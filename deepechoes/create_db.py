import numpy as np
import os
import tables as tb
import pandas as pd
import json
import librosa
from tqdm import tqdm
from scipy.signal import butter, sosfilt
from pathlib import Path
from deepechoes.dev_utils.audio_processing import load_audio
from deepechoes.dev_utils.annotation_processing import standardize, define_segments, generate_time_shifted_instances, create_random_segments
from deepechoes.dev_utils.hdf5_helper import create_table_description, insert_spectrogram_data, create_or_get_table, save_dataset_attributes
from deepechoes.dev_utils.spec_preprocessing import invertible_representation, augmentation_representation_snapshot, classifier_representation
from deepechoes.dev_utils.file_management import file_duration_table

def high_pass_filter(sig, rate, order=10, freq=400):
    butter_filter = butter(N=order, fs=rate, Wn=freq,btype="highpass",output="sos")
    filtered_signal = sosfilt(butter_filter,sig)

    return filtered_signal

def create_db(data_dir, audio_representation, annotations=None, annotation_step=0, step_min_overlap=0.5, labels=None, 
              output=None, table_name=None, random_selections=None, avoid_annotations=None, overwrite=False, seed=None, 
              n_samples=None, only_augmented=False):
    
    # Initialize random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Raise an exception if both annotations and random_selections are None
    if random_selections is None and annotations is None:
        raise Exception("Missing value: Either annotations or random_selection must be defined.") 

    selections = {}
    
    # Open and read the audio configuration file (e.g., JSON file with audio settings)
    with open(audio_representation, 'r') as f:
        config = json.load(f)

    annots = None
    if annotations is not None: # If an annotation table is provided
        annots = pd.read_csv(annotations)
        annots = standardize(annots, labels=labels) # Standardize annotations by mapping labels to integers
        
        # Get the list of labels after processing
        labels = annots.label.unique().tolist()
   
        # Remove any label equal to -1 (an "ignore" label)
        labels = [label for label in labels if label != -1]

        # Check if start and end times are present in the annotation dataframe
        if 'start' in annots.columns and 'end' in annots.columns:
            for label in labels:
                # Define segments for the given label based on annotation data
                selections[label] = define_segments(annots, duration=config['duration'], center=False)

                # If annotation_step is set, create time-shifted instances
                if annotation_step > 0:
                    shifted_segments = generate_time_shifted_instances(selections[label], step=annotation_step, min_overlap=step_min_overlap)

                    if only_augmented:
                        # Only include the time-shifted segments and discard the original ones
                        selections[label] = shifted_segments
                    else:
                        # Concatenate the original segments with the new time-shifted instances
                        selections[label] = pd.concat([selections[label], shifted_segments], ignore_index=True)
        else:
            # If start and end are not present, treat annotations as selections directly 
            for label in labels:
                selections[label] = annots.loc[annots['label'] == label]

    # Handle random selections for generating new random segments
    if random_selections is not None: 
        num_segments = random_selections[0] # Number of segments to generate
        if avoid_annotations is not None and annotations is None: # Avoid areas with existing annotations
            annots = pd.read_csv(avoid_annotations)
            annots = standardize(annots, labels=labels)
            
            if num_segments == 'same':
                raise ValueError("The number of background samples to generate cannot be 'same' when avoid_annotations is being used.")

        if num_segments == 'same': # If num_segments is 'same', generate as many samples as the largest selection
            biggest_selection = float('-inf') 
            for label in labels:
                if len(selections[label]) > biggest_selection:
                    biggest_selection = len(selections[label])

            num_segments = biggest_selection

        print(f'\nGenerating {num_segments} samples with label {random_selections[1]}...')
        files = file_duration_table(data_dir, num=None)

        # If filenames are provided, filter the file list based on them
        if random_selections[2]:
            with open(random_selections[2], 'r') as file:
                filenames = file.read().splitlines()
            files = files[files['filename'].isin(filenames)]
        # print(files)
        
        # Generate random segments based on the file durations and label
        rando = create_random_segments(files, config['duration'], num_segments, label=random_selections[1], annotations=annots)
        # print(rando)
        
        if labels is None:
            labels = []
            
        if random_selections[1] in labels: 
            # if the random selection label already exists in the selections, concatenate the generatiosn with the selections that already exist
            selections[random_selections[1]] = pd.concat([selections[random_selections[1]], rando], ignore_index=False) # concatenating the generated random selections with the existings selections
        else:
            # if the random selections label did not yet exist in the selections, add it to the list of labels
            labels.append(random_selections[1])
            selections[random_selections[1]] = rando
        # print(selections)

    if output is None:
        output = os.path.join('db', 'narw_db.h5')

    if not os.path.isabs(output): 
        output = os.path.join(os.getcwd(), output)
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True) #creating dir if it doesnt exist

    if overwrite and os.path.exists(output): 
        os.remove(output)

    print('\nCreating db...')
    with tb.open_file(output, mode='a') as h5file:
        # Load the first sample to determine the table shape
        first_key = labels[0]
        y, sr = load_audio(path=Path(data_dir) / selections[first_key].iloc[0]['filename'], start=0, end=config['duration'], new_sr=config['sr'])
        sp = classifier_representation(y, config["window"], config["step"], sr, config["num_filters"], fmin=config["fmin"], fmax=config["fmax"]).shape
        table = create_or_get_table(h5file, table_name, 'data', create_table_description(sp))
        
        # Initialize global min, max, sum, and sum of squares for normalization
        global_min = float('inf')
        global_max = float('-inf')
        global_sum = 0
        global_sum_of_squares = 0
        total_samples = 0

        # Loop through each label and add its data to the table
        for label in labels:
            print(f'\nAdding data with label {label} to table {table_name} with shape {sp}...')
            selections_label = selections[label]
            
            # If n_samples is specified, randomly sample from the selections
            if n_samples is not None:
                selections_label = selections_label.sample(n=n_samples)  

            for _, row in tqdm(selections_label.iterrows(), total=selections_label.shape[0]):
                start = 0
                if 'start' in row.index:
                    start = row['start']

                file_path = os.path.join(data_dir, row['filename'])
                
                file_duration = librosa.get_duration(path=file_path)
                if start >= file_duration: # I know this check is bizarre. but because we have some very long annotations that span two files, sometimes, when using a small duration window, and trying to centralize this window, it may end up entirely in the second file. I then adjust to try to at least span 2 files. Very rare.
                    start = file_duration - config['duration'] / 2 # the following two lines set the duration of the segment to be between two files
                end = start + config['duration']

                # Load audio segment and create its representation
                y, sr = load_audio(path=file_path, start=start, end=end, new_sr=config['sr'])
                representation_data = classifier_representation(y, config["window"], config["step"], sr, config["num_filters"], fmin=config["fmin"], fmax=config["fmax"])

                # Update global min and max
                current_min = representation_data.min()
                current_max = representation_data.max()
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max
                
                # Update global sum and sum of squares
                global_sum += representation_data.sum()
                global_sum_of_squares += np.square(representation_data).sum()
                total_samples += np.prod(representation_data.shape)

                # Insert spectrogram data into the table
                insert_spectrogram_data(table, row['filename'], start, label, representation_data)
        
        # Calculate mean and standard deviation
        global_mean = global_sum / total_samples
        global_variance = (global_sum_of_squares / total_samples) - (global_mean ** 2)
        global_std = np.sqrt(global_variance)

        # Store dataset attributes in the table (min, max, mean, std)
        attributes = {
            "min_value": global_min,
            "max_value": global_max,
            "mean_value": global_mean,
            "std_value": global_std
        }

        save_dataset_attributes(table, attributes)

def main():
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Process as key-value pairs
            setattr(namespace, self.dest, dict())
            for value in values:
                key, val = value.split('=')
                if val.isdigit():
                    val = int(val)
                getattr(namespace, self.dest)[key] = val

    class RandomSelectionsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) < 2:
                parser.error("--random_selections requires at least two arguments")
            x = values[0] if values[0] == 'same' else int(values[0])
            y = int(values[1])
            z = values[2] if len(values) > 2 else None
            setattr(namespace, self.dest, (x, y, z))

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file')
    parser.add_argument('--annotations', default=None, type=str, help='Path to the annotations .csv')
    parser.add_argument('--annotation_step', default=0, type=float, help='Produce multiple time shifted representations views for each annotated  section by shifting the annotation  \
                window in steps of length step (in seconds) both forward and backward in time. The default value is 0.')
    parser.add_argument('--step_min_overlap', default=0.5, type=float, help='Minimum required overlap between the annotated section and the representation view, expressed as a fraction of whichever of the two is shorter. Only used if step > 0.')
    parser.add_argument('--labels', default=None, nargs='*', action=ParseKwargs, help='Specify a label mapping. Example: --labels background=0 upcall=1 will map labels with the string background to 0 and labels with string upcall to 1. \
        Any label not included in this mapping will be discarded. If None, will save every label in the annotation csv and will map the labels to 0, 1, 2, 3....')
    parser.add_argument('--table_name', default=None, type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--random_selections', default=None, nargs='+', type=str, action=RandomSelectionsAction, help='Will generate random x number of samples with label y. By default, all files in the data_dir and subdirectories will be used.  \
                        To limit this, pass a .txt file with the list of filenames relative to data/dir to sample from. --random_selections x y z (Where z is the optional text file)')
    parser.add_argument('--n_samples', default=None, type=int, help='randomly select n samples from the annotations')
    parser.add_argument('--avoid_annotations', default=None, type=str, help="Path to .csv file with annotations of upcalls to avoid. Only used with --random_selections. If the annotations option is being used, this argument is ignored.")
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    parser.add_argument('--output', default=None, type=str, help='HDF5 dabase name. For isntance: db.h5')
    parser.add_argument('--overwrite', default=False, type=boolean_string, help='Overwrite the database. Otherwise append to it')
    parser.add_argument('--only_augmented', default=False, type=boolean_string, help='Only include time-shifted instances without original annotations')
    args = parser.parse_args()

    create_db(
        data_dir=args.data_dir,
        audio_representation=args.audio_representation,
        annotations=args.annotations,
        annotation_step=args.annotation_step,
        step_min_overlap=args.step_min_overlap,
        labels=args.labels,
        output=args.output,
        table_name=args.table_name,
        random_selections=args.random_selections,
        avoid_annotations=args.avoid_annotations,
        overwrite=args.overwrite,
        seed=args.seed,
        n_samples=args.n_samples,
        only_augmented=args.only_augmented
    )

if __name__ == "__main__":
    main()