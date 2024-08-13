from scipy.signal import butter, sosfilt
import librosa
from tqdm import tqdm
import numpy as np
import os
import tables as tb
import soundfile as sf
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from ketos.data_handling import selection_table as sl
from ketos.data_handling.parsing import load_audio_representation
from deepechoes.constants import IMG_HEIGHT, IMG_WIDTH
from deepechoes.utils.hdf5_helper import create_table_description, insert_spectrogram_data, create_or_get_table, save_dataset_attributes
from deepechoes.utils.spec_preprocessing import invertible_representation, augmentation_representation_snapshot, classifier_representation
from deepechoes.utils.dev_utils import file_duration_table

def load_data(path, start=None, end=None, new_sr=None):
    # Open the file to get the sample rate and total frames
    
    with sf.SoundFile(path) as file:
        sr = file.samplerate
        total_frames = file.frames
        file_duration = total_frames / sr  # Duration of the file in seconds

        # Default to the full file if neither start nor end is provided
        if start is None and end is None:
            start, end = 0, file_duration
        
        # Adjust start time if it's negative, and dynamically adjust the end time to maintain duration
        if start is not None and start < 0:
            end += -start  # Adjust end time by the amount start time was negative
            start = 0

        # Ensure end time does not exceed the file's duration
        if end is not None and end > file_duration:
            start = max(start - (end - file_duration), 0)
            end = file_duration

        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        # Read the specific segment of the audio file
        file.seek(start_frame)
        audio_segment = file.read(end_frame - start_frame)

    return_sr = sr
    # Resample the audio segment if new sample rate is provided
    if new_sr is not None:
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=new_sr)
        return_sr = new_sr
    
    return audio_segment, return_sr

def high_pass_filter(sig, rate, order=10, freq=400):
    butter_filter = butter(N=order, fs=rate, Wn=freq,btype="highpass",output="sos")
    filtered_signal = sosfilt(butter_filter,sig)

    return filtered_signal

def create_db(data_dir, audio_representation, annotations=None, annotation_step=0, step_min_overlap=0.5, labels=None, 
              output=None, table_name=None, random_selections=None, avoid_annotations=None, overwrite=False, seed=None, 
              n_samples=None):
    
    #initialiaze random seed
    if seed is not None:
        np.random.seed(seed)

    if random_selections is None and annotations is None:
        raise Exception("Missing value: Either annotations or random_selection must be defined.") 

    selections = {}
    
    #load the audio representation. We are currently only allowing 1
    config = load_audio_representation(audio_representation)
    config = config[list(config.keys())[0]]

    annots = None
    if annotations is not None: # if an annotation table is given
        annots = pd.read_csv(annotations)

        if labels is None:
            labels = annots.label.unique().tolist() # For each unique label
   
        annots = sl.standardize(table=annots, trim_table=True, labels=labels) #standardize to ketos format and remove extra columns
        annots['label'] = annots['label'].astype(int)

        labels = annots.label.unique().tolist() # get the actual list of labels after all the processing
   
        # removing label -1
        labels = [label for label in labels if label != -1]

        if 'start' in annots.columns and 'end' in annots.columns: # Checking if start and end times are in the dataframe
            for label in labels:
                selections[label] = sl.select(annotations=annots, length=config['duration'], step=annotation_step, min_overlap=step_min_overlap, center=False, label=[label]) #create the selections
        else: # if not, than the annotations are already the selections
            
            for label in labels:
                selections[label] = annots.loc[annots['label'] == label]
                

    # random_selections is a list where the first index is the number of samples to generate and the second index is the label to assign to the generations
    if random_selections is not None: 
        num_segments = random_selections[0]
        if avoid_annotations is not None and annotations is None:
            annots = pd.read_csv(avoid_annotations)
            annots = sl.standardize(table=annots, trim_table=True, labels=labels)
            if num_segments == 'same':
                raise ValueError("The number of background samples to generate cannot be 'same' when avoid_annotations is being used.")

        if num_segments == 'same':
            biggest_selection = float('-inf') 
            for label in labels:
                if len(selections[label]) > biggest_selection:
                    biggest_selection = len(selections[label])

            num_segments = biggest_selection

        print(f'\nGenerating {num_segments} samples with label {random_selections[1]}...')
        files = file_duration_table(data_dir, num=None)

        if random_selections[2]:
            with open(random_selections[2], 'r') as file:
                filenames = file.read().splitlines()
        
            files = files[files['filename'].isin(filenames)]

        rando = sl.create_rndm_selections(files=files, length=config['duration'], annotations=annots, num=num_segments, label=random_selections[1])
        del rando['duration'] # create_rndm selections returns the duration which we dont need. So lets delete it

        if labels is None:
            labels = []
            
        if random_selections[1] in labels: 
            # if the random selection label already exists in the selections, concatenate the generatiosn with the selections that already exist
            selections[random_selections[1]] = pd.concat([selections[random_selections[1]], rando], ignore_index=False) # concatenating the generated random selections with the existings selections
        else:
            # if the random selections label did not yet exist in the selections, add it to the list of labels
            labels.append(random_selections[1])
            selections[random_selections[1]] = rando

    if output is None:
        output = os.path.join('db', 'narw_db.h5')

    if not os.path.isabs(output): 
        output = os.path.join(os.getcwd(), output)
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True) #creating dir if it doesnt exist

    if overwrite and os.path.exists(output): 
        os.remove(output)

    print('\nCreating db...')
    with tb.open_file(output, mode='a') as h5file:
        # Loading one sample to get the table shape
        y, sr = load_data(path=Path(data_dir) / selections[0].iloc[0].name[0], start=0, end=config['duration'], new_sr=config['rate'])
        sp = classifier_representation(y, config["window"], config["step"], sr, config["num_filters"], fmin=config["fmin"], fmax=config["fmax"]).shape
        table = create_or_get_table(h5file, table_name, 'data', create_table_description(sp))
        # Initialize global min and max values and sums
        global_min = float('inf')
        global_max = float('-inf')
        global_sum = 0
        global_sum_of_squares = 0
        total_samples = 0

        for label in labels:
            print(f'\nAdding data with label {label} to table {table_name} with shape {sp}...')
            selections_label = selections[label]
            if n_samples is not None:
                # Filter the DataFrame by the label
                selections_label = selections_label.sample(n=n_samples)  

            for (filename, sel_id), row in tqdm(selections_label.iterrows(), total=selections_label.shape[0]):
                start = 0
                if 'start' in row.index:
                    start = row['start']
                
                y, sr = load_data(path=Path(data_dir) / filename, start=start, end=start+config['duration'], new_sr=config['rate'])
                
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

                insert_spectrogram_data(table, filename, start, label, representation_data)
        
        # Calculate mean and standard deviation
        global_mean = global_sum / total_samples
        global_variance = (global_sum_of_squares / total_samples) - (global_mean ** 2)
        global_std = np.sqrt(global_variance)

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
    args = parser.parse_args()

    create_db(**vars(args))

if __name__ == "__main__":
    main()