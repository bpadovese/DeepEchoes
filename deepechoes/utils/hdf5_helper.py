import tables as tb
import numpy as np
import pandas as pd
import os
from ketos.data_handling.data_handling import parse_datetime
from ketos.audio.waveform import get_duration
from random import sample
from ketos.data_handling.data_handling import find_files
from deepechoes.constants import IMG_HEIGHT, IMG_WIDTH

# Function to find the global min and max values in the dataset
def find_global_min_max(dataset):
    global_min = float('inf')
    global_max = float('-inf')

    for sample in dataset:
        sample_min = sample.min().item()
        sample_max = sample.max().item()

        if sample_min < global_min:
            global_min = sample_min
        if sample_max > global_max:
            global_max = sample_max
    
    return global_min, global_max

def create_table_description(item_shape):

    # Define the data structure for the table
    class SpectrogramTable(tb.IsDescription):
        filename = tb.StringCol(100)  # Assuming filenames are strings with a max length of 100
        offset = tb.Float32Col()      # Assuming offset is a float
        id = tb.UInt32Col()      
        label = tb.UInt8Col()      
        data = tb.Float32Col(shape=item_shape)  # Shape of the representation_data
    
    return SpectrogramTable

def insert_spectrogram_data(table, filename, offset, label, representation_data):
    """
    Inserts a single row of spectrogram data into the specified PyTables table.

    Parameters:
    - table: The PyTables table where the data will be inserted.
    - filename: The filename associated with the spectrogram data.
    - offset: The offset value for the spectrogram data.
    - label: The label (as an integer) for the spectrogram data.
    - representation_data: The spectrogram data as a 2D numpy array.
    """
    # Prepare the data to be inserted
    spectrogram = table.row
    spectrogram['filename'] = filename
    spectrogram['offset'] = offset
    spectrogram['id'] = table.nrows
    spectrogram['label'] = label
    spectrogram['data'] = representation_data.astype(np.float32)  # Ensure data is float32
    
    # Insert the data into the table
    spectrogram.append()
    
    # Save (commit) the changes
    table.flush()

def get_or_create_group(h5file, path):
    """
    Navigate through or create a hierarchy of groups in an HDF5 file based on the provided path.
    
    Parameters:
    - h5file: The open HDF5 file object.
    - path: The path to navigate or create, e.g., "/a/b/c/d".
    
    Returns:
    - The final group object at the end of the path.
    """
    # Split the path into components, filtering out empty strings
    groups = [group for group in path.split('/') if group]
    
    # Start at the root of the HDF5 file
    current_group = h5file.root
    
    # Navigate through or create each group in the path
    for group_name in groups:
        if not hasattr(current_group, group_name):
            current_group = h5file.create_group(current_group, group_name, f"{group_name} Data")
        else:
            current_group = getattr(current_group, group_name)
    
    return current_group

def create_or_get_table(h5file, path, table_name, table_description):
    """
    Creates or retrieves a table within a given group path in an HDF5 file.
    
    Parameters:
    - h5file: The open HDF5 file object.
    - path: The group path where the table should be located, e.g., "/train/fw".
    - table_name: The name of the table to create or retrieve.
    - table_description: The PyTables description of the table structure.
    
    Returns:
    - The table object.
    """
    # Get or create the group for the specified path
    group = get_or_create_group(h5file, path)
    
    # Define the filters
    filters = tb.Filters(complevel=1, complib='zlib', shuffle=True, fletcher32=True)
    
    # Create or get the table within the final group
    if not hasattr(group, table_name):
        table = h5file.create_table(group, table_name, table_description, f"{table_name} Data", filters=filters, chunkshape=(5,))
    else:
        table = getattr(group, table_name)
    
    return table

def save_dataset_attributes(node, attributes):
    """
    Save attributes to the given HDF5 node (table, group, etc.).

    Parameters:
    - node: The PyTables node (e.g., table or group) where the attributes will be saved.
    - attributes: A dictionary of attributes to save.
    """
    for key, value in attributes.items():
        node.attrs[key] = value

def file_duration_table(path, datetime_format=None, num=None, exclude_subdir=None):
    """ Create file duration table.

        Args:
            path: str
                Path to folder with audio files (\*.wav)
            datetime_format: str
                String defining the date-time format. 
                Example: %d_%m_%Y* would capture "14_3_1999.txt".
                See https://pypi.org/project/datetime-glob/ for a list of valid directives.
                If specified, the method will attempt to parse the datetime information from the filename.
            num: int
                Randomly sample a number of files
            exclude_subdir: str
                Exclude subdir from the search 

        Returns:
            df: pandas DataFrame
                File duration table. Columns: filename, duration, (datetime)
    """
    paths = find_files(path=path, substr=['.wav', '.WAV', '.flac', '.FLAC'], search_subdirs=True, return_path=True)

    if exclude_subdir is not None:
        paths = [path for path in paths if exclude_subdir not in path]

    if num is not None:
        paths = sample(paths, num)

    durations = get_duration([os.path.join(path,p) for p in paths])
    df = pd.DataFrame({'filename':paths, 'duration':durations})
    if datetime_format is None:
        return df

    df['datetime'] = df.apply(lambda x: parse_datetime(os.path.basename(x.filename), fmt=datetime_format), axis=1)
    return df