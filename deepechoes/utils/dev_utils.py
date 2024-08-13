import pandas as pd
import os
import soundfile as sf
import librosa
from pathlib import Path
from random import sample


def find_files(path, substr, return_path=True, search_subdirs=False, search_path=False):
    """Find all files in the specified directory containing the specified substring in their file name or path.

    Args:
        path: str or Path
            Directory path.
        substr: str
            Substring contained in file name or path.
        return_path: bool
            If True, return the path to each file, relative to the top directory. 
            If False, only return the filenames.
        search_subdirs: bool
            If True, search all subdirectories.
        search_path: bool
            If True, search for substring occurrence in the relative path rather than just the filename.

    Returns:
        files: list (str)
            Alphabetically sorted list of file names or paths.
    """
    path = Path(path)
    if isinstance(substr, str):
        substr = [substr]
    
    if search_subdirs:
        files = path.rglob('*')
    else:
        files = path.glob('*')

    matching_files = []
    
    for file in files:
        relative_path = file.relative_to(path)
        search_target = str(relative_path) if search_path else file.name
        if any(ss in search_target for ss in substr):
            matching_files.append(str(relative_path) if return_path else file.name)

    return sorted(matching_files)


def get_duration(file_paths):
    """Calculate the durations of multiple audio files.

    Args:
        file_paths (list): List of paths to audio files.

    Returns:
        list: Durations of the audio files in seconds.
    """
    durations = []
    for file_path in file_paths:
        with sf.SoundFile(file_path) as sound_file:
            durations.append(len(sound_file) / sound_file.samplerate)
    return durations


def file_duration_table(path, num=None, exclude_subdir=None):
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
    return df