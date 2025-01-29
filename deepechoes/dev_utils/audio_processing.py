import numpy as np
import soundfile as sf
import librosa

def load_segment(path, start=None, end=None, new_sr=None, pad='reflect'):
    """
    Loads an audio segment with optional padding.

    Args:
        path (str): Path to the audio file.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        new_sr (int): If provided, resample the audio to this sample rate.
        pad (str): Padding option to apply if the requested segment extends beyond the audio file.
                   Options:
                   - 'zero': Pads the beginning and/or end of the audio with zeros if the start or 
                     end time is outside the file's duration.
                   - 'reflect' (default): Pads the beginning and/or end of the audio with a reflected version 
                     of the audio if the start or end time is outside the file's duration.

    Returns:
        np.ndarray: The loaded audio segment.
        int: The sample rate of the loaded audio.
    """
    # Open the file to get the sample rate and total frames
    
    with sf.SoundFile(path) as file:
        sr = file.samplerate
        total_frames = file.frames
        file_duration = total_frames / sr  # Duration of the file in seconds

        # Default to the full file if neither start nor end is provided
        if start is None and end is None:
            start, end = 0, file_duration
        
        # Adjust start time if it's negative, and dynamically adjust the end time to maintain duration
        pad_start = 0
        pad_end = 0
        if start is not None and start < 0:
            pad_start = -start  # Calculate how much to pad at the beginning
            start = 0

        # Ensure end time does not exceed the file's duration
        if end is not None and end > file_duration:
            pad_end = end - file_duration  # Calculate how much to pad at the end
            end = file_duration

        # Convert start and end times to frame indices
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        
        # Read the specific segment of the audio file
        file.seek(start_frame)
        audio_segment = file.read(end_frame - start_frame)

    # Apply padding if necessary
    if (pad_start > 0 or pad_end > 0):
        # Calculate padding in frames
        pad_start_frames = int(pad_start * sr)
        pad_end_frames = int(pad_end * sr)

        if pad == 'zero':
            # Pad with zeros
            audio_segment = np.pad(audio_segment, (pad_start_frames, pad_end_frames), 'constant')
        elif pad == 'reflect':
            # Pad with reflection
            audio_segment = np.pad(audio_segment, (pad_start_frames, pad_end_frames), 'reflect')

    # Resample the audio segment if new sample rate is provided and different from the original
    if new_sr is not None and new_sr != sr:
        audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=new_sr)
        sr = new_sr  # Replace original sample rate with the new one
    
    return audio_segment, sr

def pad_or_crop_audio2(y, sr, duration, start, end):
    target_length = int(sr * duration)
    current_length = len(y)

    if current_length < target_length:
        pad_length = target_length - current_length
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        y = np.pad(y, (pad_left, pad_right), mode='constant')
    elif current_length > target_length:
        start_idx = (current_length - target_length) // 2
        y = y[start_idx:start_idx + target_length]

    return y

def pad_or_crop_audio(y, sr, duration, start, end):
    """
    Pads or crops the audio signal based on a given start and end time.

    Parameters:
    - y (numpy array): The audio signal.
    - sr (int): The sample rate of the audio.
    - start (float): The start time of the audio in seconds.
    - end (float): The end time of the audio in seconds.
    - duration (float): The target duration in seconds.

    Returns:
    - y (numpy array): The processed audio signal.
    """
    target_length = int(sr * duration)
    current_length = len(y)

    if current_length < target_length:  # Padding needed
        pad_amount = target_length - current_length  # Number of samples to pad
        if start < 0:  # Pad on the right
            pad_left = 0
            pad_right = pad_amount
        elif end > duration:  # Pad on the left
            pad_left = pad_amount
            pad_right = 0
        else:  # Default: Pad symmetrically
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
        
        y = np.pad(y, (pad_left, pad_right), mode='constant')

    return y

import numpy as np

def time_shift_signal_dynamic(y, sr, duration=3.0, min_overlap=0.8, num_shifts=10):
    """
    Places the signal in various positions within a 3-second interval with dynamic padding.
    
    Parameters:
    - y: np.array, the audio signal
    - sr: int, sample rate
    - duration: float, target duration in seconds (default: 3s)
    - min_overlap: float, minimum overlap required when shifting (default: 80%)
    - num_shifts: int, number of shifted versions to generate
    
    Returns:
    - list of np.array, each a version of the signal shifted within the interval
    """
    target_length = int(duration * sr)  # 3 seconds in samples
    signal_length = len(y)

    # Determine possible start positions ensuring at least min_overlap is within the view
    max_shift = target_length - int(signal_length * min_overlap)
    shifts = np.linspace(0, max_shift, num_shifts, dtype=int)  # Different shift positions

    shifted_signals = []
    for shift in shifts:
        new_y = np.zeros(target_length)  # Create a zero-padded array of target length
        start_idx = shift
        end_idx = start_idx + signal_length

        # Insert the original signal at the shifted position
        new_y[start_idx:end_idx] = y[:target_length - start_idx]
        shifted_signals.append(new_y)

    return shifted_signals


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