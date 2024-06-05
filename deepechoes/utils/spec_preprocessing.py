import numpy as np
import librosa
import skimage
from utils.image_transforms import scale_to_range, normalize_to_zero_mean_unit_variance
from constants import IMG_HEIGHT, IMG_WIDTH

def invertible_representation(y, window, step, sr, n_mels, fmin=0, fmax=8000):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(S, ref=np.max)
    
    representation_data = skimage.transform.resize(spec, (IMG_HEIGHT,IMG_WIDTH))#128x128

    # Normalize each frequency bin to have zero mean and unit variance
    normalized_spec = normalize_to_zero_mean_unit_variance(spec, clip_std=True)

    # Rescale to [-1, 1]
    representation_data = scale_to_range(normalized_spec, -1, 1)
    return representation_data

def augmentation_representation_snapshot(y, window, step, sr, n_mels, fmin=400, fmax=12000):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(S, ref=np.max)
    
    representation_data = skimage.transform.resize(spec, (IMG_HEIGHT,IMG_WIDTH))
    representation_data = scale_to_range(representation_data)

    return representation_data