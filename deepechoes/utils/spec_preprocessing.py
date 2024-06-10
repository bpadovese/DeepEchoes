import numpy as np
import librosa
import skimage
from deepechoes.utils.image_transforms import scale_to_range, normalize_to_zero_mean_unit_variance
from deepechoes.constants import IMG_HEIGHT, IMG_WIDTH
from deepechoes.utils.spec_to_wav import spec_to_wav

def invertible_representation(y, window, step, sr, n_mels, fmin=0, fmax=8000):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size 
    hop_length = int(step * sr)  # Step size
    # n_fft = 800
    # hop_length = int(n_fft * 0.31)  
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hann", n_mels=n_mels, fmin=fmin)
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hann", n_mels=n_mels, fmin=fmin, fmax=fmax)
    representation_data = librosa.power_to_db(S, ref=np.max)
    # Remove one column from the beginning and one column from the end
    representation_data = representation_data[:, 1:-1]
    # Normalize each frequency bin to have zero mean and unit variance
    # representation_data = normalize_to_zero_mean_unit_variance(representation_data, clip_std=True)
    waveform = spec_to_wav(representation_data, n_fft, hop_length, sr)
    
    # max_amplitude = np.max(np.abs(waveform))
    # if max_amplitude > 0:
    #     waveform /= max_amplitude

    # Rescale to [-1, 1]
    # representation_data = scale_to_range(representation_data, -1, 1)
    representation_data = np.clip(representation_data, -1, 1)
    return representation_data

def augmentation_representation_snapshot(y, window, step, sr, n_mels, fmin=400, fmax=12000):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hamming", n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec = librosa.power_to_db(S, ref=np.max)
    
    representation_data = skimage.transform.resize(spec, (IMG_HEIGHT,IMG_WIDTH))
    representation_data = scale_to_range(representation_data, -1, 1)

    return representation_data