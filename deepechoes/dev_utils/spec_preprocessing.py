import numpy as np
import librosa
from PIL import Image

def classifier_representation(y, window, step, sr, n_mels, fmin=0, fmax=12000, ref=np.max, top_db=80, mode='img'):
    # Converting windows size and step size to nfft and hop length (in frames) because librosa uses that.
    n_fft = int(window * sr)  # Window size
    hop_length = int(step * sr)  # Step size
    # n_fft = 1024
    # hop_length = 188
    # n_fft = 2048
    # hop_length = 282
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window="hann", n_mels=n_mels, fmin=fmin, fmax=fmax)
    # spec = librosa.power_to_db(S, ref=1.0, top_db=80.0)
    spec = librosa.power_to_db(S, ref=ref, top_db=80.0)
    if mode == 'img':
        bytedata = (((spec + top_db) * 255 / top_db).clip(0, 255) + 0.5).astype(np.uint8)
        spec = Image.fromarray(bytedata)

    return spec

def image_to_audio(self, image: Image.Image) -> np.ndarray:
    """Converts spectrogram to audio.

    Args:
        image (`PIL Image`): x_res x y_res grayscale image

    Returns:
        audio (`np.ndarray`): raw audio
    """
    bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
    log_S = bytedata.astype("float") * self.top_db / 255 - self.top_db
    S = librosa.db_to_power(log_S)
    audio = librosa.feature.inverse.mel_to_audio(
        S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
    )
    return audio