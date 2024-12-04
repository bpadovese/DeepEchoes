import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from deepechoes.dev_utils.audio_processing import load_audio  # Replace `your_module` with the actual module name.

# Test audio file creation
@pytest.fixture
def test_audio_file(tmp_path):
    """Creates a temporary test audio file with sine waves and silence."""
    sr = 16000  # Sample rate
    duration = 5  # seconds
    freq = 440  # Frequency of the sine wave (Hz)

    # Create 5 seconds of sine wave
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * freq * t)

    # Save the sine wave as a WAV file
    test_file = tmp_path / "test_audio.wav"
    sf.write(test_file, sine_wave, sr)
    return str(test_file)

# Test cases
def test_full_audio_read(test_audio_file):
    """Test loading the full audio without specifying start and end."""
    audio, sr = load_audio(test_audio_file)
    with sf.SoundFile(test_audio_file) as f:
        assert sr == f.samplerate  # Check sample rate
        assert len(audio) == f.frames  # Check full length


def test_partial_audio_read_exact_content(test_audio_file):
    """Test loading a specific segment of the audio and check exact content."""
    start = 1.0  # Start at 1 second
    end = 3.0  # End at 3 seconds

    # Load the full audio manually for reference
    with sf.SoundFile(test_audio_file) as f:
        sr = f.samplerate
        f.seek(int(start * sr))  # Move to the start frame
        expected_audio = f.read(int((end - start) * sr))  # Read the exact segment

    # Use the function to load the same segment
    audio, loaded_sr = load_audio(test_audio_file, start=start, end=end)
    
    # Check sample rate consistency
    assert loaded_sr == sr, "Sample rates do not match!"

    # Check content length
    assert len(audio) == len(expected_audio), "Audio lengths do not match!"

    # Check exact content
    assert np.allclose(audio, expected_audio), "Audio content does not match!"


def test_padding_with_reflect(test_audio_file):
    """Test loading with padding using 'reflect'."""
    start = -1.0  # Negative start time
    end = 6.0  # End beyond file duration
    audio, sr = load_audio(test_audio_file, start=start, end=end, pad='reflect')
    
    expected_length = int((6.0 - (-1.0)) * sr)  # Total padded duration
    assert len(audio) == expected_length  # Check padded length

def test_padding_with_zero(test_audio_file):
    """Test loading with padding using 'zero'."""
    start = -1.0  # Negative start time
    end = 6.0  # End beyond file duration
    audio, sr = load_audio(test_audio_file, start=start, end=end, pad='zero')
    
    expected_length = int((6.0 - (-1.0)) * sr)  # Total padded duration
    assert len(audio) == expected_length  # Check padded length
    assert np.all(audio[: int(sr)] == 0)  # Check zero padding at the start
    assert np.all(audio[-int(sr) :] == 0)  # Check zero padding at the end

def test_resampling(test_audio_file):
    """Test resampling the audio to a new sample rate."""
    new_sr = 8000  # New sample rate
    audio, sr = load_audio(test_audio_file, new_sr=new_sr)
    assert sr == new_sr  # Check new sample rate
    original_length_in_seconds = len(audio) / new_sr
    with sf.SoundFile(test_audio_file) as f:
        original_duration = f.frames / f.samplerate
    assert pytest.approx(original_length_in_seconds, 0.01) == original_duration