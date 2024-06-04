import numpy as np
import scipy.signal
import librosa
import tables as tb
import json
import skimage
from pathlib import Path
from utils.image_transforms import unnormalize_data

def create_waveforms_from_hdf5(hdf5_db, audio_representation, train_table, output_folder):
    """
    Create waveforms from spectrograms stored in an HDF5 database.

    Parameters:
        - hdf5_db (str): Path to the HDF5 database file.
        - audio_representation (str): Path to the audio representation.
        - train_table (str): Name of the table within the HDF5 database.
        - output_folder (str): Directory to save the generated waveform files.
    """
    db = tb.open_file(hdf5_db, 'r')
    table = db.get_node(train_table + '/data')

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the JSON configuration from a file
    with open(audio_representation, 'r') as file:
        config = json.load(file)
        
    sr = config['spectrogram']['rate']

    for i in range(table.nrows):
        spectrogram = table[i]['data']
        filename = table[i]['filename'].decode('utf-8')
        
        n_fft = int(config["spectrogram"]["window"] * sr)  # Window size
        hop_length = int(config["spectrogram"]["step"] * sr)  # Step size
        
        # spectrogram = unnormalize_data(spectrogram)
        # spectrogram = skimage.transform.resize(spectrogram, (150, 241))

        # Perform ISTFT
        # waveform = librosa.feature.inverse.mel_to_audio(spectrogram, n_fft=n_fft, hop_length=hop_length, sr=sr, fmin=400, fmax=int(sr/2))
        waveform = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sr, fmin=400, fmax=int(sr/2))

        # Normalize the waveform to the range [-1, 1]
        waveform /= np.max(np.abs(waveform))

        # Save the waveform to a file
        waveform_filename = output_path / f"waveform_{filename}.wav"
        scipy.io.wavfile.write(waveform_filename, sr, waveform)

    db.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create waveforms from spectrograms stored in an HDF5 database.")
    parser.add_argument("hdf5_db", type=str, help="Path to the HDF5 database file")
    parser.add_argument("audio_representation", type=str, help="Json file with the audio representation")
    parser.add_argument("table", type=str, help="Name of the table within the HDF5 database")
    parser.add_argument("output_folder", type=str, help="Directory to save the generated waveform files")
    

    args = parser.parse_args()

    create_waveforms_from_hdf5(args.hdf5_db, args.audio_representation, args.table, args.output_folder)
