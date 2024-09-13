import tensorflow as tf
import tables as tb
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from deepechoes.dev_utils.image_transforms import unscale_data
from deepechoes.dev_utils.hdf5_helper import insert_spectrogram_data, create_or_get_table, SpectrogramTable


def load_generator(model_path):
    """
    Load the generator model from a saved file.

    Parameters:
        model_path (str): The path to the saved generator model.

    Returns:
        tf.keras.Model: The loaded generator model.
    """
    generator = tf.keras.models.load_model(model_path)
    print(f"Generator model loaded from {model_path}.")
    return generator


def generate_new_specs(generator, noise_dim=100, batch_size=100, transform=None):
    """
    Generate new spectrograms using the generator model.

    Parameters:
        generator (tf.keras.Model): The pre-trained generator model.
        noise_dim (int): The dimensionality of the noise vector input to the generator. Default is 100.
        batch_size (int): The number of spectrograms to generate. Default is 100.

    Returns:
        list: A list of generated spectrograms in numpy array format.
    """
    # Assume generator expects a random noise vector as input
    noise_dim = 100  
    random_noise = tf.random.normal([batch_size, noise_dim])
    
    generated_images = generator(random_noise, training=False)
    if transform:
        return [unscale_data(generated_images[i, :, :, 0].numpy()) for i in range(batch_size)]
    return [generated_images[i, :, :, 0].numpy() for i in range(batch_size)]

def plot_images(input, output_folder, batch_num):
    """
    Plot and save generated spectrograms as images.

    Parameters:
        input (list): A list of generated spectrograms.
        output_folder (Path): The folder to save the images.
        batch_num (int): The current batch number, used for naming the output image file.
    """
    fig, axs = plt.subplots(4, 4, figsize=(34, 28))
    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust as needed

    for i in range(len(input)):
        ax = axs[i // 4, i % 4]
        ax.imshow(input[i], aspect='auto', origin='lower', cmap='viridis')
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_folder / f'image_{batch_num}.png')
    plt.close(fig)

def gans_generate_to_plot(model_path, num_samples=10, noise_dim=100, output_folder=None):
    """
    Generate and plot spectrograms using a GAN generator model, and save them as images.

    Parameters:
        model_path (str): Path to the saved generator model.
        num_samples (int): Number of spectrograms to generate. Default is 10.
        noise_dim (int): Dimensionality of the noise vector input to the generator. Default is 100.
        output_folder (str or None): Folder to save the images. Default is the current directory.
    """
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    generator = load_generator(model_path)

    batch_size=16
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed

    for batch_num in tqdm(range(num_batches)):
        batch_size_adjusted = min(batch_size, num_samples - batch_num * batch_size) # calculate the batch size for the current batch (will be different for the last batch)
        generated_images = generate_new_specs(generator, noise_dim, batch_size_adjusted)
        plot_images(generated_images, output_folder, batch_num)

def save_single_spectrogram_to_csv(model_path, noise_dim=100, csv_file_path='single_spectrogram.csv'):
    """
    Generate a single spectrogram using the GAN generator model and save it to a CSV file.

    Parameters:
        model_path (str): Path to the saved generator model.
        noise_dim (int): The dimensionality of the noise vector input to the generator.
        csv_file_path (str): Path to save the CSV file.
    """
    generator = load_generator(model_path)
    # Generate a single spectrogram using the existing function
    generated_spectrograms = generate_new_specs(generator, noise_dim, batch_size=1)
    
    # Save the generated spectrogram to a CSV file
    np.savetxt(csv_file_path, generated_spectrograms[0], delimiter=',')
    print(f"Saved generated spectrogram to CSV at {csv_file_path}")

def gans_generate_to_hdf5(model_path, num_samples=10, noise_dim=100, output_folder=None, hdf5_db_name='gans_db.h5', table_name='/train', label=1, batch_size=100):
    """
    Generate spectrograms using a GAN generator model and save them to an HDF5 database.

    Parameters:
        model_path (str): Path to the saved generator model.
        num_samples (int): Number of spectrograms to generate. Default is 10.
        noise_dim (int): Dimensionality of the noise vector input to the generator. Default is 100.
        output_folder (str or None): Folder to save the HDF5 database. Default is the current directory.
        hdf5_db_name (str): Name of the HDF5 database file. Default is 'gans_db.h5'.
        table_name (str): Name of the table within the database to store the data. Must start with a forward slash.
        label (int): Label to assign to the generated spectrograms. Default is 1.
        batch_size (int): Number of samples to generate per batch. Default is 100.
    """
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    
    output = output_folder / hdf5_db_name

    generator = load_generator(model_path)
    print('\nCreating db...')
    with tb.open_file(output, mode='a') as h5file:
        table = create_or_get_table(h5file, table_name, 'data', SpectrogramTable)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed
        print(f'\nGenerating {num_samples} samples with label {label} to table {table_name}...')
        for batch_num in tqdm(range(num_batches)):
            batch_size_adjusted = min(batch_size, num_samples - batch_num * batch_size) # calculate the batch size for the current batch (will be different for the last batch)
            generated_images = generate_new_specs(generator, noise_dim, batch_size_adjusted)

            for i, representation_data in enumerate(generated_images):
                idx = batch_num * batch_size + i
                filename = f"gans_{idx}"
                insert_spectrogram_data(table, filename, 0, label, representation_data)

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path', type=str, help='Path to where the generator is saved')
    parser.add_argument('--mode', type=str, default='hdf5', help='mode to either save to a hdf5 db or plot, or saves 1 image to a "csv"')
    parser.add_argument('--num_samples', default=10, type=int, help="How many samples to generate.")
    parser.add_argument('--noise_dim', default=100, type=int, help='The noise dim.')
    parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
    parser.add_argument('--table_name', default='/train', type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--hdf5_db_name', default='gans_db.h5', type=str, help='HDF5 dabase name. For isntance: db.h5')
    parser.add_argument('--label', default=1, type=int, help='Label to assign the generated images.')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples to generate per batch.')

    args = parser.parse_args()
    mode = args.mode
    arg_dict = vars(args)
    arg_dict.pop('mode')  # Remove mode entry from dictionary

    if mode == 'hdf5':
        gans_generate_to_hdf5(**arg_dict)
    elif mode == 'csv':
        save_single_spectrogram_to_csv(args.model_path)
    else:    
        gans_generate_to_plot(args.model_path, args.num_samples, args.noise_dim, args.output_folder)

if __name__ == "__main__":
    main()