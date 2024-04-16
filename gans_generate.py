import tensorflow as tf
import tables as tb
from tqdm import tqdm
from pathlib import Path
from utils.image_transforms import unnormalize_data
from utils.hdf5_helper import insert_spectrogram_data, create_or_get_table, SpectrogramTable


def load_generator(model_path):
    """ Load the generator model from a saved file """
    generator = tf.keras.models.load_model(model_path)
    print(f"Generator model loaded from {model_path}.")
    return generator


def generate_new_specs(generator, noise_dim=100, batch_size=100):
    """ Generate new spectrograms using the generator """
    # Assume generator expects a random noise vector as input
    noise_dim = 100  
    random_noise = tf.random.normal([batch_size, noise_dim])
    
    generated_images = generator(random_noise, training=False)
    return [unnormalize_data(generated_images[i, :, :, 0].numpy()) for i in range(batch_size)]


def gans_generate_to_hdf5(model_path, num_samples=10, noise_dim=100, output_folder=None, hdf5_db_name='gans_db.h5', table_name='/train', label=1, batch_size=100):
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    
    output = output_folder / hdf5_db_name

    generator = load_generator(model_path)
    generated_images = generate_new_specs(generator, noise_dim, num_samples)
    print('\nCreating db...')
    with tb.open_file(output, mode='a') as h5file:
        table = create_or_get_table(h5file, table_name, 'data', SpectrogramTable)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed

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
    parser.add_argument('--num_samples', default=10, type=int, help="How many samples to generate.")
    parser.add_argument('--noise_dim', default=100, type=int, help='The noise dim.')
    parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
    parser.add_argument('--table_name', default='/train', type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--hdf5_db_name', default='gans_db.h5', type=str, help='HDF5 dabase name. For isntance: db.h5')
    parser.add_argument('--label', default=1, type=int, help='Label to assign the generated images.')
    parser.add_argument('--batch_size', default=100, type=int, help='Number of samples to generate per batch.')

    args = parser.parse_args()
    gans_generate_to_hdf5(**vars(args))

if __name__ == "__main__":
    main()