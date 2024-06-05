import tensorflow as tf
import time
import tables
import librosa
import numpy as np
from pathlib import Path
from ketos.data_handling.data_feeding import BatchGenerator
from matplotlib import pyplot as plt
from deepechoes.constants import IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS
from deepechoes.gans_archs.tf.nn_archs.generators import UnetGenerator, DcgansGenerator
from deepechoes.gans_archs.tf.dcgans import DCGAN
from deepechoes.gans_archs.tf.wgan import WGAN
from deepechoes.gans_archs.tf.ssgans import SSGANS
from deepechoes.gans_archs.tf.nn_archs.discriminators import DcgansDiscriminator


def transform(X,Y):
    X = tf.reshape(X, (X.shape[0],X.shape[1], X.shape[2],1))
    return X, Y

def gans_train(hdf5_db, train_table="/train", epochs=20, batch_size=32, output_folder=None, checkpoints=None):

    db = tables.open_file(hdf5_db, 'r')
    table = db.get_node(train_table + '/data')
    # Assuming the table is a PyTables EArray or CArray, and spectrograms are stored in rows

    batch_generator = BatchGenerator(batch_size=batch_size,
                                data_table=table,
                                output_transform_func=transform,
                                shuffle=True, refresh_on_epoch_end=True, x_field="data")

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)

    if checkpoints is None:
        checkpoints = epochs
    else:
        # the checkpoint frequency cant be greater than the number of epochs
        checkpoints = min(checkpoints, epochs)
    
    generator = DcgansGenerator(apply_norm=True, norm_type='batch')
    # generator = UnetGenerator()
    discriminator = DcgansDiscriminator(apply_norm=True, norm_type='batch', phase_shift=1)
    generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    gan = DCGAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn=loss_fn)

    gan.log_dir = output_folder
    gan.checkpoint_dir = output_folder / 'checkpoints'  
    gan.generated_image_dir = output_folder / 'generations'
    
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    print("\n Training Starting ...")
    gan.train_loop(batch_generator, epochs, checkpoint_freq=checkpoints)

    gan.save(output_folder, save_discriminator=False)
    db.close()

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
    parser.add_argument('--train_table', default='/', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--output_folder', default=None, type=str, help='Output directory')
    parser.add_argument('--checkpoints', default=None, type=int, help='Checkpoint frequency in terms of epochs.')
    # parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    args = parser.parse_args()

    gans_train(**vars(args))

if __name__ == "__main__":
    main()