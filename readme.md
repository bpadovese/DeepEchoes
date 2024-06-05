# DeepEchoes: Marine Mammal Spectrogram Synthesis

## Overview
This project aims to generate spectrograms of marine mammal calls using generative algorithms, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). 

This project is currently under development.

## Project Structure

    .
    ├── constants.py # Project constants and configuration
    ├── create_db.py # Script to create and populate the HDF5 database
    ├── discriminators.py # Discriminator architectures for GANs
    ├── generators.py # Generator architectures for GANs
    ├── gans.py # Main script to train and evaluate GANs
    ├── utils/ # Utility scripts
    │   ├── hdf5_helper.py # Helper functions for managing HDF5 files
    │   ├── plot_db_from_spec.py # Script to visualize database contents
    │   └── image_transforms.py # Image transformation functions
    └── gans_archs/ # GAN architectures
    ├── base.py # Base classes for GAN components
    ├── dcgans.py # DCGAN architecture
    └── wgan.py # WGAN architecture
