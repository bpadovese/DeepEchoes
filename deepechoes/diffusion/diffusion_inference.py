from diffusers import DiffusionPipeline
from diffusers import DDIMScheduler, DDPMScheduler
from pathlib import Path
from tqdm import tqdm
from deepechoes.dev_utils.hdf5_helper import insert_spectrogram_data, create_or_get_table, create_table_description
import tables as tb
import matplotlib.pyplot as plt
import math
import os
import torch
import numpy as np
import random
from deepechoes.dev_utils.validate_generation import filter_spectrograms, load_spectrogram
from sklearn.decomposition import PCA


    
def save_specs_individually(images, output_path, prefix="spec"):
    """
    Save spectrograms as individual images.

    Parameters:
    - images: numpy array of spectrograms
    - output_path: Path object or string, directory to save the spectrograms
    - prefix: Optional prefix for filenames
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
        ax.axis('off')  # Turn off the axis
        fig.savefig(output_path / f"{prefix}_{i}.png", bbox_inches='tight')
        plt.close(fig)  # Close the figure to save memory


def make_grid_spec(images, cols):
    print(images.shape)
    num_images = images.shape[0]
    rows = math.ceil(num_images / cols)  # Calculate the number of rows
    fig_height = rows * 3  # Height of each subplot
    fig, axes = plt.subplots(rows, cols, figsize=(15, fig_height))

    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Plotting the spectrogram
        ax.imshow(image[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
        ax.axis('off')  # Turn off the axis

    # Hide any remaining empty subplots
    for j in range(num_images, rows * cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    return fig

def to_img(pipeline, num_samples, output_path='./dataset', batch_size=8, num_inference_steps=1000, display=None, real_spectrograms=None):    
    num_valid_samples = 0
    batch_num = 0
    if real_spectrograms is not None:
        # Train PCA on real spectrograms
        pca = PCA(n_components=2)
        real_pca_scores = pca.fit_transform(real_spectrograms)    

    # Outer loop for batches
    with tqdm(total=num_samples, desc="Generating Samples") as pbar:
        while num_valid_samples < num_samples:
            batch_size_adjusted = min(batch_size, num_samples - num_valid_samples)
            batch_num += 1
            # Generate an image
            images = pipeline(
                batch_size=batch_size_adjusted, 
                generator=None,
                num_inference_steps=num_inference_steps,
                return_dict=False
                )[0]
            
            if real_spectrograms is not None:
                # Apply filtering
                images_array = np.array([np.array(img) for img in images])  # Convert to numpy array
                _, keep_indices = filter_spectrograms(images_array, real_pca_scores, pca)
                images = [img for i, img in enumerate(images) if keep_indices[i]]

    
            if display:
                spec_grid = make_grid_spec(images, cols=4)
                # Save the figure
                spec_grid.savefig(output_path / f'{str(batch_num)}.png', bbox_inches='tight')
            else:
                 # Save filtered images
                for i, image in enumerate(images):
                    if num_valid_samples >= num_samples:
                        break
                    image.save(output_path / f"diffusion_{num_valid_samples}.png")
                    num_valid_samples += 1
                    pbar.update(1)
                
                # save_specs_individually(images, output_path=output_path, prefix=f'batch_{batch_num}')

def to_hdf5(pipeline, num_samples, output_path='diffusion.h5', table_name='/train', label=1, batch_size=8, num_inference_steps=1000):
    
    print('\nCreating db...')
    with tb.open_file(output_path, mode='a') as h5file:
        table = create_or_get_table(h5file, table_name, 'data', create_table_description((128,128)))
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed
        print(f'\nGenerating {num_samples} samples with label {label} to table {table_name}...')
        for batch_num in tqdm(range(num_batches)):
            batch_size_adjusted = min(batch_size, num_samples - batch_num * batch_size) # calculate the batch size for the current batch (will be different for the last batch)
            # Generate the images
            images = pipeline(batch_size=batch_size_adjusted, output_type="nd.array", num_inference_steps=num_inference_steps).images

            for i, representation_data in enumerate(images):
                idx = batch_num * batch_size + i
                filename = f"diffusion_{idx}"

                # Squeeze the last dimension if it's 1
                if representation_data.shape[-1] == 1:
                    representation_data = representation_data.squeeze(-1)
                
                insert_spectrogram_data(table, filename, 0, label, representation_data)

def diffusion_inference(model_path, mode, num_samples, output_path='diffusion.h5', table_name='/train', label=1, batch_size=8, num_inference_steps=1000, validation=None, seed=None):
    
    if seed:
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if torch.cuda.is_available():
        print("CUDA is available, using GPU.")
    else:
        print("CUDA is not available, using CPU.")
    
    if output_path is None:
        output_path = Path('.').resolve()
    else:
        output_path = Path(output_path).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the diffusion model
    pipeline = DiffusionPipeline.from_pretrained(model_path).to("cuda")
    # print(pipeline.scheduler)
    # Replace the scheduler with DDIMScheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print(pipeline.scheduler)
    pipeline.set_progress_bar_config(disable=True, leave=True, desc="Pipeline Progress") # for some reason, leave True not working
    
    real_spectrograms = None
    if validation:
        real_spectrogram_files = [
            os.path.join(validation, f) for f in os.listdir(validation) if f.endswith('.png')
        ]

        real_spectrograms = np.array([
            load_spectrogram(f, image_shape=(128,128)).flatten() for f in real_spectrogram_files
        ])

    if mode == 'hdf5':
        to_hdf5(pipeline, num_samples, output_path, table_name, label, batch_size, num_inference_steps)
    else:
        to_img(
            pipeline, 
            num_samples, 
            output_path=output_path, 
            batch_size=batch_size, 
            real_spectrograms=real_spectrograms,
            num_inference_steps=num_inference_steps
            )

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path', type=str, help='Path to where the generator is saved')
    parser.add_argument('--mode', type=str, choices=['hdf5', 'img'], default='img', help="Specify dataset mode: 'img' to generate images, 'hdf5' for tos tore in HDF5 datasets.")
    parser.add_argument('--num_samples', default=10, type=int, help="How many samples to generate.")
    parser.add_argument('--output_path', default=None, type=str, help='Output Path')
    parser.add_argument('--table_name', default='/train', type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--label', default=1, type=int, help='Label to assign the generated images.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of samples to generate per batch.')
    parser.add_argument('--num_inference_steps', default=1000, type=int, help="Number of denoising steps during the reverse-diffusion step.")
    parser.add_argument('--validation', default=None, type=str,
    help=(
        "Whether to validate the results against a reference distribution. "
        "If 'None', no validation is applied. "
        "Provide the path to a folder containing real spectrogram images "
        "to validate the generated spectrograms by comparing their distributions "
        "using Mahalanobis distance."
        )
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    diffusion_inference(
        args.model_path, 
        args.mode, 
        args.num_samples, 
        output_path=args.output_path, 
        table_name=args.table_name, 
        label=args.label, 
        batch_size=args.batch_size, 
        num_inference_steps=args.num_inference_steps,
        validation=args.validation, 
        seed=args.seed)

if __name__ == "__main__":
    main()