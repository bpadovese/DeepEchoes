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

def save_reverse_diffusion_steps(pipeline, output_dir, num_inference_steps=10, seed=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = pipeline.device  # usually 'cuda'
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = torch.Generator(device=device)

    # 1. Start from noise
    batch_size = 1
    shape = pipeline.unet.config.sample_size
    latents = torch.randn(
        (batch_size, pipeline.unet.config.in_channels, shape, shape),
        generator=generator,
        device=device,
    )

    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_pred = pipeline.unet(latents, t)["sample"]
            latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

        # Convert to numpy and reshape
        image = latents.detach().cpu().squeeze().numpy()

        image.save(output_dir / f"step_{i:03d}.png")

        # # Save as a spectrogram-like image
        # fig, ax = plt.subplots(figsize=(3, 3))
        # ax.imshow(image, aspect='auto', origin='lower', cmap='viridis')
        # ax.axis('off')
        # fig.savefig(output_dir / f"step_{i:03d}.png", bbox_inches='tight')
        # plt.close(fig)

    print(f"Saved {num_inference_steps} denoising steps to {output_dir}")

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
                # Save images
                for i, image in enumerate(images):
                    if num_valid_samples >= num_samples:
                        break
                    image.save(output_path / f"diffusion_{num_valid_samples}.png")
                    num_valid_samples += 1
                    pbar.update(1)
                

def diffusion_inference(model_path, num_samples, output_path=None, label=1, batch_size=8, num_inference_steps=1000, validation=None, seed=None):
    
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

    # save_reverse_diffusion_steps(pipeline, output_dir=output_path, num_inference_steps=num_inference_steps, seed=seed)
    # exit()
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
    parser.add_argument('--num_samples', default=10, type=int, help="How many samples to generate.")
    parser.add_argument('--output_path', default=None, type=str, help='Output Path')
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
        args.num_samples, 
        output_path=args.output_path, 
        label=args.label, 
        batch_size=args.batch_size, 
        num_inference_steps=args.num_inference_steps,
        validation=args.validation, 
        seed=args.seed)

if __name__ == "__main__":
    main()