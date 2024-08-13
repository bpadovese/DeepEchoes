from diffusers import DiffusionPipeline
from pathlib import Path
from tqdm import tqdm
from deepechoes.utils.hdf5_helper import insert_spectrogram_data, create_or_get_table, create_table_description
import tables as tb
import matplotlib.pyplot as plt
import math

def single_spec_gen():
    # Load the diffusion model
    generator = DiffusionPipeline.from_pretrained("trained_models/diffusion/accelerate").to("cuda")

    # Generate an image
    image = generator(output_type="nd.array", num_inference_steps=1000).images[0]

    # Plot and save the spectrogram
    fig, ax = plt.subplots()
    ax.imshow(image[:, :, 0], aspect='auto', origin='lower', cmap='viridis')
    ax.axis('off')  # Turn off the axis

    # Save the figure
    fig.savefig('spectrogram_image.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free up memory

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

def multiple_spec_gen():
    # Load the diffusion model
    generator = DiffusionPipeline.from_pretrained("trained_models/diffusion/accelerate").to("cuda")

    # Generate an image
    images = generator(batch_size=8, output_type="nd.array", num_inference_steps=1000).images
    spec_grid = make_grid_spec(images, cols=4)
    
    # Save the figure
    spec_grid.savefig('spectrogram_grid.png', bbox_inches='tight')
    
# multiple_spec_gen()

def diffusion_generate_to_hdf5(model_path, num_samples, output_path='diffusion.h5', table_name='/train', label=1, batch_size=8, num_inference_steps=1000):
    if output_path is None:
        output_path = Path('.').resolve()
    else:
        output_path = Path(output_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the diffusion model
    generator = DiffusionPipeline.from_pretrained(model_path).to("cuda")
    generator.set_progress_bar_config(disable=True)

    print('\nCreating db...')
    with tb.open_file(output_path, mode='a') as h5file:
        table = create_or_get_table(h5file, table_name, 'data', create_table_description((128,128)))
        num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed
        print(f'\nGenerating {num_samples} samples with label {label} to table {table_name}...')
        for batch_num in tqdm(range(num_batches)):
            batch_size_adjusted = min(batch_size, num_samples - batch_num * batch_size) # calculate the batch size for the current batch (will be different for the last batch)
            # Generate the images
            images = generator(batch_size=batch_size_adjusted, output_type="nd.array", num_inference_steps=num_inference_steps).images

            for i, representation_data in enumerate(images):
                idx = batch_num * batch_size + i
                filename = f"diffusion_{idx}"

                # Squeeze the last dimension if it's 1
                if representation_data.shape[-1] == 1:
                    representation_data = representation_data.squeeze(-1)
                
                insert_spectrogram_data(table, filename, 0, label, representation_data)

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_path', type=str, help='Path to where the generator is saved')
    parser.add_argument('--mode', type=str, default='hdf5', help='mode to either save to a hdf5 db or plot, or saves 1 image to a "csv"')
    parser.add_argument('--num_samples', default=10, type=int, help="How many samples to generate.")
    parser.add_argument('--output_path', default=None, type=str, help='Output Path')
    parser.add_argument('--table_name', default='/train', type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--label', default=1, type=int, help='Label to assign the generated images.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of samples to generate per batch.')
    parser.add_argument('--num_inference_steps', default=1000, type=int, help="Number of denoising steps during the reverse-diffusion step.")

    args = parser.parse_args()
    mode = args.mode

    if mode == 'hdf5':
        diffusion_generate_to_hdf5(args.model_path, args.num_samples, args.output_path, args.table_name, args.label, args.batch_size, args.num_inference_steps)
    

if __name__ == "__main__":
    main()