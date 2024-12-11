from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from diffusers import DDPMScheduler, DDPMPipeline
from PIL import Image
from torchvision import transforms
from diffusers.optimization import get_cosine_schedule_with_warmup
from deepechoes.diffusion.nn_architectures.unet import huggingface_unet
from matplotlib import pyplot as plt
from deepechoes.diffusion.dataset import HDF5Dataset, NormalizeToRange, get_leaf_paths
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
import os
import math
import numpy as np
import torch.nn.functional as F
import torch
import random

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def make_grid_spec(images, cols):
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

def evaluate(config, epoch, pipeline, generator):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # Save the images
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)

    if config['mode'] == 'img':
        images = pipeline(
            batch_size=config['eval_batch_size'],
            generator=generator,
            num_inference_steps=pipeline.scheduler.config.num_train_timesteps,
            return_dict=False,
        )[0]
        
        # denormalize the images and save to tensorboard
        # images = np.array(
        #     [
        #         np.frombuffer(image.tobytes(), dtype="uint8").reshape(
        #             (len(image.getbands()), image.height, image.width)
        #         )
        #         for image in images
        #     ]
        # )
        for idx, image in enumerate(images):  # `images` is a list of PIL.Image objects
            image.save(os.path.join(test_dir, f"image_epoch{epoch}_sample{idx}.png"))

        # image_grid = make_grid(images, rows=4, cols=4)
        # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    else:
        images = pipeline(
            batch_size=config['eval_batch_size'],
            generator=generator,
            output_type="nd.array"
        ).images
        # Make a grid out of the images
        image_grid = make_grid_spec(images, cols=4)
        image_grid.savefig(f"{test_dir}/{epoch:04d}.png", format='png')
        

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], "logs"),
    )
    if accelerator.is_main_process:
        if config['output_dir'] is not None:
            os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()    

        accelerator.wait_for_everyone()

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # Conditions
            is_save_image_epoch = (epoch + 1) % config['save_image_epochs'] == 0
            is_last_epoch = epoch == config['num_epochs'] - 1
            is_save_model_epoch = (epoch + 1) % config['save_model_epochs'] == 0

            if is_save_image_epoch or is_last_epoch or is_save_model_epoch:
                
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                print("saving image...")
                
            if is_save_model_epoch or is_last_epoch:
                pipeline.save_pretrained(config['output_dir'])

            if is_save_image_epoch:
                generator = torch.Generator(device=clean_images.device).manual_seed(42)

                evaluate(config, epoch, pipeline, generator)
            
        accelerator.wait_for_everyone()

    accelerator.end_training()

def select_transform(norm_type, image_size):
    match norm_type:
        case 0:
            # Default: Sample-Wise Normalization [-1, 1]
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                NormalizeToRange(new_min=-1, new_max=1)  # Normalize sample-wise
            ])
        case 2:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        case _:
            raise ValueError("Invalid norm_type value. It should be between 0 and 1.")
        
def main(dataset, mode='img', train_table="/train", image_size=128, train_batch_size=8, eval_batch_size=8, 
         num_epochs=50, num_timesteps=1000, gradient_accumulation_steps=1, learning_rate=1e-4, 
         lr_warmup_steps=500, save_image_epochs=10, save_model_epochs=10, mixed_precision="no", 
         output_dir="trained_models", overwrite_output_dir=True, norm_type=0, seed=None):

    if seed:
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if mode == "img":
        train_transform = select_transform(norm_type, image_size)
        # Load images from the folder
        train_dataset = ImageFolder(root=dataset, transform=train_transform, loader=lambda path: Image.open(path))
    else:
        # Select transforms for the training dataset
        train_transform = select_transform(norm_type, image_size)

        train_datasets = []
        # Handle train_table argument (single path or list of paths)
        if isinstance(train_table, str):
            train_table = [train_table]  # Convert to list if it's a single path
        for path in train_table:
            leaf_paths = get_leaf_paths(dataset, path)
            for leaf_path in leaf_paths:
                train_ds = HDF5Dataset(dataset, leaf_path, transform=None) 
                train_ds.set_transform(train_transform)
                train_datasets.append(train_ds)

        # Concating all train datasets together.
        train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    # Sanity check: print the total number of samples in all datasets combined
    total_samples = len(train_dataset)
    print(f"Total number of samples across all datasets: {total_samples}")

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)

    model = huggingface_unet(channels=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    from accelerate import notebook_launcher

    #  Create a config dictionary from the parsed arguments
    config = {
        "dataset": dataset,
        "mode": mode,
        "eval_batch_size": eval_batch_size,
        "num_epochs": num_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "save_image_epochs": save_image_epochs,
        "save_model_epochs": save_model_epochs,
        "mixed_precision": mixed_precision,
        "output_dir": output_dir,
        "overwrite_output_dir": overwrite_output_dir,
        "seed": seed
    }
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    notebook_launcher(train_loop, args, num_processes=1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument("dataset", type=str, default="img", help="Path to the HDF5 dataset file or image folder.")
    parser.add_argument('--mode', type=str, choices=['hdf5', 'img'], default='img', help="Specify dataset mode: 'img' to generate images, 'hdf5' for tos tore in HDF5 datasets.")
    parser.add_argument('--train_table', type=str, nargs='+', default='/train', help='HDF5 table name for training data.')
    parser.add_argument("--image_size", type=int, default=128, help="Generated image resolution.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps for the scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Epochs interval to save generated images.")
    parser.add_argument("--save_model_epochs", type=int, default=30, help="Epochs interval to save the model.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"], help="Mixed precision setting.")
    parser.add_argument("--output_dir", type=str, default="ddpm-butterflies-128", help="Output directory for the model.")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True, help="Overwrite the output directory if it exists.")
    parser.add_argument('--norm_type', type=int, default=0, help=(
        'Type of normalization/standardization to apply. Default is 0. Options are:\n'
        '0 - Normalize each sample individually to the range [-1, 1].\n'
    ))
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    
    args = parser.parse_args()
    main(args.dataset, args.mode, args.train_table, args.image_size, args.train_batch_size, args.eval_batch_size,
         args.num_epochs, args.num_timesteps, args.gradient_accumulation_steps, args.learning_rate,
         args.lr_warmup_steps, args.save_image_epochs, args.save_model_epochs, args.mixed_precision,
         args.output_dir, args.overwrite_output_dir, args.norm_type, args.seed)
    