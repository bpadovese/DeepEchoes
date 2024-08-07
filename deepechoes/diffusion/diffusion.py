import argparse
from deepechoes.diffusion.noise_schedulers import NoiseScheduler
from deepechoes.diffusion.dataset import line_dataset, dino_dataset, spec_dataset, Normalize
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from deepechoes.diffusion.nn_architectures.mlp import MLP
from deepechoes.diffusion.nn_architectures.unet import huggingface_unet, UNet
from torch.utils.data import default_collate
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
import lightning as L
import torch
import numpy as np
import time


def get_model_output(model, inputs, timesteps):
    """
    Get the output from the model, handling cases where the model's output is a dictionary.

    This function is designed to support models that may return outputs in different formats,
    including Hugging Face models that return a dictionary with the key 'sample'.

    Args:
        model (torch.nn.Module): The model to generate the output from.
        inputs (torch.Tensor): The input tensor to the model.
        timesteps (torch.Tensor): The timesteps tensor used as an additional input to the model.

    Returns:
        torch.Tensor: The processed output tensor from the model. If the model's output is a
                      dictionary containing the key 'sample', the corresponding value is returned.
                      Otherwise, the output is returned as-is.
    """
    output = model(inputs, timesteps)
    if isinstance(output, dict) and 'sample' in output:
        return output['sample']           
    return output

def custom_collate(batch):
    """
    This function checks if the batch is a tuple with only one tensor. 
    If it is, the function unpacks the tensor from the tuple and 
    calls the default collate function to process it. If the batch 
    contains more than one tensor or is not a tuple, the default 
    collate function is called directly.

    This function was written to handle tiny 1D datasets for testing
    diffusion models.

    In a typical supervised learning scenario, batch[0] would contain 
    the input features, and batch[1] would contain the labels. However,
    in this specific case, we only have one tensor representing the data
    points (coordinates), and there are no labels. 

    Args:
        batch (list): A list of samples from the dataset, where each 
                      sample is expected to be a tuple containing a 
                      single tensor.

    Returns:
        Tensor: A collated batch of tensors.
    """
    if isinstance(batch[0], tuple) and len(batch[0]) == 1:
        # Unpack the single-element tuples
        batch = [item[0] for item in batch]
        return default_collate(batch)
    else:
        # Use the default collate function for other cases
        return default_collate(batch)

# def train(fabric, model, optimizer, dataloader):
#     # Training loop
#     model.train()
#     for epoch in range(num_epochs):
#         for i, batch in enumerate(dataloader):
            

def reverse_diffusion(fabric, model, num_timesteps, noise_scheduler, batch_shape):
    model.eval()
    with torch.no_grad():
        # Start with random noise
        noisy = torch.randn(batch_shape, device=fabric.device)  # Use the shape from the training set

        for t in tqdm(reversed(range(num_timesteps)), desc="Reverse Diffusion"):
            t_batch = torch.tensor([t] * batch_shape[0], device=fabric.device).long()
            # Predict the noise using the model
            noise_pred = get_model_output(model, noisy, t_batch)
            # Remove the predicted noise
            noisy = noise_scheduler.step(noise_pred, t, noisy).prev_sample
    return noisy

def create_spec(samples, epoch, output_folder):
    # Save the evaluated image or any other required outputs
    image_folder = Path(output_folder) / "images"
    image_folder.mkdir(parents=True, exist_ok=True)

    eval_image = (samples + 1) / 2  # Assuming the image was normalized to [-1, 1]
    eval_image = eval_image.squeeze().cpu().numpy()

    plt.imshow(eval_image, cmap='viridis', aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(image_folder / f"epoch_{epoch}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def create_image(samples, epoch, output_folder):
    # Save the evaluated image or any other required outputs
    image_folder = Path(output_folder) / "images"
    image_folder.mkdir(parents=True, exist_ok=True)

    sample = samples[0]  # Remove the batch dimension
    sample = sample.permute(1, 2, 0)  # Change shape to (128, 128, 3)
    
    # Normalize or scale the sample if needed
    sample = (sample - sample.min()) / (sample.max() - sample.min())  # Normalize to [0, 1]
    sample = (sample * 255).byte().cpu().numpy()  # Convert to byte and move to CPU

    # Create an image from the numpy array
    image = Image.fromarray(sample)

    # Save the image
    image_path = image_folder / f"epoch_{epoch}.png"
    image.save(image_path)

    print(f"Saved image at: {image_path}")

def create_image_grid(samples, epoch, output_folder):
    # Save the evaluated images or any other required outputs
    image_folder = Path(output_folder) / "images"
    image_folder.mkdir(parents=True, exist_ok=True)

    if samples.ndim == 4:  # 2D images case
        eval_images = (samples + 1) / 2  # Assuming images were normalized to [-1, 1]
        eval_images = eval_images.squeeze().cpu().numpy()

        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = axs.flatten()

        for img, ax in zip(eval_images, axs):
            ax.imshow(img, cmap='viridis', aspect='auto')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(image_folder / f"epoch_{epoch}.png")
        plt.close()

    elif samples.ndim == 2:  # 1D points case
        frames = samples.cpu().numpy()
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6

        # for i, frame in enumerate(frames):  # Limiting to first 16 samples for consistency
        plt.figure(figsize=(10, 8))
        plt.scatter(frames[:, 0], frames[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(image_folder / f"epoch_{epoch}.png")
        plt.close()


def main(dataset="dino", train_table='/train', output_folder=None, train_batch_size=32, eval_batch_size=16,
         num_epochs=200, learning_rate=1e-4, num_timesteps=50, beta_schedule="linear", embedding_size=128,
         hidden_size=128, hidden_layers=3, time_embedding="sinusoidal", input_embedding="sinusoidal",
         save_images_step=1):
    
    fabric = L.Fabric(accelerator='gpu')
    # fabric = L.Fabric(accelerator='gpu', precision="16-mixed") # Depdning on how old your GPU is, mixed precision might even slow down processing
    if dataset == "dino":
        dataset = dino_dataset()
        model = MLP(
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            emb_size=embedding_size,
            time_emb=time_embedding,
            input_emb=input_embedding)
        clip_sample = False # we dont want to clip the sample as the data are simply points in a cartesian plane
    
    elif dataset == "butterfly":
        print("Loading butterfly dataset")
        from datasets import load_from_disk
        dataset = load_from_disk("common/datasets/smithsonian_butterflies_train")
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)
        clip_sample = True
        model = huggingface_unet()
    else:
        dataset = spec_dataset(dataset, train_table)
        normalize_transform = Normalize(dataset.min_value, dataset.max_value)
        dataset.set_transform(normalize_transform)
        model = huggingface_unet()
        clip_sample = True # We want to clip the values of our generations to -1 - 1

    data_loader = DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, drop_last=False, collate_fn=custom_collate
        )
    print(len(data_loader))
    # noise_scheduler = NoiseScheduler(
    #     num_timesteps=num_timesteps,
    #     beta_schedule=beta_schedule,
    #     device=fabric.device)
    print(f"Setting up DDPMScheduler with {num_timesteps} timesteps...")
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps, clip_sample=clip_sample)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(data_loader) * num_epochs),
    )

    model, optimizer, lr_scheduler = fabric.setup(model, optimizer, lr_scheduler)
    data_loader = fabric.setup_dataloaders(data_loader)

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    losses = []
    batch_shape = None
    print("Saving images at each epoch...")
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {epoch}")
        model.train()
        for step, batch in enumerate(data_loader):
            batch = batch["images"]
            # start_time = time.time()  # Record the start time
            # print(batch_idx)
            batch_shape = batch.shape  # Get the shape of the batch
            noise = torch.randn(batch_shape, device=fabric.device) # Creating a noise tensor (Gaussian distribution) with the same shape as the batch

            # For each item in the batch, this selects a random timestep from 0 to num_timesteps - 1 when noise will be added.
            timesteps = torch.randint(0, num_timesteps, (batch_shape[0],), device=fabric.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = get_model_output(model, noisy, timesteps)

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping to prevent the gradients from exploding

            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            losses.append(loss.detach().item())

            global_step += 1

            # end_time = time.time()  # Record the end time
            # batch_processing_time = end_time - start_time  # Calculate the processing time
            # print(batch_processing_time)

        if epoch % save_images_step == 0 or epoch == num_epochs - 1:
            image_folder = output_folder / "images"
            image_folder.mkdir(parents=True, exist_ok=True)
            
            bs = (eval_batch_size, *batch_shape[1:])
            samples = reverse_diffusion(fabric, model, num_timesteps, noise_scheduler, bs)
            create_image(samples, epoch, output_folder)
        
        # print(f"Epoch {epoch} completed, loss: {loss.item()}")
    
    torch.save(model.state_dict(), output_folder / f"spec.pth")
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="dino", type=str)
    parser.add_argument('--train_table', default='/train', type=str)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    args = parser.parse_args()

    main(dataset=args.dataset, train_table=args.train_table, output_folder=args.output_folder, train_batch_size=args.train_batch_size,
         eval_batch_size=args.eval_batch_size, num_epochs=args.num_epochs, learning_rate=args.learning_rate,
         num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule, embedding_size=args.embedding_size,
         hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, time_embedding=args.time_embedding,
         input_embedding=args.input_embedding, save_images_step=args.save_images_step)


    

    
