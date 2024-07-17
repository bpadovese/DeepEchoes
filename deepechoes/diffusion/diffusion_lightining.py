import argparse
from deepechoes.diffusion.noise_schedulers import NoiseScheduler
from deepechoes.diffusion.dataset import line_dataset, dino_dataset, spec_dataset
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from deepechoes.diffusion.nn_architectures.mlp import MLP
from deepechoes.diffusion.nn_architectures.unet import huggingface_unet, UNet
import lightning as L
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
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

    hidden_size = args.hidden_size
    hidden_layers = args.hidden_layers
    embedding_size = args. embedding_size
    time_embedding = args. time_embedding
    input_embedding = args.input_embedding
    num_timesteps = args.num_timesteps
    beta_schedule = args.beta_schedule
    learning_rate = args.learning_rate
    output_folder = args.output_folder
    eval_batch_size = args.eval_batch_size

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = dino_dataset()
    dataset = spec_dataset(args.dataset, args.train_table)
    data_loader = DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False
        )
    
    # model = MLP(
    #     hidden_size=hidden_size,
    #     hidden_layers=hidden_layers,
    #     emb_size=embedding_size,
    #     time_emb=time_embedding,
    #     input_emb=input_embedding).to(device)

    model = huggingface_unet().to(device)
    # model = UNet().to(device)
    noise_scheduler = NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    losses = []
    print("Saving images at each epoch...")
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            noise = torch.randn(batch.shape).to(device) # Creating a noise tensor (Gaussian distribution) with the same shape as the batch
            # For each item in the batch, this selects a random timestep from 0 to num_timesteps - 1 when noise will be added.
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long().to(device)
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            noise_pred = model(noisy, timesteps)
            print(noise_pred.sample)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping to prevent the gradients from exploding
            optimizer.step()

            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())

            global_step += 1

        if epoch % args.save_images_step == 0 or epoch == args.num_epochs - 1:
            # generated_data = model.generate_samples(noise_scheduler=noise_scheduler, eval_batch_size=eval_batch_size, device=device) 
            image_folder = output_folder / "images"
            image_folder.mkdir(parents=True, exist_ok=True)
            
            model.eval()
            with torch.no_grad():
                # Sample images for evaluation
                eval_noise = torch.randn((eval_batch_size, 1, 128, 128), device=device)
                timesteps = list(range(len(noise_scheduler)))[::-1]  # Reverse the list

                for t in timesteps:
                    t = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
                    with torch.no_grad():
                        residual = model(eval_noise, t, return_dict=False)[0]
                    eval_noise = noise_scheduler.step(residual, t[0], eval_noise)

                # Optionally apply any necessary transformations to the generated images (e.g., scaling, denormalizing)
                eval_images = (eval_noise + 1) / 2  # Assuming images were normalized to [-1, 1]
                print(eval_images.shape)
                # # Save generated images for this epoch
                # from torchvision.transforms.functional import to_pil_image
                # pil_images = [to_pil_image(img.squeeze(0)) for img in eval_images]  # Remove channel dimension if it's 1
                # Convert images to numpy array and remove extra dimensions
                eval_images = eval_images.squeeze().cpu().numpy()

                # Create a 4x4 grid of images
                fig, axs = plt.subplots(4, 4, figsize=(8, 8))
                axs = axs.flatten()

                for img, ax in zip(eval_images[:16], axs):
                    ax.imshow(img, cmap='viridis', aspect='auto')
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(image_folder / f"epoch_{epoch}.png")
                plt.close()



            # np.save(output_folder / f"generated_data_epoch_{epoch}_batch_{batch_idx}.npy", generated_data)

        print(f"Epoch {epoch} completed, loss: {loss.item()}")
    
    torch.save(model.state_dict(), output_folder / f"spec.pth")
    print("Training completed.")

    

    

    # image_folder = output_folder / "images"
    # image_folder.mkdir(parents=True, exist_ok=True)
    # frames = np.stack(frames)
    # xmin, xmax = -6, 6
    # ymin, ymax = -6, 6
    # for i, frame in enumerate(frames):
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(frame[:, 0], frame[:, 1])
    #     plt.xlim(xmin, xmax)
    #     plt.ylim(ymin, ymax)
    #     plt.savefig(image_folder / f"{i}.png")
    #     plt.close()

    # exit()
    # main(**vars(args))
    
