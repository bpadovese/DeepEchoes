import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from torch import nn, optim
from accelerate import Accelerator

# === Basic CNN-based Generator and Critic ===

class Generator(nn.Module):
    def __init__(self, z_dim, kernel_size=4, image_channels=1, features_g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 16, kernel_size, 1, 0, bias=False),  # (B, 1024, 4, 4)
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size, 2, 1, bias=False),  # 8
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size, 2, 1, bias=False),  # 64
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_g, image_channels, kernel_size, 2, 1, bias=False),  # 128
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

class PhaseShuffle(nn.Module):
    def __init__(self, shift_range=2):
        super().__init__()
        self.shift_range = shift_range

    def forward(self, x):
        if self.shift_range == 0:
            return x

        phase = int(torch.randint(-self.shift_range, self.shift_range + 1, (1,)))
        return torch.roll(x, shifts=phase, dims=3)  # Shift along width
    
class Critic(nn.Module):
    def __init__(self, image_channels=1, kernel_size=4, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, features_d, kernel_size, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),  # Phase shuffle layer

            nn.Conv2d(features_d, features_d * 2, kernel_size, 2, 1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),  # Phase shuffle layer

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size, 2, 1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),  # Phase shuffle layer

            nn.Conv2d(features_d * 4, features_d * 8, kernel_size, 2, 1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),  # Phase shuffle layer

            nn.Conv2d(features_d * 8, 1, kernel_size, 1, 0),
        )

    def forward(self, x):
        x = self.net(x)
        return x.mean(dim=(2, 3)).squeeze(1)  # shape: [B]



# === Utility Functions ===

def gradient_penalty(critic, real, fake, device="cuda"):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    prob_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)  # Per-sample L2 norm

    # penalty
    gp = ((grad_norm - 1) ** 2).mean()
    return gp, grad_norm.mean()


def save_image_grid(images, path, nrow=4):
    images = (images + 1) / 2  # denormalize from [-1, 1] to [0, 1]
    utils.save_image(images, path, nrow=nrow)


# === Dataset Loader ===

def load_image_dataset(path, transform=None):
    """
    Automatically load an ImageFolder dataset.
    
    - If the path contains subfolders → load all classes.
    - If the path is a leaf folder with images → treat it as one class.

    Returns:
        dataset (ImageFolder or Subset)
    """
    # Check if there are any subdirectories (class folders)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    if subdirs:
        # Has subdirectories — treat as full ImageFolder
        dataset = ImageFolder(root=path, transform=transform, loader=lambda path: Image.open(path))
    else:
        # Leaf folder with images — treat as single class
        parent_path = os.path.dirname(os.path.normpath(path))
        class_name = os.path.basename(os.path.normpath(path))

        # Load from parent but filter only target class
        full_dataset = ImageFolder(root=parent_path, transform=transform, loader=lambda path: Image.open(path))
        
        # Get the index of the class name
        class_to_idx = full_dataset.class_to_idx
        if class_name not in class_to_idx:
            raise ValueError(f"Class folder '{class_name}' not found in {parent_path}.")
        
        label = class_to_idx[class_name]
        indices = [i for i, (_, l) in enumerate(full_dataset.samples) if l == label]
        dataset = Subset(full_dataset, indices)
    
    return dataset


# === Training ===

def train_loop(config):
    if config['seed'] is not None:
        # Set seeds for reproducibility
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    accelerator = Accelerator(mixed_precision=config['mixed_precision'])
    device = accelerator.device

    # Models
    G = Generator(config['z_dim'], image_channels=config['channels'], features_g=64).to(device)
    D = Critic(config['channels'], features_d=64).to(device)

    # Optimizers
    if config['gan_type'] == "dcgan":
        opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    else:
        opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.9))

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = load_image_dataset(config['dataset'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    G, D, opt_G, opt_D, dataloader = accelerator.prepare(G, D, opt_G, opt_D, dataloader)


    # Training loop
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(
            total=len(dataloader),
            leave=False,  # ✅ Important: Do not keep old bars
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process
        )
        
        # pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for real, _ in dataloader:
            real = real.to(device)

            # === Train Critic ===
            for _ in range(config['critic_iters']):
                noise = torch.randn(real.size(0), config['z_dim'], 1, 1, device=device)
                fake = G(noise).detach()
                
                # Add instance noise
                noise_std = config.get("instance_noise_std", 0.01)
                real_noisy = real + noise_std * torch.randn_like(real)
                fake_noisy = fake + noise_std * torch.randn_like(fake)

                if config['gan_type'] == "wgan-gp":
                    # loss_D = -D(real).mean() + D(fake).mean()
                    # gp, grad_norm_mean = gradient_penalty(D, real, fake, device)
                    # loss_D_total = loss_D + config['lambda_gp'] * gp
                    loss_D = -D(real_noisy).mean() + D(fake_noisy).mean()
                    gp, grad_norm_mean = gradient_penalty(D, real_noisy, fake_noisy, device)
                    loss_D_total = loss_D + config['lambda_gp'] * gp
                else:  # DCGAN
                    real_labels = torch.ones(real.size(0), device=device)
                    fake_labels = torch.zeros(real.size(0), device=device)
                    loss_fn = nn.BCEWithLogitsLoss()

                    D_real_out = D(real).squeeze()
                    D_fake_out = D(fake).squeeze()

                    loss_D_real = loss_fn(D_real_out, real_labels)
                    loss_D_fake = loss_fn(D_fake_out, fake_labels)
                    loss_D_total = loss_D_real + loss_D_fake
                
                # loss_D = -D(real).mean() + D(fake).mean()

                # gp = gradient_penalty(D, real, fake, device)
                # loss_D_total = loss_D + config['lambda_gp'] * gp

                opt_D.zero_grad()
                accelerator.backward(loss_D_total)
                opt_D.step()

            # === Train Generator ===
            noise = torch.randn(real.size(0), config['z_dim'], 1, 1, device=device)
            fake = G(noise)
            
            if config['gan_type'] == "wgan-gp":
                loss_G = -D(fake).mean()
            else:
                fake_labels = torch.ones(fake.size(0), device=device)  # Trick the discriminator
                D_fake_out = D(fake).squeeze()
                loss_fn = nn.BCEWithLogitsLoss()
                loss_G = loss_fn(D_fake_out, fake_labels)

            opt_G.zero_grad()
            accelerator.backward(loss_G)
            opt_G.step()

            progress_bar.set_postfix({
                "D": f"{loss_D_total.item():.2f}",
                "G": f"{loss_G.item():.2f}",
                # "norm": f"{grad_norm_mean.item():.2f}"
            })
            progress_bar.update(1)
        progress_bar.close()

        if accelerator.is_main_process and (epoch + 1) % config['save_image_epochs'] == 0:
            with torch.no_grad():
                noise = torch.randn(16, config['z_dim'], 1, 1, device=device)
                samples = G(noise).cpu()
                os.makedirs(config['output_dir'], exist_ok=True)
                save_image_grid(samples, f"{config['output_dir']}/epoch_{epoch+1}.png")

    if accelerator.is_main_process:
        torch.save(G.state_dict(), os.path.join(config['output_dir'], "generator_final.pth"))
        torch.save(D.state_dict(), os.path.join(config['output_dir'], "critic_final.pth"))

#Inference
@torch.no_grad()
def generate_only(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(config['z_dim'], image_channels=config['channels'], features_g=64).to(device)

    if config['pretrained_generator'] is None:
        raise ValueError("You must provide --pretrained_generator to use --generate_only.")

    # Load weights
    G.load_state_dict(torch.load(config['pretrained_generator'], map_location=device))
    G.eval()

    # Generate noise and samples with 1 channel
    noise = torch.randn(config['num_samples'], config['z_dim'], 1, 1, device=device)
    samples = G(noise).cpu()

    # Save output
    os.makedirs(config['output_dir'], exist_ok=True)
    # save_path = os.path.join(config['output_dir'], f"samples.png")
    # save_image_grid(samples, save_path, nrow=4)
    # print(f"Saved generated samples to {save_path}")

    for i, img_tensor in enumerate(samples):
        img = to_pil_image(img_tensor)
        img.save(os.path.join(config['output_dir'], f"gan_sample_{i}.png"))

    print(f"Saved {config['num_samples']} GAN samples to {config['output_dir']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to image dataset. Required unless --generate_only is set.")
    parser.add_argument("--generate_only", action="store_true",
                    help="Only generate samples using a pretrained generator (no training).")
    parser.add_argument("--pretrained_generator", type=str, default=None,
                        help="Path to the pretrained generator .pth file")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples to generate when in generate_only mode.")
    parser.add_argument("--output_dir", type=str, default="wgan_output")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--gan_type", type=str, choices=["wgan-gp", "dcgan"], default="wgan-gp",
                    help="Which GAN variant to train.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--critic_iters", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"])
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    if not args.generate_only and args.dataset is None:
        raise ValueError("You must provide a dataset path unless --generate_only is used.")

    config = vars(args)
    if config["generate_only"]:
        generate_only(config)
    else:
        train_loop(config)
