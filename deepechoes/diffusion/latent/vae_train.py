from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from diffusers import AutoencoderKL
from PIL import Image
from packaging import version
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss, weights_init, NLayerDiscriminator
import warnings
import shutil
import gc
import os
import numpy as np
import math
import torch.nn.functional as F
import torch
import random
import argparse
import lpips
import glob

if is_wandb_available():
    import wandb

def sample_random_images(dataset_path, num_samples=8, extensions=("jpg", "jpeg", "png")):
    # Recursively find all image files with the given extensions
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, f"**/*.{ext}"), recursive=True))

    # Ensure there are enough images to sample
    if len(image_files) < num_samples:
        print(f"Warning: Only found {len(image_files)} images, sampling all.")
        return image_files  # Return all images if fewer than `num_samples`
    
    return random.sample(image_files, num_samples)

def to_pil(images):
    """
    Converts a batch of grayscale images (torch tensors) to uint8 [0, 255]
    """

    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images * 255).round().byte()

    # (B, 1, H, W) -> (B, H, W)
    images = images.squeeze(1) if images.shape[1] == 1 else images  

    # Convert to NumPy (B, H, W)
    images = images.cpu().numpy()  

    pil_imgs = [Image.fromarray(img, mode="L") for img in images]
    return pil_imgs[0] if len(pil_imgs) == 1 else pil_imgs

    # # Save each image in batch
    # for idx, img_array in enumerate(images):
    #     pil_image = Image.fromarray(img_array, mode="L")  # "L" mode for grayscale
    #     save_path = os.path.join(val_dir, f"image_step{step}_sample{idx}.png")
    #     pil_image.save(save_path)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("L", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

@torch.no_grad()
def validate(vae, args, accelerator, step):
    vae = accelerator.unwrap_model(vae)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    images = []
    gen_imgs = []
    orignal_imgs = []


    for i, validation_image in enumerate(args.validation_image):
        validation_image = Image.open(validation_image).convert("L")
        targets = transform(validation_image).to(accelerator.device)
        targets = targets.unsqueeze(0)

        reconstructions = vae(targets).sample # forward pass that encodes and decodes
        
        gen_imgs.append(to_pil(targets))
        orignal_imgs.append(to_pil(reconstructions))


        images.append(
            torch.cat([targets.cpu(), reconstructions.cpu()], axis=0)
        )
    
    ### Concat Images (so we can compare real vs reconstruction) ###
    img_width = orignal_imgs[0].width
    img_height = orignal_imgs[0].height
    combined_images = []
    for orig_img, gen_img in zip(orignal_imgs, gen_imgs):
        combined_img = Image.new(mode="L", size=(img_width, 2*img_height))
        combined_img.paste(orig_img, (0,0))
        combined_img.paste(gen_img, (0,img_height))
        combined_images.append(combined_img)

    ## Concatenate All Samples Together ###
    final_image = Image.new(mode="L", size=(img_width*len(combined_images), 2*img_height))
    x_offset = 0
    for img in combined_images:
        final_image.paste(img, (x_offset,0))
        x_offset += img_width

    ### Save Output ###

    val_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(val_dir, exist_ok=True)
    path_to_save = os.path.join(val_dir, f"iteration_{step}.png")
    final_image.save(path_to_save)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # np_images = np.stack([np.asarray(img) for img in images])
            np_images = torch.cat(images, dim=0)
            tracker.writer.add_images(
                "Original (left), Reconstruction (right)", np_images, step
            )
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left), Reconstruction (right)": [
                        wandb.Image(torchvision.utils.make_grid(image))
                        for _, image in enumerate(images)
                    ]
                }
            )
        gc.collect()
        torch.cuda.empty_cache()
    return images

def main(args):
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Convert to tensor [0, 1]
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    train_dataset = ImageFolder(root=args.dataset_path, transform=transform, loader=lambda path: Image.open(path))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load AutoencoderKL model
    config = AutoencoderKL.load_config("deepechoes/diffusion/latent/config.json")
    vae = AutoencoderKL.from_config(config)
    lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
    discriminator = NLayerDiscriminator(input_nc=1, n_layers=3, use_actnorm=False).apply(weights_init)

    vae.requires_grad_(True)
    # if args.decoder_only:
    #     vae.encoder.requires_grad_(False)
    #     if getattr(vae, "quant_conv", None):
    #         vae.quant_conv.requires_grad_(False)
    vae.train()
    discriminator.requires_grad_(True)
    discriminator.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                warnings.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Optimizer
    params_to_optimize = filter(lambda p: p.requires_grad, vae.parameters())
    disc_params_to_optimize = filter(lambda p: p.requires_grad, discriminator.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )
    disc_optimizer = torch.optim.AdamW(
        disc_params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_epochs

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
        num_cycles=1,
        power=1.0,
    )
    disc_lr_scheduler = get_scheduler(
        args.disc_lr_scheduler,
        optimizer=disc_optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
        num_cycles=1,
        power=1.0,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=os.path.join(args.output_dir, "logs"),
    )


    # Prepare everything. There is no order
    vae, discriminator, optimizer, disc_optimizer, train_dataloader, lr_scheduler, disc_lr_scheduler = accelerator.prepare(
        vae, discriminator, optimizer, disc_optimizer, train_dataloader, lr_scheduler, disc_lr_scheduler
    )

    # vae.to(accelerator.device)
    lpips_loss_fn.to(accelerator.device)
    # discriminator.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {num_training_steps}")

    if not args.validation_image:
        args.validation_image = sample_random_images(args.dataset_path)
        print(f"  Warning: --validation_image argument not set. Therefore sampling 8 random iamges from {args.dataset_path}")
    
    # ToDo: change later when loading from checkpoint
    initial_global_step = 0 
    first_epoch = 0
    global_step = 0

    progress_bar = tqdm(
        range(0, num_training_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Training loop
    for epoch in range(first_epoch, args.num_epochs):
        vae.train()
        discriminator.train()
        for step, batch in enumerate(train_dataloader):
            # Forward pass: Encode and decode
            # Convert images to latent space 
            targets = batch[0]
            posterior = vae.encode(targets).latent_dist
            latents = posterior.sample()

            # reconstruct from latent space
            reconstructions = vae.decode(latents).sample

            # Determine whether to train the VAE (generator) or the discriminator
            if (step // args.gradient_accumulation_steps) % 2 == 0 or global_step < args.disc_start:
                # Train the VAE (Generator)
                with accelerator.accumulate(vae):  # Accelerator will handle gradient acccumulation if set
                    
                    # Reconstruction loss: Compare pixel-level differences between input and output
                    if args.rec_loss == "l2":
                        rec_loss = F.mse_loss(reconstructions.float(), targets.float(), reduction="none")
                    else:
                        rec_loss = F.l1_loss(reconstructions.float(), targets.float(), reduction="none")
                    
                    # Perceptual loss: Measures high-level feature similarity
                    with torch.no_grad():
                        lpips_loss = lpips_loss_fn(reconstructions, targets)

                    # Combine reconstruction and perceptual losses
                    perceptual_loss = rec_loss + args.perceptual_scale * lpips_loss
                    perceptual_loss = torch.sum(perceptual_loss) / perceptual_loss.shape[0]  # Normalize loss over batch

                    # KL Divergence loss: Regularizes latent space
                    kl_loss = posterior.kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                    # Adversarial loss (Generator loss): Discriminator's feedback on reconstructions
                    logits_fake = discriminator(reconstructions)
                    g_loss = -torch.mean(logits_fake)  # Typical generator loss encourages generator to fool discriminator

                    # Compute discriminator weight using gradient norms
                    last_layer = accelerator.unwrap_model(vae).decoder.conv_out.weight
                    p_grads = torch.autograd.grad(perceptual_loss, last_layer, retain_graph=True)[0]
                    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
                    adaptive_weight = torch.norm(p_grads) / (torch.norm(g_grads) + 1e-4)
                    adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
                    adaptive_weight = adaptive_weight * args.adaptive_scale

                    # Set discriminator contribution factor
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0

                    # Final generator loss: Reconstruction + KL + adversarial loss (scaled by adaptive weight)
                    loss = perceptual_loss + args.kl_scale * kl_loss + adaptive_weight * disc_factor * g_loss

                    # Logging training stats
                    logs = {
                        "loss": loss.detach().mean().item(),
                        "p_loss": perceptual_loss.detach().mean().item(),
                        "rec_loss": rec_loss.detach().mean().item(),
                        "lpips_loss": lpips_loss.detach().mean().item(),
                        "kl_loss": kl_loss.detach().mean().item(),
                        "adaptive_w": adaptive_weight.detach().mean().item(),
                        "disc_factor": disc_factor,
                        "g_loss": g_loss.detach().mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch
                    }

                    # Backpropagation and optimizer step
                    accelerator.backward(loss)  # Compute gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()  # Update VAE weights
                    lr_scheduler.step()  
                    optimizer.zero_grad() 

            else:
                # Train the Discriminator
                with accelerator.accumulate(discriminator):  # Accumulate gradients for discriminator
                    
                    # Compute discriminator scores for real and fake (reconstructed) images
                    logits_real = discriminator(targets)
                    logits_fake = discriminator(reconstructions)

                    # Select discriminator loss function (Hinge loss or Vanilla GAN loss)
                    disc_loss_fn = hinge_d_loss if args.disc_loss == "hinge" else vanilla_d_loss
                    
                    # Apply scaling factor (discriminator only starts training after `disc_start` steps)
                    disc_factor = args.disc_factor if global_step >= args.disc_start else 0.0
                    disc_loss = disc_factor * disc_loss_fn(logits_real, logits_fake)

                    # Logging discriminator training stats
                    logs = {
                        "disc_loss": disc_loss.detach().mean().item(),
                        "logits_real": logits_real.detach().mean().item(),
                        "logits_fake": logits_fake.detach().mean().item(),
                        "disc_lr": disc_lr_scheduler.get_last_lr()[0]
                    }

                    # # Backpropagation and optimizer step for Discriminator
                    # accelerator.backward(disc_loss)
                    # if accelerator.sync_gradients:
                    #     accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)  # Gradient clipping
                    # disc_optimizer.step() # Update disc weights
                    # disc_lr_scheduler.step()
                    # disc_optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Conditions
                is_validation_step = (global_step) % args.validation_steps == 0
                is_last_epoch = epoch == args.num_epochs - 1
                is_save_model_epoch = (epoch + 1) % args.save_model_epochs == 0

                if accelerator.is_main_process:
                    if is_save_model_epoch:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # keeping up to 5 checkpoints
                        if len(checkpoints) >= 5:
                            for ckpt in checkpoints[:len(checkpoints) - 5 + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-epoch-{epoch}")
                        accelerator.save_state(save_path)

                    if is_validation_step:
                        validate(
                            vae,
                            args,
                            accelerator,
                            global_step,
                        )

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        validate(
            vae,
            args,
            accelerator,
            global_step,
        )
        vae = accelerator.unwrap_model(vae)
        discriminator = accelerator.unwrap_model(discriminator)
        vae.save_pretrained(args.output_dir)
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))

    accelerator.end_training()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the image folder containing spectrogram images.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=4.5e-6, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="my_model", help="Directory to save the trained model.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"], help="Mixed precision setting.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--disc_start", type=int, default=50001, help="Starting step for the discriminator. Usually, we want the dicriminator start after the VAE as to not overpower the VAE too early",)
    parser.add_argument("--report_to", type=str, default="tensorboard", help='Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--tracker_project_name", type=str, default="train_autoencoderkl", help="The `project_name` argument passed to Accelerator.init_trackers ")
    parser.add_argument("--rec_loss", type=str, default="l2", help="The loss function for VAE reconstruction loss.")
    parser.add_argument("--disc_loss", type=str, default="hinge",help="Loss function for the discriminator")
    parser.add_argument("--disc_factor", type=float, default=1.0, help="Scaling factor for the discriminator")
    parser.add_argument("--adaptive_scale", type=float, default=1.0, help="Scaling factor for the discriminator")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="Epochs interval to save the model.")
    parser.add_argument("--validation_steps", type=int, default=10, help="Steps interval to evaluate generated images.")
    parser.add_argument("--validation_image", type=str, default=None, nargs="+", help="A set of paths to the image be evaluated every `--validation_steps` and logged to `--report_to`.")
    parser.add_argument("--kl_scale", type=float, default=1e-6, help="Scaling factor for the Kullback-Leibler divergence penalty term.")
    parser.add_argument("--perceptual_scale", type=float, default=0.5, help="Scaling factor for the LPIPS metric")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
    parser.add_argument("--disc_lr_scheduler", type=str, default="cosine",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()
    main(args)