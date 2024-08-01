import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel



def huggingface_unet():
    model = UNet2DModel(
        sample_size=128,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        # downsample_type="resnet",
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        # upsample_type="resnet",
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

unet_config = {
    "sample_size": 128,                  # Size of the generated images (spectrograms)
    "in_channels": 1,                    # Number of input channels (1 for grayscale spectrograms)
    "out_channels": 1,                   # Number of output channels (1 for grayscale spectrograms)
    "down_block_types": (                # Types of down-sampling blocks
        "DownBlock2D", 
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",               # Adding attention layers can help the model focus on important features
        "DownBlock2D"
    ),
    "up_block_types": (                  # Types of up-sampling blocks
        "UpBlock2D",
        "AttnUpBlock2D",                 # Adding attention layers can help the model focus on important features
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
    "block_out_channels": (64, 128, 256, 512, 1024),  # Number of output channels for each block
    "layers_per_block": 2,               # Number of layers in each block
    "attention_resolutions": [16, 8],    # Resolutions at which attention is applied
    "norm_num_groups": 32,               # Number of groups for group normalization
    "dropout": 0.1                       # Dropout rate to avoid overfitting
}

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, t):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec4 = self.decoder4(F.interpolate(bottleneck, scale_factor=2, mode="nearest"))
        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2, mode="nearest"))
        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2, mode="nearest"))
        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2, mode="nearest"))

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def generate_samples(self, noise_scheduler, eval_batch_size, img_size, device='cpu'):
        self.eval()
        sample = torch.randn(eval_batch_size, 1, img_size, img_size).to(device)  # Drawing from Gaussian distribution
        timesteps = list(range(len(noise_scheduler)))[::-1]  # Reverse the list

        for t in timesteps:
            t = torch.from_numpy(np.repeat(t, eval_batch_size)).long().to(device)
            with torch.no_grad():
                residual = self(sample)
            sample = noise_scheduler.step(residual, t[0], sample)
        
        return sample.cpu().numpy()