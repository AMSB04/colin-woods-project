import torch
import torch.nn as nn
from model_util import init_weights_gan, get_device

class Discriminator(nn.Module):
    """
    PatchGAN discriminator network for distinguishing real vs synthesized images.
    Designed for paired inputs (e.g., concatenated 3T and 7T MRI scans with in_channels=2).

    Args:
            in_channels (int): Number of input channels; typically 2 for concatenated input images.
            features (list of int): Number of filters for each ConvBlock layer.
    """
    def __init__(self, in_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        # Initial conv layer with LeakyReLU, no normalization
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build intermediate layers
        layers = []
        in_ch = features[0]
        for i, out_ch in enumerate(features[1:]):
            stride = 1 if i == len(features) - 2 else 2  # stride=1 on last layer
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = out_ch

        # Final conv layer
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights_gan(m)
        
        self.device = get_device()
        self.to(self.device)

    def forward(self, x):
        x = self.initial(x)
        return self.model(x)