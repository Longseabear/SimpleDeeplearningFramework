import torch
from torch import Tensor, nn


class SimpleDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        conv_channels: list[int],
        hidden_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 3:
            raise ValueError("conv_channels must contain exactly 3 values.")

        conv_layers: list[nn.Module] = []
        current_channels = in_channels
        for next_channels in conv_channels:
            conv_layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels
        self.encoder = nn.Sequential(*conv_layers)

        flattened_dim = current_channels * image_size * image_size
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, image_size * image_size),
            nn.Sigmoid(),
        )
        self.image_size = image_size

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        restored = self.decoder(features)
        return restored.view(-1, 1, self.image_size, self.image_size)
