import torch
from torch import Tensor, nn


class BasicDenoiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SimpleDenoiser(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.blocks = nn.Sequential(
            *[BasicDenoiseBlock(hidden_channels, kernel_size=kernel_size) for _ in range(num_blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        features = torch.relu(self.stem(x))
        residual = self.head(self.blocks(features))
        return torch.clamp(x - residual, 0.0, 1.0)
