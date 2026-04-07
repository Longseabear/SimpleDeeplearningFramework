import math

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticDenoisingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, base_seed: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        row = self.dataframe.iloc[index]
        sample_seed = self.base_seed + (index * 1009) + (int(row["sample_id"]) * 17)
        generator = torch.Generator().manual_seed(sample_seed)

        clean = self._generate_clean_image(generator)
        noise = torch.randn((1, self.image_size, self.image_size), generator=generator) * 0.08
        noisy = torch.clamp(clean + noise, 0.0, 1.0)

        return {
            "input": noisy,
            "target": clean,
            "dataset_name": str(row["dataset_name"]),
            "sample_id": int(row["sample_id"]),
        }

    def _generate_clean_image(self, generator: torch.Generator) -> Tensor:
        coords = torch.linspace(-1.0, 1.0, self.image_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        freq_x = int(torch.randint(1, 4, (1,), generator=generator).item())
        freq_y = int(torch.randint(1, 4, (1,), generator=generator).item())
        phase = float(torch.rand(1, generator=generator).item()) * math.pi
        sigma = 0.2 + float(torch.rand(1, generator=generator).item()) * 0.35
        shift_x = float(torch.rand(1, generator=generator).item()) * 0.6 - 0.3
        shift_y = float(torch.rand(1, generator=generator).item()) * 0.6 - 0.3

        wave = torch.sin(freq_x * math.pi * xx + phase) * torch.cos(freq_y * math.pi * yy + phase)
        blob = torch.exp(-((xx - shift_x) ** 2 + (yy - shift_y) ** 2) / (2 * sigma**2))
        clean = 0.5 * wave + 0.5 * blob
        clean = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)
        return clean.unsqueeze(0)
