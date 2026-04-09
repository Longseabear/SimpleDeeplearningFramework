from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class MnistDenoisingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, noise_std: float, base_seed: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.noise_std = noise_std
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        row = self.dataframe.iloc[index]
        clean = _load_grayscale_image(Path(row["target_path"]))
        noise = torch.randn(clean.shape, generator=self._build_generator(index, int(row["sample_id"])))
        noisy = torch.clamp(clean + (noise * self.noise_std), 0.0, 1.0)

        return {
            "input": noisy,
            "target": clean,
            "dataset_name": str(row["dataset_name"]),
            "sample_id": int(row["sample_id"]),
        }

    def _build_generator(self, index: int, sample_id: int) -> torch.Generator:
        sample_seed = self.base_seed + (index * 1009) + (sample_id * 17)
        return torch.Generator().manual_seed(sample_seed)


def _load_grayscale_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        return pil_to_tensor(grayscale).float() / 255.0
