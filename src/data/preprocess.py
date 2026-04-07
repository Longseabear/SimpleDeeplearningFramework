from pathlib import Path
import random

import pandas as pd


def build_metadata(
    data_root: Path,
    output_csv: Path,
    samples_per_dataset: int,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    data_root = Path(data_root)
    output_csv = Path(output_csv)
    dataset_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())

    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders found under: {data_root}")

    rows: list[dict] = []
    for dataset_index, dataset_dir in enumerate(dataset_dirs):
        indices = list(range(samples_per_dataset))
        random.Random(seed + dataset_index).shuffle(indices)
        val_count = max(1, int(samples_per_dataset * val_ratio))
        val_indices = set(indices[:val_count])

        for sample_id in range(samples_per_dataset):
            rows.append(
                {
                    "dataset_name": dataset_dir.name,
                    "dataset_path": str(dataset_dir),
                    "sample_id": sample_id,
                    "split": "val" if sample_id in val_indices else "train",
                }
            )

    dataframe = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    return dataframe
