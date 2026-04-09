from pathlib import Path

import pandas as pd


def build_metadata(
    data_root: Path,
    output_csv: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    data_root = Path(data_root)
    output_csv = Path(output_csv)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0.")

    image_paths = sorted(data_root.glob("mnist_png/*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found under: {data_root}")

    rows: list[dict[str, str | int]] = []
    for image_path in image_paths:
        sample_id = int(image_path.stem)
        rows.append(
            {
                "input_path": str(image_path),
                "target_path": str(image_path),
                "dataset_name": "mnist",
                "sample_id": sample_id,
            }
        )

    dataframe = pd.DataFrame(rows).sort_values(["sample_id"]).reset_index(drop=True)

    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total_count = len(shuffled)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    shuffled["split"] = "test"
    shuffled.loc[: train_end - 1, "split"] = "train"
    shuffled.loc[train_end : val_end - 1, "split"] = "validation"
    dataframe = shuffled.sort_values(["split", "sample_id"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    return dataframe
