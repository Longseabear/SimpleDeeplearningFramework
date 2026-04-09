from argparse import ArgumentParser
from pathlib import Path
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mnist import prepare_mnist_pngs
from src.data.preprocess import build_metadata


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Download MNIST, export PNG files, and build metadata.csv.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild PNG export and metadata from scratch.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = PROJECT_ROOT

    cfg = OmegaConf.load(project_root / "configs" / "config.yaml")
    cfg.data = OmegaConf.load(project_root / "configs" / "data" / "default.yaml")

    data_root = project_root / cfg.paths.data_root
    download_root = project_root / cfg.paths.download_root
    metadata_csv = project_root / cfg.paths.metadata_csv

    image_count = prepare_mnist_pngs(
        download_root=download_root,
        export_root=data_root,
        force_rebuild=args.force_rebuild or bool(cfg.data.force_rebuild),
    )
    metadata = build_metadata(
        data_root=data_root,
        output_csv=metadata_csv,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )

    split_counts = metadata["split"].value_counts().to_dict()
    print(
        {
            "image_count": image_count,
            "metadata_csv": str(metadata_csv),
            "split_counts": split_counts,
        }
    )


if __name__ == "__main__":
    main()
