from pathlib import Path

from torchvision.datasets import MNIST


def prepare_mnist_pngs(
    download_root: Path,
    export_root: Path,
    force_rebuild: bool = False,
) -> int:
    download_root = Path(download_root)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    existing_pngs = list(export_root.glob("mnist_png/*.png"))
    if existing_pngs and not force_rebuild:
        return len(existing_pngs)

    _clear_existing_pngs(export_root)

    train_dataset = MNIST(root=str(download_root), train=True, download=True)
    test_dataset = MNIST(root=str(download_root), train=False, download=True)
    combined = list(train_dataset) + list(test_dataset)
    image_root = export_root / "mnist_png"
    image_root.mkdir(parents=True, exist_ok=True)

    for sample_index, (image, _) in enumerate(combined):
        image.save(image_root / f"{sample_index:05d}.png")

    return len(combined)


def _clear_existing_pngs(export_root: Path) -> None:
    for png_path in export_root.glob("mnist_png/*.png"):
        png_path.unlink()
