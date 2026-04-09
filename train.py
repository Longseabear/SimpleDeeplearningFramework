from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MnistDenoisingDataset
from src.data.mnist import prepare_mnist_pngs
from src.data.preprocess import build_metadata
from src.models.denoiser import SimpleDenoiser
from src.training.trainer import fit
from src.utils.logging_utils import configure_logging
from src.utils.paths import resolve_project_path
from src.utils.repro import seed_everything, seed_worker
from src.utils.runtime import collect_runtime_info, save_runtime_info


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    runtime_output_dir = Path(HydraConfig.get().runtime.output_dir)
    project_root = Path(hydra.utils.get_original_cwd())
    data_root = resolve_project_path(project_root, cfg.paths.data_root)
    metadata_csv = resolve_project_path(project_root, cfg.paths.metadata_csv)
    tensorboard_root = resolve_project_path(project_root, cfg.paths.tensorboard_root)
    download_root = resolve_project_path(project_root, cfg.paths.download_root)

    prepare_mnist_pngs(
        download_root=download_root,
        export_root=data_root,
        force_rebuild=cfg.data.force_rebuild,
    )
    metadata = build_metadata(
        data_root=data_root,
        output_csv=metadata_csv,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.seed,
    )

    train_loader, val_loader, _test_loader = build_dataloaders(cfg, metadata)
    model = build_model(cfg)

    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(runtime_output_dir / "train.log")
    OmegaConf.save(cfg, runtime_output_dir / "config.yaml")
    metadata.to_csv(runtime_output_dir / "metadata_snapshot.csv", index=False)
    save_runtime_info(runtime_output_dir / "runtime_env.txt", collect_runtime_info())
    logger.info("Training started")
    logger.info("Device setting: %s", cfg.train.device)
    logger.info(
        "Dataset split sizes | train=%d validation=%d test=%d",
        len(metadata.loc[metadata["split"] == "train"]),
        len(metadata.loc[metadata["split"] == "validation"]),
        len(metadata.loc[metadata["split"] == "test"]),
    )

    fit(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=runtime_output_dir,
        tensorboard_root=tensorboard_root,
        run_name=f"{cfg.run_name}_seed{cfg.seed}",
        logger=logger,
    )


def build_dataloaders(cfg: DictConfig, metadata):
    train_df = metadata.loc[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata.loc[metadata["split"] == "validation"].reset_index(drop=True)
    test_df = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)

    train_dataset = MnistDenoisingDataset(train_df, noise_std=cfg.data.noise_std, base_seed=cfg.seed)
    val_dataset = MnistDenoisingDataset(val_df, noise_std=cfg.data.noise_std, base_seed=cfg.seed)
    test_dataset = MnistDenoisingDataset(test_df, noise_std=cfg.data.noise_std, base_seed=cfg.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader


def build_model(cfg: DictConfig) -> SimpleDenoiser:
    return SimpleDenoiser(
        in_channels=cfg.model.in_channels,
        image_size=cfg.data.image_size,
        conv_channels=list(cfg.model.conv_channels),
        hidden_dim=cfg.model.hidden_dim,
        bottleneck_dim=cfg.model.bottleneck_dim,
    )


if __name__ == "__main__":
    main()
