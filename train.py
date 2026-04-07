from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SyntheticDenoisingDataset
from src.data.preprocess import build_metadata
from src.models.denoiser import SimpleDenoiser
from src.training.trainer import fit
from src.utils.paths import resolve_project_path
from src.utils.repro import seed_everything, seed_worker


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    runtime_output_dir = Path(HydraConfig.get().runtime.output_dir)
    project_root = Path(hydra.utils.get_original_cwd())
    data_root = resolve_project_path(project_root, cfg.paths.data_root)
    metadata_csv = resolve_project_path(project_root, cfg.paths.metadata_csv)
    tensorboard_root = resolve_project_path(project_root, cfg.paths.tensorboard_root)

    metadata = build_metadata(
        data_root=data_root,
        output_csv=metadata_csv,
        samples_per_dataset=cfg.data.samples_per_dataset,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.seed,
    )

    train_df = metadata.loc[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata.loc[metadata["split"] == "val"].reset_index(drop=True)

    train_dataset = SyntheticDenoisingDataset(
        dataframe=train_df,
        image_size=cfg.data.image_size,
        base_seed=cfg.seed,
    )
    val_dataset = SyntheticDenoisingDataset(
        dataframe=val_df,
        image_size=cfg.data.image_size,
        base_seed=cfg.seed,
    )

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

    model = SimpleDenoiser(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        hidden_channels=cfg.model.hidden_channels,
        num_blocks=cfg.model.num_blocks,
        kernel_size=cfg.model.kernel_size,
    )

    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, runtime_output_dir / "config.yaml")
    metadata.to_csv(runtime_output_dir / "metadata_snapshot.csv", index=False)

    fit(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=runtime_output_dir,
        tensorboard_root=tensorboard_root,
        run_name=f"{cfg.run_name}_seed{cfg.seed}",
    )


if __name__ == "__main__":
    main()
