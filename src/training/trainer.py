import csv
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def fit(
    cfg: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    output_dir: Path,
    tensorboard_root: Path,
    run_name: str,
    logger: logging.Logger,
) -> None:
    device = _resolve_device(cfg.train.device)
    model = model.to(device)
    logger.info("Resolved device: %s", device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    checkpoints_dir = output_dir / cfg.paths.checkpoints_dirname
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_root / run_name))

    metrics_path = output_dir / "metrics.csv"
    start_epoch, best_val_loss = _load_checkpoint_if_available(cfg, model, optimizer, device)
    if start_epoch > 1:
        logger.info("Resuming from checkpoint: %s", cfg.train.resume_from_checkpoint)
        logger.info("Resume start epoch: %d", start_epoch)

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            train_loss = _run_train_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg.train.epochs)
            val_loss, preview_batch = _run_validation_epoch(model, val_loader, criterion, device, epoch, cfg.train.epochs)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % cfg.train.log_image_every_n_epochs == 0 and preview_batch is not None:
                _log_preview(writer, preview_batch, epoch)

            csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            _save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, val_loss, best_val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, val_loss, best_val_loss)

            logger.info(
                f"[Epoch {epoch}/{cfg.train.epochs}] "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
            )

    writer.close()


def _run_train_epoch(model, data_loader, criterion, optimizer, device: torch.device, epoch: int, total_epochs: int) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    progress_bar = tqdm(data_loader, desc=f"Train {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_items, 1)


@torch.no_grad()
def _run_validation_epoch(model, data_loader, criterion, device: torch.device, epoch: int, total_epochs: int):
    model.eval()
    total_loss = 0.0
    total_items = 0
    preview_batch = None

    progress_bar = tqdm(data_loader, desc=f"Val   {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for batch in progress_bar:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        if preview_batch is None:
            preview_batch = {
                "input": inputs[:4].detach().cpu(),
                "prediction": predictions[:4].detach().cpu(),
                "target": targets[:4].detach().cpu(),
            }

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_items, 1), preview_batch


def _log_preview(writer: SummaryWriter, preview_batch: dict[str, torch.Tensor], epoch: int) -> None:
    image_stack = torch.cat(
        [preview_batch["input"], preview_batch["prediction"], preview_batch["target"]],
        dim=0,
    )
    writer.add_images("preview/input_prediction_target", image_stack, epoch)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def _load_checkpoint_if_available(
    cfg: DictConfig,
    model: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint_path = cfg.train.resume_from_checkpoint
    if checkpoint_path in (None, "", "null"):
        return 1, float("inf")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
    return start_epoch, best_val_loss


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
