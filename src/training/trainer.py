import csv
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def fit(
    cfg: DictConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    output_dir: Path,
    tensorboard_root: Path,
    run_name: str,
) -> None:
    device = _resolve_device(cfg.train.device)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    checkpoints_dir = output_dir / cfg.paths.checkpoints_dirname
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_root / run_name))

    metrics_path = output_dir / "metrics.csv"
    best_val_loss = float("inf")

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = _run_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
            )
            val_loss, preview_batch = _run_validation(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if epoch % cfg.train.log_image_every_n_epochs == 0 and preview_batch is not None:
                _log_preview(writer, preview_batch, epoch)

            csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            _save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, val_loss)

    writer.close()


def _run_epoch(model, data_loader, criterion, optimizer, device, training: bool) -> float:
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for batch in data_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def _run_validation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    preview_batch = None

    for batch in data_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        if preview_batch is None:
            preview_batch = {
                "input": inputs[:1].detach().cpu(),
                "prediction": predictions[:1].detach().cpu(),
                "target": targets[:1].detach().cpu(),
            }

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1), preview_batch


def _log_preview(writer: SummaryWriter, preview_batch: dict, epoch: int) -> None:
    image_stack = torch.cat(
        [preview_batch["input"], preview_batch["prediction"], preview_batch["target"]],
        dim=0,
    )
    writer.add_images("preview/input_prediction_target", image_stack, epoch)


def _save_checkpoint(path: Path, model: nn.Module, optimizer: Adam, epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
