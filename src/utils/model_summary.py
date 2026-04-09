import logging

import torch
from torch import nn
from torchinfo import summary


def log_model_summary(
    logger: logging.Logger,
    model: nn.Module,
    image_size: int,
    device: torch.device,
) -> None:
    summary_text = str(
        summary(
            model,
            input_size=(1, 1, image_size, image_size),
            device=str(device),
            verbose=0,
            col_names=("input_size", "output_size", "num_params", "trainable"),
        )
    )
    logger.info("Model summary\n%s", summary_text)
