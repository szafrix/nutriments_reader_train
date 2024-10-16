import torch.nn as nn
import torch
from training.img_encoders.architectures.efficientnet_v2_s.architecture import (
    EfficientNetV2SModel,
)

DATA_CONFIG = {
    "splits_path": "data/splits",
    "metadata_path": "data/products_nutrients_and_image_links.json",
}

DATALOADER_CONFIG = {
    "batch_size": 16,
    "num_workers": 31,
}

TRAIN_CONFIG = {
    "num_sanity_val_steps": 2,
    "max_epochs": 10,
    "enable_checkpointing": False,
    "val_check_interval": 0.1,
    "log_every_n_steps": 25,
    "profiler": "simple",
}

LOSS = nn.MSELoss()

OPTIMIZER = {"optimizer": torch.optim.Adam, "optimizer_params": {"lr": 1e-2}}

MODEL = EfficientNetV2SModel()
