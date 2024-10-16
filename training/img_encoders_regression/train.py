import lightning as L
from training.img_encoders_regression.dataset import FoodDataset
from lightning.pytorch.loggers import WandbLogger
from training.img_encoders_regression.architectures.l_wrapper import L_ImageEncoder
from training.img_encoders_regression.architectures.resnet50.architecture import ResNet50Model
from dotenv import dotenv_values
import os
from torch.utils.data import DataLoader
import wandb
from config import DATALOADER_CONFIG, TRAIN_CONFIG, MODEL

L_MODEL = L_ImageEncoder(MODEL)

config = dotenv_values(".env")
os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]

wandb.init(project="NutrimentsReader")
wandb_logger = WandbLogger(project="NutrimentsReader")

with open("training/img_encoders/config.py", "r") as config_file:
    config_content = config_file.read()
    wandb.log({"config_file_content": wandb.Html(f"<pre>{config_content}</pre>")})

ds_train = FoodDataset(split="train", transform=MODEL.transforms_for_model())
ds_val = FoodDataset(split="val", transform=MODEL.transforms_for_model())
dl_train = DataLoader(ds_train, **DATALOADER_CONFIG, shuffle=True)
dl_val = DataLoader(ds_val, **DATALOADER_CONFIG, shuffle=False)


trainer = L.Trainer(
    **TRAIN_CONFIG,
    logger=wandb_logger,
)
trainer.fit(L_MODEL, dl_train, dl_val)
wandb.finish()
