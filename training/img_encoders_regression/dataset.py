import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import random
from IPython.display import display


def get_all_imagenames(split: str):
    return os.listdir(f"data/splits/{split}")


def get_ids_for_split(split: str) -> list[int]:
    ids = os.listdir(f"data/splits/{split}")
    ids = [x.split("_")[0] for x in ids]
    ids = list(set(ids))
    return ids


class FoodDataset(Dataset):
    def __init__(self, split: str, transform: transforms.Compose = None):
        self.split = split
        self.all_images = get_all_imagenames(split)
        self.ids = get_ids_for_split(split)
        self.metadata = json.load(
            open("data/products_nutrients_and_image_links.json")
        )
        self.metadata = {x["id_"]: x for x in self.metadata}
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_id = str(image_name.split("_")[0])
        image_path = f"data/splits/{self.split}/{image_name}"
        image = Image.open(image_path).convert("RGB")
        metadata = self.metadata[image_id]
        target_vector = self.prepare_target_vector(metadata)
        if self.transform:
            image_array = self.transform(image)

        return image_array, target_vector

    def display_sample(self, idx):
        image_array, metadata = self[idx]
        image_array = image_array.numpy().transpose(1, 2, 0)
        image_array = (image_array * 255).astype(np.uint8)
        print(metadata)
        display(Image.fromarray(image_array))

    @staticmethod
    def prepare_target_completion(metadata: dict):
        return f"""kcal per 100g: {metadata['kcal_100g']}\nproteins per 100g: {metadata['proteins_100g']}\ncarbs per 100g: {metadata['carbs_100g']}\nfats per 100g: {metadata['fats_100g']}"""

    @staticmethod
    def prepare_target_vector(metadata: dict):
        return (
            torch.tensor(
                [
                    metadata["kcal_100g"],
                    metadata["proteins_100g"],
                    metadata["carbs_100g"],
                    metadata["fats_100g"],
                ]
            )
            # .unsqueeze(0)
            .float()
        )
