import gzip
import json
import re
import logging
from typing import Any, Dict, List
from tqdm import tqdm
import json

# define logger to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


with open("data/valid_train_files.txt", "r") as f:
    valid_train_files = f.readlines()
valid_train_files = [s.strip() for s in valid_train_files]

with open("data/valid_val_files.txt", "r") as f:
    valid_val_files = f.readlines()
valid_val_files = [s.strip() for s in valid_val_files]


def extract_nutrients_per_100g(product_data: Dict[str, Any]) -> Dict[str, Any] | None:
    id_ = product_data.get("_id", None)
    if not id_:
        logger.debug("No id found for product")
        return None
    if nutriments := product_data.get("nutriments", None):
        kcal_100g = nutriments.get("energy-kcal_100g", None)
        proteins_100g = nutriments.get("proteins_100g", None)
        carbs_100g = nutriments.get("carbohydrates_100g", None)
        fats_100g = nutriments.get("fat_100g", None)
    else:
        logger.debug(f"No nutriments found for product {id_}")
        return None
    for value, name in [
        (kcal_100g, "kcal_100g"),
        (proteins_100g, "proteins_100g"),
        (carbs_100g, "carbs_100g"),
        (fats_100g, "fats_100g"),
    ]:
        if value is None:
            logger.debug(f"No value found for product {id_} and nutrient {name}")
            return None
    return {
        "id_": id_,
        "kcal_100g": kcal_100g,
        "proteins_100g": proteins_100g,
        "carbs_100g": carbs_100g,
        "fats_100g": fats_100g,
    }


def get_image_url(product_data, image_name, resolution="400"):
    try:
        if image_name not in product_data["images"]:
            return None
        base_url = "https://images.openfoodfacts.org/images/products"
        folder_name = product_data["code"]
        if folder_name in valid_train_files or folder_name in valid_val_files:
            if len(folder_name) > 8:
                folder_name = re.sub(r"(...)(...)(...)(.*)", r"\1/\2/\3/\4", folder_name)
            if re.match(r"^\d+$", image_name):
                resolution_suffix = "" if resolution == "full" else f".{resolution}"
                filename = f"{image_name}{resolution_suffix}.jpg"
            else:
                rev = product_data["images"][image_name]["rev"]
                filename = f"{image_name}.{rev}.{resolution}.jpg"
            return f"{base_url}/{folder_name}/{filename}"
    except Exception as e:
        logger.error(
            f"Error getting image URL for product {product_data.get('_id', 'unknown')}: {e}"
        )
        return None
    


def get_all_nutrients_images(product_data: Dict[str, Any]) -> List[str] | None:
    if not product_data.get("images", None):
        logger.debug(
            f"No images found for product {product_data.get('_id', 'unknown')}"
        )
        return None

    nutrients_keys = [
        key for key in product_data["images"] if key.startswith("nutrition")
    ]
    if not nutrients_keys:
        logger.debug(
            f"No nutrients images found for product {product_data.get('_id', 'unknown')}"
        )
        return None
    return [get_image_url(product_data, key) for key in nutrients_keys]


def read_partial_jsonl_gz(filename):
    # Open the gzipped JSONL file
    data = []
    with gzip.open(filename, "rt", encoding="utf-8") as file:
        for line in tqdm(file):
            product = json.loads(line)
            nutrients = extract_nutrients_per_100g(product)
            nutrients_images = get_all_nutrients_images(product)
            if nutrients and nutrients_images:
                data.append({**nutrients, "images": nutrients_images})
    return data


filepath = "data/raw/openfoodfacts-products.jsonl.gz"
data = read_partial_jsonl_gz(filepath)

json.dump(
    data, open("data/metadata/products_nutrients_and_image_links.json", "w")
)
