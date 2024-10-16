import asyncio
import aiohttp
import aiofiles
import random
from typing import Any, Dict, List
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

"""Run in screen to avoid interruptions"""


def _select_split(
    train_split: float = 0.99, val_split: float = 0.005, test_split: float = 0.005
) -> str:
    return random.choices(
        ["train", "val", "test"], weights=[train_split, val_split, test_split]
    )[0]


async def download_image_async(
    session: aiohttp.ClientSession, url: str, output_path: str
) -> None:
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            async with aiofiles.open(output_path, "wb") as f:
                await f.write(await response.read())
    except Exception as e:
        print(f"Error downloading image {url}: {e}")


async def download_images_for_product_async(
    session: aiohttp.ClientSession, product_data: Dict[str, Any]
) -> None:
    split = _select_split()
    tasks = []
    for i, image_url in enumerate(product_data["images"]):
        output_path = (
            f"training/img2txt/data/splits/{split}/{product_data['id_']}_{i}.jpg"
        )
        tasks.append(download_image_async(session, image_url, output_path))
    await asyncio.gather(*tasks)


async def download_images_for_products_async(
    products_data: List[Dict[str, Any]], pbar: tqdm
) -> None:
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_images_for_product_async(session, product_data)
            for product_data in products_data
        ]
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)


def run_async_download(products_data: List[Dict[str, Any]], chunk_pbar: tqdm) -> None:
    with tqdm(total=len(products_data), leave=False) as pbar:
        asyncio.run(download_images_for_products_async(products_data, pbar))
    chunk_pbar.update(1)


if __name__ == "__main__":

    products_data = json.load(
        open("training/img2txt/data/products_nutrients_and_image_links.json")
    )
    chunked = np.array_split(products_data, 1000)
    with tqdm(total=len(chunked), desc="Chunks") as chunk_pbar:
        for chunk in chunked:
            run_async_download(chunk, chunk_pbar)
