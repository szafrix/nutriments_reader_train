import requests
from tqdm import tqdm


def download_database():
    url = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
    local_filename = "data/raw/openfoodfacts-products.jsonl.gz"

    # Send a HEAD request to get the content length
    response = requests.head(url)
    total_size = int(response.headers.get("content-length", 0))

    chunk_size = 10 * 1024 * 1024  # 10 MB

    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(local_filename, "wb") as handle:
            progress_bar = tqdm(
                total=total_size, unit="iB", unit_scale=True, desc="Downloading"
            )
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = handle.write(chunk)
                progress_bar.update(size)
            progress_bar.close()

    print(f"Download completed. File saved as {local_filename}")


if __name__ == "__main__":
    download_database()
