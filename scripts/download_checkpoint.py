import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from src.dataset import load_config


def download_checkpoint(url=None):
    config = load_config()
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_path):
        print(f"Checkpoint already exists at {checkpoint_path}")
        return

    if url is None:
        url = os.environ.get("CHECKPOINT_URL")

    if url is None:
        print("ERROR: No checkpoint URL provided.")
        print("Set CHECKPOINT_URL environment variable or pass URL as argument.")
        sys.exit(1)

    print(f"Downloading checkpoint from {url}")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"ERROR: Failed to download checkpoint. Status code: {response.status_code}")
        sys.exit(1)

    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    download_checkpoint(url)