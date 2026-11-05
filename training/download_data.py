"""Download the Kaggle dataset into data/raw.

Needs a Kaggle API token: put kaggle.json in %USERPROFILE%\\.kaggle\\ (Windows) or
~/.kaggle/ (Linux/macOS), or set KAGGLE_USERNAME / KAGGLE_KEY in .env.
Alternative: download the zip manually from the Kaggle page and extract it to data/raw.
"""
from dotenv import load_dotenv

from visionguard.config import DATA_RAW

DATASET = "cashbowman/ai-generated-images-vs-real-images"


def main():
    load_dotenv()
    from kaggle.api.kaggle_api_extended import KaggleApi  # import after env is loaded

    api = KaggleApi()
    api.authenticate()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET, path=str(DATA_RAW), unzip=True, quiet=False)
    print("Downloaded to", DATA_RAW)
    for p in sorted(DATA_RAW.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
