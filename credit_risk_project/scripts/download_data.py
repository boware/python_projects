import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_data():
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define dataset and path
    dataset_name = 'home-credit-default-risk'  # Home Credit Default Risk Dataset
    download_path = 'data/'  # Folder where data will be saved

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded from Kaggle to {download_path}")

if __name__ == "__main__":
    download_kaggle_data()
