# Download Datasets:
from src.data.datasets import RENIDatasetHDR, RENIDatasetLDR
from src.utils.utils import download_pretrained_models

# download datasets
print("Downloading datasets...")
RENIDatasetHDR('data/RENI_HDR', download=True)
RENIDatasetLDR('data/RENI_LDR', download=True)

# Download Pretrained Models:
folder_id = '1DkaBxMzCTt5dQyRx14BG9tzCTudY6kKy'
download_pretrained_models(folder_id, 'models')