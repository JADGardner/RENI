import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os
import gdown
import string
from typing import Optional
import natsort
import imageio
from src.utils.custom_transforms import MinMaxNormalise, UnMinMaxNormlise, UnNormalise
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import zipfile
import tqdm

class RENIDatasetHDR(Dataset):
    def __init__(self, dataset_path: string, transforms: Optional[transforms.Compose] = None, download: Optional[bool] = False):
        """
        Args:
          dataset_path (string): Directory of dataset
          transforms (callable, optional): Optional transforms to be applied on a sample
          minmax (tuple, optional): Tuple of min and max values to use for minmax normalisation
          download (bool, optional): Whether to download the dataset if it is not present
        """
        super(Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms

        if download:
          if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            id = "1NRTL-WHEKttLbvJjDaFeK7jMO1uUV8Cn"
            output = dataset_path + os.sep + "RENI_HDR.zip"
            gdown.download(id=id, output=output, quiet=False)
            with zipfile.ZipFile(output, 'r') as zip_ref:
              zip_ref.extractall(dataset_path)
          else:
            print(f'Dataset found at: {self.dataset_path}\n')
            return
        else:
          # use listdir to get all .exr files in the dataset
          files = os.listdir(self.dataset_path)
          files = [f for f in files if f.endswith(".exr")]
          self.img_names = natsort.natsorted(files)

          self.unnormalise = None
          # Calculate dataset min and max in log domain for scaling later
          # check if MinMaxNormalise is in transforms and if so that minmax value is not None
          if self.transforms is not None:
            for t in self.transforms.transforms:
              if isinstance(t, MinMaxNormalise):
                if len(t.minmax) == 0:
                  # calculate min and max
                  print("Calculating min and max values for dataset")
                  minmax = self.calculate_minmax()
                  print(f"Min: {minmax[0]}, Max: {minmax[1]}")
                  t.minmax = minmax
                else:
                  minmax = t.minmax
                self.unnormalise = UnMinMaxNormlise(minmax)
          
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = self.get_image(idx)

        img = self.transforms(img)

        img = torch.nan_to_num(img)

        return img, idx

    def get_image(self, idx):
        img_path = os.path.join(
            self.dataset_path, self.img_names[idx]
        )
        img = imageio.imread(img_path)

        return ToTensor()(img)
    
    def double_resolution(self):
        if self.transforms is not None:
            for t in self.transforms.transforms:
                if isinstance(t, transforms.Resize):
                    t.size = (t.size[0]*2, t.size[1]*2)

    def calculate_minmax(self):
        min_value = float("inf")
        max_value = float("-inf")
        for idx in tqdm.tqdm(range(len(self))):
            img = self.get_image(idx)
            img = torch.clip(img, img[img > 0.0].min(), img[img < torch.inf].max())
            img = torch.log(img)
            if img.min() < min_value:
                min_value = img.min()
            if img.max() > max_value:
                max_value = img.max()
        return [min_value, max_value]
        

class RENIDatasetLDR(Dataset):
    def __init__(self, dataset_path: string, transforms: Optional[transforms.Compose] = None, download: Optional[bool] = False):
        """
        Args:
            dataset_path (string): Path to the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool, optional): Download the dataset if it is not present.
        """
        super(Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.transforms = transforms

        if download:
          if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            id = "1vdOLFYaSXmHEr79F78fCBufSqVSV6laj"
            output = dataset_path + os.sep + "RENI_LDR.zip"
            gdown.download(id=id, output=output, quiet=False)
            with zipfile.ZipFile(output, 'r') as zip_ref:
              zip_ref.extractall(dataset_path)
          else:
            print(f'Dataset found at: {self.dataset_path}\n')
            return
        
        self.unnormalise = None
        if self.transforms is not None:
            for t in self.transforms.transforms:
              if isinstance(t,  Normalize):
                self.unnormalise = UnNormalise(t.mean, t.std)

        all_imgs = os.listdir(self.dataset_path)
        self.img_names = natsort.natsorted(all_imgs)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        # ensure no alpha channel
        img = img[:3, :, :]
        img = self.transforms(img)
        return img, idx

    def double_resolution(self):
        if self.transforms is not None:
            for t in self.transforms.transforms:
                if isinstance(t, transforms.Resize):
                    t.size = (t.size[0]*2, t.size[1]*2)
    
    def get_image(self, idx: int) -> torch.FloatTensor:
        img_path = os.path.join(
            self.dataset_path, self.img_names[idx]
        )
        return ToTensor()(Image.open(img_path))


def download_data(config):
  dataset_name = config.DATASET.NAME
  if dataset_name == 'RENI_HDR':
    RENIDatasetHDR(config.DATASET[dataset_name].PATH, download=True)
  elif dataset_name == 'RENI_LDR':
    RENIDatasetLDR(config.DATASET[dataset_name].PATH, download=True)


def get_dataset(dataset_name, dataset_path, transform, is_hdr):
  if dataset_name == 'RENI_HDR' or (dataset_name == 'CUSTOM' and is_hdr):
    return RENIDatasetHDR(dataset_path, transform, False)
  elif dataset_name == 'RENI_LDR' or (dataset_name == 'CUSTOM' and not is_hdr):
    return RENIDatasetLDR(dataset_path, transform, False)

