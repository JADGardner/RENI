import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import models.spherical_harmonics as spherical_harmonics
import utils.utils as utils
from utils.utils import get_directions, get_sineweight
import os
import json


class HDRIDataset(Dataset):
    def __init__(self, dataset_path, transform, minmax=None):
        """
        Args:
          dataset_path (string): Directory of dataset
          transform (callable): Transform to be applied
                                          on a sample
          minmax (tuple, optional): (min, max) values of the dataset
        """
        super(Dataset, self).__init__()
        self.dataset_path = dataset_path
        if os.path.exists(self.dataset_path + os.sep + "meta_data.json"):
            self.df = pd.read_json(os.path.join(self.dataset_path, "meta_data.json"))
        else:
            self.create_metadata()
            self.df = pd.read_json(os.path.join(self.dataset_path, "meta_data.json"))

        self.transform = transform

        # Calculate dataset min and max in log domain for scaling later
        if minmax is None:
            self.min_value = float("inf")
            self.max_value = float("-inf")
            for idx in range(0, len(self.df.columns)):
                img, _ = self.get_image(idx)
                img = torch.clip(img, img[img > 0.0].min(), img[img < np.inf].max())
                img = torch.log(img)
                if img.min() < self.min_value:
                    self.min_value = img.min()
                if img.max() > self.max_value:
                    self.max_value = img.max()
        else:
            self.min_value = minmax[0]
            self.max_value = minmax[1]

        self.directions = get_directions(self.transform.size[1])
        self.sineweight = get_sineweight(self.transform.size[1])

    def __len__(self):
        return len(self.df.columns)

    def __getitem__(self, idx):
        img, img_path = self.get_image(idx)

        img = self.apply_normalisation(img)

        img = torch.nan_to_num(img)
        pixels = img.permute((1, 2, 0))  # (3, H, W) -> (H, W, 3)
        pixels = pixels.view(-1, 3)  # (H, W, 3) -> (H*W, 3)
        return self.directions, self.sineweight, pixels, idx, img_path

    def get_image(self, idx, apply_transform=True):
        img_path = os.path.join(
            self.dataset_path, self.df[self.df.columns[idx]]["local_path"]
        )

        img = utils.exr_to_tensor(img_path)

        if apply_transform:
            img = self.transform(img)

        return img, img_path

    def apply_normalisation(self, img):
        img = torch.clip(img, img[img > 0.0].min(), img[img < np.inf].max())
        img = torch.log(img)
        # use min max normalisation to transform between -1 and 1 relative to the
        # dataset overall min and max
        img = 2 * (img - self.min_value) / (self.max_value - self.min_value) - 1
        return img

    def undo_normalisation(self, img):
        # undo the nomalisation applied to the dataset
        img = ((img + 1) * (self.max_value - self.min_value)) / 2 + self.min_value
        # model trained to match ln(image intesity) so need to undo this using exp
        img = torch.exp(img)
        return img

    # return image but converted to numpy array and with gamma adjusted to sRGB range
    def to_numpy_convert_sRGB(self, image_to_convert):
        img_hdr = self.undo_normalisation(image_to_convert)
        img = img_hdr.view(*self.transform.size, 3)  # (H*W, 3) -> (H, W, 3)
        # apply gamma correction
        img = utils.sRGB(img).cpu().detach().numpy()
        return img

    def get_spherical_harmonic_representation(self, idx, nBands):
        _, _, ground_truth, _, _ = self[idx]
        ground_truth = self.undo_normalisation(ground_truth)
        ground_truth = ground_truth.view(*self.transform.size, 3)
        ground_truth = ground_truth.permute((2, 0, 1))
        ground_truth = ground_truth.cpu().detach().numpy()
        ground_truth = ground_truth.transpose((1, 2, 0))
        iblCoeffs = spherical_harmonics.getCoefficientsFromImage(ground_truth, nBands)
        sh_radiance_map = spherical_harmonics.shReconstructSignal(
            iblCoeffs, width=ground_truth.shape[1]
        )
        sh_radiance_map = torch.from_numpy(sh_radiance_map)
        sh_radiance_map = sh_radiance_map.view(-1, 3)  # (H*W, 3)
        return sh_radiance_map

    def double_resolution(self):
        new_transform = transforms.Resize(
            (self.transform.size[0] * 2, self.transform.size[1] * 2)
        )
        del self.transform
        self.transform = new_transform
        del self.directions
        del self.sineweight
        self.directions = get_directions(self.transform.size[1])
        self.sineweight = get_sineweight(self.transform.size[1])

    def create_metadata(self):
        meta_data = {}
        for subdir, dirs, files in os.walk(self.dataset_path):
            dirs.sort()
            files.sort()
            for file in files:
                filepath = subdir + file

                if filepath.endswith(".exr"):
                    file_name = file[: len(file) - 4]
                    item = {file_name: {"local_path": filepath}}
                    meta_data.update(item)

        with open(self.dataset_path + os.sep + "meta_data.json", "w") as f:
            json.dump(meta_data, f)
