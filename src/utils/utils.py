import os
import torch
import numpy as np
import gdown
import zipfile
from torchvision import transforms
from PIL import Image

def SelectDevice():
    # Function to automatically select device (CPU or GPU with most free memory)
    # Returns a torch device
    # OS call to nvidia-smi can probably be replaced with nvidia python library
    if torch.cuda.is_available():
        os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp_free_gpus")
        with open("tmp_free_gpus", "r") as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [
                (idx, int(x.split()[2])) for idx, x in enumerate(frees)
            ]
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        idx = idx_freeMemory_pair[0][0]
        device = torch.device("cuda:" + str(idx))
        print("Using GPU idx: " + str(idx))
        os.remove("tmp_free_gpus")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def sRGB(imgs):
    # if shape is not B, C, H, W, then add batch dimension
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    q = torch.quantile(torch.quantile(torch.quantile(imgs, 0.98, dim=(1)), 0.98, dim=(1)), 0.98, dim=(1))
    imgs = imgs / q.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    imgs = torch.clamp(imgs, 0.0, 1.0)
    imgs = torch.where(
        imgs <= 0.0031308,
        12.92 * imgs,
        1.055 * torch.pow(torch.abs(imgs), 1 / 2.4) - 0.055,
    )
    return imgs


# Generates the unit vector associated with the direction of each pixel in the panoramic image
def get_directions(sidelen):
    """Generates a flattened grid of (x,y,z,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2,sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen,2]
    theta = np.pi * (uv[:, 0] - 1)
    phi = np.pi * uv[:, 1]
    directions = torch.stack(
        (
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
            -torch.sin(phi) * torch.cos(theta),
        ),
        -1,
    ).unsqueeze(0)  # shape=[1, sidelen/2*sidelen, 3]
    return directions

# sine of the polar angle for compensation of irregular equirectangular sampling
def get_sineweight(sidelen):
    """Returns a matrix of sampling densites"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # [sidelen/2, sidelen, 2]
    uv = uv.reshape(-1, 2)  # [sidelen/2*sidelen, 2]
    phi = np.pi * uv[:, 1]
    sineweight = torch.sin(phi) # [sidelen/2*sidelen]
    sineweight = sineweight.unsqueeze(1).repeat(1, 3).unsqueeze(0) # shape=[1, sidelen/2*sidelen, 3]
    return sineweight


def get_mask(sidelen, path):         
    mask = Image.open(path)
    mask = transforms.ToTensor()(mask)
    # if mask channel number is 1 then repeat it 3 times
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    mask_transform = transforms.Resize((sidelen//2, sidelen), interpolation=transforms.InterpolationMode.NEAREST)
    mask = mask_transform(mask)
    mask = mask.permute((1, 2, 0))  # (3, H, W) -> (H, W, 3)
    mask = mask.view(-1, 3).unsqueeze(0)  # (H, W, 3) -> (1, H*W, 3)
    return mask

def download_pretrained_models(gdrive_id, output_path):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  output = output_path + os.sep + "pre_trained.zip"
  if not os.path.exists(output):
    print("Downloading pretrained models...")
    gdown.download(id=gdrive_id, output=output, quiet=False)
  else:
    print("Pretrained models already downloaded")
  with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(output_path)
