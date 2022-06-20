import os
import torch
import argparse
import numpy as np
import wandb
import torch.distributed as dist
import OpenEXR
import Imath

# Some generally useful utilities


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


def exr_to_tensor(img_path):
    exr_image = OpenEXR.InputFile(img_path)
    # dataWindow contains the width and height information stored
    # in the image header, it is returned as an Imath.Box2i
    dw = exr_image.header()["dataWindow"]
    w, h = (dw.max.x + 1, dw.max.y + 1)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    # the string 'RGB' specifies the order we want colour channels returned in
    colour_channels = exr_image.channels("RGB", pixel_type)
    img = [
        np.frombuffer(channel, dtype=np.float32).reshape(h, w)
        for channel in colour_channels
    ]
    img_tensor = torch.from_numpy(np.array(img))
    return img_tensor


def save_exr_image(img, fname):
    """Save HDR image as .exr file.
    Args:
        img ([numpy.array]): [Image in shape (H, W, C)]
        fname ([string]): [Filepath and name, must end in .exr]
    Raises:
        IOError: [description]
    """
    try:
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        full_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header["channels"] = dict([(c, full_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(fname, header)
        R = (img[:, :, 0]).astype(np.float32).tobytes()
        G = (img[:, :, 1]).astype(np.float32).tobytes()
        B = (img[:, :, 2]).astype(np.float32).tobytes()
        out.writePixels({"R": R, "G": G, "B": B})
        out.close()
    except Exception as e:
        raise IOError("Failed writing EXR: %s" % e)


def sRGB(img):
    img = img / torch.quantile(img, 0.98)
    img = torch.clamp(img, 0.0, 1.0)
    img = torch.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * torch.pow(torch.abs(img), 1 / 2.4) - 0.055,
    )
    return img


# Generates the unit vector associated with the direction of each pixel in the panoramic image
def get_directions(sidelen):
    """Generates a flattened grid of (x,y,z,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    # indexing='ij' as future meshgrid behaviour will default to indexing='xy'
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # shape=[sidelen/2,sidelen,2]
    uv = uv.reshape(-1, 2)  # shape=[sidelen/2*sidelen,2]
    # From: https://vgl.ict.usc.edu/Data/HighResProbes/
    theta = np.pi * (uv[:, 0] - 1)
    phi = np.pi * uv[:, 1]
    directions = torch.stack(
        (
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
            -torch.sin(phi) * torch.cos(theta),
        ),
        -1,
    )  # shape=[sidelen/2*sidelen,3]
    return directions


# sine of the polar angle for compensation of irregular equirectangular sampling
def get_sineweight(sidelen):
    """Returns a matrix of sampling densites"""
    u = (torch.linspace(1, sidelen, steps=sidelen) - 0.5) / (sidelen // 2)
    v = (torch.linspace(1, sidelen // 2, steps=sidelen // 2) - 0.5) / (sidelen // 2)
    # indexing='ij' as future meshgrid behaviour will default to indexing='xy'
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((u_grid, v_grid), -1)  # shape=[sidelen/2,sidelen,2]
    uv = uv.reshape(-1, 2)  # shape=[sidelen/2*sidelen,2]
    # From: https://vgl.ict.usc.edu/Data/HighResProbes/
    phi = np.pi * uv[:, 1]
    sineweight = torch.sin(phi)  # shape=[sidelen/2*sidelen]
    # change from shape=[sidelen/2*sidelen] to shape=[sidelen/2*sidelen, 3]
    # to match output of RENI and with with weighting_matrix repeated
    # for each colour channel
    sineweight = sineweight.unsqueeze(1).repeat(1, 3)
    return sineweight


def parse_args():
    """
    Parse arguments given to the script.
    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb."
    )
    # Used for `distribution.launch`
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--log_all",
        action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch",
        default=1,
        type=int,
        metavar="N",
        help="number of data samples in one batch",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project",
    )
    parser.add_argument("-w", "--world-size", default=1, type=int, dest="world_size")
    parser.add_argument("--sweep", default=None, dest="sweep")

    args = parser.parse_args()
    return args


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = dotdict(value)
            self[key] = value


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def mp_setup(rank, world_size, config, is_sweep):
    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"{rank=} init complete")

    run = None
    if is_sweep:
        if rank == 0:
            run = wandb.init()
        else:
            run = wandb.init(mode="disabled")
        config = wandb.config
    else:
        if rank == 0:
            run = wandb.init(config=config)

    return run, config


def mp_cleanup(rank):
    dist.destroy_process_group()
    print(f"{rank=} destroy complete")
