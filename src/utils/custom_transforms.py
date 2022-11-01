import torch
import torchvision.transforms as transforms

class MinMaxNormalise(object):
    def __init__(self, minmax):
        self.minmax = minmax

    def __call__(self, img):
      img = torch.clip(img, img[img > 0.0].min(), img[img < torch.inf].max())
      img = torch.log(img)
      img = 2 * (img - self.minmax[0]) / (self.minmax[1] - self.minmax[0]) - 1
      return img

class UnMinMaxNormlise(object):
    def __init__(self, minmax):
        self.minmax = minmax

    def __call__(self, img):
      img = 0.5 * (img + 1) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
      img = torch.exp(img)
      return img

class UnNormalise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # normalise each channel in the batch of images
        tensor = tensor.permute(1, 0, 2, 3)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor.permute(1, 0, 2, 3)

def get_transform(transform_name, args):
    if transform_name == "resize":
        return transforms.Resize((args[0], args[1]))
    elif transform_name == "centercrop":
        return transforms.CenterCrop(args)
    elif transform_name == "randomcrop":
        return transforms.RandomCrop(args)
    elif transform_name == "randomhorizontalflip":
        return transforms.RandomHorizontalFlip()
    elif transform_name == "randomverticalflip":
        return transforms.RandomVerticalFlip()
    elif transform_name == "randomrotation":
        return transforms.RandomRotation(args)
    elif transform_name == "colorjitter":
        return transforms.ColorJitter(
            brightness=args[0],
            contrast=args[1],
            saturation=args[2],
            hue=args[3],
        )
    elif transform_name == "grayscale":
        return transforms.Grayscale(num_output_channels=1)
    elif transform_name == "to_tensor":
        return transforms.ToTensor()
    elif transform_name == "normalize":
        return transforms.Normalize(
            mean=args[0],
            std=args[1],
        )
    elif transform_name == "minmaxnormalise":
        return MinMaxNormalise(args)

def transform_builder(transform_config):
    # create list of transforms and then compose them
    transforms_list = []
    for transform, args in transform_config:
        transforms_list.append(get_transform(transform, args))
    return transforms.Compose(transforms_list)