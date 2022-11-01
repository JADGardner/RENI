from pytorch_lightning.callbacks import Callback
import torch
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
from src.utils.utils import sRGB
from src.utils.utils import get_directions, get_sineweight, get_mask
from matplotlib import pyplot as plt
import numpy as np


class MultiResTrainingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.multi_res_training:
            if pl_module.current_epoch + 1 in pl_module.curriculum:
                # double the resolution of directions, sineweight, mask and dataset
                # self.resize.size = (self.resize.size[0] * 2, self.resize.size[1] * 2)
                pl_module.cur_res = [2 * x for x in pl_module.cur_res]
                pl_module.directions = get_directions(pl_module.cur_res[1]) # (1, H*W, 3)
                pl_module.sineweight = get_sineweight(pl_module.cur_res[1])  # (1, H*W, 3)
                if pl_module.mask is not None:
                    pl_module.mask = get_mask(
                        pl_module.cur_res[1], pl_module.config.RENI.FIT_LATENT.MASK_PATH
                    )
                pl_module.train_dataloader().dataset.double_resolution()
                trainer.reload_dataloaders_every_n_epochs = True  # used to force the dataloader to be reloaded with updated resolution
            elif pl_module.current_epoch + 1 in [x + 1 for x in pl_module.curriculum]:
                trainer.reload_dataloaders_every_n_epochs = (
                    False  # turn off on next epoch
                )


class LogExampleImagesCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (
            not trainer.current_epoch
            % pl_module.config.TRAINER.LOGGER.EPOCHS_BETWEEN_EXAMPLES
        ):
            if pl_module.config.TRAINER.LOGGER.IMAGES_TO_SHOW == "noise":

                with torch.no_grad():
                    pl_module.eval()
                    z = pl_module.example_input_array
                    model_output = pl_module(z)
                    model_output = model_output.view(z.shape[0], *pl_module.cur_res, 3)
                    model_output = model_output.permute(0, 3, 1, 2)  # (B, C, H, W)
                    if pl_module.train_dataloader().dataset.unnormalise is not None:
                        model_output = pl_module.dataset.unnormalise(model_output)
                    if pl_module.is_hdr:
                        model_output = sRGB(model_output)
                    img_grid = make_grid(model_output, nrow=5, pad_value=2)
            else:
                # Generate a subset of images from the dataset
                if (
                    len(pl_module.train_dataloader().dataset)
                    < pl_module.config.TRAINER.LOGGER.NUMBER_OF_IMAGES
                ):
                    num_images = len(pl_module.train_dataloader().dataset)
                else:
                    num_images = pl_module.config.TRAINER.LOGGER.NUMBER_OF_IMAGES

                if pl_module.config.TRAINER.LOGGER.IMAGES_TO_SHOW == "random":
                    # select random non-overlapping images from the dataset
                    idx = torch.randperm(len(pl_module.train_dataloader().dataset))[
                        :num_images
                    ]
                elif isinstance(pl_module.config.TRAINER.LOGGER.IMAGES_TO_SHOW, list):
                    idx = pl_module.config.TRAINER.LOGGER.IMAGES_TO_SHOW
                subset = Subset(pl_module.train_dataloader().dataset, idx)
                example_loader = DataLoader(subset, batch_size=len(idx))

                with torch.no_grad():
                    pl_module.eval()
                    imgs, _ = next(iter(example_loader))
                    batch_size, _, H, W = imgs.size()

                    imgs = imgs.to(pl_module.device)
                    idx = idx.to(pl_module.device)
                    directions = pl_module.directions.repeat(batch_size, 1, 1).to(
                        pl_module.device
                    )
                    sineweight = pl_module.sineweight.repeat(batch_size, 1, 1).to(
                        pl_module.device
                    )

                    if pl_module.mask is not None:
                        mask = pl_module.mask.type_as(imgs)
                        mask = pl_module.mask.repeat(batch_size, 1, 1)
                        mask = mask.view(batch_size, H, W, 3)
                        mask = mask.permute(0, 3, 1, 2)  # (B, C, H, W)
                        mask = mask.type_as(imgs)
                        imgs = imgs * mask

                    model_output = pl_module.model(idx, directions)

                    if pl_module.task == "FIT_INVERSE":
                        # ground truth images are now renders
                        imgs = pl_module.gt_renders[idx, :, :, :]
                        # get the renders using model output
                        model_output = pl_module.dataset.unnormalise(model_output)
                        # get the rendered output one at a time
                        # to avoid memory issues
                        model_output = torch.cat(
                            [
                                pl_module.get_render(
                                    model_output[i:i+1, :, :], directions[i:i+1, :, :], sineweight[i:i+1, :, :]
                                )
                                for i in range(batch_size)
                            ], dim=0
                        )
                        imgs = imgs.permute(0, 3, 1, 2) # (B, C, H, W)
                        model_output = model_output.permute(0, 3, 1, 2)  # (B, C, H, W)
                    else:
                        model_output = model_output.view(batch_size, H, W, 3)
                        model_output = model_output.permute(0, 3, 1, 2)  # (B, C, H, W)
                        if pl_module.train_dataloader().dataset.unnormalise is not None:
                            imgs = pl_module.train_dataloader().dataset.unnormalise(
                                imgs
                            )
                            model_output = pl_module.dataset.unnormalise(model_output)

                    if pl_module.is_hdr:
                        imgs = sRGB(imgs)
                        model_output = sRGB(model_output)

                    imgs = torch.concat((imgs, model_output), dim=0)
                    img_grid = make_grid(imgs, nrow=5, pad_value=2)

            if pl_module.config.TRAINER.LOGGER_TYPE == "wandb":
                wandb_logger = pl_module.logger
                wandb_logger.log_image(
                    key=f"{pl_module.task.lower()}_images",
                    images=[img_grid],
                    step=trainer.current_epoch,
                )
            elif pl_module.config.TRAINER.LOGGER_TYPE == "tensorboard":
                tb = pl_module.logger.experiment
                tb.add_image(
                    f"{pl_module.task.lower()}_images", img_grid, trainer.current_epoch
                )
            pl_module.train()
