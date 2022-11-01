import torch
from src.data.datasets import get_dataset, download_data
import pytorch_lightning as pl
import numpy as np
from src.models.RENI import get_model
from src.utils.loss_functions import (
    RENITrainLoss,
    RENIVADTrainLoss,
    RENITestLoss,
    RENITestLossInverse,
)
from torch.utils.data import DataLoader
from src.utils.utils import get_directions, get_sineweight, get_mask
from src.utils.custom_transforms import transform_builder
from src.utils.pytorch3d_envmap_shader import build_renderer as PyTorch3DRenderer
from src.utils.pytorch3d_envmap_shader import EnvironmentMap
import os
import tqdm


class RENI(pl.LightningModule):
    def __init__(self, config, task):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.task = task
        self.model_from_checkpoint = False
        self.example_input_array = torch.randn(
            self.config.TRAINER.LOGGER.NUMBER_OF_IMAGES,
            self.config.RENI.LATENT_DIMENSION,
            3,
        )

    def prepare_data(self):
        download_data(self.config)

    def setup(self, stage=None):
        if not self.model_from_checkpoint:
            self.setup_dataset()
            self.model = get_model(self.config, len(self.dataset), self.task)

        self.model_type = self.config.RENI.MODEL_TYPE

        self.directions = get_directions(self.cur_res[1])  # (1, H*W, 3)
        self.sineweight = get_sineweight(self.cur_res[1])  # (1, H*W, 3)

        self.setup_for_task(self.task)

        self.mask = None
        if self.task == "FIT_LATENT" and self.config.RENI.FIT_LATENT.APPLY_MASK:
            print("Training with masked data")
            self.mask = get_mask(
                self.cur_res[1], self.config.RENI.FIT_LATENT.MASK_PATH
            )  # (1, H*W, 3)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.setup_dataset()
        self.model = get_model(self.config, len(self.dataset), self.task)
        self.model_from_checkpoint = True

    def load_state_dict(self, state_dict, strict: bool = True):
        self.model.load_state_dict(state_dict)

    def on_fit_start(self):
        if self.task == "FIT_INVERSE":
            self.renderer, self.R, self.T, self.mesh = PyTorch3DRenderer(
                obj_path=self.config.RENI.FIT_INVERSE.OBJECT_PATH,
                obj_rotation=0,
                img_size=self.config.RENI.FIT_INVERSE.RENDER_RESOLUTION,
                kd=self.config.RENI.FIT_INVERSE.KD_VALUE,
                device=self.device,
            )
            self.generate_gt_renders()

    def forward(self, z):
        batch_size = z.size(0)
        directions = self.directions.repeat(batch_size, 1, 1).type_as(z)
        return self.model(z, directions)

    def training_step(self, batch, batch_idx):
        imgs, idx = batch
        batch_size, _, _, _ = imgs.size()
        imgs = imgs.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        imgs = imgs.view(batch_size, -1, 3)  # (B, H, W, 3) -> (B, H*W, 3)

        if self.global_step == 0:
            self.example_input_array = self.example_input_array.type_as(imgs)

        directions = self.directions.repeat(batch_size, 1, 1).type_as(imgs)
        sineweight = self.sineweight.repeat(batch_size, 1, 1).type_as(imgs)

        if self.mask is not None:
            mask = self.mask.repeat(batch_size, 1, 1).type_as(imgs)
            sineweight = sineweight * mask

        #### GET MODEL OUTPUT ####
        if self.model_type == "AutoDecoder":
            Z = self.model.Z[idx, :, :]
        elif self.model_type == "VariationalAutoDecoder":
            if self.task == "FIT_DECODER":
                Z, mu, log_var = self.model.sample_latent(idx)
            else:
                Z = self.model.mu[idx, :, :]

        model_output = self.model(Z, directions)

        if self.task == "FIT_INVERSE":
            # ground truth images are now renders
            imgs = self.gt_renders[idx, :, :, :]
            # get the renders using model output
            model_output = self.dataset.unnormalise(model_output)
            model_output = self.get_render(model_output, directions, sineweight)

        #### COMPUTE LOSS ####
        if self.task == "FIT_DECODER":
            if self.model_type == "AutoDecoder":
                loss = self.criterion(model_output, imgs, sineweight)
                log_dict = {"loss": loss}
            elif self.model_type == "VariationalAutoDecoder":
                loss, mse_loss, kld_loss = self.criterion(
                    model_output, imgs, sineweight, mu, log_var
                )
                log_dict = {"loss": loss, "mse_loss": mse_loss, "kld_loss": kld_loss}

        elif self.task == "FIT_LATENT":
            loss, mse_loss, prior_loss, cosine_loss = self.criterion(
                model_output, imgs, sineweight, Z
            )
            log_dict = {
                "loss": loss,
                "mse_loss": mse_loss,
                "prior_loss": prior_loss,
                "cosine_loss": cosine_loss,
            }
        elif self.task == "FIT_INVERSE":
            loss, mse_loss, prior_loss, cosine_loss = self.criterion(
                model_output, imgs, Z
            )
            log_dict = {
                "loss": loss,
                "mse_loss": mse_loss,
                "prior_loss": prior_loss,
                "cosine_loss": cosine_loss,
            }

        return log_dict

    def training_epoch_end(self, training_step_outputs):
        metrics = {}
        for key in training_step_outputs[0].keys():
            metrics[f"{self.task.lower()}_{key}"] = torch.mean(
                torch.stack([x[key] for x in training_step_outputs])
            )
        metrics["step"] = self.current_epoch + 1.0

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

    def train_dataloader(self):
        return self.dataloader

    def build_optimizer(
        self,
        model,
        model_type,
        optimizer_type,
        learning_rate,
        beta1,
        beta2,
        fixed_decoder=False,
    ):
        if fixed_decoder:
            # We are only optimising the latent codes in this case
            if model_type == "VariationalAutoDecoder":
                parameters = [model.mu]
            elif model_type == "AutoDecoder":
                parameters = [model.Z]
        else:
            parameters = model.parameters()

        if optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                parameters, lr=learning_rate, betas=(beta1, beta2)
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        elif optimizer_type == "adagrad":
            optimizer = torch.optim.adagrad(parameters, lr=learning_rate)

        return optimizer

    def build_scheduler(
        self,
        scheduler_type,
        optimizer,
        step_size,
        lr_start=None,
        lr_end=None,
        gamma=None,
        epochs=None,
    ):
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "exponential":
            gamma = np.exp(np.log(lr_end / lr_start) / epochs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=gamma, patience=step_size, verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def configure_optimizers(self):
        optimizer = self.build_optimizer(
            self.model,
            self.config.RENI.MODEL_TYPE,
            self.optimiser_type,
            self.lr_start,
            self.beta1,
            self.beta2,
            self.fixed_decoder,
        )
        scheduler = self.build_scheduler(
            self.scheduler_type,
            optimizer,
            self.step_size,
            self.lr_start,
            self.lr_end,
            self.gamma,
            self.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "loss",
                "frequency": 1,
                "strict": True,
            },
        }

    def setup_dataset(self):
        dataset_name = self.config.DATASET.NAME
        dataset_path = self.config.DATASET[dataset_name].PATH
        self.is_hdr = self.config.DATASET[dataset_name].IS_HDR

        ##### SETUP TRANSFORMS #####
        if self.config.RENI[self.task].MULTI_RES_TRAINING:
            img_size = self.config.RENI[self.task].INITAL_RESOLUTION
        else:
            img_size = self.config.RENI[self.task].FINAL_RESOLUTION

        self.cur_res = img_size

        transform_list = [["resize", img_size]]
        for transform in self.config.DATASET[dataset_name].TRANSFORMS:
            transform_list.append(transform)
        transforms = transform_builder(transform_list)

        ##### SETUP DATASET #####
        if self.task == "FIT_DECODER":
            self.dataset = get_dataset(
                dataset_name, dataset_path + os.sep + "Train", transforms, self.is_hdr
            )
        else:
            self.dataset = get_dataset(
                dataset_name, dataset_path + os.sep + "Test", transforms, self.is_hdr
            )

        self.batch_size = self.config.RENI[self.task].BATCH_SIZE

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

    def setup_for_task(self, task):
        if task == "FIT_DECODER":
            self.lr_start = self.config.RENI.FIT_DECODER.LR_START
            self.lr_end = self.config.RENI.FIT_DECODER.LR_END
            self.beta1 = self.config.RENI.FIT_DECODER.OPTIMIZER_BETA_1
            self.beta2 = self.config.RENI.FIT_DECODER.OPTIMIZER_BETA_2
            self.optimiser_type = self.config.RENI.FIT_DECODER.OPTIMIZER
            self.scheduler_type = self.config.RENI.FIT_DECODER.SCHEDULER_TYPE
            self.epochs = self.config.RENI.FIT_DECODER.EPOCHS
            self.step_size = self.config.RENI.FIT_DECODER.SCHEDULER_STEP_SIZE
            self.gamma = self.config.RENI.FIT_DECODER.SCHEDULER_GAMMA
            self.multi_res_training = self.config.RENI.FIT_DECODER.MULTI_RES_TRAINING
            self.curriculum = self.config.RENI.FIT_DECODER.CURRICULUM
            h_start = self.config.RENI.FIT_DECODER.INITAL_RESOLUTION[0]
            h_end = self.config.RENI.FIT_DECODER.FINAL_RESOLUTION[0]
            self.fixed_decoder = False

            if self.model_type == "AutoDecoder":
                self.criterion = RENITrainLoss()
            elif self.model_type == "VariationalAutoDecoder":
                self.criterion = RENIVADTrainLoss(
                    beta=self.config.RENI.FIT_DECODER.KLD_WEIGHTING,
                    Z_dims=3 * self.config.RENI.LATENT_DIMENSION,
                )

        elif self.task == "FIT_LATENT":
            self.lr_start = self.config.RENI.FIT_LATENT.LR_START
            self.lr_end = self.config.RENI.FIT_LATENT.LR_END
            self.beta1 = self.config.RENI.FIT_LATENT.OPTIMIZER_BETA_1
            self.beta2 = self.config.RENI.FIT_LATENT.OPTIMIZER_BETA_2
            self.optimiser_type = self.config.RENI.FIT_LATENT.OPTIMIZER
            self.scheduler_type = self.config.RENI.FIT_LATENT.SCHEDULER_TYPE
            self.epochs = self.config.RENI.FIT_LATENT.EPOCHS
            self.step_size = self.config.RENI.FIT_LATENT.SCHEDULER_STEP_SIZE
            self.gamma = self.config.RENI.FIT_LATENT.SCHEDULER_GAMMA
            self.multi_res_training = self.config.RENI.FIT_LATENT.MULTI_RES_TRAINING
            self.curriculum = self.config.RENI.FIT_LATENT.CURRICULUM
            h_start = self.config.RENI.FIT_LATENT.INITAL_RESOLUTION[0]
            h_end = self.config.RENI.FIT_LATENT.FINAL_RESOLUTION[0]
            self.fixed_decoder = True

            self.criterion = RENITestLoss(
                alpha=self.config.RENI.FIT_LATENT.PRIOR_LOSS_WEIGHT,
                beta=self.config.RENI.FIT_LATENT.COSINE_SIMILARITY_WEIGHT,
            )

        elif self.task == "FIT_INVERSE":
            self.lr_start = self.config.RENI.FIT_INVERSE.LR_START
            self.lr_end = self.config.RENI.FIT_INVERSE.LR_END
            self.beta1 = self.config.RENI.FIT_INVERSE.OPTIMIZER_BETA_1
            self.beta2 = self.config.RENI.FIT_INVERSE.OPTIMIZER_BETA_2
            self.optimiser_type = self.config.RENI.FIT_INVERSE.OPTIMIZER
            self.scheduler_type = self.config.RENI.FIT_INVERSE.SCHEDULER_TYPE
            self.epochs = self.config.RENI.FIT_INVERSE.EPOCHS
            self.step_size = self.config.RENI.FIT_INVERSE.SCHEDULER_STEP_SIZE
            self.gamma = self.config.RENI.FIT_INVERSE.SCHEDULER_GAMMA
            self.multi_res_training = self.config.RENI.FIT_INVERSE.MULTI_RES_TRAINING
            self.curriculum = self.config.RENI.FIT_INVERSE.CURRICULUM
            h_start = self.config.RENI.FIT_INVERSE.INITAL_RESOLUTION[0]
            h_end = self.config.RENI.FIT_INVERSE.FINAL_RESOLUTION[0]
            self.fixed_decoder = True

            self.criterion = RENITestLossInverse(
                alpha=self.config.RENI.FIT_INVERSE.PRIOR_LOSS_WEIGHT,
                beta=self.config.RENI.FIT_INVERSE.COSINE_SIMILARITY_WEIGHT,
            )

        # Check config is valid
        assert max(self.curriculum) < self.epochs
        assert len(self.curriculum) >= np.log2(h_end / h_start)

    def generate_gt_renders(self):
        with torch.no_grad():
            self.gt_renders = []
            print("Generating ground truth renders...")
            for _, data in tqdm.tqdm(enumerate(self.dataloader)):
                imgs, _ = data
                if len(imgs.shape) == 3:
                    imgs = imgs.unsqueeze(0)
                imgs = imgs.to(self.device)
                imgs = self.dataset.unnormalise(imgs)
                batch_size, _, _, _ = imgs.size()
                imgs = imgs.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
                imgs = imgs.view(batch_size, -1, 3)  # (B, H, W, 3) -> (B, H*W, 3)

                directions = self.directions.repeat(batch_size, 1, 1).type_as(imgs)
                sineweight = self.sineweight.repeat(batch_size, 1, 1).type_as(imgs)

                render = self.get_render(imgs, directions, sineweight)

                self.gt_renders.append(render)

            self.gt_renders = torch.cat(self.gt_renders, dim=0)

    def get_render(self, model_output, directions, sineweight):
        envmap = EnvironmentMap(
            environment_map=model_output,
            directions=directions,
            sineweight=sineweight,
        )

        render, _ = self.renderer(
            meshes_world=self.mesh, R=self.R, T=self.T, envmap=envmap
        )
        return render
