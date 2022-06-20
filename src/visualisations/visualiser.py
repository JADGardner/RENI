from operator import mod
import os
import torch
import models.RENI as RENI
from data.hdri_dataset import HDRIDataset
import utils.utils as utils
import models.spherical_harmonics as sh
import models.spherical_gaussians as spherical_gaussians
from torchvision import transforms
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import utils.model_utils as model_utils
import torch.nn.functional as F
from pytorch3d.transforms import RotateAxisAngle
import matplotlib.colors
import matplotlib.pyplot as plt
import yaml
import math
import imageio
import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from skimage.transform import resize
from utils.pytorch3d_envmap_shader import (
    EnvironmentMap,
    build_renderer,
    get_normal_map,
)
import torch.nn.functional as f
from PIL import Image, ImageFont, ImageDraw
import roma
from torch.nn.functional import normalize

def show_images(imgs, num_rows, num_cols, col_headers, save=False, fname=None):
    fig, axarr = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    for i in range(num_cols):
        if num_rows < 2:
            axarr[i].set_title(col_headers[i], fontsize=18)
        else:
            axarr[0, i].set_title(col_headers[i], fontsize=18)
    axarr = axarr.flatten()
    for ax, im in zip(axarr, imgs):
        ax.imshow(im)
        ax.xaxis.tick_top()
        ax.axis("off")

    fig.tight_layout()

    if save:
        fig.savefig(fname, facecolor="white", transparent=False, dpi=200)


def show_heatmaps(imgs, num_rows, num_cols, min_max, save=False, fname=None):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(17, 12))

    norms = []
    for ax, im, mm in zip(axs.flatten(), imgs, min_max):
        vmin, vmax = mm
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        norms.append(norm)
        im = ax.imshow(im, aspect="equal", cmap="inferno", norm=norm)
        ax.axis("off")

    col_headers = ["Ground Truth", "RENI", "SH", "SG"]
    for i in range(num_cols):
        if num_rows < 2:
            axs[i].set_title(col_headers[i], fontsize=18)
        else:
            axs[0, i].set_title(col_headers[i], fontsize=18)

    for row in range(num_rows):
        mappable = matplotlib.cm.ScalarMappable(norm=norms[row], cmap="inferno")
        cbar = fig.colorbar(mappable, ax=axs[row, :], location="right")
        cbar.ax.tick_params(labelsize=12)

    if save:
        fig.savefig(fname + ".png", facecolor="white", transparent=False)


class Visualiser:
    def __init__(
        self,
        path_of_run,
        device,
        training_type="Test",
        img_size=None,
        epoch=None,
        kd_value=1.0,
        obj_name="teapot",
        minmax=None,
    ):
        self.path_of_run = os.getcwd() + os.sep + ".." + os.sep + path_of_run
        self.device = device

        config_path = self.path_of_run + "files/config.yaml"
        with open(config_path) as yamlfile:
            config = yaml.safe_load(yamlfile)
            config = utils.dotdict(config)

        self.training_type = training_type
        self.kd_value = kd_value
        self.obj_name = obj_name
        self.inverse = False

        if self.training_type == "Train":
            if epoch is not None:
                model_path = (
                    self.path_of_run
                    + "files/checkpoints/RENI_Epoch"
                    + str(epoch)
                    + ".pt"
                )
            else:
                model_path = self.path_of_run + "files/RENI.pt"
        elif self.training_type == "Test":
            if epoch is not None:
                model_path = (
                    self.path_of_run
                    + "files/checkpoints/RENI_Latent_Epoch"
                    + str(epoch)
                    + ".pt"
                )
            else:
                model_path = self.path_of_run + "files/RENI_Latent.pt"
        elif self.training_type == "Inverse":
            self.inverse = True
            if epoch is not None:
                model_path = (
                    self.path_of_run
                    + "files/checkpoints/RENI_Inverse_Epoch"
                    + str(epoch)
                    + ".pt"
                )
            else:
                model_path = (
                    self.path_of_run
                    + "files/inverse/"
                    + obj_name
                    + "/kd_"
                    + str(self.kd_value).replace(".", "_")
                    + "/RENI_Inverse.pt"
                )
        elif self.training_type == "EnvironmentCompletion":
            self.apply_mask = config.apply_mask.value
            self.mask_path = None
            if self.apply_mask:
                self.mask_path = (
                    os.getcwd() + os.sep + ".." + os.sep + config.mask_path.value
                )
            model_path = self.path_of_run + "files/RENI_Latent.pt"

        if config.previous_run_path.value != "None":
            prev_run_config_path = (
                os.getcwd()
                + os.sep
                + ".."
                + os.sep
                + config.previous_run_path.value
                + "files/config.yaml"
            )
            with open(prev_run_config_path) as yamlfile:
                prev_run_config = yaml.safe_load(yamlfile)
                prev_run_config = utils.dotdict(prev_run_config)
                prev_dataset_path = (
                    os.getcwd()
                    + os.sep
                    + ".."
                    + os.sep
                    + prev_run_config.dataset_path.value
                )
                self.image_size = prev_run_config.resize_transform_end.value
                self.model_type = prev_run_config.model_type.value
                self.ndims = prev_run_config.ndims.value
                RENI_hidden_features = prev_run_config.RENI_hidden_features.value
                RENI_hidden_layers = prev_run_config.RENI_hidden_layers.value
        else:
            self.image_size = config.resize_transform_end.value
            self.model_type = config.model_type.value
            self.ndims = config.ndims.value
            RENI_hidden_features = config.RENI_hidden_features.value
            RENI_hidden_layers = config.RENI_hidden_layers.value

        # Set up dataset
        if (
            self.training_type == "Test"
            or self.training_type == "Inverse"
            or self.training_type == "EnvironmentCompletion"
        ):
            if minmax is None:
                # need to get minmax from original dataset
                transform = transforms.Resize((self.image_size // 2, self.image_size))
                if config.previous_run_path.value != "None":
                    dataset_path = prev_dataset_path
                else:
                    dataset_path = (
                        os.getcwd() + os.sep + ".." + os.sep + config.dataset_path.value
                    )
                dataset = HDRIDataset(dataset_path, transform)
                minmax = (dataset.min_value, dataset.max_value)
            dataset_path = (
                os.getcwd() + os.sep + ".." + os.sep + config.test_dataset_path.value
            )
            if img_size is not None:
                self.image_size = img_size
            self.resize_tf = transforms.Resize((self.image_size // 2, self.image_size))
            self.dataset = HDRIDataset(dataset_path, self.resize_tf, minmax)
        else:
            dataset_path = (
                os.getcwd() + os.sep + ".." + os.sep + config.dataset_path.value
            )
            if img_size is not None:
                self.image_size = img_size
            self.resize_tf = transforms.Resize((self.image_size // 2, self.image_size))
            self.dataset = HDRIDataset(dataset_path, self.resize_tf)

        if self.model_type == "RENIVariationalAutoDecoder":
            self.model = RENI.RENIVariationalAutoDecoder(
                in_features=2 * self.ndims + self.ndims * self.ndims + 2,
                out_features=3,
                hidden_features=RENI_hidden_features,
                hidden_layers=RENI_hidden_layers,
                dataset_size=len(self.dataset),
                ndims=self.ndims,
            )
            self.model.log_var = None
        elif self.model_type == "RENIAutoDecoder":
            self.model = RENI.RENIAutoDecoder(
                in_features=2 * self.ndims + self.ndims * self.ndims + 2,
                out_features=3,
                hidden_features=RENI_hidden_features,
                hidden_layers=RENI_hidden_layers,
                dataset_size=len(self.dataset),
                ndims=self.ndims,
                outermost_linear=True,
            )

        self.model = self.model.to(device=self.device)
        # At test time for Varational models we set log_var to None. So
        # now loading states only with available keys
        pretrained_dict = torch.load(model_path, map_location=self.device)
        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(pretrained_dict)

        # Get dimensionality of spherical harmonic and spherical gaussian models
        self.sh_order = self.get_sh_order(self.ndims)
        self.sg_dims = self.calc_closest_factors(int(math.ceil((self.ndims * 3) / 6)))

    def calc_closest_factors(self, c: int):
        if c // 1 != c:
            raise TypeError("c must be an integer.")

        a, b, i = 1, c, 0
        while a < b:
            i += 1
            if c % i == 0:
                a = i
                b = c // a

        return [a, b]

    def get_model_output(self, img_ids, rotation=0, Z=None):
        R = (
            RotateAxisAngle(rotation, "Y")
            .get_matrix()[:, 0:3, 0:3]
            .to(device=self.device)
        )
        if Z is None:
            subset = torch.utils.data.Subset(self.dataset, img_ids)
            loader = DataLoader(subset, batch_size=len(img_ids))
            data = next(iter(loader))
            directions = data[0].to(device=self.device)
            R = R.repeat(len(img_ids), 1, 1)
        else:
            directions, _, _, _, _ = self.dataset[0]
            directions = directions.unsqueeze(0).to(device=self.device)

        if self.model_type == "RENIVariationalAutoDecoder":
            if Z == None:
                Z = self.model.mu[img_ids, :, :]
                if len(img_ids) == 1:
                    Z.unsqueeze(0)
            model_input = model_utils.InvariantRepresentation(
                torch.bmm(Z, R), directions
            )
            model_input = model_input.to(device=self.device)
            # Make predictions
            model_output = self.model(model_input)

        elif self.model_type == "RENIAutoDecoder":
            if Z == None:
                Z = self.model.Z[img_ids, :, :]
                if len(img_ids) == 1:
                    Z.unsqueeze(0)
            model_input = model_utils.InvariantRepresentation(
                torch.bmm(Z, R), directions
            )
            model_input = model_input.to(device=self.device)
            # Make predictions
            model_output = self.model(model_input)
        return model_output

    def get_sg_approximation(self, img_ids, sg_dims=[1, 3]):
        subset = torch.utils.data.Subset(self.dataset, img_ids)
        loader = DataLoader(subset, batch_size=len(img_ids))
        data = next(iter(loader))
        envmaps = data[2]
        envmaps = self.dataset.undo_normalisation(envmaps)
        envmaps = torch.permute(envmaps, (0, 2, 1))
        envmaps = envmaps.view(len(img_ids), 3, *self.dataset.transform.size)
        sineweight = data[1].to(self.device)
        sineweight = torch.permute(sineweight, (0, 2, 1))
        sineweight = sineweight.view(len(img_ids), 3, *self.dataset.transform.size)

        isCuda = not self.device.type == "cpu"
        sg_optim = spherical_gaussians.SGEnvOptim(
            envNum=envmaps.shape[0],
            envHeight=envmaps.shape[2],
            envWidth=envmaps.shape[3],
            SGRow=sg_dims[0],
            SGCol=sg_dims[1],
            gpuId=self.device.index,
            isCuda=isCuda,
        )

        _, _, _, _, recImageBest = sg_optim.optimize(envmaps, sineweight)
        recImageBest = torch.from_numpy(recImageBest)

        return recImageBest

    def calc_num_sh_coeffs(self, order):
        coeffs = 0
        for i in range(order + 1):
            coeffs += 2 * i + 1
        return coeffs

    def get_sh_order(self, ndims):
        order = 0
        while self.calc_num_sh_coeffs(order) < ndims:
            order += 1
        return order

    def comparison(
        self,
        img_ids,
        show_renders=False,
        obj_name="bunny",
        save=False,
        fname="Comparison",
    ):
        with torch.no_grad():
            images = []
            col_headers = [
                "GT\n" + f"{self.image_size * self.image_size//2} Params",
                "RENI\n" + f"{(3*self.ndims)} Params",
                "SH Order "
                + str(self.sh_order)
                + "\n"
                + f"{(self.calc_num_sh_coeffs(self.sh_order) * 3)} Params",
                "SG\n" + f"{(self.sg_dims[0]*self.sg_dims[1]*6)} Params",
            ]

            if show_renders:
                col_headers.extend(
                    ["Render GT", "Render RENI", "Render SH", "Render SG"]
                )

            sg_outputs = self.get_sg_approximation(img_ids, self.sg_dims)

            for i, idx in enumerate(img_ids):
                directions, sineweight, ground_truth, _, _ = self.dataset[idx]
                directions = directions.to(device=self.device)
                sineweight = sineweight.to(device=self.device)
                ground_truth = ground_truth.to(device=self.device)

                model_output = self.get_model_output([idx], 0)

                ground_truth_np = self.dataset.to_numpy_convert_sRGB(ground_truth)
                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                sh_output = self.dataset.get_spherical_harmonic_representation(
                    idx, self.sh_order
                )
                sh_output = sh_output.view(
                    *self.dataset.transform.size, 3
                )  # (H*W, 3) -> (H, W, 3)
                sh_output_np = utils.sRGB(sh_output).cpu().detach().numpy()
                sg_output = sg_outputs[i, :, :, :]
                sg_output = sg_output.permute((1, 2, 0))
                sg_output_np = utils.sRGB(sg_output).cpu().detach().numpy()

                if show_renders:
                    ground_truth = self.dataset.undo_normalisation(ground_truth)
                    env_map = EnvironmentMap(
                        environment_map=ground_truth,
                        directions=directions,
                        sineweight=sineweight,
                    )
                    render_gt = self.pytorch3d_render(
                        env_map, obj_name, 0, self.kd_value
                    )

                    model_output = self.dataset.undo_normalisation(model_output)
                    env_map = EnvironmentMap(
                        environment_map=model_output,
                        directions=directions,
                        sineweight=sineweight,
                    )
                    render_model = self.pytorch3d_render(
                        env_map, obj_name, 0, self.kd_value
                    )

                    sh_output = self.dataset.undo_normalisation(sh_output)
                    env_map = EnvironmentMap(
                        environment_map=model_output,
                        directions=directions,
                        sineweight=sineweight,
                    )
                    render_sh = self.pytorch3d_render(
                        env_map, obj_name, 0, self.kd_value
                    )

                    sg_output = sg_output.view(-1, 3).to(device=self.device)
                    env_map = EnvironmentMap(
                        environment_map=sg_output,
                        directions=directions,
                        sineweight=sineweight,
                    )
                    render_sg = self.pytorch3d_render(
                        env_map, obj_name, 0, self.kd_value
                    )

                images.append(ground_truth_np)
                images.append(model_output_np)
                images.append(sh_output_np)
                images.append(sg_output_np)

                if show_renders:
                    images.append(render_gt)
                    images.append(render_model)
                    images.append(render_sh)
                    images.append(render_sg)

                del directions

            show_images(
                images, len(img_ids), len(col_headers), col_headers, save, fname
            )

    def show_masked_output(self, img_ids, save, fname):
        with torch.no_grad():
            _, _, _, _, _ = self.dataset[0]
            images = []
            col_headers = ["Ground Truth", "Masked Ground Truth", "RENI", "SH"]
            for idx in img_ids:
                directions, _, ground_truth, _, _ = self.dataset[idx]
                directions = directions.to(device=self.device)
                # Get ground truth sample
                ground_truth_np = self.dataset.to_numpy_convert_sRGB(ground_truth)
                mask = Image.open(self.mask_path)
                mask = ToTensor()(mask)
                mask = self.dataset.transform(mask)
                mask = (
                    mask.permute((1, 2, 0)).cpu().detach().numpy()
                )  # (3, H, W) -> (H, W, 3)
                ground_truth_masked = ground_truth_np * mask

                model_output = self.get_model_output([idx], 0)

                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)

                C = (
                    ground_truth.view(*self.dataset.transform.size, 3).detach().cpu()
                )  # (H*W, 3) -> (H, W, 3)
                lmax = sh.sh_lmax_from_termsTorch(torch.tensor(self.ndims))
                B = (
                    sh.getCoefficientsMatrixTorch(self.image_size, lmax, self.device)
                    .detach()
                    .cpu()
                )

                mask = Image.open(self.mask_path)
                mask = mask.convert("1")
                mask = ToTensor()(mask)
                resize = transforms.Resize(
                    self.dataset.transform.size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
                mask = resize(mask)
                mask = mask.permute((1, 2, 0))  # (1, H, W) -> (H, W, 1)
                mask = np.where(mask >= 1.0, 1, 0).squeeze()
                B = B[np.ix_(mask.any(1), mask.any(0))]
                C = C[np.ix_(mask.any(1), mask.any(0))]
                B = B.reshape(-1, B.shape[2])
                C = C.reshape(-1, C.shape[2])
                X = torch.linalg.lstsq(B, C).solution
                X = X.to(device=self.device)
                sh_inpainting = sh.shReconstructSignalTorch(
                    X, self.image_size, self.device
                )
                sh_inpainting = utils.sRGB(sh_inpainting).cpu().detach().numpy()

                images.append(ground_truth_np)
                images.append(ground_truth_masked)
                images.append(model_output_np)
                images.append(sh_inpainting)
                del directions
                del model_output

            show_images(
                images, len(img_ids), len(col_headers), col_headers, save, fname
            )

    def display_heatmaps(self, img_ids, save=False, fname=None):
        with torch.no_grad():
            images = []
            min_max = []
            sg_images = self.get_sg_approximation(img_ids, self.sg_dims)
            i = 0
            for idx in img_ids:
                directions, sineweight, _, _, img_path = self.dataset[idx]
                directions = directions.to(device=self.device)

                img = utils.exr_to_tensor(img_path)
                img = torch.where(
                    torch.isinf(img) == True, img[torch.isfinite(img)].max(), img
                )
                img = torch.maximum(torch.maximum(img[0], img[1]), img[2])
                img = img.cpu().detach().numpy()

                # Get model output
                model_output = self.get_model_output([idx], 0)

                model_output = model_output * 0.5 + 0.5
                model_output = (
                    model_output * (self.dataset.max_value - self.dataset.min_value)
                    + self.dataset.min_value
                )
                # model trained to match log(image intesity) so need to undo this using exp
                model_output = torch.exp(model_output)
                model_output_np = (
                    model_output.view(*self.dataset.transform.size, 3)
                    .cpu()
                    .detach()
                    .numpy()
                )
                model_output_np = np.maximum(
                    np.maximum(model_output_np[:, :, 0], model_output_np[:, :, 1]),
                    model_output_np[:, :, 2],
                )

                sh_img = self.dataset.get_spherical_harmonic_representation(
                    idx, self.sh_order
                )
                sh_img = (
                    sh_img.view(*self.dataset.transform.size, 3).cpu().detach().numpy()
                )
                sh_img = np.maximum(
                    np.maximum(sh_img[:, :, 0], sh_img[:, :, 1]), sh_img[:, :, 2]
                )

                sg_img = sg_images[i, :, :, :]
                sg_img = sg_img.permute((1, 2, 0))
                sg_img = (
                    sg_img.view(*self.dataset.transform.size, 3).cpu().detach().numpy()
                )
                sg_img = np.maximum(
                    np.maximum(sg_img[:, :, 0], sg_img[:, :, 1]), sg_img[:, :, 2]
                )

                # matplotlib complains if min = <= 0.0 so just use next smallest value
                img[img < 0] = 0.0
                minimum = (
                    img[img.nonzero()].min()
                    if img[img.nonzero()].min()
                    < model_output_np[model_output_np.nonzero()].min()
                    else model_output_np[model_output_np.nonzero()].min()
                )
                maximum = (
                    img.max()
                    if img.max() > model_output_np.max()
                    else model_output_np.max()
                )
                images.append(img)
                images.append(model_output_np)
                images.append(sh_img)
                images.append(sg_img)
                min_max.append((minimum, maximum))
                min_max.append((minimum, maximum))
                min_max.append((minimum, maximum))
                min_max.append((minimum, maximum))

                del directions
                del model_output
                i = i + 1

            show_heatmaps(images, len(img_ids), 4, min_max, save, fname)

    def linear_combinations(
        self,
        idx_one=None,
        idx_two=None,
        Z1=None,
        Z2=None,
        anti_code=False,
        save=False,
        fname=None,
        steps=5,
    ):
        with torch.no_grad():
            col_headers = []
            if idx_one is not None:
                col_headers.append("")
            if idx_two is not None:
                col_headers.append("")
            col_headers += ["", "", "", "", "", ""]
            images = []

            # Get ground truth sample
            if idx_one is not None:
                directions, _, ground_truth_one, _, _ = self.dataset[idx_one]
                ground_truth_one_np = self.dataset.to_numpy_convert_sRGB(
                    ground_truth_one
                )
                images.append(ground_truth_one_np)
            else:
                directions, _, _, _, _ = self.dataset[0]

            directions = directions.to(device=self.device)

            # Get model output
            if self.model_type == "RENIVariationalAutoDecoder":
                if idx_one is None and Z1 is None:
                    Z1 = torch.randn((1, self.ndims, 3), device=self.device)
                else:
                    if Z1 is None:
                        Z1 = self.model.mu[idx_one, :, :]
                if idx_two is None and Z2 is not None:
                    if anti_code:
                        Z2 = -Z1
                else:
                    if Z2 is None:
                        Z2 = self.model.mu[idx_two, :, :]

            elif self.model_type == "RENIAutoDecoder":
                if idx_one is None:
                    Z1 = torch.randn((1, self.ndims, 3), device=self.device)
                else:
                    Z1 = self.model.Z[idx_one, :, :]
                if idx_two is None:
                    if anti_code:
                        Z2 = -Z1
                    else:
                        Z2 = torch.randn((1, self.ndims, 3), device=self.device)
                else:
                    Z2 = self.model.Z[idx_two, :, :]

            for i in torch.arange(0, 1.0 + 1.0 / steps, 1.0 / steps):
                Z = Z1 * (1.0 - i) + Z2 * i

                model_output = self.get_model_output(None, 0, Z)
                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                images.append(model_output_np)

                del model_output

            if idx_two is not None:
                _, _, ground_truth_two, _, _ = self.dataset[idx_two]
                ground_truth_two_np = self.dataset.to_numpy_convert_sRGB(
                    ground_truth_two
                )
                images.append(ground_truth_two_np)

            del directions
            show_images(images, 3, int(len(col_headers) / 2), col_headers, save, fname)

    def random_latent_codes(self, num=8, save=False, fname=None):
        with torch.no_grad():
            col_headers = ["", ""]
            images = []
            directions, _, _, _, _ = self.dataset[0]
            directions = directions.to(device=self.device)

            for _ in range(num):

                # Z = torch.randn((1, self.ndims, 3), device=self.device)
                # More interesting and varied results using this:
                Z = (
                    torch.FloatTensor(self.ndims, 3)
                    .uniform_(-1.8, 1.8)
                    .unsqueeze(0)
                    .to(device=self.device)
                )
                model_output = self.get_model_output(None, 0, Z)

                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                images.append(model_output_np)

                del model_output

            del directions

            show_images(images, num // 2, 2, col_headers, save, fname)

    def zero_latent_code(self, save=False, fname=None):
        with torch.no_grad():
            col_headers = ["", ""]
            text_for_images = []
            images = []
            directions, _, _, _, _ = self.dataset[0]
            directions = directions.to(device=self.device)

            # Get model output
            if self.model_type == "RENIVariationalAutoDecoder":
                Z = torch.zeros((1, self.ndims, 3), device=self.device)
                model_output = self.get_model_output(None, 0, Z)

            elif self.model_type == "RENIAutoDecoder":
                Z = torch.zeros((1, self.ndims, 3), device=self.device)
                model_output = self.get_model_output(None, 0, Z)

            model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
            images.append(model_output_np)
            text_for_images.append(f"")

            del model_output

            del directions

            show_images(images, text_for_images, 1, 2, col_headers, save, fname)

    def pytorch3d_render(
        self,
        env_map,
        obj_name,
        obj_rotation,
        kd,
        save=False,
        fname="PyTorch3D"
    ):
        with torch.no_grad():
            obj_path = (
                os.getcwd()
                + os.sep
                + ".."
                + os.sep
                + "data/graphics/3d_models/"
                + obj_name
                + ".obj"
            )
            blinn_phong_envmap_renderer, R, T, mesh = build_renderer(
                obj_path, obj_rotation, 128, kd, self.device
            )
            render, normals = blinn_phong_envmap_renderer(
                meshes_world=mesh, R=R, T=T, envmap=env_map
            )
            render = render[..., :3]  # Don't need alpha channel
            render = utils.sRGB(render).detach().cpu().numpy()

            # Masking object to set background to white
            normals = normals.cpu().detach().numpy()
            normals[normals == 0.0] = np.nan
            render = np.where(np.isnan(normals) == True, 1.0, render)
            if save:
                plt.imsave(fname + ".png", render)

            return render

    def create_teaser_animation(
        self,
        img_idx,
        obj_name,
        number_of_frames,
        mode="Pytorch3D",
        fname="teaser",
        is_ipython=False,
    ):
        with torch.no_grad():
            frames = []

            neural_field = imageio.imread(
                os.getcwd() + os.sep + ".." + os.sep + "data/other/neural_field.png",
                pilmode="RGB",
            )
            image_caption = imageio.imread(
                os.getcwd() + os.sep + ".." + os.sep + "data/other/image_caption.png",
                pilmode="RGB",
            )
            directions, sineweight, _, _, _ = self.dataset[img_idx]
            directions = directions.to(device=self.device)  # (K, 3)
            sineweight = sineweight.to(device=self.device)  # (K, 3)

            print("rendering frames")
            for i in np.arange(0, 720, 720 / number_of_frames):
                rotation = i
                if mode == "Pytorch3D":
                    model_output = self.get_model_output([img_idx], rotation)
                    model_output_unormalised = self.dataset.undo_normalisation(
                        model_output
                    )
                    env_map = EnvironmentMap(
                        environment_map=model_output_unormalised,
                        directions=directions,
                        sineweight=sineweight,
                    )
                    image = self.pytorch3d_render(env_map, obj_name, 0, self.kd_value)
                    latent_code = self.plot_latent_code(
                        img_idx, -rotation, Z=None, save=False, is_ipython=is_ipython
                    )
                    latent_code = resize(latent_code, (128, 144))
                    image = np.concatenate((image, latent_code), axis=1)
                    model_output = self.dataset.to_numpy_convert_sRGB(model_output)
                    model_output = resize(model_output, (128, 256))
                    image = np.concatenate((image, model_output), axis=1)
                elif mode == "teaser_animation":
                    latent_code = self.plot_latent_code(
                        img_idx, -rotation, Z=None, save=False, is_ipython=is_ipython
                    )
                    latent_code = resize(
                        latent_code,
                        (
                            self.dataset.transform.size[0],
                            int(
                                self.dataset.transform.size[0]
                                * latent_code.shape[1]
                                / latent_code.shape[0]
                            ),
                        ),
                    )
                    neural_field_resize = resize(
                        neural_field,
                        (
                            self.dataset.transform.size[0],
                            int(
                                self.dataset.transform.size[0]
                                * neural_field.shape[1]
                                / neural_field.shape[0]
                            ),
                        ),
                    )
                    image = np.concatenate((latent_code, neural_field_resize), axis=1)
                    model_output = self.get_model_output([img_idx], rotation)
                    envmap = self.dataset.to_numpy_convert_sRGB(model_output)
                    image = np.concatenate((image, envmap), axis=1)
                    image_caption_resize = resize(
                        image_caption,
                        (
                            int(
                                image.shape[1]
                                * image_caption.shape[0]
                                / image_caption.shape[1]
                            ),
                            image.shape[1],
                        ),
                    )
                    image = np.concatenate((image, image_caption_resize), axis=0)

                image_int = (image * 255).astype(np.uint8)
                frames.append(image_int)

            out = cv2.VideoWriter(
                fname + ".avi",
                cv2.VideoWriter_fourcc(*"DIVX"),
                30,
                (image.shape[1], image.shape[0]),
            )
            for i in range(number_of_frames):
                out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            out.release()

            print("saving gif")
            imageio.mimsave(fname + ".gif", frames, fps=30)

    def plot_latent_code(
        self, idx, rotation, Z=None, save=False, fname="latent_code", is_ipython=False
    ):
        with torch.no_grad():
            fig, ax = plt.subplots(
                subplot_kw=dict(projection="3d"),
            )
            for item in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
                item.set_fontsize(20)

            for item in (
                ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
            ):
                item.set_fontsize(10)

            fig.set_size_inches(8, 8)

            if Z is None:
                if self.model_type == "RENIVariationalAutoDecoder":
                    Z = self.model.mu[idx, :, :]
                elif self.model_type == "RENIAutoDecoder":
                    Z = self.model.Z[idx, :, :]

            c_choices = [
                [0.188, 0.204, 0.729, 1.0],
                [0.082, 0.478, 0.431, 1.0],
                [0.996, 0.290, 0.286, 1.0],
            ]

            colours = [c_choices[i % 3] for i in range((self.ndims))]
            for i in range(self.ndims * 2):
                colours.append(c_choices[i % 3])
                colours.append(c_choices[i % 3])

            R = RotateAxisAngle(rotation, "Y")
            R = R.get_matrix().squeeze(0)[0:3, 0:3]
            R = R.to(device=self.device)
            Z = Z @ R
            Z = F.pad(input=Z, pad=(3, 0), mode="constant", value=0)
            Z = Z.cpu().detach().numpy()
            ax.quiver(
                Z[:, 0],
                Z[:, 1],
                Z[:, 2],
                Z[:, 3],
                Z[:, 5],
                Z[:, 4],
                linewidths=2.5,
                arrow_length_ratio=0.2,
                length=2,
                normalize=True,
                colors=colours,
            )
            # ax.scatter(Z[:,0], Z[:,2], Z[:,1])
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("y")
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_zticks([-2, 0, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])

            if save:
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8
                )
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,)
                )
                image_from_plot = image_from_plot[98:505, 98:540, :]
                plt.imsave(fname + str(rotation) + ".png", image_from_plot)
            else:
                fig.canvas.draw()
                image_from_plot = np.frombuffer(
                    fig.canvas.tostring_rgb(), dtype=np.uint8
                )
                image_from_plot = image_from_plot.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,)
                )
                if is_ipython:
                    image_from_plot = image_from_plot[100:505, 98:538, :]
                else:
                    image_from_plot = image_from_plot[135:725, 135:750, :]
                plt.close()
                return image_from_plot

    def interpolation_animation(
        self,
        num_of_Z=None,
        img_ids=None,
        number_of_frames=10,
        fname="interpolation_animation",
    ):
        frames = []
        Zds = []

        print("rendering frames")

        k = 0

        # Get ground truth sample
        if img_ids is not None:
            directions, _, _, _, _ = self.dataset[img_ids[0]]
        else:
            directions, _, _, _, _ = self.dataset[0]

        directions = directions.to(device=self.device)

        # Get model output

        if img_ids is not None:
            for i in img_ids:
                if self.model_type == "RENIVariationalAutoDecoder":
                    Zds.append(self.model.mu[i, :, :])
                elif self.model_type == "RENIAutoDecoder":
                    Zds.append(self.model.Z[i, :, :])
        else:
            for i in range(num_of_Z):
                Zds.append(torch.randn((1, self.ndims, 3), device=self.device))
                Zds.append(
                    torch.FloatTensor(1, self.ndims, 3)
                    .uniform_(-1.8, 1.8)
                    .to(device=self.device)
                )
        Zds.append(Zds[0])

        for j in range(len(Zds) - 1):
            for i in torch.arange(
                0, 1.0 + 1.0 / number_of_frames, 1.0 / number_of_frames
            ):

                Z = Zds[j] * (1.0 - i) + Zds[j + 1] * i
                model_output = self.get_model_output(None, 0, Z)
                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                image_int = (model_output_np * 255).astype(np.uint8)
                frames.append(image_int)

                del model_output

        print("saving gif")
        imageio.mimsave(fname + ".gif", frames, fps=30)

        out = cv2.VideoWriter(
            fname + ".avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            30,
            (image_int.shape[1], image_int.shape[0]),
        )
        for i in range(len(frames)):
            out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        out.release()

    def get_sh_inverse(self, img_idx, obj_name, obj_rotation):
        directions, sineweight, ground_truth, _, _ = self.dataset[img_idx]
        directions = directions.to(device=self.device)  # (K, 3)
        sineweight = sineweight.to(device=self.device).squeeze(0)
        ground_truth = ground_truth.to(device=self.device)

        obj_path = (
            os.getcwd()
            + os.sep
            + ".."
            + os.sep
            + "data/graphics/3d_models/"
            + obj_name
            + ".obj"
        )
        normal_map = get_normal_map(obj_path, obj_rotation, self.device)

        ground_truth = self.dataset.undo_normalisation(ground_truth)
        env_map = EnvironmentMap(
            environment_map=ground_truth, directions=directions, sineweight=sineweight
        )
        pixels = self.pytorch3d_render(
            env_map, obj_name, 0, self.kd_value
        )

        normal_map[:,:,2] = -normal_map[:,:,2]
        normals = torch.tensor(normal_map).view(-1, 3)
        pixels = torch.tensor(pixels).view(-1, 3)
        N = normals[~torch.any(normals.isnan(), dim=1)]
        N = normalize(N, 2, dim=1)
        B = pixels[~torch.any(normals.isnan(), dim=1)]
        A = sh.SH_Inverse_Coefficient_Matrix(N, self.sh_order)
        A = torch.from_numpy(A).type(torch.FloatTensor)

        X = torch.linalg.lstsq(A, B).solution

        N = normals.clone().detach().view(-1, 3)
        A = sh.SH_Inverse_Coefficient_Matrix(N, self.sh_order)
        A = torch.from_numpy(A).type(torch.FloatTensor)

        pixels = A @ X
        img = pixels.view(128, 128, 3)
        img = np.where(np.isnan(normal_map) == True, 0.0, img)
        img = torch.from_numpy(img)
        img = utils.sRGB(img).cpu().detach().numpy()
        img = np.where(np.isnan(normal_map) == True, 1.0, img)
        sh_radiance_map = sh.shReconstructSignal(X, width=self.image_size)
        sh_radiance_map = torch.from_numpy(sh_radiance_map)
        sh_radiance_map = sh_radiance_map.view(-1, 3)  # (H*W, 3)
        sh_radiance_map = sh_radiance_map.view(self.image_size//2, self.image_size, 3)
        sh_radiance_map = utils.sRGB(sh_radiance_map).cpu().detach().numpy()
        return img, sh_radiance_map

    def inverse_comparison(
        self,
        img_ids,
        save=False,
        fname="Comparison",
    ):
        with torch.no_grad():
            images = []
            col_headers = [
                "GT\n" + f"{self.image_size * self.image_size//2} Params",
                "RENI\n" + f"{(3*self.ndims)} Params",
                "SH Order "
                + str(self.sh_order)
                + "\n"
                + f"{(self.calc_num_sh_coeffs(self.sh_order) * 3)} Params",
                "Render GT",
                "Render RENI",
                "Render SH",
            ]

            for i, idx in enumerate(img_ids):
                directions, sineweight, ground_truth, _, _ = self.dataset[idx]
                directions = directions.to(device=self.device)
                sineweight = sineweight.to(device=self.device)
                ground_truth = ground_truth.to(device=self.device)

                model_output = self.get_model_output([idx], 0)

                ground_truth_np = self.dataset.to_numpy_convert_sRGB(ground_truth)
                model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                render_sh, sh_output = self.get_sh_inverse(idx, self.obj_name, 0)

                ground_truth = self.dataset.undo_normalisation(ground_truth)
                env_map = EnvironmentMap(
                    environment_map=ground_truth,
                    directions=directions,
                    sineweight=sineweight,
                )
                render_gt = self.pytorch3d_render(
                    env_map, self.obj_name, 0, self.kd_value
                )

                model_output = self.dataset.undo_normalisation(model_output)
                env_map = EnvironmentMap(
                    environment_map=model_output,
                    directions=directions,
                    sineweight=sineweight,
                )
                render_model = self.pytorch3d_render(
                    env_map, self.obj_name, 0, self.kd_value
                )

                images.append(ground_truth_np)
                images.append(model_output_np)
                images.append(sh_output)
                images.append(render_gt)
                images.append(render_model)
                images.append(render_sh)

                del directions

            show_images(
                images, len(img_ids), len(col_headers), col_headers, save, fname
            )

    def PSNR(self):
        # Constant for numerical stability
        EPS = 1e-8

        PSNR_RENI = 0.0
        PSNR_SH = 0.0
        PSNR_SG = 0.0

        sg_outputs = self.get_sg_approximation(range(len(self.dataset)), self.sg_dims)

        for i in range(len(self.dataset)):
            directions, _, ground_truth, _, _ = self.dataset[i]
            directions = directions.to(device=self.device)

            model_output = self.get_model_output([i], 0).cpu().detach().squeeze(0)
            sh_output = self.dataset.get_spherical_harmonic_representation(
                i, self.sh_order
            )
            ground_truth = self.dataset.undo_normalisation(ground_truth)
            model_output = self.dataset.undo_normalisation(model_output)
            ground_truth = ground_truth.view(
                *self.dataset.transform.size, 3
            )  # (H*W, 3) -> (H, W, 3)
            ground_truth = utils.sRGB(ground_truth)
            model_output = model_output.view(
                *self.dataset.transform.size, 3
            )  # (H*W, 3) -> (H, W, 3)
            model_output = utils.sRGB(model_output)
            sh_output = sh_output.view(
                *self.dataset.transform.size, 3
            )  # (H*W, 3) -> (H, W, 3)
            sh_output = utils.sRGB(sh_output)

            sg_output = sg_outputs[i, :, :, :]
            sg_output = sg_output.permute((1, 2, 0))
            sg_output = sg_output.view(-1, 3)
            sg_output = sg_output.view(
                *self.dataset.transform.size, 3
            )  # (H*W, 3) -> (H, W, 3)
            sg_output = utils.sRGB(sg_output)

            mse_reni = torch.mean(((ground_truth - model_output) ** 2))
            mse_sh = torch.mean(((ground_truth - sh_output) ** 2))
            mse_sg = torch.mean((((ground_truth) - sg_output) ** 2))

            PSNR_RENI += -10 * torch.log10(mse_reni + EPS)
            PSNR_SH += -10 * torch.log10(mse_sh + EPS)
            PSNR_SG += -10 * torch.log10(mse_sg + EPS)

        PSNR_RENI = PSNR_RENI / len(self.dataset)
        PSNR_SH = PSNR_SH / len(self.dataset)
        PSNR_SG = PSNR_SG / len(self.dataset)
        return PSNR_RENI.item(), PSNR_SH.item(), PSNR_SG.item()

    def calculate_residual_error(self):
        errors = []
        total_error = 0.0
        # every other image
        for i in range(0, len(self.dataset) - 1, 2):
            # First image latent code
            Z1 = self.model.mu[i, :, :].detach()
            # Rotated image latent code
            Z2 = self.model.mu[i + 1, :, :].detach()
            M = torch.linalg.lstsq(Z1, Z2).solution
            R = roma.special_procrustes(M)
            error = (torch.norm(Z1 @ R - Z2) / torch.norm(Z2)).item()
            errors.append(error)
        total_error = sum(errors) / len(errors)
        return total_error, errors

    def latent_per_param_adjustments(self, frames_per_dim=10, fname="per_param"):
        frames = []

        print("rendering frames")

        directions, _, _, _, _ = self.dataset[0]
        directions = directions.to(device=self.device)

        Z = torch.randn((1, self.ndims, 3), device=self.device)
        Z = f.normalize(Z, p=2, dim=2)

        font = ImageFont.truetype(
            os.getcwd() + os.sep + ".." + os.sep + "data/other/DejaVuMathTeXGyre.ttf",
            15,
        )
        ratio = frames_per_dim / 14

        for i in range(self.ndims):
            for j in range(3):
                n = 0
                for k in torch.arange(0, 5.0, 5.0 / frames_per_dim):
                    caption = Image.new("RGB", (380, 50), "white")
                    draw = ImageDraw.Draw(caption)
                    draw.text(
                        (190, 25),
                        f"Component[{i},{j}]",
                        "black",
                        font=font,
                        anchor="mm",
                    )
                    draw.text(
                        (260, 25),
                        f"{int(n/ratio)*'-'}> {k:.1f}",
                        "black",
                        font=font,
                        anchor="lm",
                    )
                    Z = torch.zeros((1, self.ndims, 3), device=self.device)
                    Z[:, i, j] = k
                    model_output = self.get_model_output(None, 0, Z)
                    model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                    image_caption_resize = caption.resize(
                        (
                            model_output_np.shape[1],
                            int(
                                model_output_np.shape[1]
                                * caption.size[1]
                                / caption.size[0]
                            ),
                        ),
                        resample=Image.BILINEAR,
                    )
                    image = np.concatenate(
                        (model_output_np, image_caption_resize), axis=0
                    )
                    image_int = (image * 255).astype(np.uint8)
                    frames.append(image_int)
                    n += 1

                n = frames_per_dim
                for k in torch.arange(5.0, 0.0, -5.0 / frames_per_dim):
                    caption = Image.new("RGB", (380, 50), "white")
                    draw = ImageDraw.Draw(caption)
                    draw.text(
                        (190, 25),
                        f"Component[{i},{j}]",
                        "black",
                        font=font,
                        anchor="mm",
                    )
                    draw.text(
                        (260, 25),
                        f"{int(n/ratio)*'-'}> {k:.1f}",
                        "black",
                        font=font,
                        anchor="lm",
                    )
                    Z = torch.zeros((1, self.ndims, 3), device=self.device)
                    Z[:, i, j] = k
                    model_output = self.get_model_output(None, 0, Z)
                    model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                    image_caption_resize = caption.resize(
                        (
                            model_output_np.shape[1],
                            int(
                                model_output_np.shape[1]
                                * caption.size[1]
                                / caption.size[0]
                            ),
                        ),
                        resample=Image.BILINEAR,
                    )
                    image = np.concatenate(
                        (model_output_np, image_caption_resize), axis=0
                    )
                    image_int = (image * 255).astype(np.uint8)
                    frames.append(image_int)
                    n -= 1

                n = 0
                for k in torch.arange(0.0, -5.0, -5.0 / frames_per_dim):
                    caption = Image.new("RGB", (380, 50), "white")
                    draw = ImageDraw.Draw(caption)
                    draw.text(
                        (190, 25),
                        f"Component[{i},{j}]",
                        "black",
                        font=font,
                        anchor="mm",
                    )
                    draw.text(
                        (120, 25),
                        f"{k:.1f} <{int(n/ratio)*'-'}",
                        "black",
                        font=font,
                        anchor="rm",
                    )
                    Z = torch.zeros((1, self.ndims, 3), device=self.device)
                    Z[:, i, j] = k
                    model_output = self.get_model_output(None, 0, Z)
                    model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                    image_caption_resize = caption.resize(
                        (
                            model_output_np.shape[1],
                            int(
                                model_output_np.shape[1]
                                * caption.size[1]
                                / caption.size[0]
                            ),
                        ),
                        resample=Image.BILINEAR,
                    )
                    image = np.concatenate(
                        (model_output_np, image_caption_resize), axis=0
                    )
                    image_int = (image * 255).astype(np.uint8)
                    frames.append(image_int)
                    n += 1

                n = frames_per_dim
                for k in torch.arange(-5.0, 0.0, 5.0 / frames_per_dim):
                    caption = Image.new("RGB", (380, 50), "white")
                    draw = ImageDraw.Draw(caption)
                    draw.text(
                        (190, 25),
                        f"Component[{i},{j}]",
                        "black",
                        font=font,
                        anchor="mm",
                    )
                    draw.text(
                        (120, 25),
                        f"{k:.1f} <{int(n/ratio)*'-'}",
                        "black",
                        font=font,
                        anchor="rm",
                    )
                    Z = torch.zeros((1, self.ndims, 3), device=self.device)
                    Z[:, i, j] = k
                    model_output = self.get_model_output(None, 0, Z)
                    model_output_np = self.dataset.to_numpy_convert_sRGB(model_output)
                    image_caption_resize = caption.resize(
                        (
                            model_output_np.shape[1],
                            int(
                                model_output_np.shape[1]
                                * caption.size[1]
                                / caption.size[0]
                            ),
                        ),
                        resample=Image.BILINEAR,
                    )
                    image = np.concatenate(
                        (model_output_np, image_caption_resize), axis=0
                    )
                    image_int = (image * 255).astype(np.uint8)
                    frames.append(image_int)
                    n -= 1

                    del model_output

        print("saving gif")
        imageio.mimsave(fname + ".gif", frames, fps=24)

        out = cv2.VideoWriter(
            fname + ".avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            24,
            (image_int.shape[1], image_int.shape[0]),
        )
        for i in range(len(frames)):
            out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        out.release()
