import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import random
import utils.model_utils as model_utils
import utils.loss_functions as loss_functions
import utils.utils as utils

# ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

from utils.pytorch3d_envmap_shader import (
    EnvironmentMap,
)

import wandb


def fit_epoch(
    config,
    model_type,
    model,
    loader,
    optimizer,
    scheduler,
    device,
    kld_weight,
    latent_code_optimisation=False,
    mask=None,
):

    if not latent_code_optimisation:
        model.train()

    epoch_MSE_loss = 0
    epoch_kld_loss = 0
    epoch_loss = 0
    epoch_cosine_loss = 0
    epoch_prior_loss = 0

    if mask is not None:
        resize = transforms.Resize(
            loader.dataset.transform.size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask = resize(mask)
        mask = mask.permute((1, 2, 0))  # (3, H, W) -> (H, W, 3)
        mask = mask.view(-1, 3)  # (H, W, 3) -> (H*W, 3)
        mask = mask.unsqueeze(0)  # (1, HW, 3)
        mask = mask.repeat((config.latent_batch_size, 1, 1))  # (B, HW, 3)
        mask = mask.to(device=device)

    for _, (directions, sineweight, ground_truth, idx, _) in enumerate(loader):
        directions = directions.to(device=device)
        sineweight = sineweight.to(device=device)
        if mask is not None:
            sineweight = sineweight * mask
        ground_truth = ground_truth.to(device=device)

        if model_type == "RENIVariationalAutoDecoder":
            if latent_code_optimisation:
                Z = model.module.mu[idx, :, :]
            else:
                Z = model_utils.reparameterise(
                    model.module.mu[idx, :, :], model.module.log_var[idx, :, :]
                )
            model_input = model_utils.InvariantRepresentation(Z, directions)
            model_input = model_input.to(device=device)
            # Make predictions
            model_output = model(model_input)

            # Compute the loss
            MSE = loss_functions.WeightedMSE(model_output, ground_truth, sineweight)

            if latent_code_optimisation:
                prior_loss = config.latent_prior_loss_weight * torch.pow(Z, 2).sum()
                cosine_weight = config.latent_cosine_similarity_weight
                cosine_loss = cosine_weight * loss_functions.WeightedCosineSimilarity(
                    model_output, ground_truth, sineweight
                )
                loss = MSE + prior_loss + cosine_loss
                epoch_prior_loss += float(prior_loss)
                epoch_cosine_loss += float(cosine_loss)
            else:
                kld = (
                    loss_functions.KLD(
                        model.module.mu[idx, :, :],
                        model.module.log_var[idx, :, :],
                        Z.shape[1] * Z.shape[2]
                    )
                ) * kld_weight
                loss = MSE + kld
                epoch_kld_loss += float(kld)

            epoch_MSE_loss += float(MSE)

            del model_input

        elif model_type == "RENIAutoDecoder":
            Z = model.module.Z[idx, :, :]
            model_input = model_utils.InvariantRepresentation(Z, directions)
            model_input = model_input.to(device=device)
            # Make predictions
            model_output = model(model_input)

            # Compute the loss
            MSE = loss_functions.WeightedMSE(model_output, ground_truth, sineweight)

            if latent_code_optimisation:
                prior_loss = config.latent_prior_loss_weight * torch.pow(Z, 2).sum()
                cosine_weight = config.latent_cosine_similarity_weight
                cosine_loss = cosine_weight * loss_functions.WeightedCosineSimilarity(
                    model_output, ground_truth, sineweight
                )
                loss = MSE + prior_loss + cosine_loss
                epoch_prior_loss += float(prior_loss)
                epoch_cosine_loss += float(cosine_loss)
            else:
                loss = MSE

            epoch_MSE_loss += float(MSE)

            del model_input

        epoch_loss += float(loss)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        del directions
        del sineweight
        del ground_truth

    if scheduler is not None:
        scheduler.step()

    # getting average epoch loss from dataset
    epoch_MSE_loss = epoch_MSE_loss / len(loader.dataset)
    epoch_kld_loss = epoch_kld_loss / len(loader.dataset)
    epoch_prior_loss = epoch_prior_loss / len(loader.dataset)
    epoch_cosine_loss = epoch_cosine_loss / len(loader.dataset)
    epoch_loss = epoch_loss / len(loader.dataset)

    return (
        epoch_loss,
        epoch_MSE_loss,
        epoch_kld_loss,
        epoch_prior_loss,
        epoch_cosine_loss,
    )

def test(model_type, model, dataset, device, mask_path=None):
    model.eval()

    example_images = []

    if mask_path is not None:
        mask = Image.open(mask_path)
        mask = ToTensor()(mask)
        resize = transforms.Resize(
            dataset.transform.size, interpolation=transforms.InterpolationMode.NEAREST
        )
        mask = resize(mask)
        mask = mask.permute((1, 2, 0))  # (3, H, W) -> (H, W, 3)
        mask_np = mask.cpu().detach().numpy()

    with torch.no_grad():
        for idx in random.sample(range(len(dataset)), 1):
            # Get ground truth sample
            directions, _, ground_truth, _, _ = dataset[idx]
            directions = directions.to(device=device)
            ground_truth = ground_truth.to(device=device)

            ground_truth_np = dataset.to_numpy_convert_sRGB(ground_truth)
            example_images.append(wandb.Image(ground_truth_np))
            if mask_path is not None:
                masked_ground_truth = ground_truth_np * mask_np
                example_images.append(wandb.Image(masked_ground_truth))

            # Get model output
            if model_type == "RENIVariationalAutoDecoder":
                Z = model.module.mu[idx, :, :]
                model_input = model_utils.InvariantRepresentation(
                    Z.unsqueeze(0), directions.unsqueeze(0)
                )
                model_input = model_input.to(device=device)
                # Make predictions
                model_output = model(model_input)

            elif model_type == "RENIAutoDecoder":
                Z = model.module.Z[idx, :, :]
                model_input = model_utils.InvariantRepresentation(
                    Z.unsqueeze(0), directions.unsqueeze(0)
                )
                model_input = model_input.to(device=device)
                # Make predictions
                model_output = model(model_input)

            model_output = dataset.to_numpy_convert_sRGB(model_output)
            example_images.append(wandb.Image(model_output))

    return example_images


def fit_inverse_epoch(
    config,
    model_type,
    model,
    loader,
    optimizer,
    scheduler,
    renderer,
    mesh,
    R,
    T,
    device,
):

    model.eval()

    epoch_MSE_loss = 0
    epoch_loss = 0
    epoch_cosine_loss = 0
    epoch_prior_loss = 0

    MSE = torch.nn.MSELoss(reduction="mean")

    for _, (directions, sineweight, ground_truth, idx, _) in enumerate(loader):
        directions = directions.to(device=device)
        sineweight = sineweight.to(device=device)
        ground_truth = ground_truth.to(device=device)
        sineweight_sq = sineweight.detach().clone()
        sineweight_sq = sineweight_sq.to(device=device).squeeze(0)

        if model_type == "RENIVariationalAutoDecoder":
            Z = model.module.mu[idx, :, :]
        elif model_type == "RENIAutoDecoder":
            Z = model.module.Z[idx, :, :]
        
        model_input = model_utils.InvariantRepresentation(Z, directions)

        model_input = model_input.to(device=device)
        # Make predictions
        model_output = model(model_input)

        envmap_model = EnvironmentMap(
            environment_map=model_output,
            directions=directions,
            sineweight=sineweight_sq,
        )
        envmap_gt = EnvironmentMap(
            environment_map=ground_truth,
            directions=directions,
            sineweight=sineweight_sq,
        )
        render_model, _ = renderer(meshes_world=mesh, R=R, T=T, envmap=envmap_model)
        render_gt, _ = renderer(meshes_world=mesh, R=R, T=T, envmap=envmap_gt)
        render_model = render_model[..., :3]  # Don't need alpha channel
        render_gt = render_gt[..., :3]  # Don't need alpha channel

        # Compute the loss between renders
        mse = MSE(render_model, render_gt)
        prior_loss = config.inverse_prior_loss_weight * torch.pow(Z, 2).sum()
        cosine_weight = config.inverse_cosine_similarity_weight
        cosine_loss = cosine_weight * loss_functions.CosineSimilarity(
            render_model, render_gt
        )
        loss = mse + prior_loss + cosine_loss

        epoch_MSE_loss += float(mse)
        epoch_prior_loss += float(prior_loss)
        epoch_cosine_loss += float(cosine_loss)
        epoch_loss += float(loss)

        del model_input

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        del directions
        del sineweight
        del sineweight_sq
        del ground_truth

    del R
    del T

    if scheduler is not None:
        scheduler.step()

    # getting average epoch loss from dataset
    epoch_MSE_loss = epoch_MSE_loss / len(loader.dataset)
    epoch_prior_loss = epoch_prior_loss / len(loader.dataset)
    epoch_cosine_loss = epoch_cosine_loss / len(loader.dataset)
    epoch_loss = epoch_loss / len(loader.dataset)

    return epoch_loss, epoch_MSE_loss, epoch_prior_loss, epoch_cosine_loss

    

def test_inverse(model_type, model, dataset, renderer, mesh, R, T, device):
    model.eval()

    example_images = []

    with torch.no_grad():
        for idx in random.sample(range(len(dataset)), 1):
            # Get ground truth sample
            directions, sineweight, ground_truth, _, _ = dataset[idx]
            directions = directions.to(device=device)
            sineweight = sineweight.to(device=device).squeeze(0)
            ground_truth = ground_truth.to(device=device)

            ground_truth_np = dataset.to_numpy_convert_sRGB(ground_truth)

            # WandB – Log images
            example_images.append(wandb.Image(ground_truth_np))

            # Get model output
            if model_type == "RENIVariationalAutoDecoder":
                Z = model.module.mu[idx, :, :]
            elif model_type == "RENIAutoDecoder":
                Z = model.module.Z[idx, :, :]

            model_input = model_utils.InvariantRepresentation(Z.unsqueeze(0), directions.unsqueeze(0))
            model_input = model_input.to(device=device)
            # Make predictions
            model_output = model(model_input)

            envmap_model = EnvironmentMap(
                environment_map=model_output,
                directions=directions,
                sineweight=sineweight,
            )
            envmap_gt = EnvironmentMap(
                environment_map=ground_truth,
                directions=directions,
                sineweight=sineweight,
            )

            render_model, _ = renderer(meshes_world=mesh, R=R, T=T, envmap=envmap_model)
            render_gt, normal_map = renderer(meshes_world=mesh, R=R, T=T, envmap=envmap_gt)
            render_model = render_model[..., :3]  # Don't need alpha channel
            render_gt = render_gt[..., :3]  # Don't need alpha channel
            render_model_np = utils.sRGB(render_model).cpu().detach().numpy()
            render_gt_np = utils.sRGB(render_gt).cpu().detach().numpy()

            model_output = dataset.to_numpy_convert_sRGB(model_output)
            example_images.append(wandb.Image(model_output))
            example_images.append(wandb.Image(render_gt_np))
            example_images.append(wandb.Image(render_model_np))

            del directions
            del sineweight
            del ground_truth
            del R
            del T

    return example_images
