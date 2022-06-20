# %%
def run_from_ipython():
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir_name + "/")

    if run_from_ipython():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import torch
import random
import numpy as np
import run_setup as setup
import training_functions as training_functions
from torchvision import transforms
from data.hdri_dataset import HDRIDataset
from PIL import Image
from torchvision.transforms import ToTensor
from utils.pytorch3d_envmap_shader import build_renderer

import utils.utils as utils
import yaml

# ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb

os.environ["WANDB_NOTEBOOK_NAME"] = __file__
wandb.require(experiment="service")


def run_latent_training(
    config,
    run,
    do_log,
    world_size,
    rank,
    model,
    training_type,
    dataset,
    device,
):

    if model is None:  # We are starting from a previously saved model
        prev_run_config_path = config.previous_run_path + "files/config.yaml"
        with open(prev_run_config_path) as yamlfile:
            prev_run_config = yaml.safe_load(yamlfile)
            prev_run_config = utils.dotdict(prev_run_config)
            prev_dataset_path = prev_run_config.dataset_path.value
            model_type = prev_run_config.model_type.value
            ndims = prev_run_config.ndims.value
            RENI_hidden_features = prev_run_config.RENI_hidden_features.value
            RENI_hidden_layers = prev_run_config.RENI_hidden_layers.value
            prev_model_path = config.previous_run_path + "files/RENI.pt"
            state_dict = torch.load(prev_model_path)
            last_layer_linear = prev_run_config.last_layer_linear.value
    else:  # We have just finished training a model and are now testing it
        model_type = config.model_type
        ndims = config.ndims
        RENI_hidden_features = config.RENI_hidden_features
        RENI_hidden_layers = config.RENI_hidden_layers
        last_layer_linear = config.last_layer_linear
        state_dict = model.module.state_dict()

    # Get size of input layer for previous model, depends
    # on the size of dataset it was trained on
    if "mu" in state_dict:
        prev_dataset_len = state_dict["mu"].shape[0]
    else:
        prev_dataset_len = state_dict["Z"].shape[0]

    # Need to get the min and max of the dataset the model was trained on so that the scaling is the same
    if dataset is None:
        transform = transforms.Resize(
            (config.resize_transform_end // 2, config.resize_transform_end)
        )
        prev_dataset_path = parent_dir_name + "/../" + prev_dataset_path
        dataset = HDRIDataset(prev_dataset_path, transform)

    minmax = (dataset.min_value, dataset.max_value)

    if training_type == "Inverse":
        obj_path = parent_dir_name + "/../" + config.object_path
        kd_value = config.kd_value
        render_resolution = config.render_resolution
        lr_schedule = config.lr_inverse_schedule
        batch_size = 1
        epochs = config.epochs_inverse

        if config.inverse_multi_res_training:
            image_size = config.inverse_resize_transform_start
        else:
            image_size = config.inverse_resize_transform_end

        lr_start = config.lr_inverse_start
        lr_end = config.lr_inverse_end

        renderer, R, T, mesh = build_renderer(
            obj_path, 0, render_resolution, kd_value, device
        )
    else:
        if config.latent_multi_res_training:
            image_size = config.resize_transform_start
        else:
            image_size = config.resize_transform_end
        lr_start = config.lr_latent_start
        lr_end = config.lr_latent_end
        lr_schedule = config.lr_latent_schedule
        batch_size = config.latent_batch_size
        epochs = config.epochs_latent

    # setup test dataset
    dataset_path = parent_dir_name + "/../" + config.test_dataset_path
    loader, dataset = setup.build_dataset(
        dataset_path, image_size, batch_size, world_size, rank, minmax
    )

    # If a mask is being applied to the dataset at test time for
    # the environment completion task
    mask = None
    mask_path = None
    if config.apply_mask:
        mask_path = parent_dir_name + "/../" + config.mask_path
        mask = Image.open(mask_path)
        mask = ToTensor()(mask)

    # Build the model
    model = setup.build_network(
        model_type,
        ndims,
        prev_dataset_len,
        RENI_hidden_features,
        RENI_hidden_layers,
        last_layer_linear,
        device,
    )

    # We load the optimised RENI network
    model.load_state_dict(state_dict)
    # And now make new latent codes for the unseen test set images
    # These are initalised at the mean environment map
    model = setup.reinit_latent_codes(model, model_type, len(dataset), ndims)
    model = model.to(device=device)

    # Wrap the model for multi gpu training
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    optimizer = setup.build_optimizer(
        model.module,
        model_type,
        config.optimizer,
        lr_start,
        latent_code_optimisation=True,
    )

    scheduler = None
    if lr_schedule == "exponential":
        gamma = np.exp(np.log(lr_end / lr_start) / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(0, epochs):
        if training_type == "Latent":
            if (
                epoch > 0
                and config.latent_multi_res_training == True
                and not epoch % config.latent_epochs_till_res_change
            ):
                loader.dataset.double_resolution()

            (
                epoch_loss,
                epoch_mse_loss,
                _,
                epoch_prior_loss,
                epoch_cosine_loss,
            ) = training_functions.fit_epoch(
                config,
                model_type,
                model,
                loader,
                optimizer,
                scheduler,
                device,
                config.kld_weighting,
                True,
                mask,
            )
        if training_type == "Inverse":
            if (
                epoch > 0
                and config.inverse_multi_res_training == True
                and not epoch % config.inverse_epochs_till_res_change
            ):
                loader.dataset.double_resolution()

            (
                epoch_loss,
                epoch_mse_loss,
                epoch_prior_loss,
                epoch_cosine_loss,
            ) = training_functions.fit_inverse_epoch(
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
            )

        # If enough epochs have past then display images showing progress
        if not epoch % config.epochs_till_images:
            if training_type == "Latent":
                example_images = training_functions.test(
                    model_type, model, dataset, device, mask_path
                )
            else:
                example_images = training_functions.test_inverse(
                  model_type, model, dataset, renderer, mesh, R, T, device
                )
            if do_log:
                run.log(
                    {
                        "latent_loss": epoch_loss * world_size,
                        "latent_mse_loss": epoch_mse_loss * world_size,
                        "latent_prior_loss": epoch_prior_loss * world_size,
                        "latent_cosine_loss": epoch_cosine_loss * world_size,
                        "examples": example_images,
                    }
                )
        else:
            if do_log:
                run.log(
                    {
                        "latent_loss": epoch_loss * world_size,
                        "latent_mse_loss": epoch_mse_loss * world_size,
                        "latent_prior_loss": epoch_prior_loss * world_size,
                        "latent_cosine_loss": epoch_cosine_loss * world_size,
                    }
                )

    if do_log:
        torch.save(
            model.module.state_dict(),
            os.path.join(run.dir, "RENI_" + training_type + ".pt"),
        )


def run_training(config, run, do_log, world_size, rank, device):
    is_master = rank == 0
    model_type = config.model_type
    ndims = config.ndims
    RENI_hidden_features = config.RENI_hidden_features
    RENI_hidden_layers = config.RENI_hidden_layers
    last_layer_linear = config.last_layer_linear

    if config.multi_res_training:
        image_size = config.resize_transform_start
    else:
        image_size = config.resize_transform_end

    dataset_path = parent_dir_name + "/../" + config.dataset_path

    loader, dataset = setup.build_dataset(
        dataset_path,
        image_size,
        config.train_batch_size,
        world_size,
        rank,
    )
    dataset_len = len(dataset)

    model = setup.build_network(
        model_type,
        ndims,
        dataset_len,
        RENI_hidden_features,
        RENI_hidden_layers,
        last_layer_linear,
        device,
    )

    # Wrap the model
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    lr_train_start = config.lr_train_start
    lr_train_end = config.lr_train_end

    optimizer = setup.build_optimizer(
        model,
        model_type,
        config.optimizer,
        lr_train_start,
    )

    scheduler = None
    if config.lr_schedule == "exponential":
        gamma = np.exp(np.log(lr_train_end / lr_train_start) / config.epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Only log on rank 0
    if is_master:
        run.watch(model, log="all")
    if do_log:
        path = run.dir + "/checkpoints"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    for epoch in range(0, config.epochs):
        if (
            epoch > 0
            and config.multi_res_training == True
            and not epoch % config.epochs_till_res_change
        ):
            loader.dataset.double_resolution()

        if epoch > 0 and not epoch % config.epochs_till_checkpoint:
            if do_log:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(
                        run.dir, "checkpoints", "RENI_Epoch" + str(epoch) + ".pt"
                    ),
                )

        (
            epoch_loss,
            epoch_mse_loss,
            epoch_kld_loss,
            _,
            _,
        ) = training_functions.fit_epoch(
            config,
            model_type,
            model,
            loader,
            optimizer,
            scheduler,
            device,
            config.kld_weighting,
            False,
        )

        # loss is dependant on world size so need to scale, as wandb is only
        # tracking the loss of the process on rank 0, so just multiplying
        # the loss of rank 0 by the number of ranks to enable comparisons with
        # runs on more or less GPUs.
        if not epoch % config.epochs_till_images:
            example_images = training_functions.test(model_type, model, dataset, device)
            if do_log:
                if model_type == "RENIVariationalAutoDecoder":
                    run.log(
                        {
                            "loss": epoch_loss * world_size,
                            "mse_loss": epoch_mse_loss * world_size,
                            "kld_loss": epoch_kld_loss * world_size,
                            "examples": example_images,
                        }
                    )
                elif model_type == "RENIAutoDecoder":
                    run.log(
                        {
                            "loss": epoch_loss * world_size,
                            "examples": example_images,
                        }
                    )
        else:
            if do_log:
                if model_type == "RENIVariationalAutoDecoder":
                    run.log(
                        {
                            "loss": epoch_loss * world_size,
                            "mse_loss": epoch_mse_loss * world_size,
                            "kld_loss": epoch_kld_loss * world_size,
                        }
                    )
                elif model_type == "RENIAutoDecoder":
                    run.log(
                        {
                            "loss": epoch_loss * world_size,
                        }
                    )

    # Training is finished so save the model
    if do_log:
        torch.save(model.module.state_dict(), os.path.join(run.dir, "RENI.pt"))

    # if specified in the config now run latent code optimisation to test performance on test set
    if config.latent_code_optimisation and config.previous_run_path == "None":
        del loader
        run_latent_training(
            config, run, do_log, world_size, rank, model, "Latent", dataset, device
        )

    # if specified now run inverse render training to test performance on inverse task
    if config.inverse_rendering_task and config.previous_run_path == "None":
        if loader:
            del loader
        run_latent_training(
            config, run, do_log, world_size, rank, model, "Inverse", dataset, device
        )


def setup_run(run, rank, config, world_size, is_sweep):
    do_log = run is not None

    if not is_sweep:
        config = utils.dotdict(config)

    total_devices = torch.cuda.device_count()

    device = torch.device(rank % total_devices)

    torch.cuda.set_device(device)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # If this run is optimising the latent codes from a previous run then set that up
    if config.latent_code_optimisation and config.previous_run_path != "None":
        run_latent_training(
            config, run, do_log, world_size, rank, None, "Latent", None, device
        )
    # Same again but if for the inverse rendering task
    elif config.inverse_rendering_task and config.previous_run_path != "None":
        run_latent_training(
            config, run, do_log, world_size, rank, None, "Inverse", None, device
        )
    # Otherwise we are fitting to the Training set
    else:
        run_training(config, run, do_log, world_size, rank, device)

    utils.mp_cleanup(rank)


if __name__ == "__main__":

    args = utils.parse_args()

    config = None

    if args.sweep is None:
        config_path = parent_dir_name + "/../training_configs/config_main.yaml"
        latent_config_path = parent_dir_name + "/../training_configs/config_latent.yaml"
        inverse_config_path = parent_dir_name + "/../training_configs/config_inverse.yaml"
        with open(config_path, "r") as yamlfile:
            config = yaml.safe_load(yamlfile)
        if config['latent_code_optimisation']:
            with open(latent_config_path, "r") as yamlfile:
                config_latent = yaml.safe_load(yamlfile)
                config.update(config_latent)
        if config['inverse_rendering_task']:
            with open(inverse_config_path, "r") as yamlfile:
                config_inverse = yaml.safe_load(yamlfile)
                config.update(config_inverse)

    local_rank = int(os.environ["LOCAL_RANK"])
    run, config = utils.mp_setup(local_rank, args.world_size, config, args.sweep)
    setup_run(run, local_rank, config, args.world_size, args.sweep)

wandb.finish()
