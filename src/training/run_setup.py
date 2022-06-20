import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.hdri_dataset import HDRIDataset
import models.RENI as RENI


def build_dataset(dataset_path, sidelen, batch_size, world_size, rank, minmax=None):
    # define the tranformations to apply to our images
    transform = transforms.Resize((sidelen // 2, sidelen))

    dataset = HDRIDataset(dataset_path, transform, minmax)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )

    return loader, dataset


def build_network(
    model_type,
    ndims,
    dataset_size,
    hidden_features,
    hidden_layers,
    last_layer_linear,
    device,
):

    if model_type == "RENIVariationalAutoDecoder":
        model = RENI.RENIVariationalAutoDecoder(
            in_features=2 * ndims + ndims * ndims + 2,
            out_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            dataset_size=dataset_size,
            ndims=ndims,
            outermost_linear=last_layer_linear,
        )
    elif model_type == "RENIAutoDecoder":
        model = RENI.RENIAutoDecoder(
            in_features=2 * ndims + ndims * ndims + 2,
            out_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            dataset_size=dataset_size,
            ndims=ndims,
            outermost_linear=last_layer_linear,
        )

    model = model.to(device=device)

    return model


def reinit_latent_codes(model, model_type, dataset_size, ndims):
    if model_type == "RENIVariationalAutoDecoder":
        # Randomly re-initialise latent distributions for latent training
        model.mu = torch.nn.Parameter(torch.zeros((dataset_size, ndims, 3)))
        model.log_var = None
        model.net.eval()
    elif model_type == "RENIAutoDecoder":
        model.Z = torch.nn.Parameter(torch.randn((dataset_size, ndims, 3)))
        model.net.eval()

    return model


def build_optimizer(
    model,
    model_type,
    optimizer_type,
    learning_rate,
    latent_code_optimisation=False,
):
    if latent_code_optimisation:
        if model_type == "RENIVariationalAutoDecoder":
            parameters = [model.mu]
        elif model_type == "RENIAutoDecoder":
            parameters = [model.Z]
    else:
        parameters = model.parameters()

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_type == "adagrad":
        optimizer = torch.optim.adagrad(parameters, lr=learning_rate)

    return optimizer
