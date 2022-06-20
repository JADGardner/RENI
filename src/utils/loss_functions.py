import torch
import torch.nn.functional as F


def WeightedMSE(model_output, ground_truth, sineweight):
    MSE = (((model_output - ground_truth) ** 2) * sineweight).view(model_output.shape[0], -1).mean(1).sum(0)
    return MSE


def KLD(mu, log_var, Z_dims=1):
    kld = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).view(mu.shape[0], -1)).sum(1)
    kld /= Z_dims
    kld = kld.sum(0)
    return kld


def WeightedCosineSimilarity(model_output, ground_truth, sineweight):
    return (1 - (F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20) * sineweight[:, 0]).mean(1)).sum(0)


def CosineSimilarity(model_output, ground_truth):
    return 1 - F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20).mean()
