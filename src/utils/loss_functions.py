import torch
import torch.nn as nn
import torch.nn.functional as F


def WeightedMSE(model_output, ground_truth, sineweight):
    MSE = (
        (((model_output - ground_truth) ** 2) * sineweight)
        .view(model_output.shape[0], -1)
        .mean(1)
        .sum(0)
    )
    return MSE


def KLD(mu, log_var, Z_dims=1):
    kld = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).view(mu.shape[0], -1)).sum(
        1
    )
    kld /= Z_dims
    kld = kld.sum(0)
    return kld


def WeightedCosineSimilarity(model_output, ground_truth, sineweight):
    return (
        1
        - (
            F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20)
            * sineweight[:, 0]
        ).mean(1)
    ).sum(0)


def CosineSimilarity(model_output, ground_truth):
    return 1 - F.cosine_similarity(model_output, ground_truth, dim=1, eps=1e-20).mean()


class RENITrainLoss(object):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs, targets, sineweight):
        loss = WeightedMSE(inputs, targets, sineweight)
        return loss

class RENIVADTrainLoss(object):
    def __init__(self, beta=1, Z_dims=None):
        super().__init__()
        self.beta = beta
        self.Z_dims = Z_dims

    def __call__(self, inputs, targets, sineweight, mu, log_var):
        mse_loss = WeightedMSE(inputs, targets, sineweight)
        kld_loss = self.beta * KLD(mu, log_var, self.Z_dims)
        loss = mse_loss + kld_loss

        return loss, mse_loss, kld_loss

class RENITestLoss(object):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def __call__(self, inputs, targets, sineweight, Z):
        mse_loss = WeightedMSE(inputs, targets, sineweight)
        prior_loss = self.alpha * torch.pow(Z, 2).sum()
        cosine_loss = self.beta * WeightedCosineSimilarity(inputs, targets, sineweight)
        loss = mse_loss + prior_loss + cosine_loss
        return loss, mse_loss, prior_loss, cosine_loss

class RENITestLossInverse(object):
    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = torch.nn.MSELoss(reduction="mean")

    def __call__(self, inputs, targets, Z):
        mse_loss = self.mse(inputs, targets)
        prior_loss = self.alpha * torch.pow(Z, 2).sum()
        cosine_loss = self.beta * CosineSimilarity(inputs, targets)
        loss = mse_loss + prior_loss + cosine_loss
        return loss, mse_loss, prior_loss, cosine_loss
