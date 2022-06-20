import torch


def InvariantRepresentation(Z, D):
    # Derives a representation from latent code Z and direction coordinates D that is
    # invariant under y-axis rotation of Z and D simultaneously
    # B = Batchsize
    # Z is B x ndims x 3
    # D is B x npix x 3
    Z_xz = torch.stack((Z[:, :, 0], Z[:, :, 2]), -1)
    D_xz = torch.stack((D[:, :, 0], D[:, :, 2]), -1)
    # Invariant representation of Z, gram matrix G=Z*Z' is size B x ndims x ndims
    G = torch.bmm(Z_xz, torch.transpose(Z_xz, 1, 2))
    # Flatten G and replicate for all pixels, giving size B x npix x ndims^2
    Z_xz_invar = G.flatten(start_dim=1).unsqueeze(1).repeat(1, D.shape[1], 1)
    # innerprod is size B x npix x ndims
    innerprod = torch.bmm(D_xz, torch.transpose(Z_xz, 1, 2))
    D_xz_norm = torch.sqrt(D[:, :, 0] ** 2 + D[:, :, 2] ** 2).unsqueeze(2)
    # Copy Z_y for every pixel to be size B x npix x ndims
    Z_y = Z[:, :, 1].unsqueeze(1).repeat(1, innerprod.shape[1], 1)
    # Just the y component of D (B x npix x 1)
    D_y = D[:, :, 1].unsqueeze(2)
    # Conditioning via concatenation
    model_input = torch.cat((innerprod, Z_xz_invar, D_xz_norm, Z_y, D_y), 2)
    # model_input is size B x npix x 2 x ndims + ndims^2 + 2
    return model_input


def reparameterise(mu, log_var):
    """
    :param mu: mean from the encoder's latent space
    :param log_var: log variance from the encoder's latent space
    """
    std = torch.exp(0.5 * log_var)  # standard deviation
    eps = torch.randn_like(std)  # `randn_like` as we need the same size
    sample = mu + (eps * std)  # sampling as if coming from the input space
    return sample
