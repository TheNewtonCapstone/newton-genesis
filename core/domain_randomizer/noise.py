import torch


def gaussian_noise(x, std = 1.0):
    """Adds Gaussian noise to the input."""
    return x + torch.randn_like(x) * std