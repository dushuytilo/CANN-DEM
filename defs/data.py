"""Data generation and normalization for CANN training
The CANN is trained in invariant space:
    input  = [I1, J]
    target = Psi(I1, J)
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from .material import neo_hookean_energy


@dataclass
class NormalizationStats:
    """Mean and standard deviation used to normalize CANN data."""

    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def generate_cann_data(
    n_samples,
    min_I1,
    max_I1,
    min_J,
    max_J,
    mu,
    lam,
    device,
    dtype,
):
    """Generate synthetic CANN input/target pairs"""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if min_J <= 0 or max_J <= 0:
        raise ValueError("J must be strictly positive because log(J) is used")

    I1_samples = torch.linspace(
        min_I1,
        max_I1,
        n_samples,
        device=device,
        dtype=dtype,
    )
    J_samples = torch.linspace(
        min_J,
        max_J,
        n_samples,
        device=device,
        dtype=dtype,
    )

    I1, J = torch.meshgrid(I1_samples, J_samples, indexing="ij")

    I1 = I1.flatten()
    J = J.flatten()

    x = torch.stack((I1, J), dim=1)
    y = neo_hookean_energy(I1, J, mu, lam)

    return x, y


def compute_normalization_stats(x, y):
    """Compute normalization statistics from the training dataset"""
    return NormalizationStats(
        x_mean=x.mean(dim=0),
        x_std=x.std(dim=0),
        y_mean=y.mean(),
        y_std=y.std(),
    )


def normalize_data(x, y, stats):
    """Normalize CANN inputs and targets using training statistics"""
    x_normalized = (x - stats.x_mean) / stats.x_std
    y_normalized = (y - stats.y_mean) / stats.y_std

    return x_normalized, y_normalized


def denormalize_energy(y_normalized, stats):
    """Convert normalized CANN energy predictions back to physical units"""
    return y_normalized * stats.y_std + stats.y_mean


def create_dataloader(
    x,
    y,
    batch_size,
    shuffle,
    drop_last,
):
    """Create a PyTorch DataLoader for CANN training"""
    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )