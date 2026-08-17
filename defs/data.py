"""Data generation and normalization for CANN training

The CANN is trained in invariant space:
    input  = [I1, J]
    target = Psi(I1, J)

Psi is generated from the analytical compressible Neo-Hookean model defined in material.py
"""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from .material import neo_hookean_energy

# Sampling ranges used in Homework 2
I1_MIN = 0.08
I1_MAX = 200.0
J_MIN = 0.04
J_MAX = 100.0

@dataclass
class NormalizationStats:
    """Mean and standard deviation used to normalize CANN data"""
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor


def generate_cann_data(
    n_samples,
    min_I1=I1_MIN,
    max_I1=I1_MAX,
    min_J=J_MIN,
    max_J=J_MAX,
    *,
    device=None,
    dtype=None,
):
    """Generate synthetic CANN input/target pairs
    Uniformly samples I1 and J on a tensor-product grid and evaluates
    the analytical Neo-Hookean Helmholtz free-energy density

    Parameters
    n_samples : int
        Number of sample points per invariant dimension. The resulting
        dataset contains n_samples**2 input/target pairs
    min_I1, max_I1 : float, optional
        Sampling limits for the first invariant I1
    min_J, max_J : float, optional
        Sampling limits for the Jacobian J. J must remain positive
    device : torch.device or str, optional
        Device on which to create the tensors
    dtype : torch.dtype, optional
        Tensor dtype. If omitted, PyTorch's current default dtype is used

    Returns
    x : CANN inputs [I1, J] with shape (n_samples**2, 2)
    y : Analytical Helmholtz free-energy density with shape (n_samples**2,)
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if min_J <= 0 or max_J <= 0:
        raise ValueError("J must be strictly positive because log(J) is used.")

    if dtype is None:
        dtype = torch.get_default_dtype()

    I1_samples = torch.linspace(min_I1,max_I1,n_samples,device=device,dtype=dtype)
    J_samples = torch.linspace(min_J,max_J,n_samples,device=device,dtype=dtype)

    I1, J = torch.meshgrid(I1_samples, J_samples, indexing="ij")

    I1 = I1.flatten()
    J = J.flatten()

    x = torch.stack((I1, J), dim=1)
    y = neo_hookean_energy(I1, J)

    return x, y


def compute_normalization_stats(x, y):
    """Compute normalization statistics from the training dataset.

    Parameters
    x : Training inputs with shape (N, 2)
    y : Training targets with shape (N,)

    Returns NormalizationStats (mean and standard deviation of the training inputs and targets)
    """
    return NormalizationStats(
        x_mean=x.mean(dim=0),
        x_std=x.std(dim=0),
        y_mean=y.mean(),
        y_std=y.std(),
    )


def normalize_data(x, y, stats):
    """Normalize CANN inputs and targets using training statistics

    Parameters
    x : Input data with shape (N, 2)
    y : Target data with shape (N,)
    stats : NormalizationStats Statistics computed from the training dataset

    Returns
    x_normalized: Normalized CANN inputs
    y_normalized: Normalized CANN targets
    """
    x_normalized = (x - stats.x_mean) / stats.x_std
    y_normalized = (y - stats.y_mean) / stats.y_std

    return x_normalized, y_normalized


def denormalize_energy(y_normalized, stats):
    """Convert normalized CANN energy predictions back to physical units."""
    return y_normalized * stats.y_std + stats.y_mean


def create_dataloader(
    x,
    y,
    batch_size=8,
    *,
    shuffle=True,
    drop_last=False,
):
    """Create a PyTorch DataLoader for CANN training

    Parameters
    x : torch.Tensor
        Normalized CANN inputs
    y : torch.Tensor
        Normalized CANN targets
    batch_size : int, optional
        Number of samples per optimizer step
    shuffle : bool, optional
        Shuffle the dataset at the beginning of each epoch
    drop_last : bool, optional
        Drop an incomplete final batch

    Returns torch.utils.data.DataLoader containing paired CANN inputs and targets
    """
    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
