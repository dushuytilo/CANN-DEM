"""
Material model definitions for the CANN-DEM project

Implements the 2D compressible Neo-Hookean material model used in this work
"""

import torch


def compute_deformation_gradient(u, X):
    """Compute the 2D deformation gradient F = I + grad(u)

    Parameters
    u: Displacement field with shape (N, 2). It must be defined as a differentiable function of X
    X: Reference coordinates with shape (N, 2) and requires_grad=True

    Returns
        Deformation gradient with shape (N, 2, 2)
    """
    grad_u = []

    for i in range(2):
        du_dX = torch.autograd.grad(
            outputs=u[:, i],
            inputs=X,
            grad_outputs=torch.ones_like(u[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_u.append(du_dX)

    grad_u = torch.stack(grad_u, dim=1)
    identity = torch.eye(2, dtype=X.dtype, device=X.device).unsqueeze(0)

    return identity + grad_u


def compute_invariants(F):
    """Compute J and the first invariant I1 from a 2D deformation gradient"""
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    I1 = torch.einsum("bik,bik->b", F, F)

    return J, I1


def neo_hookean_energy(I1, J, mu, lam):
    """Compute the Helmholtz free-energy density

    Psi = lam/2 * (ln J)^2 - mu * ln J + mu/2 * (I1 - 2)
    """
    log_J = torch.log(J)

    return (
        lam / 2 * log_J**2
        - mu * log_J
        + mu / 2 * (I1 - 2)
    )


def neo_hookean_from_displacement(u, X, mu, lam):
    """Compute Helmholtz free-energy density directly from X and u(X)"""
    F = compute_deformation_gradient(u, X)
    J, I1 = compute_invariants(F)

    return neo_hookean_energy(I1, J, mu, lam)
