"""
Material model definitions for the CANN-DEM project

Implements the 2D compressible Neo-Hookean material model used in this work
"""

import torch


# Lamé parameters [Pa]
MU = 384.614
LAM = 576.923


def compute_deformation_gradient(u, X):
    """Compute the 2D deformation gradient F = I + grad(u)

    Parameters
    u : Displacement field with shape (N, 2). It must be defined as a differentiable function of X
    X : Reference coordinates with shape (N, 2) and requires_grad=True

    Returns Deformation gradient with shape (N, 2, 2)
    """
    grad_u = []

    for i in range(2):
        du_dX = torch.autograd.grad(outputs=u[:, i],inputs=X,grad_outputs=torch.ones_like(u[:, i]),create_graph=True,retain_graph=True,)[0]
        grad_u.append(du_dX)

    grad_u = torch.stack(grad_u, dim=1)
    identity = torch.eye(2,dtype=X.dtype,device=X.device,).unsqueeze(0)

    return identity + grad_u


def compute_invariants(F):
    """Compute J and the first invariant I1 from a 2D deformation gradient

    Parameters
    F : Deformation gradient with shape (N, 2, 2)

    Returns J and I1
    """
    # Determinant of a 2x2 matrix: ad - bc
    J = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]

    # I1 = F:F = tr(F^T F)
    I1 = torch.einsum("bik,bik->b", F, F)

    return J, I1


def neo_hookean_energy(I1, J, mu=MU, lam=LAM):
    """Compute the Helmholtz free-energy density of the material

    Implements Homework 2, Task 2, Eq. (1):

        Psi = lam/2 * (ln J)^2
              - mu * ln J
              + mu/2 * (I1 - 2)

    Parameters
    I1 : First invariant of the right Cauchy-Green tensor
    J : Determinant of the deformation gradient
    mu : First Lamé parameter [Pa]
    lam : Second Lamé parameter [Pa]

    Returns Helmholtz free-energy density Psi
    """
    log_J = torch.log(J)

    return (
        lam / 2 * log_J**2
        - mu * log_J
        + mu / 2 * (I1 - 2)
    )


def neo_hookean_from_displacement(u, X, mu=MU, lam=LAM):
    """Compute Helmholtz free-energy density directly from X and u(X)

    Parameters
    u : Displacement field with shape (N, 2)
    X : Reference coordinates with shape (N, 2) and requires_grad=True
    mu : First Lamé parameter [Pa]
    lam : Second Lamé parameter [Pa]

    Returns Helmholtz free-energy density Psi
    """
    F = compute_deformation_gradient(u, X)
    J, I1 = compute_invariants(F)

    return neo_hookean_energy(I1, J, mu=mu, lam=lam)