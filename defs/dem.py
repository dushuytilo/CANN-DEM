"""Deep Energy Method (DEM) definitions
"""

import torch
from torch import nn


class MyPow(nn.Module):
    """Custom activation used in the original DEM implementation
    Computes x**2 + slope*x with a trainable slope
    """

    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.pow(x, 2) + self.slope * x


class DEMNet(nn.Module):
    """FCNN used to approximate the displacement field in the DEM
    The network maps spatial coordinates [x, y] to displacements [u_x, u_y]
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            MyPow(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x, force_right, E, Lx):
        """Evaluate the displacement network

        Parameters
        x: Spatial coordinates with shape (N, 2)
        force_right:  Applied traction on the right boundary
        E: Young's modulus used for displacement scaling
        Lx : Domain length in x-direction

        Returns Predicted displacement field with shape (N, 2)
        """
        displacement_scale = force_right / (E / Lx)
        out = displacement_scale * self.network(x)

        # Output transformation from the original DEM implementation:
        # enforce u_x = u_y = 0 at x = 0.
        out = out * x[:, 0].unsqueeze(1)

        return out


def create_sample_points(
    Lx,
    Ly,
    samples_x,
    samples_y,
    device,
    dtype,
):
    """Create the rectangular DEM collocation grid

    Returns
    sample_points: Flattened coordinates with shape (samples_x*samples_y, 2)
    delta_x: Grid spacing in x-direction
    delta_y: Grid spacing in y-direction
    """
    x = torch.linspace(0, Lx, samples_x, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, samples_y, device=device, dtype=dtype)

    X, Y = torch.meshgrid(x, y, indexing="ij")

    sample_points = torch.cat(
        (X.reshape(-1, 1), Y.reshape(-1, 1)),
        dim=1,
    )
    sample_points.requires_grad_(True)

    delta_x = Lx / (samples_x - 1)
    delta_y = Ly / (samples_y - 1)

    return sample_points, delta_x, delta_y


def compute_linear_strain(X, U, samples_x, samples_y):
    """Compute the small-strain tensor from the displacement field

    Parameters
    X: Spatial coordinates with shape (N, 2)
    U: Predicted displacements with shape (N, 2)
    samples_x, samples_y: int Number of grid points in each direction

    Returns Symmetric strain tensor with shape (samples_x, samples_y, 2, 2).
    """
    du_dx = torch.autograd.grad(
        outputs=U[:, 0],
        inputs=X,
        grad_outputs=torch.ones_like(U[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    dv_dx = torch.autograd.grad(
        outputs=U[:, 1],
        inputs=X,
        grad_outputs=torch.ones_like(U[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0]

    displacement_gradient = torch.stack((du_dx, dv_dx), dim=1)
    displacement_gradient = displacement_gradient.reshape(
        samples_x,
        samples_y,
        2,
        2,
    )

    strain = 0.5 * (
        displacement_gradient
        + displacement_gradient.transpose(-1, -2)
    )

    return strain


def integrate_energy_trapezoid(psi, delta_x, delta_y):
    """Integrate strain-energy density over the 2D domain"""
    return torch.trapezoid(
        torch.trapezoid(psi, dx=delta_y, dim=1),
        dx=delta_x,
        dim=0,
    )


def integrate_energy_simpson(psi, delta_x, delta_y):
    """Integrate strain-energy density using composite Simpson's rule
    Both grid dimensions must contain an odd number of points, equivalent to an even number of integration intervals
    """
    nx, ny = psi.shape

    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError(
            "Simpson integration requires an odd number of grid points "
            "in both directions."
        )

    weights_x = torch.ones(nx, dtype=psi.dtype, device=psi.device)
    weights_y = torch.ones(ny, dtype=psi.dtype, device=psi.device)

    weights_x[1:-1:2] = 4
    weights_x[2:-1:2] = 2
    weights_y[1:-1:2] = 4
    weights_y[2:-1:2] = 2

    return (
        torch.sum(psi * weights_x[:, None] * weights_y[None, :])
        * delta_x
        * delta_y
        / 9.0
    )


def calculate_divergence_loss(
    sig,
    delta_x,
    delta_y,
):
    """Calculate the equilibrium residual ||div(sigma)||^2
    Central finite differences are used at interior grid points
    """
    div_sig = (
        sig[2:, 1:-1, :, 0] - sig[:-2, 1:-1, :, 0]
    ) / (2 * delta_x)

    div_sig = div_sig + (
        sig[1:-1, 2:, :, 1] - sig[1:-1, :-2, :, 1]
    ) / (2 * delta_y)

    return torch.sum(div_sig**2)


def integrate_traction_energy(
    U,
    force_right,
    delta_y,
):
    """Integrate external work on the right boundary"""
    return force_right * torch.trapezoid(
        U[-1, :, 0],
        dx=delta_y,
    )


def calculate_boundary_losses(
    U,
    sig,
    E,
    force_right,
    delta_x,
    delta_y,
    left_weight,
    right_weight,
):
    """Calculate displacement and traction boundary-condition losses
    """
    samples_x, samples_y = U.shape[:2]

    # Left boundary: fixed displacement.
    loss_left = torch.sum(
        (delta_y * U[0, :, :] * E * left_weight) ** 2
    )

    # Right boundary: prescribed normal traction.
    forces_right = torch.ones(
        samples_y,
        dtype=U.dtype,
        device=U.device,
    )
    forces_right[[0, -1]] = 0.5
    forces_right = forces_right * force_right * delta_y

    tractions_right = sig[-1, :, 0, 0] * delta_y
    endpoint_factor = torch.ones_like(tractions_right)
    endpoint_factor[[0, -1]] = 0.5
    tractions_right = tractions_right * endpoint_factor

    loss_right_normal = right_weight * torch.sum(
        (tractions_right - forces_right) ** 2
    )

    # Right boundary: zero shear traction.
    shear_right = sig[-1, :, 0, 1] * delta_y
    loss_right_shear = torch.sum(shear_right**2)

    # Lower boundary: zero normal and shear traction.
    traction_lower = sig[1:-1, 0, 1, 1] * delta_x
    shear_lower = sig[1:-1, 0, 1, 0] * delta_x
    loss_lower = torch.sum(traction_lower**2) + torch.sum(shear_lower**2)

    # Upper boundary: zero normal and shear traction.
    traction_upper = sig[1:-1, -1, 1, 1] * delta_x
    shear_upper = sig[1:-1, -1, 1, 0] * delta_x
    loss_upper = torch.sum(traction_upper**2) + torch.sum(shear_upper**2)

    loss_right = loss_right_normal + loss_right_shear

    return {
        "left": loss_left,
        "right": loss_right,
        "upper": loss_upper,
        "lower": loss_lower,
        "total": loss_left + loss_right + loss_upper + loss_lower,
    }


def approximate_displacement(
    sample_points,
    Lx,
    Ly,
    E,
    nu,
    force_right,
):
    delta_l = force_right / (E / Lx)

    target = torch.zeros_like(sample_points)

    target[:, 0] = (
        sample_points[:, 0] / Lx
        * delta_l
    )

    target[:, 1] = (
        -sample_points[:, 0]
        / Lx
        * nu
        * (sample_points[:, 1] / Ly - 0.5)
        * delta_l
    )
    return target

def combine_dem_loss(
    internal_energy,
    traction_energy,
    boundary_loss,
    divergence_loss,
    boundary_weight,
    divergence_weight,
):
    return (
        internal_energy
        - traction_energy
        + boundary_weight * boundary_loss
        + divergence_weight * divergence_loss
    )
