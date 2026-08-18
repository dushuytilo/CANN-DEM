"""Constitutive Artificial Neural Network (CANN) architecture
The network is based on the CANN used in Homework 2:
    input  -> [I1, J]
    output -> normalized Helmholtz free-energy density Psi
The hidden layers combine linear, quadratic, and exponential terms
"""
import torch
from torch import nn

class CANN(nn.Module):
    """Constitutive Artificial Neural Network for Neo-Hookean energy prediction

    Parameters
    input_dim : int
        Number of input features. For this project: I1 and J -> 2
    hidden_dim : int
        Total width of each hidden layer. Must be even because each hidden layer is split equally between two transformation types
    output_dim : int
        Number of outputs. For this project: Psi -> 1
    """

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even.")

        # First layer: half linear, half quadratic
        self.layer1_linear = nn.Linear(input_dim, hidden_dim // 2)
        self.layer1_quadratic = nn.Linear(input_dim, hidden_dim // 2)

        # Second layer: half linear, half exponential
        self.layer2_linear = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer2_exponential = nn.Linear(hidden_dim, hidden_dim // 2)

        # Final layer combines both branches into the material-energy output
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Evaluate the CANN

        Parameters
        x : Normalized invariant inputs [I1, J] with shape (N, 2)

        Returns predicted normalized Helmholtz free-energy density
        """
        linear_out = self.layer1_linear(x)
        quadratic_out = torch.pow(self.layer1_quadratic(x), 2)
        combined1 = torch.cat((linear_out, quadratic_out), dim=1)

        linear_out2 = self.layer2_linear(combined1)
        exponential_out = torch.exp(self.layer2_exponential(combined1))
        combined2 = torch.cat((linear_out2, exponential_out), dim=1)

        return self.output_layer(combined2)