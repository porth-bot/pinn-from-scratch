"""The neural field u_theta(x, t): a plain fully-connected network.

A physics-informed neural network is nothing more exotic than a smooth
function approximator u_theta whose *derivatives* are made to satisfy a PDE.
The only real requirements on the architecture are:

- Smoothness. The PDE residual involves u_x, u_xx, u_t, so the activation
  must be differentiable to as many orders as the equation needs. ``tanh``
  is the canonical PINN choice: infinitely differentiable, and its
  derivatives stay bounded (unlike ReLU, whose second derivative is zero
  almost everywhere -- a ReLU net has u_xx == 0 and cannot express a
  diffusion residual at all).

- Enough capacity, but not too much depth. Small MLPs (a few hidden layers,
  tens to a couple hundred units) are standard; the hard part of PINN
  training is the *optimization*, not the approximation.

The network maps a coordinate ``(x, t)`` (or any input dimension) to the
scalar field value. We keep it deliberately small and explicit -- no
``nn.Sequential`` magic -- so the forward pass reads like the math.

Weight initialization matters more than usual because the residual loss
differentiates through the network. Two schemes are provided:

- Xavier/Glorot (default), the standard for tanh nets: it keeps the
  activation variance roughly constant across layers at init.
- A SIREN-style "sine" first-layer scaling (Sitzmann et al. 2020) available
  as an option, useful when the target field is high-frequency; the plain
  tanh net exhibits spectral bias (Day 11 in this repo makes that failure
  concrete), and rescaling the first layer is one cheap mitigation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    """A fully-connected network u_theta: R^in -> R^out with tanh activations.

    Parameters
    ----------
    in_dim : int
        Number of input coordinates (e.g. 2 for (x, t)).
    out_dim : int
        Number of output fields (1 for a scalar PDE).
    width : int
        Hidden-layer width.
    depth : int
        Number of hidden layers (so there are ``depth + 1`` linear maps).
    activation : {"tanh", "sin"}
        Hidden nonlinearity. ``tanh`` is the default and the safe choice;
        ``sin`` turns the net into a SIREN-style periodic-activation network.
    first_omega : float
        Frequency scaling applied to the first layer's pre-activation when
        ``activation="sin"`` (the SIREN omega_0). Ignored for tanh.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
        first_omega: float = 30.0,
    ):
        super().__init__()
        if activation not in ("tanh", "sin"):
            raise ValueError(f"unknown activation {activation!r}")
        self.activation = activation
        self.first_omega = float(first_omega)

        dims = [in_dim] + [width] * depth + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights.

        tanh: Xavier/Glorot uniform, the textbook variance-preserving scheme
        for symmetric saturating activations. sin: the SIREN scheme -- the
        first layer draws from U(-1/in, 1/in) and is later multiplied by
        omega_0, every later layer from U(-sqrt(6/fan_in)/omega, ...), which
        keeps the pre-activation distribution stationary through the sines
        (Sitzmann et al. 2020, Sec. 3.2).
        """
        if self.activation == "tanh":
            for lin in self.layers:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            return

        # SIREN initialization
        omega = self.first_omega
        for i, lin in enumerate(self.layers):
            fan_in = lin.weight.shape[1]
            with torch.no_grad():
                if i == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = math.sqrt(6.0 / fan_in) / omega
                lin.weight.uniform_(-bound, bound)
                nn.init.zeros_(lin.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate u_theta at a batch of coordinates.

        coords : (N, in_dim). Returns (N, out_dim).
        """
        h = coords
        n_hidden = len(self.layers) - 1
        for i, lin in enumerate(self.layers):
            z = lin(h)
            if i < n_hidden:
                if self.activation == "tanh":
                    h = torch.tanh(z)
                else:
                    omega = self.first_omega if i == 0 else 1.0
                    h = torch.sin(omega * z)
            else:
                h = z  # linear output layer
        return h


def set_seed(seed: int) -> None:
    """Seed torch (and its CUDA state, if any) for reproducible runs.

    PINN results are sensitive to initialization and collocation sampling, so
    every experiment pins a seed and the tests assert determinism.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
