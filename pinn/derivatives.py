"""Exact derivatives of the network by reverse-mode autograd.

This is the piece that makes a PINN a PINN. A finite-difference or spectral
solver stores u on a grid and approximates u_xx by differencing neighbours;
a PINN instead has a closed-form differentiable function u_theta and asks
autograd for its *exact* derivatives at arbitrary points. No grid, no
truncation error in the derivative itself -- the only error is in how well
u_theta solves the PDE.

The one subtlety is that ``torch.autograd.grad`` gives the gradient of a
*scalar* with respect to the inputs. For a batch of N collocation points we
want du/dx at each point independently, i.e. the diagonal of the Jacobian,
not a sum-reduced gradient. The trick (standard in every PINN codebase) is:

    grad(sum_i u_i, coords)  ==  [ d(sum_i u_i)/dx_j ]_j  ==  [ du_j/dx_j ]_j

because u_j depends only on coords_j (the network is applied pointwise across
the batch), so the cross terms du_i/dx_j (i != j) are zero. Summing over the
batch and differentiating therefore returns exactly the per-point derivative.
We pass ``create_graph=True`` so the resulting derivative is itself part of
the graph and can be differentiated again -- that is how u_xx, u_xxx, ...
are obtained, and how the residual (a function of these derivatives) can be
backpropagated into theta during training.

Everything here is written out rather than hidden behind a helper library so
the mechanism is legible; ``tests/test_derivatives.py`` checks every one of
these against finite differences on functions with known derivatives.
"""

from __future__ import annotations

import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """d(outputs)/d(inputs), the per-point gradient, shape == inputs.shape.

    ``outputs`` is (N, 1) and ``inputs`` is (N, d). Returns (N, d) where row i
    is grad u(coords_i). Requires ``inputs.requires_grad_(True)`` upstream and
    that ``outputs`` was computed from ``inputs``.

    ``grad_outputs=ones`` sums the batch (see the module docstring for why the
    off-diagonal Jacobian terms vanish); ``create_graph=True`` keeps the
    result differentiable for higher-order derivatives.
    """
    (g,) = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )
    return g


def partial(outputs: torch.Tensor, inputs: torch.Tensor, dim: int) -> torch.Tensor:
    """Single partial derivative d(outputs)/d(inputs[:, dim]), shape (N, 1).

    A thin convenience over :func:`grad` that selects one input coordinate and
    keeps the trailing singleton dimension so results compose (e.g. to feed
    another :func:`partial` for a second derivative).
    """
    return grad(outputs, inputs)[:, dim : dim + 1]


def _second(outputs: torch.Tensor, inputs: torch.Tensor, dim: int) -> torch.Tensor:
    """d^2(outputs)/d(inputs[:, dim])^2 by differentiating the first partial."""
    first = partial(outputs, inputs, dim)
    return partial(first, inputs, dim)


# ---------------------------------------------------------------------------
# Named helpers for the (x, t) convention used across the experiments.
# Coordinates are stored as columns [x, t] -> dim 0 is space, dim 1 is time.
# ---------------------------------------------------------------------------

X, T = 0, 1


def u_x(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """First spatial derivative u_x."""
    return partial(u, coords, X)


def u_t(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """First time derivative u_t."""
    return partial(u, coords, T)


def u_xx(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Second spatial derivative u_xx."""
    return _second(u, coords, X)


def u_tt(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Second time derivative u_tt (for wave-type equations)."""
    return _second(u, coords, T)


def laplacian(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sum of unmixed second derivatives over *all* input dimensions.

    For coords with d spatial columns this is sum_i u_{x_i x_i}. On the (x, t)
    convention with a time column you usually want :func:`u_xx` instead; this
    helper is for purely spatial problems (e.g. a 2D Poisson demo).
    """
    total = torch.zeros_like(u)
    for d in range(coords.shape[1]):
        total = total + _second(u, coords, d)
    return total
