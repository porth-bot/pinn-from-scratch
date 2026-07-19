"""Collocation samplers and the PINN loss terms.

A PINN turns "solve the PDE" into "minimize a loss" by sampling points and
penalizing the residual there. On a domain (x, t) in [x0, x1] x [t0, t1] the
total objective is

    L(theta) = w_r * L_residual + w_ic * L_ic + w_bc * L_bc,

where

- L_residual = mean over interior collocation points of  r(x, t; theta)^2,
  with r the PDE written as r = 0 (e.g. r = u_t - alpha u_xx for the heat
  equation). This is the only term that references the physics; it is passed
  in as a callable so this module stays equation-agnostic.

- L_ic = mean squared error of u(x, t0) against the given initial condition.

- L_bc = mean squared error of the boundary values (Dirichlet here; a
  callable residual can express Neumann/periodic just as easily).

The weights w_* trade the terms off. Balancing them is a genuine PINN
difficulty (adaptive weighting is a research topic); here they are explicit
knobs and the experiments report what was used.

The samplers return coordinate tensors with ``requires_grad_(True)`` already
set on the interior points, because the residual needs to differentiate the
network with respect to them. Sampling is uniform (Latin-hypercube and
adaptive schemes are later work); every sampler takes an explicit
``torch.Generator`` so runs are reproducible.
"""

from __future__ import annotations

from typing import Callable

import torch

Residual = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _uniform(n: int, lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
    """n draws from U(lo, hi) as a column vector, using the given generator."""
    return lo + (hi - lo) * torch.rand(n, 1, generator=gen)


def interior_points(
    n: int,
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    gen: torch.Generator,
) -> torch.Tensor:
    """Uniform collocation points in the open space-time interior.

    Returns (n, 2) = [x, t] with ``requires_grad_`` enabled so the residual
    can autograd-differentiate the network at these points.
    """
    x = _uniform(n, x_range[0], x_range[1], gen)
    t = _uniform(n, t_range[0], t_range[1], gen)
    coords = torch.cat([x, t], dim=1)
    coords.requires_grad_(True)
    return coords


def initial_points(
    n: int,
    x_range: tuple[float, float],
    t0: float,
    gen: torch.Generator,
) -> torch.Tensor:
    """n points on the initial slice t = t0, x ~ U(x_range). Shape (n, 2)."""
    x = _uniform(n, x_range[0], x_range[1], gen)
    t = torch.full((n, 1), float(t0))
    return torch.cat([x, t], dim=1)


def boundary_points(
    n: int,
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """n points on each of the two spatial boundaries x = x0 and x = x1.

    Returns (left, right), each (n, 2) with t ~ U(t_range). Kept separate so a
    caller can impose different data on each wall (or wire them together for a
    periodic condition).
    """
    t_left = _uniform(n, t_range[0], t_range[1], gen)
    t_right = _uniform(n, t_range[0], t_range[1], gen)
    left = torch.cat([torch.full((n, 1), float(x_range[0])), t_left], dim=1)
    right = torch.cat([torch.full((n, 1), float(x_range[1])), t_right], dim=1)
    return left, right


def adaptive_interior_points(
    model,
    residual: Residual,
    n: int,
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    gen: torch.Generator,
    *,
    n_candidates: int = 40000,
    k: float = 1.0,
    c: float = 1.0,
) -> torch.Tensor:
    """Residual-adaptive collocation points (RAD; Wu et al. 2023).

    Uniform collocation spends the same point budget everywhere, but a PDE with
    a localized sharp feature -- the viscous shock in Burgers, a boundary layer,
    a front -- has its error concentrated in a thin region a uniform sample
    barely resolves. RAD moves the points to where the residual is large.

    Draw a large uniform candidate pool, evaluate the current residual there, and
    resample ``n`` points from the discrete density

        p_i  proportional to  ( |r_i|^k / mean_j |r_j|^k ) + c,

    with ``k`` sharpening the concentration and ``c > 0`` keeping a uniform floor
    so the rest of the domain is never abandoned (Wu et al.'s defaults k=c=1
    reproduce their RAD). At ``k=0`` or ``c -> inf`` this collapses back to
    uniform sampling. The returned tensor has ``requires_grad_`` set, like
    :func:`interior_points`, so it drops straight into :func:`residual_loss`.

    The residual is evaluated under autograd (it differentiates the network) but
    only its magnitude is used to build the sampling weights, so the pool graph
    is detached before selection -- no training signal flows through the choice
    of points, exactly as intended (the points are where we *measure* the PDE,
    not parameters to optimize).
    """
    if not 0.0 <= k or c <= 0.0:
        raise ValueError("require k >= 0 and c > 0")
    cand = interior_points(n_candidates, x_range, t_range, gen)
    r = residual(model(cand), cand)
    with torch.no_grad():
        w = r.detach().abs().flatten() ** k
        w = w / w.mean().clamp_min(1e-30) + c
        idx = torch.multinomial(w, n, replacement=n > n_candidates, generator=gen)
    pts = cand.detach()[idx].clone()
    pts.requires_grad_(True)
    return pts


def residual_loss(model, coords: torch.Tensor, residual: Residual) -> torch.Tensor:
    """Mean squared PDE residual at the interior collocation points.

    ``residual(u, coords)`` receives the network output and the (grad-enabled)
    coordinates and returns the PDE written as r = 0. We square and average.
    """
    u = model(coords)
    r = residual(u, coords)
    return torch.mean(r ** 2)


def data_loss(model, coords: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the network against supervised targets.

    Used for both the initial condition and Dirichlet boundary values -- both
    are just "u should equal this here" constraints.
    """
    return torch.mean((model(coords) - target) ** 2)
