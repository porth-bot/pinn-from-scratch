"""Collocation samplers and loss terms: domains, grad flags, determinism."""

import torch

from pinn import losses
from pinn.derivatives import u_t, u_xx
from pinn.model import MLP, set_seed

XR = (-1.0, 2.0)
TR = (0.0, 1.0)


def test_interior_points_lie_in_domain_and_track_grad():
    gen = torch.Generator().manual_seed(0)
    c = losses.interior_points(500, XR, TR, gen)
    assert c.shape == (500, 2)
    assert c.requires_grad
    x, t = c[:, 0], c[:, 1]
    assert x.min() >= XR[0] and x.max() <= XR[1]
    assert t.min() >= TR[0] and t.max() <= TR[1]


def test_initial_points_are_on_t0():
    gen = torch.Generator().manual_seed(0)
    c = losses.initial_points(100, XR, TR[0], gen)
    assert torch.allclose(c[:, 1], torch.full((100,), TR[0]))


def test_boundary_points_sit_on_the_walls():
    gen = torch.Generator().manual_seed(0)
    left, right = losses.boundary_points(100, XR, TR, gen)
    assert torch.allclose(left[:, 0], torch.full((100,), XR[0]))
    assert torch.allclose(right[:, 0], torch.full((100,), XR[1]))


def test_samplers_are_deterministic_given_the_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = losses.interior_points(64, XR, TR, g1)
    b = losses.interior_points(64, XR, TR, g2)
    assert torch.equal(a.detach(), b.detach())


def test_adaptive_points_concentrate_where_the_residual_is_large():
    # A synthetic residual that is large only in a thin band near x = 0. RAD
    # should pull the collocation points into that band; uniform sampling would
    # leave only ~(band width / domain width) of them there.
    gen = torch.Generator().manual_seed(0)

    def spiky_residual(u, coords):
        x = coords[:, 0:1]
        return torch.exp(-(x / 0.05) ** 2)          # peaked at x = 0

    ident = MLP(in_dim=2, out_dim=1, width=8, depth=2)  # model is unused here
    pts = losses.adaptive_interior_points(
        ident, spiky_residual, 2000, (-1.0, 1.0), (0.0, 1.0), gen,
        n_candidates=40000, k=2.0, c=0.1,
    )
    assert pts.shape == (2000, 2) and pts.requires_grad
    frac_in_band = (pts[:, 0].abs() < 0.1).float().mean().item()
    # a uniform sample would put ~10% (0.2 / 2.0) of points in |x| < 0.1;
    # the residual is ~100x larger there, so adaptivity must beat uniform by a lot.
    assert frac_in_band > 0.6


def test_adaptive_points_reduce_to_uniform_when_the_residual_is_flat():
    # Flat residual -> the weight is the same everywhere -> the sampled points
    # are just a uniform draw. Check the in-band fraction matches the geometric
    # 10% a uniform sampler gives, not the concentrated RAD fraction above.
    gen = torch.Generator().manual_seed(1)
    flat = lambda u, coords: torch.ones(coords.shape[0], 1)
    ident = MLP(in_dim=2, out_dim=1, width=8, depth=2)
    pts = losses.adaptive_interior_points(
        ident, flat, 4000, (-1.0, 1.0), (0.0, 1.0), gen, n_candidates=40000,
    )
    frac_in_band = (pts[:, 0].abs() < 0.1).float().mean().item()
    assert abs(frac_in_band - 0.10) < 0.03


def test_adaptive_points_reject_bad_hyperparameters():
    import pytest

    gen = torch.Generator().manual_seed(0)
    flat = lambda u, coords: torch.ones(coords.shape[0], 1)
    ident = MLP(in_dim=2, out_dim=1, width=8, depth=2)
    for bad in (dict(k=-1.0), dict(c=0.0), dict(c=-1.0)):
        with pytest.raises(ValueError):
            losses.adaptive_interior_points(
                ident, flat, 10, (-1.0, 1.0), (0.0, 1.0), gen, **bad)


def test_residual_loss_is_zero_for_the_exact_solution():
    # r = u_t - alpha u_xx on u = sin(a x) exp(-b t) with alpha = b/a^2 is 0,
    # so the residual loss of that analytic field is ~0 (float roundoff only).
    a, b = 2.0, 0.7
    alpha = b / a ** 2

    class Analytic(torch.nn.Module):
        def forward(self, coords):
            x, t = coords[:, 0:1], coords[:, 1:2]
            return torch.sin(a * x) * torch.exp(-b * t)

    def residual(u, coords):
        return u_t(u, coords) - alpha * u_xx(u, coords)

    gen = torch.Generator().manual_seed(0)
    c = losses.interior_points(256, (-1.0, 1.0), (0.0, 1.0), gen)
    loss = losses.residual_loss(Analytic(), c, residual)
    assert loss.item() < 1e-10


def test_data_loss_matches_manual_mse():
    set_seed(0)
    net = MLP(width=16, depth=2)
    gen = torch.Generator().manual_seed(0)
    c = losses.initial_points(32, XR, 0.0, gen)
    target = torch.randn(32, 1)
    manual = torch.mean((net(c) - target) ** 2)
    assert torch.allclose(losses.data_loss(net, c, target), manual)
