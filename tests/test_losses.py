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
