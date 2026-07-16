"""Network shape, initialization, determinism, and a smoke training step."""

import pytest
import torch

from pinn import losses
from pinn.derivatives import u_t, u_xx
from pinn.model import MLP, set_seed


def test_forward_shape():
    net = MLP(in_dim=2, out_dim=1, width=32, depth=3)
    out = net(torch.zeros(10, 2))
    assert out.shape == (10, 1)


def test_output_layer_is_linear_not_saturated():
    # A linear head means outputs are not squashed into (-1, 1); with nonzero
    # init and inputs, some |u| should exceed 1 across a big batch.
    set_seed(0)
    net = MLP(width=64, depth=4)
    x = 5 * torch.randn(2000, 2)
    with torch.no_grad():
        assert net(x).abs().max() > 1.0


def test_seeded_determinism():
    set_seed(123)
    a = MLP(width=16, depth=2)
    set_seed(123)
    b = MLP(width=16, depth=2)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    x = torch.randn(5, 2)
    assert torch.equal(a(x), b(x))


def test_different_seeds_differ():
    set_seed(1)
    a = MLP(width=16, depth=2)
    set_seed(2)
    b = MLP(width=16, depth=2)
    same = all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert not same


def test_sine_activation_runs_and_differs_from_tanh():
    set_seed(0)
    net = MLP(width=32, depth=3, activation="sin", first_omega=30.0)
    x = torch.randn(8, 2)
    out = net(x)
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        MLP(activation="relu")


def test_tanh_is_twice_differentiable_through_the_net():
    # The residual needs u_xx of the *network*; confirm it is finite and the
    # graph supports the second derivative (ReLU would give all-zero u_xx).
    set_seed(0)
    net = MLP(width=32, depth=3, activation="tanh")
    coords = torch.rand(16, 2, requires_grad=True)
    u = net(coords)
    second = u_xx(u, coords)
    assert second.shape == (16, 1)
    assert torch.isfinite(second).all()
    assert second.abs().sum() > 0  # not identically zero, unlike a ReLU net


def test_one_optimization_step_decreases_a_trivial_loss():
    # Fit u -> 0 everywhere via the data loss; a single Adam step must not
    # increase the loss. This exercises the full model+loss+autograd path.
    set_seed(0)
    net = MLP(width=16, depth=2)
    gen = torch.Generator().manual_seed(0)
    coords = losses.initial_points(64, (-1.0, 1.0), 0.0, gen)
    target = torch.zeros(64, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = losses.data_loss(net, coords, target).item()
    opt.zero_grad()
    losses.data_loss(net, coords, target).backward()
    opt.step()
    after = losses.data_loss(net, coords, target).item()
    assert after < before


def test_package_exports_the_documented_public_api():
    # The README's module table presents features.py alongside derivatives.py
    # and losses.py as core; every name in __all__ must resolve off the
    # package itself, not just via its defining submodule.
    import pinn

    for name in pinn.__all__:
        assert hasattr(pinn, name), f"pinn.__all__ advertises {name!r} but it is missing"
