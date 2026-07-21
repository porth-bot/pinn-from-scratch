"""The hard-constraint ansatz's whole claim is that the initial and boundary
conditions hold *by construction* -- exactly, for any network weights, before
and during training. We check that directly (the constraint errors sit at the
float32 floor for a random untrained net), that it composes with the autograd
derivative helpers (a finite PDE residual), and that a short residual-only train
still reduces the error -- so the exactness is not bought by freezing the model."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hard_bc import HardConstraintNet, _constraint_errors, train_hard  # noqa: E402
from heat import heat_exact, heat_residual, initial_condition  # noqa: E402
from pinn.model import MLP, set_seed  # noqa: E402


def _random_ansatz(seed=0, width=16, depth=3):
    set_seed(seed)
    return HardConstraintNet(MLP(in_dim=2, out_dim=1, width=width, depth=depth,
                                 activation="tanh"))


def test_ic_and_bc_exact_by_construction_untrained():
    """A *random, untrained* ansatz already satisfies both conditions to the
    float32 floor: the ``t`` factor kills the correction at ``t=0`` (so
    ``u = g``) and the ``x(1-x)`` factor kills it at the walls (so ``u = 0``).
    This is the property the whole method rests on."""
    model = _random_ansatz()
    ic_err, bc_err = _constraint_errors(model, n=300)
    assert ic_err < 1e-5   # u(x,0) == g(x) exactly
    assert bc_err < 1e-5   # u(0,t) == u(1,t) == 0 exactly


def test_ic_exact_matches_initial_condition_pointwise():
    """Spell out the IC guarantee: at t=0 the ansatz reproduces the analytic
    initial condition on random x, independent of the (random) network."""
    model = _random_ansatz(seed=3)
    x = torch.rand(50, 1)
    coords0 = torch.cat([x, torch.zeros_like(x)], dim=1)
    with torch.no_grad():
        u0 = model(coords0)
    np.testing.assert_allclose(u0.numpy(), initial_condition(x).numpy(), atol=1e-6)


def test_ansatz_composes_with_autograd_residual():
    """The wrapper must be a differentiable function of coords so u_t/u_xx of the
    *full* ansatz (network + g + envelope) are well defined and finite -- the
    residual is what the hard run trains on."""
    model = _random_ansatz(seed=1)
    coords = torch.rand(32, 2, requires_grad=True)
    u = model(coords)
    r = heat_residual(u, coords)
    assert r.shape == (32, 1)
    assert torch.isfinite(r).all()


def test_short_train_reduces_error_residual_only():
    """Exactness is not from freezing the model: a short residual-only train
    (no IC/BC penalty) still drives the relative L2 error down, and the
    constraints stay exact throughout."""
    model, hist = train_hard(n_interior=512, width=32, depth=3, steps=400,
                             lr=1e-3, seed=0)
    err0, err1 = hist[0][2], hist[-1][2]
    assert err1 < 0.6 * err0                    # clear reduction
    assert hist[-1][3] < 1e-5 and hist[-1][4] < 1e-5   # IC/BC still exact
