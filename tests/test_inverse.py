"""The inverse problem's claim is that ``alpha`` is recoverable. Check it.

Two things carry ``experiments/inverse.py``, and they are checked in that order.

1. **Sensitivity.** On the exact solution the residual with a wrong diffusivity
   is not merely "nonzero" -- it is *exactly* ``(alpha_true - alpha) u_xx``.
   That identity is the whole reason gradient descent on ``alpha`` works, it is
   available in closed form here, and it is checked as an identity rather than
   as an inequality.
2. **Recovery.** From noiseless data the optimizer really does walk a 4x-wrong
   initialization back to the true value, and with the data term removed it
   does not -- because the residual alone is satisfied by any solution of the
   PDE (``u = 0`` for every ``alpha``), so it cannot identify one.

The observation sampler is checked too, since a silent bug there (noise at the
wrong scale, points outside the requested window) would make every recovery
number in the README a measurement of something else.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from heat import ALPHA, AMPS, MODES, X_RANGE  # noqa: E402
from inverse import (  # noqa: E402
    ALPHA_INIT,
    inverse_residual,
    observations,
    train_inverse,
)
from pinn import derivatives as D  # noqa: E402


def exact_torch(coords):
    """The exact Fourier solution as a differentiable function of ``coords``.

    ``heat.heat_exact`` is numpy, so it cannot be handed to the autograd
    derivative helpers. This is the same closed form in torch, which lets the
    residual be evaluated on the *true* field instead of on a network -- the
    only way to isolate the alpha-sensitivity from any training error.
    """
    x, t = coords[:, 0:1], coords[:, 1:2]
    out = torch.zeros_like(x)
    for k, a in zip(MODES, AMPS):
        out = out + a * torch.sin(k * np.pi * x) * torch.exp(
            -ALPHA * (k * np.pi) ** 2 * t
        )
    return out


def grid_coords(n=41):
    x = torch.linspace(0.05, 0.95, n)
    t = torch.linspace(0.05, 0.95, n)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    coords = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    coords.requires_grad_(True)
    return coords


@pytest.mark.parametrize("alpha", [0.01, 0.03, 0.05, 0.2])
def test_the_residual_with_a_wrong_alpha_is_exactly_the_alpha_error_times_u_xx(alpha):
    """``u_t - alpha u_xx = (ALPHA - alpha) u_xx`` on the true solution.

    Because the true field satisfies ``u_t = ALPHA u_xx`` identically, the
    residual measured with any other diffusivity is a *known multiple* of
    ``u_xx``. This is the gradient signal the recovery uses, written down.
    """
    coords = grid_coords()
    u = exact_torch(coords)
    log_alpha = torch.tensor(float(np.log(alpha)))
    r = inverse_residual(u, coords, log_alpha).detach()
    u_xx = D.u_xx(exact_torch(coords), coords).detach()
    want = (ALPHA - alpha) * u_xx
    assert torch.allclose(r, want, atol=1e-4)
    if alpha == ALPHA:
        assert float(r.abs().max()) < 1e-4  # the true value leaves no residual
    else:
        assert float(r.abs().max()) > 1e-2  # and a wrong one leaves a large one


def test_the_residual_signal_grows_linearly_in_the_alpha_error():
    """A corollary worth asserting separately: the sensitivity does not
    saturate, so there is a usable gradient even far from the answer."""
    coords = grid_coords()
    u = exact_torch(coords)
    norms = []
    for alpha in (0.06, 0.07, 0.09):
        r = inverse_residual(u, coords, torch.tensor(float(np.log(alpha)))).detach()
        norms.append(float(r.pow(2).mean().sqrt()))
    # RMS residual proportional to |ALPHA - alpha| = 0.01, 0.02, 0.04
    assert norms[1] / norms[0] == pytest.approx(2.0, rel=0.02)
    assert norms[2] / norms[0] == pytest.approx(4.0, rel=0.02)


def test_observations_lie_in_the_requested_window_with_the_requested_noise():
    from heat import heat_exact

    gen = torch.Generator().manual_seed(0)
    coords, y = observations(4000, sigma=0.05, gen=gen, t_max=0.4)
    assert coords.shape == (4000, 2) and y.shape == (4000, 1)
    assert not coords.requires_grad  # data, not collocation
    assert float(coords[:, 0].min()) >= X_RANGE[0] and float(coords[:, 0].max()) <= X_RANGE[1]
    assert float(coords[:, 1].min()) >= 0.0 and float(coords[:, 1].max()) <= 0.4
    clean = torch.tensor(
        heat_exact(coords[:, 0:1].numpy(), coords[:, 1:2].numpy()), dtype=torch.float32
    ).reshape(-1, 1)
    resid = (y - clean).flatten()
    # 4000 draws -> s.e. of the sd estimate is sigma/sqrt(2n) ~ 0.0006
    assert float(resid.std()) == pytest.approx(0.05, rel=0.05)
    assert abs(float(resid.mean())) < 0.005


def test_noiseless_observations_are_exactly_the_solution():
    from heat import heat_exact

    coords, y = observations(64, sigma=0.0, gen=torch.Generator().manual_seed(1))
    clean = torch.tensor(
        heat_exact(coords[:, 0:1].numpy(), coords[:, 1:2].numpy()), dtype=torch.float32
    ).reshape(-1, 1)
    assert torch.allclose(y, clean, atol=1e-6)


def test_observations_are_reproducible_under_a_seeded_generator():
    a = observations(32, 0.03, torch.Generator().manual_seed(7))
    b = observations(32, 0.03, torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_alpha_is_recovered_from_noiseless_data():
    """End to end, from a deliberately 4x-wrong start.

    Small and short on purpose (32 wide, 2000 steps, ~10 s) -- the README's
    numbers come from the full runs; this asserts the mechanism works at all,
    which is what a regression here would break.
    """
    _, alpha_hat, history = train_inverse(
        n_obs=400, sigma=0.0, steps=2000, n_interior=800, width=32, lr=5e-3
    )
    assert alpha_hat == pytest.approx(ALPHA, rel=0.10)
    assert abs(alpha_hat - ALPHA) < abs(ALPHA_INIT - ALPHA) / 10  # it moved, a lot
    assert history[-1][3] < 0.05  # and the field it came with solves the PDE


def test_without_the_data_term_alpha_is_not_identified():
    """The control that says the data is doing the work.

    Set ``w_data = 0`` and the objective is the residual alone, which every
    solution of the heat equation satisfies for *its own* alpha -- including
    ``u = 0``, which satisfies it for all of them. The optimizer duly finds a
    flat field and leaves alpha near its initialization.
    """
    _, alpha_hat, history = train_inverse(
        n_obs=400, sigma=0.0, steps=2000, n_interior=800, width=32, lr=5e-3, w_data=0.0
    )
    assert abs(alpha_hat - ALPHA) / ALPHA > 1.0  # nowhere near recovered
    assert history[-1][3] > 0.5  # and the field is not the solution either


def test_alpha_stays_positive_through_the_log_parameterization():
    """A large step in ``log alpha`` cannot make ``alpha`` negative, which is
    the reason for the parameterization (a plain parameter can cross zero and
    turn the heat equation backwards -- ill-posed, and it shows up as a NaN)."""
    log_alpha = torch.nn.Parameter(torch.tensor(float(np.log(ALPHA_INIT))))
    opt = torch.optim.Adam([log_alpha], lr=10.0)
    for _ in range(5):
        opt.zero_grad()
        loss = torch.exp(log_alpha) * 3.0  # pushes alpha down as hard as it can
        loss.backward()
        opt.step()
    assert float(torch.exp(log_alpha)) > 0.0
    assert float(torch.exp(log_alpha)) < ALPHA_INIT  # it did move downward
