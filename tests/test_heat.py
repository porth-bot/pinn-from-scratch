"""The heat-equation ground truth must be a genuine solution of the PDE.

Every error in ``experiments/heat.py`` is measured against ``heat_exact``, so
that closed form is load-bearing: if it does not actually solve the PDE, the
whole experiment is comparing the PINN to the wrong thing. We therefore check
``heat_exact`` three ways --

1. it satisfies ``u_t = alpha u_xx`` to finite-difference accuracy on an
   interior grid (the general, assumption-free check),
2. it matches the initial condition ``sum_k a_k sin(k pi x)`` at ``t = 0``, and
3. it vanishes on the Dirichlet boundaries ``x = 0`` and ``x = 1``.

The finite-difference check uses a fine grid and central differences; the
tolerance is set by the ``O(h^2)`` truncation error, not by the PINN.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from heat import ALPHA, AMPS, MODES, heat_exact  # noqa: E402


def test_exact_satisfies_heat_equation_by_finite_difference():
    # Fine interior grid on (0, 1) x (0, 1); central differences need neighbours
    # on both sides, so we evaluate the residual strictly inside.
    nx, nt = 201, 201
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    hx = x[1] - x[0]
    ht = t[1] - t[0]
    XX, TT = np.meshgrid(x, t, indexing="ij")
    u = heat_exact(XX, TT)

    # central second difference in x, central first difference in t
    u_xx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / hx ** 2
    u_t = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * ht)
    residual = u_t - ALPHA * u_xx

    # O(h^2) truncation; the field amplitude is ~1, so this is a real check.
    assert np.max(np.abs(residual)) < 1e-3


def test_exact_matches_initial_condition():
    x = np.linspace(0.0, 1.0, 257)
    u0 = heat_exact(x, np.zeros_like(x))
    ic = sum(a * np.sin(k * np.pi * x) for k, a in zip(MODES, AMPS))
    assert np.allclose(u0, ic, atol=1e-12)


def test_exact_respects_dirichlet_boundaries():
    t = np.linspace(0.0, 1.0, 50)
    left = heat_exact(np.zeros_like(t), t)
    right = heat_exact(np.ones_like(t), t)
    assert np.max(np.abs(left)) < 1e-12
    assert np.max(np.abs(right)) < 1e-12


def test_modes_decay_at_the_right_rate():
    # Each mode k is multiplied by exp(-alpha (k pi)^2 t); isolate one mode by
    # projecting the field onto sin(k pi x) via a fine quadrature.
    x = np.linspace(0.0, 1.0, 4001)
    dx = x[1] - x[0]
    for k, a in zip(MODES, AMPS):
        for t0 in (0.0, 0.3, 0.7):
            u = heat_exact(x, np.full_like(x, t0))
            # <u, sin(k pi x)> / <sin, sin> = a_k exp(-alpha (k pi)^2 t).
            # Trapezoid on a uniform grid; the integrand vanishes at both ends
            # (sin(0) = sin(k pi) = 0), so the endpoint correction is zero.
            integrand = u * np.sin(k * np.pi * x)
            coeff = 2.0 * dx * (integrand.sum() - 0.5 * (integrand[0] + integrand[-1]))
            expected = a * np.exp(-ALPHA * (k * np.pi) ** 2 * t0)
            assert abs(coeff - expected) < 1e-3


def test_pinn_training_reduces_error_on_heat_equation():
    # A short but real training run: the relative L2 error against the exact
    # solution must drop well below its at-init value. Kept small so it runs in
    # a few seconds inside the test suite.
    from heat import rel_l2_error, train

    model, history = train(n_interior=800, width=32, steps=400, seed=0)
    steps = [h[0] for h in history]
    errs = [h[2] for h in history]
    assert steps[0] == 0 and steps[-1] == 400
    # started near O(1), ends much smaller
    assert errs[0] > 0.3
    assert errs[-1] < 0.1
