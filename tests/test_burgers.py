"""The Burgers ground truth must be a genuine solution of the PDE.

Every error in ``experiments/burgers.py`` is measured against ``burgers_exact``
(a Cole-Hopf integral evaluated by Gauss-Hermite quadrature), so that closed
form is load-bearing.  We check it four ways --

1. it satisfies ``u_t + u u_x = nu u_xx`` to finite-difference accuracy on the
   *smooth* part of the domain (the viscous shock at ``x = 0`` is too steep to
   difference on any practical grid, so the residual is checked for ``|x| > 0.1``
   -- the FD residual there is a real, assumption-free check, while inside the
   band it is dominated by truncation error, which the test also documents),
2. it matches the initial condition ``-sin(pi x)`` at ``t = 0``,
3. it respects the Dirichlet boundaries *and* the internal node at ``x = 0``,
   all three of which are pinned to zero exactly by the odd symmetry of the IC,
4. it is odd in ``x`` at every time, ``u(-x, t) = -u(x, t)``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from burgers import NU, burgers_exact  # noqa: E402


def test_exact_satisfies_burgers_away_from_shock():
    # Central differences on a fine grid; start slightly after t=0 so the
    # profile is smooth, and difference strictly inside the domain.
    nx, nt = 401, 201
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.02, 1.0, nt)
    hx = x[1] - x[0]
    ht = t[1] - t[0]
    XX, TT = np.meshgrid(x, t, indexing="ij")
    u = burgers_exact(XX, TT)

    u_x = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * hx)
    u_xx = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / hx ** 2
    u_t = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * ht)
    residual = u_t + u[1:-1, 1:-1] * u_x - NU * u_xx

    x_in = x[1:-1]
    far = np.abs(x_in) > 0.1
    # Away from the shock the closed form solves the PDE to FD truncation.
    assert np.max(np.abs(residual[far, :])) < 1e-2
    # Inside the thin shock band the FD residual is large -- not because the
    # solution is wrong but because the gradient is unresolvable on this grid.
    # Documenting it makes the point of the whole experiment concrete.
    assert np.max(np.abs(residual[~far, :])) > 1.0


def test_exact_matches_initial_condition():
    x = np.linspace(-1.0, 1.0, 257)
    u0 = burgers_exact(x, np.zeros_like(x))
    assert np.allclose(u0, -np.sin(np.pi * x), atol=1e-12)


def test_exact_pins_zero_at_boundaries_and_center():
    # u = 0 at x in {-1, 0, 1} for all t, by the odd symmetry of -sin(pi x).
    t = np.linspace(0.0, 1.0, 40)
    for x0 in (-1.0, 0.0, 1.0):
        vals = burgers_exact(np.full_like(t, x0), t)
        assert np.max(np.abs(vals)) < 1e-9


def test_exact_is_odd_in_x():
    x = np.linspace(-1.0, 1.0, 101)
    for t0 in (0.1, 0.5, 1.0):
        tt = np.full_like(x, t0)
        u = burgers_exact(x, tt)
        u_flip = burgers_exact(-x, tt)
        assert np.max(np.abs(u + u_flip)) < 1e-12


def test_shock_steepens_over_time():
    # The hallmark of Burgers: the slope at x=0 grows sharply negative as the
    # advection piles the profile up faster than viscosity smears it.
    x = np.array([-1e-3, 0.0, 1e-3])
    slopes = {}
    for t0 in (0.1, 0.5):
        u = burgers_exact(x, np.full_like(x, t0))
        slopes[t0] = (u[2] - u[0]) / (x[2] - x[0])
    # steepening: |slope| at t=0.5 exceeds t=0.1, and both are strongly negative
    assert slopes[0.5] < slopes[0.1] < -1.0
    assert slopes[0.5] < -50.0


def test_pinn_training_reduces_error_on_burgers():
    # A short but real training run: the relative L2 error against the exact
    # solution must drop well below its at-init value. Kept small for speed.
    from burgers import rel_l2_error, train

    model, history = train(n_interior=2000, width=32, depth=4, steps=800, seed=0)
    steps = [h[0] for h in history]
    errs = [h[2] for h in history]
    assert steps[0] == 0 and steps[-1] == 800
    assert errs[0] > 0.9     # starts near O(1) (net output ~0 vs a unit-scale field)
    # Burgers is nonlinear and converges far slower than the smooth heat
    # problem; even this tiny run cuts the error ~2.8x. The default run
    # (experiments/burgers.py) drives it to a few percent.
    assert errs[-1] < 0.6
