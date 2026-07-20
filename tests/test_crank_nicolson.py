"""The Crank-Nicolson baseline is only an honest yardstick for the PINN if it
is itself correct: the Thomas solve must invert the tridiagonal system, and the
scheme must converge to the exact heat solution at its advertised second order
in both space and time."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crank_nicolson import _thomas, crank_nicolson, rel_l2  # noqa: E402


def test_thomas_solves_tridiagonal_system():
    """The from-scratch tridiagonal solve must match a dense linear solve on a
    random diagonally-dominant tridiagonal system."""
    rng = np.random.default_rng(0)
    n = 12
    lower = rng.uniform(-1, 1, n - 1)
    upper = rng.uniform(-1, 1, n - 1)
    diag = rng.uniform(3, 5, n)  # dominant, so the system is well-conditioned
    d = rng.standard_normal(n)

    T = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    got = _thomas(lower, diag, upper, d)
    np.testing.assert_allclose(got, np.linalg.solve(T, d), atol=1e-12)


def test_cn_matches_exact_on_fine_grid():
    """On a fine grid CN should reproduce the exact three-mode solution to a
    small relative error -- the property that makes it a trustworthy baseline."""
    x, t, U, _ = crank_nicolson(nx=160, nt=160)
    assert rel_l2(x, t, U) < 2e-3


def test_cn_respects_ic_and_bcs():
    """Initial condition reproduced exactly at t=0; homogeneous Dirichlet BCs
    hold at every time level (boundary rows are pinned, never solved)."""
    x, t, U, _ = crank_nicolson(nx=40, nt=40)
    from crank_nicolson import _ic_vector

    np.testing.assert_allclose(U[:, 0], _ic_vector(x), atol=1e-12)
    np.testing.assert_allclose(U[0, :], 0.0, atol=1e-14)
    np.testing.assert_allclose(U[-1, :], 0.0, atol=1e-14)


def test_cn_is_second_order():
    """Halving both dx and dt must cut the relative L2 error by ~4x (the scheme
    is O(dx^2) + O(dt^2)). We refine twice and check both ratios sit in a band
    around 4."""
    errs = []
    for m in (20, 40, 80):
        x, t, U, _ = crank_nicolson(nx=m, nt=m)
        errs.append(rel_l2(x, t, U))
    for coarse, fine in zip(errs, errs[1:]):
        ratio = coarse / fine
        assert 3.0 < ratio < 5.0, f"convergence ratio {ratio:.2f} not ~4 (2nd order)"


def test_cn_is_unconditionally_stable():
    """A time step far larger than the explicit-Euler stability limit
    dt <= dx^2 / (2 alpha) must NOT blow up (bounded, and still converging) --
    the reason CN is the right classical baseline here. With nt=5 and nx=80 the
    step is ~orders above that limit."""
    x, t, U, _ = crank_nicolson(nx=80, nt=5)
    assert np.all(np.isfinite(U))
    assert np.max(np.abs(U)) < 2.0  # IC amplitude ~1.75; no explosive growth
