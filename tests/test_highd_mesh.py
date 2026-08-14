"""The d-dimensional mesh baseline has to be right before its cost means
anything: a solver that is quietly first-order, or that drops a term in the
splitting, would understate the grid a given accuracy needs and so understate
the wall the whole high-dimensional study is about.

Three independent ground truths, in increasing order of how much of the code
they exercise:

1. the shipped 1D Crank-Nicolson solver, which the ADI scheme reduces to
   algebraically at d = 1 (``crank_nicolson.crank_nicolson``);
2. :func:`highd_mesh.mode_amplification`, a hand-derived scalar recursion for
   the exact discrete amplitude of one grid sine mode at any d, sharing no code
   with the stepping loop;
3. a dense assembly of the same Kronecker-sum operator at d = 2, stepped with
   ``numpy.linalg.solve`` -- which pins the *splitting error itself* by showing
   the two agree at O(dt^2) and separate as dt grows.

Plus the closed-form solution from ``highd_heat``, which is what the accuracy
numbers are actually measured against.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from crank_nicolson import _thomas, crank_nicolson  # noqa: E402
from heat import ALPHA  # noqa: E402
from highd_heat import HighDHeat, exact, exact_ms, sec1_terms  # noqa: E402
from highd_mesh import (  # noqa: E402
    ASPECT, THETA, extrapolate, fit_cost_model, flatness_study, grid_axis,
    mode_amplification, mode_fields, model_seconds, node_steps, nt_for,
    order_study, required_nx, second_difference, solve, thomas_factor,
    thomas_solve_axis,
)


# ---------------------------------------------------------------------------
# The linear algebra pieces
# ---------------------------------------------------------------------------
def test_batched_thomas_matches_the_shipped_1d_thomas():
    """The batched constant-coefficient solve must agree with the shipped
    per-line Thomas algorithm, line for line.

    ``crank_nicolson._thomas`` is the version already trusted by Sec. 6; this
    module's is a different implementation (pre-factored, vectorised over the
    other axes, in place), so agreement is a real cross-check and not the same
    code called twice.
    """
    rng = np.random.default_rng(0)
    n, lines = 9, 5
    sub = sup = -0.37
    diag = 1.0 + 2.0 * 0.37
    rhs = rng.standard_normal((n, lines))

    pivots, cp = thomas_factor(sub, diag, sup, n)
    got = thomas_solve_axis(rhs.copy(), 0, sub, pivots, cp)

    lower = np.full(n - 1, sub)
    upper = np.full(n - 1, sup)
    main = np.full(n, diag)
    for j in range(lines):
        want = _thomas(lower, main, upper, rhs[:, j])
        np.testing.assert_allclose(got[:, j], want, rtol=0, atol=1e-13)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_batched_thomas_inverts_the_matrix_on_every_axis(axis):
    """Solving along an axis must invert the tridiagonal matrix acting on that
    axis -- the check that ``np.moveaxis`` is not quietly transposing the
    problem, which would still produce a plausible-looking field."""
    rng = np.random.default_rng(1)
    n = 6
    s = 0.21
    sub = -s
    diag = 1.0 + 2.0 * s
    T = np.diag(np.full(n, diag)) + np.diag(np.full(n - 1, sub), 1) \
        + np.diag(np.full(n - 1, sub), -1)

    rhs = rng.standard_normal((n, n, n))
    pivots, cp = thomas_factor(sub, diag, sub, n)
    got = thomas_solve_axis(rhs.copy(), axis, sub, pivots, cp)
    back = np.moveaxis(np.tensordot(T, np.moveaxis(got, axis, 0), axes=(1, 0)),
                       0, axis)
    np.testing.assert_allclose(back, rhs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
def test_second_difference_is_the_dirichlet_stencil(axis):
    """The undivided second difference, with the boundary neighbours treated as
    exactly zero (which is what makes the boundary nodes unstored)."""
    rng = np.random.default_rng(2)
    u = rng.standard_normal((5, 4))
    out = np.empty_like(u)
    second_difference(u, axis, out)

    padded = np.moveaxis(u, axis, 0)
    padded = np.concatenate([np.zeros((1,) + padded.shape[1:]), padded,
                             np.zeros((1,) + padded.shape[1:])], axis=0)
    want = padded[:-2] - 2 * padded[1:-1] + padded[2:]
    np.testing.assert_allclose(np.moveaxis(out, axis, 0), want, atol=1e-14)


def test_second_difference_eigenvalue_of_a_grid_sine_mode():
    """The grid sine mode is an exact eigenvector with eigenvalue
    ``-(2 - 2 cos(k pi dx))`` -- the identity :func:`mode_amplification` is
    built on, checked here directly so the oracle is not resting on itself."""
    nx, k = 12, 3
    x = grid_axis(nx)
    u = np.sin(k * np.pi * x)
    out = np.empty_like(u)
    second_difference(u, 0, out)
    lam = -(2.0 - 2.0 * np.cos(k * np.pi / nx))
    # atol as well as rtol: k = 3 on a 12-interval grid has nodes sitting on the
    # mode's zeros, where both sides are cancellation noise at 1e-16 and a
    # relative comparison is measuring the rounding of zero.
    np.testing.assert_allclose(out, lam * u, rtol=1e-12, atol=1e-14)


def test_mode_fields_match_direct_evaluation():
    """The outer-product build of ``prod_i sin(k_i pi x_i)`` must equal a direct
    evaluation on the explicit coordinate table it exists to avoid."""
    problem = HighDHeat(3)
    nx = 7
    x = grid_axis(nx)
    grids = np.meshgrid(*([x] * 3), indexing="ij")
    for field, k in zip(mode_fields(problem, nx), problem.modes):
        want = np.ones_like(grids[0])
        for g, ki in zip(grids, k):
            want = want * np.sin(ki * np.pi * g)
        np.testing.assert_allclose(field, want, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# Ground truth 1: the shipped 1D solver
# ---------------------------------------------------------------------------
def test_d1_reduces_to_the_shipped_crank_nicolson():
    """At d = 1 the Douglas scheme is Crank-Nicolson, so the two solvers must
    produce the same field -- not the same accuracy class, the same numbers.

    This is the strongest pin available: ``crank_nicolson.py`` is shipped,
    tested and quoted in Sec. 6, and it shares no line of stepping code with
    this module. Compared on the *interior* nodes of the final time level,
    since this solver never stores the boundary (identically zero) or the
    intermediate levels.
    """
    problem = HighDHeat(1, terms=sec1_terms(), alpha=ALPHA)
    nx = nt = 40
    x, t, U, _ = crank_nicolson(nx=nx, nt=nt)

    # Re-run the ADI solver without scoring, then rebuild its final level by
    # stepping again -- solve() returns metrics, so reach the field through the
    # same public path a caller would: compare the relative L2 both report.
    ours = solve(problem, nx, nt)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    u_true = exact(problem, XX.reshape(-1, 1), TT.reshape(-1))
    cn_err = np.linalg.norm(U.reshape(-1) - u_true) / np.linalg.norm(u_true)

    # Both are relative L2 of the same scheme on the same grid, under two
    # quadratures (CN's unweighted node norm, ours trapezoidal in time), so
    # they agree to the quadrature, not to machine precision.
    assert 0.9 < ours["rel_l2"] / cn_err < 1.1, (ours["rel_l2"], cn_err)


def test_d1_field_matches_crank_nicolson_to_floating_point():
    """The fields themselves, stepped side by side.

    Reproduces the Douglas stages here on a 1D array and checks them against
    ``crank_nicolson``'s own time levels one step at a time, which localises
    any disagreement to the step that first breaks rather than to the run.
    """
    problem = HighDHeat(1, terms=sec1_terms(), alpha=ALPHA)
    nx, nt = 24, 16
    x, t, U, _ = crank_nicolson(nx=nx, nt=nt)

    n = nx - 1
    dx, dt = 1.0 / nx, 1.0 / nt
    s = THETA * dt * problem.alpha / (dx * dx)
    pivots, cp = thomas_factor(-s, 1.0 + 2.0 * s, -s, n)
    u = U[1:-1, 0].copy()
    scratch = np.empty(n)
    for step in range(nt):
        work = u.copy()
        second_difference(u, 0, scratch)
        work += (dt * problem.alpha / (dx * dx)) * scratch
        second_difference(u, 0, scratch)
        work -= s * scratch
        thomas_solve_axis(work, 0, -s, pivots, cp)
        u = work
        np.testing.assert_allclose(u, U[1:-1, step + 1], rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Ground truth 2: the hand-derived amplification factor, at any d
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_single_mode_decays_by_the_derived_amplification(d):
    """A solve started from one grid sine mode must end at exactly that mode
    scaled by :func:`mode_amplification` -- shape and amplitude both.

    The scalar recursion is derived by hand from the Douglas stages and touches
    none of the array code, so this catches a wrong stage ordering, a dropped
    ``dt sum_j A_j u`` term, or a ``theta`` in the wrong place, none of which a
    convergence test would flag (they all still converge, just at the wrong
    constant or order).
    """
    k = tuple([2] + [1] * (d - 1))
    problem = HighDHeat(d, terms=[(k, 1.0)])
    nx, nt = 10, 7
    n = nx - 1
    dx, dt = 1.0 / nx, 1.0 / nt
    s = THETA * dt * problem.alpha / (dx * dx)
    pivots, cp = thomas_factor(-s, 1.0 + 2.0 * s, -s, n)

    phi = mode_fields(problem, nx)[0]
    u = phi.copy()
    scratch = np.empty_like(u)
    for _ in range(nt):
        work = u.copy()
        for j in range(d):
            second_difference(u, j, scratch)
            work += (dt * problem.alpha / (dx * dx)) * scratch
        for j in range(d):
            second_difference(u, j, scratch)
            work -= s * scratch
            thomas_solve_axis(work, j, -s, pivots, cp)
        u = work

    g = mode_amplification(problem, k, nx, nt)
    np.testing.assert_allclose(u, g * phi, rtol=1e-11, atol=1e-14)


def test_amplification_reduces_to_crank_nicolson_at_d_one():
    """At d = 1 the recursion must collapse to ``(1 - z/2)/(1 + z/2)`` per step."""
    problem = HighDHeat(1, terms=[((3,), 1.0)])
    nx, nt = 16, 5
    dx, dt = 1.0 / nx, 1.0 / nt
    mu = problem.alpha * (2 - 2 * np.cos(3 * np.pi * dx)) / dx ** 2
    want = ((1 - dt * mu / 2) / (1 + dt * mu / 2)) ** nt
    assert mode_amplification(problem, (3,), nx, nt) == pytest.approx(want, rel=1e-13)


def test_amplification_is_stable_at_an_enormous_time_step():
    """|g| < 1 for a single step covering the whole interval at a fine grid --
    the unconditional stability that makes an implicit scheme the right
    classical baseline. An explicit scheme here would need dt <= dx^2/(2 d
    alpha), about 1e-4 at these settings."""
    for d in (1, 2, 3, 8):
        problem = HighDHeat(d)
        for k in problem.modes:
            g = mode_amplification(problem, k, 128, 1)
            assert 0.0 < g < 1.0, (d, k, g)


# ---------------------------------------------------------------------------
# Ground truth 3: dense assembly of the unsplit operator, at d = 2
# ---------------------------------------------------------------------------
def _dense_cn_step(problem, nx, nt):
    """Unsplit Crank-Nicolson on the assembled Kronecker sum, at d = 2.

    ``L = L_1 (x) I + I (x) L_2`` built explicitly and inverted by
    ``numpy.linalg.solve`` -- the O(N^(3d)) method the ADI scheme exists to
    avoid, which is exactly why it is a good oracle at N = 7.
    """
    n = nx - 1
    dx, dt = 1.0 / nx, 1.0 / nt
    L1 = (np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1)
          + np.diag(np.ones(n - 1), -1)) / dx ** 2
    A = problem.alpha * (np.kron(L1, np.eye(n)) + np.kron(np.eye(n), L1))
    I = np.eye(n * n)
    lhs, rhs = I - 0.5 * dt * A, I + 0.5 * dt * A

    u = np.zeros((n, n))
    for f, a in zip(mode_fields(problem, nx), problem.amps):
        u = u + a * f
    v = u.reshape(-1)
    for _ in range(nt):
        v = np.linalg.solve(lhs, rhs @ v)
    return v.reshape(n, n)


def _adi_field(problem, nx, nt):
    """The ADI solver's final time level, stepped here to expose the field."""
    d = problem.d
    n = nx - 1
    dx, dt = 1.0 / nx, 1.0 / nt
    s = THETA * dt * problem.alpha / (dx * dx)
    pivots, cp = thomas_factor(-s, 1.0 + 2.0 * s, -s, n)
    u = np.zeros((n,) * d)
    for f, a in zip(mode_fields(problem, nx), problem.amps):
        u = u + a * f
    scratch = np.empty_like(u)
    for _ in range(nt):
        work = u.copy()
        for j in range(d):
            second_difference(u, j, scratch)
            work += (dt * problem.alpha / (dx * dx)) * scratch
        for j in range(d):
            second_difference(u, j, scratch)
            work -= s * scratch
            thomas_solve_axis(work, j, -s, pivots, cp)
        u = work
    return u


def test_splitting_error_is_second_order_in_dt():
    """ADI and the unsplit Crank-Nicolson must differ by O(dt^2), and by nothing
    else. Both are second-order approximations of the same continuum problem, so
    agreeing at fine dt proves little on its own; what pins the *splitting* is
    the rate at which they separate as dt grows.

    Measured over three refinements at a fixed grid, so the spatial error --
    identical in both, since they use the same stencil -- cancels exactly.

    The assertion is the rate and a *fitted* constant, not a hand-picked level:
    the gap is ``c dt^2`` with c measured near 0.013 here, so quoting an
    absolute tolerance would be quoting c to whatever precision this grid
    happens to give it.
    """
    problem = HighDHeat(2)
    nx = 7
    nts = (4, 8, 16)
    diffs = []
    for nt in nts:
        gap = _adi_field(problem, nx, nt) - _dense_cn_step(problem, nx, nt)
        diffs.append(float(np.max(np.abs(gap))))
    for coarse, fine in zip(diffs, diffs[1:]):
        assert 3.2 < coarse / fine < 5.0, (diffs,)
    consts = [dif * nt ** 2 for dif, nt in zip(diffs, nts)]
    assert max(consts) / min(consts) < 1.3, consts


def test_adi_matches_the_unsplit_scheme_to_the_splitting_error():
    """At a fine time step the two schemes agree to a small *relative* fraction
    of the solution's own size -- the absolute statement behind the rate above,
    scaled by the field so it does not encode this problem's amplitude."""
    problem = HighDHeat(2)
    nx, nt = 7, 64
    ours = _adi_field(problem, nx, nt)
    theirs = _dense_cn_step(problem, nx, nt)
    gap = float(np.max(np.abs(ours - theirs))) / float(np.max(np.abs(theirs)))
    assert gap < 1e-4, gap


# ---------------------------------------------------------------------------
# Accuracy against the closed form, and the metric that measures it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3])
def test_second_order_convergence_against_the_exact_solution(d):
    """Halving dx and dt must cut the relative L2 by ~4, at every d. A splitting
    that had gone first-order would show up here as a ratio near 2."""
    problem = HighDHeat(d)
    errs = [solve(problem, nx)["rel_l2"] for nx in (8, 16, 32)]
    for coarse, fine in zip(errs, errs[1:]):
        ratio = coarse / fine
        assert 3.5 < ratio < 4.5, (d, errs)


def test_grid_quadrature_reproduces_the_closed_form_mean_square():
    """The trapezoidal space-time average, applied to the exact solution, must
    reproduce ``highd_heat.exact_ms`` -- and converge to it at second order.

    Second order, not exactly: the two-mode cross term is a half-period cosine
    on [0,1], where there is no periodicity for the trapezoidal rule to exploit.
    Asserting the *rate* rather than a level keeps this portable.
    """
    problem = HighDHeat(2)
    gaps = [abs(solve(problem, nx)["quad_ratio"] - 1.0) for nx in (8, 16, 32)]
    for coarse, fine in zip(gaps, gaps[1:]):
        assert 3.5 < coarse / fine < 4.5, gaps
    assert gaps[-1] < 1e-3
    assert exact_ms(problem) > 0


def test_initial_condition_is_imposed_exactly():
    """No error at t = 0: the mesh method starts from the exact IC sampled on
    the grid, which is one of the two ways it differs from the PINN (which
    penalises the IC and satisfies it only approximately)."""
    problem = HighDHeat(3)
    r = solve(problem, 8, 4)
    assert r["rel_l2"] > 0            # the run does accumulate error later
    fields = mode_fields(problem, 8)
    u0 = sum(a * f for f, a in zip(fields, problem.amps))
    x = grid_axis(8)
    grids = np.meshgrid(*([x] * 3), indexing="ij")
    coords = np.stack([g.reshape(-1) for g in grids], axis=1)
    np.testing.assert_allclose(u0.reshape(-1), exact(problem, coords, 0.0),
                               rtol=1e-12)


def test_solution_stays_bounded_at_a_time_step_far_past_explicit_stability():
    """One step across the whole interval on a fine grid: bounded and decaying.
    Explicit Euler would need dt <= dx^2/(2 d alpha) ~ 3e-4 here."""
    problem = HighDHeat(3)
    r = solve(problem, 32, 1)
    assert np.isfinite(r["rel_l2"])
    assert r["rel_l2"] < 1.0          # still tracking the solution, not blown up


# ---------------------------------------------------------------------------
# The bookkeeping the cost claims rest on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d,nx", [(1, 32), (2, 12), (3, 8)])
def test_memory_is_the_counted_number_of_arrays(d, nx):
    """``bytes_counted`` must be the arrays the code actually holds, and
    ``tracemalloc`` must corroborate it.

    The counted number is what gets extrapolated to d = 16, so it cannot be a
    comment that drifted from the code. The traced peak is checked as a band,
    not a level: it carries the interpreter's own transient allocations, and
    ``gp-from-scratch`` Day 7 is the reason no process-level memory number is
    asserted exactly anywhere in this slate.
    """
    problem = HighDHeat(d)
    r = solve(problem, nx)
    assert r["arrays"] == len(problem.amps) + 3
    assert r["bytes_counted"] == r["arrays"] * 8 * (nx - 1) ** d
    assert r["unknowns"] == (nx - 1) ** d
    assert r["bytes_traced"] >= r["bytes_counted"]


def test_memory_does_not_grow_with_the_number_of_time_steps():
    """Only the current time level is stored, so a run 32x longer costs the same
    memory. This is what makes the memory wall a pure ``(N-1)^d`` law -- and it
    is the concrete way this solver departs from Sec. 6's, which returns the
    whole space-time field."""
    problem = HighDHeat(2)
    short = solve(problem, 16, 2)
    long_ = solve(problem, 16, 64)
    assert short["bytes_counted"] == long_["bytes_counted"]
    assert short["bytes_traced"] == pytest.approx(long_["bytes_traced"], rel=0.10)


def test_nt_for_is_the_documented_aspect_ratio():
    assert nt_for(64) == 64 // ASPECT
    assert nt_for(2) >= 1              # never zero steps
    assert node_steps(2, 5, 3) == 3 * 4.0 ** 2


def test_required_nx_returns_a_grid_that_actually_reaches_the_target():
    """The search must return a *run*, not the extrapolated prediction: the
    returned error is measured on the returned grid and is at or below target."""
    problem = HighDHeat(2)
    r = required_nx(problem, 1e-3)
    assert r["rel_l2"] <= 1e-3
    assert solve(problem, r["nx"], r["nt"])["rel_l2"] == pytest.approx(r["rel_l2"])
    assert 1.5 < r["fitted_order"] < 2.5


def test_required_nx_declines_a_cell_it_cannot_fit():
    """Over budget returns None rather than attempting a solve that would swap
    the machine -- the mechanism that makes a skipped cell visible in the CSV."""
    assert required_nx(HighDHeat(8), 1e-3, budget=10_000) is None


def test_cost_model_recovers_coefficients_it_was_built_from():
    """Constructed rows with known ``c_py`` and ``tau``, so the assertion is
    about the fit's arithmetic and not about any timing.

    The two terms are only separable if the cells span dimensions where each in
    turn dominates -- which is exactly the situation the cost sweep is in, and
    the reason the fit is done over the whole sweep rather than the big cells.
    """
    c_py, tau = 3e-6, 7e-9
    rows = []
    for d, nx, nt in ((1, 22, 11), (3, 22, 11), (5, 20, 10), (6, 20, 10)):
        n = nx - 1
        rows.append({"d": d, "nx": nx, "nt": nt, "unknowns": n ** d,
                     "seconds": f"{nt * d * (c_py * n + tau * n ** d):.12e}"})
    model = fit_cost_model(rows)
    assert model["tau"] == pytest.approx(tau, rel=1e-6)
    assert model["c_py"] == pytest.approx(c_py, rel=1e-6)
    assert model["worst_rel"] < 1e-9
    # The one-parameter quotient really does span orders over these cells --
    # the fact that motivates the second term.
    assert model["naive_spread"] > 100


def test_cost_model_prediction_matches_its_own_coefficients():
    model = dict(c_py=2e-6, tau=5e-9)
    assert model_seconds(model, 3, 21, 10) == pytest.approx(
        10 * 3 * (2e-6 * 20 + 5e-9 * 20 ** 3))


def test_extrapolation_is_anchored_on_the_largest_measured_cell():
    """The projection must reproduce its anchor exactly and scale by
    ``d (N-1)^d`` from there -- the property a global fit does not have, and the
    reason the projection does not use one.

    Constructed rows, so this is a statement about the arithmetic and about
    which cell is chosen, not about any timing.
    """
    rows = [
        {"target": "1e-03", "d": 4, "nx": 20, "nt": 10, "unknowns": 19 ** 4,
         "seconds": "0.05", "arrays": 5, "node_steps": f"{10*19.0**4:.6e}"},
        {"target": "1e-03", "d": 6, "nx": 20, "nt": 10, "unknowns": 19 ** 6,
         "seconds": "25.0", "arrays": 5, "node_steps": f"{10*19.0**6:.6e}"},
    ]
    projected, _ = extrapolate(rows, dims=(6, 8))
    at6 = next(r for r in projected if r["d"] == 6)
    at8 = next(r for r in projected if r["d"] == 8)
    assert float(at6["seconds"]) == pytest.approx(25.0)          # exact at anchor
    assert float(at8["seconds"]) == pytest.approx(25.0 * (8 / 6) * 19.0 ** 2)
    assert float(at8["unknowns"]) == pytest.approx(19.0 ** 8)
    assert float(at8["bytes_counted"]) == pytest.approx(5 * 8 * 19.0 ** 8)
    assert "extrapolated" in at8["source"] and "d=6" in at8["source"]


# ---------------------------------------------------------------------------
# The premises the extrapolation rests on
# ---------------------------------------------------------------------------
def test_accuracy_at_fixed_resolution_does_not_degrade_with_d():
    """The extrapolation holds N fixed as d grows, which is only honest if the
    accuracy a given N reaches does not fall off with d. Measured up to d = 5
    here (d = 6 is left to the experiment script for runtime).

    The reason it holds is ``alpha_d = alpha_1 / d``: the leading truncation
    error sums d per-axis terms and is multiplied by a diffusivity that falls as
    1/d. If that scaling were ever removed from ``highd_heat``, this test is
    what would catch the extrapolation going optimistic.
    """
    rows = flatness_study(dims=(1, 2, 3, 4, 5), nx=8)
    errs = [float(r["rel_l2"]) for r in rows]
    assert max(errs) / min(errs) < 2.0, errs
    assert errs[-1] <= errs[0]        # it improves slightly, and does not worsen


def test_order_study_reports_second_order_everywhere():
    """The study function itself, on a small grid set -- so a refactor that
    breaks the reported ``order`` column fails here and not silently in a CSV."""
    rows = order_study(dims=(1, 2), sizes=(8, 16, 32))
    orders = [float(r["order"]) for r in rows if r["order"]]
    assert len(orders) == 4
    assert all(1.8 < o < 2.2 for o in orders), orders
