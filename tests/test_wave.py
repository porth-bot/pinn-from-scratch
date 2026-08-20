"""The wave equation's ground truth, its energy, and the run that is scored on it.

`experiments/wave.py` scores a PINN against d'Alembert rather than against a
truncated Fourier series, so the burden of proof sits on d'Alembert. That is
where most of this file goes:

1. **The odd 2-periodic extension**, from its two symmetries -- which is what
   makes both fixed ends hold for all time -- and then the solution built from
   it, checked against the initial displacement, the zero initial velocity, both
   boundary conditions, and the PDE itself by central differences away from the
   corners (where the strong form does not exist and a stencil across it would
   be reporting the mesh).
2. **d'Alembert against separation of variables**, which shares no code with it.
3. **The conserved energy**, in closed form against the series and against
   quadrature of the exact solution -- and its conservation in time, which is
   the property that makes the measured energy ratio a real diagnostic.
4. **The trivial solution scores exactly 1.0**, so every relative error in the
   section has a baseline that is not a matter of opinion.
5. One real training cell end to end, and the committed logs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import wave as W  # noqa: E402
from common import read_csv  # noqa: E402
from pinn.model import MLP  # noqa: E402

TINY = dict(n_interior=300, n_ic=64, n_bc=64, width=16, depth=2, steps=24,
            eval_every=8)


class Zero(torch.nn.Module):
    """A network that outputs 0 everywhere -- the trivial solution, as a model.

    Written as ``0 * sum(coords^2)`` rather than ``torch.zeros`` or
    ``0 * sum(coords)``, and the square is the point: the residual takes *second*
    derivatives, and the first derivative of a linear function is a constant with
    no ``grad_fn``, so the second ``autograd.grad`` call raises. A quadratic that
    happens to be zero stays twice-differentiable, and the failure it avoids has
    nothing to do with the quantity under test.
    """

    def forward(self, coords):
        return 0.0 * (coords ** 2).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# 1. the extension and the solution it builds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ic", list(W.ICS))
def test_extension_is_odd_and_two_periodic(ic):
    y = np.linspace(-5, 5, 2001)
    F = W.odd_extension(y, ic)
    assert np.allclose(W.odd_extension(y + 2.0, ic), F, atol=1e-12)
    assert np.allclose(W.odd_extension(-y, ic), -F, atol=1e-12)
    # and it agrees with f0 on [0, 1], which is the only place f0 is defined
    x = np.linspace(0, 1, 501)
    assert np.allclose(W.odd_extension(x, ic), W.f0(x, ic), atol=1e-12)


@pytest.mark.parametrize("ic", list(W.ICS))
def test_extension_is_odd_about_x_equals_one(ic):
    """The second symmetry, which is the one that gives ``u(1,t) = 0``.

    ``F(2 - s) = -F(s)`` follows from 2-periodicity plus oddness, but it is the
    step the docstring's boundary-condition argument actually rests on, so it is
    checked rather than inferred.
    """
    s = np.linspace(-3, 3, 1201)
    assert np.allclose(W.odd_extension(2.0 - s, ic),
                       -W.odd_extension(s, ic), atol=1e-12)


@pytest.mark.parametrize("ic", list(W.ICS))
def test_dalembert_satisfies_the_data(ic):
    x = np.linspace(0, 1, 401)
    t = np.linspace(0, W.T_RANGE[1], 41)
    assert np.allclose(W.wave_exact(x, np.zeros_like(x), ic), W.f0(x, ic),
                       atol=1e-12)
    assert np.allclose(W.wave_exact(np.zeros_like(t), t, ic), 0.0, atol=1e-12)
    assert np.allclose(W.wave_exact(np.ones_like(t), t, ic), 0.0, atol=1e-12)
    h = 1e-6
    vel = (W.wave_exact(x, np.full_like(x, h), ic)
           - W.wave_exact(x, np.full_like(x, -h), ic)) / (2 * h)
    assert np.max(np.abs(vel)) < 1e-9


@pytest.mark.parametrize("ic", list(W.ICS))
def test_dalembert_satisfies_the_pde_away_from_corners(ic):
    """``u_tt = c^2 u_xx`` by central differences, corners excluded on purpose.

    The plucked string's second derivative is a delta at the travelling corner,
    so a difference stencil straddling it measures ``1/h^2`` and nothing else.
    Excluding a band is not a weakening of the test, it is the statement that
    the PDE holds in the strong form only off the characteristics -- which is
    the section's whole reason for existing.
    """
    assert W._fd_residual(ic) < 1e-6


def test_fd_residual_would_catch_a_wrong_wave_speed(monkeypatch):
    """The residual check has teeth: change c and it fails."""
    monkeypatch.setattr(W, "C", 1.7)
    assert W._fd_residual("sine") > 1.0


@pytest.mark.parametrize("ic", list(W.ICS))
def test_dalembert_agrees_with_separation_of_variables(ic):
    """Two independent constructions of the same solution.

    ``sine`` must agree to machine precision at any mode count -- it is one
    mode. ``pluck``'s series converges like 1/k^2, so the check is that the rms
    gap *falls* as modes are added, at a rate the coefficients predict, rather
    than that it is small at one count.
    """
    x = np.linspace(0.0, 1.0, 401)
    t = np.array([0.0, 0.25, 0.6, 1.3])
    ref = lambda n: np.stack(
        [W.fourier_reference(x, np.full_like(x, tv), ic, n_modes=n) for tv in t],
        axis=1)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    exact = W.wave_exact(XX, TT, ic)
    if ic == "sine":
        assert np.max(np.abs(ref(5) - exact)) < 1e-12
        return
    gaps = [float(np.sqrt(np.mean((ref(n) - exact) ** 2)))
            for n in (50, 200, 800)]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


def test_pluck_coefficients_match_a_numerical_projection():
    """The hand-integrated ``b_k`` against a quadrature of the same integral."""
    k, b = W.fourier_coefficients("pluck", 12)
    x = np.linspace(0, 1, 200_001)
    f = W.f0(x, "pluck")
    for kk, bb in zip(k, b):
        num = 2 * np.trapezoid(f * np.sin(kk * np.pi * x), x) \
            if hasattr(np, "trapezoid") else \
            2 * np.trapz(f * np.sin(kk * np.pi * x), x)
        assert num == pytest.approx(bb, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. the energy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ic", list(W.ICS))
def test_exact_energy_matches_the_series_and_a_quadrature(ic):
    k, b = W.fourier_coefficients(ic, 200_000)
    series = W.C ** 2 * np.pi ** 2 / 4 * np.sum(b ** 2 * k ** 2)
    assert series == pytest.approx(W.exact_energy(ic), rel=1e-4)
    # and against a direct quadrature of (c^2/2) int f'^2 at t = 0
    x = np.linspace(0, 1, 400_001)
    fp = np.gradient(W.f0(x, ic), x)
    inner = x[10:-10]                      # drop the one-sided end stencils
    quad = 0.5 * W.C ** 2 * (
        np.trapezoid(fp[10:-10] ** 2, inner) if hasattr(np, "trapezoid")
        else np.trapz(fp[10:-10] ** 2, inner))
    assert quad == pytest.approx(W.exact_energy(ic), rel=2e-3)


@pytest.mark.parametrize("ic", list(W.ICS))
def test_energy_is_conserved_by_the_exact_solution(ic):
    """The property that makes the measured ratio a diagnostic, not a fit.

    Differentiating the exact solution on a grid and integrating gives the same
    energy at every time -- to the accuracy of the stencil, which for the pluck
    is limited by the corner and so is checked with a loose tolerance and a
    stated reason rather than a tight one that hides where it comes from.
    """
    x = np.linspace(0, 1, 20_001)
    tol = 0.02 if ic == "sine" else 0.10
    for t in (0.0, 0.37, 0.91, 1.55):
        u = W.wave_exact(x, np.full_like(x, t), ic)
        up = W.wave_exact(x, np.full_like(x, t + 1e-5), ic)
        um = W.wave_exact(x, np.full_like(x, t - 1e-5), ic)
        ux = np.gradient(u, x)
        ut = (up - um) / 2e-5
        dens = 0.5 * (ut ** 2 + W.C ** 2 * ux ** 2)
        e = (np.trapezoid(dens[5:-5], x[5:-5]) if hasattr(np, "trapezoid")
             else np.trapz(dens[5:-5], x[5:-5]))
        assert e == pytest.approx(W.exact_energy(ic), rel=tol)


def test_energy_by_modes_is_increasing_and_reaches_one():
    fracs = [W.energy_by_modes("pluck", n) for n in (1, 2, 5, 20, 200, 20_000)]
    assert fracs == sorted(fracs)
    assert fracs[0] == pytest.approx(0.6316, abs=1e-3)
    assert fracs[-1] == pytest.approx(1.0, abs=1e-3)
    assert W.energy_by_modes("sine", 1) == pytest.approx(1.0)


def test_modes_bracketing_brackets():
    K, lo, hi = W.modes_bracketing(0.766, "pluck")
    assert K == 1 and lo <= 0.766 < hi
    assert lo == pytest.approx(W.energy_by_modes("pluck", 1))
    assert hi == pytest.approx(W.energy_by_modes("pluck", 2))


# ---------------------------------------------------------------------------
# 3. the baselines the section quotes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ic", list(W.ICS))
def test_the_trivial_solution_scores_exactly_one(ic):
    """``u = 0`` is a solution of everything except the initial displacement.

    It satisfies the residual, both boundary conditions and the zero initial
    *velocity* exactly, which is why the whole problem rides on one loss term --
    and why the relative error of a zero network is exactly 1.0, the number the
    section's tables are read against.
    """
    zero = Zero()
    assert W.rel_l2_error(zero, ic) == pytest.approx(1.0, rel=1e-12)
    assert W.energy_ratio(zero, ic) == pytest.approx(0.0, abs=1e-12)
    coords = torch.rand(64, 2, requires_grad=True)
    assert torch.allclose(W.wave_residual(zero(coords), coords),
                          torch.zeros(64, 1))


def test_residual_is_the_equation_it_says_it_is():
    """On the exact single-mode solution, written as a torch expression."""
    coords = torch.rand(200, 2, dtype=torch.float64, requires_grad=True)
    x, t = coords[:, 0:1], coords[:, 1:2]
    u = torch.sin(np.pi * x) * torch.cos(np.pi * W.C * t)
    r = W.wave_residual(u, coords)
    assert torch.max(torch.abs(r)).item() < 1e-8
    # and a function that is not a solution has a residual the same size as its
    # own second derivatives, so the check above is not vacuous
    u2 = torch.sin(np.pi * x) * torch.cos(2 * np.pi * W.C * t)
    assert torch.max(torch.abs(W.wave_residual(u2, coords))).item() > 1.0


def test_kink_split_separates_the_characteristics():
    """The band really is a neighbourhood of the travelling corners."""
    # a point on the corner's characteristic, and one far from both
    assert W._periodic_distance(np.array([W.PLUCK_X0]), W.PLUCK_X0)[0] == \
        pytest.approx(0.0, abs=1e-12)
    far = W._periodic_distance(np.array([W.PLUCK_X0 + 0.5]), W.PLUCK_X0)[0]
    assert far == pytest.approx(0.5, abs=1e-12)
    # a network that is exactly right has zero error in both regions
    class Exact(torch.nn.Module):
        def forward(self, coords):
            c = coords.detach().numpy()
            return torch.tensor(W.wave_exact(c[:, 0], c[:, 1], "pluck"),
                                dtype=torch.float32).unsqueeze(1)
    kink, smooth = W.kink_split(Exact(), "pluck")
    assert kink < 1e-6 and smooth < 1e-6


# ---------------------------------------------------------------------------
# 4. one real cell, and the committed logs
# ---------------------------------------------------------------------------
def test_training_cell_runs_and_selects_the_lowest_loss():
    model, history, best = W.train(ic="sine", seed=0, **TINY)
    assert history[-1][0] == TINY["steps"]
    assert best["loss"] <= min(row[1] for row in history) + 1e-12
    row, _ = W.run_cell("sine", 0, verbose=False, **TINY)
    assert set(row) == set(W.CELL_FIELDS)


def test_velocity_term_is_actually_in_the_objective(monkeypatch):
    """Drop it and the objective changes -- the four terms are four terms.

    A second-order-in-time equation needs two initial conditions; with only the
    displacement the problem is ill-posed. The test is that the term is doing
    something, not that the run is better with it.
    """
    _, _, with_vel = W.train(ic="sine", seed=0, **TINY)
    _, _, without = W.train(ic="sine", seed=0, w_vel=0.0, **TINY)
    assert with_vel["final_loss"] != without["final_loss"]


def test_unknown_initial_condition_is_refused():
    for fn in (W.f0, W.fourier_coefficients, W.exact_energy):
        with pytest.raises(ValueError):
            fn("triangle-ish")


def test_committed_check_log_says_dalembert_is_exact():
    rows = read_csv(W.CHECK_CSV)
    assert rows
    for r in rows:
        assert float(r["bc_left"]) < 1e-12
        assert float(r["bc_right"]) < 1e-12
        assert float(r["ic_error"]) < 1e-12
        assert float(r["initial_velocity"]) < 1e-9
        assert float(r["residual_fd"]) < 1e-6
        assert float(r["energy_exact"]) == pytest.approx(
            W.exact_energy(r["ic"]), rel=1e-5)
    # the pluck's agreement with the series improves with mode count
    pluck = sorted(((int(r["n_modes"]), float(r["rms_gap"])) for r in rows
                    if r["ic"] == "pluck"))
    assert [g for _, g in pluck] == sorted((g for _, g in pluck), reverse=True)


def test_committed_cells_log_is_consistent():
    rows = read_csv(W.CELLS_CSV)
    assert rows
    for r in rows:
        assert r["ic"] in W.ICS
        assert int(r["best_step"]) <= W.BUDGET["steps"] + 1
        assert float(r["energy_exact"]) == pytest.approx(
            W.exact_energy(r["ic"]), rel=1e-5)
    summary = {s["ic"]: s for s in W.summarize(rows)}
    # the section's two headline claims, as assertions on the committed log
    assert float(summary["pluck"]["mean"]) > 5 * float(summary["sine"]["mean"])
    assert float(summary["pluck"]["mean_kink"]) > \
        2 * float(summary["pluck"]["mean_smooth"])
    assert float(summary["sine"]["mean_kink"]) == pytest.approx(
        float(summary["sine"]["mean_smooth"]), rel=0.3)


def test_committed_trace_covers_every_cell():
    cells = read_csv(W.CELLS_CSV)
    trace = read_csv(W.TRACE_CSV)
    keyed = {(t["ic"], int(t["seed"])) for t in trace}
    for c in cells:
        assert (c["ic"], int(c["seed"])) in keyed
    assert W.BUDGET["steps"] in {int(t["step"]) for t in trace}
