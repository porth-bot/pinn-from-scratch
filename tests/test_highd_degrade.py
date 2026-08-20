"""The degradation study: its closed forms, its samplers, and its committed log.

``experiments/highd_degrade.py`` rests on one piece of algebra -- the effective
collocation count ``(2/3)^d`` uniform and ``(9/10)^d`` tilted -- and on three
arms whose comparability is a property of the code rather than of the write-up.
Neither survives a change to the module unless it is pinned, so:

1. **The closed forms**, against Monte Carlo and against the one-dimensional
   integrals they are built from (``E[sin^2] = 1/2``, ``E[sin^4] = 3/8``,
   ``E[sin^6] = 5/16``), plus the exact 25/33 correction the two-mode initial
   condition carries relative to the fundamental alone.
2. **The tilted sampler**, against the CDF it inverts rather than against
   another sampler, and against the density it claims to draw from.
3. **Arm comparability**: every arm draws the same number of interior points,
   and the ``rad`` arm is bit-for-bit the uniform arm until its first resample.
   That is the property that makes "adaptivity did this" a legitimate reading of
   a difference between them, and it broke once already -- an earlier version
   drew only the uniform *base* up front, so the adaptive arm silently trained
   on a third fewer points through the warmup.
4. **Exact resumption**, on a fixed-set arm and on the resampling arm, since the
   latter draws from the generator mid-training and a checkpoint that forgets
   the generator state would resume onto a different run.
5. **The committed logs**, against the module's declared budget and against each
   other -- the failure mode a README table cannot notice.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import highd_degrade as G  # noqa: E402
from common import read_csv  # noqa: E402
from highd_heat import HighDHeat, exact_rms, initial_condition  # noqa: E402

TINY = dict(n_ic=40, n_bc=40, width=16, depth=2, steps=24, lr=1e-3,
            eval_every=8, eval_n=2000)


def _numbers(history):
    """A history without its last column, which is a clock.

    Everything a history row carries is a deterministic function of the run
    except ``train_seconds``, which is wall time and reproduces nowhere -- this
    repo has now written that lesson down three times (Sec. 11's ms/step, Sec.
    12's mesh timings, ``gp-from-scratch``'s peak RSS). Comparing two runs on
    the numbers and not on the clock is the portable assertion.
    """
    return [row[:-1] for row in history]


# ---------------------------------------------------------------------------
# 1. the closed forms
# ---------------------------------------------------------------------------
def test_one_dimensional_moments_are_the_ones_the_derivation_uses():
    """``E[sin^{2k}(pi x)]`` for k = 1, 2, 3 on [0, 1], by quadrature.

    Every factor of ``(2/3)^d`` and ``(9/10)^d`` is a ratio of these three, so
    they are checked first and separately; if one of them is wrong the ESS laws
    are wrong in a way no Monte Carlo comparison would isolate.
    """
    x = np.linspace(0, 1, 200_001)
    s = np.sin(np.pi * x)
    for power, expected in ((2, 1 / 2), (4, 3 / 8), (6, 5 / 16)):
        got = np.trapezoid(s ** power, x) if hasattr(np, "trapezoid") else \
            np.trapz(s ** power, x)
        assert abs(got - expected) < 1e-8, (power, got, expected)


@pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
def test_ess_fraction_matches_monte_carlo(d):
    """``ESS/n`` measured on real draws agrees with the closed form.

    A 200,000-point estimate of a ratio of fourth moments is noisy at d = 16 --
    the quantity being estimated is itself concentration-limited -- so the
    tolerance is relative and generous (20%). The point of the check is that the
    law has the right *shape* in d; the exactness of the constant is settled by
    the moment test above, not by sampling.
    """
    rng = np.random.default_rng(1234)
    n = 200_000
    x = rng.random((n, d))
    phi = np.prod(np.sin(np.pi * x), axis=1)
    meas = G.ess(phi ** 2) / n
    assert meas == pytest.approx(G.ess_fraction(d, "uniform"), rel=0.2)

    xt = G.tilted_unit(torch.rand(n, d, generator=torch.Generator().manual_seed(0),
                                  dtype=torch.float64)).numpy()
    phit = np.prod(np.sin(np.pi * xt), axis=1)
    meas_t = G.ess(phit ** 2) / n
    assert meas_t == pytest.approx(G.ess_fraction(d, "tilted"), rel=0.05)


def test_tilted_beats_uniform_by_a_growing_factor():
    """The ratio ``(27/20)^d`` is the prediction the tilted arm was run on.

    ``(9/10)^d / (2/3)^d = (27/20)^d``: 1.35 at d = 1 and 122 at d = 16. It is
    asserted here because the measured outcome of the arm went the other way,
    and a reader is entitled to check that the prediction really was this large
    rather than being softened after the fact.
    """
    for d in (1, 2, 4, 8, 16):
        ratio = G.ess_fraction(d, "tilted") / G.ess_fraction(d, "uniform")
        assert ratio == pytest.approx((27 / 20) ** d, rel=1e-12)
    assert 120 < G.ess_fraction(16, "tilted") / G.ess_fraction(16, "uniform") < 123


@pytest.mark.parametrize("d", [1, 2, 4, 8])
def test_two_mode_initial_condition_costs_exactly_25_over_33(d):
    """The default IC's ESS is ``(25/33) (2/3)^d``, not ``(2/3)^d``.

    ``u_0 = phi_1 + 0.5 phi_2`` with phi_2 doubling the first axis. Expanding
    the fourth moment, the two odd cross terms vanish
    (``int_0^1 sin^3(pi x) sin(2 pi x) dx = 0`` and its partner), and what
    survives gives ``E[u_0^2]^2 / E[u_0^4] = (25/33)(2/3)^d``. The per-cell
    ``ess_ic`` column measures this quantity, so the constant is worth pinning:
    without it the logged ESS looks like a 24% disagreement with the module's
    own headline law.
    """
    rng = np.random.default_rng(7)
    n = 400_000
    problem = HighDHeat(d)
    x = rng.random((n, d))
    u0 = initial_condition(problem, torch.as_tensor(x, dtype=torch.float64))
    meas = G.ess(u0.numpy().ravel() ** 2) / n
    assert meas == pytest.approx((25 / 33) * (2 / 3) ** d, rel=0.1)


def test_n_for_effective_inverts_ess_fraction():
    for d in (1, 4, 16):
        for arm in ("uniform", "tilted"):
            n = G.n_for_effective(1000, d, arm)
            assert n * G.ess_fraction(d, arm) == pytest.approx(1000)


def test_ess_edge_cases():
    assert G.ess(np.ones(50)) == pytest.approx(50)
    assert G.ess(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert G.ess(np.zeros(5)) == 0.0
    with pytest.raises(ValueError):
        G.ess(np.array([1.0, -1.0]))


# ---------------------------------------------------------------------------
# 2. the tilted sampler
# ---------------------------------------------------------------------------
def test_tilted_unit_inverts_its_own_cdf():
    """``F(tilted_unit(u)) == u`` to float64 precision, on a fine grid."""
    u = torch.linspace(1e-6, 1 - 1e-6, 2001, dtype=torch.float64)
    x = G.tilted_unit(u)
    assert np.allclose(G.tilted_marginal_cdf(x.numpy()), u.numpy(), atol=1e-12)


def test_tilted_marginal_cdf_is_the_antiderivative_of_the_density():
    """Differencing the CDF reproduces ``2 sin^2(pi x)``.

    The sampler is only as right as the CDF it inverts, and the CDF is the one
    place where an algebra slip would be invisible in the sampler's own output.
    """
    x = np.linspace(0, 1, 100_001)
    F = G.tilted_marginal_cdf(x)
    dens = np.gradient(F, x)
    assert np.allclose(dens, 2 * np.sin(np.pi * x) ** 2, atol=1e-6)
    assert G.tilted_marginal_cdf(0.0) == pytest.approx(0.0)
    assert G.tilted_marginal_cdf(1.0) == pytest.approx(1.0)


def test_tilted_draw_has_the_claimed_moments():
    """A tilted sample's ``E[sin^2]`` is 3/4, not 1/2 -- the whole point of it."""
    gen = torch.Generator().manual_seed(3)
    x = G.tilted_x(400_000, 1, gen).numpy().ravel()
    assert x.mean() == pytest.approx(0.5, abs=0.005)
    assert np.mean(np.sin(np.pi * x) ** 2) == pytest.approx(0.75, abs=0.005)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_tilted_points_are_further_from_the_boundary():
    """The coverage the tilted arm gives up, as a number rather than a claim."""
    gen = torch.Generator().manual_seed(5)
    d = 8
    uni = torch.rand(20_000, d, generator=gen).numpy()
    til = G.tilted_x(20_000, d, gen).numpy()
    near = lambda a: float(np.mean(np.min(np.minimum(a, 1 - a), axis=1) < 0.02))
    assert near(til) < 0.5 * near(uni)


# ---------------------------------------------------------------------------
# 3. arm comparability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", list(G.ARMS))
def test_every_arm_trains_on_the_same_number_of_interior_points(arm):
    problem = HighDHeat(2)
    seen = {}
    real = G.residual

    def spy(prob, u, coords):
        seen.setdefault("n", set()).add(int(coords.shape[0]))
        return real(prob, u, coords)

    G.residual = spy
    try:
        G.train_cell(problem, arm, n_interior=300, **TINY)
    finally:
        G.residual = real
    # The rad arm evaluates the candidate pool through the same residual, so its
    # set of observed sizes is {n, n_candidates}; every training call is at n.
    assert 300 in seen["n"]
    assert seen["n"] <= {300, G.RAR_CANDIDATES}


def test_rad_equals_uniform_until_its_first_resample():
    """Sec. 7's property, restated here: the arms share a warmup exactly.

    With ``steps < RAR_WARMUP`` the adaptive arm has never resampled, so it must
    be the uniform run to the last bit. This is what caught the base-size bug.
    """
    problem = HighDHeat(2)
    assert TINY["steps"] < G.RAR_WARMUP
    _, hist_u, best_u = G.train_cell(problem, "uniform", n_interior=300, **TINY)
    _, hist_r, best_r = G.train_cell(problem, "rad", n_interior=300, **TINY)
    assert _numbers(hist_u) == _numbers(hist_r)
    assert best_u["final_loss"] == best_r["final_loss"]


def test_rad_actually_resamples_when_it_is_allowed_to():
    """And is *not* the uniform run once it has, so the check above has teeth."""
    problem = HighDHeat(2)
    kw = dict(TINY, steps=G.RAR_WARMUP + G.RAR_EVERY, eval_every=1000)
    _, _, best_u = G.train_cell(problem, "uniform", n_interior=300, **kw)
    _, _, best_r = G.train_cell(problem, "rad", n_interior=300, **kw)
    assert best_u["final_loss"] != best_r["final_loss"]


def test_tilted_arm_changes_the_points_and_not_the_model():
    """The arms differ in the draw alone: same architecture, same parameters."""
    problem = HighDHeat(3)
    mu, _, bu = G.train_cell(problem, "uniform", n_interior=200, **TINY)
    mt, _, bt = G.train_cell(problem, "tilted", n_interior=200, **TINY)
    assert bu["ess_ic"] < bt["ess_ic"]        # the tilt does what it says
    counts = [sum(p.numel() for p in m.parameters()) for m in (mu, mt)]
    assert counts[0] == counts[1] == G.n_params(3, TINY["width"], TINY["depth"])
    # and the two runs really are different runs, not the same one relabelled
    assert bu["final_loss"] != bt["final_loss"]


@pytest.mark.parametrize("arm", list(G.ARMS))
def test_replay_points_reproduces_the_training_draw(arm):
    """``replay_points`` must draw what ``train_cell`` drew, not something like it.

    Two columns of the committed log were added after the cells had already been
    run and were filled in by replaying each cell's draw rather than by spending
    its minutes again. That is only legitimate if the replay is the same draw, so
    the energy ``train_cell`` computes internally is compared against the one the
    replay computes -- exact equality, since both are the same arithmetic on the
    same points.
    """
    problem = HighDHeat(3)
    _, _, best = G.train_cell(problem, arm, n_interior=200, **TINY)
    replayed = G.sampled_ic_energy(problem, arm, 0, 200, n_ic=TINY["n_ic"])
    assert replayed == best["ic_energy_sampled"]


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError):
        G.train_cell(HighDHeat(1), "importance", **TINY)


# ---------------------------------------------------------------------------
# 4. resumption is exact
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["uniform", "rad"])
def test_resumed_run_equals_uninterrupted_run(tmp_path, arm):
    """Interrupt a cell mid-training and finish it; compare entry for entry.

    Run once straight through, then again with a deadline that expires during
    the first call, and require the two histories and the two selected losses to
    agree exactly. The ``rad`` arm is the one that matters: it draws from the
    generator mid-training, so a checkpoint that saved the weights but not the
    generator state would resume onto a different sequence of point sets and
    this test would see it.
    """
    problem = HighDHeat(2)
    kw = dict(TINY, steps=G.RAR_WARMUP + G.RAR_EVERY + 20, eval_every=50)
    _, hist_ref, best_ref = G.train_cell(problem, arm, n_interior=200, **kw)

    ckpt = str(tmp_path / "cell.pt")
    _, _, best_a = G.train_cell(problem, arm, n_interior=200, ckpt_path=ckpt,
                                ckpt_every=10, deadline=0.0, **kw)
    assert not best_a.get("completed", True)
    assert os.path.exists(ckpt)
    for _ in range(200):
        _, hist_b, best_b = G.train_cell(problem, arm, n_interior=200,
                                         ckpt_path=ckpt, ckpt_every=10,
                                         deadline=0.0, **kw)
        if best_b.get("completed", True):
            break
    assert best_b["completed"]
    assert _numbers(hist_b) == _numbers(hist_ref)
    assert best_b["final_loss"] == best_ref["final_loss"]
    assert best_b["step"] == best_ref["step"]


# ---------------------------------------------------------------------------
# 5. one real cell, and the row it writes
# ---------------------------------------------------------------------------
def test_run_cell_row_is_complete_and_self_consistent():
    row, history = G.run_cell("uniform", 2, 0, 200, verbose=False,
                              resumable=False, **TINY)
    assert set(row) == set(G.CELL_FIELDS)
    assert int(row["n_interior"]) == 200
    assert history[-1][0] == TINY["steps"]
    # rel_ic_error is sqrt(loss_ic / IC energy), and the IC energy is closed form
    ic_energy, residual_scale = G.loss_scales(HighDHeat(2))
    assert float(row["rel_ic_error"]) == pytest.approx(
        np.sqrt(float(row["loss_ic"]) / ic_energy), rel=1e-5)
    assert float(row["rel_residual"]) == pytest.approx(
        np.sqrt(float(row["loss_r"]) / residual_scale), rel=1e-5)
    assert float(row["exact_rms"]) == pytest.approx(exact_rms(HighDHeat(2)))


def test_selection_returns_the_lowest_loss_iterate():
    problem = HighDHeat(2)
    _, history, best = G.train_cell(problem, "uniform", n_interior=200,
                                    select="best_loss", eval_every=1,
                                    **{k: v for k, v in TINY.items()
                                       if k != "eval_every"})
    logged = [row[1] for row in history]
    assert best["loss"] <= min(logged) + 1e-12


def test_fit_cell_is_supervised_and_reports_both_errors():
    """The control arm fits labels: no residual, and a train-set error column."""
    problem = HighDHeat(2)
    calls = {"n": 0}
    real = G.residual

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    G.residual = spy
    try:
        row = G.fit_cell(problem, "uniform", n_points=200, steps=30, width=16,
                         depth=2, eval_n=5000)
    finally:
        G.residual = real
    assert calls["n"] == 0                       # no PDE anywhere in the control
    assert float(row["rel_fit_error"]) == pytest.approx(
        np.sqrt(float(row["final_mse"])) / exact_rms(problem), rel=1e-6)
    assert int(row["steps"]) == 30


# ---------------------------------------------------------------------------
# 6. the committed logs
# ---------------------------------------------------------------------------
def test_committed_geometry_log_matches_the_closed_forms():
    rows = read_csv(G.GEOM_CSV)
    assert rows, "geometry log is empty"
    for r in rows:
        d = int(r["d"])
        assert float(r["ess_frac_uniform"]) == pytest.approx(
            G.ess_fraction(d, "uniform"), rel=1e-6)
        assert float(r["ess_frac_tilted"]) == pytest.approx(
            G.ess_fraction(d, "tilted"), rel=1e-6)
        assert float(r["ess_frac_uniform_mc"]) == pytest.approx(
            float(r["ess_frac_uniform"]), rel=0.25)
        assert float(r["eff_points_uniform"]) == pytest.approx(
            int(r["n"]) * G.ess_fraction(d, "uniform"), rel=1e-4)
        assert float(r["cube_diameter"]) == pytest.approx(np.sqrt(d))
    # the nearest-neighbour distance is monotone in d and passes the
    # inter-point spacing that makes "a sample of the cube" meaningless
    nn = [float(r["nn_mean"]) for r in rows]
    assert nn == sorted(nn)


def test_committed_cells_log_is_consistent_with_the_budget():
    rows = read_csv(G.CELLS_CSV)
    assert rows, "cells log is empty"
    for r in rows:
        assert r["arm"] in G.ARMS
        assert int(r["params"]) == G.n_params(int(r["d"]), G.BUDGET["width"],
                                              G.BUDGET["depth"])
        assert float(r["exact_rms"]) == pytest.approx(
            exact_rms(HighDHeat(int(r["d"]))), rel=1e-5)
        assert int(r["best_step"]) <= G.BUDGET["steps"] + 1
        assert float(r["stderr"]) < 0.2 * float(r["rel_l2"])


def test_committed_trace_covers_every_committed_cell():
    cells = read_csv(G.CELLS_CSV)
    trace = read_csv(G.TRACE_CSV)
    keyed = {(t["arm"], int(t["d"]), int(t["seed"]), int(t["n_interior"]))
             for t in trace}
    for c in cells:
        assert G.cell_key(c) in keyed
    steps = {int(t["step"]) for t in trace}
    assert G.BUDGET["steps"] in steps


def test_committed_fit_log_shares_the_budget_and_the_points():
    rows = read_csv(G.FIT_CSV)
    assert rows, "fit log is empty"
    shared = [r for r in rows if int(r["steps"]) == G.BUDGET["steps"]]
    assert shared
    for r in rows:
        assert int(r["n_points"]) == G.BUDGET["n_interior"]
        assert float(r["rel_fit_error"]) == pytest.approx(
            np.sqrt(float(r["final_mse"])) / float(r["exact_rms"]), rel=1e-5)


def test_summarize_arithmetic_on_known_rows():
    rows = [
        {"arm": "uniform", "d": "8", "seed": "0", "n_interior": "4000",
         "rel_l2": "0.2", "rel_ic_error": "0.5", "rel_ic_sampled": "0.5",
         "ess_ic": "10",
         "train_seconds": "100"},
        {"arm": "uniform", "d": "8", "seed": "1", "n_interior": "4000",
         "rel_l2": "0.4", "rel_ic_error": "0.7", "rel_ic_sampled": "0.7",
         "ess_ic": "20",
         "train_seconds": "200"},
        {"arm": "uniform", "d": "8", "seed": "2", "n_interior": "999",
         "rel_l2": "9.9", "rel_ic_error": "0.1", "rel_ic_sampled": "0.1",
         "ess_ic": "1",
         "train_seconds": "1"},
    ]
    (s,) = G.summarize(rows)
    assert s["n_seeds"] == 2                 # the n = 999 cell is a different arm
    assert float(s["mean"]) == pytest.approx(0.3)
    assert float(s["min"]) == pytest.approx(0.2)
    assert float(s["spread"]) == pytest.approx(2.0)
    assert float(s["mean_ess_ic"]) == pytest.approx(15.0)
