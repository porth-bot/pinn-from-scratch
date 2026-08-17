"""The d-sweep: its arithmetic, its determinism, and its committed log.

``experiments/highd_pinn.py`` reports one headline -- how the relative L2 error
moves as d goes 1 -> 16 at a fixed budget -- and two supporting claims: that the
network's cost per step grows linearly in d, and that the seed spread it quotes
is larger than the Monte Carlo noise in the metric. The first two hours of CPU
that produce those numbers cannot run in a test suite, so what is tested here is
everything around them:

1. **The cost claim, structurally rather than by timing.** The residual takes
   ``d + 2`` reverse-mode passes -- one for u_t, one shared first gradient, and
   one per spatial axis -- which is counted by instrumenting
   ``torch.autograd.grad`` rather than by measuring a clock. ``gp-from-scratch``
   spent two CI rounds on timing and memory assertions that were measuring the
   machine; a call count is portable.
2. **The summary arithmetic**, on synthetic rows where the answer is known.
3. **One real cell end to end** at a tiny budget: the row's shape, the selection
   invariant (the selected loss is the lowest seen), and determinism in the seed.
4. **The committed sweep log**, against the module's declared budget. A log that
   drifts from the constants the module reports is the failure mode that a
   README table cannot notice -- ``gp-from-scratch``'s Day 5 shipped a stale
   figure for exactly this reason -- so the cells, the parameter counts and the
   step counts are all re-derived here from ``BUDGET`` and compared.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import highd_heat as H  # noqa: E402
import highd_pinn as P  # noqa: E402
from common import read_csv  # noqa: E402
from highd_heat import HighDHeat, interior_points, residual  # noqa: E402
from pinn import derivatives as D  # noqa: E402
from pinn.model import MLP  # noqa: E402

TINY = dict(n_interior=200, n_ic=50, n_bc=50, width=16, depth=2, steps=30, lr=1e-3)


# ---------------------------------------------------------------------------
# 1. the cost claim, as a call count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_residual_costs_d_plus_two_backward_passes(monkeypatch, d):
    """The residual is linear in d, and the constant is the point.

    ``u_t`` needs one gradient of u; the Laplacian takes one more gradient of u
    and then differentiates each of its d spatial columns. That is ``d + 2``,
    not ``2d + 1`` -- the naive ``sum_i _second(u, coords, i)`` recomputes grad u
    inside every term. Both counts are checked, so the saving is measured and
    not just described in a docstring.
    """
    calls = {"n": 0}
    real = torch.autograd.grad

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting)

    problem = HighDHeat(d)
    model = MLP(in_dim=d + 1, out_dim=1, width=8, depth=2)
    gen = torch.Generator().manual_seed(0)
    coords = interior_points(problem, 16, gen)

    residual(problem, model(coords), coords)
    assert calls["n"] == d + 2

    calls["n"] = 0
    u = model(coords)
    total = torch.zeros_like(u)
    for i in range(d):
        total = total + D.partial(D.partial(u, coords, i), coords, i)
    assert calls["n"] == 2 * d


# ---------------------------------------------------------------------------
# 2. parameter counting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 4, 8, 16])
def test_n_params_matches_a_real_model(d):
    model = MLP(**{"in_dim": d + 1, "out_dim": 1, "width": 32, "depth": 3})
    counted = sum(p.numel() for p in model.parameters())
    assert P.n_params(d, 32, 3) == counted


def test_params_grow_only_through_the_input_layer():
    """d = 1 -> 16 adds exactly ``width * 15`` weights, and nothing else.

    The sweep calls its architecture fixed. It is fixed up to this, which is
    why the count is reported per cell rather than assumed constant.
    """
    w = P.BUDGET["width"]
    delta = P.n_params(16, w, P.BUDGET["depth"]) - P.n_params(1, w, P.BUDGET["depth"])
    assert delta == w * 15


# ---------------------------------------------------------------------------
# 3. summary arithmetic
# ---------------------------------------------------------------------------
def _row(d, seed, rel, se=1e-4, final=None, ms=10.0, secs=100.0):
    return {"d": d, "seed": seed, "params": P.n_params(d, 128, 4),
            "rel_l2": f"{rel:.6e}", "stderr": f"{se:.6e}",
            "rel_l2_final": f"{(final if final is not None else rel):.6e}",
            "train_seconds": f"{secs:.2f}", "ms_per_step": f"{ms:.3f}"}


def test_summarize_arithmetic():
    rows = [_row(2, 0, 1.0), _row(2, 1, 2.0), _row(2, 2, 3.0),
            _row(4, 0, 10.0), _row(4, 1, 10.0)]
    summary = P.summarize(rows)
    assert [s["d"] for s in summary] == [2, 4]

    two = summary[0]
    assert float(two["mean"]) == pytest.approx(2.0)
    assert float(two["sd"]) == pytest.approx(1.0)          # ddof = 1 on 1, 2, 3
    assert float(two["min"]) == pytest.approx(1.0)
    assert float(two["max"]) == pytest.approx(3.0)
    assert float(two["spread"]) == pytest.approx(3.0)
    assert two["n_seeds"] == 3

    four = summary[1]
    assert float(four["sd"]) == pytest.approx(0.0)
    assert float(four["spread"]) == pytest.approx(1.0)


def test_summarize_takes_ms_per_step_from_the_rows():
    """Not recomputed from ``BUDGET``: a summary of a shorter run must not be
    labelled with the sweep's step count."""
    rows = [_row(2, 0, 1.0, ms=7.5, secs=3.0), _row(2, 1, 1.0, ms=8.5, secs=3.0)]
    assert float(P.summarize(rows)[0]["ms_per_step"]) == pytest.approx(8.0)


@pytest.mark.parametrize("d", [1, 2, 5])
def test_loss_scales_against_monte_carlo(d):
    """The two normalizing energies, checked by sampling rather than by algebra.

    Both are closed forms that lean on eigenfunction orthogonality; a Monte
    Carlo estimate of the same quantity shares none of that reasoning. The IC
    energy is the mean square of ``u(x, 0)``; the residual scale is the
    space-time mean square of ``u_t`` on the *exact* solution, which is what
    makes a nonzero residual large or small.
    """
    problem = HighDHeat(d)
    ic_energy, residual_scale = P.loss_scales(problem)

    rng = np.random.default_rng(0)
    x = rng.random((400_000, d))
    u0 = H.exact(problem, x, 0.0)
    assert float((u0 ** 2).mean()) == pytest.approx(ic_energy, rel=0.03)

    coords = H.uniform_box_points(problem, 400_000, np.random.default_rng(1))
    xs, t = coords[:, :d], coords[:, d]
    u_t = np.zeros(xs.shape[0])
    for k, a, rate in zip(problem.modes, problem.amps, problem.rates):
        spatial = np.prod(np.sin(np.pi * k[None, :] * xs), axis=1)
        u_t = u_t - a * rate * spatial * np.exp(-rate * t)
    assert float((u_t ** 2).mean()) == pytest.approx(residual_scale, rel=0.05)


def test_normalized_losses_are_relative_errors():
    """``sqrt(loss / energy)``, and nothing else. Checked on rows whose losses
    are chosen so the answer is 1 -- a network no better than zero."""
    rows = []
    for d in (1, 4):
        ic_energy, residual_scale = P.loss_scales(HighDHeat(d))
        rows.append({"d": d, "loss_ic": f"{ic_energy:.6e}",
                     "loss_r": f"{residual_scale:.6e}"})
    for r in P.normalized_losses(rows):
        assert float(r["rel_ic_error"]) == pytest.approx(1.0, rel=1e-9)
        assert float(r["rel_residual"]) == pytest.approx(1.0, rel=1e-9)


def test_committed_losses_are_not_comparable_raw(committed):
    """The reason the normalization exists, asserted on the committed log.

    At d = 16 the raw ``loss_ic`` is smaller than at d = 2, and the normalized
    IC error is *forty times larger* and above 1 -- worse than the zero network.
    If a future run made the raw comparison honest, this test should be the
    thing that notices.
    """
    rows, _, _ = committed
    norm = {r["d"]: r for r in P.normalized_losses(rows)}
    assert float(norm[16]["loss_ic"]) < float(norm[2]["loss_ic"])
    assert float(norm[16]["rel_ic_error"]) > 1.0
    assert (float(norm[16]["rel_ic_error"])
            > 40 * float(norm[2]["rel_ic_error"]))


# ---------------------------------------------------------------------------
# 4. one real cell
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_cell():
    return P.run_cell(2, 0, budget=TINY, eval_every=15, eval_n=2000,
                      score_n=5000, verbose=False)


def test_run_cell_row_has_every_logged_field(tiny_cell):
    row, _ = tiny_cell
    expected = {"d", "seed", "params", "rel_l2", "stderr", "rel_l2_final",
                "stderr_final", "best_step", "best_loss", "final_loss",
                "loss_r", "loss_ic", "loss_bc", "exact_rms",
                "train_seconds", "wall_seconds", "ms_per_step"}
    assert set(row) == expected
    assert row["params"] == P.n_params(2, TINY["width"], TINY["depth"])
    assert float(row["rel_l2"]) > 0 and float(row["stderr"]) > 0


def test_selected_iterate_really_is_the_lowest_loss(tiny_cell):
    """The selection is on training loss, so the invariant is on the loss --
    not on the error, which the selection never sees."""
    row, history = tiny_cell
    best_loss = float(row["best_loss"])
    assert best_loss <= float(row["final_loss"])
    assert best_loss <= min(h[1] for h in history)


def test_train_seconds_excludes_evaluation_and_is_monotone(tiny_cell):
    """The cost column is optimization only. Evaluation is instrumentation, and
    charging the method for it would overstate the PINN's side of Sec. 11."""
    row, history = tiny_cell
    stamps = [h[7] for h in history]
    assert stamps == sorted(stamps)
    assert stamps[0] > 0
    assert float(row["train_seconds"]) <= float(row["wall_seconds"])


def test_cell_is_deterministic_in_the_seed():
    a, _ = P.run_cell(2, 0, budget=TINY, eval_every=1000, eval_n=1000,
                      score_n=4000, verbose=False)
    b, _ = P.run_cell(2, 0, budget=TINY, eval_every=1000, eval_n=1000,
                      score_n=4000, verbose=False)
    assert a["rel_l2"] == b["rel_l2"]
    assert a["best_loss"] == b["best_loss"]

    c, _ = P.run_cell(2, 1, budget=TINY, eval_every=1000, eval_n=1000,
                      score_n=4000, verbose=False)
    assert c["rel_l2"] != a["rel_l2"]      # the seed is wired through


# ---------------------------------------------------------------------------
# 5. the committed log, against the module's declared constants
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def committed():
    return read_csv(P.SWEEP_CSV), read_csv(P.TRACE_CSV), read_csv(P.COST_CSV)


def test_committed_sweep_covers_every_cell(committed):
    rows, _, _ = committed
    cells = {(int(r["d"]), int(r["seed"])) for r in rows}
    assert cells == {(d, s) for d in P.SWEEP_DIMS for s in P.SEEDS}


def test_committed_sweep_matches_the_declared_budget(committed):
    """Every row's parameter count is re-derived from ``BUDGET`` here. If the
    sweep is ever re-run at another width, or the constant is edited without
    re-running, these disagree."""
    rows, traces, _ = committed
    for r in rows:
        assert int(r["params"]) == P.n_params(int(r["d"]), P.BUDGET["width"],
                                              P.BUDGET["depth"])
    for d in P.SWEEP_DIMS:
        for s in P.SEEDS:
            steps = [int(t["step"]) for t in traces
                     if int(t["d"]) == d and int(t["seed"]) == s]
            assert max(steps) == P.BUDGET["steps"]
            assert min(steps) == 0


def test_committed_errors_are_resolved_by_their_own_metric(committed):
    """Every quoted error is many standard errors away from zero, which is the
    condition under which the delta-method standard error means anything -- and
    the condition that fails first as d grows, since the estimator's noise grows
    like ``sqrt((3/2)^d / n)``."""
    rows, _, _ = committed
    for r in rows:
        rel, se = float(r["rel_l2"]), float(r["stderr"])
        assert rel > 0
        assert se / rel < 0.05, (r["d"], r["seed"], se / rel)


def test_seed_spread_exceeds_the_metric_noise(committed):
    """The claim the summary makes in words. At every d the sd across seeds is
    larger than the mean Monte Carlo standard error, so the spread is a property
    of the training runs and not of the scoring."""
    rows, _, _ = committed
    for s in P.summarize(rows):
        assert float(s["sd"]) > float(s["mean_stderr"]), s["d"]


def test_committed_cost_rows_are_the_swept_dimensions(committed):
    _, _, cost = committed
    assert [int(r["d"]) for r in cost] == list(P.SWEEP_DIMS)
    for r in cost:
        assert float(r["ms_per_step"]) > 0
