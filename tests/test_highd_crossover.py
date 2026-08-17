"""The equal-accuracy comparison is arithmetic on two measured curves, and the
arithmetic is where it could go wrong quietly.

Sec. 12's claim is not a measurement of anything new -- both curves were
measured in Secs. 10 and 11 -- it is the statement that one curve passes the
other later than the PINN stops working. Three things have to hold for that to
mean what it says:

1. **A crossing time read off a trajectory is the first one.** A trajectory that
   dips below the target, comes back up, and dips again must be charged the
   first dip; charging the last would inflate the PINN's cost, and charging the
   minimum would deflate it.
2. **The mesh's extrapolation is the law it claims to be.** ``mesh_cost_at``
   is checked against ``highd_mesh.extrapolate``, which was written for Sec. 10
   and shares no code with it, and against the closed form by hand.
3. **The crossover is a lower bound.** The whole result is "the PINN fails N
   dimensions before the mesh would lose", and that is only worth stating if the
   N is conservative. So the direction of the bound is tested directly: making
   the PINN's assumed cost smaller can only move the crossing earlier, never
   later.

The budget guard this section's dimensions exposed in ``required_nx`` is tested
next to that function, in ``test_highd_mesh.py``. What is left here is the one
property of it Sec. 12 depends on directly: that the coarsest legal grid holds
the loose target as d grows, since that is what lets the mesh side of the
headline comparison be measured out to d = 16 rather than extrapolated.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from highd_crossover import (  # noqa: E402
    TARGETS, budget_probe, compare_to_committed, crossover, first_crossing,
    mesh_cost_at, pinn_cost_model, pinn_costs, probe_trend,
)
from highd_heat import HighDHeat  # noqa: E402
from highd_mesh import extrapolate, required_nx, solve  # noqa: E402


# ---------------------------------------------------------------------------
# Reading a crossing off a trajectory
# ---------------------------------------------------------------------------
def test_first_crossing_takes_the_first_dip_not_the_best():
    """A non-monotone trajectory is charged where it first arrived.

    This objective is measurably non-monotone -- Sec. 1's committed history
    oscillates 14x over its last 1500 steps -- so the distinction is not
    hypothetical. The run below reaches the target at step 2, leaves it, and
    comes back lower at step 4; the cost of "getting to 1e-2" is the step 2 one.
    """
    pts = [(1, 5e-2, 10.0), (2, 9e-3, 20.0), (3, 4e-2, 30.0), (4, 1e-3, 40.0)]
    step, seconds, err = first_crossing(pts, 1e-2)
    assert step == 2
    assert seconds == 20.0
    assert err == 9e-3


def test_first_crossing_is_order_independent():
    pts = [(4, 1e-3, 40.0), (2, 9e-3, 20.0), (1, 5e-2, 10.0), (3, 4e-2, 30.0)]
    assert first_crossing(pts, 1e-2)[0] == 2


def test_first_crossing_returns_none_when_it_never_arrives():
    """A cell that never reached the target is a result, not a missing row."""
    pts = [(1, 5e-2, 10.0), (2, 2e-2, 20.0), (3, 1.01e-2, 30.0)]
    assert first_crossing(pts, 1e-2) is None


def test_first_crossing_is_inclusive_at_the_target():
    assert first_crossing([(1, 1e-2, 5.0)], 1e-2) == (1, 5.0, 1e-2)


# ---------------------------------------------------------------------------
# Aggregating the trajectories
# ---------------------------------------------------------------------------
def _trace(d, seed, errs, dt=10.0):
    return [{"d": str(d), "seed": str(seed), "step": str(250 * (i + 1)),
             "rel_l2": f"{e:.6e}", "train_seconds": f"{dt * (i + 1):.4f}"}
            for i, e in enumerate(errs)]


def test_pinn_costs_counts_only_the_seeds_that_arrived():
    rows = _trace(4, 0, [1e-1, 5e-3]) + _trace(4, 1, [1e-1, 2e-2])
    (cell,) = pinn_costs(targets=(1e-2,), trace_rows=rows)
    assert cell["n_seeds"] == 2
    assert cell["n_reached"] == 1
    assert float(cell["seconds"]) == 20.0
    # best_rel_l2 averages the best each seed reached, over ALL seeds -- it is
    # the "how far did the missing cells miss by" number, so excluding the
    # seeds that failed would be exactly backwards.
    assert float(cell["best_rel_l2"]) == pytest.approx((5e-3 + 2e-2) / 2)


def test_pinn_costs_reports_a_shortfall_when_nothing_arrived():
    rows = _trace(8, 0, [8e-1, 7e-1])
    (cell,) = pinn_costs(targets=(1e-1,), trace_rows=rows)
    assert cell["n_reached"] == 0
    assert cell["seconds"] == ""
    assert float(cell["shortfall"]) == pytest.approx(7.0)


def test_pinn_costs_on_the_committed_trajectories_matches_section_11():
    """The committed sweep's own headline, re-derived from the trajectories.

    Sec. 11 reports mean relative L2 of 8.18e-4 at d = 1 and 1.041 at d = 16 on
    a 1,000,000-point score of the selected iterate. The trajectory minimum is a
    different quantity (100,000 points, and the best of 21 evaluations rather
    than the lowest-loss one), so this checks the two agree in *shape* -- that
    the trajectory read this module runs on is the same run Sec. 11 scored.
    """
    cells = {int(c["d"]): c for c in pinn_costs(targets=(1e-1,))}
    assert set(cells) == {1, 2, 4, 8, 16}
    assert float(cells[1]["best_rel_l2"]) < 2e-3
    assert float(cells[16]["best_rel_l2"]) > 1.0
    # And the ordering that is the whole of Sec. 11: strictly worse with d.
    bests = [float(cells[d]["best_rel_l2"]) for d in (1, 2, 4, 8, 16)]
    assert bests == sorted(bests)


def test_the_pinn_reaches_the_loose_target_only_up_to_d_four():
    """The terminating point of the loose curve, off the committed logs."""
    cells = {int(c["d"]): c for c in pinn_costs(targets=(1e-1,))}
    assert [d for d in (1, 2, 4, 8, 16) if int(cells[d]["n_reached"])] == [1, 2, 4]


# ---------------------------------------------------------------------------
# The mesh's cost model
# ---------------------------------------------------------------------------
def _mesh_cells():
    return [
        {"target": "1e-01", "d": "4", "nx": "4", "nt": "2", "seconds": "0.5",
         "unknowns": "81", "arrays": "5"},
        {"target": "1e-01", "d": "6", "nx": "4", "nt": "2", "seconds": "2.0",
         "unknowns": "729", "arrays": "5"},
    ]


def test_mesh_cost_at_returns_the_measurement_where_there_is_one():
    seconds, measured = mesh_cost_at(_mesh_cells(), 6)
    assert measured is True
    assert seconds == 2.0


def test_mesh_cost_at_extrapolates_by_d_times_n_to_the_d():
    """Hand-computed against the anchor, which is the only law being claimed.

    Anchor d = 6 at 2.0 s with N - 1 = 3. At d = 8 the scheme does 8/6 as many
    stages over 3^8/3^6 = 9 times as many nodes, so 2.0 * (8/6) * 9 = 24 s.
    """
    seconds, measured = mesh_cost_at(_mesh_cells(), 8)
    assert measured is False
    assert seconds == pytest.approx(2.0 * (8 / 6) * 9.0)


def test_mesh_cost_at_agrees_with_section_10s_own_extrapolation():
    """Cross-check against ``highd_mesh.extrapolate``, written for Sec. 10.

    Two implementations of the same projection, neither calling the other. They
    have to agree, and if Sec. 10's is ever changed this is what notices.
    """
    rows = [dict(c, rel_l2="1e-3", bytes_counted="0", bytes_traced="0",
                 node_steps="0", fitted_order="2") for c in _mesh_cells()]
    projected, _ = extrapolate(rows, dims=(8, 10))
    for p in projected:
        mine, measured = mesh_cost_at(_mesh_cells(), int(p["d"]))
        assert measured is False
        assert mine == pytest.approx(float(p["seconds"]), rel=1e-12)


# ---------------------------------------------------------------------------
# The crossover, and the direction of its bound
# ---------------------------------------------------------------------------
def _pinn_cells(steps, reached_dims=(1, 2, 4)):
    return [{"target": "1e-01", "d": str(d), "n_seeds": "3",
             "n_reached": "3" if d in reached_dims else "0",
             "steps": str(steps) if d in reached_dims else "",
             "seconds": "1.0" if d in reached_dims else "",
             "best_rel_l2": "1e-2", "shortfall": "0.1"}
            for d in (1, 2, 4, 8)]


def test_crossover_is_monotone_in_the_assumed_pinn_cost():
    """The load-bearing property: a cheaper PINN can only cross earlier.

    Sec. 12 reports the crossing as a lower bound, justified by granting the
    PINN a step count that does not grow in d. That justification is only valid
    if the crossing dimension is monotone in the assumed cost -- otherwise a
    generous assumption would not give a conservative answer.
    """
    mesh = _mesh_cells()
    prev = 0
    for steps in (100, 1_000, 10_000, 100_000):
        (c,) = crossover(mesh, _pinn_cells(steps), targets=(1e-1,))
        d = int(c["d_crossover_lower_bound"])
        assert d >= prev
        prev = d


def test_crossover_reports_where_the_pinn_stopped_and_the_gap():
    mesh = _mesh_cells()
    (c,) = crossover(mesh, _pinn_cells(5_000), targets=(1e-1,))
    assert c["d_pinn_last_reached"] == 4
    assert c["headroom"] == int(c["d_crossover_lower_bound"]) - 4


def test_crossover_lands_where_the_two_cost_curves_actually_meet():
    """Recomputed by hand at the returned dimension and the one before it."""
    mesh = _mesh_cells()
    a, b = pinn_cost_model()
    (c,) = crossover(mesh, _pinn_cells(5_000), targets=(1e-1,))
    d = int(c["d_crossover_lower_bound"])
    pinn_at = lambda k: 5_000 * (a + b * k) / 1000.0        # noqa: E731
    assert mesh_cost_at(mesh, d)[0] > pinn_at(d)
    assert mesh_cost_at(mesh, d - 1)[0] <= pinn_at(d - 1)


def test_crossover_is_empty_when_the_pinn_never_reached_the_target():
    mesh = _mesh_cells()
    (c,) = crossover(mesh, _pinn_cells(5_000, reached_dims=()), targets=(1e-1,))
    assert c["d_crossover_lower_bound"] == ""
    assert c["d_pinn_last_reached"] == 0


def test_a_probe_that_reached_the_target_extends_the_pinn_curve():
    mesh = _mesh_cells()
    probe = [{"target": "1e-01", "d": "8", "reached": "1", "steps": "20000",
              "cross_step": "12000", "best_rel_l2": "5e-2"}]
    (c,) = crossover(mesh, _pinn_cells(5_000), probe_rows=probe, targets=(1e-1,))
    assert c["d_pinn_last_reached"] == 8
    assert float(c["assumed_steps"]) == 12_000


def test_a_probe_that_missed_leaves_the_curve_where_it_was():
    mesh = _mesh_cells()
    probe = [{"target": "1e-01", "d": "8", "reached": "0", "steps": "20000",
              "cross_step": "", "best_rel_l2": "5e-1"}]
    (c,) = crossover(mesh, _pinn_cells(5_000), probe_rows=probe, targets=(1e-1,))
    assert c["d_pinn_last_reached"] == 4
    assert "short" in c["probe"]


def test_a_split_probe_cell_is_reported_as_split():
    """Two seeds, one arriving and one not, is neither seed's story.

    The note used to be whichever row came last, so a cell that arrived in one
    seed of two could be reported either way depending on CSV order.
    """
    probe = [{"target": "1e-01", "d": "8", "reached": "1", "steps": "20000",
              "cross_step": "9000", "best_rel_l2": "8e-2"},
             {"target": "1e-01", "d": "8", "reached": "0", "steps": "20000",
              "cross_step": "", "best_rel_l2": "3e-1"}]
    (c,) = crossover(_mesh_cells(), _pinn_cells(5_000), probe_rows=probe,
                     targets=(1e-1,))
    assert "1/2 seeds" in c["probe"]
    assert c["d_pinn_last_reached"] == 8
    # Order must not decide the answer.
    (rev,) = crossover(_mesh_cells(), _pinn_cells(5_000),
                       probe_rows=probe[::-1], targets=(1e-1,))
    assert rev["probe"] == c["probe"]
    assert rev["assumed_steps"] == c["assumed_steps"]


def _probe_trace(d, target, errs, first=250):
    return [{"d": str(d), "target": f"{target:.0e}", "seed": "0",
             "step": str(first * (i + 1)), "rel_l2": f"{e:.6e}",
             "loss": f"{e * e:.6e}", "train_seconds": f"{i:.1f}"}
            for i, e in enumerate(errs)]


def test_probe_trend_recovers_a_clean_power_law_in_every_window():
    """The control: when the rate *is* identified, all four windows agree."""
    steps = np.arange(1, 41) * 250.0
    errs = 3.0 * steps ** -0.5
    t = probe_trend(4, 1e-2, _probe_trace(4, 1e-2, errs))
    assert all(f["exponent"] == pytest.approx(-0.5, abs=1e-6) for f in t["fits"])
    assert t["exponent_max"] - t["exponent_min"] < 1e-5
    assert t["tail_spread"] < 1.5


def test_probe_trend_shows_the_windows_disagreeing_when_the_tail_oscillates():
    """And the case that actually arose: an oscillating tail, so no rate.

    A power law with a factor-of-3 oscillation laid over it. The point of the
    function is that the four windows come apart, so that a single fitted
    exponent is visibly not a thing this trajectory has.
    """
    steps = np.arange(1, 41) * 250.0
    errs = 3.0 * steps ** -0.5 * (1.0 + 2.0 * (np.arange(40) % 4 == 3))
    t = probe_trend(4, 1e-2, _probe_trace(4, 1e-2, errs))
    assert t["exponent_max"] - t["exponent_min"] > 0.1
    assert t["tail_spread"] > 2.5


def test_probe_trend_is_none_without_enough_points():
    assert probe_trend(4, 1e-2, _probe_trace(4, 1e-2, [1.0, 0.5])) is None


def test_probe_trend_reads_only_its_own_cell():
    rows = (_probe_trace(4, 1e-2, list(3.0 * (np.arange(1, 41) * 250.0) ** -0.5))
            + _probe_trace(8, 1e-1, [9.9] * 40))
    t = probe_trend(4, 1e-2, rows)
    assert t["best"] < 1.0
    assert t["steps"] == 10_000


def test_pinn_cost_model_reproduces_section_11s_committed_fit():
    """13.4 + 13.1 d ms/step, the number Sec. 11 prints."""
    a, b = pinn_cost_model()
    assert a == pytest.approx(13.4, abs=0.15)
    assert b == pytest.approx(13.1, abs=0.15)


# ---------------------------------------------------------------------------
# The budget guard this section's dimensions found
# ---------------------------------------------------------------------------
def test_required_nx_keeps_fitting_a_rate_where_it_can():
    """The common path is untouched: Sec. 10's d = 3 cell, digit for digit."""
    r = required_nx(HighDHeat(3), 1e-2)
    assert r["nx"] == 8
    assert r["rel_l2"] == pytest.approx(6.790584e-03, rel=1e-6)
    assert r["fitted_order"] == pytest.approx(1.9962, abs=1e-4)


def test_the_loose_target_is_reachable_by_a_mesh_at_every_dimension_tested():
    """The claim the headline rests on: N = 4 holds 1e-1 as d grows.

    Sec. 10's ``flatness_study`` measured this at N = 16 for d = 1..6. It is the
    coarse grid that matters here, and the dimensions that matter are higher, so
    it is measured again on the grid and at the dimensions actually used. Capped
    at d = 10 to keep the test cheap; the sweep itself runs to 16.

    What is asserted is the *band*, not monotonicity. The error rises from
    d = 1 to d = 2 (2.94e-2 -> 3.06e-2) and falls from there -- the same shape
    Sec. 10's N = 16 study shows (1.61e-3 -> 1.90e-3 -> 1.70e-3 -> ...), so an
    assertion of monotone improvement would be asserting something the
    measurement does not say, at either resolution.
    """
    errs = [solve(HighDHeat(d), 4)["rel_l2"] for d in (1, 2, 4, 6, 8, 10)]
    assert all(e <= 1e-1 for e in errs)
    assert max(errs) / min(errs) < 2.0
    assert errs[-1] < errs[0]
    assert errs[1:] == sorted(errs[1:], reverse=True)


# ---------------------------------------------------------------------------
# The probe path, end to end but tiny
# ---------------------------------------------------------------------------
def test_budget_probe_runs_and_records_a_miss():
    """A budget far too small to reach the target must say so, not fail."""
    base = dict(n_interior=64, n_ic=16, n_bc=16, width=8, depth=2, steps=4,
                lr=1e-3)
    row, history = budget_probe(2, 1e-6, 4, seed=0, base=base, eval_every=2,
                                verbose=False)
    assert row["reached"] == 0
    assert row["cross_seconds"] == ""
    assert float(row["best_rel_l2"]) > 1e-6
    assert len(history) >= 2


def test_budget_probe_records_a_hit_with_its_cost():
    """A target any initialization clears: the crossing is the first evaluation."""
    base = dict(n_interior=64, n_ic=16, n_bc=16, width=8, depth=2, steps=4,
                lr=1e-3)
    row, _ = budget_probe(2, 1e3, 4, seed=0, base=base, eval_every=2,
                          verbose=False)
    assert row["reached"] == 1
    assert float(row["cross_seconds"]) >= 0.0
    assert float(row["cross_rel_l2"]) <= 1e3


def test_probe_sweep_resumes_without_discarding_a_finished_cell(monkeypatch):
    """The failure that motivated ``resume``: an interrupted sweep lost work.

    Rewriting the CSV from only the current invocation's rows would drop cells
    an earlier run had already paid for -- 25 minutes of CPU each. A cell in the
    committed file is skipped and carried through untouched.
    """
    import highd_crossover as hc

    existing = [{"d": "4", "target": "1e-02", "seed": "0", "steps": "20000",
                 "reached": "0", "best_rel_l2": "1.978855e-02"}]
    monkeypatch.setattr(hc, "read_csv", lambda name: existing if name ==
                        hc.PROBE_CSV else [])
    monkeypatch.setattr(hc, "budget_probe", lambda *a, **k:
                        pytest.fail("re-ran a cell that was already done"))

    rows, _ = hc.probe_sweep(cells=((4, 1e-2, 20_000, (0,)),), write=False,
                             verbose=False)
    assert rows == existing


def test_probe_sweep_reruns_a_cell_whose_budget_changed(monkeypatch):
    """Matching is on (d, target, seed, steps), so a new budget is new work."""
    import highd_crossover as hc

    existing = [{"d": "4", "target": "1e-02", "seed": "0", "steps": "20000",
                 "reached": "0", "best_rel_l2": "1.978855e-02"}]
    calls = []
    monkeypatch.setattr(hc, "read_csv", lambda name: existing if name ==
                        hc.PROBE_CSV else [])
    monkeypatch.setattr(hc, "budget_probe",
                        lambda d, t, s, seed, **k: (calls.append((d, s)) or
                                                    ({"d": d}, [])))
    hc.probe_sweep(cells=((4, 1e-2, 40_000, (0,)),), write=False, verbose=False)
    assert calls == [(4, 40_000)]


def test_targets_are_ordered_loose_to_tight():
    """Report ordering and figure panels both assume it."""
    assert list(TARGETS) == sorted(TARGETS, reverse=True)


# ---------------------------------------------------------------------------
# The reproducibility check against Sec. 10
# ---------------------------------------------------------------------------
def test_compare_to_committed_pairs_cells_and_flags_a_moved_clock():
    now = [{"target": "1e-02", "d": "3", "nx": "8", "rel_l2": "6.790584e-03",
            "seconds": "0.002"}]
    then = [{"target": "1e-02", "d": "3", "nx": "8", "rel_l2": "6.790584e-03",
             "seconds": "0.001"},
            {"target": "1e-03", "d": "9", "nx": "8", "rel_l2": "1e-3",
             "seconds": "1.0"}]
    (row,) = compare_to_committed(now, committed=then)
    assert row["grid_same"] == 1
    assert row["error_same"] == 1
    assert float(row["seconds_ratio"]) == pytest.approx(2.0)
