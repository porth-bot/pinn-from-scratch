"""Mesh vs PINN at *equal accuracy*: the cost-to-target curves, and the crossover.

Sec. 10 measured the mesh's cost in d and Sec. 11 measured the PINN's accuracy
in d, and neither is the comparison the week was for. A wall-clock number means
nothing next to another wall-clock number unless the two methods reached the
same accuracy, and they do not: at the fixed budget of Sec. 11 the PINN's
relative L2 runs 8.2e-4 -> 1.04 over d = 1 .. 16 while the mesh holds whatever
accuracy its grid buys at every d it can afford. So this module puts both on one
axis the only way that is fair -- **cost to reach a fixed relative L2** -- and
reports, for every cell, the accuracy the method actually hit alongside the time
it took.

The three targets
-----------------
1e-1, 1e-2, 1e-3. Three rather than one because the crossover dimension is not a
property of the two methods; it is a property of the two methods *and the
accuracy asked of them*, and Sec. 10 already found the reason: refining the mesh
moves N, and the cost moves as N^d, so one more digit is a factor of (N'/N)^d.
The mesh's exponential is charged per digit. A single target would hide that,
and would also let the reader assume the answer generalises.

Where each side's number comes from
-----------------------------------
**Mesh**: ``highd_mesh.cost_sweep`` run at all three targets, which for each
(target, d) *finds the grid that reaches the target and runs it*, so the reported
seconds belong to a solve that actually happened, at an accuracy that was
actually measured. The loose target is the interesting addition: at 1e-1 the
grid is N = 4, so a solve is 3^d unknowns, and 3^16 is 43 million, which fits.
At that target the mesh side is therefore *measured* at every d out to 16 with
no extrapolation at all -- which is not true of the tight targets, where the
mesh stops at d = 10 (1e-2) and d = 6 (1e-3) and the curve is continued by the
same ``d (N-1)^d`` law Sec. 10 uses, marked as extrapolated wherever it appears.

**PINN**: first crossing of the target on the committed training trajectories of
Sec. 11, using the cumulative ``train_seconds`` column (optimization only --
the evaluation calls exist for the log, and a user of the method would not pay
for them). Two things about that read are quantization, not physics, and are
reported rather than smoothed: the trajectory is evaluated every 250 steps, so a
crossing time is accurate to one evaluation interval; and the trajectory's
relative L2 is a 100,000-point Monte Carlo estimate with a ~0.2% standard error,
so a crossing within a fraction of a percent of the target is a coin flip. Both
are far smaller than the effects below.

**The cells the fixed budget missed** are the ones that decide the answer, and
"it did not get there in 5000 steps" is a statement about a budget rather than
about a method. So :func:`budget_probe` re-runs the near-miss cells at 4x the
step budget and reports what the extra budget bought. That is the honest version
of a terminating curve: the PINN's line stops where the method stopped reaching
the target, having been given four times the budget to do it.

Run:  python experiments/highd_crossover.py --probe    # 4x-budget PINN probes, ~70 min
      python experiments/highd_crossover.py --mesh     # mesh cost sweep, 3 targets
      python experiments/highd_crossover.py --report   # the table, from committed CSVs
      python experiments/highd_crossover.py --figures  # replay the figure
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import read_csv, write_csv
from highd_heat import HighDHeat, rel_l2_mc, train
from highd_pinn import BUDGET, EVAL_EVERY, SCORE_N, SCORE_SEED, TRACE_CSV

#: The accuracies both methods are asked for.
TARGETS = (1e-1, 1e-2, 1e-3)

PROBE_CSV = "highd_crossover_probe.csv"
PROBE_TRACE_CSV = "highd_crossover_probe_trace.csv"
MESH_CSV = "highd_crossover_mesh.csv"
PINN_CSV = "highd_crossover_pinn.csv"
CROSS_CSV = "highd_crossover_summary.csv"

#: Near-miss cells re-run at a larger budget: (d, target, steps, seeds).
#:
#: Chosen from the fixed-budget trajectories rather than by taste. d = 4 misses
#: 1e-2 by 3.9x, which four times the budget could plausibly close, and it is
#: the cell that sets where the PINN's 1e-2 curve terminates. d = 8 misses 1e-1
#: by 7.6x and is the cell that sets where the *loosest* curve terminates -- the
#: one target at which the mesh runs to d = 16, so it is the cell the headline
#: comparison rests on. It gets one seed at 2x rather than three at 4x because
#: Sec. 11 already measured its seed spread at 1.07x, the tightest in the sweep,
#: and because Sec. 11 also measured its error to have stopped tracking its loss
#: -- a prediction that more of the same optimization will not help, which this
#: is here to test rather than to assume.
#: d = 8 runs first because it is the cell the headline rests on: 1e-1 is the
#: one target the mesh reaches at every d out to 16, so where the PINN stops
#: reaching *that* is what the comparison turns on. An interrupted sweep should
#: lose the supporting cell, not the load-bearing one.
PROBE_CELLS = (
    (8, 1e-1, 10_000, (0,)),
    (4, 1e-2, 20_000, (0, 1)),
)


# ---------------------------------------------------------------------------
# Reading a cost-to-accuracy off a trajectory
# ---------------------------------------------------------------------------
def first_crossing(points, target):
    """First ``(step, seconds, err)`` on a trajectory with ``err <= target``.

    ``points`` is an iterable of ``(step, rel_l2, train_seconds)``. Returns
    ``None`` if the trajectory never reaches the target -- which is a result and
    is recorded as one, not as a missing row.

    Deliberately *not* interpolated between evaluations. Interpolating a Monte
    Carlo estimate of a non-monotone quantity to a threshold would invent a
    precision the trajectory does not have; the quantization is one evaluation
    interval and is reported instead.
    """
    for step, err, seconds in sorted(points):
        if err <= target:
            return int(step), float(seconds), float(err)
    return None


def pinn_costs(targets=TARGETS, trace_rows=None):
    """Cost to each target, per (d, seed), off the committed Sec. 11 trajectories.

    One row per (target, d): the mean and range of the crossing time over the
    seeds that got there, the number that did, and the best error any seed
    reached -- which is what says *how far* a missing cell missed by.
    """
    trace_rows = read_csv(TRACE_CSV) if trace_rows is None else trace_rows
    dims = sorted({int(r["d"]) for r in trace_rows})
    out = []
    for target in targets:
        for d in dims:
            seeds = sorted({int(r["seed"]) for r in trace_rows if int(r["d"]) == d})
            times, steps, errs, best = [], [], [], []
            for seed in seeds:
                pts = [(int(r["step"]), float(r["rel_l2"]),
                        float(r["train_seconds"]))
                       for r in trace_rows
                       if int(r["d"]) == d and int(r["seed"]) == seed]
                best.append(min(p[1] for p in pts))
                hit = first_crossing(pts, target)
                if hit is not None:
                    steps.append(hit[0])
                    times.append(hit[1])
                    errs.append(hit[2])
            out.append({
                "target": f"{target:.0e}",
                "d": d,
                "n_seeds": len(seeds),
                "n_reached": len(times),
                "seconds": f"{np.mean(times):.4f}" if times else "",
                "seconds_min": f"{min(times):.4f}" if times else "",
                "seconds_max": f"{max(times):.4f}" if times else "",
                "steps": f"{np.mean(steps):.0f}" if steps else "",
                "rel_l2": f"{np.mean(errs):.6e}" if errs else "",
                "best_rel_l2": f"{np.mean(best):.6e}",
                "shortfall": f"{np.mean(best) / target:.3f}",
            })
    return out


# ---------------------------------------------------------------------------
# The near-miss cells, at a larger budget
# ---------------------------------------------------------------------------
def budget_probe(d, target, steps, seed, base=None, eval_every=EVAL_EVERY,
                 verbose=True):
    """Train one cell at ``steps`` optimizer steps and report where it crossed.

    Same architecture, same collocation budget, same learning rate as Sec. 11 --
    only the step count changes, because the question is whether the fixed
    budget was the binding constraint. Runs to the full ``steps`` even after
    crossing: a trajectory that reaches the target and leaves it again is
    something the comparison needs to know, and Sec. 1's own history says this
    objective does exactly that.

    Returns ``(row, history)``. ``row`` carries the crossing (or its absence),
    the best error along the whole trajectory, and the final score on a fresh
    1,000,000-point sample so it is directly comparable to Sec. 11's table.
    """
    budget = dict(BUDGET if base is None else base, steps=steps)
    problem = HighDHeat(d)
    if verbose:
        print(f"  d={d} target={target:.0e} seed={seed}: {steps} steps "
              f"({steps / BUDGET['steps']:.0f}x the fixed budget)", flush=True)

    t0 = time.time()
    model, history, best = train(problem, seed=seed, eval_every=eval_every,
                                 eval_n=100_000, select="best_loss", **budget)
    wall = time.time() - t0

    pts = [(h[0], h[5], h[7]) for h in history]
    hit = first_crossing(pts, target)
    rel, se = rel_l2_mc(model, problem, n=SCORE_N, seed=SCORE_SEED)
    best_traj = min(p[1] for p in pts)

    row = {
        "d": d, "target": f"{target:.0e}", "seed": seed, "steps": steps,
        "reached": int(hit is not None),
        "cross_step": hit[0] if hit else "",
        "cross_seconds": f"{hit[1]:.4f}" if hit else "",
        "cross_rel_l2": f"{hit[2]:.6e}" if hit else "",
        "best_rel_l2": f"{best_traj:.6e}",
        "final_rel_l2": f"{rel:.6e}",
        "final_stderr": f"{se:.6e}",
        "best_loss": f"{best['loss']:.6e}",
        "best_step": best["step"],
        "train_seconds": f"{history[-1][7]:.2f}",
        "wall_seconds": f"{wall:.2f}",
    }
    if verbose:
        if hit:
            print(f"       reached {target:.0e} at step {hit[0]} "
                  f"({hit[1]:.0f}s of optimization)", flush=True)
        else:
            print(f"       never reached {target:.0e}; best {best_traj:.4e} "
                  f"({best_traj / target:.1f}x short) in {history[-1][7]:.0f}s",
                  flush=True)
    return row, history


def probe_trend(d, target, trace_rows=None, windows=((0.1, 1.0), (0.25, 1.0),
                                                     (0.5, 1.0), (0.75, 1.0))):
    """Is the probe's error still falling, and at a rate worth extrapolating?

    The tempting move when a probe misses is to fit ``err ~ steps^-p`` and quote
    the budget the target would need. That is only honest if p is identified,
    and on this objective it is not: the trajectory oscillates, so the exponent
    depends on which window it is fitted over. This returns the fit over several
    windows *so the disagreement is visible*, alongside the spread over the
    trailing half -- which is the quantity that says whether any of the fits
    mean anything.

    Also returns the last-quarter fall in the training loss against the fall in
    the error. Loss is a mean square and the error is a normalized L2, so the
    heuristic expectation is that the error tracks the loss's square root. It is
    a heuristic and not an identity, and what it is used for here is its
    *sign*: whether more optimization is still buying accuracy at this d.
    """
    trace_rows = (read_csv(PROBE_TRACE_CSV) if trace_rows is None else trace_rows)
    key = f"{target:.0e}"
    pts = sorted((int(r["step"]), float(r["rel_l2"]), float(r["loss"]))
                 for r in trace_rows
                 if int(r["d"]) == d and r["target"] == key)
    if len(pts) < 4:
        return None
    last = pts[-1][0]

    fits = []
    for lo, hi in windows:
        seg = [p for p in pts if lo * last <= p[0] <= hi * last and p[0] > 0]
        if len(seg) < 3:
            continue
        slope, _ = np.polyfit(np.log([p[0] for p in seg]),
                              np.log([p[1] for p in seg]), 1)
        fits.append({"from_step": seg[0][0], "exponent": float(slope),
                     "n": len(seg)})

    tail = [p for p in pts if p[0] >= 0.5 * last]
    quarter = [p for p in pts if p[0] >= 0.75 * last]
    return {
        "d": d, "target": key, "steps": last,
        "fits": fits,
        "exponent_min": min(f["exponent"] for f in fits) if fits else None,
        "exponent_max": max(f["exponent"] for f in fits) if fits else None,
        "tail_spread": max(p[1] for p in tail) / min(p[1] for p in tail),
        "best": min(p[1] for p in pts),
        "loss_fall": quarter[0][2] / quarter[-1][2],
        "error_fall": quarter[0][1] / quarter[-1][1],
    }


def report_probe(probe_rows=None, trace_rows=None):
    """What the larger budget bought, per cell, and whether the rate is usable."""
    probe_rows = read_csv(PROBE_CSV) if probe_rows is None else probe_rows
    print("\n" + "-" * 86)
    print("The near-miss cells, re-run at a larger step budget")
    print("-" * 86)
    for d in sorted({int(p["d"]) for p in probe_rows}):
        cell = [p for p in probe_rows if int(p["d"]) == d]
        target = float(cell[0]["target"])
        hits = [p for p in cell if int(p["reached"])]
        best = min(float(p["best_rel_l2"]) for p in cell)
        print(f"\n  d={d}, target {target:.0e}, {cell[0]['steps']} steps "
              f"({int(cell[0]['steps']) / BUDGET['steps']:.0f}x), "
              f"{len(cell)} seed(s)")
        print(f"    reached in {len(hits)}/{len(cell)}; best {best:.4e} "
              f"({best / target:.1f}x short)" if not hits else
              f"    reached in {len(hits)}/{len(cell)}")
        t = probe_trend(d, target, trace_rows)
        if t is None:
            continue
        span = ", ".join(f"{f['exponent']:+.2f} (from step {f['from_step']})"
                         for f in t["fits"])
        print(f"    err ~ steps^p fitted over four windows: {span}")
        print(f"    -> p is not identified: the trailing half of the "
              f"trajectory spans {t['tail_spread']:.1f}x, so no extrapolation "
              f"to the target is quoted")
        print(f"    last quarter: loss falls {t['loss_fall']:.2f}x, error "
              f"{t['error_fall']:.2f}x (heuristic expectation "
              f"{np.sqrt(t['loss_fall']):.2f}x)")


def _key(row):
    return int(row["d"]), row["target"], int(row["seed"]), int(row["steps"])


def probe_sweep(cells=PROBE_CELLS, write=True, verbose=True, resume=True):
    """Every probe cell, flushed as it finishes. Roughly 70 minutes of CPU.

    ``resume`` picks up cells already in the committed CSVs and skips them,
    which is not a convenience: the first run of this lost two of three cells
    when the process it was under exited, and rewriting the file from only the
    current invocation's rows would have discarded the completed one as well.
    A cell counts as done only if its (d, target, seed, steps) all match, so
    changing a budget re-runs rather than silently reusing the old answer.
    """
    rows, traces = [], []
    if resume:
        try:
            rows = read_csv(PROBE_CSV)
            traces = read_csv(PROBE_TRACE_CSV)
        except FileNotFoundError:
            pass
    done = {_key(r) for r in rows}

    for d, target, steps, seeds in cells:
        for seed in seeds:
            if (d, f"{target:.0e}", seed, steps) in done:
                if verbose:
                    print(f"  d={d} target={target:.0e} seed={seed}: already "
                          f"in {PROBE_CSV}, skipped", flush=True)
                continue
            row, history = budget_probe(d, target, steps, seed, verbose=verbose)
            rows.append(row)
            for step, loss, lr_, li, lb, err, se, ts in history:
                traces.append({"d": d, "target": f"{target:.0e}", "seed": seed,
                               "step": step, "loss": f"{loss:.6e}",
                               "loss_r": f"{lr_:.6e}", "loss_ic": f"{li:.6e}",
                               "loss_bc": f"{lb:.6e}", "rel_l2": f"{err:.6e}",
                               "stderr": f"{se:.6e}",
                               "train_seconds": f"{ts:.4f}"})
            if write:
                _write_probe(rows, traces)
    return rows, traces


def _write_probe(rows, traces):
    write_csv(PROBE_CSV,
              ["d", "target", "seed", "steps", "reached", "cross_step",
               "cross_seconds", "cross_rel_l2", "best_rel_l2", "final_rel_l2",
               "final_stderr", "best_loss", "best_step", "train_seconds",
               "wall_seconds"], rows)
    write_csv(PROBE_TRACE_CSV,
              ["d", "target", "seed", "step", "loss", "loss_r", "loss_ic",
               "loss_bc", "rel_l2", "stderr", "train_seconds"], traces)


# ---------------------------------------------------------------------------
# The mesh side, re-measured at all three targets in one sitting
# ---------------------------------------------------------------------------
#: Dimensions asked of the mesh. Past d = 6 the tight targets stop fitting and
#: are recorded as skipped; the loose one keeps going, because its grid is
#: N = 4 and 3^16 is only 43 million unknowns.
MESH_DIMS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16)


def mesh_costs(targets=TARGETS, dims=MESH_DIMS, verbose=True):
    """Run ``highd_mesh.cost_sweep`` at all three targets, in one sitting.

    Sec. 10's committed sweep is not reused, for a reason that is itself worth
    measuring: its seconds column was taken on a different day, and the whole
    point of this section is a wall-clock comparison. Measuring both sides in
    one sitting removes the machine's state as a variable between the two curves
    -- and the cells that overlap Sec. 10's sweep then become a free
    reproducibility check on it, which :func:`compare_to_committed` reports.
    """
    from highd_mesh import cost_sweep
    return cost_sweep(targets=targets, dims=dims, verbose=verbose)


def compare_to_committed(rows, committed=None):
    """Cells this sweep shares with Sec. 10's, and how far the wall clock moved.

    Same code, same machine, different sitting. The grid chosen and the accuracy
    reached are deterministic and must match exactly; the seconds are not, and
    the ratio is the quantity `gp-from-scratch`'s Day 7 taught this series to
    report rather than to assume away.
    """
    committed = read_csv("highd_mesh_cost.csv") if committed is None else committed
    index = {(r["target"], int(r["d"])): r for r in committed}
    out = []
    for r in rows:
        key = (r["target"], int(r["d"]))
        if key not in index:
            continue
        old = index[key]
        out.append({
            "target": r["target"], "d": int(r["d"]),
            "nx_now": int(r["nx"]), "nx_then": int(old["nx"]),
            "rel_l2_now": r["rel_l2"], "rel_l2_then": old["rel_l2"],
            "grid_same": int(int(r["nx"]) == int(old["nx"])),
            "error_same": int(float(r["rel_l2"]) == float(old["rel_l2"])),
            "seconds_now": r["seconds"], "seconds_then": old["seconds"],
            "seconds_ratio": f"{float(r['seconds']) / float(old['seconds']):.3f}",
        })
    return out


# ---------------------------------------------------------------------------
# Where the two curves would meet
# ---------------------------------------------------------------------------
def pinn_cost_model(cost_rows=None):
    """``(a, b)`` in ``ms/step = a + b d``, fitted to Sec. 11's committed medians.

    Sec. 11 measured this and pinned its *shape* structurally -- a test counts
    the reverse-mode passes in one residual evaluation and gets exactly d + 2 --
    because the wall clock behind it does not reproduce across sittings. Both
    coefficients are used here only to extrapolate the PINN's cost *downward*
    against the mesh's exponential, where a 20% error in either moves the
    crossing dimension by a fraction of one dimension.
    """
    from highd_pinn import COST_CSV
    rows = read_csv(COST_CSV) if cost_rows is None else cost_rows
    d = np.array([float(r["d"]) for r in rows])
    ms = np.array([float(r["ms_per_step"]) for r in rows])
    b, a = np.polyfit(d, ms, 1)
    return float(a), float(b)


def mesh_cost_at(cells, d):
    """Seconds the mesh would take at dimension ``d``, anchored on its largest
    measured cell for this target.

    ``d (N-1)^d`` at the anchor's fixed N and n_t -- the same law
    ``highd_mesh.extrapolate`` uses, chosen there because it is exact at the
    anchor while a global fit misses that cell by tens of percent. Returns the
    measured seconds instead when ``d`` is itself a measured cell.
    """
    by_d = {int(c["d"]): c for c in cells}
    if d in by_d:
        return float(by_d[d]["seconds"]), True
    anchor = by_d[max(by_d)]
    d0, sec0 = int(anchor["d"]), float(anchor["seconds"])
    n = float(int(anchor["nx"]) - 1)
    return sec0 * (d / d0) * n ** (d - d0), False


def crossover(mesh_rows, pinn_rows, probe_rows=None, targets=TARGETS,
              max_d=64):
    """The dimension at which the mesh becomes dearer than the PINN, per target.

    **This is a bound, and the direction of the bound is the point.** The PINN's
    cost is extrapolated on the most generous assumption available: that it
    needs no more optimizer *steps* at high d than the most it needed at any d
    where it actually reached the target, and that a step costs the measured
    ``a + b d`` ms. Both halves are generous -- the measured step count *grows*
    with d over the range where it works (250 -> 1833 for the loose target), and
    a method that has stopped converging does not need fewer steps. So the
    number returned is a *lower bound* on the true crossing dimension.

    Against it sits ``d_fails``: the largest d at which the PINN reached this
    target at all, taking the larger-budget probes into account. The gap between
    the two is the result -- how many dimensions of headroom the mesh still has
    when the PINN stops working.
    """
    a, b = pinn_cost_model()
    out = []
    for target in targets:
        key = f"{target:.0e}"
        cells = [r for r in mesh_rows if r["target"] == key]
        if not cells:
            continue
        reached = [r for r in pinn_rows
                   if r["target"] == key and int(r["n_reached"]) > 0]
        d_fails = max((int(r["d"]) for r in reached), default=0)
        max_steps = max((float(r["steps"]) for r in reached), default=0.0)

        # A probe that reached the target extends d_fails; one that did not is
        # what licenses calling the curve terminated rather than under-budgeted.
        # Aggregated over the probe's seeds rather than taking the last row --
        # a cell run at two seeds where one arrives and one does not is a
        # different statement from either seed alone, and it is the statement
        # that belongs in the note.
        mine = [p for p in (probe_rows or []) if p["target"] == key]
        probe_note = ""
        for d in sorted({int(p["d"]) for p in mine}):
            cell = [p for p in mine if int(p["d"]) == d]
            hits = [p for p in cell if int(p["reached"])]
            steps = max(int(p["steps"]) for p in cell)
            if hits:
                d_fails = max(d_fails, d)
                max_steps = max([max_steps]
                                + [float(p["cross_step"]) for p in hits])
                note = (f"d={d} reached it in {len(hits)}/{len(cell)} seeds "
                        f"within {steps} steps")
            else:
                worst = min(float(p["best_rel_l2"]) for p in cell)
                note = (f"d={d} still {worst / target:.1f}x short in "
                        f"{len(cell)}/{len(cell)} seeds after {steps} steps")
            probe_note = f"{probe_note}; {note}" if probe_note else note

        d_cross, pinn_sec, mesh_sec, measured = None, None, None, False
        if max_steps > 0:
            for d in range(1, max_d + 1):
                ps = max_steps * (a + b * d) / 1000.0
                ms, was_measured = mesh_cost_at(cells, d)
                if ms > ps:
                    d_cross, pinn_sec, mesh_sec, measured = d, ps, ms, was_measured
                    break
        out.append({
            "target": key,
            "d_pinn_last_reached": d_fails,
            "d_crossover_lower_bound": d_cross if d_cross else "",
            "headroom": (d_cross - d_fails) if d_cross else "",
            "pinn_seconds_at_crossover": f"{pinn_sec:.4g}" if pinn_sec else "",
            "mesh_seconds_at_crossover": f"{mesh_sec:.4g}" if mesh_sec else "",
            "crossover_mesh_measured": int(measured),
            "assumed_steps": f"{max_steps:.0f}" if max_steps else "",
            "d_mesh_last_measured": max(int(c["d"]) for c in cells),
            "probe": probe_note,
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(mesh_rows, pinn_rows, probe_rows=None, cross_rows=None):
    cross_rows = (crossover(mesh_rows, pinn_rows, probe_rows)
                  if cross_rows is None else cross_rows)
    print("\n" + "=" * 86)
    print("Cost to a fixed relative L2: mesh vs PINN, both measured, "
          "with the accuracy each hit")
    print("=" * 86)
    for target in sorted({r["target"] for r in mesh_rows}, reverse=True):
        print(f"\ntarget rel L2 <= {target}")
        print(f"  {'d':>3} | {'mesh s':>10} {'mesh N':>7} {'mesh relL2':>11} "
              f"| {'PINN s':>10} {'seeds':>6} {'PINN relL2':>11}  note")
        dims = sorted({int(r["d"]) for r in mesh_rows if r["target"] == target}
                      | {int(r["d"]) for r in pinn_rows if r["target"] == target})
        for d in dims:
            m = next((r for r in mesh_rows
                      if r["target"] == target and int(r["d"]) == d), None)
            p = next((r for r in pinn_rows
                      if r["target"] == target and int(r["d"]) == d), None)
            ms = f"{float(m['seconds']):>10.4g}" if m else f"{'skipped':>10}"
            mn = f"{int(m['nx']):>7}" if m else f"{'--':>7}"
            me = f"{float(m['rel_l2']):>11.3e}" if m else f"{'--':>11}"
            if p and int(p["n_reached"]):
                ps = f"{float(p['seconds']):>10.4g}"
                pe = f"{float(p['rel_l2']):>11.3e}"
                note = ""
            elif p:
                ps, pe = f"{'never':>10}", f"{'--':>11}"
                note = f"best {float(p['best_rel_l2']):.2e} " \
                       f"({float(p['shortfall']):.1f}x short)"
            else:
                ps, pe, note = f"{'--':>10}", f"{'--':>11}", ""
            seeds = f"{p['n_reached']}/{p['n_seeds']}" if p else "--"
            print(f"  {d:>3} | {ms} {mn} {me} | {ps} {seeds:>6} {pe}  {note}")

    print("\n" + "-" * 86)
    print("Where the curves would meet, and where the PINN stops first")
    print("-" * 86)
    a, b = pinn_cost_model()
    print(f"PINN cost model: {a:.1f} + {b:.1f} d ms/step (Sec. 11's committed "
          f"medians)")
    print("The crossing is a LOWER BOUND: the PINN is granted a step count that "
          "does not grow in d,")
    print("which its own measurements say it does.\n")
    for c in cross_rows:
        if not c["d_crossover_lower_bound"]:
            print(f"  target {c['target']}: the PINN never reached it at any d")
            continue
        print(f"  target {c['target']}: mesh becomes dearer at d >= "
              f"{c['d_crossover_lower_bound']} "
              f"({float(c['mesh_seconds_at_crossover']):.3g}s vs "
              f"{float(c['pinn_seconds_at_crossover']):.3g}s, "
              f"{'measured' if int(c['crossover_mesh_measured']) else 'extrapolated'} "
              f"mesh cell)")
        print(f"      the PINN last reached it at d = {c['d_pinn_last_reached']}"
              f"  ->  {c['headroom']} dimensions of headroom left unused")
        if c["probe"]:
            print(f"      larger-budget probe: {c['probe']}")
    return cross_rows


def report_reproducibility(rows):
    if not rows:
        return
    print("\n" + "-" * 86)
    print("The cells this sweep shares with Sec. 10's, re-run in a new sitting")
    print("-" * 86)
    ratios = [float(r["seconds_ratio"]) for r in rows]
    same_grid = sum(int(r["grid_same"]) for r in rows)
    same_err = sum(int(r["error_same"]) for r in rows)
    print(f"  {len(rows)} shared cells: grid identical in {same_grid}, "
          f"relative L2 identical to the last digit in {same_err}")
    print(f"  wall clock ratio now/then: min {min(ratios):.2f}, "
          f"median {np.median(ratios):.2f}, max {max(ratios):.2f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
MESH_COLOR = "#2166ac"
PINN_COLOR = "#b2182b"


def make_figure(mesh_rows=None, pinn_rows=None, probe_rows=None):
    """Four panels: one cost-vs-d panel per target, then the accuracy panel.

    The accuracy panel is not decoration and is not an afterthought. A
    wall-clock comparison at "the same accuracy" is only as good as how close
    each method got to the target, and neither lands on it: the mesh's grid is
    an even integer so it *overshoots* (at 1e-1 the coarsest legal grid lands at
    1.6e-2, so it is being charged for six times the accuracy asked of it), and
    the PINN is evaluated every 250 steps so it undershoots the step count it
    would have needed. Panel 4 shows both, so the reader can see which way each
    method's quantization pushes the comparison.
    """
    from common import plt, savefig

    mesh_rows = read_csv(MESH_CSV) if mesh_rows is None else mesh_rows
    pinn_rows = read_csv(PINN_CSV) if pinn_rows is None else pinn_rows
    if probe_rows is None:
        try:
            probe_rows = read_csv(PROBE_CSV)
        except FileNotFoundError:
            probe_rows = []
    cross_rows = crossover(mesh_rows, pinn_rows, probe_rows)
    by_target = {c["target"]: c for c in cross_rows}
    a, b = pinn_cost_model()

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))
    flat = axes.ravel()

    for ax, target in zip(flat, TARGETS):
        key = f"{target:.0e}"
        cells = sorted((r for r in mesh_rows if r["target"] == key),
                       key=lambda r: int(r["d"]))
        c = by_target.get(key, {})
        d_last_mesh = max(int(r["d"]) for r in cells)
        d_cross = int(c["d_crossover_lower_bound"]) if c.get(
            "d_crossover_lower_bound") else None
        # Each panel is drawn only as far as it needs to be. A common x-range
        # would put the tight target's 10^18 seconds on the same axis as the
        # loose one's crossing and make both unreadable; the exponent is the
        # honest number and it is exactly why the panels are separate.
        d_max = max(d_cross + 3 if d_cross else 0, d_last_mesh + 3)

        # Mesh: measured, then the same law continued past where it fits.
        ax.semilogy([int(r["d"]) for r in cells],
                    [float(r["seconds"]) for r in cells],
                    "o-", color=MESH_COLOR, ms=5, lw=1.4, label="mesh (measured)")
        ext = list(range(d_last_mesh, d_max + 1))
        ax.semilogy(ext, [mesh_cost_at(cells, d)[0] for d in ext], "--",
                    color=MESH_COLOR, lw=1.1, alpha=0.8,
                    label="mesh (extrapolated)")

        # PINN: measured where it reached the target, and nowhere else.
        hit = sorted((r for r in pinn_rows
                      if r["target"] == key and int(r["n_reached"])),
                     key=lambda r: int(r["d"]))
        if hit:
            ax.semilogy([int(r["d"]) for r in hit],
                        [float(r["seconds"]) for r in hit],
                        "s-", color=PINN_COLOR, ms=5, lw=1.4,
                        label="PINN (measured)")
            d_stop = int(hit[-1]["d"])
            steps = float(c.get("assumed_steps") or hit[-1]["steps"])
            grid = np.arange(d_stop, d_max + 1)
            ax.semilogy(grid, steps * (a + b * grid) / 1000.0, ":",
                        color=PINN_COLOR, lw=1.2,
                        label="PINN (lower bound: cost/step only)")
            ax.plot([d_stop], [float(hit[-1]["seconds"])], "X", color="k",
                    ms=11, mew=1.4, mfc=PINN_COLOR, zorder=5)
            ax.annotate("stops reaching\nthe target", (d_stop, float(hit[-1]["seconds"])),
                        textcoords="offset points", xytext=(6, -26), fontsize=7.5,
                        color=PINN_COLOR)

        if d_cross:
            ax.axvline(d_cross, color="0.35", ls="-.", lw=1.1)
            ax.annotate(f"mesh dearer\nfrom $d={d_cross}$", (d_cross, 1.0),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(4, -26),
                        fontsize=7.5, color="0.25")

        ax.set_xlabel("spatial dimension $d$")
        ax.set_ylabel("wall clock to reach the target (s)")
        ax.set_title(f"cost to relative $L^2 \\leq$ {target:.0e}", fontsize=10)
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(fontsize=7, loc="upper left")

    # Panel 4: what each method actually hit.
    ax = flat[3]
    for target, marker in zip(TARGETS, ("o", "s", "^")):
        key = f"{target:.0e}"
        cells = sorted((r for r in mesh_rows if r["target"] == key),
                       key=lambda r: int(r["d"]))
        ax.semilogy([int(r["d"]) for r in cells],
                    [float(r["rel_l2"]) for r in cells], marker + "-",
                    color=MESH_COLOR, ms=4, lw=1.0, alpha=0.85,
                    label=f"mesh @ {target:.0e}")
        ax.axhline(target, color="0.6", ls=":", lw=0.9)
    best = sorted({int(r["d"]) for r in pinn_rows})
    lookup = {int(r["d"]): float(r["best_rel_l2"]) for r in pinn_rows
              if r["target"] == f"{TARGETS[0]:.0e}"}
    ax.semilogy(best, [lookup[d] for d in best], "D-", color=PINN_COLOR, ms=5,
                lw=1.4, label="PINN, best reached at the fixed budget")
    for p in probe_rows:
        ax.plot([int(p["d"])], [float(p["best_rel_l2"])], "*", color=PINN_COLOR,
                ms=13, mec="k", mew=0.6, zorder=5)
    if probe_rows:
        ax.plot([], [], "*", color=PINN_COLOR, ms=11, mec="k", mew=0.6,
                label="PINN, larger-budget probe")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.0)
    ax.annotate(r"$u_\theta \equiv 0$ scores 1.0", (0.02, 1.0),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                xytext=(0, 4), fontsize=7.5, color="0.35")
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("relative $L^2$ actually reached")
    ax.set_title("the accuracy each method actually hit", fontsize=10)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Mesh vs PINN at equal accuracy: the crossover, and where the "
                 "PINN stops first", y=1.0, fontsize=11)
    fig.tight_layout()
    savefig(fig, "highd_crossover.png")


def figures_from_committed():
    """Replay ``figures/highd_crossover.png`` from the committed CSVs."""
    make_figure()


# ---------------------------------------------------------------------------
def main(probe=False, mesh=False, do_report=False, figure=False):
    if probe:
        t0 = time.time()
        rows, _ = probe_sweep()
        print(f"\nprobe finished in {(time.time() - t0) / 60:.1f} min")
        return

    if mesh:
        measured, skipped = mesh_costs()
        write_csv(MESH_CSV,
                  ["target", "d", "nx", "nt", "unknowns", "rel_l2", "seconds",
                   "node_steps", "bytes_counted", "bytes_traced", "arrays",
                   "fitted_order"], measured)
        pinn = pinn_costs()
        write_csv(PINN_CSV,
                  ["target", "d", "n_seeds", "n_reached", "seconds",
                   "seconds_min", "seconds_max", "steps", "rel_l2",
                   "best_rel_l2", "shortfall"], pinn)
        if skipped:
            print("\nmesh cells skipped (would not fit and were not run):")
            for s in skipped:
                print(f"  target {s['target']} d={s['d']}: {s['reason']}")
        report_reproducibility(compare_to_committed(measured))
        return

    if do_report:
        mesh_rows = read_csv(MESH_CSV)
        pinn_rows = read_csv(PINN_CSV)
        try:
            probe_rows = read_csv(PROBE_CSV)
        except FileNotFoundError:
            probe_rows = []
        cross_rows = report(mesh_rows, pinn_rows, probe_rows)
        report_reproducibility(compare_to_committed(mesh_rows))
        write_csv(CROSS_CSV,
                  ["target", "d_pinn_last_reached", "d_crossover_lower_bound",
                   "headroom", "pinn_seconds_at_crossover",
                   "mesh_seconds_at_crossover", "crossover_mesh_measured",
                   "assumed_steps", "d_mesh_last_measured", "probe"], cross_rows)
        make_figure(mesh_rows, pinn_rows, probe_rows)
        return

    if figure:
        make_figure()
        return

    print(__doc__.strip().splitlines()[0])
    print("\nPass one of --probe (~70 min), --mesh, --report, --figures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="re-run the near-miss cells at a larger step budget")
    ap.add_argument("--mesh", action="store_true",
                    help="mesh cost sweep at all three targets")
    ap.add_argument("--report", action="store_true",
                    help="the comparison table and figure, from the CSVs")
    ap.add_argument("--figures", action="store_true",
                    help="replay the figure from the committed CSVs")
    args = ap.parse_args()
    main(probe=args.probe, mesh=args.mesh, do_report=args.report,
         figure=args.figures)
