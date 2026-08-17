"""The PINN across d = 1, 2, 4, 8, 16: one architecture, one budget, three seeds.

``experiments/highd_mesh.py`` measured the classical side of this week's
question and hit its wall at d = 6: the Douglas ADI solver reaches 1e-3 on
47 million unknowns in 26 seconds, and one more dimension needs 634 GB. This is
the other side. The network's cost per step is *linear* in d -- the residual
needs one first-derivative pass plus d second derivatives, and the parameter
count grows only through the input layer -- so if its accuracy holds up, the
crossover Sec. 10 is missing exists somewhere.

Whether the accuracy holds up is the measurement, and it is the one this file
exists to make. Everything about the design below is chosen so that a trend in
the reported error is a property of the *method*, not of the way it was scored.

What is held fixed
------------------
The same architecture (width 128, depth 4, tanh), the same optimizer and step
count (Adam, 1e-3, 5000 steps), and the same collocation budget (4000 interior,
400 initial, 400 boundary points) at every d. Nothing is tuned per dimension.
That is the point: a sweep in which each cell got its own tuning would measure
the tuner. The cost of the choice is that the budget is presumably generous at
d = 1 and tight at d = 16, so this measures *one* budget's behaviour in d and
not the best each d can do -- Day 13's job.

Three things are not exactly fixed, and pretending otherwise would be wrong:

- **The parameter count grows with d**, because the input layer is
  ``width x (d + 1)``. From d = 1 to d = 16 that is 50,049 -> 51,969 parameters,
  a 3.8% increase, reported per cell as ``params``. It is not zero and it is not
  the story.
- **The target changes with d.** It has to: there is no single function on
  ``[0,1]^d`` for all d. ``highd_heat`` fixes what it can (the fundamental's
  decay rate, via ``alpha_d = alpha_1/d``) and states what it cannot (the two
  modes' rate ratio falls from 4.0 to 1.19), so the family is comparable across
  d but not identical.
- **The metric's precision degrades with d.** ``exact_ms`` falls like ``2^-d``
  while a uniform sample of the cube mostly misses the region carrying the norm,
  so the Monte Carlo estimator's relative standard error grows like
  ``sqrt(((3/2)^d - 1)/n)``. That is measured in ``highd_heat.metric_study``
  and is why every number here is quoted with a standard error, and why the
  final score uses 1,000,000 points rather than the 100,000 used along the
  trajectory.

Seeds, and why three of them are the minimum
--------------------------------------------
A single seed would not survive the repo's own evidence. Sec. 9's loss-weighting
sweep found a 6.8x seed spread on this problem class and had to be re-run at
three seeds; Sec. 1's committed history oscillates 14x over its last 1500 steps.
So every cell runs at seeds 0, 1, 2, each reported individually in the log, with
mean, sd, min and max in the summary. Three seeds give an sd with two degrees of
freedom -- weak, and the min/max are quoted next to it for that reason -- but
they are enough to tell a 10x effect from an oscillation, which is what the
d-trend needs.

Selection is on lowest *training loss*, as ``highd_heat.train`` documents: the
loss contains no ground truth, so nothing about the exact solution enters the
choice of iterate. The final iterate is scored too, in ``rel_l2_final``, so the
gap between the two is visible rather than assumed.

Run:  python experiments/highd_pinn.py --sweep    # the full sweep, ~2 h
      python experiments/highd_pinn.py --figures  # replay from committed CSVs
      python experiments/highd_pinn.py --cost     # ms/step vs d, no training
      python experiments/highd_pinn.py --quick    # tiny end-to-end smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import read_csv, savefig, write_csv
from highd_heat import HighDHeat, exact_rms, rel_l2_mc, train

SWEEP_DIMS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)

#: The fixed budget. Every cell of the sweep runs exactly this.
BUDGET = dict(n_interior=4000, n_ic=400, n_bc=400,
              width=128, depth=4, steps=5000, lr=1e-3)

EVAL_EVERY = 250        # trajectory resolution, for the cost-to-accuracy read
EVAL_N = 100_000        # trajectory sample size
SCORE_N = 1_000_000     # final score: 10x the trajectory sample
SCORE_SEED = 7          # and an independent draw from the trajectory's 12345

SWEEP_CSV = "highd_pinn_sweep.csv"
TRACE_CSV = "highd_pinn_trace.csv"
COST_CSV = "highd_pinn_cost.csv"


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def n_params(d, width, depth):
    """Parameter count of the field network at spatial dimension ``d``.

    ``depth`` hidden layers of ``width``, input ``d + 1``, scalar output:
    ``width(d+1) + width`` for the first map, ``(depth-1)`` maps of
    ``width^2 + width``, and ``width + 1`` for the output. Computed rather than
    counted off the model so the sweep can report it without building one, and
    checked against a real model in ``tests/test_highd_pinn.py``.
    """
    first = width * (d + 1) + width
    hidden = (depth - 1) * (width * width + width)
    out = width + 1
    return int(first + hidden + out)


def run_cell(d, seed, budget=None, eval_every=EVAL_EVERY, eval_n=EVAL_N,
             score_n=SCORE_N, verbose=True):
    """Train one (d, seed) cell at the fixed budget and score it.

    Returns ``(row, history)``. ``row`` carries the two scores that matter --
    the selected iterate's relative L2 and the final iterate's, both on the same
    fresh ``score_n``-point sample, each with its standard error -- alongside
    the losses, the wall clock, and the parameter count.
    """
    budget = dict(BUDGET if budget is None else budget)
    problem = HighDHeat(d)
    if verbose:
        print(f"  d={d:2d} seed={seed}: {problem}", flush=True)

    # Trained once, scored twice. ``select="final"`` leaves the last iterate in
    # the model and hands back the selected parameters in ``best["state_dict"]``,
    # so both scores come off the same run on the same evaluation sample -- the
    # gap between them is then a property of the run and not of two runs.
    t0 = time.time()
    model, history, best = train(problem, seed=seed, eval_every=eval_every,
                                 eval_n=eval_n, select="final", **budget)
    wall = time.time() - t0

    rel_final, se_final = rel_l2_mc(model, problem, n=score_n, seed=SCORE_SEED)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    rel, se = rel_l2_mc(model, problem, n=score_n, seed=SCORE_SEED)

    step, loss, loss_r, loss_ic, loss_bc, traj_rel, traj_se, train_s = history[-1]
    row = {
        "d": d,
        "seed": seed,
        "params": n_params(d, budget["width"], budget["depth"]),
        "rel_l2": f"{rel:.6e}",
        "stderr": f"{se:.6e}",
        "rel_l2_final": f"{rel_final:.6e}",
        "stderr_final": f"{se_final:.6e}",
        "best_step": best["step"],
        "best_loss": f"{best['loss']:.6e}",
        "final_loss": f"{best['final_loss']:.6e}",
        "loss_r": f"{loss_r:.6e}",
        "loss_ic": f"{loss_ic:.6e}",
        "loss_bc": f"{loss_bc:.6e}",
        "exact_rms": f"{exact_rms(problem):.6e}",
        "train_seconds": f"{train_s:.2f}",
        "wall_seconds": f"{wall:.2f}",
        "ms_per_step": f"{1000 * train_s / (budget['steps'] + 1):.3f}",
    }
    if verbose:
        print(f"       rel L2 {rel:.4e} +- {se:.1e}   "
              f"(final iterate {rel_final:.4e})   "
              f"best loss {best['loss']:.3e} at step {best['step']}   "
              f"{train_s:.0f}s train", flush=True)
    return row, history


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def sweep(dims=SWEEP_DIMS, seeds=SEEDS, budget=None, eval_every=EVAL_EVERY,
          eval_n=EVAL_N, score_n=SCORE_N, write=True, verbose=True):
    """Every (d, seed) cell, written incrementally.

    The full sweep is about two hours of CPU, so each cell's row is flushed to
    ``logs/`` as it finishes; an interrupted run leaves the cells it did
    complete rather than nothing.
    """
    rows, traces = [], []
    for d in dims:
        for seed in seeds:
            row, history = run_cell(d, seed, budget=budget, eval_every=eval_every,
                                    eval_n=eval_n, score_n=score_n, verbose=verbose)
            rows.append(row)
            for step, loss, lr_, li, lb, err, se, ts in history:
                traces.append({"d": d, "seed": seed, "step": step,
                               "loss": f"{loss:.6e}", "loss_r": f"{lr_:.6e}",
                               "loss_ic": f"{li:.6e}", "loss_bc": f"{lb:.6e}",
                               "rel_l2": f"{err:.6e}", "stderr": f"{se:.6e}",
                               "train_seconds": f"{ts:.4f}"})
            if write:
                _write(rows, traces)
    return rows, traces


def _write(rows, traces):
    write_csv(SWEEP_CSV,
              ["d", "seed", "params", "rel_l2", "stderr", "rel_l2_final",
               "stderr_final", "best_step", "best_loss", "final_loss",
               "loss_r", "loss_ic", "loss_bc", "exact_rms",
               "train_seconds", "wall_seconds", "ms_per_step"], rows)
    write_csv(TRACE_CSV,
              ["d", "seed", "step", "loss", "loss_r", "loss_ic", "loss_bc",
               "rel_l2", "stderr", "train_seconds"], traces)


def summarize(rows):
    """Per-d mean, sd, min and max of the relative L2 across seeds.

    ``mean_stderr`` is the average Monte Carlo standard error of the individual
    scores, and sits next to ``sd`` for one reason: if the seed spread is not
    comfortably larger than the estimator's own noise, the spread is not a
    property of the training runs. In this sweep it is larger everywhere, but
    the comparison is printed rather than claimed.
    """
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = [r for r in rows if int(r["d"]) == d]
        errs = np.array([float(r["rel_l2"]) for r in cells])
        ses = np.array([float(r["stderr"]) for r in cells])
        finals = np.array([float(r["rel_l2_final"]) for r in cells])
        secs = np.array([float(r["train_seconds"]) for r in cells])
        # Averaged from the rows rather than recomputed from BUDGET, so a
        # summary of a run at some other budget is not silently mislabelled.
        per_step = np.array([float(r["ms_per_step"]) for r in cells])
        out.append({
            "d": d,
            "n_seeds": len(cells),
            "params": cells[0]["params"],
            "mean": f"{errs.mean():.6e}",
            "sd": f"{errs.std(ddof=1):.6e}" if len(errs) > 1 else "",
            "min": f"{errs.min():.6e}",
            "max": f"{errs.max():.6e}",
            "spread": f"{errs.max() / errs.min():.3f}",
            "mean_stderr": f"{ses.mean():.6e}",
            "mean_final": f"{finals.mean():.6e}",
            "mean_train_seconds": f"{secs.mean():.2f}",
            "ms_per_step": f"{per_step.mean():.3f}",
        })
    return out


def loss_scales(problem):
    """The two energies that make a training loss comparable across d.

    Raw losses are not comparable between dimensions, because the target itself
    shrinks: ``exact_rms`` falls like ``2^(-d/2)``, so at d = 16 a network can
    post the second-smallest ``loss_ic`` in the whole sweep while fitting the
    initial condition worse than zero would. Each loss therefore gets divided by
    the energy of the thing it is trying to match:

    - **IC energy**, the mean square of ``u(x, 0)`` over the cube. Orthogonality
      again (``<phi_k, phi_k> = 2^-d``, cross terms vanish), so it is
      ``2^-d sum_m a_m^2`` exactly.
    - **Residual scale**, the space-time mean square of ``u_t`` for the *exact*
      solution: ``u_t = -sum_m a_m r_m phi_m e^{-r_m t}``, so it is
      ``2^-d sum_m a_m^2 r_m^2 <e^{-2 r_m t}>_t``. The residual ``u_t - alpha
      Lap u`` is a difference of two terms of this size that cancel exactly on
      the truth, so it is the scale against which a nonzero residual is small
      or large.

    The square roots of the two ratios are then relative errors, directly
    comparable across d and to the relative L2 itself.
    """
    t0, t1 = problem.t_range
    r = problem.rates
    time_avg = (np.exp(-2 * r * t0) - np.exp(-2 * r * t1)) / (2 * r * (t1 - t0))
    vol = 2.0 ** (-problem.d)
    ic_energy = float(vol * np.sum(problem.amps ** 2))
    residual_scale = float(vol * np.sum(problem.amps ** 2 * r ** 2 * time_avg))
    return ic_energy, residual_scale


def normalized_losses(rows):
    """Per-d relative IC error and relative residual, from the sweep rows.

    ``sqrt(loss / scale)`` for each of the two, averaged over seeds. This is
    what Sec. 11's second table reports, and the reason it exists is that the
    unnormalized version of that table says the d = 16 run went well.
    """
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = [r for r in rows if int(r["d"]) == d]
        ic_energy, residual_scale = loss_scales(HighDHeat(d))
        li = np.mean([float(r["loss_ic"]) for r in cells])
        lr = np.mean([float(r["loss_r"]) for r in cells])
        out.append({
            "d": d,
            "exact_rms": f"{exact_rms(HighDHeat(d)):.6e}",
            "rel_ic_error": f"{np.sqrt(li / ic_energy):.6f}",
            "rel_residual": f"{np.sqrt(lr / residual_scale):.6f}",
            "loss_ic": f"{li:.6e}", "loss_r": f"{lr:.6e}",
        })
    return out


def report(rows, budget=None):
    budget = dict(BUDGET if budget is None else budget)
    summary = summarize(rows)
    print("\n" + "=" * 78)
    print(f"PINN vs d at a fixed budget "
          f"({budget['steps']} Adam steps, {budget['n_interior']} interior points, "
          f"width {budget['width']} x depth {budget['depth']})")
    print("=" * 78)
    print(f"{'d':>3} {'params':>7} {'mean relL2':>12} {'sd':>11} "
          f"{'min':>11} {'max':>11} {'MC se':>10} {'s/run':>8} {'ms/step':>8}")
    for s in summary:
        sd = float(s["sd"]) if s["sd"] else float("nan")
        print(f"{s['d']:>3} {int(s['params']):>7} {float(s['mean']):>12.4e} "
              f"{sd:>11.2e} {float(s['min']):>11.2e} {float(s['max']):>11.2e} "
              f"{float(s['mean_stderr']):>10.1e} "
              f"{float(s['mean_train_seconds']):>8.0f} "
              f"{float(s['ms_per_step']):>8.1f}")
    first, last = summary[0], summary[-1]
    print(f"\n  error   d={first['d']} -> d={last['d']}: "
          f"{float(first['mean']):.3e} -> {float(last['mean']):.3e} "
          f"({float(last['mean']) / float(first['mean']):.1f}x)")
    print(f"  cost    d={first['d']} -> d={last['d']}: "
          f"{float(first['ms_per_step']):.1f} -> {float(last['ms_per_step']):.1f} "
          f"ms/step ({float(last['ms_per_step']) / float(first['ms_per_step']):.1f}x, "
          f"against {last['d'] / first['d']:.0f}x in d)")

    print("\nTraining losses, divided by the energy of what they are matching "
          "(see loss_scales):")
    print(f"{'d':>3} {'exact rms':>11} {'rel IC error':>13} {'rel residual':>13}")
    for r in normalized_losses(rows):
        print(f"{r['d']:>3} {float(r['exact_rms']):>11.4e} "
              f"{float(r['rel_ic_error']):>13.4f} {float(r['rel_residual']):>13.4f}")
    return summary


# ---------------------------------------------------------------------------
# Cost per step: the claim that the network's cost is linear in d
# ---------------------------------------------------------------------------
def step_cost(d, budget=None, steps=30, warmup=True):
    """Seconds per optimizer step at dimension ``d``, with evaluation disabled.

    The residual differentiates the network once for ``u_t`` and once per
    spatial axis for the Laplacian (``pinn.derivatives.laplacian`` shares the
    first gradient pass, so it is d + 1 backward passes and not 2d), and every
    weight matrix except the first is independent of d. The prediction is
    therefore ``a + b d`` with a real intercept, not a proportionality -- so it
    is fitted with two parameters and both are reported.
    """
    budget = dict(BUDGET if budget is None else budget)
    problem = HighDHeat(d)
    small = dict(budget, steps=steps)
    if warmup:
        train(problem, seed=0, eval_every=10 ** 9, eval_n=100,
              **dict(small, steps=3))
    _, _, best = train(problem, seed=0, eval_every=10 ** 9, eval_n=100, **small)
    return best["train_seconds"] / (steps + 1)


def cost_study(dims=SWEEP_DIMS, steps=30, repeats=3):
    """``ms/step`` vs d, repeated, with a least-squares ``a + b d`` fit.

    Repeated because a single run of this does not reproduce. Three passes on
    one idle machine gave 230.9, 284.5 and 213.0 ms/step at d = 16 -- a 34%
    spread, which moved the fitted slope from 12.4 to 17.0 and the intercept
    from 2.2 to 15.4. That is the same lesson ``gp-from-scratch``'s Day 7 wrote
    down about peak RSS: wall clock is a property of the machine's state, not
    of the algorithm. So the median of ``repeats`` passes is what is reported
    and fitted, the min and max travel with it, and the claim the section
    actually makes is the *shape* -- linear in d -- which every pass agrees on
    and which ``tests/test_highd_pinn.py`` pins structurally, as a count of
    backward passes rather than as a time.
    """
    rows = []
    for d in dims:
        samples = sorted(1000 * step_cost(d, steps=steps) for _ in range(repeats))
        med = float(np.median(samples))
        rows.append({"d": d, "ms_per_step": f"{med:.4f}",
                     "ms_min": f"{samples[0]:.4f}", "ms_max": f"{samples[-1]:.4f}",
                     "repeats": repeats})
        print(f"  d={d:3d}  {med:8.2f} ms/step  "
              f"[{samples[0]:.2f}, {samples[-1]:.2f}] over {repeats} passes",
              flush=True)
    ds = np.array([r["d"] for r in rows], dtype=float)
    ms = np.array([float(r["ms_per_step"]) for r in rows])
    b, a = np.polyfit(ds, ms, 1)
    print(f"\n  least squares on the medians: {a:.2f} + {b:.2f} d ms/step "
          f"(intercept is {100 * a / ms[-1]:.0f}% of the d={int(ds[-1])} cost)")
    for r in rows:
        r["fit_ms_per_step"] = f"{a + b * r['d']:.4f}"
    return rows, (a, b)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(sweep_rows=None, trace_rows=None):
    """Three panels, all replayed from the committed CSVs.

    Left: the error trajectory of every run, so the trend in d is visible as
    whole curves rather than as five end points. Middle: the final scores, one
    marker per seed plus the mean, with the Monte Carlo error bars drawn --
    which is the panel that shows the seed spread dominating the metric noise.
    Right: cost per step against d, with the linear fit.
    """
    import matplotlib.pyplot as plt

    sweep_rows = sweep_rows if sweep_rows is not None else read_csv(SWEEP_CSV)
    trace_rows = trace_rows if trace_rows is not None else read_csv(TRACE_CSV)
    cost_rows = read_csv(COST_CSV)
    summary = summarize(sweep_rows)
    dims = [s["d"] for s in summary]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(dims)))
    cmap = dict(zip(dims, colors))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    for d in dims:
        for seed in sorted({int(r["seed"]) for r in trace_rows}):
            pts = [(int(r["step"]), float(r["rel_l2"])) for r in trace_rows
                   if int(r["d"]) == d and int(r["seed"]) == seed]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cmap[d],
                    lw=1.0, alpha=0.75,
                    label=f"d = {d}" if seed == 0 else None)
    ax.axhline(1.0, color="0.35", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("relative $L^2$")
    ax.set_title("training trajectories (3 seeds per $d$)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for s in summary:
        d = s["d"]
        cells = [r for r in sweep_rows if int(r["d"]) == d]
        for r in cells:
            ax.errorbar(d, float(r["rel_l2"]), yerr=float(r["stderr"]), fmt="o",
                        ms=4, color=cmap[d], alpha=0.55, capsize=2)
        ax.plot([d], [float(s["mean"])], "_", ms=18, mew=2, color=cmap[d])
    ax.plot(dims, [float(s["mean"]) for s in summary], "-", color="0.5", lw=1,
            zorder=0)
    # The scale that makes the top of this panel readable: a network that
    # outputs 0 everywhere scores exactly 1.0, which tests/test_highd_heat.py
    # asserts. Above the line, the solve is worse than saying nothing.
    ax.axhline(1.0, color="0.35", ls=":", lw=1)
    ax.annotate(r"$u_\theta \equiv 0$", xy=(1.05, 1.06), fontsize=7,
                color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("relative $L^2$")
    ax.set_title("final error: seeds (dots), mean (bar)")

    ax = axes[2]
    cds = np.array([float(r["d"]) for r in cost_rows])
    cms = np.array([float(r["ms_per_step"]) for r in cost_rows])
    lo = np.array([float(r["ms_min"]) for r in cost_rows])
    hi = np.array([float(r["ms_max"]) for r in cost_rows])
    b, a = np.polyfit(cds, cms, 1)
    grid = np.linspace(0, max(cds) * 1.05, 50)
    ax.plot(grid, a + b * grid, "-", color="0.6", lw=1,
            label=f"{a:.1f} + {b:.1f} $d$ (fit to medians)")
    ax.errorbar(cds, cms, yerr=[cms - lo, hi - cms], fmt="o", color="C0", ms=5,
                capsize=3, label=f"median of {cost_rows[0]['repeats']}, full range")
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("ms per Adam step")
    ax.set_title("cost per step is linear in $d$")
    ax.legend(fontsize=8)

    fig.suptitle("PINN across dimension at one fixed budget "
                 f"({BUDGET['steps']} steps, {BUDGET['n_interior']} collocation "
                 f"points, width {BUDGET['width']})", y=1.02, fontsize=10)
    savefig(fig, "highd_pinn.png")


def figures_from_committed():
    """Replay ``figures/highd_pinn.png`` from the committed CSVs. No training.

    The entry point ``experiments/reproduce_figures.py`` calls; the sweep it
    replays is two hours of CPU, so the logs ship and this turns them back into
    the figure.
    """
    make_figure()


# ---------------------------------------------------------------------------
def main(quick=False, do_sweep=False, cost=False, figure=False):
    if quick:
        budget = dict(BUDGET, steps=200, n_interior=500, width=32)
        rows, traces = sweep(dims=(1, 2), seeds=(0, 1), budget=budget,
                             eval_every=100, eval_n=20_000, score_n=50_000,
                             write=False)
        report(rows, budget)
        return

    if cost:
        rows, _ = cost_study()
        write_csv(COST_CSV, ["d", "ms_per_step", "ms_min", "ms_max", "repeats",
                             "fit_ms_per_step"], rows)
        return

    if do_sweep:
        t0 = time.time()
        rows, traces = sweep()
        print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")
        report(rows)
        cost_rows, _ = cost_study()
        write_csv(COST_CSV, ["d", "ms_per_step", "ms_min", "ms_max", "repeats",
                             "fit_ms_per_step"], cost_rows)
        make_figure(rows, traces)
        return

    if figure:
        make_figure()
        return

    print(__doc__.strip().splitlines()[0])
    print("\nPass one of --sweep (~2 h), --cost, --figures, --quick.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="the full d-sweep")
    ap.add_argument("--cost", action="store_true", help="ms/step vs d, no sweep")
    ap.add_argument("--figures", action="store_true", help="replay from CSVs")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()
    main(quick=args.quick, do_sweep=args.sweep, cost=args.cost,
         figure=args.figures)
