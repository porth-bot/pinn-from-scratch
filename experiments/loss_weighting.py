"""Loss weighting: what the residual/IC/BC balance actually decides.

``pinn/losses.py`` writes the objective as

    L = L_residual + w_ic L_ic + w_bc L_bc

and says, honestly, that balancing the terms is "a genuine PINN difficulty"
with the weights left as explicit knobs. Every experiment in the repo has run
at ``w_ic = w_bc = 1`` without ever measuring what that choice costs. This does.

The heat equation is the right place for it because the failure mode the
weights are supposed to guard against has a closed form. ``u == 0`` satisfies
``u_t = alpha u_xx`` exactly and satisfies the homogeneous Dirichlet walls
exactly; the *only* term that rules it out is the initial condition
(``tests/test_loss_weighting.py`` checks all three of those directly, on a real
network with its output layer zeroed rather than on an abstract zero function).
That makes a structural prediction, and the prediction is the thing under test:

- ``w_ic`` too small and the minimum the optimizer can most easily reach is the
  trivial field. The PINN converges to a perfect solution of the wrong problem,
  with a residual loss near zero -- the diagnostic a practitioner would most
  likely be watching says everything is fine.
- ``w_bc`` too small should cost very little here, because the trivial field
  the residual prefers already satisfies the walls, and so does the true
  solution.

Five measurements. The first two sweep, the third hunts the predicted failure
past the end of the sweep, the fourth asks why, and the fifth scores the
standard fix:

1. **The symmetric sweep**: ``w_ic = w_bc = w`` over four decades, three seeds.
   Relative L2 against the exact Fourier solution, plus the amplitude ratio
   ``||u_pinn|| / ||u_exact||`` -- the number that distinguishes "inaccurate"
   from "collapsed to zero", which the relative L2 alone does not.
2. **The asymmetry**: ``w_ic`` and ``w_bc`` swept separately, holding the other
   at 1, three seeds each. Three seeds because the symmetric sweep's own
   seed-to-seed spread turns out to be several-fold, which sets the resolution
   limit on this comparison -- a single-seed version of this arm could not tell
   a weight effect from a seed.
3. **Starving one constraint at a time**: each weight taken down four further
   decades to 0 exactly, the other held at 1. Measurement 1 finds no collapse
   anywhere in its range, and "we did not see it" is only worth something if
   the search went as far as the failure could possibly hide -- at a weight of
   0 the objective does not mention that constraint at all. Both arms are run
   because the prediction is a *contrast*: showing w_ic collapse without
   showing w_bc survive would not establish the asymmetry it claims.
4. **Why**: the gradient norms ``||grad_theta L_r||``, ``||grad_theta L_ic||``
   and ``||grad_theta L_bc||`` through training. The standard diagnosis (Wang,
   Teng & Perdikaris 2021) is that the residual gradient dominates the
   constraint gradients, so equal weights are not equal treatment. That is a
   claim about this problem's numbers, and it is checkable.
5. **The adaptive rule, scored against the sweep it is trying to guess.**
   Learning-rate annealing sets ``w_i`` from the gradient-norm ratio at every
   update. The honest test is not "does it train" but "does it land where the
   sweep says the optimum is", and it costs a sweep to find out.

The repo already has the other answer to this problem: ``experiments/hard_bc.py``
builds the IC and BCs into the ansatz so they hold by construction and carry no
weight at all. This experiment is the measurement of what soft constraints cost
when that trick is unavailable.

Run:  python experiments/loss_weighting.py             # ~50 min, 60 solves
      python experiments/loss_weighting.py --figures   # replay from the CSVs
      python experiments/loss_weighting.py --quick     # 2 short solves
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import plt, read_csv, savefig, write_csv
from heat import (
    T_RANGE,
    X_RANGE,
    _eval_grid,
    heat_exact,
    heat_residual,
    initial_condition,
    model_config,
    predict,
    rel_l2_error,
)
from pinn.losses import boundary_points, initial_points, interior_points
from pinn.model import MLP, set_seed

STEPS = 3000
N_INTERIOR = 4000
WIDTH = 128
GRAD_EVERY = 250
SEEDS = (0, 1, 2)
SYMMETRIC_WEIGHTS = (0.01, 0.1, 1.0, 10.0, 100.0)
ASYMMETRIC_WEIGHTS = (0.1, 10.0, 100.0)      # the other weight held at 1.0
# One constraint weight driven toward the limit where the trivial field stops
# being ruled out, the other held at 1. Four decades below the symmetric
# sweep's floor, ending at 0 exactly -- at which point nothing in the objective
# mentions that constraint at all. Both constraints get the same treatment,
# because the prediction under test is about the *difference* between them: a
# one-armed version could show w_ic collapsing without establishing that w_bc
# does not.
STARVE_WEIGHTS = (1e-2, 1e-4, 1e-6, 0.0)
ADAPT_EVERY = 100
ADAPT_ALPHA = 0.9


def amplitude_ratio(model, nx=101, nt=101):
    """||u_pinn|| / ||u_exact|| on the evaluation grid.

    Relative L2 saturates near 1.0 for *any* badly wrong field, so it cannot
    tell "the PINN learned the wrong shape" from "the PINN learned nothing".
    This can: a collapsed run reads ~0, and a run that merely mis-fits reads
    ~1 with a large relative L2.
    """
    _, _, XX, TT = _eval_grid(nx, nt)
    return float(np.linalg.norm(predict(model, XX, TT))
                 / np.linalg.norm(heat_exact(XX, TT)))


def _flat_grad(loss, params):
    """grad_theta loss as one flat vector, without disturbing the training graph.

    ``retain_graph=True`` because the term-wise backward passes and the real
    training step all differentiate the same forward pass.

    ``allow_unused=True`` is not defensive boilerplate here, and finding out
    why cost a crash: **the residual loss does not depend on the output
    layer's bias at all**. L_r sees only u_t and u_xx, and adding a constant
    to u changes neither -- a constant is an exact solution of
    ``u_t = alpha u_xx``, so the offset lives in the residual's null space and
    the parameter that implements it gets a literally absent gradient. That is
    this experiment's whole thesis, visible in the autograd graph before any
    training has happened: the physics term cannot pin down the solution by
    itself, and something else has to. Missing gradients are read as zero,
    which is what they mathematically are.
    """
    grads = torch.autograd.grad(loss, params, retain_graph=True,
                                allow_unused=True)
    return torch.cat([torch.zeros_like(p).reshape(-1) if g is None
                      else g.reshape(-1) for g, p in zip(grads, params)])


def _grad_norm(loss, params):
    """``||grad_theta loss||_2``."""
    return float(_flat_grad(loss, params).norm())


def _grad_stats(loss, params):
    """(max |grad|, mean |grad|) over all parameters -- what the adaptive rule
    reads. Kept separate from ``_grad_norm`` because the rule is defined on
    element-wise magnitudes, not on the L2 norm of the whole gradient, and
    conflating them would change what is being implemented."""
    flat = _flat_grad(loss, params).abs()
    return float(flat.max()), float(flat.mean())


def train_weighted(w_ic=1.0, w_bc=1.0, *, adaptive=False, seed=0,
                   steps=STEPS, n_interior=N_INTERIOR, width=WIDTH,
                   lr=1e-3, n_ic=400, n_bc=200, grad_every=GRAD_EVERY,
                   verbose=False):
    """Train the heat PINN at a given weighting; return (model, trace).

    Deliberately a copy of ``heat.train``'s loop rather than a call into it:
    this one records per-term gradient norms and can rewrite the weights while
    it runs, and threading both through the shipped trainer would complicate
    the code every other experiment depends on. ``tests/test_loss_weighting.py``
    pins the two together -- at ``w_ic = w_bc = 1``, same seed, same budget,
    the two trainers must produce the same field to the last bit, which is what
    makes this copy safe.

    ``adaptive=True`` implements learning-rate annealing (Wang, Teng &
    Perdikaris 2021): every ``ADAPT_EVERY`` steps, set

        w_hat_i = max_theta |grad_theta L_r| / mean_theta |grad_theta L_i|
        w_i    <- (1 - alpha) w_i + alpha w_hat_i

    with alpha = 0.9. The rule's motivation is measurement 3: if the residual
    gradient's largest component dwarfs a constraint gradient's typical
    component, the constraint is not being optimized at all, and this ratio is
    exactly the factor that fixes that. The denominator uses the *unweighted*
    term, so ``w_i`` is the whole scale factor rather than a correction to
    itself.

    The trace rows are (step, loss, loss_r, loss_ic, loss_bc, rel_l2, w_ic,
    w_bc, gnorm_r, gnorm_ic, gnorm_bc); the gradient norms are NaN on steps
    where they were not measured.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    interior = interior_points(n_interior, X_RANGE, T_RANGE, gen)
    ic = initial_points(n_ic, X_RANGE, T_RANGE[0], gen)
    ic_target = initial_condition(ic[:, 0:1])
    left, right = boundary_points(n_bc, X_RANGE, T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    bc_target = torch.zeros(bc.shape[0], 1)

    model = MLP(**model_config(width, 4))
    params = list(model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    trace = []
    for step in range(steps + 1):
        opt.zero_grad()
        u_int = model(interior)
        loss_r = torch.mean(heat_residual(u_int, interior) ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)

        if adaptive and step % ADAPT_EVERY == 0:
            max_r, _ = _grad_stats(loss_r, params)
            for name, term in (("ic", loss_ic), ("bc", loss_bc)):
                _, mean_i = _grad_stats(term, params)
                what = max_r / max(mean_i, 1e-30)
                if name == "ic":
                    w_ic = (1 - ADAPT_ALPHA) * w_ic + ADAPT_ALPHA * what
                else:
                    w_bc = (1 - ADAPT_ALPHA) * w_bc + ADAPT_ALPHA * what

        logging = grad_every and (step % grad_every == 0 or step == steps)
        if logging:
            gr = _grad_norm(loss_r, params)
            gic = _grad_norm(loss_ic, params)
            gbc = _grad_norm(loss_bc, params)
        else:
            gr = gic = gbc = float("nan")

        loss = loss_r + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        opt.step()

        if logging:
            err = rel_l2_error(model)
            trace.append((step, loss.item(), loss_r.item(), loss_ic.item(),
                          loss_bc.item(), err, float(w_ic), float(w_bc),
                          gr, gic, gbc))
            if verbose:
                print(f"    step {step:5d}  relL2 {err:.4f}  "
                      f"r {loss_r:.2e} ic {loss_ic:.2e} bc {loss_bc:.2e}  "
                      f"|g_r| {gr:.2e} |g_ic| {gic:.2e}  "
                      f"w_ic {w_ic:.3g} w_bc {w_bc:.3g}")
    return model, trace


# ---------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------
def run_arm(label, *, w_ic, w_bc, adaptive, seed, steps, verbose=False):
    t0 = time.time()
    model, trace = train_weighted(w_ic=w_ic, w_bc=w_bc, adaptive=adaptive,
                                  seed=seed, steps=steps, verbose=verbose)
    last = trace[-1]
    row = {
        "arm": label, "w_ic_set": w_ic, "w_bc_set": w_bc,
        "adaptive": int(adaptive), "seed": seed, "steps": steps,
        "rel_l2": f"{last[5]:.6e}",
        "amplitude_ratio": f"{amplitude_ratio(model):.6e}",
        "loss_r": f"{last[2]:.6e}", "loss_ic": f"{last[3]:.6e}",
        "loss_bc": f"{last[4]:.6e}",
        "w_ic_final": f"{last[6]:.6e}", "w_bc_final": f"{last[7]:.6e}",
        "seconds": f"{time.time() - t0:.1f}",
    }
    print(f"  {label:22s} seed {seed}  relL2 {last[5]:.4f}  "
          f"amp {float(row['amplitude_ratio']):.3f}  "
          f"L_r {last[2]:.2e}  ({row['seconds']}s)")
    return row, trace


ROW_FIELDS = ["arm", "w_ic_set", "w_bc_set", "adaptive", "seed", "steps",
              "rel_l2", "amplitude_ratio", "loss_r", "loss_ic", "loss_bc",
              "w_ic_final", "w_bc_final", "seconds"]
TRACE_FIELDS = ["arm", "seed", "step", "loss", "loss_r", "loss_ic", "loss_bc",
                "rel_l2", "w_ic", "w_bc", "gnorm_r", "gnorm_ic", "gnorm_bc"]


def trace_rows(label, seed, trace):
    return [dict(zip(TRACE_FIELDS,
                     [label, seed] + [f"{v:.6e}" if i else int(v)
                                      for i, v in enumerate(t)]))
            for t in trace]


def main(quick=False, figures=False):
    if figures:
        figures_from_committed()
        return

    steps = 300 if quick else STEPS
    seeds = (0,) if quick else SEEDS
    rows, traces = [], []

    print("=" * 70)
    print(f"1. symmetric sweep  w_ic = w_bc = w   ({len(seeds)} seeds, "
          f"{steps} steps)")
    print("=" * 70)
    for w in ((1.0, 100.0) if quick else SYMMETRIC_WEIGHTS):
        for seed in seeds:
            row, tr = run_arm(f"sym_w={w:g}", w_ic=w, w_bc=w, adaptive=False,
                              seed=seed, steps=steps)
            rows.append(row)
            traces += trace_rows(row["arm"], seed, tr)

    if not quick:
        print("\n" + "=" * 70)
        print(f"2. asymmetric: one weight swept, the other pinned at 1 "
              f"({len(seeds)} seeds)")
        print("=" * 70)
        for w in ASYMMETRIC_WEIGHTS:
            for label, w_ic, w_bc in ((f"ic_only_w={w:g}", w, 1.0),
                                      (f"bc_only_w={w:g}", 1.0, w)):
                for seed in seeds:
                    row, tr = run_arm(label, w_ic=w_ic, w_bc=w_bc,
                                      adaptive=False, seed=seed, steps=steps)
                    rows.append(row)
                    traces += trace_rows(row["arm"], seed, tr)

        print("\n" + "=" * 70)
        print(f"3. starving one constraint at a time, down to 0 "
              f"({len(seeds)} seeds)")
        print("=" * 70)
        for w in STARVE_WEIGHTS:
            for label, w_ic, w_bc in ((f"ic_starve_w={w:g}", w, 1.0),
                                      (f"bc_starve_w={w:g}", 1.0, w)):
                for seed in seeds:
                    row, tr = run_arm(label, w_ic=w_ic, w_bc=w_bc,
                                      adaptive=False, seed=seed, steps=steps)
                    rows.append(row)
                    traces += trace_rows(row["arm"], seed, tr)

        print("\n" + "=" * 70)
        print("5. adaptive weights (learning-rate annealing)")
        print("   -- measurement 4 is read off the w=1 traces and costs no solves")
        print("=" * 70)
        for seed in seeds:
            row, tr = run_arm("adaptive", w_ic=1.0, w_bc=1.0, adaptive=True,
                              seed=seed, steps=steps)
            rows.append(row)
            traces += trace_rows(row["arm"], seed, tr)

    write_csv("loss_weighting.csv", ROW_FIELDS, rows)
    write_csv("loss_weighting_trace.csv", TRACE_FIELDS, traces)
    if not quick:
        # Read the CSVs back rather than reporting from the in-memory rows, so
        # the training path and the --figures replay path run the *same* code
        # over the *same* bytes. They had already diverged once: the in-memory
        # rows carry an int seed and the CSV a string, and a `seed == "0"`
        # filter silently emptied the gradient-norm panel on this path only.
        figures_from_committed()


# ---------------------------------------------------------------------------
# Reporting + figure, both from the CSV rows (so --figures replays exactly)
# ---------------------------------------------------------------------------
def _f(rows, key):
    return np.array([float(r[key]) for r in rows])


def _by_arm(rows, prefix):
    """Rows whose arm starts with ``prefix``, grouped by the weight in the name."""
    out = {}
    for r in rows:
        if r["arm"].startswith(prefix):
            out.setdefault(float(r["arm"].split("=")[1]), []).append(r)
    return dict(sorted(out.items()))


def _seed_band(rows_):
    """(median, min, max) of the relative L2 over an arm's seeds."""
    e = _f(rows_, "rel_l2")
    return float(np.median(e)), float(e.min()), float(e.max())


def report(rows, traces):
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    sym = _by_arm(rows, "sym_w=")
    print("\n1. symmetric sweep (median over seeds; amp = ||u_pinn||/||u_exact||,"
          "\n   so amp ~ 0 is a collapse to the trivial solution and amp ~ 1 with"
          "\n   a large relative L2 is a fit that is merely wrong)")
    print("        w |  rel L2 (median, [min, max])   |    amp |     L_r")
    for w, rs in sym.items():
        med, lo, hi = _seed_band(rs)
        print(f"   {w:6g} | {med:8.4f}  [{lo:.4f}, {hi:.4f}] |"
              f" {np.median(_f(rs, 'amplitude_ratio')):6.3f} |"
              f" {np.median(_f(rs, 'loss_r')):.2e}")
    best_w = min(sym, key=lambda w: np.median(_f(sym[w], "rel_l2")))
    worst_seed_spread = max(hi / lo for _, lo, hi in map(_seed_band, sym.values()))
    print(f"   best symmetric weight: w = {best_w:g}")
    print(f"   widest single-arm seed spread: {worst_seed_spread:.1f}x  "
          "-- the resolution limit on everything below")

    print("\n2. asymmetric: one weight swept, the other pinned at 1."
          "\n   The structural prediction was that w_ic matters and w_bc does not,"
          "\n   because u == 0 already satisfies the walls but not the IC. The"
          "\n   test is whether either sweep moves the error by more than seeds do.")
    # Header columns come from the rows, not from ASYMMETRIC_WEIGHTS, so the
    # replay path cannot mislabel a CSV that was written at other weights.
    swept = sorted(_by_arm(rows, "ic_only_w="))
    print("           swept |" + "".join(f"{w:>10g}" for w in swept)
          + "   | median over the sweep")
    ranges = {}
    for prefix, name in (("ic_only_w=", "w_ic (w_bc=1)"),
                         ("bc_only_w=", "w_bc (w_ic=1)")):
        group = _by_arm(rows, prefix)
        meds = [np.median(_f(rs, "rel_l2")) for rs in group.values()]
        print(f"   {name:>13s} |" + "".join(f"{m:10.4f}" for m in meds)
              + f"   | {max(meds) / max(min(meds), 1e-30):.1f}x across the sweep")
        ranges[name] = max(meds) / max(min(meds), 1e-30)
    base_med, base_lo, base_hi = _seed_band([r for r in rows
                                             if r["arm"] == "sym_w=1"])
    print(f"   {'both = 1':>13s} |{base_med:10.4f}"
          f"   (seed band [{base_lo:.4f}, {base_hi:.4f}])")
    verdict = ("larger" if ranges["w_ic (w_bc=1)"] > ranges["w_bc (w_ic=1)"]
               else "no larger")
    print(f"   -> the w_ic sweep moves the error {verdict} than the w_bc sweep "
          f"({ranges['w_ic (w_bc=1)']:.1f}x vs {ranges['w_bc (w_ic=1)']:.1f}x), "
          f"\n      against a {worst_seed_spread:.1f}x seed spread")

    print("\n3. starving one constraint at a time, the other held at 1."
          "\n   The predicted failure is a collapse onto u == 0, which solves the"
          "\n   PDE and the walls exactly but not the IC. amp is the number that"
          "\n   shows it; the prediction is that only the w_ic arm collapses.")
    for prefix, starved, field in (("ic_starve_w=", "w_ic", "loss_ic"),
                                   ("bc_starve_w=", "w_bc", "loss_bc")):
        starve = _by_arm(rows, prefix)
        if not starve:
            continue
        print(f"\n   {starved} -> 0:")
        print(f"   {starved:>6s} |  rel L2 (median, [min, max])   |    amp |"
              f" {field} (median)")
        for w, rs in starve.items():
            med, lo, hi = _seed_band(rs)
            print(f"   {w:6g} | {med:8.4f}  [{lo:.4f}, {hi:.4f}] |"
                  f" {np.median(_f(rs, 'amplitude_ratio')):6.3f} |"
                  f" {np.median(_f(rs, field)):.2e}")
        amps = {w: float(np.median(_f(rs, "amplitude_ratio")))
                for w, rs in starve.items()}
        collapsed = sorted(w for w, a in amps.items() if a < 0.1)
        if collapsed:
            survived = sorted(w for w, a in amps.items() if a >= 0.1)
            edge = (f", intact at every {starved} >= {min(survived):g}"
                    if survived else "")
            print(f"   -> collapse (amp < 0.1) at {starved} <= "
                  f"{max(collapsed):g}{edge}")
        else:
            print(f"   -> no collapse at any {starved} tried, including "
                  f"{starved} = 0 exactly")

    print("\n4. gradient norms at w_ic = w_bc = 1 (seed 0), through training."
          "\n   The premise of every gradient-balancing rule is that |g_r| dwarfs"
          "\n   the constraint gradients, so a ratio below 1 refutes the premise.")
    tr = [t for t in traces if t["arm"] == "sym_w=1" and t["seed"] == "0"]
    print("     step |   |g_r|    |g_ic|    |g_bc|  |  ratio r/ic")
    ratios = []
    for t in tr:
        gr, gic = (float(t[k]) for k in ("gnorm_r", "gnorm_ic"))
        ratios.append(gr / max(gic, 1e-30))
    for t in tr[:: max(1, len(tr) // 6)]:
        gr, gic, gbc = (float(t[k]) for k in ("gnorm_r", "gnorm_ic", "gnorm_bc"))
        print(f"   {int(t['step']):6d} | {gr:.2e}  {gic:.2e}  {gbc:.2e}  |"
              f" {gr / max(gic, 1e-30):8.2f}")
    if ratios:
        frac = sum(r < 1.0 for r in ratios) / len(ratios)
        print(f"   -> |g_r| / |g_ic| in [{min(ratios):.2f}, {max(ratios):.2f}], "
              f"below 1 on {100 * frac:.0f}% of the logged steps")

    print("\n5. adaptive weights vs the sweep's own optimum")
    ad = [r for r in rows if r["arm"] == "adaptive"]
    if ad:
        med, lo, hi = _seed_band(ad)
        print(f"   adaptive: relL2 median {med:.4f} [{lo:.4f}, {hi:.4f}]   "
              f"amp {np.median(_f(ad, 'amplitude_ratio')):.3f}   "
              f"final w_ic {np.median(_f(ad, 'w_ic_final')):.3g}  "
              f"w_bc {np.median(_f(ad, 'w_bc_final')):.3g}")
        bmed, blo, bhi = _seed_band(sym[best_w])
        print(f"   best fixed (w={best_w:g}): relL2 median {bmed:.4f} "
              f"[{blo:.4f}, {bhi:.4f}]")
        verdict = "beats" if med < bmed else "does not beat"
        print(f"   -> the rule {verdict} the best fixed weight "
              f"({med / max(bmed, 1e-30):.1f}x its error), and picks weights "
              f"\n      {np.median(_f(ad, 'w_ic_final')) / max(best_w, 1e-30):.0f}x "
              "beyond the largest the sweep ever tried")


def _band(ax, group, color, name):
    """Median relative L2 with a min-max seed band, for one swept arm family."""
    ws = np.array(list(group))
    med = np.array([np.median(_f(rs, "rel_l2")) for rs in group.values()])
    lo = np.array([_f(rs, "rel_l2").min() for rs in group.values()])
    hi = np.array([_f(rs, "rel_l2").max() for rs in group.values()])
    ax.plot(ws, med, "o-", color=color, lw=1.6, ms=4, label=name)
    ax.fill_between(ws, lo, hi, color=color, alpha=0.15)
    return ws, med, lo, hi


def figure(rows, traces):
    sym = _by_arm(rows, "sym_w=")
    best_w = min(sym, key=lambda w: np.median(_f(sym[w], "rel_l2")))
    amp = np.array([np.median(_f(rs, "amplitude_ratio")) for rs in sym.values()])

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2), constrained_layout=True)

    # -- 1. the symmetric sweep ---------------------------------------------
    ax = axes[0, 0]
    ws, _, _, _ = _band(ax, sym, "C0", "relative $L_2$")
    ax.plot([], [], color="C0", alpha=0.3, lw=6,
            label=f"min-max over {len(SEEDS)} seeds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$w_{ic} = w_{bc} = w$")
    ax.set_ylabel("relative $L_2$ vs the exact solution")
    ax.set_title("1. The optimum in $w$ is a plateau, and $w=1$ is in it",
                 loc="left", fontsize=10)
    twin = ax.twinx()
    twin.plot(ws, amp, "s--", color="C3", lw=1.3, ms=3.5)
    twin.set_ylabel("$\\|u_{pinn}\\| / \\|u_{exact}\\|$", color="C3")
    twin.tick_params(axis="y", colors="C3")
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color("C3")
    twin.axhline(1.0, color="C3", lw=0.7, ls=":")
    ax.legend(fontsize=7.5, loc="upper center")

    # -- 2. the asymmetry, against the seed spread --------------------------
    ax = axes[0, 1]
    for prefix, color, name in (("ic_only_w=", "C0", "$w_{ic}$ swept, $w_{bc}=1$"),
                                ("bc_only_w=", "C1", "$w_{bc}$ swept, $w_{ic}=1$")):
        group = dict(_by_arm(rows, prefix))
        group[1.0] = sym[1.0]                     # the shared w = 1 anchor point
        _band(ax, dict(sorted(group.items())), color, name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("the swept weight (the other held at 1)")
    ax.set_ylabel("relative $L_2$")
    ax.set_title("2. Neither constraint's weight clears the seed band",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5)

    # -- 3. starving each constraint in turn --------------------------------
    # Plotted on the amplitude ratio, not the relative L2: a collapsed field
    # reads rel L2 ~ 1.0 and so does any other badly wrong field, so the error
    # metric cannot show what this panel is about. amp separates them.
    ax = axes[0, 2]
    groups = {p: _by_arm(rows, p) for p in ("ic_starve_w=", "bc_starve_w=")}
    allw = sorted({w for g in groups.values() for w in g})
    if allw:
        floor = min([w for w in allw if w > 0] + [1e-6]) / 100.0
        for prefix, color, name in (
                ("ic_starve_w=", "C0", "$w_{ic}\\to 0$ ($w_{bc}=1$)"),
                ("bc_starve_w=", "C1", "$w_{bc}\\to 0$ ($w_{ic}=1$)")):
            g = groups[prefix]
            if not g:
                continue
            xs = np.array([w if w > 0 else floor for w in g])
            a = np.array([np.median(_f(rs, "amplitude_ratio"))
                          for rs in g.values()])
            order = np.argsort(xs)
            ax.plot(xs[order], a[order], "o-", color=color, lw=1.6, ms=4,
                    label=name)
        ax.set_xscale("log")
        ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
        ax.axhline(0.1, color="C3", lw=1.0, ls="-.")
        ax.text(floor * 1.2, 0.14, "collapse threshold", fontsize=6.5, color="C3")
        ax.set_ylim(-0.05, 1.25)
        ticks = [floor] + [w for w in allw if w > 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [f"$10^{{{np.log10(t):.0f}}}$"
                                    for t in ticks[1:]])
        ax.legend(fontsize=7.5, loc="center left")
    ax.set_xlabel("the starved weight (leftmost point is 0 exactly)")
    ax.set_ylabel("$\\|u_{pinn}\\| / \\|u_{exact}\\|$")
    ax.set_title("3. Only the IC weight can collapse the solve",
                 loc="left", fontsize=10)

    # -- 4. the gradient norms ----------------------------------------------
    ax = axes[1, 0]
    tr = [t for t in traces if t["arm"] == "sym_w=1" and t["seed"] == "0"]
    steps = np.array([int(t["step"]) for t in tr])
    for key, color, name in (("gnorm_r", "C0", "$\\|\\nabla_\\theta L_r\\|$"),
                             ("gnorm_ic", "C2", "$\\|\\nabla_\\theta L_{ic}\\|$"),
                             ("gnorm_bc", "C1", "$\\|\\nabla_\\theta L_{bc}\\|$")):
        ax.semilogy(steps, [float(t[key]) for t in tr], "-", color=color,
                    lw=1.5, label=name)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("gradient norm")
    ax.set_title("4. The residual gradient does not dominate ($w=1$)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5)

    # -- 5. what the rule picks ---------------------------------------------
    ax = axes[1, 1]
    ad_tr = [t for t in traces if t["arm"] == "adaptive" and t["seed"] == "0"]
    if ad_tr:
        st = np.array([int(t["step"]) for t in ad_tr])
        ax.semilogy(st, [float(t["w_ic"]) for t in ad_tr], "-", color="C0",
                    lw=1.6, label="$w_{ic}$, chosen by the rule")
        ax.semilogy(st, [float(t["w_bc"]) for t in ad_tr], "-", color="C1",
                    lw=1.6, label="$w_{bc}$, chosen by the rule")
    ax.axhline(best_w, color="0.35", ls="--", lw=1.2,
               label=f"best fixed $w$ from panel 1 ($w={best_w:g}$)")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("weight")
    ax.set_title("5. What the adaptive rule picks, unsupervised",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5)

    # -- 6. and where that lands it -----------------------------------------
    ax = axes[1, 2]
    _band(ax, sym, "C0", "fixed $w$, swept")
    ad = [r for r in rows if r["arm"] == "adaptive"]
    if ad:
        ax.plot(_f(ad, "w_ic_final"), _f(ad, "rel_l2"), "*", color="C3", ms=13,
                ls="none", label="adaptive runs, at the $w_{ic}$ they chose")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$w_{ic}$ (swept, or self-selected)")
    ax.set_ylabel("relative $L_2$")
    ax.set_title("6. The rule lands off the end of the sweep it replaces",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    savefig(fig, "loss_weighting.png")


def figures_from_committed():
    """The figure and the printed report, from the committed CSVs -- no training."""
    rows = read_csv("loss_weighting.csv")
    traces = read_csv("loss_weighting_trace.csv")
    report(rows, traces)
    figure(rows, traces)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true",
                    help="replay the figure from the committed CSVs, no training")
    args = ap.parse_args()
    main(quick=args.quick, figures=args.figures)
