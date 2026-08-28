"""Does a different network fix the high-dimensional collapse? (Sec. 16)

Sec. 14 ended by naming the variable it could not rule out. Having shown that
the d-collapse survives replacing the physics-informed loss with supervised
regression onto the exact solution -- same points, same budget, labels handed
to the network -- it concluded that the failure belongs to the approximator and
the optimizer rather than to the residual formulation, and then said the thing
this section exists to test:

    every high-dimensional number in this repo uses one width, one depth and
    one activation, and Sec. 14's control says that is now the binding
    constraint.

So vary them. The measurement runs on Sec. 14's **regression** control rather
than on the PINN, deliberately: regression is the setting where the objective
is already out of the way, so what is left is exactly the question "can this
family of networks, trained this way, represent and find the target from 4000
exact labels?" It is also about 30x cheaper per cell than the residual loss
(no d + 2 reverse-mode passes per step), which is what makes a sweep affordable
at all on this machine.

Design: one factor at a time from Sec. 14's baseline
----------------------------------------------------
The baseline is ``BUDGET`` from Sec. 14 -- width 128, depth 4, tanh, 4000
uniform points, 2000 Adam steps at lr 1e-3 -- and every cell changes exactly one
of width, depth, activation. A full cross would be 18 configurations per
dimension and would answer a question nobody asked; the question here is
whether *any* single axis moves the number, and one-at-a-time answers that
while keeping the baseline shared across all three panels.

Two dimensions: d = 8, where Sec. 14 found the fit lands but the test error
stalls, and d = 16, where it found the network cannot fit even its own labels.
Those are different failures and an architecture change need not move both.

Three seeds per cell. Sec. 14 measured the seed spread *shrinking* with d
(every seed converges to the same failure), so three is enough to see whether a
config's effect is larger than its seeds -- and where it is not, this says so
rather than ranking means.

What is *not* controlled, and it matters
----------------------------------------
Width and depth change the parameter count, so "wider is better" and "more
parameters is better" are not separated by this design; ``params`` is logged on
every row so the reader can see the confound rather than take my word for its
size. Where the two axes disagree at matched parameter count, that is worth
more than either axis alone, and :func:`matched_params` pulls out the pairs
that are close enough to compare.

The metric is the repo's usual uniform-L2 relative error, where **a network
that outputs zero scores 1.000** -- the number to keep in view, because at
d = 16 the baseline is already past it.

Run:  python experiments/highd_arch.py --run              # the sweep, ~20 min
                        [--seconds N]                     # time-box and resume
      python experiments/highd_arch.py --table            # from the committed CSV
      python experiments/highd_arch.py --figures          # replay the figure
      python experiments/highd_arch.py --quick            # tiny smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import read_csv, savefig, write_csv
from highd_heat import HighDHeat
from highd_pinn import n_params
from highd_degrade import BUDGET, fit_cell

ARCH_CSV = "highd_arch.csv"

ARCH_FIELDS = ["axis", "d", "width", "depth", "activation", "seed", "params",
               "rel_l2", "stderr", "rel_fit_sampled", "final_mse",
               "train_seconds"]

#: The point every axis passes through: Sec. 14's regression control.
BASE = (BUDGET["width"], BUDGET["depth"], "tanh")

DIMS = (8, 16)
SEEDS = (0, 1, 2)

#: The activation axis also runs at low d, and it is a control rather than an
#: extension. A sine-activation network is *matched* to this problem -- the
#: exact solution is a sum of products of sines -- so a win for ``sin`` here is
#: at risk of being a fact about the target rather than about dimension. If the
#: advantage is a constant factor across d, it is the former; if it grows with
#: d, it is doing something to the collapse specifically. Running d = 1, 2, 4 is
#: what tells those apart, and it is cheap at low d.
EXTRA_DIMS = (1, 2, 4)

WIDTHS = (32, 128, 512)
DEPTHS = (2, 4, 8)
ACTIVATIONS = ("tanh", "sin")


def configs():
    """(axis, width, depth, activation) for every cell, baseline included once.

    The baseline belongs to all three axes and is stored under ``"base"`` so it
    is run once and read three times; :func:`axis_rows` is what re-attaches it.
    """
    w0, d0, a0 = BASE
    out = [("base", w0, d0, a0)]
    out += [("width", w, d0, a0) for w in WIDTHS if w != w0]
    out += [("depth", w0, d, a0) for d in DEPTHS if d != d0]
    out += [("activation", w0, d0, a) for a in ACTIVATIONS if a != a0]
    return out


def cell(problem, width, depth, activation, seed, axis, steps=None):
    """One regression fit at this architecture. Sec. 14's cell, re-parameterized."""
    row = fit_cell(problem, "uniform", seed=seed, width=width, depth=depth,
                   activation=activation, steps=steps)
    return {
        "axis": axis, "d": problem.d, "width": width, "depth": depth,
        "activation": activation, "seed": seed,
        "params": n_params(problem.d, width, depth),
        "rel_l2": row["rel_l2"], "stderr": row["stderr"],
        "rel_fit_sampled": row["rel_fit_sampled"],
        "final_mse": row["final_mse"],
        "train_seconds": row["train_seconds"],
    }


def sweep(dims=DIMS, seeds=SEEDS, seconds=None, steps=None, write=True,
          verbose=True, extra_dims=EXTRA_DIMS):
    """Run every (config, d, seed) cell not already in the committed log.

    Resumable and time-boxable for the same reason Sec. 13's sweep is: this
    host will not hold a foreground process for much more than ten minutes, and
    a backgrounded one gets almost no CPU. Resumption is exact because a cell
    is a pure function of (d, width, depth, activation, seed) -- the collocation
    draw is seeded and fixed, so a cell rerun reproduces itself.
    """
    try:
        rows = read_csv(ARCH_CSV)
    except FileNotFoundError:
        rows = []
    done = {(int(r["d"]), int(r["width"]), int(r["depth"]), r["activation"],
             int(r["seed"])) for r in rows}
    start = time.perf_counter()

    queue = [(d, cfg) for d in dims for cfg in configs()]
    # The activation control at low d: baseline and sin only, no size axis.
    queue += [(d, cfg) for d in extra_dims for cfg in configs()
              if cfg[0] in ("base", "activation")]

    problems = {}
    for d, (axis, width, depth, activation) in queue:
        problem = problems.setdefault(d, HighDHeat(d))
        for seed in seeds:
            key = (d, width, depth, activation, seed)
            if key in done:
                continue
            if seconds is not None and time.perf_counter() - start > seconds:
                if verbose:
                    print(f"  time box reached; {len(done)} cells done, "
                          "rerun to continue", flush=True)
                return rows
            row = cell(problem, width, depth, activation, seed, axis, steps=steps)
            rows.append(row)
            done.add(key)
            if verbose:
                print(f"  d={d:2d} w={width:<3d} depth={depth} "
                      f"{activation:4s} seed={seed}  "
                      f"rel L2 {float(row['rel_l2']):.4e}  "
                      f"own-sample {float(row['rel_fit_sampled']):.4e}  "
                      f"{float(row['train_seconds']):.0f}s", flush=True)
            if write:
                write_csv(ARCH_CSV, ARCH_FIELDS, rows)
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _f(rows, key):
    return np.array([float(r[key]) for r in rows])


def axis_rows(rows, axis, d):
    """Cells on one axis at one d, baseline included, ordered along the axis.

    The baseline is logged under axis ``"base"``; every axis needs it as its
    own middle point, which is the whole reason for the one-at-a-time design.
    """
    w0, d0, a0 = BASE
    keep = [r for r in rows if int(r["d"]) == d
            and (r["axis"] == axis or r["axis"] == "base")]
    order = {"width": lambda r: (int(r["width"]),),
             "depth": lambda r: (int(r["depth"]),),
             "activation": lambda r: (r["activation"],)}[axis]
    groups = {}
    for r in keep:
        groups.setdefault(order(r), []).append(r)
    return [(k[0], v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])]


def summarize(rows):
    """Per (axis, d, setting): median/mean/spread of test error and own-sample fit."""
    out = []
    for axis in ("width", "depth", "activation"):
        for d in sorted({int(r["d"]) for r in rows}):
            for setting, cells in axis_rows(rows, axis, d):
                err = _f(cells, "rel_l2")
                fit = _f(cells, "rel_fit_sampled")
                out.append({
                    "axis": axis, "d": d, "setting": setting,
                    "n_seeds": len(cells),
                    "params": int(cells[0]["params"]),
                    "median": float(np.median(err)),
                    "mean": float(err.mean()),
                    "spread": float(err.max() / err.min()),
                    "fit_median": float(np.median(fit)),
                    "seconds": float(_f(cells, "train_seconds").mean()),
                    "is_base": (int(cells[0]["width"]), int(cells[0]["depth"]),
                                cells[0]["activation"]) == BASE,
                })
    return out


def matched_params(rows, tol=0.35):
    """Pairs of cells at different (width, depth) whose parameter counts agree.

    Width and depth both move the parameter count, so neither axis on its own
    separates "a better shape" from "more parameters". Where a wide-shallow and
    a narrow-deep cell land within ``tol`` of each other in ``params``, the
    comparison between them does. Returns [] when the sweep contains no such
    pair, which is the honest outcome and not an error.
    """
    pairs = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = [r for r in rows if int(r["d"]) == d and r["activation"] == "tanh"]
        shapes = sorted({(int(r["width"]), int(r["depth"])) for r in cells})
        for i, a in enumerate(shapes):
            for b in shapes[i + 1:]:
                if a[0] == b[0] or a[1] == b[1]:
                    continue  # same axis: not a shape comparison
                pa, pb = n_params(d, *a), n_params(d, *b)
                if abs(np.log(pa / pb)) > np.log(1 + tol):
                    continue
                ea = np.median(_f([r for r in cells
                                   if (int(r["width"]), int(r["depth"])) == a],
                                  "rel_l2"))
                eb = np.median(_f([r for r in cells
                                   if (int(r["width"]), int(r["depth"])) == b],
                                  "rel_l2"))
                pairs.append({"d": d, "a": a, "b": b, "params_a": pa,
                              "params_b": pb, "err_a": float(ea),
                              "err_b": float(eb)})
    return pairs


def table(rows=None):
    rows = read_csv(ARCH_CSV) if rows is None else rows
    lines = []
    w0, d0, a0 = BASE
    lines.append(f"Sec. 14 baseline: width {w0}, depth {d0}, {a0}, "
                 f"{BUDGET['n_interior']} uniform points, {BUDGET['steps']} steps.")
    lines.append("A network that outputs zero scores rel L2 = 1.000.\n")
    header = (f"{'axis':11s} {'d':>2s} {'setting':>9s} {'params':>9s} "
              f"{'rel L2 (med)':>13s} {'spread':>7s} {'own-sample':>11s} {'s':>6s}")
    lines.append(header)
    lines.append("-" * len(header))
    for s in summarize(rows):
        mark = "  <- base" if s["is_base"] else ""
        lines.append(f"{s['axis']:11s} {s['d']:2d} {str(s['setting']):>9s} "
                     f"{s['params']:9,d} {s['median']:13.4e} "
                     f"{s['spread']:6.2f}x {s['fit_median']:11.4e} "
                     f"{s['seconds']:6.0f}{mark}")
    pairs = matched_params(rows)
    if pairs:
        lines.append("\nMatched-parameter shape comparisons "
                     "(the only place width and depth separate from size):")
        for p in pairs:
            lines.append(f"  d={p['d']:2d}  w{p['a'][0]}xd{p['a'][1]} "
                         f"({p['params_a']:,} params) {p['err_a']:.4e}   vs   "
                         f"w{p['b'][0]}xd{p['b'][1]} ({p['params_b']:,}) "
                         f"{p['err_b']:.4e}")
    else:
        lines.append("\nNo two shapes in this sweep land within 35% in parameter "
                     "count, so width and depth are not separated from size here.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure(rows=None, name="highd_arch.png"):
    """Three panels: the two size axes, then the activation control across d.

    The third panel is the one that carries the result, so it plots both
    numbers: the uniform-L2 test error (solid) and the fit on the network's own
    4000 labelled points (dashed). Those two separate at high d and that
    separation is the finding -- a single-metric panel would hide it.
    """
    import matplotlib.pyplot as plt

    rows = read_csv(ARCH_CSV) if rows is None else rows
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    def zero_line(ax):
        ax.axhline(1.0, color="0.45", ls=":", lw=1.2)
        ax.annotate("a network that outputs zero", xy=(0, 1.0), xytext=(2, 3),
                    textcoords="offset points", fontsize=7, color="0.35",
                    va="bottom")

    for ax, (axis, xlabel) in zip(axs, [("width", "hidden width"),
                                        ("depth", "hidden layers")]):
        for d, color, marker in zip(DIMS, ("C0", "C3"), ("o", "s")):
            groups = axis_rows(rows, axis, d)
            if len(groups) < 2:
                continue
            xs = list(range(len(groups)))
            med = np.array([np.median(_f(c, "rel_l2")) for _, c in groups])
            lo = np.array([_f(c, "rel_l2").min() for _, c in groups])
            hi = np.array([_f(c, "rel_l2").max() for _, c in groups])
            ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt=marker + "-",
                        color=color, capsize=3, lw=1.4, label=f"d = {d}")
            ax.set_xticks(xs, [str(v) for v, _ in groups])
        zero_line(ax)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("relative $L^2$")
        ax.set_title(f"{xlabel}: the size axes barely move it", loc="left",
                     fontsize=9)
        ax.legend(fontsize=8)

    ax = axs[2]
    dims = sorted({int(r["d"]) for r in rows})
    for act, color in (("tanh", "C0"), ("sin", "C2")):
        for key, ls, lw in (("rel_l2", "-", 1.6),
                            ("rel_fit_sampled", "--", 1.2)):
            ys, xs = [], []
            for d in dims:
                cells = [r for r in rows if int(r["d"]) == d
                         and r["activation"] == act
                         and (int(r["width"]), int(r["depth"])) == BASE[:2]]
                if not cells:
                    continue
                xs.append(d)
                ys.append(float(np.median(_f(cells, key))))
            label = f"{act}, " + ("test" if key == "rel_l2" else "own sample")
            ax.plot(xs, ys, ls, color=color, lw=lw, marker="o", ms=3.5,
                    label=label)
    zero_line(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("relative $L^2$ (median of 3 seeds)")
    ax.set_title("activation: fixes the fit, not the generalization", loc="left",
                 fontsize=9)
    ax.legend(fontsize=7.5)

    w0, d0, a0 = BASE
    fig.suptitle("Sec. 16: one factor at a time from Sec. 14's regression "
                 f"control (width {w0}, depth {d0}, {a0}, 4000 labelled points, "
                 "2000 Adam steps)", fontsize=9)
    savefig(fig, name)


def figures_from_committed():
    print(table())
    figure()


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="run the sweep")
    ap.add_argument("--seconds", type=float, default=None,
                    help="stop after this many seconds and write what is done")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny end-to-end smoke run, writes nothing")
    args = ap.parse_args(argv)

    if args.quick:
        rows = sweep(dims=(2,), seeds=(0,), steps=20, write=False,
                     extra_dims=())
        print(f"\n{len(rows)} cells")
        return 0
    if args.run:
        sweep(seconds=args.seconds)
        print()
        print(table())
        return 0
    if args.table:
        print(table())
        return 0
    if args.figures:
        figures_from_committed()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
