"""Optimizer study: our Adam loop vs torch L-BFGS on the heat equation.

Every experiment in this repo trains with Adam. But the PINN literature's
default recipe (Raissi et al. 2019 and most that follow) is **Adam then
L-BFGS**: a first-order optimizer to get into a good basin, then a quasi-Newton
method that uses curvature to polish the smooth, low-dimensional PINN loss to an
accuracy first-order methods reach only very slowly. This script measures why,
honestly, on the heat problem (exact Fourier ground truth, so error is truth).

The comparison is made fair by fixing everything but the optimizer: identical
network init (same seed), identical fixed collocation / IC / BC sets, identical
loss. Cost is reported in two currencies:

  - wall-clock seconds, and
  - *loss-and-gradient evaluations* -- one full forward+backward over the
    collocation set. Adam does exactly one per step; L-BFGS calls its closure
    several times per iteration (the strong-Wolfe line search), so a per-eval
    axis is the honest way to compare a cheap-step method against an
    expensive-step one (same idea as the ESS-per-gradient axis in the sibling
    mcmc-from-scratch repo).

Three regimes: Adam only, L-BFGS only (from the same init), and the standard
hybrid (short Adam warmup, then L-BFGS). Figure: rel L2 vs evaluations.

Run:  python experiments/optimizer_study.py            # full run + figure
      python experiments/optimizer_study.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import plt, read_csv, savefig, write_csv
from heat import (
    X_RANGE,
    T_RANGE,
    heat_residual,
    initial_condition,
    rel_l2_error,
)
from pinn.losses import boundary_points, initial_points, interior_points
from pinn.model import MLP, set_seed


# ---------------------------------------------------------------------------
# Shared setup: one problem instance (fixed collocation), one loss closure.
# ---------------------------------------------------------------------------
def _build(n_interior, width, depth, seed, n_ic=400, n_bc=200):
    """Return (model, loss_fn) with a fixed collocation/IC/BC set and identical
    init, so the only thing that varies across regimes is the optimizer."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    interior = interior_points(n_interior, X_RANGE, T_RANGE, gen)
    ic = initial_points(n_ic, X_RANGE, T_RANGE[0], gen)
    ic_target = initial_condition(ic[:, 0:1])
    left, right = boundary_points(n_bc, X_RANGE, T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    bc_target = torch.zeros(bc.shape[0], 1)

    model = MLP(in_dim=2, out_dim=1, width=width, depth=depth, activation="tanh")

    def loss_fn():
        r = heat_residual(model(interior), interior)
        loss_r = torch.mean(r ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        return loss_r + loss_ic + loss_bc

    return model, loss_fn


def _record(history, evals, t0, model):
    history.append((evals, time.time() - t0, rel_l2_error(model)))


# ---------------------------------------------------------------------------
# Three training regimes, all logging (evals, seconds, rel_l2).
# ---------------------------------------------------------------------------
def train_adam(model, loss_fn, steps, lr=1e-3, log_every=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, t0 = [], time.time()
    for step in range(steps + 1):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            _record(history, step, t0, model)  # 1 eval per Adam step
    return history


def train_lbfgs(model, loss_fn, outer, max_iter=20, base_evals=0, base_t=0.0):
    """L-BFGS in chunks so the trajectory can be logged. Each .step(closure)
    runs up to ``max_iter`` inner iterations; the closure counts every
    loss-and-gradient evaluation (line search included)."""
    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-16,
    )
    evals = [base_evals]
    history, t0 = [], time.time() - base_t

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        evals[0] += 1
        return loss

    for _ in range(outer):
        opt.step(closure)
        _record(history, evals[0], t0, model)
    return history


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run(quick=False):
    width, depth, seed = 64, 4, 0
    n_interior = 2000 if quick else 4000
    if quick:
        adam_steps, lbfgs_outer, warm_steps = 200, 5, 100
    else:
        adam_steps, lbfgs_outer, warm_steps = 8000, 30, 1000

    # 1) Adam only
    m_adam, loss_adam = _build(n_interior, width, depth, seed)
    h_adam = train_adam(m_adam, loss_adam, adam_steps)

    # 2) L-BFGS only, from the SAME init (fresh build, same seed)
    m_lb, loss_lb = _build(n_interior, width, depth, seed)
    h_lbfgs = train_lbfgs(m_lb, loss_lb, lbfgs_outer)

    # 3) Hybrid: short Adam warmup, then L-BFGS continues on the same weights
    m_hy, loss_hy = _build(n_interior, width, depth, seed)
    h_warm = train_adam(m_hy, loss_hy, warm_steps, log_every=100)
    base_evals, base_t, _ = h_warm[-1]
    h_lb2 = train_lbfgs(m_hy, loss_hy, lbfgs_outer,
                        base_evals=base_evals, base_t=base_t)
    h_hybrid = h_warm + h_lb2

    histories = {"adam": h_adam, "lbfgs": h_lbfgs, "hybrid": h_hybrid}
    for name, h in histories.items():
        rows = [{"evals": e, "seconds": round(s, 3), "rel_l2": r} for e, s, r in h]
        write_csv(f"optimizer_{name}.csv", ["evals", "seconds", "rel_l2"], rows)
    return histories


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _load(name):
    return [
        (int(r["evals"]), float(r["seconds"]), float(r["rel_l2"]))
        for r in read_csv(f"optimizer_{name}.csv")
    ]


def summarize():
    print(f"{'regime':8s}  {'final relL2':>11s}  {'evals':>7s}  {'seconds':>8s}"
          f"  {'best relL2':>10s}")
    print("-" * 54)
    for name in ("adam", "lbfgs", "hybrid"):
        h = _load(name)
        e, s, r = h[-1]
        best = min(x[2] for x in h)
        print(f"{name:8s}  {r:11.3e}  {e:7d}  {s:8.1f}  {best:10.3e}")


def make_figure():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    styles = {
        "adam": ("#1f77b4", "-", "Adam (8k steps)"),
        "lbfgs": ("#a11", "-", "L-BFGS (from init)"),
        "hybrid": ("#2a2", "--", "hybrid: Adam 1k -> L-BFGS"),
    }
    for name, (c, ls, label) in styles.items():
        h = _load(name)
        evals = [x[0] for x in h]
        rel = [x[2] for x in h]
        ax.plot(evals, rel, ls, color=c, label=label, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("loss-and-gradient evaluations")
    ax.set_ylabel("relative L2 error vs exact")
    ax.set_title("Heat PINN: L-BFGS curvature beats Adam per evaluation")
    ax.legend()
    savefig(fig, "optimizer_study.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figure-only", action="store_true",
                    help="regenerate the figure/table from committed CSVs")
    args = ap.parse_args()
    if not args.figure_only:
        run(quick=args.quick)
    summarize()
    make_figure()


if __name__ == "__main__":
    main()
