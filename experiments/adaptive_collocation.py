"""Residual-adaptive collocation on the Burgers shock: a failure mode and its fix.

Uniform collocation spends its point budget evenly, but Burgers concentrates
essentially all of its error in the thin viscous shock at ``x = 0`` (the
``burgers.py`` experiment measures 94% of the squared error inside |x| <= 0.1).
The obvious idea is to move points to where the residual is large. This script
runs that idea two ways and shows why *how* you do it matters more than the idea.

Three arms, each with the SAME point budget (N total interior points) and an
identical warmup -- because all arms share the seed and the same initial uniform
set, the first WARMUP steps are bit-for-bit the same run:

  U. uniform   -- keep the warmed-up uniform set to the end (baseline).
  R. resample  -- after warmup, REPLACE the whole set by a residual-density draw
                  every RESAMPLE steps (RAD; Wu et al. 2023).
  A. RAR       -- after warmup, keep a uniform base and ADD residual-density
                  points to it every RESAMPLE steps (RAR; Lu et al. 2021),
                  matched to the same N total (base = N - add).

The finding: **resampling the whole set destabilizes a good fit.** Burgers'
residual is near-singular at the shock (it contains ``u_xx``, which reaches order
100), so a set drawn proportional to |residual| piles points onto the least
tractable region; the loss is then dominated by those points, the smooth region
loses its coverage, and a model that was at rel L2 ~0.03 collapses to ~0.4-0.9
within a few dozen steps. **RAR fixes it** by never removing the uniform base:
the smooth region stays covered, the added points sharpen the shock, and the run
stays stable and improves. Same adaptive machinery
(``pinn.losses.adaptive_interior_points``); the only difference is replace vs add.

Run:  python experiments/adaptive_collocation.py   (~12 min on CPU)
"""

import numpy as np
import torch

from common import plt, savefig, write_csv
from pinn.losses import (
    adaptive_interior_points,
    boundary_points,
    initial_points,
    interior_points,
)
from pinn.model import MLP, set_seed

# Reuse the Burgers problem definition and its Cole-Hopf ground truth.
from burgers import (  # noqa: E402
    T_RANGE,
    X_RANGE,
    _eval_grid,
    burgers_exact,
    burgers_residual,
    initial_condition,
    predict,
    rel_l2_error,
)

N_TOTAL = 3000          # every arm trains on this many interior points per step
RAR_ADD = 1000          # RAR: N_TOTAL - RAR_ADD uniform base + RAR_ADD adaptive
STEPS = 10000
WARMUP = 5000           # all arms train identically on the uniform set until here
RESAMPLE = 1000         # after warmup, the adaptive arms act this often
WIDTH, DEPTH = 32, 5
LR = 1e-3
SEED = 0
SHOCK_BAND = 0.1        # |x| <= this defines the shock region for the error split
N_CAND = 20000          # candidate pool for the residual-density draw


def _build(seed):
    """Model + shared uniform base + IC/BC tensors (identical across arms)."""
    set_seed(seed)
    model = MLP(in_dim=2, out_dim=1, width=WIDTH, depth=DEPTH, activation="tanh")
    gen = torch.Generator().manual_seed(seed)
    base = interior_points(N_TOTAL, X_RANGE, T_RANGE, gen)
    ic = initial_points(512, X_RANGE, T_RANGE[0], gen)
    ic_target = initial_condition(ic[:, 0:1])
    left, right = boundary_points(128, X_RANGE, T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    bc_target = torch.zeros(bc.shape[0], 1)
    return model, gen, base, ic, ic_target, bc, bc_target


def _train(mode):
    """Train one arm ('uniform' | 'resample' | 'rar'); return (model, evals)."""
    model, gen, base, ic, ic_target, bc, bc_target = _build(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    interior = base
    evals = 0
    for step in range(STEPS + 1):
        if mode != "uniform" and step >= WARMUP and step % RESAMPLE == 0:
            if mode == "resample":
                interior = adaptive_interior_points(
                    model, burgers_residual, N_TOTAL, X_RANGE, T_RANGE, gen,
                    n_candidates=N_CAND, k=1.0, c=1.0)
            else:  # rar: preserved uniform base + added adaptive points
                add = adaptive_interior_points(
                    model, burgers_residual, RAR_ADD, X_RANGE, T_RANGE, gen,
                    n_candidates=N_CAND, k=1.0, c=1.0)
                keep = base[: N_TOTAL - RAR_ADD].detach().clone().requires_grad_(True)
                interior = torch.cat([keep, add], dim=0)
            evals += N_CAND
        opt.zero_grad()
        r = burgers_residual(model(interior), interior)
        loss = (torch.mean(r ** 2)
                + torch.mean((model(ic) - ic_target) ** 2)
                + torch.mean((model(bc) - bc_target) ** 2))
        loss.backward()
        opt.step()
    return model, evals


def _shock_split(model):
    """Mean |u_pinn - u_exact| inside vs outside the shock band |x| <= SHOCK_BAND."""
    x, _, XX, TT = _eval_grid()
    err = np.abs(predict(model, XX, TT) - burgers_exact(XX, TT))
    band = np.abs(x) <= SHOCK_BAND
    return float(err[band].mean()), float(err[~band].mean())


def _final_adaptive_points(model):
    """Where a residual-density draw lands for the trained RAR model (for the fig)."""
    gen = torch.Generator().manual_seed(SEED + 1)
    return adaptive_interior_points(
        model, burgers_residual, RAR_ADD, X_RANGE, T_RANGE, gen,
        n_candidates=N_CAND, k=1.0, c=1.0)


def figure(models, rar_points):
    """Three |error| fields (uniform / resample / RAR, shared colour scale on the
    two stable arms) + where RAR's added points land."""
    x, t, XX, TT = _eval_grid()
    fields = {k: np.abs(predict(m, XX, TT) - burgers_exact(XX, TT))
              for k, m in models.items()}
    vmax = max(fields["uniform"].max(), fields["rar"].max())  # resample saturates

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3), constrained_layout=True)
    titles = {"uniform": "U: uniform (baseline)",
              "resample": "R: resample (replace set)",
              "rar": "A: RAR (add to base)"}
    for ax, key in zip(axes[:3], ("uniform", "resample", "rar")):
        im = ax.pcolormesh(t, x, fields[key], shading="auto", cmap="magma",
                           vmin=0, vmax=vmax)
        ax.set_title(f"{titles[key]}\n|error|  (rel L2 = {rel_l2_error(models[key]):.2e})")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax)
    pts = rar_points.detach().numpy()
    axes[3].scatter(pts[:, 1], pts[:, 0], s=3, alpha=0.35, color="C0")
    for y in (SHOCK_BAND, -SHOCK_BAND):
        axes[3].axhline(y, color="k", ls=":", lw=0.8)
    axes[3].set_xlim(T_RANGE)
    axes[3].set_ylim(X_RANGE)
    axes[3].set_title("RAR added points\ncluster on the shock")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("x")
    fig.suptitle("Residual-adaptive collocation on Burgers: replacing the set "
                 "destabilizes the near-singular shock; adding to a uniform base "
                 "(RAR) is the stable fix", fontsize=10)
    savefig(fig, "adaptive_collocation.png")


def main():
    print(f"Burgers RAD/RAR: N={N_TOTAL}, steps={STEPS}, warmup={WARMUP}, "
          f"resample/add every {RESAMPLE} after warmup\n")
    models, evals = {}, {}
    for mode in ("uniform", "resample", "rar"):
        models[mode], evals[mode] = _train(mode)

    rar_points = _final_adaptive_points(models["rar"])
    frac_in_band = float((rar_points[:, 0].abs() <= SHOCK_BAND).float().mean())

    rows = []
    for mode in ("uniform", "resample", "rar"):
        band, off = _shock_split(models[mode])
        rows.append({
            "arm": mode,
            "rel_l2": rel_l2_error(models[mode]),
            "shock_band_mae": band,
            "off_shock_mae": off,
        })
    hdr = f"{'arm':>9} {'rel L2':>10} {'shock |err|':>12} {'off-shock |err|':>15}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>9} {r['rel_l2']:>10.3e} {r['shock_band_mae']:>12.3e} "
              f"{r['off_shock_mae']:>15.3e}")
    print(f"\nRAR added points: {100 * frac_in_band:.0f}% land in the shock band "
          f"|x|<={SHOCK_BAND} vs the ~{100 * SHOCK_BAND:.0f}% a uniform draw gives")
    print(f"adaptive arms' extra cost: {evals['rar']} candidate-pool residual "
          f"evals over training ({evals['rar'] // N_TOTAL} equivalent batches)")

    write_csv("adaptive_collocation.csv",
              ["arm", "rel_l2", "shock_band_mae", "off_shock_mae"], rows)
    figure(models, rar_points)


if __name__ == "__main__":
    main()
