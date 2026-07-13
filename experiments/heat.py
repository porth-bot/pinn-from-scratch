"""The 1D heat equation vs its exact Fourier series -- the repo's first solve.

Problem.  On ``x in [0, 1]``, ``t in [0, 1]`` solve

    u_t = alpha u_xx,
    u(x, 0) = sum_k a_k sin(k pi x)        (three sine modes),
    u(0, t) = u(1, t) = 0                   (homogeneous Dirichlet).

Why this problem first.  The sine modes ``sin(k pi x)`` are exactly the
eigenfunctions of the Laplacian on ``[0, 1]`` with these boundary conditions,
each with eigenvalue ``-(k pi)^2``.  So the initial condition is already an
eigenfunction expansion, and the heat semigroup just multiplies mode ``k`` by
``exp(-alpha (k pi)^2 t)``.  The solution never leaves the span of the three
modes:

    u(x, t) = sum_k a_k sin(k pi x) exp(-alpha (k pi)^2 t).

That is an *exact* closed form (not a truncation), so the PINN's error can be
measured pointwise against truth everywhere.  The three modes decay at rates
1 : 4 : 9, so the high mode is gone by mid-time while the fundamental lingers --
a clean multi-scale target.

What this script measures.
    (1) An error heatmap ``|u_pinn - u_exact|`` over the space-time rectangle
        for the default network.
    (2) Convergence of the relative L2 error as the number of interior
        collocation points grows ``{1k .. 16k}`` (network fixed), and as the
        network width grows ``{32 .. 256}`` (collocation fixed).
Both sweeps are logged to CSV so the README numbers regenerate without
retraining.

Run:  python experiments/heat.py            # full sweeps + figures (slow)
      python experiments/heat.py --quick     # tiny run, for a smoke check
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import plt, savefig, write_csv
from pinn import derivatives as D
from pinn.losses import boundary_points, initial_points, interior_points
from pinn.model import MLP, set_seed

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
ALPHA = 0.05
MODES = (1, 2, 3)
AMPS = (1.0, 0.5, 0.25)
X_RANGE = (0.0, 1.0)
T_RANGE = (0.0, 1.0)


def heat_exact(x, t, modes=MODES, amps=AMPS, alpha=ALPHA):
    """Exact solution ``sum_k a_k sin(k pi x) exp(-alpha (k pi)^2 t)``.

    Works on numpy arrays or python floats (broadcasting ``x`` and ``t``).
    This is the ground truth every error in the repo's first experiment is
    measured against; ``tests/test_heat.py`` checks it satisfies the PDE by
    finite differences, and matches the IC / BCs.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.zeros(np.broadcast(x, t).shape)
    for k, a in zip(modes, amps):
        out = out + a * np.sin(k * np.pi * x) * np.exp(-alpha * (k * np.pi) ** 2 * t)
    return out


def initial_condition(x, modes=MODES, amps=AMPS):
    """u(x, 0) = sum_k a_k sin(k pi x), as a torch tensor matching ``x``."""
    out = torch.zeros_like(x)
    for k, a in zip(modes, amps):
        out = out + a * torch.sin(k * np.pi * x)
    return out


def heat_residual(u, coords, alpha=ALPHA):
    """PDE residual r = u_t - alpha u_xx for the heat equation."""
    return D.u_t(u, coords) - alpha * D.u_xx(u, coords)


# ---------------------------------------------------------------------------
# Evaluation grid + error metrics (against the exact solution)
# ---------------------------------------------------------------------------
def _eval_grid(nx=101, nt=101):
    x = np.linspace(X_RANGE[0], X_RANGE[1], nx)
    t = np.linspace(T_RANGE[0], T_RANGE[1], nt)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    return x, t, XX, TT


def predict(model, XX, TT):
    """Evaluate the network on a meshgrid, returning a numpy array like XX."""
    coords = np.stack([XX.ravel(), TT.ravel()], axis=1)
    with torch.no_grad():
        u = model(torch.tensor(coords, dtype=torch.float32)).numpy().reshape(XX.shape)
    return u


def rel_l2_error(model, nx=101, nt=101):
    """Relative L2 error ||u_pinn - u_exact|| / ||u_exact|| on a dense grid."""
    _, _, XX, TT = _eval_grid(nx, nt)
    u_hat = predict(model, XX, TT)
    u_true = heat_exact(XX, TT)
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    n_interior=4000,
    width=128,
    depth=4,
    steps=5000,
    lr=1e-3,
    n_ic=400,
    n_bc=200,
    seed=0,
    w_ic=1.0,
    w_bc=1.0,
    verbose=False,
):
    """Train a heat-equation PINN with Adam; return (model, history).

    Collocation is sampled once (fixed set) for reproducibility and speed.
    IC and BC are enforced as data losses; the residual as the mean squared
    ``u_t - alpha u_xx``.  History is a list of (step, total_loss, rel_l2).
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    interior = interior_points(n_interior, X_RANGE, T_RANGE, gen)
    ic = initial_points(n_ic, X_RANGE, T_RANGE[0], gen)
    ic_target = initial_condition(ic[:, 0:1])
    left, right = boundary_points(n_bc, X_RANGE, T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    bc_target = torch.zeros(bc.shape[0], 1)

    model = MLP(in_dim=2, out_dim=1, width=width, depth=depth, activation="tanh")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for step in range(steps + 1):
        opt.zero_grad()
        u_int = model(interior)
        r = heat_residual(u_int, interior)
        loss_r = torch.mean(r ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == steps:
            err = rel_l2_error(model)
            history.append((step, float(loss.item()), err))
            if verbose:
                print(
                    f"  step {step:5d}  loss {loss.item():.3e}  "
                    f"(r {loss_r.item():.2e} ic {loss_ic.item():.2e} "
                    f"bc {loss_bc.item():.2e})  relL2 {err:.4f}"
                )
    return model, history


# ---------------------------------------------------------------------------
# Experiment 1: error heatmap for the default network
# ---------------------------------------------------------------------------
def figure_error_heatmap(model):
    x, t, XX, TT = _eval_grid()
    u_hat = predict(model, XX, TT)
    u_true = heat_exact(XX, TT)
    err = np.abs(u_hat - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axes[0].pcolormesh(t, x, u_true, shading="auto", cmap="RdBu_r")
    axes[0].set_title("exact  u(x, t)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(t, x, u_hat, shading="auto", cmap="RdBu_r")
    axes[1].set_title("PINN  u_theta(x, t)")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(t, x, err, shading="auto", cmap="magma")
    axes[2].set_title(f"|error|   (rel L2 = {rel_l2_error(model):.2e})")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    fig.suptitle(
        "Heat equation: PINN vs exact Fourier solution "
        f"(alpha={ALPHA}, modes {MODES})",
        y=1.05,
    )
    savefig(fig, "heat_error.png")


# ---------------------------------------------------------------------------
# Experiment 2: convergence sweeps
# ---------------------------------------------------------------------------
def sweep_collocation(counts, width, steps, seed=0):
    rows = []
    for n in counts:
        t0 = time.time()
        model, hist = train(n_interior=n, width=width, steps=steps, seed=seed)
        err = rel_l2_error(model)
        secs = time.time() - t0
        rows.append(
            {"n_interior": n, "width": width, "steps": steps,
             "rel_l2": f"{err:.6e}", "seconds": f"{secs:.1f}"}
        )
        print(f"collocation n={n:6d}  relL2={err:.4e}  ({secs:.0f}s)")
    return rows


def sweep_width(widths, n_interior, steps, seed=0):
    rows = []
    for w in widths:
        t0 = time.time()
        model, hist = train(n_interior=n_interior, width=w, steps=steps, seed=seed)
        err = rel_l2_error(model)
        secs = time.time() - t0
        rows.append(
            {"width": w, "n_interior": n_interior, "steps": steps,
             "rel_l2": f"{err:.6e}", "seconds": f"{secs:.1f}"}
        )
        print(f"width w={w:4d}  relL2={err:.4e}  ({secs:.0f}s)")
    return rows


def figure_convergence(coll_rows, width_rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    n = [int(r["n_interior"]) for r in coll_rows]
    e = [float(r["rel_l2"]) for r in coll_rows]
    axes[0].loglog(n, e, "o-")
    axes[0].set_xlabel("interior collocation points")
    axes[0].set_ylabel("relative L2 error")
    axes[0].set_title("vs collocation count (width 128)")
    axes[0].grid(True, which="both", alpha=0.3)

    w = [int(r["width"]) for r in width_rows]
    ew = [float(r["rel_l2"]) for r in width_rows]
    axes[1].loglog(w, ew, "s-", color="C1")
    axes[1].set_xlabel("network width")
    axes[1].set_ylabel("relative L2 error")
    axes[1].set_title("vs width (4k collocation)")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "heat_convergence.png")


def main(quick=False):
    if quick:
        print("[quick] tiny run for a smoke check")
        model, hist = train(n_interior=500, width=32, steps=300, verbose=True)
        print("final rel L2:", rel_l2_error(model))
        return

    print("=" * 64)
    print("Heat equation PINN: default network")
    print("=" * 64)
    t0 = time.time()
    model, hist = train(n_interior=4000, width=128, steps=5000, verbose=True)
    print(f"trained default in {time.time() - t0:.0f}s")
    figure_error_heatmap(model)
    write_csv(
        "heat_training.csv",
        ["step", "loss", "rel_l2"],
        [{"step": s, "loss": f"{l:.6e}", "rel_l2": f"{e:.6e}"} for s, l, e in hist],
    )

    print("\n" + "=" * 64)
    print("Convergence sweep: collocation count")
    print("=" * 64)
    coll_rows = sweep_collocation(
        counts=[1000, 2000, 4000, 8000, 16000], width=128, steps=3000
    )
    write_csv("heat_collocation.csv",
              ["n_interior", "width", "steps", "rel_l2", "seconds"], coll_rows)

    print("\n" + "=" * 64)
    print("Convergence sweep: network width")
    print("=" * 64)
    width_rows = sweep_width(
        widths=[32, 64, 128, 256], n_interior=4000, steps=3000
    )
    write_csv("heat_width.csv",
              ["width", "n_interior", "steps", "rel_l2", "seconds"], width_rows)

    figure_convergence(coll_rows, width_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
