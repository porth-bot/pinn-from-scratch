"""Viscous Burgers' equation vs a Cole-Hopf ground truth -- the shock problem.

Problem.  On ``x in [-1, 1]``, ``t in [0, 1]`` solve

    u_t + u u_x = nu u_xx,      nu = 0.01 / pi,
    u(x, 0) = -sin(pi x),                       (initial condition)
    u(-1, t) = u(1, t) = 0.                      (Dirichlet)

This is the canonical PINN benchmark (Raissi et al. 2019).  Unlike the heat
equation, Burgers is *nonlinear*: the advection term ``u u_x`` steepens the
profile faster than the tiny viscosity ``nu`` can smear it, so an initially
smooth ``-sin(pi x)`` collapses into a thin internal layer -- a viscous shock --
at ``x = 0``.  That standing gradient (``u_x(0, t)`` reaches order ``-100``) is
exactly what makes the problem a good stress test: a network with spectral bias
resolves the smooth flanks long before the shock.

Ground truth by Cole-Hopf.  The nonlinear PDE has a *linear* twin.  Substitute

    u = -2 nu phi_x / phi                         (Cole-Hopf transform)

and Burgers becomes the heat equation ``phi_t = nu phi_xx`` (derivation in
``theory/derivations.md``, Day 12).  The heat equation has the closed-form
heat-kernel solution, so on the whole line

    u(x, t) = [ integral (x - y)/t * exp(-(x-y)^2/(4 nu t) - F(y)/(2 nu)) dy ]
              / [ integral         exp(-(x-y)^2/(4 nu t) - F(y)/(2 nu)) dy ],

where ``F(y) = integral_0^y u0(s) ds = (cos(pi y) - 1) / pi`` for our IC.  We
evaluate both integrals by Gauss-Hermite quadrature after the substitution
``x - y = sqrt(4 nu t) z`` turns the Gaussian factor into the ``exp(-z^2)``
Hermite weight -- so a couple hundred nodes integrate the smooth remainder
exactly, and no space grid or Gaussian-tail truncation is involved.  The
``phi_0`` factor grows like ``exp(1/(nu pi)) ~ exp(100)``, so the quadrature is
done in log-space with a per-point max subtracted (it cancels in the ratio).

``tests/test_burgers.py`` checks this ground truth satisfies the PDE by finite
differences (away from the under-resolvable shock), matches the IC, and respects
the boundary + odd symmetry that pin ``u = 0`` at ``x in {-1, 0, 1}`` exactly.

What this script measures.
    (1) An error heatmap ``|u_pinn - u_exact|`` over the space-time rectangle.
    (2) Profile slices at ``t = {0.25, 0.5, 0.75, 1.0}`` -- PINN vs exact --
        so the shock and where the network struggles are visible.
    (3) The *concentration* of the error at the shock: mean error and the
        share of total squared error inside the thin band ``|x| <= 0.1``.

Run:  python experiments/burgers.py            # full train + figures (slow)
      python experiments/burgers.py --quick     # tiny run, for a smoke check
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
NU = 0.01 / np.pi
X_RANGE = (-1.0, 1.0)
T_RANGE = (0.0, 1.0)

# Gauss-Hermite nodes/weights for the weight exp(-z^2); computed once. 220 nodes
# is far more than enough for the smooth remaining integrand (the error against a
# 400-node rule is < 1e-12 on this problem).
_GH_Z, _GH_W = np.polynomial.hermite.hermgauss(220)


def burgers_exact(x, t, nu=NU):
    """Exact Cole-Hopf solution, evaluated by Gauss-Hermite quadrature.

    Accepts numpy arrays or floats (``x`` and ``t`` are broadcast together) and
    returns an array of the broadcast shape.  ``t = 0`` is returned as the exact
    initial condition ``-sin(pi x)`` (the quadrature's ``(x - y)/t`` factor is a
    ``0/0`` in that limit).  Everything is computed in log-space for numerical
    stability -- see the module docstring.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    xb, tb = np.broadcast_arrays(x, t)
    xf = xb.ravel()
    tf = tb.ravel()
    out = np.empty_like(xf)

    small = tf < 1e-9
    out[small] = -np.sin(np.pi * xf[small])

    idx = np.where(~small)[0]
    if idx.size:
        xi = xf[idx][:, None]                    # (m, 1)
        ti = tf[idx][:, None]
        y = xi - np.sqrt(4.0 * nu * ti) * _GH_Z[None, :]   # (m, n): x - sqrt(4 nu t) z
        # log phi_0(y) = -F(y)/(2 nu) with F(y) = (cos(pi y) - 1)/pi
        logphi = (1.0 - np.cos(np.pi * y)) / (2.0 * nu * np.pi)
        m = logphi.max(axis=1, keepdims=True)              # per-point stabilizer
        w = _GH_W[None, :] * np.exp(logphi - m)            # stable positive weights
        fac = np.sqrt(4.0 * nu / ti) * _GH_Z[None, :]      # (x - y)/t = sqrt(4 nu/t) z
        num = (w * fac).sum(axis=1)
        den = w.sum(axis=1)
        out[idx] = num / den
    return out.reshape(xb.shape)


def initial_condition(x):
    """u(x, 0) = -sin(pi x), as a torch tensor matching ``x``."""
    return -torch.sin(np.pi * x)


def burgers_residual(u, coords, nu=NU):
    """PDE residual r = u_t + u u_x - nu u_xx for viscous Burgers."""
    return D.u_t(u, coords) + u * D.u_x(u, coords) - nu * D.u_xx(u, coords)


# ---------------------------------------------------------------------------
# Evaluation grid + error metrics (against the exact solution)
# ---------------------------------------------------------------------------
def _eval_grid(nx=201, nt=101):
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


def rel_l2_error(model, nx=201, nt=101):
    """Relative L2 error ||u_pinn - u_exact|| / ||u_exact|| on a dense grid."""
    _, _, XX, TT = _eval_grid(nx, nt)
    u_hat = predict(model, XX, TT)
    u_true = burgers_exact(XX, TT)
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    n_interior=10000,
    width=48,
    depth=6,
    steps=20000,
    lr=1e-3,
    n_ic=512,
    n_bc=256,
    seed=0,
    w_ic=1.0,
    w_bc=1.0,
    verbose=False,
):
    """Train a Burgers PINN with Adam; return (model, history).

    A wider/deeper net than the heat problem needs: the shock demands capacity.
    Collocation is sampled once (fixed set) for reproducibility.  IC and BC are
    enforced as data losses; the residual as the mean squared
    ``u_t + u u_x - nu u_xx``.  History is a list of (step, total_loss, rel_l2).
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
        r = burgers_residual(u_int, interior)
        loss_r = torch.mean(r ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        opt.step()

        if step % 1000 == 0 or step == steps:
            err = rel_l2_error(model)
            history.append((step, float(loss.item()), err))
            if verbose:
                print(
                    f"  step {step:6d}  loss {loss.item():.3e}  "
                    f"(r {loss_r.item():.2e} ic {loss_ic.item():.2e} "
                    f"bc {loss_bc.item():.2e})  relL2 {err:.4f}"
                )
    return model, history


# ---------------------------------------------------------------------------
# Experiment 1: error heatmap
# ---------------------------------------------------------------------------
def figure_error_heatmap(model):
    x, t, XX, TT = _eval_grid()
    u_hat = predict(model, XX, TT)
    u_true = burgers_exact(XX, TT)
    err = np.abs(u_hat - u_true)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axes[0].pcolormesh(t, x, u_true, shading="auto", cmap="RdBu_r")
    axes[0].set_title("exact  u(x, t)  (Cole-Hopf)")
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
        f"Burgers' equation: PINN vs Cole-Hopf ground truth (nu = 0.01/pi)",
        y=1.05,
    )
    savefig(fig, "burgers_error.png")


# ---------------------------------------------------------------------------
# Experiment 2: profile slices in time (the shock forming)
# ---------------------------------------------------------------------------
def figure_slices(model, times=(0.25, 0.5, 0.75, 1.0)):
    x = np.linspace(X_RANGE[0], X_RANGE[1], 401)
    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 3.0),
                             sharey=True)
    for ax, t0 in zip(np.atleast_1d(axes), times):
        u_true = burgers_exact(x, np.full_like(x, t0))
        coords = np.stack([x, np.full_like(x, t0)], axis=1)
        with torch.no_grad():
            u_hat = model(torch.tensor(coords, dtype=torch.float32)).numpy().ravel()
        ax.plot(x, u_true, "k-", lw=1.6, label="exact")
        ax.plot(x, u_hat, "C1--", lw=1.4, label="PINN")
        ax.set_title(f"t = {t0}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("u(x, t)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Burgers profiles: the shock steepens at x = 0", y=1.03)
    fig.tight_layout()
    savefig(fig, "burgers_slices.png")


# ---------------------------------------------------------------------------
# Experiment 3: quantify error concentration at the shock
# ---------------------------------------------------------------------------
def shock_concentration(model, band=0.1, nx=401, nt=201):
    """Where does the error live?  Return a table of rows for the CSV log.

    The shock is a thin layer at ``x = 0``.  We report, over the dense grid,
    the mean absolute error and the fraction of the *total squared* error that
    falls inside ``|x| <= band`` -- a small band holding a large error share is
    the quantitative statement that the PINN's error concentrates at the shock.
    """
    x, t, XX, TT = _eval_grid(nx, nt)
    u_hat = predict(model, XX, TT)
    u_true = burgers_exact(XX, TT)
    err = np.abs(u_hat - u_true)
    sq = err ** 2

    in_band = np.abs(XX) <= band
    band_width_frac = float(in_band.mean())          # share of grid area
    sq_share = float(sq[in_band].sum() / sq.sum())
    rows = [
        {"region": f"|x|<={band}", "grid_area_frac": f"{band_width_frac:.3f}",
         "mean_abs_err": f"{err[in_band].mean():.6e}",
         "max_abs_err": f"{err[in_band].max():.6e}",
         "sq_err_share": f"{sq_share:.4f}"},
        {"region": f"|x|>{band}", "grid_area_frac": f"{1 - band_width_frac:.3f}",
         "mean_abs_err": f"{err[~in_band].mean():.6e}",
         "max_abs_err": f"{err[~in_band].max():.6e}",
         "sq_err_share": f"{1 - sq_share:.4f}"},
    ]
    print(f"shock band |x|<={band}: {band_width_frac:.0%} of the area holds "
          f"{sq_share:.0%} of the squared error; "
          f"mean |err| {err[in_band].mean():.2e} in-band vs "
          f"{err[~in_band].mean():.2e} out")
    return rows


def main(quick=False):
    if quick:
        print("[quick] tiny run for a smoke check")
        model, hist = train(n_interior=1000, width=32, depth=4, steps=500,
                            verbose=True)
        print("final rel L2:", rel_l2_error(model))
        shock_concentration(model)
        return

    print("=" * 64)
    print("Burgers' PINN: default network")
    print("=" * 64)
    t0 = time.time()
    model, hist = train(verbose=True)
    print(f"trained default in {time.time() - t0:.0f}s")

    figure_error_heatmap(model)
    figure_slices(model)
    write_csv(
        "burgers_training.csv",
        ["step", "loss", "rel_l2"],
        [{"step": s, "loss": f"{l:.6e}", "rel_l2": f"{e:.6e}"} for s, l, e in hist],
    )
    rows = shock_concentration(model)
    write_csv(
        "burgers_shock.csv",
        ["region", "grid_area_frac", "mean_abs_err", "max_abs_err", "sq_err_share"],
        rows,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
