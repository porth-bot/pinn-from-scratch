"""The honest classical baseline: a Crank-Nicolson finite-difference solver for
the same heat equation the PINN solves in ``experiments/heat.py``.

The repo's README leads its limitations with "on these problems, classical
solvers win, decisively." This script makes that concrete instead of asserting
it: it solves the identical problem (same ``alpha``, same three-mode IC, same
homogeneous Dirichlet BCs, same space-time rectangle) with a textbook
second-order scheme and puts the accuracy and wall-clock next to the PINN's.

Crank-Nicolson.  Discretise ``u_t = alpha u_xx`` on a grid ``x_i`` (spacing
``dx``), ``t_n`` (step ``dt``). Approximate ``u_xx`` by the central second
difference ``L u_i = (u_{i-1} - 2 u_i + u_{i+1}) / dx^2`` and average it over
the old and new time levels (the trapezoidal rule in time):

    (u^{n+1} - u^n) / dt = (alpha / 2) (L u^{n+1} + L u^n).

With ``s = alpha dt / (2 dx^2)`` this rearranges to a tridiagonal solve per
step,

    (I - s L_mat) u^{n+1} = (I + s L_mat) u^n,   L_mat = tridiag(1, -2, 1),

which is second-order accurate in *both* dx and dt and unconditionally stable
(no ``dt <= dx^2 / 2alpha`` restriction, unlike explicit Euler). Homogeneous
Dirichlet BCs make the boundary values identically zero, so only the interior
nodes are unknowns and the matrix is constant in time -- its Thomas-algorithm
factorisation is reused every step.

The tridiagonal solve is written from scratch (the Thomas algorithm) to keep
the baseline self-contained and in the spirit of the repo; ``scipy`` is not a
dependency.

Run:  python experiments/crank_nicolson.py           # sweep + figure + table
      python experiments/crank_nicolson.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import plt, read_csv, savefig, write_csv
from heat import ALPHA, AMPS, MODES, T_RANGE, X_RANGE, heat_exact


# ---------------------------------------------------------------------------
# Crank-Nicolson solver
# ---------------------------------------------------------------------------
def _ic_vector(x):
    """u(x, 0) = sum_k a_k sin(k pi x) on the numpy grid ``x``."""
    out = np.zeros_like(x)
    for k, a in zip(MODES, AMPS):
        out = out + a * np.sin(k * np.pi * x)
    return out


def _thomas(lower, diag, upper, d):
    """Solve a tridiagonal system ``T u = d`` by the Thomas algorithm.

    ``lower``/``upper`` are the sub-/super-diagonals (length n-1), ``diag`` the
    main diagonal (length n). O(n) forward elimination + back substitution;
    stable here because the CN matrix is diagonally dominant (diag 1 + 2s,
    off-diagonals -s, so |diag| > sum|off|).
    """
    n = diag.shape[0]
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = d[0] / diag[0]
    for i in range(1, n - 1):
        m = diag[i] - lower[i - 1] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (d[i] - lower[i - 1] * dp[i - 1]) / m
    m = diag[n - 1] - lower[n - 2] * cp[n - 2]
    dp[n - 1] = (d[n - 1] - lower[n - 2] * dp[n - 2]) / m
    u = np.empty(n)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]
    return u


def crank_nicolson(nx, nt, alpha=ALPHA):
    """Solve the heat equation with Crank-Nicolson on an ``nx`` x ``nt`` grid.

    ``nx`` and ``nt`` are the number of *intervals*, so there are ``nx + 1``
    spatial points (two of them the zero Dirichlet boundaries) and ``nt + 1``
    time levels. Returns ``(x, t, U)`` with ``U`` of shape ``(nx+1, nt+1)`` and
    the wall-clock ``seconds`` spent in the stepping loop.
    """
    x = np.linspace(X_RANGE[0], X_RANGE[1], nx + 1)
    t = np.linspace(T_RANGE[0], T_RANGE[1], nt + 1)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    s = alpha * dt / (2.0 * dx * dx)

    n_int = nx - 1  # interior unknowns (boundaries pinned to 0)
    # Constant CN matrix A = I - s L_mat: diag 1 + 2s, off-diagonals -s.
    a_diag = np.full(n_int, 1.0 + 2.0 * s)
    a_off = np.full(n_int - 1, -s)

    U = np.zeros((nx + 1, nt + 1))
    U[:, 0] = _ic_vector(x)
    u = U[1:-1, 0].copy()

    t0 = time.perf_counter()
    for n in range(nt):
        # right-hand side d = B u^n, B = I + s L_mat: diag 1 - 2s, off +s.
        d = (1.0 - 2.0 * s) * u
        d[:-1] += s * u[1:]
        d[1:] += s * u[:-1]
        u = _thomas(a_off, a_diag, a_off, d)
        U[1:-1, n + 1] = u
    seconds = time.perf_counter() - t0
    return x, t, U, seconds


def rel_l2(x, t, U):
    """Relative L2 error of the CN grid solution against the exact solution,
    measured on the CN's own nodes (the same relative-L2 metric heat.py uses
    for the PINN, so the two numbers are comparable)."""
    XX, TT = np.meshgrid(x, t, indexing="ij")
    u_true = heat_exact(XX, TT)
    return float(np.linalg.norm(U - u_true) / np.linalg.norm(u_true))


# ---------------------------------------------------------------------------
# Experiment: accuracy + wall-clock across resolutions, vs the PINN
# ---------------------------------------------------------------------------
def _pinn_reference():
    """Read the PINN's heat-problem accuracy and wall-clock off committed logs
    (the Adam optimizer run in optimizer_study.py -- same problem, same Adam
    trainer). Returns (rel_l2, seconds) or None if the log is absent."""
    try:
        rows = read_csv("optimizer_adam.csv")
    except FileNotFoundError:
        return None
    last = rows[-1]
    return float(last["rel_l2"]), float(last["seconds"])


def run(quick=False):
    resolutions = [(20, 20), (40, 40), (80, 80), (160, 160)]
    if quick:
        resolutions = [(20, 20), (40, 40)]

    rows = []
    prev_err = None
    for nx, nt in resolutions:
        x, t, U, seconds = crank_nicolson(nx, nt)
        err = rel_l2(x, t, U)
        order = "" if prev_err is None else f"{np.log2(prev_err / err):.2f}"
        rows.append(
            {"nx": nx, "nt": nt, "rel_l2": err, "seconds": seconds, "order": order}
        )
        prev_err = err
        print(f"CN nx={nx:4d} nt={nt:4d}  rel_l2={err:.3e}  {seconds*1e3:7.2f} ms")

    write_csv("crank_nicolson.csv", ["nx", "nt", "rel_l2", "seconds", "order"], rows)

    pinn = _pinn_reference()
    if pinn is not None:
        p_err, p_sec = pinn
        print(f"PINN (Adam) rel_l2={p_err:.3e}  {p_sec:.1f} s")
        _make_figure(rows, pinn)
    return rows, pinn


def _make_figure(rows, pinn):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cn_sec = [r["seconds"] for r in rows]
    cn_err = [r["rel_l2"] for r in rows]
    ax.loglog(cn_sec, cn_err, "o-", color="#2166ac", label="Crank-Nicolson")
    for r in rows:
        ax.annotate(
            f"{r['nx']}x{r['nt']}",
            (r["seconds"], r["rel_l2"]),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
        )
    p_err, p_sec = pinn
    ax.loglog([p_sec], [p_err], "s", color="#b2182b", ms=9, label="PINN (Adam)")
    ax.annotate(
        "PINN", (p_sec, p_err),
        textcoords="offset points", xytext=(-8, -14), fontsize=8, color="#b2182b",
    )
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("relative $L^2$ error")
    ax.set_title("Heat equation: classical FD vs PINN")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    savefig(fig, "crank_nicolson.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
