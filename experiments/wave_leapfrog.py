"""Sec. 17: the classical baseline Sec. 15 said it did not have.

Sec. 15 closes with a sentence that is a claim and an IOU: "a leapfrog scheme on
a plucked string is a few lines and would win on every axis". This runs it. The
scheme is the explicit central-difference (Stormer-Verlet / leapfrog)
discretization of u_tt = c^2 u_xx, which is a few lines, and the answer is worse
for the PINN than Sec. 15 guessed -- but only once the flattering ways of
measuring it are taken away, and taking them away is most of what this section
is about.

The scheme
----------
On a uniform grid x_j = j dx, t_n = n dt, with r = c dt / dx (the Courant
number) and delta^2 u_j = u_{j-1} - 2 u_j + u_{j+1},

    u_j^{n+1} = 2 u_j^n - u_j^{n-1} + r^2 delta^2 u_j^n.

The first level needs the initial velocity. With u_t(x, 0) = 0 the standard
second-order start comes from the Taylor expansion
u^1 = u^0 + dt u_t^0 + (dt^2/2) u_tt^0 and the PDE itself for u_tt:

    u_j^1 = u_j^0 + (r^2/2) delta^2 u_j^0.

Truncation error is O(dt^2 + dx^2), and von Neumann analysis gives the CFL
condition r <= 1: substituting u_j^n = g^n e^{i k j dx} gives
g + 1/g = 2 - 4 r^2 sin^2(k dx/2), whose roots lie on the unit circle (bounded)
exactly when |1 - 2 r^2 sin^2(k dx/2)| <= 1, i.e. r <= 1.

**At r = 1 the scheme is exact**, and that is not a numerical accident. The
update collapses to

    u_j^{n+1} = u_{j+1}^n + u_{j-1}^n - u_j^{n-1},

which is d'Alembert's formula written on the grid: with dt = dx/c the mesh
points lie exactly on the characteristics x +- ct, so the scheme transports the
initial data along them without interpolating anything. This holds for *any*
initial data, corner included, and Study A measures it at 1e-15. It is a
property of the 1D constant-coefficient wave equation, not a property of meshes,
and this section says so rather than quoting the 1e-15 as a speed-up.

What is measured
----------------
  A. Accuracy against d'Alembert, over the same space-time grid Sec. 15 scores
     the network on, across a Courant sweep and a grid refinement. Includes
     r = 1, where the scheme is exact, and r < 1, where it is not.
  B. The observed order of convergence -- and the two ways of measuring it that
     give the wrong answer. The sine initial condition is a standing wave of
     period 2/c, so at t = 1 and t = 2 it sits at a turning point of
     cos(pi c t), where a phase error enters *quadratically*: the measured order
     there is 4, not 2. And at t = 0.5 the exact solution is identically zero,
     so a relative error has no denominator at all. Sec. 15's own time window
     ends at t = 2, which is one of the flattering points.
  C. The plucked string, where the corner costs an order: the O(dx^2)
     truncation term carries a fourth derivative the solution does not have.
  D. Against the PINN, at matched accuracy and matched wall-clock -- Sec. 6's
     comparison, on this problem. The PINN numbers are read from Sec. 15's
     committed cells rather than retrained.
  E. The discrete energy, which the scheme conserves exactly for r < 1 by a
     hand-derived identity, scored against the same exact energy Sec. 15 reads
     the network's 0.766 against.

Run:  python experiments/wave_leapfrog.py
"""

import argparse
import time

import numpy as np

import wave as W
from common import read_csv, savefig, write_csv

CFLS = (1.0, 0.9, 0.5, 0.25)
GRIDS = (26, 51, 101, 201, 401, 801)
ORDER_TIMES = (0.7, 1.0, 2.0)     # generic, half-period, full period
ORDER_GRIDS = (101, 201, 401, 801)

SWEEP_CSV = "wave_leapfrog.csv"
ORDER_CSV = "wave_leapfrog_order.csv"
SWEEP_FIELDS = ["ic", "cfl", "nx", "nt", "dx", "dt", "rel_l2", "seconds",
                "energy_ratio", "energy_drift"]
ORDER_FIELDS = ["ic", "t_end", "nx", "rel_l2_fixed_norm", "order"]


# -- the scheme ---------------------------------------------------------------

def solve(nx, cfl, ic="sine", c=W.C, t_end=None, keep=False):
    """Leapfrog the wave equation to ``t_end``. Returns ``(x, t, U or u_final)``.

    ``dt`` is set from ``cfl`` and then trimmed so that an integer number of
    steps lands exactly on ``t_end`` -- otherwise the last step overshoots and
    the error being reported is partly a different final time. The trim only
    ever *lowers* the Courant number, so it cannot cross the stability limit.

    ``keep=True`` stores every time level, which is what Study A's space-time
    error needs; the default keeps three levels, which is all the recursion
    uses.
    """
    if not 0 < cfl <= 1.0:
        raise ValueError(f"the CFL condition is 0 < r <= 1, got {cfl}")
    t_end = W.T_RANGE[1] if t_end is None else float(t_end)
    x = np.linspace(*W.X_RANGE, nx)
    dx = x[1] - x[0]
    nt = int(np.ceil(t_end / (cfl * dx / c)))
    dt = t_end / nt
    r2 = (c * dt / dx) ** 2

    u_prev = W.f0(x, ic)
    u_prev[0] = u_prev[-1] = 0.0
    u_cur = u_prev + 0.5 * r2 * _delta2(u_prev)
    u_cur[0] = u_cur[-1] = 0.0
    out = [u_prev.copy(), u_cur.copy()] if keep else None
    for _ in range(1, nt):
        u_next = 2.0 * u_cur - u_prev + r2 * _delta2(u_cur)
        u_next[0] = u_next[-1] = 0.0
        u_prev, u_cur = u_cur, u_next
        if keep:
            out.append(u_cur.copy())
    t = np.linspace(0.0, t_end, nt + 1)
    return (x, t, np.stack(out, axis=1)) if keep else (x, t, u_cur)


def _delta2(u):
    """``u_{j-1} - 2 u_j + u_{j+1}``, zero at the pinned boundary rows."""
    d = np.zeros_like(u)
    d[1:-1] = u[:-2] - 2.0 * u[1:-1] + u[2:]
    return d


def discrete_energy(x, U, dt, c=W.C):
    """The leapfrog scheme's own conserved energy, at each half time level.

    Hand-derived: multiply the update by (u^{n+1} - u^{n-1}) / (2 dt) and sum
    over j. The kinetic part telescopes and what is left is conserved exactly,

        E^{n+1/2} = ||(u^{n+1} - u^n)/dt||^2 / 2
                    + c^2 <D_x u^{n+1}, D_x u^n> / 2,

    with the inner products taken with the dx weight. The potential term is a
    *product of two different time levels*, not the square of one -- that
    staggering is what makes it conserved, and using ||D_x u^n||^2 instead
    gives a quantity that oscillates at O(dt^2) and looks like drift. Positive
    definite exactly when r < 1, which is the energy proof of the CFL
    condition.

    Returns the array of E^{n+1/2}, one per step.
    """
    dx = x[1] - x[0]
    ut = (U[:, 1:] - U[:, :-1]) / dt
    ux = (U[1:, :] - U[:-1, :]) / dx
    kinetic = 0.5 * np.sum(ut ** 2, axis=0) * dx
    potential = 0.5 * c ** 2 * np.sum(ux[:, 1:] * ux[:, :-1], axis=0) * dx
    return kinetic + potential


# -- A. accuracy, on Sec. 15's own metric -------------------------------------

def space_time_error(x, t, U, ic):
    """Relative L2 over the whole space-time grid -- Sec. 15's metric.

    Sec. 15 scores the network on a 201x201 grid over the same domain, and a
    network outputting zero scores exactly 1.0 there. Using the same
    normalization here is what makes the two numbers comparable; a single time
    slice is not, and Study B shows what goes wrong when one is used.
    """
    XX, TT = np.meshgrid(x, t, indexing="ij")
    exact = W.wave_exact(XX, TT, ic)
    return float(np.linalg.norm(U - exact) / np.linalg.norm(exact))


def sweep(ics=W.ICS, cfls=CFLS, grids=GRIDS, write=True):
    rows = []
    for ic in ics:
        e_exact = W.exact_energy(ic)
        for cfl in cfls:
            for nx in grids:
                t0 = time.perf_counter()
                x, t, U = solve(nx, cfl, ic, keep=True)
                seconds = time.perf_counter() - t0
                dt = t[1] - t[0]
                E = discrete_energy(x, U, dt)
                rows.append({
                    "ic": ic, "cfl": cfl, "nx": nx, "nt": len(t) - 1,
                    "dx": x[1] - x[0], "dt": dt,
                    "rel_l2": space_time_error(x, t, U, ic),
                    "seconds": seconds,
                    "energy_ratio": float(E[0] / e_exact),
                    "energy_drift": float(np.abs(E - E[0]).max() / E[0]),
                })
    if write:
        write_csv(SWEEP_CSV, SWEEP_FIELDS, rows)
    return rows


# -- B/C. the order, and where it is measured ---------------------------------

def order_study(ics=W.ICS, cfl=0.5, times=ORDER_TIMES, grids=ORDER_GRIDS,
                write=True):
    """Convergence order at a single time, normalized by a constant.

    The denominator is ``||f||`` -- the norm of the *initial displacement*, the
    same for every row -- rather than ``||u(., t)||``. That is not fussiness:
    the sine's exact solution is identically zero at t = 0.5, so the natural
    relative error there divides by zero, and near such a time it divides by
    something small and reports a huge error for a small absolute one. A fixed
    denominator makes the rows at different t comparable and lets the order be
    read off cleanly.
    """
    rows = []
    for ic in ics:
        for t_end in times:
            prev = None
            for nx in grids:
                x, t, u = solve(nx, cfl, ic, t_end=t_end)
                ref = np.linalg.norm(W.f0(x, ic))  # same shape as u
                err = float(np.linalg.norm(u - W.wave_exact(x, t[-1], ic)) / ref)
                order = "" if prev is None else f"{np.log2(prev / err):.2f}"
                rows.append({"ic": ic, "t_end": t_end, "nx": nx,
                             "rel_l2_fixed_norm": err, "order": order})
                prev = err
    if write:
        write_csv(ORDER_CSV, ORDER_FIELDS, rows)
    return rows


def undefined_at_a_node(ic="sine", t=0.5):
    """How small the exact solution's norm gets at a turning time.

    Returns ``||u(., t)|| / ||f||``. For the sine at t = 0.5 this is ~0, which
    is why Study B normalizes by a constant. Reported rather than asserted.
    """
    x = np.linspace(*W.X_RANGE, 801)
    return float(np.linalg.norm(W.wave_exact(x, t, ic))
                 / np.linalg.norm(W.f0(x, ic)))


# -- D. against the PINN -------------------------------------------------------

def pinn_reference():
    """Sec. 15's committed cells: mean rel L2 and mean train seconds per IC."""
    try:
        rows = read_csv(W.CELLS_CSV)
    except FileNotFoundError:
        return None
    out = {}
    for ic in W.ICS:
        cells = [r for r in rows if r["ic"] == ic]
        if not cells:
            continue
        out[ic] = {
            "rel_l2": float(np.mean([float(r["rel_l2"]) for r in cells])),
            "seconds": float(np.mean([float(r["train_seconds"]) for r in cells])),
            "energy_ratio": float(np.mean([float(r["energy_ratio"])
                                           for r in cells])),
            "n_seeds": len(cells),
        }
    return out


def head_to_head(rows, pinn, cfl=0.5):
    """The coarsest r < 1 grid that already beats the PINN, and by how much.

    r < 1 on purpose: at r = 1 the scheme is exact and the ratio is a statement
    about the 1D wave equation rather than about the two methods, so the honest
    comparison is the one where the mesh is doing ordinary approximate work.
    """
    out = []
    for ic in W.ICS:
        if pinn is None or ic not in pinn:
            continue
        cand = sorted((r for r in rows if r["ic"] == ic and r["cfl"] == cfl),
                      key=lambda r: r["nx"])
        beat = next((r for r in cand if r["rel_l2"] < pinn[ic]["rel_l2"]), None)
        row = {"ic": ic, "PINN rel_l2": pinn[ic]["rel_l2"],
               "PINN seconds": pinn[ic]["seconds"]}
        if beat is None:
            row.update({"mesh nx": "none", "mesh rel_l2": float("nan"),
                        "mesh seconds": float("nan"), "speedup": float("nan"),
                        "accuracy x": float("nan")})
        else:
            row.update({
                "mesh nx": beat["nx"], "mesh rel_l2": beat["rel_l2"],
                "mesh seconds": beat["seconds"],
                "speedup": pinn[ic]["seconds"] / beat["seconds"],
                "accuracy x": pinn[ic]["rel_l2"] / beat["rel_l2"],
            })
        out.append(row)
    return out


# -- figure -------------------------------------------------------------------

def make_figure(rows, order_rows, pinn):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    def tidy_log_x(ax):
        """Decade ticks only. The default log locator labels 2, 3, 4 x 10^k on a
        narrow range and the labels collide into unreadable mush."""
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    colors = {"sine": "#1f77b4", "pluck": "#a11"}

    ax = axes[0]
    for ic in W.ICS:
        for cfl, ls in zip(CFLS, ("-", "--", "-.", ":")):
            sub = sorted((r for r in rows if r["ic"] == ic and r["cfl"] == cfl),
                         key=lambda r: r["nx"])
            ax.loglog([r["dx"] for r in sub], [max(r["rel_l2"], 1e-16) for r in sub],
                      ls, color=colors[ic], lw=1.2, label=f"{ic}, r={cfl}")
    if pinn:
        for ic in W.ICS:
            if ic in pinn:
                ax.axhline(pinn[ic]["rel_l2"], color=colors[ic], lw=0.8, alpha=0.5)
                ax.text(0.012, pinn[ic]["rel_l2"] * 1.25, f"PINN, {ic}",
                        fontsize=6.5, color=colors[ic])
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel("relative $L^2$ over the space-time grid")
    exact_max = max(r["rel_l2"] for r in rows if r["cfl"] == 1.0)
    ax.set_title(rf"A. every $r=1$ cell is exact to {exact_max:.0e}", fontsize=9)
    tidy_log_x(ax)
    ax.legend(fontsize=5.8, ncol=2)

    ax = axes[1]
    for ic in W.ICS:
        for t_end, mk in zip(ORDER_TIMES, ("o-", "s--", "^:")):
            sub = [r for r in order_rows if r["ic"] == ic and r["t_end"] == t_end]
            ax.loglog([1.0 / (r["nx"] - 1) for r in sub],
                      [r["rel_l2_fixed_norm"] for r in sub], mk,
                      color=colors[ic], ms=3.5, lw=1.1, label=f"{ic}, t={t_end}")
    fits = {(r["ic"], r["t_end"]): r["order"] for r in order_rows if r["order"]}
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"error at one time, $\|\cdot\|/\|f\|$")
    ax.set_title(
        rf"B. sine reads order {fits[('sine', 0.7)]} at $t=0.7$ and "
        rf"{fits[('sine', 2.0)]} at $t=2$", fontsize=9)
    tidy_log_x(ax)
    ax.legend(fontsize=6)

    ax = axes[2]
    for ic in W.ICS:
        sub = sorted((r for r in rows if r["ic"] == ic and r["cfl"] == 0.5),
                     key=lambda r: r["seconds"])
        ax.loglog([r["seconds"] for r in sub], [r["rel_l2"] for r in sub],
                  "o-", color=colors[ic], ms=4, label=f"mesh, {ic}")
        if pinn and ic in pinn:
            ax.plot(pinn[ic]["seconds"], pinn[ic]["rel_l2"], "*",
                    color=colors[ic], ms=13, label=f"PINN, {ic}")
    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("relative $L^2$")
    ax.set_title(r"C. accuracy against cost, $r = 0.5$", fontsize=9)
    ax.legend(fontsize=6.5)

    fig.suptitle("The classical baseline Sec. 15 did not have", y=1.02)
    fig.tight_layout()
    savefig(fig, "wave_leapfrog.png")


def figures_from_committed():
    """Rebuild the figure from the committed CSVs rather than re-solving.

    One axis of panel C is wall-clock, so a rerun on another machine moves those
    points; the committed CSV is the measurement. The accuracy columns are
    deterministic and would come back identical, which is why the whole thing
    replays rather than half of it.
    """
    rows = [{"ic": r["ic"], "cfl": float(r["cfl"]), "nx": int(r["nx"]),
             "nt": int(r["nt"]), "dx": float(r["dx"]), "dt": float(r["dt"]),
             "rel_l2": float(r["rel_l2"]), "seconds": float(r["seconds"]),
             "energy_ratio": float(r["energy_ratio"]),
             "energy_drift": float(r["energy_drift"])}
            for r in read_csv(SWEEP_CSV)]
    order_rows = [{"ic": r["ic"], "t_end": float(r["t_end"]), "nx": int(r["nx"]),
                   "rel_l2_fixed_norm": float(r["rel_l2_fixed_norm"]),
                   "order": r["order"]}
                  for r in read_csv(ORDER_CSV)]
    make_figure(rows, order_rows, pinn_reference())


# -- reporting -----------------------------------------------------------------

def _table(rows, cols, widths=None):
    widths = widths or {}
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3e}" if (v and abs(v) < 1e-3) else f"{v:.4f}"
        return str(v)
    w = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    w.update(widths)
    print("  ".join(c.rjust(w[c]) for c in cols))
    print("-" * (sum(w.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(fmt(r[c]).rjust(w[c]) for c in cols))


def report(rows, order_rows, pinn):
    print("=" * 78)
    print("Sec. 17: leapfrog on the wave equation -- the baseline Sec. 15 owed")
    print("=" * 78)

    print("\n=== A. accuracy on Sec. 15's space-time metric (a zero field scores 1.0) ===")
    for ic in W.ICS:
        print(f"\n  {ic}:")
        _table([r for r in rows if r["ic"] == ic],
               ["cfl", "nx", "nt", "rel_l2", "seconds", "energy_ratio",
                "energy_drift"])
    exact_rows = [r for r in rows if r["cfl"] == 1.0]
    print(f"\n  every r = 1 cell is exact to {max(r['rel_l2'] for r in exact_rows):.1e}, "
          "for both initial conditions and every grid --\n  the update is d'Alembert "
          "on the characteristics. That is a fact about the 1D\n  constant-coefficient "
          "wave equation, not a mesh-versus-network number.")

    print("\n=== B/C. observed order, and the two ways of measuring it wrongly ===")
    for ic in W.ICS:
        print(f"\n  {ic}:")
        _table([r for r in order_rows if r["ic"] == ic],
               ["t_end", "nx", "rel_l2_fixed_norm", "order"])
    z = undefined_at_a_node("sine", 0.5)
    print(f"\n  the sine is a standing wave of period {2.0 / W.C:.0f}: at t = 0.5 the "
          f"exact solution has\n  norm {z:.1e} of the initial displacement's, so a "
          "*relative* error there has no\n  denominator; and at t = 1 and t = 2 it "
          "sits at a turning point of cos(pi c t),\n  where a phase error enters "
          "quadratically and the measured order reads 4 instead\n  of 2. Sec. 15's own "
          "window ends at t = 2, which is one of those points.")

    h2h = head_to_head(rows, pinn)
    if h2h:
        print("\n=== D. against Sec. 15's PINN, r = 0.5 (the mesh doing ordinary work) ===")
        _table(h2h, ["ic", "PINN rel_l2", "PINN seconds", "mesh nx", "mesh rel_l2",
                     "mesh seconds", "accuracy x", "speedup"])
        print("\n  'mesh nx' is the coarsest grid in the sweep that already beats the "
              "network.")
    else:
        print("\n=== D. skipped: logs/wave_cells.csv not found ===")

    print("\n=== E. energy ===")
    drift = max(r["energy_drift"] for r in rows if r["cfl"] < 1.0)
    print(f"  the discrete energy is conserved to {drift:.1e} at every r < 1 cell, by "
          "the\n  staggered identity in discrete_energy(). Its value against the exact "
          "conserved\n  energy is the 'energy_ratio' column above; Sec. 15's network "
          "carries 0.766 of it\n  on the pluck.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true",
                    help="rebuild the figure from committed CSVs and exit")
    args = ap.parse_args()
    if args.figures:
        figures_from_committed()
        return
    rows = sweep()
    order_rows = order_study()
    pinn = pinn_reference()
    report(rows, order_rows, pinn)
    make_figure(rows, order_rows, pinn)


if __name__ == "__main__":
    main()
