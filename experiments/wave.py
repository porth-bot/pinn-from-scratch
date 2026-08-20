"""The wave equation, a d'Alembert ground truth, and a kink the network cannot see.

Every PDE in this repo so far is parabolic (heat, in one dimension and in d) or
first order with viscosity (Burgers) or elliptic-in-time (the HJB of Sec. 13).
All of them smooth their initial data. The wave equation does not: it *transports*
it, exactly, forever, and that changes what a PINN is being asked to do.

The problem
-----------
A string of unit length with fixed ends:

    u_tt = c^2 u_xx,    u(0,t) = u(1,t) = 0,
    u(x,0) = f(x),      u_t(x,0) = 0.

**The ground truth is d'Alembert, not a Fourier series, and that is the point.**
With zero initial velocity the solution is

    u(x,t) = 1/2 [ F(x - c t) + F(x + c t) ],

where F is the **odd 2-periodic extension** of f -- odd about x = 0 and about
x = 1, which is exactly what makes the two fixed ends hold for all time (the
reflected wave arrives inverted). This is closed form for *any* f, including an
f with a corner, which is why the second initial condition below is a plucked
string rather than a sine. The Fourier sine series is exact too, but only as an
infinite sum, and for a plucked string its coefficients decay like 1/k^2 -- a
truncation is a different function with different derivatives, so it is used
here as an *independent check* of d'Alembert (:func:`fourier_reference`) and
never as the reference itself.

Two initial conditions, and the reason for each
-----------------------------------------------
- ``sine``: ``f(x) = sin(pi x)``. Smooth, a single mode, a standing wave. This is
  the control: if the PINN cannot do this, nothing below means anything.
- ``pluck``: a triangular displacement, the string pulled aside at x = 0.3 and
  released. f is continuous with a corner, so u_xx is a delta and **the PDE holds
  only weakly**. The residual a PINN minimizes is the strong form, which does not
  exist at the corner; a network with a tanh activation is smooth, so it cannot
  represent the kink either. What it does instead is the measurement.

What is measured
----------------
1. Relative L2 against d'Alembert on a grid, for both initial conditions.
2. The **energy** the network carries against the exact conserved energy. The
   continuous problem conserves ``E = int (u_t^2 + c^2 u_x^2)/2`` exactly, and
   nothing in the training objective knows that, so it is a diagnostic the loss
   cannot game -- the analogue of the calibration checks in
   ``gp-from-scratch``. A network that has quietly decayed toward the trivial
   solution loses energy, and this says so with a number.
3. **Whether the solution is the trivial one.** ``u = 0`` satisfies the residual,
   both boundary conditions *and* the zero initial velocity condition exactly, so
   the entire problem is carried by one loss term (the initial displacement) --
   the same structure Sec. 14 measured for the high-dimensional heat problem,
   arrived at from the other direction. The ``rel_l2`` of a zero network is
   exactly 1.0 by construction, reported as a baseline in every table.

Run:  python experiments/wave.py --check    # d'Alembert vs Fourier, no training
      python experiments/wave.py --train    # both ICs, 3 seeds each (~25 min)
      python experiments/wave.py --figures  # replay from committed CSVs
      python experiments/wave.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import read_csv, savefig, write_csv
from pinn import derivatives as D
from pinn.model import MLP, set_seed

C = 1.0                      # wave speed
X_RANGE = (0.0, 1.0)
T_RANGE = (0.0, 2.0)         # two crossings of the string at c = 1
PLUCK_X0 = 0.3               # where the string is pulled aside
PLUCK_H = 1.0                # by how much

ICS = ("sine", "pluck")
SEEDS = (0, 1, 2)
BUDGET = dict(n_interior=4000, n_ic=400, n_bc=400, width=128, depth=4,
              steps=5000, lr=1e-3)

CELLS_CSV = "wave_cells.csv"
TRACE_CSV = "wave_trace.csv"
CHECK_CSV = "wave_check.csv"

CELL_FIELDS = ["ic", "seed", "rel_l2", "rel_l2_final", "energy_ratio",
               "energy_exact", "best_step", "best_loss", "final_loss", "loss_r",
               "loss_ic", "loss_vel", "loss_bc", "kink_error", "smooth_error",
               "train_seconds"]
TRACE_FIELDS = ["ic", "seed", "step", "loss", "loss_r", "loss_ic", "loss_vel",
                "loss_bc", "rel_l2", "energy_ratio", "train_seconds"]


# ---------------------------------------------------------------------------
# Initial data and its odd periodic extension
# ---------------------------------------------------------------------------
def f0(x, ic="sine"):
    """Initial displacement on [0, 1]. Outside it, use :func:`odd_extension`."""
    x = np.asarray(x, dtype=float)
    if ic == "sine":
        return np.sin(np.pi * x)
    if ic == "pluck":
        # A triangle peaking at PLUCK_X0: linear up, linear down, zero at both
        # ends. Continuous, with a corner at the peak -- which is the whole
        # reason this initial condition is here.
        return PLUCK_H * np.where(x <= PLUCK_X0,
                                  x / PLUCK_X0,
                                  (1.0 - x) / (1.0 - PLUCK_X0))
    raise ValueError(f"unknown initial condition {ic!r}")


def odd_extension(y, ic="sine"):
    """F(y): the odd 2-periodic extension of ``f0`` to the whole line.

    Fold ``y`` into ``[-1, 1)`` by periodicity, then use ``F(-s) = -F(s)``. The
    two symmetries are what enforce the boundary conditions: oddness about 0
    gives ``u(0,t) = 0`` for every t, and 2-periodicity together with oddness
    gives ``F(2 - s) = F(-s) = -F(s)``, i.e. oddness about x = 1, which gives
    ``u(1,t) = 0``. Both are checked directly in ``tests/test_wave.py`` rather
    than argued for here alone.
    """
    y = np.asarray(y, dtype=float)
    s = np.mod(y + 1.0, 2.0) - 1.0          # into [-1, 1)
    return np.where(s >= 0, f0(np.abs(s), ic), -f0(np.abs(s), ic))


def wave_exact(x, t, ic="sine", c=C):
    """d'Alembert: ``u = [F(x - ct) + F(x + ct)] / 2``. Exact at every (x, t)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    return 0.5 * (odd_extension(x - c * t, ic) + odd_extension(x + c * t, ic))


def fourier_coefficients(ic="sine", n_modes=400):
    """Sine-series coefficients ``b_k`` of ``f0``, in closed form.

    ``sine`` is one mode. For the triangle, integrating by parts twice gives

        b_k = 2 h sin(k pi x0) / (k^2 pi^2 x0 (1 - x0)),

    which decays like 1/k^2 -- the reason a truncation is a genuinely different
    function and cannot be the ground truth for a PINN whose loss contains
    second derivatives.
    """
    k = np.arange(1, n_modes + 1)
    if ic == "sine":
        b = np.zeros(n_modes)
        b[0] = 1.0
        return k, b
    if ic == "pluck":
        x0 = PLUCK_X0
        b = (2 * PLUCK_H * np.sin(k * np.pi * x0)
             / (k ** 2 * np.pi ** 2 * x0 * (1 - x0)))
        return k, b
    raise ValueError(f"unknown initial condition {ic!r}")


def fourier_reference(x, t, ic="sine", c=C, n_modes=400):
    """``sum_k b_k sin(k pi x) cos(k pi c t)`` -- an independent second opinion.

    Separation of variables, written out with no code shared with
    :func:`wave_exact`. The two must agree wherever the series converges
    quickly, which is everywhere for ``sine`` and away from the travelling
    corners for ``pluck``. Used to check d'Alembert, never to replace it.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    k, b = fourier_coefficients(ic, n_modes)
    return (b * np.sin(np.outer(x, k) * np.pi)
            * np.cos(np.outer(t, k) * np.pi * c)).sum(axis=1)


def exact_energy(ic="sine", c=C):
    """The conserved energy ``int_0^1 (u_t^2 + c^2 u_x^2)/2 dx``, in closed form.

    At t = 0 the velocity is zero, so ``E = (c^2/2) int f'(x)^2 dx``, and the
    wave equation conserves it for all time. For ``sine`` that is
    ``c^2 pi^2 / 4``; for the triangle ``f'`` is piecewise constant, giving
    ``c^2 h^2 / (2 x0 (1 - x0))``. Both are exact, which is what makes the
    measured ratio a real diagnostic rather than a comparison of two estimates.
    """
    if ic == "sine":
        return float(c ** 2 * np.pi ** 2 / 4)
    if ic == "pluck":
        return float(c ** 2 * PLUCK_H ** 2 / (2 * PLUCK_X0 * (1 - PLUCK_X0)))
    raise ValueError(f"unknown initial condition {ic!r}")


def energy_by_modes(ic="pluck", n_modes=1, c=C):
    """Energy carried by the first ``n_modes`` of the sine series, as a fraction.

    ``u = sum_k b_k sin(k pi x) cos(k pi c t)`` gives
    ``E_K = (c^2 pi^2/4) sum_{k<=K} b_k^2 k^2``, and the full sum reproduces
    :func:`exact_energy` (checked to six digits, and in the tests). For the
    plucked string ``b_k ~ 1/k^2``, so ``b_k^2 k^2 ~ 1/k^2`` and the tail decays
    like 1/K -- the energy is spread over many modes, slowly.

    This exists to turn the measured energy ratio into something with units a
    reader already has. A network holding 77% of the energy is holding what the
    first one to two modes hold, which is Sec. 3's spectral bias appearing on a
    hyperbolic problem.
    """
    k, b = fourier_coefficients(ic, max(int(n_modes), 1))
    return float(c ** 2 * np.pi ** 2 / 4 * np.sum(b ** 2 * k ** 2)
                 / exact_energy(ic, c))


def modes_bracketing(fraction, ic="pluck", max_modes=2000, c=C):
    """The K with ``E_K <= fraction < E_{K+1}``, as a bracket rather than a fit.

    Returns ``(K, E_K, E_{K+1})``. A bracket rather than an interpolated
    fractional mode count, because the quantity is a step function of an integer
    and quoting "1.6 modes" would invent a resolution the measurement does not
    have.
    """
    cum = [energy_by_modes(ic, n, c) for n in range(1, max_modes + 1)]
    for i, e in enumerate(cum):
        if e > fraction:
            return (i, cum[i - 1] if i else 0.0, e)
    return (max_modes, cum[-1], float("nan"))


# ---------------------------------------------------------------------------
# The residual
# ---------------------------------------------------------------------------
def wave_residual(u, coords, c=C):
    """``u_tt - c^2 u_xx`` at grad-enabled ``coords`` of shape (N, 2)."""
    return D.u_tt(u, coords) - c ** 2 * D.u_xx(u, coords)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _grid(nx=201, nt=201):
    x = np.linspace(*X_RANGE, nx)
    t = np.linspace(*T_RANGE, nt)
    return np.meshgrid(x, t, indexing="ij")


def predict(model, XX, TT):
    coords = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1),
                          dtype=torch.float32)
    with torch.no_grad():
        return model(coords).numpy().reshape(XX.shape)


def rel_l2_error(model, ic="sine", nx=201, nt=201):
    """Relative L2 on a grid. A network that outputs zero scores exactly 1.0."""
    XX, TT = _grid(nx, nt)
    exact = wave_exact(XX, TT, ic)
    pred = predict(model, XX, TT)
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def energy_ratio(model, ic="sine", nx=401, t=0.5, c=C):
    """Network energy at time ``t``, divided by the exact conserved energy.

    ``int (u_t^2 + c^2 u_x^2)/2 dx`` by the trapezoid rule, with both
    derivatives taken by autograd rather than by differencing the grid, so the
    number measures the network and not the quadrature stencil. Nothing in the
    training loss references energy, so this is a diagnostic the optimizer
    cannot have targeted.
    """
    x = torch.linspace(*X_RANGE, nx, dtype=torch.float32).unsqueeze(1)
    tt = torch.full_like(x, float(t))
    coords = torch.cat([x, tt], dim=1).requires_grad_(True)
    u = model(coords)
    ux = D.u_x(u, coords)
    ut = D.u_t(u, coords)
    dens = 0.5 * (ut ** 2 + c ** 2 * ux ** 2)
    e = float(torch.trapezoid(dens.squeeze(1).detach(),
                              x.squeeze(1).detach()))
    return e / exact_energy(ic, c)


def kink_split(model, ic="pluck", nx=401, nt=401, band=0.05, c=C):
    """Error inside vs outside a band around the travelling corners.

    The pluck's two corners start at ``x0`` and travel at +-c, reflecting at the
    ends; a point is "in the kink band" if it is within ``band`` of either
    characteristic through the initial corner, in the odd-periodic sense. The
    split exists because the section's claim is about *where* the error lives,
    and a single relative L2 cannot support that claim.

    Returns ``(kink_rms, smooth_rms)``, both relative to the exact rms of u.
    """
    XX, TT = _grid(nx, nt)
    err = predict(model, XX, TT) - wave_exact(XX, TT, ic)
    scale = np.sqrt(np.mean(wave_exact(XX, TT, ic) ** 2))
    # distance from x to the nearest image of the corner travelling either way
    d1 = _periodic_distance(XX - c * TT, PLUCK_X0)
    d2 = _periodic_distance(XX + c * TT, PLUCK_X0)
    near = (np.minimum(d1, d2) <= band)
    kink = np.sqrt(np.mean(err[near] ** 2)) / scale if near.any() else float("nan")
    smooth = (np.sqrt(np.mean(err[~near] ** 2)) / scale if (~near).any()
              else float("nan"))
    return float(kink), float(smooth)


def _periodic_distance(y, x0):
    """Distance from y to +-x0 modulo 2 -- where the odd extension has corners.

    ``odd_extension`` has a corner wherever its argument equals ``x0`` or
    ``-x0`` up to a multiple of 2, because the triangle's peak is reflected by
    both symmetries.
    """
    a = np.abs(np.mod(y - x0 + 1.0, 2.0) - 1.0)
    b = np.abs(np.mod(y + x0 + 1.0, 2.0) - 1.0)
    return np.minimum(a, b)


def _points(n_interior, n_ic, n_bc, gen):
    """Interior (grad-enabled), initial-slice and boundary points."""
    x0, x1 = X_RANGE
    t0, t1 = T_RANGE
    xi = x0 + (x1 - x0) * torch.rand(n_interior, 1, generator=gen)
    ti = t0 + (t1 - t0) * torch.rand(n_interior, 1, generator=gen)
    interior = torch.cat([xi, ti], dim=1).requires_grad_(True)

    xic = x0 + (x1 - x0) * torch.rand(n_ic, 1, generator=gen)
    ic_pts = torch.cat([xic, torch.full_like(xic, t0)], dim=1)
    ic_pts.requires_grad_(True)     # the velocity condition needs u_t here

    tb = t0 + (t1 - t0) * torch.rand(n_bc, 1, generator=gen)
    side = torch.randint(0, 2, (n_bc, 1), generator=gen).to(tb.dtype)
    bc = torch.cat([x0 + (x1 - x0) * side, tb], dim=1)
    return interior, ic_pts, bc


def train(ic="sine", seed=0, n_interior=None, n_ic=None, n_bc=None, width=None,
          depth=None, steps=None, lr=None, w_ic=1.0, w_vel=1.0, w_bc=1.0,
          eval_every=250, verbose=False, select="best_loss"):
    """Adam on the four-term wave objective; return ``(model, history, best)``.

    Four terms rather than Sec. 1's three, because a second-order-in-time
    equation needs *two* initial conditions: the displacement ``u(x,0) = f`` and
    the velocity ``u_t(x,0) = 0``. Dropping the velocity term does not make the
    problem easier, it makes it ill-posed -- there is a one-parameter family of
    solutions with the same displacement -- and the term is weighted the same as
    the others rather than tuned, following Sec. 9's finding that the default
    weight is inside the flat optimum on this problem class.

    Selection is on the lowest *training* loss, as everywhere else in this repo:
    the loss contains no ground truth, so nothing about d'Alembert enters the
    choice of iterate.
    """
    b = dict(BUDGET)
    for name, value in (("n_interior", n_interior), ("n_ic", n_ic),
                        ("n_bc", n_bc), ("width", width), ("depth", depth),
                        ("steps", steps), ("lr", lr)):
        if value is not None:
            b[name] = value

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    interior, ic_pts, bc = _points(b["n_interior"], b["n_ic"], b["n_bc"], gen)
    ic_target = torch.tensor(f0(ic_pts[:, 0].detach().numpy(), ic),
                             dtype=torch.float32).unsqueeze(1)
    bc_target = torch.zeros(bc.shape[0], 1)

    model = MLP(in_dim=2, out_dim=1, width=b["width"], depth=b["depth"],
                activation="tanh")
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])

    history = []
    best = dict(step=-1, loss=float("inf"))
    best_state = None
    train_seconds = 0.0
    for step in range(b["steps"] + 1):
        t_step = time.perf_counter()
        opt.zero_grad()
        r = wave_residual(model(interior), interior)
        loss_r = torch.mean(r ** 2)
        u_ic = model(ic_pts)
        loss_ic = torch.mean((u_ic - ic_target) ** 2)
        loss_vel = torch.mean(D.u_t(u_ic, ic_pts) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_ic * loss_ic + w_vel * loss_vel + w_bc * loss_bc
        loss.backward()

        value = float(loss.detach())
        if value < best["loss"]:
            best = dict(step=step, loss=value)
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        opt.step()
        train_seconds += time.perf_counter() - t_step

        if step % eval_every == 0 or step == b["steps"]:
            err = rel_l2_error(model, ic)
            er = energy_ratio(model, ic)
            history.append((step, value, float(loss_r.detach()),
                            float(loss_ic.detach()), float(loss_vel.detach()),
                            float(loss_bc.detach()), err, er, train_seconds))
            if verbose:
                print(f"  step {step:5d}  loss {value:.3e}  relL2 {err:.4f}  "
                      f"E/E* {er:.3f}", flush=True)

    # The last update's own output has never been scored; let it compete.
    r = wave_residual(model(interior), interior)
    u_ic = model(ic_pts)
    final_loss = float((torch.mean(r ** 2)
                        + w_ic * torch.mean((u_ic - ic_target) ** 2)
                        + w_vel * torch.mean(D.u_t(u_ic, ic_pts) ** 2)
                        + w_bc * torch.mean((model(bc) - bc_target) ** 2)).detach())
    if final_loss < best["loss"]:
        best = dict(step=b["steps"] + 1, loss=final_loss)
        best_state = None
    best["final_loss"] = final_loss
    best["train_seconds"] = train_seconds
    best["state_dict"] = best_state
    if select == "best_loss" and best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


def run_cell(ic, seed, verbose=True, **kw):
    if verbose:
        print(f"  {ic:6s} seed={seed}", flush=True)
    model, history, best = train(ic=ic, seed=seed, select="final",
                                 verbose=verbose, **kw)
    rel_final = rel_l2_error(model, ic)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    rel = rel_l2_error(model, ic)
    er = energy_ratio(model, ic)
    kink, smooth = kink_split(model, ic)
    step, loss, loss_r, loss_ic, loss_vel, loss_bc, _, _, ts = history[-1]
    row = {
        "ic": ic, "seed": seed,
        "rel_l2": f"{rel:.6e}", "rel_l2_final": f"{rel_final:.6e}",
        "energy_ratio": f"{er:.6f}", "energy_exact": f"{exact_energy(ic):.6f}",
        "best_step": best["step"], "best_loss": f"{best['loss']:.6e}",
        "final_loss": f"{best['final_loss']:.6e}",
        "loss_r": f"{loss_r:.6e}", "loss_ic": f"{loss_ic:.6e}",
        "loss_vel": f"{loss_vel:.6e}", "loss_bc": f"{loss_bc:.6e}",
        "kink_error": f"{kink:.6e}", "smooth_error": f"{smooth:.6e}",
        "train_seconds": f"{ts:.2f}",
    }
    if verbose:
        print(f"         rel L2 {rel:.4e}  (final {rel_final:.4e})  "
              f"E/E* {er:.3f}  kink {kink:.3e} vs smooth {smooth:.3e}  "
              f"{ts:.0f}s", flush=True)
    return row, history


def sweep(ics=ICS, seeds=SEEDS, write=True, verbose=True, **kw):
    rows, traces = [], []
    for ic in ics:
        for seed in seeds:
            row, history = run_cell(ic, seed, verbose=verbose, **kw)
            rows.append(row)
            for (step, loss, lr_, li, lv, lb, err, er, ts) in history:
                traces.append({"ic": ic, "seed": seed, "step": step,
                               "loss": f"{loss:.6e}", "loss_r": f"{lr_:.6e}",
                               "loss_ic": f"{li:.6e}", "loss_vel": f"{lv:.6e}",
                               "loss_bc": f"{lb:.6e}", "rel_l2": f"{err:.6e}",
                               "energy_ratio": f"{er:.6f}",
                               "train_seconds": f"{ts:.4f}"})
            if write:
                write_csv(CELLS_CSV, CELL_FIELDS, rows)
                write_csv(TRACE_CSV, TRACE_FIELDS, traces)
    return rows, traces


# ---------------------------------------------------------------------------
# The ground truth, checked before anything is trained against it
# ---------------------------------------------------------------------------
def check(n_modes=(50, 200, 800), verbose=True):
    """d'Alembert against the Fourier series, and both against the PDE.

    Three things, none of which the training run could reveal:

    - **Boundary and initial conditions**, from the extension's symmetries:
      ``u(0,t) = u(1,t) = 0`` at every t, ``u(x,0) = f(x)``, and ``u_t(x,0) = 0``
      by central differences.
    - **d'Alembert against separation of variables**, as the mode count grows.
      For ``sine`` they agree to machine precision at every count (one mode). For
      ``pluck`` the series converges like 1/k^2 and the max gap shrinks with
      ``n_modes`` while the *rms* gap shrinks faster -- Gibbs at the corner is
      the whole difference, which is the point rather than an annoyance.
    - **The PDE itself**, by central differences away from the corners, since
      the strong form does not exist at them.
    """
    rows = []
    for ic in ICS:
        x = np.linspace(*X_RANGE, 501)
        t = np.linspace(*T_RANGE, 7)
        XX, TT = np.meshgrid(x, t, indexing="ij")
        exact = wave_exact(XX, TT, ic)

        bc0 = float(np.max(np.abs(wave_exact(np.zeros_like(t), t, ic))))
        bc1 = float(np.max(np.abs(wave_exact(np.ones_like(t), t, ic))))
        ic_err = float(np.max(np.abs(wave_exact(x, np.zeros_like(x), ic)
                                     - f0(x, ic))))
        h = 1e-5
        vel = float(np.max(np.abs(
            (wave_exact(x, np.full_like(x, h), ic)
             - wave_exact(x, np.full_like(x, -h), ic)) / (2 * h))))

        for n in n_modes:
            ref = np.stack([fourier_reference(x, np.full_like(x, tv), ic, n_modes=n)
                            for tv in t], axis=1)
            gap = np.abs(ref - exact)
            rows.append({
                "ic": ic, "n_modes": n,
                "max_gap": f"{gap.max():.6e}",
                "rms_gap": f"{np.sqrt(np.mean(gap ** 2)):.6e}",
                "bc_left": f"{bc0:.3e}", "bc_right": f"{bc1:.3e}",
                "ic_error": f"{ic_err:.3e}", "initial_velocity": f"{vel:.3e}",
                "residual_fd": f"{_fd_residual(ic):.6e}",
                "energy_exact": f"{exact_energy(ic):.6f}",
            })
    if verbose:
        print(f"{'ic':>6} {'modes':>6} {'max gap':>11} {'rms gap':>11} "
              f"{'u(0,t)':>9} {'u(1,t)':>9} {'u(x,0)-f':>10} {'u_t(x,0)':>10} "
              f"{'FD resid':>10}")
        for r in rows:
            print(f"{r['ic']:>6} {r['n_modes']:>6} {float(r['max_gap']):>11.3e} "
                  f"{float(r['rms_gap']):>11.3e} {float(r['bc_left']):>9.1e} "
                  f"{float(r['bc_right']):>9.1e} {float(r['ic_error']):>10.1e} "
                  f"{float(r['initial_velocity']):>10.1e} "
                  f"{float(r['residual_fd']):>10.2e}")
        print("\n  (the pluck's max gap is Gibbs at the travelling corner and "
              "shrinks slowly;\n   its rms gap shrinks like the series does. "
              "d'Alembert is exact at every count.)")
    return rows


def _fd_residual(ic, h=1e-4):
    """``max |u_tt - c^2 u_xx|`` by central differences, away from the corners.

    Points within ``4h`` of a travelling corner are excluded, because the second
    derivative there is a delta and a difference stencil across it reports the
    mesh rather than the solution. Excluding them is the honest thing and it is
    stated in the log's column name.
    """
    x = np.linspace(0.05, 0.95, 401)
    t = np.linspace(0.1, 1.9, 21)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    u = lambda a, b: wave_exact(a, b, ic)
    utt = (u(XX, TT + h) - 2 * u(XX, TT) + u(XX, TT - h)) / h ** 2
    uxx = (u(XX + h, TT) - 2 * u(XX, TT) + u(XX - h, TT)) / h ** 2
    res = np.abs(utt - C ** 2 * uxx)
    if ic == "pluck":
        near = np.minimum(_periodic_distance(XX - C * TT, PLUCK_X0),
                          _periodic_distance(XX + C * TT, PLUCK_X0)) <= 4 * h
        res = res[~near]
    return float(res.max())


# ---------------------------------------------------------------------------
# Summary and figure
# ---------------------------------------------------------------------------
def summarize(rows):
    out = []
    for ic in ICS:
        cells = [r for r in rows if r["ic"] == ic]
        if not cells:
            continue
        errs = np.array([float(r["rel_l2"]) for r in cells])
        ers = np.array([float(r["energy_ratio"]) for r in cells])
        out.append({
            "ic": ic, "n_seeds": len(cells),
            "mean": f"{errs.mean():.6e}", "min": f"{errs.min():.6e}",
            "max": f"{errs.max():.6e}",
            "spread": f"{errs.max() / errs.min():.3f}",
            "mean_energy_ratio": f"{ers.mean():.4f}",
            "mean_kink": f"{np.mean([float(r['kink_error']) for r in cells]):.4e}",
            "mean_smooth": f"{np.mean([float(r['smooth_error']) for r in cells]):.4e}",
        })
    return out


def report(rows):
    print("\n" + "=" * 82)
    print(f"Wave equation, {BUDGET['steps']} Adam steps "
          f"(a network outputting zero scores exactly 1.0)")
    print("=" * 82)
    print(f"{'IC':>6} {'seeds':>6} {'mean relL2':>12} {'min':>11} {'max':>11} "
          f"{'E/E*':>7} {'kink':>10} {'smooth':>10}")
    for s in summarize(rows):
        print(f"{s['ic']:>6} {s['n_seeds']:>6} {float(s['mean']):>12.4e} "
              f"{float(s['min']):>11.4e} {float(s['max']):>11.4e} "
              f"{float(s['mean_energy_ratio']):>7.3f} "
              f"{float(s['mean_kink']):>10.3e} {float(s['mean_smooth']):>10.3e}")
    for s in summarize(rows):
        frac = float(s["mean_energy_ratio"])
        K, lo, hi = modes_bracketing(frac, s["ic"])
        if K == 0:
            print(f"  {s['ic']:>6}: energy {frac:.3f} of exact -- short of what "
                  f"the first mode alone carries ({hi:.3f})")
        else:
            print(f"  {s['ic']:>6}: energy {frac:.3f} of exact -- between what "
                  f"the first {K} mode{'s' if K != 1 else ''} carr{"y" if K != 1 else "ies"} ({lo:.3f}) "
                  f"and the first {K + 1} ({hi:.3f})")


def make_figure(cells=None, traces=None):
    """Three panels from committed CSVs: the fields, the error split, the energy."""
    import matplotlib.pyplot as plt

    cells = read_csv(CELLS_CSV) if cells is None else cells
    traces = read_csv(TRACE_CSV) if traces is None else traces

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    x = np.linspace(*X_RANGE, 401)
    for t, style in ((0.0, "-"), (0.3, "--"), (0.7, ":")):
        ax.plot(x, wave_exact(x, np.full_like(x, t), "pluck"), style,
                color="C0", lw=1.3, label=f"$t = {t}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_title("the plucked string, exact\n(d'Alembert, corners travelling)")
    ax.legend(fontsize=8)

    ax = axes[1]
    summary = summarize(cells)
    width = 0.35
    for i, key in enumerate(("mean_kink", "mean_smooth")):
        vals = [float(s[key]) for s in summary]
        ax.bar(np.arange(len(summary)) + (i - 0.5) * width, vals, width,
               label="within 0.05 of a corner" if i == 0 else "elsewhere",
               color=("C3" if i == 0 else "C0"))
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels([s["ic"] for s in summary])
    ax.set_yscale("log")
    ax.set_ylabel("rms error / rms $u$")
    ax.set_title("where the error lives")
    ax.legend(fontsize=8)

    ax = axes[2]
    for ic, color in (("sine", "C0"), ("pluck", "C3")):
        for seed in sorted({int(t["seed"]) for t in traces}):
            pts = sorted((int(t["step"]), float(t["energy_ratio"]))
                         for t in traces if t["ic"] == ic and int(t["seed"]) == seed)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                        lw=1.0, alpha=0.75, label=ic if seed == 0 else None)
    ax.axhline(1.0, color="0.35", ls=":", lw=1)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("energy / exact energy")
    ax.set_title("energy, which the loss never sees")
    ax.legend(fontsize=8)

    fig.suptitle("The wave equation against a d'Alembert ground truth", y=1.03,
                 fontsize=11)
    savefig(fig, "wave.png")


def figures_from_committed():
    make_figure()


def main(do_check=False, do_train=False, figures=False, quick=False):
    if quick:
        rows, _ = sweep(ics=ICS, seeds=(0,), write=False, n_interior=400,
                        n_ic=64, n_bc=64, width=16, depth=2, steps=60,
                        eval_every=30)
        report(rows)
        return
    if do_check:
        rows = check()
        write_csv(CHECK_CSV, list(rows[0].keys()), rows)
        return
    if do_train:
        t0 = time.time()
        rows, _ = sweep()
        print(f"\nsweep finished in {(time.time() - t0) / 60:.1f} min")
        report(rows)
        make_figure()
        return
    if figures:
        make_figure()
        return
    print(__doc__.strip().splitlines()[0])
    print("\nPass one of --check, --train, --figures, --quick.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(do_check=args.check, do_train=args.train, figures=args.figures,
         quick=args.quick)
