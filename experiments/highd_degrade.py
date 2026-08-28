"""Where the high-dimensional PINN degrades, and which part of it degrades.

Sec. 11 measured a fixed-budget PINN across d = 1 .. 16 and found the relative
L2 rising 1270x while the cost per step rose 8.2x; Sec. 12 turned that into an
equal-accuracy comparison and found the network stops reaching every target
6-15 dimensions before the mesh becomes expensive; Sec. 13 ran a second PDE and
found the collapse delayed by a factor of two in d but not removed. All three
leave the same question open, and both of the first two name it explicitly:
*what exactly is failing?* "Train longer" was tested at both termini and the
answer was no. What was never tested is a differently **shaped** budget --
where the collocation points go, rather than how many optimizer steps they get.

This section tests it, in four arms, all at one reduced budget (2000 Adam steps
rather than Sec. 11's 5000; see ``BUDGET`` for why that is fair here).

1. **The geometry, in closed form** (:func:`geometry_study`, no training). How
   many of n uniform collocation points actually carry the objective, as a
   function of d. The answer is exact and it is brutal: ``n (2/3)^d``, so the
   4000-point budget of Sec. 11 is worth **six points** at d = 16.
2. **The sampler** (:func:`run_queue`). Uniform against two alternatives at
   the same n: a *tilted* draw that puts points where the initial condition has
   its mass -- which uses only data the problem statement gives you -- and the
   repo's own residual-adaptive sampler from Sec. 7, which is the obvious thing
   to reach for and which cannot work here for a reason worth writing down.
3. **The density** (also :func:`run_queue`). n from 500 to 8000 at d = 8, to see
   whether the shortfall is something a bigger uniform budget buys back.
4. **The control** (:func:`fit_study`). Supervised regression onto the *exact*
   solution at the same points, same architecture, same budget. No PDE, no
   residual, no initial-condition penalty: labels. If that fails too, the
   failure is not the physics-informed part at all.

Why the effective count is exact
--------------------------------
The initial condition is ``u_0 = sum_m a_m phi_m`` with
``phi_k(x) = prod_i sin(k_i pi x_i)``, and the loss terms are empirical means of
squared quantities whose size is set by ``phi``. For an empirical mean
``(1/n) sum_i w(x_i)`` the number of samples actually carrying it is the usual

    ESS = n (E[w])^2 / E[w^2],

and with ``w = phi^2`` every expectation factorizes over coordinates, because
``phi`` is a product and the samples are independent per axis. Under a uniform
draw, ``E[sin^2] = 1/2`` and ``E[sin^4] = 3/8``, so

    ESS_uniform / n = (2^-d)^2 / (3/8)^d = **(2/3)^d**.

Under the tilted draw below, whose density is ``prod_i 2 sin^2(pi x_i)``, the
same computation with ``E[sin^6] = 5/16`` gives

    ESS_tilted / n = (2^d (3/8)^d)^2 / (2^d (5/16)^d) = **(9/10)^d**.

At d = 16 those are 0.0015 and 0.185: 6 effective points against 741, a **121x**
difference at the identical cost per step. Both are checked against Monte Carlo
in :func:`geometry_study` and in ``tests/test_highd_degrade.py``, and the second
is the reason the tilted arm exists -- it is a prediction made before the arm
was run, not a description of its outcome.

The same ratio has already appeared in this repo wearing different clothes:
Sec. 10's metric standard error is ``sqrt(((3/2)^d - 1)/n)``, and ``(3/2)^d`` is
exactly ``1/ESS_uniform`` per point. The estimator's precision and the
optimizer's signal are limited by one quantity.

What the tilted arm is and is not
---------------------------------
It is **not** an oracle. The density ``prod_i 2 sin^2(pi x_i)`` is built from the
initial condition's fundamental mode, and the initial condition is *given data* --
it is on the right-hand side of the problem statement. Nothing about the solution
at t > 0 enters it. What it does assume is that the solution stays where the
initial condition put it, which for the heat equation on a fixed time interval is
a reasonable guess and for a transport-dominated problem would be a bad one; that
limitation is stated in the README rather than hidden.

It is also **a change of objective, not a variance reduction.** Sampling from p
and averaging ``r^2`` unweighted minimizes ``E_p[r^2]``, a p-weighted residual
norm, not the uniform one. (Weighting by ``1/p`` would recover the uniform
objective and change nothing but the variance of its estimate.) The reason to
want the p-weighted objective is that the *metric* -- uniform L2 relative error --
is itself dominated by the region where u lives, so p-weighting moves the
objective towards the metric rather than away from it.

Run:  python experiments/highd_degrade.py --geometry   # closed form + MC, ~1 min
      python experiments/highd_degrade.py --fit        # the control arm, ~10 min
      python experiments/highd_degrade.py --run        # arms 2 and 3, ~2 h
                        [--seconds N]                  # time-box and resume
      python experiments/highd_degrade.py --figures    # replay from committed CSVs
      python experiments/highd_degrade.py --quick      # tiny end-to-end smoke run
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import read_csv, savefig, write_csv
from highd_heat import (
    HighDHeat,
    boundary_points,
    exact,
    exact_rms,
    initial_condition,
    model_config,
    rel_l2_mc,
    residual,
)
from highd_pinn import loss_scales, n_params
from pinn.model import MLP, set_seed

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
#: 2000 Adam steps rather than Sec. 11's 5000, and the reduction is measured
#: rather than asserted. Read off Sec. 11's *committed* trace, seed 0 of each d,
#: the relative L2 at step 2000 against step 5000: d = 8 goes 7.58e-1 -> 7.31e-1
#: and d = 16 goes 1.165 -> 1.016. In the regime this section studies the last
#: 3000 steps move the answer by 4% and 13%, both far smaller than the effects
#: below. At low d the truncation costs more (d = 2: 9.1e-3 -> 3.9e-3), which is
#: the conservative direction -- it makes the uniform baseline look *better*
#: relative to where it is going, not worse. ``--run`` re-measures the uniform
#: arm from scratch at this budget rather than reusing that trace, so every
#: number compared here comes from one budget and one code path.
BUDGET = dict(n_interior=4000, n_ic=400, n_bc=400,
              width=128, depth=4, steps=2000, lr=1e-3)

EVAL_EVERY = 200
EVAL_N = 100_000
SCORE_N = 1_000_000
SCORE_SEED = 7

#: Residual-adaptive arm (Sec. 7's RAR, lifted to d dimensions): keep a uniform
#: base and *add* residual-drawn points to it, never replace the whole set --
#: Sec. 7 measured that replacing it destabilizes a good fit. The 1/3 added
#: fraction and the k = c = 1 density are Sec. 7's, unchanged, so this arm is
#: the repo's own shipped tool rather than a new one tuned for this section.
RAR_FRACTION = 1.0 / 3.0
RAR_WARMUP = 500
RAR_EVERY = 250
RAR_CANDIDATES = 20_000

ARMS = ("uniform", "tilted", "rad")

CELLS_CSV = "highd_degrade_cells.csv"
TRACE_CSV = "highd_degrade_trace.csv"
GEOM_CSV = "highd_degrade_geometry.csv"
FIT_CSV = "highd_degrade_fit.csv"

CELL_FIELDS = ["arm", "d", "seed", "n_interior", "params", "rel_l2", "stderr",
               "rel_l2_final", "stderr_final", "best_step", "best_loss",
               "final_loss", "loss_r", "loss_ic", "loss_bc", "rel_ic_error",
               "rel_ic_sampled", "ic_energy_sampled", "rel_residual", "ess_ic",
               "ess_frac", "exact_rms",
               "train_seconds", "wall_seconds", "ms_per_step"]
TRACE_FIELDS = ["arm", "d", "seed", "n_interior", "step", "loss", "loss_r",
                "loss_ic", "loss_bc", "rel_l2", "stderr", "train_seconds"]

#: Per-cell resume state. Scratch, not committed; the results are in ``logs/``.
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                     ".degrade_cache"))


# ---------------------------------------------------------------------------
# The tilted draw
# ---------------------------------------------------------------------------
def tilted_marginal_cdf(x):
    """CDF of the density ``2 sin^2(pi x)`` on [0, 1].

    ``int_0^x 2 sin^2(pi s) ds = x - sin(2 pi x)/(2 pi)``, which is the
    antiderivative of ``1 - cos(2 pi s)``. Written out so the sampler can be
    checked against it rather than against another sampler.
    """
    x = np.asarray(x, dtype=float)
    return x - np.sin(2 * np.pi * x) / (2 * np.pi)


def tilted_unit(u, iters=50):
    """Inverse of :func:`tilted_marginal_cdf`, by bisection, on a torch tensor.

    Bisection rather than rejection for one reason that matters here: it
    consumes exactly one uniform per sample, so the number of random draws a
    cell makes is a function of its shape alone. That is what lets a run be
    checkpointed mid-training and resumed *exactly* -- rejection would make the
    generator's position depend on how many proposals happened to be rejected
    before the interruption.

    50 halvings take the bracket to 2^-50, far below float32's resolution, and
    the tests check the round trip against the CDF above.
    """
    lo = torch.zeros_like(u)
    hi = torch.ones_like(u)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = mid - torch.sin(2 * np.pi * mid) / (2 * np.pi)
        below = f < u
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def tilted_x(n, d, gen, dtype=torch.float32):
    """(n, d) spatial points from ``prod_i 2 sin^2(pi x_i)``, coordinatewise."""
    return tilted_unit(torch.rand(n, d, generator=gen, dtype=dtype))


def sample_x(arm, n, d, gen):
    """Spatial draw for an arm: uniform for ``uniform`` and ``rad``'s base."""
    if arm == "tilted":
        return tilted_x(n, d, gen)
    return torch.rand(n, d, generator=gen)


def interior_of(problem, arm, n, gen):
    """(n, d+1) grad-enabled space-time interior points for an arm.

    Time is uniform in every arm -- the concentration this section is about is
    spatial, and tilting time as well would confound the two.
    """
    t0, t1 = problem.t_range
    x = sample_x(arm, n, problem.d, gen)
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    coords = torch.cat([x, t], dim=1)
    coords.requires_grad_(True)
    return coords


def initial_of(problem, arm, n, gen):
    """(n, d+1) points on the initial slice, drawn according to the arm."""
    x = sample_x(arm, n, problem.d, gen)
    t = torch.full((n, 1), float(problem.t_range[0]))
    return torch.cat([x, t], dim=1)


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def ess(weights):
    """``(sum w)^2 / sum w^2`` -- the number of samples carrying a mean.

    Equals n when every weight is equal and 1 when one sample carries all of it.
    ``weights`` must be non-negative; a mean of squares always is.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s == 0:
        return 0.0
    return float(s * s / np.sum(w * w))


def ess_fraction(d, arm):
    """Closed-form ``ESS/n`` for the fundamental mode's energy at dimension d.

    ``(2/3)^d`` uniform, ``(9/10)^d`` tilted; see the module docstring for the
    derivation. ``rad``'s base is uniform, so it shares the uniform value at the
    moment its adaptive points are drawn -- but its added points are drawn from
    the network's residual and have no closed form at all, which is why the
    measured ``ess_ic`` column exists per cell.
    """
    if arm == "tilted":
        return 0.9 ** d
    return (2.0 / 3.0) ** d


def n_for_effective(target, d, arm):
    """Points needed for ``target`` effective ones at dimension d."""
    return target / ess_fraction(d, arm)


# ---------------------------------------------------------------------------
# Arm 1: the geometry, no training
# ---------------------------------------------------------------------------
GEOM_DIMS = (1, 2, 3, 4, 6, 8, 12, 16)


def geometry_study(dims=GEOM_DIMS, n=None, mc=200_000, seed=0):
    """Effective collocation count against d, closed form and measured.

    The measured column draws ``mc`` points per dimension and computes the ESS
    of ``phi^2`` on them directly, so the closed form is checked against the
    thing it claims to describe. Also reported: the nearest-neighbour spacing of
    the actual n-point collocation set, which is the other half of the story --
    the points are not only badly weighted, they are far apart. The expected
    nearest-neighbour distance among n uniform points in the unit cube grows
    like ``n^{-1/d}`` towards the cube's own diameter ``sqrt(d)``, and both are
    reported so the reader can see the sample stop being a sample.
    """
    n = BUDGET["n_interior"] if n is None else n
    rng = np.random.default_rng(seed)
    rows = []
    for d in dims:
        x = rng.random((mc, d))
        phi = np.prod(np.sin(np.pi * x), axis=1)
        meas_u = ess(phi ** 2) / mc

        u = rng.random((mc, d))
        xt = np.asarray(_tilted_np(u))
        phit = np.prod(np.sin(np.pi * xt), axis=1)
        meas_t = ess(phit ** 2) / mc

        nn = _nearest_neighbour_stats(n, d, rng)
        n_ic = BUDGET["n_ic"]
        hat = ess_estimator_bias(d, n_ic)
        rows.append({
            "d": d, "n": n, "mc": mc, "n_ic": n_ic,
            "ess_hat_median": f"{hat:.4f}",
            "ess_hat_ratio": f"{hat / (n_ic * ess_fraction(d, 'uniform')):.4f}",
            "ess_frac_uniform": f"{ess_fraction(d, 'uniform'):.6e}",
            "ess_frac_uniform_mc": f"{meas_u:.6e}",
            "ess_frac_tilted": f"{ess_fraction(d, 'tilted'):.6e}",
            "ess_frac_tilted_mc": f"{meas_t:.6e}",
            "eff_points_uniform": f"{n * ess_fraction(d, 'uniform'):.2f}",
            "eff_points_tilted": f"{n * ess_fraction(d, 'tilted'):.2f}",
            "n_for_1000_uniform": f"{n_for_effective(1000, d, 'uniform'):.4g}",
            "n_for_1000_tilted": f"{n_for_effective(1000, d, 'tilted'):.4g}",
            "nn_mean": f"{nn['mean']:.6f}", "nn_max": f"{nn['max']:.6f}",
            "cube_diameter": f"{np.sqrt(d):.6f}",
        })
    return rows


def ess_estimator_bias(d, n, repeats=200, rng=None):
    """Median of ``ESS`` estimated from an n-point draw, against the closed form.

    The effective count is a ratio of moments, and estimating it from the very
    sample it condemns does not work: the estimator is bounded below by 1, and a
    small sample almost never contains the rare points that carry the mass, so
    it reads high exactly where the sample is worst. At d = 16 with the 400
    initial-condition points a cell actually draws, the median estimate is
    **7.6x** the truth.

    This is the same shape as ``mcmc-from-scratch``'s annealed-importance result,
    where the ESS diagnostic read a perfect 1.000 on runs whose answer was wrong
    by a known amount: a diagnostic computed from an inadequate sample inherits
    the inadequacy. It is measured here so the per-cell ``ess_ic`` column can be
    read with the right amount of suspicion rather than as ground truth.
    """
    rng = np.random.default_rng(20260816 + d) if rng is None else rng
    vals = [ess(np.prod(np.sin(np.pi * rng.random((n, d))), axis=1) ** 2)
            for _ in range(repeats)]
    return float(np.median(vals))


def _tilted_np(u):
    """Numpy twin of :func:`tilted_unit`, for the training-free arm."""
    return tilted_unit(torch.as_tensor(np.asarray(u), dtype=torch.float64)).numpy()


def _nearest_neighbour_stats(n, d, rng, cap=4000):
    """Mean and max nearest-neighbour distance in an n-point uniform sample.

    Capped at ``cap`` points because this is an O(n^2 d) pairwise distance and
    the statistic it reports is a property of the sample size it is given; the
    cap is reported in the log's ``n`` column so the two never drift apart.
    """
    m = min(int(n), cap)
    x = rng.random((m, d))
    dist = np.sqrt(np.maximum(
        ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    return {"mean": float(nn.mean()), "max": float(nn.max())}


def report_geometry(rows):
    print("\n" + "=" * 88)
    print("Effective collocation count: how many of n points carry the objective")
    print("=" * 88)
    print(f"{'d':>3} {'ESS/n unif':>11} {'(MC)':>10} {'ESS/n tilt':>11} {'(MC)':>10} "
          f"{'eff pts u':>10} {'eff pts t':>10} {'n for 1000':>12} {'nn dist':>9}")
    for r in rows:
        print(f"{int(r['d']):>3} {float(r['ess_frac_uniform']):>11.4e} "
              f"{float(r['ess_frac_uniform_mc']):>10.3e} "
              f"{float(r['ess_frac_tilted']):>11.4e} "
              f"{float(r['ess_frac_tilted_mc']):>10.3e} "
              f"{float(r['eff_points_uniform']):>10.1f} "
              f"{float(r['eff_points_tilted']):>10.1f} "
              f"{float(r['n_for_1000_uniform']):>12.4g} "
              f"{float(r['nn_mean']):>9.3f}")
    last = rows[-1]
    print(f"\n  at d = {last['d']}, n = {last['n']}: "
          f"{float(last['eff_points_uniform']):.1f} effective uniform points vs "
          f"{float(last['eff_points_tilted']):.1f} tilted "
          f"({float(last['eff_points_tilted']) / float(last['eff_points_uniform']):.0f}x)")
    print(f"  holding 1000 effective points needs "
          f"{float(last['n_for_1000_uniform']):.4g} uniform points and "
          f"{float(last['n_for_1000_tilted']):.4g} tilted ones")
    print(f"  mean nearest-neighbour distance among {min(int(last['n']), 4000)} "
          f"points: {float(last['nn_mean']):.3f}, against a cube diameter of "
          f"{float(last['cube_diameter']):.3f}")
    print(f"\n  and the diagnostic is optimistic about itself: ESS estimated "
          f"from the {last['n_ic']} points a cell draws reads")
    for r in rows:
        print(f"    d = {int(r['d']):2d}: {float(r['ess_hat_median']):8.2f} "
              f"against a true {int(r['n_ic']) * ess_fraction(int(r['d']), 'uniform'):8.2f} "
              f"({float(r['ess_hat_ratio']):.2f}x)")


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------
def rar_points(problem, model, n_add, gen, n_candidates=RAR_CANDIDATES,
               k=1.0, c=1.0):
    """Residual-adaptive points in d dimensions (Sec. 7's RAD density).

    Draw a uniform candidate pool, evaluate the current residual on it, and
    resample from ``p_i ~ |r_i|^k / mean|r|^k + c``. Identical formula to
    ``pinn.losses.adaptive_interior_points``, which is 1D-only because it takes
    an ``x_range``; the density and the k = c = 1 defaults are unchanged.

    The residual is evaluated under autograd but only its magnitude is used, so
    the pool is detached before selection -- no training signal flows through
    the choice of points.
    """
    cand = interior_of(problem, "uniform", n_candidates, gen)
    r = residual(problem, model(cand), cand)
    with torch.no_grad():
        w = r.detach().abs().flatten() ** k
        w = w / w.mean().clamp_min(1e-30) + c
        idx = torch.multinomial(w, n_add, replacement=n_add > n_candidates,
                                generator=gen)
    pts = cand.detach()[idx].clone()
    pts.requires_grad_(True)
    return pts


def train_cell(problem, arm, seed=0, n_interior=None, n_ic=None, n_bc=None,
               width=None, depth=None, steps=None, lr=None, w_ic=1.0, w_bc=1.0,
               eval_every=EVAL_EVERY, eval_n=EVAL_N, select="final",
               verbose=False, ckpt_path=None, ckpt_every=100, deadline=None):
    """Train one cell under one sampler; return ``(model, history, best)``.

    Same recipe as ``highd_heat.train`` -- soft IC and BC penalties, Adam,
    selection on the lowest *training* loss, the pre-update snapshot, the final
    iterate scored so it can compete -- with one thing changed: where the
    interior and initial-condition points come from. The boundary draw is
    untouched in every arm, because the Dirichlet data is 0 on all 2d faces and
    there is no concentration to exploit in a function that is identically zero.

    ``arm``:
      ``uniform``  the Sec. 11 sampler, the baseline.
      ``tilted``   interior and IC points from ``prod_i 2 sin^2(pi x_i)``.
      ``rad``      uniform base of ``n(1 - RAR_FRACTION)`` plus ``n
                   RAR_FRACTION`` residual-drawn points, refreshed every
                   ``RAR_EVERY`` steps after ``RAR_WARMUP``; IC uniform.

    A note on selecting by training loss in the ``rad`` arm: its objective is
    evaluated on a point set that changes every ``RAR_EVERY`` steps, so losses
    from different epochs are not strictly the same functional. The selection
    rule is kept anyway, because changing it per arm would make the arms
    incomparable, and the effect is reported rather than assumed -- ``best_step``
    is in the log for every cell.

    Resumption is exact. The only randomness after construction is the ``rad``
    arm's resampling, and the generator's state is checkpointed with everything
    else, so a resumed run and an uninterrupted one visit the same parameters at
    the same steps; ``tests/test_highd_degrade.py`` checks that on both a fixed
    and a resampling arm.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if select not in ("best_loss", "final"):
        raise ValueError(f"select must be 'best_loss' or 'final', got {select!r}")
    b = dict(BUDGET)
    for name, value in (("n_interior", n_interior), ("n_ic", n_ic),
                        ("n_bc", n_bc), ("width", width), ("depth", depth),
                        ("steps", steps), ("lr", lr)):
        if value is not None:
            b[name] = value

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    n_add = int(round(b["n_interior"] * RAR_FRACTION)) if arm == "rad" else 0
    n_base = b["n_interior"] - n_add

    # The full n is drawn in every arm, and the ``rad`` arm trains on all of it
    # until its first resample -- exactly Sec. 7's arrangement, where the arms
    # are bit-for-bit the same run through the warmup and only then diverge. An
    # earlier version drew only ``n_base`` up front, which quietly gave the
    # adaptive arm a third fewer points for the first RAR_WARMUP steps and would
    # have charged the difference to adaptivity.
    base = interior_of(problem, "uniform" if arm == "rad" else arm,
                       b["n_interior"], gen)
    ic = initial_of(problem, arm, b["n_ic"], gen)
    ic_target = initial_condition(problem, ic[:, : problem.d])
    bc = boundary_points(problem, b["n_bc"], gen)
    bc_target = torch.zeros(bc.shape[0], 1)
    added = None
    keep = None

    model = MLP(**model_config(problem, b["width"], b["depth"]))
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])

    history = []
    best = dict(step=-1, loss=float("inf"))
    best_state = None
    train_seconds = 0.0
    start = 0

    if ckpt_path is not None and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        gen.set_state(blob["gen"])
        history = [tuple(row) for row in blob["history"]]
        best = dict(blob["best"])
        best_state = blob["best_state"]
        train_seconds = float(blob["train_seconds"])
        start = int(blob["step"]) + 1
        if blob["added"] is not None:
            added = blob["added"].clone().requires_grad_(True)
            keep = base[:n_base].detach().clone().requires_grad_(True)
        if verbose:
            print(f"  resumed from {os.path.basename(ckpt_path)} at step {start}",
                  flush=True)

    def _save(step):
        if ckpt_path is None:
            return
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "gen": gen.get_state(), "history": history, "best": best,
                    "best_state": best_state, "train_seconds": train_seconds,
                    "added": None if added is None else added.detach(),
                    "step": step}, ckpt_path)

    def _interior():
        if added is None:
            return base
        return torch.cat([keep, added], dim=0)

    for step in range(start, b["steps"] + 1):
        t_step = time.perf_counter()
        if arm == "rad" and step >= RAR_WARMUP and (step - RAR_WARMUP) % RAR_EVERY == 0:
            if keep is None:
                keep = base[:n_base].detach().clone().requires_grad_(True)
            added = rar_points(problem, model, n_add, gen)

        opt.zero_grad()
        interior = _interior()
        r = residual(problem, model(interior), interior)
        loss_r = torch.mean(r ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()

        value = float(loss.detach().item())
        if value < best["loss"]:
            best = dict(step=step, loss=value)
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        opt.step()
        train_seconds += time.perf_counter() - t_step

        if step % eval_every == 0 or step == b["steps"]:
            err, se = rel_l2_mc(model, problem, n=eval_n, seed=12345)
            history.append((step, value, float(loss_r.item()),
                            float(loss_ic.item()), float(loss_bc.item()),
                            err, se, train_seconds))
            if verbose:
                print(f"  step {step:5d}  loss {value:.3e}  relL2 {err:.4f}",
                      flush=True)

        if (ckpt_path is not None and step > start and step % ckpt_every == 0
                and step < b["steps"]):
            _save(step)
            if deadline is not None and time.monotonic() >= deadline:
                best = dict(best)
                best["completed"] = False
                best["stopped_at"] = step
                return model, history, best

    interior = _interior()
    final_r = residual(problem, model(interior), interior)
    final_loss = float((torch.mean(final_r ** 2)
                        + w_ic * torch.mean((model(ic) - ic_target) ** 2)
                        + w_bc * torch.mean((model(bc) - bc_target) ** 2)).detach())
    if final_loss < best["loss"]:
        best = dict(step=b["steps"] + 1, loss=final_loss)
        best_state = None
    best["final_loss"] = final_loss
    best["train_seconds"] = train_seconds
    best["state_dict"] = best_state
    best["completed"] = True
    # Measured on the *actual* initial condition, which is two modes rather
    # than the fundamental alone, so it sits a constant factor below the closed
    # form in ``ess_fraction``: 25/33 of it under a uniform draw, derived and
    # checked in the tests. Measured rather than computed because the ``rad``
    # arm's interior points have no closed form at all.
    best["n_ic"] = int(ic.shape[0])
    # The energy of the initial condition *under the measure this cell sampled*.
    # ``loss_ic`` divided by the closed-form uniform energy is not comparable
    # across arms -- a tilted draw sits where u_0 is large, so the same quality
    # of fit posts a much bigger number against a uniform denominator (at d = 16
    # it reads 11.6, which looks like a catastrophic misfit and is mostly the
    # normalization). Both readings are reported per cell: ``rel_ic_error``
    # against the uniform energy, ``rel_ic_sampled`` against this one.
    best["ic_energy_sampled"] = float(torch.mean(ic_target ** 2))
    best["ess_ic"] = ess(
        initial_condition(problem, ic[:, : problem.d]).detach().numpy().ravel() ** 2)

    if select == "best_loss" and best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


def replay_points(problem, arm, seed, n_interior, n_ic=None, n_bc=None):
    """Rebuild a cell's point sets without training it.

    Draws in exactly the order :func:`train_cell` does -- interior, then initial,
    then boundary -- from a generator seeded the same way, so the sets are the
    ones that cell trained on. That is what lets a column be added to an
    already-committed row without spending the cell's minutes again, and the
    equality is a test rather than an assumption
    (``test_replay_points_reproduces_the_training_draw``).
    """
    n_ic = BUDGET["n_ic"] if n_ic is None else n_ic
    n_bc = BUDGET["n_bc"] if n_bc is None else n_bc
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    interior = interior_of(problem, "uniform" if arm == "rad" else arm,
                           n_interior, gen)
    ic = initial_of(problem, arm, n_ic, gen)
    bc = boundary_points(problem, n_bc, gen)
    return interior, ic, bc


def sampled_ic_energy(problem, arm, seed, n_interior, n_ic=None):
    """``mean u_0(x)^2`` over the initial-condition points a cell drew."""
    _, ic, _ = replay_points(problem, arm, seed, n_interior, n_ic=n_ic)
    return float(torch.mean(
        initial_condition(problem, ic[:, : problem.d]) ** 2))


# ---------------------------------------------------------------------------
# Arms 2 and 3: the cell queue
# ---------------------------------------------------------------------------
#: (arm, d, seed, n_interior). Ordered by what the section needs most, because
#: a session that runs out of time should stop having answered the main
#: question, not having half-finished a supporting one. Uniform and tilted at
#: d = 8 and 16 come first; the extra seeds and the density sweep come last.
def default_queue():
    n = BUDGET["n_interior"]
    q = []
    for d in (8, 16):                                # the sampler, where it fails
        for arm in ("uniform", "tilted"):
            for seed in (0, 1):
                q.append((arm, d, seed, n))
    q += [("rad", 8, s, n) for s in (0, 1)]          # the obvious tool, at d = 8
    for arm in ("uniform", "tilted"):                # the sampler where it works
        q += [(arm, 4, s, n) for s in (0, 1)]
    for nn in (8000, 2000, 500):                     # density, at d = 8
        q += [("uniform", 8, s, nn) for s in (0, 1)]
    # ``rad`` is run at d = 8 only. Its d = 8 cells answer the question the arm
    # exists for -- whether residual-adaptive sampling finds the concentration
    # on its own -- and a second failing dimension would cost 17 minutes to
    # confirm it. Named here rather than quietly omitted.
    for d, seeds in ((8, (2, 3, 4)), (4, (2, 3, 4)), (2, (0, 1, 2, 3, 4))):
        q += [("uniform", d, s, n) for s in seeds]   # seed spread
    return q


def cell_key(row):
    return (row["arm"], int(row["d"]), int(row["seed"]), int(row["n_interior"]))


def run_cell(arm, d, seed, n_interior, verbose=True, deadline=None,
             resumable=True, **kw):
    """One queue entry, trained and scored. ``(None, None)`` if time ran out."""
    problem = HighDHeat(d)
    if verbose:
        print(f"  {arm:8s} d={d:2d} seed={seed} n={n_interior}", flush=True)
    ckpt = (os.path.join(CACHE, f"{arm}_d{d}_s{seed}_n{n_interior}.pt")
            if resumable else None)
    t0 = time.time()
    model, history, best = train_cell(problem, arm, seed=seed,
                                      n_interior=n_interior, select="final",
                                      verbose=verbose, ckpt_path=ckpt,
                                      deadline=deadline, **kw)
    wall = time.time() - t0
    if not best.get("completed", True):
        if verbose:
            print(f"           paused at step {best['stopped_at']} "
                  f"({wall:.0f}s this call)", flush=True)
        return None, None

    rel_f, se_f = rel_l2_mc(model, problem, n=SCORE_N, seed=SCORE_SEED)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    rel, se = rel_l2_mc(model, problem, n=SCORE_N, seed=SCORE_SEED)

    step, loss, loss_r, loss_ic, loss_bc, _, _, train_s = history[-1]
    ic_energy, residual_scale = loss_scales(problem)
    row = {
        "arm": arm, "d": d, "seed": seed, "n_interior": n_interior,
        "params": n_params(d, BUDGET["width"], BUDGET["depth"]),
        "rel_l2": f"{rel:.6e}", "stderr": f"{se:.6e}",
        "rel_l2_final": f"{rel_f:.6e}", "stderr_final": f"{se_f:.6e}",
        "best_step": best["step"], "best_loss": f"{best['loss']:.6e}",
        "final_loss": f"{best['final_loss']:.6e}",
        "loss_r": f"{loss_r:.6e}", "loss_ic": f"{loss_ic:.6e}",
        "loss_bc": f"{loss_bc:.6e}",
        "rel_ic_error": f"{np.sqrt(loss_ic / ic_energy):.6f}",
        "rel_ic_sampled": f"{np.sqrt(loss_ic / best['ic_energy_sampled']):.6f}",
        "ic_energy_sampled": f"{best['ic_energy_sampled']:.6e}",
        "rel_residual": f"{np.sqrt(loss_r / residual_scale):.6f}",
        "ess_ic": f"{best['ess_ic']:.3f}",
        "ess_frac": f"{best['ess_ic'] / best['n_ic']:.6e}",
        "exact_rms": f"{exact_rms(problem):.6e}",
        "train_seconds": f"{train_s:.2f}", "wall_seconds": f"{wall:.2f}",
        "ms_per_step": f"{1000 * train_s / (BUDGET['steps'] + 1):.3f}",
    }
    if verbose:
        print(f"           rel L2 {rel:.4e} +- {se:.1e}  "
              f"(final {rel_f:.4e})  rel IC err {row['rel_ic_error']}  "
              f"{train_s:.0f}s", flush=True)
    if ckpt is not None and os.path.exists(ckpt):
        os.remove(ckpt)
    return row, history


def _load_partial(cells_csv=CELLS_CSV, trace_csv=TRACE_CSV):
    try:
        rows = read_csv(cells_csv)
    except FileNotFoundError:
        return [], []
    try:
        traces = read_csv(trace_csv)
    except FileNotFoundError:
        traces = []
    return rows, traces


def run_queue(queue=None, seconds=None, resume=True, write=True, verbose=True,
              **kw):
    """Run the queue, skipping cells already in ``logs/``, time-boxed.

    Same two-level resumption as Sec. 13's sweep: a finished cell is skipped on
    a later call because its row is committed, and an unfinished one is picked
    up mid-training from its checkpoint. Returns ``(rows, traces, complete)``.
    """
    queue = default_queue() if queue is None else queue
    rows, traces = _load_partial() if resume else ([], [])
    done = {cell_key(r) for r in rows}
    deadline = None if seconds is None else time.monotonic() + float(seconds)

    for (arm, d, seed, n) in queue:
        if (arm, d, seed, n) in done:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            return rows, traces, False
        row, history = run_cell(arm, d, seed, n, verbose=verbose,
                                deadline=deadline, **kw)
        if row is None:
            return rows, traces, False
        rows.append(row)
        for (step, loss, lr_, li, lb, err, se, ts) in history:
            traces.append({"arm": arm, "d": d, "seed": seed, "n_interior": n,
                           "step": step, "loss": f"{loss:.6e}",
                           "loss_r": f"{lr_:.6e}", "loss_ic": f"{li:.6e}",
                           "loss_bc": f"{lb:.6e}", "rel_l2": f"{err:.6e}",
                           "stderr": f"{se:.6e}", "train_seconds": f"{ts:.4f}"})
        if write:
            write_csv(CELLS_CSV, CELL_FIELDS, rows)
            write_csv(TRACE_CSV, TRACE_FIELDS, traces)
    return rows, traces, True


# ---------------------------------------------------------------------------
# Arm 4: the supervised control
# ---------------------------------------------------------------------------
FIT_DIMS = (1, 2, 4, 8, 16)
FIT_SEEDS = (0, 1, 2)
FIT_FIELDS = ["arm", "d", "seed", "n_points", "steps", "rel_l2", "stderr",
              "final_mse", "rel_fit_error", "rel_fit_sampled", "sampled_rms",
              "exact_rms", "train_seconds"]


def fit_cell(problem, arm, seed=0, n_points=None, steps=None, width=None,
             depth=None, lr=None, eval_n=SCORE_N, activation="tanh"):
    """Supervised regression onto the exact solution at the arm's points.

    The control that separates three explanations of the d-collapse: the network
    cannot represent the target, the optimizer cannot find it, or the *sample*
    cannot see it. Same architecture, same optimizer, same step count and the
    same points as the corresponding PINN arm -- the only change is the loss,
    which is the mean squared difference from the closed-form solution at those
    points. No residual, no initial-condition penalty, no boundary penalty.

    If this fails where the PINN fails, the physics-informed machinery is not
    what is failing. Reported as ``rel_l2`` (the same uniform-L2 metric every
    other number in this repo uses) and ``rel_fit_error``, the training-set fit
    error divided by the exact rms -- the gap between those two is the whole
    generalization question in one pair of numbers.
    """
    b = dict(BUDGET)
    for name, value in (("n_interior", n_points), ("steps", steps),
                        ("width", width), ("depth", depth), ("lr", lr)):
        if value is not None:
            b[name] = value

    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    t0, t1 = problem.t_range
    x = sample_x(arm, b["n_interior"], problem.d, gen)
    t = t0 + (t1 - t0) * torch.rand(b["n_interior"], 1, generator=gen)
    coords = torch.cat([x, t], dim=1)
    target = torch.as_tensor(
        exact(problem, coords[:, : problem.d].numpy(),
              coords[:, -1].numpy()), dtype=torch.float32).unsqueeze(1)

    model = MLP(**model_config(problem, b["width"], b["depth"], activation))
    opt = torch.optim.Adam(model.parameters(), lr=b["lr"])
    start = time.perf_counter()
    mse = float("nan")
    for _ in range(b["steps"] + 1):
        opt.zero_grad()
        loss = torch.mean((model(coords) - target) ** 2)
        loss.backward()
        mse = float(loss.detach())
        opt.step()
    train_seconds = time.perf_counter() - start

    rel, se = rel_l2_mc(model, problem, n=eval_n, seed=SCORE_SEED)
    sampled_ms = float(torch.mean(target ** 2))
    return {
        "arm": arm, "d": problem.d, "seed": seed, "n_points": b["n_interior"],
        "steps": b["steps"], "rel_l2": f"{rel:.6e}", "stderr": f"{se:.6e}",
        "final_mse": f"{mse:.6e}",
        "rel_fit_error": f"{np.sqrt(mse) / exact_rms(problem):.6e}",
        "rel_fit_sampled": f"{np.sqrt(mse / sampled_ms):.6e}",
        "sampled_rms": f"{np.sqrt(sampled_ms):.6e}",
        "exact_rms": f"{exact_rms(problem):.6e}",
        "train_seconds": f"{train_seconds:.2f}",
    }


#: The control's own control. If regression stalls at the shared 2000-step
#: budget, "the network cannot fit this target" and "the network was not given
#: long enough" are the same measurement, and only a longer run separates them.
#: A regression step costs ~12 ms at every d (no derivatives, and only the input
#: layer grows), so 20x the budget is minutes rather than hours -- which is why
#: the ladder exists here and could not exist for the PINN arms.
FIT_LADDER_DIMS = (4, 8, 16)
FIT_LADDER_STEPS = (10_000, 40_000)


def fit_study(dims=FIT_DIMS, arms=("uniform", "tilted"), seeds=FIT_SEEDS,
              ladder_dims=FIT_LADDER_DIMS, ladder_steps=FIT_LADDER_STEPS,
              ladder_seeds=(0,), write=True, verbose=True):
    """Regression at the shared budget, then a budget ladder at high d.

    Resumes: a cell whose (arm, d, seed, steps) is already in the committed log
    is skipped, so extending the ladder does not re-run the shared-budget grid.
    """
    try:
        rows = read_csv(FIT_CSV)
    except FileNotFoundError:
        rows = []
    done = {(r["arm"], int(r["d"]), int(r["seed"]), int(r["steps"]))
            for r in rows}

    def _run(problem, arm, seed, steps=None):
        key = (arm, problem.d, seed, BUDGET["steps"] if steps is None else steps)
        if key in done:
            return
        row = fit_cell(problem, arm, seed=seed, steps=steps)
        rows.append(row)
        done.add(key)
        if verbose:
            print(f"  fit {arm:8s} d={int(row['d']):2d} seed={seed} "
                  f"steps={row['steps']:>5}  rel L2 {float(row['rel_l2']):.4e}  "
                  f"train-set {float(row['rel_fit_sampled']):.4e}  "
                  f"{float(row['train_seconds']):.0f}s", flush=True)
        if write:
            write_csv(FIT_CSV, FIT_FIELDS, rows)

    for d in dims:
        problem = HighDHeat(d)
        for arm in arms:
            for seed in seeds:
                _run(problem, arm, seed)
    for d in ladder_dims:
        problem = HighDHeat(d)
        for steps in ladder_steps:
            for seed in ladder_seeds:
                _run(problem, "uniform", seed, steps=steps)
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(rows, n_interior=None):
    """Per (arm, d) mean/sd/min/max of the relative L2 across seeds."""
    n_interior = BUDGET["n_interior"] if n_interior is None else n_interior
    keys = sorted({(r["arm"], int(r["d"])) for r in rows
                   if int(r["n_interior"]) == n_interior},
                  key=lambda k: (k[1], k[0]))
    out = []
    for arm, d in keys:
        cells = [r for r in rows if r["arm"] == arm and int(r["d"]) == d
                 and int(r["n_interior"]) == n_interior]
        errs = np.array([float(r["rel_l2"]) for r in cells])
        out.append({
            "arm": arm, "d": d, "n_seeds": len(cells),
            "mean": f"{errs.mean():.6e}",
            "median": f"{float(np.median(errs)):.6e}",
            "sd": f"{errs.std(ddof=1):.6e}" if len(errs) > 1 else "",
            "min": f"{errs.min():.6e}", "max": f"{errs.max():.6e}",
            "spread": f"{errs.max() / errs.min():.3f}",
            "mean_rel_ic": f"{np.mean([float(r['rel_ic_error']) for r in cells]):.4f}",
            "mean_rel_ic_sampled":
                f"{np.mean([float(r['rel_ic_sampled']) for r in cells]):.4f}",
            "mean_ess_ic": f"{np.mean([float(r['ess_ic']) for r in cells]):.1f}",
            "mean_seconds": f"{np.mean([float(r['train_seconds']) for r in cells]):.1f}",
        })
    return out


def report(rows, n_interior=None):
    n_interior = BUDGET["n_interior"] if n_interior is None else n_interior
    summary = summarize(rows, n_interior=n_interior)
    print("\n" + "=" * 88)
    print(f"Sampler arms at n = {n_interior}, {BUDGET['steps']} Adam steps")
    print("=" * 88)
    print(f"{'d':>3} {'arm':>9} {'seeds':>6} {'mean relL2':>12} {'median':>11} "
          f"{'min':>11} {'max':>11} {'spread':>7} {'relIC*':>7} {'ESS(IC)':>8}")
    for s in summary:
        print(f"{s['d']:>3} {s['arm']:>9} {s['n_seeds']:>6} "
              f"{float(s['mean']):>12.4e} {float(s['median']):>11.4e} "
              f"{float(s['min']):>11.4e} {float(s['max']):>11.4e} "
              f"{float(s['spread']):>7.2f} "
              f"{float(s['mean_rel_ic_sampled']):>7.3f} "
              f"{float(s['mean_ess_ic']):>8.1f}")

    for d in sorted({s["d"] for s in summary}):
        arms = {s["arm"]: s for s in summary if s["d"] == d}
        if "uniform" in arms and "tilted" in arms:
            u, t = float(arms["uniform"]["mean"]), float(arms["tilted"]["mean"])
            print(f"  d = {d:2d}: tilted / uniform = {t / u:.3f} "
                  f"({u:.3e} -> {t:.3e})")

    dens = [r for r in rows if r["arm"] == "uniform" and int(r["d"]) == 8]
    ns = sorted({int(r["n_interior"]) for r in dens})
    if len(ns) > 1:
        print(f"\nCollocation density at d = 8 "
              f"(uniform; effective points = n x {ess_fraction(8, 'uniform'):.4f}):")
        print(f"{'n':>7} {'eff':>7} {'seeds':>6} {'mean relL2':>12} {'relIC*':>7}")
        print("  (relIC* = IC error against the IC energy on the cell's own "
              "sample, so it is comparable across arms)")
        for n in ns:
            cells = [r for r in dens if int(r["n_interior"]) == n]
            errs = np.array([float(r["rel_l2"]) for r in cells])
            ic = np.mean([float(r["rel_ic_sampled"]) for r in cells])
            print(f"{n:>7} {n * ess_fraction(8, 'uniform'):>7.1f} {len(cells):>6} "
                  f"{errs.mean():>12.4e} {ic:>7.3f}")
    return summary


def report_fit(rows):
    print("\n" + "=" * 88)
    print("Control: supervised regression onto the exact solution, same points, "
          "same budget")
    print("=" * 88)
    print(f"{'d':>3} {'arm':>9} {'steps':>7} {'seeds':>6} {'mean relL2':>12} "
          f"{'min':>11} {'max':>11} {'own-sample err':>15}")
    for d in sorted({int(r["d"]) for r in rows}):
        for steps in sorted({int(r["steps"]) for r in rows}):
            for arm in sorted({r["arm"] for r in rows}):
                cells = [r for r in rows if int(r["d"]) == d and r["arm"] == arm
                         and int(r["steps"]) == steps]
                if not cells:
                    continue
                errs = np.array([float(r["rel_l2"]) for r in cells])
                fit = np.mean([float(r["rel_fit_sampled"]) for r in cells])
                print(f"{d:>3} {arm:>9} {steps:>7} {len(cells):>6} "
                      f"{errs.mean():>12.4e} {errs.min():>11.4e} "
                      f"{errs.max():>11.4e} {fit:>14.4e}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
ARM_STYLE = {"uniform": ("C0", "o", "uniform"),
             "tilted": ("C1", "s", "tilted (IC-informed)"),
             "rad": ("C2", "^", "residual-adaptive (Sec. 7)")}


def make_figure(cells=None, geom=None, fit=None):
    """Four panels, all replayed from committed CSVs. No training."""
    import matplotlib.pyplot as plt

    cells = read_csv(CELLS_CSV) if cells is None else cells
    geom = read_csv(GEOM_CSV) if geom is None else geom
    fit = read_csv(FIT_CSV) if fit is None else fit

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9))

    # (a) the closed form and its Monte Carlo check
    ax = axes[0]
    gd = np.array([float(r["d"]) for r in geom])
    for key, color, label in (("uniform", "C0", r"uniform: $(2/3)^d$"),
                              ("tilted", "C1", r"tilted: $(9/10)^d$")):
        ax.plot(gd, [float(r[f"ess_frac_{key}"]) for r in geom], "-",
                color=color, lw=1.4, label=label)
        ax.plot(gd, [float(r[f"ess_frac_{key}_mc"]) for r in geom], "o",
                color=color, ms=4, mfc="none", label=None)
    ax.set_yscale("log")
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("effective fraction of points")
    ax.set_title("how many collocation points count\n(line: closed form, "
                 "circles: Monte Carlo)")
    ax.legend(fontsize=8)

    # (b) the sampler arms
    ax = axes[1]
    summary = summarize(cells)
    for arm, (color, marker, label) in ARM_STYLE.items():
        pts = [(s["d"], float(s["mean"])) for s in summary if s["arm"] == arm]
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color,
                marker=marker, ms=5, lw=1.3, label=label)
        for s in (s for s in summary if s["arm"] == arm):
            cs = [float(r["rel_l2"]) for r in cells if r["arm"] == arm
                  and int(r["d"]) == s["d"]
                  and int(r["n_interior"]) == BUDGET["n_interior"]]
            ax.plot([s["d"]] * len(cs), cs, marker, color=color, ms=3,
                    alpha=0.45)
    ax.axhline(1.0, color="0.35", ls=":", lw=1)
    ax.annotate(r"$u_\theta \equiv 0$", xy=(1.05, 1.06), fontsize=7, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("relative $L^2$")
    ax.set_title(f"where the points go ({BUDGET['steps']} steps, "
                 f"n = {BUDGET['n_interior']})")
    ax.legend(fontsize=8)

    # (c) density at d = 8
    ax = axes[2]
    dens = [r for r in cells if r["arm"] == "uniform" and int(r["d"]) == 8]
    ns = sorted({int(r["n_interior"]) for r in dens})
    means = [np.mean([float(r["rel_l2"]) for r in dens
                      if int(r["n_interior"]) == n]) for n in ns]
    ax.plot(ns, means, "-o", color="C0", ms=5, label="uniform, $d = 8$")
    for n in ns:
        cs = [float(r["rel_l2"]) for r in dens if int(r["n_interior"]) == n]
        ax.plot([n] * len(cs), cs, "o", color="C0", ms=3, alpha=0.45)
    tilt8 = [float(r["rel_l2"]) for r in cells if r["arm"] == "tilted"
             and int(r["d"]) == 8 and int(r["n_interior"]) == BUDGET["n_interior"]]
    if tilt8:
        ax.axhline(float(np.mean(tilt8)), color="C1", ls="--", lw=1.2,
                   label=f"tilted at n = {BUDGET['n_interior']}")
    ax.axhline(1.0, color="0.35", ls=":", lw=1, label=r"$u_\theta \equiv 0$")
    # The panel's whole content is that the blue line is flat, and a log axis
    # auto-scaled to a 3% spread would draw that flatness as a mountain range.
    # The limits are set from the other things on the plot instead.
    ax.set_ylim(0.3, 3.0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("interior collocation points $n$")
    ax.set_ylabel("relative $L^2$")
    ax.set_title(f"more uniform points, $d = 8$\n"
                 f"({max(ns) // min(ns)}x range, "
                 f"{max(means) / min(means):.2f}x in error)")
    ax.legend(fontsize=8, loc="lower left")

    # (d) the supervised control
    ax = axes[3]
    shared = [r for r in fit if int(r["steps"]) == BUDGET["steps"]]
    for arm, color in (("uniform", "C0"), ("tilted", "C1")):
        ds = sorted({int(r["d"]) for r in shared if r["arm"] == arm})
        if not ds:
            continue
        ax.plot(ds, [np.mean([float(r["rel_l2"]) for r in shared
                              if r["arm"] == arm and int(r["d"]) == d])
                     for d in ds], "-", color=color, marker="D", ms=4,
                label=f"regression, {arm}")
    ladder = sorted({int(r["steps"]) for r in fit}) [1:]
    for steps, style in zip(ladder, ("--", "-.")):
        pts = sorted((int(r["d"]), float(r["rel_l2"])) for r in fit
                     if r["arm"] == "uniform" and int(r["steps"]) == steps)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, color="0.4",
                    lw=1.1, marker="x", ms=4,
                    label=f"regression, uniform, {steps} steps")
    for arm, color in (("uniform", "C0"), ("tilted", "C1")):
        pts = [(s["d"], float(s["mean"])) for s in summary if s["arm"] == arm]
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ":", color=color,
                    marker="o", ms=4, alpha=0.7, label=f"PINN, {arm}")
    ax.axhline(1.0, color="0.35", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("spatial dimension $d$")
    ax.set_ylabel("relative $L^2$")
    ax.set_title("control: fit the exact solution\nat the same points")
    ax.legend(fontsize=7)

    fig.suptitle("Where the high-dimensional PINN degrades: the sample, not the "
                 "solver", y=1.03, fontsize=11)
    savefig(fig, "highd_degrade.png")


def figures_from_committed():
    make_figure()


# ---------------------------------------------------------------------------
def main(geometry=False, fit=False, run=False, figures=False, quick=False,
         seconds=None):
    if quick:
        rows = []
        for arm in ARMS:
            row, _ = run_cell(arm, 2, 0, 300, resumable=False,
                              steps=40, n_ic=50, n_bc=50, width=16, depth=2,
                              eval_every=20, eval_n=5000)
            rows.append(row)
        report(rows, n_interior=300)
        print()
        p = HighDHeat(2)
        for arm in ("uniform", "tilted"):
            print(fit_cell(p, arm, n_points=300, steps=40, width=16, depth=2,
                           eval_n=20_000))
        return

    if geometry:
        rows = geometry_study()
        write_csv(GEOM_CSV, list(rows[0].keys()), rows)
        report_geometry(rows)
        return

    if fit:
        rows = fit_study()
        report_fit(rows)
        return

    if run:
        t0 = time.time()
        rows, _, complete = run_queue(seconds=seconds)
        print(f"\n{len(rows)} cells on disk after {(time.time() - t0) / 60:.1f} "
              f"min this call; queue "
              f"{'complete' if complete else 'INCOMPLETE -- call again'}")
        if complete:
            report(rows)
        return

    if figures:
        make_figure()
        return

    print(__doc__.strip().splitlines()[0])
    print("\nPass one of --geometry, --fit, --run [--seconds N], --figures, "
          "--quick.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seconds", type=float, default=None,
                    help="wall-clock budget for this --run call; it checkpoints "
                         "and returns when the budget expires, and the next "
                         "call resumes mid-cell")
    args = ap.parse_args()
    main(geometry=args.geometry, fit=args.fit, run=args.run,
         figures=args.figures, quick=args.quick, seconds=args.seconds)
