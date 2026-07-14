"""Spectral bias: the failure mode, measured, and one thing that fixes it.

This is the repo's honest-negative-result experiment.  The heat and Burgers
solves show PINNs working; this one shows the standard PINN *not* working, in a
way that is predictable rather than mysterious, and then fixes it.

The claim
---------
A plain MLP trained by gradient descent learns the low-frequency content of its
target first and the high-frequency content last -- or, on a finite step
budget, never.  This is *spectral bias* (Rahaman et al. 2019).  Its cleanest
explanation is the neural tangent kernel: in the linearized regime, the
component of the residual along the NTK eigendirection with eigenvalue
``lambda_i`` decays like ``(1 - eta lambda_i)^s`` after ``s`` gradient steps.
Large-eigenvalue directions are learned fast, small ones slowly.  For an MLP on
a low-dimensional input the NTK's eigenfunctions are essentially sinusoids and
its eigenvalues *decay with frequency* -- so "high frequency" and "small
eigenvalue" are the same statement, and the learning-rate ordering follows.

That geometric-series argument is derived from scratch in the sibling repo
gp-from-scratch (``theory/derivations.md``, Sec. 6-7: the NTK as the kernel of
linearized gradient descent, with the ``(1 - eta lambda)^s`` residual decay
written out), where the NTK of a two-layer ReLU network is computed in closed
form.  This experiment is the same statement, measured, on a network solving a
PDE.

The controlled family
---------------------
Solve the heat equation with a single-mode initial condition, one PDE per
frequency ``k``:

    u_t = alpha_k u_xx,   u(x, 0) = sin(k pi x),   u(0, t) = u(1, t) = 0,
    exact:  u(x, t) = sin(k pi x) exp(-alpha_k (k pi)^2 t).

with ``alpha_k = alpha_1 / k^2``.  That choice is the whole design of the
experiment and it is not cosmetic.  With a *fixed* alpha the mode-``k``
solution decays like ``exp(-alpha k^2 pi^2 t)``, so a high-``k`` target is
essentially zero over almost all of the domain -- and a lazy network that
simply outputs ``u = 0`` would score a *small* error.  Frequency and amplitude
would be confounded, and the "failure" would be unmeasurable.  Dividing alpha
by ``k^2`` cancels the ``k^2`` in the eigenvalue exactly:

    alpha_k (k pi)^2 = alpha_1 pi^2      for every k,

so every member of the family has the *identical* time envelope
``exp(-alpha_1 pi^2 t)`` and the identical O(1) amplitude.  The only thing that
varies across the sweep is the spatial frequency of the solution.  (It also
keeps the two residual terms ``u_t`` and ``alpha_k u_xx`` the same size at
every ``k``, so the loss is not silently reweighted along the sweep either.)

What is measured
----------------
(A) The NTK claim directly, without a PDE: fit ``sum_k sin(k pi x)`` (equal
    amplitudes, k = 1, 2, 4, 8, 16) by plain supervised regression and track
    each mode's coefficient over training.  The modes should converge in
    frequency order.  This isolates the optimization question from every other
    PINN difficulty.
(B) The cost on an actual PDE solve: final relative L2 error and time-to-fit
    (first step reaching 10% error) versus ``k``, for the plain MLP and for the
    Fourier-feature model, at identical width, depth, learning rate and step
    budget.
(C) Slow, or stuck? Rerun the plain net at 3x the step budget at the two
    highest frequencies. "It failed" is a cheap claim if the run was simply
    under-trained; the only way to earn it is to buy more compute and show it
    does not help.

What came out (seed 0, width 64, depth 4, 8000 Adam steps, 4k collocation)
-------------------------------------------------------------------------
Steps to reach 10% relative L2 error, and the final error:

    k       plain               + Fourier features
    1       200    (0.0026)     4600   (0.0424)
    2       400    (0.0013)     1400   (0.0144)
    4       800    (0.0011)      400   (0.0057)
    8       1800   (0.0024)      400   (0.0035)
    16      7400   (0.0399)      600   (0.0183)
    32      never  (0.5145)     4800   (0.0944)

Three things worth reading off that table, and the third one is the reason the
experiment is written this way.

1. Spectral bias is real and it is *graded*, not a cliff. The plain net's
   time-to-fit roughly doubles per octave (200 -> 400 -> 800 -> 1800) and then
   blows up (7400 -> never). This is the NTK prediction in wall-clock form.

2. The failure at k = 32 is genuine, and k = 16 is not. Given 3x the budget the
   plain net at k = 16 goes 0.0399 -> 0.0097 (it was merely *slow*), while at
   k = 32 it goes 0.5145 -> 0.2622 -- still 2.6x above the 10% target after
   24000 steps, and 2.8x worse than what the Fourier model reaches in a *third*
   of that budget. Compute rescues k = 16; it does not rescue k = 32. Reporting
   k = 16 as "a failure" would have been wrong, and only running the 3x check
   would have caught it.

3. The mitigation is a *trade*, not a free win -- and it cuts both ways. At
   k = 1 the Fourier model is 23x SLOWER than the plain net (4600 steps vs 200)
   and ends 16x less accurate. sigma_x = 5 hands the network frequencies the
   k = 1 solution simply does not contain, and the optimizer spends its budget
   chasing them. The crossover is at k = 4. At the other end, sigma_x = 5
   covers ~5 cycles/unit while k = 32 needs 16, so the Fourier model is
   straining at its own band edge too (0.0944, its worst high-k result). Both
   edges are a consequence of fixing sigma ONCE for the whole sweep rather than
   retuning per k -- which is deliberate: a mitigation you must retune for every
   frequency is not a mitigation, it is the answer smuggled into the prior.

Run:  python experiments/spectral_bias.py           # full sweep + figures
      python experiments/spectral_bias.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from common import plt, savefig, write_csv
from pinn import derivatives as D
from pinn.features import FourierMLP
from pinn.losses import boundary_points, initial_points, interior_points
from pinn.model import MLP, set_seed

# ---------------------------------------------------------------------------
# Problem family
# ---------------------------------------------------------------------------
ALPHA1 = 0.05
K_VALUES = (1, 2, 4, 8, 16, 32)
X_RANGE = (0.0, 1.0)
T_RANGE = (0.0, 1.0)

# Frequencies at which we re-run the plain net with 3x the step budget, to
# separate "slow" from "stuck" (part C). Chosen after seeing the 1x sweep.
LONG_CHECK_KS = (16, 32)

# One shared architecture for both models, so the comparison is honest.
WIDTH, DEPTH = 64, 4
STEPS = 8000
LR = 1e-3
N_INTERIOR, N_IC, N_BC = 4000, 400, 200
EVAL_EVERY = 200
FIT_TOL = 0.10  # "fitted" = relative L2 error below 10%

# Fourier-feature hyperparameters. Chosen ONCE for the whole sweep, not tuned
# per k -- a mitigation you have to retune for every frequency is not a
# mitigation. sigma_x = 5 covers ~5 cycles/unit, and sin(16 pi x) has 8; sigma_t
# = 1 because the solution is a slow exponential in t and does not want a wide
# temporal band.
N_FEATURES = 64
SIGMA = (5.0, 1.0)


def alpha_for(k: int) -> float:
    """alpha_k = alpha_1 / k^2 -- see the module docstring."""
    return ALPHA1 / k ** 2


def exact(x, t, k: int):
    """u(x, t) = sin(k pi x) exp(-alpha_k (k pi)^2 t), on numpy arrays.

    Written in the raw heat-solution form (an eigenfunction times its decay
    factor) rather than the simplified ``exp(-alpha_1 pi^2 t)``, so it is
    manifestly a solution of ``u_t = alpha_k u_xx``. ``tests/test_spectral_bias.py``
    checks it satisfies the PDE by finite differences, and checks the two forms
    agree.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    return np.sin(k * np.pi * x) * np.exp(-alpha_for(k) * (k * np.pi) ** 2 * t)


def initial_condition(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.sin(k * math.pi * x)


def residual(u: torch.Tensor, coords: torch.Tensor, k: int) -> torch.Tensor:
    """r = u_t - alpha_k u_xx."""
    return D.u_t(u, coords) - alpha_for(k) * D.u_xx(u, coords)


# ---------------------------------------------------------------------------
# Sine-basis spectral analysis
# ---------------------------------------------------------------------------
def sine_coefficients(u: np.ndarray, x: np.ndarray, n_modes: int = 24) -> np.ndarray:
    """Coefficients ``c_k = 2 int_0^1 u(x) sin(k pi x) dx``, k = 1..n_modes.

    The sine modes are the Dirichlet eigenfunctions of the Laplacian on [0, 1]
    -- the basis this whole problem family diagonalizes in -- so they are the
    right basis to ask "what frequencies has the network learned?" in.

    One preprocessing step is load-bearing: subtract the linear interpolant
    between the endpoints first, so the remainder vanishes at x = 0 and x = 1.
    A sine series of a function with nonzero endpoints converges only like 1/k,
    which would masquerade as high-frequency content for *any* function --
    including a straight line. (The same trap in Fourier form: an ``np.fft`` of
    a smooth non-periodic ramp leaks a fake 1/f^2 tail into every bin.) After
    the subtraction the coefficients decay at the function's true smoothness
    rate, and a smooth network reads as smooth.
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    u = u - u[0] - x * (u[-1] - u[0])
    ks = np.arange(1, n_modes + 1)
    return 2.0 * np.trapezoid(
        u[None, :] * np.sin(ks[:, None] * np.pi * x[None, :]), x, axis=1
    )


def model_sine_coefficients(model, n_modes: int = 24, n: int = 1024, t: float = 0.0):
    """Sine coefficients of the network's spatial profile at time ``t``."""
    x = np.linspace(X_RANGE[0], X_RANGE[1], n)
    coords = torch.tensor(
        np.stack([x, np.full_like(x, t)], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        u = model(coords).squeeze(1).numpy().astype(float)
    return sine_coefficients(u, x, n_modes)


# ---------------------------------------------------------------------------
# Models + evaluation
# ---------------------------------------------------------------------------
def build_model(kind: str, seed: int = 0):
    """``kind`` in {"plain", "fourier"} -- identical width/depth either way."""
    set_seed(seed)
    if kind == "plain":
        return MLP(in_dim=2, out_dim=1, width=WIDTH, depth=DEPTH, activation="tanh")
    if kind == "fourier":
        return FourierMLP(
            in_dim=2,
            out_dim=1,
            width=WIDTH,
            depth=DEPTH,
            n_features=N_FEATURES,
            sigma=SIGMA,
            feature_seed=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}")


def _eval_grid(nx=201, nt=101):
    x = np.linspace(X_RANGE[0], X_RANGE[1], nx)
    t = np.linspace(T_RANGE[0], T_RANGE[1], nt)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    return x, t, XX, TT


def predict(model, XX, TT):
    coords = np.stack([XX.ravel(), TT.ravel()], axis=1)
    with torch.no_grad():
        u = model(torch.tensor(coords, dtype=torch.float32)).numpy().reshape(XX.shape)
    return u


def rel_l2_error(model, k: int) -> float:
    """||u_pinn - u_exact|| / ||u_exact|| on a dense space-time grid.

    Note the denominator is O(1) for every k by construction (that is what the
    alpha_k = alpha_1/k^2 scaling buys), so this number is comparable across
    the sweep. A network that gave up and output zero would score ~1.0 at any k.
    """
    _, _, XX, TT = _eval_grid()
    return float(
        np.linalg.norm(predict(model, XX, TT) - exact(XX, TT, k))
        / np.linalg.norm(exact(XX, TT, k))
    )


# ---------------------------------------------------------------------------
# (B)/(C) PINN training on the mode-k problem
# ---------------------------------------------------------------------------
def train_pinn(k, kind, steps=STEPS, seed=0, verbose=False):
    """Train the mode-k heat PINN. Returns (model, history[(step, loss, relL2)])."""
    model = build_model(kind, seed=seed)
    gen = torch.Generator().manual_seed(seed)

    interior = interior_points(N_INTERIOR, X_RANGE, T_RANGE, gen)
    ic = initial_points(N_IC, X_RANGE, T_RANGE[0], gen)
    ic_target = initial_condition(ic[:, 0:1], k)
    left, right = boundary_points(N_BC, X_RANGE, T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    bc_target = torch.zeros(bc.shape[0], 1)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    for step in range(steps + 1):
        opt.zero_grad()
        r = residual(model(interior), interior, k)
        loss = (
            torch.mean(r ** 2)
            + torch.mean((model(ic) - ic_target) ** 2)
            + torch.mean((model(bc) - bc_target) ** 2)
        )
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == steps:
            err = rel_l2_error(model, k)
            history.append((step, float(loss.item()), err))
            if verbose and step % (EVAL_EVERY * 5) == 0:
                print(f"    step {step:6d}  loss {loss.item():.3e}  relL2 {err:.4f}")
    return model, history


def time_to_fit(history, tol=FIT_TOL):
    """First step whose relative L2 error is below ``tol``; None if never."""
    for step, _loss, err in history:
        if err < tol:
            return step
    return None


def run_pinn_sweep(seed=0, steps=STEPS):
    """Returns (rows, runs) where runs[(kind, k)] = (model, history).

    The trained models are kept so the k=16 failure figure is drawn from the
    *same* run that produced the k=16 row of the table -- retraining it would
    cost five minutes and let the figure and the table quietly disagree.
    """
    rows = []
    runs = {}
    for kind in ("plain", "fourier"):
        for k in K_VALUES:
            t0 = time.time()
            model, hist = train_pinn(k, kind, steps=steps, seed=seed)
            runs[(kind, k)] = (model, hist)
            err = hist[-1][2]
            ttf = time_to_fit(hist)
            secs = time.time() - t0
            rows.append(
                {
                    "model": kind,
                    "k": k,
                    "alpha": f"{alpha_for(k):.6e}",
                    "rel_l2": f"{err:.6e}",
                    "steps_to_fit": "" if ttf is None else ttf,
                    "seconds": f"{secs:.1f}",
                }
            )
            print(
                f"  {kind:7s} k={k:3d}  relL2 {err:.4f}  "
                f"steps-to-10% {'never' if ttf is None else ttf}  ({secs:.0f}s)"
            )
    return rows, runs


# ---------------------------------------------------------------------------
# (A) The NTK claim without a PDE: which modes converge first?
# ---------------------------------------------------------------------------
def regression_diagnostic(steps=10000, n=256, eval_every=100, seed=0):
    """Fit ``sum_k sin(k pi x)`` (equal amplitude) and watch the modes arrive.

    Pure supervised regression -- no residual, no collocation, no boundary
    terms. If the plain network still learns k=1 long before k=16 here, the
    ordering is a property of gradient descent on the network, not an artifact
    of the PINN loss. That is exactly the NTK prediction.
    """
    x = np.linspace(X_RANGE[0], X_RANGE[1], n)
    coords = torch.tensor(
        np.stack([x, np.zeros_like(x)], axis=1), dtype=torch.float32
    )
    target = torch.zeros(n, 1)
    for k in K_VALUES:
        target += torch.sin(k * math.pi * torch.tensor(x, dtype=torch.float32)).reshape(
            -1, 1
        )

    rows = []
    for kind in ("plain", "fourier"):
        model = build_model(kind, seed=seed)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for step in range(steps + 1):
            opt.zero_grad()
            loss = torch.mean((model(coords) - target) ** 2)
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                c = model_sine_coefficients(model, n_modes=max(K_VALUES))
                for k in K_VALUES:
                    rows.append(
                        {
                            "model": kind,
                            "step": step,
                            "k": k,
                            "coeff": f"{c[k - 1]:.6f}",  # target value is 1.0
                        }
                    )
        print(f"  {kind:7s} final coefficients: "
              + ", ".join(f"k{k}={c[k-1]:.3f}" for k in K_VALUES))
    return rows


def figure_regression(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, kind, title in zip(
        axes,
        ("plain", "fourier"),
        ("plain tanh MLP", f"+ Fourier features (sigma={SIGMA})"),
    ):
        for i, k in enumerate(K_VALUES):
            sel = [r for r in rows if r["model"] == kind and int(r["k"]) == k]
            steps = [int(r["step"]) for r in sel]
            coeff = [float(r["coeff"]) for r in sel]
            ax.plot(steps, coeff, color=f"C{i}", label=f"k = {k}")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Adam step")
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=100)
    axes[0].set_ylabel("learned coefficient of sin(k pi x)")
    axes[0].legend(loc="lower right", ncol=2)
    fig.suptitle(
        "Spectral bias, no PDE: regressing sum_k sin(k pi x) with equal amplitudes.\n"
        "The plain net learns the modes in frequency order; the target coefficient is 1.",
        y=1.12,
    )
    savefig(fig, "spectral_regression.png")


# ---------------------------------------------------------------------------
# Figures for the PINN sweep and the failed run
# ---------------------------------------------------------------------------
def figure_pinn_sweep(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for i, kind in enumerate(("plain", "fourier")):
        sel = [r for r in rows if r["model"] == kind]
        ks = [int(r["k"]) for r in sel]
        err = [float(r["rel_l2"]) for r in sel]
        axes[0].plot(ks, err, "o-", color=f"C{i}", label=kind)

        fitted = [(int(r["k"]), int(r["steps_to_fit"])) for r in sel if r["steps_to_fit"]]
        never = [int(r["k"]) for r in sel if not r["steps_to_fit"]]
        if fitted:
            axes[1].plot(
                [k for k, _ in fitted], [s for _, s in fitted], "o-", color=f"C{i}",
                label=kind,
            )
        if never:
            axes[1].plot(
                never, [STEPS * 1.15] * len(never), "x", color=f"C{i}", ms=9, mew=2,
            )

    axes[0].axhline(1.0, color="k", ls=":", lw=0.8)
    axes[0].text(1.1, 1.05, "error of predicting u = 0", fontsize=7, color="0.4")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(K_VALUES)
    axes[0].set_xticklabels(K_VALUES)
    axes[0].set_xlabel("initial-condition frequency k")
    axes[0].set_ylabel("final relative L2 error")
    axes[0].set_title(f"accuracy after {STEPS} Adam steps")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].axhline(STEPS, color="k", ls="--", lw=0.8)
    axes[1].text(1.1, STEPS * 1.22, "x = never reached 10% (budget exhausted)",
                 fontsize=7, color="0.4")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(K_VALUES)
    axes[1].set_xticklabels(K_VALUES)
    axes[1].set_xlabel("initial-condition frequency k")
    axes[1].set_ylabel("Adam steps to reach 10% error")
    axes[1].set_title("time-to-fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        "Heat equation, single-mode IC sin(k pi x), alpha_k = alpha_1/k^2 "
        "(so every target has the same amplitude and time envelope)",
        y=1.06,
    )
    fig.tight_layout()
    savefig(fig, "spectral_pinn.png")


def figure_failure_and_fix(k_fail, runs, long_runs, t_slice=0.5):
    """Three panels: what the failed run looks like, why, and does compute fix it.

    ``long_runs[k] = (model, history)`` is the plain net at 3x the step budget.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))

    # (a) the spatial profile at k_fail: what "failing" actually looks like
    model_plain, hist_plain = runs[("plain", k_fail)]
    model_fourier, hist_fourier = runs[("fourier", k_fail)]
    x = np.linspace(X_RANGE[0], X_RANGE[1], 600)
    tt = np.full_like(x, t_slice)
    axes[0].plot(x, exact(x, tt, k_fail), "k-", lw=1.6, label="exact")
    for model, kind, c in ((model_plain, "plain", "C0"), (model_fourier, "fourier", "C1")):
        coords = torch.tensor(np.stack([x, tt], axis=1), dtype=torch.float32)
        with torch.no_grad():
            u = model(coords).squeeze(1).numpy()
        axes[0].plot(x, u, color=c, lw=1.3, label=kind)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(f"u(x, t={t_slice})")
    axes[0].set_title(f"k = {k_fail}: the plain net after {STEPS} steps", loc="left")
    axes[0].legend(fontsize=7)

    # (b) error trajectories at k_fail, incl. the 3x-budget plain run
    _, hist_long = long_runs[k_fail]
    for hist, label, style in (
        (hist_plain, f"plain ({STEPS} steps)", "C0-"),
        (hist_long, f"plain ({3 * STEPS} steps, 3x)", "C0--"),
        (hist_fourier, f"Fourier ({STEPS} steps)", "C1-"),
    ):
        axes[1].plot([h[0] for h in hist], [h[2] for h in hist], style, lw=1.4,
                     label=label)
    axes[1].axhline(FIT_TOL, color="k", ls=":", lw=0.8)
    axes[1].axvline(STEPS, color="0.75", ls="-", lw=0.8)
    axes[1].text(STEPS * 1.05, 1.3, "budget", fontsize=7, color="0.5")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Adam step")
    axes[1].set_ylabel("relative L2 error")
    axes[1].set_title(f"k = {k_fail}: does more compute rescue it?", loc="left")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.25)

    # (c) slow vs stuck: 1x vs 3x budget at each long-checked k
    ks = sorted(long_runs)
    width = 0.35
    idx = np.arange(len(ks))
    err_1x = [runs[("plain", k)][1][-1][2] for k in ks]
    err_3x = [long_runs[k][1][-1][2] for k in ks]
    axes[2].bar(idx - width / 2, err_1x, width, color="C0", label=f"{STEPS} steps")
    axes[2].bar(idx + width / 2, err_3x, width, color="C0", alpha=0.45,
                label=f"{3 * STEPS} steps (3x)")
    axes[2].axhline(FIT_TOL, color="k", ls=":", lw=0.8)
    axes[2].text(-0.45, FIT_TOL * 1.15, "10% target", fontsize=7, color="0.4")
    axes[2].set_xticks(idx)
    axes[2].set_xticklabels([f"k = {k}" for k in ks])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("final relative L2 error")
    axes[2].set_title("slow, or stuck?", loc="left")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, axis="y", which="both", alpha=0.25)

    fig.suptitle(
        "Spectral bias on a PDE solve: the plain net degrades with frequency, "
        "and buying compute stops working",
        y=1.04,
    )
    fig.tight_layout()
    savefig(fig, "spectral_fix.png")


# ---------------------------------------------------------------------------
def main(quick=False):
    if quick:
        print("[quick] smoke run")
        _m, hist = train_pinn(k=4, kind="plain", steps=200, verbose=True)
        print("  rel L2:", hist[-1][2])
        _m, hist = train_pinn(k=4, kind="fourier", steps=200, verbose=True)
        print("  rel L2:", hist[-1][2])
        rows = regression_diagnostic(steps=300, eval_every=100)
        print(f"  {len(rows)} regression rows")
        return

    print("=" * 70)
    print("(A) Spectral bias without a PDE: per-mode convergence in regression")
    print("=" * 70)
    reg_rows = regression_diagnostic()
    write_csv("spectral_regression.csv", ["model", "step", "k", "coeff"], reg_rows)
    figure_regression(reg_rows)

    print("\n" + "=" * 70)
    print("(B) PINN sweep over the IC frequency k")
    print("=" * 70)
    sweep_rows, runs = run_pinn_sweep()
    write_csv(
        "spectral_pinn.csv",
        ["model", "k", "alpha", "rel_l2", "steps_to_fit", "seconds"],
        sweep_rows,
    )
    figure_pinn_sweep(sweep_rows)

    k_fail = max(K_VALUES)
    print("\n" + "=" * 70)
    print("(C) Slow, or stuck? Rerun the plain net at 3x the step budget")
    print("=" * 70)
    # Reuse the sweep's own runs, so the figure and the table cannot disagree.
    long_runs = {}
    for k in LONG_CHECK_KS:
        print(f"  plain k={k}, 3x budget ({3 * STEPS} steps):")
        long_runs[k] = train_pinn(k, "plain", steps=3 * STEPS, verbose=True)
        err_1x = runs[("plain", k)][1][-1][2]
        err_3x = long_runs[k][1][-1][2]
        print(f"    k={k:2d}: plain {err_1x:.4f} at {STEPS} steps -> "
              f"{err_3x:.4f} at {3 * STEPS}  "
              f"(fourier at {STEPS}: {runs[('fourier', k)][1][-1][2]:.4f})")

    write_csv(
        "spectral_long.csv",
        ["model", "k", "step", "loss", "rel_l2"],
        [
            {"model": name, "k": k, "step": s,
             "loss": f"{l:.6e}", "rel_l2": f"{e:.6e}"}
            for k in LONG_CHECK_KS
            for name, hist in (
                ("plain", runs[("plain", k)][1]),
                ("plain_3x", long_runs[k][1]),
                ("fourier", runs[("fourier", k)][1]),
            )
            for s, l, e in hist
        ],
    )
    figure_failure_and_fix(k_fail, runs, long_runs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
