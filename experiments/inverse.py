"""The inverse problem: recover the diffusivity from sparse noisy data.

Every experiment so far solved a *forward* problem -- given the PDE, its
coefficients, and its boundary data, produce the field -- and reported honestly
that a classical solver does it better (`experiments/crank_nicolson.py`, theory
doc Sec. 6). This is the other direction, and it is the setting PINNs are
actually good at: the field is unknown *and so is a coefficient of the
equation*, and all that is available is a handful of noisy point measurements.

Problem.  On ``x in [0, 1]``, ``t in [0, 1]``, the same three-mode heat problem
as `experiments/heat.py`, but ``alpha`` is now unknown. Given ``N`` scattered
observations

    y_i = u(x_i, t_i) + eps_i,   eps_i ~ N(0, sigma^2),

recover ``alpha`` (true value 0.05) along with the field, by minimizing

    L(theta, alpha) = w_r * mean( u_t - alpha u_xx )^2  +  w_d * mean( u_theta - y )^2

over the network weights **and** ``alpha`` jointly. Nothing else is supplied:
no initial condition, no boundary condition. That is not a simplification, it
is the point -- in an inverse problem the data replaces them, and a
finite-difference solver has nothing to march from.

Why this is a one-line change to a PINN and a rewrite for a solver.  ``alpha``
enters the loss through a term that is already being differentiated by autograd,
so making it a parameter costs one ``nn.Parameter`` and an extra entry in the
optimizer's list. The classical route to the same answer is an outer
optimization loop over full forward solves (or a hand-derived adjoint), which
is exactly the machinery a PINN sidesteps.

Parameterization.  We optimize ``log alpha``, so ``alpha = exp(.) > 0`` always
and one optimizer step is a *relative* change in ``alpha`` regardless of its
magnitude -- necessary here because the initialization is deliberately wrong by
a factor of 4 and gradients in ``alpha`` scale with ``alpha`` itself.

What is identifiable, and what is not.  The heat equation has an exact
degeneracy: ``u(x, t; alpha) = u(x, alpha t; 1)``, so scaling ``alpha`` is the
same as rescaling time. ``alpha`` is therefore recoverable only because the
observations carry *absolute* time labels, and only to the extent that they
span enough time for the decay envelope ``exp(-alpha (k pi)^2 t)`` to be
visible. The third sweep below measures exactly that: shrink the observation
window toward ``t = 0`` and the recovery degrades, because a short window sees
the field's value but barely its decay.

The three measurements.
    (1) recovery vs noise ``sigma in {0, 0.01, 0.05, 0.1}`` at ``N = 200``;
    (2) recovery vs data volume ``N in {25, 50, 100, 200, 400}`` at
        ``sigma = 0.02``;
    (3) recovery vs the observation window ``t <= t_max`` for
        ``t_max in {0.1, 0.25, 0.5, 1.0}`` at ``N = 200``, ``sigma = 0.02``
        -- the identifiability study.

Every row also reports the field's own relative L2 error against the exact
solution, since a recovered ``alpha`` is only meaningful if the field it came
with is a solution.

Run:  python experiments/inverse.py            # all three sweeps + figure
      python experiments/inverse.py --figures   # replay the figure from logs
      python experiments/inverse.py --quick     # tiny run, smoke check
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import plt, read_csv, savefig, write_csv
from heat import (
    ALPHA,
    T_RANGE,
    X_RANGE,
    heat_exact,
    predict,
    rel_l2_error,
)
from pinn import derivatives as D
from pinn.losses import interior_points
from pinn.model import MLP, set_seed

# Deliberately wrong by 4x, and on the high side: an initialization that
# over-diffuses smooths the field out, which is the harder direction to climb
# back from than starting too stiff.
ALPHA_INIT = 4.0 * ALPHA

DEFAULTS = dict(n_interior=2000, width=64, depth=4, steps=6000, lr=3e-3)


def observations(n, sigma, gen, t_max=1.0, t_min=0.0):
    """``n`` scattered noisy samples of the exact solution -> (coords, values).

    Points are uniform over ``x in X_RANGE`` and ``t in [t_min, t_max]``, and
    the values are the exact Fourier solution plus Gaussian noise. Returned
    coordinates carry **no** ``requires_grad``: they are data, not collocation
    points, and no derivative of the network is taken at them.
    """
    x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * torch.rand(n, 1, generator=gen)
    t = t_min + (t_max - t_min) * torch.rand(n, 1, generator=gen)
    coords = torch.cat([x, t], dim=1)
    clean = torch.tensor(
        heat_exact(x.numpy(), t.numpy()), dtype=torch.float32
    ).reshape(n, 1)
    noise = sigma * torch.randn(n, 1, generator=gen) if sigma > 0 else torch.zeros(n, 1)
    return coords, clean + noise


def inverse_residual(u, coords, log_alpha):
    """``r = u_t - exp(log_alpha) u_xx`` -- the residual with a *learnable* alpha.

    Identical to :func:`heat.heat_residual` except that ``alpha`` is a tensor in
    the autograd graph, so ``dL/d log_alpha`` comes out of the same backward
    pass that trains the network. On the exact solution this residual equals
    ``(ALPHA - alpha) u_xx`` exactly, which is the sensitivity the recovery
    lives on and what ``tests/test_inverse.py`` checks first.
    """
    return D.u_t(u, coords) - torch.exp(log_alpha) * D.u_xx(u, coords)


def train_inverse(
    n_obs=200,
    sigma=0.02,
    t_max=1.0,
    n_interior=2000,
    width=64,
    depth=4,
    steps=6000,
    lr=3e-3,
    w_data=10.0,
    seed=0,
    verbose=False,
):
    """Joint recovery of the field and ``alpha``; returns (model, alpha, history).

    ``w_data = 10`` weights the data term above the residual: with only a few
    hundred noisy points, the residual term is satisfiable by *any* solution of
    the PDE (including ``u = 0``, for which every alpha is consistent), so the
    data has to be the term that selects which one. The residual's job is to
    say that whatever the data suggests must also be a solution -- that is what
    ties the field's shape to alpha and makes alpha observable at all.

    History rows are ``(step, loss, alpha_hat, rel_l2)``.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1000)

    interior = interior_points(n_interior, X_RANGE, T_RANGE, gen)
    obs_x, obs_y = observations(n_obs, sigma, gen, t_max=t_max)

    model = MLP(in_dim=2, out_dim=1, width=width, depth=depth, activation="tanh")
    log_alpha = torch.nn.Parameter(torch.tensor(float(np.log(ALPHA_INIT))))
    opt = torch.optim.Adam(list(model.parameters()) + [log_alpha], lr=lr)

    history = []
    for step in range(steps + 1):
        opt.zero_grad()
        r = inverse_residual(model(interior), interior, log_alpha)
        loss_r = torch.mean(r ** 2)
        loss_d = torch.mean((model(obs_x) - obs_y) ** 2)
        loss = loss_r + w_data * loss_d
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == steps:
            alpha_hat = float(torch.exp(log_alpha.detach()))
            history.append((step, float(loss.item()), alpha_hat, rel_l2_error(model)))
            if verbose:
                print(
                    f"  step {step:5d}  loss {loss.item():.3e} "
                    f"(r {loss_r.item():.2e} data {loss_d.item():.2e})  "
                    f"alpha {alpha_hat:.5f}  relL2 {history[-1][3]:.4f}"
                )
    return model, float(torch.exp(log_alpha.detach())), history


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------
def _row(tag, alpha_hat, rel_l2, secs, **kw):
    row = {
        "sweep": tag,
        "alpha_true": f"{ALPHA:.6f}",
        "alpha_init": f"{ALPHA_INIT:.6f}",
        "alpha_hat": f"{alpha_hat:.6f}",
        "rel_err": f"{abs(alpha_hat - ALPHA) / ALPHA:.6f}",
        "rel_l2": f"{rel_l2:.6e}",
        "seconds": f"{secs:.1f}",
    }
    row.update({k: str(v) for k, v in kw.items()})
    return row


def sweep(tag, cases, steps, n_interior, verbose=False):
    """Run one sweep: ``cases`` is a list of kwargs dicts for ``train_inverse``."""
    rows = []
    for case in cases:
        t0 = time.time()
        _, alpha_hat, hist = train_inverse(
            steps=steps, n_interior=n_interior, verbose=verbose, **case
        )
        rows.append(_row(tag, alpha_hat, hist[-1][3], time.time() - t0, **case))
        print(
            f"  {tag:8s} {case}  alpha={alpha_hat:.5f} "
            f"(rel err {abs(alpha_hat - ALPHA) / ALPHA:6.2%})  "
            f"relL2={hist[-1][3]:.4f}  ({time.time() - t0:.0f}s)"
        )
    return rows


FIELDNAMES = [
    "sweep", "alpha_true", "alpha_init", "alpha_hat", "rel_err", "rel_l2",
    "seconds", "n_obs", "sigma", "t_max",
]


def _pad(rows):
    """CSV needs every column present; the sweeps vary in which knob they set."""
    for r in rows:
        for k, default in (("n_obs", 200), ("sigma", 0.02), ("t_max", 1.0)):
            r.setdefault(k, str(default))
    return rows


# ---------------------------------------------------------------------------
def figure(rows, trace):
    noise = sorted((r for r in rows if r["sweep"] == "noise"), key=lambda r: float(r["sigma"]))
    data = sorted((r for r in rows if r["sweep"] == "data"), key=lambda r: int(r["n_obs"]))
    window = sorted((r for r in rows if r["sweep"] == "window"), key=lambda r: float(r["t_max"]))

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.3))

    step, alpha_trace = trace
    axes[0].plot(step, alpha_trace, "o-", ms=3, label=r"$\hat\alpha$ during training")
    axes[0].axhline(ALPHA, color="k", ls="--", lw=1, label=r"true $\alpha=0.05$")
    axes[0].axhline(ALPHA_INIT, color="C3", ls=":", lw=1, label=r"init $4\alpha$")
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel(r"$\hat\alpha$")
    axes[0].set_title(r"From a 4x-wrong start ($N=200$, $\sigma=0.02$)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    panels = (
        (1, noise, "sigma", float, r"observation noise $\sigma$", r"Noise ($N=200$)", "C0"),
        (2, data, "n_obs", int, "observations $N$", r"Data volume ($\sigma=0.02$)", "C1"),
        (3, window, "t_max", float, r"observation window $t \leq t_{max}$",
         "Identifiability: a short window\nsees the field, not its decay", "C2"),
    )
    for idx, rows_, key, cast, xlabel, title, colour in panels:
        ax = axes[idx]
        ax.plot(
            [cast(r[key]) for r in rows_],
            [100 * float(r["rel_err"]) for r in rows_], "o-", color=colour,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\hat\alpha - \alpha| / \alpha$  (%)")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Inverse heat problem: recovering alpha from sparse noisy data "
        "(one seed per cell)", y=1.04,
    )
    savefig(fig, "inverse.png")


def figures_from_committed():
    """Replay ``figures/inverse.png`` from the committed CSVs -- no training.

    The repo's replay contract (``experiments/reproduce_figures.py``, checked by
    ``tests/test_reproduce_figures.py``): every shipped figure regenerates from
    committed artifacts. This one is all curves, so the two CSVs suffice -- no
    checkpoint is needed, unlike the field figures.
    """
    rows = read_csv("inverse.csv")
    trace_rows = read_csv("inverse_trace.csv")
    figure(
        rows,
        ([int(r["step"]) for r in trace_rows], [float(r["alpha_hat"]) for r in trace_rows]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="replay the figure from logs")
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke check")
    args = ap.parse_args()

    if args.figures:
        figures_from_committed()
        return

    steps = 400 if args.quick else DEFAULTS["steps"]
    n_interior = 400 if args.quick else DEFAULTS["n_interior"]

    print("Inverse heat problem: recover alpha (true 0.05) from noisy point data")
    rows = []
    rows += sweep(
        "noise",
        [dict(n_obs=200, sigma=s) for s in (0.0, 0.01, 0.05, 0.1)],
        steps, n_interior,
    )
    rows += sweep(
        "data",
        [dict(n_obs=n, sigma=0.02) for n in (25, 50, 100, 200, 400)],
        steps, n_interior,
    )
    rows += sweep(
        "window",
        [dict(n_obs=200, sigma=0.02, t_max=tm) for tm in (0.1, 0.25, 0.5, 1.0)],
        steps, n_interior,
    )
    write_csv("inverse.csv", FIELDNAMES, _pad(rows))

    # The headline trace: one run at the reference setting, logged per checkpoint
    # so the figure's left panel replays without retraining.
    _, alpha_hat, hist = train_inverse(
        n_obs=200, sigma=0.02, steps=steps, n_interior=n_interior, verbose=True
    )
    write_csv(
        "inverse_trace.csv",
        ["step", "loss", "alpha_hat", "rel_l2"],
        [
            {"step": s, "loss": f"{l:.6e}", "alpha_hat": f"{a:.6f}", "rel_l2": f"{e:.6e}"}
            for s, l, a, e in hist
        ],
    )
    print(f"reference run: alpha_hat = {alpha_hat:.5f} (true {ALPHA})")
    figure(_pad(rows), ([h[0] for h in hist], [h[2] for h in hist]))


if __name__ == "__main__":
    main()
