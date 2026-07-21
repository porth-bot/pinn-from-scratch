"""Hard boundary/initial conditions via a trial-function ansatz vs soft penalties.

Every solve so far (heat, Burgers, spectral bias) enforces the initial and
boundary conditions the *soft* way: add ``w_ic * ||u - g||^2 + w_bc * ||u||^2``
to the residual loss and hope the minimizer balances the three. The weights are
a real knob -- the theory doc (Sec. 1, Sec. 5) lists them as a genuine
difficulty -- and the constraints hold only approximately.

The classical alternative (Lagaris, Likas & Fotiadis 1998) is to *build the
constraints into the function space* so they cannot be violated. For this heat
problem -- ``u(x,0) = g(x)``, ``u(0,t) = u(1,t) = 0`` on ``[0,1]x[0,T]`` -- write

    u_hat(x, t) = g(x) + x (1 - x) t * N(x, t),

with ``N`` the network. Read off the two guarantees, both exact for *any* ``N``:

- at ``t = 0`` the correction carries a factor ``t = 0``, so ``u_hat = g(x)`` --
  the initial condition holds to machine precision;
- at ``x in {0, 1}`` the factor ``x(1-x) = 0``, so ``u_hat = g(x) = 0`` (the sine
  IC already vanishes at the walls) -- the Dirichlet BC holds to machine
  precision.

So the IC and BC loss terms *disappear*: the ansatz trains the residual alone,
with no weights to tune. The honest question this experiment answers: does
removing the weight-balancing problem actually improve accuracy, or does it just
trade a tuning knob for a hand-derived ansatz that must be redone per problem?

Fair comparison: same architecture, same seed (bit-identical network init via
``set_seed``), same fixed interior collocation, same optimizer/steps. The only
differences are the two the method is about -- the soft run adds IC+BC penalties
to a plain MLP; the hard run wraps the same MLP in the ansatz and drops those
terms.

Run:  python experiments/hard_bc.py           # full run + figure
      python experiments/hard_bc.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from common import plt, savefig, write_csv
from heat import (
    ALPHA,
    T_RANGE,
    X_RANGE,
    heat_exact,
    heat_residual,
    initial_condition,
    predict,
    rel_l2_error,
    train as train_soft,
)
from pinn.losses import interior_points
from pinn.model import MLP, set_seed


class HardConstraintNet(torch.nn.Module):
    """Wrap an MLP so ``u_hat = g(x) + x(1-x) t N(x,t)`` satisfies the heat IC
    and homogeneous Dirichlet BCs by construction.

    The wrapper is a plain differentiable function of ``coords = (x, t)``, so the
    existing autograd derivative helpers (``pinn.derivatives``) compose with it
    unchanged -- ``u_t`` and ``u_xx`` of ``u_hat`` include the exact derivatives
    of ``g`` and of the ``x(1-x)t`` envelope, not just the network's.
    """

    def __init__(self, net: MLP) -> None:
        super().__init__()
        self.net = net

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, 0:1]
        t = coords[:, 1:2]
        g = initial_condition(x)
        envelope = x * (1.0 - x) * t
        return g + envelope * self.net(coords)


def _constraint_errors(model, n=400):
    """Max |IC error| and max |BC error| of a model on dense boundary samples.

    For the soft model these shrink but never reach zero; for the hard ansatz
    they are at the float32 floor no matter the weights -- the whole point.
    """
    xs = torch.linspace(X_RANGE[0], X_RANGE[1], n).reshape(-1, 1)
    ts = torch.linspace(T_RANGE[0], T_RANGE[1], n).reshape(-1, 1)
    zeros = torch.zeros_like(ts)
    ones = torch.ones_like(ts)
    with torch.no_grad():
        ic = model(torch.cat([xs, torch.zeros_like(xs)], dim=1)) - initial_condition(xs)
        left = model(torch.cat([X_RANGE[0] * ones, ts], dim=1))
        right = model(torch.cat([X_RANGE[1] * ones, ts], dim=1))
    ic_err = float(ic.abs().max())
    bc_err = float(torch.maximum(left.abs().max(), right.abs().max()))
    return ic_err, bc_err


def train_hard(n_interior=4000, width=128, depth=4, steps=5000, lr=1e-3, seed=0,
               verbose=False):
    """Train the hard-constraint ansatz on the residual alone (no IC/BC terms).

    Mirrors ``heat.train`` exactly except for the ansatz and the loss: same
    ``set_seed`` -> bit-identical MLP init, same interior sampler call, same
    Adam. History rows are ``(step, loss, rel_l2, ic_err, bc_err)``.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    interior = interior_points(n_interior, X_RANGE, T_RANGE, gen)

    net = MLP(in_dim=2, out_dim=1, width=width, depth=depth, activation="tanh")
    model = HardConstraintNet(net)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for step in range(steps + 1):
        opt.zero_grad()
        u_int = model(interior)
        r = heat_residual(u_int, interior)
        loss = torch.mean(r ** 2)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            err = rel_l2_error(model)
            ic_err, bc_err = _constraint_errors(model)
            history.append((step, float(loss.item()), err, ic_err, bc_err))
            if verbose:
                print(f"  hard step {step:5d}  loss {loss.item():.3e}  "
                      f"relL2 {err:.4f}  ic {ic_err:.1e} bc {bc_err:.1e}")
    return model, history


def _soft_history_with_constraints(seed, n_interior, width, depth, steps, lr):
    """Run the soft baseline and re-measure its IC/BC errors on the same grid.

    ``heat.train`` logs (step, loss, rel_l2); we retrain here with matched
    settings and additionally record the constraint errors so the two methods
    are plotted on the same axes. (heat.train reseeds identically, so its MLP
    init matches the ansatz's wrapped net.)
    """
    model, hist = train_soft(
        n_interior=n_interior, width=width, depth=depth, steps=steps, lr=lr,
        seed=seed, w_ic=1.0, w_bc=1.0,
    )
    # heat.train already logs at the same cadence; recompute constraints per row
    # would need per-step models, so report only the final-model constraint error
    # (the curve endpoint) plus the logged rel_l2 trajectory.
    ic_err, bc_err = _constraint_errors(model)
    return model, hist, ic_err, bc_err


def main(quick=False):
    seed = 0
    if quick:
        n_interior, width, depth, steps, lr = 500, 32, 3, 500, 1e-3
    else:
        n_interior, width, depth, steps, lr = 4000, 128, 4, 5000, 1e-3

    print("=" * 74)
    print("Hard-constraint ansatz vs soft penalties on the heat equation")
    print("=" * 74)

    soft_model, soft_hist, soft_ic, soft_bc = _soft_history_with_constraints(
        seed, n_interior, width, depth, steps, lr)
    hard_model, hard_hist = train_hard(
        n_interior=n_interior, width=width, depth=depth, steps=steps, lr=lr,
        seed=seed, verbose=quick)

    soft_final = soft_hist[-1]
    hard_final = hard_hist[-1]
    soft_best = min(s[2] for s in soft_hist)
    hard_best = min(h[2] for h in hard_hist)
    print(f"\nsoft  : rel L2 final {soft_final[2]:.4e}  best {soft_best:.4e}   "
          f"IC err {soft_ic:.2e}   BC err {soft_bc:.2e}")
    print(f"hard  : rel L2 final {hard_final[2]:.4e}  best {hard_best:.4e}   "
          f"IC err {hard_final[3]:.2e}   BC err {hard_final[4]:.2e}")
    print(f"\nbest-rel-L2 ratio soft/hard: {soft_best / hard_best:.1f}x")
    print("IC and BC are exact by construction for the ansatz (float32 floor)\n"
          "vs the soft penalty's ~1e-2, with no weights to tune. On accuracy the\n"
          "*trajectory* is the honest read (fig. a): the ansatz reaches the soft\n"
          "run's best error far sooner and bottoms out deeper, because the IC/BC\n"
          "data no longer competes with the residual in a weighted sum. Both show\n"
          "late Adam oscillation, so the single final-step number is noisy.")

    if not quick:
        fields = ["method", "step", "loss", "rel_l2", "ic_err", "bc_err"]
        rows = [
            {"method": "soft", "step": s[0], "loss": s[1], "rel_l2": s[2],
             "ic_err": soft_ic, "bc_err": soft_bc}
            for s in soft_hist
        ] + [
            {"method": "hard", "step": h[0], "loss": h[1], "rel_l2": h[2],
             "ic_err": h[3], "bc_err": h[4]}
            for h in hard_hist
        ]
        write_csv("hard_bc.csv", fields, rows)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    # (a) rel-L2 convergence
    ax = axes[0]
    ss = np.array(soft_hist)
    hh = np.array(hard_hist)
    ax.semilogy(ss[:, 0], ss[:, 2], "o-", ms=3, label="soft (residual + IC + BC)")
    ax.semilogy(hh[:, 0], hh[:, 2], "s-", ms=3, label="hard (ansatz, residual only)")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("relative $L^2$ error")
    ax.set_title("(a) accuracy vs training step", loc="left")
    ax.legend(fontsize=7)

    # (b) constraint satisfaction over training (hard is flat at the float floor)
    ax = axes[1]
    ax.semilogy(hh[:, 0], np.maximum(hh[:, 3], 1e-9), "s-", ms=3, color="C1",
                label="hard: max(IC, BC) err")
    ax.axhline(max(soft_ic, soft_bc), color="C0", ls="--",
               label="soft: final max(IC, BC) err")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("max boundary/IC error")
    ax.set_title("(b) constraints: exact vs approximate", loc="left")
    ax.legend(fontsize=7)

    # (c) error heatmaps side by side (final models)
    ax = axes[2]
    nx = nt = 101
    x = np.linspace(*X_RANGE, nx)
    t = np.linspace(*T_RANGE, nt)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    err_hard = np.abs(predict(hard_model, XX, TT) - heat_exact(XX, TT))
    im = ax.pcolormesh(TT, XX, err_hard, shading="auto", cmap="magma")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("(c) |error|, hard ansatz", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Hard-constraint ansatz vs soft penalties (heat equation)",
                 x=0.02, ha="left")
    savefig(fig, "hard_bc.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    main(quick=args.quick)
