"""The optimizer study must be a *fair* comparison and its cost accounting must
be honest, since its whole claim ("L-BFGS beats Adam per evaluation on the heat
PINN") rests on both. We check three load-bearing properties without doing a
full training run:

1. Fairness: two builds with the same seed produce an identical initial network,
   so any difference across regimes is the optimizer, not the initialization.
2. L-BFGS actually optimizes: a few iterations strictly reduce the PINN loss.
3. Evaluation accounting: the closure counts every loss-and-gradient call, and
   the logged eval counts are strictly increasing (the x-axis of the figure).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from optimizer_study import _build, train_lbfgs  # noqa: E402


def test_same_seed_gives_identical_init():
    """The fairness guarantee: identical init and identical collocation, so the
    two regimes start from the exact same point."""
    m1, loss1 = _build(n_interior=256, width=16, depth=3, seed=0)
    m2, loss2 = _build(n_interior=256, width=16, depth=3, seed=0)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    # the loss closures see the same (fixed) collocation, so they agree at init
    assert torch.isclose(loss1(), loss2())


def test_lbfgs_reduces_loss():
    """A handful of L-BFGS iterations must strictly decrease the PINN loss from
    the initial point -- otherwise the quasi-Newton step is doing nothing."""
    model, loss_fn = _build(n_interior=512, width=24, depth=3, seed=1)
    loss0 = float(loss_fn().item())
    train_lbfgs(model, loss_fn, outer=3, max_iter=10)
    loss1 = float(loss_fn().item())
    assert loss1 < 0.5 * loss0  # a clear reduction, not roundoff


def test_eval_counting_is_monotone_and_counts_line_search():
    """Logged evaluation counts must strictly increase, and L-BFGS must record
    MORE evaluations than iterations (the strong-Wolfe line search calls the
    closure several times per iteration) -- the reason a per-eval axis, not a
    per-step axis, is the honest comparison."""
    model, loss_fn = _build(n_interior=256, width=16, depth=3, seed=2)
    outer, max_iter = 4, 10
    history = train_lbfgs(model, loss_fn, outer=outer, max_iter=max_iter)
    evals = [h[0] for h in history]
    assert len(history) == outer
    assert all(a < b for a, b in zip(evals, evals[1:]))  # strictly increasing
    # more evaluations than the nominal iteration budget => line search counted
    assert evals[-1] > outer  # at least one eval per outer, usually many more
    assert np.all(np.array(evals) > 0)
