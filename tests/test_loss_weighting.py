"""The loss-weighting ablation: the structural claims behind it, and the copy.

Two kinds of test here.

The first kind pins ``experiments/loss_weighting.py``'s trainer to
``experiments/heat.py``'s. The ablation needs a loop that records per-term
gradient norms and can rewrite its own weights mid-run, and threading both
through the shipped trainer would complicate the function seven other
experiments depend on -- so it is a copy, and a copy is only safe if something
checks it has not drifted. At ``w_ic = w_bc = 1``, same seed and budget, the
two must produce bit-identical weights.

The second kind tests the claims the experiment is built on, without training
anything. ``u == 0`` solves the heat equation and satisfies its homogeneous
Dirichlet walls, so the initial condition is the only term that rules it out;
the output bias has an exactly absent residual gradient, because a constant
offset lives in the residual's null space. Both are checkable in a few lines
and both are the reason the sweep comes out asymmetric.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import heat  # noqa: E402
import loss_weighting as LW  # noqa: E402
from pinn.losses import (  # noqa: E402
    boundary_points,
    initial_points,
    interior_points,
    residual_loss,
)
from pinn.model import MLP, set_seed  # noqa: E402


# -- the copy is pinned to the original --------------------------------------


def test_unweighted_trainer_matches_the_shipped_one_bit_for_bit():
    """The load-bearing test: at w=1 the ablation's loop IS heat.train's loop.

    Same seed, same budget, same sampler draws, so every parameter must agree
    exactly -- not to a tolerance. A tolerance would hide precisely the kind of
    drift this is here to catch (a reordered sampler call, a different default,
    an extra generator draw).
    """
    kw = dict(seed=0, steps=20, n_interior=200, width=32)
    ref, _ = heat.train(w_ic=1.0, w_bc=1.0, depth=4, lr=1e-3, **kw)
    got, trace = LW.train_weighted(w_ic=1.0, w_bc=1.0, lr=1e-3,
                                   grad_every=0, **kw)
    for a, b in zip(ref.parameters(), got.parameters()):
        assert torch.equal(a, b)
    assert trace == []                        # grad_every=0 logs nothing


def test_gradient_logging_does_not_change_the_trained_weights():
    """Instrumenting must be free of side effects. The extra backward passes
    run with ``retain_graph=True`` on the same forward pass the optimizer step
    uses, and if any of them leaked into ``.grad`` the two runs would diverge.
    """
    kw = dict(w_ic=1.0, w_bc=1.0, seed=0, steps=20, n_interior=200, width=32)
    quiet, _ = LW.train_weighted(grad_every=0, **kw)
    loud, trace = LW.train_weighted(grad_every=5, **kw)
    for a, b in zip(quiet.parameters(), loud.parameters()):
        assert torch.equal(a, b)
    assert len(trace) == 5                    # steps 0, 5, 10, 15, 20


# -- the structural claims the sweep is designed around ----------------------


def test_the_trivial_field_satisfies_everything_except_the_initial_condition():
    """Why w_ic and w_bc are not interchangeable, checked directly.

    u == 0 has zero PDE residual and zero error on the homogeneous Dirichlet
    walls, so neither of those terms can distinguish it from the true solution.
    Only L_ic can, and its value there is the full energy of the initial
    condition. If this were not so the whole asymmetry in the experiment would
    have no mechanism.
    """
    gen = torch.Generator().manual_seed(0)
    # A real network with its output layer zeroed, not a constant tensor and
    # not ``0 * x``. The residual differentiates its input twice, so the
    # trivial field has to be zero *through a graph that survives two
    # derivatives*; both shortcuts produce a first derivative with no grad_fn
    # and raise instead of reporting the zero residual that is the point. This
    # is also the failure the experiment actually risks -- a trained net that
    # has collapsed, not an abstract zero function.
    zero = MLP(**heat.model_config(16, 2))
    with torch.no_grad():
        for p in list(zero.parameters())[-2:]:
            p.zero_()

    interior = interior_points(500, heat.X_RANGE, heat.T_RANGE, gen)
    assert residual_loss(zero, interior, heat.heat_residual).item() == 0.0

    left, right = boundary_points(200, heat.X_RANGE, heat.T_RANGE, gen)
    bc = torch.cat([left, right], dim=0)
    assert torch.mean((zero(bc) - torch.zeros(bc.shape[0], 1)) ** 2).item() == 0.0

    ic = initial_points(400, heat.X_RANGE, heat.T_RANGE[0], gen)
    target = heat.initial_condition(ic[:, 0:1])
    loss_ic = torch.mean((zero(ic) - target) ** 2).item()
    assert loss_ic > 0.1                                  # and it is not small


def test_the_residual_cannot_see_the_output_bias():
    """A constant offset is an exact solution of u_t = alpha u_xx, so the
    parameter that implements one has no residual gradient at all -- autograd
    returns None for it, which is why the experiment's gradient helper passes
    ``allow_unused=True``. The IC term does see it, which is the same
    asymmetry as the previous test, one parameter at a time.
    """
    set_seed(0)
    model = MLP(**heat.model_config(32, 2))
    out_bias = list(model.parameters())[-1]
    gen = torch.Generator().manual_seed(0)

    interior = interior_points(300, heat.X_RANGE, heat.T_RANGE, gen)
    loss_r = residual_loss(model, interior, heat.heat_residual)
    assert torch.autograd.grad(loss_r, [out_bias], retain_graph=True,
                               allow_unused=True)[0] is None

    ic = initial_points(200, heat.X_RANGE, heat.T_RANGE[0], gen)
    loss_ic = torch.mean((model(ic) - heat.initial_condition(ic[:, 0:1])) ** 2)
    g = torch.autograd.grad(loss_ic, [out_bias], allow_unused=True)[0]
    assert g is not None and float(g.abs().max()) > 0.0

    # and the helper agrees, reading the absent gradient as the zero it is
    assert LW._flat_grad(loss_r, [out_bias]).abs().max() == 0.0


def test_flat_grad_matches_an_independent_autograd_call():
    """The gradient read-out is a measurement, so it gets checked against the
    obvious spelling on a term every parameter does reach."""
    set_seed(1)
    model = MLP(**heat.model_config(16, 2))
    params = list(model.parameters())
    gen = torch.Generator().manual_seed(1)
    ic = initial_points(64, heat.X_RANGE, heat.T_RANGE[0], gen)
    loss = torch.mean((model(ic) - heat.initial_condition(ic[:, 0:1])) ** 2)

    reference = torch.cat([g.reshape(-1) for g in
                           torch.autograd.grad(loss, params, retain_graph=True)])
    assert torch.equal(LW._flat_grad(loss, params), reference)
    assert LW._grad_norm(loss, params) == float(reference.norm())
    hi, mean = LW._grad_stats(loss, params)
    assert hi == float(reference.abs().max())
    assert mean == float(reference.abs().mean())


# -- the adaptive rule -------------------------------------------------------


def test_adaptive_weights_move_off_their_initialization():
    """The rule has to actually rewrite the weights, and to a value set by the
    gradient ratio rather than by the initialization. Started at w=1, a few
    updates must land somewhere far from 1 -- the point of measurement 3 is
    that the ratio is large, so a rule that stayed near 1 would be broken.
    """
    _, trace = LW.train_weighted(w_ic=1.0, w_bc=1.0, adaptive=True, seed=0,
                                 steps=LW.ADAPT_EVERY * 2, n_interior=300,
                                 width=32, grad_every=LW.ADAPT_EVERY)
    w_ic = [t[6] for t in trace]
    assert w_ic[0] != 1.0                     # updated before the first log
    assert max(w_ic) > 5.0


def test_the_adaptive_ratio_is_inflated_by_its_own_shape():
    """Why the rule picks ~1e5 when the two gradients have comparable norms.

    The rule is ``max_theta |grad L_r| / mean_theta |grad L_i|`` -- a max over
    parameters divided by a mean over parameters. Those are not the same
    statistic, so the ratio carries a large factor from the shape of the
    gradient vectors alone, independent of the scale difference it is supposed
    to be correcting. This measures the two side by side on the real network:
    the norm ratio the diagnosis talks about, and the max/mean ratio the rule
    actually computes.

    The experiment's measurement 4 finds the norm ratio near 1 on this problem
    -- the premise (the residual gradient dominates) is false here -- so there
    is nothing to offset the shape factor and the rule runs away. That is the
    mechanism behind the 1e5 weights in the committed log, checked here rather
    than asserted in prose.
    """
    set_seed(0)
    model = MLP(**heat.model_config(128, 4))
    params = list(model.parameters())
    gen = torch.Generator().manual_seed(0)

    interior = interior_points(2000, heat.X_RANGE, heat.T_RANGE, gen)
    ic = initial_points(400, heat.X_RANGE, heat.T_RANGE[0], gen)
    loss_r = residual_loss(model, interior, heat.heat_residual)
    loss_ic = torch.mean((model(ic) - heat.initial_condition(ic[:, 0:1])) ** 2)

    norm_ratio = LW._grad_norm(loss_r, params) / LW._grad_norm(loss_ic, params)
    max_r, _ = LW._grad_stats(loss_r, params)
    _, mean_ic = LW._grad_stats(loss_ic, params)
    rule_ratio = max_r / mean_ic

    # The two gradients are within an order of magnitude of each other...
    assert 0.05 < norm_ratio < 20.0
    # ...and yet the statistic the rule forms out of them is orders larger.
    assert rule_ratio > 30.0 * norm_ratio


def test_adaptive_weight_update_is_the_documented_formula():
    """One update reproduced by hand from the two gradient statistics, so the
    implemented rule is the one the docstring claims and not a variant."""
    set_seed(0)
    _, trace = LW.train_weighted(w_ic=1.0, w_bc=1.0, adaptive=True, seed=0,
                                 steps=0, n_interior=300, width=32,
                                 grad_every=1)

    set_seed(0)
    gen = torch.Generator().manual_seed(0)
    interior = interior_points(300, heat.X_RANGE, heat.T_RANGE, gen)
    ic = initial_points(400, heat.X_RANGE, heat.T_RANGE[0], gen)
    ic_target = heat.initial_condition(ic[:, 0:1])
    boundary_points(200, heat.X_RANGE, heat.T_RANGE, gen)     # same draw order
    model = MLP(**heat.model_config(32, 4))
    params = list(model.parameters())
    loss_r = residual_loss(model, interior, heat.heat_residual)
    loss_ic = torch.mean((model(ic) - ic_target) ** 2)
    max_r, _ = LW._grad_stats(loss_r, params)
    _, mean_ic = LW._grad_stats(loss_ic, params)
    expected = (1 - LW.ADAPT_ALPHA) * 1.0 + LW.ADAPT_ALPHA * (max_r / mean_ic)

    assert np.isclose(trace[0][6], expected, rtol=1e-10)


def test_amplitude_ratio_separates_collapse_from_a_bad_fit():
    """The metric that makes measurement 1 readable: relative L2 saturates near
    1 for anything badly wrong, so it cannot tell a collapse to zero from a fit
    of the wrong shape. This can."""
    zero = lambda c: torch.zeros(c.shape[0], 1)           # noqa: E731
    assert LW.amplitude_ratio(zero) == 0.0

    exact = lambda c: torch.tensor(                       # noqa: E731
        heat.heat_exact(c[:, 0].numpy(), c[:, 1].numpy()),
        dtype=torch.float32).reshape(-1, 1)
    assert np.isclose(LW.amplitude_ratio(exact), 1.0, atol=1e-5)
