"""The d-dimensional ground truth, checked before anything is measured against it.

``experiments/highd_heat.py`` is the setup for the high-dimensional study, and
every number that study will report is an error *against its closed form*. At
d = 16 there is no mesh solution to fall back on, so if the closed form is wrong
nothing downstream can notice. It is therefore checked three ways, in order of
how much they assume:

1. **Against the shipped 1D code.** At d = 1 with the Sec. 1 modes, ``exact``
   must equal ``heat.heat_exact`` and ``residual`` must equal
   ``heat.heat_residual``, elementwise. This assumes ``experiments/heat.py`` is
   right, which ``tests/test_heat.py`` establishes independently.
2. **Against finite differences at d = 2 and 3**, which assumes nothing at all
   about either implementation -- the general check that the closed form really
   solves ``u_t = alpha Laplacian u`` in more than one spatial dimension.
3. **Against the boundary and initial data**, which the solution must satisfy on
   all 2d faces and on ``t = 0``.

The measurement machinery gets the same treatment: the closed-form norm is
checked against Monte Carlo, and the Monte Carlo error metric is checked against
a network that *is* the exact solution (relative error must come out ~0) and
against one that is identically zero (must come out exactly 1). Between them
those two pin the coordinate layout, which is the bug a d-dimensional rewrite is
most likely to have and which no smoke test would catch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import highd_heat as H  # noqa: E402
from pinn.model import MLP, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class ExactField(torch.nn.Module):
    """The closed-form solution wrapped as a network, for metric calibration.

    Written independently of ``highd_heat.exact`` (torch rather than numpy, and
    reading its coordinates itself) so that agreement between the two is
    evidence about the *layout* -- which column is time, which are space -- and
    not just the same expression evaluated twice.
    """

    def __init__(self, problem):
        super().__init__()
        self.p = problem

    def forward(self, coords):
        p = self.p
        x, t = coords[:, : p.d], coords[:, p.d : p.d + 1]
        out = torch.zeros(coords.shape[0], 1, dtype=coords.dtype)
        for k, a, rate in zip(p.modes, p.amps, p.rates):
            kt = torch.tensor(k, dtype=coords.dtype)
            sp = torch.prod(torch.sin(np.pi * kt[None, :] * x), dim=1, keepdim=True)
            out = out + float(a) * sp * torch.exp(-float(rate) * t)
        return out


class ZeroField(torch.nn.Module):
    def forward(self, coords):
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype)


def _net(problem, seed=0, width=16):
    set_seed(seed)
    return MLP(**H.model_config(problem, width=width, depth=2))


# ---------------------------------------------------------------------------
# 1. The d = 1 reduction: this module must be the shipped one
# ---------------------------------------------------------------------------
def test_exact_reduces_to_the_shipped_1d_solution():
    from heat import ALPHA, heat_exact

    p = H.HighDHeat(1, terms=H.sec1_terms(), alpha=ALPHA)
    rng = np.random.default_rng(0)
    coords = H.uniform_box_points(p, 5000, rng)
    mine = H.exact_from_coords(p, coords)
    theirs = heat_exact(coords[:, 0], coords[:, 1])
    assert np.max(np.abs(mine - theirs)) < 1e-14


def test_residual_reduces_to_the_shipped_1d_residual():
    """Elementwise equality of the two residuals on the same network and points.

    Both differentiate the same graph, so this is stronger than a tolerance
    check on a solved field: any disagreement in which column is time, or an
    extra u_tt from summing the wrong dimensions, shows up immediately.
    """
    from heat import ALPHA, heat_residual

    p = H.HighDHeat(1, terms=H.sec1_terms(), alpha=ALPHA)
    model = _net(p, seed=3)
    gen = torch.Generator().manual_seed(11)
    coords = H.interior_points(p, 256, gen)

    u = model(coords)
    mine = H.residual(p, u, coords)
    theirs = heat_residual(model(coords), coords, alpha=ALPHA)
    assert mine.shape == (256, 1)
    assert torch.allclose(mine, theirs, atol=1e-6, rtol=1e-5)


def test_the_time_column_is_not_in_the_laplacian():
    """A residual that summed all d+1 columns would pick up a spurious u_tt.

    Constructed to catch exactly that: on the d = 1 problem, compare against a
    hand-written ``u_t - alpha (u_xx + u_tt)`` and require they *differ* by
    alpha u_tt, which pins the sign and the omission at once.
    """
    from pinn import derivatives as D

    p = H.HighDHeat(1, terms=H.sec1_terms(), alpha=0.05)
    model = _net(p, seed=5)
    gen = torch.Generator().manual_seed(2)
    coords = H.interior_points(p, 128, gen)

    u = model(coords)
    r = H.residual(p, u, coords)
    wrong = D.partial(u, coords, 1) - p.alpha * D.laplacian(u, coords)
    u_tt = D.u_tt(model(coords), coords)
    assert torch.allclose(r - wrong, p.alpha * u_tt, atol=1e-6)
    # and the two are genuinely different, so the check above is not vacuous
    assert torch.max(torch.abs(r - wrong)) > 1e-4


# ---------------------------------------------------------------------------
# 2. Finite differences in more than one spatial dimension
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [2, 3, 5])
def test_exact_satisfies_the_pde_by_finite_differences(d):
    """u_t - alpha sum_i u_{x_i x_i} = 0, checked with no reference to the code
    that produced the solution beyond evaluating it.

    Points are kept away from the faces so the central differences stay inside
    the domain and the derivatives stay O(1) relative to the field.
    """
    p = H.HighDHeat(d)
    rng = np.random.default_rng(4)
    n = 400
    X = 0.2 + 0.6 * rng.random((n, d))
    t = 0.1 + 0.8 * rng.random(n)
    h = 1e-3

    lap = np.zeros(n)
    for i in range(d):
        e = np.zeros((1, d))
        e[0, i] = h
        lap += (H.exact(p, X + e, t) - 2 * H.exact(p, X, t) + H.exact(p, X - e, t)) / h ** 2
    u_t = (H.exact(p, X, t + h) - H.exact(p, X, t - h)) / (2 * h)

    resid = u_t - p.alpha * lap
    scale = np.max(np.abs(H.exact(p, X, t)))
    assert np.max(np.abs(resid)) / scale < 1e-5


def test_exact_matches_the_initial_condition():
    for d in (1, 3, 8):
        p = H.HighDHeat(d)
        rng = np.random.default_rng(d)
        X = rng.random((500, d))
        from_solution = H.exact(p, X, 0.0)
        from_ic = H.initial_condition(p, torch.tensor(X, dtype=torch.float64))
        assert np.max(np.abs(from_solution - from_ic.numpy().ravel())) < 1e-13


def test_exact_vanishes_on_every_face():
    """All 2d faces, each one explicitly -- not a random sample of them."""
    for d in (1, 2, 5):
        p = H.HighDHeat(d)
        rng = np.random.default_rng(7)
        for axis in range(d):
            for side in (0.0, 1.0):
                X = rng.random((200, d))
                X[:, axis] = side
                t = rng.random(200)
                assert np.max(np.abs(H.exact(p, X, t))) < 1e-12


def test_boundary_sampler_lands_on_faces_where_the_solution_is_zero():
    p = H.HighDHeat(4)
    gen = torch.Generator().manual_seed(0)
    bc = H.boundary_points(p, 2000, gen)
    x = bc[:, : p.d].numpy().astype(float)

    # exactly one coordinate pinned to a wall, per point
    on_wall = (x == 0.0) | (x == 1.0)
    assert np.all(on_wall.sum(axis=1) == 1)
    # every one of the 2d faces is actually visited
    for axis in range(p.d):
        for side in (0.0, 1.0):
            assert np.any(x[:, axis] == side)
    assert np.max(np.abs(H.exact_from_coords(p, bc.numpy().astype(float)))) < 1e-12


# ---------------------------------------------------------------------------
# 3. The problem's stated properties
# ---------------------------------------------------------------------------
def test_the_fundamental_decay_rate_does_not_depend_on_d():
    """The point of scaling alpha_d = alpha_1 / d, asserted rather than assumed."""
    expected = H.ALPHA_1 * np.pi ** 2
    for d in (1, 2, 4, 8, 16, 32):
        p = H.HighDHeat(d)
        assert abs(p.rates[0] - expected) < 1e-12
        assert abs(p.alpha - H.ALPHA_1 / d) < 1e-15


def test_the_second_mode_rate_ratio_compresses_with_d():
    """The documented consequence of the default target: the rate separation
    is (d+3)/d and therefore goes to 1. Stated in the module docstring as a
    property the sweep does *not* hold fixed, so it is pinned here."""
    for d in (1, 2, 4, 8, 16):
        p = H.HighDHeat(d)
        assert abs(p.rates[1] / p.rates[0] - (d + 3) / d) < 1e-12
    assert abs(H.HighDHeat(1).rates[1] / H.HighDHeat(1).rates[0] - 4.0) < 1e-12


def test_default_target_is_not_permutation_symmetric():
    """Swapping axis 0 with axis 1 must change the field, or the symmetry
    objection the default was chosen to avoid would still apply."""
    p = H.HighDHeat(4)
    rng = np.random.default_rng(0)
    X = rng.random((500, 4))
    swapped = X[:, [1, 0, 2, 3]]
    a, b = H.exact(p, X, 0.3), H.exact(p, swapped, 0.3)
    assert np.max(np.abs(a - b)) > 0.05


@pytest.mark.parametrize("bad", [
    dict(d=0),
    dict(d=2, terms=[((1,), 1.0)]),          # multi-index of the wrong length
    dict(d=2, terms=[((0, 1), 1.0)]),        # a zero entry kills the mode
    dict(d=2, terms=[((1, 1.5), 1.0)]),      # non-integer
    dict(d=2, t_range=(1.0, 1.0)),           # empty interval
])
def test_invalid_problems_are_refused(bad):
    with pytest.raises(ValueError):
        H.HighDHeat(**bad)


# ---------------------------------------------------------------------------
# 4. The norms and the Monte Carlo metric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 6])
def test_closed_form_mean_square_agrees_with_monte_carlo(d):
    """``exact_ms`` is the denominator of every relative error in the study, and
    it is derived from orthogonality rather than sampled. Check it against the
    thing it replaces, inside the sampling error the sample itself reports."""
    p = H.HighDHeat(d)
    rng = np.random.default_rng(3)
    coords = H.uniform_box_points(p, 400_000, rng)
    u2 = H.exact_from_coords(p, coords) ** 2
    sampled = u2.mean()
    se = u2.std(ddof=1) / np.sqrt(u2.size)
    assert abs(sampled - H.exact_ms(p)) < 4 * se


def test_closed_form_mean_square_matches_a_hand_computed_case():
    """One case worked out by hand, so the orthogonality argument is checked
    against arithmetic and not only against its own Monte Carlo.

    Single mode k = (1, 1), amplitude 1, so mean_x u^2 = 2^-2 exp(-2 r t) and
    the time average over [0, 1] is (1 - exp(-2r)) / (2r).
    """
    p = H.HighDHeat(2, terms=[((1, 1), 1.0)])
    r = p.alpha * np.pi ** 2 * 2
    by_hand = 0.25 * (1 - np.exp(-2 * r)) / (2 * r)
    assert abs(H.exact_ms(p) - by_hand) < 1e-15


@pytest.mark.parametrize("d", [1, 4, 12])
def test_zero_field_scores_a_relative_error_of_one(d):
    """||0 - u|| / ||u|| = 1 by definition, so this calibrates the whole metric
    path -- sampler, predictor, exact solution, closed-form denominator -- to a
    number known without any computation at all."""
    p = H.HighDHeat(d)
    rel, se = H.rel_l2_mc(ZeroField(), p, n=200_000, seed=1)
    assert abs(rel - 1.0) < 6 * se + 1e-12


@pytest.mark.parametrize("d", [1, 4, 12])
def test_the_exact_field_scores_a_relative_error_of_zero(d):
    """The other end of the same calibration, and the one that pins the column
    layout: a network computing the closed form in torch, scored by the numpy
    closed form, must agree. The residual floor is float32 rounding inside
    ``predict``, not the metric."""
    p = H.HighDHeat(d)
    rel, _ = H.rel_l2_mc(ExactField(p), p, n=100_000, seed=2)
    assert rel < 1e-3


def test_a_scaled_field_scores_the_scale_error():
    """A field that is 1.1x the truth must score 0.1, at every d -- the check
    that the metric is a relative L2 and not something that has picked up a
    d-dependent normalization."""
    for d in (1, 4, 12):
        p = H.HighDHeat(d)

        class Scaled(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, c):
                return 1.1 * self.inner(c)

        rel, se = H.rel_l2_mc(Scaled(ExactField(p)), p, n=200_000, seed=5)
        assert abs(rel - 0.1) < 6 * se + 1e-3


def test_monte_carlo_standard_error_is_honest():
    """The reported standard error must actually describe the scatter.

    Ten independent samples of the same quantity; their observed spread should
    sit near the standard error each one reports. Loose bounds, because the
    spread of a spread from ten draws is itself wide -- this is a check against
    being wrong by an order of magnitude, which is the failure that would matter.
    """
    p = H.HighDHeat(8)
    model = _net(p, seed=1)
    vals, ses = [], []
    for s in range(10):
        rel, se = H.rel_l2_mc(model, p, n=50_000, seed=1000 + s)
        vals.append(rel)
        ses.append(se)
    observed = np.std(vals, ddof=1)
    reported = np.mean(ses)
    assert 0.3 < observed / reported < 3.0


def test_concentration_reports_the_share_of_the_largest_entries():
    flat = np.ones(1000)
    assert abs(H.concentration(flat, 0.01) - 0.01) < 1e-12
    spike = np.zeros(1000)
    spike[0] = 1.0
    assert abs(H.concentration(spike, 0.01) - 1.0) < 1e-12


def test_the_concentration_follows_the_predicted_1_point_5_to_the_d_law():
    """The concentration is not just observed, it is derivable, and the
    derivation is what makes the sample sizes in Day 10 predictable rather than
    tuned.

    For one product mode the integrand of ``||u||^2`` is
    ``v = prod_i sin^2(pi x_i)``. The factors are independent under a uniform
    draw, and ``E[sin^2] = 1/2``, ``E[sin^4] = 3/8``, so

        E[v] = 2^-d,  E[v^2] = (3/8)^d,  Var(v)/E[v]^2 = (3/2)^d - 1.

    The relative standard error of the n-point mean is therefore
    ``sqrt(((3/2)^d - 1)/n)`` -- exponential in d, which is why the metric needs
    a standard error at all. Checked against the sample, with a tolerance set by
    the fourth moment of a heavy-tailed variable at n = 200k rather than by
    optimism.
    """
    rng = np.random.default_rng(11)
    for d in (1, 2, 4, 8):
        X = rng.random((200_000, d))
        v = np.prod(np.sin(np.pi * X) ** 2, axis=1)
        observed = v.var(ddof=1) / v.mean() ** 2
        predicted = 1.5 ** d - 1
        assert 0.75 < observed / predicted < 1.25, (d, observed, predicted)


def test_the_l2_integrand_concentrates_as_d_grows():
    """The reason the metric needs a standard error at all: ``prod_i sin(pi x_i)``
    has rms ``2^(-d/2)`` but typical value ``2^-d``, so in high d the mean square
    is carried by a thin set of points. Monotone in d over the range the study
    uses."""
    shares = []
    for d in (1, 2, 4, 8, 16):
        p = H.HighDHeat(d)
        _, _, top = H.mc_relative_sd(p, 100_000, seed=d)
        shares.append(top)
    assert all(b > a for a, b in zip(shares, shares[1:]))
    assert shares[0] < 0.10 and shares[-1] > 0.5


# ---------------------------------------------------------------------------
# 5. Samplers
# ---------------------------------------------------------------------------
def test_samplers_have_the_right_shapes_and_ranges():
    p = H.HighDHeat(5, t_range=(0.0, 2.0))
    gen = torch.Generator().manual_seed(0)
    interior = H.interior_points(p, 300, gen)
    ic = H.initial_points(p, 100, gen)
    bc = H.boundary_points(p, 100, gen)

    assert interior.shape == (300, 6) and ic.shape == (100, 6) and bc.shape == (100, 6)
    assert interior.requires_grad
    x = interior[:, :5].detach()
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    t = interior[:, 5].detach()
    assert float(t.min()) >= 0.0 and float(t.max()) <= 2.0
    assert torch.all(ic[:, 5] == 0.0)


def test_samplers_are_reproducible_from_the_generator():
    p = H.HighDHeat(3)
    a = H.interior_points(p, 50, torch.Generator().manual_seed(9))
    b = H.interior_points(p, 50, torch.Generator().manual_seed(9))
    assert torch.equal(a.detach(), b.detach())
    c = H.boundary_points(p, 50, torch.Generator().manual_seed(9))
    e = H.boundary_points(p, 50, torch.Generator().manual_seed(9))
    assert torch.equal(c, e)


def test_boundary_budget_does_not_grow_with_d():
    """``n_bc`` is a total across faces, not a per-face count -- the choice that
    keeps the boundary cost flat in d. Asserted because the 1D helper in
    ``pinn/losses.py`` does the opposite (one set per wall)."""
    for d in (2, 16):
        p = H.HighDHeat(d)
        bc = H.boundary_points(p, 128, torch.Generator().manual_seed(1))
        assert bc.shape[0] == 128


# ---------------------------------------------------------------------------
# 6. A real, short solve
# ---------------------------------------------------------------------------
def test_training_reduces_the_error_in_two_dimensions():
    p = H.HighDHeat(2)
    model, history, best = H.train(p, n_interior=800, width=32, steps=400, seed=0,
                                   eval_every=200, eval_n=20_000)
    steps = [h[0] for h in history]
    errs = [h[5] for h in history]
    assert steps[0] == 0 and steps[-1] == 400
    assert errs[0] > 0.5
    assert errs[-1] < errs[0] / 2
    assert 0 <= best["step"] <= 401
    assert best["loss"] <= best["final_loss"]


def test_loss_selection_returns_the_lowest_loss_iterate_not_the_last():
    """The selection rule, checked on a run where the two differ.

    Both arms must be the same trajectory -- same seed, same points -- so any
    difference is the choice of iterate and nothing else. The returned network
    has to be the one whose *training loss* was lowest, which is verified by
    recomputing that loss on the returned model rather than trusting the
    bookkeeping.
    """
    p = H.HighDHeat(2)
    kw = dict(n_interior=400, width=24, steps=600, seed=2, eval_every=300,
              eval_n=5_000)
    final, _, best = H.train(p, select="final", **kw)
    chosen, _, best2 = H.train(p, select="best_loss", **kw)
    assert best == best2

    def training_loss(model):
        gen = torch.Generator().manual_seed(2)
        interior = H.interior_points(p, 400, gen)
        ic = H.initial_points(p, 400, gen)
        ic_target = H.initial_condition(p, ic[:, : p.d])
        bc = H.boundary_points(p, 400, gen)
        r = H.residual(p, model(interior), interior)
        total = (torch.mean(r ** 2)
                 + torch.mean((model(ic) - ic_target) ** 2)
                 + torch.mean(model(bc) ** 2))
        return float(total.detach())

    assert training_loss(chosen) <= training_loss(final) + 1e-12
    # The recorded loss must belong to the weights that were saved: recomputing
    # it on the returned model reproduces ``best["loss"]`` to floating point.
    assert abs(training_loss(chosen) - best["loss"]) < 1e-9


def test_selection_mode_is_validated():
    with pytest.raises(ValueError):
        H.train(H.HighDHeat(1), steps=1, select="lowest_error")


def test_model_config_matches_the_problem_dimension():
    for d in (1, 7):
        p = H.HighDHeat(d)
        cfg = H.model_config(p)
        assert cfg["in_dim"] == d + 1
        model = MLP(**cfg)
        assert model(torch.zeros(3, d + 1)).shape == (3, 1)
