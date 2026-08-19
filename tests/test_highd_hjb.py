"""The HJB problem: its closed form, its metric, and its committed log.

``experiments/highd_hjb.py`` is the repo's second high-dimensional PDE, and the
whole point of it is that its ground truth is exact at every d. So the closed
form is what gets tested hardest here, five independent ways -- against the PDE
by autograd, against a Cole-Hopf transform of the same solution, against
Gauss-Legendre quadrature of the Riccati, against central differences of the
Riccati ODE, and against Monte Carlo for every exact moment. If any of those
disagreed, every error in Sec. 13 would be measured against the wrong answer.

Three further groups:

- **The metric.** The headline number is divided by ``sd(u)``, so a constant
  predictor must score exactly 1.0, a zero predictor must score ``rms/sd``, and
  the exact solution must score ~0. Those are checked by building the three
  predictors and scoring them, not by re-deriving the formula.
- **The resumable trainer.** The d = 16 cells are longer than the environment
  can hold a foreground process, so the sweep checkpoints and resumes. A
  resumed run and an uninterrupted one must visit the same parameters -- checked
  by running both and comparing the histories entry for entry, and the weights
  tensor for tensor.
- **The committed log**, against the budget the module declares. A log that has
  drifted from the constants the README quotes is the failure mode a table
  cannot notice.

The 2.5 hours of CPU that produce the sweep are not re-run here. The cost claim
(``d + 1`` reverse-mode passes, one fewer than the heat residual's ``d + 2``) is
checked as a call count rather than a clock, for the reason gp-from-scratch
learned twice in CI: a portable assertion is about shape, not level.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import highd_hjb as J  # noqa: E402
from common import read_csv  # noqa: E402
from pinn import derivatives as D  # noqa: E402
from pinn.model import MLP  # noqa: E402

TINY = dict(n_interior=200, n_tc=50, n_bc=50, width=16, depth=2, steps=30, lr=1e-3)


def _coords(problem, n, seed=0):
    rng = np.random.default_rng(seed)
    return J.uniform_box_points(problem, n, rng)


# ---------------------------------------------------------------------------
# 1. construction
# ---------------------------------------------------------------------------
def test_defaults_and_derived_quantities():
    p = J.HJB(3)
    assert p.d == 3
    assert p.nu == pytest.approx(J.NU_1 / 3)
    np.testing.assert_allclose(p.q, [4.0, 1.0, 1.0])
    np.testing.assert_allclose(p.k, np.sqrt(p.q / (4 * p.lam)))
    np.testing.assert_allclose(p.beta, 4 * np.sqrt(p.lam * p.q))
    # the default terminal cost is half the Riccati fixed point, so w_T is the
    # same number on every axis -- worth pinning, since it is what makes the
    # per-coordinate solutions differ only through beta
    np.testing.assert_allclose(p.w_T, -1.0 / 3.0)


def test_q_is_not_permutation_symmetric():
    """One distinguished axis, for the reason ``highd_heat`` has one."""
    for d in (2, 5, 16):
        q = J.default_q(d)
        assert q[0] != q[1]
        assert len(set(q[1:])) == 1


@pytest.mark.parametrize("bad", [dict(d=0), dict(d=2, q=[1.0, 0.0]),
                                 dict(d=2, q=[1.0]), dict(d=2, c=-1.0),
                                 dict(d=1, lam=0.0),
                                 dict(d=1, t_range=(1.0, 1.0))])
def test_invalid_problems_refuse(bad):
    with pytest.raises(ValueError):
        J.HJB(**bad)


# ---------------------------------------------------------------------------
# 2. the Riccati solution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 3, 8])
def test_p_hits_the_terminal_condition_and_the_fixed_point(d):
    p = J.HJB(d)
    np.testing.assert_allclose(J.p_of_t(p, p.T)[0], p.c, rtol=0, atol=1e-14)
    # far in the past the Riccati has relaxed onto its stable fixed point k
    far = J.HJB(d, t_range=(0.0, 40.0))
    np.testing.assert_allclose(J.p_of_t(far, 0.0)[0], far.k, rtol=1e-12)


@pytest.mark.parametrize("d", [1, 4])
def test_p_solves_its_ode(d):
    """``p' = 4 lambda p^2 - q`` by central differences of the closed form."""
    p = J.HJB(d)
    t = np.linspace(0.05, 0.95, 13)
    h = 1e-6
    fd = (J.p_of_t(p, t + h) - J.p_of_t(p, t - h)) / (2 * h)
    np.testing.assert_allclose(fd, J.dp_dt(p, t), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("d", [1, 3, 8])
def test_r_matches_quadrature_of_p(d):
    """The partial-fraction antiderivative against Gauss-Legendre. Different maths."""
    p = J.HJB(d)
    x, w = np.polynomial.legendre.leggauss(300)
    for t0 in (0.0, 0.2, 0.61, 1.0):
        nodes = 0.5 * (x + 1.0) * (p.T - t0) + t0
        wts = w * 0.5 * (p.T - t0)
        quad = 2 * p.nu * float(wts @ J.p_of_t(p, nodes).sum(axis=1))
        assert J.r_of_t(p, t0)[0] == pytest.approx(quad, rel=1e-12, abs=1e-15)


def test_r_vanishes_at_T_and_dr_dt_is_consistent():
    p = J.HJB(4)
    assert J.r_of_t(p, p.T)[0] == pytest.approx(0.0, abs=1e-15)
    t = np.linspace(0.05, 0.95, 9)
    h = 1e-6
    fd = (J.r_of_t(p, t + h) - J.r_of_t(p, t - h)) / (2 * h)
    np.testing.assert_allclose(fd, J.dr_dt(p, t), rtol=1e-6, atol=1e-9)


def test_zero_terminal_cost_gives_the_tanh_solution():
    """``c = 0`` is the textbook special case ``p = k tanh(beta (T-t)/2)``.

    Checked because the general ``w`` formula is the one the module ships, and a
    formula that reproduces a known special case is a formula that has its
    algebra right.
    """
    p = J.HJB(3, c=0.0)
    t = np.linspace(0, 1, 11)
    want = p.k[None, :] * np.tanh(0.5 * p.beta[None, :] * (p.T - t[:, None]))
    np.testing.assert_allclose(J.p_of_t(p, t), want, rtol=1e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. the exact solution against the PDE
# ---------------------------------------------------------------------------
def _exact_torch(problem, coords):
    """u from the closed form, differentiable in ``coords`` (float64)."""
    d = problem.d
    X = coords[:, :d]
    t = coords[:, d : d + 1]
    k = torch.tensor(problem.k, dtype=coords.dtype)
    beta = torch.tensor(problem.beta, dtype=coords.dtype)
    wT = torch.tensor(problem.w_T, dtype=coords.dtype)

    def w_of(tt):
        return wT[None, :] * torch.exp(-beta[None, :] * (problem.T - tt))

    def F(tt):
        w = w_of(tt)
        return (k[None, :] / beta[None, :]) * (torch.log(torch.abs(w))
                                               - 2 * torch.log(torch.abs(1 - w)))

    p = k[None, :] * (1 + w_of(t)) / (1 - w_of(t))
    FT = F(torch.full((1, 1), float(problem.T), dtype=coords.dtype))
    return ((p * X ** 2).sum(dim=1, keepdim=True)
            + 2 * problem.nu * (FT - F(t)).sum(dim=1, keepdim=True))


@pytest.mark.parametrize("d", [1, 2, 5, 10])
def test_exact_solution_kills_the_residual(d):
    problem = J.HJB(d)
    coords = torch.tensor(_coords(problem, 48, seed=d), dtype=torch.float64,
                          requires_grad=True)
    r = J.residual(problem, _exact_torch(problem, coords), coords)
    assert float(r.detach().abs().max()) < 1e-12


@pytest.mark.parametrize("d", [1, 4])
def test_numpy_and_torch_paths_agree(d):
    problem = J.HJB(d)
    cs = _coords(problem, 64, seed=7 + d)
    torch_u = _exact_torch(problem, torch.tensor(cs, dtype=torch.float64))
    np.testing.assert_allclose(torch_u.detach().numpy().ravel(),
                               J.exact_from_coords(problem, cs), rtol=1e-13)


@pytest.mark.parametrize("d", [1, 3, 6])
def test_cole_hopf_linearizes_the_equation(d):
    """``v = exp(-lam u / nu)`` solves ``v_t + nu Lap v = (lam/nu) Q v``.

    The module docstring says the nonlinearity is removable and therefore does
    *not* claim this problem is an essentially nonlinear test. That disclaimer
    is only honest if the transform actually works, so it is checked.
    """
    problem = J.HJB(d)
    coords = torch.tensor(_coords(problem, 32, seed=3 * d), dtype=torch.float64,
                          requires_grad=True)
    u = _exact_torch(problem, coords)
    v = torch.exp(-problem.lam * u / problem.nu)
    g = D.grad(v, coords)
    lap = sum(D.partial(g[:, i : i + 1], coords, i) for i in range(d))
    q = torch.tensor(problem.q, dtype=coords.dtype)
    Q = (q[None, :] * coords[:, :d] ** 2).sum(dim=1, keepdim=True)
    lhs = g[:, d : d + 1] + problem.nu * lap - (problem.lam / problem.nu) * Q * v
    # v spans many orders of magnitude in d, so the tolerance is relative to
    # the size of the terms that are cancelling
    scale = float(torch.max(torch.abs(problem.nu * lap)).detach())
    assert float(lhs.detach().abs().max()) / scale < 1e-10


@pytest.mark.parametrize("d", [1, 4, 9])
def test_terminal_condition_is_the_quadratic_terminal_cost(d):
    problem = J.HJB(d)
    X = _coords(problem, 40, seed=d)[:, :d]
    got = J.exact(problem, X, np.full(X.shape[0], problem.T))
    want = (problem.c[None, :] * X ** 2).sum(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)
    tc = J.terminal_condition(problem, torch.tensor(X, dtype=torch.float64))
    np.testing.assert_allclose(tc.numpy().ravel(), want, rtol=1e-13)


# ---------------------------------------------------------------------------
# 4. the residual's cost, as a call count
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 6])
def test_residual_costs_d_plus_one_backward_passes(monkeypatch, d):
    """One shared first gradient, then one per spatial axis: ``d + 1``.

    ``highd_heat.residual`` takes ``d + 2`` because it asks for ``u_t`` with a
    separate call. Here ``u_t`` is a column of the gradient the Laplacian
    already needs, and ``|grad u|^2`` is the rest of that same gradient, so all
    three derivative terms come out of one pass. The saving is small and the
    reason to test it is that it is *claimed* in the docstring.
    """
    calls = {"n": 0}
    real = torch.autograd.grad

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(torch.autograd, "grad", counted)
    problem = J.HJB(d)
    gen = torch.Generator().manual_seed(0)
    coords = J.interior_points(problem, 16, gen)
    model = MLP(**J.model_config(problem, width=8, depth=2))
    J.residual(problem, model(coords), coords)
    assert calls["n"] == d + 1


# ---------------------------------------------------------------------------
# 5. exact moments, against Monte Carlo
# ---------------------------------------------------------------------------
def test_quad_moments_against_monte_carlo():
    rng = np.random.default_rng(0)
    a = np.array([0.4, -1.3, 2.0, 0.7])
    const = 0.31
    x = -1 + 2 * rng.random((400_000, 4))
    v = const + (a[None, :] * x ** 2).sum(axis=1)
    mean, ms = J._quad_moments(a, const)
    assert mean == pytest.approx(v.mean(), rel=2e-3)
    assert ms == pytest.approx((v ** 2).mean(), rel=2e-3)


@pytest.mark.parametrize("d", [1, 4, 16])
def test_space_time_moments_against_monte_carlo(d):
    problem = J.HJB(d)
    mom = J.space_time_moments(problem)
    cs = _coords(problem, 400_000, seed=100 + d)
    u = J.exact_from_coords(problem, cs)
    assert mom["mean"] == pytest.approx(u.mean(), rel=3e-3)
    assert mom["ms"] == pytest.approx((u ** 2).mean(), rel=3e-3)
    assert mom["var"] == pytest.approx(u.var(), rel=2e-2)
    # var_x sits strictly below var: the gap is the variance of the time profile
    assert 0 < mom["var_x"] < mom["var"]


@pytest.mark.parametrize("nodes", [64, 128, 256])
def test_time_quadrature_has_converged(nodes):
    """The t integral is smooth, so the node count must not matter."""
    problem = J.HJB(8)
    a = J.space_time_moments(problem, nodes)
    b = J.space_time_moments(problem, 512)
    for key in ("mean", "ms", "var", "var_x"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)


@pytest.mark.parametrize("d", [2, 6])
def test_boundary_and_terminal_energies_against_monte_carlo(d):
    problem = J.HJB(d)
    gen = torch.Generator().manual_seed(5)
    bc = J.boundary_points(problem, 300_000, gen)
    u = J.exact_from_coords(problem, bc.numpy())
    assert J.boundary_ms(problem) == pytest.approx((u ** 2).mean(), rel=1e-2)

    tc = J.terminal_points(problem, 200_000, gen)
    ut = J.exact_from_coords(problem, tc.numpy())
    assert J.terminal_ms(problem) == pytest.approx((ut ** 2).mean(), rel=1e-2)


@pytest.mark.parametrize("d", [1, 5])
def test_residual_scale_is_the_mean_square_of_u_t(d):
    """``residual_scale`` against a finite-difference ``u_t`` of the exact solution."""
    problem = J.HJB(d)
    cs = _coords(problem, 200_000, seed=9 + d)
    h = 1e-5
    up = J.exact(problem, cs[:, :d], np.clip(cs[:, d] + h, 0, problem.T))
    um = J.exact(problem, cs[:, :d], np.clip(cs[:, d] - h, 0, problem.T))
    ut = (up - um) / (2 * h)
    assert J.residual_scale(problem) == pytest.approx((ut ** 2).mean(), rel=2e-2)


# ---------------------------------------------------------------------------
# 6. the metric, by scoring predictors whose score is known
# ---------------------------------------------------------------------------
class _Const(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, coords):
        return torch.full((coords.shape[0], 1), self.value, dtype=coords.dtype)


class _Profile(torch.nn.Module):
    """The exactly-known ``E_x[u | t]``: all of the t structure, none of the x."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, coords):
        t = coords[:, self.problem.d].detach().numpy().astype(float)
        mean = (J.p_of_t(self.problem, t).sum(axis=1) / 3.0
                + J.r_of_t(self.problem, t))
        return torch.tensor(mean, dtype=coords.dtype).reshape(-1, 1)


class _Exact(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, coords):
        v = J.exact_from_coords(self.problem, coords.detach().numpy().astype(float))
        return torch.tensor(v, dtype=coords.dtype).reshape(-1, 1)


@pytest.mark.parametrize("d", [1, 4, 16])
def test_best_constant_scores_exactly_one(d):
    """The metric's whole meaning: sd(u) is the error of the best constant."""
    problem = J.HJB(d)
    mom = J.space_time_moments(problem)
    rel, se, _, _ = J.rel_l2_mc(_Const(mom["mean"]), problem, n=400_000, seed=3,
                                moments=mom)
    assert rel == pytest.approx(1.0, abs=6 * se + 5e-3)


@pytest.mark.parametrize("d", [1, 16])
def test_zero_and_profile_baselines_match_their_closed_forms(d):
    problem = J.HJB(d)
    mom = J.space_time_moments(problem)
    base = J.baselines(problem, mom)
    zero, _, _, _ = J.rel_l2_mc(_Const(0.0), problem, n=400_000, seed=4, moments=mom)
    prof, _, _, _ = J.rel_l2_mc(_Profile(problem), problem, n=400_000, seed=4,
                                moments=mom)
    assert zero == pytest.approx(base["zero"], rel=1e-2)
    assert prof == pytest.approx(base["profile"], rel=1e-2)
    # zero is *worse* than the best constant here, unlike the heat problem where
    # it is the best constant -- which is why the metric had to change
    assert base["zero"] > 1.0
    assert base["profile"] < 1.0


def test_exact_solution_scores_zero_to_the_metrics_own_floor():
    """Feeding the metric the exact answer scores ~1e-7, not ~1e-16, and the
    reason is the floor on every number Sec. 13 reports.

    ``rel_l2_mc`` evaluates the predictor in float32, because that is what the
    trained network is. So an exact predictor still sees coordinates rounded to
    single precision, and the score it gets is that rounding -- about 1e-7
    relative. Every error in the sweep is at least 8.6e-3, five orders above
    this, but the floor is worth pinning rather than discovering later.
    """
    problem = J.HJB(3)
    rel, _, rel_rms, _ = J.rel_l2_mc(_Exact(problem), problem, n=50_000, seed=1)
    assert rel < 1e-6 and rel_rms < 1e-6


@pytest.mark.parametrize("d", [1, 8])
def test_the_two_normalizations_differ_by_the_exact_ratio(d):
    problem = J.HJB(d)
    mom = J.space_time_moments(problem)
    rel_sd, _, rel_rms, _ = J.rel_l2_mc(_Const(0.0), problem, n=100_000, seed=2,
                                        moments=mom)
    assert rel_sd / rel_rms == pytest.approx(np.sqrt(mom["ms"] / mom["var"]),
                                             rel=1e-12)


def test_metric_precision_does_not_degrade_with_d():
    """The heat problem's estimator loses precision exponentially in d. This one
    does not -- which is a result of Sec. 13, so it is pinned."""
    n = 200_000
    lo = J.mc_relative_sd(J.HJB(1), n, seed=0)[0]
    hi = J.mc_relative_sd(J.HJB(16), n, seed=0)[0]
    assert hi < lo
    assert J.mc_relative_sd(J.HJB(16), n, seed=0)[2] < 0.05   # top-1% share


# ---------------------------------------------------------------------------
# 7. sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 3, 7])
def test_samplers_land_where_they_should(d):
    problem = J.HJB(d)
    gen = torch.Generator().manual_seed(0)
    lo, hi = J.BOX

    interior = J.interior_points(problem, 500, gen)
    assert interior.shape == (500, d + 1)
    assert interior.requires_grad
    x = interior.detach()[:, :d]
    assert float(x.min()) >= lo and float(x.max()) <= hi
    t = interior.detach()[:, d]
    assert float(t.min()) >= 0.0 and float(t.max()) <= problem.T

    tc = J.terminal_points(problem, 200, gen)
    assert torch.all(tc[:, d] == problem.T)

    bc = J.boundary_points(problem, 400, gen)
    on_face = ((bc[:, :d] == lo) | (bc[:, :d] == hi)).sum(dim=1)
    assert torch.all(on_face >= 1)


def test_boundary_target_is_the_exact_solution_not_zero():
    """The heat problem's faces are all u = 0; these are not, and that is a
    difference the section leans on."""
    problem = J.HJB(4)
    gen = torch.Generator().manual_seed(1)
    bc = J.boundary_points(problem, 500, gen)
    target = J.exact_target(problem, bc)
    np.testing.assert_allclose(target.numpy().ravel(),
                               J.exact_from_coords(problem, bc.numpy()),
                               rtol=1e-5)
    assert float(target.abs().mean()) > 0.1


# ---------------------------------------------------------------------------
# 8. the trainer
# ---------------------------------------------------------------------------
def test_selection_returns_the_lowest_loss_iterate():
    problem = J.HJB(2)
    model, history, best = J.train(problem, seed=0, eval_every=10, eval_n=2000,
                                   **TINY)
    assert best["completed"]
    losses = [row[1] for row in history]
    assert best["loss"] <= min(losses) + 1e-12
    # the returned model *is* the selected one: recompute its objective
    set_again = J.train(problem, seed=0, eval_every=10, eval_n=2000,
                        select="final", **TINY)[2]
    assert set_again["step"] == best["step"]
    assert set_again["loss"] == pytest.approx(best["loss"], rel=1e-12)


def test_training_is_deterministic_in_the_seed():
    problem = J.HJB(2)
    a = J.train(problem, seed=3, eval_every=15, eval_n=2000, **TINY)[1]
    b = J.train(problem, seed=3, eval_every=15, eval_n=2000, **TINY)[1]
    np.testing.assert_array_equal(np.array(a)[:, :8], np.array(b)[:, :8])


def test_resume_reproduces_an_uninterrupted_run_exactly():
    """The sweep is only runnable here because cells resume. If resuming were
    approximate, every d = 16 number would depend on where the clock cut it."""
    problem = J.HJB(2)
    kw = dict(seed=1, eval_every=10, eval_n=2000, select="final",
              n_interior=200, n_tc=50, n_bc=50, width=16, depth=2, steps=60,
              lr=1e-3)
    straight, hist_a, best_a = J.train(problem, **kw)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "cell.pt")
        calls = 0
        while True:
            model, hist_b, best_b = J.train(problem, ckpt_path=ckpt, ckpt_every=10,
                                            deadline=time.monotonic() - 1, **kw)
            calls += 1
            if best_b.get("completed"):
                break
            assert calls < 20, "resume made no progress"
    assert calls > 2, "the deadline never actually interrupted anything"
    np.testing.assert_array_equal(np.array(hist_a)[:, :8], np.array(hist_b)[:, :8])
    assert best_a["step"] == best_b["step"]
    assert best_a["loss"] == pytest.approx(best_b["loss"], rel=0, abs=0)
    sa, sb = straight.state_dict(), model.state_dict()
    for key in sa:
        assert torch.equal(sa[key], sb[key])


def test_n_params_matches_a_real_model():
    for d in (1, 4, 16):
        model = MLP(**J.model_config(J.HJB(d), width=32, depth=3))
        got = sum(p.numel() for p in model.parameters())
        assert J.n_params(d, 32, 3) == got


# ---------------------------------------------------------------------------
# 9. summaries and the heat comparison
# ---------------------------------------------------------------------------
def test_summarize_arithmetic_on_known_rows():
    rows = [{"d": "2", "seed": str(s), "params": "7", "rel_sd": v,
             "stderr_sd": "1e-5", "rel_rms": "0.5", "rel_sd_final": "0.9",
             "base_zero": "1.8", "base_profile": "0.96",
             "train_seconds": "10", "ms_per_step": "2"}
            for s, v in enumerate(("0.1", "0.2", "0.4"))]
    out = J.summarize(rows)[0]
    assert float(out["mean"]) == pytest.approx(0.7 / 3)
    assert float(out["min"]) == pytest.approx(0.1)
    assert float(out["max"]) == pytest.approx(0.4)
    assert float(out["spread"]) == pytest.approx(4.0)


def test_heat_moments_against_monte_carlo():
    """The conversion factor used to put Sec. 11's numbers on this metric."""
    from highd_heat import HighDHeat, exact_from_coords, uniform_box_points

    for d in (1, 4):
        problem = HighDHeat(d)
        rng = np.random.default_rng(2 + d)
        cs = uniform_box_points(problem, 400_000, rng)
        u = exact_from_coords(problem, cs)
        mom = J.heat_moments(d)
        assert mom["mean"] == pytest.approx(u.mean(), rel=2e-2)
        assert mom["ms"] == pytest.approx((u ** 2).mean(), rel=3e-2)
    # and the factor falls toward 1 as d grows, because the heat solution's
    # spatial mean shrinks like (2 sqrt2 / pi)^d relative to its rms
    assert J.heat_rms_over_sd(1) > J.heat_rms_over_sd(16) > 1.0


# ---------------------------------------------------------------------------
# 10. the committed log
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sweep_rows():
    return read_csv(J.SWEEP_CSV)


def test_committed_sweep_covers_every_cell(sweep_rows):
    got = {(int(r["d"]), int(r["seed"])) for r in sweep_rows}
    assert got == {(d, s) for d in J.SWEEP_DIMS for s in J.SEEDS}


def test_committed_rows_agree_with_the_declared_budget(sweep_rows):
    for r in sweep_rows:
        d = int(r["d"])
        assert int(r["params"]) == J.n_params(d, J.BUDGET["width"], J.BUDGET["depth"])
        assert 0 <= int(r["best_step"]) <= J.BUDGET["steps"] + 1
        mom = J.space_time_moments(J.HJB(d))
        # the CSV stores 7 significant figures, so 1e-6 is the tightest
        # portable comparison against a recomputed quadrature
        assert float(r["exact_sd"]) == pytest.approx(np.sqrt(mom["var"]), rel=1e-6)
        assert float(r["exact_rms"]) == pytest.approx(np.sqrt(mom["ms"]), rel=1e-6)
        base = J.baselines(J.HJB(d), mom)
        assert float(r["base_zero"]) == pytest.approx(base["zero"], rel=1e-5)
        assert float(r["base_profile"]) == pytest.approx(base["profile"], rel=1e-5)


def test_committed_trace_matches_the_sweep(sweep_rows):
    traces = read_csv(J.TRACE_CSV)
    cells = {(int(t["d"]), int(t["seed"])) for t in traces}
    assert cells == {(int(r["d"]), int(r["seed"])) for r in sweep_rows}
    for t in traces:
        assert 0 <= int(t["step"]) <= J.BUDGET["steps"]


def test_committed_check_log_is_at_machine_precision():
    """The verification log ships; if it ever stopped being machine zero the
    ground truth would be wrong and every error in Sec. 13 with it."""
    for r in read_csv(J.CHECK_CSV):
        assert float(r["pde_residual"]) < 1e-12
        assert float(r["terminal"]) < 1e-12
        assert float(r["r_vs_quadrature"]) < 1e-12
        assert float(r["cole_hopf_rel"]) < 1e-10
        assert float(r["quad_64_vs_256"]) < 1e-10


def test_committed_metric_log_shows_precision_improving_with_d():
    rows = read_csv(J.METRIC_CSV)
    for n in sorted({int(r["n"]) for r in rows}):
        cells = sorted([r for r in rows if int(r["n"]) == n], key=lambda r: int(r["d"]))
        assert float(cells[-1]["pred_rel_sd"]) < float(cells[0]["pred_rel_sd"])
        # and the estimator's prediction is confirmed by the observed scatter
        for c in cells:
            assert float(c["obs_rel_sd"]) == pytest.approx(float(c["pred_rel_sd"]),
                                                           rel=1.2)
