"""The heat equation in d dimensions, with an exact solution at every d.

This is the setup for the repo's high-dimensional study. The Limitations
section has said since v1.0 that the places PINNs earn their keep are inverse
problems (measured in Sec. 8), irregular geometry, and **high dimension** -- and
that the last two are not demonstrated here. A fair high-d comparison needs a
problem where the truth is known at every d, because otherwise "the PINN's
answer" can only be scored against another approximation, and at d = 16 there
is no mesh solution to score it against. So the ground truth has to be closed
form, and that is what this module builds.

The problem
-----------
On the unit cube ``x in [0,1]^d``, ``t in [0,1]``:

    u_t = alpha_d * Laplacian_x u,
    u = 0 on all 2d faces          (homogeneous Dirichlet),
    u(x, 0) = sum_m a_m prod_i sin(k_{m,i} pi x_i).

Each product ``phi_k(x) = prod_i sin(k_i pi x_i)`` is an eigenfunction of the
Dirichlet Laplacian on the cube:

    Laplacian phi_k = -pi^2 (sum_i k_i^2) phi_k = -pi^2 |k|^2 phi_k,

because the Laplacian is a sum of unmixed second derivatives and each factor
depends on one coordinate only. The heat semigroup therefore multiplies mode k
by ``exp(-alpha pi^2 |k|^2 t)`` and

    u(x, t) = sum_m a_m phi_{k_m}(x) exp(-alpha_d pi^2 |k_m|^2 t)

is exact at every d -- not a truncation, not a fine-grid reference. This is the
d-dimensional version of the Sec. 1 argument, and at d = 1 it *is* Sec. 1:
``tests/test_highd_heat.py`` pins ``exact`` against ``heat.heat_exact`` and this
module's residual against ``heat.heat_residual``, elementwise.

Two choices that are not forced by the mathematics
--------------------------------------------------
**The diffusivity is scaled, alpha_d = alpha_1 / d.** With alpha fixed the
fundamental mode k = (1,...,1) has decay rate alpha pi^2 d, so at d = 16 and
alpha = 0.05 the solution falls to ``exp(-7.9) = 3.7e-4`` of its initial size by
t = 1. The sweep would then be measuring how fast the target goes to zero: most
of the space-time box holds a field that is numerically nothing, and "accuracy
improves with d" would be an artifact. Scaling alpha as 1/d holds the
fundamental's rate at ``alpha_1 pi^2`` for every d, so the temporal structure of
the target is the same object at d = 1 and d = 16. At d = 1 the scaling is the
identity, which is why the Sec. 1 pin still works.

**The target is not permutation-symmetric.** The default initial condition is

    prod_i sin(pi x_i)  +  0.5 sin(2 pi x_1) prod_{i>1} sin(pi x_i),

i.e. the fundamental plus a second mode that doubles the frequency along axis 1
only. A single all-ones mode would be symmetric under permuting the coordinates,
and a target lying in that much smaller family invites the objection that the
network found the symmetry rather than the solution. One distinguished axis
costs nothing and removes the objection.

The second choice has a consequence worth stating rather than discovering later:
the two decay rates are ``alpha_d pi^2 d`` and ``alpha_d pi^2 (d + 3)``, whose
*ratio* is ``(d + 3)/d`` -- 4.0 at d = 1 but 1.19 at d = 16. The d - 1 shared
unit modes dominate ``|k|^2`` in high d, so the multi-scale rate separation of
Sec. 1 is a low-dimensional feature and is not something this family holds fixed
across d. Nothing here depends on it; it is stated so it is not read into the
d-sweep as a controlled variable.

Measuring the error when there is no grid
-----------------------------------------
Sec. 1 evaluates the relative L2 error on a 101 x 101 grid. At d = 16 that grid
has 101^16 points, which is the curse this week is about, so the error is a
Monte Carlo estimate instead -- and then the *error bar on the error* becomes a
real quantity rather than a formality. Two things help:

- The denominator is closed form. The eigenfunctions are orthogonal, with
  ``<phi_k, phi_k> = 2^-d`` on the cube, so the exact space-time mean square of
  u is a finite sum (:func:`exact_ms`). Only the numerator is sampled.
- The numerator's standard error is computed and reported with every number
  (:func:`rel_l2_mc`), by the delta method for the square root.

That precision is not free, because ``prod_i sin(pi x_i)`` is log-normally
spread in high d: its typical value at a uniform point is ``2^-d`` while its
root mean square is ``2^(-d/2)``, so the L2 norm is carried by a thin set of
points near the centre of the cube. :func:`concentration` measures that directly
(the share of the mean square carried by the largest 1% of samples), and
:func:`main` reports it alongside the sample size each d needs. Day 10's sweep
quotes errors from this estimator, so its resolution limit is measured here
first rather than assumed.

Run:  python experiments/highd_heat.py --verify   # the d=1 pin, a real solve
      python experiments/highd_heat.py --metric   # estimator precision vs d
      python experiments/highd_heat.py --quick    # tiny d=2 smoke check
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import write_csv
from pinn import derivatives as D
from pinn.model import MLP, set_seed

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
ALPHA_1 = 0.05          # the Sec. 1 diffusivity; alpha_d = ALPHA_1 / d
T_RANGE = (0.0, 1.0)
BOX = (0.0, 1.0)        # every spatial axis spans [0, 1]


class HighDHeat:
    """The d-dimensional Dirichlet heat problem and its closed-form solution.

    Parameters
    ----------
    d : int
        Number of spatial dimensions.
    terms : sequence of (multi-index, amplitude), optional
        The initial condition as ``sum_m a_m prod_i sin(k_{m,i} pi x_i)``.
        Each multi-index must have length ``d`` and strictly positive integer
        entries (a zero entry would make that factor vanish identically).
        Defaults to the fundamental plus a doubled axis-1 mode; see the module
        docstring for why the default is not permutation-symmetric.
    alpha : float, optional
        Diffusivity. Defaults to ``ALPHA_1 / d`` (see the module docstring);
        pass a value to override, which is what the d = 1 pin against Sec. 1
        does not need to do, since the scaling is the identity there.
    t_range : (float, float)
        Time interval.

    Attributes
    ----------
    rates : ndarray
        ``alpha pi^2 |k_m|^2`` for each term, the decay rate of that mode.
    """

    def __init__(self, d, terms=None, alpha=None, t_range=T_RANGE):
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self.alpha = float(ALPHA_1 / d) if alpha is None else float(alpha)
        self.t_range = (float(t_range[0]), float(t_range[1]))
        if self.t_range[1] <= self.t_range[0]:
            raise ValueError(f"empty time interval {self.t_range}")

        if terms is None:
            terms = default_terms(self.d)
        self.modes = np.array([k for k, _ in terms], dtype=float)     # (M, d)
        self.amps = np.array([a for _, a in terms], dtype=float)      # (M,)
        if self.modes.ndim != 2 or self.modes.shape[1] != self.d:
            raise ValueError(f"multi-indices must have length d = {self.d}")
        if np.any(self.modes < 1) or np.any(self.modes != np.round(self.modes)):
            raise ValueError("multi-index entries must be positive integers")
        self.rates = self.alpha * np.pi ** 2 * (self.modes ** 2).sum(axis=1)

    def __repr__(self):
        return (f"HighDHeat(d={self.d}, alpha={self.alpha:.6g}, "
                f"modes={self.modes.astype(int).tolist()}, "
                f"amps={self.amps.tolist()})")


def default_terms(d):
    """The default initial condition: fundamental + a doubled first axis.

    ``[prod_i sin(pi x_i), 0.5 sin(2 pi x_1) prod_{i>1} sin(pi x_i)]``. At
    d = 1 this is the two-mode subset ``sin(pi x) + 0.5 sin(2 pi x)`` of the
    Sec. 1 initial condition.
    """
    fundamental = tuple([1] * d)
    doubled = tuple([2] + [1] * (d - 1))
    return [(fundamental, 1.0), (doubled, 0.5)]


def sec1_terms():
    """The Sec. 1 initial condition as a d = 1 term list, for the pin.

    ``experiments/heat.py`` uses modes (1, 2, 3) with amplitudes
    (1, 0.5, 0.25) at alpha = 0.05. Expressed here, ``exact`` must reproduce
    ``heat.heat_exact`` to floating point -- which is the check that this
    module's general machinery is the same mathematics as the one already
    trusted and shipped.
    """
    return [((1,), 1.0), ((2,), 0.5), ((3,), 0.25)]


# ---------------------------------------------------------------------------
# The exact solution
# ---------------------------------------------------------------------------
def exact(problem, X, t):
    """u(x, t) in closed form. ``X`` is (N, d) and ``t`` is (N,) or a scalar.

    Returns an (N,) float64 array. Evaluated term by term as
    ``a_m prod_i sin(k_i pi x_i) exp(-rate_m t)``; the product runs over the d
    spatial factors, so this costs O(N d M) and is exact at any d.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != problem.d:
        raise ValueError(f"X must be (N, {problem.d}), got {X.shape}")
    t = np.broadcast_to(np.asarray(t, dtype=float), (X.shape[0],))

    out = np.zeros(X.shape[0])
    for k, a, rate in zip(problem.modes, problem.amps, problem.rates):
        spatial = np.prod(np.sin(np.pi * k[None, :] * X), axis=1)
        out = out + a * spatial * np.exp(-rate * t)
    return out


def exact_from_coords(problem, coords):
    """``exact`` on stacked coordinates ``[x_1 .. x_d, t]`` of shape (N, d+1)."""
    coords = np.asarray(coords, dtype=float)
    return exact(problem, coords[:, : problem.d], coords[:, problem.d])


def initial_condition(problem, X):
    """u(x, 0) as a torch tensor, shape (N, 1), for the IC data loss."""
    out = torch.zeros(X.shape[0], 1, dtype=X.dtype)
    for k, a in zip(problem.modes, problem.amps):
        kt = torch.tensor(k, dtype=X.dtype)
        spatial = torch.prod(torch.sin(np.pi * kt[None, :] * X), dim=1, keepdim=True)
        out = out + float(a) * spatial
    return out


def residual(problem, u, coords):
    """PDE residual ``r = u_t - alpha Laplacian_x u`` at the collocation points.

    ``coords`` is (N, d+1) laid out as ``[x_1 .. x_d, t]``, so the spatial
    Laplacian sums columns ``0 .. d-1`` and the time derivative is column d.
    Summing over *all* columns would silently add a u_tt term, which is why
    :func:`pinn.derivatives.laplacian` takes an explicit ``dims``.
    """
    d = problem.d
    u_t = D.partial(u, coords, d)
    lap = D.laplacian(u, coords, dims=range(d))
    return u_t - problem.alpha * lap


# ---------------------------------------------------------------------------
# Exact norms: the denominator of every relative error, in closed form
# ---------------------------------------------------------------------------
def exact_ms(problem):
    """Space-time mean square of the exact solution, exactly.

    The cube has unit volume and the eigenfunctions are orthogonal on it,

        int_{[0,1]^d} phi_k phi_k' dx = delta_{kk'} 2^-d,

    (each factor contributes int_0^1 sin(k pi x) sin(k' pi x) dx = delta/2 for
    positive integer k, k'), so the cross terms drop and

        mean_x u(x,t)^2 = 2^-d sum_m a_m^2 exp(-2 rate_m t).

    Averaging over ``t in [t0, t1]`` integrates each exponential in closed form.
    Returning the mean square rather than the norm keeps the caller's arithmetic
    in one place; :func:`exact_rms` is the square root.

    This is why the relative errors below have noise in the numerator only.
    """
    t0, t1 = problem.t_range
    span = t1 - t0
    r = problem.rates
    # (1/span) int_{t0}^{t1} exp(-2 r t) dt
    time_avg = (np.exp(-2 * r * t0) - np.exp(-2 * r * t1)) / (2 * r * span)
    return float(2.0 ** (-problem.d) * np.sum(problem.amps ** 2 * time_avg))


def exact_rms(problem):
    """Root mean square of the exact solution over the space-time box."""
    return float(np.sqrt(exact_ms(problem)))


# ---------------------------------------------------------------------------
# Collocation sampling in d dimensions
# ---------------------------------------------------------------------------
def _uniform(n, d, gen, dtype=torch.float32):
    return torch.rand(n, d, generator=gen, dtype=dtype)


def interior_points(problem, n, gen):
    """(n, d+1) uniform points in the space-time interior, grad enabled.

    Uniform on the cube times uniform in time. The residual differentiates the
    network at these points, so ``requires_grad_`` is set here exactly as
    ``pinn.losses.interior_points`` does in 1D.
    """
    t0, t1 = problem.t_range
    x = _uniform(n, problem.d, gen)
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    coords = torch.cat([x, t], dim=1)
    coords.requires_grad_(True)
    return coords


def initial_points(problem, n, gen):
    """(n, d+1) points on the initial slice t = t0, x uniform on the cube."""
    x = _uniform(n, problem.d, gen)
    t = torch.full((n, 1), float(problem.t_range[0]))
    return torch.cat([x, t], dim=1)


def boundary_points(problem, n, gen):
    """(n, d+1) points uniform on the *union* of the 2d faces of the cube.

    In 1D ``pinn.losses.boundary_points`` returns the two walls separately, so a
    caller can put different data on each. That does not survive into high d: at
    d = 16 there are 32 faces, and a fixed per-face budget makes the boundary
    cost grow linearly in d for a condition that is the same (u = 0) on all of
    them. Instead draw the face uniformly -- pick an axis and a side, pin that
    coordinate to 0 or 1, leave the rest uniform -- which is exactly the uniform
    measure on the boundary, since all 2d faces have equal area.

    The homogeneous Dirichlet target is 0 on every face, so no target tensor is
    returned; every mode has a ``sin(k_i pi x_i)`` factor that vanishes there,
    and ``tests/test_highd_heat.py`` checks that on the exact solution.
    """
    t0, t1 = problem.t_range
    d = problem.d
    x = _uniform(n, d, gen)
    axis = torch.randint(0, d, (n,), generator=gen)
    side = torch.randint(0, 2, (n,), generator=gen).to(x.dtype)
    x[torch.arange(n), axis] = side
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    return torch.cat([x, t], dim=1)


def uniform_box_points(problem, n, rng):
    """(n, d+1) numpy points uniform on the space-time box, for evaluation.

    Kept on a numpy Generator rather than a torch one so the evaluation sample
    is independent of the training sample by construction, not by seed hygiene.
    """
    t0, t1 = problem.t_range
    x = rng.random((n, problem.d))
    t = t0 + (t1 - t0) * rng.random((n, 1))
    return np.concatenate([x, t], axis=1)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def predict(model, coords, chunk=100_000):
    """Evaluate the network on (N, d+1) numpy coordinates -> (N,) float64."""
    out = np.empty(coords.shape[0])
    with torch.no_grad():
        for i in range(0, coords.shape[0], chunk):
            block = torch.tensor(coords[i : i + chunk], dtype=torch.float32)
            out[i : i + chunk] = model(block).numpy().ravel().astype(float)
    return out


def rel_l2_mc(model, problem, n=200_000, seed=0, return_parts=False):
    """Relative L2 error by Monte Carlo, with its standard error.

    Returns ``(rel, se)``: the estimate of
    ``||u_theta - u||_{L2(box)} / ||u||_{L2(box)}`` and the standard error of
    that estimate. The denominator is :func:`exact_rms`, which is exact, so all
    the noise is in the numerator.

    The numerator is ``sqrt(mean_i e_i)`` with ``e_i = (u_theta - u)^2`` at n
    uniform points. ``mean_i e_i`` has standard error ``sd(e)/sqrt(n)``, and the
    square root's follows by the delta method,

        se(sqrt(m)) = se(m) / (2 sqrt(m)),

    which is the right first-order propagation as long as the relative error of
    m is small -- reported so a caller can check that it is. With
    ``return_parts`` the per-point squared errors come back too, for
    :func:`concentration`.
    """
    rng = np.random.default_rng(seed)
    coords = uniform_box_points(problem, n, rng)
    e = (predict(model, coords) - exact_from_coords(problem, coords)) ** 2

    m = float(e.mean())
    se_m = float(e.std(ddof=1) / np.sqrt(n))
    denom = exact_rms(problem)
    rel = np.sqrt(m) / denom
    se = se_m / (2 * np.sqrt(m)) / denom if m > 0 else 0.0
    if return_parts:
        return float(rel), float(se), e
    return float(rel), float(se)


def concentration(values, top_frac=0.01):
    """Share of a sum carried by its largest ``top_frac`` of entries.

    A uniform-Monte-Carlo mean is only as precise as its integrand is spread
    out. On the cube ``prod_i sin(pi x_i)`` has root mean square ``2^(-d/2)``
    but *typical* value ``2^-d`` (its log is a sum of d iid draws with mean
    ``-ln 2``), so in high d the mean square is carried by a thin set of points
    near the centre and a uniform sample sees it rarely. This is the diagnostic
    for that: at d = 1 the top 1% of points carry a few percent of the total, and
    the share climbs with d.

    The size of the effect is derivable, not just observable. For one product
    mode the integrand is ``v = prod_i sin^2(pi x_i)``, whose factors are
    independent under a uniform draw with ``E[sin^2] = 1/2`` and
    ``E[sin^4] = 3/8``, so

        Var(v) / E[v]^2 = (3/2)^d - 1,

    and the relative standard error of an n-point mean is
    ``sqrt(((3/2)^d - 1)/n)`` -- exponential in d. That law is checked directly
    in ``tests/test_highd_heat.py`` and is what :func:`metric_study` measures on
    the actual target.
    """
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(round(top_frac * v.size)))
    total = v.sum()
    return float(v[:k].sum() / total) if total > 0 else 0.0


def mc_relative_sd(problem, n, seed=0):
    """Relative sd of the *uniform-MC estimate of the exact mean square*.

    A property of the target and the sampler, with no network in it: draw n
    points, form ``mean_i u(x_i, t_i)^2``, and report ``sd/mean/sqrt(n)`` from
    the same sample -- i.e. how well n uniform points can pin down a quantity
    :func:`exact_ms` already knows exactly. That makes it a calibrated probe of
    the estimator, since the answer it is estimating is available in closed
    form and can be compared directly.

    Returns ``(rel_sd, ratio_to_exact, top1pct_share)``, where ``ratio_to_exact``
    is the sampled mean square divided by :func:`exact_ms` -- 1.0 exactly if the
    sample happened to be perfect, and its scatter across seeds is the honest
    check on ``rel_sd``.
    """
    rng = np.random.default_rng(seed)
    coords = uniform_box_points(problem, n, rng)
    u2 = exact_from_coords(problem, coords) ** 2
    m = float(u2.mean())
    rel_sd = float(u2.std(ddof=1) / np.sqrt(n) / m) if m > 0 else float("nan")
    return rel_sd, m / exact_ms(problem), concentration(u2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def model_config(problem, width=128, depth=4):
    """Constructor kwargs for the field. Input dimension is d + 1."""
    return dict(in_dim=problem.d + 1, out_dim=1, width=width, depth=depth,
                activation="tanh")


def train(
    problem,
    n_interior=4000,
    width=128,
    depth=4,
    steps=5000,
    lr=1e-3,
    n_ic=400,
    n_bc=400,
    seed=0,
    w_ic=1.0,
    w_bc=1.0,
    eval_every=500,
    eval_n=50_000,
    select="best_loss",
    verbose=False,
):
    """Train a d-dimensional heat PINN with Adam; return (model, history, best).

    Same recipe as Sec. 1 -- fixed collocation set, soft IC and BC penalties,
    Adam -- lifted to d spatial dimensions. ``n_bc`` is the total number of
    boundary points across all 2d faces (see :func:`boundary_points`), not a
    per-face count, so the boundary budget does not grow with d.

    History rows are
    ``(step, loss, loss_r, loss_ic, loss_bc, rel_l2, se, train_seconds)``.
    The error is a Monte Carlo estimate on ``eval_n`` points and carries its own
    standard error; ``eval_n`` is smaller than the default in :func:`rel_l2_mc`
    because this runs inside the training loop.

    ``train_seconds`` is cumulative wall clock spent on the optimization only --
    the evaluation calls, which exist for the log and not for the method, are
    excluded from it. That column is what makes a run's *cost to reach an
    accuracy* readable off the history, which is the comparison Sec. 11 needs;
    including the instrumentation in it would charge the PINN for measurements
    a user of the method would never take.

    **Which iterate is returned, and why it is not the last one.** Adam on this
    objective does not settle -- it spikes late in training and recovers. Sec.
    1's own committed history is the evidence: over its last 1500 steps the
    relative L2 it logs reads 3.43e-3, 2.41e-2, 4.83e-2, 3.62e-3, a 14x band
    sampled every 500 steps, and the headline number is the last of those. A
    d-sweep that quoted final iterates would be measuring the phase of that
    oscillation as much as the effect of d.

    So the returned model is the one with the lowest *training loss* seen, which
    is a legitimate selection: the loss is the objective being optimized and
    contains no ground truth, so nothing about the exact solution leaks into the
    choice. (Selecting on the relative L2 instead would be choosing the answer by
    looking at it -- ``gp-from-scratch``'s ``adam_maximize`` makes the same
    distinction for the same reason.) Pass ``select="final"`` for the last
    iterate. ``best`` reports which step won and at what loss, so a caller can
    see how far from the end it was, along with ``final_loss``, the objective at
    the last iterate. A winning step of ``steps + 1`` means the last update's
    own output won, which is the ordinary case on a run that did not spike.
    """
    if select not in ("best_loss", "final"):
        raise ValueError(f"select must be 'best_loss' or 'final', got {select!r}")
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    interior = interior_points(problem, n_interior, gen)
    ic = initial_points(problem, n_ic, gen)
    ic_target = initial_condition(problem, ic[:, : problem.d])
    bc = boundary_points(problem, n_bc, gen)
    bc_target = torch.zeros(bc.shape[0], 1)

    model = MLP(**model_config(problem, width, depth))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best = dict(step=-1, loss=float("inf"))
    best_state = None
    train_seconds = 0.0
    for step in range(steps + 1):
        t_step = time.perf_counter()
        opt.zero_grad()
        r = residual(problem, model(interior), interior)
        loss_r = torch.mean(r ** 2)
        loss_ic = torch.mean((model(ic) - ic_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()

        # Snapshot *before* the update, because ``loss`` is the objective at the
        # current parameters and ``opt.step()`` is about to replace them. Taking
        # the copy after the step would pair a loss with the weights that came
        # after it -- an off-by-one that is invisible until the two are compared,
        # which is what ``test_loss_selection_...`` does by recomputing the loss
        # on the returned model.
        value = float(loss.detach().item())
        if value < best["loss"]:
            best = dict(step=step, loss=value)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        opt.step()
        train_seconds += time.perf_counter() - t_step

        if step % eval_every == 0 or step == steps:
            err, se = rel_l2_mc(model, problem, n=eval_n, seed=12345)
            history.append((step, value, float(loss_r.item()),
                            float(loss_ic.item()), float(loss_bc.item()), err, se,
                            train_seconds))
            if verbose:
                print(f"  step {step:5d}  loss {value:.3e}  "
                      f"(r {loss_r.item():.2e} ic {loss_ic.item():.2e} "
                      f"bc {loss_bc.item():.2e})  relL2 {err:.4f} +- {se:.4f}")

    # The loop only ever sees the loss *before* each update, so the parameters
    # produced by the last update have never been scored. Score them here, with
    # one extra forward pass and no step, and let them compete: otherwise the
    # final iterate can be the best one in the run and the selection would
    # return something worse -- which is exactly what the first version did.
    final_loss = _objective(problem, model, interior, ic, ic_target, bc, bc_target,
                            w_ic, w_bc)
    if final_loss < best["loss"]:
        best = dict(step=steps + 1, loss=final_loss)
        best_state = None            # the final weights are already in the model
    best["final_loss"] = final_loss
    best["train_seconds"] = train_seconds

    # The selected parameters travel with the summary, so a caller that wants
    # *both* iterates scored -- Sec. 11's sweep does, since the gap between them
    # is one of its measurements -- does not have to train the same run twice.
    # ``None`` means the final iterate won and is already in the model.
    best["state_dict"] = best_state

    if select == "best_loss" and best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


def _objective(problem, model, interior, ic, ic_target, bc, bc_target, w_ic, w_bc):
    """The training objective at the model's current parameters, as a float."""
    r = residual(problem, model(interior), interior)
    total = (torch.mean(r ** 2)
             + w_ic * torch.mean((model(ic) - ic_target) ** 2)
             + w_bc * torch.mean((model(bc) - bc_target) ** 2))
    return float(total.detach())


# ---------------------------------------------------------------------------
# Verification: d = 1 against the shipped Sec. 1 result
# ---------------------------------------------------------------------------
VERIFY = dict(n_interior=4000, width=128, depth=4, steps=5000, n_ic=400, n_bc=400)


def verify_d1(verbose=True):
    """Solve the *Sec. 1 problem* with the d-dimensional code and score it.

    New code that reduces to old code is the cheapest real check available. The
    algebraic half of that reduction is in the tests (this module's ``exact``
    equals ``heat.heat_exact``, and its residual equals ``heat.heat_residual``,
    elementwise). This is the other half: an actual solve of the same problem at
    the same budget as Sec. 1's default, scored two ways --

    - the Monte Carlo relative L2 this module uses at every d, and
    - Sec. 1's own 101 x 101 grid metric, via ``heat.rel_l2_error``,

    which also measures what the change of metric costs, since the two differ
    only in how the same norm is approximated. The collocation points are drawn
    by different code from Sec. 1's, so the two runs are not the same seeded run
    and the numbers are not expected to agree to digits; what is being checked
    is that the general implementation lands in the same accuracy class, and
    that the two metrics agree with each other on the same network.

    Both selections are scored, because the difference between them is the
    finding this run produced: the final iterate is a sample of an oscillating
    tail (see :func:`train`), and Sec. 1's shipped 3.6e-3 is its own final
    iterate, which its committed log happens to catch between spikes.
    """
    from heat import ALPHA, rel_l2_error

    problem = HighDHeat(1, terms=sec1_terms(), alpha=ALPHA)
    if verbose:
        print(f"d = 1 pin against Sec. 1: {problem}")
        print(f"  exact rms over the box: {exact_rms(problem):.6f}")

    t0 = time.time()
    model, history, best = train(problem, verbose=verbose, select="final", **VERIFY)
    secs = time.time() - t0

    final_mc, final_se = rel_l2_mc(model, problem, n=400_000, seed=7)
    final_grid = rel_l2_error(model)

    # Re-run under loss selection rather than reaching into the loop's state, so
    # what is scored is exactly what a caller of train() gets.
    sel, _, best2 = train(problem, select="best_loss", **VERIFY)
    assert best2["step"] == best["step"]
    mc, se = rel_l2_mc(sel, problem, n=400_000, seed=7)
    grid = rel_l2_error(sel)

    if verbose:
        print(f"\n  trained in {secs:.0f}s")
        print(f"  lowest training loss at step {best['step']} "
              f"({best['loss']:.3e}); final step loss {best['final_loss']:.3e}")
        print(f"  final iterate:      MC {final_mc:.4e} +- {final_se:.1e}   "
              f"grid {final_grid:.4e}")
        print(f"  lowest-loss iterate: MC {mc:.4e} +- {se:.1e}   grid {grid:.4e}")
        print(f"  Sec. 1 shipped value = 3.6e-3 (its own final iterate, "
              f"its own sampler)")
    return dict(mc=mc, se=se, grid=grid, final_mc=final_mc, final_grid=final_grid,
                best_step=best["step"], best_loss=best["loss"],
                final_loss=best["final_loss"], seconds=secs, history=history)


# ---------------------------------------------------------------------------
# The metric study: what the Monte Carlo error metric costs in high d
# ---------------------------------------------------------------------------
METRIC_DIMS = (1, 2, 4, 8, 16)
METRIC_N = (10_000, 100_000, 1_000_000)


def metric_study(dims=METRIC_DIMS, sizes=METRIC_N, seeds=(0, 1, 2, 3, 4)):
    """How precise is the uniform-MC L2 metric, as a function of d and n?

    Day 10 reports relative L2 errors across d from :func:`rel_l2_mc`. If the
    estimator's own noise grows with d, a trend in those errors could be the
    estimator rather than the PINN -- so measure the estimator first, on the
    exact solution, where the answer is known in closed form.
    """
    rows = []
    for d in dims:
        problem = HighDHeat(d)
        for n in sizes:
            rel_sds, ratios, tops = [], [], []
            for s in seeds:
                rel_sd, ratio, top = mc_relative_sd(problem, n, seed=100 * s + d)
                rel_sds.append(rel_sd)
                ratios.append(ratio)
                tops.append(top)
            # Spread of the estimate itself across seeds, against the
            # per-sample prediction: these should agree if the delta method and
            # the sd estimate are both behaving.
            observed = float(np.std(ratios, ddof=1))
            rows.append({
                "d": d, "n": n,
                "exact_ms": f"{exact_ms(problem):.6e}",
                "pred_rel_sd": f"{np.mean(rel_sds):.4e}",
                "obs_rel_sd": f"{observed:.4e}",
                "worst_ratio": f"{max(abs(r - 1) for r in ratios):.4e}",
                "top1pct_share": f"{np.mean(tops):.4f}",
            })
            print(f"d={d:3d} n={n:8d}  predicted rel sd {np.mean(rel_sds):.3e}  "
                  f"observed {observed:.3e}  top-1% share {np.mean(tops):.3f}")
    return rows


def report_metric(rows):
    print("\n" + "=" * 70)
    print("Uniform-MC estimate of ||u||^2, against the closed form")
    print("=" * 70)
    print(f"{'d':>3} {'n':>9} {'pred rel sd':>12} {'obs rel sd':>12} "
          f"{'worst |ratio-1|':>16} {'top 1% share':>13}")
    for r in rows:
        print(f"{r['d']:>3} {int(r['n']):>9} {float(r['pred_rel_sd']):>12.3e} "
              f"{float(r['obs_rel_sd']):>12.3e} {float(r['worst_ratio']):>16.3e} "
              f"{float(r['top1pct_share']):>13.4f}")


def main(quick=False, verify=False, metric=False):
    if quick:
        problem = HighDHeat(2)
        print(f"[quick] {problem}")
        print(f"        exact rms {exact_rms(problem):.6f}")
        model, hist, best = train(problem, n_interior=800, width=32, steps=300,
                                  eval_every=100, eval_n=20_000, verbose=True)
        print(f"        lowest training loss at step {best['step']}: {best['loss']:.3e}")
        return

    if verify:
        res = verify_d1()
        write_csv("highd_verify.csv",
                  ["metric", "value"],
                  [{"metric": "rel_l2_mc_bestloss", "value": f"{res['mc']:.6e}"},
                   {"metric": "rel_l2_mc_stderr", "value": f"{res['se']:.6e}"},
                   {"metric": "rel_l2_grid_bestloss", "value": f"{res['grid']:.6e}"},
                   {"metric": "rel_l2_mc_final", "value": f"{res['final_mc']:.6e}"},
                   {"metric": "rel_l2_grid_final", "value": f"{res['final_grid']:.6e}"},
                   {"metric": "best_loss_step", "value": f"{res['best_step']}"},
                   {"metric": "best_loss", "value": f"{res['best_loss']:.6e}"},
                   {"metric": "final_loss", "value": f"{res['final_loss']:.6e}"},
                   {"metric": "seconds", "value": f"{res['seconds']:.1f}"}])
        write_csv("highd_verify_trace.csv",
                  ["step", "loss", "loss_r", "loss_ic", "loss_bc", "rel_l2", "stderr",
                   "train_seconds"],
                  [{"step": s, "loss": f"{l:.6e}", "loss_r": f"{lr_:.6e}",
                    "loss_ic": f"{li:.6e}", "loss_bc": f"{lb:.6e}",
                    "rel_l2": f"{e:.6e}", "stderr": f"{se:.6e}",
                    "train_seconds": f"{ts:.4f}"}
                   for s, l, lr_, li, lb, e, se, ts in res["history"]])
        return

    if metric:
        rows = metric_study()
        write_csv("highd_metric.csv",
                  ["d", "n", "exact_ms", "pred_rel_sd", "obs_rel_sd",
                   "worst_ratio", "top1pct_share"], rows)
        report_metric(rows)
        return

    print(__doc__.strip().splitlines()[0])
    print("\nNothing to run without a mode. Pass one of:")
    print("  --verify   the d = 1 pin against Sec. 1 (a real solve, ~2 min)")
    print("  --metric   estimator precision vs d (no training)")
    print("  --quick    a tiny d = 2 smoke check")
    print("\nThe d-sweep itself is Day 10's; this module is its problem setup.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="train at d=1 on the Sec. 1 problem and score it")
    ap.add_argument("--metric", action="store_true",
                    help="measure the Monte Carlo metric's precision vs d")
    args = ap.parse_args()
    main(quick=args.quick, verify=args.verify, metric=args.metric)
