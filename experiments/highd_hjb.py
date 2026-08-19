"""A second high-dimensional PDE: a linear-quadratic HJB equation.

Sections 10-12 measured the mesh, the network and their crossover on *one*
problem -- the d-dimensional heat equation -- and reached a negative: at a fixed
budget the PINN's relative L2 runs 8.2e-4 -> 1.04 over d = 1 -> 16, and the
crossover where the mesh becomes dearer arrives 6 to 15 dimensions after the
network has stopped delivering the accuracy. One PDE is an anecdote. This module
is the second one, chosen so that as many of the heat problem's incidental
features as possible are *different*, and the shared conclusion (if there is
one) is therefore about the method rather than about that PDE.

The equation
------------
On ``x in [-1,1]^d``, ``t in [0,T]``, with a *terminal* condition at t = T:

    u_t + nu Laplacian_x u - lambda |grad_x u|^2 + sum_i q_i x_i^2 = 0,
    u(x, T) = sum_i c_i x_i^2,
    u = (the exact solution) on all 2d faces.

This is the Hamilton-Jacobi-Bellman equation of a linear-quadratic stochastic
control problem: state ``dX = a dt + sqrt(2 nu) dW``, running cost
``sum_i q_i X_i^2 + |a|^2/(4 lambda)``, terminal cost ``sum_i c_i X_i^2``, and
``u`` the value function. Minimizing ``a . grad u + |a|^2/(4 lambda)`` over the
control gives ``a* = -2 lambda grad u`` and the ``-lambda |grad u|^2`` term
above, so the nonlinearity is not decorative -- it is the optimizer inside the
equation.

What is different from the heat problem, and why each difference was wanted
---------------------------------------------------------------------------
- **Nonlinear.** ``|grad u|^2`` is quadratic in the derivative the network
  supplies. Sec. 1-12 are linear in u everywhere except Burgers (Sec. 2), which
  is 1D.
- **Backward in time.** The data sits at t = T and information propagates to
  t = 0. Every other parabolic problem in this repo marches forward.
- **Inhomogeneous Dirichlet data.** The heat problem's faces are all u = 0, so
  its boundary loss can be satisfied by shrinking the network. Here the faces
  carry the exact solution, which is *large* there (the target grows away from
  the origin), so the boundary term cannot be satisfied by doing nothing.
- **A target that does not vanish with d.** ``prod_i sin(pi x_i)`` has rms
  ``2^(-d/2)``, so the heat sweep's target shrinks 170x from d = 1 to d = 16 and
  its uniform-MC error metric loses precision exponentially (Sec. 11 had to
  derive the estimator's standard error before it could quote anything). The
  value function here is a quadratic form: its spatial fluctuation is *flat* in
  d (sd 0.39 -> 0.94 over the same range), the Monte Carlo metric keeps its
  precision, and the whole class of "the metric did it" objections goes away.
- **No small-parameter rescaling of the solution.** The heat family needed
  ``alpha_d = alpha_1/d`` to stop the solution decaying to nothing. Here the
  same 1/d scaling is applied to ``nu`` for the same reason (it holds the
  constant part of the value function O(1) in d), but the *x*-dependent part of
  the solution, ``p_i(t)``, is independent of d exactly -- the Riccati system
  decouples -- so each coordinate's structure is literally the same function at
  every d.

And one difference that is *not* claimed. The nonlinearity is removable: under
the Cole-Hopf substitution ``v = exp(-lambda u / nu)`` the equation becomes
``v_t + nu Laplacian v = (lambda/nu) (sum_i q_i x_i^2) v``, a *linear* parabolic
equation with a quadratic potential. So this is a test against a different
target class, a different time direction and different boundary data, not
against an essentially nonlinear PDE. Saying otherwise would overclaim, and the
substitution is two lines to check.

The exact solution
------------------
Try ``u(x, t) = sum_i p_i(t) x_i^2 + r(t)``. Then ``u_t = sum p_i' x_i^2 + r'``,
``Laplacian u = 2 sum_i p_i``, and ``|grad u|^2 = sum_i 4 p_i^2 x_i^2``, so the
equation separates coordinate by coordinate:

    x_i^2 :   p_i' - 4 lambda p_i^2 + q_i = 0,          p_i(T) = c_i
    const :   r' + 2 nu sum_i p_i = 0,                  r(T) = 0

The first is a *scalar* Riccati equation per coordinate -- the matrix Riccati of
the general LQ problem, diagonalized by the isotropic control cost and the
diagonal state cost. With ``k_i = sqrt(q_i / (4 lambda))`` (its stable fixed
point) and ``beta_i = 4 sqrt(lambda q_i)``, substituting ``w = (p-k)/(p+k)``
turns it into ``w' = beta w``, so

    w_i(t) = w_i(T) exp(-beta_i (T - t)),   w_i(T) = (c_i - k_i)/(c_i + k_i),
    p_i(t) = k_i (1 + w_i) / (1 - w_i),

and r follows by quadrature, which is also elementary:

    integral p dt = (k/beta) [ log|w| - 2 log|1 - w| ] + const

(partial fractions in w, since ``dt = dw/(beta w)``). Both are closed form at
every d and every t -- no truncation, no reference grid --  and
``tests/test_highd_hjb.py`` checks the closed-form r against Gauss-Legendre
quadrature of p, and the whole thing against the PDE by autograd in float64.

Measuring the error when the solution has a large mean
------------------------------------------------------
``u`` here has a big d-dependent *average*: the constant part r(t) plus the mean
of the quadratic form. A network that learned nothing but that average would
already score well on ``||e|| / ||u||``, and the flattery grows with d
(``||u||/sd(u)`` runs 1.58 at d = 1 to 3.67 at d = 16). So the headline metric
is relative to the solution's *standard deviation* over the space-time box,

    rel = ||u_theta - u||_{L2} / sd(u),

for which the best constant predictor scores exactly 1.0 at every d -- the same
convention as Sec. 11, where a network outputting zero scores exactly 1.0. Both
normalizations are reported, along with the exact ratio between them, so nothing
here depends on which one a reader prefers.

The denominators are not sampled. The spatial integrals of a quadratic form
against a uniform measure on the cube are elementary (``E[x_i^2] = 1/3``,
``E[x_i^4] = 1/5``), leaving a smooth one-dimensional integral in t that
Gauss-Legendre resolves to machine precision. Only the numerator is Monte Carlo,
and it carries its standard error the same way Sec. 11's does.

Run:  python experiments/highd_hjb.py --check    # exact solution vs the PDE
      python experiments/highd_hjb.py --metric   # MC metric precision vs d
      python experiments/highd_hjb.py --sweep    # the d-sweep, ~2.5 h
      python experiments/highd_hjb.py --figures  # replay from committed CSVs
      python experiments/highd_hjb.py --quick    # tiny end-to-end smoke run
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from common import read_csv, savefig, write_csv
from pinn import derivatives as D
from pinn.model import MLP, set_seed

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
LAMBDA = 0.5            # control-cost weight; the coefficient of |grad u|^2
NU_1 = 0.05             # diffusivity at d = 1; nu_d = NU_1 / d (see below)
T_END = 1.0
BOX = (-1.0, 1.0)       # every spatial axis spans [-1, 1]


def default_q(d):
    """State-cost weights: one distinguished axis, the rest equal.

    ``[4, 1, 1, ...]``. A constant vector would make the problem invariant
    under permuting the coordinates, and a target living in that much smaller
    family invites the objection that the network found the symmetry rather
    than the solution -- the same argument, and the same fix, as
    ``highd_heat.default_terms``.
    """
    return np.array([4.0] + [1.0] * (d - 1), dtype=float)


class HJB:
    """The LQ Hamilton-Jacobi-Bellman problem and its closed-form value function.

    Parameters
    ----------
    d : int
        Number of state dimensions.
    q : array_like, optional
        Per-coordinate state cost, length d, strictly positive. Defaults to
        :func:`default_q`.
    c : array_like or float, optional
        Terminal cost coefficients, length d. Defaults to ``0.5 * k``, i.e.
        half the Riccati fixed point, which makes ``p_i`` rise from ``k_i/2``
        at t = T to ``k_i`` as t decreases -- non-degenerate at both ends, and
        a *nonzero* terminal condition so the terminal loss has a scale to be
        normalized against. ``c = 0`` is allowed and gives ``p = k tanh``.
    lam : float
        The ``lambda`` above.
    nu : float, optional
        Diffusivity. Defaults to ``NU_1 / d``.
    t_range : (float, float)
        ``(0, T)``; the data is at the *right* end.

    Attributes
    ----------
    k : ndarray
        ``sqrt(q / (4 lambda))``, the stable fixed point of each Riccati.
    beta : ndarray
        ``4 sqrt(lambda q)``, the rate at which p approaches it.
    w_T : ndarray
        ``(c - k)/(c + k)``, the Riccati substitution's terminal value.

    Why nu is scaled by 1/d
    -----------------------
    ``r' = -2 nu sum_i p_i`` sums d terms, so at fixed nu the constant part of
    the value function grows linearly in d and eventually dwarfs the part that
    depends on x. That would make the d-sweep partly a measurement of how well a
    network can learn a large constant. Scaling ``nu = NU_1/d`` holds r O(1)
    across the sweep. It is the same choice, for the same reason, as
    ``highd_heat``'s ``alpha_d = alpha_1/d``, and at d = 1 it is the identity.
    Note that it does *not* touch ``p_i(t)``, which does not depend on d or on
    nu at all -- the per-coordinate structure of the solution is exactly the
    same function at every d, which the heat family could not manage.
    """

    def __init__(self, d, q=None, c=None, lam=LAMBDA, nu=None, t_range=(0.0, T_END)):
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self.lam = float(lam)
        if self.lam <= 0:
            raise ValueError(f"lambda must be > 0, got {lam}")
        self.nu = float(NU_1 / d) if nu is None else float(nu)
        self.t_range = (float(t_range[0]), float(t_range[1]))
        if self.t_range[1] <= self.t_range[0]:
            raise ValueError(f"empty time interval {self.t_range}")

        self.q = default_q(self.d) if q is None else np.asarray(q, dtype=float)
        if self.q.shape != (self.d,):
            raise ValueError(f"q must have shape ({self.d},), got {self.q.shape}")
        if np.any(self.q <= 0):
            raise ValueError("q entries must be strictly positive")

        self.k = np.sqrt(self.q / (4.0 * self.lam))
        self.beta = 4.0 * np.sqrt(self.lam * self.q)
        if c is None:
            self.c = 0.5 * self.k
        else:
            self.c = np.broadcast_to(np.asarray(c, dtype=float), (self.d,)).copy()
        if np.any(self.c < 0):
            # p is monotone toward +k from any c > -k; negative c is admissible
            # mathematically but corresponds to a negative terminal cost, and
            # c <= -k escapes to the unstable branch. Refuse rather than ship a
            # silently divergent problem.
            raise ValueError("terminal cost c must be >= 0")
        self.w_T = (self.c - self.k) / (self.c + self.k)

    @property
    def T(self):
        return self.t_range[1]

    def __repr__(self):
        return (f"HJB(d={self.d}, lam={self.lam:.4g}, nu={self.nu:.6g}, "
                f"q={self.q.tolist()}, c={np.round(self.c, 6).tolist()})")


# ---------------------------------------------------------------------------
# The exact solution: p(t), r(t), u(x, t)
# ---------------------------------------------------------------------------
def _w(problem, t):
    """The Riccati substitution ``w = (p-k)/(p+k)`` at times ``t``, shape (N, d)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    s = problem.T - t[:, None]
    return problem.w_T[None, :] * np.exp(-problem.beta[None, :] * s)


def p_of_t(problem, t):
    """``p_i(t)``, shape (N, d). The Riccati solution, in closed form."""
    w = _w(problem, t)
    return problem.k[None, :] * (1.0 + w) / (1.0 - w)


def dp_dt(problem, t):
    """``p_i'(t) = 4 lambda p_i^2 - q_i``, shape (N, d).

    Taken from the ODE rather than by differentiating the closed form: the two
    agree by construction and the ODE is the definition. The tests check them
    against each other by finite differences anyway, because "by construction"
    is exactly the kind of claim that survives a sign error.
    """
    p = p_of_t(problem, t)
    return 4.0 * problem.lam * p ** 2 - problem.q[None, :]


def _antiderivative(problem, t):
    """``F(t)`` with ``F' = p``, per coordinate, shape (N, d).

    With ``dt = dw/(beta w)`` and ``p = k(1+w)/(1-w)``,

        integral p dt = (k/beta) integral (1+w)/(w(1-w)) dw
                      = (k/beta) [ log|w| - 2 log|1-w| ]

    by partial fractions ``(1+w)/(w(1-w)) = 1/w + 2/(1-w)``.
    """
    w = _w(problem, t)
    return (problem.k[None, :] / problem.beta[None, :]) * (
        np.log(np.abs(w)) - 2.0 * np.log(np.abs(1.0 - w)))


def r_of_t(problem, t):
    """``r(t) = 2 nu sum_i integral_t^T p_i``, shape (N,). Zero at t = T."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    FT = _antiderivative(problem, np.full(1, problem.T))          # (1, d)
    return float(2.0 * problem.nu) * (FT - _antiderivative(problem, t)).sum(axis=1)


def dr_dt(problem, t):
    """``r'(t) = -2 nu sum_i p_i(t)``, shape (N,)."""
    return -2.0 * problem.nu * p_of_t(problem, t).sum(axis=1)


def exact(problem, X, t):
    """``u(x, t) = sum_i p_i(t) x_i^2 + r(t)``. ``X`` is (N, d), ``t`` (N,) or scalar."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != problem.d:
        raise ValueError(f"X must be (N, {problem.d}), got {X.shape}")
    t = np.broadcast_to(np.asarray(t, dtype=float), (X.shape[0],))
    return (p_of_t(problem, t) * X ** 2).sum(axis=1) + r_of_t(problem, t)


def exact_from_coords(problem, coords):
    """``exact`` on stacked coordinates ``[x_1 .. x_d, t]`` of shape (N, d+1)."""
    coords = np.asarray(coords, dtype=float)
    return exact(problem, coords[:, : problem.d], coords[:, problem.d])


def exact_target(problem, coords):
    """The exact solution at ``coords`` as a torch (N, 1) tensor.

    Used for the inhomogeneous boundary data. Evaluated in numpy from the closed
    form and handed over as float32, since it is a fixed target and never
    differentiated.
    """
    values = exact_from_coords(problem, coords.detach().numpy())
    return torch.tensor(values, dtype=coords.dtype).reshape(-1, 1)


def terminal_condition(problem, X):
    """``u(x, T) = sum_i c_i x_i^2`` as a torch (N, 1) tensor."""
    c = torch.tensor(problem.c, dtype=X.dtype)
    return (c[None, :] * X ** 2).sum(dim=1, keepdim=True)


def residual(problem, u, coords):
    """``r = u_t + nu Lap_x u - lambda |grad_x u|^2 + sum_i q_i x_i^2``.

    ``coords`` is (N, d+1) laid out as ``[x_1 .. x_d, t]``.

    One first-derivative pass supplies *all three* derivative terms here --
    ``u_t`` is its last column and ``|grad_x u|^2`` its first d, and the
    Laplacian differentiates those same columns a second time -- so the cost is
    ``d + 1`` reverse-mode passes rather than the ``d + 2`` of
    ``highd_heat.residual``, which takes a separate pass for u_t.
    ``tests/test_highd_hjb.py`` counts them.
    """
    d = problem.d
    g = D.grad(u, coords)
    u_t = g[:, d : d + 1]
    grad_sq = (g[:, :d] ** 2).sum(dim=1, keepdim=True)
    lap = torch.zeros_like(u)
    for i in range(d):
        lap = lap + D.partial(g[:, i : i + 1], coords, i)
    q = torch.tensor(problem.q, dtype=coords.dtype)
    source = (q[None, :] * coords[:, :d] ** 2).sum(dim=1, keepdim=True)
    return u_t + problem.nu * lap - problem.lam * grad_sq + source


# ---------------------------------------------------------------------------
# Exact norms: every denominator in closed form (x) + quadrature (t)
# ---------------------------------------------------------------------------
#: Gauss-Legendre nodes used for every time average. The integrands are
#: products and exponentials of smooth functions of t, so this is machine
#: precision, not an approximation the results depend on -- ``--check`` prints
#: the change between 64 and 256 nodes.
QUAD_NODES = 128


def _quad_moments(a, const):
    """First two moments of ``const + sum_i a_i x_i^2``, x uniform on [-1,1]^d.

    ``E[x_i^2] = 1/3`` and ``E[x_i^4] = 1/5``, and distinct coordinates are
    independent, so with ``s1 = sum a_i`` and ``s2 = sum a_i^2``

        E[Q]   = const + s1/3
        E[Q^2] = s2/5 + (s1^2 - s2)/9 + 2 const s1/3 + const^2

    (the cross terms ``E[x_i^2 x_j^2] = 1/9`` for i != j supply the middle
    piece). Returns ``(mean, mean_square)``. Everything else in this section is
    this function under a time integral.
    """
    a = np.asarray(a, dtype=float)
    s1 = float(a.sum())
    s2 = float((a ** 2).sum())
    mean = const + s1 / 3.0
    ex2 = s2 / 5.0 + (s1 ** 2 - s2) / 9.0
    return mean, ex2 + 2.0 * const * s1 / 3.0 + const ** 2


def _time_nodes(problem, nodes=QUAD_NODES):
    """Gauss-Legendre nodes and weights on the problem's time interval."""
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    t0, t1 = problem.t_range
    return 0.5 * (x + 1.0) * (t1 - t0) + t0, 0.5 * w    # weights sum to 1


def space_time_moments(problem, nodes=QUAD_NODES):
    """Exact mean, mean square, variance and x-variance of u over the box.

    Returns a dict with

    - ``mean``   : ``E_{x,t}[u]``
    - ``ms``     : ``E_{x,t}[u^2]``           (the raw metric's denominator^2)
    - ``var``    : ``ms - mean^2``            (the headline metric's, and the
                   squared error of the best *constant* predictor)
    - ``var_x``  : ``E_t[Var_x(u | t)]``      (the squared error of the best
                   *time-profile* predictor, i.e. of a network that learned the
                   exactly-right x-average at every t and nothing else)

    ``var_x = (4/45) E_t[sum_i p_i^2]`` because ``Var(x_i^2) = 1/5 - 1/9``, and
    it sits below ``var`` by exactly the variance of the mean profile in t. The
    gap between the two says how much of the solution's variation is a function
    of t alone -- which is the part a network gets nearly for free.
    """
    ts, ws = _time_nodes(problem, nodes)
    P = p_of_t(problem, ts)                     # (N, d)
    R = r_of_t(problem, ts)                     # (N,)
    mean = ms = var_x = 0.0
    for p, r, w in zip(P, R, ws):
        m, s = _quad_moments(p, float(r))
        mean += w * m
        ms += w * s
        var_x += w * (4.0 / 45.0) * float((p ** 2).sum())
    return {"mean": float(mean), "ms": float(ms),
            "var": float(ms - mean ** 2), "var_x": float(var_x)}


def exact_rms(problem, nodes=QUAD_NODES):
    """``sqrt(E[u^2])`` over the space-time box."""
    return float(np.sqrt(space_time_moments(problem, nodes)["ms"]))


def exact_sd(problem, nodes=QUAD_NODES):
    """``sd(u)`` over the space-time box: the headline metric's denominator."""
    return float(np.sqrt(space_time_moments(problem, nodes)["var"]))


def terminal_ms(problem):
    """``E_x[u(x, T)^2]``, the energy the terminal loss is measured against."""
    return float(_quad_moments(problem.c, 0.0)[1])


def boundary_ms(problem, nodes=QUAD_NODES):
    """``E[u^2]`` over the uniform measure on the 2d faces, times t.

    On the face ``x_j = +-1`` the solution is ``p_j(t) + sum_{i != j} p_i x_i^2
    + r(t)``: the same quadratic form with coordinate j deleted and its
    coefficient moved into the constant. Both sides of an axis give the same
    value (x_j enters squared), so the average over the 2d faces is the average
    over the d axes.
    """
    ts, ws = _time_nodes(problem, nodes)
    P = p_of_t(problem, ts)
    R = r_of_t(problem, ts)
    total = 0.0
    for p, r, w in zip(P, R, ws):
        acc = 0.0
        for j in range(problem.d):
            a = np.delete(p, j)
            acc += _quad_moments(a, float(r) + float(p[j]))[1]
        total += w * acc / problem.d
    return float(total)


def residual_scale(problem, nodes=QUAD_NODES):
    """``E_{x,t}[u_t^2]`` for the *exact* solution: the residual's own scale.

    The residual is a sum of four terms that cancel exactly on the truth, so
    "the residual is small" only means something relative to how big those terms
    are. ``u_t = sum_i p_i'(t) x_i^2 + r'(t)`` is one of them and is the same
    choice Sec. 11 makes for the heat sweep, which keeps the two normalizations
    comparable.
    """
    ts, ws = _time_nodes(problem, nodes)
    dP = dp_dt(problem, ts)
    dR = dr_dt(problem, ts)
    total = 0.0
    for dp, dr, w in zip(dP, dR, ws):
        total += w * _quad_moments(dp, float(dr))[1]
    return float(total)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _uniform_box(n, d, gen):
    lo, hi = BOX
    return lo + (hi - lo) * torch.rand(n, d, generator=gen)


def interior_points(problem, n, gen):
    """(n, d+1) uniform points in the space-time interior, grad enabled."""
    t0, t1 = problem.t_range
    x = _uniform_box(n, problem.d, gen)
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    coords = torch.cat([x, t], dim=1)
    coords.requires_grad_(True)
    return coords


def terminal_points(problem, n, gen):
    """(n, d+1) points on the *terminal* slice t = T, x uniform on the cube."""
    x = _uniform_box(n, problem.d, gen)
    t = torch.full((n, 1), float(problem.T))
    return torch.cat([x, t], dim=1)


def boundary_points(problem, n, gen):
    """(n, d+1) points uniform on the union of the 2d faces.

    Pick an axis and a side uniformly, pin that coordinate to -1 or +1, leave
    the rest uniform -- the uniform measure on the boundary, since all faces
    have equal area. Same construction as ``highd_heat.boundary_points`` and for
    the same reason (a per-face budget would grow the boundary cost linearly in
    d), but here the target is not zero and comes back with the points.
    """
    t0, t1 = problem.t_range
    d = problem.d
    lo, hi = BOX
    x = _uniform_box(n, d, gen)
    axis = torch.randint(0, d, (n,), generator=gen)
    side = torch.randint(0, 2, (n,), generator=gen).to(x.dtype)
    x[torch.arange(n), axis] = lo + (hi - lo) * side
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    return torch.cat([x, t], dim=1)


def uniform_box_points(problem, n, rng):
    """(n, d+1) numpy points uniform on the space-time box, for evaluation.

    On a numpy Generator, so the evaluation sample is independent of the
    training sample by construction rather than by seed hygiene.
    """
    t0, t1 = problem.t_range
    lo, hi = BOX
    x = lo + (hi - lo) * rng.random((n, problem.d))
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


def rel_l2_mc(model, problem, n=200_000, seed=0, moments=None, return_parts=False):
    """Relative L2 error by Monte Carlo, both normalizations, with std errors.

    Returns ``(rel_sd, se_sd, rel_rms, se_rms)``: the error divided by ``sd(u)``
    (the headline) and by ``rms(u)`` (the flattering one), each with the
    standard error of that estimate. The denominators are exact
    (:func:`space_time_moments`), so all the noise is in the numerator; the
    square root's standard error follows by the delta method,
    ``se(sqrt(m)) = se(m)/(2 sqrt(m))``, exactly as in ``highd_heat.rel_l2_mc``.

    ``moments`` may be passed to avoid recomputing the quadrature in a loop.
    """
    mom = space_time_moments(problem) if moments is None else moments
    rng = np.random.default_rng(seed)
    coords = uniform_box_points(problem, n, rng)
    e = (predict(model, coords) - exact_from_coords(problem, coords)) ** 2

    m = float(e.mean())
    se_m = float(e.std(ddof=1) / np.sqrt(n))
    num = np.sqrt(m)
    se_num = se_m / (2.0 * num) if m > 0 else 0.0
    sd, rms = np.sqrt(mom["var"]), np.sqrt(mom["ms"])
    out = (float(num / sd), float(se_num / sd), float(num / rms), float(se_num / rms))
    return out + (e,) if return_parts else out


def baselines(problem, moments=None):
    """What "no work" scores under the headline metric, at this d.

    Three reference predictors, all scored as ``||u_pred - u|| / sd(u)``:

    - ``zero``     : u = 0. Scores ``rms/sd``, which is *above* 1 and grows with
      d -- unlike the heat problem, where zero is the best constant.
    - ``constant`` : the best constant, ``E[u]``. Scores exactly 1.0 by
      construction; this is what fixes the metric's meaning across d.
    - ``profile``  : the exactly-known time profile ``E_x[u | t]``, i.e. a
      network that learned every bit of the t dependence and nothing about x.
      Scores ``sqrt(var_x / var)``, and how far *below* 1 that sits is how much
      of the solution is a function of t alone.
    """
    mom = space_time_moments(problem) if moments is None else moments
    sd = np.sqrt(mom["var"])
    return {"zero": float(np.sqrt(mom["ms"]) / sd),
            "constant": 1.0,
            "profile": float(np.sqrt(mom["var_x"] / mom["var"]))}


def mc_relative_sd(problem, n, seed=0):
    """Relative sd of a uniform-MC estimate of ``E[u^2]``, with no network in it.

    The calibrated probe of the metric: draw n points, form
    ``mean_i u(x_i,t_i)^2``, and compare both its sampling spread and its value
    against :func:`space_time_moments`, which knows the answer exactly. Sec. 11
    needed this because the heat problem's integrand is log-normally spread in
    high d (relative standard error ``sqrt(((3/2)^d - 1)/n)``, 8.7% at d = 16).
    Here it is run for the same reason and the answer is different, which is one
    of this section's points rather than an aside.

    Returns ``(rel_sd, ratio_to_exact, top1pct_share)``.
    """
    rng = np.random.default_rng(seed)
    coords = uniform_box_points(problem, n, rng)
    u2 = exact_from_coords(problem, coords) ** 2
    m = float(u2.mean())
    rel_sd = float(u2.std(ddof=1) / np.sqrt(n) / m) if m > 0 else float("nan")
    v = np.sort(u2)[::-1]
    top = int(max(1, round(0.01 * v.size)))
    share = float(v[:top].sum() / v.sum())
    return rel_sd, m / space_time_moments(problem)["ms"], share


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def model_config(problem, width=128, depth=4):
    """Constructor kwargs for the field. Input dimension is d + 1."""
    return dict(in_dim=problem.d + 1, out_dim=1, width=width, depth=depth,
                activation="tanh")


def _objective(problem, model, interior, tc, tc_target, bc, bc_target, w_tc, w_bc):
    """The training objective at the model's current parameters, as a float."""
    r = residual(problem, model(interior), interior)
    total = (torch.mean(r ** 2)
             + w_tc * torch.mean((model(tc) - tc_target) ** 2)
             + w_bc * torch.mean((model(bc) - bc_target) ** 2))
    return float(total.detach())


def train(
    problem,
    n_interior=4000,
    width=128,
    depth=4,
    steps=5000,
    lr=1e-3,
    n_tc=400,
    n_bc=400,
    seed=0,
    w_tc=1.0,
    w_bc=1.0,
    eval_every=500,
    eval_n=50_000,
    select="best_loss",
    verbose=False,
    ckpt_path=None,
    ckpt_every=250,
    deadline=None,
):
    """Train an HJB PINN with Adam; return ``(model, history, best)``.

    Deliberately the same recipe as ``highd_heat.train`` -- fixed collocation
    set, soft data and boundary penalties, Adam, and selection on the lowest
    *training* loss (the loss contains no ground truth, so nothing about the
    exact solution enters the choice of iterate; see that function's docstring
    for the evidence that the final iterate is a poor choice on this problem
    class). It is written out here rather than shared because four things
    genuinely differ: the box is ``[-1,1]^d``, the data slice is at ``t = T``
    rather than ``t = 0``, the boundary target is the exact solution rather than
    zero, and the residual is nonlinear.

    History rows are ``(step, loss, loss_r, loss_tc, loss_bc, rel_sd, se_sd,
    rel_rms, train_seconds)``. ``train_seconds`` is cumulative optimization wall
    clock with the evaluation calls excluded, as in Sec. 11.

    Interruptible runs
    ------------------
    ``ckpt_path`` and ``deadline`` (an absolute ``time.monotonic()`` value) make
    a cell resumable: every ``ckpt_every`` steps the model, the optimizer state,
    the history and the best-so-far are written, and if the deadline has passed
    the call returns early with ``best["completed"] = False``. Calling ``train``
    again with the same arguments picks up exactly where it stopped.

    This is not a convenience. The d = 16 cells of the sweep are ~23 minutes
    each and the environment this repo is developed in cannot hold a foreground
    process longer than ten, so without resumption the sweep is not runnable
    here at all. It is exact rather than approximate because there is no
    per-step randomness -- the collocation set is drawn once from ``seed`` and
    then fixed -- so a resumed run and an uninterrupted one visit the same
    parameters at the same steps. ``test_highd_hjb.py`` checks that by running
    both and comparing the histories entry for entry.
    """
    if select not in ("best_loss", "final"):
        raise ValueError(f"select must be 'best_loss' or 'final', got {select!r}")
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    interior = interior_points(problem, n_interior, gen)
    tc = terminal_points(problem, n_tc, gen)
    tc_target = terminal_condition(problem, tc[:, : problem.d])
    bc = boundary_points(problem, n_bc, gen)
    bc_target = exact_target(problem, bc)

    model = MLP(**model_config(problem, width, depth))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    moments = space_time_moments(problem)

    history = []
    best = dict(step=-1, loss=float("inf"))
    best_state = None
    train_seconds = 0.0
    start = 0

    if ckpt_path is not None and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        history = [tuple(row) for row in blob["history"]]
        best = dict(blob["best"])
        best_state = blob["best_state"]
        train_seconds = float(blob["train_seconds"])
        start = int(blob["step"]) + 1
        if verbose:
            print(f"  resumed from {ckpt_path} at step {start}", flush=True)

    def _save(step):
        if ckpt_path is None:
            return
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "history": history, "best": best, "best_state": best_state,
                    "train_seconds": train_seconds, "step": step}, ckpt_path)

    for step in range(start, steps + 1):
        t_step = time.perf_counter()
        opt.zero_grad()
        r = residual(problem, model(interior), interior)
        loss_r = torch.mean(r ** 2)
        loss_tc = torch.mean((model(tc) - tc_target) ** 2)
        loss_bc = torch.mean((model(bc) - bc_target) ** 2)
        loss = loss_r + w_tc * loss_tc + w_bc * loss_bc
        loss.backward()

        # Snapshot before the update: ``loss`` is the objective at the current
        # parameters and ``opt.step()`` is about to replace them. Same off-by-one
        # trap, and the same fix, as ``highd_heat.train``.
        value = float(loss.detach().item())
        if value < best["loss"]:
            best = dict(step=step, loss=value)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        opt.step()
        train_seconds += time.perf_counter() - t_step

        if step % eval_every == 0 or step == steps:
            rel_sd, se_sd, rel_rms, _ = rel_l2_mc(model, problem, n=eval_n,
                                                  seed=12345, moments=moments)
            history.append((step, value, float(loss_r.item()),
                            float(loss_tc.item()), float(loss_bc.item()),
                            rel_sd, se_sd, rel_rms, train_seconds))
            if verbose:
                print(f"  step {step:5d}  loss {value:.3e}  "
                      f"(r {loss_r.item():.2e} tc {loss_tc.item():.2e} "
                      f"bc {loss_bc.item():.2e})  rel/sd {rel_sd:.4f} +- {se_sd:.4f}",
                      flush=True)

        if (ckpt_path is not None and step > start
                and step % ckpt_every == 0 and step < steps):
            _save(step)
            if deadline is not None and time.monotonic() >= deadline:
                best = dict(best)
                best["completed"] = False
                best["stopped_at"] = step
                return model, history, best

    # The loop only ever scores the loss *before* each update, so the parameters
    # the last update produced have never competed. Score them here, with one
    # extra forward pass and no step.
    final_loss = _objective(problem, model, interior, tc, tc_target, bc, bc_target,
                            w_tc, w_bc)
    if final_loss < best["loss"]:
        best = dict(step=steps + 1, loss=final_loss)
        best_state = None            # the final weights are already in the model
    best["final_loss"] = final_loss
    best["train_seconds"] = train_seconds
    best["state_dict"] = best_state
    best["completed"] = True

    if select == "best_loss" and best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
SWEEP_DIMS = (1, 2, 4, 8, 16)
SEEDS = (0, 1, 2)

#: The fixed budget. Deliberately identical to Sec. 11's ``highd_pinn.BUDGET``,
#: because the comparison between the two problems is the reason this section
#: exists and a different budget would confound it.
BUDGET = dict(n_interior=4000, n_tc=400, n_bc=400,
              width=128, depth=4, steps=5000, lr=1e-3)

EVAL_EVERY = 250
EVAL_N = 100_000
SCORE_N = 1_000_000
SCORE_SEED = 7

SWEEP_CSV = "highd_hjb_sweep.csv"
TRACE_CSV = "highd_hjb_trace.csv"
METRIC_CSV = "highd_hjb_metric.csv"
CHECK_CSV = "highd_hjb_check.csv"

SWEEP_FIELDS = ["d", "seed", "params", "rel_sd", "stderr_sd", "rel_rms",
                "rel_sd_final", "stderr_sd_final", "best_step", "best_loss",
                "final_loss", "loss_r", "loss_tc", "loss_bc", "exact_sd",
                "exact_rms", "base_zero", "base_profile",
                "train_seconds", "wall_seconds", "ms_per_step"]
TRACE_FIELDS = ["d", "seed", "step", "loss", "loss_r", "loss_tc", "loss_bc",
                "rel_sd", "stderr_sd", "rel_rms", "train_seconds"]

#: Per-cell resume state. Not committed -- it is scratch, and a 5000-step d = 16
#: checkpoint is 600 KB. The *results* are committed, in ``logs/``.
CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hjb_cache"))


def n_params(d, width, depth):
    """Parameter count of the field network at state dimension ``d``.

    ``width(d+1) + width`` for the input map, ``depth-1`` maps of
    ``width^2 + width``, and ``width + 1`` out. Same formula as
    ``highd_pinn.n_params``; checked against a real model in the tests.
    """
    return int(width * (d + 1) + width
               + (depth - 1) * (width * width + width)
               + width + 1)


def run_cell(d, seed, budget=None, eval_every=EVAL_EVERY, eval_n=EVAL_N,
             score_n=SCORE_N, verbose=True, resumable=True, deadline=None):
    """Train one (d, seed) cell at the fixed budget and score it.

    Returns ``(row, history)``, or ``(None, None)`` if a ``deadline`` cut the
    cell short -- in which case the checkpoint holds the partial run and the
    next call finishes it.

    Trained once, scored twice: ``select="final"`` leaves the last iterate in the
    model and hands the selected parameters back in ``best["state_dict"]``, so
    both scores come off the same run on the same evaluation sample.
    """
    budget = dict(BUDGET if budget is None else budget)
    problem = HJB(d)
    moments = space_time_moments(problem)
    base = baselines(problem, moments)
    if verbose:
        print(f"  d={d:2d} seed={seed}: {problem}", flush=True)

    ckpt = os.path.join(CACHE, f"cell_d{d}_s{seed}.pt") if resumable else None
    t0 = time.time()
    model, history, best = train(problem, seed=seed, eval_every=eval_every,
                                 eval_n=eval_n, select="final", verbose=verbose,
                                 ckpt_path=ckpt, deadline=deadline, **budget)
    wall = time.time() - t0
    if not best.get("completed", True):
        if verbose:
            print(f"       paused at step {best['stopped_at']} "
                  f"({wall:.0f}s this call)", flush=True)
        return None, None

    rel_f, se_f, _, _ = rel_l2_mc(model, problem, n=score_n, seed=SCORE_SEED,
                                  moments=moments)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    rel, se, rel_rms, _ = rel_l2_mc(model, problem, n=score_n, seed=SCORE_SEED,
                                    moments=moments)

    _, loss, loss_r, loss_tc, loss_bc, _, _, _, train_s = history[-1]
    row = {
        "d": d, "seed": seed,
        "params": n_params(d, budget["width"], budget["depth"]),
        "rel_sd": f"{rel:.6e}", "stderr_sd": f"{se:.6e}", "rel_rms": f"{rel_rms:.6e}",
        "rel_sd_final": f"{rel_f:.6e}", "stderr_sd_final": f"{se_f:.6e}",
        "best_step": best["step"], "best_loss": f"{best['loss']:.6e}",
        "final_loss": f"{best['final_loss']:.6e}",
        "loss_r": f"{loss_r:.6e}", "loss_tc": f"{loss_tc:.6e}",
        "loss_bc": f"{loss_bc:.6e}",
        "exact_sd": f"{np.sqrt(moments['var']):.6e}",
        "exact_rms": f"{np.sqrt(moments['ms']):.6e}",
        "base_zero": f"{base['zero']:.6f}", "base_profile": f"{base['profile']:.6f}",
        "train_seconds": f"{train_s:.2f}", "wall_seconds": f"{wall:.2f}",
        "ms_per_step": f"{1000 * train_s / (budget['steps'] + 1):.3f}",
    }
    if verbose:
        print(f"       rel/sd {rel:.4e} +- {se:.1e}  (final iterate {rel_f:.4e}, "
              f"best-constant baseline 1.0)   {train_s:.0f}s train", flush=True)
    if ckpt is not None and os.path.exists(ckpt):
        os.remove(ckpt)
    return row, history


def _load_partial():
    """Rows and traces already committed to ``logs/``, or empty lists."""
    try:
        rows = read_csv(SWEEP_CSV)
    except FileNotFoundError:
        return [], []
    try:
        traces = read_csv(TRACE_CSV)
    except FileNotFoundError:
        traces = []
    return rows, traces


def sweep(dims=SWEEP_DIMS, seeds=SEEDS, budget=None, eval_every=EVAL_EVERY,
          eval_n=EVAL_N, score_n=SCORE_N, write=True, verbose=True,
          seconds=None, resume=True):
    """Every (d, seed) cell, written incrementally and resumable twice over.

    Two independent resume mechanisms, because they cover different failures:

    - **Between cells**, a completed cell's row is in ``logs/`` and is skipped
      on a later call. That is ``highd_pinn.sweep``'s behaviour, made explicit.
    - **Within a cell**, ``seconds`` sets a wall-clock budget for this call; when
      it expires the current cell checkpoints and the call returns. The next
      call resumes it mid-training. A single d = 16 cell is ~23 minutes, which
      is longer than the maximum foreground command this repo is developed
      under, so this is what makes the sweep runnable at all here.

    Returns ``(rows, traces, complete)``.
    """
    rows, traces = _load_partial() if resume else ([], [])
    done = {(int(r["d"]), int(r["seed"])) for r in rows}
    deadline = None if seconds is None else time.monotonic() + float(seconds)

    for d in dims:
        for seed in seeds:
            if (d, seed) in done:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return rows, traces, False
            row, history = run_cell(d, seed, budget=budget, eval_every=eval_every,
                                    eval_n=eval_n, score_n=score_n,
                                    verbose=verbose, deadline=deadline)
            if row is None:
                return rows, traces, False
            rows.append(row)
            for (step, loss, lr_, lt, lb, rsd, ssd, rrms, ts) in history:
                traces.append({"d": d, "seed": seed, "step": step,
                               "loss": f"{loss:.6e}", "loss_r": f"{lr_:.6e}",
                               "loss_tc": f"{lt:.6e}", "loss_bc": f"{lb:.6e}",
                               "rel_sd": f"{rsd:.6e}", "stderr_sd": f"{ssd:.6e}",
                               "rel_rms": f"{rrms:.6e}",
                               "train_seconds": f"{ts:.4f}"})
            if write:
                write_csv(SWEEP_CSV, SWEEP_FIELDS, rows)
                write_csv(TRACE_CSV, TRACE_FIELDS, traces)
    return rows, traces, True


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(rows):
    """Per-d mean, sd, min and max of the headline relative error."""
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = [r for r in rows if int(r["d"]) == d]
        errs = np.array([float(r["rel_sd"]) for r in cells])
        ses = np.array([float(r["stderr_sd"]) for r in cells])
        finals = np.array([float(r["rel_sd_final"]) for r in cells])
        secs = np.array([float(r["train_seconds"]) for r in cells])
        per_step = np.array([float(r["ms_per_step"]) for r in cells])
        out.append({
            "d": d, "n_seeds": len(cells), "params": cells[0]["params"],
            "mean": f"{errs.mean():.6e}",
            "median": f"{np.median(errs):.6e}",
            "sd": f"{errs.std(ddof=1):.6e}" if len(errs) > 1 else "",
            "min": f"{errs.min():.6e}", "max": f"{errs.max():.6e}",
            "spread": f"{errs.max() / errs.min():.3f}",
            "mean_stderr": f"{ses.mean():.6e}",
            "mean_final": f"{finals.mean():.6e}",
            "mean_rel_rms": f"{np.mean([float(r['rel_rms']) for r in cells]):.6e}",
            "base_zero": cells[0]["base_zero"],
            "base_profile": cells[0]["base_profile"],
            "mean_train_seconds": f"{secs.mean():.2f}",
            "ms_per_step": f"{per_step.mean():.3f}",
        })
    return out


def loss_orders_error(rows):
    """Does a lower training loss mean a lower error? Per d, from the cells.

    Selection is on training loss, so this is the assumption the selection rests
    on -- and Sec. 11-12 found it failing on the heat problem at d = 8, where
    the loss kept falling while the error stopped moving. Three seeds cannot
    support a rank correlation, so what is reported is the raw pairing: the
    seed with the lowest final loss, the seed with the lowest error, and whether
    they are the same seed. When they are not, the two are printed side by side
    rather than summarized away.
    """
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = sorted((r for r in rows if int(r["d"]) == d),
                       key=lambda r: float(r["best_loss"]))
        errs = [float(r["rel_sd"]) for r in cells]
        losses = [float(r["best_loss"]) for r in cells]
        # concordant pairs among all C(n,2): 1.0 means loss ranks error perfectly
        pairs = [(i, j) for i in range(len(cells)) for j in range(i + 1, len(cells))]
        agree = sum(1 for i, j in pairs if errs[i] <= errs[j]) / len(pairs)
        out.append({
            "d": d,
            "best_loss_seed": cells[0]["seed"],
            "best_loss": f"{losses[0]:.6e}",
            "its_error": f"{errs[0]:.6e}",
            "lowest_error_seed": cells[int(np.argmin(errs))]["seed"],
            "lowest_error": f"{min(errs):.6e}",
            "concordant": f"{agree:.3f}",
        })
    return out


def normalized_losses(rows):
    """Per-d relative residual, terminal and boundary errors, from the rows.

    Sec. 11's lesson, applied before it can bite: raw losses are not comparable
    across d, because the thing each one is matching has its own d-dependent
    size. Each loss is divided by the exact energy of its target --
    :func:`residual_scale` for the residual, :func:`terminal_ms` for the
    terminal condition, :func:`boundary_ms` for the faces -- and the square root
    of the ratio is a relative error, directly comparable across d and against
    the relative L2 itself.

    The confound is smaller here than in the heat sweep (whose target shrank
    170x over the range) but it is not zero: the boundary energy alone grows
    from 0.75 at d = 1 to 4.4 at d = 16, so an unnormalized boundary loss would
    read as improving when it is not.
    """
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        cells = [r for r in rows if int(r["d"]) == d]
        problem = HJB(d)
        r_scale = residual_scale(problem)
        tc_scale = terminal_ms(problem)
        bc_scale = boundary_ms(problem)
        lr = np.mean([float(r["loss_r"]) for r in cells])
        lt = np.mean([float(r["loss_tc"]) for r in cells])
        lb = np.mean([float(r["loss_bc"]) for r in cells])
        out.append({
            "d": d,
            "rel_residual": f"{np.sqrt(lr / r_scale):.6f}",
            "rel_terminal": f"{np.sqrt(lt / tc_scale):.6f}",
            "rel_boundary": f"{np.sqrt(lb / bc_scale):.6f}",
            "residual_scale": f"{r_scale:.6e}",
            "terminal_ms": f"{tc_scale:.6e}",
            "boundary_ms": f"{bc_scale:.6e}",
            "loss_r": f"{lr:.6e}", "loss_tc": f"{lt:.6e}", "loss_bc": f"{lb:.6e}",
        })
    return out


# ---------------------------------------------------------------------------
# Putting the two problems on one axis
# ---------------------------------------------------------------------------
def heat_moments(d, nodes=QUAD_NODES):
    """Mean, mean square and variance of ``highd_heat``'s solution over its box.

    Needed to put Sec. 11's committed numbers on this section's metric without
    retraining anything. Sec. 11 reports ``||e|| / ||u||``; the conversion to
    ``||e|| / sd(u)`` is multiplication by ``||u||/sd(u)``, and that factor is
    closed form for the heat family too:

        E_x[phi_k]   = prod_i (1 - cos(k_i pi)) / (k_i pi),   (0 for even k_i)
        E_x[u | t]   = sum_m a_m E_x[phi_m] exp(-rate_m t)
        E_x[u^2 | t] = 2^-d sum_m a_m^2 exp(-2 rate_m t)      (orthogonality)

    with the t average by the same Gauss-Legendre rule. The factor is *not*
    near 1: the heat solution's spatial mean is a large fraction of its rms at
    low d and a small one at high d (``(2 sqrt 2 / pi)^d`` falls like 0.9^d), so
    switching metrics moves the low-d end of Sec. 11's curve much more than the
    high-d end. That is worth knowing before comparing the two sweeps, and it is
    why both conventions are reported here.
    """
    from highd_heat import HighDHeat

    problem = HighDHeat(d)
    x, w = np.polynomial.legendre.leggauss(int(nodes))
    t0, t1 = problem.t_range
    ts = 0.5 * (x + 1.0) * (t1 - t0) + t0
    ws = 0.5 * w

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = (1.0 - np.cos(problem.modes * np.pi)) / (problem.modes * np.pi)
    mode_means = factors.prod(axis=1)                     # (M,)
    vol = 2.0 ** (-problem.d)

    mean = ms = 0.0
    for t, wt in zip(ts, ws):
        decay = np.exp(-problem.rates * t)
        mean += wt * float((problem.amps * mode_means * decay).sum())
        ms += wt * float(vol * (problem.amps ** 2 * decay ** 2).sum())
    return {"mean": float(mean), "ms": float(ms), "var": float(ms - mean ** 2)}


def heat_rms_over_sd(d):
    """``||u|| / sd(u)`` for the heat problem at dimension d."""
    m = heat_moments(d)
    return float(np.sqrt(m["ms"] / m["var"]))


def heat_comparison(rows):
    """Side by side with Sec. 11's heat sweep, on both metric conventions.

    Reads ``logs/highd_pinn_sweep.csv`` -- the committed result of Sec. 11, at
    the same architecture, the same optimizer, the same step count, the same
    collocation budget and the same three seeds -- and converts its errors onto
    this section's ``/sd`` convention exactly, via :func:`heat_rms_over_sd`.
    Returns an empty list if that log is not present.
    """
    try:
        heat = read_csv("highd_pinn_sweep.csv")
    except FileNotFoundError:
        return []
    out = []
    for d in sorted({int(r["d"]) for r in rows}):
        hjb_cells = [r for r in rows if int(r["d"]) == d]
        heat_cells = [r for r in heat if int(r["d"]) == d]
        if not heat_cells:
            continue
        f = heat_rms_over_sd(d)
        h_raw = np.array([float(r["rel_l2"]) for r in heat_cells])
        j_raw = np.array([float(r["rel_rms"]) for r in hjb_cells])
        j_sd = np.array([float(r["rel_sd"]) for r in hjb_cells])
        out.append({
            "d": d,
            "heat_rel_rms": f"{h_raw.mean():.6e}",
            "heat_rel_sd": f"{(h_raw * f).mean():.6e}",
            "heat_rms_over_sd": f"{f:.4f}",
            "hjb_rel_rms": f"{j_raw.mean():.6e}",
            "hjb_rel_sd": f"{j_sd.mean():.6e}",
            "hjb_rms_over_sd": f"{float(hjb_cells[0]['exact_rms']) / float(hjb_cells[0]['exact_sd']):.4f}",
            "ratio_sd": f"{(h_raw * f).mean() / j_sd.mean():.3f}",
        })
    return out


# ---------------------------------------------------------------------------
# Verification of the problem itself
# ---------------------------------------------------------------------------
def check(dims=(1, 2, 3, 5, 8), n=64, verbose=True):
    """Everything about the exact solution that can be checked, checked.

    Five independent things, none of which shares code with the others:

    1. **The PDE.** Build u from the closed form in float64 torch, take
       ``u_t``, ``Laplacian u`` and ``|grad u|^2`` by autograd, and evaluate the
       residual. Machine zero or the solution is wrong.
    2. **The terminal condition**, ``u(x, T) = sum c_i x_i^2``, elementwise.
    3. **The Riccati quadrature.** ``r(t)`` in closed form against
       Gauss-Legendre quadrature of ``p``, which is a different calculation
       entirely (the closed form came from partial fractions in w).
    4. **The Riccati ODE.** ``p'`` from the ODE ``4 lambda p^2 - q`` against
       central differences of the closed-form ``p``.
    5. **Cole-Hopf.** ``v = exp(-lambda u / nu)`` must satisfy the *linear*
       equation ``v_t + nu Lap v - (lambda/nu) Q v = 0``. This is the claim in
       the module docstring that the nonlinearity is removable, and it is
       checked rather than asserted.

    Plus the quadrature's own convergence (64 vs 256 nodes) and the exact
    moments against a large Monte Carlo sample.
    """
    rows = []
    for d in dims:
        problem = HJB(d)
        rng = np.random.default_rng(11 + d)
        lo, hi = BOX
        Xn = lo + (hi - lo) * rng.random((n, d))
        tn = rng.random(n) * problem.T

        # ``coords`` is the leaf: the residual differentiates with respect to
        # it, so u has to be built from *its* columns rather than from two
        # separate tensors that happen to be concatenated into it.
        coords = torch.tensor(np.concatenate([Xn, tn[:, None]], axis=1),
                              dtype=torch.float64, requires_grad=True)
        X = coords[:, :d]
        t = coords[:, d : d + 1]

        # u from the closed form, rebuilt as differentiable functions of t so
        # autograd can supply u_t as well as the spatial derivatives
        k = torch.tensor(problem.k, dtype=torch.float64)
        beta = torch.tensor(problem.beta, dtype=torch.float64)
        wT = torch.tensor(problem.w_T, dtype=torch.float64)

        def p_torch(tt):
            w = wT[None, :] * torch.exp(-beta[None, :] * (problem.T - tt))
            return k[None, :] * (1 + w) / (1 - w)

        def F_torch(tt):
            w = wT[None, :] * torch.exp(-beta[None, :] * (problem.T - tt))
            return (k[None, :] / beta[None, :]) * (torch.log(torch.abs(w))
                                                   - 2 * torch.log(torch.abs(1 - w)))

        FT = F_torch(torch.full((1, 1), float(problem.T), dtype=torch.float64))
        u = ((p_torch(t) * X ** 2).sum(dim=1, keepdim=True)
             + 2 * problem.nu * (FT - F_torch(t)).sum(dim=1, keepdim=True))

        res = residual(problem, u, coords)
        pde = float(res.detach().abs().max())

        # the closed-form numpy path must agree with the torch one
        agree = float(np.abs(u.detach().numpy().ravel()
                             - exact(problem, Xn, tn)).max())

        # terminal condition
        term = float(np.abs(exact(problem, Xn, np.full(n, problem.T))
                            - (problem.c[None, :] * Xn ** 2).sum(axis=1)).max())

        # r(t) against Gauss-Legendre quadrature of p -- a different
        # calculation from the partial-fraction antiderivative it is checking
        gx, gw = np.polynomial.legendre.leggauss(200)
        worst_r = 0.0
        for tq in (0.0, 0.31, 0.77):
            nodes = 0.5 * (gx + 1.0) * (problem.T - tq) + tq
            wts = gw * 0.5 * (problem.T - tq)
            integral = float(wts @ p_of_t(problem, nodes).sum(axis=1))
            worst_r = max(worst_r,
                          abs(2 * problem.nu * integral - r_of_t(problem, tq)[0]))

        # p' against central differences
        h = 1e-6
        fd = (p_of_t(problem, tn + h) - p_of_t(problem, tn - h)) / (2 * h)
        pprime = float(np.abs(fd - dp_dt(problem, tn)).max())

        # Cole-Hopf: v = exp(-lam u / nu) solves the linear equation
        v = torch.exp(-problem.lam * u / problem.nu)
        g = D.grad(v, coords)
        lap_v = sum(D.partial(g[:, i : i + 1], coords, i) for i in range(d))
        qt = torch.tensor(problem.q, dtype=torch.float64)
        Q = (qt[None, :] * X ** 2).sum(dim=1, keepdim=True)
        ch = g[:, d : d + 1] + problem.nu * lap_v - (problem.lam / problem.nu) * Q * v
        # relative to the size of the terms being cancelled: v is exponentially
        # large/small in d, so an absolute tolerance would be meaningless
        scale = float(torch.max(torch.abs(problem.nu * lap_v)).detach())
        cole = float(ch.detach().abs().max() / scale)

        m64 = space_time_moments(problem, 64)
        m256 = space_time_moments(problem, 256)
        quad = max(abs(m64[key] / m256[key] - 1.0) for key in ("mean", "ms", "var"))

        big = 2_000_000
        rng2 = np.random.default_rng(5)
        cs = uniform_box_points(problem, big, rng2)
        uu = exact_from_coords(problem, cs)
        mc_mean = abs(uu.mean() / m256["mean"] - 1.0)
        mc_ms = abs((uu ** 2).mean() / m256["ms"] - 1.0)

        rows.append({"d": d, "pde_residual": f"{pde:.3e}",
                     "numpy_vs_torch": f"{agree:.3e}",
                     "terminal": f"{term:.3e}", "r_vs_quadrature": f"{worst_r:.3e}",
                     "dp_dt_vs_fd": f"{pprime:.3e}", "cole_hopf_rel": f"{cole:.3e}",
                     "quad_64_vs_256": f"{quad:.3e}",
                     "mc_mean_rel": f"{mc_mean:.3e}", "mc_ms_rel": f"{mc_ms:.3e}"})
        if verbose:
            print(f"d={d:2d}  PDE {pde:.2e}  term {term:.2e}  r-vs-quad {worst_r:.2e}  "
                  f"p' {pprime:.2e}  Cole-Hopf {cole:.2e}  quad {quad:.2e}  "
                  f"MC ms {mc_ms:.2e}", flush=True)
    return rows


def metric_study(dims=SWEEP_DIMS, sizes=(10_000, 100_000, 1_000_000),
                 seeds=(0, 1, 2, 3, 4)):
    """How precise is the uniform-MC L2 metric here, as a function of d and n?

    The same study ``highd_heat.metric_study`` runs, on the same grid, so the
    two are directly comparable -- and the comparison is one of this section's
    results rather than a formality. The heat target's integrand is
    log-normally spread (relative standard error ``sqrt(((3/2)^d - 1)/n)``,
    8.7% at d = 16 and n = 1e5); a quadratic form's is not.
    """
    rows = []
    for d in dims:
        problem = HJB(d)
        for n in sizes:
            rel_sds, ratios, tops = [], [], []
            for s in seeds:
                rel_sd, ratio, top = mc_relative_sd(problem, n, seed=100 * s + d)
                rel_sds.append(rel_sd)
                ratios.append(ratio)
                tops.append(top)
            rows.append({
                "d": d, "n": n,
                "exact_ms": f"{space_time_moments(problem)['ms']:.6e}",
                "pred_rel_sd": f"{np.mean(rel_sds):.4e}",
                "obs_rel_sd": f"{np.std(ratios, ddof=1):.4e}",
                "worst_ratio": f"{max(abs(r - 1) for r in ratios):.4e}",
                "top1pct_share": f"{np.mean(tops):.4f}",
            })
            print(f"d={d:3d} n={n:8d}  predicted rel sd {np.mean(rel_sds):.3e}  "
                  f"observed {np.std(ratios, ddof=1):.3e}  "
                  f"top-1% share {np.mean(tops):.3f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(rows, budget=None):
    budget = dict(BUDGET if budget is None else budget)
    summary = summarize(rows)
    print("\n" + "=" * 82)
    print(f"HJB PINN vs d at a fixed budget "
          f"(width {budget['width']}, depth {budget['depth']}, "
          f"{budget['steps']} Adam steps at lr {budget['lr']}, "
          f"{budget['n_interior']} collocation points)")
    print("=" * 82)
    print("relative L2 against sd(u); the best constant predictor scores 1.000\n")
    print(f"{'d':>3} {'seeds':>6} {'mean':>11} {'median':>11} {'min':>11} {'max':>11} "
          f"{'spread':>7} {'MC se':>10} {'zero':>7} {'profile':>8} {'ms/step':>9}")
    for s in summary:
        print(f"{s['d']:>3} {s['n_seeds']:>6} {float(s['mean']):>11.4e} "
              f"{float(s['median']):>11.4e} {float(s['min']):>11.4e} "
              f"{float(s['max']):>11.4e} "
              f"{float(s['spread']):>7.2f} {float(s['mean_stderr']):>10.2e} "
              f"{float(s['base_zero']):>7.3f} {float(s['base_profile']):>8.3f} "
              f"{float(s['ms_per_step']):>9.1f}")

    print("\nnormalized losses (sqrt(loss / exact energy of what it matches)):")
    print(f"{'d':>3} {'rel residual':>13} {'rel terminal':>13} {'rel boundary':>13} "
          f"{'|  residual scale':>18} {'terminal ms':>13} {'boundary ms':>13}")
    for r in normalized_losses(rows):
        print(f"{r['d']:>3} {float(r['rel_residual']):>13.4f} "
              f"{float(r['rel_terminal']):>13.4f} {float(r['rel_boundary']):>13.4f} "
              f"{float(r['residual_scale']):>18.4e} {float(r['terminal_ms']):>13.4e} "
              f"{float(r['boundary_ms']):>13.4e}")

    print("\nis the selection criterion ranking the runs? (loss vs error, per d)")
    print(f"{'d':>3} {'lowest-loss seed':>17} {'its loss':>11} {'its error':>11} "
          f"{'best seed':>10} {'best error':>11} {'concordant':>11}")
    for r in loss_orders_error(rows):
        print(f"{r['d']:>3} {r['best_loss_seed']:>17} {float(r['best_loss']):>11.3e} "
              f"{float(r['its_error']):>11.3e} {r['lowest_error_seed']:>10} "
              f"{float(r['lowest_error']):>11.3e} {float(r['concordant']):>11.3f}")

    comp = heat_comparison(rows)
    if comp:
        print("\nagainst Sec. 11's heat sweep, same budget, same seeds:")
        print(f"{'d':>3} {'heat /rms':>11} {'heat /sd':>11} {'HJB /rms':>11} "
              f"{'HJB /sd':>11} {'heat/HJB':>9}")
        for r in comp:
            print(f"{r['d']:>3} {float(r['heat_rel_rms']):>11.4e} "
                  f"{float(r['heat_rel_sd']):>11.4e} {float(r['hjb_rel_rms']):>11.4e} "
                  f"{float(r['hjb_rel_sd']):>11.4e} {float(r['ratio_sd']):>9.2f}")

    errs = [float(s["mean"]) for s in summary]
    if len(errs) > 1:
        print(f"\nheadline: mean relative error {errs[0]:.3e} at d={summary[0]['d']} "
              f"-> {errs[-1]:.3e} at d={summary[-1]['d']}, "
              f"a factor of {errs[-1] / errs[0]:.1f}")
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure(rows, traces, comparison=None, name="highd_hjb.png"):
    """Four panels: the sweep, the two problems, the normalized losses, cost."""
    import matplotlib.pyplot as plt

    summary = summarize(rows)
    dims = [s["d"] for s in summary]
    mean = np.array([float(s["mean"]) for s in summary])
    lo = np.array([float(s["min"]) for s in summary])
    hi = np.array([float(s["max"]) for s in summary])

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    # (a) the sweep, with the three reference predictors
    ax = axes[0, 0]
    ax.fill_between(dims, lo, hi, alpha=0.2, color="C0", lw=0)
    ax.plot(dims, mean, "o-", color="C0", label="PINN (mean of 3 seeds)")
    ax.axhline(1.0, color="0.35", ls="--", lw=1)
    ax.plot(dims, [float(s["base_zero"]) for s in summary], ":", color="C3",
            label="u = 0")
    ax.plot(dims, [float(s["base_profile"]) for s in summary], "-.", color="C2",
            label=r"exact $E_x[u\,|\,t]$")
    ax.annotate("best constant", (dims[-1], 1.0), xytext=(-58, 4),
                textcoords="offset points", fontsize=7, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("state dimension $d$")
    ax.set_ylabel(r"$\|u_\theta - u\|_2 \,/\, \mathrm{sd}(u)$")
    ax.set_title("(a) HJB at a fixed budget", loc="left")
    ax.legend(fontsize=7, loc="lower right")

    # (b) both problems on one axis
    ax = axes[0, 1]
    if comparison:
        cd = [r["d"] for r in comparison]
        ax.plot(cd, [float(r["heat_rel_sd"]) for r in comparison], "s-", color="C3",
                label="heat (Sec. 11)")
        ax.plot(cd, [float(r["hjb_rel_sd"]) for r in comparison], "o-", color="C0",
                label="HJB (this section)")
        ax.axhline(1.0, color="0.35", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(cd)
        ax.set_xticklabels([str(d) for d in cd])
    ax.set_xlabel("state dimension $d$")
    ax.set_ylabel(r"$\|u_\theta - u\|_2 \,/\, \mathrm{sd}(u)$")
    ax.set_title("(b) two PDEs, one budget, one metric", loc="left")
    ax.legend(fontsize=7, loc="lower right")

    # (c) normalized losses
    ax = axes[1, 0]
    nl = normalized_losses(rows)
    nd = [r["d"] for r in nl]
    ax.plot(nd, [float(r["rel_residual"]) for r in nl], "o-", label="residual")
    ax.plot(nd, [float(r["rel_terminal"]) for r in nl], "s-", label="terminal")
    ax.plot(nd, [float(r["rel_boundary"]) for r in nl], "^-", label="boundary")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(nd)
    ax.set_xticklabels([str(d) for d in nd])
    ax.set_xlabel("state dimension $d$")
    ax.set_ylabel(r"$\sqrt{\mathrm{loss}\,/\,\mathrm{energy}}$")
    ax.set_title("(c) each loss against its own exact energy", loc="left")
    ax.legend(fontsize=7)

    # (d) error against training seconds, seed 0 of each d
    ax = axes[1, 1]
    for i, d in enumerate(dims):
        cells = [t for t in traces if int(t["d"]) == d and int(t["seed"]) == 0]
        if not cells:
            continue
        secs = [float(t["train_seconds"]) for t in cells]
        err = [float(t["rel_sd"]) for t in cells]
        ax.plot(secs, err, "-", color=f"C{i}", lw=1.2, label=f"d = {d}")
    ax.axhline(1.0, color="0.35", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training seconds (optimization only)")
    ax.set_ylabel(r"$\|u_\theta - u\|_2 \,/\, \mathrm{sd}(u)$")
    ax.set_title("(d) cost to accuracy, seed 0", loc="left")
    ax.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, name)


def figures_from_committed():
    """Replay ``figures/highd_hjb.png`` from the committed CSVs. No training.

    The entry point ``experiments/reproduce_figures.py`` calls; the sweep it
    replays is ~2.5 hours of CPU, so the logs ship and this turns them back
    into the figure.
    """
    rows = read_csv(SWEEP_CSV)
    traces = read_csv(TRACE_CSV)
    figure(rows, traces, heat_comparison(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(quick=False, do_check=False, metric=False, do_sweep=False, figures=False,
         seconds=None):
    if quick:
        problem = HJB(2)
        mom = space_time_moments(problem)
        print(f"[quick] {problem}")
        print(f"        sd(u) {np.sqrt(mom['var']):.6f}   rms(u) {np.sqrt(mom['ms']):.6f}"
              f"   baselines {baselines(problem, mom)}")
        model, hist, best = train(problem, n_interior=800, width=32, steps=300,
                                  eval_every=100, eval_n=20_000, verbose=True,
                                  ckpt_path=None)
        print(f"        lowest training loss at step {best['step']}: {best['loss']:.3e}")
        return

    if do_check:
        rows = check()
        write_csv(CHECK_CSV,
                  ["d", "pde_residual", "numpy_vs_torch", "terminal",
                   "r_vs_quadrature", "dp_dt_vs_fd", "cole_hopf_rel",
                   "quad_64_vs_256", "mc_mean_rel", "mc_ms_rel"], rows)
        return

    if metric:
        rows = metric_study()
        write_csv(METRIC_CSV,
                  ["d", "n", "exact_ms", "pred_rel_sd", "obs_rel_sd",
                   "worst_ratio", "top1pct_share"], rows)
        return

    if do_sweep:
        rows, traces, complete = sweep(seconds=seconds)
        if not complete:
            done = len({(r["d"], r["seed"]) for r in rows})
            total = len(SWEEP_DIMS) * len(SEEDS)
            print(f"\nSWEEP INCOMPLETE: {done}/{total} cells. "
                  f"Run again to continue (state in {CACHE}).")
            return
        report(rows)
        figure(rows, traces, heat_comparison(rows))
        print("\nSWEEP COMPLETE")
        return

    if figures:
        rows = read_csv(SWEEP_CSV)
        traces = read_csv(TRACE_CSV)
        report(rows)
        figure(rows, traces, heat_comparison(rows))
        return

    print(__doc__.strip().splitlines()[0])
    print("\nNothing to run without a mode. Pass one of:")
    print("  --check    the exact solution against the PDE (fast, no training)")
    print("  --metric   Monte Carlo metric precision vs d (fast, no training)")
    print("  --sweep    the d-sweep: 15 cells, ~2.5 h  (--seconds N to time-box)")
    print("  --figures  replay the report and figure from committed logs")
    print("  --quick    a tiny d = 2 smoke run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="verify the closed-form solution against the PDE")
    ap.add_argument("--metric", action="store_true",
                    help="measure the Monte Carlo metric's precision vs d")
    ap.add_argument("--sweep", action="store_true", help="run the d-sweep")
    ap.add_argument("--figures", action="store_true",
                    help="replay report and figure from committed logs")
    ap.add_argument("--seconds", type=float, default=None,
                    help="wall-clock budget for this --sweep call; it checkpoints "
                         "and exits when the budget expires, and the next call "
                         "resumes mid-cell")
    args = ap.parse_args()
    main(quick=args.quick, do_check=args.check, metric=args.metric,
         do_sweep=args.sweep, figures=args.figures, seconds=args.seconds)
