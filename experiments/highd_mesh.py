"""The mesh baseline for the d-dimensional heat problem, and the wall it hits.

``experiments/highd_heat.py`` set up a heat problem on ``[0,1]^d`` whose solution
is closed form at every d. This is the other half of the comparison: the
classical method, on a grid, measured the same way -- because the claim this
week is testing ("a mesh's cost is exponential in d and a PINN's is not") is
worth nothing unless the mesh side is a real solver run to a real accuracy,
rather than a complexity formula quoted from a textbook and asserted.

The scheme
----------
Crank-Nicolson does not survive into d dimensions unchanged. In 1D the implicit
operator ``I - (dt/2) alpha L`` is tridiagonal and the Thomas algorithm inverts
it in O(N); in d dimensions ``L = sum_j L_j`` is a Kronecker *sum*, the matrix
has bandwidth ``N^(d-1)``, and a direct solve stops being linear in the number
of unknowns. The standard fix is dimensional splitting. This module uses the
Douglas ADI scheme: with ``A_j = alpha L_j`` the discretised one-dimensional
operators and ``theta = 1/2``,

    Y_0 = u^n + dt sum_j A_j u^n,
    (I - theta dt A_j) Y_j = Y_{j-1} - theta dt A_j u^n,   j = 1 .. d,
    u^{n+1} = Y_d.

Every stage inverts one *one-dimensional* operator, so each is ``N^(d-1)``
independent tridiagonal systems of size N -- O(N^d) work per stage, O(d N^d) per
step, and no ``N^(d-1)`` bandwidth anywhere. The splitting error is O(dt^2), the
same order as the trapezoidal rule it comes from, so the scheme stays second
order in both dx and dt (measured, per d, by :func:`order_study`).

At d = 1 the scheme *is* Crank-Nicolson, algebraically and not approximately:
``Y_0 = u + dt A u`` and one stage give ``(I - (dt/2)A) Y_1 = (I + (dt/2)A) u``.
``tests/test_highd_mesh.py`` pins this solver against the shipped
``crank_nicolson.crank_nicolson`` to floating point, the same way ``highd_heat``
is pinned against ``heat``. At d >= 2 the pin is :func:`mode_amplification`, a
scalar recursion for the exact discrete amplitude of one grid sine mode, derived
by hand in the docstring there and sharing no code with the stepping loop.

What is stored, and what that costs
-----------------------------------
The 1D baseline in Sec. 6 returns the whole space-time field ``U`` of shape
``(nx+1, nt+1)``. That does not lift: at d = 3 with N = 64 that is 68 MB, at
d = 5 it is 3.7 TB, and it is not what a solver needs anyway. Only the current
time level is kept here, and the error is accumulated as the solve marches. The
homogeneous Dirichlet boundary is not stored either -- those values are
identically zero, so the unknowns are the ``(N-1)^d`` interior nodes.

That makes the memory of a run a small integer multiple of ``8 (N-1)^d`` bytes,
and the multiple is *counted in the code* rather than read off a process meter:
``gp-from-scratch``'s Day 7 found peak RSS spreading 42% across five runs on one
idle machine, so the quantity extrapolated below is the deterministic one (the
arrays the solver allocates, asserted in the tests), with ``tracemalloc``
reported beside it as a check.

Measuring the error when the grid is the thing under indictment
---------------------------------------------------------------
The exact solution is known, so no reference solve is needed. The space average
is the trapezoidal rule, which for a field vanishing on the boundary is
``(1/N^d) sum_interior``; applied to the exact solution itself it must reproduce
:func:`highd_heat.exact_ms`, and every run reports that ratio (``quad_ratio``)
so the quadrature is never taken on trust. It is second-order accurate, not
exact: the cross terms of the two-mode initial condition are half-period cosines
on ``[0,1]``, where the trapezoidal rule has no periodicity to exploit.

The relative error is then ``sqrt(<(U - u)^2>) / exact_rms``, i.e. the same
quantity ``highd_heat.rel_l2_mc`` estimates for the PINN, differing only in how
the space-time average is taken -- grid quadrature here, uniform Monte Carlo
there. Day 11 puts both methods on one axis, so it has to be the same norm.

Run:  python experiments/highd_mesh.py --order    # convergence order, d = 1..6
      python experiments/highd_mesh.py --steps    # how many time steps are needed
      python experiments/highd_mesh.py --sweep    # cost to fixed accuracy vs d
      python experiments/highd_mesh.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

from common import read_csv, write_csv
from highd_heat import HighDHeat, exact_ms, exact_rms

THETA = 0.5     # trapezoidal weight; theta = 1/2 is Crank-Nicolson at d = 1
ASPECT = 2      # time steps per spatial interval count: nt = nx // ASPECT


def nt_for(nx, aspect=ASPECT):
    """Time steps paired with ``nx`` spatial intervals. See :func:`step_study`.

    Not ``nt = nx``. The error on this problem is space-dominated well before
    that -- :func:`step_study` measures where -- and tying the two wastes a
    factor of :data:`ASPECT` of wall clock for an error change in the single
    percent. The choice is a measurement, and deliberately *not* the value that
    minimises the error, which would be exploiting a sign cancellation that does
    not generalise (again, see :func:`step_study`).
    """
    return max(1, nx // aspect)


# ---------------------------------------------------------------------------
# Batched constant-coefficient tridiagonal solve
# ---------------------------------------------------------------------------
def thomas_factor(sub, diag, sup, n):
    """Pre-factor a *constant* tridiagonal system of size ``n``.

    Every line of every ADI stage inverts the same matrix -- diagonal
    ``1 + 2s``, off-diagonals ``-s``, constant in space and unchanging in time
    -- so the Thomas algorithm's forward sweep coefficients depend on nothing
    that varies and are computed once for the whole solve. Returns
    ``(pivots, cp)`` with

        pivots[0] = b,   pivots[i] = b - a cp[i-1],   cp[i] = c / pivots[i].

    ``crank_nicolson.py`` recomputes these inside every step; at O(N) against
    O(N) of solve that is invisible in 1D, and here it would be O(N) redone for
    each of ``N^(d-1)`` lines that all want the same numbers.
    """
    pivots = np.empty(n)
    cp = np.empty(n)
    pivots[0] = diag
    cp[0] = sup / pivots[0]
    for i in range(1, n):
        pivots[i] = diag - sub * cp[i - 1]
        cp[i] = sup / pivots[i]
    return pivots, cp


def thomas_solve_axis(rhs, axis, sub, pivots, cp):
    """Solve ``T y = rhs`` along ``axis`` of a d-dimensional array, in place.

    ``rhs`` is overwritten with the solution. The forward substitution writes
    its intermediate into the same buffer the back substitution reads, so a
    stage needs no scratch of its own -- which is what holds the memory count at
    a fixed number of ``(N-1)^d`` arrays independent of d.

    Vectorised over every line at once: ``np.moveaxis`` puts the solved axis
    first (a view, no copy), so the Python loop runs over the ``n`` nodes of a
    line and each iteration does O(N^(d-1)) array work.
    """
    v = np.moveaxis(rhs, axis, 0)
    n = v.shape[0]
    v[0] /= pivots[0]
    for i in range(1, n):
        v[i] -= sub * v[i - 1]
        v[i] /= pivots[i]
    for i in range(n - 2, -1, -1):
        v[i] -= cp[i] * v[i + 1]
    return rhs


def second_difference(u, axis, out):
    """``out = u_{i-1} - 2 u_i + u_{i+1}`` along ``axis``, zeros outside.

    Undivided; the caller multiplies by ``alpha / dx^2``. Homogeneous Dirichlet
    boundaries make the neighbours of the first and last interior nodes exactly
    zero, so they contribute nothing and there are no ghost cells to carry.
    """
    v = np.moveaxis(u, axis, 0)
    w = np.moveaxis(out, axis, 0)
    np.multiply(v, -2.0, out=w)
    w[1:] += v[:-1]
    w[:-1] += v[1:]
    return out


# ---------------------------------------------------------------------------
# The exact discrete solution of one grid mode: the d >= 2 oracle
# ---------------------------------------------------------------------------
def mode_amplification(problem, k, nx, nt, theta=THETA):
    """Amplitude the scheme gives a single grid sine mode after ``nt`` steps.

    All the ``A_j`` commute and share the grid sine modes as eigenvectors:
    the second difference of ``sin(k pi x_i)`` on the interior nodes is
    ``-(2 - 2 cos(k pi dx))`` times itself, so

        A_j phi = -mu_j phi,   mu_j = alpha (2 - 2 cos(k_j pi dx)) / dx^2.

    Substituting that into the Douglas stages turns the whole d-dimensional
    step into a scalar recursion. With ``a_j = theta dt mu_j`` and
    ``m = sum_j mu_j``,

        g_0 = 1 - dt m,   g_j = (g_{j-1} + a_j) / (1 + a_j),

    and the step multiplies the mode by ``g_d``. At d = 1 that is
    ``(1 - dt mu/2)/(1 + dt mu/2)``, the Crank-Nicolson amplification factor.

    This is the independent oracle the d >= 2 tests use: it is derived by hand
    from the scheme, evaluated in four lines of scalar arithmetic, and shares
    nothing with the stepping loop -- so a stepper that disagrees with it is
    wrong in a way no self-consistency check would find.

    Also the explanation for the sign cancellation :func:`step_study` measures.
    ``mu`` *underestimates* the continuum ``alpha k^2 pi^2`` (the cosine's
    fourth-order term), so the spatial discretisation decays too slowly; and
    ``(1-z/2)/(1+z/2)`` falls below ``exp(-z)`` at third order, so the time
    discretisation decays too fast. The two errors have opposite signs.
    """
    dx = 1.0 / nx
    dt = (problem.t_range[1] - problem.t_range[0]) / nt
    mu = problem.alpha * (2.0 - 2.0 * np.cos(np.asarray(k, dtype=float)
                                             * np.pi * dx)) / (dx * dx)
    g = 1.0 - dt * float(mu.sum())
    for mu_j in mu:
        a = theta * dt * mu_j
        g = (g + a) / (1.0 + a)
    return float(g ** nt)


# ---------------------------------------------------------------------------
# The Douglas ADI solver
# ---------------------------------------------------------------------------
def grid_axis(nx):
    """Interior node coordinates of ``[0,1]`` cut into ``nx`` intervals."""
    return np.linspace(0.0, 1.0, nx + 1)[1:-1]


def mode_fields(problem, nx):
    """The spatial eigenfunctions ``prod_i sin(k_i pi x_i)`` on the interior grid.

    One ``(N-1,)*d`` array per term of the initial condition, built by
    successive outer products because the mode is separable -- so this costs
    O(N^d) rather than O(d N^d) and, more to the point, never forms the
    ``(N-1)^d x d`` coordinate table a direct evaluation would need.
    """
    x = grid_axis(nx)
    fields = []
    for k in problem.modes:
        f = np.ones(1)
        for ki in k:
            f = np.multiply.outer(f, np.sin(ki * np.pi * x))
        fields.append(np.ascontiguousarray(f.reshape((x.size,) * problem.d)))
    return fields


def _trapezoid_weights(nt, dt):
    """Trapezoidal weights over ``nt + 1`` levels, summing to the interval."""
    w = np.full(nt + 1, dt)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def solve(problem, nx, nt=None, score=True, theta=THETA):
    """Douglas ADI on ``(N-1)^d`` interior nodes for ``nt`` steps.

    Parameters
    ----------
    problem : HighDHeat
    nx : int
        Spatial intervals per axis; ``(nx - 1)^d`` unknowns.
    nt : int, optional
        Time steps. Defaults to :func:`nt_for`.
    score : bool
        Accumulate the relative L2 against the exact solution as the solve
        marches. The reported wall clock excludes scoring, which a production
        solve would not do and could not do -- it is the exact solution being
        differenced.

    Returns a dict with ``rel_l2``, ``seconds`` (the stepping loop only),
    ``unknowns``, ``arrays`` (how many ``(N-1)^d`` float64 arrays the
    implementation holds -- counted in the source, asserted in the tests),
    ``bytes_counted`` = ``arrays * 8 * unknowns``, and ``bytes_traced`` from
    ``tracemalloc``.
    """
    d = problem.d
    n = nx - 1
    if n < 1:
        raise ValueError(f"nx must be >= 2, got {nx}")
    nt = nt_for(nx) if nt is None else int(nt)
    if nt < 1:
        raise ValueError(f"nt must be >= 1, got {nt}")
    dx = 1.0 / nx
    t0, t1 = problem.t_range
    dt = (t1 - t0) / nt
    s = theta * dt * problem.alpha / (dx * dx)
    lap_scale = dt * problem.alpha / (dx * dx)

    pivots, cp = thomas_factor(-s, 1.0 + 2.0 * s, -s, n)
    shape = (n,) * d

    tracemalloc.start()
    fields = mode_fields(problem, nx)                 # len(modes) arrays
    u = np.zeros(shape)                               # + 1
    for f, a in zip(fields, problem.amps):
        u += a * f
    work = np.empty(shape)                            # + 1
    scratch = np.empty(shape)                         # + 1
    n_arrays = len(fields) + 3

    # The trapezoidal rule on a grid whose endpoints carry u = 0 is (1/N^d)
    # times the interior sum; quad_ratio below checks it against the closed form.
    cell_weight = float(nx) ** (-d)
    sq_err = np.empty(nt + 1) if score else None
    sq_exact = np.empty(nt + 1) if score else None

    def exact_level(step, out):
        """u(., t_n) on the interior grid, written into ``out``."""
        t = t0 + step * dt
        out[...] = 0.0
        for f, a, r in zip(fields, problem.amps, problem.rates):
            out += (a * np.exp(-r * t)) * f
        return out

    if score:
        sq_err[0] = 0.0                   # the initial condition is imposed exactly
        sq_exact[0] = float(np.sum(u * u)) * cell_weight

    elapsed = 0.0
    for step in range(nt):
        t_start = time.perf_counter()
        # Y_0 = u^n + dt sum_j A_j u^n
        np.copyto(work, u)
        for j in range(d):
            second_difference(u, j, scratch)
            work += lap_scale * scratch
        # (I - theta dt A_j) Y_j = Y_{j-1} - theta dt A_j u^n, j = 1 .. d.
        # A_j u^n is recomputed here rather than cached from the loop above:
        # caching all d of them would make the memory d * 8 (N-1)^d instead of a
        # constant, which is the one number this whole module is measuring.
        for j in range(d):
            second_difference(u, j, scratch)
            work -= s * scratch
            thomas_solve_axis(work, j, -s, pivots, cp)
        u, work = work, u
        elapsed += time.perf_counter() - t_start

        if score:
            exact_level(step + 1, scratch)
            np.subtract(scratch, u, out=work)
            sq_err[step + 1] = float(np.sum(work * work)) * cell_weight
            sq_exact[step + 1] = float(np.sum(scratch * scratch)) * cell_weight

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    out = dict(d=d, nx=nx, nt=nt, unknowns=n ** d, seconds=elapsed,
               arrays=n_arrays, bytes_counted=n_arrays * 8 * n ** d,
               bytes_traced=int(peak))
    if score:
        w = _trapezoid_weights(nt, dt) / (t1 - t0)
        ms_err = float(np.dot(w, sq_err))
        ms_exact = float(np.dot(w, sq_exact))
        out["rel_l2"] = float(np.sqrt(ms_err) / exact_rms(problem))
        out["quad_ratio"] = float(ms_exact / exact_ms(problem))
    else:
        out["rel_l2"] = None
        out["quad_ratio"] = None
    return out


def node_steps(d, nx, nt):
    """Node-updates in a run: ``nt (nx-1)^d``. The machine-independent cost."""
    return float(nt) * float(nx - 1) ** d


# ---------------------------------------------------------------------------
# Convergence order, and whether accuracy at fixed N depends on d
# ---------------------------------------------------------------------------
ORDER_DIMS = (1, 2, 3)
ORDER_N = (8, 16, 32, 64)
FLAT_DIMS = (1, 2, 3, 4, 5, 6)
FLAT_N = 16


def order_study(dims=ORDER_DIMS, sizes=ORDER_N):
    """Relative L2 as the grid refines, per dimension.

    ``nt = nx / 2``, so a refinement halves both dx and dt and a second-order
    scheme must cut the error by 4. Reported per d because the splitting error
    is the part that is new above d = 1: if ADI were costing an order, this is
    where it would show.
    """
    rows = []
    for d in dims:
        problem = HighDHeat(d)
        prev = None
        for nx in sizes:
            r = solve(problem, nx)
            order = "" if prev is None else f"{np.log2(prev / r['rel_l2']):.3f}"
            prev = r["rel_l2"]
            rows.append({
                "d": d, "nx": nx, "nt": r["nt"], "unknowns": r["unknowns"],
                "rel_l2": f"{r['rel_l2']:.6e}", "order": order,
                "seconds": f"{r['seconds']:.6f}",
                "quad_ratio": f"{r['quad_ratio']:.9f}",
            })
            print(f"d={d} N={nx:3d} nt={r['nt']:3d}  rel_l2 {r['rel_l2']:.4e}  "
                  f"order {order:>6}  {r['seconds']*1e3:9.2f} ms  "
                  f"quad {r['quad_ratio']:.7f}")
    return rows


def flatness_study(dims=FLAT_DIMS, nx=FLAT_N):
    """Error at a *fixed* grid resolution, as d grows. The extrapolation's premise.

    Everything projected past the largest runnable d assumes the N needed for a
    given accuracy does not itself grow with d -- otherwise the projection is
    optimistic and the mesh looks better than it is. There is a reason to expect
    flatness here and it is worth stating so it can be checked: the leading
    spatial truncation error of mode k is ``alpha_d dx^2 pi^4 sum_i k_i^4 / 12``,
    which grows with the d terms in the sum but is multiplied by
    ``alpha_d = alpha_1 / d``, so the two effects cancel to leading order.

    This measures it instead of resting on it.
    """
    rows = []
    for d in dims:
        problem = HighDHeat(d)
        r = solve(problem, nx)
        rows.append({"d": d, "nx": nx, "nt": r["nt"], "unknowns": r["unknowns"],
                     "rel_l2": f"{r['rel_l2']:.6e}",
                     "seconds": f"{r['seconds']:.6f}"})
        print(f"  fixed N={nx}: d={d}  {r['unknowns']:>12,} unknowns  "
              f"rel_l2 {r['rel_l2']:.4e}  {r['seconds']:.3f}s")
    return rows


# ---------------------------------------------------------------------------
# How many time steps are actually needed
# ---------------------------------------------------------------------------
STEP_CELLS = ((1, 128), (2, 128), (3, 64), (4, 32))
STEP_NT = (1, 2, 4, 8, 16, 32, 64, 128)


def step_study(cells=STEP_CELLS, nts=STEP_NT):
    """Error against ``nt`` at fixed ``nx``: where the time step stops mattering.

    The cost of a run is ``nt (N-1)^d``, so the ``nt`` chosen for a given N is a
    free factor in every wall-clock number downstream, and picking it by habit
    (``nt = nx``, as Sec. 6 does in 1D) would inflate the mesh's cost for
    nothing. Refining dt alone at fixed dx isolates the temporal error.

    The measurement has a wrinkle that is a result rather than an annoyance: the
    error is *not monotone* in ``nt``. It falls at second order, undershoots the
    space-limited plateau, and comes back up to it. The two truncation errors
    have opposite signs -- see :func:`mode_amplification` -- so at one particular
    ``nt`` they cancel. Tuning ``nt`` to that minimum would be reporting a
    cancellation that belongs to this problem and this grid, so :data:`ASPECT` is
    set from the plateau instead and the residual cancellation credit at the
    operating point is reported per cell.
    """
    rows = []
    for d, nx in cells:
        problem = HighDHeat(d)
        errs = {}
        for nt in nts:
            errs[nt] = solve(problem, nx, nt)["rel_l2"]
        plateau = errs[max(nts)]
        best_nt = min(errs, key=errs.get)
        operating = nt_for(nx)
        for nt in nts:
            rows.append({"d": d, "nx": nx, "nt": nt,
                         "rel_l2": f"{errs[nt]:.6e}",
                         "over_plateau": f"{errs[nt]/plateau:.4f}"})
        print(f"d={d} N={nx}: plateau {plateau:.3e} at nt={max(nts)}; "
              f"minimum {errs[best_nt]:.3e} at nt={best_nt} "
              f"({errs[best_nt]/plateau:.3f}x plateau); "
              f"operating nt={operating} gives {errs[operating]:.3e} "
              f"({errs[operating]/plateau:.3f}x)")
    return rows


# ---------------------------------------------------------------------------
# Cost to a fixed accuracy, and the extrapolation past what will run
# ---------------------------------------------------------------------------
TARGETS = (1e-2, 1e-3)
SWEEP_DIMS = (1, 2, 3, 4, 5, 6)
EXTRAPOLATE_DIMS = (7, 8, 10, 12, 16)
UNKNOWN_BUDGET = 60_000_000     # ~2.4 GB at 5 arrays; what this machine will run


def required_nx(problem, target, budget=UNKNOWN_BUDGET, verbose=False):
    """Smallest even ``N`` whose solve reaches ``target``, predicted then *run*.

    A second-order scheme has ``err ~ C N^-p``, so C and p come off a small
    sweep and ``N* = (C/target)^(1/p)`` is a prediction. The prediction is not
    the answer: it is rounded up to an even N, the solve is actually performed,
    and N is stepped up until the measured error is at or below the target. The
    achieved error is returned with it, because "the grid that reaches 1e-2" is
    only meaningful next to what it actually reached.

    Returns ``None`` if reaching the target needs more than ``budget`` unknowns
    -- recorded as a skipped cell rather than dropped.
    """
    fit_sizes = [nx for nx in (4, 8, 16, 32) if (nx - 1) ** problem.d <= budget]
    if len(fit_sizes) < 2:
        fit_sizes = [4, 6]
    errs = [solve(problem, nx)["rel_l2"] for nx in fit_sizes]
    slope, intercept = np.polyfit(np.log(np.array(fit_sizes, float)),
                                  np.log(errs), 1)
    p, C = -slope, np.exp(intercept)
    guess = int(np.ceil((C / target) ** (1.0 / p)))
    nx = max(4, guess + guess % 2)

    for _ in range(12):
        if (nx - 1) ** problem.d > budget:
            if verbose:
                print(f"    d={problem.d}: N={nx} needs "
                      f"{(nx-1)**problem.d:.3g} unknowns > budget; skipped")
            return None
        r = solve(problem, nx)
        if verbose:
            print(f"    d={problem.d} try N={nx:4d} nt={r['nt']:4d} -> "
                  f"{r['rel_l2']:.4e}  ({r['seconds']:.2f}s)")
        if r["rel_l2"] <= target:
            r.update(fitted_order=float(p), predicted_nx=guess)
            return r
        nx = int(np.ceil(nx * 1.25))
        nx += nx % 2
    raise RuntimeError(f"no grid up to N={nx} reached {target} at d={problem.d}")


def cost_sweep(targets=TARGETS, dims=SWEEP_DIMS, budget=UNKNOWN_BUDGET,
               verbose=True):
    """Run the mesh solver to each accuracy target at every d that will fit.

    ``dims`` goes past the d = 1, 2, 3 the study was scoped for: d = 4, 5 and 6
    turned out to be runnable at the looser target on this machine, and a
    measured point is worth more than an extrapolated one. Cells that exceed
    ``budget`` unknowns are skipped and *recorded as skipped*, so the CSV shows
    where the measurement stops rather than implying it went further.
    """
    rows, skipped = [], []
    for target in targets:
        for d in dims:
            problem = HighDHeat(d)
            if verbose:
                print(f"  target {target:.0e}, d={d}")
            r = required_nx(problem, target, budget=budget, verbose=verbose)
            if r is None:
                skipped.append({"target": f"{target:.0e}", "d": d,
                                "reason": f"needs > {budget:,} unknowns"})
                continue
            rows.append({
                "target": f"{target:.0e}", "d": d, "nx": r["nx"], "nt": r["nt"],
                "unknowns": r["unknowns"], "rel_l2": f"{r['rel_l2']:.6e}",
                "seconds": f"{r['seconds']:.6f}",
                "node_steps": f"{node_steps(d, r['nx'], r['nt']):.6e}",
                "bytes_counted": r["bytes_counted"],
                "bytes_traced": r["bytes_traced"],
                "arrays": r["arrays"],
                "fitted_order": f"{r['fitted_order']:.4f}",
            })
            if verbose:
                print(f"    -> N={r['nx']}, {r['unknowns']:,} unknowns, "
                      f"rel_l2 {r['rel_l2']:.3e}, {r['seconds']:.3f}s, "
                      f"{r['bytes_counted']/1e6:.1f} MB counted / "
                      f"{r['bytes_traced']/1e6:.1f} MB traced")
    return rows, skipped


def fit_cost_model(rows):
    """Fit ``seconds = nt d (c_py (N-1) + tau (N-1)^d)`` to the measured cells.

    The obvious one-parameter model -- a constant ``tau`` in
    ``seconds = tau d nt (N-1)^d`` -- does not survive contact with the
    measurement: across the cells of the cost sweep that quotient spans a factor
    of about 1900, and quoting its median would be quoting a number that fits
    almost none of the data. The reason is visible in
    :func:`thomas_solve_axis`: each line sweep is a Python loop over the ``N-1``
    nodes of an axis, and each iteration does O((N-1)^(d-1)) of array work. At
    d = 1 that array work is a single float, so a small cell measures the
    interpreter, not the arithmetic.

    So the model gets the second term it needs: ``c_py`` per axis-sweep node
    (interpreter-bound, independent of d) and ``tau`` per node touched
    (array-bound). Fitted with relative residuals, since the cells span five
    orders of magnitude in wall clock and an unweighted least squares would be
    a fit to the largest cell alone.

    Returns ``tau``, ``c_py``, ``worst_rel`` (the largest |predicted/measured -
    1| over the cells), ``worst_rel_array`` (the same over the cells where the
    array term already dominates the interpreter one) and ``naive_spread``, the
    one-parameter quotient's range -- reported rather than hidden, because it is
    the reason the model has two terms.

    This model is *evidence for the scaling law*, not the projection itself.
    :func:`extrapolate` scales from the largest measured cell instead, which
    reproduces that cell exactly; a global fit misses it by tens of percent,
    since two coefficients cannot absorb the per-call numpy overheads that vary
    across five orders of magnitude of array size.
    """
    d = np.array([int(r["d"]) for r in rows], dtype=float)
    n = np.array([int(r["nx"]) - 1 for r in rows], dtype=float)
    nt = np.array([int(r["nt"]) for r in rows], dtype=float)
    sec = np.array([float(r["seconds"]) for r in rows])
    unknowns = np.array([float(r["unknowns"]) for r in rows])

    A = np.stack([nt * d * n, nt * d * unknowns], axis=1) / sec[:, None]
    coef, *_ = np.linalg.lstsq(A, np.ones_like(sec), rcond=None)
    c_py, tau = float(coef[0]), float(coef[1])
    pred = nt * d * (c_py * n + tau * unknowns)
    naive = sec / (d * nt * unknowns)
    array_dominated = tau * unknowns > c_py * n
    return dict(tau=tau, c_py=c_py,
                worst_rel=float(np.max(np.abs(pred / sec - 1.0))),
                worst_rel_array=float(np.max(np.abs(
                    (pred / sec - 1.0)[array_dominated]))),
                n_array_dominated=int(array_dominated.sum()),
                naive_spread=float(naive.max() / naive.min()))


def model_seconds(model, d, nx, nt):
    """Wall clock the fitted model predicts for a cell. Used by the projection."""
    n = float(nx - 1)
    return nt * d * (model["c_py"] * n + model["tau"] * n ** d)


def extrapolate(rows, dims=EXTRAPOLATE_DIMS):
    """Project to dimensions that will not run, by scaling the largest that did.

    **These rows are arithmetic, not measurements**, and the CSV column
    ``source`` says so on every one of them.

    The projection scales the largest measured cell for each target by the
    scheme's own complexity -- ``d`` stages times ``(N-1)^d`` nodes, at fixed
    ``nt`` -- rather than evaluating a fitted model. Both would be defensible;
    this one is preferred because it is exact at the anchor, whereas the global
    two-parameter fit of :func:`fit_cost_model` misses that same cell by tens of
    percent. The fit is still computed and reported, as the evidence that the
    law being scaled by is the law the measurements follow.

    Two assumptions, both measured elsewhere rather than assumed here: that
    seconds are proportional to ``d (N-1)^d`` (:func:`fit_cost_model`), and that
    N need not grow with d to hold the accuracy (:func:`flatness_study`).
    """
    model = fit_cost_model(rows)
    out = []
    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    for target, cells in by_target.items():
        anchor = max(cells, key=lambda c: int(c["d"]))
        nx, nt = int(anchor["nx"]), int(anchor["nt"])
        d0, arrays = int(anchor["d"]), int(anchor["arrays"])
        sec0, unknowns0 = float(anchor["seconds"]), float(anchor["unknowns"])
        for d in dims:
            unknowns = float(nx - 1) ** d
            out.append({
                "target": target, "d": d, "nx": nx, "nt": nt,
                "unknowns": f"{unknowns:.6e}",
                "bytes_counted": f"{arrays * 8 * unknowns:.6e}",
                "seconds": f"{sec0 * (d / d0) * (unknowns / unknowns0):.6e}",
                "source": f"extrapolated by d (N-1)^d from the measured "
                          f"d={d0} cell ({sec0:.4g} s)",
            })
    return out, model


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
        if abs(b) < 1024:
            return f"{b:.3g} {unit}"
        b /= 1024
    return f"{b:.3g} YB"


def _fmt_seconds(s):
    if s < 60:
        return f"{s:.3g} s"
    for div, unit in ((60.0, "min"), (60.0, "h"), (24.0, "d"), (365.25, "yr")):
        s /= div
        if s < 400 or unit == "yr":
            return f"{s:.3g} {unit}"
    return f"{s:.3g} yr"


def report(measured, projected, skipped, model):
    print("\n" + "=" * 84)
    print("Mesh cost to a fixed relative L2: measured where it runs, "
          "extrapolated where it does not")
    print("=" * 84)
    print(f"scaling law evidence: seconds = nt d (c_py (N-1) + tau (N-1)^d) "
          f"with tau = {model['tau']:.3e} s/node, c_py = {model['c_py']:.3e} s")
    print(f"  fits every measured cell to {model['worst_rel']*100:.0f}%, and "
          f"the {model['n_array_dominated']} array-dominated cells to "
          f"{model['worst_rel_array']*100:.0f}%")
    print(f"  (a single constant seconds-per-node-step spans "
          f"{model['naive_spread']:.0f}x over the same cells -- the small-d "
          f"solves measure the interpreter, not the arithmetic)")
    print("  the projection below scales the largest measured cell, not this "
          "fit, so it is exact at its anchor")
    for target in sorted({r["target"] for r in measured}, reverse=True):
        print(f"\ntarget rel L2 = {target}")
        print(f"  {'d':>3} {'N':>5} {'nt':>4} {'unknowns':>13} {'rel L2':>11} "
              f"{'wall':>11} {'memory':>11}   source")
        for r in measured:
            if r["target"] == target:
                print(f"  {r['d']:>3} {int(r['nx']):>5} {int(r['nt']):>4} "
                      f"{int(r['unknowns']):>13,} {float(r['rel_l2']):>11.3e} "
                      f"{_fmt_seconds(float(r['seconds'])):>11} "
                      f"{_fmt_bytes(float(r['bytes_counted'])):>11}   measured")
        for r in projected:
            if r["target"] == target:
                print(f"  {r['d']:>3} {int(r['nx']):>5} {int(r['nt']):>4} "
                      f"{float(r['unknowns']):>13.2e} {'--':>11} "
                      f"{_fmt_seconds(float(r['seconds'])):>11} "
                      f"{_fmt_bytes(float(r['bytes_counted'])):>11}   "
                      f"EXTRAPOLATED")
    if skipped:
        print("\nskipped (would not fit and was not run):")
        for s in skipped:
            print(f"  target {s['target']} d={s['d']}: {s['reason']}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
COLORS = {"1e-02": "#2166ac", "1e-03": "#b2182b"}


def _make_figure(measured, projected):
    from common import plt, savefig

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    for ax, key, label in ((axes[0], "bytes_counted", "memory (bytes)"),
                           (axes[1], "seconds", "wall-clock (s)")):
        for target in sorted({r["target"] for r in measured}, reverse=True):
            c = COLORS.get(target, "#444444")
            md = sorted((int(r["d"]), float(r[key])) for r in measured
                        if r["target"] == target)
            pj = sorted((int(r["d"]), float(r[key])) for r in projected
                        if r["target"] == target)
            ax.semilogy([p[0] for p in [md[-1]] + pj],
                        [p[1] for p in [md[-1]] + pj],
                        "--", color=c, lw=1.1, alpha=0.85)
            ax.semilogy([p[0] for p in pj], [p[1] for p in pj], "o",
                        mfc="none", color=c, ms=5)
            ax.semilogy([p[0] for p in md], [p[1] for p in md], "o-", color=c,
                        ms=5.5, label=f"rel $L^2 \\leq$ {float(target):.0e}")
        ax.set_xlabel("spatial dimension $d$")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.25)
    for ax, level, text in ((axes[0], 16 * 1024 ** 3, "16 GB"),
                            (axes[0], 1024 ** 5, "1 PB"),
                            (axes[1], 3600.0, "1 hour"),
                            (axes[1], 3600.0 * 24 * 365.25, "1 year")):
        ax.axhline(level, color="#666666", lw=0.9, ls=":")
        ax.annotate(text, (0.7, level), fontsize=7, color="#666666",
                    textcoords="offset points", xytext=(0, 3))
    axes[0].set_title("Douglas ADI: memory at fixed accuracy")
    axes[1].set_title("Douglas ADI: wall-clock at fixed accuracy")
    axes[0].legend(loc="upper left")
    fig.text(0.5, -0.04, "filled = measured (solve actually run); hollow = "
             "extrapolated from the measured cost model",
             ha="center", fontsize=8, color="#555555")
    savefig(fig, "highd_mesh.png")


def figures_from_committed():
    """Rebuild the figure from the committed CSVs rather than re-solving.

    One axis is wall clock, so a rerun on another machine moves every measured
    point -- and every extrapolated one with it, since the projection is
    anchored on a measured seconds-per-node-step. The committed CSV is the
    measurement.
    """
    _make_figure(read_csv("highd_mesh_cost.csv"),
                 read_csv("highd_mesh_extrapolated.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(quick=False, order=False, steps=False, sweep=False, figures=False):
    if figures:
        figures_from_committed()
        return

    if quick:
        problem = HighDHeat(2)
        r = solve(problem, 16)
        print(f"[quick] {problem}")
        print(f"  N=16 nt={r['nt']}  rel_l2 {r['rel_l2']:.4e}  "
              f"quad ratio {r['quad_ratio']:.9f}  {r['seconds']*1e3:.2f} ms")
        print(f"  {r['unknowns']:,} unknowns, {r['arrays']} arrays, "
              f"{_fmt_bytes(r['bytes_counted'])} counted / "
              f"{_fmt_bytes(r['bytes_traced'])} traced")
        print(f"  one-mode oracle at nt={r['nt']}: "
              f"{mode_amplification(problem, problem.modes[0], 16, r['nt']):.9f}")
        return

    if order:
        rows = order_study()
        write_csv("highd_mesh_order.csv",
                  ["d", "nx", "nt", "unknowns", "rel_l2", "order", "seconds",
                   "quad_ratio"], rows)
        print(f"\nerror at fixed N={FLAT_N} as d grows (the extrapolation's premise):")
        flat = flatness_study()
        write_csv("highd_mesh_flat.csv",
                  ["d", "nx", "nt", "unknowns", "rel_l2", "seconds"], flat)
        return

    if steps:
        rows = step_study()
        write_csv("highd_mesh_steps.csv",
                  ["d", "nx", "nt", "rel_l2", "over_plateau"], rows)
        return

    if sweep:
        measured, skipped = cost_sweep()
        projected, model = extrapolate(measured)
        write_csv("highd_mesh_cost.csv",
                  ["target", "d", "nx", "nt", "unknowns", "rel_l2", "seconds",
                   "node_steps", "bytes_counted", "bytes_traced", "arrays",
                   "fitted_order"], measured)
        write_csv("highd_mesh_extrapolated.csv",
                  ["target", "d", "nx", "nt", "unknowns", "bytes_counted",
                   "seconds", "source"], projected)
        report(measured, projected, skipped, model)
        _make_figure(measured, projected)
        return

    print(__doc__.strip().splitlines()[0])
    print("\nNothing to run without a mode. Pass one of:")
    print("  --order    convergence order per d, then error at fixed N vs d")
    print("  --steps    how many time steps a given grid actually needs")
    print("  --sweep    cost to a fixed accuracy vs d, measured then extrapolated")
    print("  --quick    a tiny d = 2 smoke check")
    print("  --figures  redraw from the committed CSVs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--order", action="store_true",
                    help="convergence order of the ADI scheme, per dimension")
    ap.add_argument("--steps", action="store_true",
                    help="error against the number of time steps, at fixed grid")
    ap.add_argument("--sweep", action="store_true",
                    help="cost to a fixed accuracy vs d, measured then extrapolated")
    ap.add_argument("--figures", action="store_true",
                    help="rebuild the figure from the committed CSVs")
    args = ap.parse_args()
    main(quick=args.quick, order=args.order, steps=args.steps, sweep=args.sweep,
         figures=args.figures)
