"""The leapfrog baseline is only an honest yardstick for Sec. 15's PINN if it is
itself right, and this scheme has two properties that a plausible-but-wrong
implementation would not have: it is *exact* at a Courant number of 1, and it
conserves a specific staggered discrete energy below it. Both are asserted here
against the same d'Alembert ground truth the network is scored on, so a bug
that made the mesh look good would have to fake both.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import wave as W  # noqa: E402
from wave_leapfrog import (  # noqa: E402
    _delta2,
    discrete_energy,
    order_study,
    solve,
    space_time_error,
    undefined_at_a_node,
)


def test_courant_one_is_exact_for_both_initial_conditions():
    """At r = 1 the update is u_j^{n+1} = u_{j+1}^n + u_{j-1}^n - u_j^{n-1},
    which is d'Alembert on the characteristics -- so the answer is exact for
    *any* initial data, the plucked string's corner included. This is the
    section's headline and it is a property of the scheme, not of smoothness."""
    for ic in W.ICS:
        x, t, U = solve(nx=51, cfl=1.0, ic=ic, keep=True)
        assert space_time_error(x, t, U, ic) < 1e-13


def test_courant_one_update_really_is_the_characteristic_shift():
    """The identity behind the test above, checked on the arrays rather than
    argued for: with r = 1 the interior update loses its dependence on u^n_j."""
    x, t, U = solve(nx=41, cfl=1.0, ic="pluck", keep=True)
    got = U[1:-1, 2:]
    want = U[2:, 1:-1] + U[:-2, 1:-1] - U[1:-1, :-2]
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_below_courant_one_it_is_second_order_on_smooth_data():
    """Away from a turning time of the standing wave, the sine converges at the
    scheme's advertised order. Asserted at t = 0.7 specifically -- see the next
    test for why t = 1 or 2 would not establish this."""
    rows = [r for r in order_study(ics=("sine",), times=(0.7,),
                                   grids=(101, 201, 401), write=False)
            if r["order"]]
    orders = [float(r["order"]) for r in rows]
    assert all(1.9 < o < 2.1 for o in orders), orders


def test_a_turning_time_reports_twice_the_order():
    """The trap the section is built around, asserted so it cannot quietly stop
    being true: the sine is a standing wave of period 2/c, so at t = 2 it sits
    at an extremum of cos(pi c t) and a phase error enters quadratically. The
    measured order there is 4. Sec. 15's own time window ends at t = 2."""
    rows = [r for r in order_study(ics=("sine",), times=(2.0,),
                                   grids=(101, 201, 401), write=False)
            if r["order"]]
    orders = [float(r["order"]) for r in rows]
    assert all(3.8 < o < 4.2 for o in orders), orders


def test_the_relative_error_has_no_denominator_at_a_node():
    """The other way of measuring it wrongly: at t = 0.5 the sine's exact
    solution is identically zero, so ||u - u_hat|| / ||u|| divides by nothing.
    This is why order_study normalizes by the initial displacement instead."""
    assert undefined_at_a_node("sine", 0.5) < 1e-12
    assert undefined_at_a_node("pluck", 0.5) > 0.1   # the pluck has no such node


def test_the_corner_costs_an_order():
    """The pluck is only C^0, so the O(dx^2) truncation term carries a fourth
    derivative that does not exist and the scheme drops to first order. This is
    the section's comparison with the network's own trouble at the corner, so
    it is asserted rather than only tabulated."""
    rows = [r for r in order_study(ics=("pluck",), times=(0.7,),
                                   grids=(101, 201, 401), write=False)
            if r["order"]]
    orders = [float(r["order"]) for r in rows]
    assert all(0.9 < o < 1.2 for o in orders), orders


def test_discrete_energy_is_conserved_below_the_cfl_limit():
    """The staggered identity: the potential term pairs *two* time levels. A
    same-level ||D_x u^n||^2 would oscillate at O(dt^2) and read as drift, so
    this test would fail on the natural-looking wrong formula."""
    for ic in W.ICS:
        x, t, U = solve(nx=101, cfl=0.5, ic=ic, keep=True)
        E = discrete_energy(x, U, t[1] - t[0])
        assert np.abs(E - E[0]).max() / E[0] < 1e-11


def test_discrete_energy_is_positive_and_near_the_exact_one():
    """Positive definiteness holds exactly when r < 1 -- it is the energy proof
    of the CFL condition -- and on a resolved grid the value approaches the
    closed-form conserved energy wave.py derives."""
    for ic in W.ICS:
        x, t, U = solve(nx=401, cfl=0.5, ic=ic, keep=True)
        E = discrete_energy(x, U, t[1] - t[0])
        assert E.min() > 0
        assert abs(E[0] / W.exact_energy(ic) - 1.0) < 5e-3


def test_the_scheme_respects_the_boundary_and_initial_conditions():
    for ic in W.ICS:
        x, t, U = solve(nx=61, cfl=0.5, ic=ic, keep=True)
        np.testing.assert_allclose(U[0, :], 0.0, atol=0.0)
        np.testing.assert_allclose(U[-1, :], 0.0, atol=0.0)
        np.testing.assert_allclose(U[:, 0], W.f0(x, ic), atol=1e-15)


def test_the_start_step_is_the_taylor_form_and_the_corner_breaks_it():
    """The first level uses u^1 = u^0 + (dt^2/2) u_tt = u^0 + (r^2/2) delta^2 u^0,
    so the discrete initial velocity is not zero but (dt/2) c^2 u_xx -- O(dt),
    and that is correct, since u^1 itself is second-order accurate.

    That argument needs u_xx to exist. It does not at the pluck's corner, where
    delta^2 f is O(1) instead of O(dx^2), so the start step injects an O(1)
    velocity at exactly one node. Both halves are asserted: the smooth control
    shows the O(dt) scaling, and the pluck shows the defect is confined to one
    node and does not shrink with the grid. This is the same non-smoothness the
    network's own trouble is about, arriving in the scheme as a start-up error
    rather than as a representation failure."""
    prev = None
    for nx in (61, 121, 241):
        x, t, U = solve(nx=nx, cfl=0.5, ic="sine", keep=True)
        v = np.abs((U[:, 1] - U[:, 0]) / (t[1] - t[0])).max()
        if prev is not None:
            assert 1.8 < prev / v < 2.2, (prev, v)
        prev = v

    peaks = []
    for nx in (61, 121, 241):
        x, t, U = solve(nx=nx, cfl=0.5, ic="pluck", keep=True)
        v = np.abs((U[:, 1] - U[:, 0]) / (t[1] - t[0]))
        assert (v > 1e-6).sum() == 1
        peaks.append(v.max())
    assert min(peaks) > 0.5 and max(peaks) / min(peaks) < 1.1


def test_solve_rejects_an_unstable_courant_number():
    with pytest.raises(ValueError, match="CFL"):
        solve(nx=21, cfl=1.5, ic="sine")


def test_delta2_is_the_second_difference_with_pinned_ends():
    u = np.array([0.0, 1.0, 4.0, 9.0, 0.0])
    # interior: 0-2+4 = 2, 1-8+9 = 2, 4-18+0 = -14; ends pinned to 0
    np.testing.assert_allclose(_delta2(u), [0.0, 2.0, 2.0, -14.0, 0.0])


def test_solve_lands_exactly_on_the_end_time():
    """dt is trimmed so an integer number of steps hits t_end. Without it the
    error reported would be partly a different final time, and the trim must
    only ever lower the Courant number."""
    for nx, cfl, t_end in ((37, 0.9, 2.0), (50, 0.73, 1.3), (101, 1.0, 0.7)):
        x, t, _ = solve(nx=nx, cfl=cfl, ic="sine", t_end=t_end)
        assert t[-1] == pytest.approx(t_end, abs=1e-12)
        dx, dt = x[1] - x[0], t[1] - t[0]
        assert W.C * dt / dx <= cfl + 1e-12
