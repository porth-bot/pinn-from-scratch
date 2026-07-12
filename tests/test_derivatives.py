"""Autograd derivatives vs finite differences on functions with known slopes.

The whole PINN edifice rests on the derivative helpers returning the *right*
per-point derivatives, so we check them two ways:

1. Against central finite differences of the same callable (the general,
   assumption-free check).
2. Against a hand-derived closed form on an analytic test function, so a bug
   that happened to agree with a bad FD step still gets caught.

The test function is u(x, t) = sin(a x) * exp(-b t), whose derivatives are all
known in closed form:
    u_x  =  a cos(a x) exp(-b t)
    u_xx = -a^2 sin(a x) exp(-b t) = -a^2 u
    u_t  = -b sin(a x) exp(-b t)   = -b u
It also satisfies a heat equation u_t = alpha u_xx with alpha = b / a^2, which
we use as an end-to-end residual sanity check.

Finite-difference comparisons are run in float64: the autograd derivatives are
exact, but a central difference amplifies rounding by 1/h, so at float32 the FD
*reference* itself is only good to ~1e-2 -- double precision keeps the check
meaningful rather than the test measuring float32 noise.
"""

import torch

from pinn import derivatives as D

# float64 locally (not a global default, which would leak into other test
# files sharing the pytest process) so the FD references are meaningful.
DT = torch.float64
A, B = 2.0, 0.7


def _u(coords):
    x, t = coords[:, 0:1], coords[:, 1:2]
    return torch.sin(A * x) * torch.exp(-B * t)


def _coords(n=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = -1 + 2 * torch.rand(n, 1, generator=gen, dtype=DT)
    t = torch.rand(n, 1, generator=gen, dtype=DT)
    c = torch.cat([x, t], dim=1)
    c.requires_grad_(True)
    return c


def _central_diff(f, coords, dim, h=1e-4):
    """Central finite difference of scalar-valued f along one input column."""
    e = torch.zeros_like(coords)
    e[:, dim] = h
    with torch.no_grad():
        return (f(coords + e) - f(coords - e)) / (2 * h)


def test_first_derivatives_match_finite_difference():
    coords = _coords()
    u = _u(coords)
    ux = D.u_x(u, coords)
    ut = D.u_t(u, coords)
    fd_x = _central_diff(_u, coords, D.X)
    fd_t = _central_diff(_u, coords, D.T)
    assert torch.allclose(ux, fd_x, atol=1e-5)
    assert torch.allclose(ut, fd_t, atol=1e-5)


def test_first_derivatives_match_closed_form():
    coords = _coords()
    x, t = coords[:, 0:1], coords[:, 1:2]
    u = _u(coords)
    ux_true = A * torch.cos(A * x) * torch.exp(-B * t)
    ut_true = -B * _u(coords)
    assert torch.allclose(D.u_x(u, coords), ux_true, atol=1e-6)
    assert torch.allclose(D.u_t(u, coords), ut_true, atol=1e-6)


def test_second_derivative_matches_finite_difference_of_first():
    coords = _coords()
    u = _u(coords)
    uxx = D.u_xx(u, coords)
    # closed form: u_xx = -A^2 u
    uxx_true = -(A ** 2) * u
    assert torch.allclose(uxx, uxx_true, atol=1e-5)

    # and a second-order central FD of the value itself
    h = 1e-3
    e = torch.zeros_like(coords)
    e[:, D.X] = h
    with torch.no_grad():
        fd = (_u(coords + e) - 2 * _u(coords) + _u(coords - e)) / h ** 2
    assert torch.allclose(uxx, fd, atol=1e-2)


def test_partial_selects_the_right_axis():
    coords = _coords()
    u = _u(coords)
    assert torch.allclose(D.partial(u, coords, D.X), D.u_x(u, coords))
    assert torch.allclose(D.partial(u, coords, D.T), D.u_t(u, coords))


def test_laplacian_sums_unmixed_second_derivatives():
    # On a 2D spatial field u = sin(a x0) + sin(a x1), lap = -a^2 (sin+sin).
    gen = torch.Generator().manual_seed(1)
    c = torch.rand(50, 2, generator=gen, dtype=DT)
    c.requires_grad_(True)
    u = torch.sin(A * c[:, 0:1]) + torch.sin(A * c[:, 1:2])
    lap = D.laplacian(u, c)
    lap_true = -(A ** 2) * (torch.sin(A * c[:, 0:1]) + torch.sin(A * c[:, 1:2]))
    assert torch.allclose(lap, lap_true, atol=1e-5)


def test_higher_order_graph_is_retained():
    # u_xx must itself be differentiable (create_graph=True): d/dx of u_xx
    # should equal u_xxx = -A^3 cos(A x) exp(-B t).
    coords = _coords()
    u = _u(coords)
    uxx = D.u_xx(u, coords)
    uxxx = D.partial(uxx, coords, D.X)
    x, t = coords[:, 0:1], coords[:, 1:2]
    uxxx_true = -(A ** 3) * torch.cos(A * x) * torch.exp(-B * t)
    assert torch.allclose(uxxx, uxxx_true, atol=1e-5)


def test_analytic_field_satisfies_its_heat_residual():
    # u_t = alpha u_xx with alpha = B / A^2, checked through the helpers.
    coords = _coords()
    u = _u(coords)
    alpha = B / A ** 2
    r = D.u_t(u, coords) - alpha * D.u_xx(u, coords)
    assert torch.max(torch.abs(r)) < 1e-5
