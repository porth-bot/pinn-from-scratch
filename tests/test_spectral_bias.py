"""The spectral-bias experiment's ground truth and its central design choice.

Two things here are load-bearing and both get checked.

1. ``exact(x, t, k)`` must really solve ``u_t = alpha_k u_xx`` -- every error in
   the sweep is measured against it.

2. The ``alpha_k = alpha_1 / k^2`` scaling must really do what the module
   docstring claims: equalize amplitude and time envelope across the whole
   family, so that the only thing varying along the sweep is spatial frequency.
   If it did not, a network that gave up and predicted ``u = 0`` would score a
   *small* error at high k, and the entire failure result would be an artifact.
   ``test_alpha_scaling_equalizes_the_family`` is the guard on that.

Plus the sine-basis projection, whose ramp subtraction is the difference between
"a straight line looks smooth" and "a straight line looks full of high
frequencies".
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from spectral_bias import (  # noqa: E402
    ALPHA1,
    K_VALUES,
    alpha_for,
    build_model,
    exact,
    sine_coefficients,
    time_to_fit,
    train_pinn,
)


# ---------------------------------------------------------------------------
# The ground truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", K_VALUES)
def test_exact_satisfies_the_pde_by_finite_differences(k):
    """u_t - alpha_k u_xx = 0 on an interior grid, to O(h^2) truncation error.

    The grid has to resolve the mode: sin(k pi x) at k=16 needs far more than
    the 100 points a k=1 check would get away with, or the FD second derivative
    is measuring its own truncation error rather than the residual.
    """
    h = 1.0 / (40 * k)  # ~40 points per half-wavelength, whatever k is
    dt = 1e-3
    x = np.arange(0.2, 0.8, h)
    t = np.array([0.3])
    XX, TT = np.meshgrid(x, t, indexing="ij")

    u_t = (exact(XX, TT + dt, k) - exact(XX, TT - dt, k)) / (2 * dt)
    u_xx = (exact(XX + h, TT, k) - 2 * exact(XX, TT, k) + exact(XX - h, TT, k)) / h ** 2
    res = u_t - alpha_for(k) * u_xx

    # The tolerance is the finite-difference truncation error, derived, not
    # guessed: the central second difference errs by (h^2/12)|u_xxxx|, and
    # u_xxxx = (k pi)^4 u with |u| <= 1, so after multiplying by alpha_k the
    # residual floor is
    #
    #     alpha_k (h^2/12) (k pi)^4 = alpha_1 pi^4 k^2 h^2 / 12.
    #
    # (The u_t difference contributes O(dt^2 (alpha_1 pi^2)^3) ~ 1e-10, nothing.)
    # With h ~ 1/(40k) this floor is k-independent -- another consequence of the
    # alpha_k = alpha_1/k^2 scaling. Allow 3x the leading term.
    truncation = ALPHA1 * np.pi ** 4 * k ** 2 * h ** 2 / 12.0
    assert np.abs(res).max() < 3 * truncation


@pytest.mark.parametrize("k", K_VALUES)
def test_exact_matches_the_initial_condition(k):
    x = np.linspace(0, 1, 101)
    assert np.allclose(exact(x, np.zeros_like(x), k), np.sin(k * np.pi * x), atol=1e-12)


@pytest.mark.parametrize("k", K_VALUES)
def test_exact_vanishes_on_the_dirichlet_boundaries(k):
    t = np.linspace(0, 1, 21)
    assert np.allclose(exact(np.zeros_like(t), t, k), 0.0, atol=1e-12)
    assert np.allclose(exact(np.ones_like(t), t, k), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# The design choice that makes the sweep measurable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", K_VALUES)
def test_alpha_scaling_cancels_the_eigenvalue(k):
    # alpha_k (k pi)^2 == alpha_1 pi^2, exactly, for every k.
    assert alpha_for(k) * (k * np.pi) ** 2 == pytest.approx(ALPHA1 * np.pi ** 2)


def test_alpha_scaling_equalizes_the_family():
    """Same peak amplitude and same time envelope at every k.

    This is what makes "the plain MLP fails at k=16" a statement about
    frequency rather than about amplitude. Without it, u(., t) at k=16 would
    have decayed by exp(-alpha (16 pi)^2 t) -- effectively zero -- and the
    lazy prediction u = 0 would have looked accurate.
    """
    t = np.linspace(0, 1, 51)
    x = np.linspace(0, 1, 2001)
    XX, TT = np.meshgrid(x, t, indexing="ij")

    envelopes = []
    for k in K_VALUES:
        u = exact(XX, TT, k)
        peak_over_t = np.abs(u).max(axis=0)  # sup_x |u(x, t)|, per time
        envelopes.append(peak_over_t)
        # amplitude stays O(1) -- it has not silently decayed to nothing
        assert peak_over_t[-1] > 0.5

    for e in envelopes[1:]:
        assert np.allclose(e, envelopes[0], rtol=1e-3)

    # and the shared envelope is the predicted exp(-alpha_1 pi^2 t)
    assert np.allclose(envelopes[0], np.exp(-ALPHA1 * np.pi ** 2 * t), rtol=1e-3)


def test_zero_prediction_scores_relative_error_one_at_every_k():
    """The 'give up and output zero' baseline is ~1.0 for all k, by construction.

    Makes the y-axis of the sweep figure interpretable: an error near 1 means
    the network learned nothing of the target, at any frequency.
    """
    x = np.linspace(0, 1, 201)
    t = np.linspace(0, 1, 101)
    XX, TT = np.meshgrid(x, t, indexing="ij")
    for k in K_VALUES:
        u = exact(XX, TT, k)
        assert np.linalg.norm(0 - u) / np.linalg.norm(u) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The sine-basis projection
# ---------------------------------------------------------------------------
def test_sine_coefficients_recover_known_modes():
    x = np.linspace(0, 1, 2001)
    u = 3.0 * np.sin(2 * np.pi * x) - 0.5 * np.sin(5 * np.pi * x)
    c = sine_coefficients(u, x, n_modes=8)
    assert c[1] == pytest.approx(3.0, abs=1e-3)  # k = 2
    assert c[4] == pytest.approx(-0.5, abs=1e-3)  # k = 5
    for k in (1, 3, 4, 6, 7, 8):
        assert abs(c[k - 1]) < 1e-3


def test_ramp_subtraction_makes_a_straight_line_read_as_smooth():
    """A straight line must have ~no spectral content -- the whole point.

    Without subtracting the endpoint interpolant, the sine series of ``u = x``
    has coefficients decaying only like 1/k, i.e. it would look like a function
    stuffed with high frequencies, and every "how high-frequency is this
    network?" measurement built on it would be garbage.
    """
    x = np.linspace(0, 1, 2001)
    c = sine_coefficients(2.0 * x - 0.3, x, n_modes=16)
    assert np.abs(c).max() < 1e-6

    # For contrast: the un-subtracted projection of the same line is large and
    # decays only like 1/k (c_k = -2*2/(k pi) * cos(k pi) ... i.e. O(1/k)).
    ks = np.arange(1, 17)
    raw = 2.0 * np.trapezoid(
        (2.0 * x - 0.3)[None, :] * np.sin(ks[:, None] * np.pi * x[None, :]), x, axis=1
    )
    assert np.abs(raw[15]) > 0.02  # k=16 "content" that is really just the ramp


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def test_time_to_fit_finds_the_first_crossing():
    hist = [(0, 1.0, 0.9), (100, 0.5, 0.3), (200, 0.2, 0.05), (300, 0.1, 0.02)]
    assert time_to_fit(hist, tol=0.1) == 200
    assert time_to_fit(hist, tol=0.01) is None


def test_build_model_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_model("siren")


# ---------------------------------------------------------------------------
# End to end (short): at the top frequency, only the Fourier model moves
# ---------------------------------------------------------------------------
def test_short_run_fourier_beats_plain_at_the_top_frequency():
    k = max(K_VALUES)
    _, hist_plain = train_pinn(k, "plain", steps=400)
    _, hist_fourier = train_pinn(k, "fourier", steps=400)
    err_plain = hist_plain[-1][2]
    err_fourier = hist_fourier[-1][2]
    # The plain net has not left the "predicted nothing" regime (error ~ 1);
    # the Fourier net has measurably started solving the problem. The full
    # 8000-step version of this is experiments/spectral_bias.py.
    assert err_plain > 0.9
    assert err_fourier < 0.9 * err_plain
