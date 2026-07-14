"""Fourier-feature embedding: shapes, the Bochner limit, and the mechanism.

The interesting tests here are the last three. Anyone can check a tensor shape;
the claims this module actually rests on are

  (a) the embedding's inner product converges to the Gaussian kernel
      ``exp(-2 pi^2 sigma^2 d^2)`` predicted by Bochner's theorem,
  (b) an *untrained* Fourier model already carries high-frequency energy that
      an untrained tanh MLP does not (this is the mechanism -- the frequencies
      are handed to the network, not manufactured by it), and
  (c) consequently the Fourier model fits a high-frequency target that a plain
      MLP of the same width, depth and step budget does not.
"""

import math

import numpy as np
import pytest
import torch

from pinn.derivatives import u_xx
from pinn.features import FourierFeatures, FourierMLP
from pinn.model import MLP, set_seed


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def test_embedding_shape_and_out_dim():
    emb = FourierFeatures(in_dim=2, n_features=32, sigma=1.0)
    out = emb(torch.rand(7, 2))
    assert emb.out_dim == 64
    assert out.shape == (7, 64)
    assert torch.isfinite(out).all()
    # cos/sin of anything stays in [-1, 1]
    assert out.abs().max() <= 1.0 + 1e-6


def test_B_is_a_frozen_buffer_not_a_parameter():
    model = FourierMLP(in_dim=2, n_features=16, sigma=3.0, width=8, depth=2)
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.features.B) not in param_ids
    assert "features.B" in dict(model.named_buffers())
    # and it survives a state_dict round trip
    assert "features.B" in model.state_dict()


def test_per_coordinate_sigma_scales_the_right_axis():
    # sigma = (10, 0.1): the x-column of B should be ~100x wider than the t one.
    emb = FourierFeatures(in_dim=2, n_features=4000, sigma=(10.0, 0.1), seed=0)
    sd = emb.B.std(dim=0)
    assert sd[0] == pytest.approx(10.0, rel=0.05)
    assert sd[1] == pytest.approx(0.1, rel=0.05)


def test_seeded_determinism_and_seed_sensitivity():
    a = FourierFeatures(n_features=16, sigma=2.0, seed=7)
    b = FourierFeatures(n_features=16, sigma=2.0, seed=7)
    c = FourierFeatures(n_features=16, sigma=2.0, seed=8)
    assert torch.equal(a.B, b.B)
    assert not torch.equal(a.B, c.B)


def test_bad_sigma_raises():
    with pytest.raises(ValueError):
        FourierFeatures(in_dim=2, sigma=(1.0, 2.0, 3.0))  # wrong length
    with pytest.raises(ValueError):
        FourierFeatures(in_dim=2, sigma=0.0)  # not positive


# ---------------------------------------------------------------------------
# (a) Bochner: the finite-m inner product converges to the Gaussian kernel
# ---------------------------------------------------------------------------
def test_inner_product_converges_to_the_gaussian_kernel():
    """gamma(v).gamma(v') / m  ->  exp(-2 pi^2 sum_j sigma_j^2 d_j^2).

    This is Bochner's theorem made numerical: the kernel is the characteristic
    function of the frequency-sampling density, and a Gaussian density has a
    Gaussian characteristic function. Monte-Carlo error is O(1/sqrt(m)), so at
    m = 50k we expect ~0.005 and test at 0.02.
    """
    m = 50_000
    sigma = (0.7, 0.3)
    emb = FourierFeatures(in_dim=2, n_features=m, sigma=sigma, seed=0)

    v = torch.tensor([[0.10, 0.20]])
    others = torch.tensor(
        [[0.10, 0.20], [0.15, 0.20], [0.40, 0.25], [0.90, 0.80], [0.10, 0.95]]
    )
    g_v = emb(v)  # (1, 2m)
    g_o = emb(others)  # (5, 2m)
    empirical = (g_o @ g_v.T).squeeze(1) / m
    exact = emb.kernel(others - v)

    assert torch.allclose(empirical, exact, atol=0.02)
    # sanity: it is a real kernel -- unit diagonal at zero separation
    assert empirical[0] == pytest.approx(1.0, abs=0.02)
    assert exact[0] == pytest.approx(1.0, abs=1e-6)


def test_larger_sigma_narrows_the_kernel():
    d = torch.tensor([[0.05, 0.0]])
    wide = FourierFeatures(in_dim=2, sigma=0.5).kernel(d).item()
    narrow = FourierFeatures(in_dim=2, sigma=8.0).kernel(d).item()
    assert narrow < wide  # bigger sigma -> faster decorrelation with distance
    assert narrow == pytest.approx(math.exp(-2 * math.pi ** 2 * 64 * 0.05 ** 2), rel=1e-5)


# ---------------------------------------------------------------------------
# The residual has to differentiate *through* the embedding
# ---------------------------------------------------------------------------
def test_second_derivative_flows_through_the_embedding():
    set_seed(0)
    model = FourierMLP(in_dim=2, width=32, depth=3, n_features=32, sigma=(5.0, 1.0))
    coords = torch.rand(16, 2, requires_grad=True)
    second = u_xx(model(coords), coords)
    assert second.shape == (16, 1)
    assert torch.isfinite(second).all()
    assert second.abs().sum() > 0


# ---------------------------------------------------------------------------
# (b) The mechanism: high-frequency energy is present *at initialization*
# ---------------------------------------------------------------------------
def sine_energy(model, n_modes=24, n=1024):
    """Energy of ``u(., t=0)`` in the sine basis ``sin(k pi x)``, k = 1..n_modes.

    The right basis to ask this question in is the problem's own: ``sin(k pi x)``
    are the Dirichlet eigenfunctions of the Laplacian on [0, 1], so this is
    literally the decomposition the heat equation diagonalizes in.

    One preprocessing step matters. A raw network output does not vanish at
    x = 0, 1, and a sine series of a function with nonzero endpoints converges
    only like 1/k -- which would show up as spurious "high-frequency content"
    for *any* model, including a straight line. (The same trap in Fourier form:
    an ``np.fft`` of a smooth ramp leaks a fake 1/f^2 tail across every bin,
    because the FFT assumes periodicity.) So subtract the linear interpolant
    between the endpoints first; the remainder vanishes at both ends and its
    sine coefficients decay at the true rate of the function's smoothness.

    Returns (ks, energy) with ``energy[i] = c_k^2``.
    """
    x = np.linspace(0.0, 1.0, n)
    coords = torch.tensor(np.stack([x, np.zeros_like(x)], axis=1), dtype=torch.float32)
    with torch.no_grad():
        u = model(coords).squeeze(1).numpy().astype(float)
    u = u - u[0] - x * (u[-1] - u[0])  # now u(0) = u(1) = 0
    ks = np.arange(1, n_modes + 1)
    c = 2.0 * np.trapezoid(
        u[None, :] * np.sin(ks[:, None] * np.pi * x[None, :]), x, axis=1
    )
    return ks, c ** 2


def _high_freq_fraction(model, k_cut=4):
    ks, e = sine_energy(model)
    return float(e[ks > k_cut].sum() / e.sum())


def test_untrained_fourier_model_carries_high_frequencies_tanh_does_not():
    set_seed(0)
    plain = MLP(in_dim=2, width=64, depth=4, activation="tanh")
    set_seed(0)
    fourier = FourierMLP(in_dim=2, width=64, depth=4, n_features=64, sigma=(5.0, 1.0))

    plain_hf = _high_freq_fraction(plain)
    fourier_hf = _high_freq_fraction(fourier)

    # An untrained tanh MLP on [0, 1] is a nearly-linear, very smooth function
    # of x: measured in the sine basis, ~0.01% of its energy sits above k = 4,
    # and its three strongest modes are k = 1, 2, 3 for every seed tried. The
    # Fourier model starts life already carrying the frequencies the target
    # needs -- that is the entire mechanism, and it is present before a single
    # gradient step.
    ks, e_plain = sine_energy(plain)
    assert set(ks[np.argsort(e_plain)[::-1][:3]]) == {1, 2, 3}
    assert plain_hf < 0.01
    assert fourier_hf > 0.2
    assert fourier_hf > 50 * plain_hf


# ---------------------------------------------------------------------------
# (c) The consequence: same budget, only one of them fits sin(16 pi x)
# ---------------------------------------------------------------------------
def _fit_1d(model, k, steps, lr=1e-3, n=256, seed=0):
    """Plain supervised regression of sin(k pi x) on [0, 1]. Returns rel L2."""
    set_seed(seed)
    x = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    coords = torch.cat([x, torch.zeros_like(x)], dim=1)
    target = torch.sin(k * math.pi * x)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((model(coords) - target) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(coords)
    return float(torch.norm(pred - target) / torch.norm(target))


def test_fourier_fits_a_high_frequency_target_that_the_plain_mlp_cannot():
    """The headline claim, as a unit test: identical width/depth/steps/lr.

    Regression, not a PDE solve, so it isolates the approximation-under-GD
    question from every other PINN difficulty. The PDE version of the same
    experiment is experiments/spectral_bias.py.
    """
    set_seed(0)
    plain = MLP(in_dim=2, width=64, depth=4, activation="tanh")
    set_seed(0)
    fourier = FourierMLP(in_dim=2, width=64, depth=4, n_features=64, sigma=(5.0, 1.0))

    err_plain = _fit_1d(plain, k=16, steps=2000)
    err_fourier = _fit_1d(fourier, k=16, steps=2000)

    assert err_plain > 0.8  # has learned essentially nothing of the target
    assert err_fourier < 0.1  # has essentially fitted it
