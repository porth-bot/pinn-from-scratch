"""Random Fourier-feature input embeddings -- the fix for spectral bias.

Why a plain MLP struggles with high frequencies
-----------------------------------------------
Under gradient descent a wide network's training dynamics are governed by its
neural tangent kernel: with the linearized model, the residual along the NTK
eigendirection with eigenvalue ``lambda_i`` decays like ``(1 - eta lambda_i)^s``
after ``s`` steps.  So components of the target that align with *large*-
eigenvalue eigenfunctions are learned fast, and small-eigenvalue components are
learned slowly -- or, within a finite step budget, not at all.

For a standard tanh/ReLU MLP on a low-dimensional input the NTK eigenfunctions
are (near enough) sinusoids and the eigenvalues *decay with frequency*.  That
single fact is "spectral bias": low frequencies are learned first, high
frequencies last.  ``experiments/spectral_bias.py`` measures exactly this, and
the NTK machinery it appeals to is derived from scratch in the sibling repo
gp-from-scratch (``theory/derivations.md`` Sec. 6-7, "NTK linearized-GD
geometric series"), which builds the same kernel analytically for a two-layer
ReLU net.

The embedding
-------------
Tancik et al. (2020) fix this by not feeding the raw coordinate to the network
at all.  Map the input ``v in R^d`` through a *fixed* (non-trainable) random
Fourier basis first,

    gamma(v) = [ cos(2 pi B v), sin(2 pi B v) ] in R^{2m},
    B in R^{m x d},   B_ij ~ N(0, sigma_j^2),

and train the MLP on ``gamma(v)`` instead of ``v``.  The network no longer has
to *manufacture* high frequencies out of a smooth nonlinearity; they are handed
to it in the first layer, and it only has to learn a linear-ish combination of
them.

Why this works, in one line of algebra.  The embedding's own kernel is

    gamma(v)^T gamma(v') / m
        = (1/m) sum_i [ cos(2 pi b_i.v) cos(2 pi b_i.v')
                        + sin(2 pi b_i.v) sin(2 pi b_i.v') ]
        = (1/m) sum_i cos( 2 pi b_i . (v - v') )
        -> E_b[ cos(2 pi b . d) ]                       as m -> infinity
        = exp( -2 pi^2 sum_j sigma_j^2 d_j^2 ),          d = v - v'

i.e. a *stationary* Gaussian kernel whose bandwidth is set by ``sigma``, by
Bochner's theorem (the characteristic function of the Gaussian sampling density
is the kernel).  ``tests/test_features.py`` checks this convergence numerically.
The consequence is that the composed model's tangent kernel is stationary with
a bandwidth *we choose*: crank ``sigma`` up and the kernel becomes narrow, its
eigenvalues flatten across frequencies, and high-frequency components stop
being suppressed.  Wang, Wang & Perdikaris (2021) make precisely this argument
for PINNs ("On the eigenvector bias of Fourier feature networks").

``sigma`` is per-input-coordinate on purpose.  In a space-time PDE the solution
is typically wiggly in ``x`` and smooth in ``t``; giving ``t`` the same large
bandwidth as ``x`` just injects high-frequency noise into a direction where the
target has none.  So ``sigma`` may be a scalar (shared) or one value per input
dimension.

``B`` is a **buffer, not a parameter**: it is drawn once at construction and
never trained.  That is the standard formulation and it keeps the argument
above honest -- the kernel is fixed, so the eigenvalue story is a statement
about the model we actually train.

Sizing sigma.  The features contain frequencies of order ``sigma_j`` *cycles*
per unit length in coordinate ``j``.  A target ``sin(k pi x)`` on ``[0, 1]`` has
``k / 2`` cycles per unit length, so ``sigma_x`` of order ``k / 2`` is the right
scale.  Too small and the bias is not fixed; too large and the model can
represent -- and the optimizer can chase -- frequencies the solution does not
contain.  Both failure directions show up in the Day-11 sweep.

References
----------
Tancik et al. 2020, "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains".
Wang, Wang & Perdikaris 2021, "On the eigenvector bias of Fourier feature
networks: from regression to solving multi-scale PDEs with PINNs".
Rahaman et al. 2019, "On the Spectral Bias of Neural Networks".
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from pinn.model import MLP


class FourierFeatures(nn.Module):
    """Fixed random Fourier embedding ``v -> [cos(2 pi B v), sin(2 pi B v)]``.

    Parameters
    ----------
    in_dim : int
        Dimension of the raw coordinate (2 for ``(x, t)``).
    n_features : int
        Number of random frequencies ``m``. The output dimension is ``2 m``
        (a cosine and a sine per frequency).
    sigma : float or sequence of float
        Standard deviation of the sampling density, per input coordinate. A
        scalar is broadcast to every coordinate. Sets the bandwidth of the
        induced kernel ``exp(-2 pi^2 sum_j sigma_j^2 d_j^2)``.
    seed : int
        Seed for drawing ``B``. Drawn once, then frozen.
    """

    def __init__(
        self,
        in_dim: int = 2,
        n_features: int = 64,
        sigma: float | Sequence[float] = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        if isinstance(sigma, (int, float)):
            sigmas = [float(sigma)] * in_dim
        else:
            sigmas = [float(s) for s in sigma]
            if len(sigmas) != in_dim:
                raise ValueError(
                    f"sigma has {len(sigmas)} entries but in_dim is {in_dim}"
                )
        if any(s <= 0 for s in sigmas):
            raise ValueError("every sigma must be positive")

        self.in_dim = in_dim
        self.n_features = n_features
        self.sigmas = tuple(sigmas)
        self.out_dim = 2 * n_features

        gen = torch.Generator().manual_seed(seed)
        scale = torch.tensor(sigmas, dtype=torch.float32).reshape(1, in_dim)
        B = torch.randn(n_features, in_dim, generator=gen) * scale
        # A buffer, not a Parameter: B is fixed for the life of the model, so
        # it moves with .to(device) and is saved in the state_dict, but the
        # optimizer never sees it.
        self.register_buffer("B", B)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) -> (N, 2 * n_features).

        Differentiable in ``coords`` (that is the whole point -- the PDE
        residual differentiates through the embedding by the chain rule, which
        is why the composed model can express a high-frequency ``u_xx``).
        """
        proj = 2.0 * math.pi * (coords @ self.B.T)  # (N, m)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)

    def kernel(self, d: torch.Tensor) -> torch.Tensor:
        """The exact ``m -> infinity`` kernel this embedding converges to.

        ``d`` is (N, in_dim) of coordinate differences ``v - v'``; returns (N,)
        of ``exp(-2 pi^2 sum_j sigma_j^2 d_j^2)``. Provided so the tests can
        check the finite-``m`` inner product against the limit rather than
        against a number copied from the paper.
        """
        sig2 = torch.tensor(self.sigmas, dtype=d.dtype).reshape(1, -1) ** 2
        return torch.exp(-2.0 * math.pi ** 2 * torch.sum(sig2 * d ** 2, dim=1))


class FourierMLP(nn.Module):
    """``u_theta = MLP(gamma(v))`` -- Fourier embedding then a plain tanh MLP.

    A thin composition rather than a change to :class:`~pinn.model.MLP`, so the
    plain network the other experiments train stays bit-for-bit what it was and
    the two are directly comparable: same width, same depth, same optimizer,
    the *only* difference being what the first layer sees.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        n_features: int = 64,
        sigma: float | Sequence[float] = 1.0,
        feature_seed: int = 0,
    ):
        super().__init__()
        self.features = FourierFeatures(
            in_dim=in_dim, n_features=n_features, sigma=sigma, seed=feature_seed
        )
        self.net = MLP(
            in_dim=self.features.out_dim,
            out_dim=out_dim,
            width=width,
            depth=depth,
            activation="tanh",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(coords))
