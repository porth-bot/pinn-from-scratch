"""pinn-from-scratch: physics-informed neural networks in PyTorch.

A study repo that builds PINNs from the derivatives up: the network is a
plain MLP, the PDE residual is formed from *exact* autograd derivatives of
that network (u_t, u_x, u_xx written out in :mod:`pinn.derivatives`), and the
loss is a weighted sum of residual + initial/boundary terms
(:mod:`pinn.losses`). Every problem the repo solves has a closed-form or
independently computed ground truth, so the PINN's error is always measured
against truth rather than asserted.

The core is intentionally small and readable; the experiments (heat equation
vs the exact Fourier series, Burgers' via Cole-Hopf, the spectral-bias
failure mode) live in ``experiments/`` and write their figures from committed
logs.
"""

from pinn import derivatives, features, losses  # noqa: F401
from pinn.features import FourierFeatures, FourierMLP  # noqa: F401
from pinn.model import MLP, set_seed  # noqa: F401

__all__ = [
    "MLP",
    "FourierFeatures",
    "FourierMLP",
    "set_seed",
    "derivatives",
    "features",
    "losses",
]
