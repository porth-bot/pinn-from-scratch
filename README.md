# pinn-from-scratch

![ci](https://github.com/porth-bot/pinn-from-scratch/actions/workflows/ci.yml/badge.svg)

Physics-informed neural networks (PINNs) built from the derivatives up, in
PyTorch. The network is a plain MLP; the PDE residual is formed from **exact
autograd derivatives** of that network — `u_t`, `u_x`, `u_xx` written out by
hand in [`pinn/derivatives.py`](pinn/derivatives.py) — and training minimizes a
weighted sum of the residual and the initial/boundary conditions. Every problem
this repo solves comes with a **closed-form or independently computed ground
truth**, so the PINN's error is always *measured against truth*, never asserted.

> **Status: work in progress.** This is the scaffold and verified core
> (network, autograd derivatives, loss terms, collocation samplers, and their
> tests). The worked problems — the heat equation vs its exact Fourier series,
> Burgers' via Cole-Hopf, and a documented spectral-bias failure mode — land
> over the following sessions, along with the house-style writeup and figures.

## What a PINN is

A finite-difference or spectral solver stores `u` on a grid and approximates
`u_xx` by differencing neighbours. A PINN instead represents the solution as a
smooth function `u_theta(x, t)` — a neural network — and asks automatic
differentiation for its *exact* derivatives at arbitrary points. The PDE,
written as a residual `r(x, t) = 0` (for the heat equation, `r = u_t - alpha
u_xx`), becomes a loss:

```
L(theta) = w_r · mean r(x_i, t_i)²      (PDE residual at interior points)
         + w_ic · mean (u(x, t0) - u0)²  (initial condition)
         + w_bc · mean (u(boundary) - g)² (boundary condition)
```

There is no grid and no truncation error in the derivatives themselves; the
only error is how well `u_theta` satisfies the physics. The catch is that the
minimization is nonconvex and the loss terms compete — the honest parts of this
repo are about *when this works and when it does not*.

## Core modules

| module | what it does |
|---|---|
| [`pinn/model.py`](pinn/model.py) | `MLP` — tanh (default) or SIREN-style sine activations, Xavier / SIREN init, linear output head. `set_seed` for reproducibility. |
| [`pinn/derivatives.py`](pinn/derivatives.py) | `u_x`, `u_t`, `u_xx`, `u_tt`, `laplacian` via `torch.autograd.grad` with `create_graph=True` (so derivatives are themselves differentiable). The batch-diagonal-Jacobian trick is written out in the module docstring. |
| [`pinn/losses.py`](pinn/losses.py) | `residual_loss`, `data_loss`, and uniform collocation samplers (`interior_points`, `initial_points`, `boundary_points`), each taking an explicit `torch.Generator`. |

## Tests

The derivative helpers are the foundation, so they are checked two ways —
against central finite differences *and* against hand-derived closed forms — on
`u = sin(a x) exp(-b t)`, whose every derivative is known. There are also
determinism tests (same seed → identical weights and outputs) and a check that
the tanh network has a nonzero `u_xx` (a ReLU network's is identically zero and
cannot express a diffusion residual at all).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

## Provenance

This is an **AI-assisted** study resource: the implementation was written with
Claude (Anthropic) as a from-scratch reference for learning how PINNs work,
with honest attribution. The math is checked against closed-form ground truth
in the tests rather than taken on faith. Commits carry a `Co-Authored-By`
trailer. Part of a from-scratch series alongside
[gp-from-scratch](https://github.com/porth-bot/gp-from-scratch),
[mcmc-from-scratch](https://github.com/porth-bot/mcmc-from-scratch), and
[grokking-transformer](https://github.com/porth-bot/grokking-transformer).

## License

MIT — see [LICENSE](LICENSE).
