# Derivations

Hand-derived math behind the code, section-numbered so the docstrings can point
here (`experiments/burgers.py` cites Sec. 3, "Day 12"; `pinn/features.py` and
`experiments/spectral_bias.py` cite Sec. 4). Notation follows the source: `u` is
the field, `u_theta` the network approximating it, subscripts are partial
derivatives (`u_t = du/dt`, `u_xx = d^2u/dx^2`), `alpha` is a diffusivity, `nu`
a viscosity, and a PDE is always written as a residual `r = 0`.

Everything here that can be checked is checked numerically somewhere in
`tests/`: the exact solutions satisfy their PDEs by finite differences
(`tests/test_heat.py`, `tests/test_burgers.py`), the autograd derivatives match
finite differences and closed forms (`tests/test_derivatives.py`), and the
Fourier-feature kernel identity of Sec. 4 is checked against its finite-`m`
Monte-Carlo estimate (`tests/test_features.py`).

Contents:

1. The PINN loss: solving a PDE by minimizing a residual
2. The heat equation and its exact Fourier series
3. Burgers' equation via the Cole-Hopf transform (start to finish)
4. Spectral bias through the NTK eigenspectrum (cross-link to gp-from-scratch)
5. Collocation sampling: what is sampled, and why it carries a gradient
6. When classical solvers win (and why this repo is a study, not a claim)

---

## 1. The PINN loss: solving a PDE by minimizing a residual

A PDE with boundary data is a statement about a function. Take the heat
equation on `x in [0, 1]`, `t in [0, 1]`:

```
u_t - alpha u_xx = 0          (interior)
u(x, 0) = u0(x)               (initial condition, IC)
u(0, t) = u(1, t) = 0.        (boundary conditions, BC)
```

A classical solver discretizes `u` on a grid and turns the differential
operator into a matrix. A **physics-informed neural network** instead posits a
smooth parametric function `u_theta(x, t)` — a neural net — and asks it to make
all three lines true *at sampled points*, by minimizing

```
L(theta) = w_r  * (1/N_r)  sum_i  r(x_i, t_i; theta)^2       interior residual
         + w_ic * (1/N_ic) sum_j  ( u_theta(x_j, 0) - u0(x_j) )^2
         + w_bc * (1/N_bc) sum_k  ( u_theta(boundary_k) - g_k )^2,
```

with `r = u_t - alpha u_xx` formed from the network's **exact** autograd
derivatives (`pinn/derivatives.py`), not finite differences of grid values.
This is the whole idea, and three things about it are worth stating precisely.

**Why it is a sensible objective at all.** If `L(theta) = 0` exactly, then the
residual is zero at every sampled interior point and the IC/BC errors are zero
at every sampled boundary point. For a smooth `u_theta` that is a strong
constraint: a continuous residual that vanishes on a dense sample vanishes
everywhere, and by uniqueness of the (well-posed) PDE the network *is* the
solution. In practice `L` is only driven small, on a finite sample, so the
guarantee is approximate — hence every experiment in this repo measures the
error against an independent ground truth rather than trusting the loss.

**Why the derivatives are exact and this is not circular.** `u_theta` is a
closed-form composition of linear maps and `tanh`, so `u_t`, `u_x`, `u_xx` are
themselves closed-form and computed by reverse-mode autograd to machine
precision (Sec. 5 of the module docstring in `pinn/derivatives.py` writes out
the batch-diagonal-Jacobian trick). There is **no truncation error in the
derivative operator** — unlike a finite-difference stencil, whose `O(h^2)` error
is a separate approximation layered on top of the solution error. The only error
a PINN makes is how far `u_theta` is from solving the physics; the derivative of
`u_theta` is exact by construction.

**Why the weights `w_*` matter and are a genuine difficulty.** The three terms
have different units and different natural scales, and gradient descent on their
sum implicitly trades them off. If the residual term dominates, the network can
satisfy the PDE with the *wrong* boundary data (the heat equation has a whole
family of solutions; the IC/BC pick one). If the boundary terms dominate, it
fits the edges and ignores the interior. Adaptive weighting is an active
research topic; this repo keeps `w_*` explicit knobs and each experiment reports
what it used, so the trade-off is visible rather than hidden. The heat and
Burgers solves both use `w_r = w_ic = w_bc = 1` and succeed; the honest reading
is that these problems are forgiving, not that weighting is solved.

---

## 2. The heat equation and its exact Fourier series

The heat solve (`experiments/heat.py`) is deliberately the first one, because it
has a closed-form solution that is exact rather than a truncation — so the
PINN's error can be measured pointwise everywhere.

**Setup.** On `[0, 1]` with homogeneous Dirichlet BCs,

```
u_t = alpha u_xx,   u(0, t) = u(1, t) = 0,   u(x, 0) = u0(x).
```

**The eigenfunctions of the operator.** Separate variables, `u = X(x) T(t)`.
Substituting and dividing by `alpha X T`,

```
T'(t) / (alpha T(t))  =  X''(x) / X(x)  =  -lambda   (a constant),
```

since the left side depends only on `t` and the middle only on `x`. The spatial
problem `X'' + lambda X = 0` with `X(0) = X(1) = 0` is the classic Sturm-Liouville
eigenproblem on `[0, 1]`. Its solutions are

```
X_k(x) = sin(k pi x),   lambda_k = (k pi)^2,   k = 1, 2, 3, ...
```

(the `cos` branch is killed by `X(0) = 0`, and `X(1) = 0` quantizes the
frequency to integer `k`). These `sin(k pi x)` are exactly the Laplacian
eigenfunctions for this domain and these boundary conditions — a fact Sec. 4
leans on again.

**The time factor and the full solution.** With `lambda_k` fixed, the time
equation `T' = -alpha lambda_k T` gives `T_k(t) = exp(-alpha (k pi)^2 t)`.
Superposing, and matching the IC by expanding `u0` in the sine basis
`u0(x) = sum_k a_k sin(k pi x)`,

```
u(x, t) = sum_k a_k sin(k pi x) exp( -alpha (k pi)^2 t ).
```

The experiment picks `u0` to be a *finite* combination of three modes
(`k = 1, 2, 3`, amplitudes `1, 1/2, 1/4`). Because each mode is an
eigenfunction, the heat semigroup only rescales it — the solution never leaves
the span of those three modes, so the sum above is **exact, not a truncated
series**. That is what makes it a clean ground truth. The modes decay at rates
`1 : 4 : 9`, so the high mode is gone by mid-time while the fundamental
lingers: a genuine multi-scale target on which spectral bias (Sec. 4) is
visible.

`tests/test_heat.py` verifies this closed form three ways: it satisfies
`u_t - alpha u_xx = 0` by central finite differences on a fine grid, it matches
the IC at `t = 0`, it vanishes on both spatial boundaries, and each mode decays
at its predicted rate (isolated by projecting onto `sin(k pi x)`).

---

## 3. Burgers' equation via the Cole-Hopf transform (start to finish)

Viscous Burgers (`experiments/burgers.py`) is nonlinear, so it has no
eigenfunction expansion. It has something better: a change of variables that
*linearizes it exactly* into the heat equation. This is the Cole-Hopf transform,
and here is the whole derivation.

**Setup.**

```
u_t + u u_x = nu u_xx,   nu = 0.01 / pi,   u(x, 0) = u0(x),
```

with `u0(x) = -sin(pi x)` in the experiment. The advection term `u u_x` steepens
the profile faster than the tiny `nu` can smear it, so a smooth IC collapses
into a thin viscous shock at `x = 0`.

**Step 1: log-substitution.** Try `u = -2 nu (ln phi)_x = -2 nu phi_x / phi`.
Write `psi = ln phi`, so `u = -2 nu psi_x`. Then

```
u_t   = -2 nu psi_xt
u_x   = -2 nu psi_xx
u u_x = (-2 nu psi_x)(-2 nu psi_xx) = 4 nu^2 psi_x psi_xx
u_xx  = -2 nu psi_xxx.
```

**Step 2: substitute into Burgers.** `u_t + u u_x - nu u_xx = 0` becomes

```
-2 nu psi_xt + 4 nu^2 psi_x psi_xx + 2 nu^2 psi_xxx = 0.
```

Divide by `-2 nu`:

```
psi_xt - 2 nu psi_x psi_xx - nu psi_xxx = 0.
```

**Step 3: recognize a total x-derivative.** Note `2 psi_x psi_xx = (psi_x^2)_x`
and `psi_xxx = (psi_xx)_x` and `psi_xt = (psi_t)_x`, so the whole line is

```
d/dx [ psi_t - nu ( psi_x^2 + psi_xx ) ] = 0,
```

hence `psi_t - nu (psi_x^2 + psi_xx) = C(t)`, a function of `t` alone. `C(t)`
can be absorbed into `phi` (it shifts `psi` by a function of `t`, which does not
change `u = -2 nu psi_x`), so take `C = 0`:

```
psi_t = nu ( psi_x^2 + psi_xx ).
```

**Step 4: undo the log.** With `phi = exp(psi)`: `phi_t = phi psi_t`,
`phi_x = phi psi_x`, and `phi_xx = phi (psi_xx + psi_x^2)`. Therefore

```
nu phi_xx = nu phi (psi_xx + psi_x^2) = phi psi_t = phi_t,
```

so **`phi` solves the heat equation** `phi_t = nu phi_xx`. The nonlinearity is
gone.

**Step 5: transform the initial condition.** `u(x, 0) = -2 nu phi_x(x,0)/phi(x,0)
= u0(x)` means `(ln phi)_x |_{t=0} = -u0/(2 nu)`, so

```
phi(x, 0) = exp( -F(x) / (2 nu) ),   F(x) = integral_0^x u0(s) ds.
```

For `u0 = -sin(pi x)`, `F(x) = (cos(pi x) - 1)/pi`.

**Step 6: solve the heat equation and transform back.** On the whole line the
heat equation has the Gaussian-kernel solution
`phi(x,t) = (4 pi nu t)^{-1/2} integral exp(-(x-y)^2/(4 nu t)) phi(y,0) dy`.
Differentiating and forming `u = -2 nu phi_x/phi`, the normalizing constant and
the `-2 nu` cancel and leave a ratio of integrals:

```
            integral (x - y)/t * exp( -(x-y)^2/(4 nu t) - F(y)/(2 nu) ) dy
u(x, t) = --------------------------------------------------------------------.
            integral           exp( -(x-y)^2/(4 nu t) - F(y)/(2 nu) ) dy
```

**Step 7: evaluate the integrals without a grid.** Substitute
`x - y = sqrt(4 nu t) z`. The Gaussian factor becomes exactly `exp(-z^2)`, the
Gauss-Hermite weight, and `(x - y)/t = sqrt(4 nu / t) z`. So both integrals are
`integral exp(-z^2) g(z) dz` with `g` smooth, and a few hundred Gauss-Hermite
nodes integrate them essentially exactly — no space grid, no Gaussian-tail
truncation. One numerical wrinkle: `phi(y,0) = exp(-F(y)/(2 nu))` reaches
`exp(1/(nu pi)) ~ exp(100)`, so the code works in log-space and subtracts a
per-point maximum from the exponent (it cancels in the numerator/denominator
ratio). The result is the exact Burgers solution to quadrature precision.

`tests/test_burgers.py` checks it satisfies the PDE by finite differences *away
from the shock* (inside the thin shock band the FD stencil cannot resolve the
`u_x ~ -100` gradient, and the FD residual there blows past 1 — documented, not
hidden), matches the IC, and respects the odd symmetry `u(-x,t) = -u(x,t)` that
pins `u = 0` at `x in {-1, 0, 1}`.

---

## 4. Spectral bias through the NTK eigenspectrum

The spectral-bias experiment (`experiments/spectral_bias.py`) is the repo's
honest negative result: a plain PINN learns low-frequency structure first and
high-frequency structure last, or — on a finite step budget — never. The
explanation is the neural tangent kernel, and it connects directly to the NTK
derivation in the sibling repo.

**The linearized-training picture.** Near initialization a wide network's
training dynamics are governed by its NTK, `Theta(x, x') = <grad_theta u_theta(x),
grad_theta u_theta(x')>`. Under gradient descent on a squared loss, the
linearized model's residual `e = u_theta - target` evolves (in the eigenbasis of
the NTK Gram matrix, eigenvalues `lambda_i`, learning rate `eta`) as

```
e_i(s) = (1 - eta lambda_i)^s  e_i(0).
```

Each eigendirection decays geometrically at its own rate: **large-eigenvalue
directions are learned fast, small-eigenvalue directions slowly.** This
`(1 - eta lambda)^s` geometric series is derived from scratch — for a two-layer
ReLU network whose NTK is computed in closed form — in gp-from-scratch,
`theory/derivations.md` Sec. 6-7 ("Arc-cosine kernels" and "NNGP, NTK, and
linearized gradient descent"). This section is the same statement, applied to a
network solving a PDE rather than doing regression.

**Why "high frequency" equals "small eigenvalue."** For an MLP on a
low-dimensional input, the NTK is (near enough) stationary and its eigenfunctions
are essentially sinusoids; its eigenvalues **decay with frequency**. So the
eigendirection carrying a `sin(k pi x)` component has an eigenvalue that shrinks
as `k` grows, its `(1 - eta lambda_k)^s` factor decays more slowly, and the mode
is learned later. "Spectral bias" is exactly this coupling of frequency to
eigenvalue — nothing more mysterious.

**The measurement, and the design choice that makes it honest.** The experiment
solves the heat equation with a single-mode IC `sin(k pi x)` for
`k in {1, 2, 4, 8, 16, 32}`, one PDE per frequency, with the diffusivity scaled
as `alpha_k = alpha_1 / k^2`. That scaling is load-bearing: it cancels the
eigenvalue in the time factor exactly,

```
alpha_k (k pi)^2 = alpha_1 pi^2      for every k,
```

so every target has the *identical* `O(1)` amplitude and time envelope
`exp(-alpha_1 pi^2 t)`, and the only thing varying across the sweep is the
spatial frequency. Without it, a high-`k` solution would decay to ~0 over most
of the domain and a lazy network outputting `u = 0` would score a *small* error
— frequency and amplitude would be confounded and the failure would be
unmeasurable. Measured (seed 0, width 64, depth 4, 8000 Adam steps), steps to
reach 10% relative L2 error grow `200, 400, 800, 1800, 7400, never` for
`k = 1..32`: roughly a doubling per octave, then a wall.

**The fix, as a kernel statement.** A random Fourier-feature embedding
(`pinn/features.py`, Tancik et al. 2020) maps `v -> [cos(2 pi B v), sin(2 pi B v)]`
with `B_ij ~ N(0, sigma_j^2)` drawn once and frozen. Its own kernel is, by a
one-line computation and Bochner's theorem,

```
gamma(v)^T gamma(v') / m  =  (1/m) sum_i cos( 2 pi b_i . (v - v') )
   -> E_b[ cos(2 pi b . d) ]  =  exp( -2 pi^2 sum_j sigma_j^2 d_j^2 ),   d = v - v',
```

a stationary Gaussian kernel whose bandwidth is `sigma` — a bandwidth *we
choose*. Cranking `sigma` up flattens the composed model's tangent-kernel
eigenvalues across frequencies, so high-frequency components stop being
suppressed. `tests/test_features.py` checks the finite-`m` inner product against
this limit numerically. The measured table confirms the trade in both
directions: with `sigma_x = 5` the Fourier model fixes `k = 16, 32` (which the
plain net cannot reach) but is `~23x slower` at `k = 1`, because a bandwidth
tuned for high frequencies hands the optimizer frequencies the low-`k` target
does not contain. `sigma` is fixed *once* for the whole sweep on purpose:
retuning it per frequency would smuggle the answer into the prior.

---

## 5. Collocation sampling: what is sampled, and why it carries a gradient

The loss of Sec. 1 is an average over sampled points. `pinn/losses.py` draws
them, and two design points are worth writing down.

**Interior points carry `requires_grad`.** The residual `r(x_i, t_i)` needs the
network's derivatives *at the collocation points*, so the sampler returns
interior coordinates with `requires_grad_(True)` already set — autograd then
differentiates `u_theta` with respect to those very inputs to form `u_t`,
`u_xx`. IC and BC points do not need this: they enter only through
`u_theta` values, not its derivatives, so they are plain tensors. This is why
the samplers are split (`interior_points`, `initial_points`, `boundary_points`)
rather than one uniform draw.

**Fixed sample vs. resampling.** Every sampler takes an explicit
`torch.Generator`, and the experiments draw the collocation set *once* and reuse
it — reproducible, and cheaper (the interior forward/backward graph structure is
stable). The alternative, resampling every step, is an unbiased estimate of the
continuous residual integral and can help on stiff problems; it is deferred
work. The honest cost of a fixed set is that the network can overfit the sampled
residual between points; the heat convergence sweep measures this directly and
finds the error *saturates* by ~2000 points on the smooth heat target (capacity
binds before point density does), so a fixed 4000-point set is not the
bottleneck here.

**Uniform, not adaptive.** Sampling is uniform on the rectangle. For the
Burgers shock, where 94% of the squared error concentrates in the thin band
`|x| <= 0.1`, a residual-adaptive sampler that puts more points near the shock
is the natural improvement (and a listed roadmap item); the repo measures the
uniform-sampling baseline first so the improvement has something to beat.

---

## 6. When classical solvers win (and why this repo is a study, not a claim)

An honest theory doc has to say what PINNs are *not* good for, and on the exact
problems this repo solves, they are not the tool you would reach for in
production.

**On these problems, classical solvers dominate.** The 1D heat equation is
solved to machine precision in milliseconds by a spectral method (it *is* the
Fourier series of Sec. 2) or by Crank-Nicolson on a modest grid. Viscous
Burgers has the Cole-Hopf ground truth of Sec. 3, computed here by a
few-hundred-node quadrature in a fraction of a second — orders of magnitude
faster and more accurate than the ~30-minute PINN train that reaches `~1%`
relative L2. For low-dimensional, smooth, well-posed forward problems on regular
domains, finite differences / finite elements / spectral methods win on speed,
accuracy, and convergence *guarantees* (they come with provable error orders;
the PINN comes with a nonconvex loss and no such guarantee). This repo's numbers
are reported in that spirit: they show the method *works and how well*, not that
it beats a classical solver, because it does not, here.

**Where PINNs actually earn their place** (none of which this repo claims to
demonstrate, stated so the scope is honest):

- **Inverse problems.** When a coefficient (e.g. `nu`, or a spatially varying
  `alpha(x)`) is unknown and must be recovered from sparse measurements, the PINN
  loss simply adds a data-fit term and optimizes the coefficient alongside the
  weights — no re-meshing, no adjoint solver to hand-derive. This is the setting
  where PINNs are genuinely competitive, and it is a roadmap item, not a result
  here.
- **High dimension.** A grid solver's cost is exponential in the number of
  dimensions; a PINN samples collocation points and sidesteps the mesh, so
  problems in many dimensions (some PDEs from finance or statistical mechanics)
  are reachable where a grid is not.
- **Mesh-free / irregular geometry.** No grid means no meshing of a complicated
  domain; collocation points can be sampled anywhere.

The repo is a **from-scratch study of the method's mechanics and failure
modes** — exact derivatives, exact ground truths, a measured spectral-bias
failure and its kernel explanation — not an argument that PINNs are the right
solver for a 1D heat equation. The README says the same.

---

## References

- M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-Informed Neural
  Networks: A Deep Learning Framework for Solving Forward and Inverse Problems
  Involving Nonlinear Partial Differential Equations," *J. Comput. Phys.* 2019.
  (The PINN loss formulation; the Burgers benchmark.)
- J. D. Cole, "On a quasi-linear parabolic equation occurring in aerodynamics,"
  *Quart. Appl. Math.* 1951; E. Hopf, "The partial differential equation
  u_t + u u_x = mu u_xx," *Comm. Pure Appl. Math.* 1950. (The Cole-Hopf
  linearization of Sec. 3.)
- N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht,
  Y. Bengio, and A. Courville, "On the Spectral Bias of Neural Networks,"
  *ICML* 2019. (Spectral bias, Sec. 4.)
- A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence
  and Generalization in Neural Networks," *NeurIPS* 2018. (The NTK and
  linearized training underlying Sec. 4; derived from scratch in
  gp-from-scratch `theory/derivations.md` Sec. 6-7.)
- M. Tancik, P. Srinivasan, B. Mildenhall, et al., "Fourier Features Let
  Networks Learn High Frequency Functions in Low Dimensional Domains,"
  *NeurIPS* 2020. (The random Fourier embedding, Sec. 4.)
- S. Wang, H. Wang, and P. Perdikaris, "On the eigenvector bias of Fourier
  feature networks: from regression to solving multi-scale PDEs with physics-
  informed neural networks," *Comput. Methods Appl. Mech. Engrg.* 2021. (The
  eigenvalue-flattening argument for PINNs specifically.)
- V. Sitzmann, J. Martel, A. Bergman, D. Lindell, and G. Wetzstein, "Implicit
  Neural Representations with Periodic Activation Functions" (SIREN),
  *NeurIPS* 2020. (The sine-activation option in `pinn/model.py`.)
