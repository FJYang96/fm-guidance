# Methods and Results

This document gives the mathematics behind every method in the repo and the results that have actually been recorded. Where a notebook cell was never run, or its output was not saved, the result is marked **not recorded**.

- [0. Flow-matching base](#0-flow-matching-base)
- [1. Package: per-waypoint CBF for 2D planning](#1-package-per-waypoint-cbf-for-2d-planning)
- [2. `MNIST.ipynb`: first CBF prototype](#2-mnistipynb-first-cbf-prototype)
- [3. `mnist-cbf-clf*.ipynb`: avoid digit 5](#3-mnist-cbf-clfipynb-avoid-digit-5)
- [4. `obstacle-cbf-clf.ipynb`: trajectory-level obstacle avoidance](#4-obstacle-cbf-clfipynb-trajectory-level-obstacle-avoidance)
- [5. Known issues](#5-known-issues)
- [6. Summary of evidence](#6-summary-of-evidence)

---

## 0. Flow-matching base

All models are trained with linear-interpolant conditional flow matching (`torchcfm`, $\sigma = 0$):

```math
x_t = (1-t)\,x_0 + t\,x_1,\qquad x_0\sim\mathcal N(0,I),\ x_1\sim p_{\text{data}},\ t\sim U[0,1]
```

```math
\mathcal L(\theta) = \mathbb E\,\big\|v_\theta(t,x_t) - (x_1 - x_0)\big\|^2 .
```

Sampling integrates $\dot x = v_\theta(t,x)$ from $t=0$ to $t=1$: explicit Euler with $\Delta t = 0.01$ in the notebooks, RK4 on a 32-point grid in the package.

**Identities used below.** For the optimal $v_t(x) = \mathbb E[x_1 - x_0 \mid x_t = x]$:

```math
\mathbb E[x_1\mid x_t=x] = x + (1-t)\,v_t(x),\qquad
\mathbb E[x_0\mid x_t=x] = x - t\,v_t(x),
```

```math
s_t(x) = \nabla_x\log p_t(x) = -\frac{\mathbb E[x_0\mid x_t=x]}{1-t} = \frac{t\,v_t(x) - x}{1-t}.
```

**`visualize_vf.py`** illustrates conditional vs. marginal fields for the path $x = \alpha_t z + \beta_t\varepsilon$ with $\alpha_t = t$, $\beta_t = 1-t$ and binary data $z\in\{0,1\}$:

```math
u_t(x\mid z) = \Big(\dot\alpha_t - \tfrac{\dot\beta_t}{\beta_t}\alpha_t\Big)z + \tfrac{\dot\beta_t}{\beta_t}x = \frac{z-x}{1-t},\qquad
u_t(x) = \sum_z p(z\mid x)\,u_t(x\mid z),\quad p(z\mid x)\propto\mathcal N\big(x;\,tz,\,(1-t)^2\big).
```

*Results:* interactive plots only (`plt.show()`), nothing saved.

---

## 1. Package: per-waypoint CBF for 2D planning

Files: `toyenv.py`, `architecture/`, `training.py`, `sampling/`, `script.py`, `guidancetest.py`, `cbf_test.py`.

**Model.** A conditional field $v_\theta(t,\tau,c)$ over trajectories $\tau\in\mathbb R^{32\times 2}$, conditioned on $c = (\text{start},\text{goal})\in\mathbb R^4$. The architecture is a temporal U-Net adapted from [Diffuser](https://github.com/jannerm/diffuser), or an MLP. Training data are cubic interpolants through 5 noisy points on the start→goal line.

**Control view.** Each waypoint is treated as a single integrator $\dot x = f(x) + g(x)u$ with $f = 0$, $g = I$, and nominal control $u_{\text{nom}} = v_\theta$. A circular obstacle $o$ gives

```math
h_o(x) = \|x - c_o\|^2 - r_o^2,\qquad \nabla h_o(x) = 2(x - c_o).
```

**CBF-QP**, solved at every ODE evaluation and independently for each waypoint (`per_state_barrier=True`):

```math
u^\star = \arg\min_u\ (u-u_{\text{nom}})^\top P\,(u-u_{\text{nom}})
\quad\text{s.t.}\quad L_f h_o + L_g h_o\,u + \alpha(h_o) \ge 0\quad\forall o.
```

With a single obstacle there is a closed form (`CBFQP._solve_closed_form`). Writing $a = L_f h + \alpha(h)$ and $b = (L_g h)^\top$:

```math
u^\star = u_{\text{nom}} + \frac{\max\big(0,\,-(a + b^\top u_{\text{nom}})\big)}{b^\top P^{-1} b}\,P^{-1} b .
```

With several obstacles the QP is solved with CVXPY, looping over the batch. The class-$\mathcal K$ functions are $\alpha(h) = h$ in `cbf_test.py` and $\alpha(h) = 30(e^h - 1)$ in `guidancetest.py`.

**Results: not recorded.** The outputs are GIFs written to `results/`, which is gitignored. No checkpoint is committed, and `training.py` currently crashes (see [Known issues](#5-known-issues)). Note that a per-waypoint guarantee says nothing about the segments between waypoints.

---

## 2. `MNIST.ipynb`: first CBF prototype

**Barrier.** A binary MLP gives a logit $\ell(x)$ for "the image is a 0". It does not take $t$ as input.

- Unconditional generation, avoiding 0: $h = -\ell(x)$.
- Class-0-conditional generation, staying a 0: $h = +\ell(x)$.

**Time rescaling.** The ODE runs over $\tau\in[0,10]$ with $v(\tau,x) = v_\theta(\tau/10,\,x)/10$.

**QP** (CVXPY, per sample):

```math
\min_v\ \|v - v_{\text{nom}}\|^2\quad\text{s.t.}\quad \nabla h^\top v \ge -0.1\,h .
```

**Recorded results.**

| Item | Value |
|---|---|
| Binary "is-0" classifier, test accuracy | 99.73% |
| Class-conditional flow, final loss (10 epochs) | 0.1535 |
| Class-conditional samples | clean 0–9 grid |
| "Avoid 0" (unconditional, 1 sample) | produces a non-zero digit; one sample, so no statistical weight |
| "Stay 0" (class-0-conditional + CBF) | identical to nominal; the constraint never activates |

*Caveats:* the classifier is trained on images in $[0,1]$ but applied to the flow's images in $[-1,1]$, and it is applied to noisy intermediate states it never saw in training.

---

## 3. `mnist-cbf-clf*.ipynb`: avoid digit 5

Goal: sample MNIST unconditionally while never producing a 5. `mnist-cbf-clf.ipynb` is v1; `mnist-cbf-clf-v2.ipynb` extends it.

### 3a. Barrier from a time-dependent classifier

A 10-way CNN $q_\phi(y\mid t,x)$ with logits $z$ is trained with cross-entropy on bridge samples $x_t$, $t\sim U[0,1]$. The barrier is the log-sum-exp margin

```math
h(t,x) = \log\sum_{k\ne 5} e^{z_k(t,x)} - z_5(t,x) - \gamma
= \log\frac{1 - p_5(t,x)}{p_5(t,x)} - \gamma ,
```

i.e. **the log-odds of "not 5"**, so $h\ge 0 \iff p_5 \le 1/(1+e^\gamma)$. That is $p_5\le 0.378$ for $\gamma = 0.5$ and $p_5\le 0.269$ for $\gamma = 1$.

### 3b. CBF step (first-order MM surrogate)

The nonconvex requirement $h(t+\eta,\,x+\eta u)\ge(1-\alpha\eta)\,h(t,x)$ is linearized to

```math
\partial_t h + \nabla_x h^\top u + \alpha h \ge 0 .
```

With $\hat g = \nabla_x h/\|\nabla_x h\|$ and $b = (-\partial_t h - \alpha h)/\|\nabla_x h\|$, the minimum-norm correction is the half-space projection

```math
u^\star = u_{\text{nom}} + \max\big(0,\ b - \hat g^\top u_{\text{nom}}\big)\,\hat g,
\qquad
\|u^\star - u_{\text{nom}}\| = \frac{\max\big(0,\,-(\partial_t h + \nabla h^\top u_{\text{nom}} + \alpha h)\big)}{\|\nabla h\|}.
```

Dividing by $\|\nabla h\|$ does not change the feasible set; it only puts the constraint in unit-normal form. Sampling uses $x_{k+1} = x_k + \Delta t\,u^\star$.

### 3c. Classifier-guidance baseline

```math
u = v_\theta + \lambda\,\nabla_x h,\qquad \lambda\in\{0.5,\,1,\,2,\,5\}.
```

### 3d. CLF variants and the joint QP (v2)

```math
\min_{u,\,\xi}\ \tfrac12\|u - u_{\text{nom}}\|^2 + \tfrac{\rho}{2}\xi^2
\quad\text{s.t.}\quad
\hat g^\top u \ge b_h,\qquad
\nabla V^\top u \le -c\,V + \xi .
```

- **`relu`**: $V = \tfrac12\,\mathrm{relu}(-h)^2$, so $\nabla V = -\mathrm{relu}(-h)\,\nabla h$. For $h<0$ and $\xi = 0$ the constraint reduces exactly to $\nabla h^\top u \ge -\tfrac{c}{2}h$, which is a CBF with $\alpha' = c/2$. With the default $c = 2$, $\alpha' = 1 = \alpha$, so it is **identical to the CBF** apart from the $\partial_t h$ term. For $h\ge0$ it is vacuous.
- **`proto`**: $V = \tfrac12\|\phi(t,x) - \mu_{y^\star}\|^2$. Here $\phi$ is the classifier's 256-d penultimate feature, $y^\star = \arg\max_{k\ne5} q_\phi(k\mid t,x)$, and $\mu_k$ is the mean feature of 500 real class-$k$ images at $t=1$. $\nabla V$ is computed with autograd.
- **`loglike`**: along a controlled path, the continuity equation gives
  ```math
  \frac{d}{dt}\log p_t(x_t) = -\nabla\!\cdot v_t(x_t) + s_t(x_t)^\top\big(u - v_t(x_t)\big),
  ```
  so the intended likelihood-preserving constraint is $s_t^\top u \ge s_t^\top v_t - \varepsilon - \xi$, with $s_t$ from the identity in §0.
  **The implementation encodes $V = -s_t^\top v_t + \varepsilon$, $\nabla V = -s_t$, which enforces $s_t^\top u \ge -c\,s_t^\top v_t + c\,\varepsilon - \xi$ instead (sign of $s_t^\top v_t$ flipped).** The correct encoding is $V = s_t^\top v_t - \varepsilon$ with $c = 1$.

**Closed-form solver.** Four candidate active sets (none / CBF / CLF / both). In the "both" case, with $a_V = \nabla V$ and $b_V = -cV$:

```math
\begin{bmatrix} 1 & -\hat g^\top a_V \\ \hat g^\top a_V & -(\|a_V\|^2 + 1/\rho)\end{bmatrix}
\begin{bmatrix}\lambda\\ \mu\end{bmatrix}
=
\begin{bmatrix} b_h - \hat g^\top u_{\text{nom}} \\ b_V - a_V^\top u_{\text{nom}}\end{bmatrix},
\qquad u = u_{\text{nom}} + \lambda\hat g - \mu\,a_V,\quad \xi = \mu/\rho .
```

The solver keeps the cheapest candidate that passes its feasibility check. This is exact when only the CBF is on, but approximate with the CLF on (see [Known issues](#5-known-issues)).

### 3e. Geometric barriers (v2, no trained classifier)

Both use soft-min squared pixel distances to a bank $\mathcal R_k$ of 200 real images per class.

**Tweedie one-step** ($\beta = 20$):

```math
m_t(x) = x + (1-t)\,v_\theta(t,x)\ \ (v\text{ detached}),\qquad
D_k = -\tfrac1\beta\log\!\sum_{x_i\in\mathcal R_k}\! e^{-\beta\|m_t(x) - x_i\|^2},\qquad
h = D_5 - D_{\neg5} - \gamma .
```

**Lookahead.**

- Endpoint $\hat x_1 = \Phi^\theta_{t\to1}(x)$ from 5 Euler steps, with $v$ detached.
- Bank-size-normalized soft-min: $D_k = -\tfrac1\beta\big[\log\sum_i e^{-\beta d_i^2} - \log|\mathcal R_k|\big]$, and likewise $D_{\neg5}$ over the 9 safe classes with $\log 9$.
- $\beta = 1/\mathrm{median}$ within-class nearest-neighbour $d^2$.
- A late-activation gate: $h = \sigma\big(10(t - 0.6)\big)\,\big(D_5 - D_{\neg5} - \gamma\big)$.

### 3f. Evaluation metrics (v2)

A separate reference CNN labels generated samples.

- $P(5)$: fraction labelled 5.
- $\mathrm{KL}_{\neg5} = \sum_{k\ne5}\hat p_k\log(9\hat p_k)$: divergence from uniform over the 9 safe classes.
- Feature-FID on 128-d reference features against 5000 real non-5 images:
  ```math
  \|\mu_g-\mu_r\|^2 + \mathrm{Tr}\big(\Sigma_g + \Sigma_r - 2(\Sigma_g\Sigma_r)^{1/2}\big).
  ```
- Also: confidence, entropy, and pixel-statistics drift.

### 3g. Recorded results (v2)

v1 has no saved outputs beyond download and training progress bars.

| Item | Value |
|---|---|
| 2D toy: points with $h<0$ | 12% (the print labels this "violating CBF", but it measures $h<0$) |
| Time-dependent classifier, CE loss (5 epochs) | 0.798 → 0.620 → 0.602 → 0.581 → 0.582 |
| 10-way accuracy at $t=1$ | **98.91%** |
| AUROC for detecting 5 at $t=1$ | **0.9999** |
| $h(1,x)$ histogram | 5s centred near −15 (range −27 to 0); non-5s near +25 (range 0 to 60); almost no overlap |
| Flow model | `torchcfm` UNet, 32 channels, 3 epochs, ~85 s/epoch on CUDA |
| Reference classifier accuracy | 98.83% |
| Real non-5 pixel statistics | mean −0.739, std 0.617, active fraction 0.156 |
| MM surrogate residual $h_{\text{true}} - h_{\text{lin}}$ (50 nominal steps) | mean **0.0016**, max $\lvert\cdot\rvert$ **0.0155**, against $h$ values in the tens |

**$h$-trace** (trained barrier, $\alpha = 1$, $\gamma = 0.5$, CBF only):

- **Nominal:** $\min h$ over 32 samples crosses 0 at about step 18, bottoms out near −4.5 at step 50, and ends near −3.8. At least one nominal sample ends classified as a 5 by $q_\phi$.
- **CBF:** $\min h$ over 500 samples stays positive throughout, decreasing smoothly from 1.4 to 0.65.
- **Correction norm** $\|u^\star - v\|$ (784-d): about 2.1–2.5 for the first 5 steps, about 0.3 by step 20, about 0 after step 80. Nearly all steering happens early, while $x$ is still mostly noise.
- The "CBF active" panel is identically 0 because it counts $h<0$, not active constraints. The nominal and CBF traces use different batch sizes (32 vs 500).

**Not recorded:** the main comparison table ($P(5)$, FID, KL, correction norm per method), the classifier-guidance sweep, the Pareto plots, the proto/loglike runs, the geometric-barrier histograms, and the kNN-accuracy-vs-$t$ diagnostic.

---

## 4. `obstacle-cbf-clf.ipynb`: trajectory-level obstacle avoidance

**Data.** Cubic Bézier curves with $T = 32$ waypoints:

```math
b(s) = (1-s)^3p_0 + 3(1-s)^2s\,p_1 + 3(1-s)s^2\,p_2 + s^3p_3,
```

with $p_0 = (-3,0)$, $p_3 = (3,0)$, $p_1\sim U([-2,0]\times[-2.5,2.5])$, $p_2\sim U([0,2]\times[-2.5,2.5])$, and $\mathcal N(0, 0.15^2)$ noise on each waypoint. No obstacles are seen at training time.

**Model.** An unconditional MLP $v_\theta(t,\tau)$ on $\mathbb R^{64}$, trained for 200 epochs.

**Barrier.** A soft-min over all waypoints and obstacles. Two discs at $(0,\pm0.8)$ with $r = 0.7$; margin $m = 0.1$; $\beta = 8$:

```math
h(\tau) = -\tfrac1\beta\log\sum_{t,o}\exp\!\Big(-\beta\big(\|z_t - c_o\|^2 - (r_o+m)^2\big)\Big),
\qquad
\min_{t,o}\text{slack} - \tfrac{\log(TO)}{\beta}\ \le\ h\ \le\ \min_{t,o}\text{slack}.
```

So $h\ge0$ certifies no collision, and the soft-min is conservative by at most $\log 64/8\approx0.52$ in squared-distance units.

**Methods.** The same QP as §3d with $\partial_t h = 0$:

- CBF with $\alpha = 1$ and $\alpha = 3$;
- CBF + relu-CLF ($c = 2$, $\rho = 10$);
- CBF + loglike-CLF ($\alpha = 2$, $c = 1$, $\rho = 1$, $\varepsilon = 0.5$), which has the same sign bug;
- guidance $u = v + \lambda\nabla h$, $\lambda\in\{0.5,1,2,5\}$.

**Moving obstacle.** $c(t) = (0,\,-1 + 2t)$, $r = 0.7$, $\alpha = 3$, with constraint $\hat g^\top u \ge (-\partial_t h - \alpha h)/\|\nabla h\|$, where $\partial_t h = \nabla_c h\cdot\dot c$. Note that $t$ here is generation time, not physical time along the trajectory.

**Metrics.** Hard collision rate; goal error $\|z_T - \text{goal}\|$; start error $\|z_1 - \text{start}\|$; path length $\sum_t\|z_{t+1} - z_t\|$; smoothness $\sum_t\|z_{t+1} - 2z_t + z_{t-1}\|^2$; mean correction norm.

**Results: not recorded. The notebook has not been executed.** An independent check: with these obstacles, 100% of initial noise trajectories $\tau_0\sim\mathcal N(0,I)$ already have a waypoint inside an obstacle. Starting from $h(0)<0$, the CBF only guarantees $h(1)\ge e^{-\alpha}h(0)$ (≈ $0.37\,h(0)$ for $\alpha = 1$, ≈ $0.05\,h(0)$ for $\alpha = 3$). That is recovery toward safety, not a certificate of ending safe.

---

## 5. Known issues

Ordered by severity. None are fixed on this branch; it is documentation only.

1. **loglike-CLF sign error** (v2 `cbf_signals_batched`, obstacle `cbf_signals`). Enforces $s^\top u \ge -c\,s^\top v + c\varepsilon$ rather than $s^\top u \ge s^\top v - \varepsilon$. Numerically, $u = v_{\text{nom}}$ satisfies the intended constraint always but the encoded one only ~48% of the time. Fix: $V = s^\top v - \varepsilon$, $c = 1$.
2. **4-candidate QP is approximate with the CLF on.**
   - Candidate costs omit $\tfrac\rho2\xi^2$.
   - Candidates 0/1 are marked infeasible when the soft CLF is violated, although a soft constraint is always satisfiable.
   - Candidate 3 clips $\mu\ge0$ without re-solving for $\lambda$ but is always marked feasible, so it can violate the CBF.
   - CBF-only mode is exact.
3. **`training.py` calls `backward()` twice** on the same graph, so it crashes on the first batch.
4. **`sampling/util.py` (`CBF.jac_h`) uses autograd while `Sampler.sample` runs under `torch.no_grad()`.** It crashes unless explicit Jacobians are passed.
5. **API drift.**
   - `script.py` calls `visualize_environment` without the required `ax`.
   - `toyenv.test_environment` calls a nonexistent `add_obstacle`.
   - `_solve_closed_form` divides by zero when $\nabla h = 0$.
6. **Obstacle setup** starts every sample inside the unsafe set (see §4).
7. **Minor.**
   - The sinusoidal time embedding sees $t\in[0,1]$ with frequencies ≤ 1, giving weak time conditioning.
   - In `MNIST.ipynb`, the classifier is trained on $[0,1]$ images but applied to $[-1,1]$ flow images.

---

## 6. Summary of evidence

The recorded outputs support three claims:

1. The trained "not-5" barrier is near-perfect at $t=1$ (AUROC 0.9999).
2. The first-order step surrogate is accurate at $\Delta t = 0.01$.
3. The CBF keeps the classifier's $h$ positive along the sampling path while nominal sampling goes negative, with most of the correction in the first ~20% of steps.

Still unestablished: whether CBF samples actually avoid 5s according to the independent reference classifier, at what image quality, and how that compares to classifier guidance. Those comparison cells were not run or their outputs were not saved. The obstacle experiments have not been run at all.
