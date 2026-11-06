# fm-guidance

Safety-filtered sampling from flow-matching models. A pretrained flow's velocity field is treated as a nominal controller, and at every sampling step a control barrier function (CBF) quadratic program applies the minimum correction that keeps samples out of a forbidden region. Optional control-Lyapunov-style (CLF) terms aim to keep the corrected samples close to the model's distribution.

The mathematics of every method, the recorded results, and a list of known issues are in **[docs/METHODS.md](docs/METHODS.md)**.

## Experiments

| Where | Task | Barrier | Status |
|---|---|---|---|
| `script.py`, `guidancetest.py`, `sampling/` | Conditional 2D start→goal trajectories, circular obstacles | Per-waypoint ‖x − c‖² − r², CVXPY / closed-form QP | Code only; `training.py` currently crashes |
| `cbf_test.py` | CBF filter on a constant vector field | Same | Produces a GIF in `results/` |
| `MNIST.ipynb` | Avoid / keep digit 0 | Binary classifier logit | Single qualitative samples |
| `mnist-cbf-clf.ipynb` (v1) | Unconditional MNIST, never generate a 5 | 10-way time-dependent classifier, LSE margin | No saved outputs |
| `mnist-cbf-clf-v2.ipynb` (v2) | Same, plus CLF variants and geometric (kNN) barriers | Classifier or soft-kNN in endpoint space | Barrier and diagnostic results recorded; comparison table not recorded |
| `obstacle-cbf-clf.ipynb` | Unconditional 64-d trajectories, static and moving obstacles | Soft-min over waypoints × obstacles | Not yet executed |
| `visualize_vf.py` | Conditional vs. marginal vector fields (teaching plot) | — | — |

## Recorded results so far

From `mnist-cbf-clf-v2.ipynb`:

- The time-dependent classifier reaches 98.91% 10-way accuracy and 0.9999 AUROC for detecting 5s at $t=1$. The barrier $h(1,x)$ separates 5s (around −15) from non-5s (around +25) with almost no overlap.
- Along nominal sampling, the minimum $h$ across samples goes negative by about step 18 and ends near −3.8. With the CBF it stays positive (1.4 → 0.65).
- The CBF correction norm is largest in the first ~20% of steps (≈2.5) and decays to ≈0 by step 80.
- The first-order step surrogate is accurate: the mean residual is 0.0016 and the max is 0.0155.

The headline comparison against classifier guidance (P(5), feature-FID, Pareto plots) has not been recorded yet. See [docs/METHODS.md §6](docs/METHODS.md#6-summary-of-evidence).

## Layout

```
architecture/   temporal U-Net (adapted from Diffuser), MLP, time/condition embeddings
sampling/       Sampler base (torchdiffeq), NominalSampler, CBFGuidedSampler, CBF/CBFQP utilities
toyenv.py       2D environment, trajectory generation, circular barriers
training.py     conditional flow-matching training loop
docs/           METHODS.md: math, results, known issues
```

## Setup

```bash
pip install -r requirement.txt
python cbf_test.py          # CBF on a constant field -> results/vectorfield.gif (create results/ first)
```

The notebooks are self-contained and install their own dependencies (`torch`, `torchcfm`, `torchvision`, `cvxpy`, `tqdm`). The MNIST notebooks expect a GPU.
