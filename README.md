# Trajectory-Based Bayesian Discovery of a Host-Pathogen Model from Quasi-Steady-State Observations

This repository reproduces the results and the figure of the paper entitled

```Trajectory-Based Bayesian Discovery of a Host-Pathogen Model from Quasi-Steady-State Observations by Reymart Salcedo Lagunero, Andreas Langer, Cordula Reisch```

**arXiv link:** [https://doi.org/10.48550/arXiv.2611.18815](https://doi.org/10.48550/arXiv.2611.18815)

The paper recovers the mechanisms of a bistable host–pathogen model from noise-free data on its nearly settled window [10, 20], with the initial state known.

```
du/dt = -1.0 u^3 + 1.1 u^2 - 0.2 u - 0.2 u v,   u(0) = 1.0
dv/dt =  0.5 u - 1.0 v + 0.5 u v,               v(0) = 0.5
```

| file | what it does |
|---|---|
| `pipeline.py` | the method: Stages 1–3, and the exhaustive search (`--exhaustive`) |
| `plot.py` | the paper's figure and the solution panels of the final models |
| `baselines.py` | the SINDy comparison of the paper: five sparse-regression methods on the window and x0 |
| `hpc.sbatch` | every step below in one cluster job |
| `environment.yml`, `requirements.txt` | the packages, for conda and for pip |

## The method

- **Stage 1.** Adam fits the PINN loss over the two state networks and all 20 coefficients.
- **Stage 2.**
  1. Sparse Bayesian learning (SBL) re-estimates the coefficients from the networks' derivatives.
  2. ℓ0-IHT-HTP finds a support of size k (or of every k from 19 to 2).
  3. The two-swap climb moves to the best support within two swaps while the evidence rises. Each support is
     scored by the Laplace evidence of its ODE solution, refitted to the observations from least squares on the
     Stage 1 networks.
  4. The selection: the climb's end, or with k unknown the end with the highest evidence.
  5. The ODE refit of the selected support, from least squares on the current or the Stage 1 networks, whichever
     has the higher evidence.
- **Stage 3.** The networks are refitted with the support and the coefficients fixed.

The exhaustive search scores every support of one size with the same refit and evidence.

## The paper's settings

All of them are constants at the top of `pipeline.py`. The command line sets only σ, k and the number of scoring processes.

**Problem and data**

| setting | value | meaning |
|---|---|---|
| parameters | (a1, a2, a3, a4, b1, b2, b3) = (1.0, 1.1, 0.2, 0.2, 0.5, 1.0, 0.5) | the true model |
| initial state | x0 = (1.0, 0.5) at t = 0 | known, and imposed exactly on every candidate ODE solution |
| time span | [0, 20] | where the PINN's ODE residual is enforced and the candidates are integrated |
| observations | 100 evenly spaced times on [10, 20], noise-free | the quasi-steady-state window; x0 is not one of them |
| data solver | LSODA, rtol 1e-10, atol 1e-12 | how the true solution is computed |
| library | the 10 monomials of degree ≤ 3 in u and v, in each equation | 20 mechanisms, 7 of them true |
| column scales | s_m = rms of monomial m over the observations (+1e-6) | the coefficients are fitted as θ = s_m ξ, so every column is O(1) |

**The PINN (Stages 1 and 3)**

| setting | value | meaning |
|---|---|---|
| networks | one per state; 4 tanh layers of 64 neurons; float64 | x̂(t; w) |
| input | τ = t/10 − 1 and the Fourier features sin(kπτ), cos(kπτ), k = 1, 2 | time scaled to [−1, 1]; the features give the first layer smooth functions of the whole span |
| output | softplus + 1e-8 | the states stay positive |
| initialisation | Xavier normal weights, zero biases, θ = 0, seed 42 | the same draw on every machine |
| collocation points | N_c = 1,000, evenly on [0, 20] | where the ODE residual is evaluated |
| loss weights | ω_data, ω_ode, ω_ic = 100, 10, 1 | the PINN loss; x0 enters it only as the penalty ω_ic ‖x̂(0) − x0‖² |
| Adam | learning rate 1e-3 (θ: 1e-2); warm-up over 25 epochs; gradient norms clipped at 1 | every Adam fit keeps its best state |
| Stage 1 | 15,000 epochs, networks and all of θ | |
| Stage 3 | Adam 5,000 epochs at 5e-4, then at most 300 L-BFGS steps (strong Wolfe) | the networks only; L-BFGS stops after 10 steps of relative change below 1e-12 |

**Stage 2**

| setting | value | meaning |
|---|---|---|
| SBL | MacKay's updates per equation; α = 1, β = 1/var(b) to start; at most 500 iterations, stopping when no log α changes by 1e-6; a weight whose α exceeds 1e8 is set to 0 | every mechanism stays active; SBL only sets the ranking |
| ℓ0-IHT-HTP | at most 15 alternations, stopping when the support is unchanged for 3 | |
| w-step | Adam, 200 epochs, θ fixed | the networks follow the current coefficients |
| θ-step | at most 20 gradient steps on ω_ode L_ode, step 0.9/L with L = 2 ω_ode ‖A‖²/N_c | each keeps the k largest \|θ\| (at least one per equation) and refits them by least squares |
| after ℓ0-IHT-HTP | Adam, 1,000 epochs, networks and θ, support fixed | |
| sizes | k = 7, or k = 19, …, 2 with k unknown | |
| climb | every support within one or two swaps that keeps a mechanism in each equation: at most 1,729 at k = 7 (91 one-swaps + 1,638 two-swaps; 1,719 or 1,689 when one equation holds only 2 or 1 terms; 1,729 at the truth) | moves to the best while log Z rises by more than 1e-6 |
| ODE refit (step 5) | from least squares on the ℓ0-IHT-HTP networks and on the Stage 1 networks | the higher log Z is kept |

**The evidence**

| setting | value | meaning |
|---|---|---|
| σ | 2e-7 (1e-5 for the comparison) | it sets how strongly a misfit is penalised |
| prior | c ~ N(0, τ² I), τ = 2 | weakly informative for coefficients of order one |
| log Z | −‖r‖²/2σ² − ‖c‖²/2τ² − k log τ − ½ log det(JᵀJ/σ² + I/τ²) | the Laplace approximation with the Gauss–Newton Hessian; the constant −n/2 log(2πσ²) is dropped |
| MAP fit | Levenberg–Marquardt; tolerances 1e-12; at most 100 evaluations per fit | |
| repeated fits | the fit restarts from its own result while log Z rises by more than 1e-3, at most 5 fits | in the climb and the exhaustive search only while log Z > −1e4 |
| sensitivities | forward sensitivity equations, LSODA, rtol 1e-8, atol 1e-10 | J = ∂X/∂c exactly, up to the solver tolerance |
| blow-up | a solution leaving \|x\| < 1e3 (1 + max of the data) fails | log Z = −∞ |
| time per support | 60 s in the climb, 3,600 s in the exhaustive search | a support still being refitted keeps the best score so far |

## Install

Two ways; both give the same packages. From this folder:

**With conda** (or mamba), everything from conda-forge:

```
conda env create -f environment.yml
conda activate sparse-truth
```

**With pip**, in a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # optional: the small CPU-only PyTorch
pip install -r requirements.txt
```

Both need Python ≥ 3.11 (conda installs it; pip uses the `python3` you have) and install numpy ≥ 2.0, scipy, PyTorch (the CPU build is enough), matplotlib (for `plot.py`), and PySINDy 2.1.0 and numpyro with jax (for `baselines.py` only). Without PySINDy, `pipeline.py` and `plot.py` still run; without numpyro, `baselines.py` skips sparse Bayesian regression. The paper's runs used Python 3.11–3.13, numpy 2.4, scipy 1.17 and torch 2.12–2.14; the baselines PySINDy 2.1.0, scikit-learn 1.8–1.9 and numpyro 0.22. On Windows the per-support time limit is not applied (it needs `SIGALRM`).

## Run, one step at a time

Run every command from this folder, with the environment active. `--workers` is the number of processes that score supports; use the number of cores minus one. Every step writes into `results/`.

**The paper's Stage 1 networks come with this folder,** as `results/stage1.pt`. They were trained by HPC job
12514, the paper's run. Every step loads them instead of training, so all results come from that one Stage 1, as in the paper.

- **To train your own Stage 1 instead,** delete the whole `results` folder before step 1: step 1 then trains it
  (15,000 epochs, about 8 minutes on 4 threads) and saves it there. A fresh training gives other networks on most machines.
- **Scores are saved as they come,** so a step that is interrupted resumes when it is run again. They depend on the Stage 1 networks: never replace `stage1.pt` without deleting the scores.

**1. Size known, k = 7.** About 5 minutes for SBL, ℓ0-IHT-HTP and Stage 3 on 4 threads, plus the climb, about 13 CPU-hours.

```
python pipeline.py --sigma 2e-7 --k 7 --workers 8
```

Expected output (an excerpt):

```
Stage 1 loaded from results/stage1.pt (PINN loss 3.504868e-06)
  true mechanisms among the 7 largest coefficients: 0
  (1) SBL: true mechanisms among the 7 largest coefficients: 3
  k = 7: (2) l0-IHT-HTP: 4/3; (3) the climb:
    start 4/3, log Z -754791368880.78
    -> 5/2, log Z -438.86
    -> 7/0, log Z -94.72
  (5) ODE refit: log Z -94.72, misfit 2.9e-11
  du/dt = -0.2000008236 u +1.1000023651 u^2 -0.2000000916 u*v -1.0000015797 u^3
  dv/dt = +0.5000000149 u -1.0000000936 v +0.5000001147 u*v
  relative coefficient error 1.5e-06; the largest deviation from the true coefficients, 2.4e-06, is in u^2 -> du/dt
```

`a/b` is a support with a true and b spurious mechanisms. The truth is used only for these labels.

**2. The exhaustive search at k = 7.** 77,280 supports, about 230 CPU-hours.

```
python pipeline.py --sigma 2e-7 --k 7 --exhaustive --workers 63
```

While it runs it prints its progress every 100 supports (`--progress 500`, `--progress 1000`, …: every N), for example

```
  0 of 77,280 supports already scored in results/scores_sigma2e-07_stage1_exhaustive.csv; scoring 77,280
  100 of 77,280 scored (0.1%), 19 s, about 4.1 h left
  200 of 77,280 scored (0.3%), 37 s, about 4.0 h left
```

(a resumed run first counts the supports already scored). At the end it prints the ten best supports. For the four best it also gives the misfit δ and the largest deviation from the
true solution on [0, 20], ‖Δx‖∞ (Table 1 of the paper):

```
   1. 7/0  log Z    -94.72 = -fit   0.00 - prior and Occam  94.72  misfit 2.9e-11  |dx|_inf 2.7e-08
   2. 2/5  log Z    -97.34 = -fit   4.70 - prior and Occam  92.64  misfit 4.3e-08  |dx|_inf 8.3e-02
   3. 2/5  log Z   -104.01 = -fit  11.44 - prior and Occam  92.56  misfit 6.8e-08  |dx|_inf 8.4e-02
   4. 6/1  log Z   -104.59 = -fit   8.50 - prior and Occam  96.09  misfit 5.8e-08  |dx|_inf 8.2e-04
  the true model ranks 1 (log Z -94.72); it is 2.62 above the runner-up 2/5, a Bayes factor of 13.7
```

**3. The figure.**

```
python plot.py
```

- `results/fig_rivals_half.pdf` is the paper's figure: the four best supports of step 2, each refitted and
  integrated from x0, over the true solution.
- `results/solution_sigma2e-07_k7.pdf` shows the final model's networks over the data and the true solution.
- `results/solution_sigma2e-07_k7_with_ranks.pdf` adds the two best other supports of the run's climb.

**4. Size unknown.** ℓ0-IHT-HTP and the climb at every k = 19, …, 2, about 500 CPU-hours.

```
python pipeline.py --sigma 2e-7 --workers 63
```

The climbs' ends are printed per size. The true model has the highest evidence. The best other size is k = 9 (the 7 true mechanisms and 2 spurious, −102.47), and the evidence falls to −117.23 at k = 19; at k = 6 the climb ends at −2,449.92.

**5. The exhaustive search from exact derivatives** (the diagnostic check).

```
python pipeline.py --sigma 2e-7 --k 7 --exhaustive --exact-start --workers 63
```

The true model is again first, 2.62 above the same 2/5 support.

**6. The exhaustive search at σ = 1e-5.**

```
python pipeline.py --sigma 1e-5 --k 7 --exhaustive --workers 63
```

The true model ranks 2,287th, and the best support, a 1/6, is 5.85 above it.

**7. Sizes 6, 8, 9 and 10, then the posterior.** 594,194 supports with k = 7, about 1,800 CPU-hours.

```
for k in 6 8 9 10 7; do python pipeline.py --sigma 2e-7 --k $k --exhaustive --workers 63; done
```

No support of sizes 6, 8, 9 or 10 beats the true model; the best is a 2/6 at k = 8 (−98.36). The last run, k = 7
again, scores nothing new and prints the posterior over all five sizes:

```
  its posterior probability, equal prior odds over every support of the sizes [6, 7, 8, 9, 10]: 0.81
```

**8. The figures again,** now with the panel of the size-unknown run: `python plot.py`.

**9. The SINDy baselines.** About 4–5 minutes on a laptop.

```
python baselines.py
```

Five sparse-regression methods fit the same data, the 100 observations plus x0 at t = 0 as one more row (101 rows), with the same 20-mechanism library. A setting counts as a success when its nonzero coefficients are exactly the 7 true mechanisms. Every setting is saved to `results/baselines.csv` (below). Expected output (each line also ends with its
run time):

```
window + x0: 101 rows, t = 0, 10, ..., 20; PySINDy 2.1.0
  SINDy (STLSQ)                    exact model in  0 of 116 settings
  weak SINDy                       exact model in  0 of 174 settings
  ensemble SINDy                   exact model in  0 of  60 settings
  sparse Bayesian learning         exact model in  0 of   6 settings
  sparse Bayesian regression       exact model in  0 of   4 settings
  exact derivatives (diagnostic)   exact model in 17 of  58 settings
every setting (418) saved to results/baselines.csv
```

`results/baselines.csv` has one row per setting, 418 in all:

| columns | content |
|---|---|
| `method`, `derivatives`, `column_scaling`, `threshold`, `K`, `ensemble`, `prune`, `tau0` | the setting: the swept hyperparameters (empty where a method has none) |
| `details` | the fixed hyperparameters of that method, in words |
| `exact` | `yes` if the recovered model has exactly the 7 true mechanisms |
| `n_true_kept`, `n_lost`, `n_spurious` and `true_kept`, `lost`, `spurious` | how many true mechanisms are kept and lost, how many wrong ones are added, and their names |
| `rel_coef_error` | ‖c − c_true‖ / ‖c_true‖ over all 20 coefficients |
| `sbr_max_rhat`, `sbr_divergences` | sparse Bayesian regression only: the largest split R̂ of its chains and its divergent transitions |
| `1 -> du/dt`, …, `v^3 -> dv/dt` | the recovered model: the 20 coefficients, in full precision (0 for a removed mechanism) |

`results/baselines_reference.csv` is the same file from our run (Linux, Intel Xeon, OpenBLAS), for comparison.

| method | settings swept |
|---|---|
| SINDy | STLSQ on second-order or smoothed finite differences; raw or normalised columns; 29 thresholds from 1e-6 to 10 |
| weak SINDy | PySINDy's weak form with K = 50, 100 or 200 random test functions (polynomial degree 4); raw or normalised; 29 thresholds |
| ensemble SINDy | 100 STLSQ fits on 60-row bootstrap samples, with or without random sub-libraries; a term kept if it appears in at least half, with the median coefficient; raw or normalised; 15 thresholds |
| sparse Bayesian learning | Tipping's relevance vector machine (our code; PySINDy has none), per equation, on finite differences; a term removed when its prior precision exceeds 1e2, 1e3, 1e4, 1e5, 1e6 or 1e8 |
| sparse Bayesian regression | PySINDy's SBR (regularised horseshoe prior, NUTS with 2 chains of 1,000 + 2,000 draws); a term kept if its 95% credible interval excludes 0; global shrinkage 1e-3, 1e-2, 0.1 or 1 |
| exact derivatives | a diagnostic, not a method: STLSQ on the exact derivatives f(x) of the true equations at the same rows; raw or normalised; 29 thresholds |

- **x0 in the baselines** is a data row: a regression of derivatives needs ẋ(0), which here must be estimated from
  samples ten time units away. With the exact derivatives the same regression succeeds in many settings.
- **The five zeros are reproducible.** They were the same on three machines and, for weak SINDy, for every one of 10
  draws of its random test functions (seeded here, so a rerun gives the same draws).
- **The diagnostic's count is not.** The regression is nearly singular in double precision, so the count depends
  on the last bits of the data and the linear-algebra library: 17, 19 and 21 on three machines.
- **Sparse Bayesian regression's chains do not converge** on these data (split R̂ about 50–100); its four settings are reported as they are. Without numpyro this method is skipped and the others still run.

## On a cluster

`hpc.sbatch` runs steps 1 to 9 on one node with 64 CPUs (63 scoring processes), as in the paper's run.

```
sbatch hpc.sbatch                              # every step
STEPS="1 2 3" sbatch hpc.sbatch                # only these steps
PYTHON=/path/to/python3 sbatch hpc.sbatch      # another Python (default: python3 on the PATH)
PROGRESS=500 sbatch hpc.sbatch                 # exhaustive searches: a progress line every 500 supports (default 100)
```

The job runs the cluster's `python3`, which needs numpy, scipy, torch and matplotlib; it stops at once if one is missing. Step 9 also needs PySINDy (and numpyro); without them only step 9 fails. Without conda, install the packages for that `python3` once with `python3 -m pip install --user -r requirements.txt`. With conda, activate the environment of "Install" before submitting (`conda activate sparse-truth`): sbatch passes it on to the job.

- Submit from this folder. Everything is printed to `sparse_truth_<job>.out` as it happens (`tail -f` it), and each step also to `results/step<n>.log`.
- Step 1 prints its Stage 1 loss next to the paper's run's, 3.504868e-06. With the shipped `results/stage1.pt` the two agree; if they do not, that file is not the shipped one.
- Submitting again resumes where the job stopped.
- If you train your own Stage 1 (the `results` folder deleted), keep the CPU count the same between submissions: Stage 1 depends on the thread count.

On 63 workers the steps take about:

| steps | time |
|---|---|
| 1 | 1 h |
| 2, 5, 6 | 4 h each |
| 4 | 8 h |
| 7 | 1–2 days |
| 9 | 5 min |

## Output files (`results/`)

| file | content |
|---|---|
| `stage1.pt` | the Stage 1 networks and coefficients; shipped with this folder (HPC job 12514) |
| `scores_sigma<σ>_<stage1\|exact>_<climb\|exhaustive>.csv` | every scored support: log Z, rms misfit, its mechanisms |
| `final_sigma<σ>_k<k\|auto>.npz` | the final model: support, coefficients, log Z, the Stage 3 networks on [0, 20] |
| `fig_rivals_half.pdf`, `solution_*.pdf` | the figures |
| `baselines.csv` | every setting of the SINDy baselines (step 9): hyperparameters, recovered model, true/lost/spurious |
| `baselines_reference.csv` | the same from our run; shipped with this folder |

## Reproducibility

`pipeline.py` reproduces the development code bit for bit on the same machine. That covers the data, the initial networks, the Adam steps, SBL, the θ-step, the neighbourhoods and every evidence score.

`baselines.py` reproduces the development baselines bit for bit on the same machine: all 418 coefficient sets, except the three densest weak-SINDy fits (raw columns, threshold 1e-6), which differ by at most 1.3e-3 with the same terms kept.

The paper's numbers come from one Stage 1 training, HPC job 12514 on node bead70 of the RUC-HPC cluster. Its networks are `results/stage1.pt`. From them, `pipeline.py` reproduced the paper's route and numbers exactly on another machine. ℓ0-IHT-HTP still trains the networks for 4,000 Adam epochs, which another machine could round differently.

A fresh Stage 1 training gives the same networks only on the same kind of machine. The cluster's other node type gave other networks three times out of three, with the same data, starting weights and software. Its arithmetic differs: the CPU's instruction set selects the low-level kernels, and the number of CPU threads torch uses (`OMP_NUM_THREADS`) changes the order of its sums. Another machine's ODE solver can also change the last digit of some observations.

Other Stage 1 networks give another ℓ0-IHT-HTP support. The climb then starts elsewhere and may stop on a local maximum. The exhaustive search does not depend on the start: the true model was first on every machine tried. The last digits of the coefficients can differ between machines; the paper's run gives 1.1000023652 for
u².