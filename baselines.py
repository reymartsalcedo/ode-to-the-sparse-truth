"""The SINDy baselines of the paper: five sparse-regression methods on the paper's data, the 100 observations
on [10, 20] plus the initial state x0 at t = 0 as one more row (101 rows), and the exact-derivative diagnostic.

  SINDy     STLSQ on finite / smoothed finite differences, raw / normalised columns, 29 thresholds     116 settings
  weak      weak SINDy, K = 50, 100, 200 random test functions (seeded), raw / normalised, 29 thresholds 174
  ensemble  ensemble SINDy, bagging / bagging + library ensemble, raw / normalised, 15 thresholds        60
  SBL       sparse Bayesian learning (Tipping's relevance vector machine), 6 pruning levels               6
  SBR       sparse Bayesian regression (PySINDy's SBR, horseshoe prior, NUTS), 4 global shrinkages        4
  exact     the diagnostic: STLSQ on the EXACT derivatives f(x) at the 101 rows, raw / normalised          58

A setting succeeds when its nonzero coefficients (|c| > 1e-8) are exactly the 7 true mechanisms. The data, the
library order and the truth are those of pipeline.py. Every setting is saved to results/baselines.csv: its
hyperparameters, the recovered model (the 20 coefficients), the true mechanisms kept and lost, the spurious ones,
and the relative coefficient error.

    python baselines.py            # ~5 min; prints the number of successful settings per method
"""
import csv
import os
import time
import types
import warnings

import numpy as np
import pysindy as ps

import pipeline as p

warnings.filterwarnings("ignore")
T = np.r_[p.T0, p.T_OBS]                                   # x0 at t = 0, then the window
X = np.vstack([p.X0, p.Y])
C_TRUE = p.TRUE_C.T                                        # (2, 10), PySINDy's term order is pipeline.TERMS
TRUE = C_TRUE != 0
THRESHOLDS = np.logspace(-6, 1, 29)                        # SINDy, weak SINDy, the diagnostic
THRESHOLDS_ENSEMBLE = np.logspace(-6, 1, 15)
ZERO = 1e-8                                                # |c| at or below this counts as zero
OUT = "results/baselines.csv"
COLUMNS = ["method", "derivatives", "column_scaling", "threshold", "K", "ensemble", "prune", "tau0", "details", "exact",
           "n_true_kept", "n_lost", "n_spurious", "true_kept", "lost", "spurious", "rel_coef_error",
           "sbr_max_rhat", "sbr_divergences"] + p.MECH


def mechanisms(mask):
    """The names of the mechanisms where mask (2, 10) is True, separated by semicolons."""
    return "; ".join(p.MECH[2 * k + j] for k in range(10) for j in range(2) if mask[j, k])


def grade(coef):
    """The recovered model's status against the truth: exact or not, true kept, lost, spurious, coefficient error."""
    on = np.abs(coef) > ZERO
    return dict(exact="yes" if (on == TRUE).all() else "no", n_true_kept=int((on & TRUE).sum()),
                n_lost=int((~on & TRUE).sum()), n_spurious=int((on & ~TRUE).sum()), true_kept=mechanisms(on & TRUE),
                lost=mechanisms(~on & TRUE), spurious=mechanisms(on & ~TRUE),
                rel_coef_error=f"{np.linalg.norm(coef - C_TRUE) / np.linalg.norm(C_TRUE):.6e}")


def coefficients(coef):
    """The 20 coefficients of coef (2, 10) by mechanism name, in full precision."""
    return {m: f"{coef[j, k]:.17g}" for m, (k, j) in zip(p.MECH, np.ndindex(10, 2))}


def library():
    """The degree-3 polynomial library with a constant: 10 terms per equation, 20 mechanisms."""
    return ps.PolynomialLibrary(degree=3, include_bias=True)


def stlsq(threshold, norm):
    """STLSQ on raw columns (no ridge) or on normalised columns (ridge 1e-5)."""
    return ps.STLSQ(threshold=threshold, alpha=1e-5 if norm else 0.0, normalize_columns=norm)


def scaling(norm):
    """The column scaling of an STLSQ setting, as saved."""
    return "normalised (ridge 1e-5)" if norm else "raw (no ridge)"


def sindy(optimizer, x_dot=None, differentiation=None, lib=None):
    """The coefficients (2, 10) of a SINDy model fitted to the 101 rows."""
    m = ps.SINDy(feature_library=library() if lib is None else lib, optimizer=optimizer,
                 differentiation_method=ps.FiniteDifference() if differentiation is None else differentiation)
    m.fit(X, t=T, x_dot=x_dot, feature_names=["u", "v"])
    return m.coefficients()


def strong():
    """SINDy: STLSQ on second-order finite differences and on smoothed finite differences."""
    out = []
    for name, diff in (("finite differences, 2nd order", lambda: ps.FiniteDifference(order=2)),
                       ("smoothed finite differences", ps.SmoothedFiniteDifference)):
        for norm in (False, True):
            for th in THRESHOLDS:
                s = dict(method="SINDy (STLSQ)", derivatives=name, column_scaling=scaling(norm), threshold=f"{th:.6g}",
                         details="PySINDy STLSQ, max 20 iterations")
                out.append((s, sindy(stlsq(th, norm), differentiation=diff())))
    return out


def diagnostic():
    """STLSQ on the exact derivatives f(x) of the true equations at the same rows (not a method)."""
    x_dot = np.array([p.rhs(0, x) for x in X])
    return [(dict(method="exact derivatives (diagnostic)", derivatives="exact, f(x) of the true equations",
                  column_scaling=scaling(norm), threshold=f"{th:.6g}", details="PySINDy STLSQ, max 20 iterations"),
             sindy(stlsq(th, norm), x_dot=x_dot)) for norm in (False, True) for th in THRESHOLDS]


def _scalar_uniform_redraw():
    """Work around a PySINDy 2.1 bug (a redrawn weak subdomain is a 1-element array); patches that module only."""
    from pysindy.feature_library import weak_pde_library as W

    def uniform(low=0.0, high=1.0, size=None):
        """numpy's uniform draw, a scalar when size=1."""
        x = np.random.uniform(low, high, size)
        return float(np.asarray(x).reshape(-1)[0]) if size == 1 else x
    rnd = types.SimpleNamespace(**{k: getattr(np.random, k) for k in dir(np.random) if not k.startswith("__")})
    rnd.uniform = uniform
    W.np = types.SimpleNamespace(**{**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")}, "random": rnd})


def weak():
    """Weak SINDy: STLSQ on the weak form against K test functions; every K draws its subdomains from seed K."""
    _scalar_uniform_redraw()
    out = []
    for K in (50, 100, 200):
        for norm in (False, True):
            for th in THRESHOLDS:
                np.random.seed(K)                          # the subdomains are drawn during the fit
                lib = ps.WeakPDELibrary(function_library=library(), spatiotemporal_grid=T, is_uniform=False, K=K, p=4)
                s = dict(method="weak SINDy", derivatives="none (weak form)", column_scaling=scaling(norm),
                         threshold=f"{th:.6g}", K=K, details=f"PySINDy WeakPDELibrary, test-function degree p = 4, "
                                                             f"subdomains drawn from seed {K}; STLSQ")
                out.append((s, sindy(stlsq(th, norm), lib=lib)))
    return out


def ensemble():
    """Ensemble SINDy: 100 STLSQ fits on 60-row bootstrap samples (and random sub-libraries); median of the kept."""
    out = []
    n_subset = max(10, int(0.6 * len(T)))
    for library_ensemble in (False, True):
        for norm in (False, True):
            for th in THRESHOLDS_ENSEMBLE:
                np.random.seed(0)
                opt = ps.EnsembleOptimizer(stlsq(th, norm), bagging=True, library_ensemble=library_ensemble,
                                           n_models=100, n_subset=n_subset)
                sindy(opt, differentiation=ps.FiniteDifference(order=2))
                fits = np.asarray(opt.coef_list)           # (100, 2, 10)
                coef = np.where((np.abs(fits) > 0).mean(axis=0) >= 0.5, np.median(fits, axis=0), 0.0)
                mode = "bagging + library ensemble" if library_ensemble else "bagging"
                s = dict(method="ensemble SINDy", derivatives="finite differences, 2nd order",
                         column_scaling=scaling(norm), threshold=f"{th:.6g}", ensemble=mode,
                         details=f"PySINDy EnsembleOptimizer(STLSQ), 100 models of {n_subset} rows, seed 0; "
                                 "a term kept if in >= 50% of the models, its median")
                out.append((s, coef))
    return out


def rvm(A, y, prune, n_iter=1000):
    """Tipping's relevance vector machine (MacKay updates); a term whose precision exceeds prune is removed."""
    n, k = A.shape
    alpha, beta, keep, mu = np.ones(k), 1.0 / max(np.var(y), 1e-30), np.ones(k, bool), np.zeros(k)
    for _ in range(n_iter):
        Ak = A[:, keep]
        H = beta * Ak.T @ Ak + np.diag(alpha[keep])
        try:
            S = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(H)
        m = beta * S @ Ak.T @ y
        gamma = 1 - alpha[keep] * np.diag(S)
        r = y - Ak @ m
        beta = max(n - gamma.sum(), 1e-12) / max(r @ r, 1e-300)
        mu[:] = 0
        mu[keep] = m
        old = alpha.copy()
        alpha[keep] = gamma / np.maximum(m ** 2, 1e-300)
        keep &= alpha < prune
        if not keep.any() or np.allclose(np.log(alpha[keep]), np.log(old[keep]), atol=1e-6):
            break
    mu[~keep] = 0
    return mu


def sbl():
    """Sparse Bayesian learning: the relevance vector machine per equation on finite differences."""
    A = np.asarray(library().fit_transform(X), dtype=float)
    x_dot = np.asarray(ps.FiniteDifference(order=2)(X, T), dtype=float)
    return [(dict(method="sparse Bayesian learning", derivatives="finite differences, 2nd order", prune=f"{prune:g}",
                  details="our relevance vector machine (Tipping 2001), MacKay updates, max 1000 iterations, "
                          "noise precision estimated"),
             np.vstack([rvm(A, x_dot[:, j], prune) for j in range(2)])) for prune in (1e2, 1e3, 1e4, 1e5, 1e6, 1e8)]


def sbr():
    """Sparse Bayesian regression: a term is kept if its 95% credible interval excludes 0 (NUTS, 2 chains)."""
    from pysindy.optimizers import SBR                     # PySINDy exports it only when numpyro is installed
    from numpyro.diagnostics import split_gelman_rubin
    x_dot = np.asarray(ps.FiniteDifference(order=2)(X, T), dtype=float)
    out = []
    for tau0 in (1e-3, 1e-2, 1e-1, 1.0):
        opt = SBR(sparsity_coef_tau0=tau0, num_warmup=1000, num_samples=2000,
                  mcmc_kwargs={"progress_bar": False, "num_chains": 2, "chain_method": "sequential"})
        sindy(opt, x_dot=x_dot)
        beta = np.asarray(opt.mcmc_.get_samples()["beta"])  # (samples, 2, 10)
        lo, hi = np.percentile(beta, [2.5, 97.5], axis=0)
        rhat = np.nanmax(split_gelman_rubin(np.asarray(opt.mcmc_.get_samples(group_by_chain=True)["beta"])))
        s = dict(method="sparse Bayesian regression", derivatives="finite differences, 2nd order", tau0=f"{tau0:g}",
                 sbr_max_rhat=f"{rhat:.2f}",
                 sbr_divergences=int(np.asarray(opt.mcmc_.get_extra_fields()["diverging"]).sum()),
                 details="PySINDy SBR (regularised horseshoe), NUTS 2 chains x (1000 warm-up + 2000 draws); "
                         "a term kept if its 95% credible interval excludes 0, its posterior mean")
        out.append((s, np.where((lo > 0) | (hi < 0), beta.mean(axis=0), 0.0)))
    return out


def main():
    """Every method on the 101 rows; prints the successful settings per method and saves every setting to OUT."""
    print(f"window + x0: {len(T)} rows, t = {T[0]:g}, {T[1]:g}, ..., {T[-1]:g}; PySINDy {ps.__version__}")
    rows = []
    for name, method in (("SINDy (STLSQ)", strong), ("weak SINDy", weak), ("ensemble SINDy", ensemble),
                         ("sparse Bayesian learning", sbl), ("sparse Bayesian regression", sbr),
                         ("exact derivatives (diagnostic)", diagnostic)):
        t0 = time.time()
        try:
            fits = method()
        except ImportError as e:
            print(f"  {name:32s} skipped: {e} (pip install numpyro)")
            continue
        graded = [{**s, **grade(c), **coefficients(c)} for s, c in fits]
        rows += graded
        n = sum(g["exact"] == "yes" for g in graded)
        print(f"  {name:32s} exact model in {n:2d} of {len(graded):3d} settings ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"every setting ({len(rows)}) saved to {OUT}")


if __name__ == "__main__":
    main()
