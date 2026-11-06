"""Trajectory-Based Bayesian Discovery of a Host-Pathogen Model from Quasi-Steady-State Observations
by Reymart Salcedo Lagunero, Andreas Langer, Cordula Reisch

The host-pathogen model  du/dt = -u^3 + 1.1 u^2 - 0.2 u - 0.2 u v,  dv/dt = 0.5 u - v + 0.5 u v,  x0 = (1, 0.5),
is observed (noise-free) at 100 evenly spaced times on [10, 20]; x0 is known. Library: the 10 monomials of degree
<= 3 in each equation, 20 mechanisms.

  Stage 1  Adam on the PINN loss over the networks and all 20 coefficients (15,000 epochs).
  Stage 2  (1) sparse Bayesian learning on the networks' derivatives;
           (2) l0-IHT-HTP for k (or k = 19..2);
           (3) the two-swap climb on the Laplace approximated evidence of the ODE solutions;
           (4) the selection;
           (5) the ODE refit of the selected support.
  Stage 3  the networks refitted with the support and coefficients fixed (Adam 5,000 epochs, L-BFGS <= 300 steps).

    python pipeline.py --sigma 2e-7 --k 7 --workers 8                 # k known
    python pipeline.py --sigma 2e-7 --workers 8                       # k unknown: k = 19, ..., 2
    python pipeline.py --sigma 2e-7 --k 7 --exhaustive --workers 63   # every support of size 7
    python pipeline.py --sigma 2e-7 --k 7 --exhaustive --exact-start  # the same, refits started from exact derivatives

Stage 1 is loaded from results/stage1.pt, which ships with the paper's Stage 1 networks (HPC job 12514); without
that file, Stage 1 is trained and saved there. Every scored support is appended to
results/scores_sigma<sigma>_<stage1|exact>_<climb|exhaustive>.csv and reread, so an interrupted run resumes (climbs
allow 60 s per support, the exhaustive search 3,600 s). A pipeline run saves its final model to
results/final_sigma<sigma>_k<k|auto>.npz for plot.py. The scores depend on the Stage 1 networks: delete the whole
results folder to start again.
"""
import argparse, itertools, math, multiprocessing as mp, os, signal, time
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# the paper's settings
P = dict(a1=1.0, a2=1.1, a3=0.2, a4=0.2, b1=0.5, b2=1.0, b3=0.5)
X0, T0, T1 = [1.0, 0.5], 0.0, 20.0
DATA_SOLVER = dict(method="LSODA", rtol=1e-10, atol=1e-12)        # the data
SIM_SOLVER = dict(method="LSODA", rtol=1e-8, atol=1e-10)          # the evidence's ODE solves
EXPS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
TERMS = ["1", "u", "v", "u^2", "u*v", "v^2", "u^3", "u^2*v", "u*v^2", "v^3"]
MECH = [f"{t} -> d{s}/dt" for t in TERMS for s in "uv"]
TRUE_C = np.zeros((10, 2))
TRUE_C[[1, 3, 4, 6], 0] = [-P["a3"], P["a2"], -P["a4"], -P["a1"]]
TRUE_C[[1, 2, 4], 1] = [P["b1"], -P["b2"], P["b3"]]
TRUTH = frozenset(np.flatnonzero(TRUE_C.reshape(-1)).tolist())    # used only to label supports a/b in the output
SEED, HIDDEN, LAYERS, FOURIER, N_C = 42, 64, 4, 2, 1000
W_DATA, W_ODE, W_IC = 100.0, 10.0, 1.0
LR, WARMUP, CLIP = 1e-3, 25, 1.0
TAU, MAX_NFEV = 2.0, 100
TIMEOUT = dict(climb=60, exhaustive=3600)                       # seconds per support


def rhs(t, y):
    """The true model."""
    u, v = y
    return [-P["a1"] * u ** 3 + P["a2"] * u ** 2 - P["a3"] * u - P["a4"] * u * v, P["b1"] * u - P["b2"] * v + P["b3"] * u * v]


T_OBS = np.linspace(10.0, 20.0, 100)
Y = solve_ivp(rhs, (T0, T1), X0, t_eval=T_OBS, **DATA_SOLVER).y.T      # (100, 2) observations; x0 is not among them
BOUND = 1e3 * (1.0 + max(np.abs(Y).max(), max(X0)))                    # a solution leaving |x| < BOUND has blown up
col = lambda a: torch.tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64).unsqueeze(1)
unit_time = lambda t: 2.0 * (t - T0) / (T1 - T0) - 1.0                 # time scaled to [-1, 1]
T_DATA, T_IC = unit_time(col(T_OBS)), unit_time(col([T0]))
X_DATA = [col(Y[:, j]) for j in range(2)]
T_COLL = unit_time(col(np.linspace(T0, T1, N_C))).requires_grad_(True)


def library(x):
    """The 10 monomials of the states x = [u, v] (torch columns), (N, 10)."""
    cols = []
    for e in EXPS:
        c = torch.ones_like(x[0])
        for s, k in zip(x, e):
            if k:
                c = c * s ** k
        cols.append(c)
    return torch.cat(cols, dim=1)


with torch.no_grad():
    SCALE = library(X_DATA).pow(2).mean(0, keepdim=True).sqrt() + 1e-6    # s_m: rms over the observed states
F = 1.0 / SCALE.numpy().reshape(-1, 1)                                     # physical coefficient = theta * F


class Net(nn.Module):
    """One state: tanh layers on (tau, sin k pi tau, cos k pi tau), softplus output (positive state)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, FOURIER + 1, dtype=torch.float64))
        layers = [nn.Linear(2 * FOURIER + 1, HIDDEN), nn.Tanh()]
        for _ in range(LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
        self.net = nn.Sequential(*layers, nn.Linear(HIDDEN, 1)).double()
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        a = np.pi * t * self.freqs.unsqueeze(0)
        return torch.nn.functional.softplus(self.net(torch.cat([t, torch.sin(a), torch.cos(a)], dim=1))) + 1e-8


class PINN(nn.Module):
    """The two state networks and the scaled coefficients theta (mechanism order); mask = the active mechanisms."""

    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList([Net(), Net()])
        self.theta = nn.Parameter(torch.zeros(20, dtype=torch.float64))
        self.register_buffer("mask", torch.ones(20, dtype=torch.float64))


def derivatives(model):
    """States, their time derivatives and the scaled library at the collocation points."""
    x = [net(T_COLL) for net in model.nets]
    dx = [torch.autograd.grad(s, T_COLL, torch.ones_like(s), create_graph=True)[0] * (2.0 / (T1 - T0)) for s in x]
    return x, dx, library(x) / SCALE


def loss(model):
    """The PINN loss: w_data L_data + w_ode L_ode + w_ic L_ic."""
    data = sum(torch.mean((net(T_DATA) - d) ** 2) for net, d in zip(model.nets, X_DATA))
    _, dx, A = derivatives(model)
    f = A @ (model.theta * model.mask).view(10, 2)
    ode = sum(torch.mean((d - f[:, j:j + 1]) ** 2) for j, d in enumerate(dx))
    ic = sum(((net(T_IC) - x0) ** 2).squeeze() for net, x0 in zip(model.nets, X0))
    return W_DATA * data + W_ODE * ode + W_IC * ic


def blocks(model):
    """A (scaled library) and b (derivatives) of the networks at the collocation points, numpy."""
    _, dx, A = derivatives(model)
    return A.detach().numpy(), np.column_stack([d.detach().numpy().ravel() for d in dx])


def state(model):
    """A copy of the model's weights, coefficients and mask."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def adam(model, epochs, lr=LR, train_theta=True):
    """Adam (theta at 10 x lr) with a linear warm-up and gradient clipping; keeps the best state."""
    nets = [p for net in model.nets for p in net.parameters()]
    model.theta.requires_grad_(train_theta)
    opt = torch.optim.Adam([{"params": nets, "lr": lr}] + ([{"params": [model.theta], "lr": 10 * lr}] if train_theta else []))
    base = [g["lr"] for g in opt.param_groups]
    best, best_state = loss(model).item(), state(model)
    for epoch in range(1, epochs + 1):
        if epoch <= WARMUP:
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * epoch / WARMUP
        opt.zero_grad()
        loss(model).backward()
        nn.utils.clip_grad_norm_(nets, max_norm=CLIP)
        nn.utils.clip_grad_norm_([model.theta], max_norm=CLIP)
        opt.step()
        now = loss(model).item()
        if now < best:
            best, best_state = now, state(model)
    model.load_state_dict(best_state)
    model.theta.requires_grad_(True)
    model.theta.grad = None
    return best


def lbfgs(model, steps):
    """L-BFGS on the networks only; stops after 10 steps without a relative change of 1e-12; keeps the best."""
    model.theta.requires_grad_(False)
    opt = torch.optim.LBFGS([p for net in model.nets for p in net.parameters()], lr=1e-3, line_search_fn="strong_wolfe")
    best, best_state, prev, stall = loss(model).item(), state(model), None, 0

    def closure():
        opt.zero_grad()
        value = loss(model)
        value.backward()
        return value
    for _ in range(steps):
        opt.step(closure)
        now = loss(model).item()
        if now < best:
            best, best_state = now, state(model)
        stall = stall + 1 if prev is not None and abs(now - prev) / (abs(prev) + 1e-12) < 1e-12 else 0
        prev = now
        if stall >= 10:
            break
    model.load_state_dict(best_state)
    model.theta.requires_grad_(True)
    return best


def sbl(A, y):
    """Sparse Bayesian learning (Tipping 2001), MacKay's fixed-point updates; a weight whose alpha exceeds 1e8 is 0."""
    n, p = A.shape
    alpha, beta, keep = np.ones(p), 1.0 / max(np.var(y), 1e-30), np.ones(p, bool)
    for _ in range(500):
        Ak = A[:, keep]
        H = beta * Ak.T @ Ak + np.diag(alpha[keep])
        try:
            S = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            S = np.linalg.pinv(H)
        mu = beta * S @ Ak.T @ y
        gam = 1.0 - alpha[keep] * np.diag(S)
        new = gam / np.maximum(mu ** 2, 1e-300)
        res = y - Ak @ mu
        beta = max(n - gam.sum(), 1e-12) / max(res @ res, 1e-300)
        done = np.max(np.abs(np.log(new) - np.log(alpha[keep]))) < 1e-6
        alpha[keep] = new
        keep = alpha < 1e8
        if done or not keep.any():
            break
    w = np.zeros(p)
    Ak = A[:, keep]
    w[keep] = beta * np.linalg.inv(beta * Ak.T @ Ak + np.diag(alpha[keep])) @ Ak.T @ y
    return w


def lsq(S, A, b):
    """Least squares of each equation's active mechanisms on A theta_j = b_j (theta in mechanism order)."""
    theta = np.zeros(20)
    for j in range(2):
        idx = [m for m in sorted(S) if m % 2 == j]
        if idx:
            theta[idx] = np.linalg.lstsq(A[:, [m // 2 for m in idx]], b[:, j], rcond=None)[0]
    return theta


def theta_step(theta, A, b, k):
    """Up to 20 gradient steps on w_ode L_ode (step 0.9 / L), each keeping the k largest |theta| (>= 1 per equation)
    and refitting them by least squares (HTP); stops when support and theta no longer change."""
    s = 1.0 / np.sqrt(N_C)
    J = np.zeros((2 * N_C, 20))                          # dr/dtheta of the residual r = vec(A Xi - b) / sqrt(N_c)
    for m in range(20):
        J[m % 2::2, m] = A[:, m // 2] * s
    eta = 0.9 / (2 * W_ODE * np.linalg.norm(J, 2) ** 2)
    S = set(np.flatnonzero(theta).tolist())
    for _ in range(20):
        r = ((A @ theta.reshape(10, 2) - b) * s).ravel()
        z = theta - eta * 2 * W_ODE * (J.T @ r)
        order = [int(m) for m in np.argsort(-np.abs(z), kind="stable")]
        keep = [m for m in order if z[m] != 0][:k]
        for j in range(2):                               # an equation without a term takes its largest
            if not any(m % 2 == j for m in keep):
                keep.append(next(m for m in order if m % 2 == j))
        while len(keep) > k:                             # ... and the weakest of a two-term equation makes room
            keep.remove(next(m for m in reversed(order) if m in keep and sum(q % 2 == m % 2 for q in keep) > 1))
        new = lsq(keep, A, b)
        done = set(keep) == S and np.linalg.norm(new - theta) <= 1e-12 * (1 + np.linalg.norm(theta))
        theta, S = new, set(keep)
        if done:
            break
    return theta, S


def htp(model, k):
    """l0-IHT-HTP: at most 15 alternations of a w-step (Adam, 200 epochs, theta fixed) and a theta-step, stopping when
    the support is unchanged for three; then Adam on w and theta (1,000 epochs) with the support fixed."""
    prev, stable = None, 0
    for _ in range(15):
        adam(model, 200, train_theta=False)
        theta, S = theta_step(model.theta.detach().numpy(), *blocks(model), k)
        mask = np.zeros(20)
        mask[sorted(S)] = 1.0
        with torch.no_grad():
            model.theta.copy_(torch.as_tensor(theta))
            model.mask.copy_(torch.as_tensor(mask))
        stable = stable + 1 if S == prev else 0
        prev = S
        if stable >= 3:
            break
    adam(model, 1000)
    return frozenset(prev)


def evidence(C, sigma):
    """One Levenberg-Marquardt refit of the nonzero entries c of C (physical units) through the ODE from x0, and the
    Laplace log evidence -|r|^2/2 sigma^2 - |c|^2/2 tau^2 - k log tau - 1/2 log det(J'J/sigma^2 + I/tau^2).
    Returns (log Z, rms misfit, refitted C), or None if the solution blows up or fails."""
    rows, cols = np.nonzero(C)
    p, E = rows.size, np.array(EXPS, dtype=np.float64)
    memo = {}

    def solve(c):                                     # X(c) - y and J = dX/dc (forward sensitivities), or None
        key = c.tobytes()
        if key not in memo:
            Cm = np.zeros_like(C)
            Cm[rows, cols] = c

            def f(t, z):
                x = z[:2]
                pw = x[None, :] ** E
                th = np.prod(pw, axis=1)
                grad = np.stack([np.where(E[:, s] > 0, E[:, s] * x[s] ** np.maximum(E[:, s] - 1, 0), 0.0)
                                 * np.prod(np.delete(pw, s, axis=1), axis=1) for s in range(2)], axis=1)
                dL = (Cm.T @ grad) @ z[2:].reshape(2, p)
                dL[cols, np.arange(p)] += th[rows]
                return np.concatenate([th @ Cm, dL.ravel()])

            def blowup(t, z):
                return BOUND - float(np.max(np.abs(z[:2])))
            blowup.terminal = True
            with np.errstate(all="ignore"):
                sol = solve_ivp(f, (T0, float(T_OBS[-1])), np.r_[np.asarray(X0, dtype=np.float64), np.zeros(2 * p)],
                                t_eval=T_OBS, events=blowup, **SIM_SOLVER)
            ok = sol.status == 0 and sol.y.shape[1] == len(T_OBS) and np.all(np.isfinite(sol.y))
            memo.clear()
            memo[key] = ((sol.y[:2].T - Y).ravel(), sol.y[2:].T.reshape(-1, p), True) if ok else \
                (np.full(Y.size, BOUND), np.zeros((Y.size, p)), False)
        return memo[key]
    if p == 0:
        return None
    try:
        fit = least_squares(lambda c: np.r_[solve(c)[0] / sigma, c / TAU], C[rows, cols].copy(),
                            jac=lambda c: np.vstack([solve(c)[1] / sigma, np.eye(p) / TAU]),
                            method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=MAX_NFEV)
    except Exception:
        return None
    r, J, ok = solve(fit.x)
    if not ok:
        return None
    g = np.linalg.svd(J, compute_uv=False)
    h = np.r_[g ** 2, np.zeros(p - g.size)] / sigma ** 2 + 1.0 / TAU ** 2
    Cf = np.zeros_like(C)
    Cf[rows, cols] = fit.x
    logz = -0.5 * r @ r / sigma ** 2 - 0.5 * fit.x @ fit.x / TAU ** 2 - p * np.log(TAU) - 0.5 * np.sum(np.log(h))
    return float(logz), float(np.sqrt(np.mean(r ** 2))), Cf


class _Timeout(BaseException):
    pass


def _alarm(*_):
    """The timeout's signal handler."""
    raise _Timeout


def map_refit(C, sigma, gate=-1e4, timeout=0):
    """The MAP: the refit is repeated from its own result while log Z rises (at most 5 fits; only above `gate`).
    After `timeout` seconds (0: none), or an error, the best refit so far is returned."""
    best, n = None, 1
    timeout = timeout if hasattr(signal, "SIGALRM") else 0     # no timeout where there is no SIGALRM
    if timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
    try:
        best = evidence(C, sigma)
        while best is not None and best[0] > gate and n < 5:
            nxt = evidence(best[2], sigma)
            n += 1
            if nxt is None or nxt[0] <= best[0] + 1e-3:
                if nxt is not None and nxt[0] > best[0]:
                    best = nxt
                break
            best = nxt
    except (_Timeout, Exception):
        pass
    finally:
        if timeout:
            signal.alarm(0)
    return best


START = None                                          # (A, b) of the Stage 1 networks, set in every scoring process


def _set_start(start):
    """Sets START in a scoring process."""
    global START
    START = start


def score(job):
    """(support, (log Z, rms misfit)) of one support: least squares on START, then the MAP refit through the ODE,
    at most `timeout` seconds; -inf if it fails."""
    S, sigma, timeout = job
    out = map_refit(lsq(S, *START).reshape(10, 2) * F, sigma, timeout=timeout)
    return S, (out[0], out[1]) if out else (-math.inf, math.inf)


def label(S):
    """a/b: the true and the spurious mechanisms of S."""
    return f"{len(S & TRUTH)}/{len(S - TRUTH)}"


def neighbours(S):
    """Every support within one or two swaps of S that keeps a mechanism in each equation."""
    out, found = sorted(set(range(20)) - S), set()
    for d in (1, 2):
        for drop in itertools.combinations(sorted(S), d):
            for add in itertools.combinations(out, d):
                S1 = (S - set(drop)) | set(add)
                if any(m % 2 == 0 for m in S1) and any(m % 2 == 1 for m in S1):
                    found.add(frozenset(S1))
    return found


def climb(S, scores, add):
    """The two-swap climb: move to the best support within two swaps while the evidence rises."""
    add([S])
    print(f"    start {label(S)}, log Z {scores[S][0]:.2f}", flush=True)
    while True:
        nb = neighbours(S)
        add(nb)
        best = max(nb, key=lambda q: scores[q][0])
        if scores[best][0] <= scores[S][0] + 1e-6:
            return S
        S = best
        print(f"    -> {label(S)}, log Z {scores[S][0]:.2f}", flush=True)


def main():
    """Stage 1, then Stage 2 and Stage 3 (or the exhaustive search), as the module docstring says."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, required=True, help="the noise scale of the evidence")
    ap.add_argument("--k", type=int, help="the support size; omitted: k = 19, ..., 2")
    ap.add_argument("--exhaustive", action="store_true", help="score every support of size k and stop")
    ap.add_argument("--exact-start", action="store_true", help="with --exhaustive: start refits from exact derivatives")
    ap.add_argument("--workers", type=int, default=1, help="processes scoring supports")
    a = ap.parse_args()
    if (a.exhaustive or a.exact_start) and not a.k or a.exact_start and not a.exhaustive:
        ap.error("--exhaustive needs --k, and --exact-start needs --exhaustive")
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(SEED)
    model = PINN()
    if os.path.exists("results/stage1.pt"):
        model.load_state_dict(torch.load("results/stage1.pt"))
        print(f"Stage 1 loaded from results/stage1.pt (PINN loss {loss(model).item():.6e})")
    else:
        t0 = time.time()
        print(f"Stage 1: Adam, 15,000 epochs ... PINN loss {adam(model, 15000):.6e} ({time.time() - t0:.0f} s)", flush=True)
        torch.save(model.state_dict(), "results/stage1.pt")
    A, b = blocks(model)
    top = lambda th: len(TRUTH & set(np.argsort(-np.abs(th), kind="stable")[:7].tolist()))
    print(f"  true mechanisms among the 7 largest coefficients: {top(model.theta.detach().numpy())}")
    if a.exact_start:                                 # a diagnostic: the true solution and its exact derivatives
        x = solve_ivp(rhs, (T0, T1), X0, t_eval=np.linspace(T0, T1, N_C), **DATA_SOLVER).y
        A = np.column_stack([x[0] ** e[0] * x[1] ** e[1] for e in EXPS]) / SCALE.numpy()
        b = np.asarray(rhs(0.0, x)).T

    mode = "exhaustive" if a.exhaustive else "climb"
    path = f"results/scores_sigma{a.sigma:g}_{'exact' if a.exact_start else 'stage1'}_{mode}.csv"
    scores = {}
    if os.path.exists(path):
        for line in open(path).read().splitlines()[1:]:
            z, d, s = line.split(",")
            scores[frozenset(MECH.index(n) for n in s.split(";"))] = (float(z), float(d))
    else:
        open(path, "w").write("log_evidence,misfit,support\n")
    pool = mp.Pool(a.workers, initializer=_set_start, initargs=((A, b),)) if a.workers > 1 else None
    _set_start((A, b))

    def add(supports):                                # score the supports not yet in `scores`, appending to `path`
        new = [(tuple(sorted(S)), a.sigma, TIMEOUT[mode]) for S in supports if S not in scores]
        with open(path, "a") as fh:
            for S, v in (pool.imap_unordered(score, new, chunksize=8) if pool else map(score, new)):
                scores[frozenset(S)] = v
                fh.write(f"{v[0]:.6f},{v[1]:.6e},{';'.join(MECH[m] for m in S)}\n")

    if a.exhaustive:
        every = [frozenset(S) for S in itertools.combinations(range(20), a.k)
                 if any(m % 2 == 0 for m in S) and any(m % 2 == 1 for m in S)]
        t0 = time.time()
        add(every)
        rank = sorted(every, key=lambda S: -scores[S][0])
        print(f"\nExhaustive search, k = {a.k}, sigma = {a.sigma:g}: {len(every):,} supports ({time.time() - t0:.0f} s)")
        n = Y.size
        x_true = solve_ivp(rhs, (T0, T1), X0, t_eval=np.linspace(T0, T1, 2001), **DATA_SOLVER).y
        for i, S in enumerate(rank[:10], 1):
            z, d = scores[S]
            fit = n * d ** 2 / (2 * a.sigma ** 2)
            line = f"  {i:2d}. {label(S)}  log Z {z:9.2f} = -fit {fit:6.2f} - prior and Occam {-z - fit:6.2f}  misfit {d:.1e}"
            if i <= 4:                                # the deviation from the true solution on [0, 20]
                C = map_refit(lsq(S, A, b).reshape(10, 2) * F, a.sigma)[2]
                x = solve_ivp(lambda t, y: np.prod(np.asarray(y)[None, :] ** np.array(EXPS), axis=1) @ C, (T0, T1), X0,
                              t_eval=np.linspace(T0, T1, 2001), **SIM_SOLVER).y
                line += f"  |dx|_inf {np.abs(x - x_true).max() if x.shape == x_true.shape else np.inf:.1e}"
            print(line + "   " + "; ".join(MECH[m] for m in sorted(S)))
        sizes = [k for k in range(2, 20) if sum(len(S) == k for S in scores) == math.comb(20, k) - 2 * math.comb(10, k)]
        if a.k == len(TRUTH):
            r, zt = rank.index(TRUTH) + 1, scores[TRUTH][0]
            other = next(S for S in rank if S != TRUTH)
            gap = zt - scores[other][0]
            print(f"  the true model ranks {r} (log Z {zt:.2f}); " + (
                f"it is {gap:.2f} above the runner-up {label(other)}, a Bayes factor of {math.exp(gap):.1f}" if r == 1
                else f"the best support, {label(other)}, is {-gap:.2f} above it"))
            z = np.array([v[0] for S, v in scores.items() if len(S) in sizes])
            print(f"  its posterior probability, equal prior odds over every support of the sizes {sizes}: "
                  f"{math.exp(zt - z.max() - math.log(np.exp(z - z.max()).sum())):.2f}")
        else:
            print(f"  (the true model's rank and posterior are printed by the k = {len(TRUTH)} search)")
        return

    print("\nStage 2")
    th = np.zeros(20)
    for j in range(2):
        th[j::2] = sbl(A, b[:, j])
    with torch.no_grad():
        model.theta.copy_(torch.as_tensor(th))
    print(f"  (1) SBL: true mechanisms among the 7 largest coefficients: {top(th)}")
    after_sbl, nets, ends = state(model), {}, {}
    for k in ([a.k] if a.k else range(19, 1, -1)):
        model.load_state_dict(after_sbl)
        S = htp(model, k)
        nets[k] = state(model)
        print(f"  k = {k}: (2) l0-IHT-HTP: {label(S)}; (3) the climb:", flush=True)
        ends[k] = climb(S, scores, add)
    S = max(ends.values(), key=lambda q: scores[q][0])
    if not a.k:
        print("  the climbs' ends:\n" + "\n".join(f"    k = {k:2d}: {label(E)}, log Z {scores[E][0]:.2f}" for k, E in ends.items()))
    print(f"  (4) selected: {label(S)}, log Z {scores[S][0]:.2f}" + ("" if a.k else f" (k = {len(S)})"))
    model.load_state_dict(nets[len(S)])               # (5) the MAP from the current and the Stage 1 networks' start
    fits = [map_refit(lsq(S, *start).reshape(10, 2) * F, a.sigma, -math.inf) for start in (blocks(model), START)]
    z, d, C = max((f for f in fits if f), key=lambda f: f[0])
    print(f"  (5) ODE refit: log Z {z:.2f}, misfit {d:.1e}")

    mask = np.zeros(20)
    mask[sorted(S)] = 1.0
    with torch.no_grad():
        model.theta.copy_(torch.as_tensor((C / F).reshape(-1) * mask))
        model.mask.copy_(torch.as_tensor(mask))
    adam(model, 5000, lr=LR / 2, train_theta=False)
    print(f"Stage 3: PINN loss {lbfgs(model, 300):.6e}")
    print(f"the final model ({label(S)}):")
    for j, s in enumerate("uv"):
        print(f"  d{s}/dt = " + " ".join(f"{C[i, j]:+.10f} {TERMS[i]}" for i in range(10) if C[i, j]))
    i, j = np.unravel_index(np.argmax(np.abs(C - TRUE_C)), C.shape)
    print(f"  relative coefficient error {np.linalg.norm(C - TRUE_C) / np.linalg.norm(TRUE_C):.1e}; the largest deviation "
          f"from the true coefficients, {abs(C[i, j] - TRUE_C[i, j]):.1e}, is in {MECH[2 * i + j]}")
    t = np.linspace(T0, T1, 2001)
    with torch.no_grad():
        pinn = np.column_stack([net(unit_time(col(t))).numpy().ravel() for net in model.nets])
    out = f"results/final_sigma{a.sigma:g}_k{a.k or 'auto'}.npz"
    np.savez(out, support=sorted(S), C=C, log_evidence=z, t=t, pinn=pinn, sigma=a.sigma)
    print(f"  saved to {out}")
    if pool:
        pool.close()


if __name__ == "__main__":
    main()
