"""The figure of the paper and the solution panels of the final models, from the results of pipeline.py.

  fig_rivals_half.pdf                 the paper's figure: the four best seven-term supports of the exhaustive search at
                                      sigma = 2e-7, each refitted through the ODE and integrated from x0, over the true
                                      solution; needs the complete table (pipeline.py --sigma 2e-7 --k 7 --exhaustive)
  solution_<run>.pdf                  for every results/final_<run>.npz: the initial state and the observations, the
                                      true solution and the Stage 3 networks (PINN)
  solution_<run>_with_ranks.pdf       the same with the two best other supports of that run's climb overlaid

    python plot.py            # writes into results/
"""
import glob
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pipeline as p

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
logging.getLogger("fontTools").setLevel(logging.ERROR)     # no font-subsetting notes in the terminal


def scores(path):
    """{support: log Z} of a scores file of pipeline.py"""
    out = {}
    for line in open(path).read().splitlines()[1:]:
        z, _, s = line.split(",")
        out[frozenset(p.MECH.index(n) for n in s.split(";"))] = float(z)
    return out


def solution(S, sigma, start, t):
    """x(t; c) of support S: c from least squares on `start`, then the MAP refit through the ODE (as the climb scores)"""
    C = p.map_refit(p.lsq(sorted(S), *start).reshape(10, 2) * p.F, sigma, S=S)[2]
    f = lambda _, y: np.prod(np.asarray(y)[None, :] ** np.array(p.EXPS), axis=1) @ C
    return p.solve_ivp(f, (p.T0, p.T1), p.X0, t_eval=t, **p.SIM_SOLVER).y.T


def paper_figure(start, t, x_true):
    """fig_rivals_half.pdf, as in the paper (2.3 x 1.5 inches, 600 dpi)"""
    path = "results/scores_sigma2e-07_stage1_exhaustive.csv"
    z = {S: v for S, v in scores(path).items() if len(S) == 7} if os.path.exists(path) else {}
    if len(z) != 77280:
        print(f"  fig_rivals_half.pdf skipped: {path} holds {len(z)} of the 77,280 seven-term supports")
        return
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "font.family": "serif",
                         "mathtext.fontset": "cm", "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(figsize=(2.3, 1.5))
    for i in range(2):
        ax.plot(t, x_true[:, i], color="tab:orange", lw=3, alpha=0.5)
    for S, (col, st) in zip(sorted(z, key=lambda S: -z[S])[:4], [("k", "-"), ("tab:blue", "--"), ("tab:green", "-."),
                                                              ("tab:red", ":")]):
        x = solution(S, 2e-7, start, t)
        for i in range(2):
            ax.plot(t, x[:, i], st, color=col, lw=1, label=f"${p.label(S)}$, $\\log Z={z[S]:.2f}$" if i == 0 else None)
    ax.text(0.6, 0.93, "$u$", fontsize=8)
    ax.text(0.6, 0.55, "$v$", fontsize=8)
    ax.axvspan(10, 20, color="0.92", zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=6, handlelength=2.0, borderaxespad=0.2)
    fig.tight_layout(pad=0.2)
    fig.savefig("results/fig_rivals_half.pdf", dpi=600)
    plt.close(fig)
    plt.rcdefaults()
    print("  written results/fig_rivals_half.pdf")


def panel(run, t, x_true, pinn, ranked=()):
    """The solution panel: data and initial state (dots), the true solution (thick), the PINN (black) and the ranked
    supports' solutions; the observed window shaded"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, s in enumerate("uv"):
        ax.scatter(np.r_[p.T0, p.T_OBS], np.r_[p.X0[i], p.Y[:, i]], s=40, color=COLORS[i], edgecolor="black",
                   linewidths=0.5, alpha=0.75, label=f"{s} data", zorder=3)
        ax.plot(t, x_true[:, i], "-", color=COLORS[i], lw=5, alpha=0.75, label=f"{s} true", zorder=1)
        ax.plot(t, pinn[:, i], ":" if ranked or i == 0 else "--", color="black", lw=2, label=f"{s} PINN", zorder=5)
        for (rank, x), st, col in zip(ranked, ["--", "-."], ["tab:red", "tab:green"]):
            ax.plot(t, x[:, i], st, color=col, lw=2, label=f"{s} rank {rank}", zorder=4)
    ax.axvspan(10, 20, color="gray", alpha=0.13, zorder=0)
    ax.set_xlabel("time", fontsize=15)
    ax.set_ylabel("state", fontsize=15)
    h, lab = ax.get_legend_handles_labels()
    order = [k for s in "uv" for k, x in enumerate(lab) if x.startswith(f"{s} ")]
    ax.legend([h[k] for k in order], [lab[k] for k in order], fontsize=12, ncol=2)
    fig.tight_layout()
    name = f"results/solution_{run}{'_with_ranks' if ranked else ''}.pdf"
    fig.savefig(name, dpi=300)
    plt.close(fig)
    print(f"  written {name}")


def main():
    """The paper's figure, then the panels of every final model."""
    model = p.PINN()
    model.load_state_dict(torch.load("results/stage1.pt"))
    start = p.blocks(model)
    t = np.linspace(p.T0, p.T1, 2001)
    x_true = p.solve_ivp(p.rhs, (p.T0, p.T1), p.X0, t_eval=t, **p.DATA_SOLVER).y.T
    paper_figure(start, t, x_true)
    for f in sorted(glob.glob("results/final_*.npz")):
        r = np.load(f)
        run, final, sigma = os.path.basename(f)[6:-4], frozenset(r["support"].tolist()), float(r["sigma"])
        panel(run, t, x_true, r["pinn"])
        z = scores(f"results/scores_sigma{sigma:g}_stage1_climb.csv")
        others = sorted((S for S in z if S != final and np.isfinite(z[S])), key=lambda S: -z[S])[:2]
        above = sum(z[S] > z.get(final, float(r["log_evidence"])) for S in others)
        ranked = [(j + 1 + (j >= above), solution(S, sigma, start, t)) for j, S in enumerate(others)]
        for (rank, _), S in zip(ranked, others):
            print(f"    rank {rank}: {p.label(S)}, log Z {z[S]:.2f}: " + "; ".join(p.MECH[m] for m in sorted(S)))
        panel(run, t, x_true, r["pinn"], ranked)


if __name__ == "__main__":
    main()
