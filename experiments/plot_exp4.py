"""Plot Experiment 4 recall-vs-cost curves (real datasets, by selectivity band)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"BF": "#2a78d6", "HNSW": "#eb6834", "A2H": "#1baf7a", "S2H": "#eda100"}
MK = {"BF": "o", "HNSW": "s", "A2H": "^", "S2H": "D"}
LAB = {"BF": "BF (exact)", "HNSW": "HNSW filtered (hnswlib)", "A2H": "2-hop, no seeds (ACORN-like)",
       "S2H": "2-hop + cluster seeds (S2H)"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BANDS = ["s<1%", "1%<=s<10%", "s>=10%"]


def main(datasets=("fashion", "glove"), out="results/exp4/curves_real.png"):
    fig, axes = plt.subplots(len(datasets), 3, figsize=(13, 4.2 * len(datasets)), facecolor=BG)
    for i, ds in enumerate(datasets):
        S = json.load(open(os.path.join("results", "exp4", ds, "summary.json")))
        for j, b in enumerate(BANDS):
            ax = axes[i, j]
            ax.set_facecolor(BG)
            g = S["groups"][b]
            for name, pts in g["curves"].items():
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=COL[name], marker=MK[name], lw=2, ms=8, mec=BG, mew=2,
                        label=LAB[name])
            ax.axhline(0.95, color=INK2, lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(0.3, 1.02)
            ax.set_title(f"{ds}: {b}  (n={g['n']})", color=INK, fontsize=11)
            ax.grid(True, color=GRID, lw=0.8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.tick_params(colors=INK2)
            if j == 0:
                ax.set_ylabel("mean recall@10", color=INK2)
            if i == len(datasets) - 1:
                ax.set_xlabel("mean cost / query (distances + w·edges), log", color=INK2)
    axes[0, 0].legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right")
    fig.suptitle("Recall vs cost by selectivity band (up-left is better; dotted line = 0.95)",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
