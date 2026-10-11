"""Summary: exact VA3 fast-scan (exp9) vs ACORN-gamma / ACORN-1 (exp7) on identical queries.
Writes results/summary_vs_acorn.md and results/summary_vs_acorn.png."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DS = ["fashion", "glove", "spacy", "synth_base", "synth_overlap"]
BANDS = ["s<1%", "1%<=s<10%", "s>=10%"]


def main():
    ex = json.load(open("results/exp9/summary.json"))
    lines = ["| dataset | band | BF (exact) | **VA3 (exact)** | ACORN-γ @0.90 | ACORN-γ @0.95 | ACORN-γ @0.99 | ACORN-1 @0.95 | VA3 vs ACORN-γ@0.95 |",
             "|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(DS), figsize=(16, 4), facecolor="#fcfcfb", sharey=False)
    for j, ds in enumerate(DS):
        ac = json.load(open(f"results/exp7/{ds}.json"))["groups"]
        xs, va, a95 = [], [], []
        for b in BANDS:
            e = ex[ds]["groups"][b]
            g = ac[b]
            ag = g.get("ACORN-gamma", {})
            a1 = g.get("ACORN-1", {})
            f = lambda v: "never" if v is None else f"{v:.0f}"  # noqa: E731
            t95 = ag.get("us_at_0.95")
            ratio = "∞ (ACORN never reaches 0.95)" if t95 is None else f"{t95 / e['va2_mean_us']:.2f}×"
            lines.append(f"| {ds} | {b} | {e['bf_mean_us']:.0f} | **{e['va2_mean_us']:.0f}** | {f(ag.get('us_at_0.9'))} | "
                         f"{f(t95)} | {f(ag.get('us_at_0.99'))} | {f(a1.get('us_at_0.95'))} | {ratio} |")
            xs.append(b); va.append(e["va2_mean_us"]); a95.append(t95)
        ax = axes[j]
        ax.set_facecolor("#fcfcfb")
        x = range(len(xs))
        ax.bar([i - 0.2 for i in x], va, width=0.38, color="#2a78d6", label="VA3 exact (recall 1.0)")
        ax.bar([i + 0.2 for i in x], [v if v is not None else 0 for v in a95], width=0.38, color="#eb6834",
               label="ACORN-γ at recall ≥ 0.95")
        for i, v in enumerate(a95):
            if v is None:
                ax.text(i + 0.2, max(va) * 0.05, "never\nreaches\n0.95", ha="center", fontsize=8, color="#52514e")
        ax.set_xticks(list(x)); ax.set_xticklabels(xs, fontsize=8, color="#52514e")
        ax.set_title(ds, color="#0b0b0b"); ax.set_yscale("log")
        ax.grid(True, axis="y", color="#e4e3df")
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
        if j == 0:
            ax.set_ylabel("mean µs / query (1 thread)", color="#52514e")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Exact filtered search (VA3) vs ACORN-γ (approximate) — lower is better", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig("results/summary_vs_acorn.png", dpi=130)
    open("results/summary_vs_acorn.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
