"""Experiment 1: does global selectivity alone predict the cheapest filtered-kNN strategy?

Strategies per query:  BF (pre-filter + brute force, exact)  and  HNSW(ef) for several ef.
Cost = number of distance computations (hardware-independent).
A strategy is *acceptable* for a query if recall@k >= RECALL_OK.

Planners compared on held-out TEST queries (all tuned only on DEV queries):
  * s-planner:   knows only global selectivity s; for each s picks the single strategy
                 with lowest mean dev cost among those with mean dev recall >= RECALL_PLAN.
  * mode-planner: also knows the correlation mode (an upper bound for any planner
                 that estimates correlation, i.e. the prize available to hypothesis H2).
  * oracle:      per-query cheapest acceptable strategy (unattainable lower bound).

Usage:  python experiments/exp1_h1.py --tag base
        python experiments/exp1_h1.py --tag overlap --sigma 1.2
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.bruteforce import LabelIndex, bruteforce_knn  # noqa: E402
from fknn.data import MODES, make_dataset, make_queries  # noqa: E402
from fknn.hnsw_build import build  # noqa: E402
from fknn.hnsw_replica import search  # noqa: E402

SELECTIVITIES = [0.001, 0.005, 0.01, 0.05, 0.2, 0.5]
EFS = [16, 64, 256]
K = 10
RECALL_OK = 0.9      # per-query acceptability
RECALL_PLAN = 0.95   # mean recall a planner must guarantee on dev
DEV_FRACTION = 0.7

G = {}  # filled in parent before fork; read-only in workers


def run_cell(args):
    mode, s, seed, n_queries = args
    ds, li, g, Xint, idx = G["ds"], G["li"], G["g"], G["Xint"], G["idx"]
    rows = []
    for qi, (q, P) in enumerate(make_queries(ds, n_queries, mode, s, seed)):
        ids = li.ids(P)
        t = time.perf_counter()
        exact, _, n_bf = bruteforce_knn(ds.X, q, ids, K)
        t_bf = time.perf_counter() - t
        exact = set(exact.tolist())
        allowed = np.zeros(len(ds.X), dtype=bool)
        allowed[ids] = True
        allowed_int = allowed[g.ext_label]
        row = dict(mode=mode, s=s, qi=qi, split="dev" if qi < DEV_FRACTION * n_queries else "test",
                   n_allowed=len(ids), cost_BF=n_bf, recall_BF=1.0, time_BF=t_bf)
        for ef in EFS:
            got, _, nd = search(g, Xint, q, K, ef, allowed_int)
            row[f"cost_HNSW{ef}"] = nd
            row[f"recall_HNSW{ef}"] = len(exact & set(got.tolist())) / K
        # real hnswlib wall time (single thread, python filter callback) for ef=64
        idx.set_ef(64)
        t = time.perf_counter()
        idx.knn_query(q[None], k=K, filter=lambda i: bool(allowed[i]))
        row["time_hnswlib64"] = time.perf_counter() - t
        rows.append(row)
    return rows


def strategies():
    return ["BF"] + [f"HNSW{ef}" for ef in EFS]


def analyze(rows):
    S = strategies()
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    def choose(subset):
        best, best_cost = "BF", np.mean([r["cost_BF"] for r in subset])
        for st in S[1:]:
            rec = np.mean([r[f"recall_{st}"] for r in subset])
            c = np.mean([r[f"cost_{st}"] for r in subset])
            if rec >= RECALL_PLAN and c < best_cost:
                best, best_cost = st, c
        return best

    s_plan = {s: choose([r for r in dev if r["s"] == s]) for s in SELECTIVITIES}
    m_plan = {(m, s): choose([r for r in dev if r["s"] == s and r["mode"] == m])
              for m in MODES for s in SELECTIVITIES}

    def oracle(r):
        ok = [st for st in S if r[f"recall_{st}"] >= RECALL_OK]
        return min(ok, key=lambda st: r[f"cost_{st}"])

    def evaluate(pick):
        cost = np.array([r[f"cost_{pick(r)}"] for r in test], dtype=float)
        rec = np.array([r[f"recall_{pick(r)}"] for r in test])
        orc = np.array([r[f"cost_{oracle(r)}"] for r in test], dtype=float)
        bad = (cost > orc) | (rec < RECALL_OK)
        costly = (cost >= 2 * orc) | (rec < RECALL_OK)
        return dict(total_cost=float(cost.sum()), mean_recall=float(rec.mean()),
                    frac_not_optimal=float(bad.mean()),
                    frac_costly_mistake=float(costly.mean()),
                    mean_cost_ratio_on_mistakes=float((cost[bad] / orc[bad]).mean()) if bad.any() else 1.0)

    res = {
        "s_planner_choices": {str(k): v for k, v in s_plan.items()},
        "mode_planner_choices": {f"{m}|{s}": v for (m, s), v in m_plan.items()},
        "s_planner": evaluate(lambda r: s_plan[r["s"]]),
        "mode_planner": evaluate(lambda r: m_plan[(r["mode"], r["s"])]),
        "oracle": evaluate(oracle),
        "always_BF": evaluate(lambda r: "BF"),
        "always_HNSW64": evaluate(lambda r: "HNSW64"),
        "n_test_queries": len(test), "n_dev_queries": len(dev),
    }
    res["prize_for_H2_cost_ratio_s_over_mode"] = res["s_planner"]["total_cost"] / res["mode_planner"]["total_cost"]
    sp = res["s_planner"]
    res["H1_supported"] = bool(sp["frac_not_optimal"] >= 0.20 and sp["mean_cost_ratio_on_mistakes"] >= 2.0)
    # per-cell table
    cells = []
    for m in MODES:
        for s in SELECTIVITIES:
            sub = [r for r in rows if r["mode"] == m and r["s"] == s]
            c = dict(mode=m, s=s)
            for st in S:
                c[f"cost_{st}"] = float(np.mean([r[f"cost_{st}"] for r in sub]))
                c[f"recall_{st}"] = float(np.mean([r[f"recall_{st}"] for r in sub]))
            c["time_BF_ms"] = 1e3 * float(np.median([r["time_BF"] for r in sub]))
            c["time_hnswlib64_ms"] = 1e3 * float(np.median([r["time_hnswlib64"] for r in sub]))
            cells.append(c)
    res["cells"] = cells
    return res


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"BF": "#2a78d6", "HNSW16": "#eb6834", "HNSW64": "#1baf7a", "HNSW256": "#eda100"}
    markers = {"BF": "o", "HNSW16": "s", "HNSW64": "^", "HNSW256": "D"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, facecolor="#fcfcfb")
    for j, m in enumerate(MODES):
        cells = [c for c in res["cells"] if c["mode"] == m]
        xs = [c["s"] for c in cells]
        for row, key, ylab in ((0, "cost", "distance computations / query"), (1, "recall", "recall@10")):
            ax = axes[row, j]
            ax.set_facecolor("#fcfcfb")
            for st in strategies():
                ax.plot(xs, [c[f"{key}_{st}"] for c in cells], color=colors[st], lw=2,
                        marker=markers[st], ms=8, mec="#fcfcfb", mew=2, label=st)
            ax.set_xscale("log")
            if key == "cost":
                ax.set_yscale("log")
                ax.set_title(f"{m} correlation", color=ink, fontsize=12)
            else:
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("global selectivity s", color=ink2)
            if j == 0:
                ax.set_ylabel(ylab, color=ink2)
            ax.grid(True, color=grid, lw=0.8)
            for sp in ax.spines.values():
                sp.set_color(grid)
            ax.tick_params(colors=ink2)
    axes[0, 0].legend(frameon=False, labelcolor=ink)
    fig.suptitle("Same selectivity, different correlation: cost of BF vs HNSW (mean over queries)",
                 color=ink, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--n-clusters", type=int, default=50)
    ap.add_argument("--per-cluster", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--queries", type=int, default=100, help="per (mode, s) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = os.path.join("results", f"exp1_{a.tag}")
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    ds = make_dataset(n_clusters=a.n_clusters, per_cluster=a.per_cluster, dim=a.dim,
                      sigma=a.sigma, seed=a.seed)
    idx, g, Xint = build(ds.X, os.path.join("results", "cache", f"{a.tag}.bin"))
    G.update(ds=ds, li=LabelIndex(ds.labels, ds.n_labels), g=g, Xint=Xint, idx=idx)
    print(f"built index in {time.time() - t0:.1f}s", flush=True)

    jobs = [(m, s, 1000 + 97 * i + 13 * j + a.seed, a.queries)
            for i, m in enumerate(MODES) for j, s in enumerate(SELECTIVITIES)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = [r for cell in pool.imap_unordered(run_cell, jobs) for r in cell]
    rows.sort(key=lambda r: (MODES.index(r["mode"]), r["s"], r["qi"]))

    with open(os.path.join(out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = analyze(rows)
    res["config"] = dict(vars(a), selectivities=SELECTIVITIES, efs=EFS, k=K, recall_ok=RECALL_OK,
                         recall_plan=RECALL_PLAN, hnsw=dict(M=16, ef_construction=200, seed=100),
                         python=platform.python_version(), numpy=np.__version__,
                         runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(res, os.path.join(out, "cost_recall.png"))
    for k in ("s_planner", "mode_planner", "oracle", "always_BF", "always_HNSW64"):
        print(k, json.dumps(res[k]))
    print("prize s/mode:", round(res["prize_for_H2_cost_ratio_s_over_mode"], 2),
          "H1_supported:", res["H1_supported"], f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
