"""Experiment 4: recall-matched cost curves for each strategy (no planner).

Strategies, each swept over ef:
  HNSW  - hnswlib filtered search (exact replica)
  A2H   - allowed-only 2-hop traversal from the usual entry point (ACORN-like ablation)
  S2H   - the same traversal from cluster x label seeds (our variant)
  BF    - exact baseline
Cost = distance computations + w * adjacency entries scanned (w from bench/edge_vs_dist.c;
HNSW's own adjacency reads are NOT charged, which favours HNSW).
Results are broken down by selectivity band and (for synthetic data) correlation mode.

Usage: python experiments/exp4_curves.py --datasets synth_base,fashion,glove
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.bruteforce import LabelIndex, bruteforce_knn  # noqa: E402
from fknn.data import ALL_MODES, make_dataset, make_queries  # noqa: E402
from fknn.hnsw_build import build  # noqa: E402
from fknn.hnsw_replica import search  # noqa: E402
from fknn.ivf_exact import IVFExact  # noqa: E402
from fknn.planner import LocalSelectivity  # noqa: E402
from fknn.realdata import load_fashion, load_glove, make_real_queries  # noqa: E402
from fknn.seeded import SeedPicker, acorn_like_search, seeded_2hop_search  # noqa: E402

K = 10
HNSW_EFS = [10, 16, 32, 64, 128, 256]
GRAPH_EFS = [10, 16, 32, 64, 128]
W = {"synth_base": 0.067, "synth_overlap": 0.067, "fashion": 0.003, "glove": 0.021}  # worst measured
SEL = [0.001, 0.005, 0.01, 0.05, 0.2, 0.5]
G = {}


def run_query(task):
    q, P, meta = task
    X, li, g, Xint, sp = G["X"], G["li"], G["g"], G["Xint"], G["sp"]
    ids = li.ids(P)
    exact = set(bruteforce_knn(X, q, ids, K)[0].tolist())
    allowed = np.zeros(len(X), dtype=bool)
    allowed[ids] = True
    ai = allowed[g.ext_label]
    row = dict(meta, n_allowed=len(ids), s=len(ids) / len(X), cost_BF=len(ids))
    for ef in HNSW_EFS:
        got, _, nd = search(g, Xint, q, K, ef, ai)
        row[f"HNSW{ef}_d"], row[f"HNSW{ef}_e"] = nd, 0
        row[f"HNSW{ef}_r"] = len(exact & set(got.tolist())) / K
    seeds, c0 = sp.seeds(q, P, 32, per_cluster=2)
    for ef in GRAPH_EFS:
        got, _, nd, ne = seeded_2hop_search(g, Xint, q, K, ef, ai, seeds)
        row[f"S2H{ef}_d"], row[f"S2H{ef}_e"] = nd + c0, ne
        row[f"S2H{ef}_r"] = len(exact & set(got.tolist())) / K
        got, _, nd, ne = acorn_like_search(g, Xint, q, K, ef, ai)
        row[f"A2H{ef}_d"], row[f"A2H{ef}_e"] = nd, ne
        row[f"A2H{ef}_r"] = len(exact & set(got.tolist())) / K
    return row


def tasks_for(name, n_per, seed):
    if name.startswith("synth"):
        ds = make_dataset(sigma=0.6 if name == "synth_base" else 1.2, seed=seed)
        t = [(q, P, dict(mode=m)) for i, m in enumerate(ALL_MODES) for j, s in enumerate(SEL)
             for q, P in make_queries(ds, n_per, m, s, 20000 + 31 * i + 7 * j + seed)]
        return ds.X, ds.labels, ds.n_labels, t
    rd = load_fashion(seed) if name == "fashion" else load_glove()
    t = [(q, P, dict(mode="natural")) for q, P in make_real_queries(rd, n_per * 24, 30000 + seed)]
    return rd.X, rd.labels, rd.n_labels, t


def band(s):
    return "s<1%" if s < 0.01 else ("1%<=s<10%" if s < 0.1 else "s>=10%")


def curves(rows, w):
    out = {"BF": [(float(np.mean([r["cost_BF"] for r in rows])), 1.0)]}
    for name, efs in (("HNSW", HNSW_EFS), ("A2H", GRAPH_EFS), ("S2H", GRAPH_EFS)):
        out[name] = [(float(np.mean([r[f"{name}{ef}_d"] + w * r[f"{name}{ef}_e"] for r in rows])),
                      float(np.mean([r[f"{name}{ef}_r"] for r in rows]))) for ef in efs]
    return out


def cheapest_at(curve, target):
    ok = [c for c, r in curve if r >= target]
    return min(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synth_base,synth_overlap,fashion,glove")
    ap.add_argument("--n-per", type=int, default=10, help="queries per (mode, s) cell; real: x24")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for name in a.datasets.split(","):
        t0 = time.time()
        X, labels, L, tasks = tasks_for(name, a.n_per, a.seed)
        _, g, Xint = build(X, os.path.join("results", "cache", f"exp2_{name}_{a.seed}.bin"))
        est = LocalSelectivity(X, labels, L, n_centroids=64, seed=a.seed)
        ivf = IVFExact(X, labels, L, est.C, est._assign(X, est.C))
        e2i = np.empty(len(X), np.int64)
        e2i[g.ext_label] = np.arange(len(X))
        G.update(X=X, li=LabelIndex(labels, L), g=g, Xint=Xint, sp=SeedPicker(ivf, e2i))
        with mp.get_context("fork").Pool(a.workers) as pool:
            rows = pool.map(run_query, tasks, chunksize=4)
        out = os.path.join("results", "exp4", name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "raw.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        w = W[name]
        groups = {"all": rows}
        for b in ("s<1%", "1%<=s<10%", "s>=10%"):
            groups[b] = [r for r in rows if band(r["s"]) == b]
        if name.startswith("synth"):
            for m in ALL_MODES:
                groups[m] = [r for r in rows if r["mode"] == m]
        summary = {"edge_weight": w, "n_queries": len(rows), "groups": {}}
        for gname, sub in groups.items():
            if not sub:
                continue
            cv = curves(sub, w)
            at = {t: {k: cheapest_at(v, t) for k, v in cv.items()} for t in (0.9, 0.95, 0.99)}
            summary["groups"][gname] = {"n": len(sub), "curves": cv, "cheapest_cost_at_recall": at}
            a95 = at[0.95]
            fmt = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
            print(f"[{name}/{gname}] n={len(sub)} cost@recall>=0.95: " +
                  " ".join(f"{k}={fmt(v)}" for k, v in a95.items()), flush=True)
        summary["runtime_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
