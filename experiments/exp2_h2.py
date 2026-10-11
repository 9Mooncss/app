"""Experiment 2: does a local-selectivity feature beat the strongest s-only planner?

Planners (same k-NN learner, tuned only on DEV queries, scored on TEST queries):
  A  s-only            features: log n_allowed         strategies: BF, HNSW(ef)
  B  s-only + fallback features: log n_allowed         strategies: + budgeted HNSW -> BF
  C  local             features: + local selectivity    strategies: BF, HNSW(ef)
  D  local + fallback  features: + local selectivity    strategies: all
C and D pay the estimator overhead (n_centroids distance computations) on every query.

Datasets:
  synth_base / synth_overlap: dev = 3 correlation modes; tests = held-out same modes
                              ("grid") AND an unseen "mixed" mode (out-of-distribution).
  fashion / glove:            real data, random filters, 70% dev / 30% test.

Usage: python experiments/exp2_h2.py [--datasets synth_base,synth_overlap,fashion,glove]
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
from fknn.planner import (KNNPlanner, LocalSelectivity, evaluate, oracle_pick,  # noqa: E402
                          strategy_table)
from fknn.realdata import load_fashion, load_glove, make_real_queries  # noqa: E402

SELECTIVITIES = [0.001, 0.005, 0.01, 0.05, 0.2, 0.5]
EFS = [16, 64, 256]
K = 10
G = {}


def run_query(task):
    q, P, meta = task
    X, li, g, Xint, est = G["X"], G["li"], G["g"], G["Xint"], G["est"]
    ids = li.ids(P)
    exact, _, n_bf = bruteforce_knn(X, q, ids, K)
    exact = set(exact.tolist())
    allowed = np.zeros(len(X), dtype=bool)
    allowed[ids] = True
    allowed_int = allowed[g.ext_label]
    row = dict(meta, n_allowed=len(ids), s=len(ids) / len(X), cost_BF=n_bf)
    for ef in EFS:
        got, _, nd = search(g, Xint, q, K, ef, allowed_int)
        row[f"cost_HNSW{ef}"] = nd
        row[f"recall_HNSW{ef}"] = len(exact & set(got.tolist())) / K
    for i, f in enumerate(est.features(q, P)):
        row[f"local{est.ms[i]}"] = f
    return row


def queries_for(name, seed):
    """Return (X, labels, n_labels, dev_tasks, {test_name: tasks})."""
    if name.startswith("synth"):
        sigma = 0.6 if name == "synth_base" else 1.2
        ds = make_dataset(sigma=sigma, seed=seed)

        def grid(modes, n, base):
            out = []
            for i, m in enumerate(modes):
                for j, s in enumerate(SELECTIVITIES):
                    for qi, (q, P) in enumerate(make_queries(ds, n, m, s, base + 97 * i + 13 * j)):
                        out.append((q, P, dict(mode=m)))
            return out
        dev = grid(MODES, 70, 5000 + seed)
        tests = {"grid": grid(MODES, 30, 7000 + seed), "mixed": grid(["mixed"], 90, 9000 + seed)}
        return ds.X, ds.labels, ds.n_labels, dev, tests
    rd = load_fashion(seed) if name == "fashion" else load_glove()
    qs = [(q, P, dict(mode="natural")) for q, P in make_real_queries(rd, 1000, seed + 11)]
    return rd.X, rd.labels, rd.n_labels, qs[:700], {"natural": qs[700:]}


def features(rows, local):
    F = [[np.log10(r["n_allowed"])] + ([r["local1"], r["local4"], r["local16"]] if local else [])
         for r in rows]
    return np.array(F, dtype=float)


PLANNERS = {"A_s_only": (False, False), "B_s_fallback": (False, True),
            "C_local": (True, False), "D_local_fallback": (True, True)}


def analyze(dev, tests, overhead):
    out = {}
    for pname, (local, fb) in PLANNERS.items():
        names, cdev, rdev = strategy_table(dev, EFS, fb)
        pl = KNNPlanner().fit(features(dev, local), cdev, rdev)
        out[pname] = {"k": pl.k}
        for tname, rows in tests.items():
            _, ct, rt = strategy_table(rows, EFS, fb)
            pick = pl.predict(features(rows, local))
            res = evaluate(pick, ct, rt, overhead if local else 0.0)
            res["strategy_use"] = {names[i]: int(c) for i, c in enumerate(np.bincount(pick, minlength=len(names))) if c}
            out[pname][tname] = res
    for tname, rows in tests.items():
        names, ct, rt = strategy_table(rows, EFS, True)
        out.setdefault("oracle", {})[tname] = evaluate(oracle_pick(ct, rt), ct, rt)
        out.setdefault("always_BF", {})[tname] = evaluate(np.zeros(len(rows), int), ct, rt)
        out[f"gain_B_over_D_{tname}"] = out["B_s_fallback"][tname]["total_cost"] / out["D_local_fallback"][tname]["total_cost"]
        out[f"gain_A_over_C_{tname}"] = out["A_s_only"][tname]["total_cost"] / out["C_local"][tname]["total_cost"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synth_base,synth_overlap,fashion,glove")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--centroids", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    summary = {}
    for name in a.datasets.split(","):
        t0 = time.time()
        X, labels, n_labels, dev_tasks, test_tasks = queries_for(name, a.seed)
        idx, g, Xint = build(X, os.path.join("results", "cache", f"exp2_{name}_{a.seed}.bin"))
        est = LocalSelectivity(X, labels, n_labels, n_centroids=a.centroids, seed=a.seed)
        G.update(X=X, li=LabelIndex(labels, n_labels), g=g, Xint=Xint, est=est)
        print(f"[{name}] n={len(X)} d={X.shape[1]} index+estimator {time.time() - t0:.0f}s", flush=True)
        with mp.get_context("fork").Pool(a.workers) as pool:
            dev = pool.map(run_query, dev_tasks, chunksize=8)
            tests = {k: pool.map(run_query, v, chunksize=8) for k, v in test_tasks.items()}
        out = os.path.join("results", f"exp2{a.tag}", name)
        os.makedirs(out, exist_ok=True)
        for split, rows in [("dev", dev)] + list(tests.items()):
            with open(os.path.join(out, f"raw_{split}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        res = analyze(dev, tests, est.cost)
        res["config"] = dict(dataset=name, n=len(X), d=X.shape[1], seed=a.seed, centroids=a.centroids,
                             efs=EFS, k=K, n_dev=len(dev), n_test={k: len(v) for k, v in tests.items()},
                             python=platform.python_version(), numpy=np.__version__,
                             runtime_s=round(time.time() - t0, 1))
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(res, f, indent=2)
        summary[name] = res
        for tname in tests:
            line = " | ".join(f"{p}: {res[p][tname]['mean_cost']:.0f} r{res[p][tname]['mean_recall']:.3f}"
                              for p in list(PLANNERS) + ["oracle", "always_BF"])
            print(f"[{name}/{tname}] {line} | B/D={res[f'gain_B_over_D_{tname}']:.2f}", flush=True)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
