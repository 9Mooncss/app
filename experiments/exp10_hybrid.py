"""Experiment 10: hybrid planner = exact VA3 below a selectivity threshold, ACORN-gamma above.

Per-query times: VA3/BF measured here in C; ACORN-gamma per query from Experiment 7
(results/exp7_perquery/<ds>_ACORN-gamma.csv.gz, same queries, same machine).
Queries are split by index parity: even = dev, odd = test. On dev we choose
(threshold s*, ACORN ef) minimising mean time subject to mean recall >= TARGET
(VA3 recall = 1). ACORN-only is tuned the same way (s* = 0). Report on test.

Usage: python experiments/exp10_hybrid.py
"""
import csv
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exp5_exact_scan import workload  # noqa: E402
from fknn.fastscan import FastScanIndex  # noqa: E402

TARGET = 0.95


def load_acorn(ds, nq):
    T, R = {}, {}
    with gzip.open(f"results/exp7_perquery/{ds}_ACORN-gamma.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            ef = int(r["ef"])
            T.setdefault(ef, np.zeros(nq))[int(r["query"])] = float(r["time_us"])
            R.setdefault(ef, np.zeros(nq))[int(r["query"])] = float(r["recall"])
    return T, R


def evaluate(mask, s, tv, T, R, thr, ef):
    use_va = s < thr
    t = np.where(use_va, tv, T[ef])[mask]
    r = np.where(use_va, 1.0, R[ef])[mask]
    return float(t.mean()), float(r.mean()), float(np.percentile(r, 10)), float(use_va[mask].mean())


def tune(mask, s, tv, T, R, thresholds):
    best = None
    for thr in thresholds:
        for ef in sorted(T):
            t, r, _, _ = evaluate(mask, s, tv, T, R, thr, ef)
            if r >= TARGET and (best is None or t < best[0]):
                best = (t, thr, ef)
    return best


def main():
    out = {}
    for ds in ["fashion", "glove", "spacy"]:
        X, labels, L, qs = workload(ds)
        fs = FastScanIndex(X, labels, L, bits3=6)
        Q = np.stack([q for q, _, _ in qs]); Ps = [P for _, P, _ in qs]
        tv = np.full(len(qs), np.inf); tb = np.full(len(qs), np.inf)
        for _ in range(3):
            _, _, t, nb = fs.bf(Q, Ps); tb = np.minimum(tb, t)
            _, _, t, _, _ = fs.va3(Q, Ps); tv = np.minimum(tv, t)
        tv, tb = tv / 1e3, tb / 1e3
        s = nb / len(X)
        T, R = load_acorn(ds, len(qs))
        dev = np.arange(len(qs)) % 2 == 0
        test = ~dev
        thresholds = [0.0] + sorted(set(np.quantile(s[dev], np.linspace(0.05, 1, 20)).tolist())) + [1.01]
        h = tune(dev, s, tv, T, R, thresholds)
        a = tune(dev, s, tv, T, R, [0.0])
        res = {"n_test": int(test.sum())}
        res["BF_exact"] = {"mean_us": float(tb[test].mean()), "recall": 1.0}
        res["VA3_exact"] = {"mean_us": float(tv[test].mean()), "recall": 1.0}
        if a:
            t, r, p10, _ = evaluate(test, s, tv, T, R, 0.0, a[2])
            res["ACORN_gamma_only"] = {"ef": a[2], "mean_us": t, "recall": r, "p10_recall": p10}
        else:
            res["ACORN_gamma_only"] = {"note": f"no ef reaches mean recall {TARGET} on dev"}
        t, r, p10, frac = evaluate(test, s, tv, T, R, h[1], h[2])
        res["hybrid"] = {"threshold_s": h[1], "ef": h[2], "mean_us": t, "recall": r, "p10_recall": p10, "frac_to_VA3": frac}
        out[ds] = res
        print(ds, json.dumps(res), flush=True)
    os.makedirs("results/exp10", exist_ok=True)
    json.dump(out, open("results/exp10/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
