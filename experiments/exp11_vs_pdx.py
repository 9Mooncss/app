"""Experiment 11: exact VA3 vs PDX (SIGMOD'25; github.com/cwida/PDX), filtered search.

PDX IndexPDXIVF (float32, normalize=False, i.e. plain L2), filtered_search with the
allowed row ids. "exhaustive" = nprobe = all clusters (PDX's pruned exhaustive search;
its ADSampling pruning is probabilistic, so recall is measured, not assumed);
"default" = nprobe 32 (approximate). Single-thread search, timed per query from Python
(min of 2), Python call overhead measured on a trivial index and subtracted.

Usage: python experiments/exp11_vs_pdx.py [datasets]
"""
import json
import os
import sys
import time

import numpy as np
from pdxearch import IndexPDXIVF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exp5_exact_scan import band, workload  # noqa: E402
from fknn.fastscan import FastScanIndex  # noqa: E402

K = 10


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fashion", "glove", "spacy", "synth_big"]
    out = {}
    for name in names:
        X, labels, L, qs = workload(name)
        n, d = X.shape
        fs = FastScanIndex(X, labels, L, bits3=6)
        Q = np.stack([q for q, _, _ in qs]); Ps = [P for _, P, _ in qs]
        tv = np.full(len(qs), np.inf); tb = np.full(len(qs), np.inf)
        for _ in range(3):
            eid, ed, t, nb = fs.bf(Q, Ps); tb = np.minimum(tb, t)
            _, _, t, _, _ = fs.va3(Q, Ps); tv = np.minimum(tv, t)
        tv, tb = tv / 1e3, tb / 1e3
        t0 = time.time()
        idx = IndexPDXIVF(num_dimensions=d, normalize=False, n_threads=4)
        idx.build(np.ascontiguousarray(X, np.float32))
        build = time.time() - t0
        ncl = idx.num_clusters
        tiny = IndexPDXIVF(num_dimensions=d, normalize=False, n_threads=1)
        tiny.build(np.random.default_rng(0).normal(size=(256, d)).astype(np.float32))
        ov = []
        for _ in range(300):
            a = time.perf_counter(); tiny.filtered_search(Q[0], K, np.arange(10, dtype=np.uint64), nprobe=1); ov.append(time.perf_counter() - a)
        ovh = float(np.median(ov)) * 1e6
        label_sorted = np.argsort(labels, kind="stable")
        lstart = np.concatenate([[0], np.cumsum(np.bincount(labels, minlength=L))])
        rows = [np.concatenate([label_sorted[lstart[l]:lstart[l + 1]] for l in P]).astype(np.uint64) for P in Ps]
        res = {"n": n, "d": d, "pdx_build_s": round(build, 1), "pdx_clusters": ncl, "pdx_call_overhead_us": ovh, "groups": {}}
        modes = {"exhaustive": ncl, "nprobe32": 32}
        tp = {m: np.zeros(len(qs)) for m in modes}; rp = {m: np.zeros(len(qs)) for m in modes}
        for m, npb in modes.items():
            for i in range(len(qs)):
                best = np.inf
                for _ in range(2):
                    a = time.perf_counter(); ids, _ = idx.filtered_search(Q[i], K, rows[i], nprobe=npb); best = min(best, time.perf_counter() - a)
                tp[m][i] = max(best * 1e6 - ovh, 0)
                rp[m][i] = len(set(np.asarray(ids).ravel().tolist()) & set(eid[i].tolist())) / K
        s = nb / n
        for g in ("all", "s<1%", "1%<=s<10%", "s>=10%"):
            mk = np.ones(len(qs), bool) if g == "all" else np.array([band(x) == g for x in s])
            r = {"n": int(mk.sum()), "bf_us": float(tb[mk].mean()), "va3_us": float(tv[mk].mean())}
            for m in modes:
                r[f"pdx_{m}_us"] = float(tp[m][mk].mean()); r[f"pdx_{m}_recall"] = float(rp[m][mk].mean())
            res["groups"][g] = r
            print(f"[{name}/{g}] n={r['n']} BF {r['bf_us']:.0f} | VA3 exact {r['va3_us']:.0f} | "
                  f"PDX exhaustive {r['pdx_exhaustive_us']:.0f} (recall {r['pdx_exhaustive_recall']:.3f}) | "
                  f"PDX nprobe32 {r['pdx_nprobe32_us']:.0f} (recall {r['pdx_nprobe32_recall']:.3f})", flush=True)
        out[name] = res
    os.makedirs("results/exp11", exist_ok=True)
    json.dump(out, open("results/exp11/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
