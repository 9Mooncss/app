"""Experiment 12: kernel-vs-kernel without filters (s = 100%): exact VA3 vs PDX pruned
exhaustive search vs contiguous brute force. Removes VA3's filter-layout advantage.
Usage: python experiments/exp12_unfiltered_kernel.py"""
import json
import os
import sys
import time

import numpy as np
from pdxearch import IndexPDXIVF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.fastscan import FastScanIndex  # noqa: E402
from fknn.realdata import load_fashion, load_glove, load_spacy  # noqa: E402

K, NQ = 10, 100


def main():
    out = {}
    for name, loader in (("fashion", load_fashion), ("glove", load_glove), ("spacy", load_spacy)):
        rd = loader()
        X = rd.X
        n, d = X.shape
        labels = np.zeros(n, np.int32)            # one label = everything
        fs = FastScanIndex(X, labels, 1, bits3=6)
        Q = np.ascontiguousarray(rd.Q[np.random.default_rng(1).choice(len(rd.Q), NQ, replace=False)])
        Ps = [np.array([0])] * NQ
        tv = np.full(NQ, np.inf); tb = np.full(NQ, np.inf); t4 = np.full(NQ, np.inf)
        for _ in range(3):
            eid, _, t, _ = fs.bf(Q, Ps); tb = np.minimum(tb, t)
            _, _, t, nr, _ = fs.va3(Q, Ps); tv = np.minimum(tv, t)
            _, _, t, _, nd4 = fs.va4(Q, Ps); t4 = np.minimum(t4, t)
        idx = IndexPDXIVF(num_dimensions=d, normalize=False, n_threads=4)
        idx.build(np.ascontiguousarray(X, np.float32))
        tp = np.zeros(NQ); rp = np.zeros(NQ)
        for i in range(NQ):
            best = np.inf
            for _ in range(2):
                a = time.perf_counter(); ids, _ = idx.search(Q[i], K, nprobe=idx.num_clusters); best = min(best, time.perf_counter() - a)
            tp[i] = best * 1e6
            rp[i] = len(set(np.asarray(ids).ravel().tolist()) & set(eid[i].tolist())) / K
        r = {"n": n, "d": d, "bf_us": float(tb.mean() / 1e3), "va3_us": float(tv.mean() / 1e3),
             "va3_refined_frac": float(nr.mean() / n), "va4_us": float(t4.mean() / 1e3),
             "va4_dims_scanned_frac": float(nd4.sum() / (NQ * d * ((n + 63) // 64))), "pdx_exhaustive_us": float(tp.mean()),
             "pdx_recall": float(rp.mean())}
        out[name] = r
        print(name, json.dumps(r), flush=True)
    os.makedirs("results/exp12", exist_ok=True)
    json.dump(out, open("results/exp12/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
