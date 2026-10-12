"""Experiment 17: exact top-k MIPS on REAL attention keys/queries (RoBERTa-base, see exp16).
Database = keys of one (layer, head) from ~200 chunks of 512 tokens; queries = that head's
queries on held-out chunks (128 sampled). Same baselines as Experiment 15.
Usage: OPENBLAS_NUM_THREADS=1 python experiments/exp17_mips_real_attention.py"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from exp15_mips import bench_one  # noqa: E402


def main():
    z = np.load("results/cache/data/attn_keys.npz")
    rng = np.random.default_rng(0)
    out = {}
    for name in sorted(k[2:] for k in z.files if k.startswith("K_")):
        K, Q = z["K_" + name], z["Q_" + name]
        Q = Q[rng.choice(len(Q), 128, replace=False)]
        kn = np.linalg.norm(K, axis=1)
        print(f"{name}: keys {K.shape}, key-norm p5/p50/p95 {np.percentile(kn, [5, 50, 95]).round(2)}", flush=True)
        out[name] = bench_one("attn_" + name, K, Q)
    os.makedirs("results/exp17", exist_ok=True)
    json.dump(out, open("results/exp17/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
