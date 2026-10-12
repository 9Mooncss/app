"""Experiment 15: exact top-k maximum inner product search (MIPS).
VA upper-bound fast scan (fknn/mips.py, 6-bit) vs exact baselines, single thread:
  * BF-C      : contiguous dot products in C (same compiler flags)
  * BLAS-gemv : numpy/OpenBLAS X @ q + argpartition, one query at a time
  * BLAS-gemm : batches of 64 queries (Q @ X.T) - throughput per query, the strongest
                brute force when queries arrive in batches
Datasets: GloVe and spaCy WITHOUT normalisation (norms vary: classic MIPS), synthetic
"recommendation" (Gaussian x log-normal norms), synthetic "attention keys" (power-law
spectrum + 4 outlier channels; queries = noisy keys).
Usage: OPENBLAS_NUM_THREADS=1 python experiments/exp15_mips.py"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.mips import MipsIndex  # noqa: E402
from fknn.realdata import load_glove, load_spacy  # noqa: E402

K, NQ = 10, 128


def datasets():
    rng = np.random.default_rng(0)
    g = load_glove(normalize=False)
    yield "glove_raw", g.X, g.Q[:NQ]
    s = load_spacy(normalize=False)
    yield "spacy_raw", s.X, s.Q[:NQ]
    n, d = 1_000_000, 64
    X = (rng.normal(size=(n, d)) * np.exp(0.5 * rng.normal(size=(n, 1)))).astype(np.float32)
    yield "synth_rec_1M", X, rng.normal(size=(NQ, d)).astype(np.float32)
    n, d = 300_000, 128
    spec = (np.arange(1, d + 1) ** -0.8).astype(np.float32)
    K_ = (rng.normal(size=(n, d)) * np.sqrt(spec)).astype(np.float32)
    K_[:, :4] *= 20.0
    R = np.linalg.qr(rng.normal(size=(d, d)))[0].astype(np.float32)
    K_ = K_ @ R
    K_[:, rng.choice(d, 4, replace=False)] += rng.normal(scale=3.0, size=(n, 4)).astype(np.float32)
    Qa = K_[rng.choice(n, NQ, replace=False)] + 0.3 * rng.normal(size=(NQ, d)).astype(np.float32)
    yield "synth_attn_300k", K_, Qa


def bench_one(name, X, Q):
    """All exact MIPS baselines + VA variants on one (database, queries) pair; returns a dict."""
    if True:
        X = np.ascontiguousarray(X, np.float32); Q = np.ascontiguousarray(Q, np.float32)
        mi = MipsIndex(X, bits=6)
        mn = MipsIndex(X, bits=6, norm_order=True)
        tb = tv = np.full(len(Q), np.inf)
        for _ in range(3):
            bid, bs, t = mi.bf(Q); tb = np.minimum(tb, t)
            _, vs, t, nr = mi.va(Q); tv = np.minimum(tv, t)
        assert np.allclose(bs, vs, rtol=1e-4, atol=1e-3), "VA MIPS differs from brute force"
        tbn = tvn = np.full(len(Q), np.inf)
        for _ in range(3):
            _, s1, t, nsc = mn.bf_norm(Q); tbn = np.minimum(tbn, t)
            _, s2, t, nr2, nblk = mn.va_norm(Q); tvn = np.minimum(tvn, t)
        assert np.allclose(bs, s1, rtol=1e-4, atol=1e-3) and np.allclose(bs, s2, rtol=1e-4, atol=1e-3)
        tg = []
        for q in Q:
            best = np.inf
            for _ in range(2):
                a = time.perf_counter(); sc = X @ q; idx = np.argpartition(-sc, K)[:K]; best = min(best, time.perf_counter() - a)
            tg.append(best)
        tm = []
        for b in range(0, len(Q), 64):
            best = np.inf
            for _ in range(2):
                a = time.perf_counter(); S = Q[b:b + 64] @ X.T; np.argpartition(-S, K, axis=1)[:, :K]; best = min(best, time.perf_counter() - a)
            tm.append(best / len(Q[b:b + 64]))
        r = {"n": X.shape[0], "d": X.shape[1], "bf_c_us": tb.mean() / 1e3, "blas_gemv_us": np.mean(tg) * 1e6,
             "blas_gemm_per_query_us": np.mean(tm) * 1e6, "va_us": tv.mean() / 1e3, "refined_frac": nr.mean() / X.shape[0],
             "bf_norm_stop_us": tbn.mean() / 1e3, "bf_norm_scanned_frac": nsc.mean() / X.shape[0],
             "va_norm_stop_us": tvn.mean() / 1e3, "va_norm_blocks_frac": nblk.mean() / ((X.shape[0] + 63) // 64)}
        best_exact = min(r["bf_c_us"], r["blas_gemv_us"], r["blas_gemm_per_query_us"], r["bf_norm_stop_us"])
        r["speedup_vs_best_exact"] = best_exact / min(r["va_us"], r["va_norm_stop_us"])
        res = {k: float(v) for k, v in r.items()}
        print(f"[{name}] n={r['n']} d={r['d']} | BF-C {r['bf_c_us']:.0f}us | BLAS gemv {r['blas_gemv_us']:.0f}us | "
              f"BLAS gemm/q {r['blas_gemm_per_query_us']:.0f}us | BF+norm-stop {r['bf_norm_stop_us']:.0f}us (scan {r['bf_norm_scanned_frac']:.2f}) | "
              f"VA {r['va_us']:.0f}us | VA+norm-stop {r['va_norm_stop_us']:.0f}us (blocks {r['va_norm_blocks_frac']:.2f}) | "
              f"VA vs best exact {r['speedup_vs_best_exact']:.2f}x", flush=True)
    return res


def main():
    out = {name: bench_one(name, X, Q) for name, X, Q in datasets()}
    os.makedirs("results/exp15", exist_ok=True)
    json.dump(out, open("results/exp15/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
