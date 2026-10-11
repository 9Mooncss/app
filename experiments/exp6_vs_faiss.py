"""Experiment 6: our exact VA fast-scan vs FAISS HNSW with a native bitmap filter.

FAISS HNSWFlat (M=32, efConstruction=200), search with SearchParametersHNSW
(sel = IDSelectorBitmap, efSearch = ef), single thread. FAISS is timed per query
from Python; the fixed Python->C++ call overhead is measured on a 16-vector index
and subtracted (reported separately), so FAISS is not penalised for the binding.
For each group, HNSW's time is taken at the smallest ef whose mean recall@10 >= target.
Our VA scan and BF are exact (recall 1.0), timed inside C (see exp5).

Usage: python experiments/exp6_vs_faiss.py
"""
import json
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exp5_exact_scan import band, workload  # noqa: E402
from fknn.data import ALL_MODES  # noqa: E402
from fknn.fastscan import FastScanIndex  # noqa: E402

EFS = [16, 32, 64, 128, 256, 512, 1024]
TARGETS = (0.9, 0.95, 0.99)
K = 10


def bitmap(ids, n):
    b = np.zeros(n, dtype=bool)
    b[ids] = True
    return np.packbits(b, bitorder="little")


def faiss_overhead(d, reps=2000):
    idx = faiss.IndexHNSWFlat(d, 32)
    idx.add(np.random.default_rng(0).normal(size=(16, d)).astype(np.float32))
    q = np.zeros((1, d), np.float32)
    bm = np.packbits(np.ones(16, bool), bitorder="little")
    sel = faiss.IDSelectorBitmap(16, faiss.swig_ptr(bm))
    p = faiss.SearchParametersHNSW(); p.sel = sel; p.efSearch = 16
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); idx.search(q, K, params=p); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    out = {}
    for name in ["fashion", "glove", "synth_base", "synth_overlap"]:
        X, labels, L, qs = workload(name)
        n, d = X.shape
        fs = FastScanIndex(X, labels, L)
        Q = np.stack([q for q, _, _ in qs]); Ps = [P for _, P, _ in qs]
        tb = np.full(len(qs), np.inf); tv = np.full(len(qs), np.inf)
        for _ in range(3):
            eid, ed, t, nb = fs.bf(Q, Ps); tb = np.minimum(tb, t)
            _, _, t, _, _ = fs.va(Q, Ps); tv = np.minimum(tv, t)
        tb, tv = tb / 1e3, tv / 1e3          # microseconds
        t0 = time.time()
        faiss.omp_set_num_threads(4)
        idx = faiss.IndexHNSWFlat(d, 32)
        idx.hnsw.efConstruction = 200
        idx.add(np.ascontiguousarray(X, dtype=np.float32))
        build = time.time() - t0
        faiss.omp_set_num_threads(1)
        ovh = faiss_overhead(d) * 1e6
        # exact neighbor sets by id (BF ids are original ids)
        exact = [set(r.tolist()) for r in eid]
        label_sorted = np.argsort(labels, kind="stable")
        lstart = np.concatenate([[0], np.cumsum(np.bincount(labels, minlength=L))])
        bms = [bitmap(np.concatenate([label_sorted[lstart[l]:lstart[l + 1]] for l in P]), n) for P in Ps]
        th = {ef: np.empty(len(qs)) for ef in EFS}
        rh = {ef: np.empty(len(qs)) for ef in EFS}
        for ef in EFS:
            for i, (q, bm) in enumerate(zip(Q, bms)):
                sel = faiss.IDSelectorBitmap(n, faiss.swig_ptr(bm))
                p = faiss.SearchParametersHNSW(); p.sel = sel; p.efSearch = ef
                best = np.inf
                for _ in range(2):
                    t1 = time.perf_counter(); _, I = idx.search(q[None], K, params=p); best = min(best, time.perf_counter() - t1)
                th[ef][i] = max(best * 1e6 - ovh, 0.0)
                rh[ef][i] = len(exact[i] & set(I[0].tolist())) / K
        s = nb / n
        groups = {"all": np.ones(len(qs), bool)}
        for b in ("s<1%", "1%<=s<10%", "s>=10%"):
            groups[b] = np.array([band(x) == b for x in s])
        if name.startswith("synth"):
            for m in ALL_MODES:
                groups[m] = np.array([mm == m for _, _, mm in qs])
        res = {"faiss_build_s": round(build, 1), "faiss_call_overhead_us": ovh, "groups": {}}
        for g, mk in groups.items():
            if not mk.any():
                continue
            r = {"n": int(mk.sum()), "bf_us": float(tb[mk].mean()), "va_exact_us": float(tv[mk].mean()),
                 "hnsw_curve": [(ef, float(th[ef][mk].mean()), float(rh[ef][mk].mean())) for ef in EFS]}
            for tgt in TARGETS:
                ok = [(t, ef) for ef, t, rr in r["hnsw_curve"] if rr >= tgt]
                r[f"hnsw_us_at_{tgt}"] = min(ok)[0] if ok else None
            res["groups"][g] = r
            h95 = r["hnsw_us_at_0.95"]
            print(f"[{name}/{g}] n={r['n']} BF {r['bf_us']:.0f}us | VA exact {r['va_exact_us']:.0f}us | "
                  f"FAISS-HNSW@0.95 {('%.0fus' % h95) if h95 is not None else 'never'} | "
                  f"HNSW curve " + " ".join(f"ef{ef}:{t:.0f}us/r{rr:.2f}" for ef, t, rr in r["hnsw_curve"]), flush=True)
        out[name] = res
    os.makedirs("results/exp6", exist_ok=True)
    out["config"] = dict(faiss=faiss.__version__, M=32, efConstruction=200, efs=EFS, k=K, threads=1)
    with open("results/exp6/summary.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
