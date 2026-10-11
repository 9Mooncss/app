"""Experiment 5: exact filtered k-NN in C - contiguous brute force vs VA fast-scan.

Both methods return the exact k-NN (checked on every query). Single thread,
wall time measured inside C per query; each batch is run 3 times and the
per-query minimum is kept (standard micro-benchmark practice against noise).
Workloads: real data with random filters; synthetic data with every correlation
mode x selectivity. Results grouped by selectivity band (and mode).

Usage: python experiments/exp5_exact_scan.py
"""
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.data import ALL_MODES, make_dataset, make_queries  # noqa: E402
from fknn.fastscan import FastScanIndex  # noqa: E402
from fknn.realdata import load_fashion, load_glove, load_spacy, make_real_queries  # noqa: E402

SEL = [0.001, 0.005, 0.01, 0.05, 0.2, 0.5]
REPS = 3


def band(s):
    return "s<1%" if s < 0.01 else ("1%<=s<10%" if s < 0.1 else "s>=10%")


def workload(name):
    if name == "synth_big":   # 1M x 128 (512 MB): out-of-cache regime
        ds = make_dataset(n_clusters=100, per_cluster=10000, dim=128, labels_per_cluster=10, sigma=1.2)
        qs = [(q, P, m) for i, m in enumerate(ALL_MODES) for j, s in enumerate(SEL)
              for q, P in make_queries(ds, 25, m, s, 60000 + 31 * i + 7 * j)]
        return ds.X, ds.labels, ds.n_labels, qs
    if name.startswith("synth"):
        ds = make_dataset(sigma=0.6 if name == "synth_base" else 1.2)
        qs = [(q, P, m) for i, m in enumerate(ALL_MODES) for j, s in enumerate(SEL)
              for q, P in make_queries(ds, 25, m, s, 40000 + 31 * i + 7 * j)]
        return ds.X, ds.labels, ds.n_labels, qs
    rd = {"fashion": load_fashion, "glove": load_glove, "spacy": load_spacy}[name]()
    return rd.X, rd.labels, rd.n_labels, [(q, P, "natural") for q, P in make_real_queries(rd, 600, 50000)]


def main():
    out_all = {}
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fashion", "glove", "synth_base", "synth_overlap"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "exp5"
    for name in names:
        t0 = time.time()
        X, labels, L, qs = workload(name)
        fs = FastScanIndex(X, labels, L, bits3=int(os.environ.get("BITS3", "6")))
        build_s = time.time() - t0
        Q = np.stack([q for q, _, _ in qs]); Ps = [P for _, P, _ in qs]
        tb = np.full(len(qs), np.inf); tv = np.full(len(qs), np.inf); t2 = np.full(len(qs), np.inf)
        for _ in range(REPS):
            _, bd, t, nb = fs.bf(Q, Ps)
            tb = np.minimum(tb, t)
            _, vd, t, nr, ns = fs.va(Q, Ps)
            tv = np.minimum(tv, t)
            _, vd2, t, nr2, _ = getattr(fs, os.environ.get("ALT", "va2"))(Q, Ps)
            t2 = np.minimum(t2, t)
        assert np.allclose(bd, vd, rtol=1e-4, atol=1e-5), "VA result differs from brute force"
        assert np.allclose(bd, vd2, rtol=1e-4, atol=1e-5), "alternative kernel result differs from brute force"
        s = nb / len(X)
        groups = {"all": np.ones(len(qs), bool)}
        for b in ("s<1%", "1%<=s<10%", "s>=10%"):
            groups[b] = np.array([band(x) == b for x in s])
        if name.startswith("synth"):
            for m in ALL_MODES:
                groups[m] = np.array([mm == m for _, _, mm in qs])
        res = {"n": len(X), "d": X.shape[1], "n_queries": len(qs), "index_build_s": round(build_s, 1),
               "code_bytes_per_vector": fs.codes.nbytes / len(X), "groups": {}}
        for g, mask in groups.items():
            if not mask.any():
                continue
            r = dict(n=int(mask.sum()),
                     bf_mean_us=float(tb[mask].mean() / 1e3), va_mean_us=float(tv[mask].mean() / 1e3),
                     bf_median_us=float(np.median(tb[mask]) / 1e3), va_median_us=float(np.median(tv[mask]) / 1e3),
                     speedup_total=float(tb[mask].sum() / tv[mask].sum()),
                     speedup_median_query=float(np.median(tb[mask] / tv[mask])),
                     frac_queries_va_slower=float((tv[mask] > tb[mask]).mean()),
                     refined_frac=float(nr[mask].sum() / nb[mask].sum()),
                     va2_mean_us=float(t2[mask].mean() / 1e3),
                     va2_speedup_total=float(tb[mask].sum() / t2[mask].sum()),
                     va2_refined_frac=float(nr2[mask].sum() / nb[mask].sum()))
            res["groups"][g] = r
            print(f"[{name}/{g}] n={r['n']} BF {r['bf_mean_us']:.0f}us  VA {r['va_mean_us']:.0f}us  "
                  f"speedup(total) {r['speedup_total']:.2f}x  median-query {r['speedup_median_query']:.2f}x  "
                  f"VA slower on {100 * r['frac_queries_va_slower']:.0f}%  refined {r['refined_frac']:.3f} | "
                  f"VA2 {r['va2_mean_us']:.0f}us {r['va2_speedup_total']:.2f}x refined {r['va2_refined_frac']:.3f}", flush=True)
        out_all[name] = res
    out_all["config"] = dict(reps=REPS, k=10, bits=4, python=platform.python_version(), numpy=np.__version__,
                             cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
                             compiler="gcc -O3 -march=native")
    os.makedirs(f"results/{tag}", exist_ok=True)
    with open(f"results/{tag}/summary.json", "w") as f:
        json.dump(out_all, f, indent=2)


if __name__ == "__main__":
    main()
