"""Experiment 7: our exact VA fast-scan vs the official ACORN implementation.

ACORN (stanford-futuredata/ACORN, SIGMOD'24) is built from source (see bench/README.md)
and driven by bench/acorn_bench.cpp: one query at a time, its own filter bitmap,
single thread, per-query time measured in C++. Variants:
  ACORN-1      M=32, gamma=1,  M_beta=64
  ACORN-gamma  M=32, gamma=12, M_beta=64   (paper's setting for low selectivity)
Same queries as Experiment 5. VA/BF are exact and timed in C (fknn/c/fscan.c).

Usage: ACORN_BENCH=/path/to/acorn_bench python experiments/exp7_vs_acorn.py [datasets]
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from exp5_exact_scan import band, workload  # noqa: E402
from fknn.data import ALL_MODES  # noqa: E402
from fknn.fastscan import FastScanIndex  # noqa: E402

EFS = [16, 32, 64, 128, 256, 512]
VARIANTS = {"ACORN-1": (32, 1, 64), "ACORN-gamma": (32, 12, 64)}
K = 10


def main():
    bench = os.environ["ACORN_BENCH"]
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["glove", "fashion", "synth_base", "synth_overlap"]
    variants = os.environ.get("ACORN_VARIANTS", ",".join(VARIANTS)).split(",")
    os.makedirs("results/exp7", exist_ok=True)
    for name in names:
        X, labels, L, qs = workload(name)
        n, d = X.shape
        fs = FastScanIndex(X, labels, L)
        Q = np.stack([q for q, _, _ in qs]).astype(np.float32)
        Ps = [P for _, P, _ in qs]
        tb = np.full(len(qs), np.inf); tv = np.full(len(qs), np.inf)
        for _ in range(3):
            eid, _, t, nb = fs.bf(Q, Ps); tb = np.minimum(tb, t)
            _, _, t, _, _ = fs.va(Q, Ps); tv = np.minimum(tv, t)
        tb, tv = tb / 1e3, tv / 1e3
        tmp = tempfile.mkdtemp(prefix=f"acorn_{name}_", dir=os.environ.get("TMPDIR", "/tmp"))
        F = np.zeros((len(qs), n), dtype=np.uint8)
        for i, P in enumerate(Ps):
            F[i, np.isin(labels, P)] = 1
        files = {"base": np.ascontiguousarray(X, np.float32), "q": Q, "f": F,
                 "e": np.ascontiguousarray(eid, np.int64), "l": labels.astype(np.int32)}
        for key, arr in files.items():
            arr.tofile(os.path.join(tmp, key + ".bin"))
        s = nb / n
        groups = {"all": np.ones(len(qs), bool)}
        for b in ("s<1%", "1%<=s<10%", "s>=10%"):
            groups[b] = np.array([band(x) == b for x in s])
        if name.startswith("synth"):
            for m in ALL_MODES:
                groups[m] = np.array([mm == m for _, _, mm in qs])
        res = {"groups": {g: {"n": int(mk.sum()), "bf_us": float(tb[mk].mean()), "va_exact_us": float(tv[mk].mean())}
                          for g, mk in groups.items() if mk.any()}}
        for var in variants:
            M, gamma, Mb = VARIANTS[var]
            out_csv = os.path.join(tmp, f"{var}.csv")
            cmd = [bench] + [os.path.join(tmp, f + ".bin") for f in ("base", "q", "f", "e", "l")] + \
                  [str(x) for x in (n, d, len(qs), K, M, gamma, Mb, out_csv)] + [str(e) for e in EFS]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"[{name}] {var} FAILED: {p.stderr[-500:]}", flush=True)
                continue
            build = p.stderr.split("build ")[1].split("s")[0]
            rows = list(csv.DictReader(open(out_csv)))
            T = {ef: np.zeros(len(qs)) for ef in EFS}; R = {ef: np.zeros(len(qs)) for ef in EFS}
            for r in rows:
                T[int(r["ef"])][int(r["query"])] = float(r["time_us"])
                R[int(r["ef"])][int(r["query"])] = float(r["recall"])
            for g, mk in groups.items():
                if not mk.any():
                    continue
                curve = [(ef, float(T[ef][mk].mean()), float(R[ef][mk].mean())) for ef in EFS]
                gr = res["groups"][g]
                gr[var] = {"build_s": float(build), "curve": curve}
                for tgt in (0.9, 0.95, 0.99):
                    ok = [t for ef, t, rr in curve if rr >= tgt]
                    gr[var][f"us_at_{tgt}"] = min(ok) if ok else None
                a95 = gr[var]["us_at_0.95"]
                print(f"[{name}/{g}] n={gr['n']} BF {gr['bf_us']:.0f}us | VA exact {gr['va_exact_us']:.0f}us | "
                      f"{var}@0.95 {('%.0fus' % a95) if a95 is not None else 'never'} | curve " +
                      " ".join(f"ef{ef}:{t:.0f}us/r{rr:.2f}" for ef, t, rr in curve), flush=True)
        with open(f"results/exp7/{name}.json", "w") as f:
            json.dump(res, f, indent=2)
        subprocess.run(["rm", "-rf", tmp])


if __name__ == "__main__":
    main()
