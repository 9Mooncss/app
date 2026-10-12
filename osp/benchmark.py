"""Held-out evaluation: our C++ SA vs published results (authors' SA 5M evaluations, LNS, CP, ILP)
on instances 81-120 (250/500 jobs) of all three use cases. Several seeds per instance.
Usage: python -m osp.benchmark OUT.json TIME SEEDS [first_instance last_instance]"""
import json
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from .fast import solve
from .instance import instance_files

ROOT = "/tmp/claude-0/-home-user-app/1282557a-9b1d-593e-89bf-391d46ab0c81/scratchpad/osp-ls"
T0F, TEF = 0.005, 2e-6   # overwritten by env after tuning


def published(uc):
    P = f"{ROOT}/processed-datasets"
    ex = pd.read_csv(f"{P}/exact-methods/exact-methods-uc-{uc}.csv", index_col=0).replace(-1, np.nan)
    sa = pd.read_csv(f"{P}/sa-5-milioni/sa-uc{uc}-5-milion.csv", index_col=0)
    ln = pd.read_csv(f"{P}/lns-results/lns-uc-{uc}.csv", index_col=0).replace(-1, np.nan)
    df = ex.merge(sa, on="instance").merge(ln, on="instance").set_index("instance")
    df["best_published"] = df[["ilp", "cp", "total_cost_min", "LNS"]].min(axis=1)
    return df


def job(a):
    uc, idx, path, t, seed, params = a
    st, _, ev = solve(path, t, seed=seed, params=params)
    return uc, idx, seed, ev["normalized"], st["iterations"]


def main():
    import os
    out, t, seeds = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (81, 120)
    params = [float(os.environ.get("T0F", T0F)), float(os.environ.get("TEF", TEF))]
    tasks = []
    for uc in (1, 2, 3):
        files = instance_files(f"{ROOT}/instances/use-case-{uc}")
        for idx in range(lo, hi + 1):
            for s in range(seeds):
                tasks.append((uc, idx, files[idx], t, 100 + s, params))
    cache = out.replace(".json", ".runs.jsonl")
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            r = json.loads(line)
            done[(r[0], r[1], r[2])] = r
    todo = [a for a in tasks if (a[0], a[1], a[4]) not in done]
    with mp.Pool(4) as pool, open(cache, "a") as f:
        for r in pool.imap_unordered(job, todo, chunksize=1):
            f.write(json.dumps(list(r)) + "\n"); f.flush()
            done[(r[0], r[1], r[2])] = list(r)
    res = [done[(a[0], a[1], a[4])] for a in tasks]
    rows = []
    for uc in (1, 2, 3):
        pub = published(uc)
        for idx in range(lo, hi + 1):
            v = [r[3] for r in res if r[0] == uc and r[1] == idx]
            p = pub.loc[idx]
            rows.append(dict(uc=uc, instance=idx, ours_mean=float(np.mean(v)), ours_min=float(np.min(v)),
                             sa_mean=float(p.total_cost_mean), sa_min=float(p.total_cost_min), sa_time=float(p.time_seconds_mean),
                             lns=float(p.LNS), cp=float(p.cp), best_published=float(p.best_published)))
    df = pd.DataFrame(rows)
    rel = lambda a, b: (df[a] - df[b]) / df[b]
    summ = {}
    for uc in (1, 2, 3):
        d = df[df.uc == uc]
        r = lambda a, b: float(((d[a] - d[b]) / d[b]).mean())
        summ[uc] = dict(
            n=len(d),
            mean_rel_ours_mean_vs_sa_mean=r("ours_mean", "sa_mean"),
            mean_rel_ours_min_vs_best_published=r("ours_min", "best_published"),
            ours_mean_better_than_sa_mean=int((d.ours_mean < d.sa_mean - 1e-12).sum()),
            ours_mean_worse_than_sa_mean=int((d.ours_mean > d.sa_mean + 1e-12).sum()),
            new_best_known=int((d.ours_min < d.best_published - 1e-12).sum()),
            matched_best_known=int((abs(d.ours_min - d.best_published) <= 1e-12).sum()),
            sa_mean_time_s=float(d.sa_time.mean()))
        print(f"UC{uc} (t={t}s x {seeds} seeds): " + json.dumps(summ[uc]), flush=True)
    df.to_csv(out.replace(".json", ".csv"), index=False)
    json.dump({"time": t, "seeds": seeds, "params": params, "summary": summ}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
