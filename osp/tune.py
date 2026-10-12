"""Small grid tuning of the C++ SA on the TUNING instances only (n >= 200, 3 use cases).
Score = mean relative gap to the best value found by any config on that instance.
Usage: python -m osp.tune OUT.json TIME"""
import itertools
import json
import multiprocessing as mp
import re
import sys

import numpy as np

from .fast import solve
from .instance import load

SRC = "/tmp/claude-0/-home-user-app/1282557a-9b1d-593e-89bf-391d46ab0c81/scratchpad/osp-ls/instances"


def tuning_instances():
    import glob
    out = []
    for uc in (1, 2, 3):
        for p in sorted(glob.glob(f"{SRC}/tuning-use-case-{uc}/*.dzn")):
            n = int(re.search(r"(?m)^n=(\d+)", open(p).read()).group(1))
            if n >= 200:
                out.append(p)
    return out


def job(args):
    path, cfg, t = args
    st, _, ev = solve(path, t, seed=7, params=cfg)
    return path, tuple(cfg), ev["normalized"]


def main():
    out, t = sys.argv[1], float(sys.argv[2])
    insts = tuning_instances()
    import os
    t0s = [float(x) for x in os.environ.get("T0S", "0.005,0.02,0.05").split(",")]
    tes = [float(x) for x in os.environ.get("TES", "2e-6,2e-5,2e-4").split(",")]
    extra = [float(x) for x in os.environ.get("RATES", "").split(",") if x]
    grid = [tuple([a, b] + extra) for a, b in itertools.product(t0s, tes)]
    tasks = [(p, list(c), t) for p in insts for c in grid]
    with mp.Pool(4) as pool:
        res = pool.map(job, tasks, chunksize=1)
    best = {}
    for p, c, v in res:
        best[p] = min(best.get(p, np.inf), v)
    score = {}
    for p, c, v in res:
        score.setdefault(c, []).append((v - best[p]) / max(best[p], 1e-9))
    table = sorted(((float(np.mean(v)), c) for c, v in score.items()))
    for s, c in table:
        print(f"T0={c[0]:<6} Tend={c[1]:<7} mean rel gap {s:.4f}")
    json.dump({"instances": len(insts), "time": t, "table": [(s, list(c)) for s, c in table]}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
