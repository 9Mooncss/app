"""Equal-time comparison on the SAME machine: our SA vs the authors' SA (easylocal), for a slice of
the held-out large instances. Their evaluation budget is calibrated per instance (50k-evaluation run)
to fill the same wall time. Designed to run as many parallel CI jobs (GitHub Actions matrix).
Usage: python -m osp.equal_time OUT.json SECONDS SEEDS CHUNK NCHUNKS"""
import json
import multiprocessing as mp
import os
import sys

from .benchmark import ROOT, published
from .fast import solve
from .instance import instance_files
from .their_sa import run as their_run

PARAMS = [0.002, 6e-7]


def task(a):
    uc, idx, secs, seed = a
    cal = their_run((uc, idx, 50_000, seed))
    evals = max(50_000, 50_000 * secs / max(cal["time"], 1e-3))
    th = their_run((uc, idx, evals, seed))
    path = instance_files(f"{ROOT}/instances/use-case-{uc}")[idx]
    st, _, ev = solve(path, secs, seed=seed, params=PARAMS)
    return dict(uc=uc, instance=idx, seed=seed, their=th["normalized"], their_time=th["time"], their_evals=int(evals),
                ours=ev["normalized"], ours_iters=st["iterations"])


def main():
    out, secs, seeds, chunk, nchunks = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    items = [(uc, i) for uc in (1, 2, 3) for i in range(81, 121)]
    mine = items[chunk::nchunks]
    tasks = [(uc, i, secs, 100 + s) for uc, i in mine for s in range(seeds)]
    workers = int(os.environ.get("WORKERS", os.cpu_count() // 2 or 1))   # 2 solvers alternate; keep cores free
    with mp.Pool(workers) as pool:
        res = pool.map(task, tasks, chunksize=1)
    for r in res:
        p = published(r["uc"]).loc[r["instance"]]
        r["published_sa_mean"], r["best_published"] = float(p.total_cost_mean), float(p.best_published)
        print("RESULT " + json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
