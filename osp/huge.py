"""Huge instances (1000/2500/5000 jobs, use case 3; no published results): ours vs the authors' SA
at EQUAL wall time on this machine. Their evaluation budget is calibrated per instance from a
short run (50k evaluations). Usage: python -m osp.huge OUT.json SECONDS"""
import json
import multiprocessing as mp
import re
import sys

from .benchmark import ROOT
from .fast import solve
from .instance import load
from .their_sa import BIN  # noqa: F401
from . import their_sa


def their(path, evals, seed=1):
    import os
    import subprocess
    out = f"/tmp/their_huge_{os.path.basename(path)}_{int(evals)}.json"
    cmd = [their_sa.BIN, "--main::instance", path, "--main::seed", str(seed), "--main::solution_method", "2",
           "--metaheuristic::method", "SA_all", "--metaheuristic::initial_solution", "1",
           "--metaheuristic::swap_rate", "0.2", "--metaheuristic::insert_rate", "0.2", "--metaheuristic::inverse_rate", "0.1",
           "--metaheuristic::single_job_to_new_batch_rate", "0.2", "--metaheuristic::more_jobs_to_new_batch_rate", "0.1",
           "--metaheuristic::job_to_existing_batch_rate", "0.2", "--SA_all::compute_start_temperature-enable",
           "--SA_all::min_temperature", "0.004", "--SA_all::cooling_rate", "0.988", "--SA_all::neighbors_accepted_ratio", "0.309",
           "--SA_all::max_evaluations", str(int(evals)), "--main::output_file", out]
    subprocess.run(cmd, capture_output=True, check=True)
    summ = json.loads(open(out).read().split("}}", 1)[1].strip())
    return summ["total_cost"] / load(path).ub, summ["time_seconds"]


def task(a):
    path, secs = a
    _, t50 = their(path, 50_000)
    evals = max(50_000, 50_000 * secs / max(t50, 1e-3))
    tc, tt = their(path, evals)
    st, _, ev = solve(path, secs, seed=1, params=[0.002, 6e-7])
    n = int(re.search(r"(?m)^n=(\d+)", open(path).read()).group(1))
    return dict(instance=path.split("/")[-1], n=n, their=tc, their_time=tt, their_evals=int(evals),
                ours=ev["normalized"], ours_iters=st["iterations"])


def main():
    import glob
    out, secs = sys.argv[1], float(sys.argv[2])
    paths = sorted(glob.glob(f"{ROOT}/instances/huge-instances-use-case-3/*.dzn"))
    with mp.Pool(2) as pool:   # each task runs two solvers sequentially; 2 tasks x 1 thread at a time
        res = pool.map(task, [(p, secs) for p in paths], chunksize=1)
    for r in res:
        r["rel"] = (r["ours"] - r["their"]) / r["their"]
        print(json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
