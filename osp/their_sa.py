"""Run the authors' SA (osp-ls, easylocal) on THIS machine, for calibration and time-matched comparison.
Parameters as in our reproduction (rates 0.2/0.2/0.1/0.2/0.1/0.2, T_end 0.004, cooling 0.988,
accepted ratio 0.309; from arXiv 2410.01368 Table 5 where available).
Usage: python -m osp.their_sa OUT.json MODE   (MODE = 5M | matched:<seconds>:<calibration.json>)"""
import json
import multiprocessing as mp
import subprocess
import sys

from .benchmark import ROOT, published
from .instance import instance_files, load

BIN = f"{ROOT}/bin/osp"
SET = [(uc, i) for uc in (1, 2, 3) for i in (85, 95, 105, 115)]


def run(a):
    uc, idx, evals, seed = a
    path = instance_files(f"{ROOT}/instances/use-case-{uc}")[idx]
    out = f"/tmp/their_{uc}_{idx}_{evals}_{seed}.json"
    cmd = [BIN, "--main::instance", path, "--main::seed", str(seed), "--main::solution_method", "2",
           "--metaheuristic::method", "SA_all", "--metaheuristic::initial_solution", "1",
           "--metaheuristic::swap_rate", "0.2", "--metaheuristic::insert_rate", "0.2", "--metaheuristic::inverse_rate", "0.1",
           "--metaheuristic::single_job_to_new_batch_rate", "0.2", "--metaheuristic::more_jobs_to_new_batch_rate", "0.1",
           "--metaheuristic::job_to_existing_batch_rate", "0.2", "--SA_all::compute_start_temperature-enable",
           "--SA_all::min_temperature", "0.004", "--SA_all::cooling_rate", "0.988", "--SA_all::neighbors_accepted_ratio", "0.309",
           "--SA_all::max_evaluations", str(int(evals)), "--main::output_file", out]
    subprocess.run(cmd, capture_output=True, check=True)
    txt = open(out).read()
    summ = json.loads(txt.split("}}", 1)[1].strip())
    ub = load(path).ub
    return dict(uc=uc, instance=idx, evals=int(evals), seed=seed, normalized=summ["total_cost"] / ub, time=summ["time_seconds"])


def main():
    out, mode = sys.argv[1], sys.argv[2]
    if mode == "5M":
        tasks = [(uc, i, 5_000_000, 1) for uc, i in SET]
    else:
        _, secs, cal = mode.split(":")
        calib = {(r["uc"], r["instance"]): r for r in json.load(open(cal))}
        tasks = [(uc, i, 5_000_000 * float(secs) / calib[(uc, i)]["time"], s) for uc, i in SET for s in (1, 2, 3)]
    with mp.Pool(4) as pool:
        res = pool.map(run, tasks, chunksize=1)
    for r in res:
        p = published(r["uc"]).loc[r["instance"]]
        r["published_sa_mean"], r["published_sa_time"] = float(p.total_cost_mean), float(p.time_seconds_mean)
        print(json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
