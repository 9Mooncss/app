"""OSP: our evaluator must reproduce the authors' solver costs exactly; our solver must return
verified, feasible solutions. Needs external/osp-ls (scripts/setup_osp_ls.sh); skipped otherwise."""
import glob
import json
import os
import re

import pytest

from osp.benchmark import ROOT
from osp.instance import evaluate, load

REF = os.path.join(os.path.dirname(__file__), "data", "osp")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "instances")), reason="osp-ls instances not present")


@pytest.mark.parametrize("ref", sorted(glob.glob(REF + "/*.json")))
def test_evaluator_matches_reference_solver(ref):
    b = os.path.basename(ref)
    uc, name = b[2], b[4:-5]
    inst = load(f"{ROOT}/instances/use-case-{uc}/{name}.dzn")
    sol_txt, summ_txt = open(ref).read().split("}}", 1)
    grab = lambda k: [int(x) for x in re.search(rf'"{k}": \[([^\]]*)\]', sol_txt).group(1).split(",") if x.strip()]  # noqa: E731
    sched = [[] for _ in range(inst.M)]
    for j, (m, p) in enumerate(zip(grab("machine_for_job"), grab("batch_for_job"))):
        while len(sched[m - 1]) < p:
            sched[m - 1].append([])
        sched[m - 1][p - 1].append(j)
    sched = [[x for x in ms if x] for ms in sched]
    r = evaluate(inst, sched)
    assert r["total"] == json.loads(summ_txt.strip())["total_cost"]
    assert not r["violations"]


def test_solver_returns_verified_solution():
    from osp.fast import solve
    from osp.instance import instance_files
    path = instance_files(f"{ROOT}/instances/use-case-3")[61]
    stats, sched, ev = solve(path, 1.0, seed=3, params=[0.002, 6e-7])   # solve() asserts feasibility + cost match
    assert ev["total"] == stats["cost"]
