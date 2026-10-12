"""Run the C++ OSP solver (osp/cpp/solver.cpp) and verify its solution with the reference evaluator."""
import json
import os
import subprocess
import tempfile

from .instance import evaluate, load

_DIR = os.path.join(os.path.dirname(__file__), "cpp")
_BIN = os.path.join(_DIR, "solver")


def _build():
    src = os.path.join(_DIR, "solver.cpp")
    if not os.path.exists(_BIN) or os.path.getmtime(_BIN) < os.path.getmtime(src):
        subprocess.check_call(["g++", "-O3", "-march=native", "-std=c++17", "-o", _BIN, src])


def write_txt(inst, path):
    L = [inst.horizon, inst.A, inst.M, inst.S, inst.N]
    L += inst.setup_cost.ravel().tolist() + inst.setup_time.ravel().tolist()
    L += inst.max_cap.tolist() + inst.init_state.tolist() + inst.av_start.ravel().tolist() + inst.av_end.ravel().tolist()
    for v in (inst.est, inst.lst, inst.min_time, inst.max_time, inst.size, inst.attr):
        L += v.tolist()
    for e in inst.eligible:
        L += [len(e)] + sorted(e)
    L += [inst.w_runtime, inst.w_tardy, inst.w_setupcost, inst.ub]
    with open(path, "w") as f:
        f.write(" ".join(map(str, L)))


def solve(dzn_path, time_limit, seed=1, params=()):
    _build()
    inst = load(dzn_path)
    if inst.w_setuptime != 0:
        raise NotImplementedError("setup-time weight != 0 is not used by any benchmark instance")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        path = tf.name
    write_txt(inst, path)
    out = subprocess.run([_BIN, path, str(time_limit), str(seed)] + [str(p) for p in params],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    os.unlink(path)
    stats = json.loads(out[0])
    sched = []
    for line in out[1:]:
        parts = line.split("|")[1:]
        sched.append([[int(x) for x in p.split()] for p in parts])
    sched += [[] for _ in range(inst.M - len(sched))]
    jobs = sorted(j for ms in sched for b in ms for j in b)
    assert jobs == list(range(inst.N)), "every job must appear exactly once"
    ev = evaluate(inst, sched)
    assert not ev["violations"], ev["violations"][:5]
    assert ev["total"] == stats["cost"] == stats["check"], (ev["total"], stats["cost"], stats["check"])
    stats["normalized_verified"] = ev["normalized"]
    return stats, sched, ev
