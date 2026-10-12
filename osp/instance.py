"""OSP instance (.dzn) parser and the reference cost evaluator.

Semantics follow the published local-search code (github.com/iolab-uniud/osp-ls,
local-search/OSP_data.cc) exactly, so costs are directly comparable:
  * a batch = jobs on one machine at one position; all share the attribute;
    capacity: sum(size) <= max_cap[m]; duration = max(min_time) <= min(max_time)
  * batches on a machine run in position order; start = earliest t with
      t >= max(prev_end + setup_time(prev_attr, attr), max earliest_start of jobs)
      and [t - setup, t + duration] inside one availability interval of the machine
      (t is pushed to interval_start + setup when needed); if none fits within the
      horizon the batch (and every later batch on that machine) is "not scheduled"
  * cost = w_setupcost * sum setup_cost + w_tardy * #jobs ending after latest_end
           + w_runtime * sum batch durations + 2*UB * #jobs in unscheduled batches
    (only scheduled batches contribute to the first three terms)
  * normalized cost = cost / upper_bound_integer_objective
"""
import re
from dataclasses import dataclass

import numpy as np


def _arr(txt, name):
    m = re.search(rf"(?<![A-Za-z_]){name}\s*=\s*\[(.*?)\];", txt, re.S)
    return m.group(1)


def _int(txt, name):
    return int(re.search(rf"(?<![A-Za-z_]){name}\s*=\s*(-?\d+)\s*;", txt).group(1))


def _ints(s):
    return [int(x) for x in re.findall(r"-?\d+", s)]


def _matrix(txt, name, rows, cols):
    rows_txt = [r for r in _arr(txt, name).split("|") if r.strip(", \n")]
    return np.array([_ints(r)[:cols] for r in rows_txt[:rows]], dtype=np.int64)


@dataclass
class Instance:
    name: str
    horizon: int
    A: int
    setup_cost: np.ndarray      # (A, A) [prev, next]
    setup_time: np.ndarray
    M: int
    max_cap: np.ndarray
    init_state: np.ndarray      # (M,) 0-based attribute
    S: int
    av_start: np.ndarray        # (M, S)
    av_end: np.ndarray
    N: int
    eligible: list              # list of sets of 0-based machines
    est: np.ndarray
    lst: np.ndarray             # latest end
    min_time: np.ndarray
    max_time: np.ndarray
    size: np.ndarray
    attr: np.ndarray            # 0-based
    w_runtime: int
    w_tardy: int
    w_setuptime: int
    w_setupcost: int
    ub: int


def load(path):
    t = open(path).read()
    A, M, S, N = _int(t, "a"), _int(t, "m"), _int(t, "s"), _int(t, "n")
    elig = [set(int(x) - 1 for x in re.findall(r"\d+", g)) for g in re.findall(r"\{([^}]*)\}", _arr(t, "eligible_machine"))]
    return Instance(
        name=path.split("/")[-1], horizon=_int(t, "l"), A=A,
        setup_cost=_matrix(t, "setup_costs", A, A), setup_time=_matrix(t, "setup_times", A, A),
        M=M, max_cap=np.array(_ints(_arr(t, "max_cap"))), init_state=np.array(_ints(_arr(t, "initState"))) - 1,
        S=S, av_start=np.array(_ints(_arr(t, "m_a_s"))).reshape(M, S), av_end=np.array(_ints(_arr(t, "m_a_e"))).reshape(M, S),
        N=N, eligible=elig, est=np.array(_ints(_arr(t, "earliest_start"))), lst=np.array(_ints(_arr(t, "latest_end"))),
        min_time=np.array(_ints(_arr(t, "min_time"))), max_time=np.array(_ints(_arr(t, "max_time"))),
        size=np.array(_ints(_arr(t, "size"))), attr=np.array(_ints(_arr(t, "attribute"))) - 1,
        w_runtime=_int(t, "mult_factor_total_runtime"), w_tardy=_int(t, "mult_factor_finished_toolate"),
        w_setuptime=_int(t, "mult_factor_total_setuptimes"), w_setupcost=_int(t, "mult_factor_total_setupcosts"),
        ub=_int(t, "upper_bound_integer_objective"))


def batch_start(inst, m, earliest, setup, dur, prev_end):
    """Exact port of CalculateBatchStartTime + CalculateEarliestSuitableMachineIntervalStart."""
    e = max(earliest, prev_end + setup)
    if e + dur > inst.horizon:
        return inst.horizon + 1
    for s in range(inst.S):
        es = max(e, inst.av_start[m, s] + setup)
        if es - setup >= inst.av_start[m, s] and es + dur <= inst.av_end[m, s]:
            return es
    return inst.horizon + 1


def evaluate(inst, schedule):
    """schedule: list over machines of lists of batches (each batch = list of job ids, 0-based).
    Returns dict of components, total cost and feasibility violations (capacity/attribute/eligibility/time)."""
    H = inst.horizon
    setupcost = tardy = runtime = unsched = 0
    viol = []
    for m in range(inst.M):
        prev_attr, prev_end, prev_start = inst.init_state[m], 0, 0
        for p, jobs in enumerate(schedule[m]):
            a = inst.attr[jobs[0]]
            dur = int(inst.min_time[jobs].max())
            if any(inst.attr[j] != a for j in jobs): viol.append(("attr", m, p))
            if inst.size[jobs].sum() > inst.max_cap[m]: viol.append(("cap", m, p))
            if dur > inst.max_time[jobs].min(): viol.append(("time", m, p))
            if any(m not in inst.eligible[j] for j in jobs): viol.append(("elig", m, p))
            st = inst.setup_time[prev_attr, a]
            if prev_start > H:
                start = H + 1
            else:
                start = batch_start(inst, m, int(inst.est[jobs].max()), st, dur, prev_end)
            end = start + dur
            if start <= H:
                setupcost += inst.setup_cost[prev_attr, a]
                runtime += dur
                tardy += int((inst.lst[jobs] < end).sum())
            else:
                unsched += len(jobs)
            prev_attr, prev_end, prev_start = a, end, start
    total = (inst.w_setupcost * setupcost + inst.w_tardy * tardy + inst.w_runtime * runtime
             + 2 * inst.ub * unsched)
    return dict(total=int(total), normalized=total / inst.ub, setup_cost=int(setupcost), tardy=tardy,
                runtime=runtime, unscheduled=unsched, violations=viol)
