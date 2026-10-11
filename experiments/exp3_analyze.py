"""Experiment 3 analysis: planners with the new strategies (IVFX, S2H), with S2H
charged for adjacency scans:  cost = distances + w * edges_scanned,
where w = (time of one edge scan) / (time of one distance), measured by
bench/edge_vs_dist.c at the dataset's n and d.

Usage: python experiments/exp3_analyze.py --w synth_base=0.046,synth_overlap=0.046,fashion=0.003,glove=0.021
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import exp2_h2 as E  # noqa: E402

ROOT = os.path.join("results", "exp3")


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "mode":
                r[k] = float(v)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--edge-multiplier", type=float, default=1.0,
                    help="sensitivity check: scale every w by this factor")
    a = ap.parse_args()
    W = {k: float(v) * a.edge_multiplier for k, v in (p.split("=") for p in a.w.split(","))}
    E.EXTRA[:] = ["IVFX"] + [f"S2H{ef}" for ef in E.S2H_EFS]
    table = {}
    for ds, w in W.items():
        d = os.path.join(ROOT, ds)
        splits = {f[4:-4]: load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("raw_")}
        for rows in splits.values():
            for r in rows:
                for ef in E.S2H_EFS:
                    r[f"cost_S2H{ef}"] += w * r[f"edges_S2H{ef}"]
        dev = splits.pop("dev")
        res = E.analyze(dev, splits, overhead=64)
        res["edge_weight"] = w
        with open(os.path.join(d, f"summary_weighted_x{a.edge_multiplier:g}.json"), "w") as f:
            json.dump(res, f, indent=2)
        table[ds] = res
        for t in splits:
            ps = ["B0_s_fallback_no_new", "D0_local_fallback_no_new", "A_s_only", "B_s_fallback",
                  "C_local", "D_local_fallback", "oracle"]
            line = " | ".join(f"{p}: {res[p][t]['mean_cost']:.0f} r{res[p][t]['mean_recall']:.3f} p10 {res[p][t]['p10_recall']:.1f}"
                              for p in ps)
            print(f"[{ds}/{t}] w={w:g} {line}")
            print(f"   D uses {res['D_local_fallback'][t]['strategy_use']}")
            print(f"   B uses {res['B_s_fallback'][t]['strategy_use']}")
    return table


if __name__ == "__main__":
    main()
