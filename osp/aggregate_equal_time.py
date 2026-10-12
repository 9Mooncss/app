"""Aggregate equal-time CI results (osp.equal_time parts) into a summary; prints a table.
Usage: python -m osp.aggregate_equal_time PARTS_DIR OUT.json"""
import glob
import json
import sys

import pandas as pd


def main():
    parts, out = sys.argv[1], sys.argv[2]
    rows = [r for f in glob.glob(f"{parts}/**/*.json", recursive=True) for r in json.load(open(f))]
    d = pd.DataFrame(rows)
    g = d.groupby(["uc", "instance"]).agg(their=("their", "mean"), ours=("ours", "mean"), ours_min=("ours", "min"),
                                          sa_pub=("published_sa_mean", "first"), best_pub=("best_published", "first")).reset_index()
    g["rel"] = (g.ours - g.their) / g.their
    summ = {"instances": int(len(g)), "runs": int(len(d)),
            "ours_better_than_their_equal_time": int((g.ours < g.their - 1e-12).sum()),
            "ours_worse_than_their_equal_time": int((g.ours > g.their + 1e-12).sum()),
            "mean_rel_ours_vs_their": float(g.rel.mean()),
            "new_best_known_vs_published": int((g.ours_min < g.best_pub - 1e-12).sum())}
    for uc, gg in g.groupby("uc"):
        summ[f"uc{uc}_mean_rel"] = float(gg.rel.mean())
        summ[f"uc{uc}_better"] = f"{int((gg.ours < gg.their - 1e-12).sum())}/{len(gg)}"
    print(json.dumps(summ, indent=2))
    json.dump({"summary": summ, "per_instance": g.to_dict(orient="records")}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
