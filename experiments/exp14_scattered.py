"""Experiment 14 (H7): exact k-NN under arbitrary scattered predicates (allowed rows chosen at
random, unrelated to any data layout). BF over the id list vs VA3 masked full scan vs VA3
transpose-gather. Exact results are checked. Single thread, min of 3 runs per query.
Usage: python experiments/exp14_scattered.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.realdata import load_fashion, load_glove, load_spacy  # noqa: E402
from fknn.scattered import ScatteredIndex  # noqa: E402

SEL = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
NQ = 60


def main():
    out = {}
    for name, loader in (("fashion", load_fashion), ("glove", load_glove), ("spacy", load_spacy)):
        rd = loader()
        si = ScatteredIndex(rd.X, bits=6)
        rng = np.random.default_rng(7)
        out[name] = {}
        for s in SEL:
            idl = [np.sort(rng.choice(si.n, int(s * si.n), replace=False)) for _ in range(NQ)]
            Q = rd.Q[rng.choice(len(rd.Q), NQ, replace=False)]
            tb = tm = tg = np.full(NQ, np.inf)
            for _ in range(3):
                _, bd, t = si.bf(Q, idl); tb = np.minimum(tb, t)
                _, md, t, _ = si.masked(Q, idl); tm = np.minimum(tm, t)
                _, gd, t, nr = si.gather(Q, idl); tg = np.minimum(tg, t)
            assert np.allclose(bd, md, rtol=1e-4) and np.allclose(bd, gd, rtol=1e-4)
            r = {"bf_us": tb.mean() / 1e3, "masked_us": tm.mean() / 1e3, "gather_us": tg.mean() / 1e3,
                 "refined_frac": float(nr.sum() / sum(len(x) for x in idl))}
            r["best_speedup_vs_bf"] = r["bf_us"] / min(r["masked_us"], r["gather_us"])
            out[name][str(s)] = {k: float(v) for k, v in r.items()}
            print(f"[{name} s={s}] BF {r['bf_us']:.0f}us | masked {r['masked_us']:.0f}us | gather {r['gather_us']:.0f}us | "
                  f"best {r['best_speedup_vs_bf']:.2f}x | refined {r['refined_frac']:.3f}", flush=True)
    os.makedirs("results/exp14", exist_ok=True)
    json.dump(out, open("results/exp14/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
