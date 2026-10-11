# Filtered k-NN: research on where search wastes work

Research project: can a planner that sees *where* the allowed points are (not only
how many there are) choose a cheaper filtered nearest-neighbor strategy than a
planner that only knows global filter selectivity?

Plan and hypotheses: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).
Results log: [`docs/RESULTS.md`](docs/RESULTS.md).

## Layout
| Path | What |
|---|---|
| `fknn/data.py` | Synthetic data: Gaussian clusters, labels tied to clusters, filters with controlled selectivity and correlation |
| `fknn/bruteforce.py` | Exact baseline: inverted label index + brute force |
| `fknn/hnsw_build.py` | Builds a real `hnswlib` index (deterministic, 1 thread) |
| `fknn/hnsw_replica.py` | Loads the hnswlib graph from its file and replays hnswlib's filtered search in Python, counting distance computations |
| `tests/test_correctness.py` | Brute force vs naive reference; replica vs real hnswlib (≥99% identical results) |
| `experiments/exp1_h1.py` | Experiment 1 (hypothesis H1) |

## Run
```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests
python3 experiments/exp1_h1.py --tag base               # ~10 min on 4 cores
python3 experiments/exp1_h1.py --tag overlap --sigma 1.2
```
Each run writes `results/exp1_<tag>/raw.csv` (one row per query),
`summary.json` (planner comparison + config) and `cost_recall.png`.

Cost metric is **number of distance computations**, which does not depend on the
machine. Wall times are recorded too, but `hnswlib` times include Python filter
callback overhead, so treat them as indicative only.
