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
| `fknn/planner.py` | Local-selectivity estimator, k-NN planner, budget+fallback strategies |
| `fknn/realdata.py` | Fashion-MNIST and GloVe loaders with natural attributes |
| `fknn/ivf_exact.py` | Exact filtered search with triangle-inequality cluster pruning |
| `fknn/seeded.py` | Allowed-only 2-hop graph search (S2H, seeded) and ACORN-like ablation |
| `bench/edge_vs_dist.c` | C micro-benchmark: cost of an adjacency scan vs a distance |
| `experiments/exp1_h1.py` | Experiment 1 (hypothesis H1) |
| `experiments/exp2_h2.py` | Experiments 2 and 3 (`--new-strategies --tag 3`) |
| `experiments/exp3_analyze.py` | Experiment 3 with edge-cost weighting |
| `experiments/exp4_curves.py`, `plot_exp4.py` | Experiment 4: recall-matched cost curves |

## Run
```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q tests
python3 experiments/exp1_h1.py --tag base               # ~10 min on 4 cores
python3 experiments/exp1_h1.py --tag overlap --sigma 1.2
python3 experiments/exp2_h2.py                           # ~15 min
python3 experiments/exp2_h2.py --new-strategies --tag 3  # ~15 min; then: mv results/exp23 results/exp3
gcc -O3 -march=native bench/edge_vs_dist.c -o edge_vs_dist && ./edge_vs_dist 198000 100
python3 experiments/exp3_analyze.py --w synth_base=0.067,synth_overlap=0.067,fashion=0.003,glove=0.021
python3 experiments/exp4_curves.py && python3 experiments/plot_exp4.py   # ~6 min
```
Real datasets are downloaded on first use into `results/cache/data` (~165 MB).
Each run writes `results/exp1_<tag>/raw.csv` (one row per query),
`summary.json` (planner comparison + config) and `cost_recall.png`.

Cost metric is **number of distance computations**, which does not depend on the
machine. Wall times are recorded too, but `hnswlib` times include Python filter
callback overhead, so treat them as indicative only.
