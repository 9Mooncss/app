# Filtered k-NN: research on where search wastes work

Research project: can a planner that sees *where* the allowed points are (not only
how many there are) choose a cheaper filtered nearest-neighbor strategy than a
planner that only knows global filter selectivity?

Plan and hypotheses: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).
Results log: [`docs/RESULTS.md`](docs/RESULTS.md). Commercial notes: [`docs/COMMERCIAL.md`](docs/COMMERCIAL.md).

**Current best result:** exact filtered k-NN (`fknn/c/fscan.c`, kernels `va3`/`va4`) 4–9× faster than
contiguous brute force, faster than ACORN-γ (at recall 0.95) for selectivity < 10%. See RESULTS, cycles 5–8.

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
| `fknn/fastscan.py`, `fknn/c/fscan.c` | Exact filtered fast-scan kernels (v1 AVX2, v3 AVX-512 VBMI 6-bit, v4 early-abandon) |
| `fknn/scattered.py` | Kernels for arbitrary (scattered) predicates |
| `fknn/exact_lb.py`, `fknn/exact_va.py` | Python reference lower bounds (PCA, VA) |
| `bench/acorn_bench.cpp` | Harness for the official ACORN implementation |
| `bench/edge_vs_dist.c` | C micro-benchmark: cost of an adjacency scan vs a distance |
| `experiments/exp1_h1.py` | Experiment 1 (hypothesis H1) |
| `experiments/exp2_h2.py` | Experiments 2 and 3 (`--new-strategies --tag 3`) |
| `experiments/exp3_analyze.py` | Experiment 3 with edge-cost weighting |
| `experiments/exp4_curves.py`, `plot_exp4.py` | Experiment 4: recall-matched cost curves |
| `experiments/exp5_exact_scan.py` | Exact kernels vs BF (exp5/8/9/13 via args and `ALT`/`BITS3` env) |
| `experiments/exp6_vs_faiss.py`, `exp7_vs_acorn.py`, `exp11_vs_pdx.py`, `exp12_unfiltered_kernel.py` | Competitor comparisons |
| `experiments/exp10_hybrid.py` | Hybrid planner VA3 + ACORN-γ |
| `experiments/exp14_scattered.py` | Arbitrary predicates |

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
Exact kernels need x86-64 with AVX-512 BW+VBMI (v3/v4) or AVX2 (v1); built automatically with gcc.
spaCy vectors: extract `vocab/vectors` from the en_core_web_lg-3.8.0 wheel to `results/cache/data/spacy_lg_vectors.npy`.
Real datasets are downloaded on first use into `results/cache/data` (~165 MB).
Each run writes `results/exp1_<tag>/raw.csv` (one row per query),
`summary.json` (planner comparison + config) and `cost_recall.png`.

Cost metric is **number of distance computations**, which does not depend on the
machine. Wall times are recorded too, but `hnswlib` times include Python filter
callback overhead, so treat them as indicative only.
