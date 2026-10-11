# Results log

## Cycle 1 — Experiment 1 (H1), 2026-10-11

Setup: n = 100 000, d = 64, 50 Gaussian clusters, 1000 labels (20 per cluster),
hnswlib M=16, ef_construction=200, k=10. 100 queries per (mode, s) cell,
1 800 per dataset; 70% dev / 30% held-out test. Cost = distance computations,
counted by an exact Python replica of hnswlib's filtered search on the graph
hnswlib itself built (tests: ≥99% identical results to real hnswlib).
Two geometries: separated clusters (`sigma=0.6`) and heavily overlapping (`sigma=1.2`).
Runtime: 277 s and 223 s on 4 cores. Files: `results/exp1_{base,overlap}/`.

### Observation 1: the same s can mean ~1 000× different HNSW cost
`sigma=0.6`, s = 0.1%: BF computes ~100 distances; HNSW(ef=64) computes ~2 400
(positive), ~82 000 (neutral), ~100 000 (negative, i.e. the whole database).
Real hnswlib wall time agrees in direction: 1.0 ms vs 50 ms vs 71 ms (BF: 0.1 ms).

### Observation 2: under negative correlation HNSW is both expensive and wrong
`sigma=0.6`, s = 50%: HNSW(ef=64) costs 38 700 distances (BF: 50 000)
and returns recall@10 = 0.53. The search gets stuck around the query and never
reaches the far-away allowed points. Overlapping clusters soften this (0.82)
but do not remove it.

### H1 verdict (held-out test queries)
| | separated (`sigma=0.6`) | overlapping (`sigma=1.2`) |
|---|---|---|
| s-planner: share of queries with a non-optimal choice | **37.8%** | **39.8%** |
| s-planner: mean cost ÷ oracle on those queries | **26.2×** | **3.0×** |
| total cost: s-planner ÷ mode-aware planner | 2.59× | 1.31× |
| total cost: mode-aware planner ÷ per-query oracle | 1.04× | 1.33× |

Pre-registered criterion (≥20% non-optimal and ≥2× cost on them) → **H1 supported
on both synthetic geometries.**

How to read it:
* With `sigma=0.6` the s-only planner degenerates to "always BF": at large s
  the negative queries pull mean HNSW recall below 0.95, so it cannot risk HNSW.
  It then overpays heavily on positive/neutral queries.
* The **prize for H2** (a planner that knows correlation) is at most 2.6× / 1.3×
  total cost on this workload. That is an upper bound: it assumes perfect
  knowledge of the correlation mode.

### What this does NOT show (honest limits)
1. **Synthetic, with an extreme negative mode** (allowed points in the farthest
   clusters). Real correlation is milder; the prize may be smaller.
2. **The prize depends on the workload mix** (here 1/3 per mode).
3. **A stronger s-only competitor exists and was not tested yet:** run HNSW with
   a distance budget and fall back to BF when it overruns (similar in spirit to
   pgvector's iterative scans and to fallback logic in vector DBs). It would fix
   the cost blow-up at small s *without* estimating correlation. It would NOT
   obviously fix observation 2 (cheap but wrong answers), because there the
   search stops early and looks "successful".
4. Only hnswlib-style filtered HNSW was tested; ACORN-type graphs are better on
   neutral filters.
5. The distance-count metric ignores memory-access patterns. BF reads contiguous
   memory and vectorises well, so per distance it is cheaper than graph search.
   This bias favours HNSW, i.e. it makes our BF-vs-HNSW gaps conservative.

### Next: Experiment 2 (H2)
Compare on the same data, held-out queries:
* A. s-planner (as now);
* B. **budget + fallback** (strongest s-only baseline, see limit 3);
* C. **local-selectivity planner**: k-means (64 centroids) + per-centroid label
  histogram; for query q, the share of allowed points among the m nearest
  centroids (cost: 64 distances + table lookups) → choose BF / HNSW(ef);
* D. C + fallback.
Report total cost, mean recall, worst-decile recall. H2 holds if C or D beats B
by ≥1.3× total cost at equal-or-better recall on **both** geometries. Then
validate on a real dataset with natural labels before any claim.

## Cycle 2 — Experiment 2 (H2), 2026-10-11

Question: does a cheap *local* selectivity estimate (k-means, 64 centroids,
per-centroid label histogram; 64 extra distance computations per query) let a
planner beat the strongest planner that only knows global selectivity?

All planners are the same learner (k-NN regression over dev queries, k chosen by
leave-one-out); they differ only in features and allowed strategies.
Fallback = run HNSW with a budget of β·n_allowed distances, on overrun switch to BF.
Real data: Fashion-MNIST (60k × 784, filter = classes × synthetic price range) and
GloVe-100 (198k words, filter = frequency-rank range). 700 dev / 300 test queries.
Synthetic: dev on 3 modes; tests on held-out same modes ("grid") and on an unseen
"mixed" mode (out of distribution). Files: `results/exp2/<dataset>/`.

Mean distance computations per query on TEST (mean recall@10 in all cells ≥ 0.984):

| dataset / test | A: s only | B: s + fallback | C: local | D: local + fallback | oracle | best of A,B ÷ D |
|---|---|---|---|---|---|---|
| synth separated / grid | 9 782 | 6 187 | 5 027 | 5 025 | 4 757 | **1.23×** |
| synth separated / mixed (OOD) | 13 970 | 13 276 | 11 903 | 11 903 | 10 527 | **1.12×** |
| synth overlap / grid | 6 024 | 5 047 | 4 467 | 4 485 | 4 066 | **1.13×** |
| synth overlap / mixed (OOD) | 8 292 | 9 170 | 8 375 | 8 388 | 7 405 | **0.99×** |
| Fashion-MNIST (real) | 2 116 | 2 233 | 1 799 | 1 799 | 1 634 | **1.18×** |
| GloVe (real) | 6 445 | 6 509 | 5 987 | 6 040 | 4 253 | **1.07×** |

("best of A,B" is picked on the test set itself, which favours the baseline.)

### Verdict on H2: **not confirmed** (pre-registered bar: ≥1.3× on both geometries)
* The local feature helps in 5 of 6 tests, by 7–23%, and never hurts recall.
  On the out-of-distribution "mixed" overlap test it gives nothing (0.99×).
* The budget-and-fallback baseline is strong on synthetic data, as expected; on real
  data its extra options did not help the learner (B ≈ A).
* The realised gain (1.23×) is far below the Experiment 1 ceiling (2.6×) because
  real planners only see noisy features, and a good fallback already removes most
  of the catastrophic cases.
* Headroom left: on GloVe the oracle is 30% cheaper than every planner, so the
  *strategy set* (not the planner) is now the limit. That motivates Experiment 3.

### Side results from this cycle
* **Exact cluster pruning (triangle inequality, `fknn/ivf_exact.py`): rejected as a
  main idea.** Probe: saves 30–36% of BF distances on synthetic data, ~0% on GloVe,
  and gets worse with more clusters (bounds are weak in high dimension).
* **Seeding plain hnswlib search near allowed points: no effect.** The best-first
  rule still expands every non-allowed point closer than the current answers.
* **Allowed-only 2-hop traversal from cluster seeds (`fknn/seeded.py`, ACORN-1-style
  on the unmodified hnswlib graph):** probe on synthetic data, s = 5%, neutral
  correlation: 370 distance computations vs 5 650 for HNSW and 5 015 for BF at
  recall 0.97, but ~30 000 adjacency entries scanned, and recall collapses under
  negative correlation (0.2–0.65). This is a known weakness of graph search for
  out-of-distribution queries (flat distance landscape), so S2H needs a planner
  to keep it away from those queries → Experiment 3.
