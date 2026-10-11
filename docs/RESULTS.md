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
