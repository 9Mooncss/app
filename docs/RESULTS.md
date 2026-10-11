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

## Cycle 3 — Experiment 3: planners with a richer strategy set

Same protocol as Experiment 2, but planners may also choose
IVFX (exact cluster pruning) and S2H (allowed-only 2-hop graph traversal from
cluster × label seeds, ef ∈ {16, 64}). S2H is charged for adjacency scans:
cost = distances + w·edges, with w = time(edge scan)/time(distance) measured in C
(`bench/edge_vs_dist.c`): 0.035–0.067 at d=64, 0.016–0.021 at d=100,
0.002–0.003 at d=784. We use the worst (largest) value. Files: `results/exp3/`.

Mean cost per query on TEST (w ×1; recall@10 in brackets):

| dataset | best planner, old strategies (Exp 2) | best planner, + new strategies | gain | same with w ×3 |
|---|---|---|---|---|
| synth separated / grid | 5 025 (0.992) | 4 948 (0.988) | 1.02× | 1.01× |
| synth separated / mixed | 11 903 (0.996) | 11 828 (0.994) | 1.01× | 1.01× |
| synth overlap / grid | 4 485 (0.986) | 4 353 (0.985) | 1.03× | 1.01× |
| synth overlap / mixed | 8 388 (0.984) | 8 288 (0.988) | 1.01× | 1.01× |
| **Fashion-MNIST** | 1 799 (0.998) | **446 (0.978)** | **4.0×** | 3.6× |
| **GloVe** | 6 040 (0.991) | **3 456 (0.986)** | **1.75×** | 1.56× |

* On real data the new strategy set changes the picture; on synthetic data it
  does not (cluster structure there is too clean for graph traversal across clusters).
* With the new strategies the **local-selectivity features no longer help** on
  real data (C/D are worse than A/B: 847 vs 446 on Fashion, 4 010 vs 3 456 on
  GloVe). Extra noisy features hurt the k-NN learner, and the 64-distance
  overhead matters when queries cost a few hundred. H2 is now clearly rejected.
* Recall is slightly lower (0.978 vs 0.998 on Fashion) — hence Experiment 4.

## Cycle 4 — Experiment 4: recall-matched curves and the seed ablation

Each strategy swept over ef, no planner. 240 queries per dataset, broken down by
selectivity band. A2H = the same 2-hop traversal started from the usual hnswlib
entry point (ACORN-like ablation, NOT real ACORN: ACORN also builds a denser graph,
which would help it). Figure: `results/exp4/curves_real.png`.
Cheapest mean cost reaching mean recall@10 ≥ 0.95 ("–" = never reaches it):

| dataset / band | BF | HNSW (hnswlib) | A2H (ACORN-like) | S2H (seeded) |
|---|---|---|---|---|
| Fashion s<1% | **314** | 11 874 | – (max 0.62) | – (max 0.84) |
| Fashion 1–10% | 2 505 | 7 622 | 691 | **457** |
| Fashion ≥10% | 12 075 | 2 362 | **578** | 586 |
| GloVe s<1% | **742** | 46 287 | – (max 0.69) | – (max 0.80) |
| GloVe 1–10% | 7 501 | 24 011 | 3 860 | **3 784** |
| GloVe ≥10% | 52 557 | 4 914 | 3 932 | **3 849** |
| synth separated, neutral | 12 774 | 17 420 | – | **2 380** |
| synth overlap, s<1% | **433** | 45 468 | 3 567 | 1 510 |
| synth overlap, neutral | 12 774 | 21 507 | 3 525 | **3 048** |

What this shows:
1. **The big win is allowed-only 2-hop traversal itself** (the ACORN idea), even on
   the unmodified hnswlib graph: 2–5.5× fewer operations than BF and 1.3–17× fewer
   than hnswlib's filtered search at s ≥ 1%. This is *known* and is what modern
   vector databases already ship (ACORN in Weaviate and Qdrant).
2. **Our addition (cluster × label seeds) helps only where the allowed subgraph is
   fragmented**: separated clusters (A2H never reaches 0.95, S2H does at 2 380),
   Fashion 1–10% (1.5×), and recall at s<1% (+0.1–0.2). On GloVe it is ≈ 1.0–1.02×.
3. **Below 1% selectivity nothing graph-based beats brute force** on these sizes.
4. **Negative correlation stays unsolved by graphs**: every graph method fails to
   reach recall 0.95; only BF does.

## Overall verdict after 4 cycles (honest)

* **Confirmed, but not new:** (a) global selectivity alone mispredicts cost (H1);
  (b) a per-query choice between brute force and 2-hop filtered graph search beats
  hnswlib's filtered search by large factors on real data. Both are consistent with
  the published literature (ACORN, RACORN-1, PostgreSQL study 2603.23710).
* **Our own ideas, measured:**
  - Local-selectivity planner (H2): +7–23% with old strategies, **0% or negative**
    with modern strategies → rejected.
  - Exact triangle-inequality cluster pruning: ≤ 36% on synthetic, ~0% on real →
    rejected.
  - Seeding plain hnswlib search: no effect → rejected.
  - Cluster × label seeds for 2-hop traversal: real but **narrow** gain (fragmented
    allowed subgraphs; 1.5× on Fashion 1–10%), close to existing ideas (filtered
    entry points in Weaviate; cluster restarts in arXiv:2604.00102) → not a moat.
* **No defensible algorithmic advantage over the state of the art has been shown.**
  We also did not run real ACORN or RACORN-1, so even the narrow gain is not
  established against them.

### Where the remaining opportunity is (by the data)
* The **s < 1% band**: graphs fail and brute force costs s·n, which grows linearly
  with the database (at 1B vectors and s = 0.1%, 1M distances per query). This is
  the regime targeted by Curator (arXiv:2601.01291) and RACORN-1 — active, crowded.
* **Negative correlation** (out-of-distribution w.r.t. the allowed subset): all graph
  methods fail. Exact or partition-based methods are the only reliable option today.

## Cycle 5 — exact filtered search faster than brute force (where graphs fail)

Motivation: Experiments 1–4 show that below ~1% selectivity and under negative
correlation every graph method misses recall 0.95 and brute force (BF) is the only
reliable method. BF is the fallback in every vector DB. Goal: exact answers, less work.

Hypotheses tested (all methods exact; tests in `tests/test_correctness.py`):
| id | idea | result | verdict |
|---|---|---|---|
| H5.1 | PCA m-dim lower bound + Seidl–Kriegel multi-step (`fknn/exact_lb.py`) | Fashion 3–12× fewer multiply-adds; GloVe 0.74–1.04× (flat spectrum: 32 PCs = 49% variance) | works only for fast-decaying spectra |
| H5.2 | + residual-norm term (r(q)−r(x))² | +3–7% fewer ops | small, free |
| H5.6 | local (per-cluster) PCA | no robust gain (apparent GloVe gain is a small-sample artefact; Fashion worse) | rejected |
| H5.7 | 4-bit per-dimension VA codes, exact LB via table lookups, table rounded **down** to uint8 (keeps exactness) (`fknn/exact_va.py`) | refine only 1.5–30% of allowed points on GloVe *and* Fashion; PCA rotation hurts | **works on both spectra** |
| H5.5 | C/AVX2 fast-scan kernel (`fknn/c/fscan.c`), per-label contiguous code blocks, vs the strongest BF (contiguous label-sorted floats) | see table | **confirmed (real time)** |

Experiment 5 — single-thread wall time, exact results verified on every query
(`experiments/exp5_exact_scan.py`, `results/exp5/`). Speedup = total BF time ÷ total VA time:

| dataset | all queries | s<1% | 1–10% | ≥10% |
|---|---|---|---|---|
| Fashion-MNIST (60k×784) | **7.2×** | 2.3× | 5.8× | 8.7× |
| GloVe (198k×100) | **3.4×** | 0.97× | 2.4× | 4.0× |
| synthetic separated (100k×64) | 1.9× | 0.73× | 1.2× | 2.0× |
| synthetic overlap (100k×64) | 2.3× | 0.87× | 1.6× | 2.5× |

Caveats: this CPU has a 260 MB L3 cache, so every dataset is cache-resident; BF is
then compute-bound, which *favours BF* (VA reads 8× fewer bytes). At s<1% with small
d, the fixed cost of building the lookup table loses to BF → use BF there (simple rule).
Prior art: VA-file (1998), VA+-file (2000), FAISS fast-scan (2019+), multi-step k-NN
(1998). The combination for exact *filtered* search is engineering, not a new theory.

## Cycle 6 — real competitors, scale, and kernel iterations

### Competitors (all single thread, same queries, same machine)
* **ACORN** — the official C++ implementation (stanford-futuredata/ACORN, SIGMOD'24),
  built from source; harness `bench/acorn_bench.cpp`, driver `experiments/exp7_vs_acorn.py`.
  ACORN-γ (M=32, γ=12, M_β=64; the paper's low-selectivity setting) and ACORN-1.
* **FAISS 1.15 HNSW** with a native bitmap `IDSelector` (`experiments/exp6_vs_faiss.py`).
  Its filtered search caps graph expansion, so recall stays low (≤0.73 on Fashion even at
  ef=1024): it is a weak baseline here and is not used for conclusions.

### Kernel iterations (exact results verified on every query)
| kernel | idea | outcome |
|---|---|---|
| v1 | 4-bit codes, AVX2 `pshufb`, 8-bit table | 2–9× over BF; loses at s<1% |
| v2 | AVX-512, 6-bit *table*, 4 dims pre-summed in uint8 | **slower**: coarser table → 1.5–3× more refinement; refinement, not scanning, dominated (my estimate was wrong) |
| H5.8 | PCA top-m dims + residual-norm term | rejected (Fashion refines 12–14% vs 2.6%; flat spectra can't drop dims) |
| **v3** | **6-bit codes via AVX-512 VBMI `vpermb` (64-entry LUT, 64 lookups/instr)**, 8-bit table rounded down, table build vectorized (14.7 µs vs 130 µs for d=784) | **best**: refinement 5–20× lower than v1 |
| v3-7bit | `vpermi2b`, 128-entry LUT | no further gain (table cost) |

### Experiment 9 — VA3 (6-bit) vs strongest brute force, exact, single thread
(`results/exp9/summary.json`; speedup = total BF time / total VA3 time)
| dataset | n × d | all | s<1% | 1–10% | ≥10% |
|---|---|---|---|---|---|
| Fashion-MNIST | 60k × 784 | **8.8×** | 3.4× | 8.0× | 9.8× |
| GloVe | 198k × 100 | **4.0×** | 1.7× | 3.7× | 4.2× |
| spaCy (out of cache, 409 MB) | 341k × 300 | **7.1×** | 2.5× | 5.6× | 7.9× |
| synthetic (out of cache, 512 MB) | 1M × 128 | **6.3×** | 3.5× | 5.1× | 6.5× |
| synthetic separated | 100k × 64 | 2.6× | 1.2× | 2.1× | 2.7× |
| synthetic overlap | 100k × 64 | 2.8× | 1.2× | 2.4× | 2.9× |

### Exact VA3 vs ACORN-γ (approximate) — `results/summary_vs_acorn.md`, `.png`
Mean µs/query; ACORN-γ at the smallest ef reaching mean recall ≥ 0.95:
| dataset | s<1% | 1–10% | ≥10% |
|---|---|---|---|
| Fashion | VA3 53 · ACORN never | VA3 **162** · ACORN 292 | VA3 775 · ACORN **202** |
| GloVe | VA3 21 · ACORN never | VA3 **99** · ACORN 696 | VA3 536 · ACORN 576 |
| spaCy | VA3 124 · ACORN never | VA3 **589** · ACORN 2 696 | VA3 2 663 · ACORN 2 589 |
| synthetic ×2 | ACORN never reaches 0.95 in any band | | |

Reading: **below 10% selectivity, exact VA3 is faster than ACORN-γ even when ACORN is
allowed 5% errors** (1.8–7× where ACORN reaches 0.95 at all). At ≥10% it ties on
GloVe/spaCy and loses 3.8× on Fashion. Caveats: VA3 is linear in the allowed set, so at
much larger n and high selectivity graphs win more; ACORN was not re-tuned per dataset
(paper defaults); all data except spaCy/synth-1M are cache-resident; ACORN build takes
minutes and much more memory than VA3 codes (d bytes per vector).

## Cycle 7 — prior art, PDX, hybrid planner, early abandon

### Prior-art check (literature agent, 2026-10-11; arxiv/vldb pages were not reachable, so partly from abstracts)
* **Prior art:** PQ Fast Scan (VLDB'16) — quantized-LUT lower bounds with saturating SIMD adds and
  heap-bound pruning (exact w.r.t. PQ distance, not true L2); Quicker ADC (TPAMI'19) — AVX-512 VBMI
  `vpermi2b` LUT scans; VA-file / VA+-file; PDX / PDX-BOND (SIGMOD'25) — exact pruned scans beating
  SIMD brute force; Curator / ACORN / RACORN-1 — brute force wins at low selectivity; Vespa (5%),
  Weaviate (~15%) switch to exact search below a selectivity threshold.
* **Not found** (absence of evidence ≠ proof): exact filtered k-NN with *true-L2* deterministic
  bounds from floor-rounded LUTs + byte-permute SIMD + per-label contiguous code blocks; `vpermb`
  on 6-bit per-dimension scalar codes.
* **Consequence:** "exact beats ACORN at low selectivity" is *not* new by itself (pre-filter BF is
  known to win there). The defensible claim is: **exact filtered search 4–9× faster than
  contiguous exact brute force**, competitive with the best exact pruned scan (PDX).

### Experiment 11 — VA3 vs PDX filtered search (`results/exp11`)
| dataset | VA3 exact | PDX exhaustive (recall) | PDX nprobe 32 (recall) |
|---|---|---|---|
| Fashion | **246 µs** | 464 µs (0.985) | 269 µs (0.69) |
| GloVe | **181 µs** | 1 695 µs (0.996) | 243 µs (0.60) |
| spaCy | **870 µs** | 3 648 µs (0.995) | 906 µs (0.63) |
| synthetic 1M | **2 021 µs** | 8 020 µs (0.999) | 1 528 µs (0.34) |
Part of this gap is the filter-aware layout (PDX takes an arbitrary row-id list).

### Experiment 12 — no filter, kernel vs kernel (`results/exp12`)
| dataset | BF | VA3 | VA4 (early abandon) | PDX exhaustive (recall) |
|---|---|---|---|---|
| Fashion (d=784) | 38.7 ms | 2.21 ms | 1.34 ms | **0.67–1.17 ms** (0.992) |
| GloVe | 8.9 ms | 1.63 ms | **1.04 ms** | 3.25 ms (1.0) |
| spaCy | 85.3 ms | 8.30 ms | **7.46 ms** | 9.24 ms (1.0) |
PDX timings varied ~1.7× between runs on Fashion. Kernels are comparable; PDX wins on the
784-d image data (dimension pruning), VA wins on the word-vector data.

### Experiment 10 — hybrid planner (`results/exp10`): VA3 below a selectivity threshold, ACORN-γ above
Tuned on even queries (target mean recall 0.95), evaluated on odd queries:
| dataset | BF | VA3 (exact) | ACORN-γ alone | hybrid (recall) |
|---|---|---|---|---|
| Fashion | 2 462 µs | 281 µs | cannot reach 0.95 | **82 µs** (0.946) |
| GloVe | 656 µs | 171 µs | cannot reach 0.95 | **56 µs** (0.934) |
| spaCy | 6 733 µs | 884 µs | cannot reach 0.95 | **393 µs** (0.958) |
ACORN-γ alone never reaches mean recall 0.95 on a mixed-selectivity workload (low-selectivity
queries sink it). The hybrid is 2.2–3.4× faster than exact VA3, but its held-out recall fell
slightly below target on 2 of 3 datasets — threshold tuning does not transfer perfectly.
ACORN-γ itself varied between rebuilds (Fashion 1–10%: 292 µs vs 1 500 µs at 0.95).

### Experiment 13 — VA4 (variance-ordered dims, streaming refine, block early-abandon), filtered
Speedup over BF, all queries: Fashion 8.9×, GloVe **5.1×** (VA3: 4.0×), spaCy 6.0× (VA3: 7.1×),
synthetic 1M 6.6× (VA3: 6.3×). Mixed: better on GloVe and at high selectivity, worse at s<1% and
on spaCy (streaming refinement starts with a loose threshold). Rule of thumb: VA3 for small
allowed sets, VA4 for large.
