# Stage 8 — commercial assessment (draft, honest)

## What the technology is
An **exact** filtered vector-search engine: for queries like "10 nearest products *in these
categories*", it returns the provably exact answer 4–9× faster than brute force (the fallback every
vector database uses at low selectivity), and below ~10% selectivity it is faster than the
state-of-the-art approximate filtered index (ACORN) even when ACORN is allowed 5% errors.

## Who has the problem
* Vector databases and search engines: their planners switch to brute force below a selectivity
  threshold (Vespa 5%, Weaviate ~15%, Qdrant/Lucene planners). Those queries are their slowest.
* Applications where filters are selective and correctness matters: e-commerce search inside a
  category/brand/price band; multi-tenant SaaS (tenant = filter: every query is a low-selectivity
  query); legal/compliance/medical retrieval where "approximately right" is not acceptable;
  RAG over per-user document sets.

## What a customer would pay for
* Lower p95/p99 latency and CPU cost on filtered queries (the measured 4–9× is per-core CPU time).
* "Exact" as a guarantee (audits, reproducibility), not as an expensive mode.
* Smaller memory than graph indexes (d bytes/vector of codes, no adjacency lists).

## Competition and moat (weak)
* The components are public; a strong vector-DB team could re-implement this in weeks. FAISS,
  PDX, Lucene could add exact lower-bound scans.
* A moat would have to come from: (1) the per-attribute layout + planner as a complete engine,
  (2) speed of integration (e.g. a drop-in library/extension: pgvector, DuckDB, SQLite, Lucene),
  (3) benchmarks and trust. None of these is an algorithmic monopoly.

## Most plausible paths
1. **Open-source library + paid support/consulting** (low barrier, builds credibility).
2. **Extension for PostgreSQL (pgvector users) / DuckDB**: exact filtered k-NN operator. The
   PostgreSQL study (arXiv 2603.23710, 2602.11443) shows filtered queries are a pain point there.
3. Contribution to an existing engine (Qdrant/Weaviate/Lucene) as a portfolio piece.

## How to prove value to a customer
Run their real filtered workload (vectors + filter logs) through `experiments/` style benchmarks:
report p50/p99 latency, CPU per query, recall (=1.0), memory, vs their current setup.

## Not yet validated (blocks any commercial claim)
Multi-threaded throughput; datasets ≥ 10M; inserts/deletes; scattered multi-attribute filters;
ARM / AVX2-only CPUs (VA3 needs AVX-512 VBMI; the AVX2 v1 kernel is 2–4× over BF).
