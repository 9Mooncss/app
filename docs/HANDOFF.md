# Handoff — read this first in a new session

Goal of the project (user's brief): find a computational problem where a better *algorithm*
(not hardware) gives a measurable, honest advantage, then check money potential. User is a
beginner, communicates in Russian, wants autonomous work, honest results, plain explanations.

## Where things stand (2026-10-12)
1. **Filtered vector search (cycles 1–8)** — `fknn/`, `docs/RESULTS.md` cycles 1–8.
   Confirmed: exact filtered k-NN (VA3 kernel, AVX-512 VBMI) 4–9× faster than contiguous brute
   force; faster than ACORN-γ for selectivity < 10%. Narrow, copyable → modest money (consulting /
   OSS library). Limits: scattered predicates, high selectivity.
2. **MIPS / LLM attention transfer (cycle 9)** — closed: ≈1× vs batched BLAS on real data
   (incl. real RoBERTa attention keys). Lesson: self-made synthetic data can fake a win.
3. **Oven Scheduling Problem (cycle 10, CURRENT)** — `osp/`, `docs/OSP_PLAN.md`, RESULTS cycle 10.
   Benchmark + authors' solver: github.com/iolab-uniud/osp-ls (setup: `scripts/setup_osp_ls.sh`).
   Our C++ SA (`osp/cpp/solver.cpp`, incremental suffix evaluation with re-synchronisation):
   * 30 s beats the published SA (5M evals, 300–900 s) on 115/120 held-out large instances;
     new best-known solutions on 106/120.
   * Same machine: equal time 12/12 better (−1.85%); our 30 s beats their 5M runs (225–540 s) 12/12.
   * Evaluator `osp/instance.py` matches the authors' solver exactly (tests/test_osp.py).

## Running / pending
* GitHub Actions (repo is PUBLIC → free minutes):
  `tests.yml` on every push; `osp-bench.yml` (workflow_dispatch: seconds, seeds) = equal-time
  comparison on all 120 large instances over 12 parallel jobs; summary artifact `summary`.
  First run: https://github.com/9Mooncss/app/actions/runs/38240543826
* Local: `python -m osp.huge results/osp/huge_120s.json 120` (huge 1000–5000-job instances, ours vs
  theirs at equal time) — results in results/osp/huge_120s.json / log_huge.txt when done.

## Next steps (planned)
1. Collect GitHub equal-time results (all 120) and huge-instance results; add to RESULTS.md.
2. Compare with published lower bounds (arXiv 2410.01368) to know the remaining gap.
3. Quality improvements: exact re-batching moves, ruin-and-recreate (H10.2).
4. Money: validate with real heat-treatment shops (pain, current tools, energy cost); see the
   commercial discussion in chat / docs/COMMERCIAL.md (vector-search draft) — OSP version to write.

## Practical notes
* Environment: 4 cores, AVX-512 BW+VBMI, 260 MB L3, 15 GB RAM. HuggingFace/zenodo/arxiv blocked;
  GitHub, PyPI, GCS reachable. Long jobs: use run_in_background with timeout up to 2 h and
  make scripts resumable (see osp/benchmark.py run cache).
* Timing experiments must not share CPU with other heavy jobs.
