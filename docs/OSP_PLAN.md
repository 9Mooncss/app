# Cycle 10 — Oven Scheduling Problem (OSP): plan

## Why this problem
* Industrial (heat-treatment ovens: energy, setups, due dates), open benchmark
  (github.com/iolab-uniud/osp-ls; Lackner et al. CP'21/Constraints'23; Da Ros et al. GECCO'24, C&OR'25).
* Young benchmark, few competitors. State of the art (from the repo's processed results):
  small instances (10–100 jobs) are essentially solved (CP/ILP/LNS within 0–1.7% of best);
  on large instances (250–500 jobs) the authors' multi-neighbourhood simulated annealing (SA,
  5M evaluations, ~470 s) is best, LNS/CP are 15–31% worse and CP finds no solution for 7–18 of 40;
  the 1000-job "huge" instances have no published results in the repo.
* Lower bounds exist (arXiv 2410.01368) → gaps can be measured.

## Ground truth
`osp/instance.py` re-implements the authors' cost exactly; it matches their C++ solver on
12/12 reference solutions (3 use cases × 4 sizes), total cost and all components.

## Hypotheses
* H10.1 (speed): an SA with O(affected-suffix) incremental evaluation in C++ runs ≥10× more
  evaluations per second than the authors' easylocal implementation → better solutions at equal time.
* H10.2 (neighbourhoods): exact re-batching moves (re-partition one machine's same-attribute jobs)
  and ruin-and-recreate improve quality beyond what more iterations give.
* Success: on held-out large instances (81–120, all 3 use cases), at equal single-thread wall time,
  mean normalized cost lower than the authors' SA, and/or reach their published 5M-evaluation
  quality in ≤ 1/5 of the time. Parameters tuned only on the tuning-use-case-* instances.
