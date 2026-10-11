"""Synthetic data with controllable filter selectivity and query-filter correlation.

Model:
  * C Gaussian clusters, each with `per_cluster` points in R^d.
  * Every cluster owns a block of `labels_per_cluster` labels; each point gets a
    uniformly random label from its cluster's block. So a label "lives" in exactly
    one cluster, and choosing labels = choosing *where* the allowed points are.
  * A query is a fresh point drawn from a random cluster.
  * A filter is a set of labels P. Its global selectivity s = |P| / n_labels is the
    same in every correlation mode; only the location of allowed points differs:
      positive: labels from clusters nearest to q   (allowed points near q)
      neutral:  labels sampled uniformly at random  (allowed points everywhere)
      negative: labels from clusters farthest from q (allowed points far from q)
"""
from dataclasses import dataclass

import numpy as np

MODES = ("positive", "neutral", "negative")


@dataclass
class Dataset:
    X: np.ndarray          # (n, d) float32
    labels: np.ndarray     # (n,) int32
    centers: np.ndarray    # (C, d) float32
    cluster_of: np.ndarray  # (n,) int32
    labels_per_cluster: int
    sigma: float

    @property
    def n_labels(self) -> int:
        return len(self.centers) * self.labels_per_cluster


def make_dataset(n_clusters=50, per_cluster=2000, dim=64, labels_per_cluster=20,
                 center_scale=1.0, sigma=0.6, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, center_scale, size=(n_clusters, dim)).astype(np.float32)
    cluster_of = np.repeat(np.arange(n_clusters, dtype=np.int32), per_cluster)
    X = centers[cluster_of] + rng.normal(0.0, sigma, size=(len(cluster_of), dim)).astype(np.float32)
    labels = (cluster_of * labels_per_cluster
              + rng.integers(0, labels_per_cluster, size=len(cluster_of))).astype(np.int32)
    return Dataset(X.astype(np.float32), labels, centers, cluster_of, labels_per_cluster, sigma)


def _labels_from_cluster_order(ds: Dataset, order: np.ndarray, m: int, rng) -> np.ndarray:
    """Take whole label blocks of clusters in `order` until m labels are chosen."""
    lpc = ds.labels_per_cluster
    chosen = []
    for c in order:
        need = m - len(chosen)
        if need <= 0:
            break
        block = np.arange(c * lpc, (c + 1) * lpc)
        chosen.extend(block if need >= lpc else rng.choice(block, size=need, replace=False))
    return np.sort(np.asarray(chosen, dtype=np.int32))


def make_queries(ds: Dataset, n_queries: int, mode: str, selectivity: float, seed: int):
    """Return list of (q, allowed_labels). Filters with |P| = round(s * n_labels)."""
    if mode not in MODES:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    m = max(1, int(round(selectivity * ds.n_labels)))
    d = ds.X.shape[1]
    out = []
    for _ in range(n_queries):
        c = rng.integers(len(ds.centers))
        q = (ds.centers[c] + rng.normal(0.0, ds.sigma, size=d)).astype(np.float32)
        if mode == "neutral":
            P = np.sort(rng.choice(ds.n_labels, size=m, replace=False).astype(np.int32))
        else:
            dc = ((ds.centers - q) ** 2).sum(1)
            order = np.argsort(dc)
            if mode == "negative":
                order = order[::-1]
            P = _labels_from_cluster_order(ds, order, m, rng)
        out.append((q, P))
    return out
