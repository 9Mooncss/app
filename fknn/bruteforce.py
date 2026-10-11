"""Exact filtered k-NN: pre-filter with an inverted label index, then brute force."""
import numpy as np


class LabelIndex:
    """label -> sorted array of point ids (an inverted index, built once)."""

    def __init__(self, labels: np.ndarray, n_labels: int):
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=n_labels)
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        self.order = order.astype(np.int64)

    def ids(self, P: np.ndarray) -> np.ndarray:
        return np.concatenate([self.order[self.starts[l]:self.starts[l + 1]] for l in P])


def bruteforce_knn(X: np.ndarray, q: np.ndarray, ids: np.ndarray, k: int):
    """Return (ids sorted by distance, squared distances, n_distance_computations)."""
    diff = X[ids] - q
    d2 = np.einsum("ij,ij->i", diff, diff)
    kk = min(k, len(ids))
    part = np.argpartition(d2, kk - 1)[:kk]
    part = part[np.lexsort((ids[part], d2[part]))]
    return ids[part], d2[part], len(ids)
