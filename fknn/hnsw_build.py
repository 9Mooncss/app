"""Build an hnswlib index and load it into the instrumented replica."""
import os

import hnswlib
import numpy as np

from .hnsw_replica import load_hnswlib_index


def build(X: np.ndarray, path: str, M=16, ef_construction=200, seed=100, threads=1):
    """threads=1 keeps graph construction deterministic.
    Returns (hnswlib.Index, HNSWGraph, Xint) where Xint[i] = vector of internal id i."""
    n, d = X.shape
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(max_elements=n, M=M, ef_construction=ef_construction, random_seed=seed)
    idx.set_num_threads(threads)
    idx.add_items(X, np.arange(n))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    idx.save_index(path)
    g = load_hnswlib_index(path, d)
    Xint = np.ascontiguousarray(X[g.ext_label])
    return idx, g, Xint
