"""Instrumented re-implementation of hnswlib's (v0.8.0) filtered search.

Why: hnswlib's Python API does not expose how many distances a query computed,
and its Python filter callback adds per-node overhead that distorts timings.
We load the *exact graph* hnswlib built (by parsing its saved index file) and
replay the same search algorithm in Python, counting distance computations.
That count is our hardware-independent cost metric.

Algorithm mirrored: HierarchicalNSW::searchKnn -> greedy descent through upper
layers (no filter), then searchBaseLayerST with filter on layer 0, ef' = max(ef, k).
"""
import heapq
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class HNSWGraph:
    level0: np.ndarray        # (N, maxM0) int64, neighbor internal ids, padded with -1
    level0_count: np.ndarray  # (N,) number of valid neighbors
    upper: list               # upper[i] = {level: np.ndarray of neighbor ids} for internal id i
    ext_label: np.ndarray     # internal id -> external label
    entry: int
    maxlevel: int


def load_hnswlib_index(path: str, dim: int) -> HNSWGraph:
    with open(path, "rb") as f:
        buf = f.read()
    (offset_level0, max_elements, cur_count, size_per_elem, label_offset, offset_data
     ) = struct.unpack_from("<6Q", buf, 0)
    pos = 48
    maxlevel, entry = struct.unpack_from("<iI", buf, pos); pos += 8
    maxM, maxM0, M = struct.unpack_from("<3Q", buf, pos); pos += 24
    pos += 8 + 8  # mult (double), ef_construction (size_t)
    assert offset_level0 == 0 and offset_data == 4 + 4 * maxM0, "unexpected hnswlib layout"

    raw = np.frombuffer(buf, dtype=np.uint8, count=cur_count * size_per_elem, offset=pos)
    raw = raw.reshape(cur_count, size_per_elem)
    pos += cur_count * size_per_elem
    head = raw[:, :4 + 4 * maxM0].copy().view(np.uint32)
    level0_count = (head[:, 0] & 0xFFFF).astype(np.int64)
    level0 = head[:, 1:].astype(np.int64)
    level0[np.arange(maxM0)[None, :] >= level0_count[:, None]] = -1
    ext_label = raw[:, label_offset:label_offset + 8].copy().view(np.uint64)[:, 0].astype(np.int64)

    size_links = 4 * maxM + 4
    upper = [None] * cur_count
    for i in range(cur_count):
        (ll_size,) = struct.unpack_from("<I", buf, pos); pos += 4
        if ll_size:
            levels = {}
            for lev in range(ll_size // size_links):
                o = pos + lev * size_links
                (cnt,) = struct.unpack_from("<I", buf, o)
                cnt &= 0xFFFF
                levels[lev + 1] = np.frombuffer(buf, dtype=np.uint32, count=cnt, offset=o + 4).astype(np.int64)
            upper[i] = levels
            pos += ll_size
    return HNSWGraph(level0, level0_count, upper, ext_label, int(entry), int(maxlevel))


def search(g: HNSWGraph, Xint: np.ndarray, q: np.ndarray, k: int, ef: int,
           allowed_int: np.ndarray | None = None):
    """Filtered kNN as in hnswlib. Xint: vectors indexed by *internal* id.
    allowed_int: bool mask over internal ids (None = no filter).
    Returns (external labels nearest-first, squared dists, n_distance_computations)."""
    ndist = 0

    def dist1(i):
        v = Xint[i] - q
        return float(np.dot(v, v))

    # 1) greedy descent through upper layers (unfiltered, as in hnswlib)
    cur = g.entry
    curd = dist1(cur); ndist += 1
    for level in range(g.maxlevel, 0, -1):
        changed = True
        while changed:
            changed = False
            nb = g.upper[cur][level]
            if len(nb) == 0:
                continue
            diff = Xint[nb] - q
            ds = np.einsum("ij,ij->i", diff, diff); ndist += len(nb)
            for c, d in zip(nb.tolist(), ds.tolist()):
                if d < curd:
                    curd, cur, changed = d, c, True

    # 2) layer-0 best-first search with filter
    ef = max(ef, k)
    visited = np.zeros(len(Xint), dtype=bool)
    top = []    # max-heap via (-dist, -id): top[0] = farthest kept result
    cand = []   # min-heap via (dist, -id): cand[0] = closest frontier node
    inf = float("inf")
    if allowed_int is None or allowed_int[cur]:
        d0 = dist1(cur); ndist += 1
        lower = d0
        heapq.heappush(top, (-d0, -cur))
        heapq.heappush(cand, (d0, -cur))
    else:
        lower = inf
        heapq.heappush(cand, (inf, -cur))
    visited[cur] = True

    while cand:
        cd, negc = cand[0]
        if cd > lower and len(top) == ef:
            break
        heapq.heappop(cand)
        c = -negc
        nb = g.level0[c, :g.level0_count[c]]
        nb = nb[~visited[nb]]
        if len(nb) == 0:
            continue
        visited[nb] = True
        diff = Xint[nb] - q
        ds = np.einsum("ij,ij->i", diff, diff); ndist += len(nb)
        ok = allowed_int[nb] if allowed_int is not None else np.ones(len(nb), dtype=bool)
        for j, d, a in zip(nb.tolist(), ds.tolist(), ok.tolist()):
            if len(top) < ef or lower > d:
                heapq.heappush(cand, (d, -j))
                if a:
                    heapq.heappush(top, (-d, -j))
                while len(top) > ef:
                    heapq.heappop(top)
                if top:
                    lower = -top[0][0]

    res = sorted((-nd, -ni) for nd, ni in top)[:k]
    ids = np.array([g.ext_label[i] for _, i in res], dtype=np.int64)
    dists = np.array([d for d, _ in res], dtype=np.float32)
    return ids, dists, ndist
