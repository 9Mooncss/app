"""Seeded filtered HNSW: start the layer-0 search from allowed points located by
the (cluster x label) posting lists, in addition to the normal greedy entry point.

Motivation (Experiment 1): under negative query-filter correlation, hnswlib's
filtered search starts next to q, far from every allowed point, and either explores
the whole graph or stops early with a wrong answer. Seeds placed in the nearest
clusters that *contain* allowed points let the search start where answers can be.
Related prior art: label-specific start nodes in Filtered-DiskANN (Gollapudi et al.,
WWW 2023). Here seeds depend on both the filter and the query.
"""
import heapq

import numpy as np


class SeedPicker:
    """Per (cluster, label) lists of point ids sorted by distance to the centroid."""

    def __init__(self, ivf, ext_to_int):
        self.ivf = ivf
        # re-sort each (cluster, label) posting range by distance to its centroid
        order = ivf.order.copy()
        st = ivf.start
        for a, b in zip(st[:-1], st[1:]):
            if b - a > 1:
                seg = order[a:b]
                order[a:b] = seg[np.argsort(ivf.dc[seg], kind="stable")]
        self.order_int = ext_to_int[order]

    def seeds(self, q, P, n_seeds, per_cluster=4):
        """Up to n_seeds allowed internal ids: the per_cluster most central allowed
        points of each cluster, clusters taken in order of distance to q.
        Returns (seed ids, n_distance_computations for the centroids)."""
        ivf = self.ivf
        dq = ((ivf.C - q) ** 2).sum(1)
        has = ivf.counts[:, P].sum(1) > 0
        L = ivf.n_labels
        out = []
        for c in np.argsort(dq):
            if not has[c]:
                continue
            base = c * L
            got = []
            for l in P:
                a, b = ivf.start[base + l], ivf.start[base + l + 1]
                if b > a:
                    got.extend(self.order_int[a:min(b, a + per_cluster)].tolist())
            # keep the most central ones of this cluster
            out.extend(got[:per_cluster])
            if len(out) >= n_seeds:
                break
        return np.asarray(out[:n_seeds], dtype=np.int64), len(ivf.C)


def seeded_search(g, Xint, q, k, ef, allowed_int, seeds):
    """hnswlib filtered layer-0 search, but candidate/result heaps start with
    `seeds` (allowed internal ids) plus the usual greedy-descent entry point."""
    ndist = 0
    cur = g.entry
    v = Xint[cur] - q
    curd = float(np.dot(v, v)); ndist += 1
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

    ef = max(ef, k)
    visited = np.zeros(len(Xint), dtype=bool)
    top, cand = [], []
    start = np.unique(np.concatenate([seeds, [cur]]))
    diff = Xint[start] - q
    ds = np.einsum("ij,ij->i", diff, diff); ndist += len(start)
    for j, d in zip(start.tolist(), ds.tolist()):
        visited[j] = True
        heapq.heappush(cand, (d, -j))
        if allowed_int[j]:
            heapq.heappush(top, (-d, -j))
            if len(top) > ef:
                heapq.heappop(top)
    lower = -top[0][0] if top else float("inf")

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
        ok = allowed_int[nb]
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
    return ids, np.array([d for d, _ in res], dtype=np.float32), ndist


def seeded_2hop_search(g, Xint, q, k, ef, allowed_int, seeds):
    """Best-first search restricted to ALLOWED nodes, started from `seeds`.

    Neighbors of an expanded node: its allowed 1-hop neighbors plus the allowed
    neighbors of its non-allowed 1-hop neighbors (2-hop), as in ACORN-1
    (Patel et al., SIGMOD 2024) but on the unmodified hnswlib layer-0 graph.
    Distances are computed only for allowed nodes, each at most once, so
    n_distance_computations <= |seeds| + n_allowed (never worse than brute force
    by more than the seed/centroid overhead).
    Returns (ids, dists, n_distance_computations, n_edges_scanned)."""
    ef = max(ef, k)
    visited = np.zeros(len(Xint), dtype=bool)
    top, cand = [], []
    ndist, nedges = 0, 0
    seeds = np.unique(seeds)
    if len(seeds):
        diff = Xint[seeds] - q
        ds = np.einsum("ij,ij->i", diff, diff); ndist += len(seeds)
        visited[seeds] = True
        for j, d in zip(seeds.tolist(), ds.tolist()):
            heapq.heappush(cand, (d, -j))
            heapq.heappush(top, (-d, -j))
            if len(top) > ef:
                heapq.heappop(top)
    lower = -top[0][0] if top else float("inf")
    L0, cnt = g.level0, g.level0_count
    while cand:
        cd, negc = cand[0]
        if cd > lower and len(top) == ef:
            break
        heapq.heappop(cand)
        c = -negc
        n1 = L0[c, :cnt[c]]
        nedges += len(n1)
        a1 = allowed_int[n1]
        blocked = n1[~a1]
        if len(blocked):
            n2 = L0[blocked].ravel()
            n2 = n2[n2 >= 0]
            nedges += len(n2)
            nb = np.concatenate([n1[a1], n2[allowed_int[n2]]])
        else:
            nb = n1
        nb = np.unique(nb[~visited[nb]])
        if len(nb) == 0:
            continue
        visited[nb] = True
        diff = Xint[nb] - q
        ds = np.einsum("ij,ij->i", diff, diff); ndist += len(nb)
        for j, d in zip(nb.tolist(), ds.tolist()):
            if len(top) < ef or lower > d:
                heapq.heappush(cand, (d, -j))
                heapq.heappush(top, (-d, -j))
                if len(top) > ef:
                    heapq.heappop(top)
                lower = -top[0][0]
    res = sorted((-nd, -ni) for nd, ni in top)[:k]
    ids = np.array([g.ext_label[i] for _, i in res], dtype=np.int64)
    return ids, np.array([d for d, _ in res], dtype=np.float32), ndist, nedges
