"""Exact filtered k-NN over a clustered inverted index, with triangle-inequality pruning.

Classic branch-and-bound (Fukunaga & Narendra 1975; same spirit as exact IVF search),
combined with per-(cluster, label) posting lists so only allowed points are touched.

For centroid c with radius r_c and a point x in it:
    d(q, x) >= |d(q, c) - d(x, c)|   >= d(q, c) - r_c
So once we hold k candidates with k-th distance t, a cluster with d(q,c) - r_c >= t
and a point with |d(q,c) - d(x,c)| >= t can be skipped without computing d(q, x).
Answers are exact. Cost = n_centroids + number of point distances actually computed.
Bound checks are scalar ops on precomputed numbers; reported separately.
"""
import numpy as np


class IVFExact:
    def __init__(self, X, labels, n_labels, centroids, assign):
        self.X, self.C = X, centroids
        nc = len(centroids)
        key = assign.astype(np.int64) * n_labels + labels
        self.order = np.argsort(key, kind="stable")
        counts = np.bincount(key, minlength=nc * n_labels)
        self.start = np.concatenate([[0], np.cumsum(counts)]).reshape(-1)  # flat (c, l) -> range
        self.n_labels = n_labels
        self.counts = counts.reshape(nc, n_labels)
        self.dc = np.sqrt(((X - centroids[assign]) ** 2).sum(1))           # ||x - c(x)||
        self.dc_sorted = self.dc[self.order]
        self.radius = np.zeros(nc)
        np.maximum.at(self.radius, assign, self.dc)

    def search(self, q, P, k):
        """Return (ids, squared dists, n_distance_computations, n_bound_checks)."""
        L = self.n_labels
        dq = np.sqrt(((self.C - q) ** 2).sum(1))
        ndist, nbound = len(self.C), 0
        best_d = np.full(k, np.inf)
        best_i = np.full(k, -1, dtype=np.int64)
        t = np.inf
        has = self.counts[:, P].sum(1) > 0
        for c in np.argsort(dq):
            if not has[c]:
                continue
            nbound += 1
            if dq[c] - self.radius[c] >= t:
                continue
            base = c * L
            segs = [np.arange(self.start[base + l], self.start[base + l + 1]) for l in P
                    if self.start[base + l + 1] > self.start[base + l]]
            if not segs:
                continue
            pos = np.concatenate(segs)
            nbound += len(pos)
            pos = pos[np.abs(dq[c] - self.dc_sorted[pos]) < t]
            if len(pos) == 0:
                continue
            ids = self.order[pos]
            diff = self.X[ids] - q
            d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            ndist += len(ids)
            allD = np.concatenate([best_d, d])
            allI = np.concatenate([best_i, ids])
            sel = np.argsort(allD, kind="stable")[:k]
            best_d, best_i = allD[sel], allI[sel]
            t = best_d[-1]
        ok = best_i >= 0
        return best_i[ok], best_d[ok] ** 2, ndist, nbound
