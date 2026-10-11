"""Exact filtered k-NN with quantization lower bounds (VA-file / VA+-file idea:
Weber et al. 1998; Ferhatosmanoglu et al. 2000), multi-step refinement.

Each coordinate (optionally after a PCA rotation) is scalar-quantized into 2^b
cells with per-dimension equi-depth boundaries. For a query, a table
    T[i][c] = (distance from q_i to cell c of dimension i)^2   (0 if q_i inside)
gives an exact lower bound  LB(x) = sum_i T[i][code_i(x)] <= ||q - x||^2
(the rotation is orthonormal, so distances are preserved).
If T is rounded DOWN to small integers (as SIMD "fast-scan" kernels need), LB stays
a valid lower bound - exactness is kept.

Cost model: LB = d table lookups on b-bit codes (counted as "lookups", cheap);
exact distance = d multiply-adds; query rotation = d*d (if rotate=True).
"""
import numpy as np


class VABound:
    def __init__(self, X, bits=4, rotate=False, sample=20000, seed=0):
        rng = np.random.default_rng(seed)
        S = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)].astype(np.float64)
        self.mu = S.mean(0).astype(np.float32)
        if rotate:
            evals, evecs = np.linalg.eigh(np.cov(S - self.mu, rowvar=False))
            self.R = evecs[:, np.argsort(evals)[::-1]].astype(np.float32)
        else:
            self.R = None
        Z = self._tr(X)
        Zs = self._tr(S.astype(np.float32))
        L = 2 ** bits
        qs = np.linspace(0, 1, L + 1)[1:-1]
        self.edges = np.quantile(Zs, qs, axis=0).T.astype(np.float32)  # (d, L-1) inner boundaries
        lo = np.concatenate([np.full((Z.shape[1], 1), -np.inf, np.float32), self.edges], 1)
        hi = np.concatenate([self.edges, np.full((Z.shape[1], 1), np.inf, np.float32)], 1)
        self.lo, self.hi = lo, hi                                        # (d, L) cell intervals
        self.codes = np.empty(Z.shape, dtype=np.uint8)
        for i in range(Z.shape[1]):
            self.codes[:, i] = np.searchsorted(self.edges[i], Z[:, i], side="right")
        self.X, self.d, self.bits = X, X.shape[1], bits

    def _tr(self, V):
        V = V - self.mu
        return V @ self.R if self.R is not None else V

    def table(self, q):
        z = self._tr(q[None])[0][:, None]
        gap = np.maximum(self.lo - z, 0) + np.maximum(z - self.hi, 0)    # (d, L)
        return (gap * gap).astype(np.float32)

    def search(self, q, ids, k, batch=16, int8_table=False):
        T = self.table(q)
        if int8_table:   # round DOWN to uint8 levels with a shared scale -> still a lower bound
            scale = max(float(T.max()) / 255.0, 1e-12)
            T = np.floor(T / scale).astype(np.float32) * scale
        codes = self.codes[ids]
        lb = T[np.arange(self.d)[None, :], codes].sum(1)
        order = np.argsort(lb, kind="stable")
        best_d = np.full(k, np.inf)
        best_i = np.full(k, -1, dtype=np.int64)
        refined, pos, n = 0, 0, len(ids)
        while pos < n:
            if refined >= k and lb[order[pos]] >= best_d[-1]:
                break
            sel = order[pos:pos + batch]
            sel = sel[lb[sel] < best_d[-1]] if refined >= k else sel
            pos += batch
            if len(sel) == 0:
                continue
            xi = ids[sel]
            dd = self.X[xi] - q
            dist = np.einsum("ij,ij->i", dd, dd)
            refined += len(sel)
            allD = np.concatenate([best_d, dist])
            allI = np.concatenate([best_i, xi])
            s = np.argsort(allD, kind="stable")[:k]
            best_d, best_i = allD[s], allI[s]
        ok = best_i >= 0
        stats = dict(refined=refined, n=n, lookups=n * self.d,
                     flops=refined * self.d + (self.d * self.d if self.R is not None else 0),
                     ops_bf=n * self.d)
        return best_i[ok], best_d[ok], stats
