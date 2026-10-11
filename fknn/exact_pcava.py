"""H5.8: PCA rotation + 4-bit VA codes on the top-m components + residual-norm term.

For orthonormal top-m directions V (d x m) and residual norms r(.) of the dropped
(d - m)-dim subspace:
    ||q - x||^2 = ||V^T(q - x)||^2 + ||res(q) - res(x)||^2
                >= sum_{i<m} gap_i(q, cell_i(x))^2 + (r(q) - r(x))^2
The first term is the VA cell bound on the m kept coordinates; the second the
reverse triangle inequality on the dropped subspace. Exact, and the code scan
touches m instead of d coordinates.
"""
import numpy as np


class PCAVABound:
    def __init__(self, X, m, bits=4, sample=20000, seed=0):
        rng = np.random.default_rng(seed)
        S = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)].astype(np.float64)
        self.mu = S.mean(0).astype(np.float32)
        evals, evecs = np.linalg.eigh(np.cov(S - self.mu, rowvar=False))
        self.V = evecs[:, np.argsort(evals)[::-1][:m]].astype(np.float32)
        Xc = X - self.mu
        Z = Xc @ self.V
        self.r = np.sqrt(np.maximum(np.einsum("ij,ij->i", Xc, Xc) - np.einsum("ij,ij->i", Z, Z), 0)).astype(np.float32)
        L = 2 ** bits
        Zs = (S.astype(np.float32) - self.mu) @ self.V
        edges = np.quantile(Zs, np.linspace(0, 1, L + 1)[1:-1], axis=0).T.astype(np.float32)
        self.lo = np.concatenate([np.full((m, 1), -np.inf, np.float32), edges], 1)
        self.hi = np.concatenate([edges, np.full((m, 1), np.inf, np.float32)], 1)
        self.codes = np.empty(Z.shape, dtype=np.uint8)
        for i in range(m):
            self.codes[:, i] = np.searchsorted(edges[i], Z[:, i], side="right")
        self.X, self.m, self.d = X, m, X.shape[1]

    def search(self, q, ids, k, batch=16, int8_table=True):
        qc = q - self.mu
        z = qc @ self.V
        rq = np.sqrt(max(float(qc @ qc) - float(z @ z), 0.0))
        gap = np.maximum(self.lo - z[:, None], 0) + np.maximum(z[:, None] - self.hi, 0)
        T = (gap * gap).astype(np.float32)
        if int8_table:
            scale = max(float(T.max()) / 255.0, 1e-12)
            T = np.floor(T / scale) * scale
        lb = T[np.arange(self.m)[None, :], self.codes[ids]].sum(1) + (self.r[ids] - rq) ** 2
        order = np.argsort(lb, kind="stable")
        best_d = np.full(k, np.inf); best_i = np.full(k, -1, dtype=np.int64)
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
            allD = np.concatenate([best_d, dist]); allI = np.concatenate([best_i, xi])
            s = np.argsort(allD, kind="stable")[:k]
            best_d, best_i = allD[s], allI[s]
        ok = best_i >= 0
        return best_i[ok], best_d[ok], dict(refined=refined, n=n, scan_dims=self.m)
