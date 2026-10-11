"""Exact filtered k-NN cheaper than brute force: lower bounds + multi-step refinement.

Lower bound (exact, never overestimates the true squared L2 distance):
  Let mu be the data mean, V_m the top-m principal directions (orthonormal), and
  for any vector v: p(v) = V_m^T (v - mu)  (m-dim projection),
                    r(v) = || (v - mu) - V_m p(v) ||  (norm of the residual).
  Because the projection and the residual live in orthogonal subspaces,
      ||q - x||^2 = ||p(q) - p(x)||^2 + ||res(q) - res(x)||^2
                 >= ||p(q) - p(x)||^2 + (r(q) - r(x))^2           (reverse triangle ineq.)
  The first term alone is the classic PCA partial-distance bound; the second
  term ("residual-norm term") costs one stored float per point.

Search = optimal multi-step k-NN (Seidl & Kriegel, SIGMOD 1998):
  compute LB for every allowed point, visit points in increasing LB order, compute
  the exact distance, stop as soon as the next LB >= current k-th exact distance.
  The answer is exact.

Cost model ("dim-ops" = multiply-adds): projection of q: m*d; residual norm: d;
LB per allowed point: m+1; exact distance: d. Sorting is counted separately.
"""
import numpy as np


class PCABound:
    def __init__(self, X, m, sample=20000, seed=0, residual_term=True):
        rng = np.random.default_rng(seed)
        S = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)].astype(np.float64)
        self.mu = S.mean(0)
        C = np.cov(S - self.mu, rowvar=False)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        self.V = evecs[:, order[:m]].astype(np.float32)          # (d, m)
        self.explained = float(evals[order[:m]].sum() / evals.sum())
        Xc = (X - self.mu).astype(np.float32)
        self.Y = Xc @ self.V                                      # (n, m)
        tot = np.einsum("ij,ij->i", Xc, Xc)
        self.r = np.sqrt(np.maximum(tot - np.einsum("ij,ij->i", self.Y, self.Y), 0)).astype(np.float32)
        self.X = X
        self.m, self.d = m, X.shape[1]
        self.residual_term = residual_term

    def search(self, q, ids, k, batch=16):
        """Exact k-NN of q among X[ids]. Returns (ids, sq. dists, stats dict)."""
        d, m = self.d, self.m
        qc = (q - self.mu).astype(np.float32)
        pq = qc @ self.V
        rq = np.sqrt(max(float(qc @ qc) - float(pq @ pq), 0.0))
        diff = self.Y[ids] - pq
        lb = np.einsum("ij,ij->i", diff, diff)
        if self.residual_term:
            lb = lb + (self.r[ids] - rq) ** 2
        order = np.argsort(lb, kind="stable")
        best_d = np.full(k, np.inf)
        best_i = np.full(k, -1, dtype=np.int64)
        refined = 0
        pos = 0
        n = len(ids)
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
        ops = m * d + d + n * (m + 1) + refined * d
        stats = dict(ops=ops, refined=refined, n=n, ops_bf=n * d)
        return best_i[ok], best_d[ok], stats
