"""Strategy planners for filtered k-NN.

Every planner is the SAME learner (k-nearest-neighbour regression over past
queries); planners differ only in the features they see and the strategies they
may choose. This isolates the value of a feature from the value of a learner.

Strategies:
  BF               pre-filter + brute force (exact)
  HNSW{ef}         hnswlib-style filtered HNSW search
  HNSW{ef}fb{b}    HNSW with a distance budget b * n_allowed; on overrun, abort
                   and run BF (cost = budget + n_allowed, exact answer).
                   Derived offline from the full HNSW run: a run that costs c <= budget
                   is unaffected; otherwise it would have been aborted at the budget.
"""
import numpy as np

FALLBACK_BETAS = (0.25, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------- local estimator
class LocalSelectivity:
    """k-means centroids + per-centroid label histogram.

    For a query q and allowed label set P, returns the fraction of points that pass
    the filter among the points of the m nearest centroids, for several m.
    Query-time cost: n_centroids distance computations + O(m * |P|) table lookups.
    """

    def __init__(self, X, labels, n_labels, n_centroids=64, iters=15, sample=20000, seed=0,
                 ms=(1, 4, 16)):
        rng = np.random.default_rng(seed)
        S = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)]
        C = S[rng.choice(len(S), size=n_centroids, replace=False)].copy()
        for _ in range(iters):
            a = self._assign(S, C)
            for j in range(n_centroids):
                pts = S[a == j]
                if len(pts):
                    C[j] = pts.mean(0)
        self.C = C
        assign = self._assign(X, C)
        self.H = np.zeros((n_centroids, n_labels), dtype=np.int64)
        np.add.at(self.H, (assign, labels), 1)
        self.size = self.H.sum(1)
        self.ms = ms
        self.cost = n_centroids

    @staticmethod
    def _assign(X, C, chunk=20000):
        out = np.empty(len(X), dtype=np.int64)
        cc = (C * C).sum(1)
        for i in range(0, len(X), chunk):
            B = X[i:i + chunk]
            out[i:i + chunk] = np.argmin(cc[None, :] - 2 * B @ C.T, axis=1)
        return out

    def features(self, q, P):
        d = ((self.C - q) ** 2).sum(1)
        order = np.argsort(d)
        allowed = self.H[:, P].sum(1)
        f = []
        for m in self.ms:
            top = order[:m]
            f.append(np.log10(allowed[top].sum() / max(self.size[top].sum(), 1) + 1e-4))
        return f


# ---------------------------------------------------------------- strategy table
def strategy_table(rows, efs, with_fallback, extra=()):
    """Return names, cost matrix (n, S), recall matrix (n, S).
    `extra`: additional strategy names with cost_<name>/recall_<name> columns."""
    names = ["BF"]
    cost = [np.array([r["cost_BF"] for r in rows], dtype=float)]
    rec = [np.ones(len(rows))]
    nb = np.array([r["n_allowed"] for r in rows], dtype=float)
    for ef in efs:
        c = np.array([r[f"cost_HNSW{ef}"] for r in rows], dtype=float)
        rr = np.array([r[f"recall_HNSW{ef}"] for r in rows])
        names.append(f"HNSW{ef}")
        cost.append(c)
        rec.append(rr)
        if with_fallback:
            for b in FALLBACK_BETAS:
                budget = b * nb
                over = c > budget
                names.append(f"HNSW{ef}fb{b:g}")
                cost.append(np.where(over, budget + nb, c))
                rec.append(np.where(over, 1.0, rr))
    for name in extra:
        names.append(name)
        cost.append(np.array([r[f"cost_{name}"] for r in rows], dtype=float))
        rec.append(np.array([r[f"recall_{name}"] for r in rows]))
    return names, np.stack(cost, 1), np.stack(rec, 1)


# ---------------------------------------------------------------- learner
class KNNPlanner:
    """Predict mean cost / mean recall of every strategy from the k most similar
    past queries (in standardized feature space); pick the cheapest strategy whose
    predicted recall >= recall_target. Column 0 (BF, exact) is always allowed.
    k is chosen by leave-one-out on the training queries."""

    def __init__(self, recall_target=0.95, ks=(5, 15, 50, 150), seed=0):
        self.rt, self.ks, self.seed = recall_target, ks, seed

    def fit(self, F, cost, rec):
        perm = np.random.default_rng(self.seed).permutation(len(F))  # break ties fairly
        self.F, self.cost, self.rec = F[perm], cost[perm], rec[perm]
        self.mu, self.sd = self.F.mean(0), self.F.std(0) + 1e-9
        Z = (self.F - self.mu) / self.sd
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        order = np.argsort(D, axis=1, kind="stable")
        best = None
        for k in self.ks:
            if k >= len(F):
                continue
            pick = self._choose(order[:, :k])
            tot = self.cost[np.arange(len(F)), pick].sum()
            if best is None or tot < best[0]:
                best = (tot, k)
        self.k = best[1]
        return self

    def _choose(self, nbr_idx):
        pc = self.cost[nbr_idx].mean(1)   # (n, S)
        pr = self.rec[nbr_idx].mean(1)
        pc = np.where(pr >= self.rt, pc, np.inf)
        return np.argmin(pc, axis=1)

    def predict(self, F):
        Z = (F - self.mu) / self.sd
        Zt = (self.F - self.mu) / self.sd
        D = ((Z[:, None, :] - Zt[None, :, :]) ** 2).sum(-1)
        return self._choose(np.argsort(D, axis=1, kind="stable")[:, :self.k])


def evaluate(pick, cost, rec, overhead=0.0, recall_ok=0.9):
    idx = np.arange(len(pick))
    c = cost[idx, pick] + overhead
    r = rec[idx, pick]
    return dict(total_cost=float(c.sum()), mean_cost=float(c.mean()), mean_recall=float(r.mean()),
                p10_recall=float(np.percentile(r, 10)), frac_recall_below_ok=float((r < recall_ok).mean()))


def oracle_pick(cost, rec, recall_ok=0.9):
    c = np.where(rec >= recall_ok, cost, np.inf)
    return np.argmin(c, axis=1)
