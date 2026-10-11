import numpy as np
import pytest

from fknn.bruteforce import LabelIndex, bruteforce_knn
from fknn.data import MODES, make_dataset, make_queries
from fknn.hnsw_build import build
from fknn.hnsw_replica import search


@pytest.fixture(scope="module")
def small():
    ds = make_dataset(n_clusters=10, per_cluster=500, dim=16, labels_per_cluster=10, seed=1)
    return ds, LabelIndex(ds.labels, ds.n_labels)


def naive(ds, q, P, k):
    mask = np.isin(ds.labels, P)
    ids = np.nonzero(mask)[0]
    d2 = ((ds.X[ids].astype(np.float64) - q) ** 2).sum(1)
    o = np.lexsort((ids, d2))[:k]
    return ids[o]


def test_bruteforce_matches_naive(small):
    ds, li = small
    for mode in MODES:
        for s in (0.01, 0.1, 0.5):
            for q, P in make_queries(ds, 20, mode, s, seed=7):
                ids, _, nd = bruteforce_knn(ds.X, q, li.ids(P), 10)
                assert nd == np.isin(ds.labels, P).sum()
                assert set(ids) == set(naive(ds, q, P, 10))


def test_selectivity_equal_across_modes_and_correlation_ordered(small):
    ds, li = small
    for s in (0.01, 0.1, 0.3):
        sizes, mean_d = {}, {}
        for mode in MODES:
            qs = make_queries(ds, 30, mode, s, seed=3)
            sizes[mode] = {len(li.ids(P)) for _, P in qs}
            mean_d[mode] = np.mean([((ds.X[li.ids(P)] - q) ** 2).sum(1).mean() for q, P in qs])
        # same number of labels -> same expected size (exact per-label counts vary)
        all_sizes = np.concatenate([list(v) for v in sizes.values()])
        assert abs(all_sizes.mean() / len(ds.X) - s) < 0.35 * s
        assert mean_d["positive"] < mean_d["neutral"] < mean_d["negative"]


@pytest.fixture(scope="module")
def hnsw(small, tmp_path_factory):
    ds, _ = small
    idx, g, Xint = build(ds.X, str(tmp_path_factory.mktemp("h") / "i.bin"), M=8, ef_construction=100)
    return idx, g, Xint


def test_loaded_graph_matches_vectors(small, hnsw):
    ds, _ = small
    idx, g, Xint = hnsw
    assert len(g.ext_label) == len(ds.X)
    assert sorted(g.ext_label.tolist()) == list(range(len(ds.X)))
    np.testing.assert_array_equal(np.asarray(idx.get_items(g.ext_label[:50])), Xint[:50])


@pytest.mark.parametrize("filtered", [False, True])
def test_replica_matches_hnswlib(small, hnsw, filtered):
    ds, li = small
    idx, g, Xint = hnsw
    idx.set_ef(32)
    idx.set_num_threads(1)
    agree, total = 0, 0
    for q, P in make_queries(ds, 60, "neutral", 0.1, seed=5):
        allowed_ext = np.isin(ds.labels, P)
        if filtered:
            ref, _ = idx.knn_query(q[None], k=10, filter=lambda i: bool(allowed_ext[i]))
            got, _, _ = search(g, Xint, q, 10, 32, allowed_ext[g.ext_label])
        else:
            ref, _ = idx.knn_query(q[None], k=10)
            got, _, _ = search(g, Xint, q, 10, 32, None)
        agree += len(set(ref[0].tolist()) & set(got.tolist()))
        total += 10
    # float32 rounding in SIMD vs numpy can flip rare ties; demand near-identity
    assert agree / total >= 0.99


def test_mixed_mode_between_extremes(small):
    ds, li = small
    md = {m: np.mean([((ds.X[li.ids(P)] - q) ** 2).sum(1).mean()
                      for q, P in make_queries(ds, 40, m, 0.1, seed=4)])
          for m in ("positive", "mixed", "negative")}
    assert md["positive"] < md["mixed"] < md["negative"]


def test_local_selectivity_counts(small):
    from fknn.planner import LocalSelectivity
    ds, _ = small
    est = LocalSelectivity(ds.X, ds.labels, ds.n_labels, n_centroids=8, seed=0, ms=(1, 8))
    assign = est._assign(ds.X, est.C)
    q, P = make_queries(ds, 1, "positive", 0.05, seed=2)[0]
    f1, f8 = est.features(q, P)
    near = np.argmin(((est.C - q) ** 2).sum(1))
    in_c = assign == near
    expect = np.isin(ds.labels[in_c], P).mean()
    assert abs(f1 - np.log10(expect + 1e-4)) < 1e-9
    assert abs(f8 - np.log10(np.isin(ds.labels, P).mean() + 1e-4)) < 1e-9  # all centroids = global s


def test_fallback_derivation():
    from fknn.planner import strategy_table
    rows = [dict(n_allowed=100, cost_BF=100, cost_HNSW16=50, recall_HNSW16=0.5),
            dict(n_allowed=100, cost_BF=100, cost_HNSW16=500, recall_HNSW16=1.0)]
    names, c, r = strategy_table(rows, [16], True)
    j = names.index("HNSW16fb1")
    assert c[0, j] == 50 and r[0, j] == 0.5      # within budget: unchanged (silent failure kept)
    assert c[1, j] == 200 and r[1, j] == 1.0     # overrun: budget + BF, exact


def test_planner_learns_obvious_rule():
    from fknn.planner import KNNPlanner
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1, size=(300, 1))
    cost = np.stack([np.where(F[:, 0] < 0.5, 1.0, 10.0), np.where(F[:, 0] < 0.5, 10.0, 1.0)], 1)
    rec = np.ones_like(cost)
    pl = KNNPlanner().fit(F, cost, rec)
    pick = pl.predict(np.array([[0.1], [0.9]]))
    assert list(pick) == [0, 1]


def test_ivf_exact_matches_bruteforce(small):
    from fknn.ivf_exact import IVFExact
    from fknn.planner import LocalSelectivity
    ds, li = small
    est = LocalSelectivity(ds.X, ds.labels, ds.n_labels, n_centroids=16, seed=0)
    ivf = IVFExact(ds.X, ds.labels, ds.n_labels, est.C, est._assign(ds.X, est.C))
    for mode in MODES:
        for s in (0.01, 0.1, 0.5):
            for q, P in make_queries(ds, 15, mode, s, seed=9):
                ids, d2, nd, _ = ivf.search(q, P, 10)
                ref, rd2, nbf = bruteforce_knn(ds.X, q, li.ids(P), 10)
                np.testing.assert_allclose(np.sort(d2), np.sort(rd2), rtol=1e-4)
                assert nd <= nbf + 16


@pytest.mark.parametrize("residual", [False, True])
def test_pca_bound_is_exact_and_lower_bound(small, residual):
    from fknn.exact_lb import PCABound
    ds, li = small
    pb = PCABound(ds.X, m=4, residual_term=residual)
    for mode in MODES:
        for s in (0.01, 0.1, 0.5):
            for q, P in make_queries(ds, 10, mode, s, seed=12):
                ids = li.ids(P)
                got, gd, st = pb.search(q, ids, 10)
                ref, rd, _ = bruteforce_knn(ds.X, q, ids, 10)
                np.testing.assert_allclose(np.sort(gd), np.sort(rd), rtol=1e-4)
                assert st["refined"] <= len(ids)
    # the bound itself never exceeds the true distance
    q, P = make_queries(ds, 1, "neutral", 0.5, seed=1)[0]
    qc = q - pb.mu
    pq = qc @ pb.V
    rq = np.sqrt(max(qc @ qc - pq @ pq, 0))
    lb = ((pb.Y - pq) ** 2).sum(1) + ((pb.r - rq) ** 2 if residual else 0)
    true = ((ds.X - q) ** 2).sum(1)
    assert np.all(lb <= true * (1 + 1e-4) + 1e-4)


@pytest.mark.parametrize("rotate,int8", [(False, False), (True, False), (True, True)])
def test_va_bound_exact(small, rotate, int8):
    from fknn.exact_va import VABound
    ds, li = small
    va = VABound(ds.X, bits=3, rotate=rotate)
    for mode in MODES:
        for s in (0.01, 0.2):
            for q, P in make_queries(ds, 8, mode, s, seed=13):
                ids = li.ids(P)
                _, gd, _ = va.search(q, ids, 10, int8_table=int8)
                _, rd, _ = bruteforce_knn(ds.X, q, ids, 10)
                np.testing.assert_allclose(np.sort(gd), np.sort(rd), rtol=1e-4)
    q = make_queries(ds, 1, "neutral", 0.5, seed=1)[0][0]
    T = va.table(q)
    lb = T[np.arange(va.d)[None, :], va.codes].sum(1)
    assert np.all(lb <= ((ds.X - q) ** 2).sum(1) * (1 + 1e-4) + 1e-4)


def test_fastscan_c_exact(small):
    from fknn.fastscan import FastScanIndex
    ds, li = small
    fs = FastScanIndex(ds.X, ds.labels, ds.n_labels)
    for mode in MODES:
        for s in (0.01, 0.1, 0.5):
            qs = make_queries(ds, 10, mode, s, seed=17)
            Q = np.stack([q for q, _ in qs]); Ps = [P for _, P in qs]
            _, bd, _, bn = fs.bf(Q, Ps)
            _, vd, _, nr, ns = fs.va(Q, Ps)
            for i, (q, P) in enumerate(qs):
                _, rd, nbf = bruteforce_knn(ds.X, q, li.ids(P), 10)
                np.testing.assert_allclose(bd[i], np.sort(rd), rtol=1e-4)
                np.testing.assert_allclose(vd[i], np.sort(rd), rtol=1e-4)
                assert bn[i] == nbf and ns[i] == nbf and nr[i] <= nbf


def test_pcava_exact(small):
    from fknn.exact_pcava import PCAVABound
    ds, li = small
    pv = PCAVABound(ds.X, m=6)
    for mode in MODES:
        for s in (0.01, 0.2):
            for q, P in make_queries(ds, 8, mode, s, seed=14):
                ids = li.ids(P)
                _, gd, _ = pv.search(q, ids, 10)
                _, rd, _ = bruteforce_knn(ds.X, q, ids, 10)
                np.testing.assert_allclose(np.sort(gd), np.sort(rd), rtol=1e-4)
