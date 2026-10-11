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
