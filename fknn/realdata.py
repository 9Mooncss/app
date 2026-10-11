"""Real datasets with natural attributes, plus random filter workloads.

Both are public ANN-benchmark datasets. Downloads are cached in results/cache/data.

fashion: Fashion-MNIST, 60k base images (784-d pixels / 255), 10k test images as
         queries. Attribute = (product class, price bucket). Class is real and
         strongly tied to geometry; price (100 buckets) is synthetic and independent,
         as a stand-in for a typical numeric product field.
         label = class * 100 + price_bucket  (1000 labels).
         Filter = random set of 1..5 classes  x  random price window.
glove:   GloVe 6B 100-d word vectors (first 200k words, L2-normalised, so L2
         order == cosine order). Every 100th word is held out as a query.
         Attribute = frequency rank bucket (rank // 200), a real property:
         the file is sorted by corpus frequency. Filter = random window of buckets
         ("words with frequency rank in [a, b)"), i.e. a range filter.
spacy:   spaCy en_core_web_lg word vectors (342,918 x 300, L2-normalised), 2,000 held out as
         queries. No natural attribute is available, so labels are uniform random in 0..999
         (neutral correlation). Used for the out-of-cache (memory-bound) regime: 411 MB.
"""
import gzip
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "cache", "data")
FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
GLOVE = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
         "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz")


@dataclass
class RealDataset:
    name: str
    X: np.ndarray       # (n, d) float32 base vectors
    labels: np.ndarray  # (n,) int32
    n_labels: int
    Q: np.ndarray       # (m, d) float32 query vectors (not in base)
    Qlabels: np.ndarray  # (m,) natural label of each query (informational)


def _get(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _idx(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def load_fashion(seed=0) -> RealDataset:
    xb = _idx(_get(FASHION + "train-images-idx3-ubyte.gz", "fm_train_x.gz"), 16).reshape(-1, 784)
    yb = _idx(_get(FASHION + "train-labels-idx1-ubyte.gz", "fm_train_y.gz"), 8)
    xq = _idx(_get(FASHION + "t10k-images-idx3-ubyte.gz", "fm_test_x.gz"), 16).reshape(-1, 784)
    yq = _idx(_get(FASHION + "t10k-labels-idx1-ubyte.gz", "fm_test_y.gz"), 8)
    rng = np.random.default_rng(seed)
    price = rng.integers(0, 100, size=len(yb))
    labels = (yb.astype(np.int32) * 100 + price).astype(np.int32)
    return RealDataset("fashion", (xb / 255.0).astype(np.float32), labels, 1000,
                       (xq / 255.0).astype(np.float32), yq.astype(np.int32))


def load_glove(n_words=200_000) -> RealDataset:
    path = _get(GLOVE, "glove100.gz")
    vecs = np.empty((n_words, 100), dtype=np.float32)
    with gzip.open(path, "rt", encoding="utf8") as f:
        first = f.readline().split()
        i = 0
        if len(first) != 2:  # no "count dim" header
            vecs[0] = np.asarray(first[1:], dtype=np.float32)
            i = 1
        for line in f:
            if i >= n_words:
                break
            vecs[i] = np.asarray(line.rstrip().split(" ")[1:], dtype=np.float32)
            i += 1
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    rank = np.arange(n_words)
    bucket = (rank // 200).astype(np.int32)
    is_q = rank % 100 == 0
    return RealDataset("glove", vecs[~is_q], bucket[~is_q], n_words // 200, vecs[is_q], bucket[is_q])


def load_spacy(seed=0) -> RealDataset:
    path = os.path.join(CACHE, "spacy_lg_vectors.npy")
    if not os.path.exists(path):
        raise FileNotFoundError("extract vocab/vectors from the en_core_web_lg-3.8.0 wheel into " + path)
    V = np.load(path).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(seed)
    is_q = np.zeros(len(V), bool)
    is_q[rng.choice(len(V), 2000, replace=False)] = True
    labels = rng.integers(0, 1000, size=len(V)).astype(np.int32)
    return RealDataset("spacy", V[~is_q], labels[~is_q], 1000, V[is_q], labels[is_q])


def make_real_queries(ds: RealDataset, n_queries: int, seed: int):
    """Random filters; correlation with the query is whatever the data implies."""
    rng = np.random.default_rng(seed)
    qi = rng.choice(len(ds.Q), size=n_queries, replace=False)
    out = []
    for i in qi:
        if ds.name == "fashion":
            classes = rng.choice(10, size=rng.integers(1, 6), replace=False)
            w = int(np.clip(round(10 ** rng.uniform(0, 2)), 1, 100))
            a = rng.integers(0, 100 - w + 1)
            P = (classes[:, None] * 100 + np.arange(a, a + w)[None, :]).ravel()
        else:  # glove (frequency-rank range) and spacy (random labels): a window of labels
            L = ds.n_labels
            w = int(np.clip(round(10 ** rng.uniform(0, np.log10(L / 2))), 1, L))
            a = rng.integers(0, L - w + 1)
            P = np.arange(a, a + w)
        out.append((ds.Q[i], np.sort(P.astype(np.int32))))
    return out
