"""Extract REAL attention keys and queries from RoBERTa-base (weights from spaCy
en_core_web_trf-3.8.0) over Python-stdlib docstrings, for selected (layer, head) pairs.
Database chunks -> keys (a long "KV cache"); held-out chunks -> queries.
Writes results/cache/data/attn_keys.npz.  Usage: python experiments/exp16_extract_attention.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fknn.roberta_np import BPE, Roberta, forward_collect, load, stdlib_corpus  # noqa: E402

LH = [(1, 0), (6, 3), (10, 7)]
N_DB_CHUNKS, N_Q_CHUNKS, CH = 200, 12, 512


def main():
    W, bpe = load("results/cache/data/trf/trf_model.bin")
    m, tok = Roberta(W), BPE(bpe)
    texts = stdlib_corpus(12_000_000)
    ids = []
    for t in texts:
        ids.extend(tok.encode(t)[1:-1])
        if len(ids) > (N_DB_CHUNKS + N_Q_CHUNKS) * (CH - 2):
            break
    ids = np.array(ids)
    chunks = [np.concatenate([[0], ids[i:i + CH - 2], [2]]) for i in range(0, len(ids) - CH, CH - 2)]
    print(len(texts), "docstrings,", len(ids), "tokens,", len(chunks), "chunks", flush=True)
    keys = {lh: [] for lh in LH}; queries = {lh: [] for lh in LH}
    t0 = time.time()
    for ci, ch in enumerate(chunks[:N_DB_CHUNKS + N_Q_CHUNKS]):
        out, _ = forward_collect(m, ch, LH)
        for lh, (q, k) in out.items():
            (keys if ci < N_DB_CHUNKS else queries)[lh].append(k if ci < N_DB_CHUNKS else q)
        if ci % 20 == 0:
            print(f"chunk {ci} {time.time() - t0:.0f}s", flush=True)
    save = {}
    for (l, h) in LH:
        save[f"K_l{l}_h{h}"] = np.concatenate(keys[(l, h)]).astype(np.float32)
        save[f"Q_l{l}_h{h}"] = np.concatenate(queries[(l, h)]).astype(np.float32)
    np.savez("results/cache/data/attn_keys.npz", **save)
    print({k: v.shape for k, v in save.items()})


if __name__ == "__main__":
    main()
