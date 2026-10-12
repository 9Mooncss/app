"""Load the RoBERTa-base weights shipped inside spaCy's en_core_web_trf-3.8.0 (no PyTorch needed)
and run a numpy forward pass to collect REAL attention keys/queries per layer and head.

The model file is thinc msgpack; shim 6 holds a torch.save zip (data.pkl + raw storages), which
we read with a minimal unpickler. Tokenizer: byte-level BPE (vocab + merges from the same file).
Used only as a source of realistic attention vectors for the MIPS experiments.
"""
import io
import json
import pickle
import re
import zipfile

import msgpack
import numpy as np

_DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "LongStorage": np.int64,
       "IntStorage": np.int32, "BFloat16Storage": np.uint16, "BoolStorage": np.bool_}


def _load_torch_zip(raw: bytes):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    prefix = names[0].split("/")[0]

    class _Storage:
        def __init__(self, key, dtype):
            self.arr = np.frombuffer(zf.read(f"{prefix}/data/{key}"), dtype=dtype)

    def _rebuild(storage, offset, size, stride, *args):
        a = storage.arr
        if not size:
            return a[offset:offset + 1].reshape(())
        itemsize = a.itemsize
        return np.lib.stride_tricks.as_strided(a[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return _rebuild
            if name.endswith("Storage"):
                return name
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if mod == "torch" and name in ("float32", "float16"):
                return name
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            stype = stype if isinstance(stype, str) else getattr(stype, "__name__", str(stype))
            return _Storage(key, _DT.get(stype, np.float32))

    return _U(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()


def load(path):
    obj = msgpack.unpackb(open(path, "rb").read(), raw=False, strict_map_key=False)
    inner = msgpack.unpackb(obj["shims"][6][0], raw=False, strict_map_key=False)
    state = _load_torch_zip(inner["state"])
    bpe = msgpack.unpackb(obj["attrs"][3]["byte_bpe_processor"], raw=False, strict_map_key=False)
    return {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}, bpe


# ---------------- byte-level BPE (GPT-2/RoBERTa style) ----------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class BPE:
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

    def __init__(self, bpe):
        vocab = bpe["vocab"]
        self.vocab = vocab if isinstance(vocab, dict) else json.loads(vocab)
        merges = bpe["merges"]
        merges = merges.split("\n") if isinstance(merges, str) else merges
        pairs = [tuple(m.split()) if isinstance(m, str) else tuple(m) for m in merges if m and not str(m).startswith("#")]
        self.ranks = {p: i for i, p in enumerate(pairs) if len(p) == 2}
        self.b2u = _bytes_to_unicode()

    def _bpe(self, tok):
        word = list(tok)
        while len(word) > 1:
            best = min(((self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(word, word[1:]))))
            if best[0] == 1 << 30:
                break
            i = best[1]
            word = word[:i] + [word[i] + word[i + 1]] + word[i + 2:]
        return word

    def encode(self, text):
        ids = [self.vocab.get("<s>", 0)]
        for t in self.PAT.findall(text):
            u = "".join(self.b2u[b] for b in t.encode("utf-8"))
            ids += [self.vocab.get(p, self.vocab.get("<unk>", 3)) for p in self._bpe(u)]
        return ids + [self.vocab.get("</s>", 2)]


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu_unused(x):
    from math import sqrt
    import numpy as _np
    return 0.5 * x * (1 + _np.vectorize(lambda z: __import__("math").erf(z / sqrt(2)))(x)) if x.size < 0 else \
        0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


class Roberta:
    def __init__(self, W):
        self.W = W
        keys = list(W)
        self.layer_keys = sorted({int(m.group(1)) for k in keys for m in [re.search(r"layers\.(\d+)\.", k)] if m})
        self.keys = keys

    def find(self, *parts):
        c = [k for k in self.keys if all(p in k for p in parts)]
        if len(c) != 1:
            raise KeyError((parts, c[:5]))
        return self.W[c[0]]


def forward_collect(model, ids, layers_heads):
    """Run RoBERTa on one sequence of token ids; return {(layer, head): (Q, K)} (L x 64 each)
    and the final hidden states. Post-LN encoder, fused qkv projection [q; k; v]."""
    W = model.W
    pre = "curated_encoder."
    E = W[pre + "embeddings.inner.word_embeddings.weight"]
    P = W[pre + "embeddings.inner.position_embeddings.weight"]
    T = W[pre + "embeddings.inner.token_type_embeddings.weight"]
    L = len(ids)
    h = E[ids] + P[np.arange(2, 2 + L)] + T[0]
    h = _ln(h, W[pre + "embeddings.inner.layer_norm.weight"], W[pre + "embeddings.inner.layer_norm.bias"])
    out = {}
    nh, hd = 12, 64
    for li in model.layer_keys:
        p = f"{pre}layers.{li}."
        qkv = h @ W[p + "mha.input.weight"].T + W[p + "mha.input.bias"]
        q, k, v = np.split(qkv, 3, axis=1)
        q = q.reshape(L, nh, hd); k = k.reshape(L, nh, hd); v = v.reshape(L, nh, hd)
        for (lw, hw) in layers_heads:
            if lw == li:
                out[(lw, hw)] = (q[:, hw].copy(), k[:, hw].copy())
        s = np.matmul(q.transpose(1, 0, 2), k.transpose(1, 2, 0)) / np.sqrt(hd)
        s -= s.max(-1, keepdims=True)
        a = np.exp(s); a /= a.sum(-1, keepdims=True)
        o = np.matmul(a, v.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(L, nh * hd)
        o = o @ W[p + "mha.output.weight"].T + W[p + "mha.output.bias"]
        h = _ln(h + o, W[p + "attn_output_layernorm.weight"], W[p + "attn_output_layernorm.bias"])
        f = _gelu(h @ W[p + "ffn.intermediate.weight"].T + W[p + "ffn.intermediate.bias"])
        f = f @ W[p + "ffn.output.weight"].T + W[p + "ffn.output.bias"]
        h = _ln(h + f, W[p + "ffn_output_layernorm.weight"], W[p + "ffn_output_layernorm.bias"])
    return out, h


def stdlib_corpus(max_chars=8_000_000):
    """Real English prose available offline: docstrings of the Python standard library."""
    import ast
    import glob
    import os
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    texts, total = [], 0
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if "/test" in path or "site-packages" in path:
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                ds = ast.get_docstring(node)
                if ds and len(ds) > 200:
                    texts.append(" ".join(ds.split()))
                    total += len(texts[-1])
        if total > max_chars:
            break
    return texts


def _gelu(x):
    """tanh approximation of GELU (max abs error ~1e-3 vs erf GELU)."""
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
