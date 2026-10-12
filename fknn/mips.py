"""Exact top-k maximum inner product search (MIPS): brute force vs VA upper-bound fast scan.
Cells per dimension are equi-depth; the outer cells end at the data min/max so every cell is
finite (needed for an inner-product upper bound)."""
import ctypes

import numpy as np

from .fastscan import _lib, _p


class MipsIndex:
    def __init__(self, X, bits=6, sample=50000, seed=0, norm_order=False):
        """norm_order=True stores rows sorted by decreasing norm (enables Cauchy-Schwarz stop);
        returned ids are always original row ids."""
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1)
        self.order = np.argsort(-norms, kind="stable") if norm_order else np.arange(len(X))
        X = X[self.order]
        self.norms = np.ascontiguousarray(norms[self.order], dtype=np.float32)
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        rng = np.random.default_rng(seed)
        S = X[rng.choice(n, size=min(sample, n), replace=False)]
        C = 2 ** bits
        inner = np.quantile(S, np.linspace(0, 1, C + 1)[1:-1], axis=0).T.astype(np.float32)   # (d, C-1)
        mn, mx = X.min(0).astype(np.float32), X.max(0).astype(np.float32)
        self.lo = np.ascontiguousarray(np.concatenate([mn[:, None], inner], 1))
        self.hi = np.ascontiguousarray(np.concatenate([inner, mx[:, None]], 1))
        codes = np.empty((n, d), np.uint8)
        for i in range(d):
            codes[:, i] = np.searchsorted(inner[i], X[:, i], side="right")
        nb = (n + 63) // 64
        pad = np.zeros((nb * 64, d), np.uint8); pad[:n] = codes
        self.codes = np.ascontiguousarray(pad.reshape(nb, 64, d).transpose(0, 2, 1))
        self.n, self.d, self.bits = n, d, bits
        self.ub_buf = np.empty(nb * 64, np.uint16); self.row_buf = np.empty(nb * 64, np.int64)
        self.bnorm = np.ascontiguousarray(np.pad(self.norms, (0, nb * 64 - n)).reshape(nb, 64).max(1), dtype=np.float32)

    def bf(self, Q, k=10):
        Q = np.ascontiguousarray(Q, np.float32); nq = len(Q)
        oid = np.empty((nq, k), np.int64); os_ = np.empty((nq, k), np.float32); t = np.empty(nq)
        _lib.mips_bf_batch(_p(self.X), ctypes.c_int64(self.n), ctypes.c_int(self.d), _p(Q), ctypes.c_int(nq), ctypes.c_int(k), _p(oid), _p(os_), _p(t))
        return self.order[np.maximum(oid, 0)], os_, t

    def va(self, Q, k=10):
        Q = np.ascontiguousarray(Q, np.float32); nq = len(Q)
        oid = np.empty((nq, k), np.int64); os_ = np.empty((nq, k), np.float32); t = np.empty(nq); nr = np.empty(nq, np.int64)
        _lib.mips_va_batch(_p(self.X), ctypes.c_int64(self.n), ctypes.c_int(self.d), ctypes.c_int(self.bits), _p(self.codes),
                           _p(self.lo), _p(self.hi), _p(Q), ctypes.c_int(nq), ctypes.c_int(k), _p(oid), _p(os_), _p(t), _p(nr),
                           _p(self.ub_buf), _p(self.row_buf))
        return self.order[np.maximum(oid, 0)], os_, t, nr

    def bf_norm(self, Q, k=10):
        """brute force over norm-sorted rows with the Cauchy-Schwarz stop (needs norm_order=True)."""
        Q = np.ascontiguousarray(Q, np.float32); nq = len(Q)
        oid = np.empty((nq, k), np.int64); os_ = np.empty((nq, k), np.float32); t = np.empty(nq); ns = np.empty(nq, np.int64)
        _lib.mips_bfn_batch(_p(self.X), ctypes.c_int64(self.n), ctypes.c_int(self.d), _p(self.norms), _p(Q), ctypes.c_int(nq),
                            ctypes.c_int(k), _p(oid), _p(os_), _p(t), _p(ns))
        return self.order[np.maximum(oid, 0)], os_, t, ns

    def va_norm(self, Q, k=10):
        """VA bounds + streaming refinement + Cauchy-Schwarz block stop (needs norm_order=True)."""
        Q = np.ascontiguousarray(Q, np.float32); nq = len(Q)
        oid = np.empty((nq, k), np.int64); os_ = np.empty((nq, k), np.float32); t = np.empty(nq)
        nr = np.empty(nq, np.int64); nb = np.empty(nq, np.int64)
        _lib.mips_van_batch(_p(self.X), ctypes.c_int64(self.n), ctypes.c_int(self.d), ctypes.c_int(self.bits), _p(self.codes),
                            _p(self.lo), _p(self.hi), _p(self.bnorm), _p(Q), ctypes.c_int(nq), ctypes.c_int(k),
                            _p(oid), _p(os_), _p(t), _p(nr), _p(nb))
        return self.order[np.maximum(oid, 0)], os_, t, nr, nb
