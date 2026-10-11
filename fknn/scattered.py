"""H7: exact k-NN under arbitrary (scattered) predicates - wrappers for fscan.c kernels.
Rows stay in original order; the filter is a row-id list / bitmap."""
import ctypes

import numpy as np

from .exact_va import VABound
from .fastscan import _lib, _p


class ScatteredIndex:
    def __init__(self, X, bits=6, seed=0):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        va = VABound(X, bits=bits, rotate=False, seed=seed)
        self.lo = np.ascontiguousarray(va.lo + va.mu[:, None], dtype=np.float32)
        self.hi = np.ascontiguousarray(va.hi + va.mu[:, None], dtype=np.float32)
        self.codesR = np.ascontiguousarray(va.codes)                        # [n][d]
        nb = (n + 63) // 64
        pad = np.zeros((nb * 64, d), np.uint8); pad[:n] = va.codes
        self.codesG = np.ascontiguousarray(pad.reshape(nb, 64, d).transpose(0, 2, 1))
        self.n, self.d, self.nb, self.bits = n, d, nb, bits
        self.lb_buf = np.empty(nb * 64, np.uint16); self.row_buf = np.empty(nb * 64, np.int64)

    def _ids(self, id_lists):
        ids = np.ascontiguousarray(np.concatenate(id_lists).astype(np.int64))
        off = np.concatenate([[0], np.cumsum([len(x) for x in id_lists])]).astype(np.int64)
        return ids, off

    def bf(self, Q, id_lists, k=10):
        Q = np.ascontiguousarray(Q, np.float32); ids, off = self._ids(id_lists); nq = len(Q)
        oid = np.empty((nq, k), np.int64); od = np.empty((nq, k), np.float32); t = np.empty(nq)
        _lib.bf_ids_batch(_p(self.X), ctypes.c_int(self.d), _p(Q), ctypes.c_int(nq), _p(ids), _p(off), ctypes.c_int(k), _p(oid), _p(od), _p(t))
        return oid, od, t

    def masked(self, Q, id_lists, k=10):
        Q = np.ascontiguousarray(Q, np.float32); nq = len(Q)
        bm = np.zeros((nq, self.nb * 64), bool)
        for i, ids in enumerate(id_lists):
            bm[i, ids] = True
        words = np.ascontiguousarray(np.packbits(bm, axis=1, bitorder="little").view(np.uint64))
        oid = np.empty((nq, k), np.int64); od = np.empty((nq, k), np.float32); t = np.empty(nq); nr = np.empty(nq, np.int64)
        _lib.va3m_batch(_p(self.X), ctypes.c_int64(self.n), ctypes.c_int(self.d), ctypes.c_int(self.bits), _p(self.codesG),
                        _p(self.lo), _p(self.hi), _p(Q), ctypes.c_int(nq), _p(words), ctypes.c_int(k),
                        _p(oid), _p(od), _p(t), _p(nr), _p(self.lb_buf), _p(self.row_buf))
        return oid, od, t, nr

    def gather(self, Q, id_lists, k=10):
        Q = np.ascontiguousarray(Q, np.float32); ids, off = self._ids(id_lists); nq = len(Q)
        oid = np.empty((nq, k), np.int64); od = np.empty((nq, k), np.float32); t = np.empty(nq); nr = np.empty(nq, np.int64)
        _lib.va3g_batch(_p(self.X), ctypes.c_int(self.d), ctypes.c_int(self.bits), _p(self.codesR), _p(self.lo), _p(self.hi),
                        _p(Q), ctypes.c_int(nq), _p(ids), _p(off), ctypes.c_int(k), _p(oid), _p(od), _p(t), _p(nr),
                        _p(self.lb_buf), _p(self.row_buf))
        return oid, od, t, nr
