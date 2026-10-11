"""ctypes wrapper around fknn/c/fscan.c: exact filtered k-NN, brute force vs VA fast-scan.

Build: gcc -O3 -march=native -fPIC -shared fknn/c/fscan.c -o fknn/c/libfscan.so -lm
(done automatically on first import if the .so is missing or older than the .c).
"""
import ctypes
import os
import subprocess

import numpy as np

from .exact_va import VABound

_DIR = os.path.join(os.path.dirname(__file__), "c")
_SRC, _LIB = os.path.join(_DIR, "fscan.c"), os.path.join(_DIR, "libfscan.so")
if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-fPIC", "-shared", _SRC, "-o", _LIB, "-lm"])
_lib = ctypes.CDLL(_LIB)
_P = ctypes.c_void_p


def _p(a):
    return a.ctypes.data_as(_P)


class FastScanIndex:
    def __init__(self, X, labels, n_labels, bits=4, seed=0):
        assert bits == 4
        order = np.argsort(labels, kind="stable")
        self.row_to_id = order.astype(np.int64)
        self.Xs = np.ascontiguousarray(X[order], dtype=np.float32)
        counts = np.bincount(labels, minlength=n_labels)
        self.lstart = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        va = VABound(X, bits=4, rotate=False, seed=seed)
        # VABound works in centered coordinates (x - mu); shift cell edges back to raw space
        lo = va.lo + va.mu[:, None]
        hi = va.hi + va.mu[:, None]               # (d, 16)
        self.edges_lo = np.ascontiguousarray(lo, dtype=np.float32)
        self.edges_hi = np.ascontiguousarray(hi, dtype=np.float32)
        codes_sorted = va.codes[order]            # (n, d), rows in label-sorted order
        nblk = (counts + 31) // 32
        self.lblock = np.concatenate([[0], np.cumsum(nblk)]).astype(np.int64)
        nb = int(self.lblock[-1])
        d = X.shape[1]
        self.slot_row = np.full(nb * 32, -1, dtype=np.int64)
        codes = np.zeros((nb, 32, d), dtype=np.uint8)
        for l in range(n_labels):
            a, b = self.lstart[l], self.lstart[l + 1]
            if b == a:
                continue
            s0 = self.lblock[l] * 32
            self.slot_row[s0:s0 + (b - a)] = np.arange(a, b)
            flat = codes.reshape(nb * 32, d)
            flat[s0:s0 + (b - a)] = codes_sorted[a:b]
        self.codes = np.ascontiguousarray(codes.transpose(0, 2, 1))   # (nb, d, 32)
        self.d, self.nslots = d, nb * 32
        self.lb_buf = np.empty(self.nslots, dtype=np.uint16)
        self.slot_buf = np.empty(self.nslots, dtype=np.int64)

    @staticmethod
    def _qarrays(Q, label_sets):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        qlab = np.ascontiguousarray(np.concatenate(label_sets).astype(np.int64))
        qoff = np.concatenate([[0], np.cumsum([len(p) for p in label_sets])]).astype(np.int64)
        return Q, qlab, qoff

    def bf(self, Q, label_sets, k=10):
        Q, qlab, qoff = self._qarrays(Q, label_sets)
        nq = len(Q)
        oid = np.empty((nq, k), np.int64); od = np.empty((nq, k), np.float32)
        t = np.empty(nq); n = np.empty(nq, np.int64)
        _lib.bf_batch(_p(self.Xs), ctypes.c_int(self.d), _p(self.lstart), _p(Q), ctypes.c_int(nq),
                      _p(qlab), _p(qoff), ctypes.c_int(k), _p(oid), _p(od), _p(t), _p(n))
        return self.row_to_id[np.maximum(oid, 0)], od, t, n

    def va(self, Q, label_sets, k=10):
        Q, qlab, qoff = self._qarrays(Q, label_sets)
        nq = len(Q)
        oid = np.empty((nq, k), np.int64); od = np.empty((nq, k), np.float32)
        t = np.empty(nq); nr = np.empty(nq, np.int64); ns = np.empty(nq, np.int64)
        _lib.va_batch(_p(self.Xs), ctypes.c_int(self.d), _p(self.codes), _p(self.slot_row), _p(self.lblock),
                      _p(self.edges_lo), _p(self.edges_hi), _p(Q), ctypes.c_int(nq), _p(qlab), _p(qoff),
                      ctypes.c_int(k), _p(oid), _p(od), _p(t), _p(nr), _p(ns), _p(self.lb_buf), _p(self.slot_buf))
        return self.row_to_id[np.maximum(oid, 0)], od, t, nr, ns
