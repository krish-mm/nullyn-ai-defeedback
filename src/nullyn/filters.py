"""Audio-path filters: RBJ peaking EQ and the dynamic IIR notch bank.

All audio-path filtering is causal IIR (no look-ahead). Algorithmic latency
of the notch bank itself is 0 samples; coefficient updates take effect on
subsequent samples.
"""

from __future__ import annotations

import numpy as np
from .constants import RESEARCH_FS
from ._accel import sosfilt, warmup as _accel_warmup


def peaking_eq(f0, gain_db, q, fs=RESEARCH_FS):
    """RBJ cookbook peaking EQ biquad, returned as normalised (b, a)."""
    out = np.empty((1, 6))
    peaking_eq_rows(np.asarray([f0], dtype=float), np.asarray([gain_db], dtype=float),
                    q, fs, out)
    return out[0, :3].copy(), out[0, 3:].copy()


def peaking_eq_rows(f0, gain_db, q, fs, out):
    """Vectorised RBJ peaking EQ, writing normalised SOS rows into ``out``."""
    f0 = np.asarray(f0, dtype=float)
    gain_db = np.asarray(gain_db, dtype=float)
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    a0 = 1 + al / A
    out[:, 0] = (1 + al * A) / a0
    out[:, 1] = (-2 * c) / a0
    out[:, 2] = (1 - al * A) / a0
    out[:, 3] = 1.0
    out[:, 4] = (-2 * c) / a0
    out[:, 5] = (1 - al / A) / a0
    return out


class NotchBank:
    """Dynamic notch bank (the zero-added-latency audio path).

    ``sos`` and ``zi`` are views of preallocated ``max_notches``-row buffers,
    sliced to the active count, so ``hit`` / ``tick`` never grow or shrink
    the underlying arrays (issue #14).
    """

    def __init__(self, fs=RESEARCH_FS, max_notches=24, q=20.0, first_db=-6.0,
                 step_db=-3.0, min_db=-30.0, hold_s=8.0, release_db_per_s=1.5):
        self.fs = float(fs)
        self.max = max_notches
        self.q = q
        self.first, self.step, self.min = first_db, step_db, min_db
        self.hold, self.rel = hold_s, release_db_per_s
        self.reset()
        _accel_warmup()

    def reset(self):
        M = self.max
        self.notches = []
        self._zi = np.zeros((M, 2))
        self._sos = np.zeros((M, 6))
        self._dirty_rows = np.zeros(M, dtype=bool)
        self._coef = np.empty((M, 6))
        self._n = 0
        self.zi = self._zi[:0]
        self.sos = self._sos[:0]
        self.n_created = 0
        self.created = []
        self.dirty = False

    def _set_n(self, n):
        self._n = n
        self.zi = self._zi[:n]
        self.sos = self._sos[:n]

    def _drop_row(self, j):
        """Remove active row ``j``, shifting later rows down in place."""
        n = self._n
        for k in range(j, n - 1):
            self._zi[k] = self._zi[k + 1]
            self._sos[k] = self._sos[k + 1]
            self._dirty_rows[k] = self._dirty_rows[k + 1]
        self._zi[n - 1] = 0.0
        self._sos[n - 1] = 0.0
        self._dirty_rows[n - 1] = False
        self._set_n(n - 1)

    def _rebuild_dirty(self):
        n = self._n
        if n == 0:
            self._dirty_rows[:self.max] = False
            return
        idx = np.flatnonzero(self._dirty_rows[:n])
        if idx.size:
            f = np.fromiter((self.notches[i]["f"] for i in idx), dtype=float, count=idx.size)
            g = np.fromiter((self.notches[i]["g"] for i in idx), dtype=float, count=idx.size)
            peaking_eq_rows(f, g, self.q, self.fs, self._coef[:idx.size])
            self._sos[idx] = self._coef[:idx.size]
        self._dirty_rows[:n] = False

    def hit(self, f, now, strength=1.0):
        for i, n in enumerate(self.notches):
            if abs(n["f"] - f) < max(8.0, 0.6 * n["f"] / self.q):
                n["f"] = 0.7 * n["f"] + 0.3 * f
                n["g"] = max(self.min, n["g"] + self.step * strength)
                n["last"] = now
                self._dirty_rows[i] = True
                self.dirty = True
                return
        if self._n >= self.max:
            j = 0
            best = self.notches[0]["g"]
            for i, n in enumerate(self.notches):
                if n["g"] > best:
                    best = n["g"]
                    j = i
            self.notches.pop(j)
            self._drop_row(j)
        self.notches.append(dict(f=f, g=self.first * max(1.0, strength), last=now))
        self.created.append((now, f))
        i = self._n
        self._zi[i] = 0.0
        self._dirty_rows[i] = True
        self._set_n(i + 1)
        self.n_created += 1
        self.dirty = True

    def tick(self, now, dt):
        write = 0
        n = self._n
        for j in range(n):
            ch = self.notches[j]
            if now - ch["last"] > self.hold:
                ch["g"] += self.rel * dt
                self._dirty_rows[j] = True
                self.dirty = True
            if ch["g"] < -1.0:
                if write != j:
                    self.notches[write] = ch
                    self._zi[write] = self._zi[j]
                    self._sos[write] = self._sos[j]
                    self._dirty_rows[write] = self._dirty_rows[j]
                write += 1
        if write != n:
            del self.notches[write:]
            self._zi[write:n] = 0.0
            self._sos[write:n] = 0.0
            self._dirty_rows[write:n] = False
            self._set_n(write)
            self.dirty = True

    def process(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        n = self._n
        if self.dirty:
            self._rebuild_dirty()
            self.dirty = False
        if n == 0:
            return x
        y, zf = sosfilt(self._sos[:n], x, zi=self._zi[:n])
        self._zi[:n] = zf
        return y
