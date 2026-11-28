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
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([1 + al * A, -2 * c, 1 - al * A])
    a = np.array([1 + al / A, -2 * c, 1 - al / A])
    return b / a[0], a / a[0]


class NotchBank:
    """Dynamic notch bank (the zero-added-latency audio path)."""

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
        self.notches = []
        self.zi = np.zeros((0, 2))
        self.sos = np.zeros((0, 6))
        self.n_created = 0
        self.created = []
        self.dirty = False

    def hit(self, f, now, strength=1.0):
        for n in self.notches:
            if abs(n["f"] - f) < max(8.0, 0.6 * n["f"] / self.q):
                n["f"] = 0.7 * n["f"] + 0.3 * f
                n["g"] = max(self.min, n["g"] + self.step * strength)
                n["last"] = now
                self.dirty = True
                return
        if len(self.notches) >= self.max:
            j = int(np.argmax([n["g"] for n in self.notches]))
            self.notches.pop(j)
            self.zi = np.delete(self.zi, j, axis=0)
        self.notches.append(dict(f=f, g=self.first * max(1.0, strength), last=now))
        self.created.append((now, f))
        self.zi = np.vstack([self.zi, np.zeros((1, 2))])
        self.n_created += 1
        self.dirty = True

    def tick(self, now, dt):
        keep = []
        for j, n in enumerate(self.notches):
            if now - n["last"] > self.hold:
                n["g"] += self.rel * dt
                self.dirty = True
            if n["g"] < -1.0:
                keep.append(j)
        if len(keep) != len(self.notches):
            self.notches = [self.notches[j] for j in keep]
            self.zi = self.zi[keep] if len(keep) else np.zeros((0, 2))
            self.dirty = True

    def process(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.dirty:
            rows = []
            for n in self.notches:
                b, a = peaking_eq(n["f"], n["g"], self.q, self.fs)
                rows.append(np.r_[b, a])
            self.sos = np.array(rows).reshape(-1, 6)
            self.dirty = False
        if len(self.notches) == 0:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y
