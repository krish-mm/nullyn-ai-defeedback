"""PEM-prewhitened partitioned-block frequency-domain NLMS canceller.

Audio path: e[n] = mic[n] - yhat[n], where yhat is the adaptive estimate of
the speaker-to-mic feedback, computed from earlier reference blocks only.
That is why the canceller adds no samples of audio latency, provided the
electro-acoustic loop delay outside the processor (mixer + speakers + air,
plus any USB buffer delay on the reference) exceeds one AFC block.

Adaptation follows Spriet et al. 2005 and van Waterschoot & Moonen 2011:
partitioned-block frequency-domain NLMS (overlap-save, gradient-constrained)
on PEM-prewhitened signals (LPC of the error, plus an optional pitch
predictor). Default parameters match the closed-loop simulation that produced
the published results: P=64 partitions (256 ms at 16 kHz), mu=0.02, 3 s
sound-check identification.

This module is the part of the method that may be withheld from a public
repository until a provisional patent is filed.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz

from ..constants import DEFAULT_BLOCK_MS, RESEARCH_FS
from ..notch import NullynV0


def afc_block_size(fs, block_ms=DEFAULT_BLOCK_MS):
    """Return the AFC block length in samples (4 ms by default)."""
    return max(8, int(round(block_ms * 1e-3 * fs)))


class _NoBank:
    created = []
    n_created = 0
    notches = []


class ClosedLoopAFC:
    """Nullyn AI closed-loop canceller with optional Nullyn AI v0 notch fallback.

    Call ``process(mic_block)`` to get the residual (and fallback notches),
    then ``push_ref(ref_block)`` with the speaker-feed samples that belong to
    the same instant. ``yhat`` for the current block is formed only from
    earlier reference blocks, so the current reference can be pushed after
    ``process`` without adding audio-path delay.

    During ``soundcheck_s`` seconds after ``reset()``, ``process`` outputs
    white probe noise and identifies the path with plain NLMS (open loop).
    """

    name = "closedloop"
    latency_samples = 0

    def __init__(self, fs=RESEARCH_FS, v0params=None, fallback=True, detector=None,
                 P=64, mu=0.02, lpc_order=16, lpc_every=4, lpc_win=None,
                 pem=True, delta_rel=1e-2, adapt=True, ltp=True,
                 soundcheck_s=0.0, mu_id=0.5, probe_rms=0.05, probe_seed=0,
                 block_ms=DEFAULT_BLOCK_MS, out_fs=1.0, ref_fs=np.inf):
        self.fs = float(fs)
        self.B = afc_block_size(self.fs, block_ms)
        self.P, self.mu, self.ord, self.every = P, mu, lpc_order, lpc_every
        scale = self.fs / RESEARCH_FS
        self.win = int(round((lpc_win if lpc_win is not None else 1024) * scale))
        if self.win % 2:
            self.win += 1
        self.pem, self.fallback, self.delta_rel, self.adapt = pem, fallback, delta_rel, adapt
        self.ltp = ltp
        self.n_probe = int(round(soundcheck_s * self.fs / self.B))
        self.mu_id, self.probe_rms, self.probe_seed = mu_id, probe_rms, probe_seed
        # Pitch lag 40–320 samples at 16 kHz (2.5–20 ms).
        self.lag_min = max(2, int(round(40 * scale)))
        self.lag_max = max(self.lag_min + 8, int(round(320 * scale)))
        self.corr_len = max(16, int(round(256 * scale)))
        self.SMAX = lpc_order + self.lag_max + 16
        self.out_fs = out_fs
        self.ref_fs = ref_fs
        if detector is not None:
            self.inner = detector
        elif fallback:
            self.inner = NullynV0(params=v0params, fs=self.fs)
        else:
            self.inner = None
        self.reset()

    def reset(self):
        P, K, B = self.P, self.B + 1, self.B
        self.W = np.zeros((P, K), dtype=complex)
        self.X = np.zeros((P, K), dtype=complex)
        self.Xw = np.zeros((P, K), dtype=complex)
        self.Xp = np.zeros((P, K))
        self.Xp0 = np.zeros((P, K))
        self.prev = np.zeros(B)
        self.prevw = np.zeros(B)
        self.A = np.r_[1.0, np.zeros(self.ord)]
        self.eh = np.zeros(self.win)
        self.eh_w = np.zeros(self.SMAX)
        self.mh_w = np.zeros(self.SMAX)
        self.Af = self.A.copy()
        self.kt = 0
        self.k = 0
        self.prng = np.random.default_rng(self.probe_seed)
        self.lpc_win = np.hanning(self.win)
        if self.inner is not None:
            self.inner.reset()
        self.bank = self.inner.bank if self.inner is not None else _NoBank()

    def stats(self):
        return self.inner.stats() if self.inner is not None else {}

    def _fir(self, x, st):
        z = np.concatenate([st, x])
        n = len(self.Af) - 1
        y = np.convolve(z[len(st) - n:], self.Af, "valid")
        return y, z[-len(st):]

    def _probe_step(self, x):
        """Sound-check identification: output white probe, adapt with plain NLMS."""
        B = self.B
        yhat = np.fft.irfft(np.sum(self.X * self.W, axis=0), n=2 * B)[B:]
        e = x - yhat
        E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
        nrm = np.sum(self.Xp0, axis=0)
        nrm = nrm + self.delta_rel * np.mean(nrm) + 1e-12
        gt = np.fft.irfft(np.conj(self.X) * (E / nrm)[None, :], axis=1, n=2 * B)[:, :B]
        self.W += self.mu_id * np.fft.rfft(
            np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1
        )
        return self.prng.standard_normal(B) * self.probe_rms

    def process(self, x):
        """Cancel predicted feedback in one AFC block and run the notch fallback.

        ``x`` must be exactly one AFC block (``self.B`` samples).
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        B = self.B
        if len(x) != B:
            raise ValueError(f"ClosedLoopAFC.process expects a block of {B} samples, got {len(x)}")
        self.kt += 1
        if self.kt <= self.n_probe:
            return self._probe_step(x)
        if self.kt == self.n_probe + 1 and self.n_probe:
            self.Xp[:] = 0.0
            self.Xw[:] = 0.0
        yhat = np.fft.irfft(np.sum(self.X * self.W, axis=0), n=2 * B)[B:]
        e = x - yhat
        if self.adapt and self.mu > 0:
            if self.pem:
                ew, self.eh_w = self._fir(e, self.eh_w)
            else:
                ew = e
            E = np.fft.rfft(np.concatenate([np.zeros(B), ew]))
            Xs = self.Xw if self.pem else self.X
            nrm = np.sum(self.Xp, axis=0)
            nrm = nrm + self.delta_rel * np.mean(nrm) + 1e-12
            G = np.conj(Xs) * (E / nrm)[None, :]
            gt = np.fft.irfft(G, axis=1, n=2 * B)[:, :B]
            self.W += self.mu * np.fft.rfft(
                np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1
            )
            self.eh = np.concatenate([self.eh[B:], e]) if self.win >= B else e[-self.win:]
            if len(self.eh) > self.win:
                self.eh = self.eh[-self.win:]
            if len(self.eh) < self.win:
                self.eh = np.concatenate([np.zeros(self.win - len(self.eh)), self.eh])
            self.k += 1
            if self.pem and self.k % self.every == 0:
                self._update_lpc()
        out = self.inner.process(e) if self.inner is not None else e
        return np.clip(out, -self.out_fs, self.out_fs)

    def _update_lpc(self):
        w = self.eh * self.lpc_win
        rr = np.correlate(w, w, "full")[len(w) - 1:len(w) + self.ord]
        if rr[0] > 1e-12:
            rr = rr * np.exp(-0.5 * (0.01 * np.arange(self.ord + 1)) ** 2)
            rr[0] *= 1.0001
            try:
                a = solve_toeplitz(rr[:-1], -rr[1:])
                self.A = np.r_[1.0, a]
            except Exception:
                pass
            Af = self.A
            if self.ltp:
                res = np.convolve(self.eh, self.A, "valid")
                cur = res[-self.corr_len:]
                # Search lags lag_min..lag_max, same geometry as the 16 kHz original.
                left = self.corr_len + self.lag_max
                right = self.lag_min
                if len(res) > left:
                    seg = res[-left:-right]
                    c = np.correlate(seg, cur, "valid")[::-1]
                    sq = np.concatenate([[0.0], np.cumsum(seg ** 2)])
                    den = (sq[self.corr_len:] - sq[:-self.corr_len])[::-1] + 1e-12
                    n = min(len(c), len(den))
                    c, den = c[:n], den[:n]
                    sc_ = np.where(c > 0, c * c / den, 0.0)
                    j = int(np.argmax(sc_))
                    best = float(sc_[j])
                    T = self.lag_min + j
                    bt = float(c[j] / den[j])
                    if T and best / (np.dot(cur, cur) + 1e-12) > 0.2:
                        lt = np.zeros(T + 1)
                        lt[0] = 1.0
                        lt[T] = -min(bt, 1.0)
                        Af = np.convolve(self.A, lt)
            self.Af = Af

    def push_ref(self, m):
        """Accept one AFC block of speaker-feed (reference) samples."""
        m = np.asarray(m, dtype=float).reshape(-1)
        B = self.B
        if len(m) != B:
            raise ValueError(f"ClosedLoopAFC.push_ref expects a block of {B} samples, got {len(m)}")
        m = np.clip(m, -self.ref_fs, self.ref_fs)
        self.X[1:] = self.X[:-1]
        self.X[0] = np.fft.rfft(np.concatenate([self.prev, m]))
        self.prev = m
        if self.pem:
            mw, self.mh_w = self._fir(m, self.mh_w)
            self.Xw[1:] = self.Xw[:-1]
            self.Xw[0] = np.fft.rfft(np.concatenate([self.prevw, mw]))
            self.prevw = mw
            src = self.Xw[0]
        else:
            src = self.X[0]
        self.Xp[1:] = self.Xp[:-1]
        self.Xp[0] = np.abs(src) ** 2
        if self.n_probe and self.kt <= self.n_probe:
            self.Xp0[1:] = self.Xp0[:-1]
            self.Xp0[0] = np.abs(self.X[0]) ** 2
