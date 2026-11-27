"""Causal side-chain analysis: spectral peak candidates and per-candidate features.

Runs on the mic (or residual) signal every hop samples on a Hann window.
Nothing here touches the audio path; it only produces (frequency, features)
for each candidate peak, which a detector turns into notch decisions.

Window and hop lengths scale with sample rate so that a 64 ms window and
16 ms hop are used at any rate (1,024 / 256 samples at the 16 kHz research rate).
"""

from __future__ import annotations

import math

import numpy as np

from .constants import RESEARCH_FS

NFFT_AT_16K = 1024  # 64 ms window at 16 kHz
HOP_AT_16K = 256    # 16 ms hop
NCAND = 5           # candidate peaks per frame
HIST = 64           # frames of history (~1 s)
FMIN, FMAX = 80.0, 7800.0

_J8 = np.arange(8)
_X4 = np.arange(4) - 1.5      # centred abscissae for the closed-form slopes
_X8 = np.arange(8) - 3.5
_D4 = float(_X4 @ _X4)
_D8 = float(_X8 @ _X8)

def _std(x):
    """np.std(x) for a short 1-D float array, without the ufunc-dispatch overhead.

    Same operations in the same order as NumPy's _var (sum/n, subtract,
    square, sum/n, sqrt), so the result is identical.
    """
    n = x.shape[0]
    d = x - x.sum() / n
    return math.sqrt((d * d).sum() / n)


FEATURE_NAMES = [
    "papr", "pnpr", "phpr", "subhr", "ipmp", "slope4", "slope8", "fstd", "amstd",
    "dur", "birth_flux", "width", "logf", "lvl_vs_avg", "flux", "n_strong", "lvl_rel_max",
]


class SideChain:
    def __init__(self, fs=RESEARCH_FS, fmax=None):
        self.fs = float(fs)
        scale = self.fs / RESEARCH_FS
        self.nfft = int(round(NFFT_AT_16K * scale))
        if self.nfft % 2:
            self.nfft += 1
        self.hop = max(1, int(round(HOP_AT_16K * scale)))
        fmax = FMAX if fmax is None else fmax
        self.win = np.hanning(self.nfft)
        self.buf = np.zeros(self.nfft)
        self.nb = self.nfft // 2 + 1
        self.kmin = max(2, int(FMIN / self.fs * self.nfft))
        self.kmax = min(int(fmax / self.fs * self.nfft), self.nb - 12)
        self.hist = np.full((HIST, self.nb), -120.0)
        self.pk_hist = np.zeros((HIST, self.nb), dtype=bool)
        self.fr_hist = np.zeros((HIST, self.nb))
        self.flux_hist = np.zeros(HIST)
        self.run = np.zeros(self.nb, dtype=int)
        self.birth_flux = np.zeros(self.nb)
        self.avg = np.full(self.nb, -120.0)
        self.t = 0
        self.pending = 0

    def reset(self):
        self.buf[:] = 0
        self.hist[:] = -120.0
        self.pk_hist[:] = False
        self.fr_hist[:] = 0
        self.flux_hist[:] = 0
        self.run[:] = 0
        self.birth_flux[:] = 0
        self.avg[:] = -120.0
        self.t = 0
        self.pending = 0

    def push(self, x):
        """Feed a block; returns a list of frames, each a list of candidates.

        The analysis window slides through ``x`` at hop intervals, so a long
        array and a sequence of short blocks produce the same frames.
        """
        out = []
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return out
        work = np.concatenate([self.buf, x])
        n = len(x)
        first = self.hop - self.pending
        if first <= n:
            for k in range(first, n + 1, self.hop):
                self.buf = work[k:k + self.nfft].copy()
                out.append(self.analyze())
            last_k = first + ((n - first) // self.hop) * self.hop
            self.pending = n - last_k
        else:
            self.pending += n
        self.buf = work[-self.nfft:].copy()
        return out

    def analyze(self):
        X = np.fft.rfft(self.buf * self.win)
        P = 10 * np.log10(np.abs(X) ** 2 + 1e-12)
        prev = self.hist[(self.t - 1) % HIST]
        flux = float(np.mean(np.maximum(P - prev, 0)[self.kmin:self.kmax]))
        i = self.t % HIST
        self.hist[i] = P
        self.flux_hist[i] = flux
        lo, hi = self.kmin, self.kmax
        seg = P[lo - 1:hi + 2]
        is_pk = np.zeros(self.nb, dtype=bool)
        is_pk[lo:hi + 1] = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
        d = np.zeros(self.nb)
        k = np.nonzero(is_pk)[0]
        if len(k):
            a, b, c = P[k - 1], P[k], P[k + 1]
            den = a - 2 * b + c
            d[k] = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
        self.pk_hist[i] = is_pk
        self.fr_hist[i] = d
        prev_pk = self.pk_hist[(self.t - 1) % HIST]
        near_prev = prev_pk | np.roll(prev_pk, 1) | np.roll(prev_pk, -1)
        prev_run = np.maximum(np.maximum(self.run, np.roll(self.run, 1)), np.roll(self.run, -1))
        newrun = np.where(is_pk & near_prev, prev_run + 1, np.where(is_pk, 1, 0))
        born = is_pk & (newrun == 1)
        cont = is_pk & ~born
        bf = np.maximum(np.maximum(self.birth_flux, np.roll(self.birth_flux, 1)),
                        np.roll(self.birth_flux, -1))
        self.birth_flux = np.where(born, flux, np.where(cont, bf, 0.0))
        self.run = newrun
        meanP = 10 * np.log10(np.mean(10 ** (P[lo:hi] / 10)) + 1e-12)
        maxP = np.max(P[lo:hi])
        cands = []
        if len(k):
            order = k[np.argsort(P[k])[::-1]][:NCAND]
            n_strong = int(np.sum(P[k] > meanP + 20))
            for kk in order:
                if P[kk] < meanP + 6:
                    continue
                cands.append(self._features(kk, P, d[kk], meanP, maxP, flux, n_strong))
        self.avg = 0.94 * self.avg + 0.06 * P if self.t else P.copy()
        self.t += 1
        return cands

    def _features(self, k, P, dk, meanP, maxP, flux, n_strong):
        nb = self.nb

        def lv(j):
            j = int(round(j))
            if j < 1 or j >= nb - 1:
                return -120.0
            return P[j - 1:j + 2].max()

        neigh = np.concatenate((P[max(k - 8, 0):max(k - 3, 0)], P[k + 4:k + 9]))
        pnpr = P[k] - 10 * np.log10(np.mean(10 ** (neigh / 10)) + 1e-12)
        kf = k + dk
        phpr = P[k] - max(lv(2 * kf), lv(3 * kf), lv(4 * kf))
        subhr = P[k] - lv(0.5 * kf)
        # Last 8 frames (newest first), gathered once with fancy indexing.
        idx = (self.t - _J8) % HIST
        lo, hi = max(k - 1, 0), k + 2
        loc = self.hist[idx, lo:hi].max(axis=1)
        pk_rows = self.pk_hist[idx[:6], lo:hi]
        has_pk = pk_rows.any(axis=1)
        ipmp = np.count_nonzero(has_pk[:5]) / 5.0
        # Least-squares slopes in closed form (same fit as np.polyfit(deg=1)).
        slope4 = float(_X4 @ loc[3::-1]) / _D4
        slope8 = float(_X8 @ loc[::-1]) / _D8
        if has_pk.any():
            rows = np.nonzero(has_pk)[0]
            j = lo + np.argmax(pk_rows[rows], axis=1)
            fpos = j + self.fr_hist[idx[rows], j]
        else:
            fpos = ()
        fstd = _std(fpos) if len(fpos) > 1 else 1.0
        amstd = _std(loc)
        thr = P[k] - 6
        l = k
        while l > 1 and P[l - 1] > thr and k - l < 10:
            l -= 1
        r = k
        while r < nb - 2 and P[r + 1] > thr and r - k < 10:
            r += 1
        width = float(r - l + 1)
        f_hz = kf * self.fs / self.nfft
        return dict(k=k, f=f_hz, x=np.array([
            P[k] - meanP, pnpr, phpr, subhr, ipmp, slope4, slope8, fstd, amstd,
            min(self.run[k], HIST), self.birth_flux[k], width, np.log2(f_hz / 1000),
            P[k] - self.avg[k], flux, n_strong, P[k] - maxP,
        ]))
