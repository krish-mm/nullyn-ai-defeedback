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
the published v0.1 results: P=64 partitions (256 ms at 16 kHz), mu=0.02, 3 s
sound-check identification.

v0.2 tracking (``tracking=True``, the default): a slow/accurate filter runs
alongside a faster filter; a path-change detector raises the fast step size
and briefly leans on the notch fallback when the residual jumps. This is
aimed at moving microphones without giving up the static-mic result. Set
``tracking=False`` to recover the v0.1 canceller exactly.

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


def _ring_push(buf, head, P, row):
    """Prepend ``row`` to the (P, K) window ``buf[head:head+P]`` in O(K).

    ``buf`` has 2P rows. Returns the new head and the new window view. The
    window holds exactly the values a shift-by-one would give.
    """
    if head == 0:
        buf[P:] = buf[:P]  # once every P blocks
        head = P
    head -= 1
    buf[head] = row
    return head, buf[head:head + P]


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
                 block_ms=DEFAULT_BLOCK_MS, out_fs=1.0, ref_fs=np.inf,
                 tracking=True, mu_fast=0.08, erle_drop_db=6.0,
                 leak_on_change=1.0, change_hang_blocks=None,
                 prefer_slow_db=1.5, copy_fast_after=None,
                 notch_thr_unreliable=0.4, notch_need_unreliable=1):
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
        # Tracking controls (v0.2). hang defaults to ~250 ms; copy after ~0.5 s.
        self.tracking = bool(tracking)
        self.mu_fast = float(mu_fast)
        self.erle_drop_db = float(erle_drop_db)
        self.leak_on_change = float(leak_on_change)
        hop = max(1, int(round(0.25 * self.fs / self.B)))
        self.change_hang = int(change_hang_blocks) if change_hang_blocks is not None else hop
        self.prefer_slow_db = float(prefer_slow_db)
        self.copy_fast_after = int(copy_fast_after) if copy_fast_after is not None else 2 * hop
        self.notch_thr_unreliable = float(notch_thr_unreliable)
        self.notch_need_unreliable = int(notch_need_unreliable)
        if detector is not None:
            self.inner = detector
        elif fallback:
            self.inner = NullynV0(params=v0params, fs=self.fs)
        else:
            self.inner = None
        self._thr0 = getattr(self.inner, "thr", None)
        self._need0 = getattr(self.inner, "need", None)
        self.reset()

    def reset(self):
        P, K, B = self.P, self.B + 1, self.B
        self.W = np.zeros((P, K), dtype=complex)
        self.W_fast = np.zeros((P, K), dtype=complex)
        # Partition histories (newest first) live in sliding windows over
        # buffers of 2P rows: a new block writes one row and moves the window
        # start back by one, so there is no O(P*K) shift per block. When the
        # window reaches row 0 it is copied to the upper half once (every P
        # blocks). self.X etc. are always contiguous (P, K) views.
        self._Xr = np.zeros((2 * P, K), dtype=complex)
        self._Xwr = np.zeros((2 * P, K), dtype=complex)
        self._Xpr = np.zeros((2 * P, K))
        self._Xp0r = np.zeros((2 * P, K))
        self._hx = self._hxw = self._hxp = self._hxp0 = P
        self.X = self._Xr[P:]
        self.Xw = self._Xwr[P:]
        self.Xp = self._Xpr[P:]
        self.Xp0 = self._Xp0r[P:]
        self.prev = np.zeros(B)
        self.prevw = np.zeros(B)
        self.A = np.r_[1.0, np.zeros(self.ord)]
        self.eh = np.zeros(self.win)
        self.eh_w = np.zeros(self.SMAX)
        self.eh_w_fast = np.zeros(self.SMAX)
        self.mh_w = np.zeros(self.SMAX)
        self.Af = self.A.copy()
        self.kt = 0
        self.k = 0
        self.prng = np.random.default_rng(self.probe_seed)
        self.lpc_win = np.hanning(self.win)
        self.pe_avg = 0.0
        self.px_avg = 0.0
        self.erle_peak = 0.0
        self.gamma = 1.0
        self.use_fast = False
        self.fast_wins = 0
        self.change_timer = 0
        self.mu_fast_now = self.mu_fast
        self.hurt_run = 0
        self.n_path_changes = 0
        self.n_fast_blocks = 0
        self.n_blocks = 0
        if self.inner is not None:
            self.inner.reset()
            if self._thr0 is not None:
                self.inner.thr = self._thr0
            if self._need0 is not None:
                self.inner.need = self._need0
        self.bank = self.inner.bank if self.inner is not None else _NoBank()

    def stats(self):
        d = self.inner.stats() if self.inner is not None else {}
        d.update(dict(gamma=float(self.gamma), use_fast=bool(self.use_fast),
                      change_timer=int(self.change_timer), tracking=bool(self.tracking),
                      n_path_changes=int(self.n_path_changes),
                      n_fast_blocks=int(self.n_fast_blocks),
                      n_blocks=int(self.n_blocks),
                      frac_fast=(self.n_fast_blocks / self.n_blocks) if self.n_blocks else 0.0))
        return d

    def _fir(self, x, st):
        z = np.concatenate([st, x])
        n = len(self.Af) - 1
        y = np.convolve(z[len(st) - n:], self.Af, "valid")
        return y, z[-len(st):]

    def _yhat(self, W):
        B = self.B
        return np.fft.irfft(np.sum(self.X * W, axis=0), n=2 * B)[B:]

    def _adapt(self, W, mu, e, fir_state):
        """Gradient-constrained PEM (or plain) NLMS update of W.

        ``fir_state`` is the prewhitening FIR memory for this filter, so the
        slow and fast filters do not share residual history.
        """
        if not (self.adapt and mu > 0):
            return W, fir_state
        B = self.B
        if self.pem:
            ew, fir_state = self._fir(e, fir_state)
        else:
            ew = e
        E = np.fft.rfft(np.concatenate([np.zeros(B), ew]))
        Xs = self.Xw if self.pem else self.X
        nrm = np.sum(self.Xp, axis=0)
        nrm = nrm + self.delta_rel * np.mean(nrm) + 1e-12
        G = np.conj(Xs) * (E / nrm)[None, :]
        gt = np.fft.irfft(G, axis=1, n=2 * B)[:, :B]
        W = W + mu * np.fft.rfft(np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1)
        return W, fir_state

    def _probe_step(self, x):
        """Sound-check identification: output white probe, adapt with plain NLMS."""
        B = self.B
        yhat = self._yhat(self.W)
        e = x - yhat
        E = np.fft.rfft(np.concatenate([np.zeros(B), e]))
        nrm = np.sum(self.Xp0, axis=0)
        nrm = nrm + self.delta_rel * np.mean(nrm) + 1e-12
        gt = np.fft.irfft(np.conj(self.X) * (E / nrm)[None, :], axis=1, n=2 * B)[:, :B]
        self.W += self.mu_id * np.fft.rfft(
            np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1
        )
        return self.prng.standard_normal(B) * self.probe_rms

    def _set_notch_mode(self, unreliable):
        if self.inner is None or self._thr0 is None:
            return
        if unreliable:
            self.inner.thr = min(self._thr0, self.notch_thr_unreliable)
            if self._need0 is not None:
                self.inner.need = min(self._need0, self.notch_need_unreliable)
        else:
            self.inner.thr = self._thr0
            if self._need0 is not None:
                self.inner.need = self._need0

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
            if self.tracking:
                self.W_fast = self.W.copy()

        yhat_slow = self._yhat(self.W)
        e_slow = x - yhat_slow
        pe_slow = float(np.mean(e_slow * e_slow))
        px = float(np.mean(x * x)) + 1e-16

        if self.tracking:
            yhat_fast = self._yhat(self.W_fast)
            e_fast = x - yhat_fast
            pe_fast = float(np.mean(e_fast * e_fast))
        else:
            yhat_fast = None
            e_fast = None
            pe_fast = pe_slow

        # ERLE of the slow (careful) model decides whether we dare leave it.
        erle_slow = 10.0 * np.log10(px / (pe_slow + 1e-16))
        erle_fast = 10.0 * np.log10(px / (pe_fast + 1e-16)) if self.tracking else erle_slow

        if self.pe_avg <= 0.0:
            self.pe_avg = pe_slow
            self.px_avg = px
        erle_avg = 10.0 * np.log10((self.px_avg + 1e-16) / (self.pe_avg + 1e-16))
        jump = pe_slow > self.pe_avg * (10 ** (self.erle_drop_db / 10.0))
        dropped = (erle_avg - erle_slow) >= self.erle_drop_db
        # A path change is a jump or ERLE drop from a *previously good*
        # slow model down to a poor one. Speech onsets can raise the residual
        # without the feedback path moving; those keep high ERLE. Cold start
        # (no good model yet) must not look like a path change either.
        erle_floor = 3.0
        had_good = erle_avg >= 6.0
        path_change = (self.tracking and self.kt > self.n_probe + 8
                       and had_good
                       and (jump or dropped or pe_slow > px)
                       and erle_slow < erle_floor)
        if path_change:
            self.change_timer = self.change_hang
            self.mu_fast_now = min(0.2, max(self.mu_fast, 4.0 * self.mu))
            # Never leak the slow filter: that is the static-mic result.
            if 0.0 < self.leak_on_change < 1.0:
                self.W_fast *= self.leak_on_change
            self.n_path_changes += 1

        # Prefer the slow filter whenever it still has useful ERLE. Switch to
        # the fast filter only if it is clearly quieter *and* the slow model
        # has lost its grip (or we are inside a detected path-change hang).
        if self.tracking:
            fast_better = pe_fast * (10 ** (self.prefer_slow_db / 10.0)) < pe_slow
            if fast_better and erle_slow < 6.0:
                self.fast_wins += 1
            else:
                self.fast_wins = 0
            if self.change_timer > 0 and erle_slow < 6.0:
                self.use_fast = True
            elif self.fast_wins >= 4:
                self.use_fast = True
            elif erle_slow >= 3.0 or pe_slow <= pe_fast:
                self.use_fast = False
            yhat = yhat_fast if self.use_fast else yhat_slow
            pe_sel = pe_fast if self.use_fast else pe_slow
        else:
            yhat = yhat_slow
            pe_sel = pe_slow

        if self.change_timer > 0 and not path_change:
            self.change_timer -= 1
            if self.change_timer == 0:
                self.mu_fast_now = self.mu_fast
        self.pe_avg = 0.9 * self.pe_avg + 0.1 * pe_slow
        self.px_avg = 0.9 * self.px_avg + 0.1 * px
        self.n_blocks += 1
        if self.use_fast:
            self.n_fast_blocks += 1

        # Safety gate: only back off after several consecutive blocks where
        # the selected prediction is louder than the microphone.
        if self.tracking and pe_sel > px:
            self.hurt_run += 1
        else:
            self.hurt_run = 0
        if self.hurt_run >= 3:
            target = px / (pe_sel + 1e-16)
            self.gamma = 0.8 * self.gamma + 0.2 * float(np.clip(target, 0.0, 1.0))
        else:
            self.gamma = 0.8 * self.gamma + 0.2 * 1.0
        e = x - self.gamma * yhat

        unreliable = self.tracking and (self.change_timer > 0 or self.gamma < 0.7)
        self._set_notch_mode(unreliable)

        if self.adapt and self.mu > 0:
            # Adapt the slow filter on the slow residual; the fast filter on
            # its own residual (and a little harder during a path change).
            self.W, self.eh_w = self._adapt(self.W, self.mu, e_slow, self.eh_w)
            if self.tracking:
                self.W_fast, self.eh_w_fast = self._adapt(
                    self.W_fast, self.mu_fast_now, e_fast, self.eh_w_fast
                )
                if self.fast_wins >= self.copy_fast_after and pe_fast < px:
                    self.W = self.W_fast.copy()
                    self.eh_w = self.eh_w_fast.copy()
                    self.fast_wins = 0
            self.eh = np.concatenate([self.eh[B:], e]) if self.win >= B else e[-self.win:]
            if len(self.eh) > self.win:
                self.eh = self.eh[-self.win:]
            if len(self.eh) < self.win:
                self.eh = np.concatenate([np.zeros(self.win - len(self.eh)), self.eh])
            self.k += 1
            every = 1 if (self.tracking and self.change_timer > 0) else self.every
            if self.pem and self.k % every == 0:
                self._update_lpc()

        out = self.inner.process(e) if self.inner is not None else e
        return np.clip(out, -self.out_fs, self.out_fs)

    def _update_lpc(self):
        w = self.eh * self.lpc_win
        # Only lags 0..ord are needed. np.correlate(w, w, "full") computed all
        # 2*len(w)-1 lags (O(len(w)^2)); these p+1 dot products are O(len(w)*p)
        # and give the same values (each lag is the same dot product).
        n = len(w)
        rr = np.array([np.dot(w[:n - k], w[k:]) for k in range(self.ord + 1)])
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
        self._hx, self.X = _ring_push(self._Xr, self._hx, self.P,
                                      np.fft.rfft(np.concatenate([self.prev, m])))
        self.prev = m
        if self.pem:
            mw, self.mh_w = self._fir(m, self.mh_w)
            self._hxw, self.Xw = _ring_push(self._Xwr, self._hxw, self.P,
                                            np.fft.rfft(np.concatenate([self.prevw, mw])))
            self.prevw = mw
            src = self.Xw[0]
        else:
            src = self.X[0]
        self._hxp, self.Xp = _ring_push(self._Xpr, self._hxp, self.P, np.abs(src) ** 2)
        if self.n_probe and self.kt <= self.n_probe:
            self._hxp0, self.Xp0 = _ring_push(self._Xp0r, self._hxp0, self.P, np.abs(self.X[0]) ** 2)
