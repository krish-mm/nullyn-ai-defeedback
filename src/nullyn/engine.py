"""Streaming Nullyn AI engine: ``process(mic_block, ref_block=None)``."""

from __future__ import annotations

import numpy as np

from .constants import DEFAULT_BLOCK_MS, RESEARCH_FS
from .notch import ClassicNotch, NullynV0


def _mono(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise ValueError(f"{name} must be a mono vector; got shape {x.shape}")
    elif x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {x.shape}")
    return x


class NullynEngine:
    """Block-based streaming processor for the Nullyn AI algorithm.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    mode : {'auto', 'notch', 'closedloop'}
        ``auto`` uses the closed-loop canceller when a reference block is
        supplied, and the notch layer alone otherwise. ``notch`` never uses
        the canceller. ``closedloop`` requires ``nullyn.closedloop`` and a
        reference block on every call (zeros may be passed until the
        speaker feed is connected).
    detector : {'nullyn_v0', 'nullyn_v0.1', 'classic'}
        Howl detector that steers the notch bank.
    soundcheck_s : float
        If greater than zero, the closed-loop canceller starts in sound-check
        mode (probe noise + path identification) for this many seconds.
    """

    def __init__(self, fs=48000, mode="auto", detector="nullyn_v0",
                 soundcheck_s=0.0, block_ms=DEFAULT_BLOCK_MS, **kw):
        self.fs = float(fs)
        self.mode = mode
        self.detector_name = detector
        self.block_ms = float(block_ms)
        self.block = max(8, int(round(self.block_ms * 1e-3 * self.fs)))
        self.notch = self._make_detector(detector, kw)
        self.afc = None
        if mode in ("auto", "closedloop"):
            try:
                from .closedloop import ClosedLoopAFC
            except ImportError as exc:
                if mode == "closedloop":
                    raise ImportError(
                        "nullyn.closedloop is not available. Use mode='notch' "
                        "or restore the closedloop package."
                    ) from exc
            else:
                fallback = kw.pop("fallback", True)
                self.afc = ClosedLoopAFC(
                    fs=self.fs,
                    detector=self.notch if fallback else None,
                    fallback=fallback,
                    soundcheck_s=soundcheck_s,
                    block_ms=self.block_ms,
                    **{k: v for k, v in kw.items() if k in (
                        "P", "mu", "lpc_order", "lpc_every", "lpc_win", "pem",
                        "delta_rel", "adapt", "ltp", "mu_id", "probe_rms",
                        "probe_seed", "out_fs", "ref_fs", "tracking", "mu_fast",
                        "erle_drop_db", "leak_on_change", "change_hang_blocks",
                        "prefer_slow_db", "copy_fast_after",
                        "notch_thr_unreliable", "notch_need_unreliable",
                    )},
                )
                self.block = self.afc.B
        self.reset()

    def _make_detector(self, detector, kw):
        bank_kw = {k: kw[k] for k in (
            "max_notches", "q", "first_db", "step_db", "min_db", "hold_s", "release_db_per_s",
        ) if k in kw}
        name = str(detector).lower()
        if name in ("classic", "classic_notch"):
            return ClassicNotch(fs=self.fs, **bank_kw)
        if name in ("nullyn_v0", "v0"):
            return NullynV0(model="v0", fs=self.fs, **bank_kw)
        if name in ("nullyn_v0.1", "v0.1", "nullyn_v0_1"):
            proc = NullynV0(model="v0.1", fs=self.fs, **bank_kw)
            proc.name = "nullyn_v0.1"
            return proc
        raise KeyError(f"unknown detector {detector!r}")

    def reset(self):
        self.notch.reset()
        if self.afc is not None:
            self.afc.reset()
        self._in_m = np.zeros(0)
        self._in_r = np.zeros(0)
        self._have_ref = False
        self._out = np.zeros(0)

    def stats(self):
        if self.afc is not None:
            d = dict(self.afc.stats())
            d["mode"] = "closedloop" if self._have_ref or self.mode == "closedloop" else "notch"
            return d
        d = dict(self.notch.stats())
        d["mode"] = "notch"
        return d

    def process(self, mic_block, ref_block=None):
        """Process a block of microphone samples.

        Parameters
        ----------
        mic_block : array-like, shape (n,)
            Microphone (or channel-insert) samples.
        ref_block : array-like, shape (n,), optional
            Speaker-feed reference, same length as ``mic_block``. When
            omitted, only the notch layer runs (in ``auto`` / ``notch``
            modes).

        Returns
        -------
        ndarray, shape (n,)
            Processed samples, the same length as ``mic_block``.
        """
        mic = _mono(mic_block, "mic_block")
        n = len(mic)
        use_afc = self.afc is not None and (
            self.mode == "closedloop" or (self.mode == "auto" and ref_block is not None) or self._have_ref
        )
        if self.mode == "closedloop" and self.afc is None:
            raise RuntimeError("closed-loop mode requested but nullyn.closedloop is missing")
        if not use_afc:
            # Notch layer accepts any length; emit immediately with no extra delay.
            return self.notch.process(mic)

        if ref_block is not None:
            ref = _mono(ref_block, "ref_block")
            if len(ref) != n:
                raise ValueError(
                    f"ref_block length ({len(ref)}) must match mic_block length ({n})"
                )
            self._have_ref = True
        else:
            ref = np.zeros(n)

        self._in_m = np.concatenate([self._in_m, mic])
        self._in_r = np.concatenate([self._in_r, ref])
        produced = []
        B = self.block
        while len(self._in_m) >= B:
            m = self._in_m[:B]
            r = self._in_r[:B]
            self._in_m = self._in_m[B:]
            self._in_r = self._in_r[B:]
            y = self.afc.process(m)
            self.afc.push_ref(r)
            produced.append(y)
        if produced:
            self._out = np.concatenate([self._out, *produced])
        if len(self._out) < n:
            out = np.concatenate([np.zeros(n - len(self._out)), self._out])
            self._out = np.zeros(0)
            return out
        out = self._out[:n]
        self._out = self._out[n:]
        return out
