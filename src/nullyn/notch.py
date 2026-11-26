"""Notch-layer processors: classic rule-based detector and Nullyn AI v0.

Audio-path filtering is causal IIR (no look-ahead). Decisions come from the
side chain with a reaction time of about one analysis window (~64–100 ms).
"""

from __future__ import annotations

import numpy as np

from .constants import RESEARCH_FS
from .features import SideChain
from .filters import NotchBank
from .models import load_classic_config, load_detector_params


class TinyMLP:
    """NumPy forward pass of the small MLP (17-32-16-1, 1,121 parameters)."""

    def __init__(self, params):
        self.mu, self.sd = params["mu"], params["sd"]
        self.W, self.b = params["W"], params["b"]

    def __call__(self, X):
        h = (X - self.mu) / self.sd
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return 1 / (1 + np.exp(-h[:, 0]))


class NotchSuppressor:
    """Detector + NotchBank. Subclasses implement ``decide(cands) -> [(f, strength)]``."""

    name = "notch"
    latency_samples = 0

    def __init__(self, fs=RESEARCH_FS, **bank_kw):
        self.fs = float(fs)
        self.bank = NotchBank(self.fs, **bank_kw)
        self.reset()

    def reset(self):
        self.bank.reset()
        self.sc = SideChain(self.fs)
        self.now = 0.0
        self.persist = {}
        self.log = []

    def stats(self):
        return dict(
            notches_created=self.bank.n_created,
            notches_active=len(self.bank.notches),
            depth_sum_db=float(sum(-n["g"] for n in self.bank.notches)),
        )

    def _consume(self, chunk):
        y = self.bank.process(chunk)
        for cands in self.sc.push(chunk):
            self.now += self.sc.hop / self.sc.fs
            self.bank.tick(self.now, self.sc.hop / self.sc.fs)
            for f, s in self.decide(cands):
                self.bank.hit(f, self.now, s)
                self.log.append((self.now, f))
        return y

    def process(self, x):
        """Filter ``x`` and update notches when an analysis frame completes.

        Long arrays are split at hop boundaries so coefficient updates take
        effect on later samples of the same call, matching block-wise use.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return x
        outs = []
        i = 0
        while i < len(x):
            until_frame = self.sc.hop - self.sc.pending
            if until_frame <= 0:
                until_frame = self.sc.hop
            n = min(len(x) - i, until_frame)
            outs.append(self._consume(x[i:i + n]))
            i += n
        return np.concatenate(outs)

    def _persistent(self, flagged_freqs, need):
        new = {}
        out = []
        for f in flagged_freqs:
            key = int(round(f / 10.0))
            c = 1 + max(self.persist.get(key + d, 0) for d in (-1, 0, 1))
            new[key] = c
            if c >= need:
                out.append(f)
        self.persist = new
        return out

    def decide(self, cands):
        raise NotImplementedError


class ClassicNotch(NotchSuppressor):
    """Classic automatic notch suppressor (van Waterschoot & Moonen 2011 style).

    Default thresholds are the values tuned on the training rooms
    (``classic_tuned.json``): PNPR, PHPR, PAPR, plus IPMP persistence.
    A peak must also be flagged in ``need`` consecutive frames.
    """

    name = "classic_notch"

    def __init__(self, fs=RESEARCH_FS, pnpr=None, phpr=None, papr=None,
                 ipmp=0.6, need=None, **kw):
        cfg = load_classic_config()
        self.th = (
            cfg["pnpr"] if pnpr is None else pnpr,
            cfg["phpr"] if phpr is None else phpr,
            cfg["papr"] if papr is None else papr,
            ipmp,
        )
        self.need = cfg.get("need", 2) if need is None else need
        super().__init__(fs=fs, **kw)

    def decide(self, cands):
        pn, ph, pa, ip = self.th
        fl = [c["f"] for c in cands
              if c["x"][1] >= pn and c["x"][2] >= ph and c["x"][0] >= pa and c["x"][4] >= ip]
        return [(f, 1.0) for f in self._persistent(fl, self.need)]


class NullynV0(NotchSuppressor):
    """Nullyn AI v0: same notch bank, with a learned classifier on 17 features."""

    name = "nullyn_v0"

    def __init__(self, params=None, model="v0", thr=None, need=2, fs=RESEARCH_FS, **kw):
        if params is None:
            params = load_detector_params(model)
        self.model = TinyMLP(params)
        self.thr = params.get("thr", 0.5) if thr is None else thr
        self.need = need
        super().__init__(fs=fs, **kw)

    def decide(self, cands):
        if not cands:
            self._persistent([], self.need)
            return []
        X = np.array([c["x"] for c in cands])
        p = self.model(X)
        fl = [c["f"] for c, pi in zip(cands, p) if pi >= self.thr]
        return [(f, 1.0) for f in self._persistent(fl, self.need)]
