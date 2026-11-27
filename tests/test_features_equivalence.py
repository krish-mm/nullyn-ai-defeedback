"""The optimised SideChain._features must match the v0.2.1 implementation (issue #9)."""
import numpy as np

from nullyn.features import HIST, SideChain


def _features_v021(self, k, P, dk, meanP, maxP, flux, n_strong):
    """Verbatim copy of the v0.2.1 code, kept as the reference."""
    nb = self.nb

    def lv(j):
        j = int(round(j))
        if j < 1 or j >= nb - 1:
            return -120.0
        return max(P[j - 1], P[j], P[j + 1])

    neigh = np.r_[P[max(k - 8, 0):max(k - 3, 0)], P[k + 4:k + 9]]
    pnpr = P[k] - 10 * np.log10(np.mean(10 ** (neigh / 10)) + 1e-12)
    harm = [lv(h * (k + dk)) for h in (2, 3, 4)]
    phpr = P[k] - max(harm)
    subhr = P[k] - lv(0.5 * (k + dk))
    idx = [(self.t - j) % HIST for j in range(8)]
    loc = np.array([self.hist[ii][max(k - 1, 0):k + 2].max() for ii in idx])
    pk_near = np.array([self.pk_hist[ii][max(k - 1, 0):k + 2].any() for ii in idx[:5]])
    ipmp = float(pk_near.mean())
    slope4 = float(np.polyfit(np.arange(4), loc[:4][::-1], 1)[0])
    slope8 = float(np.polyfit(np.arange(8), loc[::-1], 1)[0])
    fpos = []
    for ii in idx[:6]:
        row = self.pk_hist[ii][max(k - 1, 0):k + 2]
        if row.any():
            j = max(k - 1, 0) + int(np.argmax(row))
            fpos.append(j + self.fr_hist[ii][j])
    fstd = float(np.std(fpos)) if len(fpos) > 1 else 1.0
    amstd = float(np.std(loc))
    thr = P[k] - 6
    l = k
    while l > 1 and P[l - 1] > thr and k - l < 10:
        l -= 1
    r = k
    while r < nb - 2 and P[r + 1] > thr and r - k < 10:
        r += 1
    width = float(r - l + 1)
    f_hz = (k + dk) * self.fs / self.nfft
    return dict(k=k, f=f_hz, x=np.array([
        P[k] - meanP, pnpr, phpr, subhr, ipmp, slope4, slope8, fstd, amstd,
        min(self.run[k], HIST), self.birth_flux[k], width, np.log2(f_hz / 1000),
        P[k] - self.avg[k], flux, n_strong, P[k] - maxP,
    ]))


class _RefSideChain(SideChain):
    _features = _features_v021


def _signal(fs, seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = 0.02 * rng.standard_normal(n)
    # growing tones (howl-like), a glide and a harmonic stack
    x += 0.3 * np.minimum(1, t / seconds * 2) * np.sin(2 * np.pi * 1250 * t)
    x += 0.1 * np.sin(2 * np.pi * (600 + 200 * t) * t)
    for h in (1, 2, 3):
        x += 0.05 / h * np.sin(2 * np.pi * 220 * h * t)
    return x


def test_features_match_reference():
    for fs in (16000, 48000):
        new, ref = SideChain(fs), _RefSideChain(fs)
        x = _signal(fs, 3.0, seed=fs)
        n_cands = 0
        for i in range(0, len(x), 192):
            a, b = new.push(x[i:i + 192]), ref.push(x[i:i + 192])
            assert len(a) == len(b)
            for fa, fb in zip(a, b):
                assert [c["k"] for c in fa] == [c["k"] for c in fb]
                for ca, cb in zip(fa, fb):
                    n_cands += 1
                    np.testing.assert_allclose(ca["x"], cb["x"], rtol=0, atol=1e-9)
                    assert ca["f"] == cb["f"]
        assert n_cands > 50
