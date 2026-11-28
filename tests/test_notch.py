import numpy as np

from nullyn.filters import NotchBank
from nullyn.notch import ClassicNotch, NullynV0, TinyMLP
from nullyn.models import load_classic_config, load_detector_params


def test_notch_bank_attenuates_tone():
    fs = 16000
    f0 = 1250.0
    t = np.arange(int(0.25 * fs)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    bank = NotchBank(fs=fs)
    bank.hit(f0, now=0.0, strength=1.0)
    # Deepen the notch a few times, as the live suppressor would.
    for _ in range(6):
        bank.hit(f0, now=0.05, strength=1.0)
    y = bank.process(x)
    # Compare steady-state tail energy.
    n = fs // 20
    in_e = np.mean(x[-n:] ** 2)
    out_e = np.mean(y[-n:] ** 2)
    assert out_e < 0.25 * in_e


def test_classic_config_and_detector_flags_howl_tone():
    cfg = load_classic_config()
    assert set(cfg) >= {"pnpr", "phpr", "papr", "need"}
    fs = 16000
    proc = ClassicNotch(fs=fs)
    t = np.arange(int(0.8 * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 1500 * t)
    _ = proc.process(x)
    assert proc.bank.n_created >= 1
    freqs = [n["f"] for n in proc.bank.notches]
    assert any(abs(f - 1500) < 40 for f in freqs)


def test_mlp_loads_and_scores_howl_like_vector():
    params = load_detector_params("v0")
    assert params["nparams"] == 1121
    assert params["W"][0].shape == (17, 32)
    model = TinyMLP(params)
    # A synthetic howl-like feature vector: high PAPR/PNPR/PHPR, high IPMP,
    # low frequency wobble, growing slope. Column order matches FEATURE_NAMES.
    x = np.array([[
        25.0, 20.0, 25.0, 20.0, 1.0, 2.0, 2.0, 0.05, 0.2,
        20.0, 1.0, 2.0, 0.0, 15.0, 1.0, 1.0, 0.0,
    ]])
    p = model(x)
    assert p.shape == (1,)
    assert 0.0 <= p[0] <= 1.0
    # The kept v0 model should treat a clean, growing, isolated peak as howl.
    assert p[0] >= 0.5


def test_nullyn_v0_places_a_notch_on_a_howl():
    fs = 16000
    proc = NullynV0(model="v0", fs=fs)
    t = np.arange(int(1.0 * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 1800 * t)
    y = proc.process(x)
    assert y.shape == x.shape
    assert proc.bank.n_created >= 1


# --- issue #14: preallocated bank must match the v0.4.0 vstack/delete bank ----

class _LegacyNotchBank:
    """The v0.4.0 NotchBank (grows zi with vstack / np.delete)."""

    def __init__(self, fs=16000, max_notches=24, q=20.0, first_db=-6.0,
                 step_db=-3.0, min_db=-30.0, hold_s=8.0, release_db_per_s=1.5):
        from nullyn.filters import peaking_eq
        self._peaking_eq = peaking_eq
        self.fs = float(fs)
        self.max = max_notches
        self.q = q
        self.first, self.step, self.min = first_db, step_db, min_db
        self.hold, self.rel = hold_s, release_db_per_s
        self.reset()

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
        from nullyn._accel import sosfilt
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.dirty:
            rows = []
            for n in self.notches:
                b, a = self._peaking_eq(n["f"], n["g"], self.q, self.fs)
                rows.append(np.r_[b, a])
            self.sos = np.array(rows).reshape(-1, 6)
            self.dirty = False
        if len(self.notches) == 0:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y


def test_peaking_eq_matches_original_formula():
    from nullyn.filters import peaking_eq
    rng = np.random.default_rng(14)
    for _ in range(30):
        f0 = float(rng.uniform(80, 8000))
        g = float(rng.uniform(-30, -1))
        q, fs = 20.0, 48000.0
        A = 10 ** (g / 40)
        w0 = 2 * np.pi * f0 / fs
        al = np.sin(w0) / (2 * q)
        c = np.cos(w0)
        b = np.array([1 + al * A, -2 * c, 1 - al * A])
        a = np.array([1 + al / A, -2 * c, 1 - al / A])
        br, ar = peaking_eq(f0, g, q, fs)
        np.testing.assert_array_equal(br, b / a[0])
        np.testing.assert_array_equal(ar, a / a[0])


def test_peaking_eq_rows_matches_scalar():
    from nullyn.filters import peaking_eq, peaking_eq_rows
    rng = np.random.default_rng(14)
    f = rng.uniform(80, 8000, size=16)
    g = rng.uniform(-30, -1, size=16)
    out = np.empty((16, 6))
    peaking_eq_rows(f, g, 20.0, 48000.0, out)
    for i in range(16):
        b, a = peaking_eq(f[i], g[i], 20.0, 48000.0)
        np.testing.assert_array_equal(out[i], np.r_[b, a])


def test_notch_bank_matches_legacy_bookkeeping():
    """Preallocated hit/tick/process must be bit-exact with the v0.4.0 bank."""
    kw = dict(fs=16000, max_notches=8, q=20.0, hold_s=0.05, release_db_per_s=80.0)
    new = NotchBank(**kw)
    old = _LegacyNotchBank(**kw)
    rng = np.random.default_rng(14)
    now = 0.0
    freqs = [400, 900, 1600, 2500, 3700, 5100, 6400, 800, 900, 1200, 2500]
    for i, f0 in enumerate(freqs * 3):
        now += 0.01
        new.hit(f0 + rng.normal(0, 3), now, strength=1.0 + 0.1 * (i % 3))
        old.hit(f0 + rng.normal(0, 3), now, strength=1.0 + 0.1 * (i % 3))
        # The two banks must see the same sequence; use a shared draw.
    # Re-run with a recorded sequence so both banks get identical hits.
    new.reset(); old.reset()
    now = 0.0
    events = []
    rng = np.random.default_rng(14)
    for i, f0 in enumerate(freqs * 3):
        now += 0.01
        events.append(("hit", f0 + float(rng.normal(0, 3)), now, 1.0 + 0.1 * (i % 3)))
        if i % 4 == 3:
            events.append(("tick", now, 0.01))
            events.append(("process", rng.standard_normal(64)))
        if i % 7 == 6:
            # jump past the hold so release starts
            now += 0.06
            events.append(("tick", now, 0.02))
            events.append(("process", rng.standard_normal(64)))
    for ev in events:
        if ev[0] == "hit":
            _, f, t, s = ev
            new.hit(f, t, s); old.hit(f, t, s)
        elif ev[0] == "tick":
            _, t, dt = ev
            new.tick(t, dt); old.tick(t, dt)
        else:
            x = ev[1]
            y_new = new.process(x)
            y_old = old.process(x)
            np.testing.assert_array_equal(y_new, y_old)
            np.testing.assert_array_equal(new.zi, old.zi)
            np.testing.assert_array_equal(new.sos, old.sos)
    assert [(n["f"], n["g"], n["last"]) for n in new.notches] == [
        (n["f"], n["g"], n["last"]) for n in old.notches
    ]
    assert new.n_created == old.n_created


def test_hit_tick_steady_state_does_not_allocate_arrays(monkeypatch):
    bank = NotchBank(fs=16000, max_notches=8)
    for f in (800.0, 1500.0, 2400.0):
        bank.hit(f, now=0.0)
    bank.process(np.zeros(64))
    allocs = []

    def wrap(name, fn):
        def inner(*a, **k):
            allocs.append(name)
            return fn(*a, **k)
        return inner

    import nullyn.filters as fl
    for name in ("vstack", "delete", "zeros", "empty", "array", "zeros_like", "fromiter"):
        monkeypatch.setattr(fl.np, name, wrap(name, getattr(np, name)))
    # Update existing notches and tick well inside the hold window.
    for i in range(20):
        bank.hit(800.0, now=0.01 * i)
        bank.tick(now=0.01 * i, dt=0.01)
    assert allocs == [], allocs
    assert len(bank.notches) == 3
    # The underlying buffers must stay the preallocated max-row arrays.
    assert bank._zi.shape == (8, 2)
    assert bank._sos.shape == (8, 6)
