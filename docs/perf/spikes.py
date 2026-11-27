"""Micro-benchmark spikes for candidate optimisations (48 kHz sizes unless noted)."""
import gc, json, sys, time, os
import numpy as np, scipy, scipy.fft as sfft
from scipy import signal
from scipy.linalg import solve_toeplitz
import numba
from numba import njit
import pyfftw, pyfftw.interfaces.numpy_fft as pfft
pyfftw.interfaces.cache.enable()

def tm(f, n=300, warm=30):
    for _ in range(warm): f()
    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns(); f(); t.append(time.perf_counter_ns() - t0)
    return float(np.median(t)) * 1e-6

rows = []
def rec(stage, variant, ms, base_ms, err=None, note=""):
    rows.append(dict(stage=stage, variant=variant, ms=ms, speedup=base_ms / ms if ms else None, max_abs_err=err, note=note))
    print(f"{stage:<22}{variant:<42}{ms*1e3:>9.1f}us {base_ms/ms:>6.2f}x  err={err if err is None else f'{err:.2e}'} {note}", flush=True)

rng = np.random.default_rng(0)
fs = 48000; B = 192; P = 64; K = B + 1; ORD = 16
win = 3072; corr_len = 768; lag_min = 120; lag_max = 960

# ---------------- S1 LPC autocorrelation
eh = rng.standard_normal(win); lw = np.hanning(win); w = eh * lw
f0 = lambda: np.correlate(w, w, "full")[win - 1:win + ORD]
r0 = f0(); b0 = tm(f0, 100)
rec("lpc.autocorr", "np.correlate full (baseline)", b0, b0, 0.0)
def f1():
    return np.array([np.dot(w[:win - k], w[k:]) for k in range(ORD + 1)])
rec("lpc.autocorr", "17 lag dot products", tm(f1), b0, float(np.max(np.abs(f1() - r0))))
def f1b():
    return np.correlate(w, w[: win - ORD], "valid") if False else None
from numpy.lib.stride_tricks import sliding_window_view
def f2():
    # r[k] = sum_n w[n] w[n+k]
    sw = sliding_window_view(np.r_[w, np.zeros(ORD)], ORD + 1)  # (win, ORD+1)
    return w @ sw
rec("lpc.autocorr", "sliding_window_view matvec", tm(f2), b0, float(np.max(np.abs(f2() - r0))))
def f3():
    X = np.fft.rfft(w, 2 * win); return np.fft.irfft(np.abs(X) ** 2)[:ORD + 1]
rec("lpc.autocorr", "FFT (Wiener-Khinchin)", tm(f3), b0, float(np.max(np.abs(f3() - r0))))
@njit(cache=True, fastmath=False)
def ac_nb(w, order):
    n = w.shape[0]; r = np.zeros(order + 1)
    for k in range(order + 1):
        s = 0.0
        for i in range(n - k): s += w[i] * w[i + k]
        r[k] = s
    return r
rec("lpc.autocorr", "numba njit loops", tm(lambda: ac_nb(w, ORD)), b0, float(np.max(np.abs(ac_nb(w, ORD) - r0))))

# ---------------- S2 Levinson
rr = r0 * np.exp(-0.5 * (0.01 * np.arange(ORD + 1)) ** 2); rr[0] *= 1.0001
g0 = lambda: solve_toeplitz(rr[:-1], -rr[1:]); a0 = g0(); bl = tm(g0)
rec("lpc.levinson", "scipy solve_toeplitz (baseline)", bl, bl, 0.0)
@njit(cache=True)
def levinson(r, p):
    a = np.zeros(p + 1); a[0] = 1.0; e = r[0]; tmp = np.zeros(p + 1)
    for i in range(1, p + 1):
        acc = r[i]
        for j in range(1, i): acc += a[j] * r[i - j]
        k = -acc / e
        tmp[:i + 1] = a[:i + 1]
        for j in range(1, i): a[j] = tmp[j] + k * tmp[i - j]
        a[i] = k; e *= (1 - k * k)
    return a[1:]
rec("lpc.levinson", "numba Levinson-Durbin", tm(lambda: levinson(rr, ORD)), bl, float(np.max(np.abs(levinson(rr, ORD) - a0))))
def lev_py(r, p):
    a = np.zeros(p + 1); a[0] = 1.0; e = r[0]
    for i in range(1, p + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i + 1] = a[1:i + 1] + k * np.r_[a[i - 1:0:-1], 1.0]
        e *= (1 - k * k)
    return a[1:]
rec("lpc.levinson", "pure numpy Levinson", tm(lambda: lev_py(rr, ORD)), bl, float(np.max(np.abs(lev_py(rr, ORD) - a0))))

# ---------------- S3 LTP correlation
res = rng.standard_normal(win - ORD); cur = res[-corr_len:]; left = corr_len + lag_max; seg = res[-left:-lag_min]
h0 = lambda: np.correlate(seg, cur, "valid"); c0 = h0(); bt = tm(h0, 100)
rec("ltp.xcorr", "np.correlate valid (baseline)", bt, bt, 0.0)
h1 = lambda: signal.correlate(seg, cur, "valid", method="fft")
rec("ltp.xcorr", "scipy.signal.correlate fft", tm(h1), bt, float(np.max(np.abs(h1() - c0))))
nf = sfft.next_fast_len(len(seg) + corr_len)
def h2():
    S = sfft.rfft(seg, nf); C = sfft.rfft(cur[::-1], nf)
    return sfft.irfft(S * C, nf)[corr_len - 1:len(seg)]
rec("ltp.xcorr", "scipy.fft rfft next_fast_len", tm(h2), bt, float(np.max(np.abs(h2() - c0))))
h3 = lambda: sliding_window_view(seg, corr_len) @ cur
rec("ltp.xcorr", "sliding_window_view matvec (BLAS)", tm(h3), bt, float(np.max(np.abs(h3() - c0))))

# ---------------- S4 AFC adapt
Xw = (rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K)))
Xp = np.abs(Xw) ** 2; W = np.zeros((P, K), complex); ew = rng.standard_normal(B)
def ad0(W):
    E = np.fft.rfft(np.concatenate([np.zeros(B), ew]))
    nrm = np.sum(Xp, axis=0); nrm = nrm + 1e-2 * np.mean(nrm) + 1e-12
    G = np.conj(Xw) * (E / nrm)[None, :]
    gt = np.fft.irfft(G, axis=1, n=2 * B)[:, :B]
    return W + 0.02 * np.fft.rfft(np.concatenate([gt, np.zeros_like(gt)], axis=1), axis=1)
W0 = ad0(W); ba = tm(lambda: ad0(W))
rec("afc.adapt", "numpy.fft baseline (1 filter)", ba, ba, 0.0)
def ad1(W):
    E = np.fft.rfft(ew, 2 * B) if False else np.fft.rfft(np.concatenate([np.zeros(B), ew]))
    nrm = np.sum(Xp, axis=0); nrm = nrm + 1e-2 * np.mean(nrm) + 1e-12
    G = np.conj(Xw) * (E / nrm)
    gt = np.fft.irfft(G, axis=1, n=2 * B)[:, :B]
    W += 0.02 * np.fft.rfft(gt, n=2 * B, axis=1)
    return W
rec("afc.adapt", "numpy: rfft(n=2B) pad, in-place W", tm(lambda: ad1(W.copy())), ba, float(np.max(np.abs(ad1(W.copy()) - W0))))
def ad2(W, workers=1):
    E = sfft.rfft(np.concatenate([np.zeros(B), ew]))
    nrm = np.sum(Xp, axis=0); nrm = nrm + 1e-2 * np.mean(nrm) + 1e-12
    G = np.conj(Xw) * (E / nrm)
    gt = sfft.irfft(G, axis=1, n=2 * B, workers=workers)[:, :B]
    W += 0.02 * sfft.rfft(gt, n=2 * B, axis=1, workers=workers)
    return W
rec("afc.adapt", "scipy.fft workers=1", tm(lambda: ad2(W.copy())), ba, float(np.max(np.abs(ad2(W.copy()) - W0))))
rec("afc.adapt", "scipy.fft workers=4", tm(lambda: ad2(W.copy(), 4)), ba, float(np.max(np.abs(ad2(W.copy(), 4) - W0))))
# pyfftw builders
Gbuf = pyfftw.empty_aligned((P, K), dtype="complex128"); tbuf = pyfftw.empty_aligned((P, 2 * B), dtype="float64")
Fbuf = pyfftw.empty_aligned((P, K), dtype="complex128")
inv = pyfftw.FFTW(Gbuf, tbuf, axes=(1,), direction="FFTW_BACKWARD", flags=("FFTW_MEASURE",), threads=1)
fwd = pyfftw.FFTW(tbuf, Fbuf, axes=(1,), direction="FFTW_FORWARD", flags=("FFTW_MEASURE",), threads=1)
E0 = np.fft.rfft(np.concatenate([np.zeros(B), ew]))
def ad3(W):
    nrm = np.sum(Xp, axis=0); nrm = nrm + 1e-2 * np.mean(nrm) + 1e-12
    np.multiply(np.conj(Xw), E0 / nrm, out=Gbuf)
    inv(); tbuf[:, B:] = 0.0  # FFTW backward is unnormalised; irfft normalises by 1/n
    fwd()
    W += 0.02 * Fbuf  # FFTW() call normalises the inverse by default
    return W
rec("afc.adapt", "pyfftw FFTW_MEASURE planned, preallocated", tm(lambda: ad3(W.copy())), ba, float(np.max(np.abs(ad3(W.copy()) - W0))))
Xw32 = Xw.astype(np.complex64); Xp32 = Xp.astype(np.float32); ew32 = ew.astype(np.float32)
def ad4(W):
    E = sfft.rfft(np.concatenate([np.zeros(B, np.float32), ew32]))
    nrm = np.sum(Xp32, axis=0); nrm = nrm + np.float32(1e-2) * np.mean(nrm) + np.float32(1e-12)
    G = np.conj(Xw32) * (E / nrm)
    gt = sfft.irfft(G, axis=1, n=2 * B)[:, :B]
    W += np.float32(0.02) * sfft.rfft(gt, n=2 * B, axis=1)
    return W
W32 = W.astype(np.complex64)
rec("afc.adapt", "scipy.fft complex64 (float32)", tm(lambda: ad4(W32.copy())), ba, float(np.max(np.abs(ad4(W32.copy()) - W0))), "rel err vs |dW|=%.1e" % np.max(np.abs(W0)))
# batched slow+fast
def ad_two_base():
    ad0(W); ad0(W)
bb2 = tm(ad_two_base)
rec("afc.adapt x2", "baseline: two separate adapts (slow+fast)", bb2, bb2, 0.0)
Ew2 = np.stack([E0, E0])
def ad_batched(W2):
    nrm = np.sum(Xp, axis=0); nrm = nrm + 1e-2 * np.mean(nrm) + 1e-12
    G = np.conj(Xw)[None] * (Ew2 / nrm)[:, None, :]
    gt = sfft.irfft(G, axis=-1, n=2 * B)[..., :B]
    W2 += 0.02 * sfft.rfft(gt, n=2 * B, axis=-1)
    return W2
rec("afc.adapt x2", "batched (2,P,K) one FFT call, shared nrm", tm(lambda: ad_batched(np.zeros((2, P, K), complex))), bb2, float(np.max(np.abs(ad_batched(np.zeros((2, P, K), complex))[0] - W0))))
nrm_once = None

# ---------------- S5 X history shift vs double-buffer ring
Xs = np.zeros((P, K), complex); newrow = np.ones(K, complex)
def sh0():
    Xs[1:] = Xs[:-1]; Xs[0] = newrow
bs = tm(sh0)
rec("afc.history", "shift P rows (baseline, x3 arrays/blk)", bs, bs)
R = np.zeros((2 * P, K), complex); head = [P]
def sh1():
    h = head[0] - 1
    if h < 0: h = P - 1
    R[h] = newrow; R[h + P] = newrow; head[0] = h
    return R[h:h + P]
rec("afc.history", "double-buffer ring (view, O(K) write)", tm(sh1), bs)

# ---------------- S6 yhat
Wc = W0
y0f = lambda: np.fft.irfft(np.sum(Xw * Wc, axis=0), n=2 * B)[B:]
by = tm(y0f); yy = y0f()
rec("afc.yhat", "np.sum(X*W) + irfft (baseline)", by, by, 0.0)
y1f = lambda: np.fft.irfft(np.einsum("pk,pk->k", Xw, Wc), n=2 * B)[B:]
rec("afc.yhat", "einsum pk,pk->k", tm(y1f), by, float(np.max(np.abs(y1f() - yy))))
@njit(cache=True)
def accum(X, W):
    P, K = X.shape; o = np.zeros(K, np.complex128)
    for p in range(P):
        for k in range(K): o[k] += X[p, k] * W[p, k]
    return o
y2f = lambda: np.fft.irfft(accum(Xw, Wc), n=2 * B)[B:]
rec("afc.yhat", "numba accumulate", tm(y2f), by, float(np.max(np.abs(y2f() - yy))))

# ---------------- S7 side-chain FFT
buf = rng.standard_normal(3072); hw = np.hanning(3072)
f_np = lambda: np.fft.rfft(buf * hw); bf = tm(f_np); X0 = f_np()
rec("sidechain.fft", "numpy.fft.rfft 3072 (baseline)", bf, bf, 0.0)
rec("sidechain.fft", "scipy.fft.rfft", tm(lambda: sfft.rfft(buf * hw)), bf, float(np.max(np.abs(sfft.rfft(buf * hw) - X0))))
rec("sidechain.fft", "pyfftw interfaces (cached)", tm(lambda: pfft.rfft(buf * hw)), bf, float(np.max(np.abs(pfft.rfft(buf * hw) - X0))))
P_ = 10 * np.log10(np.abs(X0) ** 2 + 1e-12)
l0 = lambda: 10 * np.log10(np.abs(X0) ** 2 + 1e-12); bl0 = tm(l0)
rec("sidechain.logpow", "10log10(abs(X)**2) (baseline)", bl0, bl0, 0.0)
l1 = lambda: 10 * np.log10(X0.real ** 2 + X0.imag ** 2 + 1e-12)
rec("sidechain.logpow", "re^2+im^2 (no sqrt)", tm(l1), bl0, float(np.max(np.abs(l1() - P_))))

# ---------------- S8 polyfit slope
loc = rng.standard_normal(8)
p0 = lambda: (np.polyfit(np.arange(4), loc[:4][::-1], 1)[0], np.polyfit(np.arange(8), loc[::-1], 1)[0])
bp = tm(p0); pv = p0()
rec("features.slope", "2x np.polyfit (baseline, per candidate)", bp, bp, 0.0)
x4 = np.arange(4) - 1.5; x8 = np.arange(8) - 3.5; d4 = x4 @ x4; d8 = x8 @ x8
p1 = lambda: (float(x4 @ loc[:4][::-1]) / d4, float(x8 @ loc[::-1]) / d8)
rec("features.slope", "closed-form least squares", tm(p1), bp, float(max(abs(p1()[0] - pv[0]), abs(p1()[1] - pv[1]))))

# ---------------- S9 notch bank sosfilt K biquads
for Kn in (4, 12, 24):
    sos = np.array([np.r_[signal.iirpeak(1000 + 200 * i, 20, fs)[0], signal.iirpeak(1000 + 200 * i, 20, fs)[1]] for i in range(Kn)])
    sos = np.array([np.r_[b, a] for b, a in [signal.iirpeak(1000 + 200 * i, 20, fs) for i in range(Kn)]])
    x = rng.standard_normal(B); zi = np.zeros((Kn, 2))
    s0 = lambda: signal.sosfilt(sos, x, zi=zi); bsos = tm(s0); ys = s0()[0]
    rec(f"notch.sosfilt K={Kn}", "scipy sosfilt (baseline)", bsos, bsos, 0.0)
    @njit(cache=True)
    def sos_nb(sos, x, zi):
        y = x.copy()
        for s in range(sos.shape[0]):
            b0, b1, b2, a0, a1, a2 = sos[s]
            z0, z1 = zi[s, 0], zi[s, 1]
            for n in range(y.shape[0]):
                xn = y[n]; yn = b0 * xn + z0
                z0 = b1 * xn - a1 * yn + z1; z1 = b2 * xn - a2 * yn
                y[n] = yn
            zi[s, 0] = z0; zi[s, 1] = z1
        return y
    zi2 = np.zeros((Kn, 2))
    rec(f"notch.sosfilt K={Kn}", "numba DF2T cascade, in-place zi", tm(lambda: sos_nb(sos, x, zi2.copy())), bsos, float(np.max(np.abs(sos_nb(sos, x, np.zeros((Kn, 2))) - ys))))

# ---------------- S10 engine buffering
from nullyn.engine import NullynEngine
mic = rng.standard_normal(B) * 0.01
im = np.zeros(0)
def buf0():
    global im
    im = np.concatenate([im, mic]); m = im[:B]; im = im[B:]; o = np.concatenate([np.zeros(0), m]); return o[:B]
bb = tm(buf0)
rec("engine.buffering", "concatenate FIFOs (baseline)", bb, bb)
rec("engine.buffering", "fast path n==B (no copies)", tm(lambda: mic), bb)

json.dump(dict(numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__, pyfftw=pyfftw.__version__, rows=rows),
          open("spikes.json", "w"), indent=1)
