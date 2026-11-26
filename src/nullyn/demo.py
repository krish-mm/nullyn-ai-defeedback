"""Simulate a single-mode acoustic feedback loop and show Nullyn AI suppressing the howl.

This is a pedagogical delay-line demonstration, not a venue simulation and not
a performance result. The published numbers in the paper come from the
pyroomacoustics study, which lives in a separate research repository.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .engine import NullynEngine
from .filters import peaking_eq


def _pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X))
    f[0] = 1
    x = np.fft.irfft(X / np.sqrt(f), n)
    return x / (np.std(x) + 1e-12)


def _path_filter(howl_hz, fs, boost_db=18.0, q=25.0):
    b, a = peaking_eq(howl_hz, boost_db, q, fs)
    # Plus a mild speaker-like band limit.
    sos_hp = signal.butter(2, 80, "hp", fs=fs, output="sos")
    sos_lp = signal.butter(4, min(0.45 * fs, 7000), "lp", fs=fs, output="sos")
    return b, a, np.vstack([sos_hp, sos_lp])


def _impulse_response(b, a, sos, delay, n_ir, fs):
    x = np.zeros(n_ir)
    x[delay] = 1.0
    y = signal.lfilter(b, a, x)
    y = signal.sosfilt(sos, y)
    return y


def _msg0_db(h, fs):
    _, H = signal.freqz(h, worN=1 << 16, fs=fs)
    return -20 * np.log10(np.max(np.abs(H)) + 1e-16)


def _howl_report(u, dry, fs, start_s=0.5, hold_s=0.5, thresh_db=12.0):
    """Oracle-style howl: a band of u sits thresh_db above dry for hold_s."""
    nfft = int(round(0.064 * fs))
    hop = max(1, nfft // 4)
    win = np.hanning(nfft)
    start = int(start_s * fs)
    frames_u, frames_d = [], []
    for i in range(start, min(len(u), len(dry)) - nfft + 1, hop):
        U = np.fft.rfft(u[i:i + nfft] * win)
        D = np.fft.rfft(dry[i:i + nfft] * win)
        frames_u.append(20 * np.log10(np.abs(U) + 1e-12))
        frames_d.append(20 * np.log10(np.abs(D) + 1e-12))
    if not frames_u:
        return dict(howl=False, howl_time_s=0.0, max_excess_db=0.0)
    U = np.stack(frames_u)
    D = np.stack(frames_d)
    excess = U - D
    over = excess > thresh_db
    # Longest consecutive run of frames where any bin is over threshold.
    any_over = over.any(axis=1)
    longest = cur = 0
    for flag in any_over:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    howl_time = longest * hop / fs
    return dict(
        howl=howl_time >= hold_s,
        howl_time_s=float(howl_time),
        max_excess_db=float(excess.max()) if excess.size else 0.0,
    )


def run_loop(source, fs, delay, b, a, sos, K, processor=None, use_ref=False):
    """Block-accurate delay-line loop: mic = source + path(speaker_delayed)."""
    B = processor.block if processor is not None else 64
    n = len(source) - len(source) % B
    source = source[:n]
    # Delay must exceed one block so the current block's speaker samples
    # do not reach the mic in the same block (same condition as the research sim).
    if delay < B:
        raise ValueError(f"loop delay ({delay}) must be >= block size ({B})")
    # Circular history of speaker samples; path starts after `delay` samples.
    hist = np.zeros(delay + n + B)
    zi_ba = signal.lfilter(b, a, np.zeros(8), zi=np.zeros(max(len(a), len(b)) - 1))[1]
    zi_sos = np.zeros((sos.shape[0], 2))
    out = np.zeros(n)
    dry = K * source
    if processor is not None:
        processor.reset()
    # Seed the delay line with silence; apply the path filter as samples emerge.
    # We filter the delayed speaker ( tail of hist ) in blocks.
    spk_hist = np.zeros(n + delay)
    for i in range(0, n, B):
        delayed = spk_hist[i:i + B]  # samples that were written at time i-delay
        filt, zi_ba = signal.lfilter(b, a, delayed, zi=zi_ba)
        fb, zi_sos = signal.sosfilt(sos, filt, zi=zi_sos)
        mic = source[i:i + B] + fb
        if processor is None:
            y = mic
        else:
            ref = spk_hist[delay + i - B:delay + i] if use_ref else None
            # First block has no previous speaker; pass zeros as the reference.
            if use_ref and (ref is None or len(ref) != B):
                ref = np.zeros(B)
            y = processor.process(mic, ref)
        spk = np.clip(K * y, -4.0, 4.0)  # soft safety; the research sim uses tanh
        spk_hist[delay + i:delay + i + B] = spk
        out[i:i + B] = spk
    return out, dry


def run_demo(seconds=4.0, fs=16000, howl_hz=1250.0, gain_db=6.0,
             delay_ms=12.0, out_wav=None, mode="auto", detector="nullyn_v0",
             seed=0, quiet=False):
    fs = int(fs)
    n = int(round(seconds * fs))
    delay = int(round(delay_ms * 1e-3 * fs))
    b, a, sos = _path_filter(howl_hz, fs)
    h = _impulse_response(b, a, sos, delay, int(0.25 * fs), fs)
    msg0 = _msg0_db(h, fs)
    K = 10 ** ((msg0 + gain_db) / 20)
    rng = np.random.default_rng(seed)
    # Low-level pink noise seeds the mode without sounding like a tone at t=0.
    source = 0.02 * _pink(n, rng)

    none_u, dry = run_loop(source, fs, delay, b, a, sos, K, processor=None)
    none = _howl_report(none_u, dry, fs)

    engine = NullynEngine(fs=fs, mode=mode, detector=detector, soundcheck_s=0.0)
    use_ref = mode in ("auto", "closedloop")
    proc_u, dry2 = run_loop(source, fs, delay, b, a, sos, K, processor=engine, use_ref=use_ref)
    proc = _howl_report(proc_u, dry2, fs)
    stats = engine.stats()

    if out_wav:
        import soundfile as sf
        # Two-channel: unprocessed speaker, processed speaker.
        sf.write(out_wav, np.stack([none_u, proc_u], axis=1), fs)

    ok = (not proc["howl"]) or (proc["howl_time_s"] < 0.5 * none["howl_time_s"])
    if not quiet:
        print("Nullyn AI feedback demo (synthetic delay-line loop, not a venue measurement)")
        print(f"  {seconds:.1f} s at {fs} Hz, mode around {howl_hz:.0f} Hz, "
              f"delay {delay_ms:.1f} ms, gain {gain_db:+.1f} dB re loop MSG0 ({msg0:.1f} dB)")
        print(f"  no processing : howl={none['howl']}  "
              f"howl_time={none['howl_time_s']:.2f}s  max excess={none['max_excess_db']:.1f} dB")
        print(f"  Nullyn AI     : howl={proc['howl']}  "
              f"howl_time={proc['howl_time_s']:.2f}s  max excess={proc['max_excess_db']:.1f} dB  "
              f"notches={stats.get('notches_created', 0)}")
        if ok:
            print("  result: howl was suppressed relative to the unprocessed loop.")
        else:
            print("  result: howl was NOT clearly suppressed — this is unexpected on this demo loop.")
        if out_wav:
            print(f"  wrote {out_wav} (ch1 unprocessed, ch2 Nullyn AI)")
    return dict(ok=ok, none=none, processed=proc, stats=stats, msg0_db=msg0, fs=fs, K=K)
