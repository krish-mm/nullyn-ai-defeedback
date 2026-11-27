# Performance: real-time path, complexity and latency budget

Status: **phase 1 baseline (v0.2.0)**. This page maps the per-block pipeline,
gives time and space complexity per stage, and records the measured baseline
that the `performance` issues work against. Every number here is offline
compute time on the machine named next to it. It is not analog-to-analog
latency, and it is not a Windows ASIO or macOS CoreAudio measurement.

Tools:

- `scripts/bench_latency.py`: seeded closed-loop latency benchmark (median, p95, p99, max, jitter, deadline misses, GC, transient allocation peak)
- `scripts/profile_hotspots.py`: per-stage timers, or `--cprofile` for a function-level profile
- `scripts/quality_gate.py`: replays the SIMULATED held-out steady protocol and compares howl flags, LSD and the output signal against a baseline (needs the private research workspace)

## 1. The per-block pipeline

`NullynEngine.process(mic, ref)` with a speaker-feed reference (closed loop):

```
callback block (N samples)
  └─ engine FIFO → AFC blocks of B samples (B = 4 ms: 192 @ 48 kHz, 64 @ 16 kHz)
       ├─ ClosedLoopAFC.process(mic_B)
       │    ├─ yhat_slow = IFFT( Σ_p X[p]·W[p] )          partitioned-block FDAF filter
       │    ├─ yhat_fast = IFFT( Σ_p X[p]·W_fast[p] )     v0.2 tracking filter
       │    ├─ ERLE / path-change / filter selection / safety gain (scalars)
       │    ├─ e = mic − γ·yhat
       │    ├─ adapt(W, e_slow) and adapt(W_fast, e_fast)   PEM-prewhitened, gradient-constrained NLMS
       │    ├─ every `lpc_every` blocks (every block during a path change):
       │    │     LPC (autocorrelation + Levinson) and LTP pitch predictor on the residual history
       │    └─ notch fallback: NullynV0.process(e)
       │          ├─ NotchBank.process: K_n cascaded biquads (sosfilt), zero added latency
       │          └─ every hop H samples: SideChain.analyze → candidates → TinyMLP → NotchBank.hit/tick
       └─ ClosedLoopAFC.push_ref(ref_B): FFT of the reference, prewhitening FIR, history shift
```

Without a reference (`mode="notch"`), only the notch fallback runs, on the
callback block directly, with no FIFO.

## 2. Symbols (values at 48 kHz, v0.2.0 defaults)

| Symbol | Meaning | 48 kHz | 16 kHz |
|---|---|---|---|
| N | callback block | 128 (CLI default) | — |
| B | AFC block (4 ms) | 192 | 64 |
| P | FDAF partitions (filter length P·B) | 64 (256 ms) | 64 |
| K | FFT bins per partition, B + 1 (FFT size 2B) | 193 (384) | 65 (128) |
| p | LPC order | 16 | 16 |
| N_w | LPC window | 3072 | 1024 |
| L_c, L_r | LTP correlation length, lag range | 768, 840 | 256, 280 |
| \|A_f\| | prewhitening FIR length (LPC ⊛ LTP) | ≤ 977 | ≤ 337 |
| N_f, H | side-chain FFT size, hop | 3072, 768 (every 4 AFC blocks) | 1024, 256 |
| N_b | side-chain bins N_f/2 + 1 | 1537 | 513 |
| C | candidates per frame | ≤ 5 | ≤ 5 |
| M | side-chain history frames | 64 | 64 |
| K_n | active notches | ≤ 24 | ≤ 24 |
| F | filters adapted (slow + fast) | 2 | 2 |

## 3. Time and space complexity per stage (v0.2.0 code as written)

Per AFC block unless noted. "Measured" is the share of `process()` time from
`scripts/profile_hotspots.py` (48 kHz, seeded closed-loop scenarios; the
instrumentation adds overhead, so read the shares, not the absolute values).

| Stage | Time | Persistent space | Transient per call | Measured share @48k |
|---|---|---|---|---|
| Engine FIFO (`np.concatenate` in/out) | O(N + B) copies | O(B) | O(N + B) | small |
| `push_ref`: ref FFT, prewhiten FIR, shift X/Xw/Xp(/Xp0) | O(B log B + B·\|A_f\| + P·K) | X, Xw: 2·P·K complex; Xp, Xp0: 2·P·K real | O(B) | ~6% |
| `_yhat` ×F | O(F·(P·K + B log B)) | — | O(P·K) temp product | ~6% |
| Path-change logic, ERLE, γ | O(B) | O(1) | O(B) | small |
| `_adapt` ×F: prewhiten FIR, error FFT, normaliser Σ_p Xp, gradient, **P IFFTs + P FFTs** (gradient constraint) | O(F·(B·\|A_f\| + P·K + P·B log B)) | W, W_fast: 2·P·K complex | O(P·B) real + O(P·K) complex, several copies | ~34% |
| `_update_lpc` (every 4th block): autocorrelation via `np.correlate(w, w, "full")` | **O(N_w²)** in v0.2.0 (all 2N_w−1 lags computed, p+1 used); O(N_w·p) from 0.2.1 (#7) | O(N_w) | O(N_w) | ~20% (incl. LTP) |
| … Levinson (`solve_toeplitz`) | O(p²) | O(p) | O(p) | |
| … LTP: residual + cross-correlation | O(N_w·p + L_c·L_r) | O(\|A_f\|) | O(L_c + L_r) | |
| `SideChain.analyze` (every H samples): FFT, log power, peak picking, run tracking (`np.roll` ×6) | O(N_f log N_f + N_b) | hist, fr_hist: 2·M·N_b float; pk_hist: M·N_b bool | O(N_b) several arrays | ~21% |
| … `_features` per candidate (Python loops, 2× `np.polyfit`) | O(C·M_used) with large constant | — | small | ~15% of total |
| TinyMLP (17-32-16-1) | O(C·1,072) MACs | 1,121 params | O(C·32) | small |
| `NotchBank.process` (sosfilt) | O(B·K_n) (+ O(K_n) coefficient rebuild when dirty) | O(K_n) | O(B) | ~6% |

**Amortised per AFC block** (dominant terms):

```
T_block ≈ F·P·B·log B                     (gradient constraint, every block)
        + (N_w² + N_w·p + L_c·L_r) / 4    (LPC/LTP, every 4th block — as written)
        + (N_f log N_f + C·c_feat) / 4    (side chain, every 4th block)
        + B·K_n                            (notch bank)
```

**Worst-case block** (the one that sets p95/p99): when the LPC update and a
side-chain frame land on the same block, which in v0.2.0 is every 4th block
at both rates, the cost is the sum of all three heavy terms. That is why
the distribution is bimodal: about 0.7 ms on most blocks, about 3 ms on one
block in four at 48 kHz.

**Total persistent state at 48 kHz** ≈ 4·P·K·16 B (complex) + 2·P·K·8 B
(real) ≈ 0.99 MB for the canceller, plus ≈ 1.67 MB of side-chain history:
about 2.7 MB in total, O(P·B + M·N_f).

## 4. Target complexity (what the issues move toward)

| Stage | Now | Target | How (issue) |
|---|---|---|---|
| LPC autocorrelation | O(N_w²) | O(N_w·p) | compute only p+1 lags |
| History shift | O(P·K) copy ×3–4 | O(K) | double-buffer ring, view |
| Normaliser Σ_p Xp | O(P·K) ×F | O(K) | running sum, computed once per block |
| Gradient constraint | O(F·P·B log B) every block | O(F·B log B) + amortised | partitioned/alternating constraint (algorithm change, gated) |
| Side-chain features | Python loops, polyfit | O(C) vectorised, closed-form slopes | |
| Engine FIFO | O(N + B) allocs | O(N) no allocs, zero-copy when N = B | |
| Worst-case block | LPC + side chain + AFC coincide | at most one heavy job per block | stagger schedule (gated) |

The README gets a complexity section once these land and the numbers are
re-measured.

## 5. Baseline measurement (v0.2.0)

`scripts/bench_latency.py`, seed 20261126, 4 simulated closed-loop scenarios
(static and walking, low and high gain) × 8 s, 0.5 s warm-up dropped.
Shared Linux box (Intel Xeon, 8 logical CPUs, Python 3.13.5, NumPy 2.5.5,
SciPy 1.18.1), 26 Nov 2026, 20:12–20:35 CT. The box runs other jobs, so the
max column is mostly scheduler noise; read median, p95 and p99. Three
48 kHz closed-loop runs are shown to give the spread.

| Mode | fs | Block | Budget | Median | p95 | p99 | Max | Jitter (sd) | Blocks > budget | Transient alloc peak |
|---|---|---|---|---|---|---|---|---|---|---|
| Closed loop | 48 kHz | 192 | 4.000 ms | 0.749–0.785 ms | 2.960–3.160 ms | 3.434–3.633 ms | 10.8–14.7 ms | 0.92–0.98 ms | 32–48 of 7,500 | 799 KiB |
| Closed loop | 16 kHz | 64 | 4.000 ms | 0.375 ms | 1.638 ms | 1.782 ms | 8.3 ms | 0.52 ms | 3 of 7,500 | 270 KiB |
| Notch only | 48 kHz | 192 | 4.000 ms | 0.082 ms | 1.140 ms | 1.347 ms | 8.1 ms | 0.41 ms | 2 of 7,500 | 52 KiB |

Raw results: `docs/perf/bench-v0.2.0.json`.

**Per-stage profile, 48 kHz** (`scripts/profile_hotspots.py`, 4 scenarios × 4 s;
the timers add overhead): canceller adapt (two filters) 33.5%, side-chain
analysis 20.6% (of which per-candidate features 15.4%), LPC/LTP update
20.3%, filter output (two filters) 6.2%, reference push 5.9%, notch bank
5.6%. Files: `docs/perf/stages-48k-v0.2.0.txt`, `docs/perf/stages-16k-v0.2.0.txt`.

**Added latency from the engine FIFO.** With a reference, the canceller runs
on 4 ms blocks. An impulse through `NullynEngine` at 48 kHz comes out
**128 samples (2.67 ms) late** with 64-, 128- (the `nullyn rt` default), or
256-sample callbacks, and **0 samples late** with 192-sample callbacks.
Fixing the default is issue #8.

**Quality baseline.** `scripts/quality_gate.py` on all 575 held-out steady
scenarios (2,875 runs: 3 matched gains + 2 high gains each) reproduces the
published v0.2.0 numbers exactly: 331 / 1,725 matched runs howled, 4 / 690
static, 327 / 1,035 walking (`docs/perf/quality-gate-v0.2.0-summary.json`).
SIMULATED, like every closed-loop number in this repository.

## 6. Library and technique spikes

Micro-benchmarks of one stage at a time at 48 kHz sizes (median of 300
calls), with the maximum absolute difference from the current code.
Script: `docs/perf/spikes.py` (needs numba and pyFFTW); output:
`docs/perf/spikes-2026-11-26.txt` and `docs/perf/spike_fir-2026-11-26.txt`.
numba 0.68.1, pyFFTW 0.15.1.

| Stage | Current | Candidate | Speed-up | Max diff | Verdict |
|---|---|---|---|---|---|
| LPC autocorrelation | `np.correlate` full, 787 µs | p+1 dot products, 15 µs | 52× | 0 | adopt (#7) |
| | | numba loops / FFT / sliding-window matvec | 26× / 11× / 20× | ≤ 3e-12 | dot products win |
| Levinson | `solve_toeplitz` 16.1 µs | numba Levinson-Durbin 0.9 µs | 17× | 7e-18 | adopt as optional (#13) |
| | | pure-NumPy Levinson | 0.2× | | reject |
| LTP cross-correlation | `np.correlate` valid 54 µs | scipy.fft + next_fast_len 43 µs | 1.27× | 7e-14 | small (#18) |
| | | `scipy.signal.correlate(fft)` / BLAS matvec | 0.6× / 0.16× | | reject |
| Canceller adapt (1 filter) | numpy.fft 171 µs | pyFFTW planned + aligned buffers 111 µs | 1.54× | 4e-18 | optional (#15) |
| | | complex64 / float32 | 1.70× | 2e-9 per update | needs gate (#16) |
| | | scipy.fft workers=1 / 4 | 1.01× / 1.01× | 0 | no gain |
| | | `rfft(n=2B)` + in-place W | 0.99× | 0 | allocation only (#11) |
| Adapt, slow + fast | two calls 333 µs | one batched (2,P,K) call | 0.38× | 0 | reject |
| History shift | copy P rows 8.2 µs | double-buffer ring 0.5 µs | 16× | 0 | adopt (#10) |
| Filter output | `np.sum(X*W)` 17.1 µs | einsum / numba | 0.54× / 0.99× | 7e-18 | keep |
| Side-chain FFT (3072) | numpy 15.8 µs | scipy.fft / pyFFTW | 1.08× / 1.41× | 4e-14 | (#19) |
| Log power | `abs(X)**2` 6.7 µs | `re²+im²` 6.1 µs | 1.09× | 7e-15 | (#19) |
| Feature slopes | 2× `np.polyfit` 31.3 µs | closed form 1.7 µs | 18× | 8e-17 | adopt (#9) |
| Notch cascade, 4 / 12 / 24 notches | sosfilt 19.0 / 22.0 / 26.2 µs | numba DF2T in place | 5.9× / 2.9× / 1.8× | 0 | optional (#13) |
| Prewhitening FIR, 977 taps | dense `np.convolve` 17.4 µs | LPC then 2-tap LTP | 1.46× (none at short lags) | 4e-16 | low priority (#22) |
| Engine FIFO | concatenate 1.4 µs | zero-copy when N = B | 24× | 0 | (#20) |
| GC during the timed loop | on: p99 3.45 ms | off 3.99 / freeze 3.87 ms | no gain on this box | — | re-test on a laptop (#17) |
| BLAS/OpenMP threads = 1 | p99 3.45 ms | 3.71 ms | no gain on this box | — | no change |

Tried and not installable here: nothing. All candidates in the brief
(NumPy in-place/preallocation, float32, SciPy `sosfilt`/`lfilter` state,
`scipy.fft` workers/`next_fast_len`, pyFFTW, numba `@njit(cache=True)`)
installed and were measured.

## 7. Reproduce

```bash
pip install -e ".[dev]"
python scripts/bench_latency.py --json bench.json          # ~5 min, add --quick for ~1 min
python scripts/profile_hotspots.py --fs 48000              # per-stage shares
python scripts/profile_hotspots.py --cprofile --top 30     # function-level
# quality gate (private research workspace + pyroomacoustics):
python scripts/quality_gate.py --research ../nullyn --count 575 --out base.json
python scripts/quality_gate.py --research ../nullyn --count 575 --out new.json --compare base.json
```

## 8. Progress log

| Step | Issue | Change | 48 kHz closed loop p95 / p99 (ms) | Quality gate |
|---|---|---|---|---|
| baseline | — | v0.2.0 | 2.96–3.16 / 3.43–3.63 | 331 / 1,725 matched howls (reference) |
| 1 | #7 | LPC autocorrelation: only p+1 lags, O(N_w²) → O(N_w·p) | 2.26–2.35 / 2.49–2.82 | bit-exact, 0 flips (`docs/perf/issue-7/`) |
| 2 | #8 | rt CLI default block = engine block, `latency='low'` | DSP unchanged; added engine delay 128 → 0 samples at the default | identical, 0 flips (`docs/perf/issue-8/`) |
