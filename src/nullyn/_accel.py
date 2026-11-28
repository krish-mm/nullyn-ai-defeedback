"""Optional numba accelerators for Levinson-Durbin and the notch cascade.

Numba is an optional extra (``pip install nullyn[fast]``). When it is not
installed, every public function here is a thin wrapper around the same SciPy
routines the package used before this module existed. When it is installed,
the implementations are compiled with ``cache=True`` and without ``fastmath``,
and :func:`warmup` must be called once at construction so the first audio
callback does not pay JIT time.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz

HAS_NUMBA = False
_NUMBA = None

# Set NULLYN_NUMBA=0 to force the SciPy path even when numba is installed.
if os.environ.get("NULLYN_NUMBA", "1") not in ("0", "false", "False"):
    try:
        from numba import njit as _NUMBA
        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        _NUMBA = None


def levinson_scipy(r):
    """LPC coefficients a[1:] from autocorrelation ``r`` of length p+1."""
    r = np.asarray(r, dtype=float).reshape(-1)
    return solve_toeplitz(r[:-1], -r[1:])


def sosfilt_scipy(sos, x, zi):
    """DF2T biquad cascade; same contract as ``scipy.signal.sosfilt``."""
    return signal.sosfilt(sos, x, zi=zi)


if HAS_NUMBA:
    @_NUMBA(cache=True, fastmath=False)
    def _levinson_nb(r):
        p = r.shape[0] - 1
        a = np.zeros(p + 1)
        a[0] = 1.0
        e = r[0]
        tmp = np.zeros(p + 1)
        for i in range(1, p + 1):
            acc = r[i]
            for j in range(1, i):
                acc += a[j] * r[i - j]
            k = -acc / e
            tmp[:i + 1] = a[:i + 1]
            for j in range(1, i):
                a[j] = tmp[j] + k * tmp[i - j]
            a[i] = k
            e *= (1.0 - k * k)
        return a[1:]

    @_NUMBA(cache=True, fastmath=False)
    def _sosfilt_nb(sos, x, zi):
        y = x.copy()
        for s in range(sos.shape[0]):
            b0 = sos[s, 0]
            b1 = sos[s, 1]
            b2 = sos[s, 2]
            a0 = sos[s, 3]
            a1 = sos[s, 4]
            a2 = sos[s, 5]
            if a0 != 1.0:
                inv = 1.0 / a0
                b0 *= inv
                b1 *= inv
                b2 *= inv
                a1 *= inv
                a2 *= inv
            z0 = zi[s, 0]
            z1 = zi[s, 1]
            for n in range(y.shape[0]):
                xn = y[n]
                yn = b0 * xn + z0
                z0 = b1 * xn - a1 * yn + z1
                z1 = b2 * xn - a2 * yn
                y[n] = yn
            zi[s, 0] = z0
            zi[s, 1] = z1
        return y
else:
    _levinson_nb = None
    _sosfilt_nb = None


def levinson(r):
    """LPC coefficients a[1:] from autocorrelation ``r`` of length p+1."""
    r = np.ascontiguousarray(r, dtype=np.float64).reshape(-1)
    if HAS_NUMBA:
        return _levinson_nb(r)
    return levinson_scipy(r)


def sosfilt(sos, x, zi):
    """DF2T biquad cascade with state ``zi``, matching ``scipy.signal.sosfilt``."""
    if not HAS_NUMBA:
        return sosfilt_scipy(sos, x, zi)
    sos = np.ascontiguousarray(sos, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    z = np.array(zi, dtype=np.float64, copy=True, order="C")
    y = _sosfilt_nb(sos, x, z)
    return y, z


def warmup():
    """Compile the numba kernels (if any) so the first callback is not a JIT."""
    if not HAS_NUMBA:
        return
    r = np.zeros(17, dtype=np.float64)
    r[0] = 1.0001
    r[1] = 0.5
    _levinson_nb(r)
    sos = np.zeros((4, 6), dtype=np.float64)
    sos[:, 0] = 1.0
    sos[:, 3] = 1.0
    x = np.zeros(192, dtype=np.float64)
    zi = np.zeros((4, 2), dtype=np.float64)
    _sosfilt_nb(sos, x, zi)
