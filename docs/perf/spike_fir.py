import numpy as np, time
from scipy import signal
def tm(f, n=500):
    for _ in range(30): f()
    t=[]
    for _ in range(n):
        t0=time.perf_counter_ns(); f(); t.append(time.perf_counter_ns()-t0)
    return float(np.median(t))*1e-3
rng=np.random.default_rng(1); B=192; A=np.r_[1.0, rng.standard_normal(16)*0.1]
for T in (0, 300, 960):
    if T:
        lt=np.zeros(T+1); lt[0]=1; lt[T]=-0.5; Af=np.convolve(A, lt)
    else: Af=A
    SMAX=16+960+16; st=rng.standard_normal(SMAX); x=rng.standard_normal(B)
    def base():
        z=np.concatenate([st,x]); n=len(Af)-1
        return np.convolve(z[len(st)-n:], Af, "valid")
    y0=base(); b=tm(base)
    def casc():
        z=np.concatenate([st,x])
        p=len(A)-1
        if T:
            # r = A * z over the last B+T samples, then y = r[n] - bT * r[n-T]
            r=np.convolve(z[len(st)-p-T:], A, "valid")
            return r[T:] - 0.5*r[:-T]
        return np.convolve(z[len(st)-p:], A, "valid")
    y1=casc()
    print(f"prewhiten FIR |Af|={len(Af):4d} (T={T:3d}): np.convolve {b:6.1f}us  sparse cascade A then LTP {tm(casc):6.1f}us  {b/tm(casc):5.2f}x  max err {np.max(np.abs(y1-y0)):.1e}")
