# Constant-phase pacer, the algorithm ported 1:1 to anti-lag.cpp (PhaseStep / PhaseRelease):
#  - anchor: the smoothed release grid (period P = the cap or flip rate x cadence).
#  - each game step (first pad read after a flip) waits until the next time with phase phi on that grid (no event);
#    within half a period after it: no wait (late).
#  - phi moves at most 0.5 ms per step toward the queue target: q (gpu done -> flip of each released frame, FlipQueue's
#    own stamps, EMA) -> q_target = margin + Q x P; Q = 1 (one frame of cushion) while a frame was late in the last 5 s,
#    0 after 10 s without. Start Q = 1.
# Game model: pad read, input -> GPU done W, its loop C to the next read, its own backpressure (k-3 released).
import math, random
def sim(name, P, vb, Wf, C, n=9000, pace=True, seed=7, margin=2.0):
    R = random.Random(seed)
    t_in=[]; rdy=[]; rel=[]; lat=[]
    anchor=None; phi=None; qe=None; Q=1; last_late=-1e9; q0=None
    for k in range(n):
        W = Wf(R)
        cand = t_in[-1] + C if t_in else 0.0
        if k >= 3: cand = max(cand, rel[k-3])
        t = cand
        if pace and anchor is not None and qe is not None:
            now = t
            if now - last_late < 5000: Q = 1
            elif now - last_late > 10000: Q = 0
            target = max(margin, vb + 1.0) + Q * P
            if phi is None: phi = (now - anchor) % P
            phi += max(-0.5, min(0.5, 0.1 * (qe - target)))
            x = (now - anchor - phi) % P          # time since the last phase point
            t = now + ((P - x) % P)
            if t_in: t = min(t, max(now, t_in[-1] + P + 1.0))   # a step never over P + 1 ms by our wait
        t_in.append(t); r0 = t + W; rdy.append(r0)
        prev = rel[-1] if rel else -1e9
        r = max(r0, prev + P - 1e-6)
        if vb: r = math.ceil(r / vb - 1e-9) * vb
        if rel and r - prev > P + max(vb / 2, 0.5): last_late = r
        rel.append(r)
        if anchor is None: anchor = r
        else:
            e = r - anchor; e -= round(e / P) * P; anchor += 0.1 * e
        q = r - r0; qe = q if qe is None else qe + (q - qe) * 0.2
        lat.append(r + 0.6 - t)
    a = n // 3
    def sd(v): m = sum(v)/len(v); return (sum((x-m)**2 for x in v)/len(v))**.5, m
    st = [t_in[i]-t_in[i-1] for i in range(a, n)]; sc = [rel[i]-rel[i-1] for i in range(a, n)]
    s1, _ = sd(st); s2, m2 = sd(sc); qv = [rel[i]-rdy[i] for i in range(a, n)]
    return (f"{name:30s} fps {1000/m2:5.2f}  game steps sd {s1:4.2f} (max {max(st):5.1f})  screen sd {s2:4.2f}  "
            f"input->present {sum(lat[a:])/len(lat[a:]):5.1f}  gpu done->flip {sum(qv)/len(qv):5.1f}  Q {Q}")
VB = 1000/360
w212 = lambda R: R.uniform(7, 14)
w212s = lambda R: R.uniform(7, 14) + (R.uniform(5, 15) if R.random() < 0.02 else 0)
w202 = lambda R: 28.8 + R.uniform(-1, 1)
w202s = lambda R: 28.8 + R.uniform(-1, 1) + (R.uniform(3, 10) if R.random() < 0.01 else 0)
for nm, P, vb, wf, C in (("212307 30x2 FSR3", 12*VB, VB, w212, 8.5), ("212307 + 2% spikes", 12*VB, VB, w212s, 8.5),
                         ("202407 game 30", 1000/30, 0.0, w202, 9.0), ("202407 + 1% spikes", 1000/30, 0.0, w202s, 9.0)):
    print(sim(nm + " before", P, vb, wf, C, pace=False)); print(sim(nm + " after", P, vb, wf, C))
