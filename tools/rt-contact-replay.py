#!/usr/bin/env python3
# Deterministic offline replay of rt-contact.comp's temporal + composite (modes 1, 3) on a still camera:
# static scene, the upscaler's jitter flipping a ray's binary hit, the RT sun pass on 62 of 500 frames
# (HWRTCONTACT log 20261027-200029-085). Prints the RTFLICKER metric (% of touched points whose light
# changed > 10 % frame to frame) for the old (6991c474) and new (ds-rt-stable-1027) code. Exit 1 if new >= 1 %.
import random
def run(new, frames=500, pixels=4000, seed=1):
    rnd = random.Random(seed); exact = set(rnd.sample(range(frames), 62))
    K, gain, maxd = 1.5, 1.0, 0.6
    px = [(rnd.random() < 0.5, rnd.random() * 0.6, rnd.random()) for _ in range(pixels)]  # shadowed?, p(hit flips), base occ
    hist = [0.0] * pixels; n = [0.0] * pixels; held = [None] * pixels; held_w = 0.0; held_ok = False
    prev = [None] * pixels; changed = touched = 0
    for f in range(frames):
        ex = f in exact; held_ok |= ex; held_w = min(1.0, held_w + 1 / 60) if (new and held_ok) else held_w
        for i, (shad, pf, occ0) in enumerate(px):
            cur = occ0 if rnd.random() > pf else (1.0 if occ0 < 0.5 else 0.0)   # jitter-flipped binary hit
            nn = min(n[i] + 1, 64 if new else 24); a = max(1 / nn, 1 / 64 if new else 0.08)
            hist[i] = cur if n[i] == 0 else hist[i] + (cur - hist[i]) * a; n[i] = nn
            Vs = 0.2 if shad else 1.0
            if new:
                if ex: held[i] = Vs
                V = 1.0 if held[i] is None else 1.0 + (held[i] - 1.0) * held_w
            else:
                V = Vs if ex else 1.0
            os_ = min(max(hist[i] * gain, 0), 1); fs = (1 + K * min(V, 1 - os_)) / (1 + K * V)
            m = min(max(fs, 1 - maxd), 1.0)
            if f >= 120 and prev[i] is not None and (m < 0.98 or prev[i] < 0.98):
                touched += 1; changed += abs(m - prev[i]) / max(prev[i], 1e-3) > 0.10
            prev[i] = m
    return 100.0 * changed / max(touched, 1)
old, new = run(False), run(True)
print(f"RTFLICKER replay contact shadows: old {old:.3f} %  new {new:.3f} %  (target < 1 %)")
raise SystemExit(0 if new < 1.0 else 1)
