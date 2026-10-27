#!/usr/bin/env python3
"""(10-25) Mouse flick / pan -> right stick -> Demon's Souls camera at 60 game fps: the "holgura" (turning on after the
hand stops) and the delay, old (combo14-1025: camera speed 5, 8 ms smoothing, 200 ms carry, counts/s * sens / 2000)
vs new (mouse3-1025: camera speed raised to 10, 2 ms smoothing, one frame of carry, linearized to 0.05 deg/count *
sens with the learned top rate).

The game: a full stick turns 18 deg/s per camera speed step (5: 90 deg/s), dead zone 0.12, stick read once a frame.
"""
import math

GAME_DZ, ADZ, FPS = 0.12, 0.10, 60.0


def game_rate(m, rmax):
    m = abs(m)
    return 0.0 if m <= GAME_DZ else rmax * min(1.0, (m - GAME_DZ) / (1.0 - GAME_DZ))


def hand(kind):
    """counts per ms over 1.2 s."""
    out = []
    for i in range(1200):
        t = i / 1000.0
        if kind == "flick":
            v = 6000.0 if 0.1 <= t < 0.22 else 0.0  # 720 counts in 120 ms
        else:
            v = 900.0 if 0.1 <= t < 0.7 else 0.0  # a pan
        out.append(v / 1000.0)
    return out


def run(kind, old, sens, speed):
    rmax = 18.0 * speed
    dpc = 0.05 * sens
    smooth = 0.008 if old else 0.002
    counts = hand(kind)
    acc, cam, last = 0.0, 0.0, None
    T = 1.0 / FPS
    path = []
    ms = 0
    t = 0.0
    while t < 1.2:
        while ms < len(counts) and ms / 1000.0 <= t:
            acc += counts[ms]
            ms += 1
        dt = T
        used = acc * (1.0 - math.exp(-dt / smooth))
        if old:
            mag = abs(used) / dt * sens / 2000.0
            cpf = 2000.0 / sens
            lim = 0.2 * cpf
        else:
            cpf = rmax / dpc
            mag = abs(used) / dt / cpf
            lim = dt * cpf
        if mag > 1.0:
            used /= mag
            mag = 1.0
        acc -= used
        if abs(acc) > lim:
            acc = math.copysign(lim, acc)
        stick = 0.0 if mag < 0.005 else ADZ + (1.0 - ADZ) * mag
        cam += game_rate(stick, rmax) * dt
        path.append((t + T, cam))  # shown a frame later
        t += T
    hand_end = 0.22 if kind == "flick" else 0.7
    total = path[-1][1]
    after = total - max(c for tt, c in path if tt <= hand_end + T)
    moving = [tt for (tt, c), (_, c0) in zip(path[1:], path[:-1]) if c - c0 > 1e-6]
    tail = max(0.0, (moving[-1] - hand_end) * 1000.0) if moving else 0.0
    half = next(tt for tt, c in path if c >= 0.5 * total) * 1000.0 if total > 0 else 0.0
    wanted = sum(counts) * (0.05 * sens if not old else 18.0 * 5 * sens / 2000.0)
    return total, wanted, after, tail, half - 100.0


print(f"{'case':38s} {'move':6s} {'turned':>7s} {'hand':>6s} {'after stop':>11s} {'keeps turning':>14s} {'to 50%':>7s}")
for kind in ("flick", "pan"):
    for name, old, sens, speed in [
        ("old: speed 5, sens 1.0", True, 1.0, 5),
        ("old: speed 5, sens 2.0", True, 2.0, 5),
        ("new: speed 10 (auto), sens 1.0", False, 1.0, 10),
        ("new: speed 10 (auto), sens 2.0", False, 2.0, 10),
        ("new: speed 15, sens 2.0", False, 2.0, 15),
    ]:
        tot, want, after, tail, half = run(kind, old, sens, speed)
        print(f"{name:38s} {kind:6s} {tot:6.1f}d {want:5.1f}d {after:9.1f} d {tail:11.0f} ms {half:5.0f} ms")
