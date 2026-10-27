#!/usr/bin/env python3
"""Mouse -> right stick -> Demon's Souls camera -> warp -> (FSR 3 FG x2) -> 240 Hz display.

Simulates the user's settings (KYTY_KBM_SENS 1.00, KYTY_KBM_ANTIDZ 0.10, game camera speed 5 so a ~90 deg/s top
rate, no kbm_curve.txt = the linear mapping, KYTY_KBM_CARRY_MS 200, KYTY_KBM_WARP=1) at 30 and 60 game fps, with and
without FG x2, on a 240 Hz display, for the old (combo12-1024) and the new (mouse2-1025) code.

What is measured, at every 240 Hz vblank, is the camera angle on screen (game camera + warp shift):
  judder  = RMS of the shown angle minus its own 40 ms centred moving average (px at 2560 wide, 70 deg hfov):
            the frame-to-frame back-and-forth, independent of latency;
  back    = shown updates that step against the hand's direction during the steady pan (count / updates);
  lag     = the delay (ms) that best aligns the shown angle with the hand's angle (cross-correlation).
"""
import math
import random
import sys

SENS, ADZ, GAME_DZ, RMAX, FULL = 1.0, 0.10, 0.12, 90.0, 2000.0
CARRY_MS = 200.0
DPC = 0.05  # KYTY_KBM_DEG_PER_COUNT default (the old warp's linear-mode degrees per count)
G_TRUE = RMAX * SENS / FULL  # deg per count the game really turns (linear stick, anti-deadzone ~ its dead zone)
PX_PER_DEG = (1280.0 / math.tan(math.radians(35.0))) * math.pi / 180.0
VBLANK = 1.0 / 240.0
SIM_IRR = []


def hand_velocity(t):
    """counts/s: a pan with a human's uneven speed, a slow aim, a reversal, a stop."""
    if t < 0.3:
        return 0.0
    if t < 1.8:  # steady-ish pan, ~27 deg/s on screen, with 7 Hz tremor and slow speed changes
        return 600.0 + 120.0 * math.sin(2 * math.pi * 1.3 * t) + 60.0 * math.sin(2 * math.pi * 7.0 * t)
    if t < 2.6:  # slow aim (a few counts per frame: quantization matters)
        return 90.0 + 40.0 * math.sin(2 * math.pi * 2.0 * t)
    if t < 3.2:  # flick back, past the game's top speed (saturation and carry)
        return -2600.0
    return 0.0


def mouse_counts(duration, seed):
    """1000 Hz raw input: integer counts per ms (the hand's angle in counts at each ms too)."""
    rnd = random.Random(seed)
    counts, hand = [], []
    frac, pos = 0.0, 0.0
    for i in range(int(duration * 1000)):
        t = i / 1000.0
        v = hand_velocity(t)
        frac += v / 1000.0
        c = int(math.floor(frac + 0.5)) if frac >= 0 else -int(math.floor(-frac + 0.5))
        frac -= c
        # USB polling jitter: now and then a count arrives one poll late
        counts.append(c)
        pos += v / 1000.0
        hand.append(pos)
    return counts, hand


def game_rate(stick):
    m = abs(stick)
    if m <= GAME_DZ:
        return 0.0
    return math.copysign(RMAX * min(1.0, (m - GAME_DZ) / (1.0 - GAME_DZ)), stick)


class Old:
    """combo12-1024 linear mapping + KYTY_KBM_WARP shift = unread counts * dpc."""

    name = "old"

    def __init__(self):
        self.acc, self.last, self.out = 0.0, None, 0.0

    def read(self, t, new_frame):
        dt = 1.0 / 60.0 if self.last is None else t - self.last
        dt = min(max(dt, 0.004), 0.1)
        self.last = t
        mag = abs(self.acc) / dt * SENS / FULL
        lim = 1.0 + CARRY_MS * 1e-3 / dt
        if mag > lim:
            self.acc *= lim / mag
            mag = lim
        if mag > 0.0:
            f = 1.0 / mag if mag > 1.0 else 1.0
            used = self.acc * f
            self.acc -= used
            m = min(mag, 1.0)
            self.out = math.copysign(ADZ + (1.0 - ADZ) * m, used)
        else:
            self.out = 0.0  # KYTY_KBM_DECAY_MS 0: centred the read the mouse has nothing
        return self.out

    def stamp(self):
        return 0

    def warp_deg(self, stamp):
        return self.acc * DPC


class New:
    """mouse2-1025: one stick per game frame (a second read in the same frame repeats it), the counts drained by a
    sub-frame exponential (KYTY_KBM_SMOOTH_MS), the warp = the learned degrees per count * (counts not read + counts the
    reads after the presented frame's flip sent, which the game has applied but not shown)."""

    name = "new"

    def __init__(self, tau_ms=8.0, gain_err=0.0, behind=0):
        self.acc, self.last, self.out = 0.0, None, 0.0
        self.tau = tau_ms * 1e-3
        self.seq = 0
        self.sent = {}  # seq -> counts
        self.g = G_TRUE * (1.0 + gain_err)
        self.behind = behind

    def read(self, t, new_frame):
        if not new_frame and self.last is not None:
            return self.out  # the same frame: the same stick, nothing consumed
        dt = 1.0 / 60.0 if self.last is None else t - self.last
        dt = min(max(dt, 0.004), 0.1)
        self.last = t
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        used = self.acc * a
        mag = abs(used) / dt * SENS / FULL
        if mag > 1.0:
            used /= mag
            mag = 1.0
        self.acc -= used
        lim = CARRY_MS * 1e-3 * FULL / SENS
        if abs(self.acc) > lim:
            self.acc = math.copysign(lim, self.acc)
        if mag < 0.005:
            used, self.out = 0.0, 0.0
            if abs(self.acc) < 2.0:
                self.acc = 0.0  # a hand at rest: the last fraction of a count dropped
        else:
            self.out = math.copysign(ADZ + (1.0 - ADZ) * mag, used)
        self.seq += 1
        self.sent[self.seq] = used
        return self.out

    def stamp(self):
        return self.seq

    def warp_deg(self, stamp):
        inflight = sum(c for s, c in self.sent.items() if s > stamp - self.behind)
        return self.g * (self.acc + inflight)


def simulate(model_factory, fps, fg, warp, seed=1, double_read=0.25, duration=3.6):
    rnd = random.Random(seed * 7 + fps)
    model = model_factory()
    counts, hand = mouse_counts(duration, seed)
    T = 1.0 / fps
    events = []
    t = 0.0
    fid = 0
    while t < duration - 0.1:
        dt_frame = T + rnd.gauss(0.0, 0.0012)
        events.append((t, 0, "read", fid, True))
        if rnd.random() < double_read:
            events.append((t + 0.00005, 1, "read", fid, False))
        events.append((t + 0.8 * T, 2, "flip", fid, None))
        events.append((t + 0.8 * T + 0.034 + rnd.gauss(0.0, 0.0013), 3, "present", fid, None))
        t += dt_frame
        fid += 1
    events.sort(key=lambda e: (e[0], e[1]))
    ms_done = 0
    reads = {}  # fid -> (time, stick)
    per_frame = {}  # fid -> every stick value a read of that frame got
    flips = {}  # fid -> (camera, stamp)
    cam = 0.0
    prev_read_t = None
    shown_real = []  # (present_time, angle_deg)
    for et, _, kind, f, new in events:
        while ms_done < len(counts) and ms_done / 1000.0 <= et:
            model.acc += counts[ms_done]
            ms_done += 1
        if kind == "read":
            s = model.read(et, new)
            if new:
                reads[f] = [et, s]
                per_frame[f] = [s]
            else:
                reads[f][1] = s
                per_frame[f].append(s)
        elif kind == "flip":
            rt, s = reads[f]
            dt_g = rt - reads[f - 1][0] if f - 1 in reads else T
            cam += game_rate(s) * dt_g
            flips[f] = (cam, model.stamp())
        else:
            c, stamp = flips[f]
            shift = model.warp_deg(stamp) if warp else 0.0
            # the warp's clamp: KYTY_KBM_WARP_MAX 6% of the width (~4.2 deg at 70 deg hfov) and the edge margin
            shift = max(-4.0, min(4.0, shift))
            shown_real.append((et, c + shift))
    # what the display shows at each vblank
    if fg:
        # FSR 3 FG x2: the real frame k is shown half a frame late, the generated (k-1, k) midpoint first
        shown = []
        for i in range(1, len(shown_real)):
            p0, a0 = shown_real[i - 1]
            p1, a1 = shown_real[i]
            half = (p1 - p0) / 2.0
            shown.append((p1, 0.5 * (a0 + a1)))
            shown.append((p1 + half, a1))
    else:
        shown = shown_real
    vb = []
    j = 0
    cur = 0.0
    k = 0
    tv = 0.0
    while tv < duration - 0.15:
        while j < len(shown) and shown[j][0] <= tv:
            cur = shown[j][1]
            j += 1
        vb.append((tv, cur))
        tv += VBLANK
    differ = sum(1 for v in per_frame.values() if len(v) > 1 and max(v) != min(v))
    multi = sum(1 for v in per_frame.values() if len(v) > 1)
    # step irregularity: each shown update's step vs its neighbours' (pan only), % of the mean step
    steps = [(shown[i][0], shown[i][1] - shown[i - 1][1]) for i in range(1, len(shown))]
    pan = [d for t, d in steps if 0.5 <= t <= 1.8]
    irr = 0.0
    if len(pan) > 5:
        e = [(pan[i] - sum(pan[i - 2:i + 3]) / 5.0) ** 2 for i in range(2, len(pan) - 2)]
        irr = 100.0 * math.sqrt(sum(e) / len(e)) / max(1e-9, sum(abs(d) for d in pan) / len(pan))
    SIM_IRR.append(irr)
    return vb, hand, differ, multi


def metrics(vb, hand, t0=0.5, t1=1.8):
    n = int(round(0.040 / VBLANK)) | 1
    h = n // 2
    angles = [a for _, a in vb]
    sq, cnt = 0.0, 0
    back, ups = 0, 0
    for i in range(h, len(vb) - h):
        t = vb[i][0]
        if not (t0 <= t <= t1):
            continue
        avg = sum(angles[i - h:i + h + 1]) / n
        sq += (angles[i] - avg) ** 2
        cnt += 1
        d = angles[i] - angles[i - 1]
        if abs(d) > 1e-9:
            ups += 1
            if d < 0:  # the pan is to +x
                back += 1
    judder_px = math.sqrt(sq / max(cnt, 1)) * PX_PER_DEG
    # lag: the hand's angle (deg the game would turn) vs the shown angle, over the pan
    best, best_lag = None, 0
    for lag_ms in range(0, 160, 2):
        e = 0.0
        m = 0
        for t, a in vb:
            if t0 <= t <= t1 + 0.5:
                ti = int((t - lag_ms / 1000.0) * 1000)
                if 0 <= ti < len(hand):
                    e += (a - G_TRUE * hand[ti] - (vb[0][1])) ** 2
                    m += 1
        # remove the constant offset: compare increments over the window instead
        if best is None or e < best:
            best, best_lag = e, lag_ms
    return judder_px, back, ups, best_lag


def lag_by_increments(vb, hand, t0=0.5, t1=3.4):
    best, best_lag = None, 0
    for lag_ms in range(0, 200, 2):
        e = 0.0
        for i in range(1, len(vb)):
            t = vb[i][0]
            if t0 <= t <= t1:
                ti = int(round((t - lag_ms / 1000.0) * 1000))
                tj = int(round((vb[i - 6][0] - lag_ms / 1000.0) * 1000)) if i >= 6 else ti
                if 0 <= tj < len(hand) and 0 <= ti < len(hand):
                    want = G_TRUE * (hand[ti] - hand[tj])
                    got = vb[i][1] - vb[i - 6][1]
                    e += (want - got) ** 2
        if best is None or e < best:
            best, best_lag = e, lag_ms
    return best_lag


def main():
    rows = []
    cases = [
        ("old", lambda: Old(), True),
        ("old, warp off", lambda: Old(), False),
        ("new, warp on", lambda: New(), True),
        ("new, warp on, gain 20% off", lambda: New(gain_err=0.2), True),
                ("new, warp off", lambda: New(), False),
    ]
    print(f"{'case':28s} {'fps':>3s} {'FG':>3s} {'step irregularity':>18s} {'back-steps':>11s} {'lag ms':>7s} {'reads differ in a frame':>24s}")
    for fps in (30, 60):
        for fg in (False, True):
            for name, fac, warp in cases:
                if fg and "behind" in name:
                    continue
                js, bs, us, ls = [], [], [], []
                jslow = []
                ds, ms = [], []
                SIM_IRR.clear()
                for seed in range(1, 6):
                    vb, hand, dif, mul = simulate(fac, fps, fg, warp, seed=seed)
                    ds.append(dif); ms.append(mul)
                    j, b, u, _ = metrics(vb, hand)
                    js.append(j)
                    bs.append(b)
                    us.append(u)
                    jslow.append(metrics(vb, hand, 1.9, 2.5)[0])
                    ls.append(lag_by_increments(vb, hand))
                label = name
                print(f"{label:28s} {fps:3d} {'x2' if fg else 'off':>3s} {sum(SIM_IRR)/len(SIM_IRR):16.0f} % {sum(bs):4d}/{sum(us):<5d} {sum(ls)/len(ls):7.0f} {sum(ds):14d} of {sum(ms):<3d}")
        print()


if __name__ == "__main__":
    main()
