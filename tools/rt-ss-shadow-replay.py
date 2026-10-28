#!/usr/bin/env python3
# Offline replay of src/local/cshadow.comp (KYTY_RT_SHADOWS, the screen-space sun shadows: the default sun path again
# since ds-ssrt-default-1028; KYTY_RT_REAL=1 selects the hardware-RT path) on a KYTY_RT_CAPTURE set.
# Modes 0 (trace), 1 (temporal), 2 (a-trous 1, 2, 4 at level 2), 4 (composite), frame by frame on the capture's still
# camera (static depth / colour, the per-frame IGN jitter of the shader), in numpy. Prints:
#   - the darkening (% of pixels with occlusion > 0.1, the mean colour change),
#   - RTFLICKER (as rt-flicker.cpp: % of touched points whose contribution changed > 10 % of the pixel's light frame to
#     frame, camera still), after a warm-up,
#   - a stripe index (the share of the darkening's row / column profile power at periods 2-8 px; ~0 for no stripes).
# Usage: rt-ss-shadow-replay.py <capture>/still [--level 2] [--frames 48] [--light x y z (view space, toward the sun)]
#        [--png out.png]. Exit 1 when the darkening is zero, the flicker >= 1 % or the stripe index >= 0.05.
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("set_dir")
ap.add_argument("--level", type=int, default=2)
ap.add_argument("--frames", type=int, default=48)
ap.add_argument("--warmup", type=int, default=16)
# (the 20261028-001032 log, the capture's session: "the light in view space now (-0.397 0.701 -0.593)")
ap.add_argument("--light", type=float, nargs=3, default=[-0.397, 0.701, -0.593])
ap.add_argument("--strength", type=float, default=0.5)
ap.add_argument("--png", default="")
a = ap.parse_args()

fj = json.load(open(os.path.join(a.set_dir, "frame.json")))
W, H = fj["render"]
M = fj["view_to_clip_rows"]
def img(name):
    return np.fromfile(os.path.join(a.set_dir, name + ".bin"), dtype=np.float32).reshape(H, W, 4)
color = img("color")[..., :3].astype(np.float64)
depth = img("depth")[..., 0].astype(np.float64)
motion = img("motion")[..., :2].astype(np.float64)
ms = fj["motion_scale"]

P = (M[0], M[5], M[8], M[9]); DM = (M[10], M[14], M[11], M[15])
YS = -1.0  # row 0 = NDC y +1 (the log's calibration)
q = a.level
half = q < 2
TW, TH = (W // 2, H // 2) if half else (W, H)

def view_z(d):
    den = d * DM[2] - DM[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(den) > 1e-12, (DM[1] - d * DM[3]) / den, 1e6)
    return z
def is_sky(d):
    return (d <= 1e-7) | (d >= 0.9999999) | (np.abs(view_z(d)) > 2e3)
def view_pos(u, v, d):
    z = view_z(d); w = z * DM[2] + DM[3]
    nx = u * 2 - 1; ny = YS * (v * 2 - 1)
    return np.stack([(nx * w - z * P[2]) / P[0], (ny * w - z * P[3]) / P[1], z], -1)
def project(p):
    w = p[..., 2] * DM[2] + DM[3]
    nx = (p[..., 0] * P[0] + p[..., 2] * P[2]) / w; ny = (p[..., 1] * P[1] + p[..., 2] * P[3]) / w
    return nx * 0.5 + 0.5, YS * ny * 0.5 + 0.5
def fetch_pt(tex, u, v, w_, h_):  # nearest, clamp to edge
    x = np.clip(np.floor(u * w_).astype(np.int64), 0, w_ - 1); y = np.clip(np.floor(v * h_).astype(np.int64), 0, h_ - 1)
    return tex[y, x]
def fetch_lin(tex, u, v):
    h_, w_ = tex.shape[:2]
    x = u * w_ - 0.5; y = v * h_ - 0.5
    x0 = np.floor(x); y0 = np.floor(y); fx = x - x0; fy = y - y0
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    c = lambda xx, yy: tex[np.clip(yy, 0, h_ - 1), np.clip(xx, 0, w_ - 1)]
    if tex.ndim == 3: fx = fx[..., None]; fy = fy[..., None]
    return (c(x0, y0) * (1 - fx) * (1 - fy) + c(x0 + 1, y0) * fx * (1 - fy) + c(x0, y0 + 1) * (1 - fx) * fy + c(x0 + 1, y0 + 1) * fx * fy)
def dpt(u, v):
    return fetch_pt(depth, u, v, W, H)
def ign(px, py):
    return np.modf(52.9829189 * np.modf(px * 0.06711056 + py * 0.00583715)[0])[0] % 1.0
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-20)

# Trace-grid geometry (frame invariant).
gy, gx = np.mgrid[0:TH, 0:TW].astype(np.float64)
tu = (gx + 0.5) / TW; tv = (gy + 0.5) / TH
td = dpt(tu, tv); tsky = is_sky(td)
p = view_pos(tu, tv, td)
rs = (1.0 / W, 1.0 / H)
def vp_at(du, dv):
    return view_pos(tu + du, tv + dv, dpt(tu + du, tv + dv))
px_, nx_, py_, ny_ = vp_at(rs[0], 0), vp_at(-rs[0], 0), vp_at(0, rs[1]), vp_at(0, -rs[1])
dx = np.where((np.abs(px_[..., 2] - p[..., 2]) < np.abs(p[..., 2] - nx_[..., 2]))[..., None], px_ - p, p - nx_)
dy = np.where((np.abs(py_[..., 2] - p[..., 2]) < np.abs(p[..., 2] - ny_[..., 2]))[..., None], py_ - p, p - ny_)
n = np.cross(dx, dy); ln = np.linalg.norm(n, axis=-1, keepdims=True)
n = np.where(ln > 1e-20, n / np.maximum(ln, 1e-30), normalize(-p))
n = np.where((np.sum(n * -p, -1) < 0)[..., None], -n, n)
z = np.abs(p[..., 2])
gbuf = np.concatenate([n, np.where(tsky, 1e6, z)[..., None]], -1)
gbuf[tsky] = (0, 0, 1, 1e6)
l = np.array(a.light) / np.linalg.norm(a.light)
ndl = n @ l
nw = 1 - smoothstep(6.0, 40.0, z)
g0 = np.where(True, 0.3 + (0.15 - 0.3) * nw, 0); g1 = 0.6 + (0.4 - 0.6) * nw
az = lambda du, dv: np.abs(view_z(dpt(tu + du, tv + dv)))
tol = 0.01 * z + 0.03
bad = (np.abs(az(rs[0], 0) + az(-rs[0], 0) - 2 * z) > tol) | (np.abs(az(0, rs[1]) + az(0, -rs[1]) - 2 * z) > tol)
g0 = np.where(bad, 2.0, g0); g1 = np.where(bad, 3.0, g1)
fade_m = 100.0
cand3 = (400.0 if q >= 3 else 280.0 if q >= 2 else 160.0, 1.0, fade_m, fade_m * 2)
wgt = smoothstep(g0, g1, ndl) * (1 - smoothstep(cand3[3], cand3[3] * 1.2, z))
length_m = 1.5
b1 = normalize(np.cross(l, np.array([0, 1, 0]) if abs(l[1]) < 0.9 else np.array([1, 0, 0]))); b2 = np.cross(l, b1)

def trace(frame):
    f = float(frame % 64)
    u1 = ign(gx + f * 7.123 + 13.0, gy + f * 7.123 + 13.0); u2 = ign(gy + f * 3.731 + 71.0, gx + f * 3.731 + 71.0)
    ang = 6.2831853 * u2; rad = 0.05 * np.sqrt(u1)
    lj = normalize(l + (b1[None, None] * np.cos(ang)[..., None] + b2[None, None] * np.sin(ang)[..., None]) * rad[..., None])
    steps = 48 if q >= 3 else 28 if q >= 2 else 16
    px_m = abs(P[1]) * 0.5 * H / np.maximum(z, 1e-3)
    cap = cand3[0] * H / 1440.0
    farx = smoothstep(60.0, max(cand3[3] * 0.5, 61.0), z) if cand3[3] > 121.0 else np.zeros_like(z)
    ln_ = np.minimum((1.25 if q >= 3 else 1.0) * length_m * (0.45 + 0.55 * nw) + farx * 0.03 * z, cap / px_m)
    foot = 1.0 / px_m
    thick = (np.clip(0.12 + 0.004 * z, 0.12, 0.3) + farx * foot) * cand3[1]
    bias = 0.004 * z + 0.01 + farx * foot * (0.5 + (1.0 - ndl))
    ndl_c = np.clip(ndl, -1, 1)
    t_min = (0.04 + 0.008 * z) / np.maximum(ndl_c, 0.3)
    jit = ign(gx + f * 5.588238, gy + f * 5.588238)
    o = p + n * (0.006 * z + 0.02)[..., None]
    t0 = 1.0 / px_m
    active = (wgt > 0) & ~tsky & (t0 < ln_)
    hits = np.zeros(z.shape, np.int32); res = np.zeros(z.shape); done = ~active
    for s in range(steps):
        k = (s + jit) / steps
        t = t0 + (ln_ - t0) * k * k
        qp = o + lj * t[..., None]
        u, v = project(qp)
        off = (u < 0.01) | (v < 0.01) | (u > 0.99) | (v > 0.99)
        done |= off
        sd = dpt(u, v); sky = is_sky(sd)
        oz = np.abs(view_z(sd)); delta = np.abs(qp[..., 2]) - oz
        brk = ~sky & (delta > 4 * thick)
        c = ~sky & (delta > bias) & (delta < thick) & (t > t_min)
        r2 = (1.5 / W, 1.5 / H)
        a0 = np.abs(view_z(dpt(u + r2[0], v))); a1 = np.abs(view_z(dpt(u - r2[0], v)))
        a2 = np.abs(view_z(dpt(u, v + r2[1]))); a3 = np.abs(view_z(dpt(u, v - r2[1])))
        tl = 0.02 * oz + 0.05
        c &= (np.abs(a0 + a1 - 2 * oz) < tl) & (np.abs(a2 + a3 - 2 * oz) < tl)
        live = ~done
        hits = np.where(live, np.where(c, hits + 1, 0), hits)
        got = live & (hits >= 2)
        e = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v))
        val = smoothstep(0, 0.04, e) * (1 - smoothstep(0.85, 1.0, k)) * (1 - smoothstep(0.5 * thick, thick, delta)) * \
              smoothstep(t_min, 2 * t_min, t) * (1.0 + (0.35 - 1.0) * np.clip(t / ln_, 0, 1))
        res = np.where(got, val, res)
        done |= got | (live & brk)
    occ = np.where(active, res * wgt, 0.0)
    v_ = 1 - occ
    v_[tsky] = 1.0
    return np.clip(v_, 0, 1)

def depth_slope(zz):
    sh = lambda dx_, dy_: np.pad(zz, 1, mode="edge")[1 + dy_:1 + dy_ + TH, 1 + dx_:1 + dx_ + TW]
    gxv = np.minimum(np.abs(sh(1, 0) - zz), np.abs(sh(-1, 0) - zz)); gyv = np.minimum(np.abs(sh(0, 1) - zz), np.abs(sh(0, -1) - zz))
    return np.minimum(gxv + gyv, 0.25 * zz)
def shift(arr, dx_, dy_):
    pad = ((1 * abs(dy_), 1 * abs(dy_)), (abs(dx_), abs(dx_))) + ((0, 0),) * (arr.ndim - 2)
    a_ = np.pad(arr, pad, mode="edge")
    return a_[abs(dy_) + dy_:abs(dy_) + dy_ + arr.shape[0], abs(dx_) + dx_:abs(dx_) + dx_ + arr.shape[1]]
gw = gbuf[..., 3]; slope = depth_slope(gw); far_k = smoothstep(10.0, 60.0, gw)
tmot = fetch_lin(motion, tu, tv)
puv_u = tu + tmot[..., 0] * ms[0] / W; puv_v = tv + tmot[..., 1] * ms[1] / H
pin = (puv_u >= 0) & (puv_u <= 1) & (puv_v >= 0) & (puv_v <= 1)
pg = fetch_lin(gbuf, puv_u, puv_v)

def temporal(cur, hist, count_prev, valid):
    m1 = np.zeros_like(cur); m2 = np.zeros_like(cur); wn = np.zeros_like(cur)
    ntol = 0.03 * gw + 2 * slope; ndot = 0.8 + (0.65 - 0.8) * far_k
    for yy in (-1, 0, 1):
        for xx in (-1, 0, 1):
            sg = shift(gbuf, xx, yy); sv = shift(cur, xx, yy)
            w_ = ((np.abs(sg[..., 3] - gw) < ntol) & (np.sum(sg[..., :3] * gbuf[..., :3], -1) > ndot)).astype(float)
            m1 += sv * w_; m2 += sv * sv * w_; wn += w_
    wn = np.maximum(wn, 1); m1 /= wn; m2 /= wn
    sigma = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    a_min = 0.08 if q >= 3 else 0.1
    count = np.ones_like(cur); res = 0.5 * (m1 + cur)
    if valid:
        ptol = 0.04 * gw + 1.5 * slope + 0.15
        ok = pin & (pg[..., 3] < 1e5) & (np.abs(pg[..., 3] - gw) < ptol) & (np.sum(pg[..., :3] * gbuf[..., :3], -1) > 0.9 + (0.7 - 0.9) * far_k)
        pc = fetch_lin(count_prev, puv_u, puv_v)
        cnt = np.minimum(np.maximum(pc, 1) + 1, 32.0)
        alpha = np.maximum(1 / cnt, a_min)
        kk = 1.0 + np.clip(cnt / 16.0, 0, 1)
        h = np.clip(fetch_lin(hist, puv_u, puv_v), m1 - kk * sigma - 0.03, m1 + kk * sigma + 0.03)
        res = np.where(ok, h + (cur - h) * alpha, res); count = np.where(ok, cnt, count)
    res = np.clip(res, 0, 1)
    sk = gw > 1e5
    res[sk] = 1; count[sk] = 0
    return res, count

kern = (0.375, 0.25, 0.0625)
def atrous(src, step):
    st = int(step)
    zden = np.maximum((gw * 0.015 + 1.5 * slope) * step, 1e-3)
    sv_den = 0.9 if step <= 2 else 0.5
    s_ = np.zeros_like(src); ws = np.zeros_like(src)
    for yy in range(-2, 3):
        for xx in range(-2, 3):
            sg = shift(gbuf, xx * st, yy * st); sc = shift(src, xx * st, yy * st)
            w_ = kern[abs(xx)] * kern[abs(yy)] * np.maximum(np.sum(sg[..., :3] * gbuf[..., :3], -1), 0) ** 16 * \
                 np.exp(-np.abs(sg[..., 3] - gw) / zden) * np.exp(-np.abs(sc - src) / sv_den)
            w_ = np.where(sg[..., 3] > 1e5, 0, w_)
            s_ += sc * w_; ws += w_
    r = np.where(ws > 1e-6, s_ / np.maximum(ws, 1e-30), src)
    r = np.where(gw > 1e5, src, r)
    return np.clip(r, 0, 1)

# Composite (render resolution == trace resolution at level >= 2).
rz = np.abs(view_z(depth)); rsky = is_sky(depth)
fade = np.where(rsky, 0.0, 1 - smoothstep(cand3[2], cand3[3], rz))
lum = color @ np.array([0.2126, 0.7152, 0.0722])
lref = lum.copy()
for (ox, oy) in ((6, 0), (-6, 0), (0, 6), (0, -6), (10, 10), (-10, 10), (10, -10), (-10, -10)):
    sz = shift(rz, ox, oy); sl = shift(lum, ox, oy)
    lref = np.where(np.abs(sz - rz) < 0.08 * rz + 0.2, np.maximum(lref, sl), lref)
inv_k = 0.5
rel = lum / np.maximum(0.85 * lref, 1e-8)
lit = smoothstep(0.55, 0.9, (rel - inv_k) / (1 - inv_k))
nearr = 0.6 + 0.4 * (1 - smoothstep(6.0, 60.0, rz))
def composite(vis, count):
    if half:
        vis = np.kron(vis, np.ones((2, 2)))[:H, :W]; count = np.kron(count, np.ones((2, 2)))[:H, :W]
    vis = np.clip(np.nan_to_num(vis, nan=1.0), 0, 1)
    occ = np.clip((1 - vis - 0.04) / 0.96, 0, 1)
    stable = 0.4 + 0.6 * smoothstep(1.0, 4.0, count)
    amt = a.strength * fade
    factor = 1 - np.clip(amt * nearr * stable * occ * lit, 0, 0.5)
    dark = np.clip(fade * stable * occ * lit, 0, 1)
    return color * np.minimum(factor, 1.0)[..., None], dark, factor

hist = np.ones((TH, TW)); cnt = np.zeros((TH, TW)); prev_contrib = None
changed = touched = 0; changed20 = 0; maxj = 0.0
for fr in range(a.frames):
    raw = trace(fr)
    tmp_a, cnt_new = temporal(raw, hist, cnt, fr > 0)
    h1 = atrous(tmp_a, 1.0)
    vis = atrous(h1, 2.0)
    if q >= 2: vis = atrous(vis, 4.0)
    if q >= 3: vis = atrous(vis, 8.0)
    hist = h1; cnt = cnt_new
    out, dark, factor = composite(vis, cnt)
    # RTFLICKER as rt-flicker.comp: the pass's contribution (out - in) vs the previous frame's, relative to the light.
    contrib = lum - out @ np.array([0.2126, 0.7152, 0.0722])
    if prev_contrib is not None and fr >= a.warmup:
        t_ = (contrib > 1e-3 * lum) | (prev_contrib > 1e-3 * lum)
        d_ = np.abs(contrib - prev_contrib) / np.maximum(lum, 1e-6)
        touched += int(t_.sum()); changed += int((t_ & (d_ > 0.10)).sum()); changed20 += int((t_ & (d_ > 0.20)).sum())
        if t_.any(): maxj = max(maxj, float(d_[t_].max()))
    prev_contrib = contrib

shadowed = 100.0 * float((dark > 0.1).mean())
geo = ~rsky
mean_change = 100.0 * float(1 - (factor[geo]).mean()) if geo.any() else 0.0
# Stripes: the darkening's column / row profiles (over the geometry), power at periods 2-8 px over the total (no DC).
def stripe(profile):
    pr = profile - profile.mean()
    if not np.any(pr): return 0.0
    sp = np.abs(np.fft.rfft(pr)) ** 2; fr_ = np.fft.rfftfreq(len(pr))
    hi = (fr_ >= 1 / 8) & (fr_ <= 0.5)
    return float(sp[hi].sum() / max(sp[1:].sum(), 1e-30))
dk = 1 - factor
rows = dk.sum(1) / np.maximum(geo.sum(1), 1); cols = dk.sum(0) / np.maximum(geo.sum(0), 1)
r_idx, c_idx = stripe(rows[geo.any(1)]), stripe(cols[geo.any(0)])
# Local stripes (the HWRT path's artefact: alternating rows / columns): on darkened pixels, the mean |second
# difference| across rows and across columns over the mean darkening.
dd = dk
sec_y = np.abs(dd[2:, :] - 2 * dd[1:-1, :] + dd[:-2, :])[:, 1:-1]; sec_x = np.abs(dd[:, 2:] - 2 * dd[:, 1:-1] + dd[:, :-2])[1:-1, :]
m_ = (dd[1:-1, 1:-1] > 0.02)
alt = float(max(sec_y[m_].mean(), sec_x[m_].mean()) / max(dd[1:-1, 1:-1][m_].mean(), 1e-9)) if m_.any() else 0.0
pct = 100.0 * changed / max(touched, 1); pct20 = 100.0 * changed20 / max(touched, 1)
print(f"RTSHADOW replay (screen-space, KYTY_RT_SHADOWS level {q}, trace {TW}x{TH} of {W}x{H}, strength {a.strength}, "
      f"{a.frames} frames, still camera): {shadowed:.2f} % of pixels shadowed (occlusion > 0.1), mean darkening on geometry "
      f"{mean_change:.3f} %, max {100 * float(dk.max()):.1f} %")
print(f"RTFLICKER replay sun shadows (screen-space): changed > 10 % {pct:.3f} % (> 20 % {pct20:.3f} %) of {touched} touched "
      f"points ({a.frames - a.warmup - 1} frames), max jump {100 * maxj:.1f} %")
print(f"STRIPES replay: profile index rows {r_idx:.4f} cols {c_idx:.4f}; local alternation {alt:.3f} (|2nd diff| / darkening)")
if a.png:
    from PIL import Image
    v = np.clip(dk / max(float(dk.max()), 1e-6), 0, 1)
    Image.fromarray((255 * (1 - v)).astype(np.uint8)).save(a.png)
ok = shadowed > 0 and mean_change > 0 and pct < 1.0 and max(r_idx, c_idx) < 0.05
sys.exit(0 if ok else 1)
