// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB_METRICS_OUT=<file>: once a second, the same lines as Kyty writes for Demon's Souls, read by
// metricas-bb.ps1 (-FpsFile):
//   [fps] N frames: mean X ms, max Y ms
//   [lat] input->present mean A ms, p95 B ms
//   [up] <upscaler> <mode> <render> -> <output>   (or "[up] off")
// Appended and flushed per line; the previous run's file is kept as <file>.1.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace BB::Metrics {
bool Enabled();
void OnInput();                // the game read the pad
void OnPresent();              // a new guest frame was presented
void SetUpscaleLine(const std::string& text);
void SetResLine(const std::string& text); // written each second as "[res] game WxH -> output WxH ..."
void Line(const char* text);
// bb-dlss5perf-106: counters for the per-second "[perf]" line (vk_presenter.cpp). Lock-free.
void PerfDlss5(double cpu_ms);              // one DLSS 5 Apply: CPU time to record it
void PerfDlss5Gpu(double gpu_ms);           // a DLSS 5 GPU time read back (async, frames late)
void PerfPresentWait(double ms);            // the presenter waited for a free frame / its fence
void PerfLine(int frames_in_flight, int frames_total);
// bb-cpcpu-112: GPU command processor time by category (ns, summed; read and cleared by PerfLine).
// Draw/Dispatch total; nested parts: pipeline lookup, BindResources (buffers + textures inside),
// buffer cache ObtainBuffer, texture cache FindImage/FindTexture, BB camera scan, object motion,
// render setup + command recording (BeginRendering, vertex/index, barriers, dynamic state, cmds).
// bb-perf-132: + render-target lookup (PrepareRenderState), BB constant patch scan (inside bind),
// BB sample hooks (upscale / HUD per sampled image, inside bind), samplers (inside bind).
enum CpCat : int {
    CpDraw, CpPipe, CpBind, CpBuf, CpTex, CpScan, CpObjMv, CpRec,
    CpPrep, CpPatch, CpHook, CpSamp, CpCount
};
extern std::atomic<uint64_t> g_cp_ns[CpCount];
extern std::atomic<uint64_t> g_cp_draws, g_cp_dispatches;
// bb-perf-132: constant-patch scans (buffers looked at), bytes scanned, buffers rejected by the
// fast pre-filter, buffers patched.
extern std::atomic<uint64_t> g_cp_patch_scans, g_cp_patch_bytes, g_cp_patch_fast, g_cp_patch_hits;
// BB_CP_FASTPATCH=0: the 1.31 constant-patch scan (exact float loop over every buffer).
bool CpFastPatch();
// BB_CP_DETAIL=0: no sub-timers for the bb-perf-132 categories (prep/patch/hook/samplers).
bool CpDetail();
struct CpTimer {
    using C = std::chrono::steady_clock;
    int cat;
    C::time_point t0;
    explicit CpTimer(int c) : cat(c), t0(C::now()) {}
    ~CpTimer() {
        g_cp_ns[cat].fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(C::now() - t0).count()),
            std::memory_order_relaxed);
    }
}; // once per presented game frame // a line as it is (frame generation: "[dlssg] ... real X fps, shown Y fps")

// ---- (bb-metrics-043) The in-game metrics panel (bb_metrics_panel.cpp): the Demon's Souls port's F8
// panel (ds-menu.cpp MetricsPanel, metricas.ps1's look), top-centre of the game image. F8 shows /
// hides it; BB_METRICS=1 (the launcher's "Metricas") starts it shown. BB_METRICS_OVERLAY=0: no
// panel at all (F8 then stays shadPS4's "reload inputs"); BB_METRICS_KEY=0: F8 not taken.
// Hidden it costs one relaxed atomic load per present / pad read / FG line.
bool PanelShown();
bool PanelKeyEnabled(); // F8 is the panel's (not shadPS4's reload inputs)
void TogglePanel();     // SDL thread (F8)
void SetPanelShown(bool shown);
void InitPanel();       // after ImGui's init: the Consolas Bold face, BB_METRICS=1 shows it

// The upscaler now (bb_upscale.cpp, every frame while the panel or the metrics file is on).
struct Upscale {
    int kind = 0;    // BB::SR::Kind
    int quality = 0; // BB::SR::Quality
    bool running = false; // upscaled this frame
    bool scene = false;   // a 3D scene seen (render size learned)
    unsigned render_w = 0, render_h = 0, output_w = 0, output_h = 0;
};
void SetUpscale(const Upscale& up);

// DLSS 5 (the bb-dlss5 branch, BB_DLSS5): its code calls this with its state; until it does the
// panel reads BB_DLSS5 itself and shows "no disponible" when it is asked for (this engine has no
// DLSS 5 yet). state: 0 off, 1 active, 2 asked but not running; detail: shown after it (may be null).
void SetDlss5State(int state, const char* detail);

struct Snapshot {
    // The game's frames, last 1 s window (as the [fps] line); key moves with each new window.
    double mean_ms = 0, max_ms = 0;
    uint64_t key = 0;
    double window_age_s = 1e9; // seconds since the last window closed
    // Frame generation's last real / shown line ([dlssg] / [fsr3] / [fsr3-mfg]) and multiplier.
    double fg_real = 0, fg_shown = 0, fg_age_s = 1e9;
    int fg_mult = 0;
    // [lat] (BB_LATENCY=1, bb_antilag.cpp): total (input->present + scanout + fg hold) mean / p95.
    double lat_mean = 0, lat_p95 = 0, lat_age_s = 1e9;
    std::string lat_mode; // "reflex-sl", "antilag2", "reflex", "limiter", "lowlat", "off"
    // Our own input (scePadRead) -> present, the last window (when [lat] is off).
    double own_lat_mean = 0, own_lat_p95 = 0;
    Upscale up;
    bool has_up = false;
    int dlss5_state = -1; // -1: never set by a DLSS 5 back end
    std::string dlss5_detail;
};
Snapshot Get();
} // namespace BB::Metrics
