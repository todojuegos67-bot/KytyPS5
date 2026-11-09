// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// (bb-metrics-043) The in-game metrics panel: the Demon's Souls PC port's F8 panel (KytyPS5
// ds-pc-port51 src/port/ds-menu.cpp, namespace MetricsPanel: metricas.ps1's window drawn in the
// game image) with the same rows, order, colours, font, sizes, position and 1 s update, its values
// from Bloodborne's own systems instead of the Kyty log:
//   GPU / GPU reloj / VRAM   NVML (nvml.dll, the driver's), as DS
//   CPU / CPU reloj / RAM / Commit   GetSystemTimes, PDH "% Processor Performance", memory status
//   Frame                    BB::Metrics' 1 s window of the game's frames (the [fps] line)
//   Latencia                 bb_antilag's [lat] total (BB_LATENCY=1), else our scePadRead -> present
//   Escala / Resol. / Modelo bb_upscale's state (BB::Metrics::SetUpscale)
//   DLSS 5                   BB::Metrics::SetDlss5State (the DLSS 5 branch), else BB_DLSS5
//   FG / FG ahora            BB::FG::Cfg() and the [dlssg] / [fsr3] / [fsr3-mfg] lines
//   Anti-Lag                 BB::AntiLag::CurrentModeName()
//   mode line                BB_MODE (the launcher's mode)
//   FPS (real)               shown = FG's shown rate, real beside it
//   frametime graph          the last 60 windows' mean (green) and max (orange)
#include <algorithm>
#include <array>
#include <cfloat>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <imgui.h>
#include "imgui/imgui_layer.h"
#include "video_core/bb/bb_antilag.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_metrics.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace BB::Metrics {
namespace {

struct Row {
    std::string name, value;
    ImU32 color;
};
constexpr ImU32 kGreen = IM_COL32(50, 205, 50, 255), kRed = IM_COL32(255, 0, 0, 255),
                kBlue = IM_COL32(0, 191, 255, 255), kOrange = IM_COL32(255, 165, 0, 255),
                kWhite = IM_COL32(255, 255, 255, 255), kKhaki = IM_COL32(240, 230, 140, 255),
                kGold = IM_COL32(255, 215, 0, 255), kName = IM_COL32(150, 150, 150, 255);
std::mutex g_mutex;
std::vector<Row> g_rows;
std::string g_fps_text;
ImU32 g_fps_color = kWhite;
std::vector<double> g_ft, g_ft_max;
std::atomic<bool> g_started{false}, g_layer_added{false};

bool EnvOff(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && (v[0] == '0' || std::strcmp(v, "off") == 0);
}
bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0' && std::strcmp(v, "off") != 0;
}

ImU32 UseColor(double used, double total) {
    if (total <= 0) return kOrange;
    const double f = used / total;
    return f < 0.70 ? kGreen : f < 0.85 ? kGold : f < 0.95 ? kOrange : kRed;
}
std::string Fmt(const char* f, ...) {
    char text[256];
    va_list args;
    va_start(args, f);
    std::vsnprintf(text, sizeof(text), f, args);
    va_end(args);
    return text;
}

#if defined(_WIN32)
// NVML (nvml.dll, the driver's: what nvidia-smi queries), as DS.
struct Nvml {
    bool ok = false;
    void* dev = nullptr;
    int (*util)(void*, unsigned int*) = nullptr;
    int (*temp)(void*, int, unsigned int*) = nullptr;
    int (*clock)(void*, int, unsigned int*) = nullptr;
    int (*mem)(void*, unsigned long long*) = nullptr;
    void Init() {
        HMODULE m = LoadLibraryW(L"nvml.dll");
        if (m == nullptr) {
            wchar_t sys[MAX_PATH] = {};
            GetSystemDirectoryW(sys, MAX_PATH);
            m = LoadLibraryW((std::wstring(sys) + L"\\nvml.dll").c_str());
        }
        if (m == nullptr) return;
        auto init = reinterpret_cast<int (*)()>(reinterpret_cast<void*>(GetProcAddress(m, "nvmlInit_v2")));
        auto get = reinterpret_cast<int (*)(unsigned int, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetHandleByIndex_v2")));
        util = reinterpret_cast<int (*)(void*, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetUtilizationRates")));
        temp = reinterpret_cast<int (*)(void*, int, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetTemperature")));
        clock = reinterpret_cast<int (*)(void*, int, unsigned int*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetClockInfo")));
        mem = reinterpret_cast<int (*)(void*, unsigned long long*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "nvmlDeviceGetMemoryInfo")));
        ok = init != nullptr && get != nullptr && util != nullptr && temp != nullptr && mem != nullptr &&
             init() == 0 && get(0, &dev) == 0;
    }
};
// "\Processor Information(_Total)\% Processor Performance" (pdh.dll by name), as DS.
struct CpuPerf {
    void* query = nullptr;
    void* counter = nullptr;
    long (*collect)(void*) = nullptr;
    long (*value)(void*, unsigned long, unsigned long*, void*) = nullptr;
    int max_mhz = 0;
    void Init() {
        HKEY key = nullptr;
        if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                          KEY_READ, &key) == ERROR_SUCCESS) {
            DWORD mhz = 0, size = sizeof(mhz);
            if (RegQueryValueExW(key, L"~MHz", nullptr, nullptr, reinterpret_cast<BYTE*>(&mhz), &size) ==
                ERROR_SUCCESS)
                max_mhz = static_cast<int>(mhz);
            RegCloseKey(key);
        }
        HMODULE m = LoadLibraryW(L"pdh.dll");
        if (m == nullptr) return;
        auto open = reinterpret_cast<long (*)(const wchar_t*, uintptr_t, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhOpenQueryW")));
        auto add = reinterpret_cast<long (*)(void*, const wchar_t*, uintptr_t, void**)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhAddEnglishCounterW")));
        collect = reinterpret_cast<long (*)(void*)>(reinterpret_cast<void*>(GetProcAddress(m, "PdhCollectQueryData")));
        value = reinterpret_cast<long (*)(void*, unsigned long, unsigned long*, void*)>(
            reinterpret_cast<void*>(GetProcAddress(m, "PdhGetFormattedCounterValue")));
        if (open == nullptr || add == nullptr || collect == nullptr || value == nullptr) return;
        if (open(nullptr, 0, &query) != 0) return;
        if (add(query, L"\\Processor Information(_Total)\\% Processor Performance", 0, &counter) != 0)
            counter = nullptr;
        if (counter != nullptr) collect(query);
    }
    int Mhz() {
        if (counter == nullptr || max_mhz <= 0) return 0;
        struct {
            unsigned long status;
            double value;
        } v{};
        collect(query);
        if (value(counter, 0x200 /* PDH_FMT_DOUBLE */, nullptr, &v) != 0) return 0;
        return static_cast<int>(std::lround(v.value * max_mhz / 100.0));
    }
};
#endif

// DS's FrameGenLabel (ds-tables.h kFgModes) for BB_FG.
std::string FgLabel(const BB::FG::Config& c) {
    using BB::FG::Kind;
    if (c.kind == Kind::Off) return "Sin frame generation";
    const bool amd = c.kind == Kind::Amd;
    if (c.dynamic) return Fmt("%s MFG dinamico (x2-x%d)", amd ? "AMD" : "DLSS", std::max(c.multiplier, 2));
    return Fmt("%s x%d", amd ? "AMD" : "DLSS", c.multiplier);
}

// DS's mode names (Launcher-souls $resModes, written out as ds-menu.cpp's mode_name does).
std::string ModeName(int kind, int quality) {
    using BB::SR::Kind;
    using BB::SR::Quality;
    const auto k = static_cast<Kind>(kind);
    switch (static_cast<Quality>(quality)) {
    case Quality::Native:
        return k == Kind::Taa ? "TAA" : k == Kind::Dlss ? "DLAA" : k == Kind::Xess ? "AA" : k == Kind::Fsr4 ? "Nativo (AA)" : "AA nativo";
    case Quality::UltraQualityPlus: return "Ultra calidad+";
    case Quality::UltraQuality: return "Ultra calidad";
    case Quality::Quality: return "Calidad";
    case Quality::Balanced: return "Equilibrado";
    case Quality::Performance: return "Rendimiento";
    case Quality::UltraPerformance: return "Ultra rendimiento";
    case Quality::Dynamic: return "Dinámico";
    }
    return "";
}

void Sample(
#if defined(_WIN32)
    Nvml& nv, CpuPerf& cpu,
#endif
    uint64_t& idle0, uint64_t& total0, uint64_t& last_key) {
    std::vector<Row> rows;
    const Snapshot snap = Get();
#if defined(_WIN32)
    if (nv.ok) {
        unsigned int u[2] = {}, t = 0, clk = 0;
        unsigned long long m[3] = {};
        nv.util(nv.dev, u);
        nv.temp(nv.dev, 0, &t);
        if (nv.clock != nullptr) nv.clock(nv.dev, 0 /* graphics */, &clk);
        nv.mem(nv.dev, m);
        const double used = static_cast<double>(m[2] >> 20), total = static_cast<double>(m[0] >> 20);
        rows.push_back({"GPU", Fmt("%3u %%   %u C", u[0], t), kOrange});
        if (clk > 0) rows.push_back({"GPU reloj", Fmt("%4u MHz", clk), kOrange});
        rows.push_back({"VRAM", Fmt("%.1f / %.0f GB", used / 1024.0, total / 1024.0), UseColor(used, total)});
    } else {
        rows.push_back({"GPU", "(sin nvidia-smi)", kOrange});
    }
    {
        FILETIME fi, fk, fu;
        if (GetSystemTimes(&fi, &fk, &fu)) {
            const auto q = [](FILETIME f) {
                return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime;
            };
            const uint64_t idle = q(fi), total = q(fk) + q(fu);
            const double busy = total > total0 ? 100.0 * (1.0 - static_cast<double>(idle - idle0) /
                                                                     static_cast<double>(total - total0))
                                               : 0.0;
            idle0 = idle, total0 = total;
            rows.push_back(
                {"CPU", Fmt("%3d %%", static_cast<int>(std::lround(std::clamp(busy, 0.0, 100.0)))), kOrange});
        }
        if (const int mhz = cpu.Mhz(); mhz > 0) rows.push_back({"CPU reloj", Fmt("%4d MHz", mhz), kOrange});
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            const double gb = 1024.0 * 1024.0 * 1024.0;
            const double used = static_cast<double>(ms.ullTotalPhys - ms.ullAvailPhys) / gb,
                         total = static_cast<double>(ms.ullTotalPhys) / gb;
            rows.push_back({"RAM", Fmt("%.1f / %.0f GB", used, total), UseColor(used, total)});
            const double climit = static_cast<double>(ms.ullTotalPageFile) / gb,
                         cused = climit - static_cast<double>(ms.ullAvailPageFile) / gb;
            if (climit > 0)
                rows.push_back({"Commit", Fmt("%.1f / %.0f GB", cused, climit),
                                cused / climit > 0.90 ? kRed : UseColor(cused, climit)});
        }
    }
#endif
    const BB::FG::Config& fg = BB::FG::Cfg();
    const std::string fg_label = FgLabel(fg);
    const bool fg_on = fg.kind != BB::FG::Kind::Off;
    const bool fg_nv = fg.kind == BB::FG::Kind::Dlss;
    const bool fg_running = fg_nv ? BB::FG::DlssgEnabled() : BB::FG::Fsr3Active();
    const bool fg_fresh = snap.fg_age_s < 3.0 && snap.fg_shown > 0 &&
                          std::lround(snap.fg_shown * 10) != std::lround(snap.fg_real * 10);

    std::vector<double> ft, ftm;
    {
        std::scoped_lock lock(g_mutex);
        ft = g_ft, ftm = g_ft_max;
    }
    const bool any = snap.key != 0 && snap.window_age_s < 3.0;
    double shown = 0;
    if (any) {
        shown = fg_fresh && fg_on ? std::round(snap.fg_shown) : std::round(1000.0 / snap.mean_ms);
        rows.push_back({"Frame", Fmt("%.1f ms", snap.mean_ms), kOrange});
    }
    if (snap.key != 0 && snap.key != last_key) {
        last_key = snap.key;
        ft.push_back(snap.mean_ms), ftm.push_back(snap.max_ms);
        while (ft.size() > 60) ft.erase(ft.begin()), ftm.erase(ftm.begin());
    }
    {
        double lat = 0, p95 = 0;
        if (snap.lat_age_s < 3.0 && snap.lat_mean > 0) lat = snap.lat_mean, p95 = snap.lat_p95;
        else if (any && snap.own_lat_mean > 0) lat = snap.own_lat_mean, p95 = snap.own_lat_p95;
        if (lat > 0) {
            const ImU32 lc = lat < 40 ? kGreen : lat < 60 ? kGold : lat < 80 ? kOrange : kRed;
            rows.push_back({"Latencia", Fmt("%.0f ms (p95 %.0f)", lat, p95), lc});
        }
    }
    // Escala / Resol. / Modelo (DS's Get-Scale).
    {
        using BB::SR::Kind;
        const char* dot = " \xC2\xB7 ";
        const char* arrow = "\xE2\x86\x92";
        const auto& up = snap.up;
        const auto kind = static_cast<Kind>(up.kind);
        std::string rk, mode, res, model;
        ImU32 col = kOrange;
        if (snap.has_up && up.running) {
            // bb-dcideas-094: what really runs: FSR 4.1.1 (bbport's replay) or FSR 4.0 for fsr4.
            const bool v411 = kind == Kind::Fsr4 && BB::SR::Fsr4::Active411();
            rk = kind == Kind::Taa ? "TAA" : kind == Kind::Dlss ? "DLSS" : kind == Kind::Xess ? "XeSS" : kind == Kind::Fsr4 ? (v411 ? "FSR 4.1.1" : "FSR 4.0") : "FSR 3.1";
            mode = ModeName(up.kind, up.quality);
            col = kind == Kind::Dlss ? kGreen : kind == Kind::Xess ? kBlue : kRed;
            const double ratio = up.render_h != 0 ? static_cast<double>(up.output_h) / up.render_h : 0.0;
            res = up.render_h != 0 ? Fmt("%up %s %up (x%.3g)", up.render_h, arrow, up.output_h, ratio)
                                   : Fmt("%s %up", arrow, up.output_h);
            if (kind == Kind::Dlss) {
                const char* p = std::getenv("BB_DLSS_PRESET");
                const std::string pk = p != nullptr && *p >= 'A' && *p <= 'O' ? std::string(1, *p) : "";
                model = pk == "K"   ? "K (Transformer, DLSS 4)"
                        : pk == "L" ? "L (Transformer, DLSS 4.5)"
                        : pk == "M" ? "M (Transformer, DLSS 4.5)"
                        : pk.empty() ? "auto (NVIDIA)"
                                     : pk;
            } else if (kind == Kind::Xess) {
                model = "XeSS";
            } else if (kind == Kind::Fsr4) {
                model = BB::SR::Fsr4::Active411() ? std::string("FSR 4.1.1 (IA, ") + BB::SR::Fsr4::ModelName411() + ")"
                                                  : "FSR 4.0 (IA, modelo v07)";
            } else {
                model = "FSR 3.1";
            }
        } else if (snap.has_up) {
            // No upscale this frame: the game's own size to the window (DS's "Nativo").
            rk = "Nativo";
            if (kind != Kind::Off && kind != Kind::Native) mode = std::string(BB::SR::KindName(kind)) + " en espera";
            if (up.render_h != 0 || up.output_h != 0)
                res = Fmt("%up %s %up", up.render_h != 0 ? up.render_h : up.output_h, arrow, up.output_h);
        }
        if (!rk.empty()) {
            rows.push_back({"Escala", mode.empty() ? rk : rk + dot + mode, col});
            if (!res.empty()) rows.push_back({"Resol.", res, col});
            if (!model.empty()) rows.push_back({"Modelo", model, col});
        }
    }
    // DLSS 5 (not in DS's panel: asked for here). The DLSS 5 back end reports through SetDlss5State.
    {
        const bool asked = EnvOn("BB_DLSS5");
        if (snap.dlss5_state == 1) {
            rows.push_back({"DLSS 5", snap.dlss5_detail.empty() ? "Activo" : "Activo" + std::string(" \xC2\xB7 ") + snap.dlss5_detail, kGreen});
        } else if (snap.dlss5_state == 2) {
            rows.push_back({"DLSS 5", snap.dlss5_detail.empty() ? "Pedido, no activo" : snap.dlss5_detail, kOrange});
        } else if (snap.dlss5_state == 0 || !asked) {
            rows.push_back({"DLSS 5", "No", kKhaki});
        } else {
            rows.push_back({"DLSS 5", "no disponible", kName});
        }
    }
    if (fg_on) {
        const ImU32 fg_col = fg_nv ? kGreen : kRed;
        std::string label = fg_label;
        if (fg_nv) {
            const char* p = BB::FG::Env("KYTY_DLSSG_PRESET");
            label += Fmt(" (modelo %c)", p != nullptr && *p >= 'A' && *p <= 'C' ? *p : 'C');
        }
        if (!fg_running) label += " (inactivo)";
        rows.push_back({"FG", label, fg_col});
        int now = 0;
        if (fg_running) {
            if (fg_nv) now = BB::FG::DlssgGenerated() > 0 ? BB::FG::DlssgGenerated() + 1 : 0;
            else now = snap.fg_mult > 0 ? snap.fg_mult : (fg.dynamic ? 0 : fg.multiplier);
        }
        if (now > 0) rows.push_back({"FG ahora", Fmt("x%d", now), fg_col});
    }
    {
        const std::string m = BB::AntiLag::CurrentModeName();
        const std::string al = m == "reflex-sl" ? (fg_nv && fg_running ? "Reflex (DLSS-G)" : "Reflex")
                               : m == "antilag2" ? "Anti-Lag 2 (AMD)"
                               : m == "reflex"   ? "Reflex"
                               : m == "limiter"  ? "Normal"
                               : m == "lowlat"   ? "Ultra"
                                                 : "No";
        rows.push_back({"Anti-Lag", al, al == "No" ? kKhaki : kGreen});
    }
    {
        const char* bm = std::getenv("BB_MODE");
        const std::string m = bm != nullptr ? bm : "";
        const char* name = m == "ligero"        ? "Modo Ligero"
                           : m == "equilibrado" ? "Modo Equilibrado"
                           : m == "ultra"       ? "Modo Ultra"
                           : m == "extremo"     ? "Modo Extremo"
                                                : "Personalizado";
        rows.push_back({"", name, kKhaki});
    }
    std::scoped_lock lock(g_mutex);
    g_rows = std::move(rows);
    g_fps_text = shown > 0 ? (fg_on && fg_fresh && snap.fg_real > 0
                                  ? Fmt("FPS %.0f  (real %.0f)", shown, std::round(snap.fg_real))
                                  : Fmt("FPS %.0f", shown))
                           : std::string();
    g_fps_color = fg_on ? (fg_nv ? kGreen : kRed) : kWhite;
    g_ft = std::move(ft), g_ft_max = std::move(ftm);
}

void Start() {
    if (g_started.exchange(true)) return;
    std::thread([] {
#if defined(_WIN32)
        Nvml nv;
        CpuPerf cpu;
        nv.Init();
        cpu.Init();
#endif
        uint64_t idle0 = 0, total0 = 0, last_key = 0;
        for (;;) {
            if (PanelShown()) {
                Sample(
#if defined(_WIN32)
                    nv, cpu,
#endif
                    idle0, total0, last_key);
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
}

ImFont* FindFont(const char* name) {
    for (ImFont* font : ImGui::GetIO().Fonts->Fonts)
        if (std::strstr(font->GetDebugName(), name) != nullptr) return font;
    return nullptr;
}

// Top-centre, 10 px x S from the top (DS's Draw, unchanged).
void Draw(float width, float height) {
    const float S = std::clamp(height / 1080.0f, 1.0f, 2.5f);
    ImFont* f = FindFont("consolab");
    if (f == nullptr) f = ImGui::GetFont();
    std::vector<Row> rows;
    std::string fps;
    ImU32 fps_col;
    std::vector<double> ft, ftm;
    {
        std::scoped_lock lock(g_mutex);
        rows = g_rows, fps = g_fps_text, fps_col = g_fps_color, ft = g_ft, ftm = g_ft_max;
    }
    if (rows.empty()) rows.push_back({"", "Cargando...", kOrange});
    const float row_px = std::round(15 * S), title_px = std::round(16 * S), fps_px = std::round(34 * S),
                small_px = std::round(10 * S);
    const float pad = std::round(8 * S), row_h = std::ceil(row_px * 1.12f), title_h = std::ceil(title_px * 1.4f),
                fps_h = fps.empty() ? 0.0f : std::ceil(fps_px * 1.1f);
    float name_w = std::round(90 * S), val_w = std::round(230 * S), wide_w = 0;
    for (const auto& r : rows) {
        const float w = f->CalcTextSizeA(row_px, FLT_MAX, 0, r.value.c_str()).x;
        if (!r.name.empty()) {
            name_w = std::max(name_w, f->CalcTextSizeA(row_px, FLT_MAX, 0, r.name.c_str()).x + std::round(10 * S));
            val_w = std::max(val_w, w);
        } else
            wide_w = std::max(wide_w, w);
    }
    const float W = std::min(std::max(name_w + val_w, wide_w) + 2 * pad, width * 0.4f);
    const float graph_h = std::round(90 * S);
    const float x0 = std::floor((width - W) * 0.5f), y0 = std::round(10 * S);
    auto* dl = ImGui::GetForegroundDrawList();
    const char* title = "BLOODBORNE";
    const float tw = f->CalcTextSizeA(title_px, FLT_MAX, 0, title).x;
    dl->AddText(f, title_px, {std::floor(x0 + (W - tw) * 0.5f), y0 + pad}, kWhite, title);
    float y = y0 + pad + title_h;
    const ImVec4 clip{x0, y0, x0 + W, y0 + 4000};
    for (const auto& r : rows) {
        if (!r.name.empty()) {
            dl->AddText(f, row_px, {x0 + pad, y}, kName, r.name.c_str(), nullptr, 0, &clip);
            dl->AddText(f, row_px, {x0 + pad + name_w, y}, r.color, r.value.c_str(), nullptr, 0, &clip);
        } else
            dl->AddText(f, row_px, {x0 + pad, y}, r.color, r.value.c_str(), nullptr, 0, &clip);
        y += row_h;
    }
    if (!fps.empty()) {
        dl->AddText(f, fps_px, {x0 + pad, y}, fps_col, fps.c_str());
        y += fps_h;
    }
    y += pad;
    // The frametime graph: 40 ms at the top, 8.3 / 16.7 / 33.3 lines, the peaks (orange) and the mean (green).
    const float gy = y, gw = W, top = 40.0f;
    for (float ms : {8.3f, 16.7f, 33.3f}) {
        const float ly = gy + graph_h - ms / top * graph_h;
        dl->AddLine({x0, ly}, {x0 + gw, ly}, IM_COL32(60, 60, 60, 255));
    }
    dl->AddText(f, small_px, {x0, gy}, IM_COL32(128, 128, 128, 255), "frametime (ms)  40   verde media, naranja picos");
    dl->AddText(f, small_px, {x0, gy + graph_h - 16.7f / top * graph_h - small_px}, IM_COL32(128, 128, 128, 255),
                "16.7");
    const auto line = [&](const std::vector<double>& list, ImU32 col) {
        if (list.size() < 2) return;
        const float step = gw / 59.0f, xs = x0 + gw - (list.size() - 1) * step;
        for (size_t i = 1; i < list.size(); i++) {
            const float y1 = gy + graph_h - std::min(static_cast<float>(list[i - 1]), top) / top * graph_h;
            const float y2 = gy + graph_h - std::min(static_cast<float>(list[i]), top) / top * graph_h;
            dl->AddLine({xs + (i - 1) * step, y1}, {xs + i * step, y2}, col, 2.0f * S);
        }
    };
    line(ftm, IM_COL32(255, 150, 40, 255));
    line(ft, IM_COL32(0, 255, 0, 255));
}

class PanelLayer final : public ImGui::Layer {
public:
    void Draw() override {
        if (!PanelShown()) return;
        const auto& io = ImGui::GetIO();
        Metrics::Draw(io.DisplaySize.x, io.DisplaySize.y);
    }
    bool ShouldKeepDrawing() override {
        return false; // the game's frames redraw it
    }
};
PanelLayer g_layer;

bool PanelAllowed() {
    static const bool allowed = !EnvOff("BB_METRICS_OVERLAY");
    return allowed;
}

void Show(bool shown) {
    if (!PanelAllowed()) return;
    if (shown) {
        if (!g_layer_added.exchange(true)) ImGui::Layer::AddLayer(&g_layer);
        Start();
    }
    SetPanelShown(shown);
}

} // namespace

bool PanelKeyEnabled() {
    static const bool on = PanelAllowed() && !EnvOff("BB_METRICS_KEY");
    return on;
}

void TogglePanel() {
    Show(!PanelShown());
}

void InitPanel() {
#if defined(_WIN32)
    // Consolas Bold, the face of metricas.ps1 / DS's panel (found by name in Draw). The atlas
    // rasterises its glyphs only when the panel draws.
    if (PanelAllowed()) {
        const char* windows = std::getenv("WINDIR");
        const std::string path = std::string(windows != nullptr ? windows : "C:\\Windows") + "\\Fonts\\consolab.ttf";
        if (std::FILE* file = std::fopen(path.c_str(), "rb"); file != nullptr) {
            std::fclose(file);
            ImFontConfig cfg{};
            std::snprintf(cfg.Name, sizeof(cfg.Name), "consolab.ttf");
            (void)ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), 32.0f, &cfg);
        }
    }
#endif
    // The launcher's "Metricas" (BB_METRICS=1): shown from the start, as DS's launcher metrics.
    if (EnvOn("BB_METRICS")) Show(true);
}

} // namespace BB::Metrics
