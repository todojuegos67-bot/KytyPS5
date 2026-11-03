// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// BB_FG settings, the device / swapchain plumbing, and NVIDIA DLSS Frame Generation (DLSS-G)
// through Streamline, ported from the Demon's Souls fork's src/local/frame-gen.cpp (BeforePresent,
// AfterPresent, SwapchainChanging, the dynamic multiplier and its pacing, the per-second lines).
// Streamline itself is loaded by bb_sr_dlss.cpp (NVIDIA-signed sl.interposer.dll, the Vulkan
// loader from then on), which also gives its entry points here.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <cctype>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include "video_core/bb/bb_antilag.h"
#include "video_core/bb/bb_dyn_multiplier.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_fg_internal.h"
#include "video_core/bb/bb_kyty_compat.h"
#include "video_core/bb/bb_metrics.h"
#include "video_core/bb/bb_sr.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#undef min
#undef max
#endif
#if defined(_WIN32) && defined(BB_STREAMLINE)
// (sl_pcl.h's C++23 branch aliases std::to_underlying, which is not valid C++: its own helper.)
#pragma push_macro("__cplusplus")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#undef __cplusplus
#define __cplusplus 202002L
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#pragma clang diagnostic pop
#pragma pop_macro("__cplusplus")
#endif

namespace BB::FG {

const char* Env(const char* kyty_name) {
    if (kyty_name == nullptr) {
        return nullptr;
    }
    const std::string_view name{kyty_name};
    if (name.starts_with("KYTY_")) {
        const std::string bb = "BB_" + std::string(name.substr(5));
        if (const char* v = std::getenv(bb.c_str()); v != nullptr) {
            return v;
        }
    }
    return std::getenv(kyty_name);
}

int DynBaseCap(double base_fps, int& state) {
    static const bool on = [] {
        const char* v = Env("KYTY_FG_DYN_CAP");
        return v == nullptr || *v != '0';
    }();
    if (!on || !(base_fps > 0.0)) {
        return 64;
    }
    const auto cap_for = [](double fps) { return fps < 50.0 ? 2 : fps < 70.0 ? 3 : 64; };
    const int down = cap_for(base_fps);      // lowered at once
    const int up = cap_for(base_fps - 5.0);  // raised only 5 fps past the line
    const int before = state;
    if (state == 0 || down < state) {
        state = down;
    } else if (up > state) {
        state = up;
    }
    if (state != before) {
        Printf("[dyn] ceiling %s for a base of %.1f real fps (BB_FG_DYN_CAP=0: none)",
               state >= 64 ? "none" : state == 3 ? "x3" : "x2", base_fps);
    }
    return state;
}

void Printf(const char* format, ...) {
    char text[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    std::size_t n = std::strlen(text);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r')) {
        text[--n] = '\0';
    }
    BBLogf("%s", text);
    BB::Metrics::Line(text);
}

namespace {

#if defined(_WIN32)
void SetProcessEnv(const char* name, const char* value) {
    _putenv_s(name, value);
    SetEnvironmentVariableA(name, value);
}
#else
void SetProcessEnv(const char* name, const char* value) {
    setenv(name, value, 1);
}
#endif

Config ParseConfig() {
    Config c{};
    const char* v = std::getenv("BB_FG");
    std::string s = v != nullptr ? v : "off";
    // Aliases: fsr3-xN / fsr3-dyn / amd-xN / dlssg-xN / dlss-g-xN, "xN" or "N" spelled either way.
    for (const char* from : {"fsr3-", "fsr-", "dlssg-", "dlss-g-"}) {
        if (s.starts_with(from)) {
            s = std::string(std::string_view(from).starts_with("fsr") ? "amd-" : "dlss-") +
                s.substr(std::strlen(from));
            break;
        }
    }
    if (const auto dash = s.find('-'); dash != std::string::npos && dash + 1 < s.size() &&
                                       s[dash + 1] >= '2' && s[dash + 1] <= '6') {
        s.insert(dash + 1, "x");
    }
    if (s.starts_with("dlss-")) {
        c.kind = Kind::Dlss;
        const std::string m = s.substr(5);
        if (m == "dyn") {
            c.dynamic = true;
            c.multiplier = 6; // the maximum (x6), capped by what the GPU makes
        } else if (m.size() == 2 && m[0] == 'x' && m[1] >= '2' && m[1] <= '6') {
            c.multiplier = m[1] - '0';
        } else {
            c.kind = Kind::Off;
        }
    } else if (s.starts_with("amd-")) {
        c.kind = Kind::Amd;
        const std::string m = s.substr(4);
        if (m == "dyn") {
            c.dynamic = true;
            c.multiplier = 4;
        } else if (m.size() == 2 && m[0] == 'x' && m[1] >= '2' && m[1] <= '4') {
            c.multiplier = m[1] - '0';
        } else {
            c.kind = Kind::Off;
        }
    }
    // The Demon's Souls code reads its settings by their KYTY_ names (KYTY_FSR3_FG, KYTY_FSR3_MFG,
    // KYTY_FSR3_DYNAMIC; the patched FidelityFX DLL reads KYTY_FSR3_FG_FRAMES / _ACTIVE).
    if (c.kind == Kind::Amd) {
        SetProcessEnv("KYTY_FSR3_FG", "1");
        if (c.dynamic) {
            SetProcessEnv("KYTY_FSR3_DYNAMIC", "1");
            if (Env("KYTY_FSR3_MFG") == nullptr) SetProcessEnv("KYTY_FSR3_MFG", "4");
        } else if (c.multiplier > 2) {
            const std::string n = std::to_string(c.multiplier);
            SetProcessEnv("KYTY_FSR3_MFG", n.c_str());
        }
    }
    // The Demon's Souls launcher's defaults: KYTY_DLSSG_PACE=grid, KYTY_VRR_CAP_FPS=auto for the
    // x4 FSR 3 configuration, KYTY_DYN_QUALITY=0.95 for the dynamic ones.
    if (Env("KYTY_DLSSG_PACE") == nullptr) SetProcessEnv("BB_DLSSG_PACE", "grid");
    if (c.kind == Kind::Amd && !c.dynamic && c.multiplier == 4 && Env("KYTY_VRR_CAP_FPS") == nullptr) {
        SetProcessEnv("BB_VRR_CAP_FPS", "auto");
    }
    // (bb_dyn_multiplier.h reads KYTY_DYN_QUALITY itself.)
    if (const char* q = std::getenv("BB_DYN_QUALITY"); q != nullptr) {
        SetProcessEnv("KYTY_DYN_QUALITY", q);
    } else if (c.dynamic && std::getenv("KYTY_DYN_QUALITY") == nullptr) {
        SetProcessEnv("KYTY_DYN_QUALITY", "0.95");
    }
    // BB_REFLEX=on|boost|off: NVIDIA Reflex. With Streamline (DLSS / DLSS-G) its sl.reflex mode;
    // without, VK_NV_low_latency2 through the latency module (as BB_ANTILAG=1 picks on NVIDIA).
    if (const char* r = std::getenv("BB_REFLEX"); r != nullptr && *r != '\0') {
        const std::string mode = r;
        if (std::getenv("BB_DLSSG_REFLEX") == nullptr) {
            SetProcessEnv("BB_DLSSG_REFLEX", mode == "0" || mode == "off"   ? "off"
                                             : mode == "boost" || mode == "2" ? "boost"
                                                                              : "low");
        }
        if (mode != "0" && mode != "off" && std::getenv("BB_ANTILAG") == nullptr) {
            SetProcessEnv("BB_ANTILAG", "1");
        }
    }
    return c;
}

} // namespace

const Config& Cfg() {
    static const Config config = [] {
        Config c = ParseConfig();
        if (c.kind != Kind::Off) {
            BBLogf("Frame generation: BB_FG=%s (%s, %s x%d)", std::getenv("BB_FG"),
                   c.kind == Kind::Dlss ? "NVIDIA DLSS-G" : "AMD FSR 3",
                   c.dynamic ? "dynamic up to" : "fixed", c.multiplier);
        }
        return c;
    }();
    return config;
}

const char* Name() {
    static const std::string name = [] {
        const auto& c = Cfg();
        if (c.kind == Kind::Off) return std::string("off");
        return std::string(c.kind == Kind::Dlss ? "dlss-" : "amd-") +
               (c.dynamic ? std::string("dyn") : "x" + std::to_string(c.multiplier));
    }();
    return name.c_str();
}

bool StreamlineWanted() {
    return Cfg().kind == Kind::Dlss;
}

bool StreamlineLoaded() {
    return SR::Dlss::Loaded() && SR::Dlss::LoaderPath() != nullptr;
}

bool Fsr3Requested() {
    return Amd::Requested();
}

bool Fsr3Active() {
    return Amd::Active();
}


// ---------------------------------------------------------------------------------------------
// DLSS-G preset (Demon's Souls frame-gen.cpp DlssgPreset, as it is; its launcher's default C).
// KYTY_DLSSG_PRESET (BB_DLSSG_PRESET) = A | B | C (default) | 0.
// ---------------------------------------------------------------------------------------------
#if defined(_WIN32) && defined(BB_STREAMLINE)
namespace DlssgPreset {

constexpr uint32_t kGetSettingInterface = 0x73BF8338u;
constexpr uint32_t kSettingId           = 0x10E41DF1u;
constexpr int32_t  kOk                  = 0;
constexpr int32_t  kSettingNotFound     = -160;

#pragma pack(push, 4)
union SettingValue {
	uint32_t dword;
	struct {
		uint32_t length;
		uint8_t  data[4096];
	} binary;
	uint16_t text[2048];
	uint64_t qword;
};
struct Setting { // NVDRS_SETTING_V1
	uint32_t     version;
	uint16_t     name[2048];
	uint32_t     id;
	uint32_t     type;
	uint32_t     location;
	uint32_t     current_predefined;
	uint32_t     predefined_valid;
	SettingValue predefined;
	SettingValue current;
};
#pragma pack(pop)
static_assert(sizeof(Setting) == 12320);
constexpr uint32_t kSettingVersion = sizeof(Setting) | (1u << 16);

using QueryInterfaceFn = void*(__cdecl*)(uint32_t);
using GetSettingFn     = int32_t(__cdecl*)(void*, void*, uint32_t, Setting*);

uint32_t                  g_preset = 0; // 0 off, else 1..3
std::atomic<QueryInterfaceFn> g_query {nullptr};
std::atomic<GetSettingFn>     g_get_setting {nullptr};
std::atomic<uint64_t>         g_reads {0};
std::atomic<uint64_t>         g_applied {0};
std::atomic<int>              g_provider_c {-1}; // -1 unknown, 0 lacks C, 1 has C
std::mutex                    g_log_mutex;

// The module's VS_FIXEDFILEINFO (from its own resources: no version.dll).
bool ModuleVersion(HMODULE module, uint16_t v[4]) {
	HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16));
	if (resource == nullptr) return false;
	HGLOBAL     handle = LoadResource(module, resource);
	const DWORD size   = SizeofResource(module, resource);
	const auto* bytes  = handle != nullptr ? static_cast<const uint8_t*>(LockResource(handle)) : nullptr;
	if (bytes == nullptr) return false;
	for (DWORD i = 0; i + 52 <= size; i += 4) {
		uint32_t signature = 0;
		std::memcpy(&signature, bytes + i, 4);
		if (signature != 0xFEEF04BDu) continue;
		uint32_t ms = 0, ls = 0;
		std::memcpy(&ms, bytes + i + 8, 4);
		std::memcpy(&ls, bytes + i + 12, 4);
		v[0] = static_cast<uint16_t>(ms >> 16);
		v[1] = static_cast<uint16_t>(ms);
		v[2] = static_cast<uint16_t>(ls >> 16);
		v[3] = static_cast<uint16_t>(ls);
		return true;
	}
	return false;
}

char Letter(uint32_t preset) { return preset == 0 ? '0' : static_cast<char>('A' + preset - 1); }

// Whether the reading module (nvngx_dlssg.dll) implements Preset C; logs once.
bool ProviderHasC(const void* caller) {
	int known = g_provider_c.load(std::memory_order_acquire);
	if (known >= 0) return known == 1;
	std::lock_guard lock(g_log_mutex);
	known = g_provider_c.load(std::memory_order_acquire);
	if (known >= 0) return known == 1;
	HMODULE module = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                   static_cast<LPCWSTR>(caller), &module);
	char     name[MAX_PATH] = "?";
	uint16_t v[4] {};
	const bool versioned = module != nullptr && ModuleVersion(module, v);
	if (module != nullptr && GetModuleFileNameA(module, name, MAX_PATH) != 0) {
		if (const char* slash = std::strrchr(name, '\\'); slash != nullptr) std::memmove(name, slash + 1, std::strlen(slash));
	}
	const uint64_t version = versioned ? (uint64_t(v[0]) << 32) | (uint64_t(v[1]) << 16) | v[2] : 0;
	const bool     has_c   = versioned && version >= ((uint64_t(310) << 32) | (uint64_t(9) << 16) | 16);
	if (g_preset == 3) {
		Printf("[dlssg] preset C requested; provider %s %u.%u.%u: %s\n", name, v[0], v[1], v[2],
		     has_c ? "applied" : "not supported (needs >= 310.9.16), left to the driver");
		std::printf("[dlssg] preset C requested; provider %s %u.%u.%u: %s\n", name, v[0], v[1], v[2],
		            has_c ? "applied" : "not supported (needs >= 310.9.16)");
	} else {
		Printf("[dlssg] preset %c requested; provider %s %u.%u.%u: applied\n", Letter(g_preset), name, v[0], v[1], v[2]);
		std::printf("[dlssg] preset %c requested; provider %s %u.%u.%u: applied\n", Letter(g_preset), name, v[0], v[1],
		            v[2]);
	}
	
	g_provider_c.store(has_c ? 1 : 0, std::memory_order_release);
	return has_c;
}

__declspec(noinline) int32_t __cdecl GetSettingHook(void* session, void* profile, uint32_t id, Setting* setting) {
	const GetSettingFn original = g_get_setting.load(std::memory_order_acquire);
	const int32_t      result   = original(session, profile, id, setting);
	if (id != kSettingId || setting == nullptr || setting->version != kSettingVersion) return result;
	if (!(result == kSettingNotFound || (result == kOk && setting->id == id && setting->type == 0))) return result;
	// Only nvngx_dlssg*.dll's own reads are answered (as MFG-Unlocker does: its override is limited to the
	// retained DLSS-G provider); other readers of the key (NGX core, Streamline, the driver) keep the driver's
	// value. Each distinct reader is logged once, to see who reads the key. KYTY_DLSSG_PRESET_ALL=1: every reader.
	{
		HMODULE caller = nullptr;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                   static_cast<LPCWSTR>(_ReturnAddress()), &caller);
		char path[MAX_PATH] = "?";
		if (caller != nullptr) GetModuleFileNameA(caller, path, MAX_PATH);
		const char* slash = std::strrchr(path, '\\');
		const char* name  = slash != nullptr ? slash + 1 : path;
		const bool  provider = _strnicmp(name, "nvngx_dlssg", 11) == 0;
		static const bool all = [] {
			const char* v = Env("KYTY_DLSSG_PRESET_ALL");
			return v != nullptr && *v != '\0' && *v != '0';
		}();
		{
			static HMODULE seen[16] {};
			std::lock_guard lock(g_log_mutex);
			bool known = false;
			for (HMODULE& m: seen) {
				if (m == caller) { known = true; break; }
				if (m == nullptr) { m = caller; break; }
			}
			if (!known) {
				Printf("[dlssg] FG preset key read by %s (driver %s %u): %s\n", name, result == kOk ? "value" : "none",
				     result == kOk ? setting->current.dword : 0u, provider || all ? "overridden" : "left as is");
				
			}
		}
		if (!provider && !all) return result;
	}
	const uint64_t reads = g_reads.fetch_add(1, std::memory_order_relaxed) + 1;
	const uint32_t was   = result == kOk ? setting->current.dword : 0;
	const bool has_c = ProviderHasC(_ReturnAddress()); // (Also logs the provider once.)
	if (g_preset == 3 && !has_c) return result;
	if (result == kSettingNotFound) {
		std::memset(setting, 0, sizeof(*setting));
		setting->version = kSettingVersion;
		setting->id      = id;
		constexpr char label[] = "Override DLSS-FG preset";
		for (size_t i = 0; i < sizeof(label); ++i) setting->name[i] = static_cast<uint16_t>(label[i]);
	}
	setting->location           = 0;
	setting->current_predefined = 0;
	std::memset(&setting->current, 0, sizeof(setting->current));
	setting->current.dword = g_preset;
	const uint64_t applied = g_applied.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((reads & (reads - 1)) == 0) {
		Printf("[dlssg] FG preset read #%llu by the provider (driver %s %u, answered %c, %llu overridden)\n",
		     static_cast<unsigned long long>(reads), result == kOk ? "value" : "none", was, Letter(g_preset),
		     static_cast<unsigned long long>(applied));
	}
	return kOk;
}

void* __cdecl QueryInterfaceHook(uint32_t id) {
	void* function = g_query.load(std::memory_order_acquire)(id);
	if (id != kGetSettingInterface || function == nullptr) return function;
	GetSettingFn expected = nullptr;
	if (!g_get_setting.compare_exchange_strong(expected, reinterpret_cast<GetSettingFn>(function)) &&
	    expected != reinterpret_cast<GetSettingFn>(function)) {
		return function; // (A second, different implementation: left alone.)
	}
	return reinterpret_cast<void*>(&GetSettingHook);
}

// Points nvapi64.dll's nvapi_QueryInterface export at QueryInterfaceHook (later GetProcAddress calls and imports
// resolve to it). Before sl.interposer.dll loads.
void Install() {
	const char* value = Env("KYTY_DLSSG_PRESET");
	if (value == nullptr || *value == '\0') return;
	switch (*value) {
		case 'A': case 'a': case '1': g_preset = 1; break;
		case 'B': case 'b': case '2': g_preset = 2; break;
		case 'C': case 'c': case '3': g_preset = 3; break;
		case '0': return; // game / driver
		default: Printf("[dlssg] KYTY_DLSSG_PRESET=%s: not A, B, C or 0, ignored\n", value); return;
	}
	const auto fail = [](const char* why) {
		Printf("[dlssg] preset %c requested: not applied (%s)\n", Letter(g_preset), why);
		g_preset = 0;
	};
	HMODULE nvapi = LoadLibraryExW(L"nvapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	if (nvapi == nullptr) return fail("no nvapi64.dll");
	HMODULE pinned = nullptr; // (Never unloaded: the export now points into our stub.)
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
	                   reinterpret_cast<LPCWSTR>(nvapi), &pinned);
	auto*       base = reinterpret_cast<uint8_t*>(nvapi);
	const auto* dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
	const auto* nt   = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
	const auto& dir  = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
	if (dir.VirtualAddress == 0) return fail("nvapi64.dll has no exports");
	const auto* exports   = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
	const auto* names     = reinterpret_cast<const DWORD*>(base + exports->AddressOfNames);
	const auto* ordinals  = reinterpret_cast<const WORD*>(base + exports->AddressOfNameOrdinals);
	auto*       functions = reinterpret_cast<DWORD*>(base + exports->AddressOfFunctions);
	DWORD*      slot      = nullptr;
	for (DWORD i = 0; i < exports->NumberOfNames; ++i) {
		if (std::strcmp(reinterpret_cast<const char*>(base + names[i]), "nvapi_QueryInterface") == 0) {
			slot = &functions[ordinals[i]];
			break;
		}
	}
	if (slot == nullptr) return fail("no nvapi_QueryInterface");
	auto* original = reinterpret_cast<QueryInterfaceFn>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
	if (original == nullptr || reinterpret_cast<uint8_t*>(original) != base + *slot) return fail("unexpected export");
	// A jump stub above the image, within the 32-bit export RVA range.
	const uintptr_t image_end = reinterpret_cast<uintptr_t>(base) + nt->OptionalHeader.SizeOfImage;
	uint8_t*        stub      = nullptr;
	for (uintptr_t at = (image_end + 0xFFFF) & ~uintptr_t(0xFFFF);
	     stub == nullptr && at < reinterpret_cast<uintptr_t>(base) + 0x7FFF0000u; at += 0x10000) {
		stub = static_cast<uint8_t*>(
		    VirtualAlloc(reinterpret_cast<void*>(at), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	}
	if (stub == nullptr) return fail("no memory near nvapi64.dll");
	const uint64_t target = reinterpret_cast<uint64_t>(&QueryInterfaceHook);
	stub[0] = 0xFF; stub[1] = 0x25; std::memset(stub + 2, 0, 4); // jmp [rip+0]
	std::memcpy(stub + 6, &target, 8);
	DWORD protection = 0;
	VirtualProtect(stub, 4096, PAGE_EXECUTE_READ, &protection);
	FlushInstructionCache(GetCurrentProcess(), stub, 14);
	g_query.store(original, std::memory_order_release);
	if (!VirtualProtect(slot, sizeof(DWORD), PAGE_READWRITE, &protection)) return fail("export table is read-only");
	*slot = static_cast<DWORD>(reinterpret_cast<uintptr_t>(stub) - reinterpret_cast<uintptr_t>(base));
	VirtualProtect(slot, sizeof(DWORD), protection, &protection);
	Printf("[dlssg] preset %c requested: NVAPI DRS read hook installed (driver profile untouched)\n", Letter(g_preset));
	if (GetModuleHandleW(L"nvngx_dlssg.dll") != nullptr) Printf("[dlssg] warning: nvngx_dlssg.dll was already loaded\n");
}

} // namespace DlssgPreset
#endif

void InstallDlssgPreset() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    static bool done = false;
    if (done || Cfg().kind != Kind::Dlss) return;
    done = true;
    if (Env("KYTY_DLSSG_PRESET") == nullptr) SetProcessEnv("BB_DLSSG_PRESET", "C");
    DlssgPreset::Install();
#endif
}

// ---------------------------------------------------------------------------------------------
// DLSS-G
// ---------------------------------------------------------------------------------------------
// BB_FG_PACE=guest (default) | present. The real frames' limit (target / multiplier, or the VRR
// cap) is applied where the game submits its flip, holding the game's own frames to that rate on a
// fixed grid, as Demon's Souls holds its flips (10-23: VideoOut::SetDynamicFlipFps, "game flips at
// 56"); the old place, the present thread, beat against the game's 60 (present interval sd 1.9 ms
// at x4 on 224). present: the old place.
bool GuestPacing() {
    static const bool guest = [] {
        const char* v = std::getenv("BB_FG_PACE");
        return v == nullptr || std::strcmp(v, "present") != 0;
    }();
    return guest;
}

namespace Dlssg {
double PaceInterval();
bool Generating();
void NoteSlept(double seconds);
}

// The adaptive grid (BB_FG_PACE_ADAPT=0: off). With no cap, the real flips are held on a grid at
// the mean free frame time of the last 16 flips: the game's own time between flips, without our
// waits. The real presents then come out evenly spaced, and a flip is never held longer than one
// such frame (added latency at most one frame). A cap (KYTY_VRR_CAP_FPS, dynamic) still applies
// when its interval is longer.
namespace Adapt {
constexpr int kWindow = 16;
constexpr int kMinSamples = 8;
double g_ring[kWindow]{};
int g_count = 0;
int g_next = 0;
std::atomic<double> g_last{0.0}; // the interval the last generated flip was held to (log)

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_FG_PACE_ADAPT");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return on;
}

void Add(double free_seconds) {
    g_ring[g_next] = free_seconds;
    g_next = (g_next + 1) % kWindow;
    ++g_count;
}

// Mean free frame time (0: too few samples yet).
double Mean() {
    if (g_count < kMinSamples) return 0.0;
    const int n = std::min(g_count, kWindow);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += g_ring[i];
    return sum / n;
}
} // namespace Adapt

double PaceEffective() {
    return Adapt::g_last.load();
}

void PaceGuestFlip() {
    if (!GuestPacing()) return;
    using Clock = std::chrono::steady_clock;
    static Clock::time_point next{};
    static Clock::time_point exited{}; // when the previous flip's wait ended
    static std::mutex mutex;
    std::unique_lock lock{mutex};
    const auto arrive = Clock::now();
    const bool generating = Dlssg::Generating() || Amd::Generating();
    if (generating && Adapt::Enabled() && exited != Clock::time_point{}) {
        const double frame_s = std::chrono::duration<double>(arrive - exited).count();
        if (frame_s > 0.0 && frame_s < 0.1) Adapt::Add(frame_s);
    }
    double interval = Dlssg::PaceInterval();
    if (interval <= 0.0) interval = Amd::PaceInterval();
    if (generating && Adapt::Enabled()) interval = std::max(interval, Adapt::Mean());
    Adapt::g_last = interval;
    if (interval <= 0.0 || interval > 0.1) {
        next = Clock::time_point{};
        exited = Clock::now();
        return;
    }
    const auto step =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(interval));
    auto now = arrive;
    if (next > now) {
        const auto until = next;
        lock.unlock();
        std::this_thread::sleep_until(until - std::chrono::milliseconds(1));
        while (Clock::now() < until) std::this_thread::yield();
        lock.lock();
        Amd::NoteSlept(std::chrono::duration<double>(until - now).count());
        Dlssg::NoteSlept(std::chrono::duration<double>(until - now).count());
        now = until;
    }
    // A late flip resets the grid (no burst to catch up).
    next = std::max(next, now - step) + step;
    exited = Clock::now();
}

namespace Dlssg {
namespace {
using Clock = std::chrono::steady_clock;

int g_generate = 0; // frames to generate per rendered frame (the dynamic multiplier's maximum)
bool g_ready = false;
bool g_dynamic = false;
float g_target_fps = 0.0f;
int g_dyn_gen = 1; // the current generated count in dynamic mode
int g_dyn_max = 3; // min(asked, the GPU's numFramesToGenerateMax), 3 until known
double g_dyn_real_fps = 0.0;
double g_slept = 0.0; // seconds our pacer waited in this window
Kyty::Local::DynMultiplier g_dm;
std::atomic<double> g_real_interval{0.0}; // the pacer's real-frame interval (0: none)
bool g_fixed_cap = false;                 // fixed DLSS-G under BB_VRR_CAP_FPS
std::atomic<uint32_t> g_presented_frames{1};
int g_mode_set = -1; // DLSS-G last set on (1) or off (0)
uint32_t g_rx_open = 0;               // the frame begun at the last guest flip
std::atomic<uint32_t> g_rx_present{0}; // the index BeforePresent presents next (SetPresentIndex)

// 10-23 (Demon's Souls): KYTY_DLSSG_MARKERS=guest (default) | present. Reflex's sleep and the
// simulation / render submit markers at the guest flip (it paces the game's frames, not the
// presents); present start / end at the present with the same frame index.
bool GuestMarkers() {
    static const bool guest = [] {
        const char* v = Env("KYTY_DLSSG_MARKERS");
        return v == nullptr || std::strcmp(v, "present") != 0;
    }();
    return guest;
}

int CurrentGenerate() {
    return g_dynamic ? g_dyn_gen : g_generate;
}

#if defined(_WIN32) && defined(BB_STREAMLINE)
struct Api {
    PFun_slGetNewFrameToken* get_new_frame_token = nullptr;
    PFun_slSetConstants* set_constants = nullptr;
    PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
    PFun_slDLSSGSetOptions* dlssg_set_options = nullptr;
    PFun_slDLSSGGetState* dlssg_get_state = nullptr;
    PFun_slReflexSetOptions* reflex_set_options = nullptr;
    PFun_slReflexSleep* reflex_sleep = nullptr;
    PFun_slPCLSetMarker* pcl_set_marker = nullptr;
} g_api;
sl::ReflexOptions g_reflex_opts{};
sl::FrameToken* g_token = nullptr;
uint32_t g_frame_index = 0; // the shared Streamline frame index stream (SR::Dlss::NextFrameIndex)
bool g_off_only = false;    // Streamline loaded for DLSS alone: DLSS-G kept off with valid formats
vk::Extent2D g_off_backbuffer{};
vk::Format g_off_format = vk::Format::eUndefined;
bool g_options_set = false;
vk::Extent2D g_options_backbuffer{}, g_options_render{};
vk::Format g_options_format = vk::Format::eUndefined;
uint64_t g_presents = 0;
Clock::time_point g_present_begin{};
// The per-second line.
Clock::time_point g_stat_start{};
uint32_t g_stat_real = 0, g_stat_on = 0, g_stat_hudless = 0;
uint32_t g_last_status = 0;
double g_stat_present_ms = 0, g_stat_present_max = 0;
double g_pace_n = 0, g_pace_sum = 0, g_pace_sq = 0, g_pace_max = 0;

int PaceMode() { // KYTY_DLSSG_PACE=grid (our pacer, default) | reflex (Reflex's limit only)
    static const int mode = [] {
        const char* v = Env("KYTY_DLSSG_PACE");
        return v != nullptr && std::string(v) == "reflex" ? 2 : 1;
    }();
    return mode;
}

uint32_t ReflexLimitUs(double interval) {
    static const bool force = [] {
        const char* v = Env("KYTY_DLSSG_REFLEX_LIMIT");
        return v != nullptr && *v == '1';
    }();
    return (force || PaceMode() >= 2) && interval > 0.0
               ? static_cast<uint32_t>(1000000.0 * interval)
               : 0u;
}

// The real-frame limit for `gen` generated frames: target / (gen + 1), so the frames shown stay
// under the target (the G-SYNC range). gen < 0: none (DLSS-G off: menus, movies, loading).
void ApplyLimit(int gen) {
    const double interval =
        gen < 0 ? 0.0
                : static_cast<double>(gen + 1) / std::max(static_cast<double>(g_target_fps), 1.0);
    if (g_real_interval.load() != interval) {
        Printf("[dyn] pacer interval %.0f us for x%d", interval * 1e6, gen + 1);
    }
    g_real_interval = PaceMode() >= 2 ? 0.0 : interval;
    const uint32_t us = ReflexLimitUs(interval);
    if (g_reflex_opts.frameLimitUs == us) {
        return;
    }
    g_reflex_opts.frameLimitUs = us;
    if (const auto result = g_api.reflex_set_options(g_reflex_opts); result != sl::Result::eOk) {
        Printf("frame generation: slReflexSetOptions (limit) failed (%d)", static_cast<int>(result));
    }
}

sl::float4x4 Matrix(const float* m) {
    sl::float4x4 r;
    for (uint32_t i = 0; i < 4; ++i) {
        r[i] = sl::float4(m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
    }
    return r;
}

bool Inverted(const float* m, float* out) {
    double a[4][8]{};
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) a[i][j] = m[i * 4 + j];
        a[i][4 + i] = 1.0;
    }
    for (uint32_t c = 0; c < 4; ++c) {
        uint32_t pivot = c;
        for (uint32_t r = c + 1; r < 4; ++r) {
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
        }
        if (std::fabs(a[pivot][c]) < 1e-12) return false;
        for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
        const double scale = 1.0 / a[c][c];
        for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
        for (uint32_t r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<float>(a[i][4 + j]);
    }
    return true;
}

// The same constants the super resolution path gives Streamline (bb_sr_dlss.cpp): our matrices
// are row major for column vectors, Streamline's row-vector, so each is given transposed.
sl::Constants MakeConstants(const Inputs* in) {
    sl::Constants c{};
    const float identity[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const auto transposed = [](const float* m) {
        float t[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) t[j * 4 + i] = m[i * 4 + j];
        return Matrix(t);
    };
    const bool camera = in != nullptr && in->camera_valid;
    float inverse[16]{}, prev_to_clip[16]{};
    c.cameraViewToClip = transposed(camera ? in->view_to_clip : identity);
    c.clipToCameraView = camera && Inverted(in->view_to_clip, inverse) ? transposed(inverse)
                                                                      : Matrix(identity);
    c.clipToLensClip = Matrix(identity);
    c.clipToPrevClip = transposed(camera ? in->clip_to_prev_clip : identity);
    c.prevClipToClip = camera && Inverted(in->clip_to_prev_clip, prev_to_clip)
                           ? transposed(prev_to_clip)
                           : Matrix(identity);
    const vk::Extent2D render = in != nullptr ? in->render : vk::Extent2D{1, 1};
    c.jitterOffset = in != nullptr ? sl::float2(in->jitter[0], in->jitter[1]) : sl::float2(0, 0);
    c.mvecScale = sl::float2(1.0f / static_cast<float>(std::max(render.width, 1u)),
                             1.0f / static_cast<float>(std::max(render.height, 1u)));
    c.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    c.cameraPos = sl::float3(0, 0, 0);
    c.cameraUp = sl::float3(0, 1, 0);
    c.cameraRight = sl::float3(1, 0, 0);
    c.cameraFwd = sl::float3(0, 0, 1);
    c.cameraNear = in != nullptr ? in->z_near : 0.1f;
    c.cameraFar = in != nullptr ? in->z_far : 1000.0f;
    c.cameraFOV = in != nullptr ? in->fov_y : 1.0f;
    c.cameraAspectRatio =
        static_cast<float>(std::max(render.width, 1u)) / static_cast<float>(std::max(render.height, 1u));
    c.motionVectorsInvalidValue = sl::INVALID_FLOAT;
    c.depthInverted = in == nullptr || in->depth_inverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = in == nullptr || in->reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    return c;
}

sl::Resource ToResource(const Image& image) {
    sl::Resource r(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
                   static_cast<VkImageView>(image.view),
                   static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    r.width = image.extent.width;
    r.height = image.extent.height;
    r.nativeFormat = static_cast<uint32_t>(static_cast<VkFormat>(image.format));
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
    return r;
}

template <typename F>
bool Feature(sl::Feature feature, const char* name, F*& function) {
    function = reinterpret_cast<F*>(SR::Dlss::FeatureProc(static_cast<uint32_t>(feature), name));
    if (function == nullptr) {
        Printf("Frame generation: %s is unavailable", name);
        return false;
    }
    return true;
}
#endif

} // namespace

void OnDevice(vk::PhysicalDevice physical) {
    if (Cfg().kind != Kind::Dlss) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
        // DLSS super resolution alone: sl.dlss_g is loaded too (device creation needs it) and
        // checks its formats at every present: kept off with valid formats (KeepOff).
        if (SR::Dlss::Loaded() &&
            SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeatureDLSS_G), physical)) {
            g_api.dlssg_set_options = reinterpret_cast<PFun_slDLSSGSetOptions*>(
                SR::Dlss::FeatureProc(static_cast<uint32_t>(sl::kFeatureDLSS_G), "slDLSSGSetOptions"));
            g_off_only = g_api.dlssg_set_options != nullptr;
        }
#endif
        return;
    }
    g_generate = std::clamp(Cfg().multiplier - 1, 1, 5);
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!SR::Dlss::Loaded()) {
        Printf("Frame generation: NVIDIA's Streamline runtime was not loaded (not found or not "
               "signed by NVIDIA): DLSS-G off");
        return;
    }
    if (Cfg().dynamic) {
        g_dynamic = true;
        if (const char* t = Env("KYTY_DLSSG_TARGET_FPS"); t != nullptr && std::atof(t) > 0) {
            g_target_fps = static_cast<float>(std::atof(t));
            Printf("[dlssg] dynamic target %.0f fps (BB_DLSSG_TARGET_FPS)",
                   static_cast<double>(g_target_fps));
        } else {
            DEVMODEA dm{};
            dm.dmSize = sizeof(dm);
            if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &dm) != 0 &&
                dm.dmDisplayFrequency > 30) {
                const double hz = dm.dmDisplayFrequency;
                g_target_fps = static_cast<float>(std::floor(hz * 14.0 / 15.0 + 0.01));
                Printf("[dlssg] display refresh %u Hz (current mode): dynamic target %.0f fps",
                       static_cast<unsigned>(dm.dmDisplayFrequency),
                       static_cast<double>(g_target_fps));
            } else {
                Printf("[dlssg] display refresh unknown: dynamic target 224 fps");
            }
        }
        if (g_target_fps <= 0.0f) g_target_fps = 224.0f;
        g_dyn_max = std::min(g_generate, 3);
        g_dyn_gen =
            std::clamp(static_cast<int>(std::floor(g_target_fps / 30.0)) - 1, 1, g_dyn_max);
    }
    if (!SR::Dlss::FeatureReady(static_cast<uint32_t>(sl::kFeatureDLSS_G), physical)) {
        Printf("Frame generation: DLSS-G is not supported on this device");
        return;
    }
    g_api.get_new_frame_token =
        reinterpret_cast<PFun_slGetNewFrameToken*>(SR::Dlss::Proc("slGetNewFrameToken"));
    g_api.set_constants = reinterpret_cast<PFun_slSetConstants*>(SR::Dlss::Proc("slSetConstants"));
    g_api.set_tag_for_frame =
        reinterpret_cast<PFun_slSetTagForFrame*>(SR::Dlss::Proc("slSetTagForFrame"));
    if (g_api.get_new_frame_token == nullptr || g_api.set_constants == nullptr ||
        g_api.set_tag_for_frame == nullptr ||
        !Feature(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
        !Feature(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
        !Feature(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
        !Feature(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
        return;
    }
    // slReflexSleep each frame (KYTY_DLSSG_SLEEP=0: not): DLSS-G requires Reflex active.
    Feature(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
    sl::ReflexOptions& reflex = g_reflex_opts;
    reflex = {};
    reflex.mode = sl::ReflexMode::eLowLatency;
    if (const char* mode = Env("KYTY_DLSSG_REFLEX"); mode != nullptr) {
        if (std::string(mode) == "off") reflex.mode = sl::ReflexMode::eOff;
        if (std::string(mode) == "boost") reflex.mode = sl::ReflexMode::eLowLatencyWithBoost;
    }
    // G-SYNC: the frames shown (rendered x (1 + generated)) held under the refresh
    // (KYTY_VRR_CAP_FPS=<shown fps> or auto = refresh x 14 / 15; unset: off).
    {
        uint32_t cap = 0;
        const char* text = Env("KYTY_VRR_CAP_FPS");
        if (text != nullptr && *text != '\0' && std::string(text) != "auto") {
            cap = static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
        } else if (text != nullptr && std::string(text) == "auto") {
            DEVMODEA mode{};
            mode.dmSize = sizeof(mode);
            if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 &&
                mode.dmDisplayFrequency > 30) {
                cap = static_cast<uint32_t>(std::floor(mode.dmDisplayFrequency * 14.0 / 15.0 + 0.01));
            }
        }
        if (g_dynamic) {
            cap = 0;
            g_real_interval =
                static_cast<double>(g_dyn_gen + 1) / static_cast<double>(g_target_fps);
            reflex.frameLimitUs = ReflexLimitUs(g_real_interval);
            Printf("Frame generation: dynamic multiplier, target %.0f fps: Reflex limit %u us (x%d)",
                   static_cast<double>(g_target_fps), reflex.frameLimitUs, g_dyn_gen + 1);
        }
        if (cap != 0) {
            g_target_fps = static_cast<float>(cap);
            g_fixed_cap = true;
            g_real_interval = static_cast<double>(g_generate + 1) / cap;
            reflex.frameLimitUs = ReflexLimitUs(g_real_interval);
            Printf("Frame generation: shown frames held under %u fps (rendered at most %.1f, "
                   "Reflex limit %u us)",
                   cap, static_cast<double>(cap) / (g_generate + 1), reflex.frameLimitUs);
        }
    }
    Printf("Frame generation: Reflex mode %d", static_cast<int>(reflex.mode));
    if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
        Printf("frame generation: slReflexSetOptions failed (%d)", static_cast<int>(result));
    }
    g_ready = true;
    Printf("Frame generation: DLSS-G ready (%d generated frame(s) per frame%s)", g_generate,
           g_dynamic ? ", dynamic" : "");
#else
    (void)physical;
    Printf("Frame generation: this build has no Streamline SDK: DLSS-G off");
#endif
}


bool Ready() {
    return g_ready;
}

double PaceInterval() {
    return g_ready && g_mode_set == 1 ? g_real_interval.load() : 0.0;
}

bool Generating() {
    return g_ready && g_mode_set == 1;
}

void NoteSlept(double seconds) {
    g_slept += seconds;
}

uint32_t GuestFlip() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_ready || !GuestMarkers() || g_api.pcl_set_marker == nullptr) return 0;
    uint32_t done = 0;
    sl::FrameToken* token = nullptr;
    if (g_rx_open != 0) {
        uint32_t index = g_rx_open;
        if (g_api.get_new_frame_token(token, &index) == sl::Result::eOk && token != nullptr) {
            g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, *token);
            done = g_rx_open;
        }
    }
    uint32_t next = SR::Dlss::NextFrameIndex();
    token = nullptr;
    if (g_api.get_new_frame_token(token, &next) != sl::Result::eOk || token == nullptr) {
        g_rx_open = 0;
        return done;
    }
    static const bool sleep = [] {
        const char* value = Env("KYTY_DLSSG_SLEEP");
        return value == nullptr || *value != '0';
    }();
    if (sleep && g_api.reflex_sleep != nullptr) g_api.reflex_sleep(*token);
    g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, *token);
    g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, *token);
    g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, *token);
    g_rx_open = next;
    return done;
#else
    return 0;
#endif
}

void SetPresentIndex(uint32_t index) {
    g_rx_present.store(index, std::memory_order_relaxed);
}

int Generated() {
    return g_ready && g_mode_set == 1 ? CurrentGenerate() : 0;
}

uint32_t PresentedFrames() {
    return g_presented_frames.load(std::memory_order_relaxed);
}

void BeforePresent(const Inputs* inputs, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_ready) {
        return;
    }
    const sl::ViewportHandle viewport{0};
    // Menus, movies and loading screens (no 3D scene upscaled) are presented as they are.
    const bool usable = inputs != nullptr && inputs->valid;
    // The pacer: real frames at most every target / m while generating (one limiter, ours).
    if (const double interval = g_real_interval.load(); interval > 0.0 && usable && !GuestPacing()) {
        static Clock::time_point next{};
        const auto step = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(interval));
        auto t = Clock::now();
        if (next > t) {
            g_slept += std::chrono::duration<double>(next - t).count();
            std::this_thread::sleep_until(next - std::chrono::milliseconds(1));
            while (Clock::now() < next) std::this_thread::yield();
            t = next;
        }
        next = std::max(next, t - step) + step;
    }
    {
        const auto now = Clock::now();
        if (g_stat_start == Clock::time_point{}) g_stat_start = now;
        ++g_stat_real;
        if (usable) ++g_stat_on;
        {
            static Clock::time_point prev{};
            const double dt = prev == Clock::time_point{}
                                  ? 0.0
                                  : std::chrono::duration<double>(now - prev).count();
            prev = now;
            if (usable && dt > 0.0 && dt < 0.1) {
                g_pace_n += 1.0;
                g_pace_sum += dt;
                g_pace_sq += dt * dt;
                g_pace_max = std::max(g_pace_max, dt);
            }
        }
        const double window = std::chrono::duration<double>(now - g_stat_start).count();
        if (window >= 1.0) {
            const double n = g_stat_real != 0 ? static_cast<double>(g_stat_real) : 1.0;
            if (g_pace_n >= 10.0) {
                const double mean = g_pace_sum / g_pace_n;
                const double sd = std::sqrt(std::max(g_pace_sq / g_pace_n - mean * mean, 0.0));
                Printf("[dlssg] present interval sd %.2f ms (mean %.2f, max %.1f, %.0f frames); "
                       "pacer %.2f ms (%s)",
                       sd * 1e3, mean * 1e3, g_pace_max * 1e3, g_pace_n,
                       PaceEffective() * 1e3,
                       PaceMode() == 2 ? "reflex" : GuestPacing() ? "grid, guest flips" : "grid");
            }
            g_pace_n = g_pace_sum = g_pace_sq = g_pace_max = 0.0;
            if (g_dynamic && g_mode_set == 1) {
                const double real = g_stat_real / window;
                const double freer = g_stat_real / std::max(window - g_slept, 0.05);
                g_slept = 0.0;
                g_dyn_real_fps = g_dyn_real_fps <= 0.0 ? freer : 0.5 * g_dyn_real_fps + 0.5 * freer;
                const double T = static_cast<double>(g_target_fps);
                static int cap_state = 0;
                const int maxgen =
                    std::min(g_dyn_max, std::max(DynBaseCap(g_dyn_real_fps, cap_state) - 1, 1));
                const int next = std::clamp(g_dm.Window(g_dyn_gen + 1, real, T, maxgen + 1, false) - 1,
                                            1, std::max(maxgen, 1));
                if (next != g_dyn_gen) {
                    const int from_m = g_dyn_gen + 1;
                    g_dyn_gen = next;
                    g_options_set = false; // send numFramesToGenerate again
                    ApplyLimit(g_dyn_gen);
                    Printf("[dyn] x%d (from x%d): real %.1f fps, shown %.1f (target %.0f), free %.1f",
                           g_dyn_gen + 1, from_m, real, std::min(real * from_m, T), T,
                           g_dm.last_free);
                }
                Printf("[dlssg] dynamic x%d; real %.1f fps (smoothed %.1f), shown %.1f, target %.0f",
                       g_dyn_gen + 1, real, g_dyn_real_fps, real * (g_dyn_gen + 1), T);
            }
            const uint32_t shown_per = g_mode_set == 1 ? PresentedFrames() : 1u;
            Printf("[dlssg] %s x%u; real %.1f fps, shown %.1f fps; generation on %u of %u frames; "
                   "present call %.2f ms (max %.1f); HUD-less %u, state 0x%x",
                   g_dynamic ? "dynamic" : "fixed", shown_per, g_stat_real / window,
                   g_stat_real / window * static_cast<double>(shown_per), g_stat_on, g_stat_real,
                   g_stat_present_ms / n, g_stat_present_max, g_stat_hudless, g_last_status);
            g_stat_hudless = 0;
            g_stat_start = now;
            g_stat_real = 0;
            g_stat_on = 0;
            g_stat_present_ms = g_stat_present_max = 0;
        }
    }
    // DLSS-G on or off (its resources kept, eRetainResourcesWhenOff): off after 30 presents in a
    // row without inputs (menus, movies, loading), on after 10 with them, so a fade or a cut does
    // not flap it.
    static uint32_t run_with = 0, run_without = 0;
    run_with = usable ? run_with + 1 : 0;
    run_without = usable ? 0 : run_without + 1;
    const bool mode_on = g_mode_set == 1 ? run_without < 30 : run_with >= 10;
    if (g_dynamic) {
        ApplyLimit(mode_on ? g_dyn_gen : -1);
    } else if (g_fixed_cap) {
        ApplyLimit(mode_on ? g_generate : -1);
    }
    if (!g_options_set || g_mode_set != (mode_on ? 1 : 0) || g_options_format != format ||
        (usable && (g_options_backbuffer != backbuffer || g_options_render != inputs->render))) {
        sl::DLSSGOptions options{};
        options.mode = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
        options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
        {
            static bool once = false;
            if (!once) {
                once = true;
                if (g_dynamic) {
                    Printf("[dlssg] mode: app-side dynamic multiplier (eOn; eDynamic is D3D12 only), "
                           "up to %d generated, target %.0f fps, starting x%d",
                           g_generate, static_cast<double>(g_target_fps), g_dyn_gen + 1);
                } else {
                    Printf("[dlssg] mode: fixed x%d (eOn)", g_generate + 1);
                }
            }
        }
        if (g_mode_set != (mode_on ? 1 : 0)) {
            Printf("DLSS-G: %s", mode_on ? "on (3D frame: depth, motion vectors, camera)"
                                         : "off (no 3D frame: menu, loading or movie)");
        }
        g_mode_set = mode_on ? 1 : 0;
        options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
        static const bool parallel = [] {
            const char* value = Env("KYTY_DLSSG_PARALLEL");
            return value != nullptr && *value == '1';
        }();
        if (parallel) {
            options.queueParallelismMode = sl::DLSSGQueueParallelismMode::eBlockNoClientQueues;
        }
        options.numBackBuffers = image_count;
        options.colorWidth = backbuffer.width;
        options.colorHeight = backbuffer.height;
        options.colorBufferFormat = static_cast<uint32_t>(static_cast<VkFormat>(format));
        // Every format defined, inputs or not (Streamline: "Cannot have undefined format" and no
        // generation otherwise): the inputs' own, else what they will be (D32S8 depth, RG16F
        // vectors, the HUD-less image in the backbuffer's format).
        options.mvecDepthWidth = g_options_render.width != 0 ? g_options_render.width : backbuffer.width;
        options.mvecDepthHeight =
            g_options_render.height != 0 ? g_options_render.height : backbuffer.height;
        options.mvecBufferFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
        options.depthBufferFormat = static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT);
        options.hudLessBufferFormat = static_cast<uint32_t>(static_cast<VkFormat>(format));
        if (usable) {
            options.mvecDepthWidth = inputs->render.width;
            options.mvecDepthHeight = inputs->render.height;
            if (inputs->motion.format != vk::Format::eUndefined) {
                options.mvecBufferFormat =
                    static_cast<uint32_t>(static_cast<VkFormat>(inputs->motion.format));
            }
            if (inputs->depth.format != vk::Format::eUndefined) {
                options.depthBufferFormat =
                    static_cast<uint32_t>(static_cast<VkFormat>(inputs->depth.format));
            }
            if (inputs->has_hudless && inputs->hudless.format != vk::Format::eUndefined) {
                options.hudLessBufferFormat =
                    static_cast<uint32_t>(static_cast<VkFormat>(inputs->hudless.format));
            }
            g_options_render = inputs->render;
        }
        if (const auto result = g_api.dlssg_set_options(viewport, options);
            result != sl::Result::eOk) {
            Printf("frame generation: slDLSSGSetOptions failed (%d)", static_cast<int>(result));
        }
        g_options_set = true;
        g_options_backbuffer = backbuffer;
        g_options_format = format;
    }

    // 10-23 (Demon's Souls): a frame begun at the guest flip (GuestFlip) keeps its index; others
    // (menus, a repeated image) take a new one with all markers here, as before.
    const uint32_t guest_index = g_rx_present.exchange(0, std::memory_order_relaxed);
    {
        const uint32_t own = guest_index != 0 ? guest_index
                             : usable && inputs->sl_frame > g_frame_index ? inputs->sl_frame
                                                                          : SR::Dlss::NextFrameIndex();
        g_frame_index = std::max(own, g_frame_index + 1);
    }
    if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
        g_token = nullptr;
        return;
    }
    auto& token = *g_token;
    static const bool sleep = [] {
        const char* value = Env("KYTY_DLSSG_SLEEP");
        return value == nullptr || *value != '0';
    }();
    if (guest_index == 0) {
        if (sleep && g_api.reflex_sleep != nullptr && !GuestMarkers()) g_api.reflex_sleep(token);
        g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
        g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
        g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);
    }

    sl::Constants constants = MakeConstants(usable ? inputs : nullptr);
    if (const auto result = g_api.set_constants(constants, token, viewport);
        result != sl::Result::eOk) {
        static uint32_t failures = 0;
        if (failures++ < 8 || failures % 600 == 0) {
            Printf("[dlssg] slSetConstants failed (%d), %u time(s)", static_cast<int>(result),
                   failures);
        }
    }
    if (usable) {
        sl::Resource depth = ToResource(inputs->depth);
        sl::Resource motion = ToResource(inputs->motion);
        sl::Resource hudless = ToResource(inputs->hudless);
        const sl::Extent full{0, 0, inputs->render.width, inputs->render.height};
        // sl::Extent is {left, top, width, height}: the game image's offset is x (left), y (top).
        const sl::Extent game{static_cast<uint32_t>(region.offset.x),
                              static_cast<uint32_t>(region.offset.y), region.extent.width,
                              region.extent.height};
        const bool partial = region.offset.x != 0 || region.offset.y != 0 ||
                             region.extent.width != backbuffer.width ||
                             region.extent.height != backbuffer.height;
        const sl::Extent hudless_extent{0, 0, inputs->hudless.extent.width,
                                        inputs->hudless.extent.height};
        std::vector<sl::ResourceTag> tags{
            sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent,
                            &full),
            sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors,
                            sl::ResourceLifecycle::eValidUntilPresent, &full),
        };
        if (partial) {
            tags.emplace_back(nullptr, sl::kBufferTypeBackbuffer,
                              sl::ResourceLifecycle::eValidUntilPresent, &game);
        }
        static const bool hudless_tag = [] {
            const char* value = Env("KYTY_FRAMEGEN_HUDLESS");
            return value == nullptr || *value != '0';
        }();
        // The HUD-less image is the game's image (the window size, output size), which sits in
        // the backbuffer at region: it can be smaller than the backbuffer (2556 of 2560 wide).
        const bool tag_hudless = hudless_tag && inputs->has_hudless &&
                                 inputs->hudless.extent.width == region.extent.width &&
                                 inputs->hudless.extent.height == region.extent.height;
        if (tag_hudless) {
            ++g_stat_hudless;
            tags.emplace_back(&hudless, sl::kBufferTypeHUDLessColor,
                              sl::ResourceLifecycle::eValidUntilPresent, &hudless_extent);
        }
        const auto tag_result = g_api.set_tag_for_frame(token, viewport, tags.data(),
                                                        static_cast<uint32_t>(tags.size()), nullptr);
        static uint32_t tag_failures = 0;
        if (tag_result != sl::Result::eOk && (tag_failures++ < 8 || tag_failures % 600 == 0)) {
            Printf("[dlssg] slSetTagForFrame failed (%d), %u time(s)", static_cast<int>(tag_result),
                   tag_failures);
        }
        static bool tags_dumped = false;
        if (!tags_dumped) {
            // Once per run: the format and view of each tag as Streamline gets it.
            tags_dumped = true;
            const auto show = [](const char* name, const Image& img) {
                Printf("[dlssg] tag %s: format %u, image %s, view %s, %ux%u", name,
                       static_cast<uint32_t>(static_cast<VkFormat>(img.format)),
                       static_cast<VkImage>(img.image) != VK_NULL_HANDLE ? "yes" : "NULL",
                       static_cast<VkImageView>(img.view) != VK_NULL_HANDLE ? "yes" : "NULL",
                       img.extent.width, img.extent.height);
            };
            show("depth", inputs->depth);
            show("motion", inputs->motion);
            show("hudless", inputs->hudless);
            Printf("[dlssg] tag hudless: tagged %s, options hudless format %u, backbuffer format %u",
                   tag_hudless ? "yes" : "no",
                   static_cast<uint32_t>(static_cast<VkFormat>(inputs->hudless.format)),
                   static_cast<uint32_t>(static_cast<VkFormat>(format)));
        }
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            Printf("[dlssg] inputs: frame %u camera %s, mvecScale %.6g %.6g jitter %.4g %.4g near "
                   "%.4g far %.4g fov %.4g depthInverted %d | depth %ux%u fmt %u, motion %ux%u fmt "
                   "%u, hudless %s (%s) %ux%u, backbuffer %ux%u, region %d,%d %ux%u, setTag %d",
                   g_frame_index, inputs->camera_valid ? "valid" : "INVALID",
                   constants.mvecScale.x, constants.mvecScale.y, constants.jitterOffset.x,
                   constants.jitterOffset.y, constants.cameraNear, constants.cameraFar,
                   constants.cameraFOV, static_cast<int>(constants.depthInverted), depth.width,
                   depth.height, depth.nativeFormat, motion.width, motion.height,
                   motion.nativeFormat, tag_hudless ? "tagged" : "not tagged", inputs->hudless_source,
                   hudless.width,
                   hudless.height, backbuffer.width, backbuffer.height, region.offset.x,
                   region.offset.y, region.extent.width, region.extent.height,
                   static_cast<int>(tag_result));
        }
    } else {
        // No 3D frame: no tags at all. The former null depth and motion tags had no format
        // (Streamline: "Cannot have undefined format", about 2 a frame on menus and loads).
        static uint32_t empty_failures = 0;
        const auto empty_result =
            g_api.set_tag_for_frame(token, viewport, nullptr, 0, nullptr);
        if (empty_result != sl::Result::eOk && empty_failures++ < 4) {
            Printf("[dlssg] empty tag set refused (%d)", static_cast<int>(empty_result));
        }
    }
    if (guest_index == 0) g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
    g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
    g_present_begin = Clock::now();
#else
    (void)inputs;
    (void)backbuffer;
    (void)region;
    (void)format;
    (void)image_count;
#endif
}

void AfterPresent() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (g_present_begin != Clock::time_point{}) {
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - g_present_begin).count();
        g_stat_present_ms += ms;
        g_stat_present_max = std::max(g_stat_present_max, ms);
        g_present_begin = {};
    }
    if (!g_ready || g_token == nullptr) {
        return;
    }
    g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
    const auto presents = ++g_presents;
    if (presents % 30 != 1) {
        return;
    }
    sl::DLSSGState state{};
    if (const auto result = g_api.dlssg_get_state(sl::ViewportHandle{0}, state, nullptr);
        result != sl::Result::eOk) {
        if (presents < 300) Printf("[dlssg] slDLSSGGetState failed (%d)", static_cast<int>(result));
        return;
    }
    {
        static uint32_t last_status = ~0u;
        const auto status = static_cast<uint32_t>(state.status);
        g_last_status = status;
        if (status != last_status || presents < 120) {
            last_status = status;
            Printf("[dlssg] state: status 0x%x, presented %u, generate max %u", status,
                   state.numFramesActuallyPresented, state.numFramesToGenerateMax);
        }
    }
    g_presented_frames.store(std::max(state.numFramesActuallyPresented, 1u),
                             std::memory_order_relaxed);
    // More generated frames asked for than the GPU makes (x5 / x6 off a 50 series): as many as
    // it makes.
    if (state.numFramesToGenerateMax != 0 &&
        static_cast<uint32_t>(g_generate) > state.numFramesToGenerateMax) {
        Printf("Frame generation: %d generated frame(s) asked for, the GPU makes up to %u",
               g_generate, state.numFramesToGenerateMax);
        g_generate = static_cast<int>(state.numFramesToGenerateMax);
        g_options_set = false;
    }
    if (g_dynamic && state.numFramesToGenerateMax != 0) {
        const int max = std::min(g_generate, static_cast<int>(state.numFramesToGenerateMax));
        if (max != g_dyn_max) {
            Printf("[dyn] up to x%d (GPU makes up to %u generated)", max + 1,
                   state.numFramesToGenerateMax);
            g_dyn_max = max;
            if (g_dyn_gen > max) {
                g_dyn_gen = max;
                g_options_set = false;
                ApplyLimit(g_dyn_gen);
            }
        }
    }
    if (presents % 600 == 1) {
        Printf("Frame generation: status=0x%x presented=%u max_generated=%u vram=%llu MiB",
               static_cast<uint32_t>(state.status), state.numFramesActuallyPresented,
               state.numFramesToGenerateMax,
               static_cast<unsigned long long>(state.estimatedVRAMUsageInBytes >> 20u));
    }
#endif
}

void KeepOff(vk::Extent2D backbuffer, vk::Format format, uint32_t image_count) {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    if (!g_off_only || (g_off_backbuffer == backbuffer && g_off_format == format)) {
        return;
    }
    sl::DLSSGOptions options{};
    options.mode = sl::DLSSGMode::eOff;
    options.numBackBuffers = image_count;
    options.colorWidth = options.mvecDepthWidth = backbuffer.width;
    options.colorHeight = options.mvecDepthHeight = backbuffer.height;
    options.colorBufferFormat = options.hudLessBufferFormat =
        static_cast<uint32_t>(static_cast<VkFormat>(format));
    options.mvecBufferFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
    options.depthBufferFormat = static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT);
    const auto result = g_api.dlssg_set_options(sl::ViewportHandle{0}, options);
    Printf("DLSS-G: kept off (DLSS super resolution only), formats given (%d)",
           static_cast<int>(result));
    g_off_backbuffer = backbuffer;
    g_off_format = format;
#else
    (void)backbuffer;
    (void)format;
    (void)image_count;
#endif
}

void SwapchainChanging() {
#if defined(_WIN32) && defined(BB_STREAMLINE)
    g_off_backbuffer = {};
    if (!g_ready || g_mode_set != 1) {
        return;
    }
    sl::DLSSGOptions options{};
    options.mode = sl::DLSSGMode::eOff;
    options.flags = sl::DLSSGFlags::eRetainResourcesWhenOff;
    options.numFramesToGenerate = static_cast<uint32_t>(CurrentGenerate());
    if (const auto result = g_api.dlssg_set_options(sl::ViewportHandle{0}, options);
        result != sl::Result::eOk) {
        Printf("frame generation: slDLSSGSetOptions (off) failed (%d)", static_cast<int>(result));
    }
    g_mode_set = 0;
    g_options_set = false;
#endif
}

} // namespace Dlssg

// ---------------------------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------------------------
namespace {
bool g_antilag_ext = false;
bool g_reflex_ext = false;
bool g_present_id_ext = false;
VkPhysicalDevicePresentIdFeaturesKHR g_present_id{};

bool Has(const std::vector<std::string>& available, const char* name) {
    return std::find(available.begin(), available.end(), name) != available.end();
}
bool Enabled(const std::vector<const char*>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](const char* e) { return std::strcmp(e, name) == 0; });
}
} // namespace

bool DlssgEnabled() {
    return Dlssg::Ready();
}

int DlssgGenerated() {
    return Dlssg::Generated();
}

uint32_t GuestFlip() {
    return Dlssg::GuestFlip();
}

void SetPresentIndex(uint32_t index) {
    Dlssg::SetPresentIndex(index);
}

bool WantInputs() {
    return Dlssg::Ready() || Amd::Active();
}

namespace {
// BB_ANTILAG=auto|amd2|reflex|normal|ultra|off and BB_ANTILAG_DLSSG=reflex|normal|ultra: the Demon's
// Souls port's antilag_modo / antilag_dlssg (src/port/ds-settings.cpp, 10-24), resolved here once,
// before the latency module reads its settings, into the KYTY_ names that module reads (BB_ first):
//  - DLSS-G: reflex (default) = Streamline's Reflex; normal / ultra = ours with Reflex off
//    (KYTY_DLSSG_REFLEX=off, KYTY_LOWLAT=1, KYTY_LOWLAT_QUEUE 2 / 1).
//  - else: amd2 = AMD Anti-Lag 2 (KYTY_ANTILAG=1, VK_AMD_anti_lag); reflex = NVIDIA Reflex
//    (KYTY_ANTILAG=1, VK_NV_low_latency2); normal = ours, 2 frames queued; ultra (Anti-Lag+) = ours,
//    1 frame queued (KYTY_LOWLAT=1, KYTY_LOWLAT_QUEUE=2 / 1); off = none.
//  - auto: Anti-Lag 2 where VK_AMD_anti_lag is (AMD); with AMD FSR 3 frame generation on any other
//    GPU ours (normal), as the DS rule; without frame generation NVIDIA Reflex where
//    VK_NV_low_latency2 is, else ours (normal).
// The launcher's fixed Souls modes (Ligero / Equilibrado) pass ultra and BB_PAD_LATE_MS (Anti-Lag+).
// Unset BB_ANTILAG (or the old numeric 0 / 1 / 2): unchanged, the earlier behaviour.
std::string g_antilag_label;

void ResolveAntiLag(const std::vector<std::string>& available) {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    const auto lower = [](const char* v) {
        std::string s = v != nullptr ? v : "";
        for (auto& ch : s) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return s;
    };
    const std::string mode = lower(std::getenv("BB_ANTILAG"));
    const std::string dlssg = lower(std::getenv("BB_ANTILAG_DLSSG"));
    const bool has_al2 = Has(available, "VK_AMD_anti_lag");
    const bool has_reflex = Has(available, VK_NV_LOW_LATENCY_2_EXTENSION_NAME) &&
                            Has(available, VK_KHR_PRESENT_ID_EXTENSION_NAME);
    const auto own = [&](bool ultra, bool dlss) {
        SetProcessEnv("BB_LOWLAT", "1");
        SetProcessEnv("BB_LOWLAT_QUEUE", ultra ? "1" : "2");
        if (dlss) {
            SetProcessEnv("BB_DLSSG_REFLEX", "off");
        }
    };
    const bool dlss_fg = Cfg().kind == Kind::Dlss;
    const bool amd_fg = Cfg().kind == Kind::Amd;
    if (dlss_fg) {
        if (dlssg == "normal" || dlssg == "ultra") {
            own(dlssg == "ultra", true);
            SetProcessEnv("BB_ANTILAG", "0");
            g_antilag_label = dlssg == "ultra" ? "Anti-Lag+ (ultra, ours: 1 frame queued; Reflex off)"
                                               : "Anti-Lag (normal, ours: 2 frames queued; Reflex off)";
        } else if (!mode.empty() || !dlssg.empty()) {
            SetProcessEnv("BB_ANTILAG", "0"); // (Streamline's own Reflex)
            g_antilag_label = "Reflex (DLSS-G, Streamline)";
        }
    } else if (!mode.empty() && mode != "0" && mode != "1" && mode != "2") {
        std::string m = mode == "no" ? "off" : mode;
        if (m == "auto") {
            m = has_al2 ? "amd2" : (amd_fg || !has_reflex) ? "normal" : "reflex";
        }
        if (m == "amd2" || m == "reflex") {
            SetProcessEnv("BB_ANTILAG", "1");
            g_antilag_label = m == "amd2" ? (has_al2 ? "AMD Anti-Lag 2" : "AMD Anti-Lag 2 (VK_AMD_anti_lag missing)")
                                          : (has_reflex ? "NVIDIA Reflex" : "NVIDIA Reflex (VK_NV_low_latency2 missing)");
        } else if (m == "normal" || m == "ultra") {
            own(m == "ultra", false);
            SetProcessEnv("BB_ANTILAG", "0");
            g_antilag_label = m == "ultra" ? "Anti-Lag+ (ultra, ours: 1 frame queued)"
                                           : "Anti-Lag (normal, ours: 2 frames queued)";
        } else {
            SetProcessEnv("BB_ANTILAG", "0");
            g_antilag_label = "off";
        }
        if (mode == "auto") {
            g_antilag_label = "auto -> " + g_antilag_label;
        }
    }
    if (!g_antilag_label.empty()) {
        const char* pad = Env("KYTY_PAD_LATE_MS");
        BBLogf("Anti-Lag mode: %s (BB_ANTILAG=%s, BB_ANTILAG_DLSSG=%s, frame generation %s%s%s)",
               g_antilag_label.c_str(), mode.empty() ? "-" : mode.c_str(),
               dlssg.empty() ? "-" : dlssg.c_str(), Name(), pad != nullptr ? ", pad late ms " : "",
               pad != nullptr ? pad : "");
    }
}
} // namespace

void AddDeviceExtensions(const std::vector<std::string>& available,
                         std::vector<const char*>& extensions) {
    ResolveAntiLag(available);
    const auto add = [&](const char* name) {
        if (!Has(available, name)) {
            return false;
        }
        if (!Enabled(extensions, name)) {
            extensions.push_back(name);
            BBLogf("Frame generation / latency: device extension %s added", name);
        }
        return true;
    };
    if (Amd::Requested()) {
        // The FidelityFX backend calls vkGetBufferMemoryRequirements2KHR by name.
        add(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
        add(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
    if (BB::AntiLag::Requested() && !StreamlineLoaded()) {
        g_antilag_ext = add(BB::AntiLag::ExtensionName());
        // NVIDIA Reflex without Streamline (VK_NV_low_latency2 needs present id).
        if (Has(available, VK_NV_LOW_LATENCY_2_EXTENSION_NAME) &&
            Has(available, VK_KHR_PRESENT_ID_EXTENSION_NAME)) {
            g_present_id_ext = !Enabled(extensions, VK_KHR_PRESENT_ID_EXTENSION_NAME);
            add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
            g_reflex_ext = add(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
        }
    }
}

void* ChainDeviceFeatures(const std::vector<const char*>& extensions, void* next) {
    next = BB::AntiLag::ChainDeviceFeatures(g_antilag_ext && Enabled(extensions, "VK_AMD_anti_lag"),
                                            next);
    if (g_present_id_ext) {
        g_present_id = {};
        g_present_id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR;
        g_present_id.presentId = VK_TRUE;
        g_present_id.pNext = next;
        next = &g_present_id;
    }
    return next;
}

QueuePlan PlanQueues(vk::PhysicalDevice physical, uint32_t graphics_family) {
    return Amd::PlanQueues(physical, graphics_family);
}

void OnDevice(vk::Instance, vk::PhysicalDevice physical, vk::Device device,
              uint32_t graphics_family, vk::Queue graphics_queue, const QueuePlan& plan,
              std::mutex* submit_mutex) {
    Amd::OnDevice(physical, device, graphics_family, graphics_queue, plan, submit_mutex);
    Dlssg::OnDevice(physical);
    BB::AntiLag::OnDevice(device, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
                          g_antilag_ext, g_reflex_ext);
}

bool CreateSwapchain(const vk::SwapchainCreateInfoKHR& info, vk::SwapchainKHR* out) {
    return Amd::CreateSwapchain(info, out);
}

void DestroySwapchain(vk::SwapchainKHR) {
    Amd::DestroySwapchain();
}

vk::Result GetSwapchainImages(vk::SwapchainKHR swapchain, uint32_t* count, vk::Image* images) {
    return Amd::GetSwapchainImages(swapchain, count, images);
}

vk::Result AcquireNextImage(vk::SwapchainKHR swapchain, vk::Semaphore semaphore, uint32_t* index) {
    return Amd::AcquireNextImage(swapchain, semaphore, index);
}

vk::Result QueuePresent(vk::Queue queue, const vk::PresentInfoKHR& info) {
    return Amd::QueuePresent(queue, info);
}

void SwapchainChanging() {
    Dlssg::SwapchainChanging();
}

void* ChainSwapchain(void* next) {
    return BB::AntiLag::ChainSwapchain(next, Amd::Requested());
}

void OnSwapchain(vk::SwapchainKHR swapchain) {
    BB::AntiLag::OnSwapchain(static_cast<VkSwapchainKHR>(swapchain), Amd::Active());
}

void BeforePresent(const Inputs* inputs, vk::SwapchainKHR swapchain, vk::Extent2D backbuffer,
                   vk::Format format, vk::Rect2D region, uint32_t image_count) {
    if (Dlssg::Ready()) {
        Dlssg::BeforePresent(inputs, backbuffer, region, format, image_count);
    } else {
        Dlssg::KeepOff(backbuffer, format, image_count);
    }
    if (Amd::Active()) {
        Amd::BeforePresent(inputs, swapchain, backbuffer, format);
    }
}

void AfterPresent() {
    Dlssg::AfterPresent();
}

} // namespace BB::FG
