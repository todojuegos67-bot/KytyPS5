// DLSS 5 Neural Rendering on Vulkan with the foveated working grid (ds-foveated-1029). dlss-nr.h has the switches;
// dlss-nr.comp the warp, the proxy's encoding and the ratio resolve.
//
// Where it comes from:
// - The model, its feature id (18) and its parameter contract (DLSSNR.* keys, create-time steering, the scaling-ratio
//   callback, a parameter block the host owns, the caller-identity check the runtime makes on its caller's module and the
//   import patch that answers it) follow RenoDX's DLSS 5 add-on (github.com/clshortfuse/renodx, MIT, Carlos Lopez Jr.) as
//   FoveatedDLSS5 carries it (src/addons/dlss5: dlssnr.hpp, bridge_params.hpp). That add-on drives the runtime's
//   NVSDK_NGX_D3D12_* exports; nvngx_dlssnr.dll also exports the NVSDK_NGX_VULKAN_* set (Init_Ext, CreateFeature,
//   EvaluateFeature, ReleaseFeature, ...: OptiScaler_DLSSNR's notes, which also name the two vendor extensions NGX's
//   Vulkan path needs), so here the same contract goes through those, with Vulkan resources (NVSDK_NGX_Resource_VK).
//   Signatures: the NGX SDK's nvsdk_ngx_vk.h, its NGX_SNIPPET_BUILD forms (the core-to-snippet ones, as the D3D12 path).
// - The foveated grid is FoveatedDLSS5's (github.com/maximedgoulet/FoveatedDLSS5, MIT): foveation.hpp's per-axis
//   density, fall-off and edge, its < 5 % stand-down, its warped guides (fov_guides) and its warp-aware resolve, ported
//   from HLSL / D3D12 to GLSL / Vulkan (dlss-nr.comp) and to pixel units: the working extent keeps the output's parity
//   and the edge density is solved so the grid ends exactly at the screen edge (the plateau is texel-exact both ways).
// Credit and licences: LICENSES/FoveatedDLSS5-RenoDX-MIT.txt. No NVIDIA file is shipped: nvngx_dlssnr.dll is the
// user's (NVIDIA driver / app packages carry it); without it this does nothing and says so once.
#include "local/local-pass-cache.h" // (ds-perf-1031: the passes' pipelines in the disk cache)
#include "local/dlss-nr.h"

#include "graphics/presentation/videoOut.h"
#include "gpu_tiler_shaders/dlss_nr_spv.h"
#include "local/local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

struct ID3D11Resource;
struct ID3D12Resource;

namespace Libs::Graphics::FrameGen {
namespace {

// ---- NGX's ABI, declared here (no NGX SDK in the build) ----
// NVSDK_NGX_Parameter: the same virtual functions in the same order as nvsdk_ngx_params.h, so the same vtable under the
// MSVC ABI (clang-cl) the runtime was built with. No destructor (it would add a slot).
struct NgxParameter {
	virtual void     Set(const char* name, unsigned long long value)       = 0;
	virtual void     Set(const char* name, float value)                    = 0;
	virtual void     Set(const char* name, double value)                   = 0;
	virtual void     Set(const char* name, unsigned int value)             = 0;
	virtual void     Set(const char* name, int value)                      = 0;
	virtual void     Set(const char* name, ID3D11Resource* value)          = 0;
	virtual void     Set(const char* name, ID3D12Resource* value)          = 0;
	virtual void     Set(const char* name, void* value)                    = 0;
	virtual uint32_t Get(const char* name, unsigned long long* out) const  = 0;
	virtual uint32_t Get(const char* name, float* out) const               = 0;
	virtual uint32_t Get(const char* name, double* out) const              = 0;
	virtual uint32_t Get(const char* name, unsigned int* out) const        = 0;
	virtual uint32_t Get(const char* name, int* out) const                 = 0;
	virtual uint32_t Get(const char* name, ID3D11Resource** out) const     = 0;
	virtual uint32_t Get(const char* name, ID3D12Resource** out) const     = 0;
	virtual uint32_t Get(const char* name, void** out) const               = 0;
	virtual void     Reset()                                               = 0;
};
constexpr uint32_t NgxSuccess = 0x1, NgxFailMissingInput = 0xBAD0000Au, NgxFailInvalidParameter = 0xBAD00005u;
constexpr int      NgxVersionApi = 0x15;       // NVSDK_NGX_Version_API (1.5.0)
constexpr int      NgxFeatureNr  = 18;         // NVSDK_NGX_Feature_Reserved18: DLSS Neural Rendering
constexpr int      NgxFlagDepthInverted = 1 << 3; // NVSDK_NGX_DLSS_Feature_Flags_DepthInverted
// The application id RenoDX's add-on initializes the runtime with (dlssnr.hpp kDirectApplicationId).
constexpr unsigned long long NgxApplicationId = 0x876232cULL;

// A parameter block the host owns (RenoDX bridge_params.hpp OwnedParameters): a number is stored once and read back as
// any numeric type, a pointer as any pointer type; a 64-bit value is also a pointer (resources set through the 64-bit
// setter); a missing key answers FAIL_MissingInput.
class OwnedParameters final : public NgxParameter {
public:
	void Set(const char* n, unsigned long long v) override { StoreInt(n, v, static_cast<double>(v), true); }
	void Set(const char* n, float v) override { StoreNum(n, v); }
	void Set(const char* n, double v) override { StoreNum(n, v); }
	void Set(const char* n, unsigned int v) override { StoreInt(n, v, static_cast<double>(v), false); }
	void Set(const char* n, int v) override { StoreInt(n, static_cast<unsigned long long>(static_cast<long long>(v)), static_cast<double>(v), false); }
	void Set(const char* n, ID3D11Resource* v) override { StorePtr(n, v); }
	void Set(const char* n, ID3D12Resource* v) override { StorePtr(n, v); }
	void Set(const char* n, void* v) override { StorePtr(n, v); }
	uint32_t Get(const char* n, unsigned long long* o) const override { return LoadNum(n, o); }
	uint32_t Get(const char* n, float* o) const override { return LoadNum(n, o); }
	uint32_t Get(const char* n, double* o) const override { return LoadNum(n, o); }
	uint32_t Get(const char* n, unsigned int* o) const override { return LoadNum(n, o); }
	uint32_t Get(const char* n, int* o) const override { return LoadNum(n, o); }
	uint32_t Get(const char* n, ID3D11Resource** o) const override { return LoadPtr(n, o); }
	uint32_t Get(const char* n, ID3D12Resource** o) const override { return LoadPtr(n, o); }
	uint32_t Get(const char* n, void** o) const override { return LoadPtr(n, o); }
	void     Reset() override { m_entries.clear(); }

private:
	struct Entry {
		double             number = 0.0;
		unsigned long long bits   = 0;
		void*              pointer = nullptr;
		bool               has_number = false, integral = false, has_pointer = false;
	};
	void StoreInt(const char* n, unsigned long long bits, double number, bool pointer_too) {
		if (n == nullptr) return;
		auto& e      = m_entries[n];
		e.has_number = true;
		e.integral   = true;
		e.bits       = bits;
		e.number     = number;
		if (pointer_too) {
			e.has_pointer = true;
			e.pointer     = reinterpret_cast<void*>(static_cast<uintptr_t>(bits));
		}
	}
	void StoreNum(const char* n, double v) {
		if (n == nullptr) return;
		auto& e      = m_entries[n];
		e.has_number = true;
		e.integral   = false;
		e.number     = v;
		e.bits       = static_cast<unsigned long long>(static_cast<long long>(v));
	}
	void StorePtr(const char* n, void* v) {
		if (n == nullptr) return;
		auto& e       = m_entries[n];
		e.has_pointer = true;
		e.pointer     = v;
		e.has_number  = true;
		e.integral    = true;
		e.bits        = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(v));
		e.number      = static_cast<double>(e.bits);
	}
	template <typename T>
	uint32_t LoadNum(const char* n, T* o) const {
		if (n == nullptr || o == nullptr) return NgxFailInvalidParameter;
		const auto it = m_entries.find(n);
		if (it == m_entries.end() || !it->second.has_number) return NgxFailMissingInput;
		if constexpr (std::is_same_v<T, unsigned long long>) {
			*o = it->second.integral ? it->second.bits : static_cast<unsigned long long>(it->second.number);
		} else {
			*o = static_cast<T>(it->second.number);
		}
		return NgxSuccess;
	}
	template <typename T>
	uint32_t LoadPtr(const char* n, T** o) const {
		if (n == nullptr || o == nullptr) return NgxFailInvalidParameter;
		const auto it = m_entries.find(n);
		if (it == m_entries.end() || !it->second.has_pointer) return NgxFailMissingInput;
		*o = static_cast<T*>(it->second.pointer);
		return NgxSuccess;
	}
	std::map<std::string, Entry, std::less<>> m_entries;
};

// NVSDK_NGX_Resource_VK (nvsdk_ngx_defs_vk.h).
struct NgxImageViewInfoVk {
	VkImageView             ImageView;
	VkImage                 Image;
	VkImageSubresourceRange SubresourceRange;
	VkFormat                Format;
	unsigned int            Width;
	unsigned int            Height;
};
struct NgxBufferInfoVk {
	VkBuffer     Buffer;
	unsigned int SizeInBytes;
};
struct NgxResourceVk {
	union {
		NgxImageViewInfoVk ImageViewInfo;
		NgxBufferInfoVk    BufferInfo;
	} Resource;
	int  Type; // 0: image view
	bool ReadWrite;
};

#if defined(_WIN32)
#define KYTY_NGX_CONV __cdecl
#else
#define KYTY_NGX_CONV
#endif
// The runtime's Vulkan exports, NGX_SNIPPET_BUILD forms (nvsdk_ngx_vk.h).
using InitExtFn  = uint32_t(KYTY_NGX_CONV*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, int,
                                           const NgxParameter*);
using CreateFn   = uint32_t(KYTY_NGX_CONV*)(VkCommandBuffer, int, NgxParameter*, void**);
using EvaluateFn = uint32_t(KYTY_NGX_CONV*)(VkCommandBuffer, const void*, const NgxParameter*, void*);
using ReleaseFn  = uint32_t(KYTY_NGX_CONV*)(void*);

// The scaling-ratio callback the runtime asks a create for (RenoDX DlssNrScalingRatioCallback): NR runs 1:1.
uint32_t KYTY_NGX_CONV ScalingRatioCallback(NgxParameter* parameters) {
	if (parameters == nullptr) return NgxFailInvalidParameter;
	parameters->Set("DLSSNR.ScalingRatio", 1.0f);
	return NgxSuccess;
}

float EnvF(const char* name, float fallback, float lo, float hi) {
	const char* v = std::getenv(name);
	if (v == nullptr || *v == '\0') return fallback;
	// ("0.35" from the launcher; a "0,35" taken as 0.35 too. The process keeps the C locale.)
	std::string t(v);
	for (auto& c: t)
		if (c == ',') c = '.';
	const float f = static_cast<float>(std::strtod(t.c_str(), nullptr));
	return std::isfinite(f) ? std::clamp(f, lo, hi) : fallback;
}
bool EnvOn(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && *v != '\0' && *v != '0' && (*v | 0x20) != 'n' && (*v | 0x20) != 'f';
}
// KYTY_DLSS_NR_PRESR=1: NR on the render-res colour BEFORE the upscaler (OptiScaler-DLSSNR-PreSR's idea), uniform grid.
bool PreSr() {
	static const bool on = EnvOn("KYTY_DLSS_NR_PRESR");
	return on;
}
void Say(const char* text) {
	std::printf("%s\n", text);
	std::fflush(stdout);
}

// ---- the grid (FoveatedDLSS5 foveation.hpp MakeAxis / MakeGrid, in pixels) ----
struct Axis {
	bool     on   = false;
	float    a    = 0.0f, b = 0.0f, e = 1.0f; // plateau half-width, fall-off (px), edge density (solved)
	uint32_t full = 0, work = 0;
	uint32_t center_px = 0, falloff_px = 0;   // the effective zone and the fall-off on screen
	float    edge_reached = 1.0f;             // the density at the screen edge
};
Axis MakeAxis(uint32_t full, double center, double falloff, double edge) {
	Axis g;
	g.full = g.work = full;
	const double half = 0.5 * full;
	const double a    = std::clamp(0.5 * center, 0.0, half);
	const double b    = std::max(0.0, falloff);
	g.center_px       = static_cast<uint32_t>(std::lround(2.0 * a));
	if (full < 16 || a >= half || edge >= 0.999) return g;
	const double k_user = 1.0 - std::clamp(edge, 0.25, 1.0);
	// G(half) = half - k B (linear in k = 1 - e).
	double B = 0.0;
	if (b > 0.0 && half < a + b) {
		const double q = (half - a) / b;
		B              = b * (q * q * q - 0.5 * q * q * q * q);
	} else {
		B = 0.5 * b + (half - a - b);
	}
	if (B <= 1e-6) return g;
	const double desired = 2.0 * (half - k_user * B);
	auto         work    = static_cast<uint32_t>(std::ceil(desired - 1e-6));
	if (((work ^ full) & 1u) != 0) ++work; // the output's parity: the plateau's texel centres are full-res texel centres
	if (work >= full || work < 16) return g;
	const double k = (half - 0.5 * work) / B; // <= k_user: the grid ends exactly at the screen edge
	g.on           = true;
	g.a            = static_cast<float>(a);
	g.b            = static_cast<float>(b);
	g.e            = static_cast<float>(1.0 - k);
	g.work         = work;
	g.falloff_px   = static_cast<uint32_t>(std::lround(std::clamp(half - a, 0.0, b)));
	if (b <= 0.0 || half >= a + b) {
		g.edge_reached = g.e;
	} else {
		const double q = (half - a) / b;
		g.edge_reached = static_cast<float>(1.0 - k * q * q * (3.0 - 2.0 * q));
	}
	return g;
}
struct Grid {
	Axis x, y;
	bool   active() const { return x.on || y.on; }
	double PixelFactor() const {
		return static_cast<double>(x.work) * y.work / (static_cast<double>(x.full) * y.full);
	}
};
enum class Why { Off, CentreCoversScreen, SavingTooSmall, MeasuredTooSmall, Active };
Grid MakeGrid(bool foveated, vk::Extent2D full, Why& why) {
	Grid g;
	g.x = MakeAxis(full.width, full.width, 0, 1.0);
	g.y = MakeAxis(full.height, full.height, 0, 1.0);
	if (!foveated) {
		why = Why::Off;
		return g;
	}
	const auto size = [](const char* name, double fallback_fraction, uint32_t extent) {
		const float v = EnvF(name, static_cast<float>(fallback_fraction), 0.0f, 100000.0f);
		return v <= 1.0f ? v * extent : static_cast<double>(v);
	};
	const double width   = size("KYTY_FOV_WIDTH", 0.52, full.width);
	const double height  = size("KYTY_FOV_HEIGHT", 0.51, full.height);
	const float  fall    = EnvF("KYTY_FOV_FALLOFF", 400.0f / 2160.0f, 0.0f, 100000.0f);
	const double falloff = fall <= 1.0f ? fall * full.height : static_cast<double>(fall);
	const double edge    = EnvF("KYTY_FOV_EDGE", 0.5f, 0.25f, 1.0f);
	Grid         f;
	f.x = MakeAxis(full.width, width, falloff, edge);
	f.y = MakeAxis(full.height, height, falloff, edge);
	if (!f.active()) {
		why = Why::CentreCoversScreen;
		g.x.center_px = f.x.center_px, g.y.center_px = f.y.center_px;
		return g;
	}
	if (f.PixelFactor() > 0.95) {
		why = Why::SavingTooSmall; // FoveatedDLSS5 kMinSaving: the warp would cost more than it saves
		g.x.center_px = f.x.center_px, g.y.center_px = f.y.center_px;
		return g;
	}
	why = Why::Active;
	return f;
}

// ---- Vulkan objects ----
struct Image {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = VK_NULL_HANDLE;
	vk::ImageView view   = nullptr;
	vk::Format    format = vk::Format::eUndefined;
	vk::Extent2D  extent {};
	bool          general = false;
	NgxResourceVk ngx {};
};
struct Retired {
	void*    feature = nullptr;
	Image    images[4];
	uint64_t frame = 0;
};
constexpr uint32_t Bindings = 11, Stamps = 4, TimingEntries = 8;
struct Push {
	int32_t full[2], work[2], render[2];
	int32_t mode, exposure_index;
	float   ax[4], ay[4];
	float   mv_scale[2];
	float   strength, debug;
};
static_assert(sizeof(Push) == 80);

struct Runtime {
	bool       tried = false, ready = false;
#if defined(_WIN32)
	HMODULE    module = nullptr;
	void**     iat_w = nullptr;
	void**     iat_a = nullptr;
#endif
	InitExtFn  init = nullptr;
	CreateFn   create = nullptr;
	EvaluateFn evaluate = nullptr;
	ReleaseFn  release = nullptr;
	std::string path, version;
};
Runtime g_rt;

struct State {
	bool                    disabled = false;
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout     = nullptr;
	vk::Pipeline            pipeline   = nullptr;
	vk::Sampler             linear     = nullptr, point = nullptr;
	VkBuffer                one = VK_NULL_HANDLE;
	VmaAllocation           one_mem = VK_NULL_HANDLE;
	bool                    one_ready = false;
	Image                   keep, out16;             // full res
	Image                   proxy, nr, wdepth, wmv;  // working grid
	vk::Extent2D            full {};
	Grid                    grid;
	Why                     why = Why::Off;
	bool                    foveated_now = false;    // the grid in use is the foveated one
	OwnedParameters*        params  = nullptr;
	void*                   feature = nullptr;
	vk::Extent2D            feature_extent {};
	bool                    pending_reset = true;
	uint32_t                eval_failures = 0;
	std::vector<Retired>    retired;
	uint64_t                frame = 0;
	// Timing: start, after encode, after NR, after resolve; a ring of 8, read 4 frames later.
	vk::QueryPool           pool = nullptr;
	double                  period_ns = 1.0;
	uint64_t                written = 0, read = 0;
	std::array<bool, TimingEntries> entry_foveated {};
	double                  sum[3] {};
	uint32_t                n = 0;
	// The A/B (KYTY_FOV_MEASURE): 0 the uniform grid measured, 1 the foveated one, 2 decided.
	int                     ab_stage = 2;
	double                  ab_sum[2] {};
	uint32_t                ab_n[2] {};
	double                  ab_uniform_ms = 0.0, ab_foveated_ms = 0.0;
	bool                    ab_done = false;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g;

constexpr uint32_t AbFrames = 240;

#if defined(_WIN32)
// ---- the runtime's caller check (RenoDX dlssnr.hpp CallerGetModuleFileNameW / InstallCallerIdentity) ----
// The runtime names its caller's module (the one holding the return address of the call into it) and refuses any whose
// path does not contain "nvngx.dll" (FAIL_PlatformError) before it reads an argument. Its import of GetModuleFileNameW
// is pointed at this: for this executable's module (where every call into it is made) it answers "nvngx.dll"; every
// other question goes to Windows.
using GetModuleFileNameWFn = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);
using GetModuleFileNameAFn = DWORD(WINAPI*)(HMODULE, LPSTR, DWORD);
GetModuleFileNameWFn g_real_w = nullptr;
GetModuleFileNameAFn g_real_a = nullptr;
HMODULE              g_self   = nullptr;

DWORD WINAPI IdentityW(HMODULE module, LPWSTR name, DWORD size) {
	if (module == g_self && name != nullptr && size > 9) {
		std::memcpy(name, L"nvngx.dll", sizeof(L"nvngx.dll"));
		return 9;
	}
	return g_real_w != nullptr ? g_real_w(module, name, size) : 0;
}
DWORD WINAPI IdentityA(HMODULE module, LPSTR name, DWORD size) {
	if (module == g_self && name != nullptr && size > 9) {
		std::memcpy(name, "nvngx.dll", sizeof("nvngx.dll"));
		return 9;
	}
	return g_real_a != nullptr ? g_real_a(module, name, size) : 0;
}
void** FindImport(HMODULE module, const char* import_name) {
	auto* base = reinterpret_cast<unsigned char*>(module);
	auto* dos  = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
	if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
	auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
	if (nt->Signature != IMAGE_NT_SIGNATURE) return nullptr;
	const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
	if (directory.VirtualAddress == 0 || directory.Size == 0) return nullptr;
	for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress); d->Name != 0; ++d) {
		auto* names     = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (d->OriginalFirstThunk != 0 ? d->OriginalFirstThunk : d->FirstThunk));
		auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
		for (; names->u1.AddressOfData != 0; ++names, ++addresses) {
			if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
			auto* import = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
			if (std::strcmp(reinterpret_cast<const char*>(import->Name), import_name) == 0)
				return reinterpret_cast<void**>(&addresses->u1.Function);
		}
	}
	return nullptr;
}
bool Patch(void** slot, void* value, void** old) {
	DWORD protection = 0;
	if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection)) return false;
	if (old != nullptr) *old = *slot;
	*slot = value;
	FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
	DWORD ignored = 0;
	VirtualProtect(slot, sizeof(void*), protection, &ignored);
	return true;
}

// The module's file version from its own VS_FIXEDFILEINFO resource (as frame-gen.cpp ModuleVersion: no version.dll).
std::string ModuleVersionText(HMODULE module) {
	HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16));
	if (resource == nullptr) return {};
	HGLOBAL     handle = LoadResource(module, resource);
	const DWORD size   = SizeofResource(module, resource);
	const auto* bytes  = handle != nullptr ? static_cast<const uint8_t*>(LockResource(handle)) : nullptr;
	if (bytes == nullptr) return {};
	for (DWORD i = 0; i + 52 <= size; i += 4) {
		uint32_t signature = 0;
		std::memcpy(&signature, bytes + i, 4);
		if (signature != 0xFEEF04BDu) continue;
		uint32_t ms = 0, ls = 0;
		std::memcpy(&ms, bytes + i + 8, 4);
		std::memcpy(&ls, bytes + i + 12, 4);
		char text[64];
		std::snprintf(text, sizeof(text), "%u.%u.%u.%u", ms >> 16, ms & 0xffffu, ls >> 16, ls & 0xffffu);
		return text;
	}
	return {};
}

// The calls into the runtime, each in a frame of its own (never a tail call: the runtime checks the module holding its
// return address) and under structured exception handling: a fault inside the runtime turns NR off, not the game.
#if defined(_MSC_VER)
#define KYTY_NR_TRY __try
#define KYTY_NR_EXCEPT __except (EXCEPTION_EXECUTE_HANDLER)
#else
#define KYTY_NR_TRY if (true)
#define KYTY_NR_EXCEPT else
#endif
__declspec(noinline) uint32_t CallInit(InitExtFn fn, const wchar_t* path, VkInstance i, VkPhysicalDevice p, VkDevice d) {
	volatile uint32_t r = 0xBAD00000u;
	KYTY_NR_TRY { r = fn(NgxApplicationId, path, i, p, d, NgxVersionApi, nullptr); }
	KYTY_NR_EXCEPT { r = 0xBAD0FFFFu; }
	return r;
}
__declspec(noinline) uint32_t CallCreate(CreateFn fn, VkCommandBuffer c, NgxParameter* params, void** handle) {
	volatile uint32_t r = 0xBAD00000u;
	KYTY_NR_TRY { r = fn(c, NgxFeatureNr, params, handle); }
	KYTY_NR_EXCEPT { r = 0xBAD0FFFFu; }
	return r;
}
__declspec(noinline) uint32_t CallEvaluate(EvaluateFn fn, VkCommandBuffer c, const void* handle, const NgxParameter* params) {
	volatile uint32_t r = 0xBAD00000u;
	KYTY_NR_TRY { r = fn(c, handle, params, nullptr); }
	KYTY_NR_EXCEPT { r = 0xBAD0FFFFu; }
	return r;
}
__declspec(noinline) uint32_t CallRelease(ReleaseFn fn, void* handle) {
	volatile uint32_t r = 0xBAD00000u;
	KYTY_NR_TRY { r = fn(handle); }
	KYTY_NR_EXCEPT { r = 0xBAD0FFFFu; }
	return r;
}
#else
uint32_t CallInit(InitExtFn, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice) { return 0xBAD00001u; }
uint32_t CallCreate(CreateFn, VkCommandBuffer, NgxParameter*, void**) { return 0xBAD00001u; }
uint32_t CallEvaluate(EvaluateFn, VkCommandBuffer, const void*, const NgxParameter*) { return 0xBAD00001u; }
uint32_t CallRelease(ReleaseFn, void*) { return 0xBAD00001u; }
#endif

#if defined(_WIN32)
std::filesystem::path ExeDir() {
	wchar_t exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	return std::filesystem::path(exe).parent_path();
}
std::string PathText(const std::filesystem::path& p) {
	try {
		const auto u8 = p.u8string();
		return std::string(u8.begin(), u8.end());
	} catch (...) {
		return "?";
	}
}
// nvngx_dlssnr.dll beside the emulator, in its streamline folder, or in KYTY_STREAMLINE_DIR; `searched` lists where.
std::filesystem::path FindRuntime(std::string& searched) {
	const auto dir = ExeDir();
	std::vector<std::filesystem::path> candidates {dir / "nvngx_dlssnr.dll", dir / "streamline" / "nvngx_dlssnr.dll"};
	if (const wchar_t* env = _wgetenv(L"KYTY_STREAMLINE_DIR"); env != nullptr && *env != L'\0')
		candidates.push_back(std::filesystem::path(env) / "nvngx_dlssnr.dll");
	std::filesystem::path found;
	for (const auto& c: candidates) {
		std::error_code error;
		if (!searched.empty()) searched += ", ";
		searched += PathText(c);
		if (found.empty() && std::filesystem::exists(c, error)) found = c;
	}
	return found;
}
#endif

// The runtime: found, signed, loaded, its exports and the caller identity, initialized on this device. Once.
bool EnsureRuntime(GraphicContext& graphics) {
	if (g_rt.ready) return true;
	if (g_rt.tried) return false;
	g_rt.tried = true;
#if defined(_WIN32)
	const auto dir = ExeDir();
	std::string searched;
	const std::filesystem::path found = FindRuntime(searched);
	if (found.empty()) {
		std::printf("DLSS-NR: nvngx_dlssnr.dll not found (looked in %s and %s\\streamline): DLSS 5 Neural Rendering and the "
		            "foveated grid are off, the upscaler's picture as usual. Place NVIDIA's nvngx_dlssnr.dll there to use it.\n",
		            PathText(dir).c_str(), PathText(dir).c_str());
		std::fflush(stdout);
		return false;
	}
	g_rt.path = PathText(found);
	if (!LocalPlatform::IsNvidiaSigned(found.wstring()) && !EnvOn("KYTY_DLSS_NR_UNSIGNED")) {
		std::printf("DLSS-NR: %s is not signed by NVIDIA: not loaded (KYTY_DLSS_NR_UNSIGNED=1 loads it anyway). NR off.\n",
		            g_rt.path.c_str());
		std::fflush(stdout);
		return false;
	}
	g_rt.module = LoadLibraryExW(found.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (g_rt.module == nullptr) {
		std::printf("DLSS-NR: cannot load %s (error %lu). NR off.\n", g_rt.path.c_str(), GetLastError());
		std::fflush(stdout);
		return false;
	}
	g_rt.version = ModuleVersionText(g_rt.module);
	const auto load = [&](auto& fn, const char* name) {
		fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(reinterpret_cast<void*>(GetProcAddress(g_rt.module, name)));
		return fn != nullptr;
	};
	const bool init_ok = load(g_rt.init, "NVSDK_NGX_VULKAN_Init_Ext");
	const bool create_ok = load(g_rt.create, "NVSDK_NGX_VULKAN_CreateFeature");
	const bool eval_ok = load(g_rt.evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
	const bool release_ok = load(g_rt.release, "NVSDK_NGX_VULKAN_ReleaseFeature");
	const bool d3d12 = GetProcAddress(g_rt.module, "NVSDK_NGX_D3D12_CreateFeature") != nullptr;
	if (!init_ok || !create_ok || !eval_ok || !release_ok) {
		std::printf("DLSS-NR: %s (version %s) has no complete Vulkan entry points (Init_Ext %d, CreateFeature %d, EvaluateFeature %d, "
		            "ReleaseFeature %d; D3D12 ones %s): this runtime has no Vulkan path. NR off.\n",
		            g_rt.path.c_str(), g_rt.version.empty() ? "?" : g_rt.version.c_str(), init_ok, create_ok, eval_ok, release_ok,
		            d3d12 ? "present" : "absent");
		std::fflush(stdout);
		return false;
	}
	g_self = GetModuleHandleW(nullptr);
	g_rt.iat_w = FindImport(g_rt.module, "GetModuleFileNameW");
	g_rt.iat_a = FindImport(g_rt.module, "GetModuleFileNameA");
	void* old = nullptr;
	if (g_rt.iat_w == nullptr || !Patch(g_rt.iat_w, reinterpret_cast<void*>(&IdentityW), &old)) {
		std::printf("DLSS-NR: %s imports no GetModuleFileNameW to answer its caller check (or it could not be patched). NR off.\n",
		            g_rt.path.c_str());
		std::fflush(stdout);
		return false;
	}
	g_real_w = reinterpret_cast<GetModuleFileNameWFn>(old);
	if (g_rt.iat_a != nullptr && Patch(g_rt.iat_a, reinterpret_cast<void*>(&IdentityA), &old)) g_real_a = reinterpret_cast<GetModuleFileNameAFn>(old);
	const std::wstring data_path = dir.wstring();
	const uint32_t     result    = CallInit(g_rt.init, data_path.c_str(), static_cast<VkInstance>(graphics.instance),
	                                        static_cast<VkPhysicalDevice>(graphics.physical_device), static_cast<VkDevice>(graphics.device));
	if (result != NgxSuccess) {
		std::printf("DLSS-NR: NVSDK_NGX_VULKAN_Init_Ext of %s (version %s) returned 0x%08x%s. NR off (update the NVIDIA driver; "
		            "0xbad00002: the caller check refused it; 0xbad00001: not supported on this GPU / driver).\n",
		            g_rt.path.c_str(), g_rt.version.empty() ? "?" : g_rt.version.c_str(), result,
		            result == 0xBAD0FFFFu ? " (it raised an exception)" : "");
		std::fflush(stdout);
		return false;
	}
	std::printf("DLSS-NR: %s (version %s) initialized on the Vulkan device (NGX feature 18, native Vulkan entry points)\n",
	            g_rt.path.c_str(), g_rt.version.empty() ? "?" : g_rt.version.c_str());
	std::fflush(stdout);
	g_rt.ready = true;
	return true;
#else
	(void)graphics;
	Say("DLSS-NR: Windows only (NVIDIA's nvngx_dlssnr.dll). NR off.");
	return false;
#endif
}

bool MakeImage(GraphicContext& graphics, vk::Format format, vk::Extent2D extent, Image& img, bool read_write) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	                       VK_IMAGE_USAGE_STORAGE_BIT;
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &img.image, &img.memory, nullptr) != VK_SUCCESS) {
		img.image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(img.image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &img.view) != vk::Result::eSuccess) {
		img.view = nullptr;
		return false;
	}
	img.format  = format;
	img.extent  = extent;
	img.general = false;
	img.ngx     = {};
	auto& info  = img.ngx.Resource.ImageViewInfo;
	info.ImageView        = static_cast<VkImageView>(img.view);
	info.Image            = img.image;
	info.SubresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	info.Format           = static_cast<VkFormat>(format);
	info.Width            = extent.width;
	info.Height           = extent.height;
	img.ngx.Type          = 0;
	img.ngx.ReadWrite     = read_write;
	return true;
}
void FreeImage(GraphicContext& graphics, Image& img) {
	if (img.view != nullptr) graphics.device.destroyImageView(img.view, nullptr);
	if (img.image != VK_NULL_HANDLE) vmaDestroyImage(graphics.allocator, img.image, img.memory);
	img = {};
}

vk::DescriptorType DescType(uint32_t i) {
	if (i == 10) return vk::DescriptorType::eStorageBuffer;
	if (i == 3 || i == 4 || i == 5 || i == 9) return vk::DescriptorType::eStorageImage;
	return vk::DescriptorType::eCombinedImageSampler;
}

bool EnsurePipeline(GraphicContext& graphics) {
	if (g.pipeline != nullptr) return true;
	const auto device = graphics.device;
	std::array<vk::DescriptorSetLayoutBinding, Bindings> bindings {};
	for (uint32_t i = 0; i < Bindings; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = DescType(i);
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = Bindings;
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &g.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &g.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &g.layout) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = sizeof(DLSS_NR_SPV);
	module_info.pCode    = DLSS_NR_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = g.layout;
	const auto created         = device.createComputePipelines(LocalPassCache::Get(device), 1, &pipeline_info, nullptr, &g.pipeline);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		g.pipeline = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eLinear;
	sampler.minFilter    = vk::Filter::eLinear;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = 0.0f;
	if (device.createSampler(&sampler, nullptr, &g.linear) != vk::Result::eSuccess) return false;
	sampler.magFilter = vk::Filter::eNearest;
	sampler.minFilter = vk::Filter::eNearest;
	if (device.createSampler(&sampler, nullptr, &g.point) != vk::Result::eSuccess) return false;
	VkBufferCreateInfo buffer {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer.size  = 16;
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo info {};
	info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, &buffer, &info, &g.one, &g.one_mem, nullptr) != VK_SUCCESS) return false;
	vk::QueryPoolCreateInfo query {};
	query.queryType  = vk::QueryType::eTimestamp;
	query.queryCount = Stamps * TimingEntries;
	if (device.createQueryPool(&query, nullptr, &g.pool) != vk::Result::eSuccess) g.pool = nullptr;
	g.period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	return true;
}

// Freed 32 frames after they were last used (never under the GPU; RenoDX parks retired features the same way).
void RetireWorking(GraphicContext& graphics) {
	(void)graphics;
	Retired r;
	r.feature   = g.feature;
	r.images[0] = g.proxy, r.images[1] = g.nr, r.images[2] = g.wdepth, r.images[3] = g.wmv;
	r.frame     = g.frame;
	g.retired.push_back(r);
	g.feature = nullptr;
	g.proxy = {}, g.nr = {}, g.wdepth = {}, g.wmv = {};
	g.feature_extent = {};
}
void FreeRetired(GraphicContext& graphics) {
	for (auto it = g.retired.begin(); it != g.retired.end();) {
		if (g.frame < it->frame + 32) {
			++it;
			continue;
		}
		if (it->feature != nullptr && g_rt.release != nullptr) CallRelease(g_rt.release, it->feature);
		for (auto& img: it->images) FreeImage(graphics, img);
		it = g.retired.erase(it);
	}
}

void ToGeneral(vk::CommandBuffer command, Image& img) {
	if (img.general) return;
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = vk::PipelineStageFlagBits2::eNone;
	b.dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	b.dstAccessMask       = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	b.oldLayout           = vk::ImageLayout::eUndefined;
	b.newLayout           = vk::ImageLayout::eGeneral;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(img.image);
	b.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
	img.general = true;
}
void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src;
	barrier.dstStageMask  = dst_stage;
	barrier.dstAccessMask = dst;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// The model's steering, read once (RenoDX's defaults).
struct Steering {
	uint32_t preset = 0, style = 0;
	float    intensity = 1.0f, tone = 1.0f, structure = 1.0f, skin = -1.0f;
	int      auto_mask = 0;
};
const Steering& Steer() {
	static const Steering s = [] {
		Steering m;
		m.preset    = static_cast<uint32_t>(EnvF("KYTY_DLSS_NR_PRESET", 0.0f, 0.0f, 32.0f));
		m.style     = static_cast<uint32_t>(EnvF("KYTY_DLSS_NR_STYLE", 0.0f, 0.0f, 32.0f));
		m.intensity = EnvF("KYTY_DLSS_NR_INTENSITY", 1.0f, 0.0f, 2.0f);
		m.tone      = EnvF("KYTY_DLSS_NR_TONE", 1.0f, 0.0f, 2.0f);
		m.structure = EnvF("KYTY_DLSS_NR_STRUCTURE", 1.0f, 0.0f, 2.0f);
		m.skin      = EnvF("KYTY_DLSS_NR_SKIN", -1.0f, -1.0f, 2.0f);
		m.auto_mask = EnvOn("KYTY_DLSS_NR_AUTOMASK") ? 1 : 0;
		return m;
	}();
	return s;
}
void SetSizes(NgxParameter& p, vk::Extent2D work) {
	p.Set("Width", work.width), p.Set("Height", work.height), p.Set("OutWidth", work.width), p.Set("OutHeight", work.height);
	p.Set("DLSSNR.Width", work.width), p.Set("DLSSNR.Height", work.height);
	p.Set("DLSSNR.InputWidth", work.width), p.Set("DLSSNR.InputHeight", work.height);
	p.Set("DLSSNR.OutputWidth", work.width), p.Set("DLSSNR.OutputHeight", work.height);
	p.Set("DLSSNR.Output.Width", work.width), p.Set("DLSSNR.Output.Height", work.height);
	p.Set("DLSSNR.ScalingRatio", 1.0f), p.Set("DLSSNR.Scale", 1.0f);
	p.Set("DLSSNRComputeScalingRatioCallback", reinterpret_cast<void*>(&ScalingRatioCallback));
}
void SetSteering(NgxParameter& p) {
	const auto& s = Steer();
	p.Set("DLSSNR.Intensity", s.intensity);
	p.Set("DLSSNR.Style", s.style);
	p.Set("DLSSNR.LocalToneStrength", s.tone);
	p.Set("DLSSNR.LocalStructureStrength", s.structure);
	p.Set("DLSSNR.SkinStructureStrength", s.skin);
	p.Set("DLSSNR.UseAutoMask", s.auto_mask);
	p.Set("DLSSNR.UICorrection", 0);
}

bool EnsureFeature(GraphicContext& graphics, vk::CommandBuffer command, vk::Extent2D work) {
	if (g.feature != nullptr && g.feature_extent == work) return true;
	if (g.feature != nullptr) RetireWorking(graphics);
	bool ok = MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, work, g.proxy, false);
	ok      = ok && MakeImage(graphics, vk::Format::eR16G16B16A16Sfloat, work, g.nr, true);
	ok      = ok && MakeImage(graphics, vk::Format::eR32Sfloat, work, g.wdepth, false);
	ok      = ok && MakeImage(graphics, vk::Format::eR16G16Sfloat, work, g.wmv, false);
	if (!ok) {
		for (auto* img: {&g.proxy, &g.nr, &g.wdepth, &g.wmv}) FreeImage(graphics, *img);
		Say("DLSS-NR: could not create its working images. NR off.");
		return false;
	}
	if (g.params == nullptr) g.params = new OwnedParameters();
	auto& p = *g.params;
	p.Reset();
	SetSizes(p, work);
	p.Set("DLSSNR.Hint.Render.Preset", Steer().preset);
	p.Set("PerfQualityValue", 0);
	p.Set("DLSS.Feature.Create.Flags", NgxFlagDepthInverted);
	p.Set("CreationNodeMask", 1u), p.Set("VisibilityNodeMask", 1u);
	p.Set("DLSSNR.Enabled", 1);
	SetSteering(p);
	void*          handle = nullptr;
	const uint32_t result = CallCreate(g_rt.create, static_cast<VkCommandBuffer>(command), &p, &handle);
	if (result != NgxSuccess || handle == nullptr) {
		std::printf("DLSS-NR: creating feature 18 at %ux%u returned 0x%08x%s. NR off.\n", work.width, work.height, result,
		            result == 0xBAD0FFFFu ? " (it raised an exception)" : "");
		std::fflush(stdout);
		for (auto* img: {&g.proxy, &g.nr, &g.wdepth, &g.wmv}) FreeImage(graphics, *img);
		return false;
	}
	g.feature        = handle;
	g.feature_extent = work;
	g.pending_reset  = true;
	std::printf("DLSS-NR: feature 18 created at %ux%u (preset %u, style %u, intensity %g, tone %g, structure %g, skin %g, auto mask %d)\n",
	            work.width, work.height, Steer().preset, Steer().style, Steer().intensity, Steer().tone, Steer().structure, Steer().skin,
	            Steer().auto_mask);
	std::fflush(stdout);
	return true;
}

const char* WhyText(Why why) {
	switch (why) {
		case Why::Off: return "uniform grid (KYTY_FOVEATED off)";
		case Why::CentreCoversScreen: return "uniform grid (the centre zone covers the screen)";
		case Why::SavingTooSmall: return "uniform grid (the foveated grid would save < 5 % of the NR pixels: auto-off)";
		case Why::MeasuredTooSmall: return "uniform grid (the A/B measured < 5 % GPU saved: auto-off)";
		case Why::Active: return "foveated";
	}
	return "";
}

void ReadTiming(GraphicContext& graphics) {
	while (g.pool != nullptr && g.read + 4 <= g.written) {
		const auto                   entry = static_cast<uint32_t>(g.read % TimingEntries);
		std::array<uint64_t, Stamps> stamps {};
		if (graphics.device.getQueryPoolResults(g.pool, entry * Stamps, Stamps, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess)
			break;
		double ms[3];
		for (int i = 0; i < 3; ++i) ms[i] = static_cast<double>(stamps[i + 1] - stamps[i]) * g.period_ns / 1e6;
		for (int i = 0; i < 3; ++i) g.sum[i] += ms[i];
		++g.n;
		const double total = ms[0] + ms[1] + ms[2];
		if (g.ab_stage < 2) {
			const int which = g.entry_foveated[entry] ? 1 : 0;
			if (which == g.ab_stage) g.ab_sum[which] += total, ++g.ab_n[which];
		}
		++g.read;
	}
}

void Log(vk::Extent2D full) {
	const auto now = std::chrono::steady_clock::now();
	if (now - g.last_log < std::chrono::seconds(10) || g.n == 0) return;
	const auto&  gr  = g.grid;
	const double pf  = gr.PixelFactor();
	char         ab[256] {};
	if (g.ab_done)
		std::snprintf(ab, sizeof(ab), "; A/B: uniform %.3f ms -> foveated %.3f ms, GPU saved %.3f ms (%.1f%%)", g.ab_uniform_ms,
		              g.ab_foveated_ms, g.ab_uniform_ms - g.ab_foveated_ms,
		              g.ab_uniform_ms > 0.0 ? 100.0 * (g.ab_uniform_ms - g.ab_foveated_ms) / g.ab_uniform_ms : 0.0);
	else if (g.ab_stage < 2)
		std::snprintf(ab, sizeof(ab), "; A/B measuring (%s)", g.ab_stage == 0 ? "uniform" : "foveated");
	std::printf("DLSS-NR: 10 s: %s, zone %ux%u px, fall-off %u/%u px, edge %.2f/%.2f (reached %.2f/%.2f), NR grid %ux%u of %ux%u = %.1f%% "
	            "of the pixels (%.1f%% saved); GPU encode %.3f + NR %.3f + resolve %.3f = %.3f ms (mean of %u)%s\n",
	            WhyText(g.why), gr.x.center_px, gr.y.center_px, gr.x.falloff_px, gr.y.falloff_px, gr.x.e, gr.y.e, gr.x.edge_reached,
	            gr.y.edge_reached, gr.x.work, gr.y.work, full.width, full.height, 100.0 * pf, 100.0 * (1.0 - pf), g.sum[0] / g.n,
	            g.sum[1] / g.n, g.sum[2] / g.n, (g.sum[0] + g.sum[1] + g.sum[2]) / g.n, g.n, ab);
	std::fflush(stdout);
	g.sum[0] = g.sum[1] = g.sum[2] = 0.0;
	g.n                            = 0;
	g.last_log                     = now;
}

// The grid for this output and the A/B's stage.
void UpdateGrid(vk::Extent2D full) {
	static const bool foveated = EnvOn("KYTY_FOVEATED") && !PreSr();
	static const bool measure  = [] {
		const char* v = std::getenv("KYTY_FOV_MEASURE");
		return v == nullptr || *v != '0';
	}();
	if (g.full != full) {
		g.full = full;
		Why        why;
		const Grid fov = MakeGrid(foveated, full, why);
		g.why          = why;
		g.grid         = fov;
		g.foveated_now = fov.active();
		g.ab_stage     = 2;
		g.ab_done      = false;
		if (g.foveated_now && measure && g.pool != nullptr) {
			// Uniform first: its GPU time is the reference.
			g.ab_stage = 0;
			g.ab_sum[0] = g.ab_sum[1] = 0.0;
			g.ab_n[0] = g.ab_n[1] = 0;
			g.grid.x = MakeAxis(full.width, full.width, 0, 1.0);
			g.grid.y = MakeAxis(full.height, full.height, 0, 1.0);
			g.foveated_now = false;
		}
		const auto& f = fov;
		std::printf("DLSS-NR: output %ux%u: %s; zone %ux%u px, fall-off %u/%u px, edge %.2f/%.2f, NR grid %ux%u (%.1f%% of the pixels)%s\n",
		            full.width, full.height, WhyText(why), f.x.center_px, f.y.center_px, f.x.falloff_px, f.y.falloff_px, f.x.e, f.y.e, f.x.work,
		            f.y.work, 100.0 * f.PixelFactor(), g.ab_stage == 0 ? "; A/B: the uniform grid first, then the foveated one" : "");
		std::fflush(stdout);
		return;
	}
	if (g.ab_stage == 0 && g.ab_n[0] >= AbFrames) {
		g.ab_uniform_ms = g.ab_sum[0] / g.ab_n[0];
		Why why;
		g.grid         = MakeGrid(foveated, full, why);
		g.foveated_now = true;
		g.ab_stage     = 1;
	} else if (g.ab_stage == 1 && g.ab_n[1] >= AbFrames) {
		g.ab_foveated_ms = g.ab_sum[1] / g.ab_n[1];
		g.ab_stage       = 2;
		g.ab_done        = true;
		const double saved = g.ab_uniform_ms - g.ab_foveated_ms;
		const bool   keep  = g.ab_uniform_ms > 0.0 && saved >= 0.05 * g.ab_uniform_ms;
		std::printf("DLSS-NR: A/B (%u frames each): uniform %.3f ms, foveated %.3f ms (encode + NR + resolve): GPU saved %.3f ms (%.1f%%), "
		            "%.1f%% of the NR pixels saved: %s\n",
		            AbFrames, g.ab_uniform_ms, g.ab_foveated_ms, saved, g.ab_uniform_ms > 0.0 ? 100.0 * saved / g.ab_uniform_ms : 0.0,
		            100.0 * (1.0 - g.grid.PixelFactor()), keep ? "foveated grid kept" : "less than 5 %: back to the uniform grid (auto-off)");
		std::fflush(stdout);
		if (!keep) {
			g.why          = Why::MeasuredTooSmall;
			g.grid.x       = MakeAxis(full.width, full.width, 0, 1.0);
			g.grid.y       = MakeAxis(full.height, full.height, 0, 1.0);
			g.foveated_now = false;
		}
	}
}

} // namespace

bool DlssNrWanted() {
	static const bool wanted = [] {
		const bool on = EnvOn("KYTY_FOVEATED") || EnvOn("KYTY_DLSS_NR");
		if (on) {
			std::printf("DLSS-NR: KYTY_DLSS_NR=%s KYTY_FOVEATED=%s read (edge %s, zone %s x %s, fall-off %s)\n",
			            std::getenv("KYTY_DLSS_NR") != nullptr ? std::getenv("KYTY_DLSS_NR") : "(unset)",
			            std::getenv("KYTY_FOVEATED") != nullptr ? std::getenv("KYTY_FOVEATED") : "(unset)",
			            std::getenv("KYTY_FOV_EDGE") != nullptr ? std::getenv("KYTY_FOV_EDGE") : "0.5",
			            std::getenv("KYTY_FOV_WIDTH") != nullptr ? std::getenv("KYTY_FOV_WIDTH") : "0.52",
			            std::getenv("KYTY_FOV_HEIGHT") != nullptr ? std::getenv("KYTY_FOV_HEIGHT") : "0.51",
			            std::getenv("KYTY_FOV_FALLOFF") != nullptr ? std::getenv("KYTY_FOV_FALLOFF") : "400 px at 2160");
			std::fflush(stdout);
		}
		return on;
	}();
	return wanted;
}

// One start-up line, whatever the switches: what the emulator was given, where the DLL was looked for, whether the
// NR hook (FrameGen::Upscale, the PS5 Pro path's upscaler pass) is reachable in this mode.
static void StartupLine(const std::vector<vk::ExtensionProperties>* available) {
	static bool said = false;
	if (said) return;
	said = true;
	const auto env = [](const char* name) {
		const char* v = std::getenv(name);
		return std::string(v != nullptr ? (*v != '\0' ? v : "(empty)") : "(unset)");
	};
	std::string dll = "Windows only";
#if defined(_WIN32)
	std::string searched;
	const auto  found = FindRuntime(searched);
	dll               = found.empty() ? "nvngx_dlssnr.dll NOT found (looked at " + searched + ")" : "nvngx_dlssnr.dll found at " + PathText(found);
#endif
	const std::string up = env("KYTY_UPSCALER");
	const bool reachable = up == "dlss" || up == "xess" || up == "fsr" || up == "fsr4";
	std::string ext = "-";
	if (available != nullptr) {
		const auto has = [&](const char* name) {
			return std::any_of(available->begin(), available->end(),
			                   [&](const vk::ExtensionProperties& e) { return std::strcmp(e.extensionName.data(), name) == 0; });
		};
		ext = std::string("VK_NVX_binary_import ") + (has("VK_NVX_binary_import") ? "yes" : "no") + ", VK_NVX_image_view_handle " +
		      (has("VK_NVX_image_view_handle") ? "yes" : "no");
	}
	std::printf("DLSS-NR: start-up: KYTY_FOVEATED=%s KYTY_DLSS_NR=%s KYTY_DLSS_NR_PRESR=%s KYTY_DLSS_NR_STRENGTH=%s (%g) KYTY_DLSS_NR_UNSIGNED=%s KYTY_FOV_EDGE=%s KYTY_FOV_WIDTH=%s KYTY_FOV_HEIGHT=%s KYTY_FOV_FALLOFF=%s -> NR %s; "
	            "%s; KYTY_UPSCALER=%s: the NR hook (after the upscaler on the PS5 Pro path) is %s; device: %s\n",
	            env("KYTY_FOVEATED").c_str(), env("KYTY_DLSS_NR").c_str(), env("KYTY_DLSS_NR_PRESR").c_str(), env("KYTY_DLSS_NR_STRENGTH").c_str(),
	            static_cast<double>(EnvF("KYTY_DLSS_NR_STRENGTH", 1.0f, 0.0f, 1.0f)), env("KYTY_DLSS_NR_UNSIGNED").c_str(), env("KYTY_FOV_EDGE").c_str(), env("KYTY_FOV_WIDTH").c_str(),
	            env("KYTY_FOV_HEIGHT").c_str(), env("KYTY_FOV_FALLOFF").c_str(),
	            DlssNrWanted() ? (PreSr() ? "wanted, PreSR (render-res colour, before the upscaler)" : EnvOn("KYTY_FOVEATED") ? "wanted, foveated, after the upscaler" : "wanted, uniform, after the upscaler") : "off (neither switch is 1)", dll.c_str(),
	            up.c_str(), reachable ? "reachable (it runs once the game's 3D view reaches the upscaler)"
	                                  : "NOT reachable: no upscaler (the game's own TAA); choose DLSS, XeSS or FSR in Reescalado",
	            ext.c_str());
	std::fflush(stdout);
}

void DlssNrAddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& extensions) {
	StartupLine(&available);
	if (!DlssNrWanted()) return;
	for (const char* name: {"VK_NVX_binary_import", "VK_NVX_image_view_handle", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}) {
		const bool have = std::any_of(available.begin(), available.end(), [&](const vk::ExtensionProperties& e) {
			return std::strcmp(e.extensionName.data(), name) == 0;
		});
		const bool added = std::any_of(extensions.begin(), extensions.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
		if (have && !added) extensions.push_back(name);
		if (!have) std::printf("DLSS-NR: the device has no %s (NGX's Vulkan path needs it)\n", name);
	}
	std::fflush(stdout);
}

// The NR pass on `out` (the upscaled output, written in place; or with `pre` the render-res colour, read only, the
// result in out16 handed back in `result`).
static bool RunNr(vk::CommandBuffer command, const UpscaleInputs& inputs, const UpscaleImage& out, bool pre, UpscaleImage* result) {
	const auto skip = [](const char* why) {
		static std::string last;
		if (last != why) {
			last = why;
			std::printf("DLSS-NR: skipped: %s\n", why);
			std::fflush(stdout);
		}
		return false;
	};
	GraphicContext* graphics = inputs.graphics;
	if (graphics == nullptr || command == nullptr) return skip("no graphics context / command buffer");
	if (out.view == nullptr || out.image == nullptr) return skip("the image has no view");
	if (inputs.depth.view == nullptr || inputs.motion.view == nullptr) return skip("no depth or motion vectors from the game");
	if (out.extent.width < 64 || out.extent.height < 64) return skip("the output is smaller than 64x64");
	if (!pre && !(out.usage & vk::ImageUsageFlagBits::eTransferSrc)) return skip("the output cannot be copied (no transfer-src usage)");
	if (pre && out.extent != inputs.depth.extent) return skip("PreSR: the colour and the depth differ in size");
	if (VideoOut::MoviePlaying()) {
		g.pending_reset = true;
		return false;
	}
	if (!EnsureRuntime(*graphics) || !EnsurePipeline(*graphics)) {
		g.disabled = true;
		return false;
	}
	++g.frame;
	FreeRetired(*graphics);
	ReadTiming(*graphics);
	const vk::Extent2D full = out.extent;
	const vk::Extent2D old_full = g.full;
	UpdateGrid(full);
	const vk::Extent2D work {g.grid.x.work, g.grid.y.work};
	// Full-res images (the copy of the frame; the rgba16f stand-in when the output takes no storage writes).
	const bool direct = !pre && (out.usage & vk::ImageUsageFlagBits::eStorage) &&
	                    (graphics->physical_device.getFormatProperties(out.format).optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);
	if (!pre && !direct && !(out.usage & vk::ImageUsageFlagBits::eTransferDst)) {
		Say("DLSS-NR: the upscaled output takes neither storage writes nor transfers. NR off.");
		g.disabled = true;
		return false;
	}
	if ((!pre && g.keep.image == VK_NULL_HANDLE) || old_full != full || (!direct && g.out16.image == VK_NULL_HANDLE)) {
		// (Resized: the old full-res images go with the working ones, 32 frames later.)
		if (g.keep.image != VK_NULL_HANDLE || g.out16.image != VK_NULL_HANDLE) {
			Retired r;
			r.images[0] = g.keep, r.images[1] = g.out16, r.frame = g.frame;
			g.retired.push_back(r);
			g.keep = {}, g.out16 = {};
		}
		bool ok = pre || MakeImage(*graphics, vk::Format::eR16G16B16A16Sfloat, full, g.keep, false);
		if (ok && !direct) ok = MakeImage(*graphics, vk::Format::eR16G16B16A16Sfloat, full, g.out16, false);
		if (!ok) {
			Say("DLSS-NR: could not create its full-resolution images. NR off.");
			g.disabled = true;
			return false;
		}
	}
	if (!EnsureFeature(*graphics, command, work)) {
		g.disabled = true;
		return false;
	}
	for (auto* img: {&g.keep, &g.out16, &g.proxy, &g.nr, &g.wdepth, &g.wmv})
		if (img->image != VK_NULL_HANDLE) ToGeneral(command, *img);

	const bool timed = g.pool != nullptr;
	const auto entry = static_cast<uint32_t>(g.written % TimingEntries);
	if (timed) {
		command.resetQueryPool(g.pool, entry * Stamps, Stamps);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * Stamps);
		g.entry_foveated[entry] = g.foveated_now;
	}
	// The frame as the upscaler left it -> keep (rgba16f, an exact copy at the same size).
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
	if (!g.one_ready) {
		g.one_ready = true;
		const float one[4] {1.0f, 1.0f, 1.0f, 1.0f};
		command.updateBuffer(vk::Buffer(g.one), 0, sizeof(one), one);
	}
	if (!pre) {
		vk::ImageBlit blit {};
		blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, out.level, out.layer, 1};
		blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(full.width), static_cast<int32_t>(full.height), 1};
		blit.dstOffsets[1]  = blit.srcOffsets[1];
		command.blitImage(out.image, out.layout, vk::Image(g.keep.image), vk::ImageLayout::eGeneral, 1, &blit, vk::Filter::eNearest);
	}
	Barrier(command, vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);

	static const float strength = EnvF("KYTY_DLSS_NR_STRENGTH", 1.0f, 0.0f, 1.0f);
	static const bool  view     = EnvOn("KYTY_FOV_VIEW");
	const bool         game_exposure = inputs.exposure_buffer != nullptr && inputs.exposure_range >= sizeof(float);
	Push               push {};
	push.full[0] = static_cast<int32_t>(full.width), push.full[1] = static_cast<int32_t>(full.height);
	push.work[0] = static_cast<int32_t>(work.width), push.work[1] = static_cast<int32_t>(work.height);
	push.render[0] = static_cast<int32_t>(inputs.depth.extent.width), push.render[1] = static_cast<int32_t>(inputs.depth.extent.height);
	push.exposure_index = game_exposure ? static_cast<int32_t>(inputs.exposure_index) : 0;
	const auto axis = [](const Axis& a, float* o) {
		o[0] = a.a, o[1] = a.b, o[2] = a.e, o[3] = a.on ? 1.0f : 0.0f;
	};
	axis(g.grid.x, push.ax);
	axis(g.grid.y, push.ay);
	push.mv_scale[0] = inputs.motion_scale_x, push.mv_scale[1] = inputs.motion_scale_y;
	push.strength = strength;
	push.debug    = view && g.foveated_now ? 1.0f : 0.0f;

	const vk::ImageView dst_view = direct ? out.view : g.out16.view;
	const auto          run      = [&](int mode, vk::Extent2D size) {
        const auto              gen = vk::ImageLayout::eGeneral;
        vk::DescriptorImageInfo infos[Bindings] {};
        infos[0]  = pre ? vk::DescriptorImageInfo {g.linear, out.view, out.layout} : vk::DescriptorImageInfo {g.linear, g.keep.view, gen};
        infos[1]  = {g.point, inputs.depth.view, inputs.depth.layout};
        infos[2]  = {g.point, inputs.motion.view, inputs.motion.layout};
        infos[3]  = {nullptr, g.proxy.view, gen};
        infos[4]  = {nullptr, g.wdepth.view, gen};
        infos[5]  = {nullptr, g.wmv.view, gen};
        infos[6]  = {g.linear, g.proxy.view, gen};
        infos[7]  = {g.linear, g.nr.view, gen};
        infos[8]  = {g.point, g.wdepth.view, gen};
        infos[9]  = {nullptr, mode == 1 ? dst_view : g.out16.view != nullptr ? g.out16.view : g.keep.view, gen};
        const vk::DescriptorBufferInfo exp_info = game_exposure
                                                      ? vk::DescriptorBufferInfo {inputs.exposure_buffer, inputs.exposure_base, inputs.exposure_range}
                                                      : vk::DescriptorBufferInfo {vk::Buffer(g.one), 0, 16};
        vk::WriteDescriptorSet writes[Bindings] {};
        for (uint32_t i = 0; i < Bindings; ++i) {
            writes[i].dstBinding      = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = DescType(i);
            if (i == 10) writes[i].pBufferInfo = &exp_info;
            else writes[i].pImageInfo = &infos[i];
        }
        push.mode = mode;
        command.bindPipeline(vk::PipelineBindPoint::eCompute, g.pipeline);
        command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.layout, 0, Bindings, writes);
        command.pushConstants(g.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        command.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
	};
	// 1. Encode: the proxy and the warped guides on the working grid.
	run(0, work);
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	if (timed) command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * Stamps + 1);
	// 2. The model: one evaluate on the working grid, its guides already on it (MVecScale 1).
	auto& p = *g.params;
	p.Set("DLSSNR.Color", static_cast<void*>(&g.proxy.ngx));
	p.Set("DLSSNR.Output", static_cast<void*>(&g.nr.ngx));
	p.Set("DLSSNR.MVec", static_cast<void*>(&g.wmv.ngx));
	p.Set("DLSSNR.Depth", static_cast<void*>(&g.wdepth.ngx));
	SetSizes(p, work);
	for (const char* prefix: {"DLSSNR.Color", "DLSSNR.Output", "DLSSNR.Depth", "DLSSNR.MVec"}) {
		const std::string base(prefix);
		p.Set((base + "SubrectBaseX").c_str(), 0u), p.Set((base + "SubrectBaseY").c_str(), 0u);
		p.Set((base + "SubrectWidth").c_str(), work.width), p.Set((base + "SubrectHeight").c_str(), work.height);
	}
	p.Set("DLSSNR.MVecScaleX", 1.0f), p.Set("DLSSNR.MVecScaleY", 1.0f);
	p.Set("DLSSNR.DepthInverted", 1);
	p.Set("DLSSNR.Enabled", 1);
	if (pre) p.Set("DLSSNR.JitterOffsetX", inputs.jitter_x), p.Set("DLSSNR.JitterOffsetY", inputs.jitter_y); // (the render-res colour is jittered)
	const bool reset = g.pending_reset || inputs.reset;
	p.Set("DLSSNR.Reset", reset ? 1 : 0);
	g.pending_reset = false;
	SetSteering(p);
	const uint32_t eval_result = CallEvaluate(g_rt.evaluate, static_cast<VkCommandBuffer>(command), g.feature, &p);
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	if (timed) command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * Stamps + 2);
	const bool evaluated = eval_result == NgxSuccess;
	if (!evaluated) {
		++g.eval_failures;
		std::printf("DLSS-NR: evaluate returned 0x%08x%s (failure %u of 3)\n", eval_result, eval_result == 0xBAD0FFFFu ? " (it raised an exception)" : "",
		            g.eval_failures);
		std::fflush(stdout);
		if (g.eval_failures >= 3) {
			Say("DLSS-NR: the model refused 3 evaluations. NR off, the upscaler's picture as usual.");
			g.disabled = true;
		}
	}
	// 3. Resolve onto the frame (only when the model wrote its answer: else the frame stays as it was).
	if (evaluated) {
		run(1, full);
		if (!direct && !pre) {
			Barrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
			        vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
			vk::ImageBlit blit {};
			blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, out.level, out.layer, 1};
			blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(full.width), static_cast<int32_t>(full.height), 1};
			blit.dstOffsets[1]  = blit.srcOffsets[1];
			command.blitImage(vk::Image(g.out16.image), vk::ImageLayout::eGeneral, out.image, out.layout, 1, &blit, vk::Filter::eNearest);
		}
	}
	Barrier(command, vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer,
	        vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	if (timed) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, entry * Stamps + 3);
		++g.written;
	}
	{
		static bool said = false;
		if (!said && evaluated) {
			said = true;
			std::printf("DLSS-NR: on%s: %ux%u image (format %d, %s), NR grid %ux%u, guides from %ux%u, exposure %s, strength %g%s\n", pre ? " (PreSR, before the upscaler)" : " (after the upscaler)", full.width,
			            full.height, static_cast<int>(out.format), pre ? "into an rgba16f colour the upscaler reads" : direct ? "written in place" : "via rgba16f + blit", work.width,
			            work.height, inputs.depth.extent.width, inputs.depth.extent.height, game_exposure ? "the game's" : "1.0", strength,
			            view ? ", density view" : "");
			std::fflush(stdout);
		}
	}
	Log(full);
	if (evaluated && pre && result != nullptr) {
		result->image  = vk::Image(g.out16.image);
		result->view   = g.out16.view;
		result->layout = vk::ImageLayout::eGeneral;
		result->format = vk::Format::eR16G16B16A16Sfloat;
		result->extent = full;
		result->usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
		                vk::ImageUsageFlagBits::eTransferDst;
		result->level = 0;
		result->layer = 0;
	}
	return evaluated;
}

static void FirstReach(const char* where, const UpscaleImage& image) {
	static bool said = false;
	if (said) return;
	said = true;
	std::printf("DLSS-NR: the upscaler pass reached the NR hook (%s, %ux%u, format %d, usage 0x%x): NR %s%s\n", where, image.extent.width,
	            image.extent.height, static_cast<int>(image.format), static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage)),
	            DlssNrWanted() ? (g.disabled ? "wanted but off (see above)" : "wanted") : "off (KYTY_FOVEATED / KYTY_DLSS_NR not 1)",
	            DlssNrWanted() ? (PreSr() ? ", PreSR: before the upscaler, on the render-res colour" : ", after the upscaler") : "");
	std::fflush(stdout);
}

bool DlssNrApply(vk::CommandBuffer command, const UpscaleInputs& inputs) {
	StartupLine(nullptr);
	FirstReach("after the upscaler: its output", inputs.output);
	if (!DlssNrWanted() || g.disabled || PreSr()) return false;
	return RunNr(command, inputs, inputs.output, false, nullptr);
}

bool DlssNrPreSr(vk::CommandBuffer command, UpscaleInputs& inputs) {
	StartupLine(nullptr);
	if (!DlssNrWanted() || g.disabled || !PreSr()) return false;
	FirstReach("PreSR: the render-res colour", inputs.color);
	UpscaleImage result;
	if (!RunNr(command, inputs, inputs.color, true, &result)) return false;
	inputs.color = result;
	return true;
}

} // namespace Libs::Graphics::FrameGen
