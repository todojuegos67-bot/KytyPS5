// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bb-pathtracing-2: DLSS Ray Reconstruction for the path traced frame (pt_rr.h).

#ifdef ENABLE_PATHTRACING

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "common/logging/log.h"
#include "video_core/bb/bb_fg.h"
#include "video_core/bb/bb_sr.h"
#include "video_core/bb/bb_upscale.h"
#include "video_core/renderer_vulkan/rt/pt_compat.h"
#include "video_core/renderer_vulkan/rt/pt_rr.h"
#include "video_core/renderer_vulkan/rt/rt_gi.h"
#include "video_core/renderer_vulkan/rt/vk_dlss_rr.h"
#include "video_core/renderer_vulkan/rt/vk_ray_tracing.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <softpub.h>
#if defined(_MSC_VER)
#pragma comment(lib, "wintrust.lib")
#pragma comment(lib, "crypt32.lib")
#endif
#endif

namespace Vulkan {

namespace {

PtRayReconstruction* g_current = nullptr;

// ---- NGX's ABI (nvsdk_ngx_params.h / nvsdk_ngx_defs_vk.h, DLSS SDK 310.x); no SDK in the build.
// The same declarations as bb_dlss5.cpp (NVSDK_NGX_Parameter's vtable order under the MSVC ABI).
struct ID3D11Resource;
struct ID3D12Resource;
struct NgxParameter {
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;
    virtual u32 Get(const char* name, unsigned long long* out) const = 0;
    virtual u32 Get(const char* name, float* out) const = 0;
    virtual u32 Get(const char* name, double* out) const = 0;
    virtual u32 Get(const char* name, unsigned int* out) const = 0;
    virtual u32 Get(const char* name, int* out) const = 0;
    virtual u32 Get(const char* name, ID3D11Resource** out) const = 0;
    virtual u32 Get(const char* name, ID3D12Resource** out) const = 0;
    virtual u32 Get(const char* name, void** out) const = 0;
    virtual void Reset() = 0;
};
constexpr u32 NgxSuccess = 0x1, NgxFailMissingInput = 0xBAD0000Au,
              NgxFailInvalidParameter = 0xBAD00005u;
constexpr int NgxVersionApi = 0x15;         // NVSDK_NGX_Version_API
constexpr int NgxFeatureRr = 13;            // NVSDK_NGX_Feature_RayReconstruction
constexpr int NgxFlagHdr = 1 << 0;          // NVSDK_NGX_DLSS_Feature_Flags_IsHDR
constexpr int NgxFlagMvLowRes = 1 << 1;     // NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
constexpr int NgxFlagDepthInverted = 1 << 3; // NVSDK_NGX_DLSS_Feature_Flags_DepthInverted
constexpr int NgxFlagAutoExposure = 1 << 6; // NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
// NVSDK_NGX_PerfQuality_Value: MaxPerf 0, Balanced 1, MaxQuality 2, UltraPerformance 3,
// UltraQuality 4, DLAA 5.
constexpr unsigned long long NgxApplicationId = 0x876232cULL; // as bb_dlss5.cpp

// A parameter block we own (bb_dlss5.cpp OwnedParameters): any numeric type in, any out.
class OwnedParameters final : public NgxParameter {
public:
    void Set(const char* n, unsigned long long v) override {
        Store(n, double(v), v, true, reinterpret_cast<void*>(uintptr_t(v)));
    }
    void Set(const char* n, float v) override {
        Store(n, v, u64(s64(v)), false, nullptr);
    }
    void Set(const char* n, double v) override {
        Store(n, v, u64(s64(v)), false, nullptr);
    }
    void Set(const char* n, unsigned int v) override {
        Store(n, double(v), v, false, nullptr);
    }
    void Set(const char* n, int v) override {
        Store(n, double(v), u64(s64(v)), false, nullptr);
    }
    void Set(const char* n, ID3D11Resource* v) override {
        Pointer(n, v);
    }
    void Set(const char* n, ID3D12Resource* v) override {
        Pointer(n, v);
    }
    void Set(const char* n, void* v) override {
        Pointer(n, v);
    }
    u32 Get(const char* n, unsigned long long* o) const override {
        return Number(n, o);
    }
    u32 Get(const char* n, float* o) const override {
        return Number(n, o);
    }
    u32 Get(const char* n, double* o) const override {
        return Number(n, o);
    }
    u32 Get(const char* n, unsigned int* o) const override {
        return Number(n, o);
    }
    u32 Get(const char* n, int* o) const override {
        return Number(n, o);
    }
    u32 Get(const char* n, ID3D11Resource** o) const override {
        return Ptr(n, o);
    }
    u32 Get(const char* n, ID3D12Resource** o) const override {
        return Ptr(n, o);
    }
    u32 Get(const char* n, void** o) const override {
        return Ptr(n, o);
    }
    void Reset() override {
        entries.clear();
    }

private:
    struct Entry {
        double number{};
        u64 bits{};
        void* pointer{};
        bool integral{}, has_pointer{};
    };
    void Store(const char* n, double number, u64 bits, bool pointer_too, void* p) {
        if (!n) {
            return;
        }
        auto& e = entries[n];
        e.number = number;
        e.bits = bits;
        e.integral = true;
        e.has_pointer = pointer_too;
        e.pointer = p;
    }
    void Pointer(const char* n, void* v) {
        if (!n) {
            return;
        }
        auto& e = entries[n];
        e.has_pointer = true;
        e.pointer = v;
        e.bits = u64(reinterpret_cast<uintptr_t>(v));
        e.number = double(e.bits);
        e.integral = true;
    }
    template <typename T>
    u32 Number(const char* n, T* o) const {
        if (!n || !o) {
            return NgxFailInvalidParameter;
        }
        const auto it = entries.find(n);
        if (it == entries.end()) {
            return NgxFailMissingInput;
        }
        if constexpr (std::is_same_v<T, unsigned long long>) {
            *o = it->second.bits;
        } else {
            *o = T(it->second.number);
        }
        return NgxSuccess;
    }
    template <typename T>
    u32 Ptr(const char* n, T** o) const {
        if (!n || !o) {
            return NgxFailInvalidParameter;
        }
        const auto it = entries.find(n);
        if (it == entries.end() || !it->second.has_pointer) {
            return NgxFailMissingInput;
        }
        *o = static_cast<T*>(it->second.pointer);
        return NgxSuccess;
    }
    std::map<std::string, Entry, std::less<>> entries;
};

// NVSDK_NGX_Resource_VK.
struct NgxImageViewInfoVk {
    VkImageView ImageView;
    VkImage Image;
    VkImageSubresourceRange SubresourceRange;
    VkFormat Format;
    unsigned int Width;
    unsigned int Height;
};
struct NgxBufferInfoVk {
    VkBuffer Buffer;
    unsigned int SizeInBytes;
};
struct NgxResourceVk {
    union {
        NgxImageViewInfoVk ImageViewInfo;
        NgxBufferInfoVk BufferInfo;
    } Resource;
    int Type; // NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW = 0
    bool ReadWrite;
};

NgxResourceVk Wrap(vk::Image image, vk::ImageView view, vk::Format format, u32 w, u32 h,
                   bool write, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
    NgxResourceVk r{};
    r.Type = 0;
    r.ReadWrite = write;
    r.Resource.ImageViewInfo.ImageView = static_cast<VkImageView>(view);
    r.Resource.ImageViewInfo.Image = static_cast<VkImage>(image);
    r.Resource.ImageViewInfo.SubresourceRange =
        static_cast<VkImageSubresourceRange>(vk::ImageSubresourceRange{aspect, 0, 1, 0, 1});
    r.Resource.ImageViewInfo.Format = static_cast<VkFormat>(format);
    r.Resource.ImageViewInfo.Width = w;
    r.Resource.ImageViewInfo.Height = h;
    return r;
}

#if defined(_WIN32)
#define BB_RR_CONV __cdecl
#else
#define BB_RR_CONV
#endif
using InitExtFn = u32(BB_RR_CONV*)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice,
                                    VkDevice, int, const NgxParameter*);
using CreateFn = u32(BB_RR_CONV*)(VkCommandBuffer, int, NgxParameter*, void**);
using EvaluateFn = u32(BB_RR_CONV*)(VkCommandBuffer, const void*, const NgxParameter*, void*);
using ReleaseFn = u32(BB_RR_CONV*)(void*);

std::string Hex(u32 v) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", v);
    return text;
}

int QualityValue(BB::SR::Quality q) {
    switch (q) {
    case BB::SR::Quality::Native:
        return 5; // DLAA
    case BB::SR::Quality::Balanced:
        return 1;
    case BB::SR::Quality::Performance:
        return 0;
    case BB::SR::Quality::UltraPerformance:
        return 3;
    default:
        return 2; // MaxQuality
    }
}

// BB_PT_DLSS_RR_PRESET: D or E (NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D = 4, E = 5);
// default 0 (the runtime's own default model).
u32 PresetValue() {
    const char* v = std::getenv("BB_PT_DLSS_RR_PRESET");
    if (!v || !*v) {
        return 0;
    }
    const char c = char(*v & ~0x20);
    return c == 'D' ? 4u : c == 'E' ? 5u : 0u;
}

#if defined(_WIN32)
// The runtime's caller check (bb_dlss5.cpp IdentityW): it refuses callers whose module path does
// not contain "nvngx.dll"; its import of GetModuleFileNameW answers that for our executable.
using GetModuleFileNameWFn = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);
using GetModuleFileNameAFn = DWORD(WINAPI*)(HMODULE, LPSTR, DWORD);
GetModuleFileNameWFn g_real_w = nullptr;
GetModuleFileNameAFn g_real_a = nullptr;
HMODULE g_self = nullptr;

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
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        return nullptr;
    }
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) {
        return nullptr;
    }
    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (directory.VirtualAddress == 0 || directory.Size == 0) {
        return nullptr;
    }
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress);
         d->Name != 0; ++d) {
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(
            base + (d->OriginalFirstThunk != 0 ? d->OriginalFirstThunk : d->FirstThunk));
        auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
        for (; names->u1.AddressOfData != 0; ++names, ++addresses) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) {
                continue;
            }
            auto* import = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(import->Name), import_name) == 0) {
                return reinterpret_cast<void**>(&addresses->u1.Function);
            }
        }
    }
    return nullptr;
}
bool Patch(void** slot, void* value, void** old) {
    DWORD protection = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection)) {
        return false;
    }
    if (old != nullptr) {
        *old = *slot;
    }
    *slot = value;
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), protection, &ignored);
    return true;
}
bool IsNvidiaSigned(const std::wstring& path) {
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof(file);
    file.pcwszFilePath = path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof(data);
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    const LONG status = WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
    bool nvidia = false;
    if (status == ERROR_SUCCESS) {
        auto* provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        auto* signer =
            provider != nullptr ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        auto* cert = signer != nullptr ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[256]{};
        if (cert != nullptr && cert->pCert != nullptr &&
            CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256) >
                1) {
            nvidia = std::wstring(name) == L"NVIDIA Corporation";
        }
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    (void)WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
    return nvidia;
}
std::filesystem::path ExeDir() {
    wchar_t exe[MAX_PATH]{};
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
// Each call in a frame of its own, under SEH: a fault in the runtime turns RR off, not the game.
#if defined(_MSC_VER)
#define BB_RR_TRY __try
#define BB_RR_EXCEPT __except (EXCEPTION_EXECUTE_HANDLER)
#else
#define BB_RR_TRY if (true)
#define BB_RR_EXCEPT else
#endif
__declspec(noinline) u32 CallInit(InitExtFn fn, const wchar_t* path, VkInstance i,
                                  VkPhysicalDevice p, VkDevice d) {
    volatile u32 r = 0xBAD00000u;
    BB_RR_TRY {
        r = fn(NgxApplicationId, path, i, p, d, NgxVersionApi, nullptr);
    }
    BB_RR_EXCEPT {
        r = 0xBAD0FFFFu;
    }
    return r;
}
__declspec(noinline) u32 CallCreate(CreateFn fn, VkCommandBuffer c, NgxParameter* params,
                                    void** handle) {
    volatile u32 r = 0xBAD00000u;
    BB_RR_TRY {
        r = fn(c, NgxFeatureRr, params, handle);
    }
    BB_RR_EXCEPT {
        r = 0xBAD0FFFFu;
    }
    return r;
}
__declspec(noinline) u32 CallEvaluate(EvaluateFn fn, VkCommandBuffer c, const void* handle,
                                      const NgxParameter* params) {
    volatile u32 r = 0xBAD00000u;
    BB_RR_TRY {
        r = fn(c, handle, params, nullptr);
    }
    BB_RR_EXCEPT {
        r = 0xBAD0FFFFu;
    }
    return r;
}
__declspec(noinline) u32 CallRelease(ReleaseFn fn, void* handle) {
    volatile u32 r = 0xBAD00000u;
    BB_RR_TRY {
        r = fn(handle);
    }
    BB_RR_EXCEPT {
        r = 0xBAD0FFFFu;
    }
    return r;
}
#endif // _WIN32

} // namespace

// The NGX runtime and the feature.
struct PtRayReconstruction::Ngx {
    bool tried = false, ready = false;
    std::string why = "not tried";
    std::string path, version;
#if defined(_WIN32)
    HMODULE module = nullptr;
#endif
    InitExtFn init = nullptr;
    CreateFn create = nullptr;
    EvaluateFn evaluate = nullptr;
    ReleaseFn release = nullptr;
    OwnedParameters params;
    void* feature = nullptr;
    struct Key {
        u32 w, h, ow, oh;
        int quality;
        bool hdr, inverted;
        bool operator==(const Key&) const = default;
    } key{};

    bool Load(const Instance& instance) {
        if (tried) {
            return ready;
        }
        tried = true;
#if defined(_WIN32)
        const auto dir = ExeDir();
        std::vector<std::filesystem::path> candidates{dir / "nvngx_dlssd.dll",
                                                      dir / "streamline" / "nvngx_dlssd.dll"};
        for (const wchar_t* name : {L"BB_PT_DLSS_RR_DIR", L"BB_STREAMLINE_DIR"}) {
            if (const wchar_t* env = _wgetenv(name); env && *env) {
                candidates.insert(candidates.begin(), std::filesystem::path(env) / "nvngx_dlssd.dll");
            }
        }
        std::filesystem::path found;
        std::string searched;
        for (const auto& c : candidates) {
            std::error_code error;
            searched += (searched.empty() ? "" : ", ") + PathText(c);
            if (found.empty() && std::filesystem::exists(c, error)) {
                found = c;
            }
        }
        if (found.empty()) {
            why = "nvngx_dlssd.dll not found (looked at " + searched + ")";
            return false;
        }
        path = PathText(found);
        const char* unofficial = std::getenv("BB_PT_DLSS_RR_UNOFFICIAL");
        if (!IsNvidiaSigned(found.wstring()) && !(unofficial && unofficial[0] == '1')) {
            why = path + " is not signed by NVIDIA (BB_PT_DLSS_RR_UNOFFICIAL=1 loads it anyway)";
            return false;
        }
        module = LoadLibraryExW(found.wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!module) {
            why = "cannot load " + path + " (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        const auto load = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
                reinterpret_cast<void*>(GetProcAddress(module, name)));
            return fn != nullptr;
        };
        if (!load(init, "NVSDK_NGX_VULKAN_Init_Ext") ||
            !load(create, "NVSDK_NGX_VULKAN_CreateFeature") ||
            !load(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature") ||
            !load(release, "NVSDK_NGX_VULKAN_ReleaseFeature")) {
            why = path + " has no complete NGX Vulkan entry points";
            return false;
        }
        g_self = GetModuleHandleW(nullptr);
        void* old = nullptr;
        if (void** iat = FindImport(module, "GetModuleFileNameW");
            !iat || !Patch(iat, reinterpret_cast<void*>(&IdentityW), &old)) {
            why = path + ": its caller check could not be answered (no GetModuleFileNameW import)";
            return false;
        }
        g_real_w = reinterpret_cast<GetModuleFileNameWFn>(old);
        if (void** iat = FindImport(module, "GetModuleFileNameA");
            iat && Patch(iat, reinterpret_cast<void*>(&IdentityA), &old)) {
            g_real_a = reinterpret_cast<GetModuleFileNameAFn>(old);
        }
        const std::wstring data_path = dir.wstring();
        const u32 result = CallInit(init, data_path.c_str(), static_cast<VkInstance>(instance.GetInstance()),
                                    static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice()),
                                    static_cast<VkDevice>(instance.GetDevice()));
        if (result != NgxSuccess) {
            why = "NVSDK_NGX_VULKAN_Init_Ext of " + path + " returned " + Hex(result) +
                  " (0xbad00001: not supported on this GPU / driver)";
            return false;
        }
        ready = true;
        why = "ready";
        return true;
#else
        (void)instance;
        why = "Windows only (NVIDIA's nvngx_dlssd.dll)";
        return false;
#endif
    }

    void Release(const Instance& instance) {
#if defined(_WIN32)
        if (feature && release) {
            (void)instance.GetDevice().waitIdle();
            (void)CallRelease(release, feature);
        }
#else
        (void)instance;
#endif
        feature = nullptr;
    }

    bool Create(const Instance& instance, vk::CommandBuffer cmd, const Key& k) {
        if (feature && key == k) {
            return true;
        }
        Release(instance);
#if defined(_WIN32)
        OwnedParameters p;
        p.Set("CreationNodeMask", 1u);
        p.Set("VisibilityNodeMask", 1u);
        p.Set("Width", k.w);
        p.Set("Height", k.h);
        p.Set("OutWidth", k.ow);
        p.Set("OutHeight", k.oh);
        p.Set("PerfQualityValue", k.quality);
        p.Set("DLSS.Feature.Create.Flags",
              NgxFlagMvLowRes | (k.inverted ? NgxFlagDepthInverted : 0) |
                  (k.hdr ? NgxFlagHdr | NgxFlagAutoExposure : 0));
        p.Set("DLSS.Enable.Output.Subrects", 0);
        p.Set("DLSS.Denoise.Mode", 1);      // NVSDK_NGX_DLSS_Denoise_Mode_DLUnified
        p.Set("DLSS.Roughness.Mode", 1u);   // packed: normals.w
        p.Set("DLSS.Use.HW.Depth", 1u);     // hardware depth
        const u32 preset = PresetValue();
        for (const char* q : {"DLAA", "Quality", "Balanced", "Performance", "UltraPerformance",
                              "UltraQuality"}) {
            p.Set((std::string("RayReconstruction.Hint.Render.Preset.") + q).c_str(), preset);
        }
        void* handle = nullptr;
        const u32 r = CallCreate(create, static_cast<VkCommandBuffer>(cmd), &p, &handle);
        if (r != NgxSuccess || !handle) {
            why = "CreateFeature (Ray Reconstruction) returned " + Hex(r);
            return false;
        }
        feature = handle;
        key = k;
        return true;
#else
        (void)cmd;
        (void)k;
        return false;
#endif
    }
};

PtRayReconstruction::PtRayReconstruction(const Instance& instance_, Scheduler& scheduler_,
                                         Runtime& runtime_, VideoCore::TextureCache& texture_cache_,
                                         RayTracing& ray_tracing_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      texture_cache{texture_cache_}, ray_tracing{ray_tracing_}, ngx{std::make_unique<Ngx>()} {
    g_current = this;
    const char* v = std::getenv("BB_PT_DLSS_RR");
    LOG_INFO(Render_Vulkan,
             "[pt] DLSS Ray Reconstruction: BB_PT_DLSS_RR={} ({}); runs with BB_UPSCALER=dlss and "
             "nvngx_dlssd.dll beside shadPS4.exe, in its streamline folder or BB_PT_DLSS_RR_DIR",
             v && *v ? v : "unset", Wanted() ? "wanted" : "off");
}

PtRayReconstruction::~PtRayReconstruction() {
    if (g_current == this) {
        g_current = nullptr;
    }
    if (ngx) {
        ngx->Release(instance);
    }
}

PtRayReconstruction* PtRayReconstruction::Current() {
    return g_current;
}

bool PtRayReconstruction::Wanted() const {
    return ray_tracing.RayReconstructionWanted();
}

bool PtRayReconstruction::Evaluate(const BB::SR::Desc& desc) {
    const auto give_up = [&](const std::string& reason) {
        if (!failed) {
            LOG_WARNING(Render_Vulkan,
                        "[pt] DLSS Ray Reconstruction off for this session: {}; DLSS Super "
                        "Resolution and the built-in GI denoiser continue",
                        reason);
        }
        failed = true;
        active = false;
        return false;
    };
    const bool was_active = active;
    active = false;
    if (failed || !Wanted() || RayTracing::Paused() || !ray_tracing.Enabled()) {
        return false;
    }
    // Streamline's DLSS-G takes its constants from the DLSS SR evaluate (slSetConstants): with it
    // on, RR stays off unless BB_PT_DLSS_RR=1 asks for it explicitly.
    if (BB::FG::DlssSl::Requested()) {
        const char* v = std::getenv("BB_PT_DLSS_RR");
        if (!(v && v[0] == '1')) {
            return give_up("Streamline DLSS frame generation is on (BB_PT_DLSS_RR=1 forces RR)");
        }
    }
    if (!ngx->Load(instance)) {
        return give_up(ngx->why);
    }
    if (!ngx->ready) {
        return false;
    }
    if (!guides) {
        static SceneTargets no_scene_targets;
        guides = std::make_unique<DlssGuides>(instance, scheduler, runtime, texture_cache,
                                              no_scene_targets);
    }
    const vk::CommandBuffer cmd = desc.cmd;
    // The guides move G-buffer images through the texture cache (runtime.Transit records into the
    // scheduler's command buffer): only the scene pass's upscale (bb_ui_cut.cpp, recorded into
    // that buffer) qualifies; the presenter's own buffer (frames without a HUD cut) stays on SR.
    if (cmd != scheduler.CommandBuffer()) {
        if (fallbacks++ == 0) {
            LOG_INFO(Render_Vulkan, "[pt] DLSS Ray Reconstruction: a frame upscaled outside the "
                                    "scene pass (no HUD cut) uses DLSS SR");
        }
        return false;
    }
    const u32 w = desc.render.width, h = desc.render.height;
    const u32 ow = desc.upscale.width, oh = desc.upscale.height;

    // Camera: the scene block's inverse view and projection (bb_upscale.cpp PtCamera).
    DlssGuides::Camera camera{};
    float inv_view[12]{}, proj[4]{};
    camera.valid = BB::Upscale::PtCamera(inv_view, proj);
    std::memcpy(camera.inv_view.data(), inv_view, sizeof(inv_view));
    std::memcpy(camera.projection.data(), proj, sizeof(proj));
    camera.jitter = {desc.jitter[0], desc.jitter[1]};

    Dlss::Resource depth{desc.depth.image, desc.depth.view, desc.depth.format,
                         vk::ImageAspectFlagBits::eDepth, w, h};
    guides->SetInputLayouts(desc.color.layout, desc.depth.layout);
    Dlss::RrGuides g{};
    if (!guides->Record(cmd, ray_tracing.RayReconstructionSource(), depth, false, camera, g)) {
        if (fallbacks++ == 0) {
            LOG_INFO(Render_Vulkan, "[pt] DLSS Ray Reconstruction: no G-buffer guides or camera "
                                    "this frame (first such frame): DLSS SR for it");
        }
        return false; // no G-buffer or camera this frame: SR this frame
    }
    // The game's image is tonemapped LDR: RR is given linear HDR (the mod's inverse tonemap,
    // BB_PT_DLSS_RR_HDR=0 keeps LDR) and its output is tonemapped back in place.
    Dlss::Resource color{desc.color.image, desc.color.view, desc.color.format,
                         vk::ImageAspectFlagBits::eColor, w, h};
    Dlss::Resource input = color;
    const char* hdr_env = std::getenv("BB_PT_DLSS_RR_HDR");
    const bool linear = !desc.hdr && !(hdr_env && hdr_env[0] == '0') &&
                        guides->Linearize(cmd, color, input);
    const Ngx::Key key{w, h, ow, oh, QualityValue(desc.quality), desc.hdr || linear,
                       desc.depth_inverted};
    if (!ngx->Create(instance, cmd, key)) {
        return give_up(ngx->why);
    }
#if defined(_WIN32)
    const auto& p = camera.projection;
    const bool matrices = camera.valid && p[0] != 0.0f && p[1] != 0.0f && p[3] != 0.0f;
    const Rt::Affine view = Rt::InverseRigid(camera.inv_view);
    // Row-vector matrices (p * M), as the mod gives them (SuperMedo's bridge).
    float world_to_view[16]{view[0], view[4], view[8],  0.0f, view[1], view[5], view[9],  0.0f,
                            view[2], view[6], view[10], 0.0f, view[3], view[7], view[11], 1.0f};
    float view_to_clip[16]{p[0], 0.0f, 0.0f, 0.0f, 0.0f, p[1], 0.0f, 0.0f,
                           0.0f, 0.0f, p[2], 1.0f, 0.0f, 0.0f, p[3], 0.0f};
    NgxResourceVk r_color = Wrap(input.image, input.view, input.format, w, h, false);
    NgxResourceVk r_output = Wrap(desc.output.image, desc.output.view, desc.output.format, ow, oh, true);
    NgxResourceVk r_depth =
        Wrap(desc.depth.image, desc.depth.view, desc.depth.format, w, h, false,
             vk::ImageAspectFlagBits::eDepth);
    NgxResourceVk r_motion = Wrap(desc.motion.image, desc.motion.view, desc.motion.format, w, h, false);
    const auto guide = [](const Dlss::Resource& r) {
        return Wrap(r.image, r.view, r.format, r.width, r.height, false);
    };
    NgxResourceVk r_diffuse = guide(g.diffuse_albedo);
    NgxResourceVk r_specular = guide(g.specular_albedo);
    NgxResourceVk r_normals = guide(g.normal_roughness);
    NgxResourceVk r_hit = g.specular_hit.image ? guide(g.specular_hit) : NgxResourceVk{};
    auto& q = ngx->params;
    q.Reset();
    q.Set("Color", static_cast<void*>(&r_color));
    q.Set("Output", static_cast<void*>(&r_output));
    q.Set("Depth", static_cast<void*>(&r_depth));
    q.Set("MotionVectors", static_cast<void*>(&r_motion));
    q.Set("DLSS.Input.DiffuseAlbedo", static_cast<void*>(&r_diffuse));
    q.Set("DLSS.Input.SpecularAlbedo", static_cast<void*>(&r_specular));
    q.Set("GBuffer.Normals", static_cast<void*>(&r_normals));
    q.Set("GBuffer.Roughness", static_cast<void*>(&r_normals));
    q.Set("DLSSD.SpecularHitDistance", g.specular_hit.image ? static_cast<void*>(&r_hit) : nullptr);
    q.Set("ExposureTexture", static_cast<void*>(nullptr));
    q.Set("DLSS.Input.Bias.Current.Color.Mask", static_cast<void*>(nullptr));
    q.Set("TransparencyMask", static_cast<void*>(nullptr));
    q.Set("DLSSD.Alpha", static_cast<void*>(nullptr));
    q.Set("DLSSD.OutputAlpha", static_cast<void*>(nullptr));
    q.Set("WorldToViewMatrix", matrices ? static_cast<void*>(world_to_view) : nullptr);
    q.Set("ViewToClipMatrix", matrices ? static_cast<void*>(view_to_clip) : nullptr);
    // Our jitter and vectors are render pixels, as for DLSS SR (bb_sr_dlss.cpp: mvecScale 1/size
    // for Streamline's UV units; NGX takes pixels, scale 1). BB_DLSS_JITTER_SIGN applies here too.
    const char* js = std::getenv("BB_DLSS_JITTER_SIGN");
    const float sign = js && *js && std::atof(js) < 0 ? -1.0f : 1.0f;
    q.Set("Jitter.Offset.X", sign * desc.jitter[0]);
    q.Set("Jitter.Offset.Y", sign * desc.jitter[1]);
    q.Set("MV.Scale.X", 1.0f);
    q.Set("MV.Scale.Y", 1.0f);
    q.Set("Reset", (desc.reset || !was_active) ? 1 : 0);
    q.Set("DLSS.Render.Subrect.Dimensions.Width", w);
    q.Set("DLSS.Render.Subrect.Dimensions.Height", h);
    for (const char* base : {"DLSS.Input.Color.Subrect.Base.X", "DLSS.Input.Color.Subrect.Base.Y",
                             "DLSS.Input.Depth.Subrect.Base.X", "DLSS.Input.Depth.Subrect.Base.Y",
                             "DLSS.Input.MV.SubrectBase.X", "DLSS.Input.MV.SubrectBase.Y",
                             "DLSS.Output.Subrect.Base.X", "DLSS.Output.Subrect.Base.Y"}) {
        q.Set(base, 0u);
    }
    q.Set("DLSS.Pre.Exposure", 1.0f);
    q.Set("DLSS.Exposure.Scale", 1.0f);
    q.Set("FrameTimeDeltaInMsec", desc.delta_ms);
    q.Set("TonemapperType", 0u);
    const u32 result = CallEvaluate(ngx->evaluate, static_cast<VkCommandBuffer>(cmd), ngx->feature, &q);
    scheduler.GetDynamicState().Invalidate();
    if (result != NgxSuccess) {
        ngx->Release(instance);
        return give_up("EvaluateFeature returned " + Hex(result));
    }
    if (linear) {
        guides->Tonemap(cmd, {desc.output.image, desc.output.view, desc.output.format,
                              vk::ImageAspectFlagBits::eColor, ow, oh});
    }
    if (frames++ == 0) {
        LOG_INFO(Render_Vulkan,
                 "[pt] DLSS Ray Reconstruction on: {}x{} -> {}x{}, {} input, specular hit "
                 "distance {}, camera matrices {}, preset {} (BB_PT_DLSS_RR_PRESET), runtime {}",
                 w, h, ow, oh, linear ? "linearised (inverse tonemap)" : desc.hdr ? "HDR" : "LDR",
                 g.specular_hit.image ? "given" : "not given", matrices ? "given" : "not given",
                 PresetValue(), ngx->path);
    }
    active = true;
    return true;
#else
    return give_up(ngx->why);
#endif
}

} // namespace Vulkan

#endif // ENABLE_PATHTRACING
