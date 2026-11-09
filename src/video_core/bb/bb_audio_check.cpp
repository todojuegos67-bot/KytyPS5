// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Windows audio session check (bb_audio_check.h). Runs on its own thread with its own COM
// apartment; it never touches the emulator's audio code, SDL or guest memory.
#include "video_core/bb/bb_audio_check.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include "video_core/bb/bb_upscale.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <thread>
#include <windows.h>
#include <audiopolicy.h>
#include <endpointvolume.h>
#include <mmdeviceapi.h>
#undef min
#undef max

namespace BB::Audio {
namespace {

template <typename T>
struct Com {
    T* p = nullptr;
    ~Com() {
        if (p) p->Release();
    }
    T** operator&() {
        return &p;
    }
    T* operator->() const {
        return p;
    }
    explicit operator bool() const {
        return p != nullptr;
    }
};

std::string Narrow(const wchar_t* w) {
    if (w == nullptr) return "?";
    char buf[512]{};
    WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
    return buf;
}

// PKEY_Device_FriendlyName (functiondiscoverykeys_devpkey.h), defined here so no GUID library is
// needed.
constexpr PROPERTYKEY FriendlyNameKey{
    {0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};

std::string DeviceName(IMMDevice* device) {
    Com<IPropertyStore> props;
    if (FAILED(device->OpenPropertyStore(STGM_READ, &props))) return "?";
    PROPVARIANT v;
    PropVariantInit(&v);
    std::string name = "?";
    if (SUCCEEDED(props->GetValue(FriendlyNameKey, &v)) && v.vt == VT_LPWSTR) {
        name = Narrow(v.pwszVal);
    }
    PropVariantClear(&v);
    return name;
}

// Our process's sessions on one endpoint: volume, mute and peak. Returns how many were found.
int CheckEndpoint(IMMDevice* device, bool is_default, bool fix, int pass) {
    const DWORD pid = GetCurrentProcessId();
    const std::string name = DeviceName(device);
    Com<IAudioSessionManager2> manager;
    if (FAILED(device->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr,
                                reinterpret_cast<void**>(&manager)))) {
        return 0;
    }
    Com<IAudioSessionEnumerator> sessions;
    if (FAILED(manager->GetSessionEnumerator(&sessions))) return 0;
    int count = 0, found = 0;
    sessions->GetCount(&count);
    for (int i = 0; i < count; ++i) {
        Com<IAudioSessionControl> control;
        if (FAILED(sessions->GetSession(i, &control))) continue;
        Com<IAudioSessionControl2> control2;
        if (FAILED(control->QueryInterface(__uuidof(IAudioSessionControl2),
                                           reinterpret_cast<void**>(&control2)))) {
            continue;
        }
        DWORD session_pid = 0;
        if (FAILED(control2->GetProcessId(&session_pid)) || session_pid != pid) continue;
        ++found;
        AudioSessionState state = AudioSessionStateInactive;
        control->GetState(&state);
        float volume = -1.0f, peak = -1.0f;
        BOOL mute = FALSE;
        Com<ISimpleAudioVolume> simple;
        if (SUCCEEDED(control->QueryInterface(__uuidof(ISimpleAudioVolume),
                                              reinterpret_cast<void**>(&simple)))) {
            simple->GetMasterVolume(&volume);
            simple->GetMute(&mute);
        }
        Com<IAudioMeterInformation> meter;
        if (SUCCEEDED(control->QueryInterface(__uuidof(IAudioMeterInformation),
                                              reinterpret_cast<void**>(&meter)))) {
            // The loudest of 20 reads over 1 s: what Windows actually mixes for this process.
            peak = 0.0f;
            for (int k = 0; k < 20; ++k) {
                float now = 0.0f;
                meter->GetPeakValue(&now);
                peak = std::max(peak, now);
                Sleep(50);
            }
        }
        BB::Upscale::Checkpoint(fmt::format(
            "audio check {}: shadPS4's Windows audio session on \"{}\"{}: state {}, app volume "
            "{:.2f}, muted {}, output peak {:.4f}",
            pass, name, is_default ? " (the default output device)" : " (NOT the default device)",
            state == AudioSessionStateActive ? "active"
            : state == AudioSessionStateExpired ? "expired"
                                                : "inactive",
            volume, mute ? "YES" : "no", peak));
        if (fix && simple && (mute || (volume >= 0.0f && volume < 0.05f))) {
            simple->SetMute(FALSE, nullptr);
            simple->SetMasterVolume(1.0f, nullptr);
            BB::Upscale::Checkpoint(
                "audio check: this exe's per-app volume in the Windows mixer was muted / at 0 "
                "(Windows keeps it per exe path): set back to 100% for this run. "
                "(BB_AUDIO_FIX=0 leaves it alone.)");
        }
        if (!is_default && state == AudioSessionStateActive) {
            BB::Upscale::Checkpoint(fmt::format(
                "audio check: WARNING: Windows routes this exe to \"{}\", not to the default "
                "output device (Settings > System > Sound > Volume mixer > shadPS4: output "
                "device). Set it to \"Default\".",
                name));
        }
    }
    return found;
}

std::atomic<bool> g_opened{false};
std::atomic<int> g_open_count{0};

std::string Hr(HRESULT hr) {
    return fmt::format("0x{:08x}", static_cast<unsigned>(hr));
}

void Pass(int pass, bool fix) {
    Com<IMMDeviceEnumerator> enumerator;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                  __uuidof(IMMDeviceEnumerator),
                                  reinterpret_cast<void**>(&enumerator));
    if (FAILED(hr)) {
        BB::Upscale::Checkpoint(fmt::format(
            "audio check {}: CoCreateInstance(MMDeviceEnumerator) failed {}", pass, Hr(hr)));
        return;
    }
    Com<IMMDevice> def;
    std::wstring def_id;
    hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &def);
    if (FAILED(hr)) {
        BB::Upscale::Checkpoint(fmt::format(
            "audio check {}: no default output device (GetDefaultAudioEndpoint {})", pass, Hr(hr)));
    } else {
        LPWSTR id = nullptr;
        if (SUCCEEDED(def->GetId(&id))) {
            def_id = id;
            CoTaskMemFree(id);
        }
        Com<IAudioEndpointVolume> endpoint;
        float level = -1.0f;
        BOOL mute = FALSE;
        hr = def->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr,
                           reinterpret_cast<void**>(&endpoint));
        if (SUCCEEDED(hr)) {
            endpoint->GetMasterVolumeLevelScalar(&level);
            endpoint->GetMute(&mute);
        }
        BB::Upscale::Checkpoint(fmt::format(
            "audio check {}: default output device \"{}\": system volume {:.2f}, muted {}{}", pass,
            DeviceName(def.p), level, mute ? "YES" : "no",
            SUCCEEDED(hr) ? "" : " (endpoint volume " + Hr(hr) + ")"));
    }
    Com<IMMDeviceCollection> devices;
    UINT n = 0;
    int found = 0;
    hr = enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices);
    if (FAILED(hr)) {
        BB::Upscale::Checkpoint(
            fmt::format("audio check {}: EnumAudioEndpoints failed {}", pass, Hr(hr)));
    } else {
        devices->GetCount(&n);
        for (UINT i = 0; i < n; ++i) {
            Com<IMMDevice> device;
            if (FAILED(devices->Item(i, &device))) continue;
            LPWSTR id = nullptr;
            bool is_default = false;
            if (SUCCEEDED(device->GetId(&id))) {
                is_default = def_id == id;
                CoTaskMemFree(id);
            }
            found += CheckEndpoint(device.p, is_default, fix, pass);
        }
    }
    if (found == 0) {
        BB::Upscale::Checkpoint(fmt::format(
            "audio check {}: no Windows audio session for this process on any of the {} "
            "output devices ({} SDL audio stream(s) opened so far)",
            pass, n, g_open_count.load()));
    }
}

void Run() {
    BB::Upscale::Checkpoint("audio check: thread started");
    const char* fix_env = std::getenv("BB_AUDIO_FIX");
    const bool fix = fix_env == nullptr || fix_env[0] != '0';
    // Its own MTA; if this thread somehow has an apartment already, use it as it is.
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(init) && init != RPC_E_CHANGED_MODE) {
        BB::Upscale::Checkpoint(fmt::format("audio check: CoInitializeEx failed {}", Hr(init)));
        return;
    }
    // Pass 1 three seconds after the game opens its first audio stream (at most 40 s after start),
    // then 30 s and 90 s later.
    for (int waited = 0; !g_opened.load() && waited < 40000; waited += 250) Sleep(250);
    if (!g_opened.load()) {
        BB::Upscale::Checkpoint("audio check: the game has not opened an audio stream after 40 s");
    }
    for (int pass = 1; pass <= 3; ++pass) {
        Sleep(pass == 1 ? 3000 : pass == 2 ? 30000 : 60000);
        Pass(pass, fix);
    }
    if (SUCCEEDED(init)) CoUninitialize();
}

} // namespace

void StartCheck() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    try {
        std::thread(Run).detach();
    } catch (...) {
        BB::Upscale::Checkpoint("audio check: could not start its thread");
    }
}

void OnStreamOpened() {
    g_open_count.fetch_add(1);
    g_opened.store(true);
}

} // namespace BB::Audio

#else

namespace BB::Audio {
void StartCheck() {}
void OnStreamOpened() {}
} // namespace BB::Audio

#endif
