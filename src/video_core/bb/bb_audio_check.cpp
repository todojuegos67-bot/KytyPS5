// SPDX-FileCopyrightText: Copyright 2026 bb-port contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Windows audio session check (bb_audio_check.h). Runs on its own thread with its own COM
// apartment; it never touches the emulator's audio code, SDL or guest memory.
#include "video_core/bb/bb_audio_check.h"

#include <algorithm>
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

void Run() {
    const char* fix_env = std::getenv("BB_AUDIO_FIX");
    const bool fix = fix_env == nullptr || fix_env[0] != '0';
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return;
    for (int pass = 1; pass <= 3; ++pass) {
        Sleep(pass == 1 ? 25000 : 60000);
        Com<IMMDeviceEnumerator> enumerator;
        if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                    __uuidof(IMMDeviceEnumerator),
                                    reinterpret_cast<void**>(&enumerator)))) {
            break;
        }
        Com<IMMDevice> def;
        std::wstring def_id;
        if (SUCCEEDED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &def))) {
            LPWSTR id = nullptr;
            if (SUCCEEDED(def->GetId(&id))) {
                def_id = id;
                CoTaskMemFree(id);
            }
            Com<IAudioEndpointVolume> endpoint;
            if (SUCCEEDED(def->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr,
                                        reinterpret_cast<void**>(&endpoint)))) {
                float level = -1.0f;
                BOOL mute = FALSE;
                endpoint->GetMasterVolumeLevelScalar(&level);
                endpoint->GetMute(&mute);
                BB::Upscale::Checkpoint(fmt::format(
                    "audio check {}: default output device \"{}\": system volume {:.2f}, muted {}",
                    pass, DeviceName(def.p), level, mute ? "YES" : "no"));
            }
        }
        Com<IMMDeviceCollection> devices;
        UINT n = 0;
        int found = 0;
        if (SUCCEEDED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices))) {
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
                "output devices (the emulator has not opened its audio stream)",
                pass, n));
        }
    }
    CoUninitialize();
}

} // namespace

void StartCheck() {
    static bool started = false;
    if (started) return;
    started = true;
    std::thread(Run).detach();
}

} // namespace BB::Audio

#else

namespace BB::Audio {
void StartCheck() {}
} // namespace BB::Audio

#endif
