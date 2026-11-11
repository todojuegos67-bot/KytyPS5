#include "libs/dualSenseHaptics.h"

#include "SDL.h"
#include "common/logging/log.h"
#include "libs/dualSenseBluetooth.h"
#include "SDL_hidapi.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

// Shared by the Windows and Linux routes below: the rumble motors driven from each haptic block, and the
// knobs of that drive (the per-platform code opens the pad's endpoint and queues the blocks).
namespace Libs::Controller::DualSenseHaptics {
namespace {

// How hard the rumble motors are driven from a haptic block's peak: 2.0 is the mainline's reference, 0
// leaves the work to the actuators alone (faint), 4 doubles it. Strength is taste - too low reads as "I
// cannot feel it", too high as "it buzzes through a whole spell charge" - so it is a setting, not a
// constant: KYTY_DUALSENSE_VIBRATION_MOTOR_GAIN.
float    g_vibration_motor_gain = 2.0f;
// How long the motors stay on after each block, and how fast the hold falls between blocks. Together
// they are the tail of a hit: a report lasts motor_ms, and the level carries over at motor_decay per
// 21.3 ms report. A hit is a single block, so 60 ms / 0.6 ends it in about 250 ms; longer or slower
// reads as a hit that keeps rumbling (and a whole spell charge as one long one).
uint32_t g_vibration_motor_ms    = 60;
float    g_vibration_motor_decay = 0.6f;
// The game's own level (before the vibration gain) a block needs to drive the motors: a quiet effect the game keeps
// playing stays on the actuators, as on a PS5. An enchanted weapon's hum (0.005-0.01, the whole buff long) rumbled
// until the buff ran out (issue #239); hits reach 0.25. KYTY_DUALSENSE_VIBRATION_MOTOR_ONSET. (Upstream 5aa407e.)
float    g_vibration_motor_onset = 0.025f;
// A block's level less its average over this time (0: off), as KYTY_PAD_RUMBLE_SUSTAIN_MS for other pads: a hit drives
// the motors, an effect that goes on fades from them (onto the actuators alone, as on a PS5). Issue #239, again: in
// v20261031 an enchanted weapon's hum came at 0.13-0.39 after the gain 4 (the earlier log's 0.02-0.04 was a quieter
// spell), a minute long, over the onset all along; hits reach 0.8-1.0. KYTY_DUALSENSE_VIBRATION_MOTOR_SUSTAIN_MS.
float    g_vibration_motor_sustain_ms = 500.0f;

void ReadMotorSettings() {
	// KYTY_DUALSENSE_VIBRATION_MOTOR_GAIN=<float> scales the rumble motors' drive (0 leaves them off
	// and the actuators alone, 2.0 is the mainline's reference, 4 doubles it).
	if (const char* value = std::getenv("KYTY_DUALSENSE_VIBRATION_MOTOR_GAIN"); value != nullptr) {
		const auto gain = std::strtof(value, nullptr);
		if (gain >= 0.0f && gain <= 16.0f) {
			g_vibration_motor_gain = gain;
		}
	}
	// KYTY_DUALSENSE_VIBRATION_MOTOR_MS=<0..500> and KYTY_DUALSENSE_VIBRATION_MOTOR_DECAY=<0..1>: the
	// tail of a hit - how long each report lasts and how fast the hold falls between reports.
	if (const char* value = std::getenv("KYTY_DUALSENSE_VIBRATION_MOTOR_MS"); value != nullptr) {
		const auto ms = std::strtoul(value, nullptr, 10);
		// At least 1: SDL2 reads 0 ms as "no expiry", which would run until the next call.
		if (ms >= 1 && ms <= 500) {
			g_vibration_motor_ms = static_cast<uint32_t>(ms);
		}
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_VIBRATION_MOTOR_DECAY"); value != nullptr) {
		const auto decay = std::strtof(value, nullptr);
		// Below 1.0: at exactly 1 the hold never reaches the floor.
		if (decay > 0.0f && decay < 1.0f) {
			g_vibration_motor_decay = decay;
		}
	}
	// KYTY_DUALSENSE_VIBRATION_MOTOR_ONSET=<0..1>: the game's level a block needs to drive the motors (0: any).
	if (const char* value = std::getenv("KYTY_DUALSENSE_VIBRATION_MOTOR_ONSET"); value != nullptr) {
		const auto onset = std::strtof(value, nullptr);
		if (onset >= 0.0f && onset <= 1.0f) {
			g_vibration_motor_onset = onset;
		}
	}
	// KYTY_DUALSENSE_VIBRATION_MOTOR_SUSTAIN_MS=<0..5000>: how long a steady effect keeps driving the motors (0: off).
	if (const char* value = std::getenv("KYTY_DUALSENSE_VIBRATION_MOTOR_SUSTAIN_MS"); value != nullptr) {
		const auto ms = std::strtof(value, nullptr);
		if (ms >= 0.0f && ms <= 5000.0f) {
			g_vibration_motor_sustain_ms = ms;
		}
	}
}

extern float g_gain; // the route's vibration gain (the Windows and Linux sections below each define it)

// Drives the pad's rumble motors from a block of haptic audio: the mainline's fallback for pads with no
// haptic-audio path (its Queue(): peak -> SDL_RumbleGamepad), used here for a DualSense on Bluetooth,
// where the actuators are heard far more than they are felt.
void RumbleFromHaptics(float left, float right, float block_ms) {
	// Hold with a decay instead of dropping to the block's own peak: consecutive hits arrive in separate
	// blocks, and the motors should not fall silent in the gaps between them.
	//
	// The decay ends at a floor, on purpose. An asymptotic one never reached zero, so every block kept
	// re-arming the motors with a smaller-but-nonzero strength and a single hit rumbled for over a
	// second. 0.6 per 21.3 ms report falls from a full hit to the floor in about 190 ms, and the
	// duration below adds its own tail - short enough to end with the hit, long enough to bridge two or
	// three blocks.
	constexpr float FLOOR = 0.01f;
	static float    held[2] = {0.0f, 0.0f};
	// (The average by the block's own length: the routes' writes can come in bursts. #239's stream, simulated: the
	// motors on 50.1 of 60 s without, 3.4 s with 500 ms, each hit's first block still 0.66-0.75, full at the gain 2.)
	static float average[2] = {0.0f, 0.0f};
	if (g_vibration_motor_sustain_ms > 0.0f) {
		const float weight = 1.0f - std::exp(-std::max(block_ms, 0.0f) / g_vibration_motor_sustain_ms);
		average[0] += (left - average[0]) * weight;
		average[1] += (right - average[1]) * weight;
		left  = std::max(0.0f, left - average[0]);
		right = std::max(0.0f, right - average[1]);
	}
	// (Under the onset a block only lets the hold fall: the peaks here carry the vibration gain. Upstream 5aa407e.)
	const float onset = g_vibration_motor_onset * g_gain;
	left              = left >= onset ? left : 0.0f;
	right             = right >= onset ? right : 0.0f;
	held[0] = std::max(std::max(left, 0.0f), held[0] * g_vibration_motor_decay);
	held[1] = std::max(std::max(right, 0.0f), held[1] * g_vibration_motor_decay);
	const bool live = held[0] >= FLOOR || held[1] >= FLOOR;
	// Just crossed the floor: stop the motors explicitly. Relying on the report's own duration would not
	// do - SDL2 reads 0 ms as "no expiry" - and a game that stops writing in the middle of a hit would
	// otherwise leave them running at the last level, a fade that never finishes.
	const bool stop = !live && (held[0] != 0.0f || held[1] != 0.0f);
	if (!live) {
		held[0] = held[1] = 0.0f;
		if (!stop) {
			return;
		}
	}
	const auto to_motor = [](float value) {
		return static_cast<uint16_t>(
		    std::clamp(value * g_vibration_motor_gain, 0.0f, 1.0f) * 65535.0f);
	};
	const int count = SDL_NumJoysticks();
	for (int i = 0; i < count; i++) {
		auto* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i));
		if (pad == nullptr || SDL_GameControllerGetType(pad) != SDL_CONTROLLER_TYPE_PS5) {
			continue;
		}
		if (stop) {
			(void)SDL_GameControllerRumble(pad, 0, 0, 0);
		} else {
			(void)SDL_GameControllerRumble(pad, to_motor(held[0]), to_motor(held[1]),
			                               g_vibration_motor_ms);
		}
		return;
	}
}

} // namespace
} // namespace Libs::Controller::DualSenseHaptics

#if defined(_WIN32)

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <windows.h>

namespace Libs::Controller::DualSenseHaptics {

namespace {

// PKEY_DeviceInterface_FriendlyName: the device's own name, "USB\VID_054C&PID_0CE6&MI_00" for the
// pad's audio interface. The endpoint's own name (PKEY_Device_DeviceDesc) is just "Speakers".
constexpr PROPERTYKEY PKEY_INTERFACE_FRIENDLY_NAME = {
    {0xb3f8fa53, 0x0004, 0x438e, {0x90, 0x03, 0x51, 0xa4, 0x6e, 0x13, 0x9b, 0xfc}}, 2};
// PKEY_Device_FriendlyName: the endpoint's whole name, "Speakers (DualSense Wireless Controller)".
constexpr PROPERTYKEY PKEY_DEVICE_FRIENDLY_NAME = {
    {0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};

// The pad's USB interface path, as both the audio driver and the registry report it: a DualSense's, or a DualSense
// Edge's (the same four channels; matched on Bluetooth already, while wired it vibrated nowhere).
constexpr const char* PAD_DEVICE_PATHS[] = {"VID_054C&PID_0CE6", "VID_054C&PID_0DF2"};

// The endpoint's four channels: 0-1 play on the pad's speaker, 2-3 drive its haptic actuators.

constexpr uint32_t ROUTE_SPEAKER   = 1U << 0U;
constexpr uint32_t ROUTE_VIBRATION = 1U << 1U;

// One client per route - [0] the actuators, [1] the speaker - as the mainline has one audio stream per
// route. Sharing one client meant each route's whole-buffer memset silenced the other route's channels,
// and both routes counted into the same padding, so the endpoint kept dropping writes: the speaker then
// sounded muffled and chopped. Each client now writes only its own channel pair and leaves the rest at
// zero, which the endpoint's mixer sums back together.
IAudioClient*       g_client[2]        = {nullptr, nullptr};
IAudioRenderClient* g_render[2]        = {nullptr, nullptr};
WAVEFORMATEX*       g_mix[2]           = {nullptr, nullptr};
uint32_t            g_buffer_frames[2] = {0, 0};
uint32_t            g_routes        = 0;

std::mutex g_mutex;
int        g_found    = -1; // -1 unknown, 0 no endpoint, 1 yes
bool       g_enabled  = true;
// The same pad over Bluetooth: no audio endpoint exists there, its speaker and actuators travel in
// HID reports (dualSenseBluetooth.cpp). One stream per route, opened on first use.
DualSenseBluetooth::Stream* g_bt_speaker   = nullptr;
DualSenseBluetooth::Stream* g_bt_vibration = nullptr;

// True when a DualSense sits on the Bluetooth bus. SDL2's hidapi has no bus type, so the Windows
// device path's HID-over-Bluetooth service GUID says which bus it is on.
bool BluetoothPadPresent() {
	for (const uint16_t product: {static_cast<uint16_t>(0x0ce6), static_cast<uint16_t>(0x0df2)}) {
		auto* devices = SDL_hid_enumerate(0x054c, product);
		bool  found   = false;
		for (auto* device = devices; device != nullptr; device = device->next) {
			if (device->path == nullptr) {
				continue;
			}
			std::string path;
			for (const char* c = device->path; *c != 0; ++c) {
				path.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*c))));
			}
			if (path.find("00001124-0000-1000-8000-00805f9b34fb") != std::string::npos) {
				found = true;
				break;
			}
		}
		SDL_hid_free_enumeration(devices);
		if (found) {
			return true;
		}
	}
	return false;
}
bool       g_log      = false;
// 4.0 on the vibration route is deliberate: the actuators are felt rather than heard, and at unity they
// are barely there in play. The speaker route below stays at unity because there the pad's own amplifier
// turns the same difference into clipping. Both are overridable (KYTY_DUALSENSE_HAPTICS_GAIN and
// _SPEAKER_GAIN), and the per-route "clipped" count in KYTY_DUALSENSE_HAPTICS_LOG output says whether a
// value runs past full scale.
float      g_gain     = 4.0f;
bool       g_all_channels  = false;
uint32_t   g_first_channel = 2; // the actuator pair; KYTY_DUALSENSE_HAPTICS_CHANNELS overrides it
uint32_t   g_channel_count = 2;
// The speaker route has its own channels and gain; the gain itself is explained below.
uint32_t   g_speaker_first_channel = 0;
uint32_t   g_speaker_channel_count = 2;
// Unity by default, as the Bluetooth route uses: the pad's own amplifier and the +6 dB preamp the
// routing report asks for are what make this audible, and anything above unity clips in the pad -
// heard as tearing, not as loudness. Raise it here only to taste.
float      g_speaker_gain          = 1.0f;
// The controller's own +6 dB speaker preamp (HID output report byte 37). github.com/Jetsku/KytyPS5
// sends it, Linux does too, and so does the DualSense add-on that is known to play without tearing -
// all with a digital gain of 1.0. The tearing came from this route's own gain, not from the preamp, so
// it is on by default and the gain above is what has to stay inside the scale.
bool       g_speaker_preamp        = true;
// The controller's own speaker volume (HID output report byte 5, 0..100). Lowering it leaves the
// pad's amplifier more headroom, and this route's digital gain can take over the loudness.
uint32_t   g_speaker_volume        = 100;
// A soft limiter above 0.8 on the speaker route only: it keeps loud moments from reaching full
// scale, where the controller's amplifier clips (heard as tearing). Off by default - it colours
// what it touches, and the gain knobs are the first thing to try.
bool       g_speaker_limiter       = false;

// What the routes actually did, so a log can tell clipping from silence and from a stalled endpoint.
uint64_t   g_speaker_writes        = 0;
uint64_t   g_vibration_writes      = 0;
uint64_t   g_speaker_clips         = 0;
uint64_t   g_vibration_clips       = 0;
uint64_t   g_dropped_writes        = 0;
float      g_speaker_peak_max      = 0.0f;
float      g_vibration_peak_max    = 0.0f;

std::string Lower(std::string text) {
	for (auto& c: text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return text;
}

std::string Narrow(const wchar_t* text) {
	std::string out;
	if (text == nullptr) {
		return out;
	}
	for (; *text != L'\0'; text++) {
		out.push_back(*text < 128 ? static_cast<char>(*text) : '?');
	}
	return out;
}

// The pad's speaker stays silent until the host routes the endpoint's audio to it: the controller's
// own mixer sits on the headphone output, and the USB endpoint only carries samples to it. These are
// the values the Linux driver sends with no headphones plugged in, through output report 0x02: route
// the right channel to the speaker, mute the headphones, speaker volume 100 and the +6 dB preamp.
// github.com/Jetsku/KytyPS5 routes it the same way ("the values Linux uses without headphones"), and
// without it the speaker plays nothing here while the pad's own "test" tone stays faint. Passing
// false restores the headphone routing.
bool RouteSpeaker(bool speaker) {
	std::array<uint8_t, 38> report {};
	report[0] = 0x80; // audio control valid
	if (speaker) {
		report[0] |= 0x20; // speaker volume valid
		report[1]  = 0x80; // audio control 2 valid
		report[5]  = static_cast<uint8_t>(g_speaker_volume > 100 ? 100 : g_speaker_volume);
		report[7]  = 0x30; // output path: right channel to the speaker, headphones muted
		report[37] = g_speaker_preamp ? 0x02 : 0x00; // speaker preamp, +6 dB when asked for
	}

	const int devices = SDL_NumJoysticks();
	for (int i = 0; i < devices; i++) {
		auto* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i));
		if (pad == nullptr || SDL_GameControllerGetType(pad) != SDL_CONTROLLER_TYPE_PS5) {
			continue;
		}
		const bool sent =
		    SDL_GameControllerSendEffect(pad, report.data(), static_cast<int>(report.size())) == 0;
		std::printf("DualSense: audio routing -> %s%s%s\n", speaker ? "the pad's speaker" : "headphones",
		            speaker && g_speaker_preamp ? " (+6 dB preamp)" : "",
		            sent ? "" : " (the HID report was refused)");
		return sent;
	}
	return false;
}

void ReadSettings() {
	static bool read = false;
	if (read) {
		return;
	}
	read = true;

	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS"); value != nullptr) {
		g_enabled = std::string(value) != "0";
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_CHANNELS"); value != nullptr) {
		const std::string channels(value);
		if (channels == "all") {
			g_all_channels = true;
		} else if (channels == "01") {
			g_first_channel = 0;
			g_channel_count = 2;
		} else if (channels == "23") {
			g_first_channel = 2;
			g_channel_count = 2;
		} else if (!channels.empty()) {
			g_first_channel = static_cast<uint32_t>(channels[0] - '0');
			g_channel_count = channels.size() > 1 ? static_cast<uint32_t>(channels[1] - '0') : 1;
			g_channel_count = g_channel_count > g_first_channel
			                      ? g_channel_count - g_first_channel + 1
			                      : 1;
		}
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_GAIN"); value != nullptr) {
		const auto gain = std::strtof(value, nullptr);
		if (gain > 0.0f && gain <= 4.0f) {
			g_gain = gain;
		}
	}
	ReadMotorSettings();
	// KYTY_DUALSENSE_SPEAKER_CHANNELS=01|23|all and KYTY_DUALSENSE_SPEAKER_GAIN=<float>: which
	// channels the pad speaker port plays on (0-1 by default, as the mainline and this module's
	// first version do) and how much level it gets. Both keep the pad's USB id selection.
	if (const char* value = std::getenv("KYTY_DUALSENSE_SPEAKER_CHANNELS"); value != nullptr) {
		const std::string channels(value);
		if (channels == "23") {
			g_speaker_first_channel = 2;
			g_speaker_channel_count = 2;
		} else if (channels == "01") {
			g_speaker_first_channel = 0;
			g_speaker_channel_count = 2;
		} else if (channels == "all") {
			g_all_channels = true;
		}
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_SPEAKER_GAIN"); value != nullptr) {
		const auto gain = std::strtof(value, nullptr);
		if (gain >= 0.0f && gain <= 64.0f) {
			g_speaker_gain = gain;
		}
	}
	// KYTY_DUALSENSE_SPEAKER_PREAMP=0 turns the controller's own +6 dB speaker preamp off. It is on by
	// default: the mainline, Linux and the DualSense add-on all send it, and with unity digital gain it
	// is what makes the speaker audible without clipping inside the pad.
	if (const char* value = std::getenv("KYTY_DUALSENSE_SPEAKER_PREAMP"); value != nullptr) {
		g_speaker_preamp = std::string(value) != "0";
	}
	// KYTY_DUALSENSE_SPEAKER_VOLUME=0..100 is the controller's own volume, and
	// KYTY_DUALSENSE_SPEAKER_LIMITER=1 a soft limiter above 0.8 on the speaker route.
	if (const char* value = std::getenv("KYTY_DUALSENSE_SPEAKER_VOLUME"); value != nullptr) {
		const auto volume = std::strtoul(value, nullptr, 10);
		if (volume <= 100) {
			g_speaker_volume = static_cast<uint32_t>(volume);
		}
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_SPEAKER_LIMITER"); value != nullptr) {
		g_speaker_limiter = std::string(value) != "0";
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_LOG"); value != nullptr) {
		g_log = std::string(value) != "0";
	}

	// The settings actually in use, once: the values the routes and the motors run with.
	std::printf("DualSense: settings: %s, vibration gain %.2f on channels %u-%u, motors %.2f (%u ms, "
	            "decay %.2f, onset %.3f, sustain %.0f ms), speaker gain %.2f on channels %u-%u, preamp %s, volume %u, "
	            "limiter %s%s\n",
	            g_enabled ? "on" : "off", static_cast<double>(g_gain), g_first_channel,
	            g_first_channel + g_channel_count - 1u, static_cast<double>(g_vibration_motor_gain),
	            g_vibration_motor_ms, static_cast<double>(g_vibration_motor_decay),
	            static_cast<double>(g_vibration_motor_onset), static_cast<double>(g_vibration_motor_sustain_ms),
	            static_cast<double>(g_speaker_gain),
	            g_speaker_first_channel, g_speaker_first_channel + g_speaker_channel_count - 1u,
	            g_speaker_preamp ? "on (+6 dB)" : "off", g_speaker_volume,
	            g_speaker_limiter ? "on" : "off", g_all_channels ? ", all channels" : "");
}

// COM is initialized per thread and never uninitialized: the stream outlives any single call, and
// releasing its last reference after CoUninitialize is not allowed. S_FALSE (already initialized)
// and RPC_E_CHANGED_MODE (another apartment) are both fine here.
void EnsureCom() {
	static thread_local bool done = false;
	if (!done) {
		done = true;
		(void)CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	}
}

// The pad's output endpoint: exactly one active render device whose interface path is the pad's.
IMMDevice* FindEndpoint() {
	IMMDeviceEnumerator* enumerator = nullptr;
	auto hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_INPROC_SERVER,
	                           __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator));
	if (FAILED(hr)) {
		if (g_log) {
			std::printf("DualSense: CoCreateInstance failed: 0x%08x\n", static_cast<uint32_t>(hr));
		}
		return nullptr;
	}

	IMMDeviceCollection* collection = nullptr;
	hr = enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &collection);
	if (FAILED(hr) || collection == nullptr) {
		enumerator->Release();
		return nullptr;
	}

	UINT  count   = 0;
	collection->GetCount(&count);

	// By its interface path (any case), else by its name: issue #146's wired pad was listed by SDL as "Speakers
	// (DualSense Wireless Controller)" while no path here named its USB id, and it never vibrated.
	IMMDevice*  matched = nullptr;
	IMMDevice*  by_name = nullptr;
	uint32_t    names   = 0;
	std::string named;
	for (UINT i = 0; i < count; i++) {
		IMMDevice* device = nullptr;
		if (FAILED(collection->Item(i, &device)) || device == nullptr) {
			continue;
		}

		IPropertyStore* store = nullptr;
		bool            is_pad = false, has_name = false;
		std::string     path, name;
		if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &store)) && store != nullptr) {
			PROPVARIANT value;
			PropVariantInit(&value);
			if (SUCCEEDED(store->GetValue(PKEY_INTERFACE_FRIENDLY_NAME, &value)) &&
			    value.vt == VT_LPWSTR && value.pwszVal != nullptr) {
				path   = Lower(Narrow(value.pwszVal));
				is_pad = std::ranges::any_of(PAD_DEVICE_PATHS,
				                             [&](const char* pad) { return path.find(Lower(pad)) != std::string::npos; });
			}
			PropVariantClear(&value);
			PropVariantInit(&value);
			if (SUCCEEDED(store->GetValue(PKEY_DEVICE_FRIENDLY_NAME, &value)) && value.vt == VT_LPWSTR &&
			    value.pwszVal != nullptr) {
				name     = Narrow(value.pwszVal);
				has_name = Lower(name).find("dualsense") != std::string::npos;
			}
			PropVariantClear(&value);
			store->Release();
		}
		if (g_log) {
			std::printf("DualSense: endpoint %u: %s (%s)%s\n", i, path.c_str(), name.c_str(),
			            is_pad ? "  <- the pad" : has_name ? "  <- named the pad" : "");
		}

		if (is_pad) {
			if (matched != nullptr) {
				// Two pads: which one is playing is not knowable here, and driving the wrong one
				// would be worse than not driving any (the mainline refuses ambiguous endpoints too).
				device->Release();
				matched->Release();
				matched = nullptr;
				std::printf("DualSense: more than one pad endpoint, none is used\n");
				break;
			}
			matched = device;
			continue;
		}
		if (has_name && names++ == 0) {
			by_name = device;
			named   = path + " (" + name + ")";
			continue;
		}
		device->Release();
	}
	if (matched == nullptr && by_name != nullptr && names == 1) {
		std::printf("DualSense: the pad's endpoint found by its name, not its USB id: %s\n", named.c_str());
		matched = by_name;
		by_name = nullptr;
	} else if (names > 1 && matched == nullptr) {
		std::printf("DualSense: %u endpoints named like the pad and none by its USB id, none is used\n", names);
	}
	if (by_name != nullptr) {
		by_name->Release();
	}

	collection->Release();
	enumerator->Release();
	return matched;
}

} // namespace

bool Available() {
	ReadSettings();
	if (!g_enabled) {
		return false;
	}
	if (g_found >= 0) {
		return g_found == 1;
	}

	EnsureCom();

	IMMDevice* device = FindEndpoint();
	g_found           = device != nullptr ? 1 : 0;
	if (device != nullptr) {
		device->Release();
		std::printf("DualSense: the pad's audio endpoint is available (speaker 0-1, actuators %u-%u)\n",
		            g_first_channel, g_first_channel + g_channel_count - 1u);
	} else if (BluetoothPadPresent()) {
		g_found = 1;
		std::printf("DualSense: no audio endpoint; a Bluetooth pad carries the speaker and the haptics\n");
		return true;
	} else {
		std::printf("DualSense: no audio endpoint found (no pad speaker and no haptics)\n");
	}
	return g_found == 1;
}

uint32_t SampleRate() {
	std::lock_guard lock(g_mutex);
	const auto* mix = g_mix[0] != nullptr ? g_mix[0] : g_mix[1];
	return mix != nullptr ? mix->nSamplesPerSec : 0;
}

uint32_t EndpointChannels() {
	std::lock_guard lock(g_mutex);
	const auto* mix = g_mix[0] != nullptr ? g_mix[0] : g_mix[1];
	return mix != nullptr ? mix->nChannels : 0;
}

bool SpeakerActive() {
	std::lock_guard lock(g_mutex);
	return (g_routes & ROUTE_SPEAKER) != 0U;
}

bool Open(bool speaker) {
	ReadSettings();
	if (!g_enabled) {
		return false;
	}

	std::lock_guard lock(g_mutex);
	const auto route = speaker ? ROUTE_SPEAKER : ROUTE_VIBRATION;
	const int  index = speaker ? 1 : 0;
	if ((g_routes & route) != 0U) {
		return true; // this route is already playing
	}

	if (g_client[index] == nullptr) {
		// A Bluetooth pad first: it has no audio endpoint; its speaker and actuators are HID reports.
		// Tried per route rather than once: the game can close a route and open it again, and a close
		// takes this route's stream away. A pad that had already given up on Bluetooth would then look
		// for a USB endpoint it does not have and stay silent until the game restarted.
		if (speaker ? g_bt_speaker == nullptr : g_bt_vibration == nullptr) {
			if (speaker) {
				g_bt_speaker = DualSenseBluetooth::Open(48000, true, 0, nullptr);
			} else {
				g_bt_vibration = DualSenseBluetooth::Open(48000, false, 0, nullptr);
			}
			if (g_bt_vibration != nullptr || g_bt_speaker != nullptr) {
				std::printf("DualSense: a Bluetooth pad carries the haptics (%s) and the speaker (%s)\n",
				            g_bt_vibration != nullptr ? "yes" : "no",
				            g_bt_speaker != nullptr ? "yes" : "no");
			}
		}
		if (speaker ? g_bt_speaker != nullptr : g_bt_vibration != nullptr) {
			// The pad plays its speaker only after the audio-control report routes it there (the
			// same report the USB path sends): without it the reports carry audio it never plays.
			if (speaker) {
				(void)RouteSpeaker(true);
			}
			g_routes |= route;
			return true;
		}

		EnsureCom();

		IMMDevice* device = FindEndpoint();
		if (device == nullptr) {
			return false;
		}
		g_found = 1;

		bool opened = false;
		do {
			if (FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_INPROC_SERVER, nullptr,
			                            reinterpret_cast<void**>(&g_client[index]))) ||
			    g_client[index] == nullptr) {
				break;
			}

			if (FAILED(g_client[index]->GetMixFormat(&g_mix[index])) || g_mix[index] == nullptr) {
				break;
			}
			if (g_mix[index]->wFormatTag != WAVE_FORMAT_IEEE_FLOAT &&
			    g_mix[index]->wFormatTag != WAVE_FORMAT_EXTENSIBLE) {
				break;
			}
			if (g_mix[index]->wBitsPerSample != 32) {
				break;
			}

			// 100 ms of buffer; the pad's own path starts about a second late, which the queue
			// covers as long as the game keeps writing.
			if (FAILED(
			        g_client[index]->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 1000000LL, 0, g_mix[index],
			                                   nullptr))) {
				break;
			}
			if (FAILED(g_client[index]->GetBufferSize(&g_buffer_frames[index]))) {
				break;
			}
			if (FAILED(g_client[index]->GetService(__uuidof(IAudioRenderClient),
			                                       reinterpret_cast<void**>(&g_render[index]))) ||
			    g_render[index] == nullptr) {
				break;
			}
			if (FAILED(g_client[index]->Start())) {
				break;
			}
			opened = true;
		} while (false);

		device->Release();

		if (!opened) {
			Close(speaker);
			std::printf("DualSense: the pad's endpoint could not be opened\n");
			return false;
		}

		std::printf("DualSense: %s stream open: %u Hz, %u ch, buffer %u frames, gains %.2f/%.2f "
		            "(vibration/speaker)\n",
		            speaker ? "speaker" : "vibration",
		            static_cast<unsigned>(g_mix[index]->nSamplesPerSec),
		            static_cast<unsigned>(g_mix[index]->nChannels), g_buffer_frames[index],
		            static_cast<double>(g_gain), static_cast<double>(g_speaker_gain));
	}

	g_routes |= route;
	const auto first = speaker ? g_speaker_first_channel : g_first_channel;
	const auto count = speaker ? g_speaker_channel_count : g_channel_count;
	std::printf("DualSense: %s route on channels %u-%u (gain %.2f)\n", speaker ? "speaker" : "vibration",
	            first, first + count - 1u, static_cast<double>(speaker ? g_speaker_gain : g_gain));
	if (speaker) {
		// Without this the endpoint's samples never reach the pad's speaker (see RouteSpeaker).
		(void)RouteSpeaker(true);
	}
	return true;
}

void Close(bool speaker) {
	std::lock_guard lock(g_mutex);
	const bool had_speaker = (g_routes & ROUTE_SPEAKER) != 0U;
	g_routes &= ~(speaker ? ROUTE_SPEAKER : ROUTE_VIBRATION);
	if (g_routes != 0U) {
		return; // another route is still playing
	}

	if (had_speaker) {
		(void)RouteSpeaker(false); // the pad's headphone routing goes back, as on the console
	}

	if (speaker) {
		DualSenseBluetooth::Close(g_bt_speaker);
		g_bt_speaker = nullptr;
	} else {
		DualSenseBluetooth::Close(g_bt_vibration);
		g_bt_vibration = nullptr;
	}

	const int index = speaker ? 1 : 0;
	if (g_client[index] != nullptr) {
		g_client[index]->Stop();
		g_client[index]->Release();
		g_client[index] = nullptr;
	}
	if (g_render[index] != nullptr) {
		g_render[index]->Release();
		g_render[index] = nullptr;
	}
	if (g_mix[index] != nullptr) {
		CoTaskMemFree(g_mix[index]);
		g_mix[index] = nullptr;
	}
	g_buffer_frames[index] = 0;
}

bool Write(const void* data, uint32_t frames, uint32_t source_rate, uint32_t source_channels,
           bool is_float, bool speaker, const float* gains) {
	std::lock_guard lock(g_mutex);
	if (data == nullptr || frames == 0 || source_channels == 0) {
		return false;
	}
	if ((g_routes & (speaker ? ROUTE_SPEAKER : ROUTE_VIBRATION)) == 0U) {
		return false;
	}

	// Bluetooth: the same mapping and gain, resampled to the 48 kHz its reports carry, then queued.
	if (speaker ? g_bt_speaker != nullptr : g_bt_vibration != nullptr) {
		auto* bt = speaker ? g_bt_speaker : g_bt_vibration;
		constexpr uint32_t bt_rate     = 48000;
		const auto         out_frames  = source_rate == bt_rate
		                                     ? frames
		                                     : static_cast<uint32_t>(static_cast<uint64_t>(frames) * bt_rate / source_rate);
		if (out_frames == 0) {
			return false;
		}
		std::vector<float> mixed(static_cast<size_t>(out_frames) * 2);
		// Unity gain for the speaker, not the USB path's g_speaker_gain: the mainline's Bluetooth path
		// multiplies only by the guest's per-channel gains (its Queue() uses gain = 1), and the extra
		// +12 dB this used to apply drove the pad's small amplifier into clipping - heard as tearing,
		// never as loudness. The USB path keeps its tuning knobs; this one matches the mainline.
		const auto         route_gain = speaker ? 1.0f : g_gain;
		float              motor_peak[2] = {0.0f, 0.0f};
		for (uint32_t i = 0; i < out_frames; i++) {
			const auto source_index = source_rate == bt_rate
			                              ? i
			                              : static_cast<uint32_t>(static_cast<uint64_t>(i) * source_rate / bt_rate);
			if (source_index >= frames) {
				break;
			}
			for (uint32_t c = 0; c < 2; c++) {
				const auto source_channel =
				    source_channels == 1 ? 0u : (c < source_channels ? c : source_channels - 1u);
				float value = 0.0f;
				if (is_float) {
					value = reinterpret_cast<const float*>(data)[static_cast<size_t>(source_index) *
					                                                 source_channels + source_channel];
				} else {
					value = static_cast<float>(reinterpret_cast<const int16_t*>(data)[
					            static_cast<size_t>(source_index) * source_channels + source_channel]) /
					        32768.0f;
				}
				if (gains != nullptr) {
					value *= gains[source_channel];
				}
				value *= route_gain;
				if (speaker ? g_speaker_limiter : false) {
					value = value > 0.8f ? 0.8f + (value - 0.8f) / (1.0f + (value - 0.8f) * 4.0f) : value;
					value = value < -0.8f ? -0.8f + (value + 0.8f) / (1.0f - (value + 0.8f) * 4.0f) : value;
				}
				mixed[static_cast<size_t>(i) * 2 + c] = value;
				motor_peak[c] = std::max(motor_peak[c], std::fabs(value));
			}
		}
		(void)DualSenseBluetooth::Queue(bt, mixed.data(), out_frames);
		if (!speaker) {
			RumbleFromHaptics(motor_peak[0], motor_peak[1], static_cast<float>(frames) * 1000.0f / static_cast<float>(source_rate));
		}
		if (speaker) {
			g_speaker_writes++;
		} else {
			g_vibration_writes++;
		}
		return true;
	}

	// The USB endpoint's own client, one per route, so neither silences the other's channels.
	const int index = speaker ? 1 : 0;
	if (g_client[index] == nullptr || g_mix[index] == nullptr) {
		return false;
	}

	const auto target_rate     = g_mix[index]->nSamplesPerSec;
	const auto target_channels = g_mix[index]->nChannels;
	if (target_rate == 0 || target_channels < 2) {
		return false;
	}

	const auto first_channel = speaker ? g_speaker_first_channel : g_first_channel;
	const auto channel_count = speaker ? g_speaker_channel_count : g_channel_count;
	const auto route_gain    = speaker ? g_speaker_gain : g_gain;
	const auto channels      = g_all_channels ? target_channels : channel_count;

	// Resample (linear: enough for haptics) and put the signal in the route's channels.
	const auto out_frames = source_rate == target_rate
	                            ? frames
	                            : static_cast<uint32_t>(static_cast<uint64_t>(frames) * target_rate /
	                                                    source_rate);
	if (out_frames == 0) {
		return false;
	}

	UINT32 padding = 0;
	if (FAILED(g_client[index]->GetCurrentPadding(&padding))) {
		return false;
	}
	const auto free =
	    g_buffer_frames[index] > padding ? g_buffer_frames[index] - padding : 0;
	if (free < out_frames) {
		g_dropped_writes++;
		return false; // behind: drop rather than delay the guest's audio thread
	}

	BYTE* buffer = nullptr;
	if (FAILED(g_render[index]->GetBuffer(out_frames, &buffer)) || buffer == nullptr) {
		return false;
	}

	auto* out = reinterpret_cast<float*>(buffer);
	std::memset(out, 0, static_cast<size_t>(out_frames) * target_channels * sizeof(float));

	float peak = 0.0f;

	for (uint32_t i = 0; i < out_frames; i++) {
		const auto source_index =
		    source_rate == target_rate
		        ? i
		        : static_cast<uint32_t>(static_cast<uint64_t>(i) * source_rate / target_rate);
		if (source_index >= frames) {
			break;
		}

		// Mono fills every channel of the route (a one-channel pad speaker port has to drive both of
		// its channels, or only one side plays), stereo maps one to one.
		for (uint32_t c = 0; c < channels; c++) {
			const auto source_channel =
			    source_channels == 1 ? 0u : (c < source_channels ? c : source_channels - 1u);
			float value = 0.0f;
			if (is_float) {
				value = reinterpret_cast<const float*>(data)[static_cast<size_t>(source_index) *
				                                                 source_channels +
				                                             source_channel];
			} else {
				value = static_cast<float>(reinterpret_cast<const int16_t*>(data)[
				            static_cast<size_t>(source_index) * source_channels + source_channel]) /
				        32768.0f;
			}
			value *= route_gain;
			if (gains != nullptr) {
				value *= gains[source_channel];
			}
			// Count what the gain pushed past full scale before clamping it: the digital domain is
			// where this branch can act, and the controller's own amplifier clips after it.
			if (value > 1.0f || value < -1.0f) {
				if (speaker) {
					g_speaker_clips++;
				} else {
					g_vibration_clips++;
				}
			}
			if (speaker && g_speaker_limiter) {
				// Soft knee above 0.8: what would clamp at full scale bends towards it instead, so the
				// controller's own amplifier sees a peak it can take.
				constexpr float knee     = 0.8f;
				const auto      magnitude = value < 0.0f ? -value : value;
				if (magnitude > knee) {
					const auto excess = (magnitude - knee) / (1.0f - knee);
					value = (value < 0.0f ? -1.0f : 1.0f) *
					        (knee + (1.0f - knee) * (excess / (1.0f + excess)));
				}
			}
			if (value > 1.0f) {
				value = 1.0f;
			} else if (value < -1.0f) {
				value = -1.0f;
			}
			if (value > peak) {
				peak = value;
			} else if (-value > peak) {
				peak = -value;
			}

			if (g_all_channels) {
				for (uint32_t target = 0; target < target_channels; target++) {
					out[static_cast<size_t>(i) * target_channels + target] = value;
				}
				continue;
			}

			const auto channel = first_channel + c;
			if (channel < target_channels) {
				out[static_cast<size_t>(i) * target_channels + channel] = value;
			}
		}
	}

	g_render[index]->ReleaseBuffer(out_frames, 0);

	if (!speaker) {
		// The same motor drive the Bluetooth route uses, from the same block: the endpoint's actuator
		// pair alone is faint too, and a wired pad should feel like a wireless one.
		RumbleFromHaptics(peak, peak, static_cast<float>(frames) * 1000.0f / static_cast<float>(source_rate));
	}

	// What actually reaches the pad, so a log says whether the game fills this route at all, how loud
	// it got and whether the gain ran past full scale: the first writes, then one every 512 (about
	// every 2.7 s of 5 ms calls), with the peak and the run's maximum, clipped samples and drops.
	auto& route_writes = speaker ? g_speaker_writes : g_vibration_writes;
	auto& route_peak   = speaker ? g_speaker_peak_max : g_vibration_peak_max;
	auto& route_clips  = speaker ? g_speaker_clips : g_vibration_clips;
	route_writes++;
	if (peak > route_peak) {
		route_peak = peak;
	}
	if (route_writes <= 4 || (route_writes % 512) == 0) {
		std::printf("DualSense: %s write %llu: %u frames at %u Hz, %u ch, %s, gains %s%.2f, peak "
		            "%.3f (max %.3f), clipped %llu, dropped %llu, queue %u/%u\n",
		            speaker ? "speaker" : "vibration", static_cast<unsigned long long>(route_writes),
		            frames, source_rate, source_channels, is_float ? "float" : "s16",
		            gains != nullptr ? "" : "(none) ",
		            static_cast<double>(gains != nullptr ? gains[0] : 1.0f),
		            static_cast<double>(peak), static_cast<double>(route_peak),
		            static_cast<unsigned long long>(route_clips),
		            static_cast<unsigned long long>(g_dropped_writes), padding, g_buffer_frames[index]);
	}
	return true;
}

} // namespace Libs::Controller::DualSenseHaptics

#elif defined(__linux__)

// Linux (KYTY_DUALSENSE_HAPTICS=1, default OFF here): the pad's USB audio interface is an output device of
// SDL2's audio list too (PipeWire or PulseAudio name it "DualSense Wireless Controller ..."; ALSA "Wireless
// Controller"), four channels, the actuators on the second pair. The vibration port opens that device
// through SDL as the main output is opened (audio.cpp OpenSdlDevice), asks for four float channels at
// 48 kHz, and queues each block into the actuator pair with the Windows route's mapping and gain; the pad
// speaker route stays on the main output here (SpeakerActive() false), as it always did. Best effort as on
// Windows: a block that would push the queue past MAX_QUEUE_MS is dropped. A Bluetooth pad has no audio
// device; its reports (dualSenseBluetooth.cpp) are not wired here yet.
//
//   KYTY_DUALSENSE_HAPTICS_DEVICE=<text>   Linux only: the SDL audio device whose name holds this text
//                                          (case-insensitive) is the pad; default: "dualsense", else
//                                          "wireless controller".

#include "libs/audio.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace Libs::Controller::DualSenseHaptics {

namespace {

constexpr uint32_t ROUTE_VIBRATION = 1U << 1U;
constexpr int      WANTED_RATE     = 48000;
constexpr uint8_t  WANTED_CHANNELS = 4;
constexpr uint16_t WANTED_SAMPLES  = 256;
constexpr uint32_t MAX_QUEUE_MS    = 60; // (the Windows client's buffer is about this deep)

std::mutex g_mutex;
bool       g_settings_read = false;
bool       g_enabled       = false; // KYTY_DUALSENSE_HAPTICS (not "0") turns it on; the Windows default is on
bool       g_log           = false;
float      g_gain          = 4.0f; // as the Windows route: the actuators are felt, not heard
bool       g_all_channels  = false;
uint32_t   g_first_channel = 2; // the actuator pair; KYTY_DUALSENSE_HAPTICS_CHANNELS overrides it
uint32_t   g_channel_count = 2;
std::string g_device_match;

std::string       g_device_name; // (empty: not found yet; looked for again while wanted, a pad plugs in late)
// The game opened its vibration port: the route is wanted, and while it has no device (the pad's device listed a
// moment after the port opened, a pad plugged in later, an unplug) the open is tried again from the guest's audio
// thread, at most once a second (not on every 5 ms grain).
bool                                  g_wanted = false;
std::chrono::steady_clock::time_point g_next_try {};
SDL_AudioDeviceID g_device = 0;
SDL_AudioSpec     g_spec {};
uint32_t          g_routes = 0;

std::vector<float> g_mix; // one block, reused (Write runs on the guest's audio thread)
uint64_t g_vibration_writes   = 0;
uint64_t g_vibration_clips    = 0;
uint64_t g_dropped_writes     = 0;
float    g_vibration_peak_max = 0.0f;

void ReadSettings() {
	if (g_settings_read) {
		return;
	}
	g_settings_read = true;
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS"); value != nullptr) {
		g_enabled = std::string(value) != "0";
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_CHANNELS"); value != nullptr) {
		const std::string channels = Common::ToLower(value);
		if (channels == "all") {
			g_all_channels = true;
		} else if (channels == "01") {
			g_first_channel = 0;
			g_channel_count = 2;
		} else if (channels == "23") {
			g_first_channel = 2;
			g_channel_count = 2;
		} else if (!channels.empty() && std::isdigit(static_cast<unsigned char>(channels[0])) != 0) {
			g_first_channel = static_cast<uint32_t>(channels[0] - '0');
			g_channel_count = channels.size() > 1 && std::isdigit(static_cast<unsigned char>(channels[1])) != 0
			                      ? static_cast<uint32_t>(channels[1] - '0')
			                      : 1u;
			g_channel_count = g_channel_count > g_first_channel ? g_channel_count - g_first_channel + 1u : 1u;
		}
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_GAIN"); value != nullptr) {
		const float gain = std::strtof(value, nullptr);
		if (gain >= 0.0f && gain <= 64.0f) {
			g_gain = gain;
		}
	}
	ReadMotorSettings();
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_DEVICE"); value != nullptr && value[0] != 0) {
		g_device_match = Common::ToLower(value);
	}
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_LOG"); value != nullptr) {
		g_log = std::string(value) != "0";
	}
}

// The pad's output device in SDL's list: the name the caller gave (KYTY_DUALSENSE_HAPTICS_DEVICE), else
// audio.cpp's finder, which takes a DualSense-named device and refuses an ambiguous list (two pads). Looked
// for again while nothing was found: PipeWire adds the pad's device a moment after it is plugged in.
bool FindDevice() {
	if (!g_device_name.empty()) {
		return true;
	}
	if (!Audio::EnsureSdlAudio()) {
		return false;
	}
	const int devices = SDL_GetNumAudioDevices(0);
	if (!g_device_match.empty()) {
		for (int i = 0; i < devices; i++) {
			const char* name = SDL_GetAudioDeviceName(i, 0);
			if (name != nullptr && Common::ToLower(name).find(g_device_match) != std::string::npos) {
				g_device_name = name;
				break;
			}
		}
	} else {
		g_device_name = Audio::FindControllerAudioDevice();
	}
	if (g_device_name.empty()) {
		static int reported = 0;
		if (reported++ < 4) {
			std::printf("DualSense: no pad output device among %d SDL audio devices (USB pad plugged in before the "
			            "game? two pads are refused; AudioOut's device list names them, "
			            "KYTY_DUALSENSE_HAPTICS_DEVICE=<text> picks one)\n",
			            devices);
		}
		return false;
	}
	return true;
}

// The device's own channel count (SDL_GetAudioDeviceSpec), 0 when unknown: a stream opened with four channels
// on a stereo profile is remixed by the sound server into the pad's SPEAKER, so a stereo-profile pad is refused.
uint32_t DeviceChannels() {
	const int devices = SDL_GetNumAudioDevices(0);
	for (int i = 0; i < devices; i++) {
		const char* name = SDL_GetAudioDeviceName(i, 0);
		if (name == nullptr || g_device_name != name) {
			continue;
		}
		SDL_AudioSpec spec {};
		if (SDL_GetAudioDeviceSpec(i, 0, &spec) == 0) {
			return spec.channels;
		}
		break;
	}
	return 0;
}

} // namespace

static bool OpenLocked(bool force);

bool Available() {
	std::lock_guard lock(g_mutex);
	ReadSettings();
	if (!g_enabled || !g_wanted) {
		return false;
	}
	return OpenLocked(false);
}

uint32_t SampleRate() {
	std::lock_guard lock(g_mutex);
	return g_device != 0 ? static_cast<uint32_t>(g_spec.freq) : 0u;
}

uint32_t EndpointChannels() {
	std::lock_guard lock(g_mutex);
	return g_device != 0 ? g_spec.channels : 0u;
}

bool SpeakerActive() { return false; }

// The route's device, opened: `force` skips the retry's rate limit (the port's own Open). Called with g_mutex held.
static bool OpenLocked(bool force) {
	if ((g_routes & ROUTE_VIBRATION) != 0U && g_device != 0) {
		return true; // this route is already playing (as the Windows route answers)
	}
	const auto now = std::chrono::steady_clock::now();
	if (!force && now < g_next_try) {
		return false;
	}
	g_next_try = now + std::chrono::seconds(1);
	if (!FindDevice()) {
		return false;
	}
	if (g_device == 0) {
		if (const auto device_channels = DeviceChannels();
		    device_channels != 0 && !g_all_channels && g_first_channel + g_channel_count > device_channels) {
			std::printf("DualSense: \"%s\" is a %u-channel output: the actuators need the pad's 4-channel profile (PipeWire/"
			            "PulseAudio: set the card's profile to Analog Surround 4.0 Output, e.g. in pavucontrol); not opened\n",
			            g_device_name.c_str(), device_channels);
			return false;
		}
		SDL_AudioSpec desired {};
		desired.freq     = WANTED_RATE;
		desired.format   = AUDIO_F32SYS;
		desired.channels = WANTED_CHANNELS;
		desired.samples  = WANTED_SAMPLES;
		desired.callback = nullptr;
		SDL_AudioSpec obtained {};
		// The rate and the block size may differ (resampled below); the format stays float (SDL converts);
		// the channel count is allowed to differ so a stereo-only device is seen, and then refused.
		g_device = SDL_OpenAudioDevice(g_device_name.c_str(), 0, &desired, &obtained,
		                               SDL_AUDIO_ALLOW_FREQUENCY_CHANGE | SDL_AUDIO_ALLOW_SAMPLES_CHANGE |
		                                   SDL_AUDIO_ALLOW_CHANNELS_CHANGE);
		if (g_device == 0) {
			std::printf("DualSense: cannot open \"%s\": %s\n", g_device_name.c_str(), SDL_GetError());
			return false;
		}
		if (obtained.channels < 2 || obtained.freq <= 0 ||
		    (!g_all_channels && g_first_channel + g_channel_count > obtained.channels)) {
			std::printf("DualSense: \"%s\" opened with %u channel(s) at %d Hz; the actuator pair needs four "
			            "(the pad's Analog Surround 4.0 profile); not used\n",
			            g_device_name.c_str(), obtained.channels, obtained.freq);
			SDL_CloseAudioDevice(g_device);
			g_device = 0;
			return false;
		}
		g_spec = obtained;
		SDL_PauseAudioDevice(g_device, 0);
		std::printf("DualSense: haptics on \"%s\" (%d Hz, %u ch, format 0x%04x, %u-frame blocks), channels %s, gain %.2f\n",
		            g_device_name.c_str(), g_spec.freq, g_spec.channels, g_spec.format, g_spec.samples,
		            g_all_channels ? "all" : (g_first_channel == 2 && g_channel_count == 2 ? "23" : "custom"),
		            static_cast<double>(g_gain));
	}
	g_routes |= ROUTE_VIBRATION;
	return true;
}

bool Open(bool speaker) {
	std::lock_guard lock(g_mutex);
	ReadSettings();
	if (!g_enabled || speaker) {
		return false; // (the pad speaker port keeps the main output here)
	}
	g_wanted = true;
	return OpenLocked(true);
}

void Close(bool speaker) {
	std::lock_guard lock(g_mutex);
	if (speaker) {
		return;
	}
	g_wanted = false;
	g_routes &= ~ROUTE_VIBRATION;
	if (g_device != 0) {
		SDL_ClearQueuedAudio(g_device);
		SDL_CloseAudioDevice(g_device);
		g_device = 0;
		g_spec   = {};
	}
}

bool Write(const void* data, uint32_t frames, uint32_t source_rate, uint32_t source_channels,
           bool is_float, bool speaker, const float* gains) {
	std::lock_guard lock(g_mutex);
	if (speaker || data == nullptr || frames == 0 || source_channels == 0 || source_rate == 0 ||
	    (g_routes & ROUTE_VIBRATION) == 0U || g_device == 0) {
		return false;
	}
	if (SDL_GetAudioDeviceStatus(g_device) == SDL_AUDIO_STOPPED) {
		// The pad was unplugged (SDL stops its device; the queue would never drain): closed, and looked for again.
		std::printf("DualSense: \"%s\" stopped (unplugged?): closed; looked for again while the port is open\n",
		            g_device_name.c_str());
		SDL_ClearQueuedAudio(g_device);
		SDL_CloseAudioDevice(g_device);
		g_device = 0;
		g_spec   = {};
		g_device_name.clear();
		g_routes &= ~ROUTE_VIBRATION;
		return false;
	}
	const auto target_rate     = static_cast<uint32_t>(g_spec.freq);
	const auto target_channels = static_cast<uint32_t>(g_spec.channels);
	const auto channels        = g_all_channels ? target_channels : g_channel_count;
	const auto out_frames      = source_rate == target_rate
	                                 ? frames
	                                 : static_cast<uint32_t>(static_cast<uint64_t>(frames) * target_rate / source_rate);
	if (out_frames == 0) {
		return false;
	}
	// Behind: drop rather than delay the guest's audio thread (and keep a hit from arriving late).
	const auto frame_bytes = target_channels * sizeof(float);
	const auto queued      = SDL_GetQueuedAudioSize(g_device);
	if (queued / frame_bytes > static_cast<uint64_t>(target_rate) * MAX_QUEUE_MS / 1000u) {
		g_dropped_writes++;
		return false;
	}

	auto& out = g_mix;
	out.assign(static_cast<size_t>(out_frames) * target_channels, 0.0f);
	float peak = 0.0f;
	for (uint32_t i = 0; i < out_frames; i++) {
		const auto source_index =
		    source_rate == target_rate ? i : static_cast<uint32_t>(static_cast<uint64_t>(i) * source_rate / target_rate);
		if (source_index >= frames) {
			break;
		}
		// Mono fills every channel of the route, stereo maps one to one (as the Windows route).
		for (uint32_t c = 0; c < channels; c++) {
			const auto source_channel = source_channels == 1 ? 0u : (c < source_channels ? c : source_channels - 1u);
			float      value          = 0.0f;
			if (is_float) {
				value = reinterpret_cast<const float*>(data)[static_cast<size_t>(source_index) * source_channels + source_channel];
			} else {
				value = static_cast<float>(reinterpret_cast<const int16_t*>(data)[static_cast<size_t>(source_index) * source_channels +
				                                                                 source_channel]) /
				        32768.0f;
			}
			value *= g_gain;
			if (gains != nullptr) {
				value *= gains[source_channel];
			}
			if (value > 1.0f || value < -1.0f) {
				g_vibration_clips++;
			}
			value = std::clamp(value, -1.0f, 1.0f);
			peak  = std::max(peak, std::fabs(value));
			if (g_all_channels) {
				for (uint32_t target = 0; target < target_channels; target++) {
					out[static_cast<size_t>(i) * target_channels + target] = value;
				}
				continue;
			}
			const auto channel = g_first_channel + c;
			if (channel < target_channels) {
				out[static_cast<size_t>(i) * target_channels + channel] = value;
			}
		}
	}
	if (SDL_QueueAudio(g_device, out.data(), static_cast<Uint32>(out.size() * sizeof(float))) < 0) {
		g_dropped_writes++;
		return false;
	}
	RumbleFromHaptics(peak, peak, static_cast<float>(frames) * 1000.0f / static_cast<float>(source_rate));

	g_vibration_writes++;
	g_vibration_peak_max = std::max(g_vibration_peak_max, peak);
	if (g_log && (g_vibration_writes <= 4 || (g_vibration_writes % 512) == 0)) {
		std::printf("DualSense: vibration write %llu: %u frames at %u Hz, %u ch, %s, gains %s%.2f, peak %.3f (max %.3f), "
		            "clipped %llu, dropped %llu, queued %u bytes\n",
		            static_cast<unsigned long long>(g_vibration_writes), frames, source_rate, source_channels,
		            is_float ? "float" : "s16", gains != nullptr ? "" : "(none) ",
		            static_cast<double>(gains != nullptr ? gains[0] : 1.0f), static_cast<double>(peak),
		            static_cast<double>(g_vibration_peak_max), static_cast<unsigned long long>(g_vibration_clips),
		            static_cast<unsigned long long>(g_dropped_writes), queued);
	}
	return true;
}

} // namespace Libs::Controller::DualSenseHaptics

#else

namespace Libs::Controller::DualSenseHaptics {

bool     Available() { return false; }
uint32_t SampleRate() { return 0; }
uint32_t EndpointChannels() { return 0; }
bool     Open(bool) { return false; }
void     Close(bool) {}
bool     Write(const void*, uint32_t, uint32_t, uint32_t, bool, bool, const float*) { return false; }
bool     SpeakerActive() { return false; }

} // namespace Libs::Controller::DualSenseHaptics

#endif
