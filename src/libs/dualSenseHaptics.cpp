#include "libs/dualSenseHaptics.h"

#include "SDL.h"
#include "common/logging/log.h"

#if defined(_WIN32)

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#define WIN32_LEAN_AND_MEAN
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <windows.h>

namespace Libs::Controller::DualSenseHaptics {

namespace {

// PKEY_DeviceInterface_FriendlyName: the device's own name, "USB\VID_054C&PID_0CE6&MI_00" for the
// pad's audio interface. The endpoint's own name (PKEY_Device_FriendlyName) is just "Speakers".
constexpr PROPERTYKEY PKEY_INTERFACE_FRIENDLY_NAME = {
    {0xb3f8fa53, 0x0004, 0x438e, {0x90, 0x03, 0x51, 0xa4, 0x6e, 0x13, 0x9b, 0xfc}}, 2};

// The pad's USB interface path, as both the audio driver and the registry report it.
constexpr const char* PAD_DEVICE_PATH = "VID_054C&PID_0CE6";

// The endpoint's four channels: 0-1 play on the pad's speaker, 2-3 drive its haptic actuators.
constexpr uint32_t SPEAKER_FIRST_CHANNEL = 0;
constexpr uint32_t SPEAKER_CHANNELS      = 2;

constexpr uint32_t ROUTE_SPEAKER   = 1U << 0U;
constexpr uint32_t ROUTE_VIBRATION = 1U << 1U;

IMMDevice*          g_device        = nullptr;
IAudioClient*       g_client        = nullptr;
IAudioRenderClient* g_render        = nullptr;
WAVEFORMATEX*       g_mix           = nullptr;
uint32_t            g_buffer_frames = 0;
uint32_t            g_routes        = 0;

std::mutex g_mutex;
int        g_found    = -1; // -1 unknown, 0 no endpoint, 1 yes
bool       g_enabled  = true;
bool       g_log      = false;
float      g_gain     = 1.0f;
bool       g_all_channels  = false;
uint32_t   g_first_channel = 2; // the actuator pair; KYTY_DUALSENSE_HAPTICS_CHANNELS overrides it
uint32_t   g_channel_count = 2;
// The speaker route has its own channels and gain: the pad's speaker is small, so its port often
// needs more level than the actuators, and which pair carries it is worth an experiment.
uint32_t   g_speaker_first_channel = 0;
uint32_t   g_speaker_channel_count = 2;
float      g_speaker_gain          = 8.0f;

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
		report[5]  = 0x64; // speaker volume, 100 of 100
		report[7]  = 0x30; // output path: right channel to the speaker, headphones muted
		report[37] = 0x02; // speaker preamp, +6 dB
	}

	const int devices = SDL_NumJoysticks();
	for (int i = 0; i < devices; i++) {
		auto* pad = SDL_GameControllerFromInstanceID(SDL_JoystickGetDeviceInstanceID(i));
		if (pad == nullptr || SDL_GameControllerGetType(pad) != SDL_CONTROLLER_TYPE_PS5) {
			continue;
		}
		const bool sent =
		    SDL_GameControllerSendEffect(pad, report.data(), static_cast<int>(report.size())) == 0;
		std::printf("DualSense: audio routing -> %s%s\n", speaker ? "the pad's speaker" : "headphones",
		            sent ? "" : " (the HID report was refused)");
		return sent;
	}
	return false;
}

std::string Trimmed(std::string text) {	const auto spaces = " \t\r\n";
	const auto first  = text.find_first_not_of(spaces);
	if (first == std::string::npos) {
		return {};
	}
	const auto last = text.find_last_not_of(spaces);
	return text.substr(first, last - first + 1);
}

std::string Lowered(std::string text) {
	for (auto& c : text) {
		if (c >= 'A' && c <= 'Z') {
			c = static_cast<char>(c - 'A' + 'a');
		}
	}
	return text;
}

// Channel pairs as text: "01" (the speaker pair), "23" (the actuator pair), "all".
void ParseChannels(const std::string& text, uint32_t* first, uint32_t* count, bool* all) {
	if (text == "all" || text == "4") {
		*all = true;
		return;
	}
	if (text == "01" || text == "0") {
		*first = 0;
		*count = 2;
	} else if (text == "23" || text == "2") {
		*first = 2;
		*count = 2;
	}
}

// dualsense-haptics.txt next to the emulator (the launcher's working directory): one "key=value" a
// line, "#" starts a comment. The environment is awkward to set under the launcher - editing
// run.cmd or a shortcut - and this file is easy to edit, to keep, and to delete. It is applied
// after the environment, so a line here wins.
void ReadSettingsFile() {
	static bool read = false;
	if (read) {
		return;
	}
	read = true;

	auto* file = std::fopen("dualsense-haptics.txt", "rb");
	if (file == nullptr) {
		return;
	}

	char line[256] = {};
	while (std::fgets(line, sizeof(line), file) != nullptr) {
		std::string text(line);
		const auto  hash = text.find('#');
		if (hash != std::string::npos) {
			text.erase(hash);
		}
		const auto equals = text.find('=');
		if (equals == std::string::npos) {
			continue;
		}
		const auto key   = Lowered(Trimmed(text.substr(0, equals)));
		const auto value = Trimmed(text.substr(equals + 1));
		if (key.empty() || value.empty()) {
			continue;
		}

		if (key == "enabled") {
			g_enabled = value != "0" && Lowered(value) != "false" && Lowered(value) != "no";
		} else if (key == "log") {
			g_log = value != "0" && Lowered(value) != "false" && Lowered(value) != "no";
		} else if (key == "gain" || key == "vibration_gain") {
			const auto gain = std::strtof(value.c_str(), nullptr);
			if (gain > 0.0f && gain <= 8.0f) {
				g_gain = gain;
			}
		} else if (key == "speaker_gain") {
			const auto gain = std::strtof(value.c_str(), nullptr);
			if (gain >= 0.0f && gain <= 64.0f) {
				g_speaker_gain = gain;
			}
		} else if (key == "channels" || key == "vibration_channels") {
			ParseChannels(value, &g_first_channel, &g_channel_count, &g_all_channels);
		} else if (key == "speaker_channels") {
			ParseChannels(value, &g_speaker_first_channel, &g_speaker_channel_count, &g_all_channels);
		}
	}
	std::fclose(file);

	std::printf("DualSense: dualsense-haptics.txt: %s, vibration gain %.2f on channels %u-%u, "
	            "speaker gain %.2f on channels %u-%u%s\n",
	            g_enabled ? "on" : "off", static_cast<double>(g_gain), g_first_channel,
	            g_first_channel + g_channel_count - 1u, static_cast<double>(g_speaker_gain),
	            g_speaker_first_channel, g_speaker_first_channel + g_speaker_channel_count - 1u,
	            g_all_channels ? ", all channels" : "");
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
	if (const char* value = std::getenv("KYTY_DUALSENSE_HAPTICS_LOG"); value != nullptr) {
		g_log = std::string(value) != "0";
	}

	// The file last, so it wins over the environment.
	ReadSettingsFile();
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

	IMMDevice* matched = nullptr;
	for (UINT i = 0; i < count; i++) {
		IMMDevice* device = nullptr;
		if (FAILED(collection->Item(i, &device)) || device == nullptr) {
			continue;
		}

		IPropertyStore* store = nullptr;
		bool            is_pad = false;
		if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &store)) && store != nullptr) {
			PROPVARIANT value;
			PropVariantInit(&value);
			if (SUCCEEDED(store->GetValue(PKEY_INTERFACE_FRIENDLY_NAME, &value)) &&
			    value.vt == VT_LPWSTR && value.pwszVal != nullptr) {
				const auto path = Narrow(value.pwszVal);
				is_pad          = path.find(PAD_DEVICE_PATH) != std::string::npos;
				if (g_log) {
					std::printf("DualSense: endpoint %u: %s%s\n", i, path.c_str(),
					            is_pad ? "  <- the pad" : "");
				}
			}
			PropVariantClear(&value);
			store->Release();
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
		device->Release();
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
	} else {
		std::printf("DualSense: no audio endpoint found (no pad speaker and no haptics)\n");
	}
	return g_found == 1;
}

uint32_t SampleRate() {
	std::lock_guard lock(g_mutex);
	return g_mix != nullptr ? g_mix->nSamplesPerSec : 0;
}

uint32_t EndpointChannels() {
	std::lock_guard lock(g_mutex);
	return g_mix != nullptr ? g_mix->nChannels : 0;
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
	if ((g_routes & route) != 0U) {
		return true; // this route is already playing
	}

	if (g_client == nullptr) {
		EnsureCom();

		IMMDevice* device = FindEndpoint();
		if (device == nullptr) {
			return false;
		}
		g_found = 1;

		bool opened = false;
		do {
			if (FAILED(device->Activate(__uuidof(IAudioClient), CLSCTX_INPROC_SERVER, nullptr,
			                            reinterpret_cast<void**>(&g_client))) ||
			    g_client == nullptr) {
				break;
			}

			if (FAILED(g_client->GetMixFormat(&g_mix)) || g_mix == nullptr) {
				break;
			}
			if (g_mix->wFormatTag != WAVE_FORMAT_IEEE_FLOAT &&
			    g_mix->wFormatTag != WAVE_FORMAT_EXTENSIBLE) {
				break;
			}
			if (g_mix->wBitsPerSample != 32) {
				break;
			}

			// 100 ms of buffer; the pad's own path starts about a second late, which the queue
			// covers as long as the game keeps writing.
			if (FAILED(
			        g_client->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 1000000LL, 0, g_mix, nullptr))) {
				break;
			}
			if (FAILED(g_client->GetBufferSize(&g_buffer_frames))) {
				break;
			}
			if (FAILED(g_client->GetService(__uuidof(IAudioRenderClient),
			                                reinterpret_cast<void**>(&g_render))) ||
			    g_render == nullptr) {
				break;
			}
			if (FAILED(g_client->Start())) {
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

		std::printf("DualSense: stream open: %u Hz, %u ch, buffer %u frames, gains %.2f/%.2f "
		            "(vibration/speaker)\n",
		            g_mix->nSamplesPerSec, g_mix->nChannels, g_buffer_frames,
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

	if (g_client != nullptr) {
		g_client->Stop();
		g_client->Release();
		g_client = nullptr;
	}
	if (g_render != nullptr) {
		g_render->Release();
		g_render = nullptr;
	}
	if (g_mix != nullptr) {
		CoTaskMemFree(g_mix);
		g_mix = nullptr;
	}
	g_buffer_frames = 0;
}

bool Write(const void* data, uint32_t frames, uint32_t source_rate, uint32_t source_channels,
           bool is_float, bool speaker, const float* gains) {
	std::lock_guard lock(g_mutex);
	if (g_client == nullptr || data == nullptr || frames == 0 || source_channels == 0 ||
	    g_mix == nullptr) {
		return false;
	}
	if ((g_routes & (speaker ? ROUTE_SPEAKER : ROUTE_VIBRATION)) == 0U) {
		return false;
	}

	const auto target_rate     = g_mix->nSamplesPerSec;
	const auto target_channels = g_mix->nChannels;
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
	if (FAILED(g_client->GetCurrentPadding(&padding))) {
		return false;
	}
	const auto free = g_buffer_frames > padding ? g_buffer_frames - padding : 0;
	if (free < out_frames) {
		return false; // behind: drop rather than delay the guest's audio thread
	}

	BYTE* buffer = nullptr;
	if (FAILED(g_render->GetBuffer(out_frames, &buffer)) || buffer == nullptr) {
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

	g_render->ReleaseBuffer(out_frames, 0);

	// What actually reaches the pad, so a log says whether the game fills this route at all: the
	// first writes, then one every 512 (about every 2.7 s of 5 ms calls), with their peak level.
	static uint32_t speaker_writes  = 0;
	static uint32_t vibration_writes = 0;
	auto&           writes          = speaker ? speaker_writes : vibration_writes;
	writes++;
	if (writes <= 4 || (writes % 512) == 0) {
		std::printf("DualSense: %s write %u: %u frames at %u Hz, %u ch, %s, gains %s%.2f, peak %.3f, "
		            "queue %u/%u\n",
		            speaker ? "speaker" : "vibration", writes, frames, source_rate, source_channels,
		            is_float ? "float" : "s16", gains != nullptr ? "" : "(none) ",
		            static_cast<double>(gains != nullptr ? gains[0] : 1.0f),
		            static_cast<double>(peak), padding, g_buffer_frames);
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
