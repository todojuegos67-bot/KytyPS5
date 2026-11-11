#include "libs/dualSenseBluetooth.h"

#include "SDL.h"
#include "SDL_hidapi.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <opus.h>
#include <string>
#include <vector>

namespace Libs::Controller::DualSenseBluetooth {

namespace {
// Report 0x39 and its sub-packets are described by awalol/DS5Dongle (MIT):
// https://github.com/awalol/DS5Dongle
// Each report plays two 32-frame stereo haptic blocks at 3 kHz (21.333 ms).
constexpr uint64_t PERIOD_NS            = 64000000000ULL / 3000;
constexpr int      HAPTIC_RATE          = 3000;
constexpr int      HAPTIC_REPORT_FLOATS = 2 * 32 * 2;
// Two 480-frame Opus packets occupy the same 21.333 ms slot. Resampling 48 kHz input to 45 kHz
// before encoding at 48 kHz compensates for playback.
constexpr int      SPEAKER_RATE = 45000;
constexpr int      OPUS_FRAMES  = 480;
constexpr int      OPUS_BYTES   = 200;
constexpr size_t   REPORT_BYTES = 547;
// 40 ms: the same report carries the actuators, so a shorter queue trades a little tolerance for a
// noticeably smaller delay before a hit is felt.
constexpr int      MAX_QUEUE_MS = 40;
constexpr uint64_t COALESCE_NS  = 1000000;

struct Session;
std::mutex                              g_mutex;
std::condition_variable                 g_wake;
std::map<int, std::shared_ptr<Session>> g_sessions;
SDL_Thread*                             g_thread   = nullptr;
bool                                    g_stopping = false;
void (*g_tick)()                                  = nullptr;

template <typename Char>
std::string Address(const Char* serial) {
	std::string address;
	if (serial == nullptr) {
		return address;
	}
	for (; *serial != 0; ++serial) {
		if (*serial >= '0' && *serial <= '9') {
			address.push_back(static_cast<char>(*serial));
		} else if (*serial >= 'a' && *serial <= 'f') {
			address.push_back(static_cast<char>(*serial));
		} else if (*serial >= 'A' && *serial <= 'F') {
			address.push_back(static_cast<char>(*serial - 'A' + 'a'));
		} else if (*serial != ':' && *serial != '-') {
			return {};
		}
	}
	return address.size() == 12 ? address : std::string {};
}

// SDL2's hidapi says nothing about the bus a device sits on (bus_type is an SDL3 field). On Windows a
// Bluetooth HID device's path carries the HID-over-Bluetooth service's GUID, and a wired one's does
// not, so the path is what tells them apart here.
static bool IsBluetoothPath(const char* path) {
	if (path == nullptr) {
		return false;
	}
	std::string lower;
	for (const char* c = path; *c != 0; ++c) {
		lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*c))));
	}
	return lower.find("00001124-0000-1000-8000-00805f9b34fb") != std::string::npos;
}

// The DualSense on the Bluetooth bus whose serial is the joystick's own. A wired one has no
// Bluetooth-bus HID device, so this answers nullptr for it and the USB audio path stays in use.
SDL_hid_device* OpenDevice(int controller) {
	std::string serial;
	if (auto* joystick = SDL_JoystickFromInstanceID(static_cast<SDL_JoystickID>(controller))) {
		serial = Address(SDL_JoystickGetSerial(joystick));
	}
	for (const uint16_t product: {static_cast<uint16_t>(0x0ce6), static_cast<uint16_t>(0x0df2)}) {
		auto*       devices = SDL_hid_enumerate(0x054c, product);
		std::string path;
		bool        ambiguous = false;
		for (auto* device = devices; device != nullptr; device = device->next) {
			if (!IsBluetoothPath(device->path)) {
				continue;
			}
			if (!serial.empty() && Address(device->serial_number) != serial) {
				continue;
			}
			if (device->usage_page != 0 && !(device->usage_page == 1 && device->usage == 5)) {
				continue;
			}
			if (!path.empty()) {
				ambiguous = true;
				break;
			}
			path = device->path;
		}
		SDL_hid_free_enumeration(devices);
		if (!ambiguous && !path.empty()) {
			return SDL_hid_open_path(path.c_str(), 0);
		}
	}
	return nullptr;
}

void FinishReport(std::array<uint8_t, REPORT_BYTES>& report) {
	const uint8_t prefix = 0xa2;
	auto          crc    = SDL_crc32(0, &prefix, 1);
	crc                  = SDL_crc32(crc, report.data(), report.size() - 4);
	for (size_t i = 0; i < 4; i++) {
		report[report.size() - 4 + i] = static_cast<uint8_t>(crc >> (8 * i));
	}
}

uint64_t NowNs() {
	return SDL_GetTicks64() * 1000000ULL;
}

struct Session {
	SDL_hid_device*       hid     = nullptr;
	OpusEncoder*          encoder = nullptr;
	std::vector<Stream*>  streams;
	std::mutex            io_mutex;
	std::atomic<bool>     active {true};
	std::atomic<bool>     failed {false};
	std::atomic<uint64_t> generation {0};
	uint64_t              next_send        = 0;
	uint64_t              pending_deadline = 0;
	uint8_t               sequence         = 0;
	uint8_t               counter          = 0;

	~Session() {
		if (encoder != nullptr) {
			opus_encoder_destroy(encoder);
		}
		if (hid != nullptr) {
			SDL_hid_close(hid);
		}
	}

	bool CreateEncoder() {
		if (encoder != nullptr) {
			return true;
		}
		int error = OPUS_OK;
		encoder   = opus_encoder_create(48000, 2, OPUS_APPLICATION_AUDIO, &error);
		if (encoder == nullptr || error != OPUS_OK) {
			return false;
		}
		if (opus_encoder_ctl(encoder, OPUS_SET_BITRATE(OPUS_BYTES * 8 * 100)) == OPUS_OK &&
		    opus_encoder_ctl(encoder, OPUS_SET_VBR(0)) == OPUS_OK &&
		    opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(0)) == OPUS_OK) {
			return true;
		}
		opus_encoder_destroy(encoder);
		encoder = nullptr;
		return false;
	}
};
} // namespace

struct Stream {
	std::shared_ptr<Session> session;
	SDL_AudioStream*         audio   = nullptr;
	uint32_t                 freq    = 0;
	bool                     speaker = false;
};

namespace {
// The large buffers of the sender and of Queue live on the heap, not the stack. With LTO, inlining SDL's
// functions (built with /Gs1048576) gives their callers SDL's stack probe size, so a frame of more than a
// page skipped the thread's guard page: the sender thread crashed at its first call (0xC0000005).
template <size_t N>
void Mix(SDL_AudioStream* stream, std::array<float, N>& mixed) {
	static thread_local std::vector<float> pcm;
	pcm.assign(N, 0.0f);
	const int bytes = SDL_AudioStreamGet(stream, pcm.data(), static_cast<int>(N * sizeof(float)));
	for (int i = 0; i < std::max(0, bytes) / static_cast<int>(sizeof(float)); i++) {
		mixed[i] += pcm[i];
	}
}

struct PendingReport {
	std::shared_ptr<Session>               session;
	uint64_t                               generation;
	std::array<uint8_t, REPORT_BYTES>      data {};
	std::array<float, 2 * OPUS_FRAMES * 2> audio {};
	bool                                   speaker = false;
};

bool PrepareReport(Session& session, PendingReport& pending, uint64_t now) {
	bool has_data      = false;
	bool ready         = false;
	bool speaker_ready = false;
	for (const auto* stream: session.streams) {
		const bool available = SDL_AudioStreamAvailable(stream->audio) > 0;
		// One report carries HAPTIC_REPORT_FLOATS floats, or two Opus packets for the speaker. SDL2's
		// SDL_AudioStreamAvailable answers the converted (output) side where SDL3's
		// SDL_GetAudioStreamQueued answers the input side, so a full block is measured in output
		// frames here: measuring it in input frames made a complete block look partial, and every
		// report then waited for the deadline path before it was sent.
		const int needed = (stream->speaker ? 2 * OPUS_FRAMES : HAPTIC_REPORT_FLOATS / 2) * 2 *
		                   static_cast<int>(sizeof(float));
		const bool full = available && SDL_AudioStreamAvailable(stream->audio) >= needed;
		has_data |= available;
		ready |= full;
		speaker_ready |= stream->speaker && full;
		pending.speaker |= stream->speaker;
	}
	if (!has_data) {
		session.pending_deadline = 0;
		return false;
	}
	if (!ready || (!speaker_ready && (pending.speaker || session.next_send == 0))) {
		// Accumulate short blocks for at most one report period. A full stream must not wait on a
		// partial peer; only briefly coalesce haptics with speaker audio in the same emulated batch.
		const auto deadline = now + (ready ? COALESCE_NS : PERIOD_NS);
		session.pending_deadline =
		    session.pending_deadline == 0 ? deadline : std::min(session.pending_deadline, deadline);
		if (now < session.pending_deadline) {
			return false;
		}
	}
	session.pending_deadline = 0;
	std::array<float, HAPTIC_REPORT_FLOATS> haptics {};
	for (auto* stream: session.streams) {
		if (stream->speaker) {
			Mix(stream->audio, pending.audio);
		} else {
			Mix(stream->audio, haptics);
		}
	}
	auto& report     = pending.data;
	report[0]        = 0x39;
	report[1]        = static_cast<uint8_t>(session.sequence << 4);
	session.sequence = (session.sequence + 1) & 0x0f;
	report[2]        = 0x91;
	report[3]        = 6;
	report[4]        = 0x7e;
	std::fill_n(report.begin() + 5, 4, 48);
	session.counter += 2;
	report[9]  = session.counter;
	report[10] = 0xd2;
	report[11] = 64;
	for (size_t i = 0; i < haptics.size(); i++) {
		const auto value = std::isfinite(haptics[i]) ? haptics[i] : 0.0f;
		report[12 + i]   = static_cast<uint8_t>(
            static_cast<int>(std::lround(std::clamp(value, -1.0f, 1.0f) * 127)));
	}
	if (pending.speaker) {
		report[140] = 0xd3;
		report[141] = OPUS_BYTES;
	}
	return true;
}

bool EncodeReport(PendingReport& pending) {
	if (!pending.session->active || pending.session->failed) {
		return false;
	}
	if (pending.speaker) {
		for (auto& sample: pending.audio) {
			sample = std::isfinite(sample) ? std::clamp(sample, -1.0f, 1.0f) : 0.0f;
		}
		for (int packet = 0; packet < 2; packet++) {
			if (opus_encode_float(pending.session->encoder,
			                      pending.audio.data() + packet * OPUS_FRAMES * 2, OPUS_FRAMES,
			                      pending.data.data() + 142 + packet * OPUS_BYTES,
			                      OPUS_BYTES) != OPUS_BYTES) {
				pending.session->failed = true;
				return false;
			}
		}
	}
	FinishReport(pending.data);
	return true;
}

std::vector<PendingReport> Prepare(uint64_t now) {
	std::vector<PendingReport> pending;
	{
		std::lock_guard lock(g_mutex);
		for (auto& [id, session]: g_sessions) {
			if (session->failed || now < session->next_send) {
				continue;
			}
			auto& report      = pending.emplace_back();
			report.session    = session;
			report.generation = session->generation.load();
			if (!PrepareReport(*session, report, now)) {
				pending.pop_back();
			} else {
				// Regain timing after a late HID write without sending reports in a burst. Reset the
				// cadence only on the first report.
				session->next_send = session->next_send == 0
				                         ? now + PERIOD_NS
				                         : std::max(session->next_send + PERIOD_NS,
				                                    now + PERIOD_NS / 2);
			}
		}
	}
	// Opus encoding can take milliseconds; the emulation audio thread must be able to queue its
	// next block while this work runs.
	pending.erase(std::remove_if(pending.begin(), pending.end(),
	                             [](PendingReport& report) { return !EncodeReport(report); }),
	              pending.end());
	return pending;
}

void Send(const std::vector<PendingReport>& pending) {
	for (const auto& report: pending) {
		auto&           session = *report.session;
		std::lock_guard io_lock(session.io_mutex);
		if (!session.active || session.failed || report.generation != session.generation) {
			continue;
		}
		if (SDL_hid_write(session.hid, report.data.data(), report.data.size()) !=
		    static_cast<int>(report.data.size())) {
			session.failed = true;
			std::printf("DualSenseBluetooth: audio HID write failed: %s\n", SDL_GetError());
		}
	}
}

int SDLCALL Sender(void*) {
	std::unique_lock lock(g_mutex);
	while (!g_stopping) {
		auto tick = g_tick;
		lock.unlock();
		auto pending = Prepare(NowNs());
		Send(pending);
		if (tick != nullptr) {
			tick();
		}
		lock.lock();
		if (g_sessions.empty()) {
			g_wake.wait(lock, [] { return g_stopping || !g_sessions.empty(); });
		} else {
			const auto now     = NowNs();
			uint64_t   wait_ns = 10000000; // Check rumble expiry at most 10 ms late.
			for (const auto& [id, session]: g_sessions) {
				if (session->next_send > now) {
					wait_ns = std::min(wait_ns, session->next_send - now);
				}
				if (session->pending_deadline > now) {
					wait_ns = std::min(wait_ns, session->pending_deadline - now);
				}
			}
			g_wake.wait_for(lock, std::chrono::nanoseconds(wait_ns));
		}
	}
	return 0;
}
} // namespace

Stream* Open(uint32_t freq, bool speaker, int controller, void (*tick)()) {
	if (freq == 0 || freq > 192000) {
		return nullptr;
	}
	std::lock_guard lock(g_mutex);
	if (g_stopping) {
		return nullptr;
	}
	auto                     it = g_sessions.find(controller);
	std::shared_ptr<Session> session;
	if (it != g_sessions.end() && !it->second->failed) {
		session = it->second;
	} else {
		session      = std::make_shared<Session>();
		session->hid = OpenDevice(controller);
		if (session->hid == nullptr) {
			return nullptr;
		}
	}
	if (speaker && !session->CreateEncoder()) {
		return nullptr;
	}
	auto* audio = SDL_NewAudioStream(AUDIO_F32SYS, 2, static_cast<int>(freq), AUDIO_F32SYS, 2,
	                                 speaker ? SPEAKER_RATE : HAPTIC_RATE);
	if (audio == nullptr) {
		return nullptr;
	}
	if (g_thread == nullptr) {
		g_thread = SDL_CreateThread(Sender, "DualSense Bluetooth audio", nullptr);
		if (g_thread == nullptr) {
			SDL_FreeAudioStream(audio);
			return nullptr;
		}
	}
	g_tick       = tick;
	auto* stream = new Stream {session, audio, freq, speaker};
	session->streams.push_back(stream);
	session->pending_deadline = 0;
	g_sessions[controller]    = session;
	g_wake.notify_one();
	return stream;
}

uint64_t Queue(Stream* stream, const float* stereo, uint32_t frames) {
	if (stream == nullptr || stereo == nullptr || frames == 0) {
		return 0;
	}
	std::lock_guard lock(g_mutex);
	if (!stream->session->active || stream->session->failed) {
		return 0;
	}
	const uint32_t max_frames = stream->freq * MAX_QUEUE_MS / 1000;
	if (frames > max_frames) {
		stereo += static_cast<size_t>(frames - max_frames) * 2;
		frames = max_frames;
	}
	// The budget is a duration, so it is counted in output frames: that is the side SDL2 answers.
	const auto out_rate  = stream->speaker ? SPEAKER_RATE : HAPTIC_RATE;
	const int  max_out   = static_cast<int>(static_cast<uint64_t>(out_rate) * MAX_QUEUE_MS / 1000) * 2 *
	                       static_cast<int>(sizeof(float));
	const int  converted = static_cast<int>(static_cast<uint64_t>(frames) * out_rate / stream->freq) * 2 *
	                       static_cast<int>(sizeof(float));
	int       queued = std::max(0, SDL_AudioStreamAvailable(stream->audio));
	const int bytes  = static_cast<int>(frames * 2 * sizeof(float));
	if (queued + converted > max_out) {
		// Discard complete old report slots, preserving the newest queued audio. Clearing the whole
		// stream here creates an audible gap after a brief Windows Bluetooth write stall.
		std::vector<float> discarded(2 * OPUS_FRAMES * 2);
		const int discard_bytes = stream->speaker
		                              ? static_cast<int>(discarded.size() * sizeof(float))
		                              : static_cast<int>(HAPTIC_REPORT_FLOATS * sizeof(float));
		const int target        = max_out - converted;
		while (queued > target) {
			if (SDL_AudioStreamGet(stream->audio, discarded.data(), discard_bytes) <= 0) {
				SDL_AudioStreamClear(stream->audio);
				break;
			}
			queued = std::max(0, SDL_AudioStreamAvailable(stream->audio));
		}
	}
	queued = std::max(0, SDL_AudioStreamAvailable(stream->audio));
	if (SDL_AudioStreamPut(stream->audio, stereo, bytes) != 0) {
		return 0;
	}
	g_wake.notify_one();
	return static_cast<uint64_t>(queued + bytes) * 1000000 /
	       (static_cast<uint64_t>(stream->freq) * 2 * sizeof(float));
}

void Close(Stream* stream) {
	if (stream == nullptr) {
		return;
	}
	auto session = stream->session;
	{
		std::lock_guard lock(g_mutex);
		auto&           streams = session->streams;
		streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
		session->generation++;
		session->pending_deadline = 0;
		if (streams.empty()) {
			session->active = false;
			for (auto it = g_sessions.begin(); it != g_sessions.end(); ++it) {
				if (it->second == session) {
					g_sessions.erase(it);
					break;
				}
			}
		}
		SDL_FreeAudioStream(stream->audio);
		delete stream;
	}
	// Drain a write already in flight before the caller changes speaker routing.
	std::lock_guard io_lock(session->io_mutex);
}

void Shutdown() {
	SDL_Thread* thread;
	{
		std::lock_guard lock(g_mutex);
		g_stopping = true;
		for (auto& [id, session]: g_sessions) {
			session->active = false;
		}
		thread = g_thread;
		g_wake.notify_one();
	}
	if (thread != nullptr) {
		SDL_WaitThread(thread, nullptr);
	}
	std::lock_guard lock(g_mutex);
	g_sessions.clear();
	g_thread   = nullptr;
	g_tick     = nullptr;
	g_stopping = false;
}

} // namespace Libs::Controller::DualSenseBluetooth
