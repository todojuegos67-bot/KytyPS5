#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "kernel/pthread.h"
#include "libs/audio.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Audio {

namespace {

constexpr int AUDIO_OUT_PORT_TYPE_MAIN      = 0;
constexpr int AUDIO_OUT_PORT_TYPE_BGM       = 1;
constexpr int AUDIO_OUT_PORT_TYPE_VOICE     = 2;
constexpr int AUDIO_OUT_PORT_TYPE_PERSONAL  = 3;
constexpr int AUDIO_OUT_PORT_TYPE_PADSPK    = 4;
constexpr int AUDIO_OUT_PORT_TYPE_VIBRATION = 10;
constexpr int AUDIO_OUT_PORT_TYPE_AUX       = 127;

} // namespace

namespace AudioOut2 {

LIB_NAME("AudioOut2", "AudioOut");

struct AudioOut2ContextParam {
	uint32_t max_ports;
	uint32_t max_object_ports;
	uint32_t guarantee_object_ports;
	uint32_t queue_depth;
	uint32_t num_grains;
	uint32_t flags;
	uint32_t reserved[10];
};

struct AudioOut2PortParam {
	uint16_t            port_type;
	uint16_t            pad;
	uint32_t            data_format;
	uint32_t            sampling_freq;
	uint32_t            flags;
	AudioOut2UserHandle user_handle;
	uint32_t            reserved[10];
};

struct AudioOut2Attribute {
	uint32_t    attribute_id;
	int32_t     reserved;
	const void* value;
	size_t      value_size;
};

struct AudioOut2Pcm {
	const void* data;
};

struct AudioOut2Position {
	float x;
	float y;
	float z;
};

struct AudioOut2PortState {
	uint16_t output;
	uint8_t  num_channels;
	uint8_t  pad1;
	int16_t  volume;
	uint16_t reroute_counter;
	uint32_t flags;
	uint32_t pad2;
	uint64_t reserved[6];
};

struct AudioOut2SystemState {
	float    loudness;
	uint32_t pad;
	uint64_t reserved[7];
};

struct AudioOut2SpeakerAngle {
	int16_t azimuth;
	int16_t elevation;
};

struct AudioOut2SpeakerInfo {
	uint8_t               type;
	uint8_t               pad1;
	int16_t               pad2;
	uint32_t              available_bits;
	uint32_t              flags;
	uint32_t              pad3;
	AudioOut2SpeakerAngle speaker_angle[16];
};

struct AudioOut2SystemDebugStateParam {
	uint32_t debug_state_id;
	int32_t  reserved;
	void*    param;
	size_t   param_size;
};

struct AudioOut2MasteringParamsHeader {
	uint32_t params_id;
};

struct AudioOut2MasteringStatesHeader {
	uint32_t states_id;
};

struct AudioOut2MasteringStatesDescriptor {
	uint32_t id;
	uint32_t size;
};

static constexpr uint32_t AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS = 8;
static constexpr uint32_t AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS    = 3;
static constexpr uint32_t AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2 = 4;

struct AudioOut2MasteringCompressorStates {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_rms[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS]
	                                            [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              compression_coeff[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS]
	                                                    [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringCompressorStatesV2 {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_rms[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2]
	                                            [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              compression_coeff[AUDIO_OUT2_MASTERING_COMPRESSOR_BANDS_V2]
	                                                    [AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringLimiterStates {
	AudioOut2MasteringStatesDescriptor descriptor;
	uint32_t                           reserved[2];
	float                              input_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              output_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
	float                              gain_peak[AUDIO_OUT2_MASTERING_MAX_SYSTEM_CHANNELS];
};

struct AudioOut2MasteringStates {
	AudioOut2MasteringStatesHeader     states_header;
	uint32_t                           reserved[3];
	AudioOut2MasteringCompressorStates compressor_states;
	AudioOut2MasteringLimiterStates    limiter_states;
};

struct AudioOut2MasteringStatesV2 {
	AudioOut2MasteringStatesHeader       states_header;
	uint32_t                             reserved[3];
	AudioOut2MasteringCompressorStatesV2 compressor_states;
	AudioOut2MasteringLimiterStates      limiter_states;
};

static std::atomic_uint64_t g_audioout2_next_context {1};
static std::atomic_uint64_t g_audioout2_next_port {1};
static std::atomic_uint64_t g_audioout2_next_user {1};

struct AudioOut2ContextState {
	bool                   used        = false;
	AudioOut2ContextHandle handle      = 0;
	uint32_t               queue_depth = 4;
	uint32_t               queued      = 0;
	uint32_t               num_grains  = 512;
	uint64_t               last_update = 0;
	uint64_t               drained_at  = 0; // (KYTY_AUDIO_LOG) the device time its queue ran out, until the next push
};

struct AudioOut2PortStateEntry {
	bool                   used          = false;
	AudioOut2PortHandle    handle        = 0;
	AudioOut2ContextHandle context       = 0;
	uint16_t               port_type     = 0;
	uint32_t               data_format   = 0;
	uint32_t               sampling_freq = 48000;
	uint32_t               samples_num   = 512;
	AudioInternal::Format  audio_format  = AudioInternal::Format::Unknown;
	int                    audio_handle  = 0;
	const void*            pcm_data      = nullptr;
	float                  gain          = 1.0f; // a mono (object) port's gain attribute
	uint32_t               ambisonics    = 0;    // an object port's attribute 8: 0x40 | ACN when it carries one
	// Attribute 1 as the guest sends it: one gain for each of the port's channels. Its speaker and
	// vibration ports carry their level here, and a vibration port's is two channels wide - the old
	// single-float read dropped both.
	std::array<float, 16>  gains {};
	bool                   gains_set = false;
	// A PCM attribute holds for one grain: set since the context's last push (pcm_fresh), and the pushes in a row
	// without one (pcm_stale). An object port the game stopped feeding is no longer mixed (issue #40).
	bool                   pcm_fresh = false;
	uint32_t               pcm_stale = 0;
	// (Diagnostic, issue #40.) The pushes in a row that were fed the same bytes.
	uint64_t               pcm_hash   = 0;
	uint32_t               pcm_repeat = 0;
};

struct AudioOut2SpeakerArrayState {
	bool                        used          = false;
	uint32_t                    num_speakers  = 0;
	uint8_t                     is_3d         = 0;
	uint8_t                     is_ambisonics = 0;
	AudioOut2SpeakerArrayHandle handle        = nullptr;
};

struct AudioOut2LatencyState {
	bool     used       = false;
	uint32_t user_id    = 0;
	uint32_t output     = 0;
	uint32_t latency_us = 0;
};

static Common::Mutex                              g_audioout2_context_mutex;
static std::array<AudioOut2ContextState, 16>      g_audioout2_contexts;
static Common::Mutex                              g_audioout2_port_mutex;
static std::array<AudioOut2PortStateEntry, 256>   g_audioout2_ports;
static Common::Mutex                              g_audioout2_speaker_array_mutex;
static std::array<AudioOut2SpeakerArrayState, 32> g_audioout2_speaker_arrays;
static Common::Mutex                              g_audioout2_latency_mutex;
static std::array<AudioOut2LatencyState, 16>      g_audioout2_latencies;

static constexpr int AUDIO_OUT2_ERROR_NOT_READY                   = -2144960504; /* 0x80268008 */
static constexpr int AUDIO_OUT2_ERROR_PORT_FULL                   = -2144960494; /* 0x80268012 */
static constexpr int AUDIO_OUT2_ERROR_INVALID_PARAM               = -2144960511; /* 0x80268001 */
static constexpr int AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM = -2144959999; /* 0x80268201 */
static constexpr int AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID = -2144959996; /* 0x80268204 */
static constexpr uint32_t AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM        = 0;
static constexpr uint32_t AUDIO_OUT2_PORT_ATTRIBUTE_ID_GAIN       = 1;
static constexpr uint32_t AUDIO_OUT2_PORT_ATTRIBUTE_ID_AMBISONICS = 8;
static constexpr uint32_t AUDIO_OUT2_AMBISONICS_CHANNEL           = 0x40;
static constexpr uint32_t AUDIO_OUT2_MASTERING_OUTPUT_RECORDING   = 2;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT  = 1;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_ID_V2       = 2;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_DEFAULT = 0x01010001;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_V2      = 0x01020001;
static constexpr uint32_t AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT    = 0x01010003;

static AudioOut2ContextState* audioout2_find_context_locked(AudioOut2ContextHandle ctx) {
	for (auto& state: g_audioout2_contexts) {
		if (state.used && state.handle == ctx) {
			return &state;
		}
	}

	return nullptr;
}

static uint32_t audioout2_grain_micros(uint32_t grains) {
	const auto sample_count = (grains == 0 ? 512u : grains);
	return std::max<uint32_t>((sample_count * 1000000u) / 48000u, 1000u);
}

static uint8_t audioout2_data_format_channels(uint32_t data_format) {
	const auto channels = (data_format >> 8u) & 0xffu;
	return static_cast<uint8_t>(channels == 0 ? 2u : std::min(channels, 16u));
}

static AudioInternal::Format audioout2_data_format_to_audio_format(uint32_t data_format) {
	const auto channels  = audioout2_data_format_channels(data_format);
	const auto data_type = data_format & 0x7fu;
	const auto is_std    = (data_format & 0x80u) != 0;

	switch (data_type) {
		case 0:
			switch (channels) {
				case 1: return AudioInternal::Format::FloatMono;
				case 2: return AudioInternal::Format::FloatStereo;
				case 8:
					return is_std ? AudioInternal::Format::Float8ChStd
					              : AudioInternal::Format::Float8Ch;
				case 12:
					if (!is_std) {
						return AudioInternal::Format::Float12Ch;
					}
					break;
				default: break;
			}
			break;
		case 1:
			switch (channels) {
				case 1: return AudioInternal::Format::Signed16bitMono;
				case 2: return AudioInternal::Format::Signed16bitStereo;
				case 8:
					return is_std ? AudioInternal::Format::Signed16bit8ChStd
					              : AudioInternal::Format::Signed16bit8Ch;
				default: break;
			}
			break;
		default: break;
	}

	return AudioInternal::Format::Unknown;
}

static int audioout2_port_type_to_audio_out_type(uint16_t port_type) {
	switch (port_type & 0xffu) {
		case 0: return AUDIO_OUT_PORT_TYPE_MAIN;
		case 1: return AUDIO_OUT_PORT_TYPE_BGM;
		case 2: return AUDIO_OUT_PORT_TYPE_VOICE;
		case 3: return AUDIO_OUT_PORT_TYPE_PADSPK;
		case 4: return AUDIO_OUT_PORT_TYPE_PERSONAL;
		case 5: return AUDIO_OUT_PORT_TYPE_AUX;
		case 6: return AUDIO_OUT_PORT_TYPE_VIBRATION;
		default: return AUDIO_OUT_PORT_TYPE_MAIN;
	}
}

static bool audioout2_port_type_is_object(uint16_t port_type) {
	return (port_type & 0xff00u) == 0x0100u;
}

// (Diagnostic, KYTY_AUDIO_LOG.) The pushes by mode, and the device time the game missed with the queue drained (its
// push came late): counted, summed and the longest, printed with the object grains every 2000 pushes.
static std::atomic<uint64_t> g_audioout2_pushes_blocking {0}, g_audioout2_pushes_async {0}, g_audioout2_pushes_device {0};
static std::atomic<uint64_t> g_audioout2_late_count {0}, g_audioout2_late_us {0}, g_audioout2_late_max_us {0};

static void audioout2_update_context_locked(AudioOut2ContextState* state) {
	if (state == nullptr) {
		return;
	}

	// KYTY_AUDIO_CATCHUP_MS (a test, default 0: the time dropped as before): with the queue drained, the device time a
	// late game missed stays owed for up to that long, and its next pushes are taken at once until it caught up. (The
	// suspicion it tests: dropped time leaves the host device's queue that much less sound after each late push, and
	// dropouts with a neural rendering add-on's frame spikes.) A context unpushed for longer starts again from now.
	static const uint64_t catchup_us = [] {
		const char* value = std::getenv("KYTY_AUDIO_CATCHUP_MS");
		return static_cast<uint64_t>(value != nullptr ? std::clamp(std::atoi(value), 0, 1000) : 0) * 1000u;
	}();
	const auto grain_micros = static_cast<uint64_t>(audioout2_grain_micros(state->num_grains));
	// With the queue drained: the device time since last_update the game missed is dropped beyond catchup_us
	// (last_update = now), as before with catchup_us 0.
	const auto drained_queue = [&](uint64_t now) {
		if (now - state->last_update > catchup_us) {
			state->last_update = now;
		}
	};

	const auto now = LibKernel::KernelGetProcessTime();
	if (state->last_update == 0 || now < state->last_update) {
		state->last_update = now;
		return;
	}
	if (state->queued == 0) {
		drained_queue(now);
		return;
	}

	if (now == state->last_update || grain_micros == 0) {
		return;
	}

	const auto elapsed = now - state->last_update;
	const auto drained = std::min<uint64_t>(state->queued, elapsed / grain_micros);
	if (drained == 0) {
		return;
	}

	state->queued -= static_cast<uint32_t>(drained);
	state->last_update += drained * grain_micros;
	if (state->queued == 0) {
		if (state->drained_at == 0) {
			state->drained_at = state->last_update;
		}
		drained_queue(now);
	}
}

// (KYTY_AUDIO_LOG.) A push into a drained queue: how long after the device ran out it came (more than a grain: late).
static void audioout2_note_push_locked(AudioOut2ContextState* state, uint64_t now) {
	if (state->queued != 0 || state->drained_at == 0) {
		return;
	}
	const auto late = now > state->drained_at ? now - state->drained_at : 0;
	state->drained_at = 0;
	if (late < audioout2_grain_micros(state->num_grains)) {
		return;
	}
	g_audioout2_late_count.fetch_add(1, std::memory_order_relaxed);
	g_audioout2_late_us.fetch_add(late, std::memory_order_relaxed);
	if (late > g_audioout2_late_max_us.load(std::memory_order_relaxed)) {
		g_audioout2_late_max_us.store(late, std::memory_order_relaxed);
	}
	static uint32_t count = 0;
	if (++count <= 40 || count % 100 == 0) {
		std::printf("AudioOut2: the game's push came %.1f ms after its queue ran out (#%u)\n",
		            static_cast<double>(late) / 1000.0, count);
	}
}

static AudioOut2PortStateEntry* audioout2_find_port_locked(AudioOut2PortHandle port) {
	for (auto& state: g_audioout2_ports) {
		if (state.used && state.handle == port) {
			return &state;
		}
	}

	return nullptr;
}

static bool audioout2_context_has_queueable_device(AudioOut2ContextHandle ctx) {
	Common::LockGuard lock(g_audioout2_port_mutex);
	for (const auto& state: g_audioout2_ports) {
		if (state.used && state.context == ctx && state.audio_handle > 0 &&
		    state.pcm_data != nullptr && AudioInternal::AudioOutHasDevice(state.audio_handle)) {
			return true;
		}
	}
	return false;
}

// The object ports (mono float) are for the system's 3D audio to render. Demon's Souls puts its whole
// game mix there as ambisonics: two fifth-order scenes, one channel per port (attribute 8 = 0x40 | ACN,
// SN3D, its beds silent). Decoded here for stereo from the first order (W, Y left positive, X front
// positive) by two virtual microphones facing about 74 degrees left and right: a sound straight ahead
// plays at full level on both sides, as a binaural render plays it, one behind at -3 dB (the level of
// a surround downmix), and one at the side with the power it has ahead. Cardioids facing 90 degrees
// (W and Y alone) played what is ahead, where people talk, 3 dB below what is at the sides. An object
// that is no ambisonics channel plays in the centre. Each port's gain applies, and the result goes to
// the front left and right of the context's first float main port. Adding the 36 channels up instead
// cancelled most directions.
static const void* audioout2_mix_objects_locked(AudioOut2ContextHandle ctx, const AudioOut2PortStateEntry& target) {
	constexpr float HALF_POWER = 0.70710677f;
	// Ahead W + X = 1, behind W - X = 1/sqrt(2), at the side (W + Y)^2 + (W - Y)^2 = 2.
	constexpr float W_GAIN = 0.85355339f;
	constexpr float X_GAIN = 0.14644661f;
	constexpr float Y_GAIN = 0.52100538f;
	thread_local std::vector<float> mix;
	const auto channels = audioout2_data_format_channels(target.data_format);
	const auto frames   = target.samples_num;
	bool       mixed    = false;
	for (const auto& object: g_audioout2_ports) {
		if (!object.used || object.context != ctx || !audioout2_port_type_is_object(object.port_type) ||
		    object.pcm_data == nullptr || !object.pcm_fresh || object.audio_format != AudioInternal::Format::FloatMono ||
		    object.samples_num != frames) {
			continue;
		}
		float left = HALF_POWER, right = HALF_POWER;
		if ((object.ambisonics & AUDIO_OUT2_AMBISONICS_CHANNEL) != 0) {
			const auto acn = object.ambisonics & ~AUDIO_OUT2_AMBISONICS_CHANNEL;
			if (acn == 0) {
				left = right = W_GAIN;
			} else if (acn == 1) {
				left  = Y_GAIN;
				right = -Y_GAIN;
			} else if (acn == 3) {
				left = right = X_GAIN;
			} else {
				continue;
			}
		}
		if (!mixed) {
			const auto* bed = static_cast<const float*>(target.pcm_data);
			mix.assign(bed, bed + static_cast<size_t>(frames) * channels);
			mixed = true;
		}
		const auto* in = static_cast<const float*>(object.pcm_data);
		left *= object.gain;
		right *= object.gain;
		for (uint32_t frame = 0; frame < frames; frame++) {
			mix[static_cast<size_t>(frame) * channels] += in[frame] * left;
			mix[static_cast<size_t>(frame) * channels + 1] += in[frame] * right;
		}
	}
	return mixed ? mix.data() : target.pcm_data;
}

// Before a push queues the context's grain: the ports the game gave no PCM since the last push count as unfed. Issue #40:
// with Mono Mix Mode on, Demon's Souls stops setting PCM on its 36 ambisonics object ports (it mixes into its main port
// itself) and keeps them; each push mixed their last grain again, a high tone that never ended. An object port without
// PCM for this grain is left out of the mix (on the console a grain's PCM holds for that grain only).
// (Diagnostic.) A port fed the same non-silent bytes for 100 pushes in a row (its first 256 hashed) is reported once:
// the game's own mix stopped moving.
static void audioout2_note_grain_locked(AudioOut2ContextHandle ctx) {
	static std::atomic<uint32_t> reported {0};
	// (KYTY_AUDIO_LOG: every 2000 pushes, the object grains fed and left out.)
	static const bool log    = std::getenv("KYTY_AUDIO_LOG") != nullptr;
	static uint64_t   pushes = 0, fed = 0, unfed = 0;
	if (log && ++pushes % 2000 == 0) {
		std::printf("AudioOut2: %llu pushes, object grains fed %llu, left out %llu\n", static_cast<unsigned long long>(pushes),
		            static_cast<unsigned long long>(fed), static_cast<unsigned long long>(unfed));
		std::printf("AudioOut2: at %.1f s: pushes blocking %llu (device-paced %llu), async %llu; late pushes %llu, "
		            "%.1f ms missed in all, the longest %.1f ms\n",
		            static_cast<double>(LibKernel::KernelGetProcessTime()) / 1e6,
		            static_cast<unsigned long long>(g_audioout2_pushes_blocking.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_audioout2_pushes_device.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_audioout2_pushes_async.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_audioout2_late_count.load(std::memory_order_relaxed)),
		            static_cast<double>(g_audioout2_late_us.load(std::memory_order_relaxed)) / 1000.0,
		            static_cast<double>(g_audioout2_late_max_us.load(std::memory_order_relaxed)) / 1000.0);
	}
	for (auto& state: g_audioout2_ports) {
		if (!state.used || state.context != ctx || state.pcm_data == nullptr) continue;
		state.pcm_stale = state.pcm_fresh ? 0 : state.pcm_stale + 1;
		if (audioout2_port_type_is_object(state.port_type)) (state.pcm_fresh ? fed : unfed)++;
		if (state.pcm_stale == 100 && audioout2_port_type_is_object(state.port_type) &&
		    reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("AudioOut2: object port 0x%llx (ambisonics 0x%x) not fed for 100 grains: left out of the mix\n",
			            static_cast<unsigned long long>(state.handle), state.ambisonics);
		}
		if (!state.pcm_fresh) continue;
		const auto bytes = std::min<uint64_t>(256, static_cast<uint64_t>(state.samples_num) *
		                                                audioout2_data_format_channels(state.data_format) * 2);
		const auto* data = static_cast<const uint8_t*>(state.pcm_data);
		uint64_t    hash = 0xcbf29ce484222325ull;
		uint64_t    any  = 0;
		for (uint64_t i = 0; i + 8 <= bytes; i += 8) {
			uint64_t word = 0;
			std::memcpy(&word, data + i, 8);
			any |= word;
			hash = (hash ^ word) * 0x100000001b3ull;
		}
		const bool repeat = any != 0 && hash == state.pcm_hash;
		state.pcm_hash    = hash;
		state.pcm_repeat  = repeat ? state.pcm_repeat + 1 : 0;
		if (state.pcm_repeat == 100 && reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("AudioOut2: port 0x%llx (type %u, ambisonics 0x%x) was fed the same grain 100 times in a row\n",
			            static_cast<unsigned long long>(state.handle), static_cast<uint32_t>(state.port_type),
			            state.ambisonics);
		}
	}
}

// After the push queued it: the next grain needs PCM of its own.
static void audioout2_end_grain_locked(AudioOut2ContextHandle ctx) {
	for (auto& state: g_audioout2_ports) {
		if (state.used && state.context == ctx) state.pcm_fresh = false;
	}
}

static void audioout2_queue_context_audio(AudioOut2ContextHandle ctx, bool blocking) {
	std::vector<AudioInternal::OutputParam> params;
	params.reserve(AudioInternal::OUT_PORTS_MAX);
	// The gains live in the port states, which the call below reads after the lock is dropped: keep
	// a copy per parameter (fixed size, so these addresses stay put).
	std::array<std::array<float, 16>, AudioInternal::OUT_PORTS_MAX> gain_storage {};

	g_audioout2_port_mutex.Lock();
	audioout2_note_grain_locked(ctx);
	bool objects_mixed = false;
	for (const auto& state: g_audioout2_ports) {
		if (state.used && state.context == ctx && state.audio_handle > 0 &&
		    state.pcm_data != nullptr && params.size() < AudioInternal::OUT_PORTS_MAX) {
			const void* data = state.pcm_data;
			if (!objects_mixed && (state.port_type & 0xffu) == 0 &&
			    (state.audio_format == AudioInternal::Format::FloatStereo ||
			     state.audio_format == AudioInternal::Format::Float8Ch ||
			     state.audio_format == AudioInternal::Format::Float8ChStd)) {
				data          = audioout2_mix_objects_locked(ctx, state);
				objects_mixed = true;
			}
			const float* gains = nullptr;
			if (state.gains_set) {
				gain_storage[params.size()] = state.gains;
				gains                       = gain_storage[params.size()].data();
			}
			params.push_back(AudioInternal::OutputParam {state.audio_handle, data, gains});
		}
	}
	audioout2_end_grain_locked(ctx);
	g_audioout2_port_mutex.Unlock();

	if (params.empty()) {
		return;
	}

	(void)AudioInternal::AudioOutOutputs(params.data(), static_cast<uint32_t>(params.size()),
	                                     blocking);
}

static void audioout2_close_audio_handle(int audio_handle) {
	if (audio_handle > 0) {
		AudioInternal::AudioOutClose(audio_handle);
	}
}

int KYTY_SYSV_ABI AudioOut2Initialize() {
	PRINT_NAME();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextResetParam(AudioOut2ContextParam* params) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);

	std::memset(params, 0, sizeof(AudioOut2ContextParam));
	params->max_ports              = 256;
	params->max_object_ports       = 256;
	params->guarantee_object_ports = 0;
	params->queue_depth            = 4;
	params->num_grains             = 512;
	params->flags                  = 1;

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextQueryMemory(const AudioOut2ContextParam* params,
                                              size_t*                      memory_size) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(memory_size == nullptr);

	const auto queue_depth = (params->queue_depth == 0 ? 4u : params->queue_depth);
	*memory_size           = 0x10000u + static_cast<size_t>(queue_depth) * 0x590u;

	LOGF("\t memory_size = 0x%016" PRIx64 "\n", static_cast<uint64_t>(*memory_size));

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextCreate(const AudioOut2ContextParam* params, void* buffer,
                                         size_t buffer_size, AudioOut2ContextHandle* ctx) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(ctx == nullptr);

	*ctx = g_audioout2_next_context.fetch_add(1, std::memory_order_relaxed);

	g_audioout2_context_mutex.Lock();
	auto* state = audioout2_find_context_locked(0);
	if (state == nullptr) {
		for (auto& candidate: g_audioout2_contexts) {
			if (!candidate.used) {
				state = &candidate;
				break;
			}
		}
	}
	EXIT_NOT_IMPLEMENTED(state == nullptr);
	*state             = AudioOut2ContextState {};
	state->used        = true;
	state->handle      = *ctx;
	state->queue_depth = (params->queue_depth == 0 ? 4u : params->queue_depth);
	state->queued      = 0;
	state->num_grains  = (params->num_grains == 0 ? 512u : params->num_grains);
	state->last_update = LibKernel::KernelGetProcessTime();
	g_audioout2_context_mutex.Unlock();

	LOGF("\t buffer      = 0x%016" PRIx64 "\n"
	     "\t buffer_size = 0x%016" PRIx64 "\n"
	     "\t ctx         = 0x%016" PRIx64 "\n"
	     "\t queue_depth = %" PRIu32 ", num_grains = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(buffer), static_cast<uint64_t>(buffer_size), *ctx,
	     state->queue_depth, state->num_grains);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextDestroy(AudioOut2ContextHandle ctx) {
	PRINT_NAME();
	LOGF("\t ctx = 0x%016" PRIx64 "\n", ctx);

	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		*state = AudioOut2ContextState {};
	}
	g_audioout2_context_mutex.Unlock();

	std::array<int, 256> audio_handles {};
	size_t               audio_handles_num = 0;

	g_audioout2_port_mutex.Lock();
	for (auto& port_state: g_audioout2_ports) {
		if (port_state.used && port_state.context == ctx) {
			if (port_state.audio_handle > 0 && audio_handles_num < audio_handles.size()) {
				audio_handles[audio_handles_num++] = port_state.audio_handle;
			}
			port_state = AudioOut2PortStateEntry {};
		}
	}
	g_audioout2_port_mutex.Unlock();

	for (size_t i = 0; i < audio_handles_num; i++) {
		audioout2_close_audio_handle(audio_handles[i]);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextSetAttributes(AudioOut2ContextHandle    ctx,
                                                const AudioOut2Attribute* attributes,
                                                uint32_t                  num) {
	PRINT_NAME();
	LOGF("\t ctx = 0x%016" PRIx64 ", num = %" PRIu32 "\n", ctx, num);
	EXIT_NOT_IMPLEMENTED(num != 0 && attributes == nullptr);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextAdvance(AudioOut2ContextHandle ctx) {
	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		audioout2_update_context_locked(state);
	}
	g_audioout2_context_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2ContextPush(AudioOut2ContextHandle ctx, uint32_t blocking) {
	uint32_t sleep_micros = audioout2_grain_micros(512);
	static const bool log = std::getenv("KYTY_AUDIO_LOG") != nullptr;
	if (log) {
		(blocking != 0 ? g_audioout2_pushes_blocking : g_audioout2_pushes_async).fetch_add(1, std::memory_order_relaxed);
		if (blocking != 0 && audioout2_context_has_queueable_device(ctx)) {
			g_audioout2_pushes_device.fetch_add(1, std::memory_order_relaxed);
		}
	}

	for (;;) {
		// Only a synchronous submission carrying PCM to a real device can rely on the SDL queue for
		// pacing. Async pushes must retain queue-depth backpressure, and a handle without PCM (or a
		// vibration/failed-open handle) has no downstream operation that can block this call.
		const bool use_device_clock =
		    blocking != 0 && audioout2_context_has_queueable_device(ctx);

		g_audioout2_context_mutex.Lock();
		if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
			audioout2_update_context_locked(state);
			sleep_micros = audioout2_grain_micros(state->num_grains);
			if (state->queued < state->queue_depth || use_device_clock) {
				// (With the queue drained, audioout2_update_context_locked just set last_update: now, or the owed time
				// of a late game, KYTY_AUDIO_CATCHUP_MS.)
				if (log) {
					audioout2_note_push_locked(state, LibKernel::KernelGetProcessTime());
				}
				if (state->queued < state->queue_depth) {
					state->queued++;
				}
				g_audioout2_context_mutex.Unlock();
				audioout2_queue_context_audio(ctx, blocking != 0);
				return OK;
			}
		}
		g_audioout2_context_mutex.Unlock();

		if (blocking == 0) {
			return AUDIO_OUT2_ERROR_NOT_READY;
		}

		Common::Thread::SleepMicro(sleep_micros);
	}
}

int KYTY_SYSV_ABI AudioOut2ContextGetQueueLevel(AudioOut2ContextHandle ctx, uint32_t* queue_level,
                                                uint32_t* available_queues) {
	if (queue_level != nullptr) {
		*queue_level = 0;
	}
	if (available_queues != nullptr) {
		*available_queues = 4;
	}

	g_audioout2_context_mutex.Lock();
	if (auto* state = audioout2_find_context_locked(ctx); state != nullptr) {
		audioout2_update_context_locked(state);
		if (queue_level != nullptr) {
			*queue_level = state->queued;
		}
		if (available_queues != nullptr) {
			*available_queues =
			    (state->queued < state->queue_depth ? state->queue_depth - state->queued : 0);
		}
	}
	g_audioout2_context_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortCreate(AudioOut2ContextHandle ctx, const AudioOut2PortParam* params,
                                      AudioOut2PortHandle* port) {
	EXIT_NOT_IMPLEMENTED(params == nullptr);
	EXIT_NOT_IMPLEMENTED(port == nullptr);

	const auto next_port    = g_audioout2_next_port.fetch_add(1, std::memory_order_relaxed);
	const auto audio_format = audioout2_data_format_to_audio_format(params->data_format);
	const auto audio_type   = audioout2_port_type_to_audio_out_type(params->port_type);

	g_audioout2_context_mutex.Lock();
	const auto* context_state = audioout2_find_context_locked(ctx);
	if (context_state == nullptr) {
		g_audioout2_context_mutex.Unlock();
		return AUDIO_OUT2_ERROR_INVALID_PARAM;
	}
	const auto samples_num = context_state->num_grains == 0 ? 512u : context_state->num_grains;

	g_audioout2_port_mutex.Lock();
	AudioOut2PortStateEntry* port_state = nullptr;
	for (auto& candidate: g_audioout2_ports) {
		if (!candidate.used) {
			port_state = &candidate;
			break;
		}
	}
	if (port_state != nullptr) {
		*port_state               = AudioOut2PortStateEntry {};
		port_state->used          = true;
		port_state->handle        = next_port;
		port_state->context       = ctx;
		port_state->port_type     = params->port_type;
		port_state->data_format   = params->data_format;
		port_state->sampling_freq = params->sampling_freq;
		port_state->samples_num   = samples_num;
		port_state->audio_format  = audio_format;
	}
	g_audioout2_port_mutex.Unlock();
	g_audioout2_context_mutex.Unlock();

	if (port_state == nullptr) {
		return AUDIO_OUT2_ERROR_PORT_FULL;
	}

	int audio_handle = 0;

	if (audio_format != AudioInternal::Format::Unknown &&
	    !audioout2_port_type_is_object(params->port_type)) {
		audio_handle = AudioInternal::AudioOutOpen(audio_type, samples_num, params->sampling_freq,
		                                           audio_format);
	}

	g_audioout2_port_mutex.Lock();
	const bool reserved = port_state->used && port_state->handle == next_port;
	if (reserved) {
		port_state->audio_handle = audio_handle;
	}
	g_audioout2_port_mutex.Unlock();
	if (!reserved) {
		audioout2_close_audio_handle(audio_handle);
		return AUDIO_OUT2_ERROR_INVALID_PARAM;
	}

	*port = next_port;

	if (next_port <= 16 || (next_port % 600) == 0) {
		PRINT_NAME();
		LOGF("\t ctx           = 0x%016" PRIx64 "\n"
		     "\t port          = 0x%016" PRIx64 "\n"
		     "\t port_type     = %" PRIu16 "\n"
		     "\t data_format   = 0x%08" PRIx32 "\n"
		     "\t sampling_freq = %" PRIu32 "\n",
		     ctx, *port, params->port_type, params->data_format, params->sampling_freq);
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortDestroy(AudioOut2PortHandle port) {
	PRINT_NAME();
	LOGF("\t port = 0x%016" PRIx64 "\n", port);

	int audio_handle = 0;
	g_audioout2_port_mutex.Lock();
	if (auto* state = audioout2_find_port_locked(port); state != nullptr) {
		audio_handle = state->audio_handle;
		*state       = AudioOut2PortStateEntry {};
	}
	g_audioout2_port_mutex.Unlock();

	audioout2_close_audio_handle(audio_handle);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortSetAttributes(AudioOut2PortHandle       port,
                                             const AudioOut2Attribute* attributes, uint32_t num) {
	EXIT_NOT_IMPLEMENTED(num != 0 && attributes == nullptr);

	const void* pcm_data       = nullptr;
	bool        has_pcm        = false;
	float       gain           = 1.0f;
	bool        has_gain       = false;
	std::array<float, 16> gains {};
	uint32_t    gains_count    = 0;
	bool        has_gains      = false;
	uint32_t    ambisonics     = 0;
	bool        has_ambisonics = false;
	for (uint32_t i = 0; i < num; i++) {
		if (attributes[i].attribute_id == AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM &&
		    attributes[i].value != nullptr && attributes[i].value_size >= sizeof(AudioOut2Pcm)) {
			AudioOut2Pcm pcm {};
			std::memcpy(&pcm, attributes[i].value, sizeof(AudioOut2Pcm));
			pcm_data = pcm.data;
			has_pcm  = true;
		}
		// Attribute 1: one gain for each of the port's channels (a mono port's is one float). Only a
		// finite, non-negative set is taken - a NaN fails both comparisons - and anything else keeps
		// the last gains, as a DualSense does.
		if (attributes[i].attribute_id == AUDIO_OUT2_PORT_ATTRIBUTE_ID_GAIN &&
		    attributes[i].value != nullptr && attributes[i].value_size >= sizeof(float) &&
		    attributes[i].value_size <= gains.size() * sizeof(float) &&
		    (attributes[i].value_size % sizeof(float)) == 0) {
			const auto count = static_cast<uint32_t>(attributes[i].value_size / sizeof(float));
			std::array<float, 16> candidate {};
			std::memcpy(candidate.data(), attributes[i].value, attributes[i].value_size);
			const bool valid = std::all_of(candidate.begin(), candidate.begin() + count,
			                               [](float value) { return value >= 0.0f && value <= 64.0f; });
			if (valid) {
				gains       = candidate;
				gains_count = count;
				has_gains   = true;
				gain        = candidate[0];
				has_gain    = true;
			}
		}
		if (attributes[i].attribute_id == AUDIO_OUT2_PORT_ATTRIBUTE_ID_AMBISONICS && attributes[i].value != nullptr &&
		    attributes[i].value_size == sizeof(uint32_t)) {
			std::memcpy(&ambisonics, attributes[i].value, sizeof(uint32_t));
			has_ambisonics = true;
		}
	}

	if (has_pcm || has_gain || has_gains || has_ambisonics) {
		g_audioout2_port_mutex.Lock();
		if (auto* state = audioout2_find_port_locked(port); state != nullptr) {
			if (has_pcm) {
				state->pcm_data  = pcm_data;
				state->pcm_fresh = true;
			}
			if (has_gain) state->gain = gain;
			if (has_gains) {
				state->gains     = gains;
				state->gains_set = true;
			}
			if (has_ambisonics) state->ambisonics = ambisonics;
		}
		g_audioout2_port_mutex.Unlock();
	}

	return OK;
}

int KYTY_SYSV_ABI AudioOut2PortGetState(AudioOut2PortHandle port, AudioOut2PortState* state) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(state == nullptr);

	std::memset(state, 0, sizeof(AudioOut2PortState));
	state->output          = 1;
	state->num_channels    = 2;
	state->volume          = 127;
	state->reroute_counter = 0;
	state->flags           = 0;

	g_audioout2_port_mutex.Lock();
	if (auto* port_state = audioout2_find_port_locked(port); port_state != nullptr) {
		state->num_channels = audioout2_data_format_channels(port_state->data_format);
	}
	g_audioout2_port_mutex.Unlock();

	// LOGF("\t port = 0x%016" PRIx64 "\n", port);
	// LOGF("\t num_channels = %" PRIu8 "\n", state->num_channels);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSystemState(AudioOut2SystemState* state) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(state == nullptr);
	std::memset(state, 0, sizeof(AudioOut2SystemState));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserCreate(uint32_t user_id, AudioOut2UserHandle* handle) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(handle == nullptr);
	*handle = static_cast<AudioOut2UserHandle>(
	    g_audioout2_next_user.fetch_add(1, std::memory_order_relaxed));
	LOGF("\t user_id = %" PRIu32 ", handle = 0x%016" PRIx64 "\n", user_id,
	     static_cast<uint64_t>(*handle));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserDestroy(AudioOut2UserHandle handle) {
	PRINT_NAME();
	LOGF("\t handle = 0x%016" PRIx64 "\n", static_cast<uint64_t>(handle));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2UserGetSupportedAttributes(AudioOut2UserHandle handle,
                                                      uint32_t* context_attributes,
                                                      uint32_t* port_attributes) {
	PRINT_NAME();

	if (context_attributes == nullptr || port_attributes == nullptr) {
		return AUDIO_OUT2_ERROR_INVALID_PARAM;
	}

	// The context attributes this branch answers, and the port attributes it understands: the
	// per-channel gain (attribute 1) and the PCM buffer (attribute 0). The console works the same
	// way, and the guest asks this before it relies on either.
	*context_attributes = 0;
	*port_attributes    = (1u << AUDIO_OUT2_PORT_ATTRIBUTE_ID_PCM) |
	                   (1u << AUDIO_OUT2_PORT_ATTRIBUTE_ID_GAIN);

	// printf, not LOGF: the console's LOGF is silent in the release configuration.
	std::printf("AudioOut2: user 0x%016" PRIx64
	            " supported attributes: context 0x%x, port 0x%x (PCM and gain)\n",
	            static_cast<uint64_t>(handle), *context_attributes, *port_attributes);
	return OK;
}

size_t KYTY_SYSV_ABI AudioOut2GetSpeakerArrayMemorySize(uint32_t num_speakers, uint8_t is_3d,
                                                        uint8_t is_ambisonics) {
	PRINT_NAME();

	const auto speakers = std::clamp<uint32_t>(num_speakers, 1, 32);
	const auto size = static_cast<size_t>(0x400 + speakers * (is_ambisonics != 0 ? 0x100 : 0x40) +
	                                      (is_3d != 0 ? 0x200 : 0));

	LOGF("\t num_speakers  = %" PRIu32 "\n"
	     "\t is_3d         = %" PRIu8 "\n"
	     "\t is_ambisonics = %" PRIu8 "\n"
	     "\t memory_size   = 0x%016" PRIx64 "\n",
	     num_speakers, is_3d, is_ambisonics, static_cast<uint64_t>(size));

	return size;
}

int KYTY_SYSV_ABI AudioOut2SpeakerArrayCreate(AudioOut2SpeakerArrayHandle* handle,
                                              const void* vbap_params, const void* ambi_params) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(handle == nullptr);

	*handle = nullptr;

	g_audioout2_speaker_array_mutex.Lock();
	for (auto& state: g_audioout2_speaker_arrays) {
		if (!state.used) {
			state              = AudioOut2SpeakerArrayState {};
			state.used         = true;
			state.handle       = &state;
			state.num_speakers = 2;
			*handle            = state.handle;
			break;
		}
	}
	g_audioout2_speaker_array_mutex.Unlock();

	LOGF("\t handle      = 0x%016" PRIx64 "\n"
	     "\t vbap_params = 0x%016" PRIx64 "\n"
	     "\t ambi_params = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(*handle), reinterpret_cast<uint64_t>(vbap_params),
	     reinterpret_cast<uint64_t>(ambi_params));

	return (*handle != nullptr ? OK : AUDIO_OUT2_ERROR_PORT_FULL);
}

int KYTY_SYSV_ABI AudioOut2SpeakerArrayDestroy(AudioOut2SpeakerArrayHandle handle) {
	PRINT_NAME();
	LOGF("\t handle = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(handle));

	g_audioout2_speaker_array_mutex.Lock();
	for (auto& state: g_audioout2_speaker_arrays) {
		if (state.used && state.handle == handle) {
			state = AudioOut2SpeakerArrayState {};
			break;
		}
	}
	g_audioout2_speaker_array_mutex.Unlock();

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerArrayCoefficients(
    AudioOut2SpeakerArrayHandle handle, AudioOut2Position pos, float spread, float* coefficients,
    uint32_t num_coefficients, uint8_t height_aware, float downmix_spread_radius) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(coefficients == nullptr && num_coefficients != 0);

	if (coefficients != nullptr) {
		std::fill(coefficients, coefficients + num_coefficients, 0.0f);
		if (num_coefficients > 0) {
			coefficients[0] = 1.0f;
		}
		if (num_coefficients > 1) {
			coefficients[1] = 1.0f;
		}
	}

	LOGF("\t handle = 0x%016" PRIx64 ", coeffs = %" PRIu32
	     ", pos = (%f, %f, %f), spread = %f, height = %" PRIu8 ", downmix = %f\n",
	     reinterpret_cast<uint64_t>(handle), num_coefficients, static_cast<double>(pos.x),
	     static_cast<double>(pos.y), static_cast<double>(pos.z), static_cast<double>(spread),
	     height_aware, static_cast<double>(downmix_spread_radius));

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerArrayAmbisonicsCoefficients(AudioOut2SpeakerArrayHandle handle,
                                                                 uint32_t ambisonics_channel,
                                                                 float*   coefficients,
                                                                 uint32_t num_coefficients) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(coefficients == nullptr && num_coefficients != 0);

	if (coefficients != nullptr) {
		std::fill(coefficients, coefficients + num_coefficients, 0.0f);
		if (num_coefficients > 0) {
			coefficients[0] =
			    (ambisonics_channel == 0 || ambisonics_channel == 64 ? 0.70710677f : 1.0f);
		}
	}

	LOGF("\t handle = 0x%016" PRIx64 ", channel = %" PRIu32 ", coeffs = %" PRIu32 "\n",
	     reinterpret_cast<uint64_t>(handle), ambisonics_channel, num_coefficients);

	return OK;
}

int KYTY_SYSV_ABI AudioOut2GetSpeakerInfo(AudioOut2SpeakerInfo* info, uint32_t flags) {
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	std::memset(info, 0, sizeof(AudioOut2SpeakerInfo));
	info->type             = 0;
	info->available_bits   = 0x03;
	info->flags            = 0;
	info->speaker_angle[0] = {-30, 0};
	info->speaker_angle[1] = {30, 0};

	return OK;
}

int KYTY_SYSV_ABI AudioOut2SetSystemDebugState(const AudioOut2SystemDebugStateParam* param) {
	PRINT_NAME();
	LOGF("\t param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(param));
	return OK;
}

int KYTY_SYSV_ABI AudioOut2Set3DLatency(uint32_t user_id, uint32_t output, uint32_t latency_us) {
	// Not sure

	PRINT_NAME();
	LOGF("\t user_id = %" PRIu32 ", output = %" PRIu32 ", latency_us = %" PRIu32 "\n", user_id,
	     output, latency_us);

	if (output > AUDIO_OUT2_MASTERING_OUTPUT_RECORDING) {
		return AUDIO_OUT2_ERROR_INVALID_PARAM;
	}

	g_audioout2_latency_mutex.Lock();
	AudioOut2LatencyState* free_state = nullptr;
	for (auto& state: g_audioout2_latencies) {
		if (state.used && state.user_id == user_id && state.output == output) {
			state.latency_us = latency_us;
			g_audioout2_latency_mutex.Unlock();
			return OK;
		}
		if (!state.used && free_state == nullptr) {
			free_state = &state;
		}
	}

	if (free_state != nullptr) {
		*free_state = AudioOut2LatencyState {true, user_id, output, latency_us};
	}
	g_audioout2_latency_mutex.Unlock();

	return (free_state != nullptr ? OK : AUDIO_OUT2_ERROR_PORT_FULL);
}

int KYTY_SYSV_ABI AudioOut2MasteringInit(uint32_t flags) {
	PRINT_NAME();
	LOGF("\t flags = 0x%08" PRIx32 "\n", flags);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2MasteringSetParam(const AudioOut2MasteringParamsHeader* param,
                                             uint32_t output, uint32_t flags) {
	PRINT_NAME();
	LOGF("\t param = 0x%016" PRIx64 ", output = %" PRIu32 ", flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(param), output, flags);
	return OK;
}

int KYTY_SYSV_ABI AudioOut2MasteringGetState(AudioOut2MasteringStatesHeader* state, uint32_t output,
                                             AudioOut2UserHandle user) {
	PRINT_NAME();
	LOGF("\t state = 0x%016" PRIx64 ", output = %" PRIu32 ", user = 0x%016" PRIx64 "\n",
	     reinterpret_cast<uint64_t>(state), output, static_cast<uint64_t>(user));

	if (state == nullptr) {
		return AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM;
	}

	const auto states_id = state->states_id;
	switch (states_id) {
		case AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT: {
			auto* full_state = reinterpret_cast<AudioOut2MasteringStates*>(state);
			std::memset(full_state, 0, sizeof(AudioOut2MasteringStates));
			full_state->states_header.states_id = AUDIO_OUT2_MASTERING_STATES_ID_DEFAULT;
			full_state->compressor_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_DEFAULT;
			full_state->compressor_states.descriptor.size =
			    sizeof(AudioOut2MasteringCompressorStates);
			full_state->limiter_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT;
			full_state->limiter_states.descriptor.size = sizeof(AudioOut2MasteringLimiterStates);
			return OK;
		}
		case AUDIO_OUT2_MASTERING_STATES_ID_V2: {
			auto* full_state = reinterpret_cast<AudioOut2MasteringStatesV2*>(state);
			std::memset(full_state, 0, sizeof(AudioOut2MasteringStatesV2));
			full_state->states_header.states_id = AUDIO_OUT2_MASTERING_STATES_ID_V2;
			full_state->compressor_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_COMPRESSOR_V2;
			full_state->compressor_states.descriptor.size =
			    sizeof(AudioOut2MasteringCompressorStatesV2);
			full_state->limiter_states.descriptor.id =
			    AUDIO_OUT2_MASTERING_STATES_STRUCT_ID_LIMITER_DEFAULT;
			full_state->limiter_states.descriptor.size = sizeof(AudioOut2MasteringLimiterStates);
			return OK;
		}
		default: return AUDIO_OUT2_ERROR_MASTERING_INVALID_STATES_ID;
	}
}

int KYTY_SYSV_ABI AudioOut2MasteringTerm() {
	PRINT_NAME();
	return OK;
}

} // namespace AudioOut2
} // namespace Libs::Audio
