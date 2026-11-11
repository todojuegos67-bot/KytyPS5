#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "libs/ajm/aac_decoder.h"
#include "libs/ajm/atrac9_decoder.h"
#include "libs/ajm/decoder.h"
#include "libs/ajm/mp3_decoder.h"
#include "libs/audio.h"
#include "libs/audio_internal.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Audio::Ajm {

LIB_NAME("Ajm", "Ajm");

struct AjmBatchInfo {
	void*       buffer;
	size_t      offset;
	size_t      size;
	void*       last_good_job;
	const void* last_good_job_ra;
};

struct AjmBatchError {
	int         error_code;
	const void* job_addr;
	uint32_t    cmd_offset;
	const void* job_ra;
};

struct AjmBuffer {
	void*  address;
	size_t size;
};

struct AjmSidebandResult {
	int32_t result;
	int32_t internal_result;
};

struct AjmSidebandStream {
	int32_t  size_consumed;
	int32_t  size_produced;
	uint64_t total_decoded_samples;
};

struct AjmSidebandMFrame {
	uint32_t num_frames;
	uint32_t reserved;
};

struct AjmSidebandResampleInfo {
	float    ratio;
	int32_t  num_samples;
	uint32_t reserved[8];
};

struct AjmGetStatisticsResult {
	AjmSidebandResult result;
	float             engine_usage_batch;
	float             engine_usage_interval[3];
	uint32_t          memory[6];
};

struct AjmDecOpusInitializeParameters {
	uint32_t channel_num;
	uint32_t sample_rate;
	uint32_t mapping_family;
};

struct AjmSidebandDecOpusCodecInfo {
	uint32_t frames_per_packet;
};

enum class AjmCodec : uint32_t {
	DecMp3           = 0,
	DecAt9           = 1,
	DecM4aac         = 2,
	DecUnknownNewABI = 14,
	DecOpus          = 24,
};

constexpr int32_t AJM_ERROR_INVALID_CONTEXT     = static_cast<int32_t>(0x80930002u);
constexpr int32_t AJM_ERROR_INVALID_PARAMETER   = static_cast<int32_t>(0x80930005u);
constexpr int32_t AJM_ERROR_CODEC_NOT_SUPPORTED = static_cast<int32_t>(0x80930008u);
constexpr int32_t AJM_ERROR_JOB_CREATION        = static_cast<int32_t>(0x80930012u);
constexpr size_t  AJM_JOB_CONTROL_SIZE          = 48;
constexpr size_t  AJM_JOB_RUN_SIZE              = 64;
constexpr size_t  AJM_JOB_GET_STATISTICS_SIZE =
    sizeof(uint32_t) + sizeof(float) * 3 + 8 + AJM_JOB_RUN_SIZE;
constexpr size_t AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE =
    sizeof(float) * 2 + sizeof(uint32_t) + 12 + AJM_JOB_CONTROL_SIZE;
constexpr size_t   AJM_SIDEBAND_RESULT_SIZE           = sizeof(AjmSidebandResult);
constexpr size_t   AJM_SIDEBAND_STREAM_SIZE           = sizeof(AjmSidebandStream);
constexpr size_t   AJM_SIDEBAND_MFRAME_SIZE           = sizeof(AjmSidebandMFrame);
constexpr size_t   AJM_SIDEBAND_FORMAT_SIZE           = sizeof(AjmSidebandFormat);
constexpr size_t   AJM_SIDEBAND_GAPLESS_SIZE          = sizeof(AjmSidebandGaplessDecode);
constexpr size_t   AJM_SIDEBAND_RESAMPLE_INFO_SIZE    = sizeof(AjmSidebandResampleInfo);
constexpr uint64_t AJM_FLAG_RUN_GET_CODEC_INFO        = 1ull << 11u;
constexpr uint64_t AJM_FLAG_RUN_MULTIPLE_FRAMES       = 1ull << 12u;
constexpr uint64_t AJM_FLAG_CONTROL_RESET             = 1ull << 13u;
constexpr uint64_t AJM_FLAG_CONTROL_INITIALIZE        = 1ull << 14u;
constexpr uint64_t AJM_FLAG_SIDEBAND_RESAMPLE_INFO    = 1ull << 44u;
constexpr uint64_t AJM_FLAG_SIDEBAND_GAPLESS          = 1ull << 45u;
constexpr uint64_t AJM_FLAG_SIDEBAND_FORMAT           = 1ull << 46u;
constexpr uint64_t AJM_FLAG_SIDEBAND_STREAM           = 1ull << 47u;
constexpr uint64_t AJM_INSTANCE_FLAG_FORMAT_OFFSET    = 7u;
constexpr uint64_t AJM_INSTANCE_FLAG_FORMAT_MASK      = 0x7u;
constexpr uint64_t AJM_INSTANCE_FLAG_MAX_CHANNEL_MASK = 0x7fu;
constexpr uint32_t AJM_DEC_OPUS_MAX_CHANNELS_FOR_10CH = 10;
constexpr uint32_t AJM_DEC_OPUS_FRAME_SAMPLES         = 960;

static std::atomic_uint32_t g_ajm_next_instance {1};
static std::atomic_uint32_t g_ajm_next_batch {1};

// Jobs run when their batch starts (sceAjmBatchStart), as on the console, not when they are added: a game may add a
// decode job before the input it names is filled (a streamed file read that completes before the batch starts) -
// decoded at once, that input was silence or stale data. KYTY_AJM_DEFER=0: run them as they are added (as before).
static bool AjmDeferJobs() {
	static const bool defer = [] {
		const char* value = std::getenv("KYTY_AJM_DEFER");
		return value == nullptr || *value == '\0' || *value != '0';
	}();
	return defer;
}
static std::mutex                                                        g_ajm_deferred_mutex;
static std::unordered_map<const void*, std::vector<std::function<void()>>> g_ajm_deferred;

static uint32_t AjmGetFlagChannelCount(uint64_t flags) {
	return static_cast<uint32_t>(flags & AJM_INSTANCE_FLAG_MAX_CHANNEL_MASK);
}

static AjmSampleEncoding AjmGetFlagSampleEncoding(uint64_t flags) {
	const auto encoding = static_cast<uint32_t>((flags >> AJM_INSTANCE_FLAG_FORMAT_OFFSET) &
	                                            AJM_INSTANCE_FLAG_FORMAT_MASK);
	switch (encoding) {
		case static_cast<uint32_t>(AjmSampleEncoding::S16): return AjmSampleEncoding::S16;
		case static_cast<uint32_t>(AjmSampleEncoding::S32): return AjmSampleEncoding::S32;
		case static_cast<uint32_t>(AjmSampleEncoding::Float): return AjmSampleEncoding::Float;
		default: EXIT("unsupported AJM PCM sample encoding %u\n", encoding);
	}
	return AjmSampleEncoding::S16;
}

static const char* AjmCodecName(uint32_t codec) {
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3): return "MP3 decoder";
		case static_cast<uint32_t>(AjmCodec::DecAt9): return "ATRAC9 decoder";
		case static_cast<uint32_t>(AjmCodec::DecM4aac): return "MPEG4-AAC decoder";
		case 3: return "CELP8 decoder";
		case 4: return "CELP8 encoder";
		case 12: return "CELP16 decoder";
		case 13: return "CELP16 encoder";
		case 22: return "HE-VAG decoder";
		case 23: return "LPCM decoder";
		case static_cast<uint32_t>(AjmCodec::DecOpus): return "Opus decoder";
		default: return "unknown";
	}
}

static_assert(sizeof(AjmDecAt9ConfigDataInfo) == 20);

int KYTY_SYSV_ABI AjmDecAt9ParseConfigData(const void*              config_data,
                                           AjmDecAt9ConfigDataInfo* config_info) {
	PRINT_NAME();
	if (config_data == nullptr || config_info == nullptr) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	auto* handle = Atrac9GetHandle();
	if (handle == nullptr) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	uint8_t config[ATRAC9_CONFIG_DATA_SIZE] {};
	std::memcpy(config, config_data, sizeof(config));
	// (A multichannel config: a channel's mono stream, then the channels' superframe.)
	At9MultiConfig  multi {};
	const bool      is_multi = At9ParseMultiConfig(config, &multi);
	Atrac9CodecInfo codec_info {};
	const int       init_result = Atrac9InitDecoder(handle, is_multi ? multi.mono : config);
	const int       info_result =
	    init_result == 0 ? Atrac9GetCodecInfo(handle, &codec_info) : init_result;
	Atrac9ReleaseHandle(handle);
	if (is_multi && info_result == 0) {
		codec_info.channels       = static_cast<int>(multi.channels);
		codec_info.superframeSize = codec_info.superframeSize * static_cast<int>(multi.channels);
	}

	if (info_result != 0 || codec_info.channels <= 0 || codec_info.samplingRate <= 0 ||
	    codec_info.frameSamples <= 0 || codec_info.framesInSuperframe <= 0 ||
	    codec_info.superframeSize <= 0) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	config_info->channels                  = static_cast<uint32_t>(codec_info.channels);
	config_info->sample_rate               = static_cast<uint32_t>(codec_info.samplingRate);
	config_info->frame_samples_per_channel = static_cast<uint32_t>(codec_info.frameSamples);
	config_info->superframe_samples_per_channel =
	    static_cast<uint32_t>(codec_info.frameSamples) *
	    static_cast<uint32_t>(codec_info.framesInSuperframe);
	config_info->superframe_size = static_cast<uint32_t>(codec_info.superframeSize);
	return OK;
}

class AjmOpusDecoder final: public AjmDecoder {
public:
	AjmOpusDecoder(uint32_t channels, uint32_t sample_rate, AjmSampleEncoding encoding,
	               uint64_t flags)
	    : AjmDecoder(channels, sample_rate, encoding) {
		(void)flags;
	}

	AjmDecodeResult Initialize(const void* codec_parameters,
	                           size_t      codec_parameters_size) override {
		auto result = MakeResult();

		if (codec_parameters == nullptr ||
		    codec_parameters_size < sizeof(AjmDecOpusInitializeParameters)) {
			result.result = AJM_RESULT_INVALID_PARAMETER;
			return result;
		}

		const auto* params = static_cast<const AjmDecOpusInitializeParameters*>(codec_parameters);
		if (params->channel_num == 0 || params->channel_num > AJM_DEC_OPUS_MAX_CHANNELS_FOR_10CH ||
		    params->sample_rate == 0) {
			result.result = AJM_RESULT_INVALID_PARAMETER;
			return result;
		}

		SetFormat(params->channel_num, params->sample_rate, m_sample_encoding);
		m_mapping_family        = params->mapping_family;
		m_total_decoded_samples = 0;
		m_is_initialized        = true;
		m_frames_per_packet     = 1;

		LOGF("AJM Opus initialized: %" PRIu32 " Hz, %" PRIu32 " ch, mapping=%" PRIu32 "\n",
		     params->sample_rate, params->channel_num, params->mapping_family);

		return MakeResult();
	}

	void Reset() override { m_total_decoded_samples = 0; }

	AjmDecodeResult Decode(const void* input, size_t input_size, void* output, size_t output_size,
	                       bool multiple_frames, AjmGaplessState* gapless) override {
		(void)multiple_frames;
		auto result = MakeResult();

		if (input == nullptr || input_size == 0) {
			result.result = AJM_RESULT_PARTIAL_INPUT;
			return result;
		}
		if (!m_is_initialized) {
			result.result = AJM_RESULT_NOT_INITIALIZED;
			return result;
		}

		auto samples_to_write = AJM_DEC_OPUS_FRAME_SAMPLES;
		auto skip_samples     = 0u;
		if (gapless != nullptr) {
			skip_samples = std::min<uint32_t>(samples_to_write, gapless->current.skip_samples);
			samples_to_write -= skip_samples;
			if (gapless->HasSampleLimit()) {
				samples_to_write =
				    std::min<uint32_t>(samples_to_write, gapless->current.total_samples);
			}
		}

		const auto output_bytes = static_cast<size_t>(samples_to_write) *
		                          static_cast<size_t>(m_channels) *
		                          AjmBytesPerSample(m_sample_encoding);
		if (output_bytes != 0 && output == nullptr) {
			result.result = AJM_RESULT_NOT_ENOUGH_ROOM;
			return result;
		}
		if (output_bytes > output_size) {
			result.result = AJM_RESULT_NOT_ENOUGH_ROOM;
			return result;
		}

		if (output_bytes != 0) {
			std::memset(output, 0, output_bytes);
		}

		if (gapless != nullptr) {
			gapless->current.skip_samples =
			    static_cast<uint16_t>(gapless->current.skip_samples - skip_samples);
			if (gapless->HasSampleLimit()) {
				gapless->current.total_samples =
				    (gapless->current.total_samples > samples_to_write
				         ? gapless->current.total_samples - samples_to_write
				         : 0);
			}

			const auto skipped_samples = AJM_DEC_OPUS_FRAME_SAMPLES - samples_to_write;
			const auto skipped_total   = std::min<uint32_t>(
			    std::numeric_limits<uint16_t>::max(),
			    static_cast<uint32_t>(gapless->current.skipped_samples) + skipped_samples);
			gapless->current.skipped_samples = static_cast<uint16_t>(skipped_total);
		}

		result.input_consumed    = input_size;
		result.output_written    = output_bytes;
		result.frames            = 1;
		result.frames_per_packet = m_frames_per_packet;
		m_total_decoded_samples += samples_to_write;
		result.total_decoded_samples = m_total_decoded_samples;
		result.format                = GetFormat();
		return result;
	}

	void WriteCodecInfo(void* output, size_t output_size,
	                    const AjmDecodeResult& result) const override {
		(void)result;
		if (output == nullptr || output_size < sizeof(AjmSidebandDecOpusCodecInfo)) {
			return;
		}

		auto* info              = static_cast<AjmSidebandDecOpusCodecInfo*>(output);
		info->frames_per_packet = m_frames_per_packet;
	}

	[[nodiscard]] size_t CodecInfoSize() const override {
		return sizeof(AjmSidebandDecOpusCodecInfo);
	}

private:
	bool     m_is_initialized    = false;
	uint32_t m_mapping_family    = 0;
	uint32_t m_frames_per_packet = 1;
};

struct AjmInstanceStats {
	uint32_t decodes = 0, frames = 0, riff = 0, partial = 0, no_room = 0, errors = 0, empty = 0;
	uint64_t in_bytes = 0, out_bytes = 0;
	float    peak = 0.0f;
};

struct AjmInstanceState {
	uint32_t                    context = 0;
	uint32_t                    codec   = 0;
	uint64_t                    flags   = 0;
	AjmGaplessState             gapless;
	std::unique_ptr<AjmDecoder> decoder;
	AjmInstanceStats            stats;
};

static std::mutex                                     g_ajm_instances_mutex;
static std::unordered_map<uint32_t, AjmInstanceState> g_ajm_instances;

static bool AjmCodecIsSupported(uint32_t codec) {
	return codec == static_cast<uint32_t>(AjmCodec::DecMp3) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecAt9) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecM4aac) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecOpus) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecUnknownNewABI);
}

static bool AjmLogCodecSupport(uint32_t codec) {
	const auto supported = AjmCodecIsSupported(codec);
	if (supported) {
		static std::atomic_bool logged {false};
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			std::printf("AJM codec=%u, name=%s\n", codec, AjmCodecName(codec));
		}
	} else {
		std::printf("AJM codec=%u, name=%s\n", codec, AjmCodecName(codec));
		std::printf("AJM unsupported codec=%u (%s)\n", codec, AjmCodecName(codec));
	}
	return supported;
}

static std::unique_ptr<AjmDecoder> AjmCreateDecoder(uint32_t codec, uint64_t flags) {
	if (!AjmCodecIsSupported(codec)) {
		return {};
	}

	auto channels = AjmGetFlagChannelCount(flags);
	if (channels == 0) {
		channels = 2;
	}

	const auto encoding = AjmGetFlagSampleEncoding(flags);
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3):
			return std::make_unique<AjmMp3Decoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecAt9):
			return std::make_unique<AjmAt9Decoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecM4aac):
			return std::make_unique<AjmAacDecoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecOpus):
			return std::make_unique<AjmOpusDecoder>(channels, 48000, encoding, flags);
		default: break;
	}

	return {};
}

static AjmInstanceState* AjmFindInstanceLocked(uint32_t instance) {
	const auto it = g_ajm_instances.find(instance);
	if (it == g_ajm_instances.end()) {
		return nullptr;
	}
	return &it->second;
}

static uint8_t* AjmReserveJob(AjmBatchInfo* info, size_t job_size) {
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	if (info->offset + job_size > info->size) {
		return nullptr;
	}

	auto* job              = static_cast<uint8_t*>(info->buffer) + info->offset;
	info->last_good_job    = job;
	info->last_good_job_ra = nullptr;
	info->offset += job_size;

	std::memset(job, 0, job_size);

	return job;
}

static int AjmAppendJob(AjmBatchInfo* info, size_t job_size, const char* name) {
	auto* job = AjmReserveJob(info, job_size);
	if (job == nullptr) {
		LOGF("\t %s: not enough batch space, offset=0x%016" PRIx64 ", size=0x%016" PRIx64
		     ", job=0x%016" PRIx64 "\n",
		     name, static_cast<uint64_t>(info->offset), static_cast<uint64_t>(info->size),
		     static_cast<uint64_t>(job_size));
		return AJM_ERROR_JOB_CREATION;
	}

	return OK;
}

// A job of the batch: its space taken now, its work run when the batch starts (AjmDeferJobs), or now.
static int AjmQueueJob(AjmBatchInfo* info, size_t job_size, const char* name, std::function<void()> work) {
	if (!AjmDeferJobs()) {
		work();
		return AjmAppendJob(info, job_size, name);
	}
	if (const int result = AjmAppendJob(info, job_size, name); result != OK) return result;
	std::scoped_lock lock(g_ajm_deferred_mutex);
	g_ajm_deferred[info->buffer].push_back(std::move(work));
	return OK;
}

static int32_t AjmClampInt32(size_t value) {
	return static_cast<int32_t>(
	    std::min<size_t>(value, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
}

static void AjmWriteResult(void* result, size_t size, const AjmDecodeResult& decode_result = {}) {
	if (result == nullptr || size == 0) {
		return;
	}

	std::memset(result, 0, size);

	if (size >= AJM_SIDEBAND_RESULT_SIZE) {
		auto* sideband            = static_cast<AjmSidebandResult*>(result);
		sideband->result          = decode_result.result;
		sideband->internal_result = decode_result.internal_result;
	}
}

static void AjmWriteStreamResult(void* result, size_t result_size,
                                 const AjmDecodeResult& decode_result, bool multiple_frames) {
	if (result == nullptr) {
		return;
	}

	AjmWriteResult(result, result_size, decode_result);

	if (result_size >= AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE) {
		auto* stream          = reinterpret_cast<AjmSidebandStream*>(static_cast<uint8_t*>(result) +
		                                                             AJM_SIDEBAND_RESULT_SIZE);
		stream->size_consumed = AjmClampInt32(decode_result.input_consumed);
		stream->size_produced = AjmClampInt32(decode_result.output_written);
		stream->total_decoded_samples = decode_result.total_decoded_samples;
	}

	if (multiple_frames && result_size >= AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE +
	                                          AJM_SIDEBAND_MFRAME_SIZE) {
		auto* mframe = reinterpret_cast<AjmSidebandMFrame*>(
		    static_cast<uint8_t*>(result) + AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE);
		mframe->num_frames = decode_result.frames;
	}
}

static void AjmWriteFormat(void* result, size_t result_size, const AjmSidebandFormat& format,
                           const AjmDecodeResult& decode_result = {}) {
	if (result == nullptr || result_size < AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_FORMAT_SIZE) {
		AjmWriteResult(result, result_size, decode_result);
		return;
	}

	AjmWriteResult(result, result_size, decode_result);

	auto* format_output = reinterpret_cast<AjmSidebandFormat*>(static_cast<uint8_t*>(result) +
	                                                           AJM_SIDEBAND_RESULT_SIZE);
	*format_output      = format;
}

static void AjmWriteCodecInfo(AjmDecoder* decoder, void* result, size_t result_size,
                              const AjmDecodeResult& decode_result) {
	if (decoder == nullptr || result == nullptr || result_size <= AJM_SIDEBAND_RESULT_SIZE) {
		return;
	}
	decoder->WriteCodecInfo(static_cast<uint8_t*>(result) + AJM_SIDEBAND_RESULT_SIZE,
	                        result_size - AJM_SIDEBAND_RESULT_SIZE, decode_result);
}

static void AjmWriteSideband(uint64_t flags, void* sideband_output, size_t sideband_output_size,
                             AjmDecoder* decoder, const AjmGaplessState* gapless,
                             const AjmDecodeResult& decode_result) {
	if (sideband_output == nullptr || sideband_output_size == 0) {
		return;
	}

	AjmWriteResult(sideband_output, sideband_output_size, decode_result);

	size_t offset = AJM_SIDEBAND_RESULT_SIZE;

	if ((flags & AJM_FLAG_RUN_GET_CODEC_INFO) != 0 && decoder != nullptr) {
		const auto codec_info_size = decoder->CodecInfoSize();
		if (offset + codec_info_size <= sideband_output_size) {
			decoder->WriteCodecInfo(static_cast<uint8_t*>(sideband_output) + offset,
			                        sideband_output_size - offset, decode_result);
			offset += codec_info_size;
		}
	}

	if ((flags & AJM_FLAG_SIDEBAND_RESAMPLE_INFO) != 0 &&
	    offset + AJM_SIDEBAND_RESAMPLE_INFO_SIZE <= sideband_output_size) {
		auto* info = reinterpret_cast<AjmSidebandResampleInfo*>(
		    static_cast<uint8_t*>(sideband_output) + offset);
		info->ratio = 1.0f;
		offset += AJM_SIDEBAND_RESAMPLE_INFO_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_GAPLESS) != 0 &&
	    offset + AJM_SIDEBAND_GAPLESS_SIZE <= sideband_output_size) {
		auto* gapless_output = reinterpret_cast<AjmSidebandGaplessDecode*>(
		    static_cast<uint8_t*>(sideband_output) + offset);
		if (gapless != nullptr) {
			*gapless_output = gapless->current;
		}
		offset += AJM_SIDEBAND_GAPLESS_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_FORMAT) != 0 &&
	    offset + AJM_SIDEBAND_FORMAT_SIZE <= sideband_output_size) {
		auto* format =
		    reinterpret_cast<AjmSidebandFormat*>(static_cast<uint8_t*>(sideband_output) + offset);
		*format = decode_result.format;
		offset += AJM_SIDEBAND_FORMAT_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_STREAM) != 0 &&
	    offset + AJM_SIDEBAND_STREAM_SIZE <= sideband_output_size) {
		auto* stream =
		    reinterpret_cast<AjmSidebandStream*>(static_cast<uint8_t*>(sideband_output) + offset);
		stream->size_consumed         = AjmClampInt32(decode_result.input_consumed);
		stream->size_produced         = AjmClampInt32(decode_result.output_written);
		stream->total_decoded_samples = decode_result.total_decoded_samples;
		offset += AJM_SIDEBAND_STREAM_SIZE;
	}

	if ((flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0 &&
	    offset + AJM_SIDEBAND_MFRAME_SIZE <= sideband_output_size) {
		auto* mframe =
		    reinterpret_cast<AjmSidebandMFrame*>(static_cast<uint8_t*>(sideband_output) + offset);
		mframe->num_frames = decode_result.frames;
	}
}

static bool AjmCodecIsValid(uint32_t codec) {
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3):
		case static_cast<uint32_t>(AjmCodec::DecAt9):
		case static_cast<uint32_t>(AjmCodec::DecM4aac):
		case 3:
		case 4:
		case 12:
		case 13:
		case static_cast<uint32_t>(AjmCodec::DecUnknownNewABI):
		case 22:
		case 23:
		case static_cast<uint32_t>(AjmCodec::DecOpus): return true;
		default: return false;
	}
}

// In the run log: the first decode jobs that failed (a stream the game then drops: missing voice lines), and a count.
static void AjmLogDecodeResult(const char* job, uint32_t instance, const AjmDecodeResult& result, size_t in,
                               size_t out) {
	constexpr uint32_t failures = static_cast<uint32_t>(AJM_RESULT_CODEC_ERROR) | static_cast<uint32_t>(AJM_RESULT_FATAL) |
	                              static_cast<uint32_t>(AJM_RESULT_INVALID_DATA) | static_cast<uint32_t>(AJM_RESULT_NOT_INITIALIZED) |
	                              static_cast<uint32_t>(AJM_RESULT_INVALID_PARAMETER);
	if ((static_cast<uint32_t>(result.result) & failures) == 0) return;
	static std::atomic<uint32_t> count {0};
	const auto n = count.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 32 || (n & (n - 1)) == 0)
		std::printf("AJM %s failed #%u: instance 0x%08" PRIx32 " (codec %u) result 0x%08x, in %zu, out %zu\n", job, n,
		            instance, instance >> 14u, static_cast<unsigned>(result.result), in, out);
}

// KYTY_AUDIO_STATS: each instance's decodes over 5 s (called with g_ajm_instances_mutex held): what was decoded and
// how loud, whether new files (RIFF headers) started, and the results that produce nothing (partial input, no room,
// errors). A voice the game decodes as silence, or never decodes, shows here.
static void AjmNoteDecode(uint32_t instance, AjmInstanceState& state, const void* input, size_t input_size, const void* output,
                          const AjmDecodeResult& result) {
	auto& st = state.stats;
	st.decodes++;
	st.frames += result.frames;
	st.in_bytes += result.input_consumed;
	st.out_bytes += result.output_written;
	if (input != nullptr && input_size >= 4 && std::memcmp(input, "RIFF", 4) == 0) st.riff++;
	if ((result.result & AJM_RESULT_PARTIAL_INPUT) != 0) st.partial++;
	if ((result.result & AJM_RESULT_NOT_ENOUGH_ROOM) != 0) st.no_room++;
	if ((result.result & (AJM_RESULT_CODEC_ERROR | AJM_RESULT_FATAL | AJM_RESULT_INVALID_DATA | AJM_RESULT_NOT_INITIALIZED |
	                      AJM_RESULT_INVALID_PARAMETER)) != 0)
		st.errors++;
	if (result.output_written == 0) st.empty++;
	const auto encoding = static_cast<int>(result.format.sample_encoding);
	const auto bytes    = encoding == 0 ? 2u : 4u;
	st.peak = std::max(st.peak, Libs::Audio::AudioInternal::PcmPeak(output, result.output_written / bytes, encoding, 7));
	(void)instance;

	static double last = 0.0;
	const double  now  = Libs::Audio::AudioInternal::StatsSeconds();
	if (last == 0.0) last = now;
	if (now - last < 5.0) return;
	last = now;
	uint32_t active = 0;
	for (const auto& [id, s]: g_ajm_instances)
		if (s.stats.decodes != 0) active++;
	std::printf("[ajm] %.0f s: %u instances decoded (instance: channels, rate, decodes, frames, new files, KiB in/out, peak; "
	            "partial/no room/errors/nothing out)\n",
	            5.0, active);
	for (auto& [id, s]: g_ajm_instances) {
		auto& t = s.stats;
		if (t.decodes == 0) continue;
		const auto format = s.decoder != nullptr ? s.decoder->GetFormat() : AjmSidebandFormat {};
		std::printf("[ajm]  0x%05x %uch %u Hz: %u dec, %u fr, %u riff, %llu/%llu KiB, peak %.3f; %u/%u/%u/%u\n", id, format.channel_num,
		            format.sampling_frequency, t.decodes, t.frames, t.riff, static_cast<unsigned long long>(t.in_bytes / 1024),
		            static_cast<unsigned long long>(t.out_bytes / 1024), static_cast<double>(t.peak), t.partial, t.no_room, t.errors,
		            t.empty);
		t = {};
	}
}

static AjmDecodeResult AjmDecodeInstance(uint32_t instance, const void* input, size_t input_size,
                                         void* output, size_t output_size, bool multiple_frames) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr) {
		AjmDecodeResult result {};
		result.result = AJM_RESULT_INVALID_PARAMETER;
		return result;
	}

	auto result = state->decoder->Decode(input, input_size, output, output_size, multiple_frames, &state->gapless);
	if (Libs::Audio::AudioInternal::StatsWanted()) AjmNoteDecode(instance, *state, input, input_size, output, result);
	return result;
}

static AjmDecodeResult AjmDecodeSplitInstance(uint32_t instance, const AjmBuffer* input_buffers,
                                              size_t           input_buffers_num,
                                              const AjmBuffer* output_buffers,
                                              size_t output_buffers_num, bool multiple_frames) {
	size_t input_size  = 0;
	size_t output_size = 0;
	for (size_t i = 0; i < input_buffers_num && input_buffers != nullptr; i++) {
		input_size += input_buffers[i].size;
	}
	for (size_t i = 0; i < output_buffers_num && output_buffers != nullptr; i++) {
		output_size += output_buffers[i].size;
	}

	// (Zeros past the input: LibAtrac9's bit reader takes up to 4 bytes at a time and reads past a frame's last byte. With
	// the input's exact size it read past the allocation, which a heap without slack after its blocks turned into an
	// access violation in PeekInt.)
	constexpr size_t     ReadSlack = 16;
	std::vector<uint8_t> input(input_size + ReadSlack);
	size_t               input_offset = 0;
	for (size_t i = 0; i < input_buffers_num && input_buffers != nullptr; i++) {
		if (input_buffers[i].address != nullptr && input_buffers[i].size != 0) {
			std::memcpy(input.data() + input_offset, input_buffers[i].address,
			            input_buffers[i].size);
		}
		input_offset += input_buffers[i].size;
	}

	std::vector<uint8_t> output(output_size);
	auto result = AjmDecodeInstance(instance, input.data(), input_size, output.data(),
	                                output.size(), multiple_frames);

	size_t output_offset = 0;
	for (size_t i = 0; i < output_buffers_num && output_buffers != nullptr; i++) {
		const auto bytes = std::min(
		    output_buffers[i].size,
		    (result.output_written > output_offset ? result.output_written - output_offset : 0));
		if (output_buffers[i].address != nullptr && bytes != 0) {
			std::memcpy(output_buffers[i].address, output.data() + output_offset, bytes);
		}
		output_offset += output_buffers[i].size;
	}

	return result;
}

static AjmDecodeResult AjmInitializeInstance(uint32_t instance, const void* codec_parameters,
                                             size_t codec_parameters_size) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr) {
		AjmDecodeResult result {};
		result.result = AJM_RESULT_INVALID_PARAMETER;
		return result;
	}

	state->gapless.Reset();
	return state->decoder->Initialize(codec_parameters, codec_parameters_size);
}

static AjmDecodeResult AjmControlInstance(uint32_t instance, uint64_t flags,
                                          const void* sideband_input, size_t sideband_input_size,
                                          AjmDecoder** out_decoder, AjmGaplessState** out_gapless) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr) {
		AjmDecodeResult result {};
		result.result = AJM_RESULT_INVALID_PARAMETER;
		return result;
	}

	auto* decoder = state->decoder.get();
	if (out_decoder != nullptr) {
		*out_decoder = decoder;
	}
	if (out_gapless != nullptr) {
		*out_gapless = &state->gapless;
	}

	AjmDecodeResult result = decoder->MakeResult();
	if ((flags & AJM_FLAG_SIDEBAND_GAPLESS) != 0) {
		if (sideband_input == nullptr || sideband_input_size < AJM_SIDEBAND_GAPLESS_SIZE) {
			result.result = AJM_RESULT_INVALID_PARAMETER;
			return result;
		}
		state->gapless.Set(*static_cast<const AjmSidebandGaplessDecode*>(sideband_input),
		                   (flags & AJM_FLAG_CONTROL_RESET) != 0);
		sideband_input = static_cast<const uint8_t*>(sideband_input) + AJM_SIDEBAND_GAPLESS_SIZE;
		sideband_input_size -= AJM_SIDEBAND_GAPLESS_SIZE;
	}
	if ((flags & AJM_FLAG_CONTROL_RESET) != 0) {
		state->gapless.Reset();
		decoder->Reset();
		result = decoder->MakeResult();
	}
	if ((flags & AJM_FLAG_CONTROL_INITIALIZE) != 0) {
		state->gapless.Reset();
		result = decoder->Initialize(sideband_input, sideband_input_size);
	}

	return result;
}

static AjmDecodeResult AjmSetGaplessDecode(uint32_t instance, const void* gapless_decode,
                                           int reset) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr || gapless_decode == nullptr) {
		AjmDecodeResult result {};
		result.result = AJM_RESULT_INVALID_PARAMETER;
		return result;
	}

	state->gapless.Set(*static_cast<const AjmSidebandGaplessDecode*>(gapless_decode), reset != 0);
	return state->decoder->MakeResult();
}

static AjmDecodeResult AjmGetGaplessDecode(uint32_t                  instance,
                                           AjmSidebandGaplessDecode* gapless_decode) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr) {
		AjmDecodeResult result {};
		result.result = AJM_RESULT_INVALID_PARAMETER;
		return result;
	}

	if (gapless_decode != nullptr) {
		*gapless_decode = state->gapless.current;
	}
	return state->decoder->MakeResult();
}

static AjmSidebandFormat AjmGetInstanceFormat(uint32_t          instance,
                                              AjmDecodeResult*  out_result  = nullptr,
                                              AjmDecoder**      out_decoder = nullptr,
                                              AjmGaplessState** out_gapless = nullptr) {
	std::scoped_lock lock(g_ajm_instances_mutex);
	auto*            state = AjmFindInstanceLocked(instance);
	if (state == nullptr || state->decoder == nullptr) {
		if (out_result != nullptr) {
			out_result->result = AJM_RESULT_INVALID_PARAMETER;
		}
		return AjmMakeFormat(2, 48000, AjmSampleEncoding::S16);
	}

	if (out_decoder != nullptr) {
		*out_decoder = state->decoder.get();
	}
	if (out_gapless != nullptr) {
		*out_gapless = &state->gapless;
	}
	if (out_result != nullptr) {
		*out_result        = state->decoder->MakeResult();
		out_result->format = state->decoder->GetFormat();
	}
	return state->decoder->GetFormat();
}

int KYTY_SYSV_ABI AjmInitialize(int64_t reserved, uint32_t* context) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(context == nullptr);
	if (reserved != 0) {
		LOGF("\t ignoring reserved = 0x%016" PRIx64 "\n", static_cast<uint64_t>(reserved));
	}

	*context = 1;

	return OK;
}

int KYTY_SYSV_ABI AjmFinalize(uint32_t context) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 "\n", context);
	return OK;
}

int KYTY_SYSV_ABI AjmModuleRegister(uint32_t context, uint32_t codec, int64_t reserved) {
	PRINT_NAME();

	if (reserved != 0) {
		LOGF("\t ignoring reserved = 0x%016" PRIx64 "\n", static_cast<uint64_t>(reserved));
	}

	LOGF("\t codec = %u\n", codec);
	LOGF("\t %s\n", AjmCodecName(codec));

	const auto valid     = AjmCodecIsValid(codec);
	const auto supported = valid && AjmLogCodecSupport(codec);
	int        result    = OK;
	if (context != 1) {
		result = AJM_ERROR_INVALID_CONTEXT;
	} else if (!valid) {
		result = AJM_ERROR_INVALID_PARAMETER;
	} else if (!supported) {
		result = AJM_ERROR_CODEC_NOT_SUPPORTED;
	}

	if (!supported) {
		LOGF("\t codec is not implemented\n");
	}

	return result;
}

int KYTY_SYSV_ABI AjmModuleUnregister(uint32_t context, uint32_t codec) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", codec = %" PRIu32 "\n", context, codec);
	return OK;
}

int KYTY_SYSV_ABI AjmInstanceCreate(uint32_t context, uint32_t codec, uint64_t flags,
                                    uint32_t* instance) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(instance == nullptr);
	const auto supported = AjmLogCodecSupport(codec);

	const auto slot = (g_ajm_next_instance.fetch_add(1, std::memory_order_relaxed) & 0x3fffu);
	*instance       = (codec << 14u) | slot;

	auto state    = AjmInstanceState {};
	state.context = context;
	state.codec   = codec;
	state.flags   = flags;
	state.decoder = AjmCreateDecoder(codec, flags);

	// (In the run log, the first instances: which codecs the game decodes with, and how many channels.)
	static std::atomic<uint32_t> logged_instances {0};
	if (logged_instances.fetch_add(1, std::memory_order_relaxed) < 32)
		std::printf("AJM instance 0x%08" PRIx32 ": codec %" PRIu32 " (%s), flags 0x%016" PRIx64 ", %u channels%s\n", *instance, codec,
		            AjmCodecName(codec), flags, AjmGetFlagChannelCount(flags), state.decoder ? "" : ", no decoder");
	{
		std::scoped_lock lock(g_ajm_instances_mutex);
		g_ajm_instances[*instance] = std::move(state);
	}

	LOGF("\t context  = %" PRIu32 "\n"
	     "\t codec    = %" PRIu32 "\n"
	     "\t flags    = 0x%016" PRIx64 "\n"
	     "\t instance = 0x%08" PRIx32 "\n",
	     context, codec, flags, *instance);
	if (!supported) {
		LOGF("\t unsupported codec instance created without decoder\n");
	}

	return OK;
}

int KYTY_SYSV_ABI AjmInstanceDestroy(uint32_t context, uint32_t instance) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", instance = 0x%08" PRIx32 "\n", context, instance);

	std::scoped_lock lock(g_ajm_instances_mutex);
	g_ajm_instances.erase(instance);

	return OK;
}

int KYTY_SYSV_ABI AjmMemoryRegister(uint32_t context, void* ptr, size_t pages) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", ptr = 0x%016" PRIx64 ", pages = 0x%016" PRIx64 "\n", context,
	     reinterpret_cast<uint64_t>(ptr), static_cast<uint64_t>(pages));
	return OK;
}

int KYTY_SYSV_ABI AjmMemoryUnregister(uint32_t context, void* ptr) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", ptr = 0x%016" PRIx64 "\n", context,
	     reinterpret_cast<uint64_t>(ptr));
	return OK;
}

int KYTY_SYSV_ABI AjmBatchInitialize(void* buffer, size_t size, AjmBatchInfo* info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(buffer == nullptr);
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	{
		// (Jobs added to this buffer before and never started are dropped, as a new batch overwrites them.)
		std::scoped_lock lock(g_ajm_deferred_mutex);
		g_ajm_deferred.erase(buffer);
	}
	info->buffer           = buffer;
	info->offset           = 0;
	info->size             = size;
	info->last_good_job    = nullptr;
	info->last_good_job_ra = nullptr;

	return OK;
}

int KYTY_SYSV_ABI AjmBatchStart(uint32_t context, const AjmBatchInfo* info, int priority,
                                AjmBatchError* error, uint32_t* batch) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(info == nullptr);
	EXIT_NOT_IMPLEMENTED(batch == nullptr);

	(void)context;
	(void)priority;

	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}

	*batch = g_ajm_next_batch.fetch_add(1, std::memory_order_relaxed);

	// The batch's jobs, in order (AjmQueueJob).
	std::vector<std::function<void()>> jobs;
	{
		std::scoped_lock lock(g_ajm_deferred_mutex);
		if (const auto found = g_ajm_deferred.find(info->buffer); found != g_ajm_deferred.end()) {
			jobs = std::move(found->second);
			g_ajm_deferred.erase(found);
		}
	}
	for (auto& job: jobs) job();

	return OK;
}

int KYTY_SYSV_ABI AjmBatchWait(uint32_t context, uint32_t batch, uint32_t timeout,
                               AjmBatchError* error) {
	(void)context;
	(void)batch;
	(void)timeout;

	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}

	return OK;
}

int KYTY_SYSV_ABI AjmBatchCancel(uint32_t context, uint32_t batch) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", batch = %" PRIu32 "\n", context, batch);
	return OK;
}

int KYTY_SYSV_ABI AjmBatchErrorDump(const AjmBatchInfo* info, AjmBatchError* error) {
	PRINT_NAME();
	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}
	LOGF("\t info = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(info));
	return OK;
}

int KYTY_SYSV_ABI AjmBatchJobInitialize(AjmBatchInfo* info, uint32_t instance,
                                        const void* codec_parameters, size_t codec_parameters_size,
                                        void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	// (The parameters are copied into the batch, as the console's library does.)
	std::vector<uint8_t> parameters;
	if (codec_parameters != nullptr && codec_parameters_size != 0) {
		const auto* bytes = static_cast<const uint8_t*>(codec_parameters);
		parameters.assign(bytes, bytes + codec_parameters_size);
	}
	return AjmQueueJob(info, AJM_JOB_CONTROL_SIZE, "initialize", [instance, parameters = std::move(parameters), result] {
		const auto decode_result =
		    AjmInitializeInstance(instance, parameters.empty() ? nullptr : parameters.data(), parameters.size());
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobClearContext(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_CONTROL_SIZE, "clear-context", [instance, result] {
		AjmDecodeResult decode_result {};
		{
			std::scoped_lock lock(g_ajm_instances_mutex);
			auto*            state = AjmFindInstanceLocked(instance);
			if (state != nullptr && state->decoder != nullptr) {
				state->gapless.Reset();
				state->decoder->Reset();
				decode_result = state->decoder->MakeResult();
			} else {
				decode_result.result = AJM_RESULT_INVALID_PARAMETER;
			}
		}
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobDecode(AjmBatchInfo* info, uint32_t instance,
                                    const void* bitstream_input, size_t bitstream_input_size,
                                    void* pcm_output, size_t pcm_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "decode",
	                   [instance, bitstream_input, bitstream_input_size, pcm_output, pcm_output_size, result] {
		                   const auto decode_result = AjmDecodeInstance(instance, bitstream_input, bitstream_input_size,
		                                                                pcm_output, pcm_output_size, true);
		                   AjmLogDecodeResult("decode", instance, decode_result, bitstream_input_size, pcm_output_size);
		                   AjmWriteStreamResult(
		                       result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE + AJM_SIDEBAND_MFRAME_SIZE,
		                       decode_result, true);
	                   });
}

int KYTY_SYSV_ABI AjmBatchJobDecodeSingle(AjmBatchInfo* info, uint32_t instance,
                                          const void* bitstream_input, size_t bitstream_input_size,
                                          void* pcm_output, size_t pcm_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "decode-single",
	                   [instance, bitstream_input, bitstream_input_size, pcm_output, pcm_output_size, result] {
		                   const auto decode_result = AjmDecodeInstance(instance, bitstream_input, bitstream_input_size,
		                                                                pcm_output, pcm_output_size, false);
		                   AjmLogDecodeResult("decode-single", instance, decode_result, bitstream_input_size,
		                                      pcm_output_size);
		                   AjmWriteStreamResult(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE,
		                                        decode_result, false);
	                   });
}

int KYTY_SYSV_ABI AjmBatchJobDecodeSplit(AjmBatchInfo* info, uint32_t instance,
                                         const AjmBuffer* input_buffers, size_t input_buffers_num,
                                         const AjmBuffer* output_buffers, size_t output_buffers_num,
                                         void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	// (The buffer lists are copied into the batch: the caller's arrays may be gone when it starts.)
	std::vector<AjmBuffer> inputs(input_buffers, input_buffers + (input_buffers != nullptr ? input_buffers_num : 0));
	std::vector<AjmBuffer> outputs(output_buffers, output_buffers + (output_buffers != nullptr ? output_buffers_num : 0));
	return AjmQueueJob(info, 16 * (input_buffers_num + output_buffers_num) + 32, "decode-split",
	                   [instance, inputs = std::move(inputs), outputs = std::move(outputs), result] {
		                   const auto decode_result = AjmDecodeSplitInstance(instance, inputs.data(), inputs.size(),
		                                                                     outputs.data(), outputs.size(), true);
		                   AjmLogDecodeResult("decode-split", instance, decode_result, inputs.size(), outputs.size());
		                   AjmWriteStreamResult(
		                       result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE + AJM_SIDEBAND_MFRAME_SIZE,
		                       decode_result, true);
	                   });
}

int KYTY_SYSV_ABI AjmBatchJobEncode(AjmBatchInfo* info, uint32_t instance, const void* pcm_input,
                                    size_t pcm_input_size, void* bitstream_output,
                                    size_t bitstream_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	AjmDecodeResult encode_result {};
	encode_result.input_consumed = pcm_input_size;
	encode_result.output_written = bitstream_output_size;
	encode_result.frames         = (pcm_input_size != 0 || bitstream_output_size != 0 ? 1u : 0u);
	encode_result.total_decoded_samples = bitstream_output_size;
	if (bitstream_output != nullptr && bitstream_output_size != 0) {
		std::memset(bitstream_output, 0, bitstream_output_size);
	}

	AjmWriteStreamResult(
	    result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE + AJM_SIDEBAND_MFRAME_SIZE,
	    encode_result, true);
	LOGF("\t instance = 0x%08" PRIx32 ", input = 0x%016" PRIx64 ", output = 0x%016" PRIx64 "\n",
	     instance, reinterpret_cast<uint64_t>(pcm_input),
	     reinterpret_cast<uint64_t>(bitstream_output));
	return AjmAppendJob(info, AJM_JOB_RUN_SIZE, "encode");
}

int KYTY_SYSV_ABI AjmBatchJobGetInfo(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "get-info", [instance, result] {
		AjmDecodeResult decode_result {};
		const auto      format = AjmGetInstanceFormat(instance, &decode_result);
		AjmWriteFormat(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_FORMAT_SIZE, format, decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobGetCodecInfo(AjmBatchInfo* info, uint32_t instance, void* result,
                                          size_t result_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "get-codec-info", [instance, result, result_size] {
		AjmDecodeResult decode_result {};
		AjmDecoder*     decoder = nullptr;
		(void)AjmGetInstanceFormat(instance, &decode_result, &decoder);
		AjmWriteResult(result, result_size, decode_result);
		AjmWriteCodecInfo(decoder, result, result_size, decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobSetGaplessDecode(AjmBatchInfo* info, uint32_t instance,
                                              const void* gapless_decode, int reset, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	AjmSidebandGaplessDecode gapless {};
	const bool               has_gapless = gapless_decode != nullptr;
	if (has_gapless) std::memcpy(&gapless, gapless_decode, sizeof(gapless));
	return AjmQueueJob(info, AJM_JOB_CONTROL_SIZE, "set-gapless", [instance, gapless, has_gapless, reset, result] {
		const auto decode_result = AjmSetGaplessDecode(instance, has_gapless ? &gapless : nullptr, reset);
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobGetGaplessDecode(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "get-gapless", [instance, result] {
		AjmSidebandGaplessDecode gapless_decode {};
		const auto               decode_result = AjmGetGaplessDecode(instance, &gapless_decode);
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_GAPLESS_SIZE, decode_result);
		if (result != nullptr) {
			auto* gapless = reinterpret_cast<AjmSidebandGaplessDecode*>(static_cast<uint8_t*>(result) +
			                                                            AJM_SIDEBAND_RESULT_SIZE);
			*gapless      = gapless_decode;
		}
	});
}

int KYTY_SYSV_ABI AjmBatchJobSetResampleParameters(AjmBatchInfo* info, uint32_t instance,
                                                   float ratio, uint32_t flags, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE);
	LOGF("\t instance = 0x%08" PRIx32 ", ratio = %f, flags = 0x%08" PRIx32 "\n", instance,
	     static_cast<double>(ratio), flags);
	{
		// (Not done here: decoded audio keeps its rate. In the run log, the first requests.)
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("AJM resample requested (ignored): instance 0x%05x ratio %f flags 0x%x\n", instance, static_cast<double>(ratio), flags);
	}
	return AjmAppendJob(info, AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample");
}

int KYTY_SYSV_ABI AjmBatchJobSetResampleParametersEx(AjmBatchInfo* info, uint32_t instance,
                                                     float ratio_start,
                                                     float ratio_change_per_sample, uint32_t flags,
                                                     void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE);
	{
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("AJM resample-ex requested (ignored): instance 0x%05x ratio %f change %f flags 0x%x\n", instance,
			            static_cast<double>(ratio_start), static_cast<double>(ratio_change_per_sample), flags);
	}
	LOGF("\t instance = 0x%08" PRIx32 ", ratio_start = %f, ratio_change = %f, flags = 0x%08" PRIx32
	     "\n",
	     instance, static_cast<double>(ratio_start), static_cast<double>(ratio_change_per_sample),
	     flags);
	return AjmAppendJob(info, AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample-ex");
}

int KYTY_SYSV_ABI AjmBatchJobGetResampleInfo(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "get-resample-info", [instance, result] {
		AjmDecodeResult decode_result {};
		(void)AjmGetInstanceFormat(instance, &decode_result);
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_RESAMPLE_INFO_SIZE, decode_result);
		if (result != nullptr) {
			auto* resample  = reinterpret_cast<AjmSidebandResampleInfo*>(static_cast<uint8_t*>(result) +
			                                                             AJM_SIDEBAND_RESULT_SIZE);
			resample->ratio = 1.0f;
		}
	});
}

int KYTY_SYSV_ABI AjmBatchJobGetStatistics(AjmBatchInfo* info, float interval, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	AjmWriteResult(result, sizeof(AjmGetStatisticsResult));
	(void)interval;
	return AjmAppendJob(info, AJM_JOB_GET_STATISTICS_SIZE, "get-statistics");
}

int KYTY_SYSV_ABI AjmBatchJobControl(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                     const void* sideband_input, size_t sideband_input_size,
                                     void* sideband_output, size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	std::vector<uint8_t> input;
	if (sideband_input != nullptr && sideband_input_size != 0) {
		const auto* bytes = static_cast<const uint8_t*>(sideband_input);
		input.assign(bytes, bytes + sideband_input_size);
	}
	return AjmQueueJob(info, AJM_JOB_CONTROL_SIZE, "control",
	                   [instance, flags, input = std::move(input), sideband_output, sideband_output_size] {
		                   AjmDecoder*      decoder = nullptr;
		                   AjmGaplessState* gapless = nullptr;
		                   const auto decode_result = AjmControlInstance(instance, flags, input.empty() ? nullptr : input.data(),
		                                                                 input.size(), &decoder, &gapless);
		                   AjmWriteSideband(flags, sideband_output, sideband_output_size, decoder, gapless, decode_result);
	                   });
}

int KYTY_SYSV_ABI AjmBatchJobRun(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                 const void* data_input, size_t data_input_size, void* data_output,
                                 size_t data_output_size, void* sideband_output,
                                 size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	return AjmQueueJob(info, AJM_JOB_RUN_SIZE, "run",
	                   [instance, flags, data_input, data_input_size, data_output, data_output_size, sideband_output,
	                    sideband_output_size] {
		                   auto decode_result = AjmDecodeInstance(instance, data_input, data_input_size, data_output,
		                                                          data_output_size, (flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0);
		                   AjmLogDecodeResult("run", instance, decode_result, data_input_size, data_output_size);
		                   AjmDecoder*      decoder = nullptr;
		                   AjmGaplessState* gapless = nullptr;
		                   (void)AjmGetInstanceFormat(instance, nullptr, &decoder, &gapless);
		                   AjmWriteSideband(flags, sideband_output, sideband_output_size, decoder, gapless, decode_result);
	                   });
}

int KYTY_SYSV_ABI AjmBatchJobRunSplit(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                      const AjmBuffer* input_buffers, size_t input_buffers_num,
                                      const AjmBuffer* output_buffers, size_t output_buffers_num,
                                      void* sideband_output, size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	std::vector<AjmBuffer> inputs(input_buffers, input_buffers + (input_buffers != nullptr ? input_buffers_num : 0));
	std::vector<AjmBuffer> outputs(output_buffers, output_buffers + (output_buffers != nullptr ? output_buffers_num : 0));
	return AjmQueueJob(info, 16 * (input_buffers_num + output_buffers_num) + 32, "run-split",
	                   [instance, flags, inputs = std::move(inputs), outputs = std::move(outputs), sideband_output,
	                    sideband_output_size] {
		                   auto decode_result = AjmDecodeSplitInstance(instance, inputs.data(), inputs.size(), outputs.data(),
		                                                               outputs.size(), (flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0);
		                   AjmLogDecodeResult("run-split", instance, decode_result, inputs.size(), outputs.size());
		                   AjmDecoder*      decoder = nullptr;
		                   AjmGaplessState* gapless = nullptr;
		                   (void)AjmGetInstanceFormat(instance, nullptr, &decoder, &gapless);
		                   AjmWriteSideband(flags, sideband_output, sideband_output_size, decoder, gapless, decode_result);
	                   });
}

const char* KYTY_SYSV_ABI AjmStrError(int error) {
	PRINT_NAME();
	LOGF("\t error = %d\n", error);
	return "AJM";
}

} // namespace Libs::Audio::Ajm
