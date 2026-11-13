#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_

#include "common/common.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace Libs::Audio::AudioInternal {

enum class Format {
	Unknown,
	Signed16bitMono,
	Signed16bitStereo,
	Signed16bit8Ch,
	FloatMono,
	FloatStereo,
	Float8Ch,
	Signed16bit8ChStd,
	Float8ChStd,
	Float12Ch,
};

struct OutputParam {
	int         handle = 0;
	const void* data   = nullptr;
	// One gain for each of the port's channels, from the guest's per-channel gain attribute
	// (AudioOut2 port attribute 1), or null when the guest never set one: the pad's speaker and
	// vibration ports ask for their level there, and without it they play far too quietly.
	const float* gains = nullptr;
};

static constexpr int OUT_PORTS_MAX = 32;

int      AudioOutOpen(int type, uint32_t samples_num, uint32_t freq, Format format);
void     AudioOutClose(int handle);
bool     AudioOutHasDevice(int handle);
uint32_t AudioOutOutputs(const OutputParam* params, uint32_t num, bool blocking = true);

// KYTY_AUDIO_STATS (on unless =0): every 5 s the run log shows what each audio stage carried (AJM decodes,
// AudioOut2 ports and objects, the AudioOut devices), each with its peak level, to find where a sound is lost.
inline bool StatsWanted() {
	static const bool wanted = [] {
		// (ds-native-lean) Off unless KYTY_AUDIO_STATS=1: 3700 lines in a 3-minute log, a peak scan of every block
		// and a lock on every output.
		const char* value = std::getenv("KYTY_AUDIO_STATS");
		return value != nullptr && *value != '\0' && *value != '0';
	}();
	return wanted;
}
// Seconds on a steady clock (the stats' 5 s periods).
inline double StatsSeconds() {
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// The largest absolute sample (full scale 1.0) of `count` samples, every `stride`th one read.
// encoding: 0 = int16, 1 = int32, 2 = float.
inline float PcmPeak(const void* data, size_t count, int encoding, size_t stride) {
	if (data == nullptr || count == 0) return 0.0f;
	if (stride == 0) stride = 1;
	float peak = 0.0f;
	for (size_t i = 0; i < count; i += stride) {
		float v = 0.0f;
		switch (encoding) {
			case 0: v = static_cast<float>(static_cast<const int16_t*>(data)[i]) / 32768.0f; break;
			case 1: v = static_cast<float>(static_cast<const int32_t*>(data)[i]) / 2147483648.0f; break;
			default: v = static_cast<const float*>(data)[i]; break;
		}
		v = std::fabs(v);
		if (v > peak && std::isfinite(v)) peak = v;
	}
	return peak;
}

} // namespace Libs::Audio::AudioInternal

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_INTERNAL_H_
