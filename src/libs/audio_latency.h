#ifndef EMULATOR_SRC_LIBS_AUDIO_LATENCY_H_
#define EMULATOR_SRC_LIBS_AUDIO_LATENCY_H_
// (ds-pc-port50) A main port's latency cap (audio.cpp, Audio::QueueSdlAudio). The SDL queue had no limit on the
// asynchronous path (an AudioOut2 push with blocking 0 is paced by the modelled queue on the process clock, not by the
// device) nor after the host device stalled: whatever the guest submitted beyond what the device played stayed queued,
// and the sound fell behind the picture for good. Above `target_us` queued, each grain is played 2% shorter (resampled,
// linear: a pitch change of ~0.3 semitone while it lasts, the excess paid back at ~1 ms per 50 ms); above
// target_us + 250 ms (a stall, a load) the queue is cleared and starts again from this grain.
// KYTY_AUDIO_MAX_LATENCY_MS: the target (default 100, 0: no cap, as before).
#include <algorithm>
#include <cstdint>

namespace Libs::Audio::Latency {

enum class Action { Keep, Shorten, Clear };

constexpr uint64_t CLEAR_ABOVE_US = 250000;

struct Decision {
	Action   action;
	uint32_t keep_frames; // the frames of the grain to queue
};

inline Decision Decide(uint64_t queued_us, uint64_t target_us, uint32_t frames) {
	if (target_us == 0 || frames == 0) return {Action::Keep, frames};
	if (queued_us > target_us + CLEAR_ABOVE_US) return {Action::Clear, frames};
	if (queued_us > target_us) return {Action::Shorten, frames - std::max<uint32_t>(1u, frames / 50u)};
	return {Action::Keep, frames};
}

// `frames` interleaved frames of `channels` resampled into `keep` frames (linear, first and last frames kept).
template <typename T>
void Shorten(const T* in, uint32_t frames, uint32_t channels, T* out, uint32_t keep) {
	if (keep == 0 || frames == 0) return;
	if (keep == 1 || frames == 1) {
		for (uint32_t c = 0; c < channels; c++) out[c] = in[c];
		return;
	}
	const double step = static_cast<double>(frames - 1) / static_cast<double>(keep - 1);
	for (uint32_t i = 0; i < keep; i++) {
		const double   pos  = step * i;
		const uint32_t a    = std::min(static_cast<uint32_t>(pos), frames - 1);
		const uint32_t b    = std::min(a + 1, frames - 1);
		const double   frac = pos - a;
		for (uint32_t c = 0; c < channels; c++) {
			const double v = static_cast<double>(in[a * channels + c]) * (1.0 - frac) +
			                 static_cast<double>(in[b * channels + c]) * frac;
			out[i * channels + c] = static_cast<T>(v);
		}
	}
}

} // namespace Libs::Audio::Latency

#endif // EMULATOR_SRC_LIBS_AUDIO_LATENCY_H_
