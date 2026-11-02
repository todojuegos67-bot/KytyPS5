// (ds-pc-port50) The main ports' latency cap (src/libs/audio_latency.h) in a lab model of Audio::QueueSdlAudio's queue:
// a guest submitting 256-frame grains at 48 kHz on its own clock (the asynchronous path: nothing waits on the device),
// a host device that plays slightly slower (clock drift, a resampler's rounding) and stalls now and then (a hitch, a
// load, a device switch). Without the cap the delay only grows; with it, it stays at the target.
#include "libs/audio_latency.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Latency = Libs::Audio::Latency;

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AudioLatencyTests: failed: %s\n", text);
		std::exit(1);
	}
}

struct Result {
	double   max_ms = 0, end_ms = 0, mean_last_min_ms = 0;
	uint64_t shortened = 0, clears = 0;
};

// `drain_ratio`: the device's playback rate over the guest's (1: none). Stalls: `stall_s` without playing every
// `stall_every_s` (0: none). `cap_ms` 0: as before (no cap). Steps of 1 ms.
Result Run(double minutes, double drain_ratio, double stall_every_s, double stall_s, uint64_t cap_ms, uint64_t reserve_us = 0) {
	constexpr double   rate      = 48000.0;
	constexpr uint32_t grain     = 256;
	const double       grain_us  = grain * 1e6 / rate;
	double             queued    = 0; // frames in the device queue
	double             next_push = 0;
	Result             r;
	double             sum = 0;
	uint64_t           n   = 0;
	const double       end_us = minutes * 60e6;
	for (double t = 0; t < end_us; t += 1000.0) {
		const bool stalled = stall_every_s > 0 && std::fmod(t / 1e6, stall_every_s) > stall_every_s - stall_s;
		if (!stalled) queued = std::max(0.0, queued - rate * drain_ratio / 1000.0);
		while (next_push <= t) {
			next_push += grain_us;
			const auto queued_us = static_cast<uint64_t>(queued * 1e6 / rate);
			const auto target_us = cap_ms == 0 ? 0 : std::max<uint64_t>(cap_ms * 1000u, 100000u + reserve_us);
			const auto d         = Latency::Decide(queued_us, target_us, grain);
			if (d.action == Latency::Action::Clear) queued = 0, ++r.clears;
			if (d.action == Latency::Action::Shorten) ++r.shortened;
			queued += d.keep_frames;
		}
		const double ms = queued * 1000.0 / rate;
		r.max_ms        = std::max(r.max_ms, ms);
		if (t > end_us - 60e6) sum += ms, ++n;
		r.end_ms = ms;
	}
	r.mean_last_min_ms = n ? sum / n : 0;
	return r;
}

void Print(const char* name, const Result& r) {
	std::printf("  %-44s max %8.1f ms, last minute mean %8.1f ms, end %8.1f ms; grains shortened %llu, restarts %llu\n", name,
	            r.max_ms, r.mean_last_min_ms, r.end_ms, static_cast<unsigned long long>(r.shortened),
	            static_cast<unsigned long long>(r.clears));
}

void TestDecide() {
	Check(Latency::Decide(50000, 100000, 256).action == Latency::Action::Keep, "under the target: kept");
	Check(Latency::Decide(100000, 100000, 256).action == Latency::Action::Keep, "at the target: kept");
	const auto s = Latency::Decide(120000, 100000, 256);
	Check(s.action == Latency::Action::Shorten && s.keep_frames == 251, "over the target: 2% shorter");
	Check(Latency::Decide(400000, 100000, 256).action == Latency::Action::Clear, "far over: restarted");
	Check(Latency::Decide(9000000, 0, 256).action == Latency::Action::Keep, "cap 0: as before");
}

void TestShortenIsSmooth() {
	// A 1 kHz sine, stereo: the shortened grain is the same waveform 2% faster, no step at its edges.
	constexpr uint32_t frames = 256, keep = 251, ch = 2;
	std::vector<float> in(frames * ch), out(keep * ch);
	for (uint32_t i = 0; i < frames; i++)
		for (uint32_t c = 0; c < ch; c++) in[i * ch + c] = static_cast<float>(std::sin(2 * 3.14159265358979 * 1000.0 * i / 48000.0));
	Latency::Shorten(in.data(), frames, ch, out.data(), keep);
	Check(out[0] == in[0] && out[(keep - 1) * ch] == in[(frames - 1) * ch], "first and last frames kept");
	double max_step = 0, max_step_in = 0;
	for (uint32_t i = 1; i < keep; i++) max_step = std::max(max_step, std::fabs(double(out[i * ch]) - out[(i - 1) * ch]));
	for (uint32_t i = 1; i < frames; i++) max_step_in = std::max(max_step_in, std::fabs(double(in[i * ch]) - in[(i - 1) * ch]));
	std::printf("  shorten: 1 kHz sine, largest sample step %.4f (input %.4f)\n", max_step, max_step_in);
	Check(max_step < max_step_in * 1.03, "no discontinuity");
	std::vector<int16_t> in16(frames), out16(keep);
	for (uint32_t i = 0; i < frames; i++) in16[i] = static_cast<int16_t>(i * 100);
	Latency::Shorten(in16.data(), frames, 1, out16.data(), keep);
	Check(out16[0] == 0 && out16[keep - 1] == 25500, "s16 ramp ends kept");
}

void TestLab() {
	std::printf("lab: 30 min, guest 48 kHz grains of 256 on its own clock\n");
	// The drift seen in a real log (lowest queue 1 -> 12 grains over 13 min) is ~0.008%; the user's 6-10 s needs more:
	// a 0.34% resampler surplus (48 -> 44.1 kHz, 235.2 frames rounded to 236), or device stalls of 1 s every 5 min.
	const Result drift_before = Run(30, 1.0 - 0.00008, 0, 0, 0);
	const Result drift_after  = Run(30, 1.0 - 0.00008, 0, 0, 100);
	const Result worst_before = Run(30, 1.0 - 0.0034, 0, 0, 0);
	const Result worst_after  = Run(30, 1.0 - 0.0034, 0, 0, 100);
	const Result stall_before = Run(30, 1.0, 300, 1, 0);
	const Result stall_after  = Run(30, 1.0, 300, 1, 100);
	const Result normal       = Run(30, 1.0, 0, 0, 100);
	Print("drift 0.008%, no cap (before)", drift_before);
	Print("drift 0.008%, cap 100 ms (after)", drift_after);
	Print("surplus 0.34%, before", worst_before);
	Print("surplus 0.34%, after", worst_after);
	Print("1 s stalls / 5 min, before", stall_before);
	Print("1 s stalls / 5 min, after", stall_after);
	Print("matched clocks, no stall, cap (normal play)", normal);
	Check(worst_before.end_ms > 6000.0, "the model reproduces a delay over 6 s");
	Check(stall_before.end_ms > 5000.0, "stalls alone pile up");
	Check(worst_after.mean_last_min_ms < 110.0 && worst_after.max_ms < 110.0, "capped: at the target");
	Check(stall_after.mean_last_min_ms < 110.0, "capped after stalls");
	Check(drift_after.max_ms < 110.0, "capped drift");
	Check(normal.shortened == 0 && normal.clears == 0, "normal play untouched");
}

} // namespace

int main() {
	TestDecide();
	TestShortenIsSmooth();
	TestLab();
	std::printf("AudioLatencyTests: ok\n");
	return 0;
}
