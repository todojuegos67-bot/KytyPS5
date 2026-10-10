#pragma once
// Local measurement hook (KYTY_LIVE_FILE): one boot, many same-process A/B experiments.
// A side thread polls the command file every 100 ms and runs its commands once per new
// "id N" first line, printing LIVE_* lines to stdout (tools/local/live-bench.py).
//   poke32 <host address> <hex>   write a switch (live-bench resolves `sym NAME VALUE`)
//   peek <host address> <bytes>   hex dump (<= 256 bytes)
//   measure <seconds> <label>     presented frames, fps, per-frame CPU of the render and
//                                 recording threads
//   prof <seconds> <path>         4 kHz samples of render-thread CPU time
//   profw <seconds> <path>        4 kHz samples of render-thread wall time (waits included)
//   profp <seconds> <path>        4 kHz samples of process CPU time (with thread ids)
//   proft <tid> <seconds> <path>  Windows: samples of another thread (e.g. the guest main thread)
//   pinthread <tid> <cpus>        Windows: set any thread's affinity
// A sample is 16 words: pc, the word at rsp, then up to 14 frame-pointer return addresses
// (process mode: pc, thread id | 1 << 63, zeros).
//   timecensus <seconds>          time-related HLE calls per guest caller (time-census.h)
//   sleep <seconds>
// Without KYTY_LIVE_FILE nothing runs; the flip hook is one relaxed increment.

#include "common/assert.h"
#include "frame-capture.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace.h"
#include "loader/demonsSoulsWarp.h"
#include "local-platform.h"
#include "slow-log.h"
#include "time-census.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <ctime>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#include <pthread.h>
#include <ucontext.h>
#endif

#if defined(KYTY_PGO_GENERATE)
// compiler-rt profile runtime (instrumented builds only).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

namespace Loader {
std::string DescribeGuestAddressForDiagnostics(uint64_t vaddr);
} // namespace Loader

namespace LiveControl {

inline std::atomic_uint64_t g_flips {0};
// The steady-clock time (ns) of flip n at [n % FlipTimes], for frame-time percentiles (Measure).
constexpr size_t             FlipTimes = 8192;
inline std::atomic<int64_t>  g_flip_times[FlipTimes] {};
inline std::atomic_bool     g_render_known {false};
// pthread_t on Linux, a thread HANDLE on Windows (LocalPlatform::CurrentThreadHandle).
inline uint64_t              g_render_thread {};
inline std::atomic<uint32_t> g_render_tid {0};
// Render-thread stack bounds for the frame-pointer walk.
inline uint64_t             g_stack_low = 0, g_stack_high = 0;
constexpr size_t             SampleWords = 16;
inline uint64_t              g_samples[1u << 23];
inline uint64_t              g_sample_tsc[(1u << 23) / SampleWords]; // Windows: each sample's TSC
inline std::atomic<uint32_t> g_sample_count {0};
inline std::atomic_bool      g_process {false};

// The render thread's time waiting for guest work (GuestGpu::ThreadRun), in steady-clock ns.
inline int64_t g_render_idle_ns = 0;
// When the render thread began waiting for the guest's next submission (steady-clock ns; 0 while it has one): its
// idle waits set it once, a submission taken clears it (guest commands do not: a guest thread may wait on them).
inline std::atomic<int64_t> g_render_idle_since {0};
inline void NoteRenderIdle(int64_t now_ns) {
	int64_t expected = 0;
	(void)g_render_idle_since.compare_exchange_strong(expected, now_ns, std::memory_order_relaxed);
}

// Render thread. With KYTY_HITCH_LOG_MS (or KYTY_SLOW_LOG_MS), a frame (flip to flip) that long and
// at least 20 ms is a SLOW line too, on the same TSC timeline: the hitch the slow calls before it explain.
// The line also has the frame's render-thread wait for guest work, what some counters did in it and its
// wall-clock time.
inline void Flip() {
	const auto flip = g_flips.load(std::memory_order_relaxed);
	g_flip_times[flip % FlipTimes].store(std::chrono::steady_clock::now().time_since_epoch().count(),
	                                     std::memory_order_relaxed);
	g_flips.store(flip + 1, std::memory_order_release);
	if (SlowLog::HitchThreshold() > 0.0) {
		using C = LiveCounters::Id;
		static constexpr std::array counted {C::XprStores, C::XprStored, C::XprStorePending, C::XprRefuseBind,
		                                     C::XprMissRefused, C::XprMissValidate, C::XprMissUnseen, C::XprMissBudget,
		                                     C::UploadBytes, C::AsyncImageBytes, C::FullUploadBytes,
		                                     C::BufferRegistrations, C::RegionSyncs, C::SyncDownloads, C::AsyncReadbacks,
		                                     C::ReadbackParts, C::ReadbackRegions, C::GuestCommands, C::TextureUnmaps, C::AsyncPipelines,
		                                     C::ImageInitUs, C::BufferSyncUs, C::DrawUs, C::TextureUnmapUs, C::BackingReadUs, C::RegionSyncUs, C::AsyncDrawWaitUs, C::AsyncDrawSkips, C::QueueLockUs,
		                                     C::TranslateUs, C::Translates, C::ComputePipelineUs, C::ComputePipelines,
		                                     C::GraphicsPipelineUs, C::GraphicsPipelines, C::FlipSlotWaitUs,
		                                     C::TableDraws, C::TableStores, C::TableStoreVariant, C::TableStoreTargets,
		                                     C::TableRefusedSets, C::TableNoSet, C::TableDrawNative, C::TableEvaluations,
		                                     C::TableStoreDeferred};
		static constexpr const char* waits[] = {"gpu_wait", "readback_wait", "download_wait", "compile", "record_wait"};
		static std::chrono::steady_clock::time_point     last {};
		static uint64_t                                  last_tsc = 0;
		static std::array<uint64_t, LiveCensus::Kinds>   last_kind_cycles {};
		static std::array<uint64_t, LiveCensus::Phases>  last_dispatch_phases {}, last_draw_phases {};
		static std::array<uint64_t, 4>                   last_xpr {};
		static int64_t                                   last_idle = 0;
		static int64_t                                   last_upload_wait = 0;
		static std::array<int64_t, LiveCensus::Waits>    last_waits {};
		static std::array<uint64_t, counted.size()>      last_counts {};
		// The render thread's run time (TSC ticks) and a TSC rate from the first flip on: busy wall time
		// (the frame less idle) well over cpu is time the thread was ready but not running (preempted).
		static const uint64_t                            first_tsc  = __rdtsc();
		static const auto                                first_time = std::chrono::steady_clock::now();
		static uint64_t                                  last_cycles = 0;
		const uint64_t                                   cycles      = LocalPlatform::CurrentThreadCycles();
		const auto                                       now = std::chrono::steady_clock::now();
		const uint64_t                                   tsc = __rdtsc();
		const double ms = std::chrono::duration<double, std::milli>(now - last).count();
		// KYTY_FPS_LOG=1: a line every second with the frames of that second (count, mean and longest), the render
		// thread's mean run time, idle and waits per frame, and how many frames passed 17 ms (missed 60 fps): what a
		// frame rate that fluctuates under 60 is made of, below the SLOW Frame threshold.
		static const bool fps_log = [] {
			const char* text = std::getenv("KYTY_FPS_LOG");
			return text != nullptr && *text != '\0' && std::string_view(text) != "0";
		}();
		if (fps_log && last != std::chrono::steady_clock::time_point {}) {
			struct Second {
				std::chrono::steady_clock::time_point start {};
				uint32_t frames = 0, missed = 0, slow = 0;
				double   sum_ms = 0, max_ms = 0, sum_cpu = 0, sum_idle = 0;
				std::array<double, LiveCensus::Waits> sum_waits {};
			};
			static Second second;
			if (second.start == std::chrono::steady_clock::time_point {}) second.start = now;
			const double since_first = std::chrono::duration<double, std::milli>(now - first_time).count();
			const double cpu_ms      = cycles != 0 && last_cycles != 0 && since_first > 1000.0
			                               ? static_cast<double>(cycles - last_cycles) * since_first /
			                                     static_cast<double>(__rdtsc() - first_tsc)
			                               : 0.0;
			second.frames++;
			second.sum_ms += ms;
			second.max_ms = std::max(second.max_ms, ms);
			second.sum_cpu += cpu_ms;
			second.sum_idle += static_cast<double>(g_render_idle_ns - last_idle) / 1e6;
			for (size_t i = 0; i < LiveCensus::Waits; ++i)
				second.sum_waits[i] += static_cast<double>(LiveCensus::g_waits_ns[i] - last_waits[i]) / 1e6;
			if (ms > 17.0) second.missed++;
			if (ms > 25.0) second.slow++;
			if (now - second.start >= std::chrono::seconds(1)) {
				const double n = static_cast<double>(second.frames);
				std::printf("[fps] %u frames: mean %.1f ms, max %.1f, cpu %.1f, idle %.1f, gpu_wait %.2f, readback %.2f, download %.2f, "
				            "compile %.2f, record %.2f; over 17 ms: %u, over 25 ms: %u\n",
				            second.frames, second.sum_ms / n, second.max_ms, second.sum_cpu / n, second.sum_idle / n,
				            second.sum_waits[0] / n, second.sum_waits[1] / n, second.sum_waits[2] / n, second.sum_waits[3] / n,
				            second.sum_waits[4] / n, second.missed, second.slow);
				// The guest threads' tax that second: their write and read faults, protection re-arms and
				// synchronous GPU reads (what the game's own thread spends outside its code).
				using G = LiveCounters::Id;
				static constexpr std::array guest {G::WindowFault, G::WriteFault, G::ReadFault, G::Reprotect, G::Unprotect,
				                                   G::ProtectCalls, G::SyncReadsGuest, G::SyncDownloads, G::GuestCommands,
				                                   G::BackingLockWaitUs, G::BackingWriteWaitUs, G::UploadWaitUs};
				static std::array<uint64_t, guest.size()> last_guest {};
				std::printf("[fps+]");
				for (size_t i = 0; i < guest.size(); ++i) {
					const auto value = LiveCounters::Value(guest[i]);
					std::printf(" %s=%llu", LiveCounters::Names[guest[i]], static_cast<unsigned long long>(value - last_guest[i]));
					last_guest[i] = value;
				}
				std::printf("\n");
				std::fflush(stdout);
				second       = {};
				second.start = now;
			}
		}
		if (last != std::chrono::steady_clock::time_point {} && ms >= std::max(20.0, SlowLog::HitchThreshold())) {
			std::printf("[tsc %llu] SLOW Frame %.1f ms idle=%.1f", static_cast<unsigned long long>(tsc), ms,
			            static_cast<double>(g_render_idle_ns - last_idle) / 1e6);
			if (const double since = std::chrono::duration<double, std::milli>(now - first_time).count();
			    cycles != 0 && last_cycles != 0 && since > 1000.0)
				std::printf(" cpu=%.1f", static_cast<double>(cycles - last_cycles) * since /
				                             static_cast<double>(__rdtsc() - first_tsc));
			// The render thread's time by call kind (ms), from the frame's own TSC rate.
			const double ms_per_cycle = tsc > last_tsc ? ms / static_cast<double>(tsc - last_tsc) : 0.0;
			for (size_t i = 0; i < LiveCensus::Kinds; ++i) {
				const double kind_ms = static_cast<double>(LiveCensus::g_kind_cycles[i] - last_kind_cycles[i]) * ms_per_cycle;
				if (kind_ms >= 0.05) std::printf(" %s=%.1f", LiveCensus::KindNames[i], kind_ms);
			}
			for (size_t i = 0; i < LiveCensus::Phases; ++i) {
				const double phase_ms = static_cast<double>(LiveCensus::g_dispatch_phase_cycles[i] - last_dispatch_phases[i]) * ms_per_cycle;
				if (phase_ms >= 0.5) std::printf(" dispatch.%s=%.1f", LiveCensus::DispatchPhaseNames[i], phase_ms);
			}
			for (size_t i = 0; i < LiveCensus::Phases; ++i) {
				const double phase_ms = static_cast<double>(LiveCensus::g_draw_phase_cycles[i] - last_draw_phases[i]) * ms_per_cycle;
				if (phase_ms >= 0.5) std::printf(" draw.%s=%.1f", LiveCensus::DrawPhaseNames[i], phase_ms);
			}
			for (size_t i = 0; i < 4; ++i) {
				const double xpr_ms = static_cast<double>(LiveCensus::g_xpr_cycles[i] - last_xpr[i]) * ms_per_cycle;
				if (xpr_ms >= 0.5) std::printf(" xpr.%zu=%.1f", i, xpr_ms);
			}
			for (size_t i = 0; i < LiveCensus::Waits; ++i)
				if (const auto ns = LiveCensus::g_waits_ns[i] - last_waits[i]; ns != 0)
					std::printf(" %s=%.1f", waits[i], static_cast<double>(ns) / 1e6);
			if (const auto ns = LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed) - last_upload_wait; ns != 0)
				std::printf(" upload_wait=%.1f", static_cast<double>(ns) / 1e6);
			for (size_t i = 0; i < counted.size(); ++i) {
				const auto delta = LiveCounters::Value(counted[i]) - last_counts[i];
				if (delta != 0) std::printf(" %s=%llu", LiveCounters::Names[counted[i]], static_cast<unsigned long long>(delta));
			}
			// Wall-clock time, to find the frame in a screen capture.
			std::printf(" unix_ms=%lld\n", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
			                                   std::chrono::system_clock::now().time_since_epoch())
			                                   .count()));
			std::fflush(stdout);
		}
		last      = now;
		last_tsc  = tsc;
		last_idle = g_render_idle_ns;
		last_upload_wait = LiveCensus::g_upload_wait_ns.load(std::memory_order_relaxed);
		for (size_t i = 0; i < LiveCensus::Kinds; ++i) last_kind_cycles[i] = LiveCensus::g_kind_cycles[i];
		for (size_t i = 0; i < LiveCensus::Phases; ++i) {
			last_dispatch_phases[i] = LiveCensus::g_dispatch_phase_cycles[i];
			last_draw_phases[i]     = LiveCensus::g_draw_phase_cycles[i];
		}
		for (size_t i = 0; i < 4; ++i) last_xpr[i] = LiveCensus::g_xpr_cycles[i];
		last_cycles = cycles;
		for (size_t i = 0; i < LiveCensus::Waits; ++i) last_waits[i] = LiveCensus::g_waits_ns[i];
		for (size_t i = 0; i < counted.size(); ++i) last_counts[i] = LiveCounters::Value(counted[i]);
	}
}

inline double ThreadCpuSeconds(uint64_t thread) {
	return LocalPlatform::ThreadCpuSeconds(thread);
}

inline double NamedThreadsCpuSeconds(const char* name) {
	return LocalPlatform::NamedThreadsCpuSeconds(name);
}

#if !defined(_WIN32)
inline void ProfSignal(int /*signal*/, siginfo_t* /*info*/, void* context) {
	const auto* uc  = static_cast<const ucontext_t*>(context);
	const auto  pc  = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
	const auto  rsp = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
	const auto  idx = g_sample_count.fetch_add(1, std::memory_order_relaxed);
	if (SampleWords * (size_t(idx) + 1) <= std::size(g_samples)) {
		auto* record = g_samples + SampleWords * size_t(idx);
		record[0]    = pc;
		if (g_process.load(std::memory_order_relaxed)) {
			record[1] = static_cast<uint64_t>(LocalPlatform::ThreadId()) | (1ull << 63);
			for (size_t i = 2; i < SampleWords; ++i) record[i] = 0;
		} else {
			// Word at rsp (the caller of a leaf routine), then return addresses from the
			// frame-pointer chain, bounded to this stack.
			record[1]    = *reinterpret_cast<const uint64_t*>(rsp);
			auto rbp     = static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RBP]);
			for (size_t i = 2; i < SampleWords; ++i) {
				record[i] = 0;
				if (rbp < rsp || rbp >= g_stack_high || g_stack_high - rbp < 16 || (rbp & 7u) != 0) continue;
				const auto* frame = reinterpret_cast<const uint64_t*>(rbp);
				record[i]         = frame[1];
				if (frame[0] <= rbp) {
					rbp = 0;
					continue;
				}
				rbp = frame[0];
			}
		}
	}
}

inline void Profile(uint64_t id, double seconds, const char* path, bool process, bool wall = false) {
	if (g_render_tid.load() == 0) return;
	g_process.store(process);
	struct sigaction action {};
	action.sa_sigaction = ProfSignal;
	action.sa_flags     = SA_SIGINFO | SA_RESTART;
	sigemptyset(&action.sa_mask);
	sigaction(SIGPROF, &action, nullptr);
	clockid_t clock = CLOCK_PROCESS_CPUTIME_ID;
	if (wall) {
		clock = CLOCK_MONOTONIC;
	} else if (!process && pthread_getcpuclockid(static_cast<pthread_t>(g_render_thread), &clock) != 0) {
		return;
	}
	sigevent event {};
	event.sigev_signo = SIGPROF;
	if (process) {
		event.sigev_notify = SIGEV_SIGNAL;
	} else {
		event.sigev_notify   = SIGEV_THREAD_ID;
		event._sigev_un._tid = g_render_tid.load();
	}
	timer_t timer {};
	if (timer_create(clock, &event, &timer) != 0) return;
	g_sample_count.store(0);
	itimerspec spec {};
	spec.it_interval.tv_nsec = 1000000000L / 4000;
	spec.it_value            = spec.it_interval;
	timer_settime(timer, 0, &spec, nullptr);
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	timer_delete(timer);
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	const auto count = std::min<uint32_t>(g_sample_count.load(), std::size(g_samples) / SampleWords);
	if (auto* out = std::fopen(path, "wb")) {
		std::fwrite(g_samples, sizeof(uint64_t), SampleWords * size_t(count), out);
		std::fclose(out);
	}
	const std::string maps_path = std::string(path) + ".maps";
	if (auto* in = std::fopen("/proc/self/maps", "re")) {
		if (auto* out = std::fopen(maps_path.c_str(), "we")) {
			char buffer[65536];
			for (size_t got; (got = std::fread(buffer, 1, sizeof(buffer), in)) > 0;) std::fwrite(buffer, 1, got, out);
			std::fclose(out);
		}
		std::fclose(in);
	}
	std::printf("LIVE_PROF id=%" PRIu64 " samples=%u path=%s\n", id, count, path);
}
#else
// Windows: no per-thread CPU-time timers; prof and profw both sample the render thread at
// 4 kHz of wall time by suspending it. Process-wide sampling (profp) is not available.
// proft <tid>: another thread (a guest thread on its own stack) instead of the render thread.
inline void Profile(uint64_t id, double seconds, const char* path, bool process, bool /*wall*/ = false,
                    uint32_t tid = 0) {
	if (g_render_tid.load() == 0 || g_render_thread == 0) return;
	if (process) {
		std::printf("LIVE_ERROR id=%" PRIu64 " line=profp (not supported on Windows)\n", id);
		return;
	}
	uint64_t thread = g_render_thread, stack_low = g_stack_low, stack_high = g_stack_high;
	if (tid != 0) {
		thread    = LocalPlatform::OpenThreadForSampling(tid);
		stack_low = stack_high = 0;
		if (thread == 0) {
			std::printf("LIVE_ERROR id=%" PRIu64 " line=proft %u (cannot open the thread)\n", id, tid);
			return;
		}
	}
	g_sample_count.store(0);
	LocalPlatform::PrepareSampling();
	// KYTY_PROF_HZ (default 4000): fewer suspensions distort lock and page-fault timing less.
	static const long hz = [] {
		const char* value = std::getenv("KYTY_PROF_HZ");
		const long  rate  = value != nullptr ? std::strtol(value, nullptr, 10) : 4000;
		return rate >= 100 && rate <= 20000 ? rate : 4000;
	}();
	const auto period = std::chrono::nanoseconds(1000000000L / hz);
	const auto begin  = std::chrono::steady_clock::now();
	const auto end    = begin + std::chrono::duration<double>(seconds);
	const auto tsc0   = __rdtsc();
	auto       next   = begin;
	while (next < end) {
		const auto idx = g_sample_count.load(std::memory_order_relaxed);
		if (SampleWords * (size_t(idx) + 1) > std::size(g_samples)) break;
		if (LocalPlatform::SampleThread(thread, stack_low, stack_high, g_samples + SampleWords * size_t(idx),
		                                SampleWords)) {
			g_sample_tsc[idx] = __rdtsc();
			g_sample_count.store(idx + 1, std::memory_order_relaxed);
		}
		next += period;
		while (std::chrono::steady_clock::now() < next) _mm_pause();
	}
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const double tsc_hz  = static_cast<double>(__rdtsc() - tsc0) / elapsed;
	if (tid != 0) LocalPlatform::CloseThreadForSampling(thread);
	const auto count = std::min<uint32_t>(g_sample_count.load(), std::size(g_samples) / SampleWords);
	if (auto* out = std::fopen(path, "wb")) {
		std::fwrite(g_samples, sizeof(uint64_t), SampleWords * size_t(count), out);
		std::fclose(out);
	}
	// <path>.tsc: each sample's TSC (slow-frame lines carry TSC stamps: hitch-prof.py picks their samples).
	if (auto* out = std::fopen((std::string(path) + ".tsc").c_str(), "wb")) {
		std::fwrite(g_sample_tsc, sizeof(uint64_t), size_t(count), out);
		std::fclose(out);
	}
	std::printf("LIVE_PROF id=%" PRIu64 " samples=%u path=%s tsc_hz=%.0f\n", id, count, path, tsc_hz);
}
#endif

inline void Measure(uint64_t id, double seconds, const char* label) {
	std::array<uint64_t, LiveCounters::Count> counters0 {};
	for (size_t i = 0; i < counters0.size(); ++i) counters0[i] = LiveCounters::Value(i);
	const auto   flips0  = g_flips.load();
	const double render0 = g_render_known ? ThreadCpuSeconds(g_render_thread) : 0;
	const double record0 = NamedThreadsCpuSeconds("Kyty.Record");
	const auto   begin   = std::chrono::steady_clock::now();
	std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
	const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
	const auto   frames  = g_flips.load() - flips0;
	const double render  = (g_render_known ? ThreadCpuSeconds(g_render_thread) : 0) - render0;
	const double record  = NamedThreadsCpuSeconds("Kyty.Record") - record0;
	const double per     = frames ? 1000.0 / static_cast<double>(frames) : 0;
	// Frame times (flip to flip) of the window: the worst 1% as fps (1% low), the 99th percentile, the
	// longest and the frames of 50 ms or more and of 100 ms or more.
	std::vector<double> times;
	for (auto flip = flips0 + 1; flip < flips0 + frames && flip + FlipTimes > flips0 + frames; ++flip)
		times.push_back(static_cast<double>(g_flip_times[flip % FlipTimes].load(std::memory_order_relaxed) -
		                                    g_flip_times[(flip - 1) % FlipTimes].load(std::memory_order_relaxed)) / 1e6);
	std::sort(times.begin(), times.end(), std::greater<>());
	const size_t worst = (times.size() + 99) / 100;
	double       sum   = 0;
	for (size_t i = 0; i < worst; ++i) sum += times[i];
	const double low1 = worst ? 1000.0 * static_cast<double>(worst) / sum : 0;
	const double p99  = times.empty() ? 0 : times[std::min(times.size() - 1, times.size() / 100)];
	const auto   over = [&](double ms) { return std::ranges::count_if(times, [&](double t) { return t >= ms; }); };
	std::printf("LIVE_MEASURE id=%" PRIu64 " label=%s seconds=%.2f frames=%" PRIu64
	            " fps=%.2f render_ms=%.2f record_ms=%.2f render_busy=%.1f%% low1=%.2f p99_ms=%.1f max_ms=%.1f"
	            " over50=%lld over100=%lld\n",
	            id, label, elapsed, frames, static_cast<double>(frames) / elapsed, render * per, record * per,
	            100.0 * render / elapsed, low1, p99, times.empty() ? 0 : times.front(),
	            static_cast<long long>(over(50)), static_cast<long long>(over(100)));
	std::string counters;
	for (size_t i = 0; i < counters0.size(); ++i) {
		char text[96];
		std::snprintf(text, sizeof(text), " %s=%.1f", LiveCounters::Names[i],
		              static_cast<double>(LiveCounters::Value(i) - counters0[i]) / (frames ? frames : 1));
		counters += text;
	}
	std::printf("LIVE_COUNTERS id=%" PRIu64 " label=%s%s\n", id, label, counters.c_str());
}

inline void Run(uint64_t id, const std::string& line) {
	char      command[16] {}, arg1[256] {}, arg2[256] {};
	const int n = std::sscanf(line.c_str(), "%15s %255s %255s", command, arg1, arg2);
	if (n < 1) return;
	const std::string_view cmd = command;
	if ((cmd == "poke32" || cmd == "peek") && n == 3 && std::strtoull(arg1, nullptr, 16) < 0x100000000ull) {
		// Host addresses of switches are far above 4 GiB; refuse guest-looking values.
		std::printf("LIVE_ERROR id=%" PRIu64 " line=%s (address below 4 GiB)\n", id, line.c_str());
	} else if (cmd == "poke32" && n == 3) {
		auto*      target = reinterpret_cast<void*>(std::strtoull(arg1, nullptr, 16));
		const auto value  = static_cast<uint32_t>(std::strtoul(arg2, nullptr, 16));
		std::memcpy(target, &value, 4);
	} else if (cmd == "peek" && n == 3) {
		const auto* data  = reinterpret_cast<const uint8_t*>(std::strtoull(arg1, nullptr, 16));
		const auto  bytes = std::min<size_t>(std::strtoul(arg2, nullptr, 0), 256);
		std::string hex;
		for (size_t i = 0; i < bytes; ++i) {
			char b[4];
			std::snprintf(b, sizeof(b), "%02x", data[i]);
			hex += b;
		}
		std::printf("LIVE_PEEK id=%" PRIu64 " va=%s bytes=%s\n", id, arg1, hex.c_str());
	} else if (cmd == "measure" && n >= 2) {
		Measure(id, std::strtod(arg1, nullptr), n == 3 ? arg2 : "-");
	} else if (cmd == "flips" && n == 3) {
		// flips <count> <path>: the last flips' intervals (us), oldest first, one a line (periodic slow frames).
		const auto last  = g_flips.load();
		const auto count = std::min<uint64_t>({std::strtoull(arg1, nullptr, 10), FlipTimes - 1, last ? last - 1 : 0});
		if (auto* out = std::fopen(arg2, "w")) {
			for (auto flip = last - count; flip < last; ++flip)
				std::fprintf(out, "%lld\n", static_cast<long long>((g_flip_times[flip % FlipTimes].load() -
				                                                    g_flip_times[(flip - 1) % FlipTimes].load()) / 1000));
			std::fclose(out);
		}
		std::printf("LIVE_FLIPS id=%" PRIu64 " count=%llu path=%s\n", id, static_cast<unsigned long long>(count), arg2);
	} else if ((cmd == "prof" || cmd == "profp" || cmd == "profw") && n == 3) {
		Profile(id, std::strtod(arg1, nullptr), arg2, cmd == "profp", cmd == "profw");
#if defined(_WIN32)
	} else if (cmd == "pinthread" && n == 3) {
		// pinthread <tid> <cpus>: any thread's affinity (placement A/B of the worker threads).
		const auto handle = LocalPlatform::OpenThreadForSampling(static_cast<uint32_t>(std::strtoul(arg1, nullptr, 10)));
		const bool ok     = handle != 0 && LocalPlatform::PinThreadHandleToCpuList(handle, arg2);
		LocalPlatform::CloseThreadForSampling(handle);
		std::printf("LIVE_PIN id=%" PRIu64 " thread=%s cpus=%s ok=%d\n", id, arg1, arg2, ok ? 1 : 0);
	} else if (cmd == "proft" && n == 3) {
		// proft <tid> <seconds> <path>: sample that thread (the file name follows the seconds).
		char path[256] {};
		if (std::sscanf(line.c_str(), "%*s %*s %*s %255s", path) == 1)
			Profile(id, std::strtod(arg2, nullptr), path, false, false, static_cast<uint32_t>(std::strtoul(arg1, nullptr, 10)));
#endif
	} else if ((cmd == "trace" || cmd == "tracew" || cmd == "tracem") && n == 3) {
		LiveTrace::g_count.store(0);
		LiveTrace::g_writes_on.store(cmd == "tracew");
		LiveTrace::g_mark_next.store(0);
		LiveTrace::g_marks_on.store(cmd == "tracem");
		const auto tsc0   = __rdtsc();
		const auto clock0 = std::chrono::steady_clock::now();
		LiveTrace::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveTrace::g_on.store(false);
		LiveTrace::g_writes_on.store(false);
		LiveTrace::g_marks_on.store(false);
		const auto tsc1   = __rdtsc();
		const auto clock1 = std::chrono::steady_clock::now();
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		if (cmd == "tracem" && LiveTrace::g_read_marks)
			LiveTrace::g_read_marks(std::min(LiveTrace::g_mark_next.load(), LiveTrace::MarkSlots));
		LiveTrace::Dump(arg2);
		const double seconds = std::chrono::duration<double>(clock1 - clock0).count();
		std::printf("LIVE_TRACE id=%" PRIu64 " records=%" PRIu64 " tsc_hz=%.0f path=%s\n", id,
		            std::min<uint64_t>(LiveTrace::g_count.load(), LiveTrace::Capacity),
		            static_cast<double>(tsc1 - tsc0) / seconds, arg2);
	} else if (cmd == "census" && n >= 2) {
		// census <seconds>: render-thread time per call kind and shader over the window.
		LiveCensus::g_on.store(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		for (auto& entry: LiveCensus::g_table) entry = {};
		const auto flips0 = g_flips.load();
		const auto tsc0   = __rdtsc();
		const auto t0     = std::chrono::steady_clock::now();
		LiveCensus::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveCensus::g_on.store(false);
		const auto   frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		const double ns_per_cycle =
		    std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() /
		    static_cast<double>(__rdtsc() - tsc0);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
#if defined(_WIN32)
		uint64_t base = 0;
		char     exe[512] {};
		if (!LocalPlatform::ModuleOf(reinterpret_cast<const void*>(&Flip), &base, exe, sizeof(exe))) exe[0] = '\0';
		std::printf("LIVE_CENSUS id=%" PRIu64 " frames=%" PRIu64 " base=%016" PRIx64 " exe=%s\n", id, frames, base,
		            exe[0] != '\0' ? exe : "?");
#else
		Dl_info self {};
		dladdr(reinterpret_cast<void*>(&Flip), &self);
		std::printf("LIVE_CENSUS id=%" PRIu64 " frames=%" PRIu64 " base=%016" PRIx64 " exe=%s\n", id, frames,
		            reinterpret_cast<uint64_t>(self.dli_fbase), self.dli_fname ? self.dli_fname : "?");
#endif
		for (const auto& entry: LiveCensus::g_table) {
			if (!entry.used) continue;
			std::printf("LIVE_CENSUS_ENTRY id=%" PRIu64 " kind=%u a=%016" PRIx64 " b=%016" PRIx64
			            " calls_per_frame=%.2f ms_per_frame=%.4f\n",
			            id, entry.kind, entry.a, entry.b, static_cast<double>(entry.calls) / frames,
			            static_cast<double>(entry.cycles) * ns_per_cycle / 1e6 / frames);
		}
	} else if (cmd == "timecensus" && n >= 2) {
		// timecensus <seconds>: time-related HLE calls per caller (guest return address).
		TimeCensus::g_on.store(false);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		for (auto& entry: TimeCensus::g_table) {
			entry.key.store(0);
			entry.calls.store(0);
			entry.arg_sum.store(0);
			entry.arg_last.store(0);
		}
		TimeCensus::g_overflow.store(0);
		const auto flips0   = g_flips.load();
		const auto vblanks0 = LiveCounters::Value(LiveCounters::Vblanks);
		const auto t0       = std::chrono::steady_clock::now();
		TimeCensus::g_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		TimeCensus::g_on.store(false);
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		const auto   frames  = g_flips.load() - flips0;
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		const auto   vblanks = LiveCounters::Value(LiveCounters::Vblanks) - vblanks0;
		std::printf("LIVE_TIMECENSUS id=%" PRIu64 " seconds=%.2f frames=%" PRIu64 " vblanks=%" PRIu64
		            " overflow=%" PRIu64 "\n",
		            id, seconds, frames, vblanks, TimeCensus::g_overflow.load());
		for (const auto& entry: TimeCensus::g_table) {
			const auto key = entry.key.load();
			if (key == 0) continue;
			const auto calls = entry.calls.load();
			std::printf("LIVE_TIMECENSUS_ENTRY id=%" PRIu64 " source=%u caller=%012" PRIx64
			            " per_s=%.1f per_frame=%.2f arg_avg=%.1f arg_last=%" PRIu64 "\n",
			            id, static_cast<unsigned>(key & 0xffu), key >> 8u, static_cast<double>(calls) / seconds,
			            static_cast<double>(calls) / static_cast<double>(frames ? frames : 1),
			            calls ? static_cast<double>(entry.arg_sum.load()) / static_cast<double>(calls) : 0.0,
			            entry.arg_last.load());
		}
	} else if (cmd == "granules" && n >= 2) {
		// granules <seconds>: per 1 MiB granule, per frame: window pages, re-armed pages,
		// upload bytes and upload copies.
		for (auto& granule: LiveCounters::g_granules) {
			granule.tag.store(0);
			for (auto& value: granule.values) value.store(0);
		}
		const auto flips0 = g_flips.load();
		LiveCounters::g_granules_on.store(true);
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		LiveCounters::g_granules_on.store(false);
		const auto frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		for (const auto& granule: LiveCounters::g_granules) {
			const auto tag = granule.tag.load();
			if (tag == 0) continue;
			std::printf("LIVE_GRANULE id=%" PRIu64 " address=%09" PRIx64 " window_pages=%.1f reprotect_pages=%.1f"
			            " upload_kb=%.1f upload_calls=%.1f gpu_writes=%.2f gpu_write_kb=%.1f\n",
			            id, (tag - 1) << 20u, double(granule.values[0].load()) / frames,
			            double(granule.values[1].load()) / frames, double(granule.values[2].load()) / 1024.0 / frames,
			            double(granule.values[3].load()) / frames, double(granule.values[4].load()) / frames,
			            double(granule.values[5].load()) / 1024.0 / frames);
		}
		std::printf("LIVE_GRANULE_OVERFLOW id=%" PRIu64 " count=%" PRIu64 "\n", id, LiveCounters::g_granule_overflow.load());
	} else if (cmd == "pm4" && n >= 2) {
		// pm4 <seconds>: PM4 packets per frame by opcode.
		std::array<uint64_t, 256> before {};
		for (size_t i = 0; i < before.size(); ++i) before[i] = LiveCounters::g_pm4[i].load();
		const auto flips0 = g_flips.load();
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
		const auto frames = std::max<uint64_t>(g_flips.load() - flips0, 1);
		for (size_t i = 0; i < before.size(); ++i) {
			const auto count = LiveCounters::g_pm4[i].load() - before[i];
			if (count != 0)
				std::printf("LIVE_PM4 id=%" PRIu64 " opcode=0x%02zx per_frame=%.2f\n", id, i,
				            static_cast<double>(count) / static_cast<double>(frames));
		}
	} else if (cmd == "pin" && n >= 2) {
		// pin <cpus>: the render thread's affinity (for same-process placement A/B).
		const bool ok = LocalPlatform::PinThreadHandleToCpuList(g_render_thread, arg1);
		std::printf("LIVE_PIN id=%" PRIu64 " cpus=%s ok=%d\n", id, arg1, ok ? 1 : 0);
	} else if (cmd == "cpuset" && n >= 2) {
		// cpuset <cpus|all>: the default CPU set of threads without their own (Windows only).
		const bool ok = LocalPlatform::SetProcessDefaultCpuList(std::string_view(arg1) == "all" ? "" : arg1);
		std::printf("LIVE_CPUSET id=%" PRIu64 " cpus=%s ok=%d\n", id, arg1, ok ? 1 : 0);
#if defined(KYTY_PGO_GENERATE)
	} else if (cmd == "pgo" && n >= 2) {
		// pgo <file>: write the instrumentation profile now (the emulator exits with quick_exit).
		__llvm_profile_set_filename(arg1);
		const int result = __llvm_profile_write_file();
		std::printf("LIVE_PGO id=%" PRIu64 " path=%s result=%d\n", id, arg1, result);
#endif
	} else if (cmd == "vma" && n >= 2) {
		// vma <path>: the GPU allocator's detailed statistics (JSON: every allocation's type and size).
		if (LiveCounters::g_vma_report != nullptr) LiveCounters::g_vma_report(arg1);
		std::printf("LIVE_VMA id=%" PRIu64 " path=%s written=%d\n", id, arg1, LiveCounters::g_vma_report != nullptr ? 1 : 0);
	} else if (cmd == "images" && n >= 2) {
		// images <path>: the texture cache's images, one tab-separated row each (what the video memory holds).
		if (LiveCounters::g_image_report != nullptr) LiveCounters::g_image_report(arg1);
		std::printf("LIVE_IMAGES id=%" PRIu64 " path=%s written=%d\n", id, arg1, LiveCounters::g_image_report != nullptr ? 1 : 0);
	} else if (cmd == "sync") {
		// sync: GPU timelines, deferred submissions and pending guest readbacks (a stalled GPU thread).
		if (LiveCounters::g_sync_report != nullptr) LiveCounters::g_sync_report();
		std::printf("LIVE_SYNC id=%" PRIu64 "\n", id);
	} else if (cmd == "warp" && n >= 2) {
		// warp <map> <spawn> | warp off: the debug warp (loader/demonsSoulsWarp.h).
		const bool off = std::string_view(arg1) == "off";
		const bool ok  = off ? (Loader::DemonsSoulsWarp::Disarm(), true)
		                     : n == 3 && Loader::DemonsSoulsWarp::Arm(arg1, arg2);
		std::printf("LIVE_WARP id=%" PRIu64 " map=%s spawn=%s ok=%d spawns=%zu\n", id, arg1, n == 3 ? arg2 : "-",
		            ok ? 1 : 0, Loader::DemonsSoulsWarp::Spawns().size());
	} else if (cmd == "fatal") {
		// fatal: a fatal error on this thread, the path a crash takes (log, then the process ends).
		EXIT("live fatal: a test of the fatal-error exit\n");
	} else if (cmd == "sleep" && n >= 2) {
		std::this_thread::sleep_for(std::chrono::duration<double>(std::strtod(arg1, nullptr)));
	} else if (cmd == "capture" && n >= 2) {
		// capture <dir> [frames] [hash,hash,...]: every draw/dispatch of the next frames, with
		// the constants of the listed shaders (frame-capture.h).
		char hashes[1024] {};
		int  frames = 2;
		std::sscanf(line.c_str(), "%*s %*s %d %1023s", &frames, hashes);
		frames        = std::max(1, frames);
		const bool ok = FrameCapture::Arm(arg1, frames, hashes, 120.0);
		std::printf("LIVE_CAPTURE id=%" PRIu64 " frames=%d dir=%s data=%s ok=%d\n", id, frames, arg1,
		            hashes[0] != '\0' ? hashes : "-", ok ? 1 : 0);
	} else {
		std::printf("LIVE_ERROR id=%" PRIu64 " line=%s\n", id, line.c_str());
	}
	std::fflush(stdout);
}

// Called on the render thread before it consumes commands.
#if defined(_WIN32)
// KYTY_STALL_DUMP_MS: the game stopped submitting work for that long in the middle of play (the 60 frames before the
// wait averaged under 50 ms): every thread but the emulator's own (Kyty.*) and SDL's audio threads is sampled, three times 150 ms apart,
// to show what the game waits on. STALL lines: the frames of each thread's stack, guest code as module+offset, host
// code as addresses (symbolize with the build's map). Not with a live `prof` (both prepare the unwind tables).
inline void StallWatch(int64_t threshold_ns) {
	LocalPlatform::SetThreadName("Kyty.StallWatch");
	int64_t stall = 0;
	int     dumps = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		const int64_t since = g_render_idle_since.load(std::memory_order_relaxed);
		if (since == 0) continue;
		const int64_t idle = std::chrono::steady_clock::now().time_since_epoch().count() - since;
		if (since != stall) {
			stall = since;
			dumps = 0;
		}
		if (dumps >= 3 || idle < threshold_ns + dumps * int64_t {150'000'000}) continue;
		const auto flips = g_flips.load(std::memory_order_acquire);
		if (flips < 120) continue;
		const int64_t last  = g_flip_times[(flips - 1) % FlipTimes].load(std::memory_order_relaxed);
		const int64_t first = g_flip_times[(flips - 61) % FlipTimes].load(std::memory_order_relaxed);
		if ((last - first) / 60 > 50'000'000 || since - last > 50'000'000) {
			dumps = 3; // (a loading screen or a pause: not play)
			continue;
		}
		++dumps;
		LocalPlatform::PrepareSampling();
		const auto self = LocalPlatform::ThreadId();
		std::string out;
		char        line[160];
		std::snprintf(line, sizeof(line), "STALL idle=%.0f ms (dump %d), render thread waiting since %.0f ms after the last flip\n",
		              static_cast<double>(idle) / 1e6, dumps, static_cast<double>(since - last) / 1e6);
		out += line;
		for (const auto& [tid, name]: LocalPlatform::ProcessThreads()) {
			if (tid == self || (std::getenv("KYTY_STALL_DUMP_ALL") == nullptr && name.rfind("Kyty.", 0) == 0) || name.rfind("SDLAudio", 0) == 0) continue;
			const auto handle = LocalPlatform::OpenThreadForSampling(tid);
			if (handle == 0) continue;
			uint64_t words[SampleWords] {};
			const bool sampled = LocalPlatform::SampleThread(handle, 0, 0, words, SampleWords);
			LocalPlatform::CloseThreadForSampling(handle);
			if (!sampled) continue;
			std::snprintf(line, sizeof(line), "STALL   %u %s:", tid, name.empty() ? "-" : name.c_str());
			out += line;
			for (const auto word: words) {
				if (word == 0) break;
				const auto guest = Loader::DescribeGuestAddressForDiagnostics(word);
				if (!guest.empty()) out += " " + guest;
				else {
					std::snprintf(line, sizeof(line), " %llx", static_cast<unsigned long long>(word));
					out += line;
				}
			}
			out += "\n";
		}
		std::fputs(out.c_str(), stdout);
		std::fflush(stdout);
	}
}
#endif

inline void Start() {
	g_render_thread = LocalPlatform::CurrentThreadHandle();
	g_render_tid.store(LocalPlatform::ThreadId());
	LiveCensus::g_render = true;
	LiveCounters::g_single_writer = true;
	(void)LocalPlatform::CurrentThreadStack(&g_stack_low, &g_stack_high);
	g_render_known.store(true);
#if defined(_WIN32)
	if (const char* stall = std::getenv("KYTY_STALL_DUMP_MS"); stall != nullptr && std::atoi(stall) > 0)
		std::thread(StallWatch, int64_t {std::atoi(stall)} * 1'000'000).detach();
#endif
	static const char* const path = std::getenv("KYTY_LIVE_FILE");
	if (path == nullptr) return;
	std::thread([] {
		LocalPlatform::SetThreadName("Kyty.Live");
		// Commands already in the file belong to an earlier process: only newer ids run.
		uint64_t last = 0;
		if (std::FILE* file = std::fopen(path, "rb")) {
			unsigned long long id = 0;
			if (std::fscanf(file, "id %llu", &id) == 1) last = id;
			std::fclose(file);
		}
		for (;;) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			std::FILE* file = std::fopen(path, "rb");
			if (file == nullptr) continue;
			std::string text;
			char        buffer[4096];
			for (size_t got; (got = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) text.append(buffer, got);
			std::fclose(file);
			unsigned long long id = 0;
			if (std::sscanf(text.c_str(), "id %llu", &id) != 1 || id == last) continue;
			last         = id;
			size_t begin = text.find('\n');
			while (begin != std::string::npos && begin + 1 < text.size()) {
				const size_t end = text.find('\n', begin + 1);
				Run(id, text.substr(begin + 1, end == std::string::npos ? std::string::npos : end - begin - 1));
				begin = end;
			}
			std::printf("LIVE_DONE id=%llu\n", id);
			std::fflush(stdout);
		}
	}).detach();
}

} // namespace LiveControl
