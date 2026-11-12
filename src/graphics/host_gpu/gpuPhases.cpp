#include "graphics/host_gpu/gpuPhases.h"

#include "common/logging/log.h"
#include "common/perfStats.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/speculation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <fmt/format.h>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::GpuPhases {

namespace {

constexpr auto ReportInterval = std::chrono::seconds(10);

struct Reporter {
	std::chrono::steady_clock::time_point last_time {};
	uint64_t                              last_tsc    = 0;
	uint64_t                              last_frames = 0;
	uint64_t                              next_check  = 0;
	// Round 131: the frame whose timing was decided last, and the timed frames at the last report.
	uint64_t                                            timing_frame      = UINT64_MAX;
	uint64_t                                            last_timed_frames = 0;
	Totals                                last {};
	std::array<uint64_t, PhaseCount + 1>  last_protect_cycles {};
	std::array<uint64_t, PhaseCount + 1>  last_protect_count {};
	std::array<uint64_t, PhaseCount + 1>                last_protect_pages {};
	std::array<std::array<uint64_t, 3>, PhaseCount + 1> last_protect_modes {};
	std::array<uint64_t, PhaseCount + 1>                last_upload_bytes {};
	std::array<uint64_t, PhaseCount + 1>                last_upload_count {};
	std::array<uint64_t, PhaseCount + 1>                last_upload_copies {};
	// The longest scope of each phase in the window just reported (Totals::longest).
	std::array<uint64_t, PhaseCount> last_longest {};
};

// The names of the steps page protection changes are charged to (Detail::ChargesProtects).
const char* StepName(size_t index) {
	switch (static_cast<Phase>(index)) {
		case Phase::Process: return "a slice";
		case Phase::Command: return "commands";
		case Phase::Draw: return "draws";
		case Phase::Dispatch: return "dispatches";
		case Phase::BdaWalk: return "bda walks";
		case Phase::DrawPending: return "draw pending";
		case Phase::DrawConsume: return "draw consume";
		case Phase::DrawPrograms: return "draw programs";
		case Phase::DrawTargets: return "draw targets";
		case Phase::DrawTextures: return "draw textures";
		case Phase::DrawFind: return "draw find";
		case Phase::DrawBda: return "draw bda";
		case Phase::DrawImages: return "draw images";
		case Phase::DrawBuffers: return "draw buffers";
		case Phase::DrawVertex: return "draw vertex";
		case Phase::DrawPipeline: return "draw pipeline";
		case Phase::DrawRendering: return "draw rendering";
		case Phase::DrawCommit: return "draw commit";
		case Phase::DrawEmit: return "draw emit";
		case Phase::DispatchPending: return "dispatch pending";
		case Phase::DispatchProgram: return "dispatch program";
		case Phase::DispatchClears: return "dispatch clears";
		case Phase::DispatchFill: return "dispatch fill";
		case Phase::DispatchPipeline: return "dispatch pipeline";
		case Phase::DispatchTextures: return "dispatch textures";
		case Phase::DispatchFind: return "dispatch find";
		case Phase::DispatchBda: return "dispatch bda";
		case Phase::DispatchImages: return "dispatch images";
		case Phase::DispatchBuffers: return "dispatch buffers";
		case Phase::DispatchCommit: return "dispatch commit";
		default: return "elsewhere";
	}
}

Reporter g_reporter;

// KYTY_OP_STATS (round 84): the operations by group since the last report (GPU thread only).
struct OpGroup {
	uint64_t extra          = 0;
	uint64_t count          = 0;
	uint64_t cycles         = 0;
	uint64_t bda_cycles     = 0;
	uint64_t protect_cycles = 0;
};
std::unordered_map<uint64_t, OpGroup> g_dispatch_groups;
std::unordered_map<uint64_t, OpGroup> g_draw_groups;

// The groups that took the most cycles, at most `shown` of them, as one line's text.
std::string DescribeGroups(const std::unordered_map<uint64_t, OpGroup>& groups, bool dispatch,
                           double cycles_per_us, uint64_t frames, size_t shown) {
	uint64_t                                         total_cycles = 0;
	uint64_t                                         total_count  = 0;
	std::vector<std::pair<uint64_t, const OpGroup*>> order;
	order.reserve(groups.size());
	for (const auto& [key, group]: groups) {
		total_cycles += group.cycles;
		total_count += group.count;
		order.emplace_back(key, &group);
	}
	const auto take = std::min(shown, order.size());
	std::partial_sort(
	    order.begin(), order.begin() + static_cast<ptrdiff_t>(take), order.end(),
	    [](const auto& a, const auto& b) { return a.second->cycles > b.second->cycles; });
	const auto per_frame = [&](double value) {
		return frames != 0 ? value / static_cast<double>(frames) : 0.0;
	};
	std::string text =
	    fmt::format("{} {}, {} {} ({:.1f} a frame), {:.1f} ms a frame; the most:",
	                dispatch ? "dispatches" : "draws", total_count, groups.size(),
	                dispatch ? "shaders" : "targets", per_frame(static_cast<double>(total_count)),
	                per_frame(static_cast<double>(total_cycles) / cycles_per_us / 1000.0));
	for (size_t i = 0; i < take; i++) {
		const auto& [key, group] = order[i];
		const auto  n            = static_cast<double>(group->count);
		const auto  us_each      = static_cast<double>(group->cycles) / cycles_per_us / n;
		std::string what;
		if (dispatch) {
			what = fmt::format("cs {:016x} ({:.0f} groups)", key,
			                   static_cast<double>(group->extra) / n);
		} else {
			const auto extent  = group->extra;
			const auto layers  = extent >> 32u;
			const auto address = key & ~(uint64_t {1} << 63u);
			what = fmt::format("{} 0x{:x} {}x{}", (key >> 63u) != 0 ? "depth" : "color", address,
			                   extent & 0xffffu, (extent >> 16u) & 0xffffu);
			if (layers > 1u) {
				what += fmt::format("x{}", layers);
			}
		}
		text += fmt::format("{} {}: {:.1f} a frame x {:.2f} us = {:.2f} ms a frame ({:.0f}%; bda "
		                    "{:.2f} us, protects {:.2f} us each)",
		                    i == 0 ? "" : ";", what, per_frame(n), us_each,
		                    per_frame(static_cast<double>(group->cycles) / cycles_per_us / 1000.0),
		                    total_cycles != 0 ? 100.0 * static_cast<double>(group->cycles) /
		                                            static_cast<double>(total_cycles)
		                                      : 0.0,
		                    static_cast<double>(group->bda_cycles) / cycles_per_us / n,
		                    static_cast<double>(group->protect_cycles) / cycles_per_us / n);
	}
	return text;
}

bool EnabledFromEnvironment() {
	const char* value = std::getenv("KYTY_GPU_PHASES");
	return value == nullptr || value[0] != '0';
}

// Round 131: KYTY_GPU_PHASES_EVERY=<n> times the draws and dispatches of one frame in n (4 by
// default; 1, every frame, as before). With KYTY_OP_STATS, which times whole operations, every
// frame.
uint64_t TimedEvery() {
	static const uint64_t every = [] {
		const char* value = std::getenv("KYTY_GPU_PHASES_EVERY");
		const auto  n = value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 4u;
		return std::clamp<uint64_t>(n, 1u, 64u);
	}();
	return OpStatsOn() ? 1u : every;
}

} // namespace

bool OpStatsOn() noexcept {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_OP_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool TakeFrameHold() noexcept {
	static const uint32_t threshold = [] {
		const char* value = std::getenv("KYTY_HOLD_SKIPPED_DRAWS");
		const auto n = value != nullptr && *value != '\0' ? std::strtoul(value, nullptr, 10) : 100u;
		LOGF("Frame holds: a frame that left out %lu draws or more while their pipelines were "
		     "looked up is not shown, the one before it stays (KYTY_HOLD_SKIPPED_DRAWS; 0 or F8's "
		     "hold:off shows every frame)\n",
		     n);
		return static_cast<uint32_t>(n);
	}();
	static const auto max_held = [] {
		const char* value = std::getenv("KYTY_HOLD_MAX_MS");
		return std::chrono::milliseconds(
		    value != nullptr && *value != '\0' ? std::strtoul(value, nullptr, 10) : 5000u);
	}();
	static const bool compiling = [] {
		const char* value = std::getenv("KYTY_HOLD_COMPILING");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Frame holds: draws left out while their pipelines compiled count too; frames are "
			     "held in a row for up to %lld ms (KYTY_HOLD_COMPILING, KYTY_HOLD_MAX_MS)\n",
			     static_cast<long long>(max_held.count()));
		}
		return on;
	}();
	static const uint32_t MaxInARow  = compiling ? UINT32_MAX : 2u;
	static const uint32_t ShownAfter = compiling ? 1u : 8u;
	struct Holds {
		std::chrono::steady_clock::time_point since    = std::chrono::steady_clock::now();
		std::chrono::steady_clock::time_point held_since; // the first frame of this run of holds
		uint32_t                              in_a_row = 0;
		uint32_t                              shown    = ShownAfter;
		// Since the last line.
		uint64_t frames   = 0;
		uint64_t skipping = 0; // frames that left draws out
		uint64_t skipped  = 0;
		uint64_t most     = 0;
		uint64_t over_32  = 0;
		uint64_t over_100 = 0;
		uint64_t over_300 = 0;
		uint64_t held     = 0;
	};
	static Holds   s;
	const uint64_t compile_skips = g_frame_compile_skips.exchange(0, std::memory_order_relaxed);
	const uint64_t skips = g_frame_lookup_skips.exchange(0, std::memory_order_relaxed) +
	                       (compiling ? compile_skips : 0u);
	const auto now = std::chrono::steady_clock::now();
	s.frames++;
	if (skips != 0) {
		s.skipping++;
		s.skipped += skips;
		s.most = std::max(s.most, skips);
		s.over_32 += skips >= 32 ? 1u : 0u;
		s.over_100 += skips >= 100 ? 1u : 0u;
		s.over_300 += skips >= 300 ? 1u : 0u;
	}
	const bool hold = threshold != 0 && !Bisect::FrameHoldOff() && skips >= threshold &&
	                  s.in_a_row < MaxInARow && (s.in_a_row != 0 || s.shown >= ShownAfter) &&
	                  (!compiling || s.in_a_row == 0 || now - s.held_since < max_held);
	if (hold) {
		if (s.in_a_row == 0) {
			s.held_since = now;
		}
		s.in_a_row++;
		s.shown = 0;
		s.held++;
	} else {
		s.in_a_row = 0;
		s.shown    = std::min(s.shown + 1u, ShownAfter);
	}
	if (now - s.since >= ReportInterval) {
		if (s.skipped != 0) {
			LOGF("Pipeline lookups (%.1f s): %" PRIu64 " draws left out while their pipelines were "
			     "looked up, in %" PRIu64 " of %" PRIu64 " frames (most in one frame %" PRIu64
			     "; frames with 32 or more %" PRIu64 ", 100 or more %" PRIu64
			     ", 300 or more %" PRIu64 "); frames not shown (the one before stayed) %" PRIu64
			     "\n",
			     std::chrono::duration<double>(now - s.since).count(), s.skipped, s.skipping,
			     s.frames, s.most, s.over_32, s.over_100, s.over_300, s.held);
		}
		const auto in_a_row   = s.in_a_row;
		const auto shown      = s.shown;
		const auto held_since = s.held_since;
		s                     = {};
		s.since               = now;
		s.in_a_row            = in_a_row;
		s.shown               = shown;
		s.held_since          = held_since;
	}
	return hold;
}

void NoteOp(bool dispatch, uint64_t key, uint64_t extra, uint64_t cycles, uint64_t bda_cycles,
            uint64_t protect_cycles) noexcept {
	auto& group = (dispatch ? g_dispatch_groups : g_draw_groups)[key];
	group.extra = dispatch ? group.extra + extra : extra;
	group.count++;
	group.cycles += cycles;
	group.bda_cycles += bda_cycles;
	group.protect_cycles += protect_cycles;
}

void Start() {
	if (!EnabledFromEnvironment()) {
		return;
	}
	Detail::t_on          = true;
	g_reporter.last_time  = std::chrono::steady_clock::now();
	g_reporter.last_tsc   = Now();
	g_reporter.next_check = g_reporter.last_tsc;
	LOGF("GPU phases: every 10 s, a \"GPU phases\" line with the GPU thread's time by step of a "
	     "draw and of a dispatch (KYTY_GPU_PHASES=0 turns it off), the draws and dispatches of one "
	     "frame in %" PRIu64 " timed (KYTY_GPU_PHASES_EVERY; F8's phases:all times every frame)\n",
	     TimedEvery());
	if (OpStatsOn()) {
		LOGF("GPU phases: with it, \"Dispatch shaders\" and \"Draw targets\" lines: the GPU "
		     "thread's time by dispatch shader and by draw target (KYTY_OP_STATS)\n");
	}
}

void ReportIfDue(uint64_t frames) {
	if (!Detail::t_on) {
		return;
	}
	if (frames != g_reporter.timing_frame) {
		// Round 131: whether this frame's draws and dispatches are timed.
		g_reporter.timing_frame = frames;
		const auto every        = Bisect::PhaseTimingAll() ? uint64_t {1} : TimedEvery();
		Detail::t_timed         = frames % every == 0;
		if (Detail::t_timed) {
			Detail::g_timed_frames++;
		}
	}
	const auto tsc = Now();
	// A clock read only every ~50 million cycles (~10 ms).
	if (tsc < g_reporter.next_check) {
		return;
	}
	g_reporter.next_check = tsc + 50'000'000u;
	const auto now        = std::chrono::steady_clock::now();
	if (now - g_reporter.last_time < ReportInterval) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - g_reporter.last_time).count();
	const double cycles_per_us =
	    static_cast<double>(tsc - g_reporter.last_tsc) / (seconds * 1'000'000.0);
	const auto& totals = Detail::g_totals;
	Totals      delta;
	for (size_t i = 0; i < PhaseCount; i++) {
		delta.cycles[i] = totals.cycles[i] - g_reporter.last.cycles[i];
		delta.count[i]  = totals.count[i] - g_reporter.last.count[i];
		delta.timed[i]  = totals.timed[i] - g_reporter.last.timed[i];
	}
	std::array<uint64_t, PhaseCount + 1> protect_cycles {};
	std::array<uint64_t, PhaseCount + 1> protect_count {};
	std::array<uint64_t, PhaseCount + 1>                protect_pages {};
	std::array<std::array<uint64_t, 3>, PhaseCount + 1> protect_modes {};
	for (size_t i = 0; i <= PhaseCount; i++) {
		protect_cycles[i] = Detail::g_protect_cycles[i] - g_reporter.last_protect_cycles[i];
		protect_count[i]  = Detail::g_protect_count[i] - g_reporter.last_protect_count[i];
		protect_pages[i]  = Detail::g_protect_pages[i] - g_reporter.last_protect_pages[i];
		for (size_t m = 0; m < 3; m++) {
			protect_modes[i][m] =
			    Detail::g_protect_modes[i][m] - g_reporter.last_protect_modes[i][m];
		}
	}
	g_reporter.last_protect_cycles = Detail::g_protect_cycles;
	g_reporter.last_protect_count  = Detail::g_protect_count;
	g_reporter.last_protect_pages  = Detail::g_protect_pages;
	g_reporter.last_protect_modes  = Detail::g_protect_modes;
	std::array<uint64_t, PhaseCount + 1> upload_bytes {};
	std::array<uint64_t, PhaseCount + 1> upload_count {};
	std::array<uint64_t, PhaseCount + 1> upload_copies {};
	for (size_t i = 0; i <= PhaseCount; i++) {
		upload_bytes[i]  = Detail::g_upload_bytes[i] - g_reporter.last_upload_bytes[i];
		upload_count[i]  = Detail::g_upload_count[i] - g_reporter.last_upload_count[i];
		upload_copies[i] = Detail::g_upload_copies[i] - g_reporter.last_upload_copies[i];
	}
	g_reporter.last_upload_bytes  = Detail::g_upload_bytes;
	g_reporter.last_upload_count  = Detail::g_upload_count;
	g_reporter.last_upload_copies = Detail::g_upload_copies;
	g_reporter.last_longest       = totals.longest;
	Detail::g_totals.longest.fill(0);
	g_reporter.last        = totals;
	g_reporter.last_time   = now;
	g_reporter.last_tsc    = tsc;
	const auto new_frames  = frames - g_reporter.last_frames;
	g_reporter.last_frames = frames;
	const auto timed_frames      = Detail::g_timed_frames - g_reporter.last_timed_frames;
	g_reporter.last_timed_frames = Detail::g_timed_frames;
	if (cycles_per_us <= 0.0) {
		return;
	}

	// A phase's time: its timed scopes' average times its count (round 131: the scopes of the
	// frames not timed are counted only).
	const auto us = [&](Phase phase) {
		const auto index = static_cast<size_t>(phase);
		if (delta.timed[index] == 0) {
			return 0.0;
		}
		return static_cast<double>(delta.cycles[index]) / cycles_per_us *
		       static_cast<double>(delta.count[index]) / static_cast<double>(delta.timed[index]);
	};
	const auto count = [&](Phase phase) { return delta.count[static_cast<size_t>(phase)]; };
	// Microseconds of `phase` per operation of `per`.
	const auto each = [&](Phase phase, Phase per) {
		const auto n = count(per);
		return n != 0 ? us(phase) / static_cast<double>(n) : 0.0;
	};

	std::string line = fmt::format("GPU phases ({:.1f} s)", seconds);
	if (const auto step = Common::PerfStats::WindowConfiguration(); !step.empty()) {
		line += fmt::format(" [{}]", step);
	}
	line += fmt::format(": {} frames", new_frames);
	if (timed_frames != new_frames) {
		line += fmt::format(" ({} timed)", timed_frames);
	}

	const auto steps = [&](Phase                                                total,
	                       std::initializer_list<std::pair<const char*, Phase>> parts) {
		double sum = 0.0;
		for (const auto& [name, phase]: parts) {
			const auto value = each(phase, total);
			sum += value;
			line += fmt::format(" {} {:.2f},", name, value);
		}
		line += fmt::format(" other {:.2f}", each(total, total) - sum);
	};

	line += fmt::format(" | draws {}, {:.2f} us each:", count(Phase::Draw),
	                    each(Phase::Draw, Phase::Draw));
	steps(Phase::Draw, {{"pending", Phase::DrawPending},
	                    {"consume", Phase::DrawConsume},
	                    {"programs", Phase::DrawPrograms},
	                    {"targets", Phase::DrawTargets},
	                    {"textures", Phase::DrawTextures},
	                    {"find", Phase::DrawFind},
	                    {"bda", Phase::DrawBda},
	                    {"images", Phase::DrawImages},
	                    {"buffers", Phase::DrawBuffers},
	                    {"vertex", Phase::DrawVertex},
	                    {"pipeline", Phase::DrawPipeline},
	                    {"rendering", Phase::DrawRendering},
	                    {"commit", Phase::DrawCommit},
	                    {"emit", Phase::DrawEmit}});
	line += fmt::format(" (describe {:.2f})", each(Phase::Describe, Phase::Draw));
	if (count(Phase::TargetMemoHit) + count(Phase::TargetMemoMiss) != 0) {
		line += fmt::format(" (target memo {} hits, {} misses)", count(Phase::TargetMemoHit),
		                    count(Phase::TargetMemoMiss));
	}
	if (count(Phase::ImageSetHit) + count(Phase::ImageSetMissNew) +
	        count(Phase::ImageSetMissCollision) + count(Phase::ImageSetMissStorage) +
	        count(Phase::ImageSetMissPool) !=
	    0) {
		line += fmt::format(" (image sets {} hits, {} written; misses: {} new, {} collision, {} "
		                    "storage, {} pool; dropped: {} views, {} layouts, {} full)",
		                    count(Phase::ImageSetHit), count(Phase::ImageSetWritten),
		                    count(Phase::ImageSetMissNew), count(Phase::ImageSetMissCollision),
		                    count(Phase::ImageSetMissStorage), count(Phase::ImageSetMissPool),
		                    count(Phase::ImageSetRetireViews), count(Phase::ImageSetRetireLayouts),
		                    count(Phase::ImageSetRetireFull));
	}
	if (count(Phase::DepthMemoHit) + count(Phase::DepthMemoMiss) != 0) {
		line += fmt::format(" (depth memo {} hits, {} misses)", count(Phase::DepthMemoHit),
		                    count(Phase::DepthMemoMiss));
	}
	// Round 78: the tessellated ones among them (GT7's car bodies), and their shaders' share.
	line += fmt::format(" | tessellated draws {}, {:.2f} us each (shaders {:.2f})",
	                    count(Phase::TessDraw), each(Phase::TessDraw, Phase::TessDraw),
	                    each(Phase::TessPrograms, Phase::TessDraw));

	line += fmt::format(" | dispatches {}, {:.2f} us each:", count(Phase::Dispatch),
	                    each(Phase::Dispatch, Phase::Dispatch));
	steps(Phase::Dispatch, {{"pending", Phase::DispatchPending},
	                        {"program", Phase::DispatchProgram},
	                        {"clears", Phase::DispatchClears},
	                        {"fill", Phase::DispatchFill},
	                        {"pipeline", Phase::DispatchPipeline},
	                        {"textures", Phase::DispatchTextures},
	                        {"find", Phase::DispatchFind},
	                        {"bda", Phase::DispatchBda},
	                        {"images", Phase::DispatchImages},
	                        {"buffers", Phase::DispatchBuffers},
	                        {"commit", Phase::DispatchCommit}});

	const auto counted = [&](const char* name, Phase phase) {
		return fmt::format("{} {} x {:.2f} us", name, count(phase), each(phase, phase));
	};
	line += fmt::format(
	    " | program gets {} x {:.2f} us: key {:.2f}, {}, {}", count(Phase::ProgramGet),
	    each(Phase::ProgramGet, Phase::ProgramGet), each(Phase::ProgramKey, Phase::ProgramGet),
	    counted("scanned", Phase::ProgramScanned), counted("walks", Phase::ProgramWalk));
	if (count(Phase::ProgramMemoHit) + count(Phase::ProgramMemoMiss) != 0) {
		line += fmt::format(" (memo {} hits, {} misses)", count(Phase::ProgramMemoHit),
		                    count(Phase::ProgramMemoMiss));
	}
	line += fmt::format(" (match {:.2f})", each(Phase::ProgramMatch, Phase::ProgramGet));
	line += fmt::format(" | {}, {}, {}, {}, {}", counted("draw bda", Phase::DrawBda),
	                    counted("dispatch bda", Phase::DispatchBda),
	                    counted("buffer syncs", Phase::BufferSync),
	                    counted("protects", Phase::Protect), counted("flushes", Phase::Flush));
	const auto walks = count(Phase::BdaWalk);
	const auto per_walk = [&](Phase phase) {
		return walks != 0 ? static_cast<double>(count(phase)) / static_cast<double>(walks) : 0.0;
	};
	line += fmt::format(" | {} ({:.1f} regions, {:.1f} protects each; {} after a wait in their "
	                    "submission, {} draws or dispatches after CPU writes left to the last one; "
	                    "{} waits on memory passed), {}, {}, stores on watched pages {}",
	                    counted("bda walks", Phase::BdaWalk), per_walk(Phase::BdaRegions),
	                    per_walk(Phase::BdaProtects), count(Phase::BdaAfterWait),
	                    count(Phase::BdaHeldBack), count(Phase::WaitPassed),
	                    counted("spec checks", Phase::SpecCheck),
	                    counted("buffers made", Phase::BufferCreate), count(Phase::CleanStore));
	{
		// The protection changes by the step they were made in, the costliest first (round 72).
		std::array<size_t, PhaseCount + 1> order {};
		for (size_t i = 0; i < order.size(); i++) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(),
		          [&](size_t a, size_t b) { return protect_cycles[a] > protect_cycles[b]; });
		std::string by_step;
		for (size_t k = 0; k < order.size() && k < 8; k++) {
			const auto i = order[k];
			if (protect_count[i] == 0) {
				break;
			}
			by_step += fmt::format("{}{} {} x {:.2f} us", by_step.empty() ? "" : ", ", StepName(i),
			                       protect_count[i],
			                       static_cast<double>(protect_cycles[i]) / cycles_per_us /
			                           static_cast<double>(protect_count[i]));
			// Round 80: the pages a change covered, and what the changes made the pages.
			const auto changes = static_cast<double>(protect_count[i]);
			by_step += fmt::format(
			    " ({:.1f} pages; {:.0f}% to no access, {:.0f}% read-only, {:.0f}% writable)",
			    static_cast<double>(protect_pages[i]) / changes,
			    100.0 * static_cast<double>(protect_modes[i][0]) / changes,
			    100.0 * static_cast<double>(protect_modes[i][1]) / changes,
			    100.0 * static_cast<double>(protect_modes[i][2]) / changes);
		}
		if (!by_step.empty()) {
			line += " | protects by step: " + by_step;
		}
	}
	{
		// Round 108: the blocks protected most, with what the changes made their pages.
		auto&                                                     blocks = Detail::g_protect_blocks;
		std::vector<std::pair<uint64_t, std::array<uint32_t, 3>>> sorted(blocks.begin(),
		                                                                 blocks.end());
		const auto total = [](const std::array<uint32_t, 3>& modes) {
			return static_cast<uint64_t>(modes[0]) + modes[1] + modes[2];
		};
		const auto shown = std::min<size_t>(sorted.size(), 10);
		std::partial_sort(
		    sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(shown), sorted.end(),
		    [&](const auto& a, const auto& b) { return total(a.second) > total(b.second); });
		std::string churn;
		for (size_t k = 0; k < shown; k++) {
			const auto& [block, modes] = sorted[k];
			churn += fmt::format("{}0x{:x} {} ({} no access, {} read-only, {} writable)",
			                     k == 0 ? "" : ", ", block << 16u, total(modes), modes[0], modes[1],
			                     modes[2]);
		}
		if (!churn.empty()) {
			line += fmt::format(" | protect churn, {} blocks of 64 KB, the most: {}", sorted.size(),
			                    churn);
		}
		blocks.clear();
	}
	{
		// Round 113: the slices by the kind of queue they ran, a frame (of the timed frames, which
		// alone measure them: round 131).
		const double frames_d = timed_frames != 0 ? static_cast<double>(timed_frames) : 1.0;
		const auto   ms_frame = [&](uint64_t cycles) {
            return static_cast<double>(cycles) / cycles_per_us / 1000.0 / frames_d;
		};
		const auto  per_frame = [&](uint64_t n) { return static_cast<double>(n) / frames_d; };
		const auto& queues    = Detail::g_queue_stats;
		line += fmt::format(
		    " | by queue, a frame: graphics {:.1f} ms in {:.0f} slices ({:.0f} draws {:.1f} ms, "
		    "{:.0f} dispatches {:.1f} ms), compute queues {:.1f} ms in {:.0f} slices ({:.0f} "
		    "dispatches {:.1f} ms)",
		    ms_frame(queues[0][0]), per_frame(queues[0][1]), per_frame(queues[0][2]),
		    ms_frame(queues[0][4]), per_frame(queues[0][3]), ms_frame(queues[0][5]),
		    ms_frame(queues[1][0]), per_frame(queues[1][1]), per_frame(queues[1][3]),
		    ms_frame(queues[1][5]));
		Detail::g_queue_stats = {};
	}
	if (Detail::g_yields != 0) {
		// Round 108: graphics streams stopped for compute work, and how far they ran first.
		line += fmt::format(" | compute yields {} ({:.0f} packets before each)", Detail::g_yields,
		                    static_cast<double>(Detail::g_yield_packets) /
		                        static_cast<double>(Detail::g_yields));
		Detail::g_yields        = 0;
		Detail::g_yield_packets = 0;
	}
	{
		// Round 81: the guest bytes uploaded to the GPU by the step they were uploaded in, the
		// most first (CPU writes made since a range was last uploaded; a frame's share with
		// the frames below).
		std::array<size_t, PhaseCount + 1> order {};
		for (size_t i = 0; i < order.size(); i++) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(),
		          [&](size_t a, size_t b) { return upload_bytes[a] > upload_bytes[b]; });
		std::string by_step;
		uint64_t    total = 0;
		for (const auto bytes: upload_bytes) {
			total += bytes;
		}
		for (size_t k = 0, shown = 0; k < order.size() && shown < 8; k++) {
			const auto i = order[k];
			if (upload_count[i] == 0) {
				continue;
			}
			shown++;
			by_step += fmt::format("{}{} {} x {:.1f} KB ({:.1f} copies)",
			                       by_step.empty() ? "" : ", ", StepName(i), upload_count[i],
			                       static_cast<double>(upload_bytes[i]) / 1024.0 /
			                           static_cast<double>(upload_count[i]),
			                       static_cast<double>(upload_copies[i]) /
			                           static_cast<double>(upload_count[i]));
		}
		if (!by_step.empty()) {
			line +=
			    fmt::format(" | buffer uploads {:.1f} MB", static_cast<double>(total) / 1048576.0);
			if (new_frames != 0) {
				line += fmt::format(" ({:.2f} MB a frame)", static_cast<double>(total) / 1048576.0 /
				                                                static_cast<double>(new_frames));
			}
			line += " by step: " + by_step;
		}
	}

	if (new_frames != 0) {
		const auto frame = [&](Phase phase) {
			return us(phase) / 1000.0 / static_cast<double>(new_frames);
		};
		line +=
		    fmt::format(" | a frame: processing {:.1f} ms (draws {:.1f}, dispatches {:.1f}, "
		                "bda {:.1f}), commands {:.1f} ms",
		                frame(Phase::Process), frame(Phase::Draw), frame(Phase::Dispatch),
		                frame(Phase::DrawBda) + frame(Phase::DispatchBda), frame(Phase::Command));
	}
	// GT7 port: the line goes where the Perf lines go (the port's log), so a run shows where
	// the GPU thread's time per draw goes without a console.
	Log::WriteUnmuted(line + "\n");
	// gt7-spec-1112: KYTY_SPECULATE's counters, on a line of their own.
	Speculation::Report(cycles_per_us, seconds);
	{
		// Round 165: what the renderer recorded besides draws and dispatches, a frame.
		namespace RC = RecordedCommands;
		static std::array<uint64_t, RC::Count> last {};
		std::array<uint64_t, RC::Count>        now_counts {};
		for (size_t i = 0; i < RC::Count; i++) {
			now_counts[i] = RC::g_counts[i].load(std::memory_order_relaxed);
		}
		const double frames_d = new_frames != 0 ? static_cast<double>(new_frames) : 1.0;
		const auto   n        = [&](RC::Kind kind) {
            return static_cast<double>(now_counts[kind] - last[kind]) / frames_d;
		};
		const auto mb = [&](RC::Kind kind) { return n(kind) / 1048576.0; };
		LOGF("Recorded (%.1f s), a frame: %.0f barriers (%.0f wait for every stage; %.0f memory, "
		     "%.0f buffer, %.0f image barriers, %.0f of them layout changes), %.0f render passes "
		     "(%.0f attachments loaded), %.0f buffer copies (%.2f MB), %.0f buffer updates "
		     "(%.2f MB), %.0f fills (%.2f MB), %.0f image uploads, %.0f image downloads, %.0f "
		     "image copies or resolves, %.0f clears\n",
		     seconds, n(RC::Barriers), n(RC::FullBarriers), n(RC::MemoryBarriers),
		     n(RC::BufferBarriers), n(RC::ImageBarriers), n(RC::LayoutChanges), n(RC::RenderPasses),
		     n(RC::LoadedAttachments), n(RC::BufferCopies), mb(RC::BufferCopyBytes),
		     n(RC::BufferUpdates), mb(RC::BufferUpdateBytes), n(RC::BufferFills),
		     mb(RC::BufferFillBytes), n(RC::ImageUploads), n(RC::ImageDownloads),
		     n(RC::ImageCopies), n(RC::Clears));
		// Round 172: pipeline binds and dynamic state calls, and those that changed nothing.
		if (n(RC::StateCalls) + n(RC::PipelineBinds) > 0.0) {
			LOGF("Recorded state (%.1f s), a frame: %.0f dynamic state calls (%.0f set what was "
			     "set already), %.0f pipeline binds (%.0f of the pipeline bound already); %.0f "
			     "left out (state:filter)\n",
			     seconds, n(RC::StateCalls), n(RC::StateUnchanged), n(RC::PipelineBinds),
			     n(RC::PipelineUnchanged), n(RC::StateLeftOut));
		}
		// Round 167: copies:lean's batches and images:lean's skipped barriers, when there were.
		if (n(RC::BatchedCopies) + n(RC::ImageBarriersSkipped) > 0.0) {
			LOGF("Lean barriers (%.1f s), a frame: %.0f copies in batches (%.0f barriers before "
			     "them, %.0f after them on their own, %.0f taken into another barrier, %.0f "
			     "between copies of the same bytes), %.0f image uses with no barrier\n",
			     seconds, n(RC::BatchedCopies), n(RC::BatchBefores), n(RC::BatchAfters),
			     n(RC::BatchMerged), n(RC::BatchOverlaps), n(RC::ImageBarriersSkipped));
		}
		last = now_counts;
		// Round 166: the barriers by the line that made them, the most first.
		static std::array<uint64_t, RC::g_barrier_sites.size()> last_site_count {};
		static std::array<uint64_t, RC::g_barrier_sites.size()> last_site_full {};
		struct Place {
			const char* file;
			uint32_t    line;
			uint64_t    count;
			uint64_t    full;
		};
		std::vector<Place> places;
		for (size_t i = 0; i < RC::g_barrier_sites.size(); i++) {
			const auto& site = RC::g_barrier_sites[i];
			const char* file = site.file.load(std::memory_order_acquire);
			if (file == nullptr) {
				continue;
			}
			const auto count = site.count.load(std::memory_order_relaxed);
			const auto full  = site.full.load(std::memory_order_relaxed);
			if (count != last_site_count[i]) {
				places.push_back({file, site.line.load(std::memory_order_acquire),
				                  count - last_site_count[i], full - last_site_full[i]});
			}
			last_site_count[i] = count;
			last_site_full[i]  = full;
		}
		std::sort(places.begin(), places.end(),
		          [](const Place& a, const Place& b) { return a.count > b.count; });
		std::string by_place;
		for (size_t k = 0; k < places.size() && k < 14; k++) {
			std::string_view file(places[k].file);
			if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
				file.remove_prefix(slash + 1);
			}
			by_place += fmt::format(
			    "{}{}:{} {:.0f}{}", by_place.empty() ? "" : ", ", file, places[k].line,
			    static_cast<double>(places[k].count) / frames_d,
			    places[k].full != 0
			        ? fmt::format(" ({:.0f} full)", static_cast<double>(places[k].full) / frames_d)
			        : std::string());
		}
		if (!by_place.empty()) {
			LOGF("Recorded barriers by place, a frame: %s\n", by_place.c_str());
		}
	}
	{
		// Round 86: the GPU thread's shader and pipeline work, when there was some.
		const auto longest_ms = [&](Phase phase) {
			return static_cast<double>(g_reporter.last_longest[static_cast<size_t>(phase)]) /
			       cycles_per_us / 1000.0;
		};
		const auto work = [&](const char* name, Phase phase) {
			return fmt::format("{} {} in {:.0f} ms (longest {:.1f} ms)", name, count(phase),
			                   us(phase) / 1000.0, longest_ms(phase));
		};
		// Round 130: the lookups made on worker threads since the last report.
		const uint64_t worker_lookups =
		    g_worker_lookups.count.exchange(0, std::memory_order_relaxed);
		const uint64_t worker_us = g_worker_lookups.us.exchange(0, std::memory_order_relaxed);
		const uint64_t worker_longest =
		    g_worker_lookups.longest_us.exchange(0, std::memory_order_relaxed);
		const uint64_t worker_found = g_worker_lookups.found.exchange(0, std::memory_order_relaxed);
		// Round 134: the shader disk cache reads made on worker threads.
		const uint64_t disk_reads =
		    g_worker_disk_reads.count.exchange(0, std::memory_order_relaxed);
		const uint64_t disk_us = g_worker_disk_reads.us.exchange(0, std::memory_order_relaxed);
		const uint64_t disk_longest =
		    g_worker_disk_reads.longest_us.exchange(0, std::memory_order_relaxed);
		const uint64_t disk_found =
		    g_worker_disk_reads.found.exchange(0, std::memory_order_relaxed);
		if (count(Phase::ShaderDisk) + count(Phase::ShaderCompile) + count(Phase::PipelineWait) +
		        count(Phase::PipelineCompile) + count(Phase::TranslationSkip) +
		        count(Phase::PipelineSkip) + count(Phase::PipelineProbe) +
		        count(Phase::PipelineLookupSkip) + worker_lookups + disk_reads +
		        count(Phase::DiskReadSkip) !=
		    0) {
			std::string workers;
			if (worker_lookups + count(Phase::PipelineLookupSkip) != 0) {
				workers = fmt::format(
				    "; on worker threads pipeline cache lookups {} in {:.0f} ms (longest {:.1f} "
				    "ms) (found {}, to compile {}), draws skipped while their pipeline was looked "
				    "up {}",
				    worker_lookups, static_cast<double>(worker_us) / 1000.0,
				    static_cast<double>(worker_longest) / 1000.0, worker_found,
				    worker_lookups - worker_found, count(Phase::PipelineLookupSkip));
				// Round 132: each lookup's average time by how many ran at once as it began.
				std::string                          by;
				constexpr std::array<const char*, 4> Names {"alone", "2-3", "4-7", "8+"};
				for (size_t i = 0; i < Names.size(); i++) {
					const auto n = g_worker_lookups.by_concurrency_count[i].exchange(
					    0, std::memory_order_relaxed);
					const auto t = g_worker_lookups.by_concurrency_us[i].exchange(
					    0, std::memory_order_relaxed);
					if (n != 0) {
						by +=
						    fmt::format("{}{} {} x {:.1f} ms", by.empty() ? "" : ", ", Names[i], n,
						                static_cast<double>(t) / 1000.0 / static_cast<double>(n));
					}
				}
				if (!by.empty()) {
					workers += " (by lookups at once: " + by + ")";
				}
			}
			if (disk_reads + count(Phase::DiskReadSkip) != 0) {
				workers += fmt::format(
				    "; on worker threads shader disk cache reads {} in {:.0f} ms (longest {:.1f} "
				    "ms) ({} found), draws skipped while their shaders were read {}",
				    disk_reads, static_cast<double>(disk_us) / 1000.0,
				    static_cast<double>(disk_longest) / 1000.0, disk_found,
				    count(Phase::DiskReadSkip));
			}
			LOGF("Shader work (%.1f s): on the GPU thread %s, %s, %s (found %" PRIu64
			     ", to compile %" PRIu64 "), %s, %s; draws skipped while their shaders "
			     "translated %" PRIu64 ", while their pipeline compiled %" PRIu64 "%s\n",
			     seconds, work("disk cache reads", Phase::ShaderDisk).c_str(),
			     work("translations and compiles", Phase::ShaderCompile).c_str(),
			     work("pipeline cache lookups", Phase::PipelineProbe).c_str(),
			     count(Phase::PipelineProbeHit), count(Phase::PipelineProbeMiss),
			     work("pipeline waits", Phase::PipelineWait).c_str(),
			     work("pipeline compiles", Phase::PipelineCompile).c_str(),
			     count(Phase::TranslationSkip), count(Phase::PipelineSkip), workers.c_str());
		}
	}
	if (OpStatsOn()) {
		LOGF("Dispatch shaders (%.1f s): %s\n", seconds,
		     DescribeGroups(g_dispatch_groups, true, cycles_per_us, new_frames, 16).c_str());
		LOGF("Draw targets (%.1f s): %s\n", seconds,
		     DescribeGroups(g_draw_groups, false, cycles_per_us, new_frames, 16).c_str());
		g_dispatch_groups.clear();
		g_draw_groups.clear();
	}
}

} // namespace Libs::Graphics::GpuPhases
