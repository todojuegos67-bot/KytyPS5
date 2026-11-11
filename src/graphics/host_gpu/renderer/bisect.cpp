#include "graphics/host_gpu/renderer/bisect.h"

#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/perfStats.h"
#include "graphics/host_gpu/renderer/cache/pagedRangeSet.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "kernel/memory.h"
#include "kernel/syncOnAddress.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

namespace Libs::Graphics::Bisect {
namespace {

struct Skip {
	uint64_t    prefix = 0;
	uint32_t    shift  = 0; // the low bits of the hash a shorter prefix leaves out
	std::string text;
};

struct Step {
	std::vector<Skip> skips;
	bool              lit_big     = false;
	bool              lit_all     = false;
	bool              whole_cs    = false;
	bool              log_off     = false;
	bool              writes_off  = false;
	bool              nan_off     = false;
	bool              fence       = false;
	bool              gc_off      = false;
	bool              gpu_watch   = false;
	bool              tess_off    = false;
	int32_t           tess_bit        = -1; // tess:bit:<k> (round 82); -1: none
	bool              readahead   = false;
	bool              cmask_off    = false;
	bool              fastread_off = false;
	bool              fastread_any = false;
	bool              fastread_gpu    = false; // round 152
	bool              liftwait_old    = false; // round 153
	int32_t           faultblock_kb   = -1;    // faultblock:<kb> (round 155); -1: the usual
	bool              cuts156_off     = false; // round 156
	bool              aamask_off      = false; // round 159
	bool              recheck      = false;
	bool              locks_spin   = false;
	bool              htile_depth  = false;
	bool              pattern_off    = false;
	bool              faultblock_off = false;
	bool              pop_wait       = false;
	bool              texmemo_off    = false;
	bool              rebind_twice    = false;
	bool              drawprep_off    = false;
	bool              gpuwrites_fault = false;
	bool              bdawalk_all     = false;
	bool              checkpoints_on  = false;
	bool              speccheck_all   = false;
	bool              drawprep_taint  = false;
	bool              bdaspan_off     = false;
	bool              scanplan_shared = false;
	bool              views_old       = false;
	bool              grids_off       = false;
	bool              grids_raw       = false;
	bool              scan_one        = false;
	bool              hot_pages       = false;
	bool              memsync_async   = false; // memsync:async (round 111)
	float             equal_relaxed   = 0.0f; // depth:equal-relaxed[:<units>]; 0: off
	bool              hiz_off         = false;
	bool              unorm_lod       = false;
	bool              unorm_old       = false;
	bool              drawprep_notess = false;
	bool              texmemo_global  = false;
	bool              faultblock_old  = false;
	bool              alias_old       = false;
	bool              spechash_off    = false;
	bool              dirtyindex_off  = false;
	bool              flipcheck_old   = false;
	bool              bdaprotect_old  = false;
	bool              listreset_off   = false;
	// Round 82.
	bool              scanwave_old    = false;
	bool              drawstate_old   = false;
	bool              refresh_old     = false;
	bool              expand_old      = false;
	bool              backing_locked  = false;
	// Round 84.
	bool              buflen_old      = false;
	// Round 85.
	uint32_t          cube_every      = 0;    // cube:<n>; 0: KYTY_CUBE_EVERY
	// Round 90.
	bool              bdasync_submission = false;
	// Round 91.
	bool              gbuffer_sync    = false;
	bool              depthread_old   = false;
	// Round 92.
	bool              stream_host       = false;
	bool              buffers_dedicated = false;
	float             tess_max    = 0.0f; // tess:max:<f>; 0: none
	uint32_t          flush       = 0;    // flush:<n>; 0: the usual
	int32_t           dflush          = -1;   // dflush:<n>; -1: the usual
	uint32_t          gpu_frames  = 0;    // gputime:<n>
	int64_t           readback_kb = -1; // -1: the usual window
	// Round 108.
	bool              yield_off = false;
	int64_t           yield_us  = -1;   // yield:<microseconds> (round 109); -1: the usual
	// Round 114: slow:gpu, slow:faults, slow:reads (microseconds; 0: off).
	uint32_t          slow_gpu_us   = 0;
	uint32_t          slow_fault_us = 0;
	uint32_t          slow_read_us  = 0;
	// Round 130.
	bool              probe_sync = false;
	// Round 131.
	bool              phases_all = false;
	// Round 132.
	bool              memsync_sync   = false;
	bool              syncwait_old   = false;
	bool              storeclean_off = false;
	bool              hold_off       = false; // hold:off (round 133)
	// Round 135.
	bool              cleanpage_old   = false; // cleanpage:old
	bool              readwait_writer = false; // readwait:writer
	bool              barriers_global = false; // barriers:global
	bool              barriers_guest  = false; // barriers:guest (round 166)
	// Round 167.
	bool              barriers_renderer = false; // barriers:renderer
	bool              copies_lean       = false; // copies:lean
	bool              images_lean       = false; // images:lean
	bool              lean_off          = false; // lean:off (round 169)
	bool              scan_shared       = false; // scan:shared (round 170)
	bool              scan_oldtaint     = false; // scan:oldtaint (round 170)
	bool              state_filter      = false; // state:filter (round 172)
	bool              protect_excl    = false; // protect:exclusive
	std::string       cpu_set;          // empty: as at the start
	std::string       name;             // for the screen and the picture's file name
	std::string       text;             // name and actions, for the log
};

std::string_view Trim(std::string_view text) {
	while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
		text.remove_prefix(1);
	}
	while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
		text.remove_suffix(1);
	}
	return text;
}

std::vector<std::string_view> Split(std::string_view text, char separator) {
	std::vector<std::string_view> parts;
	size_t                        start = 0;
	while (start <= text.size()) {
		const auto end = (std::min)(text.find(separator, start), text.size());
		if (const auto part = Trim(text.substr(start, end - start)); !part.empty()) {
			parts.push_back(part);
		}
		start = end + 1;
	}
	return parts;
}

bool ParseAction(std::string_view action, Step& step) {
	if (action == "lit:big") {
		step.lit_big = true;
		return true;
	}
	if (action == "lit:all") {
		step.lit_all = true;
		return true;
	}
	if (action == "mips:whole" || action == "mips:cs") {
		step.whole_cs = true;
		return true;
	}
	if (action == "log:off") {
		step.log_off = true;
		return true;
	}
	if (action == "writes:off") {
		step.writes_off = true;
		return true;
	}
	if (action == "nancheck:off") {
		step.nan_off = true;
		return true;
	}
	if (action == "present:fence") {
		step.fence = true;
		return true;
	}
	if (action == "gc:off") {
		step.gc_off = true;
		return true;
	}
	if (action == "gpuwatch") {
		step.gpu_watch = true;
		return true;
	}
	if (action == "tess:off") {
		step.tess_off = true;
		return true;
	}
	if (action.starts_with("tess:bit:")) {
		const auto text = action.substr(9);
		if (text.empty() || text.size() > 2 ||
		    text.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		const auto bit = std::strtoul(std::string(text).c_str(), nullptr, 10);
		if (bit > 15) {
			return false;
		}
		step.tess_bit = static_cast<int32_t>(bit);
		return true;
	}
	if (action == "readahead") {
		step.readahead = true;
		return true;
	}
	if (action == "cmask:off") {
		step.cmask_off = true;
		return true;
	}
	if (action == "fastread:off") {
		step.fastread_off = true;
		return true;
	}
	if (action == "fastread:any") {
		step.fastread_any = true;
		return true;
	}
	if (action == "fastread:gpu") {
		step.fastread_gpu = true;
		return true;
	}
	if (action == "liftwait:old") {
		step.liftwait_old = true;
		return true;
	}
	if (action == "cuts156:off") {
		step.cuts156_off = true;
		return true;
	}
	if (action == "aamask:off") {
		step.aamask_off = true;
		return true;
	}
	if (action == "recheck") {
		step.recheck = true;
		return true;
	}
	if (action == "locks:spin") {
		step.locks_spin = true;
		return true;
	}
	if (action == "htile:depth") {
		step.htile_depth = true;
		return true;
	}
	if (action == "patternfill:off") {
		step.pattern_off = true;
		return true;
	}
	if (action == "faultblock:off") {
		step.faultblock_off = true;
		return true;
	}
	if (action == "popwait") {
		step.pop_wait = true;
		return true;
	}
	if (action == "texmemo:off") {
		step.texmemo_off = true;
		return true;
	}
	if (action == "rebind2") {
		step.rebind_twice = true;
		return true;
	}
	if (action == "drawprep:off") {
		step.drawprep_off = true;
		return true;
	}
	if (action == "gpuwrites:fault") {
		step.gpuwrites_fault = true;
		return true;
	}
	if (action == "bdawalk:all") {
		step.bdawalk_all = true;
		return true;
	}
	if (action == "checkpoints:on") {
		step.checkpoints_on = true;
		return true;
	}
	if (action == "speccheck:all") {
		step.speccheck_all = true;
		return true;
	}
	if (action == "drawprep:taint") {
		step.drawprep_taint = true;
		return true;
	}
	if (action == "bdaspan:off") {
		step.bdaspan_off = true;
		return true;
	}
	if (action == "scanplan:shared") {
		step.scanplan_shared = true;
		return true;
	}
	if (action == "views:old") {
		step.views_old = true;
		return true;
	}
	if (action == "grids:off") {
		step.grids_off = true;
		return true;
	}
	if (action == "grids:raw") {
		step.grids_raw = true;
		return true;
	}
	if (action == "scan:one") {
		step.scan_one = true;
		return true;
	}
	if (action == "hotpages") {
		step.hot_pages = true;
		return true;
	}
	if (action == "memsync:async") {
		step.memsync_async = true;
		return true;
	}
	if (action == "depth:equal-relaxed") {
		step.equal_relaxed = 256.0f;
		return true;
	}
	if (action == "hiz:off") {
		step.hiz_off = true;
		return true;
	}
	if (action == "unorm:lod") {
		step.unorm_lod = true;
		return true;
	}
	if (action == "unorm:old") {
		step.unorm_old = true;
		return true;
	}
	if (action == "yield:off") {
		step.yield_off = true;
		return true;
	}
	if (action.starts_with("yield:")) {
		const auto text = action.substr(6);
		if (text.empty() || text.size() > 6 ||
		    text.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		step.yield_us = static_cast<int64_t>(std::strtoul(std::string(text).c_str(), nullptr, 10));
		return true;
	}
	// Round 114: slow:gpu, slow:faults and slow:reads, each with an optional :<microseconds>.
	for (const auto& [prefix, target, fallback]:
	     {std::tuple {std::string_view("slow:gpu"), &step.slow_gpu_us, 3u},
	      std::tuple {std::string_view("slow:faults"), &step.slow_fault_us, 15u},
	      std::tuple {std::string_view("slow:reads"), &step.slow_read_us, 2500u}}) {
		if (action == prefix) {
			*target = fallback;
			return true;
		}
		if (action.size() > prefix.size() + 1 && action.starts_with(prefix) &&
		    action[prefix.size()] == ':') {
			const auto text = action.substr(prefix.size() + 1);
			if (text.size() > 6 || text.find_first_not_of("0123456789") != std::string_view::npos) {
				return false;
			}
			*target = static_cast<uint32_t>(std::strtoul(std::string(text).c_str(), nullptr, 10));
			return true;
		}
	}
	if (action == "drawprep:notess") {
		step.drawprep_notess = true;
		return true;
	}
	if (action == "texmemo:global") {
		step.texmemo_global = true;
		return true;
	}
	if (action == "faultblock:old") {
		step.faultblock_old = true;
		return true;
	}
	if (action == "alias:old") {
		step.alias_old = true;
		return true;
	}
	if (action == "spechash:off") {
		step.spechash_off = true;
		return true;
	}
	if (action == "dirtyindex:off") {
		step.dirtyindex_off = true;
		return true;
	}
	if (action == "flipcheck:old") {
		step.flipcheck_old = true;
		return true;
	}
	if (action == "bdaprotect:old") {
		step.bdaprotect_old = true;
		return true;
	}
	if (action == "listreset:off") {
		step.listreset_off = true;
		return true;
	}
	if (action == "scanwave:old") {
		step.scanwave_old = true;
		return true;
	}
	if (action == "drawstate:old") {
		step.drawstate_old = true;
		return true;
	}
	if (action == "refresh:old") {
		step.refresh_old = true;
		return true;
	}
	if (action == "expand:old") {
		step.expand_old = true;
		return true;
	}
	if (action == "backing:locked") {
		step.backing_locked = true;
		return true;
	}
	if (action == "buflen:old") {
		step.buflen_old = true;
		return true;
	}
	if (action == "storeclean:off") {
		step.storeclean_off = true;
		return true;
	}
	if (action == "hold:off") {
		step.hold_off = true;
		return true;
	}
	if (action == "cleanpage:old") {
		step.cleanpage_old = true;
		return true;
	}
	if (action == "readwait:writer") {
		step.readwait_writer = true;
		return true;
	}
	if (action == "barriers:global") {
		step.barriers_global = true;
		return true;
	}
	if (action == "barriers:guest") {
		step.barriers_guest = true;
		return true;
	}
	if (action == "barriers:renderer") {
		step.barriers_renderer = true;
		return true;
	}
	if (action == "copies:lean") {
		step.copies_lean = true;
		return true;
	}
	if (action == "images:lean") {
		step.images_lean = true;
		return true;
	}
	if (action == "lean:off") {
		step.lean_off = true;
		return true;
	}
	if (action == "scan:shared") {
		step.scan_shared = true;
		return true;
	}
	if (action == "scan:oldtaint") {
		step.scan_oldtaint = true;
		return true;
	}
	if (action == "state:filter") {
		step.state_filter = true;
		return true;
	}
	if (action == "protect:exclusive") {
		step.protect_excl = true;
		return true;
	}
	if (action == "memsync:sync") {
		step.memsync_sync = true;
		return true;
	}
	if (action == "syncwait:old") {
		step.syncwait_old = true;
		return true;
	}
	if (action == "phases:all") {
		step.phases_all = true;
		return true;
	}
	if (action == "probe:sync") {
		step.probe_sync = true;
		return true;
	}
	if (action == "bdasync:submission") {
		step.bdasync_submission = true;
		return true;
	}
	if (action == "gbuffer:sync") {
		step.gbuffer_sync = true;
		return true;
	}
	if (action == "depthread:old") {
		step.depthread_old = true;
		return true;
	}
	if (action == "stream:host") {
		step.stream_host = true;
		return true;
	}
	if (action == "buffers:dedicated") {
		step.buffers_dedicated = true;
		return true;
	}
	if (action.starts_with("cube:")) {
		const auto text = action.substr(5);
		if (text.size() != 1 || text[0] < '1' || text[0] > '8') {
			return false;
		}
		step.cube_every = static_cast<uint32_t>(text[0] - '0');
		return true;
	}
	if (action.starts_with("depth:equal-relaxed:")) {
		const auto text = std::string(action.substr(20));
		if (text.empty() || text.size() > 7 ||
		    text.find_first_not_of("0123456789") != std::string::npos) {
			return false;
		}
		const auto units = std::strtof(text.c_str(), nullptr);
		if (!(units >= 1.0f && units <= 1000000.0f)) {
			return false;
		}
		step.equal_relaxed = units;
		return true;
	}
	if (action.starts_with("tess:max:")) {
		const auto text = std::string(action.substr(9));
		if (text.empty() || text.size() > 8 ||
		    text.find_first_not_of("0123456789.") != std::string::npos) {
			return false;
		}
		const auto factor = std::strtof(text.c_str(), nullptr);
		// The host tessellates factors from 1 to 64.
		if (!(factor >= 1.0f && factor <= 64.0f)) {
			return false;
		}
		step.tess_max = factor;
		return true;
	}
	if (action.starts_with("flush:")) {
		const auto draws = action.substr(6);
		if (draws.empty() || draws.size() > 5 ||
		    draws.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		step.flush = static_cast<uint32_t>(std::strtoul(std::string(draws).c_str(), nullptr, 10));
		return step.flush != 0;
	}
	if (action.starts_with("dflush:")) {
		const auto dispatches = action.substr(7);
		if (dispatches.empty() || dispatches.size() > 5 ||
		    dispatches.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		step.dflush =
		    static_cast<int32_t>(std::strtoul(std::string(dispatches).c_str(), nullptr, 10));
		return true;
	}
	if (action.starts_with("gputime:")) {
		const auto frames = action.substr(8);
		if (frames.empty() || frames.size() > 4 ||
		    frames.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		step.gpu_frames =
		    static_cast<uint32_t>(std::strtoul(std::string(frames).c_str(), nullptr, 10));
		return step.gpu_frames != 0;
	}
	if (action == "cpu:vcache" || action == "cpu:frequency" || action == "cpu:all" ||
	    action == "cpu:gpuvcache" || action == "cpu:gpufrequency" || action == "cpu:gpucore") {
		step.cpu_set = std::string(action.substr(4));
		return true;
	}
	if (action.starts_with("faultblock:") && action != "faultblock:off" &&
	    action != "faultblock:old") {
		// Round 155: a write fault's block of that many KB (8 to 1024, a power of two).
		const auto kb = action.substr(11);
		if (kb.empty() || kb.size() > 4 ||
		    kb.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		const auto value = std::strtoul(std::string(kb).c_str(), nullptr, 10);
		if (value < 8 || value > 1024 || (value & (value - 1u)) != 0) {
			return false;
		}
		step.faultblock_kb = static_cast<int32_t>(value);
		return true;
	}
	if (action.starts_with("readback:")) {
		const auto kb = action.substr(9);
		if (kb.empty() || kb.find_first_not_of("0123456789") != std::string_view::npos) {
			return false;
		}
		step.readback_kb =
		    static_cast<int64_t>(std::strtoull(std::string(kb).c_str(), nullptr, 10));
		return true;
	}
	if (action == "mips:graphics") {
		LOGF("Bisect: mips:graphics changes nothing: graphics stages keep the whole view "
		     "anyway\n");
		return true;
	}
	if (action.starts_with("skip:")) {
		auto hex = action.substr(5);
		if (hex.starts_with("0x") || hex.starts_with("0X")) {
			hex.remove_prefix(2);
		}
		if (hex.size() < 8 || hex.size() > 16 ||
		    hex.find_first_not_of("0123456789abcdefABCDEF") != std::string_view::npos) {
			return false;
		}
		Skip skip;
		skip.prefix = std::strtoull(std::string(hex).c_str(), nullptr, 16);
		skip.shift  = static_cast<uint32_t>(4u * (16u - hex.size()));
		skip.text   = std::string(hex);
		step.skips.push_back(std::move(skip));
		return true;
	}
	return false;
}

// 0: none on (normal); n: Steps()[n - 1].
std::atomic<uint32_t> g_current {0};

const std::vector<Step>& Steps() {
	static const std::vector<Step> steps = [] {
		std::vector<Step> result;
		const char*       value = std::getenv("KYTY_BISECT");
		if (value == nullptr) {
			return result;
		}
		for (auto text: Split(value, ';')) {
			Step step;
			bool understood = true;
			// "<name>=<actions>": the name is what the screen shows.
			std::string_view name = text;
			if (const auto equals = text.find('='); equals != std::string_view::npos) {
				name = Trim(text.substr(0, equals));
				text = Trim(text.substr(equals + 1));
			}
			for (const auto action: Split(text, '+')) {
				if (!ParseAction(action, step)) {
					LOGF("Bisect: KYTY_BISECT action \"%.*s\" is not understood; its step is left "
					     "out\n",
					     static_cast<int>(action.size()), action.data());
					understood = false;
				}
			}
			if (!understood) {
				continue;
			}
			step.name = std::string(name.empty() ? text : name);
			step.text = name == text || name.empty() ? std::string(text)
			                                         : fmt::format("{} ({})", name, text);
			result.push_back(std::move(step));
		}
		std::string list;
		for (size_t i = 0; i < result.size(); i++) {
			list += fmt::format("{}{}: {}", i == 0 ? "" : ", ", i + 1, result[i].text);
		}
		LOGF("Bisect: KYTY_BISECT has %zu steps; each F8 saves a picture, then turns on the next "
		     "one: %s\n",
		     result.size(), list.c_str());
		// KYTY_BISECT_START=<n>: step n is on from the start (for what is made while loading).
		if (const char* start = std::getenv("KYTY_BISECT_START"); start != nullptr) {
			const auto n = std::strtoul(start, nullptr, 10);
			if (n >= 1 && n <= result.size()) {
				g_current.store(static_cast<uint32_t>(n), std::memory_order_release);
				LOGF("Bisect: KYTY_BISECT_START: step %lu is on from the start: %s\n", n,
				     result[n - 1].text.c_str());
			}
		}
		return result;
	}();
	return steps;
}

// Whether this step's first use of each action was logged (bit 0 skip, 1 lit, 2 mips, 3 tess:off).
std::atomic<uint32_t> g_logged {0};
// Advance's picture and the step that follows it.
std::mutex              g_pending_mutex;
std::string             g_picture;
std::optional<uint32_t> g_pending;
uint32_t                g_presses = 0;

// KYTY_BISECT_AB=1 (round 79): the step that is on alternates with normal at each Perf line.
bool AbMode() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BISECT_AB");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
// While AbMode alternates: the step that is on is off for now (its "base" lines).
std::atomic<bool> g_ab_base {false};

const Step* Current() {
	const auto& steps   = Steps();
	const auto  current = g_current.load(std::memory_order_acquire);
	if (current == 0 || current > steps.size() || g_ab_base.load(std::memory_order_acquire)) {
		return nullptr;
	}
	return &steps[current - 1];
}

bool FirstUse(uint32_t bit) {
	return (g_logged.fetch_or(bit, std::memory_order_acq_rel) & bit) == 0;
}

// The placement the run started with (Common::StartCpuSet: KYTY_CPU_SET, or round 80's default
// on a Ryzen with a 3D V-Cache die beside another); "start" when it left the placement alone,
// which gives the process back the processors it started on.
std::string StartCpuSet() {
	static const std::string start = [] {
		auto set = Common::StartCpuSet();
		return set.empty() ? std::string("start") : set;
	}();
	return start;
}

// What the step that is on asks of the process (the CPU set, the log) and the name the Perf lines
// give it. Once at the start, then at every change (toggled: KYTY_BISECT_AB's alternation, which
// does not time GPU frames again).
void ApplyEffects(bool toggled = false) {
	const auto& steps   = Steps();
	const auto  current = g_current.load(std::memory_order_acquire);
	const Step* chosen  = current == 0 || current > steps.size() ? nullptr : &steps[current - 1];
	const bool  base    = chosen != nullptr && g_ab_base.load(std::memory_order_acquire);
	const Step* step    = base ? nullptr : chosen;
	Common::PerfStats::SetConfiguration(chosen == nullptr ? std::string("normal")
	                                    : base            ? chosen->name + " base"
	                                                      : chosen->name);
	// The 10 s reports span both halves of an alternation.
	Common::PerfStats::SetWindowConfiguration(chosen != nullptr && AbMode() ? chosen->name + " A/B"
	                                                                        : std::string());
	Log::SetMuted(step != nullptr && step->log_off);
	if (step != nullptr && step->gpu_frames != 0 && !toggled) {
		GpuTiming::RequestFrames(step->gpu_frames);
		Log::WriteUnmuted(fmt::format("Bisect: gputime: timing the next {} frames on the GPU\n",
		                              step->gpu_frames));
	}
	GpuTiming::SetWatch(step != nullptr && step->gpu_watch);
	PagedRangeSet::SetIndexOff(step != nullptr && step->dirtyindex_off);
	LibKernel::Memory::SetBackingCopyLocked(step != nullptr && step->backing_locked);
	LibKernel::Memory::SetProtectLockExclusive(step != nullptr && step->protect_excl);
	LibKernel::SyncOnAddress::SetOldWaits(step != nullptr && step->syncwait_old);
	Common::PerfStats::SetSpinLocks(Common::PerfStats::SpinLocksFromEnvironment() ||
	                                (step != nullptr && step->locks_spin));
	static std::string applied = StartCpuSet();
	const auto wanted = step != nullptr && !step->cpu_set.empty() ? step->cpu_set : StartCpuSet();
	if (wanted != applied) {
		std::string description;
		Common::SetProcessCpuSet(wanted.c_str(), &description);
		Log::WriteUnmuted(fmt::format("Bisect: cpu:{}: {}\n", wanted, description));
		applied = wanted;
	}
}

// g_pending_mutex held.
void ApplyLocked() {
	if (!g_pending.has_value()) {
		return;
	}
	g_logged.store(0, std::memory_order_release);
	g_ab_base.store(false, std::memory_order_release);
	g_current.store(*g_pending, std::memory_order_release);
	g_pending.reset();
	ApplyEffects();
}

// The Perf reporter, after each Perf line (AbMode): the step that is on goes off, or on again.
void OnPerfReport() {
	std::scoped_lock lock(g_pending_mutex);
	if (g_pending.has_value() || g_current.load(std::memory_order_acquire) == 0) {
		return;
	}
	const bool base = !g_ab_base.load(std::memory_order_acquire);
	g_ab_base.store(base, std::memory_order_release);
	ApplyEffects(true);
}

std::string FileName(std::string_view name) {
	std::string out;
	for (const char c: name) {
		const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
		                  c == '-';
		if (keep) {
			out += c;
		} else if (!out.empty() && out.back() != '_') {
			out += '_';
		}
		if (out.size() >= 40) {
			break;
		}
	}
	while (!out.empty() && out.back() == '_') {
		out.pop_back();
	}
	return out.empty() ? "step" : out;
}

} // namespace

std::string Advance() {
	const auto& steps = Steps();
	if (steps.empty()) {
		static std::atomic<bool> told {false};
		if (!told.exchange(true)) {
			LOGF("Bisect: F8 does nothing: KYTY_BISECT is not set\n");
		}
		return {};
	}
	std::scoped_lock lock(g_pending_mutex);
	// A press before the last one's picture was taken: that step goes on now, unpictured.
	ApplyLocked();
	const auto now  = g_current.load(std::memory_order_acquire);
	const auto next = (now + 1u) % static_cast<uint32_t>(steps.size() + 1u);
	g_presses++;
	g_picture = fmt::format("bisect_{:02}_{}", g_presses,
	                        now == 0 ? std::string("normal") : FileName(steps[now - 1].name));
	g_pending = next;
	// Written even under log:off, which the press may be leaving.
	Log::WriteUnmuted(fmt::format(
	    "Bisect: F8 press {}: a picture of {} ({}.jpg), then {}\n", g_presses,
	    now == 0 ? std::string("normal") : fmt::format("step {}", now), g_picture,
	    next == 0 ? std::string("back to normal")
	              : fmt::format("step {} of {}: {}", next, steps.size(), steps[next - 1].text)));
	if (next == 0) {
		return fmt::format("Bisect: normal (all {} steps off)", steps.size());
	}
	return fmt::format("Bisect step {} of {}: {}", next, steps.size(), steps[next - 1].name);
}

void RequestPicture(const std::string& name) {
	std::scoped_lock lock(g_pending_mutex);
	if (g_picture.empty()) {
		g_picture = name;
	}
}

std::string TakePictureRequest() {
	std::scoped_lock lock(g_pending_mutex);
	std::string      picture;
	picture.swap(g_picture);
	return picture;
}

void ApplyPending() {
	std::scoped_lock lock(g_pending_mutex);
	ApplyLocked();
}

bool Active() {
	return Current() != nullptr;
}

bool TessBitsConfigured() {
	static const bool configured = [] {
		const auto& steps = Steps();
		return std::any_of(steps.begin(), steps.end(),
		                   [](const Step& step) { return step.tess_bit >= 0; });
	}();
	return configured;
}

int32_t TessSkipBit() {
	const auto* step = Current();
	return step != nullptr ? step->tess_bit : -1;
}

uint32_t StepNumber() {
	return g_ab_base.load(std::memory_order_acquire) ? 0u
	                                                 : g_current.load(std::memory_order_acquire);
}

bool SkipsShader(uint64_t hash) {
	const auto* step = Current();
	if (step == nullptr) {
		return false;
	}
	for (const auto& skip: step->skips) {
		if ((hash >> skip.shift) == skip.prefix) {
			if (FirstUse(1u)) {
				LOGF("Bisect: skipping shader %016" PRIx64 " (skip:%s), the first this step\n", hash,
				     skip.text.c_str());
			}
			return true;
		}
	}
	return false;
}

bool PassesDepthCompare(bool big_depth_image) {
	const auto* step = Current();
	if (step == nullptr || !(step->lit_all || (step->lit_big && big_depth_image))) {
		return false;
	}
	if (FirstUse(2u)) {
		LOGF("Bisect: a depth comparison%s passes (%s), the first this step\n",
		     big_depth_image ? " against a big depth image" : "",
		     step->lit_all ? "lit:all" : "lit:big");
	}
	return true;
}

void Start() {
	if (Steps().empty()) {
		return;
	}
	if (AbMode()) {
		LOGF("Bisect: KYTY_BISECT_AB=1: the step that is on alternates with normal at each Perf "
		     "line (\"<step>\" and \"<step> base\")\n");
		Common::PerfStats::SetReportHook(&OnPerfReport);
	}
	std::scoped_lock lock(g_pending_mutex);
	ApplyEffects();
}

int64_t ReadbackWindowKB() {
	const auto* step = Current();
	return step != nullptr ? step->readback_kb : -1;
}

bool CleanWritesOff() {
	const auto* step = Current();
	return step != nullptr && step->writes_off;
}

bool NanCheckOff() {
	const auto* step = Current();
	return step != nullptr && step->nan_off;
}

bool PresentFence() {
	const auto* step = Current();
	return step != nullptr && step->fence;
}

bool GcOff() {
	const auto* step = Current();
	return step != nullptr && step->gc_off;
}

bool TessOff() {
	const auto* step = Current();
	if (step == nullptr || !step->tess_off) {
		return false;
	}
	if (FirstUse(8u)) {
		LOGF("Bisect: tess:off: a tessellated draw is left out, the first this step\n");
	}
	return true;
}

float TessMaxLevel() {
	const auto* step = Current();
	return step != nullptr ? step->tess_max : 0.0f;
}

uint32_t FlushInterval() {
	const auto* step = Current();
	return step != nullptr ? step->flush : 0u;
}

bool Readahead() {
	const auto* step = Current();
	return step != nullptr && step->readahead;
}

bool MsaaCmaskOff() {
	const auto* step = Current();
	return step != nullptr && step->cmask_off;
}

bool FastReadbackOff() {
	const auto* step = Current();
	return step != nullptr && step->fastread_off;
}

bool FastReadbackAnyAddress() {
	const auto* step = Current();
	return step != nullptr && step->fastread_any;
}

bool FastReadbackOnGpuThread() {
	const auto* step = Current();
	return step != nullptr && step->fastread_gpu;
}

bool LiftWaitOld() {
	const auto* step = Current();
	return step != nullptr && step->liftwait_old;
}

bool Cuts156Off() {
	const auto* step = Current();
	return step != nullptr && step->cuts156_off;
}

bool AaMaskOff() {
	const auto* step = Current();
	return step != nullptr && step->aamask_off;
}

bool GpuClearRecheck() {
	const auto* step = Current();
	return step != nullptr && step->recheck;
}

bool LocksSpin() {
	const auto* step = Current();
	return step != nullptr && step->locks_spin;
}

bool HtileDepthOnly() {
	const auto* step = Current();
	return step != nullptr && step->htile_depth;
}

bool PatternFillsOff() {
	const auto* step = Current();
	return step != nullptr && step->pattern_off;
}

bool FaultBlocksOff() {
	const auto* step = Current();
	return step != nullptr && step->faultblock_off;
}

int32_t FaultBlockKB() {
	const auto* step = Current();
	return step != nullptr ? step->faultblock_kb : -1;
}

bool PopWait() {
	const auto* step = Current();
	return step != nullptr && step->pop_wait;
}

bool TextureMemoOff() {
	const auto* step = Current();
	return step != nullptr && step->texmemo_off;
}

bool RebindTwice() {
	const auto* step = Current();
	return step != nullptr && step->rebind_twice;
}

bool DrawPrepOff() {
	const auto* step = Current();
	return step != nullptr && step->drawprep_off;
}

bool GpuWritesFault() {
	const auto* step = Current();
	return step != nullptr && step->gpuwrites_fault;
}

bool BdaWalkAll() {
	const auto* step = Current();
	return step != nullptr && step->bdawalk_all;
}

bool CheckpointsOn() {
	const auto* step = Current();
	return step != nullptr && step->checkpoints_on;
}

bool SpecCheckAll() {
	const auto* step = Current();
	return step != nullptr && step->speccheck_all;
}

int32_t DispatchFlushInterval() {
	const auto* step = Current();
	return step != nullptr ? step->dflush : -1;
}

bool DrawPrepTaint() {
	const auto* step = Current();
	return step != nullptr && step->drawprep_taint;
}

bool BdaSpanOff() {
	const auto* step = Current();
	return step != nullptr && step->bdaspan_off;
}

bool ScanPlanShared() {
	const auto* step = Current();
	return step != nullptr && step->scanplan_shared;
}

bool OldViewBase() {
	const auto* step = Current();
	return step != nullptr && step->views_old;
}

bool LightGridsOff() {
	const auto* step = Current();
	return step != nullptr && step->grids_off;
}

bool LightGridsRaw() {
	const auto* step = Current();
	return step != nullptr && step->grids_raw;
}

bool ScanOneThread() {
	const auto* step = Current();
	return step != nullptr && step->scan_one;
}

bool ScanShared() {
	const auto* step = Current();
	return step != nullptr && step->scan_shared;
}

bool ScanOldTaint() {
	const auto* step = Current();
	return step != nullptr && step->scan_oldtaint;
}

bool StateFilter() {
	static const bool whole_run = [] {
		const char* value = std::getenv("KYTY_STATE_FILTER");
		const bool  on    = value == nullptr || std::strcmp(value, "0") != 0;
		LOGF("State filter: %s (KYTY_STATE_FILTER=1, or KYTY_BISECT's state:filter for a step: "
		     "pipeline binds and dynamic state calls that set what the command buffer holds "
		     "already are left out)\n",
		     on ? "on" : "off");
		return on;
	}();
	if (whole_run) {
		return true;
	}
	const auto* step = Current();
	return step != nullptr && step->state_filter;
}

bool HotPages() {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_HOT_PAGES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	const auto* step = Current();
	return always || (step != nullptr && step->hot_pages);
}

bool AsyncMemorySync() {
	// Round 132: on unless KYTY_ASYNC_MEMORY_SYNC=0 (or F8's memsync:sync, for a step). In a race
	// A/B it took 0.4 us off each draw and dispatch of the GPU thread.
	static const bool on = [] {
		const char* value = std::getenv("KYTY_ASYNC_MEMORY_SYNC");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	const auto* step = Current();
	if (step != nullptr && step->memsync_async) {
		return true;
	}
	return on && !(step != nullptr && step->memsync_sync);
}

bool HiZOff() {
	const auto* step = Current();
	return step != nullptr && step->hiz_off;
}

bool UnnormalizedLod() {
	const auto* step = Current();
	return step != nullptr && step->unorm_lod;
}

bool UnnormalizedOld() {
	const auto* step = Current();
	return step != nullptr && step->unorm_old;
}

bool DrawPrepNoTess() {
	const auto* step = Current();
	return step != nullptr && step->drawprep_notess;
}

bool TextureMemoGlobal() {
	const auto* step = Current();
	return step != nullptr && step->texmemo_global;
}

bool FaultBlockAroundImagesOff() {
	const auto* step = Current();
	return step != nullptr && step->faultblock_old;
}

bool ReadOnlyAliasesOff() {
	const auto* step = Current();
	return step != nullptr && step->alias_old;
}

bool SpecHashOff() {
	const auto* step = Current();
	return step != nullptr && step->spechash_off;
}

bool FlipCheckOld() {
	const auto* step = Current();
	return step != nullptr && step->flipcheck_old;
}

bool BdaProtectOld() {
	const auto* step = Current();
	return step != nullptr && step->bdaprotect_old;
}

bool ComputeYieldOff() {
	const auto* step = Current();
	return step != nullptr && step->yield_off;
}

int64_t ComputeYieldMicroseconds() {
	const auto* step = Current();
	return step != nullptr ? step->yield_us : -1;
}

uint32_t SlowGpuMicroseconds() {
	const auto* step = Current();
	return step != nullptr ? step->slow_gpu_us : 0;
}

uint32_t SlowFaultMicroseconds() {
	const auto* step = Current();
	return step != nullptr ? step->slow_fault_us : 0;
}

uint32_t SlowReadMicroseconds() {
	const auto* step = Current();
	return step != nullptr ? step->slow_read_us : 0;
}

bool ZeroNewMemory() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ZERO_NEW_MEMORY");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Zero new memory (KYTY_ZERO_NEW_MEMORY=1): new game buffers are zeroed before "
			     "their contents are copied in, and new images nothing uploads into are cleared "
			     "to zero\n");
		}
		return on;
	}();
	return enabled;
}

void SpinMicroseconds(uint32_t microseconds) {
	if (microseconds == 0) {
		return;
	}
	const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(microseconds);
	while (std::chrono::steady_clock::now() < until) {
	}
}

bool ListResetOff() {
	const auto* step = Current();
	return step != nullptr && step->listreset_off;
}

bool ScanWaveOld() {
	const auto* step = Current();
	return step != nullptr && step->scanwave_old;
}

bool DrawStateOld() {
	const auto* step = Current();
	return step != nullptr && step->drawstate_old;
}

bool RefreshOld() {
	const auto* step = Current();
	return step != nullptr && step->refresh_old;
}

bool ExpandOld() {
	const auto* step = Current();
	return step != nullptr && step->expand_old;
}

bool BufferLengthOld() {
	const auto* step = Current();
	return step != nullptr && step->buflen_old;
}

uint32_t CubeEvery() {
	const auto* step = Current();
	return step != nullptr ? step->cube_every : 0u;
}

float DepthEqualRelaxed() {
	const auto* step = Current();
	return step != nullptr ? step->equal_relaxed : 0.0f;
}

bool BdaSyncSubmission() {
	const auto* step = Current();
	return step != nullptr && step->bdasync_submission;
}

bool StoreCleanOff() {
	const auto* step = Current();
	return step != nullptr && step->storeclean_off;
}

bool FrameHoldOff() {
	const auto* step = Current();
	return step != nullptr && step->hold_off;
}

bool CleanPageOld() {
	const auto* step = Current();
	return step != nullptr && step->cleanpage_old;
}

bool ReadWaitWriter() {
	const auto* step = Current();
	return step != nullptr && step->readwait_writer;
}

bool GlobalPm4Barriers() {
	const auto* step = Current();
	return step != nullptr && step->barriers_global;
}

bool GuestPm4Barriers() {
	const auto* step = Current();
	return step != nullptr && step->barriers_guest;
}

bool RendererPm4Barriers() {
	const auto* step = Current();
	return step != nullptr && step->barriers_renderer;
}

// Round 169: KYTY_LEAN_BARRIERS=1 turns copies:lean and images:lean on for the whole run (F8's
// lean:off turns them off for a step). Off by default: with round 168's shaders a race A/B found
// no difference (with the shaders of before, about 5 % more frames).
static bool LeanBarriersDefault() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LEAN_BARRIERS");
		const bool  lean  = value == nullptr || std::strcmp(value, "0") != 0;
		LOGF("Lean barriers: %s (KYTY_LEAN_BARRIERS=1: copies share their barriers, image uses in "
		     "an unchanged layout make none; KYTY_BISECT's copies:lean, images:lean and lean:off "
		     "for a step)\n",
		     lean ? "on" : "off");
		return lean;
	}();
	return on;
}

bool LeanCopies() {
	const auto* step = Current();
	if (step != nullptr && (step->copies_lean || step->lean_off)) {
		return !step->lean_off;
	}
	return LeanBarriersDefault();
}

bool LeanImages() {
	const auto* step = Current();
	if (step != nullptr && (step->images_lean || step->lean_off)) {
		return !step->lean_off;
	}
	return LeanBarriersDefault();
}

bool PhaseTimingAll() {
	const auto* step = Current();
	return step != nullptr && step->phases_all;
}

bool PipelineProbeSync() {
	const auto* step = Current();
	return step != nullptr && step->probe_sync;
}

bool GbufferSync() {
	const auto* step = Current();
	return step != nullptr && step->gbuffer_sync;
}

bool DepthReadOld() {
	const auto* step = Current();
	return step != nullptr && step->depthread_old;
}

bool StreamHostConfigured() {
	static const bool configured = [] {
		const auto& steps = Steps();
		return std::any_of(steps.begin(), steps.end(),
		                   [](const Step& step) { return step.stream_host; });
	}();
	return configured;
}

bool StreamHost() {
	const auto* step = Current();
	return step != nullptr && step->stream_host;
}

bool BuffersDedicated() {
	const auto* step = Current();
	return step != nullptr && step->buffers_dedicated;
}

bool WholeUnnormalizedViews() {
	const auto* step = Current();
	if (step == nullptr || !step->whole_cs) {
		return false;
	}
	if (FirstUse(4u)) {
		LOGF("Bisect: an unnormalized sampler of a compute shader keeps the whole view, the first "
		     "this step\n");
	}
	return true;
}

} // namespace Libs::Graphics::Bisect
