#include "graphics/host_gpu/renderer/speculation.h"

#include "common/logging/log.h"
#include "common/perfStats.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

namespace Libs::Graphics::Speculation {

namespace {

struct Totals {
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> counts {};
	uint64_t fast_cycles    = 0;
	uint64_t fast_n         = 0;
	uint64_t classic_cycles = 0;
	uint64_t classic_n      = 0;
	uint64_t hit_serial     = 0;
};
Totals g_totals;
Totals g_last;
bool   g_announced = false;

} // namespace

uint32_t Threads() {
	static const uint32_t threads = [] {
		const char* value = std::getenv("KYTY_SPECULATE");
		if (value == nullptr || *value == '\0') {
			return 0u;
		}
		return static_cast<uint32_t>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 8ul));
	}();
	return threads;
}

void Add(Counter counter, uint64_t value) {
	g_totals.counts[static_cast<size_t>(counter)] += value;
}

void AddFastCycles(uint64_t cycles) {
	g_totals.fast_cycles += cycles;
	g_totals.fast_n++;
}

void AddClassicCycles(uint64_t cycles) {
	g_totals.classic_cycles += cycles;
	g_totals.classic_n++;
}

bool CalibrateNext() {
	return (++g_totals.hit_serial & 63u) == 0u;
}

void Report(double cycles_per_us, double seconds) {
	if (!Enabled()) {
		return;
	}
	if (!g_announced) {
		g_announced = true;
		Log::WriteUnmuted(fmt::format(
		    "Spec: KYTY_SPECULATE={}: at least {} draw-prep walk workers, validated speculated "
		    "program refreshes on a short path; a \"Spec\" line every 10 s\n",
		    Threads(), Threads()));
	}
	const auto delta = [&](Counter c) {
		const auto i = static_cast<size_t>(c);
		return g_totals.counts[i] - g_last.counts[i];
	};
	const auto fast_n         = g_totals.fast_n - g_last.fast_n;
	const auto classic_n      = g_totals.classic_n - g_last.classic_n;
	const auto fast_us        = fast_n != 0 && cycles_per_us > 0.0
	                                ? static_cast<double>(g_totals.fast_cycles - g_last.fast_cycles) /
	                                      cycles_per_us / static_cast<double>(fast_n)
	                                : 0.0;
	const auto classic_us     = classic_n != 0 && cycles_per_us > 0.0
	                                ? static_cast<double>(g_totals.classic_cycles -
	                                                      g_last.classic_cycles) /
	                                      cycles_per_us / static_cast<double>(classic_n)
	                                : 0.0;
	const auto hits           = delta(Counter::FastHit);
	const auto misses         = delta(Counter::FastMiss);
	const auto total          = hits + misses;
	const double saved_us     = classic_n != 0 && fast_n != 0
	                                ? std::max(0.0, classic_us - fast_us) * static_cast<double>(hits)
	                                : 0.0;
	std::string line = fmt::format(
	    "Spec ({:.1f} s): KYTY_SPECULATE={} | program refreshes speculated {} hits, {} misses "
	    "({:.1f}% hit) | short path {:.3f} us, classic {:.3f} us ({} samples) | "
	    "GPU thread saved ~{:.1f} ms ({:.2f} ms/s)",
	    seconds, Threads(), hits, misses,
	    total != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0,
	    fast_us, classic_us, classic_n, saved_us / 1000.0,
	    seconds > 0.0 ? saved_us / 1000.0 / seconds : 0.0);
	g_last = g_totals;
	Log::WriteUnmuted(line + "\n");
}

} // namespace Libs::Graphics::Speculation
