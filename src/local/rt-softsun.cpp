#include "local/rt-softsun.h"
#include "local/env-level.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics::FrameGen::SoftSun {
namespace {
struct State {
	std::mutex              mutex;
	uint64_t                use_flip = ~0ull;
	uint64_t                source   = 0;
	std::array<uint64_t, 8> others {};
	uint32_t                n_others = 0;
	bool                    said     = false;
	// The 10 s line.
	std::chrono::steady_clock::time_point t0 {};
	uint64_t fresh = 0, held = 0, refused = 0, copies = 0;
} g;
} // namespace

int Level() {
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_SOFTSUN");
		const int   q     = ::Libs::Graphics::FrameGen::ParseEnvLevel(value);
		if (q > 0) {
			static const char* names[] {"No", "Bajo", "Medio", "Alto"};
			std::printf("SOFTSUN: KYTY_HWRT_SOFTSUN=%d (%s): sun %.2f deg, cascade copy held while in use (up to %d flips)\n", q,
			            names[q], 0.53 * (q == 1 ? 1.0 : q == 2 ? 1.5 : 2.0),
			            [] {
				            const char* h = std::getenv("KYTY_HWRT_SOFTSUN_HOLD");
				            return h != nullptr && *h != '\0' ? std::clamp(std::atoi(h), 8, 1000000) : 3600;
			            }());
			std::fflush(stdout);
		}
		return q;
	}();
	return level;
}

float AngleDeg() {
	const int q = Level();
	return 0.53f * (q <= 1 ? 1.0f : q == 2 ? 1.5f : 2.0f);
}

void NoteUse(uint64_t flip) {
	std::lock_guard lock(g.mutex);
	g.use_flip = flip;
}

void NoteCopy(uint64_t flip, uint64_t source, std::span<const uint64_t> others) {
	std::lock_guard lock(g.mutex);
	g.use_flip = flip, g.source = source, ++g.copies;
	g.n_others = 0;
	for (uint64_t a: others)
		if (a != 0 && a != source && g.n_others < g.others.size() &&
		    std::find(g.others.begin(), g.others.begin() + g.n_others, a) == g.others.begin() + g.n_others)
			g.others[g.n_others++] = a;
	if (!g.said) {
		g.said = true;
		std::printf("SOFTSUN: cascade copy from 0x%llx; its use tracked on it and %u companion array(s):", static_cast<unsigned long long>(source),
		            g.n_others);
		for (uint32_t i = 0; i < g.n_others; ++i) std::printf(" 0x%llx", static_cast<unsigned long long>(g.others[i]));
		std::printf("\n");
		std::fflush(stdout);
	}
}

bool IsCompanion(uint64_t address) {
	std::lock_guard lock(g.mutex);
	if (address == 0) return false;
	if (address == g.source) return true;
	for (uint32_t i = 0; i < g.n_others; ++i)
		if (g.others[i] == address) return true;
	return false;
}

bool CopyUsable(uint64_t now, uint64_t copy_flip, bool& held) {
	held = false;
	if (copy_flip == ~0ull) return false;
	if (now <= copy_flip + 8) return true;
	if (Level() <= 0) return false;
	static const uint64_t hold = [] {
		const char* h = std::getenv("KYTY_HWRT_SOFTSUN_HOLD");
		return static_cast<uint64_t>(h != nullptr && *h != '\0' ? std::clamp(std::atoi(h), 8, 1000000) : 3600);
	}();
	uint64_t use = ~0ull;
	{
		std::lock_guard lock(g.mutex);
		use = g.use_flip;
	}
	if (use == ~0ull || now > use + 8 || now > copy_flip + hold) return false;
	held = true;
	return true;
}

void NoteFrame(bool ran, bool held) {
	if (Level() <= 0) return;
	std::lock_guard lock(g.mutex);
	const auto now = std::chrono::steady_clock::now();
	if (g.t0.time_since_epoch().count() == 0) g.t0 = now;
	(ran ? (held ? g.held : g.fresh) : g.refused)++;
	if (now - g.t0 >= std::chrono::seconds(10)) {
		std::printf("SOFTSUN: 10 s: sun pass ran %llu frames (%llu on a fresh cascade copy, %llu on a held one), %llu without a usable "
		            "copy; %llu copies\n",
		            static_cast<unsigned long long>(g.fresh + g.held), static_cast<unsigned long long>(g.fresh),
		            static_cast<unsigned long long>(g.held), static_cast<unsigned long long>(g.refused),
		            static_cast<unsigned long long>(g.copies));
		std::fflush(stdout);
		g.t0 = now, g.fresh = g.held = g.refused = g.copies = 0;
	}
}

} // namespace Libs::Graphics::FrameGen::SoftSun
