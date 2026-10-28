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
	return 0; // (ds-rt-stable-1027) removed: the owner keeps only RT reflections, RT sun / lamp shadows and volumetric shafts
	static const int level = [] {
		const char* value = std::getenv("KYTY_HWRT_SOFTSUN");
		const int   q     = ::Libs::Graphics::FrameGen::ParseEnvLevel(value);
		if (q > 0) {
			static const char* names[] {"No", "Bajo", "Medio", "Alto"};
			std::printf("SOFTSUN: KYTY_HWRT_SOFTSUN=%d (%s): sun %.2f deg, cascade copy held while in use (up to %d flips; -1: no cap)\n", q,
			            names[q], 0.53 * (q == 1 ? 1.0 : q == 2 ? 1.5 : 2.0),
			            [] {
				            const char* h = std::getenv("KYTY_HWRT_SOFTSUN_HOLD");
				            return h != nullptr && *h != '\0' ? std::clamp(std::atoi(h), 8, 1000000) : -1;
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

bool HoldOn() {
	static const bool on = [] {
		const auto lvl = [](const char* n) {
			const char* v = std::getenv(n);
			return v != nullptr && *v != '\0' ? std::atoi(v) : 0;
		};
		const char* h = std::getenv("KYTY_HWRT_SOFTSUN_HOLD");
		if (h != nullptr && *h == '0' && h[1] == '\0') return false;
		return Level() > 0 || lvl("KYTY_HWRT_SHADOW") > 0 || lvl("KYTY_RT_SUN") > 0;
	}();
	return on;
}

bool CopyUsable(uint64_t now, uint64_t copy_flip, bool& held, const char** why) {
	held = false;
	const auto refuse = [&](const char* w) {
		if (why != nullptr) *why = w;
		return false;
	};
	if (copy_flip == ~0ull) return refuse("no cascade copy yet");
	if (now <= copy_flip + 8) return true;
	if (!HoldOn()) return refuse("copy older than 8 flips (hold off)");
	// (ds-rt-shadow) no age cap by default: the game re-renders (and binds, so we copy) the far cascade only when it moves;
	// a still camera kept it for minutes and the old 3600-flip cap turned the RT sun off -> the game's shadow popped back.
	static const uint64_t hold = [] {
		const char* h = std::getenv("KYTY_HWRT_SOFTSUN_HOLD");
		return h != nullptr && *h != '\0' ? static_cast<uint64_t>(std::clamp(std::atoi(h), 8, 1000000)) : ~0ull / 2;
	}();
	uint64_t use = ~0ull;
	{
		std::lock_guard lock(g.mutex);
		use = g.use_flip;
	}
	// (ds-rt-shadow 10-28) no use gate: the consumer dispatches run on the table path, which never reaches
	// RtSunAfterDispatch, so "in use" was seen only on copy frames (20261028-001032 log: refused 579-665 of ~700 frames,
	// held 0) and the pass toggled. The copy is the game's current cascade until it copies again; the composite compares
	// against the game's own lookup on the same copy, so where the game itself is shadowed (indoors) nothing changes.
	// KYTY_HWRT_SOFTSUN_GATE=1: the old gate.
	static const bool gate = [] {
		const char* v = std::getenv("KYTY_HWRT_SOFTSUN_GATE");
		return v != nullptr && v[0] == '1';
	}();
	if (gate && (use == ~0ull || now > use + 8)) return refuse("game not using the sun shadows (indoors?)");
	if (now > copy_flip + hold) return refuse("copy older than KYTY_HWRT_SOFTSUN_HOLD");
	held = true;
	return true;
}

void NoteFrame(bool ran, bool held) {
	if (!HoldOn()) return;
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
