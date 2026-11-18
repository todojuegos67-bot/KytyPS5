#include "graphics/host_gpu/renderer/mtGpuStageA.h"

#include "common/logging/log.h"
#include "common/perfStats.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <fmt/format.h>

namespace Libs::Graphics::MtGpu {

namespace {

constexpr uint32_t ZoneBits   = 16u;   // 64 KiB zones
constexpr uint32_t TableBits  = 18u;   // 256 Ki stamps (2 MiB): aliasing only makes it stricter
constexpr uint64_t TableMask  = (uint64_t {1} << TableBits) - 1u;
constexpr uint64_t WideZones  = 256u;  // a write over more zones (16 MiB) stamps everything
constexpr uint32_t PrepBits   = 12u;   // prepared chunks remembered
constexpr uint32_t CostSample = 64u;   // one call in 64 is timed

// (epoch << 2) | kind.
std::unique_ptr<std::atomic<uint64_t>[]> g_zones;
std::atomic<uint64_t>                    g_epoch {0};
std::atomic<uint64_t>                    g_wide {0};

struct PrepSlot {
	std::atomic<uint64_t> address {0};
	std::atomic<uint64_t> epoch {0};
};
std::array<PrepSlot, size_t {1} << PrepBits> g_prepared;

enum Stat : uint32_t {
	Chunks,          // chunks with draws
	Accepted,
	RejectGpu,
	RejectHost,
	RejectImage,
	Uncertain,       // a zone read was rewritten after the chunk started (by itself or others)
	Unprepared,      // the scanner never walked it
	Writes,
	Reads,
	WriteCostNs,     // sampled
	WriteSamples,
	ReadCostNs,
	ReadSamples,
	StatCount,
};
std::array<std::atomic<uint64_t>, StatCount> g_stats {};
std::array<uint64_t, StatCount>              g_last {};

void Add(Stat s, uint64_t v = 1) {
	g_stats[s].fetch_add(v, std::memory_order_relaxed);
}

// GPU thread state of the current chunk.
struct Chunk {
	bool     open      = false;
	bool     prepared  = false;
	uint64_t prep      = 0;
	uint64_t start     = 0;
	uint32_t draws     = 0;
	uint32_t verdict   = 0; // 0 ok, 1 gpu, 2 host, 3 image, 4 uncertain
};
thread_local Chunk t_chunk;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool Sampled() {
	static thread_local uint32_t n = 0;
	return (++n % CostSample) == 0;
}

uint64_t SlotOf(uint64_t address) {
	return (address >> 2u) * 0x9E3779B97F4A7C15ull >> (64u - PrepBits);
}

void Classify(uint64_t stamp) {
	auto&      c     = t_chunk;
	const auto epoch = stamp >> 2u;
	if (epoch <= c.prep || c.verdict != 0) {
		return;
	}
	if (epoch <= c.start) {
		c.verdict = static_cast<uint32_t>(stamp & 3u); // 1..3
		if (c.verdict == 0) {
			c.verdict = 4;
		}
	} else {
		c.verdict = 4;
	}
}

void CloseChunk() {
	auto& c = t_chunk;
	if (!c.open) {
		return;
	}
	c.open = false;
	if (c.draws == 0) {
		return;
	}
	Add(Chunks);
	if (!c.prepared) {
		Add(Unprepared);
		return;
	}
	switch (c.verdict) {
		case 0: Add(Accepted); break;
		case 1: Add(RejectGpu); break;
		case 2: Add(RejectHost); break;
		case 3: Add(RejectImage); break;
		default: Add(Uncertain); break;
	}
}

} // namespace

bool StageAEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MT_GPU");
		const bool  on    = value != nullptr && std::strcmp(value, "2") == 0;
		if (on) {
			g_zones = std::make_unique<std::atomic<uint64_t>[]>(size_t {1} << TableBits);
			Log::WriteUnmuted("MT GPU stage A: record only (KYTY_MT_GPU=2)\n");
		}
		return on;
	}();
	return enabled;
}

void NoteWrite(WriteKind kind, uint64_t vaddr, uint64_t size) {
	if (!StageAEnabled() || size == 0) {
		return;
	}
	const bool     timed = Sampled();
	const uint64_t t0    = timed ? NowNs() : 0;
	const uint64_t epoch = g_epoch.fetch_add(1, std::memory_order_relaxed) + 1u;
	const uint64_t stamp = (epoch << 2u) | static_cast<uint64_t>(kind);
	const uint64_t first = vaddr >> ZoneBits;
	const uint64_t last  = (vaddr + size - 1u) >> ZoneBits;
	if (last - first >= WideZones) {
		g_wide.store(stamp, std::memory_order_relaxed);
	} else {
		for (uint64_t z = first; z <= last; z++) {
			g_zones[z & TableMask].store(stamp, std::memory_order_relaxed);
		}
	}
	Add(Writes);
	if (timed) {
		Add(WriteCostNs, NowNs() - t0);
		Add(WriteSamples);
	}
}

bool SelfCheckEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_MT_GPU");
		const bool  on    = value != nullptr && std::strcmp(value, "3") == 0;
		if (on) {
			Log::WriteUnmuted("MT GPU stage B: self-check of the translation context (KYTY_MT_GPU=3)\n");
		}
		return on;
	}();
	return enabled;
}

namespace {
thread_local bool             t_check_pending = false;
std::atomic<uint64_t>         g_checked {0};
std::atomic<uint64_t>         g_differ {0};
uint64_t                      g_checked_last = 0;
uint64_t                      g_differ_last  = 0;
} // namespace

bool TakeSelfCheckDraw() {
	if (!SelfCheckEnabled() || !t_check_pending) {
		return false;
	}
	t_check_pending = false;
	return true;
}

void NoteSelfCheck(bool same) {
	g_checked.fetch_add(1, std::memory_order_relaxed);
	if (!same) {
		g_differ.fetch_add(1, std::memory_order_relaxed);
	}
}

void NotePrepared(uint64_t address) {
	if (!StageAEnabled()) {
		return;
	}
	auto& slot = g_prepared[SlotOf(address)];
	slot.epoch.store(g_epoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
	slot.address.store(address, std::memory_order_release);
}

void ChunkBegin(uint64_t address, uint64_t size_bytes) {
	if (SelfCheckEnabled()) {
		t_check_pending = true;
	}
	if (!StageAEnabled()) {
		return;
	}
	CloseChunk();
	auto& c    = t_chunk;
	auto& slot = g_prepared[SlotOf(address)];
	c          = {};
	c.open     = true;
	c.start    = g_epoch.load(std::memory_order_relaxed);
	if (slot.address.load(std::memory_order_acquire) == address) {
		c.prepared = true;
		c.prep     = slot.epoch.load(std::memory_order_relaxed);
		// Used once: a buffer reused later must be walked again.
		slot.address.store(0, std::memory_order_relaxed);
	}
	NoteRead(address, size_bytes);
}

void StreamEnd() {
	if (StageAEnabled()) {
		CloseChunk();
	}
}

void NoteDraw() {
	if (StageAEnabled() && t_chunk.open) {
		t_chunk.draws++;
	}
}

void NoteRead(uint64_t vaddr, uint64_t size) {
	if (!StageAEnabled() || !t_chunk.open || !t_chunk.prepared || size == 0) {
		return;
	}
	const bool     timed = Sampled();
	const uint64_t t0    = timed ? NowNs() : 0;
	Classify(g_wide.load(std::memory_order_relaxed));
	const uint64_t first = vaddr >> ZoneBits;
	uint64_t       last  = (vaddr + size - 1u) >> ZoneBits;
	if (last - first >= WideZones) {
		last = first + WideZones - 1u; // a read that large is a buffer bound whole; its head
	}
	for (uint64_t z = first; z <= last && t_chunk.verdict == 0; z++) {
		Classify(g_zones[z & TableMask].load(std::memory_order_relaxed));
	}
	Add(Reads);
	if (timed) {
		Add(ReadCostNs, NowNs() - t0);
		Add(ReadSamples);
	}
}

void Report(uint64_t frames, double seconds) {
	if (SelfCheckEnabled()) {
		const uint64_t checked = g_checked.load(std::memory_order_relaxed);
		const uint64_t differ  = g_differ.load(std::memory_order_relaxed);
		Log::WriteUnmuted(fmt::format(
		    "MT GPU stage B ({:.0f} s): draws comprobados {}, con diferencias {} (total {} / {})\n",
		    seconds, checked - g_checked_last, differ - g_differ_last, checked, differ));
		g_checked_last = checked;
		g_differ_last  = differ;
	}
	if (!StageAEnabled()) {
		return;
	}
	std::array<uint64_t, StatCount> d {};
	for (uint32_t i = 0; i < StatCount; i++) {
		const auto now = g_stats[i].load(std::memory_order_relaxed);
		d[i]           = now - g_last[i];
		g_last[i]      = now;
	}
	const auto pct = [&](Stat s) {
		return d[Chunks] == 0 ? 0.0 : 100.0 * static_cast<double>(d[s]) / static_cast<double>(d[Chunks]);
	};
	const auto avg = [&](Stat cost, Stat samples) {
		return d[samples] == 0 ? 0.0 : static_cast<double>(d[cost]) / static_cast<double>(d[samples]);
	};
	const double ns = avg(WriteCostNs, WriteSamples) * static_cast<double>(d[Writes]) +
	                  avg(ReadCostNs, ReadSamples) * static_cast<double>(d[Reads]);
	const double us_frame = frames == 0 ? 0.0 : ns / 1000.0 / static_cast<double>(frames);
	Log::WriteUnmuted(fmt::format(
	    "MT GPU stage A ({:.0f} s): trozos con draws {}, aceptables {:.1f} %, rechazados por: "
	    "escritura GPU {:.1f} % / escritura host {:.1f} % / imagen retirada {:.1f} %, inciertos "
	    "{:.1f} %, sin preparar {:.1f} %, coste del registro {:.1f} us/fotograma ({} escrituras, "
	    "{} lecturas, {:.0f}/{:.0f} ns c/u)\n",
	    seconds, d[Chunks], pct(Accepted), pct(RejectGpu), pct(RejectHost), pct(RejectImage),
	    pct(Uncertain), pct(Unprepared), us_frame, d[Writes], d[Reads],
	    avg(WriteCostNs, WriteSamples), avg(ReadCostNs, ReadSamples)));
}

} // namespace Libs::Graphics::MtGpu
