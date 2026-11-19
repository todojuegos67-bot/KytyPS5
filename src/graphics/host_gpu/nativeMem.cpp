#include "graphics/host_gpu/nativeMem.h"

#include "common/logging/log.h"
#include "common/perfStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics::NativeMem {

namespace {

constexpr uint64_t ChunkBytes  = 0x200000;
constexpr int      ChunkFlags  = 0x90; // GUEST_MAP_FIXED | GUEST_MAP_NO_OVERWRITE
constexpr uint64_t WindowBytes = uint64_t {1} << 40u;
constexpr uint64_t Granules    = WindowBytes / ChunkBytes;

std::array<std::atomic<uint64_t>, Granules / 64> g_pool {};
std::atomic<uint64_t>                            g_pool_chunks {0};

uint64_t ReadEnv(const char* name, uint64_t fallback, uint64_t low, uint64_t high) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	const auto parsed = std::strtoull(value, nullptr, 10);
	return parsed < low || parsed > high ? fallback : parsed;
}

struct Counters {
	std::atomic<uint64_t> faults {0};
	std::atomic<uint64_t> released {0};
	std::atomic<uint64_t> marked {0};
	std::atomic<uint64_t> refreshes {0};
	std::atomic<uint64_t> wait_refreshes {0};
	std::atomic<uint64_t> wait_skipped {0};
	std::atomic<uint64_t> redirtied {0};
	std::atomic<uint64_t> written {0};
	std::atomic<uint64_t> dropped {0};
	std::atomic<uint64_t> hashed {0};
	std::atomic<uint64_t> hash_ns {0};
	std::atomic<uint64_t> kept_now {0};
	std::atomic<uint64_t> kept_peak {0};
};
Counters              g_counters;
std::atomic<uint64_t> g_report_ms {0};

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_MEM");
		const bool  on    = value != nullptr && value[0] == '1' && value[1] == '\0';
		if (on) {
			Log::WriteUnmuted(fmt::sprintf("Native mem: on (KYTY_NATIVE_MEM=1): GT7's 2 MiB GPU pool chunks are written "
			     "without page faults (a page written since its upload, by its hash, is "
			     "uploaded again after each slice and passed wait); a kept page unchanged "
			     "%" PRIu64 " us is tracked again (KYTY_NATIVE_MEM_IDLE_US), at most %zu pages "
			     "kept (KYTY_NATIVE_MEM_CAP)\n",
			     IdleMicros(), PageCap()));
		}
		return on;
	}();
	return enabled;
}

uint64_t IdleMicros() {
	static const auto micros = ReadEnv("KYTY_NATIVE_MEM_IDLE_US", 3000, 100, 1000000);
	return micros;
}

size_t PageCap() {
	static const auto cap = static_cast<size_t>(ReadEnv("KYTY_NATIVE_MEM_CAP", 1024, 16, 262144));
	return cap;
}

bool IsPoolMapping(uint64_t vaddr, uint64_t len, int flags, uint64_t alignment,
                   bool gpu_read_write) {
	return Enabled() && gpu_read_write && len == ChunkBytes && alignment == ChunkBytes &&
	       flags == ChunkFlags && vaddr % ChunkBytes == 0 && vaddr != 0 &&
	       vaddr + len <= WindowBytes;
}

void AddPool(uint64_t vaddr, uint64_t len) {
	if (len == 0 || vaddr >= WindowBytes) {
		return;
	}
	const auto end = std::min(vaddr + len, WindowBytes);
	for (auto granule = vaddr / ChunkBytes; granule < (end + ChunkBytes - 1) / ChunkBytes;
	     granule++) {
		const auto bit = uint64_t {1} << (granule % 64);
		if ((g_pool[granule / 64].fetch_or(bit, std::memory_order_acq_rel) & bit) == 0) {
			g_pool_chunks.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

void RemovePool(uint64_t vaddr, uint64_t len) {
	if (len == 0 || vaddr >= WindowBytes || g_pool_chunks.load(std::memory_order_relaxed) == 0) {
		return;
	}
	const auto end = std::min(vaddr + len, WindowBytes);
	// Only chunks wholly inside the range: a partial unmap leaves the rest of the chunk mapped,
	// and InPool is a hint for the fault handler (a fault outside guest GPU memory is refused
	// before it is asked).
	for (auto granule = (vaddr + ChunkBytes - 1) / ChunkBytes; granule < end / ChunkBytes;
	     granule++) {
		const auto bit = uint64_t {1} << (granule % 64);
		if ((g_pool[granule / 64].fetch_and(~bit, std::memory_order_acq_rel) & bit) != 0) {
			g_pool_chunks.fetch_sub(1, std::memory_order_relaxed);
		}
	}
}

bool InPool(uint64_t vaddr) noexcept {
	if (vaddr >= WindowBytes) {
		return false;
	}
	const auto granule = vaddr / ChunkBytes;
	return (g_pool[granule / 64].load(std::memory_order_relaxed) >> (granule % 64) & 1u) != 0;
}

void NoteFault(bool released_block, size_t kept) noexcept {
	g_counters.faults.fetch_add(1, std::memory_order_relaxed);
	if (released_block) {
		g_counters.released.fetch_add(1, std::memory_order_relaxed);
	}
	g_counters.marked.fetch_add(kept, std::memory_order_relaxed);
}

void NoteRefresh(const Refresh& refresh) noexcept {
	auto& c = g_counters;
	if (refresh.skipped) {
		c.wait_skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	(refresh.at_wait ? c.wait_refreshes : c.refreshes).fetch_add(1, std::memory_order_relaxed);
	c.redirtied.fetch_add(refresh.redirtied, std::memory_order_relaxed);
	c.written.fetch_add(refresh.written, std::memory_order_relaxed);
	c.dropped.fetch_add(refresh.dropped, std::memory_order_relaxed);
	c.hashed.fetch_add(refresh.hashed, std::memory_order_relaxed);
	c.hash_ns.fetch_add(refresh.hash_ns, std::memory_order_relaxed);
	c.kept_now.store(refresh.kept, std::memory_order_relaxed);
	if (refresh.kept > c.kept_peak.load(std::memory_order_relaxed)) {
		c.kept_peak.store(refresh.kept, std::memory_order_relaxed);
	}
}

void MaybeReport(uint64_t now_ms) {
	auto last = g_report_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		g_report_ms.store(now_ms, std::memory_order_relaxed);
		return;
	}
	if (now_ms - last < 10000) {
		return;
	}
	g_report_ms.store(now_ms, std::memory_order_relaxed);
	auto&        c       = g_counters;
	const double seconds = static_cast<double>(now_ms - last) / 1000.0;
	const auto   rate    = [&](std::atomic<uint64_t>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) / seconds;
	};
	const double faults    = rate(c.faults);
	const double released  = rate(c.released);
	const double marked    = rate(c.marked);
	const double slices    = rate(c.refreshes);
	const double waits     = rate(c.wait_refreshes);
	const double skipped   = rate(c.wait_skipped);
	const double redirtied = rate(c.redirtied);
	const double written   = rate(c.written);
	const double dropped   = rate(c.dropped);
	const double hashed    = rate(c.hashed);
	const double hash_ms   = rate(c.hash_ns) / 1e6;
	const auto   peak      = c.kept_peak.exchange(0, std::memory_order_relaxed);
	// "faults avoided": kept pages found written (hash changed) after their last upload: each of
	// those writes was a write fault before (a lower bound: a page written several times between
	// two refreshes faulted once at most then too). "uploaded at most": the pages made CPU-dirty
	// again, 4 KB each, which the next use of each uploads (only those inside a cached buffer are
	// copied; the "GPU phases" line's "buffer uploads" has the bytes really copied).
	Log::WriteUnmuted(fmt::sprintf("Native mem (%.0f s): pool %" PRIu64 " chunks (%" PRIu64 " MB) | pool write faults "
	     "%.0f/s (%.0f released their 64 KB block), %.0f pages kept/s | kept now %" PRIu64
	     " (peak %" PRIu64 ") | faults avoided %.0f/s | refreshes %.0f/s at slices, %.0f/s at "
	     "waits (%.0f/s waits not released by the CPU) | made dirty again %.0f pages/s, uploaded "
	     "at most %.1f MB/s | tracked again %.0f/s | hashed %.0f pages/s, %.1f ms/s\n",
	     seconds, g_pool_chunks.load(std::memory_order_relaxed),
	     g_pool_chunks.load(std::memory_order_relaxed) * 2u, faults, released, marked,
	     c.kept_now.load(std::memory_order_relaxed), peak, written, slices, waits, skipped,
	     redirtied, redirtied * 4096.0 / (1024.0 * 1024.0), dropped, hashed, hash_ms));
}

} // namespace Libs::Graphics::NativeMem
