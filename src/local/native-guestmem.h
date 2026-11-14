#pragma once
// KYTY_NATIVE_GUESTMEM=1 (default off): the emulator's own large reads of guest memory through the backing alias never
// touch 64 KiB granules nobody wrote (src/local/fresh-memory.h: their bytes are zeros). They read as zeros instead.
//
// Why: on a native PC port a GPU-only resource (a depth array, a storage image, a GPU-written buffer) has no bytes in
// system RAM. Here every such resource also has guest memory in the 13.8 GiB page-file section. When the emulator reads
// it through its alias view (to hash a layer, to stage an upload), each never-written page becomes a physical page of
// zeros in the working set (and, with KYTY_GUEST_LAZY_COMMIT, a commit). The 30 s "GPU residency" lines of the
// 2026-11-14 field logs show 410-515 MiB "resident but never written" that only the emulator's alias holds, on a 16 GB
// PC at 95-97% RAM (see /home/claude/static/native-guestmem.md).
//
// The rule is the one the async upload path already applies by default (bufferCache.cpp PushUploadCopy): a granule
// that is not marked written holds zeros. This module extends it to the reads that did not apply it yet:
//   - HashImage:    TextureCache's HashGuestRange (depth-array layers rendered / refreshed, partial image hashes);
//   - ImageRead:    the synchronous image staging read (ObtainBufferForImage when the async upload is not used);
//   - BufferRead:   the synchronous buffer upload read (ReadUploadSource);
//   - SparseUpload: image uploads over partly unmapped ranges (the mapped pieces went to the worker unchecked).
// Reads under 64 KiB (one granule) are left as they were: they touch at most two granules and are the hot path.
// The bytes produced are identical to the read's whenever the fresh-memory invariant holds (the same invariant the
// default async upload relies on); the hash is XXH3's streaming form over the same bytes, equal to the one-shot hash.
// Nothing is freed, decommitted, trimmed or moved; no guest byte is written.

#include "fresh-memory.h"

#include <xxhash.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace NativeGuestMem {

enum Site : uint8_t { HashImage = 0, ImageRead = 1, BufferRead = 2, SparseUpload = 3, Sites = 4 };

inline const char* SiteName(Site site) {
	switch (site) {
		case HashImage: return "image hash";
		case ImageRead: return "image staging read";
		case BufferRead: return "buffer upload read";
		case SparseUpload: return "sparse image upload";
		default: return "?";
	}
}

constexpr uint64_t MinBytes = FreshMemory::Granule;

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_NATIVE_GUESTMEM");
		const bool  set   = value != nullptr && std::string_view(value) == "1";
		if (set && !FreshMemory::Enabled())
			std::printf("Native guest memory: KYTY_NATIVE_GUESTMEM=1 needs fresh memory (KYTY_FRESH_MEMORY=0 set): off\n");
		return set && FreshMemory::Enabled();
	}();
	return on;
}

// Bytes since start, per site: never-written bytes produced as zeros without a read (skipped), and bytes of reads of
// MinBytes or more that went to memory as before (read).
struct Counters {
	std::atomic<uint64_t> skipped[Sites] {};
	std::atomic<uint64_t> read[Sites] {};
};
inline Counters g_counters;

inline void Count(Site site, uint64_t skipped, uint64_t read) {
	if (site >= Sites) return;
	if (skipped != 0) g_counters.skipped[site].fetch_add(skipped, std::memory_order_relaxed);
	if (read != 0) g_counters.read[site].fetch_add(read, std::memory_order_relaxed);
}

// The runs (offset, bytes, fresh) of backing-view bytes [backing, backing + size), appended to `runs` at `base`; true
// when one of them is fresh.
inline bool AppendRuns(const uint8_t* backing, uint64_t size, uint64_t base,
                       std::vector<std::tuple<uint64_t, uint64_t, bool>>* runs) {
	bool any = false;
	(void)FreshMemory::ForEachBackingRun(backing, size, [&](uint64_t offset, uint64_t bytes, bool fresh) {
		runs->emplace_back(base + offset, bytes, fresh);
		any = any || fresh;
	});
	return any;
}

// Reads guest [vaddr, vaddr + size) into `data` with `read(vaddr, data, size)` (TryReadBacking-like: true when done),
// never-written granules as zeros without reading them. Only when the whole range is mapped (one piece per guest
// mapping, Memory::TryGetBackingPieces); otherwise, or when nothing in it is fresh, or off, one call of `read` as before.
template <typename ReadFn>
bool Read(uint64_t vaddr, void* data, uint64_t size, Site site, ReadFn&& read) {
	if (data == nullptr || size < MinBytes || !Enabled()) return read(vaddr, data, size);
	thread_local std::vector<std::pair<const uint8_t*, uint64_t>>       pieces;
	thread_local std::vector<std::tuple<uint64_t, uint64_t, bool>> runs;
	runs.clear();
	bool any = false;
	if (Libs::LibKernel::Memory::TryGetBackingPieces(vaddr, size, pieces)) {
		uint64_t offset = 0;
		for (const auto& [backing, bytes]: pieces) {
			any = AppendRuns(backing, bytes, offset, &runs) || any;
			offset += bytes;
		}
		any = any && offset == size;
	}
	if (!any) {
		Count(site, 0, size);
		return read(vaddr, data, size);
	}
	auto*    out     = static_cast<uint8_t*>(data);
	uint64_t skipped = 0, done = 0;
	for (const auto& [offset, bytes, fresh]: runs) {
		if (fresh) {
			std::memset(out + offset, 0, bytes);
			skipped += bytes;
		} else {
			if (!read(vaddr + offset, out + offset, bytes)) {
				Count(site, skipped, done);
				return false;
			}
			done += bytes;
		}
	}
	Count(site, skipped, done);
	return true;
}

// XXH3_64bits of backing-view bytes [backing, backing + size), never-written granules hashed as the zeros they hold
// without reading them (XXH3's streaming state: the same value as the one-shot hash of the same bytes).
inline uint64_t HashBacking(const uint8_t* backing, uint64_t size, Site site) {
	if (backing == nullptr || size < MinBytes || !Enabled()) return XXH3_64bits(backing, static_cast<size_t>(size));
	thread_local std::vector<std::tuple<uint64_t, uint64_t, bool>> runs;
	runs.clear();
	if (!AppendRuns(backing, size, 0, &runs)) {
		Count(site, 0, size);
		return XXH3_64bits(backing, static_cast<size_t>(size));
	}
	struct State {
		XXH3_state_t* state = XXH3_createState();
		~State() { XXH3_freeState(state); }
	};
	thread_local State                   owner;
	static const std::vector<uint8_t>    zeros(FreshMemory::Granule, 0);
	if (owner.state == nullptr || XXH3_64bits_reset(owner.state) == XXH_ERROR) {
		Count(site, 0, size);
		return XXH3_64bits(backing, static_cast<size_t>(size));
	}
	uint64_t skipped = 0;
	for (const auto& [offset, bytes, fresh]: runs) {
		if (!fresh) {
			(void)XXH3_64bits_update(owner.state, backing + offset, static_cast<size_t>(bytes));
			continue;
		}
		for (uint64_t left = bytes; left != 0;) {
			const auto chunk = std::min<uint64_t>(left, zeros.size());
			(void)XXH3_64bits_update(owner.state, zeros.data(), static_cast<size_t>(chunk));
			left -= chunk;
		}
		skipped += bytes;
	}
	Count(site, skipped, size - skipped);
	return XXH3_64bits_digest(owner.state);
}

// One line with the PC memory line (TextureCache), when on.
inline void Report() {
	if (!Enabled()) return;
	constexpr uint64_t MiB = uint64_t {1} << 20u;
	std::printf("Native guest memory (KYTY_NATIVE_GUESTMEM=1, MiB since start): never-written guest bytes given as zeros "
	            "instead of read through the alias / read as before:");
	for (uint8_t site = 0; site < Sites; site++) {
		std::printf("%s %s %llu/%llu", site == 0 ? "" : ",", SiteName(static_cast<Site>(site)),
		            static_cast<unsigned long long>(g_counters.skipped[site].load(std::memory_order_relaxed) / MiB),
		            static_cast<unsigned long long>(g_counters.read[site].load(std::memory_order_relaxed) / MiB));
	}
	std::printf("\n");
}

} // namespace NativeGuestMem
