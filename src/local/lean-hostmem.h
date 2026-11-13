#pragma once
// KYTY_LEAN_HOSTMEM=1 (default off until reviewed): less system RAM held for the emulator outside its working set.
//
// On a 16 GB PC Windows showed ~3.5-4 GB "in use" beyond the game's working set and the rest of the PC: memory the
// emulator causes but that is not in its working set - the modified page list (dirty pages of the 13.8 GB page-file
// section of guest memory, trimmed from the working set and not yet written to the page file) and system RAM the GPU
// driver holds for Vulkan memory outside video memory. With this on:
//
//   1. Writes into the guest backing store skip what would not change it. A write-back of GPU results
//      (TryWriteBacking), a CPU-owned linear copy (TryWriteCpuBacking) and the zero fill of reused backing
//      (ZeroRange) compare each 4 KiB page first and write only pages that differ. An unchanged page written anyway
//      becomes dirty: once trimmed it waits on the modified list ("in use") until Windows writes it to the page file.
//      An equal page left alone stays clean: trimmed, it goes to the standby list (available).
//   2. Upload rings the CPU only writes in order (the shader-constant ring, the table ring, the speculative
//      translators' rings) are made in device-local host-visible memory where the GPU has a large BAR (ReBAR / SAM):
//      video memory instead of system RAM the driver locks (streamBuffer.cpp, LeanUploadUsage).
//   3. The image upload ring is 128 MiB when KYTY_STAGING_RING_MB is not set (bufferCache.cpp).
//   4. On a GPU short of video memory, textures are no longer sent to the driver's plain memory type at half the
//      budget (vma.cpp, GraphicContext::CreateImage): that type lives in system RAM once video memory is full (2.6 GB
//      of it, private commit +4.6 GB, on a simulated 8 GB card); textures stay device-local under the caches' budget
//      and go to system memory only when video memory is really full, as with KYTY_LOWMEM_VRAM=1.
//
// Nothing is freed, resized or moved at runtime; every size is decided once at start.
//
// Dropped pages: a page dropped with DiscardVirtualMemory or MEM_RESET (the RAM guard's FlushFreed of released guest
// memory, the texture offload's DiscardRange) keeps its old bytes only until Windows takes it: read and not written it
// stays droppable, and later reads zeros. A write skipped there because the old bytes matched would be lost (zeros in
// place of the data: black textures, broken structures). Such pages are marked (NoteDiscarded) and written in full by
// the next lean write, which unmarks them; skipping resumes after that.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace LeanHostMem {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_LEAN_HOSTMEM");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

// Bytes not written because they were already equal (copies) or zero (fills), and bytes asked to be written (on only).
inline std::atomic<uint64_t> g_skipped_copy {0};
inline std::atomic<uint64_t> g_skipped_zero {0};
inline std::atomic<uint64_t> g_written {0};

constexpr size_t Page = 4096;

// The guest backing store's bytes [base, base + size) and one bit per 4 KiB page of it: dropped (DiscardVirtualMemory /
// MEM_RESET) and not written by a lean write since. Only made when on (SetStore, by the store once it is mapped).
struct Discarded {
	std::atomic<uint64_t> base {0};
	uint64_t              size  = 0;
	std::atomic<uint64_t>* words = nullptr; // (never freed: the store lives for the whole run)
};
inline Discarded g_discarded;

inline void SetStore(const void* base, uint64_t size) {
	if (!Enabled() || base == nullptr || size == 0 || g_discarded.base.load(std::memory_order_relaxed) != 0) return;
	const uint64_t words = (size / Page + 63) / 64;
	g_discarded.words    = new std::atomic<uint64_t>[words] {};
	g_discarded.size     = size;
	g_discarded.base.store(reinterpret_cast<uint64_t>(base), std::memory_order_release);
}

// Pages of [address, address + size) dropped (any page it touches). Before the pages can be written again (the drop
// runs under the lock that keeps the memory free, or with the range protected).
inline void NoteDiscarded(const void* address, uint64_t size) {
	const uint64_t base = g_discarded.base.load(std::memory_order_acquire);
	if (base == 0 || size == 0) return;
	const auto at = reinterpret_cast<uint64_t>(address);
	if (at < base || at - base >= g_discarded.size) return;
	const uint64_t first = (at - base) / Page;
	const uint64_t last  = (std::min(at - base + size, g_discarded.size) - 1) / Page;
	for (uint64_t page = first; page <= last; ++page)
		g_discarded.words[page / 64].fetch_or(uint64_t {1} << (page % 64), std::memory_order_release);
}

// Whether the page holding `address` was dropped and not written since; `unmark`: it is written in full now.
inline bool Dropped(const uint8_t* address, bool unmark) {
	const uint64_t base = g_discarded.base.load(std::memory_order_acquire);
	const auto     at   = reinterpret_cast<uint64_t>(address);
	if (base == 0 || at < base || at - base >= g_discarded.size) return false;
	const uint64_t page = (at - base) / Page;
	const uint64_t bit  = uint64_t {1} << (page % 64);
	auto&          word = g_discarded.words[page / 64];
	if ((word.load(std::memory_order_acquire) & bit) == 0) return false;
	if (unmark) word.fetch_and(~bit, std::memory_order_acq_rel);
	return true;
}

[[nodiscard]] inline bool IsZero(const uint8_t* data, size_t size) {
	size_t i = 0;
	for (; i + 32 <= size; i += 32) {
		uint64_t w[4];
		std::memcpy(w, data + i, sizeof(w));
		if ((w[0] | w[1] | w[2] | w[3]) != 0) return false;
	}
	for (; i < size; ++i)
		if (data[i] != 0) return false;
	return true;
}

// memcpy(destination, source, size), writing only the 4 KiB pages of destination whose bytes differ.
inline void CopyChanged(uint8_t* destination, const uint8_t* source, size_t size) {
	uint64_t skipped = 0;
	while (size != 0) {
		const size_t room = Page - (reinterpret_cast<uintptr_t>(destination) & (Page - 1u));
		const size_t n    = room < size ? room : size;
		// (A dropped page is written whatever it holds: unmarked once written whole.)
		if (Dropped(destination, n == Page) || std::memcmp(destination, source, n) != 0) std::memcpy(destination, source, n);
		else skipped += n;
		destination += n;
		source += n;
		size -= n;
	}
	if (skipped != 0) g_skipped_copy.fetch_add(skipped, std::memory_order_relaxed);
}

// memset(destination, 0, size), writing only the 4 KiB pages that are not zero already.
inline void ZeroChanged(uint8_t* destination, size_t size) {
	uint64_t skipped = 0;
	while (size != 0) {
		const size_t room = Page - (reinterpret_cast<uintptr_t>(destination) & (Page - 1u));
		const size_t n    = room < size ? room : size;
		if (Dropped(destination, n == Page) || !IsZero(destination, n)) std::memset(destination, 0, n);
		else skipped += n;
		destination += n;
		size -= n;
	}
	if (skipped != 0) g_skipped_zero.fetch_add(skipped, std::memory_order_relaxed);
}

// The guest backing store's writes (memoryAddressSpace.inc): the lean versions when on, plain memcpy/memset otherwise.
// (Counted only when on: the old path is exactly the memcpy / memset it was.)
inline void BackingCopy(uint8_t* destination, const void* source, size_t size) {
	if (!Enabled()) {
		std::memcpy(destination, source, size);
		return;
	}
	g_written.fetch_add(size, std::memory_order_relaxed);
	CopyChanged(destination, static_cast<const uint8_t*>(source), size);
}
inline void BackingZero(uint8_t* destination, size_t size) {
	if (!Enabled()) {
		std::memset(destination, 0, size);
		return;
	}
	g_written.fetch_add(size, std::memory_order_relaxed);
	ZeroChanged(destination, size);
}

} // namespace LeanHostMem
