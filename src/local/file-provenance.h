#pragma once
// File provenance census (stage 1: counts only, nothing changes; KYTY_PROVENANCE=0 turns it off, and it is off with
// KYTY_FRESH_MEMORY=0, whose write marks it relies on).
//
// Per 4 KiB page of the guest backing store (by backing offset, not guest address): whether its bytes came only from a
// read of a read-only game file (/app0, archives in it) and nothing wrote it since, and where in which file:
// valid:1 | discarded:1 | file_id:14 | file_byte_offset:48. Such a page could later be dropped from RAM and read again
// from the file. Recorded after the file reads into guest memory (APR reads, read/pread), pages wholly covered only;
// cleared by every write the emulator sees (the fresh-memory marks: write-backs, host writes into the backing view,
// first-write faults, mappings that let the CPU write unwatched; host overwrites; GPU writes noted by the buffer cache;
// frees and zeroing of backing). Not seen: guest CPU stores into pages already writable and unwatched (and host code
// writing guest memory without telling anyone) - the census's known gap. The verify (KYTY_PROVENANCE_VERIFY=1) measures
// it: intact pages sampled, read again from the file and compared.
//
// A second, conservative figure leaves out pages whose 64 KiB granule got any write mark after they were recorded.
//
// Hot paths: atomics only (a relaxed load per page cleared, and a store per granule).

#include <atomic>
#include <cstdint>
#include <string>

namespace FileProvenance {

constexpr uint64_t PageBits     = 12;
constexpr uint64_t Page         = uint64_t {1} << PageBits;
constexpr uint64_t GranuleBits  = 16; // (the fresh-memory granule)
constexpr uint64_t ValidBit     = uint64_t {1} << 63u;
constexpr uint64_t DiscardedBit = uint64_t {1} << 62u;
constexpr uint64_t IdShift      = 48;
constexpr uint64_t IdMask       = 0x3fff;
constexpr uint64_t OffsetMask   = (uint64_t {1} << 48u) - 1;

struct State {
	std::atomic<uint64_t>* pages          = nullptr; // a word per 4 KiB page of the backing store
	std::atomic<uint32_t>* granule_mark   = nullptr; // the sequence at the granule's last write mark (0: none)
	std::atomic<uint32_t>* granule_record = nullptr; // the sequence of the oldest record among its intact pages
	uint64_t               base = 0, size = 0, page_count = 0, granules = 0;
	std::atomic<uint32_t>  sequence {1};
	std::atomic<int64_t>   valid_pages {0};
	// Counters: pages recorded, pages cleared by a write, reads not recorded (file table full, offset past 48 bits).
	std::atomic<uint64_t>  recorded {0}, cleared {0}, unrecorded {0};
};

inline std::atomic<State*> g_state {nullptr};
inline std::atomic<bool>   g_off {false};
// Inside a file read being recorded: its own marks (preparing the range, faults of its copy) do not count as writes
// after the record for the conservative figure. (Its pages are still cleared: the record comes after.)
inline thread_local int t_reading = 0;

State* Init(); // file-provenance.cpp: nullptr while the backing store does not exist yet, or when off

inline State* Get() {
	if (auto* state = g_state.load(std::memory_order_acquire); state != nullptr) return state;
	if (g_off.load(std::memory_order_relaxed)) return nullptr;
	return Init();
}

// Pages [first, last) not intact any more (bits only: callable under any lock).
inline void ClearPages(State& state, uint64_t first, uint64_t last) {
	last = last < state.page_count ? last : state.page_count;
	for (uint64_t page = first; page < last; ++page) {
		auto& word = state.pages[page];
		if ((word.load(std::memory_order_relaxed) & ValidBit) == 0) continue;
		if ((word.fetch_and(~ValidBit, std::memory_order_acq_rel) & ValidBit) != 0) {
			state.valid_pages.fetch_sub(1, std::memory_order_relaxed);
			state.cleared.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

// Backing-view bytes [backing, backing + size) written: their pages cleared, their granules marked (bits only).
inline void ClearBacking(uint64_t backing, uint64_t size) {
	auto* state = Get();
	if (state == nullptr || size == 0 || backing < state->base || backing - state->base >= state->size) return;
	// (Nothing intact: nothing to clear, and a mark matters only to pages recorded before it.)
	if (state->valid_pages.load(std::memory_order_relaxed) <= 0) return;
	const uint64_t begin = backing - state->base;
	const uint64_t end   = size > state->size - begin ? state->size : begin + size;
	ClearPages(*state, begin >> PageBits, (end + Page - 1) >> PageBits);
	if (t_reading != 0) return;
	const uint32_t sequence = state->sequence.load(std::memory_order_acquire);
	for (uint64_t granule = begin >> GranuleBits, last = (end - 1) >> GranuleBits; granule <= last && granule < state->granules;
	     ++granule) {
		state->granule_mark[granule].store(sequence, std::memory_order_release);
	}
}

// The same by backing offset (frees, zeroing).
inline void ClearBackingOffset(uint64_t backing_offset, uint64_t size) {
	auto* state = Get();
	if (state == nullptr) return;
	ClearBacking(state->base + backing_offset, size);
}

// Guest range [vaddr, vaddr + size) written (its bytes in the backing store; queries the kernel's mappings: not under
// the address-space lock).
void ClearGuestRange(uint64_t vaddr, uint64_t size);

// The id of a read-only game file at host path `path` (archive members "x.zar!/dir/file" too); 0 when off or the
// table is full. The caller decides that the file is a read-only game file.
uint16_t FileId(const std::string& path);

// After a successful copy of file `file_id`'s bytes [file_offset, file_offset + size) into guest range
// [vaddr, vaddr + size): its wholly covered pages recorded.
void Record(uint64_t vaddr, uint64_t size, uint16_t file_id, uint64_t file_offset);

// Texture cache and buffer cache registrations (for the overlap figures).
enum class Kind : uint8_t { Texture, Buffer };
void NoteRange(Kind kind, uint64_t vaddr, uint64_t size, bool registered);

class ReadScope {
public:
	ReadScope() { ++t_reading; }
	~ReadScope() { --t_reading; }
	ReadScope(const ReadScope&)            = delete;
	ReadScope& operator=(const ReadScope&) = delete;
};

} // namespace FileProvenance
