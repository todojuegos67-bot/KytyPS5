#include "graphics/host_gpu/nativeTex.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsNativeTex.h"
#include "loader/demonsSoulsNativeTexStream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// KYTY_NATIVE_TEX (nativeTex.h).
//
// Locks: one mutex `m` guards the maps, the ring and the page manager watcher changes they imply (applied under it, so the
// watcher count of a page is 0 or 1 here). Order: the texture cache's lock -> m -> page manager. `m` is never held across
// a GPU wait, render_sync or a host-write notification. A write-back of entries to guest memory (slow: a GPU download)
// runs with `m` released: those entries are marked `inflight` and every operation over them (and a fault on their pages)
// waits for `done` first.
//
// Game-thread callbacks that may run inside the engine's allocator lock (OnTextureRelease, OnTextureMove: Loader's
// demonsSoulsNativeTexStream.h) never wait for the render thread: a release drops the ranges at once (only `m` and the
// page manager's locks, neither of which is ever held while waiting for a game lock), a move shares ring bytes or records
// "bytes of image X" for the destination (the GPU copy is made when an image needs them: Rehome).
//
// Who writes authoritative bytes back (MaterializeAny): Host::run_render's thread classes. The one class that can do it
// neither inline nor through the render thread - the async upload worker, a deferred operation, a thread holding the
// texture cache's lock off the render thread - only ever reads guest memory through host pointers the render thread
// resolved (and guarded: BeforeGuestRead) for it; a touch of bytes an image holds by one of them is a broken invariant
// and stops the emulator with a message (never stale bytes). Bytes in the ring it gets at once (a CPU copy).

namespace Libs::Graphics::NativeTex {

namespace {

constexpr uint64_t Page       = 4096; // TRACKER_PAGE_SIZE
constexpr uint64_t MinPiece   = 4096; // smaller loads go to guest memory as before
constexpr uint64_t RingAlign  = 256;  // the tiler's source alignment
constexpr size_t   MaxEntries = size_t {1} << 18u;
constexpr uint64_t GranuleBits = 21, GranuleSpace = uint64_t {1} << 40u;
constexpr size_t   GranuleWords = (GranuleSpace >> GranuleBits) / 64;

struct Entry {
	uint64_t end         = 0;
	bool     owned       = false;
	bool     undef       = false; // (not owned: no bytes; nativeTex.h Piece::undefined)
	bool     inflight    = false;
	uint64_t ring_offset = 0;
	uint64_t image = 0, serial = 0, src = 0;
	uint64_t ready_tick = 0;
	uint32_t alloc      = 0;
	uint64_t flip       = 0; // (when it came: spill age)
};

struct Alloc {
	uint64_t offset = 0, size = 0;
	uint32_t refs      = 0; // ring entries over it (+1 while its creator fills it)
	uint64_t free_tick = 0; // GPU work reading or writing it (0: none pending)
};

enum Why : uint32_t { WhyTouch, WhyHostWrite, WhyRead, WhyImage, WhyGone, WhySpill, WhyMove, WhyOther, WhyCount };
constexpr const char* WhyNames[WhyCount] = {"touch", "host-write", "emulator-read", "image", "image-gone", "spill", "move", "other"};

struct State {
	bool                      active = false;
	Host                      host {};
	std::unique_ptr<Buffer>   ring;
	uint8_t*                  mapped   = nullptr;
	uint64_t                  capacity = 0;
	std::mutex                m;
	std::condition_variable   done;
	std::map<uint64_t, Entry> entries; // by guest start; disjoint
	std::unordered_map<uint64_t, uint32_t> owned_count; // image serial -> owned entries
	std::deque<Alloc>         allocs;  // FIFO, front = oldest
	uint32_t                  alloc_first = 0;
	uint64_t                  tail = 0, used = 0, owned_bytes = 0, flip = 0;
	uint64_t                  ready_entries = 0; // entries a GPU copy still writes (ready_tick != 0): Pump's walk
	std::atomic<uint64_t>     lo {~uint64_t {0}}, hi {0};
	// 2 MiB granules any range ever touched (a lock-free filter for BeforeGuestRead; never cleared, 2^40 bytes).
	std::unique_ptr<std::atomic<uint64_t>[]> granules = std::make_unique<std::atomic<uint64_t>[]>(GranuleWords);
	// Counters (cumulative).
	std::atomic<uint64_t> routed_bytes {0}, routed_ranges {0}, refused {0}, refused_bytes {0};
	std::atomic<uint64_t> uploads {0}, upload_bytes {0}, rehomes {0}, rehome_bytes {0};
	std::atomic<uint64_t> moves_shared {0}, moves_guest {0}, releases_count {0}, release_bytes {0};
	std::atomic<uint64_t> fallback[WhyCount] {};
	std::atomic<uint64_t> fallback_bytes[WhyCount] {};
	std::atomic<uint64_t> lost {0}, lost_bytes {0}, commits {0}, commit_bytes {0}, decommits {0}, declined {0}, declined_bytes {0};
	std::chrono::steady_clock::time_point last_log {};
};

State& S() {
	static State state;
	return state;
}

thread_local bool t_suppress_host_write = false; // (our own notifications)

struct Delta {
	std::vector<std::pair<uint64_t, uint64_t>> on, off;
};

uint64_t PageDown(uint64_t v) { return v & ~(Page - 1); }
uint64_t PageUp(uint64_t v) { return (v + Page - 1) & ~(Page - 1); }

Why WhyOf(const char* why) {
	for (uint32_t i = 0; i < WhyCount; i++)
		if (why != nullptr && std::strcmp(why, WhyNames[i]) == 0) return static_cast<Why>(i);
	return WhyOther;
}

using Lock = std::unique_lock<std::mutex>;

std::map<uint64_t, Entry>::iterator FirstOverLocked(State& s, uint64_t begin) {
	auto it = s.entries.upper_bound(begin);
	if (it != s.entries.begin() && std::prev(it)->second.end > begin) --it;
	return it;
}

bool TouchedLocked(State& s, uint64_t begin, uint64_t end) {
	const auto it = FirstOverLocked(s, begin);
	return it != s.entries.end() && it->first < end;
}

bool InflightLocked(State& s, uint64_t begin, uint64_t end) {
	for (auto it = FirstOverLocked(s, begin); it != s.entries.end() && it->first < end; ++it)
		if (it->second.inflight) return true;
	return false;
}

void WaitInflight(State& s, Lock& lock, uint64_t begin, uint64_t end) {
	s.done.wait(lock, [&] { return !InflightLocked(s, begin, end); });
}

// Whether entries cover all of [begin, end) (texture memory: committed, loaded or held by an image).
bool CoveredLocked(State& s, uint64_t begin, uint64_t end) {
	uint64_t at = begin;
	for (auto it = FirstOverLocked(s, begin); it != s.entries.end() && it->first < end && at < end; ++it) {
		if (it->first > at) return false;
		at = std::max(at, it->second.end);
	}
	return at >= end;
}

Alloc* AllocLocked(State& s, uint32_t id) {
	const uint32_t index = id - s.alloc_first;
	return index < s.allocs.size() ? &s.allocs[index] : nullptr;
}

void ReleaseRefLocked(State& s, uint32_t id, uint64_t free_tick) {
	if (auto* a = AllocLocked(s, id); a != nullptr && a->refs != 0) {
		a->refs--;
		a->free_tick = std::max(a->free_tick, free_tick);
	}
}

void AddRefLocked(State& s, uint32_t id) {
	if (auto* a = AllocLocked(s, id)) a->refs++;
}

void ReclaimLocked(State& s) {
	while (!s.allocs.empty() && s.allocs.front().refs == 0 && s.allocs.front().free_tick == 0) {
		s.used -= s.allocs.front().size;
		s.allocs.pop_front();
		s.alloc_first++;
	}
	if (s.allocs.empty()) s.tail = 0;
}

// `size` bytes of the ring with one reference for the caller, or false (full).
bool AllocateLocked(State& s, uint64_t size, uint32_t* id, uint64_t* offset) {
	size = (size + RingAlign - 1) & ~(RingAlign - 1);
	if (size > s.capacity) return false;
	ReclaimLocked(s);
	uint64_t at = 0;
	if (!s.allocs.empty()) {
		const uint64_t head = s.allocs.front().offset;
		if (s.tail > head) {
			if (s.capacity - s.tail >= size) at = s.tail;
			else if (head >= size) at = 0; // (wrap: the end of the ring is skipped)
			else return false;
		} else {
			if (head - s.tail >= size) at = s.tail; // (wrapped: free space is [tail, head))
			else return false;
		}
	}
	s.allocs.push_back({at, size, 1, 0});
	*id     = s.alloc_first + static_cast<uint32_t>(s.allocs.size() - 1);
	*offset = at;
	s.tail  = at + size;
	s.used += size;
	return true;
}

// Watchers for [begin, end), an empty range about to get an entry: the pages no other entry touches.
void AddCoverageLocked(State& s, uint64_t begin, uint64_t end, Delta& d) {
	uint64_t pb = PageDown(begin), pe = PageUp(end);
	if (pb < pe && TouchedLocked(s, pb, pb + Page)) pb += Page;
	if (pb < pe && TouchedLocked(s, pe - Page, pe)) pe -= Page;
	if (pb < pe) d.on.emplace_back(pb, pe - pb);
}

// Watchers off for the pages of the removed ranges (sorted) no entry touches now.
void RemoveCoverageLocked(State& s, const std::vector<std::pair<uint64_t, uint64_t>>& removed, Delta& d) {
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	for (const auto& [b, e]: removed) {
		const uint64_t pb = PageDown(b), pe = PageUp(e);
		if (!runs.empty() && runs.back().second > pb) runs.back().second = std::max(runs.back().second, pe);
		else runs.emplace_back(pb, pe);
	}
	for (auto [pb, pe]: runs) {
		if (pb < pe && TouchedLocked(s, pb, pb + Page)) pb += Page;
		if (pb < pe && TouchedLocked(s, pe - Page, pe)) pe -= Page;
		if (pb < pe) d.off.emplace_back(pb, pe - pb);
	}
}

// The watcher changes, new ones first (a page passing from one entry to another is never open).
void ApplyLocked(State& s, const Delta& d) {
	if (s.host.watch == nullptr) return;
	for (const auto& [a, n]: d.on) s.host.watch(a, n, true);
	for (const auto& [a, n]: d.off) s.host.watch(a, n, false);
}

void NoteOwnedLocked(State& s, const Entry& e, uint64_t begin, bool add) {
	if (!e.owned) return;
	if (add) {
		s.owned_bytes += e.end - begin;
		s.owned_count[e.serial]++;
	} else {
		s.owned_bytes -= e.end - begin;
		if (auto it = s.owned_count.find(e.serial); it != s.owned_count.end() && --it->second == 0) s.owned_count.erase(it);
	}
}

// The entry containing x strictly inside split at x (not inflight: the callers waited).
void SplitAtLocked(State& s, uint64_t x) {
	auto it = s.entries.upper_bound(x);
	if (it == s.entries.begin()) return;
	--it;
	if (it->first >= x || it->second.end <= x) return;
	Entry right = it->second;
	if (right.owned) {
		right.src += x - it->first;
		s.owned_count[right.serial]++;
	} else if (!right.undef) {
		right.ring_offset += x - it->first;
		AddRefLocked(s, right.alloc);
	}
	it->second.end = x;
	s.entries.emplace(x, right);
}

void MarkGranules(State& s, uint64_t begin, uint64_t end) {
	if (begin >= GranuleSpace) return;
	const uint64_t last = (std::min(end, GranuleSpace) - 1) >> GranuleBits;
	for (uint64_t g = begin >> GranuleBits; g <= last; g++)
		s.granules[g / 64].fetch_or(uint64_t {1} << (g % 64), std::memory_order_release);
}

bool MaybeGranules(const State& s, uint64_t begin, uint64_t end) {
	if (end > GranuleSpace) return true;
	const uint64_t last = (end - 1) >> GranuleBits;
	for (uint64_t g = begin >> GranuleBits; g <= last; g++)
		if ((s.granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) return true;
	return false;
}

void InsertLocked(State& s, uint64_t begin, const Entry& e) {
	MarkGranules(s, begin, e.end);
	s.entries.emplace(begin, e);
	NoteOwnedLocked(s, e, begin, true);
	if (begin < s.lo.load(std::memory_order_relaxed)) s.lo.store(begin, std::memory_order_relaxed);
	if (e.end > s.hi.load(std::memory_order_relaxed)) s.hi.store(e.end, std::memory_order_relaxed);
}

// Everything over [begin, end) dropped (clipped; the callers waited for inflight ones), ring references released.
void DropLocked(State& s, uint64_t begin, uint64_t end, Delta& d) {
	SplitAtLocked(s, begin);
	SplitAtLocked(s, end);
	std::vector<std::pair<uint64_t, uint64_t>> removed;
	for (auto it = FirstOverLocked(s, begin); it != s.entries.end() && it->first < end;) {
		NoteOwnedLocked(s, it->second, it->first, false);
		if (!it->second.owned && !it->second.undef) ReleaseRefLocked(s, it->second.alloc, it->second.ready_tick);
		removed.emplace_back(it->first, it->second.end);
		it = s.entries.erase(it);
	}
	RemoveCoverageLocked(s, removed, d);
}

Piece PieceOf(uint64_t begin, const Entry& e) {
	Piece p {};
	p.begin       = begin;
	p.end         = e.end;
	p.owned       = e.owned;
	p.undefined   = e.undef;
	p.ring_offset = e.ring_offset;
	p.image       = e.image;
	p.serial      = e.serial;
	p.src         = e.src;
	p.ready_tick  = e.ready_tick;
	p.alloc       = e.alloc;
	return p;
}

void NoteLost(State& s, uint64_t begin, uint64_t end, const char* why) {
	s.lost.fetch_add(1, std::memory_order_relaxed);
	s.lost_bytes.fetch_add(end - begin, std::memory_order_relaxed);
	std::printf("NativeTex: MISSED 0x%llx..0x%llx (%s): guest bytes stale there\n", static_cast<unsigned long long>(begin),
	            static_cast<unsigned long long>(end), why);
}

// The pieces' bytes into guest memory (m not held). `render`: on the render thread (owned pieces downloaded from their
// images, GPU copies into the ring waited for); else ring bytes only, the others lost.
uint64_t WriteBack(State& s, const std::vector<Piece>& pieces, bool render) {
	uint64_t           bytes = 0;
	std::vector<Piece> owned;
	for (const auto& p: pieces) {
		if (p.owned) {
			owned.push_back(p);
			continue;
		}
		if (p.undefined) continue; // (no bytes: the guest's, whatever they are, are as good)
		if (p.ready_tick != 0) {
			if (!render) {
				NoteLost(s, p.begin, p.end, "a GPU copy in flight; touched by a thread that cannot wait for the GPU");
				std::fflush(stdout);
				std::abort();
			}
			if (s.host.tick_done != nullptr && !s.host.tick_done(p.ready_tick) && s.host.wait_tick != nullptr)
				s.host.wait_tick(p.ready_tick);
			s.ring->Invalidate(p.ring_offset, p.end - p.begin);
		}
		LibKernel::Memory::WriteBacking(p.begin, s.mapped + p.ring_offset, p.end - p.begin, "native-tex");
		bytes += p.end - p.begin;
	}
	if (owned.empty()) return bytes;
	if (render && s.host.materialize_owned != nullptr) {
		s.host.materialize_owned(owned);
		for (const auto& p: owned) bytes += p.end - p.begin;
	} else {
		// (A broken invariant, see the top of this file: stop rather than let stale bytes be read.)
		for (const auto& p: owned) NoteLost(s, p.begin, p.end, "its image holds it; touched by a thread that can neither record GPU work nor wait for the render thread");
		std::fflush(stdout);
		std::abort();
	}
	return bytes;
}

// Removes the (inflight) entries of `pieces` after their write-back, their pages tracked normally again.
void EraseWrittenLocked(State& s, const std::vector<Piece>& pieces) {
	Delta                                      d;
	std::vector<std::pair<uint64_t, uint64_t>> removed;
	for (const auto& p: pieces) {
		auto it = s.entries.find(p.begin);
		if (it == s.entries.end()) continue;
		NoteOwnedLocked(s, it->second, it->first, false);
		if (!it->second.owned && !it->second.undef) ReleaseRefLocked(s, it->second.alloc, 0);
		removed.emplace_back(it->first, it->second.end);
		s.entries.erase(it);
	}
	std::sort(removed.begin(), removed.end());
	RemoveCoverageLocked(s, removed, d);
	ApplyLocked(s, d);
	ReclaimLocked(s);
}

// Every entry intersecting [addr, addr + size) (whole entries; `serial` != 0: owned by that image only) written into
// guest memory and dropped. Not `render` (a thread that may not record GPU work): false, and nothing done, when some of
// them need the GPU (bytes an image holds, a GPU copy into the ring in flight).
bool MaterializeHere(uint64_t addr, uint64_t size, Why why, uint64_t serial, bool render) {
	auto&              s = S();
	std::vector<Piece> pieces;
	{
		Lock lock(s.m);
		WaitInflight(s, lock, addr, addr + size);
		for (auto it = FirstOverLocked(s, addr); it != s.entries.end() && it->first < addr + size; ++it) {
			if (serial != 0 && (!it->second.owned || it->second.serial != serial)) continue;
			if (!render && (it->second.owned || (!it->second.undef && it->second.ready_tick != 0))) return false;
		}
		for (auto it = FirstOverLocked(s, addr); it != s.entries.end() && it->first < addr + size; ++it) {
			if (serial != 0 && (!it->second.owned || it->second.serial != serial)) continue;
			it->second.inflight = true;
			pieces.push_back(PieceOf(it->first, it->second));
		}
	}
	if (pieces.empty()) return true;
	const uint64_t bytes = WriteBack(s, pieces, render);
	{
		Lock lock(s.m);
		EraseWrittenLocked(s, pieces);
	}
	s.done.notify_all();
	s.fallback[why].fetch_add(1, std::memory_order_relaxed);
	s.fallback_bytes[why].fetch_add(bytes, std::memory_order_relaxed);
	return true;
}

struct MaterializeJob {
	uint64_t addr, size;
	Why      why;
};

void MaterializeTrampoline(void* arg) {
	const auto* job = static_cast<const MaterializeJob*>(arg);
	(void)MaterializeHere(job->addr, job->size, job->why, 0, true);
}

// From any thread: where GPU work can be recorded and waited for (Host::run_render), else here (ring bytes only).
void MaterializeAny(uint64_t addr, uint64_t size, Why why) {
	auto& s = S();
	// Bytes in the ring (written) or none (undefined): a CPU copy or nothing, on this thread - no wait for the render
	// thread (a loader on a fiber, a game thread in the engine's allocator lock).
	if (MaterializeHere(addr, size, why, 0, false)) return;
	MaterializeJob job {addr, size, why};
	if (s.host.run_render != nullptr && s.host.run_render(&MaterializeTrampoline, &job)) return;
	// (A thread that can do neither: see the top of this file.)
	std::printf("NativeTex: 0x%llx..0x%llx needs the GPU, touched by a thread that can neither record GPU work nor wait for "
	            "the render thread\n", static_cast<unsigned long long>(addr), static_cast<unsigned long long>(addr + size));
	std::fflush(stdout);
	std::abort();
}

// Render thread: the owned entries over [addr, end) selected by `pick` copied by the GPU into the ring (one copy of each
// image's bytes, no wait); those that cannot be (ring full, image not downloadable) written into guest memory instead.
template <typename Pick>
void RehomeHere(uint64_t addr, uint64_t end, Pick pick) {
	auto& s = S();
	struct Group {
		uint64_t           image = 0, serial = 0;
		std::vector<Piece> pieces;
	};
	std::vector<Group> groups;
	{
		Lock lock(s.m);
		WaitInflight(s, lock, addr, end);
		for (auto it = FirstOverLocked(s, addr); it != s.entries.end() && it->first < end; ++it) {
			auto& e = it->second;
			if (!e.owned || !pick(e)) continue;
			e.inflight = true;
			auto g     = std::find_if(groups.begin(), groups.end(), [&](const Group& x) { return x.serial == e.serial; });
			if (g == groups.end()) g = groups.insert(groups.end(), Group {e.image, e.serial, {}});
			g->pieces.push_back(PieceOf(it->first, e));
		}
	}
	if (groups.empty()) return;
	std::vector<Piece> fallback;
	for (auto& g: groups) {
		uint64_t ib = 0, isize = 0, tick = 0, offset = 0;
		uint32_t alloc = 0;
		bool     ok    = s.host.owned_extent != nullptr && s.host.owned_extent(g.image, g.serial, &ib, &isize);
		if (ok) {
			Lock lock(s.m);
			ok = AllocateLocked(s, isize, &alloc, &offset);
		}
		if (ok && (s.host.copy_owned == nullptr || !s.host.copy_owned(g.image, g.serial, offset, &tick))) {
			Lock lock(s.m);
			ReleaseRefLocked(s, alloc, 0);
			ok = false;
		}
		if (!ok) {
			fallback.insert(fallback.end(), g.pieces.begin(), g.pieces.end());
			continue;
		}
		Lock lock(s.m);
		for (const auto& p: g.pieces) {
			auto it = s.entries.find(p.begin);
			if (it == s.entries.end()) continue;
			auto& e = it->second;
			NoteOwnedLocked(s, e, it->first, false);
			e.owned       = false;
			e.inflight    = false;
			e.ring_offset = offset + (p.src - ib);
			e.alloc       = alloc;
			e.ready_tick  = tick;
			s.ready_entries++;
			AddRefLocked(s, alloc);
			s.rehomes.fetch_add(1, std::memory_order_relaxed);
			s.rehome_bytes.fetch_add(p.end - p.begin, std::memory_order_relaxed);
		}
		ReleaseRefLocked(s, alloc, tick); // (the creator's reference)
	}
	if (!fallback.empty()) {
		const uint64_t bytes = WriteBack(s, fallback, true);
		{
			Lock lock(s.m);
			EraseWrittenLocked(s, fallback);
		}
		s.fallback[WhyGone].fetch_add(1, std::memory_order_relaxed);
		s.fallback_bytes[WhyGone].fetch_add(bytes, std::memory_order_relaxed);
	}
	s.done.notify_all();
}

void Log(bool force) {
	auto&      s   = S();
	const auto now = std::chrono::steady_clock::now();
	if (!force && now - s.last_log < std::chrono::seconds(30)) return;
	s.last_log = now;
	size_t   entries = 0;
	uint64_t used = 0, owned = 0;
	{
		Lock lock(s.m);
		entries = s.entries.size();
		used    = s.used;
		owned   = s.owned_bytes;
	}
	constexpr double MiB = 1048576.0;
	std::printf("NativeTex: routed %.1f MiB in %llu ranges (refused %llu, %.1f MiB); ring uploads %llu (%.1f MiB); now %zu ranges, "
	            "%.1f MiB held by images only, ring %.1f/%.1f MiB; back to the ring %llu (%.1f MiB); moves %llu shared, %llu in "
	            "guest memory; releases %llu (%.1f MiB); fallbacks to guest memory:",
	            static_cast<double>(s.routed_bytes.load()) / MiB, static_cast<unsigned long long>(s.routed_ranges.load()),
	            static_cast<unsigned long long>(s.refused.load()), static_cast<double>(s.refused_bytes.load()) / MiB,
	            static_cast<unsigned long long>(s.uploads.load()), static_cast<double>(s.upload_bytes.load()) / MiB, entries,
	            static_cast<double>(owned) / MiB, static_cast<double>(used) / MiB, static_cast<double>(s.capacity) / MiB,
	            static_cast<unsigned long long>(s.rehomes.load()), static_cast<double>(s.rehome_bytes.load()) / MiB,
	            static_cast<unsigned long long>(s.moves_shared.load()), static_cast<unsigned long long>(s.moves_guest.load()),
	            static_cast<unsigned long long>(s.releases_count.load()), static_cast<double>(s.release_bytes.load()) / MiB);
	for (uint32_t i = 0; i < WhyCount; i++)
		std::printf(" %s %llu (%.1f MiB)", WhyNames[i], static_cast<unsigned long long>(s.fallback[i].load()),
		            static_cast<double>(s.fallback_bytes[i].load()) / MiB);
	std::printf("; lost %llu (%.1f MiB); commits %llu (%.1f MiB undefined until written), decommits %llu; loads declined (not "
	            "committed texture memory) %llu (%.1f MiB)\n",
	            static_cast<unsigned long long>(s.lost.load()), static_cast<double>(s.lost_bytes.load()) / MiB,
	            static_cast<unsigned long long>(s.commits.load()), static_cast<double>(s.commit_bytes.load()) / MiB,
	            static_cast<unsigned long long>(s.decommits.load()), static_cast<unsigned long long>(s.declined.load()),
	            static_cast<double>(s.declined_bytes.load()) / MiB);
}

// Loader::NativeTex stream callbacks (demonsSoulsNativeTexStream.h).
void StreamRelease(uint64_t addr, uint64_t size) noexcept {
	OnTextureRelease(addr, size);
}
bool StreamMove(uint64_t dst, uint64_t src, uint64_t size) noexcept {
	return OnTextureMove(dst, src, size);
}
void StreamCommit(uint64_t addr, uint64_t size) noexcept {
	OnTextureCommit(addr, size);
}
void StreamDecommit(uint64_t addr, uint64_t size) noexcept {
	OnTextureDecommit(addr, size);
}

} // namespace

bool Enabled() noexcept {
#if defined(_WIN32)
	static const bool on = [] {
		const char* value = std::getenv("KYTY_NATIVE_TEX");
		return value != nullptr && value[0] == '1';
	}();
	return on;
#else
	return false; // (the touch fallback needs the Windows probe handler)
#endif
}

bool Active() noexcept {
	return Enabled() && S().active;
}

Buffer* Ring() noexcept {
	return S().ring.get();
}

void Init(GraphicContext& graphics, CommandScheduler& scheduler, const Host& host) {
	auto& s = S();
	if (!Enabled() || s.active) return;
	uint64_t mb = 256;
	if (const char* value = std::getenv("KYTY_NATIVE_TEX_RING_MB"); value != nullptr && value[0] != '\0')
		mb = std::clamp<uint64_t>(std::strtoull(value, nullptr, 10), 32, 4096);
	s.capacity = mb << 20u;
	if (const char* value = std::getenv("KYTY_NATIVE_TEX_RING_KB"); value != nullptr && value[0] != '\0') // (tests)
		s.capacity = std::clamp<uint64_t>(std::strtoull(value, nullptr, 10), 64, 4u << 20u) << 10u;
	// Device-local and host-visible with ReBAR (a heap over 1 GiB of such memory), else host-visible.
	MemoryUsage usage  = MemoryUsage::Upload;
	const auto& memory = graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
		const auto         properties = static_cast<uint32_t>(memory.memoryTypes[i].propertyFlags);
		constexpr uint32_t wanted     = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		const uint32_t     heap       = memory.memoryTypes[i].heapIndex;
		if ((properties & wanted) == wanted && heap < memory.memoryHeapCount && memory.memoryHeaps[heap].size > (uint64_t {1} << 30u)) {
			usage = MemoryUsage::Stream;
			break;
		}
	}
	s.ring     = std::make_unique<Buffer>(graphics, scheduler, usage, 0, AllFlags, s.capacity);
	s.mapped   = s.ring->Mapped().data();
	if (s.mapped == nullptr || s.ring->Mapped().size() < s.capacity) {
		std::printf("NativeTex: off (the staging ring could not be mapped)\n");
		s.ring.reset();
		return;
	}
	s.host     = host;
	s.last_log = std::chrono::steady_clock::now();
	s.active   = true;
	// Agent A's load hooks (0x8fed17 texture files, 0x8fef15 pool layers) and agent B's streaming hooks (VirtualMemoryGPU
	// releases, the texture -> pool-slice copy 0x900f26). A CPU read of loaded texture memory by the game faults on its
	// no-access pages: OnFault.
	::Loader::NativeTex::SetConsumer(+[](const ::Loader::NativeTex::TextureBytes& bytes) noexcept {
		return OnTextureBytes(bytes.guest_dst, bytes.host_src, bytes.size);
	});
	::Loader::NativeTex::SetStreamConsumer(&StreamRelease, &StreamMove, &StreamCommit, &StreamDecommit);
	std::printf("NativeTex: on (KYTY_NATIVE_TEX=1): texture loads staged in a %llu MiB %s ring and uploaded to their images; "
	            "their guest pages are left unwritten\n",
	            static_cast<unsigned long long>(s.capacity >> 20u), usage == MemoryUsage::Stream ? "device-local host-visible (ReBAR)" : "host-visible");
}

bool OnTextureBytes(uint64_t guest_dst, const void* src, uint64_t size) {
	auto& s = S();
	if (!Active() || src == nullptr || size < MinPiece || guest_dst + size < guest_dst) return false;
	// Whole pages only: the partial first and last pages (shared with whatever else lives there) get their bytes in
	// guest memory, tracked as any host write.
	const uint64_t whole_begin = PageUp(guest_dst), whole_end = PageDown(guest_dst + size);
	if (whole_end <= whole_begin || whole_end - whole_begin < MinPiece) return false;
	const auto* bytes = static_cast<const uint8_t*>(src);
	src               = bytes + (whole_begin - guest_dst);
	const uint64_t head = whole_begin - guest_dst, tail = guest_dst + size - whole_end;
	const uint64_t full_dst = guest_dst, full_size = size;
	guest_dst = whole_begin;
	size      = whole_end - whole_begin;
	uint32_t alloc  = 0;
	uint64_t offset = 0;
	{
		Lock lock(s.m);
		if (!CoveredLocked(s, guest_dst, guest_dst + size)) {
			s.declined.fetch_add(1, std::memory_order_relaxed);
			s.declined_bytes.fetch_add(full_size, std::memory_order_relaxed);
			return false; // (not committed texture memory: OnTextureCommit)
		}
		if (s.entries.size() >= MaxEntries || !AllocateLocked(s, size, &alloc, &offset)) {
			s.refused.fetch_add(1, std::memory_order_relaxed);
			s.refused_bytes.fetch_add(size, std::memory_order_relaxed);
			return false;
		}
	}
	std::memcpy(s.mapped + offset, src, size); // (the one copy of the load)
	s.ring->Flush(offset, size);
	// As a file read's host write: the trackers told before (images and buffers over the range CPU-dirty) and after (the
	// images refreshed - from the ring now: InitializeImage - closing the window of an upload from stale guest bytes).
	t_suppress_host_write = true;
	const bool prepared   = LibKernel::Memory::PrepareHostOverwrite(full_dst, full_size);
	t_suppress_host_write = false;
	// (An older range over an edge page: its bytes back first, the page is then the guest's.)
	if (head != 0 && Overlaps(PageDown(full_dst), Page)) MaterializeAny(PageDown(full_dst), Page, WhyHostWrite);
	if (tail != 0 && Overlaps(whole_end, Page)) MaterializeAny(whole_end, Page, WhyHostWrite);
	if (head != 0) LibKernel::Memory::WriteBacking(full_dst, bytes, head, "native-tex edge");
	if (tail != 0) LibKernel::Memory::WriteBacking(whole_end, bytes + (whole_end - full_dst), tail, "native-tex edge");
	{
		Lock  lock(s.m);
		Delta d;
		WaitInflight(s, lock, guest_dst, guest_dst + size);
		DropLocked(s, guest_dst, guest_dst + size, d); // (newer bytes replace them)
		AddCoverageLocked(s, guest_dst, guest_dst + size, d);
		Entry e {};
		e.end         = guest_dst + size;
		e.ring_offset = offset;
		e.alloc       = alloc;
		e.flip        = s.flip;
		InsertLocked(s, guest_dst, e); // (it takes the allocation's first reference)
		ApplyLocked(s, d);
	}
	t_suppress_host_write = true;
	LibKernel::Memory::FinishHostOverwrite(full_dst, full_size, prepared);
	t_suppress_host_write = false;
	s.routed_bytes.fetch_add(size, std::memory_order_relaxed);
	s.routed_ranges.fetch_add(1, std::memory_order_relaxed);
	return true;
}

bool OnTextureMove(uint64_t dst, uint64_t src, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0 || dst == src || dst + size < dst || src + size < src) return false;
	bool shared = false;
	{
		Lock lock(s.m);
		const uint64_t lo = std::min(dst, src), hi = std::max(dst, src) + size;
		WaitInflight(s, lock, lo, hi);
		// All of the source ours: the destination gets the same bytes without a copy (ring bytes shared; bytes an image
		// holds recorded as "that image's", copied by the GPU when an image needs them: Rehome).
		std::vector<std::pair<uint64_t, Entry>> moved;
		uint64_t                                at = src;
		for (auto it = FirstOverLocked(s, src); it != s.entries.end() && it->first < src + size; ++it) {
			const uint64_t b = std::max(it->first, src), e = std::min(it->second.end, src + size);
			if (b != at) break;
			Entry copy = it->second;
			copy.end   = e - src + dst;
			copy.flip  = s.flip;
			if (copy.owned) copy.src += b - it->first;
			else if (!copy.undef) copy.ring_offset += b - it->first;
			moved.emplace_back(b - src + dst, copy);
			at = e;
		}
		if (at == src + size && !moved.empty()) {
			for (const auto& [b, e]: moved)
				if (!e.owned && !e.undef) AddRefLocked(s, e.alloc); // (before the drop: dst may overlap src)
			Delta d;
			DropLocked(s, dst, dst + size, d);
			AddCoverageLocked(s, dst, dst + size, d);
			for (const auto& [b, e]: moved) {
				InsertLocked(s, b, e);
				if (e.ready_tick != 0) s.ready_entries++;
			}
			ApplyLocked(s, d);
			shared = true;
		} else if (TouchedLocked(s, dst, dst + size)) {
			Delta d;
			DropLocked(s, dst, dst + size, d); // (the guest move overwrites them)
			ApplyLocked(s, d);
		}
	}
	if (shared) {
		// The images over the destination refreshed (as the default move tells the trackers: B's OnTextureMove).
		t_suppress_host_write = true;
		(void)LibKernel::Memory::TryPrepareHostOverwrite(dst, size);
		t_suppress_host_write = false;
		s.moves_shared.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	// Partly ours: the caller moves guest memory; its read of the source faults its bytes back (OnFault).
	s.moves_guest.fetch_add(1, std::memory_order_relaxed);
	return false;
}

void OnTextureRelease(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0) return;
	if (addr + size <= s.lo.load(std::memory_order_relaxed) || addr >= s.hi.load(std::memory_order_relaxed)) return;
	// At once: the pages may be reused by any allocator as soon as the engine's call returns. (Under the engine's
	// allocator lock: `m` and the page manager's locks only, and no holder of either waits for a game lock.)
	Forget(addr, size);
	s.releases_count.fetch_add(1, std::memory_order_relaxed);
	s.release_bytes.fetch_add(size, std::memory_order_relaxed);
}

void OnTextureCommit(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0 || addr + size < addr) return;
	const uint64_t begin = PageUp(addr), end = PageDown(addr + size);
	if (end <= begin) return;
	// (Under the engine's allocator lock: `m` and the page manager's locks only, as OnTextureRelease.)
	Lock  lock(s.m);
	Delta d;
	WaitInflight(s, lock, begin, end);
	DropLocked(s, begin, end, d);
	AddCoverageLocked(s, begin, end, d);
	Entry e {};
	e.end   = end;
	e.undef = true;
	e.flip  = s.flip;
	InsertLocked(s, begin, e);
	ApplyLocked(s, d);
	ReclaimLocked(s);
	s.commits.fetch_add(1, std::memory_order_relaxed);
	s.commit_bytes.fetch_add(end - begin, std::memory_order_relaxed);
}

void OnTextureDecommit(uint64_t addr, uint64_t size) {
	OnTextureCommit(addr, size); // (the same: undefined content, the texture keeps the pages)
	S().decommits.fetch_add(1, std::memory_order_relaxed);
}

bool Overlaps(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0) return false;
	if (addr + size <= s.lo.load(std::memory_order_relaxed) || addr >= s.hi.load(std::memory_order_relaxed)) return false;
	Lock lock(s.m);
	return TouchedLocked(s, addr, addr + size);
}

std::vector<Piece> Collect(uint64_t addr, uint64_t size) {
	auto&              s = S();
	std::vector<Piece> out;
	if (!Active() || size == 0) return out;
	Lock lock(s.m);
	WaitInflight(s, lock, addr, addr + size);
	for (auto it = FirstOverLocked(s, addr); it != s.entries.end() && it->first < addr + size; ++it) {
		const uint64_t b = std::max(it->first, addr), e = std::min(it->second.end, addr + size);
		auto           p = PieceOf(it->first, it->second);
		if (p.owned) p.src += b - it->first;
		else if (!p.undefined) p.ring_offset += b - it->first;
		p.begin = b;
		p.end   = e;
		out.push_back(p);
	}
	return out;
}

void MarkOwned(const std::vector<Piece>& uploaded, uint64_t image, uint64_t serial, uint64_t tick) {
	auto& s = S();
	Lock  lock(s.m);
	for (const auto& p: uploaded) {
		if (p.owned || p.begin >= p.end) continue;
		WaitInflight(s, lock, p.begin, p.end);
		SplitAtLocked(s, p.begin);
		SplitAtLocked(s, p.end);
		for (auto it = s.entries.lower_bound(p.begin); it != s.entries.end() && it->first < p.end; ++it) {
			auto& e = it->second;
			// Only the bytes the upload took: the same allocation at the same place (a load since replaced them: its
			// own Finish made the image dirty again, the next refresh takes them).
			const bool same = !e.owned && e.undef == p.undefined &&
			                  (e.undef || (e.alloc == p.alloc && e.ring_offset == p.ring_offset + (it->first - p.begin)));
			if (!same) continue;
			if (!e.undef) ReleaseRefLocked(s, e.alloc, std::max(tick, e.ready_tick));
			if (e.ready_tick != 0 && s.ready_entries != 0) s.ready_entries--;
			e.undef       = false;
			e.owned       = true;
			e.image       = image;
			e.serial      = serial;
			e.src         = it->first;
			e.ring_offset = 0;
			e.ready_tick  = 0;
			NoteOwnedLocked(s, e, it->first, true);
		}
	}
}

void Materialize(uint64_t addr, uint64_t size, const char* why, uint64_t serial) {
	if (!Active() || size == 0) return;
	(void)MaterializeHere(addr, size, WhyOf(why), serial, true);
}

void ImageGone(uint64_t image, uint64_t serial) {
	auto& s = S();
	if (!Active() || serial == 0) return;
	{
		Lock lock(s.m);
		if (s.owned_count.find(serial) == s.owned_count.end()) return;
	}
	const auto pick = [serial](const Entry& e) { return e.serial == serial; };
	// Its own range first (a lookup), the whole map only when bytes it holds were moved elsewhere (OnTextureMove).
	uint64_t begin = 0, size = 0;
	if (s.host.owned_extent != nullptr && s.host.owned_extent(image, serial, &begin, &size)) RehomeHere(begin, begin + size, pick);
	{
		Lock lock(s.m);
		if (s.owned_count.find(serial) == s.owned_count.end()) return;
	}
	RehomeHere(0, ~uint64_t {0}, pick);
}

void Rehome(uint64_t addr, uint64_t size, uint64_t keep_serial) {
	if (!Active() || size == 0) return;
	RehomeHere(addr, addr + size, [keep_serial](const Entry& e) { return e.serial != keep_serial; });
}

void BeforeGuestRead(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0 || addr + size < addr) return;
	if (addr + size <= s.lo.load(std::memory_order_relaxed) || addr >= s.hi.load(std::memory_order_relaxed)) return;
	if (!MaybeGranules(s, addr, addr + size) || t_suppress_host_write) return;
	if (!Overlaps(addr, size)) return;
	MaterializeAny(addr, size, WhyRead);
}

void BeforeHostWrite(uint64_t addr, uint64_t size) {
	if (t_suppress_host_write || !Overlaps(addr, size)) return;
	MaterializeAny(addr, size, WhyHostWrite);
}

void Forget(uint64_t addr, uint64_t size) {
	auto& s = S();
	if (!Active() || size == 0) return;
	if (addr + size <= s.lo.load(std::memory_order_relaxed) || addr >= s.hi.load(std::memory_order_relaxed)) return;
	Lock  lock(s.m);
	Delta d;
	WaitInflight(s, lock, addr, addr + size);
	DropLocked(s, addr, addr + size, d);
	ApplyLocked(s, d);
	ReclaimLocked(s);
}

bool OnFault(uint64_t address, uint64_t /*rip*/, const void* /*native_context*/, bool /*write*/) {
	auto& s = S();
	if (!Active() || address < PageDown(s.lo.load(std::memory_order_relaxed)) ||
	    address >= PageUp(s.hi.load(std::memory_order_relaxed)))
		return false;
	const uint64_t page = PageDown(address);
	{
		Lock lock(s.m);
		if (InflightLocked(s, page, page + Page)) {
			WaitInflight(s, lock, page, page + Page);
			return true; // (written back by another thread meanwhile: the access goes again)
		}
		// Nothing of ours there (now): if a release or write-back just took our watchers off it, the page's protection is
		// no longer ours and the access goes again; else not our fault.
		if (!TouchedLocked(s, page, page + Page)) return false;
	}
	// The bytes of every range touching the page into guest memory, its watchers gone: the access goes again and the page
	// is tracked as any other from then on.
	MaterializeAny(page, Page, WhyTouch);
	return true;
}

void Pump() {
	auto& s = S();
	if (!Active()) return;
	std::vector<std::pair<uint64_t, uint64_t>> spill;
	{
		Lock lock(s.m);
		s.flip++;
		if (s.host.tick_done != nullptr) {
			for (auto& a: s.allocs)
				if (a.free_tick != 0 && s.host.tick_done(a.free_tick)) a.free_tick = 0;
			// (Only while GPU copies are pending: no walk of the ranges per flip otherwise - nothing per frame grows with
			// the MiB routed.)
			if (s.ready_entries != 0) {
				uint64_t pending = 0;
				for (auto& [b, e]: s.entries) {
					if (e.ready_tick != 0 && !e.inflight && s.host.tick_done(e.ready_tick)) e.ready_tick = 0;
					pending += e.ready_tick != 0 ? 1 : 0;
				}
				s.ready_entries = pending;
			}
		}
		ReclaimLocked(s);
		// Spill policy: past 75% of the ring (or of the range cap), the oldest ring bytes no image took go to guest memory
		// (the old behavior for them) until below 50%; entries 2 flips old or more only (a load in progress keeps its own).
		if (s.used > s.capacity / 4 * 3 || s.entries.size() > MaxEntries / 4 * 3) {
			std::vector<std::tuple<uint32_t, uint64_t, uint64_t>> ring;
			for (const auto& [b, e]: s.entries)
				if (!e.owned && !e.undef && !e.inflight && e.flip + 2 <= s.flip) ring.emplace_back(e.alloc, b, e.end);
			std::sort(ring.begin(), ring.end());
			uint64_t projected = s.used;
			size_t   count     = s.entries.size();
			for (const auto& [alloc, b, e]: ring) {
				if (projected <= s.capacity / 2 && count <= MaxEntries / 2) break;
				spill.emplace_back(b, e - b);
				projected -= std::min(projected, e - b);
				count--;
			}
		}
	}
	for (const auto& [b, n]: spill) (void)MaterializeHere(b, n, WhySpill, 0, true);
	Log(false);
}

void NoteNativeUpload(uint64_t bytes) {
	S().uploads.fetch_add(1, std::memory_order_relaxed);
	S().upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void NoteFallback(const char* why, uint64_t bytes) {
	const auto w = WhyOf(why);
	S().fallback[w].fetch_add(1, std::memory_order_relaxed);
	S().fallback_bytes[w].fetch_add(bytes, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::NativeTex
