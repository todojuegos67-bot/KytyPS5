#pragma once
// The speculative translation of the calling thread (KYTY_SPECULATE, docs/PARALLEL-TRANSLATION.md), as the caches
// see it. A speculation records the GPU work of a graphics command buffer into its own command buffers without
// changing what other translations see: what the work needs from the shared caches is noted here and done when the
// translation is committed, in guest order, before its command buffers are submitted (uploads of the ranges it
// reads, page-table preparation for its BDA reads), or after them (the ranges it writes become GPU-owned, image
// layouts, guest memory writes of the command processor). Its own later decisions see those notes (a range it will
// upload counts as clean, one it writes as GPU-modified, an image as its work left it).
//
// A packet whose translation needs more (a buffer or image to create, a texture to upload, a program to compile,
// the normal path) is refused (Refuse). A draw or dispatch the speculation cannot take becomes a hole: the work
// recorded so far is a segment, the commit translates the packet the normal way between it and the next segment.
// At any other refused packet the speculation stops; what it recorded before is committed and the command buffer is
// translated on from there.

#include "common/guardedCopy.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "live-switch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {
class Image;
class StreamBuffer;
} // namespace Libs::Graphics

namespace Libs::Graphics::Spec {

struct Range {
	uint64_t begin = 0, end = 0;
};

// A guest memory write of the command processor (a label) or a copy made on the CPU (a linear copy:
// BufferCache::CopyBuffer), applied at commit in order.
struct MemoryWrite {
	uint64_t address = 0;
	uint64_t value   = 0; // a copy's source address
	uint32_t size    = 0; // 4 or 8; a copy's bytes
	bool     copy    = false;
};

// A word read from the pending guest memory writes of a speculation before it (State::previous), and a wait
// found satisfied: the commit checks memory still holds that word / the wait is still satisfied before the segment.
struct ChainRead {
	uint64_t address = 0;
	uint32_t value   = 0;
};
struct WaitCheck {
	uint64_t address = 0, reference = 0, mask = 0;
	uint32_t function = 0, bytes = 4;
};

// The notes' counts: a segment's notes are those after the previous segment's counts, up to its own.
struct Counts {
	size_t current = 0, written = 0, bda = 0, invalidated = 0, writes = 0, touched_images = 0, touched_buffers = 0,
	       clean_reads = 0, feedbacks = 0, buffer_ranges = 0, sets = 0, chain_reads = 0, waits = 0, cpu_copies = 0;
};

// An image's barrier state as a segment found it (the commit transitions the image there first if other work moved
// it since) and as it left it.
struct ImageUse {
	Image*                        image  = nullptr;
	uint64_t                      serial = 0;
	VulkanImageState              found, left;
	std::vector<VulkanImageState> found_subresources, left_subresources;
	uint64_t                      found_group = 0, left_group = 0; // Image::transit_group
	bool                          found_buffer_modified = false; // (Image::IsBufferModified when the segment found it)
};

// The 64 KiB granules ranges were added over, hashed into 4096 bits, and up to BigRanges ranges over 64 granules or
// more as they are (more: all): one look before a range set's search, for reads mostly of memory nothing pending
// writes. (With every range of 4 MiB or more taken as all, 88% of a speculation thread's ~125K reads a frame went on to
// four range set searches: about a quarter of its translation, 10-31.)
struct GranuleFilter {
	static constexpr size_t  BigRanges = 8;
	std::array<uint64_t, 64> bits {};
	bool                     all = false;
	uint32_t                 big_count = 0;
	std::array<std::pair<uint64_t, uint64_t>, BigRanges> big {}; // [begin, end)
	[[nodiscard]] static uint32_t Bit(uint64_t granule) noexcept {
		return static_cast<uint32_t>((granule * 0x9E3779B97F4A7C15ull) >> 52u);
	}
	void AddBig(uint64_t begin, uint64_t end) noexcept {
		// (KYTY_SPECULATE_BIG_FILTER, live speculate_big_filter: 0 takes any as all, as before.)
		static auto& kept = LiveSwitch::Find("speculate_big_filter", 1);
		if (kept.load(std::memory_order_relaxed) == 0) {
			all = true;
			return;
		}
		for (uint32_t i = 0; i < big_count; ++i)
			if (big[i].first <= end && begin <= big[i].second) { // (overlapping or touching: one range)
				big[i] = {std::min(big[i].first, begin), std::max(big[i].second, end)};
				return;
			}
		if (big_count == BigRanges) {
			all = true;
			return;
		}
		big[big_count++] = {begin, end};
	}
	void Add(uint64_t address, uint64_t size) noexcept {
		if (size == 0) return;
		const auto first = address >> 16u, last = (address + size - 1) >> 16u;
		if (last - first >= 64) return AddBig(address, address + size);
		for (auto granule = first; granule <= last; ++granule) bits[Bit(granule) >> 6u] |= uint64_t {1} << (Bit(granule) & 63u);
	}
	[[nodiscard]] bool MayHold(uint64_t address, uint64_t size) const noexcept {
		if (all) return true;
		if (size == 0) return false;
		for (uint32_t i = 0; i < big_count; ++i)
			if (big[i].first < address + size && address < big[i].second) return true;
		const auto first = address >> 16u, last = (address + size - 1) >> 16u;
		if (last - first >= 64) return true;
		for (auto granule = first; granule <= last; ++granule)
			if (((bits[Bit(granule) >> 6u] >> (Bit(granule) & 63u)) & 1u) != 0) return true;
		return false;
	}
	void Merge(const GranuleFilter& other) noexcept {
		all = all || other.all;
		for (size_t i = 0; i < bits.size(); ++i) bits[i] |= other.bits[i];
		for (uint32_t i = 0; i < other.big_count && !all; ++i) AddBig(other.big[i].first, other.big[i].second);
	}
	void Clear() noexcept {
		bits.fill(0);
		all       = false;
		big_count = 0;
	}
};

struct Segment {
	Counts                end;
	bool                  bda_all = false; // all BDA ranges prepared before it (an unbounded plan)
	std::vector<ImageUse> images;
	size_t                recorded = 0; // the recorder's recorded command buffers up to this count
};

// An image's barrier state as the speculation's work leaves it so far (Image::GetBarriers uses it), and as the open
// segment found it (`used`: the open segment transitioned it).
struct ImageEntry {
	Image*                        image  = nullptr;
	uint64_t                      serial = 0;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresources;
	uint64_t                      group = 0;
	bool                          used  = false;
	VulkanImageState              found;
	std::vector<VulkanImageState> found_subresources;
	uint64_t                      found_group = 0;
	bool                          found_buffer_modified = false;
};

struct State {
	// What the work needs, in order, for all segments (Segment::end splits them):
	// - ranges it reads, made current before it (buffer uploads); ranges it writes (obtained written before it:
	//   their CPU-dirty pages uploaded, then GPU-owned);
	// - BDA read ranges (GpuResourceManager::PrepareBdaReadRanges) before it;
	// - TextureCache::InvalidateMemoryFromGPU calls (formatted buffers the work writes), before it;
	// - guest memory writes (labels, linear copies made on the CPU) and copy feedback snapshots of what its linear copies
	//   made on the GPU (BufferCache::ScheduleCopyFeedback), after it;
	// - the images and buffers it uses, touched in the caches' LRU after it (an image only a speculation uses must
	//   not age out under it);
	// - guest memory it read on the CPU as no GPU work had written it: a hole that writes it there since stops the
	//   commit before the segment.
	std::vector<Range>                       current, written, invalidated, clean_reads, feedbacks;
	// - the ranges of the game buffers it uses on the GPU (their buffers registered over them still: none registered
	//   or unregistered in their tracker regions since it began);
	std::vector<Range> buffer_ranges;
	// - the descriptor sets it binds (a hole that retires one stops the commit before it: it may be reused under it);
	std::vector<uint64_t> sets;
	// - words it read from a previous speculation's pending writes, and the waits it found satisfied (ChainRead);
	std::vector<ChainRead> chain_reads;
	std::vector<WaitCheck> waits;
	// - the sources and destinations of the copies it makes on the CPU at its commit (in order: what the guest or the
	//   processor writes there before is copied, or copied over): no GPU work may have written them since.
	std::vector<Range> cpu_copies;
	std::vector<GuestRange>                  bda;
	bool                                     bda_all = false; // the open segment's
	std::vector<MemoryWrite>                 writes;
	std::vector<std::pair<Image*, uint64_t>> touched_images; // and the serial
	std::vector<Common::SlotId>              touched_buffers;
	// All segments' current and written ranges and guest memory writes, for the caches' questions; the open
	// segment's guest memory writes (made after its work: its work cannot read them) and the sources of its copies
	// (read after its work: its work cannot write them).
	RangeSet current_set, written_set, journal_set, open_journal_set, open_copy_sources;
	std::vector<Segment>                     closed;
	std::vector<ImageEntry>                  overlay;
	std::unordered_map<const Image*, size_t> overlay_index;
	std::vector<size_t>                      used; // overlay entries the open segment transitioned
	// The upload rings its work reads (BufferCache::GetTableUploadBuffer, GetShaderUploadBuffer): its own, what they
	// fence gets the commit's tick.
	StreamBuffer* table_upload  = nullptr;
	StreamBuffer* shader_upload = nullptr;
	// Why the packet being translated cannot be (null: it can); whether the speculation stops there even when the
	// packet could be a hole (Stop).
	const char* refused = nullptr;
	bool        stop    = false;
	// The speculation of the command buffer before this one while it is not committed (run-ahead): its pending guest
	// memory writes, the ranges its work writes and the image states it leaves are this one's too (ReadJournaled,
	// JournalIntersects, WrittenIntersects, ChainImage). Null: none. What the speculations before it write, merged
	// (Inherit: one lookup, the chain walked when it hits).
	std::atomic<const State*> previous {nullptr}; // (unlinked by the thread that commits)
	RangeSet                  chain_journal, chain_written;
	// The granules of journal_set, written_set, chain_journal and chain_written (a look before their searches).
	GranuleFilter pending;
	// The notes as the packet being translated found them (Mark): a refused packet leaves none (Rollback). (Its image
	// transitions stay: they are recorded, the overlay holds where they leave the images.)
	Counts   checkpoint;
	bool     checkpoint_bda_all = false;
	uint64_t checkpoint_work    = 0; // LocalVulkanRecording::RecordedWrites()

	[[nodiscard]] Counts Sizes() const noexcept {
		return {current.size(),        written.size(),         bda.size(),         invalidated.size(), writes.size(),
		        touched_images.size(), touched_buffers.size(), clean_reads.size(), feedbacks.size(), buffer_ranges.size(),
		        sets.size(),           chain_reads.size(),     waits.size(),       cpu_copies.size()};
	}
	// Its pending guest memory writes or a previous speculation's over the range; the ranges its work or a previous
	// one's writes.
	[[nodiscard]] bool JournalIntersects(uint64_t address, uint64_t size) const {
		return pending.MayHold(address, size) &&
		       (journal_set.Intersects(address, size) || chain_journal.Intersects(address, size));
	}
	[[nodiscard]] bool WrittenIntersects(uint64_t address, uint64_t size) const {
		return pending.MayHold(address, size) &&
		       (written_set.Intersects(address, size) || chain_written.Intersects(address, size));
	}
	// After `previous` is set: what the speculations before it not committed yet write (one committed is unlinked:
	// memory holds its writes).
	void Inherit() {
		chain_journal.Clear();
		chain_written.Clear();
		for (const State* state = previous.load(); state != nullptr; state = state->previous.load()) {
			state->journal_set.ForEach([&](uint64_t begin, uint64_t end) { chain_journal.Add(begin, end - begin); });
			state->written_set.ForEach([&](uint64_t begin, uint64_t end) { chain_written.Add(begin, end - begin); });
			pending.Merge(state->pending);
		}
	}
	// An image as the latest speculation before it leaves it (null: none used it).
	[[nodiscard]] const ImageEntry* ChainImage(const Image* image, uint64_t serial) const {
		for (const State* state = previous.load(); state != nullptr; state = state->previous.load())
			if (const auto found = state->overlay_index.find(image); found != state->overlay_index.end())
				return state->overlay[found->second].serial == serial ? &state->overlay[found->second] : nullptr;
		return nullptr;
	}
	[[nodiscard]] Counts Sealed() const noexcept { return closed.empty() ? Counts {} : closed.back().end; }

	void NoteCurrent(uint64_t address, uint64_t size) {
		if (Note(current, Sealed().current, address, size)) current_set.Add(address, size);
	}
	void NoteWritten(uint64_t address, uint64_t size) {
		if (Note(written, Sealed().written, address, size)) {
			written_set.Add(address, size);
			pending.Add(address, size);
		}
	}
	void NoteInvalidated(uint64_t address, uint64_t size) { (void)Note(invalidated, Sealed().invalidated, address, size); }
	void NoteCleanRead(uint64_t address, uint64_t size) { (void)Note(clean_reads, Sealed().clean_reads, address, size); }
	void NoteBufferRange(uint64_t address, uint64_t size) { (void)Note(buffer_ranges, Sealed().buffer_ranges, address, size); }
	void NoteCpuCopy(uint64_t address, uint64_t size) { (void)Note(cpu_copies, Sealed().cpu_copies, address, size); }
	void NoteSet(uint64_t set) {
		if (sets.size() == Sealed().sets || sets.back() != set) sets.push_back(set);
	}
	// What it holds of surface metadata clears (TextureCache::IsMetaCleared), by slice (all: UINT32_MAX), from a segment
	// on: found not cleared (the commit checks it before the segments that relied on it, first..last); consumed by a
	// hole whose draw the normal path clears a target for (NativeXprEmitTargets, TableTransit: the segment after it
	// finds it not cleared, checked as well); cleared by a hole (a compute clear the normal path consumes: the draws
	// after it are holes that apply it).
	struct MetaFact {
		uint64_t address = 0;
		uint32_t slice   = 0;
		bool     cleared = false;
		size_t   first = 0, last = 0;
	};
	std::vector<MetaFact> meta_facts;

	// IsMetaCleared's answer for the open segment from the latest fact: 1 cleared, 0 not, -1 none (the cache answers).
	// Without one of its own, as a speculation before it not committed yet leaves the surface: cleared by a hole of
	// its (the draw here is a hole that applies the clear), or not (noted: the commit checks it).
	[[nodiscard]] int KnownMeta(uint64_t address, uint32_t slice) noexcept {
		MetaFact* latest = nullptr;
		for (auto& fact: meta_facts)
			if (fact.address == address && (fact.slice == slice || fact.slice == UINT32_MAX) && fact.first <= closed.size() &&
			    (latest == nullptr || fact.first >= latest->first))
				latest = &fact;
		if (latest != nullptr) {
			if (latest->cleared) return 1;
			latest->last = std::max(latest->last, closed.size());
			return 0;
		}
		for (const State* state = previous.load(); state != nullptr; state = state->previous.load()) {
			const MetaFact* last = nullptr;
			for (const auto& fact: state->meta_facts)
				if (fact.address == address && (fact.slice == slice || fact.slice == UINT32_MAX) &&
				    (last == nullptr || fact.first >= last->first))
					last = &fact;
			if (last == nullptr) continue;
			if (last->cleared) return 1;
			NoteMetaRead(address, slice);
			return 0;
		}
		return -1;
	}
	void NoteMetaRead(uint64_t address, uint32_t slice) {
		meta_facts.push_back({address, slice, false, closed.size(), closed.size()});
	}
	// The open segment's packet is a hole that consumes the clear / clears the surface.
	void ConsumeMeta(uint64_t address, uint32_t slice) {
		meta_facts.push_back({address, slice, false, closed.size() + 1, closed.size() + 1});
	}
	void PredictClear(uint64_t address) { meta_facts.push_back({address, UINT32_MAX, true, closed.size() + 1, closed.size() + 1}); }
	void Journal(uint64_t address, uint64_t value, uint32_t size, bool copy = false) {
		writes.push_back({address, value, size, copy});
		journal_set.Add(address, size);
		pending.Add(address, size);
		open_journal_set.Add(address, size);
		if (copy) open_copy_sources.Add(value, size);
	}
	// The words at `address` as the translation reads them: as the processor's pending writes leave them (in order it
	// writes before the commands after it; a copy's word is its source's), its own, then a previous speculation's;
	// false when one is the GPU's (its own work's or a previous one's), a write it took back (a hole's, made before the
	// work after it) or a copy's source a write is pending over. A word a previous one gives is noted (chain_reads),
	// one memory gives is a clean read (`clean`: noted; the source a copy of its own reads too).
	[[nodiscard]] bool ReadJournaled(uint64_t address, uint32_t count, uint32_t* out, bool clean = true) {
		// (Mostly nothing pending over the words: one look for all of them.)
		const uint64_t bytes = uint64_t {4} * count;
		if (bytes != 0 && !pending.MayHold(address, bytes)) {
			if (!CopyGuest(out, address, bytes)) return false;
			if (clean) NoteCleanRead(address, bytes);
			return true;
		}
		if (bytes != 0 && !journal_set.Intersects(address, bytes) && !written_set.Intersects(address, bytes) &&
		    !chain_written.Intersects(address, bytes) &&
		    (previous.load(std::memory_order_relaxed) == nullptr || !chain_journal.Intersects(address, bytes))) {
			if (!CopyGuest(out, address, bytes)) return false;
			if (clean) NoteCleanRead(address, bytes);
			return true;
		}
		// (KYTY_SPECULATE_COPY_RUNS, live speculate_copy_runs 0: word by word, as before.)
		static auto& copy_runs = LiveSwitch::Find("speculate_copy_runs", 1);
		if (count > 1 && copy_runs.load(std::memory_order_relaxed) != 0)
			if (const auto run = ReadCopyRun(address, count, out); run != Run::NotOne) return run == Run::Read;
		for (uint32_t i = 0; i < count; ++i) {
			const uint64_t at = address + 4ull * i;
			const State*   by = nullptr;
			if (!ReadWord(at, out[i], by)) return false;
			if (by == nullptr && clean) NoteCleanRead(at, 4);
			else if (by != nullptr && by != this) chain_reads.push_back({at, out[i]});
		}
		return true;
	}

	// Guest memory as a speculation's thread reads it: a page the GPU wrote (read-protected) is not read, the
	// speculation refused, instead of the fault reading it back through the GPU thread (SendCommandSync), which
	// may wait for this thread: a mode 4 hang at the stone arch had a speculation thread fault in TableEvaluate
	// holding the catalog's shared lock while the GPU thread waited for it exclusively (10-31). The callers' page
	// checks race with watchers added meanwhile.
	[[nodiscard]] bool CopyGuest(void* out, uint64_t address, uint64_t bytes) noexcept {
		if (Common::GuardedCopy(out, reinterpret_cast<const void*>(address), bytes) == 0) return true;
		if (refused == nullptr) refused = "GPU-written guest memory (read fault)";
		return false;
	}
	// One word (ReadJournaled): `by` the speculation whose pending write gave it (null: memory).
	[[nodiscard]] bool ReadWord(uint64_t at, uint32_t& value, const State*& by) {
		if (journal_set.Intersects(at, 4)) return Resolve(*this, at, value, by);
		if (written_set.Intersects(at, 4)) return false;
		if (chain_journal.Intersects(at, 4))
			for (const State* state = previous.load(); state != nullptr; state = state->previous.load()) {
				if (state->journal_set.Intersects(at, 4)) return Resolve(*state, at, value, by);
				if (state->written_set.Intersects(at, 4)) return false;
			}
		// (A previous speculation's work's; a write of one finished since is in memory, or will be: a clean read.)
		if (chain_written.Intersects(at, 4)) return false;
		return CopyGuest(&value, at, 4);
	}
	// The words ReadWord gives one by one, in one read when one copy gives them all: the latest pending write over any
	// of them (of this speculation, else of the first one before it with any) is a copy over all of them, with nothing
	// pending over its source and no write of the work of a speculation walked before it over them. (The game's 4 MiB
	// copy of its constants each frame put every descriptor read there on the word path: ~11% of a speculation's
	// thread.) NotOne: not that case, the word path decides.
	enum class Run { NotOne, Read, Refused };
	[[nodiscard]] Run ReadCopyRun(uint64_t address, uint32_t count, uint32_t* out) {
		const uint64_t bytes = uint64_t {4} * count;
		const State*   owner = nullptr;
		if (journal_set.Intersects(address, bytes)) {
			owner = this;
		} else {
			if (written_set.Intersects(address, bytes) || !chain_journal.Intersects(address, bytes)) return Run::NotOne;
			for (const State* state = previous.load(); state != nullptr && owner == nullptr; state = state->previous.load()) {
				if (state->written_set.Intersects(address, bytes)) return Run::NotOne;
				if (state->journal_set.Intersects(address, bytes)) owner = state;
			}
			if (owner == nullptr) return Run::NotOne;
		}
		const auto latest = std::find_if(owner->writes.rbegin(), owner->writes.rend(), [&](const MemoryWrite& write) {
			return write.address < address + bytes && address < write.address + write.size;
		});
		if (latest == owner->writes.rend() || !latest->copy || latest->address > address ||
		    address + bytes > latest->address + latest->size)
			return Run::NotOne;
		const auto source = latest->value + (address - latest->address);
		if (owner->JournalIntersects(source, bytes) || owner->WrittenIntersects(source, bytes)) return Run::Refused;
		if (!CopyGuest(out, source, bytes)) return Run::Refused;
		if (owner == this) NoteCleanRead(source, bytes);
		else
			for (uint32_t i = 0; i < count; ++i) chain_reads.push_back({address + 4ull * i, out[i]});
		return Run::Read;
	}
	// The word `state`'s latest pending write over it gives (a copy: its source's, with no write of that speculation
	// or an earlier one pending over the source).
	[[nodiscard]] bool Resolve(const State& state, uint64_t at, uint32_t& value, const State*& by) {
		const auto latest = std::find_if(state.writes.rbegin(), state.writes.rend(), [&](const MemoryWrite& write) {
			return write.address <= at && at + 4 <= write.address + write.size;
		});
		if (latest == state.writes.rend()) return false;
		by = &state;
		if (!latest->copy) {
			value = static_cast<uint32_t>(latest->value);
			return true;
		}
		const auto source = latest->value + (at - latest->address);
		if (state.JournalIntersects(source, 4) || state.WrittenIntersects(source, 4)) return false;
		if (!CopyGuest(&value, source, 4)) return false;
		if (&state == this) NoteCleanRead(source, 4);
		return true;
	}

	void Mark(uint64_t work) noexcept {
		checkpoint         = Sizes();
		checkpoint_bda_all = bda_all;
		checkpoint_work    = work;
	}
	// (The cumulative sets keep what the packet noted: they only make later decisions more careful.)
	void Rollback() {
		current.resize(std::min(current.size(), checkpoint.current));
		written.resize(std::min(written.size(), checkpoint.written));
		bda.resize(std::min(bda.size(), checkpoint.bda));
		invalidated.resize(std::min(invalidated.size(), checkpoint.invalidated));
		writes.resize(std::min(writes.size(), checkpoint.writes));
		touched_images.resize(std::min(touched_images.size(), checkpoint.touched_images));
		touched_buffers.resize(std::min(touched_buffers.size(), checkpoint.touched_buffers));
		clean_reads.resize(std::min(clean_reads.size(), checkpoint.clean_reads));
		feedbacks.resize(std::min(feedbacks.size(), checkpoint.feedbacks));
		buffer_ranges.resize(std::min(buffer_ranges.size(), checkpoint.buffer_ranges));
		sets.resize(std::min(sets.size(), checkpoint.sets));
		chain_reads.resize(std::min(chain_reads.size(), checkpoint.chain_reads));
		waits.resize(std::min(waits.size(), checkpoint.waits));
		cpu_copies.resize(std::min(cpu_copies.size(), checkpoint.cpu_copies));
		bda_all = checkpoint_bda_all;
	}

	void Clear() {
		current.clear();
		written.clear();
		invalidated.clear();
		clean_reads.clear();
		feedbacks.clear();
		buffer_ranges.clear();
		sets.clear();
		chain_reads.clear();
		waits.clear();
		cpu_copies.clear();
		bda.clear();
		bda_all = false;
		writes.clear();
		touched_images.clear();
		touched_buffers.clear();
		current_set.Clear();
		written_set.Clear();
		journal_set.Clear();
		open_journal_set.Clear();
		open_copy_sources.Clear();
		closed.clear();
		meta_facts.clear();
		overlay.clear();
		overlay_index.clear();
		used.clear();
		refused  = nullptr;
		stop     = false;
		previous = nullptr;
		chain_journal.Clear();
		chain_written.Clear();
		pending.Clear();
	}
	// The open segment ends (its command buffers: the recorder's first `recorded_count`): its image uses taken from
	// the overlay.
	void CloseSegment(size_t recorded_count) {
		auto& segment    = closed.emplace_back();
		segment.end      = Sizes();
		segment.bda_all  = bda_all;
		segment.recorded = recorded_count;
		for (const auto index: used) {
			auto& entry = overlay[index];
			segment.images.push_back({entry.image, entry.serial, entry.found, entry.state,
			                          std::move(entry.found_subresources), entry.subresources, entry.found_group,
			                          entry.group, entry.found_buffer_modified});
			entry.used = false;
		}
		used.clear();
		bda_all = false;
		open_journal_set.Clear();
		open_copy_sources.Clear();
	}

private:
	// A range noted, merged into the open segment's last one when it extends it (draws mostly read consecutive
	// ranges); false when one of the open segment's last few covers it (a repeat only costs the commit a lookup).
	static bool Note(std::vector<Range>& ranges, size_t sealed, uint64_t address, uint64_t size) {
		if (size == 0) return false;
		if (ranges.size() > sealed && ranges.back().begin <= address && address <= ranges.back().end) {
			ranges.back().end = std::max(ranges.back().end, address + size);
			return true;
		}
		const auto first = std::max(sealed, ranges.size() - std::min<size_t>(ranges.size(), 8));
		if (std::any_of(ranges.begin() + static_cast<ptrdiff_t>(first), ranges.end(),
		                [&](const Range& r) { return r.begin <= address && address + size <= r.end; }))
			return false;
		ranges.push_back({address, address + size});
		return true;
	}
};

inline thread_local State* t_state = nullptr;
// The state pass's record of the command buffer it passes (KYTY_SPECULATE=4, CommandProcessor::SetStateOnly): the guest
// memory writes of its processor (journaled, not made; one it cannot tell: written) and the surfaces its compute clears
// clear (predicted). A speculation of a command buffer after it reads through it while it is not done.
inline thread_local State* t_record = nullptr;

// (KYTY_SPECULATE unset or 0: no speculation state on any thread. The global is read first, so mode 0's per-draw and
// per-packet callers read no thread-local, as the base.)
[[nodiscard]] inline bool On() noexcept {
	return kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0;
}

[[nodiscard]] inline State* Current() noexcept {
	return On() ? t_state : nullptr;
}
[[nodiscard]] inline State* Record() noexcept {
	return On() ? t_record : nullptr;
}

inline void Refuse(const char* why) noexcept {
	if (!On()) return;
	if (t_state != nullptr && t_state->refused == nullptr) t_state->refused = why;
}
// The speculation stops at the packet (not a hole).
inline void Stop(const char* why) noexcept {
	if (!On()) return;
	Refuse(why);
	if (t_state != nullptr) t_state->stop = true;
}

// The guest memory the command processor writes (labels): a wait on one is satisfied by work of the GPU queues, which
// the commit checks a speculation against (EffectsLog); one on memory only the CPU writes orders CPU writes the
// speculation cannot see (it stops there).
struct Labels {
	std::mutex                   mutex;
	std::unordered_set<uint64_t> addresses;
};
inline Labels g_labels;
inline void   NoteLabel(uint64_t address) {
	  std::lock_guard lock(g_labels.mutex);
	  g_labels.addresses.insert(address & ~uint64_t {3});
}
[[nodiscard]] inline bool IsLabel(uint64_t address) {
	std::lock_guard lock(g_labels.mutex);
	return g_labels.addresses.contains(address & ~uint64_t {3});
}

// What the thread that commits is doing (its effects' source, for the commits' stop reasons): translating the normal
// way, running a hole, committing a speculation's segments.
enum class EffectSource : uint8_t { Normal, Hole, Commit };
inline thread_local EffectSource t_effect_source = EffectSource::Normal;
struct EffectSourceScope {
	explicit EffectSourceScope(EffectSource source) noexcept: saved(std::exchange(t_effect_source, source)) {}
	~EffectSourceScope() { t_effect_source = saved; }
	EffectSourceScope(const EffectSourceScope&)            = delete;
	EffectSourceScope& operator=(const EffectSourceScope&) = delete;
	EffectSource       saved;
};

// The log index (+ 1) of the latest effect over each 256-byte unit (4 KiB for host writes), and over each 4 MiB
// granule for writes larger than 1 MiB (KYTY_SPECULATE_STAMPS, on by default; Effects::Clear): a commit asks whether
// anything was written over what a segment read since it began by looking at the units it read, instead of adding every
// entry since to a range set and searching it for every read (commit bookkeeping was ~20% of the thread that commits,
// 10-31). A unit stamped since is a candidate only: EffectsLog::FirstWriteOver decides. The thread that commits alone
// writes and reads it.
template <uint64_t UnitBits>
struct WriteStamps {
	static constexpr uint64_t GranuleBits  = 22; // (4 MiB a leaf)
	static constexpr uint64_t BlockBits    = 16; // (64 KiB: writes over MidWrite are stamped by block)
	static constexpr uint64_t LeafBits     = GranuleBits - UnitBits;
	static constexpr uint64_t AddressLimit = uint64_t {1} << 40u;
	static constexpr uint64_t BigWrite     = uint64_t {1} << 20u;
	static constexpr uint64_t MidWrite     = uint64_t {1} << 14u;
	struct Leaf {
		std::array<uint64_t, size_t {1} << LeafBits>                     units {};
		std::array<uint64_t, size_t {1} << (GranuleBits - BlockBits)> blocks {};
	};
	struct Granule {
		uint64_t              coarse = 0; // (a big write over some of it)
		uint64_t              latest = 0; // (the latest stamp in it: a granule not written since is passed whole)
		std::unique_ptr<Leaf> leaf;
	};
	std::vector<Granule> granules = std::vector<Granule>(AddressLimit >> GranuleBits);
	// (A write of 16 KiB to 1 MiB stamps its 64 KiB blocks, one over 1 MiB its granules: at most 64 stores a write.
	// Every unit cost ~1.5 ms a frame of the render thread for the async queue's culling buffers, 10-31.)
	void Stamp(uint64_t address, uint64_t size, uint64_t stamp) {
		if (size == 0 || address >= AddressLimit) return;
		const auto end = std::min(address + size, AddressLimit);
		if (end - address > BigWrite) {
			for (auto g = address >> GranuleBits, last = (end - 1) >> GranuleBits; g <= last; ++g)
				granules[g].coarse = granules[g].latest = stamp;
			return;
		}
		const bool by_block = end - address > MidWrite;
		const auto bits     = by_block ? BlockBits : UnitBits;
		for (auto at = address >> bits, last = (end - 1) >> bits; at <= last; ++at) {
			auto& granule = granules[(at << bits) >> GranuleBits];
			if (!granule.leaf) granule.leaf = std::make_unique<Leaf>();
			if (by_block) granule.leaf->blocks[at & ((uint64_t {1} << (GranuleBits - BlockBits)) - 1)] = stamp;
			else granule.leaf->units[at & ((uint64_t {1} << LeafBits) - 1)] = stamp;
			granule.latest = stamp;
		}
	}
	// Whether an effect logged at index `from` or later may be over [address, address + size).
	[[nodiscard]] bool MaybeSince(uint64_t address, uint64_t size, uint64_t from) const {
		if (size == 0) return false;
		if (address >= AddressLimit || address + size > AddressLimit) return true;
		const auto last = (address + size - 1) >> UnitBits;
		for (auto unit = address >> UnitBits; unit <= last;) {
			const auto& granule = granules[unit >> LeafBits];
			if (granule.latest <= from) { // (nothing in the granule since: to the next one)
				unit = ((unit >> LeafBits) + 1) << LeafBits;
				continue;
			}
			if (granule.coarse > from) return true;
			if (granule.leaf) {
				const auto& leaf = *granule.leaf;
				if (leaf.units[unit & ((uint64_t {1} << LeafBits) - 1)] > from) return true;
				if (leaf.blocks[((unit << UnitBits) >> BlockBits) & ((uint64_t {1} << (GranuleBits - BlockBits)) - 1)] > from)
					return true;
			}
			++unit;
		}
		return false;
	}
};

// What the translation of the thread that commits does (the normal translation, holes) that a speculation translated
// before it must see, in order (each speculation is checked against what was logged since it began): the ranges its
// GPU work writes, the guest memory it writes on the CPU (the processor's writes, copies, fills), the descriptor sets
// it retires.
struct EffectsLog {
	enum class Kind : uint8_t { GpuWrite, HostWrite, RetiredSet };
	struct Entry {
		Kind         kind   = Kind::GpuWrite;
		EffectSource source = EffectSource::Normal;
		uint64_t     address = 0, size = 0; // a set: its handle
	};
	std::deque<Entry>     entries;
	uint64_t              base = 0;  // the index of entries.front() since the log began
	std::atomic<uint64_t> end {0};   // entries since the log began (read where a speculation begins)
	[[nodiscard]] uint64_t End() const noexcept { return end.load(std::memory_order_acquire); }
	WriteStamps<8>         stamps;      // GPU and host writes
	WriteStamps<12>        host_stamps; // host writes
	void                   Add(Kind kind, uint64_t address, uint64_t size) {
		                  if (kind != Kind::RetiredSet) stamps.Stamp(address, size, base + entries.size() + 1);
		                  if (kind == Kind::HostWrite) host_stamps.Stamp(address, size, base + entries.size() + 1);
		                  entries.push_back({kind, t_effect_source, address, size});
		                  end.store(base + entries.size(), std::memory_order_release);
	}
	// The first write logged from `from` on over the range (null: none).
	[[nodiscard]] const Entry* FirstWriteOver(uint64_t from, uint64_t address, uint64_t size) const {
		for (auto i = std::max(from, base); i < base + entries.size(); ++i) {
			const auto& entry = entries[i - base];
			if (entry.kind != Kind::RetiredSet && entry.address < address + size && address < entry.address + entry.size)
				return &entry;
		}
		return nullptr;
	}
	// The first GPU write logged from `from` on over the range (null: none).
	[[nodiscard]] const Entry* FirstGpuWriteOver(uint64_t from, uint64_t address, uint64_t size) const {
		for (auto i = std::max(from, base); i < base + entries.size(); ++i) {
			const auto& entry = entries[i - base];
			if (entry.kind == Kind::GpuWrite && entry.address < address + size && address < entry.address + entry.size)
				return &entry;
		}
		return nullptr;
	}
	// Each host write logged from `from` on over the range, clipped to it.
	template <typename F>
	void ForEachHostWriteOver(uint64_t from, uint64_t address, uint64_t size, F&& f) const {
		if (!host_stamps.MaybeSince(address, size, from)) return;
		for (auto i = std::max(from, base); i < base + entries.size(); ++i) {
			const auto& entry = entries[i - base];
			if (entry.kind != Kind::HostWrite || entry.address >= address + size || address >= entry.address + entry.size)
				continue;
			const auto begin = std::max(address, entry.address), end = std::min(address + size, entry.address + entry.size);
			f(begin, end - begin);
		}
	}
	// What no speculation began before.
	void DropBefore(uint64_t index) {
		while (base < index && !entries.empty()) {
			entries.pop_front();
			++base;
		}
	}
};
// The log's entries since a speculation began, as its commit checks them.
struct Effects {
	RangeSet              gpu_writes, host_writes;
	std::vector<uint64_t> retired_sets;
	uint64_t              next = 0; // the log's next entry to take
	bool                  stamped = false; // (writes looked up in the log's stamps, not gathered here)
	void                  Clear(uint64_t from) {
		// (KYTY_SPECULATE_STAMPS, live: `set speculate_stamps 0` gathers GPU writes in a range set as before.)
		static auto& on = LiveSwitch::Find("speculate_stamps", 1);
		stamped = on.load(std::memory_order_relaxed) != 0;
		gpu_writes.Clear();
		host_writes.Clear();
		retired_sets.clear();
		next = from;
	}
	void Take(const EffectsLog& log) {
		for (next = std::max(next, log.base); next < log.base + log.entries.size(); ++next) {
			const auto& entry = log.entries[next - log.base];
			switch (entry.kind) {
				case EffectsLog::Kind::GpuWrite:
					if (!stamped) gpu_writes.Add(entry.address, entry.size);
					break;
				case EffectsLog::Kind::HostWrite:
					if (!stamped) host_writes.Add(entry.address, entry.size);
					break;
				case EffectsLog::Kind::RetiredSet: retired_sets.push_back(entry.address); break;
			}
		}
	}
};
inline thread_local EffectsLog* t_effects = nullptr;

// The speculation threads' packets (KYTY_SPECULATE=4): each thread's number while it translates one (its slot:
// t_packet_slot), 0 between translations. A buffer or image the thread that commits releases (unregistered: no later
// lookup finds it) is destroyed after the packets it may have been looked up in: PacketNow when it is released,
// WaitPacketPassed before it is destroyed.
inline constexpr size_t                                MaxPacketSlots = 12; // (8 graphics queue threads, 4 async)
inline std::array<std::atomic<uint64_t>, MaxPacketSlots> g_packet {};
inline thread_local size_t                             t_packet_slot = 0;
inline thread_local uint64_t                           t_packets     = 0;
inline void EnterPacket() noexcept { g_packet[t_packet_slot].store(++t_packets, std::memory_order_seq_cst); }
inline void LeavePackets() noexcept { g_packet[t_packet_slot].store(0, std::memory_order_seq_cst); }
using PacketMarks = std::array<uint64_t, MaxPacketSlots>;
[[nodiscard]] inline PacketMarks PacketNow() noexcept {
	if (!On()) return {}; // (mode 0: no speculation thread, no packet; no fence)
	std::atomic_thread_fence(std::memory_order_seq_cst); // (after the release: a later packet does not find it)
	PacketMarks marks {};
	for (size_t i = 0; i < MaxPacketSlots; ++i) marks[i] = g_packet[i].load(std::memory_order_seq_cst);
	return marks;
}
// (Not waited for: the thread that releases may hold what a packet waits for. What a packet may still read waits in
// a list of the cache's, destroyed once the packets passed: BufferCache::EraseRetired, TextureCache::EraseRetired.)
[[nodiscard]] inline bool PacketsPassed(const PacketMarks& marks) noexcept {
	for (size_t i = 0; i < MaxPacketSlots; ++i)
		if (marks[i] != 0 && g_packet[i].load(std::memory_order_acquire) == marks[i]) return false;
	return true;
}

// (KYTY_SPECULATE_ASYNC: the async queue's speculation logs them too.)
inline thread_local EffectsLog* t_effects_async = nullptr;
// (KYTY_SPECULATE unset: no effects log on any thread, and the hot callers (ObtainBuffer's writes, the host writes)
// read a plain global instead of two thread-locals.)
extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode; // KYTY_SPECULATE (graphics/guest_gpu/speculation.cpp)
}
inline void NoteGpuWrite(uint64_t address, uint64_t size) {
	if (size == 0 || kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0) return;
	if (t_effects != nullptr) t_effects->Add(EffectsLog::Kind::GpuWrite, address, size);
	if (t_effects_async != nullptr) t_effects_async->Add(EffectsLog::Kind::GpuWrite, address, size);
}
inline void NoteHostWrite(uint64_t address, uint64_t size) {
	if (size == 0 || kyty_local_speculate_mode.load(std::memory_order_relaxed) == 0) return;
	if (t_effects != nullptr) t_effects->Add(EffectsLog::Kind::HostWrite, address, size);
	if (t_effects_async != nullptr) t_effects_async->Add(EffectsLog::Kind::HostWrite, address, size);
}
// The parts of a host write that change memory, in 16-byte granules: found before the write, logged after it
// (NoteHostWrite's order). A speculation that read only the rest reads the same bytes after the write: the game's
// copies rewrite tables of which a few words change (the culling chain's 720-byte descriptor table every frame), and
// the whole write logged stopped the commits of every translation that read a descriptor there. (A write over 64 KiB,
// or one that changes more than 16 parts, is logged whole.)
struct HostWriteParts {
	static constexpr uint64_t Granule = 16, MaxBytes = uint64_t {64} << 10u;
	std::array<std::pair<uint64_t, uint64_t>, 16> parts {};
	uint32_t count = 0;
	uint64_t address = 0, size = 0;
	bool     whole = true;
	void Find(uint64_t dst, const void* src, uint64_t bytes) {
		// (KYTY_SPECULATE_WRITE_PARTS, live speculate_write_parts: 0 logs every write whole, as before.)
		static auto& on = LiveSwitch::Find("speculate_write_parts", 1);
		address = dst, size = bytes, count = 0;
		whole   = !On() || (t_effects == nullptr && t_effects_async == nullptr) || bytes > MaxBytes ||
		        on.load(std::memory_order_relaxed) == 0;
		if (whole) return;
		const auto* from = static_cast<const uint8_t*>(src);
		for (uint64_t at = 0; at < bytes;) {
			const auto n = std::min(Granule - (dst + at) % Granule, bytes - at);
			if (std::memcmp(reinterpret_cast<const void*>(dst + at), from + at, n) != 0) {
				if (count != 0 && parts[count - 1].first + parts[count - 1].second == dst + at) {
					parts[count - 1].second += n;
				} else if (count == parts.size()) {
					whole = true;
					return;
				} else {
					parts[count++] = {dst + at, n};
				}
			}
			at += n;
		}
	}
	void Note() const;
};
inline void NoteRetiredSet(uint64_t set) {
	if (!On()) return;
	if (t_effects != nullptr) t_effects->Add(EffectsLog::Kind::RetiredSet, set, 0);
	if (t_effects_async != nullptr) t_effects_async->Add(EffectsLog::Kind::RetiredSet, set, 0);
}
inline void HostWriteParts::Note() const {
	if (whole) return NoteHostWrite(address, size);
	for (uint32_t i = 0; i < count; ++i) NoteHostWrite(parts[i].first, parts[i].second);
}

} // namespace Libs::Graphics::Spec
