#include "graphics/host_gpu/renderer/drawPrep.h"

#include "common/logging/log.h"
#include "common/perfStats.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/mtGpuStageA.h"
#include "kernel/memory.h"
#include "kernel/scanStamp.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>

#include <fmt/format.h>

namespace Libs::Graphics::DrawPrep {

namespace {

// Results waiting for the GPU thread; the scanner waits while there are more.
constexpr size_t MaxPending = 8192;

// Results the GPU thread is done with go back to the scanner with their buffers, so that neither
// thread frees what the other allocated (slow across threads on the Windows heap).
constexpr size_t MaxRecycled = 4096;

struct Store {
	std::mutex                                                       mutex;
	std::condition_variable                                          space;
	std::array<std::deque<std::unique_ptr<PacketResult>>, MaxQueues> pending; // per queue
	std::vector<std::unique_ptr<PacketResult>>                       recycled;
};

Store& GetStore() {
	static Store store;
	return store;
}

thread_local Key  t_current {};
thread_local bool t_has_current = false;
// GPU thread progress per queue, packed: submission (40 bits), ordinal (24 bits).
std::array<std::atomic<uint64_t>, MaxQueues> g_progress {};

// Each counter on a cache line of its own (round 155): the GPU thread's (takes, uses, misses) and
// the scanner threads' (scanned, prepared, skips) shared lines, and each count pulled the line
// from the other's core.
struct alignas(64) PaddedCounter {
	std::atomic<uint64_t> value {0};
};
std::array<PaddedCounter, static_cast<size_t>(Counter::Count)> g_counters {};
// Round 170: the counts of the compute queues' packets again (the scanner's item, or the GPU
// thread's current packet), for a line of their own.
std::array<PaddedCounter, static_cast<size_t>(Counter::Count)> g_compute_counters {};
thread_local bool                                              t_scanner_compute = false;

// GPU thread (round 155): the packet the current draw's first take found, for its other stages'
// takes (a pixel stage after the vertex stage, a tessellated draw's four) without the store's
// lock. Only this thread pops a result, and only for a later key, so the packet stays while the
// key does.
thread_local PacketResult* t_found = nullptr;
thread_local Key           t_found_key {};

struct GoodPairs {
	std::shared_mutex            mutex;
	std::unordered_set<uint64_t> pairs;
};

GoodPairs& GetGoodPairs() {
	static GoodPairs pairs;
	return pairs;
}

uint64_t PairKey(uint64_t vs_addr, uint64_t ps_addr) {
	// Shader addresses are 256-byte aligned guest addresses below 2^48.
	return (vs_addr >> 8u) * 0x9e3779b97f4a7c15ull ^ (ps_addr >> 8u);
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		// On by default since 2026-10-09 (verify: 0 mismatches over the new games of 2026-10-08
		// and -09; +9 % at the Leap Attack prompt, +13 % in the fight); KYTY_DRAW_PREP=0 turns it off.
		const char* value = std::getenv("KYTY_DRAW_PREP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

int VerifyMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_DRAW_PREP_VERIFY");
		if (value == nullptr || value[0] == '\0' || value[0] == '0') {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool CountsAsDrawOrDispatch(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT: return true;
		default: return false;
	}
}

void ReadSet::Add(uint64_t address, const void* data, uint32_t size) {
	if (size == 0) {
		return;
	}
	const auto offset = static_cast<uint32_t>(m_bytes.size());
	m_bytes.resize(m_bytes.size() + size);
	std::memcpy(m_bytes.data() + offset, data, size);
	if (!m_ranges.empty()) {
		auto& last = m_ranges.back();
		if (last.address + last.size == address && last.offset + last.size == offset) {
			last.size += size;
			return;
		}
	}
	m_ranges.push_back({address, offset, size});
}

// Page by page: a page that is GPU-clean as a whole is compared straight from its backing bytes
// (one dirty-range query per page, as the walk's page cache does); otherwise the range's bytes are
// read with the exact clean read.
bool ReadSet::Validate() const {
	KYTY_PROFILER_BLOCK("DrawPrep::Validate");
	constexpr uint64_t PageSize = 4096;
	struct Page {
		uint64_t       page    = UINT64_MAX;
		const uint8_t* backing = nullptr;
	};
	std::array<Page, 8> pages {};
	size_t              next = 0;
	const auto map = [&](uint64_t page) {
		for (const auto& entry: pages) {
			if (entry.page == page) {
				return entry.backing;
			}
		}
		auto& slot = pages[next++ % pages.size()];
		slot       = {page, LibKernel::Memory::FindGpuCleanBacking(page, PageSize)};
		return slot.backing;
	};
	thread_local std::vector<uint8_t> current;
	for (const auto& range: m_ranges) {
		const uint8_t* expected = m_bytes.data() + range.offset;
		uint64_t       address  = range.address;
		uint64_t       left     = range.size;
		while (left != 0) {
			const uint64_t page    = address & ~(PageSize - 1u);
			const uint64_t in_page = std::min<uint64_t>(left, page + PageSize - address);
			if (const auto* backing = map(page); backing != nullptr) {
				if (std::memcmp(backing + (address - page), expected, in_page) != 0) {
					return false;
				}
			} else if (const auto* range_backing =
			               LibKernel::Memory::FindGpuCleanBacking(address, in_page);
			           range_backing != nullptr) {
				// A page with GPU-written bytes elsewhere: this range compared where it lies, as
				// the copy below would compare it (round 78: those copies were ~2 % of the GPU
				// thread in a race).
				if (std::memcmp(range_backing, expected, in_page) != 0) {
					return false;
				}
			} else {
				if (current.size() < in_page) {
					current.resize(in_page);
				}
				if (!LibKernel::Memory::TryReadGpuCleanBacking(address, current.data(), in_page) ||
				    std::memcmp(current.data(), expected, in_page) != 0) {
					return false;
				}
			}
			address += in_page;
			expected += in_page;
			left -= in_page;
		}
	}
	return true;
}

// gt7-drawmemo-1118, KYTY_SCAN_STAMP (kernel/scanStamp.h).
namespace {

struct StampCounters {
	std::atomic<uint64_t> checks {0};          // validations asked for
	std::atomic<uint64_t> unsealed {0};        // results prepared without stamps
	std::atomic<uint64_t> accepted {0};        // the stamps accepted
	std::atomic<uint64_t> accepted_bytes {0};  // ... comparing some bytes of unwatched pages
	std::atomic<uint64_t> rejected {0};        // unwatched bytes differ
	std::atomic<uint64_t> written {0};         // a page read was stamped since: Validate
	std::atomic<uint64_t> layout {0};          // the mappings changed since: Validate
	std::atomic<uint64_t> ranges {0};          // ranges the stamps looked at
	std::atomic<uint64_t> ranges_watched {0};  // ... on write-protected pages
	std::atomic<uint64_t> false_accepts {0};   // KYTY_SCAN_STAMP=2: accepted, the bytes differ
	std::atomic<uint64_t> false_rejects {0};   // KYTY_SCAN_STAMP=2: rejected, the bytes agree
};
StampCounters g_stamp;

void Bump(std::atomic<uint64_t>& counter, uint64_t value = 1) {
	counter.fetch_add(value, std::memory_order_relaxed);
}

} // namespace

void ReadSet::BeginStamp() {
	m_stamped      = false;
	m_layout_epoch = ScanStamp::LayoutEpoch();
	m_stamp_epoch  = ScanStamp::Epoch();
}

void ReadSet::SealStamp() {
	for (auto& range: m_ranges) {
		range.backing = LibKernel::Memory::FindBackingAny(range.address, range.size);
		range.watched = range.backing != nullptr &&
		                PageManager::HostWriteProtected(range.address, range.size) &&
		                !ScanStamp::OverlapsLiveStack(range.address, range.size);
	}
	m_stamped = true;
}

ReadSet::Verdict ReadSet::StampVerdict(uint64_t& compared, bool& bytes) const {
	if (ScanStamp::LayoutEpoch() != m_layout_epoch) {
		Bump(g_stamp.layout);
		return Verdict::Unknown;
	}
	for (const auto& range: m_ranges) {
		if (range.backing == nullptr ||
		    ScanStamp::WrittenSince(range.address, range.size, m_stamp_epoch)) {
			Bump(g_stamp.written);
			return Verdict::Unknown;
		}
	}
	// Nothing stamped the pages since the read, and the mappings are the same: they are still
	// GPU-clean and their backing is where it was. A write-protected page cannot have changed;
	// another one is compared there.
	uint64_t watched = 0;
	for (const auto& range: m_ranges) {
		if (range.watched) {
			watched++;
			continue;
		}
		bytes = true;
		compared += range.size;
		if (std::memcmp(range.backing, m_bytes.data() + range.offset, range.size) != 0) {
			Bump(g_stamp.ranges, m_ranges.size());
			Bump(g_stamp.ranges_watched, watched);
			return Verdict::Rejected;
		}
	}
	Bump(g_stamp.ranges, m_ranges.size());
	Bump(g_stamp.ranges_watched, watched);
	return Verdict::Accepted;
}

bool ReadSet::ValidateStamped(uint64_t& compared) const {
	const int mode = ScanStamp::Mode();
	if (mode == 0) {
		compared = ByteCount();
		return Validate();
	}
	Bump(g_stamp.checks);
	if (!m_stamped) {
		Bump(g_stamp.unsealed);
		compared = ByteCount();
		return Validate();
	}
	uint64_t   stamp_bytes = 0;
	bool       bytes       = false;
	const auto verdict     = StampVerdict(stamp_bytes, bytes);
	if (verdict == Verdict::Accepted) {
		Bump(g_stamp.accepted);
		if (bytes) {
			Bump(g_stamp.accepted_bytes);
		}
	} else if (verdict == Verdict::Rejected) {
		Bump(g_stamp.rejected);
	}
	if (mode == 1) {
		if (verdict != Verdict::Unknown) {
			compared = stamp_bytes;
			return verdict == Verdict::Accepted;
		}
		compared = ByteCount();
		return Validate();
	}
	// KYTY_SCAN_STAMP=2: the bytes decide; the stamps' verdict is checked against them.
	const bool ok = Validate();
	compared      = ByteCount() + stamp_bytes;
	if (verdict == Verdict::Accepted && !ok) {
		Bump(g_stamp.false_accepts);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::string where;
			for (const auto& range: m_ranges) {
				where += fmt::format(" 0x{:x}+{}{}", range.address, range.size,
				                     range.watched ? "(vigilada)" : "");
			}
			Log::WriteUnmuted(fmt::format(
			    "Sello del escaner: FALSO aceptado (los bytes difieren), rangos:{}\n", where));
		}
	} else if (verdict == Verdict::Rejected && ok) {
		Bump(g_stamp.false_rejects);
	}
	return ok;
}

std::string TakeStampReport() {
	const int mode = ScanStamp::Mode();
	if (mode == 0) {
		return {};
	}
	const auto take = [](std::atomic<uint64_t>& counter) {
		return counter.exchange(0, std::memory_order_relaxed);
	};
	const auto checks         = take(g_stamp.checks);
	const auto unsealed       = take(g_stamp.unsealed);
	const auto accepted       = take(g_stamp.accepted);
	const auto accepted_bytes = take(g_stamp.accepted_bytes);
	const auto rejected       = take(g_stamp.rejected);
	const auto written        = take(g_stamp.written);
	const auto layout         = take(g_stamp.layout);
	const auto ranges         = take(g_stamp.ranges);
	const auto ranges_watched = take(g_stamp.ranges_watched);
	const auto false_accepts  = take(g_stamp.false_accepts);
	const auto false_rejects  = take(g_stamp.false_rejects);
	const auto percent        = [](uint64_t n, uint64_t of) {
        return of != 0 ? 100.0 * static_cast<double>(n) / static_cast<double>(of) : 0.0;
	};
	std::string text = fmt::format(
	    " | sello (KYTY_SCAN_STAMP={}): validaciones {}, aceptadas por sello {} ({:.1f}%; {} "
	    "comparando bytes de paginas no vigiladas), rechazadas por bytes {}, a la comparacion "
	    "completa {} (escritas desde la preparacion {}, mapas cambiados {}, sin sello {}), rangos "
	    "en paginas vigiladas {:.1f}%",
	    mode, checks, accepted, percent(accepted, checks), accepted_bytes, rejected,
	    written + layout + unsealed, written, layout, unsealed, percent(ranges_watched, ranges));
	if (mode == 2) {
		text += fmt::format("; verificador: falsos aceptados {} (tienen que ser 0), rechazos que "
		                    "los bytes aceptan {}",
		                    false_accepts, false_rejects);
	}
	return text;
}

void SetCurrent(const Key& key) {
	t_current     = key;
	t_has_current = true;
	g_progress[key.queue % MaxQueues].store((key.seq << 24u) | (key.ordinal & 0xffffffu),
	                                        std::memory_order_relaxed);
}

void ClearCurrent() {
	t_has_current = false;
}

const Key* Current() {
	return t_has_current ? &t_current : nullptr;
}

Key GpuProgress(uint32_t queue) {
	const auto packed = g_progress[queue % MaxQueues].load(std::memory_order_relaxed);
	return {.queue   = queue,
	        .seq     = packed >> 24u,
	        .ordinal = static_cast<uint32_t>(packed & 0xffffffu)};
}

void Publish(std::unique_ptr<PacketResult> result) {
	if (result == nullptr) {
		return;
	}
	auto&            store = GetStore();
	std::unique_lock lock(store.mutex);
	// While the GPU thread is far behind, wait for it a little; a result that does not fit is
	// dropped (the GPU thread walks that draw itself), so the scanner never waits for good.
	auto& pending = store.pending[result->key.queue % MaxQueues];
	for (int i = 0; i < 100 && pending.size() >= MaxPending; i++) {
		store.space.wait_for(lock, std::chrono::milliseconds(1));
	}
	if (pending.size() >= MaxPending) {
		if (store.recycled.size() < MaxRecycled) {
			store.recycled.push_back(std::move(result));
		}
		return;
	}
	pending.push_back(std::move(result));
}

void Discard(std::unique_ptr<PacketResult> result) {
	if (result == nullptr) {
		return;
	}
	auto&           store = GetStore();
	std::lock_guard lock(store.mutex);
	if (store.recycled.size() < MaxRecycled) {
		store.recycled.push_back(std::move(result));
	}
}

std::unique_ptr<PacketResult> Recycle() {
	std::unique_ptr<PacketResult> result;
	{
		auto&           store = GetStore();
		std::lock_guard lock(store.mutex);
		if (!store.recycled.empty()) {
			result = std::move(store.recycled.back());
			store.recycled.pop_back();
		}
	}
	if (result == nullptr) {
		result = std::make_unique<PacketResult>();
	}
	result->key         = {};
	result->stage_count = 0;
	for (auto& stage: result->stages) {
		stage.ok       = false;
		stage.consumed = false;
	}
	return result;
}

StageResult* Take(uint32_t stage) {
	KYTY_PROFILER_BLOCK("DrawPrep::Take");
	const auto* key = Current();
	Add(Counter::Takes);
	if (key == nullptr) {
		Add(Counter::MissNone);
		Add(Counter::MissNoKey);
		return nullptr;
	}
	if (t_found != nullptr && t_found_key == *key) {
		for (uint32_t i = 0; i < t_found->stage_count; i++) {
			auto& result = t_found->stages[i];
			if (result.ok && result.stage == stage) {
				// The scanner wrote the packet before publishing it, and changes it no more.
				return &result;
			}
		}
		Add(Counter::MissNone);
		Add(Counter::MissStage);
		return nullptr;
	}
	t_found                 = nullptr;
	auto&           store   = GetStore();
	std::lock_guard lock(store.mutex);
	auto&           pending = store.pending[key->queue % MaxQueues];
	bool            popped = false;
	while (!pending.empty() && pending.front()->key.Before(*key)) {
		if (store.recycled.size() < MaxRecycled) {
			store.recycled.push_back(std::move(pending.front()));
		}
		pending.pop_front();
		popped = true;
	}
	if (popped) {
		store.space.notify_one();
	}
	if (pending.empty()) {
		Add(Counter::MissNone);
		Add(Counter::MissEmpty);
		return nullptr;
	}
	auto& front = *pending.front();
	if (front.key.seq != key->seq || front.key.ordinal != key->ordinal) {
		Add(Counter::MissNone);
		Add(Counter::MissLater);
		return nullptr;
	}
	if (front.key.packet != key->packet) {
		Add(Counter::MissNone);
		Add(Counter::MissPacket);
		return nullptr;
	}
	t_found     = &front;
	t_found_key = *key;
	for (uint32_t i = 0; i < front.stage_count; i++) {
		auto& result = front.stages[i];
		if (result.ok && result.stage == stage) {
			// Held by pointer: the scanner's push_back moves only the pointer, and only this
			// thread pops, so the result stays valid until the next Take.
			return &result;
		}
	}
	Add(Counter::MissNone);
	Add(Counter::MissStage);
	return nullptr;
}

void Add(Counter counter, uint64_t value) {
	if (MtGpu::OnWorker()) {
		return; // KYTY_MT_GPU=1/4: a translation worker's walks are not the scanner's
	}
	g_counters[static_cast<size_t>(counter)].value.fetch_add(value, std::memory_order_relaxed);
	if (t_has_current ? t_current.queue != 0 : t_scanner_compute) {
		g_compute_counters[static_cast<size_t>(counter)].value.fetch_add(value,
		                                                                 std::memory_order_relaxed);
	}
}

void SetScannerQueue(uint32_t queue) {
	t_scanner_compute = queue != 0;
}

void LogCountersIfDue() {
	// Round 170: two scanner threads call this.
	static std::mutex                                                mutex;
	std::lock_guard                                                  lock(mutex);
	static std::chrono::steady_clock::time_point last {};
	static std::array<uint64_t, static_cast<size_t>(Counter::Count)> previous {};
	static std::array<uint64_t, static_cast<size_t>(Counter::Count)> previous_compute {};
	const auto now = std::chrono::steady_clock::now();
	if (last.time_since_epoch().count() == 0) {
		last = now;
		return;
	}
	if (now - last < std::chrono::seconds(10)) {
		return;
	}
	last = now;
	std::array<uint64_t, static_cast<size_t>(Counter::Count)> delta {};
	for (size_t i = 0; i < delta.size(); i++) {
		const auto value = g_counters[i].value.load(std::memory_order_relaxed);
		delta[i]         = value - previous[i];
		previous[i]      = value;
	}
	const auto c = [&](Counter counter) { return static_cast<unsigned long long>(delta[static_cast<size_t>(counter)]); };
	LOGF("DrawPrep (10 s): scanned %llu, prepared %llu stages (tessellated draws %llu), skipped "
	     "behind %llu / unknown pair "
	     "%llu / unsupported %llu / no entry %llu, walk failed %llu, tainted %llu (indirect %llu, "
	     "context %llu); GPU takes %llu, "
	     "used %llu (again %llu), miss none %llu (no packet %llu, nothing waiting %llu, a later "
	     "packet waiting %llu, another packet %llu, not this stage %llu, used before %llu) / entry "
	     "%llu / inputs "
	     "%llu / reads %llu; COND_EXEC: scanner %llu (stepped over %llu, stopped %llu), GPU "
	     "thread %llu (stepped over %llu); verify %llu checks, %llu mismatches\n",
	     c(Counter::Scanned), c(Counter::Prepared), c(Counter::TessScanned), c(Counter::SkipBehind),
	     c(Counter::SkipUnknownPair), c(Counter::SkipUnsupported), c(Counter::SkipNoEntry),
	     c(Counter::FailWalk), c(Counter::Tainted), c(Counter::TaintIndirect),
	     c(Counter::TaintContext), c(Counter::Takes), c(Counter::Used) + c(Counter::Reused),
	     c(Counter::Reused), c(Counter::MissNone), c(Counter::MissNoKey), c(Counter::MissEmpty),
	     c(Counter::MissLater), c(Counter::MissPacket), c(Counter::MissStage),
	     c(Counter::MissAgain), c(Counter::MissEntry), c(Counter::MissInputs),
	     c(Counter::MissReads), c(Counter::ScanCondExec), c(Counter::ScanCondSkip),
	     c(Counter::ScanCondStop), c(Counter::GpuCondExec), c(Counter::GpuCondSkip),
	     c(Counter::VerifyChecks), c(Counter::VerifyMismatches));
	for (size_t i = 0; i < delta.size(); i++) {
		const auto value    = g_compute_counters[i].value.load(std::memory_order_relaxed);
		delta[i]            = value - previous_compute[i];
		previous_compute[i] = value;
	}
	LOGF("DrawPrep compute queues (10 s): scanned %llu, prepared %llu, skipped behind %llu / "
	     "unknown shader %llu / unsupported %llu / no entry %llu, walk failed %llu, tainted %llu "
	     "(taints %llu); GPU takes %llu, used %llu, miss none %llu (nothing waiting %llu, a later "
	     "packet waiting %llu, another packet %llu) / entry %llu / inputs %llu / reads %llu\n",
	     c(Counter::Scanned), c(Counter::Prepared), c(Counter::SkipBehind),
	     c(Counter::SkipUnknownPair), c(Counter::SkipUnsupported), c(Counter::SkipNoEntry),
	     c(Counter::FailWalk), c(Counter::Tainted),
	     c(Counter::TaintIndirect) + c(Counter::TaintContext), c(Counter::Takes),
	     c(Counter::Used) + c(Counter::Reused), c(Counter::MissNone), c(Counter::MissEmpty),
	     c(Counter::MissLater), c(Counter::MissPacket), c(Counter::MissEntry),
	     c(Counter::MissInputs), c(Counter::MissReads));
}

void NoteGoodPair(uint64_t vs_addr, uint64_t ps_addr) {
	thread_local std::unordered_set<uint64_t> known;
	const auto                                key = PairKey(vs_addr, ps_addr);
	if (!known.insert(key).second) {
		return;
	}
	auto&           pairs = GetGoodPairs();
	std::unique_lock lock(pairs.mutex);
	pairs.pairs.insert(key);
}

bool IsGoodPair(uint64_t vs_addr, uint64_t ps_addr) {
	auto&            pairs = GetGoodPairs();
	std::shared_lock lock(pairs.mutex);
	return pairs.pairs.contains(PairKey(vs_addr, ps_addr));
}

} // namespace Libs::Graphics::DrawPrep
