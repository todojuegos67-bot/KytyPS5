#pragma once
// (trust1) Delivery 1 of "A acotada" (native/bigpath/NOTAS.md): the XPR depth draws' trusted results, and the
// measurement that decides whether it pays. Render thread only (GuestGpu's), nothing here is touched by other threads
// but DsTrustFeed's atomics.
//
// KYTY_RENDER_STATS=1 (measurement, nothing drawn differently): a "[trust1]" line every 5 s with
//   - the table/native draw CALLS by class: XPR runs (DRAW_INDEX_INDIRECT, multi draws) into a depth image alone, XPR
//     runs into anything else, direct draws (the render lists) into a depth image alone, direct into anything else, the
//     rest (draws of vertices and every try the table path refused): us per call and calls a frame (ClassClock: the
//     whole Try* call, lookups and emission included; the census scopes nested in it are counted in it too);
//   - the calls a frame per depth image (its address, layers; the top 8);
//   - where the XPR depth draws read: one call in 8 has its slot runs (the table plans' reads) and its read-only V#s'
//     ranges classified - static memory the hooks announce / frame memory / other (DsTrustFeed when the audit feeds it,
//     else the memory map's labels: heaps as "static" there, unproven) -, in bytes, by vertex shader (top 6);
//   - how often an XPR depth draw's key (pair, draw state, both stages' user data) was seen before: in the same frame,
//     in the previous one, earlier, or never (the registry's ceiling: a key never seen again is never trusted).
//
// KYTY_DS_TRUSTED_DEPTH (off by default; needs the audit's feed, DsTrustFeed: without it nothing is done, and the line
// says "sin feed"):
//   =2  verification: an XPR depth draw whose key has a trusted entry (stored at an earlier draw of the same key, all
//       its reads in Static memory, none of their granules announced since, no buffer (un)registered over its V#s, same
//       mappings and image starts) is ALSO made the normal way; the entry's block (the slots its runs read, each V#'s
//       5 dwords: device address, size, word, stride), T#/S# words (the image set's key) and LOD words are compared
//       word by word with the normal ones. Counted and listed ("DISCREPANCIA" rows); what is drawn is the normal path's.
//   =1  the trusted draws skip TableEvaluate, TableGather and TableBlocks: the entry's words are put back and its blocks
//       copied into the upload ring. The image set (TableSet: validated as always), the page table ranges
//       (TablePrepareBda), arguments, indices and the emission are as always. Only after =2 is clean.
// Never trusted: written or atomic V#s, indirect images, ring copies (small CPU-written ranges), anything Excluded (the
// 0x20-byte table 0x9f6800-0x9f6b20 rewrites, Bink), Frame, PerUse or Other memory, speculation and other executors.
//
// (draw1) KYTY_DRAW1 (off by default): the same registry with another proof, which needs no feed: an entry's slot runs
// (the words TableEvaluate read, at the addresses it read them) are read again with TableEvaluate's own reader and
// compared with the entry's words, after the mappings and image starts are found the same (so the runs' pages are as
// mapped as when they were read). Equal words at equal addresses from equal user data are what TableEvaluate would read
// now, level after level (a run's address comes from the user data and the slots of earlier levels), so its result
// would be the entry's; the read-only V#s keep the trust1 proofs (no (un)registration over them, the same CPU
// modification epochs). No memory class is asked for (Frame or Other memory is trusted while its words stay the same).
//   =1  an XPR run (DRAW_INDEX_INDIRECT, multi draws) into any target, not only depth, is drawn from a confirmed entry;
//   =2  verification, as KYTY_DS_TRUSTED_DEPTH=2 (the normal path draws; differences listed as "DISCREPANCIA").
// A key whose words changed twice is no longer stored (its draws go the normal way) until the registry forgets.
// With KYTY_DS_TRUSTED_DEPTH set, that variable gives the mode and KYTY_DRAW1 only the proof.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <x86intrin.h>

#include "ds-trust-feed.h"
#include "live-census.h"
#include "loader/demonsSoulsMemMap.h"

namespace TrustedDepth {

// KYTY_DS_TRUSTED_DEPTH: 0 off, 1 trust, 2 verify. Read at its first use (the port sets its variables late).
// (draw1) KYTY_DRAW1: 0 off, 1 trust, 2 verify, with the words as the proof (ByteProof).
[[nodiscard]] inline int Draw1Mode() {
	static const int mode = [] {
		const char* text = std::getenv("KYTY_DRAW1");
		if (text == nullptr || text[0] == '\0' || text[1] != '\0') return 0;
		const int value = text[0] == '1' ? 1 : text[0] == '2' ? 2 : 0;
		if (value != 0) {
			std::printf("draw1: KYTY_DRAW1=%d: XPR table draws %s a registry proven by their words read again (no feed "
			            "needed); the [trust1] line counts it\n",
			            value, value == 1 ? "drawn from" : "compared with");
			std::fflush(stdout);
		}
		return value;
	}();
	return mode;
}
[[nodiscard]] inline bool ByteProof() {
	return Draw1Mode() != 0;
}
[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* text = std::getenv("KYTY_DS_TRUSTED_DEPTH");
		if (text == nullptr || text[0] == '\0' || text[1] != '\0') return Draw1Mode();
		return text[0] == '1' ? 1 : text[0] == '2' ? 2 : 0;
	}();
	return mode;
}
[[nodiscard]] inline bool StatsOn() {
	return LiveCensus::g_render && LiveCensus::RenderStatsOn();
}

// ---- (b) calls by class -------------------------------------------------------------------------------------------
enum Class : uint32_t { XprDepth, XprColor, ListDepth, ListColor, Rest, Classes };
inline constexpr const char* ClassNames[Classes] = {"xpr_profundidad", "xpr_color", "listas_profundidad", "listas_color",
                                                    "resto"};
inline uint64_t g_class[Classes][2] {}; // [class][0 calls, 1 TSC cycles]
// What the table draw of the current Try* call drew into: 0 none (refused, or not a table draw), 1 a depth image
// alone, 2 anything else. Set by TableDraw once drawn.
inline thread_local uint32_t t_target = 0;
// One per CommandProcessor::TryNative*Draw(s) call (kind: 0 runs and multi draws, 1 direct draws, 2 vertices).
class ClassClock {
public:
	explicit ClassClock(uint32_t kind): m_on(StatsOn()) {
		if (m_on) {
			m_kind   = kind;
			t_target = 0;
			m_start  = __rdtsc();
		}
	}
	~ClassClock() {
		if (!m_on) return;
		const uint64_t cycles = __rdtsc() - m_start;
		uint32_t       which  = Rest;
		if (t_target != 0 && m_kind < 2)
			which = m_kind == 0 ? (t_target == 1 ? XprDepth : XprColor) : (t_target == 1 ? ListDepth : ListColor);
		g_class[which][0] += 1;
		g_class[which][1] += cycles;
		t_target = 0;
	}
	ClassClock(const ClassClock&)            = delete;
	ClassClock& operator=(const ClassClock&) = delete;

private:
	bool     m_on;
	uint32_t m_kind  = 0;
	uint64_t m_start = 0;
};

// ---- (b) calls per depth image ------------------------------------------------------------------------------------
struct DepthImage {
	uint64_t address = 0, calls = 0, xpr_calls = 0, draws = 0;
	uint32_t layers = 0;
};
constexpr size_t                         DepthImages = 256;
inline std::array<DepthImage, DepthImages> g_depth_images {};
inline uint64_t                          g_depth_images_dropped = 0;
inline void NoteDepthImage(uint64_t address, uint32_t layers, uint64_t draws, bool xpr) {
	const uint64_t hash = (address * 0x9e3779b97f4a7c15ull) >> 40u;
	for (size_t probe = 0; probe < 16; ++probe) {
		auto& entry = g_depth_images[(hash + probe) & (DepthImages - 1)];
		if (entry.calls != 0 && entry.address != address) continue;
		entry.address = address;
		entry.layers  = layers;
		entry.calls += 1;
		entry.xpr_calls += xpr ? 1 : 0;
		entry.draws += draws;
		return;
	}
	++g_depth_images_dropped;
}

// ---- (b) where the XPR depth draws read ---------------------------------------------------------------------------
enum ReadClass : uint32_t { ReadStatic, ReadFrame, ReadOther, ReadExcluded, ReadPerUse, ReadClasses };
inline constexpr const char* ReadClassNames[ReadClasses] = {"estatica", "frame", "otra", "excluida", "por_uso"};
// The memory map's labels when the audit gives no feed (by 64 KiB granule, forgotten every window: LabelAt scans).
struct LabelClass {
	uint64_t granule = UINT64_MAX;
	uint32_t read    = ReadOther;
};
inline std::array<LabelClass, 4096> g_label_classes {};
[[nodiscard]] inline uint32_t ClassOfLabel(uint64_t address) {
	auto& known = g_label_classes[((address >> 16u) * 0x9e3779b97f4a7c15ull) >> 52u];
	if (known.granule == (address >> 16u)) return known.read;
	char label[32] {};
	uint32_t read = ReadOther;
	if (Loader::DemonsSoulsMemMap::LabelAt(address, label, sizeof(label))) {
		const auto starts = [&](const char* prefix) { return std::strncmp(label, prefix, std::strlen(prefix)) == 0; };
		if (starts("ring") || starts("ctx") || starts("dcb") || starts("DefaultTagg")) read = ReadFrame;
		else if (starts("cpuheap") || starts("gpuheap") || starts("cpupage") || starts("gpupage") || starts("cpubkt") ||
		         starts("gpubkt"))
			read = ReadStatic;
	}
	known = {address >> 16u, read};
	return read;
}
inline bool g_feed_seen = false; // (the last window's classifications came from the feed)
[[nodiscard]] inline uint32_t Classify(uint64_t address, uint64_t size) {
	if (const auto classify = DsTrustFeed::g_classify.load(std::memory_order_acquire); classify != nullptr) {
		g_feed_seen = true;
		switch (classify(address, size)) {
			case DsTrustFeed::Static: return ReadStatic;
			case DsTrustFeed::Frame: return ReadFrame;
			case DsTrustFeed::Excluded: return ReadExcluded;
			case DsTrustFeed::PerUse: return ReadPerUse;
			default: return ReadOther;
		}
	}
	return ClassOfLabel(address);
}
struct ShaderReads {
	uint64_t shader = 0, calls = 0, all_static = 0;
	uint64_t bytes[2][ReadClasses] {}; // [0 slot runs, 1 read-only V# ranges]
};
constexpr size_t                          ShaderReadEntries = 128;
inline std::array<ShaderReads, ShaderReadEntries> g_shader_reads {};
inline uint64_t                           g_shader_reads_dropped = 0;
[[nodiscard]] inline ShaderReads* ReadsOf(uint64_t shader) {
	const uint64_t hash = (shader * 0x9e3779b97f4a7c15ull) >> 40u;
	for (size_t probe = 0; probe < 16; ++probe) {
		auto& entry = g_shader_reads[(hash + probe) & (ShaderReadEntries - 1)];
		if (entry.calls != 0 && entry.shader != shader) continue;
		entry.shader = shader;
		return &entry;
	}
	++g_shader_reads_dropped;
	return nullptr;
}
inline uint64_t g_sample_counter = 0;
[[nodiscard]] inline bool Sample() {
	return (g_sample_counter++ & 7u) == 0;
}

// ---- (b) key repetition -------------------------------------------------------------------------------------------
inline uint64_t g_frame = 1; // flips seen (OnFlip)
struct SeenKey {
	uint64_t key = 0, frame = 0;
};
inline std::array<SeenKey, 16384> g_seen {};
enum Seen : uint32_t { SeenNew, SeenSameFrame, SeenPreviousFrame, SeenOlder, SeenKinds };
inline uint64_t g_seen_counts[SeenKinds] {};
inline void NoteKey(uint64_t key) {
	auto& entry = g_seen[(key * 0x9e3779b97f4a7c15ull) >> 50u];
	if (entry.key != key || entry.frame == 0) ++g_seen_counts[SeenNew];
	else if (entry.frame == g_frame) ++g_seen_counts[SeenSameFrame];
	else if (entry.frame + 1 == g_frame) ++g_seen_counts[SeenPreviousFrame];
	else ++g_seen_counts[SeenOlder];
	entry = {key, g_frame};
}

// (draw1, KYTY_DRAW1 with KYTY_RENDER_STATS) The same repetition for what the registry does not take yet - XPR runs into
// colour (it takes them), the render lists' direct draws and the table dispatches - a key each (pair, state, user data):
// the ceiling of a registry for each, the data a next step needs.
enum Draw1Kind : uint32_t { KindXprColor, KindList, KindDispatch, Draw1Kinds };
inline constexpr const char* Draw1KindNames[Draw1Kinds] = {"xpr_color", "listas", "dispatches"};
inline std::array<std::array<SeenKey, 16384>, Draw1Kinds> g_seen_kind {};
inline uint64_t g_seen_kind_counts[Draw1Kinds][SeenKinds] {};
inline void NoteKindKey(uint32_t kind, uint64_t key) {
	auto& entry  = g_seen_kind[kind][(key * 0x9e3779b97f4a7c15ull) >> 50u];
	auto& counts = g_seen_kind_counts[kind];
	if (entry.key != key || entry.frame == 0) ++counts[SeenNew];
	else if (entry.frame == g_frame) ++counts[SeenSameFrame];
	else if (entry.frame + 1 == g_frame) ++counts[SeenPreviousFrame];
	else ++counts[SeenOlder];
	entry = {key, g_frame};
}

// ---- (c) the registry ---------------------------------------------------------------------------------------------
// The slot runs TableEvaluate read for the draw being made (captured while `on`).
struct Capture {
	struct Run {
		uint32_t st = 0, first = 0, count = 0;
		uint64_t address = 0;
	};
	std::vector<Run> runs;
	bool             on = false;
};
inline thread_local Capture t_capture;
inline void NoteRun(uint32_t st, uint32_t first, uint32_t count, uint64_t address) {
	if (t_capture.on) t_capture.runs.push_back({st, first, count, address});
}

struct Range {
	uint64_t address = 0, size = 0;
	uint32_t generation = 0;
};
struct Entry {
	uint64_t                         key = 0;
	uint64_t                         forget = 0;   // g_forget when stored
	uint32_t                         sightings = 0; // (stored at the second sighting of the key)
	bool                             valid = false;
	bool                             confirmed = false; // a second sighting found the same result, nothing announced
	const void*                      pair = nullptr;
	std::array<const void*, 2>       programs {};
	uint64_t                         state_key = 0;
	const void*                      draw_state = nullptr;
	std::array<std::vector<uint32_t>, 2> user, block;
	std::vector<uint32_t>            words, lod;
	std::vector<Capture::Run>        runs;    // the slot runs (compared in =2)
	std::vector<std::array<uint32_t, 2>> buffer_dwords; // [stage, the V#'s first block dword] (5 dwords each)
	std::vector<Range>               ranges;  // the runs' and the read-only V#s' memory, with its generations
	std::vector<std::pair<uint64_t, uint64_t>> buffers; // the read-only V#s' ranges (registrations)
	std::vector<uint64_t>            cpu_epochs; // ... their BufferCache::CpuModificationEpoch (the tracking's own proof)
	uint64_t                         registration = 0, mapping = 0, start_epoch = 0;
	uint32_t                         changed = 0; // (draw1) times its words were found changed: 2 and it is not stored again
};
// (draw1: 16384, was 4096. Direct-mapped: with ~2 000-3 500 XPR keys a frame at Boletaria [trust1 line], 4096 entries
// would keep only e^(-keys/4096) = 43-58 % of them from one frame to the next; 16384 keeps 81-89 %.)
constexpr size_t   EntryCount = 16384;
inline std::vector<Entry>& Entries() {
	static std::vector<Entry> entries(EntryCount);
	return entries;
}
[[nodiscard]] inline size_t EntryIndex(uint64_t key) {
	return static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 50u);
}
static_assert(EntryCount == size_t {1} << 14u, "EntryIndex takes the top 14 bits");
// Moves when pairs, variants or draw states may have gone (TableReset, TableSweepPairs, TableForgetStates): every
// entry holds pointers to them.
inline uint64_t g_forget = 1;
inline void     Forget() {
	++g_forget;
}
constexpr size_t MaxRanges = 64;

enum Count : uint32_t {
	CEligible,    // XPR depth draws of a pair the registry can hold (mode on)
	CNoFeed,      // ... with no feed: nothing done
	CHit,         // =1: drawn from the entry
	CWouldHit,    // =2: a valid entry, compared
	CStored,      // entries stored or refreshed
	CFirstSight,  // first sighting of a key (not stored yet)
	CInvalidKey,  // an entry of another key or a forgotten one
	CInvalidGen,  // a range announced (or added/removed) since
	CInvalidReg,  // a buffer (un)registered over a V#, or mappings / image starts moved
	CRefuseClass, // a read not Static (refused to store)
	CRefuseRing,  // a V# read from a ring copy
	CRefuseRanges, // more than MaxRanges ranges
	CRefuseDirty,  // a V#'s range with no CPU modification epoch (CPU-dirty or untracked)
	CInvalidCpu,   // a V#'s CPU modification epoch moved since (the protection tracking saw a write)
	CMismatchCalls,
	CMismatchWords,
	CPairIneligible, // XPR depth draws of a pair with written/atomic V#s or indirect images
	CInvalidBytes,   // (draw1) a run's words read again differ from the entry's
	CUnstable,       // (draw1) draws of a key whose words changed twice: not stored again
	CHitColor,       // (draw1) of the hits (=1) or comparisons (=2), draws into anything but a depth image alone
	Counts
};
inline uint64_t g_counts[Counts] {};
inline uint64_t g_refused_class[ReadClasses] {};
struct Mismatch {
	uint64_t shader = 0;
	uint32_t stage = 0, index = 0, trusted = 0, fresh = 0;
	char     what = '?'; // s slot, a run address, b V# dword, w T#/S# word, l LOD word, n count/size
};
inline std::array<Mismatch, 16> g_mismatches {};
inline size_t                   g_mismatch_count = 0;
inline void NoteMismatch(uint64_t shader, uint32_t stage, char what, uint32_t index, uint32_t trusted, uint32_t fresh) {
	++g_counts[CMismatchWords];
	if (g_mismatch_count < g_mismatches.size()) g_mismatches[g_mismatch_count++] = {shader, stage, index, trusted, fresh, what};
}

// ---- the line -----------------------------------------------------------------------------------------------------
// LiveControl::Flip (the render thread): every 5 s while KYTY_RENDER_STATS or KYTY_DS_TRUSTED_DEPTH is on.
inline void OnFlip(int64_t flip_ns) {
	++g_frame;
	const bool stats = LiveCensus::RenderStatsOn();
	if (!stats && Mode() == 0) return;
	static int64_t  last_ns  = 0;
	static uint64_t last_tsc = 0, frames = 0;
	static int64_t  last_list_ns = 0;
	++frames;
	if (last_ns != 0 && flip_ns - last_ns < 5'000'000'000ll) return;
	const uint64_t tsc = __rdtsc();
	if (last_ns != 0 && frames != 0 && tsc > last_tsc) {
		const double n         = static_cast<double>(frames);
		const double wall_ms   = static_cast<double>(flip_ns - last_ns) / 1e6;
		const double us_per_tc = wall_ms * 1e3 / static_cast<double>(tsc - last_tsc);
		std::printf("[trust1] %.1f s, %llu frames", wall_ms / 1e3, static_cast<unsigned long long>(frames));
		if (stats) {
			std::printf(" | llamadas por clase, us/llamada (llamadas/frame):");
			for (uint32_t c = 0; c < Classes; ++c)
				if (g_class[c][0] != 0)
					std::printf(" %s=%.3f (%.0f)", ClassNames[c],
					            static_cast<double>(g_class[c][1]) * us_per_tc / static_cast<double>(g_class[c][0]),
					            static_cast<double>(g_class[c][0]) / n);
			// The depth images, most calls first.
			std::array<const DepthImage*, 8> top {};
			for (const auto& image: g_depth_images) {
				if (image.calls == 0) continue;
				for (size_t i = 0; i < top.size(); ++i)
					if (top[i] == nullptr || image.calls > top[i]->calls) {
						for (size_t j = top.size() - 1; j > i; --j) top[j] = top[j - 1];
						top[i] = &image;
						break;
					}
			}
			std::printf(" | imagenes de profundidad, llamadas/frame (de ellas xpr) capas:");
			for (const auto* image: top)
				if (image != nullptr)
					std::printf(" 0x%llx=%.0f (%.0f) %u", static_cast<unsigned long long>(image->address),
					            static_cast<double>(image->calls) / n, static_cast<double>(image->xpr_calls) / n, image->layers);
			if (g_depth_images_dropped != 0)
				std::printf(" (+%llu sin sitio)", static_cast<unsigned long long>(g_depth_images_dropped));
			// Where the XPR depth draws read (sampled calls).
			std::array<uint64_t, ReadClasses> slots {}, buffers {};
			uint64_t                          calls = 0, all_static = 0;
			for (const auto& entry: g_shader_reads) {
				calls += entry.calls;
				all_static += entry.all_static;
				for (uint32_t r = 0; r < ReadClasses; ++r) {
					slots[r] += entry.bytes[0][r];
					buffers[r] += entry.bytes[1][r];
				}
			}
			const auto share = [](const std::array<uint64_t, ReadClasses>& bytes, uint32_t r) {
				uint64_t total = 0;
				for (const auto value: bytes) total += value;
				return total != 0 ? 100.0 * static_cast<double>(bytes[r]) / static_cast<double>(total) : 0.0;
			};
			std::printf(" | lecturas xpr_profundidad (%s, %llu llamadas muestreadas, todo estatico %.1f%%): slots",
			            g_feed_seen ? "feed de ganchos" : "etiquetas memmap: estatica = monton, sin probar",
			            static_cast<unsigned long long>(calls),
			            calls != 0 ? 100.0 * static_cast<double>(all_static) / static_cast<double>(calls) : 0.0);
			for (uint32_t r = 0; r < ReadClasses; ++r)
				if (slots[r] != 0) std::printf(" %s=%.1f%%", ReadClassNames[r], share(slots, r));
			std::printf(" buffers");
			for (uint32_t r = 0; r < ReadClasses; ++r)
				if (buffers[r] != 0) std::printf(" %s=%.1f%%", ReadClassNames[r], share(buffers, r));
			std::array<const ShaderReads*, 6> shaders {};
			for (const auto& entry: g_shader_reads) {
				if (entry.calls == 0) continue;
				for (size_t i = 0; i < shaders.size(); ++i)
					if (shaders[i] == nullptr || entry.calls > shaders[i]->calls) {
						for (size_t j = shaders.size() - 1; j > i; --j) shaders[j] = shaders[j - 1];
						shaders[i] = &entry;
						break;
					}
			}
			std::printf(" | por shader (llamadas, slots estatica/frame/otra %%):");
			for (const auto* entry: shaders)
				if (entry != nullptr) {
					std::array<uint64_t, ReadClasses> own {};
					for (uint32_t r = 0; r < ReadClasses; ++r) own[r] = entry->bytes[0][r];
					std::printf(" %016llx=%llu %.0f/%.0f/%.0f", static_cast<unsigned long long>(entry->shader),
					            static_cast<unsigned long long>(entry->calls), share(own, ReadStatic), share(own, ReadFrame),
					            share(own, ReadOther));
				}
			const double keys = static_cast<double>(g_seen_counts[0] + g_seen_counts[1] + g_seen_counts[2] + g_seen_counts[3]);
			if (keys != 0)
				std::printf(" | claves xpr_profundidad: nuevas %.1f%%, mismo frame %.1f%%, frame anterior %.1f%%, antes %.1f%%",
				            100.0 * static_cast<double>(g_seen_counts[SeenNew]) / keys,
				            100.0 * static_cast<double>(g_seen_counts[SeenSameFrame]) / keys,
				            100.0 * static_cast<double>(g_seen_counts[SeenPreviousFrame]) / keys,
				            100.0 * static_cast<double>(g_seen_counts[SeenOlder]) / keys);
			if (ByteProof()) {
				std::printf(" | draw1 claves (nuevas/mismo frame/frame anterior/antes %%, llamadas/frame):");
				for (uint32_t k = 0; k < Draw1Kinds; ++k) {
					const auto&  c     = g_seen_kind_counts[k];
					const double total = static_cast<double>(c[0] + c[1] + c[2] + c[3]);
					if (total != 0)
						std::printf(" %s=%.0f/%.0f/%.0f/%.0f (%.0f)", Draw1KindNames[k], 100.0 * static_cast<double>(c[0]) / total,
						            100.0 * static_cast<double>(c[1]) / total, 100.0 * static_cast<double>(c[2]) / total,
						            100.0 * static_cast<double>(c[3]) / total, total / n);
				}
			}
		}
		if (Mode() != 0) {
			const auto per = [&](Count c) { return static_cast<double>(g_counts[c]) / n; };
			if (ByteProof())
				std::printf(" | draw1 (prueba por palabras): de ellas color %.0f, palabras cambiadas %.1f, claves inestables %.1f",
				            per(CHitColor), per(CInvalidBytes), per(CUnstable));
			std::printf(" | confianza=%d%s: elegibles/frame %.0f (pares no aptos %.0f), %s %.0f, guardadas %.1f, primera vez %.1f, "
			            "invalidas: clave %.1f ganchos %.1f registro %.1f cpu %.1f; rechazadas: no estatica %.1f (frame %.1f otra %.1f "
			            "excluida %.1f por_uso %.1f) copia_anillo %.1f rangos %.1f sucias %.1f; DISCREPANCIAS llamadas %llu palabras %llu",
			            Mode(), DsTrustFeed::Live() || ByteProof() ? "" : " SIN FEED DE GANCHOS: NO ACTUA (falta el diff del audit de NOTAS o KYTY_DS_WRITE_AUDIT=1)",
			            per(CEligible), per(CPairIneligible), Mode() == 1 ? "aciertos" : "comparadas", per(Mode() == 1 ? CHit : CWouldHit),
			            per(CStored), per(CFirstSight), per(CInvalidKey), per(CInvalidGen), per(CInvalidReg), per(CInvalidCpu), per(CRefuseClass),
			            static_cast<double>(g_refused_class[ReadFrame]) / n, static_cast<double>(g_refused_class[ReadOther]) / n,
			            static_cast<double>(g_refused_class[ReadExcluded]) / n, static_cast<double>(g_refused_class[ReadPerUse]) / n,
			            per(CRefuseRing), per(CRefuseRanges), per(CRefuseDirty), static_cast<unsigned long long>(g_counts[CMismatchCalls]),
			            static_cast<unsigned long long>(g_counts[CMismatchWords]));
		}
		std::printf("\n");
		// The mismatches (=2), at most every 60 s.
		if (g_mismatch_count != 0 && (last_list_ns == 0 || flip_ns - last_list_ns >= 60'000'000'000ll)) {
			last_list_ns = flip_ns;
			for (size_t i = 0; i < g_mismatch_count; ++i) {
				const auto& m = g_mismatches[i];
				std::printf("[trust1]   DISCREPANCIA vs %016llx etapa %u %c[%u]: confiado 0x%08x, normal 0x%08x\n",
				            static_cast<unsigned long long>(m.shader), m.stage, m.what, m.index, m.trusted, m.fresh);
			}
			g_mismatch_count = 0;
		}
		std::fflush(stdout);
	}
	// A new window.
	for (auto& c: g_class) c[0] = c[1] = 0;
	g_depth_images.fill({});
	g_depth_images_dropped = 0;
	g_shader_reads.fill({});
	g_shader_reads_dropped = 0;
	g_label_classes.fill({});
	g_feed_seen = false;
	for (auto& c: g_seen_counts) c = 0;
	for (auto& kind: g_seen_kind_counts)
		for (auto& c: kind) c = 0;
	// (The mismatch totals stay: a clean verification is one that never counted any.)
	const uint64_t mismatch_calls = g_counts[CMismatchCalls], mismatch_words = g_counts[CMismatchWords];
	for (auto& c: g_counts) c = 0;
	g_counts[CMismatchCalls] = mismatch_calls;
	g_counts[CMismatchWords] = mismatch_words;
	for (auto& c: g_refused_class) c = 0;
	last_ns  = flip_ns;
	last_tsc = tsc;
	frames   = 0;
}

} // namespace TrustedDepth
