#ifndef EMULATOR_SRC_LOCAL_RENDER1_H_
#define EMULATOR_SRC_LOCAL_RENDER1_H_
// (render1 F0+F1) KYTY_RENDER1=1 (off by default; Demon's Souls 01.007.000 only): a probe of the game's mesh draw loop,
// eboot+0xad8bc0 (obj, visible mask, pass descriptor, draw context; called from 0xa1f290). Nothing is replaced: the
// hook (loader/demonsSoulsCopy.cpp) reads what the game's loop is about to do, runs the game's function whole and
// writes nothing the game or the GPU reads. What it reads (Ghidra, projelf):
//   obj+0x38 R base (u64)            obj+0x50 tables (u64*): [0] material breaks (u16), [1] stream breaks (u16),
//   obj+0x60 materials (u16)                                  [2] materials (u64), [3] streams (0x20 each, +0x18 u32),
//   obj+0x62 streams (u16)                                    [4] pass bits per submesh (u32)
//   obj+0x64 submeshes (u16)         pass+0x20 the pass (u32; bit `pass & 31` of [4][i])
//   **ctx: the sce::Agc command buffer the draws go to (bottom, top, cursor_up, ...).
// For each visible submesh i whose pass bit is set the game draws DRAW_INDEX_INDIRECT(offset i * 0x14) after setting
// R = obj+0x38 + (stream[+0x18] & 0xffffff) * 4 (eboot+0x9819d0) when the stream or the material changed.
// F0: per pass, calls / submeshes / visible / draws / material and stream changes.
// F1: one marker per draw {i, i*0x14, R, material, stream} in a side registry keyed by the command buffer range the
// game's function wrote (cursor_up before and after); the command processor, at each DRAW_INDEX_INDIRECT it really
// runs, pairs it with the next marker of the range holding the packet (same order) and counts matches and mismatches.
// A line [render1] every 5 s.
// F2 (KYTY_RENDER1=2; render1-native.inc): the probe is lighter (passes other than p0/p1 are only counted: no submesh
// read, no marker; the pairing reuses the last range), and the p0/p1 depth draws are drawn a second time, the native
// way, into a depth image of their own, compared with the normal depth at the end of each batch: a second line
// "[render1] F2" every 5 s. The image shown is always the normal path's.
// F3 (KYTY_RENDER1=3; render1-native.inc Render1Draw): the p0/p1 depth draws F2 takes are drawn ONLY the native way,
// into the game's own depth image, where the packet is (no batch: the order with every other packet is the walk's);
// the normal path neither translates nor records them. A draw that does not meet F2's conditions, or whose native
// preparation refuses, goes the normal way in its place. The native way does on its own the minimum a draw needs: the
// pair / variant / draw state lookup (nothing created), F2's target checks, the table evaluation of its slots and
// T#/S# words from its user data (R), its image set, its block, its arguments and indices, and the emission. To
// measure, the native way alternates 5 s on / 5 s off in the same process (KYTY_RENDER1_ALTERNA=0: always on) and a
// line "[render1] F3 ON|OFF" every 5 s gives both sides.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <x86intrin.h>

#include "live-census.h" // (F3: the GPU thread's submissions)

namespace Render1 {

// KYTY_RENDER1=1: F0+F1 (observe only). =2: F2 (render1-native.inc): the depth passes p0 and p1 drawn again the native
// way into a depth image of their own and compared with the normal one at the end of each batch; the image shown is
// always the normal path's.
[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_RENDER1");
		if (text == nullptr || text[0] == '\0' || text[1] != '\0') return 0u;
		return text[0] == '1' ? 1u : text[0] == '2' ? 2u : text[0] == '3' ? 3u : 0u;
	}();
	return mode;
}
[[nodiscard]] inline bool On() { return Mode() != 0; }
// (F2) The passes drawn natively (bit p: pass p; Ghidra + the F1 log: p0 and p1 are the depth passes).
constexpr uint32_t NativePasses = 0x3u;
[[nodiscard]] inline bool NativePass(uint32_t pass) { return pass < 32 && ((NativePasses >> pass) & 1u) != 0; }
// (F3) The native way's window: on, or off (the normal path draws everything; the same probe and pairing run).
// KYTY_RENDER1_ALTERNA=0: always on; otherwise each [render1] line (every 5 s) switches it.
inline std::atomic<bool> g_f3_on {true};
[[nodiscard]] inline bool F3Alternates() {
	static const bool alternate = [] {
		const char* text = std::getenv("KYTY_RENDER1_ALTERNA");
		return !(text != nullptr && text[0] == '0' && text[1] == '\0');
	}();
	return alternate;
}
// The hook was installed (the eboot's bytes are 01.007.000's).
inline std::atomic<bool> g_installed {false};

constexpr uint32_t Passes = 16; // (pass values above 15 are counted in 15)

struct Marker {
	uint32_t submesh, offset;
	uint64_t r;
	uint16_t material, stream;
};
struct Record {
	uintptr_t           end = 0;
	uint32_t            pass = 0;
	std::vector<Marker> markers;
	size_t              next = 0;
	uintptr_t           last = 0;
	uint64_t            born_ms = 0;
	bool                count_only = false; // (F2) a pass not drawn natively: its packets are counted, not paired
};

enum Mismatch : uint32_t {
	Offset,      // the packet's offset is not the next marker's (nor one of the 64 after it)
	Gap,         // markers passed over: the game's loop drew no packet for them (or a packet not seen)
	ExtraPm4,    // a packet in a range after its last marker
	NoMarker,    // a packet outside any range (not from the hook's calls)
	Stale,       // markers never reached by a packet (range swept after 2 s or overwritten)
	Split,       // the call's packets did not stay in one block of the command buffer (not registered)
	Repeated,    // a packet seen again (a speculation's, or the run walked twice)
	Overflow,    // too many markers / records (not registered)
	Mismatches
};
inline constexpr const char* MismatchNames[Mismatches] = {"offset", "hueco", "sobra_pm4", "sin_marcador",
                                                          "falta_pm4", "bloque_partido", "repetido", "desbordado"};

struct Stats {
	uint64_t calls[Passes] {}, submeshes[Passes] {}, visible[Passes] {}, draws[Passes] {}, materials[Passes] {},
	    streams[Passes] {}, matched[Passes] {}, matched_depth[Passes] {};
	uint64_t pm4 = 0, pm4_depth = 0, mismatch[Mismatches] {}, hook_cycles = 0, observe_cycles = 0;
	uint64_t pm4_count_only = 0; // (F2) packets in the range of a pass not drawn natively (not paired)
};

inline std::mutex                   g_mutex;
inline Stats                        g_stats; // (under g_mutex)
inline std::map<uintptr_t, Record>  g_ranges; // begin -> record (under g_mutex)
inline uint64_t                     g_markers_held = 0;
inline uint64_t                     g_erase_epoch  = 0; // (F2) moves with every erase from g_ranges (under g_mutex)
constexpr size_t                    MaxMarkersPerCall = 1u << 16u, MaxMarkersHeld = 1u << 21u;

[[nodiscard]] inline uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// --- the hook's side (a guest thread) ---------------------------------------------------------------------------
// What the game's loop will draw (read-only; the same reads the loop makes, in its order).
struct Plan {
	uint32_t            pass = 0, submeshes = 0, visible = 0, materials = 0, streams = 0;
	std::vector<Marker> markers;
	bool                overflow   = false;
	bool                count_only = false; // (F2) a pass not drawn natively: no marker is written
};
inline void Read(uint64_t obj, const uint64_t* mask, uint64_t pass_desc, Plan& plan) {
	plan.pass            = *reinterpret_cast<const uint32_t*>(pass_desc + 0x20);
	// (F2, lighter probe) Another pass is only counted (its calls): no read of its submeshes, no marker.
	if (Mode() >= 2 && !NativePass(plan.pass)) {
		plan.count_only = true;
		return;
	}
	const uint32_t count = *reinterpret_cast<const uint16_t*>(obj + 0x64);
	plan.submeshes       = count;
	if (count == 0 || mask == nullptr) return;
	plan.markers.reserve(count < 4096u ? count : 4096u);
	const auto* tables  = *reinterpret_cast<const uint64_t* const*>(obj + 0x50);
	const auto* mbreak  = reinterpret_cast<const uint16_t*>(tables[0]);
	const auto* sbreak  = reinterpret_cast<const uint16_t*>(tables[1]);
	const auto  streams = tables[3];
	const auto* bits    = reinterpret_cast<const uint32_t*>(tables[4]);
	const uint32_t mcount = *reinterpret_cast<const uint16_t*>(obj + 0x60);
	const uint32_t scount = *reinterpret_cast<const uint16_t*>(obj + 0x62);
	const uint64_t rbase  = *reinterpret_cast<const uint64_t*>(obj + 0x38);
	const uint32_t shift  = plan.pass & 31u;
	const uint32_t words  = (count + 63u) >> 6u;
	uint32_t       m = 0, s = 0, last_m = UINT32_MAX, last_s = UINT32_MAX;
	uint64_t       r = 0;
	for (uint32_t w = 0; w < words; ++w) {
		for (uint64_t word = mask[w]; word != 0; word &= word - 1) {
			const uint32_t i = w * 64u + static_cast<uint32_t>(__builtin_ctzll(word));
			if (i >= count) return;
			plan.visible++;
			if (((bits[i] >> shift) & 1u) == 0) continue;
			while (m < mcount && mbreak[m] <= i) m++;
			while (s < scount && sbreak[s] <= i) s++;
			if (m != last_m) {
				plan.materials++;
				last_m = m;
				last_s = UINT32_MAX; // (the game sets R again after a material)
			}
			if (s != last_s) {
				plan.streams++;
				last_s = s;
				r = s != 0 ? rbase + uint64_t {(*reinterpret_cast<const uint32_t*>(streams + uint64_t {s - 1} * 0x20 + 0x18) &
				                                0xffffffu)}
				                         * 4u
				           : 0;
			}
			if (plan.markers.size() >= MaxMarkersPerCall) {
				plan.overflow = true;
				continue;
			}
			plan.markers.push_back({i, i * 0x14u, r, static_cast<uint16_t>(m), static_cast<uint16_t>(s)});
		}
	}
}

// After the game's function: the range [begin, end) of the command buffer its packets went to (split: another block).
inline void Register(Plan&& plan, uintptr_t begin, uintptr_t end, bool split, uint64_t cycles) {
	const uint32_t p = plan.pass < Passes ? plan.pass : Passes - 1;
	std::lock_guard lock(g_mutex);
	auto& st = g_stats;
	st.calls[p]++;
	st.submeshes[p] += plan.submeshes;
	st.visible[p] += plan.visible;
	st.draws[p] += plan.markers.size();
	st.materials[p] += plan.materials;
	st.streams[p] += plan.streams;
	st.hook_cycles += cycles;
	if (plan.markers.empty() && !plan.count_only) return;
	if (split || end <= begin) {
		st.mismatch[Split] += plan.markers.size();
		return;
	}
	if (plan.overflow || g_markers_held + plan.markers.size() > MaxMarkersHeld) {
		st.mismatch[Overflow] += plan.markers.size();
		return;
	}
	// Ranges this one overlaps are older (the command buffer's memory used again): their markers left are stale.
	auto it = g_ranges.lower_bound(begin);
	if (it != g_ranges.begin() && std::prev(it)->second.end > begin) --it;
	while (it != g_ranges.end() && it->first < end) {
		st.mismatch[Stale] += it->second.markers.size() - it->second.next;
		g_markers_held -= it->second.markers.size();
		it = g_ranges.erase(it);
		++g_erase_epoch;
	}
	Record record;
	record.end        = end;
	record.pass       = p;
	record.born_ms    = NowMs();
	record.count_only = plan.count_only;
	g_markers_held += plan.markers.size();
	record.markers = std::move(plan.markers);
	g_ranges.emplace(begin, std::move(record));
}

// --- the command processor's side --------------------------------------------------------------------------------
// A DRAW_INDEX_INDIRECT packet the processor runs (`depth_only`: z write on, no color target bound). Returns the pass
// of the marker it was paired with (-1: none). (F2, lighter: the range of the last packet is kept per thread and used
// again while no range was erased, instead of a map lookup per packet.)
inline thread_local uint64_t t_last_r = 0; // (F3) the R of the marker the last paired packet took
inline int ObserveDraw(const uint32_t* packet, bool depth_only) {
	const uint64_t start = __rdtsc();
	const auto     at    = reinterpret_cast<uintptr_t>(packet);
	struct Last {
		uintptr_t begin = 0, end = 0;
		Record*   record = nullptr;
		uint64_t  epoch  = UINT64_MAX;
	};
	thread_local Last last;
	int             paired = -1;
	std::lock_guard lock(g_mutex);
	auto& st = g_stats;
	st.pm4++;
	if (depth_only) st.pm4_depth++;
	Record* found = nullptr;
	if (last.record != nullptr && last.epoch == g_erase_epoch && at >= last.begin && at < last.end) {
		found = last.record;
	} else {
		auto it = g_ranges.upper_bound(at);
		if (it != g_ranges.begin() && at < std::prev(it)->second.end) {
			--it;
			found = &it->second;
			last  = {it->first, it->second.end, found, g_erase_epoch};
		}
	}
	if (found == nullptr) {
		st.mismatch[NoMarker]++;
	} else if (found->count_only) {
		st.pm4_count_only++;
	} else {
		auto& rec = *found;
		if (at <= rec.last) {
			st.mismatch[Repeated]++;
		} else {
			rec.last = at;
			if (rec.next >= rec.markers.size()) {
				st.mismatch[ExtraPm4]++;
			} else {
				const size_t limit = std::min(rec.markers.size(), rec.next + 64);
				size_t       k     = rec.next;
				while (k < limit && rec.markers[k].offset != packet[1]) k++;
				if (k == limit) {
					st.mismatch[Offset]++;
				} else {
					st.mismatch[Gap] += k - rec.next;
					rec.next = k + 1;
					st.matched[rec.pass]++;
					if (depth_only) st.matched_depth[rec.pass]++;
					paired   = static_cast<int>(rec.pass);
					t_last_r = rec.markers[k].r; // (F3: the stream's R, for the in-frame repetition count)
				}
			}
		}
	}
	st.observe_cycles += __rdtsc() - start;
	return paired;
}

// --- F2 (KYTY_RENDER1=2): the native depth passes ----------------------------------------------------------------
// The command processor's thread only (the graphics queue's: not a speculation's). What it found about the draw packet
// being processed, for the executor (render1-native.inc).
struct Current {
	bool     native   = false; // paired with a marker of a native pass (p0/p1) and depth only
	bool     consumed = false; // the executor took it up (Render1Begin)
	bool     captured = false; // ... and added it to the batch (Render1Add)
	bool     noted    = false; // ... or counted it under a reason (not "via_normal")
	uint32_t pass     = 0;
	uint64_t args     = 0; // the marker's arguments: args base + i*0x14
	// (F3)
	bool     f3_on         = false; // the native way's window was on at the step's start
	bool     rest_observed = false; // the run's other draws were paired before it was drawn (TryNativeXprDraws)
	bool     drawn         = false; // drawn the native way (Render1Draw)
	bool     f3_noted      = false; // a fallback counted under its reason
	uint64_t r             = 0;     // the first draw's R (its marker's)
};
inline thread_local Current t_current;
// Cycles the executor's native work took during the packet (not the normal path's).
inline thread_local uint64_t t_native_cycles = 0;
// Matched native-pass draws in the packet's step (a run's first packet and the rest after it).
inline thread_local uint32_t t_step_draws = 0;

// Why a draw of p0/p1 was not drawn natively, a batch was not compared, or a batch ended.
enum Reason : uint32_t {
	RViaNormal,     // drawn by the normal path (not a table / native record emission): not reproducible here
	RDirect,        // a direct draw (not DRAW_INDEX_INDIRECT arguments)
	RWrites,        // its shaders write buffers
	RHwrt,          // KYTY_HWRT's capturing pipeline
	RTarget,        // not one depth target alone (color, or no image)
	RLayers,        // layered / mip target
	RMsaa,          // multisampled
	RNoSampled,     // the depth image cannot be sampled (no compare)
	RClear,         // a fast clear pending on the target (the normal path clears first)
	RNoIndex,       // its index buffer is not known (bound before the batch by another path)
	RArgs,          // the arguments are not the marker's (base + i*0x14)
	RRunMixed,      // a run with a draw of another pass or no marker (batch not compared)
	RCbChanged,     // the command buffer changed under an open batch (batch dropped)
	RCap,           // too many batches this frame
	RRingFull,      // no result slot free (batch drawn, not compared)
	RNoResources,   // the shadow image or the compare pipeline could not be made
	RTooMany,       // more draws than a batch holds
	Reasons
};
inline constexpr const char* ReasonNames[Reasons] = {"via_normal", "directo", "escribe", "hwrt", "objetivo", "capas",
                                                     "msaa", "no_muestreable", "clear", "indice", "args", "racha_mixta",
                                                     "cb_cambiado", "tope_lotes", "anillo_lleno", "recursos", "lote_lleno"};
// Why a batch ended (it is drawn and compared then).
enum End : uint32_t { EOther, ETarget, EDrawOther, EPm4End, Ends };
inline constexpr const char* EndNames[Ends] = {"paquete", "objetivo", "draw_otro", "fin_dcb"};

// (F2c) What a p0/p1 draw's shaders read of its depth target (filled by the emission before Render1Begin).
struct ReadInfo {
	bool     known = false, reads_depth = false;
	uint32_t images = 0, base_layer = 0, layers = 0, base_level = 0, levels = 0;
};
inline thread_local ReadInfo t_read;
// (F2c) Per line: the draws by whether they read their depth target, and the most frequent (read vs written) cases.
struct ReadSig {
	uint32_t format = 0, width = 0, height = 0, image_layers = 0, write_layer = 0, reads = 0, read_layer = 0,
	         read_layers = 0, real_feedback = 0, taken = 0;
	bool operator<(const ReadSig& o) const {
		return std::tie(format, width, height, image_layers, write_layer, reads, read_layer, read_layers, real_feedback, taken) <
		       std::tie(o.format, o.width, o.height, o.image_layers, o.write_layer, o.reads, o.read_layer, o.read_layers,
		                o.real_feedback, o.taken);
	}
};
inline std::map<ReadSig, uint64_t> g_read_sigs; // (under g_target_mutex)

// (F2b) Why a draw was refused as "objetivo", and the signature of its target (the most frequent go in the log).
enum TargetWhy : uint32_t {
	TNoDepth,           // no depth target
	TTwoDepths,         // more than one
	TFeedback,          // (F2c) a real attachment feedback loop: the depth attachment in the feedback layout
	TColorOrder,        // the color targets are not the rendering's color attachments, in order, before the depth
	TColorImage,        // a color attachment's image not found
	TNoImage,           // the depth image not found
	TSerial,            // another image object in its slot
	TFormat,            // not a depth format
	TType,              // not a 2D image
	TNoDepthAttachment, // the rendering has no depth attachment
	TargetWhys
};
inline constexpr const char* TargetWhyNames[TargetWhys] = {"sin_prof", "dos_prof", "feedback", "orden_color", "img_color",
                                                           "sin_img", "serial", "formato", "tipo", "sin_adjunto_prof"};
struct TargetSig {
	uint32_t why = 0, format = 0, width = 0, height = 0, layers = 0, samples = 0, type = 0, colors = 0, targets = 0,
	         stencil = 0;
	bool operator<(const TargetSig& o) const {
		return std::tie(why, format, width, height, layers, samples, type, colors, targets, stencil) <
		       std::tie(o.why, o.format, o.width, o.height, o.layers, o.samples, o.type, o.colors, o.targets, o.stencil);
	}
};
inline std::mutex                    g_target_mutex;
inline std::map<TargetSig, uint64_t> g_target_sigs; // (under g_target_mutex; cleared every line)
inline void NoteTarget(const TargetSig& sig, uint64_t n) {
	std::lock_guard lock(g_target_mutex);
	if (g_target_sigs.size() < 256 || g_target_sigs.count(sig) != 0) g_target_sigs[sig] += n;
}
inline void NoteRead(const ReadSig& sig, uint64_t n) {
	std::lock_guard lock(g_target_mutex);
	if (g_read_sigs.size() < 256 || g_read_sigs.count(sig) != 0) g_read_sigs[sig] += n;
}

struct F2Stats {
	std::atomic<uint64_t> native_draws {0}, native_items {0}, batches {0}, batches_compared {0}, batches_tainted {0},
	    batches_dropped {0}, native_cycles {0}, check_cycles {0}, normal_cycles {0}, normal_draws {0}, pixels {0},
	    diff_pixels {0}, results {0}, max_diff_bits {0}, reasons[Reasons] {}, ends[Ends] {};
	void Max(uint32_t bits) {
		uint64_t now = max_diff_bits.load(std::memory_order_relaxed);
		while (bits > now && !max_diff_bits.compare_exchange_weak(now, bits, std::memory_order_relaxed)) {}
	}
};
inline F2Stats g_f2;
inline void Note(Reason reason, uint64_t n = 1) { g_f2.reasons[reason].fetch_add(n, std::memory_order_relaxed); }

// --- F3 (KYTY_RENDER1=3): the native way draws p0/p1 in the game's own image --------------------------------------
// Why a p0/p1 depth draw paired in an "on" window went the normal way (each counts the step's draws).
enum F3Reason : uint32_t {
	FViaNormal,     // never reached the native way (a path before it took it: TryNativeXprDraws refused early, a decal,
	                // a capture, another draw form...)
	FTableOff,      // KYTY_TABLE_XPR off (or the native XPR diag bit 8)
	FBuffer,        // the command buffer invalid, no index buffer, no arguments
	FNoPair,        // its pair, programs, variant, pipeline or draw state not ready (the normal path stores them), or
	                // its shaders changed, or its targets are not current
	FCamera,        // frame generation wants this G-buffer draw's camera (TableTry's rule)
	FHwrt,          // KYTY_HWRT's capturing variant (F2 rule)
	FWrites,        // its shaders write buffers (F2 rule)
	FTarget,        // not one depth target (+ color attachments in order) of a known 2D depth image (F2 "objetivo")
	FFeedback,      // a real attachment feedback loop (F2c)
	FLayers,        // layered / mip rendering (F2)
	FMsaa,          // multisampled (F2)
	FNoSampled,     // the depth image cannot be sampled / copied (F2)
	FClear,         // a fast clear pending on the target (F2)
	FEvaluate,      // the table evaluation refused (as TableDraw's)
	FSet,           // no image set
	FBlocks,        // the blocks refused
	FArgs,          // the arguments' buffer not obtained
	FIndex,         // the indices' buffer not obtained
	FRegistration,  // a buffer registered during the lookups (as TableDraw's)
	FTargets,       // the targets or the set's images could not be placed (as TableDraw's)
	FRunMixed,      // the run has a draw not paired with a p0/p1 marker (or not depth only)
	F3Reasons
};
inline constexpr const char* F3ReasonNames[F3Reasons] = {
	"via_normal", "tabla_off", "cb", "sin_par", "camara_fg", "hwrt", "escribe", "objetivo", "feedback", "capas", "msaa",
	"no_muestreable", "clear", "evaluacion", "set", "bloques", "args", "indices", "registro", "colocar", "racha_mixta"};
// The native way's own work by phase (TSC cycles, its calls taken and refused).
enum F3Phase : uint32_t { F3PLookup, F3PEvaluate, F3PSet, F3PBlocks, F3PBuffers, F3PEmit, F3PRefused, F3Phases };
inline constexpr const char* F3PhaseNames[F3Phases] = {"buscar+comprobar", "evaluar", "set", "bloques", "buffers",
                                                       "emitir", "rechazos"};
struct F3Stats {
	// Steps of paired p0/p1 draws (the processor's whole step, the hook's pairing included): drawn natively, or the
	// normal way by window ([0] off, [1] on: the fallbacks).
	std::atomic<uint64_t> native_draws {0}, native_calls {0}, native_cycles {0};
	std::atomic<uint64_t> normal_draws[2] {}, normal_calls[2] {}, normal_cycles[2] {};
	std::atomic<uint64_t> phase_cycles[F3Phases] {}, reasons[F3Reasons] {};
	// The native calls whose R (the marker's) an earlier native call of the frame had (the R-dependent reads a later
	// phase could share between the cascades and the main depth), and the draws whose set samples their depth image.
	std::atomic<uint64_t> r_calls {0}, r_repeated {0}, reads_own_depth {0};
};
inline F3Stats g_f3;
inline void F3Note(F3Reason reason, uint64_t n) { g_f3.reasons[reason].fetch_add(n, std::memory_order_relaxed); }
// The step of a paired p0/p1 draw ended (`cycles`: its whole time).
inline void F3EndStep(uint64_t cycles, uint32_t draws) {
	auto&          c = t_current;
	const uint64_t n = draws != 0 ? draws : 1;
	if (c.drawn) {
		g_f3.native_draws.fetch_add(n, std::memory_order_relaxed);
		g_f3.native_calls.fetch_add(1, std::memory_order_relaxed);
		g_f3.native_cycles.fetch_add(cycles, std::memory_order_relaxed);
		return;
	}
	const int w = c.f3_on ? 1 : 0;
	g_f3.normal_draws[w].fetch_add(n, std::memory_order_relaxed);
	g_f3.normal_calls[w].fetch_add(1, std::memory_order_relaxed);
	g_f3.normal_cycles[w].fetch_add(cycles, std::memory_order_relaxed);
	if (c.f3_on && !c.f3_noted) F3Note(FViaNormal, n);
}

// The packets that keep a batch open (state; the draw packets decide at their pairing): anything else ends it first
// (a dispatch, a copy, a wait, a write, an unpaired draw...), so that the comparison sees the depth as the pass left it.
// `plain_nop`: an IT_NOP of R_ZERO that is not the emulator's own (the linear copy hook's announcement).
[[nodiscard]] inline bool KeepsBatch(uint32_t opcode, bool plain_nop) {
	switch (opcode) {
		case 0x25: // IT_DRAW_INDEX_INDIRECT (decided when paired)
		case 0x11: // IT_SET_BASE
		case 0x13: // IT_INDEX_BUFFER_SIZE
		case 0x26: // IT_INDEX_BASE
		case 0x2A: // IT_INDEX_TYPE
		case 0x2F: // IT_NUM_INSTANCES
		case 0x63: // IT_SET_SH_REG_INDIRECT
		case 0x64: // IT_SET_UCONFIG_REG_INDIRECT
		case 0x69: // IT_SET_CONTEXT_REG
		case 0x76: // IT_SET_SH_REG
		case 0x79: // IT_SET_UCONFIG_REG
		case 0x9F: // IT_SET_CONTEXT_REG_INDIRECT
			return true;
		case 0x10: return plain_nop; // IT_NOP: the game's markers; the emulator's own codes do work
		default: return false;
	}
}

// (F3) The line of the window that ends (ON: the native way drew; OFF: the normal path drew everything), then the
// window switches. The GPU thread's submissions (LiveCensus Submission scopes, taken with or without
// KYTY_RENDER_STATS) and the frame, per window and over the alternation.
inline void ReportF3(double secs, double frames, double us_per_cycle) {
	static uint64_t last_sub = 0;
	static bool     started  = false;
	static double   acc[2][3] {}; // per window kind: windows, frame ms, submission ms
	static double   normal_off_us = 0.0; // the last OFF window's normal us per p0/p1 draw
	const uint64_t  sub_now = LiveCensus::g_kind_cycles[LiveCensus::Submission];
	const double    sub_ms  = started && sub_now >= last_sub
	                              ? static_cast<double>(sub_now - last_sub) * us_per_cycle / 1000.0 / frames
	                              : 0.0;
	last_sub = sub_now;
	const bool on = g_f3_on.load(std::memory_order_relaxed);
	auto&      g  = g_f3;
	const auto take = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
	const auto f    = [&](uint64_t v) { return static_cast<double>(v) / frames; };
	const auto us   = [&](uint64_t cycles, uint64_t n) {
		return n != 0 ? static_cast<double>(cycles) * us_per_cycle / static_cast<double>(n) : 0.0;
	};
	const uint64_t nd = take(g.native_draws), nc = take(g.native_calls), ncy = take(g.native_cycles);
	uint64_t       od[2], oc[2], ocy[2];
	for (int w = 0; w < 2; ++w) od[w] = take(g.normal_draws[w]), oc[w] = take(g.normal_calls[w]), ocy[w] = take(g.normal_cycles[w]);
	uint64_t phases[F3Phases], reasons[F3Reasons];
	for (uint32_t i = 0; i < F3Phases; ++i) phases[i] = take(g.phase_cycles[i]);
	for (uint32_t i = 0; i < F3Reasons; ++i) reasons[i] = take(g.reasons[i]);
	const uint64_t r_calls = take(g.r_calls), r_rep = take(g.r_repeated), reads = take(g.reads_own_depth);
	const double   frame_ms = secs * 1000.0 / frames;
	const double   native_us = us(ncy, nd);
	const uint64_t normal_n = od[0] + od[1], normal_cy = ocy[0] + ocy[1];
	const double   normal_us = us(normal_cy, normal_n);
	if (!on && od[0] != 0) normal_off_us = us(ocy[0], od[0]);
	std::printf("[render1] F3 %s %.1fs | por frame: nativos=%.0f (llamadas %.1f) p0p1_normales=%.0f (llamadas %.1f) | "
	            "CPU us/draw (paso entero): nativo=%.3f normal=%.3f | us/llamada: nativo=%.3f normal=%.3f | "
	            "coste propio nativo us/llamada:",
	            on ? "ON" : "OFF", secs, f(nd), f(nc), f(normal_n), f(oc[0] + oc[1]), native_us, normal_us,
	            us(ncy, nc), us(normal_cy, oc[0] + oc[1]));
	for (uint32_t i = 0; i < F3PRefused; ++i) std::printf(" %s=%.3f", F3PhaseNames[i], us(phases[i], nc));
	std::printf(" %s=%.3f ms/frame | caidas por frame:", F3PhaseNames[F3PRefused],
	            static_cast<double>(phases[F3PRefused]) * us_per_cycle / 1000.0 / frames);
	for (uint32_t i = 0; i < F3Reasons; ++i)
		if (reasons[i] != 0) std::printf(" %s=%.1f", F3ReasonNames[i], f(reasons[i]));
	std::printf(" | R ya visto en el frame %.1f %% de %.0f llamadas/frame | leen su profundidad %.1f/frame | hilo GPU "
	            "submissions %.2f ms/frame | frame %.2f ms",
	            r_calls != 0 ? 100.0 * static_cast<double>(r_rep) / static_cast<double>(r_calls) : 0.0, f(r_calls),
	            f(reads), sub_ms, frame_ms);
	if (on && nd != 0 && normal_off_us > 0.0)
		std::printf(" | ahorro estimado por draw (normal OFF %.3f - nativo %.3f us) x %.0f = %.3f ms/frame", normal_off_us,
		            native_us, f(nd), (normal_off_us - native_us) * f(nd) / 1000.0);
	if (started && sub_ms > 0.0 && F3Alternates()) {
		auto& a = acc[on ? 1 : 0];
		a[0] += 1.0;
		a[1] += frame_ms;
		a[2] += sub_ms;
	}
	if (acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int w, int i) { return acc[w][i] / acc[w][0]; };
		std::printf(" | ALTERNA (%.0f ventanas OFF / %.0f ON): submissions %.2f -> %.2f ms, frame %.2f -> %.2f ms: AHORRO "
		            "MEDIDO %.3f ms/frame (submissions), %.3f ms/frame (frame)",
		            acc[0][0], acc[1][0], m(0, 2), m(1, 2), m(0, 1), m(1, 1), m(0, 2) - m(1, 2), m(0, 1) - m(1, 1));
	}
	std::printf("\n");
	started = true;
	if (F3Alternates()) g_f3_on.store(!on, std::memory_order_relaxed);
}

// Every 5 s (any caller; `flips` the frames shown so far).
inline void MaybeReport(uint64_t flips) {
	static std::atomic<uint64_t> next_ms {0};
	static uint64_t              last_tsc = 0, last_flips = 0, last_ms = 0;
	const uint64_t               now = NowMs();
	uint64_t                     due = next_ms.load(std::memory_order_relaxed);
	if (now < due || !next_ms.compare_exchange_strong(due, now + 5000)) return;
	Stats st;
	{
		std::lock_guard lock(g_mutex);
		for (auto it = g_ranges.begin(); it != g_ranges.end();) {
			if (now - it->second.born_ms < 2000) {
				++it;
				continue;
			}
			g_stats.mismatch[Stale] += it->second.markers.size() - it->second.next;
			g_markers_held -= it->second.markers.size();
			it = g_ranges.erase(it);
			++g_erase_epoch;
		}
		st      = g_stats;
		g_stats = Stats {};
	}
	const uint64_t tsc = __rdtsc();
	if (due == 0 || last_ms == 0) { // (the first call only starts the window)
		last_tsc = tsc, last_flips = flips, last_ms = now;
		return;
	}
	const double secs   = static_cast<double>(now - last_ms) / 1000.0;
	const double frames = static_cast<double>(flips > last_flips ? flips - last_flips : 1);
	const double us_per_cycle =
	    tsc > last_tsc ? secs * 1e6 / static_cast<double>(tsc - last_tsc) : 0.0;
	last_tsc = tsc, last_flips = flips, last_ms = now;
	const auto f = [&](uint64_t v) { return static_cast<double>(v) / frames; };
	uint64_t   draws = 0, matched = 0, matched_depth = 0, subs = 0, vis = 0, mats = 0, strs = 0, calls = 0;
	for (uint32_t p = 0; p < Passes; ++p) {
		draws += st.draws[p], matched += st.matched[p], matched_depth += st.matched_depth[p];
		subs += st.submeshes[p], vis += st.visible[p], mats += st.materials[p], strs += st.streams[p], calls += st.calls[p];
	}
	std::printf("[render1] %.1fs frames=%.0f | por frame: llamadas=%.1f submallas=%.0f visibles=%.0f draws_hook=%.0f "
	            "materiales=%.1f streams=%.1f | pm4 DII=%.0f (prof=%.0f) | emparejados=%.0f (prof=%.0f) | desajustes",
	            secs, frames, f(calls), f(subs), f(vis), f(draws), f(mats), f(strs), f(st.pm4), f(st.pm4_depth), f(matched),
	            f(matched_depth));
	for (uint32_t i = 0; i < Mismatches; ++i) std::printf(" %s=%.1f", MismatchNames[i], f(st.mismatch[i]));
	std::printf(" | coste hook=%.3f ms/frame obs=%.3f ms/frame |",
	            static_cast<double>(st.hook_cycles) * us_per_cycle / 1000.0 / frames,
	            static_cast<double>(st.observe_cycles) * us_per_cycle / 1000.0 / frames);
	// Per pass value: draws seen by the hook, matched, matched into a depth image alone (the depth pass: prof ~= draws).
	for (uint32_t p = 0; p < Passes; ++p)
		if (st.calls[p] != 0) {
			if (Mode() >= 2 && !NativePass(p))
				std::printf(" p%u: llamadas=%.1f", p, f(st.calls[p]));
			else
				std::printf(" p%u: draws=%.0f emp=%.0f prof=%.0f mat=%.1f", p, f(st.draws[p]), f(st.matched[p]),
				            f(st.matched_depth[p]), f(st.materials[p]));
		}
	if (Mode() >= 2) std::printf(" | pm4 de otras pasadas=%.0f", f(st.pm4_count_only));
	std::printf("\n");
	if (Mode() == 2) {
		// (F2) The native depth passes: the counters since the last line, taken and cleared.
		auto&      g    = g_f2;
		const auto take = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
		const uint64_t ndraws = take(g.native_draws), items = take(g.native_items), batches = take(g.batches),
		               compared = take(g.batches_compared), tainted = take(g.batches_tainted),
		               dropped = take(g.batches_dropped), native_cyc = take(g.native_cycles),
		               check_cyc = take(g.check_cycles), normal_cyc = take(g.normal_cycles),
		               normal_draws = take(g.normal_draws), pixels = take(g.pixels), diff = take(g.diff_pixels),
		               results = take(g.results), max_bits = take(g.max_diff_bits);
		float max_diff = 0.0f;
		const auto bits = static_cast<uint32_t>(max_bits);
		std::memcpy(&max_diff, &bits, sizeof(max_diff));
		const double us = us_per_cycle;
		std::printf("[render1] F2 %.1fs | por frame: nativos=%.0f (llamadas %.1f) lotes=%.1f comparados=%.1f "
		            "contaminados=%.1f perdidos=%.1f | CPU us/draw: nativo=%.3f normal=%.3f (draws normales medidos %.0f) "
		            "| validacion (copia+comparacion, no cuenta) %.3f ms/frame | comparacion: resultados=%llu pixeles=%.0f "
		            "distintos=%llu (%.6f %%) dif_max=%.3g | motivos",
		            secs, f(ndraws), f(items), f(batches), f(compared), f(tainted), f(dropped),
		            ndraws != 0 ? static_cast<double>(native_cyc) * us / static_cast<double>(ndraws) : 0.0,
		            normal_draws != 0 ? static_cast<double>(normal_cyc) * us / static_cast<double>(normal_draws) : 0.0,
		            f(normal_draws), static_cast<double>(check_cyc) * us / 1000.0 / frames,
		            static_cast<unsigned long long>(results), f(pixels), static_cast<unsigned long long>(diff),
		            pixels != 0 ? 100.0 * static_cast<double>(diff) / static_cast<double>(pixels) : 0.0,
		            static_cast<double>(max_diff));
		for (uint32_t i = 0; i < Reasons; ++i) std::printf(" %s=%.1f", ReasonNames[i], f(take(g.reasons[i])));
		std::printf(" | fin de lote");
		for (uint32_t i = 0; i < Ends; ++i) std::printf(" %s=%.1f", EndNames[i], f(take(g.ends[i])));
		std::printf("\n");
		// (F2b) The refusals "objetivo": by cause, and the 4 most frequent targets (vk::Format as its number).
		std::vector<std::pair<uint64_t, TargetSig>> sigs;
		std::vector<std::pair<uint64_t, ReadSig>>   reads;
		{
			std::lock_guard lock(g_target_mutex);
			for (const auto& [sig, n]: g_target_sigs) sigs.emplace_back(n, sig);
			g_target_sigs.clear();
			for (const auto& [sig, n]: g_read_sigs) reads.emplace_back(n, sig);
			g_read_sigs.clear();
		}
		// (F2c) The p0/p1 draws by what their shaders read of their own depth image: read layer(s) vs written layer.
		if (!reads.empty()) {
			std::sort(reads.begin(), reads.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			std::printf("[render1] F2 lectura de la profundidad propia (por frame, 6 mas frecuentes):");
			for (size_t i = 0; i < reads.size() && i < 6; ++i) {
				const auto& rs = reads[i].second;
				std::printf(" [%.1f: vkformat=%u %ux%u capas=%u escribe_capa=%u %s", f(reads[i].first), rs.format, rs.width,
				            rs.height, rs.image_layers, rs.write_layer, rs.reads ? "LEE" : "no_lee");
				if (rs.reads) std::printf(" capa=%u n=%u%s", rs.read_layer, rs.read_layers,
				                          rs.read_layer <= rs.write_layer && rs.write_layer < rs.read_layer + rs.read_layers
				                              ? " (MISMA capa)"
				                              : " (otra capa)");
				std::printf(" feedback_real=%u %s]", rs.real_feedback, rs.taken ? "entra" : "fuera");
			}
			std::printf("\n");
		}
		if (!sigs.empty()) {
			std::array<uint64_t, TargetWhys> by {};
			for (const auto& [n, sig]: sigs) by[sig.why < TargetWhys ? sig.why : 0] += n;
			std::printf("[render1] F2 objetivo por frame:");
			for (uint32_t i = 0; i < TargetWhys; ++i)
				if (by[i] != 0) std::printf(" %s=%.1f", TargetWhyNames[i], f(by[i]));
			std::sort(sigs.begin(), sigs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			std::printf(" | mas frecuentes:");
			for (size_t i = 0; i < sigs.size() && i < 4; ++i) {
				const auto& sg = sigs[i].second;
				std::printf(" [%s %.1f/frame: vkformat=%u %ux%u capas=%u muestras=%u tipo=%u color_adjuntos=%u objetivos=%u "
				            "stencil=%u]",
				            sg.why < TargetWhys ? TargetWhyNames[sg.why] : "?", f(sigs[i].first), sg.format, sg.width,
				            sg.height, sg.layers, sg.samples, sg.type, sg.colors, sg.targets, sg.stencil);
			}
			std::printf("\n");
		}
	}
	if (Mode() == 3) ReportF3(secs, frames, us_per_cycle);
	std::fflush(stdout);
}

} // namespace Render1

#endif // EMULATOR_SRC_LOCAL_RENDER1_H_
