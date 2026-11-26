#ifndef EMULATOR_SRC_LOCAL_PORT_R_H_
#define EMULATOR_SRC_LOCAL_PORT_R_H_
// (port-r) KYTY_PORT_R=0|1|2 (0 by default: nothing installed, the game and the emulator as they are).
//
// The port of Demon's Souls' common geometry list to the host, stage R1 (port-r compilation 1 of 3).
// Every geometry pass (DepthPrePass, GBuffer, the shadow cascades and their lists) draws its object lists through
// eboot+0xab57d0 (01.007.000; 58 references from 27 functions). loader/demonsSoulsPortR.cpp replaces that function by
// a host one (PortR::List, the same calls in the same order: the game's material, draw and binder functions are
// called as they are) and builds, for every draw, the native description of it from the game's own data (Record
// below). Nothing of the emulator's translation is changed in this stage:
//   1  the host loop draws (ON window) or the game's function does (OFF window), alternating every 5 s
//      (KYTY_PORT_R_ALTERNA=0: always ON). In the ON window a table identical to the previous one of the list (same
//      material mask, same ten source pointers) is not allocated and written again: the previous one is passed
//      (the same bytes at another address: what the GPU reads is the same). Line [port-r] every 5 s.
//   2  verification: the host loop draws exactly as the game (every table new, as today), each draw's Record goes to
//      a registry keyed by its table address, and the GPU thread's table draw (local/table-xpr.inc, TableDraw) that
//      carries that address in its user data compares, field by field, what the emulator translated with what the
//      native description says (DIFERENCIAS per field), and which of the draw's slot runs come from which game source
//      and whether their words stay the same while the source pointer does (the data the native emission of
//      compilation 2 is built on). Nothing native is drawn; the image is the emulator's as always.
//
// What the game does per element (Ghidra, 01.007.000, port-r/ghidra; offsets below in GameLayout):
//   ab57d0(owner, view, ctx, list, shadow)  view+0x178+list*0x28: [begin, end) of 0x20-byte elements
//     per element e: inst = e[+8]; obj = inst[0]; mesh = inst[2]; residency touch of obj (8b13f0, obj+0x28 vcalls);
//       obj+0x6d = 1; obj+0xa3 &= ~2; material = obj[+0x20]
//       material changed -> a84ee0(material, ctx): a84de0 (SetShaders 01d108a0 when W[0x37]/W[0x36] change; the
//         blend block 9113a0 -> context registers 01d182f0; 911900 depth/raster), ctx[0x1a] = material+0x4e8 (u48)
//       same material and nothing drawn yet -> skipped
//       abc4b0(mesh, ctx, e+0x14, e+0x18, inst[5]) -> abc100: ctx[0x19] = mesh+0xa8 (u48, instance constants);
//         submesh = mesh[+0x130] + e14*0x48; 9f80a0(submesh): ctx[0x16] = LOD record +0x58; inst[5] != 0 ->
//         a2c160: ctx[0x1f] = inst[5]+8 (u48); aa1e20 -> 9142c0 (index buffer); passes 2/3: a579f0 (stencil)
//       mask = material+0x3f4; table = 7f9cf0(popcount(mask)*8, frame allocator 0x2f643f0, 8, 0) (frame memory, new
//         every draw); table[k] = ctx[0x16 + j] for every set bit j of the mask, in order
//       9819d0(W, table): the table's address as user data of each active stage (W+0x18 bits 0-3: VS, bit 4: PS)
//       aba900(inst, ctx, e+0x18): e18 bit 3 clear (or W+8 bit 2) -> ac3580 -> 914360(W, 4, args+0x38, args+0x3c)
//         = DrawIndexOffset(offset, count) (args = inst[4]); else a2a150 = DrawIndexIndirectMulti
//   after the list: ctx+0x144 bit 3 or ctx[0xd]+0xc96 == 1 -> telemetry pass (CSV): the hook leaves such a call to
//   the game's function whole.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>
#include <x86intrin.h>

namespace PortR {

[[nodiscard]] inline uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* text = std::getenv("KYTY_PORT_R");
		if (text == nullptr || text[0] == '\0' || text[1] != '\0') return 0u;
		return text[0] == '1' ? 1u : text[0] == '2' ? 2u : 0u;
	}();
	return mode;
}
[[nodiscard]] inline bool Alternates() {
	static const bool alternate = [] {
		const char* text = std::getenv("KYTY_PORT_R_ALTERNA");
		return !(text != nullptr && text[0] == '0' && text[1] == '\0');
	}();
	return alternate;
}
inline std::atomic<bool> g_installed {false};
inline std::atomic<bool> g_window_on {true}; // (1) the host loop draws; off: the game's function
// (=2, installed) Off: TableDraw and TableEvaluate pay one relaxed load of this global, nothing else.
inline std::atomic<bool> g_verify {false};
[[nodiscard]] inline bool Verifying() { return g_verify.load(std::memory_order_relaxed); }

// ---- the game's structures (01.007.000; Ghidra) ----
namespace GameLayout {
constexpr uint64_t ListFunction      = 0xab57d0; // the common geometry list
constexpr uint32_t ListFunctionSize  = 0x555;    // to its ud2
constexpr uint64_t ListFunctionFnv   = 0xc7938948622db474ull;
constexpr uint64_t MaterialApply     = 0xa84ee0; // (material, ctx)
constexpr uint64_t DrawSubmesh       = 0xabc4b0; // (mesh, ctx, e+0x14, e+0x18, inst[5])
constexpr uint64_t FrameAlloc        = 0x7f9cf0; // (bytes, descriptor*, align, flags) -> address
constexpr uint64_t FrameAllocDesc    = 0x2f643f0; // 24 bytes copied for every call
constexpr uint64_t BindTable         = 0x9819d0; // (W, table)
constexpr uint64_t IssueDraw         = 0xaba900; // (inst, ctx, e+0x18)
constexpr uint64_t IsDefaultObject   = 0x8b13f0; // (obj) -> bool
// view
constexpr uint64_t ViewLists = 0x178, ViewListStride = 0x28;
// element (0x20)
constexpr uint64_t ElemInst = 0x08, ElemSubmesh = 0x14, ElemFlags = 0x18, ElemSize = 0x20;
// instance (u64 words)
constexpr uint64_t InstObj = 0, InstMesh = 2, InstArgs = 4, InstExtra = 5;
// drawable object
constexpr uint64_t ObjOwner = 0x28, ObjResident = 0x68, ObjKind = 0x6c, ObjDrawn = 0x6d, ObjFlags = 0xa3, ObjMaterial = 0x20;
constexpr uint8_t  ObjKindNoTouch = 0x0b;
// material
constexpr uint64_t MatMask = 0x3f4, MatConstants = 0x4e8;
// draw arguments (DrawIndexOffset)
constexpr uint64_t ArgsOffset = 0x38, ArgsCount = 0x3c;
// draw context (u64 words): [0] W, [0xd] view state (+0xc96: telemetry), [0x16..0x1f] the table's sources, [0x27] pass
constexpr uint64_t CtxW = 0, CtxView = 0xd, CtxSources = 0x16, CtxPass = 0x27;
constexpr uint64_t CtxTelemetryByte = 0x144, ViewTelemetry = 0xc96;
// W (Agc state wrapper): [0] the command buffer
constexpr uint64_t WFlags = 0x08, WStages = 0x18, WBlend = 0x60, WDepth = 0xd0, WViewport = 0x184, WStencil = 0xe8,
                   WPs = 0x1b0, WVs = 0x1b8, WPrim = 0x1a4, WMisc = 0x180;
constexpr uint32_t Sources = 10;
} // namespace GameLayout

inline constexpr const char* SourceNames[GameLayout::Sources + 2] = {
    "malla_lod", "s17", "s18", "instancia", "material", "s1b", "s1c", "s1d", "s1e", "extra", "tabla", "otra"};
constexpr uint32_t SrcTable = GameLayout::Sources, SrcOther = GameLayout::Sources + 1, SrcClasses = GameLayout::Sources + 2;

// ---- the native description of one draw (the hook builds it from the game's data) ----
struct Record {
	uint64_t table = 0;                         // the table's guest address (the user data the binder writes)
	uint32_t mask = 0, words = 0;               // the material's table mask, popcount
	std::array<uint64_t, GameLayout::Sources> src {}; // ctx[0x16..0x1f] at the draw
	uint64_t material = 0, mesh = 0, inst = 0;
	uint64_t vs_obj = 0, ps_obj = 0;            // W[0x37], W[0x36] (the game's shader objects)
	uint64_t state = 0;                         // hash of W's register shadows (blend x8, depth, viewport, stencil, prim)
	uint32_t list = 0, shadow = 0, pass = 0, stages = 0;
	uint32_t path = 0;                          // 0 DrawIndexOffset, 1 DrawIndexIndirectMulti
	uint32_t index_offset = 0, index_count = 0;
	uint32_t submesh = 0;                       // e+0x14 (the mesh's submesh: its index buffer)
	uint64_t born_ms = 0;
};

[[nodiscard]] inline uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
[[nodiscard]] inline uint64_t Mix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
	return h * 0xff51afd7ed558ccdull;
}
[[nodiscard]] inline uint64_t HashWords(const uint32_t* w, uint32_t n) {
	uint64_t h = 0xcbf29ce484222325ull ^ n;
	for (uint32_t i = 0; i < n; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
	return h;
}

// ---- the game thread's counters (the hook) ----
enum Fall : uint32_t { FTelemetry, FWindowOff, FSkipped, FIndirect, FNoMaterial, FRegistryFull, Falls };
inline constexpr const char* FallNames[Falls] = {"telemetria", "ventana_off", "saltado_como_juego", "indirecto",
                                                 "sin_material", "registro_lleno"};
struct GameStats {
	std::atomic<uint64_t> calls[2] {}, elements[2] {}, draws[2] {}, cycles[2] {}; // [0] OFF / game's, [1] host loop
	std::atomic<uint64_t> materials {0}, tables_new {0}, tables_reused {0}, falls[Falls] {}, records {0};
	std::atomic<uint64_t> per_pass[16] {};
};
inline GameStats g_game;

// ---- the registry (KYTY_PORT_R=2): table address -> record ----
inline std::mutex                             g_rec_mutex;
inline std::unordered_map<uint64_t, Record>   g_records; // (under g_rec_mutex)
constexpr size_t                              MaxRecords = size_t {1} << 18u;
inline uint64_t                               g_replaced = 0, g_aged = 0; // (under g_rec_mutex)
inline bool Put(const Record& r) {
	std::lock_guard lock(g_rec_mutex);
	if (g_records.size() >= MaxRecords && g_records.count(r.table) == 0) return false;
	auto [it, fresh] = g_records.try_emplace(r.table, r);
	if (!fresh) {
		++g_replaced; // (a table address of 2 frames ago, its draw never paired)
		it->second = r;
	}
	return true;
}

// ---- the GPU thread (local/table-xpr.inc, TableDraw / TableEvaluate) ----
struct Run {
	uint32_t st = 0, count = 0;
	uint64_t address = 0, hash = 0;
	std::array<uint32_t, 24> head {};
};
struct Capture {
	bool             on = false, ended = false;
	Record           rec;
	std::vector<Run> runs;
	uint64_t         start = 0;
	uint64_t         own   = 0; // End's own cycles (taken out of the draw's clock: the verification's cost)
};
inline thread_local Capture t_cap;
inline thread_local bool    t_capture = false; // (one load in TableEvaluate's read_run when off)
// A run TableEvaluate read (its words just read into `out`).
inline void NoteRun(uint32_t st, uint64_t address, uint32_t count, const uint32_t* out) {
	if (t_cap.runs.size() >= 256) return;
	Run r;
	r.st      = st;
	r.count   = count;
	r.address = address;
	r.hash    = HashWords(out, count);
	std::copy_n(out, std::min<uint32_t>(count, 24u), r.head.begin());
	t_cap.runs.push_back(r);
}

enum Field : uint32_t {
	DUdVs,      // the table's address not in the VS user data (the binder's stage bits say VS)
	DUdPs,      // ... nor in the PS user data (bit 4)
	DTable,     // the table's words read by the emulator are not the pointers the game wrote (the record's)
	DTableNot,  // no run of the evaluation read the table (its slots come from elsewhere)
	DPair,      // the same game shaders + state gave another shader pair (vs, ps) than before
	DState,     // ... another pipeline state (variant)
	DTargets,   // ... another draw state (targets)
	DTextures,  // the same material + shaders gave other T#/S# words than before
	DCount,     // DrawIndexOffset: another index count than the game's arguments
	DStart,     // ... the index buffer implied (first index address - the game's offset) moved for the same submesh
	DForm,      // the emulator drew it not as one direct indexed draw
	DNotEval,   // the evaluation did not run (trust1 entry, or refused): nothing compared but the draw
	Fields
};
inline constexpr const char* FieldNames[Fields] = {"ud_vs",   "ud_ps",    "tabla_mem", "tabla_no_leida",
                                                   "par",     "estado",   "objetivos", "texturas",
                                                   "args_cuenta", "args_inicio", "args_forma", "no_evaluado"};
struct GpuStats {
	uint64_t paired = 0, compared = 0, covered = 0, learned_pipe = 0, learned_tex = 0;
	uint64_t diff[Fields] {};
	uint64_t src_same[SrcClasses] {}, src_changed[SrcClasses] {}, src_new[SrcClasses] {}, src_words[SrcClasses] {};
	uint64_t draw_cycles = 0, draw_n = 0, eval_cycles = 0, eval_n = 0;
	uint64_t examples = 0;
};
inline std::mutex g_gpu_mutex;
inline GpuStats   g_gpu; // (under g_gpu_mutex)
struct PipeSeen {
	uint64_t vs = 0, ps = 0, state = 0;
	const void* draw_state = nullptr;
};
inline std::unordered_map<uint64_t, PipeSeen> g_pipes;    // (under g_gpu_mutex) game shaders + state -> emulator's
inline std::unordered_map<uint64_t, uint64_t> g_textures; // material + shaders -> T#/S# words hash
inline std::unordered_map<uint64_t, uint64_t> g_sources;  // (pair, stage, run, source pointer) -> words hash
inline std::unordered_map<uint64_t, uint64_t> g_bases;    // (mesh, submesh) -> index buffer implied by the game's offset
constexpr size_t MaxMemo = size_t {1} << 20u;

// Before TableEvaluate: the record whose table address the draw's user data carries (true: found, capture on).
inline bool Begin(std::span<const uint32_t> ud_vs, std::span<const uint32_t> ud_ps) {
	t_cap.on = t_cap.ended = false;
	t_cap.own = 0;
	t_capture = false;
	const std::array<std::span<const uint32_t>, 2> stages {ud_vs, ud_ps};
	std::lock_guard lock(g_rec_mutex);
	if (g_records.empty()) return false;
	for (const auto& ud: stages) {
		for (size_t i = 0; i + 1 < ud.size(); ++i) {
			const uint64_t v = uint64_t {ud[i]} | uint64_t {ud[i + 1]} << 32u;
			if ((v & 7u) != 0 || (v >> 48u) != 0 || v == 0) continue;
			auto it = g_records.find(v);
			if (it == g_records.end()) continue;
			t_cap.rec = it->second;
			g_records.erase(it);
			t_cap.runs.clear();
			t_cap.on    = true;
			t_cap.start = __rdtsc();
			t_capture   = true;
			return true;
		}
	}
	return false;
}
[[nodiscard]] inline bool UdHas(std::span<const uint32_t> ud, uint64_t v) {
	for (size_t i = 0; i + 1 < ud.size(); ++i)
		if (ud[i] == static_cast<uint32_t>(v) && ud[i + 1] == static_cast<uint32_t>(v >> 32u)) return true;
	return false;
}
// After TableEvaluate (+ TableGather): the comparison. `evaluated` false: trust1 entry (no runs captured).
struct Translated {
	uint64_t                  vs = 0, ps = 0, state = 0;
	const void*               draw_state = nullptr;
	std::span<const uint32_t> words;          // T#/S# words (table.words)
	std::span<const uint32_t> ud_vs, ud_ps;
	bool                      evaluated = true;
	bool                      direct = false; // one direct indexed draw
	uint64_t                  index_base = 0, index_address = 0; // (TableDraw's index_base: the first index's address)
	uint32_t                  index_size = 2, index_count = 0;
};
inline void End(const Translated& t) {
	if (!t_cap.on || t_cap.ended) return;
	t_cap.ended     = true;
	t_capture       = false;
	const auto& rec = t_cap.rec;
	const uint64_t end_start   = __rdtsc();
	const uint64_t eval_cycles = end_start - t_cap.start;
	struct Own {
		uint64_t s;
		~Own() { t_cap.own = __rdtsc() - s; }
	} own {end_start};
	std::array<bool, Fields> d {};
	// (`translated` false: TableDraw refused the draw before the comparison: only "no_evaluado" is counted.)
	const bool translated = t.vs != 0 || t.ps != 0;
	if (translated && (rec.stages & 0x0fu) != 0 && !UdHas(t.ud_vs, rec.table)) d[DUdVs] = true;
	if (translated && (rec.stages & 0x10u) != 0 && !UdHas(t.ud_ps, rec.table)) d[DUdPs] = true;
	std::lock_guard lock(g_gpu_mutex);
	auto& g = g_gpu;
	g.paired++;
	g.eval_cycles += eval_cycles;
	g.eval_n++;
	if (!translated) {
		g.diff[DNotEval]++;
		g.compared++;
		return;
	}
	if (!t.evaluated) d[DNotEval] = true;
	// The runs: which source each comes from, the table's words against the record's, and per source whether the
	// words stay the same while its pointer does.
	bool table_read = false;
	if (g_sources.size() > MaxMemo) g_sources.clear();
	for (size_t i = 0; i < t_cap.runs.size(); ++i) {
		const auto& run = t_cap.runs[i];
		uint32_t    cls = SrcOther;
		uint64_t    base = run.address;
		const uint64_t table_end = rec.table + uint64_t {rec.words} * 8u;
		if (run.address >= rec.table && run.address < table_end) {
			cls        = SrcTable;
			table_read = true;
			// The record's pointers as dwords, from the run's first slot.
			std::array<uint32_t, 20> mine {};
			uint32_t k = 0;
			for (uint32_t j = 0; j < GameLayout::Sources; ++j)
				if ((rec.mask >> j) & 1u) {
					mine[2 * k]     = static_cast<uint32_t>(rec.src[j]);
					mine[2 * k + 1] = static_cast<uint32_t>(rec.src[j] >> 32u);
					++k;
				}
			const uint64_t first = (run.address - rec.table) / 4u;
			const uint32_t n     = std::min<uint32_t>(run.count, 24u);
			for (uint32_t w = 0; w < n && first + w < 2u * rec.words; ++w)
				if (run.head[w] != mine[first + w]) d[DTable] = true;
		} else {
			uint64_t best = 0;
			for (uint32_t j = 0; j < GameLayout::Sources; ++j) {
				const uint64_t p = rec.src[j] & 0x0000ffffffffffffull;
				if (((rec.mask >> j) & 1u) == 0 || p == 0 || run.address < p || run.address - p >= 0x10000u) continue;
				if (p >= best) {
					best = p;
					cls  = j;
				}
			}
			if (cls != SrcOther) base = best;
		}
		g.src_words[cls] += run.count;
		const uint64_t key = Mix(Mix(Mix(Mix(t.vs, t.ps), run.st), i), base * 31u + (run.address - base));
		auto [it, fresh]   = g_sources.try_emplace(key, run.hash);
		if (fresh) g.src_new[cls]++;
		else if (it->second == run.hash) g.src_same[cls]++;
		else {
			g.src_changed[cls]++;
			it->second = run.hash;
		}
	}
	if (t.evaluated && !table_read) d[DTableNot] = true;
	// The pipeline the game's shaders and state map to.
	if (g_pipes.size() > MaxMemo) g_pipes.clear();
	if (g_textures.size() > MaxMemo) g_textures.clear();
	const uint64_t pkey = Mix(Mix(Mix(Mix(Mix(rec.vs_obj, rec.ps_obj), rec.state), rec.list), rec.shadow), rec.pass);
	bool           pipe_known = false, tex_known = false;
	if (auto it = g_pipes.find(pkey); it == g_pipes.end()) {
		g_pipes.emplace(pkey, PipeSeen {t.vs, t.ps, t.state, t.draw_state});
		g.learned_pipe++;
	} else {
		pipe_known = true;
		auto& s    = it->second;
		if (s.vs != t.vs || s.ps != t.ps) d[DPair] = true;
		else if (s.state != t.state) d[DState] = true;
		if (s.draw_state != t.draw_state) d[DTargets] = true;
		s = {t.vs, t.ps, t.state, t.draw_state};
	}
	if (t.evaluated) {
		const uint64_t tkey = Mix(Mix(Mix(rec.material, rec.vs_obj), rec.ps_obj), rec.pass);
		const uint64_t th   = HashWords(t.words.data(), static_cast<uint32_t>(t.words.size()));
		if (auto it = g_textures.find(tkey); it == g_textures.end()) {
			g_textures.emplace(tkey, th);
			g.learned_tex++;
		} else {
			tex_known = true;
			if (it->second != th) {
				d[DTextures] = true;
				it->second   = th;
			}
		}
	}
	if (rec.path == 0) {
		if (!t.direct) d[DForm] = true;
		else {
			if (t.index_count != rec.index_count) d[DCount] = true;
			// (TableDraw gets the draw's first index address, not the buffer's: the buffer the game's offset implies
			// is learned per submesh and must stay.)
			const uint64_t implied = t.index_address - uint64_t {rec.index_offset} * t.index_size;
			if (g_bases.size() > MaxMemo) g_bases.clear();
			auto [it, fresh] = g_bases.try_emplace(Mix(rec.mesh, rec.submesh), implied);
			if (!fresh && it->second != implied) {
				d[DStart]  = true;
				it->second = implied;
			}
		}
	}
	bool any = false;
	for (uint32_t f = 0; f < Fields; ++f)
		if (d[f]) {
			g.diff[f]++;
			any = true;
		}
	g.compared++;
	if (!any && pipe_known && tex_known) g.covered++;
	if (any && g.examples < 24) {
		++g.examples;
		std::printf("[port-r] DIFERENCIA tabla=0x%llx lista=%u sombra=%u pase=%u material=0x%llx vs_obj=0x%llx "
		            "ps_obj=0x%llx par=0x%llx/0x%llx cuenta juego=%u emu=%u offset juego=%u direccion emu=0x%llx campos:",
		            static_cast<unsigned long long>(rec.table), rec.list, rec.shadow, rec.pass,
		            static_cast<unsigned long long>(rec.material), static_cast<unsigned long long>(rec.vs_obj),
		            static_cast<unsigned long long>(rec.ps_obj), static_cast<unsigned long long>(t.vs),
		            static_cast<unsigned long long>(t.ps), rec.index_count, t.index_count, rec.index_offset,
		            static_cast<unsigned long long>(t.direct ? t.index_address : 0));
		for (uint32_t f = 0; f < Fields; ++f)
			if (d[f]) std::printf(" %s", FieldNames[f]);
		std::printf("\n");
	}
}
// TableDraw's whole step for a paired draw (RAII: every return of TableDraw).
struct DrawClock {
	bool     on    = false;
	uint64_t start = 0;
	explicit DrawClock(bool paired): on(paired), start(paired ? __rdtsc() : 0) {}
	DrawClock(const DrawClock&)            = delete;
	DrawClock& operator=(const DrawClock&) = delete;
	~DrawClock() {
		if (!on) return;
		if (!t_cap.ended) {
			// (refused before the comparison: an evaluation that failed, no set...) counted as not evaluated.
			Translated none;
			none.evaluated = false;
			End(none);
		}
		t_capture = false;
		t_cap.on  = false;
		const uint64_t all = __rdtsc() - start;
		const uint64_t c   = all > t_cap.own ? all - t_cap.own : 0;
		std::lock_guard lock(g_gpu_mutex);
		g_gpu.draw_cycles += c;
		g_gpu.draw_n++;
	}
};

// ---- the [port-r] line (every 5 s, from the hook's thread) ----
// `submission_cycles`: LiveCensus::g_kind_cycles[Submission] now (the GPU thread's submissions, always taken).
inline void Report(double secs, double frames, double us_per_cycle, uint64_t submission_cycles) {
	static uint64_t last_sub = 0;
	static bool     started  = false;
	static double   acc[2][3] {}; // per window kind (OFF, ON): windows, frame ms, submission ms
	if (frames < 1.0) frames = 1.0;
	const double sub_ms = started && submission_cycles >= last_sub
	                          ? static_cast<double>(submission_cycles - last_sub) * us_per_cycle / 1000.0 / frames
	                          : 0.0;
	last_sub            = submission_cycles;
	const bool on       = Mode() == 2 || g_window_on.load(std::memory_order_relaxed);
	auto&      g        = g_game;
	const auto take     = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
	const auto f        = [&](uint64_t v) { return static_cast<double>(v) / frames; };
	const auto us       = [&](uint64_t cycles, uint64_t n) {
        return n != 0 ? static_cast<double>(cycles) * us_per_cycle / static_cast<double>(n) : 0.0;
	};
	uint64_t calls[2], elems[2], draws[2], cyc[2];
	for (int w = 0; w < 2; ++w)
		calls[w] = take(g.calls[w]), elems[w] = take(g.elements[w]), draws[w] = take(g.draws[w]), cyc[w] = take(g.cycles[w]);
	const uint64_t mats = take(g.materials), tnew = take(g.tables_new), treused = take(g.tables_reused),
	               recs = take(g.records);
	uint64_t falls[Falls], passes[16];
	for (uint32_t i = 0; i < Falls; ++i) falls[i] = take(g.falls[i]);
	for (uint32_t i = 0; i < 16; ++i) passes[i] = take(g.per_pass[i]);
	const double frame_ms = secs * 1000.0 / frames;
	std::printf("[port-r] modo=%u %s %.1fs %.0f frames | lista ab57d0 por frame: llamadas=%.1f (nativas %.1f, del juego %.1f) "
	            "draws=%.0f (nativos %.0f, del juego <=%.0f) materiales=%.0f tablas nuevas=%.0f reutilizadas=%.0f (%.1f %%) | "
	            "us/elemento hilo del juego (lista entera): nativo=%.3f juego=%.3f | caidas/frame:",
	            Mode(), Mode() == 2 ? "VERIFICA" : (on ? "ON" : "OFF"), secs, frames, f(calls[0] + calls[1]), f(calls[1]),
	            f(calls[0]), f(draws[0] + draws[1]), f(draws[1]), f(draws[0]), f(mats), f(tnew), f(treused),
	            tnew + treused != 0 ? 100.0 * static_cast<double>(treused) / static_cast<double>(tnew + treused) : 0.0,
	            us(cyc[1], elems[1]), us(cyc[0], elems[0]));
	for (uint32_t i = 0; i < Falls; ++i)
		if (falls[i] != 0) std::printf(" %s=%.1f", FallNames[i], f(falls[i]));
	std::printf(" | draws por pase (ctx[0x27]):");
	for (uint32_t i = 0; i < 16; ++i)
		if (passes[i] != 0) std::printf(" p%u=%.0f", i, f(passes[i]));
	std::printf(" | hilo GPU submissions %.2f ms/frame | frame %.2f ms", sub_ms, frame_ms);
	if (Mode() == 2) {
		GpuStats gs;
		{
			std::lock_guard lock(g_gpu_mutex);
			gs    = g_gpu;
			g_gpu = {};
			g_gpu.examples = gs.examples;
		}
		uint64_t aged = 0, replaced = 0, held = 0;
		{
			const uint64_t now = NowMs();
			std::lock_guard lock(g_rec_mutex);
			for (auto it = g_records.begin(); it != g_records.end();) {
				if (now - it->second.born_ms > 3000) {
					++aged;
					it = g_records.erase(it);
				} else {
					++it;
				}
			}
			replaced   = g_replaced;
			g_replaced = 0;
			held       = g_records.size();
		}
		const auto pct = [](uint64_t a, uint64_t b) { return b != 0 ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0; };
		std::printf(" | GPU (registros %.0f/frame): emparejados=%.0f/frame (%.1f %%) no_vistos=%.0f/frame (+%.0f pisados, %llu "
		            "pendientes) | us/draw traduccion TableDraw=%.3f evaluar+gather=%.3f | aprendidos: pipelines=%.0f texturas=%.0f "
		            "| DIFERENCIAS (draws/frame de %.0f comparados):",
		            f(recs), f(gs.paired), pct(gs.paired, recs), f(aged), f(replaced), static_cast<unsigned long long>(held),
		            us(gs.draw_cycles, gs.draw_n), us(gs.eval_cycles, gs.eval_n), f(gs.learned_pipe), f(gs.learned_tex),
		            f(gs.compared));
		for (uint32_t i = 0; i < Fields; ++i) std::printf(" %s=%.1f", FieldNames[i], f(gs.diff[i]));
		std::printf(" | CUBIERTOS (todo igual y ya aprendido) %.0f/frame = %.1f %% de los comparados | TECHO si fueran gratis: "
		            "%.3f ms/frame | fuentes (runs igual/cambia/nuevo, palabras/draw):",
		            f(gs.covered), pct(gs.covered, gs.compared),
		            static_cast<double>(gs.covered) * us(gs.draw_cycles, gs.draw_n) / 1000.0 / frames);
		for (uint32_t i = 0; i < SrcClasses; ++i)
			if (gs.src_same[i] + gs.src_changed[i] + gs.src_new[i] != 0)
				std::printf(" %s=%.0f/%.0f/%.0f (%.1f)", SourceNames[i], f(gs.src_same[i]), f(gs.src_changed[i]),
				            f(gs.src_new[i]),
				            gs.paired != 0 ? static_cast<double>(gs.src_words[i]) / static_cast<double>(gs.paired) : 0.0);
	}
	if (Mode() == 1 && started && sub_ms > 0.0 && Alternates()) {
		auto& a = acc[on ? 1 : 0];
		a[0] += 1.0;
		a[1] += frame_ms;
		a[2] += sub_ms;
	}
	if (acc[0][0] > 0 && acc[1][0] > 0) {
		const auto m = [&](int w, int i) { return acc[w][i] / acc[w][0]; };
		std::printf(" | ALTERNA (%.0f ventanas OFF / %.0f ON): submissions %.2f -> %.2f ms, frame %.2f -> %.2f ms: AHORRO "
		            "MEDIDO %.3f ms/frame (hilo GPU), %.3f ms/frame (frame)",
		            acc[0][0], acc[1][0], m(0, 2), m(1, 2), m(0, 1), m(1, 1), m(0, 2) - m(1, 2), m(0, 1) - m(1, 1));
	}
	std::printf("\n");
	std::fflush(stdout);
	started = true;
	if (Mode() == 1 && Alternates()) g_window_on.store(!on, std::memory_order_relaxed);
}

} // namespace PortR

#endif // EMULATOR_SRC_LOCAL_PORT_R_H_
