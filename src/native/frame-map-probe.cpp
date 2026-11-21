// (f5) KYTY_NATIVE_RENDER=map: the frame map at run time (native-render.h).
//
// Game threads (AgcCbReleaseMem): a GPU-clock RELEASE_MEM of event 0x28 may be a profiler scope's. The record's name
// pointers are read there (the thread that just wrote the record): +0x48 from the address (a begin's record) and
// +0x28 (an end's: the address is record + 0x20). Both kept, with the address, in a side table keyed by the packet's
// address (a seqlock per slot).
//
// GPU thread: at each such packet it translates, the side table says which name; the queue's stack of open scopes
// decides begin or end (an end's address is the open record + 0x20). Between checkpoints (slice begin / end, scope
// marks, flip) the GPU thread's time and the PM4 draws / dispatches / WAIT_REG_MEM it counted
// (LiveCounters::g_pm4) are charged to the innermost open pass of the queue being translated (exclusive) and to every
// open one (inclusive). Every 5 s: "[f5]" lines, ranked by exclusive ms per frame.
#include "native/frame-map-probe.h"
#include "native/native-render.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"

#include "local/live-counters.h"
#include "local/pathtrace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace NativeRender {
namespace {

// ---- the bytes this relies on (01.007.000; FNV-1a 64, as port2.h) ----
struct EbootRange {
	uint64_t    offset;
	uint32_t    size;
	uint64_t    fnv;
	const char* what;
};
constexpr EbootRange Checked[] = {
    {0x999220, 512, 0xf8e2e4c252522598ull, "ScopeBegin 0x999220 (+ following code to 0x999420)"},
    {0x999420, 13, 0xe2d241971aa57034ull, "ScopeEnd 0x999420 (tail to 0x99a0d0)"},
    {0x99a080, 160, 0xc05e1abc124e5965ull, "scope clock writes 0x99a080 / 0x99a0d0"},
    {0x1d0a920, 116, 0x4dd40411757187e0ull, "binder ReleaseMem 0x1d0a920"},
};
uint64_t Fnv(const uint8_t* p, uint32_t n) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
	return h;
}

std::atomic<bool> g_checked {false}; // the bytes matched: the probe works

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// ---- side table (game threads write, GPU thread reads) ----
struct Slot {
	std::atomic<uint64_t> key {0}; // packet address; 0 while written
	std::atomic<uint64_t> address {0};
	std::atomic<uint64_t> name_begin {0}; // *(address + 0x48) when in the eboot, else 0
	std::atomic<uint64_t> name_end {0};   // *(address + 0x28) when in the eboot, else 0
};
constexpr uint32_t SlotBits = 16;
constexpr uint32_t SlotMask = (1u << SlotBits) - 1u;
Slot*              g_slots  = nullptr; // allocated in ReadRunMode when on (2 MiB)

uint32_t SlotOf(uint64_t key) { return static_cast<uint32_t>((key >> 2u) * 0x9e3779b97f4a7c15ull >> (64u - SlotBits)) & SlotMask; }

std::atomic<uint64_t> g_noted {0}, g_noted_named {0};

// ---- GPU thread state (one thread; plain data) ----
constexpr uint32_t Queues   = 8; // queue ids above 7 share the last
constexpr uint32_t MaxDepth = 16;
struct Open {
	uint32_t        id;
	uint64_t        record;
	const uint32_t* buffer; // (f5b cut) the command block of the instance's last packet the walk ran
	uint32_t        blocks; // (f5b cut) command blocks its packets lay in
};
struct QueueState {
	Open     stack[MaxDepth];
	uint32_t depth = 0;
};
struct Acc {
	int64_t  excl_ns = 0, incl_ns = 0;
	uint64_t begins = 0, draws = 0, dispatches = 0, waits = 0;
	uint8_t  queues = 0; // 1 << queue id seen
};
struct Pm4Snap {
	uint64_t draws = 0, dispatches = 0, waits = 0;
};

struct GpuState {
	std::thread::id thread;
	bool            thread_set = false;
	int32_t         queue      = -1; // translating now
	int64_t         last_ns    = 0;
	Pm4Snap         last_pm4;
	QueueState      q[Queues];
	Acc             pass[MaxPasses];
	Acc             outside[Queues];
	int64_t         window_start = 0;
	uint64_t        frames       = 0;
	// counters (window)
	uint64_t marks = 0, unknown = 0, mismatched = 0, foreign = 0, overflow = 0, ends_unmatched = 0, cost_ns = 0;
	// one frame's order, captured after each report
	bool                                       capture = true, capturing = false;
	std::vector<std::array<uint32_t, 3>>       order; // id, queue, begin(1)/end(0)
	uint32_t                                   names_printed = 0;
	int64_t                                    last_summary  = 0;
	// whole-run totals for the summary
	Acc      total[MaxPasses];
	uint64_t total_frames = 0;
	// (f5b cut) never reset: the per-pass tools read differences
	int64_t  cum_excl[MaxPasses] {};
	int64_t  cum_incl[MaxPasses] {};
	int64_t  cum_slice = 0;
	// (f5b cut) instances closed in the window / of them in more than one command block (only with the walk hook on)
	uint64_t instances = 0, split = 0, walk_packets = 0;
};
GpuState* g_gpu = nullptr;

Pm4Snap ReadPm4() {
	using LiveCounters::g_pm4;
	auto v = [](uint32_t op) { return g_pm4[op].load(std::memory_order_relaxed); };
	Pm4Snap s;
	// IT_DRAW_INDIRECT 0x24, IT_DRAW_INDEX_INDIRECT 0x25, IT_DRAW_INDEX_2 0x27, IT_DRAW_INDIRECT_MULTI 0x2c,
	// IT_DRAW_INDEX_AUTO 0x2d, IT_DRAW_INDEX_OFFSET_2 0x35, IT_DRAW_INDEX_INDIRECT_MULTI 0x38 (graphicsRun.cpp Pm4ClassOf)
	s.draws      = v(0x24) + v(0x25) + v(0x27) + v(0x2c) + v(0x2d) + v(0x35) + v(0x38);
	s.dispatches = v(0x15) + v(0x16); // IT_DISPATCH_DIRECT, IT_DISPATCH_INDIRECT
	s.waits      = v(0x3c);           // IT_WAIT_REG_MEM
	return s;
}

uint32_t QueueIndex(uint32_t queue_id) { return queue_id < Queues ? queue_id : Queues - 1u; }

// Charges the time and PM4 counts since the last checkpoint to the open passes of the queue being translated.
void Checkpoint(GpuState& g, int64_t now) {
	const Pm4Snap pm4 = ReadPm4();
	if (g.queue >= 0) {
		const int64_t  dt = now - g.last_ns;
		const uint64_t dd = pm4.draws - g.last_pm4.draws, dq = pm4.dispatches - g.last_pm4.dispatches,
		               dw = pm4.waits - g.last_pm4.waits;
		QueueState&    qs = g.q[g.queue];
		Acc&           a  = qs.depth > 0 ? g.pass[qs.stack[qs.depth - 1].id] : g.outside[g.queue];
		a.excl_ns += dt;
		if (qs.depth > 0) g.cum_excl[qs.stack[qs.depth - 1].id] += dt;
		g.cum_slice += dt;
		a.draws += dd;
		a.dispatches += dq;
		a.waits += dw;
		a.queues |= static_cast<uint8_t>(1u << g.queue);
		for (uint32_t i = 0; i < qs.depth; ++i) {
			g.pass[qs.stack[i].id].incl_ns += dt;
			g.cum_incl[qs.stack[i].id] += dt;
		}
	}
	g.last_ns  = now;
	g.last_pm4 = pm4;
}

const char* QueueName(uint8_t mask) {
	const bool d = (mask & 1u) != 0, a = (mask & ~1u) != 0;
	return d && a ? "DCB+ACB" : d ? "DCB" : a ? "ACB" : "-";
}

void PrintNames(GpuState& g) {
	const uint32_t n = PassRegistry::Count();
	for (uint32_t id = g.names_printed; id < n; ++id)
		std::printf("[f5] nombre id=%u \"%s\" eboot+0x%llx %s cat=%u\n", id, PassRegistry::Name(id),
		            static_cast<unsigned long long>(PassRegistry::NameOffset(id)),
		            PassRegistry::IsDynamic(id) ? "dinamico" : "estatico", PassRegistry::Category(id));
	g.names_printed = n;
}

void Report(GpuState& g, int64_t now) {
	const double frames = g.frames > 0 ? static_cast<double>(g.frames) : 1.0;
	const double secs   = static_cast<double>(now - g.window_start) / 1e9;
	int64_t      in_ns = 0, out_ns = 0, out_dcb = 0, out_acb = 0;
	const uint32_t n = PassRegistry::Count();
	for (uint32_t id = 0; id < n; ++id) in_ns += g.pass[id].excl_ns;
	for (uint32_t q = 0; q < Queues; ++q) {
		out_ns += g.outside[q].excl_ns;
		(q == 0 ? out_dcb : out_acb) += g.outside[q].excl_ns;
	}
	PrintNames(g);
	if (WalkOn())
		std::printf("[f5] corte: %.1f instancias/frame cerradas, partidas en varios bloques del CB %.1f/frame (%.1f%%; el corte por "
		            "ambitos las sigue enteras), paquetes vistos %.0f/frame\n",
		            static_cast<double>(g.instances) / frames, static_cast<double>(g.split) / frames,
		            g.instances > 0 ? 100.0 * static_cast<double>(g.split) / static_cast<double>(g.instances) : 0.0,
		            static_cast<double>(g.walk_packets) / frames);
	g.instances = g.split = g.walk_packets = 0;
	std::printf("[f5] %.1f s, %llu frames | hilo GPU: en pasadas %.3f ms/frame, fuera %.3f (DCB %.3f, ACB %.3f) | marcas %.1f/frame, "
	            "sin registro %llu, no casan %llu, ajenas %llu, pila llena %llu, fin sin inicio %llu | coste f5 %.3f ms/frame\n",
	            secs, static_cast<unsigned long long>(g.frames), static_cast<double>(in_ns) / 1e6 / frames,
	            static_cast<double>(out_ns) / 1e6 / frames, static_cast<double>(out_dcb) / 1e6 / frames,
	            static_cast<double>(out_acb) / 1e6 / frames, static_cast<double>(g.marks) / frames,
	            static_cast<unsigned long long>(g.unknown), static_cast<unsigned long long>(g.mismatched),
	            static_cast<unsigned long long>(g.foreign), static_cast<unsigned long long>(g.overflow),
	            static_cast<unsigned long long>(g.ends_unmatched), static_cast<double>(g.cost_ns) / 1e6 / frames);
	std::vector<uint32_t> ids;
	for (uint32_t id = 0; id < n; ++id)
		if (g.pass[id].begins > 0 || g.pass[id].excl_ns > 0) ids.push_back(id);
	std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return g.pass[a].excl_ns > g.pass[b].excl_ns; });
	for (const uint32_t id: ids) {
		const Acc& a = g.pass[id];
		std::printf("[f5] pasada id=%u \"%s\" cola=%s excl=%.3f incl=%.3f ms/frame llamadas=%.2f draws=%.1f dispatches=%.1f "
		            "esperas=%.2f /frame\n",
		            id, PassRegistry::Name(id), QueueName(a.queues), static_cast<double>(a.excl_ns) / 1e6 / frames,
		            static_cast<double>(a.incl_ns) / 1e6 / frames, static_cast<double>(a.begins) / frames,
		            static_cast<double>(a.draws) / frames, static_cast<double>(a.dispatches) / frames,
		            static_cast<double>(a.waits) / frames);
	}
	for (uint32_t q = 0; q < Queues; ++q) {
		const Acc& a = g.outside[q];
		if (a.excl_ns == 0) continue;
		std::printf("[f5] fuera cola=%u excl=%.3f ms/frame draws=%.1f dispatches=%.1f esperas=%.2f /frame\n", q,
		            static_cast<double>(a.excl_ns) / 1e6 / frames, static_cast<double>(a.draws) / frames,
		            static_cast<double>(a.dispatches) / frames, static_cast<double>(a.waits) / frames);
	}
	if (!g.order.empty()) {
		std::string line = "[f5] orden (1 frame; id:cola, + inicio, - fin):";
		char        item[32];
		for (const auto& o: g.order) {
			std::snprintf(item, sizeof(item), " %s%u:%u", o[2] != 0 ? "+" : "-", o[0], o[1]);
			line += item;
		}
		std::printf("%s\n", line.c_str());
	}
	std::fflush(stdout);
	for (uint32_t id = 0; id < n; ++id) {
		Acc& t = g.total[id];
		t.excl_ns += g.pass[id].excl_ns;
		t.incl_ns += g.pass[id].incl_ns;
		t.begins += g.pass[id].begins;
		t.draws += g.pass[id].draws;
		t.dispatches += g.pass[id].dispatches;
		t.waits += g.pass[id].waits;
		t.queues |= g.pass[id].queues;
		g.pass[id] = Acc {};
	}
	g.total_frames += g.frames;
	for (auto& o: g.outside) o = Acc {};
	g.frames = g.marks = g.unknown = g.mismatched = g.foreign = g.overflow = g.ends_unmatched = g.cost_ns = 0;
	g.order.clear();
	g.capture      = true;
	g.window_start = now;
}

} // namespace

RunMode ReadRunMode() noexcept {
	const char* v = std::getenv("KYTY_NATIVE_RENDER");
	// (pt1) BB_PATHTRACING=1 needs the game's scopes (its passes and capture are placed by them): the map, unless set.
	if ((v == nullptr || v[0] == '\0') && Libs::Graphics::PathTrace::Requested()) v = "map";
	if (v == nullptr || v[0] == '\0' || std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0) return RunMode::Off;
	RunMode mode = RunMode::Off;
	if (std::strcmp(v, "map") == 0) mode = RunMode::FrameMap;
	else if (std::strcmp(v, "hybrid") == 0) mode = RunMode::Hybrid;
	if (mode == RunMode::Off) {
		std::printf("[f5] KYTY_NATIVE_RENDER=%s: valor desconocido (map | hybrid); apagado\n", v);
		std::fflush(stdout);
		return mode;
	}
	g_slots = new Slot[size_t {1} << SlotBits];
	g_gpu   = new GpuState;
	std::printf("[f5] KYTY_NATIVE_RENDER=%s: mapa del frame (solo medida: tiempo del hilo GPU, draws, dispatches y esperas por "
	            "pasada del juego; nada cambia en la imagen)%s\n",
	            v, mode == RunMode::Hybrid ? "; hybrid: la unica implementacion de esta build es la pasada de CONTROL de la verificacion (Nombre:2 en KYTY_NATIVE_PASSES); por lo demas funciona como map" : "");
	std::fflush(stdout);
	return mode;
}

void InstallCheckSlow(uint64_t base, uint64_t size, bool is_eboot) {
	if (!is_eboot || g_checked.load(std::memory_order_acquire)) return;
	for (const auto& r: Checked) {
		if (size < r.offset + r.size) {
			std::printf("[f5] DESACTIVADO: el eboot es demasiado corto para ser 01.007.000\n");
			std::fflush(stdout);
			return;
		}
		const uint64_t h = Fnv(reinterpret_cast<const uint8_t*>(base + r.offset), r.size);
		if (h != r.fnv) {
			std::printf("[f5] DESACTIVADO: eboot+0x%llx (%s) hash 0x%016llx, 01.007.000 tiene 0x%016llx; no se mide nada\n",
			            static_cast<unsigned long long>(r.offset), r.what, static_cast<unsigned long long>(h),
			            static_cast<unsigned long long>(r.fnv));
			std::fflush(stdout);
			return;
		}
	}
	PassRegistry::Init(base, size);
	size_t            rows   = 0;
	const StaticPass* table  = PassRegistry::StaticRows(&rows);
	size_t            wrong  = 0;
	for (size_t i = 0; i < rows; ++i) {
		const auto*  p = reinterpret_cast<const char*>(base + table[i].name_offset);
		const size_t n = std::strlen(table[i].name);
		if (table[i].name_offset + n + 1 > size || std::memcmp(p, table[i].name, n) != 0 || p[n] != '\0') ++wrong;
	}
	if (wrong != 0) {
		std::printf("[f5] DESACTIVADO: %zu de %zu nombres estaticos no coinciden con el eboot\n", wrong, rows);
		std::fflush(stdout);
		return;
	}
	g_checked.store(true, std::memory_order_release);
	std::printf("[f5] eboot 01.007.000: ambitos del perfilador comprobados (%zu tramos), %zu nombres estaticos; mapa activo\n",
	            sizeof(Checked) / sizeof(Checked[0]), rows);
	std::fflush(stdout);
}

void NoteReleaseMemSlow(const uint32_t* packet, uint64_t address, uint32_t /*action*/, uint32_t /*data_sel*/) {
	if (!g_checked.load(std::memory_order_relaxed) || packet == nullptr || address < 0x30) return;
	g_noted.fetch_add(1, std::memory_order_relaxed);
	// The record was just written by this thread (ScopeBegin stores +0x40/+0x48 before its clock write).
	// An end's address is record + 0x20: its name is at +0x28 and +0x48 would be past the record (maybe past the pool),
	// so +0x48 is read only when +0x28 is not a name. Both are inside the record for a begin.
	uint64_t nb = 0, ne = 0;
	std::memcpy(&ne, reinterpret_cast<const void*>(address + 0x28), sizeof(ne));
	if (!PassRegistry::InEboot(ne)) {
		ne = 0;
		std::memcpy(&nb, reinterpret_cast<const void*>(address + 0x48), sizeof(nb));
		if (!PassRegistry::InEboot(nb)) nb = 0;
	}
	if (nb == 0 && ne == 0) return;
	g_noted_named.fetch_add(1, std::memory_order_relaxed);
	const auto key  = reinterpret_cast<uint64_t>(packet);
	Slot&      slot = g_slots[SlotOf(key)];
	slot.key.store(0, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	slot.address.store(address, std::memory_order_relaxed);
	slot.name_begin.store(nb, std::memory_order_relaxed);
	slot.name_end.store(ne, std::memory_order_relaxed);
	slot.key.store(key, std::memory_order_release);
}

void TimestampPacketSlow(const uint32_t* packet, uint64_t address) {
	GpuState& g = *g_gpu;
	if (!g_checked.load(std::memory_order_relaxed)) return;
	if (!g.thread_set || g.thread != std::this_thread::get_id()) {
		++g.foreign;
		return;
	}
	const int64_t t0 = NowNs();
	Checkpoint(g, t0);
	const auto key  = reinterpret_cast<uint64_t>(packet);
	Slot&      slot = g_slots[SlotOf(key)];
	const uint64_t k1 = slot.key.load(std::memory_order_acquire);
	const uint64_t a  = slot.address.load(std::memory_order_relaxed);
	const uint64_t nb = slot.name_begin.load(std::memory_order_relaxed);
	const uint64_t ne = slot.name_end.load(std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_acquire);
	const uint64_t k2 = slot.key.load(std::memory_order_relaxed);
	if (k1 != key || k2 != key) {
		++g.unknown; // not a scope's, or its slot was taken by another packet
	} else if (a != address) {
		++g.mismatched;
	} else if (g.queue >= 0) {
		++g.marks;
		QueueState& qs = g.q[g.queue];
		// An end: the open record + 0x20 on this queue (innermost first; deeper ones closed with it).
		int32_t at = -1;
		for (int32_t i = static_cast<int32_t>(qs.depth) - 1; i >= 0; --i)
			if (qs.stack[i].record + 0x20 == address) {
				at = i;
				break;
			}
		if (at >= 0) {
			if (g.capturing) g.order.push_back({qs.stack[at].id, static_cast<uint32_t>(g.queue), 0u});
			if (static_cast<uint32_t>(at) != qs.depth - 1) ++g.mismatched;
			const uint32_t closed = qs.stack[at].id;
			for (uint32_t i = static_cast<uint32_t>(at); i < qs.depth; ++i) {
				++g.instances;
				if (qs.stack[i].blocks > 1) ++g.split;
			}
			// (pt1) inner scopes closed with it end first
			for (uint32_t i = qs.depth; i > static_cast<uint32_t>(at) + 1u; --i) Libs::Graphics::PathTrace::ScopeEnd(qs.stack[i - 1].id, g.queue);
			qs.depth = static_cast<uint32_t>(at);
			if (Verify::Configured()) Verify::End(closed, g.queue);
			Libs::Graphics::PathTrace::ScopeEnd(closed, g.queue); // (pt1; one static load when off)
		} else if (nb != 0) {
			const uint32_t id = PassRegistry::IdOfName(nb);
			if (id == NoPass) {
				++g.unknown;
			} else if (qs.depth >= MaxDepth) {
				++g.overflow;
			} else {
				qs.stack[qs.depth++] = Open {id, address, nullptr, 0};
				++g.pass[id].begins;
				if (g.capturing) g.order.push_back({id, static_cast<uint32_t>(g.queue), 1u});
				if (Verify::Configured()) Verify::Begin(id, g.queue);
				Libs::Graphics::PathTrace::ScopeBegin(id, g.queue); // (pt1)
			}
		} else if (ne != 0) {
			++g.ends_unmatched; // an end whose begin this queue did not see (begun before the probe, or on another queue)
		} else {
			++g.unknown;
		}
	}
	const int64_t t1 = NowNs();
	g.cost_ns += static_cast<uint64_t>(t1 - t0);
	g.last_ns = t1; // the probe's own time is not charged to the pass
}

void SliceBeginSlow(uint32_t queue_id) {
	if (!g_checked.load(std::memory_order_relaxed)) return;
	GpuState& g = *g_gpu;
	if (!g.thread_set) {
		g.thread       = std::this_thread::get_id();
		g.thread_set   = true;
		g.window_start = NowNs();
	}
	if (g.thread != std::this_thread::get_id()) return;
	g.queue    = static_cast<int32_t>(QueueIndex(queue_id));
	g.last_ns  = NowNs();
	g.last_pm4 = ReadPm4();
}

void SliceEndSlow() {
	if (!g_checked.load(std::memory_order_relaxed)) return;
	GpuState& g = *g_gpu;
	if (!g.thread_set || g.thread != std::this_thread::get_id() || g.queue < 0) return;
	Checkpoint(g, NowNs());
	g.queue = -1;
}

void FlipSlow(int64_t idle_ns, uint64_t sub_cycles) {
	if (!g_checked.load(std::memory_order_relaxed)) return;
	GpuState& g = *g_gpu;
	if (!g.thread_set || g.thread != std::this_thread::get_id()) return;
	const int64_t now = NowNs();
	Checkpoint(g, now);
	++g.frames;
	if (Techo::Configured()) Techo::Flip(now, idle_ns, sub_cycles); // (f5b)
	if (Verify::Configured()) Verify::Flip();                        // (f5b)
	// One frame's order after each report: from this flip to the next.
	if (g.capturing) g.capturing = false;
	if (g.capture) {
		g.capture   = false;
		g.capturing = true;
	}
	if (now - g.window_start >= 5'000'000'000LL) {
		Report(g, now);
		if (now - g.last_summary >= 60'000'000'000LL) {
			FrameMapProbe::PrintSummary();
			g.last_summary = now;
		}
	}
}

// ---- (f5b) the cut: the walk's packets and what the per-pass tools read ----------------------------------------------
bool ReadWalkOn() noexcept {
	const bool techo = Techo::Configured(), verify = Verify::Configured();
	if (techo && verify) {
		std::printf("[f5] KYTY_NATIVE_TECHO y KYTY_NATIVE_PASSES a la vez: la medida del techo no vale con la verificacion "
		            "encendida (sus copias cuestan en las dos ventanas por igual, pero no son el juego). Usa uno por sesion\n");
		std::fflush(stdout);
	}
	return techo || verify;
}

uint32_t WalkPacketSlow(uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw, const uint32_t* buffer,
                        uint32_t /*buffer_dw*/) {
	if (!g_checked.load(std::memory_order_relaxed)) return 0;
	GpuState& g = *g_gpu;
	if (!g.thread_set || g.thread != std::this_thread::get_id() || g.queue < 0) return 0;
	++g.walk_packets;
	QueueState& qs = g.q[g.queue];
	for (uint32_t i = 0; i < qs.depth; ++i) {
		Open& o = qs.stack[i];
		if (o.buffer != buffer) {
			o.buffer = buffer;
			++o.blocks;
		}
	}
	const uint32_t header = packet[0];
	const uint32_t len    = header == 0x80000000u ? 1u : ((header >> 16u) & 0x3fffu) + 2u;
	if (Verify::Configured() && (header >> 30u) == 3u && len <= remaining_dw) Verify::Packet(opcode, packet, len, buffer);
	if (qs.depth == 0 || !Techo::Configured()) return 0;
	// Only a whole type-3 packet that fits is passed over.
	if ((header >> 30u) != 3u || len > remaining_dw) return 0;
	return Techo::Skip(opcode, qs.stack[qs.depth - 1].id) ? len : 0u;
}

namespace Cut {
bool OnGpuThread() noexcept {
	return g_gpu != nullptr && g_gpu->thread_set && g_gpu->thread == std::this_thread::get_id();
}
int32_t Queue() noexcept { return g_gpu != nullptr ? g_gpu->queue : -1; }
uint32_t Innermost() noexcept {
	if (g_gpu == nullptr || g_gpu->queue < 0) return NoPass;
	const QueueState& qs = g_gpu->q[g_gpu->queue];
	return qs.depth > 0 ? qs.stack[qs.depth - 1].id : NoPass;
}
bool IsOpen(uint32_t id) noexcept {
	if (g_gpu == nullptr || g_gpu->queue < 0) return false;
	const QueueState& qs = g_gpu->q[g_gpu->queue];
	for (uint32_t i = 0; i < qs.depth; ++i)
		if (qs.stack[i].id == id) return true;
	return false;
}
int64_t ExclNs(uint32_t id) noexcept { return g_gpu != nullptr && id < MaxPasses ? g_gpu->cum_excl[id] : 0; }
int64_t InclNs(uint32_t id) noexcept { return g_gpu != nullptr && id < MaxPasses ? g_gpu->cum_incl[id] : 0; }
int64_t SliceNs() noexcept { return g_gpu != nullptr ? g_gpu->cum_slice : 0; }
} // namespace Cut

namespace FrameMapProbe {
void PrintSummary() {
	if (g_gpu == nullptr || !g_checked.load(std::memory_order_relaxed)) return;
	GpuState&      g      = *g_gpu;
	const double   frames = g.total_frames > 0 ? static_cast<double>(g.total_frames) : 1.0;
	const uint32_t n      = PassRegistry::Count();
	std::vector<uint32_t> ids;
	for (uint32_t id = 0; id < n; ++id)
		if (g.total[id].excl_ns > 0) ids.push_back(id);
	std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return g.total[a].excl_ns > g.total[b].excl_ns; });
	std::printf("[f5] RESUMEN %llu frames, %zu pasadas con tiempo, notadas %llu (con nombre %llu); orden de conversion por coste "
	            "exclusivo del hilo GPU:\n",
	            static_cast<unsigned long long>(g.total_frames), ids.size(),
	            static_cast<unsigned long long>(g_noted.load(std::memory_order_relaxed)),
	            static_cast<unsigned long long>(g_noted_named.load(std::memory_order_relaxed)));
	uint32_t rank = 0;
	for (const uint32_t id: ids) {
		const Acc& a = g.total[id];
		std::printf("[f5] RESUMEN #%u id=%u \"%s\" cola=%s excl=%.3f incl=%.3f ms/frame draws=%.1f dispatches=%.1f\n", ++rank, id,
		            PassRegistry::Name(id), QueueName(a.queues), static_cast<double>(a.excl_ns) / 1e6 / frames,
		            static_cast<double>(a.incl_ns) / 1e6 / frames, static_cast<double>(a.draws) / frames,
		            static_cast<double>(a.dispatches) / frames);
	}
	std::fflush(stdout);
}
} // namespace FrameMapProbe

} // namespace NativeRender
