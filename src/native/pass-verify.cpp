// (f5b) KYTY_NATIVE_PASSES=<pass>:2,...: the per-pass verification framework (=2), CPU side.
//
// For a pass in Verify both the emulated pass and its native implementation run; the image shown is the emulated
// one. At the pass's scope end its exit state is snapshot and compared with what the native implementation leaves:
//   1. State: the register values the pass's range sets (SET_CONTEXT/SH/UCONFIG/CONFIG_REG, last value per register:
//      the "exit delta" the next pass relies on) and a hash of every packet of the range. Taken twice: as the GPU
//      thread's walk ran the packets (walk time) and decoded again from the range's command memory at the scope end
//      (the delta a collapsed native pass would apply: skip the range, apply its delta). Different = DISCREPANCIA
//      "estado": the range changed after the walk read it (the game patching packets, f5 H2) or the cut lost packets.
//      The range is the walk's (pass-cut.h): every block of the command buffer the instance's packets lay in, as
//      segments; packets the walk did not see one by one (KYTY_PORT2 spans applied from a flow, runs of draws taken
//      at once) are gaps between segments, counted and decoded too ("registros en huecos").
//   2. Image: the render targets the pass wrote (renderDraw.cpp / table-xpr.inc NoteTarget) are copied at the scope
//      end (sampled tiles, verify-gpu.cpp) as the emulated pass left them, and as the native implementation's shadow
//      target holds them; read back 8 frames later and compared texel by texel. Different = DISCREPANCIA "imagen".
//      The same targets are copied at the scope begin too (the ones the pass's previous instance wrote): what the pass
//      changed ("cambiados") shows the comparator does see a pass's writes.
// The control pass (the only implementation in this build) is native but trivially equivalent: its shadow target is
// filled by copying the emulated target, and its exit delta is the range's own. It must give 0 DISCREPANCIA; that
// validates the framework itself (cut, segments, snapshot, shadow targets, copies, readback, comparison) before a real
// native pass (B3: the XPR depth pass) is put in its place.
// Sampling: one instance per pass every KYTY_NATIVE_VERIFY_FRAMES frames (default 1; the passes' frames staggered). Lines "[f5v]" every 5 s,
// "[f5v] RESUMEN" every 60 s and at exit; "[f5v] DISCREPANCIA" (the first 8 per pass).
#include "native/native-render.h"
#include "native/pass-cut.h"
#include "native/pass-registry.h"
#include "native/verify-jobs.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace NativeRender::Verify {
namespace {

constexpr uint32_t MaxActive = 4;

struct Segment {
	const uint32_t* buffer = nullptr;
	const uint32_t* begin  = nullptr;
	const uint32_t* end    = nullptr;
};

struct Instance {
	bool                                       live = false;
	uint32_t                                   pass = 0;
	int32_t                                    queue = -1;
	uint64_t                                   serial = 0, begin_flip = 0;
	std::vector<Segment>                       segments;
	std::vector<std::pair<uint32_t, uint32_t>> writes; // (space << 24 | register, value), walk order
	uint64_t                                   hash = 0;
	uint64_t                                   packets = 0, indirect = 0;
	uint32_t                                   ntargets = 0;
	TargetRef                                  targets[MaxTargets];
};

struct Acc {
	uint64_t instances = 0, segments = 0, gaps = 0, gap_dwords = 0, gap_regs = 0, gap_unparsed = 0, packets = 0;
	uint64_t regs = 0, indirect = 0;
	uint64_t state_bad = 0, regs_differing = 0, hash_bad = 0;
	uint64_t targets = 0, images = 0, texels = 0, differing = 0, image_bad = 0, entries = 0, changed = 0, changed_images = 0;
	uint64_t refused[Refusals] {};
	uint64_t no_targets = 0, too_many_targets = 0, skipped_busy = 0, no_end = 0;
};

struct State {
	bool                  configured = false;
	uint32_t              every      = 1;
	Instance              active[MaxActive];
	uint64_t              serial     = 0;
	uint64_t              flips      = 0;
	uint64_t              last_frame[MaxPasses] {};
	bool                  seen[MaxPasses] {};
	TargetRef             last_targets[MaxPasses][MaxTargets] {};
	uint32_t              last_count[MaxPasses] {};
	Acc                   window[MaxPasses], total[MaxPasses];
	uint32_t              reports[MaxPasses] {};
	std::deque<Job>       jobs;
	int64_t               window_start = 0, last_summary = 0;
	bool                  exit_hooked = false;
	uint32_t              open_count  = 0; // live instances
};
State g;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

uint64_t Mix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
	return h * 0xff51afd7ed558ccdull;
}


// The register space of a SET_*_REG opcode (0: not one); indirect ones (values from memory) are 0xff.
uint32_t SpaceOf(uint32_t op) {
	switch (op) {
		case 0x68: return 1;  // SET_CONFIG_REG
		case 0x69: return 2;  // SET_CONTEXT_REG
		case 0x76: return 3;  // SET_SH_REG
		case 0x79: return 4;  // SET_UCONFIG_REG
		case 0x7a: return 5;  // SET_UCONFIG_REG_INDEX
		case 0x63:            // SET_SH_REG_INDIRECT
		case 0x64:            // SET_UCONFIG_REG_INDIRECT
		case 0x9f: return 0xff; // SET_CONTEXT_REG_INDIRECT
		default: return 0;
	}
}

// One packet into a delta / hash. `len` its dwords (header included).
void Decode(const uint32_t* p, uint32_t len, std::vector<std::pair<uint32_t, uint32_t>>* writes, uint64_t* hash,
            uint64_t* indirect) {
	const uint32_t op = (p[0] >> 8u) & 0xffu;
	if (hash != nullptr)
		for (uint32_t i = 0; i < len; ++i) *hash = Mix(*hash, p[i]);
	const uint32_t space = SpaceOf(op);
	if (space == 0xff) {
		if (indirect != nullptr) ++*indirect;
		return;
	}
	if (space == 0 || len < 3 || writes == nullptr) return;
	const uint32_t reg = p[1] & 0xffffu;
	for (uint32_t i = 2; i < len; ++i) writes->emplace_back(space << 24u | ((reg + i - 2) & 0xffffffu), p[i]);
}

// Type-3 packets of [p, end) (type-2 one-dword NOPs too); false if one does not parse or runs past `end`.
template <typename F>
bool Walk(const uint32_t* p, const uint32_t* end, F&& f) {
	while (p < end) {
		const uint32_t header = p[0];
		uint32_t       len    = 1;
		if (header != 0x80000000u) {
			if ((header >> 30u) != 3u) return false;
			len = ((header >> 16u) & 0x3fffu) + 2u;
			if (len > static_cast<uint64_t>(end - p)) return false;
			f(p, len);
		}
		p += len;
	}
	return true;
}

std::unordered_map<uint32_t, uint32_t> Final(const std::vector<std::pair<uint32_t, uint32_t>>& writes) {
	std::unordered_map<uint32_t, uint32_t> m;
	m.reserve(writes.size());
	for (const auto& w: writes) m[w.first] = w.second;
	return m;
}

const char* Name(uint32_t id) { return PassRegistry::Name(id); }

void Discrepancy(uint32_t pass, const char* what) {
	if (g.reports[pass] >= 8) return;
	++g.reports[pass];
	std::printf("[f5v] DISCREPANCIA \"%s\" %s\n", Name(pass), what);
	std::fflush(stdout);
}

void Finish(Instance& in) {
	Acc& a = g.window[in.pass];
	++a.instances;
	a.segments += in.segments.size();
	a.packets += in.packets;
	a.indirect += in.indirect;
	// The exit delta decoded again from the range's command memory, now.
	std::vector<std::pair<uint32_t, uint32_t>> again;
	again.reserve(in.writes.size());
	uint64_t hash = 0x84222325cbf29ce4ull;
	bool     parsed = true;
	for (size_t i = 0; i < in.segments.size(); ++i) {
		const Segment& s = in.segments[i];
		parsed &= Walk(s.begin, s.end, [&](const uint32_t* p, uint32_t len) { Decode(p, len, &again, &hash, nullptr); });
		// The gap to the next segment of the same block: what the walk did not run packet by packet.
		if (i + 1 < in.segments.size() && in.segments[i + 1].buffer == s.buffer && in.segments[i + 1].begin > s.end &&
		    in.segments[i + 1].begin - s.end < (1 << 20)) {
			++a.gaps;
			a.gap_dwords += static_cast<uint64_t>(in.segments[i + 1].begin - s.end);
			std::vector<std::pair<uint32_t, uint32_t>> gap;
			if (!Walk(s.end, in.segments[i + 1].begin, [&](const uint32_t* p, uint32_t len) { Decode(p, len, &gap, nullptr, nullptr); }))
				++a.gap_unparsed;
			a.gap_regs += gap.size();
		}
	}
	const auto walk = Final(in.writes);
	const auto end  = Final(again);
	a.regs += walk.size();
	uint64_t differing = 0;
	for (const auto& [key, value]: walk) {
		const auto it = end.find(key);
		if (it == end.end() || it->second != value) ++differing;
	}
	for (const auto& [key, value]: end)
		if (walk.find(key) == walk.end()) ++differing;
	if (!parsed) ++differing;
	if (differing != 0) {
		++a.state_bad;
		a.regs_differing += differing;
		char text[160];
		std::snprintf(text, sizeof(text), "estado: instancia %llu, %llu registros de salida distintos (de %zu)%s",
		              static_cast<unsigned long long>(in.serial), static_cast<unsigned long long>(differing), walk.size(),
		              parsed ? "" : ", un segmento ya no se analiza");
		Discrepancy(in.pass, text);
	}
	if (hash != in.hash) ++a.hash_bad;
	// The image part: the targets this instance wrote, at the scope end.
	if (in.ntargets == 0) {
		++a.no_targets;
	} else {
		Job job;
		job.kind     = JobKind::Exit;
		job.pass     = in.pass;
		job.instance = in.serial;
		job.count    = in.ntargets;
		job.queue    = in.queue;
		std::copy(in.targets, in.targets + in.ntargets, job.targets);
		g.jobs.push_back(job);
		std::copy(in.targets, in.targets + in.ntargets, g.last_targets[in.pass]);
		g.last_count[in.pass] = in.ntargets;
		a.targets += in.ntargets;
	}
	in = Instance {};
}

void PrintAcc(const char* title, const Acc* acc, double frames) {
	const uint32_t n = PassRegistry::Count();
	for (uint32_t id = 0; id < n && id < MaxPasses; ++id) {
		const Acc& a = acc[id];
		if (PassRegistry::Requested(id) != PassState::Verify || (a.instances == 0 && !g.seen[id])) continue;
		std::string refused;
		for (uint32_t r = 0; r < Refusals; ++r)
			if (a.refused[r] != 0) refused += std::string(" ") + RefusalNames[r] + "=" + std::to_string(a.refused[r]);
		const bool image_ok = a.images > 0 && a.differing == 0;
		const bool state_ok = a.instances > 0 && a.state_bad == 0;
		std::printf("[f5v] %s \"%s\" control: instancias %llu (%.2f/frame) | estado: %s, registros de salida %.1f/instancia, distintos "
		            "%llu en %llu instancias, hash de paquetes distinto %llu, bloques %.2f/instancia, huecos %.1f/instancia "
		            "(%.0f dwords, %.1f registros, sin analizar %llu), registros por memoria %llu | imagen: %s, objetivos %llu, "
		            "comparados %llu, texeles %llu, DISCREPANCIA texeles %llu en %llu objetivos, entrada comparada %llu (cambiados "
		            "por la pasada en %llu, texeles %llu), sin objetivos %llu, sin fin %llu, rechazados:%s | %s\n",
		            title, Name(id), static_cast<unsigned long long>(a.instances),
		            frames > 0 ? static_cast<double>(a.instances) / frames : 0.0, state_ok ? "0 DISCREPANCIA" : a.instances ? "DISCREPANCIA" : "-",
		            a.instances ? static_cast<double>(a.regs) / static_cast<double>(a.instances) : 0.0,
		            static_cast<unsigned long long>(a.regs_differing), static_cast<unsigned long long>(a.state_bad),
		            static_cast<unsigned long long>(a.hash_bad),
		            a.instances ? static_cast<double>(a.segments) / static_cast<double>(a.instances) : 0.0,
		            a.instances ? static_cast<double>(a.gaps) / static_cast<double>(a.instances) : 0.0,
		            a.instances ? static_cast<double>(a.gap_dwords) / static_cast<double>(a.instances) : 0.0,
		            a.instances ? static_cast<double>(a.gap_regs) / static_cast<double>(a.instances) : 0.0,
		            static_cast<unsigned long long>(a.gap_unparsed), static_cast<unsigned long long>(a.indirect),
		            image_ok ? "0 DISCREPANCIA" : a.images ? "DISCREPANCIA" : "sin comparar todavia",
		            static_cast<unsigned long long>(a.targets), static_cast<unsigned long long>(a.images),
		            static_cast<unsigned long long>(a.texels), static_cast<unsigned long long>(a.differing),
		            static_cast<unsigned long long>(a.image_bad), static_cast<unsigned long long>(a.entries),
		            static_cast<unsigned long long>(a.changed_images), static_cast<unsigned long long>(a.changed),
		            static_cast<unsigned long long>(a.no_targets), static_cast<unsigned long long>(a.no_end), refused.empty() ? " ninguno" : refused.c_str(),
		            state_ok && image_ok ? "CONTROL OK" : (a.state_bad || a.differing) ? "CONTROL FALLA: el marco no vale aun"
		                                                                               : "faltan datos");
	}
	std::fflush(stdout);
}

void AddInto(Acc& t, const Acc& w) {
	const uint64_t* src = reinterpret_cast<const uint64_t*>(&w);
	uint64_t*       dst = reinterpret_cast<uint64_t*>(&t);
	static_assert(sizeof(Acc) % sizeof(uint64_t) == 0);
	for (size_t i = 0; i < sizeof(Acc) / sizeof(uint64_t); ++i) dst[i] += src[i];
}

void AtExit() {
	if (!g.configured) return;
	for (uint32_t id = 0; id < MaxPasses; ++id) {
		AddInto(g.total[id], g.window[id]);
		g.window[id] = Acc {};
	}
	PrintAcc("RESUMEN FINAL", g.total, static_cast<double>(g.flips));
}

bool ReadConfig() {
	const char* v = std::getenv("KYTY_NATIVE_PASSES");
	if (v == nullptr || std::strstr(v, ":2") == nullptr) return false;
	if (const char* e = std::getenv("KYTY_NATIVE_VERIFY_FRAMES"); e != nullptr && std::atoi(e) >= 1)
		g.every = static_cast<uint32_t>(std::atoi(e));
	std::printf("[f5v] verificacion por pasada (=2) con la pasada de CONTROL: estado de salida (registros que deja el rango, "
	            "leidos al recorrerlo y otra vez de su memoria al final) e imagen (los objetivos que escribio, copiados al "
	            "final tal como los dejo la pasada emulada y por la sombra del control; teselas muestreadas, leidas 8 frames "
	            "despues). Tiene que dar 0 DISCREPANCIA. Una instancia por pasada cada %u frames. La imagen no cambia\n",
	            g.every);
	std::fflush(stdout);
	return true;
}

} // namespace

bool Configured() noexcept {
	static const bool on = [] {
		g.configured = ReadConfig();
		return g.configured;
	}();
	return on;
}

void Begin(uint32_t id, int32_t queue) {
	if (id >= MaxPasses || PassRegistry::Effective(id) != PassState::Verify) return;
	g.seen[id] = true;
	// One instance per pass per sampled frame; the passes' frames staggered (pass id) so the copies spread out.
	if (g.last_frame[id] == g.flips + 1 || (g.flips + id) % g.every != 0) return;
	Instance* slot = nullptr;
	for (auto& a: g.active)
		if (!a.live) {
			slot = &a;
			break;
		}
	if (slot == nullptr) {
		++g.window[id].skipped_busy;
		return;
	}
	g.last_frame[id] = g.flips + 1;
	*slot        = Instance {};
	slot->live   = true;
	slot->pass   = id;
	slot->queue  = queue;
	slot->serial = ++g.serial;
	slot->begin_flip = g.flips;
	slot->hash   = 0x84222325cbf29ce4ull;
	++g.open_count;
	g_note_targets.store(true, std::memory_order_relaxed);
	// The entry copy: the targets the previous instance wrote, as they are before this one draws.
	if (g.last_count[id] != 0) {
		Job job;
		job.kind     = JobKind::Entry;
		job.pass     = id;
		job.instance = slot->serial;
		job.count    = g.last_count[id];
		job.queue    = queue;
		std::copy(g.last_targets[id], g.last_targets[id] + job.count, job.targets);
		g.jobs.push_back(job);
	}
}

void End(uint32_t id, int32_t queue) {
	for (auto& a: g.active)
		if (a.live && a.pass == id && a.queue == queue) {
			Finish(a);
			--g.open_count;
			break;
		}
	if (g.open_count == 0) g_note_targets.store(false, std::memory_order_relaxed);
}

void Packet(uint32_t /*opcode*/, const uint32_t* packet, uint32_t len_dw, const uint32_t* buffer) {
	if (g.open_count == 0) return;
	const int32_t queue = Cut::Queue();
	for (auto& in: g.active) {
		if (!in.live || in.queue != queue) continue;
		if (in.segments.empty() || in.segments.back().buffer != buffer || in.segments.back().end != packet)
			in.segments.push_back(Segment {buffer, packet, packet});
		in.segments.back().end = packet + len_dw;
		++in.packets;
		Decode(packet, len_dw, &in.writes, &in.hash, &in.indirect);
	}
}

void Flip() {
	++g.flips;
	CollectGpu(g.flips);
	// An instance whose end the cut never saw (2 frames): dropped, counted.
	for (auto& a: g.active)
		if (a.live && g.flips > a.begin_flip + 2) {
			++g.window[a.pass].no_end;
			a = Instance {};
			--g.open_count;
		}
	if (g.open_count == 0) g_note_targets.store(false, std::memory_order_relaxed);
	// Jobs no walk of their queue took (a queue that stopped): dropped, counted.
	while (g.jobs.size() > 256) {
		ImageRefused(g.jobs.front().pass, RNoBuffer);
		g.jobs.pop_front();
	}
	const int64_t now = NowNs();
	if (g.window_start == 0) {
		g.window_start = g.last_summary = now;
		if (!g.exit_hooked) {
			g.exit_hooked = true;
			std::atexit(AtExit);
		}
		return;
	}
	static uint64_t window_flips = 0;
	++window_flips;
	if (now - g.window_start < 5'000'000'000LL) return;
	PrintAcc("5 s", g.window, static_cast<double>(window_flips));
	for (uint32_t id = 0; id < MaxPasses; ++id) {
		AddInto(g.total[id], g.window[id]);
		g.window[id] = Acc {};
	}
	window_flips   = 0;
	g.window_start = now;
	if (now - g.last_summary >= 60'000'000'000LL) {
		g.last_summary = now;
		PrintAcc("RESUMEN", g.total, static_cast<double>(g.flips));
	}
}

bool TakeJob(Job& out) {
	if (g.jobs.empty() || g.jobs.front().queue != Cut::Queue()) return false;
	out = g.jobs.front();
	g.jobs.pop_front();
	return true;
}
uint64_t FlipCount() noexcept { return g.flips; }

void ImageResult(const ImageOutcome& o) {
	if (o.pass >= MaxPasses) return;
	Acc& a = g.window[o.pass];
	++a.images;
	a.texels += o.texels;
	a.differing += o.differing;
	if (o.differing != 0) {
		++a.image_bad;
		char text[200];
		std::snprintf(text, sizeof(text),
		              "imagen: instancia %llu, objetivo %s formato %u %ux%u nivel %u capa %u: %llu de %llu texeles distintos",
		              static_cast<unsigned long long>(o.instance), o.depth ? "profundidad" : "color", o.format, o.width,
		              o.height, o.level, o.layer, static_cast<unsigned long long>(o.differing),
		              static_cast<unsigned long long>(o.texels));
		Discrepancy(o.pass, text);
	}
	if (o.entry_known) {
		++a.entries;
		a.changed += o.changed;
		if (o.changed != 0) ++a.changed_images;
	}
}

void ImageRefused(uint32_t pass, Refusal why) {
	if (pass < MaxPasses && why < Refusals) ++g.window[pass].refused[why];
}

} // namespace NativeRender::Verify

namespace NativeRender {

bool GpuWorkPending() noexcept {
	return Verify::Configured() && !Verify::g.jobs.empty() && Verify::g.jobs.front().queue == Cut::Queue();
}

void NoteTargetSlow(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth) {
	using namespace Verify;
	if (!Cut::OnGpuThread() || g.open_count == 0) return;
	const int32_t   queue = Cut::Queue();
	const TargetRef t {index, generation, level, layer, depth};
	for (auto& in: g.active) {
		if (!in.live || in.queue != queue) continue;
		bool known = false;
		for (uint32_t i = 0; i < in.ntargets; ++i) known |= in.targets[i] == t;
		if (known) continue;
		if (in.ntargets < MaxTargets) in.targets[in.ntargets++] = t;
		else ++g.window[in.pass].too_many_targets;
	}
}

} // namespace NativeRender
