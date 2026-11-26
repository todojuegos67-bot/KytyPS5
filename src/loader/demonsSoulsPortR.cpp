#include "loader/demonsSoulsPortR.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/runtimeLinker.h"
#include "live-census.h"  // (the GPU thread's submissions, for the [port-r] line)
#include "live-control.h" // (LiveControl::g_flips: the line's frames)
#include "port-r.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <x86intrin.h>

// (port-r, KYTY_PORT_R; local/port-r.h) The game's common geometry list, eboot+0xab57d0 of 01.007.000, run by the
// host: PortR::List below does what the game's function does, element by element, calling the game's own material,
// draw and binder functions (they write the game's state and its command buffer as always). Its entry jumps to List;
// the trampoline (the 12 bytes the jump replaced + a jump back) runs the game's function whole where List leaves it
// to the game (telemetry pass, the OFF window of KYTY_PORT_R=1).

namespace Loader::DemonsSoulsPortR {
namespace {

namespace L = PortR::GameLayout;

using ListFunction     = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t*, int32_t, int32_t);
using MaterialFunction = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t*);
using SubmeshFunction  = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t*, uint32_t, uint32_t, uint64_t);
using AllocFunction    = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t*, int32_t, uint32_t);
using BindFunction     = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t);
using DrawFunction     = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t*, uint32_t);
using TestFunction     = KYTY_SYSV_ABI bool (*)(uint64_t);
using GetFunction      = KYTY_SYSV_ABI uint64_t (*)(uint64_t);
using TouchFunction    = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t);

struct Game {
	ListFunction     original = nullptr; // the trampoline
	MaterialFunction material = nullptr;
	SubmeshFunction  submesh  = nullptr;
	AllocFunction    alloc    = nullptr;
	BindFunction     bind     = nullptr;
	DrawFunction     draw     = nullptr;
	TestFunction     is_default = nullptr;
	uint64_t         alloc_desc = 0; // eboot+0x2f643f0
};
Game     g_game;
uint64_t g_trampoline = 0;

// The 12 bytes the jump replaces: push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12 (no jump of the
// function lands in them, none is position-dependent, rax is first written at +0x14). Next: push rbx.
constexpr size_t                   Moved = 12;
constexpr std::array<uint8_t, Moved> Prologue {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54};

template <typename T>
[[nodiscard]] inline T At(uint64_t address) {
	T v;
	std::memcpy(&v, reinterpret_cast<const void*>(address), sizeof(T));
	return v;
}

// The [port-r] line, every 5 s, from whichever game thread calls the list then.
void MaybeReport() {
	static std::atomic<uint32_t> tick {0};
	if ((tick.fetch_add(1, std::memory_order_relaxed) & 63u) != 0) return;
	static std::atomic<bool> busy {false};
	static uint64_t          last_ms = 0, last_tsc = 0, last_flips = 0;
	const uint64_t           now     = PortR::NowMs();
	if (last_ms != 0 && now - last_ms < 5000) return;
	if (busy.exchange(true, std::memory_order_acquire)) return;
	const uint64_t tsc   = __rdtsc();
	const uint64_t flips = LiveControl::g_flips.load(std::memory_order_relaxed);
	if (last_ms != 0 && now > last_ms && tsc > last_tsc) {
		const double secs         = static_cast<double>(now - last_ms) / 1000.0;
		const double us_per_cycle = static_cast<double>(now - last_ms) * 1000.0 / static_cast<double>(tsc - last_tsc);
		PortR::Report(secs, static_cast<double>(flips - last_flips), us_per_cycle,
		              LiveCensus::g_kind_cycles[LiveCensus::Submission]);
	}
	last_ms    = now;
	last_tsc   = tsc;
	last_flips = flips;
	busy.store(false, std::memory_order_release);
}

// eboot+0xab57d0 (owner, view, ctx, list, shadow), on the host. SysV: rdi, rsi, rdx, ecx, r8d.
KYTY_SYSV_ABI uint64_t ListR(uint64_t owner, uint64_t view, uint64_t* ctx, int32_t list, int32_t shadow) {
	MaybeReport();
	auto&          st    = PortR::g_game;
	const uint32_t mode  = PortR::Mode();
	const uint64_t slot  = view + L::ViewLists + static_cast<uint64_t>(static_cast<int64_t>(list)) * L::ViewListStride;
	const uint64_t begin = At<uint64_t>(slot), end = At<uint64_t>(slot + 8);
	if (begin == end) return 0; // (the game's function does nothing either)
	const uint64_t elements = (end - begin) / L::ElemSize;
	const bool     telemetry =
	    (At<uint8_t>(reinterpret_cast<uint64_t>(ctx) + L::CtxTelemetryByte) & 8u) != 0 ||
	    At<uint8_t>(ctx[L::CtxView] + L::ViewTelemetry) == 1;
	if (telemetry || (mode == 1 && !PortR::g_window_on.load(std::memory_order_relaxed))) {
		st.falls[telemetry ? PortR::FTelemetry : PortR::FWindowOff].fetch_add(1, std::memory_order_relaxed);
		const uint64_t t0 = __rdtsc();
		const uint64_t r  = g_game.original(owner, view, ctx, list, shadow);
		st.calls[0].fetch_add(1, std::memory_order_relaxed);
		st.elements[0].fetch_add(elements, std::memory_order_relaxed);
		st.draws[0].fetch_add(elements, std::memory_order_relaxed); // (the game's: at most one draw per element)
		st.cycles[0].fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
		return r;
	}
	const uint64_t t0      = __rdtsc();
	const uint64_t w       = ctx[L::CtxW];
	const bool     reuse   = mode == 1; // (=2: every table new, as the game: the verification sees today's stream)
	const bool     records = mode == 2;
	const auto     pass    = static_cast<uint32_t>(At<int32_t>(reinterpret_cast<uint64_t>(&ctx[L::CtxPass])));
	uint64_t       last_material = 0;
	bool           drawn         = false;
	uint32_t       draws = 0, materials = 0, tnew = 0, treused = 0, skipped = 0, indirect = 0, full = 0;
	uint32_t       last_mask  = 0;
	uint64_t       last_table = 0;
	std::array<uint64_t, L::Sources> last_values {};
	for (uint64_t e = begin; e != end; e += L::ElemSize) {
		const uint64_t inst = At<uint64_t>(e + L::ElemInst);
		const uint64_t obj  = At<uint64_t>(inst + L::InstObj * 8);
		const uint64_t mesh = At<uint64_t>(inst + L::InstMesh * 8);
		// The object's residency touch (as the game: kind 0xb, the default object or a resident one are left alone).
		if (At<uint8_t>(obj + L::ObjKind) != L::ObjKindNoTouch && !g_game.is_default(obj) &&
		    At<int32_t>(obj + L::ObjResident) == 0) {
			const uint64_t owner_obj = At<uint64_t>(obj + L::ObjOwner);
			const auto     get       = reinterpret_cast<GetFunction>(At<uint64_t>(At<uint64_t>(owner_obj) + 0x68));
			const uint64_t streamer  = get(owner_obj);
			const auto     touch     = reinterpret_cast<TouchFunction>(At<uint64_t>(At<uint64_t>(streamer) + 0x10));
			touch(streamer, obj);
		}
		*reinterpret_cast<uint8_t*>(obj + L::ObjDrawn) = 1;
		*reinterpret_cast<uint8_t*>(obj + L::ObjFlags) &= 0xfdu;
		const uint64_t material = At<uint64_t>(obj + L::ObjMaterial);
		if (material == last_material) {
			if (!drawn) { // (the game's: an element with the list's first material null is passed over)
				++skipped;
				continue;
			}
		} else {
			g_game.material(material, ctx);
			last_material = material;
			++materials;
		}
		const uint32_t e14 = At<uint32_t>(e + L::ElemSubmesh), e18 = At<uint32_t>(e + L::ElemFlags);
		g_game.submesh(mesh, ctx, e14, e18, At<uint64_t>(inst + L::InstExtra * 8));
		// The table: the sources the material's mask selects, in order.
		const uint32_t mask = At<uint32_t>(material + L::MatMask);
		std::array<uint64_t, L::Sources> values {};
		for (uint32_t j = 0; j < L::Sources; ++j) values[j] = ctx[L::CtxSources + j];
		bool same = reuse && last_table != 0 && mask == last_mask;
		for (uint32_t j = 0; same && j < L::Sources; ++j)
			if (((mask >> j) & 1u) != 0 && values[j] != last_values[j]) same = false;
		uint64_t table = 0;
		if (same) {
			table = last_table;
			++treused;
		} else {
			std::array<uint64_t, 3> desc {};
			std::memcpy(desc.data(), reinterpret_cast<const void*>(g_game.alloc_desc), sizeof(desc));
			table = g_game.alloc(uint64_t {static_cast<uint32_t>(std::popcount(mask))} << 3u, desc.data(), 8, 0);
			if (table != 0) {
				auto* out = reinterpret_cast<uint64_t*>(table);
				for (uint32_t j = 0; j < L::Sources; ++j)
					if (((mask >> j) & 1u) != 0) *out++ = values[j];
			}
			++tnew;
			last_table  = table;
			last_mask   = mask;
			last_values = values;
		}
		g_game.bind(w, table);
		g_game.draw(inst, ctx, e18);
		drawn = true;
		++draws;
		const bool offset_path = (e18 & 8u) == 0 || (At<uint32_t>(w + L::WFlags) & 4u) != 0;
		if (!offset_path) ++indirect;
		if (records && table != 0) {
			PortR::Record r;
			r.table    = table;
			r.mask     = mask;
			r.words    = static_cast<uint32_t>(std::popcount(mask));
			r.src      = values;
			r.material = material;
			r.mesh     = mesh;
			r.inst     = inst;
			r.vs_obj   = At<uint64_t>(w + L::WVs);
			r.ps_obj   = At<uint64_t>(w + L::WPs);
			std::array<uint32_t, 28> regs {};
			std::memcpy(regs.data(), reinterpret_cast<const void*>(w + L::WBlend), 64);
			std::memcpy(regs.data() + 16, reinterpret_cast<const void*>(w + L::WDepth), 8);
			std::memcpy(regs.data() + 18, reinterpret_cast<const void*>(w + L::WStencil), 16);
			std::memcpy(regs.data() + 22, reinterpret_cast<const void*>(w + L::WViewport), 16);
			regs[26]   = At<uint32_t>(w + L::WPrim);
			regs[27]   = At<uint32_t>(w + L::WMisc);
			r.state    = PortR::HashWords(regs.data(), static_cast<uint32_t>(regs.size()));
			r.list     = static_cast<uint32_t>(list);
			r.shadow   = static_cast<uint32_t>(shadow);
			r.pass     = pass;
			r.stages   = At<uint32_t>(w + L::WStages) & 0x3fu;
			r.path     = offset_path ? 0u : 1u;
			r.submesh  = e14;
			if (offset_path) {
				const uint64_t args = At<uint64_t>(inst + L::InstArgs * 8);
				r.index_offset      = At<uint32_t>(args + L::ArgsOffset);
				r.index_count       = At<uint32_t>(args + L::ArgsCount);
			}
			r.born_ms = PortR::NowMs();
			if (!PortR::Put(r)) ++full;
		}
	}
	st.calls[1].fetch_add(1, std::memory_order_relaxed);
	st.elements[1].fetch_add(elements, std::memory_order_relaxed);
	st.draws[1].fetch_add(draws, std::memory_order_relaxed);
	st.materials.fetch_add(materials, std::memory_order_relaxed);
	st.tables_new.fetch_add(tnew, std::memory_order_relaxed);
	st.tables_reused.fetch_add(treused, std::memory_order_relaxed);
	if (skipped != 0) st.falls[PortR::FSkipped].fetch_add(skipped, std::memory_order_relaxed);
	if (indirect != 0) st.falls[PortR::FIndirect].fetch_add(indirect, std::memory_order_relaxed);
	if (full != 0) st.falls[PortR::FRegistryFull].fetch_add(full, std::memory_order_relaxed);
	if (records) st.records.fetch_add(draws - full, std::memory_order_relaxed);
	st.per_pass[pass < 16 ? pass : 15].fetch_add(draws, std::memory_order_relaxed);
	st.cycles[1].fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
	// (The game's return: the list's element count, read again as it does.)
	return (At<uint64_t>(slot + 8) - At<uint64_t>(slot)) >> 5u;
}

// ---- (portg, KYTY_PORTG=1|2) contador de listas identicas: cada llamada a ab57d0 (clave: vista, lista, sombra, pase)
// se resume en un hash de sus elementos y de lo que cada elemento dibuja (instancia, objeto, malla, material, args del
// draw). Igual que en el frame anterior = la lista se podria reproducir sin generar ni traducir PM4. Solo lee memoria
// que la funcion del juego va a leer justo despues; no cambia nada de lo que se dibuja.
namespace PortG {
[[nodiscard]] uint32_t Mode() {
	static const uint32_t mode = [] {
		const char* t = std::getenv("KYTY_PORTG");
		return (t != nullptr && (t[0] == '1' || t[0] == '2') && t[1] == '\0') ? static_cast<uint32_t>(t[0] - '0') : 0u;
	}();
	return mode;
}
struct Entry {
	uint64_t hash = 0, flip = 0;
};
std::mutex                              g_mutex;
std::unordered_map<uint64_t, Entry>     g_lists;
std::atomic<uint64_t>                   g_calls {0}, g_same {0}, g_draws {0}, g_draws_same {0}, g_new {0}, g_cycles {0};
std::atomic<uint64_t>                   g_pass_draws[16] {}, g_pass_same[16] {};

void Report() {
	static std::atomic<bool> busy {false};
	static uint64_t          last_ms = 0, last_flips = 0, last_tsc = 0;
	static uint64_t          c0 = 0, s0 = 0, d0 = 0, ds0 = 0, cy0 = 0;
	const uint64_t           now = PortR::NowMs();
	if (last_ms != 0 && now - last_ms < 5000) return;
	if (busy.exchange(true, std::memory_order_acquire)) return;
	const uint64_t flips = LiveControl::g_flips.load(std::memory_order_relaxed), tsc = __rdtsc();
	const uint64_t c = g_calls.load(), s = g_same.load(), d = g_draws.load(), ds = g_draws_same.load(), cy = g_cycles.load();
	if (last_ms != 0 && flips > last_flips && tsc > last_tsc) {
		const double fr    = static_cast<double>(flips - last_flips);
		const double us_cy = static_cast<double>(now - last_ms) * 1000.0 / static_cast<double>(tsc - last_tsc);
		const double dpf = static_cast<double>(d - d0) / fr, dspf = static_cast<double>(ds - ds0) / fr;
		std::printf("[portg] %.0f frames: listas/frame %.0f (identicas al frame anterior %.1f %%); draws/frame %.0f "
		            "(en listas identicas %.0f = %.1f %%); TECHO del replay en el hilo de traduccion %.2f ms/frame "
		            "(a 1,33 us/draw de tabla, hilo2) ; coste del contador %.3f ms/frame\n",
		            fr, static_cast<double>(c - c0) / fr, c > c0 ? 100.0 * static_cast<double>(s - s0) / static_cast<double>(c - c0) : 0.0,
		            dpf, dspf, dpf > 0 ? 100.0 * dspf / dpf : 0.0, dspf * 1.33 / 1000.0,
		            static_cast<double>(cy - cy0) * us_cy / 1000.0 / fr);
		std::printf("[portg] por pase (draws/identicos acumulados):");
		for (uint32_t p = 0; p < 16; ++p)
			if (const uint64_t pd = g_pass_draws[p].load(); pd != 0)
				std::printf(" p%u %llu/%llu", p, static_cast<unsigned long long>(pd),
				            static_cast<unsigned long long>(g_pass_same[p].load()));
		std::printf("\n");
		std::fflush(stdout);
	}
	last_ms = now; last_flips = flips; last_tsc = tsc;
	c0 = c; s0 = s; d0 = d; ds0 = ds; cy0 = cy;
	busy.store(false, std::memory_order_release);
}

void Count(uint64_t view, const uint64_t* ctx, int32_t list, int32_t shadow, uint64_t begin, uint64_t end) {
	const uint64_t t0   = __rdtsc();
	const auto     pass = static_cast<uint32_t>(At<int32_t>(reinterpret_cast<uint64_t>(&ctx[L::CtxPass])));
	uint64_t       key  = 0xcbf29ce484222325ull;
	for (const uint64_t v: {view, static_cast<uint64_t>(static_cast<uint32_t>(list)), static_cast<uint64_t>(static_cast<uint32_t>(shadow)),
	                        static_cast<uint64_t>(pass)})
		key = (key ^ v) * 0x100000001b3ull;
	uint64_t h = 0x84222325cbf29ce4ull, n = 0;
	const auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; h ^= h >> 29; };
	for (uint64_t e = begin; e < end; e += L::ElemSize, ++n) {
		mix(At<uint64_t>(e)); mix(At<uint64_t>(e + 8)); mix(At<uint64_t>(e + 16)); mix(At<uint64_t>(e + 24));
		const uint64_t inst = At<uint64_t>(e + L::ElemInst);
		if (inst == 0) continue;
		const uint64_t obj = At<uint64_t>(inst + L::InstObj * 8), mesh = At<uint64_t>(inst + L::InstMesh * 8),
		               args = At<uint64_t>(inst + L::InstArgs * 8);
		mix(obj); mix(mesh); mix(args); mix(At<uint64_t>(inst + L::InstExtra * 8));
		if (obj != 0) mix(At<uint64_t>(obj + L::ObjMaterial));
		if (args != 0) mix(At<uint64_t>(args + L::ArgsOffset));
	}
	const uint64_t flip = LiveControl::g_flips.load(std::memory_order_relaxed);
	bool           same = false;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (g_lists.size() > 65536) g_lists.clear();
		auto& en = g_lists[key];
		same     = en.flip != 0 && en.hash == h && flip > en.flip && flip - en.flip <= 2;
		en.hash  = h;
		en.flip  = flip == 0 ? 1 : flip;
	}
	g_calls.fetch_add(1, std::memory_order_relaxed);
	g_draws.fetch_add(n, std::memory_order_relaxed);
	g_pass_draws[pass & 15u].fetch_add(n, std::memory_order_relaxed);
	if (same) {
		g_same.fetch_add(1, std::memory_order_relaxed);
		g_draws_same.fetch_add(n, std::memory_order_relaxed);
		g_pass_same[pass & 15u].fetch_add(n, std::memory_order_relaxed);
	}
	g_cycles.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
}
} // namespace PortG

KYTY_SYSV_ABI uint64_t List(uint64_t owner, uint64_t view, uint64_t* ctx, int32_t list, int32_t shadow) {
	if (PortG::Mode() != 0) {
		PortG::Report();
		const uint64_t slot  = view + L::ViewLists + static_cast<uint64_t>(static_cast<int64_t>(list)) * L::ViewListStride;
		const uint64_t begin = At<uint64_t>(slot), end = At<uint64_t>(slot + 8);
		if (begin < end && (end - begin) % L::ElemSize == 0 && end - begin < (1ull << 24)) PortG::Count(view, ctx, list, shadow, begin, end);
		if (PortR::Mode() == 0) return g_game.original(owner, view, ctx, list, shadow);
	}
	return ListR(owner, view, ctx, list, shadow);
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	const uint32_t mode = PortR::Mode();
	if ((mode == 0 && PortG::Mode() == 0) || program == nullptr || program->file_name.filename() != "eboot.bin") return;
	const auto refuse = [&](const char* why) {
		std::printf("[port-r] NO instalado (KYTY_PORT_R=%u): %s; el juego va como siempre\n", mode, why);
		std::fflush(stdout);
	};
	if (!Libs::Graphics::DemonsSouls::IsSupportedGame()) return refuse("no es Demon's Souls");
	if (program->mapped_size < L::FrameAllocDesc + 24 || program->base_vaddr > UINT64_MAX - L::FrameAllocDesc - 24)
		return refuse("el eboot es demasiado corto para ser el 01.007.000");
	const uint64_t base    = program->base_vaddr;
	const uint64_t address = base + L::ListFunction;
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, L::ListFunctionSize))
		return refuse("su codigo no se puede leer");
	uint64_t h = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < L::ListFunctionSize; ++i) h = (h ^ reinterpret_cast<const uint8_t*>(address)[i]) * 0x100000001b3ull;
	if (h != L::ListFunctionFnv) {
		std::printf("[port-r] eboot+0x%llx: hash 0x%016llx, el 01.007.000 tiene 0x%016llx\n",
		            static_cast<unsigned long long>(L::ListFunction), static_cast<unsigned long long>(h),
		            static_cast<unsigned long long>(L::ListFunctionFnv));
		return refuse("eboot+0xab57d0 no es la lista de geometria del 01.007.000 (otra version, o ya parcheada)");
	}
	if (g_trampoline == 0)
		g_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
		                                                              "demons_souls_port_r");
	if (g_trampoline == 0) return refuse("no hay memoria para su trampolin");
	std::array<uint8_t, Moved + 12> trampoline {};
	std::memcpy(trampoline.data(), Prologue.data(), Moved);
	const uint64_t resume  = address + Moved;
	trampoline[Moved]      = 0x48; // mov rax, resume; jmp rax
	trampoline[Moved + 1]  = 0xb8;
	std::memcpy(trampoline.data() + Moved + 2, &resume, sizeof(resume));
	trampoline[Moved + 10] = 0xff;
	trampoline[Moved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_trampoline, trampoline.size()))
		return refuse("no se pudo vaciar la cache de instrucciones del trampolin");
	g_game.original   = reinterpret_cast<ListFunction>(g_trampoline);
	g_game.material   = reinterpret_cast<MaterialFunction>(base + L::MaterialApply);
	g_game.submesh    = reinterpret_cast<SubmeshFunction>(base + L::DrawSubmesh);
	g_game.alloc      = reinterpret_cast<AllocFunction>(base + L::FrameAlloc);
	g_game.bind       = reinterpret_cast<BindFunction>(base + L::BindTable);
	g_game.draw       = reinterpret_cast<DrawFunction>(base + L::IssueDraw);
	g_game.is_default = reinterpret_cast<TestFunction>(base + L::IsDefaultObject);
	g_game.alloc_desc = base + L::FrameAllocDesc;
	std::array<uint8_t, Moved> entry {0x48, 0xb8}; // mov rax, List; jmp rax
	const auto                 hook = reinterpret_cast<uint64_t>(&List);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), Prologue.data(), Moved);
		Common::VirtualMemory::FlushInstructionCache(address, Moved);
		return refuse("no se pudo vaciar la cache de instrucciones de su entrada");
	}
	if (mode != 0) PortR::g_installed.store(true, std::memory_order_release);
	if (PortG::Mode() != 0)
		std::printf("[portg] instalado (KYTY_PORTG=%u): contador de listas identicas de eboot+0xab57d0; linea [portg] cada 5 s\n", PortG::Mode());
	if (mode == 2) PortR::g_verify.store(true, std::memory_order_release);
	if (mode != 0)
	std::printf("[port-r] instalado (KYTY_PORT_R=%u): la lista de geometria eboot+0xab57d0 del 01.007.000 la ejecuta el host "
	            "(PortR::List)%s; linea [port-r] cada 5 s\n",
	            mode,
	            mode == 2 ? "; VERIFICACION: tablas como el juego, cada draw comparado campo a campo con lo que traduce el "
	                        "emulador (DIFERENCIAS), nada nativo se dibuja"
	            : PortR::Alternates() ? "; tablas iguales reutilizadas; ventanas de 5 s ON (host) / OFF (funcion del juego)"
	                                  : "; tablas iguales reutilizadas; siempre ON (KYTY_PORT_R_ALTERNA=0)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

} // namespace Loader::DemonsSoulsPortR
