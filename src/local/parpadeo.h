#pragma once
// (parpadeo2) KYTY_PARPADEO: the one-frame loss of the world's background geometry (Boletaria's cliffs and walls) with
// KYTY_RAIZ=1. Those objects are XPR draws whose arguments a GPU compute pass writes in the same frame (Ghidra,
// projelf_un3: 0xab6540 "XPR Index Cull" on the DCB - a cs_memset32 of the 15,000 x 0x14-byte argument entries
// (entry[5], e+0x28) with a 20-byte zero pattern from a frame allocator, one dispatch per object, a WRITE_DATA of its
// label and a WAIT_REG_MEM on it in the same DCB -; 0xab6080 then SET_BASE_INDIRECT_ARGS(entry[5]) and one
// DRAW_INDEX_INDIRECT per object, argument offset index*0x14; 0xa1f020, the CPU reading the counters, is not called
// in this mode: 0 calls/s in 20261116-070419). A frame whose arguments the draws read as zero loses exactly that
// geometry for that frame.
//
// KYTY_PARPADEO (off by default; the launcher sets it):
//   1  counts and the first events, on the GPU thread, no GPU work added:
//      - every DRAW_INDEX_INDIRECT run (native XPR, table, normal path): its argument base (learned: the bases are the
//        index culls' outputs), whether its arguments are CPU-DIRTY at the draw (the draw then uploads the game's guest
//        bytes over what the GPU wrote: hypothesis 2), whether they are neither GPU- nor CPU-written (the GPU buffer then
//        holds the guest bytes: read here, counted when every draw of the run has index_count 0), whether the index
//        range is CPU-dirty at the pass's first draw, whether the Vulkan buffer holding the arguments changed since the
//        last frame (a buffer made again / moved);
//      - every cs_memset32 dispatch into a known argument range (the cull's zeroing): the guest bytes of its 20-byte
//        source pattern (must be zero), whether the buffer cache sees that source CPU-written (if not, the GPU reads its
//        cached copy of the frame allocator's page: a stale source fills the arguments with old bytes);
//      - normal-path draws that read their arguments on the CPU (after SyncGpuCleanBacking) with index_count 0.
//   2  as 1, plus the GPU's truth: at each argument base's first draw run of a frame its entries are copied on the GPU,
//      right where the draw reads them (after the draw's own synchronization), to a host buffer; read once that work
//      completed. A base whose entries were mostly non-zero and are ALL zero in one frame is the defect seen on the
//      GPU ("VACIO" lines, with what the CPU side saw in that frame). Costs one end of the render pass a base a frame.
//   3  (parpadeo3) as 2, but the snapshot is ALL of the base's argument range (as far as its draws have reached, at
//      most 15,000 entries): per base and frame the draws whose index_count is not 0 and the triangles (sum / 3) as the
//      GPU draws them; a frame where they fall under 80 % of BOTH neighbouring frames' (and by >= 16 draws or >= 2,000
//      triangles) is a "PRIMITIVAS CAEN (GPU)" line. Costs a copy of up to 300 KB per base and frame (~26 bases).
//   Any mode (parpadeo3): per base, the draws the GAME recorded each frame (CPU side); a frame with under 80 % of both
//   neighbours' (by >= 4) is a "DRAWS CAEN (CPU)" line, and a base that drew in frames F-2, F-1 and F+1 but not in F a
//   "PASADA SIN DRAWS" line. The game records one draw per visible chunk (0xa1f560 reads the chunk bits a GPU dispatch
//   wrote: KYTY_PARPADEO_XPR), so a drop here with the GPU's arguments intact points at the CPU's decision.
// KYTY_PARPADEO_FIX (off by default; bits):
//   1  the cull's zeroing (cs_memset32 with a period shorter than the count, into a known argument range, its source
//      pattern zero in guest memory) made as a GPU fill of zeros (vkCmdFillBuffer, in order in the command buffer)
//      instead of the shader reading the frame allocator's 20 bytes through the buffer cache. Removes the stale-source
//      path; the arguments stay GPU-only (as the cull's own writes make them).
// One "[parpadeo]" line every 5 s; "[parpadeo2] ..." event lines (the first 48 of each kind).
//
// Threads: the GPU thread (the command processor); with KYTY_PAR_QUEUES the ACB's too: one mutex (uncontended).

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "local/parpadeo5.h" // (parpadeo5: the game's side of each base, KYTY_PARPADEO5)

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>

namespace Libs::Graphics::Parpadeo {

inline int EnvLevel(const char* name, int fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	return static_cast<int>(std::strtol(value, nullptr, 0));
}
// (Read at first use: the port sets the settings' variables after the process started.)
[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const int m = EnvLevel("KYTY_PARPADEO", 0);
		if (m != 0)
			std::printf("[parpadeo2] KYTY_PARPADEO=%d (%s), KYTY_PARPADEO_FIX=%d (%s). Linea [parpadeo] cada 5 s.\n", m,
			            m >= 3 ? "contadores + copia en la GPU de TODOS los argumentos de cada base y frame (parpadeo3)"
			            : m >= 2 ? "contadores + copia en la GPU de los argumentos al dibujar" : "contadores",
			            EnvLevel("KYTY_PARPADEO_FIX", 0),
			            (EnvLevel("KYTY_PARPADEO_FIX", 0) & 1) != 0 ? "puesta a cero de argumentos como relleno en la GPU"
			                                                         : "sin arreglo");
		std::fflush(stdout);
		return m;
	}();
	return mode;
}
[[nodiscard]] inline bool On() {
	return Mode() != 0;
}
[[nodiscard]] inline uint32_t Fix() {
	static const uint32_t fix = static_cast<uint32_t>(EnvLevel("KYTY_PARPADEO_FIX", 0));
	return fix;
}

enum Path : uint32_t { PathNative = 0, PathMulti, PathNormal, PathNormalMulti, Paths };
inline constexpr const char* PathNames[Paths] = {"nativo", "nativo-multi", "normal", "normal-multi"};
inline constexpr uint64_t    Entry            = 0x14; // (DrawIndexedIndirectArgs)

struct Base {
	uint64_t address = 0, extent = Entry;
	// The frame of the fields below.
	uint64_t frame = UINT64_MAX;
	uint32_t draws = 0, cpu_dirty = 0, clean_zero = 0, index_dirty = 0, fills = 0, fill_bad = 0;
	// The cull's zeroing comes before the pass's draws: counted here, taken by the next frame Roll starts.
	uint32_t pending_fills = 0, pending_bad = 0;
	bool     first_done = false, snapped = false, buffer_changed = false;
	// The entries this frame's draws noted (a refused native run is offered again packet by packet).
	std::array<uint64_t, 15360 / 64> noted {};
	VkBuffer last_buffer = VK_NULL_HANDLE; // (the arguments' Vulkan buffer at the last frame's first draw)
	// GPU snapshots (mode 2): the last read result.
	int64_t  last_nonzero = -1;
	uint32_t last_entries = 0;
	uint64_t last_frame   = 0;
	// (parpadeo3) The game's draws in the two frames before `frame` (CPU side), and the mode 3 results of the last two
	// snapshots (frame, non-zero draws, indices).
	uint64_t h_frame[2] = {UINT64_MAX, UINT64_MAX};
	uint32_t h_draws[2] = {0, 0};
	uint64_t g_frame[2] = {UINT64_MAX, UINT64_MAX};
	uint64_t g_nonzero[2] = {0, 0}, g_sum[2] = {0, 0};
	uint32_t g_flags[2] = {0, 0};
};

struct Slot {
	uint64_t tick = 0, frame = 0;
	uint32_t base = 0, entries = 0, flags = 0;
	bool     busy = false;
};
// Flags of the frame a snapshot was taken in (what the CPU side saw for its base by then).
enum SnapFlag : uint32_t { FlagCpuDirty = 1, FlagCleanZero = 2, FlagIndexDirty = 4, FlagFillBad = 8, FlagBufferChanged = 16,
	                       FlagFixed = 32 };

struct Counters {
	uint64_t runs = 0, draws = 0, path[Paths] {}, cpu_dirty = 0, clean_zero = 0, index_dirty = 0, buffer_changed = 0,
	         cpu_zero = 0, cpu_reads = 0, fills = 0, fill_nonzero_src = 0, fill_src_not_dirty = 0, fill_src_gpu = 0,
	         fixed = 0, snaps = 0, snaps_read = 0, empty = 0, frames = 0;
	uint64_t cpu_drops = 0, pass_gaps = 0, gpu_drops = 0; // (parpadeo3)
};

struct State {
	std::mutex                    mutex;
	std::array<Base, 64>          bases {};
	uint32_t                      count = 0;
	Counters                      c {};
	std::array<uint32_t, 8>       logged {};
	std::chrono::steady_clock::time_point last {};
	uint64_t                      first_frame = UINT64_MAX, last_frame = 0;
	std::unique_ptr<Buffer>       readback;
	// (parpadeo3: mode 3 copies a base's whole range, 15,000 x 0x14 bytes at most.)
	static constexpr uint32_t     SlotBytes = 0x4a000, SlotCount = 64;
	std::array<Slot, SlotCount>   slots {};
	uint32_t                      cursor = 0;
};
[[nodiscard]] inline State& S() {
	static State state;
	return state;
}

// Event kinds (the first 48 lines of each).
enum Kind : uint32_t { KindCpuDirty = 0, KindCleanZero, KindIndexDirty, KindBuffer, KindCpuZero, KindFill, KindEmpty,
	                   KindCpuDrop, KindPassGap, KindGpuDrop, Kinds };
inline bool Log(State& s, Kind kind) {
	return s.logged[kind]++ < 48;
}

inline Base* FindBase(State& s, uint64_t address, bool add) {
	for (uint32_t i = 0; i < s.count; ++i)
		if (s.bases[i].address == address) return &s.bases[i];
	if (!add || s.count >= s.bases.size()) return nullptr;
	auto& base   = s.bases[s.count++];
	base         = {};
	base.address = address;
	return &base;
}
// The known argument range [address, address + extent) that [vaddr, vaddr + size) overlaps.
inline Base* OverlappingBase(State& s, uint64_t vaddr, uint64_t size) {
	for (uint32_t i = 0; i < s.count; ++i) {
		auto& base = s.bases[i];
		if (vaddr < base.address + std::max<uint64_t>(base.extent, Entry) && base.address < vaddr + size) return &base;
	}
	return nullptr;
}
inline uint64_t UnixMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
// (parpadeo3) `frame` begins for the base: its last frame (base.frame, base.draws) is complete. The middle of the last
// three frames the base drew in, judged against its neighbours.
inline void JudgeDraws(State& s, Base& base, uint64_t frame) {
	if (base.frame == UINT64_MAX) return;
	const uint64_t f0 = base.h_frame[0], f1 = base.h_frame[1], f2 = base.frame;
	const uint32_t d0 = base.h_draws[0], d1 = base.h_draws[1], d2 = base.draws;
	if (f0 != UINT64_MAX && f1 == f0 + 1 && f2 == f1 + 1) {
		const uint32_t low = (std::min)(d0, d2);
		if (d1 * 5 < low * 4 && low >= d1 + 4) {
			++s.c.cpu_drops;
			if (Log(s, KindCpuDrop)) {
				std::printf("[parpadeo3] DRAWS CAEN (CPU): frame %" PRIu64 ", base 0x%" PRIx64 ": el juego grabo %u draws (frame anterior %u, "
				            "siguiente %u); unix_ms~%" PRIu64 "\n",
				            f1, base.address, d1, d0, d2, UnixMs());
				std::fflush(stdout);
				// (parpadeo5) What the game's 0xa1f290 recorded for this base (KYTY_PARPADEO5).
				Parpadeo5::PrintRecent(base.address);
			}
		}
	}
	// Drew in f1 and f2 = f1 + 1 (and the frame before), not in f2 + 1, again in `frame` = f2 + 2.
	if (f1 != UINT64_MAX && f2 == f1 + 1 && frame == f2 + 2 && d2 >= 4) {
		++s.c.pass_gaps;
		if (Log(s, KindPassGap)) {
			std::printf("[parpadeo3] PASADA SIN DRAWS: frame %" PRIu64 ", base 0x%" PRIx64 ": ningun draw ese frame (frames %" PRIu64
			            " y %" PRIu64 ": %u y %u draws; vuelve en el %" PRIu64 "); unix_ms~%" PRIu64 "\n",
			            f2 + 1, base.address, f1, f2, d1, d2, frame, UnixMs());
			std::fflush(stdout);
		}
	}
	base.h_frame[0] = f1, base.h_draws[0] = d1;
	base.h_frame[1] = f2, base.h_draws[1] = d2;
}
inline void Roll(State& s, Base& base, uint64_t frame) {
	if (base.frame == frame) return;
	JudgeDraws(s, base, frame);
	base.frame          = frame;
	base.draws          = 0;
	base.cpu_dirty      = 0;
	base.clean_zero     = 0;
	base.index_dirty    = 0;
	base.fills          = base.pending_fills;
	base.fill_bad       = base.pending_bad;
	base.pending_fills  = 0;
	base.pending_bad    = 0;
	base.first_done     = false;
	base.snapped        = false;
	base.buffer_changed = false;
	base.noted.fill(0);
}
// True the first time this frame (entries past the table's 15,360 always true).
inline bool NoteEntry(Base& base, uint64_t address) {
	if (address < base.address) return true;
	const uint64_t index = (address - base.address) / Entry;
	if (index >= base.noted.size() * 64) return true;
	auto&          word = base.noted[index / 64];
	const uint64_t bit  = uint64_t {1} << (index % 64);
	if ((word & bit) != 0) return false;
	word |= bit;
	return true;
}
inline uint32_t FrameFlags(const Base& base) {
	return (base.cpu_dirty ? FlagCpuDirty : 0u) | (base.clean_zero ? FlagCleanZero : 0u) |
	       (base.index_dirty ? FlagIndexDirty : 0u) | (base.fill_bad ? FlagFillBad : 0u) |
	       (base.buffer_changed ? FlagBufferChanged : 0u) | ((Fix() & 1u) != 0 && base.fills != 0 ? FlagFixed : 0u);
}
inline void PrintFlags(uint32_t flags, char* out, size_t size) {
	std::snprintf(out, size, "%s%s%s%s%s%s", (flags & FlagCpuDirty) ? " args-sucios-de-CPU" : "",
	              (flags & FlagCleanZero) ? " args-limpios-a-cero" : "", (flags & FlagIndexDirty) ? " indices-sucios-de-CPU" : "",
	              (flags & FlagFillBad) ? " puesta-a-cero-dudosa" : "", (flags & FlagBufferChanged) ? " buffer-cambiado" : "",
	              (flags & FlagFixed) ? " (relleno-GPU)" : "");
	if (out[0] == '\0') std::snprintf(out, size, " nada");
}

// Completed snapshots (mode 2): read and judged. Under the mutex.
inline void ReadSnapshots(State& s, CommandScheduler& scheduler) {
	if (!s.readback) return;
	const uint64_t done = scheduler.GetMasterSemaphore().KnownGpuTick();
	for (uint32_t i = 0; i < State::SlotCount; ++i) {
		auto& slot = s.slots[i];
		if (!slot.busy || slot.tick > done) continue;
		slot.busy = false;
		s.readback->Invalidate(uint64_t {i} * State::SlotBytes, (std::max)(uint64_t {slot.entries} * Entry, uint64_t {256}));
		const uint8_t* bytes   = s.readback->Mapped().data() + uint64_t {i} * State::SlotBytes;
		int64_t        nonzero = 0;
		uint64_t       sum     = 0;
		for (uint32_t e = 0; e < slot.entries; ++e) {
			uint32_t count = 0;
			std::memcpy(&count, bytes + e * Entry, sizeof(count));
			nonzero += count != 0 ? 1 : 0;
			sum += count;
		}
		++s.c.snaps_read;
		if (slot.base >= s.count) continue;
		auto& base = s.bases[slot.base];
		// A frame after the previous snapshot's (snapshots complete in order).
		if (base.last_nonzero >= 4 && nonzero == 0 && slot.entries >= 4 && slot.frame > base.last_frame) {
			++s.c.empty;
			if (Log(s, KindEmpty)) {
				char flags[160];
				PrintFlags(slot.flags, flags, sizeof(flags));
				std::printf("[parpadeo2] VACIO en la GPU: frame %" PRIu64 ", base 0x%" PRIx64
				            ": los %u argumentos del primer grupo de draws tienen index_count 0 al dibujar (frame %" PRIu64
				            ": %" PRId64 " de %u no nulos). Visto en la CPU ese frame:%s\n",
				            slot.frame, base.address, slot.entries, base.last_frame, base.last_nonzero, base.last_entries,
				            flags);
				std::fflush(stdout);
			}
		}
		if (slot.frame >= base.last_frame) {
			base.last_nonzero = nonzero;
			base.last_entries = slot.entries;
			base.last_frame   = slot.frame;
		}
		// (parpadeo3) Mode 3: the middle of the last three consecutive frames against both neighbours.
		if (Mode() >= 3 && (base.g_frame[1] == UINT64_MAX || slot.frame > base.g_frame[1])) {
			const uint64_t f0 = base.g_frame[0], f1 = base.g_frame[1], f2 = slot.frame;
			if (f0 != UINT64_MAX && f1 == f0 + 1 && f2 == f1 + 1) {
				const uint64_t n0 = base.g_nonzero[0], n1 = base.g_nonzero[1], n2 = static_cast<uint64_t>(nonzero);
				const uint64_t t0 = base.g_sum[0] / 3, t1 = base.g_sum[1] / 3, t2 = sum / 3;
				const uint64_t nl = (std::min)(n0, n2), tl = (std::min)(t0, t2);
				if ((n1 * 5 < nl * 4 && nl >= n1 + 16) || (t1 * 5 < tl * 4 && tl >= t1 + 2000)) {
					++s.c.gpu_drops;
					if (Log(s, KindGpuDrop)) {
						char flags[160];
						PrintFlags(base.g_flags[1], flags, sizeof(flags));
						std::printf("[parpadeo3] PRIMITIVAS CAEN (GPU): frame %" PRIu64 ", base 0x%" PRIx64
						            ": draws no nulos %" PRIu64 " (anterior %" PRIu64 ", siguiente %" PRIu64 "), triangulos %" PRIu64
						            " (anterior %" PRIu64 ", siguiente %" PRIu64 "). Visto en la CPU ese frame:%s\n",
						            f1, base.address, n1, n0, n2, t1, t0, t2, flags);
						std::fflush(stdout);
					}
				}
			}
			base.g_frame[0] = f1, base.g_nonzero[0] = base.g_nonzero[1], base.g_sum[0] = base.g_sum[1];
			base.g_flags[0] = base.g_flags[1];
			base.g_frame[1] = f2, base.g_nonzero[1] = static_cast<uint64_t>(nonzero), base.g_sum[1] = sum;
			base.g_flags[1] = slot.flags;
		}
	}
}

inline void Report(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (s.last == std::chrono::steady_clock::time_point {}) s.last = now;
	const double seconds = std::chrono::duration<double>(now - s.last).count();
	if (seconds < 5.0) return;
	s.last               = now;
	const auto& c        = s.c;
	const uint64_t frames = s.last_frame >= s.first_frame && s.first_frame != UINT64_MAX ? s.last_frame - s.first_frame + 1 : 0;
	std::printf("[parpadeo] %.1f s, %" PRIu64 " frames: bases de argumentos %u; grupos de draws indirectos %" PRIu64
	            " (%" PRIu64 " draws: nativo %" PRIu64 ", nativo-multi %" PRIu64 ", normal %" PRIu64 ", normal-multi %" PRIu64
	            "); ARGS SUCIOS DE CPU al dibujar %" PRIu64 " (la GPU recibe los bytes del juego encima de los del computo); "
	            "args limpios con todos los contadores a 0 %" PRIu64 "; indices sucios de CPU %" PRIu64
	            "; buffer de args cambiado entre frames %" PRIu64 "; lecturas de args en la CPU %" PRIu64 " (con contador 0 %" PRIu64
	            "); puestas a cero de args (cs_memset32) %" PRIu64 ": fuente NO NULA en memoria %" PRIu64
	            ", fuente no marcada escrita %" PRIu64 ", fuente escrita por la GPU %" PRIu64 ", hechas como relleno en la GPU %" PRIu64
	            "; GPU (modo 2): copias %" PRIu64 ", leidas %" PRIu64 ", VACIOS %" PRIu64
	            "; (parpadeo3) DRAWS CAEN (CPU) %" PRIu64 ", PASADAS SIN DRAWS %" PRIu64 ", PRIMITIVAS CAEN (GPU, modo 3) %" PRIu64 "\n",
	            seconds, frames, s.count, c.runs, c.draws, c.path[PathNative], c.path[PathMulti], c.path[PathNormal],
	            c.path[PathNormalMulti], c.cpu_dirty, c.clean_zero, c.index_dirty, c.buffer_changed, c.cpu_reads, c.cpu_zero,
	            c.fills, c.fill_nonzero_src, c.fill_src_not_dirty, c.fill_src_gpu, c.fixed, c.snaps, c.snaps_read, c.empty,
	            c.cpu_drops, c.pass_gaps, c.gpu_drops);
	std::fflush(stdout);
	s.c           = {};
	s.first_frame = UINT64_MAX;
}

// A DRAW_INDEX_INDIRECT run (or one draw) about to be drawn: `args` the argument addresses (base + offset each),
// `index_base`/`index_bytes` the bound index range. GPU thread, not a speculation's (the caller checks). `mutex`: the
// renderer's (mode 2 records GPU work).
inline void OnDraws(BufferCache& cache, CommandBuffer& command, CommandScheduler& scheduler, RenderMutex& mutex,
                    uint64_t base_address, std::span<const uint64_t> args, uint64_t stride, uint64_t index_base,
                    uint64_t index_bytes, Path path) {
	if (args.empty() || base_address == 0) return;
	// (parpadeo5) The chunk-bit ranges the game read since: how the buffer cache holds them now (its own mutex first,
	// not under ours).
	Parpadeo5::CheckPending([&](uint64_t at, uint64_t bytes) -> uint32_t {
		if (cache.IsRegionCpuModified(at, bytes)) return Parpadeo5::FlagGpuCpuDirty;
		if (cache.IsRegionGpuModified(at, bytes)) return Parpadeo5::FlagGpuGpuWritten;
		if (!cache.IsRegionRegistered(at, bytes)) return Parpadeo5::FlagGpuUnknown;
		return 0u;
	});
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	const uint64_t         frame = command.GetContext().FrameNumber();
	if (s.first_frame == UINT64_MAX) s.first_frame = frame;
	s.last_frame = std::max(s.last_frame, frame);
	auto* base   = FindBase(s, base_address, true);
	if (base == nullptr) return Report(s);
	Roll(s, *base, frame);
	// Only the entries not noted yet this frame (a refused run is offered again from its next packet).
	std::array<uint64_t, 64> fresh {};
	size_t                   kept = 0;
	for (const auto a: args)
		if (kept < fresh.size() && NoteEntry(*base, a)) fresh[kept++] = a;
	if (kept == 0) return Report(s);
	args = {fresh.data(), kept};
	++s.c.runs;
	s.c.draws += args.size();
	s.c.path[path] += args.size();
	base->draws += static_cast<uint32_t>(args.size());
	for (const auto a: args)
		if (a >= base_address) base->extent = std::max(base->extent, a - base_address + Entry);
	// The arguments' state, a group of consecutive entries at a time.
	for (size_t first = 0; first < args.size();) {
		size_t n = 1;
		while (first + n < args.size() && args[first + n] == args[first] + n * stride) ++n;
		const uint64_t at = args[first], bytes = (n - 1) * stride + Entry;
		if (cache.IsRegionCpuModified(at, bytes)) {
			++s.c.cpu_dirty;
			++base->cpu_dirty;
			if (Log(s, KindCpuDirty)) {
				uint32_t guest = 0xffffffffu;
				(void)Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(at, &guest, sizeof(guest));
				std::printf("[parpadeo2] ARGS SUCIOS DE CPU: frame %" PRIu64 ", ruta %s, base 0x%" PRIx64 " entrada %" PRIu64
				            ", %zu draws: el draw sube la memoria del juego (index_count alli %u) encima de lo que dejo la GPU; "
				            "escritos por la GPU %d\n",
				            frame, PathNames[path], base_address, (at - base_address) / Entry, n, guest,
				            cache.IsRegionGpuModified(at, bytes) ? 1 : 0);
				std::fflush(stdout);
			}
		} else if (!cache.IsRegionGpuModified(at, bytes)) {
			// Neither: the buffer holds the guest's bytes. All zero counts in the group is a frame without these objects.
			bool all_zero = true, read = true;
			for (size_t i = 0; i < n && all_zero; ++i) {
				uint32_t count = 0;
				read           = Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(args[first + i], &count, sizeof(count));
				if (!read) break;
				all_zero = count == 0;
			}
			if (read && all_zero && n >= 4) {
				++s.c.clean_zero;
				++base->clean_zero;
				if (Log(s, KindCleanZero)) {
					std::printf("[parpadeo2] ARGS LIMPIOS A CERO: frame %" PRIu64 ", ruta %s, base 0x%" PRIx64 " entrada %" PRIu64
					            ", %zu draws: ni la GPU ni la CPU los marcan escritos y en memoria todos tienen index_count 0\n",
					            frame, PathNames[path], base_address, (at - base_address) / Entry, n);
					std::fflush(stdout);
				}
			}
		}
		first += n;
	}
	if (!base->first_done) {
		base->first_done = true;
		if (index_base != 0 && index_bytes != 0 && cache.IsRegionCpuModified(index_base, index_bytes)) {
			++s.c.index_dirty;
			++base->index_dirty;
			if (Log(s, KindIndexDirty)) {
				std::printf("[parpadeo2] INDICES SUCIOS DE CPU: frame %" PRIu64 ", base de args 0x%" PRIx64 ", indices 0x%" PRIx64
				            " +0x%" PRIx64 " (escritos por la GPU %d): el draw sube la memoria del juego encima\n",
				            frame, base_address, index_base, index_bytes, cache.IsRegionGpuModified(index_base, index_bytes) ? 1 : 0);
				std::fflush(stdout);
			}
		}
		const auto     id    = cache.RegisteredBuffer(args[0], Entry);
		const VkBuffer owner = id ? static_cast<VkBuffer>(cache.GetBuffer(id).Handle()) : VK_NULL_HANDLE;
		if (base->last_buffer != VK_NULL_HANDLE && owner != base->last_buffer) {
			++s.c.buffer_changed;
			base->buffer_changed = true;
			if (Log(s, KindBuffer)) {
				std::printf("[parpadeo2] BUFFER DE ARGS CAMBIADO: frame %" PRIu64 ", base 0x%" PRIx64 " (%s)\n", frame,
				            base_address, owner == VK_NULL_HANDLE ? "ninguno registrado" : "otro objeto Vulkan");
				std::fflush(stdout);
			}
		}
		base->last_buffer = owner;
	}
	// Mode 2: the first group's entries as the GPU holds them here (after the draw's own synchronization).
	if (Mode() >= 2 && !base->snapped && !command.IsInvalid()) {
		base->snapped = true;
		ReadSnapshots(s, scheduler);
		size_t n = 1;
		while (n < args.size() && args[n] == args[0] + n * stride) ++n;
		// (parpadeo3) Mode 3: the whole range the base's draws have reached, from its start.
		const bool     whole = Mode() >= 3;
		const uint64_t from  = whole ? base->address : args[0];
		const uint64_t bytes = whole ? std::min<uint64_t>(std::max<uint64_t>(base->extent, Entry), State::SlotBytes)
		                             : std::min<uint64_t>((n - 1) * stride + Entry, State::SlotBytes);
		auto&          slot  = s.slots[s.cursor];
		if (!slot.busy) {
			RenderLockGuard render(mutex);
			if (!s.readback)
				s.readback = std::make_unique<Buffer>(command.GetGraphics(), scheduler, MemoryUsage::Download, 0,
				                                      vk::BufferUsageFlagBits::eTransferDst,
				                                      uint64_t {State::SlotBytes} * State::SlotCount);
			const auto [owner, offset] = cache.ObtainBuffer(from, bytes, false);
			if (owner != nullptr) {
				s.readback->CopyFrom(command, *owner, offset, uint64_t {s.cursor} * State::SlotBytes, bytes,
				                     vk::AccessFlagBits::eMemoryWrite,
				                     vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
				                         vk::AccessFlagBits::eHostRead,
				                     vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
				                     vk::AccessFlagBits::eHostRead);
				slot = {scheduler.CurrentTick(), frame, static_cast<uint32_t>(base - s.bases.data()),
				        static_cast<uint32_t>(whole ? bytes / Entry : std::min<uint64_t>(n, State::SlotBytes / Entry)),
				        FrameFlags(*base), true};
				s.cursor = (s.cursor + 1) % State::SlotCount;
				++s.c.snaps;
			}
		}
	} else if (Mode() >= 2) {
		ReadSnapshots(s, scheduler);
	}
	Report(s);
}

// A normal-path draw read its arguments on the CPU (after SyncGpuCleanBacking): its index count.
inline void OnCpuArgs(uint64_t frame, uint64_t base_address, uint64_t at, uint32_t index_count) {
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	++s.c.cpu_reads;
	auto* base = FindBase(s, base_address, false);
	if (index_count != 0 || base == nullptr) return;
	++s.c.cpu_zero;
	if (Log(s, KindCpuZero)) {
		std::printf("[parpadeo2] ARGS LEIDOS EN LA CPU CON CONTADOR 0: frame %" PRIu64 ", base 0x%" PRIx64 " entrada %" PRIu64 "\n",
		            frame, base_address, (at - base_address) / Entry);
		std::fflush(stdout);
	}
}

// The game's cs_memset32 dispatch (dst[i] = src[i % period], `count` dwords) about to run. True: done here (the fix).
// Not a speculation's (the caller checks). The frame: the last one a draw saw.
inline bool OnFill(BufferCache& cache, uint64_t dst, uint64_t src, uint64_t count, uint64_t period) {
	if (dst == 0 || count == 0 || period == 0 || period >= count || period > 64) return false;
	auto&                  s = S();
	const std::scoped_lock lock(s.mutex);
	const uint64_t         bytes = count * 4;
	auto*                  base  = OverlappingBase(s, dst, bytes);
	if (base == nullptr) return false;
	const uint64_t frame = s.last_frame; // (the previous frame's number while this one has not drawn yet)
	// (A dispatch is offered twice when the register check refuses it first: DispatchDirect's two tries. The same
	// dispatch again in a row is not counted again; the fix, if it applies, applied at the first.)
	static uint64_t last_dst = 0, last_src = 0, last_runs = UINT64_MAX;
	const bool      again = last_dst == dst && last_src == src && last_runs == s.c.runs;
	last_dst = dst, last_src = src, last_runs = s.c.runs;
	if (again) return false;
	++s.c.fills;
	++base->pending_fills;
	std::array<uint32_t, 64> pattern {};
	const bool read     = Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(src, pattern.data(), period * 4);
	bool       zero     = read;
	for (uint64_t i = 0; i < period && zero; ++i) zero = pattern[i] == 0;
	const bool src_gpu   = cache.IsRegionGpuModified(src, period * 4);
	const bool src_dirty = cache.IsRegionCpuModified(src, period * 4);
	if (read && !zero) ++s.c.fill_nonzero_src;
	if (!src_dirty) ++s.c.fill_src_not_dirty;
	if (src_gpu) ++s.c.fill_src_gpu;
	if ((read && !zero) || src_gpu) {
		++base->pending_bad;
		if (Log(s, KindFill)) {
			std::printf("[parpadeo2] PUESTA A CERO DUDOSA: frame %" PRIu64 ", args 0x%" PRIx64 " (+0x%" PRIx64 "), fuente 0x%" PRIx64
			            " (%" PRIu64 " dwords): en memoria %s, marcada escrita por la CPU %d, por la GPU %d\n",
			            frame, dst, bytes, src, period, !read ? "ilegible" : zero ? "cero" : "NO NULA", src_dirty ? 1 : 0,
			            src_gpu ? 1 : 0);
			std::fflush(stdout);
		}
	}
	// The fix: the pattern the game wrote is zero - the arguments zeroed on the GPU, in order, without the source.
	if ((Fix() & 1u) == 0 || !read || !zero || (dst & 3u) != 0) return false;
	const auto [owner, offset] = cache.ObtainBuffer(dst, bytes, true, true);
	if (owner == nullptr) return false;
	owner->Fill(offset, bytes, 0);
	++s.c.fixed;
	return true;
}

} // namespace Libs::Graphics::Parpadeo
