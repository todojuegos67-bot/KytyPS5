#include "loader/demonsSoulsCopy.h"

#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "ds-linear-copy-hook.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "loader/demonsSoulsTextureProbe.h"
#include "local/buf-nativo.h" // (BUF-NATIVO)
#include "local/malla-nativa.h" // (MALLA-NATIVA)
#include "loader/demonsSoulsPortR.h" // (port-r, KYTY_PORT_R)
#include "loader/demonsSoulsWriteAudit.h"
#include "loader/runtimeLinker.h"
#include "local/tex-directa.h" // (etapa2.1b) KYTY_TEX_DIRECTA: el memcpy real del juego pasa por CoherentMove
#include "render1.h" // (render1)
#include "f5p1-techo.h" // (f5p1 techo, KYTY_F5P1_TECHO)
#include "port2.h" // (port2, KYTY_PORT2)
#include "native/native-render.h" // (f5, KYTY_NATIVE_RENDER)
#include "native/g1-probe.h" // (g1)

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <xxhash.h>
#include <x86intrin.h>

#include "local/frame-nativo2.h" // (FRAME-NATIVO2 v7: el gancho de NextFrame)
#include "local/arena-dueno.h" // (ARENA-DUENO: KYTY_ARENA_DUENO=1, medidor)
#include "local/ds-nativo-pase.h" // (F5 pase 1: KYTY_NATIVO_PASE_SSRCLEAR)
#include "local/ds-nativo-sombras.h" // (F5 pase 2: KYTY_NATIVO_PASE_SOMBRAS=2, extractor)

extern "C" {
// KYTY_GUEST_MOVE_COMPARE (default on; 0 for an A/B): a large move writes only the destination pages whose bytes
// differ (CompareMove).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_move_compare {1};
}

namespace Libs::Graphics::TexAntEarly {
void Install(); // (etapa1b) descriptors.cpp: ganchos de KYTY_TEX_ANTICIPADA al cargar el eboot
} // namespace Libs::Graphics::TexAntEarly

namespace Loader::DemonsSoulsCopy {
namespace {
uint64_t                site = 0;
std::array<uint8_t, 13> original {}, patch {};

// A large move into memory the GPU caches, page by page: a page the GPU wrote (its bytes here are stale, and reading them
// would fault) or whose bytes differ is written (runs prepared as overwritten, Memory::TryPrepareHostOverwrite); a page
// that already holds the source's bytes is left alone, clean: no upload of it again, no protection change.
void* CompareMove(uint64_t to, uint64_t from, size_t size) {
	constexpr uint64_t Page = 4096;
	// (The images over all of it as before: a GPU-written image's bytes here are stale, equal or not.)
	(void)Libs::LibKernel::Memory::TryPrepareHostImageWrite(to, size);
	uint64_t           run = 0, run_end = 0, same = 0, gpu = 0;
	const auto         write = [&] {
        if (run_end == run) return;
        (void)Libs::LibKernel::Memory::TryPrepareHostOverwrite(to + run, run_end - run);
        std::memmove(reinterpret_cast<void*>(to + run), reinterpret_cast<const void*>(from + run), run_end - run);
	};
	for (uint64_t at = 0; at < size;) {
		const auto next  = std::min<uint64_t>(size, ((to + at) / Page + 1) * Page - to);
		const auto bytes = next - at;
		bool       differs = Libs::LibKernel::Memory::ReadFaults(to + at, bytes);
		if (differs) gpu += bytes;
		else differs = std::memcmp(reinterpret_cast<const void*>(to + at), reinterpret_cast<const void*>(from + at), bytes) != 0;
		if (!differs) {
			same += bytes;
		} else {
			if (run_end != at) {
				write();
				run = at;
			}
			run_end = next;
		}
		at = next;
	}
	write();
	LiveCounters::Add(LiveCounters::GuestMoveBytes, size);
	LiveCounters::Add(LiveCounters::GuestMoveSameBytes, same);
	LiveCounters::Add(LiveCounters::GuestMoveGpuBytes, gpu);
	return reinterpret_cast<void*>(to);
}

void* KYTY_SYSV_ABI     CoherentMove(void* destination, const void* source, size_t size) {
	// KYTY_TEXTURE_PROBE: reached by a jump with the guest caller's return address on top (Tailcall pops the entry's
	// saved RBP), so this is the eboot call site's. Measurement only (demonsSoulsTextureProbe.h).
	DemonsSoulsTextureProbe::OnCopy(reinterpret_cast<uint64_t>(__builtin_return_address(0)), destination, size);
	// (etapa2.1b) KYTY_TEX_DIRECTA: el memcpy de eboot+0x901459 va por el stub PLT (eboot+0x1e73d00) al memcpy del
	// libc.prx del juego, cuyo cuerpo sustituye este CoherentMove: LibcInternal::memcpy (libC.cpp) no lo ve nunca.
	// (N2+N3) KYTY_TEX_NATIVA: un trozo de textura nativa va directo a la GPU y NO se escribe en la RAM emulada.
	if (::TexDirecta::NativeChunkCopy(destination, source, size)) return destination;
	::TexDirecta::OnMemcpy(destination, source, size);
	// (audit1) KYTY_DS_WRITE_AUDIT: the copy this thread makes, known to the audit until it returns (one load when off).
	// (policy1: and its source, for the "[read-audit]" table.)
	const DemonsSoulsWriteAudit::CopyScope audit_scope(reinterpret_cast<uint64_t>(__builtin_return_address(0)), destination, size,
	                                                   source);
	// A move within one range reads bytes it then writes: their GPU data is read back first. Else the destination's is
	// replaced whole and dropped (Memory::InvalidateMemoryForWrite).
	const auto to = reinterpret_cast<uint64_t>(destination), from = reinterpret_cast<uint64_t>(source);
	const bool overlap = to < from + size && from < to + size;
	if (!overlap && size >= 256 * 1024 && kyty_local_move_compare.load(std::memory_order_relaxed) != 0)
		return CompareMove(to, from, size);
	if (size >= 256 * 1024) LiveCounters::Add(LiveCounters::GuestMoveBytes, size);
	return Move(destination, source, size,
	            overlap ? &Libs::LibKernel::Memory::TryPrepareHostWrite : &Libs::LibKernel::Memory::TryPrepareHostOverwrite);
}
// libc.prx's memmove/bcopy/memcpy body, shared by the three entries, which jump to it after the guest's
// argument/stack checks with one saved RBP. Hashed whole, not just the prologue, in each build audited; nothing
// else branches into the replaced bytes (the body's size jump table neither). Other builds, relocations or
// platform loader patches fail closed.
struct Body {
	const char* build;
	uint64_t    offset, size, hash;
};
constexpr std::array<Body, 2> Bodies {{
    {"01.007.000", 0x3c36, 2048, 0xe654325b8be848a9ull},
    {"01.005.000", 0x3d06, 1705, 0xc4be692ee02e9f77ull}, // an older body (another small-size table), the same entries
}};

// --- (render4) KYTY_DS_LINEAR_COPY_HOOK=1 (off by default; local/ds-linear-copy-hook.h) ------------------------------
//
// 01.007.000's linear copy, eboot+0x912e60 (ctx, dst, dst dwords, src, src dwords): it binds cs_memset32, makes three
// temporary buffers (src, dst and a {dst dwords, src dwords} block at the top of the command buffer) and dispatches
// (dst dwords + 63) / 64 groups; some 650 calls a frame (the bone matrices of every animated skeleton, blend shape
// weights, dynamic meshes, 4 MiB of material parameters). Its entry jumps to LinearCopyHook, which writes the
// announcement packet at the command buffer's cursor, runs the game's function whole (through a copy of the 12 bytes
// the jump replaced) and then seals the announcement if the packets the function wrote are exactly the ones a plain
// copy writes. The game's state and its packets are what they are without the hook; only the announcement is added.
struct AgcCommandBuffer { // (libs/agc.cpp's CommandBuffer: sce::Agc::CommandBuffer)
	uint32_t* bottom;
	uint32_t* top;
	uint32_t* cursor_up;
	uint32_t* cursor_down;
	void*     callback;
	void*     user_data;
	uint32_t  reserved_dw;
};
using LinearCopyFunction = KYTY_SYSV_ABI void (*)(uint64_t*, uint64_t, int32_t, uint64_t, int32_t);
LinearCopyFunction g_linear_copy_original   = nullptr; // (the trampoline; set before the entry is patched)
uint64_t           g_linear_copy_trampoline = 0;

constexpr uint64_t LinearCopyOffset = 0x912e60;
// The function as 01.007.000 has it, whole (to its ud2): compared before patching.
constexpr std::array<uint8_t, 807> LinearCopyCode {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x98, 0x01, 0x00, 0x00,
    0x48, 0x8b, 0x05, 0x45, 0xf4, 0xdb, 0x01, 0x44, 0x89, 0x85, 0x54, 0xfe, 0xff, 0xff, 0x49, 0x89, 0xcd, 0x89, 0xd3, 0x49,
    0x89, 0xf4, 0x49, 0x89, 0xfe, 0x48, 0x8b, 0x00, 0x48, 0x89, 0x45, 0xd0, 0xf6, 0x47, 0x08, 0x01, 0x75, 0x07, 0x41, 0x83,
    0x7e, 0x18, 0x20, 0x74, 0x08, 0x41, 0xc7, 0x46, 0x18, 0x20, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x05, 0x48, 0x77, 0x5f, 0x02,
    0x48, 0x8b, 0x00, 0x48, 0x8b, 0x80, 0x50, 0x09, 0x00, 0x00, 0x80, 0xa0, 0xa3, 0x00, 0x00, 0x00, 0xfd, 0x48, 0x8b, 0x40,
    0x20, 0x48, 0x8b, 0x40, 0x68, 0x4c, 0x8b, 0x78, 0x10, 0x41, 0x80, 0x7f, 0x6c, 0x0b, 0x74, 0x29, 0x4c, 0x89, 0xff, 0xe8,
    0x14, 0xe5, 0xf9, 0xff, 0x84, 0xc0, 0x75, 0x1d, 0x41, 0x83, 0x7f, 0x68, 0x00, 0x75, 0x16, 0x49, 0x8b, 0x7f, 0x28, 0x48,
    0x8b, 0x07, 0xff, 0x50, 0x68, 0x48, 0x8b, 0x08, 0x48, 0x89, 0xc7, 0x4c, 0x89, 0xfe, 0xff, 0x51, 0x10, 0x41, 0xc6, 0x47,
    0x6d, 0x01, 0x41, 0x80, 0xa7, 0xa3, 0x00, 0x00, 0x00, 0xfd, 0x49, 0x8b, 0x47, 0x20, 0x49, 0x8b, 0x3e, 0x4c, 0x8b, 0x78,
    0x60, 0x49, 0x8b, 0x37, 0xe8, 0x73, 0xdb, 0x3f, 0x01, 0x4d, 0x89, 0xbe, 0xa8, 0x01, 0x00, 0x00, 0x49, 0x8b, 0x07, 0x4c,
    0x8d, 0xbd, 0xa8, 0xfe, 0xff, 0xff, 0x4c, 0x89, 0xff, 0x49, 0x89, 0x86, 0xc0, 0x01, 0x00, 0x00, 0xe8, 0x63, 0x64, 0xff,
    0xff, 0xbf, 0x04, 0x00, 0x00, 0x00, 0xe8, 0x09, 0x6a, 0xff, 0xff, 0x8b, 0x8d, 0x54, 0xfe, 0xff, 0xff, 0x4c, 0x89, 0xff,
    0xbe, 0x04, 0x00, 0x00, 0x00, 0x89, 0xc2, 0x41, 0xb8, 0x00, 0x08, 0x00, 0x00, 0x4d, 0x89, 0xe9, 0x48, 0xc7, 0x04, 0x24,
    0x00, 0x00, 0x00, 0x00, 0xe8, 0xf3, 0x64, 0xff, 0xff, 0xc5, 0xf8, 0x10, 0x85, 0xa8, 0xfe, 0xff, 0xff, 0x45, 0x31, 0xed,
    0x31, 0xf6, 0xc5, 0xf8, 0x11, 0x85, 0xd0, 0xfe, 0xff, 0xff, 0x49, 0x8b, 0x3e, 0x48, 0x83, 0xc7, 0x38, 0xe8, 0x62, 0xbe,
    0x3f, 0x01, 0x48, 0x8d, 0x8d, 0xd0, 0xfe, 0xff, 0xff, 0x48, 0x89, 0xc7, 0xba, 0x01, 0x00, 0x00, 0x00, 0x31, 0xf6, 0xe8,
    0x9c, 0xa7, 0x3f, 0x01, 0x4c, 0x8d, 0xbd, 0x80, 0xfe, 0xff, 0xff, 0x4c, 0x89, 0xff, 0xe8, 0xed, 0x63, 0xff, 0xff, 0xbf,
    0x04, 0x00, 0x00, 0x00, 0xe8, 0x93, 0x69, 0xff, 0xff, 0x4c, 0x89, 0xff, 0xbe, 0x04, 0x00, 0x00, 0x00, 0x89, 0xc2, 0x89,
    0xd9, 0x41, 0xb8, 0x00, 0x08, 0x00, 0x00, 0x4d, 0x89, 0xe1, 0x48, 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0xe8, 0x81,
    0x64, 0xff, 0xff, 0xc5, 0xf8, 0x10, 0x85, 0x80, 0xfe, 0xff, 0xff, 0x31, 0xf6, 0xc5, 0xf8, 0x11, 0x85, 0xd0, 0xfe, 0xff,
    0xff, 0x49, 0x8b, 0x3e, 0x48, 0x83, 0xc7, 0x38, 0xe8, 0xf3, 0xbd, 0x3f, 0x01, 0x48, 0x8d, 0x8d, 0xd0, 0xfe, 0xff, 0xff,
    0x48, 0x89, 0xc7, 0xba, 0x01, 0x00, 0x00, 0x00, 0x31, 0xf6, 0xe8, 0x2d, 0xa8, 0x3f, 0x01, 0x4d, 0x8b, 0x26, 0x49, 0x8b,
    0x44, 0x24, 0x18, 0x41, 0x8b, 0x74, 0x24, 0x30, 0x48, 0x89, 0xc1, 0x49, 0x2b, 0x4c, 0x24, 0x10, 0x48, 0xc1, 0xf9, 0x02,
    0x48, 0x29, 0xf1, 0x4c, 0x0f, 0x43, 0xe9, 0x49, 0x83, 0xfd, 0x01, 0x77, 0x3f, 0x83, 0xfe, 0xfd, 0x77, 0x4c, 0x49, 0x8b,
    0x54, 0x24, 0x28, 0x4c, 0x89, 0xe7, 0x83, 0xc6, 0x02, 0x41, 0xff, 0x54, 0x24, 0x20, 0x84, 0xc0, 0x74, 0x38, 0x49, 0x8b,
    0x44, 0x24, 0x18, 0x41, 0x8b, 0x54, 0x24, 0x30, 0x45, 0x31, 0xed, 0x48, 0x89, 0xc1, 0x49, 0x2b, 0x4c, 0x24, 0x10, 0x48,
    0xc1, 0xf9, 0x02, 0x48, 0x29, 0xd1, 0x49, 0x0f, 0x42, 0xcd, 0x48, 0x83, 0xf9, 0x02, 0x72, 0x15, 0x48, 0x83, 0xc0, 0xf8,
    0x48, 0x83, 0xe0, 0xfc, 0x49, 0x89, 0xc5, 0x49, 0x89, 0x44, 0x24, 0x18, 0xeb, 0x03, 0x45, 0x31, 0xed, 0x8b, 0x85, 0x54,
    0xfe, 0xff, 0xff, 0x4c, 0x8d, 0xbd, 0x58, 0xfe, 0xff, 0xff, 0x41, 0x89, 0x5d, 0x00, 0x4c, 0x89, 0xff, 0x41, 0x89, 0x45,
    0x04, 0xe8, 0xf6, 0x62, 0xff, 0xff, 0x4c, 0x89, 0xff, 0xbe, 0x1e, 0x00, 0x00, 0x00, 0xba, 0x10, 0x00, 0x00, 0x00, 0xb9,
    0x01, 0x00, 0x00, 0x00, 0x41, 0xb8, 0x00, 0x08, 0x00, 0x00, 0x4d, 0x89, 0xe9, 0x48, 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00,
    0x00, 0xe8, 0x8e, 0x63, 0xff, 0xff, 0xc5, 0xf8, 0x10, 0x85, 0x58, 0xfe, 0xff, 0xff, 0x31, 0xf6, 0xc5, 0xf8, 0x11, 0x85,
    0xd0, 0xfe, 0xff, 0xff, 0x49, 0x8b, 0x3e, 0x48, 0x83, 0xc7, 0x38, 0xe8, 0x00, 0xbd, 0x3f, 0x01, 0x48, 0x8d, 0x8d, 0xd0,
    0xfe, 0xff, 0xff, 0x48, 0x89, 0xc7, 0xba, 0x01, 0x00, 0x00, 0x00, 0x31, 0xf6, 0xe8, 0x3a, 0xa8, 0x3f, 0x01, 0x4d, 0x8b,
    0x36, 0x8d, 0x43, 0x3f, 0x83, 0xc3, 0x7e, 0x85, 0xc0, 0x0f, 0x49, 0xd8, 0xc1, 0xfb, 0x06, 0x4c, 0x89, 0xf7, 0xe8, 0x91,
    0xda, 0x3f, 0x01, 0x4c, 0x89, 0xf7, 0x89, 0xde, 0xba, 0x01, 0x00, 0x00, 0x00, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x41, 0x89,
    0xc0, 0xe8, 0xca, 0x77, 0x3f, 0x01, 0x4c, 0x89, 0xf7, 0xe8, 0x32, 0xd7, 0x3f, 0x01, 0x4c, 0x89, 0xff, 0xe8, 0x7a, 0x62,
    0xff, 0xff, 0x48, 0x8d, 0xbd, 0x80, 0xfe, 0xff, 0xff, 0xe8, 0x6e, 0x62, 0xff, 0xff, 0x48, 0x8d, 0xbd, 0xa8, 0xfe, 0xff,
    0xff, 0xe8, 0x62, 0x62, 0xff, 0xff, 0x48, 0x8b, 0x05, 0x5b, 0xf1, 0xdb, 0x01, 0x48, 0x8b, 0x00, 0x48, 0x3b, 0x45, 0xd0,
    0x75, 0x12, 0x48, 0x81, 0xc4, 0x98, 0x01, 0x00, 0x00, 0x5b, 0x41, 0x5c, 0x41, 0x5d, 0x41, 0x5e, 0x41, 0x5f, 0x5d, 0xc3,
    0xe8, 0xbb, 0x0a, 0x56, 0x01, 0x0f, 0x0b};
// The 12 bytes the jump replaces: push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12 (no jump lands
// inside them and none of them is position-dependent). The next instruction is push rbx, at +12.
constexpr size_t LinearCopyMoved = 12;

// The copy the dispatch's register check takes (DemonsSouls::LinearCopyOf): a plain copy of whole dwords.
bool PlainLinearCopy(uint64_t dst, int32_t dst_dwords, uint64_t src, int32_t src_dwords) {
	if (dst_dwords <= 0 || src_dwords < dst_dwords) return false;
	const uint64_t bytes = static_cast<uint64_t>(dst_dwords) * 4;
	return dst != 0 && src != 0 && ((dst | src) & 3) == 0 && ((dst | src) >> 48u) == 0 && bytes <= 16 * 1024 * 1024 &&
	       !(src < dst + bytes && dst < src + bytes);
}

// The announcement at the cursor, not sealed (a plain NOP until SealLinearCopy). Null: no room without the buffer's
// callback (the game's function asks it for room as it always did).
uint32_t* AnnounceLinearCopy(AgcCommandBuffer& commands) {
	uint32_t* const up   = commands.cursor_up;
	uint32_t* const down = commands.cursor_down;
	if (up == nullptr || down == nullptr || reinterpret_cast<uintptr_t>(down) <= reinterpret_cast<uintptr_t>(up))
		return nullptr;
	const uint64_t available = (reinterpret_cast<uintptr_t>(down) - reinterpret_cast<uintptr_t>(up)) / sizeof(uint32_t);
	if (available <= commands.reserved_dw || available - commands.reserved_dw < DsLinearCopyHook::PacketDwords)
		return nullptr;
	up[0] = DsLinearCopyHook::Header;
	up[1] = DsLinearCopyHook::Magic0;
	up[2] = DsLinearCopyHook::Magic1;
	for (uint32_t i = 3; i < DsLinearCopyHook::PacketDwords; ++i) up[i] = 0;
	commands.cursor_up = up + DsLinearCopyHook::PacketDwords;
	return up;
}

// The packets the game's function wrote after the announcement, when they are a plain copy's and say what the
// function was asked: register writes of the shader stage only (the shader's own, and the 12 user data registers
// with the source, destination and parameter descriptors LinearCopyOf accepts) and, last, the dispatch of the
// groups. Then the announcement is sealed. False: left a plain NOP (`why`: what was found). `down_before`: the
// buffer's downward cursor before the game's function ran.
bool SealLinearCopy(const AgcCommandBuffer& commands, uint32_t* packet, uintptr_t down_before, uint64_t dst,
                    int32_t dst_dwords, uint64_t src, int32_t src_dwords, const char** why) {
	namespace H = DsLinearCopyHook;
	const uint32_t* const begin      = packet + H::PacketDwords;
	const auto            begin_addr = reinterpret_cast<uintptr_t>(begin);
	const auto            end_addr   = reinterpret_cast<uintptr_t>(commands.cursor_up);
	if (end_addr <= begin_addr || (end_addr - begin_addr) % sizeof(uint32_t) != 0 ||
	    (end_addr - begin_addr) / sizeof(uint32_t) > H::MaxSpanDwords)
		return *why = "the cursor is not just after its packets (another buffer?)", false;
	const auto     span      = static_cast<uint32_t>((end_addr - begin_addr) / sizeof(uint32_t));
	const auto     count     = static_cast<uint32_t>(dst_dwords);
	const uint32_t groups    = (count + 63u) / 64u;
	bool           user_data = false, dispatch = false;
	for (uint32_t at = 0; at < span;) {
		const uint32_t header = begin[at];
		if ((header >> 30u) != 3u || (header & 1u) != 0) return *why = "a packet that is not a plain type-3 one", false;
		const uint32_t length = ((header >> 16u) & 0x3fffu) + 2u;
		if (length > span - at) return *why = "a packet past the cursor", false;
		const uint32_t* const words = begin + at;
		switch ((header >> 8u) & 0xffu) {
			case 0x63: break; // IT_SET_SH_REG_INDIRECT: the shader's registers
			case 0x76: { // IT_SET_SH_REG
				if (length != 14 || words[1] != 0x240u) break; // (not COMPUTE_USER_DATA_0..11: other shader registers)
				const uint32_t* const source      = words + 2;
				const uint32_t* const destination = words + 6;
				const uint32_t* const parameters  = words + 10;
				const auto base48 = [](const uint32_t* d) { return (d[0] | uint64_t {d[1]} << 32u) & 0xFFFFFFFFFFFFull; };
				for (const auto* d: {source, destination})
					if (d[3] != 0x00014004u || (d[1] & 0xffff0000u) != 0x00040000u)
						return *why = "a buffer descriptor of another format", false;
				if (base48(source) != src || base48(destination) != dst || source[2] < count || destination[2] < count)
					return *why = "buffer descriptors of other ranges", false;
				const uint64_t block = base48(parameters);
				if (parameters[3] != 0x0004dfacu || (parameters[1] & 0xffff0000u) != 0x00100000u || parameters[2] != 1 ||
				    block == 0 || (block & 3) != 0)
					return *why = "a parameter descriptor of another format", false;
				// (The block the function just took from the top of this command buffer, between the downward cursor now
				// and before the call: host-readable here.)
				const auto block_addr = static_cast<uintptr_t>(block);
				if (block_addr < reinterpret_cast<uintptr_t>(commands.cursor_down) ||
				    block_addr + 2 * sizeof(uint32_t) > down_before)
					return *why = "a parameter block not taken from the command buffer in this call", false;
				uint32_t values[2];
				std::memcpy(values, reinterpret_cast<const void*>(block_addr), sizeof(values));
				if (values[0] != count || values[1] != static_cast<uint32_t>(src_dwords)) return *why = "other parameters", false;
				user_data = true;
				break;
			}
			case 0x15: // IT_DISPATCH_DIRECT: the last packet, (count + 63) / 64 x 1 x 1 groups, initiator 0x41
				if (at + length != span || length != 5 || words[1] != groups || words[2] != 1 || words[3] != 1 ||
				    words[4] != 0x41u)
					return *why = "another dispatch", false;
				dispatch = true;
				break;
			default: return *why = "a packet of another kind", false;
		}
		at += length;
	}
	if (!user_data || !dispatch) return *why = "no user data registers or no dispatch", false;
	const uint64_t hash   = H::SpanHash(begin, span);
	packet[H::DstWord]      = static_cast<uint32_t>(dst);
	packet[H::DstWord + 1]  = static_cast<uint32_t>(dst >> 32u);
	packet[H::SrcWord]      = static_cast<uint32_t>(src);
	packet[H::SrcWord + 1]  = static_cast<uint32_t>(src >> 32u);
	packet[H::CountWord]    = count;
	packet[H::PeriodWord]   = static_cast<uint32_t>(src_dwords);
	packet[H::HashWord]     = static_cast<uint32_t>(hash);
	packet[H::HashWord + 1] = static_cast<uint32_t>(hash >> 32u);
	packet[H::SpanWord]     = span; // (last: the packet is sealed)
	return true;
}

// The game's linear copy function's entry jumps here (a guest thread recording a command buffer).
void KYTY_SYSV_ABI LinearCopyHook(uint64_t* context, uint64_t dst, int32_t dst_dwords, uint64_t src, int32_t src_dwords) {
	namespace H = DsLinearCopyHook;
	// (audit1) KYTY_DS_WRITE_AUDIT counts the copy; without KYTY_DS_LINEAR_COPY_HOOK the game's function then runs as
	// it is: nothing announced, none of this hook's counters moved.
	if (DemonsSoulsWriteAudit::On()) DemonsSoulsWriteAudit::OnLinearCopy(dst, dst_dwords, src, src_dwords);
	if (H::Mode() == 0) {
		g_linear_copy_original(context, dst, dst_dwords, src, src_dwords);
		return;
	}
	H::Count(H::Calls);
	auto* const commands = context != nullptr ? reinterpret_cast<AgcCommandBuffer*>(context[0]) : nullptr;
	uint32_t*   packet   = nullptr;
	uintptr_t   down_before = 0;
	if (commands != nullptr) {
		if (!PlainLinearCopy(dst, dst_dwords, src, src_dwords)) {
			H::Count(H::NotPlain);
		} else {
			packet = AnnounceLinearCopy(*commands);
			if (packet == nullptr) H::Count(H::NoRoom);
			else down_before = reinterpret_cast<uintptr_t>(commands->cursor_down);
		}
	}
	{
		// (port2) No marker after the copy's dispatch: SealLinearCopy wants the function's own packets, the dispatch last.
		const Port2::Suppress no_port2_marker;
		g_linear_copy_original(context, dst, dst_dwords, src, src_dwords);
	}
	if (packet == nullptr) return;
	const char* why = nullptr;
	if (reinterpret_cast<AgcCommandBuffer*>(context[0]) != commands) why = "another command buffer";
	if (why == nullptr && SealLinearCopy(*commands, packet, down_before, dst, dst_dwords, src, src_dwords, &why)) {
		H::Count(H::Sealed);
		return;
	}
	H::Count(H::OtherPackets);
	static std::atomic<uint32_t> printed {0};
	if (printed.fetch_add(1, std::memory_order_relaxed) < 8) {
		// (With the first dwords the function wrote after the announcement, when the cursor is still just after them.)
		char       words[16 * 9 + 1] = "";
		const auto begin_addr        = reinterpret_cast<uintptr_t>(packet + H::PacketDwords);
		const auto end_addr          = reinterpret_cast<uintptr_t>(commands->cursor_up);
		if (reinterpret_cast<AgcCommandBuffer*>(context[0]) == commands && end_addr > begin_addr &&
		    (end_addr - begin_addr) / sizeof(uint32_t) <= H::MaxSpanDwords) {
			const auto count = (std::min<size_t>)((end_addr - begin_addr) / sizeof(uint32_t), 16);
			for (size_t i = 0; i < count; ++i)
				std::snprintf(words + i * 9, 10, " %08x", static_cast<unsigned>(packet[H::PacketDwords + i]));
		}
		std::printf("[ds-copy-hook] not sealed: %s (dst 0x%llx, %d dwords, src 0x%llx, %d dwords);%s\n",
		            why != nullptr ? why : "?", static_cast<unsigned long long>(dst), dst_dwords,
		            static_cast<unsigned long long>(src), src_dwords, words);
		std::fflush(stdout);
	}
}

void InstallLinearCopyHook(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	// (audit1) Also for KYTY_DS_WRITE_AUDIT=1 alone: the same entry patch, the hook then only counts.
	const bool for_audit = DemonsSoulsWriteAudit::WantsLinearCopy();
	if ((DsLinearCopyHook::Mode() == 0 && !for_audit) || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto refuse = [](const char* why) {
		std::printf("Demon's Souls linear copy hook: NOT installed (KYTY_DS_LINEAR_COPY_HOOK): %s; the game's linear copies "
		            "stay as they were\n",
		            why);
		std::fflush(stdout);
	};
	if (program->mapped_size < LinearCopyOffset + LinearCopyCode.size() ||
	    program->base_vaddr > UINT64_MAX - LinearCopyOffset - LinearCopyCode.size())
		return refuse("the eboot is too short to be 01.007.000's");
	const auto  address = program->base_vaddr + LinearCopyOffset;
	const auto* code    = reinterpret_cast<const uint8_t*>(address);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, LinearCopyCode.size())) return refuse("its code is not readable");
	for (size_t i = 0; i < LinearCopyCode.size(); ++i) {
		if (code[i] == LinearCopyCode[i]) continue;
		std::printf("Demon's Souls linear copy hook: eboot+0x%llx holds 0x%02x where 01.007.000 has 0x%02x\n",
		            static_cast<unsigned long long>(LinearCopyOffset + i), static_cast<unsigned>(code[i]),
		            static_cast<unsigned>(LinearCopyCode[i]));
		return refuse("eboot+0x912e60 is not 01.007.000's linear copy function (another build, or code already patched)");
	}
	// The trampoline: the moved bytes, then mov rax, <entry + moved>; jmp rax (rax is dead there: the function loads it
	// at +0x14 before any read of it).
	if (g_linear_copy_trampoline == 0)
		g_linear_copy_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_linear_copy");
	if (g_linear_copy_trampoline == 0) return refuse("no memory for its trampoline");
	std::array<uint8_t, LinearCopyMoved + 12> trampoline {};
	std::memcpy(trampoline.data(), LinearCopyCode.data(), LinearCopyMoved);
	const uint64_t resume           = address + LinearCopyMoved;
	trampoline[LinearCopyMoved]     = 0x48;
	trampoline[LinearCopyMoved + 1] = 0xb8;
	std::memcpy(trampoline.data() + LinearCopyMoved + 2, &resume, sizeof(resume));
	trampoline[LinearCopyMoved + 10] = 0xff;
	trampoline[LinearCopyMoved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_linear_copy_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_linear_copy_trampoline, trampoline.size()))
		return refuse("its trampoline could not be flushed");
	g_linear_copy_original = reinterpret_cast<LinearCopyFunction>(g_linear_copy_trampoline);
	// The entry: mov rax, LinearCopyHook; jmp rax (12 bytes, the moved ones). Installation occurs before guest threads
	// run; program memory is writable to the host.
	std::array<uint8_t, LinearCopyMoved> entry {0x48, 0xb8};
	const auto                           hook = reinterpret_cast<uint64_t>(&LinearCopyHook);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), LinearCopyCode.data(), LinearCopyMoved);
		Common::VirtualMemory::FlushInstructionCache(address, LinearCopyMoved);
		return refuse("its entry could not be flushed");
	}
	if (for_audit) DemonsSoulsWriteAudit::NoteLinearCopyHook(true);
	if (DsLinearCopyHook::Mode() == 0) {
		std::printf("Demon's Souls linear copy hook: installed for the write audit only (KYTY_DS_WRITE_AUDIT=1; eboot+0x912e60 of "
		            "01.007.000 is counted: nothing is announced, the game's linear copies run as they were)\n");
		std::fflush(stdout);
		return;
	}
	std::printf("Demon's Souls linear copy hook: installed (KYTY_DS_LINEAR_COPY_HOOK=%u; eboot+0x912e60 of 01.007.000 "
	            "announces each linear copy in the command buffer: %s)\n",
	            DsLinearCopyHook::Mode(),
	            DsLinearCopyHook::Mode() == 2 ? "verify only, nothing is passed over"
	                                          : "the command processor copies and passes over the game's packets");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

// --- (render1 F0+F1) KYTY_RENDER1=1 (off by default; local/render1.h) ----------------------------------------------
//
// 01.007.000's mesh draw loop, eboot+0xad8bc0 (obj, visible mask, pass descriptor, draw context; SysV: rdi, rsi, rdx,
// rcx; called from 0xa1f290). Its entry jumps to DrawLoopHook, which reads what the loop will draw (Render1::Read: the
// same reads, nothing written), notes the command buffer's cursor, runs the game's function whole through a copy of
// the 12 bytes the jump replaced, and registers the markers with the range the function's packets went to. The
// game's state, its packets and the image are what they are without the hook.
using DrawLoopFunction = KYTY_SYSV_ABI void (*)(uint64_t, const uint64_t*, uint64_t, uint64_t*);
DrawLoopFunction g_draw_loop_original   = nullptr;
uint64_t         g_draw_loop_trampoline = 0;

constexpr uint64_t DrawLoopOffset = 0xad8bc0;
// The function as 01.007.000 has it, whole (to its ud2 at +0x417): compared before patching.
constexpr std::array<uint8_t, 1049> DrawLoopCode {
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x98, 0x00, 0x00, 0x00,
    0x4c, 0x8b, 0x05, 0xe5, 0x96, 0xbf, 0x01, 0x49, 0x8b, 0x00, 0x48, 0x89, 0x45, 0xd0, 0x44, 0x0f, 0xb7, 0x4f, 0x64, 0x45,
    0x85, 0xc9, 0x0f, 0x84, 0xc7, 0x03, 0x00, 0x00, 0x45, 0x8d, 0x51, 0x3f, 0x4c, 0x8b, 0x5f, 0x50, 0x48, 0x89, 0xcb, 0x49,
    0x89, 0xf6, 0x48, 0xc7, 0xc0, 0xc1, 0xff, 0xff, 0xff, 0x49, 0x89, 0xf7, 0x41, 0xc1, 0xea, 0x06, 0x44, 0x89, 0xd1, 0xc1,
    0xe1, 0x06, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x8b, 0x37, 0x48,
    0x85, 0xf6, 0x75, 0x18, 0x48, 0x8d, 0x74, 0x01, 0xc0, 0x48, 0x83, 0xc0, 0xc0, 0x49, 0x83, 0xc7, 0x08, 0x48, 0x83, 0xfe,
    0xc1, 0x75, 0xe5, 0xe9, 0x77, 0x03, 0x00, 0x00, 0xc4, 0xe2, 0xf0, 0xf3, 0xde, 0xf3, 0x4c, 0x0f, 0xbd, 0xe1, 0x49, 0x01,
    0xc4, 0x49, 0xf7, 0xdc, 0x4d, 0x39, 0xcc, 0x0f, 0x83, 0x5e, 0x03, 0x00, 0x00, 0x49, 0x89, 0xff, 0x48, 0x8d, 0x7d, 0xa8,
    0x4c, 0x89, 0x4d, 0x80, 0x49, 0x89, 0xd5, 0x4c, 0x89, 0x95, 0x78, 0xff, 0xff, 0xff, 0x4c, 0x89, 0x5d, 0x90, 0xe8, 0x29,
    0x07, 0xe3, 0xff, 0x4d, 0x8b, 0x4f, 0x40, 0x48, 0x8d, 0x7d, 0xa8, 0xbe, 0x1c, 0x00, 0x00, 0x00, 0xba, 0x04, 0x00, 0x00,
    0x00, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x41, 0xb8, 0x00, 0x08, 0x00, 0x00, 0x4c, 0x89, 0xbd, 0x58, 0xff, 0xff, 0xff, 0x48,
    0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0x4d, 0x89, 0xef, 0xe8, 0xb5, 0x07, 0xe3, 0xff, 0x48, 0x8b, 0x03, 0x8b, 0x75,
    0xa8, 0x48, 0x8b, 0x38, 0x0f, 0xb7, 0x45, 0xac, 0x48, 0xc1, 0xe0, 0x20, 0x48, 0x09, 0xc6, 0xe8, 0xfc, 0x24, 0x23, 0x01,
    0x4c, 0x8b, 0x55, 0x90, 0x48, 0x8b, 0xb5, 0x78, 0xff, 0xff, 0xff, 0xb8, 0xff, 0xff, 0xff, 0xff, 0x45, 0x31, 0xdb, 0x45,
    0x31, 0xc0, 0xc7, 0x45, 0xa4, 0xff, 0xff, 0xff, 0xff, 0x48, 0x89, 0x9d, 0x60, 0xff, 0xff, 0xff, 0x4c, 0x89, 0xad, 0x68,
    0xff, 0xff, 0xff, 0x4c, 0x89, 0xb5, 0x50, 0xff, 0xff, 0xff, 0x49, 0x8b, 0x4a, 0x20, 0x41, 0x8b, 0x57, 0x20, 0x42, 0x8b,
    0x0c, 0xa1, 0x0f, 0xa3, 0xd1, 0x0f, 0x83, 0x33, 0x02, 0x00, 0x00, 0x4c, 0x8b, 0xad, 0x58, 0xff, 0xff, 0xff, 0x41, 0x0f,
    0xb7, 0x4d, 0x60, 0x41, 0x39, 0xc8, 0x73, 0x44, 0x49, 0x8b, 0x32, 0x44, 0x89, 0xc2, 0x0f, 0xb7, 0x3c, 0x56, 0x49, 0x39,
    0xfc, 0x72, 0x35, 0x89, 0xcf, 0x48, 0x8d, 0x74, 0x56, 0x02, 0x48, 0xf7, 0xdf, 0x4c, 0x8d, 0x44, 0x3a, 0x01, 0x31, 0xff,
    0x0f, 0x1f, 0x40, 0x00, 0x4d, 0x89, 0xc1, 0x49, 0x01, 0xf9, 0x74, 0x64, 0x44, 0x0f, 0xb7, 0x0c, 0x7e, 0x48, 0xff, 0xc7,
    0x4d, 0x39, 0xcc, 0x73, 0xeb, 0x01, 0xfa, 0x41, 0x89, 0xd0, 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00, 0x41, 0x0f, 0xb7, 0x4d,
    0x62, 0x4c, 0x89, 0x45, 0x98, 0x41, 0x39, 0xcb, 0x73, 0x58, 0x49, 0x8b, 0x72, 0x08, 0x44, 0x89, 0xda, 0x0f, 0xb7, 0x3c,
    0x56, 0x49, 0x39, 0xfc, 0x72, 0x48, 0x89, 0xcf, 0x48, 0x8d, 0x74, 0x56, 0x02, 0x48, 0xf7, 0xdf, 0x4c, 0x8d, 0x44, 0x3a,
    0x01, 0x31, 0xff, 0x90, 0x4d, 0x89, 0xc1, 0x49, 0x01, 0xf9, 0x74, 0x27, 0x44, 0x0f, 0xb7, 0x0c, 0x7e, 0x48, 0xff, 0xc7,
    0x4d, 0x39, 0xcc, 0x73, 0xeb, 0x01, 0xfa, 0x41, 0x89, 0xd3, 0xeb, 0x16, 0x41, 0x89, 0xc8, 0x41, 0x0f, 0xb7, 0x4d, 0x62,
    0x4c, 0x89, 0x45, 0x98, 0x41, 0x39, 0xcb, 0x72, 0xb1, 0xeb, 0x07, 0x41, 0x89, 0xcb, 0x4c, 0x8b, 0x45, 0x98, 0x4c, 0x89,
    0x9d, 0x70, 0xff, 0xff, 0xff, 0x44, 0x39, 0x45, 0xa4, 0x74, 0x2b, 0x49, 0x8b, 0x42, 0x10, 0x41, 0x8d, 0x48, 0xff, 0x48,
    0x89, 0xde, 0x48, 0x8b, 0x3c, 0xc8, 0xe8, 0xf9, 0xbf, 0xfa, 0xff, 0x4c, 0x8b, 0x9d, 0x70, 0xff, 0xff, 0xff, 0x4c, 0x8b,
    0x55, 0x90, 0xb8, 0xff, 0xff, 0xff, 0xff, 0x48, 0x8b, 0x4d, 0x98, 0x89, 0x4d, 0xa4, 0x48, 0x8b, 0x3b, 0x44, 0x39, 0xd8,
    0x0f, 0x84, 0xe7, 0x00, 0x00, 0x00, 0x4d, 0x8b, 0x7a, 0x18, 0x49, 0x89, 0xde, 0x41, 0x8d, 0x5b, 0xff, 0x48, 0xc1, 0xe3,
    0x05, 0x41, 0x8b, 0x74, 0x1f, 0x18, 0x49, 0x8d, 0x04, 0x1f, 0x48, 0x89, 0x45, 0x88, 0xc1, 0xe6, 0x02, 0x81, 0xe6, 0xfc,
    0xff, 0xff, 0x03, 0x49, 0x03, 0x75, 0x38, 0xe8, 0x98, 0x8b, 0xea, 0xff, 0x41, 0x8b, 0x44, 0x1f, 0x0c, 0x4d, 0x8b, 0x36,
    0x45, 0x8b, 0x6c, 0x1f, 0x08, 0x0f, 0xb7, 0xc8, 0x48, 0xc1, 0xe1, 0x20, 0x49, 0x09, 0xcd, 0x41, 0xf6, 0x46, 0x08, 0x01,
    0x75, 0x0b, 0xb1, 0x01, 0x4d, 0x39, 0xae, 0x98, 0x01, 0x00, 0x00, 0x74, 0x30, 0x48, 0x8b, 0x5d, 0x88, 0x49, 0x8b, 0x3e,
    0x4c, 0x89, 0xee, 0x44, 0x8b, 0x7b, 0x10, 0xe8, 0x7c, 0x23, 0x23, 0x01, 0x49, 0x8b, 0x3e, 0x44, 0x89, 0xfe, 0xe8, 0x81,
    0x23, 0x23, 0x01, 0x4d, 0x89, 0xae, 0x98, 0x01, 0x00, 0x00, 0x41, 0xf6, 0x46, 0x08, 0x01, 0x8b, 0x43, 0x0c, 0x0f, 0x94,
    0xc1, 0x25, 0x00, 0x00, 0xff, 0x3f, 0x45, 0x31, 0xed, 0x3d, 0x00, 0x00, 0x04, 0x00, 0x0f, 0x94, 0xc0, 0x84, 0xc9, 0x74,
    0x09, 0x41, 0x38, 0x86, 0xa0, 0x01, 0x00, 0x00, 0x74, 0x19, 0x49, 0x8b, 0x3e, 0x41, 0x88, 0xc5, 0x31, 0xd2, 0x31, 0xc9,
    0x44, 0x89, 0xee, 0xe8, 0x20, 0x23, 0x23, 0x01, 0x45, 0x88, 0xae, 0xa0, 0x01, 0x00, 0x00, 0x48, 0x8b, 0x9d, 0x60, 0xff,
    0xff, 0xff, 0x4c, 0x8b, 0xbd, 0x68, 0xff, 0xff, 0xff, 0x83, 0xbb, 0x38, 0x01, 0x00, 0x00, 0x02, 0x48, 0x8b, 0x3b, 0x75,
    0x10, 0x48, 0x8b, 0x45, 0x88, 0x0f, 0xb6, 0x70, 0x1b, 0xe8, 0x02, 0xeb, 0xf7, 0xff, 0x48, 0x8b, 0x3b, 0x4c, 0x8b, 0x2f,
    0x42, 0x8d, 0x04, 0xa5, 0x00, 0x00, 0x00, 0x00, 0x44, 0x8d, 0x34, 0x80, 0x4c, 0x89, 0xef, 0xe8, 0x68, 0x7c, 0x23, 0x01,
    0x4c, 0x89, 0xef, 0x44, 0x89, 0xf6, 0x48, 0x89, 0xc2, 0xe8, 0xfa, 0x21, 0x23, 0x01, 0x4c, 0x89, 0xef, 0xe8, 0x22, 0x79,
    0x23, 0x01, 0x4c, 0x8b, 0x9d, 0x70, 0xff, 0xff, 0xff, 0x4c, 0x8b, 0xb5, 0x50, 0xff, 0xff, 0xff, 0x48, 0x8b, 0xb5, 0x78,
    0xff, 0xff, 0xff, 0x4c, 0x8b, 0x55, 0x90, 0x4c, 0x8b, 0x45, 0x98, 0x44, 0x89, 0xd8, 0x44, 0x89, 0xe2, 0x4c, 0x89, 0xe1,
    0x48, 0xc7, 0xc7, 0xff, 0xff, 0xff, 0xff, 0xf6, 0xd2, 0x48, 0xc1, 0xe9, 0x06, 0xc4, 0xe2, 0xeb, 0xf7, 0xd7, 0xc4, 0xc2,
    0xe8, 0xf2, 0x14, 0xce, 0xc4, 0xe2, 0xe8, 0xf3, 0xda, 0x74, 0x0d, 0x49, 0x83, 0xcc, 0x3f, 0xeb, 0x2c, 0x0f, 0x1f, 0x80,
    0x00, 0x00, 0x00, 0x00, 0x49, 0x89, 0xcc, 0x48, 0xff, 0xc1, 0x49, 0xc1, 0xe4, 0x06, 0x49, 0x83, 0xcc, 0x3f, 0x66, 0x90,
    0x48, 0x39, 0xf1, 0x73, 0x22, 0x49, 0x83, 0xc4, 0x40, 0xc4, 0xc2, 0xe8, 0xf3, 0x1c, 0xce, 0x48, 0x8d, 0x49, 0x01, 0x74,
    0xeb, 0xf3, 0x48, 0x0f, 0xbd, 0xca, 0x49, 0x29, 0xcc, 0x4c, 0x3b, 0x65, 0x80, 0x0f, 0x82, 0x4f, 0xfd, 0xff, 0xff, 0x48,
    0x8d, 0x7d, 0xa8, 0xe8, 0x10, 0x04, 0xe3, 0xff, 0x4c, 0x8b, 0x05, 0x09, 0x93, 0xbf, 0x01, 0x49, 0x8b, 0x00, 0x48, 0x3b,
    0x45, 0xd0, 0x75, 0x12, 0x48, 0x81, 0xc4, 0x98, 0x00, 0x00, 0x00, 0x5b, 0x41, 0x5c, 0x41, 0x5d, 0x41, 0x5e, 0x41, 0x5f,
    0x5d, 0xc3, 0xe8, 0x69, 0xac, 0x39, 0x01, 0x0f, 0x0b
};
// The 12 bytes the jump replaces: push rbp; mov rbp, rsp; push r15; push r14; push r13; push r12 (no jump of the
// function lands inside them, none is position-dependent; rax is first written at +0x1b, read after). Next: push rbx.
constexpr size_t DrawLoopMoved = 12;

void KYTY_SYSV_ABI DrawLoopHook(uint64_t obj, const uint64_t* mask, uint64_t pass, uint64_t* context) {
	const uint64_t t0 = __rdtsc();
	Render1::Plan plan;
	const bool    render1 = Render1::On(); // (f5p1 techo: the hook may be installed for it alone)
	if (render1 && obj != 0 && pass != 0) Render1::Read(obj, mask, pass, plan);
	auto* const commands =
	    context != nullptr && context[0] != 0 ? *reinterpret_cast<AgcCommandBuffer* const*>(context[0]) : nullptr;
	uint32_t* const bottom = commands != nullptr ? commands->bottom : nullptr;
	const auto      begin  = commands != nullptr ? reinterpret_cast<uintptr_t>(commands->cursor_up) : uintptr_t {0};
	const uint64_t  t1     = __rdtsc();
	g_draw_loop_original(obj, mask, pass, context);
	const uint64_t t2      = __rdtsc();
	auto* const    after   = context != nullptr && context[0] != 0
	                             ? *reinterpret_cast<AgcCommandBuffer* const*>(context[0])
	                             : nullptr;
	const bool     split   = commands == nullptr || after != commands || commands->bottom != bottom;
	const auto     end     = commands != nullptr ? reinterpret_cast<uintptr_t>(commands->cursor_up) : uintptr_t {0};
	if (render1) Render1::Register(std::move(plan), begin, end, split, (t1 - t0) + (__rdtsc() - t2));
	F5p1Techo::Register(pass, begin, end, split); // (f5p1 techo, KYTY_F5P1_TECHO; one load when off)
}

void InstallDrawLoopHook(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if ((!Render1::On() && !F5p1Techo::On()) || program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    program->file_name.filename() != "eboot.bin")
		return;
	const auto refuse = [](const char* why) {
		std::printf("[render1] hook NOT installed (KYTY_RENDER1=%u): %s; nothing is counted, the game runs as it is\n",
		            Render1::Mode(), why);
		std::fflush(stdout);
	};
	if (program->mapped_size < DrawLoopOffset + DrawLoopCode.size() ||
	    program->base_vaddr > UINT64_MAX - DrawLoopOffset - DrawLoopCode.size())
		return refuse("the eboot is too short to be 01.007.000's");
	const auto  address = program->base_vaddr + DrawLoopOffset;
	const auto* code    = reinterpret_cast<const uint8_t*>(address);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(address, DrawLoopCode.size())) return refuse("its code is not readable");
	for (size_t i = 0; i < DrawLoopCode.size(); ++i) {
		if (code[i] == DrawLoopCode[i]) continue;
		std::printf("[render1] eboot+0x%llx holds 0x%02x where 01.007.000 has 0x%02x\n",
		            static_cast<unsigned long long>(DrawLoopOffset + i), static_cast<unsigned>(code[i]),
		            static_cast<unsigned>(DrawLoopCode[i]));
		return refuse("eboot+0xad8bc0 is not 01.007.000's mesh draw loop (another build, or code already patched)");
	}
	if (g_draw_loop_trampoline == 0)
		g_draw_loop_trampoline = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_render1");
	if (g_draw_loop_trampoline == 0) return refuse("no memory for its trampoline");
	std::array<uint8_t, DrawLoopMoved + 12> trampoline {};
	std::memcpy(trampoline.data(), DrawLoopCode.data(), DrawLoopMoved);
	const uint64_t resume         = address + DrawLoopMoved;
	trampoline[DrawLoopMoved]     = 0x48;
	trampoline[DrawLoopMoved + 1] = 0xb8;
	std::memcpy(trampoline.data() + DrawLoopMoved + 2, &resume, sizeof(resume));
	trampoline[DrawLoopMoved + 10] = 0xff;
	trampoline[DrawLoopMoved + 11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(g_draw_loop_trampoline), trampoline.data(), trampoline.size());
	if (!Common::VirtualMemory::FlushInstructionCache(g_draw_loop_trampoline, trampoline.size()))
		return refuse("its trampoline could not be flushed");
	g_draw_loop_original = reinterpret_cast<DrawLoopFunction>(g_draw_loop_trampoline);
	std::array<uint8_t, DrawLoopMoved> entry {0x48, 0xb8};
	const auto                         hook = reinterpret_cast<uint64_t>(&DrawLoopHook);
	std::memcpy(entry.data() + 2, &hook, sizeof(hook));
	entry[10] = 0xff;
	entry[11] = 0xe0;
	std::memcpy(reinterpret_cast<void*>(address), entry.data(), entry.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, entry.size())) {
		std::memcpy(reinterpret_cast<void*>(address), DrawLoopCode.data(), DrawLoopMoved);
		Common::VirtualMemory::FlushInstructionCache(address, DrawLoopMoved);
		return refuse("its entry could not be flushed");
	}
	Render1::g_installed.store(true);
	std::printf("[render1] hook installed (KYTY_RENDER1=%u; eboot+0xad8bc0 of 01.007.000 observed only: counts and markers, "
	            "nothing replaced%s)\n",
	            Render1::Mode(),
	            Render1::Mode() == 2   ? "; F2: p0/p1 drawn again natively into a depth image of their own and compared, "
	                                     "the image shown is the normal one"
	            : Render1::Mode() == 3 ? (Render1::F3Alternates()
	                                          ? "; F3: p0/p1 drawn ONLY natively into the game's own depth, alternating 5 s "
	                                            "on / 5 s off (normal path), \"[render1] F3\" line every 5 s"
	                                          : "; F3: p0/p1 drawn ONLY natively into the game's own depth, always on "
	                                            "(KYTY_RENDER1_ALTERNA=0)")
	                                   : "");
	std::fflush(stdout);
#else
	(void)program;
#endif
}
// --- (port2) KYTY_PORT2=1|2 (off by default; local/port2.h) ---------------------------------------------------------
//
// Nothing of the game's code is patched: the flow is recorded in the emulator's Agc, where the game's wrappers jump to.
// What it relies on is the binder's and the wrappers' behaviour (the user data hole reserved through
// SetShRegisterRangeDirect and filled up to the draw; the "emitted" flags cleared after it; each draw / dispatch a
// tail-jump to its import): those functions of 01.007.000 are compared here, by a hash of their bytes, before any guest
// thread runs. Other bytes: port2 stays off and says so.
void InstallPort2Check(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (Port2::Mode() == 0 || program == nullptr || program->file_name.filename() != "eboot.bin") return;
	const auto off = [](const char* why) {
		Port2::g_eboot.store(-1, std::memory_order_relaxed);
		std::printf("[port2] DESACTIVADO (KYTY_PORT2=%u): %s; nothing recorded, the PM4 is walked as always\n", Port2::Mode(), why);
		std::fflush(stdout);
	};
	if (!Libs::Graphics::DemonsSouls::IsSupportedGame()) return off("not Demon's Souls");
	for (const auto& f: Port2::EbootFunctions) {
		if (program->mapped_size < f.offset + f.size || program->base_vaddr > UINT64_MAX - f.offset - f.size)
			return off("the eboot is too short to be 01.007.000's");
		const auto address = program->base_vaddr + f.offset;
		if (!Libs::Graphics::HostMemoryRangeIsReadable(address, f.size)) return off("the binder's code is not readable");
		const uint64_t h = Port2::Fnv1a64(reinterpret_cast<const uint8_t*>(address), f.size);
		if (h != f.fnv) {
			std::printf("[port2] eboot+0x%llx (%s, %u bytes): hash 0x%016llx, 01.007.000 has 0x%016llx\n",
			            static_cast<unsigned long long>(f.offset), f.what, f.size, static_cast<unsigned long long>(h),
			            static_cast<unsigned long long>(f.fnv));
			return off("the eboot is not 01.007.000 (or its Agc binder was patched)");
		}
	}
	Port2::g_eboot.store(1, std::memory_order_relaxed);
	std::printf("[port2] eboot 01.007.000: Agc binder and draw / dispatch wrappers checked (%u functions); KYTY_PORT2=%u %s\n",
	            static_cast<unsigned>(sizeof(Port2::EbootFunctions) / sizeof(Port2::EbootFunctions[0])), Port2::Mode(),
	            Port2::Mode() == 2 ? "(verification)" : "(on, 5 s ON / 5 s OFF)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

// --- (BUF-NATIVO) KYTY_BUF_NATIVO=1 (por defecto; =0 lo apaga): local/buf-nativo.h ------------------------------
//
// 01.007.000: 0xaa06a0 (malla, &bloque, &indices, ...: crea los CBuffer de la malla) y 0xaa1d10 (destructor de la malla).
// Cada entrada salta (mov r11, gancho; jmp r11: 13 bytes) a un gancho que llama a la funcion del juego entera por un
// trampolin con los 13 bytes movidos (sin direcciones relativas a RIP) y vuelve a la instruccion siguiente con r11 (libre
// en SysV: ni argumento ni preservado; la funcion no lo ha usado aun en esos 13 bytes). Comprobados byte a byte antes de
// que corra ningun hilo del juego; otros bytes: no se instala y lo dice.
constexpr uint64_t MeshBuildOffset = 0xaa06a0, MeshFreeOffset = 0xaa1d10, MeshCopyReturn = 0x909550;
constexpr size_t   MeshMoved       = 13;
// push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx
constexpr std::array<uint8_t, MeshMoved> MeshBuildCode {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53};
// push rbp; mov rbp,rsp; push rbx; push rax; mov rsi,[rdi+0x298]
constexpr std::array<uint8_t, MeshMoved> MeshFreeCode {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x8b, 0xb7, 0x98, 0x02, 0x00, 0x00};
// (y lo que sigue, para no instalar sobre otra build: el memcpy de CBuffer::Create, call rel32 justo antes de 0x909550)
using MeshBuildFunction = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using MeshFreeFunction  = KYTY_SYSV_ABI void (*)(uint64_t);
MeshBuildFunction g_mesh_build_original = nullptr;
MeshFreeFunction  g_mesh_free_original  = nullptr;

uint64_t KYTY_SYSV_ABI MeshBuildHook(uint64_t mesh, uint64_t block, uint64_t indices, uint64_t a3, uint64_t a4, uint64_t a5) {
	const uint64_t prev   = ::BufNativo::Enter(mesh);
	const uint64_t result = g_mesh_build_original(mesh, block, indices, a3, a4, a5);
	::BufNativo::Leave(prev);
	return result;
}

void KYTY_SYSV_ABI MeshFreeHook(uint64_t mesh) {
	if (mesh != 0) ::BufNativo::OnMeshFree(mesh);
	g_mesh_free_original(mesh);
}

bool PatchEntry(uint64_t address, const std::array<uint8_t, MeshMoved>& code, uint64_t hook, uint64_t trampoline) {
	// El trampolin: los 13 bytes, luego mov r11, <entrada + 13>; jmp r11.
	std::array<uint8_t, MeshMoved + 13> t {};
	std::memcpy(t.data(), code.data(), MeshMoved);
	const uint64_t resume = address + MeshMoved;
	t[MeshMoved]          = 0x49;
	t[MeshMoved + 1]      = 0xbb;
	std::memcpy(t.data() + MeshMoved + 2, &resume, sizeof(resume));
	t[MeshMoved + 10] = 0x41;
	t[MeshMoved + 11] = 0xff;
	t[MeshMoved + 12] = 0xe3;
	std::memcpy(reinterpret_cast<void*>(trampoline), t.data(), t.size());
	if (!Common::VirtualMemory::FlushInstructionCache(trampoline, t.size())) return false;
	// La entrada: mov r11, gancho; jmp r11 (13 bytes).
	std::array<uint8_t, MeshMoved> e {0x49, 0xbb};
	std::memcpy(e.data() + 2, &hook, sizeof(hook));
	e[10] = 0x41;
	e[11] = 0xff;
	e[12] = 0xe3;
	std::memcpy(reinterpret_cast<void*>(address), e.data(), e.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, e.size())) {
		std::memcpy(reinterpret_cast<void*>(address), code.data(), MeshMoved);
		Common::VirtualMemory::FlushInstructionCache(address, MeshMoved);
		return false;
	}
	return true;
}

void InstallBufNativo(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || program->file_name.filename() != "eboot.bin" || !Libs::Graphics::DemonsSouls::IsSupportedGame()) return;
	if (!::BufNativo::On()) {
		std::printf("[buf-nativo] apagado (KYTY_BUF_NATIVO=1 lo enciende): mallas en la RAM emulada como antes\n");
		std::fflush(stdout);
		return;
	}
	const auto refuse = [](const char* why) {
		std::printf("[buf-nativo] NO instalado: %s; mallas en la RAM emulada como antes\n", why);
		std::fflush(stdout);
	};
	const uint64_t base = program->base_vaddr;
	if (program->mapped_size < MeshFreeOffset + 64 || base > UINT64_MAX - MeshFreeOffset - 64) return refuse("eboot demasiado corto");
	const auto same = [&](uint64_t offset, const std::array<uint8_t, MeshMoved>& code) {
		return Libs::Graphics::HostMemoryRangeIsReadable(base + offset, MeshMoved) &&
		       std::memcmp(reinterpret_cast<const void*>(base + offset), code.data(), MeshMoved) == 0;
	};
	if (!same(MeshBuildOffset, MeshBuildCode)) return refuse("eboot+0xaa06a0 no es el de 01.007.000 (u otro gancho ya lo parcheo)");
	if (!same(MeshFreeOffset, MeshFreeCode)) return refuse("eboot+0xaa1d10 no es el de 01.007.000 (u otro gancho ya lo parcheo)");
	const auto* call = reinterpret_cast<const uint8_t*>(base + MeshCopyReturn - 5);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(base + MeshCopyReturn - 5, 5) || call[0] != 0xe8)
		return refuse("eboot+0x90954b no es la llamada al memcpy de CBuffer::Create");
	const uint64_t mem = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                    "demons_souls_buf_nativo");
	if (mem == 0) return refuse("sin memoria para los trampolines");
	g_mesh_build_original = reinterpret_cast<MeshBuildFunction>(mem);
	g_mesh_free_original  = reinterpret_cast<MeshFreeFunction>(mem + 0x80);
	// Primero el destructor: si el constructor no se engancha, no hay nada registrado que olvidar.
	if (!PatchEntry(base + MeshFreeOffset, MeshFreeCode, reinterpret_cast<uint64_t>(&MeshFreeHook), mem + 0x80))
		return refuse("no se pudo parchear eboot+0xaa1d10");
	if (!PatchEntry(base + MeshBuildOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&MeshBuildHook), mem))
		return refuse("no se pudo parchear eboot+0xaa06a0 (0xaa1d10 queda enganchado: no olvida nada)");
	DemonsSoulsTextureProbe::g_bufnativo_return_address.store(base + MeshCopyReturn, std::memory_order_relaxed);
	::BufNativo::Get().hooked.store(true, std::memory_order_release);
	std::printf("[buf-nativo] ganchos puestos (KYTY_BUF_NATIVO=1; =0 lo apaga): 0xaa06a0 registra los CBuffer de cada malla "
	            "(buf7: sus bloques +0x298 vertices y +0x2a0 indices, paginas enteras, 64 KiB de margen), 0xaa1d10 los olvida "
	            "antes del free; tras %d s pasan al almacen (VRAM) y salen de la RAM; linea [buf-nativo] cada 30 s\n",
	            ::BufNativo::DelaySeconds());
	std::fflush(stdout);
#else
	(void)program;
#endif
}

// --- (FRAME-NATIVO2 v7) KYTY_FRAME_NATIVO2=1: gancho de TaggedFrameAllocator::NextFrame -------------------------------
// Juego 0x811860 (Ghidra, archivo 0x81d610): idx=([+0x90]+1)%[+0x94] y el padre (vtable+0x100) recupera los trozos de
// ese indice. Es el unico sitio donde memoria del arena DefaultTagg vuelve a quedar libre: FrameNativo2::OnNextFrame sube
// la epoca ANTES de llamar al original, y la copia COPIA de una epoca se reutiliza solo mientras la epoca no cambie.
// Entrada: push rbp; mov rbp,rsp; push rbx; push rax; lea rcx,[rdi+0xc8] (13 bytes, sin RIP-relativo).
constexpr uint64_t                       NextFrameOffset = 0x811860, NextFrameLoopReturn = 0xa988eb;
constexpr std::array<uint8_t, MeshMoved> NextFrameCode {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x8d, 0x8f, 0xc8, 0x00, 0x00, 0x00};
using NextFrameFunction = KYTY_SYSV_ABI void (*)(uint64_t);
NextFrameFunction g_next_frame_original = nullptr;

// (FRAME-NATIVO2 v9, =3) FreeTag del padre: juego 0x7fe540 (archivo 0x80a2f0) = vtable+0x100 del asignador de 0x730
// bytes que Init (0x8114a0) guarda en [+0x88]. Entrada: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12;
// push rbx (13 bytes, los de MeshBuildCode). Antes de liberar apunta los bloques exactos (FrameNativo2::CollectFreed).
constexpr uint64_t TagFreeOffset = 0x7fe540;
using TagFreeFunction             = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t);
TagFreeFunction g_tag_free_original = nullptr;

uint64_t KYTY_SYSV_ABI TagFreeHook(uint64_t parent, uint64_t key) {
	if (parent != 0) ::FrameNativo2::CollectFreed(parent, key);
	return g_tag_free_original(parent, key);
}

void KYTY_SYSV_ABI NextFrameHook(uint64_t self) {
	::FrameNativo2::OnNextFrame(reinterpret_cast<uint64_t>(__builtin_return_address(0))); // (entrada por jmp: el llamador)
	if ((::FrameNativo2::NoProt() || ::FrameNativo2::SoloLectura()) && ::FrameNativo2::g_vtable_ok.load(std::memory_order_relaxed) != 1 && self != 0) {
		// (v9) Comprobacion con datos: padre = [self+0x88]; su vtable[+0x100] debe ser FreeTag (eboot+0x7fe540) (hasta
		// que una instancia coincida: otra con otro padre, p. ej. el de 0x2eedbe0 / 0x808b00, solo se imprime una vez).
		const uint64_t parent = *reinterpret_cast<const uint64_t*>(self + 0x88);
		const uint64_t fn     = parent != 0 ? *reinterpret_cast<const uint64_t*>(*reinterpret_cast<const uint64_t*>(parent) + 0x100) : 0;
		const uint64_t want   = ::FrameNativo2::g_frame_loop_ret.load(std::memory_order_relaxed) - NextFrameLoopReturn + TagFreeOffset;
		if (::FrameNativo2::g_vtable_ok.exchange(fn == want ? 1 : 0, std::memory_order_relaxed) != (fn == want ? 1 : 0)) {
			std::printf("[frame-nativo2 v9] padre 0x%llx: vtable+0x100 = 0x%llx (FreeTag eboot+0x7fe540 = 0x%llx): %s\n",
			            static_cast<unsigned long long>(parent), static_cast<unsigned long long>(fn),
			            static_cast<unsigned long long>(want), fn == want ? "COINCIDE" : "NO COINCIDE");
			std::fflush(stdout);
		}
	}
	g_next_frame_original(self);
}

void InstallFrameNativo2(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || program->file_name.filename() != "eboot.bin" || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    (::FrameNativo2::Mode() != 1 && !::FrameNativo2::NoProt() && !::FrameNativo2::SoloLectura()))
		return; // (=1 v7 y, desde v9, =3: v8 lo necesitaba y no se ponia; v11: =4)
	const auto refuse = [](const char* why) {
		std::printf("[frame-nativo2] gancho NextFrame NO puesto: %s; COPIA sin reutilizar copias (como v6)\n", why);
		if (::FrameNativo2::Nativo()) std::printf("[frame-nativo2 v13] =6 NO ACTIVO: %s\n", why);
		std::fflush(stdout);
	};
	const uint64_t base = program->base_vaddr;
	if (program->mapped_size < NextFrameLoopReturn + 16) return refuse("eboot demasiado corto");
	if (!Libs::Graphics::HostMemoryRangeIsReadable(base + NextFrameOffset, MeshMoved) ||
	    std::memcmp(reinterpret_cast<const void*>(base + NextFrameOffset), NextFrameCode.data(), MeshMoved) != 0)
		return refuse("eboot+0x811860 no es el NextFrame de 01.007.000 (u otro gancho ya lo parcheo)");
	const auto* call = reinterpret_cast<const uint8_t*>(base + NextFrameLoopReturn - 5);
	if (!Libs::Graphics::HostMemoryRangeIsReadable(base + NextFrameLoopReturn - 5, 5) || call[0] != 0xe8)
		return refuse("eboot+0xa988e6 no es la llamada del bucle de fotograma");
	const uint64_t mem = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                    "demons_souls_frame_nativo2");
	if (mem == 0) return refuse("sin memoria para el trampolin");
	g_next_frame_original = reinterpret_cast<NextFrameFunction>(mem);
	::FrameNativo2::g_frame_loop_ret.store(base + NextFrameLoopReturn, std::memory_order_relaxed);
	if (!PatchEntry(base + NextFrameOffset, NextFrameCode, reinterpret_cast<uint64_t>(&NextFrameHook), mem))
		return refuse("no se pudo parchear eboot+0x811860");
	::FrameNativo2::g_hooked.store(true, std::memory_order_release);
	std::printf("[frame-nativo2] gancho NextFrame puesto (eboot+0x811860): copias COPIA reutilizadas dentro de la misma "
	            "epoca y tick\n");
	std::fflush(stdout);
	if (!::FrameNativo2::NoProt() && !::FrameNativo2::SoloLectura()) return;
	// (v9, =3) FreeTag: la invalidacion exacta donde el juego reutiliza.
	if (program->mapped_size < TagFreeOffset + 64 || !Libs::Graphics::HostMemoryRangeIsReadable(base + TagFreeOffset, MeshMoved) ||
	    std::memcmp(reinterpret_cast<const void*>(base + TagFreeOffset), MeshBuildCode.data(), MeshMoved) != 0) {
		std::printf("[frame-nativo2 v9] gancho FreeTag NO puesto: eboot+0x7fe540 no es el de 01.007.000; sin invalidacion\n");
		if (::FrameNativo2::Nativo()) std::printf("[frame-nativo2 v13] =6 NO ACTIVO: sin gancho FreeTag no hay cuarentena (todo vigilado)\n");
		std::fflush(stdout);
		return;
	}
	g_tag_free_original = reinterpret_cast<TagFreeFunction>(mem + 0x80);
	if (!PatchEntry(base + TagFreeOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&TagFreeHook), mem + 0x80)) {
		std::printf("[frame-nativo2 v9] gancho FreeTag NO puesto: no se pudo parchear eboot+0x7fe540\n");
		std::fflush(stdout);
		return;
	}
	::FrameNativo2::g_freetag_hooked.store(true, std::memory_order_release); // (v13, =6: sin esto no hay cuarentena)
	if (::FrameNativo2::Nativo())
		std::printf("[frame-nativo2 v13] =6 ACTIVO: ganchos NextFrame y FreeTag puestos (cuarentena); AllocTag lo pone [arena-dueno]; "
		            "el arena se importa en el primer uso (linea 'arena N importado') y la linea [frame-nativo2 v13] sale cada 10 s\n");
	std::printf("[frame-nativo2 v9] gancho FreeTag puesto (eboot+0x7fe540): los bloques exactos liberados se descartan en "
	            "la cache antes de reutilizarse\n");
	std::fflush(stdout);
#else
	(void)program;
#endif
}

// --- (MALLA-NATIVA) encendido por defecto (KYTY_MALLA_NATIVA=0 lo apaga): local/malla-nativa.h ---------------------------------
// Los mismos dos puntos que BUF-NATIVO (0xaa06a0 al volver, 0xaa1d10 antes), con sus propios trampolines; no se instala
// si BUF-NATIVO esta encendido (los dos parchean las mismas entradas). Las lecturas de CPU llegan por los ganchos de
// 0x909940/0x909970 de KYTY_DS_READBACK_HOOK (loader/demonsSoulsWriteAudit.cpp, encendido por defecto).
MeshBuildFunction g_malla_build_original = nullptr;
MeshFreeFunction  g_malla_free_original  = nullptr;

uint64_t KYTY_SYSV_ABI MallaBuildHook(uint64_t mesh, uint64_t block, uint64_t indices, uint64_t a3, uint64_t a4, uint64_t a5) {
	const uint64_t result = g_malla_build_original(mesh, block, indices, a3, a4, a5);
	::MallaNativa::OnBuild(mesh);
	return result;
}

void KYTY_SYSV_ABI MallaFreeHook(uint64_t mesh) {
	if (mesh != 0) ::MallaNativa::OnFree(mesh);
	g_malla_free_original(mesh);
}

// (fase 1g, diagnostico) CBuffer::Create 0x909460 (Ghidra archivo 0x915210): (slot, fmt, stride, n, flags, datos), sin
// argumentos en la pila (objdump: ningun [rbp+0x10..]). Mismos 13 bytes de entrada que 0xaa06a0. Al volver: el origen del V#.
constexpr uint64_t CBufferCreateOffset = 0x909460;
using CBufferCreateFunction            = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
CBufferCreateFunction g_cbuffer_create_original = nullptr;

uint64_t KYTY_SYSV_ABI MallaCBufferCreateHook(uint64_t slot, uint64_t fmt, uint64_t stride, uint64_t n, uint64_t flags, uint64_t data) {
	const auto     ret    = reinterpret_cast<uint64_t>(__builtin_return_address(0)); // (entrada por jmp: el llamador del juego)
	const uint64_t result = g_cbuffer_create_original(slot, fmt, stride, n, flags, data);
	::MallaNativa::OnCBufferCreate(slot, ret);
	return result;
}

void InstallMallaNativa(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || program->file_name.filename() != "eboot.bin" || !Libs::Graphics::DemonsSouls::IsSupportedGame()) return;
	if (!::MallaNativa::On()) {
		std::printf("[malla-nativa] apagado (KYTY_MALLA_NATIVA=0; quitarlo del extra lo enciende)\n");
		std::fflush(stdout);
		return;
	}
	const auto refuse = [](const char* why) {
		std::printf("[malla-nativa] NO instalado: %s; mallas como antes\n", why);
		std::fflush(stdout);
	};
	if (::BufNativo::On()) return refuse("KYTY_BUF_NATIVO=1 usa las mismas entradas (apagalo para probar esto)");
	const uint64_t base = program->base_vaddr;
	if (program->mapped_size < MeshFreeOffset + 64 || base > UINT64_MAX - MeshFreeOffset - 64) return refuse("eboot demasiado corto");
	const auto same = [&](uint64_t offset, const std::array<uint8_t, MeshMoved>& code) {
		return Libs::Graphics::HostMemoryRangeIsReadable(base + offset, MeshMoved) &&
		       std::memcmp(reinterpret_cast<const void*>(base + offset), code.data(), MeshMoved) == 0;
	};
	if (!same(MeshBuildOffset, MeshBuildCode)) return refuse("eboot+0xaa06a0 no es el de 01.007.000 (u otro gancho ya lo parcheo)");
	if (!same(MeshFreeOffset, MeshFreeCode)) return refuse("eboot+0xaa1d10 no es el de 01.007.000 (u otro gancho ya lo parcheo)");
	const uint64_t mem = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                    "demons_souls_malla_nativa");
	if (mem == 0) return refuse("sin memoria para los trampolines");
	g_malla_build_original = reinterpret_cast<MeshBuildFunction>(mem);
	g_malla_free_original  = reinterpret_cast<MeshFreeFunction>(mem + 0x80);
	// Primero el destructor: si el constructor no se engancha, no hay nada registrado.
	if (!PatchEntry(base + MeshFreeOffset, MeshFreeCode, reinterpret_cast<uint64_t>(&MallaFreeHook), mem + 0x80))
		return refuse("no se pudo parchear eboot+0xaa1d10");
	if (!PatchEntry(base + MeshBuildOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&MallaBuildHook), mem))
		return refuse("no se pudo parchear eboot+0xaa06a0 (0xaa1d10 queda enganchado: no hace nada)");
	::MallaNativa::Get().hooked.store(true, std::memory_order_release);
	::MallaNativa::Get().eboot.store(base, std::memory_order_relaxed);
	// (fase 1g) Diagnostico del origen de los huecos; si no se puede, la linea lo dira (origen conocido 0).
	// (fase 1i) Solo con KYTY_MALLA_NATIVA_DIAG=1: el gancho corre en cada CBuffer::Create (489 llamadores, tambien por
	// dibujo desde 0xab6080) y costaba fotograma. Por defecto NO se pone.
	if (!::MallaNativa::Diag()) {
	} else if (program->mapped_size > CBufferCreateOffset + 64 && same(CBufferCreateOffset, MeshBuildCode)) {
		g_cbuffer_create_original = reinterpret_cast<CBufferCreateFunction>(mem + 0x100);
		if (!PatchEntry(base + CBufferCreateOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&MallaCBufferCreateHook), mem + 0x100))
			std::printf("[malla-nativa] AVISO: no se pudo parchear eboot+0x909460 (sin origen de los huecos)\n");
	} else {
		std::printf("[malla-nativa] AVISO: eboot+0x909460 no es el de 01.007.000 (sin origen de los huecos)\n");
	}
	std::printf("[malla-nativa] ganchos puestos (encendido por defecto): 0xaa06a0 registra los bloques +0x298/+0x2a0 de cada malla, "
	            "su primer uso por la GPU crea un VkBuffer con ellos (sin vigilar ni sincronizar esa memoria), 0xaa1d10 lo destruye; "
	            "RAM: %s; linea [malla-nativa] cada 30 s\n",
	            ::MallaNativa::RamOn() ? "KYTY_MALLA_NATIVA_RAM=1: los bloques sin lecturas de CPU en 30 s sueltan sus paginas internas"
	                                   : "intacta (KYTY_MALLA_NATIVA_RAM=1 la suelta)");
	std::fflush(stdout);
#else
	(void)program;
#endif
}
// --- (ARENA-DUENO) KYTY_ARENA_DUENO=1 (apagado por defecto): medidor, local/arena-dueno.h -----------------------------
// Init 0x8114a0 (archivo 0x81d250): mov byte [rdi+8],1; lea rcx,[rdi+0xc8]; vxorps xmm0,xmm0,xmm0 (15 bytes, sin
// RIP-relativo; la funcion es hoja). AllocTag del padre (vtable+0xf8) 0x7fe350 (archivo 0x80a100): los 13 bytes de
// MeshBuildCode. Los ganchos llaman al original entero y apuntan; no cambian nada del juego.
constexpr uint64_t                 ArenaInitOffset = 0x8114a0, ArenaAllocTagOffset = 0x7fe350;
constexpr std::array<uint8_t, 15> ArenaInitCode {0xc6, 0x47, 0x08, 0x01, 0x48, 0x8d, 0x8f, 0xc8, 0x00, 0x00, 0x00, 0xc5, 0xf8, 0x57, 0xc0};
using ArenaInitFunction     = KYTY_SYSV_ABI void (*)(uint64_t, uint64_t, uint32_t);
using ArenaAllocTagFunction = KYTY_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
ArenaInitFunction     g_arena_init_original     = nullptr;
ArenaAllocTagFunction g_arena_alloctag_original = nullptr;

void KYTY_SYSV_ABI ArenaInitHook(uint64_t self, uint64_t parent, uint32_t nbuf) {
	const auto ret = reinterpret_cast<uint64_t>(__builtin_return_address(0)); // (entrada por jmp: el llamador del juego)
	g_arena_init_original(self, parent, nbuf);
	if (self != 0) ::ArenaDueno::OnInit(self, parent, nbuf, ret);
}

// (ARENA-DUENO paso 0, KYTY_ARENA_DUENO=2) AllocTag del otro padre 0x7fcb50 (vtable 0x25accc8+0xf8; mismos 13 bytes de
// entrada que 0x7fe350 y mismos argumentos: padre, instancia, bytes, bloque, clave, origen). Solo apunta el dueno.
constexpr uint64_t    ArenaAllocTagBOffset    = 0x7fcb50;
ArenaAllocTagFunction g_arena_alloctagB_original = nullptr;
uint64_t KYTY_SYSV_ABI ArenaAllocTagBHook(uint64_t parent, uint64_t self, uint64_t bytes, uint64_t block, uint64_t key, uint64_t src) {
	const uint64_t result = g_arena_alloctagB_original(parent, self, bytes, block, key, src);
	::ArenaDueno::OnAllocTagB(parent, self, bytes, key, result);
	return result;
}

uint64_t KYTY_SYSV_ABI ArenaAllocTagHook(uint64_t parent, uint64_t self, uint64_t bytes, uint64_t block, uint64_t key, uint64_t src) {
	const uint64_t result = g_arena_alloctag_original(parent, self, bytes, block, key, src);
	::ArenaDueno::OnAllocTag(parent, self, bytes, key, result);
	// (FRAME-NATIVO2 v11, =4) Bloques de una instancia de solo lectura: fuera de la vigilancia y RW ya.
	if (::FrameNativo2::SoloLectura()) {
		uint64_t lo = 0, size = 0;
		if (::FrameNativo2::OnAllocTagRo(parent, bytes, result, &lo, &size)) {
#if defined(_WIN32)
			if (!Libs::LibKernel::Memory::RestoreGuestWritable(lo, size))
				::FrameNativo2::g_ro_restore_fail.fetch_add(1, std::memory_order_relaxed);
#endif
		}
	}
	return result;
}

// Como PatchEntry, para una entrada de N bytes (N >= 13): trampolin = los N bytes + mov r11, entrada+N; jmp r11.
template <size_t N>
bool PatchEntryN(uint64_t address, const std::array<uint8_t, N>& code, uint64_t hook, uint64_t trampoline) {
	static_assert(N >= 13);
	std::array<uint8_t, N + 13> t {};
	std::memcpy(t.data(), code.data(), N);
	const uint64_t resume = address + N;
	t[N]                  = 0x49;
	t[N + 1]              = 0xbb;
	std::memcpy(t.data() + N + 2, &resume, sizeof(resume));
	t[N + 10] = 0x41;
	t[N + 11] = 0xff;
	t[N + 12] = 0xe3;
	std::memcpy(reinterpret_cast<void*>(trampoline), t.data(), t.size());
	if (!Common::VirtualMemory::FlushInstructionCache(trampoline, t.size())) return false;
	std::array<uint8_t, N> e {};
	for (auto& x: e) x = 0x90; // (los bytes sobrantes tras el jmp: nop, nunca se ejecutan)
	e[0] = 0x49;
	e[1] = 0xbb;
	std::memcpy(e.data() + 2, &hook, sizeof(hook));
	e[10] = 0x41;
	e[11] = 0xff;
	e[12] = 0xe3;
	std::memcpy(reinterpret_cast<void*>(address), e.data(), e.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, e.size())) {
		std::memcpy(reinterpret_cast<void*>(address), code.data(), N);
		Common::VirtualMemory::FlushInstructionCache(address, N);
		return false;
	}
	return true;
}

// (RAM-NATIVA) KYTY_RAM_NATIVA=1/2: el hilo [ram-nativa] (kernel/memory.cpp) toma los padres de los Init vistos aqui.
static bool RamNativaWanted() {
	const char* v = std::getenv("KYTY_RAM_NATIVA");
	return v != nullptr && (v[0] == '1' || v[0] == '2');
}

void InstallArenaDueno(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (program == nullptr || program->file_name.filename() != "eboot.bin" || !Libs::Graphics::DemonsSouls::IsSupportedGame() ||
	    (!::ArenaDueno::On() && !::FrameNativo2::SoloLectura() && !RamNativaWanted())) // (v11: =4; RAM-NATIVA: los padres)
		return;
	const auto refuse = [](const char* why) {
		std::printf("[arena-dueno] NO instalado: %s; sin medida\n", why);
		std::fflush(stdout);
	};
	const uint64_t base = program->base_vaddr;
	if (program->mapped_size < ArenaInitOffset + 64 || base > UINT64_MAX - ArenaInitOffset - 64) return refuse("eboot demasiado corto");
	if (!Libs::Graphics::HostMemoryRangeIsReadable(base + ArenaInitOffset, ArenaInitCode.size()) ||
	    std::memcmp(reinterpret_cast<const void*>(base + ArenaInitOffset), ArenaInitCode.data(), ArenaInitCode.size()) != 0)
		return refuse("eboot+0x8114a0 no es el Init de 01.007.000 (u otro gancho ya lo parcheo)");
	if (!Libs::Graphics::HostMemoryRangeIsReadable(base + ArenaAllocTagOffset, MeshMoved) ||
	    std::memcmp(reinterpret_cast<const void*>(base + ArenaAllocTagOffset), MeshBuildCode.data(), MeshMoved) != 0)
		return refuse("eboot+0x7fe350 no es el AllocTag de 01.007.000 (u otro gancho ya lo parcheo)");
	const uint64_t mem = Libs::LibKernel::Memory::AllocateRuntimeMemory(0, 0x1000, Common::VirtualMemory::Mode::ExecuteReadWrite,
	                                                                    "demons_souls_arena_dueno");
	if (mem == 0) return refuse("sin memoria para los trampolines");
	::ArenaDueno::g_eboot.store(base, std::memory_order_relaxed);
	g_arena_alloctag_original = reinterpret_cast<ArenaAllocTagFunction>(mem + 0x80);
	if (!PatchEntry(base + ArenaAllocTagOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&ArenaAllocTagHook), mem + 0x80))
		return refuse("no se pudo parchear eboot+0x7fe350");
	g_arena_init_original = reinterpret_cast<ArenaInitFunction>(mem);
	if (!PatchEntryN(base + ArenaInitOffset, ArenaInitCode, reinterpret_cast<uint64_t>(&ArenaInitHook), mem))
		return refuse("no se pudo parchear eboot+0x8114a0 (0x7fe350 queda enganchado: apunta bloques sin llamador)");
	::ArenaDueno::g_hooked.store(true, std::memory_order_release);
	if (::ArenaDueno::Dos()) {
		if (!Libs::Graphics::HostMemoryRangeIsReadable(base + ArenaAllocTagBOffset, MeshMoved) ||
		    std::memcmp(reinterpret_cast<const void*>(base + ArenaAllocTagBOffset), MeshBuildCode.data(), MeshMoved) != 0) {
			std::printf("[arena-dueno paso0] NO ACTIVO: eboot+0x7fcb50 no es el AllocTag de 01.007.000\n");
		} else {
			g_arena_alloctagB_original = reinterpret_cast<ArenaAllocTagFunction>(mem + 0x100);
			if (PatchEntry(base + ArenaAllocTagBOffset, MeshBuildCode, reinterpret_cast<uint64_t>(&ArenaAllocTagBHook), mem + 0x100))
				std::printf("[arena-dueno paso0] ACTIVO (KYTY_ARENA_DUENO=2): gancho AllocTag eboot+0x7fcb50 del otro padre; dueno y "
				            "tamano real del bloque; usos de la GPU por instancia en la linea [arena-dueno] cada 10 s\n");
			else
				std::printf("[arena-dueno paso0] NO ACTIVO: no se pudo parchear eboot+0x7fcb50\n");
		}
		std::fflush(stdout);
	}
	std::printf("[arena-dueno] ganchos puestos (KYTY_ARENA_DUENO=1): Init eboot+0x8114a0 (instancia, padre, buferes, tag, "
	            "llamador) y AllocTag eboot+0x7fe350 (bloque de 1 MiB -> instancia); linea [arena-dueno] cada 10 s\n");
	std::fflush(stdout);
#else
	(void)program;
#endif
}
#include "loader/demonsSoulsNativoPase.inc" // (F5 pase 1) KYTY_NATIVO_PASE_SSRCLEAR, off by default
#include "loader/demonsSoulsNativoSombras.inc" // (F5 pase 2) KYTY_NATIVO_PASE_SOMBRAS=2, off by default
} // namespace

void Install(Program* program) {
	InstallPort2Check(program); // (port2) eboot.bin only; KYTY_PORT2, off by default (a check, nothing patched)
	if (program != nullptr) // (f5) KYTY_NATIVE_RENDER, off by default: the profiler scopes' bytes checked, nothing patched
		NativeRender::InstallCheck(program->base_vaddr, program->mapped_size,
		                           program->file_name.filename() == "eboot.bin" && Libs::Graphics::DemonsSouls::IsSupportedGame());
	if (program != nullptr) // (g1, KYTY_G1_PROBE / KYTY_G1_STATE, off by default: the binder's copies checked, nothing patched)
		G1::Install(program->base_vaddr, program->mapped_size,
		            program->file_name.filename() == "eboot.bin" && Libs::Graphics::DemonsSouls::IsSupportedGame());
	DemonsSoulsTextureProbe::Install(program); // eboot.bin only; measurement (KYTY_TEXTURE_PROBE=0 off)
	if (program != nullptr && program->file_name.filename() == "eboot.bin" && Libs::Graphics::DemonsSouls::IsSupportedGame())
		::Libs::Graphics::TexAntEarly::Install(); // (etapa1b) KYTY_TEX_ANTICIPADA: los ganchos antes de que el juego cree texturas
	InstallBufNativo(program); // (BUF-NATIVO) eboot.bin only; KYTY_BUF_NATIVO=1 turns it on (off by default)
	InstallFrameNativo2(program); // (FRAME-NATIVO2 v7) KYTY_FRAME_NATIVO2=1 only: NextFrame hook
	InstallArenaDueno(program); // (ARENA-DUENO) eboot.bin only; KYTY_ARENA_DUENO=1, off by default (medidor)
	InstallMallaNativa(program); // (MALLA-NATIVA) eboot.bin only; on by default; KYTY_MALLA_NATIVA=0 turns it off
	InstallLinearCopyHook(program); // (render4) eboot.bin only; KYTY_DS_LINEAR_COPY_HOOK, off by default
	InstallNativoPaseHook(program); // (F5 pase 1) eboot.bin only; KYTY_NATIVO_PASE_SSRCLEAR=1|2, off by default
	InstallDrawLoopHook(program); // (render1) eboot.bin only; KYTY_RENDER1=1|2|3, off by default (the hook observes only)
	DemonsSoulsPortR::Install(program); // (port-r) eboot.bin only; KYTY_PORT_R=1|2 o KYTY_PORTG=1|2, off by default (eboot+0xab57d0 on the host)
	InstallNativoSombrasHook(program); // (F5 pase 2) eboot.bin only; KYTY_NATIVO_PASE_SOMBRAS=2 (mide), off by default
#if defined(__x86_64__) || defined(_M_X64)
	if (site || !program || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "libc.prx")
		return;
	const auto body = std::find_if(Bodies.begin(), Bodies.end(), [&](const Body& b) {
		const auto address = program->base_vaddr + b.offset;
		return program->mapped_size >= b.offset + b.size && program->base_vaddr <= UINT64_MAX - b.offset - b.size &&
		       Libs::Graphics::HostMemoryRangeIsReadable(address, b.size) &&
		       XXH3_64bits(reinterpret_cast<const void*>(address), b.size) == b.hash;
	});
	if (body == Bodies.end()) {
		std::printf("Demon's Souls copy: libc.prx of an unknown build; retaining its memmove\n");
		return;
	}
	const auto address = program->base_vaddr + body->offset;
	std::memcpy(original.data(), reinterpret_cast<const void*>(address), original.size());
	patch = Tailcall(reinterpret_cast<uint64_t>(&CoherentMove));
	std::memcpy(reinterpret_cast<void*>(address), patch.data(), patch.size());
	if (!Common::VirtualMemory::FlushInstructionCache(address, patch.size())) {
		std::memcpy(reinterpret_cast<void*>(address), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(address, original.size());
		return;
	}
	site = address;
	std::printf("Demon's Souls copy: installed verified coherent memmove (libc.prx of %s)\n", body->build);
#endif
}

void Clear() {
	if (!site) return;
	if (std::memcmp(reinterpret_cast<const void*>(site), patch.data(), patch.size()) == 0) {
		std::memcpy(reinterpret_cast<void*>(site), original.data(), original.size());
		Common::VirtualMemory::FlushInstructionCache(site, original.size());
	}
	site = 0;
}
} // namespace Loader::DemonsSoulsCopy
