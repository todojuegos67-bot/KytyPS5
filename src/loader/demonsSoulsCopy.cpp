#include "loader/demonsSoulsCopy.h"

#include "common/logging/log.h"
#include "common/virtualMemory.h"
#include "ds-linear-copy-hook.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "loader/demonsSoulsTextureProbe.h"
#include "loader/demonsSoulsWriteAudit.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <xxhash.h>

extern "C" {
// KYTY_GUEST_MOVE_COMPARE (default on; 0 for an A/B): a large move writes only the destination pages whose bytes
// differ (CompareMove).
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_move_compare {1};
}

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
	g_linear_copy_original(context, dst, dst_dwords, src, src_dwords);
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
} // namespace

void Install(Program* program) {
	DemonsSoulsTextureProbe::Install(program); // eboot.bin only; measurement (KYTY_TEXTURE_PROBE=0 off)
	InstallLinearCopyHook(program); // (render4) eboot.bin only; KYTY_DS_LINEAR_COPY_HOOK, off by default
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
