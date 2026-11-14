// KYTY_NATIVE_TEX=1, streaming part: see demonsSoulsNativeTexStream.h. Static facts (01.007.000 eboot, ELF vaddrs;
// /home/claude/static/native-tex/B-streaming.md, verified instruction by instruction):
//   VirtualMemoryGPU       static IAllocator 0x2f65eb8, vtable 0x25b38c0 (object pointer = +0x10), handle 0x2f64438
//                          (every reference to the handle is texture code: FTextureFactory 0x8febed / 0x8fec22,
//                          CTexture slice commit 0x9010b3 / decommit 0x901193, SetCurrentStreamingChunk 0x9013da /
//                          0x9014bd, TextureArrayManager init 0xac4189)
//   vtable slots           [vptr+0xb0] = Free 0xaa3430, [vptr+0xc0] = Realloc 0xaa2a20 (called by bpe Realloc 0x7f9f70
//                          at 0x7fa1b0, by bpe Free 0x7fa380 at 0x7fa5b1)
//   Realloc 0xaa2a20       rdi = this, rsi = ptr, rdx = size, ecx = mode (0 commit [ptr, ptr+size), 1 resize the run,
//                          2 decommit [ptr, ptr+size)); r11 = FindPages(ptr) 0x801b80 (0 when ptr = 0)
//     0xaa2b90..0xaa2bc1   mode 1: ResizePages(this, run r11, 0, (size + 0xffff) & ~0xffff, op 1, align, [rsp])
//     0xaa3325..0xaa3346   modes 0/2 (after the run's sub-range list is updated): the pages not shared with another
//                          committed sub-range, ResizePages(this, run, rdx = page start (absolute), rcx = length,
//                          op r8d = 0 or 2, 0x10000, [rsp]); jbe 0xaa3359 skips the call when no whole page changes
//   ResizePages 0x801da0   called from exactly those two sites in the whole eboot
//   Free 0xaa3430          0xaa34bf FindPages(this, ptr) -> r12 = the run; 0xaa353d ReleasePages 0x801a10(this, run)
//   PageAllocationInfo     VA [+0x60], size [+0x58] (FindPages 0x801b80 compares against them)
#include "loader/demonsSoulsNativeTexStream.h"
#include "loader/demonsSoulsNativeTex.h"

#include "common/abi.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

namespace Loader {

namespace NativeTex {
static std::atomic<StreamRelease> g_stream_release {nullptr};
static std::atomic<StreamMove>    g_stream_move {nullptr};
static std::atomic<StreamCommit>  g_stream_commit {nullptr};
static std::atomic<StreamCommit>  g_stream_decommit {nullptr};
void SetStreamConsumer(StreamRelease release, StreamMove move, StreamCommit commit, StreamCommit decommit) noexcept {
	g_stream_release.store(release, std::memory_order_release);
	g_stream_move.store(move, std::memory_order_release);
	g_stream_commit.store(commit, std::memory_order_release);
	g_stream_decommit.store(decommit, std::memory_order_release);
}
// (op 2: to the emulator's decommit consumer, else as a release.)
static void OnTextureDecommit(uint64_t addr, uint64_t size) noexcept {
	if (const auto decommit = g_stream_decommit.load(std::memory_order_acquire); decommit != nullptr) decommit(addr, size);
	else OnTextureRelease(addr, size);
}
// Defaults: the original behaviour (the engine's own code runs right after each notification).
void OnTextureRelease(uint64_t addr, uint64_t size) noexcept {
	if (const auto release = g_stream_release.load(std::memory_order_acquire); release != nullptr) release(addr, size);
}
void OnTextureCommit(uint64_t addr, uint64_t size) noexcept {
	if (const auto commit = g_stream_commit.load(std::memory_order_acquire); commit != nullptr) commit(addr, size);
}
// The game's own copy, told to the trackers once for the whole destination (as KYTY_NATIVE_MEM=3's MipCopy and a file
// read do); the source is read as the game's memcpy reads it (a GPU-written source faults its bytes back).
void OnTextureMove(uint64_t dst, uint64_t src, uint64_t size) noexcept {
	if (size == 0 || dst == src) return;
	if (const auto move = g_stream_move.load(std::memory_order_acquire); move != nullptr && move(dst, src, size)) return;
	const bool prepared = Libs::LibKernel::Memory::PrepareHostOverwrite(dst, size);
	std::memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size);
	Libs::LibKernel::Memory::FinishHostOverwrite(dst, size, prepared);
}
} // namespace NativeTex

namespace DemonsSoulsNativeTexStream {

std::atomic<uint32_t> g_installed {0};

namespace {

// ---- static facts (01.007.000) ----
constexpr uint64_t VmGpuVtable = 0x25b38c0, VmGpuFree = 0xaa3430, VmGpuRealloc = 0xaa2a20;
constexpr uint64_t ResizePages = 0x801da0, ReleasePages = 0x801a10, FindPages = 0x801b80;
constexpr uint64_t ResizeCall = 0xaa3346, RunResizeCall = 0xaa2bc1, ReleaseCall = 0xaa353d;
constexpr uint64_t InfoVa = 0x60, InfoSize = 0x58;
// TextureArrayManager's texture -> pool slice copy 0x900e50(pool rdi, layer esi, source texture rdx, rcx): with rcx = 0,
// memcpy(pool base + layer * faces * slice, source base, faces * slice) at 0x900f26 (callers 0xac51ed, the manager's
// deferred list run from UpdateTextureStreamingGPUData 0xa16a50, then 0x900f70 frees the source; and 0x9d5034).
constexpr uint64_t PoolCopyCall = 0x900f26, MemcpyStub = 0x1e73d00;
// The cave (Idle 0x8000000, SceneGuard 0x8008000, Camera 0x8020000, NativeMem 0x8040000, GpuPages 0x8080000..+0x8000).
constexpr uint64_t CaveOffset = 0x80b4000, CaveSize = 0x4000;

struct Window {
	uint64_t    offset;
	const char* hex;
};
constexpr std::array<Window, 12> Exact {{
    // modes 0/2: lea rax,[rbp-0x70]; mov rdi,rbx; mov rsi,r11; mov r8d,2; mov [rbp-0x70],0; mov [rsp],rax; mov r9d,0x10000
    {0xaa3325, "488d45904889df4c89de41b802000000c7459000000000" "4889042441b900000100"},
    // ... and the mode 0 entry into it: mov [rbp-0x70],0; mov rdi,rbx; mov rsi,r11; xor r8d,r8d; jmp 0xaa333c
    {0xaa3135, "c74590000000004889df4c89de4531c0e9f2010000"},
    // page range: rax = ptr & ~0xffff ... rcx = end - start; jbe skip
    {0xaa32ed, "4889f84981c6ffff0000482500"},
    // mode 1: ... mov rdi,rbx; mov rsi,r11; mov r8d,1; xor r15d,r15d; xor edx,edx; and r14,~0xffff; mov rcx,r14;
    // mov [rsp],rax
    {0xaa2ba2, "4889df4c89de41b8010000004531ff31d24981e60000ffff4c89f148890424"},
    // Free: FindPages(this, ptr) -> r12; ...; mov rdi,rbx; mov rsi,r12 before ReleasePages
    {0xaa34b7, "4889df4c89f631d2"},
    {0xaa34c4, "4c8b68684989c4"},
    {0xaa3537, "4889df4c89e6"},
    // the targets' first bytes: ResizePages cmp byte [rdi+9],1; ReleasePages / FindPages entries
    {ResizePages, "807f0901"},
    {FindPages, "488b47604883c750"},
    // 0x900e50: mov r15,rcx; lea r12,[rbp-0x3c8]; mov ebx,esi; mov r14,rdi ... test r15,r15; jne 0x900f2b (no copy)
    {0x900e6b, "4989cf4c8da538fcffff89f34989fe"},
    {0x900ece, "4d85ff7558"},
    // ... imul rbx,rcx; add rbx,rax; mov rdi,rbx (rsi = source base, rdx = size) before the memcpy
    {0x900f1c, "480fafd94801c34889df"},
}};
struct Call {
	uint64_t at, target;
};
constexpr std::array<Call, 5> Calls {{
    {ResizeCall, ResizePages},
    {RunResizeCall, ResizePages},
    {ReleaseCall, ReleasePages},
    {0xaa34bf, FindPages},
    {PoolCopyCall, MemcpyStub},
}};

int HexNibble(char c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	return -1;
}

bool Matches(uint64_t address, const char* hex) {
	const auto* p = reinterpret_cast<const uint8_t*>(address);
	size_t      i = 0;
	for (const char* h = hex; *h != '\0';) {
		if (*h == ' ') {
			h++;
			continue;
		}
		const int hi = HexNibble(h[0]), lo = h[1] != '\0' ? HexNibble(h[1]) : -1;
		if (hi < 0 || lo < 0) return false;
		if (!Libs::Graphics::HostMemoryRangeIsReadable(address + i, 1) || p[i] != static_cast<uint8_t>(hi * 16 + lo))
			return false;
		h += 2;
		i++;
	}
	return i != 0;
}

uint64_t CallTarget(uint64_t site) {
	if (!Libs::Graphics::HostMemoryRangeIsReadable(site, 5)) return 0;
	const auto* p = reinterpret_cast<const uint8_t*>(site);
	if (p[0] != 0xe8) return 0;
	int32_t rel = 0;
	std::memcpy(&rel, p + 1, sizeof(rel));
	return site + 5 + static_cast<int64_t>(rel);
}

const char* CheckStatic(uint64_t base, uint64_t* where) {
	for (const auto& w: Exact) {
		*where = w.offset;
		if (!Matches(base + w.offset, w.hex)) return "bytes differ";
	}
	for (const auto& c: Calls) {
		*where = c.at;
		if (CallTarget(base + c.at) != base + c.target) return "call target differs";
	}
	// The relocated vtable slots: the hooks are VirtualMemoryGPU's own Realloc and Free.
	for (const auto& [slot, function]: {std::pair<uint64_t, uint64_t> {0xc0, VmGpuFree}, {0xd0, VmGpuRealloc}}) {
		*where = VmGpuVtable + slot;
		uint64_t value = 0;
		if (!Libs::Graphics::HostMemoryRangeIsReadable(base + VmGpuVtable + slot, 8)) return "vtable unreadable";
		std::memcpy(&value, reinterpret_cast<const void*>(base + VmGpuVtable + slot), 8);
		if (value != base + function) return "VirtualMemoryGPU vtable slot differs";
	}
	*where = 0;
	return nullptr;
}

// ---- counters (the hooks run on game threads; the report thread prints) ----
std::atomic<uint64_t> g_commits {0}, g_commit_bytes {0}, g_releases {0}, g_release_bytes {0}, g_run_frees {0},
    g_run_free_bytes {0}, g_run_shrinks {0}, g_moves {0}, g_move_bytes {0};

uint64_t ReadInfo(uint64_t info, uint64_t field) {
	uint64_t value = 0;
	std::memcpy(&value, reinterpret_cast<const void*>(info + field), sizeof(value));
	return value;
}

// ResizePages(this, run, start, length, op, ...) from VirtualMemoryGPU::Realloc, before the call.
void KYTY_SYSV_ABI OnResizePages(uint64_t run, uint64_t start, uint64_t length, uint32_t op) {
	if (op == 0 && length != 0) {
		g_commits.fetch_add(1, std::memory_order_relaxed);
		g_commit_bytes.fetch_add(length, std::memory_order_relaxed);
		NativeTex::OnTextureCommit(start, length);
	} else if (op == 2 && length != 0) {
		g_releases.fetch_add(1, std::memory_order_relaxed);
		g_release_bytes.fetch_add(length, std::memory_order_relaxed);
		NativeTex::OnTextureDecommit(start, length);
	} else if (op == 1 && run != 0) {
		// The new size is already rounded to 64 KiB (0xaa2bb3); the run keeps its VA (Realloc returns ptr, 0xaa3375).
		const uint64_t va = ReadInfo(run, InfoVa), old_size = ReadInfo(run, InfoSize);
		if (length < old_size) {
			g_run_shrinks.fetch_add(1, std::memory_order_relaxed);
			g_release_bytes.fetch_add(old_size - length, std::memory_order_relaxed);
			NativeTex::OnTextureRelease(va + length, old_size - length);
		}
	}
}

// ReleasePages(this, run) from VirtualMemoryGPU::Free, before the call.
void KYTY_SYSV_ABI OnReleaseRun(uint64_t run) {
	if (run == 0) return;
	const uint64_t va = ReadInfo(run, InfoVa), size = ReadInfo(run, InfoSize);
	g_run_frees.fetch_add(1, std::memory_order_relaxed);
	g_run_free_bytes.fetch_add(size, std::memory_order_relaxed);
	if (size != 0) NativeTex::OnTextureRelease(va, size);
}

// The pool-slice memcpy (0x900f26): memcpy's contract (returns dst).
void* KYTY_SYSV_ABI PoolCopy(void* destination, const void* source, size_t size) {
	g_moves.fetch_add(1, std::memory_order_relaxed);
	g_move_bytes.fetch_add(size, std::memory_order_relaxed);
	NativeTex::OnTextureMove(reinterpret_cast<uint64_t>(destination), reinterpret_cast<uint64_t>(source), size);
	return destination;
}

template <size_t N>
void Put(uint8_t*& p, const std::array<uint8_t, N>& bytes) {
	std::memcpy(p, bytes.data(), N);
	p += N;
}
void PutImm64(uint8_t*& p, uint64_t value) {
	std::memcpy(p, &value, sizeof(value));
	p += sizeof(value);
}

// The stub for ResizePages: push rdi,rsi,rdx,rcx,r8,r9; sub rsp,8 (16-byte aligned again); OnResizePages(rsi, rdx, rcx,
// r8d); restore; mov rax, target; jmp rax (rax is not an argument; the stack argument [rsp] of the caller is where it
// was, the return address is the original one).
size_t EmitResizeStub(uint8_t* p0, uint64_t target) {
	uint8_t* p = p0;
	Put(p, std::array<uint8_t, 8> {0x57, 0x56, 0x52, 0x51, 0x41, 0x50, 0x41, 0x51});
	Put(p, std::array<uint8_t, 4> {0x48, 0x83, 0xec, 0x08});
	Put(p, std::array<uint8_t, 12> {0x48, 0x89, 0xf7, 0x48, 0x89, 0xd6, 0x48, 0x89, 0xca, 0x44, 0x89, 0xc1});
	Put(p, std::array<uint8_t, 2> {0x48, 0xb8});
	PutImm64(p, reinterpret_cast<uint64_t>(&OnResizePages));
	Put(p, std::array<uint8_t, 2> {0xff, 0xd0});
	Put(p, std::array<uint8_t, 4> {0x48, 0x83, 0xc4, 0x08});
	Put(p, std::array<uint8_t, 8> {0x41, 0x59, 0x41, 0x58, 0x59, 0x5a, 0x5e, 0x5f});
	Put(p, std::array<uint8_t, 2> {0x48, 0xb8});
	PutImm64(p, target);
	Put(p, std::array<uint8_t, 2> {0xff, 0xe0});
	return static_cast<size_t>(p - p0);
}

// The stub for ReleasePages: push rdi,rsi; sub rsp,8; OnReleaseRun(rsi); restore; jmp target.
size_t EmitReleaseStub(uint8_t* p0, uint64_t target) {
	uint8_t* p = p0;
	Put(p, std::array<uint8_t, 6> {0x57, 0x56, 0x48, 0x83, 0xec, 0x08});
	Put(p, std::array<uint8_t, 3> {0x48, 0x89, 0xf7});
	Put(p, std::array<uint8_t, 2> {0x48, 0xb8});
	PutImm64(p, reinterpret_cast<uint64_t>(&OnReleaseRun));
	Put(p, std::array<uint8_t, 2> {0xff, 0xd0});
	Put(p, std::array<uint8_t, 6> {0x48, 0x83, 0xc4, 0x08, 0x5e, 0x5f});
	Put(p, std::array<uint8_t, 2> {0x48, 0xb8});
	PutImm64(p, target);
	Put(p, std::array<uint8_t, 2> {0xff, 0xe0});
	return static_cast<size_t>(p - p0);
}

bool PointCall(uint64_t site, uint64_t stub) {
	const int64_t displacement = static_cast<int64_t>(stub) - static_cast<int64_t>(site + 5);
	if (displacement != static_cast<int32_t>(displacement)) return false;
	std::array<uint8_t, 5> call {0xe8};
	const auto             rel = static_cast<int32_t>(displacement);
	std::memcpy(call.data() + 1, &rel, sizeof(rel));
	std::memcpy(reinterpret_cast<void*>(site), call.data(), call.size());
	return Common::VirtualMemory::FlushInstructionCache(site, call.size());
}

void ReportThread() {
	uint64_t last = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(10));
		const uint64_t c = g_commits.load(), r = g_releases.load(), f = g_run_frees.load(), s = g_run_shrinks.load(),
		               m = g_moves.load();
		if (c + r + f + s + m == last) continue;
		last = c + r + f + s + m;
		constexpr uint64_t MiB = 1024 * 1024;
		std::printf("[native-tex] VirtualMemoryGPU since start: %llu commits (%llu MiB), %llu releases + %llu run shrinks "
		            "(%llu MiB), %llu runs freed (%llu MiB), %llu slice copies (%llu MiB)\n",
		            static_cast<unsigned long long>(c), static_cast<unsigned long long>(g_commit_bytes.load() / MiB),
		            static_cast<unsigned long long>(r), static_cast<unsigned long long>(s),
		            static_cast<unsigned long long>(g_release_bytes.load() / MiB), static_cast<unsigned long long>(f),
		            static_cast<unsigned long long>(g_run_free_bytes.load() / MiB), static_cast<unsigned long long>(m),
		            static_cast<unsigned long long>(g_move_bytes.load() / MiB));
		std::fflush(stdout);
	}
}

} // namespace

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	if (!::Loader::NativeTex::HooksRequested()) return;
	if (program == nullptr || !Libs::Graphics::DemonsSouls::IsSupportedGame() || program->file_name.filename() != "eboot.bin" ||
	    program->mapped_size > CaveOffset || program->base_vaddr > UINT64_MAX - CaveOffset - CaveSize)
		return;
	const uint64_t base  = program->base_vaddr;
	uint64_t       where = 0;
	if (const char* failed = CheckStatic(base, &where)) {
		std::printf("[native-tex] streaming hooks NOT installed: eboot+0x%llx: %s (checked for 01.007.000)\n",
		            static_cast<unsigned long long>(where), failed);
		std::fflush(stdout);
		return;
	}
	const uint64_t cave = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    base + CaveOffset, CaveSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_native_tex_stream", true);
	if (cave != base + CaveOffset) {
		if (cave != 0) Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		std::printf("[native-tex] streaming hooks NOT installed: no cave memory at eboot+0x%llx\n",
		            static_cast<unsigned long long>(CaveOffset));
		std::fflush(stdout);
		return;
	}
	auto*        bytes   = reinterpret_cast<uint8_t*>(cave);
	const size_t resize  = EmitResizeStub(bytes, base + ResizePages);
	const size_t release = EmitReleaseStub(bytes + 0x80, base + ReleasePages);
	uint8_t* move = bytes + 0x100; // mov rax, PoolCopy; jmp rax
	Put(move, std::array<uint8_t, 2> {0x48, 0xb8});
	PutImm64(move, reinterpret_cast<uint64_t>(&PoolCopy));
	Put(move, std::array<uint8_t, 2> {0xff, 0xe0});
	if (resize > 0x80 || release > 0x80 || !Common::VirtualMemory::FlushInstructionCache(cave, 0x110)) {
		Libs::LibKernel::Memory::FreeGuestMemory(cave, CaveSize);
		std::printf("[native-tex] streaming hooks NOT installed: cave stubs\n");
		std::fflush(stdout);
		return;
	}
	uint32_t installed = 0;
	if (PointCall(base + ResizeCall, cave)) installed |= 1u;
	if (PointCall(base + RunResizeCall, cave)) installed |= 2u;
	if (PointCall(base + ReleaseCall, cave + 0x80)) installed |= 4u;
	if (PointCall(base + PoolCopyCall, cave + 0x100)) installed |= 8u;
	g_installed.store(installed, std::memory_order_release);
	std::printf("[native-tex] streaming hooks (mask %u of 15): VirtualMemoryGPU commit/decommit (eboot+0x%llx), run resize "
	            "(eboot+0x%llx), run free (eboot+0x%llx) notify the native texture path before the engine's own code "
	            "runs (notification only); the texture -> array-slice copy (eboot+0x%llx) goes through OnTextureMove\n",
	            installed, static_cast<unsigned long long>(ResizeCall), static_cast<unsigned long long>(RunResizeCall),
	            static_cast<unsigned long long>(ReleaseCall), static_cast<unsigned long long>(PoolCopyCall));
	std::fflush(stdout);
	if (installed != 0) std::thread(ReportThread).detach();
#else
	(void)program;
#endif
}

} // namespace DemonsSoulsNativeTexStream
} // namespace Loader
