// (audit1) KYTY_DS_WRITE_AUDIT=1: the audit of the "write hooks" design. See loader/demonsSoulsWriteAudit.h.
//
// Demon's Souls 01.007.000 (Ghidra, vaddr + the eboot's base at run time). What is hooked, and how:
//
//   create       eboot+0x909819  call 0x1d10480, the single exit of CBuffer::Create 0x909460 (rbx = rdi = the
//                CBuffer). A cave stub makes the call and then tells OnCreateAfter(CBuffer*): the descriptor's low
//                48 bits are the data address, +0x18 flags, +0x1c format, +0x20 bytes.
//   destroy      eboot+0x9093c0 and 0x909410 (the two CBuffer destroy functions; entry: mov rsi,[rdi+0x10]; test
//                rsi,rsi - 7 bytes, no relative operand, no jump into them): OnDestroy(CBuffer*).
//   decommit     eboot+0x7f9f70 (addr, size, op, ...): the engine's virtual-memory commit (op 0) / decommit (op 2);
//                its entry (push rbp; mov rbp,rsp; push r15 - 6 bytes): OnVirtualOp. Every decommit goes through it:
//                SetCurrentStreamingChunk's (the call at 0x9014f4) and the texture arrays' (0x9011c7).
//   texload      eboot+0x8fed17  call 0x813130 (stream, destination, bytes) in FTextureFactory 0x8fe750: the read of a
//                texture's data from its file. A cave stub tells OnTexLoadBefore, makes the call, tells OnTexLoadAfter.
//   deferred free eboot+0x903460 (backend, pointer): the deferred free every CBuffer / CTexture block goes through;
//                its entry (6 bytes as above): OnDeferredFree.
//   heap free    eboot+0x7fa380 (pointer, ...): the engine's free; its entry (6 bytes as above): OnHeapFree. Every
//                free of the game: one hash and one load unless the pointer may be a registered range's start.
//   linear copy  eboot+0x912e60: render4's entry patch (loader/demonsSoulsCopy.cpp), shared: OnLinearCopy.
//   memcpy sites no patch: DemonsSoulsCopy::CoherentMove (the guest memcpy's body) tells every copy with the eboot
//                call site's return address: 0x909550 (CBuffer::Create's copy of the initial data, the call at
//                0x90954b) and 0x90145e (SetCurrentStreamingChunk's copy of a mip chunk, the call at 0x901459).
//
// Each hook compares the bytes around what it replaces with 01.007.000's before it patches; one that differs is left
// out and said. The patches are written before any guest thread runs.
#include "loader/demonsSoulsWriteAudit.h"

#include "common/abi.h"
#include "common/guardedCopy.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "kernel/memory.h"
#include "loader/demonsSoulsMemMap.h"
#include "loader/demonsSoulsWriteAuditRegistry.h"
#include "loader/demonsSoulsWriteAuditStubs.h"
#include "loader/runtimeLinker.h"
#include "local/parpadeo5.h" // (parpadeo5: KYTY_PARPADEO5)
#include "local/parpadeo6.h" // (parpadeo6: KYTY_PARPADEO6)
#include "local/parpadeo8.h" // (parpadeo8, w21: lecturas peligrosas de los bits; KYTY_PARPADEO8)
#include "local/parpadeo7.h" // (parpadeo7: KYTY_PARPADEO7)
#include "local/juego3.h" // (juego3: KYTY_JUEGO3)
#include "local/serie1.h"    // (serie1: KYTY_SERIE1)
#include "local/raiz.h"      // (raiz12: KYTY_RAIZ_SETTLE)
#include "local/frame-nativo.h" // (FRAME-NATIVO) KYTY_FRAME_NATIVO
#include "local/nativo-cubierto.h" // (VIGILANCIA-DOBLE) KYTY_SIN_VIGILANCIA_DOBLE
#include "local/prot-menos.h"   // (PROT-MENOS) KYTY_PROT_MENOS: its counters in the [frame-nativo] line
#include "local/copia1.h"       // (FRAME-NATIVO) the flips counted (Copia1::g_frames)
#include "live-counters.h"      // (FRAME-NATIVO) UploadBytes
#include "local/tex-nativa.h"   // (VRAM-FREE) free del motor sobre memoria de texturas
#include "local/malla-nativa.h"  // (MALLA-NATIVA) KYTY_MALLA_NATIVA: lecturas de CPU de las mallas

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>

namespace Loader::DemonsSoulsWriteAudit {
namespace {

namespace Reg  = DemonsSoulsWriteAuditRegistry;
namespace Stub = DemonsSoulsWriteAuditStubs;

constexpr double MiB = 1024.0 * 1024.0;

// ---- The switches -----------------------------------------------------------------------------------------------
enum HookBit : uint32_t {
	HookCreate       = 0x01,
	HookDestroy      = 0x02,
	HookDecommit     = 0x04,
	HookTexLoad      = 0x08,
	HookDeferredFree = 0x10,
	HookHeapFree     = 0x20,
	HookLinearCopy   = 0x40,
	HookCopySites    = 0x80,
	// (policy1)
	HookAcbCopy     = 0x100,
	HookLayers      = 0x200,
	HookBink        = 0x400,
	HookStreamEntry = 0x800,
	HookReadback    = 0x1000, // (policy2: only with KYTY_DS_READBACK_HOOK=1)
	HookFrameAudit  = 0x2000, // (policy3: only with KYTY_DS_FRAME_AUDIT=1)
	HookSettle      = 0x4000, // (raiz12: only with KYTY_RAIZ=1 and KYTY_RAIZ_SETTLE=1 or 2)
	HookChunkBits   = 0x8000, // (parpadeo3: only with KYTY_PARPADEO_XPR=1..3)
	HookAll         = 0xffff,
};
uint32_t HookMask() noexcept {
	static const uint32_t mask = [] {
		const char* text = std::getenv("KYTY_DS_WRITE_AUDIT_HOOKS");
		if (text == nullptr || text[0] == 0) return static_cast<uint32_t>(HookAll);
		return static_cast<uint32_t>(std::strtoul(text, nullptr, 16)) & HookAll;
	}();
	return mask;
}
// (policy1) A variable that is one digit from 0 to `top` (anything else: 0). Read once each (statics below), at their
// first use: the loader's Execute, after the port set its variables (as KYTY_DS_WRITE_AUDIT).
int EnvLevel(const char* name, int top) noexcept {
	const char* text = std::getenv(name);
	if (text == nullptr || text[0] < '0' || text[0] > '9' || text[1] != 0) return 0;
	return (std::min)(text[0] - '0', top);
}
int SplitWanted() noexcept {
	static const int level = EnvLevel("KYTY_DS_NEIGHBOR_SPLIT", 2);
	return level;
}
int AnnouncedWanted() noexcept {
	static const int level = EnvLevel("KYTY_DS_WRITE_ANNOUNCED", 2);
	return level;
}
// (policy6) KYTY_DS_WRITE_FRAME=1/2, KYTY_DS_WRITE_FRAME_LOTS=N (R1's margin, default 2).
// (anuncio1) Unset, with KYTY_RAIZ=1 and KYTY_RAIZ_ANNOUNCE=1: 1 (2 with KYTY_RAIZ_VERIFY=1: the byte check of
// policy7 beside the write-watch audit). Set (=0 too), it is what it says.
int AnnounceWantedLevel() noexcept;
int FrameWanted() noexcept {
	static const int level = [] {
		const char* text = std::getenv("KYTY_DS_WRITE_FRAME");
		if (text != nullptr && text[0] != 0) return EnvLevel("KYTY_DS_WRITE_FRAME", 2);
		const int announce = AnnounceWantedLevel();
		// (VIGILANCIA-DOBLE, KYTY_SIN_VIGILANCIA_DOBLE=1 por defecto) Log 20261125-061906-403: DefaultTagg 23 878
		// protecciones/s (42 %), write-policy "no admitidas: frame" 1 336 966 y "reservas de frame que pidieron paginas de
		// vuelta" 5 318 430 en 180 s: la memoria de frame se vigila aunque el juego la anuncia (ganchos 0x811f60 Alloc,
		// 0xa08ae0 KickoffSubmitJob, 0xa188bc envio de lote; Ghidra 01.007.000: 0x811f60 = TaggedFrameAllocator Alloc,
		// cursores por hilo en +0xb8+t*0x30 y ranura compartida +0x478, en la vtable 0x25ad518). Politica de frame nivel 1
		// (sin la tabla de 32 MiB del nivel 2); la seguridad es la VERIFY por write watch del anuncio (nivel 2 abajo).
		if (::NativoCubierto::On() && std::getenv("KYTY_RAIZ_ANNOUNCE") == nullptr) return 1;
		if (announce > 0) return announce;
		// (FRAME-NATIVO) KYTY_FRAME_NATIVO (default 1): the frame policy on without the stale check (local/frame-nativo.h).
		return FrameNativo::Wanted() ? 1 : 0;
	}();
	return level;
}
uint64_t FrameLotsWanted() noexcept {
	static const uint64_t lots = [] {
		const int n = EnvLevel("KYTY_DS_WRITE_FRAME_LOTS", 9);
		return static_cast<uint64_t>(n == 0 ? 2 : n);
	}();
	return lots;
}
// (policy3) KYTY_DS_FRAME_AUDIT=1.
bool FrameAuditWanted() noexcept {
	static const bool wanted = EnvLevel("KYTY_DS_FRAME_AUDIT", 1) != 0;
	return wanted;
}
// (policy2) KYTY_DS_READBACK_HOOK=1.
bool ReadbackWanted() noexcept {
	static const bool wanted = EnvLevel("KYTY_DS_READBACK_HOOK", 1) != 0;
	return wanted;
}
bool ReadHook2Wanted() noexcept { // (READ-HOOK2)
	// (11-26 verde) encendido por defecto: validado en pruebas 62-69; KYTY_DS_READ_HOOK2=0 lo apaga.
	static const char* const v = std::getenv("KYTY_DS_READ_HOOK2");
	static const bool wanted = v != nullptr && v[0] == '1'; // (11-26 limpieza) apagado por defecto
	return wanted;
}
// (READ-HOOK2) 0x8e9760 por separado, APAGADO por defecto (KYTY_DS_READ_COPY12=1 lo enciende). Log
// 20261125-103354-397: 125310 entradas en 5 s (~25000/s) y solo 40 descargaron algo (0.03 %): el coste es entrar al
// emulador por cada copia de 12 bytes. 0xadabb0 (4636 entradas, 330 descargas) sigue encendido.
bool Copy12Wanted() noexcept {
	static const bool wanted = EnvLevel("KYTY_DS_READ_COPY12", 0) != 0;
	return wanted;
}
bool VerifyWanted() noexcept {
	static const bool wanted = EnvLevel("KYTY_DS_WRITE_VERIFY", 1) != 0;
	return wanted;
}
// The kinds the split trusts (Reg::TrustedKinds): KYTY_DS_WRITE_ANNOUNCED's level, 1 when only the split is set.
std::atomic<uint32_t> g_trust {0};
std::atomic<bool>     g_lines {false};  // KYTY_DS_WRITE_AUDIT=1: the "[write-audit]" and "[read-audit]" lines
std::atomic<bool>     g_verify {false}; // KYTY_DS_WRITE_VERIFY=1
std::atomic<bool>     g_readback {false}; // (policy2) KYTY_DS_READBACK_HOOK=1 and its two entries patched

// ---- The game (01.007.000) --------------------------------------------------------------------------------------
constexpr uint64_t CaveOffset = 0x8000000 + 16 * 0x4000; // (beside the other caves: idle +0, scene guard +0x8000,
constexpr uint64_t CaveSize   = 0x4000;                  //  camera +0x20000, GPU buffer pages +0x80000)

// (policy1) Checked in Ghidra and in the ELF's bytes (native/policy1/ghidra):
constexpr uint64_t AcbCopyEntry    = 0x914e10; // the compute contexts' linear copy (ctx, dst, dst dwords, src, src dwords)
constexpr uint64_t StreamEntry     = 0x901330; // CTexture::SetCurrentStreamingChunk(texture, chunk)
constexpr uint64_t BinkSubmitEntry = 0x9808a0; // BinkGPU's "the CPU wrote the frame" callback (ctx); called by pointer
constexpr uint64_t LayerReadSite   = 0x8fef15; // call 0x813130 (stream, layer, bytes) in the layer loader 0x8fedc0
constexpr uint64_t LayerConvSite   = 0x8feed1; // call 0x1d2fb70 (base, .., temp, bytes, .., 0, [rsp] index): converted
constexpr uint64_t LayerConvTarget = 0x1d2fb70;
constexpr uint64_t LayerSliceLoad  = 0x8fee53; // mov rsi,[rbp-0x310]: 0x8fedc0 keeps one slice's bytes there
// (policy3) KYTY_DS_FRAME_AUDIT. Checked in Ghidra and in the ELF (native/policy3/ghidra):
//   0x811f60: TaggedFrameAllocator's allocation (vtable +0x80; (this, size, align), the address in rax; no direct
//     call, only through the table). Every frame allocator with a name is one of these objects. (0x811d50, vtable
//     +0, only reserves a chunk for the thread's cursor ahead of time and returns nothing: not hooked.)
//   0xa08ae0 KickoffSubmitJob: hands the recording jobs launched since the last kickoff to a new lot.
//   0xa188bc: the call to 0x831a50 in SubmitJob 0xa18840, right after it waited for every job of its lot (the loop of
//     0x8351b0 above it) and before it submits the lot's contexts (0x911c90, 0x915540).
constexpr uint64_t FrameAllocB      = 0x811f60;
constexpr uint64_t FrameAllocFirst  = 0x811d50; // (policy7: the allocator's code, 0x811d50 .. 0x81222f, Ghidra)
constexpr uint64_t FrameAllocEnd    = 0x812230;
constexpr uint64_t KickoffEntry     = 0xa08ae0;
constexpr uint64_t LotSubmitSite    = 0xa188bc; // call 0x831a50
constexpr uint64_t LotSubmitTarget  = 0x831a50;
constexpr uint64_t GameToGpuObject  = 0x2f64b20;
// 0x811f60: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x48; mov rax,[rip+..]; mov ebx,edx; mov r13,rsi
constexpr uint8_t FrameAllocBHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83,
                                       0xec, 0x48, 0x48, 0x8b, 0x05, 0x48, 0x03, 0xec, 0x01, 0x89, 0xd3, 0x49, 0x89, 0xf5, 0x49};
// 0xa08ae0: the same pushes; sub rsp,0x218; mov r12,[rip+..]; mov r14d,edx
constexpr uint8_t KickoffHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81,
                                   0xec, 0x18, 0x02, 0x00, 0x00, 0x4c, 0x8b, 0x25, 0xc5, 0x97, 0xcc, 0x01, 0x41, 0x89, 0xd6};
// (raiz12) 0xa1f020 (object, pass, &table_2cd30[i]): push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12;
// push rbx; sub rsp,0x18; mov rax,[rdi+0x38]; test rax,rax; je +0xe; mov r9,[rsi+0x68] (Ghidra; no reference into its
// first 0x10 bytes). Its one caller, 0xab6080, waits for the index cull's job (0x8351b0 on [entry[1] + 8]) and then calls
// it for each object with rdx = the entry; it reads the cull's counters at *(entry[1] + 0x28) (local/raiz.h).
constexpr uint64_t XprReadEntry = 0xa1f020;
constexpr uint8_t  XprReadHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83,
                                    0xec, 0x18, 0x48, 0x8b, 0x47, 0x38, 0x48, 0x85, 0xc0, 0x74, 0x0e, 0x4c, 0x8b, 0x4e, 0x68};
constexpr uint64_t XprCounterBytes = 15000u * 0x14u; // (0xab6540: CBuffer::Create(.., 0x14, 15000, 0x804, entry[5]))
// (parpadeo3) 0xa1f560 (section, view, pass entry), called by the XPR index cull 0xab6540 ("XPR Index Cull") once per
// XPR section (Ghidra, projelf_par1; native/parpadeo3/ghidra): it reads ON THE CPU the section's chunk-visibility bits
//   bits = *(u64*)(*(view + 0x58) + *(u32*)(section + 0x198) * 0x2d0 + idx * 8), idx from the pass (+0x44, +0x48):
//   0 -> +0x48 + 4, 1 -> 1, 3 -> 2, 4 -> 3, 6 -> +0x48 + 0x18, 2 / 5 -> 0;  chunks = *(u16*)(*(section + 0x38) + 100)
// and records one triangle-cull dispatch (and so one draw) per chunk whose bit is set (0xada200 / 0xada6e0): a chunk
// whose bit reads 0 is not drawn this frame. Those bits are WRITTEN BY THE GPU: 0xa1f930 (idx 0..3, called by
// 0xa0b020 / 0xa0b480, job "XPRPreRender") and 0xa20750 (idx 4.., by 0xa0b120) take them from a frame allocator
// (0x7f9cf0, never written by the CPU), bind them as RW buffers (CBuffer 0x805) of a compute dispatch (0x1d0a900) and
// store their addresses in that table; the job then publishes a label with RELEASE_MEM (0x912960): *(view + 0x120) +
// 0x1c0 after 0xa1f930's dispatches, + 0x120 after 0xa20750's (0xa0de50 sets both to 0 before the job).
// Head: cmp qword [rdi+0x38],0 (5 bytes, moved: the stub runs it after the host call, so its flags reach the je);
// je +0x44; movzx eax,word [rdi+0x114]. One reference: the call at 0xab6c09.
constexpr uint64_t ChunkBitsEntry = 0xa1f560;
constexpr uint8_t  ChunkBitsHead[] = {0x48, 0x83, 0x7f, 0x38, 0x00, 0x74, 0x44, 0x0f, 0xb7, 0x87, 0x14, 0x01, 0x00, 0x00};
// (parpadeo5) 0xa1f290 (section, view, pass entry): the reader that runs (parpadeo4 saw 0xa1f560 never entered). Its one
// caller, 0xab6080, first waits on the CPU for *entry[0] == 1 (0x8351b0 at 0xab6107: the label XPRPreRender's
// RELEASE_MEM writes) and then calls it per section (0xab630a); it tail-jumps to 0xad8bc0 / 0xad92f0, which read the
// chunk bits on the CPU and record one DRAW_INDEX_INDIRECT per visible chunk (local/parpadeo5.h; native/parpadeo5/ghidra).
// Head: push r14; push rbx; mov rax,rdi (6 bytes moved, position-independent; the only reference into the first 0x20
// bytes is the call at 0xab630a; a jump lands at +0x20); mov rdi,[rdi+0x38]; test rdi,rdi; je +0x11; mov r9,[rsi+0x68];
// mov r8,rsi; cmp byte [r9+0xce2],0.
constexpr uint64_t ChunkDrawsEntry  = 0xa1f290;
constexpr uint8_t  ChunkDrawsHead[] = {0x41, 0x56, 0x53, 0x48, 0x89, 0xf8, 0x48, 0x8b, 0x7f, 0x38, 0x48, 0x85, 0xff, 0x74, 0x11,
                                       0x4c, 0x8b, 0x4e, 0x68, 0x49, 0x89, 0xf0, 0x41, 0x80, 0xb9, 0xe2, 0x0c, 0x00, 0x00, 0x00};
// 0xa188ae: mov rax,[rbx+0x10]; mov rcx,[rbx+0x18]; jmp -0x38; lea rdi,[rbp-0x38]; call 0x831a50; mov r15,[rbx+0x210]
constexpr uint8_t LotSubmitCall[] = {0x48, 0x8b, 0x43, 0x10, 0x48, 0x8b, 0x4b, 0x18, 0xeb, 0xc8, 0x48, 0x8d, 0x7d, 0xc8,
                                     0xe8, 0x8f, 0x91, 0xe1, 0xff, 0x4c, 0x8b, 0xbb, 0x10, 0x02, 0x00, 0x00};

// (policy2) The CBuffer readback accessors (write-hooks2 7.1): 0x909940 returns the data pointer (8 call sites, no
// jump), 0x909970 copies the data out (18 call sites). Both are leaves; their first 6 bytes are two whole
// instructions with no relative operand.
constexpr uint64_t ReadPointerEntry = 0x909940;
constexpr uint64_t ReadCopyEntry    = 0x909970;
// 0x909940: movzx ecx,word [rdi+4]; mov eax,[rdi]; shl rcx,0x20; or rax,rcx; ret; int3; int3
constexpr uint8_t ReadPointerHead[] = {0x0f, 0xb7, 0x4f, 0x04, 0x8b, 0x07, 0x48, 0xc1, 0xe1, 0x20, 0x48, 0x09, 0xc8, 0xc3, 0xcc, 0xcc};
// 0x909970: xor eax,eax; cmp dword [rdi+0x1c],0x1f; je +0x69; mov r8d,[rdi+4]; mov r9d,0xe10; bextr r9d,r8d,r9d
constexpr uint8_t ReadCopyHead[] = {0x31, 0xc0, 0x83, 0x7f, 0x1c, 0x1f, 0x74, 0x69, 0x44, 0x8b, 0x47, 0x04,
                                    0x41, 0xb9, 0x10, 0x0e, 0x00, 0x00, 0xc4, 0x42, 0x30, 0xf7, 0xc8};
// (readback2) 0xa2e6f0 (its one caller 0xa2e693): reads the list a compute pass wrote, from the buffer whose 48-bit
// address is at [rdi+0x28] (dword +0x28, word +0x2c): the count at +0x8000 (at most 0x400 used), the entries from
// +0x8004 (4 bytes each). Ghidra 01.007.000. Its entry: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12;
// push rbx; sub rsp,0x58; mov rax,[rsi+0x18] (6 bytes moved: push rbp; mov rbp,rsp; push r15).
constexpr uint64_t VisListEntry  = 0xa2e6f0;
constexpr uint64_t VisListOffset = 0x8000, VisListBytes = 4 + 0x400 * 4;
constexpr uint8_t  VisListHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54,
                                    0x53, 0x48, 0x83, 0xec, 0x58, 0x48, 0x8b, 0x46, 0x18};
// (READ-HOOK2, KYTY_DS_READ_HOOK2=1 por defecto) Las dos RVA que mas leen por fault en la tabla [read-audit] "otros"
// (log 20261125-090852-271: 0xadac62 x14354, 0x8e9760 x8725 de 33471 en 240 s). Ghidra (projelf_par1):
// 0xadabb0 (su unico llamador 0xa20cce): recorre una mascara de bits que la GPU escribio (DefaultTagg): rsi = la
//   mascara, [rdi] (dword) = numero de bits -> lee ((n + 63) / 64) * 8 bytes desde rsi. Entrada: push rbp; mov rbp,rsp;
//   push r15 (6 bytes movidos, sin operando relativo).
// 0x8e9760 (10 llamadores): copia 12 bytes de [rsi] a [rdi] (mov eax,[rsi+8]; mov [rdi+8],eax; mov rax,[rsi]; mov
//   [rdi],rax; ret). Se mueven las dos primeras (6 bytes, sin operando relativo ni flags).
constexpr uint64_t ChunkMaskEntry  = 0xadabb0;
constexpr uint8_t  ChunkMaskHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};
constexpr uint64_t Copy12Entry     = 0x8e9760;
constexpr uint8_t  Copy12Head[]    = {0x8b, 0x46, 0x08, 0x89, 0x47, 0x08, 0x48, 0x8b, 0x06, 0x48, 0x89, 0x07, 0xc3};
constexpr uint64_t RewrittenReturn = 0x9f67dc; // after the call to CBuffer::Create at 0x9f67d7 (the 0x20-byte table)
// 0x914e10: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x198; mov rax,[rip+..];
// mov [rbp-0x1b8],rsi; mov r12,rcx
constexpr uint8_t AcbCopyHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x98, 0x01, 0x00,
                                   0x00, 0x48, 0x8b, 0x05, 0x95, 0xd4, 0xdb, 0x01, 0x48, 0x89, 0xb5, 0x48, 0xfe, 0xff, 0xff, 0x49, 0x89, 0xcc};
// 0x901330: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x28; mov r12,[rip+..];
// lea rdx,[rip+..]; mov ebx,esi; mov r14,rdi
constexpr uint8_t StreamHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28, 0x4c,
                                  0x8b, 0x25, 0x78, 0x0f, 0xdd, 0x01, 0x48, 0x8d, 0x15, 0x21, 0x16, 0xb8, 0x01, 0x89, 0xf3, 0x49, 0x89, 0xfe};
// 0x9808a0: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x58; mov rcx,[rip+..];
// vpxor xmm0,xmm0,xmm0; mov rax,[rcx]; mov [rbp-0x30],rax; mov r14,[rdi+0x230]
constexpr uint8_t BinkHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48,
                                0x83, 0xec, 0x58, 0x48, 0x8b, 0x0d, 0x08, 0x1a, 0xd5, 0x01, 0xc5, 0xf9, 0xef, 0xc0,
                                0x48, 0x8b, 0x01, 0x48, 0x89, 0x45, 0xd0, 0x4c, 0x8b, 0xb7, 0x30, 0x02, 0x00, 0x00};
// 0x8fef02: mov eax,ebx; mov rdi,[rbp-0x3f0]; mov rdx,r14; imul rsi,rax; add rsi,r12; call 0x813130; mov rax,[rip+..]
constexpr uint8_t LayerReadCall[] = {0x89, 0xd8, 0x48, 0x8b, 0xbd, 0x10, 0xfc, 0xff, 0xff, 0x4c, 0x89, 0xf2, 0x48, 0x0f, 0xaf, 0xf0,
                                     0x4c, 0x01, 0xe6, 0xe8, 0x16, 0x42, 0xf1, 0xff, 0x48, 0x8b, 0x05, 0x9f, 0x33, 0xdd, 0x01};
// 0x8feeb4: mov rsi,[rbp-0x308]; lea r8,[rbp-0x380]; mov rdi,r12; mov rdx,r15; mov rcx,r14; xor r9d,r9d;
// mov [rsp],ebx; call 0x1d2fb70; mov rdi,r15; xor esi,esi
constexpr uint8_t LayerConvCall[] = {0x48, 0x8b, 0xb5, 0xf8, 0xfc, 0xff, 0xff, 0x4c, 0x8d, 0x85, 0x80, 0xfc, 0xff, 0xff, 0x4c, 0x89, 0xe7, 0x4c, 0x89, 0xfa,
                                     0x4c, 0x89, 0xf1, 0x45, 0x31, 0xc9, 0x89, 0x1c, 0x24, 0xe8, 0x9a, 0x0c, 0x43, 0x01, 0x4c, 0x89, 0xff, 0x31, 0xf6};
// 0x8fee53: mov rsi,[rbp-0x310]; mov eax,[rax+0xc]
constexpr uint8_t LayerSlice[] = {0x48, 0x8b, 0xb5, 0xf0, 0xfc, 0xff, 0xff, 0x8b, 0x40, 0x0c};
// 0x9f67ca: mov qword [rsp],0; vmovups [rbp-0x40],xmm0; call 0x909460
constexpr uint8_t RewrittenCall[] = {0x48, 0xc7, 0x04, 0x24, 0x00, 0x00, 0x00, 0x00, 0xc5, 0xf8, 0x11, 0x45, 0xc0, 0xe8, 0x84, 0x2c, 0xf1, 0xff};

// The game code that reads memory the GPU wrote (write-hooks2 7): [begin, end) of each function.
struct ReadSite {
	uint64_t begin, end;
	uint32_t group; // 0 CBuffer copy-out 0x909970, 1 the readers after 0x909940, 2 0xa2e6f0, 3 0xa1f560 (culling masks)
};
constexpr ReadSite ReadSites[] = {
    {0x909970, 0x9099e6, 0}, {0xa25130, 0xa25266, 1}, {0xa2bb00, 0xa2bc20, 1}, {0xa2f910, 0xa2fa03, 1}, {0xa34ab0, 0xa34b53, 1},
    {0xa4fd60, 0xa4fdcf, 1}, {0xa6a8b0, 0xa6a92b, 1}, {0xa8f2f0, 0xa8f85b, 1}, {0xa2e6f0, 0xa2e9ac, 2}, {0xa1f560, 0xa1f930, 3},
    // (parpadeo5) the chunk-bit readers that run: 0xad8bc0 (.. 0xad8fd8) and 0xad92f0 (.. 0xad9691), group 3 as well.
    {0xad8bc0, 0xad8fe0, 3}, {0xad92f0, 0xad96a0, 3},
};
constexpr uint32_t    ReadGroups = 5; // (4: any other code)
std::atomic<uint64_t> g_reads[ReadGroups] {};
std::atomic<uint64_t> g_rewritten_return {0}; // (absolute; 0 until Install found the code around the call)

constexpr uint64_t CreateFunction   = 0x909460;
constexpr uint64_t CreateCallSite   = 0x909819; // call 0x1d10480
constexpr uint64_t CreateCallTarget = 0x1d10480;
constexpr uint64_t CreateCopyReturn = 0x909550; // (after the call at 0x90954b)
constexpr uint64_t StreamCopyReturn = 0x90145e; // (after the call at 0x901459)
constexpr uint64_t DestroyA         = 0x9093c0;
constexpr uint64_t DestroyB         = 0x909410;
constexpr uint64_t VirtualOp        = 0x7f9f70;
constexpr uint64_t TexLoadCallSite  = 0x8fed17; // call 0x813130
constexpr uint64_t TexLoadTarget    = 0x813130;
constexpr uint64_t DeferredFreeEntry     = 0x903460;
constexpr uint64_t HeapFreeEntry         = 0x7fa380;

// The bytes 01.007.000 has (Ghidra listing, the same in the ELF) at and around what each hook replaces.
// 0x909460: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x48
constexpr uint8_t CreateHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};
// 0x909813: mov rdi,rbx; mov rsi,rax; call 0x1d10480; mov rax,[r12]; cmp rax,[rbp-0x30]
constexpr uint8_t CreateExit[] = {0x48, 0x89, 0xdf, 0x48, 0x89, 0xc6, 0xe8, 0x62, 0x6c, 0x40, 0x01, 0x49, 0x8b, 0x04, 0x24, 0x48, 0x3b, 0x45, 0xd0};
// 0x909542: mov edx,[rbx+0x20]; mov rdi,r15; mov rsi,r9; call 0x1e73d00 (memcpy's stub)
constexpr uint8_t CreateCopy[] = {0x8b, 0x53, 0x20, 0x4c, 0x89, 0xff, 0x4c, 0x89, 0xce, 0xe8, 0xb0, 0xa7, 0x56, 0x01};
// 0x901450: mov rdi,r12; mov rsi,rax; mov rdx,r15; call 0x1e73d00; mov r12,[rip+..]
constexpr uint8_t StreamCopy[] = {0x4c, 0x89, 0xe7, 0x48, 0x89, 0xc6, 0x4c, 0x89, 0xfa, 0xe8, 0xa2, 0x28, 0x57, 0x01, 0x4c, 0x8b, 0x25, 0x5b, 0x0e, 0xdd, 0x01};
// 0x9093c0 and 0x909410 (the same bytes): mov rsi,[rdi+0x10]; test rsi,rsi; jz +0x2a; test byte [rdi+0x19],4;
// jnz +0x24; push rbp; mov rbp,rsp; push rbx; push rax
constexpr uint8_t DestroyHead[] = {0x48, 0x8b, 0x77, 0x10, 0x48, 0x85, 0xf6, 0x74, 0x2a, 0xf6, 0x47, 0x19, 0x04, 0x75, 0x24, 0x55, 0x48, 0x89, 0xe5, 0x53, 0x50};
// 0x7f9f70: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x58; mov r14,[rip+..]
constexpr uint8_t VirtualOpHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x58, 0x4c, 0x8b, 0x35, 0x38, 0x83, 0xed, 0x01};
// 0x8fed14: mov rdx,r13; call 0x813130; add [rbx+0x27c],r13d
constexpr uint8_t TexLoadCall[] = {0x4c, 0x89, 0xea, 0xe8, 0x14, 0x44, 0xf1, 0xff, 0x44, 0x01, 0xab, 0x7c, 0x02, 0x00, 0x00};
// 0x903460: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x18; mov r14,rsi; mov rbx,rdi
constexpr uint8_t DeferredFreeHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
// 0x7fa380: push rbp; mov rbp,rsp; push r15; push r14; push r12; push rbx; sub rsp,0x30; mov r14,[rip+..]
constexpr uint8_t HeapFreeHead[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x30, 0x4c, 0x8b, 0x35, 0x2a, 0x7f, 0xed, 0x01};

uint64_t g_base = 0, g_size = 0; // the eboot (set by Install before g_on)
// The absolute return addresses of the two memcpy calls, once Install found their code to be 01.007.000's (0: not).
std::atomic<uint64_t> g_create_copy_return {0}, g_stream_copy_return {0};
std::atomic<bool>     g_copies {false}; // the guest's copies are looked at (the mask's bit)

// ---- Counters ---------------------------------------------------------------------------------------------------
enum Counter : uint32_t {
	CCreate, // CBuffer::Create calls seen
	CCreateData,
	CCreateGpu,
	CCreateWrap,
	CCreatePlain,
	CCreateSkipFlags, // flags 2 / 0x1000 / 0x400: per-frame memory by construction
	CCreateSkipFrame, // its memory is in a frame arena
	CCreateSkipEmpty, // no memory (address 0 or 0 bytes)
	CCreateUnreadable,
	CCreateNested, // a wrapper inside a registered range (that one stays)
	CReplaced,     // registered ranges a new one overlapped (memory reused without a removal seen)
	CDestroy,
	CDestroyRemoved,
	CCreateCopy, // CBuffer::Create's memcpy (by return address)
	CStreamCopy, // the streaming chunk memcpy (by return address)
	CTexLoad,
	CDeferredFree,
	CDeferredFreeRemoved,
	CHeapFreeRemoved,
	CDecommit,
	CDecommitRemoved,
	CLinearCopy,
	CLcSrcFrame,
	CLcSrcRegistered,
	CLcSrcUnknown,
	CLcDstRegistered,
	CLcDstFrame,
	CLcDstUnknown,
	CSweptRuns,
	CSweptFrameBytes,
	CSweptRegisteredBytes,
	CSweptUnknownBytes,
	// (policy1)
	CCreateRewrap,    // a wrapper (0x800) over a range registered already: announced again
	CCreateRewritten, // the 0x20-byte table (return address 0x9f67dc)
	CDestroyReleased, // a wrapper destroyed: its range stays, one live wrapper less
	CStreamOwned,     // streamed chunks registered with their texture's block
	CFreeOwned,       // ranges taken out because their owner block was freed
	CLinearCopyAcb,   // linear copies of the compute contexts (0x914e10), also in CLinearCopy
	CLcSrcNew,        // linear copy sources registered (outside the frame arenas and the registry)
	CLayerRead,       // texture array layers read from the file in place (0x8fef15)
	CLayerConv,       // ... converted from a temporary (0x8feed1)
	CLayerConvUnknown,
	CBink,
	CStaleSwept, // per-use ranges not announced for a while, taken out
	CStaleHit,   // faults inside one of those (counted as desconocido)
	// The split, had it been on (by the fault's page): its faults a level would have ended.
	CWouldNeighbor1, // vecino, the page's ranges all trusted at KYTY_DS_WRITE_ANNOUNCED=1
	CWouldNeighbor2, // ... only at =2
	CWouldNeighborNo, // ... not trusted at any level (GPU-written, plain, rewritten, Bink)
	CWouldPerUse2,   // por-uso, the page's ranges all trusted at =2
	CWouldPerUseNo,
	CWouldUnknown,   // desconocido: KYTY_DS_NEIGHBOR_SPLIT=2
	// The policies.
	PNoWatchNeighbor, // pages made no-watch: a write beside trusted ranges
	PNoWatchPerUse,   // ... inside a per-use range
	PNoWatchUnknown,  // ... in a page with no registered range
	PRefusedTracker,  // the tracker refused (GPU-written, refused before, no-watch already with another watcher)
	PKeptFrame,       // not wanted: frame memory
	PKeptInside,      // not wanted: inside a range written only through hooks (a FUGA) or one not trusted
	PKeptUntrusted,   // not wanted: the page holds a range not trusted
	PKeptUnknown,     // not wanted: no registered range in the page (needs KYTY_DS_NEIGHBOR_SPLIT=2)
	PAnnounced,       // announcements queued for the GPU thread
	PAnnouncedBytes,
	PRevertQueued,    // a range not trusted appeared where there may be no-watch pages
	PQueueDropped,    // the queue was full (the GPU thread then takes every no-watch page as written)
	PDirtyPages,      // no-watch pages the announcements made CPU-dirty
	PRevertedPagesHook,
	PRevertedPagesUse, // no-watch pages watched again: a consumer asked for bytes no hook covers
	PRevertedPagesGpu, // ... GPU work writes them
	PRevertedUses,
	PReadCovered,
	PEverything,
	VSampled,
	VSampledBytes,
	VSame,
	VAnnounced, // the range was announced, replaced or freed between the two looks
	VUnreadable,
	VMismatch,
	// (policy2) KYTY_DS_READBACK_HOOK
	RPointer,  // 0x909940 entered
	RCopy,     // 0x909970 entered
	RNeeded,   // ... with pages the GPU wrote (read watchers) in the buffer
	RDone,     // ... read back at once
	RDoneBytes,
	RRefused,  // ... not done (the reads fault as always)
	RVisList,  // (readback2) 0xa2e6f0 entered
	RVisDone,  // ... its list read back at once
	RMask,     // (READ-HOOK2) 0xadabb0 entered
	RMaskDone, // ... its mask read back at once
	RMaskEnt,     // (READ-HOOK2b) ... its 16-byte entries (read by 0xadad90 at 0xadada1) faulted
	RMaskEntDone, // ... and read back at once
	RCopy12,   // (READ-HOOK2) 0x8e9760 entered
	RCopy12Done,
	// (SIN-VIGILAR-GANCHO) read back by the hooks with no read watcher (m_noread pages: CPUHEAP-/DEFAULTTAGG-SIN-VIGILAR)
	RUnwCBuf, RUnwVis, RUnwMask, RUnwMaskEnt, RUnwCopy12, RUnwBytes,
	CWriteUntrusted, // (policy2) faults inside a range never trusted (Bink, the rewritten table): not leaks
	// (policy3) KYTY_DS_FRAME_AUDIT
	FAllocs,
	FAllocPages,
	FKickoffs,
	FSubmits,
	FPendingLost, // frame faults not judged: the pending list was full
	FTableFull,   // pages not noted: the page table was full
	FOrderBroken, // lot submits with no kickoff of their own seen (their faults are not judged)
	// (policy4)
	PRevertedPagesBda, // no-watch pages watched again: they lie in a buffer shaders read by address (BDA)
	PRevertedBdaUses,
	PKeptBink,         // not wanted: a Bink video is playing (no page is made no-watch meanwhile)
	PBinkRevertAll,    // a video began: every no-watch page was asked back
	// (policy5)
	PKeptFilter,       // not wanted: KYTY_DS_SPLIT_ONLY / KYTY_DS_SPLIT_EXCEPT (the page's allocator label)
	PFrameReserved,    // frame reservations that asked no-watch pages back
	// (policy6) KYTY_DS_WRITE_FRAME
	PNoWatchFrame, // frame pages made no-watch
	FrLots,        // lot submits that marked (R1)
	FrMarks,       // ranges marked (R1, R2, R3)
	FrMarkedBytes,
	FrSubmitMarks, // command buffer submits that marked (R2)
	FrCopyMarks,   // emulator copies into frame memory that marked (R3)
	FrCopySkipped, // (policy7) ... that did not: R1 marks them at a lot submit to come
	FrOverflow,    // everything taken as written (queue full, lots out of order)
	FrDrainPages,  // no-watch pages the marks made CPU-dirty
	VCleanNoted,   // (KYTY_DS_WRITE_FRAME=2) sampled no-watch pages hashed when they became clean
	VUseChecked,   // ... looked at again when GPU work was about to use them, still clean
	VUseStale,     // ... and their bytes had changed: the GPU would use stale data
	VUseStaleFrame,
	VUseRereserved, // (policy7) changed, but handed out again since: R1 marks it at the lot to come
	VUseOutside,    // (policy7) changed only outside the range the use reads
	VUseWide,       // (policy7) the use is too wide to tell (a BDA buffer, over 1 MiB)
	CounterCount
};
std::atomic<uint64_t> g_counters[CounterCount] {};
void Count(Counter counter, uint64_t value = 1) noexcept {
	g_counters[counter].fetch_add(value, std::memory_order_relaxed);
}

// The classes of a CPU write fault.
enum Class : uint32_t { KFrame, KAnnounced, KAnnouncedNet, KNeighbor, KUnknown, KLeak, KPerUse, ClassCount };
// (policy1) The frame faults by arena: 0 the backend's rings, 1 the render contexts' and small DCBs' blocks, 2 the rest
// of the Default TaggedArenas.
std::atomic<uint64_t> g_frame_faults[3] {};
constexpr uint32_t    FaultKinds = 3; // 0 write window, 1 write, 2 fresh first write
std::atomic<uint64_t> g_faults[FaultKinds][ClassCount] {};

// ---- The registry and the frame ranges --------------------------------------------------------------------------
std::atomic<Reg::Registry*> g_registry {nullptr};

// The per-frame arenas: sorted, disjoint [begin, end) pairs, two buffers and the one to read. Written by the audit's
// thread only (every 500 ms, from DemonsSoulsMemMap's snapshot: the rings, the Default TaggedArenas' blocks, the small
// DCBs' block). A reader takes the current buffer; the writer fills the other one and then switches, so a reader is
// in a buffer being written only if it was held up for the 500 ms between two refreshes (its sequence number says so).
constexpr size_t FrameMax = 4096;
struct FrameSet {
	std::atomic<uint32_t> seq {0}; // odd: being written
	std::atomic<uint32_t> count {0};
	std::atomic<uint64_t> begin[FrameMax] {};
	std::atomic<uint64_t> end[FrameMax] {};
};
struct FrameIndex {
	FrameSet         sets[2];
	std::atomic<int> current {-1};
};
// All of it; (policy1) and, for the faults' breakdown, the rings alone and the DCB blocks alone.
FrameIndex            g_frame_all, g_frame_ring, g_frame_ctx;
std::atomic<uint64_t> g_frame_bytes {0};

bool InSet(const FrameIndex& index, uint64_t address) noexcept {
	const int current = index.current.load(std::memory_order_acquire);
	if (current < 0) return false;
	const auto&    set = index.sets[current];
	const uint32_t seq = set.seq.load(std::memory_order_acquire);
	if ((seq & 1u) != 0) return false;
	const uint32_t count = (std::min)(set.count.load(std::memory_order_relaxed), static_cast<uint32_t>(FrameMax));
	// The last range that begins at or before the address.
	uint32_t low = 0, high = count;
	while (low < high) {
		const uint32_t middle = low + (high - low) / 2;
		if (set.begin[middle].load(std::memory_order_relaxed) <= address) low = middle + 1;
		else high = middle;
	}
	if (low == 0) return false;
	const bool inside = address < set.end[low - 1].load(std::memory_order_relaxed);
	return inside && set.seq.load(std::memory_order_acquire) == seq;
}
// (policy5) A page a frame allocator handed out at some time (the reservation hook, 0x811f60: installed with the
// split and with the frame audit). The memory map's ranges lag a new arena block by up to ~2.5 s; this does not.
bool FrameReservedPage(uint64_t address) noexcept;
bool InFrame(uint64_t address) noexcept {
	return InSet(g_frame_all, address) || FrameReservedPage(address);
}

// The bytes of [begin, end) inside the frame arenas.
uint64_t FrameOverlap(uint64_t begin, uint64_t end) noexcept {
	const int current = g_frame_all.current.load(std::memory_order_acquire);
	if (current < 0 || end <= begin) return 0;
	const auto&    set = g_frame_all.sets[current];
	const uint32_t seq = set.seq.load(std::memory_order_acquire);
	if ((seq & 1u) != 0) return 0;
	const uint32_t count = (std::min)(set.count.load(std::memory_order_relaxed), static_cast<uint32_t>(FrameMax));
	// The first range that ends after `begin`.
	uint32_t low = 0, high = count;
	while (low < high) {
		const uint32_t middle = low + (high - low) / 2;
		if (set.end[middle].load(std::memory_order_relaxed) <= begin) low = middle + 1;
		else high = middle;
	}
	uint64_t bytes = 0;
	for (uint32_t i = low; i < count; ++i) {
		const uint64_t range_begin = set.begin[i].load(std::memory_order_relaxed);
		const uint64_t range_end   = set.end[i].load(std::memory_order_relaxed);
		if (range_begin >= end) break;
		const uint64_t from = (std::max)(range_begin, begin), to = (std::min)(range_end, end);
		if (to > from) bytes += to - from;
	}
	return set.seq.load(std::memory_order_acquire) == seq ? (std::min)(bytes, end - begin) : 0;
}

// (The audit's thread.) The ranges of `labels` sorted, merged and published in `index`; their bytes. SIZE_MAX: the map
// has nothing to give yet (KYTY_MEMMAP=0, or not read yet).
uint64_t PublishFrame(FrameIndex& index, const char* const* labels, size_t label_count) {
	static uint64_t pairs[FrameMax * 2];
	// (Never destroyed: this thread is detached and may still run while the process exits.)
	static auto& ranges = *new std::vector<std::pair<uint64_t, uint64_t>>();
	const size_t found  = DemonsSoulsMemMap::RangesOf(labels, label_count, pairs, FrameMax);
	if (found == SIZE_MAX) return SIZE_MAX;
	ranges.clear();
	for (size_t i = 0; i < found; ++i)
		if (pairs[2 * i + 1] > pairs[2 * i]) ranges.emplace_back(pairs[2 * i], pairs[2 * i + 1]);
	std::sort(ranges.begin(), ranges.end());
	size_t merged = 0;
	for (size_t i = 0; i < ranges.size(); ++i) {
		if (merged != 0 && ranges[i].first <= ranges[merged - 1].second) {
			ranges[merged - 1].second = (std::max)(ranges[merged - 1].second, ranges[i].second);
		} else {
			ranges[merged++] = ranges[i];
		}
	}
	const int current = index.current.load(std::memory_order_relaxed);
	const int next    = current == 0 ? 1 : 0;
	auto&     set     = index.sets[next];
	set.seq.fetch_add(1, std::memory_order_acq_rel);
	uint64_t bytes = 0;
	for (size_t i = 0; i < merged; ++i) {
		set.begin[i].store(ranges[i].first, std::memory_order_relaxed);
		set.end[i].store(ranges[i].second, std::memory_order_relaxed);
		bytes += ranges[i].second - ranges[i].first;
	}
	set.count.store(static_cast<uint32_t>(merged), std::memory_order_relaxed);
	set.seq.fetch_add(1, std::memory_order_release);
	index.current.store(next, std::memory_order_release);
	return bytes;
}
bool RefreshFrame() {
	static constexpr const char* All[]  = {"ring", "DefaultTagg", "ctx"};
	static constexpr const char* Ring[] = {"ring"};
	static constexpr const char* Ctx[]  = {"ctx"};
	const uint64_t               bytes  = PublishFrame(g_frame_all, All, std::size(All));
	if (bytes == SIZE_MAX) return false;
	g_frame_bytes.store(bytes, std::memory_order_relaxed);
	(void)PublishFrame(g_frame_ring, Ring, std::size(Ring));
	(void)PublishFrame(g_frame_ctx, Ctx, std::size(Ctx));
	return true;
}

// ---- Small lock-free count tables (the reports' detail) ---------------------------------------------------------
template <size_t N>
struct CountTable {
	struct Slot {
		std::atomic<uint64_t> key {0};
		std::atomic<uint64_t> count {0};
	};
	Slot                  slots[N];
	std::atomic<uint64_t> dropped {0};
	void                  Add(uint64_t key) noexcept { // (key != 0)
        size_t i = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 40u) % N;
        for (size_t probe = 0; probe < 32; ++probe, i = (i + 1) % N) {
            uint64_t current = slots[i].key.load(std::memory_order_relaxed);
            if (current == 0 && slots[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) current = key;
            if (current == key) {
                slots[i].count.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        dropped.fetch_add(1, std::memory_order_relaxed);
	}
};
// tag << 60 | value. Sites (code addresses): 1 a direct write to unknown memory (rip), 2 a copy into unknown memory
// (the memcpy's call site), 3 a copy into a registered range from a site that is not hooked (the memcpy net).
// Granules (address >> 16): 1 unknown, 2 neighbour, 3 a linear copy's source outside the frame arenas and the registry.
constexpr uint64_t TagShift = 60;
constexpr uint64_t TagMask  = (uint64_t {1} << TagShift) - 1;
CountTable<4096>   g_sites;
CountTable<16384>  g_granules;
// (policy1) The code that reads GPU-written memory outside the known readers (a memcpy's call site when the read is
// the copy's source: value | 1 << 59). g_read_first: the first address each such site read (for its allocator label).
CountTable<2048> g_read_sites;
struct ReadFirst {
	std::atomic<uint64_t> site {0}, address {0};
};
constexpr size_t   ReadFirstMax = 512;
ReadFirst          g_read_first[ReadFirstMax];
constexpr uint64_t CopyFlag = uint64_t {1} << 59u;
void NoteSite(uint64_t tag, uint64_t code) noexcept {
	if (code != 0) g_sites.Add(tag << TagShift | (code & TagMask));
}
void NoteGranule(uint64_t tag, uint64_t address) noexcept {
	g_granules.Add(tag << TagShift | ((address >> 16u) + 1));
}

// ---- The FUGA table (the first 64 distinct range + writer pairs) ------------------------------------------------
struct Leak {
	uint64_t begin = 0, end = 0; // the registered range
	uint64_t address = 0;        // the first write seen
	uint64_t rip = 0, ret = 0;   // the writer, and the word at its stack top when it is not the eboot's code
	uint64_t count = 0;
	uint32_t flags = 0, format = 0;
	uint8_t  kind = 0, fault_kind = 0;
};
constexpr size_t LeakMax = 64;
std::mutex       g_leak_mutex;
Leak             g_leaks[LeakMax];
size_t           g_leak_count   = 0;
uint64_t         g_leak_dropped = 0;

bool InEboot(uint64_t address) noexcept {
	return address >= g_base && address - g_base < g_size;
}

void RecordLeak(uint64_t address, uint64_t rip, uint64_t rsp, int fault_kind, const Reg::Hit& hit) noexcept {
	std::lock_guard lock(g_leak_mutex);
	for (size_t i = 0; i < g_leak_count; ++i) {
		if (g_leaks[i].begin == hit.begin && g_leaks[i].rip == rip) {
			g_leaks[i].count++;
			return;
		}
	}
	if (g_leak_count == LeakMax) {
		g_leak_dropped++;
		return;
	}
	Leak& leak      = g_leaks[g_leak_count++];
	leak.begin      = hit.begin;
	leak.end        = hit.end;
	leak.address    = address;
	leak.rip        = rip;
	leak.count      = 1;
	leak.flags      = hit.flags;
	leak.format     = hit.format;
	leak.kind       = hit.kind;
	leak.fault_kind = static_cast<uint8_t>(fault_kind);
	// (Host or libc code - a memset, an HLE write: the word at the stack top is the caller when the writer is a leaf.)
	// (Read as the loader's fault handler reads it for its live trace: the faulting thread's own stack. No guarded
	// copy here: this runs inside the fault handler.)
	if (!InEboot(rip) && rsp >= 0x10000 && (rsp & 7u) == 0) leak.ret = *reinterpret_cast<const uint64_t*>(rsp);
}

// ---- This thread's copy / read in progress, and the texture reads of every thread -------------------------------
enum OpKind : uint64_t { OpNone = 0, OpCopy, OpStreamCopy, OpCreateCopy, OpTexLoad };
struct ThreadState {
	uint64_t op_begin = 0, op_end = 0, op_site = 0, op_kind = OpNone;
	uint64_t fault_rip = 0, fault_rsp = 0;
	int      texload_slot = -1;
	// (policy1)
	uint64_t op_source  = 0; // the source of the copy in progress (op_end - op_begin bytes)
	uint64_t stream_tex = 0; // the CTexture of the last SetCurrentStreamingChunk this thread entered
	uint64_t returns[8] {};  // the game's return addresses the stack stub replaced (EmitCallAroundStackStub)
	uint32_t return_depth = 0;
	uint8_t  split_reason = 0; // WantsNoWatch's reason, for NoteNoWatch
	bool     gpu_thread   = false; // (policy7) the emulator's GPU thread (NoteGpuThread)
	// (policy3) The frame reservations under way on this thread (EmitEntryAroundStub): the caller's return address,
	// the allocator and the bytes asked for.
	struct Reservation {
		uint64_t back = 0, object = 0, bytes = 0;
	};
	Reservation reservations[8] {};
	uint32_t    reservation_depth = 0;
};
thread_local ThreadState t_state;

// FTextureFactory's reads in progress (another thread may fill the destination: a job).
constexpr size_t      TexLoadSlots = 16;
std::atomic<uint64_t> g_texloads[TexLoadSlots][2] {}; // {begin, end}; end stored last, cleared first

bool InTexLoad(uint64_t address) noexcept {
	for (const auto& slot: g_texloads) {
		const uint64_t end = slot[1].load(std::memory_order_acquire);
		if (end == 0) continue;
		const uint64_t begin = slot[0].load(std::memory_order_acquire);
		if (address >= begin && address < end && slot[1].load(std::memory_order_acquire) == end) return true;
	}
	return false;
}

// ---- (policy1) What the hooks tell the GPU thread ----------------------------------------------------------------
// One bit per 4 MiB tracker region: some page of it was made no-watch at some time (never cleared). The hooks queue an
// announcement only for ranges that touch such a region: without the split nothing is ever queued.
constexpr uint64_t    RegionShift = 22;
constexpr size_t      RegionWords = (uint64_t {1} << (40u - RegionShift)) / 64;
std::atomic<uint64_t> g_nowatch_regions[RegionWords] {};
bool MayHaveNoWatch(uint64_t address, uint64_t size) noexcept {
	if (size == 0 || address + size < address) return false;
	const uint64_t first = address >> RegionShift, last = (address + size - 1) >> RegionShift;
	if (last >= RegionWords * 64) return false;
	for (uint64_t region = first; region <= last; ++region)
		if ((g_nowatch_regions[region / 64].load(std::memory_order_relaxed) >> (region % 64) & 1u) != 0) return true;
	return false;
}

// The queue: a ring under a mutex held for a few stores (game threads put, the GPU thread takes at each submission).
constexpr size_t RequestMax = 65536; // (policy4: was 16384; it overflowed once while a zone loaded)
std::mutex       g_request_mutex;
Request          g_requests[RequestMax];
size_t           g_request_head = 0, g_request_count = 0; // (under the mutex)
bool             g_request_lost = false;
std::atomic<size_t> g_request_pending {0};

void QueueRequest(uint64_t address, uint64_t size, bool revert) noexcept {
	if (!SplitOn() || !MayHaveNoWatch(address, size)) return;
	Count(revert ? PRevertQueued : PAnnounced);
	if (!revert) Count(PAnnouncedBytes, size);
	std::lock_guard lock(g_request_mutex);
	if (g_request_count != 0) {
		// (The same range announced again before the GPU thread came: once.)
		const Request& last = g_requests[(g_request_head + g_request_count - 1) % RequestMax];
		if (last.address == address && last.size == size && last.revert == revert) return;
	}
	if (g_request_count == RequestMax) {
		g_request_lost = true;
		Count(PQueueDropped);
		return;
	}
	Request& slot = g_requests[(g_request_head + g_request_count) % RequestMax];
	slot.address  = address;
	slot.size     = size;
	slot.revert   = revert;
	g_request_count++;
	g_request_pending.store(g_request_count, std::memory_order_relaxed);
}
// A range of `kind` was registered or announced at [address, address + size): its no-watch pages are written (a kind
// the split trusts) or must be watched again (any other kind).
void PolicyRange(uint64_t address, uint64_t size, uint8_t kind) noexcept {
	if (!SplitOn()) return;
	QueueRequest(address, size, ((1u << kind) & g_trust.load(std::memory_order_relaxed)) == 0);
}

// ---- (policy4) Who wrote each no-watch page, and the pages given back ----------------------------------------------
// The instruction whose write fault made a page no-watch (page number + 1 -> rip), for the report of the pages that
// had to be watched again. Open addressing; a full neighbourhood drops the entry.
constexpr size_t SplitRipSlots = size_t {1} << 17u;
struct SplitRip {
	std::atomic<uint64_t> key {0}, rip {0};
};
SplitRip g_split_rips[SplitRipSlots];
void NoteSplitRip(uint64_t page, uint64_t rip) noexcept {
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 47u) % SplitRipSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % SplitRipSlots) {
		uint64_t current = g_split_rips[i].key.load(std::memory_order_relaxed);
		if (current == 0 && g_split_rips[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) current = key;
		if (current == key) {
			g_split_rips[i].rip.store(rip, std::memory_order_relaxed);
			return;
		}
	}
}
uint64_t SplitRipOf(uint64_t page) noexcept {
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 47u) % SplitRipSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % SplitRipSlots) {
		const uint64_t current = g_split_rips[i].key.load(std::memory_order_relaxed);
		if (current == key) return g_split_rips[i].rip.load(std::memory_order_relaxed);
		if (current == 0) return 0;
	}
	return 0;
}
// The pages given back, by reason and writer (the first address of each pair). why: 0 a consumer asked for bytes no
// hook covers, 1 GPU work writes them, 2 a shader reads their buffer by address (BDA).
struct Reverted {
	uint64_t address = 0, rip = 0, uses = 0, pages = 0;
	int      why = 0;
};
constexpr size_t RevertedMax = 64;
std::mutex       g_reverted_mutex;
Reverted         g_reverted[RevertedMax];
size_t           g_reverted_count = 0;
// (policy6) KYTY_DS_WRITE_FRAME=2: the pages the GPU would have used stale, by the code whose write fault released them.
struct StaleUse {
	uint64_t address = 0, rip = 0, count = 0; // (address: the first byte that differs)
	bool     frame = false;
	uint32_t bytes = 0;   // (policy7) bytes that differ inside the use's range
	uint64_t use   = 0;   // (policy7) the use's size
};
constexpr size_t StaleMax = 32;
std::mutex       g_stale_mutex;
StaleUse         g_stale[StaleMax];
size_t           g_stale_count = 0;
// The registry clock's value at the last Bink callback (0: none yet).
std::atomic<uint32_t> g_bink_tick {0};
std::atomic<bool>     g_bink_seen {false};
// A video began: the GPU thread is to watch every no-watch page again (TakeRequests' `everything`).
std::atomic<bool> g_revert_all {false};
// (policy5) KYTY_DS_SPLIT_ONLY / KYTY_DS_SPLIT_EXCEPT: comma-separated allocator labels (DemonsSoulsMemMap::LabelAt's:
// cpuheap, cpubkt, cpupage, gpuheap, gpubkt, gpupage, DefaultTagg..., and "sin" for memory with no label). A
// diagnostic: only (all but) the pages of those allocators are made no-watch. Set at Install, read-only after.
char g_filter_list[256] = "";
int  g_filter_mode      = 0; // 0 none, 1 only, 2 except
bool FilterAllows(uint64_t address) noexcept {
	if (g_filter_mode == 0) return true;
	char label[24] = "";
	if (!DemonsSoulsMemMap::LabelAt(address, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin");
	bool         listed = false;
	const size_t length = std::strlen(label);
	for (const char* at = g_filter_list; *at != 0;) {
		const char*  end  = std::strchr(at, ',');
		const size_t size = end != nullptr ? static_cast<size_t>(end - at) : std::strlen(at);
		// (A listed name matches a label it is the beginning of: "cpu" takes cpuheap, cpubkt and cpupage.)
		if (size != 0 && size <= length && std::strncmp(at, label, size) == 0) listed = true;
		if (end == nullptr) break;
		at = end + 1;
	}
	return g_filter_mode == 1 ? listed : !listed;
}

// ---- (policy3) KYTY_DS_FRAME_AUDIT ---------------------------------------------------------------------------------
// The rule under test, for the frame arenas (native/policy3/NOTAS.md): "when a lot is submitted (SubmitJob, after it
// waited for the lot's jobs), every 4 KiB page that holds memory reserved since the kickoff BEFORE that lot's kickoff
// is taken as written". Measured here without changing anything: every reservation of a TaggedFrameAllocator is noted
// (its pages get the event count of then), every kickoff and lot submit too; a frame write fault waits for the next
// lot submit and is "covered" when its page's last reservation is not older than the kickoff before that lot's.
std::atomic<bool>     g_frame_audit {false};
std::atomic<uint64_t> g_fa_clock {1}; // one count per reservation, kickoff and lot submit
// The pages' last reservation: open addressing, key = page number + 1, value = clock << 8 | allocator index.
constexpr size_t FaPageSlots = size_t {1} << 21u;
struct FaPage {
	std::atomic<uint64_t> key {0}, value {0};
};
FaPage* g_fa_pages = nullptr; // (new[] at Install, never freed)
// The allocator objects seen (index 1...; 0 none).
constexpr size_t      FaAllocators = 48;
std::atomic<uint64_t> g_fa_allocators[FaAllocators] {};
uint64_t FaAllocatorIndex(uint64_t object) noexcept {
	for (size_t i = 1; i < FaAllocators; ++i) {
		uint64_t current = g_fa_allocators[i].load(std::memory_order_relaxed);
		if (current == object) return i;
		if (current == 0 && g_fa_allocators[i].compare_exchange_strong(current, object, std::memory_order_relaxed)) return i;
		if (current == object) return i;
	}
	return FaAllocators - 1; // (full: the last one stands for the rest)
}
void FaNotePage(uint64_t page, uint64_t value) noexcept {
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 43u) % FaPageSlots;
	for (size_t probe = 0; probe < 64; ++probe, i = (i + 1) % FaPageSlots) {
		uint64_t current = g_fa_pages[i].key.load(std::memory_order_relaxed);
		if (current == 0 && g_fa_pages[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) current = key;
		if (current == key) {
			// (The newest reservation stays: two threads never reserve in one page at once but for the shared cursor.)
			uint64_t old = g_fa_pages[i].value.load(std::memory_order_relaxed);
			while (old < value && !g_fa_pages[i].value.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
			}
			return;
		}
	}
	Count(FTableFull);
}
uint64_t FaPageValue(uint64_t page) noexcept;
bool     FrameReservedPage(uint64_t address) noexcept {
	return g_fa_pages != nullptr && FaPageValue(address >> 12u) != 0;
}
uint64_t FaPageValue(uint64_t page) noexcept {
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 43u) % FaPageSlots;
	for (size_t probe = 0; probe < 64; ++probe, i = (i + 1) % FaPageSlots) {
		const uint64_t current = g_fa_pages[i].key.load(std::memory_order_relaxed);
		if (current == key) return g_fa_pages[i].value.load(std::memory_order_relaxed);
		if (current == 0) return 0;
	}
	return 0;
}
// The kickoffs' clocks (a ring) and how many kickoffs and lot submits there were. (policy6) With each kickoff also the
// frame policy's run ring position then (FrRun).
constexpr size_t      FaKicks = 256;
std::atomic<uint64_t> g_fa_kick_clock[FaKicks] {};
std::atomic<uint64_t> g_fa_kick_runs[FaKicks] {};
std::atomic<uint64_t> g_fa_kickoffs {0}, g_fa_submits {0};
// The frame faults waiting for the next lot submit, and the results (under g_fa_mutex: the fault handler puts, the
// lot submit judges; both short).
struct FaPending {
	uint64_t address = 0, clock = 0, rip = 0;
	uint8_t  arena = 0;     // 0 rings, 1 DCB blocks, 2 the rest
	bool     copy  = false; // (policy6) made inside a guest memcpy the emulator runs (CoherentMove): rule R3
	bool     gpu   = false; // (policy7) made by the emulator's GPU thread (the command processor writing a label)
};
constexpr size_t FaPendingMax = 8192;
std::mutex       g_fa_mutex;
FaPending        g_fa_pending[FaPendingMax];
size_t           g_fa_pending_count = 0;
// Results by class: 0 rings, 1 DCB, 2 TaggedArena (not GameToGPU), 3 GameToGPU. (policy6) The rules, in the order the
// policy applies them (native/policy6/NOTAS.md): R1 the lot (kickoff n-1, then n-2: "margin"), R3 the emulator's copy,
// R2 a command buffer submitted after the write that holds the page (its range, its reservation, its context block).
constexpr uint32_t FaClasses = 4;
struct FaResult {
	std::atomic<uint64_t> faults {0}, covered {0}, covered_slack {0}, by_copy {0}, by_submit {0}, uncovered {0}, unreserved {0};
	// (policy7) Not the policy's business: the frame allocators' own free-list nodes, written into free memory by
	// 0x811d50 / 0x811f60 (nobody reads them but the allocator); the emulated GPU's own writes (its GPU thread).
	std::atomic<uint64_t> by_allocator {0}, by_gpu {0};
};
FaResult g_fa_results[FaClasses];
// The faults no lot rule covered, waiting for the command buffers submitted after them (FaResolveLate).
struct FaLate {
	uint64_t address = 0, clock = 0, rip = 0, allocator = 0;
	uint32_t tick = 0, kickoffs_old = 0;
	uint8_t  cls = 0;
	bool     unreserved = false;
};
constexpr size_t FaLateMax = 16384;
FaLate           g_fa_late[FaLateMax];
size_t           g_fa_late_count = 0;
struct FaSample {
	uint64_t address = 0, rip = 0, allocator = 0, count = 0;
	uint32_t kickoffs_old = 0; // kickoffs between the page's last reservation and the fault
	uint8_t  cls = 0;
	bool     unreserved = false;
};
constexpr size_t FaSampleMax = 48;
FaSample         g_fa_samples[FaSampleMax]; // (under g_fa_mutex)
size_t           g_fa_sample_count = 0;

// The clock of kickoff number `index` (0 when it is out of the ring or before the first one).
uint64_t FaKickClock(uint64_t index, uint64_t kickoffs) noexcept {
	if (index >= kickoffs || kickoffs - index > FaKicks) return 0;
	return g_fa_kick_clock[index % FaKicks].load(std::memory_order_relaxed);
}

// (policy6) The page's reservation counts as frame memory for the frame policy: a frame allocator handed it out in the
// last few seconds (g_fa_recent_clock: the clock as it was ~4 s ago, set by the audit's thread). A block an arena gave
// back long ago is not frame memory any more.
std::atomic<uint64_t> g_fa_recent_clock {0};
bool FrameRecentPage(uint64_t address) noexcept {
	if (g_fa_pages == nullptr) return false;
	const uint64_t value = FaPageValue(address >> 12u);
	return value != 0 && (value >> 8u) >= g_fa_recent_clock.load(std::memory_order_relaxed);
}

// A frame write fault (the fault handler): kept for the next lot submit.
void FaNoteFault(uint64_t address, uint64_t rip, int arena, bool copy, bool gpu) noexcept {
	std::lock_guard lock(g_fa_mutex);
	if (g_fa_pending_count == FaPendingMax) {
		Count(FPendingLost);
		return;
	}
	FaPending& pending = g_fa_pending[g_fa_pending_count++];
	pending.address    = address;
	pending.clock      = g_fa_clock.load(std::memory_order_relaxed);
	pending.rip        = rip;
	pending.arena      = static_cast<uint8_t>(arena);
	pending.copy       = copy;
	pending.gpu        = gpu;
}

// A lot is submitted (its jobs are done): the faults since the last lot submit are judged against R1 and R3; the
// others wait for R2 (FaResolveLate). `lot`: this lot's number; bounds: the clocks of kickoffs lot-1 and lot-2.
void FaJudge(uint64_t kickoffs, uint64_t bound, uint64_t bound_slack, bool broken) noexcept {
	std::lock_guard lock(g_fa_mutex);
	// (A submit with no kickoff of its own, or one too old for the ring: the order assumed does not hold; not judged.)
	if (broken) {
		Count(FOrderBroken);
		Count(FPendingLost, g_fa_pending_count);
		g_fa_pending_count = 0;
		return;
	}
	const uint32_t tick = Reg::Registry::Now();
	for (size_t i = 0; i < g_fa_pending_count; ++i) {
		const FaPending& pending = g_fa_pending[i];
		const uint64_t   value   = FaPageValue(pending.address >> 12u);
		const uint64_t   reserved = value >> 8u, allocator = g_fa_allocators[value & 0xffu].load(std::memory_order_relaxed);
		const uint32_t   cls = pending.arena == 0 ? 0u : pending.arena == 1 ? 1u : allocator == g_base + GameToGpuObject ? 3u : 2u;
		FaResult&        result = g_fa_results[cls];
		result.faults.fetch_add(1, std::memory_order_relaxed);
		const bool unreserved = value == 0;
		if (!unreserved && reserved >= bound) {
			result.covered.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		if (!unreserved && reserved >= bound_slack) {
			result.covered_slack.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		if (pending.copy) {
			result.by_copy.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		// (policy7) The allocator's free-list nodes (0x811d50..0x812230: the remnant's {next, prev, size} and the previous
		// node's next, write-hooks: TaggedFrameAllocator 0x811f60 at 0x812046..0x8120d8, 0x811d50 at 0x811ebd).
		if (pending.rip >= g_base + FrameAllocFirst && pending.rip < g_base + FrameAllocEnd) {
			result.by_allocator.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		if (pending.gpu) {
			result.by_gpu.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		// How many kickoffs passed between the page's last reservation and the fault.
		uint32_t old = 0;
		for (uint64_t k = kickoffs; k != 0 && kickoffs - k < FaKicks; --k) {
			const uint64_t clock = g_fa_kick_clock[(k - 1) % FaKicks].load(std::memory_order_relaxed);
			if (clock <= reserved) break;
			if (clock < pending.clock) old++;
		}
		if (g_fa_late_count < FaLateMax)
			g_fa_late[g_fa_late_count++] = {pending.address, pending.clock, pending.rip, allocator, tick, old, static_cast<uint8_t>(cls), unreserved};
		else Count(FPendingLost);
	}
	g_fa_pending_count = 0;
}

// (policy6) The command buffers the game submitted (NoteSubmitted, the HLE submit: before the GPU thread is given
// them), with the clock then, the reservation that holds each one's first page (its chunk: the pages around it with the
// same reservation value) and its context block (the memory map's). Under g_fa_submit_mutex.
struct FaSubmit {
	uint64_t clock = 0, begin = 0, end = 0, chunk_begin = 0, chunk_end = 0, block_begin = 0, block_end = 0;
};
constexpr size_t FaSubmitKeep = 65536;
std::mutex       g_fa_submit_mutex;
FaSubmit*        g_fa_submits_kept = nullptr; // (new[] with the page table)
uint64_t         g_fa_submit_next  = 0;

// The pages around `address` that a frame allocator handed out in one reservation (one value in the page table):
// [*begin, *end). False: the page has no reservation. At most MaxPages on each side. A small cache per thread.
bool ReservationOf(uint64_t address, uint64_t* begin, uint64_t* end) noexcept {
	constexpr uint64_t MaxPages = 4096;
	if (g_fa_pages == nullptr) return false;
	const uint64_t page  = address >> 12u;
	const uint64_t value = FaPageValue(page);
	if (value == 0) return false;
	struct Cached {
		uint64_t value = 0, first = 0, last = 0;
	};
	thread_local Cached cache[8];
	thread_local size_t cache_next = 0;
	for (const Cached& c: cache) {
		if (c.value == value && page >= c.first && page <= c.last && FaPageValue(c.first) == value && FaPageValue(c.last) == value) {
			*begin = c.first << 12u;
			*end   = (c.last + 1) << 12u;
			return true;
		}
	}
	uint64_t first = page, last = page;
	while (page - first < MaxPages && first != 0 && FaPageValue(first - 1) == value) --first;
	while (last - page < MaxPages && FaPageValue(last + 1) == value) ++last;
	cache[cache_next++ % std::size(cache)] = {value, first, last};
	*begin = first << 12u;
	*end   = (last + 1) << 12u;
	return true;
}

// (policy6) The frame policy's marks for the GPU thread: page-aligned ranges to take as written. A vector under a
// mutex; past FrMarkMax, `g_fr_everything` (every no-watch page is then taken as written: more uploads, never stale).
constexpr size_t      FrMarkMax = 262144;
std::mutex            g_fr_mark_mutex;
std::vector<Request>* g_fr_marks = nullptr; // (new at Install)
std::atomic<bool>     g_fr_everything {false};
std::atomic<size_t>   g_fr_pending {0};
void FrMark(uint64_t begin, uint64_t end) noexcept {
	if (end <= begin || !MayHaveNoWatch(begin, end - begin)) return;
	begin &= ~uint64_t {0xfff};
	end = (end + 0xfff) & ~uint64_t {0xfff};
	Count(FrMarks);
	Count(FrMarkedBytes, end - begin);
	std::lock_guard lock(g_fr_mark_mutex);
	auto& marks = *g_fr_marks;
	if (!marks.empty() && marks.back().address <= begin && marks.back().address + marks.back().size >= end) return; // (in the last)
	if (marks.size() >= FrMarkMax) {
		g_fr_everything.store(true, std::memory_order_relaxed);
		Count(FrOverflow);
		return;
	}
	marks.push_back({begin, end - begin, false});
	g_fr_pending.store(marks.size(), std::memory_order_relaxed);
}

// (policy6) The runs of pages the frame allocators handed out (rule R1 marks those of the last lots at each lot
// submit): each thread extends its open run while its reservations stay contiguous, and closes it into the ring.
struct FrRun {
	uint64_t begin = 0, end = 0, clock = 0; // (byte addresses, page-aligned; clock: the run's last reservation)
};
constexpr size_t FrRunMax  = 65536;
constexpr size_t FrSlots   = 128;
struct FrOpen {
	std::mutex m;
	FrRun      run;
};
FrOpen*               g_fr_open = nullptr; // (new[] at Install)
std::atomic<size_t>   g_fr_open_used {0};
std::mutex            g_fr_ring_mutex;
FrRun*                g_fr_ring = nullptr; // (new[] at Install)
std::atomic<uint64_t> g_fr_ring_next {0};
void FrNoteReservation(uint64_t address, uint64_t bytes, uint64_t clock) noexcept {
	thread_local int slot = -1;
	if (slot < 0) {
		const size_t index = g_fr_open_used.fetch_add(1, std::memory_order_relaxed);
		slot               = static_cast<int>(index < FrSlots ? index : FrSlots - 1); // (past the slots: one shared, locked)
	}
	const uint64_t begin = address & ~uint64_t {0xfff}, end = (address + bytes + 0xfff) & ~uint64_t {0xfff};
	FrOpen&        open  = g_fr_open[slot];
	std::lock_guard lock(open.m);
	if (open.run.end != 0 && begin <= open.run.end && end >= open.run.begin) {
		open.run.begin = (std::min)(open.run.begin, begin);
		open.run.end   = (std::max)(open.run.end, end);
		open.run.clock = clock;
		return;
	}
	if (open.run.end != 0) {
		std::lock_guard ring(g_fr_ring_mutex);
		const uint64_t  next = g_fr_ring_next.load(std::memory_order_relaxed);
		g_fr_ring[next % FrRunMax] = open.run;
		g_fr_ring_next.store(next + 1, std::memory_order_release);
	}
	open.run = {begin, end, clock};
}
// At a lot submit: every run closed since kickoff `from_kick` and every open run extended since `bound`, to FrMark.
void FrMarkLot(uint64_t from_kick, uint64_t kickoffs, uint64_t bound, bool broken) noexcept {
	Count(FrLots);
	if (broken || from_kick >= kickoffs || kickoffs - from_kick > FaKicks) {
		g_fr_everything.store(true, std::memory_order_relaxed); // (nothing to bound the marks with: all of it)
		Count(FrOverflow);
		return;
	}
	const uint64_t from = g_fa_kick_runs[from_kick % FaKicks].load(std::memory_order_relaxed);
	const uint64_t next = g_fr_ring_next.load(std::memory_order_acquire);
	if (next - from > FrRunMax) {
		g_fr_everything.store(true, std::memory_order_relaxed);
		Count(FrOverflow);
		return;
	}
	thread_local std::vector<std::pair<uint64_t, uint64_t>> runs;
	runs.clear();
	{
		std::lock_guard ring(g_fr_ring_mutex);
		for (uint64_t i = from; i < next; ++i) runs.emplace_back(g_fr_ring[i % FrRunMax].begin, g_fr_ring[i % FrRunMax].end);
	}
	const size_t used = (std::min)(g_fr_open_used.load(std::memory_order_relaxed), FrSlots);
	for (size_t i = 0; i < used; ++i) {
		std::lock_guard lock(g_fr_open[i].m);
		if (g_fr_open[i].run.end != 0 && g_fr_open[i].run.clock >= bound) runs.emplace_back(g_fr_open[i].run.begin, g_fr_open[i].run.end);
	}
	std::sort(runs.begin(), runs.end());
	uint64_t begin = 0, end = 0;
	for (const auto& [b, e]: runs) {
		if (end != 0 && b <= end) {
			end = (std::max)(end, e);
			continue;
		}
		if (end != 0) FrMark(begin, end);
		begin = b;
		end   = e;
	}
	if (end != 0) FrMark(begin, end);
}

// (policy6; the audit's thread, every 500 ms) Each fault no lot rule covered, once it is a second old, is looked for
// in the command buffers submitted AFTER it (clock): inside one's range, its reservation or its context block, it is
// covered by R2 (that submit marks all three); else it is NOT covered, and listed.
void FaResolveLate() {
	const uint32_t now = Reg::Registry::Now();
	static FaLate late[FaLateMax];
	size_t        late_count = 0;
	{
		std::lock_guard lock(g_fa_mutex);
		size_t left = 0;
		for (size_t i = 0; i < g_fa_late_count; ++i) {
			if (now - g_fa_late[i].tick >= 8) late[late_count++] = g_fa_late[i];
			else g_fa_late[left++] = g_fa_late[i];
		}
		g_fa_late_count = left;
	}
	if (late_count == 0) return;
	// The submits kept, newest first, down to the oldest fault's clock.
	uint64_t oldest = UINT64_MAX;
	for (size_t i = 0; i < late_count; ++i) oldest = (std::min)(oldest, late[i].clock);
	static auto& submits = *new std::vector<FaSubmit>();
	submits.clear();
	{
		std::lock_guard lock(g_fa_submit_mutex);
		for (uint64_t i = g_fa_submit_next; i != 0 && g_fa_submit_next - i < FaSubmitKeep; --i) {
			const FaSubmit& submit = g_fa_submits_kept[(i - 1) % FaSubmitKeep];
			if (submit.clock < oldest) break;
			submits.push_back(submit);
		}
	}
	for (size_t i = 0; i < late_count; ++i) {
		const FaLate& fault = late[i];
		bool          inside = false;
		for (const FaSubmit& submit: submits) {
			if (submit.clock <= fault.clock) break; // (newest first: the rest are older)
			const uint64_t a = fault.address;
			if ((a >= submit.begin && a < submit.end) || (a >= submit.chunk_begin && a < submit.chunk_end) ||
			    (a >= submit.block_begin && a < submit.block_end)) {
				inside = true;
				break;
			}
		}
		FaResult& result = g_fa_results[fault.cls < FaClasses ? fault.cls : 0];
		if (inside) {
			result.by_submit.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		(fault.unreserved ? result.unreserved : result.uncovered).fetch_add(1, std::memory_order_relaxed);
		std::lock_guard lock(g_fa_mutex);
		size_t found = 0;
		while (found < g_fa_sample_count && !(g_fa_samples[found].rip == fault.rip && g_fa_samples[found].allocator == fault.allocator &&
		                                      g_fa_samples[found].cls == fault.cls))
			++found;
		if (found == g_fa_sample_count) {
			if (g_fa_sample_count == FaSampleMax) continue;
			FaSample& sample  = g_fa_samples[g_fa_sample_count++];
			sample.address    = fault.address;
			sample.rip        = fault.rip;
			sample.allocator  = fault.allocator;
			sample.cls        = fault.cls;
			sample.unreserved = fault.unreserved;
		}
		g_fa_samples[found].count++;
		if (!fault.unreserved) g_fa_samples[found].kickoffs_old = (std::max)(g_fa_samples[found].kickoffs_old, fault.kickoffs_old);
	}
}

// (policy6) The frame policy's level (0 off) and its lot margin (R1 marks back to kickoff n - margin; default 2).
uint64_t g_fr_margin = 2;

void KYTY_SYSV_ABI OnLotSubmit(uint64_t /*rdi*/) {
	const bool audit = g_frame_audit.load(std::memory_order_relaxed), frame = FrameOn();
	if (!audit && !frame) return;
	g_fa_clock.fetch_add(1, std::memory_order_relaxed);
	const uint64_t submits  = g_fa_submits.fetch_add(1, std::memory_order_relaxed); // this lot's number, from 0
	const uint64_t kickoffs = g_fa_kickoffs.load(std::memory_order_acquire);
	Count(FSubmits);
	// Lot n was kicked off by kickoff n (lots are submitted in the order they were kicked off: each SubmitJob is a
	// job that depends on the one before, or runs inside its kickoff). R1's bound is kickoff n - 1, its margin n - 2.
	const uint64_t bound       = submits == 0 ? 0 : FaKickClock(submits - 1, kickoffs);
	const uint64_t bound_slack = submits < 2 ? 0 : FaKickClock(submits - 2, kickoffs);
	const bool     broken      = submits >= kickoffs || (submits != 0 && bound == 0);
	if (audit) FaJudge(kickoffs, bound, bound_slack, broken);
	if (frame) {
		const uint64_t from = submits >= g_fr_margin ? submits - g_fr_margin : 0;
		FrMarkLot(from, kickoffs, submits >= g_fr_margin ? FaKickClock(from, kickoffs) : 0, broken || submits < g_fr_margin);
	}
}
void KYTY_SYSV_ABI OnKickoff() {
	if (!g_frame_audit.load(std::memory_order_relaxed) && !FrameOn()) return;
	const uint64_t now   = g_fa_clock.fetch_add(1, std::memory_order_relaxed) + 1;
	// (The clock is stored before the count moves; kickoffs come from one thread at a time [H]: the render thread.)
	const uint64_t index = g_fa_kickoffs.load(std::memory_order_relaxed);
	g_fa_kick_clock[index % FaKicks].store(now, std::memory_order_relaxed);
	g_fa_kick_runs[index % FaKicks].store(g_fr_ring_next.load(std::memory_order_acquire), std::memory_order_relaxed);
	g_fa_kickoffs.store(index + 1, std::memory_order_release);
	Count(FKickoffs);
}

// ---- The hooks (game threads; SysV: reached from the cave stubs) ------------------------------------------------
template <typename T>
bool Peek(uint64_t address, T* out) noexcept {
	return address >= 0x10000 && address < (uint64_t {1} << 47u) &&
	       Common::GuardedCopy(out, reinterpret_cast<const void*>(address), sizeof(T)) == 0;
}

struct CBufferFields {
	uint64_t descriptor[2]; // +0x00 (the low 48 bits of the first: the data address)
	uint64_t block;         // +0x10 the memory it allocated itself, if any
	uint32_t flags;         // +0x18
	uint32_t format;        // +0x1c
	uint32_t bytes;         // +0x20
	uint32_t pad;
};
static_assert(sizeof(CBufferFields) == 0x28);
constexpr uint64_t AddressMask = 0xFFFFFFFFFFFFull;

// (The SysV entry points below hold no try/catch of their own: the ones that may allocate forward to these plain
// functions, kept out of line.)
[[gnu::noinline]] void CreateAfter(uint64_t buffer, uint64_t frame) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CCreate);
	CBufferFields fields {};
	if (!Peek(buffer, &fields)) {
		Count(CCreateUnreadable);
		return;
	}
	const uint64_t address = fields.descriptor[0] & AddressMask;
	const uint64_t bytes   = fields.bytes;
	try {
		// Flags 2 / 0x1000 (memory taken from RenderSingleFrameGPU at each use) and 0x400 (renamed to frame memory).
		if ((fields.flags & (0x2u | 0x1000u | 0x400u)) != 0) {
			Count(CCreateSkipFlags);
			if (address >= 0x10000) (void)registry->RemoveAt(address, true);
			return;
		}
		if (address < 0x10000 || bytes == 0) {
			Count(CCreateSkipEmpty);
			return;
		}
		if (InFrame(address)) {
			Count(CCreateSkipFrame);
			(void)registry->RemoveAt(address, true);
			return;
		}
		// (policy1) A wrapper (0x800) over memory registered already by the same wrap: announced again, under the
		// shared lock (most of the persistent creates of a frame: the engine wraps the same memory at every use).
		if ((fields.flags & 0x800u) != 0 && registry->Touch(address, bytes, Reg::CBufWrap, 1)) {
			Count(CCreateRewrap);
			PolicyRange(address, bytes, Reg::CBufWrap);
			return;
		}
		// (policy1) The 0x20-byte stream table made at eboot+0x9f67d7: six functions rewrite its 16-byte entries in
		// place with no hook (write-hooks2 N3). Told by Create's own return address: Create keeps rbp as its frame
		// pointer (push rbp; mov rbp, rsp), so it is at [rbp + 8].
		if (bytes == 0x20 && (fields.flags & 0x800u) == 0) {
			const uint64_t wanted = g_rewritten_return.load(std::memory_order_relaxed);
			uint64_t       from   = 0;
			if (wanted != 0 && frame >= 0x10000 && Peek(frame + 8, &from) && from == wanted) {
				const size_t replaced = registry->Add(address, bytes, Reg::CBufRewritten, fields.flags, fields.format, false, false);
				if (replaced > 1) Count(CReplaced, replaced - 1); // (one is the pending range of its own copy)
				Count(CCreateRewritten);
				PolicyRange(address, bytes, Reg::CBufRewritten);
				return;
			}
		}
		// (Created with its data, whatever its flags - 4 or 0x804 too, write-hooks2 N5 -: the copy was seen.)
		if (registry->Confirm(address, address + bytes, fields.flags, fields.format)) {
			Count(CCreateData);
			PolicyRange(address, bytes, Reg::CBufData);
			return;
		}
		const uint8_t kind = (fields.flags & 0x800u) != 0   ? Reg::CBufWrap
		                     : (fields.flags & 0x5u) != 0 ? Reg::CBufGpu
		                                                  : Reg::CBufPlain;
		bool          nested   = false;
		const size_t  replaced = registry->Add(address, bytes, kind, fields.flags, fields.format, false, kind == Reg::CBufWrap, &nested, 0,
		                                       kind == Reg::CBufWrap);
		if (nested) {
			Count(CCreateNested);
			// (A wrapper over part of a registered range: that range's bytes are the ones being used.)
			(void)registry->Announce(address);
			if (SplitOn()) QueueRequest(address, bytes, false);
			return;
		}
		Count(kind == Reg::CBufWrap ? CCreateWrap : kind == Reg::CBufGpu ? CCreateGpu : CCreatePlain);
		if (replaced != 0) Count(CReplaced, replaced);
		PolicyRange(address, bytes, kind);
	} catch (...) { // (no memory for the map's node: this buffer is not registered)
	}
}

void KYTY_SYSV_ABI OnCreateAfter(uint64_t buffer, uint64_t frame) {
	CreateAfter(buffer, frame);
}

void KYTY_SYSV_ABI OnDestroy(uint64_t buffer) {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CDestroy);
	CBufferFields fields {};
	if (!Peek(buffer, &fields)) return;
	const uint64_t address = fields.descriptor[0] & AddressMask;
	// (policy1) A wrapper (0x800) owns no memory: its range stays registered (the memory is its owner's, and the
	// engine wraps it again at the next use), with one live wrapper less.
	if ((fields.flags & 0x800u) != 0) {
		if (address >= 0x10000 && registry->Release(address, Reg::CBufWrap)) Count(CDestroyReleased);
		return;
	}
	bool removed = false;
	if (address >= 0x10000) removed = registry->RemoveAt(address);
	if (fields.block >= 0x10000 && fields.block != address) removed = registry->RemoveAt(fields.block) || removed;
	if (removed) Count(CDestroyRemoved);
}

// (policy1) A block was freed: the range that starts there, and the ranges it owns (a streamed texture's chunks).
void Freed(Reg::Registry* registry, uint64_t pointer, Counter counter) {
	if (registry->RemoveAt(pointer)) Count(counter);
	if (const size_t owned = registry->RemoveOwned(pointer); owned != 0) Count(CFreeOwned, owned);
}

void KYTY_SYSV_ABI OnDeferredFree(uint64_t /*backend*/, uint64_t pointer) {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CDeferredFree);
	if (pointer >= 0x10000 && registry->MayBeginAt(pointer)) Freed(registry, pointer, CDeferredFreeRemoved);
}

// (Every free of the game: the filter's one load unless the pointer may start a registered range.)
void KYTY_SYSV_ABI OnHeapFree(uint64_t pointer) {
	TexNativa::OnEngineFree(pointer); // (VRAM-FREE) CTexture destruida (DeletePendingItems 0x901f30)
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr || !registry->MayBeginAt(pointer) || pointer < 0x10000) return;
	Freed(registry, pointer, CHeapFreeRemoved);
}

void KYTY_SYSV_ABI OnVirtualOp(uint64_t address, uint64_t size, uint64_t op) {
	if (static_cast<uint32_t>(op) != 2) return; // (edx: 0 commit, 1 reserve, 2 decommit)
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CDecommit);
	if (address < 0x10000 || size == 0 || size > (uint64_t {1} << 40u)) return;
	const size_t removed = registry->RemoveIn(address, size);
	if (removed != 0) Count(CDecommitRemoved, removed);
}

// A read of a file (or a conversion) into [destination, destination + bytes) begins: registered as `kind`, and the
// range known to be in the hooks' hands until ReadEnd (another thread may fill it: a job).
[[gnu::noinline]] void ReadBegin(uint64_t destination, uint64_t bytes, uint8_t kind, uint64_t site) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	if (destination < 0x10000 || bytes == 0 || bytes > (uint64_t {1} << 40u)) return;
	auto& state = t_state;
	if (!InFrame(destination)) {
		try {
			const size_t replaced = registry->Add(destination, bytes, kind, 0, 0, false, false);
			if (replaced != 0) Count(CReplaced, replaced);
		} catch (...) {
		}
	}
	state.op_begin = destination;
	state.op_end   = destination + bytes;
	state.op_site  = site;
	state.op_kind  = OpTexLoad;
	state.texload_slot = -1;
	for (size_t i = 0; i < TexLoadSlots; ++i) {
		uint64_t expected = 0;
		if (g_texloads[i][0].compare_exchange_strong(expected, destination, std::memory_order_acq_rel)) {
			g_texloads[i][1].store(destination + bytes, std::memory_order_release);
			state.texload_slot = static_cast<int>(i);
			break;
		}
	}
}

void KYTY_SYSV_ABI OnTexLoadBefore(uint64_t /*stream*/, uint64_t destination, uint64_t bytes) {
	Count(CTexLoad);
	ReadBegin(destination, bytes, Reg::TexLoad, g_base + TexLoadCallSite + 5);
}

// The read ended: the range's generation moves (its bytes are final now) and its no-watch pages are written.
void ReadEnd() noexcept {
	auto& state = t_state;
	if (state.op_kind == OpTexLoad && state.op_end > state.op_begin) {
		if (auto* const registry = g_registry.load(std::memory_order_relaxed); registry != nullptr)
			(void)registry->Announce(state.op_begin);
		if (SplitOn()) QueueRequest(state.op_begin, state.op_end - state.op_begin, false);
	}
}

// (policy1) Texture array layers (write-hooks2 N2), the layer loader 0x8fedc0 (array, stream, layer, desc):
// the same format: the file is read straight into the layer, 0x813130(stream, base + slice * index, bytes) at 0x8fef15.
void KYTY_SYSV_ABI OnLayerReadBefore(uint64_t /*stream*/, uint64_t destination, uint64_t bytes) {
	Count(CLayerRead);
	ReadBegin(destination, bytes, Reg::TexLayer, g_base + LayerReadSite + 5);
}
// Another format: read into a temporary and converted by 0x1d2fb70(base, .., temp, bytes, desc, 0, index) at 0x8feed1,
// its only call. The layer is where the other path reads it: base + slice * index, with the slice's bytes in the
// loader's frame at [rbp - 0x310] (the mov at 0x8fee53, checked) and bytes = slice or 6 * slice (a cube's faces).
// `frame`: the game's return address (kept for OnLayerConvAfter) and then the original's stack arguments.
void KYTY_SYSV_ABI OnLayerConvBefore(uint64_t base, uint64_t /*rsi*/, uint64_t /*temp*/, uint64_t bytes, uint64_t /*r8*/,
                                     uint64_t /*r9*/, const uint64_t* frame, uint64_t rbp) {
	auto& state = t_state;
	state.returns[state.return_depth % std::size(state.returns)] = frame[0];
	state.return_depth++;
	Count(CLayerConv);
	const uint64_t index = static_cast<uint32_t>(frame[1]);
	uint64_t       slice = 0;
	if (rbp >= 0x10000 && Peek(rbp - 0x310, &slice) && slice != 0 && bytes % slice == 0 && bytes / slice <= 6 &&
	    index < (uint64_t {1} << 20u) && slice < (uint64_t {1} << 32u)) {
		ReadBegin(base + slice * index, bytes, Reg::TexLayer, g_base + LayerConvSite + 5);
	} else {
		Count(CLayerConvUnknown);
	}
}

void KYTY_SYSV_ABI OnTexLoadAfter() {
	ReadEnd();
	auto& state = t_state;
	if (state.texload_slot >= 0) {
		g_texloads[state.texload_slot][1].store(0, std::memory_order_release);
		g_texloads[state.texload_slot][0].store(0, std::memory_order_release);
		state.texload_slot = -1;
	}
	if (state.op_kind == OpTexLoad) {
		state.op_begin = state.op_end = state.op_site = 0;
		state.op_kind                                 = OpNone;
	}
}

// (The stack stub's second half: the game's return address back.)
uint64_t KYTY_SYSV_ABI OnLayerConvAfter() {
	OnTexLoadAfter();
	auto& state = t_state;
	state.return_depth--;
	return state.returns[state.return_depth % std::size(state.returns)];
}

// (policy1) CTexture::SetCurrentStreamingChunk's entry: the texture whose chunk the copy at 0x901459 fills. Its
// virtual block ([texture + 0x98], what its destructor 0x900bd0 gives to the deferred free) owns the chunk.
void KYTY_SYSV_ABI OnStreamEntry(uint64_t texture) {
	t_state.stream_tex = texture;
}

// (policy1) BinkGPU's callback 0x9808a0(ctx): the CPU wrote this frame's input buffers inside the video's block
// ([ctx + 0x218], [ctx + 0x220] bytes; write-hooks2 6). No hook sees those writes one by one: the block is
// registered as not trusted (its pages are never left no-watch).
[[gnu::noinline]] void BinkSubmit(uint64_t context) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CBink);
	// (policy4) A video is playing: WantsNoWatch makes no page no-watch for the next two seconds. And when it begins
	// (no callback in the last two seconds) every page that was no-watch is watched again: while a video plays the
	// tracking is the normal one everywhere.
	{
		const uint32_t now   = Reg::Registry::Now();
		const bool     begun = !g_bink_seen.load(std::memory_order_relaxed) ||
		                       now - g_bink_tick.load(std::memory_order_relaxed) > Reg::Registry::FreshTicks;
		g_bink_tick.store(now, std::memory_order_relaxed);
		g_bink_seen.store(true, std::memory_order_relaxed);
		if (begun && SplitOn()) {
			g_revert_all.store(true, std::memory_order_release);
			Count(PBinkRevertAll);
		}
	}
	uint64_t block = 0, bytes = 0;
	if (!Peek(context + 0x218, &block) || !Peek(context + 0x220, &bytes)) return;
	if (block < 0x10000 || bytes == 0 || bytes > (uint64_t {1} << 32u) || InFrame(block)) return;
	try {
		if (!registry->Touch(block, bytes, Reg::Bink, 0)) {
			const size_t replaced = registry->Add(block, bytes, Reg::Bink, 0, 0, false, false);
			if (replaced != 0) Count(CReplaced, replaced);
		}
		PolicyRange(block, bytes, Reg::Bink);
	} catch (...) {
	}
}
void KYTY_SYSV_ABI OnBinkSubmit(uint64_t context) {
	BinkSubmit(context);
}

// (policy2) KYTY_DS_READBACK_HOOK: the game entered a readback accessor with this CBuffer: it is about to read the
// buffer's data (the pointer 0x909940 returns, or the bytes 0x909970 copies out). Pages of it the GPU wrote would fault
// one by one as the reads reach them; all of the buffer is read back now instead (Memory::TryReadbackRange: the very
// readback a fault makes, once). Nothing else changes: a page not read back here still faults.
// (raiz12, KYTY_RAIZ_SETTLE) Then the buffer settled (Memory::RaizSettleRange): its deferred releases applied and any
// put-in of kept GPU bytes over it waited for - without it a buffer read back earlier (nothing for TryReadbackRange to
// do) could be read while the release thread was putting its bytes in: the page readable with its OLD bytes -.
// (SIN-VIGILAR-GANCHO) A range the reader is about to read with no read watcher: with CPUHEAP-/DEFAULTTAGG-SIN-VIGILAR its
// GPU-written pages stay readable with OLD bytes (no fault reads them back), so the hook reads them back itself.
bool ReadbackUnwatched(uint64_t address, uint64_t bytes, Counter which) noexcept {
	if (!::CpuheapSinVigilar::On() && !::DefaultTaggSinVigilar::On()) return false;
	if (!Libs::LibKernel::Memory::TryReadbackRangeUnwatched(address, bytes)) return false;
	Count(which);
	Count(RUnwBytes, bytes);
	return true;
}
[[gnu::noinline]] void ReadbackAccess(uint64_t buffer, Counter which) noexcept {
	Count(which);
	if (!g_readback.load(std::memory_order_relaxed)) return;
	CBufferFields fields {};
	if (!Peek(buffer, &fields)) return;
	const uint64_t address = fields.descriptor[0] & AddressMask;
	const uint64_t bytes   = fields.bytes;
	if (address < 0x10000 || bytes == 0 || bytes > (uint64_t {64} << 20u)) return;
	struct Settle {
		uint64_t address, bytes;
		~Settle() {
			if (Raiz::SettleMode() != 0 && bytes <= (uint64_t {16} << 20u))
				(void)Libs::LibKernel::Memory::RaizSettleRange(address, bytes, Raiz::SettleAccessor);
		}
	} settle {address, bytes};
	if (!Libs::LibKernel::Memory::ReadFaults(address, bytes)) {
		(void)ReadbackUnwatched(address, bytes, RUnwCBuf); // (SIN-VIGILAR-GANCHO)
		return;
	}
	Count(RNeeded);
	if (Libs::LibKernel::Memory::TryReadbackRange(address, bytes)) {
		Count(RDone);
		Count(RDoneBytes, bytes);
	} else {
		Count(RRefused);
	}
}
void KYTY_SYSV_ABI OnReadPointer(uint64_t buffer) {
	Juego3::Scope juego3_wait(Juego3::TagAccessor); // (juego3)
	if (MallaNativa::On() && buffer != 0) MallaNativa::OnCpuRead(MallaNativa::VsharpBase(buffer)); // (MALLA-NATIVA: antes de leer)
	ReadbackAccess(buffer, RPointer);
}
// (readback2) 0xa2e6f0 entered: the count and entries it reads ([rdi+0x28] + 0x8000, 0x1004 bytes) read back at once
// and settled, as ReadbackAccess does for a CBuffer.
void KYTY_SYSV_ABI OnVisList(uint64_t object) {
	Juego3::Scope juego3_wait(Juego3::TagAccessor);
	Count(RVisList);
	if (!g_readback.load(std::memory_order_relaxed)) return;
	uint32_t low = 0;
	uint16_t high = 0;
	if (!Peek(object + 0x28, &low) || !Peek(object + 0x2c, &high)) return;
	const uint64_t base = (static_cast<uint64_t>(high) << 32u) | low;
	if (base < 0x10000) return;
	const uint64_t address = base + VisListOffset;
	if (Libs::LibKernel::Memory::ReadFaults(address, VisListBytes)) {
		if (Libs::LibKernel::Memory::TryReadbackRange(address, VisListBytes)) Count(RVisDone);
	} else {
		(void)ReadbackUnwatched(address, VisListBytes, RUnwVis); // (SIN-VIGILAR-GANCHO)
	}
	if (Raiz::SettleMode() != 0) (void)Libs::LibKernel::Memory::RaizSettleRange(address, VisListBytes, Raiz::SettleAccessor);
}
// (READ-HOOK2) 0xadabb0 entered: the bit mask it walks (rsi, ((n + 63) / 64) * 8 bytes, n the dword at [rdi]) read
// back at once when it would fault, as OnVisList does.
void KYTY_SYSV_ABI OnChunkMask(uint64_t object, uint64_t mask) {
	Juego3::Scope juego3_wait(Juego3::TagAccessor);
	Count(RMask);
	if (!g_readback.load(std::memory_order_relaxed) || mask < 0x10000) return;
	uint32_t bits = 0;
	if (!Peek(object, &bits) || bits == 0 || bits > (1u << 24u)) return;
	const uint64_t bytes = ((static_cast<uint64_t>(bits) + 63) / 64) * 8;
	if (Libs::LibKernel::Memory::ReadFaults(mask, bytes)) {
		if (Libs::LibKernel::Memory::TryReadbackRange(mask, bytes)) Count(RMaskDone);
	} else {
		(void)ReadbackUnwatched(mask, bytes, RUnwMask); // (SIN-VIGILAR-GANCHO)
	}
	// (READ-HOOK2b) Ghidra 0xadabb0/0xadaee0: por cada bit llama a 0xadad90 con la entrada uVar11 * 0x10 + *(qword*)(rdi + 8);
	// 0xadad90 lee de ella un qword (+0), un word (+8) y un byte (+10) (0xadada1: x3826 lecturas por fault en cpuheap,
	// log 20261126-005934-791). Se descargan las n entradas de una vez, como la mascara.
	uint64_t entries = 0;
	if (!Peek(object + 8, &entries) || entries < 0x10000) return;
	const uint64_t entry_bytes = static_cast<uint64_t>(bits) * 0x10;
	if (Libs::LibKernel::Memory::ReadFaults(entries, entry_bytes)) {
		Count(RMaskEnt);
		if (Libs::LibKernel::Memory::TryReadbackRange(entries, entry_bytes)) Count(RMaskEntDone);
	} else {
		(void)ReadbackUnwatched(entries, entry_bytes, RUnwMaskEnt); // (SIN-VIGILAR-GANCHO)
	}
}
// (READ-HOOK2) 0x8e9760 entered: the 12 bytes it copies from rsi.
void KYTY_SYSV_ABI OnCopy12(uint64_t /*to*/, uint64_t from) {
	Count(RCopy12);
	if (!g_readback.load(std::memory_order_relaxed) || from < 0x10000) return;
	if (Libs::LibKernel::Memory::ReadFaults(from, 12)) {
		Juego3::Scope juego3_wait(Juego3::TagAccessor);
		if (Libs::LibKernel::Memory::TryReadbackRange(from, 12)) Count(RCopy12Done);
	} else if (::CpuheapSinVigilar::On() || ::DefaultTaggSinVigilar::On()) {
		Juego3::Scope juego3_wait(Juego3::TagAccessor);
		(void)ReadbackUnwatched(from, 12, RUnwCopy12); // (SIN-VIGILAR-GANCHO)
	}
}
void KYTY_SYSV_ABI OnReadCopy(uint64_t buffer) {
	Juego3::Scope juego3_wait(Juego3::TagAccessor); // (juego3)
	if (MallaNativa::On() && buffer != 0) MallaNativa::OnCpuRead(MallaNativa::VsharpBase(buffer)); // (MALLA-NATIVA: antes de copiar)
	ReadbackAccess(buffer, RCopy);
}

// (raiz12, KYTY_RAIZ_SETTLE) 0xa1f020's entry and return (EmitEntryAroundStub): before it reads the index cull's
// counters, they are settled (Memory::RaizSettleRange: read back, deferred releases applied, put-ins under way waited
// for); while it runs, its buffer counts as being read (Raiz::SettleNotePutIn: a put-in over it then is a game read of
// old bytes, counted). The caller's return address kept here (a stack: a nested call - none known - nests).
struct XprCall {
	uint64_t back;
	int      slot, outer; // (outer: the thread's Raiz::t_settle_slot before the call)
};
constexpr size_t             XprDepth = 64;
thread_local XprCall         t_xpr[XprDepth];
thread_local size_t          t_xpr_depth = 0;
void KYTY_SYSV_ABI OnXprReadBefore(uint64_t /*object*/, uint64_t /*pass*/, uint64_t entry, uint64_t /*rcx*/, uint64_t /*r8*/,
                                   uint64_t /*r9*/, const uint64_t* frame, uint64_t /*rbp*/) {
	Juego3::Scope juego3_wait(Juego3::TagAccessor); // (juego3: the raiz12 settle of the index cull's counters)
	if (t_xpr_depth >= XprDepth) {
		std::printf("[raiz12] 0xa1f020 nested %zu deep: not expected, stopping (KYTY_RAIZ_SETTLE=0 leaves the hook out)\n", t_xpr_depth);
		std::fflush(stdout);
		std::abort();
	}
	auto& call = t_xpr[t_xpr_depth++];
	call.back  = frame[0];
	call.slot  = -1;
	call.outer = Raiz::t_settle_slot;
	uint64_t table = 0, counters = 0;
	if (entry < 0x10000 || Common::GuardedCopy(&table, reinterpret_cast<const void*>(entry + 8), 8) != 0 || table < 0x10000 ||
	    Common::GuardedCopy(&counters, reinterpret_cast<const void*>(table + 0x28), 8) != 0 || counters < 0x10000)
		return;
	(void)Libs::LibKernel::Memory::RaizSettleRange(counters, XprCounterBytes, Raiz::SettleXpr);
	call.slot = Raiz::SettleBufferSlot(counters, counters + XprCounterBytes);
	if (call.slot >= 0) {
		Raiz::g_settle_buffers[call.slot].active.fetch_add(1, std::memory_order_seq_cst);
		Raiz::t_settle_slot = call.slot;
	}
}
uint64_t KYTY_SYSV_ABI OnXprReadAfter(uint64_t /*result*/) {
	const auto& call = t_xpr[--t_xpr_depth];
	if (call.slot >= 0) Raiz::g_settle_buffers[call.slot].active.fetch_sub(1, std::memory_order_seq_cst);
	Raiz::t_settle_slot = call.outer;
	return call.back;
}

// ---- (parpadeo3) KYTY_PARPADEO_XPR: 0xa1f560's read of the chunk-visibility bits (see ChunkBitsEntry) -------------------
// 1  observes, changes nothing: at each call, the label of the job that wrote the bits (1: the GPU thread translated
//    the RELEASE_MEM after the dispatches - the emulator writes labels when it translates them -), whether reading the
//    bits faults (Memory::ReadFaults: GPU-written bytes not read back yet), the bits set (read as the game will); per
//    (section, idx) a call whose bits drop to under half of the previous call's (>= 8 before) is a "CAIDA", told with
//    the next call's bits. A "[parpadeo3]" line every 5 s (with the pages of raiz11's near misses, by label).
// 2  as 1, and the bits settled first (Memory::RaizSettleRange, site SettleChunkBits: read back if the GPU owns them,
//    deferred releases applied, put-ins waited for; with KYTY_RAIZ_SETTLE=2 it only counts).
// 3  as 2, and first a wait (at most 4 ms) for the label to be 1 when it is not.
int ChunkBitsWanted() noexcept {
	static const int level = EnvLevel("KYTY_PARPADEO_XPR", 3);
	return level;
}
struct ChunkKey {
	uint64_t section = 0;
	int32_t  idx     = 0;
	uint32_t last_bits = 0, last_chunks = 0;
	uint64_t calls = 0;
	// A drop waiting for the next call (to tell whether it came back).
	bool     pending = false;
	uint32_t p_before = 0, p_now = 0, p_flags = 0;
	uint64_t p_label = 0, p_bits_at = 0, p_unix_ms = 0;
};
enum ChunkFlag : uint32_t { ChunkLabelNot1 = 1, ChunkFaults = 2, ChunkUnreadable = 4, ChunkWaitedLabel = 8, ChunkLabelLate = 16,
	                        ChunkSettleBusy = 32, ChunkSettlePending = 64, ChunkSettleReadback = 128, ChunkSettleWaited = 256 };
struct ChunkState {
	std::mutex                         mutex;
	std::array<ChunkKey, 2048>         keys {};
	uint32_t                           used = 0;
	std::array<uint64_t, 1024>         pages {}; // the bits' pages seen (for the near misses)
	uint32_t                           page_cursor = 0;
	uint64_t calls = 0, skipped = 0, label_not1 = 0, label_late = 0, waited = 0, faults = 0, unreadable = 0, drops = 0,
	         drops_label_not1 = 0, drops_faults = 0, drops_busy = 0, drops_pending = 0, settle_busy = 0, settle_pending = 0,
	         settle_readback = 0, bits_total = 0;
	// (parpadeo4) every entry of the hook, and why the ones that read no bits stopped.
	uint64_t entries = 0, stop_mesh0 = 0, stop_read = 0, stop_idx = 0, stop_table = 0;
	int32_t  last_kind = -1;
	uint32_t                           logged = 0;
	std::chrono::steady_clock::time_point last {};
	uint64_t                           near_seen = 0;
};
ChunkState& Chunks() {
	static ChunkState state;
	return state;
}
uint64_t UnixMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
template <typename T>
bool GuestRead(uint64_t address, T* out) {
	return address >= 0x10000 && Common::GuardedCopy(out, reinterpret_cast<const void*>(address), sizeof(T)) == 0;
}
void ChunkFlagsText(uint32_t f, char* out, size_t size) {
	std::snprintf(out, size, "%s%s%s%s%s%s%s%s%s", (f & ChunkLabelNot1) ? " etiqueta-no-1" : "",
	              (f & ChunkWaitedLabel) ? " (esperada)" : "", (f & ChunkLabelLate) ? " (no llego en 4 ms)" : "",
	              (f & ChunkFaults) ? " bytes-de-GPU-sin-leer(fault)" : "", (f & ChunkUnreadable) ? " ilegible" : "",
	              (f & ChunkSettleBusy) ? " put-in-EN-CURSO" : "", (f & ChunkSettlePending) ? " bytes-guardados-sin-meter" : "",
	              (f & ChunkSettleReadback) ? " leidos-de-vuelta" : "", (f & ChunkSettleWaited) ? " esperado-put-in" : "");
	if (out[0] == 0) std::snprintf(out, size, " nada");
}
// The 5 s line (under the mutex).
void ChunkReport(ChunkState& c) {
	const auto now = std::chrono::steady_clock::now();
	if (c.last == std::chrono::steady_clock::time_point {}) c.last = now;
	const double seconds = std::chrono::duration<double>(now - c.last).count();
	if (seconds < 5.0) return;
	c.last = now;
	// raiz11's near misses (game read faults just before a put-in) since the last line: their pages, by label.
	const uint64_t next = Raiz::g_near_page_next.load(std::memory_order_relaxed);
	const uint64_t from = next - (std::min<uint64_t>)(next - (std::min)(c.near_seen, next), Raiz::NearPageRing);
	struct Hit {
		uint64_t page = 0;
		uint32_t n = 0;
		bool     bits = false;
	};
	std::array<Hit, 32> hits {};
	uint32_t            nh = 0, in_bits = 0;
	for (uint64_t i = from; i < next; ++i) {
		const uint64_t page = Raiz::g_near_page[i % Raiz::NearPageRing].load(std::memory_order_relaxed);
		bool           bits = false;
		for (uint32_t k = 0; k < c.pages.size() && !bits; ++k) bits = c.pages[k] == page;
		in_bits += bits ? 1u : 0u;
		uint32_t h = 0;
		while (h < nh && hits[h].page != page) ++h;
		if (h == nh && nh < hits.size()) hits[nh++] = {page, 0, bits};
		if (h < nh) hits[h].n++;
	}
	c.near_seen = next;
	std::printf("[parpadeo4] %.1f s: 0xa1f560 entradas %llu; paradas: sin malla %llu, lectura fallida %llu, pase/indice fuera %llu "
	            "(ultimo tipo %d), tabla/bits invalidos %llu\n",
	            seconds, static_cast<unsigned long long>(c.entries), static_cast<unsigned long long>(c.stop_mesh0),
	            static_cast<unsigned long long>(c.stop_read), static_cast<unsigned long long>(c.stop_idx), c.last_kind,
	            static_cast<unsigned long long>(c.stop_table));
	c.entries = c.stop_mesh0 = c.stop_read = c.stop_idx = c.stop_table = 0;
	std::sort(hits.begin(), hits.begin() + nh, [](const Hit& a, const Hit& b) { return a.n > b.n; });
	std::printf("[parpadeo3] %.1f s: 0xa1f560 (bits de chunks del XPR, KYTY_PARPADEO_XPR=%d) llamadas %llu (sin bits %llu), "
	            "etiqueta del trabajo GPU NO a 1 al entrar %llu (esperadas %llu, sin llegar en 4 ms %llu), bytes de GPU sin leer "
	            "(fault) %llu, ilegibles %llu, asentado: put-in en curso %llu / guardados sin meter %llu / leidos de vuelta %llu; "
	            "bits puestos de media %.1f; CAIDAS %llu (con etiqueta no 1 %llu, con fault %llu, con put-in en curso %llu, con "
	            "guardados %llu); lecturas viejas estimadas de raiz11 (paginas): %llu, en paginas de estos bits %u",
	            seconds, ChunkBitsWanted(), static_cast<unsigned long long>(c.calls), static_cast<unsigned long long>(c.skipped),
	            static_cast<unsigned long long>(c.label_not1), static_cast<unsigned long long>(c.waited),
	            static_cast<unsigned long long>(c.label_late), static_cast<unsigned long long>(c.faults),
	            static_cast<unsigned long long>(c.unreadable), static_cast<unsigned long long>(c.settle_busy),
	            static_cast<unsigned long long>(c.settle_pending), static_cast<unsigned long long>(c.settle_readback),
	            c.calls != 0 ? static_cast<double>(c.bits_total) / static_cast<double>(c.calls) : 0.0,
	            static_cast<unsigned long long>(c.drops), static_cast<unsigned long long>(c.drops_label_not1),
	            static_cast<unsigned long long>(c.drops_faults), static_cast<unsigned long long>(c.drops_busy),
	            static_cast<unsigned long long>(c.drops_pending), static_cast<unsigned long long>(next - from), in_bits);
	for (uint32_t h = 0; h < nh && h < 6; ++h) {
		char label[96] = {};
		if (!::Loader::DemonsSoulsMemMap::LabelAt(hits[h].page, label, sizeof(label)) || label[0] == 0)
			std::snprintf(label, sizeof(label), "?");
		std::printf("%s 0x%llx x%u (%s%s)", h == 0 ? ": " : ",", static_cast<unsigned long long>(hits[h].page), hits[h].n, label,
		            hits[h].bits ? ", BITS DE CHUNKS" : "");
	}
	std::printf("\n");
	std::fflush(stdout);
	c.calls = c.skipped = c.label_not1 = c.label_late = c.waited = c.faults = c.unreadable = c.drops = 0;
	c.drops_label_not1 = c.drops_faults = c.drops_busy = c.drops_pending = 0;
	c.settle_busy = c.settle_pending = c.settle_readback = c.bits_total = 0;
}
ChunkKey* ChunkKeyOf(ChunkState& c, uint64_t section, int32_t idx) {
	const uint64_t h = ((section >> 4u) * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(idx) * 0x632BE59BD9B4E019ull) >> 53u;
	for (uint32_t probe = 0; probe < c.keys.size(); ++probe) {
		auto& k = c.keys[(h + probe) % c.keys.size()];
		if (k.section == section && k.idx == idx) return &k;
		if (k.section == 0) {
			if (c.used + 1 >= c.keys.size()) return nullptr;
			++c.used;
			k         = {};
			k.section = section;
			k.idx     = idx;
			return &k;
		}
	}
	return nullptr;
}
void KYTY_SYSV_ABI OnChunkBits(uint64_t section, uint64_t view, uint64_t pass) {
	const int level = ChunkBitsWanted();
	if (level == 0) return;
	try {
		// The bits' address and size, as 0xa1f560 finds them.
		uint64_t mesh = 0, table = 0, bits_at = 0, labels = 0;
		uint32_t sec_index = 0, kind = 0;
		int32_t  extra = 0;
		uint16_t chunks = 0;
		auto& c0   = Chunks();
		auto  stop = [&](uint64_t ChunkState::*counter) {
			const std::scoped_lock lock(c0.mutex);
			++c0.entries;
			++(c0.*counter);
			ChunkReport(c0);
		};
		if (!GuestRead(section + 0x38, &mesh)) return stop(&ChunkState::stop_read);
		if (mesh == 0) return stop(&ChunkState::stop_mesh0); // (0xa1f560 returns at once)
		if (!GuestRead(mesh + 100, &chunks) || !GuestRead(section + 0x198, &sec_index) || !GuestRead(pass + 0x44, &kind) ||
		    !GuestRead(pass + 0x48, &extra) || !GuestRead(view + 0x58, &table) || !GuestRead(view + 0x120, &labels))
			return stop(&ChunkState::stop_read);
		int32_t idx = -1;
		switch (kind) {
			case 0: idx = extra + 4; break;
			case 1: idx = 1; break;
			case 3: idx = 2; break;
			case 4: idx = 3; break;
			case 6: idx = extra + 0x18; break;
			case 2:
			case 5: idx = 0; break;
			default: idx = -1; break;
		}
		auto& c = Chunks();
		if (idx < 0 || idx >= 0x5a || chunks == 0 || table < 0x10000 ||
		    !GuestRead(table + uint64_t {sec_index} * 0x2d0 + static_cast<uint64_t>(idx) * 8, &bits_at) || bits_at < 0x10000) {
			const std::scoped_lock lock(c.mutex);
			++c.entries;
			++c.calls;
			++c.skipped;
			c.last_kind = static_cast<int32_t>(kind);
			if (idx < 0 || idx >= 0x5a || chunks == 0) ++c.stop_idx; else ++c.stop_table;
			ChunkReport(c);
			return;
		}
		const uint64_t bytes = ((uint64_t {chunks} + 63u) / 64u) * 8u;
		uint32_t       flags = 0;
		// The label of the job that wrote them (written when the GPU thread translates its RELEASE_MEM).
		const uint64_t label_at = labels >= 0x10000 ? labels + (idx < 4 ? 0x1c0u : 0x120u) : 0;
		uint64_t       label    = 0;
		const bool     label_ok = label_at != 0 && GuestRead(label_at, &label);
		if (!label_ok || static_cast<uint32_t>(label) != 1u) {
			flags |= ChunkLabelNot1;
			if (level >= 3 && label_ok) {
				flags |= ChunkWaitedLabel;
				const auto start = std::chrono::steady_clock::now();
				for (;;) {
					std::this_thread::yield();
					if (GuestRead(label_at, &label) && static_cast<uint32_t>(label) == 1u) break;
					if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(4)) {
						flags |= ChunkLabelLate;
						break;
					}
				}
			}
		}
		if (level >= 2) {
			const uint32_t settle = Libs::LibKernel::Memory::RaizSettleRange(bits_at, bytes, Raiz::SettleChunkBits);
			if ((settle & Libs::LibKernel::Memory::RaizSettleBusy) != 0) flags |= ChunkSettleBusy;
			if ((settle & Libs::LibKernel::Memory::RaizSettlePending) != 0) flags |= ChunkSettlePending;
			if ((settle & Libs::LibKernel::Memory::RaizSettleReadback) != 0) flags |= ChunkSettleReadback;
			if ((settle & Libs::LibKernel::Memory::RaizSettleWaited) != 0) flags |= ChunkSettleWaited;
		}
		if (Libs::LibKernel::Memory::ReadFaults(bits_at, bytes)) flags |= ChunkFaults;
		// The bits set, read as the game reads them (a page without access: ilegible - the game's read faults there). The
		// first 8,192 chunks (1 KiB here: this runs on the game thread's stack).
		uint64_t words[128] = {};
		const uint64_t n_words = (std::min<uint64_t>)(bytes / 8u, std::size(words));
		uint32_t       set     = 0;
		if (Common::GuardedCopy(words, reinterpret_cast<const void*>(bits_at), n_words * 8u) != 0) {
			flags |= ChunkUnreadable;
		} else {
			for (uint64_t w = 0; w < n_words; ++w) {
				uint64_t v = words[w];
				const uint64_t first = w * 64u;
				if (first + 64u > chunks) v &= (chunks > first) ? ((uint64_t {1} << (chunks - first)) - 1u) : 0u;
				set += static_cast<uint32_t>(std::popcount(v));
			}
		}
		const std::scoped_lock lock(c.mutex);
		++c.entries;
		++c.calls;
		c.bits_total += set;
		if (flags & ChunkLabelNot1) ++c.label_not1;
		if (flags & ChunkWaitedLabel) ++c.waited;
		if (flags & ChunkLabelLate) ++c.label_late;
		if (flags & ChunkFaults) ++c.faults;
		if (flags & ChunkUnreadable) ++c.unreadable;
		if (flags & ChunkSettleBusy) ++c.settle_busy;
		if (flags & ChunkSettlePending) ++c.settle_pending;
		if (flags & ChunkSettleReadback) ++c.settle_readback;
		{
			const uint64_t page = bits_at & ~uint64_t {0xfff};
			bool           known = false;
			for (uint32_t k = 0; k < c.pages.size() && !known; ++k) known = c.pages[k] == page;
			if (!known) c.pages[c.page_cursor++ % c.pages.size()] = page;
		}
		ChunkKey* key = ChunkKeyOf(c, section, idx);
		if (key != nullptr && (flags & ChunkUnreadable) == 0) {
			if (key->pending) {
				key->pending = false;
				if (c.logged++ < 64) {
					char ft[256], fl[96] = {};
					ChunkFlagsText(key->p_flags, ft, sizeof(ft));
					if (!::Loader::DemonsSoulsMemMap::LabelAt(key->p_bits_at, fl, sizeof(fl)) || fl[0] == 0)
						std::snprintf(fl, sizeof(fl), "?");
					std::printf("[parpadeo3] CAIDA de bits de chunks: unix_ms=%llu, seccion 0x%llx (indice %u), pase idx %d, %u chunks: "
					            "bits puestos antes %u, AHORA %u, siguiente llamada %u; etiqueta 0x%llx; al leer:%s; bits en 0x%llx (%s)\n",
					            static_cast<unsigned long long>(key->p_unix_ms), static_cast<unsigned long long>(section), sec_index, idx,
					            key->last_chunks, key->p_before, key->p_now, set, static_cast<unsigned long long>(key->p_label), ft,
					            static_cast<unsigned long long>(key->p_bits_at), fl);
					std::fflush(stdout);
				}
			}
			if (key->calls != 0 && key->last_bits >= 8 && set * 2 < key->last_bits) {
				++c.drops;
				if (flags & ChunkLabelNot1) ++c.drops_label_not1;
				if (flags & ChunkFaults) ++c.drops_faults;
				if (flags & ChunkSettleBusy) ++c.drops_busy;
				if (flags & ChunkSettlePending) ++c.drops_pending;
				key->pending   = true;
				key->p_before  = key->last_bits;
				key->p_now     = set;
				key->p_flags   = flags;
				key->p_label   = label;
				key->p_bits_at = bits_at;
				key->p_unix_ms = UnixMs();
			}
			key->last_bits   = set;
			key->last_chunks = chunks;
			key->calls++;
		}
		ChunkReport(c);
	} catch (...) {
	}
}

// ---- (parpadeo5) KYTY_PARPADEO5: 0xa1f290's entry (see ChunkDrawsEntry and local/parpadeo5.h) ---------------------------
// The bits index of a pass kind, as 0xa1f290 computes it (kind >= 7: -1).
int ChunkIndexOf(uint32_t kind, int32_t extra) noexcept {
	switch (kind) {
		case 0: return extra + 4;
		case 1: return 1;
		case 3: return 2;
		case 4: return 3;
		case 6: return extra + 0x18;
		case 2:
		case 5: return 0;
		default: return -1;
	}
}
// The bits words at `at` as the game will read them: when their pages are no-access (GPU bytes not in), the readback
// the game's read fault would make, made first. False: unreadable.
bool ReadChunkWords(uint64_t at, uint64_t* out, uint64_t words, uint32_t* flags) {
	if (at < 0x10000 || words == 0) return false;
	if (Libs::LibKernel::Memory::ReadFaults(at, words * 8u)) {
		*flags |= Parpadeo5::FlagNoAccess;
		(void)Libs::LibKernel::Memory::TryReadbackRange(at, words * 8u);
	}
	return Common::GuardedCopy(out, reinterpret_cast<const void*>(at), words * 8u) == 0;
}
// (parpadeo6) KYTY_PARPADEO6=1: the bits (and mask2 on the 0xad92f0 path, `mask2_at` != 0) settled before the game reads
// them, KYTY_RAIZ_SETTLE's mode 1 whatever that variable says (local/parpadeo6.h). Returns Memory::RaizSettle* flags.
uint32_t SettleChunkBitsFix(uint64_t bits_at, uint64_t mask2_at, uint64_t bytes, uint64_t* us_out) {
	const auto start  = std::chrono::steady_clock::now();
	uint32_t   flags  = Libs::LibKernel::Memory::RaizSettleRange(bits_at, bytes, Raiz::SettleChunkBits, 1);
	uint32_t   ranges = 1;
	// (parpadeo7) What the settle met of the readbacks begun at the label (diagnostics: local/parpadeo7.h's line).
	// (juego3) Also with KYTY_JUEGO3: its prefetch at the label shares those counters ([juego3] line).
	const bool prefetch = Parpadeo7::On() || Juego3::On();
	if (prefetch) Parpadeo7::NoteSettle(bits_at, bytes, (flags & Libs::LibKernel::Memory::RaizSettleReadback) != 0);
	if (mask2_at != 0) {
		const uint32_t more = Libs::LibKernel::Memory::RaizSettleRange(mask2_at, bytes, Raiz::SettleChunkBits, 1);
		if (prefetch) Parpadeo7::NoteSettle(mask2_at, bytes, (more & Libs::LibKernel::Memory::RaizSettleReadback) != 0);
		flags |= more;
		++ranges;
	}
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	Parpadeo6::Note(flags, ranges, us);
	Parpadeo6::MaybeReport();
	Juego3::NoteSettle(us, (flags & Libs::LibKernel::Memory::RaizSettleReadback) != 0); // (juego3)
	// (serie1) KYTY_SERIE1: the wall time from the culling's label to the bits in (the game waiting for the GPU to run it).
	if (Serie1::Mode() != 0 && (flags & Libs::LibKernel::Memory::RaizSettleReadback) != 0) Serie1::NoteSettleReadback();
	if (Parpadeo7::On()) Parpadeo7::MaybeReport();
	if (us_out != nullptr) *us_out = us;
	return flags;
}
// (parpadeo8, w21) Tras el asentado: lectura peligrosa (paginas ni de GPU ni leidas de vuelta en este frame) contada y,
// con KYTY_PARPADEO8=1 y un cull traducido sin correr en la GPU, esperado y asentado de nuevo (local/parpadeo8.h).
void Parpadeo8Check(uint64_t table, uint64_t bits_at, uint64_t mask2_at, uint64_t bytes, uint32_t flags) {
	namespace Mem           = Libs::LibKernel::Memory;
	const uint32_t gpu_mask = static_cast<uint32_t>(Mem::RaizSettleBusy) | static_cast<uint32_t>(Mem::RaizSettlePending) |
	                          static_cast<uint32_t>(Mem::RaizSettleReadback);
	(void)Parpadeo8::Check(table, bits_at, bytes, (flags & gpu_mask) != 0, [&]() noexcept {
		uint32_t f = Mem::RaizSettleRange(bits_at, bytes, Raiz::SettleChunkBits, 1);
		if (mask2_at != 0) f |= Mem::RaizSettleRange(mask2_at, bytes, Raiz::SettleChunkBits, 1);
		return (f & gpu_mask) != 0;
	});
	Parpadeo8::MaybeReport();
}
void KYTY_SYSV_ABI OnChunkDraws(uint64_t section, uint64_t view, uint64_t pass) {
	const int  level = Parpadeo5::Level();
	const bool fix   = Parpadeo6::On(); // (parpadeo6)
	if (level == 0 && !fix) return;
	Juego3::Scope juego3_wait(Juego3::TagSettle); // (juego3: the game thread in 0xa1f290's settle)
	try {
		// (8 KiB each: 65,535 chunks at most; thread_local, not on the game thread's stack.)
		thread_local std::array<uint64_t, 1024> bits {}, before {}, mask2 {}, drawn {}, old {};
		Parpadeo5::Call call;
		call.key = (section * 0x100000001b3ull) ^ (pass + 1u);
		uint64_t mesh = 0, flags_obj = 0;
		uint8_t  ce2  = 0;
		// (0xa1f290 returns at once: nothing recorded, not counted.)
		if (!GuestRead(section + 0x38, &mesh) || mesh == 0) return;
		if (!GuestRead(view + 0x68, &flags_obj) || !GuestRead(flags_obj + 0xce2, &ce2) || ce2 != 0) return;
		uint64_t config = 0, table = 0, label_at = 0, objs = 0, mask_at = 0;
		uint32_t sec_index = 0, pkind = 0, ckind = 0;
		int32_t  pextra = 0, cextra = 0;
		uint16_t chunks = 0;
		uint8_t  c4c = 0, cac = 0, ca8 = 0;
		const bool read = GuestRead(pass + 8, &config) && GuestRead(config + 0x44, &ckind) && GuestRead(config + 0x48, &cextra) &&
		                  GuestRead(config + 0x4c, &c4c) && GuestRead(pass + 0x20, &pkind) && GuestRead(pass + 0x24, &pextra) &&
		                  GuestRead(pass, &label_at) && GuestRead(section + 0x198, &sec_index) && GuestRead(view + 0x58, &table) &&
		                  GuestRead(mesh + 0x64, &chunks) && GuestRead(mesh + 0x50, &objs) && GuestRead(objs + 0x20, &mask_at) &&
		                  GuestRead(flags_obj + 0xcac, &cac) && GuestRead(flags_obj + 0xca8, &ca8);
		const int  i5 = ChunkIndexOf(pkind, pextra), i7 = ChunkIndexOf(ckind, cextra);
		const bool path_a = c4c == 0 && (pkind == 0 || pkind == 6 || cac != 1);
		int        slot   = i5;
		if (!path_a && ca8 == 1) {
			switch (pkind) {
				case 1: slot = 0x2e; break;
				case 2: slot = 0x2d; break;
				case 3: slot = 0x2f; break;
				case 4: slot = 0x30; break;
				default: break;
			}
		}
		const uint64_t row = table + uint64_t {sec_index} * 0x2d0u;
		uint64_t       bits_at = 0, mask2_at = 0;
		if (!read || chunks == 0 || table < 0x10000 || slot < 0 || (!path_a && i7 < 0) ||
		    !GuestRead(row + static_cast<uint64_t>(slot) * 8u, &bits_at) || bits_at < 0x10000 ||
		    (!path_a && (!GuestRead(row + static_cast<uint64_t>(i7) * 8u, &mask2_at) || mask2_at < 0x10000)) ||
		    !GuestRead(path_a ? mesh + 0x40 : config + 0x28, &call.base) || call.base == 0) {
			if (level == 0) return; // (parpadeo6 alone: nothing the game reads here that this hook can name)
			call.skipped = true;
			Parpadeo5::Note(call, 0);
			return;
		}
		const uint64_t words = (uint64_t {chunks} + 63u) / 64u;
		// (parpadeo7) What the game reads after this label, for the next frame's label: the bits slot (and mask2's).
		if (Parpadeo7::On()) {
			uint64_t base = 0;
			if (!GuestRead(view + 0x120, &base)) base = 0;
			Parpadeo7::Learn(view, label_at, base, sec_index, static_cast<uint32_t>(slot), static_cast<uint32_t>(words), table);
			if (!path_a) Parpadeo7::Learn(view, label_at, base, sec_index, static_cast<uint32_t>(i7), static_cast<uint32_t>(words), table);
		}
		// (parpadeo6, without the detector) The settle, and the game's code runs.
		if (level == 0) {
			const uint32_t settled = SettleChunkBitsFix(bits_at, path_a ? 0 : mask2_at, words * 8u, nullptr);
			Parpadeo8Check(table, bits_at, path_a ? 0 : mask2_at, words * 8u, settled); // (parpadeo8)
			return;
		}
		if (!path_a) call.flags |= Parpadeo5::FlagPathB;
		call.bits_at         = bits_at;
		call.bytes           = words * 8u;
		// The label 0xab6080 waited for (8 bytes compared with 1).
		uint64_t label = 0;
		if (label_at < 0x10000 || !GuestRead(label_at, &label) || label != 1u) call.flags |= Parpadeo5::FlagLabelNot1;
		// The previous frame's bits of this (section, pass), read again now: did they change after the game read them?
		int      later = 0;
		uint64_t o_at = 0, o_bytes = 0, o_hash = 0;
		if (Parpadeo5::PreviousRead(call.key, &o_at, &o_bytes, &o_hash) && o_at != bits_at && o_bytes != 0 && o_bytes <= old.size() * 8u &&
		    !Libs::LibKernel::Memory::ReadFaults(o_at, o_bytes) &&
		    Common::GuardedCopy(old.data(), reinterpret_cast<const void*>(o_at), o_bytes) == 0)
			later = Parpadeo5::Hash(old.data(), o_bytes / 8u) != o_hash ? 2 : 1;
		// Level 2 (or KYTY_PARPADEO6, parpadeo6's settle): the bytes before the settle, the settle, the bytes after (what
		// the game reads).
		bool had_before = false;
		if (level >= 2 || fix) {
			had_before = !Libs::LibKernel::Memory::ReadFaults(bits_at, call.bytes) &&
			             Common::GuardedCopy(before.data(), reinterpret_cast<const void*>(bits_at), call.bytes) == 0;
			uint32_t settle = 0;
			if (fix) {
				settle = SettleChunkBitsFix(bits_at, path_a ? 0 : mask2_at, call.bytes, &call.settle_us);
				Parpadeo8Check(table, bits_at, path_a ? 0 : mask2_at, call.bytes, settle); // (parpadeo8)
			} else {
				const auto start = std::chrono::steady_clock::now();
				settle           = Libs::LibKernel::Memory::RaizSettleRange(bits_at, call.bytes, Raiz::SettleChunkBits);
				if (!path_a) settle |= Libs::LibKernel::Memory::RaizSettleRange(mask2_at, call.bytes, Raiz::SettleChunkBits);
				call.settle_us = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
			}
			if (settle & Libs::LibKernel::Memory::RaizSettleBusy) call.flags |= Parpadeo5::FlagSettleBusy;
			if (settle & Libs::LibKernel::Memory::RaizSettlePending) call.flags |= Parpadeo5::FlagSettlePending;
			if (settle & Libs::LibKernel::Memory::RaizSettleReadback) call.flags |= Parpadeo5::FlagSettleReadback;
			if (settle & Libs::LibKernel::Memory::RaizSettleLeft) call.flags |= Parpadeo5::FlagSettleLeft;
		}
		call.readable = ReadChunkWords(bits_at, bits.data(), words, &call.flags);
		if (call.readable && !path_a) call.readable = ReadChunkWords(mask2_at, mask2.data(), words, &call.flags);
		if (!call.readable) {
			call.flags |= Parpadeo5::FlagUnreadable;
		} else {
			if (had_before && std::memcmp(before.data(), bits.data(), call.bytes) != 0) {
				call.flags |= Parpadeo5::FlagSettleChanged;
				if (fix) Parpadeo6::NoteChanged(); // (parpadeo6)
			}
			call.hash = Parpadeo5::Hash(bits.data(), words);
			// The draws 0xad8bc0 / 0xad92f0 record: chunk c < chunks with its bit (and mask2's on path B) and the mesh's
			// per-chunk pass mask (*(u32*)(mask_at + c*4) >> kind) & 1.
			uint32_t m[64];
			for (uint64_t w = 0; w < words; ++w) {
				uint64_t       v     = bits[w] & (path_a ? ~uint64_t {0} : mask2[w]);
				const uint64_t first = w * 64u;
				const uint64_t n     = (std::min<uint64_t>)(64u, chunks - first);
				if (n < 64u) v &= (uint64_t {1} << n) - 1u;
				drawn[w] = 0;
				if (v == 0) continue;
				if (mask_at < 0x10000 || Common::GuardedCopy(m, reinterpret_cast<const void*>(mask_at + first * 4u), n * 4u) != 0) {
					drawn[w] = v; // (the mesh's mask unreadable: every bit counted)
				} else {
					for (uint64_t b = 0; b < n; ++b)
						if (((v >> b) & 1u) != 0 && ((m[b] >> (pkind & 31u)) & 1u) != 0) drawn[w] |= uint64_t {1} << b;
				}
				call.draws += static_cast<uint32_t>(std::popcount(drawn[w]));
			}
			if (path_a) {
				call.drawn = drawn.data();
				call.words = words;
			}
		}
		Parpadeo5::Note(call, later);
	} catch (...) {
	}
}

// (policy3) TaggedFrameAllocator's allocation (this, bytes, alignment): before it runs and after, with its result.
void KYTY_SYSV_ABI OnFrameAllocBefore(uint64_t object, uint64_t bytes, uint64_t /*alignment*/, uint64_t /*rcx*/, uint64_t /*r8*/,
                                      uint64_t /*r9*/, const uint64_t* frame, uint64_t /*rbp*/) {
	auto& state = t_state;
	auto& slot  = state.reservations[state.reservation_depth % std::size(state.reservations)];
	slot.back   = frame[0];
	slot.object = object;
	slot.bytes  = bytes;
	state.reservation_depth++;
}
uint64_t KYTY_SYSV_ABI OnFrameAllocAfter(uint64_t address) {
	auto& state = t_state;
	state.reservation_depth--;
	const auto& slot = state.reservations[state.reservation_depth % std::size(state.reservations)];
	// (policy5: with the split too - its guard: frame memory is never left no-watch.)
	if (g_fa_pages != nullptr && address >= 0x10000 && slot.bytes != 0 && slot.bytes <= (uint64_t {64} << 20u)) {
		const uint64_t value = (g_fa_clock.fetch_add(1, std::memory_order_relaxed) + 1) << 8u | FaAllocatorIndex(slot.object);
		const uint64_t first = address >> 12u, last = (address + slot.bytes - 1) >> 12u;
		for (uint64_t page = first; page <= last; ++page) FaNotePage(page, value);
		Count(FAllocs);
		Count(FAllocPages, last - first + 1);
		if (FrameOn()) {
			// (policy6) R1's runs. A no-watch page handed out now is frame memory: R1 marks it at the lot's submit.
			FrNoteReservation(address, slot.bytes, value >> 8u);
		} else if (SplitOn() && MayHaveNoWatch(address, slot.bytes)) {
			// A page made no-watch before the allocator handed it out (the map did not know the block yet): back.
			Count(PFrameReserved);
			QueueRequest(address, slot.bytes, true);
		}
	}
	return slot.back;
}

// (policy1) The compute contexts' linear copy, eboot+0x914e10 (ctx, dst, dst dwords, src, src dwords): as the render
// contexts' (OnLinearCopy), whose entry DemonsSoulsCopy patches.
void KYTY_SYSV_ABI OnLinearCopyAcb(uint64_t /*context*/, uint64_t dst, uint64_t dst_dwords, uint64_t src, uint64_t src_dwords) {
	Count(CLinearCopyAcb);
	OnLinearCopy(dst, static_cast<int32_t>(dst_dwords), src, static_cast<int32_t>(src_dwords));
}

// ---- The patches ------------------------------------------------------------------------------------------------
struct Patch {
	const char* name     = "";
	uint64_t    rva      = 0;
	size_t      size     = 0;
	uint8_t     original[8] {};
	uint8_t     installed[8] {};
	bool        written = false;
	uint32_t    bit     = 0; // (policy1: its HookBit)
};
constexpr size_t PatchMax = 32; // (READ-HOOK2: +2)
Patch            g_patches[PatchMax];
size_t           g_patch_count = 0;
uint64_t         g_cave        = 0;
bool             g_linear_copy_hook = false;
bool             g_thread_started   = false;

// The loaded bytes at eboot+rva are `expected`. Otherwise says the first one that differs.
bool Matches(const char* hook, uint64_t rva, const uint8_t* expected, size_t size) {
	if (rva + size > g_size || !Libs::Graphics::HostMemoryRangeIsReadable(g_base + rva, size)) {
		std::printf("[write-audit]   %s: eboot+0x%llx no se puede leer\n", hook, static_cast<unsigned long long>(rva));
		return false;
	}
	const auto* code = reinterpret_cast<const uint8_t*>(g_base + rva);
	for (size_t i = 0; i < size; ++i) {
		if (code[i] == expected[i]) continue;
		std::printf("[write-audit]   %s: eboot+0x%llx tiene 0x%02x donde 01.007.000 tiene 0x%02x\n", hook,
		            static_cast<unsigned long long>(rva + i), static_cast<unsigned>(code[i]), static_cast<unsigned>(expected[i]));
		return false;
	}
	return true;
}

// Queues the patch of `size` bytes at eboot+rva: `opcode` (0xe8 call / 0xe9 jmp) to `target`, NOPs after it.
bool Queue(const char* name, uint64_t rva, size_t size, uint8_t opcode, uint64_t target, uint32_t bit) {
	if (g_patch_count == PatchMax || size > sizeof(Patch {}.original)) return false;
	Patch& patch = g_patches[g_patch_count];
	patch.name   = name;
	patch.bit    = bit;
	patch.rva    = rva;
	patch.size   = size;
	std::memcpy(patch.original, reinterpret_cast<const void*>(g_base + rva), size);
	if (!Stub::EmitPatch(patch.installed, size, opcode, g_base + rva, target)) return false;
	g_patch_count++;
	return true;
}

// ---- The reports (the audit's thread; Clear) --------------------------------------------------------------------
struct Totals {
	uint64_t counters[CounterCount] {};
	uint64_t faults[FaultKinds][ClassCount] {};
	uint64_t frame[3] {};          // (policy1)
	uint64_t reads[ReadGroups] {}; // (policy1)
	uint64_t kept = 0;             // (policy1: g_kept_pages)
};
std::atomic<uint64_t> g_kept_pages {0};
Totals ReadTotals() {
	Totals totals;
	for (uint32_t i = 0; i < CounterCount; ++i) totals.counters[i] = g_counters[i].load(std::memory_order_relaxed);
	for (uint32_t k = 0; k < FaultKinds; ++k)
		for (uint32_t c = 0; c < ClassCount; ++c) totals.faults[k][c] = g_faults[k][c].load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < 3; ++i) totals.frame[i] = g_frame_faults[i].load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < ReadGroups; ++i) totals.reads[i] = g_reads[i].load(std::memory_order_relaxed);
	totals.kept = g_kept_pages.load(std::memory_order_relaxed);
	return totals;
}
Totals Minus(const Totals& now, const Totals& before) {
	Totals delta;
	for (uint32_t i = 0; i < CounterCount; ++i) delta.counters[i] = now.counters[i] - before.counters[i];
	for (uint32_t k = 0; k < FaultKinds; ++k)
		for (uint32_t c = 0; c < ClassCount; ++c) delta.faults[k][c] = now.faults[k][c] - before.faults[k][c];
	for (uint32_t i = 0; i < 3; ++i) delta.frame[i] = now.frame[i] - before.frame[i];
	for (uint32_t i = 0; i < ReadGroups; ++i) delta.reads[i] = now.reads[i] - before.reads[i];
	delta.kept = now.kept - before.kept;
	return delta;
}

void Describe(uint64_t code, char* out, size_t size) {
	if (code == 0) std::snprintf(out, size, "?");
	else if (InEboot(code)) std::snprintf(out, size, "eboot+0x%llx", static_cast<unsigned long long>(code - g_base));
	else std::snprintf(out, size, "0x%llx", static_cast<unsigned long long>(code));
}

// "[write-audit] <seconds> s: faults N = frame a (x%) + anunciada b + vecino c + desconocido d + FUGA e; ..."
void PrintLine(const Totals& t, double seconds, const char* prefix, bool frame_known) {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	// (The faults the task counts: write-window and write. The fresh first writes are given apart.)
	uint64_t by_class[ClassCount] {};
	for (uint32_t c = 0; c < ClassCount; ++c) by_class[c] = t.faults[0][c] + t.faults[1][c];
	uint64_t window = 0, write = 0, fresh = 0, total = 0;
	for (uint32_t c = 0; c < ClassCount; ++c) {
		window += t.faults[0][c];
		write += t.faults[1][c];
		fresh += t.faults[2][c];
		total += by_class[c];
	}
	const auto percent = [&](uint64_t n) { return total == 0 ? 0.0 : 100.0 * static_cast<double>(n) / static_cast<double>(total); };
	uint64_t   ranges = 0, bytes = 0;
	for (uint8_t k = 0; k < Reg::KindCount; ++k) {
		ranges += registry->Count(k);
		bytes += registry->Bytes(k);
	}
	const auto& c    = t.counters;
	const auto  u    = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	char        frame[48];
	if (frame_known) std::snprintf(frame, sizeof(frame), "%.0f MiB", static_cast<double>(g_frame_bytes.load(std::memory_order_relaxed)) / MiB);
	else std::snprintf(frame, sizeof(frame), "? (sin mapa: KYTY_MEMMAP=0 o aun sin leer)");
	std::printf("[write-audit] %s%.1f s: faults %llu = frame %llu (%.1f%%) + anunciada %llu + vecino %llu + desconocido %llu + FUGA %llu + por-uso %llu; "
	            "registro: %llu rangos anunciados (%.0f MiB), frame %s; ganchos: create %llu, destroy %llu, stream %llu, texload %llu, "
	            "free %llu, linearcopy %llu",
	            prefix, seconds, u(total), u(by_class[KFrame]), percent(by_class[KFrame]),
	            u(by_class[KAnnounced] + by_class[KAnnouncedNet]), u(by_class[KNeighbor]), u(by_class[KUnknown]), u(by_class[KLeak]), u(by_class[KPerUse]),
	            u(ranges), static_cast<double>(bytes) / MiB, frame, u(c[CCreate]), u(c[CDestroy]), u(c[CStreamCopy]), u(c[CTexLoad]),
	            u(c[CDeferredFreeRemoved] + c[CHeapFreeRemoved] + c[CDecommitRemoved]), u(c[CLinearCopy]));
	// The detail the main part leaves out.
	std::printf(" | ventana %llu, escritura %llu; anunciada por red memcpy %llu; bajas: destroy %llu, free diferido %llu de %llu, "
	            "free heap %llu, decommit %llu rangos en %llu; copia de create %llu; fresh aparte %llu (frame %llu)",
	            u(window), u(write), u(by_class[KAnnouncedNet]), u(c[CDestroyRemoved]), u(c[CDeferredFreeRemoved]), u(c[CDeferredFree]),
	            u(c[CHeapFreeRemoved]), u(c[CDecommitRemoved]), u(c[CDecommit]), u(c[CCreateCopy]), u(fresh), u(t.faults[2][KFrame]));
	if (c[CSweptRuns] != 0) {
		const uint64_t swept = c[CSweptFrameBytes] + c[CSweptRegisteredBytes] + c[CSweptUnknownBytes];
		std::printf("; write watch: %llu tramos, %llu paginas = frame %llu + en registro %llu + desconocido %llu", u(c[CSweptRuns]),
		            u(swept / 4096), u(c[CSweptFrameBytes] / 4096), u(c[CSweptRegisteredBytes] / 4096), u(c[CSweptUnknownBytes] / 4096));
	}
	// (policy1)
	std::printf(" | frame: anillos %llu, DCB %llu, TaggedArena %llu; create: reenvuelto %llu, reescrito (0x9f67dc) %llu; destroy de "
	            "envoltorio %llu; stream con bloque %llu, bajas por bloque %llu; linearcopy ACB %llu, origenes nuevos %llu; capas: "
	            "leidas %llu, convertidas %llu (sin situar %llu); bink %llu; por-uso caducados %llu (faults en ellos %llu)",
	            u(t.frame[0]), u(t.frame[1]), u(t.frame[2]), u(c[CCreateRewrap]), u(c[CCreateRewritten]), u(c[CDestroyReleased]),
	            u(c[CStreamOwned]), u(c[CFreeOwned]), u(c[CLinearCopyAcb]), u(c[CLcSrcNew]), u(c[CLayerRead]), u(c[CLayerConv]),
	            u(c[CLayerConvUnknown]), u(c[CBink]), u(c[CStaleSwept]), u(c[CStaleHit]));
	std::printf("\n");
}

// (policy1) "[read-audit] ...": the CPU read faults by the game code that reads.
void PrintReadLine(const Totals& t, double seconds, const char* prefix) {
	const auto u     = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	uint64_t   total = 0;
	for (const uint64_t n: t.reads) total += n;
	const uint64_t known = total - t.reads[ReadGroups - 1];
	std::printf("[read-audit] %s%.1f s: lecturas %llu = copia de CBuffer (0x909970) %llu + lectores tras 0x909940 %llu + 0xa2e6f0 %llu + "
	            "bits de chunks 0xa1f560/0xad8bc0/0xad92f0 %llu + otros %llu; en accesores conocidos %.1f%%\n",
	            prefix, seconds, u(total), u(t.reads[0]), u(t.reads[1]), u(t.reads[2]), u(t.reads[3]), u(t.reads[4]),
	            total == 0 ? 0.0 : 100.0 * static_cast<double>(known) / static_cast<double>(total));
}

// (policy1) "[write-policy] ...": what KYTY_DS_NEIGHBOR_SPLIT did.
void PrintPolicyLine(const Totals& t, double seconds, const char* prefix) {
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	std::printf("[write-policy] %s%.1f s: paginas sin vigilancia nuevas %llu (junto a rangos anunciados %llu, dentro de por-uso %llu, sin "
	            "rango %llu); protecciones evitadas al subir: %llu paginas; anuncios de ganchos %llu (%.0f KiB) -> %llu paginas marcadas "
	            "escritas; SEGURIDAD: vueltas a la vigilancia normal %llu paginas (consumo de bytes sin gancho %llu en %llu usos, "
	            "escritura de GPU %llu, rango no fiable nuevo o cola desbordada %llu, buffer leido por BDA %llu); cubiertos por ganchos "
	            "%llu usos; no admitidas: frame %llu, dentro de rango %llu, pagina con rango no fiable %llu, sin rango %llu, video en "
	            "marcha %llu, filtro de diagnostico %llu, ya suelta o vetada (sigue el camino normal) %llu; reservas de frame que "
	            "pidieron paginas de vuelta %llu; cola: perdidos %llu, todo devuelto %llu veces (%llu por "
	            "empezar un video)\n",
	            prefix, seconds, u(c[PNoWatchNeighbor] + c[PNoWatchPerUse] + c[PNoWatchUnknown]), u(c[PNoWatchNeighbor]),
	            u(c[PNoWatchPerUse]), u(c[PNoWatchUnknown]), u(t.kept), u(c[PAnnounced]), static_cast<double>(c[PAnnouncedBytes]) / 1024.0,
	            u(c[PDirtyPages]), u(c[PRevertedPagesUse] + c[PRevertedPagesGpu] + c[PRevertedPagesHook] + c[PRevertedPagesBda]),
	            u(c[PRevertedPagesUse]), u(c[PRevertedUses]), u(c[PRevertedPagesGpu]), u(c[PRevertedPagesHook]), u(c[PRevertedPagesBda]),
	            u(c[PReadCovered]), u(c[PKeptFrame]), u(c[PKeptInside]), u(c[PKeptUntrusted]), u(c[PKeptUnknown]), u(c[PKeptBink]),
	            u(c[PKeptFilter]), u(c[PRefusedTracker]), u(c[PFrameReserved]), u(c[PQueueDropped]), u(c[PEverything]), u(c[PBinkRevertAll]));
}
// (FRAME-NATIVO) "[frame-nativo] 30 s: ...": every protection change by label (calls/s, pages/s), what the policies
// spared (pages not protected again after an upload, /s), the frame pages marked written instead (/s), and the bytes
// uploaded per frame (LiveCounters UploadBytes / flips). From the 5 s loop only.
void PrintFrameNativoLine(const Totals& totals, std::chrono::steady_clock::time_point now) {
	static bool                                  started = false;
	static std::chrono::steady_clock::time_point last;
	static uint64_t calls[FrameNativo::CCount] {}, pages[FrameNativo::CCount] {};
	static uint64_t kept = 0, drained = 0, upload = 0, frames = 0;
	static uint64_t avoided[FrameNativo::CCount] {}; // (PROT-MENOS)
	static uint64_t promoted = 0, refused = 0, redirty = 0, demoted = 0; // (PROT-MENOS)
	static uint64_t ww_calls = 0, ww_failed = 0, ww_unchanged = 0, ww_late = 0, step_us = 0; // (PROT-MENOS2)
	const auto take = [&] {
		for (size_t i = 0; i < FrameNativo::CCount; ++i) {
			calls[i] = FrameNativo::g_calls[i].load(std::memory_order_relaxed);
			pages[i] = FrameNativo::g_pages[i].load(std::memory_order_relaxed);
			avoided[i] = FrameNativo::g_avoided[i].load(std::memory_order_relaxed); // (PROT-MENOS)
		}
		promoted = ProtMenos::g_promoted.load(std::memory_order_relaxed); // (PROT-MENOS)
		refused  = ProtMenos::g_refused.load(std::memory_order_relaxed);
		redirty  = ProtMenos::g_redirty_pages.load(std::memory_order_relaxed);
		demoted  = ProtMenos::g_demoted.load(std::memory_order_relaxed);
		ww_calls     = ProtMenos::g_ww_calls.load(std::memory_order_relaxed); // (PROT-MENOS2)
		ww_failed    = ProtMenos::g_ww_failed.load(std::memory_order_relaxed);
		ww_unchanged = ProtMenos::g_ww_unchanged.load(std::memory_order_relaxed);
		ww_late      = ProtMenos::g_ww_late.load(std::memory_order_relaxed);
		step_us      = ProtMenos::g_step_us.load(std::memory_order_relaxed);
		kept    = totals.kept;
		drained = totals.counters[FrDrainPages];
		upload  = LiveCounters::Value(LiveCounters::UploadBytes);
		frames  = Copia1::g_frames.load(std::memory_order_relaxed);
		last    = now;
	};
	if (!started) {
		started = true;
		take();
		return;
	}
	if (now - last < std::chrono::seconds(30)) return;
	const double s = std::chrono::duration<double>(now - last).count();
	const auto   u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	uint64_t     total_calls = 0;
	uint64_t     dc[FrameNativo::CCount], dp[FrameNativo::CCount];
	for (size_t i = 0; i < FrameNativo::CCount; ++i) {
		dc[i] = FrameNativo::g_calls[i].load(std::memory_order_relaxed) - calls[i];
		dp[i] = FrameNativo::g_pages[i].load(std::memory_order_relaxed) - pages[i];
		total_calls += dc[i];
	}
	const uint64_t df = Copia1::g_frames.load(std::memory_order_relaxed) - frames;
	const uint64_t du = LiveCounters::Value(LiveCounters::UploadBytes) - upload;
	std::printf("[frame-nativo] %.0f s (KYTY_FRAME_NATIVO=%d, politica de frame %s): cambios de proteccion %.0f/s por etiqueta:", s,
	            FrameNativo::Wanted() ? 1 : 0, FrameOn() ? "ACTIVA" : "apagada", static_cast<double>(total_calls) / s);
	for (size_t i = 0; i < FrameNativo::CCount; ++i)
		std::printf(" %s %.0f/s (%.0f%%, %.0f pag/s)", FrameNativo::kNames[i], static_cast<double>(dc[i]) / s,
		            total_calls != 0 ? 100.0 * static_cast<double>(dc[i]) / static_cast<double>(total_calls) : 0.0,
		            static_cast<double>(dp[i]) / s);
	std::printf("; protecciones evitadas al subir %.0f pag/s; paginas de frame marcadas escritas sin fault %.0f/s; subido %.2f MB/frame "
	            "(%llu frames)\n",
	            static_cast<double>(totals.kept - kept) / s, static_cast<double>(totals.counters[FrDrainPages] - drained) / s,
	            df != 0 ? static_cast<double>(du) / 1e6 / static_cast<double>(df) : 0.0, u(df));
	// (PROT-MENOS) Per label: protection changes now ("despues") and what they would have been without the hot pages
	// ("antes" = now + 2 x the reprotects spared: each spares its call and the unprotect of the next write fault).
	{
		uint64_t da[FrameNativo::CCount], total_avoided = 0;
		for (size_t i = 0; i < FrameNativo::CCount; ++i) {
			da[i] = FrameNativo::g_avoided[i].load(std::memory_order_relaxed) - avoided[i];
			total_avoided += da[i];
		}
		std::printf("[prot-menos] %.0f s (KYTY_PROT_MENOS=%d): protecciones/s antes -> despues: total %.0f -> %.0f;", s,
		            ProtMenos::On() ? 1 : 0, static_cast<double>(total_calls + 2 * total_avoided) / s, static_cast<double>(total_calls) / s);
		for (size_t i = 0; i < FrameNativo::CCount; ++i)
			std::printf(" %s %.0f -> %.0f;", FrameNativo::kNames[i], static_cast<double>(dc[i] + 2 * da[i]) / s, static_cast<double>(dc[i]) / s);
		std::printf(" paginas calientes ahora %lld (tope %lld), nuevas %.0f/s, rechazadas por tope %.0f/s, devueltas a la vigilancia "
		            "%.0f/s, re-sucias antes de cada envio %.0f pag/s (%.2f MB/frame como maximo a resubir)",
		            static_cast<long long>(ProtMenos::g_hot_pages.load(std::memory_order_relaxed)), static_cast<long long>(ProtMenos::kMaxHotPages),
		            static_cast<double>(ProtMenos::g_promoted.load(std::memory_order_relaxed) - promoted) / s,
		            static_cast<double>(ProtMenos::g_refused.load(std::memory_order_relaxed) - refused) / s,
		            static_cast<double>(ProtMenos::g_demoted.load(std::memory_order_relaxed) - demoted) / s,
		            static_cast<double>(ProtMenos::g_redirty_pages.load(std::memory_order_relaxed) - redirty) / s,
		            df != 0 ? static_cast<double>((ProtMenos::g_redirty_pages.load(std::memory_order_relaxed) - redirty) * 4096) / 1e6 /
		                          static_cast<double>(df)
		                    : 0.0);
		// (PROT-MENOS2) Write watch of the hot pages: MB re-uploaded per frame (pages made dirty x 4 KiB), pages left clean
		// (not written: not re-uploaded), GetWriteWatch reads, refused ones, pages caught after a demotion, cost per frame.
		const auto d = [](const std::atomic<uint64_t>& v, uint64_t old_v) { return v.load(std::memory_order_relaxed) - old_v; };
		const double frames_d = df != 0 ? static_cast<double>(df) : 1.0;
		std::printf("; (PROT-MENOS2) vigilancia de escritura %s: resubido %.2f MB/frame, sin cambios (no resubidas) %.2f MB/frame, "
		            "GetWriteWatch %.0f/frame (%.0f fallidas/s), tras devolver %.0f pag/s, coste %.3f ms/frame\n",
		            ProtMenos::WatchOn() ? "SI" : "NO (todas re-sucias)",
		            static_cast<double>(d(ProtMenos::g_redirty_pages, redirty) * 4096) / 1e6 / frames_d,
		            static_cast<double>(d(ProtMenos::g_ww_unchanged, ww_unchanged) * 4096) / 1e6 / frames_d,
		            static_cast<double>(d(ProtMenos::g_ww_calls, ww_calls)) / frames_d, static_cast<double>(d(ProtMenos::g_ww_failed, ww_failed)) / s,
		            static_cast<double>(d(ProtMenos::g_ww_late, ww_late)) / s, static_cast<double>(d(ProtMenos::g_step_us, step_us)) / 1000.0 / frames_d);
	}
	take();
}
// (policy6) "[write-frame] ...": what KYTY_DS_WRITE_FRAME did, and (=2) the stale-use check.
void PrintFrameLine(const Totals& t, double seconds, const char* prefix, bool list) {
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	std::printf("[write-frame] %s%.1f s: paginas de frame sin vigilancia nuevas %llu; protecciones evitadas al subir (todas las "
	            "politicas) %llu paginas; marcas: lotes %llu, envios de command buffer %llu, copias del emulador %llu (otras %llu ya las "
	            "cubria el lote) = %llu rangos "
	            "(%.0f MiB) -> %llu paginas marcadas escritas; TODO marcado %llu veces (cola llena o lotes fuera de orden)",
	            prefix, seconds, u(c[PNoWatchFrame]), u(t.kept), u(c[FrLots]), u(c[FrSubmitMarks]), u(c[FrCopyMarks]), u(c[FrCopySkipped]), u(c[FrMarks]),
	            static_cast<double>(c[FrMarkedBytes]) / MiB, u(c[FrDrainPages]), u(c[FrOverflow]));
	if (CheckStaleOn())
		std::printf("; COMPROBACION (=2): paginas copiadas al quedar limpias %llu, vistas otra vez al ir a usarlas %llu: cambiadas pero "
		            "entregadas de nuevo por el asignador (las marca el lote) %llu, cambiadas solo fuera del rango que se usa %llu, uso "
		            "demasiado amplio para saberlo (BDA, > 1 MiB) %llu, CON DATOS VIEJOS en el rango usado %llu (de frame %llu)",
		            u(c[VCleanNoted]), u(c[VUseChecked]), u(c[VUseRereserved]), u(c[VUseOutside]), u(c[VUseWide]), u(c[VUseStale]),
		            u(c[VUseStaleFrame]));
	std::printf("\n");
	if (!list || !CheckStaleOn()) return;
	StaleUse rows[StaleMax];
	size_t   count = 0;
	{
		std::lock_guard lock(g_stale_mutex);
		count = g_stale_count;
		for (size_t i = 0; i < count; ++i) rows[i] = g_stale[i];
	}
	std::sort(rows, rows + count, [](const StaleUse& a, const StaleUse& b) { return a.count > b.count; });
	for (size_t i = 0; i < count; ++i) {
		char where[40], label[24] = "";
		Describe(rows[i].rip, where, sizeof(where));
		if (!DemonsSoulsMemMap::LabelAt(rows[i].address, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[write-frame]   DATOS VIEJOS x%llu: 0x%llx (%s, %s), %u bytes distintos dentro de un uso de %llu bytes; pagina soltada "
		            "por una escritura de %s\n",
		            u(rows[i].count), u(rows[i].address), label, rows[i].frame ? "frame" : "no frame", rows[i].bytes, u(rows[i].use), where);
	}
}

// (policy4) The pages given back, by reason and by the code that wrote them when they were made no-watch.
void PrintReverted() {
	static constexpr const char* Why[] = {"un consumidor pidio bytes sin gancho", "la GPU escribe ahi", "su buffer lo lee un shader por BDA"};
	Reverted rows[RevertedMax];
	size_t   count = 0;
	{
		std::lock_guard lock(g_reverted_mutex);
		count = g_reverted_count;
		for (size_t i = 0; i < count; ++i) rows[i] = g_reverted[i];
	}
	std::sort(rows, rows + count, [](const Reverted& a, const Reverted& b) { return a.pages > b.pages; });
	for (size_t i = 0; i < count && i < 24; ++i) {
		char where[40], label[24] = "";
		Describe(rows[i].rip, where, sizeof(where));
		if (!DemonsSoulsMemMap::LabelAt(rows[i].address, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[write-policy]   devueltas %llu paginas en %llu veces: %s; soltada por una escritura de %s; primera 0x%llx (%s)\n",
		            static_cast<unsigned long long>(rows[i].pages), static_cast<unsigned long long>(rows[i].uses),
		            Why[rows[i].why < 0 || rows[i].why > 2 ? 0 : rows[i].why], where, static_cast<unsigned long long>(rows[i].address), label);
	}
}

// (policy2) "[readback-hook] ...": what KYTY_DS_READBACK_HOOK did, and its check: the read faults that still come
// from the accessors' readers (they should go to nearly none; the ones left are buffers written again mid-read).
void PrintReadbackLine(const Totals& t, double seconds, const char* prefix) {
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	std::printf("[readback-hook] %s%.1f s: accesos %llu (puntero 0x909940 %llu, copia 0x909970 %llu); con paginas de GPU en su granulo de "
	            "64 KiB %llu -> descargados de una vez %llu (%.0f KiB), sin paginas de GPU propias o no mapeado (nada hecho) %llu; COMPROBACION: faults de lectura que aun llegan desde esos "
	            "accesores %llu (copia %llu, lectores %llu); otras lecturas por fault %llu (lista 0xa2e6f0 %llu [entradas %llu, descargadas %llu], "
	            "bits de chunks %llu, resto %llu); READ-HOOK2: mascara 0xadabb0 %llu (descargadas %llu, entradas 0xadad90 con fault %llu, descargadas %llu), copia 0x8e9760 %llu (descargadas %llu); DESCARGADOS SIN VIGILANTE (paginas SIN-VIGILAR): CBuffer %llu, lista %llu, mascara %llu, entradas %llu, copia12 %llu (%.0f KiB)\n",
	            prefix, seconds, u(c[RPointer] + c[RCopy]), u(c[RPointer]), u(c[RCopy]), u(c[RNeeded]), u(c[RDone]),
	            static_cast<double>(c[RDoneBytes]) / 1024.0, u(c[RRefused]), u(t.reads[0] + t.reads[1]), u(t.reads[0]), u(t.reads[1]),
	            u(t.reads[2] + t.reads[3] + t.reads[4]), u(t.reads[2]), u(c[RVisList]), u(c[RVisDone]), u(t.reads[3]), u(t.reads[4]), u(c[RMask]),
	            u(c[RMaskDone]), u(c[RMaskEnt]), u(c[RMaskEntDone]), u(c[RCopy12]), u(c[RCopy12Done]),
	            u(c[RUnwCBuf]), u(c[RUnwVis]), u(c[RUnwMask]), u(c[RUnwMaskEnt]), u(c[RUnwCopy12]), static_cast<double>(c[RUnwBytes]) / 1024.0);
}

// (policy3) "[frame-audit] ...". Its counts are totals since the start (the judging lags the faults by one lot).
const char* FaAllocatorName(uint64_t object, char* out, size_t size) {
	static constexpr struct {
		uint64_t    rva;
		const char* name;
	} Known[] = {{0x2f64678, "GameToRenderThread"}, {0x2f64b20, "GameToGPU"}, {0x2f64fc8, "CPU_RenderToGPUToGame"},
	             {0x2f65470, "RenderSingleFrameGPU"}, {0x2f65918, "RenderSingleFrameCPU"}};
	if (object == 0) {
		std::snprintf(out, size, "sin reserva vista");
		return out;
	}
	for (const auto& known: Known) {
		if (object == g_base + known.rva) {
			std::snprintf(out, size, "%s", known.name);
			return out;
		}
	}
	if (InEboot(object)) std::snprintf(out, size, "asignador eboot+0x%llx", static_cast<unsigned long long>(object - g_base));
	else std::snprintf(out, size, "asignador 0x%llx", static_cast<unsigned long long>(object));
	return out;
}
void PrintFrameAuditLine(const Totals& t, double seconds, const char* prefix, bool list) {
	static constexpr const char* Names[FaClasses] = {"anillos", "DCB", "TaggedArena", "GameToGPU"};
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	uint64_t    faults = 0, covered = 0, pending = 0;
	for (const auto& result: g_fa_results) {
		const uint64_t f = result.faults.load(std::memory_order_relaxed), unreserved = result.unreserved.load(std::memory_order_relaxed);
		const uint64_t done = result.covered.load(std::memory_order_relaxed) + result.covered_slack.load(std::memory_order_relaxed) +
		                      result.by_copy.load(std::memory_order_relaxed) + result.by_submit.load(std::memory_order_relaxed) +
		                      result.by_allocator.load(std::memory_order_relaxed); // (policy7: free memory: nothing to cover)
		const uint64_t gpu  = result.by_gpu.load(std::memory_order_relaxed);
		const uint64_t judged = done + gpu + result.uncovered.load(std::memory_order_relaxed) + unreserved;
		// (The policy takes only reserved pages: the faults on pages with no reservation are not its business.)
		faults += f - unreserved;
		covered += done;
		pending += f > judged ? f - judged : 0;
	}
	faults -= pending; // (still waiting for the submits after them)
	std::printf("[frame-audit] %s%.1f s: en este tramo reservas %llu (%llu paginas), kickoffs %llu, envios de lote %llu, sin juzgar %llu "
	            "(orden roto %llu veces), tabla llena %llu. TOTAL desde el arranque, paginas reservadas: faults de frame juzgados %llu, "
	            "CUBIERTOS por R1+R2+R3 %llu (%.3f%%)",
	            prefix, seconds, u(c[FAllocs]), u(c[FAllocPages]), u(c[FKickoffs]), u(c[FSubmits]), u(c[FPendingLost]), u(c[FOrderBroken]),
	            u(c[FTableFull]), u(faults), u(covered), faults == 0 ? 0.0 : 100.0 * static_cast<double>(covered) / static_cast<double>(faults));
	for (uint32_t k = 0; k < FaClasses; ++k) {
		const FaResult& r = g_fa_results[k];
		std::printf("; %s: %llu = lote %llu + lote con margen %llu + copia del emulador %llu + envio posterior de command buffer %llu + "
		            "metadatos del asignador en memoria libre %llu + escrituras del hilo de GPU del emulador %llu + NO cubiertos %llu + sin "
		            "reserva vista %llu",
		            Names[k], u(r.faults.load(std::memory_order_relaxed)), u(r.covered.load(std::memory_order_relaxed)),
		            u(r.covered_slack.load(std::memory_order_relaxed)), u(r.by_copy.load(std::memory_order_relaxed)),
		            u(r.by_submit.load(std::memory_order_relaxed)), u(r.by_allocator.load(std::memory_order_relaxed)),
		            u(r.by_gpu.load(std::memory_order_relaxed)), u(r.uncovered.load(std::memory_order_relaxed)),
		            u(r.unreserved.load(std::memory_order_relaxed)));
	}
	std::printf("\n");
	if (!list) return;
	FaSample samples[FaSampleMax];
	size_t   count = 0;
	{
		std::lock_guard lock(g_fa_mutex);
		count = g_fa_sample_count;
		for (size_t i = 0; i < count; ++i) samples[i] = g_fa_samples[i];
	}
	std::sort(samples, samples + count, [](const FaSample& a, const FaSample& b) { return a.count > b.count; });
	for (size_t i = 0; i < count && i < 24; ++i) {
		char where[40], name[48];
		Describe(samples[i].rip, where, sizeof(where));
		std::printf("[frame-audit]   NO cubierto x%llu: %s, %s, escribe %s, primera direccion 0x%llx; kickoffs entre la reserva de su "
		            "pagina y la escritura: %u%s\n",
		            u(samples[i].count), Names[samples[i].cls], FaAllocatorName(samples[i].allocator, name, sizeof(name)), where,
		            u(samples[i].address), samples[i].kickoffs_old,
		            samples[i].unreserved ? " (pagina sin ninguna reserva vista)"
		                                  : samples[i].kickoffs_old >= 2 ? " (escrita despues del envio que debia cubrirla)" : "");
	}
}

struct Row {
	uint64_t key, count;
};
template <size_t N>
std::vector<Row> Rows(const CountTable<N>& table, uint64_t tag) {
	std::vector<Row> rows;
	for (const auto& slot: table.slots) {
		const uint64_t key = slot.key.load(std::memory_order_relaxed);
		if (key == 0 || (key >> TagShift) != tag) continue;
		const uint64_t count = slot.count.load(std::memory_order_relaxed);
		if (count != 0) rows.push_back({key & TagMask, count});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.count > b.count; });
	return rows;
}

void PrintSites(const char* title, uint64_t tag) {
	const auto rows = Rows(g_sites, tag);
	if (rows.empty()) return;
	uint64_t total = 0;
	for (const auto& row: rows) total += row.count;
	std::printf("[write-audit]   %s (%zu sitios, %llu faults):", title, rows.size(), static_cast<unsigned long long>(total));
	for (size_t i = 0; i < rows.size() && i < 10; ++i) {
		char where[40];
		Describe(rows[i].key, where, sizeof(where));
		std::printf(" %s x%llu", where, static_cast<unsigned long long>(rows[i].count));
	}
	std::printf("\n");
}

// The granules of one tag summed by the allocator label of each (DemonsSoulsMemMap).
void PrintLabels(const char* title, uint64_t tag) {
	const auto rows = Rows(g_granules, tag);
	if (rows.empty()) return;
	struct Label {
		char     name[24];
		uint64_t count;
	};
	std::vector<Label> labels;
	uint64_t           total = 0;
	for (const auto& row: rows) {
		char name[24] = "";
		if (!DemonsSoulsMemMap::LabelAt((row.key - 1) << 16u, name, sizeof(name)) || name[0] == 0)
			std::snprintf(name, sizeof(name), "sin etiqueta");
		total += row.count;
		auto found = std::find_if(labels.begin(), labels.end(), [&](const Label& l) { return std::strcmp(l.name, name) == 0; });
		if (found == labels.end()) {
			Label label {};
			std::snprintf(label.name, sizeof(label.name), "%s", name);
			label.count = row.count;
			labels.push_back(label);
		} else {
			found->count += row.count;
		}
	}
	std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b) { return a.count > b.count; });
	std::printf("[write-audit]   %s por etiqueta de asignador (%llu, en %zu granulos de 64 KiB):", title,
	            static_cast<unsigned long long>(total), rows.size());
	for (size_t i = 0; i < labels.size() && i < 12; ++i)
		std::printf(" %s %llu (%.1f%%)", labels[i].name, static_cast<unsigned long long>(labels[i].count),
		            100.0 * static_cast<double>(labels[i].count) / static_cast<double>(total));
	std::printf("\n");
}

void PrintTables(const Totals& t, double seconds, bool frame_known) {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	PrintLine(t, seconds, "ACUMULADO ", frame_known);
	std::printf("[write-audit]   registro por tipo:");
	for (uint8_t k = 0; k < Reg::KindCount; ++k)
		std::printf(" %s %llu (%.1f MiB)", Reg::KindName(k), u(registry->Count(k)), static_cast<double>(registry->Bytes(k)) / MiB);
	std::printf("\n");
	std::printf("[write-audit]   create %llu: con datos %llu, gpu (flag 4/1) %llu, envuelto (0x800) %llu, sin datos %llu; sin registrar: "
	            "frame por flags %llu, en arena de frame %llu, sin memoria %llu, envuelto dentro de otro %llu, ilegible %llu; altas sobre "
	            "un rango vivo (memoria reutilizada sin baja vista) %llu\n",
	            u(c[CCreate]), u(c[CCreateData]), u(c[CCreateGpu]), u(c[CCreateWrap]), u(c[CCreatePlain]), u(c[CCreateSkipFlags]),
	            u(c[CCreateSkipFrame]), u(c[CCreateSkipEmpty]), u(c[CCreateNested]), u(c[CCreateUnreadable]), u(c[CReplaced]));
	std::printf("[write-audit]   linear copy %llu: origen en frame %llu, en registro %llu, desconocido %llu; destino en registro %llu, "
	            "en frame %llu, desconocido %llu\n",
	            u(c[CLinearCopy]), u(c[CLcSrcFrame]), u(c[CLcSrcRegistered]), u(c[CLcSrcUnknown]), u(c[CLcDstRegistered]),
	            u(c[CLcDstFrame]), u(c[CLcDstUnknown]));
	// The FUGA table.
	Leak     leaks[LeakMax];
	size_t   leak_count = 0;
	uint64_t dropped    = 0;
	{
		std::lock_guard lock(g_leak_mutex);
		leak_count = g_leak_count;
		dropped    = g_leak_dropped;
		for (size_t i = 0; i < leak_count; ++i) leaks[i] = g_leaks[i];
	}
	uint64_t leak_faults = 0;
	for (uint32_t k = 0; k < FaultKinds; ++k) leak_faults += t.faults[k][KLeak];
	std::printf("[write-audit]   FUGA: %llu faults, %zu parejas rango+escritor distintas%s\n", u(leak_faults), leak_count,
	            dropped != 0 ? " (tabla llena: hay mas)" : "");
	std::sort(leaks, leaks + leak_count, [](const Leak& a, const Leak& b) { return a.count > b.count; });
	for (size_t i = 0; i < leak_count; ++i) {
		const Leak& leak = leaks[i];
		char        label[24] = "", rip[40], ret[56] = "";
		if (!DemonsSoulsMemMap::LabelAt(leak.address, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		Describe(leak.rip, rip, sizeof(rip));
		if (leak.ret != 0) {
			char where[40];
			Describe(leak.ret, where, sizeof(where));
			std::snprintf(ret, sizeof(ret), " (tope de pila %s)", where);
		}
		std::printf("[write-audit]     FUGA x%llu: 0x%llx en [0x%llx +%llu] %s flags 0x%x fmt %u, %s, escribe %s%s, %s\n", u(leak.count),
		            u(leak.address), u(leak.begin), u(leak.end - leak.begin), Reg::KindName(leak.kind), leak.flags, leak.format, label, rip,
		            ret, leak.fault_kind == 0 ? "ventana" : leak.fault_kind == 1 ? "escritura" : "fresh");
	}
	PrintLabels("desconocido", 1);
	PrintLabels("vecino", 2);
	PrintLabels("origen de linear copy fuera de frame y de registro", 3);
	PrintLabels("por-uso (memoria envuelta 0x800 / origen de linear copy escrita por su dueno)", 4);
	if (SplitOn()) PrintLabels("paginas devueltas a la vigilancia normal", 5);
	// (policy1) What each level of the split would end, by the faults seen (ventana + escritura).
	{
		const uint64_t faults = [&] {
			uint64_t n = 0;
			for (uint32_t k = 0; k < 2; ++k)
				for (uint32_t cl = 0; cl < ClassCount; ++cl) n += t.faults[k][cl];
			return n;
		}();
		const auto share = [&](uint64_t n) { return faults == 0 ? 0.0 : 100.0 * static_cast<double>(n) / static_cast<double>(faults); };
		const uint64_t one = c[CWouldNeighbor1], two = one + c[CWouldNeighbor2] + c[CWouldPerUse2], three = two + c[CWouldUnknown];
		std::printf("[write-audit]   lo que quitaria KYTY_DS_NEIGHBOR_SPLIT (de %llu faults): =1 con ANNOUNCED=1 %llu (%.1f%%); =1 con "
		            "ANNOUNCED=2 %llu (%.1f%%); =2 con ANNOUNCED=2 %llu (%.1f%%); se quedan: frame %llu (%.1f%%), paginas con rango no "
		            "fiable %llu, FUGA %llu\n",
		            u(faults), u(one), share(one), u(two), share(two), u(three), share(three),
		            u(t.faults[0][KFrame] + t.faults[1][KFrame]), share(t.faults[0][KFrame] + t.faults[1][KFrame]),
		            u(c[CWouldNeighborNo] + c[CWouldPerUseNo]), u(t.faults[0][KLeak] + t.faults[1][KLeak]));
	}
	PrintSites("desconocido, escritura directa: codigo que escribe", 1);
	PrintSites("desconocido, dentro de un memcpy: sitio que llama", 2);
	PrintSites("anunciada solo por la red memcpy: sitio que llama", 3);
	if (g_sites.dropped.load(std::memory_order_relaxed) != 0 || g_granules.dropped.load(std::memory_order_relaxed) != 0)
		std::printf("[write-audit]   (tablas de detalle llenas: %llu sitios y %llu granulos sin contar)\n",
		            u(g_sites.dropped.load(std::memory_order_relaxed)), u(g_granules.dropped.load(std::memory_order_relaxed)));
	std::fflush(stdout);
}

std::atomic<bool> g_frame_known {false};
std::chrono::steady_clock::time_point g_start;

// (policy1) The read faults' table: ALL the sites outside the known readers, most reads first (antes solo 16; prueba 68).
void PrintReadTable(const Totals& t, double seconds) {
	PrintReadLine(t, seconds, "ACUMULADO ");
	std::vector<Row> rows;
	for (const auto& slot: g_read_sites.slots) {
		const uint64_t key = slot.key.load(std::memory_order_relaxed), count = slot.count.load(std::memory_order_relaxed);
		if (key != 0 && count != 0) rows.push_back({key, count});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.count > b.count; });
	std::printf("[read-audit]   otros: %zu sitios%s\n", rows.size(),
	            g_read_sites.dropped.load(std::memory_order_relaxed) != 0 ? " (tabla llena: hay mas)" : "");
	for (size_t i = 0; i < rows.size(); ++i) {
		uint64_t first = 0;
		for (const auto& entry: g_read_first)
			if (entry.site.load(std::memory_order_relaxed) == rows[i].key) first = entry.address.load(std::memory_order_relaxed);
		char where[40], label[24] = "";
		Describe(rows[i].key & (CopyFlag - 1), where, sizeof(where));
		if (first == 0 || !DemonsSoulsMemMap::LabelAt(first, label, sizeof(label)) || label[0] == 0)
			std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[read-audit]     x%llu: %s%s, primera lectura en 0x%llx (%s)\n", static_cast<unsigned long long>(rows[i].count), where,
		            (rows[i].key & CopyFlag) != 0 ? " (llama a memcpy: la lectura es el origen de la copia)" : "",
		            static_cast<unsigned long long>(first), label);
	}
	std::fflush(stdout);
}

// ---- (policy1) KYTY_DS_WRITE_VERIFY ------------------------------------------------------------------------------
// The audit's thread, every 500 ms: ranges of the kinds the split may trust are hashed, and hashed again at the next
// step. A range whose generation did not move in between (no hook announced it, it was not registered again, and for
// a per-use range a wrapper stayed alive over it) must hold the same bytes: the GPU's copy of them is the one taken
// when they were last announced. Bounded: VerifyRanges ranges and VerifyBytes bytes a step, VerifyPiece of each.
constexpr size_t   VerifyRanges = 24;
constexpr uint64_t VerifyBytes  = uint64_t {2} << 20u;
constexpr uint64_t VerifyPiece  = uint64_t {256} << 10u;
struct VerifySample {
	uint64_t begin = 0, end = 0, at = 0, bytes = 0, hash = 0;
	uint32_t generation = 0;
	uint8_t  kind       = 0;
	bool     valid      = false;
};
VerifySample g_verify_samples[VerifyRanges];
struct Mismatch {
	uint64_t begin = 0, end = 0, at = 0, count = 0;
	uint32_t flags = 0;
	uint8_t  kind  = 0;
};
constexpr size_t MismatchMax = 32;
Mismatch         g_mismatches[MismatchMax]; // (the audit's thread only)
size_t           g_mismatch_count = 0;

// FNV-1a over 8-byte words of the bytes read through GuardedCopy (a page without access ends it: false).
bool HashGuest(uint64_t address, uint64_t bytes, uint64_t* out) {
	static uint8_t buffer[64 * 1024];
	uint64_t       hash = 0xcbf29ce484222325ull;
	for (uint64_t done = 0; done < bytes;) {
		const uint64_t piece = (std::min)(bytes - done, static_cast<uint64_t>(sizeof(buffer)));
		if (Common::GuardedCopy(buffer, reinterpret_cast<const void*>(address + done), piece) != 0) return false;
		uint64_t i = 0;
		for (; i + 8 <= piece; i += 8) {
			uint64_t word;
			std::memcpy(&word, buffer + i, 8);
			hash = (hash ^ word) * 0x100000001b3ull;
		}
		for (; i < piece; ++i) hash = (hash ^ buffer[i]) * 0x100000001b3ull;
		done += piece;
	}
	*out = hash;
	return true;
}

bool VerifyUsable(const Reg::Hit& hit) {
	if (hit.pending || ((1u << hit.kind) & Reg::TrustedKinds(2)) == 0) return false;
	// (A per-use range is only bound to hold its bytes while a wrapper is alive over it.)
	return ((1u << hit.kind) & Reg::PerUseKinds) == 0 || (hit.kind == Reg::CBufWrap && hit.live > 0);
}

void VerifyStep() {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	// The second look of the last step's samples.
	for (auto& sample: g_verify_samples) {
		if (!sample.valid) continue;
		sample.valid = false;
		Reg::Hit hit;
		if (!registry->At(sample.begin, sample.end, &hit) || hit.kind != sample.kind || hit.generation != sample.generation ||
		    !VerifyUsable(hit)) {
			Count(VAnnounced);
			continue;
		}
		uint64_t hash = 0;
		if (!HashGuest(sample.at, sample.bytes, &hash)) {
			Count(VUnreadable);
			continue;
		}
		// (Announced while it was being hashed: not a finding.)
		Reg::Hit after;
		if (!registry->At(sample.begin, sample.end, &after) || after.generation != sample.generation || !VerifyUsable(after)) {
			Count(VAnnounced);
			continue;
		}
		if (hash == sample.hash) {
			Count(VSame);
			continue;
		}
		Count(VMismatch);
		size_t i = 0;
		while (i < g_mismatch_count && g_mismatches[i].begin != sample.begin) ++i;
		if (i == g_mismatch_count) {
			if (g_mismatch_count == MismatchMax) continue;
			g_mismatches[g_mismatch_count++] = {sample.begin, sample.end, sample.at, 0, hit.flags, sample.kind};
		}
		g_mismatches[i].count++;
	}
	// New samples: a run of ranges from a place that moves through the address space.
	static uint64_t seed   = 0x9E3779B97F4A7C15ull;
	seed                   = seed * 6364136223846793005ull + 1442695040888963407ull;
	const uint64_t from    = (seed >> 24u) & ((uint64_t {1} << 40u) - 1);
	uint64_t       budget  = VerifyBytes;
	size_t         slot    = 0;
	uint64_t       first   = 0;
	for (size_t index = 0; index < VerifyRanges * 4 && slot < VerifyRanges && budget != 0; ++index) {
		Reg::Hit hit;
		if (!registry->Nth(from, index, &hit)) break;
		if (index != 0 && hit.begin == first) break; // (went round)
		if (index == 0) first = hit.begin;
		if (!VerifyUsable(hit) || InFrame(hit.begin)) continue;
		VerifySample sample;
		sample.begin = hit.begin;
		sample.end   = hit.end;
		sample.bytes = (std::min)((std::min)(hit.end - hit.begin, VerifyPiece), budget);
		// (A piece of a long range, at a 4 KiB step that moves.)
		const uint64_t room = hit.end - hit.begin - sample.bytes;
		sample.at           = hit.begin + (room == 0 ? 0 : ((seed >> 12u) % (room / 4096 + 1)) * 4096);
		if (sample.at + sample.bytes > hit.end) sample.at = hit.begin;
		sample.generation = hit.generation;
		sample.kind       = hit.kind;
		if (!HashGuest(sample.at, sample.bytes, &sample.hash)) {
			Count(VUnreadable);
			continue;
		}
		Reg::Hit after;
		if (!registry->At(sample.begin, sample.end, &after) || after.generation != sample.generation) continue;
		sample.valid = true;
		budget -= sample.bytes;
		Count(VSampled);
		Count(VSampledBytes, sample.bytes);
		g_verify_samples[slot++] = sample;
	}
}

void PrintVerifyLine(const Totals& t, double seconds, const char* prefix, bool list) {
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const auto& c = t.counters;
	std::printf("[write-verify] %s%.1f s: muestras %llu (%.0f KiB): iguales %llu, anunciadas o cambiadas de dueno entre las dos lecturas "
	            "%llu, ilegibles %llu, DISCREPANCIAS %llu (bytes de un rango fiable que cambiaron sin que ningun gancho lo anunciara)\n",
	            prefix, seconds, u(c[VSampled]), static_cast<double>(c[VSampledBytes]) / 1024.0, u(c[VSame]), u(c[VAnnounced]),
	            u(c[VUnreadable]), u(c[VMismatch]));
	if (!list) return;
	for (size_t i = 0; i < g_mismatch_count; ++i) {
		const Mismatch& m         = g_mismatches[i];
		char            label[24] = "";
		if (!DemonsSoulsMemMap::LabelAt(m.at, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[write-verify]   DISCREPANCIA x%llu: [0x%llx +%llu] %s flags 0x%x, %s (trozo comparado desde 0x%llx)\n", u(m.count),
		            u(m.begin), u(m.end - m.begin), Reg::KindName(m.kind), m.flags, label, u(m.at));
	}
}

// ---- (anuncio1) KYTY_RAIZ_ANNOUNCE=1 (with KYTY_RAIZ=1) ---------------------------------------------------------------
// Measured (logs 11-15/16, native/anuncio1/INFORME.md): with the raiz policies (split on, frame policy off) ~44% of the
// CPU write faults are frame memory the split refuses ("no admitidas: frame") and ~53% are pages it would take but the
// tracker refuses ("ya suelta o vetada"); with the frame policy on (old memory, 215649) frame pages a GPU write had
// refused ("la GPU escribe ahi", 80,595 pages in 120 s, nearly all DefaultTagg / frame) stay refused for good and keep
// faulting. So: the frame policy on, and a frame page refused by a GPU write is made no-watch again once a frame
// allocator has handed it out again after the refusal (new owner: the GPU write belonged to the reservation before).
// That is as safe as the frame policy: BufferCache::PolicyBeforeUse still takes any no-watch page back before GPU work
// writes it or reads bytes no rule covers. Plus the counters that say why the tracker refuses (the minimal counter the
// raiz log lacked) and, with KYTY_RAIZ_VERIFY=1, the write-watch audit of the no-watch pages GPU work is about to use.
int AnnounceWantedLevel() noexcept {
	static const int level = [] {
		const char* a = std::getenv("KYTY_RAIZ_ANNOUNCE");
		// (VIGILANCIA-DOBLE) Sin poner, con KYTY_SIN_VIGILANCIA_DOBLE=1 y KYTY_RAIZ=1: anuncio + VERIFY (nivel 2): cada pagina
		// sin vigilancia que la GPU va a usar se mira en el write watch; lo escrito sin anuncio se cuenta (SIN ANUNCIO), se
		// resube y vuelve a la vigilancia normal (BufferCache::PolicyCheckAnnounced).
		if (a == nullptr && ::NativoCubierto::On()) {
			const char* r = std::getenv("KYTY_RAIZ");
			return (r != nullptr && r[0] == '1') ? 2 : 0;
		}
		if (a == nullptr || a[0] != '1' || a[1] != 0) return 0;
		const char* r = std::getenv("KYTY_RAIZ");
		if (r == nullptr || r[0] != '1') return -1; // (asked for, without raiz: ignored and said)
		const char* v = std::getenv("KYTY_RAIZ_VERIFY");
		return (v != nullptr && v[0] == '1') ? 2 : 1;
	}();
	return level;
}

// The clock (g_fa_clock) when each page was last refused (ClearNoWatch with `block`): open addressing as FaPage, key =
// page number + 1. Not found: unknown, never unblocked (the page keeps today's behaviour).
constexpr size_t BlockSlots   = size_t {1} << 20u; // (16 MiB; 4 GiB of pages)
constexpr uint64_t BlockRunMax = 65536;           // pages stamped a run at most (a longer run: not stamped)
FaPage*          g_block_pages = nullptr;         // (new[] at Install with the variable, never freed)
constexpr uint64_t BlockUnknown = ~uint64_t {0};
uint64_t BlockClock(uint64_t page) noexcept {
	if (g_block_pages == nullptr) return BlockUnknown;
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 44u) % BlockSlots;
	for (size_t probe = 0; probe < 32; ++probe, i = (i + 1) % BlockSlots) {
		const uint64_t current = g_block_pages[i].key.load(std::memory_order_acquire);
		if (current == key) return g_block_pages[i].value.load(std::memory_order_acquire);
		if (current == 0) return BlockUnknown;
	}
	return BlockUnknown;
}
bool StampBlock(uint64_t page, uint64_t clock) noexcept {
	const uint64_t key = page + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 44u) % BlockSlots;
	for (size_t probe = 0; probe < 32; ++probe, i = (i + 1) % BlockSlots) {
		uint64_t current = g_block_pages[i].key.load(std::memory_order_acquire);
		if (current == 0) {
			// (The value first, the key published after it: a reader that finds the key reads this clock or a newer one.)
			if (g_block_pages[i].value.load(std::memory_order_relaxed) != 0) continue; // (another thread is taking it)
			uint64_t expected = 0;
			if (!g_block_pages[i].value.compare_exchange_strong(expected, clock, std::memory_order_acq_rel)) continue;
			g_block_pages[i].key.store(key, std::memory_order_release);
			return true;
		}
		if (current == key) {
			uint64_t old = g_block_pages[i].value.load(std::memory_order_relaxed);
			while (old < clock && !g_block_pages[i].value.compare_exchange_weak(old, clock, std::memory_order_acq_rel)) {
			}
			return true;
		}
	}
	return false;
}

// Counters since start (the lines print differences). Refusals: ARefused + frame * 4 + why (why 0 untracked or inside
// an upload, 1 GPU-dirty, 2 no-watch already - another watcher -, 3 refused before).
enum AnnounceCounter : int {
	ARefused         = 0, // 8 of them
	ABlockedRuns     = 8,
	ABlockedPages,
	AStampLost,
	AUnblockTried,
	AUnblockDone,
	AUnblockFailed,
	AUnblockSame,
	AUnblockUnknown,
	// (KYTY_RAIZ_VERIFY)
	AResetRuns,
	AResetPages,
	AChecks,
	ACheckedWritten,
	ACheckedDirty,
	ATruncated,
	AStale,               // 4 of them, by class: 0 frame (not DCB), 1 DCB (a render context's block), 2 registered range, 3 other
	AStaleWide = AStale + 4, // of them, in a use over 1 MiB or a BDA read
	ARepaired,
	// (anuncio2)
	AGpuWriteUses,       // GPU writes over no-watch pages (PolicyBeforeUseSlow, KYTY_RAIZ_ANNOUNCE)
	AGpuWriteKept,       // ... frame-policy pages left no-watch (CPU-dirty for the upload): no veto, no protection
	AGpuWriteReverted,   // ... other pages watched again and refused (as before)
	ACoveredMemoHits,    // ReadCovered answered by its memo (no page walk)
	ACoveredMemoMisses,  // ... walked (memo empty, stale, or a range with pages outside frame memory)
	AVetoOther,          // 3 of them, pages outside frame memory watched again and refused, by why (0 read, 1 GPU write, 2 BDA)
	AUploadBegins = AVetoOther + 3, // (verify) upload copies whose whole pages had their write watch reset just before
	AUploadSnaps,        // (verify) sampled pages whose clean copy was taken from the staging bytes
	ACleanAwaiting,      // (verify) uses that met a sampled page whose upload copy had not run yet (not compared)
	AnnounceCounterCount
};
std::atomic<uint64_t> g_an[AnnounceCounterCount] {};
void ACount(int counter, uint64_t value = 1) noexcept {
	g_an[counter].fetch_add(value, std::memory_order_relaxed);
}
void AnnounceSnapshot(uint64_t* out) noexcept {
	for (int i = 0; i < AnnounceCounterCount; ++i) out[i] = g_an[i].load(std::memory_order_relaxed);
}

// The code whose write faults the tracker refused, by (rip, frame, why): open addressing, a full neighbourhood drops it.
constexpr size_t RefusedSlots = 2048;
struct RefusedRip {
	std::atomic<uint64_t> key {0};   // rip << 3 | frame << 2 | why (+1 when 0)
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> page {0};  // the first page
};
RefusedRip g_refused_rips[RefusedSlots];
void NoteRefusedRip(uint64_t rip, bool frame, int why, uint64_t page) noexcept {
	const uint64_t key = ((rip << 3u) | (frame ? 4u : 0u) | static_cast<uint64_t>(why & 3)) + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 53u) % RefusedSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % RefusedSlots) {
		uint64_t current = g_refused_rips[i].key.load(std::memory_order_relaxed);
		if (current == 0 && g_refused_rips[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) {
			g_refused_rips[i].page.store(page, std::memory_order_relaxed);
			current = key;
		}
		if (current == key) {
			g_refused_rips[i].count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
}

// (KYTY_RAIZ_VERIFY) The pages found written with no announcement: page -> class, count, the use's size.
constexpr size_t StaleSlots = 1024;
struct StalePage {
	std::atomic<uint64_t> key {0}; // page number + 1
	std::atomic<uint64_t> count {0}, use_size {0};
	std::atomic<uint32_t> cls {0};
};
StalePage g_stale_pages[StaleSlots];
void NoteStalePage(uint64_t page, uint32_t cls, uint64_t use_size) noexcept {
	const uint64_t key = (page >> 12u) + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 54u) % StaleSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % StaleSlots) {
		uint64_t current = g_stale_pages[i].key.load(std::memory_order_relaxed);
		if (current == 0 && g_stale_pages[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) {
			g_stale_pages[i].cls.store(cls, std::memory_order_relaxed);
			g_stale_pages[i].use_size.store(use_size, std::memory_order_relaxed);
			current = key;
		}
		if (current == key) {
			g_stale_pages[i].count.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
}

// "[anuncio] ..." every 5 s (and ACUMULADO with the tables every 60 s) while the split or the frame policy is on: the
// tracker's refusals by reason (the counter the raiz log lacked), and with KYTY_RAIZ_ANNOUNCE what it did.
void PrintAnnounceLine(const uint64_t* now, const uint64_t* before, double seconds, const char* prefix) {
	uint64_t d[AnnounceCounterCount];
	for (int i = 0; i < AnnounceCounterCount; ++i) d[i] = now[i] - (before != nullptr ? before[i] : 0);
	const auto  u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	const uint64_t* r = d + ARefused;
	std::printf("[anuncio] %s%.1f s: faults que el tracker NO dejo sin vigilancia (por que): frame = escrita por la GPU %llu + ya sin "
	            "vigilancia (otro vigilante) %llu + VETADA antes %llu + sin region %llu; no frame = escrita por la GPU %llu + ya sin "
	            "vigilancia %llu + VETADA antes %llu + sin region %llu",
	            prefix, seconds, u(r[4 + 1]), u(r[4 + 2]), u(r[4 + 3]), u(r[4 + 0]), u(r[1]), u(r[2]), u(r[3]), u(r[0]));
	if (AnnounceOn()) {
		std::printf("; KYTY_RAIZ_ANNOUNCE: vetos anotados %llu paginas en %llu tramos (sin sitio %llu); paginas de frame vetadas y "
		            "REENTREGADAS -> sin vigilancia otra vez %llu (de %llu intentos; fallidas %llu); no levantadas: misma reserva %llu, "
		            "veto sin fecha %llu",
		            u(d[ABlockedPages]), u(d[ABlockedRuns]), u(d[AStampLost]), u(d[AUnblockDone]), u(d[AUnblockTried]), u(d[AUnblockFailed]),
		            u(d[AUnblockSame]), u(d[AUnblockUnknown]));
		// (anuncio2)
		std::printf("; (anuncio2) ESCRITURAS DE GPU sobre paginas sin vigilancia %llu usos: paginas de frame en ellos %llu (las sin vigilancia "
		            "SIGUEN asi: sin veto ni proteccion), otras vigiladas y vetadas %llu; vetos fuera de frame %llu paginas (lectura no cubierta %llu, escritura de GPU "
		            "%llu, BDA %llu); ReadCovered memo %llu aciertos / %llu recorridos",
		            u(d[AGpuWriteUses]), u(d[AGpuWriteKept]), u(d[AGpuWriteReverted]), u(d[AVetoOther] + d[AVetoOther + 1] + d[AVetoOther + 2]),
		            u(d[AVetoOther]), u(d[AVetoOther + 1]), u(d[AVetoOther + 2]), u(d[ACoveredMemoHits]), u(d[ACoveredMemoMisses]));
	}
	if (AnnounceVerifyOn()) {
		std::printf("; VERIFY (write watch): paginas puestas a cero al quedar limpias %llu en %llu tramos; usos comprobados %llu, paginas "
		            "escritas en ellos %llu (anunciadas o sucias %llu); SIN ANUNCIO (la GPU las usaria viejas) frame %llu + DCB %llu + rango "
		            "registrado %llu + otra %llu (en usos > 1 MiB o BDA %llu), reparadas %llu; usos con mas de 512 escritas %llu",
		            u(d[AResetPages]), u(d[AResetRuns]), u(d[AChecks]), u(d[ACheckedWritten]), u(d[ACheckedDirty]), u(d[AStale + 0]),
		            u(d[AStale + 1]), u(d[AStale + 2]), u(d[AStale + 3]), u(d[AStaleWide]), u(d[ARepaired]), u(d[ATruncated]));
		std::printf("; (anuncio2) copias de subida: write watch a cero justo antes %llu, copias limpias tomadas de lo subido %llu, usos "
		            "con la copia aun pendiente (no comparados) %llu",
		            u(d[AUploadBegins]), u(d[AUploadSnaps]), u(d[ACleanAwaiting]));
	}
	std::printf("\n");
}

// (anuncio2) The vetoes outside frame memory, by (why, the allocator's label of the range's first page, the use's size rounded up
// to a power of two): open addressing, a full neighbourhood drops it. Printed every 60 s with the other tables.
constexpr size_t VetoSlots = 512;
struct VetoRow {
	std::atomic<uint64_t> key {0}; // (why << 8 | log2 size) << 48 | label hash, + 1
	std::atomic<uint64_t> uses {0}, pages {0}, first {0}, size {0};
};
VetoRow g_veto_rows[VetoSlots];
void PrintVetoTable() {
	struct Row {
		uint64_t key, uses, pages, first, size;
	};
	std::vector<Row> rows;
	for (auto& slot: g_veto_rows) {
		const uint64_t key = slot.key.load(std::memory_order_relaxed);
		if (key != 0)
			rows.push_back({key - 1, slot.uses.load(std::memory_order_relaxed), slot.pages.load(std::memory_order_relaxed),
			                slot.first.load(std::memory_order_relaxed), slot.size.load(std::memory_order_relaxed)});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.pages > b.pages; });
	static const char* const why_name[3] = {"lectura que ninguna regla cubre", "escritura de la GPU", "buffer leido por BDA"};
	for (size_t k = 0; k < rows.size() && k < 10; ++k) {
		char label[24] = "";
		if (!DemonsSoulsMemMap::LabelAt(rows[k].first, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[anuncio]   vetos no frame x%llu usos, %llu paginas: %s, uso de %llu bytes (ej. 0x%llx, %s)\n",
		            static_cast<unsigned long long>(rows[k].uses), static_cast<unsigned long long>(rows[k].pages),
		            why_name[(rows[k].key >> 56u) & 3u], static_cast<unsigned long long>(rows[k].size),
		            static_cast<unsigned long long>(rows[k].first), label);
	}
}

// Every 60 s: who writes the refused pages (the code), and the pages the verify found written with no announcement.
void PrintAnnounceTables() {
	struct Row {
		uint64_t key, count, page;
	};
	std::vector<Row> rows;
	for (auto& slot: g_refused_rips) {
		const uint64_t key = slot.key.load(std::memory_order_relaxed);
		if (key != 0) rows.push_back({key - 1, slot.count.load(std::memory_order_relaxed), slot.page.load(std::memory_order_relaxed)});
	}
	std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.count > b.count; });
	static const char* const why_name[4] = {"sin region", "escrita por la GPU", "ya sin vigilancia", "VETADA antes"};
	for (size_t k = 0; k < rows.size() && k < 12; ++k) {
		char site[40], label[24] = "";
		Describe(rows[k].key >> 3u, site, sizeof(site));
		if (!DemonsSoulsMemMap::LabelAt(rows[k].page, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[anuncio]   rechazadas x%llu: %s, %s, escribe %s; primera 0x%llx (%s)\n", static_cast<unsigned long long>(rows[k].count),
		            (rows[k].key & 4u) != 0 ? "frame" : "no frame", why_name[rows[k].key & 3u], site,
		            static_cast<unsigned long long>(rows[k].page), label);
	}
	if (AnnounceOn()) PrintVetoTable(); // (anuncio2)
	if (!AnnounceVerifyOn()) return;
	struct Stale {
		uint64_t page, count, use_size;
		uint32_t cls;
	};
	std::vector<Stale> stale;
	for (auto& slot: g_stale_pages) {
		const uint64_t key = slot.key.load(std::memory_order_relaxed);
		if (key != 0)
			stale.push_back({(key - 1) << 12u, slot.count.load(std::memory_order_relaxed), slot.use_size.load(std::memory_order_relaxed),
			                 slot.cls.load(std::memory_order_relaxed)});
	}
	std::sort(stale.begin(), stale.end(), [](const Stale& a, const Stale& b) { return a.count > b.count; });
	static const char* const cls_name[4] = {"frame", "DCB", "rango registrado", "otra"};
	for (size_t k = 0; k < stale.size() && k < 12; ++k) {
		char label[24] = "";
		if (!DemonsSoulsMemMap::LabelAt(stale[k].page, label, sizeof(label)) || label[0] == 0) std::snprintf(label, sizeof(label), "sin etiqueta");
		std::printf("[anuncio]   SIN ANUNCIO x%llu: pagina 0x%llx (%s, %s), uso de %llu bytes\n", static_cast<unsigned long long>(stale[k].count),
		            static_cast<unsigned long long>(stale[k].page), cls_name[stale[k].cls & 3u], label,
		            static_cast<unsigned long long>(stale[k].use_size));
	}
}

void Thread() {
	using Clock      = std::chrono::steady_clock;
	auto   last_line = Clock::now(), last_tables = last_line;
	Totals before    = ReadTotals();
	uint64_t announce_before[AnnounceCounterCount] {}; // (anuncio1)
	AnnounceSnapshot(announce_before);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		if (!On()) continue;
		if (RefreshFrame()) g_frame_known.store(true, std::memory_order_relaxed);
		const auto now = Clock::now();
		// (policy1) The registry's clock, in 1/8 s.
		Reg::Registry::s_now.store(static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start).count() / 125),
		                           std::memory_order_relaxed);
		if (g_verify.load(std::memory_order_relaxed)) VerifyStep();
		if (g_frame_audit.load(std::memory_order_relaxed)) FaResolveLate(); // (policy4)
		// (policy6) The clock ~4 s ago: reservations older than it no longer make a page the frame policy's.
		{
			static uint64_t samples[8] {};
			static size_t   at = 0;
			g_fa_recent_clock.store(samples[at % 8], std::memory_order_relaxed);
			samples[at % 8] = g_fa_clock.load(std::memory_order_relaxed);
			++at;
		}
		const bool lines = g_lines.load(std::memory_order_relaxed);
		if (now - last_line >= std::chrono::seconds(5)) {
			// (policy1) Per-use ranges nothing announced for 8 s: taken out (their memory may be anything now).
			if (auto* const registry = g_registry.load(std::memory_order_relaxed); registry != nullptr) {
				try {
					if (const size_t swept = registry->SweepStale(64); swept != 0) Count(CStaleSwept, swept);
				} catch (...) {
				}
			}
			const Totals totals  = ReadTotals();
			const Totals delta   = Minus(totals, before);
			const double seconds = std::chrono::duration<double>(now - last_line).count();
			if (lines) {
				PrintLine(delta, seconds, "", g_frame_known.load(std::memory_order_relaxed));
				PrintReadLine(delta, seconds, "");
			}
			if (SplitOn()) PrintPolicyLine(delta, seconds, "");
			if (g_readback.load(std::memory_order_relaxed)) PrintReadbackLine(delta, seconds, "");
			if (g_frame_audit.load(std::memory_order_relaxed)) PrintFrameAuditLine(delta, seconds, "", false);
			if (FrameOn()) PrintFrameLine(delta, seconds, "", false);
			if (g_verify.load(std::memory_order_relaxed)) PrintVerifyLine(delta, seconds, "", false);
			// (anuncio1) Why the tracker refuses, and KYTY_RAIZ_ANNOUNCE's work.
			if (SplitOn()) {
				uint64_t announce_now[AnnounceCounterCount];
				AnnounceSnapshot(announce_now);
				PrintAnnounceLine(announce_now, announce_before, seconds, "");
				std::copy(std::begin(announce_now), std::end(announce_now), std::begin(announce_before));
			}
			// (FRAME-NATIVO) The label map of the protection counter, and every 30 s the "[frame-nativo]" line.
			FrameNativo::Refresh([](const char* const* l, size_t n, uint64_t* out, size_t max) {
				return DemonsSoulsMemMap::RangesOf(l, n, out, max);
			});
			PrintFrameNativoLine(totals, now);
			std::fflush(stdout);
			before    = totals;
			last_line = now;
		}
		if (now - last_tables >= std::chrono::seconds(60)) {
			const Totals totals  = ReadTotals();
			const double seconds = std::chrono::duration<double>(now - g_start).count();
			if (lines) {
				PrintTables(totals, seconds, g_frame_known.load(std::memory_order_relaxed));
				PrintReadTable(totals, seconds);
			}
			if (SplitOn()) {
				PrintPolicyLine(totals, seconds, "ACUMULADO ");
				PrintReverted();
			}
			if (g_readback.load(std::memory_order_relaxed)) {
				PrintReadbackLine(totals, seconds, "ACUMULADO ");
				if (!lines) PrintReadTable(totals, seconds); // (readback2) the sites of "resto", by code address
			}
			if (g_frame_audit.load(std::memory_order_relaxed)) PrintFrameAuditLine(totals, seconds, "ACUMULADO ", true);
			if (FrameOn()) PrintFrameLine(totals, seconds, "ACUMULADO ", true);
			if (g_verify.load(std::memory_order_relaxed)) PrintVerifyLine(totals, seconds, "ACUMULADO ", true);
			if (SplitOn()) { // (anuncio1)
				uint64_t announce_now[AnnounceCounterCount];
				AnnounceSnapshot(announce_now);
				PrintAnnounceLine(announce_now, nullptr, seconds, "ACUMULADO ");
				PrintAnnounceTables();
			}
			std::fflush(stdout);
			last_tables = now;
		}
	}
}

} // namespace

bool Wanted() noexcept {
	static const bool wanted = [] {
		const char* text = std::getenv("KYTY_DS_WRITE_AUDIT");
		return text != nullptr && std::strcmp(text, "1") == 0;
	}();
	return wanted;
}

bool HooksWanted() noexcept {
	return Wanted() || SplitWanted() != 0 || AnnouncedWanted() != 0 || VerifyWanted() || ReadbackWanted() || FrameAuditWanted() ||
	       FrameWanted() != 0;
}

bool WantsLinearCopy() noexcept {
	return HooksWanted() && (HookMask() & HookLinearCopy) != 0;
}

void NoteKeptPages(uint64_t total) noexcept {
	g_kept_pages.store(total, std::memory_order_relaxed);
}

void NoteLinearCopyHook(bool installed) noexcept {
	g_linear_copy_hook = installed;
}

void OnLinearCopy(uint64_t dst, int32_t dst_dwords, uint64_t src, int32_t /*src_dwords*/) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return;
	Count(CLinearCopy);
	if (dst_dwords <= 0) return;
	const uint64_t bytes = static_cast<uint64_t>(dst_dwords) * 4;
	if (InFrame(src)) {
		Count(CLcSrcFrame);
	} else if (src >= 0x10000) {
		// (policy1) The copy reads [src, src + bytes) now: the game's own word that those bytes are final. A source
		// inside a registered range announces it; any other becomes a per-use range of its own.
		if (registry->Find(src).where == Reg::Where::Inside) {
			Count(CLcSrcRegistered);
			(void)registry->Announce(src);
		} else {
			Count(CLcSrcUnknown);
			NoteGranule(3, src);
			try {
				if (!registry->Touch(src, bytes, Reg::LinearSrc, 0)) {
					bool         nested   = false;
					const size_t replaced = registry->Add(src, bytes, Reg::LinearSrc, 0, 0, false, true, &nested);
					if (replaced != 0) Count(CReplaced, replaced);
					if (!nested) Count(CLcSrcNew);
				}
			} catch (...) {
			}
		}
		if (SplitOn()) QueueRequest(src, bytes, false);
	}
	if (registry->Find(dst).where == Reg::Where::Inside) Count(CLcDstRegistered);
	else if (InFrame(dst)) Count(CLcDstFrame);
	else Count(CLcDstUnknown);
}

void NoteFaultSite(uint64_t rip, uint64_t rsp) noexcept {
	auto& state     = t_state;
	state.fault_rip = rip;
	state.fault_rsp = rsp;
}

void NoteWriteFault(uint64_t vaddr, int kind) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr || kind < 0 || kind >= static_cast<int>(FaultKinds)) return;
	auto&          state = t_state;
	const uint64_t rip   = state.fault_rip, rsp = state.fault_rsp;
	state.fault_rip = state.fault_rsp = 0;
	Class cls;
	if (InFrame(vaddr)) {
		cls = KFrame;
		uint64_t block_begin = 0, block_end = 0;
		const int arena = InSet(g_frame_ring, vaddr) ? 0
		                  : InSet(g_frame_ctx, vaddr) || DemonsSoulsMemMap::CtxBlockOf(vaddr, &block_begin, &block_end) ? 1 : 2;
		if (kind != 2) g_frame_faults[arena].fetch_add(1, std::memory_order_relaxed);
		// (policy3) KYTY_DS_FRAME_AUDIT: judged at the next lot submit.
		if (kind != 2 && g_frame_audit.load(std::memory_order_relaxed))
			FaNoteFault(vaddr, rip, arena, vaddr >= state.op_begin && vaddr < state.op_end && state.op_kind == OpCopy, state.gpu_thread);
	} else {
		// This thread's copy or texture read in progress covers the write; or another thread's texture read does.
		const bool in_op = vaddr >= state.op_begin && vaddr < state.op_end;
		Reg::Hit   hit   = registry->Find(vaddr);
		// (policy1) A per-use range nothing announced for a while: its memory may be anything now.
		if (hit.where == Reg::Where::Inside && !Reg::Registry::Fresh(hit.kind, hit.live, hit.stamp)) {
			Count(CStaleHit);
			hit.where = Reg::Where::None;
		}
		if (hit.where == Reg::Where::Inside) {
			if (in_op && state.op_kind == OpCopy) {
				cls = KAnnouncedNet;
				NoteSite(3, state.op_site);
			} else if (in_op || InTexLoad(vaddr)) {
				cls = KAnnounced;
			} else if (((1u << hit.kind) & Reg::PerUseKinds) != 0) {
				// (policy1) Memory the engine wraps or copies from at each use: its owner writes it between two uses.
				cls = KPerUse;
				NoteGranule(4, vaddr);
			} else if (hit.kind == Reg::Bink || hit.kind == Reg::CBufRewritten) {
				// (policy2) Memory the CPU writes all the time with no hook (the Bink decoder's block, the 0x20-byte
				// table): registered so that its pages keep their watchers, not as something that only hooks write.
				// Counted with the unknown writes, which is how the tracking treats them.
				cls = KUnknown;
				Count(CWriteUntrusted);
			} else {
				cls = KLeak;
				RecordLeak(vaddr, rip, rsp, kind, hit);
			}
		} else {
			cls = hit.where == Reg::Where::Neighbor ? KNeighbor : KUnknown;
			NoteGranule(cls == KNeighbor ? 2 : 1, vaddr);
			if (cls == KUnknown) {
				if (in_op) NoteSite(2, state.op_site);
				else NoteSite(1, rip);
			}
		}
		// (policy1) What the split would do with this fault's page at each level.
		if (kind != 2 && (cls == KNeighbor || cls == KUnknown || cls == KPerUse)) {
			const uint32_t kinds = registry->PageKinds(vaddr);
			// (policy2: a page with a range never trusted stays watched at every level.)
			const bool     one = (kinds & ~Reg::TrustedKinds(1)) == 0, two = (kinds & ~Reg::TrustedKinds(2)) == 0;
			if (cls == KPerUse) Count(two ? CWouldPerUse2 : CWouldPerUseNo);
			else if (kinds == 0) Count(CWouldUnknown);
			else if (cls == KUnknown) Count(CWouldNeighborNo);
			else Count(one ? CWouldNeighbor1 : two ? CWouldNeighbor2 : CWouldNeighborNo);
		}
	}
	g_faults[kind][cls].fetch_add(1, std::memory_order_relaxed);
}

// ---- (policy1) The read faults ------------------------------------------------------------------------------------

void NoteReadFault(uint64_t vaddr) noexcept {
	if (g_registry.load(std::memory_order_relaxed) == nullptr) return;
	auto&    state = t_state;
	uint64_t code  = state.fault_rip;
	state.fault_rip = state.fault_rsp = 0;
	bool in_copy = false;
	// The read is the source of the guest memcpy this thread is in: the game code is the copy's caller.
	if (state.op_source != 0 && state.op_end > state.op_begin && vaddr >= state.op_source &&
	    vaddr - state.op_source < state.op_end - state.op_begin && state.op_site != 0) {
		code    = state.op_site;
		in_copy = true;
	}
	uint32_t group = ReadGroups - 1;
	if (InEboot(code)) {
		const uint64_t rva = code - g_base;
		for (const auto& site: ReadSites) {
			if (rva >= site.begin && rva < site.end) {
				group = site.group;
				break;
			}
		}
	}
	g_reads[group].fetch_add(1, std::memory_order_relaxed);
	if (group != ReadGroups - 1 || code == 0) return;
	const uint64_t key = (code & (CopyFlag - 1)) | (in_copy ? CopyFlag : 0);
	g_read_sites.Add(key);
	size_t i = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 40u) % ReadFirstMax;
	for (size_t probe = 0; probe < 8; ++probe, i = (i + 1) % ReadFirstMax) {
		uint64_t current = g_read_first[i].site.load(std::memory_order_relaxed);
		if (current == key) break;
		if (current == 0 && g_read_first[i].site.compare_exchange_strong(current, key, std::memory_order_relaxed)) {
			g_read_first[i].address.store(vaddr, std::memory_order_relaxed);
			break;
		}
		if (current == key) break;
	}
}

// ---- (anuncio1) KYTY_RAIZ_ANNOUNCE ----------------------------------------------------------------------------------
void NoteBlockedRun(uint64_t begin, uint64_t end) noexcept {
	if (g_block_pages == nullptr || end <= begin) return;
	const uint64_t first = begin >> 12u, last = (end - 1) >> 12u;
	ACount(ABlockedRuns);
	ACount(ABlockedPages, last - first + 1);
	if (last - first + 1 > BlockRunMax) {
		ACount(AStampLost, last - first + 1);
		return;
	}
	// (The clock now: a reservation made after this returns has a larger one - FaNotePage's value >> 8.)
	const uint64_t clock = g_fa_clock.load(std::memory_order_acquire);
	for (uint64_t page = first; page <= last; ++page)
		if (!StampBlock(page, clock)) ACount(AStampLost);
}

bool UnblockWanted(uint64_t vaddr) noexcept {
	// (WantsNoWatch took it as the frame policy's page just before, on this thread: split_reason 3.)
	if (!AnnounceOn() || g_block_pages == nullptr || g_fa_pages == nullptr || t_state.split_reason != 3) return false;
	ACount(AUnblockTried);
	const uint64_t page    = vaddr >> 12u;
	const uint64_t blocked = BlockClock(page);
	if (blocked == BlockUnknown) {
		ACount(AUnblockUnknown);
		return false;
	}
	const uint64_t reserved = FaPageValue(page) >> 8u;
	if (reserved <= blocked) {
		// (Refused during the reservation it has now: the GPU writes this owner's bytes. Stays refused: no ping-pong.)
		ACount(AUnblockSame);
		return false;
	}
	return true;
}

void NoteUnblocked(uint64_t /*vaddr*/, bool done) noexcept {
	ACount(done ? AUnblockDone : AUnblockFailed);
}

void NoteNoWatchRefused(uint64_t vaddr, int why) noexcept {
	const bool frame = t_state.split_reason == 3;
	const int  w     = (why < 0 || why > 3) ? 0 : why;
	ACount(ARefused + (frame ? 4 : 0) + w);
	NoteRefusedRip(t_state.fault_rip, frame, w, vaddr & ~uint64_t {0xfff});
}

void NoteNoWatchCleanRun(uint64_t vaddr, uint64_t size) noexcept {
	if (!AnnounceVerifyOn() || size == 0) return;
	Libs::LibKernel::Memory::NativeAnnounceReset(vaddr, size);
	ACount(AResetRuns);
	ACount(AResetPages, (size + 0xfff) >> 12u);
}

void NoteUnannounced(uint64_t page, uint64_t /*use_begin*/, uint64_t use_size, bool wide, bool repaired) noexcept {
	uint32_t cls = 3;
	uint64_t block_begin = 0, block_end = 0;
	if (DemonsSoulsMemMap::CtxBlockOf(page, &block_begin, &block_end)) cls = 1;
	else if (InFrame(page)) cls = 0;
	else if (auto* const registry = g_registry.load(std::memory_order_relaxed); registry != nullptr) {
		try {
			if (registry->Find(page).where == Reg::Where::Inside) cls = 2;
		} catch (...) {
		}
	}
	ACount(AStale + static_cast<int>(cls));
	if (wide) ACount(AStaleWide);
	if (repaired) ACount(ARepaired);
	NoteStalePage(page, cls, use_size);
}

void NoteAnnounceCheck(uint64_t written, uint64_t dirty, uint64_t /*stale*/, bool truncated) noexcept {
	ACount(AChecks);
	ACount(ACheckedWritten, written);
	ACount(ACheckedDirty, dirty);
	if (truncated) ACount(ATruncated);
}

// ---- (policy1) KYTY_DS_NEIGHBOR_SPLIT ------------------------------------------------------------------------------
bool WantsNoWatch(uint64_t vaddr) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	const int   level    = g_split.load(std::memory_order_relaxed);
	const bool  frame    = g_frame.load(std::memory_order_relaxed) != 0;
	if (registry == nullptr || (level == 0 && !frame)) return false;
	// (policy4) While a Bink video plays (its callback 0x9808a0 ran in the last two seconds) no page is made no-watch:
	// the decoder writes its block, and whatever the engine wraps inside it, with no hook.
	if (g_bink_seen.load(std::memory_order_relaxed) &&
	    Reg::Registry::Now() - g_bink_tick.load(std::memory_order_relaxed) <= Reg::Registry::FreshTicks) {
		Count(PKeptBink);
		return false;
	}
	// Frame memory: never left without its watcher by the split. (policy6) The frame policy takes the pages a frame
	// allocator handed out lately (not the rings, not the small DCBs, not a block given back): R1-R3 mark them.
	// (VIGILANCIA-DOBLE) A page a native piece serves (NativoCubierto::MarcarCubierto): left without its watcher.
	if (::NativoCubierto::Cubierto(vaddr)) {
		t_state.split_reason = 2;
		return true;
	}
	if (InFrame(vaddr)) {
		if (frame && FrameRecentPage(vaddr) && !InSet(g_frame_ring, vaddr)) {
			t_state.split_reason = 3;
			::NativoCubierto::NoteFrameNoWatch(); // (VIGILANCIA-DOBLE: the [vigilancia] line)
			return true;
		}
		Count(PKeptFrame);
		return false;
	}
	if (level == 0) return false;
	const uint32_t trust = g_trust.load(std::memory_order_relaxed);
	const Reg::Hit hit   = registry->Find(vaddr);
	uint8_t        reason = 0; // 0 beside trusted ranges, 1 inside a per-use range, 2 no range in the page
	if (hit.where == Reg::Where::Inside && Reg::Registry::Fresh(hit.kind, hit.live, hit.stamp)) {
		// Inside a range: only one announced at each use (its owner writes it in between) is left alone. A write
		// inside a range that changes only through hooks is either a hooked write under way (it is announced when
		// it ends) or a leak: both keep the watcher.
		if (((1u << hit.kind) & trust & Reg::PerUseKinds) == 0) {
			Count(PKeptInside);
			return false;
		}
		reason = 1;
	}
	// Every registered range with bytes in the page must be one the hooks announce.
	const uint32_t kinds = registry->PageKinds(vaddr);
	if ((kinds & ~trust) != 0) {
		Count(PKeptUntrusted);
		return false;
	}
	if (kinds == 0) {
		if (level < 2) {
			Count(PKeptUnknown);
			return false;
		}
		reason = 2;
	}
	// (policy5) The diagnostic filter by allocator label (a scan of the map: once per page that would be made no-watch).
	if (!FilterAllows(vaddr)) {
		Count(PKeptFilter);
		return false;
	}
	t_state.split_reason = reason;
	return true;
}

void NoteNoWatch(uint64_t vaddr, bool done) noexcept {
	if (!done) {
		Count(PRefusedTracker);
		return;
	}
	const uint8_t reason = t_state.split_reason;
	Count(reason == 3 ? PNoWatchFrame : reason == 1 ? PNoWatchPerUse : reason == 2 ? PNoWatchUnknown : PNoWatchNeighbor);
	NoteSplitRip(vaddr >> 12u, t_state.fault_rip); // (policy4; the fault's instruction: NoteFaultSite)
	const uint64_t region = vaddr >> RegionShift;
	if (region < RegionWords * 64) g_nowatch_regions[region / 64].fetch_or(uint64_t {1} << (region % 64), std::memory_order_relaxed);
}

bool ReadCovered(uint64_t vaddr, uint64_t size) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr) return false;
	const uint32_t trust = g_trust.load(std::memory_order_relaxed);
	bool           covered;
	// (A BDA-read buffer over a whole arena block is 120 MiB: up to 256 MiB is walked, ~30k lookups at most; only the
	// rare whole-buffer calls are that long, the synchronizations go by 4 MiB pieces.)
	if (!FrameOn() || size == 0 || size > (uint64_t {256} << 20u)) {
		covered = trust != 0 && registry->Covered(vaddr, size, trust);
	} else {
		// (anuncio2, KYTY_RAIZ_ANNOUNCE) The memo: a range found made of frame-policy pages only stays covered while the
		// oldest reservation among its pages is still "recent" (g_fa_recent_clock only grows, a page's reservation
		// clock only grows: FaNotePage keeps the newest) and the rings' set is the one seen then. The walk below costs
		// one hash lookup a page (a 4 MiB synchronization piece: 1024): with the variable ~5x more ranges reach it
		// (035513: 1.04 M covered uses per 5 s; region_sync_us 1169 us/frame against raiz8's 382).
		struct Memo {
			uint64_t vaddr = 0, size = 0, oldest = 0;
			uint32_t ring = 0;
		};
		constexpr size_t  MemoSlots = 256;
		thread_local Memo memo[MemoSlots];
		const bool        use_memo = AnnounceOn();
		// (The rings' set as it is now: which buffer and its sequence number. The audit's thread republishes it every 500 ms.)
		const int      ring_current = g_frame_ring.current.load(std::memory_order_acquire);
		const uint32_t ring = ring_current < 0 ? 0u
		                                       : (static_cast<uint32_t>(ring_current + 1) << 28u) ^
		                                             g_frame_ring.sets[ring_current].seq.load(std::memory_order_acquire);
		Memo* const    slot = &memo[static_cast<size_t>(((vaddr >> 12u) ^ (size << 7u)) * 0x9E3779B97F4A7C15ull >> 56u) % MemoSlots];
		if (use_memo && slot->vaddr == vaddr && slot->size == size && slot->ring == ring &&
		    slot->oldest >= g_fa_recent_clock.load(std::memory_order_relaxed)) {
			ACount(ACoveredMemoHits);
			Count(PReadCovered);
			return true;
		}
		if (use_memo) ACount(ACoveredMemoMisses);
		// (policy6) Page by page: the frame policy's pages are covered by its rules (R1-R3); the runs of other pages
		// need the split's trusted ranges.
		covered             = true;
		bool           all_ours = true;
		uint64_t       oldest   = ~uint64_t {0};
		const uint64_t end  = vaddr + size;
		uint64_t       run  = 0; // (start of a run of other pages; 0: none)
		for (uint64_t at = vaddr; at < end && covered;) {
			const uint64_t next  = (std::min)((at | 0xfff) + 1, end);
			// (FrameRecentPage, with the page's reservation clock kept for the memo.)
			const uint64_t value = g_fa_pages != nullptr ? FaPageValue(at >> 12u) : 0;
			const bool     ours  = value != 0 && (value >> 8u) >= g_fa_recent_clock.load(std::memory_order_relaxed) && !InSet(g_frame_ring, at);
			if (ours) oldest = (std::min)(oldest, value >> 8u);
			if (!ours) all_ours = false;
			if (!ours && run == 0) run = at;
			if ((ours || next == end) && run != 0) {
				const uint64_t stop = ours ? at : end;
				covered             = trust != 0 && registry->Covered(run, stop - run, trust);
				run                 = 0;
			}
			at = next;
		}
		if (use_memo) {
			if (covered && all_ours && oldest != ~uint64_t {0}) *slot = {vaddr, size, oldest, ring};
			else if (slot->vaddr == vaddr && slot->size == size) *slot = {};
		}
	}
	if (covered) Count(PReadCovered);
	return covered;
}

// ---- (policy6) KYTY_DS_WRITE_FRAME -----------------------------------------------------------------------------------
void NoteSubmitted(const void* commands, uint64_t size_in_bytes) noexcept {
	const bool audit = g_frame_audit.load(std::memory_order_relaxed), frame = FrameOn();
	if ((!audit && !frame) || commands == nullptr || size_in_bytes == 0) return;
	FaSubmit submit;
	submit.clock = g_fa_clock.fetch_add(1, std::memory_order_relaxed) + 1;
	submit.begin = reinterpret_cast<uint64_t>(commands);
	submit.end   = submit.begin + size_in_bytes;
	// The reservation that holds the command buffer (its chunk: the packets up, the embedded data down from its top)
	// and the context's block the memory map knows (100 ms old at most).
	(void)ReservationOf(submit.begin, &submit.chunk_begin, &submit.chunk_end);
	(void)DemonsSoulsMemMap::CtxBlockOf(submit.begin, &submit.block_begin, &submit.block_end);
	if (frame) {
		Count(FrSubmitMarks);
		FrMark(submit.begin, submit.end);
		if (submit.chunk_end > submit.chunk_begin) FrMark(submit.chunk_begin, submit.chunk_end);
		if (submit.block_end > submit.block_begin) FrMark(submit.block_begin, submit.block_end);
		// (A command buffer that runs past its first reservation: the last page's too.)
		uint64_t last_begin = 0, last_end = 0;
		if (submit.end - 1 >= submit.chunk_end && ReservationOf(submit.end - 1, &last_begin, &last_end)) FrMark(last_begin, last_end);
	}
	if (audit && g_fa_submits_kept != nullptr) {
		std::lock_guard lock(g_fa_submit_mutex);
		g_fa_submits_kept[g_fa_submit_next++ % FaSubmitKeep] = submit;
	}
}

size_t TakeFrameMarks(Request* out, size_t max, bool* everything) noexcept {
	if (g_fr_everything.exchange(false, std::memory_order_acq_rel) && everything != nullptr) *everything = true;
	if (g_fr_marks == nullptr || g_fr_pending.load(std::memory_order_relaxed) == 0) return 0;
	std::lock_guard lock(g_fr_mark_mutex);
	auto&           marks = *g_fr_marks;
	const size_t    count = (std::min)(max, marks.size());
	// (From the end: the order does not matter, each mark is applied whole.)
	for (size_t i = 0; i < count; ++i) {
		out[i] = marks.back();
		marks.pop_back();
	}
	g_fr_pending.store(marks.size(), std::memory_order_relaxed);
	return count;
}

void NoteGpuThread() noexcept {
	t_state.gpu_thread = true;
}

void NoteFrameDrained(uint64_t pages) noexcept {
	Count(FrDrainPages, pages);
}

// KYTY_DS_WRITE_FRAME=2: the stale-use check's copies of the sampled pages when they became clean, by page (open
// addressing; a full neighbourhood drops the page). (policy7: a copy, not a hash: the check compares only the bytes the
// use reads.) 8192 pages, 32 MiB, made at Install with =2.
constexpr size_t CleanSlots = 8192;
struct CleanPage {
	std::atomic<uint64_t> key {0};   // (page number + 1)
	std::atomic<uint32_t> state {0}; // 0 none, 1 being written, 2 valid
	uint64_t              value = 0; // the page's reservation then (frame pages)
	uint8_t               bytes[4096];
};
CleanPage* g_clean = nullptr;
CleanPage* CleanSlot(uint64_t page_address, bool insert) noexcept {
	if (g_clean == nullptr) return nullptr;
	const uint64_t key = (page_address >> 12u) + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 48u) % CleanSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % CleanSlots) {
		uint64_t current = g_clean[i].key.load(std::memory_order_relaxed);
		if (current == 0 && insert && g_clean[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) current = key;
		if (current == key) return &g_clean[i];
		if (current == 0) return nullptr;
	}
	return nullptr;
}

void NoteNoWatchClean(uint64_t page_address) noexcept {
	if (!CheckStaleOn()) return;
	CleanPage* slot = CleanSlot(page_address, true);
	if (slot == nullptr) return;
	uint32_t expected = slot->state.load(std::memory_order_acquire);
	if (expected == 1 || !slot->state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) return; // (busy)
	const bool ok = Common::GuardedCopy(slot->bytes, reinterpret_cast<const void*>(page_address), sizeof(slot->bytes)) == 0;
	slot->value   = g_fa_pages != nullptr ? FaPageValue(page_address >> 12u) : 0;
	// (anuncio2, KYTY_RAIZ_ANNOUNCE) 3: the upload's copy has not run yet; NoteUploadCopied puts the bytes it read and
	// makes it 2. Until then a use is not compared (its bytes may be uploaded still).
	slot->state.store(ok ? (UploadCopyHookOn() ? 3u : 2u) : 0u, std::memory_order_release);
	if (ok) Count(VCleanNoted);
}

bool HasCleanHash(uint64_t page_address) noexcept {
	const CleanPage* slot = CleanSlot(page_address, false);
	return slot != nullptr && slot->state.load(std::memory_order_acquire) == 2;
}

void CheckNoWatchUse(uint64_t page_address, bool still_clean, uint64_t use_begin, uint64_t use_end, bool wide) noexcept {
	CleanPage* slot = CleanSlot(page_address, false);
	if (slot == nullptr) return;
	uint32_t expected = 2;
	if (!slot->state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
		if (expected == 3 && still_clean) ACount(ACleanAwaiting); // (anuncio2: its copy has not run yet)
		return; // (none, or busy)
	}
	if (!still_clean) { // (it will be uploaded again: a new copy when it is clean again)
		slot->state.store(0, std::memory_order_release);
		return;
	}
	Count(VUseChecked);
	alignas(8) static thread_local uint8_t now[4096];
	if (Common::GuardedCopy(now, reinterpret_cast<const void*>(page_address), sizeof(now)) != 0) {
		slot->state.store(0, std::memory_order_release);
		return;
	}
	// The bytes that differ: the first and how many, in the use's range and outside it.
	const uint64_t from = use_begin > page_address ? use_begin - page_address : 0;
	const uint64_t to   = use_end < page_address + 4096 ? (use_end > page_address ? use_end - page_address : 0) : 4096;
	uint32_t       inside = 0, outside = 0, first = UINT32_MAX;
	for (uint32_t i = 0; i < 4096; ++i) {
		if (now[i] == slot->bytes[i]) continue;
		if (i >= from && i < to) {
			if (inside++ == 0) first = i;
		} else {
			outside++;
		}
	}
	const uint64_t value_now   = g_fa_pages != nullptr ? FaPageValue(page_address >> 12u) : 0;
	const bool     rereserved  = slot->value != 0 && value_now != slot->value;
	std::memcpy(slot->bytes, now, sizeof(now)); // (once per change)
	slot->value = value_now;
	slot->state.store(2, std::memory_order_release);
	if (inside == 0 && outside == 0) return;
	// Classes: a reservation handed the page out again since (R1 marks it at the lot to come); bytes changed only
	// outside the range this use reads; a use too wide to tell (a BDA buffer, over 1 MiB); else STALE.
	if (rereserved) {
		Count(VUseRereserved);
		return;
	}
	if (inside == 0) {
		Count(VUseOutside);
		return;
	}
	if (wide) {
		Count(VUseWide);
		return;
	}
	const bool frame = FrameRecentPage(page_address);
	Count(VUseStale);
	if (frame) Count(VUseStaleFrame);
	const uint64_t rip = SplitRipOf(page_address >> 12u);
	std::lock_guard lock(g_stale_mutex);
	size_t          i = 0;
	while (i < g_stale_count && !(g_stale[i].rip == rip && g_stale[i].frame == frame)) ++i;
	if (i == g_stale_count) {
		if (g_stale_count == StaleMax) return;
		g_stale[g_stale_count++] = {page_address + first, rip, 0, frame, inside, use_end - use_begin};
	}
	g_stale[i].count++;
}

// ---- (anuncio2) ----------------------------------------------------------------------------------------------------
bool FramePolicyPage(uint64_t address) noexcept {
	return FrameOn() && FrameRecentPage(address) && !InSet(g_frame_ring, address);
}

void NoteGpuWriteUse(uint64_t kept, uint64_t reverted) noexcept {
	ACount(AGpuWriteUses);
	if (kept != 0) ACount(AGpuWriteKept, kept);
	if (reverted != 0) ACount(AGpuWriteReverted, reverted);
}

void NoteVetoOther(uint64_t vaddr, uint64_t size, uint64_t pages, int why) noexcept {
	if (pages == 0 || InFrame(vaddr)) return;
	const int w = (why < 0 || why > 2) ? 0 : why;
	ACount(AVetoOther + w, pages);
	char label[24] = "";
	(void)DemonsSoulsMemMap::LabelAt(vaddr, label, sizeof(label));
	uint64_t hash = 1469598103934665603ull;
	for (const char* c = label; *c != 0; ++c) hash = (hash ^ static_cast<uint8_t>(*c)) * 1099511628211ull;
	uint64_t log2 = 0;
	while (log2 < 63 && (uint64_t {1} << log2) < size) ++log2;
	const uint64_t key = ((static_cast<uint64_t>(w) << 8u | log2) << 48u | (hash & ((uint64_t {1} << 48u) - 1))) + 1;
	size_t         i   = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 55u) % VetoSlots;
	for (size_t probe = 0; probe < 16; ++probe, i = (i + 1) % VetoSlots) {
		uint64_t current = g_veto_rows[i].key.load(std::memory_order_relaxed);
		if (current == 0 && g_veto_rows[i].key.compare_exchange_strong(current, key, std::memory_order_relaxed)) {
			g_veto_rows[i].first.store(vaddr, std::memory_order_relaxed);
			g_veto_rows[i].size.store(size, std::memory_order_relaxed);
			current = key;
		}
		if (current == key) {
			g_veto_rows[i].uses.fetch_add(1, std::memory_order_relaxed);
			g_veto_rows[i].pages.fetch_add(pages, std::memory_order_relaxed);
			return;
		}
	}
}
// The upload worker (or the GPU thread for a copy it makes itself), just before / after it reads guest memory for an
// upload. Begin: KYTY_RAIZ_VERIFY's write watch of the copy's whole pages starts over (a page only partly copied keeps
// the reset made when it became clean: bytes of it outside the copy may have been written since). Copied: the sampled
// pages' clean copies take the bytes uploaded.
void NoteUploadCopyBegin(uint64_t guest, uint64_t size) noexcept {
	if (!AnnounceVerifyOn() || size == 0) return;
	const uint64_t first = (guest + 0xfff) & ~uint64_t {0xfff}, last = (guest + size) & ~uint64_t {0xfff};
	if (last <= first) return;
	Libs::LibKernel::Memory::NativeAnnounceReset(first, last - first);
	ACount(AUploadBegins);
}
void NoteUploadCopied(uint64_t guest, const uint8_t* bytes, uint64_t size) noexcept {
	if (!CheckStaleOn() || bytes == nullptr || size == 0) return;
	const uint64_t end = guest + size;
	for (uint64_t page = guest & ~uint64_t {0xfff}; page < end; page += 0x1000) {
		if (!SampledPage(page)) continue;
		CleanPage* slot = CleanSlot(page, false);
		if (slot == nullptr) continue;
		uint32_t expected = slot->state.load(std::memory_order_acquire);
		if ((expected != 2 && expected != 3) || !slot->state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) continue;
		const uint64_t from = (std::max)(page, guest), to = (std::min)(page + 0x1000, end);
		std::memcpy(slot->bytes + (from - page), bytes + (from - guest), to - from);
		slot->state.store(2, std::memory_order_release);
		ACount(AUploadSnaps);
	}
}

void NoteReverted(uint64_t vaddr, uint64_t pages, int why) noexcept {
	if (why == 2) {
		Count(PRevertedBdaUses);
		Count(PRevertedPagesBda, pages);
	} else {
		Count(PRevertedUses);
		Count(why == 1 ? PRevertedPagesGpu : PRevertedPagesUse, pages);
	}
	if (pages == 0) return;
	NoteGranule(5, vaddr);
	// (policy4) Listed by reason and by the code that wrote the page when it was made no-watch (the range's first
	// page that has one).
	uint64_t rip = 0, address = vaddr;
	for (uint64_t page = vaddr >> 12u; rip == 0 && page <= (vaddr + 0x10000) >> 12u; ++page) {
		rip = SplitRipOf(page);
		if (rip != 0) address = page << 12u;
	}
	std::lock_guard lock(g_reverted_mutex);
	size_t i = 0;
	while (i < g_reverted_count && !(g_reverted[i].rip == rip && g_reverted[i].why == why)) ++i;
	if (i == g_reverted_count) {
		if (g_reverted_count == RevertedMax) return;
		g_reverted[g_reverted_count++] = {address, rip, 0, 0, why};
	}
	g_reverted[i].uses++;
	g_reverted[i].pages += pages;
}

size_t TakeRequests(Request* out, size_t max, bool* everything) noexcept {
	const bool revert_all = g_revert_all.exchange(false, std::memory_order_acq_rel); // (policy4: a video began)
	if (revert_all && everything != nullptr) *everything = true;
	if (g_request_pending.load(std::memory_order_relaxed) == 0 && !g_request_lost) return 0;
	std::lock_guard lock(g_request_mutex);
	if (g_request_lost) {
		g_request_lost = false;
		if (everything != nullptr) *everything = true;
	}
	size_t count = 0;
	while (count < max && g_request_count != 0) {
		out[count++]   = g_requests[g_request_head];
		g_request_head = (g_request_head + 1) % RequestMax;
		g_request_count--;
	}
	g_request_pending.store(g_request_count, std::memory_order_relaxed);
	return count;
}

void NoteDrained(uint64_t dirty_pages, uint64_t reverted_pages, bool everything) noexcept {
	Count(PDirtyPages, dirty_pages);
	Count(PRevertedPagesHook, reverted_pages);
	if (everything) Count(PEverything);
}

void NoteSweptRun(uint64_t vaddr, uint64_t size) noexcept {
	auto* const registry = g_registry.load(std::memory_order_relaxed);
	if (registry == nullptr || size == 0 || vaddr + size < vaddr) return;
	Count(CSweptRuns);
	// By bytes: those inside the frame arenas, those inside registered ranges (outside the arenas), the rest.
	const uint64_t frame = FrameOverlap(vaddr, vaddr + size);
	uint64_t registered = frame == size ? 0 : registry->Overlap(vaddr, size);
	if (registered > size - frame) registered = size - frame;
	Count(CSweptFrameBytes, frame);
	Count(CSweptRegisteredBytes, registered);
	Count(CSweptUnknownBytes, size - frame - registered);
}

void BeginCopy(uint64_t return_address, uint64_t destination, uint64_t size, uint64_t* saved, uint64_t source) noexcept {
	auto& state = t_state;
	saved[0]    = state.op_begin;
	saved[1]    = state.op_end;
	saved[2]    = state.op_site;
	saved[3]    = state.op_kind;
	saved[4]    = state.op_source;
	if (size == 0 || destination + size < destination) return;
	if (!g_copies.load(std::memory_order_relaxed) || return_address == 0) return; // (left out: copies are not looked at)
	uint64_t kind = OpCopy;
	// A copy inside the texture read this thread is in: the read's.
	if (state.op_kind == OpTexLoad && destination >= state.op_begin && destination + size <= state.op_end) {
		kind = OpTexLoad;
	} else if (return_address == g_stream_copy_return.load(std::memory_order_relaxed)) {
		kind = OpStreamCopy;
		Count(CStreamCopy);
		if (auto* const registry = g_registry.load(std::memory_order_relaxed); registry != nullptr && destination >= 0x10000) {
			try {
				// (policy1) The chunk belongs to its texture's virtual block ([texture + 0x98]: the texture this thread's
				// SetCurrentStreamingChunk was entered with): the block's free ends it. audit1 registered it by its own
				// start only, which no free names: the ranges outlived their textures (every FUGA of its first run).
				uint64_t owner = 0;
				if (state.stream_tex == 0 || !Peek(state.stream_tex + 0x98, &owner) || owner < 0x10000 || destination < owner ||
				    destination - owner >= (uint64_t {1} << 36u))
					owner = 0;
				if (owner != 0) Count(CStreamOwned);
				const size_t replaced = registry->Add(destination, size, Reg::TexStream, 0, 0, false, false, nullptr, owner);
				if (replaced != 0) Count(CReplaced, replaced);
			} catch (...) {
			}
		}
	} else if (return_address == g_create_copy_return.load(std::memory_order_relaxed)) {
		kind = OpCreateCopy;
		Count(CCreateCopy);
		// (Create's copy runs before Create's exit hook: registered here, pending, so that the copy's own faults find
		// it; the exit hook confirms it, or takes it out when the buffer is per-frame.)
		if (auto* const registry = g_registry.load(std::memory_order_relaxed);
		    registry != nullptr && destination >= 0x10000 && !InFrame(destination)) {
			try {
				const size_t replaced = registry->Add(destination, size, Reg::CBufData, 0, 0, true, false);
				if (replaced != 0) Count(CReplaced, replaced);
			} catch (...) {
			}
		}
	}
	state.op_begin  = destination;
	state.op_end    = destination + size;
	state.op_site   = return_address;
	state.op_kind   = kind;
	state.op_source = source;
}

void EndCopy(const uint64_t* saved) noexcept {
	auto& state = t_state;
	// (policy1) A copy the hooks know ended: the range's generation moves (its bytes are final now) and its no-watch
	// pages are written. (Any other copy into a registered range - the memcpy net - announces it too.)
	if (state.op_kind == OpStreamCopy || state.op_kind == OpCreateCopy || state.op_kind == OpCopy) {
		// (The net costs every memcpy of the game one or two loads: only while a policy or the verification is on.)
		if (auto* const registry = g_registry.load(std::memory_order_relaxed);
		    registry != nullptr && state.op_end > state.op_begin &&
		    (state.op_kind != OpCopy ||
		     ((SplitOn() || g_verify.load(std::memory_order_relaxed)) && registry->MayContain(state.op_begin, state.op_end - state.op_begin)))) {
			const bool known = registry->Announce(state.op_begin);
			if (known && SplitOn()) QueueRequest(state.op_begin, state.op_end - state.op_begin, false);
		}
	}
	// (policy6) R3: a guest memcpy the emulator ran into frame memory: its destination is written. (policy7) Only when
	// R1 will not mark it anyway: the destination's reservation is older than the second next lot's bound (the next
	// lot may be being submitted right now). policy6 marked every one: ~1.2 million a second.
	if (state.op_kind == OpCopy && state.op_end > state.op_begin && FrameOn() && g_fa_pages != nullptr) {
		const uint64_t value = FaPageValue(state.op_begin >> 12u);
		if (value != 0 && (value >> 8u) >= g_fa_recent_clock.load(std::memory_order_relaxed)) {
			const uint64_t kickoffs = g_fa_kickoffs.load(std::memory_order_acquire);
			const uint64_t lot      = g_fa_submits.load(std::memory_order_acquire) + 1;
			const uint64_t bound    = lot >= g_fr_margin ? FaKickClock(lot - g_fr_margin, kickoffs) : 0;
			if (bound != 0 && (value >> 8u) >= bound) {
				Count(FrCopySkipped);
			} else {
				Count(FrCopyMarks);
				FrMark(state.op_begin, state.op_end);
			}
		}
	}
	state.op_source = saved[4];
	state.op_begin = saved[0];
	state.op_end   = saved[1];
	state.op_site  = saved[2];
	state.op_kind  = saved[3];
}

void Install(Program* program) {
#if defined(__x86_64__) || defined(_M_X64)
	// (policy1) The variables that only say they are not implemented (once, whatever else is set).
	static bool told = false;
	if (!told && program != nullptr && program->file_name.filename() == "eboot.bin") {
		told = true;
		// (policy6: KYTY_DS_WRITE_FRAME is implemented now.)
		std::fflush(stdout);
	}
	if (!HooksWanted() || On() || g_registry.load(std::memory_order_relaxed) != nullptr || program == nullptr ||
	    program->file_name.filename() != "eboot.bin" || !Libs::Graphics::DemonsSouls::IsSupportedGame())
		return;
	const uint32_t mask = HookMask();
	std::printf("[write-audit] KYTY_DS_WRITE_AUDIT=%d: %s. Ganchos pedidos 0x%03x (KYTY_DS_WRITE_AUDIT_HOOKS). Politicas pedidas: "
	            "KYTY_DS_NEIGHBOR_SPLIT=%d, KYTY_DS_WRITE_ANNOUNCED=%d, KYTY_DS_WRITE_VERIFY=%d\n",
	            Wanted() ? 1 : 0,
	            Wanted() ? "auditoria de escrituras y lecturas (sus lineas solo miden)" : "sin lineas de auditoria (ganchos y registro para las politicas)",
	            mask, SplitWanted(), AnnouncedWanted(), VerifyWanted() ? 1 : 0);
	if (program->mapped_size >= CaveOffset || program->mapped_size < 0x1e00000 ||
	    program->base_vaddr > UINT64_MAX - CaveOffset - CaveSize) {
		std::printf("[write-audit] NO activa: el eboot no tiene el tamano del de 01.007.000\n");
		std::fflush(stdout);
		return;
	}
	g_base = program->base_vaddr;
	g_size = program->mapped_size;
	// The version: CBuffer::Create's head and exit. Without them this is not the audited build: nothing at all.
	if (!Matches("version", CreateFunction, CreateHead, sizeof(CreateHead)) ||
	    !Matches("version", CreateCallSite - 6, CreateExit, sizeof(CreateExit))) {
		std::printf("[write-audit] NO activa: el codigo de CBuffer::Create no es el de 01.007.000 (otra version, o ya parcheado)\n");
		std::fflush(stdout);
		return;
	}
	Reg::Registry* registry = nullptr;
	try {
		registry = new Reg::Registry();
	} catch (...) {
		std::printf("[write-audit] NO activa: sin memoria para el registro\n");
		std::fflush(stdout);
		return;
	}
	const auto host = [](auto* function) { return reinterpret_cast<uint64_t>(function); };

	// The cave and its stubs.
	std::array<uint8_t, 0x1000> image {};
	image.fill(0xcc);
	Stub::Emitter emitter;
	emitter.buffer   = image.data();
	emitter.capacity = image.size();
	emitter.address  = g_base + CaveOffset;
	const auto queue_entry = [&](const char* name, uint32_t bit, uint64_t rva, const uint8_t* head, size_t head_size, size_t moved,
	                             uint64_t function) {
		if ((mask & bit) == 0) {
			std::printf("[write-audit]   %s: fuera por la mascara\n", name);
			return false;
		}
		if (!Matches(name, rva, head, head_size)) {
			std::printf("[write-audit]   %s: NO instalado (el codigo no coincide)\n", name);
			return false;
		}
		const uint64_t stub = Stub::EmitEntryStub(emitter, function, head, moved, g_base + rva);
		if (!emitter.ok || !Queue(name, rva, moved, 0xe9, stub, bit)) {
			std::printf("[write-audit]   %s: NO instalado (la cueva queda fuera de alcance)\n", name);
			return false;
		}
		return true;
	};
	const auto queue_call = [&](const char* name, uint32_t bit, uint64_t site, uint64_t target, const uint8_t* around, size_t around_size,
	                            size_t before_site, uint64_t first, uint64_t second, int kind = 0) {
		if ((mask & bit) == 0) {
			std::printf("[write-audit]   %s: fuera por la mascara\n", name);
			return false;
		}
		if (!Matches(name, site - before_site, around, around_size) ||
		    Stub::RelativeTarget(reinterpret_cast<const uint8_t*>(g_base + site), g_base + site) != g_base + target) {
			std::printf("[write-audit]   %s: NO instalado (el codigo no coincide)\n", name);
			return false;
		}
		// (policy1) kind 1: the host is told after the call, with the caller's frame pointer; 2: around a call that takes
		// stack arguments.
		const uint64_t stub = kind == 2   ? Stub::EmitCallAroundStackStub(emitter, first, second, g_base + target)
		                      : kind == 1 ? Stub::EmitCallAfterFrameStub(emitter, first, g_base + target)
		                      : second != 0 ? Stub::EmitCallAroundStub(emitter, first, second, g_base + target)
		                                    : Stub::EmitCallAfterStub(emitter, first, g_base + target);
		if (!emitter.ok || !Queue(name, site, 5, 0xe8, stub, bit)) {
			std::printf("[write-audit]   %s: NO instalado (la cueva queda fuera de alcance)\n", name);
			return false;
		}
		return true;
	};
	bool any = false;
	any = queue_call("create (eboot+0x909819)", HookCreate, CreateCallSite, CreateCallTarget, CreateExit, sizeof(CreateExit), 6,
	                 host(&OnCreateAfter), 0, 1) || any;
	any = queue_entry("destroy (eboot+0x9093c0)", HookDestroy, DestroyA, DestroyHead, sizeof(DestroyHead), 7, host(&OnDestroy)) || any;
	any = queue_entry("destroy (eboot+0x909410)", HookDestroy, DestroyB, DestroyHead, sizeof(DestroyHead), 7, host(&OnDestroy)) || any;
	any = queue_entry("decommit (eboot+0x7f9f70)", HookDecommit, VirtualOp, VirtualOpHead, sizeof(VirtualOpHead), 6, host(&OnVirtualOp)) || any;
	any = queue_call("texload (eboot+0x8fed17)", HookTexLoad, TexLoadCallSite, TexLoadTarget, TexLoadCall, sizeof(TexLoadCall), 3,
	                 host(&OnTexLoadBefore), host(&OnTexLoadAfter)) || any;
	any = queue_entry("free diferido (eboot+0x903460)", HookDeferredFree, DeferredFreeEntry, DeferredFreeHead, sizeof(DeferredFreeHead), 6,
	                  host(&OnDeferredFree)) || any;
	any = queue_entry("free heap (eboot+0x7fa380)", HookHeapFree, HeapFreeEntry, HeapFreeHead, sizeof(HeapFreeHead), 6, host(&OnHeapFree)) || any;
	// (policy1) Each entry below begins push rbp; mov rbp,rsp; push r15 (6 bytes moved, none position-dependent).
	any = queue_entry("linear copy ACB (eboot+0x914e10)", HookAcbCopy, AcbCopyEntry, AcbCopyHead, sizeof(AcbCopyHead), 6,
	                  host(&OnLinearCopyAcb)) || any;
	any = queue_entry("entrada de streaming (eboot+0x901330)", HookStreamEntry, StreamEntry, StreamHead, sizeof(StreamHead), 6,
	                  host(&OnStreamEntry)) || any;
	any = queue_entry("bink (eboot+0x9808a0)", HookBink, BinkSubmitEntry, BinkHead, sizeof(BinkHead), 6, host(&OnBinkSubmit)) || any;
	// (policy2) KYTY_DS_READBACK_HOOK=1 only: the two readback accessors' entries (6 bytes moved each: two whole
	// instructions, no relative operand; the stub runs them after the host call, so the flags 0x909970's cmp sets
	// reach its je).
	bool read_pointer = false, read_copy = false, read_vis = false, read_mask = false, read_c12 = false;
	if (ReadbackWanted()) {
		read_pointer = queue_entry("readback, puntero (eboot+0x909940)", HookReadback, ReadPointerEntry, ReadPointerHead,
		                           sizeof(ReadPointerHead), 6, host(&OnReadPointer));
		read_copy = queue_entry("readback, copia (eboot+0x909970)", HookReadback, ReadCopyEntry, ReadCopyHead, sizeof(ReadCopyHead), 6,
		                        host(&OnReadCopy));
		read_vis = queue_entry("readback, lista de compute (eboot+0xa2e6f0)", HookReadback, VisListEntry, VisListHead, sizeof(VisListHead),
		                       6, host(&OnVisList));
		any = read_pointer || read_copy || read_vis || any;
		if (ReadHook2Wanted()) { // (READ-HOOK2)
			read_mask = queue_entry("readback, mascara de bits (eboot+0xadabb0)", HookReadback, ChunkMaskEntry, ChunkMaskHead,
			                        sizeof(ChunkMaskHead), 6, host(&OnChunkMask));
			if (Copy12Wanted())
			read_c12  = queue_entry("readback, copia de 12 bytes (eboot+0x8e9760)", HookReadback, Copy12Entry, Copy12Head, sizeof(Copy12Head),
			                        6, host(&OnCopy12));
			any = read_mask || read_c12 || any;
		}
	}
	// (raiz12) KYTY_RAIZ_SETTLE=1/2 (KYTY_RAIZ=1 only): 0xa1f020's entry and return (6 bytes moved: push rbp; mov
	// rbp,rsp; push r15).
	bool xpr_read = false;
	if (Raiz::SettleMode() != 0) {
		const char* name = "lectura de contadores del index cull (eboot+0xa1f020)";
		if ((mask & HookSettle) == 0 || !Matches(name, XprReadEntry, XprReadHead, sizeof(XprReadHead))) {
			std::printf("[write-audit]   %s: NO instalado (mascara o codigo distinto)\n", name);
		} else {
			const uint64_t stub = Stub::EmitEntryAroundStub(emitter, host(&OnXprReadBefore), host(&OnXprReadAfter), XprReadHead, 6,
			                                                g_base + XprReadEntry);
			xpr_read = emitter.ok && Queue(name, XprReadEntry, 6, 0xe9, stub, HookSettle);
			any      = xpr_read || any;
		}
	}
	// (parpadeo3) KYTY_PARPADEO_XPR=1..3: 0xa1f560's entry (5 bytes moved: its cmp, run after the host call).
	bool chunk_bits = false;
	if (ChunkBitsWanted() != 0)
		chunk_bits = queue_entry("bits de chunks del XPR (eboot+0xa1f560)", HookChunkBits, ChunkBitsEntry, ChunkBitsHead,
		                         sizeof(ChunkBitsHead), 5, host(&OnChunkBits));
	any = chunk_bits || any;
	// (parpadeo5) KYTY_PARPADEO5=1/2: 0xa1f290's entry (6 bytes moved: push r14; push rbx; mov rax,rdi). (parpadeo6) Also
	// for KYTY_PARPADEO6=1 alone (the settle).
	bool chunk_draws = false;
	if (Parpadeo5::Level() != 0 || Parpadeo6::On())
		chunk_draws = queue_entry("draws por chunks del XPR (eboot+0xa1f290)", HookChunkBits, ChunkDrawsEntry, ChunkDrawsHead,
		                          sizeof(ChunkDrawsHead), 6, host(&OnChunkDraws));
	any = chunk_draws || any;
	// (policy3) KYTY_DS_FRAME_AUDIT=1 only: the allocation function of the frame allocators (told before and
	// after, with the address), the kickoff's entry and the point of SubmitJob after its waits.
	uint32_t frame_hooks = 0;
	// (policy5) The reservation hook goes in with the split too (its guard against frame memory left no-watch).
	if (!FrameAuditWanted() && FrameWanted() == 0 && SplitWanted() != 0) {
		if ((mask & HookFrameAudit) != 0 && Matches("frame: reserva (eboot+0x811f60)", FrameAllocB, FrameAllocBHead, sizeof(FrameAllocBHead))) {
			const uint64_t stub = Stub::EmitEntryAroundStub(emitter, host(&OnFrameAllocBefore), host(&OnFrameAllocAfter), FrameAllocBHead, 6,
			                                                g_base + FrameAllocB);
			if (emitter.ok && Queue("frame: reserva (eboot+0x811f60)", FrameAllocB, 6, 0xe9, stub, HookFrameAudit)) any = true;
		} else {
			std::printf("[write-audit]   frame: reserva (eboot+0x811f60): NO instalado (mascara o codigo distinto)\n");
		}
	}
	if (FrameAuditWanted() || FrameWanted() != 0) { // (policy6: the frame policy needs the same three)
		const auto queue_alloc = [&](const char* name, uint64_t rva, const uint8_t* head, size_t head_size) {
			if ((mask & HookFrameAudit) == 0 || !Matches(name, rva, head, head_size)) {
				std::printf("[write-audit]   %s: NO instalado (mascara o codigo distinto)\n", name);
				return false;
			}
			const uint64_t stub =
			    Stub::EmitEntryAroundStub(emitter, host(&OnFrameAllocBefore), host(&OnFrameAllocAfter), head, 6, g_base + rva);
			return emitter.ok && Queue(name, rva, 6, 0xe9, stub, HookFrameAudit);
		};
		frame_hooks += queue_alloc("frame: reserva (eboot+0x811f60)", FrameAllocB, FrameAllocBHead, sizeof(FrameAllocBHead)) ? 1 : 0;
		frame_hooks += queue_entry("frame: kickoff (eboot+0xa08ae0)", HookFrameAudit, KickoffEntry, KickoffHead, sizeof(KickoffHead), 6,
		                           host(&OnKickoff)) ? 1 : 0;
		frame_hooks += queue_call("frame: envio de lote (eboot+0xa188bc)", HookFrameAudit, LotSubmitSite, LotSubmitTarget, LotSubmitCall,
		                          sizeof(LotSubmitCall), 14, host(&OnLotSubmit), 0) ? 1 : 0;
		any = frame_hooks != 0 || any;
	}
	any = queue_call("capa leida (eboot+0x8fef15)", HookLayers, LayerReadSite, TexLoadTarget, LayerReadCall, sizeof(LayerReadCall), 19,
	                 host(&OnLayerReadBefore), host(&OnTexLoadAfter)) || any;
	// (The converted layer is placed with the slice size the loader keeps at [rbp - 0x310]: that load is checked too.)
	if ((mask & HookLayers) == 0 || Matches("capa convertida (eboot+0x8feed1)", LayerSliceLoad, LayerSlice, sizeof(LayerSlice)))
		any = queue_call("capa convertida (eboot+0x8feed1)", HookLayers, LayerConvSite, LayerConvTarget, LayerConvCall, sizeof(LayerConvCall),
		                 29, host(&OnLayerConvBefore), host(&OnLayerConvAfter), 2) || any;
	else std::printf("[write-audit]   capa convertida (eboot+0x8feed1): NO instalado (el codigo no coincide)\n");

	if (any) {
		const uint64_t requested = g_base + CaveOffset;
		const uint64_t allocated = Libs::LibKernel::Memory::AllocateRuntimeMemory(
		    requested, CaveSize, Common::VirtualMemory::Mode::ExecuteReadWrite, "demons_souls_write_audit", true);
		if (allocated != requested) {
			if (allocated != 0) Libs::LibKernel::Memory::FreeGuestMemory(allocated, CaveSize);
			std::printf("[write-audit]   sin memoria ejecutable en eboot+0x%llx: NINGUN gancho con parche instalado\n",
			            static_cast<unsigned long long>(CaveOffset));
			g_patch_count = 0;
		} else {
			g_cave = allocated;
			std::memset(reinterpret_cast<void*>(g_cave), 0xcc, CaveSize);
			std::memcpy(reinterpret_cast<void*>(g_cave), image.data(), emitter.size);
			if (!Common::VirtualMemory::FlushInstructionCache(g_cave, CaveSize)) {
				std::printf("[write-audit]   la cueva no se pudo preparar: NINGUN gancho con parche instalado\n");
				g_patch_count = 0;
			}
		}
	}
	// The registry is there before the first patch can call a hook.
	g_registry.store(registry, std::memory_order_release);
	uint32_t missing = HookCreate | HookDestroy | HookDecommit | HookDeferredFree | HookHeapFree; // (policy1: all must be in)
	uint32_t seen    = 0, failed = 0;
	for (size_t i = 0; i < g_patch_count; ++i) {
		Patch&     patch   = g_patches[i];
		const auto address = g_base + patch.rva;
		std::memcpy(reinterpret_cast<void*>(address), patch.installed, patch.size);
		patch.written = true;
		if (!Common::VirtualMemory::FlushInstructionCache(address, patch.size)) {
			std::memcpy(reinterpret_cast<void*>(address), patch.original, patch.size);
			Common::VirtualMemory::FlushInstructionCache(address, patch.size);
			patch.written = false;
		}
		std::printf("[write-audit]   %s: %s\n", patch.name, patch.written ? "instalado" : "NO instalado (no se pudo escribir)");
		(patch.written ? seen : failed) |= patch.bit;
	}
	missing &= ~seen | failed;
	// (policy1) HookDestroy is two patches: both must be in.
	{
		size_t destroys = 0;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].bit == HookDestroy && g_patches[i].written) destroys++;
		if (destroys != 2) missing |= HookDestroy;
	}
	// (policy1) The 0x20-byte table's Create: no patch, its return address is trusted when the code around the call is
	// 01.007.000's.
	if (Matches("tabla reescrita (eboot+0x9f67d7)", RewrittenReturn - sizeof(RewrittenCall), RewrittenCall, sizeof(RewrittenCall)))
		g_rewritten_return.store(g_base + RewrittenReturn, std::memory_order_relaxed);
	else std::printf("[write-audit]   tabla reescrita (eboot+0x9f67d7): NO reconocida (su buffer se registra como uno con datos)\n");
	// The memcpy sites: no patch; the return addresses are trusted only when the code around the calls is 01.007.000's.
	if ((mask & HookCopySites) == 0) {
		std::printf("[write-audit]   sitios de memcpy: fuera por la mascara (ni copias de create/streaming ni red memcpy)\n");
	} else {
		const bool create_copy = Matches("copia de create (eboot+0x90954b)", CreateCopyReturn - sizeof(CreateCopy), CreateCopy, sizeof(CreateCopy));
		const bool stream_copy = Matches("copia de streaming (eboot+0x901459)", StreamCopyReturn - 14, StreamCopy, sizeof(StreamCopy));
		if (create_copy) g_create_copy_return.store(g_base + CreateCopyReturn, std::memory_order_relaxed);
		if (stream_copy) g_stream_copy_return.store(g_base + StreamCopyReturn, std::memory_order_relaxed);
		g_copies.store(true, std::memory_order_relaxed);
		if (!create_copy || !stream_copy) {
			std::printf("[write-audit]   sitios de memcpy: copia de create %s, copia de streaming %s (el codigo de la que falta no "
			            "coincide; sus copias cuentan solo como red memcpy)\n",
			            create_copy ? "activa" : "NO", stream_copy ? "activa" : "NO");
		} else {
			std::printf("[write-audit]   sitios de memcpy (retornos eboot+0x909550 y eboot+0x90145e): activos si el memmove de libc.prx "
			            "queda sustituido (linea \"Demon's Souls copy: installed\"); sin parche propio\n");
		}
	}
	std::printf("[write-audit]   linear copy (eboot+0x912e60): %s\n",
	            g_linear_copy_hook ? "instalado (parche compartido con KYTY_DS_LINEAR_COPY_HOOK)"
	                               : (mask & HookLinearCopy) == 0 ? "fuera por la mascara" : "NO instalado (ver la linea del linear copy hook)");
	std::printf("[write-audit]   decommit de streaming (eboot+0x9014f4): cubierto por el gancho de entrada de eboot+0x7f9f70\n");
	if (Wanted())
		std::printf("[write-audit] activa: lineas [write-audit] y [read-audit] cada 5 s, tablas cada 60 s y al salir.\n");
	// (policy1) The policies. The split needs every hook a range's life depends on: without one of them a range could
	// outlive its memory and be trusted, so it stays off.
	const int split = SplitWanted(), announced = AnnouncedWanted();
	if (announced != 0 || split != 0) {
		const int level = (std::max)(announced, split != 0 ? 1 : 0);
		g_trust.store(Reg::TrustedKinds(level), std::memory_order_relaxed);
		std::printf("[write-policy] KYTY_DS_WRITE_ANNOUNCED=%d: rangos fiables = %s\n", level,
		            level >= 2 ? "buffers creados con datos, texturas, memoria envuelta (0x800) y origenes de linear copy"
		                       : "buffers creados con datos y texturas");
	}
	if (split != 0 && g_rewritten_return.load(std::memory_order_relaxed) == 0) missing |= HookCreate;
	// (policy5) The split needs the frame reservations (the hook written, and its table).
	if (split != 0) {
		bool reservation = false;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].rva == FrameAllocB && g_patches[i].written) reservation = true;
		if (reservation && g_fa_pages == nullptr) {
			try {
				g_fa_pages = new FaPage[FaPageSlots];
			} catch (...) {
			}
		}
		if (!reservation || g_fa_pages == nullptr) missing |= HookFrameAudit;
		// The diagnostic filter.
		const char* only   = std::getenv("KYTY_DS_SPLIT_ONLY");
		const char* except = std::getenv("KYTY_DS_SPLIT_EXCEPT");
		const char* list   = only != nullptr && only[0] != 0 ? only : except != nullptr && except[0] != 0 ? except : nullptr;
		if (list != nullptr) {
			std::snprintf(g_filter_list, sizeof(g_filter_list), "%s", list);
			g_filter_mode = list == only ? 1 : 2;
			std::printf("[write-policy] DIAGNOSTICO: %s=%s: %s las paginas de esos asignadores se sueltan\n",
			            g_filter_mode == 1 ? "KYTY_DS_SPLIT_ONLY" : "KYTY_DS_SPLIT_EXCEPT", g_filter_list,
			            g_filter_mode == 1 ? "SOLO" : "todas MENOS");
		}
	}
	if (split != 0 && missing != 0) {
		std::printf("[write-policy] KYTY_DS_NEIGHBOR_SPLIT=%d: NO activa: faltan ganchos (mascara 0x%03x: create 01, destroy 02, decommit 04, "
		            "free diferido 10, free heap 20, reserva de frame 2000; o la tabla reescrita). Todo queda con la vigilancia normal.\n",
		            split, missing);
	} else if (split != 0) {
		std::printf("[write-policy] KYTY_DS_NEIGHBOR_SPLIT=%d: ACTIVA: una pagina que recibe una escritura fuera de todo rango registrado%s "
		            "deja de protegerse si sus rangos registrados son todos fiables. Linea [write-policy] cada 5 s.\n",
		            split, split >= 2 ? " (tambien las paginas sin ningun rango)" : " (solo paginas con algun rango fiable)");
	}
	bool frame_ready = false; // (policy6: the three frame hooks in and the tables made)
	if (FrameAuditWanted() || FrameWanted() != 0) {
		size_t written = 0;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].bit == HookFrameAudit && g_patches[i].written) written++;
		FaPage* pages = g_fa_pages;
		if (written == 3 && pages == nullptr) {
			try {
				pages = new FaPage[FaPageSlots];
			} catch (...) {
			}
		}
		if (written == 3 && pages != nullptr) {
			try {
				g_fa_submits_kept = new FaSubmit[FaSubmitKeep];
				g_fr_marks        = new std::vector<Request>();
				g_fr_marks->reserve(4096);
				g_fr_open = new FrOpen[FrSlots];
				g_fr_ring = new FrRun[FrRunMax];
				frame_ready = true;
			} catch (...) {
			}
		}
		if (frame_ready) g_fa_pages = pages;
	}
	if (FrameAuditWanted()) {
		size_t written = frame_ready ? 3 : 0;
		if (frame_ready) {
			g_frame_audit.store(true, std::memory_order_release);
			std::printf("[frame-audit] KYTY_DS_FRAME_AUDIT=1: ACTIVA (solo mide): reservas de los asignadores de frame, kickoffs y envios de "
			            "lote; cada fault de frame se juzga contra la regla en el envio de lote siguiente. Linea [frame-audit] cada 5 s.\n");
		} else {
			// (With only some of the three in, the ones in do nothing but pass through: g_frame_audit stays false.)
			std::printf("[frame-audit] KYTY_DS_FRAME_AUDIT=1: NO activa (faltan ganchos de frame o memoria para sus tablas). No mide nada.\n");
			(void)written;
		}
	}
	if (ReadbackWanted()) {
		size_t entries = 0;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].bit == HookReadback && g_patches[i].written) entries++;
		if (entries != 0) {
			g_readback.store(true, std::memory_order_relaxed);
			// (MALLA-NATIVA) Los dos accesores enganchados: la fase 2 (KYTY_MALLA_NATIVA_RAM) puede soltar RAM de mallas.
			MallaNativa::Get().read_hooked.store(read_pointer && read_copy, std::memory_order_release);
			std::printf("[readback-hook] KYTY_DS_READBACK_HOOK=1: ACTIVA en %zu de 5 accesores (0x909940 %s, 0x909970 %s, 0xa2e6f0 %s, 0xadabb0 %s, 0x8e9760 %s; KYTY_DS_READ_HOOK2=0 quita los dos ultimos): al "
			            "entrar se descarga de una vez el buffer que la GPU escribio. Linea [readback-hook] cada 5 s.\n",
			            entries, read_pointer ? "si" : "no", read_copy ? "si" : "no", read_vis ? "si" : "no", read_mask ? "si" : "no",
			            read_c12 ? "si" : "no");
		} else {
			std::printf("[readback-hook] KYTY_DS_READBACK_HOOK=1: NO activa (ningun accesor instalado). Las lecturas siguen por fault.\n");
		}
	}
	if (Raiz::SettleMode() != 0) {
		bool written = false;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].bit == HookSettle && g_patches[i].written) written = true;
		std::printf("[raiz12] KYTY_RAIZ_SETTLE=%d (%s): 0xa1f020 (contadores del index cull) %s; accesores 0x909940/0x909970 %s. "
		            "Linea [raiz] SETTLE.\n",
		            Raiz::SettleMode(), Raiz::SettleMode() == 1 ? "arreglo: el juego espera a que entren los bytes de la GPU" : "solo cuenta",
		            written ? "enganchado" : "NO enganchado", g_readback.load(std::memory_order_relaxed) ? "con el mismo asentado" : "sin gancho");
	}
	if (ChunkBitsWanted() != 0) {
		bool written = false;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].rva == ChunkBitsEntry && g_patches[i].written) written = true;
		static const char* const what[4] = {"", "solo mira", "mira y asienta los bits (KYTY_RAIZ_SETTLE decide si arregla o cuenta)",
		                                    "mira, espera la etiqueta del trabajo GPU (max 4 ms) y asienta"};
		std::printf("[parpadeo3] KYTY_PARPADEO_XPR=%d (%s): 0xa1f560 (lectura en la CPU de los bits de chunks del XPR) %s. Linea "
		            "[parpadeo3] cada 5 s.\n",
		            ChunkBitsWanted(), what[ChunkBitsWanted()], written ? "enganchado" : "NO enganchado");
	}
	if (Parpadeo5::Level() != 0) {
		bool written = false;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].rva == ChunkDrawsEntry && g_patches[i].written) written = true;
		std::printf("[parpadeo5] KYTY_PARPADEO5=%d (%s): 0xa1f290 (el lector de bits de chunks que de verdad corre: 0xad8bc0/0xad92f0) %s. "
		            "Linea [parpadeo5] cada 5 s; con KYTY_PARPADEO>=1 cada DRAWS CAEN (CPU) lleva lo que grabo el juego.\n",
		            Parpadeo5::Level(),
		            Parpadeo5::Level() >= 2 ? "mide y asienta los bits antes de que el juego los lea (arreglo candidato; KYTY_RAIZ_SETTLE=1)"
		                                    : "solo mide",
		            written ? "enganchado" : "NO enganchado");
	}
	if (Parpadeo6::On()) {
		bool written = false;
		for (size_t i = 0; i < g_patch_count; ++i)
			if (g_patches[i].rva == ChunkDrawsEntry && g_patches[i].written) written = true;
		std::printf("[parpadeo6] KYTY_PARPADEO6=1 (arreglo: los bits de chunks del XPR se asientan en 0xa1f290 antes de que el juego los "
		            "lea: lo que la GPU aun tiene leido de vuelta y ningun put-in de bytes guardados en curso; %s): 0xa1f290 %s. Linea "
		            "[parpadeo6] cada 5 s.\n",
		            Raiz::On() ? "KYTY_RAIZ=1" : "SIN EFECTO: sin KYTY_RAIZ no hay bytes guardados", written ? "enganchado" : "NO enganchado");
	}
	if (Parpadeo7::On()) // (parpadeo7)
		std::printf("[parpadeo7] KYTY_PARPADEO7=1 (los bits del XPR se leen de vuelta al traducir la etiqueta que los entrega; el "
		            "asentado de parpadeo6 sigue como red): aprende en 0xa1f290 (ver la linea [parpadeo6] de arriba: enganchado). Linea "
		            "[parpadeo7] cada 5 s.\n");
	if (VerifyWanted()) {
		g_verify.store(true, std::memory_order_relaxed);
		std::printf("[write-policy] KYTY_DS_WRITE_VERIFY=1: verificacion por hash de rangos fiables, %zu rangos y %llu KiB cada 0,5 s. Linea "
		            "[write-verify] cada 5 s.\n",
		            VerifyRanges, static_cast<unsigned long long>(VerifyBytes >> 10u));
	}
	std::fflush(stdout);
	g_lines.store(Wanted(), std::memory_order_relaxed);
	g_start = std::chrono::steady_clock::now();
	g_on.store(true, std::memory_order_release);
	if (split != 0 && missing == 0) g_split.store(split, std::memory_order_release);
	// (policy6) The frame policy.
	if (const int frame = FrameWanted(); frame != 0) {
		if (!frame_ready) {
			std::printf("[write-frame] KYTY_DS_WRITE_FRAME=%d: NO activa (faltan los ganchos de frame 0x811f60 / 0xa08ae0 / 0xa188bc, mascara "
			            "2000, o memoria para sus tablas). Todo queda con la vigilancia normal.\n",
			            frame);
		} else {
			g_fr_margin = FrameLotsWanted();
			if (frame >= 2) {
				try {
					g_clean = new CleanPage[CleanSlots];
				} catch (...) {
				}
			}
			g_check_stale.store(frame >= 2 && g_clean != nullptr, std::memory_order_relaxed);
			std::printf("[write-frame] KYTY_DS_WRITE_FRAME=%d: ACTIVA: la memoria que entregan los asignadores de frame deja de protegerse "
			            "tras su primera escritura; se marca escrita en cada envio de lote (lo reservado desde el kickoff n-%llu), en cada "
			            "envio de command buffer (su rango, su reserva, su bloque de contexto) y al acabar cada copia del emulador sobre "
			            "ella.%s Linea [write-frame] cada 5 s.\n",
			            frame, static_cast<unsigned long long>(g_fr_margin),
			            frame >= 2 ? " COMPROBACION de datos viejos activa (1 pagina de cada 64)." : "");
			g_frame.store(frame, std::memory_order_release);
		}
		std::fflush(stdout);
	}
	// (anuncio1) KYTY_RAIZ_ANNOUNCE=1.
	if (const int announce = AnnounceWantedLevel(); announce != 0) {
		if (announce < 0) {
			std::printf("[anuncio] KYTY_RAIZ_ANNOUNCE=1: ignorada (necesita KYTY_RAIZ=1). Nada cambia.\n");
		} else if (!SplitOn()) {
			std::printf("[anuncio] KYTY_RAIZ_ANNOUNCE=1: NO activa (ni la politica de frame ni la de vecinos estan activas: faltan ganchos). "
			            "Nada cambia.\n");
		} else {
			if (FrameOn()) {
				try {
					g_block_pages = new FaPage[BlockSlots];
				} catch (...) {
					g_block_pages = nullptr;
				}
			}
			g_announce.store(true, std::memory_order_release);
			g_announce_verify.store(announce >= 2, std::memory_order_release);
			std::printf("[anuncio] KYTY_RAIZ_ANNOUNCE=1: ACTIVA: politica de frame %s (KYTY_DS_WRITE_FRAME=%d); una pagina de frame vetada "
			            "porque la GPU la escribio vuelve a quedar sin vigilancia cuando un asignador de frame la entrega otra vez%s.%s Linea "
			            "[anuncio] cada 5 s.\n",
			            FrameOn() ? "activa" : "NO activa", g_frame.load(std::memory_order_relaxed),
			            g_block_pages != nullptr ? "" : " (NO: sin la politica de frame o sin memoria para la tabla)",
			            announce >= 2 ? " KYTY_RAIZ_VERIFY=1: AUDITORIA por write watch de las paginas sin vigilancia que la GPU va a usar "
			                            "(lo escrito sin anuncio se cuenta, se lista y se vuelve a subir)."
			                          : "");
		}
		std::fflush(stdout);
	}
	if (!g_thread_started) {
		g_thread_started = true;
		std::thread(Thread).detach();
	}
#else
	(void)program;
#endif
}

void Clear() {
	if (!On()) return;
	{
		const Totals totals  = ReadTotals();
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
		if (g_lines.load(std::memory_order_relaxed)) {
			PrintTables(totals, seconds, g_frame_known.load(std::memory_order_relaxed));
			PrintReadTable(totals, seconds);
		}
		if (SplitOn()) {
			PrintPolicyLine(totals, seconds, "ACUMULADO ");
			PrintReverted();
		}
		if (g_verify.load(std::memory_order_relaxed)) PrintVerifyLine(totals, seconds, "ACUMULADO ", true);
		std::fflush(stdout);
	}
	if (g_readback.load(std::memory_order_relaxed))
		PrintReadbackLine(ReadTotals(), std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(), "ACUMULADO ");
	g_readback.store(false, std::memory_order_relaxed);
	if (g_frame_audit.load(std::memory_order_relaxed))
		PrintFrameAuditLine(ReadTotals(), std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(), "ACUMULADO ", true);
	g_frame_audit.store(false, std::memory_order_relaxed);
	if (FrameOn())
		PrintFrameLine(ReadTotals(), std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(), "ACUMULADO ", true);
	if (SplitOn()) { // (anuncio1)
		uint64_t announce_now[AnnounceCounterCount];
		AnnounceSnapshot(announce_now);
		PrintAnnounceLine(announce_now, nullptr, std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count(), "ACUMULADO ");
		PrintAnnounceTables();
		std::fflush(stdout);
	}
	g_announce_verify.store(false, std::memory_order_release);
	g_announce.store(false, std::memory_order_release);
	g_frame.store(0, std::memory_order_release);
	// (policy1: the pages already without their watcher stay so; no new one is made.)
	g_split.store(0, std::memory_order_release);
	g_on.store(false, std::memory_order_release);
	// The game's bytes back (the cave stays: a thread may still be inside a stub).
	for (size_t i = 0; i < g_patch_count; ++i) {
		Patch& patch = g_patches[i];
		if (!patch.written) continue;
		auto* const address = reinterpret_cast<void*>(g_base + patch.rva);
		if (std::memcmp(address, patch.installed, patch.size) == 0) {
			std::memcpy(address, patch.original, patch.size);
			Common::VirtualMemory::FlushInstructionCache(g_base + patch.rva, patch.size);
		}
		patch.written = false;
	}
	g_patch_count = 0;
}

// (compat) declarada en el .h (nativo3, linea [nativo-check]); esta copia no la traia.
uint64_t StaleUsesTotal() noexcept {
	std::lock_guard lock(g_stale_mutex);
	uint64_t total = 0;
	for (size_t i = 0; i < g_stale_count; ++i) total += g_stale[i].count;
	return total;
}

} // namespace Loader::DemonsSoulsWriteAudit
