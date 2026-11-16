#pragma once
// (audit1) KYTY_DS_WRITE_AUDIT=1 (off by default; Demon's Souls 01.007.000 only): the AUDIT of the "write hooks" design
// (native/write-hooks/NOTAS.md). Measurement only: the game is hooked where it creates, fills and frees the memory the
// GPU reads, a registry of those ranges is kept, and every CPU write the tracking sees (a write fault, a write-window
// fault, a write-watch page) is classified against it. NO tracking policy changes: every page is protected, swept and
// uploaded exactly as without the variable. Without it nothing is installed (the two inline checks below are the cost).
//
//   frame       the write is in per-frame memory (the backend's two rings, the Default TaggedArenas, the small DCBs)
//   anunciada   in a registered range, made by a copy/read the hooks see (a hooked site, or any memcpy: "red memcpy")
//   vecino      outside every registered range, in a 4 KiB page that holds part of one
//   desconocido in no registered range and no such page
//   FUGA        in a registered range, by code no hook sees: what the "announced" policy would miss
//
// "[write-audit]" lines: one every 5 s, the tables every 60 s and when the game ends.
// KYTY_DS_WRITE_AUDIT_HOOKS=<hex mask> (default 3fff) leaves hooks out: 01 create, 02 destroy, 04 decommit, 08 texload,
// 10 deferred free, 20 heap free, 40 linear copy, 80 the memcpy sites (create's copy, the streaming copy, the net),
// (policy1) 100 the compute contexts' linear copy, 200 the texture array layers, 400 Bink, 800 the streaming
// function's entry (the owner block of the streamed chunks), (policy2) 1000 the readback accessors, (policy3) 2000 the frame audit's.
//
// (policy1) More classes, more hooks and the read faults:
//   por-uso     in memory a CBuffer only wraps (flag 0x800) or a linear copy reads, announced again at each wrap or
//               copy: what KYTY_DS_WRITE_ANNOUNCED=2 trusts
//   "[read-audit]" lines: the CPU read faults (memory the GPU wrote) by the game code that reads.
//
// (policy1) The policies, each OFF by default and independent of the audit's lines (any of them installs the hooks
// and keeps the registry; none of them changes anything without its variable):
//   KYTY_DS_NEIGHBOR_SPLIT=1   a page that takes a CPU write OUTSIDE every registered range, and whose registered
//                              ranges are all ones the hooks announce, stops being write-protected (a "no-watch"
//                              page: RegionManager). Its announced ranges are marked written when a hook says so.
//                              =2: also the pages with no registered range at all.
//   KYTY_DS_WRITE_ANNOUNCED=1  which ranges the split trusts: 1 (the default with the split) buffers created with
//                              their data and textures; 2 also wrapped memory (0x800) and linear copy sources.
//   KYTY_DS_WRITE_VERIFY=1     samples trusted ranges and hashes them twice: bytes that changed with no announcement
//                              in between are counted and listed ("[write-verify]").
//   KYTY_DS_READBACK_HOOK=1    (policy2) the entries of the CBuffer readback accessors 0x909940 / 0x909970: the buffer
//                              the game is about to read is read back from the GPU at once, instead of one read
//                              fault per page. "[readback-hook]" line.
//   KYTY_DS_FRAME_AUDIT=1      (policy3) measurement only: every reservation of the frame allocators, every
//                              kickoff and lot submit; each frame write fault is judged against the rule "at a
//                              lot's submit, what was reserved since the kickoff before that lot's is written".
//                              "[frame-audit]" line.
//   KYTY_DS_WRITE_FRAME=1      (policy6) frame memory a frame allocator handed out stops being write-protected after
//                              its first write fault; it is taken as written at each lot submit (what was reserved
//                              since kickoff n-2: R1), at each submit of a command buffer that holds it (R2) and at
//                              the end of each guest memcpy the emulator runs into it (R3). =2: also the stale-use
//                              check (sampled no-watch pages hashed when they become clean and when GPU work is about
//                              to use them). KYTY_DS_WRITE_FRAME_LOTS=N: R1's margin (default 2). "[write-frame]".
//   KYTY_RAIZ_ANNOUNCE=1       (anuncio1; only with KYTY_RAIZ=1) the frame policy on by default, a frame page refused
//                              for good made no-watch again once handed out again, and with KYTY_RAIZ_VERIFY=1 the
//                              write-watch audit of the no-watch pages GPU work uses. "[anuncio]" (see below).
// "[write-policy]" line every 5 s while a policy is on.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Loader {
struct Program;
namespace DemonsSoulsWriteAudit {

// True once Install found the game and started the audit (or a policy: the hooks and the registry are the same). One
// relaxed load: the only cost when off.
inline std::atomic<bool> g_on {false};
[[nodiscard]] inline bool On() noexcept {
	return g_on.load(std::memory_order_relaxed);
}
// (policy1) KYTY_DS_NEIGHBOR_SPLIT's level once Install started it (0: off). One relaxed load.
inline std::atomic<int> g_split {0};
// (policy6) KYTY_DS_WRITE_FRAME's level once Install started it (0 off, 1 on, 2 on with the stale-use check).
inline std::atomic<int> g_frame {0};
[[nodiscard]] inline bool FrameOn() noexcept {
	return g_frame.load(std::memory_order_relaxed) != 0;
}
// The no-watch pages' machinery is on: either policy (they share it: RegionManager's no-watch pages, the queue).
[[nodiscard]] inline bool SplitOn() noexcept {
	return g_split.load(std::memory_order_relaxed) != 0 || g_frame.load(std::memory_order_relaxed) != 0;
}

// KYTY_DS_WRITE_AUDIT=1, read at its first use (the loader's Execute: the port has set its variables by then).
[[nodiscard]] bool Wanted() noexcept;
// (policy1) ... or a policy that needs the hooks (KYTY_DS_NEIGHBOR_SPLIT, KYTY_DS_WRITE_ANNOUNCED, KYTY_DS_WRITE_VERIFY).
[[nodiscard]] bool HooksWanted() noexcept;
// ... and its linear copy hook not left out by the mask (DemonsSoulsCopy installs that entry patch for the audit too).
[[nodiscard]] bool WantsLinearCopy() noexcept;

// From RuntimeLinker::Execute, after every other patch of the eboot (DemonsSoulsGpuPages verifies functions this
// patches), before any guest code runs. eboot.bin of 01.007.000 only; each hook checks the bytes it replaces.
void Install(Program* program);
// From RuntimeLinker::Clear: the last tables, and the game's bytes put back.
void Clear();

// DemonsSoulsCopy: the linear copy function's entry patch (shared with KYTY_DS_LINEAR_COPY_HOOK) is in / is not.
void NoteLinearCopyHook(bool installed) noexcept;
// DemonsSoulsCopy's LinearCopyHook (a game thread), before the game's function runs.
void OnLinearCopy(uint64_t dst, int32_t dst_dwords, uint64_t src, int32_t src_dwords) noexcept;

// The loader's access-violation handler, before the GPU tracking handles the fault: the faulting instruction and stack
// pointer of this thread (for the FUGA table).
void NoteFaultSite(uint64_t rip, uint64_t rsp) noexcept;
// GpuResourceManager::HandleFault: a CPU write fault the tracking took. kind: 0 write window, 1 write, 2 the first
// write of fresh memory.
void NoteWriteFault(uint64_t vaddr, int kind) noexcept;
// GpuResourceManager::SweepCpuWrites (write watch, poll): a run of pages the game wrote.
void NoteSweptRun(uint64_t vaddr, uint64_t size) noexcept;

// The guest's memcpy/memmove (DemonsSoulsCopy::CoherentMove): the copy this thread is making, for as long as the
// scope lives. `return_address`: the eboot call site's.
void BeginCopy(uint64_t return_address, uint64_t destination, uint64_t size, uint64_t* saved, uint64_t source = 0) noexcept;
void EndCopy(const uint64_t* saved) noexcept;
struct CopyScope {
	CopyScope(uint64_t return_address, const void* destination, size_t size, const void* source = nullptr) noexcept
	    : active(On()) {
		if (active) BeginCopy(return_address, reinterpret_cast<uint64_t>(destination), size, saved, reinterpret_cast<uint64_t>(source));
	}
	~CopyScope() {
		if (active) EndCopy(saved);
	}
	CopyScope(const CopyScope&)            = delete;
	CopyScope& operator=(const CopyScope&) = delete;
	bool     active;
	uint64_t saved[5] {};
};

// ---- (policy1) -------------------------------------------------------------------------------------------------------
// GpuResourceManager::HandleFault, a CPU read fault the tracking took (memory the GPU wrote): counted by the game code
// that reads (NoteFaultSite's instruction, or the call site of the memcpy this thread is in).
void NoteReadFault(uint64_t vaddr) noexcept;

// KYTY_DS_NEIGHBOR_SPLIT. GpuResourceManager::HandleFault, a CPU write fault (any thread): the page of `vaddr` should
// stop being write-protected. Lock-free but for one shared lookup in the registry.
[[nodiscard]] bool WantsNoWatch(uint64_t vaddr) noexcept;
// ... and whether it did (false: the tracker refused; the fault takes the normal path).
void NoteNoWatch(uint64_t vaddr, bool done) noexcept;
// BufferCache::PolicyBeforeUse: every byte of the range lies in ranges whose writes the hooks announce (the frame
// arenas never have no-watch pages).
[[nodiscard]] bool ReadCovered(uint64_t vaddr, uint64_t size) noexcept;
// ... it did not: `pages` no-watch pages were watched again. why: 0 bytes no hook covers were asked for, 1 GPU work
// writes the range, (policy4) 2 the range is a buffer shaders read by address (BufferCache::PolicyBdaRead).
void NoteReverted(uint64_t vaddr, uint64_t pages, int why) noexcept;
// What the hooks announced since the last call, for the GPU thread (GpuResourceManager::SweepCpuWrites): ranges whose
// no-watch pages are written (revert false) or must be watched again (revert true). Returns how many were put in
// `out` (call again while it returns `max`). *everything: the queue overflowed since the last call: every no-watch
// page is to be taken as written.
struct Request {
	uint64_t address = 0, size = 0;
	bool     revert = false;
};
size_t TakeRequests(Request* out, size_t max, bool* everything) noexcept;
// ---- (policy6) KYTY_DS_WRITE_FRAME ----------------------------------------------------------------------------------
// The HLE's command buffer submit (libs/agc.cpp, before the GPU thread is given it; any thread): rule R2's marks, and
// the frame audit's record.
void NoteSubmitted(const void* commands, uint64_t size_in_bytes) noexcept;
// The GPU thread (GpuResourceManager::SweepCpuWrites, with the split's requests): the frame policy's marks, page-aligned
// ranges whose no-watch pages are to be taken as written. As TakeRequests; *everything: all no-watch pages are.
size_t TakeFrameMarks(Request* out, size_t max, bool* everything) noexcept;
void   NoteFrameDrained(uint64_t pages) noexcept;
// KYTY_DS_WRITE_FRAME=2: the stale-use check. One page in 64 is sampled (by its address). RegionManager: a sampled
// no-watch page became clean (an upload copies it now): its bytes are hashed. BufferCache, before GPU work uses a range:
// for each sampled page with a hash, `still_clean` (no-watch and not CPU-dirty): a page whose bytes changed since is
// one the GPU would use stale. Both cheap when the check is off (one relaxed load, inline).
inline std::atomic<bool> g_check_stale {false};
[[nodiscard]] inline bool CheckStaleOn() noexcept {
	return g_check_stale.load(std::memory_order_relaxed);
}
[[nodiscard]] inline bool SampledPage(uint64_t page_address) noexcept {
	return (((page_address >> 12u) * 0x9E3779B97F4A7C15ull) >> 58u) == 0;
}
void               NoteNoWatchClean(uint64_t page_address) noexcept;
[[nodiscard]] bool HasCleanHash(uint64_t page_address) noexcept;
// (policy7) `use_begin`, `use_end`: the range this use reads; `wide`: it is a BDA buffer or over 1 MiB (the bytes read
// cannot be told). Only bytes that changed inside the range are stale data.
void CheckNoWatchUse(uint64_t page_address, bool still_clean, uint64_t use_begin, uint64_t use_end, bool wide) noexcept;
// (policy7) The calling thread is the emulator's GPU thread (GpuResourceManager::SweepCpuWrites): its write faults are
// the emulated GPU's own writes (labels), for the frame audit.
void NoteGpuThread() noexcept;

// ---- (anuncio1) KYTY_RAIZ_ANNOUNCE=1 (with KYTY_RAIZ=1; off by default, read at Install) ----------------------------
// The frame policy on by default (KYTY_DS_WRITE_FRAME=1, =2 with KYTY_RAIZ_VERIFY=1, unless set), and a frame page the
// tracker refused for good (RegionManager::ClearNoWatch with `block`: GPU work wrote it, or a consumer asked for bytes
// no rule covers) is made no-watch again at a write fault once a frame allocator handed it out after the refusal.
// KYTY_RAIZ_VERIFY=1 too: the write-watch audit of the no-watch pages GPU work is about to use (a page written since it
// became clean with no announcement is counted, listed and made CPU-dirty). "[anuncio]" line every 5 s (also without
// the variable, while a policy is on: why the tracker refuses pages).
inline std::atomic<bool> g_announce {false};
inline std::atomic<bool> g_announce_verify {false};
[[nodiscard]] inline bool AnnounceOn() noexcept {
	return g_announce.load(std::memory_order_relaxed);
}
[[nodiscard]] inline bool AnnounceVerifyOn() noexcept {
	return g_announce_verify.load(std::memory_order_relaxed);
}
// RegionManager::ClearNoWatch with `block` (region lock held): the pages of [begin, end) were refused now.
void NoteBlockedRun(uint64_t begin, uint64_t end) noexcept;
// GpuResourceManager::HandleFault, right after WantsNoWatch (same thread) said yes and the tracker refused the page as
// refused before (why 3): whether it is a frame page handed out again since its refusal (lift it and try again).
[[nodiscard]] bool UnblockWanted(uint64_t vaddr) noexcept;
void               NoteUnblocked(uint64_t vaddr, bool done) noexcept;
// ... the tracker refused it, and why (RegionManager::SetNoWatch's `why`): the counters of the "[anuncio]" line.
void NoteNoWatchRefused(uint64_t vaddr, int why) noexcept;
// (KYTY_RAIZ_VERIFY) RegionManager: a run of no-watch pages became clean (before the upload copies them): their write
// watch is reset. BufferCache::PolicyCheckAnnounced: a written no-watch page that is clean (no announcement since), and
// the summary of one check.
void NoteNoWatchCleanRun(uint64_t vaddr, uint64_t size) noexcept;
void NoteUnannounced(uint64_t page, uint64_t use_begin, uint64_t use_size, bool wide, bool repaired) noexcept;
void NoteAnnounceCheck(uint64_t written, uint64_t dirty, uint64_t stale, bool truncated) noexcept;

// ---- (anuncio2) KYTY_RAIZ_ANNOUNCE=1: frame pages GPU work writes stay no-watch; ReadCovered memo; exact verify -----
// A page the frame policy owns (a frame allocator handed it out lately - FrameRecentPage -, not one of the rings): its
// CPU writes are announced by R1-R3. False without the frame policy.
[[nodiscard]] bool FramePolicyPage(uint64_t address) noexcept;
// BufferCache::PolicyBeforeUseSlow (KYTY_RAIZ_ANNOUNCE): GPU work is about to write [vaddr, vaddr + size). `kept`:
// no-watch frame-policy pages left no-watch (made CPU-dirty, so the upload before the write takes their bytes);
// `reverted`: other no-watch pages watched again and refused (as before).
void NoteGpuWriteUse(uint64_t kept, uint64_t reverted) noexcept;
// BufferCache: a use watched pages of [vaddr, vaddr + size) again and refused them (why: 0 a read no rule covers, 1 a
// GPU write, 2 a buffer read by address): for the "[anuncio] vetos no frame" table (only pages outside frame memory).
void NoteVetoOther(uint64_t vaddr, uint64_t size, uint64_t pages, int why) noexcept;
// (KYTY_DS_WRITE_FRAME=2 or KYTY_RAIZ_VERIFY with KYTY_RAIZ_ANNOUNCE) The upload's copy of guest [guest, guest + size):
// Begin just before it reads guest memory (the write watch of its whole pages starts over there), Copied with the
// bytes it read (`bytes`: the staging copy, what the GPU gets): the stale-use check compares a use with what was
// really uploaded, not with the bytes when the page became clean (the copy may run ms later on the upload worker:
// a write in between is uploaded, and was counted stale - anuncio2, the "datos viejos" of 035808).
[[nodiscard]] inline bool UploadCopyHookOn() noexcept {
	return AnnounceOn() && (CheckStaleOn() || AnnounceVerifyOn()); // (without KYTY_RAIZ_ANNOUNCE: as before)
}
void NoteUploadCopyBegin(uint64_t guest, uint64_t size) noexcept;
void NoteUploadCopied(uint64_t guest, const uint8_t* bytes, uint64_t size) noexcept;

// The count of clean pages uploads left without their write watcher so far (regionManager.h's g_nowatch_kept_pages),
// for the policy's line.
void NoteKeptPages(uint64_t total) noexcept;
void   NoteDrained(uint64_t dirty_pages, uint64_t reverted_pages, bool everything) noexcept;

} // namespace DemonsSoulsWriteAudit
} // namespace Loader
