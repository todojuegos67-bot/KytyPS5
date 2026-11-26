#pragma once
// (f0) KYTY_F0=1 (off by default; read once, at first use). Measurement only: nothing the game sees or the GPU draws
// changes. One build that measures what decides phase F2 of the graphics study (the game's frame memory shared with the
// GPU through Resizable BAR / placed mapping, and the game's linear copies on the GPU) and the real ceiling:
//
//  1. GPU time: a timestamp pool of its own; a TOP_OF_PIPE stamp when a scheduler's command buffer begins, a
//     BOTTOM_OF_PIPE stamp when it is submitted, and one more where the GPU thread switches between the graphics queue
//     (DCB) and a compute queue (ACB) of the game inside it. Resolved without waiting (results with availability, read
//     by the f0 thread, the queries reset on the host afterwards). Per frame: GPU busy (the union of the buffers'
//     intervals), and the time by side: DCB, ACB, the emulator's own (uploads' prologues, presentation, host commands).
//  2. The game's CPU reads of its frame memory: a few pages at a time of the memory map's labels (ring, DefaultTagg,
//     the heaps... local/../loader/demonsSoulsMemMap.h) get PAGE_GUARD added to their protection (the low byte kept: raiz
//     sees its own protection), with the protection gate taken alone, as the cold store probes (local/cold-store.h). The
//     first touch of each raises a guard violation (Windows clears the guard itself); counted as read or write, game code
//     or not; the access then runs again. A protection change or a kernel-mode access of guest memory clears ours first
//     (Ensure). Untouched ones lose their guard after ~60 ms.
//  3. Uploads (BufferCache::UploadCopies) and the table path's ring copies, bytes and count per frame by label; the game's
//     linear copies (DemonsSouls::TryLinearCopy*): count, bytes and the CPU time they take, by destination label.
//  4. The game's side: its frame (AgcSuspendPoint to AgcSuspendPoint, the game's submit thread), the part of it spent
//     waiting in the emulator (GuestGpu::Done), its thread's CPU, its submissions.
//  5. A "[f0]" line every 5 s and a row of logs\<run>-f0.csv; "[f0] RESUMEN" every 60 s and at exit with the verdict
//     "F2 viable / no viable / dudoso" and the fps ceiling when the GPU bounds the frame.
// Its own cost (hooks' cycles, its thread, the gate held, guard violations) is measured and printed.
//
// Off: each hook is one test of a static bool.

#include <cstdint>

namespace Libs::Graphics {
struct GraphicContext;
}

#if defined(__clang__) || defined(__GNUC__)
#define KYTY_F0_RET() __builtin_return_address(0)
#else
#define KYTY_F0_RET() nullptr
#endif

namespace F0Measure {

[[nodiscard]] bool Enabled() noexcept; // (reads KYTY_F0 once)
[[nodiscard]] inline bool On() noexcept {
	static const bool on = Enabled();
	return on;
}
// (esperas) KYTY_ESPERAS=1 (or N > 1: threshold N ms; else 20): light, alone, without KYTY_F0. Each frame of the game's
// submit thread (sceAgcSuspendPoint to the next) that takes the threshold or more prints one "[esperas]" line: its CPU,
// its waits by kind (the WaitScope hooks + file calls) and the eboot call sites of the longest, the time neither on a
// CPU nor in those waits, and its unix_ms (to match the "SLOW Frame" line). Also the game's wait for the emulator in
// the suspend point when it takes the threshold. Measurement only. Off: one more test of a static bool per hook.
[[nodiscard]] bool WaitsEnabled() noexcept; // (reads KYTY_ESPERAS once)
[[nodiscard]] inline bool WaitsOn() noexcept {
	static const bool on = WaitsEnabled();
	return on;
}

// (esperas2) The time a thread spends in the emulator's access-violation handler (write/read faults on guest pages:
// raiz's write faults, GPU readbacks, deferred releases put in by the fault). Counted only on a thread that calls
// sceAgcSuspendPoint (the "[esperas]" line's "fallos" part). Off (no KYTY_ESPERAS): one test of a static bool.
void EsperasFault(int64_t ns, uint64_t rip) noexcept;
[[nodiscard]] int64_t WaitNowNsFault() noexcept;
class FaultScope {
public:
	explicit FaultScope(uint64_t rip) noexcept {
		if (!WaitsOn()) return;
		m_rip = rip, m_t0 = WaitNowNsFault();
	}
	~FaultScope() {
		if (m_t0 != 0) EsperasFault(WaitNowNsFault() - m_t0, m_rip);
	}
	FaultScope(const FaultScope&)            = delete;
	FaultScope& operator=(const FaultScope&) = delete;

private:
	uint64_t m_rip = 0;
	int64_t  m_t0  = 0;
};

// ---- GPU timestamps (command scheduler; any scheduler, the thread that records it) ----
// A command buffer `raw` (VkCommandBuffer) began on this thread; `main`: the scheduler's open buffer (not an upload
// prologue). Its begin stamp is recorded into it.
void BufferBegin(Libs::Graphics::GraphicContext* graphics, void* raw, bool main);
// The buffers of one submission, before it is queued (CommandScheduler::SubmitBuffers): each one this thread began gets
// its end stamp; `ended`: already ended elsewhere (none recorded).
void BufferSubmit(void* raw, bool ended, int lane);

// ---- the GPU thread's slices (GuestGpu::ThreadRun around Process) ----
void SliceBegin(bool compute_queue);
void SliceEnd();

// ---- the render thread's flip (LiveControl::Flip) ----
// `render_idle_ns`: LiveControl::g_render_idle_ns; `gpu_wait_ns`: the GPU thread's waits on the GPU since start;
// `ab`: KYTY_PORT2=1's window (1 ON, 2 OFF; 0 none): the frame times split by it.
void Flip(int64_t flip_ns, int64_t render_idle_ns, int64_t gpu_wait_ns, int ab);

// ---- uploads and copies (render thread / upload worker) ----
void NoteUploadSlow(uint64_t vaddr, uint64_t size);
inline void NoteUpload(uint64_t vaddr, uint64_t size) {
	if (On()) NoteUploadSlow(vaddr, size);
}
void NoteRingCopySlow(uint64_t vaddr, uint64_t size);
inline void NoteRingCopy(uint64_t vaddr, uint64_t size) {
	if (On()) NoteRingCopySlow(vaddr, size);
}
// The game's linear copy made by the emulator, its CPU time from construction to destruction.
class LinearCopyScope {
public:
	LinearCopyScope(uint64_t dst, uint64_t src, uint64_t bytes) noexcept;
	~LinearCopyScope();
	LinearCopyScope(const LinearCopyScope&)            = delete;
	LinearCopyScope& operator=(const LinearCopyScope&) = delete;

private:
	uint64_t m_dst = 0, m_bytes = 0, m_start = 0;
};

// ---- the game's side (libs/agc.cpp) ----
void GameSubmit(bool compute_queue, uint32_t dwords);
// AgcSuspendPoint: `entering` before GuestGpu::Done, then false after it.
void GameSuspend(bool entering);

// (cpu1) The game thread's waits inside its frame ("construye": its build between two Agc suspends), by primitive
// (0 semaphore, 1 condition variable, 2 mutex, 3 event flag, 4 event queue, 5 sleep, 6 join, 7 rwlock; 8 file call,
// KYTY_ESPERAS only) and by the game's
// call site (the return address into the eboot): the "[f0w]" line every 5 s. Only the thread that suspends in Agc
// (the game's render-submitting thread) is timed; nested waits count once. Measurement only.
void WaitNote(int kind, int64_t ns, const void* ret) noexcept;
[[nodiscard]] bool WaitTimed() noexcept; // (this thread is the game's, f0 on)
[[nodiscard]] int64_t WaitNowNs() noexcept;
class WaitScope {
public:
	WaitScope(int kind, const void* ret) noexcept {
		if (!On() && !WaitsOn()) return;
		if (Depth()++ != 0 || !WaitTimed()) {
			m_kind = -2;
			return;
		}
		m_kind = kind, m_ret = ret, m_t0 = WaitNowNs();
	}
	~WaitScope() {
		if (m_kind == -1) return;
		--Depth();
		if (m_kind >= 0) WaitNote(m_kind, WaitNowNs() - m_t0, m_ret);
	}
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	static int& Depth() noexcept {
		thread_local int depth = 0;
		return depth;
	}
	int         m_kind = -1;
	int64_t     m_t0   = 0;
	const void* m_ret  = nullptr;
};

// ---- guest memory (kernel/memoryAddressSpace.inc, native memory only) ----
// The protection gate taken alone and let go (the cold store's hooks).
void SetGate(void (*lock)(), void (*unlock)());
// Before a protection change or a kernel-mode access of guest [vaddr, vaddr + size): our guards there cleared.
void EnsureSlow(uint64_t vaddr, uint64_t size);
inline void Ensure(uint64_t vaddr, uint64_t size) {
	if (On()) EnsureSlow(vaddr, size);
}

} // namespace F0Measure
