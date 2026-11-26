#pragma once
// (juego3) KYTY_JUEGO3: where the game's frame thread (the one that calls sceAgcSuspendPoint) loses the time it is not on
// a CPU, and a fix for the most likely cause. Off by default (0 = exactly as before: one relaxed load per hook).
//
// Why (native/juego3/INFORME.md): in the "CPU-juego" scenes the game thread takes ~11 ms from one suspend point to the
// next but runs ~4.5 ms of it; its waits the f0 WaitScope sees (semaphores, conditions, mutexes, flags, queues, sleeps,
// joins, rwlocks) are ~0. The XPR culling bits' settle (KYTY_PARPADEO6, 0xa1f290) waits ~6 times a frame, ~1.3 ms each
// (summed over the game's threads), for a readback whose copy is recorded behind everything translated by then: the GPU
// has to run all of it before the bits come in. A fault-driven read of GPU-written bytes waits the same way.
//
// KYTY_JUEGO3 = 1: measure, and the fix in alternating 5 s windows (OFF, ON, OFF, ...: one run gives the breakdown and
//                  the fix's effect in the same scene); 2: measure + fix always; 3: measure only.
// Measure (only the game thread):
//   - per frame: the period (suspend to suspend), its build (suspend exit to the next entry), its CPU (QueryThreadCycleTime
//     of the same thread; a frame whose thread changed is dropped), the CCD it ended on;
//   - scopes (Juego3::Scope) around every emulator path that can block it: the settle in 0xa1f290 (TagSettle), the
//     readback accessors 0x909940/0x909970 and the raiz12 settle in 0xa1f020 (TagAccessor), BufferCache::ReadMemory for a
//     guest thread (TagReadback), GuestGpu::SendCommandSync (TagSendSync), the emulator's fault handler (TagFault): wall
//     time per tag (inclusive) and, through the sampler, the time OFF the CPU inside each;
//   - a sampler thread ("Kyty.Juego3", 1 ms): the game thread's cycle count -> on/off CPU in each interval, given to its
//     innermost tag at that moment; when it was mostly off the CPU, at most every 5 ms, its stack (LocalPlatform::
//     SampleThread: rip and callers) -> the first eboot return address, the first emulator (exe) frames, and whether rip is
//     in ntdll (blocked in a system call) or not (ready, preempted). Lines "[juego3]" / "[juego3s]" every 5 s and
//     "[juego3] RESUMEN" every 60 s (ON - OFF from OFF/ON/OFF triplets of the same scene; loading windows dropped).
// Fix (FixNow(), the ON windows): the readback that a game read would ask is begun at the label after the work that wrote
//   the bytes, not when the game reads them:
//   - learn (GPU thread, a guest readback in BufferCache::ReadMemory): the compute shader that last wrote the range read back
//     (a ring of the last GPU writes with the dispatch's shader, BufferCache::NoteGpuWrite); at most 32 shaders;
//   - note (the translating thread, NoteGpuWrite): the ranges those shaders write;
//   - at the next label (CommandProcessor::WriteAtEndOfPipe): one command queued for the GPU thread that begins their
//     readbacks right behind that work (GpuResourceManager::PrefetchGuestReads / BufferCache::PrefetchGuestRead, the
//     parpadeo7 machinery: asynchronous, a helper thread copies the bytes in when the GPU is done, the GPU thread never
//     waits). The game's read then finds the bytes in, or waits for that copy begun earlier.
//   Correctness: the same readback a read fault or the settle makes, earlier; the settle (KYTY_PARPADEO6) stays as the net.

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace Juego3 {

enum Tag : uint32_t { TagNone = 0, TagSettle, TagAccessor, TagReadback, TagSendSync, TagFault, TagCount };

int ModeSlow() noexcept;
inline std::atomic<int> g_mode {-1};
// 0 off, 1 alternate ON/OFF, 2 fix always, 3 measure only.
[[nodiscard]] inline int Mode() noexcept {
	const int m = g_mode.load(std::memory_order_relaxed);
	return m >= 0 ? m : ModeSlow();
}
[[nodiscard]] inline bool On() noexcept {
	return Mode() != 0;
}
// The fix in this window (mode 1: ON windows; mode 2: always).
inline std::atomic<bool> g_fix_now {false};
[[nodiscard]] inline bool FixNow() noexcept {
	return g_fix_now.load(std::memory_order_relaxed);
}

// ---- the game thread ----
inline std::atomic<uint32_t> g_game_tid {0};
uint32_t                     CallerTid() noexcept;
[[nodiscard]] inline bool    IsGameThread() noexcept {
	const uint32_t g = g_game_tid.load(std::memory_order_relaxed);
	return g != 0 && CallerTid() == g;
}
// The emulator's GPU thread (GuestGpu::ProcessCommands): its CPU per window, beside the game thread's.
inline std::atomic<uint32_t> g_gpu_tid {0};
inline void                  NoteGpuThread() noexcept {
	if (Mode() != 0 && g_gpu_tid.load(std::memory_order_relaxed) == 0) g_gpu_tid.store(CallerTid(), std::memory_order_relaxed);
}
// sceAgcSuspendPoint: `entering` before GuestGpu::Done, then false after it.
void GameSuspend(bool entering) noexcept;

int64_t                       NowNs() noexcept;
void                          ScopeEnd(uint32_t tag, int64_t t0) noexcept;
inline std::atomic<uint32_t>  g_cur_tag {0}; // the game thread's innermost tag now (the sampler reads it)
// Wall time of a blocking path on the game thread (other threads: nothing). A tag already open counts once.
class Scope {
public:
	explicit Scope(uint32_t tag) noexcept {
		if (Mode() == 0 || tag == 0 || tag >= TagCount || !IsGameThread()) return;
		uint32_t& active = Active();
		if ((active & (1u << tag)) != 0) return;
		active |= 1u << tag;
		m_tag   = tag;
		m_outer = g_cur_tag.load(std::memory_order_relaxed);
		g_cur_tag.store(tag, std::memory_order_relaxed);
		m_t0 = NowNs();
	}
	~Scope() {
		if (m_tag == 0) return;
		Active() &= ~(1u << m_tag);
		g_cur_tag.store(m_outer, std::memory_order_relaxed);
		ScopeEnd(m_tag, m_t0);
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	static uint32_t& Active() noexcept {
		thread_local uint32_t active = 0;
		return active;
	}
	uint32_t m_tag = 0, m_outer = 0;
	int64_t  m_t0 = 0;
};

// The settle in 0xa1f290 (any thread): its microseconds and whether it read back.
void NoteSettle(uint64_t us, bool read_back) noexcept;

// ---- the fix ----
[[nodiscard]] inline bool FixCapable() noexcept {
	const int m = Mode();
	return m == 1 || m == 2;
}
// The compute shader whose dispatch this thread translates now (0 outside RenderExecutor::DispatchDirect).
inline thread_local uint64_t t_cs = 0;
class DispatchScope {
public:
	explicit DispatchScope(uint64_t cs) noexcept {
		if (!FixCapable()) return;
		m_on   = true;
		m_prev = t_cs;
		t_cs   = cs;
	}
	~DispatchScope() {
		if (m_on) t_cs = m_prev;
	}
	DispatchScope(const DispatchScope&)            = delete;
	DispatchScope& operator=(const DispatchScope&) = delete;

private:
	bool     m_on   = false;
	uint64_t m_prev = 0;
};
void NoteWriteSlow(uint64_t shader, uint64_t vaddr, uint64_t size) noexcept;
// BufferCache::NoteGpuWrite (translating threads): a range a dispatch writes (nothing outside one).
inline void NoteWrite(uint64_t vaddr, uint64_t size) noexcept {
	if (!FixCapable()) return;
	const uint64_t shader = t_cs;
	if (shader != 0 && size != 0) NoteWriteSlow(shader, vaddr, size);
}
// BufferCache::ReadMemory, a guest thread's readback, on the GPU thread: learn the range's writer.
void LearnReadback(uint64_t vaddr, uint64_t size) noexcept;
// The label (CommandProcessor::WriteAtEndOfPipe): the ranges noted since the last label, joined and capped. False: none.
inline std::atomic<uint32_t> g_pending_n {0};
bool                         TakePendingSlow(std::vector<std::pair<uint64_t, uint64_t>>& out) noexcept;
inline bool                  TakePending(std::vector<std::pair<uint64_t, uint64_t>>& out) noexcept {
	if (g_pending_n.load(std::memory_order_relaxed) == 0) return false;
	return TakePendingSlow(out);
}

} // namespace Juego3
