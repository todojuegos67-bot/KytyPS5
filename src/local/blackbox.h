#pragma once
// The crash / freeze black box's counters (local/blackbox-run.h writes them out). Relaxed atomics the hot paths bump
// (an increment, a decrement and a clock read at most per event): file reads, pipeline creations, queue submissions,
// GPU fence waits, presents, the render thread's waits, the frames' kind, the pause and the window's state. Nothing
// here allocates, locks or changes what the code around it does.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Blackbox {

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The phase a key thread entered last (its InFlightScope, below): the kind of work (PhaseCode), with After set once it
// left it, and since when (steady-clock ns). The render (GPU) thread claims its slot at its start; the thread that
// presents (the window's, or the render thread itself when it presents) claims the other at its first present.
enum PhaseCode : uint32_t {
	PhaseNone     = 0,
	PhaseSubmit   = 1, // vkQueueSubmit
	PhasePresent  = 2, // vkQueuePresentKHR (its queue lock's wait included)
	PhaseFence    = 3, // a wait for the GPU (MasterSemaphore::Wait)
	PhasePipeline = 4, // vkCreate*Pipelines
	PhaseRead     = 5, // a file read into guest memory
	PhaseFault    = 6, // the guest memory's access-violation handler (hostException.cpp)
	PhaseProtect  = 7, // a protection change of guest memory (pageManager.cpp ProtectHost: VirtualProtect)
	PhaseCodes    = 8,
	PhaseAfter    = 0x100, // (or'ed: left it, nothing tracked entered since)
};
inline const char* PhaseName(uint32_t code) {
	static constexpr const char* Names[PhaseCodes] = {"-", "submit", "present", "fence", "pso", "read", "fault", "protect"};
	return Names[(code & 0xffu) < PhaseCodes ? (code & 0xffu) : 0];
}
struct ThreadPhase {
	std::atomic<uint32_t> code {0};
	std::atomic<int64_t>  since {0};
};
enum Role : uint32_t { RoleRender = 0, RolePresent = 1, Roles = 2 };
inline ThreadPhase                g_phases[Roles];
inline thread_local ThreadPhase*  t_phase = nullptr;
// The calling thread's role (the render thread, at its start: it keeps it).
inline void SetThreadRole(Role role) noexcept {
	t_phase = &g_phases[role];
}
// The role, unless the calling thread has one already (the presenting thread: the render thread when it presents).
inline void ClaimThreadRole(Role role) noexcept {
	if (t_phase == nullptr) t_phase = &g_phases[role];
}

// Work in flight of one kind: how many now, how many started in all, and when the oldest of a busy period began (the
// first started while none was in flight; steady-clock ns, 0: none in flight). With several in flight the age is the
// busy period's, not exactly the oldest one's (good enough to see something that never ends). `last`: the last one's
// start; `left`: the last one's end (steady-clock ns).
struct InFlight {
	constexpr explicit InFlight(uint32_t code) noexcept: phase(code) {}
	const uint32_t        phase;
	std::atomic<int32_t>  now {0};
	std::atomic<uint64_t> total {0};
	std::atomic<int64_t>  since {0};
	std::atomic<int64_t>  last {0};
	std::atomic<int64_t>  left {0};
	void Enter(int64_t at) {
		total.fetch_add(1, std::memory_order_relaxed);
		last.store(at, std::memory_order_relaxed);
		if (now.fetch_add(1, std::memory_order_relaxed) == 0) since.store(at, std::memory_order_relaxed);
	}
	void Leave(int64_t at) {
		left.store(at, std::memory_order_relaxed);
		if (now.fetch_sub(1, std::memory_order_relaxed) == 1) since.store(0, std::memory_order_relaxed);
	}
};
// One unit of work in flight; on a thread with a role, also its phase (the one it was in restored at the end when that
// was still in progress - a fault inside a submit - else this one marked left).
class InFlightScope {
public:
	explicit InFlightScope(InFlight& counter) noexcept: m_counter(counter), m_phase(t_phase) {
		const int64_t at = NowNs();
		m_counter.Enter(at);
		if (m_phase != nullptr) {
			m_previous       = m_phase->code.load(std::memory_order_relaxed);
			m_previous_since = m_phase->since.load(std::memory_order_relaxed);
			m_phase->since.store(at, std::memory_order_relaxed);
			m_phase->code.store(m_counter.phase, std::memory_order_relaxed);
		}
	}
	~InFlightScope() {
		const int64_t at = NowNs();
		m_counter.Leave(at);
		if (m_phase != nullptr) {
			const bool resume = m_previous != PhaseNone && (m_previous & PhaseAfter) == 0;
			m_phase->since.store(resume ? m_previous_since : at, std::memory_order_relaxed);
			m_phase->code.store(resume ? m_previous : (m_counter.phase | PhaseAfter), std::memory_order_relaxed);
		}
	}
	InFlightScope(const InFlightScope&)            = delete;
	InFlightScope& operator=(const InFlightScope&) = delete;

private:
	InFlight&    m_counter;
	ThreadPhase* m_phase;
	uint32_t     m_previous       = 0;
	int64_t      m_previous_since = 0;
};

inline InFlight g_reads {PhaseRead};          // the game's file reads into guest memory (libAmpr ReadHostFileToGuest)
inline InFlight g_pipelines {PhasePipeline};  // vkCreate*Pipelines of the game's shaders (shaders.cpp CreatePipelineInternal)
inline InFlight g_submits {PhaseSubmit};      // vkQueueSubmit (commandScheduler.cpp, direct and deferred)
inline InFlight g_fence_waits {PhaseFence};   // MasterSemaphore::Wait's waits for the GPU (timeline not reached yet)
inline InFlight g_presents {PhasePresent};    // vkQueuePresentKHR (swapchain.cpp)
inline InFlight g_faults {PhaseFault};        // guest memory access violations being handled (hostException.cpp)
inline InFlight g_protects {PhaseProtect};    // guest memory protection changes (pageManager.cpp)
inline std::atomic<uint64_t> g_read_bytes {0};
inline std::atomic<int64_t>  g_last_present_ns {0}; // the last present's return (steady-clock ns)

// The last file read's name (its last 47 bytes), in words so that a torn read is only a garbled name.
constexpr size_t                  FileWords = 6;
inline std::atomic<uint64_t>      g_last_file[FileWords] {};
inline void NoteRead(const char* path, size_t length, uint64_t bytes) {
	g_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
	char         text[FileWords * 8] {};
	const size_t keep = std::min(length, sizeof(text) - 1);
	std::memcpy(text, path + (length - keep), keep);
	for (size_t i = 0; i < FileWords; ++i) {
		uint64_t word = 0;
		std::memcpy(&word, text + i * 8, 8);
		g_last_file[i].store(word, std::memory_order_relaxed);
	}
}

// The render thread's wait in progress (LiveCensus::WaitScope): its kind + 1 (0: none) and start (steady-clock ns).
inline std::atomic<uint32_t> g_render_wait {0};
inline std::atomic<int64_t>  g_render_wait_since {0};

// The frames presented: 1 the game's 3D view, 2 none (a loading screen, a menu, a movie), 0 not known yet; and since when
// (steady-clock ns of the change).
inline std::atomic<uint32_t> g_scene_frame {0};
inline std::atomic<int64_t>  g_scene_frame_since {0};
inline void NoteSceneFrame(bool scene) {
	const uint32_t kind = scene ? 1u : 2u;
	if (g_scene_frame.load(std::memory_order_relaxed) != kind) {
		g_scene_frame_since.store(NowNs(), std::memory_order_relaxed);
		g_scene_frame.store(kind, std::memory_order_relaxed);
	}
}

// The guest paused (the pause key, F9's snapshot, the port's menus) and the window minimized: a stop of the presents
// then is not a stall.
inline std::atomic<bool> g_paused {false};
inline std::atomic<bool> g_minimized {false};
// The game asked to end (the window's quit): the shutdown's pauses (caches saved) are not stalls either.
inline std::atomic<bool> g_quitting {false};

// The block the black box shares with its watchdog process (blackbox-run.h, LocalPlatform::WatchdogMain): a named
// mapping (Windows: Local\\KytyBlackbox-<pid>) the 1 Hz thread updates first thing each second, before any call that
// could block. The watchdog reads it and tells "the game hung" (the heartbeat stopped, the watchdog's lines go on) from
// "the PC froze" (the watchdog's lines stop too).
constexpr uint64_t SharedMagic = 0x31584242'5954594bull; // "KYTYBBX1"
constexpr size_t   SharedPath  = 1024;
// The 1 Hz thread's stage: what it was doing when it stopped.
enum Stage : uint32_t {
	StageStart     = 0,
	StageSleep     = 1, // between two seconds
	StageBeatWrite = 2, // the heartbeat line's write to the ring (write-through)
	StageCpu       = 3, // the render thread's CPU time (GetThreadTimes)
	StageScene     = 4, // the scene tags
	StageProcMem   = 5, // the process's commit and working set (K32GetProcessMemoryInfo: the working set's lock)
	StageSysMem    = 6, // the system's memory (GlobalMemoryStatusEx)
	StageLineWrite = 7, // the full line's write to the ring
	Stages         = 8,
};
inline const char* StageName(uint32_t stage) {
	static constexpr const char* Names[Stages] = {"start", "sleep", "beat-write", "thread-cpu", "scene", "process-memory",
	                                              "system-memory", "line-write"};
	return stage < Stages ? Names[stage] : "?";
}
struct Shared {
	uint64_t              magic = 0; // (SharedMagic once the paths are written)
	uint32_t              pid   = 0;
	uint32_t              size  = 0; // sizeof(Shared)
	char                  watchdog_path[SharedPath] {}; // UTF-8: logs/watchdog-<stamp>.txt
	char                  dump_path[SharedPath] {};     // UTF-8: logs/hang-<stamp>.dmp
	std::atomic<uint64_t> beat {0};                     // the 1 Hz thread's seconds, counted at their start
	std::atomic<int64_t>  beat_ns {0};
	std::atomic<uint32_t> stage {0};
	std::atomic<int64_t>  stage_ns {0};
	std::atomic<uint64_t> flips {0};
	std::atomic<int64_t>  flip_ns {0};
	std::atomic<uint64_t> presents {0};
	std::atomic<int64_t>  present_ns {0};
	std::atomic<uint32_t> render_phase {0};
	std::atomic<int64_t>  render_phase_ns {0};
	std::atomic<uint32_t> flags {0}; // 1 paused, 2 minimized, 4 quitting
	std::atomic<int32_t>  reads {0}, faults {0}, pipelines {0}, submits {0};
	std::atomic<uint64_t> private_bytes {0}, working_set {0}; // the last measured
};
constexpr uint32_t SharedPaused = 1, SharedMinimized = 2, SharedQuitting = 4;

} // namespace Blackbox
