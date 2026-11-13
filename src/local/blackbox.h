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

// Work in flight of one kind: how many now, how many started in all, and when the oldest of a busy period began (the
// first started while none was in flight; steady-clock ns, 0: none in flight). With several in flight the age is the
// busy period's, not exactly the oldest one's (good enough to see something that never ends).
struct InFlight {
	std::atomic<int32_t>  now {0};
	std::atomic<uint64_t> total {0};
	std::atomic<int64_t>  since {0};
	void Enter() {
		total.fetch_add(1, std::memory_order_relaxed);
		if (now.fetch_add(1, std::memory_order_relaxed) == 0) since.store(NowNs(), std::memory_order_relaxed);
	}
	void Leave() {
		if (now.fetch_sub(1, std::memory_order_relaxed) == 1) since.store(0, std::memory_order_relaxed);
	}
};
class InFlightScope {
public:
	explicit InFlightScope(InFlight& counter): m_counter(counter) { m_counter.Enter(); }
	~InFlightScope() { m_counter.Leave(); }
	InFlightScope(const InFlightScope&)            = delete;
	InFlightScope& operator=(const InFlightScope&) = delete;

private:
	InFlight& m_counter;
};

inline InFlight g_reads;      // the game's file reads into guest memory (libAmpr ReadHostFileToGuest)
inline InFlight g_pipelines;  // vkCreate*Pipelines of the game's shaders (shaders.cpp CreatePipelineInternal)
inline InFlight g_submits;    // vkQueueSubmit (commandScheduler.cpp, direct and deferred)
inline InFlight g_fence_waits; // MasterSemaphore::Wait's waits for the GPU (timeline not reached yet)
inline InFlight g_presents;   // vkQueuePresentKHR (swapchain.cpp)
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

} // namespace Blackbox
