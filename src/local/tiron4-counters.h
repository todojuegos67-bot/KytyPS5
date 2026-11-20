#pragma once
// (tiron4) Counters with no dependencies for KYTY_TIRON3=2 ([tiron3] per slow frame): time in the guest's file calls
// (KernelOpen / KernelRead / KernelPread, fileSystem.cpp) and RAM17's reads back from its disk store (texture-offload.h).
// Always counted (two relaxed adds and two clock reads a call): the probe reads their difference over a slow frame.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace Tiron4 {
inline std::atomic<uint64_t> g_file_ns {0}, g_file_calls {0}, g_file_bytes {0}, g_open_ns {0}, g_opens {0};
inline std::atomic<uint64_t> g_ram17_us {0}, g_ram17_reads {0};
// (tiron5) TextureOffload::FileRestore (RAM17 reads back on a touch, waits included), wall time and calls.
inline std::atomic<uint64_t> g_restore_us {0}, g_restores {0};
// (tiron5) KYTY_TIRON5_TEX=1: the textures that came back stay in RAM longer (RAM17 hold and RAM16 retake 300 s).
inline bool TexOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON5_TEX");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
// (tiron5) KYTY_TIRON5_PIPE=1: new compute pipelines off the render thread (KYTY_ASYNC_COMPUTE_PIPELINES path) and the
// RT passes' prewarm thread at shader-worker (below normal) priority.
inline bool PipeOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON5_PIPE");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

struct FileScope {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	bool                                  open;
	explicit FileScope(bool is_open, uint64_t bytes = 0): open(is_open) {
		if (!open) g_file_bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
	~FileScope() {
		const auto ns = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		(open ? g_open_ns : g_file_ns).fetch_add(ns, std::memory_order_relaxed);
		(open ? g_opens : g_file_calls).fetch_add(1, std::memory_order_relaxed);
	}
};
} // namespace Tiron4
