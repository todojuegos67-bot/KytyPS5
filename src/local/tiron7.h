#pragma once
// (tiron7, KYTY_TIRON7=1) Native texture source: a large image upload (ObtainBufferForImage) reads the guest's bytes
// straight from guest memory on the GPU - the backing pages imported as a VkBuffer (VK_EXT_external_memory_host) -
// instead of the upload worker copying them into the staging ring first and every submission waiting for that copy
// (20261119-204418: 310 MB at ~1 GB/s, upload_wait 303 ms in one frame; 62, 34 and 13 ms in three others).
// The GPU's copy then reads system memory over the bus (~12+ GB/s) at execution time; the render thread neither copies
// nor waits. An import is freed only after the GPU work that read it is complete; any drain of guest memory over its
// range (unmap, offload, a command processor write: AsyncUpload::DrainRange) first waits for that work, so no GPU read
// outlives the pages or sees bytes written after the work was recorded. Where the import fails (alignment, a range not
// importable) the old path runs: nothing changes. 0 / unset: as before.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Tiron7 {

[[nodiscard]] inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TIRON7");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// (v3) The import itself only with KYTY_TIRON7_IMPORT=1: NVIDIA refuses RAIZ's private guest memory (20261120-000053: every
// vkAllocateMemory VK_ERROR_OUT_OF_DEVICE_MEMORY over MEM_PRIVATE PAGE_READONLY pages, 81 tries, 53.7 ms lost).
[[nodiscard]] inline bool ImportOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_TIRON7_IMPORT");
		return On() && v != nullptr && v[0] == '1';
	}();
	return on;
}

// Smallest upload that goes the native way (KYTY_TIRON7_MIN_KB, default 1024): small ones cost less copied.
[[nodiscard]] inline uint64_t MinBytes() {
	static const uint64_t bytes = [] {
		const char* v = std::getenv("KYTY_TIRON7_MIN_KB");
		return (v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 10) : 1024ull) * 1024ull;
	}();
	return bytes;
}

// Set at device creation (vulkanWindow.cpp): VK_EXT_external_memory_host enabled.
inline std::atomic<bool> g_extension {false};

// Counters for the [tiron7] line.
inline std::atomic<uint64_t> g_imports {0}, g_import_bytes {0}, g_import_us {0}, g_fails {0}, g_frees {0},
    g_drain_waits {0}, g_drain_wait_us {0}, g_live {0}, g_live_bytes {0}, g_fail_us {0}, g_skipped {0};

} // namespace Tiron7
