#pragma once
// (mem1 etapa 2, 11-23) KYTY_MEM1_STAGE2=1 (the port's default on PCs of 20 GB of RAM or less; =0 off): the native
// offload's store (texture-offload.h, RAM17: the bytes of a range taken out of RAM) kept in video memory instead of the
// disk file, and RAM18's pass (ram-guard.cpp) takes the streamed textures' zone (VirtualMemo: the game's CPU never reads
// it, Ghidra 01.007.000: written by the file read 0x8fed14 and the chunk memcpy 0x901459 only) once no emulator window
// read it for KYTY_MEM1_STAGE2_IDLE_S (30) s, not only what Windows found cold. Any touch (a game write, an emulator
// window, an upload) reads the range back from video memory first (a copy on the GPU, synchronous), exactly the bytes
// that were dropped: nothing ever comes back from a GPU image.
//
// Video memory: only on a discrete GPU, and only out of the room the caches' budget leaves (GetTotalMemoryBudget: the
// caches collect from the capped budget; the store's bytes are added to that cap, so the caches keep exactly their room)
// below the driver's budget less KYTY_MEM1_STAGE2_MARGIN_MB (1024). Under 1 GiB of such room (8 GB cards, the 8 GB
// simulation) it stays off and nothing changes. At most KYTY_MEM1_STAGE2_MB (3072), grown in 64 MiB blocks.
//
// The implementation is in graphics/host_gpu/vramStore.cpp (registered at the allocator's creation); this header is the
// registry the offload and ram-guard read: no link dependency on the graphics code.

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace VramStore {

using IoFn    = bool (*)(bool write, uint64_t offset, uint8_t* data, uint64_t bytes);
using ProbeFn = bool (*)();

struct Registry {
	std::atomic<IoFn>     io {nullptr};     // set at registration
	std::atomic<ProbeFn>  probe {nullptr};  // set at registration (first call creates the resources and decides)
	std::atomic<bool>     active {false};   // the probe said yes: the store takes new ranges
	std::atomic<uint64_t> cap {0};          // bytes the store may hold (decided by the probe)
	std::atomic<uint64_t> allocated {0};    // video memory the store holds (its blocks), added to the caches' cap
	std::atomic<uint64_t> writes {0}, write_bytes {0}, write_us {0};
	std::atomic<uint64_t> reads {0}, read_bytes {0}, read_us {0}, read_us_max {0};
	std::atomic<uint64_t> failed {0}, refused {0}; // (failed: a GPU copy failed; refused: no room for a block)
};

inline Registry& Get() {
	static Registry registry;
	return registry;
}

inline bool Asked() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_MEM1_STAGE2");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

inline uint64_t AskedBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_MEM1_STAGE2_MB");
		const unsigned long long mb = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : 3072ull;
		return static_cast<uint64_t>(mb > 16384 ? 16384 : mb) << 20u;
	}();
	return bytes;
}

inline uint64_t MarginBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_MEM1_STAGE2_MARGIN_MB");
		const unsigned long long mb = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 10) : 1024ull;
		return static_cast<uint64_t>(mb) << 20u;
	}();
	return bytes;
}

inline int IdleSeconds() {
	static const int seconds = [] {
		const char* value = std::getenv("KYTY_MEM1_STAGE2_IDLE_S");
		const long  asked = value != nullptr && value[0] != '\0' ? std::strtol(value, nullptr, 10) : 30;
		return static_cast<int>(asked < 5 ? 5 : asked > 3600 ? 3600 : asked);
	}();
	return seconds;
}

inline bool Active() {
	return Get().active.load(std::memory_order_acquire);
}

// Once (any thread): creates the store's GPU resources and decides whether it may take ranges (false: off, the disk
// store as before). Cheap after the first call.
inline bool Probe() {
	if (!Asked()) return false;
	const ProbeFn probe = Get().probe.load(std::memory_order_acquire);
	return probe != nullptr && probe();
}

// The store's offsets carry this bit (texture-offload.h's FileAlloc / FileFree / FileIo route by it).
constexpr uint64_t OffsetBit = uint64_t {1} << 62u;

} // namespace VramStore
