#ifndef EMULATOR_SRC_LOCAL_MEM_GUARD_H_
#define EMULATOR_SRC_LOCAL_MEM_GUARD_H_
// Running out of host memory without crashing (16 GB PCs: the RAM and the commit limit fill up).
//
// An allocation that fails for lack of memory (VirtualAlloc/MapViewOfFile3 commit, operator new, vkAllocateMemory,
// vmaCreate*) asks ReleaseNow() to run every registered releaser, waits a little and tries again a few times
// (Retry/ShouldRetry). Only when it still fails, FatalOutOfMemory() says so plainly ("not enough RAM: close other
// programs", in the log and a message box) and ends the process, instead of an access violation later.
//
// Nothing here trims the working set, writes to the page file or changes a Windows setting: releasers drop memory the
// emulator can rebuild (caches, pools' free lists, staging buffers while idle), and the built-in one decommits the
// heaps' free blocks (HeapCompact).
//
// Compiled into the common library (src/common/CMakeLists.txt), so the platform layer and the tests can call it.
// Pressure monitoring is not here: a monitor (KYTY_LOWRAM budgets) calls OnPressure(), or polls ReleaseEpoch().
#include <cstddef>
#include <cstdint>
#include <functional>

namespace MemGuard {

// A releaser runs on the guard's own thread ("Kyty.MemGuard"), never on the thread that ran out: that thread may hold
// any lock, so a releaser must not wait for guest address-space locks or the render thread (try_lock, or flag the work
// for its owner and return 0). Returns the bytes it dropped (0 when unknown). Thread-safe.
using ReleaseFn = std::function<uint64_t()>;
int  RegisterReleaser(const char* name, ReleaseFn fn); // returns an id for UnregisterReleaser
void UnregisterReleaser(int id);

// Runs every releaser and the heap compaction now, on the guard thread, and waits up to `wait_ms` for that pass
// (concurrent calls share one pass). Returns the bytes released as reported (0 on timeout). `reason` is logged.
uint64_t ReleaseNow(const char* reason, uint32_t wait_ms = 300);

// Thin hook for a pressure monitor: a release pass without waiting, at most one every 2 s.
void OnPressure(const char* reason);

// +1 per release pass: a subsystem working on its own thread (a texture collector) can poll it and collect harder.
[[nodiscard]] uint64_t ReleaseEpoch();

// Starts the guard thread and installs the operator new handler (call early: a thread made at the moment memory ran
// out may fail). Safe to call more than once.
void Start();

// Windows error codes that mean "out of memory / commit" (worth a release and a retry).
[[nodiscard]] bool IsOutOfMemoryError(uint32_t error);

// An allocation of `bytes` for `what` failed for lack of memory; attempt counts from 0. Runs a release pass and waits
// (longer each attempt) and returns true while another try is worth it (MaxRetries in all).
constexpr int MaxRetries = 4;
bool ShouldRetry(const char* what, uint64_t bytes, int attempt);

// Calls `alloc` until it gives a truthy value or the retries end.
template <typename F>
auto Retry(const char* what, uint64_t bytes, F&& alloc) -> decltype(alloc()) {
	auto result = alloc();
	for (int attempt = 0; !result && ShouldRetry(what, bytes, attempt); ++attempt) result = alloc();
	return result;
}

// The same for an OS call (VirtualAlloc, MapViewOfFile3...): retried only while its error (GetLastError / errno) says
// out of memory; the error is kept for the caller's own failure path.
[[nodiscard]] uint32_t LastOsError();
void                   SetLastOsError(uint32_t error);
template <typename F>
auto RetryOs(const char* what, uint64_t bytes, F&& alloc) -> decltype(alloc()) {
	for (int attempt = 0;; ++attempt) {
		auto result = alloc();
		if (result) return result;
		const uint32_t error = LastOsError();
		if (!IsOutOfMemoryError(error) || !ShouldRetry(what, bytes, attempt)) {
			SetLastOsError(error);
			return result;
		}
	}
}

// The last word: the memory state in the log, a clear message (log + message box unless KYTY_NO_MESSAGEBOX=1), exit.
[[noreturn]] void FatalOutOfMemory(const char* what, uint64_t bytes);

// "process commit .. MiB, working set .. MiB; system RAM available .. of .. MiB; commit .. of .. MiB" (no allocation:
// safe in crash reports).
void FormatState(char* out, size_t size);
void LogState(const char* label); // stdout, flushed
// The system is nearly out (commit within 3% / 512 MiB of its limit, or under 256 MiB of RAM available).
[[nodiscard]] bool SystemNearlyOut();
// A fatal error is ending the process: when the system is nearly out, say that RAM is the likely cause (message box).
void NoteFatalExit();

} // namespace MemGuard

#endif // EMULATOR_SRC_LOCAL_MEM_GUARD_H_
