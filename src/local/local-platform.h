#pragma once
// Host-OS helpers for the local performance and diagnostic code (thread names, thread
// ids, CPU pinning), so the local headers stay free of <windows.h> and Linux-only APIs.

#include <cstddef>
#include <cstdint>

namespace LocalPlatform {

// Names the calling thread (Linux: 15 characters are kept).
void SetThreadName(const char* name);

// The OS id of the calling thread (Linux: gettid, Windows: GetCurrentThreadId).
uint32_t ThreadId();

// Pins the calling thread to a comma-separated CPU list ("1,2,3"); an empty or invalid
// list leaves the affinity alone.
void PinThreadToCpuList(const char* list);

// Pins a thread (a CurrentThreadHandle) to a CPU list; false if the list or the call fails.
bool PinThreadHandleToCpuList(uint64_t handle, const char* list);

// Windows: the process's default CPU set as a CPU list; "" restores all CPUs. Linux: false.
bool SetProcessDefaultCpuList(const char* list);

// Opaque handle of the calling thread for ThreadCpuSeconds (valid while the thread runs).
uint64_t CurrentThreadHandle();
double   ThreadCpuSeconds(uint64_t handle);

// Total CPU seconds of the threads with this name (0 where the OS cannot enumerate them).
double NamedThreadsCpuSeconds(const char* name);

// Stack bounds of the calling thread; false when unknown.
bool CurrentThreadStack(uint64_t* low, uint64_t* high);

// The calling thread kept off the CPUs of a list (KYTY_RENDER_CPUS; null or empty: any CPU); with
// below-normal priority too: background work that must not hold up the game's threads.
void AvoidCpuList(const char* avoid_cpus);
void MakeBackgroundThread(const char* avoid_cpus);

// A temporary file for scratch data, deleted when it is closed or the process ends (Windows keeps it
// in memory while it can: FILE_ATTRIBUTE_TEMPORARY); 0 when none could be made. Writes and reads go
// to any offset, from any thread.
uint64_t OpenScratchFile();
bool     WriteScratchFile(uint64_t file, uint64_t offset, const void* data, size_t size);
bool     ReadScratchFile(uint64_t file, uint64_t offset, void* data, size_t size);
void     CloseScratchFile(uint64_t file);
// An existing file (a UTF-8 path) for ReadScratchFile and CloseScratchFile; 0 when it cannot be opened.
uint64_t OpenFileForReading(const char* path);

#if defined(_WIN32)
// With its output in files (run-windows.ps1's logs), the console window a launcher gave the process
// stays empty, and closing it would end the game: the process leaves it (it closes).
void ReleaseRedirectedConsole();

// Standard output written to a file: through a pipe a background thread empties into the file (a write to the
// redirected log blocked the writer ~6 ms in WriteFile: the GPU thread's per-slow-frame flush took 9% of its time,
// 10-06). DrainStdout, at the process's end, sends standard output to NUL and waits until what was written before
// reached the file.
void AsyncStdout();
void DrainStdout();

// Collects the unwind tables of the loaded images for SampleThread (call before sampling).
void PrepareSampling();

// Suspends the thread (a CurrentThreadHandle) and records pc and its callers (unwound with
// the tables of PrepareSampling, else the word at rsp and the frame-pointer chain) within its
// stack [stack_low, stack_high) into out[0..words). stack_high 0: the stack is unknown (a guest
// thread), the walk stays inside the committed region that holds rsp.
bool SampleThread(uint64_t handle, uint64_t stack_low, uint64_t stack_high, uint64_t* out, size_t words);
// A handle to another thread of this process for SampleThread and PinThreadHandleToCpuList (0 if
// it cannot be opened), and its release.
uint64_t OpenThreadForSampling(uint32_t thread_id);
void     CloseThreadForSampling(uint64_t handle);

// Base address and path of the module that contains the address.
bool ModuleOf(const void* address, uint64_t* base, char* path, size_t path_size);
#endif

// The process's memory (bytes): private commit (Windows PrivateUsage / Linux RSS-shared) and the working
// set (resident); zeros where unknown. For the run log's 30-second "System memory:" line.
void ProcessMemory(uint64_t* private_bytes, uint64_t* working_set);

} // namespace LocalPlatform
