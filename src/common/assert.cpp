#include "common/assert.h"

#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"
#include "local/mem-guard.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <thread>

#if defined(__linux__)
#include <execinfo.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Common {

// RecoverableExitScope: the message of the pending EXIT on this thread.
static thread_local bool        t_recoverable = false;
static thread_local std::string t_recoverable_message;

RecoverableExitScope::RecoverableExitScope() {
	t_recoverable = true;
}

RecoverableExitScope::~RecoverableExitScope() {
	t_recoverable = false;
}

// The raw return addresses of a fatal error (resolve with addr2line against the executable;
// Windows: llvm-symbolizer --obj=kyty_emulator.exe with the build's PDB next to it).
static void WriteFatalBacktrace() {
#if defined(__linux__)
	void*     frames[48];
	const int count = backtrace(frames, 48);
	std::fflush(nullptr);
	(void)!write(STDOUT_FILENO, "--- Backtrace ---\n", 18);
	backtrace_symbols_fd(frames, count, STDOUT_FILENO);
#elif defined(_WIN32)
	void*      frames[48];
	const auto count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
	std::printf("--- Backtrace --- (exe base %p)\n", static_cast<void*>(GetModuleHandleW(nullptr)));
	for (USHORT i = 0; i < count; ++i) std::printf("  %p\n", frames[i]);
	std::fflush(nullptr);
#endif
}

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	// The memory state too (16 GB PCs: a fatal error at the commit limit reads as RAM, not as a bug).
	char memory[256];
	MemGuard::FormatState(memory, sizeof(memory));
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n--- Memory ---\n{}\n", KYTY_BUILD_LABEL, title, text,
	                   file, line, memory);
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	if (t_recoverable) {
		t_recoverable_message = fmt::format("{} in {}:{}", text, file, line);
		return 1;
	}
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	WriteFatalBacktrace();
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	if (t_recoverable) {
		t_recoverable_message = fmt::format("{} in {}:{}", text, file, line);
		return 1;
	}
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	WriteFatalBacktrace();
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	if (t_recoverable) {
		t_recoverable_message = fmt::format("{} in {}:{}", text, file, line);
		return 1;
	}
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	WriteFatalBacktrace();
	return 1;
}

static void (*g_exit_drain)() = nullptr;

void SetExitDrain(void (*drain)()) noexcept {
	g_exit_drain = drain;
}

void RunExitDrain() noexcept {
	if (g_exit_drain != nullptr) g_exit_drain();
}

static std::atomic<int> g_exit_blocking_reports {0};

void BeginExitBlockingReport() noexcept {
	g_exit_blocking_reports.fetch_add(1, std::memory_order_acq_rel);
}

void EndExitBlockingReport() noexcept {
	g_exit_blocking_reports.fetch_sub(1, std::memory_order_acq_rel);
}

void DbgExit(int status) {
	if (t_recoverable) {
		throw RecoverableExit {std::move(t_recoverable_message)};
	}
	// (A device loss: the thread that met it first writes what the driver reports while the others' submissions fail.)
	for (int i = 0; i < 500 && g_exit_blocking_reports.load(std::memory_order_acquire) > 0; ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	// (System nearly out of memory: a message box says RAM is the likely cause.)
	MemGuard::NoteFatalExit();
	RunExitDrain();
#if defined(_WIN32)
	// Not ExitProcess (std::_Exit): it kills the other threads wherever they are and then runs the DLLs'
	// detach routines on this one, where a driver's waited forever on what a killed thread held (a
	// crashed emulator stayed a one-thread process, unkillable, holding its memory).
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(status));
#endif
	std::_Exit(status);
}

} // namespace Common
