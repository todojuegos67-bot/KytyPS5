// Instrumented builds only (KYTY_PGO_GENERATE, CMakeLists.txt). The port quits with TerminateProcess (menu) or
// quick_exit (window), which skip LLVM's atexit profile writer: write the profile (LLVM_PROFILE_FILE) every 20 s
// on a helper thread and once more at quick_exit. The counters are cumulative, so the last write holds the whole run.
#include <chrono>
#include <cstdlib>
#include <thread>

extern "C" int __llvm_profile_write_file(void);

namespace {
int StartPgoDump() {
	std::at_quick_exit([] { (void)__llvm_profile_write_file(); });
	std::thread([] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(20));
			(void)__llvm_profile_write_file();
		}
	}).detach();
	return 0;
}
const int g_pgo_dump_started = StartPgoDump();
} // namespace
