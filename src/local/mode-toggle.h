#ifndef EMULATOR_SRC_LOCAL_MODE_TOGGLE_H_
#define EMULATOR_SRC_LOCAL_MODE_TOGGLE_H_
// KYTY_SPECULATE's mode defaults in one place, and the live mode switch. Every switch whose default depends on the mode
// takes it from Table (Default, at its first Find); the live `mode 0|4` (live-control.h) asks for the mode to change
// while the game runs, and the GPU thread applies it between submissions, when no speculated queue has one started and
// not done (GuestGpu::ThreadRun): to 0 the speculation's threads are stopped and its results not committed finished as
// lost ones (Speculation::Stop), to 4 they are made and the queued command buffers handed to them; the table's values
// for the new mode are set at the same moment, a switch whose environment variable is set kept. With KYTY_SPECULATE
// unset nothing of mode 4 is made until a `mode 4`; the GPU thread's loop reads one relaxed atomic.
#include "live-switch.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
extern "C" {
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
extern volatile std::atomic_uint32_t kyty_local_table_target_read;
#endif
}
namespace ModeToggle {
struct Row {
	const char* name; // the live switch (its environment variable: KYTY_<NAME>)
	int64_t     on4;  // its default under KYTY_SPECULATE=4
	int64_t     on0;  // and otherwise
};
// The mode-defaulted switches: under KYTY_SPECULATE=4 its measured set (the stone arch, 1-2 Nov: 55-59 fps with the set
// against 51-53 without), else 0 (serve_in_waits: 1, the upload waits; eager_feedback, feedback_batch, table_serials
// and table_memo: off on the normal path upstream, cfbf1148).
inline constexpr Row Table[] = {
    {"fill_buffer_clears", 1, 0},       {"spec_draw_indirect", 1, 0},    {"spec_predict_dcc", 3, 0},
    {"spec_read_one_set", 1, 0},        {"speculate_flush_every", 8, 0}, {"speculate_keep_pass", 1, 0},
    {"speculate_texel_sync", 1, 0},     {"spec_late_reads", 1, 0},       {"spec_start_instance_free", 1, 0},
    {"spec_chunk_in_order", 1, 0},      {"serve_in_waits", 7, 1},        {"write_guard_spec", 1, 0},
    {"eager_feedback", 1, 0},           {"feedback_batch", 1, 0},        {"table_serials", 1, 0},
    {"table_memo", 1, 0},               {"spec_worker_packets", 1, 0},  {"table_learn_portable", 2, 0},
};
// KYTY_TABLE_TARGET_READ (not a live switch: read where image sets are made): on under mode 4.
inline constexpr uint32_t TableTargetRead4 = 1, TableTargetRead0 = 0;
// The switch's default in the mode now (a name not in the table: 0).
inline int64_t Default(const char* name) {
	const bool four = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4;
	for (const auto& row: Table)
		if (std::strcmp(row.name, name) == 0) return four ? row.on4 : row.on0;
	return 0;
}
// The mode asked for (0 or 4), -1 none.
inline std::atomic<int> g_request {-1};
inline void Request(int mode) {
	if (mode < 0) mode = ::kyty_local_speculate_mode.load(std::memory_order_relaxed) == 4 ? 0 : 4;
	g_request.store(mode == 4 ? 4 : 0, std::memory_order_release);
}
// The table's values for `mode`, a switch not made yet made with them (its later Find keeps the value): on a live flip
// and at a KYTY_SPECULATE=4 boot (GuestGpu::ThreadRun, before the first packet), so a switch whose Find has main's
// default (eager_feedback, feedback_batch, table_serials, table_memo: 0) is on under mode 4 all the same.
inline void ApplyTable(int mode) {
	for (const auto& row: Table) {
		std::string variable = "KYTY_";
		for (const char* c = row.name; *c != '\0'; ++c) variable += static_cast<char>(*c >= 'a' && *c <= 'z' ? *c - 32 : *c);
		const char* env = std::getenv(variable.c_str());
		const int64_t value = mode == 4 ? row.on4 : row.on0;
		if (env == nullptr || *env == '\0') LiveSwitch::Find(row.name, value).store(value, std::memory_order_relaxed);
	}
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	const char* target_read = std::getenv("KYTY_TABLE_TARGET_READ");
	if (target_read == nullptr || *target_read == '\0')
		::kyty_local_table_target_read.store(mode == 4 ? TableTargetRead4 : TableTargetRead0, std::memory_order_relaxed);
#endif
}
} // namespace ModeToggle
#endif // EMULATOR_SRC_LOCAL_MODE_TOGGLE_H_
