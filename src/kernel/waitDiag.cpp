#include "kernel/waitDiag.h"

#include "common/logging/log.h"
#include "common/threads.h"
#include "loader/gt7Title.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace Libs::LibKernel::WaitDiag {

namespace {

struct Slot {
	std::atomic<const char*> what {nullptr}; // nullptr: not waiting
	std::atomic<uint64_t>    object {0};
	std::atomic<uint64_t>    since_ns {0};
	std::atomic<int64_t>     timeout_us {-1};
	std::atomic<int>         thread_id {0};
	char                     name[32] {};
};

constexpr size_t           SlotCount = 512;
std::array<Slot, SlotCount> g_slots;
std::atomic<size_t>        g_next_slot {0};
thread_local Slot*         t_slot      = nullptr;
thread_local bool          t_no_slot   = false;

bool Enabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_CPU_WAIT_DIAG");
		if (env != nullptr) return env[0] == '1';
		return ::Loader::IsGranTurismo7();
	}();
	return enabled;
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void Monitor() {
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		const auto now     = NowNs();
		const auto count   = std::min(g_next_slot.load(std::memory_order_acquire), SlotCount);
		uint32_t   printed = 0;
		for (size_t index = 0; index < count && printed < 48u; index++) {
			auto&       slot  = g_slots[index];
			const auto* what  = slot.what.load(std::memory_order_acquire);
			const auto  since = slot.since_ns.load(std::memory_order_relaxed);
			if (what == nullptr || since == 0 || now < since || now - since < 5'000'000'000ull) continue;
			char name[sizeof(slot.name)] {};
			std::memcpy(name, slot.name, sizeof(name) - 1);
			const auto timeout = slot.timeout_us.load(std::memory_order_relaxed);
			char       line[256];
			std::snprintf(line, sizeof(line),
			              "CPU wait: thread %d blocked %.1f s in %s object=0x%016" PRIx64 " %s timeout=%s%" PRId64
			              "\n",
			              slot.thread_id.load(std::memory_order_relaxed), static_cast<double>(now - since) / 1e9,
			              what, slot.object.load(std::memory_order_relaxed), name, timeout < 0 ? "none" : "",
			              timeout < 0 ? int64_t {0} : timeout);
			std::printf("%s", line);
			LOGF("%s", line);
			printed++;
		}
		if (printed != 0) std::fflush(stdout);
	}
}

Slot* ThisSlot() {
	if (t_slot != nullptr || t_no_slot) return t_slot;
	const auto index = g_next_slot.fetch_add(1, std::memory_order_acq_rel);
	if (index >= SlotCount) {
		t_no_slot = true;
		return nullptr;
	}
	t_slot = &g_slots[index];
	t_slot->thread_id.store(Common::Thread::GetThreadIdUnique(), std::memory_order_relaxed);
	return t_slot;
}

} // namespace

Scope::Scope(const char* what, uint64_t object, std::string_view name, int64_t timeout_us) {
	if (!Enabled()) return;
	static std::once_flag started;
	std::call_once(started, [] { std::thread(Monitor).detach(); });
	auto* slot = ThisSlot();
	if (slot == nullptr) return;
	const auto length = std::min(name.size(), sizeof(slot->name) - 1);
	std::memcpy(slot->name, name.data(), length);
	slot->name[length] = '\0';
	slot->object.store(object, std::memory_order_relaxed);
	slot->timeout_us.store(timeout_us, std::memory_order_relaxed);
	slot->since_ns.store(NowNs(), std::memory_order_relaxed);
	slot->what.store(what, std::memory_order_release);
	m_active = true;
}

Scope::~Scope() {
	if (m_active && t_slot != nullptr) {
		t_slot->what.store(nullptr, std::memory_order_release);
	}
}

} // namespace Libs::LibKernel::WaitDiag
