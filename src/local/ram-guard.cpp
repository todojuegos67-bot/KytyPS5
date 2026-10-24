#include "local/ram-guard.h"

#include "local/local-platform.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace RamGuard {

namespace {

constexpr uint64_t MiB = uint64_t {1} << 20u;
constexpr uint64_t GiB = uint64_t {1} << 30u;

std::atomic<int> g_level {None};

std::mutex                                                        g_flush_mutex;
std::vector<std::pair<const char*, uint64_t (*)(int, uint64_t)>> g_flushes;

const char* Env(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

uint64_t EnvMiB(const char* name, uint64_t fallback_bytes) {
	const char* value = Env(name);
	return value != nullptr ? std::strtoull(value, nullptr, 10) * MiB : fallback_bytes;
}

const char* LevelName(int level) {
	return level == Critical ? "critical" : level == Low ? "low" : "none";
}

struct Counted {
	uint64_t ram = 0, available = 0, headroom = 0, private_bytes = 0, working_set = 0;
	bool     valid = false;
};

Counted Count() {
	Counted    c;
	const auto m = LocalPlatform::QueryMemory();
	if (!m.valid) return c;
	c.valid         = true;
	c.ram           = m.total_phys;
	c.available     = m.avail_phys;
	c.headroom      = m.commit_limit > m.commit_total ? m.commit_limit - m.commit_total : 0;
	c.private_bytes = m.private_bytes;
	c.working_set   = m.working_set;
	if (const uint64_t sim = EnvMiB("KYTY_SIMULATE_RAM_MB", 0); sim != 0) {
		const uint64_t os        = EnvMiB("KYTY_SIMULATE_OS_MB", 3072 * MiB);
		const uint64_t used      = os + m.working_set;
		const uint64_t available = sim > used ? sim - used : 0;
		const uint64_t limit     = sim + EnvMiB("KYTY_SIMULATE_PAGEFILE_MB", sim);
		const uint64_t charge    = m.private_bytes + EnvMiB("KYTY_SIMULATE_OS_COMMIT_MB", 4096 * MiB);
		c.ram                    = std::min(c.ram, sim);
		c.available              = std::min(c.available, available);
		c.headroom               = std::min(c.headroom, limit > charge ? limit - charge : 0);
	}
	return c;
}

uint64_t FlushPass(int level, uint64_t budget) {
	uint64_t freed = 0;
	(void)LocalPlatform::CompactHeaps(); // (the heaps' free blocks back to the system: no size reported)
	std::vector<std::pair<const char*, uint64_t (*)(int, uint64_t)>> flushes;
	{
		std::lock_guard lock(g_flush_mutex);
		flushes = g_flushes;
	}
	for (const auto& [name, flush]: flushes) {
		if (freed >= budget) break;
		freed += flush(level, budget - freed);
	}
	return freed;
}

void Watch() {
	const bool     low_ram   = LowRam();
	const double   factor    = low_ram ? 1.25 : 1.0;
	const int      log_s     = Env("KYTY_RAM_LOG_SECONDS") != nullptr ? std::atoi(Env("KYTY_RAM_LOG_SECONDS")) : 30;
	int            flush_s   = Env("KYTY_RAM_FLUSH_SECONDS") != nullptr ? std::atoi(Env("KYTY_RAM_FLUSH_SECONDS"))
	                                                                    : (low_ram ? 30 : 0);
	const uint64_t flush_cap = std::max<uint64_t>(MiB, EnvMiB("KYTY_RAM_FLUSH_MB", 1024 * MiB));
	int            seconds = 0, since_pass = 0;
	int            last_level = None;
	std::printf("RAM guard: on (pressure watcher; log every %d s; periodic flush %s)\n", log_s,
	            flush_s > 0 ? (std::to_string(flush_s) + " s").c_str() : "off");
	std::fflush(stdout);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		++seconds;
		++since_pass;
		const auto c = Count();
		if (!c.valid) continue;
		const uint64_t low_avail  = static_cast<uint64_t>(
		    static_cast<double>(EnvMiB("KYTY_RAM_LOW_AVAIL_MB", std::max<uint64_t>(1024 * MiB, c.ram / 16))) * factor);
		const uint64_t low_commit = static_cast<uint64_t>(static_cast<double>(EnvMiB("KYTY_RAM_LOW_COMMIT_MB", 2048 * MiB)) * factor);
		int level = None;
		if (c.available < low_avail || c.headroom < low_commit || LocalPlatform::LowMemorySignaled()) level = Low;
		if (c.available < low_avail / 2 || c.headroom < low_commit / 2) level = Critical;
		g_level.store(level, std::memory_order_relaxed);
		if (level != None && (level > last_level || since_pass >= 10)) {
			const auto     start = std::chrono::steady_clock::now();
			const uint64_t freed = FlushPass(level, flush_cap);
			since_pass           = 0;
			std::printf("RAM: pressure %s (available %llu MiB, commit headroom %llu MiB), freed %llu MB in %.1f ms\n",
			            LevelName(level), static_cast<unsigned long long>(c.available / MiB),
			            static_cast<unsigned long long>(c.headroom / MiB), static_cast<unsigned long long>(freed / MiB),
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
			std::fflush(stdout);
		} else if (level == None && last_level != None) {
			std::printf("RAM: pressure over (available %llu MiB, commit headroom %llu MiB)\n",
			            static_cast<unsigned long long>(c.available / MiB), static_cast<unsigned long long>(c.headroom / MiB));
			std::fflush(stdout);
		}
		last_level = level;
		if (flush_s > 0 && seconds % flush_s == 0) {
			const auto     start  = std::chrono::steady_clock::now();
			const uint64_t freed  = FlushPass(None, flush_cap);
			const auto     after  = Count();
			std::printf("RAM flush: freed %llu MB in %.1f ms; private %llu -> %llu MiB, working set %llu -> %llu MiB\n",
			            static_cast<unsigned long long>(freed / MiB),
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
			            static_cast<unsigned long long>(c.private_bytes / MiB),
			            static_cast<unsigned long long>(after.private_bytes / MiB),
			            static_cast<unsigned long long>(c.working_set / MiB),
			            static_cast<unsigned long long>(after.working_set / MiB));
			std::fflush(stdout);
		}
		if (log_s > 0 && seconds % log_s == 0) {
			std::printf("RAM (%d s): private %llu MiB, working set %llu MiB, available %llu MiB, commit headroom %llu MiB%s\n",
			            log_s, static_cast<unsigned long long>(c.private_bytes / MiB),
			            static_cast<unsigned long long>(c.working_set / MiB), static_cast<unsigned long long>(c.available / MiB),
			            static_cast<unsigned long long>(c.headroom / MiB),
			            Env("KYTY_SIMULATE_RAM_MB") != nullptr ? " (simulated PC)" : "");
			std::fflush(stdout);
		}
	}
}

} // namespace

bool LowRam() {
	static const bool on = [] {
		if (const char* value = Env("KYTY_LOWRAM"); value != nullptr) return value[0] == '1';
		uint64_t ram = LocalPlatform::QueryMemory().total_phys;
		if (const uint64_t sim = EnvMiB("KYTY_SIMULATE_RAM_MB", 0); sim != 0) ram = ram != 0 ? std::min(ram, sim) : sim;
		return ram != 0 && ram <= 20 * GiB;
	}();
	return on;
}

void ApplyLowRamDefaults() {
	static std::once_flag once;
	std::call_once(once, [] {
		const uint64_t ram = LocalPlatform::QueryMemory().total_phys;
		std::printf("RAM: low-RAM profile %s (%llu MiB of RAM%s; KYTY_LOWRAM=%d turns it %s)\n", LowRam() ? "on" : "off",
		            static_cast<unsigned long long>(ram / MiB),
		            Env("KYTY_SIMULATE_RAM_MB") != nullptr ? ", simulated less" : "", LowRam() ? 0 : 1,
		            LowRam() ? "off" : "on");
		std::fflush(stdout);
	});
}

void Start() {
	static std::once_flag once;
	std::call_once(once, [] {
		if (const char* value = Env("KYTY_RAM_GUARD"); value != nullptr && value[0] == '0') {
			std::printf("RAM guard: off (KYTY_RAM_GUARD=0)\n");
			std::fflush(stdout);
			return;
		}
		std::thread(Watch).detach();
	});
}

int PressureLevel() {
	return g_level.load(std::memory_order_relaxed);
}

void RegisterFlush(const char* name, uint64_t (*flush)(int, uint64_t)) {
	std::lock_guard lock(g_flush_mutex);
	g_flushes.emplace_back(name, flush);
}

} // namespace RamGuard
