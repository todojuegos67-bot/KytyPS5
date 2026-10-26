#include "ram-guard.h"

#include "guest-lazy-commit.h"
#include "local-platform.h"
#include "mem-guard.h"
#include "soft-release.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace RamGuard {

namespace {

constexpr uint64_t MiB = uint64_t {1} << 20u;
constexpr uint64_t GiB = uint64_t {1} << 30u;

std::atomic<int>      g_level {None};
// The RAM governor (ram-guard.h): its stage, the offload bytes wanted, what scans took since the guard's last sample.
std::atomic<int>      g_gov_stage {0};
std::atomic<uint64_t> g_gov_demand {0};
std::atomic<uint64_t> g_gov_taken {0}, g_gov_taken_total {0};
std::atomic<bool>     g_gov_urgent {false};
std::atomic<uint64_t> g_available {0};
std::atomic<uint64_t> g_released_count[ReleasedKinds] {};
std::atomic<uint64_t> g_released_bytes[ReleasedKinds] {};
std::atomic<uint64_t> g_released_count_total[ReleasedKinds] {};
std::atomic<uint64_t> g_released_bytes_total[ReleasedKinds] {};

std::atomic<const void*> g_store_alias {nullptr};
std::atomic<uint64_t>    g_store_size {0};
std::atomic<int>         g_store_charge {static_cast<int>(StoreCharge::AllCommitted)};

std::mutex                                           g_flush_mutex;
std::vector<std::pair<const char*, uint64_t (*)(int)>> g_flushes;
struct BudgetFlush {
	const char*   name;
	FlushClass    kind;
	BudgetFlushFn fn;
};
std::vector<BudgetFlush> g_budget_flushes;
std::atomic<bool>        g_periodic {false};
std::vector<std::pair<const char*, uint64_t (*)()>> g_figures;
std::string                                         g_flush_defaults; // what the flush turned on (the log line)

const char* Env(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

[[maybe_unused]] uint64_t EnvMiB(const char* name, uint64_t fallback_bytes) {
	const char* value = Env(name);
	return value != nullptr ? std::strtoull(value, nullptr, 10) * MiB : fallback_bytes;
}

bool SetDefault(const char* name, const char* value, std::string* set) {
	if (Env(name) != nullptr) return false;
#if defined(_WIN32)
	if (_putenv_s(name, value) != 0) return false;
	(void)SetEnvironmentVariableA(name, value);
#else
	if (setenv(name, value, 0) != 0) return false;
#endif
	if (!set->empty()) *set += ' ';
	*set += name;
	*set += '=';
	*set += value;
	return true;
}

[[maybe_unused]] const char* LevelName(int level) {
	return level == Critical ? "critical" : level == Low ? "low" : "none";
}

} // namespace

bool LowRam() {
	static const bool on = [] {
		if (const char* value = Env("KYTY_LOWRAM"); value != nullptr) return value[0] == '1';
		const uint64_t ram = LocalPlatform::PhysicalMemory();
		return ram != 0 && ram <= 20 * GiB;
	}();
	return on;
}

int64_t FlushSeconds() {
	static const int64_t seconds = [] {
		const char* value = Env("KYTY_RAM_FLUSH_SECONDS");
		return value != nullptr ? std::max<int64_t>(0, std::strtoll(value, nullptr, 10)) : 0;
	}();
	return seconds;
}

int FlushLevel() {
	static const int level = [] {
		const char* value = Env("KYTY_RAM_FLUSH_LEVEL");
		const long  n     = value != nullptr ? std::strtol(value, nullptr, 10) : 1;
		return n >= 2 ? 2 : 1;
	}();
	return level;
}

uint64_t FlushImageIdleSeconds() {
	static const uint64_t seconds = [] {
		if (FlushSeconds() <= 0) return uint64_t {0};
		const char* value = Env("KYTY_RAM_FLUSH_IMAGE_IDLE_SECONDS");
		if (value != nullptr) return std::strtoull(value, nullptr, 10);
		return FlushLevel() >= 2 ? uint64_t {120} : uint64_t {0};
	}();
	return seconds;
}

void RegisterFigure(const char* name, uint64_t (*figure)()) {
	if (name == nullptr || figure == nullptr) return;
	std::lock_guard lock(g_flush_mutex);
	g_figures.emplace_back(name, figure);
}

namespace {
// The switches the periodic flush turns on where the environment leaves them (ram-guard.h): the pass alone returns
// only what is free at that moment; these lower what is held all along.
void ApplyFlushDefaults() {
	if (FlushSeconds() <= 0) return;
	std::string set;
	SetDefault("KYTY_RAM_DIET", "1", &set);
	SetDefault("KYTY_SOFT_RELEASE", "1", &set);
	{
		const char* file    = Env("KYTY_BACKING_FILE");
		const bool  in_file = file != nullptr && !(file[0] == '0' && file[1] == '\0');
		const char* lazy    = Env("KYTY_GUEST_LAZY_COMMIT");
		const char* demand  = Env("KYTY_LOWRAM_ON_DEMAND");
		if (!in_file && !(lazy != nullptr && lazy[0] == '1') && !(demand != nullptr && demand[0] == '0'))
			SetDefault("KYTY_BACKING_ON_DEMAND", "1", &set);
	}
	if (FlushLevel() >= 2) {
		SetDefault("KYTY_TEXTURE_OFFLOAD", "1", &set);
		SetDefault("KYTY_RAM_FLUSH_IMAGE_IDLE_SECONDS", "120", &set);
	}
	g_flush_defaults = set;
}
// KYTY_MEMCUT=1|2 (memcut-1025): the emulator's own memory held all along, lowered only by switches that change where
// or how long memory is kept, never what a draw reads: the image upload ring (KYTY_STAGING_RING_MB: a full ring waits
// for the GPU or uploads through a temporary buffer, the same bytes), VMA's block size (KYTY_VMA_BLOCK_MB: placement
// only, less free space held inside part-used blocks; 32 is KYTY_LOWMEM_VRAM's tested value) and idle pipelines'
// lifetime (KYTY_PIPELINE_EVICT_SECONDS: made again from the pipeline cache on next use). Level 1: ring 384 MiB,
// blocks 32 MiB, pipelines idle 120 s; level 2: ring 256 MiB, pipelines idle 60 s. Every switch set by hand wins.
void ApplyMemcutDefaults() {
	const char* value = Env("KYTY_MEMCUT");
	const long  level = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
	if (level <= 0) return;
	std::string set;
	SetDefault("KYTY_STAGING_RING_MB", level >= 2 ? "256" : "384", &set);
	SetDefault("KYTY_VMA_BLOCK_MB", "32", &set);
	SetDefault("KYTY_RAM_DIET", "1", &set);
	SetDefault("KYTY_PIPELINE_EVICT_SECONDS", level >= 2 ? "60" : "120", &set);
	std::printf("[memcut] level %ld: %s (no step changes data a draw reads: verification mismatches 0 by construction)\n",
	            level >= 2 ? 2L : 1L, set.empty() ? "every switch already set" : set.c_str());
	std::fflush(stdout);
}
} // namespace

// KYTY_RAM_TARGET_MB (the governor): what it escalates to, turned on where unset - texture offload (it takes nothing
// below the soft threshold), the RAM diet's pipeline eviction, soft release's idle scratch buffers.
static void ApplyGovernorDefaults() {
	if (!GovernorOn()) return;
	std::string set;
	SetDefault("KYTY_TEXTURE_OFFLOAD", "1", &set);
	SetDefault("KYTY_RAM_DIET", "1", &set);
	SetDefault("KYTY_SOFT_RELEASE", "1", &set);
	std::printf("RAM target: on (KYTY_RAM_TARGET_MB=%s); %s\n", Env("KYTY_RAM_TARGET_MB"),
	            set.empty() ? "every switch already set" : set.c_str());
	std::fflush(stdout);
}

void ApplyLowRamDefaults() {
	static std::once_flag once;
	std::call_once(once, [] {
		ApplyMemcutDefaults();
		ApplyGovernorDefaults();
		ApplyFlushDefaults();
		const uint64_t ram    = LocalPlatform::PhysicalMemory();
		const char*    forced = Env("KYTY_LOWRAM");
		if (!LowRam()) {
			std::printf("RAM: low-RAM profile off (%llu MiB of RAM%s; KYTY_LOWRAM=1 turns it on)\n",
			            static_cast<unsigned long long>(ram / MiB), forced != nullptr ? ", KYTY_LOWRAM=0" : "");
			std::fflush(stdout);
			return;
		}
		std::string set;
		SetDefault("KYTY_SOFT_RELEASE", "1", &set);
		SetDefault("KYTY_RAM_DIET", "1", &set);
		// (ram16-1026) 1536, not 3072: a 16 GB PC sits near 3 GB free in play, and a held prefetch left shaders to
		// translate at first use on the render thread (xpr spikes); the guard's own low level is ~1 GB.
		SetDefault("KYTY_PREFETCH_MIN_AVAIL_MB", "1536", &set);
		// (No video memory budget of its own: 5120 MiB, 10-24, left the video memory in use at 9.1-9.4 GB anyway - what
		// holds it is not the collector's images - and only made the collector and pipeline eviction churn.)
		// The guest store committed as it is mapped (the game maps ~11.3 of its 13.5 GiB): only for the page-file
		// section committed at launch, the default.
		const char* file     = Env("KYTY_BACKING_FILE");
		const bool  in_file  = file != nullptr && !(file[0] == '0' && file[1] == '\0');
		const char* lazy     = Env("KYTY_GUEST_LAZY_COMMIT");
		const char* demand   = Env("KYTY_LOWRAM_ON_DEMAND");
		if (!in_file && !(lazy != nullptr && lazy[0] == '1') && !(demand != nullptr && demand[0] == '0'))
			SetDefault("KYTY_BACKING_ON_DEMAND", "1", &set);
		std::printf("RAM: low-RAM profile on (%llu MiB of RAM%s; KYTY_LOWRAM=0 turns it off): %s\n",
		            static_cast<unsigned long long>(ram / MiB), forced != nullptr ? ", KYTY_LOWRAM=1" : ", 20 GiB or less",
		            set.empty() ? "every switch already set" : set.c_str());
		std::fflush(stdout);
	});
}

int PressureLevel() {
	return g_level.load(std::memory_order_relaxed);
}

uint64_t AvailablePhysical() {
	const uint64_t value = g_available.load(std::memory_order_relaxed);
	return value != 0 ? value : LocalPlatform::AvailablePhysicalMemory();
}

void NoteGuestStore(const void* alias, uint64_t size, StoreCharge charge) {
	g_store_alias.store(alias, std::memory_order_relaxed);
	g_store_size.store(size, std::memory_order_relaxed);
	g_store_charge.store(static_cast<int>(charge), std::memory_order_relaxed);
}

void NoteReleased(Released kind, uint64_t count, uint64_t bytes) {
	if (kind < 0 || kind >= ReleasedKinds) return;
	g_released_count[kind].fetch_add(count, std::memory_order_relaxed);
	g_released_bytes[kind].fetch_add(bytes, std::memory_order_relaxed);
	g_released_count_total[kind].fetch_add(count, std::memory_order_relaxed);
	g_released_bytes_total[kind].fetch_add(bytes, std::memory_order_relaxed);
}

void RegisterFlush(const char* name, uint64_t (*flush)(int level)) {
	std::lock_guard lock(g_flush_mutex);
	g_flushes.emplace_back(name, flush);
}

void RegisterBudgetFlush(const char* name, FlushClass kind, BudgetFlushFn flush) {
	if (flush == nullptr || kind < 0 || kind >= FlushClasses) return;
	std::lock_guard lock(g_flush_mutex);
	g_budget_flushes.push_back({name, kind, flush});
}

bool PeriodicFlush() {
	return g_periodic.load(std::memory_order_relaxed);
}

bool GovernorOn() {
	static const bool on = Env("KYTY_RAM_TARGET_MB") != nullptr && std::strtoull(Env("KYTY_RAM_TARGET_MB"), nullptr, 10) != 0;
	return on;
}

int GovernorStage() {
	return g_gov_stage.load(std::memory_order_relaxed);
}

uint64_t GovernorOffloadDemand() {
	const uint64_t demand = g_gov_demand.load(std::memory_order_relaxed);
	const uint64_t taken  = g_gov_taken.load(std::memory_order_relaxed);
	return demand > taken ? demand - taken : 0;
}

void NoteGovernorOffload(uint64_t bytes) {
	g_gov_taken.fetch_add(bytes, std::memory_order_relaxed);
	g_gov_taken_total.fetch_add(bytes, std::memory_order_relaxed);
}

bool TakeGovernorUrgent() {
	return g_gov_urgent.load(std::memory_order_relaxed) && g_gov_urgent.exchange(false, std::memory_order_acq_rel);
}

#if defined(_WIN32)
namespace {

// The guest store's share of the commit charge.
uint64_t StoreCommitted() {
	const auto  charge = static_cast<StoreCharge>(g_store_charge.load(std::memory_order_relaxed));
	const auto  size   = g_store_size.load(std::memory_order_relaxed);
	const auto* alias  = static_cast<const uint8_t*>(g_store_alias.load(std::memory_order_relaxed));
	switch (charge) {
		case StoreCharge::File: return 0;
		case StoreCharge::Lazy: return GuestLazyCommit::Get().committed.load(std::memory_order_relaxed);
		case StoreCharge::OnDemand: {
			// (A walk of the alias's regions: committed runs as the game mapped them, a few thousand at most.)
			if (alias == nullptr) return size;
			uint64_t                 committed = 0;
			MEMORY_BASIC_INFORMATION info {};
			for (uint64_t at = 0; at < size;) {
				if (VirtualQuery(alias + at, &info, sizeof(info)) != sizeof(info)) return size;
				const uint64_t end  = static_cast<uint64_t>(static_cast<const uint8_t*>(info.BaseAddress) - alias) + info.RegionSize;
				const uint64_t stop = std::min(end, size);
				if (stop <= at) break;
				if (info.State == MEM_COMMIT) committed += stop - at;
				at = stop;
			}
			return committed;
		}
		case StoreCharge::AllCommitted:
		default: return size;
	}
}

struct Sample {
	uint64_t private_bytes = 0, working_set = 0;
	uint64_t total_phys = 0, available = 0;          // as counted (simulated when asked)
	uint64_t commit_total = 0, commit_limit = 0;     // as counted
	uint64_t real_available = 0, real_headroom = 0;  // Windows' own
	uint64_t store_committed = 0;
	bool     signal    = false;                      // Windows' low-memory notification
	bool     simulated = false;
	[[nodiscard]] uint64_t Headroom() const { return commit_limit > commit_total ? commit_limit - commit_total : 0; }
};

struct Settings {
	uint64_t sim_ram = 0, sim_commit_limit = 0, sim_os = 0, sim_os_commit = 0;
	uint64_t low_avail = 0, low_commit = 0;
	bool     sim_abort = false;                      // the simulated PC's end when its commit is used up
	int64_t  flush_seconds = 0;                      // KYTY_RAM_FLUSH_SECONDS: the periodic flush (0 off)
	uint64_t flush_bytes   = 0;                      // KYTY_RAM_FLUSH_MB
	bool     flush_trim    = false;                  // KYTY_RAM_FLUSH_TRIM=1: the working set trim too
};

Settings ReadSettings() {
	Settings s;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	const uint64_t real_ram = GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
	s.sim_ram          = EnvMiB("KYTY_SIMULATE_RAM_MB", 0);
	if (real_ram != 0 && s.sim_ram >= real_ram) s.sim_ram = 0; // (not a smaller PC)
	s.sim_commit_limit = EnvMiB("KYTY_SIMULATE_COMMIT_MB", 0);
	if (s.sim_commit_limit == 0 && s.sim_ram != 0) s.sim_commit_limit = s.sim_ram + EnvMiB("KYTY_SIMULATE_PAGEFILE_MB", s.sim_ram);
	s.sim_os        = EnvMiB("KYTY_SIMULATE_OS_MB", 3 * GiB);
	s.sim_os_commit = EnvMiB("KYTY_SIMULATE_OS_COMMIT_MB", 4 * GiB);
	const uint64_t ram = s.sim_ram != 0 ? s.sim_ram : real_ram;
	s.low_avail  = EnvMiB("KYTY_RAM_LOW_AVAIL_MB", std::max<uint64_t>(GiB, ram / 16));
	s.low_commit = EnvMiB("KYTY_RAM_LOW_COMMIT_MB", 2 * GiB);
	{
		const char* abort_text = Env("KYTY_SIMULATE_RAM_ABORT");
		s.sim_abort            = s.sim_commit_limit != 0 && !(abort_text != nullptr && abort_text[0] == '0');
		s.flush_seconds        = FlushSeconds();
		s.flush_bytes          = std::max<uint64_t>(MiB, EnvMiB("KYTY_RAM_FLUSH_MB", 1024 * MiB));
		const char* trim       = Env("KYTY_RAM_FLUSH_TRIM");
		s.flush_trim           = trim != nullptr && trim[0] == '1';
	}
	return s;
}

Sample Take(const Settings& s, HANDLE low_memory, bool walk_store, uint64_t last_store) {
	Sample out;
	const auto use    = LocalPlatform::ProcessMemory();
	out.private_bytes = use.private_bytes;
	out.working_set   = use.working_set;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (GlobalMemoryStatusEx(&status) != 0) {
		out.total_phys     = status.ullTotalPhys;
		out.available      = status.ullAvailPhys;
		out.real_available = status.ullAvailPhys;
		// (ullTotalPageFile/ullAvailPageFile: the commit limit and what is left of it.)
		out.commit_limit  = status.ullTotalPageFile;
		out.commit_total  = status.ullTotalPageFile - std::min(status.ullTotalPageFile, status.ullAvailPageFile);
		out.real_headroom = status.ullAvailPageFile;
	}
	out.store_committed = walk_store ? StoreCommitted() : last_store;
	if (s.sim_ram != 0) {
		out.simulated  = true;
		out.total_phys = s.sim_ram;
		const uint64_t used = s.sim_os + out.working_set;
		out.available  = std::min(out.real_available, s.sim_ram > used ? s.sim_ram - used : 0);
	}
	if (s.sim_commit_limit != 0) {
		out.simulated = true;
		// (Not the KYTY_SIMULATE_VRAM_MB ballast's driver commit: 7.8 GB of the private bytes in a simulated 8 GB card
		// run, 10-24, which a real 8 GB card does not have.)
		const uint64_t ballast  = LocalPlatform::SimulationBallast();
		const uint64_t charged  = s.sim_os_commit + (out.private_bytes - std::min(out.private_bytes, ballast)) + out.store_committed;
		const uint64_t headroom = std::min(out.real_headroom, s.sim_commit_limit > charged ? s.sim_commit_limit - charged : 0);
		out.commit_limit        = s.sim_commit_limit;
		out.commit_total        = s.sim_commit_limit - std::min(s.sim_commit_limit, headroom);
	}
	if (low_memory != nullptr) {
		BOOL state = FALSE;
		out.signal = QueryMemoryResourceNotification(low_memory, &state) != 0 && state != FALSE;
	}
	return out;
}

int LevelOf(const Settings& s, const Sample& sample) {
	const uint64_t headroom = sample.Headroom();
	const bool     known    = sample.total_phys != 0;
	if (!known) return None;
	if (sample.available < s.low_avail / 2 || (sample.commit_limit != 0 && headroom < s.low_commit / 2)) return Critical;
	if (sample.available < s.low_avail || (sample.commit_limit != 0 && headroom < s.low_commit) || sample.signal) return Low;
	return None;
}

std::string Figures(const Sample& sample) {
	char       ballast_text[96] = "";
	if (const uint64_t ballast = LocalPlatform::SimulationBallast(); ballast != 0)
		std::snprintf(ballast_text, sizeof(ballast_text), " (%llu MiB of the private: the KYTY_SIMULATE_VRAM_MB ballast, not counted)",
		              static_cast<unsigned long long>(ballast / MiB));
	char line[640];
	std::snprintf(line, sizeof(line),
	              "available %llu of %llu MiB, commit %llu of %llu MiB (headroom %llu MiB)%s%s; emulator private %llu MiB, "
	              "working set %llu MiB, guest store committed %llu MiB%s",
	              static_cast<unsigned long long>(sample.available / MiB),
	              static_cast<unsigned long long>(sample.total_phys / MiB),
	              static_cast<unsigned long long>(sample.commit_total / MiB),
	              static_cast<unsigned long long>(sample.commit_limit / MiB),
	              static_cast<unsigned long long>(sample.Headroom() / MiB), sample.signal ? ", Windows low-memory signal" : "",
	              sample.simulated ? " (simulated: KYTY_SIMULATE_RAM_MB / KYTY_SIMULATE_COMMIT_MB)" : "",
	              static_cast<unsigned long long>(sample.private_bytes / MiB),
	              static_cast<unsigned long long>(sample.working_set / MiB),
	              static_cast<unsigned long long>(sample.store_committed / MiB), ballast_text);
	return line;
}

// What owners released since the last line (pipelines, images), and clears it.
std::string TakeReleased(uint64_t* bytes) {
	std::string out;
	static const char* const names[ReleasedKinds] = {"idle pipelines", "unused images (video memory)"};
	for (int kind = 0; kind < ReleasedKinds; kind++) {
		const uint64_t count = g_released_count[kind].exchange(0, std::memory_order_relaxed);
		const uint64_t b     = g_released_bytes[kind].exchange(0, std::memory_order_relaxed);
		if (count == 0) continue;
		*bytes += b;
		char piece[128];
		std::snprintf(piece, sizeof(piece), ", %s %llu (~%llu MiB)", names[kind], static_cast<unsigned long long>(count),
		              static_cast<unsigned long long>(b / MiB));
		out += piece;
	}
	return out;
}

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// The simulated PC ran out of commit: the run ends as it would have there (an allocation refused, the game closed).
[[noreturn]] void SimulatedOutOfMemory(const Settings& s, const Sample& sample) {
	char pc[64];
	if (s.sim_ram != 0)
		std::snprintf(pc, sizeof(pc), "un PC de %llu GB", static_cast<unsigned long long>((s.sim_ram + GiB / 2) / GiB));
	else
		std::snprintf(pc, sizeof(pc), "un PC con %llu MiB de commit", static_cast<unsigned long long>(s.sim_commit_limit / MiB));
	std::printf("RAM simulada agotada: en %s el juego se habria cerrado (simulated RAM used up: on that PC the game "
	            "would have closed; KYTY_SIMULATE_RAM_ABORT=0 keeps running on the real memory). %s\n",
	            pc, Figures(sample).c_str());
	std::fflush(stdout);
	std::fflush(stderr);
	// (No exit handlers: the game's threads are running; STATUS_NO_MEMORY, as a refused allocation's end would be.)
	TerminateProcess(GetCurrentProcess(), 0xC0000017u);
	for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}

struct FlushState {
	uint64_t passes = 0;
	uint64_t released_total[FlushClasses] {};
	uint64_t owners_total = 0;
	double   ms_total = 0.0, ms_max = 0.0;
	double   heaps_ms_last = 0.0;
	Clock::time_point heaps_last {};
};

// One periodic flush pass (ram-guard.h): the sources by class, in order, until s.flush_bytes are released.
void FlushPass(const Settings& s, HANDLE low_memory, uint64_t store, FlushState* state) {
	const auto   start  = Clock::now();
	const Sample before = Take(s, low_memory, false, store);
	std::vector<BudgetFlush> list;
	{
		std::lock_guard lock(g_flush_mutex);
		list = g_budget_flushes;
	}
	uint64_t    remaining = s.flush_bytes;
	uint64_t    released[FlushClasses] {};
	uint64_t    held_us = 0;
	std::string parts[FlushClasses];
	const auto  add_part = [&](int kind, const char* name, uint64_t bytes, double ms) {
		char piece[160];
		std::snprintf(piece, sizeof(piece), "%s%s %.1f MiB %.1f ms", parts[kind].empty() ? "" : ", ", name,
		              static_cast<double>(bytes) / 1048576.0, ms);
		parts[kind] += piece;
	};
	for (int kind = 0; kind < FlushClasses; kind++) {
		if (kind == FlushTrim && !s.flush_trim) break;
		const auto use_before = LocalPlatform::ProcessMemory();
		uint64_t   reported   = 0;
		for (const auto& source: list) {
			if (source.kind != kind || remaining <= reported) continue;
			uint64_t   held = 0;
			const auto t0   = Clock::now();
			const auto b    = source.fn(remaining - reported, &held);
			held_us         = std::max(held_us, held);
			reported += b;
			add_part(kind, source.name, b, MsSince(t0));
		}
		if (kind == FlushCaches && remaining > reported) {
			// Idle scratch buffers and the heaps' free blocks (soft-release.h). The compaction holds each heap's lock: when
			// it took more than 5 ms, it runs at most every 30 s.
			const auto t0 = Clock::now();
			if (state->heaps_ms_last <= 5.0 || t0 - state->heaps_last >= std::chrono::seconds(30)) {
				char detail[192];
				(void)SoftRelease::FlushNow(10'000, detail, sizeof(detail));
				state->heaps_ms_last = MsSince(t0);
				state->heaps_last    = t0;
				held_us              = std::max(held_us, static_cast<uint64_t>(state->heaps_ms_last * 1000.0));
				char piece[256];
				std::snprintf(piece, sizeof(piece), "%s%s %.1f ms", parts[kind].empty() ? "" : ", ", detail, state->heaps_ms_last);
				parts[kind] += piece;
			} else {
				parts[kind] += parts[kind].empty() ? "heaps skipped (slow last time)" : ", heaps skipped (slow last time)";
			}
		}
		// Caches: what the commit dropped (or the sources said); guest memory and the trim: what the working set dropped
		// (their commit stays).
		const auto use_after = LocalPlatform::ProcessMemory();
		const auto dropped   = kind == FlushCaches ? (use_before.private_bytes > use_after.private_bytes
		                                                  ? use_before.private_bytes - use_after.private_bytes : 0)
		                                           : (use_before.working_set > use_after.working_set
		                                                  ? use_before.working_set - use_after.working_set : 0);
		released[kind] = kind == FlushCaches ? std::max(reported, dropped) : dropped;
		remaining -= std::min(remaining, std::max(released[kind], kind == FlushCaches ? 0 : reported));
		state->released_total[kind] += released[kind];
		if (remaining == 0) break;
	}
	const double ms    = MsSince(start);
	const Sample after = Take(s, low_memory, false, store);
	state->passes++;
	state->ms_total += ms;
	state->ms_max = std::max(state->ms_max, ms);
	// What the owners released on their own threads since the last pass (pipelines idle 180 s with KYTY_RAM_DIET, images
	// the collector dropped for the flush): their commit went between the passes, not inside this one.
	uint64_t    owners_bytes = 0;
	std::string owners       = TakeReleased(&owners_bytes);
	state->owners_total += owners_bytes;
	const auto mb = [](uint64_t b) { return static_cast<unsigned long long>(b / 1000000u); };
	const auto mi = [](uint64_t b) { return static_cast<unsigned long long>(b / MiB); };
	// The system's figures as the overlay shows them ("RAM x / y GB", "Commit x / y GB"): RAM in use = total less
	// available; the commit charge of the whole PC. The process's own: private (its commit) and working set (its RAM).
	const uint64_t sys_before = before.total_phys - std::min(before.total_phys, before.available);
	const uint64_t sys_after  = after.total_phys - std::min(after.total_phys, after.available);
	std::printf("RAM flush #%llu (every %lld s, up to %llu MB, level %d): released %llu MB: caches %llu MB [%s], freed guest "
	            "memory %llu MB of RAM [%s], trim %s%llu MB%s%s%s; owners since the last pass %llu MB%s; %.1f ms (longest "
	            "lock a game thread could wait on %.2f ms); process: private %llu -> %llu MiB, working set %llu -> %llu MiB; "
	            "PC: RAM in use %llu -> %llu of %llu MiB, commit %llu -> %llu of %llu MiB%s\n",
	            static_cast<unsigned long long>(state->passes), static_cast<long long>(s.flush_seconds), mb(s.flush_bytes),
	            FlushLevel(), mb(released[0] + released[1] + released[2]), mb(released[0]), parts[0].c_str(), mb(released[1]),
	            parts[1].empty() ? "none" : parts[1].c_str(), s.flush_trim ? "" : "off ", mb(released[2]),
	            s.flush_trim ? " [" : "", s.flush_trim ? (parts[2].empty() ? "none" : parts[2].c_str()) : "",
	            s.flush_trim ? "] (no commit change)" : "", mb(owners_bytes), owners.c_str(), ms,
	            static_cast<double>(held_us) / 1000.0, mi(before.private_bytes), mi(after.private_bytes),
	            mi(before.working_set), mi(after.working_set), mi(sys_before), mi(sys_after), mi(after.total_phys),
	            mi(before.commit_total), mi(after.commit_total), mi(after.commit_limit), after.simulated ? " (simulated)" : "");
	// The census: where the memory the flush cannot touch is (the owners' figures), with the first pass and then every
	// 12th (a minute at the launcher's 5 s).
	if (state->passes == 1 || state->passes % 12 == 0) {
		std::vector<std::pair<const char*, uint64_t (*)()>> figures;
		{
			std::lock_guard lock(g_flush_mutex);
			figures = g_figures;
		}
		std::string line = "RAM census (what is held; the flush returns only what is free or re-creatable):";
		char        piece[160];
		std::snprintf(piece, sizeof(piece), " guest store committed %llu of %llu MiB,", mi(after.store_committed),
		              mi(g_store_size.load(std::memory_order_relaxed)));
		line += piece;
		for (const auto& [name, figure]: figures) {
			std::snprintf(piece, sizeof(piece), " %s %llu MiB,", name, mi(figure()));
			line += piece;
		}
		std::snprintf(piece, sizeof(piece), " flush passes %llu: caches %llu MB, freed guest memory %llu MB, owners %llu MB in all",
		              static_cast<unsigned long long>(state->passes), mb(state->released_total[0]), mb(state->released_total[1]),
		              mb(state->owners_total));
		line += piece;
		std::printf("%s\n", line.c_str());
	}
	std::fflush(stdout);
}

// The RAM governor (ram-guard.h, KYTY_RAM_TARGET_MB): its thresholds and state, on the guard's thread.
struct Governor {
	bool              on = false;
	uint64_t          target = 0, soft = 0, emergency = 0, hysteresis = 256 * MiB;
	int               stage = GovIdle;
	Clock::time_point next_pass {};
	uint64_t          passes = 0, entered[4] {};
};

Governor MakeGovernor(const Settings& s, const Sample& first) {
	Governor g;
	g.on = GovernorOn();
	if (!g.on) return g;
	const uint64_t total = first.total_phys != 0 ? first.total_phys : (s.sim_ram != 0 ? s.sim_ram : 16 * GiB);
	g.target    = EnvMiB("KYTY_RAM_TARGET_MB", 0);
	g.target    = std::min(g.target, total > 1024 * MiB ? total - 1024 * MiB : total / 2);
	g.soft      = std::min(g.target, EnvMiB("KYTY_RAM_SOFT_MB", g.target > 512 * MiB ? g.target - 512 * MiB : g.target));
	g.emergency = std::max(g.target, EnvMiB("KYTY_RAM_EMERGENCY_MB", g.target + 512 * MiB));
	g.emergency = std::min(g.emergency, total > 512 * MiB ? total - 512 * MiB : total); // (the hard ceiling below the total)
	g.emergency = std::max(g.emergency, g.target);
	std::printf("RAM target: soft %llu MiB, target %llu MiB, emergency %llu MiB, of %llu MiB%s; below the soft threshold "
	            "nothing extra runs; above it: flush pass every 2 s (1 s over target), idle pipelines 60/30/15 s, texture "
	            "offload of the longest-idle verified textures for what is over (hot window %s s)\n",
	            static_cast<unsigned long long>(g.soft / MiB), static_cast<unsigned long long>(g.target / MiB),
	            static_cast<unsigned long long>(g.emergency / MiB), static_cast<unsigned long long>(total / MiB),
	            first.simulated ? " (simulated)" : "", Env("KYTY_RAM_TARGET_HOT_SECONDS") != nullptr ? Env("KYTY_RAM_TARGET_HOT_SECONDS") : "10");
	std::fflush(stdout);
	return g;
}

const char* GovStageName(int stage) {
	return stage == GovEmergency ? "emergency" : stage == GovOver ? "over target" : stage == GovSoft ? "soft" : "under";
}

uint64_t InUse(const Sample& sample) {
	return sample.total_phys - std::min(sample.total_phys, sample.available);
}

// Every second: the stage from the RAM in use (down a stage only 256 MiB below its threshold), the offload demand, and
// above the soft threshold a flush pass (every 2 s, 1 s from the target up) with its "RAM target:" line.
void GovernorStep(const Settings& s, HANDLE low_memory, uint64_t store, const Sample& sample, FlushState* flush_state,
                  Governor* g, Clock::time_point now) {
	const uint64_t in_use = InUse(sample);
	const auto     threshold = [g](int stage) {
        return stage >= GovEmergency ? g->emergency : stage == GovOver ? g->target : g->soft;
	};
	const int raw   = in_use >= g->emergency ? GovEmergency : in_use >= g->target ? GovOver : in_use >= g->soft ? GovSoft : GovIdle;
	int       stage = g->stage;
	if (raw > stage) stage = raw;
	else if (raw < stage && in_use + g->hysteresis < threshold(stage)) stage = std::max(raw, stage - 1);
	const int before_stage = g->stage;
	if (stage != g->stage) {
		g->entered[stage]++;
		std::printf("RAM target: stage %s -> %s, in use %llu MiB (soft %llu, target %llu, emergency %llu)%s\n",
		            GovStageName(g->stage), GovStageName(stage), static_cast<unsigned long long>(in_use / MiB),
		            static_cast<unsigned long long>(g->soft / MiB), static_cast<unsigned long long>(g->target / MiB),
		            static_cast<unsigned long long>(g->emergency / MiB), sample.simulated ? " (simulated)" : "");
		std::fflush(stdout);
		g->stage = stage;
		if (stage > before_stage) g->next_pass = now; // (escalation acts at once)
	}
	g_gov_stage.store(stage, std::memory_order_relaxed);
	const uint64_t goal = g->soft > g->hysteresis ? g->soft - g->hysteresis : 0;
	g_gov_demand.store(stage >= GovSoft && in_use > goal ? in_use - goal : 0, std::memory_order_relaxed);
	if (stage >= GovEmergency) g_gov_urgent.store(true, std::memory_order_release);
	if (stage == GovIdle) {
		g_gov_taken.store(0, std::memory_order_relaxed);
		return;
	}
	if (now < g->next_pass) return;
	g->next_pass = now + std::chrono::seconds(stage == GovSoft ? 2 : 1);
	g->passes++;
	const uint64_t taken       = g_gov_taken.exchange(0, std::memory_order_relaxed);
	const uint64_t owners_prev = flush_state->owners_total;
	const uint64_t caches_prev = flush_state->released_total[FlushCaches] + flush_state->released_total[FlushGuestFreed];
	FlushPass(s, low_memory, store, flush_state);
	const Sample   after = Take(s, low_memory, false, store);
	const uint64_t flushed =
	    flush_state->released_total[FlushCaches] + flush_state->released_total[FlushGuestFreed] - caches_prev;
	std::printf("RAM target #%llu: in use %llu -> %llu MiB vs target %llu (soft %llu, emergency %llu), stage %s; actions: "
	            "flush pass %llu MiB, pipelines idle %d s or more out (%llu MiB since the last pass), texture offload "
	            "wanted %llu MiB, taken since the last pass %llu MiB (%llu MiB in all)%s\n",
	            static_cast<unsigned long long>(g->passes), static_cast<unsigned long long>(in_use / MiB),
	            static_cast<unsigned long long>(InUse(after) / MiB), static_cast<unsigned long long>(g->target / MiB),
	            static_cast<unsigned long long>(g->soft / MiB), static_cast<unsigned long long>(g->emergency / MiB),
	            GovStageName(stage), static_cast<unsigned long long>(flushed / MiB),
	            stage >= GovEmergency ? 15 : stage == GovOver ? 30 : 60,
	            static_cast<unsigned long long>((flush_state->owners_total - owners_prev) / MiB),
	            static_cast<unsigned long long>(g_gov_demand.load(std::memory_order_relaxed) / MiB),
	            static_cast<unsigned long long>(taken / MiB),
	            static_cast<unsigned long long>(g_gov_taken_total.load(std::memory_order_relaxed) / MiB),
	            after.simulated ? " (simulated)" : "");
	std::fflush(stdout);
}

void Watch() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.RamGuard");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	SoftRelease::LowerThreadMemoryPriority();
	const Settings s          = ReadSettings();
	const char*    log_text   = Env("KYTY_RAM_LOG_SECONDS");
	const int64_t  log_period = log_text != nullptr ? std::strtoll(log_text, nullptr, 10) : 30;
	HANDLE         low_memory = CreateMemoryResourceNotification(LowMemoryResourceNotification);
	std::printf("RAM guard: on (KYTY_RAM_GUARD=0 turns it off): pressure below %llu MiB available or %llu MiB of commit "
	            "headroom (critical at half)%s; low-RAM profile %s; a line every %lld s\n",
	            static_cast<unsigned long long>(s.low_avail / MiB), static_cast<unsigned long long>(s.low_commit / MiB),
	            s.sim_ram != 0 || s.sim_commit_limit != 0 ? ", simulated PC" : "", LowRam() ? "on" : "off",
	            static_cast<long long>(log_period));
	if (s.sim_ram != 0 || s.sim_commit_limit != 0)
		std::printf("RAM guard: simulating %llu MiB of RAM (%llu MiB for Windows) and a %llu MiB commit limit (%llu MiB for "
		            "Windows)\n",
		            static_cast<unsigned long long>(s.sim_ram / MiB), static_cast<unsigned long long>(s.sim_os / MiB),
		            static_cast<unsigned long long>(s.sim_commit_limit / MiB),
		            static_cast<unsigned long long>(s.sim_os_commit / MiB));
	if (s.sim_abort)
		std::printf("RAM guard: the simulated PC's end is simulated too: with no commit headroom left the run is closed "
		            "(\"RAM simulada agotada\"; KYTY_SIMULATE_RAM_ABORT=0 keeps running)\n");
	if (s.flush_seconds > 0)
		std::printf("RAM flush (vaciado): every %lld s up to %llu MB, level %d: caches (re-creatable), guest memory the game "
		            "released%s (KYTY_RAM_FLUSH_SECONDS / KYTY_RAM_FLUSH_MB / KYTY_RAM_FLUSH_LEVEL / KYTY_RAM_FLUSH_TRIM); turned "
		            "on with it: %s%s\n",
		            static_cast<long long>(s.flush_seconds), static_cast<unsigned long long>(s.flush_bytes / 1000000u),
		            FlushLevel(), s.flush_trim ? ", then the working set trim of cold guest pages" : "; no working set trim",
		            g_flush_defaults.empty() ? "nothing (every switch already set)" : g_flush_defaults.c_str(),
		            FlushImageIdleSeconds() != 0 ? "; images no draw used for KYTY_RAM_FLUSH_IMAGE_IDLE_SECONDS leave video memory"
		                                         : "");
	std::fflush(stdout);
	g_periodic.store(s.flush_seconds > 0, std::memory_order_relaxed);
	FlushState flush_state;
	auto       next_periodic = Clock::now() + std::chrono::seconds(std::max<int64_t>(s.flush_seconds, 1));
	uint64_t   seconds_under_1g = 0, seconds_under_512 = 0;
	Governor   governor = MakeGovernor(s, Take(s, low_memory, false, 0));
	int        sim_exhausted = 0;
	auto     next_log   = Clock::now() + std::chrono::seconds(std::max<int64_t>(log_period, 1));
	auto     next_flush = Clock::now();
	auto     next_walk  = Clock::now();
	int      backoff    = 0;
	int      logged     = None;
	uint64_t store      = 0, flushes = 0, freed_total = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		const auto now  = Clock::now();
		const bool walk = now >= next_walk;
		if (walk) next_walk = now + std::chrono::seconds(5);
		const Sample sample = Take(s, low_memory, walk, store);
		store               = sample.store_committed;
		g_available.store(sample.available, std::memory_order_relaxed);
		// (The simulated commit used up: as on that PC, a release pass first - what MemGuard does when an allocation is
		// refused - and the end only when a second later there is still none.)
		if (s.sim_abort && sample.commit_limit != 0 && sample.Headroom() == 0) {
			if (sim_exhausted++ != 0) SimulatedOutOfMemory(s, sample);
			std::printf("RAM simulada: sin commit libre (no simulated commit headroom left): a release pass, the run ends if "
			            "there is still none; %s\n",
			            Figures(sample).c_str());
			std::fflush(stdout);
			(void)MemGuard::ReleaseNow("simulated commit used up", 500);
		} else {
			sim_exhausted = 0;
		}
		if (s.sim_ram != 0) {
			seconds_under_1g += sample.available < GiB ? 1 : 0;
			seconds_under_512 += sample.available < GiB / 2 ? 1 : 0;
		}
		const int level = LevelOf(s, sample);
		g_level.store(level, std::memory_order_relaxed);
		if (governor.on) GovernorStep(s, low_memory, store, sample, &flush_state, &governor, now);
		if (governor.on && governor.stage != GovIdle) {
			next_periodic = Clock::now() + std::chrono::seconds(std::max<int64_t>(s.flush_seconds, 1)); // (its passes run)
		} else if (s.flush_seconds > 0 && now >= next_periodic) {
			FlushPass(s, low_memory, store, &flush_state);
			next_periodic = Clock::now() + std::chrono::seconds(s.flush_seconds);
		}
		if (level != logged && level == None) {
			uint64_t    bytes    = 0;
			const auto  released = TakeReleased(&bytes);
			std::printf("RAM: pressure %s -> none; %s%s\n", LevelName(logged), Figures(sample).c_str(), released.c_str());
			std::fflush(stdout);
			backoff = 0;
		}
		logged = level;
		if (level != None && now >= next_flush) {
			// The safe flush: only free or re-creatable memory (soft-release.h's pass and the registered sources); the
			// owners of pipelines and images release theirs on their own threads while the level stays up.
			char           detail[256];
			const uint64_t before = sample.private_bytes;
			uint64_t       freed  = SoftRelease::FlushNow(level == Critical ? 500 : 2000, detail, sizeof(detail));
			std::string    sources;
			{
				std::vector<std::pair<const char*, uint64_t (*)(int)>> list;
				{
					std::lock_guard lock(g_flush_mutex);
					list = g_flushes;
				}
				for (const auto& [name, flush]: list) {
					const uint64_t bytes = flush(level);
					if (bytes == 0) continue;
					char piece[128];
					std::snprintf(piece, sizeof(piece), ", %s %.1f MiB", name, static_cast<double>(bytes) / 1048576.0);
					sources += piece;
					freed += bytes;
				}
			}
			uint64_t   owners   = 0;
			const auto released = TakeReleased(&owners);
			freed += owners;
			freed_total += freed;
			flushes++;
			const auto after = LocalPlatform::ProcessMemory().private_bytes;
			std::printf("RAM: pressure %s (%s), freed %llu MB (%s%s%s; private %llu -> %llu MiB); %llu passes, %llu MB in all\n",
			            LevelName(level), Figures(sample).c_str(), static_cast<unsigned long long>(freed / 1000000u), detail,
			            sources.c_str(), released.c_str(), static_cast<unsigned long long>(before / MiB),
			            static_cast<unsigned long long>(after / MiB), static_cast<unsigned long long>(flushes),
			            static_cast<unsigned long long>(freed_total / 1000000u));
			std::fflush(stdout);
			// Again in 5 s (critical) or 15 s (low); a pass that freed little waits twice as long, up to 2 minutes:
			// nothing left to give is not looked for every few seconds.
			backoff = freed < 32 * MiB ? std::min(backoff + 1, 4) : 0;
			const int base = level == Critical ? 5 : 15;
			next_flush     = now + std::chrono::seconds(std::min(120, base << backoff));
		}
		if (log_period > 0 && now >= next_log) {
			next_log = now + std::chrono::seconds(log_period);
			std::printf("RAM (%lld s): %s; pressure %s, real available %llu MiB, real commit headroom %llu MiB; pressure "
			            "passes %llu (%llu MB), pipelines released %llu, images collected %llu (%llu MiB)\n",
			            static_cast<long long>(log_period), Figures(sample).c_str(), LevelName(level),
			            static_cast<unsigned long long>(sample.real_available / MiB),
			            static_cast<unsigned long long>(sample.real_headroom / MiB), static_cast<unsigned long long>(flushes),
			            static_cast<unsigned long long>(freed_total / 1000000u),
			            static_cast<unsigned long long>(g_released_count_total[Pipelines].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_count_total[Images].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_bytes_total[Images].load(std::memory_order_relaxed) / MiB));
			if (s.flush_seconds > 0 || s.sim_ram != 0) {
				std::printf("RAM (%lld s): flush passes %llu (caches %llu MB, freed guest memory %llu MB, trim %llu MB; %.1f ms "
				            "a pass on average, at most %.1f ms); simulated PC under 1 GiB available %llu s, under 512 MiB "
				            "%llu s (a real one pages there: stutter)\n",
				            static_cast<long long>(log_period), static_cast<unsigned long long>(flush_state.passes),
				            static_cast<unsigned long long>(flush_state.released_total[0] / 1000000u),
				            static_cast<unsigned long long>(flush_state.released_total[1] / 1000000u),
				            static_cast<unsigned long long>(flush_state.released_total[2] / 1000000u),
				            flush_state.passes != 0 ? flush_state.ms_total / static_cast<double>(flush_state.passes) : 0.0,
				            flush_state.ms_max, static_cast<unsigned long long>(seconds_under_1g),
				            static_cast<unsigned long long>(seconds_under_512));
			}
			std::fflush(stdout);
		}
	}
}

} // namespace
#endif

void Start() {
#if defined(_WIN32)
	static std::once_flag once;
	std::call_once(once, [] {
		if (const char* value = Env("KYTY_RAM_GUARD"); value != nullptr && value[0] == '0') {
			std::printf("RAM guard: off (KYTY_RAM_GUARD=0)\n");
			std::fflush(stdout);
			return;
		}
		std::thread(Watch).detach();
	});
#endif
}

} // namespace RamGuard
