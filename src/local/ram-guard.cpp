#include "ram-guard.h"

#include "fresh-memory.h"
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
#include <psapi.h>
#endif

namespace RamGuard {

namespace {

constexpr uint64_t MiB = uint64_t {1} << 20u;
constexpr uint64_t GiB = uint64_t {1} << 30u;

std::atomic<int>      g_level {None};
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

void ApplyLowRamDefaults() {
	static std::once_flag once;
	std::call_once(once, [] {
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
		SetDefault("KYTY_PREFETCH_MIN_AVAIL_MB", "3072", &set);
		// Clean sampled textures' guest bytes out of RAM (their images give them back), and the levels KYTY_TEXTURE_REDUCE
		// leaves out of large textures (no image holds them) into a scratch file: texture-offload.h.
		SetDefault("KYTY_TEXTURE_OFFLOAD", "1", &set);
		SetDefault("KYTY_TEXTURE_SPILL", "1", &set);
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
		const char* seconds    = Env("KYTY_RAM_FLUSH_SECONDS");
		s.flush_seconds        = seconds != nullptr ? std::max<int64_t>(0, std::strtoll(seconds, nullptr, 10)) : 0;
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
	const auto mb = [](uint64_t b) { return static_cast<unsigned long long>(b / 1000000u); };
	const auto mi = [](uint64_t b) { return static_cast<unsigned long long>(b / MiB); };
	std::printf("RAM flush #%llu (every %lld s, up to %llu MB): released %llu MB: caches %llu MB [%s], freed guest memory "
	            "%llu MB of RAM [%s], trim %s%llu MB%s%s%s; %.1f ms (longest lock a game thread could wait on %.2f ms); "
	            "private %llu -> %llu MiB, working set %llu -> %llu MiB, commit %llu -> %llu of %llu MiB, available %llu -> "
	            "%llu MiB%s\n",
	            static_cast<unsigned long long>(state->passes), static_cast<long long>(s.flush_seconds), mb(s.flush_bytes),
	            mb(released[0] + released[1] + released[2]), mb(released[0]), parts[0].c_str(), mb(released[1]),
	            parts[1].empty() ? "none" : parts[1].c_str(), s.flush_trim ? "" : "off ", mb(released[2]),
	            s.flush_trim ? " [" : "", s.flush_trim ? (parts[2].empty() ? "none" : parts[2].c_str()) : "",
	            s.flush_trim ? "] (no commit change)" : "", ms, static_cast<double>(held_us) / 1000.0,
	            mi(before.private_bytes), mi(after.private_bytes), mi(before.working_set), mi(after.working_set),
	            mi(before.commit_total), mi(after.commit_total), mi(after.commit_limit), mi(before.available),
	            mi(after.available), after.simulated ? " (simulated)" : "");
	std::fflush(stdout);
}

// KYTY_WS_BREAKDOWN (ram-guard.h): the working set by kind, from QueryWorkingSetEx on one page in every `stride`
// bytes of the committed address space (an estimate: each sampled page stands for `stride` bytes), in batches of
// 2048 pages with the thread yielding between them (each call holds the working set lock over its pages only).
// Guest memory is counted at both of its addresses: the game's views (a sampled page there is also looked up at its
// store alias address, so pages resident at both - counted twice in the working set, once in RAM - are known) and the
// emulator's alias; "never written" is the fresh-memory granules' (zero pages someone read).
struct WsCategory {
	uint64_t committed = 0, resident = 0, fresh = 0, both = 0;
};
enum WsKind : int {
	WsGuestView = 0,
	WsAlias,
	WsOtherMapped,
	WsHeap,
	WsExec,
	WsRwLarge,
	WsRwMid,
	WsRwSmall,
	WsPrivateOther,
	WsImage,
	WsKinds
};

bool WsBreakdownWanted(const Settings& s) {
	if (const char* value = Env("KYTY_WS_BREAKDOWN"); value != nullptr) return value[0] == '1';
	return LowRam() || s.sim_ram != 0 || s.sim_commit_limit != 0;
}

void WsBreakdown() {
	static const uint64_t stride = [] {
		const char*    text = Env("KYTY_WS_BREAKDOWN_STRIDE_KB");
		const uint64_t kb   = text != nullptr ? std::strtoull(text, nullptr, 10) : 64;
		return (std::clamp<uint64_t>(kb, 4, 1024) * 1024u) & ~uint64_t {4095};
	}();
	const auto start = Clock::now();
	struct Region {
		uint64_t base, size, alloc;
		DWORD    type, protect;
	};
	std::vector<Region> regions;
	regions.reserve(65536);
	std::vector<std::pair<uint64_t, uint64_t>> heaps; // (allocation bases of the process heaps)
	{
		HANDLE      handles[256];
		const DWORD n = GetProcessHeaps(256, handles);
		for (DWORD i = 0; i < n && i < 256; i++) {
			MEMORY_BASIC_INFORMATION h {};
			if (VirtualQuery(handles[i], &h, sizeof(h)) == sizeof(h)) heaps.emplace_back(reinterpret_cast<uint64_t>(h.AllocationBase), 0);
		}
	}
	MEMORY_BASIC_INFORMATION info {};
	for (uint64_t at = 0x10000; VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == sizeof(info);) {
		const uint64_t base = reinterpret_cast<uint64_t>(info.BaseAddress);
		// (No-access pages too: the trackers' watched pages stay resident.)
		if (info.State == MEM_COMMIT)
			regions.push_back({base, info.RegionSize, reinterpret_cast<uint64_t>(info.AllocationBase), info.Type, info.Protect});
		const uint64_t next = base + info.RegionSize;
		if (next <= at) break;
		at = next;
	}
	// Committed private bytes per allocation (its size class) and the store alias's range.
	std::vector<std::pair<uint64_t, uint64_t>> alloc_committed; // base, committed
	for (const auto& r: regions) {
		if (r.type != MEM_PRIVATE) continue;
		if (alloc_committed.empty() || alloc_committed.back().first != r.alloc) alloc_committed.emplace_back(r.alloc, 0);
		alloc_committed.back().second += r.size;
	}
	const auto committed_of = [&](uint64_t alloc) {
		auto it = std::lower_bound(alloc_committed.begin(), alloc_committed.end(), std::pair<uint64_t, uint64_t> {alloc, 0});
		return it != alloc_committed.end() && it->first == alloc ? it->second : 0;
	};
	std::sort(alloc_committed.begin(), alloc_committed.end());
	const auto alias      = reinterpret_cast<uint64_t>(g_store_alias.load(std::memory_order_relaxed));
	const auto alias_size = g_store_size.load(std::memory_order_relaxed);
	auto*      fresh      = FreshMemory::Get();
	const auto is_fresh   = [&](uint64_t backing) {
        if (fresh == nullptr) return false;
        const auto granule = FreshMemory::GranuleOf(*fresh, backing);
        return granule != UINT64_MAX && !FreshMemory::IsWritten(*fresh, granule);
	};
	WsCategory                       cat[WsKinds] {};
	std::vector<std::pair<uint64_t, uint64_t>> private_resident; // allocation base, resident (sampled)
	// One batch: entries and what each is (kind; for a guest-view page, the index of its alias twin, or -1).
	constexpr size_t                          Batch = 2048;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries;
	struct Tag {
		int      kind;
		int32_t  twin;   // (guest view: the alias entry's index; -1 none)
		bool     fresh;
		uint64_t alloc;  // (private: its allocation base)
	};
	std::vector<Tag> tags;
	entries.reserve(Batch + 2);
	tags.reserve(Batch + 2);
	uint64_t calls = 0;
	const auto flush = [&] {
		if (entries.empty()) return;
		if (QueryWorkingSetEx(GetCurrentProcess(), entries.data(), static_cast<DWORD>(entries.size() * sizeof(entries[0]))) != 0) {
			for (size_t i = 0; i < entries.size(); i++) {
				const auto& tag = tags[i];
				if (tag.kind < 0) continue; // (an alias twin: counted through its guest page)
				const bool valid = entries[i].VirtualAttributes.Valid != 0;
				if (!valid) continue;
				auto& c = cat[tag.kind];
				c.resident += stride;
				if (tag.fresh) c.fresh += stride;
				if (tag.twin >= 0 && entries[static_cast<size_t>(tag.twin)].VirtualAttributes.Valid != 0) c.both += stride;
				if (tag.kind >= WsHeap && tag.kind <= WsPrivateOther) {
					if (private_resident.empty() || private_resident.back().first != tag.alloc) private_resident.emplace_back(tag.alloc, 0);
					private_resident.back().second += stride;
				}
			}
		}
		calls++;
		entries.clear();
		tags.clear();
		SwitchToThread();
	};
	const auto add = [&](uint64_t address, Tag tag) {
		PSAPI_WORKING_SET_EX_INFORMATION e {};
		e.VirtualAddress = reinterpret_cast<void*>(address);
		entries.push_back(e);
		tags.push_back(tag);
	};
	for (const auto& r: regions) {
		int kind = WsImage;
		if (r.type == MEM_IMAGE) kind = WsImage;
		else if (r.type == MEM_MAPPED) kind = (alias != 0 && r.base >= alias && r.base < alias + alias_size) ? WsAlias : WsOtherMapped;
		else {
			const bool heap = std::any_of(heaps.begin(), heaps.end(), [&](const auto& h) { return h.first == r.alloc; });
			const auto all  = committed_of(r.alloc);
			kind = heap                                                                             ? WsHeap
			       : (r.protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0 ? WsExec
			       : (r.protect & PAGE_READWRITE) != 0 ? (all >= 64 * MiB ? WsRwLarge : all >= MiB ? WsRwMid : WsRwSmall)
			                                           : WsPrivateOther;
		}
		// A view the guest maps (its bytes the store's): looked up at both addresses.
		const uint8_t* backing     = nullptr;
		uint64_t       backing_end = 0; // (guest address where that backing run ends)
		if (kind == WsOtherMapped) {
			const auto end = Libs::LibKernel::Memory::BackingMappingEnd(r.base);
			if (end > r.base) {
				const auto piece = std::min(end, r.base + r.size) - r.base;
				backing          = Libs::LibKernel::Memory::TryGetBackingPointer(r.base, piece);
				if (backing != nullptr) {
					kind        = WsGuestView;
					backing_end = r.base + piece;
				}
			}
		}
		cat[kind].committed += r.size;
		// (The first sampled page of a region: its base rounded up to the stride, so every region's pages are sampled alike.)
		for (uint64_t page = (r.base + stride - 1) & ~(stride - 1); page < r.base + r.size; page += stride) {
			if (kind == WsGuestView && page < backing_end) {
				const uint64_t twin = reinterpret_cast<uint64_t>(backing) + (page - r.base);
				add(twin, {-1, -1, false, 0});
				add(page, {WsGuestView, static_cast<int32_t>(entries.size() - 1), is_fresh(twin), 0});
			} else if (kind == WsGuestView) {
				add(page, {WsOtherMapped, -1, false, 0});
			} else {
				add(page, {kind, -1, kind == WsAlias && is_fresh(page), r.alloc});
			}
			if (entries.size() >= Batch) flush();
		}
	}
	flush();
	// The guest store: committed, and never written (fresh granules), over the alias's committed runs.
	uint64_t store_committed = 0, store_fresh = 0;
	if (alias != 0) {
		for (const auto& r: regions) {
			if (r.type != MEM_MAPPED || r.base < alias || r.base >= alias + alias_size) continue;
			store_committed += r.size;
			if (fresh == nullptr) continue;
			for (uint64_t at = r.base; at < r.base + r.size; at += FreshMemory::Granule)
				if (is_fresh(at)) store_fresh += std::min<uint64_t>(FreshMemory::Granule, r.base + r.size - at);
		}
	}
	std::sort(private_resident.begin(), private_resident.end());
	std::vector<std::pair<uint64_t, uint64_t>> merged; // resident, base
	for (const auto& [base, bytes]: private_resident) {
		if (!merged.empty() && merged.back().second == base) merged.back().first += bytes;
		else merged.emplace_back(bytes, base);
	}
	std::sort(merged.begin(), merged.end(), std::greater<>());
	const auto mi  = [](uint64_t b) { return static_cast<unsigned long long>(b / MiB); };
	uint64_t   all = 0;
	for (const auto& c: cat) all += c.resident;
	const auto& g = cat[WsGuestView];
	const auto& a = cat[WsAlias];
	uint64_t    priv = 0;
	for (int k = WsHeap; k <= WsPrivateOther; k++) priv += cat[k].resident;
	const auto  use = LocalPlatform::ProcessMemory();
	std::string largest;
	for (size_t i = 0; i < merged.size() && i < 6; i++) {
		char piece[96];
		std::snprintf(piece, sizeof(piece), "%s0x%llx=%llu/%llu", i == 0 ? "" : " ", static_cast<unsigned long long>(merged[i].second),
		              mi(merged[i].first), mi(committed_of(merged[i].second)));
		largest += piece;
	}
	std::printf("RAM breakdown (working set by kind, 1 page in %llu sampled, %.0f ms, %llu calls): %llu MiB resident (working set "
	            "%llu MiB), about %llu MiB of RAM (pages at both guest addresses counted once); guest memory: game's views %llu "
	            "MiB (never written %llu), emulator's alias %llu MiB (never written %llu), resident at both %llu; emulator "
	            "private %llu MiB [heaps %llu, read-write allocations >=64 MiB %llu, 1-64 MiB %llu, <1 MiB %llu, executable "
	            "%llu, other %llu; most resident (MiB resident/committed): %s]; other mapped views %llu MiB, code (exe and "
	            "DLLs) %llu MiB; guest store committed %llu MiB, of it never written %llu MiB (not in RAM unless read)\n",
	            static_cast<unsigned long long>(stride / 4096u), MsSince(start), static_cast<unsigned long long>(calls), mi(all),
	            mi(use.working_set), mi(all - std::min(all, g.both)), mi(g.resident), mi(g.fresh), mi(a.resident), mi(a.fresh),
	            mi(g.both), mi(priv), mi(cat[WsHeap].resident), mi(cat[WsRwLarge].resident), mi(cat[WsRwMid].resident),
	            mi(cat[WsRwSmall].resident), mi(cat[WsExec].resident), mi(cat[WsPrivateOther].resident),
	            largest.empty() ? "none" : largest.c_str(), mi(cat[WsOtherMapped].resident), mi(cat[WsImage].resident),
	            mi(store_committed), mi(store_fresh));
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
		std::printf("RAM flush (vaciado): every %lld s up to %llu MB: caches (re-creatable), guest memory the game released%s "
		            "(KYTY_RAM_FLUSH_SECONDS / KYTY_RAM_FLUSH_MB / KYTY_RAM_FLUSH_TRIM)\n",
		            static_cast<long long>(s.flush_seconds), static_cast<unsigned long long>(s.flush_bytes / 1000000u),
		            s.flush_trim ? ", then the working set trim of cold guest pages" : "; no working set trim");
	std::fflush(stdout);
	g_periodic.store(s.flush_seconds > 0, std::memory_order_relaxed);
	const bool ws_breakdown = log_period > 0 && WsBreakdownWanted(s);
	if (ws_breakdown)
		std::printf("RAM breakdown: on (KYTY_WS_BREAKDOWN=0 turns it off): the working set by kind with the RAM line, from one "
		            "page in every KYTY_WS_BREAKDOWN_STRIDE_KB (default 64) KiB\n");
	FlushState flush_state;
	auto       next_periodic = Clock::now() + std::chrono::seconds(std::max<int64_t>(s.flush_seconds, 1));
	uint64_t   seconds_under_1g = 0, seconds_under_512 = 0;
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
		if (s.flush_seconds > 0 && now >= next_periodic) {
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
			if (ws_breakdown) WsBreakdown();
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
