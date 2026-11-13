#include "ram-guard.h"

#include "fresh-memory.h"
#include "gpu-residency.h"
#include "guest-lazy-commit.h"
#include "local-platform.h"
#include "mem-guard.h"
#include "soft-release.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstddef>
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
// (After windows.h.)
#include <psapi.h>
#endif

namespace Libs::LibKernel::Memory {
void ReportGuestAccess(bool force);
}
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

#if defined(_WIN32)
namespace {
std::atomic<uint64_t> g_ram_ballast {0};
// (11-12) KYTY_SIMULATE_RAM_MB=<n> on a bigger PC: hold (physical - n MiB - 1 GiB margin for the bigger PC's own
// OS headroom is NOT kept: the rest of the machine behaves as an n MiB PC) of real RAM, committed, touched and locked,
// so the game and Windows see the memory pressure of an n MiB PC, as the VRAM ballast does for KYTY_SIMULATE_VRAM_MB.
void HoldRamBallast() {
	const char* text = std::getenv("KYTY_SIMULATE_RAM_MB");
	if (text == nullptr || *text == '\0') return;
	const uint64_t want = std::strtoull(text, nullptr, 10) << 20u;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	if (want == 0 || GlobalMemoryStatusEx(&status) == 0 || status.ullTotalPhys <= want) return;
	uint64_t ballast = status.ullTotalPhys - want;
	// Never more than what is free now less 2 GiB (the game must still start).
	if (status.ullAvailPhys < (2ull << 30u)) return;
	ballast = std::min<uint64_t>(ballast, status.ullAvailPhys - (2ull << 30u));
	SIZE_T wmin = 0, wmax = 0;
	GetProcessWorkingSetSize(GetCurrentProcess(), &wmin, &wmax);
	SetProcessWorkingSetSizeEx(GetCurrentProcess(), wmin + ballast + (64ull << 20u), wmax + ballast + (256ull << 20u),
	                           QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
	constexpr uint64_t kChunk = 256ull << 20u;
	uint64_t held = 0, locked = 0;
	for (; held + kChunk <= ballast; held += kChunk) {
		void* p = VirtualAlloc(nullptr, kChunk, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		if (p == nullptr) break;
		for (uint64_t o = 0; o < kChunk; o += 4096) static_cast<volatile char*>(p)[o] = 1;
		if (VirtualLock(p, kChunk) != 0) locked += kChunk;
	}
	g_ram_ballast = held;
	std::printf("RAM simulation: %llu MiB PC on %llu MiB: %llu MiB of RAM held (%llu MiB locked)\n",
	            static_cast<unsigned long long>(want >> 20u), static_cast<unsigned long long>(status.ullTotalPhys >> 20u),
	            static_cast<unsigned long long>(held >> 20u), static_cast<unsigned long long>(locked >> 20u));
	std::fflush(stdout);
}
} // namespace
#endif

uint64_t RamBallast() {
#if defined(_WIN32)
	return g_ram_ballast.load();
#else
	return 0;
#endif
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

// KYTY_RESIDENCY_LOG=0 turns it off (default on): with the 30 s RAM line, where the working set is, from the process's
// whole working-set list (K32QueryWorkingSet, one walk on the guard's thread, ~10-40 ms for 4M pages): pages of the guest
// store's emulator-side alias, pages of the other mapped views (the game's own views of the guest store: direct and
// flexible memory), private pages (the emulator's own allocations and the driver's), images (DLLs); against what of the
// store is committed and what of it was ever written (fresh-memory granules: CPU, host or write-back writes). Committed
// but never touched pages of the store are commit charge only: no RAM. A page touched through both the alias and a game
// view is in the working set twice (one physical page): the alias and view figures can overlap.
bool ResidencyLogEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RESIDENCY_LOG");
		return v == nullptr || !(v[0] == '0' && v[1] == '\0');
	}();
	return on;
}

void PrintResidency(uint64_t store_committed) {
	if (!ResidencyLogEnabled()) return;
	const auto t0 = std::chrono::steady_clock::now();
	// (Freed after each line: ~8 bytes a resident page, not kept.)
	std::vector<ULONG_PTR> buffer(size_t {1} << 16u);
	HANDLE                 process = GetCurrentProcess();
	bool ok = false;
	for (int attempt = 0; attempt < 4; attempt++) {
		if (K32QueryWorkingSet(process, buffer.data(), static_cast<DWORD>(buffer.size() * sizeof(ULONG_PTR))) != 0) {
			ok = true;
			break;
		}
		if (GetLastError() != ERROR_BAD_LENGTH) break;
		const auto needed = static_cast<size_t>(buffer[0]); // NumberOfEntries
		buffer.resize(std::max(buffer.size() * 2, needed + needed / 8 + 4096 + 1));
	}
	if (!ok) {
		std::printf("RAM residency (30 s): K32QueryWorkingSet failed (0x%08x)\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	const size_t count = std::min(static_cast<size_t>(buffer[0]), buffer.size() - 1);
	uint64_t     shared_pages = 0;
	// (In place: the entries become page addresses, sorted, so the region walk queries each region once.)
	for (size_t i = 1; i <= count; i++) {
		if ((buffer[i] & 0x100u) != 0) shared_pages++; // PSAPI_WORKING_SET_BLOCK.Shared (bit 8)
		buffer[i] &= ~static_cast<ULONG_PTR>(0xfff);
	}
	std::sort(buffer.begin() + 1, buffer.begin() + 1 + static_cast<std::ptrdiff_t>(count));
	const auto     alias      = reinterpret_cast<uint64_t>(g_store_alias.load(std::memory_order_relaxed));
	const uint64_t store_size = g_store_size.load(std::memory_order_relaxed);
	constexpr uint64_t Page   = 4096;
	uint64_t in_alias = 0, in_views = 0, in_private = 0, in_image = 0, in_other = 0;
	uint64_t region_begin = 0, region_end = 0;
	DWORD    region_type  = 0;
	for (size_t i = 1; i <= count; i++) {
		const auto page = static_cast<uint64_t>(buffer[i]);
		if (alias != 0 && page >= alias && page - alias < store_size) {
			in_alias += Page;
			continue;
		}
		if (page < region_begin || page >= region_end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) != sizeof(info)) {
				in_other += Page;
				region_begin = region_end = 0;
				continue;
			}
			region_begin = reinterpret_cast<uint64_t>(info.BaseAddress);
			region_end   = region_begin + info.RegionSize;
			region_type  = info.Type;
		}
		(region_type == MEM_MAPPED ? in_views : region_type == MEM_PRIVATE ? in_private : region_type == MEM_IMAGE ? in_image : in_other) += Page;
	}
	uint64_t written = 0;
	if (const auto* fresh = FreshMemory::Get(); fresh != nullptr) {
		const uint64_t words = (fresh->granules + 63) / 64;
		for (uint64_t w = 0; w < words; w++)
			for (uint64_t bits = fresh->written[w].load(std::memory_order_relaxed); bits != 0; bits &= bits - 1) written++;
		written <<= FreshMemory::GranuleBits;
	}
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::printf("RAM residency (30 s, MiB; KYTY_RESIDENCY_LOG=0 off): working set %llu (shared pages %llu) = guest store "
	            "through the emulator's alias %llu + other mapped views (the game's views of the guest store) %llu + private "
	            "%llu + images %llu + other %llu; guest store committed %llu, ever written %s%llu, so committed and never "
	            "written <= %llu (commit charge, no RAM unless read); %.1f ms\n",
	            static_cast<unsigned long long>(count * Page / MiB), static_cast<unsigned long long>(shared_pages * Page / MiB),
	            static_cast<unsigned long long>(in_alias / MiB), static_cast<unsigned long long>(in_views / MiB),
	            static_cast<unsigned long long>(in_private / MiB), static_cast<unsigned long long>(in_image / MiB),
	            static_cast<unsigned long long>(in_other / MiB), static_cast<unsigned long long>(store_committed / MiB),
	            FreshMemory::Enabled() ? "" : "(fresh memory off: unknown) ", static_cast<unsigned long long>(written / MiB),
	            static_cast<unsigned long long>((store_committed > written ? store_committed - written : 0) / MiB), ms);
	std::fflush(stdout);
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
	uint64_t budget        = 0;                      // KYTY_RAM_BUDGET_MB: the working set kept under it (0 off)
	uint64_t budget_step   = 0;                      // KYTY_RAM_BUDGET_STEP_MB: the most trimmed a second
	uint64_t budget_floor  = 0;                      // the hard working set minimum kept meanwhile (KYTY_RESIDENT_MB)
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
	s.budget      = EnvMiB("KYTY_RAM_BUDGET_MB", 0);
	if (s.budget != 0) s.budget = std::max<uint64_t>(s.budget, 2 * GiB);
	s.budget_step = std::clamp<uint64_t>(EnvMiB("KYTY_RAM_BUDGET_STEP_MB", 384 * MiB), 64 * MiB, 2 * GiB);
	// (KYTY_RESIDENT_MB's hard minimum kept, but under the budget's trim target.)
	s.budget_floor = s.budget != 0 ? std::min(EnvMiB("KYTY_RESIDENT_MB", 0), s.budget - GiB) : 0;
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
	if (s.budget != 0) {
		std::printf("RAM budget: working set kept under %llu MiB (KYTY_RAM_BUDGET_MB; 0 turns it off): over it, up to %llu "
		            "MiB a second trimmed to the page file / standby list in quiet windows (hard minimum %llu MiB), the "
		            "release passes run as under pressure (critical 1 GiB over), released guest memory dropped\n",
		            static_cast<unsigned long long>(s.budget / MiB), static_cast<unsigned long long>(s.budget_step / MiB),
		            static_cast<unsigned long long>(s.budget_floor / MiB));
	}
	std::fflush(stdout);
	g_periodic.store(s.flush_seconds > 0, std::memory_order_relaxed);
	FlushState flush_state;
	auto       next_periodic = Clock::now() + std::chrono::seconds(std::max<int64_t>(s.flush_seconds, 1));
	uint64_t   seconds_under_1g = 0, seconds_under_512 = 0;
	// The RAM budget: what was trimmed (in all and since the last line), the seconds over it, the trims refused.
	uint64_t   budget_trimmed = 0, budget_trimmed_line = 0, budget_seconds_over = 0, budget_refused = 0;
	uint64_t   budget_ws_max_line = 0;
	auto       budget_next_guest = Clock::now();
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
		int level = LevelOf(s, sample);
		if (s.budget != 0) {
			budget_ws_max_line = std::max(budget_ws_max_line, sample.working_set);
			if (sample.working_set > s.budget) {
				budget_seconds_over++;
				// Over the budget: the release passes and the owners (pipelines, images) as under pressure; critical a GiB
				// over. Then the working set trimmed one step (a few hundred MiB, inside the guest memory's quiet window)
				// toward 256 MiB under the budget: cold pages first (Windows trims the least recently used).
				level = std::max(level, sample.working_set > s.budget + GiB ? static_cast<int>(Critical) : static_cast<int>(Low));
				const uint64_t goal   = s.budget - 256 * MiB;
				const uint64_t target = std::max(goal, sample.working_set - std::min(sample.working_set, s.budget_step));
				if (target > s.budget_floor) {
					if (LocalPlatform::TrimWorkingSetTo(target, s.budget_floor)) {
						const uint64_t after = LocalPlatform::ProcessMemory().working_set;
						const uint64_t cut   = sample.working_set > after ? sample.working_set - after : 0;
						budget_trimmed += cut;
						budget_trimmed_line += cut;
					} else {
						budget_refused++;
					}
				}
				// Guest memory the game released but still in RAM, dropped (no data lost), at most every 30 s.
				if (now >= budget_next_guest) {
					budget_next_guest = now + std::chrono::seconds(30);
					Settings guest  = s;
					guest.flush_trim = false;
					FlushPass(guest, low_memory, store, &flush_state);
				}
			}
		}
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
			Libs::LibKernel::Memory::ReportGuestAccess(true);
			PrintResidency(sample.store_committed);
			std::printf("RAM (%lld s): %s; pressure %s, real available %llu MiB, real commit headroom %llu MiB; pressure "
			            "passes %llu (%llu MB), pipelines released %llu, images collected %llu (%llu MiB)\n",
			            static_cast<long long>(log_period), Figures(sample).c_str(), LevelName(level),
			            static_cast<unsigned long long>(sample.real_available / MiB),
			            static_cast<unsigned long long>(sample.real_headroom / MiB), static_cast<unsigned long long>(flushes),
			            static_cast<unsigned long long>(freed_total / 1000000u),
			            static_cast<unsigned long long>(g_released_count_total[Pipelines].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_count_total[Images].load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_released_bytes_total[Images].load(std::memory_order_relaxed) / MiB));
			if (s.budget != 0) {
				const uint64_t ws_now = LocalPlatform::ProcessMemory().working_set;
				std::printf("RAM budget (%lld s): budget %llu MiB, working set %llu MiB now (peak %llu MiB this period), "
				            "trimmed %llu MiB this period, %llu MiB in all; %llu s over budget in all, %llu trims refused\n",
				            static_cast<long long>(log_period), static_cast<unsigned long long>(s.budget / MiB),
				            static_cast<unsigned long long>(ws_now / MiB), static_cast<unsigned long long>(budget_ws_max_line / MiB),
				            static_cast<unsigned long long>(budget_trimmed_line / MiB),
				            static_cast<unsigned long long>(budget_trimmed / MiB),
				            static_cast<unsigned long long>(budget_seconds_over), static_cast<unsigned long long>(budget_refused));
				budget_trimmed_line = 0;
				budget_ws_max_line  = 0;
			}
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

namespace GpuResidency {

#if defined(_WIN32)
namespace {

constexpr uint64_t SampleBits = 16; // one page sampled in each 64 KiB granule of guest address space
constexpr uint64_t SampleSpan = uint64_t {1} << SampleBits;
constexpr uint64_t PageSize   = 4096;
constexpr size_t   MaxSamples = size_t {1} << 21u; // (48 MiB of samples at most; ~13 GiB of ranges is ~220K)

struct Sample {
	uint64_t page   = 0; // guest address (the game's view)
	uint64_t alias  = 0; // the same page in the backing store's alias (0: no backing: unmapped)
	uint32_t weight = 0; // the range's bytes in the sample's 64 KiB granule
	Kind     kind   = Texture;
	uint8_t  state  = 0; // StateView | StateAlias | StateWritten | StateKnown
};
constexpr uint8_t StateView    = 1; // in the working set through the guest address
constexpr uint8_t StateAlias   = 2; // in the working set through the alias
constexpr uint8_t StateWritten = 4; // fresh memory: its 64 KiB granule was ever written (CPU, host or write-back)
constexpr uint8_t StateKnown   = 8; // fresh memory knows the granule

struct Totals {
	uint64_t bytes = 0, view_only = 0, alias_only = 0, both = 0, unmapped = 0, resident_unwritten = 0;
	[[nodiscard]] uint64_t Resident() const { return view_only + alias_only + both; }
	void Add(const Sample& sample, uint64_t weight) {
		bytes += weight;
		const bool view = (sample.state & StateView) != 0, alias = (sample.state & StateAlias) != 0;
		if (!view && !alias) {
			if (sample.alias == 0) unmapped += weight; // (no backing under it: nothing to be resident)
			return;
		}
		if (view && alias) both += weight;
		else if (view) view_only += weight;
		else alias_only += weight;
		if ((sample.state & StateKnown) != 0 && (sample.state & StateWritten) == 0) resident_unwritten += weight;
	}
};

// The order a granule several kinds share is counted under in the union.
int UnionRank(Kind kind) {
	switch (kind) {
		case Target: return 0;
		case Storage: return 1;
		case GpuWrittenBuffer: return 2;
		case Texture: return 3;
		case GpuBuffer:
		default: return 4;
	}
}

const char* KindName(Kind kind) {
	switch (kind) {
		case Target: return "render targets/depth/video out";
		case Storage: return "storage/GPU-written images";
		case GpuWrittenBuffer: return "GPU-written buffers";
		case GpuBuffer: return "other GPU buffers";
		case Texture:
		default: return "sampled textures";
	}
}

double ToMiB(uint64_t bytes) {
	return static_cast<double>(bytes) / 1048576.0;
}

// The samples of the ranges: one page in each 64 KiB granule a range covers (the page at the pass's offset in the
// granule, clamped into the range: over passes every 16th page), weighted by the range's bytes in the granule; each
// with its alias address (the backing pieces of the range: the guest address space's shared map lock, a lookup a
// range).
bool MakeSamples(const std::vector<Range>& ranges, uint64_t pass, std::vector<Sample>* samples) {
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	const uint64_t                                   offset = (pass % (SampleSpan / PageSize)) * PageSize;
	for (const auto& range: ranges) {
		if (range.size == 0 || UINT64_MAX - range.address < range.size) continue;
		const uint64_t end    = range.address + range.size;
		const bool     backed = Libs::LibKernel::Memory::TryGetSparseBackingPieces(range.address, range.size, pieces);
		size_t         piece  = 0;
		uint64_t       piece_begin = 0; // (offset in the range of pieces[piece])
		for (uint64_t granule = range.address & ~(SampleSpan - 1); granule < end; granule += SampleSpan) {
			if (samples->size() >= MaxSamples) return false;
			const uint64_t lo    = std::max(granule, range.address);
			const uint64_t hi    = std::min(granule + SampleSpan, end);
			const uint64_t first = lo & ~(PageSize - 1), last = (hi - 1) & ~(PageSize - 1);
			Sample         sample;
			sample.page   = std::clamp(granule + offset, first, last);
			sample.weight = static_cast<uint32_t>(hi - lo);
			sample.kind   = range.kind;
			if (backed) {
				const uint64_t at = sample.page > range.address ? sample.page - range.address : 0;
				while (piece < pieces.size() && at >= piece_begin + pieces[piece].second) piece_begin += pieces[piece++].second;
				if (piece < pieces.size() && pieces[piece].first != nullptr)
					sample.alias = (reinterpret_cast<uint64_t>(pieces[piece].first) + (at - piece_begin)) & ~(PageSize - 1);
			}
			samples->push_back(sample);
			if (UINT64_MAX - granule < SampleSpan) break;
		}
	}
	return true;
}

// The working-set state of each sample's two pages (K32QueryWorkingSetEx, in batches); false when it failed.
bool QuerySamples(std::vector<Sample>* samples) {
	// (Samples a call, two entries each: a short call, then the CPU yielded, so the kernel's walk of the working set
	// never holds it long against the game's page faults.)
	constexpr size_t                               Batch = 1024;
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> entries(Batch * 2);
	HANDLE                                         process = GetCurrentProcess();
	const auto*                                    fresh   = FreshMemory::Get();
	for (size_t begin = 0; begin < samples->size(); begin += Batch) {
		const size_t count = std::min(Batch, samples->size() - begin);
		for (size_t i = 0; i < count; i++) {
			const auto& sample               = (*samples)[begin + i];
			entries[i * 2]                   = {};
			entries[i * 2 + 1]               = {};
			entries[i * 2].VirtualAddress     = reinterpret_cast<PVOID>(sample.page);
			entries[i * 2 + 1].VirtualAddress = reinterpret_cast<PVOID>(sample.alias != 0 ? sample.alias : sample.page);
		}
		if (begin != 0) SwitchToThread();
		if (K32QueryWorkingSetEx(process, entries.data(), static_cast<DWORD>(count * 2 * sizeof(entries[0]))) == 0)
			return false;
		for (size_t i = 0; i < count; i++) {
			auto& sample = (*samples)[begin + i];
			// (PSAPI_WORKING_SET_EX_BLOCK.Valid, bit 0: the page is in the process's working set at that address.)
			const bool view  = (entries[i * 2].VirtualAttributes.Flags & 1u) != 0;
			const bool alias = sample.alias != 0 && (entries[i * 2 + 1].VirtualAttributes.Flags & 1u) != 0;
			sample.state     = static_cast<uint8_t>((view ? StateView : 0) | (alias ? StateAlias : 0));
			if (fresh != nullptr && sample.alias != 0) {
				const uint64_t granule = FreshMemory::GranuleOf(*fresh, sample.alias);
				if (granule != UINT64_MAX) {
					sample.state = static_cast<uint8_t>(sample.state | StateKnown |
					                                    (FreshMemory::IsWritten(*fresh, granule) ? StateWritten : 0));
				}
			}
		}
	}
	return true;
}

// The lines (see the header gpu-residency.h, and the report in the commit): per kind as registered (a granule two
// kinds share is in both), then the union (each 64 KiB granule once, under the first kind by UnionRank).
void PrintGpuResidency(std::vector<Range>& ranges, uint64_t images, uint64_t buffers, double copy_ms, uint64_t pass) {
	const auto          t0 = std::chrono::steady_clock::now();
	std::vector<Sample> samples;
	samples.reserve(std::min<size_t>(MaxSamples, 1u << 18u));
	// (The backing lookups take the address space's shared lock, and the working-set queries the kernel's working-set
	// lock: at normal priority, so neither is held by a descheduled low-priority thread while a map, an unmap or a page
	// fault waits.)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
	const bool complete = MakeSamples(ranges, pass, &samples);
	ranges.clear();
	ranges.shrink_to_fit();
	const double lookup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	const bool   queried   = QuerySamples(&samples);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	if (!queried) {
		std::printf("GPU residency (30 s): K32QueryWorkingSetEx failed (0x%08x)\n", static_cast<unsigned>(GetLastError()));
		std::fflush(stdout);
		return;
	}
	Totals by_kind[Kinds];
	for (const auto& sample: samples) by_kind[static_cast<size_t>(sample.kind)].Add(sample, sample.weight);
	// The union: by granule, the first kind by rank; the granule's largest weight; resident through a view where any of
	// its samples is; written where any of its samples' granules was.
	std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
		const uint64_t ga = a.page >> SampleBits, gb = b.page >> SampleBits;
		return ga != gb ? ga < gb : UnionRank(a.kind) < UnionRank(b.kind);
	});
	Totals union_kind[Kinds], union_all;
	for (size_t i = 0; i < samples.size();) {
		Sample   merged = samples[i];
		uint64_t weight = merged.weight;
		size_t   j      = i + 1;
		for (; j < samples.size() && (samples[j].page >> SampleBits) == (merged.page >> SampleBits); j++) {
			weight = std::max<uint64_t>(weight, samples[j].weight);
			if (merged.alias == 0) merged.alias = samples[j].alias;
			merged.state = static_cast<uint8_t>(merged.state | samples[j].state); // (any sample's view, any written)
		}
		union_kind[static_cast<size_t>(merged.kind)].Add(merged, weight);
		union_all.Add(merged, weight);
		i = j;
	}
	const auto   ws = LocalPlatform::ProcessMemory().working_set;
	const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::printf("GPU residency (30 s, MiB of guest memory resident in the working set, 1 page in 64 KiB sampled; "
	            "KYTY_GPU_RESIDENCY=0 off): %llu images and %llu buffers registered (copied on the GPU thread in %.2f ms), "
	            "%zu samples%s, %.1f ms on this thread (backing lookups %.1f ms); process working set %.0f\n",
	            static_cast<unsigned long long>(images), static_cast<unsigned long long>(buffers), copy_ms, samples.size(),
	            complete ? "" : " (capped: the rest not measured)", ms, lookup_ms, ToMiB(ws));
	const auto print = [](const char* what, const char* name, const Totals& t) {
		std::printf("GPU residency %s %s: range %.0f, resident %.0f = through the game's view only %.0f + the emulator's "
		            "alias only %.0f + both %.0f; not resident %.0f, unmapped %.0f; resident but never written %.0f\n",
		            what, name, ToMiB(t.bytes), ToMiB(t.Resident()), ToMiB(t.view_only), ToMiB(t.alias_only), ToMiB(t.both),
		            ToMiB(t.bytes - std::min(t.bytes, t.Resident() + t.unmapped)), ToMiB(t.unmapped),
		            ToMiB(t.resident_unwritten));
	};
	for (size_t kind = 0; kind < static_cast<size_t>(Kinds); kind++) print("[by kind]", KindName(static_cast<Kind>(kind)), by_kind[kind]);
	for (size_t kind = 0; kind < static_cast<size_t>(Kinds); kind++) print("[union]", KindName(static_cast<Kind>(kind)), union_kind[kind]);
	print("[union]", "all GPU-cached guest memory", union_all);
	std::fflush(stdout);
}

void Watch() {
	SetThreadDescription(GetCurrentThread(), L"Kyty.GpuResidency");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
	std::printf("GPU residency: on (KYTY_GPU_RESIDENCY=0 turns it off): every 30 s, the working set over the guest ranges "
	            "of the GPU caches' images and buffers, read-only\n");
	std::fflush(stdout);
	auto& exchange = Get();
	for (uint64_t pass = 1;; pass++) {
		std::this_thread::sleep_for(std::chrono::seconds(30));
		std::vector<Range> ranges;
		ranges.reserve(size_t {1} << 16u); // (here, not on the GPU thread)
		std::vector<Range> stale;            // (a late answer to a pass not measured: freed outside the lock)
		uint64_t           images = 0, buffers = 0;
		double             copy_ms = 0.0;
		{
			// (Held only for these swaps and the wait's checks: the GPU thread only tries it, never waits.)
			std::unique_lock lock(exchange.mutex);
			stale.swap(exchange.ranges);
			exchange.ranges.swap(ranges);
			exchange.request = pass;
			exchange.requested.store(true, std::memory_order_release);
			// (No flip in 10 s - a loading screen held, the game paused in a debugger, or the answer dropped: this pass is
			// not measured.)
			if (!exchange.delivered.wait_for(lock, std::chrono::seconds(10), [&] { return exchange.answer == pass; })) {
				exchange.requested.store(false, std::memory_order_relaxed);
				lock.unlock();
				std::printf("GPU residency (30 s): no answer from the GPU thread in 10 s, not measured\n");
				std::fflush(stdout);
				continue;
			}
			ranges.swap(exchange.ranges);
			images  = exchange.images;
			buffers = exchange.buffers;
			copy_ms = exchange.copy_ms;
		}
		PrintGpuResidency(ranges, images, buffers, copy_ms, pass);
	}
}

} // namespace
#endif

void Start() {
#if defined(_WIN32)
	if (!Enabled()) {
		std::printf("GPU residency: off (KYTY_GPU_RESIDENCY=0)\n");
		std::fflush(stdout);
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] { std::thread(Watch).detach(); });
#endif
}

} // namespace GpuResidency
