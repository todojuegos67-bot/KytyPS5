// KYTY_PIN_WATCHED: see pin-watched.h.
#include "pin-watched.h"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace PinWatched {

namespace {

constexpr uint64_t PageSize        = 0x1000;
constexpr uint64_t GranuleBits     = 16; // 64 KiB: the allocation granularity, a view never shares one with another
constexpr uint64_t GranuleSize     = uint64_t {1} << GranuleBits;
constexpr uint64_t PagesInGranule  = GranuleSize / PageSize;
constexpr uint16_t FullMask        = 0xffffu;
constexpr uint64_t MiB             = uint64_t {1} << 20u;
constexpr uint64_t MandatorySlack  = 512 * MiB; // what protections that cannot be skipped may lock past the budget
constexpr uint64_t WorkingSetSpare = 256 * MiB; // the working-set minimum: the budget, the slack and this

struct Granule {
	uint16_t prot   = 0; // pages that keep a protection other than read-write (and are locked)
	uint16_t locked = 0; // pages this module locked
	uint32_t busy   = 0; // protection changes between Begin and Finish
};

struct State {
	SRWLOCK                               lock = SRWLOCK_INIT;
	std::unordered_map<uint64_t, Granule> granules;
	uint64_t                              locked_bytes = 0; // under `lock`
	std::atomic<uint64_t>                 locked_view {0};  // a copy for the log
	uint64_t                              budget = 0;

	std::atomic<uint64_t> protects {0}, pins {0}, unpins {0}, pinned_bytes_total {0};
	std::atomic<uint64_t> unpinned_protects {0}, unpinned_bytes {0};
	std::atomic<uint64_t> fallbacks {0}, fallback_bytes {0};
	std::atomic<uint64_t> lock_failures {0}, last_lock_error {0}, quota_grows {0}, private_skips {0};
	std::atomic<uint64_t> drained {0};

	std::mutex                                 queue_mutex;
	std::vector<std::pair<uint64_t, uint64_t>> queue;
	std::atomic<bool>                          queued {false};
};

State& Get() {
	static auto* state = new State; // (never destroyed: protections may run during static destruction)
	return *state;
}

bool Writable(uint32_t protect) {
	switch (protect & 0xffu) {
		case PAGE_READWRITE:
		case PAGE_WRITECOPY:
		case PAGE_EXECUTE_READWRITE:
		case PAGE_EXECUTE_WRITECOPY: return true;
		default: return false;
	}
}

uint16_t MaskOf(uint64_t granule, uint64_t begin, uint64_t end) {
	const uint64_t g_begin = granule << GranuleBits;
	const uint64_t from    = std::max(begin, g_begin);
	const uint64_t to      = std::min(end, g_begin + GranuleSize);
	if (from >= to) return 0;
	const uint64_t first = (from - g_begin) / PageSize, count = (to - from) / PageSize;
	return static_cast<uint16_t>(count >= PagesInGranule ? FullMask : (((1u << count) - 1u) << first));
}

uint64_t Bytes(uint16_t mask) {
	return static_cast<uint64_t>(std::popcount(static_cast<unsigned>(mask))) * PageSize;
}

void WorkingSet(SIZE_T* minimum, SIZE_T* maximum, DWORD* flags) {
	*minimum = 0;
	*maximum = 0;
	*flags   = 0;
	(void)GetProcessWorkingSetSizeEx(GetCurrentProcess(), minimum, maximum, flags);
}

// The working-set minimum raised (soft: Windows does not keep that much resident for it; VirtualLock's quota is the
// minimum) to `want`; the maximum soft too. A hard minimum the process has (KYTY_RESIDENT_MB) stays hard.
bool RaiseWorkingSet(uint64_t want) {
	SIZE_T minimum = 0, maximum = 0;
	DWORD  flags   = 0;
	WorkingSet(&minimum, &maximum, &flags);
	if (minimum >= want) return true;
	const SIZE_T new_max   = std::max<SIZE_T>(maximum, static_cast<SIZE_T>(want + 1024 * MiB));
	const DWORD  min_flags = (flags & QUOTA_LIMITS_HARDWS_MIN_ENABLE) != 0 ? QUOTA_LIMITS_HARDWS_MIN_ENABLE
	                                                                         : QUOTA_LIMITS_HARDWS_MIN_DISABLE;
	return SetProcessWorkingSetSizeEx(GetCurrentProcess(), static_cast<SIZE_T>(want), new_max,
	                                  min_flags | QUOTA_LIMITS_HARDWS_MAX_DISABLE) != 0;
}

void StartLog();

void Init() {
	static std::once_flag once;
	std::call_once(once, [] {
		auto& s = Get();
		MEMORYSTATUSEX status {};
		status.dwLength = sizeof(status);
		const uint64_t physical = GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 16384 * MiB;
		uint64_t       budget   = std::clamp<uint64_t>(physical / 6, 512 * MiB, 3072 * MiB);
		if (const char* text = std::getenv("KYTY_PIN_BUDGET_MB"); text != nullptr && *text != '\0')
			budget = std::max<uint64_t>(std::strtoull(text, nullptr, 10), 64) * MiB;
		// (Refused: smaller, down to 256 MiB of budget.)
		bool raised = false;
		for (;;) {
			if (RaiseWorkingSet(budget + MandatorySlack + WorkingSetSpare)) {
				raised = true;
				break;
			}
			if (budget <= 256 * MiB) break;
			budget = std::max<uint64_t>(budget / 2, 256 * MiB);
		}
		s.budget = budget;
		SIZE_T minimum = 0, maximum = 0;
		DWORD  flags   = 0;
		WorkingSet(&minimum, &maximum, &flags);
		std::printf("Pin watched: on (KYTY_PIN_WATCHED=1): pages under a protection are locked in RAM; budget %llu MiB "
		            "(+%llu MiB for protections that cannot be skipped) of %llu MiB RAM; working set min %llu / max %llu "
		            "MiB%s; the emulator's working-set trims are off\n",
		            static_cast<unsigned long long>(budget / MiB), static_cast<unsigned long long>(MandatorySlack / MiB),
		            static_cast<unsigned long long>(physical / MiB), static_cast<unsigned long long>(minimum / MiB),
		            static_cast<unsigned long long>(maximum / MiB),
		            raised ? "" : " (raise refused by Windows: pins beyond the quota fail and fall back)");
		std::fflush(stdout);
		StartLog();
	});
}

// Unlocks the pages of `mask` in granule g that are locked (lock held): QueryWorkingSetEx first, as VirtualUnlock of a
// page that is not locked takes it out of the working set (a trim).
void UnlockLocked(State& s, uint64_t granule, uint16_t mask) {
	if (mask == 0) return;
	const uint64_t                    base = granule << GranuleBits;
	PSAPI_WORKING_SET_EX_INFORMATION info[PagesInGranule] {};
	for (uint64_t i = 0; i < PagesInGranule; i++) info[i].VirtualAddress = reinterpret_cast<void*>(base + i * PageSize);
	uint16_t really = mask;
	if (K32QueryWorkingSetEx(GetCurrentProcess(), info, sizeof(info)) != 0) {
		really = 0;
		for (uint64_t i = 0; i < PagesInGranule; i++)
			if ((mask & (1u << i)) != 0 && info[i].VirtualAttributes.Valid != 0 && info[i].VirtualAttributes.Locked != 0)
				really = static_cast<uint16_t>(really | (1u << i));
	} else {
		really = 0; // (unknown: left locked rather than trimmed)
	}
	for (uint64_t i = 0; i < PagesInGranule;) {
		if ((really & (1u << i)) == 0) {
			++i;
			continue;
		}
		uint64_t j = i;
		while (j < PagesInGranule && (really & (1u << j)) != 0) ++j;
		(void)VirtualUnlock(reinterpret_cast<void*>(base + i * PageSize), (j - i) * PageSize);
		i = j;
	}
	s.locked_bytes -= std::min(s.locked_bytes, Bytes(mask));
	s.locked_view.store(s.locked_bytes, std::memory_order_relaxed);
	s.unpins.fetch_add(std::popcount(static_cast<unsigned>(mask)), std::memory_order_relaxed);
}

// Lock held: the granule is dropped when nothing keeps it.
void Settle(State& s, std::unordered_map<uint64_t, Granule>::iterator it) {
	auto& g = it->second;
	if (g.busy != 0 || g.prot != 0) return;
	UnlockLocked(s, it->first, g.locked);
	s.granules.erase(it);
}

bool LockRange(State& s, uint64_t begin, uint64_t end) {
	for (int attempt = 0; attempt < 2; attempt++) {
		if (VirtualLock(reinterpret_cast<void*>(begin), end - begin) != 0) return true;
		const DWORD error = GetLastError();
		if (error != ERROR_WORKING_SET_QUOTA || attempt != 0) {
			s.last_lock_error.store(error, std::memory_order_relaxed);
			return false;
		}
		// (Quota: the minimum is what VirtualLock may hold; raised once more.)
		SIZE_T minimum = 0, maximum = 0;
		DWORD  flags   = 0;
		WorkingSet(&minimum, &maximum, &flags);
		s.quota_grows.fetch_add(1, std::memory_order_relaxed);
		if (!RaiseWorkingSet(static_cast<uint64_t>(minimum) + 256 * MiB)) {
			s.last_lock_error.store(error, std::memory_order_relaxed);
			return false;
		}
	}
	return false;
}

struct Need {
	uint64_t granule = 0;
	uint16_t need    = 0; // pages to lock
	uint16_t got     = 0; // pages locked now
	uint16_t skip    = 0; // pages that are not views of a section (private memory): nothing to pin
};

// Locks the needed pages (no lock held): runs of whole granules in one call, else a granule, else its exact pages.
void LockNeeds(State& s, std::vector<Need>& needs) {
	uint64_t region_end = 0;
	bool     mapped     = false;
	for (size_t i = 0; i < needs.size();) {
		auto&          n    = needs[i];
		const uint64_t base = n.granule << GranuleBits;
		if (base >= region_end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<void*>(base), &info, sizeof(info)) != sizeof(info)) {
				n.skip = n.need;
				++i;
				continue;
			}
			region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
			mapped     = info.Type == MEM_MAPPED;
		}
		if (!mapped) {
			n.skip = n.need;
			s.private_skips.fetch_add(1, std::memory_order_relaxed);
			++i;
			continue;
		}
		// A run of consecutive whole granules inside this region: one call.
		size_t j = i;
		while (j + 1 < needs.size() && needs[j].need == FullMask && needs[j + 1].need == FullMask &&
		       needs[j + 1].granule == needs[j].granule + 1 && (needs[j + 1].granule << GranuleBits) < region_end)
			++j;
		if (j > i && LockRange(s, base, (needs[j].granule + 1) << GranuleBits)) {
			for (size_t k = i; k <= j; k++) needs[k].got = FullMask;
			i = j + 1;
			continue;
		}
		// The whole granule (batching: its other pages are pinned too), else exactly the pages needed.
		if (LockRange(s, base, base + GranuleSize)) {
			n.got = FullMask;
		} else {
			for (uint64_t p = 0; p < PagesInGranule;) {
				if ((n.need & (1u << p)) == 0) {
					++p;
					continue;
				}
				uint64_t q = p;
				while (q < PagesInGranule && (n.need & (1u << q)) != 0) ++q;
				if (LockRange(s, base + p * PageSize, base + q * PageSize))
					n.got = static_cast<uint16_t>(n.got | (((1u << (q - p)) - 1u) << p));
				else
					s.lock_failures.fetch_add(1, std::memory_order_relaxed);
				p = q;
			}
		}
		++i;
	}
}

void StartLog() {
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"Kyty.PinWatched");
		auto&    s               = Get();
		uint64_t last_pins       = 0, last_unpins = 0, last_protects = 0;
		auto     last            = std::chrono::steady_clock::now();
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(30));
			const auto   now     = std::chrono::steady_clock::now();
			const double seconds = std::max(1e-3, std::chrono::duration<double>(now - last).count());
			last                 = now;
			const uint64_t pins = s.pins.load(), unpins = s.unpins.load(), protects = s.protects.load();
			SIZE_T         minimum = 0, maximum = 0;
			DWORD          flags   = 0;
			WorkingSet(&minimum, &maximum, &flags);
			size_t granules = 0;
			AcquireSRWLockShared(&s.lock);
			granules = s.granules.size();
			ReleaseSRWLockShared(&s.lock);
			std::printf("Pin watched (30 s): pinned %llu MiB of %llu MiB budget (%zu granules), pins %.1f/s, unpins %.1f/s "
			            "(pages), protects %.1f/s; protects on unpinned pages %llu (%llu MiB; must be 0); fallbacks "
			            "(write watches left unprotected, re-uploaded) %llu (%llu MiB), drained %llu; lock failures %llu "
			            "(last error %llu), quota raises %llu, private ranges skipped %llu; working set min %llu / max %llu "
			            "MiB%s\n",
			            static_cast<unsigned long long>(s.locked_view.load() / MiB),
			            static_cast<unsigned long long>(s.budget / MiB), granules,
			            static_cast<double>(pins - last_pins) / seconds, static_cast<double>(unpins - last_unpins) / seconds,
			            static_cast<double>(protects - last_protects) / seconds,
			            static_cast<unsigned long long>(s.unpinned_protects.load()),
			            static_cast<unsigned long long>(s.unpinned_bytes.load() / MiB),
			            static_cast<unsigned long long>(s.fallbacks.load()),
			            static_cast<unsigned long long>(s.fallback_bytes.load() / MiB),
			            static_cast<unsigned long long>(s.drained.load()),
			            static_cast<unsigned long long>(s.lock_failures.load()),
			            static_cast<unsigned long long>(s.last_lock_error.load()),
			            static_cast<unsigned long long>(s.quota_grows.load()),
			            static_cast<unsigned long long>(s.private_skips.load()),
			            static_cast<unsigned long long>(minimum / MiB), static_cast<unsigned long long>(maximum / MiB),
			            (flags & QUOTA_LIMITS_HARDWS_MIN_ENABLE) != 0 ? " (hard min)" : "");
			std::fflush(stdout);
			last_pins     = pins;
			last_unpins   = unpins;
			last_protects = protects;
		}
	}).detach();
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_PIN_WATCHED");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

Token Begin(uint64_t address, uint64_t size, uint32_t protect, bool optional) {
	Token token;
	if (!Enabled() || size == 0 || address > UINT64_MAX - size - PageSize) return token;
	Init();
	auto& s          = Get();
	token.begin      = address & ~(PageSize - 1u);
	token.end        = (address + size + PageSize - 1u) & ~(PageSize - 1u);
	token.persistent = !Writable(protect);
	token.active     = true;
	s.protects.fetch_add(1, std::memory_order_relaxed);

	thread_local std::vector<Need> needs;
	needs.clear();
	uint64_t need_bytes = 0, locked_now = 0;
	AcquireSRWLockExclusive(&s.lock);
	for (uint64_t g = token.begin >> GranuleBits, last = (token.end - 1u) >> GranuleBits; g <= last; g++) {
		auto&          granule = s.granules[g];
		const uint16_t mask    = MaskOf(g, token.begin, token.end);
		granule.busy++; // (from here until Finish nothing unlocks the granule)
		const uint16_t need = static_cast<uint16_t>(mask & ~granule.locked);
		if (need != 0) {
			needs.push_back({g, need, 0, 0});
			need_bytes += Bytes(need);
		}
	}
	locked_now = s.locked_bytes;
	ReleaseSRWLockExclusive(&s.lock);
	if (needs.empty()) return token;

	// The budget: a write watch over it is not made (the caller's fallback); other persistent protections may use the
	// slack; a transient one (made writable) locks whatever it needs until Finish.
	const uint64_t cap = !token.persistent ? UINT64_MAX : optional ? s.budget : s.budget + MandatorySlack;
	if (locked_now + need_bytes > cap) {
		if (optional) {
			token.fallback = true;
			s.fallbacks.fetch_add(1, std::memory_order_relaxed);
			s.fallback_bytes.fetch_add(token.end - token.begin, std::memory_order_relaxed);
		} else {
			s.unpinned_protects.fetch_add(1, std::memory_order_relaxed);
			s.unpinned_bytes.fetch_add(need_bytes, std::memory_order_relaxed);
		}
		return token;
	}
	LockNeeds(s, needs);
	uint64_t missing = 0;
	AcquireSRWLockExclusive(&s.lock);
	for (const auto& n: needs) {
		auto&          granule = s.granules[n.granule];
		const uint16_t add     = static_cast<uint16_t>(n.got & ~granule.locked);
		granule.locked         = static_cast<uint16_t>(granule.locked | n.got);
		s.locked_bytes += Bytes(add);
		s.pins.fetch_add(std::popcount(static_cast<unsigned>(add)), std::memory_order_relaxed);
		missing += Bytes(static_cast<uint16_t>(n.need & ~n.got & ~n.skip));
	}
	s.locked_view.store(s.locked_bytes, std::memory_order_relaxed);
	ReleaseSRWLockExclusive(&s.lock);
	if (missing != 0) {
		if (optional) {
			token.fallback = true;
			s.fallbacks.fetch_add(1, std::memory_order_relaxed);
			s.fallback_bytes.fetch_add(token.end - token.begin, std::memory_order_relaxed);
		} else {
			s.unpinned_protects.fetch_add(1, std::memory_order_relaxed);
			s.unpinned_bytes.fetch_add(missing, std::memory_order_relaxed);
		}
	}
	return token;
}

void Finish(const Token& token, bool applied) {
	if (!token.active) return;
	auto& s = Get();
	AcquireSRWLockExclusive(&s.lock);
	for (uint64_t g = token.begin >> GranuleBits, last = (token.end - 1u) >> GranuleBits; g <= last; g++) {
		auto it = s.granules.find(g);
		if (it == s.granules.end()) continue;
		auto&          granule = it->second;
		const uint16_t mask    = MaskOf(g, token.begin, token.end);
		if (granule.busy != 0) granule.busy--;
		if (applied && !token.fallback) {
			if (token.persistent) granule.prot = static_cast<uint16_t>(granule.prot | (mask & granule.locked));
			else granule.prot = static_cast<uint16_t>(granule.prot & ~mask);
		}
		// (Unlocked AFTER the protection is back to read-write, under the lock: no Begin of another thread in between.)
		Settle(s, it);
	}
	ReleaseSRWLockExclusive(&s.lock);
}

void Forget(uint64_t address, uint64_t size) {
	if (!Enabled() || size == 0 || address > UINT64_MAX - size - PageSize) return;
	auto&          s     = Get();
	const uint64_t begin = address & ~(PageSize - 1u), end = (address + size + PageSize - 1u) & ~(PageSize - 1u);
	const uint64_t first = begin >> GranuleBits, last = (end - 1u) >> GranuleBits;
	AcquireSRWLockExclusive(&s.lock);
	const auto drop = [&](std::unordered_map<uint64_t, Granule>::iterator it) {
		auto&          granule = it->second;
		const uint16_t mask    = MaskOf(it->first, begin, end);
		// (The unmapped pages' locks went with the view.)
		s.locked_bytes -= std::min(s.locked_bytes, Bytes(static_cast<uint16_t>(granule.locked & mask)));
		granule.locked = static_cast<uint16_t>(granule.locked & ~mask);
		granule.prot   = static_cast<uint16_t>(granule.prot & ~mask);
	};
	if (last - first + 1 > s.granules.size()) {
		for (auto it = s.granules.begin(); it != s.granules.end();) {
			auto next = std::next(it);
			if (it->first >= first && it->first <= last) {
				drop(it);
				Settle(s, it);
			}
			it = next;
		}
	} else {
		for (uint64_t g = first; g <= last; g++) {
			auto it = s.granules.find(g);
			if (it == s.granules.end()) continue;
			drop(it);
			Settle(s, it);
		}
	}
	s.locked_view.store(s.locked_bytes, std::memory_order_relaxed);
	ReleaseSRWLockExclusive(&s.lock);
}

void QueueUnprotectedWrite(uint64_t address, uint64_t size) {
	auto&           s = Get();
	std::lock_guard lock(s.queue_mutex);
	if (!s.queue.empty() && s.queue.back().first + s.queue.back().second == address) s.queue.back().second += size;
	else s.queue.emplace_back(address, size);
	s.queued.store(true, std::memory_order_release);
}

bool TakeUnprotectedWrites(std::vector<std::pair<uint64_t, uint64_t>>* out) {
	if (!Enabled()) return false;
	auto& s = Get();
	if (!s.queued.load(std::memory_order_acquire)) return false;
	std::lock_guard lock(s.queue_mutex);
	out->swap(s.queue);
	s.queue.clear();
	s.queued.store(false, std::memory_order_relaxed);
	s.drained.fetch_add(out->size(), std::memory_order_relaxed);
	return !out->empty();
}

void NoteTrimSkipped(const char* name) {
	std::printf("%s: off (KYTY_PIN_WATCHED=1: no emulator trim of guest pages)\n", name);
	std::fflush(stdout);
}

} // namespace PinWatched

#endif
