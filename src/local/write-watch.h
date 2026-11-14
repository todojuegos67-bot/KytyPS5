#ifndef EMULATOR_SRC_LOCAL_WRITE_WATCH_H_
#define EMULATOR_SRC_LOCAL_WRITE_WATCH_H_

// KYTY_WRITE_WATCH=1 (off by default): the move from protection-based write tracking to Windows write watch
// (MEM_WRITE_WATCH + GetWriteWatch/ResetWriteWatch). Design and stages: docs/write-watch-design.md.
//
// Bug check 0x1A/0x41202 (MEMORY_MANAGEMENT, the user's reboots): a page of the page-file-backed guest section is
// paged out while its protection changes. The trackers change protections of section views tens of thousands of
// times a second. Write watch needs private VirtualAlloc memory (a section view cannot be write-watched), so every
// tracker moved to it also leaves the section, and its protection changes stop being the dangerous kind.
//
// This stage (2a) changes no tracking and no memory: it measures, in the field, what must move and whether the system
// gives what the move needs.
//  - Every page-manager protection change (Memory::ProtectGuestHostMemory) is classified: on a view of the section
//    in direct memory, on a view in flexible memory (the store past the direct part), or not on a view (private
//    memory: no 0x41202 risk). Logged every 10 s; the target of the next stages is "on section views" trending to 0.
//  - Once at start-up, a probe: a placeholder replaced by MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH (what guest
//    memory would need, VirtualAlloc2 on the reserved guest space), and a plain VirtualAlloc MEM_WRITE_WATCH one;
//    pages written, GetWriteWatch with WRITE_WATCH_FLAG_RESET timed (the per-frame poll cost per GiB watched), the
//    written pages checked (exactly those, none after the reset). Allocated, measured, freed: nothing the game uses.
//
// Nothing here frees, moves or protects guest memory.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

namespace WriteWatch {

enum Class : uint32_t { SectionDirect = 0, SectionFlexible = 1, Private = 2, Classes = 3 };

struct State {
	std::atomic<uint64_t> calls[Classes] {};
	std::atomic<uint64_t> bytes[Classes] {};
	std::atomic<uint64_t> next_ms {0};
	std::atomic<uint64_t> since_ms {0};
	std::atomic<uint64_t> last_frame {0};
	std::atomic<bool>     probed {false};
};

inline State& Get() {
	static State s;
	return s;
}

[[nodiscard]] inline bool Requested() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_WRITE_WATCH");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

[[nodiscard]] inline uint64_t NowMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// The start-up probe (see the top). Windows only; prints one line per kind of memory.
inline void Probe() {
#if defined(_WIN32)
	constexpr uint64_t Size = uint64_t {64} << 20u, Page = 4096;
	const uint64_t     pages = Size / Page;
	using VirtualAlloc2Fn    = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
	auto* alloc2 = reinterpret_cast<VirtualAlloc2Fn>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernelbase.dll"), "VirtualAlloc2")));
	for (int kind = 0; kind < 2; kind++) {
		const char* name = kind == 0 ? "placeholder replaced (VirtualAlloc2)" : "plain VirtualAlloc";
		uint8_t*    p    = nullptr;
		DWORD       err  = 0;
		if (kind == 0) {
			if (alloc2 == nullptr) {
				std::printf("[write-watch] probe %s: VirtualAlloc2 not found\n", name);
				continue;
			}
			auto* ph = static_cast<uint8_t*>(
			    alloc2(GetCurrentProcess(), nullptr, Size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
			if (ph == nullptr) {
				std::printf("[write-watch] probe %s: placeholder refused 0x%08lx\n", name, GetLastError());
				continue;
			}
			p = static_cast<uint8_t*>(alloc2(GetCurrentProcess(), ph, Size,
			                                 MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH | MEM_REPLACE_PLACEHOLDER,
			                                 PAGE_READWRITE, nullptr, 0));
			err = p == nullptr ? GetLastError() : 0;
			if (p == nullptr) VirtualFree(ph, 0, MEM_RELEASE);
		} else {
			p   = static_cast<uint8_t*>(VirtualAlloc(nullptr, Size, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
			err = p == nullptr ? GetLastError() : 0;
		}
		if (p == nullptr) {
			std::printf("[write-watch] probe %s + MEM_WRITE_WATCH: REFUSED 0x%08lx (guest memory cannot use it this way)\n",
			            name, err);
			continue;
		}
		std::vector<PVOID> hits(pages);
		ULONG_PTR          count = pages;
		ULONG              gran  = 0;
		GetWriteWatch(WRITE_WATCH_FLAG_RESET, p, Size, hits.data(), &count, &gran); // (the commit itself)
		// Written: every 64th page (256 of 16384 pages; 1 MiB resident at most).
		uint64_t written = 0;
		for (uint64_t i = 0; i < pages; i += 64, written++) p[i * Page] = 1;
		count            = pages;
		const auto t0    = std::chrono::steady_clock::now();
		const UINT r1    = GetWriteWatch(WRITE_WATCH_FLAG_RESET, p, Size, hits.data(), &count, &gran);
		const auto t1    = std::chrono::steady_clock::now();
		const ULONG_PTR found = count;
		bool            exact = r1 == 0 && found == written;
		for (ULONG_PTR i = 0; exact && i < found; i++)
			exact = (static_cast<uint8_t*>(hits[i]) - p) % (64 * Page) == 0;
		count            = pages;
		const auto t2    = std::chrono::steady_clock::now();
		const UINT r2    = GetWriteWatch(WRITE_WATCH_FLAG_RESET, p, Size, hits.data(), &count, &gran);
		const auto t3    = std::chrono::steady_clock::now();
		exact            = exact && r2 == 0 && count == 0;
		const double us1 = std::chrono::duration<double, std::micro>(t1 - t0).count();
		const double us2 = std::chrono::duration<double, std::micro>(t3 - t2).count();
		std::printf("[write-watch] probe %s + MEM_WRITE_WATCH: ok, %s; GetWriteWatch+reset over 64 MiB: %.1f us with %llu "
		            "dirty pages, %.1f us clean (per GiB watched: %.1f / %.1f us)\n",
		            name, exact ? "exact (written pages only, none after reset)" : "NOT EXACT", us1,
		            static_cast<unsigned long long>(found), us2, us1 * 16, us2 * 16);
		VirtualFree(p, 0, MEM_RELEASE);
	}
#else
	std::printf("[write-watch] probe: not Windows (Linux would use soft-dirty or userfaultfd write-protect)\n");
#endif
	std::fflush(stdout);
}

// Every 10 s (called by Note and by the frame boundary): the per-second protection changes by class.
inline void Tick(uint64_t frame = 0) {
	if (!Requested()) return;
	auto&      s   = Get();
	const auto now = NowMs();
	if (!s.probed.exchange(true)) {
		Probe();
		s.since_ms.store(now);
		s.next_ms.store(now + 10000);
	}
	if (frame != 0) s.last_frame.store(frame, std::memory_order_relaxed);
	uint64_t next = s.next_ms.load(std::memory_order_relaxed);
	if (now < next || !s.next_ms.compare_exchange_strong(next, now + 10000)) return;
	static std::atomic<uint64_t> last_frame_logged {0};
	const uint64_t               since  = s.since_ms.exchange(now);
	const double                 secs   = since < now ? static_cast<double>(now - since) / 1000.0 : 10.0;
	const uint64_t               frame1 = s.last_frame.load(std::memory_order_relaxed);
	const uint64_t               frames = frame1 - last_frame_logged.exchange(frame1);
	uint64_t                     c[Classes], b[Classes];
	for (uint32_t i = 0; i < Classes; i++) {
		c[i] = s.calls[i].exchange(0);
		b[i] = s.bytes[i].exchange(0);
	}
	const uint64_t section = c[SectionDirect] + c[SectionFlexible], total = section + c[Private];
	std::printf("[write-watch] %.0f s: per s protect calls on section memory %.0f (%.0f MiB; direct %.0f, flexible %.0f) "
	            "= %.1f%% of %.0f, on private memory %.0f (%.0f MiB); %.1f frames/s; GetWriteWatch polls 0 (stage 2a: "
	            "nothing polls yet)\n",
	            secs, section / secs, (b[SectionDirect] + b[SectionFlexible]) / secs / 1048576.0, c[SectionDirect] / secs,
	            c[SectionFlexible] / secs, total != 0 ? 100.0 * static_cast<double>(section) / static_cast<double>(total) : 0.0,
	            total / secs, c[Private] / secs, b[Private] / secs / 1048576.0, frames / secs);
	std::fflush(stdout);
}

inline void Note(Class c, uint64_t size) {
	auto& s = Get();
	s.calls[c].fetch_add(1, std::memory_order_relaxed);
	s.bytes[c].fetch_add(size, std::memory_order_relaxed);
	Tick();
}

} // namespace WriteWatch

#endif // EMULATOR_SRC_LOCAL_WRITE_WATCH_H_
