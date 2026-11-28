#pragma once
// (ram-emu) The "[ram-emu]" line, every 5 s (Windows; KYTY_RAM_EMU=0 turns it off): the process's private memory in RAM
// split into the GAME's (committed pages inside the guest's reserved ranges, kernel/memoryAddressSpace.inc m_owned) and
// the EMULATOR's own (every other committed private page: C++ heap, large malloc/new blocks, Vulkan driver memory, JIT,
// stacks, rings), the emulator's part by consumer, and its largest allocations with their growth since the last line.
// Why: the "[ram16]" line's "emulador: privado en RAM" is the whole process (game included); this tells how much of it is
// the emulator's and where, so a cut is decided with data. Cost: one VirtualQuery walk and K32QueryWorkingSetEx (the
// emulator's pages one sample per 64 KiB, the game's one per MiB), on its own lowest-priority thread, read-only.
#include <windows.h>
#include "ram-dueno.h"
#include <psapi.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::LibKernel::Memory {
size_t RamEmuGuestRanges(uint64_t* out, size_t max_pairs); // (memory.cpp)
}
namespace Libs::Graphics {
bool RamLineVulkan(uint64_t* video_blocks, uint64_t* system_blocks, uint64_t* ballast); // (vma.cpp)
}

namespace RamEmu {

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_RAM_EMU");
		return v == nullptr || v[0] != '0';
	}();
	return on;
}

struct Sample {
	uint64_t address, weight;
	uint32_t slot; // index into the allocations
};

inline void Line(std::unordered_map<uint64_t, uint64_t>* last_ram) {
	const auto t0 = std::chrono::steady_clock::now();
	static uint64_t guest[2 * 4096];
	const size_t    guest_n = Libs::LibKernel::Memory::RamEmuGuestRanges(guest, 4096);
	if (guest_n == 0) return;
	std::vector<std::pair<uint64_t, uint64_t>> owned;
	owned.reserve(guest_n);
	for (size_t i = 0; i < guest_n; i++) owned.emplace_back(guest[2 * i], guest[2 * i] + guest[2 * i + 1]);
	std::sort(owned.begin(), owned.end());
	const auto is_guest = [&](uint64_t a) {
		auto it = std::upper_bound(owned.begin(), owned.end(), std::make_pair(a, UINT64_MAX));
		if (it == owned.begin()) return false;
		--it;
		return a >= it->first && a < it->second;
	};
	std::unordered_set<uint64_t> heaps;
	{
		HANDLE list[256];
		const DWORD n = GetProcessHeaps(256, list);
		for (DWORD i = 0; i < n && i < 256; i++) heaps.insert(reinterpret_cast<uint64_t>(list[i]));
	}
	struct Alloc {
		uint64_t base = 0, commit = 0, ram = 0;
		DWORD    protect = 0;
		bool     heap = false, exec = false;
	};
	std::vector<Alloc>                     allocs;
	std::unordered_map<uint64_t, uint32_t> index;
	std::vector<Sample>                    host_samples, guest_samples;
	uint64_t                               guest_commit = 0;
	MEMORY_BASIC_INFORMATION               info {};
	for (uint64_t at = 0x10000; VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == sizeof(info);) {
		const uint64_t begin = reinterpret_cast<uint64_t>(info.BaseAddress), size = info.RegionSize;
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE && (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0) {
			if (is_guest(begin)) {
				guest_commit += size;
				for (uint64_t s = begin; s < begin + size && guest_samples.size() < (size_t {1} << 18u); s += uint64_t {1} << 20u)
					guest_samples.push_back({s, std::min<uint64_t>(uint64_t {1} << 20u, begin + size - s), 0});
			} else {
				const uint64_t base = reinterpret_cast<uint64_t>(info.AllocationBase);
				auto [it, fresh]    = index.try_emplace(base, static_cast<uint32_t>(allocs.size()));
				if (fresh) {
					Alloc a;
					a.base    = base;
					a.protect = info.AllocationProtect;
					if (heaps.count(base) != 0) a.heap = true;
					else if (begin == base && info.Protect == PAGE_READWRITE && size >= 0x1000) {
						// (an NT heap's further segment: header signature 0xFFEEFFEE at +0x10, as local-platform.cpp ram4)
						uint8_t head[0x18] {};
						SIZE_T  got = 0;
						uint32_t sig = 0;
						if (ReadProcessMemory(GetCurrentProcess(), info.BaseAddress, head, sizeof(head), &got) != 0 && got == sizeof(head)) {
							std::memcpy(&sig, head + 0x10, sizeof(sig));
							a.heap = sig == 0xFFEEFFEEu;
						}
					}
					allocs.push_back(a);
				}
				Alloc& a = allocs[it->second];
				a.commit += size;
				if ((info.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0) a.exec = true;
				for (uint64_t s = begin; s < begin + size && host_samples.size() < (size_t {1} << 19u); s += 0x10000)
					host_samples.push_back({s, std::min<uint64_t>(0x10000, begin + size - s), it->second});
			}
		}
		const uint64_t next = begin + size;
		if (next <= at) break;
		at = next;
	}
	std::vector<PSAPI_WORKING_SET_EX_INFORMATION> e(4096);
	uint64_t                                      guest_ram = 0;
	const auto query = [&](std::vector<Sample>& samples, bool host) {
		for (size_t at = 0; at < samples.size(); at += e.size()) {
			const size_t n = std::min(e.size(), samples.size() - at);
			for (size_t i = 0; i < n; i++) {
				e[i]                = {};
				e[i].VirtualAddress = reinterpret_cast<PVOID>(samples[at + i].address);
			}
			if (K32QueryWorkingSetEx(GetCurrentProcess(), e.data(), static_cast<DWORD>(n * sizeof(e[0]))) == 0) return false;
			for (size_t i = 0; i < n; i++)
				if ((e[i].VirtualAttributes.Flags & 1u) != 0) {
					if (host) allocs[samples[at + i].slot].ram += samples[at + i].weight;
					else guest_ram += samples[at + i].weight;
				}
			SwitchToThread();
		}
		return true;
	};
	if (!query(host_samples, true) || !query(guest_samples, false)) return;
	uint64_t host_ram = 0, host_commit = 0, heap_ram = 0, exec_ram = 0, small_ram = 0, mid_ram = 0, big_ram = 0;
	uint64_t small_n = 0, mid_n = 0, big_n = 0;
	for (const auto& a: allocs) {
		host_ram += a.ram;
		host_commit += a.commit;
		if (a.heap) heap_ram += a.ram;
		else if (a.exec) exec_ram += a.ram;
		else if (a.commit < (uint64_t {1} << 20u)) small_ram += a.ram, small_n++;
		else if (a.commit < (uint64_t {64} << 20u)) mid_ram += a.ram, mid_n++;
		else big_ram += a.ram, big_n++;
	}
	uint64_t vk_video = 0, vk_system = 0, ballast = 0;
	(void)Libs::Graphics::RamLineVulkan(&vk_video, &vk_system, &ballast);
	std::vector<uint32_t> top(allocs.size());
	for (uint32_t i = 0; i < top.size(); i++) top[i] = i;
	const size_t k = std::min<size_t>(8, top.size());
	std::partial_sort(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(k), top.end(),
	                  [&](uint32_t x, uint32_t y) { return allocs[x].ram > allocs[y].ram; });
	constexpr double MiB = 1024.0 * 1024.0;
	char             text[2400];
	int              used = std::snprintf(
        text, sizeof(text),
        "[ram-emu] 5 s, MiB en RAM (commit): juego %.0f (%.0f) | EMULADOR %.0f (%.0f) = monton C++ %.0f + codigo JIT/ejecutable %.0f "
        "+ bloques <1 MiB %.0f (%llu) + 1-64 MiB %.0f (%llu) + >=64 MiB %.0f (%llu); Vulkan en RAM del sistema (VMA) %.0f; mayores "
        "(base=RAM/commit, +crecido desde la linea anterior):",
        guest_ram / MiB, guest_commit / MiB, host_ram / MiB, host_commit / MiB, heap_ram / MiB, exec_ram / MiB, small_ram / MiB,
        static_cast<unsigned long long>(small_n), mid_ram / MiB, static_cast<unsigned long long>(mid_n), big_ram / MiB,
        static_cast<unsigned long long>(big_n), vk_system / MiB);
	std::unordered_map<uint64_t, uint64_t> now;
	for (size_t i = 0; i < allocs.size(); i++)
		if (allocs[i].ram >= (uint64_t {16} << 20u)) now[allocs[i].base] = allocs[i].ram;
	for (size_t i = 0; i < k && used > 0 && used < static_cast<int>(sizeof(text)) - 96; i++) {
		const Alloc&   a    = allocs[top[i]];
		const auto     prev = last_ram->find(a.base);
		const long long d   = prev == last_ram->end() ? 0 : (static_cast<long long>(a.ram) - static_cast<long long>(prev->second)) / (1 << 20);
		const char* owner = a.heap ? "monton" : a.exec ? "exec" : RamDueno::Find(a.base);
		used += std::snprintf(text + used, sizeof(text) - used, " 0x%llx=%.0f/%.0f[%s]%s%+lld", static_cast<unsigned long long>(a.base),
		                      a.ram / MiB, a.commit / MiB, owner != nullptr ? owner : "desconocido (driver?)",
		                      prev == last_ram->end() ? "(nuevo)" : "", d);
	}
	last_ram->swap(now);
	std::printf("%s; %zu+%zu muestras, %.1f ms\n", text, host_samples.size(), guest_samples.size(),
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
	std::fflush(stdout);
}

inline void Start() {
	if (!On()) return;
	static bool started = false;
	if (started) return;
	started = true;
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"Kyty.RamEmu");
		(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
		std::unordered_map<uint64_t, uint64_t> last;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(5));
			Line(&last);
		}
	}).detach();
}

} // namespace RamEmu
