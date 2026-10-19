#include "graphics/guest_gpu/threadSampler.h"

#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS

namespace Libs::Graphics::ThreadSampler {

void RegisterCurrentThread(const char* /*name*/) {}

} // namespace Libs::Graphics::ThreadSampler

#else

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <tlhelp32.h>

#include "common/hostException.h"
#include "common/liveSwitches.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ThreadSampler {

namespace {

struct Target {
	const char*                                  name   = nullptr;
	HANDLE                                       handle = nullptr;
	DWORD                                        id     = 0;
	uint64_t                                     samples = 0;
	std::unordered_map<uint64_t, uint64_t>       rips;    // exe offset -> count
	std::unordered_map<std::string, uint64_t>    outside; // module name -> count
	std::unordered_map<uint64_t, uint64_t>       callers; // CallerKey(first exe caller, module id) -> count
	std::unordered_map<std::string, uint64_t>    chains;  // "module|exe+a<exe+b<..." (up to 4 exe frames) -> count
	std::unordered_map<uint64_t, uint64_t>       guest;   // guest code (no module): 64-byte block of rip -> count
};

std::mutex          g_mutex; // NOLINT
std::vector<Target> g_targets; // NOLINT
std::once_flag      g_start; // NOLINT

uintptr_t g_image_base = 0;
uintptr_t g_image_end  = 0;

std::unordered_map<uintptr_t, std::string> g_page_modules; // NOLINT, sampler thread only

// Module ids for the "by first exe caller" table (sampler thread only). Id 0 is "exe".
std::vector<std::string>                     g_module_names {"exe"}; // NOLINT
std::unordered_map<std::string, uint32_t>    g_module_ids {{"exe", 0}}; // NOLINT

constexpr uint64_t kNoCaller = (uint64_t(1) << 48u) - 1; // no exe frame found within the walk limit
constexpr int      kMaxFrames = 32;

uint64_t CallerKey(uint64_t exe_offset, uint32_t module_id) {
	return (exe_offset << 16u) | module_id;
}

// Stack copy for unwinding. The stack is copied while the target is suspended (no user-mode
// locks: VirtualQuery/ReadProcessMemory are plain syscalls), then the thread is resumed and the
// copy is walked. Walking the live stack of a running thread races with it, and walking while it
// is suspended can deadlock inside RtlLookupFunctionEntry if the target holds the function table
// lock. g_stack has kStackSlack bytes after the copied part so an unwinder read slightly past the
// copy (large frame allocation in a bogus frame) still hits mapped memory.
constexpr size_t kStackCopyMax = size_t(512) * 1024;
constexpr size_t kStackSlack   = size_t(1024) * 1024;
uint8_t*         g_stack       = nullptr; // NOLINT

// Copies [rsp, end of rsp's committed region) (capped) into g_stack. Returns copied size or 0.
size_t CopyStack(uintptr_t rsp) {
	if (g_stack == nullptr || rsp == 0) {
		return 0;
	}
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<LPCVOID>(rsp), &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT ||
	    (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
	    (mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
	                    PAGE_EXECUTE_WRITECOPY)) == 0) {
		return 0;
	}
	const uintptr_t region_end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
	size_t          size       = std::min(static_cast<size_t>(region_end - rsp), kStackCopyMax);
	size &= ~static_cast<size_t>(7);
	SIZE_T read = 0;
	if (size == 0 ||
	    ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(rsp), g_stack, size, &read) == 0 ||
	    read != size) {
		return 0;
	}
	return size;
}

DWORD64* IntReg(CONTEXT& c, unsigned reg) {
	DWORD64* const regs[16] = {&c.Rax, &c.Rcx, &c.Rdx, &c.Rbx, &c.Rsp, &c.Rbp, &c.Rsi, &c.Rdi,
	                           &c.R8,  &c.R9,  &c.R10, &c.R11, &c.R12, &c.R13, &c.R14, &c.R15};
	return regs[reg & 15u];
}

// The unwinder reads the target's saved registers through pointers the sampled thread held; a
// sample taken mid-prologue/epilogue or in code using a frame register as a scratch register can
// make it read garbage. A fault there must end the walk, not the process.
bool SafeVirtualUnwind(DWORD64 image_base, PRUNTIME_FUNCTION fn, CONTEXT* ctx, PVOID* handler_data,
                       DWORD64* establisher) {
	// The emulator's vectored handler runs before this __except and would end the process.
	Common::HostException::SetIgnoreOnThisThread(true);
	bool ok = true;
	__try {
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx->Rip, fn, ctx, handler_data, establisher, nullptr);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		ok = false;
	}
	Common::HostException::SetIgnoreOnThisThread(false);
	return ok;
}

// Walks the copied stack from ctx (registers of the sampled thread at the sample) and returns the
// exe offset of the first frame inside the exe image, or kNoCaller.
uint64_t FirstExeCaller(const CONTEXT& sampled, uintptr_t orig_rsp, size_t copied, uint64_t* chain = nullptr,
                        int* chain_len = nullptr, int chain_max = 0) {
	const auto buf     = reinterpret_cast<uintptr_t>(g_stack);
	const auto buf_end = buf + copied;
	// Rebase every stack-pointing qword (saved Rbp chains etc.) and register into the copy.
	auto rebase = [&](DWORD64 v) -> DWORD64 {
		return (v >= orig_rsp && v < orig_rsp + copied) ? v - orig_rsp + buf : v;
	};
	for (size_t i = 0; i < copied; i += 8) {
		DWORD64 v = 0;
		std::memcpy(&v, g_stack + i, 8);
		const DWORD64 r = rebase(v);
		if (r != v) {
			std::memcpy(g_stack + i, &r, 8);
		}
	}
	CONTEXT ctx = sampled;
	for (unsigned reg = 0; reg < 16; reg++) {
		DWORD64* p = IntReg(ctx, reg);
		*p         = rebase(*p);
	}
	auto in_buf = [&](DWORD64 v, size_t bytes) { return v >= buf && v + bytes <= buf_end; };

	for (int frame = 0; frame < kMaxFrames; frame++) {
		const DWORD64 rsp = ctx.Rsp;
		if (!in_buf(rsp, 8)) {
			break;
		}
		DWORD64 image_base = 0;
		auto*   fn         = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
		if (fn == nullptr) {
			// Leaf (or code without unwind info): return address is at [Rsp].
			std::memcpy(&ctx.Rip, reinterpret_cast<const void*>(rsp), 8);
			ctx.Rsp = rsp + 8;
		} else {
			// If the function uses a frame register, the unwinder reads memory relative to it:
			// only proceed when it points into the copy.
			const RUNTIME_FUNCTION* primary = fn;
			if ((primary->UnwindData & 1u) != 0) {
				primary = reinterpret_cast<const RUNTIME_FUNCTION*>(image_base + (primary->UnwindData & ~DWORD(1)));
			}
			const auto* unwind_info = reinterpret_cast<const uint8_t*>(image_base + primary->UnwindData);
			const unsigned frame_reg = unwind_info[3] & 0x0fu;
			if (frame_reg != 0 && !in_buf(*IntReg(ctx, frame_reg), 8)) {
				break;
			}
			PVOID   handler_data = nullptr;
			DWORD64 establisher  = 0;
			if (!SafeVirtualUnwind(image_base, fn, &ctx, &handler_data, &establisher)) {
				break;
			}
		}
		if (ctx.Rip == 0 || ctx.Rsp <= rsp) {
			break;
		}
		const auto rip = static_cast<uintptr_t>(ctx.Rip);
		if (rip >= g_image_base && rip < g_image_end) {
			if (chain == nullptr) {
				return rip - g_image_base;
			}
			chain[(*chain_len)++] = rip - g_image_base;
			if (*chain_len >= chain_max) {
				break;
			}
		}
	}
	if (chain != nullptr && *chain_len > 0) {
		return chain[0];
	}
	return kNoCaller;
}

const std::string& ModuleName(uintptr_t rip) {
	const uintptr_t page = rip & ~static_cast<uintptr_t>(0xfff);
	auto            it   = g_page_modules.find(page);
	if (it != g_page_modules.end()) {
		return it->second;
	}
	std::string name = "outside:?";
	HMODULE     mod  = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(rip), &mod) != 0 &&
	    mod != nullptr) {
		char path[MAX_PATH] = {};
		if (GetModuleFileNameA(mod, path, MAX_PATH) != 0) {
			const char* base = path;
			for (const char* p = path; *p != '\0'; p++) {
				if (*p == '\\' || *p == '/') {
					base = p + 1;
				}
			}
			name = std::string("outside:") + base;
		}
	} else {
		name = "outside:<no module>";
	}
	return g_page_modules.emplace(page, std::move(name)).first->second;
}

template <class K>
std::vector<std::pair<K, uint64_t>> Sorted(const std::unordered_map<K, uint64_t>& map) {
	std::vector<std::pair<K, uint64_t>> v(map.begin(), map.end());
	std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
	return v;
}

void Print(Target& t) {
	::printf("Profile (5 s) %s: %" PRIu64 " samples\n", t.name, t.samples);
	if (t.samples != 0) {
		const double total = static_cast<double>(t.samples);
		auto         rips  = Sorted(t.rips);
		for (size_t i = 0; i < rips.size() && i < 40; i++) {
			::printf("  %5.2f%%  exe+0x%" PRIx64 "\n", 100.0 * static_cast<double>(rips[i].second) / total, rips[i].first);
		}
		std::unordered_map<uint64_t, uint64_t> buckets;
		for (const auto& [off, n]: t.rips) {
			buckets[off >> 8u] += n;
		}
		auto sorted_buckets = Sorted(buckets);
		::printf(" %s by 256-byte bucket:\n", t.name);
		for (size_t i = 0; i < sorted_buckets.size() && i < 25; i++) {
			::printf("  %5.2f%%  exe+0x%" PRIx64 "..0x%" PRIx64 "\n",
			         100.0 * static_cast<double>(sorted_buckets[i].second) / total, sorted_buckets[i].first << 8u,
			         (sorted_buckets[i].first << 8u) + 0xffu);
		}
		auto outside = Sorted(t.outside);
		for (const auto& [mod, n]: outside) {
			::printf("  %5.2f%%  %s\n", 100.0 * static_cast<double>(n) / total, mod.c_str());
		}
		auto callers = Sorted(t.callers);
		::printf("  by first exe caller:\n");
		for (size_t i = 0; i < callers.size() && i < 30; i++) {
			const uint64_t off    = callers[i].first >> 16u;
			const auto     mod_id = static_cast<uint32_t>(callers[i].first & 0xffffu);
			const char*    mod    = mod_id < g_module_names.size() ? g_module_names[mod_id].c_str() : "?";
			const double   pct    = 100.0 * static_cast<double>(callers[i].second) / total;
			if (off == kNoCaller) {
				::printf("  %5.2f%%  <no exe caller>  [%s]\n", pct, mod);
			} else {
				::printf("  %5.2f%%  exe+0x%" PRIx64 "  [%s]\n", pct, off, mod);
			}
		}
		auto guest = Sorted(t.guest);
		if (!guest.empty()) {
			// Guest code by 64-byte block (absolute guest addresses): one hot block is a spin loop.
			::printf("  guest code by 64-byte block:\n");
			for (size_t i = 0; i < guest.size() && i < 30; i++) {
				::printf("  %5.2f%%  guest 0x%" PRIx64 "\n", 100.0 * static_cast<double>(guest[i].second) / total,
				         guest[i].first << 6u);
			}
			std::unordered_map<uint64_t, uint64_t> pages;
			for (const auto& [block, n]: t.guest) {
				pages[block >> 6u] += n;
			}
			auto sorted_pages = Sorted(pages);
			::printf("  guest code by 4 KiB page:\n");
			for (size_t i = 0; i < sorted_pages.size() && i < 20; i++) {
				::printf("  %5.2f%%  guest 0x%" PRIx64 "\n",
				         100.0 * static_cast<double>(sorted_pages[i].second) / total, sorted_pages[i].first << 12u);
			}
		}
		auto chains = Sorted(t.chains);
		::printf("  by call chain (sample frame < its callers):\n");
		for (size_t i = 0; i < chains.size() && i < 30; i++) {
			::printf("  %5.2f%%  %s\n", 100.0 * static_cast<double>(chains[i].second) / total, chains[i].first.c_str());
		}
	}
	t.samples = 0;
	t.rips.clear();
	t.outside.clear();
	t.callers.clear();
	t.chains.clear();
	t.guest.clear();
}

void SamplerThread() {
	static auto& prof = Common::LiveSwitches::Get("KYTY_PROF", 0);
	auto report_time = std::chrono::steady_clock::now();
	bool was_on      = false;
	for (;;) {
		const bool on = prof.load(std::memory_order_relaxed) != 0;
		if (!on) {
			was_on = false;
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}
		const auto now = std::chrono::steady_clock::now();
		if (!was_on) {
			was_on      = true;
			report_time = now;
		}
		{
			std::lock_guard lock(g_mutex);
			for (auto& t: g_targets) {
				if (t.handle == nullptr) {
					continue;
				}
				if (SuspendThread(t.handle) == static_cast<DWORD>(-1)) {
					continue;
				}
				CONTEXT ctx {};
				ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
				const BOOL   ok     = GetThreadContext(t.handle, &ctx);
				const auto   rip    = static_cast<uintptr_t>(ctx.Rip);
				const bool   in_exe = rip >= g_image_base && rip < g_image_end;
				// While suspended: only syscalls and memcpy, no heap or loader locks.
				// Unwinding from an arbitrary exe instruction crashed the process (build 1108695):
				// only samples outside the exe (waits, driver, CRT), which sit at call sites, are
				// walked.
				const size_t copied = (ok != 0 && !in_exe) ? CopyStack(static_cast<uintptr_t>(ctx.Rsp)) : 0;
				ResumeThread(t.handle);
				if (ok == 0) {
					continue;
				}
				// Nothing below may take a lock the target could hold: it is already resumed.
				t.samples++;
				uint64_t caller    = kNoCaller;
				uint32_t module_id = 0;
				if (in_exe) {
					caller = rip - g_image_base;
					t.rips[caller]++;
				} else {
					const std::string& mod = ModuleName(rip);
					t.outside[mod]++;
					if (mod == "outside:<no module>") {
						t.guest[rip >> 6u]++;
					}
					const std::string short_name = mod.compare(0, 8, "outside:") == 0 ? mod.substr(8) : mod;
					auto              it         = g_module_ids.find(short_name);
					if (it == g_module_ids.end()) {
						it = g_module_ids.emplace(short_name, static_cast<uint32_t>(g_module_names.size())).first;
						g_module_names.push_back(short_name);
					}
					module_id = it->second;
					if (copied != 0) {
						caller = FirstExeCaller(ctx, static_cast<uintptr_t>(ctx.Rsp), copied);
					}
				}
				t.callers[CallerKey(caller, module_id)]++;
				{
					// Call chain: the sample's exe frame (if any) plus the exe frames above it.
					uint64_t chain[4] {};
					int      len = 0;
					if (in_exe) {
						chain[len++] = rip - g_image_base;
					}
					if (copied != 0) {
						FirstExeCaller(ctx, static_cast<uintptr_t>(ctx.Rsp), copied, chain, &len, 4);
					}
					std::string key = g_module_names[module_id] + "|";
					char        buf[24];
					for (int i = 0; i < len; i++) {
						std::snprintf(buf, sizeof(buf), "%sexe+0x%" PRIx64, i == 0 ? "" : "<", chain[i]);
						key += buf;
					}
					t.chains[key]++;
				}
			}
			if (now - report_time >= std::chrono::seconds(5)) {
				report_time = now;
				for (auto& t: g_targets) {
					Print(t);
				}
				::fflush(stdout);
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

// KYTY_CPU_LOAD=1 (live, default 0): every 5 s prints the CPU use of the busiest threads of the
// process (user+kernel time from GetThreadTimes over wall time; 100% = one full core) with their
// names. A thread near 100% is a serial bottleneck; if none is, the frame waits on the GPU or on
// synchronization. Costs nothing while off (one relaxed load every 200 ms).
void CpuLoadThread() {
	static auto& on = Common::LiveSwitches::Get("KYTY_CPU_LOAD", 0);
	std::unordered_map<DWORD, uint64_t> last; // thread id -> 100 ns CPU time
	auto                               last_time = std::chrono::steady_clock::now();
	bool                               primed    = false;
	const DWORD                        pid       = GetCurrentProcessId();
	for (;;) {
		if (on.load(std::memory_order_relaxed) == 0) {
			primed = false;
			last.clear();
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			continue;
		}
		const auto now  = std::chrono::steady_clock::now();
		const double wall_100ns = std::chrono::duration<double>(now - last_time).count() * 1e7;
		last_time = now;
		std::vector<std::pair<double, std::string>> rows;
		std::unordered_map<DWORD, uint64_t>         cur;
		double                                      total = 0.0;
		HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (snap != INVALID_HANDLE_VALUE) {
			THREADENTRY32 te {};
			te.dwSize = sizeof(te);
			for (BOOL ok = Thread32First(snap, &te); ok != 0; ok = Thread32Next(snap, &te)) {
				if (te.th32OwnerProcessID != pid) {
					continue;
				}
				HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
				if (h == nullptr) {
					continue;
				}
				FILETIME c {}, e {}, k {}, u {};
				if (GetThreadTimes(h, &c, &e, &k, &u) != 0) {
					const uint64_t t = (static_cast<uint64_t>(k.dwHighDateTime) << 32u | k.dwLowDateTime) +
					                   (static_cast<uint64_t>(u.dwHighDateTime) << 32u | u.dwLowDateTime);
					cur[te.th32ThreadID] = t;
					auto it = last.find(te.th32ThreadID);
					if (primed && it != last.end() && wall_100ns > 0.0) {
						const double pct = 100.0 * static_cast<double>(t - it->second) / wall_100ns;
						total += pct;
						if (pct >= 3.0) {
							std::string name = "tid " + std::to_string(te.th32ThreadID);
							PWSTR desc = nullptr;
							if (SUCCEEDED(GetThreadDescription(h, &desc)) && desc != nullptr) {
								if (desc[0] != 0) {
									char buf[128] {};
									WideCharToMultiByte(CP_UTF8, 0, desc, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
									name = buf;
								}
								LocalFree(desc);
							}
							std::lock_guard lock(g_mutex);
							for (const auto& tg: g_targets) {
								if (tg.id == te.th32ThreadID) {
									name = tg.name;
								}
							}
							rows.emplace_back(pct, name);
						}
					}
				}
				CloseHandle(h);
			}
			CloseHandle(snap);
		}
		if (primed) {
			std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
			::printf("CPU load (5 s): process %.0f%% of one core; busiest threads:\n", total);
			for (size_t i = 0; i < rows.size() && i < 12; i++) {
				::printf("  %5.1f%%  %s\n", rows[i].first, rows[i].second.c_str());
			}
			::fflush(stdout);
		}
		last   = std::move(cur);
		primed = true;
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
}

void Start() {
	auto* base   = reinterpret_cast<const uint8_t*>(GetModuleHandleA(nullptr));
	g_image_base = reinterpret_cast<uintptr_t>(base);
	const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
	const auto* nt  = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
	g_image_end     = g_image_base + nt->OptionalHeader.SizeOfImage;
	g_stack = static_cast<uint8_t*>(VirtualAlloc(nullptr, kStackCopyMax + kStackSlack, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	std::thread(SamplerThread).detach();
	std::thread(CpuLoadThread).detach();
}

} // namespace

void RegisterCurrentThread(const char* name) {
	std::call_once(g_start, Start);
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0) == 0) {
		return;
	}
	std::lock_guard lock(g_mutex);
	for (auto& t: g_targets) {
		if (std::strcmp(t.name, name) == 0) {
			// A restarted thread (the recording thread) replaces its earlier registration.
			if (t.handle != nullptr) {
				CloseHandle(t.handle);
			}
			t.handle = handle;
			t.id     = GetCurrentThreadId();
			return;
		}
	}
	Target t;
	t.name   = name;
	t.handle = handle;
	t.id     = GetCurrentThreadId();
	g_targets.push_back(std::move(t));
}

} // namespace Libs::Graphics::ThreadSampler

#endif
