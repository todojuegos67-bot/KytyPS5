#include "local-platform.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iterator>
#include <map>
#include <set>
#include <tuple>
#include <mutex>
#include <unordered_map>
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
#include <dxgi1_4.h>
#include <psapi.h>
#include <softpub.h>
#include <tlhelp32.h>
#include <wincrypt.h>
#include <wintrust.h>
#pragma comment(lib, "wintrust.lib")
#pragma comment(lib, "crypt32.lib")
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <string>
#include <chrono>
#include <thread>
#else
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <x86intrin.h>
#endif

namespace LocalPlatform {

// "1,2,3" -> bit mask of CPUs 0..63; 0 for an empty or invalid list.
static uint64_t CpuListMask(const char* list) {
	uint64_t mask = 0;
	if (list == nullptr) return 0;
	for (const char* p = list; *p != '\0';) {
		char*      end = nullptr;
		const long cpu = std::strtol(p, &end, 10);
		if (end == p) break;
		if (cpu >= 0 && cpu < 64) mask |= uint64_t(1) << cpu;
		p = (*end == ',') ? end + 1 : end;
	}
	return mask;
}

#if defined(_WIN32)

static double FileTimeSeconds(const FILETIME& time) {
	const uint64_t ticks = (uint64_t(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	return static_cast<double>(ticks) / 1e7;
}

static double HandleCpuSeconds(HANDLE thread) {
	FILETIME creation {}, exit {}, kernel {}, user {};
	if (GetThreadTimes(thread, &creation, &exit, &kernel, &user) == 0) return 0;
	return FileTimeSeconds(kernel) + FileTimeSeconds(user);
}

void SetThreadName(const char* name) {
	wchar_t wide[64] {};
	for (size_t i = 0; name[i] != '\0' && i + 1 < std::size(wide); ++i) wide[i] = static_cast<wchar_t>(name[i]);
	(void)SetThreadDescription(GetCurrentThread(), wide);
}

void ReleaseRedirectedConsole() {
	if (GetConsoleWindow() != nullptr && GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_DISK &&
	    GetFileType(GetStdHandle(STD_ERROR_HANDLE)) == FILE_TYPE_DISK)
		(void)FreeConsole();
}

static HANDLE                g_stdout_done = nullptr; // set once the writer has written all the pipe held

void AsyncStdout() {
	const HANDLE original = GetStdHandle(STD_OUTPUT_HANDLE);
	if (original == nullptr || original == INVALID_HANDLE_VALUE || GetFileType(original) != FILE_TYPE_DISK) return;
	// (Its own handle to the file: replacing descriptor 1 closes the one the C runtime holds.)
	HANDLE file = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), original, GetCurrentProcess(), &file, 0, FALSE, DUPLICATE_SAME_ACCESS) == 0)
		return;
	HANDLE read_end = nullptr, write_end = nullptr;
	if (CreatePipe(&read_end, &write_end, nullptr, 4u << 20u) == 0) {
		CloseHandle(file);
		return;
	}
	const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(write_end), _O_TEXT);
	if (fd < 0) {
		CloseHandle(read_end);
		CloseHandle(write_end);
		CloseHandle(file);
		return;
	}
	std::fflush(stdout);
	if (_dup2(fd, _fileno(stdout)) != 0) {
		_close(fd);
		CloseHandle(read_end);
		CloseHandle(file);
		return;
	}
	_close(fd);
	(void)SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdout))));
	g_stdout_done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
	std::thread([read_end, file] {
		SetThreadName("Kyty.Stdout");
		std::vector<char> buffer(256u << 10u);
		for (DWORD read = 0; ReadFile(read_end, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) != 0 && read != 0;)
			for (DWORD written = 0, at = 0; at < read; at += written)
				if (WriteFile(file, buffer.data() + at, read - at, &written, nullptr) == 0 || written == 0) break;
		SetEvent(g_stdout_done);
	}).detach();
}

void DrainStdout() {
	static std::atomic<bool> closed {false};
	std::fflush(stdout);
	if (g_stdout_done == nullptr) return;
	// The pipe cannot be polled: a request on its read end waits behind the writer's pending read (a crash report
	// hung there, 10-06). Standard output goes to NUL instead, which closes the pipe's only write end: the writer's
	// read ends once it has taken all the pipe held (not _close: a later write to a closed descriptor would end
	// the process before the writer is done). Bounded: a crash must not hang the end.
	if (!closed.exchange(true)) {
		if (const int nul = _open("NUL", _O_WRONLY); nul >= 0) {
			(void)_dup2(nul, _fileno(stdout));
			_close(nul);
		}
	}
	WaitForSingleObject(g_stdout_done, 2000);
}

uint32_t ThreadId() {
	return static_cast<uint32_t>(GetCurrentThreadId());
}

void PinThreadToCpuList(const char* list) {
	if (const auto mask = CpuListMask(list); mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(mask));
}

bool PinThreadHandleToCpuList(uint64_t handle, const char* list) {
	const auto mask = CpuListMask(list);
	return handle != 0 && mask != 0 &&
	       SetThreadAffinityMask(reinterpret_cast<HANDLE>(handle), DWORD_PTR(mask)) != 0;
}

// CPU set ids of the logical processors in the mask (all of them for mask 0 = no restriction).
static bool SetProcessDefaultCpuMask(uint64_t mask) {
	if (mask == 0) return SetProcessDefaultCpuSets(GetCurrentProcess(), nullptr, 0) != 0;
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	auto* first = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
	if (length == 0 || !GetSystemCpuSetInformation(first, length, &length, GetCurrentProcess(), 0)) return false;
	std::vector<ULONG> ids;
	for (ULONG offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 && info->CpuSet.LogicalProcessorIndex < 64 &&
		    (mask >> info->CpuSet.LogicalProcessorIndex) & 1u)
			ids.push_back(info->CpuSet.Id);
		offset += info->Size;
	}
	return !ids.empty() && SetProcessDefaultCpuSets(GetCurrentProcess(), ids.data(), static_cast<ULONG>(ids.size())) != 0;
}

bool SetProcessDefaultCpuList(const char* list) {
	return SetProcessDefaultCpuMask(CpuListMask(list));
}

uint64_t CurrentThreadHandle() {
	HANDLE handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION | THREAD_SUSPEND_RESUME |
	                               THREAD_GET_CONTEXT,
	                           FALSE, GetCurrentThreadId());
	return reinterpret_cast<uint64_t>(handle);
}

uint64_t OpenThreadForSampling(uint32_t thread_id) {
	HANDLE handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION | THREAD_SUSPEND_RESUME |
	                               THREAD_GET_CONTEXT,
	                           FALSE, thread_id);
	return reinterpret_cast<uint64_t>(handle);
}

void CloseThreadForSampling(uint64_t handle) {
	if (handle != 0) CloseHandle(reinterpret_cast<HANDLE>(handle));
}

// Unwind tables of the loaded images, collected before sampling: the sampler must not take
// loader or function-table locks (RtlLookupFunctionEntry does) while the sampled thread,
// which may hold them, is suspended. RtlVirtualUnwind itself takes no lock.
namespace {
struct UnwindImage {
	uint64_t                base = 0, end = 0;
	const RUNTIME_FUNCTION* table = nullptr;
	size_t                  count = 0;
};
std::vector<UnwindImage> g_unwind_images;

const RUNTIME_FUNCTION* FindUnwindEntry(uint64_t pc, uint64_t* image_base) {
	for (const auto& image: g_unwind_images) {
		if (pc < image.base || pc >= image.end) continue;
		const auto rva = static_cast<DWORD>(pc - image.base);
		size_t     lo = 0, hi = image.count;
		while (lo < hi) {
			const size_t mid = (lo + hi) / 2;
			if (image.table[mid].EndAddress <= rva) lo = mid + 1;
			else hi = mid;
		}
		if (lo == image.count || image.table[lo].BeginAddress > rva) return nullptr;
		*image_base = image.base;
		// Chained entries (UNW_FLAG_CHAININFO) are resolved by UnwindFrame.
		return &image.table[lo];
	}
	return nullptr;
}

// One frame of RtlVirtualUnwind's algorithm (epilogue emulation, else the unwind codes of the
// function and its chain) with every stack read inside [low, high): a sample can catch a thread at
// an instruction its unwind data does not describe, and RtlVirtualUnwind then reads wherever a
// stale frame register points (a profiling run died reading 3.6 KiB above the render stack's top).
// XMM saves are not read (they never steer the unwind). False ends the walk.
bool UnwindFrame(uint64_t image_base, const RUNTIME_FUNCTION* entry, CONTEXT* context, uint64_t low, uint64_t high) {
	// x64 unwind operation codes (not in the SDK headers).
	enum : uint32_t {
		UWOP_PUSH_NONVOL      = 0,
		UWOP_ALLOC_LARGE      = 1,
		UWOP_ALLOC_SMALL      = 2,
		UWOP_SET_FPREG        = 3,
		UWOP_SAVE_NONVOL      = 4,
		UWOP_SAVE_NONVOL_FAR  = 5,
		UWOP_SAVE_XMM128      = 8,
		UWOP_SAVE_XMM128_FAR  = 9,
		UWOP_PUSH_MACHFRAME   = 10,
	};
	DWORD64*   registers = &context->Rax;
	const auto read      = [&](uint64_t address, uint64_t* value) {
        if (address < low || high < 8 || address > high - 8) return false;
        *value = *reinterpret_cast<const uint64_t*>(address);
        return true;
	};
	const uint64_t pc    = context->Rip;
	const auto*    info  = reinterpret_cast<const uint8_t*>(image_base + entry->UnwindData);
	const uint32_t count = info[2], frame_register = info[3] & 0x0fu;
	const auto*    codes = reinterpret_cast<const uint16_t*>(info + 4);
	if ((info[0] & 7u) != 1 && (info[0] & 7u) != 2) return false;
	const auto slots = [](uint32_t op, uint32_t op_info) -> uint32_t {
		switch (op) {
			case UWOP_ALLOC_LARGE: return op_info != 0 ? 3 : 2;
			case UWOP_SAVE_NONVOL:
			case UWOP_SAVE_XMM128: return 2;
			case UWOP_SAVE_NONVOL_FAR:
			case UWOP_SAVE_XMM128_FAR: return 3;
			default: return 1; // version 2 epilogue descriptors take one slot each
		}
	};
	auto prolog_offset = static_cast<uint32_t>(pc - (image_base + entry->BeginAddress));
	uint64_t frame     = context->Rsp;
	if (frame_register != 0) {
		bool established = prolog_offset >= info[1] || ((info[0] >> 3) & UNW_FLAG_CHAININFO) != 0;
		for (uint32_t i = 0; !established && i < count; i += slots((codes[i] >> 8) & 0x0fu, codes[i] >> 12)) {
			if (((codes[i] >> 8) & 0x0fu) == UWOP_SET_FPREG) {
				established = prolog_offset >= (codes[i] & 0xffu);
				break;
			}
		}
		if (established) frame = registers[frame_register] - (info[3] >> 4u) * 16ull;
	}
	// An epilogue (add rsp / lea rsp,[fp], pops, then ret or a jump out) is emulated forward.
	const auto* next          = reinterpret_cast<const uint8_t*>(pc);
	const auto  rex           = [](uint8_t byte) { return (byte & 0xf0u) == 0x40u; };
	uint64_t    epilogue_rsp  = context->Rsp;
	if (next[0] == 0x48 && next[1] == 0x83 && next[2] == 0xc4) {
		epilogue_rsp += static_cast<int8_t>(next[3]);
		next += 4;
	} else if (next[0] == 0x48 && next[1] == 0x81 && next[2] == 0xc4) {
		int32_t displacement = 0;
		std::memcpy(&displacement, next + 3, 4);
		epilogue_rsp += displacement;
		next += 7;
	} else if ((next[0] & 0xfeu) == 0x48 && next[1] == 0x8d) {
		const uint32_t base = ((next[0] & 1u) << 3u) | (next[2] & 7u);
		if (base != 0 && base == frame_register) {
			if ((next[2] & 0xf8u) == 0x60) {
				epilogue_rsp = registers[base] + static_cast<int8_t>(next[3]);
				next += 4;
			} else if ((next[2] & 0xf8u) == 0xa0) {
				int32_t displacement = 0;
				std::memcpy(&displacement, next + 3, 4);
				epilogue_rsp = registers[base] + displacement;
				next += 7;
			}
		}
	}
	const auto* pops = next;
	while ((next[0] & 0xf8u) == 0x58 || (rex(next[0]) && (next[1] & 0xf8u) == 0x58)) next += (next[0] & 0xf8u) == 0x58 ? 1 : 2;
	bool epilogue = next[0] == 0xc3 || next[0] == 0xc2 || (next[0] == 0xf3 && next[1] == 0xc3) ||
	                (next[0] == 0xff && next[1] == 0x25) || (rex(next[0]) && next[1] == 0xff && (next[2] & 0x38u) == 0x20);
	if (next[0] == 0xeb || next[0] == 0xe9) {
		// A jump out of the function (or to its start) is a tail call.
		int32_t displacement = 0;
		if (next[0] == 0xeb) displacement = static_cast<int8_t>(next[1]);
		else std::memcpy(&displacement, next + 1, 4);
		const uint64_t target = reinterpret_cast<uint64_t>(next) + (next[0] == 0xeb ? 2 : 5) + displacement - image_base;
		epilogue = target < entry->BeginAddress || target >= entry->EndAddress ||
		           (target == entry->BeginAddress && ((info[0] >> 3) & UNW_FLAG_CHAININFO) == 0);
	}
	if (epilogue) {
		uint64_t rsp = epilogue_rsp;
		for (next = pops; (next[0] & 0xf8u) == 0x58 || (rex(next[0]) && (next[1] & 0xf8u) == 0x58);) {
			const bool     extended = (next[0] & 0xf8u) != 0x58;
			const uint32_t reg      = (extended ? ((next[0] & 1u) << 3u) | (next[1] & 7u) : (next[0] & 7u));
			uint64_t       value    = 0;
			if (!read(rsp, &value)) return false;
			registers[reg] = value;
			rsp += 8;
			next += extended ? 2 : 1;
		}
		uint64_t ret = 0;
		if (!read(rsp, &ret)) return false;
		context->Rip = ret;
		context->Rsp = rsp + 8;
		return true;
	}
	// The unwind codes the thread has executed, through the chain of the function's fragments.
	bool machine_frame = false;
	for (int chain = 0; chain < 32; ++chain) {
		info                 = reinterpret_cast<const uint8_t*>(image_base + entry->UnwindData);
		codes                = reinterpret_cast<const uint16_t*>(info + 4);
		const uint32_t total = info[2];
		prolog_offset        = static_cast<uint32_t>(pc - (image_base + entry->BeginAddress));
		for (uint32_t i = 0; i < total;) {
			const uint32_t op = (codes[i] >> 8) & 0x0fu, op_info = codes[i] >> 12;
			if (prolog_offset < (codes[i] & 0xffu) || op == UWOP_SAVE_XMM128 || op == UWOP_SAVE_XMM128_FAR ||
			    op == 6 /* version 2 epilogue descriptor */) {
				i += slots(op, op_info);
				continue;
			}
			uint64_t value = 0;
			switch (op) {
				case UWOP_PUSH_NONVOL:
					if (!read(context->Rsp, &value)) return false;
					registers[op_info] = value;
					context->Rsp += 8;
					break;
				case UWOP_ALLOC_LARGE:
					if (i + slots(op, op_info) > total) return false;
					context->Rsp += op_info != 0 ? codes[i + 1] | (uint64_t(codes[i + 2]) << 16u) : codes[i + 1] * 8ull;
					break;
				case UWOP_ALLOC_SMALL: context->Rsp += op_info * 8ull + 8; break;
				case UWOP_SET_FPREG: context->Rsp = registers[info[3] & 0x0fu] - (info[3] >> 4u) * 16ull; break;
				case UWOP_SAVE_NONVOL:
				case UWOP_SAVE_NONVOL_FAR: {
					if (i + slots(op, op_info) > total) return false;
					const uint64_t offset = op == UWOP_SAVE_NONVOL ? codes[i + 1] * 8ull
					                                               : codes[i + 1] | (uint64_t(codes[i + 2]) << 16u);
					if (!read(frame + offset, &value)) return false;
					registers[op_info] = value;
					break;
				}
				case UWOP_PUSH_MACHFRAME: {
					const uint64_t base = context->Rsp + (op_info != 0 ? 8 : 0);
					uint64_t       rsp  = 0;
					if (!read(base, &value) || !read(base + 24, &rsp)) return false;
					context->Rip  = value;
					context->Rsp  = rsp;
					machine_frame = true;
					break;
				}
				default: return false;
			}
			i += slots(op, op_info);
		}
		if (((info[0] >> 3) & UNW_FLAG_CHAININFO) == 0) break;
		entry = reinterpret_cast<const RUNTIME_FUNCTION*>(codes + ((total + 1) & ~1u));
	}
	if (machine_frame) return true;
	uint64_t ret = 0;
	if (!read(context->Rsp, &ret)) return false;
	context->Rip = ret;
	context->Rsp += 8;
	return true;
}
} // namespace

void PrepareSampling() {
	g_unwind_images.clear();
	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
	if (snapshot == INVALID_HANDLE_VALUE) return;
	MODULEENTRY32W entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Module32FirstW(snapshot, &entry); ok; ok = Module32NextW(snapshot, &entry)) {
		const auto base = reinterpret_cast<uint64_t>(entry.modBaseAddr);
		const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE ||
		    nt->OptionalHeader.NumberOfRvaAndSizes <= IMAGE_DIRECTORY_ENTRY_EXCEPTION)
			continue;
		const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
		if (directory.VirtualAddress == 0 || directory.Size < sizeof(RUNTIME_FUNCTION)) continue;
		g_unwind_images.push_back({base, base + entry.modBaseSize,
		                           reinterpret_cast<const RUNTIME_FUNCTION*>(base + directory.VirtualAddress),
		                           directory.Size / sizeof(RUNTIME_FUNCTION)});
	}
	CloseHandle(snapshot);
}

bool SampleThread(uint64_t handle, uint64_t stack_low, uint64_t stack_high, uint64_t* out, size_t words) {
	auto* thread = reinterpret_cast<HANDLE>(handle);
	if (thread == nullptr || words < 2 || SuspendThread(thread) == static_cast<DWORD>(-1)) return false;
	CONTEXT context {};
	context.ContextFlags = CONTEXT_FULL;
	const bool got       = GetThreadContext(thread, &context) != 0;
	if (got && stack_high == 0) {
		// Unknown stack (a guest thread on its own stack): walk only inside the committed
		// region that holds rsp, so no read can fault while the thread is suspended.
		MEMORY_BASIC_INFORMATION region {};
		if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &region, sizeof(region)) != 0 &&
		    region.State == MEM_COMMIT) {
			stack_low  = context.Rsp;
			stack_high = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
		}
	}
	if (got) {
		// Layout of the Linux SIGPROF sampler: pc, then callers. With unwind tables
		// (PrepareSampling) the callers come from the unwind codes (UnwindFrame), so frames
		// without a frame pointer (ntdll, the driver, leaf code) are walked too; code without
		// unwind data (guest code) ends the walk with the word at rsp and the frame-pointer chain.
		// Only [rsp, stack top) is read: the pages below rsp may be uncommitted.
		stack_low = std::max(stack_low, static_cast<uint64_t>(context.Rsp));
		for (size_t i = 0; i < words; ++i) out[i] = 0;
		out[0]      = context.Rip;
		size_t next = 1;
		for (int depth = 0; next < words && depth < 64; ++depth) {
			const uint64_t rsp = context.Rsp;
			if (rsp < stack_low || rsp + 8 > stack_high) break;
			uint64_t    image_base = 0;
			const auto* entry      = g_unwind_images.empty() ? nullptr : FindUnwindEntry(context.Rip, &image_base);
			if (entry == nullptr) {
				// No unwind data: the word at rsp (a leaf's caller), then frame pointers.
				if (next < words) out[next++] = *reinterpret_cast<const uint64_t*>(rsp);
				uint64_t rbp = context.Rbp;
				while (next < words) {
					if (rbp < rsp || rbp >= stack_high || stack_high - rbp < 16 || (rbp & 7u) != 0) break;
					const auto* frame = reinterpret_cast<const uint64_t*>(rbp);
					out[next++]       = frame[1];
					rbp               = frame[0] <= rbp ? 0 : frame[0];
				}
				break;
			}
			if (!UnwindFrame(image_base, entry, &context, stack_low, stack_high) || context.Rip == 0 ||
			    context.Rsp <= rsp)
				break;
			out[next++] = context.Rip;
		}
	}
	ResumeThread(thread);
	return got;
}

bool ModuleOf(const void* address, uint64_t* base, char* path, size_t path_size) {
	HMODULE module = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       static_cast<LPCSTR>(address), &module) == 0)
		return false;
	*base = reinterpret_cast<uint64_t>(module);
	if (path_size != 0 && GetModuleFileNameA(module, path, static_cast<DWORD>(path_size)) == 0) path[0] = '\0';
	return true;
}

double ThreadCpuSeconds(uint64_t handle) {
	return handle != 0 ? HandleCpuSeconds(reinterpret_cast<HANDLE>(handle)) : 0;
}

uint64_t CurrentThreadCycles() {
	ULONG64 cycles = 0;
	return QueryThreadCycleTime(GetCurrentThread(), &cycles) != 0 ? cycles : 0;
}

int CurrentThreadPriority() {
	return GetThreadPriority(GetCurrentThread());
}

namespace {
struct KeptFile {
	HANDLE   handle     = INVALID_HANDLE_VALUE;
	uint64_t size       = 0;
	bool     prefetched = false;
	bool     ahead      = false; // opened by FollowKeptFiles's threads, not read by the game yet
};
std::mutex                                g_kept_files_mutex;
std::unordered_map<std::string, KeptFile> g_kept_files; // (never closed: the game's files, read-only)

std::wstring WidePath(const std::string& path) {
	const int wide_size = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
	if (wide_size <= 0) return {};
	std::wstring wide(static_cast<size_t>(wide_size), L' ');
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), wide_size);
	return wide;
}

// PrefetchKeptFile's thread: the files queued, each read through once in 1 MiB pieces at background priority (low I/O
// priority too: the game's own reads go first) and by a handle of its own (a synchronous handle runs one read at a
// time, so the game's next read would wait behind a piece of this one; the cache is the file's, whatever the handle).
std::mutex              g_prefetch_mutex;
std::condition_variable g_prefetch_ready;
std::vector<std::wstring> g_prefetch_queue;

void PrefetchThread() {
	SetThreadName("Kyty.FilePrefetch");
	SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_BEGIN);
	std::vector<uint8_t> piece(1u << 20u);
	for (;;) {
		std::wstring path;
		{
			std::unique_lock lock(g_prefetch_mutex);
			g_prefetch_ready.wait(lock, [] { return !g_prefetch_queue.empty(); });
			path = std::move(g_prefetch_queue.front());
			g_prefetch_queue.erase(g_prefetch_queue.begin());
		}
		HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
		                          FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
		if (file == INVALID_HANDLE_VALUE) continue;
		DWORD      got   = 0;
		uint64_t   total = 0;
		const auto begin = std::chrono::steady_clock::now();
		while (ReadFile(file, piece.data(), static_cast<DWORD>(piece.size()), &got, nullptr) != 0 && got != 0) total += got;
		CloseHandle(file);
		// KYTY_APR_LOG (libAmpr.cpp's): each file read through.
		static const bool log_reads = std::getenv("KYTY_APR_LOG") != nullptr;
		if (log_reads)
			std::printf("APRPREFETCH %llu bytes %.1f ms %ls\n", static_cast<unsigned long long>(total),
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(), path.c_str());
	}
}
} // namespace

// FollowKeptFiles: the recorded opens and the files queued for the openers. A 16 GB laptop loading 1-1 opened 28,528
// files for 31,390 reads, one after another on the game's file thread (0.9 ms a read at the median, 11 ms at p90:
// metadata and the antivirus's scan of each first open); its file thread was 60% in CreateFileW over the ~100 s
// loading fog.
struct FileFollow {
	std::mutex                                mutex; // (never held with g_kept_files_mutex)
	std::condition_variable                   ready;
	std::unordered_map<std::string, uint32_t> index; // host path -> place in the recording
	std::vector<const std::string*>           paths; // index's keys in recorded order
	std::vector<uint8_t>                      queued; // per place: queued, opened or opened by the game
	std::deque<uint32_t>                      queue;
	std::string                               root; // (with its separator)
	FILE*                                     out       = nullptr;
	uint32_t                                  unflushed = 0;
	bool                                      on        = false;
};
FileFollow g_follow;
// The opens wait (metadata, the antivirus), each thread one at a time: the 1-1 fog took 28.5-30.7 s with 4 threads and
// windows of 256, 24.3 s with 8 and 512, 17.6-20.6 s with 16 and 1024, 18.8 s with 32 and 2048. (KYTY_FILE_FOLLOW_WINDOW,
// KYTY_FILE_FOLLOW_THREADS: other counts.)
uint32_t g_follow_window = 1024, g_follow_threads = 16;

void FollowOpenThread() {
	SetThreadName("Kyty.FileOpen");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	for (;;) {
		std::string path;
		{
			std::unique_lock lock(g_follow.mutex);
			g_follow.ready.wait(lock, [] { return !g_follow.queue.empty(); });
			path = *g_follow.paths[g_follow.queue.front()];
			g_follow.queue.pop_front();
		}
		{
			const std::lock_guard lock(g_kept_files_mutex);
			if (g_kept_files.contains(path)) continue;
		}
		const auto wide = WidePath(path);
		if (wide.empty()) continue;
		HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
		                            FILE_ATTRIBUTE_NORMAL, nullptr);
		if (handle == INVALID_HANDLE_VALUE) continue; // (a file of an earlier session gone)
		LARGE_INTEGER length {};
		if (GetFileSizeEx(handle, &length) == 0) {
			CloseHandle(handle);
			continue;
		}
		const std::lock_guard lock(g_kept_files_mutex);
		const auto [kept, added] =
		    g_kept_files.try_emplace(path, KeptFile {handle, static_cast<uint64_t>(length.QuadPart), false, true});
		if (!added) CloseHandle(handle);
	}
}

// The game opened (or first read) `path`: the recorded files after it queued, or it recorded.
void FollowFrom(const std::string& path) {
	std::lock_guard lock(g_follow.mutex);
	if (!g_follow.on) return;
	const auto found = g_follow.index.find(path);
	if (found == g_follow.index.end()) {
		const auto place = static_cast<uint32_t>(g_follow.paths.size());
		const auto added = g_follow.index.emplace(path, place).first;
		g_follow.paths.push_back(&added->first);
		g_follow.queued.push_back(1);
		if (g_follow.out != nullptr) {
			const bool under = !g_follow.root.empty() && path.starts_with(g_follow.root);
			std::fprintf(g_follow.out, "%s\n", path.c_str() + (under ? g_follow.root.size() : 0));
			if (++g_follow.unflushed >= 32) {
				std::fflush(g_follow.out);
				g_follow.unflushed = 0;
			}
		}
		return;
	}
	g_follow.queued[found->second] = 1;
	// Ahead of the windows queued before (the game left them, or is past them): the next file first.
	const auto end = std::min<size_t>(size_t {found->second} + 1 + g_follow_window, g_follow.paths.size());
	bool       any = false;
	for (auto place = end; place-- > size_t {found->second} + 1;)
		if (g_follow.queued[place] == 0) {
			g_follow.queued[place] = 1;
			g_follow.queue.push_front(static_cast<uint32_t>(place));
			any = true;
		}
	if (any) g_follow.ready.notify_all();
}

uint64_t OpenKeptReadFile(const std::string& path, uint64_t* size) {
	{
		std::unique_lock lock(g_kept_files_mutex);
		if (const auto found = g_kept_files.find(path); found != g_kept_files.end()) {
			*size              = found->second.size;
			const auto handle  = reinterpret_cast<uint64_t>(found->second.handle);
			const bool first   = std::exchange(found->second.ahead, false);
			lock.unlock();
			if (first) FollowFrom(path);
			return handle;
		}
	}
	FollowFrom(path);
	// Opened outside the lock: a 16 GB laptop's opens took 100-390 ms while the game streamed a level in, and each
	// read of a file kept open already (an audio stream's next piece) waited for them behind it (open=150-250 ms).
	const auto wide = WidePath(path);
	if (wide.empty()) return 0;
	HANDLE handle = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
	                            FILE_ATTRIBUTE_NORMAL, nullptr);
	if (handle == INVALID_HANDLE_VALUE) return 0;
	LARGE_INTEGER length {};
	if (GetFileSizeEx(handle, &length) == 0) {
		CloseHandle(handle);
		return 0;
	}
	const std::lock_guard lock(g_kept_files_mutex);
	const auto [kept, added] = g_kept_files.try_emplace(path, KeptFile {handle, static_cast<uint64_t>(length.QuadPart)});
	if (!added) CloseHandle(handle); // (another thread opened it meanwhile: its handle is the one kept)
	*size = kept->second.size;
	return reinterpret_cast<uint64_t>(kept->second.handle);
}

uint32_t ReadOpenFileAt(uint64_t file, uint64_t offset, void* buffer, uint32_t size) {
	// A positional read (the handle is synchronous: it returns when done), so threads share the handle.
	OVERLAPPED at {};
	at.Offset     = static_cast<DWORD>(offset);
	at.OffsetHigh = static_cast<DWORD>(offset >> 32u);
	DWORD got     = 0;
	if (ReadFile(reinterpret_cast<HANDLE>(file), buffer, size, &got, &at) == 0) return 0;
	return got;
}

void FollowKeptFiles(const std::string& recording, const std::string& root) {
	static std::once_flag once;
	std::call_once(once, [&] {
		if (const char* value = std::getenv("KYTY_FILE_FOLLOW"); value != nullptr && std::string_view(value) == "0") return;
		if (const char* value = std::getenv("KYTY_FILE_FOLLOW_WINDOW"); value != nullptr && *value != '\0')
			g_follow_window = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 65536u);
		if (const char* value = std::getenv("KYTY_FILE_FOLLOW_THREADS"); value != nullptr && *value != '\0')
			g_follow_threads = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 64u);
		std::lock_guard lock(g_follow.mutex);
		g_follow.root = root;
		if (!g_follow.root.empty() && g_follow.root.back() != '/' && g_follow.root.back() != '\\') g_follow.root += '/';
		if (FILE* in = std::fopen(recording.c_str(), "rb")) {
			std::string line;
			for (int c; (c = std::fgetc(in)) != EOF;) {
				if (c != '\n') {
					line += static_cast<char>(c);
					continue;
				}
				if (!line.empty() && line.back() == '\r') line.pop_back();
				if (!line.empty()) {
					// (Relative: under the root. Absolute: a drive or a separator first.)
					const bool absolute = line.size() > 1 && (line[1] == ':' || line[0] == '/' || line[0] == '\\');
					auto       full     = absolute ? line : g_follow.root + line;
					const auto place    = static_cast<uint32_t>(g_follow.paths.size());
					if (const auto [added, inserted] = g_follow.index.emplace(std::move(full), place); inserted)
						g_follow.paths.push_back(&added->first);
				}
				line.clear();
			}
			std::fclose(in);
		}
		g_follow.queued.assign(g_follow.paths.size(), 0);
		std::error_code error;
		std::filesystem::create_directories(std::filesystem::path(recording).parent_path(), error);
		g_follow.out = std::fopen(recording.c_str(), "ab");
		g_follow.on  = true;
		for (uint32_t t = 0; t < g_follow_threads; ++t) std::thread(FollowOpenThread).detach();
	});
}

void PrefetchKeptFile(const std::string& path) {
	{
		const std::lock_guard lock(g_kept_files_mutex);
		const auto            found = g_kept_files.find(path);
		if (found == g_kept_files.end() || std::exchange(found->second.prefetched, true)) return;
	}
	auto wide = WidePath(path);
	if (wide.empty()) return;
	static std::once_flag started;
	std::call_once(started, [] { std::thread(PrefetchThread).detach(); });
	{
		const std::lock_guard lock(g_prefetch_mutex);
		g_prefetch_queue.push_back(std::move(wide));
	}
	g_prefetch_ready.notify_one();
}

double NamedThreadsCpuSeconds(const char* name) {
	std::wstring wanted;
	for (const char* p = name; *p != '\0'; ++p) wanted.push_back(static_cast<wchar_t>(*p));
	double       total    = 0;
	const DWORD  process  = GetCurrentProcessId();
	HANDLE       snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return 0;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			if (wanted == description) total += HandleCpuSeconds(thread);
			LocalFree(description);
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return total;
}

std::vector<std::pair<uint32_t, std::string>> ProcessThreads() {
	std::vector<std::pair<uint32_t, std::string>> threads;
	const DWORD process  = GetCurrentProcessId();
	HANDLE      snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return threads;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		PWSTR       description = nullptr;
		std::string name;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			for (const wchar_t* c = description; *c != L'\0'; ++c) name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
			LocalFree(description);
		}
		threads.emplace_back(entry.th32ThreadID, std::move(name));
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return threads;
}

bool FlushProcessWriteBuffers() {
	::FlushProcessWriteBuffers();
	return true;
}

bool CurrentThreadStack(uint64_t* low, uint64_t* high) {
	ULONG_PTR stack_low = 0, stack_high = 0;
	GetCurrentThreadStackLimits(&stack_low, &stack_high);
	*low  = stack_low;
	*high = stack_high;
	return stack_high > stack_low;
}

void AvoidCpuList(const char* avoid_cpus) {
	DWORD_PTR process = 0, system = 0;
	if (const auto avoid = static_cast<DWORD_PTR>(CpuListMask(avoid_cpus));
	    avoid != 0 && GetProcessAffinityMask(GetCurrentProcess(), &process, &system) != 0 && (process & ~avoid) != 0)
		(void)SetThreadAffinityMask(GetCurrentThread(), process & ~avoid);
}

void MakeBackgroundThread(const char* avoid_cpus) {
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	AvoidCpuList(avoid_cpus);
}

void MakeCriticalThread() {
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

namespace {
// d3dkmthk.h (WDK) structures for the display's vertical blank, taken from gdi32 at run time.
using KmtHandle = UINT;
struct KmtOpenAdapterFromGdiDisplayName {
	WCHAR     DeviceName[32];
	KmtHandle hAdapter;
	LUID      AdapterLuid;
	UINT      VidPnSourceId;
};
struct KmtWaitForVerticalBlankEvent {
	KmtHandle hAdapter;
	KmtHandle hDevice;
	UINT      VidPnSourceId;
};
struct KmtCloseAdapter {
	KmtHandle hAdapter;
};
using KmtOpenFn  = LONG(APIENTRY*)(KmtOpenAdapterFromGdiDisplayName*);
using KmtWaitFn  = LONG(APIENTRY*)(const KmtWaitForVerticalBlankEvent*);
using KmtCloseFn = LONG(APIENTRY*)(const KmtCloseAdapter*);

struct DisplayVblanks {
	std::mutex              mutex;
	std::condition_variable changed;
	uint64_t                count = 0;
	std::atomic<HWND>       window {nullptr};
	std::atomic<bool>       started {false};
};
DisplayVblanks g_display_vblanks;

bool MonitorDevice(HWND window, HMONITOR& monitor, MONITORINFOEXW& info) {
	monitor     = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
	info        = {};
	info.cbSize = sizeof(info);
	return monitor != nullptr && GetMonitorInfoW(monitor, &info) != FALSE;
}

void CountDisplayVblanks(KmtOpenFn open, KmtWaitFn wait, KmtCloseFn close) {
	SetThreadName("Kyty.DisplayVblank");
	MakeCriticalThread();
	HMONITOR  monitor = nullptr;
	KmtHandle adapter = 0;
	UINT      source  = 0;
	uint32_t  waits   = 0;
	const auto release = [&] {
		if (adapter != 0) {
			const KmtCloseAdapter request {adapter};
			(void)close(&request);
			adapter = 0;
		}
	};
	for (;;) {
		// The window's display: at the start, after a failed wait and every 120 vblanks (the window may have moved).
		if (adapter == 0 || ++waits >= 120) {
			waits = 0;
			HMONITOR       now = nullptr;
			MONITORINFOEXW info {};
			if (!MonitorDevice(g_display_vblanks.window.load(), now, info)) {
				release();
				Sleep(100);
				continue;
			}
			if (adapter == 0 || now != monitor) {
				release();
				KmtOpenAdapterFromGdiDisplayName request {};
				wcsncpy_s(request.DeviceName, info.szDevice, _TRUNCATE);
				if (open(&request) != 0) {
					Sleep(100);
					continue;
				}
				monitor = now;
				adapter = request.hAdapter;
				source  = request.VidPnSourceId;
			}
		}
		const KmtWaitForVerticalBlankEvent request {adapter, 0, source};
		if (wait(&request) != 0) {
			release();
			Sleep(16);
			continue;
		}
		{
			std::lock_guard lock(g_display_vblanks.mutex);
			g_display_vblanks.count++;
		}
		g_display_vblanks.changed.notify_all();
	}
}
} // namespace

void SleepUntilPrecise(int64_t deadline_ns) {
	using clock = std::chrono::steady_clock;
	const auto deadline = clock::time_point(std::chrono::duration_cast<clock::duration>(std::chrono::nanoseconds(deadline_ns)));
	const auto coarse   = deadline - std::chrono::microseconds(500);
	if (clock::now() < coarse) {
		static thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* HIGH_RESOLUTION */, TIMER_ALL_ACCESS);
		const auto wait_100ns = std::chrono::duration_cast<std::chrono::nanoseconds>(coarse - clock::now()).count() / 100;
		LARGE_INTEGER due {};
		due.QuadPart = -static_cast<LONGLONG>(wait_100ns);
		if (timer != nullptr && wait_100ns > 0 && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
			WaitForSingleObject(timer, INFINITE);
		} else if (wait_100ns > 0) {
			std::this_thread::sleep_until(coarse);
		}
	}
	while (clock::now() < deadline) YieldProcessor();
}

void* MonitorHandle(void* window) {
	return window != nullptr ? static_cast<void*>(MonitorFromWindow(static_cast<HWND>(window), MONITOR_DEFAULTTONEAREST)) : nullptr;
}

int DisplayRefreshHz(void* window) {
	HMONITOR       monitor = nullptr;
	MONITORINFOEXW info {};
	if (!MonitorDevice(static_cast<HWND>(window), monitor, info)) return 0;
	DEVMODEW mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) == FALSE) return 0;
	return mode.dmDisplayFrequency > 1 ? static_cast<int>(mode.dmDisplayFrequency) : 0;
}

uint32_t DisplayHeight(void* window) {
	HMONITOR monitor = window != nullptr ? MonitorFromWindow(static_cast<HWND>(window), MONITOR_DEFAULTTONEAREST)
	                                     : MonitorFromPoint(POINT {0, 0}, MONITOR_DEFAULTTOPRIMARY);
	MONITORINFOEXW info {};
	info.cbSize = sizeof(info);
	if (monitor == nullptr || GetMonitorInfoW(monitor, &info) == FALSE) return 0;
	DEVMODEW mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) == FALSE) return 0;
	return static_cast<uint32_t>(mode.dmPelsHeight);
}

bool StartDisplayVblanks(void* window) {
	g_display_vblanks.window.store(static_cast<HWND>(window));
	if (g_display_vblanks.started.load()) return true;
	HMODULE    gdi   = GetModuleHandleW(L"gdi32.dll");
	if (gdi == nullptr) gdi = LoadLibraryW(L"gdi32.dll");
	const auto open  = gdi != nullptr ? reinterpret_cast<KmtOpenFn>(reinterpret_cast<void*>(
	                                       GetProcAddress(gdi, "D3DKMTOpenAdapterFromGdiDisplayName")))
	                                  : nullptr;
	const auto wait  = gdi != nullptr ? reinterpret_cast<KmtWaitFn>(reinterpret_cast<void*>(
	                                       GetProcAddress(gdi, "D3DKMTWaitForVerticalBlankEvent")))
	                                  : nullptr;
	const auto close = gdi != nullptr ? reinterpret_cast<KmtCloseFn>(reinterpret_cast<void*>(
	                                        GetProcAddress(gdi, "D3DKMTCloseAdapter")))
	                                  : nullptr;
	if (open == nullptr || wait == nullptr || close == nullptr) return false;
	if (!g_display_vblanks.started.exchange(true)) std::thread(CountDisplayVblanks, open, wait, close).detach();
	return true;
}

bool WaitDisplayVblank(uint64_t& seen, uint32_t timeout_us) {
	if (!g_display_vblanks.started.load()) return false;
	std::unique_lock lock(g_display_vblanks.mutex);
	if (seen == 0) seen = g_display_vblanks.count; // (a caller's first wait: the next vblank)
	if (!g_display_vblanks.changed.wait_for(lock, std::chrono::microseconds(timeout_us),
	                                        [&] { return g_display_vblanks.count > seen; })) {
		return false;
	}
	// One vblank a call: those that came while the caller was away are caught up, four at most.
	seen = std::max(seen + 1, g_display_vblanks.count > 4 ? g_display_vblanks.count - 4 : uint64_t {0});
	return true;
}

uint64_t OpenScratchFile() {
	wchar_t directory[MAX_PATH + 1] {}, path[MAX_PATH + 1] {};
	if (GetTempPathW(MAX_PATH + 1, directory) == 0 || GetTempFileNameW(directory, L"kyt", 0, path) == 0) return 0;
	// FILE_ATTRIBUTE_TEMPORARY kept every written page dirty in RAM (Windows does not flush a temporary
	// file while memory lasts): the shader prefetch's ~6.7 GB of SPIR-V stayed as modified pages, which
	// is not "available" memory, and Task Manager showed the PC's memory filling. A normal file is
	// written out in the background and its pages become standby cache (reclaimable, still fast to
	// read back). KYTY_SCRATCH_IN_MEMORY=1: the temporary file as before.
	static const bool in_memory = [] {
		const char* value = std::getenv("KYTY_SCRATCH_IN_MEMORY");
		return value != nullptr && *value != '\0' && *value != '0';
	}();
	HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
	                          (in_memory ? FILE_ATTRIBUTE_TEMPORARY : FILE_ATTRIBUTE_NORMAL) | FILE_FLAG_DELETE_ON_CLOSE,
	                          nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		(void)DeleteFileW(path);
		return 0;
	}
	return reinterpret_cast<uint64_t>(file);
}

// ReadFile or WriteFile at an offset, in pieces a DWORD can count.
template <typename Call, typename Byte>
static bool ScratchTransfer(uint64_t file, uint64_t offset, Byte* data, size_t size, Call call) {
	while (size != 0) {
		OVERLAPPED at {};
		at.Offset         = static_cast<DWORD>(offset);
		at.OffsetHigh     = static_cast<DWORD>(offset >> 32u);
		DWORD      done   = 0;
		const auto length = static_cast<DWORD>(size < (1u << 30u) ? size : (1u << 30u));
		if (call(reinterpret_cast<HANDLE>(file), data, length, &done, &at) == 0 || done == 0) return false;
		data += done;
		offset += done;
		size -= done;
	}
	return true;
}

bool WriteScratchFile(uint64_t file, uint64_t offset, const void* data, size_t size) {
	return file != 0 && ScratchTransfer(file, offset, static_cast<const uint8_t*>(data), size, WriteFile);
}

bool ReadScratchFile(uint64_t file, uint64_t offset, void* data, size_t size) {
	return file != 0 && ScratchTransfer(file, offset, static_cast<uint8_t*>(data), size, ReadFile);
}

void CloseScratchFile(uint64_t file) {
	if (file != 0) CloseHandle(reinterpret_cast<HANDLE>(file));
}

uint64_t OpenFileForReading(const char* path) {
	wchar_t wide[MAX_PATH * 2] {};
	if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH * 2) == 0) return 0;
	HANDLE file = CreateFileW(wide, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
	                          FILE_FLAG_RANDOM_ACCESS, nullptr);
	return file == INVALID_HANDLE_VALUE ? 0 : reinterpret_cast<uint64_t>(file);
}

bool WriteHeapHistogram(const char* path) {
	std::vector<HANDLE> heaps(256);
	const DWORD         count = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
	if (count == 0 || count > heaps.size()) return false;
	// Nothing is allocated from a heap while it is walked: blocks under 16 bytes are counted by size in arrays made
	// before, larger ones are recorded (size, first qword) in pages of their own and grouped after.
	constexpr uint64_t Small = 16;
	struct Block {
		uint32_t heap;
		uint64_t size;
		uint64_t first;
	};
	constexpr size_t Capacity = size_t {16} << 20u;
	auto* blocks = static_cast<Block*>(VirtualAlloc(nullptr, Capacity * sizeof(Block), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
	if (blocks == nullptr) return false;
	std::vector<std::pair<uint64_t, uint64_t>> by_size(static_cast<size_t>(count) * Small); // heap, size -> blocks, bytes
	size_t recorded = 0, dropped = 0;
	// KYTY_HEAP_SAMPLE=<size>: the bytes of some busy blocks of that size (the first 32, then every 32768th) in
	// <path>.samples.txt, to tell what they are.
	const char*       sample_value = std::getenv("KYTY_HEAP_SAMPLE");
	const uint64_t    sample_size  = sample_value != nullptr ? std::strtoull(sample_value, nullptr, 10) : 0;
	constexpr size_t  SampleCount = 128, SampleBytes = 256;
	struct Sample {
		uint64_t address;
		uint64_t size;
		uint8_t  bytes[SampleBytes];
	};
	auto*  samples = sample_size != 0 ? static_cast<Sample*>(VirtualAlloc(nullptr, SampleCount * sizeof(Sample),
	                                                                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
	                                  : nullptr;
	size_t sampled = 0, sample_seen = 0;
	// Where the heaps' memory lies: their regions (segments) and blocks of a megabyte or more (each its own
	// allocation): the private allocations of <path>.vm.tsv that are the heaps'.
	constexpr size_t      MaxPlaces = 65536;
	std::vector<uint64_t> places(MaxPlaces);
	size_t                place_count = 0;
	for (DWORD i = 0; i < count; ++i) {
		if (!HeapLock(heaps[i])) continue;
		PROCESS_HEAP_ENTRY entry {};
		while (HeapWalk(heaps[i], &entry)) {
			if ((entry.wFlags & PROCESS_HEAP_REGION) != 0 && place_count < MaxPlaces)
				places[place_count++] = reinterpret_cast<uint64_t>(entry.lpData);
			if ((entry.wFlags & PROCESS_HEAP_ENTRY_BUSY) == 0) continue;
			if (samples != nullptr && entry.cbData == sample_size && sampled < SampleCount &&
			    (sample_seen++ < 32 || sample_seen % 32768 == 0)) {
				samples[sampled].address = reinterpret_cast<uint64_t>(entry.lpData);
				samples[sampled].size    = entry.cbData;
				std::memcpy(samples[sampled].bytes, entry.lpData, std::min<size_t>(entry.cbData, SampleBytes));
				sampled++;
			}
			if (entry.cbData >= (uint64_t {1} << 20u) && place_count < MaxPlaces)
				places[place_count++] = reinterpret_cast<uint64_t>(entry.lpData);
			if (entry.cbData < Small) {
				auto& use = by_size[i * Small + entry.cbData];
				use.first++;
				use.second += entry.cbData;
			} else if (recorded < Capacity) {
				uint64_t first = 0;
				std::memcpy(&first, entry.lpData, sizeof(first));
				blocks[recorded++] = {i, entry.cbData, first};
			} else {
				dropped++;
			}
		}
		HeapUnlock(heaps[i]);
	}
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> sizes; // block size -> blocks, bytes
	for (size_t i = 0; i < by_size.size(); i++) {
		if (by_size[i].first == 0) continue;
		auto& use = sizes[i % Small];
		use.first += by_size[i].first;
		use.second += by_size[i].second;
	}
	for (size_t i = 0; i < recorded; i++) {
		auto& use = sizes[blocks[i].size];
		use.first++;
		use.second += blocks[i].size;
	}
	FILE* file = std::fopen(path, "w");
	if (file == nullptr) {
		VirtualFree(blocks, 0, MEM_RELEASE);
		return false;
	}
	std::fprintf(file, "size\tcount\tbytes\n");
	for (const auto& [size, use]: sizes)
		std::fprintf(file, "%llu\t%llu\t%llu\n", static_cast<unsigned long long>(size),
		             static_cast<unsigned long long>(use.first), static_cast<unsigned long long>(use.second));
	std::fclose(file);
	if (samples != nullptr) {
		if (FILE* out = std::fopen((std::string(path) + ".samples.txt").c_str(), "w"); out != nullptr) {
			for (size_t i = 0; i < sampled; i++) {
				std::fprintf(out, "%llx %llu:", static_cast<unsigned long long>(samples[i].address),
				             static_cast<unsigned long long>(samples[i].size));
				for (size_t at = 0; at + 8 <= std::min<size_t>(samples[i].size, SampleBytes); at += 8) {
					uint64_t qword = 0;
					std::memcpy(&qword, samples[i].bytes + at, sizeof(qword));
					std::fprintf(out, " %llx", static_cast<unsigned long long>(qword));
				}
				std::fprintf(out, "\n");
			}
			std::fclose(out);
		}
		VirtualFree(samples, 0, MEM_RELEASE);
	}
	// What the blocks are, per heap: <path>.sig.tsv groups blocks of 16 bytes or more by size and first qword: SPIR-V's
	// magic, an address in a module (a vtable: module+offset), zero, else data; groups under 64 KiB and 16 blocks go
	// to "rest" per heap.
	struct Module {
		uint64_t    begin;
		uint64_t    end;
		std::string name;
	};
	std::vector<Module>  modules;
	std::vector<HMODULE> handles(2048);
	DWORD                needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), handles.data(), static_cast<DWORD>(handles.size() * sizeof(HMODULE)),
	                       &needed) != 0) {
		handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
		for (HMODULE handle: handles) {
			MODULEINFO module {};
			char       name[MAX_PATH] {};
			if (GetModuleInformation(GetCurrentProcess(), handle, &module, sizeof(module)) != 0 &&
			    GetModuleBaseNameA(GetCurrentProcess(), handle, name, MAX_PATH) != 0) {
				const auto begin = reinterpret_cast<uint64_t>(module.lpBaseOfDll);
				modules.push_back({begin, begin + module.SizeOfImage, name});
			}
		}
	}
	std::sort(modules.begin(), modules.end(), [](const Module& a, const Module& b) { return a.begin < b.begin; });
	constexpr int64_t Data = -1, Spirv = -2, Zero = -3;
	// heap, size, kind (module index or Data/Spirv/Zero), offset in the module -> blocks, bytes
	std::map<std::tuple<uint32_t, uint64_t, int64_t, uint64_t>, std::pair<uint64_t, uint64_t>> groups;
	for (size_t i = 0; i < recorded; i++) {
		const uint64_t first  = blocks[i].first;
		int64_t        kind   = Data;
		uint64_t       offset = 0;
		if (static_cast<uint32_t>(first) == 0x07230203u) {
			kind = Spirv;
		} else if (first == 0) {
			kind = Zero;
		} else {
			auto module = std::upper_bound(modules.begin(), modules.end(), first,
			                               [](uint64_t value, const Module& m) { return value < m.begin; });
			if (module != modules.begin() && first < std::prev(module)->end) {
				kind   = std::prev(module) - modules.begin();
				offset = first - std::prev(module)->begin;
			}
		}
		auto& use = groups[{blocks[i].heap, blocks[i].size, kind, offset}];
		use.first++;
		use.second += blocks[i].size;
	}
	VirtualFree(blocks, 0, MEM_RELEASE);
	const std::string sig_path = std::string(path) + ".sig.tsv";
	file                       = std::fopen(sig_path.c_str(), "w");
	if (file == nullptr) return false;
	std::fprintf(file, "heap\tsize\tkind\tcount\tbytes\n");
	std::vector<std::pair<uint64_t, uint64_t>> rest(count); // per heap: blocks, bytes
	for (const auto& [key, use]: groups) {
		const auto& [heap, size, kind, offset] = key;
		if (use.first < 16 && use.second < (uint64_t {64} << 10u)) {
			rest[heap].first += use.first;
			rest[heap].second += use.second;
			continue;
		}
		char text[320];
		if (kind >= 0) {
			std::snprintf(text, sizeof(text), "%s+0x%llx", modules[kind].name.c_str(), static_cast<unsigned long long>(offset));
		} else {
			std::snprintf(text, sizeof(text), "%s", kind == Spirv ? "spirv" : kind == Zero ? "zero" : "data");
		}
		std::fprintf(file, "%lu\t%llu\t%s\t%llu\t%llu\n", static_cast<unsigned long>(heap), static_cast<unsigned long long>(size),
		             text, static_cast<unsigned long long>(use.first), static_cast<unsigned long long>(use.second));
	}
	for (DWORD i = 0; i < count; ++i) {
		uint64_t blocks_small = 0, bytes_small = 0;
		for (uint64_t size = 0; size < Small; size++) {
			blocks_small += by_size[i * Small + size].first;
			bytes_small += by_size[i * Small + size].second;
		}
		if (rest[i].first != 0)
			std::fprintf(file, "%lu\t0\trest\t%llu\t%llu\n", static_cast<unsigned long>(i),
			             static_cast<unsigned long long>(rest[i].first), static_cast<unsigned long long>(rest[i].second));
		if (blocks_small != 0)
			std::fprintf(file, "%lu\t0\tsmall\t%llu\t%llu\n", static_cast<unsigned long>(i),
			             static_cast<unsigned long long>(blocks_small), static_cast<unsigned long long>(bytes_small));
	}
	if (dropped != 0) std::fprintf(file, "-1\t0\tdropped\t%llu\t0\n", static_cast<unsigned long long>(dropped));
	std::fclose(file);
	// Committed private memory per allocation (VirtualAlloc'd pools, heap segments, driver memory): <path>.vm.tsv.
	const std::string vm_path = std::string(path) + ".vm.tsv";
	file                      = std::fopen(vm_path.c_str(), "w");
	if (file == nullptr) return false;
	std::fprintf(file, "base\tcommitted\tregions\theap\n");
	std::set<uint64_t> heap_bases;
	for (size_t i = 0; i < place_count; i++) {
		MEMORY_BASIC_INFORMATION place {};
		if (VirtualQuery(reinterpret_cast<void*>(places[i]), &place, sizeof(place)) == sizeof(place))
			heap_bases.insert(reinterpret_cast<uint64_t>(place.AllocationBase));
	}
	std::map<uint64_t, std::pair<uint64_t, uint64_t>> bases; // allocation base -> committed bytes, regions
	MEMORY_BASIC_INFORMATION                          info {};
	for (uint64_t address = 0x10000; VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) == sizeof(info);) {
		if (info.State == MEM_COMMIT && info.Type == MEM_PRIVATE) {
			auto& base = bases[reinterpret_cast<uint64_t>(info.AllocationBase)];
			base.first += info.RegionSize;
			base.second++;
		}
		const uint64_t next = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		if (next <= address) break;
		address = next;
	}
	for (const auto& [base, use]: bases)
		std::fprintf(file, "%llx\t%llu\t%llu\t%d\n", static_cast<unsigned long long>(base),
		             static_cast<unsigned long long>(use.first), static_cast<unsigned long long>(use.second),
		             heap_bases.contains(base) ? 1 : 0);
	std::fclose(file);
	return true;
}

namespace AllocTrace {

// Live blocks of the traced sizes: open addressing by address (tombstones on free), stacks interned by hash. All in
// pages of their own: the hooks never allocate from a heap.
constexpr size_t   MaxSize    = 1u << 20u;
constexpr size_t   BlockSlots = size_t {1} << 22u;
constexpr size_t   StackSlots = size_t {1} << 18u;
constexpr int      Frames     = 20;
constexpr uint64_t Empty = 0, Tombstone = 1;

struct BlockSlot {
	uint64_t address;
	uint32_t stack;
	uint32_t size;
};
struct StackSlot {
	uint64_t hash;
	void*    frames[Frames];
	uint32_t count;
};

using Allocate   = void*(NTAPI*)(void*, ULONG, SIZE_T);
using Free       = BOOLEAN(NTAPI*)(void*, ULONG, void*);
using Reallocate = void*(NTAPI*)(void*, ULONG, void*, SIZE_T);

std::atomic_flag  g_lock       = ATOMIC_FLAG_INIT;
uint8_t*          g_sizes      = nullptr; // MaxSize flags
BlockSlot*        g_blocks     = nullptr;
BlockSlot*        g_scratch    = nullptr; // a rehash's copy of the live blocks
size_t            g_used       = 0;       // slots not empty (live and tombstones)
size_t            g_live       = 0;
StackSlot*        g_stacks     = nullptr;
Allocate          g_allocate   = nullptr;
Free              g_free       = nullptr;
Reallocate        g_reallocate = nullptr;
std::atomic<bool> g_full {false};

struct Guard {
	Guard() {
		while (g_lock.test_and_set(std::memory_order_acquire)) YieldProcessor();
	}
	~Guard() { g_lock.clear(std::memory_order_release); }
};

uint64_t Mix(uint64_t value) {
	value ^= value >> 33u;
	value *= 0xff51afd7ed558ccdull;
	value ^= value >> 33u;
	return value;
}

void Record(void* block, size_t size) {
	void*      frames[Frames] {};
	const auto captured = RtlCaptureStackBackTrace(2, Frames, frames, nullptr);
	uint64_t   hash     = size;
	for (USHORT i = 0; i < captured; i++) hash = Mix(hash ^ reinterpret_cast<uint64_t>(frames[i]));
	if (hash <= Tombstone) hash += 2;
	Guard    guard;
	uint32_t stack = 0;
	for (size_t i = hash % StackSlots, probes = 0; probes < StackSlots; i = (i + 1) % StackSlots, probes++) {
		if (g_stacks[i].hash == hash) {
			stack = static_cast<uint32_t>(i);
			break;
		}
		if (g_stacks[i].hash == Empty) {
			g_stacks[i].hash = hash;
			std::memcpy(g_stacks[i].frames, frames, sizeof(frames));
			stack = static_cast<uint32_t>(i);
			break;
		}
	}
	g_stacks[stack].count++;
	// Half the slots used (tombstones of freed blocks mostly): the live blocks go in again from scratch. (A free racing
	// with it may miss its block's record and leave it listed.)
	if ((g_used + 1) * 2 > BlockSlots) {
		size_t live = 0;
		for (size_t i = 0; i < BlockSlots; i++)
			if (g_blocks[i].address > Tombstone) g_scratch[live++] = g_blocks[i];
		std::memset(g_blocks, 0, BlockSlots * sizeof(BlockSlot));
		for (size_t k = 0; k < live; k++) {
			size_t i = Mix(g_scratch[k].address) % BlockSlots;
			while (g_blocks[i].address != Empty) i = (i + 1) % BlockSlots;
			g_blocks[i] = g_scratch[k];
		}
		g_used = g_live = live;
	}
	const uint64_t address = reinterpret_cast<uint64_t>(block);
	for (size_t i = Mix(address) % BlockSlots, probes = 0; probes < BlockSlots; i = (i + 1) % BlockSlots, probes++) {
		if (g_blocks[i].address <= Tombstone) {
			if (g_blocks[i].address == Empty) g_used++;
			g_live++;
			g_blocks[i] = {address, stack, static_cast<uint32_t>(size)};
			return;
		}
	}
	g_full = true;
}

void Forget(void* block) {
	const uint64_t address = reinterpret_cast<uint64_t>(block);
	// Most blocks freed were never traced: an empty first slot says so without the lock (slots never become empty
	// again, and a traced block's record was written before the block could be freed).
	if (std::atomic_ref<uint64_t>(g_blocks[Mix(address) % BlockSlots].address).load(std::memory_order_relaxed) == Empty)
		return;
	Guard guard;
	for (size_t i = Mix(address) % BlockSlots, probes = 0; probes < BlockSlots; i = (i + 1) % BlockSlots, probes++) {
		if (g_blocks[i].address == address) {
			g_stacks[g_blocks[i].stack].count--;
			g_blocks[i].address = Tombstone;
			g_live--;
			return;
		}
		if (g_blocks[i].address == Empty) return;
	}
}

void* NTAPI HookAllocate(void* heap, ULONG flags, SIZE_T size) {
	void* block = g_allocate(heap, flags, size);
	if (block != nullptr && size < MaxSize && g_sizes[size] != 0) Record(block, size);
	return block;
}

BOOLEAN NTAPI HookFree(void* heap, ULONG flags, void* block) {
	if (block != nullptr) Forget(block);
	return g_free(heap, flags, block);
}

void* NTAPI HookReallocate(void* heap, ULONG flags, void* block, SIZE_T size) {
	void* result = g_reallocate(heap, flags, block, size);
	if (result != nullptr) {
		if (block != nullptr) Forget(block);
		if (size < MaxSize && g_sizes[size] != 0) Record(result, size);
	}
	return result;
}

// Every loaded module's import slots that hold `from` now hold `to`.
int Patch(void* from, void* to) {
	std::vector<HMODULE> handles(2048);
	DWORD                needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), handles.data(), static_cast<DWORD>(handles.size() * sizeof(HMODULE)),
	                       &needed) == 0)
		return 0;
	handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
	int patched = 0;
	for (HMODULE handle: handles) {
		auto*       base = reinterpret_cast<uint8_t*>(handle);
		const auto* dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) continue;
		const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IAT];
		if (dir.VirtualAddress == 0 || dir.Size == 0) continue;
		auto* slots = reinterpret_cast<void**>(base + dir.VirtualAddress);
		for (size_t i = 0; i < dir.Size / sizeof(void*); i++) {
			if (slots[i] != from) continue;
			DWORD old = 0;
			if (VirtualProtect(&slots[i], sizeof(void*), PAGE_READWRITE, &old) == 0) continue;
			slots[i] = to;
			VirtualProtect(&slots[i], sizeof(void*), old, &old);
			patched++;
		}
	}
	return patched;
}

} // namespace AllocTrace

bool StartAllocTrace(const char* sizes) {
	using namespace AllocTrace;
	if (g_sizes == nullptr) {
		HMODULE ntdll = GetModuleHandleA("ntdll.dll");
		g_allocate    = reinterpret_cast<Allocate>(GetProcAddress(ntdll, "RtlAllocateHeap"));
		g_free        = reinterpret_cast<Free>(GetProcAddress(ntdll, "RtlFreeHeap"));
		g_reallocate  = reinterpret_cast<Reallocate>(GetProcAddress(ntdll, "RtlReAllocateHeap"));
		if (g_allocate == nullptr || g_free == nullptr || g_reallocate == nullptr) return false;
		const auto reserve = [](size_t bytes) {
			return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
		};
		g_sizes  = static_cast<uint8_t*>(reserve(MaxSize));
		g_blocks  = static_cast<BlockSlot*>(reserve(BlockSlots * sizeof(BlockSlot)));
		g_scratch = static_cast<BlockSlot*>(reserve(BlockSlots * sizeof(BlockSlot)));
		g_stacks = static_cast<StackSlot*>(reserve(StackSlots * sizeof(StackSlot)));
		if (g_sizes == nullptr || g_blocks == nullptr || g_scratch == nullptr || g_stacks == nullptr) return false;
	}
	for (const char* text = sizes; *text != '\0';) {
		char*      end  = nullptr;
		const auto size = std::strtoull(text, &end, 10);
		if (end == text) break;
		auto last = size; // "first-last": a range
		if (*end == '-') {
			text = end + 1;
			last = std::strtoull(text, &end, 10);
		}
		for (auto at = size; at <= last && at < MaxSize; at++) g_sizes[at] = 1;
		text = *end == ',' ? end + 1 : end;
	}
	// Frees first: a block traced from here on is forgotten whichever module frees it.
	const int patched = Patch(reinterpret_cast<void*>(g_free), reinterpret_cast<void*>(&HookFree)) +
	                    Patch(reinterpret_cast<void*>(g_reallocate), reinterpret_cast<void*>(&HookReallocate)) +
	                    Patch(reinterpret_cast<void*>(g_allocate), reinterpret_cast<void*>(&HookAllocate));
	std::printf("Alloc trace: %d import slots hooked\n", patched);
	std::fflush(stdout);
	return patched != 0;
}

bool WriteAllocTrace(const char* path) {
	using namespace AllocTrace;
	if (g_stacks == nullptr) return false;
	struct Module {
		uint64_t    begin;
		uint64_t    end;
		std::string name;
	};
	std::vector<Module>  modules;
	std::vector<HMODULE> handles(2048);
	DWORD                needed = 0;
	if (EnumProcessModules(GetCurrentProcess(), handles.data(), static_cast<DWORD>(handles.size() * sizeof(HMODULE)),
	                       &needed) != 0) {
		handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
		for (HMODULE handle: handles) {
			MODULEINFO module {};
			char       name[MAX_PATH] {};
			if (GetModuleInformation(GetCurrentProcess(), handle, &module, sizeof(module)) != 0 &&
			    GetModuleBaseNameA(GetCurrentProcess(), handle, name, MAX_PATH) != 0) {
				const auto begin = reinterpret_cast<uint64_t>(module.lpBaseOfDll);
				modules.push_back({begin, begin + module.SizeOfImage, name});
			}
		}
	}
	// The live stacks with their blocks' bytes, copied under the lock (no allocation there), written after.
	struct Live {
		uint32_t stack;
		uint64_t count;
		uint64_t bytes;
		uint32_t size;
	};
	std::vector<uint64_t> bytes(StackSlots), counts(StackSlots), sizes(StackSlots);
	std::vector<void*>    frames(StackSlots * Frames);
	{
		Guard guard;
		for (size_t i = 0; i < BlockSlots; i++) {
			if (g_blocks[i].address <= Tombstone) continue;
			bytes[g_blocks[i].stack] += g_blocks[i].size;
			counts[g_blocks[i].stack]++;
			sizes[g_blocks[i].stack] = g_blocks[i].size;
		}
		for (size_t i = 0; i < StackSlots; i++)
			if (counts[i] != 0) std::memcpy(&frames[i * Frames], g_stacks[i].frames, sizeof(g_stacks[i].frames));
	}
	std::vector<Live> live;
	for (uint32_t i = 0; i < StackSlots; i++)
		if (counts[i] != 0) live.push_back({i, counts[i], bytes[i], static_cast<uint32_t>(sizes[i])});
	std::sort(live.begin(), live.end(), [](const Live& a, const Live& b) { return a.bytes > b.bytes; });
	FILE* file = std::fopen(path, "w");
	if (file == nullptr) return false;
	std::fprintf(file, "count\tbytes\tsize\tframes%s\n", g_full ? " (table full: some blocks missing)" : "");
	for (const auto& entry: live) {
		std::fprintf(file, "%llu\t%llu\t%u\t", static_cast<unsigned long long>(entry.count),
		             static_cast<unsigned long long>(entry.bytes), entry.size);
		for (int f = 0; f < Frames; f++) {
			const auto address = reinterpret_cast<uint64_t>(frames[static_cast<size_t>(entry.stack) * Frames + f]);
			if (address == 0) break;
			const Module* found = nullptr;
			for (const auto& module: modules)
				if (address >= module.begin && address < module.end) found = &module;
			if (found != nullptr) {
				std::fprintf(file, "%s%s+0x%llx", f == 0 ? "" : " ", found->name.c_str(),
				             static_cast<unsigned long long>(address - found->begin));
			} else {
				std::fprintf(file, "%s0x%llx", f == 0 ? "" : " ", static_cast<unsigned long long>(address));
			}
		}
		std::fprintf(file, "\n");
	}
	std::fclose(file);
	return true;
}

MemoryUse ProcessMemory() {
	PROCESS_MEMORY_COUNTERS_EX counters {};
	counters.cb = sizeof(counters);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
	                            sizeof(counters)) == 0)
		return {};
	return {counters.PrivateUsage, counters.WorkingSetSize};
}

static std::atomic<void (*)(void (*)())> g_trim_runner {nullptr};

void KeepResident() {
	static std::once_flag once;
	std::call_once(once, [] {
		const char* text = std::getenv("KYTY_RESIDENT_MB");
		const auto  mb   = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
		if (mb == 0) return;
		const SIZE_T minimum = static_cast<SIZE_T>(mb << 20u);
		// (The maximum soft: the process may grow past it; the minimum hard: Windows keeps that much in RAM.)
		const SIZE_T maximum = minimum + (SIZE_T {512} << 20u);
		const bool   ok = SetProcessWorkingSetSizeEx(GetCurrentProcess(), minimum, maximum,
		                                             QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) != 0;
		std::printf("Memory: %llu MiB kept in RAM (KYTY_RESIDENT_MB)%s\n", static_cast<unsigned long long>(mb),
		            ok ? "" : ": refused by Windows");
		std::fflush(stdout);
	});
}
static SIZE_T g_ws_min = 0, g_ws_max = 0; // the working set sizes the process started with
void SetTrimRunner(void (*runner)(void (*call)())) {
	g_trim_runner.store(runner, std::memory_order_release);
}

void StartWorkingSetTrim() {
	static std::once_flag once;
	std::call_once(once, [] {
		const char* text    = std::getenv("KYTY_TRIM_RAM_SECONDS");
		const auto  seconds = text != nullptr ? std::strtoul(text, nullptr, 10) : 0ul;
		if (seconds == 0) return;
		// KYTY_TRIM_RAM_MB (default 8192): only a working set past it is emptied. Emptying it every minute
		// regardless re-faulted the 4-5 GB the game touches in a minute each time (10-10, 1-1 to the boss).
		const char* limit_text = std::getenv("KYTY_TRIM_RAM_MB");
		const auto  limit      = (limit_text != nullptr ? std::strtoull(limit_text, nullptr, 10) : 8192ull) << 20u;
		{
			DWORD flags = 0;
			GetProcessWorkingSetSizeEx(GetCurrentProcess(), &g_ws_min, &g_ws_max, &flags);
		}
		std::thread([seconds, limit] {
			SetThreadDescription(GetCurrentThread(), L"Kyty.TrimRam");
			for (uint32_t count = 0;; ++count) {
				std::this_thread::sleep_for(std::chrono::seconds(seconds));
				const auto before = ProcessMemory().working_set;
				if (before <= limit) continue;
				// In steps of 512 MiB down to half the limit (a hard working set maximum lowered each step, then soft
				// again), each with every guest map, unmap and protection change held off (the guest memory's runner):
				// emptying it at once stalled a frame 135-230 ms (10-19, Turbo 2.0).
				const auto run = [](void (*call)()) {
					if (auto* runner = g_trim_runner.load(std::memory_order_acquire); runner != nullptr) runner(call);
					else call();
				};
				static std::atomic<uint64_t> g_step_target {0};
				const uint64_t               floor = limit / 2;
				for (uint64_t ws = before; ws > floor;) {
					const uint64_t target = ws > floor + (512ull << 20u) ? ws - (512ull << 20u) : floor;
					g_step_target.store(target, std::memory_order_relaxed);
					run([] {
						SetProcessWorkingSetSizeEx(GetCurrentProcess(), 1u << 20u,
						                           static_cast<SIZE_T>(g_step_target.load(std::memory_order_relaxed)),
						                           QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);
					});
					std::this_thread::sleep_for(std::chrono::milliseconds(50));
					const auto now = ProcessMemory().working_set;
					if (now >= ws) break; // (no progress)
					ws = now;
				}
				// (Soft again: the game grows back as it needs.)
				// ((-1, -1) would empty it: the sizes the process started with.)
				SetProcessWorkingSetSizeEx(GetCurrentProcess(), g_ws_min, g_ws_max,
				                           QUOTA_LIMITS_HARDWS_MAX_DISABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE);
				if (count < 8 || count % 16 == 0) {
					std::printf("RAM trim: working set %llu -> %llu MiB\n", static_cast<unsigned long long>(before >> 20u),
					            static_cast<unsigned long long>(ProcessMemory().working_set >> 20u));
					std::fflush(stdout);
				}
			}
		}).detach();
	});
}

uint64_t PhysicalMemory() {
	if (const char* simulate = std::getenv("KYTY_SIMULATE_RAM_MB"); simulate != nullptr && *simulate != '\0')
		return std::strtoull(simulate, nullptr, 10) << 20u;
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	return GlobalMemoryStatusEx(&status) != 0 ? status.ullTotalPhys : 0;
}

uint64_t AvailablePhysicalMemory() {
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	return GlobalMemoryStatusEx(&status) != 0 ? status.ullAvailPhys : 0;
}

void LogAddressSpace(const char* label, uint64_t begin, uint64_t end) {
	struct Allocation {
		uint64_t base = 0, size = 0, committed = 0;
		DWORD    type = 0;
	};
	std::vector<Allocation>                    allocations;
	std::vector<std::pair<uint64_t, uint64_t>> gaps;
	for (uint64_t address = begin; address < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) break;
		const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto size = std::min<uint64_t>(static_cast<uint64_t>(info.RegionSize), end - base);
		if (info.State == MEM_FREE) {
			if (size >= (uint64_t {1} << 30u)) gaps.emplace_back(base, size);
		} else {
			const auto allocation = reinterpret_cast<uint64_t>(info.AllocationBase);
			if (allocations.empty() || allocations.back().base != allocation)
				allocations.push_back({.base = allocation, .type = info.Type});
			allocations.back().size += size;
			if (info.State == MEM_COMMIT) allocations.back().committed += size;
		}
		address = base + static_cast<uint64_t>(info.RegionSize);
	}
	std::printf("Address space %s (0x%llx-0x%llx):\n", label, static_cast<unsigned long long>(begin),
	            static_cast<unsigned long long>(end));
	uint64_t small_count = 0, small_bytes = 0;
	for (const auto& allocation: allocations) {
		if (allocation.size < (uint64_t {16} << 20u)) {
			++small_count;
			small_bytes += allocation.size;
			continue;
		}
		const char* type = allocation.type == MEM_IMAGE ? "image" : allocation.type == MEM_MAPPED ? "mapped" : "private";
		std::printf("  0x%012llx %9.2f GiB %s, %.2f GiB committed\n", static_cast<unsigned long long>(allocation.base),
		            static_cast<double>(allocation.size) / (1u << 30u), type,
		            static_cast<double>(allocation.committed) / (1u << 30u));
	}
	std::printf("  and %llu allocations under 16 MiB (%.2f GiB)\n", static_cast<unsigned long long>(small_count),
	            static_cast<double>(small_bytes) / (1u << 30u));
	for (const auto& [base, size]: gaps)
		std::printf("  free 0x%012llx %9.2f GiB\n", static_cast<unsigned long long>(base),
		            static_cast<double>(size) / (1u << 30u));
	std::fflush(stdout);
}

static IDXGIAdapter3* g_video_memory_adapter = nullptr;

bool OpenVideoMemoryAdapter(const uint8_t* luid) {
	HMODULE dxgi = LoadLibraryA("dxgi.dll");
	if (dxgi == nullptr || g_video_memory_adapter != nullptr) return g_video_memory_adapter != nullptr;
	using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
	const auto     create  = reinterpret_cast<CreateFactory>(GetProcAddress(dxgi, "CreateDXGIFactory1"));
	IDXGIFactory4* factory = nullptr;
	if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) return false;
	LUID id {};
	std::memcpy(&id, luid, sizeof(id));
	if (FAILED(factory->EnumAdapterByLuid(id, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&g_video_memory_adapter))))
		g_video_memory_adapter = nullptr;
	factory->Release();
	return g_video_memory_adapter != nullptr;
}

bool QueryVideoMemory(uint64_t* usage, uint64_t* budget) {
	DXGI_QUERY_VIDEO_MEMORY_INFO info {};
	if (g_video_memory_adapter == nullptr ||
	    FAILED(g_video_memory_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
		return false;
	*usage  = info.CurrentUsage;
	*budget = info.Budget;
	return true;
}

bool QuerySharedGpuMemory(uint64_t* usage) {
	DXGI_QUERY_VIDEO_MEMORY_INFO info {};
	if (g_video_memory_adapter == nullptr ||
	    FAILED(g_video_memory_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info)))
		return false;
	*usage = info.CurrentUsage;
	return true;
}

bool IsNvidiaSigned(const std::wstring& path) {
	WINTRUST_FILE_INFO file {};
	file.cbStruct      = sizeof(file);
	file.pcwszFilePath = path.c_str();
	WINTRUST_DATA data {};
	data.cbStruct            = sizeof(data);
	data.dwUIChoice          = WTD_UI_NONE;
	data.fdwRevocationChecks = WTD_REVOKE_NONE;
	data.dwUnionChoice       = WTD_CHOICE_FILE;
	data.pFile               = &file;
	data.dwStateAction       = WTD_STATEACTION_VERIFY;
	data.dwProvFlags         = WTD_CACHE_ONLY_URL_RETRIEVAL;
	GUID       action        = WINTRUST_ACTION_GENERIC_VERIFY_V2;
	const LONG status        = WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	bool       nvidia        = false;
	if (status == ERROR_SUCCESS) {
		// The signer: the leaf certificate of the first signature's chain.
		auto* provider = WTHelperProvDataFromStateData(data.hWVTStateData);
		auto* signer   = provider != nullptr ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
		auto* cert     = signer != nullptr ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
		wchar_t name[256] {};
		if (cert != nullptr && cert->pCert != nullptr &&
		    CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256) > 1)
			nvidia = std::wstring(name) == L"NVIDIA Corporation";
	}
	data.dwStateAction = WTD_STATEACTION_CLOSE;
	(void)WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	return nvidia;
}

#else

void SetThreadName(const char* name) {
	(void)pthread_setname_np(pthread_self(), name);
}

uint32_t ThreadId() {
	return static_cast<uint32_t>(syscall(SYS_gettid));
}

static bool MaskToSet(uint64_t mask, cpu_set_t* set) {
	CPU_ZERO(set);
	for (int cpu = 0; cpu < 64; ++cpu)
		if ((mask >> cpu) & 1u) CPU_SET(cpu, set);
	return mask != 0;
}

void PinThreadToCpuList(const char* list) {
	cpu_set_t set;
	if (MaskToSet(CpuListMask(list), &set)) (void)sched_setaffinity(0, sizeof(set), &set);
}

bool PinThreadHandleToCpuList(uint64_t handle, const char* list) {
	cpu_set_t set;
	return handle != 0 && MaskToSet(CpuListMask(list), &set) &&
	       pthread_setaffinity_np(static_cast<pthread_t>(handle), sizeof(set), &set) == 0;
}

bool SetProcessDefaultCpuList(const char* /*list*/) {
	return false;
}

static double ClockSeconds(clockid_t clock) {
	timespec now {};
	if (clock_gettime(clock, &now) != 0) return 0;
	return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) / 1e9;
}

uint64_t CurrentThreadHandle() {
	return static_cast<uint64_t>(pthread_self());
}

double ThreadCpuSeconds(uint64_t handle) {
	clockid_t clock {};
	if (pthread_getcpuclockid(static_cast<pthread_t>(handle), &clock) != 0) return 0;
	return ClockSeconds(clock);
}

// The thread's CPU time in TSC ticks, as Windows' thread cycle count is (the slow-frame lines' cpu=: busy wall time well
// over it is time the thread was blocked or not running). The TSC rate is measured once.
uint64_t CurrentThreadCycles() {
	static const double ticks_per_ns = [] {
		const auto     start = std::chrono::steady_clock::now();
		const uint64_t first = __rdtsc();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
		return ns > 0 ? static_cast<double>(__rdtsc() - first) / static_cast<double>(ns) : 0.0;
	}();
	timespec time {};
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) != 0) return 0;
	return static_cast<uint64_t>((static_cast<double>(time.tv_sec) * 1e9 + static_cast<double>(time.tv_nsec)) * ticks_per_ns);
}

int CurrentThreadPriority() {
	return 0;
}

uint64_t OpenKeptReadFile(const std::string& /*path*/, uint64_t* /*size*/) {
	return 0;
}

uint32_t ReadOpenFileAt(uint64_t /*file*/, uint64_t /*offset*/, void* /*buffer*/, uint32_t /*size*/) {
	return 0;
}

void PrefetchKeptFile(const std::string& /*path*/) {}

void FollowKeptFiles(const std::string& /*recording*/, const std::string& /*root*/) {}

std::vector<std::pair<uint32_t, std::string>> ProcessThreads() {
	return {};
}

// Windows samples another thread's stack (SuspendThread + GetThreadContext + unwind data). Linux has no such
// cross-thread sampler here: ProcessThreads() is empty, so the diagnostic's per-thread dump stays empty and these
// are never reached; they exist so the callers compile and link.
void PrepareSampling() {}

uint64_t OpenThreadForSampling(uint32_t /*thread_id*/) {
	return 0;
}

void CloseThreadForSampling(uint64_t /*handle*/) {}

bool SampleThread(uint64_t /*handle*/, uint64_t /*stack_low*/, uint64_t /*stack_high*/, uint64_t* /*out*/,
                  size_t /*words*/) {
	return false;
}

double NamedThreadsCpuSeconds(const char* name) {
	double     total = 0;
	const long ticks = sysconf(_SC_CLK_TCK);
	if (auto* dir = opendir("/proc/self/task")) {
		while (auto* entry = readdir(dir)) {
			if (entry->d_name[0] == '.') continue;
			char path[64], comm[32] {};
			std::snprintf(path, sizeof(path), "/proc/self/task/%s/comm", entry->d_name);
			if (auto* f = std::fopen(path, "re")) {
				if (std::fgets(comm, sizeof(comm), f)) comm[std::strcspn(comm, "\n")] = 0;
				std::fclose(f);
			}
			if (std::strcmp(comm, name) != 0) continue;
			std::snprintf(path, sizeof(path), "/proc/self/task/%s/stat", entry->d_name);
			if (auto* f = std::fopen(path, "re")) {
				char line[1024] {};
				if (std::fgets(line, sizeof(line), f)) {
					const char*        rest  = std::strrchr(line, ')');
					unsigned long long utime = 0, stime = 0;
					if (rest && std::sscanf(rest + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu",
					                        &utime, &stime) == 2)
						total += static_cast<double>(utime + stime) / static_cast<double>(ticks);
				}
				std::fclose(f);
			}
		}
		closedir(dir);
	}
	return total;
}

bool FlushProcessWriteBuffers() {
	return false;
}

bool CurrentThreadStack(uint64_t* low, uint64_t* high) {
	pthread_attr_t attr;
	bool           found = false;
	if (pthread_getattr_np(pthread_self(), &attr) == 0) {
		void*  stack = nullptr;
		size_t size  = 0;
		if (pthread_attr_getstack(&attr, &stack, &size) == 0) {
			*low  = reinterpret_cast<uint64_t>(stack);
			*high = *low + size;
			found = true;
		}
		pthread_attr_destroy(&attr);
	}
	return found;
}

void AvoidCpuList(const char* avoid_cpus) {
	cpu_set_t set;
	if (const auto avoid = CpuListMask(avoid_cpus); avoid != 0 && sched_getaffinity(0, sizeof(set), &set) == 0) {
		for (int cpu = 0; cpu < 64; ++cpu)
			if ((avoid >> cpu) & 1u) CPU_CLR(cpu, &set);
		if (CPU_COUNT(&set) != 0) (void)sched_setaffinity(0, sizeof(set), &set);
	}
}

void MakeBackgroundThread(const char* avoid_cpus) {
	(void)setpriority(PRIO_PROCESS, static_cast<id_t>(ThreadId()), 10);
	AvoidCpuList(avoid_cpus);
}

void MakeCriticalThread() {}

void SleepUntilPrecise(int64_t deadline_ns) {
	using clock = std::chrono::steady_clock;
	const auto deadline = clock::time_point(std::chrono::duration_cast<clock::duration>(std::chrono::nanoseconds(deadline_ns)));
	const auto coarse   = deadline - std::chrono::microseconds(500);
	if (clock::now() < coarse) std::this_thread::sleep_until(coarse);
	while (clock::now() < deadline) {}
}

void* MonitorHandle(void* /*window*/) {
	return nullptr;
}

int DisplayRefreshHz(void* /*window*/) {
	return 0;
}

uint32_t DisplayHeight(void* /*window*/) {
	return 0;
}

bool StartDisplayVblanks(void* /*window*/) {
	return false;
}

bool WaitDisplayVblank(uint64_t& /*seen*/, uint32_t /*timeout_us*/) {
	return false;
}

// Handles are the file descriptor + 1 (0: none).
uint64_t OpenScratchFile() {
	const char* directory = std::getenv("TMPDIR");
	std::string path      = std::string(directory != nullptr && *directory != '\0' ? directory : "/tmp") + "/kyty-XXXXXX";
	const int   fd        = mkstemp(path.data());
	if (fd < 0) return 0;
	(void)unlink(path.c_str());
	return static_cast<uint64_t>(fd) + 1;
}

bool WriteScratchFile(uint64_t file, uint64_t offset, const void* data, size_t size) {
	for (const auto* bytes = static_cast<const uint8_t*>(data); file != 0 && size != 0;) {
		const auto done = pwrite(static_cast<int>(file - 1), bytes, size, static_cast<off_t>(offset));
		if (done <= 0) return false;
		bytes += done;
		offset += static_cast<uint64_t>(done);
		size -= static_cast<size_t>(done);
	}
	return file != 0;
}

bool ReadScratchFile(uint64_t file, uint64_t offset, void* data, size_t size) {
	for (auto* bytes = static_cast<uint8_t*>(data); file != 0 && size != 0;) {
		const auto done = pread(static_cast<int>(file - 1), bytes, size, static_cast<off_t>(offset));
		if (done <= 0) return false;
		bytes += done;
		offset += static_cast<uint64_t>(done);
		size -= static_cast<size_t>(done);
	}
	return file != 0;
}

void CloseScratchFile(uint64_t file) {
	if (file != 0) close(static_cast<int>(file - 1));
}

uint64_t OpenFileForReading(const char* path) {
	const int fd = open(path, O_RDONLY | O_CLOEXEC);
	return fd < 0 ? 0 : static_cast<uint64_t>(fd) + 1;
}

bool WriteHeapHistogram(const char*) {
	return false;
}

bool StartAllocTrace(const char*) {
	return false;
}

bool WriteAllocTrace(const char*) {
	return false;
}

MemoryUse ProcessMemory() {
	// statm: total, resident, shared pages; private resident = resident - shared.
	MemoryUse use;
	if (FILE* file = std::fopen("/proc/self/statm", "r"); file != nullptr) {
		unsigned long long total = 0, resident = 0, shared = 0;
		if (std::fscanf(file, "%llu %llu %llu", &total, &resident, &shared) == 3) {
			const auto page   = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
			use.working_set   = resident * page;
			use.private_bytes = (resident - std::min(resident, shared)) * page;
		}
		std::fclose(file);
	}
	return use;
}

void StartWorkingSetTrim() {}
void KeepResident() {}
void SetTrimRunner(void (*)(void (*)())) {}

uint64_t PhysicalMemory() {
	if (const char* simulate = std::getenv("KYTY_SIMULATE_RAM_MB"); simulate != nullptr && *simulate != '\0')
		return std::strtoull(simulate, nullptr, 10) << 20u;
	const long pages = sysconf(_SC_PHYS_PAGES);
	const long size  = sysconf(_SC_PAGESIZE);
	return pages > 0 && size > 0 ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(size) : 0;
}

void LogAddressSpace(const char* /*label*/, uint64_t /*begin*/, uint64_t /*end*/) {}

uint64_t AvailablePhysicalMemory() {
	uint64_t available = 0;
	if (FILE* file = std::fopen("/proc/meminfo", "r")) {
		char line[256];
		while (std::fgets(line, sizeof(line), file) != nullptr) {
			unsigned long long kib = 0;
			if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
				available = static_cast<uint64_t>(kib) << 10u;
				break;
			}
		}
		std::fclose(file);
	}
	return available;
}

bool OpenVideoMemoryAdapter(const uint8_t* /*luid*/) {
	return false;
}

bool QueryVideoMemory(uint64_t* /*usage*/, uint64_t* /*budget*/) {
	return false;
}

bool QuerySharedGpuMemory(uint64_t* /*usage*/) {
	return false;
}

bool IsNvidiaSigned(const std::wstring& /*path*/) {
	return false;
}

#endif

size_t ReadableBytes(const void* address, size_t size) noexcept {
#if defined(_WIN32)
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(address, &info, sizeof(info)) == 0 || info.State != MEM_COMMIT || info.Protect == 0 ||
	    (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0)
		return 0;
	const auto end = reinterpret_cast<const uint8_t*>(info.BaseAddress) + info.RegionSize;
	return std::min<size_t>(size, static_cast<size_t>(end - static_cast<const uint8_t*>(address)));
#else
	return size;
#endif
}
} // namespace LocalPlatform
