#include "perf-overlay.h"

#include "local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <tlhelp32.h>
#include <cwchar>
#include <cwctype>
#undef min
#undef max
#else
#include <cpuid.h>
#if defined(__linux__)
#include <time.h>
#endif
#endif

// The detailed panel's sampling (a diagnostic switch, poked live): 1 the GPU (NVML), 2 the cores' clocks (performance
// counters), 4 the rest (processor times, memory); 0 none.
extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_overlay_sampling {7};
}

namespace PerfOverlay {

namespace {

// The frames' flip times and the render thread's idle totals at them (one writer: the GPU thread).
constexpr size_t      Ring = 4096;
std::atomic<int64_t>  g_flip_ns[Ring] {};
std::atomic<int64_t>  g_flip_idle_ns[Ring] {};
std::atomic<uint64_t> g_flips {0};

std::mutex     g_machine_mutex;
Machine        g_machine; // (g_machine_mutex)
std::once_flag g_machine_started;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// The processor's brand string without trademarks, generation and core count ("Intel Core i9-14900K").
std::string CpuName() {
	char brand[49] {};
#if defined(_WIN32)
	int regs[4] {};
	__cpuid(regs, static_cast<int>(0x80000000u));
	if (static_cast<uint32_t>(regs[0]) < 0x80000004u) return {};
	for (int i = 0; i < 3; ++i) {
		__cpuid(regs, static_cast<int>(0x80000002u) + i);
		std::memcpy(brand + 16 * i, regs, 16);
	}
#else
	uint32_t regs[4] {};
	if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000004u) return {};
	for (uint32_t i = 0; i < 3; ++i) {
		__get_cpuid(0x80000002u + i, &regs[0], &regs[1], &regs[2], &regs[3]);
		std::memcpy(brand + 16 * i, regs, 16);
	}
#endif
	std::string name = brand;
	for (const std::string_view text: {"(R)", "(TM)", "(tm)", " CPU", " Processor"})
		for (size_t at; (at = name.find(text)) != std::string::npos;) name.erase(at, text.size());
	for (const std::string_view tail: {" @", " with "})
		if (const auto at = name.find(tail); at != std::string::npos) name.erase(at);
	if (const auto at = name.find("-Core"); at != std::string::npos) { // "AMD Ryzen 9 5950X 16-Core"
		const auto word  = name.rfind(' ', at);
		const auto first = word == std::string::npos ? at : word;
		name.erase(first, at + 5 - first);
	}
	if (const auto at = name.find(" Gen "); at != std::string::npos && at < 6) name.erase(0, at + 5); // "13th Gen "
	std::string compact;
	for (const char c: name)
		if (c != ' ' || (!compact.empty() && compact.back() != ' ')) compact.push_back(c);
	while (!compact.empty() && compact.back() == ' ') compact.pop_back();
	return compact;
}

void Publish(Machine&& machine) {
	machine.valid = true;
	std::lock_guard lock(g_machine_mutex);
	g_machine = std::move(machine);
}

#if defined(_WIN32)

template <typename Function>
bool Load(HMODULE module, Function& function, const char* symbol) {
	function = reinterpret_cast<Function>(reinterpret_cast<void*>(GetProcAddress(module, symbol)));
	return function != nullptr;
}

// NVIDIA's management library (the driver's nvml.dll), the few calls the panel makes by their C ABI.
struct Nvml {
	using Init    = int (*)();
	using Count   = int (*)(unsigned int*);
	using ByIndex = int (*)(unsigned int, void**);
	using Uuid    = int (*)(void*, char*, unsigned int);
	using Rates   = int (*)(void*, unsigned int*);         // nvmlUtilization_t: GPU and memory %
	using Value   = int (*)(void*, int, unsigned int*);    // a temperature by sensor, a clock by kind
	using Power   = int (*)(void*, unsigned int*);         // milliwatts
	using Memory  = int (*)(void*, unsigned long long*);   // nvmlMemory_t: total, free, used
	static constexpr int TemperatureGpu = 0, ClockGraphics = 0, ClockMemory = 2;

	Rates  rates       = nullptr;
	Value  temperature = nullptr, clock = nullptr;
	Power  power       = nullptr;
	Memory memory      = nullptr;
	void*  device      = nullptr;

	bool Open(const std::array<uint8_t, 16>& uuid, const std::string& name) {
		HMODULE module = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		if (module == nullptr) {
			// (Drivers before 2019 kept it with nvidia-smi.)
			wchar_t path[MAX_PATH] {};
			if (ExpandEnvironmentStringsW(L"%ProgramW6432%\\NVIDIA Corporation\\NVSMI\\nvml.dll", path, MAX_PATH) != 0)
				module = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		}
		if (module == nullptr) return false;
		Init    init     = nullptr;
		Count   count    = nullptr;
		ByIndex by_index = nullptr;
		Uuid    get_uuid = nullptr;
		if (!Load(module, init, "nvmlInit_v2") || !Load(module, count, "nvmlDeviceGetCount_v2") ||
		    !Load(module, by_index, "nvmlDeviceGetHandleByIndex_v2") || !Load(module, get_uuid, "nvmlDeviceGetUUID") ||
		    !Load(module, rates, "nvmlDeviceGetUtilizationRates") || !Load(module, temperature, "nvmlDeviceGetTemperature") ||
		    !Load(module, clock, "nvmlDeviceGetClockInfo") || !Load(module, power, "nvmlDeviceGetPowerUsage") ||
		    !Load(module, memory, "nvmlDeviceGetMemoryInfo") || init() != 0)
			return false;
		// NVML's id of a GPU is the Vulkan device's UUID ("GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
		char wanted[48] {};
		std::snprintf(wanted, sizeof(wanted),
		              "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", uuid[0], uuid[1],
		              uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7], uuid[8], uuid[9], uuid[10], uuid[11], uuid[12],
		              uuid[13], uuid[14], uuid[15]);
		unsigned int devices = 0;
		if (count(&devices) != 0) return false;
		for (unsigned int i = 0; i < devices && device == nullptr; ++i) {
			void* handle = nullptr;
			char  text[96] {};
			if (by_index(i, &handle) == 0 && get_uuid(handle, text, sizeof(text)) == 0 && _stricmp(text, wanted) == 0)
				device = handle;
		}
		// (One NVIDIA GPU and the Vulkan device is NVIDIA's: that one.)
		if (device == nullptr && devices == 1 && name.find("NVIDIA") != std::string::npos) (void)by_index(0, &device);
		return device != nullptr;
	}

	void Sample(Machine* machine) const {
		unsigned int load[2] {}, value = 0;
		if (rates(device, load) == 0) {
			machine->gpu_load  = load[0];
			machine->gpu_valid = true;
		}
		if (temperature(device, TemperatureGpu, &value) == 0) machine->gpu_temperature = value;
		if (clock(device, ClockGraphics, &value) == 0) machine->gpu_clock = value;
		if (clock(device, ClockMemory, &value) == 0) machine->gpu_memory_clock = value;
		if (power(device, &value) == 0) machine->gpu_power = (value + 500) / 1000;
		unsigned long long bytes[3] {};
		if (memory(device, bytes) == 0) {
			machine->gpu_memory_total = bytes[0];
			machine->gpu_memory_used  = bytes[2];
		}
	}
};

// The logical processors' effective clocks as Task Manager shows them: each one's nominal frequency times its
// "% Processor Performance" (performance counters, pdh.dll).
struct Clocks {
	using Open    = PDH_STATUS(WINAPI*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
	using Add     = PDH_STATUS(WINAPI*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
	using Collect = PDH_STATUS(WINAPI*)(PDH_HQUERY);
	using Array   = PDH_STATUS(WINAPI*)(PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W);

	Collect              collect     = nullptr;
	Array                array       = nullptr;
	PDH_HQUERY           query       = nullptr;
	PDH_HCOUNTER         performance = nullptr, frequency = nullptr;
	std::vector<uint8_t> buffer;

	bool Start() {
		HMODULE module = LoadLibraryExW(L"pdh.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
		Open    open   = nullptr;
		Add     add    = nullptr;
		return module != nullptr && Load(module, open, "PdhOpenQueryW") && Load(module, add, "PdhAddEnglishCounterW") &&
		       Load(module, collect, "PdhCollectQueryData") && Load(module, array, "PdhGetFormattedCounterArrayW") &&
		       open(nullptr, 0, &query) == ERROR_SUCCESS &&
		       add(query, L"\\Processor Information(*)\\% Processor Performance", 0, &performance) == ERROR_SUCCESS &&
		       add(query, L"\\Processor Information(*)\\Processor Frequency", 0, &frequency) == ERROR_SUCCESS &&
		       collect(query) == ERROR_SUCCESS;
	}

	// Each instance "0,<processor>" of processor group 0 (not "_Total" and "0,_Total").
	void Read(PDH_HCOUNTER counter, std::vector<double>* values) {
		DWORD bytes = 0, items = 0;
		if (array(counter, PDH_FMT_DOUBLE, &bytes, &items, nullptr) != PDH_MORE_DATA) return;
		buffer.resize(bytes);
		auto* item = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
		if (array(counter, PDH_FMT_DOUBLE, &bytes, &items, item) != ERROR_SUCCESS) return;
		for (DWORD i = 0; i < items; ++i) {
			const wchar_t* name = item[i].szName;
			if (name == nullptr || name[0] != L'0' || name[1] != L',' || std::iswdigit(name[2]) == 0) continue;
			const auto index = std::wcstoul(name + 2, nullptr, 10);
			const auto status = item[i].FmtValue.CStatus;
			if (index < values->size() && (status == PDH_CSTATUS_VALID_DATA || status == PDH_CSTATUS_NEW_DATA))
				(*values)[index] = item[i].FmtValue.doubleValue;
		}
	}

	void Sample(size_t cpus, std::vector<uint32_t>* mhz) {
		if (collect(query) != ERROR_SUCCESS) return;
		std::vector<double> percent(cpus, -1.0), nominal(cpus, -1.0);
		Read(performance, &percent);
		Read(frequency, &nominal);
		mhz->assign(cpus, 0);
		for (size_t i = 0; i < cpus; ++i)
			if (percent[i] >= 0.0 && nominal[i] > 0.0) (*mhz)[i] = static_cast<uint32_t>(nominal[i] * percent[i] / 100.0 + 0.5);
	}
};

// Each logical processor's idle, kernel (idle included) and user time (NtQuerySystemInformation's
// SystemProcessorPerformanceInformation).
struct ProcessorTimes {
	LARGE_INTEGER idle, kernel, user, dpc, interrupt;
	ULONG         interrupts;
};
using QuerySystemInformation = LONG(NTAPI*)(ULONG, PVOID, ULONG, PULONG);

uint64_t FileTimeTicks(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}

// The render thread (Kyty.Gpu), for its CPU time.
HANDLE OpenRenderThread() {
	HANDLE      found    = nullptr;
	const DWORD process  = GetCurrentProcessId();
	HANDLE      snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) return nullptr;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0 && found == nullptr; more = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != process) continue;
		HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(thread, &description)) && description != nullptr) {
			if (std::wcscmp(description, L"Kyty.Gpu") == 0) found = thread;
			LocalFree(description);
		}
		if (found != thread) CloseHandle(thread);
	}
	CloseHandle(snapshot);
	return found;
}

void Run(std::array<uint8_t, 16> uuid, std::string gpu_name) {
	LocalPlatform::SetThreadName("Kyty.Overlay");
	(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	Nvml       nvml;
	const bool gpu = nvml.Open(uuid, gpu_name);
	Clocks     clocks;
	const bool have_clocks = clocks.Start();
	const auto query       = reinterpret_cast<QuerySystemInformation>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
	const size_t cpus = std::min<size_t>(GetActiveProcessorCount(0), 64);
	// The efficiency class of each logical processor (group 0).
	std::vector<uint8_t> classes(cpus, 0);
	{
		ULONG length = 0;
		(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
		std::vector<uint8_t> buffer(length);
		if (length != 0 && GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data()), length,
		                                              &length, GetCurrentProcess(), 0)) {
			for (ULONG offset = 0; offset < length;) {
				const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
				if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 && info->CpuSet.LogicalProcessorIndex < cpus)
					classes[info->CpuSet.LogicalProcessorIndex] = info->CpuSet.EfficiencyClass;
				offset += info->Size;
			}
		}
	}
	const std::string cpu_name = CpuName();
	HANDLE            render   = nullptr;
	std::vector<ProcessorTimes> last(cpus), now(cpus);
	uint64_t last_process = 0, last_render = 0, last_tsc = 0, samples = 0;
	int64_t  last_ns      = 0;
	// (first: no last sample to take differences from.)
	for (bool first = true;; std::this_thread::sleep_for(std::chrono::milliseconds(500))) {
		Machine machine;
		machine.gpu_name   = gpu_name;
		machine.cpu_name   = cpu_name;
		machine.core_class = classes;
		const int64_t  ns  = NowNs();
		const uint64_t tsc = __rdtsc();
		const double   seconds = static_cast<double>(ns - last_ns) / 1e9;
		const uint32_t parts = kyty_local_overlay_sampling.load(std::memory_order_relaxed);
		if (parts == 0) {
			first = true;
			continue;
		}
		// (Each part's time: a slow one is logged.)
		const auto timed = [&](const char* what, auto&& part) {
			const auto start = std::chrono::steady_clock::now();
			part();
			const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
			if (ms >= 5.0 || samples < 2) std::printf("Overlay sample: %s %.2f ms\n", what, ms);
		};
		if (gpu && (parts & 1u) != 0) timed("GPU", [&] { nvml.Sample(&machine); });
		if ((parts & 4u) != 0) (void)LocalPlatform::QueryVideoMemory(&machine.video_usage, &machine.video_budget);
		// Processors: the load of each since the last sample (busy = kernel + user - idle).
		ULONG returned = 0;
		if ((parts & 4u) != 0 && query != nullptr &&
		    query(8, now.data(), static_cast<ULONG>(now.size() * sizeof(ProcessorTimes)), &returned) == 0) {
			if (!first) {
				machine.core_load.resize(cpus);
				double busy_sum = 0, total_sum = 0;
				for (size_t i = 0; i < cpus; ++i) {
					const auto idle  = static_cast<double>(now[i].idle.QuadPart - last[i].idle.QuadPart);
					const auto total = static_cast<double>(now[i].kernel.QuadPart - last[i].kernel.QuadPart +
					                                       now[i].user.QuadPart - last[i].user.QuadPart);
					machine.core_load[i] = total > 0 ? static_cast<float>(std::clamp(1.0 - idle / total, 0.0, 1.0)) : 0.0f;
					busy_sum += total - idle;
					total_sum += total;
				}
				machine.cpu_load = total_sum > 0 ? static_cast<float>(std::clamp(busy_sum / total_sum, 0.0, 1.0)) : 0.0f;
			}
			last = now;
		}
		if (have_clocks && (parts & 2u) != 0) timed("clocks", [&] { clocks.Sample(cpus, &machine.core_clock); });
		// The emulator's CPU time, and its render thread's (cycles: precise over half a second).
		FILETIME creation {}, exit {}, kernel {}, user {};
		if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) != 0) {
			const uint64_t process = FileTimeTicks(kernel) + FileTimeTicks(user);
			if (!first && seconds > 0) machine.process_cores = static_cast<float>(static_cast<double>(process - last_process) / 1e7 / seconds);
			last_process = process;
		}
		if (render == nullptr) render = OpenRenderThread();
		ULONG64 cycles = 0;
		if (render != nullptr && QueryThreadCycleTime(render, &cycles) != 0) {
			if (last_render != 0 && tsc > last_tsc)
				machine.render_thread = static_cast<float>(std::clamp(static_cast<double>(cycles - last_render) /
				                                                          static_cast<double>(tsc - last_tsc),
				                                                      0.0, 1.0));
			last_render = cycles;
		}
		MEMORYSTATUSEX status {};
		status.dwLength = sizeof(status);
		if (GlobalMemoryStatusEx(&status) != 0) {
			machine.ram_total = status.ullTotalPhys;
			machine.ram_used  = status.ullTotalPhys - status.ullAvailPhys;
		}
		machine.process_memory = LocalPlatform::ProcessMemory().working_set;
		last_ns  = ns;
		last_tsc = tsc;
		samples++;
		if (!first) Publish(std::move(machine));
		first = false;
	}
}

#else

#if defined(__linux__)
struct CpuTimes {
	uint64_t total = 0, idle = 0;
};

std::vector<CpuTimes> ReadCpuTimes() {
	std::vector<CpuTimes> result;
	if (FILE* file = std::fopen("/proc/stat", "re")) {
		char line[512];
		while (std::fgets(line, sizeof(line), file)) {
			if (std::strncmp(line, "cpu", 3) != 0 || line[3] < '0' || line[3] > '9') continue;
			unsigned int id = 0;
			unsigned long long user = 0, nice = 0, system = 0, idle = 0, wait = 0, irq = 0, softirq = 0, steal = 0;
			if (std::sscanf(line, "cpu%u %llu %llu %llu %llu %llu %llu %llu %llu", &id, &user, &nice, &system,
			                &idle, &wait, &irq, &softirq, &steal) < 5 || id >= 65536)
				continue;
			if (result.size() <= id) result.resize(id + 1);
			// Guest times are already included in user/nice, so do not add them a second time.
			result[id] = {user + nice + system + idle + wait + irq + softirq + steal, idle + wait};
		}
		std::fclose(file);
	}
	return result;
}

std::vector<uint32_t> ReadCpuClocks() {
	std::vector<uint32_t> result;
	if (FILE* file = std::fopen("/proc/cpuinfo", "re")) {
		char line[512];
		unsigned int id = 65536;
		while (std::fgets(line, sizeof(line), file)) {
			unsigned int processor = 0;
			double mhz = 0;
			if (std::sscanf(line, "processor : %u", &processor) == 1) {
				id = processor;
				if (id < 65536 && result.size() <= id) result.resize(id + 1, 0);
			} else if (id < result.size() && std::sscanf(line, "cpu MHz : %lf", &mhz) == 1 && mhz > 0)
				result[id] = static_cast<uint32_t>(mhz);
		}
		std::fclose(file);
	}
	return result;
}
#endif

void Run(std::array<uint8_t, 16> /*uuid*/, std::string gpu_name) {
	LocalPlatform::SetThreadName("Kyty.Overlay");
	const std::string cpu_name = CpuName();
#if defined(__linux__)
	std::vector<CpuTimes> last;
	double last_process = 0, last_render = 0;
	int64_t last_ns = 0;
#endif
	for (;;) {
		Machine machine;
		machine.gpu_name        = gpu_name;
		machine.cpu_name        = cpu_name;
		machine.ram_total       = LocalPlatform::PhysicalMemory();
		machine.process_memory  = LocalPlatform::ProcessMemory().working_set;
#if defined(__linux__)
		const auto parts = kyty_local_overlay_sampling.load(std::memory_order_relaxed);
		const int64_t ns = NowNs();
		if ((parts & 4u) != 0) {
			machine.ram_used = machine.ram_total - std::min(machine.ram_total, LocalPlatform::AvailablePhysicalMemory());
			auto now = ReadCpuTimes();
			machine.core_load.resize(now.size(), 0);
			machine.core_class.resize(now.size(), 0);
			double busy_sum = 0, total_sum = 0;
			for (size_t i = 0; i < now.size() && i < last.size(); ++i) {
				if (last[i].total == 0 || now[i].total < last[i].total || now[i].idle < last[i].idle) continue;
				const auto total = now[i].total - last[i].total;
				const auto idle = std::min(total, now[i].idle - last[i].idle);
				if (total != 0) machine.core_load[i] = static_cast<float>(static_cast<double>(total - idle) / total);
				busy_sum += static_cast<double>(total - idle);
				total_sum += static_cast<double>(total);
			}
			if (total_sum > 0) machine.cpu_load = static_cast<float>(busy_sum / total_sum);
			last = std::move(now);
			const double seconds = static_cast<double>(ns - last_ns) / 1e9;
			timespec process_time {};
			if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &process_time) == 0) {
				const double process = static_cast<double>(process_time.tv_sec) + static_cast<double>(process_time.tv_nsec) / 1e9;
				if (last_ns != 0 && seconds > 0 && process >= last_process)
					machine.process_cores = static_cast<float>((process - last_process) / seconds);
				last_process = process;
			}
			const double render = LocalPlatform::NamedThreadsCpuSeconds("Kyty.Gpu");
			if (last_ns != 0 && last_render > 0 && seconds > 0 && render >= last_render)
				machine.render_thread = static_cast<float>(std::clamp((render - last_render) / seconds, 0.0, 1.0));
			last_render = render;
			last_ns = ns;
		} else {
			last.clear();
			last_ns = 0;
			last_process = last_render = 0;
		}
		if ((parts & 2u) != 0) machine.core_clock = ReadCpuClocks();
#endif
		Publish(std::move(machine));
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
}

#endif

} // namespace

std::atomic<int>& LevelState() {
	static std::atomic<int> level = [] {
		const char* value = std::getenv("KYTY_FPS_HUD");
		if (value == nullptr || *value == '\0' || std::string_view(value) == "0") return 0;
		return std::atoi(value) >= 2 ? 2 : 1;
	}();
	return level;
}

int Level() {
	return LevelState().load(std::memory_order_relaxed);
}

int CycleLevel() {
	auto& level = LevelState();
	int   next  = (level.load(std::memory_order_relaxed) + 1) % 3;
	level.store(next, std::memory_order_relaxed);
	return next;
}

void NoteFlip(int64_t now_ns, int64_t idle_ns) {
	const auto flip = g_flips.load(std::memory_order_relaxed);
	g_flip_ns[flip % Ring].store(now_ns, std::memory_order_relaxed);
	g_flip_idle_ns[flip % Ring].store(idle_ns, std::memory_order_relaxed);
	g_flips.store(flip + 1, std::memory_order_release);
}

void RecentFrames(size_t count, std::vector<float>* frame_ms, std::vector<float>* busy_ms) {
	const auto     flips = g_flips.load(std::memory_order_acquire);
	const uint64_t n     = std::min<uint64_t>({count, flips > 1 ? flips - 1 : 0, Ring / 2});
	frame_ms->resize(n);
	if (busy_ms != nullptr) busy_ms->resize(n);
	for (uint64_t i = 0; i < n; ++i) {
		// (Frame k: from flip k - 1 to flip k.)
		const uint64_t k  = flips - n + i;
		const auto     ns = g_flip_ns[k % Ring].load(std::memory_order_relaxed) -
		                g_flip_ns[(k - 1) % Ring].load(std::memory_order_relaxed);
		(*frame_ms)[i] = static_cast<float>(static_cast<double>(ns) / 1e6);
		if (busy_ms != nullptr) {
			const auto idle = g_flip_idle_ns[k % Ring].load(std::memory_order_relaxed) -
			                  g_flip_idle_ns[(k - 1) % Ring].load(std::memory_order_relaxed);
			(*busy_ms)[i] = static_cast<float>(static_cast<double>(std::max<int64_t>(ns - idle, 0)) / 1e6);
		}
	}
}

Summary Summarize(double seconds) {
	Summary    summary;
	const auto flips = g_flips.load(std::memory_order_acquire);
	const auto from  = NowNs() - static_cast<int64_t>(seconds * 1e9);
	// (Frame k: from flip k - 1 to flip k; the ones that ended since `from`.)
	std::vector<double> times;
	for (uint64_t k = flips; k >= 2 && flips - k < Ring / 2;) {
		--k;
		const auto end = g_flip_ns[k % Ring].load(std::memory_order_relaxed);
		if (end < from) break;
		times.push_back(static_cast<double>(end - g_flip_ns[(k - 1) % Ring].load(std::memory_order_relaxed)) / 1e6);
	}
	if (times.empty()) return summary;
	summary.fps = static_cast<double>(times.size()) / seconds;
	std::sort(times.begin(), times.end(), std::greater<>());
	const size_t worst = (times.size() + 99) / 100;
	double       sum   = 0;
	for (size_t i = 0; i < worst; ++i) sum += times[i];
	summary.low1   = sum > 0 ? 1000.0 * static_cast<double>(worst) / sum : 0;
	summary.max_ms = times.front();
	return summary;
}

void StartMachine(const uint8_t* device_uuid, const std::string& device_name) {
	std::call_once(g_machine_started, [&] {
		std::array<uint8_t, 16> uuid {};
		if (device_uuid != nullptr) std::memcpy(uuid.data(), device_uuid, uuid.size());
		std::thread(Run, uuid, device_name).detach();
	});
}

Machine LatestMachine() {
	std::lock_guard lock(g_machine_mutex);
	return g_machine;
}

} // namespace PerfOverlay
