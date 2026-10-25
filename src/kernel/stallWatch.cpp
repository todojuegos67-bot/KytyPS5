#include "kernel/stallWatch.h"

#include "common/common.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/renderer/threadSampler.h"
#include "kernel/pthread.h"
#include "loader/elf.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#undef min
#undef max
#else
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel {

namespace {

constexpr size_t SlotCount = 1024;

std::array<StallWatchSlot, SlotCount> g_slots;
std::atomic<bool>                     g_enabled {true};

// Files read, per window of FileWindowS seconds (the current and the previous one are printed).
constexpr double FileWindowS = 30.0;
struct FileReads {
	uint64_t count       = 0;
	uint64_t bytes       = 0;
	int64_t  last_offset = 0;
	uint64_t last_size   = 0;
	uint64_t distinct    = 0; // Reads whose offset differed from the previous read's.
};
struct FileReadState {
	std::mutex                                 mutex;
	std::unordered_map<std::string, FileReads> current;
	std::unordered_map<std::string, FileReads> previous;
	std::unordered_map<std::string, FileReads> ten; // since the last [loader] line
};
FileReadState* const g_file_reads = new FileReadState; // (never destroyed, as g_ranges)

void RotateFileReads() {
	std::scoped_lock lock(g_file_reads->mutex);
	g_file_reads->previous = std::move(g_file_reads->current);
	g_file_reads->current.clear();
}

void DescribeFileReads(std::string* out) {
	std::unordered_map<std::string, FileReads> merged;
	{
		std::scoped_lock lock(g_file_reads->mutex);
		merged = g_file_reads->previous;
		for (const auto& [path, reads]: g_file_reads->current) {
			auto& m = merged[path];
			m.count += reads.count;
			m.bytes += reads.bytes;
			m.distinct += reads.distinct;
			m.last_offset = reads.last_offset;
			m.last_size   = reads.last_size;
		}
	}
	std::vector<std::pair<std::string, FileReads>> rows(merged.begin(), merged.end());
	std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	uint64_t total = 0;
	for (const auto& row: rows) {
		total += row.second.count;
	}
	char line[1024];
	std::snprintf(line, sizeof(line),
	              "STALLWATCH files read in the last %.0f-%.0f s: %" PRIu64 " reads of %zu files (most read first)\n",
	              FileWindowS, FileWindowS * 2.0, total, rows.size());
	*out += line;
	for (size_t i = 0; i < rows.size() && i < 16; i++) {
		const auto& [path, reads] = rows[i];
		std::snprintf(line, sizeof(line),
		              "STALLWATCH file %s: %" PRIu64 " reads, %.2f MB, %" PRIu64
		              " at a new offset; last offset 0x%" PRIx64 " size 0x%" PRIx64 "\n",
		              path.c_str(), reads.count, static_cast<double>(reads.bytes) / 1e6, reads.distinct,
		              static_cast<uint64_t>(reads.last_offset), reads.last_size);
		*out += line;
	}
}

int64_t NowNs() {
	return std::chrono::steady_clock::now().time_since_epoch().count();
}

std::string LoadName(const StallWatchSlot& slot) {
	std::string name;
	for (const auto& c: slot.name) {
		const char value = c.load(std::memory_order_relaxed);
		if (value == '\0') {
			break;
		}
		name.push_back(value >= 32 && value < 127 ? value : '?');
	}
	return name;
}

struct CodeRange {
	uint64_t    low  = 0;
	uint64_t    high = 0;
	uint64_t    base = 0;
	std::string name;
};

std::vector<CodeRange> GuestCodeRanges() {
	std::vector<CodeRange> ranges;
	auto*                  linker = Common::Singleton<Loader::RuntimeLinker>::Instance();
	if (linker == nullptr) {
		return ranges;
	}
	int misses = 0;
	for (int32_t id = 0; id < 4096 && misses < 64; id++) {
		auto* program = linker->FindProgramById(id);
		if (program == nullptr || program->elf == nullptr) {
			misses += id == 0 ? 0 : 1;
			continue;
		}
		misses           = 0;
		const auto* ehdr = program->elf->GetEhdr();
		const auto* phdr = program->elf->GetPhdr();
		if (ehdr == nullptr || phdr == nullptr) {
			continue;
		}
		const auto name = Common::PathToString(program->file_name.filename());
		for (Loader::Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
			if (phdr[i].p_type == Loader::PT_LOAD && (phdr[i].p_flags & Loader::PF_X) != 0 && phdr[i].p_memsz != 0) {
				const uint64_t low = program->base_vaddr + phdr[i].p_vaddr;
				if (std::ranges::none_of(ranges, [&](const CodeRange& r) { return r.low == low; })) {
					ranges.push_back({low, low + phdr[i].p_memsz, program->base_vaddr, name});
				}
			}
		}
	}
	return ranges;
}

// Refreshed every 10 s by its own thread, so a dump never takes the linker's lock (a hang may hold it).
// (Never destroyed: the watch threads may still run while the process exits.)
struct RangesState {
	std::mutex             mutex;
	std::vector<CodeRange> ranges;
};
RangesState* const g_ranges = new RangesState;

void RefreshRanges() {
	auto ranges = GuestCodeRanges();
	std::scoped_lock lock(g_ranges->mutex);
	g_ranges->ranges = std::move(ranges);
}

std::vector<CodeRange> CachedRanges() {
	std::scoped_lock lock(g_ranges->mutex);
	return g_ranges->ranges;
}

const CodeRange* FindCode(const std::vector<CodeRange>& ranges, uint64_t address) {
	for (const auto& range: ranges) {
		if (address >= range.low && address < range.high) {
			return &range;
		}
	}
	return nullptr;
}

std::string DescribeCode(const std::vector<CodeRange>& ranges, uint64_t address) {
	char text[160];
	if (const auto* range = FindCode(ranges, address)) {
		std::snprintf(text, sizeof(text), "%s+0x%" PRIx64, range->name.c_str(), address - range->base);
	} else {
		std::snprintf(text, sizeof(text), "host:0x%" PRIx64, address);
	}
	return text;
}

// Reads another thread's stack without faulting on an unmapped page.
size_t SafeRead(uint64_t address, void* out, size_t size) {
	if (address == 0 || size == 0) {
		return 0;
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION region {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &region, sizeof(region)) == 0 ||
	    region.State != MEM_COMMIT) {
		return 0;
	}
	const auto end = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
	size           = static_cast<size_t>(std::min<uint64_t>(size, end - address));
	SIZE_T got     = 0;
	if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), out, size,
	                      &got) == 0) {
		return static_cast<size_t>(got);
	}
	return static_cast<size_t>(got);
#else
	iovec local {out, size};
	iovec remote {reinterpret_cast<void*>(address), size};
	const auto got = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
	return got > 0 ? static_cast<size_t>(got) : 0;
#endif
}

constexpr size_t StackScanBytes = 32 * 1024;
constexpr int    MaxGuestFrames = 16;

void ScanStack(const std::vector<CodeRange>& ranges, const uint8_t* copy, size_t size,
               std::string* out) {
	int      found = 0;
	uint64_t last  = 0;
	for (size_t offset = 0; offset + 8 <= size && found < MaxGuestFrames; offset += 8) {
		uint64_t value = 0;
		std::memcpy(&value, copy + offset, 8);
		if (value == last || FindCode(ranges, value) == nullptr) {
			continue;
		}
		last = value;
		*out += found == 0 ? " stack:" : "";
		*out += " " + DescribeCode(ranges, value);
		found++;
	}
	if (found == 0) {
		*out += " stack: (no guest return address found)";
	}
}

struct Snapshot {
	uint64_t flips    = 0;
	uint64_t gpu_work = 0;
	uint64_t io       = 0;
	int64_t  time     = 0;
};

void Dump(const char* reason, double seconds, double work_per_flip, double io_age_s, int index) {
	const auto ranges = CachedRanges();
	const auto now    = NowNs();
	std::string out;
	char        line[512];
	std::snprintf(line, sizeof(line),
	              "STALLWATCH dump %d: %s for %.0f s (flips %" PRIu64 ", %.0f draws+dispatches a flip, last file read %.0f s ago, "
	              "file reads %" PRIu64 ", AMPR commands rejected %" PRIu64 ")\n",
	              index, reason, seconds, StallWatchCounters::g_flips.load(), work_per_flip, io_age_s,
	              StallWatchCounters::g_io.load(), StallWatchCounters::g_ampr_rejected.load());
	out += line;
	auto stack = std::make_unique<uint8_t[]>(StackScanBytes);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto self = static_cast<uint64_t>(GetCurrentThreadId());
#endif
	for (auto& slot: g_slots) {
		auto* thread = static_cast<Pthread>(slot.thread.load(std::memory_order_acquire));
		if (thread == nullptr) {
			continue;
		}
		char name[32] {};
		if (PthreadGetname(thread, name) != 0) {
			std::snprintf(name, sizeof(name), "?");
		}
		const auto  tid   = slot.host_tid.load(std::memory_order_relaxed);
		const char* what  = slot.what.load(std::memory_order_acquire);
		const auto  since = slot.since.load(std::memory_order_relaxed);
		std::snprintf(line, sizeof(line), "STALLWATCH thread '%s' tid %" PRIu64 ": ", name, tid);
		out += line;
		if (what != nullptr) {
			const auto object_name = LoadName(slot);
			std::snprintf(line, sizeof(line), "waits in %s on 0x%" PRIx64 "%s%s%s for %.1f s", what,
			              slot.object.load(std::memory_order_relaxed), object_name.empty() ? "" : " '",
			              object_name.c_str(), object_name.empty() ? "" : "'",
			              static_cast<double>(now - since) / 1e9);
		} else {
			std::snprintf(line, sizeof(line), "no known wait (running, or blocked elsewhere)");
		}
		out += line;
		std::snprintf(line, sizeof(line), "; waits so far %" PRIu64 ";",
		              slot.waits.load(std::memory_order_relaxed));
		out += line;
		size_t got = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (tid != 0 && tid != self) {
			HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
			                               THREAD_QUERY_LIMITED_INFORMATION,
			                           FALSE, static_cast<DWORD>(tid));
			if (handle != nullptr) {
				CONTEXT context {};
				context.ContextFlags = CONTEXT_FULL;
				bool have            = false;
				// Nothing may allocate or take a lock while the thread is stopped.
				if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
					if (GetThreadContext(handle, &context) != 0) {
						have = true;
						got  = SafeRead(context.Rsp, stack.get(), StackScanBytes);
					}
					ResumeThread(handle);
				}
				CloseHandle(handle);
				if (have) {
					out += " rip " + DescribeCode(ranges, context.Rip) + ";";
				}
			}
		}
#endif
		if (got == 0) {
			got = SafeRead(slot.sp.load(std::memory_order_relaxed), stack.get(), StackScanBytes);
		}
		ScanStack(ranges, stack.get(), got, &out);
		out += "\n";
	}
	LibAmpr::DescribeStateForStallWatch(&out);
	DescribeFileReads(&out);
	std::fputs(out.c_str(), stdout);
	std::fflush(stdout);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Host stacks of every thread (the emulator's own too): module+offset, kyty_emulator.map.
	Libs::Graphics::DumpAllThreadStacks(stdout, "STALLWATCH host ");
	std::fflush(stdout);
#endif
	std::printf("STALLWATCH dump %d end\n", index);
	std::fflush(stdout);
}

// astro-tunnel-1025: the game's loader threads (Team Asobi's engine: Astro Bot), for the [loader] line and the
// "loaders idle in the same scene" dump. Matched by prefix of the guest thread name.
constexpr const char* LoaderThreads[] = {"LevelTask_ATQT", "RoomLoad_ATQT", "OdxAsyncLoader", "ProductNextLoad_ATQT",
                                         "ProductNextUnload_ATQT", "GfxTextureStreamerThread",
                                         "GfxModelStreamerThread", "sndz_stream_task_service"};

bool IsLoaderThread(const char* name) {
	for (const char* loader: LoaderThreads) {
		if (std::strncmp(name, loader, std::strlen(loader)) == 0) {
			return true;
		}
	}
	return false;
}

// Sum of the loader threads' finished waits (each wake-up of a loader is one): progress of the loaders.
uint64_t LoaderProgress() {
	uint64_t total = 0;
	for (auto& slot: g_slots) {
		auto* thread = static_cast<Pthread>(slot.thread.load(std::memory_order_acquire));
		if (thread == nullptr) {
			continue;
		}
		char name[32] {};
		if (PthreadGetname(thread, name) == 0 && IsLoaderThread(name) &&
		    std::strncmp(name, "Gfx", 3) != 0 && std::strncmp(name, "sndz", 4) != 0) {
			total += slot.waits.load(std::memory_order_relaxed);
		}
	}
	return total;
}

// Every 10 s with KYTY_FPS_LOG: the files read since the last line and the guest threads that wait longest, plus
// each loader thread's wake-ups since the last line. No thread is stopped; slots are read with relaxed loads.
void LoaderLine(double seconds) {
	struct Row {
		std::string name;
		std::string text;
		double      wait_s = 0.0;
	};
	static std::unordered_map<const void*, uint64_t> s_last_waits;
	std::unordered_map<const void*, uint64_t>        waits_now;
	std::vector<Row>                                 waiting;
	std::string                                      loaders;
	const int64_t                                    now = NowNs();
	char                                             line[512];
	for (auto& slot: g_slots) {
		auto* thread = static_cast<Pthread>(slot.thread.load(std::memory_order_acquire));
		if (thread == nullptr) {
			continue;
		}
		char name[32] {};
		if (PthreadGetname(thread, name) != 0) {
			std::snprintf(name, sizeof(name), "?");
		}
		const auto  waits = slot.waits.load(std::memory_order_relaxed);
		const char* what  = slot.what.load(std::memory_order_acquire);
		const auto  since = slot.since.load(std::memory_order_relaxed);
		const auto  key   = static_cast<const void*>(&slot);
		waits_now[key]    = waits;
		const auto  last  = s_last_waits.find(key);
		const auto  delta = last == s_last_waits.end() ? waits : waits - last->second;
		const double wait_s = what != nullptr ? static_cast<double>(now - since) / 1e9 : 0.0;
		std::string  wait_text;
		if (what != nullptr) {
			const auto object_name = LoadName(slot);
			std::snprintf(line, sizeof(line), "%s%s%s%s %.1fs", what, object_name.empty() ? "" : " '",
			              object_name.c_str(), object_name.empty() ? "" : "'", wait_s);
			wait_text = line;
		} else {
			wait_text = "running";
		}
		if (IsLoaderThread(name)) {
			std::snprintf(line, sizeof(line), "%s%s +%" PRIu64 " (%s)", loaders.empty() ? "" : ", ", name, delta,
			              wait_text.c_str());
			loaders += line;
		}
		if (what != nullptr && wait_s >= 1.0) {
			waiting.push_back({name, wait_text, wait_s});
		}
	}
	s_last_waits = std::move(waits_now);
	std::ranges::sort(waiting, [](const Row& a, const Row& b) { return a.wait_s > b.wait_s; });

	std::unordered_map<std::string, FileReads> ten;
	{
		std::scoped_lock lock(g_file_reads->mutex);
		ten.swap(g_file_reads->ten);
	}
	std::vector<std::pair<std::string, FileReads>> files(ten.begin(), ten.end());
	std::ranges::sort(files, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	uint64_t reads = 0;
	uint64_t bytes = 0;
	for (const auto& [path, r]: files) {
		reads += r.count;
		bytes += r.bytes;
	}
	std::string out;
	std::snprintf(line, sizeof(line), "[loader] %.0f s: file reads %" PRIu64 " (%.1f MB, %zu files)", seconds, reads,
	              static_cast<double>(bytes) / 1e6, files.size());
	out += line;
	for (size_t i = 0; i < files.size() && i < 6; i++) {
		auto path = files[i].first;
		if (const auto cut = path.find_last_of("/\\"); cut != std::string::npos && path.size() - cut > 1) {
			// The last two parts of the path are enough to name the file.
			const auto cut2 = cut == 0 ? std::string::npos : path.find_last_of("/\\", cut - 1);
			path            = path.substr(cut2 == std::string::npos ? 0 : cut2 + 1);
		}
		std::snprintf(line, sizeof(line), "%s %s x%" PRIu64 " %.1fMB", i == 0 ? ":" : ",", path.c_str(),
		              files[i].second.count, static_cast<double>(files[i].second.bytes) / 1e6);
		out += line;
	}
	out += "\n[loader] loaders: " + (loaders.empty() ? std::string("none") : loaders) + "\n[loader] longest waits:";
	for (size_t i = 0; i < waiting.size() && i < 8; i++) {
		out += (i == 0 ? " " : ", ") + waiting[i].name + " " + waiting[i].text;
	}
	out += "\n";
	std::fputs(out.c_str(), stdout);
	std::fflush(stdout);
}

bool FpsLogEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_FPS_LOG");
		return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
	}();
	return on;
}

std::atomic<bool> g_dump_requested {false};

// A manual dump: F10 in the game window, or a file named stalldump.txt in the emulator's folder (removed then).
bool TakeDumpRequest() {
	if (g_dump_requested.exchange(false)) {
		return true;
	}
	std::error_code error;
	if (std::filesystem::exists("stalldump.txt", error)) {
		std::filesystem::remove("stalldump.txt", error);
		return true;
	}
	return false;
}

void Run(double threshold_s) {
	constexpr double LightFrame = 300.0; // draws+dispatches a flip
	constexpr int    MaxDumps   = 6;
	constexpr size_t History    = 512;   // samples, 500 ms apart
	std::vector<Snapshot> history;
	history.reserve(History);
	int     dumps          = 0;
	bool    armed_flip     = true;
	bool    armed_io       = true;
	bool    armed_light    = true;
	int64_t last_flip_time = NowNs();
	int64_t last_io_time   = 0;
	uint64_t last_flips    = 0;
	uint64_t last_io       = 0;
	int64_t  file_window   = NowNs();
	int64_t  loader_line   = NowNs();
	int      manual_dumps  = 0;
	// Loaders idle in the same scene: the loaders' progress, when it last changed, and the draws per flip of
	// the 10 s windows since (a loading tunnel draws the same each frame; play varies).
	constexpr double LoaderIdleS   = 45.0;
	uint64_t         loader_progress = LoaderProgress();
	int64_t          loader_change   = 0; // 0: no loader activity seen since the last such dump
	bool             armed_loader    = true;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		if (NowNs() - file_window >= static_cast<int64_t>(FileWindowS * 1e9)) {
			file_window = NowNs();
			RotateFileReads();
		}
		if (NowNs() - loader_line >= 10'000'000'000LL) {
			const double seconds = static_cast<double>(NowNs() - loader_line) / 1e9;
			loader_line          = NowNs();
			if (FpsLogEnabled()) {
				LoaderLine(seconds);
			}
		}
		if (TakeDumpRequest() && manual_dumps < 20) {
			Dump("manual request (F10 or stalldump.txt)", 0.0, -1.0, -1.0, 100 + ++manual_dumps);
		}
		if (const auto progress = LoaderProgress(); progress != loader_progress) {
			loader_progress = progress;
			loader_change   = NowNs();
			armed_loader    = true;
		}
		const Snapshot now {StallWatchCounters::g_flips.load(std::memory_order_relaxed),
		                    StallWatchCounters::g_gpu_work.load(std::memory_order_relaxed),
		                    StallWatchCounters::g_io.load(std::memory_order_relaxed), NowNs()};
		if (history.size() == History) {
			history.erase(history.begin());
		}
		history.push_back(now);
		if (now.flips != last_flips) {
			last_flips     = now.flips;
			last_flip_time = now.time;
			armed_flip     = true;
		}
		if (now.io != last_io) {
			last_io      = now.io;
			last_io_time = now.time;
			armed_io     = true;
		}
		const int64_t threshold = static_cast<int64_t>(threshold_s * 1e9);
		// Work per flip over the last `seconds`.
		const auto per_flip = [&](double seconds, bool* full) {
			const int64_t from = now.time - static_cast<int64_t>(seconds * 1e9);
			const Snapshot* first = nullptr;
			for (const auto& sample: history) {
				if (sample.time >= from) {
					first = &sample;
					break;
				}
			}
			*full = !history.empty() && history.front().time <= from;
			if (first == nullptr || now.flips == first->flips) {
				return -1.0;
			}
			return static_cast<double>(now.gpu_work - first->gpu_work) /
			       static_cast<double>(now.flips - first->flips);
		};
		if (dumps >= MaxDumps) {
			continue; // (manual dumps and the [loader] line go on)
		}
		const double io_age = last_io_time == 0 ? -1.0 : static_cast<double>(now.time - last_io_time) / 1e9;
		if (armed_flip && last_flips > 0 && now.time - last_flip_time >= threshold) {
			armed_flip = false;
			Dump("no flip", static_cast<double>(now.time - last_flip_time) / 1e9, -1.0, io_age, ++dumps);
			continue;
		}
		bool         full  = false;
		const double light = per_flip(threshold_s, &full);
		if (armed_io && last_io_time != 0 && now.time - last_io_time >= threshold && full && light >= 0.0 &&
		    light < LightFrame) {
			armed_io = false;
			Dump("light frames (a loading screen or a menu) and no file read", io_age, light, io_age, ++dumps);
			continue;
		}
		// The loaders woke up earlier, then none for 45 s, while flips go on and the scene draws the same: a level
		// load that never ends behind a loading animation (Astro Bot's light tunnel), which the light-frame test
		// misses (the tunnel is not light, and streamed audio keeps reading files).
		if (armed_loader && loader_change != 0 && now.time - loader_change >= static_cast<int64_t>(LoaderIdleS * 1e9) &&
		    now.time - last_flip_time < threshold) {
			bool         full_a = false;
			bool         full_b = false;
			const double recent = per_flip(LoaderIdleS / 3.0, &full_a);
			const double whole  = per_flip(LoaderIdleS, &full_b);
			if (full_b && recent > 0.0 && whole > 0.0 && std::abs(recent - whole) <= 0.15 * whole) {
				armed_loader = false;
				Dump("loaders idle in the same scene (a load that never ends?)",
				     static_cast<double>(now.time - loader_change) / 1e9, whole, io_age, ++dumps);
				continue;
			}
		}
		bool         long_full = false;
		const double long_light = per_flip(threshold_s * 6.0, &long_full);
		if (long_light >= LightFrame || long_light < 0.0) {
			armed_light = true;
		} else if (armed_light && long_full) {
			armed_light = false;
			Dump("light frames (a loading screen or a menu), file reads or not", threshold_s * 6.0, long_light,
			     io_age, ++dumps);
		}
	}
}

} // namespace

void StallWatchRequestDump() noexcept {
	g_dump_requested.store(true, std::memory_order_relaxed);
}

void StallWatchNoteFileRead(const std::filesystem::path& path, int64_t offset, uint64_t size) noexcept {
	if (!g_enabled.load(std::memory_order_relaxed)) {
		return;
	}
	try {
		auto             name = Common::PathToString(path);
		std::scoped_lock lock(g_file_reads->mutex);
		if (g_file_reads->current.size() >= 4096 && !g_file_reads->current.contains(name)) {
			name = "(other files)";
		}
		for (auto* map: {&g_file_reads->current, &g_file_reads->ten}) {
			if (map == &g_file_reads->ten && map->size() >= 1024 && !map->contains(name)) {
				continue;
			}
			auto& reads = (*map)[name];
			reads.distinct += reads.count == 0 || reads.last_offset != offset ? 1 : 0;
			reads.count++;
			reads.bytes += size;
			reads.last_offset = offset;
			reads.last_size   = size;
		}
	} catch (...) {
	}
}

void StallWatchRegisterThread(void* pthread, uint64_t host_tid, uint64_t /*stack_low*/,
                              uint64_t /*stack_high*/) noexcept {
	if (!g_enabled.load(std::memory_order_relaxed) || pthread == nullptr || t_stall_watch_slot != nullptr) {
		return;
	}
	for (auto& slot: g_slots) {
		void* expected = nullptr;
		if (slot.thread.load(std::memory_order_relaxed) == nullptr &&
		    slot.thread.compare_exchange_strong(expected, pthread, std::memory_order_acq_rel)) {
			slot.host_tid.store(host_tid, std::memory_order_relaxed);
			slot.what.store(nullptr, std::memory_order_relaxed);
			slot.waits.store(0, std::memory_order_relaxed);
			slot.sp.store(0, std::memory_order_relaxed);
			slot.StoreName(nullptr);
			t_stall_watch_slot = &slot;
			return;
		}
	}
}

void StallWatchUnregisterThread() noexcept {
	auto* slot = t_stall_watch_slot;
	if (slot == nullptr) {
		return;
	}
	t_stall_watch_slot = nullptr;
	slot->what.store(nullptr, std::memory_order_relaxed);
	slot->host_tid.store(0, std::memory_order_relaxed);
	slot->thread.store(nullptr, std::memory_order_release);
}

void StallWatchStart() {
	static std::atomic<bool> started {false};
	if (started.exchange(true)) {
		return;
	}
	if (const char* off = std::getenv("KYTY_STALL_WATCH"); off != nullptr && std::strcmp(off, "0") == 0) {
		g_enabled.store(false);
		std::printf("Stall watch: off (KYTY_STALL_WATCH=0)\n");
		return;
	}
	double seconds = 10.0;
	if (const char* value = std::getenv("KYTY_STALL_WATCH_S"); value != nullptr && std::atof(value) > 0.0) {
		seconds = std::atof(value);
	}
	std::printf("Stall watch: on (a STALLWATCH thread dump after %.0f s without a flip, or of light frames "
	            "without file reads; KYTY_STALL_WATCH=0 off, KYTY_STALL_WATCH_S=<seconds>)\n",
	            seconds);
	std::fflush(stdout);
	std::thread([seconds] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SetThreadDescription(GetCurrentThread(), L"Kyty.StallWatch");
#endif
		Run(seconds);
	}).detach();
	std::thread([] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SetThreadDescription(GetCurrentThread(), L"Kyty.StallWatchModules");
#endif
		for (;;) {
			RefreshRanges();
			std::this_thread::sleep_for(std::chrono::seconds(10));
		}
	}).detach();
}

} // namespace Libs::LibKernel
