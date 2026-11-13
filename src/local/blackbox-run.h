#pragma once
// The crash / freeze black box (default on; KYTY_BLACKBOX=0: off). A PC that froze so hard it had to be restarted left
// an .out.log that stops in the middle of a line: what the process was doing then was lost. Here a watchdog thread
// (Kyty.Blackbox, THREAD_PRIORITY_HIGHEST, started with the GPU thread) writes one line a second to
// logs/blackbox-<the run's log stamp>.txt, each write through to the disk (FILE_FLAG_WRITE_THROUGH; FlushFileBuffers only
// after the file is made and after a STALL record's parts), so the last second before a freeze is on the disk after the
// restart.
//
// The file: a header (HeaderSlots slots), a ring of RingSlots one-line slots (Slot bytes each, padded with spaces; the
// newest line is the one before the "#------" end marker, or sort by the #sequence), then the STALL records (at most
// StallBytes). Its size is bounded: ~300 KB of ring and header, written at the start, plus at most 2 MiB of records.
//
// A STALL record: when the guest's flips or the host's presents stop for StallNs (3 s) while the game is not paused, its
// window not minimized and it is not quitting, a second thread (Kyty.BlackboxDump, waiting from the start so that nothing is created then)
// samples every thread's stack twice, a second apart (LocalPlatform::SampleThread: suspended for microseconds, pc and
// callers), with each thread's CPU time: raw addresses first (written and flushed), then the same symbolized (guest code
// as module+offset; host code as dll+offset), which takes the runtime linker's lock and the loader's (written after,
// so a lock held by a stuck thread costs only the symbols). At most MaxStalls a session, one per stop.
//
// Each second the line is written twice: first a heartbeat (seq, time, flips, presents, the render thread's phase; it
// ends in HB-ONLY), before any call that could block (GetThreadTimes, the scene tags, K32GetProcessMemoryInfo - which
// takes the working set's lock -, GlobalMemoryStatusEx), then the full line over it. A newest line still HB-ONLY: that
// second stopped between the two.
//
// The external watchdog (KYTY_BLACKBOX_WATCHDOG=0: none): this exe again as a small process of its own
// (LocalPlatform::WatchdogMain, --kyty-watchdog <pid>; nothing of the emulator runs in it), reading the block the game
// shares with it (blackbox.h Shared: the heartbeat, the 1 Hz thread's stage, flips, presents, the render thread's
// phase) and writing logs/watchdog-<stamp>.txt a line a second, through to the disk. The game's heartbeat stopped >= 5 s
// while the watchdog's lines go on: "GAME HUNG, PC alive", and once a small minidump (threads, stacks, memory map; no
// memory contents) in logs/hang-<stamp>.dmp. Both files stopping at the same second: the whole PC (or its disk) froze.
//
// It reads only atomics and process / system counters: no guest memory is allocated or freed, nothing touches Vulkan,
// nothing the game or the renderer does changes.

#include "blackbox.h"
#include "live-census.h"
#include "live-control.h"
#include "local-platform.h"
#include "scene-tags.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Blackbox {

constexpr uint64_t Slot        = 512;
constexpr uint64_t HeaderSlots = 5;
constexpr uint64_t RingSlots   = 600; // 10 minutes
constexpr uint64_t RingBase    = HeaderSlots * Slot;
constexpr uint64_t StallBase   = RingBase + RingSlots * Slot;
constexpr uint64_t StallBytes  = uint64_t {2} << 20u;
constexpr int64_t  StallNs     = 3'000'000'000;
constexpr int      MaxStalls   = 3;
constexpr size_t   StackWords  = 24;
constexpr uint64_t ArmFlips    = 120; // (the game flipped for a while: not the start-up)

struct Box {
	uint64_t              file = 0;
	uint32_t              watch_tid = 0;
	// The dump's request: the watchdog fills `reason` and `line` while `busy` is its, then counts `requests` up.
	std::atomic<bool>     busy {false};
	std::atomic<uint32_t> requests {0};
	uint32_t              stall_number = 0;
	char                  reason[256] {};
	char                  line[Slot] {};
	uint64_t              stall_end = StallBase; // (the dump thread's)
	Shared*               shared    = nullptr;   // (the watchdog's block: blackbox.h Shared; nullptr without one)
};
inline Box g_box;

inline void Append(char* out, size_t size, size_t& used, const char* format, ...) {
	if (used + 1 >= size) return;
	va_list args;
	va_start(args, format);
	const int wrote = std::vsnprintf(out + used, size - used, format, args);
	va_end(args);
	if (wrote > 0) used = std::min(size - 1, used + static_cast<size_t>(wrote));
}

// Local wall time: "YYYY-MM-DD HH:MM:SS.mmm" (date) or "HH:MM:SS.mmm".
inline void WallTime(char* out, size_t size, bool date) {
	const auto        now = std::chrono::system_clock::now();
	const std::time_t t   = std::chrono::system_clock::to_time_t(now);
	const auto ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
	std::tm tm {};
#if defined(_WIN32)
	(void)localtime_s(&tm, &t);
#else
	(void)localtime_r(&t, &tm);
#endif
	if (date)
		std::snprintf(out, size, "%04d-%02d-%02d %02d:%02d:%02d.%03d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
		              tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
	else
		std::snprintf(out, size, "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
}

// One slot: the text padded with spaces, a newline last.
inline void FillSlot(char* slot, const char* text) {
	const size_t length = std::min(std::strlen(text), static_cast<size_t>(Slot - 1));
	std::memset(slot, ' ', Slot);
	std::memcpy(slot, text, length);
	slot[Slot - 1] = '\n';
}
inline void WriteSlot(uint64_t offset, const char* text) {
	char slot[Slot];
	FillSlot(slot, text);
	(void)LocalPlatform::WriteScratchFile(g_box.file, offset, slot, sizeof(slot));
}

inline double Gib(uint64_t bytes) {
	return static_cast<double>(bytes) / static_cast<double>(uint64_t {1} << 30u);
}

inline int64_t AgeMs(int64_t now, int64_t since) {
	return since != 0 && now > since ? (now - since) / 1'000'000 : 0;
}

// The 1 Hz thread's stage, in the shared block (nothing without one): the watchdog's lines say where it stopped.
inline void SetStage(Stage stage) {
	if (Shared* shared = g_box.shared; shared != nullptr) {
		shared->stage_ns.store(NowNs(), std::memory_order_relaxed);
		shared->stage.store(stage, std::memory_order_release);
	}
}

// A ring slot and the end marker after it: one write when they are next to each other (all but the ring's last slot).
// No FlushFileBuffers here: the handle writes through already (the data is on the disk when the write returns; the
// file's size and extents were flushed by Open), and a flush a second would empty a SATA drive's command queue (FLUSH
// CACHE is not queued) under the game's streaming reads.
inline void WriteRingLine(uint64_t sequence, const char* line) {
	static constexpr const char* EndMarker = "#------ (the ring's end: the line above is the newest, older ones follow)";
	const uint64_t               at        = sequence % RingSlots;
	char                         slots[Slot * 2];
	FillSlot(slots, line);
	FillSlot(slots + Slot, EndMarker);
	if (at + 1 < RingSlots) {
		(void)LocalPlatform::WriteScratchFile(g_box.file, RingBase + at * Slot, slots, sizeof(slots));
	} else {
		(void)LocalPlatform::WriteScratchFile(g_box.file, RingBase + at * Slot, slots, Slot);
		(void)LocalPlatform::WriteScratchFile(g_box.file, RingBase, slots + Slot, Slot);
	}
}

// A thread's phase for the line: " <label>=<phase>:<age>ms" (in it) or " <label>=after-<phase>:<age>ms" (left it).
inline void AppendPhase(char* line, size_t size, size_t& used, const char* label, const ThreadPhase& phase, int64_t now) {
	const uint32_t code  = phase.code.load(std::memory_order_relaxed);
	const int64_t  since = phase.since.load(std::memory_order_relaxed);
	if (code == PhaseNone) return;
	Append(line, size, used, " %s=%s%s:%lldms", label, (code & PhaseAfter) != 0 ? "after-" : "", PhaseName(code),
	       static_cast<long long>(AgeMs(now, since)));
}

// The STALL record's text past the record area's end is dropped.
inline void StallWrite(const std::string& text) {
	const uint64_t limit = StallBase + StallBytes;
	if (g_box.stall_end >= limit || text.empty()) return;
	const auto size = static_cast<size_t>(std::min<uint64_t>(text.size(), limit - g_box.stall_end));
	(void)LocalPlatform::WriteScratchFile(g_box.file, g_box.stall_end, text.data(), size);
	(void)LocalPlatform::FlushWrittenFile(g_box.file);
	g_box.stall_end += size;
}

struct Sampled {
	uint32_t    tid = 0;
	std::string name;
	uint64_t    handle = 0;
	bool        opened = false;
	double      cpu[2] {};
	bool        ok[2] {};
	uint64_t    words[2][StackWords] {};
};

// An address as guest module+offset, host module+offset (Windows) or hex.
inline std::string Symbol(uint64_t address, std::unordered_map<uint64_t, std::string>& known) {
	if (const auto found = known.find(address); found != known.end()) return found->second;
	std::string text = Loader::DescribeGuestAddressForDiagnostics(address);
#if defined(_WIN32)
	if (text.empty()) {
		uint64_t base = 0;
		char     path[260] {};
		if (LocalPlatform::ModuleOf(reinterpret_cast<const void*>(address), &base, path, sizeof(path))) {
			const char* name = path;
			for (const char* c = path; *c != '\0'; ++c)
				if (*c == '\\' || *c == '/') name = c + 1;
			char line[320];
			std::snprintf(line, sizeof(line), "%s+0x%llx", *name != '\0' ? name : "?",
			              static_cast<unsigned long long>(address - base));
			text = line;
		}
	}
#endif
	if (text.empty()) {
		char line[32];
		std::snprintf(line, sizeof(line), "%llx", static_cast<unsigned long long>(address));
		text = line;
	}
	known.emplace(address, text);
	return text;
}

inline void SampleRound(std::vector<Sampled>& threads, int round) {
	for (auto& thread: threads) {
		if (!thread.opened) continue;
		thread.cpu[round] = LocalPlatform::ThreadCpuSeconds(thread.handle);
		thread.ok[round]  = LocalPlatform::SampleThread(thread.handle, 0, 0, thread.words[round], StackWords);
	}
}

inline std::string RawLines(const std::vector<Sampled>& threads, int round) {
	std::string out;
	char        line[96];
	for (const auto& thread: threads) {
		if (!thread.opened) continue;
		if (round == 0)
			std::snprintf(line, sizeof(line), "S1 %u %s cpu=%.2fs:", thread.tid, thread.name.empty() ? "-" : thread.name.c_str(),
			              thread.cpu[0]);
		else
			std::snprintf(line, sizeof(line), "S2 %u %s +cpu=%.3fs%s:", thread.tid, thread.name.empty() ? "-" : thread.name.c_str(),
			              thread.cpu[1] - thread.cpu[0],
			              thread.ok[0] && thread.ok[1] && thread.words[0][0] == thread.words[1][0] ? " same-pc" : "");
		out += line;
		if (!thread.ok[round]) out += " (not sampled)";
		for (const auto word: thread.words[round]) {
			if (!thread.ok[round] || word == 0) break;
			std::snprintf(line, sizeof(line), " %llx", static_cast<unsigned long long>(word));
			out += line;
		}
		out += '\n';
	}
	return out;
}

inline void Dump() {
	char now_text[32];
	WallTime(now_text, sizeof(now_text), true);
	std::string head = "\n==== STALL #" + std::to_string(g_box.stall_number) + " at " + now_text + ": " + g_box.reason +
	                   "\n  last line: " + g_box.line + "\n";
	StallWrite(head);

	// Every thread of the process but these two, sampled twice a second apart (raw), then symbolized.
	LocalPlatform::PrepareSampling();
	std::vector<Sampled> threads;
	const auto           self = LocalPlatform::ThreadId();
	for (auto& [tid, name]: LocalPlatform::ProcessThreads()) {
		if (tid == self || tid == g_box.watch_tid) continue;
		Sampled thread;
		thread.tid    = tid;
		thread.name   = std::move(name);
		thread.handle = LocalPlatform::OpenThreadForSampling(tid);
		thread.opened = thread.handle != 0;
		threads.push_back(std::move(thread));
	}
	SampleRound(threads, 0);
	StallWrite("  threads: " + std::to_string(threads.size()) +
	           " (S1: pc and callers, raw; S2: a second later, with the CPU time between; Y1/Y2: the same symbolized)\n" +
	           RawLines(threads, 0));
	std::this_thread::sleep_for(std::chrono::seconds(1));
	SampleRound(threads, 1);
	StallWrite(RawLines(threads, 1));
	for (auto& thread: threads) {
		LocalPlatform::CloseThreadForSampling(thread.handle);
		thread.handle = 0;
	}

	std::unordered_map<uint64_t, std::string> known;
	for (int round = 0; round < 2; ++round) {
		std::string out;
		for (const auto& thread: threads) {
			if (!thread.opened || !thread.ok[round]) continue;
			out += round == 0 ? "Y1 " : "Y2 ";
			out += std::to_string(thread.tid) + " " + (thread.name.empty() ? "-" : thread.name) + ":";
			for (const auto word: thread.words[round]) {
				if (word == 0) break;
				out += " " + Symbol(word, known);
			}
			out += '\n';
		}
		StallWrite(out);
	}
	StallWrite("==== STALL #" + std::to_string(g_box.stall_number) + " end\n");
}

inline void DumpLoop() {
	LocalPlatform::SetThreadName("Kyty.BlackboxDump");
	uint32_t done = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (g_box.requests.load(std::memory_order_acquire) == done) continue;
		done = g_box.requests.load(std::memory_order_acquire);
		Dump();
		g_box.busy.store(false, std::memory_order_release);
	}
}

// The file made, its header and blank ring written (on the watchdog's thread: nothing on the GPU thread's).
inline bool Open(const std::string& path) {
	g_box.file = LocalPlatform::OpenWriteThroughFile(path.c_str());
	if (g_box.file == 0) {
		std::printf("[blackbox] %s could not be made: no black box this run\n", path.c_str());
		std::fflush(stdout);
		return false;
	}
	(void)SceneTags::Current(); // (its state made before the loop reads it)

	// The header and the ring, blank, written once: the file's size stays the same while the ring turns.
	std::string blank(static_cast<size_t>(StallBase), ' ');
	for (uint64_t at = Slot - 1; at < StallBase; at += Slot) blank[static_cast<size_t>(at)] = '\n';
	(void)LocalPlatform::WriteScratchFile(g_box.file, 0, blank.data(), blank.size());
	char when[32];
	WallTime(when, sizeof(when), true);
	char header[Slot];
	std::snprintf(header, sizeof(header), "KYTY BLACKBOX started %s, log %s; a line a second in the ring of %llu below (newest: "
	              "above the #------ line), STALL records after it", when, path.c_str(), static_cast<unsigned long long>(RingSlots));
	WriteSlot(0, header);
	WriteSlot(Slot, "fields: #seq time up=s since start | flip=guest flips/age of the last | pres=host presents/age, inpres=one in "
	                "progress for | rt=render thread: wait-<gpu|readback|download|compile|record>:for, idle:waiting for guest "
	                "work for, busy | rtcpu=its CPU share of the second");
	WriteSlot(Slot * 2, "fields: sub=queue submits (insub=one in progress for) | fence=GPU waits in progress/for | pso=pipelines "
	                    "being made/total/for | rd=file reads in flight/total/MiB asked/for | scene=last map loaded | fr=frames "
	                    "3d or no3d (loading, menu, movie)/for s | file=last read (its end)");
	WriteSlot(Slot * 3, "fields: PAUSED MINIMIZED QUITTING | priv=process commit ws=working set commit=system commit/limit avail=free RAM "
	                    "| max1s=longest frame of the second | slow=last frames >= 20 ms@age | STALL=flips or presents "
	                    "stopped > 3 s (a record below)");
	WriteSlot(Slot * 4, "fields: ph/pph=the render / presenting thread's last phase (submit fence pso read fault protect present; "
	                    "after-X: left it, nothing since):for | rd .../@-ms=the last read began | flt/prot=guest faults / protection "
	                    "changes in progress/total(/for) | a newest line ending HB-ONLY: the heartbeat, the second stopped "
	                    "before its full line (a memory query or the disk: watchdog-*.txt names the stage)");
	(void)LocalPlatform::FlushWrittenFile(g_box.file);
	std::printf("[blackbox] %s (a line a second, written through to the disk; KYTY_BLACKBOX=0: off)\n", path.c_str());
	std::fflush(stdout);
	return true;
}

// The external watchdog (KYTY_BLACKBOX_WATCHDOG=0: none): this exe again, as its own small process
// (LocalPlatform::WatchdogMain, --kyty-watchdog <pid>), which reads the shared block every second and writes
// logs/watchdog-<stamp>.txt: the game's heartbeat stopping while its own lines go on is a hung game (then, once, a small
// minidump, logs/hang-<stamp>.dmp); its lines stopping at the same moment is the whole PC.
inline void StartWatchdog(const std::string& path) {
	if (const char* value = std::getenv("KYTY_BLACKBOX_WATCHDOG"); value != nullptr && std::strcmp(value, "0") == 0) return;
	const auto        slash = path.find_last_of("/\\");
	const std::string dir   = slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
	std::string       stamp = slash == std::string::npos ? path : path.substr(slash + 1);
	if (stamp.rfind("blackbox-", 0) == 0) stamp.erase(0, 9);
	if (const auto dot = stamp.rfind(".txt"); dot != std::string::npos) stamp.resize(dot);
	const std::string watchdog = dir + "watchdog-" + stamp + ".txt";
	const std::string dump     = dir + "hang-" + stamp + ".dmp";
	if (watchdog.size() >= SharedPath || dump.size() >= SharedPath) return;
	void* memory = LocalPlatform::CreateBlackboxShared(sizeof(Shared));
	if (memory == nullptr) {
		std::printf("[blackbox] no watchdog: its shared block could not be made\n");
		std::fflush(stdout);
		return;
	}
	auto* shared = new (memory) Shared;
	shared->pid  = LocalPlatform::ProcessId();
	shared->size = sizeof(Shared);
	std::memcpy(shared->watchdog_path, watchdog.c_str(), watchdog.size() + 1);
	std::memcpy(shared->dump_path, dump.c_str(), dump.size() + 1);
	std::atomic_thread_fence(std::memory_order_release);
	shared->magic = SharedMagic;
	g_box.shared  = shared;
	const bool started = LocalPlatform::StartBlackboxWatchdog();
	std::printf("[blackbox] watchdog %s: %s (a line a second; the game hung: its heartbeat stops there, the PC froze: "
	            "those lines stop too; KYTY_BLACKBOX_WATCHDOG=0: none)\n",
	            started ? "started" : "could not be started", watchdog.c_str());
	std::fflush(stdout);
}

inline void WatchLoop(std::string path) {
	LocalPlatform::SetThreadName("Kyty.Blackbox");
	LocalPlatform::MakeCriticalThread(); // (THREAD_PRIORITY_HIGHEST: it runs ~1 ms a second)
	g_box.watch_tid = LocalPlatform::ThreadId();
	if (!Open(path)) return;
	std::thread(DumpLoop).detach();
	StartWatchdog(path);
	static constexpr const char* WaitNames[] = {"gpu", "readback", "download", "compile", "record"};
	static_assert(std::size(WaitNames) == LiveCensus::Waits);
	struct Slow {
		double  ms = 0;
		int64_t at = 0;
	};
	Slow          slow[2] {};
	uint64_t      scanned     = 0; // flips whose intervals were looked at
	int64_t       excused     = 0; // the last time seen paused or minimized
	double        last_cpu    = 0;
	int64_t       last_cpu_at = 0;
	bool          in_stall    = false;
	int           stalls      = 0;
	const int64_t start       = NowNs();
	auto          next        = std::chrono::steady_clock::now();
	for (uint64_t sequence = 0;; ++sequence) {
		const int64_t now = NowNs();
		char          line[Slot] {};
		size_t        used = 0;
		char          wall[32];
		WallTime(wall, sizeof(wall), false);
		Append(line, sizeof(line), used, "#%06llu %s up=%llds", static_cast<unsigned long long>(sequence), wall,
		       static_cast<long long>((now - start) / 1'000'000'000));

		const uint64_t flips        = LiveControl::g_flips.load(std::memory_order_acquire);
		const int64_t  last_flip    = flips != 0 ? LiveControl::g_flip_times[(flips - 1) % LiveControl::FlipTimes].load(std::memory_order_relaxed) : 0;
		const int64_t  last_present = g_last_present_ns.load(std::memory_order_relaxed);
		const bool     paused       = g_paused.load(std::memory_order_relaxed);
		const bool     minimized    = g_minimized.load(std::memory_order_relaxed);
		const bool     quitting     = g_quitting.load(std::memory_order_relaxed);

		// The heartbeat, before anything that could block (a memory query under the working set's lock, a write to a
		// disk the page file keeps busy): the shared block first (memory only), then a short line in this second's slot,
		// which the full line replaces at the end. A ring whose newest line ends in HB-ONLY: that second stopped between.
		if (Shared* shared = g_box.shared; shared != nullptr) {
			shared->flips.store(flips, std::memory_order_relaxed);
			shared->flip_ns.store(last_flip, std::memory_order_relaxed);
			shared->presents.store(g_presents.total.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->present_ns.store(last_present, std::memory_order_relaxed);
			shared->render_phase.store(g_phases[RoleRender].code.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->render_phase_ns.store(g_phases[RoleRender].since.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->flags.store((paused ? SharedPaused : 0u) | (minimized ? SharedMinimized : 0u) | (quitting ? SharedQuitting : 0u),
			                    std::memory_order_relaxed);
			shared->reads.store(g_reads.now.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->faults.store(g_faults.now.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->pipelines.store(g_pipelines.now.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->submits.store(g_submits.now.load(std::memory_order_relaxed), std::memory_order_relaxed);
			shared->beat_ns.store(now, std::memory_order_relaxed);
			shared->beat.store(sequence + 1, std::memory_order_release);
		}
		{
			char   beat[Slot] {};
			size_t beat_used = used;
			std::memcpy(beat, line, used);
			Append(beat, sizeof(beat), beat_used, " flip=%llu/%lldms pres=%llu/%lldms", static_cast<unsigned long long>(flips),
			       static_cast<long long>(AgeMs(now, last_flip)),
			       static_cast<unsigned long long>(g_presents.total.load(std::memory_order_relaxed)),
			       static_cast<long long>(AgeMs(now, last_present)));
			AppendPhase(beat, sizeof(beat), beat_used, "ph", g_phases[RoleRender], now);
			Append(beat, sizeof(beat), beat_used, " HB-ONLY");
			SetStage(StageBeatWrite);
			WriteRingLine(sequence, beat);
		}

		// The guest's flips: how many, the age of the last; the slow ones among the new (>= 20 ms, as SLOW Frame), the
		// longest of this second.
		double longest = 0;
		for (uint64_t i = std::max(scanned, flips > 4096 ? flips - 4096 : uint64_t {0}); i < flips; ++i) {
			if (i == 0) continue;
			const int64_t at     = LiveControl::g_flip_times[i % LiveControl::FlipTimes].load(std::memory_order_relaxed);
			const int64_t before = LiveControl::g_flip_times[(i - 1) % LiveControl::FlipTimes].load(std::memory_order_relaxed);
			if (at <= before) continue;
			const double ms = static_cast<double>(at - before) / 1e6;
			longest         = std::max(longest, ms);
			if (ms >= 20.0) {
				slow[1] = slow[0];
				slow[0] = {ms, at};
			}
		}
		scanned = flips;
		Append(line, sizeof(line), used, " flip=%llu/%lldms", static_cast<unsigned long long>(flips),
		       static_cast<long long>(AgeMs(now, last_flip)));

		// The host's presents: how many, the age of the last; one in progress, and for how long.
		Append(line, sizeof(line), used, " pres=%llu/%lldms", static_cast<unsigned long long>(g_presents.total.load(std::memory_order_relaxed)),
		       static_cast<long long>(AgeMs(now, last_present)));
		if (g_presents.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, " inpres=%lldms", static_cast<long long>(AgeMs(now, g_presents.since.load(std::memory_order_relaxed))));

		// The render (GPU) thread: waiting for something of its own (LiveCensus::Wait), for the guest's work, or busy;
		// its CPU share of the last second.
		if (const uint32_t wait = g_render_wait.load(std::memory_order_relaxed); wait != 0 && wait <= std::size(WaitNames))
			Append(line, sizeof(line), used, " rt=wait-%s:%lldms", WaitNames[wait - 1],
			       static_cast<long long>(AgeMs(now, g_render_wait_since.load(std::memory_order_relaxed))));
		else if (const int64_t idle = LiveControl::g_render_idle_since.load(std::memory_order_relaxed); idle != 0)
			Append(line, sizeof(line), used, " rt=idle:%lldms", static_cast<long long>(AgeMs(now, idle)));
		else
			Append(line, sizeof(line), used, " rt=busy");
		AppendPhase(line, sizeof(line), used, "ph", g_phases[RoleRender], now);
		AppendPhase(line, sizeof(line), used, "pph", g_phases[RolePresent], now);
		if (LiveControl::g_render_known.load(std::memory_order_acquire)) {
			SetStage(StageCpu);
			const double cpu = LocalPlatform::ThreadCpuSeconds(LiveControl::g_render_thread);
			if (last_cpu_at != 0 && now > last_cpu_at)
				Append(line, sizeof(line), used, " rtcpu=%.0f%%", (cpu - last_cpu) * 1e11 / static_cast<double>(now - last_cpu_at));
			last_cpu    = cpu;
			last_cpu_at = now;
		}

		// Submissions, GPU waits, pipelines being made, file reads: counts, and the age of those in progress.
		Append(line, sizeof(line), used, " sub=%llu", static_cast<unsigned long long>(g_submits.total.load(std::memory_order_relaxed)));
		if (g_submits.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, " insub=%lldms", static_cast<long long>(AgeMs(now, g_submits.since.load(std::memory_order_relaxed))));
		Append(line, sizeof(line), used, " fence=%d", g_fence_waits.now.load(std::memory_order_relaxed));
		if (g_fence_waits.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, "/%lldms", static_cast<long long>(AgeMs(now, g_fence_waits.since.load(std::memory_order_relaxed))));
		Append(line, sizeof(line), used, " pso=%d/%llu", g_pipelines.now.load(std::memory_order_relaxed),
		       static_cast<unsigned long long>(g_pipelines.total.load(std::memory_order_relaxed)));
		if (g_pipelines.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, "/%lldms", static_cast<long long>(AgeMs(now, g_pipelines.since.load(std::memory_order_relaxed))));
		Append(line, sizeof(line), used, " rd=%d/%llu/%lluM", g_reads.now.load(std::memory_order_relaxed),
		       static_cast<unsigned long long>(g_reads.total.load(std::memory_order_relaxed)),
		       static_cast<unsigned long long>(g_read_bytes.load(std::memory_order_relaxed) >> 20u));
		if (g_reads.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, "/%lldms", static_cast<long long>(AgeMs(now, g_reads.since.load(std::memory_order_relaxed))));
		if (const int64_t began = g_reads.last.load(std::memory_order_relaxed); began != 0)
			Append(line, sizeof(line), used, "@-%lldms", static_cast<long long>(AgeMs(now, began)));
		Append(line, sizeof(line), used, " flt=%d/%llu", g_faults.now.load(std::memory_order_relaxed),
		       static_cast<unsigned long long>(g_faults.total.load(std::memory_order_relaxed)));
		if (g_faults.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, "/%lldms", static_cast<long long>(AgeMs(now, g_faults.since.load(std::memory_order_relaxed))));
		Append(line, sizeof(line), used, " prot=%d/%llu", g_protects.now.load(std::memory_order_relaxed),
		       static_cast<unsigned long long>(g_protects.total.load(std::memory_order_relaxed)));
		if (g_protects.now.load(std::memory_order_relaxed) > 0)
			Append(line, sizeof(line), used, "/%lldms", static_cast<long long>(AgeMs(now, g_protects.since.load(std::memory_order_relaxed))));

		// The scene (the map the game loaded last), the frames' kind and since when, the last file read.
		SetStage(StageScene);
		const uint32_t scene = SceneTags::Current();
		if (scene != SceneTags::None)
			Append(line, sizeof(line), used, " scene=m%02x_%02x_%02x_%02x", scene >> 24u, (scene >> 16u) & 0xffu,
			       (scene >> 8u) & 0xffu, scene & 0xffu);
		else
			Append(line, sizeof(line), used, " scene=-");
		const uint32_t frame_kind = g_scene_frame.load(std::memory_order_relaxed);
		Append(line, sizeof(line), used, " fr=%s/%llds", frame_kind == 1 ? "3d" : frame_kind == 2 ? "no3d" : "-",
		       static_cast<long long>(AgeMs(now, g_scene_frame_since.load(std::memory_order_relaxed)) / 1000));
		char file[FileWords * 8 + 1] {};
		for (size_t i = 0; i < FileWords; ++i) {
			const uint64_t word = g_last_file[i].load(std::memory_order_relaxed);
			std::memcpy(file + i * 8, &word, 8);
		}
		file[FileWords * 8] = '\0';
		for (char& c: file)
			if (c != '\0' && (static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) >= 0x7f || c == ' ')) c = '_';
		Append(line, sizeof(line), used, " file=%s", file[0] != '\0' ? file : "-");

		if (paused) Append(line, sizeof(line), used, " PAUSED");
		if (minimized) Append(line, sizeof(line), used, " MINIMIZED");
		if (quitting) Append(line, sizeof(line), used, " QUITTING");

		// A stall: the guest's flips or the host's presents stopped StallNs while not paused nor minimized (their ages from
		// the last time they were, or from the last flip / present).
		if (paused || minimized || quitting) excused = now;
		const int64_t flip_age    = now - std::max(last_flip, excused);
		const int64_t present_age = last_present != 0 ? now - std::max(last_present, excused) : 0;
		const bool    stalled     = flips >= ArmFlips && (flip_age > StallNs || present_age > StallNs);
		if (stalled) Append(line, sizeof(line), used, " STALL");

		// Memory: the process's commit and working set, the system's commit and available RAM.
		SetStage(StageProcMem);
		const auto process = LocalPlatform::ProcessMemory();
		SetStage(StageSysMem);
		const auto machine = LocalPlatform::SystemMemoryStatus();
		if (Shared* shared = g_box.shared; shared != nullptr) {
			shared->private_bytes.store(process.private_bytes, std::memory_order_relaxed);
			shared->working_set.store(process.working_set, std::memory_order_relaxed);
		}
		Append(line, sizeof(line), used, " priv=%.1fG ws=%.1fG commit=%.1f/%.1fG avail=%.1fG", Gib(process.private_bytes),
		       Gib(process.working_set), Gib(machine.commit), Gib(machine.commit_limit), Gib(machine.available));

		Append(line, sizeof(line), used, " max1s=%.0fms", longest);
		for (const auto& frame: slow)
			if (frame.at != 0)
				Append(line, sizeof(line), used, " slow=%.0fms@-%llds", frame.ms, static_cast<long long>(AgeMs(now, frame.at) / 1000));

		// The full line, over this second's heartbeat line.
		SetStage(StageLineWrite);
		WriteRingLine(sequence, line);

		if (stalled && !in_stall && stalls < MaxStalls && !g_box.busy.exchange(true, std::memory_order_acq_rel)) {
			++stalls;
			g_box.stall_number = static_cast<uint32_t>(stalls);
			std::snprintf(g_box.reason, sizeof(g_box.reason),
			              "guest flips stopped %.1f s ago, host presents %.1f s ago%s (render thread %s)",
			              static_cast<double>(AgeMs(now, last_flip)) / 1000.0, static_cast<double>(AgeMs(now, last_present)) / 1000.0,
			              g_presents.now.load(std::memory_order_relaxed) > 0 ? ", one present in progress" : "",
			              g_render_wait.load(std::memory_order_relaxed) != 0 ? "waiting"
			              : LiveControl::g_render_idle_since.load(std::memory_order_relaxed) != 0 ? "idle (no guest work)"
			                                                                                      : "busy");
			std::memcpy(g_box.line, line, sizeof(g_box.line));
			g_box.line[sizeof(g_box.line) - 1] = '\0';
			g_box.requests.fetch_add(1, std::memory_order_release);
		}
		in_stall = stalled;

		// Once a second, on time (a late wake-up - the PC stalled - does not make up the seconds missed).
		next += std::chrono::seconds(1);
		if (const auto wake = std::chrono::steady_clock::now(); next <= wake) next = wake + std::chrono::seconds(1);
		SetStage(StageSleep);
		std::this_thread::sleep_until(next);
	}
}

// The GPU thread, at its start (LiveControl::Start): once.
inline void Start() {
	static std::atomic_bool started {false};
	if (started.exchange(true)) return;
	if (const char* value = std::getenv("KYTY_BLACKBOX"); value != nullptr && std::strcmp(value, "0") == 0) return;
	// logs/blackbox-<stamp>.txt beside the run's log (KYTY_RUN_LOG: logs/<stamp>.out.log), else logs/ here with the time.
	std::string path;
	if (const char* log = std::getenv("KYTY_RUN_LOG"); log != nullptr && *log != '\0') {
		const std::string run   = log;
		const auto        slash = run.find_last_of("/\\");
		std::string       stamp = slash == std::string::npos ? run : run.substr(slash + 1);
		if (const auto dot = stamp.find(".out.log"); dot != std::string::npos) stamp.resize(dot);
		path = (slash == std::string::npos ? std::string() : run.substr(0, slash + 1)) + "blackbox-" + stamp + ".txt";
	} else {
		const std::time_t t = std::time(nullptr);
		std::tm           tm {};
#if defined(_WIN32)
		(void)localtime_s(&tm, &t);
#else
		(void)localtime_r(&t, &tm);
#endif
		char stamp[32];
		std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
		              tm.tm_hour, tm.tm_min, tm.tm_sec);
		path = std::string("logs/blackbox-") + stamp + ".txt";
	}
	std::thread(WatchLoop, std::move(path)).detach();
}

} // namespace Blackbox
