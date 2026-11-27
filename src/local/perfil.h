#pragma once
// (perfil) KYTY_PERFIL=1 (off by default; off = one static bool read once at the GPU thread's start): a sampling
// profiler of the render (GPU, PM4 translation) thread for a whole run, so one run says where that thread's time goes.
//   - A thread (Kyty.Perfil) wakes every KYTY_PERFIL_US (default 1000) microseconds (high-resolution waitable timer)
//     and takes one sample with LocalPlatform::SampleThread: SuspendThread, GetThreadContext, unwind with the images'
//     .pdata (RtlLookupFunctionEntry-style tables of PrepareSampling, frame pointers past them), ResumeThread. Nothing
//     is allocated or printed while the thread is suspended: the frames go into a preallocated buffer.
//   - The samples add up in fixed, preallocated open-addressing tables: per leaf address (self), per frame address
//     counted once per sample (inclusive, by call site), per whole stack (16 frames).
//   - Every 30 s and at the process's exit: logs/perfil-<the run's log stamp>.txt (beside KYTY_RUN_LOG, as the black
//     box), addresses as module+RVA (exe+RVA: symbolize with the same build's .map, tools/perfil-simbolos.py), and one
//     [perfil] line in the log.
// Cost when on: one suspension a millisecond of the render thread (~10-40 us, ~1-3%), the tables ~6 MB.
#include "local-platform.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Perfil {

constexpr size_t Frames = 16;

inline bool On() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PERFIL");
		return v != nullptr && v[0] != '\0' && v[0] != '0';
	}();
	return on;
}

#if defined(_WIN32)
struct Slot {
	uint64_t key;
	uint64_t count;
};
struct StackSlot {
	uint64_t hash;
	uint64_t count;
	uint64_t words[Frames];
};

constexpr size_t AddrSlots  = 1u << 16u;
constexpr size_t StackSlots = 1u << 14u;

struct Tables {
	Slot      self[AddrSlots];
	Slot      incl[AddrSlots];
	StackSlot stacks[StackSlots];
	uint64_t  samples;
	uint64_t  failed;
	uint64_t  dropped;
};

inline Tables*     g_t = nullptr;
inline SRWLOCK     g_lock = SRWLOCK_INIT; // (only between the sampler and the writer; never while the target is suspended)
inline std::string g_path;
inline uint64_t    g_period_us = 1000;

inline uint64_t Mix(uint64_t x) {
	x ^= x >> 33u;
	x *= 0xff51afd7ed558ccdull;
	x ^= x >> 33u;
	return x;
}

inline void Add(Slot* table, uint64_t key) {
	if (key == 0) return;
	size_t i = Mix(key) & (AddrSlots - 1);
	for (size_t n = 0; n < AddrSlots; ++n, i = (i + 1) & (AddrSlots - 1)) {
		if (table[i].key == key) {
			++table[i].count;
			return;
		}
		if (table[i].key == 0) {
			table[i].key   = key;
			table[i].count = 1;
			return;
		}
	}
	++g_t->dropped;
}

inline void AddStack(const uint64_t* w) {
	uint64_t h = 0x9e3779b97f4a7c15ull;
	for (size_t k = 0; k < Frames; ++k) h = Mix(h ^ w[k]) + k;
	if (h == 0) h = 1;
	size_t i = h & (StackSlots - 1);
	for (size_t n = 0; n < 64; ++n, i = (i + 1) & (StackSlots - 1)) {
		StackSlot& s = g_t->stacks[i];
		if (s.hash == h && std::memcmp(s.words, w, sizeof(s.words)) == 0) {
			++s.count;
			return;
		}
		if (s.hash == 0) {
			s.hash  = h;
			s.count = 1;
			std::memcpy(s.words, w, sizeof(s.words));
			return;
		}
	}
	++g_t->dropped;
}

// Address -> "module+0xRVA" (module list cached; refreshed on a miss).
struct Module {
	uint64_t    base;
	uint64_t    end;
	std::string name;
};
inline std::vector<Module>& Modules() {
	static std::vector<Module> m;
	return m;
}
inline const Module* FindModule(uint64_t a) {
	for (const auto& m: Modules())
		if (a >= m.base && a < m.end) return &m;
	uint64_t base = 0;
	char     path[MAX_PATH] {};
	if (!LocalPlatform::ModuleOf(reinterpret_cast<const void*>(a), &base, path, sizeof(path))) return nullptr;
	const char* slash = std::strrchr(path, '\\');
	std::string name  = slash != nullptr ? slash + 1 : path;
	uint64_t                 end = base + 1;
	// The image's size from its PE header (SizeOfImage).
	const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
	if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + static_cast<uint64_t>(dos->e_lfanew));
		if (nt->Signature == IMAGE_NT_SIGNATURE) end = base + nt->OptionalHeader.SizeOfImage;
	}
	if (base == reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr))) name = "exe";
	Modules().push_back({base, end, std::move(name)});
	return &Modules().back();
}
inline std::string Name(uint64_t a) {
	char buf[96];
	if (const Module* m = FindModule(a)) {
		std::snprintf(buf, sizeof(buf), "%s+0x%llx", m->name.c_str(), static_cast<unsigned long long>(a - m->base));
	} else {
		std::snprintf(buf, sizeof(buf), "?0x%llx", static_cast<unsigned long long>(a));
	}
	return buf;
}

inline SRWLOCK g_write_lock = SRWLOCK_INIT; // (the module cache and the file: one writer at a time)
inline void     WriteLocked(bool at_exit);
inline void     Write(bool at_exit) {
	if (g_t == nullptr) return;
	AcquireSRWLockExclusive(&g_write_lock);
	WriteLocked(at_exit);
	ReleaseSRWLockExclusive(&g_write_lock);
}
inline void WriteLocked(bool at_exit) {
	std::vector<Slot>      self, incl;
	std::vector<StackSlot> stacks;
	uint64_t               samples = 0, failed = 0, dropped = 0;
	AcquireSRWLockExclusive(&g_lock);
	for (const auto& s: g_t->self)
		if (s.key != 0) self.push_back(s);
	for (const auto& s: g_t->incl)
		if (s.key != 0) incl.push_back(s);
	for (const auto& s: g_t->stacks)
		if (s.hash != 0) stacks.push_back(s);
	samples = g_t->samples;
	failed  = g_t->failed;
	dropped = g_t->dropped;
	ReleaseSRWLockExclusive(&g_lock);
	if (samples == 0) return;
	auto by = [](const auto& a, const auto& b) { return a.count > b.count; };
	std::sort(self.begin(), self.end(), by);
	std::sort(incl.begin(), incl.end(), by);
	std::sort(stacks.begin(), stacks.end(), by);
	// Modules (by the leaf).
	std::vector<std::pair<std::string, uint64_t>> mods;
	for (const auto& s: self) {
		const Module* m    = FindModule(s.key);
		std::string   name = m != nullptr ? m->name : "?";
		auto it = std::find_if(mods.begin(), mods.end(), [&](const auto& p) { return p.first == name; });
		if (it == mods.end()) mods.emplace_back(name, s.count);
		else it->second += s.count;
	}
	std::sort(mods.begin(), mods.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

	FILE* f = std::fopen(g_path.c_str(), "w");
	if (f == nullptr) return;
	const double total = static_cast<double>(samples);
	const uint64_t exe = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
	std::fprintf(f, "# perfil KYTY_PERFIL: hilo de render (GPU), cada %llu us%s\n",
	             static_cast<unsigned long long>(g_period_us), at_exit ? " (final)" : "");
	std::fprintf(f, "samples %llu failed %llu dropped %llu exe_base 0x%llx\n", static_cast<unsigned long long>(samples),
	             static_cast<unsigned long long>(failed), static_cast<unsigned long long>(dropped),
	             static_cast<unsigned long long>(exe));
	std::fprintf(f, "\n[modules]\n");
	for (const auto& [n, c]: mods) std::fprintf(f, "%8llu %6.2f%% %s\n", static_cast<unsigned long long>(c), 100.0 * c / total, n.c_str());
	std::fprintf(f, "\n[self top 60]\n");
	for (size_t i = 0; i < self.size() && i < 60; ++i)
		std::fprintf(f, "%8llu %6.2f%% %s\n", static_cast<unsigned long long>(self[i].count), 100.0 * self[i].count / total,
		             Name(self[i].key).c_str());
	std::fprintf(f, "\n[inclusive top 40] (por direccion de retorno; perfil-simbolos.py agrupa por funcion)\n");
	for (size_t i = 0; i < incl.size() && i < 40; ++i)
		std::fprintf(f, "%8llu %6.2f%% %s\n", static_cast<unsigned long long>(incl[i].count), 100.0 * incl[i].count / total,
		             Name(incl[i].key).c_str());
	std::fprintf(f, "\n[stacks top 200] (hoja primero)\n");
	for (size_t i = 0; i < stacks.size() && i < 200; ++i) {
		std::fprintf(f, "%8llu %6.2f%%", static_cast<unsigned long long>(stacks[i].count), 100.0 * stacks[i].count / total);
		for (uint64_t w: stacks[i].words)
			if (w != 0) std::fprintf(f, " %s", Name(w).c_str());
		std::fprintf(f, "\n");
	}
	// Everything, for the script (per function totals).
	std::fprintf(f, "\n[raw self]\n");
	for (const auto& s: self) std::fprintf(f, "%llu %s\n", static_cast<unsigned long long>(s.count), Name(s.key).c_str());
	std::fprintf(f, "\n[raw inclusive]\n");
	for (const auto& s: incl) std::fprintf(f, "%llu %s\n", static_cast<unsigned long long>(s.count), Name(s.key).c_str());
	std::fclose(f);

	std::string line = "[perfil] " + std::to_string(samples) + " muestras -> " + g_path + " | top:";
	for (size_t i = 0; i < self.size() && i < 3; ++i) {
		char part[128];
		std::snprintf(part, sizeof(part), " %s %.1f%%", Name(self[i].key).c_str(), 100.0 * self[i].count / total);
		line += part;
	}
	std::printf("%s%s\n", line.c_str(), at_exit ? " (final)" : "");
	std::fflush(stdout);
}

inline void Run(uint32_t tid, uint64_t stack_low, uint64_t stack_high) {
	LocalPlatform::SetThreadName("Kyty.Perfil");
	const uint64_t handle = LocalPlatform::OpenThreadForSampling(tid);
	if (handle == 0) {
		std::printf("[perfil] no se pudo abrir el hilo de render %u: perfil apagado\n", tid);
		std::fflush(stdout);
		return;
	}
	LocalPlatform::PrepareSampling();
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
	if (timer == nullptr) timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
	std::printf("[perfil] KYTY_PERFIL=1: hilo de render %u cada %llu us -> %s (timer %s)\n", tid,
	            static_cast<unsigned long long>(g_period_us), g_path.c_str(), timer != nullptr ? "ok" : "sleep");
	std::fflush(stdout);
	uint64_t words[Frames];
	ULONGLONG next_write = GetTickCount64() + 30000;
	for (;;) {
		if (timer != nullptr) {
			LARGE_INTEGER due {};
			due.QuadPart = -static_cast<LONGLONG>(g_period_us * 10); // relative, 100 ns units
			if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, 100);
			else Sleep(1);
		} else {
			Sleep(1);
		}
		std::memset(words, 0, sizeof(words));
		const bool ok = LocalPlatform::SampleThread(handle, stack_low, stack_high, words, Frames); // (suspended inside)
		AcquireSRWLockExclusive(&g_lock);
		if (ok && words[0] != 0) {
			++g_t->samples;
			Add(g_t->self, words[0]);
			for (size_t k = 0; k < Frames; ++k) {
				const uint64_t w = words[k];
				if (w == 0) break;
				bool seen = false;
				for (size_t j = 0; j < k; ++j) seen = seen || words[j] == w;
				if (!seen) Add(g_t->incl, w);
			}
			AddStack(words);
		} else {
			++g_t->failed;
		}
		ReleaseSRWLockExclusive(&g_lock);
		if (GetTickCount64() >= next_write) {
			next_write = GetTickCount64() + 30000;
			Write(false);
		}
	}
}

inline void AtExit() {
	Write(true);
}
#endif

// The GPU thread, at its start (GuestGpu::ThreadRun): once.
inline void Start() {
	if (!On()) return;
#if defined(_WIN32)
	static bool started = false;
	if (started) return;
	started = true;
	if (const char* v = std::getenv("KYTY_PERFIL_US"); v != nullptr) {
		const long us = std::strtol(v, nullptr, 10);
		if (us >= 200 && us <= 20000) g_period_us = static_cast<uint64_t>(us);
	}
	g_t = new (std::nothrow) Tables {};
	if (g_t == nullptr) return;
	if (const char* log = std::getenv("KYTY_RUN_LOG"); log != nullptr && *log != '\0') {
		const std::string run   = log;
		const auto        slash = run.find_last_of("/\\");
		std::string       stamp = slash == std::string::npos ? run : run.substr(slash + 1);
		if (const auto dot = stamp.find(".out.log"); dot != std::string::npos) stamp.resize(dot);
		g_path = (slash == std::string::npos ? std::string() : run.substr(0, slash + 1)) + "perfil-" + stamp + ".txt";
	} else {
		const std::time_t t = std::time(nullptr);
		std::tm           tm {};
		(void)localtime_s(&tm, &t);
		char stamp[32];
		std::snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
		              tm.tm_hour, tm.tm_min, tm.tm_sec);
		g_path = std::string("logs/perfil-") + stamp + ".txt";
	}
	uint64_t low = 0, high = 0;
	if (!LocalPlatform::CurrentThreadStack(&low, &high)) low = high = 0;
	std::printf("[perfil] ON: hilo de render %u muestreado cada %llu us -> %s (cada 30 s y al salir)\n", LocalPlatform::ThreadId(),
	            static_cast<unsigned long long>(g_period_us), g_path.c_str());
	std::fflush(stdout);
	std::atexit(AtExit);
	std::thread(Run, LocalPlatform::ThreadId(), low, high).detach();
#endif
}

} // namespace Perfil
