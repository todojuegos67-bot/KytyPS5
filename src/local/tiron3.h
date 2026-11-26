#pragma once
// (lab1 tiron3) KYTY_TIRON3=1 (off by default; "1" or "2" turn it on; 2 = tiron4, below): what the render (GPU) thread does in the frames of
// ~100-130 ms with ~96-131 ms of its CPU and nothing in its counters: the PM4 walk under 30 ms ([tiron2] silent),
// queue_run < 10 ms, no uploads, images or pipelines (log 20261119-063917-906, 8 such frames in ~2 min, every 4-35 s).
//   - A sampler thread (Kyty.Tiron3) suspends the render thread every KYTY_TIRON3_US (default 1000) microseconds and
//     keeps the last 8192 samples (pc and up to 15 callers, LocalPlatform::SampleThread, as `prof` does).
//   - At each flip (render thread, one compare when on): a frame of KYTY_TIRON3_MS (default 60) ms or more hands its
//     flip-to-flip window to the sampler, which prints a [tiron3] line: the window's samples by their first three
//     emulator (exe) frames, with the module of the leaf (exe / ntdll = waiting in the kernel / other dll), top 8.
//   - Host code is printed as exe+offset: symbolize with the same build's link map
//     (_Build/windows/kyty_emulator_clang_lld_link.map). Guest code (eboot) is not in these frames' stacks' leaves.
// (tiron4) KYTY_TIRON3=2: as 1, and: every other thread (guest, compilers) sampled every KYTY_TIRON3_ALL_US (4000) us,
//   one [tiron3t] line per interesting thread per slow frame (top 3 stacks: leaf, first guest frame, exe frames); the
//   [tiron3] line ends with the frame's counters (guest file calls ms, RAM17 disk reads ms, faults, pipelines, upload_busy).
//   Cost: ~N threads x ~20 us every 4 ms on the sampler thread (one core ~20-40%), suspensions of microseconds each.
// Costs, when on: one suspension a millisecond of the render thread (~10-40 us), one line per slow frame.
#include "local-platform.h"
#include "live-counters.h"
#include "tiron4-counters.h" // (tiron4)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Loader {
std::string DescribeGuestAddressForDiagnostics(uint64_t vaddr);
} // namespace Loader

namespace Tiron3 {

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON3");
		return value != nullptr && (std::string_view(value) == "1" || std::string_view(value) == "2");
	}();
	return on;
}
// (tiron4) KYTY_TIRON3=2: also every other thread of the process (guest threads, pipeline compilers, ...) and the frame's
// counters (guest file calls, RAM17 reads back, faults, pipelines, upload_busy).
inline bool All() {
	static const bool all = [] {
		const char* value = std::getenv("KYTY_TIRON3");
		return value != nullptr && std::string_view(value) == "2";
	}();
	return all;
}

inline double EnvNumber(const char* name, double fallback, double low, double high) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	const double number = std::atof(value);
	return number < low ? low : number > high ? high : number;
}

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr size_t Words = 16, Ring = 8192;
struct Sample {
	int64_t  ns = 0;
	uint64_t words[Words] {};
};
inline Sample*               g_ring = nullptr;     // (allocated by the sampler, never freed)
inline std::atomic<uint64_t> g_written {0};        // samples written (the ring index is % Ring)
inline std::atomic<int64_t>  g_win_begin {0}, g_win_end {0}; // a slow frame's window for the sampler (0: none)
inline std::atomic<double>   g_win_ms {0.0};
inline std::atomic<uint64_t> g_slow_frames {0}, g_skipped {0};

// (tiron4) The counters' snapshots (sampler thread, each round) and the other threads' samples.
enum Cnt : uint32_t { CFileNs, CFileCalls, CFileBytes, COpenNs, COpens, CRam17Us, CRam17Reads, CFaults, CPipes, CPipeUs, CUploadBusyUs, CRestoreUs, CRestores, CImageInitUs, CProgramUs, CCount };
struct Snap {
	int64_t  ns = 0;
	uint64_t v[CCount] {};
};
constexpr size_t SnapRing = 4096, AllRing = 1u << 16;
inline Snap* g_snaps = nullptr;
inline uint64_t g_snaps_written = 0; // (sampler thread only)
struct AllSample {
	int64_t  ns  = 0;
	uint32_t tid = 0;
	uint64_t words[Words] {};
};
inline AllSample* g_all = nullptr;
inline uint64_t   g_all_written = 0; // (sampler thread only)
inline std::unordered_map<uint32_t, std::string> g_names; // tid -> thread name (sampler thread only)

inline uint64_t Counter(LiveCounters::Id id) {
	return LiveCounters::g_values[id].load(std::memory_order_relaxed) + LiveCounters::g_render_values[id].load(std::memory_order_relaxed);
}
inline void TakeSnap() {
	Snap& s = g_snaps[g_snaps_written % SnapRing];
	s.ns    = NowNs();
	s.v[CFileNs]       = Tiron4::g_file_ns.load(std::memory_order_relaxed);
	s.v[CFileCalls]    = Tiron4::g_file_calls.load(std::memory_order_relaxed);
	s.v[CFileBytes]    = Tiron4::g_file_bytes.load(std::memory_order_relaxed);
	s.v[COpenNs]       = Tiron4::g_open_ns.load(std::memory_order_relaxed);
	s.v[COpens]        = Tiron4::g_opens.load(std::memory_order_relaxed);
	s.v[CRam17Us]      = Tiron4::g_ram17_us.load(std::memory_order_relaxed);
	s.v[CRam17Reads]   = Tiron4::g_ram17_reads.load(std::memory_order_relaxed);
	s.v[CFaults]       = Counter(LiveCounters::WindowFault) + Counter(LiveCounters::WriteFault) + Counter(LiveCounters::ReadFault);
	s.v[CPipes]        = Counter(LiveCounters::GraphicsPipelines) + Counter(LiveCounters::ComputePipelines) + Counter(LiveCounters::AsyncPipelines);
	s.v[CPipeUs]       = Counter(LiveCounters::GraphicsPipelineUs) + Counter(LiveCounters::ComputePipelineUs);
	s.v[CUploadBusyUs] = Counter(LiveCounters::UploadBusyUs);
	s.v[CRestoreUs]    = Tiron4::g_restore_us.load(std::memory_order_relaxed); // (tiron5)
	s.v[CRestores]     = Tiron4::g_restores.load(std::memory_order_relaxed);
	s.v[CImageInitUs]  = Counter(LiveCounters::ImageInitUs);
	s.v[CProgramUs]    = Counter(LiveCounters::TranslateUs);
	++g_snaps_written;
}
// The counters' change over [begin, end]: the last snapshot at or before begin to the first at or after end.
inline bool SnapDelta(int64_t begin, int64_t end, uint64_t* out) {
	const uint64_t first = g_snaps_written > SnapRing ? g_snaps_written - SnapRing : 0;
	const Snap *   a = nullptr, *b = nullptr;
	for (uint64_t i = first; i < g_snaps_written; ++i) {
		const Snap& s = g_snaps[i % SnapRing];
		if (s.ns <= begin) a = &s;
		if (s.ns >= end && b == nullptr) b = &s;
	}
	if (a == nullptr || b == nullptr) return false;
	for (uint32_t c = 0; c < CCount; ++c) out[c] = b->v[c] - a->v[c];
	return true;
}

#if defined(_WIN32)
// The module base of an address (cached by 64 KiB granule), 0 if none.
inline uint64_t ModuleBase(uint64_t address, std::unordered_map<uint64_t, uint64_t>& cache, std::string* name) {
	const uint64_t granule = address >> 16u;
	if (name == nullptr) {
		if (auto it = cache.find(granule); it != cache.end()) return it->second;
	}
	uint64_t base = 0;
	char     path[260] {};
	if (!LocalPlatform::ModuleOf(reinterpret_cast<const void*>(address), &base, path, sizeof(path))) base = 0;
	cache[granule] = base;
	if (name != nullptr) {
		const char* slash = std::strrchr(path, '\\');
		*name             = slash != nullptr ? slash + 1 : path;
	}
	return base;
}

inline void Report(int64_t begin, int64_t end, double frame_ms, uint64_t exe_base,
                   std::unordered_map<uint64_t, uint64_t>& cache) {
	struct Key {
		uint64_t a = 0, b = 0, c = 0, leaf_base = 0;
		bool     operator==(const Key& o) const { return a == o.a && b == o.b && c == o.c && leaf_base == o.leaf_base; }
	};
	struct KeyHash {
		size_t operator()(const Key& k) const { return std::hash<uint64_t>()(k.a * 31 + k.b * 17 + k.c * 7 + k.leaf_base); }
	};
	std::unordered_map<Key, uint32_t, KeyHash> counts;
	uint32_t total = 0, in_exe_leaf = 0, kernel_leaf = 0;
	const uint64_t written = g_written.load(std::memory_order_acquire);
	const uint64_t first   = written > Ring ? written - Ring : 0;
	std::string    leaf_name;
	std::unordered_map<uint64_t, std::string> names; // leaf module base -> name
	for (uint64_t i = first; i < written; ++i) {
		const Sample& s = g_ring[i % Ring];
		if (s.ns < begin || s.ns > end) continue;
		++total;
		Key k;
		k.leaf_base = s.words[0] != 0 ? ModuleBase(s.words[0], cache, nullptr) : 0;
		if (k.leaf_base != 0 && names.find(k.leaf_base) == names.end()) {
			(void)ModuleBase(s.words[0], cache, &leaf_name);
			names[k.leaf_base] = leaf_name;
		}
		if (k.leaf_base == exe_base) ++in_exe_leaf;
		else if (auto it = names.find(k.leaf_base); it != names.end() && (it->second == "ntdll.dll" || it->second == "KERNELBASE.dll"))
			++kernel_leaf;
		uint64_t* out[3] = {&k.a, &k.b, &k.c};
		int       got    = 0;
		for (size_t w = 0; w < Words && got < 3; ++w) {
			const uint64_t x = s.words[w];
			if (x == 0) break;
			if (ModuleBase(x, cache, nullptr) != exe_base || exe_base == 0) continue;
			const uint64_t off = x - exe_base;
			if (got > 0 && *out[got - 1] == off) continue;
			*out[got++] = off;
		}
		++counts[k];
	}
	std::vector<std::pair<Key, uint32_t>> sorted(counts.begin(), counts.end());
	std::sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
	std::string line;
	char        part[256];
	std::snprintf(part, sizeof(part),
	              "[tiron3] frame %.1f ms: %u muestras (hoja en el exe %u, en ntdll/kernelbase = esperando %u, otra dll %u) | top",
	              frame_ms, total, in_exe_leaf, kernel_leaf, total - in_exe_leaf - kernel_leaf);
	line += part;
	for (size_t i = 0; i < sorted.size() && i < 8; ++i) {
		const auto& [k, n] = sorted[i];
		const auto  it     = names.find(k.leaf_base);
		std::snprintf(part, sizeof(part), " | %.0f%% exe+0x%llx <- exe+0x%llx <- exe+0x%llx [hoja %s]",
		              total != 0 ? 100.0 * n / total : 0.0, static_cast<unsigned long long>(k.a),
		              static_cast<unsigned long long>(k.b), static_cast<unsigned long long>(k.c),
		              k.leaf_base == 0 ? "?" : it != names.end() ? it->second.c_str() : "?");
		line += part;
	}
	if (All()) {
		uint64_t d[CCount] {};
		if (SnapDelta(begin, end, d)) {
			std::snprintf(part, sizeof(part),
			              " || contadores del frame: ficheros %.1f ms (%llu lecturas, %.1f MiB; open %.1f ms x%llu) | RAM17 de disco %.1f ms "
			              "(%llu) | fallos de pagina %llu | pipelines %llu (%.1f ms) | upload_busy %.1f ms",
			              d[CFileNs] / 1e6, static_cast<unsigned long long>(d[CFileCalls]), d[CFileBytes] / 1048576.0, d[COpenNs] / 1e6,
			              static_cast<unsigned long long>(d[COpens]), d[CRam17Us] / 1e3, static_cast<unsigned long long>(d[CRam17Reads]),
			              static_cast<unsigned long long>(d[CFaults]), static_cast<unsigned long long>(d[CPipes]), d[CPipeUs] / 1e3,
			              d[CUploadBusyUs] / 1e3);
			line += part;
			std::snprintf(part, sizeof(part), " | RAM17 restaurado %.1f ms (%llu) | image_init %.1f ms | traduccion %.1f ms",
			              d[CRestoreUs] / 1e3, static_cast<unsigned long long>(d[CRestores]), d[CImageInitUs] / 1e3,
			              d[CProgramUs] / 1e3); // (tiron5)
			line += part;
		} else {
			line += " || contadores del frame: sin instantaneas";
		}
	}
	line += "\n";
	std::fputs(line.c_str(), stdout);
	std::fflush(stdout);
}

// (tiron4) An address as guest module+offset (Loader), host dll+offset, or hex (cached).
inline const std::string& Sym(uint64_t address, std::unordered_map<uint64_t, std::string>& known,
                              std::unordered_map<uint64_t, uint64_t>& cache, uint64_t exe_base) {
	if (auto it = known.find(address); it != known.end()) return it->second;
	std::string text;
	std::string module;
	const uint64_t base = ModuleBase(address, cache, &module);
	char        part[320];
	if (base == exe_base && base != 0) {
		std::snprintf(part, sizeof(part), "exe+0x%llx", static_cast<unsigned long long>(address - base));
		text = part;
	} else if (base != 0) {
		std::snprintf(part, sizeof(part), "%s+0x%llx", module.c_str(), static_cast<unsigned long long>(address - base));
		text = part;
	} else {
		text = Loader::DescribeGuestAddressForDiagnostics(address);
		if (text.empty()) {
			std::snprintf(part, sizeof(part), "0x%llx", static_cast<unsigned long long>(address));
			text = part;
		}
	}
	return known.emplace(address, std::move(text)).first->second;
}

// (tiron4) Every other thread over the slow frame's window: per thread, its samples, how many with the leaf outside
// ntdll / kernelbase (running or ready), and its top 3 stacks as: leaf (module) | first guest frame | first 3 exe frames.
// A thread is printed if it ran in >= 5% of its samples, has a guest frame (a game thread: its waits are the point), or
// is one of the emulator's compilers / loaders (Kyty.Warmup / Follow / Prefetch / TableCache). At most 48 lines.
inline void ReportAll(int64_t begin, int64_t end, double frame_ms, uint64_t frame_no, uint64_t exe_base,
                      std::unordered_map<uint64_t, uint64_t>& cache, std::unordered_map<uint64_t, std::string>& known) {
	struct Per {
		uint32_t samples = 0, running = 0, guest = 0;
		std::unordered_map<std::string, uint32_t> stacks;
	};
	std::unordered_map<uint32_t, Per> per;
	const uint64_t first = g_all_written > AllRing ? g_all_written - AllRing : 0;
	std::string    leaf_name;
	for (uint64_t i = first; i < g_all_written; ++i) {
		const AllSample& s = g_all[i % AllRing];
		if (s.ns < begin || s.ns > end) continue;
		Per& p = per[s.tid];
		++p.samples;
		const uint64_t leaf_base = s.words[0] != 0 ? ModuleBase(s.words[0], cache, &leaf_name) : 0;
		const bool     waiting   = leaf_base != 0 && (leaf_name == "ntdll.dll" || leaf_name == "KERNELBASE.dll" || leaf_name == "win32u.dll");
		if (!waiting) ++p.running;
		std::string key = "hoja ";
		key += leaf_base == 0 ? (s.words[0] != 0 ? "guest" : "?") : leaf_base == exe_base ? "exe" : leaf_name;
		if (leaf_base == exe_base || leaf_base == 0) key += " " + Sym(s.words[0], known, cache, exe_base);
		std::string guest, exe;
		int         exe_n = 0;
		uint64_t    last  = 0;
		for (size_t w = 1; w < Words; ++w) {
			const uint64_t x = s.words[w];
			if (x == 0) break;
			const uint64_t b = ModuleBase(x, cache, nullptr);
			if (b == 0 && guest.empty()) guest = Sym(x, known, cache, exe_base);
			else if (b == exe_base && exe_n < 3 && x != last) {
				exe += (exe_n == 0 ? "" : " <- ") + Sym(x, known, cache, exe_base);
				++exe_n;
				last = x;
			}
		}
		if (!guest.empty()) ++p.guest;
		key += " | juego " + (guest.empty() ? std::string("-") : guest) + " | " + (exe.empty() ? std::string("-") : exe);
		++p.stacks[key];
	}
	std::vector<std::pair<uint32_t, const Per*>> order;
	for (const auto& [tid, p]: per) order.emplace_back(tid, &p);
	std::sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.second->running > y.second->running; });
	int printed = 0;
	for (const auto& [tid, p]: order) {
		const auto        name_it = g_names.find(tid);
		const std::string name    = name_it != g_names.end() && !name_it->second.empty() ? name_it->second : "-";
		const bool        tool    = name.rfind("Kyty.Warmup", 0) == 0 || name.rfind("Kyty.Follow", 0) == 0 ||
		                    name.rfind("Kyty.Prefetch", 0) == 0 || name.rfind("Kyty.TableCache", 0) == 0 ||
		                    // (tiron9) the threads the GPU thread waits on, printed even while blocked (their wait IS the answer)
		                    name.rfind("Kyty.Upload", 0) == 0 || name.rfind("Kyty.Record", 0) == 0 ||
		                    name.rfind("Kyty.Present", 0) == 0 || name.rfind("Kyty.Offload", 0) == 0 ||
		                    name.rfind("Kyty.TexRelease", 0) == 0 || name.rfind("Kyty.FreedRelease", 0) == 0;
		if (!(p->running * 20 >= p->samples && p->running > 0) && p->guest == 0 && !tool) continue;
		if (++printed > 48) break;
		std::vector<std::pair<std::string, uint32_t>> top(p->stacks.begin(), p->stacks.end());
		std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
		std::string line;
		char        part[200];
		std::snprintf(part, sizeof(part), "[tiron3t] frame#%llu %.1f ms | hilo %u %s: %u muestras, corriendo %u (%.0f%%)",
		              static_cast<unsigned long long>(frame_no), frame_ms, tid, name.c_str(), p->samples, p->running,
		              p->samples != 0 ? 100.0 * p->running / p->samples : 0.0);
		line += part;
		for (size_t i = 0; i < top.size() && i < 3; ++i) {
			std::snprintf(part, sizeof(part), " || %.0f%% ", 100.0 * top[i].second / p->samples);
			line += part;
			line += top[i].first;
		}
		line += "\n";
		std::fputs(line.c_str(), stdout);
	}
	std::fflush(stdout);
}

inline void Run(uint32_t tid, uint64_t stack_low, uint64_t stack_high) {
	LocalPlatform::SetThreadName("Kyty.Tiron3");
	const uint64_t handle = LocalPlatform::OpenThreadForSampling(tid);
	if (handle == 0) {
		std::printf("[tiron3] no se pudo abrir el hilo de render %u: sonda apagada\n", tid);
		std::fflush(stdout);
		return;
	}
	g_ring = new (std::nothrow) Sample[Ring];
	if (g_ring == nullptr) return;
	if (All()) {
		g_snaps = new (std::nothrow) Snap[SnapRing];
		g_all   = new (std::nothrow) AllSample[AllRing];
		if (g_snaps == nullptr || g_all == nullptr) return;
	}
	LocalPlatform::PrepareSampling();
	uint64_t exe_base = 0;
	char     path[260] {};
	(void)LocalPlatform::ModuleOf(reinterpret_cast<const void*>(&Run), &exe_base, path, sizeof(path));
	const auto period = std::chrono::microseconds(static_cast<int64_t>(EnvNumber("KYTY_TIRON3_US", 1000.0, 200.0, 20000.0)));
	std::printf("[tiron3] KYTY_TIRON3=1: muestreo del hilo de render %u cada %lld us, frames >= %.0f ms; exe base 0x%llx (%s). "
	            "Simbolos: el .map de esta misma compilacion\n",
	            tid, static_cast<long long>(period.count()), EnvNumber("KYTY_TIRON3_MS", 60.0, 20.0, 1000.0),
	            static_cast<unsigned long long>(exe_base), path);
	std::fflush(stdout);
	std::unordered_map<uint64_t, uint64_t> cache;
	// (tiron4) The other threads: refreshed every second; sampled every KYTY_TIRON3_ALL_US (default 4000) microseconds.
	std::unordered_map<uint32_t, uint64_t>   handles;
	std::unordered_map<uint64_t, std::string> known;
	const int64_t all_period_ns = static_cast<int64_t>(EnvNumber("KYTY_TIRON3_ALL_US", 4000.0, 1000.0, 50000.0) * 1000.0);
	int64_t       next_all = 0, next_list = 0;
	uint64_t      frame_no = 0;
	const uint32_t self = LocalPlatform::ThreadId();
	if (All()) {
		std::printf("[tiron3] KYTY_TIRON3=2: ademas todos los hilos cada %lld us y los contadores del frame; lineas [tiron3t] por hilo\n",
		            static_cast<long long>(all_period_ns / 1000));
		std::fflush(stdout);
	}
	auto next = std::chrono::steady_clock::now();
	for (;;) {
		next += period;
		std::this_thread::sleep_until(next);
		const uint64_t i = g_written.load(std::memory_order_relaxed);
		Sample&        s = g_ring[i % Ring];
		std::memset(s.words, 0, sizeof(s.words));
		if (LocalPlatform::SampleThread(handle, stack_low, stack_high, s.words, Words)) {
			s.ns = NowNs();
			g_written.store(i + 1, std::memory_order_release);
		}
		if (All()) {
			const int64_t now = NowNs();
			if (now >= next_list) {
				next_list = now + 1000000000;
				std::unordered_map<uint32_t, uint64_t> fresh;
				for (auto& [t, name]: LocalPlatform::ProcessThreads()) {
					if (t == self || t == tid || name.rfind("Kyty.Blackbox", 0) == 0) continue;
					auto it = handles.find(t);
					fresh[t] = it != handles.end() ? it->second : LocalPlatform::OpenThreadForSampling(t);
					if (it != handles.end()) handles.erase(it);
					g_names[t] = std::move(name);
				}
				for (auto& [t, h]: handles)
					if (h != 0) LocalPlatform::CloseThreadForSampling(h);
				handles.swap(fresh);
			}
			TakeSnap();
			if (now >= next_all) {
				next_all = now + all_period_ns;
				for (const auto& [t, h]: handles) {
					if (h == 0) continue;
					AllSample& a = g_all[g_all_written % AllRing];
					std::memset(a.words, 0, sizeof(a.words));
					if (LocalPlatform::SampleThread(h, 0, 0, a.words, Words)) {
						a.ns  = NowNs();
						a.tid = t;
						++g_all_written;
					}
				}
			}
		}
		const int64_t end = g_win_end.load(std::memory_order_acquire);
		if (end != 0 && NowNs() > end + (All() ? 5000000 : 0)) { // (tiron4: 5 ms more for the counters' end snapshot)
			const int64_t begin = g_win_begin.load(std::memory_order_relaxed);
			const double  ms    = g_win_ms.load(std::memory_order_relaxed);
			Report(begin, end, ms, exe_base, cache);
			if (All()) ReportAll(begin, end, ms, ++frame_no, exe_base, cache, known);
			g_win_end.store(0, std::memory_order_release);
		}
		if (std::chrono::steady_clock::now() > next + std::chrono::milliseconds(50)) next = std::chrono::steady_clock::now();
	}
}
#endif

// The render thread, at each flip (LiveControl::Flip).
inline void OnFlip(int64_t flip_ns, uint32_t render_tid, uint64_t stack_low, uint64_t stack_high) {
	if (!On()) return;
#if defined(_WIN32)
	static bool    started = false;
	static int64_t last    = 0;
	if (!started && render_tid != 0) {
		started = true;
		std::thread(Run, render_tid, stack_low, stack_high).detach();
	}
	static const double limit_ms = EnvNumber("KYTY_TIRON3_MS", 60.0, 20.0, 1000.0);
	if (last != 0) {
		const double ms = static_cast<double>(flip_ns - last) / 1e6;
		if (ms >= limit_ms) {
			g_slow_frames.fetch_add(1, std::memory_order_relaxed);
			if (g_win_end.load(std::memory_order_acquire) == 0) {
				g_win_begin.store(last, std::memory_order_relaxed);
				g_win_ms.store(ms, std::memory_order_relaxed);
				g_win_end.store(flip_ns, std::memory_order_release);
			} else {
				g_skipped.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	last = flip_ns;
#else
	(void)flip_ns, (void)render_tid, (void)stack_low, (void)stack_high;
#endif
}

} // namespace Tiron3
