// (f0) KYTY_F0=1: see f0-measure.h. Measurement only.

#include "local/f0-measure.h"

#include "graphics/host_gpu/graphicContext.h"
#include "loader/demonsSoulsMemMap.h"
#include "local/live-trace-gpu.h"
#include "local/local-platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#include <intrin.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef small
#undef small
#endif
#ifdef far
#undef far
#endif
#ifdef near
#undef near
#endif
#else
#include <ctime>
#include <x86intrin.h>
#endif

namespace F0Measure {
namespace {

// ---------------------------------------------------------------------------------------------------------------------
// Constants
constexpr uint32_t QueryCount  = 65536; // timestamp queries (ring)
constexpr uint32_t RecordCount = 8192;  // command buffers in flight (ring)
constexpr uint32_t MaxMarks    = 48;    // stamps per command buffer (begin, slice switches, end)
constexpr uint32_t OpenPerThread = 8;
constexpr uint32_t FrameSlots  = 64;
constexpr uint32_t MaxLabels   = 32;
constexpr uint32_t MaxProbes   = 64;
constexpr uint32_t Retired     = 512;
[[maybe_unused]] constexpr uint64_t Page = 0x1000;
constexpr uint64_t GranuleBits = 16; // label table: 64 KiB granules
[[maybe_unused]] constexpr uint64_t GameCodeBegin = 0x900000000ull; // eboot 01.007.000 image
[[maybe_unused]] constexpr uint64_t GameCodeEnd   = 0x904000000ull;

// GPU sides
enum Side : uint8_t { SideDcb = 0, SideAcb, SideHost, SideUpload, SideOther, Sides };

// Read-sampler outcomes
enum Probe : uint32_t {
	PrArmedRo, PrArmedRw, PrReadGame, PrReadLib, PrReadOther, PrWriteGame, PrWriteOther, PrUntouched, PrLost, PrEnsured,
	PrAgain, PrLifeMs, PrCounters
};

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t Tsc() {
	return __rdtsc();
}

// ---------------------------------------------------------------------------------------------------------------------
// State (heap, never freed: nothing of it is destroyed at exit while the f0 thread or a hook may still run).
struct Record {
	std::atomic<uint32_t> state {0}; // 0 free, 1 open, 2 submitted, 3 dead (its queries are not reused)
	void*                 cmd     = nullptr;
	int64_t               open_ns = 0;
	uint64_t              frame   = 0;
	uint8_t               kind    = 0; // 0 main, 1 prologue
	uint8_t               lane    = 0;
	uint8_t               n       = 0;
	uint8_t               last    = SideHost;
	uint32_t              q[MaxMarks] {};
	uint8_t               side[MaxMarks] {};
};

struct FrameGpu {
	double busy_ms = 0, span_ms = 0;
	double side_ms[Sides] {};
};
struct FrameAcc {
	uint64_t                                id = UINT64_MAX;
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	double                                  side_ns[Sides] {};
};
struct FlipSample {
	double frame_ms = 0, busy_ms = 0, gpu_wait_ms = 0, cpu_ms = 0;
	int    ab = 0;
};
struct GameFrame {
	double build_ms = 0, wait_ms = 0, cpu_ms = 0, submits = 0;
};
struct ProbeSlot {
	std::atomic<uint32_t> state {0}; // 0 free, 1 arming, 2 armed, 3 retiring
	std::atomic<uint64_t> page {0};
	std::atomic<uint32_t> hit {0}; // bit 0 read, 1 write, 2 game code, 3 game via a library, 4 other
	uint32_t              label   = 0;
	uint32_t              base    = 0; // the protection kept (low byte)
	int64_t               armed_ns = 0;
};

struct Window { // one 5 s window's row (for the summary and the CSV)
	double t_s = 0, frames = 0, fps = 0, low1 = 0, ft50 = 0, ft99 = 0, hitches = 0, ft50_on = 0, ft50_off = 0;
	double gpu50 = 0, gpu90 = 0, gpu_side50[Sides] {}, gpu_frames = 0;
	double thread50 = 0, gpu_wait = 0, cpu_thread = 0;
	double game_build50 = 0, game_build90 = 0, game_wait50 = 0, game_cpu50 = 0, game_frames = 0;
	double cpu_process = 0, ram_ws = 0, ram_commit = 0, vram_used = 0, vram_budget = 0;
	double up_mb = 0, up_n = 0, f2_up_mb = 0, ring_mb = 0, ring_n = 0, lc_n = 0, lc_mb = 0, lc_ms = 0, f2_lc_ms = 0;
	double cost_gpu = 0, cost_game = 0, cost_thread_pct = 0, gate_ms_s = 0;
	int    verdict = 0; // 0 vsync/mixed, 1 GPU, 2 CPU GPU thread, 3 CPU game
};
constexpr const char* VerdictNames[4] = {"tope/mixto", "GPU", "CPU-hilo-GPU", "CPU-juego"};

struct State {
	// ---- GPU
	std::once_flag                     gpu_once;
	std::atomic<bool>                  gpu_ok {false};
	std::atomic<bool>                  graphics_ready {false}; // (graphics set)
	Libs::Graphics::GraphicContext*    graphics = nullptr;
	vk::QueryPool                      pool {};
	double                             period_ns = 1.0;
	uint32_t                           valid_bits = 64;
	uint64_t                           mask = ~uint64_t {0};
	std::atomic<uint8_t>               qstate[QueryCount] {};
	std::atomic<uint32_t>              qnext {0};
	Record                             records[RecordCount];
	std::atomic<uint32_t>              rnext {0};
	std::atomic<uint64_t>              flips {0};
	FrameAcc                           frames[FrameSlots];
	std::mutex                         gpu_mutex; // window_gpu
	std::vector<FrameGpu>              window_gpu;
	std::atomic<uint64_t>              dropped_marks {0}, dropped_records {0}, late_records {0}, dead_records {0};
	uint64_t                           unwrap_last = 0, unwrap_high = 0;
	// ---- device facts (f0 thread, once)
	std::atomic<bool>                  device_done {false};
	double                             rebar_mb = 0, bar_write_mbs = 0, bar_read_mbs = 0, bar_line_ns = 0, host_line_ns = 0;
	bool                               placed = false, host_import = false, budget_ext = false;
	// ---- labels
	std::mutex                         label_mutex; // names, ranges (f0 thread writes; reports read)
	char                               names[MaxLabels][20] {};
	std::atomic<uint32_t>              label_count {1};
	std::vector<std::pair<uint64_t, uint64_t>> ranges[MaxLabels];
	double                             label_mb[MaxLabels] {};
	struct Table {
		uint64_t             lo = 0, hi = 0;
		std::vector<uint8_t> gran;
	};
	std::atomic<const Table*>          table {nullptr};
	std::vector<const Table*>          retired_tables;
	std::atomic<uint64_t>              unknown[256] {};
	std::atomic<uint32_t>              unknown_next {0};
	// ---- uploads, ring copies, linear copies (by label)
	std::atomic<uint64_t>              up_bytes[MaxLabels] {}, up_n[MaxLabels] {};
	std::atomic<uint64_t>              ring_bytes[MaxLabels] {}, ring_n[MaxLabels] {};
	std::atomic<uint64_t>              lc_bytes[MaxLabels] {}, lc_n[MaxLabels] {}, lc_cycles[MaxLabels] {};
	// ---- read sampler
	void (*gate_lock)()   = nullptr;
	void (*gate_unlock)() = nullptr;
	ProbeSlot                          probes[MaxProbes];
	std::atomic<uint32_t>              armed {0};
	std::atomic<uint64_t>              retired[Retired] {};
	std::atomic<int64_t>               retired_ns[Retired] {};
	std::atomic<uint32_t>              retired_next {0};
	std::atomic<uint64_t>              probe[MaxLabels][PrCounters] {};
	std::atomic<uint64_t>              gate_ns {0}, guard_hits {0};
	std::atomic<uint32_t>              targets[8] {};
	std::atomic<uint32_t>              target_count {0};
	// ---- game side
	std::atomic<int64_t>               game_last_exit {0}, game_enter {0};
	std::atomic<uint64_t>              game_cycles_exit {0};
	std::atomic<uint64_t>              game_submits {0}, game_dwords {0}, game_acb_submits {0};
	std::mutex                         game_mutex;
	std::vector<GameFrame>             window_game;
	// ---- costs
	std::atomic<uint64_t>              cost_gpu_cycles {0}, cost_game_cycles {0}, cost_handler_cycles {0};
	uint64_t                           thread_handle = 0;
	std::atomic<double>                tsc_per_ns {3.0};
	// ---- render-thread report state
	std::vector<FlipSample>            flips_window;
	int64_t                            last_flip_ns = 0, last_idle = 0, last_wait = 0, window_start = 0, start_ns = 0;
	uint64_t                           last_cycles = 0;
	double                             last_process_s = 0;
	std::array<uint64_t, MaxLabels>    last_up {}, last_up_n {}, last_ring {}, last_ring_n {}, last_lc {}, last_lc_n {},
	    last_lc_cycles {};
	std::array<std::array<uint64_t, PrCounters>, MaxLabels> last_probe {};
	uint64_t                           last_cost_gpu = 0, last_cost_game = 0, last_gate = 0, last_handler = 0;
	double                             last_thread_s = 0;
	std::mutex                         summary_mutex;
	std::vector<Window>                windows;
	FILE*                              csv = nullptr;
	int64_t                            last_summary_ns = 0;
};
State* g_s = nullptr; // (made by Enabled when on)

// ---------------------------------------------------------------------------------------------------------------------
// Percentiles
double Percentile(std::vector<double> v, double p) {
	if (v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	const double at = p * static_cast<double>(v.size() - 1);
	const auto   i  = static_cast<size_t>(at);
	const double f  = at - static_cast<double>(i);
	return i + 1 < v.size() ? v[i] * (1.0 - f) + v[i + 1] * f : v[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// Labels
uint32_t LabelOf(uint64_t vaddr) {
	const auto* t = g_s->table.load(std::memory_order_acquire);
	if (t == nullptr || vaddr < t->lo || vaddr >= t->hi) return 0;
	return t->gran[(vaddr - t->lo) >> GranuleBits];
}

void NoteUnknown(uint64_t vaddr) {
	const uint32_t i = g_s->unknown_next.fetch_add(1, std::memory_order_relaxed);
	if ((i & 15u) == 0) g_s->unknown[(i >> 4u) & 255u].store(vaddr, std::memory_order_relaxed);
}

uint32_t LabelIndexLocked(const char* name) {
	const uint32_t n = g_s->label_count.load(std::memory_order_relaxed);
	for (uint32_t i = 1; i < n; ++i)
		if (std::strcmp(g_s->names[i], name) == 0) return i;
	if (n >= MaxLabels) return 0;
	std::snprintf(g_s->names[n], sizeof(g_s->names[n]), "%s", name);
	g_s->label_count.store(n + 1, std::memory_order_release);
	return n;
}

// f0 thread: unknown addresses named (LabelAt: a scan, never on the hot path), then every label's ranges read again and
// the 64 KiB table painted (larger ranges first, so the smallest range holding an address names it).
void RefreshLabels() {
	using namespace Loader::DemonsSoulsMemMap;
	{
		std::lock_guard lock(g_s->label_mutex);
		for (auto& u: g_s->unknown) {
			const uint64_t a = u.exchange(0, std::memory_order_relaxed);
			if (a == 0) continue;
			char name[20] {};
			if (LabelAt(a, name, sizeof(name)) && name[0] != 0) (void)LabelIndexLocked(name);
		}
	}
	const uint32_t n = g_s->label_count.load(std::memory_order_acquire);
	std::vector<uint64_t> pairs(2 * 16384);
	struct Painted {
		uint64_t b, e;
		uint32_t l;
	};
	std::vector<Painted> all;
	std::vector<std::pair<uint64_t, uint64_t>> per[MaxLabels];
	for (uint32_t l = 1; l < n; ++l) {
		const char*  name = g_s->names[l];
		const size_t got  = RangesOf(&name, 1, pairs.data(), pairs.size() / 2);
		if (got == SIZE_MAX) return; // (no snapshot now: next time)
		for (size_t i = 0; i < got; ++i) {
			if (pairs[2 * i + 1] <= pairs[2 * i]) continue;
			all.push_back({pairs[2 * i], pairs[2 * i + 1], l});
			per[l].emplace_back(pairs[2 * i], pairs[2 * i + 1]);
		}
	}
	if (all.empty()) return;
	uint64_t lo = UINT64_MAX, hi = 0;
	for (const auto& p: all) {
		lo = std::min(lo, p.b);
		hi = std::max(hi, p.e);
	}
	lo &= ~((uint64_t {1} << GranuleBits) - 1u);
	constexpr uint64_t MaxSpan = uint64_t {64} << 30u;
	if (hi - lo > MaxSpan) hi = lo + MaxSpan;
	auto* t = new State::Table;
	t->lo   = lo;
	t->hi   = hi;
	t->gran.assign(static_cast<size_t>((hi - lo + (uint64_t {1} << GranuleBits) - 1u) >> GranuleBits), 0);
	std::sort(all.begin(), all.end(), [](const Painted& a, const Painted& b) { return a.e - a.b > b.e - b.b; });
	for (const auto& p: all) {
		const uint64_t b = std::max(p.b, lo), e = std::min(p.e, hi);
		for (uint64_t g = (b - lo) >> GranuleBits; g < ((e - lo + (uint64_t {1} << GranuleBits) - 1u) >> GranuleBits); ++g)
			t->gran[g] = static_cast<uint8_t>(p.l);
	}
	{
		std::lock_guard lock(g_s->label_mutex);
		for (uint32_t l = 1; l < n; ++l) {
			double mb = 0;
			for (const auto& r: per[l]) mb += static_cast<double>(r.second - r.first) / 1048576.0;
			g_s->ranges[l]   = std::move(per[l]);
			g_s->label_mb[l] = mb;
		}
	}
	g_s->table.store(t, std::memory_order_release);
	g_s->retired_tables.push_back(t);
	// (A reader holds a table for one lookup: kept for 8 refreshes, 16 s, before it is freed.)
	if (g_s->retired_tables.size() > 8) {
		delete g_s->retired_tables.front();
		g_s->retired_tables.erase(g_s->retired_tables.begin());
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU timestamps
thread_local Record* t_open[OpenPerThread] {};
thread_local Record* t_main = nullptr;
thread_local uint8_t t_side = SideHost;

void InitGpuOnce(Libs::Graphics::GraphicContext* graphics);
void InitGpu(Libs::Graphics::GraphicContext* graphics) {
	InitGpuOnce(graphics);
	g_s->graphics_ready.store(true, std::memory_order_release);
}
void InitGpuOnce(Libs::Graphics::GraphicContext* graphics) {
	auto& s     = *g_s;
	s.graphics  = graphics;
	const auto& props = graphics->GetPhysicalDeviceProperties();
	s.period_ns = static_cast<double>(props.limits.timestampPeriod);
	uint32_t families = 0;
	graphics->physical_device.getQueueFamilyProperties(&families, nullptr);
	std::vector<vk::QueueFamilyProperties> family(families);
	graphics->physical_device.getQueueFamilyProperties(&families, family.data());
	s.valid_bits = graphics->queue_family < families ? family[graphics->queue_family].timestampValidBits : 0;
	if (s.valid_bits == 0 || s.period_ns <= 0.0) {
		std::printf("[f0] GPU: la cola no admite marcas de tiempo (timestampValidBits=%u): sin tiempo de GPU\n", s.valid_bits);
		return;
	}
	s.mask = s.valid_bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << s.valid_bits) - 1u;
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = QueryCount;
	if (graphics->device.createQueryPool(&info, nullptr, &s.pool) != vk::Result::eSuccess) {
		std::printf("[f0] GPU: no se pudo crear el pool de marcas: sin tiempo de GPU\n");
		return;
	}
	graphics->device.resetQueryPool(s.pool, 0, QueryCount);
	s.gpu_ok.store(true, std::memory_order_release);
	std::printf("[f0] GPU: marcas de tiempo listas (periodo %.3f ns, %u bits validos, %u consultas)\n", s.period_ns, s.valid_bits,
	            QueryCount);
}

bool AllocQuery(uint32_t* q) {
	auto&          s = *g_s;
	const uint32_t i = s.qnext.fetch_add(1, std::memory_order_relaxed) % QueryCount;
	uint8_t        e = 0;
	if (!s.qstate[i].compare_exchange_strong(e, 1, std::memory_order_acq_rel)) {
		s.dropped_marks.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	*q = i;
	return true;
}

void Stamp(void* cmd, uint32_t q, bool top) {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdWriteTimestamp(static_cast<VkCommandBuffer>(cmd),
	                                                  top ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                                                  static_cast<VkQueryPool>(g_s->pool), q);
}

void Mark(Record* r, uint8_t side) {
	if (r == nullptr || r->n + 1u >= MaxMarks || r->last == side) return;
	uint32_t q = 0;
	if (!AllocQuery(&q)) return;
	Stamp(r->cmd, q, false);
	r->q[r->n]    = q;
	r->side[r->n] = side;
	r->n++;
	r->last = side;
}

uint64_t Unwrap(uint64_t raw) {
	auto& s = *g_s;
	raw &= s.mask;
	if (s.valid_bits >= 64) return raw;
	const uint64_t range = s.mask + 1u;
	uint64_t       v     = s.unwrap_high + raw;
	if (v + range / 2 < s.unwrap_last) v += range, s.unwrap_high += range;
	else if (v > s.unwrap_last + range / 2 && s.unwrap_high >= range) v -= range;
	s.unwrap_last = std::max(s.unwrap_last, v);
	return v;
}

void FinishFrame(FrameAcc& acc) {
	if (acc.id == UINT64_MAX) return;
	FrameGpu out;
	auto&    v = acc.spans;
	std::sort(v.begin(), v.end());
	uint64_t busy = 0, cb = 0, ce = 0;
	bool     open = false;
	for (const auto& [b, e]: v) {
		if (!open || b > ce) {
			if (open) busy += ce - cb;
			cb = b, ce = e, open = true;
		} else {
			ce = std::max(ce, e);
		}
	}
	if (open) busy += ce - cb;
	out.busy_ms = static_cast<double>(busy) * g_s->period_ns / 1e6;
	out.span_ms = v.empty() ? 0.0 : static_cast<double>(ce - v.front().first) * g_s->period_ns / 1e6;
	for (uint32_t i = 0; i < Sides; ++i) out.side_ms[i] = acc.side_ns[i] / 1e6;
	if (!v.empty()) {
		std::lock_guard lock(g_s->gpu_mutex);
		if (g_s->window_gpu.size() < 4096) g_s->window_gpu.push_back(out);
	}
	acc.id = UINT64_MAX;
	acc.spans.clear();
	for (double& x: acc.side_ns) x = 0;
}

void FreeRecord(Record& r) {
	for (uint32_t i = 0; i < r.n; ++i) {
		g_s->graphics->device.resetQueryPool(g_s->pool, r.q[i], 1);
		g_s->qstate[r.q[i]].store(0, std::memory_order_release);
	}
	r.n = 0;
	r.state.store(0, std::memory_order_release);
}

// f0 thread: submitted records whose stamps are all available, into their frame.
void Resolve() {
	auto& s = *g_s;
	if (!s.gpu_ok.load(std::memory_order_acquire)) return;
	const uint64_t flips = s.flips.load(std::memory_order_acquire);
	const int64_t  now   = NowNs();
	for (auto& r: s.records) {
		const uint32_t st = r.state.load(std::memory_order_acquire);
		if (st == 3) {
			r.state.store(0, std::memory_order_release); // (its queries stay taken: a buffer that may still run)
			continue;
		}
		if (st != 2) continue;
		uint64_t values[MaxMarks] {};
		bool     ready = true;
		for (uint32_t i = 0; i < r.n && ready; ++i) {
			uint64_t pair[2] {};
			const auto result = s.graphics->device.getQueryPoolResults(
			    s.pool, r.q[i], 1, sizeof(pair), pair, sizeof(pair), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			if ((result != vk::Result::eSuccess && result != vk::Result::eNotReady) || pair[1] == 0) ready = false;
			values[i] = pair[0];
		}
		if (!ready) {
			if (now - r.open_ns > 10'000'000'000ll) { // (never available: dropped, its queries kept)
				r.state.store(0, std::memory_order_release);
				s.dead_records.fetch_add(1, std::memory_order_relaxed);
			}
			continue;
		}
		if (r.n >= 2) {
			for (uint32_t i = 0; i < r.n; ++i) values[i] = Unwrap(values[i]);
			if (r.frame + 6 <= flips) { // (its frame already closed)
				s.late_records.fetch_add(1, std::memory_order_relaxed);
			} else {
				auto& acc = s.frames[r.frame % FrameSlots];
				if (acc.id != r.frame) {
					FinishFrame(acc);
					acc.id = r.frame;
				}
				const uint64_t b = values[0], e = std::max(values[r.n - 1], values[0]);
				acc.spans.emplace_back(b, e);
				for (uint32_t i = 0; i + 1 < r.n; ++i) {
					const uint64_t from = values[i], to = std::max(values[i + 1], from);
					const uint8_t  side = r.side[i] < Sides ? r.side[i] : SideOther;
					acc.side_ns[side] += static_cast<double>(to - from) * s.period_ns;
				}
			}
		}
		FreeRecord(r);
	}
	// Frames 6 flips old: complete (their records resolved within a few frames).
	for (auto& acc: s.frames)
		if (acc.id != UINT64_MAX && acc.id + 6 <= flips) FinishFrame(acc);
}

// ---------------------------------------------------------------------------------------------------------------------
// Device facts: Resizable BAR, placed mapping, host import, CPU access cost of BAR memory (f0 thread, once).
void DeviceFacts() {
	auto& s = *g_s;
	if (!s.graphics_ready.load(std::memory_order_acquire) || s.graphics == nullptr) return;
	auto*       g     = s.graphics;
	const auto& mem   = g->GetPhysicalDeviceMemoryProperties();
	uint32_t    type  = UINT32_MAX;
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
		const auto f = mem.memoryTypes[i].propertyFlags;
		if ((f & vk::MemoryPropertyFlagBits::eDeviceLocal) && (f & vk::MemoryPropertyFlagBits::eHostVisible)) {
			const double mb = static_cast<double>(mem.memoryHeaps[mem.memoryTypes[i].heapIndex].size) / 1048576.0;
			if (mb > s.rebar_mb || (mb == s.rebar_mb && (f & vk::MemoryPropertyFlagBits::eHostCoherent))) {
				s.rebar_mb = mb;
				type       = i;
			}
		}
	}
	uint32_t count = 0;
	if (g->physical_device.enumerateDeviceExtensionProperties(nullptr, &count, nullptr) == vk::Result::eSuccess) {
		std::vector<vk::ExtensionProperties> ext(count);
		if (g->physical_device.enumerateDeviceExtensionProperties(nullptr, &count, ext.data()) == vk::Result::eSuccess)
			for (const auto& e: ext) {
				if (std::strcmp(e.extensionName, "VK_EXT_map_memory_placed") == 0) s.placed = true;
				if (std::strcmp(e.extensionName, "VK_EXT_external_memory_host") == 0) s.host_import = true;
			}
	}
	s.budget_ext = g->memory_budget_ext_enabled;
	// The CPU's cost on BAR memory: 8 MiB written and read with memcpy, and 4096 dependent 8-byte reads (one per cache
	// line, random order) against the same on host memory.
	constexpr uint64_t Bytes = 8u << 20u;
	std::vector<uint8_t> host(Bytes, 1), back(Bytes);
	auto lines = [&](const volatile uint8_t* base) {
		constexpr uint32_t N = 4096;
		uint64_t           x = 0x9e3779b97f4a7c15ull, sum = 0;
		const auto         t0 = std::chrono::steady_clock::now();
		for (uint32_t i = 0; i < N; ++i) {
			x ^= x << 13u, x ^= x >> 7u, x ^= x << 17u;
			const uint64_t at = ((x + sum) % (Bytes / 64)) * 64;
			sum += *reinterpret_cast<const volatile uint64_t*>(base + at) & 1u;
		}
		return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / N;
	};
	s.host_line_ns = lines(host.data());
	if (type != UINT32_MAX) {
		vk::MemoryAllocateInfo info {};
		info.allocationSize  = Bytes;
		info.memoryTypeIndex = type;
		vk::DeviceMemory memory {};
		void*            mapped = nullptr;
		if (g->device.allocateMemory(&info, nullptr, &memory) == vk::Result::eSuccess) {
			if (g->device.mapMemory(memory, 0, Bytes, {}, &mapped) == vk::Result::eSuccess && mapped != nullptr) {
				auto t0 = std::chrono::steady_clock::now();
				std::memcpy(mapped, host.data(), Bytes);
				const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				t0             = std::chrono::steady_clock::now();
				std::memcpy(back.data(), mapped, Bytes);
				const double r  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
				s.bar_write_mbs = w > 0 ? 8.0 / w : 0;
				s.bar_read_mbs  = r > 0 ? 8.0 / r : 0;
				s.bar_line_ns   = lines(static_cast<const uint8_t*>(mapped));
				g->device.unmapMemory(memory);
			}
			g->device.freeMemory(memory, nullptr);
		}
	}
	std::printf("[f0] dispositivo: ReBAR (DEVICE_LOCAL|HOST_VISIBLE) %.0f MiB; VK_EXT_map_memory_placed %s; "
	            "VK_EXT_external_memory_host %s; memory_budget %s | CPU sobre BAR: escribe %.0f MB/s, lee %.0f MB/s, "
	            "lectura suelta %.0f ns/linea (RAM %.0f ns/linea)\n",
	            s.rebar_mb, s.placed ? "si" : "NO", s.host_import ? "si" : "no", s.budget_ext ? "si" : "no", s.bar_write_mbs,
	            s.bar_read_mbs, s.bar_line_ns, s.host_line_ns);
	std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------------------------------
// Read sampler (Windows, native guest memory only)
#if defined(_WIN32)
void Retire(uint64_t page) {
	const uint32_t i = g_s->retired_next.fetch_add(1, std::memory_order_relaxed) % Retired;
	g_s->retired_ns[i].store(NowNs(), std::memory_order_relaxed);
	g_s->retired[i].store(page, std::memory_order_release);
}

bool RecentlyOurs(uint64_t page) {
	const int64_t now = NowNs();
	for (uint32_t i = 0; i < Retired; ++i)
		if (g_s->retired[i].load(std::memory_order_acquire) == page &&
		    now - g_s->retired_ns[i].load(std::memory_order_relaxed) < 3'000'000'000ll)
			return true;
	return false;
}

bool InGameCode(uint64_t a) {
	return a >= GameCodeBegin && a < GameCodeEnd;
}

LONG CALLBACK GuardHandler(PEXCEPTION_POINTERS info) {
	const auto* r = info->ExceptionRecord;
	if (r->ExceptionCode != STATUS_GUARD_PAGE_VIOLATION || r->NumberParameters < 2 || g_s == nullptr)
		return EXCEPTION_CONTINUE_SEARCH;
	const uint64_t t0   = Tsc();
	const uint64_t page = static_cast<uint64_t>(r->ExceptionInformation[1]) & ~(Page - 1u);
	const bool     write = r->ExceptionInformation[0] == 1;
	for (auto& p: g_s->probes) {
		const uint32_t st = p.state.load(std::memory_order_acquire);
		if ((st != 2 && st != 3) || p.page.load(std::memory_order_relaxed) != page) continue;
		uint32_t who = 4u; // other: emulator threads, HLE code
		const auto rip = static_cast<uint64_t>(info->ContextRecord->Rip);
		if (InGameCode(rip)) {
			who = 1u;
		} else if (!LiveTrace::g_mark_thread) {
			// A library routine (memcpy...) called from game code: a return address into the eboot near the stack top.
			const auto* sp = reinterpret_cast<const uint64_t*>(info->ContextRecord->Rsp);
			for (int i = 0; i < 16; ++i)
				if (InGameCode(sp[i])) {
					who = 2u;
					break;
				}
		}
		uint32_t expected = 0;
		const uint32_t bits = (write ? 2u : 1u) | (who << 2u);
		if (!p.hit.compare_exchange_strong(expected, bits)) g_s->probe[p.label][PrAgain].fetch_add(1, std::memory_order_relaxed);
		g_s->guard_hits.fetch_add(1, std::memory_order_relaxed);
		g_s->cost_handler_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	if (RecentlyOurs(page)) { // (a touch that raced the retirement: the guard is gone, the access runs again)
		g_s->guard_hits.fetch_add(1, std::memory_order_relaxed);
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

// Our guard off `page` (its protection otherwise as it is). True when it had ours.
bool ClearGuard(uint64_t page) {
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(page), &mbi, sizeof(mbi)) != sizeof(mbi) || mbi.State != MEM_COMMIT) return false;
	if ((mbi.Protect & PAGE_GUARD) == 0) return false;
	DWORD old = 0;
	if (VirtualProtect(reinterpret_cast<void*>(page), Page, mbi.Protect & ~static_cast<DWORD>(PAGE_GUARD), &old) == 0) return false;
	// (Somebody else's protection set in between, without our guard: theirs back.)
	if ((old & ~static_cast<DWORD>(PAGE_GUARD)) != (mbi.Protect & ~static_cast<DWORD>(PAGE_GUARD))) {
		DWORD ignored = 0;
		(void)VirtualProtect(reinterpret_cast<void*>(page), Page, old & ~static_cast<DWORD>(PAGE_GUARD), &ignored);
	}
	return true;
}

void Classify(ProbeSlot& p, bool ensured) {
	auto&          c   = g_s->probe[p.label];
	const uint32_t hit = p.hit.load(std::memory_order_acquire);
	c[PrLifeMs].fetch_add(static_cast<uint64_t>(std::max<int64_t>(0, (NowNs() - p.armed_ns) / 1'000'000)),
	                      std::memory_order_relaxed);
	if (hit != 0) {
		const uint32_t who = hit >> 2u;
		if ((hit & 1u) != 0) c[who == 1 ? PrReadGame : who == 2 ? PrReadLib : PrReadOther].fetch_add(1, std::memory_order_relaxed);
		else c[who == 1 || who == 2 ? PrWriteGame : PrWriteOther].fetch_add(1, std::memory_order_relaxed);
	} else {
		c[ensured ? PrEnsured : PrUntouched].fetch_add(1, std::memory_order_relaxed);
	}
}

void RetireProbe(ProbeSlot& p, bool ensured) {
	uint32_t e = 2;
	if (!p.state.compare_exchange_strong(e, 3, std::memory_order_acq_rel)) return;
	const uint64_t page = p.page.load(std::memory_order_relaxed);
	const bool     had  = ClearGuard(page);
	if (!had && p.hit.load(std::memory_order_acquire) == 0 && !ensured)
		g_s->probe[p.label][PrLost].fetch_add(1, std::memory_order_relaxed); // (its protection changed by somebody else)
	else
		Classify(p, ensured);
	Retire(page);
	p.page.store(0, std::memory_order_relaxed);
	p.hit.store(0, std::memory_order_relaxed);
	p.state.store(0, std::memory_order_release);
	g_s->armed.fetch_sub(1, std::memory_order_relaxed);
}

uint64_t Rng() {
	static uint64_t x = 0x2545f4914f6cdd1dull ^ static_cast<uint64_t>(NowNs());
	x ^= x << 13u, x ^= x >> 7u, x ^= x << 17u;
	return x;
}

// One page of label `l`, chosen at random by size.
uint64_t PickPage(uint32_t l) {
	std::lock_guard lock(g_s->label_mutex);
	const auto&     v = g_s->ranges[l];
	if (v.empty()) return 0;
	uint64_t total = 0;
	for (const auto& r: v) total += (r.second - r.first) / Page;
	if (total == 0) return 0;
	uint64_t at = Rng() % total;
	for (const auto& r: v) {
		const uint64_t pages = (r.second - r.first) / Page;
		if (at < pages) return ((r.first + Page - 1u) & ~(Page - 1u)) + at * Page;
		at -= pages;
	}
	return 0;
}

void Arm(ProbeSlot& p, uint32_t l, uint64_t page) {
	uint32_t e = 0;
	if (!p.state.compare_exchange_strong(e, 1, std::memory_order_acq_rel)) return;
	bool ok = false;
	MEMORY_BASIC_INFORMATION mbi {};
	if (VirtualQuery(reinterpret_cast<const void*>(page), &mbi, sizeof(mbi)) == sizeof(mbi) && mbi.State == MEM_COMMIT &&
	    mbi.Type == MEM_PRIVATE && (mbi.Protect & (PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE)) == 0) {
		const DWORD base = mbi.Protect & 0xffu;
		if (base == PAGE_READONLY || base == PAGE_READWRITE) {
			p.page.store(page, std::memory_order_relaxed);
			p.hit.store(0, std::memory_order_relaxed);
			p.label    = l;
			p.base     = base;
			p.armed_ns = NowNs();
			p.state.store(2, std::memory_order_release); // (before the guard: a touch right after it finds the slot)
			DWORD old = 0;
			if (VirtualProtect(reinterpret_cast<void*>(page), Page, mbi.Protect | PAGE_GUARD, &old) != 0) {
				if (old != mbi.Protect) { // (cannot happen with the gate held alone; theirs back if it does)
					DWORD ignored = 0;
					(void)VirtualProtect(reinterpret_cast<void*>(page), Page, old, &ignored);
				} else {
					ok = true;
				}
			}
			if (!ok) {
				Retire(page);
				p.page.store(0, std::memory_order_relaxed);
			}
		}
	}
	if (ok) {
		g_s->armed.fetch_add(1, std::memory_order_relaxed);
		g_s->probe[l][p.base == PAGE_READONLY ? PrArmedRo : PrArmedRw].fetch_add(1, std::memory_order_relaxed);
	} else {
		p.state.store(0, std::memory_order_release);
	}
}

// f0 thread, every ~30 ms: the probes armed ~60 ms ago retired, new ones armed; the gate alone around each page only.
void SampleStep() {
	auto& s = *g_s;
	if (s.gate_lock == nullptr) return;
	const int64_t  now  = NowNs();
	static int64_t last = 0;
	if (now - last < 30'000'000) return; // (every 30 ms: at most 8 pages armed, 32 at a time, each ~60 ms)
	last = now;
	auto gated = [&](auto&& work) {
		const int64_t t0 = NowNs();
		s.gate_lock();
		work();
		s.gate_unlock();
		s.gate_ns.fetch_add(static_cast<uint64_t>(NowNs() - t0), std::memory_order_relaxed);
	};
	for (auto& p: s.probes)
		if (p.state.load(std::memory_order_acquire) == 2 && (p.hit.load(std::memory_order_relaxed) != 0 || now - p.armed_ns > 60'000'000))
			gated([&] { RetireProbe(p, false); });
	const uint32_t targets = s.target_count.load(std::memory_order_acquire);
	if (targets == 0) return;
	static uint32_t round = 0;
	uint32_t        armed_now = 0;
	for (auto& p: s.probes) {
		if (armed_now >= 8 || s.armed.load(std::memory_order_relaxed) >= 32) break;
		if (p.state.load(std::memory_order_acquire) != 0) continue;
		const uint32_t l    = s.targets[round++ % targets].load(std::memory_order_relaxed);
		const uint64_t page = PickPage(l);
		if (page == 0) continue;
		gated([&] { Arm(p, l, page); });
		++armed_now;
	}
}
#else
void SampleStep() {}
#endif

// The labels sampled: ring and DefaultTagg (what F2 would share), the heaps and pool, and the two labels with most upload
// bytes since start. (f0 thread)
void ChooseTargets() {
	auto&                 s = *g_s;
	std::vector<uint32_t> t;
	const uint32_t        n = s.label_count.load(std::memory_order_acquire);
	auto                  add = [&](uint32_t l) {
        if (l != 0 && l < n && std::find(t.begin(), t.end(), l) == t.end() && t.size() < 8) t.push_back(l);
	};
	{
		std::lock_guard lock(s.label_mutex);
		for (const char* name: {"ring", "DefaultTagg", "gpuheap", "cpuheap", "pool"})
			for (uint32_t l = 1; l < n; ++l)
				if (std::strcmp(s.names[l], name) == 0) add(l);
	}
	std::vector<std::pair<uint64_t, uint32_t>> by;
	for (uint32_t l = 1; l < n; ++l) by.emplace_back(s.up_bytes[l].load(std::memory_order_relaxed), l);
	std::sort(by.rbegin(), by.rend());
	for (size_t i = 0; i < by.size() && i < 2; ++i)
		if (by[i].first != 0) add(by[i].second);
	for (size_t i = 0; i < t.size(); ++i) s.targets[i].store(t[i], std::memory_order_relaxed);
	s.target_count.store(static_cast<uint32_t>(t.size()), std::memory_order_release);
}

void Calibrate() {
	const uint64_t c0 = Tsc();
	const auto     t0 = std::chrono::steady_clock::now();
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	const uint64_t c1 = Tsc();
	const double   ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
	if (ns > 0 && c1 > c0) g_s->tsc_per_ns.store(static_cast<double>(c1 - c0) / ns, std::memory_order_relaxed);
}

void Summary(bool at_exit);

void ThreadRun() {
	LocalPlatform::SetThreadName("Kyty.F0");
	g_s->thread_handle = LocalPlatform::CurrentThreadHandle();
	Calibrate();
#if defined(_WIN32)
	if (AddVectoredExceptionHandler(1, &GuardHandler) == nullptr) std::printf("[f0] sin manejador de guardas: sin muestreo de lecturas\n");
#endif
	int64_t last_labels = 0, last_targets = 0;
	for (;;) {
		const int64_t now = NowNs();
		Resolve();
		if (!g_s->device_done.load(std::memory_order_relaxed) && g_s->graphics_ready.load(std::memory_order_acquire) && now - g_s->start_ns > 20'000'000'000ll) {
			DeviceFacts();
			g_s->device_done.store(true, std::memory_order_release);
		}
		if (now - last_labels > 2'000'000'000ll) {
			RefreshLabels();
			last_labels = now;
		}
		if (now - last_targets > 5'000'000'000ll) {
			ChooseTargets();
			last_targets = now;
		}
		SampleStep();
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Process figures
double ProcessCpuSeconds() {
#if defined(_WIN32)
	FILETIME c {}, e {}, k {}, u {};
	if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) == 0) return 0;
	const auto v = [](const FILETIME& f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32u) | f.dwLowDateTime; };
	return static_cast<double>(v(k) + v(u)) / 1e7;
#else
	timespec t {};
	(void)clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
	return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_nsec) / 1e9;
#endif
}

void ProcessMemory(double* ws_mb, double* commit_mb) {
	*ws_mb = *commit_mb = 0;
#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS_EX pmc {};
	pmc.cb = sizeof(pmc);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)) != 0) {
		*ws_mb     = static_cast<double>(pmc.WorkingSetSize) / 1048576.0;
		*commit_mb = static_cast<double>(pmc.PrivateUsage) / 1048576.0;
	}
#endif
}

void Vram(double* used_mb, double* budget_mb) {
	*used_mb = *budget_mb = 0;
	if (!g_s->graphics_ready.load(std::memory_order_acquire)) return;
	auto* g = g_s->graphics;
	if (g == nullptr || g->allocator == nullptr) return;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(g->allocator, budgets);
	const auto& mem = g->GetPhysicalDeviceMemoryProperties();
	for (uint32_t h = 0; h < mem.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; ++h) {
		if (!(mem.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) continue;
		*used_mb += static_cast<double>(budgets[h].usage) / 1048576.0;
		*budget_mb += static_cast<double>(budgets[h].budget) / 1048576.0;
	}
}

bool IsF2Label(uint32_t l) {
	return std::strcmp(g_s->names[l], "ring") == 0 || std::strcmp(g_s->names[l], "DefaultTagg") == 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The 5 s report (render thread, at a flip)
void Report(int64_t flip_ns) {
	auto&        s      = *g_s;
	const double tpn    = s.tsc_per_ns.load(std::memory_order_relaxed);
	const double secs   = static_cast<double>(flip_ns - s.window_start) / 1e9;
	Window       w;
	w.t_s               = static_cast<double>(flip_ns - s.start_ns) / 1e9;
	// Frames
	std::vector<double> ft, busy, cpu, ft_on, ft_off;
	double              sum_ft = 0, wait = 0;
	for (const auto& f: s.flips_window) {
		ft.push_back(f.frame_ms);
		if (f.ab == 1) ft_on.push_back(f.frame_ms);
		if (f.ab == 2) ft_off.push_back(f.frame_ms);
		busy.push_back(f.busy_ms);
		sum_ft += f.frame_ms;
		wait += f.gpu_wait_ms;
		cpu.push_back(f.cpu_ms);
	}
	w.frames = static_cast<double>(ft.size());
	if (ft.empty()) return;
	w.fps    = sum_ft > 0 ? 1000.0 * w.frames / sum_ft : 0;
	{
		std::vector<double> sorted = ft;
		std::sort(sorted.rbegin(), sorted.rend());
		const size_t n1 = std::max<size_t>(1, sorted.size() / 100);
		double       m  = 0;
		for (size_t i = 0; i < n1; ++i) m += sorted[i];
		w.low1 = m > 0 ? 1000.0 * static_cast<double>(n1) / m : 0;
		for (double x: ft)
			if (x > 33.3) w.hitches += 1;
	}
	w.ft50       = Percentile(ft, 0.5);
	w.ft50_on    = Percentile(ft_on, 0.5);
	w.ft50_off   = Percentile(ft_off, 0.5);
	w.ft99       = Percentile(ft, 0.99);
	w.thread50   = Percentile(busy, 0.5);
	w.gpu_wait   = wait / w.frames;
	double cpu_sum = 0;
	for (double x: cpu) cpu_sum += x;
	w.cpu_thread = cpu_sum / w.frames;
	// GPU
	std::vector<FrameGpu> gpu;
	{
		std::lock_guard lock(s.gpu_mutex);
		gpu.swap(s.window_gpu);
	}
	{
		std::vector<double> b, sides[Sides];
		for (const auto& f: gpu) {
			b.push_back(f.busy_ms);
			for (uint32_t i = 0; i < Sides; ++i) sides[i].push_back(f.side_ms[i]);
		}
		w.gpu_frames = static_cast<double>(gpu.size());
		w.gpu50      = Percentile(b, 0.5);
		w.gpu90      = Percentile(b, 0.9);
		for (uint32_t i = 0; i < Sides; ++i) w.gpu_side50[i] = Percentile(sides[i], 0.5);
	}
	// Game side
	std::vector<GameFrame> game;
	{
		std::lock_guard lock(s.game_mutex);
		game.swap(s.window_game);
	}
	{
		std::vector<double> b, wt, c;
		double              submits = 0;
		for (const auto& f: game) {
			b.push_back(f.build_ms);
			wt.push_back(f.wait_ms);
			c.push_back(f.cpu_ms);
			submits += f.submits;
		}
		w.game_frames  = static_cast<double>(game.size());
		w.game_build50 = Percentile(b, 0.5);
		w.game_build90 = Percentile(b, 0.9);
		w.game_wait50  = Percentile(wt, 0.5);
		w.game_cpu50   = Percentile(c, 0.5);
		(void)submits;
	}
	// Process
	const double process_s = ProcessCpuSeconds();
	w.cpu_process          = s.last_process_s > 0 ? (process_s - s.last_process_s) * 1000.0 / w.frames : 0;
	s.last_process_s       = process_s;
	ProcessMemory(&w.ram_ws, &w.ram_commit);
	Vram(&w.vram_used, &w.vram_budget);
	// Uploads, ring copies, linear copies by label
	const uint32_t nl = s.label_count.load(std::memory_order_acquire);
	struct Lab {
		uint32_t l;
		double   up, up_n, ring, lc, lc_n, lc_ms;
	};
	std::vector<Lab> labs;
	for (uint32_t l = 0; l < nl; ++l) {
		const auto d = [&](std::atomic<uint64_t>& v, uint64_t& last) {
			const uint64_t x = v.load(std::memory_order_relaxed);
			const uint64_t r = x - last;
			last             = x;
			return static_cast<double>(r);
		};
		Lab a {l, d(s.up_bytes[l], s.last_up[l]) / 1048576.0 / w.frames, d(s.up_n[l], s.last_up_n[l]) / w.frames,
		       d(s.ring_bytes[l], s.last_ring[l]) / 1048576.0 / w.frames, 0, 0, 0};
		w.ring_n += d(s.ring_n[l], s.last_ring_n[l]) / w.frames;
		a.lc    = d(s.lc_bytes[l], s.last_lc[l]) / 1048576.0 / w.frames;
		a.lc_n  = d(s.lc_n[l], s.last_lc_n[l]) / w.frames;
		a.lc_ms = d(s.lc_cycles[l], s.last_lc_cycles[l]) / tpn / 1e6 / w.frames;
		w.up_mb += a.up;
		w.up_n += a.up_n;
		w.ring_mb += a.ring;
		w.lc_n += a.lc_n;
		w.lc_mb += a.lc;
		w.lc_ms += a.lc_ms;
		if (l != 0 && IsF2Label(l)) {
			w.f2_up_mb += a.up + a.ring;
			w.f2_lc_ms += a.lc_ms;
		}
		labs.push_back(a);
	}
	// Costs
	const uint64_t cg = s.cost_gpu_cycles.load(), cm = s.cost_game_cycles.load(), gate = s.gate_ns.load(),
	               hc = s.cost_handler_cycles.load();
	w.cost_gpu        = static_cast<double>(cg - s.last_cost_gpu) / tpn / 1e6 / w.frames;
	w.cost_game       = static_cast<double>(cm - s.last_cost_game + hc - s.last_handler) / tpn / 1e6 / w.frames;
	w.gate_ms_s       = secs > 0 ? static_cast<double>(gate - s.last_gate) / 1e6 / secs : 0;
	const double th_s = LocalPlatform::ThreadCpuSeconds(s.thread_handle);
	w.cost_thread_pct = secs > 0 ? 100.0 * (th_s - s.last_thread_s) / secs : 0;
	s.last_cost_gpu = cg, s.last_cost_game = cm, s.last_gate = gate, s.last_handler = hc, s.last_thread_s = th_s;
	// Verdict
	if (w.gpu50 > 0 && (w.gpu50 >= 0.9 * w.ft50 || w.gpu_wait >= 1.0)) w.verdict = 1;
	else if (w.thread50 >= 0.85 * w.ft50) w.verdict = 2;
	else if (w.game_build50 >= 0.85 * w.ft50) w.verdict = 3;
	else w.verdict = 0;

	// ---- lines
	std::printf("[f0] %.1f s, %.0f frames | fps media %.1f, 1%% bajo %.1f, frametime p50 %.2f p99 %.2f ms, tirones>33ms %.0f | "
	            "GPU ms/frame p50 %.2f p90 %.2f (dcb %.2f acb %.2f emu %.2f subidas %.2f otros %.2f; %.0f frames medidos) | "
	            "hilo GPU ocupado p50 %.2f, espera a la GPU %.2f, cpu %.2f ms/frame | juego: construye p50 %.2f p90 %.2f, "
	            "espera al emulador p50 %.2f, cpu p50 %.2f ms (%.0f frames) | cpu proceso %.1f ms/frame | RAM ws %.0f MB "
	            "commit %.0f MB | VRAM %.0f/%.0f MB | port2 ON/OFF frametime p50 %.2f/%.2f | VEREDICTO %s, techo GPU %.0f fps (p90 %.0f)\n",
	            secs, w.frames, w.fps, w.low1, w.ft50, w.ft99, w.hitches, w.gpu50, w.gpu90, w.gpu_side50[SideDcb],
	            w.gpu_side50[SideAcb], w.gpu_side50[SideHost], w.gpu_side50[SideUpload], w.gpu_side50[SideOther], w.gpu_frames,
	            w.thread50, w.gpu_wait, w.cpu_thread, w.game_build50, w.game_build90, w.game_wait50, w.game_cpu50, w.game_frames,
	            w.cpu_process, w.ram_ws, w.ram_commit, w.vram_used, w.vram_budget, w.ft50_on, w.ft50_off, VerdictNames[w.verdict],
	            w.gpu50 > 0 ? 1000.0 / w.gpu50 : 0.0, w.gpu90 > 0 ? 1000.0 / w.gpu90 : 0.0);
	std::sort(labs.begin(), labs.end(), [](const Lab& a, const Lab& b) { return a.up + a.ring > b.up + b.ring; });
	std::printf("[f0] subidas %.2f MB/frame (%.0f), anillo %.2f MB/frame (%.0f), en ring+DefaultTagg %.2f MB/frame | por etiqueta:",
	            w.up_mb, w.up_n, w.ring_mb, w.ring_n, w.f2_up_mb);
	for (size_t i = 0; i < labs.size() && i < 8; ++i)
		if (labs[i].up + labs[i].ring > 0.001)
			std::printf(" %s=%.2f/%.2f(%.0f)", labs[i].l == 0 ? "?" : s.names[labs[i].l], labs[i].up, labs[i].ring, labs[i].up_n);
	std::printf(" | copias lineales %.1f/frame %.2f MB/frame cpu %.3f ms/frame (en ring+DefaultTagg %.3f) destino:", w.lc_n, w.lc_mb,
	            w.lc_ms, w.f2_lc_ms);
	std::sort(labs.begin(), labs.end(), [](const Lab& a, const Lab& b) { return a.lc_ms > b.lc_ms; });
	for (size_t i = 0; i < labs.size() && i < 3; ++i)
		if (labs[i].lc_n > 0.01)
			std::printf(" %s=%.1f/%.3fms", labs[i].l == 0 ? "?" : s.names[labs[i].l], labs[i].lc_n, labs[i].lc_ms);
	std::printf("\n");
	std::printf("[f0] lecturas de CPU (muestreo de paginas; armadas ro/rw, 1er acceso: lee juego/lib/otro, escribe juego/otro, "
	            "intactas, perdidas):");
	const uint32_t tc = s.target_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < tc; ++i) {
		const uint32_t l = s.targets[i].load(std::memory_order_relaxed);
		std::array<uint64_t, PrCounters> d {};
		for (uint32_t k = 0; k < PrCounters; ++k) {
			const uint64_t v = s.probe[l][k].load(std::memory_order_relaxed);
			d[k]             = v - s.last_probe[l][k];
			s.last_probe[l][k] = v;
		}
		std::printf(" %s %llu/%llu: %llu/%llu/%llu, %llu/%llu, %llu, %llu;", s.names[l], (unsigned long long)d[PrArmedRo],
		            (unsigned long long)d[PrArmedRw], (unsigned long long)d[PrReadGame], (unsigned long long)d[PrReadLib],
		            (unsigned long long)d[PrReadOther], (unsigned long long)d[PrWriteGame], (unsigned long long)d[PrWriteOther],
		            (unsigned long long)d[PrUntouched], (unsigned long long)d[PrLost]);
	}
	std::printf(" | coste f0: hilo GPU %.3f ms/frame, juego %.3f ms/frame, hilo f0 %.1f%% de un nucleo, puerta %.2f ms/s, "
	            "guardas %llu | marcas perdidas %llu, registros tarde %llu, muertos %llu\n",
	            w.cost_gpu, w.cost_game, w.cost_thread_pct, w.gate_ms_s, (unsigned long long)s.guard_hits.load(),
	            (unsigned long long)s.dropped_marks.load(), (unsigned long long)s.late_records.load(),
	            (unsigned long long)s.dead_records.load());
	std::fflush(stdout);
	// ---- CSV
	if (s.csv == nullptr) {
		std::string path = "f0.csv";
		if (const char* log = std::getenv("KYTY_RUN_LOG"); log != nullptr && *log != '\0') {
			path = log;
			if (const auto dot = path.find(".out.log"); dot != std::string::npos) path.resize(dot);
			path += "-f0.csv";
		}
		s.csv = std::fopen(path.c_str(), "w");
		if (s.csv != nullptr) {
			std::fprintf(s.csv, "t_s,frames,fps,fps_low1,ft_p50,ft_p99,hitches33,gpu_p50,gpu_p90,gpu_dcb,gpu_acb,gpu_emu,gpu_subidas,"
			                    "gpu_otros,hilo_gpu_p50,espera_gpu,cpu_hilo_gpu,juego_construye_p50,juego_construye_p90,"
			                    "juego_espera_p50,juego_cpu_p50,cpu_proceso,ram_ws_mb,ram_commit_mb,vram_mb,vram_presupuesto_mb,"
			                    "subidas_mb,subidas_n,subidas_f2_mb,anillo_mb,copias_n,copias_mb,copias_ms,copias_f2_ms,"
			                    "coste_f0_gpu_ms,coste_f0_juego_ms,ft_p50_port2_on,ft_p50_port2_off,veredicto\n");
			std::printf("[f0] CSV: %s\n", path.c_str());
		}
	}
	if (s.csv != nullptr) {
		std::fprintf(s.csv,
		             "%.1f,%.0f,%.2f,%.2f,%.3f,%.3f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,"
		             "%.0f,%.0f,%.0f,%.0f,%.3f,%.1f,%.3f,%.3f,%.2f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%s\n",
		             w.t_s, w.frames, w.fps, w.low1, w.ft50, w.ft99, w.hitches, w.gpu50, w.gpu90, w.gpu_side50[0], w.gpu_side50[1],
		             w.gpu_side50[2], w.gpu_side50[3], w.gpu_side50[4], w.thread50, w.gpu_wait, w.cpu_thread, w.game_build50,
		             w.game_build90, w.game_wait50, w.game_cpu50, w.cpu_process, w.ram_ws, w.ram_commit, w.vram_used, w.vram_budget,
		             w.up_mb, w.up_n, w.f2_up_mb, w.ring_mb, w.lc_n, w.lc_mb, w.lc_ms, w.f2_lc_ms, w.cost_gpu, w.cost_game,
		             w.ft50_on, w.ft50_off, VerdictNames[w.verdict]);
		std::fflush(s.csv);
	}
	{
		std::lock_guard lock(s.summary_mutex);
		s.windows.push_back(w);
	}
	if (flip_ns - s.last_summary_ns >= 60'000'000'000ll) {
		s.last_summary_ns = flip_ns;
		Summary(false);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Summary: medians of the windows, the read sampler since start, the device, the F2 verdict.
void Summary(bool at_exit) {
	auto& s = *g_s;
	std::lock_guard lock(s.summary_mutex);
	// Game windows only (at least 20 fps and some GPU frames measured): menus and loads left out.
	std::vector<const Window*> v;
	for (const auto& w: s.windows)
		if (w.fps >= 20.0 && w.frames >= 50) v.push_back(&w);
	if (v.empty()) {
		std::printf("[f0] RESUMEN%s: sin ventanas de juego todavia\n", at_exit ? " FINAL" : "");
		std::fflush(stdout);
		return;
	}
	const auto med = [&](double Window::*field, double p = 0.5) {
		std::vector<double> x;
		for (const auto* w: v) x.push_back(w->*field);
		return Percentile(x, p);
	};
	const auto med_pos = [&](double Window::*field) { // (windows that have it)
		std::vector<double> x;
		for (const auto* w: v)
			if (w->*field > 0) x.push_back(w->*field);
		return Percentile(x, 0.5);
	};
	uint32_t verdicts[4] {};
	for (const auto* w: v) verdicts[w->verdict]++;
	const double n    = static_cast<double>(v.size());
	const double gpu  = med(&Window::gpu50), gpu90 = med(&Window::gpu50, 0.9), ft = med(&Window::ft50);
	const double f2up = med(&Window::f2_up_mb), f2lc = med(&Window::f2_lc_ms);
	// Reads of the F2 labels since start: the share of sampled pages whose first touch was a game read, and an estimate of
	// the pages a frame reads there (the share x the label's pages x frames per sample life). [H]
	double read_pages_frame = 0, armed_all = 0, reads_all = 0, lib_all = 0;
	std::printf("[f0] RESUMEN%s (%u ventanas de juego): fps media p50 %.1f, 1%% bajo p50 %.1f, frametime p50 %.2f ms, p99 p50 %.2f ms, "
	            "tirones>33ms %.0f en total | GPU p50 %.2f ms/frame (p90 de ventanas %.2f), hilo GPU %.2f, juego construye %.2f, "
	            "espera al emulador %.2f | ventanas GPU %u, CPU-hilo-GPU %u, CPU-juego %u, tope/mixto %u\n",
	            at_exit ? " FINAL" : "", static_cast<unsigned>(v.size()), med(&Window::fps), med(&Window::low1), ft, med(&Window::ft99),
	            [&] {
		            double h = 0;
		            for (const auto* w: v) h += w->hitches;
		            return h;
	            }(),
	            gpu, gpu90, med(&Window::thread50), med(&Window::game_build50), med(&Window::game_wait50), verdicts[1], verdicts[2],
	            verdicts[3], verdicts[0]);
	std::printf("[f0] RESUMEN lecturas:");
	const uint32_t nl = s.label_count.load(std::memory_order_acquire);
	for (uint32_t l = 1; l < nl; ++l) {
		const auto   c     = [&](uint32_t k) { return static_cast<double>(s.probe[l][k].load(std::memory_order_relaxed)); };
		const double armed = c(PrArmedRo) + c(PrArmedRw);
		if (armed < 1) continue;
		const double reads = c(PrReadGame) + c(PrReadLib);
		const double life  = std::max(1.0, c(PrLifeMs) / std::max(1.0, armed - c(PrLost)));
		const double share = reads / armed;
		const double pages = s.label_mb[l] * 256.0;
		const double per_frame = std::min(pages, share * pages * (ft / life));
		std::printf(" %s: %.0f armadas, lee juego %.0f + lib %.0f + otro %.0f (%.2f%%), escribe %.0f, vida media %.0f ms, "
		            "~%.0f paginas leidas/frame;",
		            s.names[l], armed, c(PrReadGame), c(PrReadLib), c(PrReadOther), 100.0 * share, c(PrWriteGame) + c(PrWriteOther),
		            life, per_frame);
		if (IsF2Label(l)) {
			read_pages_frame += per_frame;
			armed_all += armed;
			reads_all += c(PrReadGame);
			lib_all += c(PrReadLib);
		}
	}
	std::printf("\n");
	// F2 verdict
	const double line_ns = s.bar_line_ns > 0 ? s.bar_line_ns : 1000.0;
	const double read_bw = s.bar_read_mbs > 0 ? s.bar_read_mbs : 50.0;
	const double cost_lo = read_pages_frame * line_ns / 1e6;                        // one line per page read
	const double cost_hi = read_pages_frame * 4096.0 / (read_bw * 1048576.0) * 1e3; // the whole page
	const char*  verdict = "viable";
	std::string  why;
	if (!s.device_done.load(std::memory_order_acquire)) {
		verdict = "sin datos";
		why += "el dispositivo no se llego a medir (menos de 20 s con GPU); ";
	} else if (s.rebar_mb < 256) {
		verdict = "NO viable";
		why += "sin ReBAR (memoria DEVICE_LOCAL|HOST_VISIBLE < 256 MiB); ";
	}
	if (armed_all < 200) {
		if (std::strcmp(verdict, "viable") == 0) verdict = "dudoso";
		why += "pocas muestras de lectura en ring/DefaultTagg; ";
	}
	if (cost_hi >= 1.0 && cost_lo >= 0.2 && std::strcmp(verdict, "sin datos") != 0) {
		verdict = "NO viable";
		why += "la CPU del juego lee la memoria de frame (coste estimado en BAR alto); ";
	} else if (cost_hi >= 0.3 && std::strcmp(verdict, "viable") == 0) {
		verdict = "dudoso";
		why += "la CPU lee algo de la memoria de frame; ";
	}
	if (f2up < 2.0 && f2lc < 0.2) {
		if (std::strcmp(verdict, "viable") == 0) verdict = "dudoso";
		why += "poco que quitar (subidas de ring+DefaultTagg < 2 MB/frame y copias < 0,2 ms); ";
	}
	if (verdicts[1] * 2 >= v.size()) {
		if (std::strcmp(verdict, "viable") == 0) verdict = "dudoso";
		why += "la mayoria de ventanas las limita la GPU: quitar CPU no sube fps; ";
	}
	if (!s.placed) {
		if (std::strcmp(verdict, "viable") == 0) verdict = "dudoso";
		why += "sin VK_EXT_map_memory_placed (habria que copiar o remapear); ";
	}
	std::printf("[f0] RESUMEN F2: %s. Datos: ReBAR %.0f MiB, map_memory_placed %s, CPU lee BAR %.0f MB/s y %.0f ns/linea; "
	            "subidas+anillo de ring/DefaultTagg p50 %.2f MB/frame; copias lineales en esos rangos p50 %.3f ms/frame de CPU; "
	            "lecturas del juego en ring/DefaultTagg: %.0f de %.0f paginas muestreadas (+%.0f via biblioteca), ~%.0f paginas/frame, "
	            "coste estimado en BAR %.3f-%.3f ms/frame [H]; GPU-bound en %u de %.0f ventanas. %s\n",
	            verdict, s.rebar_mb, s.placed ? "si" : "no", s.bar_read_mbs, s.bar_line_ns, f2up, f2lc, reads_all, armed_all, lib_all,
	            read_pages_frame, cost_lo, cost_hi, verdicts[1], n, why.empty() ? "Sin objeciones medidas." : why.c_str());
	std::printf("[f0] RESUMEN port2 (KYTY_PORT2=1 alterna 5 s): frametime p50 con ON %.2f ms, OFF %.2f ms\n", med_pos(&Window::ft50_on),
	            med_pos(&Window::ft50_off));
	std::printf("[f0] RESUMEN techo: GPU p50 %.2f ms/frame = %.0f fps si manda la GPU (p90 de ventanas %.2f ms = %.0f fps); "
	            "frame actual p50 %.2f ms = %.0f fps; hilo GPU %.2f ms, juego %.2f ms (techo de CPU ~%.0f fps)\n",
	            gpu, gpu > 0 ? 1000.0 / gpu : 0.0, gpu90, gpu90 > 0 ? 1000.0 / gpu90 : 0.0, ft, ft > 0 ? 1000.0 / ft : 0.0,
	            med(&Window::thread50), med(&Window::game_build50),
	            std::max(med(&Window::thread50), med(&Window::game_build50)) > 0
	                ? 1000.0 / std::max(med(&Window::thread50), med(&Window::game_build50))
	                : 0.0);
	std::fflush(stdout);
}

void AtExit() {
	if (g_s != nullptr) Summary(true);
}

} // namespace

// =====================================================================================================================
bool Enabled() noexcept {
	const char* v = std::getenv("KYTY_F0");
	if (v == nullptr || v[0] != '1') return false;
	g_s           = new State;
	g_s->start_ns = NowNs();
	std::snprintf(g_s->names[0], sizeof(g_s->names[0]), "?");
	for (const char* name: {"ring", "DefaultTagg", "ctx", "dcb", "cpuheap", "gpuheap", "pool", "cpupage", "gpupage", "cpubkt",
	                        "gpubkt", "VirtualMemo"})
		(void)LabelIndexLocked(name);
	std::printf("[f0] KYTY_F0=1: medicion F0 (tiempo real de GPU por cola, lecturas de CPU de la memoria de frame, subidas por "
	            "etiqueta, construccion del frame del juego). Sin cambio de imagen.\n");
	std::fflush(stdout);
	std::thread(ThreadRun).detach();
	std::atexit(AtExit);
	return true;
}

void BufferBegin(Libs::Graphics::GraphicContext* graphics, void* raw, bool main) {
	if (!On() || raw == nullptr) return;
	const uint64_t t0 = Tsc();
	std::call_once(g_s->gpu_once, [graphics] { InitGpu(graphics); });
	if (!g_s->gpu_ok.load(std::memory_order_relaxed)) return;
	auto&          s = *g_s;
	const uint32_t i = s.rnext.fetch_add(1, std::memory_order_relaxed) % RecordCount;
	Record&        r = s.records[i];
	uint32_t       e = 0;
	uint32_t       q = 0;
	if (!r.state.compare_exchange_strong(e, 1, std::memory_order_acq_rel)) {
		s.dropped_records.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (!AllocQuery(&q)) {
		r.state.store(0, std::memory_order_release);
		return;
	}
	const bool gpu_thread = LiveTrace::g_mark_thread;
	r.cmd     = raw;
	r.open_ns = NowNs();
	r.kind    = main ? 0 : 1;
	r.n       = 1;
	r.q[0]    = q;
	r.side[0] = !main ? SideUpload : gpu_thread ? t_side : SideOther;
	r.last    = r.side[0];
	Stamp(raw, q, true);
	// (This thread's open buffers: a slot taken by one never submitted is let go.)
	Record** slot = nullptr;
	for (auto& o: t_open)
		if (o == nullptr || o->cmd == raw) {
			slot = &o;
			break;
		}
	if (slot == nullptr) {
		slot            = &t_open[0];
		uint32_t expect = 1;
		if (t_main == *slot) t_main = nullptr;
		if ((*slot)->state.compare_exchange_strong(expect, 3)) s.dead_records.fetch_add(1, std::memory_order_relaxed);
	}
	*slot = &r;
	if (main && gpu_thread) t_main = &r;
	if (gpu_thread) s.cost_gpu_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void BufferSubmit(void* raw, bool ended, int lane) {
	if (!On() || g_s == nullptr || !g_s->gpu_ok.load(std::memory_order_relaxed) || raw == nullptr) return;
	const uint64_t t0 = Tsc();
	Record*        r  = nullptr;
	for (auto& o: t_open)
		if (o != nullptr && o->cmd == raw) {
			r = o;
			o = nullptr;
			break;
		}
	if (r == nullptr) return;
	if (t_main == r) t_main = nullptr;
	if (ended || r->n + 1u > MaxMarks) {
		uint32_t expect = 1;
		if (r->state.compare_exchange_strong(expect, 3)) g_s->dead_records.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	uint32_t q = 0;
	if (!AllocQuery(&q)) {
		uint32_t expect = 1;
		(void)r->state.compare_exchange_strong(expect, 3);
		return;
	}
	Stamp(raw, q, false);
	r->q[r->n]    = q;
	r->side[r->n] = r->last;
	r->n++;
	r->lane    = static_cast<uint8_t>(lane);
	r->frame   = g_s->flips.load(std::memory_order_relaxed);
	r->open_ns = NowNs(); // (from here: the time its stamps may take to be available)
	uint32_t expect = 1;
	(void)r->state.compare_exchange_strong(expect, 2, std::memory_order_acq_rel);
	if (LiveTrace::g_mark_thread) g_s->cost_gpu_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void SliceBegin(bool compute_queue) {
	if (!On() || g_s == nullptr) return;
	const uint64_t t0 = Tsc();
	t_side            = compute_queue ? SideAcb : SideDcb;
	if (g_s->gpu_ok.load(std::memory_order_relaxed)) Mark(t_main, t_side);
	g_s->cost_gpu_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void SliceEnd() {
	if (!On() || g_s == nullptr) return;
	const uint64_t t0 = Tsc();
	t_side            = SideHost;
	if (g_s->gpu_ok.load(std::memory_order_relaxed)) Mark(t_main, t_side);
	g_s->cost_gpu_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void Flip(int64_t flip_ns, int64_t render_idle_ns, int64_t gpu_wait_ns, int ab) {
	if (!On() || g_s == nullptr) return;
	const uint64_t t0 = Tsc();
	auto&          s  = *g_s;
	s.flips.fetch_add(1, std::memory_order_release);
	const uint64_t cycles = LocalPlatform::CurrentThreadCycles();
	if (s.last_flip_ns != 0) {
		FlipSample f;
		f.frame_ms    = static_cast<double>(flip_ns - s.last_flip_ns) / 1e6;
		f.ab          = ab;
		const double idle = static_cast<double>(render_idle_ns - s.last_idle) / 1e6;
		f.busy_ms     = std::max(0.0, f.frame_ms - idle);
		f.gpu_wait_ms = static_cast<double>(gpu_wait_ns - s.last_wait) / 1e6;
		f.cpu_ms      = cycles > s.last_cycles && s.last_cycles != 0
		                    ? static_cast<double>(cycles - s.last_cycles) / s.tsc_per_ns.load(std::memory_order_relaxed) / 1e6
		                    : 0.0;
		if (s.flips_window.size() < 16384) s.flips_window.push_back(f);
	} else {
		s.window_start = flip_ns;
	}
	s.last_flip_ns = flip_ns;
	s.last_idle    = render_idle_ns;
	s.last_wait    = gpu_wait_ns;
	s.last_cycles  = cycles;
	if (flip_ns - s.window_start >= 5'000'000'000ll) {
		Report(flip_ns);
		s.flips_window.clear();
		s.window_start = flip_ns;
	}
	s.cost_gpu_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void NoteUploadSlow(uint64_t vaddr, uint64_t size) {
	const uint32_t l = LabelOf(vaddr);
	if (l == 0) NoteUnknown(vaddr);
	g_s->up_bytes[l].fetch_add(size, std::memory_order_relaxed);
	g_s->up_n[l].fetch_add(1, std::memory_order_relaxed);
}

void NoteRingCopySlow(uint64_t vaddr, uint64_t size) {
	const uint32_t l = LabelOf(vaddr);
	if (l == 0) NoteUnknown(vaddr);
	g_s->ring_bytes[l].fetch_add(size, std::memory_order_relaxed);
	g_s->ring_n[l].fetch_add(1, std::memory_order_relaxed);
}

LinearCopyScope::LinearCopyScope(uint64_t dst, uint64_t /*src*/, uint64_t bytes) noexcept {
	if (!On()) return;
	m_dst   = dst;
	m_bytes = bytes;
	m_start = Tsc();
}

LinearCopyScope::~LinearCopyScope() {
	if (m_start == 0) return;
	const uint64_t c = Tsc() - m_start;
	const uint32_t l = LabelOf(m_dst);
	if (l == 0) NoteUnknown(m_dst);
	g_s->lc_n[l].fetch_add(1, std::memory_order_relaxed);
	g_s->lc_bytes[l].fetch_add(m_bytes, std::memory_order_relaxed);
	g_s->lc_cycles[l].fetch_add(c, std::memory_order_relaxed);
}

// ---- (cpu1) The game thread's waits ----
namespace {
uint64_t ThisThreadKey() noexcept {
	return static_cast<uint64_t>(std::hash<std::thread::id> {}(std::this_thread::get_id())) | 1u;
}
std::atomic<uint64_t> g_game_thread {0};
struct WaitAcc {
	std::mutex                                mutex;
	double                                    ns[8] {};
	uint64_t                                  count[8] {};
	std::vector<std::pair<uintptr_t, double>> sites; // (return address, ns), at most 64
	uint64_t                                  frames   = 0;
	double                                    build_ms = 0.0, cpu_ms = 0.0;
	int64_t                                   since    = 0;
};
WaitAcc& Acc() {
	static auto* acc = new WaitAcc;
	return *acc;
}
void WaitMarkGameThread() noexcept {
	const uint64_t key = ThisThreadKey();
	if (g_game_thread.load(std::memory_order_relaxed) != key) g_game_thread.store(key, std::memory_order_relaxed);
}
// The eboot is mapped at 0x9'0000'0000 (01.007.000; the crash logs' "eboot.bin+" offsets).
void Site(char* out, size_t size, uintptr_t a) {
	if (a >= 0x900000000ull && a < 0x904000000ull)
		std::snprintf(out, size, "eboot+0x%llx", static_cast<unsigned long long>(a - 0x900000000ull));
	else
		std::snprintf(out, size, "0x%llx", static_cast<unsigned long long>(a));
}
void WaitFrame(double build_ms, double cpu_ms, int64_t now) {
	auto&           w = Acc();
	std::lock_guard lock(w.mutex);
	if (w.since == 0) w.since = now;
	w.frames++, w.build_ms += build_ms, w.cpu_ms += cpu_ms;
	if (now - w.since < 5000000000ll) return;
	const double n = static_cast<double>(w.frames), ms = 1e6 * n;
	double       waits = 0.0;
	for (double v: w.ns) waits += v;
	const double off = w.build_ms / n - w.cpu_ms / n - waits / ms;
	std::sort(w.sites.begin(), w.sites.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
	char   top[512] = {};
	size_t used     = 0;
	for (size_t i = 0; i < w.sites.size() && i < 6 && used < sizeof(top); ++i) {
		char name[48];
		Site(name, sizeof(name), w.sites[i].first);
		const int k = std::snprintf(top + used, sizeof(top) - used, "%s%s %.2f", i == 0 ? "" : ", ", name, w.sites[i].second / ms);
		if (k <= 0) break;
		used += static_cast<size_t>(k);
	}
	std::printf("[f0w] hilo del juego (%.0f frames): construye %.2f ms/frame = cpu %.2f + esperas %.2f (sema %.2f, cond %.2f, mutex %.2f, "
	            "flag %.2f, equeue %.2f, sleep %.2f, join %.2f, rwlock %.2f) + sin CPU y sin espera %.2f | esperas por sitio ms/frame: %s\n",
	            n, w.build_ms / n, w.cpu_ms / n, waits / ms, w.ns[0] / ms, w.ns[1] / ms, w.ns[2] / ms, w.ns[3] / ms, w.ns[4] / ms,
	            w.ns[5] / ms, w.ns[6] / ms, w.ns[7] / ms, off, top);
	std::fflush(stdout);
	for (auto& v: w.ns) v = 0.0;
	for (auto& v: w.count) v = 0;
	w.sites.clear();
	w.frames = 0, w.build_ms = 0.0, w.cpu_ms = 0.0, w.since = now;
}
// ---- (esperas) KYTY_ESPERAS: one line per slow frame of the game's submit thread ----
int64_t g_esp_thr_ns = 20000000; // (the threshold)
const char* const EspKind[9] = {"sema", "cond", "mutex", "flag", "equeue", "sleep", "join", "rwlock", "disco"};
struct EspSite {
	uintptr_t ret  = 0;
	int       kind = 0;
	int64_t   ns   = 0;
	uint32_t  n    = 0;
};
// Only the game's submit thread writes these (WaitScope on that thread, the suspend point on that thread).
struct EspState {
	int64_t  ns[9] {};
	uint32_t n[9] {};
	EspSite  sites[16];
	uint32_t site_count = 0, site_lost = 0;
	int64_t  last_exit_ns = 0, enter_ns = 0;
	uint64_t last_exit_cycles = 0;
	int64_t  cal_ns = 0;
	uint64_t cal_tsc = 0;
	double   tsc_per_ns = 0.0;
	int64_t  line_window_ns = 0;
	uint32_t lines_in_window = 0, lines_dropped = 0;
	uint64_t frames = 0, slow_frames = 0;
	// (esperas2) per thread now: each thread that calls sceAgcSuspendPoint has its own state (the CPU counter of
	// QueryThreadCycleTime is per thread: one shared state mixed two threads' counters -> "CPU 8562 ms" in 200 ms).
	uint32_t idx = 0;
	// (esperas2) this thread's time in the access-violation handler during the frame
	uint64_t fault_n = 0;
	int64_t  fault_ns = 0, fault_max_ns = 0;
	EspSite  fsites[8];
	uint32_t fsite_count = 0;
};
std::atomic<uint32_t>   g_esp_threads {0};
thread_local EspState* t_esp = nullptr;
EspState& Esp() {
	if (t_esp == nullptr) {
		t_esp      = new EspState;
		t_esp->idx = g_esp_threads.fetch_add(1, std::memory_order_relaxed) + 1u;
	}
	return *t_esp;
}
void EsperasNote(int kind, int64_t ns, const void* ret) noexcept {
	if (kind < 0 || kind > 8 || ns < 5000 || t_esp == nullptr) return;
	auto& e = *t_esp;
	e.ns[kind] += ns;
	e.n[kind]++;
	const auto a = reinterpret_cast<uintptr_t>(ret);
	for (uint32_t i = 0; i < e.site_count; ++i)
		if (e.sites[i].ret == a && e.sites[i].kind == kind) {
			e.sites[i].ns += ns, e.sites[i].n++;
			return;
		}
	if (e.site_count < 16) {
		e.sites[e.site_count++] = EspSite {a, kind, ns, 1};
		return;
	}
	// full: the smallest goes if this one is longer
	uint32_t m = 0;
	for (uint32_t i = 1; i < 16; ++i)
		if (e.sites[i].ns < e.sites[m].ns) m = i;
	if (e.sites[m].ns < ns) e.sites[m] = EspSite {a, kind, ns, 1};
	e.site_lost++;
}
void EsperasFaultNote(int64_t ns, uint64_t rip) noexcept {
	auto& e = *t_esp;
	e.fault_n++;
	e.fault_ns += ns;
	if (ns > e.fault_max_ns) e.fault_max_ns = ns;
	if (ns < 20000) return; // (sites: the faults of 20 us or more)
	const auto a = static_cast<uintptr_t>(rip);
	for (uint32_t i = 0; i < e.fsite_count; ++i)
		if (e.fsites[i].ret == a) {
			e.fsites[i].ns += ns, e.fsites[i].n++;
			return;
		}
	if (e.fsite_count < 8) {
		e.fsites[e.fsite_count++] = EspSite {a, 0, ns, 1};
		return;
	}
	uint32_t m = 0;
	for (uint32_t i = 1; i < 8; ++i)
		if (e.fsites[i].ns < e.fsites[m].ns) m = i;
	if (e.fsites[m].ns < ns) e.fsites[m] = EspSite {a, 0, ns, 1};
}
bool EspLineAllowed(EspState& e, int64_t now) {
	if (now - e.line_window_ns >= 1000000000ll) {
		if (e.lines_dropped != 0) std::printf("[esperas] (%u lineas no escritas en el ultimo segundo: tope 20/s)\n", e.lines_dropped);
		e.line_window_ns = now, e.lines_in_window = 0, e.lines_dropped = 0;
	}
	if (e.lines_in_window >= 20) {
		e.lines_dropped++;
		return false;
	}
	e.lines_in_window++;
	return true;
}
unsigned long long UnixMs() {
	return static_cast<unsigned long long>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
void EsperasSuspend(bool entering) noexcept {
	WaitMarkGameThread();
	auto&          e      = Esp();
	const int64_t  now    = NowNs();
	const uint64_t cycles = LocalPlatform::CurrentThreadCycles();
	if (e.cal_ns == 0) {
		e.cal_ns = now, e.cal_tsc = Tsc();
	} else if (e.tsc_per_ns == 0.0 && now - e.cal_ns > 1000000000ll) {
		e.tsc_per_ns = static_cast<double>(Tsc() - e.cal_tsc) / static_cast<double>(now - e.cal_ns);
	}
	if (entering) {
		e.enter_ns = now;
		if (e.last_exit_ns == 0) return;
		e.frames++;
		const int64_t build = now - e.last_exit_ns;
		if (build < g_esp_thr_ns) return;
		e.slow_frames++;
		if (!EspLineAllowed(e, now)) return;
		int64_t waits = 0;
		for (int64_t v: e.ns) waits += v;
		const double build_ms = static_cast<double>(build) / 1e6, waits_ms = static_cast<double>(waits) / 1e6;
		double       cpu_ms   = e.tsc_per_ns > 0.0 && cycles > e.last_exit_cycles && e.last_exit_cycles != 0
		                            ? static_cast<double>(cycles - e.last_exit_cycles) / e.tsc_per_ns / 1e6
		                            : -1.0;
		if (cpu_ms > build_ms * 1.1 + 1.0) cpu_ms = -1.0; // (esperas2: never print a CPU time longer than the frame)
		// (esperas2) the faults' part: count, time, the longest, the costliest sites
		char faults[400] = {};
		{
			EspSite fs[8];
			const uint32_t fc = e.fsite_count;
			for (uint32_t i = 0; i < fc; ++i) fs[i] = e.fsites[i];
			std::sort(fs, fs + fc, [](const EspSite& x, const EspSite& y) { return x.ns > y.ns; });
			size_t fu = 0;
			int    k  = std::snprintf(faults, sizeof(faults), "fallos de pagina %llu (%.1f ms, el mayor %.2f ms)",
			                          static_cast<unsigned long long>(e.fault_n), static_cast<double>(e.fault_ns) / 1e6,
			                          static_cast<double>(e.fault_max_ns) / 1e6);
			if (k > 0) fu = static_cast<size_t>(k);
			for (uint32_t i = 0; i < fc && i < 4 && fu < sizeof(faults); ++i) {
				char name[48];
				Site(name, sizeof(name), fs[i].ret);
				k = std::snprintf(faults + fu, sizeof(faults) - fu, "%s%s %.1f ms x%u", i == 0 ? " en " : ", ", name,
				                  static_cast<double>(fs[i].ns) / 1e6, fs[i].n);
				if (k <= 0) break;
				fu += static_cast<size_t>(k);
			}
		}
		const double fault_ms = static_cast<double>(e.fault_ns) / 1e6;
		// sites, longest first
		EspSite sorted[16];
		const uint32_t c = e.site_count;
		for (uint32_t i = 0; i < c; ++i) sorted[i] = e.sites[i];
		std::sort(sorted, sorted + c, [](const EspSite& x, const EspSite& y) { return x.ns > y.ns; });
		char   top[640] = {};
		size_t used     = 0;
		for (uint32_t i = 0; i < c && i < 6 && used < sizeof(top); ++i) {
			char name[48];
			Site(name, sizeof(name), sorted[i].ret);
			const int k = std::snprintf(top + used, sizeof(top) - used, "%s%s %s %.1f ms x%u", i == 0 ? "" : ", ", name,
			                            EspKind[sorted[i].kind], static_cast<double>(sorted[i].ns) / 1e6, sorted[i].n);
			if (k <= 0) break;
			used += static_cast<size_t>(k);
		}
		char kinds[320] = {};
		size_t ku       = 0;
		for (int k = 0; k < 9 && ku < sizeof(kinds); ++k) {
			if (e.n[k] == 0) continue;
			const int w = std::snprintf(kinds + ku, sizeof(kinds) - ku, "%s%s %.1f (%u)", ku == 0 ? "" : ", ", EspKind[k],
			                            static_cast<double>(e.ns[k]) / 1e6, e.n[k]);
			if (w <= 0) break;
			ku += static_cast<size_t>(w);
		}
		if (cpu_ms >= 0.0)
			std::printf("[esperas] hilo %u fotograma del juego %.1f ms (unix_ms %llu): CPU %.1f + esperas %.1f [%s] + ni CPU ni "
			            "espera %.1f (expulsado del nucleo, cerrojos sin gancho) | %s | sitios: %s\n",
			            e.idx, build_ms, UnixMs(), cpu_ms, waits_ms, ku == 0 ? "ninguna" : kinds,
			            std::max(0.0, build_ms - cpu_ms - waits_ms), faults, used == 0 ? "-" : top);
		else
			std::printf("[esperas] hilo %u fotograma del juego %.1f ms (unix_ms %llu): esperas %.1f [%s] (CPU sin dato) | %s | "
			            "sitios: %s\n",
			            e.idx, build_ms, UnixMs(), waits_ms, ku == 0 ? "ninguna" : kinds, faults, used == 0 ? "-" : top);
		(void)fault_ms;
		std::fflush(stdout);
		return;
	}
	// leaving the suspend point: the game's wait for the emulator (GuestGpu::Done)
	if (e.enter_ns != 0 && now - e.enter_ns >= g_esp_thr_ns && EspLineAllowed(e, now)) {
		std::printf("[esperas] el juego espero al emulador en sceAgcSuspendPoint %.1f ms (unix_ms %llu)\n",
		            static_cast<double>(now - e.enter_ns) / 1e6, UnixMs());
		std::fflush(stdout);
	}
	for (auto& v: e.ns) v = 0;
	for (auto& v: e.n) v = 0;
	e.site_count = 0, e.site_lost = 0;
	e.fault_n = 0, e.fault_ns = 0, e.fault_max_ns = 0, e.fsite_count = 0;
	e.last_exit_ns = now, e.last_exit_cycles = cycles;
}
} // namespace

bool WaitsEnabled() noexcept {
	const char* v = std::getenv("KYTY_ESPERAS");
	if (v == nullptr || v[0] == '\0' || v[0] == '0') return false;
	const long n = std::strtol(v, nullptr, 10);
	if (n > 1) g_esp_thr_ns = static_cast<int64_t>(n) * 1000000ll;
	(void)Esp();
	std::printf("[esperas] KYTY_ESPERAS=%s: cada fotograma del hilo del juego de %lld ms o mas -> una linea [esperas] (CPU, esperas "
	            "por tipo y sitio del eboot, lo demas). Solo medida.\n",
	            v, static_cast<long long>(g_esp_thr_ns / 1000000ll));
	std::fflush(stdout);
	return true;
}

bool WaitTimed() noexcept {
	// (esperas2) KYTY_ESPERAS: any thread with its own state (it calls sceAgcSuspendPoint); f0: the game thread
	if (WaitsOn() && t_esp != nullptr) return true;
	return g_s != nullptr && g_game_thread.load(std::memory_order_relaxed) == ThisThreadKey();
}
void EsperasFault(int64_t ns, uint64_t rip) noexcept {
	if (t_esp != nullptr) EsperasFaultNote(ns, rip);
}
int64_t WaitNowNsFault() noexcept {
	return t_esp != nullptr ? NowNs() : 0;
}
int64_t WaitNowNs() noexcept {
	return NowNs();
}
void WaitNote(int kind, int64_t ns, const void* ret) noexcept {
	if (WaitsOn()) EsperasNote(kind, ns, ret); // (esperas)
	if (g_s == nullptr) return;                // (f0 off: only esperas)
	if (g_game_thread.load(std::memory_order_relaxed) != ThisThreadKey()) return; // (esperas2: f0 counts the game thread only)
	// (waits under 5 us are an uncontended lock's own work: the thread's CPU time, not counted here; also keeps the cost low)
	if (kind < 0 || kind > 7 || ns < 5000) return;
	auto&           w = Acc();
	std::lock_guard lock(w.mutex);
	w.ns[kind] += static_cast<double>(ns);
	w.count[kind]++;
	const auto a = reinterpret_cast<uintptr_t>(ret);
	for (auto& s: w.sites)
		if (s.first == a) {
			s.second += static_cast<double>(ns);
			return;
		}
	if (w.sites.size() < 64) w.sites.emplace_back(a, static_cast<double>(ns));
}

void GameSubmit(bool compute_queue, uint32_t dwords) {
	if (!On() || g_s == nullptr) return;
	g_s->game_submits.fetch_add(1, std::memory_order_relaxed);
	if (compute_queue) g_s->game_acb_submits.fetch_add(1, std::memory_order_relaxed);
	g_s->game_dwords.fetch_add(dwords, std::memory_order_relaxed);
}

void GameSuspend(bool entering) {
	if (WaitsOn()) EsperasSuspend(entering); // (esperas, KYTY_ESPERAS)
	if (!On() || g_s == nullptr) return;
	WaitMarkGameThread(); // (cpu1)
	const uint64_t t0  = Tsc();
	auto&          s   = *g_s;
	const int64_t  now = NowNs();
	if (entering) {
		s.game_enter.store(now, std::memory_order_relaxed);
		return;
	}
	const int64_t  enter  = s.game_enter.load(std::memory_order_relaxed);
	const int64_t  last   = s.game_last_exit.exchange(now, std::memory_order_relaxed);
	const uint64_t cycles = LocalPlatform::CurrentThreadCycles();
	const uint64_t lastc  = s.game_cycles_exit.exchange(cycles, std::memory_order_relaxed);
	if (last != 0 && enter > last) {
		GameFrame f;
		f.build_ms = static_cast<double>(enter - last) / 1e6;
		f.wait_ms  = static_cast<double>(now - enter) / 1e6;
		f.cpu_ms   = cycles > lastc && lastc != 0 ? static_cast<double>(cycles - lastc) / s.tsc_per_ns.load() / 1e6 : 0.0;
		WaitFrame(f.build_ms, f.cpu_ms, now); // (cpu1)
		f.submits  = static_cast<double>(s.game_submits.exchange(0, std::memory_order_relaxed));
		std::lock_guard lock(s.game_mutex);
		if (s.window_game.size() < 16384) s.window_game.push_back(f);
	}
	s.cost_game_cycles.fetch_add(Tsc() - t0, std::memory_order_relaxed);
}

void SetGate(void (*lock)(), void (*unlock)()) {
	if (!On() || g_s == nullptr) return;
	g_s->gate_unlock = unlock;
	g_s->gate_lock   = lock;
	std::printf("[f0] memoria nativa: muestreo de lecturas de CPU activo (PAGE_GUARD con la puerta de proteccion)\n");
}

void EnsureSlow(uint64_t vaddr, uint64_t size) {
#if defined(_WIN32)
	if (g_s == nullptr || g_s->armed.load(std::memory_order_acquire) == 0 || size == 0) return;
	const uint64_t lo = vaddr & ~(Page - 1u), hi = vaddr + size;
	for (auto& p: g_s->probes) {
		if (p.state.load(std::memory_order_acquire) != 2) continue;
		const uint64_t page = p.page.load(std::memory_order_relaxed);
		if (page >= lo && page < hi) RetireProbe(p, true);
	}
#else
	(void)vaddr;
	(void)size;
#endif
}

} // namespace F0Measure
