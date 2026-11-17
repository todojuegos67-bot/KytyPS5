#pragma once
// (envio1) KYTY_ENVIO1=1 (off by default; only "1" turns it on): a diagnostic, no change of behaviour or image. Of the
// time the thread translating the compute queues (ACB) spends in their submissions, how much falls in the operations
// that today force it to run in series with the graphics queue (envio1/INFORME.md section 5):
//   buf-miss  BufferCache registration, merge, deletion (CreateBuffer/ResolveOverlaps/JoinOverlap/DeleteBuffer)
//   buf-mod   BufferCache modification by a shader write (GPU-modified ranges, feedback, write notes)
//   buf-hit   the rest of ObtainBuffer: the read-only hit (NOT serial: it would take a shared lock)
//   tex-new   TextureCache image creation (InsertImage)
//   tex-lay   image layout transitions that record a barrier (Image::Transit)
//   dirty     uploads of dirty data (BufferCache::UploadDirtyRanges, the small CPU-written ring copies, Image::Upload)
//   lock      taking the global render lock (renderCompute.cpp DsLinearCopy / DispatchDirect)
//   record    recording into the shared command buffer (bindings commit, barriers, bind and dispatch)
// Time is exclusive: a category nested in another counts in the inner one only. A "[envio1]" line every 5 s.
// Off: one predictable branch per scope (a static bool), no clock read.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Envio1 {

enum Category : uint32_t { BufMiss, BufMod, BufHit, TexNew, TexLayout, Dirty, Lock, Record, Count };

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_ENVIO1");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Totals {
	std::atomic<uint64_t> acb_ns {0}, acb_submissions {0}, dispatches {0}, frames {0};
	std::array<std::atomic<uint64_t>, Count> ns {}, events {};
};
inline Totals g_totals;

// The calling thread's: inside a compute queue submission, and the open category scopes.
struct ThreadState {
	uint32_t                     acb_depth = 0;
	uint32_t                     depth     = 0;
	std::array<uint32_t, 16>     stack {};
	uint64_t                     mark = 0; // when the innermost open scope (re)started counting
	std::array<uint64_t, Count>  ns {}, events {};
	uint64_t                     dispatches = 0;
};
inline thread_local ThreadState t_state;

inline bool Active() {
	return On() && t_state.acb_depth != 0;
}

// A compute queue submission (GuestGpu::Process): its time, and the categories' counted into the totals at its end.
class AcbScope {
public:
	explicit AcbScope(bool acb) {
		if (!acb || !On()) return;
		m_on = true;
		if (t_state.acb_depth++ == 0) m_start = NowNs();
	}
	~AcbScope() {
		if (!m_on || --t_state.acb_depth != 0) return;
		auto& t = t_state;
		g_totals.acb_ns.fetch_add(NowNs() - m_start, std::memory_order_relaxed);
		g_totals.acb_submissions.fetch_add(1, std::memory_order_relaxed);
		g_totals.dispatches.fetch_add(t.dispatches, std::memory_order_relaxed);
		t.dispatches = 0;
		for (uint32_t c = 0; c < Count; ++c) {
			if (t.ns[c] != 0) g_totals.ns[c].fetch_add(t.ns[c], std::memory_order_relaxed);
			if (t.events[c] != 0) g_totals.events[c].fetch_add(t.events[c], std::memory_order_relaxed);
			t.ns[c] = t.events[c] = 0;
		}
	}
	AcbScope(const AcbScope&)            = delete;
	AcbScope& operator=(const AcbScope&) = delete;

private:
	bool     m_on    = false;
	uint64_t m_start = 0;
};

inline void NoteDispatch() {
	if (Active()) ++t_state.dispatches;
}

// A category's time, exclusive of the scopes opened inside it. Drop(): the scope turned out not to be the category
// (a transition with no barrier): its time goes to the enclosing scope and it is not an event.
class Scope {
public:
	explicit Scope(Category category) {
		if (!Active() || t_state.depth >= t_state.stack.size()) return;
		auto&      t   = t_state;
		const auto now = NowNs();
		if (t.depth != 0) t.ns[t.stack[t.depth - 1]] += now - t.mark;
		t.stack[t.depth++] = category;
		t.mark             = now;
		m_on               = true;
		m_category         = category;
	}
	void Drop() { m_dropped = true; }
	~Scope() {
		if (!m_on) return;
		auto&      t   = t_state;
		const auto now = NowNs();
		--t.depth;
		const auto elapsed = now - t.mark;
		if (m_dropped && t.depth != 0) t.ns[t.stack[t.depth - 1]] += elapsed;
		else if (!m_dropped) {
			t.ns[m_category] += elapsed;
			++t.events[m_category];
		}
		t.mark = now;
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool     m_on       = false;
	bool     m_dropped  = false;
	Category m_category = Count;
};

// At each frame boundary (GPU thread): a line every 5 s.
inline void NoteFrame() {
	if (!On()) return;
	g_totals.frames.fetch_add(1, std::memory_order_relaxed);
	static uint64_t last_ns = NowNs();
	const auto      now     = NowNs();
	if (now - last_ns < 5'000'000'000ull) return;
	last_ns = now;
	static const char* const names[Count] = {"buf-miss", "buf-mod", "buf-hit", "tex-new", "tex-lay", "dirty", "lock", "record"};
	auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const uint64_t frames   = std::max<uint64_t>(take(g_totals.frames), 1);
	const uint64_t acb_ns   = take(g_totals.acb_ns);
	const uint64_t subs     = take(g_totals.acb_submissions);
	const uint64_t dispatch = take(g_totals.dispatches);
	std::array<uint64_t, Count> ns {}, ev {};
	for (uint32_t c = 0; c < Count; ++c) {
		ns[c] = take(g_totals.ns[c]);
		ev[c] = take(g_totals.events[c]);
	}
	const double f      = static_cast<double>(frames);
	const double acb_ms = static_cast<double>(acb_ns) / 1e6 / f;
	auto         pct    = [&](uint64_t x) { return acb_ns != 0 ? 100.0 * static_cast<double>(x) / static_cast<double>(acb_ns) : 0.0; };
	// Serial: everything but the read-only hit.
	uint64_t serial = 0;
	for (uint32_t c = 0; c < Count; ++c)
		if (c != BufHit) serial += ns[c];
	std::printf("[envio1] %llu frames: ACB %.2f ms/frame (%.1f submissions, %.1f dispatches a frame);",
	            static_cast<unsigned long long>(frames), acb_ms, static_cast<double>(subs) / f,
	            static_cast<double>(dispatch) / f);
	for (uint32_t c = 0; c < Count; ++c)
		std::printf(" %s %.2f ms %.1f%% (%.1f/frame);", names[c], static_cast<double>(ns[c]) / 1e6 / f, pct(ns[c]),
		            static_cast<double>(ev[c]) / f);
	const double serial_pct = pct(serial);
	std::printf(" serial %.2f ms %.1f%% -> %s\n", static_cast<double>(serial) / 1e6 / f, serial_pct,
	            acb_ns == 0 ? "sin ACB" : serial_pct > 15.0 ? "no compensa" : "compensa");
	std::fflush(stdout);
}

} // namespace Envio1
