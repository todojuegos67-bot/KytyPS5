#pragma once
// (envio2) KYTY_ENVIO2 (off by default). 1: the compute queues' (ACB) submissions are translated on their own thread
// (Kyty.Acb2) while the GPU thread goes on with the graphics queue (DCB); the GPU thread commits each one, in guest order,
// into the one Vulkan queue (Speculation's machinery, speculation.h: the ACB thread records into its own command buffers
// with a Recorder and pending ticks, reads the caches without changing them and refuses what would; the commit makes the
// buffers current / GPU-owned, the images' layouts, the labels (EOP / RELEASE_MEM / WRITE_DATA) and the real ticks, and
// checks that nothing the translation assumed changed since; what it refused or what changed is translated the normal
// way, on the GPU thread, from there: the safe fallback). 2: the same translation and commit, but the GPU thread waits
// for it right away (no overlap, the order of today): a verification mode, see envio2/INFORME.md.
// A "[envio2]" line every 5 s (with KYTY_ENVIO2=1 or 2). Off: one static bool.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Envio2 {

inline uint32_t Mode() {
	static const uint32_t mode = [] {
		// (envio3: KYTY_ENVIO3=1/2 implies KYTY_ENVIO2 in the same mode, local/envio3.h.)
		const char* value3 = std::getenv("KYTY_ENVIO3");
		if (value3 != nullptr && (value3[0] == '1' || value3[0] == '2') && value3[1] == '\0') return value3[0] == '1' ? 1u : 2u;
		const char* value = std::getenv("KYTY_ENVIO2");
		if (value == nullptr || value[1] != '\0') return 0u;
		return value[0] == '1' ? 1u : value[0] == '2' ? 2u : 0u;
	}();
	return mode;
}

inline uint64_t NowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Why a compute submission was not (wholly) taken by the ACB thread.
enum Fallback : uint32_t {
	Busy,      // the ACB thread had another one (in flight, or a commit blocked at a wait)
	NotFirst,  // a slice resumed after a wait the normal way stopped at
	None,      // committed nothing (a wait not satisfied at its start, a refusal before any work, a cache changed)
	Partly,    // committed up to a refusal / change, the rest the normal way
	Blocked,   // the commit waits at a wait hole (counted per time)
	FallbackCount
};

struct Totals {
	std::atomic<uint64_t> frames {0};
	std::atomic<uint64_t> gpu_acb_ns {0};  // the GPU thread's time in compute submissions (hand-off, commit, fallback)
	std::atomic<uint64_t> acb2_ns {0};     // the ACB thread's translation time
	std::atomic<uint64_t> submissions {0}; // compute submissions (first slices)
	std::atomic<uint64_t> handed {0};      // handed to the ACB thread
	std::atomic<uint64_t> whole {0};       // committed whole (in parallel all the way)
	std::atomic<uint64_t> forced {0};      // commits that waited for the ACB thread (it was still translating)
	std::atomic<uint64_t> forced_ns {0};   // that wait
	std::atomic<uint64_t> order_errors {0}; // mode 2's check: a queue's commits out of serial order
	std::array<std::atomic<uint64_t>, FallbackCount> fallback {};
};
inline Totals g_totals;

inline void Add(std::atomic<uint64_t>& counter, uint64_t value = 1) {
	counter.fetch_add(value, std::memory_order_relaxed);
}

// The GPU thread's time in a compute submission.
class GpuScope {
public:
	explicit GpuScope(bool acb): m_on(acb && Mode() != 0), m_start(m_on ? NowNs() : 0) {}
	~GpuScope() {
		if (m_on) Add(g_totals.gpu_acb_ns, NowNs() - m_start);
	}
	GpuScope(const GpuScope&)            = delete;
	GpuScope& operator=(const GpuScope&) = delete;

private:
	bool     m_on;
	uint64_t m_start;
};

// At each frame boundary (GPU thread): a line every 5 s (returns the frames it covered, 0: no line; envio3's).
inline double NoteFrame() {
	if (Mode() == 0) return 0;
	Add(g_totals.frames);
	static uint64_t last_ns = NowNs();
	const auto      now     = NowNs();
	if (now - last_ns < 5'000'000'000ull) return 0;
	last_ns   = now;
	auto take = [](std::atomic<uint64_t>& a) { return a.exchange(0, std::memory_order_relaxed); };
	const double f        = static_cast<double>(std::max<uint64_t>(take(g_totals.frames), 1));
	const auto   ms       = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / f; };
	const auto   per      = [&](uint64_t n) { return static_cast<double>(n) / f; };
	const auto   gpu_ns   = take(g_totals.gpu_acb_ns);
	const auto   acb2_ns  = take(g_totals.acb2_ns);
	const auto   subs     = take(g_totals.submissions);
	const auto   handed   = take(g_totals.handed);
	const auto   whole    = take(g_totals.whole);
	const auto   forced   = take(g_totals.forced);
	const auto   forced_n = take(g_totals.forced_ns);
	const auto   order    = take(g_totals.order_errors);
	std::array<uint64_t, FallbackCount> fb {};
	for (uint32_t i = 0; i < FallbackCount; ++i) fb[i] = take(g_totals.fallback[i]);
	std::printf("[envio2] mode %u: hilo GPU (ACB) %.2f ms/frame, hilo Acb2 %.2f ms/frame; submissions %.1f/frame: "
	            "paralelo %.1f (enteras %.1f), fallback: ocupado %.1f, reanudada %.1f, nada %.1f, parcial %.1f, "
	            "bloqueada %.1f; envios forzados %.1f/frame (espera %.2f ms/frame); esperas de candado 0 (sin candados "
	            "nuevos); orden %s\n",
	            Mode(), ms(gpu_ns), ms(acb2_ns), per(subs), per(handed), per(whole), per(fb[Busy]), per(fb[NotFirst]),
	            per(fb[None]), per(fb[Partly]), per(fb[Blocked]), per(forced), ms(forced_n),
	            order == 0 ? "ok" : "MAL");
	std::fflush(stdout);
	return f;
}

} // namespace Envio2
