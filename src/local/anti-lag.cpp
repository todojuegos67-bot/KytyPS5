#include "local/anti-lag.h"

#include "local/frame-gen.h"
#include "local/fsr3-fg.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::Graphics::AntiLag {

// VK_AMD_anti_lag, declared here (older Vulkan headers do not have it; the values are the registry's).
namespace {
struct OwnState {
	bool     on     = false;
	uint32_t queue  = 2;
	bool     pace   = false; // KYTY_LOWLAT_PACE_BEFORE_INPUT (default on with KYTY_LOWLAT=1)
	double   margin = 4.0;   // KYTY_LOWLAT_PACE_MARGIN_MS
	OwnState() {
		const char* v = std::getenv("KYTY_LOWLAT");
		on            = v != nullptr && v[0] == '1';
		if (const char* q = std::getenv("KYTY_LOWLAT_QUEUE"); q != nullptr && q[0] == '1') queue = 1;
		const char* p = std::getenv("KYTY_LOWLAT_PACE_BEFORE_INPUT");
		pace          = on && (p == nullptr || p[0] != '0');
		if (const char* m = std::getenv("KYTY_LOWLAT_PACE_MARGIN_MS"); m != nullptr) margin = std::clamp(std::atof(m), 0.0, 20.0);
		if (on) {
			std::printf("Low latency (own): on, max %u frames queued at the controller read; pace before input %s "
			            "(KYTY_LOWLAT_PACE_BEFORE_INPUT, margin %.1f ms)\n",
			            queue, pace ? "on" : "off", margin);
			std::fflush(stdout);
		}
	}
};
std::atomic<PFN_vkWaitSemaphores> g_own_wait {nullptr};
const OwnState& Own() {
	static const OwnState state;
	return state;
}
}
namespace {
constexpr VkStructureType kFeaturesType     = static_cast<VkStructureType>(1000476000); // ..._ANTI_LAG_FEATURES_AMD
constexpr VkStructureType kDataType         = static_cast<VkStructureType>(1000476001); // ..._ANTI_LAG_DATA_AMD
constexpr VkStructureType kPresentationType = static_cast<VkStructureType>(1000476002); // ..._PRESENTATION_INFO_AMD
constexpr int32_t         kModeOn           = 1;                                         // VK_ANTI_LAG_MODE_ON_AMD
constexpr int32_t         kStageInput       = 0;                                         // VK_ANTI_LAG_STAGE_INPUT_AMD
constexpr int32_t         kStagePresent     = 1;                                         // VK_ANTI_LAG_STAGE_PRESENT_AMD

struct FeaturesAmd {
	VkStructureType sType   = kFeaturesType;
	void*           pNext   = nullptr;
	VkBool32        antiLag = VK_FALSE;
};
struct PresentationInfoAmd {
	VkStructureType sType      = kPresentationType;
	void*           pNext      = nullptr;
	int32_t         stage      = kStageInput;
	uint64_t        frameIndex = 0;
};
struct DataAmd {
	VkStructureType            sType             = kDataType;
	const void*                pNext             = nullptr;
	int32_t                    mode              = kModeOn;
	uint32_t                   maxFPS            = 0;
	const PresentationInfoAmd* pPresentationInfo = nullptr;
};
using UpdateFn = void(VKAPI_PTR*)(VkDevice, const DataAmd*);

enum class Mode { Off, Amd, Limiter, Reflex };

std::atomic<Mode>     g_mode {Mode::Off};
VkDevice              g_device = VK_NULL_HANDLE;
UpdateFn              g_update = nullptr;
FeaturesAmd           g_features {};
std::atomic<uint64_t> g_input_index {0};   // the last frame index marked INPUT
std::atomic<uint64_t> g_tagged_index {0};  // the last index a flip took
std::atomic<uint64_t> g_flips {0};         // flips reserved
std::atomic<uint64_t> g_flips_at_input {~uint64_t {0}};
std::atomic<uint64_t> g_presented {0};     // the last index presented
std::mutex            g_mutex;
// NVIDIA Reflex (VK_NV_low_latency2): the window's swapchain (not FSR 3's: its real swapchain is AMD's), under the lock.
PFN_vkSetLatencySleepModeNV g_sleep_mode = nullptr;
PFN_vkLatencySleepNV        g_sleep      = nullptr;
PFN_vkSetLatencyMarkerNV    g_marker     = nullptr;
PFN_vkWaitSemaphores        g_wait       = nullptr;
std::mutex                  g_reflex_mutex;
VkSwapchainKHR              g_swapchain = VK_NULL_HANDLE;
VkSemaphore                 g_semaphore = VK_NULL_HANDLE;
uint64_t                    g_semaphore_value = 0;
VkSwapchainLatencyCreateInfoNV g_latency_create {};
std::atomic<bool>           g_fsr3_logged {false};
bool                        g_fsr3_real = false; // g_swapchain is FSR 3's real one: the DLL marks its presents

// The patched amd_fidelityfx_vk.dll (tools/local/fsr3-mfg): the driver swapchain behind its frame interpolation one.
using RealSwapchainFn = VkSwapchainKHR (*)(VkSwapchainKHR);
RealSwapchainFn Fsr3RealSwapchain() {
#if defined(_WIN32)
	HMODULE module = GetModuleHandleA("amd_fidelityfx_vk.dll");
	if (module == nullptr) return nullptr;
	return reinterpret_cast<RealSwapchainFn>(GetProcAddress(module, "kytyFfxGetRealSwapchainVK"));
#else
	return nullptr;
#endif
}

void ReflexMarker(VkLatencyMarkerNV marker, uint64_t index) {
	std::lock_guard lock(g_reflex_mutex);
	if (g_swapchain == VK_NULL_HANDLE || index == 0) return;
	VkSetLatencyMarkerInfoNV info {};
	info.sType     = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV;
	info.presentID = index;
	info.marker    = marker;
	g_marker(g_device, g_swapchain, &info);
}
std::condition_variable g_cond;

void Mark(int32_t stage, uint64_t index) {
	PresentationInfoAmd info {};
	info.stage      = stage;
	info.frameIndex = index;
	DataAmd data {};
	data.pPresentationInfo = &info;
	g_update(g_device, &data);
}
} // namespace

bool PhaseOn() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PHASE_PACE");
		const bool  r = v != nullptr ? v[0] == '1' : Own().on;
		std::printf("Phase pacing: %s (KYTY_PHASE_PACE: each game step starts at a fixed phase of the flip grid, the queue "
		            "drained 1 ms a step; under a flip rate or cap only)\n",
		            r ? "on" : "off");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

// KYTY_PAD_LATE_MS (anti-lag.h): L ms before the frame's flip slot, 0 off.
static double PadLateMs() {
	static const double ms = [] {
		const char* v = std::getenv("KYTY_PAD_LATE_MS");
		const double r = v != nullptr ? std::atof(v) : 0.0;
		const double c = r > 0.0 ? std::clamp(r, 10.0, 80.0) : 0.0;
		if (c > 0.0) {
			std::printf("Pad late: the controller read waits until %.1f ms before its frame's flip slot (KYTY_PAD_LATE_MS; "
			            "under a 20-45 ms flip limit only)\n", c);
			std::fflush(stdout);
		}
		return c;
	}();
	return ms;
}

bool PadLateOn() {
	return PadLateMs() > 0.0;
}

// (10-26 latq) KYTY_PAD_LATE_AUTO (default on with KYTY_PAD_LATE_MS; =0 off): the read's lead L follows the frame's
// own input -> gpu done (p95 of the last 64 paced frames + a margin that grows 2 ms on each miss the wait caused and
// decays back), KYTY_PAD_LATE_MS the ceiling, KYTY_PAD_LATE_MIN_MS (default 14) the floor. The fixed 43 / 29 ms of the
// launcher sat ~20 ms above what a 5080 frame needs: that idle time was pure latency.
static bool PadLateAuto() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_PAD_LATE_AUTO");
		const bool  r = PadLateMs() > 0.0 && (v == nullptr || v[0] != '0');
		if (PadLateMs() > 0.0) {
			std::printf("Pad late auto: %s (KYTY_PAD_LATE_AUTO; L = p95 input->gpu done + margin, at most %.1f ms)\n",
			            r ? "on" : "off", PadLateMs());
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}
static double PadLateMinMs() {
	static const double ms = [] {
		const char* v = std::getenv("KYTY_PAD_LATE_MIN_MS");
		return std::clamp(v != nullptr ? std::atof(v) : 14.0, 6.0, 80.0);
	}();
	return ms;
}

bool PaceOn() {
	static const bool on = [] {
		const char* off = std::getenv("KYTY_LOWLAT_PACE_BEFORE_INPUT");
		const char* q   = std::getenv("KYTY_FLIP_QUEUE_LIMIT");
		// (Off by default since the constant-phase pacer: its event wait made the game's steps uneven, 212307.)
		const bool  r   = !(off != nullptr && off[0] == '0') && q != nullptr && q[0] == '1' && !PhaseOn();
		std::printf("Flip queue limit: %s (the game's controller read waits for frame N - 1's release under a flip rate or "
		            "cap; KYTY_FLIP_QUEUE_LIMIT=1 on, off with KYTY_PHASE_PACE)\n",
		            r ? "on" : "off");
		std::fflush(stdout);
		return r;
	}();
	return on;
}

bool AutoReflex() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_LOWLAT_REFLEX");
		return Own().on && Fsr3::Requested() && v != nullptr && v[0] == '1'; // (opt-in: the phase pacer throttles)
	}();
	return on;
}

bool Requested() {
	static const bool requested = [] {
		const char* v = std::getenv("KYTY_ANTILAG");
		return v != nullptr && (v[0] == '1' || v[0] == '2');
	}();
	return requested;
}

const char* ExtensionName() {
	return "VK_AMD_anti_lag";
}

void* ChainDeviceFeatures(bool enabled, void* next) {
	if (!enabled) return next;
	g_features         = {};
	g_features.antiLag = VK_TRUE;
	g_features.pNext   = next;
	return &g_features;
}

void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc, bool extension_enabled, bool reflex_enabled) {
	static bool logged = false;
	if (Own().on && get_proc != nullptr) // KYTY_LOWLAT: core 1.2 (the emulator's timelines need it)
		g_own_wait.store(reinterpret_cast<PFN_vkWaitSemaphores>(get_proc(static_cast<VkDevice>(device), "vkWaitSemaphores")),
		                 std::memory_order_release);
	Mode        mode   = Mode::Off;
	const char* text   = nullptr;
	const bool auto_reflex = !Requested() && AutoReflex(); // (NVIDIA Reflex only: AMD / Intel keep our pacer)
	if (!Requested() && !auto_reflex) {
		text = "Anti-Lag: off (KYTY_ANTILAG=0)";
	} else if (FrameGen::StreamlineLoaded()) {
		text = "Anti-Lag: off (NVIDIA Streamline loaded: Reflex is active)";
	} else {
		if (extension_enabled && get_proc != nullptr && !auto_reflex)
			g_update = reinterpret_cast<UpdateFn>(get_proc(static_cast<VkDevice>(device), "vkAntiLagUpdateAMD"));
		if (reflex_enabled && get_proc != nullptr) {
			const auto dev = static_cast<VkDevice>(device);
			g_sleep_mode = reinterpret_cast<PFN_vkSetLatencySleepModeNV>(get_proc(dev, "vkSetLatencySleepModeNV"));
			g_sleep      = reinterpret_cast<PFN_vkLatencySleepNV>(get_proc(dev, "vkLatencySleepNV"));
			g_marker     = reinterpret_cast<PFN_vkSetLatencyMarkerNV>(get_proc(dev, "vkSetLatencyMarkerNV"));
			g_wait       = reinterpret_cast<PFN_vkWaitSemaphores>(get_proc(dev, "vkWaitSemaphores"));
			auto create_semaphore = reinterpret_cast<PFN_vkCreateSemaphore>(get_proc(dev, "vkCreateSemaphore"));
			if (g_sleep_mode != nullptr && g_sleep != nullptr && g_marker != nullptr && g_wait != nullptr &&
			    create_semaphore != nullptr) {
				VkSemaphoreTypeCreateInfo type {};
				type.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
				type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
				VkSemaphoreCreateInfo info {};
				info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
				info.pNext = &type;
				if (create_semaphore(dev, &info, nullptr, &g_semaphore) != VK_SUCCESS) g_semaphore = VK_NULL_HANDLE;
			}
		}
		if (g_update == nullptr && g_semaphore != VK_NULL_HANDLE) {
			g_device = static_cast<VkDevice>(device);
			mode     = Mode::Reflex;
			text     = auto_reflex ? "Anti-Lag: NVIDIA Reflex for FSR 3 frame generation (KYTY_LOWLAT; VK_NV_low_latency2, low latency on, "
			                         "boost off, no frame limit; our pacer only without it)"
			                       : "Anti-Lag: NVIDIA Reflex (VK_NV_low_latency2; low latency on, boost off, no frame limit)";
		} else if (g_update != nullptr) {
			g_device = static_cast<VkDevice>(device);
			mode     = Mode::Amd;
			text     = "Anti-Lag: AMD Anti-Lag 2 (VK_AMD_anti_lag)";
		} else if (auto_reflex) {
			text = "Anti-Lag: no NVIDIA Reflex here (KYTY_LOWLAT: our own pacer)";
		} else if (const char* v = std::getenv("KYTY_ANTILAG"); v == nullptr || v[0] != '2') {
			// (The limiter's wait in the controller read corrupted the picture in a test on NVIDIA: opt-in only.)
			text = "Anti-Lag: off (neither VK_AMD_anti_lag nor VK_NV_low_latency2; KYTY_ANTILAG=2 for the experimental "
			       "emulator latency limiter)";
		} else {
			mode = Mode::Limiter;
			text = "Anti-Lag: emulator latency limiter (no VK_AMD_anti_lag)";
		}
	}
	g_mode.store(mode, std::memory_order_release);
	if (!logged) {
		logged = true;
		std::printf("%s\n", text);
		std::fflush(stdout);
	}
}

namespace {
constexpr uint64_t           kOwnRing = 16;
std::atomic<VkDevice>        g_own_device {VK_NULL_HANDLE};
std::atomic<VkSemaphore>     g_own_timeline {VK_NULL_HANDLE};
std::atomic<uint64_t>        g_own_ticks[kOwnRing] {};
std::atomic<uint64_t>        g_own_frames {0}; // guest frames recorded
std::atomic<uint64_t>        g_own_waited {0}; // the frame count the last controller read saw

// The game's thread at its controller read, once per new frame: the GPU work of frame (frames - queue + 1) must be
// complete. Only a host GPU timeline is waited for (never a present, swapchain or a lock), with a 50 ms cap.
void OwnLimit() {
	const uint64_t frames = g_own_frames.load(std::memory_order_acquire);
	uint64_t       seen   = g_own_waited.load(std::memory_order_relaxed);
	if (frames == seen || !g_own_waited.compare_exchange_strong(seen, frames, std::memory_order_acq_rel)) return;
	const uint64_t queue = Own().queue;
	if (frames < queue) return;
	const uint64_t frame = frames - queue; // 0-based: frames - 1 is the newest
	if (frames - frame > kOwnRing) return;
	const uint64_t tick     = g_own_ticks[frame % kOwnRing].load(std::memory_order_acquire);
	VkDevice       device   = g_own_device.load(std::memory_order_acquire);
	VkSemaphore    timeline = g_own_timeline.load(std::memory_order_acquire);
	auto           wait     = g_own_wait.load(std::memory_order_acquire);
	if (tick == 0 || device == VK_NULL_HANDLE || timeline == VK_NULL_HANDLE || wait == nullptr) return;
	VkSemaphoreWaitInfo info {};
	info.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
	info.semaphoreCount = 1;
	info.pSemaphores    = &timeline;
	info.pValues        = &tick;
	(void)wait(device, &info, 50'000'000ull); // (timeout / device lost: the game simply goes on)
}

// Per frame index stage stamps (LatNow ticks), for the pacing and the [lat] stages line.
int64_t LatNow();
double  LatTicksPerMs();
struct StageStamp {
	std::atomic<uint64_t> index {0};
	std::atomic<int64_t>  input {0}, reserve {0}, ready {0}, release {0};
	std::atomic<int64_t>  slot {0};      // KYTY_PAD_LATE_MS: the flip slot the read was placed against (0: not paced)
	std::atomic<double>   slept {0.0};   // KYTY_PAD_LATE_MS: ms slept before the read
};
constexpr uint64_t      kStageRing = 64;
StageStamp              g_stage[kStageRing];
std::mutex              g_pace_mutex;
std::condition_variable g_pace_cond;
uint64_t                g_rel_index = 0;   // the last released (flipped) index, under g_pace_mutex
int64_t                 g_rel_ticks = 0;
double                  g_pace_iv   = 0.0; // smoothed release interval, ms
double                  g_pace_lead = 0.0; // smoothed input -> GPU done, ms
std::atomic<double>     g_pace_slept_sum {0.0};
std::atomic<uint32_t>   g_pace_slept_n {0};
std::atomic<int>        g_pace_k {0};
std::atomic<double>     g_pace_drain_sum {0.0};
std::atomic<uint32_t>   g_pace_drain_n {0};

StageStamp* StageFor(uint64_t index) {
	if (index == 0) return nullptr;
	auto& s = g_stage[index % kStageRing];
	return s.index.load(std::memory_order_acquire) == index ? &s : nullptr;
}

// Reflex-style sleep before the input sample: the cap (KYTY_FLIP_FPS, a whole number of vblanks) releases a game-view
// flip at most every `cap` ms; a game faster than that ran ahead and its frames waited in the flip queue (2 frames,
// ~67 ms at 30 fps). Here, at the first controller read of frame `index`, the game's thread sleeps until
// (release of frame index - 1) + cap - (lead + margin), lead = the measured input -> GPU done.
//  - The interval is the configured cap (videoOut's floor for this frame, SetPaceCap), never a measured one (that
//    drifted to 39 ms: every late frame made the next sleep longer, 20261024-072608).
//  - Only own low latency: never with Reflex / Anti-Lag 2 / the limiter, Streamline (DLSS-G) or our frame generation,
//    and only while game-view frames under a cap are being flipped (no movie, menu, cinematic, loading screen).
//  - A late frame (released over half a vblank after its slot) grows the margin by 2 ms (max 16) and turns the sleep
//    off for the next 8 frames; the margin comes back down by 0.05 ms per frame on time (no fps lost to the pacer).
struct PaceCap {
	double   cap_ms    = 0.0; // the configured interval of the frame just released (0: no pacing)
	double   vblank_ms = 0.0;
	double   margin    = 0.0; // the adaptive margin, ms (from Own().margin)
	uint32_t skip      = 0;   // frames left without a sleep after a late one
	uint32_t misses    = 0;   // late frames this window (the [lat] line)
};
PaceCap g_pace; // under g_pace_mutex

// Every mode but DLSS-G (Streamline's frame generation paces itself): 202407 (DLSS SR, Streamline Reflex on, the game's
// own 30 fps flip rate) kept two frames between GPU done and flip (65.9 ms) like our 30 cap with FSR 3: Reflex's sleep
// does not see a queue in front of a flip rate. The wait only ever holds the game for frame N - k's release.
bool PaceAllowed() {
	return PaceOn() && !FrameGen::Enabled();
}

const char*       g_pace_site = "scePadRead";
std::atomic<bool> g_pace_logged {false};
void PaceLogOnce(const char* what, double ms) {
	if (g_pace_logged.exchange(true)) return;
#if defined(_WIN32)
	const unsigned long tid = GetCurrentThreadId();
#else
	const unsigned long tid = static_cast<unsigned long>(std::hash<std::thread::id> {}(std::this_thread::get_id()));
#endif
	std::printf("[pace] first %s (%.1f ms) on thread %lu at %s (the game's controller read)\n", what, ms, tid, g_pace_site);
	std::fflush(stdout);
}

// The last 64 input -> GPU done samples (the game's frame lead): k = 2 (one frame queued) while their p99 exceeds the
// cap minus a vblank, back to k = 1 (none queued) after 60 frames under it. Modelled on 195625 (tools sim depth.py):
// a slot sleep on top (v3) made frames late and the back-off then turned the pacer off on most frames.
double   g_lead_ring[64] {};
uint32_t g_lead_n = 0, g_k_hold = 0;
int      g_k = 1; // under g_pace_mutex

int PaceDepthMode() { // KYTY_LOWLAT_PACE_DEPTH: auto (default), 0 (never a frame queued), 1 (always one)
	static const int mode = [] {
		const char* v = std::getenv("KYTY_LOWLAT_PACE_DEPTH");
		return v == nullptr ? -1 : (v[0] == '0' ? 0 : (v[0] == '1' ? 1 : -1));
	}();
	return mode;
}

void PaceBeforeInput(uint64_t index) {
	if (!PaceAllowed()) return;
	const double per_ms = LatTicksPerMs();
	std::unique_lock lock(g_pace_mutex);
	const double cap = g_pace.cap_ms;
	g_pace_iv        = cap;
	if (cap < 4.0 || cap > 70.0 || g_rel_index == 0 || index < 3) return;
	if (PaceDepthMode() >= 0) {
		g_k = PaceDepthMode() + 1;
	} else if (g_lead_n >= 16) {
		const uint32_t n = std::min<uint32_t>(g_lead_n, 64);
		double         v[64];
		std::copy(g_lead_ring, g_lead_ring + n, v);
		const uint32_t q = std::min<uint32_t>(n - 1, static_cast<uint32_t>(n * 0.99));
		std::nth_element(v, v + q, v + n);
		const double p99 = v[q];
		const double vb  = 2.0; // (headroom before the slot; not a 60 Hz vblank: a ready frame flips when due)
		if (g_k == 1 && p99 > cap - vb) {
			g_k      = 2;
			g_k_hold = 0;
		} else if (g_k == 2) {
			g_k_hold = p99 <= cap - vb ? g_k_hold + 1 : 0;
			if (g_k_hold >= 60) g_k = 1;
		}
	}
	const int k = g_k;
	g_pace_k.store(k, std::memory_order_relaxed);
	// Frames queued behind the cap: the read of frame `index` waits until frame index - k has been released (at most
	// (k + 1) caps + 10 ms). The game then has a whole cap for its frame before that slot; no other sleep.
	const uint64_t target = index - static_cast<uint64_t>(k);
	if (g_rel_index >= target) return;
	const int64_t start = LatNow();
	g_pace_cond.wait_for(lock, std::chrono::microseconds(static_cast<int64_t>((cap * (k + 1) + 10.0) * 1000.0)),
	                     [target] { return g_rel_index >= target; });
	const double waited = static_cast<double>(LatNow() - start) / per_ms;
	g_pace_drain_sum.store(g_pace_drain_sum.load(std::memory_order_relaxed) + waited, std::memory_order_relaxed);
	g_pace_drain_n.fetch_add(1, std::memory_order_relaxed);
	lock.unlock();
	PaceLogOnce("drain wait", waited);
}

// KYTY_PAD_LATE_MS: per-5 s counters (under g_pace_mutex).
struct PadLateStats {
	int64_t  start = 0;
	uint32_t reads = 0, slept = 0, already_late = 0, idle = 0, released = 0, misses = 0, caused = 0, stutters = 0;
	double   slept_ms = 0.0, lead_sum = 0.0, late_sum = 0.0;
	uint32_t lead_n = 0;
};
PadLateStats g_padlate;
// KYTY_PAD_LATE_AUTO state (under g_pace_mutex).
double   g_pl_lead[64] {};
uint32_t g_pl_lead_n   = 0;
double   g_pl_margin   = 4.0;
double   g_pl_cur      = 0.0; // the L in use (0: not yet: the ceiling)
double   g_pl_cur_sum  = 0.0; // [latbd] mean L
uint32_t g_pl_cur_n    = 0;
double PadLateCurLocked() {
	if (!PadLateAuto() || g_pl_cur <= 0.0) return PadLateMs();
	return g_pl_cur;
}
uint32_t g_pl_skip     = 0; // reads left without the wait after a miss it caused (instant back-off)
// `lead` < 0: no sample (a frame late for another reason: the guest's own hitch, which no L could have saved).
void PadLateAdaptLocked(double lead, bool caused_miss, double cap) {
	if (!PadLateAuto()) return;
	// Only leads an L under the cap could cover (a guest hitch longer than the cap missed its slot whatever L was:
	// counting it kept the p95 and L ~25 ms too high, and the frame's queue wait with them).
	const bool sample = lead > 0.0 && lead < 100.0 && (cap <= 0.0 || lead < cap - 2.0);
	if (sample) g_pl_lead[g_pl_lead_n++ % 64] = lead;
	if (caused_miss) {
		g_pl_margin = std::min(g_pl_margin + 2.0, 8.0);
		g_pl_skip   = 8; // the next 8 reads at once (as the pacer's own miss rule)
	} else {
		g_pl_margin = std::max(g_pl_margin - 0.05, 4.0);
	}
	const uint32_t n = std::min<uint32_t>(g_pl_lead_n, 64);
	if (n < 16) return; // (the ceiling until 16 frames were seen)
	double v[64];
	std::copy(g_pl_lead, g_pl_lead + n, v);
	const uint32_t k = std::min<uint32_t>(n - 1, static_cast<uint32_t>(n * 0.95));
	std::nth_element(v, v + k, v + n);
	double target = std::clamp(v[k] + g_pl_margin, PadLateMinMs(), PadLateMs());
	if (caused_miss && sample && lead + g_pl_margin > target) target = std::min(lead + g_pl_margin, PadLateMs()); // up at once
	if (g_pl_cur <= 0.0) g_pl_cur = PadLateMs();
	// Up at once (a miss the wait caused), down 0.5 ms a frame (no sudden phase jump of the game's step).
	g_pl_cur = target >= g_pl_cur ? target : std::max(target, g_pl_cur - 0.5);
}

// The read of frame `index` waits until L ms before its slot: the last release + cap for each frame since. Returns the
// slot (LatNow ticks, 0: not paced) and the ms slept.
int64_t PadLateStep(uint64_t index, double& slept_ms) {
	slept_ms = 0.0;
	if (!PadLateOn()) return 0;
	const double per_ms = LatTicksPerMs();
	std::unique_lock lock(g_pace_mutex);
	const double cap = g_pace.cap_ms;
	g_padlate.reads++;
	const int64_t now = LatNow();
	if (cap < 20.0 || cap > 45.0 || g_rel_index == 0 || index <= g_rel_index || index - g_rel_index > 3 ||
	    static_cast<double>(now - g_rel_ticks) / per_ms > 200.0) {
		g_padlate.idle++;
		return 0;
	}
	const double  ahead = cap * static_cast<double>(index - g_rel_index);
	const int64_t slot  = g_rel_ticks + static_cast<int64_t>(ahead * per_ms);
	const double  lead_ms = PadLateCurLocked();
	g_pl_cur_sum += lead_ms;
	g_pl_cur_n++;
	const int64_t until = slot - static_cast<int64_t>(lead_ms * per_ms);
	if (g_pl_skip > 0) { // (back-off after a miss the wait caused: read at once)
		g_pl_skip--;
		g_padlate.already_late++;
		return slot;
	}
	if (until <= now) {
		g_padlate.already_late++;
		return slot;
	}
	lock.unlock();
	const double wait = std::min(static_cast<double>(until - now) / per_ms, cap);
	const int64_t end = now + static_cast<int64_t>(wait * per_ms);
	if (wait > 1.5) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>((wait - 1.0) * 1000.0)));
	while (LatNow() < end) std::this_thread::yield();
	slept_ms = static_cast<double>(LatNow() - now) / per_ms;
	lock.lock();
	g_padlate.slept++;
	g_padlate.slept_ms += slept_ms;
	return slot;
}

// At a release: a paced frame later than its slot is a miss; caused by the wait when it is late by no more than it slept.
void PadLateRelease(StageStamp& st, int64_t now, double dt_ms) {
	if (!PadLateOn()) return;
	const double per_ms = LatTicksPerMs();
	auto&        p      = g_padlate;
	if (p.start == 0) p.start = now;
	p.released++;
	const double vb = g_pace.vblank_ms > 0.0 ? g_pace.vblank_ms : 1.0;
	if (g_pace.cap_ms > 0.0 && dt_ms > g_pace.cap_ms + vb * 1.5 && dt_ms < 200.0) p.stutters++;
	const int64_t slot = st.slot.load(std::memory_order_relaxed);
	const int64_t in   = st.input.load(std::memory_order_relaxed);
	const int64_t rd   = st.ready.load(std::memory_order_relaxed);
	double lead = 0.0;
	if (in != 0 && rd > in) {
		lead = static_cast<double>(rd - in) / per_ms;
		p.lead_sum += lead;
		p.lead_n++;
	}
	bool caused = false;
	if (slot != 0) {
		const double late = static_cast<double>(now - slot) / per_ms;
		if (late > vb * 0.5 + 0.5) {
			p.misses++;
			p.late_sum += late;
			// Caused by the wait only when the frame would have made its slot without it (late by no more than it slept,
			// within half a vblank). (late <= slept + vb counted most guest hitches a whole cap late as the wait's: the
			// margin climbed to its max and L to ~38 ms, a 32 ms queue wait at 30x2.)
			if (late - st.slept.load(std::memory_order_relaxed) <= vb * 0.5) {
				p.caused++;
				caused = true;
			}
		}
	}
	if (slot != 0) {
		const bool late_other = !caused && static_cast<double>(now - slot) / per_ms > vb * 0.5 + 0.5;
		PadLateAdaptLocked(late_other ? -1.0 : lead, caused, g_pace.cap_ms);
	}
	if (static_cast<double>(now - p.start) / per_ms >= 5000.0) {
		std::printf("[padlate] 5 s: L %.1f ms (auto %.1f, margin %.1f), cap %.1f: %u reads (%u slept, mean %.1f ms; %u already past L; %u not under a "
		            "30-ish limit); %u releases, %u stutters; %u paced frames missed their slot (mean %.1f ms late), %u of "
		            "them caused by the wait; input -> gpu done %.1f ms\n",
		            PadLateMs(), PadLateCurLocked(), g_pl_margin, g_pace.cap_ms, p.reads, p.slept, p.slept ? p.slept_ms / p.slept : 0.0, p.already_late, p.idle,
		            p.released, p.stutters, p.misses, p.misses ? p.late_sum / p.misses : 0.0, p.caused,
		            p.lead_n ? p.lead_sum / p.lead_n : 0.0);
		std::fflush(stdout);
		p       = PadLateStats {};
		p.start = now;
	}
}
} // namespace

void SetPaceCap(double cap_ms, double vblank_ms) {
	if (!PaceOn() && !PhaseOn() && !PadLateOn()) return;
	std::lock_guard lock(g_pace_mutex);
	if (cap_ms <= 0.0) { // (not a capped game-view frame: off, and start over)
		g_pace.cap_ms = 0.0;
		g_pace.skip   = 8;
		return;
	}
	g_pace.cap_ms    = cap_ms;
	g_pace.vblank_ms = vblank_ms;
}

void OnFlipReady(uint64_t frame_index) {
	auto* st = StageFor(frame_index);
	if (st == nullptr) return;
	const int64_t now = LatNow();
	st->ready.store(now, std::memory_order_relaxed);
	if (!PaceOn()) return;
	const int64_t in = st->input.load(std::memory_order_relaxed);
	if (in == 0 || now <= in) return;
	const double sample = static_cast<double>(now - in) / LatTicksPerMs();
	if (sample > 100.0) return;
	std::lock_guard lock(g_pace_mutex);
	g_lead_ring[g_lead_n++ % 64] = sample;
	if (g_pace_lead <= 0.0) g_pace_lead = sample;
	else g_pace_lead += (sample - g_pace_lead) * (sample > g_pace_lead ? 0.5 : 0.05); // up fast, down slowly
}

void OnFlipReleased(uint64_t frame_index) {
	auto* st = StageFor(frame_index);
	if (st == nullptr) return;
	const int64_t now = LatNow();
	st->release.store(now, std::memory_order_relaxed);
	{
		std::lock_guard lock(g_pace_mutex);
		if (g_pace.margin <= 0.0) g_pace.margin = Own().margin;
		if (g_rel_index != 0 && frame_index == g_rel_index + 1 && g_pace.cap_ms > 0.0) {
			const double dt   = static_cast<double>(now - g_rel_ticks) / LatTicksPerMs();
			const bool   late = dt > g_pace.cap_ms + g_pace.vblank_ms * 0.5 && dt < 200.0;
			if (late) {
				g_pace.margin = std::min(g_pace.margin + 2.0, 16.0);
				g_pace.skip   = 8;
				g_pace.misses++;
			} else {
				if (g_pace.skip > 0) g_pace.skip--;
				g_pace.margin = std::max(g_pace.margin - 0.05, Own().margin);
			}
		}
		if (PadLateOn()) {
			const double dt = g_rel_ticks != 0 ? static_cast<double>(now - g_rel_ticks) / LatTicksPerMs() : 0.0;
			PadLateRelease(*st, now, frame_index == g_rel_index + 1 ? dt : 0.0);
		}
		if (frame_index > g_rel_index) {
			g_rel_index = frame_index;
			g_rel_ticks = now;
		}
	}
	g_pace_cond.notify_all();
}

bool OwnLowLatencyOn() {
	return Own().on;
}

// KYTY_LATENCY (anti-lag.h): the input stamps by frame index, and the present thread's one-second window.
namespace {
int64_t LatNow() {
#if defined(_WIN32)
	LARGE_INTEGER v;
	QueryPerformanceCounter(&v);
	return v.QuadPart;
#else
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}
double LatTicksPerMs() {
	static const double per_ms = [] {
#if defined(_WIN32)
		LARGE_INTEGER f;
		QueryPerformanceFrequency(&f);
		return static_cast<double>(f.QuadPart) / 1000.0;
#else
		return 1e6;
#endif
	}();
	return per_ms;
}
constexpr uint64_t kLatRing = 64;
struct LatStamp {
	std::atomic<uint64_t> index {0};
	std::atomic<int64_t>  ticks {0};
};
LatStamp g_lat_stamps[kLatRing];
struct LatWindow {
	std::mutex          mutex;
	std::vector<double> ms;
	double              st_sum[4] {};
	uint32_t            st_n = 0;
	int64_t             start = 0, first_present = 0, last_present = 0;
	uint32_t            presents = 0;
} g_lat;

const char* LatModeName(Mode mode) {
	if (FrameGen::StreamlineLoaded()) return "reflex-sl"; // (Streamline's Reflex with DLSS / DLSS-G)
	switch (mode) {
		case Mode::Amd: return "antilag2";
		case Mode::Reflex: return "reflex";
		case Mode::Limiter: return "limiter";
		default: break;
	}
	return Own().on ? "lowlat" : "off";
}
// Generated frames per real one right now (0: no frame generation).
int LatGenerated() {
	if (FrameGen::Enabled()) return std::max(FrameGen::GeneratedFrames(), 0);
	if (Fsr3::Active()) {
#if defined(_WIN32)
		char text[8] {};
		if (GetEnvironmentVariableA("KYTY_FSR3_FG_ACTIVE", text, sizeof(text)) > 0) return std::clamp(std::atoi(text), 1, 7);
#endif
		return 1;
	}
	return 0;
}
double LatRefreshMs() {
#if defined(_WIN32)
	DEVMODEA mode {};
	mode.dmSize = sizeof(mode);
	if (EnumDisplaySettingsA(nullptr, ENUM_CURRENT_SETTINGS, &mode) != 0 && mode.dmDisplayFrequency > 1)
		return 1000.0 / static_cast<double>(mode.dmDisplayFrequency);
#endif
	return 0.0;
}
} // namespace

bool LatencyOn() {
	static const bool on = [] {
		const char* v       = std::getenv("KYTY_LATENCY");
		const bool  enabled = v != nullptr && v[0] == '1';
		if (enabled) {
			std::printf("[lat] on: input (controller read) -> present of that real frame, once a second\n");
			std::fflush(stdout);
		}
		return enabled;
	}();
	return on;
}

void OnPresented(uint64_t frame_index) {
	if (!LatencyOn() || frame_index == 0) return;
	const int64_t now  = LatNow();
	auto&         slot = g_lat_stamps[frame_index % kLatRing];
	if (slot.index.load(std::memory_order_acquire) != frame_index) return;
	const int64_t at = slot.ticks.load(std::memory_order_relaxed);
	if (at == 0 || now <= at) return;
	const double per_ms = LatTicksPerMs();
	const double ms     = static_cast<double>(now - at) / per_ms;
	std::lock_guard lock(g_lat.mutex);
	if (g_lat.start == 0) {
		g_lat.start = now;
		g_lat.ms.reserve(512);
	}
	if (ms < 1000.0) g_lat.ms.push_back(ms); // (a pause / loading screen: not a frame's latency)
	if (const auto* st = StageFor(frame_index); st != nullptr && ms < 1000.0) {
		const int64_t in = st->input.load(), rs = st->reserve.load(), rd = st->ready.load(), rl = st->release.load();
		if (in != 0 && rs >= in && rd >= rs && rl >= rd && now >= rl) {
			g_lat.st_sum[0] += static_cast<double>(rs - in) / per_ms;
			g_lat.st_sum[1] += static_cast<double>(rd - rs) / per_ms;
			g_lat.st_sum[2] += static_cast<double>(rl - rd) / per_ms;
			g_lat.st_sum[3] += static_cast<double>(now - rl) / per_ms;
			g_lat.st_n++;
		}
	}
	if (g_lat.presents == 0) g_lat.first_present = now;
	g_lat.last_present = now;
	g_lat.presents++;
	if (static_cast<double>(now - g_lat.start) / per_ms < 1000.0) return;
	auto&        v = g_lat.ms;
	const size_t n = v.size();
	if (n > 0) {
		double sum = 0.0, max = 0.0;
		for (double x : v) {
			sum += x;
			max = std::max(max, x);
		}
		const double mean = sum / static_cast<double>(n);
		const size_t k    = std::min(n - 1, static_cast<size_t>(static_cast<double>(n) * 0.95));
		std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
		const double p95      = v[k];
		const double scanout  = LatRefreshMs() * 0.5;
		const int    g        = LatGenerated();
		const double interval = g_lat.presents > 1 ? static_cast<double>(g_lat.last_present - g_lat.first_present) / per_ms /
		                                                 static_cast<double>(g_lat.presents - 1)
		                                           : 0.0;
		const double fg = g > 0 ? interval * g / (g + 1.0) : 0.0;
		std::printf("[lat] input->present mean %.1f ms, p95 %.1f, max %.1f; +scanout %.1f; fg +%.1f (n %u); total %.1f, p95 %.1f; "
		            "mode %s, fg x%d\n",
		            mean, p95, max, scanout, fg, static_cast<unsigned>(n), mean + scanout + fg, p95 + scanout + fg,
		            LatModeName(g_mode.load(std::memory_order_relaxed)), g + 1);
		if (g_lat.st_n > 0) {
			const double d   = static_cast<double>(g_lat.st_n);
			const uint32_t sn = g_pace_slept_n.exchange(0, std::memory_order_relaxed);
			const double   ss = g_pace_slept_sum.exchange(0.0, std::memory_order_relaxed);
			const uint32_t dn = g_pace_drain_n.exchange(0, std::memory_order_relaxed);
			const double   ds = g_pace_drain_sum.exchange(0.0, std::memory_order_relaxed);
			double         lead = 0.0, iv = 0.0, margin = 0.0;
			uint32_t       misses = 0;
			{
				std::lock_guard pl(g_pace_mutex);
				lead   = g_pace_lead;
				iv     = g_pace.cap_ms;
				margin = g_pace.margin;
				misses = g_pace.misses;
				g_pace.misses = 0;
			}
			std::printf("[lat] stages: input->submit %.1f ms, submit->gpu done %.1f, gpu done->flip %.1f, flip->present %.1f "
			            "(n %u); pace %s: drain %.1f ms (n %u), sleep %.1f ms (n %u), lead %.1f + margin %.1f, cap %.1f, k %d, late %u\n",
			            g_lat.st_sum[0] / d, g_lat.st_sum[1] / d, g_lat.st_sum[2] / d, g_lat.st_sum[3] / d, g_lat.st_n,
			            !PaceOn() ? "off" : (PaceAllowed() && iv > 0.0 ? "on" : "idle"), dn > 0 ? ds / dn : 0.0, dn,
			            sn > 0 ? ss / sn : 0.0, sn, lead, margin, iv, g_pace_k.load(std::memory_order_relaxed), misses);
			std::fflush(stdout);
		}
	}
	// (10-26 latq) Every 10 s: where the overlay's latency goes, one line.
	{
		static double   bd_sum[6] {};
		static uint32_t bd_n = 0, bd_w = 0;
		static int64_t  bd_start = 0;
		if (bd_start == 0) bd_start = now;
		if (n > 0 && g_lat.st_n > 0) {
			const double d     = static_cast<double>(g_lat.st_n);
			const int    g     = LatGenerated();
			const double ivl   = g_lat.presents > 1 ? static_cast<double>(g_lat.last_present - g_lat.first_present) / per_ms /
			                                            static_cast<double>(g_lat.presents - 1)
			                                      : 0.0;
			bd_sum[0] += g_lat.st_sum[0] / d;
			bd_sum[1] += g_lat.st_sum[1] / d;
			bd_sum[2] += g_lat.st_sum[2] / d;
			bd_sum[3] += g_lat.st_sum[3] / d;
			bd_sum[4] += g > 0 ? ivl * g / (g + 1.0) : 0.0;
			bd_sum[5] += LatRefreshMs() * 0.5;
			bd_n++;
		}
		if (static_cast<double>(now - bd_start) / per_ms >= 10000.0) {
			if (bd_n > 0) {
				const double k = static_cast<double>(bd_n);
				double       l = 0.0;
				{
					std::lock_guard pl(g_pace_mutex);
					l           = g_pl_cur_n ? g_pl_cur_sum / g_pl_cur_n : 0.0;
					g_pl_cur_sum = 0.0;
					g_pl_cur_n   = 0;
				}
				double tot = 0.0;
				for (double x : bd_sum) tot += x / k;
				std::printf("[latbd] 10 s: input->frame submit %.1f ms + frame gpu %.1f + queue/pace wait (gpu done->flip) %.1f + "
				            "flip->present %.1f + fg hold %.1f + scanout %.1f = %.1f ms; pad late L %.1f ms\n",
				            bd_sum[0] / k, bd_sum[1] / k, bd_sum[2] / k, bd_sum[3] / k, bd_sum[4] / k, bd_sum[5] / k, tot, l);
				std::fflush(stdout);
			}
			for (double& x : bd_sum) x = 0.0;
			bd_n     = 0;
			bd_start = now;
			(void)bd_w;
		}
	}
	for (double& x : g_lat.st_sum) x = 0.0;
	g_lat.st_n = 0;
	v.clear();
	g_lat.start    = now;
	g_lat.presents = 0;
}

void OnFrameSubmitted(VkDevice device, VkSemaphore timeline, uint64_t tick) {
	if (!Own().on || device == VK_NULL_HANDLE || timeline == VK_NULL_HANDLE || tick == 0) return;
	g_own_device.store(device, std::memory_order_release);
	g_own_timeline.store(timeline, std::memory_order_release);
	const uint64_t frame = g_own_frames.load(std::memory_order_relaxed);
	g_own_ticks[frame % kOwnRing].store(tick, std::memory_order_release);
	g_own_frames.store(frame + 1, std::memory_order_release);
}


// KYTY_PHASE_PACE (anti-lag.h): the constant-phase pacer, ported 1:1 from tools sim phase2.py (proved there on 212307 and
// 202407: game steps sd <= 0.29 ms, screen sd <= 0.28 ms, input->present 100 -> 31-41 ms).
namespace {
struct Phase {
	double   anchor = 0.0;  // a smoothed release time on the flip grid, ms (LatNow)
	double   last_rel = 0.0;
	bool     have_anchor = false;
	double   phi = 0.0;     // the step phase on that grid, ms (unwrapped)
	bool     have_phi = false;
	double   qe = -1.0;     // EMA of gpu done -> flip (FlipQueue's stamps), ms
	double   last_late = -1e12;
	int      q = 1;         // frames of cushion (1 while late in the last 5 s, 0 after 10 s)
	double   last_step = 0.0;
	// cadence: releases per game step over 2 s windows
	double   cad_start = 0.0;
	uint32_t cad_steps = 0, cad_rel = 0;
	int      cadence = 1;
	// per-second stats
	double   st_start = 0.0;
	uint32_t st_steps = 0, st_waits = 0, st_rel = 0, st_late = 0, st_depth_n = 0;
	double   st_wait = 0.0, st_sum = 0.0, st_sum2 = 0.0, st_max = 0.0, st_q = 0.0, st_depth = 0.0;
	double   rs_sum = 0.0, rs_sum2 = 0.0;
	uint32_t rs_n = 0;
	bool     logged_thread = false;
};
Phase                g_phase;
std::mutex           g_phase_mutex;
std::atomic<int>     g_flip_depth {0};
double NowMs() { return static_cast<double>(LatNow()) / LatTicksPerMs(); }

bool PhaseActiveLocked() {
	return PhaseOn() && !FrameGen::Enabled() && g_pace.cap_ms >= 4.0 && g_pace.cap_ms <= 70.0;
}

void PhaseStep(const char* site) {
	if (!PhaseOn() || FrameGen::Enabled()) return;
	double cap = 0.0, vb = 0.0;
	{
		std::lock_guard pl(g_pace_mutex);
		cap = g_pace.cap_ms;
		vb  = g_pace.vblank_ms;
	}
	const double now = NowMs();
	std::unique_lock lock(g_phase_mutex);
	auto& ph = g_phase;
	ph.cad_steps++;
	if (cap < 4.0 || cap > 70.0 || !ph.have_anchor || ph.qe < 0.0 || now - ph.last_rel > 200.0) {
		ph.have_phi  = false;
		ph.last_step = now;
		return;
	}
	const double P = cap * ph.cadence;
	if (now - ph.last_late < 5000.0) ph.q = 1;
	else if (now - ph.last_late > 10000.0) ph.q = 0;
	const double target = std::max(2.0, vb + 1.0) + ph.q * cap;
	if (!ph.have_phi) {
		ph.phi      = std::fmod(now - ph.anchor, P);
		ph.have_phi = true;
	}
	ph.phi += std::clamp(0.1 * (ph.qe - target), -0.5, 0.5);
	double x = std::fmod(now - ph.anchor - ph.phi, P);
	if (x < 0.0) x += P;
	double until = now + std::fmod(P - x, P);
	if (ph.last_step > 0.0) until = std::min(until, std::max(now, ph.last_step + P + 1.0)); // (never a step over P + 1 ms)
	const double wait = until - now;
	lock.unlock();
	if (wait > 0.05) {
		if (wait > 1.5) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>((wait - 1.0) * 1000.0)));
		while (NowMs() < until) std::this_thread::yield();
	}
	const double t = NowMs();
	lock.lock();
	if (ph.last_step > 0.0 && t - ph.last_step < 200.0) {
		const double iv = t - ph.last_step;
		ph.st_steps++;
		ph.st_sum += iv;
		ph.st_sum2 += iv * iv;
		ph.st_max = std::max(ph.st_max, iv);
	}
	if (wait > 0.05) {
		ph.st_waits++;
		ph.st_wait += wait;
	}
	ph.last_step = t;
	if (!ph.logged_thread && wait > 0.05) {
		ph.logged_thread = true;
#if defined(_WIN32)
		const unsigned long tid = GetCurrentThreadId();
#else
		const unsigned long tid = static_cast<unsigned long>(std::hash<std::thread::id> {}(std::this_thread::get_id()));
#endif
		std::printf("[phase] first wait %.1f ms on thread %lu at %s (the game's controller read)\n", wait, tid, site);
		std::fflush(stdout);
	}
}
} // namespace

void NoteFlipDepth(int depth) {
	g_flip_depth.store(depth, std::memory_order_relaxed);
}

void PhaseRelease(double queued_ms) {
	if (!PhaseOn()) return;
	double cap = 0.0, vb = 0.0;
	{
		std::lock_guard pl(g_pace_mutex);
		cap = g_pace.cap_ms;
		vb  = g_pace.vblank_ms;
	}
	const double now = NowMs();
	std::lock_guard lock(g_phase_mutex);
	auto& ph = g_phase;
	ph.cad_rel++;
	if (ph.cad_start == 0.0) ph.cad_start = now;
	if (now - ph.cad_start >= 2000.0) {
		if (ph.cad_steps > 0) ph.cadence = std::clamp(static_cast<int>(std::lround(static_cast<double>(ph.cad_rel) / ph.cad_steps)), 1, 3);
		ph.cad_start = now;
		ph.cad_steps = ph.cad_rel = 0;
	}
	if (cap < 4.0 || cap > 70.0) {
		ph.have_anchor = false;
		ph.qe          = -1.0;
		ph.last_rel    = now;
		return;
	}
	if (ph.have_anchor && now - ph.last_rel < 200.0) {
		const double iv = now - ph.last_rel;
		if (iv > cap + std::max(vb * 0.5, 0.5)) {
			ph.last_late = now;
			ph.st_late++;
		}
		ph.rs_n++;
		ph.rs_sum += iv;
		ph.rs_sum2 += iv * iv;
	}
	if (!ph.have_anchor || now - ph.last_rel > 200.0) {
		ph.anchor      = now;
		ph.have_anchor = true;
	} else {
		double e = now - ph.anchor;
		e -= std::round(e / cap) * cap;
		ph.anchor += 0.1 * e;
		ph.anchor += std::round((now - ph.anchor) / cap) * cap; // (kept near now: fmod stays precise)
	}
	ph.last_rel = now;
	if (queued_ms >= 0.0 && queued_ms < 500.0) ph.qe = ph.qe < 0.0 ? queued_ms : ph.qe + (queued_ms - ph.qe) * 0.2;
	ph.st_rel++;
	ph.st_q += std::max(queued_ms, 0.0);
	ph.st_depth += g_flip_depth.load(std::memory_order_relaxed);
	ph.st_depth_n++;
	if (ph.st_start == 0.0) ph.st_start = now;
	if (now - ph.st_start >= 1000.0) {
		auto sd = [](double sum, double sum2, uint32_t n) {
			if (n == 0) return 0.0;
			const double m = sum / n;
			return std::sqrt(std::max(0.0, sum2 / n - m * m));
		};
		std::printf("[phase] %s: period %.1f ms (cap %.1f x cadence %d), cushion %d, phase %.1f, gpu done->flip %.1f ms, "
		            "flip queue %.2f, waits %u (mean %.1f ms), game steps %u sd %.2f ms (max %.1f), screen sd %.2f ms, late %u\n",
		            (cap >= 4.0 && cap <= 70.0 && !FrameGen::Enabled()) ? "on" : "idle", cap * ph.cadence, cap, ph.cadence, ph.q,
		            std::fmod(ph.phi, cap * ph.cadence), ph.st_rel ? ph.st_q / ph.st_rel : 0.0,
		            ph.st_depth_n ? ph.st_depth / ph.st_depth_n : 0.0, ph.st_waits, ph.st_waits ? ph.st_wait / ph.st_waits : 0.0,
		            ph.st_steps, sd(ph.st_sum, ph.st_sum2, ph.st_steps), ph.st_max, sd(ph.rs_sum, ph.rs_sum2, ph.rs_n), ph.st_late);
		std::fflush(stdout);
		ph.st_start = now;
		ph.st_steps = ph.st_waits = ph.st_rel = ph.st_late = ph.st_depth_n = ph.rs_n = 0;
		ph.st_wait = ph.st_sum = ph.st_sum2 = ph.st_max = ph.st_q = ph.st_depth = ph.rs_sum = ph.rs_sum2 = 0.0;
	}
}

void OnInput(const char* site) {
	if (Own().on) OwnLimit();
	const Mode mode = g_mode.load(std::memory_order_relaxed);
	if (mode == Mode::Off && !LatencyOn() && !PaceOn() && !PhaseOn() && !PadLateOn()) return;
	// Once per flip: the first controller read after the game flipped its last frame starts the next one.
	const uint64_t flips = g_flips.load(std::memory_order_acquire);
	uint64_t       seen  = g_flips_at_input.load(std::memory_order_relaxed);
	if (seen == flips) return;
	if (!g_flips_at_input.compare_exchange_strong(seen, flips, std::memory_order_acq_rel)) return;
	const uint64_t index = g_input_index.load(std::memory_order_relaxed) + 1;
	bool phase_active = false;
	{
		std::lock_guard pl(g_pace_mutex);
		phase_active = PhaseActiveLocked();
	}
	if (mode == Mode::Limiter && index > 2 && !phase_active) {
		// At most one frame (index - 1) between the game and the present: frame index - 2 must have been presented
		// (the presenter waited for its GPU work first). A frame the flip pacing holds is the one queued: no judder.
		std::unique_lock lock(g_mutex);
		g_cond.wait_for(lock, std::chrono::milliseconds(100),
		                [index] { return g_presented.load(std::memory_order_acquire) + 2 >= index; });
	}
	if (mode == Mode::Amd) Mark(kStageInput, index); // (the driver may sleep here)
	if (mode == Mode::Reflex) {
		// The driver's sleep: it signals the timeline when the frame should start; waited for without the lock (at most
		// 50 ms), on the game's thread at its controller read (no emulator GPU / translation thread waits here).
		uint64_t value = 0;
		{
			std::lock_guard lock(g_reflex_mutex);
			if (g_swapchain != VK_NULL_HANDLE) {
				VkLatencySleepInfoNV info {};
				info.sType           = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV;
				info.signalSemaphore = g_semaphore;
				info.value           = ++g_semaphore_value;
				if (g_sleep(g_device, g_swapchain, &info) == VK_SUCCESS) value = info.value;
			}
		}
		if (value != 0 && !phase_active) { // (the phase pacer's step wait instead: one sleeper)
			VkSemaphoreWaitInfo wait {};
			wait.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
			wait.semaphoreCount = 1;
			wait.pSemaphores    = &g_semaphore;
			wait.pValues        = &value;
			g_wait(g_device, &wait, 50'000'000);
		}
		ReflexMarker(VK_LATENCY_MARKER_SIMULATION_START_NV, index);
		ReflexMarker(VK_LATENCY_MARKER_INPUT_SAMPLE_NV, index);
	}
	g_pace_site = site;
	PaceBeforeInput(index); // KYTY_LOWLAT_PACE_BEFORE_INPUT: sleep before the sample, not after
	PhaseStep(site);        // KYTY_PHASE_PACE: the step at its phase of the flip grid
	double        pad_slept = 0.0;
	const int64_t pad_slot  = PadLateStep(index, pad_slept); // KYTY_PAD_LATE_MS: the read L ms before its slot
	{
		auto& st = g_stage[index % kStageRing];
		st.index.store(0, std::memory_order_release);
		st.slot.store(pad_slot, std::memory_order_relaxed);
		st.slept.store(pad_slept, std::memory_order_relaxed);
		st.reserve.store(0, std::memory_order_relaxed);
		st.ready.store(0, std::memory_order_relaxed);
		st.release.store(0, std::memory_order_relaxed);
		st.input.store(LatNow(), std::memory_order_relaxed);
		st.index.store(index, std::memory_order_release);
	}
	if (LatencyOn()) { // KYTY_LATENCY: the input sample, after any wait above
		auto& slot = g_lat_stamps[index % kLatRing];
		slot.ticks.store(LatNow(), std::memory_order_relaxed);
		slot.index.store(index, std::memory_order_release);
	}
	g_input_index.store(index, std::memory_order_release);
}

uint64_t OnFlipReserved() {
	if (g_mode.load(std::memory_order_relaxed) == Mode::Off && !LatencyOn() && !PaceOn() && !PhaseOn() && !PadLateOn()) return 0;
	g_flips.fetch_add(1, std::memory_order_acq_rel);
	const uint64_t index  = g_input_index.load(std::memory_order_acquire);
	uint64_t       tagged = g_tagged_index.load(std::memory_order_relaxed);
	// One flip per INPUT mark (a frame without a controller read has none).
	if (index == 0 || tagged >= index) return 0;
	if (!g_tagged_index.compare_exchange_strong(tagged, index, std::memory_order_acq_rel)) return 0;
	if (auto* st = StageFor(index)) st->reserve.store(LatNow(), std::memory_order_relaxed);
	if (g_mode.load(std::memory_order_relaxed) == Mode::Reflex) {
		// The game's CPU frame ends at its flip; the emulator's translation and submission of it follow.
		ReflexMarker(VK_LATENCY_MARKER_SIMULATION_END_NV, index);
		ReflexMarker(VK_LATENCY_MARKER_RENDERSUBMIT_START_NV, index);
	}
	return index;
}

void OnPresent(uint64_t frame_index) {
	const Mode mode = g_mode.load(std::memory_order_relaxed);
	if (mode == Mode::Off || frame_index == 0) return;
	if (frame_index <= g_presented.load(std::memory_order_relaxed)) return;
	if (mode == Mode::Amd) Mark(kStagePresent, frame_index);
	if (mode == Mode::Reflex) {
		ReflexMarker(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, frame_index);
		bool fsr3 = false;
		{
			std::lock_guard lock(g_reflex_mutex);
			fsr3 = g_fsr3_real;
		}
		if (!fsr3) ReflexMarker(VK_LATENCY_MARKER_PRESENT_START_NV, frame_index);
	}
	{
		std::lock_guard lock(g_mutex);
		g_presented.store(frame_index, std::memory_order_release);
	}
	g_cond.notify_all();
}

bool ReflexActive() {
	return g_mode.load(std::memory_order_relaxed) == Mode::Reflex;
}

void* ChainSwapchain(void* next, bool fsr3) {
	if (!ReflexActive()) return next;
	if (fsr3 && Fsr3RealSwapchain() == nullptr) return next; // (AMD's signed DLL refuses unknown structures)
	g_latency_create                    = {};
	g_latency_create.sType              = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV;
	g_latency_create.latencyModeEnable  = VK_TRUE;
	g_latency_create.pNext              = next;
	return &g_latency_create;
}

void OnSwapchain(VkSwapchainKHR swapchain, bool fsr3) {
	if (!ReflexActive()) return;
	std::lock_guard lock(g_reflex_mutex);
	g_swapchain = VK_NULL_HANDLE;
	g_fsr3_real = false;
	if (swapchain == VK_NULL_HANDLE) return;
	if (fsr3) {
		// FSR 3's handle is its frame interpolation swapchain: the patched DLL gives the driver swapchain behind it (made
		// with latency mode), whose real-frame presents it marks PRESENT_START / END with our frame index.
		const auto            get  = Fsr3RealSwapchain();
		const VkSwapchainKHR  real = get != nullptr ? get(swapchain) : VK_NULL_HANDLE;
		if (!g_fsr3_logged.exchange(true)) {
			std::printf(real != VK_NULL_HANDLE
			                ? "Anti-Lag: NVIDIA Reflex on FSR 3 frame generation's real swapchain (patched FidelityFX DLL)\n"
			                : "Anti-Lag: NVIDIA Reflex off for the FSR 3 frame generation swapchain (its real swapchain is AMD's: "
			                  "this amd_fidelityfx_vk.dll does not give it, the patched one does): no sleep, no markers\n");
			std::fflush(stdout);
		}
		if (real == VK_NULL_HANDLE) return;
		swapchain   = real;
		g_fsr3_real = true;
	}
	VkLatencySleepModeInfoNV info {};
	info.sType             = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV;
	info.lowLatencyMode    = VK_TRUE;
	info.lowLatencyBoost   = VK_FALSE;
	info.minimumIntervalUs = 0; // (the flip pacing keeps the rate)
	const VkResult result  = g_sleep_mode(g_device, swapchain, &info);
	if (result == VK_SUCCESS) {
		g_swapchain = swapchain;
	} else {
		g_fsr3_real = false;
		std::printf("Anti-Lag: vkSetLatencySleepModeNV failed (%d): Reflex off for this swapchain\n", static_cast<int>(result));
		std::fflush(stdout);
	}
}

void AfterPresent(uint64_t frame_index) {
	if (!ReflexActive()) return;
	{
		std::lock_guard lock(g_reflex_mutex);
		if (g_fsr3_real) return; // (the DLL's present thread marks the real frame's present)
	}
	ReflexMarker(VK_LATENCY_MARKER_PRESENT_END_NV, frame_index);
}

uint64_t PresentId(uint64_t frame_index) {
	if (!ReflexActive()) return 0;
	std::lock_guard lock(g_reflex_mutex);
	return g_swapchain != VK_NULL_HANDLE ? frame_index : 0;
}

} // namespace Libs::Graphics::AntiLag
