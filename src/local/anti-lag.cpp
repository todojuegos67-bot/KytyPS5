#include "local/anti-lag.h"

#include "local/frame-gen.h"
#include "local/fsr3-fg.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::Graphics::AntiLag {

// VK_AMD_anti_lag, declared here (older Vulkan headers do not have it; the values are the registry's).
namespace {
struct OwnState {
	bool     on    = false;
	uint32_t queue = 2;
	OwnState() {
		const char* v = std::getenv("KYTY_LOWLAT");
		on            = v != nullptr && v[0] == '1';
		if (const char* q = std::getenv("KYTY_LOWLAT_QUEUE"); q != nullptr && q[0] == '1') queue = 1;
		if (on) {
			std::printf("Low latency (own): on, max %u frames queued at the controller read\n", queue);
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
	if (!Requested()) {
		text = "Anti-Lag: off (KYTY_ANTILAG=0)";
	} else if (FrameGen::StreamlineLoaded()) {
		text = "Anti-Lag: off (NVIDIA Streamline loaded: Reflex is active)";
	} else {
		if (extension_enabled && get_proc != nullptr)
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
			text     = "Anti-Lag: NVIDIA Reflex (VK_NV_low_latency2; low latency on, boost off, no frame limit)";
		} else if (g_update != nullptr) {
			g_device = static_cast<VkDevice>(device);
			mode     = Mode::Amd;
			text     = "Anti-Lag: AMD Anti-Lag 2 (VK_AMD_anti_lag)";
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
} // namespace

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
		std::fflush(stdout);
	}
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

void OnInput() {
	if (Own().on) OwnLimit();
	const Mode mode = g_mode.load(std::memory_order_relaxed);
	if (mode == Mode::Off && !LatencyOn()) return;
	// Once per flip: the first controller read after the game flipped its last frame starts the next one.
	const uint64_t flips = g_flips.load(std::memory_order_acquire);
	uint64_t       seen  = g_flips_at_input.load(std::memory_order_relaxed);
	if (seen == flips) return;
	if (!g_flips_at_input.compare_exchange_strong(seen, flips, std::memory_order_acq_rel)) return;
	const uint64_t index = g_input_index.load(std::memory_order_relaxed) + 1;
	if (mode == Mode::Limiter && index > 2) {
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
		if (value != 0) {
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
	if (LatencyOn()) { // KYTY_LATENCY: the input sample, after any wait above
		auto& slot = g_lat_stamps[index % kLatRing];
		slot.ticks.store(LatNow(), std::memory_order_relaxed);
		slot.index.store(index, std::memory_order_release);
	}
	g_input_index.store(index, std::memory_order_release);
}

uint64_t OnFlipReserved() {
	if (g_mode.load(std::memory_order_relaxed) == Mode::Off && !LatencyOn()) return 0;
	g_flips.fetch_add(1, std::memory_order_acq_rel);
	const uint64_t index  = g_input_index.load(std::memory_order_acquire);
	uint64_t       tagged = g_tagged_index.load(std::memory_order_relaxed);
	// One flip per INPUT mark (a frame without a controller read has none).
	if (index == 0 || tagged >= index) return 0;
	if (!g_tagged_index.compare_exchange_strong(tagged, index, std::memory_order_acq_rel)) return 0;
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
