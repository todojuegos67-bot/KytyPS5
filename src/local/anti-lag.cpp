#include "local/anti-lag.h"

#include "local/frame-gen.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics::AntiLag {

// VK_AMD_anti_lag, declared here (older Vulkan headers do not have it; the values are the registry's).
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

enum class Mode { Off, Amd, Limiter };

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

void OnDevice(vk::Device device, PFN_vkGetDeviceProcAddr get_proc, bool extension_enabled) {
	static bool logged = false;
	Mode        mode   = Mode::Off;
	const char* text   = nullptr;
	if (!Requested()) {
		text = "Anti-Lag: off (KYTY_ANTILAG=0)";
	} else if (FrameGen::StreamlineLoaded()) {
		text = "Anti-Lag: off (NVIDIA Streamline loaded: Reflex is active)";
	} else {
		if (extension_enabled && get_proc != nullptr)
			g_update = reinterpret_cast<UpdateFn>(get_proc(static_cast<VkDevice>(device), "vkAntiLagUpdateAMD"));
		if (g_update != nullptr) {
			g_device = static_cast<VkDevice>(device);
			mode     = Mode::Amd;
			text     = "Anti-Lag: AMD Anti-Lag 2 (VK_AMD_anti_lag)";
		} else if (const char* v = std::getenv("KYTY_ANTILAG"); v == nullptr || v[0] != '2') {
			// (The limiter's wait in the controller read corrupted the picture in a test on NVIDIA: opt-in only.)
			text = "Anti-Lag: off (no VK_AMD_anti_lag; KYTY_ANTILAG=2 for the experimental emulator latency limiter)";
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

void OnInput() {
	const Mode mode = g_mode.load(std::memory_order_relaxed);
	if (mode == Mode::Off) return;
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
	g_input_index.store(index, std::memory_order_release);
}

uint64_t OnFlipReserved() {
	if (g_mode.load(std::memory_order_relaxed) == Mode::Off) return 0;
	g_flips.fetch_add(1, std::memory_order_acq_rel);
	const uint64_t index  = g_input_index.load(std::memory_order_acquire);
	uint64_t       tagged = g_tagged_index.load(std::memory_order_relaxed);
	// One flip per INPUT mark (a frame without a controller read has none).
	if (index == 0 || tagged >= index) return 0;
	if (!g_tagged_index.compare_exchange_strong(tagged, index, std::memory_order_acq_rel)) return 0;
	return index;
}

void OnPresent(uint64_t frame_index) {
	const Mode mode = g_mode.load(std::memory_order_relaxed);
	if (mode == Mode::Off || frame_index == 0) return;
	if (frame_index <= g_presented.load(std::memory_order_relaxed)) return;
	if (mode == Mode::Amd) Mark(kStagePresent, frame_index);
	{
		std::lock_guard lock(g_mutex);
		g_presented.store(frame_index, std::memory_order_release);
	}
	g_cond.notify_all();
}

} // namespace Libs::Graphics::AntiLag
