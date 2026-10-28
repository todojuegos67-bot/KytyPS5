// Upscaling internals shared by upscale.cpp and the backends (up_dlss.cpp, up_fsr4.cpp, up_fsr3.cpp, up_xess.cpp).
#pragma once
#include "camera.h"
#include "common.h"

#include <memory>
#include <string>
#include <vector>

namespace inj::upscale {

enum Kind : int { kOff = 0, kDlss = 1, kFsr4 = 2, kFsr3 = 3, kXess = 4, kKinds = 5 };
const char* KindName(int kind);

// Quality modes (resolved, never 0): 1 native AA, 2 quality, 3 balanced, 4 performance, 5 ultra performance.
const char* QualityName(int quality);
int         QualityForRatio(float ratio);  // the mode a render / display ratio corresponds to
float       ScaleForQuality(int quality);  // the game's resolution scale that mode wants (1, 0.67, 0.58, 0.5, 0.33)

// ---- our Vulkan objects (device-local unless said) ----
struct OwnImage {
	VkImage           image = VK_NULL_HANDLE;
	VkDeviceMemory    memory = VK_NULL_HANDLE;
	VkImageView       view = VK_NULL_HANDLE;
	VkFormat          format = VK_FORMAT_UNDEFINED;
	VkExtent2D        extent {};
	VkImageUsageFlags usage = 0;
	VkImageLayout     layout = VK_IMAGE_LAYOUT_UNDEFINED;  // as recorded so far
};
bool CreateOwnImage(OwnImage& img, VkFormat format, VkExtent2D extent, VkImageUsageFlags usage);
void DestroyOwnImage(OwnImage& img);
struct OwnBuffer {
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	VkDeviceSize   size = 0;
	void*          mapped = nullptr;  // host-visible ones
};
bool CreateOwnBuffer(OwnBuffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host);
void DestroyOwnBuffer(OwnBuffer& b);

bool IsDepthFormat(VkFormat f);
bool HasStencil(VkFormat f);
bool IsFloatFormat(VkFormat f);  // HDR-capable colour (16 / 32-bit float, B10G11R11)
VkImageAspectFlags ViewAspect(VkFormat f);     // depth only for depth formats
VkImageAspectFlags BarrierAspect(VkFormat f);  // depth + stencil where it has both
void Transition(VkCommandBuffer cb, OwnImage& img, VkImageLayout to);
void Transition(VkCommandBuffer cb, VkImage image, VkFormat format, VkImageLayout from, VkImageLayout to);
void FullBarrier(VkCommandBuffer cb);

// A compute pipeline with one push-descriptor set (VK_KHR_push_descriptor) and push constants.
struct ComputePipe {
	VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
	VkPipelineLayout      layout = VK_NULL_HANDLE;
	VkPipeline            pipeline = VK_NULL_HANDLE;
};
bool CreateComputePipe(ComputePipe& p, const uint32_t* code, size_t bytes, const char* entry,
                       const std::vector<VkDescriptorType>& bindings, uint32_t push_bytes);
void DestroyComputePipe(ComputePipe& p);
bool PushDescriptorsAvailable();

// ---- one frame's work for a backend ----
struct Frame {
	VkCommandBuffer cb = VK_NULL_HANDLE;
	OwnImage*  color = nullptr;   // render size, SHADER_READ_ONLY_OPTIMAL (our copies)
	OwnImage*  depth = nullptr;
	OwnImage*  motion = nullptr;
	OwnImage*  output = nullptr;  // display size, R16G16B16A16_SFLOAT, storage, GENERAL
	VkExtent2D render {}, display {};
	float      jitter[2] {0, 0};     // pixels (FidelityFX convention)
	float      mv_scale[2] {0, 0};   // vectors * mv_scale = render pixels toward the previous frame
	bool       reversed = false;     // depth: 1 near, 0 far
	bool       hdr = false;          // float colour (linear, maybe above 1)
	bool       reset = false;
	int        quality = 2;          // resolved (1..5)
	float      dt_ms = 16.7f;
	camera::State cam;
	uint64_t   index = 0;
};

class Upscaler {
public:
	virtual ~Upscaler() = default;
	virtual const char* Name() const = 0;
	// Records the upscale of f into f.output; false: nothing usable written this frame.
	virtual bool Dispatch(const Frame& f) = 0;
	// For the log: the mode / preset / model actually in use.
	virtual std::string Detail() const { return std::string(); }
};

// ---- backends ----
// Each: what device creation needs, whether it can run here (and why not), and its upscaler.
namespace dlss {
void BeforeCreateInstance();
void InstanceExtensions(std::vector<const char*>& names);
void DeviceExtensions(std::vector<const char*>& names);
void DeviceFeatures(VkPhysicalDeviceVulkan12Features& f12, VkPhysicalDeviceVulkan13Features& f13);
void OnFirstPresent(VkDevice device);
bool Ready(std::string& why);
std::unique_ptr<Upscaler> Create(std::string& why);
void Shutdown();
} // namespace dlss
namespace xess {
void BeforeCreateInstance();
void InstanceExtensions(std::vector<const char*>& names);
void DeviceExtensions(VkPhysicalDevice physical, std::vector<const char*>& names);
void* DeviceFeatureChain(VkPhysicalDevice physical);  // XeSS's own structures (to merge), or null
bool Ready(std::string& why);
std::unique_ptr<Upscaler> Create(std::string& why);
} // namespace xess
namespace fsr4 {
void DeviceExtensions(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& names);
// What the device supports (queried at device creation); sets the bits it needs in the game's structures.
void DeviceFeatures(VkPhysicalDevice physical, VkPhysicalDeviceFeatures& f, VkPhysicalDeviceVulkan12Features& f12,
                    VkPhysicalDeviceVulkan13Features& f13, bool derivatives_ext, bool& want_derivatives);
bool Ready(std::string& why);
std::unique_ptr<Upscaler> Create(std::string& why);
} // namespace fsr4
namespace fsr3 {
bool Ready(std::string& why);
std::unique_ptr<Upscaler> Create(std::string& why);
} // namespace fsr3

// Extensions enabled on the device (for Ready() checks), and its queue count per family.
bool     DeviceHas(const char* extension);
uint32_t QueueCount(uint32_t family);

} // namespace inj::upscale
