#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;
class PipelineBinaries;
class PipelineBinaryWriter;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	// VK_EXT_memory_priority, enabled only for KYTY_SIMULATE_VRAM_MB (vma.cpp).
	bool                               memory_priority_enabled               = false;
	bool                               rt_extensions_enabled                 = false;
	bool                               compute_subgroup_size_control_enabled = false;
	// With subgroup size control: wave64 compute programs run on 64-lane subgroups (else emulated on 32-lane ones, two
	// guest lanes an invocation). VulkanInitSubgroupSizeControl decides it (KYTY_COMPUTE_WAVE64).
	bool                               compute_wave64_native                 = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               supports_block_texel_view              = false;
	// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT is usable.
	bool                               pipeline_cache_control_enabled        = false;
	// The static precompile's pipelines (PipelineCache): looked up first when a pipeline is
	// created, never compiled into outside the precompile.
	vk::PipelineCache                  static_pipeline_cache                 = nullptr;
	// VK_KHR_pipeline_binary (and maintenance5's create flags) is enabled: the static precompile's
	// pipelines are its binaries instead (pipelineBinaries.h); in the precompile, its output.
	bool                               pipeline_binaries_enabled             = false;
	// VK_NV_copy_memory_indirect: an upload prologue's copies are one command (BufferCache::UploadDirtyRanges).
	bool                               copy_memory_indirect_enabled          = false;
	// False (the emulator): the driver keeps no copy of its own of the pipelines it makes (the
	// precompile's captures of static cache hits come from that copy).
	bool                               pipeline_binary_internal_cache        = true;
	const PipelineBinaries*            pipeline_binaries                     = nullptr;
	PipelineBinaryWriter*              pipeline_binary_writer                = nullptr;
	bool                                      mesh_shader_enabled                   = false;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	// Fragment shaders support subgroup vote/arithmetic (LOD statistics reduction).
	bool                               fragment_subgroup_reduction           = false;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	// KYTY_READBACK_QUEUE (set at device creation): a transfer-only queue for guest readback
	// copies (src/local/readback-queue.h); buffers are then shared with its family.
	uint32_t                           readback_family = static_cast<uint32_t>(-1);
	vk::Queue                          readback_queue  = nullptr;
	// Windows: a second queue of the graphics family used only for vkQueuePresentKHR. The
	// driver can block a present in the kernel until earlier GPU work on its queue is done;
	// on the graphics queue that held queue_mutex against the very submissions that work
	// waited for. Null: presents use `queue` under queue_mutex.
	Common::Mutex                      present_queue_mutex;
	vk::Queue                          present_queue = nullptr;
	bool                               present_queue_created = false; // queue index 1 of queue_family

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] const vk::PhysicalDeviceMemoryProperties&
	GetPhysicalDeviceMemoryProperties() const {
		return physical_device_memory_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	// Wave64 compute programs run natively: chosen with subgroup size control, else when every subgroup has 64 lanes.
	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return compute_subgroup_size_control_enabled ? compute_wave64_native : subgroup_size == 64u;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetHostMemoryUsage() const;
	// The driver's usage figures read again (VMA otherwise refreshes them every 30 of its own allocations: memory
	// the driver takes itself, such as pipelines, shows late).
	void                   RefreshMemoryBudget();
	// Windows' figures for the process's video memory (LocalPlatform::QueryVideoMemory) without the simulation's
	// ballast; false where there are none.
	[[nodiscard]] bool     QueryLocalVideoMemory(uint64_t* usage, uint64_t* budget) const;
	// A log line on where the video memory goes (Windows' and VMA's figures): diagnostics.
	void                   LogVideoMemory(const char* label) const;
	// capped: the caches' budget (KYTY_VRAM_BUDGET_MB, room kept for Windows); false: what the driver allows.
	[[nodiscard]] uint64_t GetTotalMemoryBudget(bool capped = true) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);
	// Frees the images kept for reuse (KYTY_IMAGE_POOL): video memory ran out.
	void                   TrimImagePool();
	void                   ReportMemoryFallback(const char* what, uint64_t bytes) const;
	// KYTY_SIMULATE_VRAM_MB: the video memory taken at start-up (left out of the budget and usage above).
	// A budget under 12 GiB (CreateAllocator): less than the game's working set (Boletaria ~11 GB). Textures yield video
	// memory to render targets and buffers (CreateImage); the pipeline warmup leaves its pipelines to first use.
	bool                          small_video_memory = false;
	// An image only sampled and copied (CreateImage places these apart on such a GPU).
	[[nodiscard]] static bool     IsTextureUsage(vk::ImageUsageFlags usage) {
		return !(usage & (vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment |
		                  vk::ImageUsageFlagBits::eStorage));
	}
	std::vector<vk::DeviceMemory> simulation_ballast;
	uint64_t                      simulation_ballast_bytes = 0;
	void                   AppendHardwareRayTracingDeviceExtensions(
	    const std::vector<vk::ExtensionProperties>& available_extensions,
	    std::vector<const char*>&                   device_extensions);
	void LoadHardwareRayTracingFunctions() const;

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
	bool                    guest       = false; // set by a guest draw's or dispatch's bindings (a transit group)
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
};



// Offline tools: the emulator's device without a window (vulkanWindow.cpp).
bool CreateHeadlessGraphicContext(GraphicContext& graphic_ctx);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
