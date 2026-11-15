#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export
#include "graphics/host_gpu/renderer/cmdRecorder.h"

#include <map>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

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
	bool                               compute_subgroup_size_control_enabled = false;
	// Compute pipelines may require full subgroups (computeFullSubgroups), each of exactly
	// subgroup_size invocations because the size cannot vary (min == max): what a wave64 program
	// split over two 32-wide subgroups needs (ShaderRecompiler::WaveSplitProblem).
	bool                               compute_full_subgroups_enabled = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               maintenance8_enabled                  = false;
	// KYTY_GPL=1 and VK_EXT_graphics_pipeline_library with fast linking: graphics pipelines from libraries.
	bool                               graphics_pipeline_library_enabled     = false;
	bool                               robust_buffer_access2                 = false;
	uint32_t                           robust_storage_buffer_alignment       = 4; // robustStorageBufferAccessSizeAlignment
	bool                               r8_srgb_views                         = false;
	bool                               r8g8_srgb_views                       = false;
	bool                               conditional_rendering_enabled         = false;
	bool                               sampler_filter_minmax_enabled         = false;
	bool                               occlusion_queries_enabled             = false;
	bool                               occlusion_query_precise_enabled       = false;
	bool                               supports_block_texel_view              = false;
	// VK_KHR_maintenance6 with blockTexelViewCompatibleMultipleLayers: an uncompressed view of a
	// block-compressed image may hold several layers (VUID-VkImageViewCreateInfo-image-09487).
	bool                               block_texel_view_multiple_layers       = false;
	// sparseBinding and sparseResidencyImage2D on a queue that binds sparse: a streamed texture's
	// levels get memory one by one (TextureCache::HoldsLevels).
	bool                               sparse_textures_enabled                = false;
	// The BDA page table is bound only where buffers are (sparse residency; an unbound page reads 0).
	bool                               sparse_page_table                      = false;
	// KYTY_SPARSE_GEOMETRY (on unless 0): game buffers of 1 GiB or more are sparse (BufferCache::CreateBuffer), backed where the guest has
	// memory mapped; the other pages sit on one shared zero page (sparse_buffers_aliased) or unbound (residencyNonResidentStrict).
	bool                               sparse_buffers                         = false;
	bool                               sparse_buffers_aliased                 = false;
	uint64_t                           max_buffer_size                        = 0; // Vulkan 1.3 maxBufferSize
	uint32_t                           sparse_queue_index                     = 0;       // of queue_family
	vk::Queue                          sparse_queue                           = nullptr; // binds only (CommandScheduler)
	bool                                      mesh_shader_enabled                   = false;
	// Upstream abc509d33's task shader for indirect mesh draws (KYTY_UPSTREAM_PATHS).
	bool                                      task_shader_enabled                   = false;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	// The guest image heap (descriptorHeap.h, --bindless) is descriptor set 1 of every guest
	// pipeline layout when the device can hold it; the renderer's heap object fills in the other
	// two. Without it, the resident image heap (imageHeap.h) is set 1 of the pipelines that use it.
	bool                               guest_image_heap_enabled              = false;
	vk::DescriptorSetLayout            guest_image_heap_layout               = nullptr;
	class GuestImageHeap*              guest_image_heap                      = nullptr;
	uint32_t                           device_clock_period_ps                = 0;
	vk::DescriptorSetLayout            image_heap_layout                     = nullptr;
	// No bindings: the set numbers a guest pipeline layout skips below its image groups' sets (IR::GroupSet).
	vk::DescriptorSetLayout            empty_set_layout                      = nullptr;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;

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

	// Sampled views of format filter linearly, as guest samplers may.
	[[nodiscard]] bool SamplesFiltered(vk::Format format) const {
		constexpr vk::FormatFeatureFlags features =
		    vk::FormatFeatureFlagBits::eSampledImage |
		    vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
		return (GetFormatProperties(format).optimalTilingFeatures & features) == features;
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

	// Graphics pipelines from libraries (KYTY_GPL): a pixel shader's resources are set IR::FragmentSet (2), the vertex
	// stages' set 0, so each library's layout holds its own; set 1 must then be the guest image heap (--bindless).
	[[nodiscard]] bool SplitsFragmentSet() const noexcept {
		return graphics_pipeline_library_enabled && guest_image_heap_layout != nullptr;
	}
	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		// Where compute can require 32 (AMD), wave64 compute runs as two 32-wide subgroups, as on NVIDIA. AMD's
		// driver (32.0.21045) crashed compiling chained DPP-row shuffles at subgroup size 64.
		if (compute_subgroup_size_control_enabled && min_subgroup_size <= 32u) return false;
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
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
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;

	struct HeapUsage {
		uint64_t usage        = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		bool     device_local = false;
	};
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	// A sparse image, with no memory: Image binds it block by block.
	[[nodiscard]] bool     CreateSparseImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);
	// Device memory for `size` bytes of a sparse image's blocks, null when there is none.
	[[nodiscard]] VmaAllocation AllocateSparse(const vk::MemoryRequirements& image, uint64_t size,
	                                           vk::DeviceMemory& memory, uint64_t& offset);

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

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
