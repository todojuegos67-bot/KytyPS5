#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct DiagnosticCheckpoint {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint32_t arg0      = 0;
	uint32_t arg1      = 0;
	uint32_t arg2      = 0;
	uint32_t arg3      = 0;
	uint64_t arg4      = 0;
	uint64_t sequence  = 0;
};

struct GraphicContext;

[[nodiscard]] const DiagnosticCheckpoint*
RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint);
void DumpDeviceLossDiagnostics(GraphicContext& graphics);

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	// VK_EXT_device_address_binding_report (addressBindingReport.h).
	vk::DebugUtilsMessengerEXT         address_binding_messenger             = nullptr;
	bool                               address_binding_report_enabled        = false;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               diagnostic_checkpoints_enabled        = false;
	bool                               device_fault_enabled                  = false;
	bool                               shader_device_clock_enabled           = false;
	bool                               compute_subgroup_size_control_enabled = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               maintenance8_enabled                  = false;
	bool                               supports_block_texel_view              = false;
	bool                                      mesh_shader_enabled                   = false;
	// Descriptor indexing for bindless images: runtime arrays indexed non-uniformly, partially
	// bound and updated after bind.
	bool                               bindless_enabled                      = false;
	uint32_t                           bindless_max_sampled_images           = 0;
	uint32_t                           bindless_max_samplers                 = 0;
	vk::DescriptorSetLayout            bindless_layout                       = nullptr;
	vk::DescriptorSet                  bindless_set                          = nullptr;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	// Queue 1 of queue_family, if the device has it; used by the GPU thread only (BufferCache).
	uint32_t              queue_count    = 1;
	vk::Queue             readback_queue = nullptr;
	std::atomic<uint64_t> presented_frames {0};

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

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
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
	// What the driver currently lets this process keep in device-local memory (no safety margin,
	// unlike GetTotalMemoryBudget); 0 when it cannot report it.
	[[nodiscard]] uint64_t GetDriverMemoryBudget() const;
	// Device-local bytes VMA has handed out, and the bytes of the memory blocks holding them
	// (the difference is free space inside blocks).
	void GetDeviceAllocationStats(uint64_t& allocation_bytes, uint64_t& block_bytes) const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;

	struct HeapUsage {
		uint64_t usage        = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		bool     device_local = false;
	};
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	// KYTY_IMAGE_RECYCLE_MB (see vma.cpp): deleted images kept for reuse by an identical
	// CreateImage, instead of a driver allocation each time.
	struct PooledImage {
		std::tuple<vk::ImageCreateFlags, vk::ImageType, vk::Format, uint32_t, uint32_t, uint32_t,
		           uint32_t, uint32_t, vk::SampleCountFlagBits, vk::ImageTiling,
		           vk::ImageUsageFlags, vk::SharingMode>
		                                      key;
		vk::Image                             image      = nullptr;
		VmaAllocation                         allocation = nullptr;
		uint64_t                              bytes      = 0;
		std::chrono::steady_clock::time_point parked;
	};
	[[nodiscard]] static decltype(PooledImage::key) PoolKey(const vk::ImageCreateInfo& info);
	// Destroys parked images beyond the byte cap, older than the age limit, or all of them.
	void TrimImagePool(uint64_t cap_bytes, bool all);

	std::mutex               m_image_pool_mutex;
	std::vector<PooledImage> m_image_pool;
	uint64_t                 m_image_pool_bytes = 0;

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
	// The rest of the create info, for KYTY_IMAGE_RECYCLE_MB: only an image created without a
	// pNext chain may be parked and handed to an identical CreateImage.
	vk::ImageTiling tiling     = vk::ImageTiling::eOptimal;
	vk::SharingMode sharing    = vk::SharingMode::eExclusive;
	bool            recyclable = false;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
