#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

// The render targets of a draw, for device-loss reports: guest addresses, extents (width |
// height << 16), host formats (vk::Format values) and the render area.
struct DiagnosticTargets {
	uint64_t color_address = 0;
	uint64_t depth_address = 0;
	uint32_t color_extent  = 0;
	uint32_t depth_extent  = 0;
	uint32_t color_format  = 0;
	uint32_t depth_format  = 0;
	uint32_t area          = 0;
	// Bits 0-7: color attachments; bit 8: color 0 clears; bits 9, 10: depth, stencil clear;
	// bit 11: has stencil.
	uint32_t flags         = 0;
	// The color attachments after the first (slot | samples << 8 in more_slots).
	static constexpr uint32_t        MoreColors = 3;
	std::array<uint64_t, MoreColors> more_addresses {};
	std::array<uint32_t, MoreColors> more_extents {};
	std::array<uint32_t, MoreColors> more_formats {};
	std::array<uint32_t, MoreColors> more_slots {};
};

struct DiagnosticCheckpoint {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint32_t arg0      = 0;
	uint32_t arg1      = 0;
	uint32_t arg2      = 0;
	uint32_t arg3      = 0;
	uint64_t arg4      = 0;
	// Transfers (CommandBuffer::MarkTransfer): the second guest address (a copy's source).
	uint64_t arg5      = 0;
	uint64_t sequence  = 0;
	// Filled in once a draw has chosen its pipeline (CommandBuffer::AnnotateDiagnosticCheckpoint),
	// so a device-loss report names the shaders that were running.
	uint64_t vs_hash       = 0;
	uint64_t ps_hash       = 0;
	uint32_t color_samples = 0;
	uint32_t depth_samples = 0;
	// Bit 0: the pixel stage writes buffers or storage images; bit 1: a vertex stage does.
	uint32_t write_flags = 0;
	DiagnosticTargets targets;
};

struct GraphicContext;

// Records a checkpoint in the ring and returns its slot (for annotations). vkCmdSetCheckpointNV
// takes DiagnosticCheckpointMarker of it, its sequence: a device-loss report then knows when the
// slot has been reused for a later checkpoint.
[[nodiscard]] DiagnosticCheckpoint*
RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint);
[[nodiscard]] inline const void*
DiagnosticCheckpointMarker(const DiagnosticCheckpoint& checkpoint) {
	return reinterpret_cast<const void*>(static_cast<uintptr_t>(checkpoint.sequence));
}
// After a device loss: VK_EXT_device_fault info, the NVIDIA crash dump written to
// _device_fault.nv-gpudmp, the queue's last diagnostic checkpoints, and every checkpoint the GPU
// had started but not finished (with a few before and after), from the ring.
void DumpDeviceLossDiagnostics(GraphicContext& graphics);

// Diagnostics: image views the renderer destroyed (the guest address of the image they viewed,
// 0 when not known). A command that names one (a stale handle) is reported; the Scrub functions
// replace it with a null view, which reads zero and discards writes, instead of leaving it to
// fault the GPU. They change the arrays the pointers name. A view created again with a destroyed
// handle is live again.
void NoteImageViewCreated(vk::ImageView view);
void NoteImageViewDestroyed(vk::ImageView view, uint64_t guest_address, vk::Format format);
// KYTY_FASTDRAW_IMAGESET (gt7-fastdraw-1112): how many image views NoteImageViewDestroyed was told of
// so far (any thread). A descriptor set written while it had a value names live views only while
// it keeps that value (the image set memo, descriptors.cpp).
[[nodiscard]] uint64_t ImageViewDestroyGeneration();
[[nodiscard]] bool DiagnosticViewIsLive(vk::ImageView view);
void               ScrubDeadRenderingViews(const vk::RenderingInfo& info, const char* where);
// Storage image descriptors only.
void ScrubDeadDescriptorViews(const vk::WriteDescriptorSet* writes, uint32_t count,
                              const char* where);

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	// KYTY_ADDRESS_BINDING_REPORT=1: logs every GPU address range bound or unbound, so a
	// device-fault address can be matched to the object that owned it.
	vk::DebugUtilsMessengerEXT         address_binding_messenger             = nullptr;
	bool                               address_binding_report_enabled        = false;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               diagnostic_checkpoints_enabled        = false;
	// With --profile, for the host GPU timeline (renderer/gpuProfiler).
	bool                               host_query_reset_enabled              = false;
	// Round 119: occlusionQueryPrecise, for the guest's occlusion counters (OcclusionQueries).
	bool                               occlusion_query_precise_enabled       = false;
	bool                               calibrated_timestamps_enabled         = false;
	bool                               device_fault_enabled                  = false;
	// pipelineStatisticsQuery: GPU timing's counts for each timed draw (gpuTiming.cpp).
	bool                               pipeline_statistics_enabled           = false;
	bool                               shader_device_clock_enabled           = false;
	bool                               compute_subgroup_size_control_enabled = false;
	// subgroupSizeControl is enabled: the device has more than one subgroup size.
	bool                               subgroup_size_control_enabled         = false;
	bool                               sample_rate_shading_enabled           = false;
	// pipelineCreationCacheControl: a pipeline can be looked up in the driver's caches alone
	// (FAIL_ON_PIPELINE_COMPILE_REQUIRED), PipelineCache's probe (round 87).
	bool pipeline_cache_control_enabled = false;
	// Attachments with different sample counts in one draw (PS5 EQAA: depth at the coverage
	// count, color at the fragment count). AMD allows either side to be larger; NV only allows
	// color <= depth, with rasterization at the depth count.
	enum class MixedAttachmentSamples : uint8_t { None, Amd, Nv };
	MixedAttachmentSamples             mixed_attachment_samples              = MixedAttachmentSamples::None;
	// Whether one draw can attach color and depth at these sample counts (0 color: depth only).
	[[nodiscard]] bool AcceptsMixedSamples(uint32_t color_samples,
	                                       uint32_t depth_samples) const noexcept {
		if (color_samples == 0 || color_samples == depth_samples) {
			return true;
		}
		switch (mixed_attachment_samples) {
			case MixedAttachmentSamples::Amd: return true;
			case MixedAttachmentSamples::Nv: return color_samples < depth_samples;
			case MixedAttachmentSamples::None: return false;
		}
		return false;
	}
	// Round 108: the raster sample count and the depth/stencil and color sample count masks
	// VK_NV_coverage_reduction_mode takes TRUNCATE with (as the device lists them): color sample i
	// is covered by raster sample i alone, as the PS5's color
	// buffer sees only the samples exposed to it. Without it, NV merges the raster samples into
	// the color sample: two triangles sharing an edge then both blend into the pixels along it.
	// Empty: MERGE (KYTY_COVERAGE_REDUCTION=merge, or no such support).
	std::vector<std::array<uint32_t, 3>> coverage_truncate_combinations;
	[[nodiscard]] bool CoverageTruncate(uint32_t raster, uint32_t depth, uint32_t color) const {
		for (const auto& combination: coverage_truncate_combinations) {
			if (combination[0] == raster && (combination[1] & depth) != 0 &&
			    (combination[2] & color) != 0) {
				return true;
			}
		}
		return false;
	}
	bool                               shader_image_int64_atomics_enabled    = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               supports_block_texel_view              = false;
	bool                                      mesh_shader_enabled                   = false;
	// Descriptor indexing for bindless images: runtime arrays indexed non-uniformly, partially
	// bound and updated after bind.
	bool                               bindless_enabled                      = false;
	// Round 116, KYTY_HOST_IMPORT_PROBE=1 (renderer/cache/hostImportProbe.h):
	// VK_EXT_external_memory_host is enabled, and the alignment an imported host pointer and size
	// need.
	bool                                      external_memory_host_enabled          = false;
	uint64_t                                  imported_host_pointer_alignment       = 0;
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
	// The queue for game threads' reads of bytes the GPU has finished writing (BufferCache's fast
	// readback, ReadbackQueue), and its family: the copy engines' (transfer-only) family when the
	// device has one, else the universal family's second queue; null with KYTY_FAST_READBACK=0 or
	// neither. Buffers are shared with a family other than queue_family (Buffer). Only the GPU
	// thread submits to it.
	vk::Queue readback_queue  = nullptr;
	uint32_t  readback_family = static_cast<uint32_t>(-1);
	// Frames presented so far: the clock the caches age their entries by.
	std::atomic<uint64_t>              presented_frames {0};

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

	// The subgroup size a graphics stage must require so that one host subgroup is one guest
	// wave, or 0 when the default already is or the device cannot require it for the stage. On
	// AMD the default is 64; a wave32 pixel or mesh shader would otherwise share a subgroup with
	// another wave, and its ballots, lane reads and reductions would mix the two.
	[[nodiscard]] uint32_t GraphicsSubgroupSize(vk::ShaderStageFlagBits stage,
	                                            uint32_t                wave_size) const noexcept {
		if (!subgroup_size_control_enabled || wave_size == subgroup_size ||
		    !(required_subgroup_size_stages & stage) || wave_size < min_subgroup_size ||
		    wave_size > max_subgroup_size) {
			return 0;
		}
		return wave_size;
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
	// The driver's budget for the device-local heaps (VK_EXT_memory_budget), without the reserve
	// and the video memory target GetTotalMemoryBudget takes off; 0 when it cannot be read.
	[[nodiscard]] uint64_t GetRawMemoryBudget() const;
	// Memory pressure in bytes, read at most every 250 ms; the caches' collectors lower their
	// thresholds by it, and DeviceMemoryAtLeast reports true while it is not 0. The larger of:
	// - the system's free commit below a reserve (KYTY_COMMIT_RESERVE_MB; default 0 = off, see
	//   MemoryShortfall for why). On Windows every GB of video memory in use is also charged to
	//   the commit limit (WDDM backs pageable allocations), so a PC with a small page file runs
	//   out of commit with video memory to spare (ISSUES #21);
	// - everything the device holds past 90 % of the budget, which KYTY_VRAM_TARGET_MB caps
	//   (default 14336: the emulator plans for a 16 GB card whatever card it runs on).
	[[nodiscard]] uint64_t MemoryShortfall() const;
	// Whether everything the device holds (images, buffers, the driver's own) has reached percent
	// of the device-local budget, or memory is short (MemoryShortfall). False when the driver
	// cannot report usage and nothing is short.
	[[nodiscard]] bool DeviceMemoryAtLeast(uint32_t percent) const {
		if (MemoryShortfall() != 0) {
			return true;
		}
		if (!CanReportMemoryUsage()) {
			return false;
		}
		const auto budget = GetTotalMemoryBudget();
		return budget != 0 && GetDeviceMemoryUsage() / 100 >= budget / 100 * percent / 100;
	}
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);

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
