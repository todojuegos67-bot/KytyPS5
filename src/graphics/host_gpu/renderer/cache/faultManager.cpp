#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "live-counters.h"
#include "live-switch.h"
#include "live-trace-gpu.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults    = 1024;
// Bitmap words each invocation of fault_buffer_process.comp reads (FAULT_WORDS_PER_INVOCATION in CMakeLists.txt).
constexpr uint64_t FaultWordsPerInvocation = 16;
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::CACHING_NUMPAGES / 8),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * PageFaultAreaSize),
      m_granule_buffer(graphics, scheduler, MemoryUsage::Upload, 0, AllFlags,
                       MaxPendingFaults * MaxGranules * sizeof(uint32_t)) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = std::size(FAULT_BUFFER_PROCESS_SPV) * sizeof(uint32_t);
	module_info.pCode    = FAULT_BUFFER_PROCESS_SPV;
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(m_graphics.device.createShaderModule(&module_info, nullptr, &module),
	                     "create fault-buffer shader module");

	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_fault_process_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	// (The subgroup size the game's compute pipelines require here: GraphicContext::InternalComputeSubgroupSize.)
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};
	subgroup.requiredSubgroupSize = m_graphics.InternalComputeSubgroupSize();
	if (subgroup.requiredSubgroupSize != 0) stage.pNext = &subgroup;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

void FaultManager::ProcessFaultBuffer() {
	const LiveTrace::MarkSite site(LiveTrace::SiteFaultProcess);
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, PageFaultAreaSize);
	m_download_buffer.Flush(offset, PageFaultAreaSize);

	// Only the bitmap's uvec4s over committed guest memory (one per 2 MiB granule: Demon's Souls commits ~11 GB of the
	// bitmap's 1 TiB). A shader's read elsewhere is zero whether or not a buffer is made there, and its bit stays set
	// until the memory is committed (then scanned with the rest).
	static_assert(LibKernel::Memory::CommittedGranuleBits == BufferCache::CACHING_PAGEBITS + 7, "a uvec4 is 128 pages");
	(void)LibKernel::Memory::CommittedGranules(&m_granule_version, &m_granules);
	static auto& listed       = LiveSwitch::Find("fault_granules", 1); // (0: the whole bitmap, for A/B)
	const auto granule_count  = m_granules.size() <= MaxGranules && listed.load(std::memory_order_relaxed) != 0
	                                ? static_cast<uint32_t>(m_granules.size())
	                                : 0u;
	const auto granule_offset = m_current_area * MaxGranules * sizeof(uint32_t);
	if (granule_count != 0 && m_area_granule_versions[m_current_area] != m_granule_version) {
		std::memcpy(m_granule_buffer.Mapped().data() + granule_offset, m_granules.data(), granule_count * sizeof(uint32_t));
		m_granule_buffer.Flush(granule_offset, granule_count * sizeof(uint32_t));
		m_area_granule_versions[m_current_area] = m_granule_version;
	}

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, m_fault_buffer.Size()},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	    {m_granule_buffer.Handle(), granule_offset, MaxGranules * sizeof(uint32_t)},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	command.pushConstants(m_fault_process_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(granule_count),
	                      &granule_count);
	const auto num_threads    = granule_count != 0 ? granule_count
	                                               : BufferCache::CACHING_NUMPAGES / 32 / FaultWordsPerInvocation;
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	const auto area = m_current_area;
	m_scheduler.DeferOperation([this, mapped, offset, area] {
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		// The parser counts every record but stores MaxPageFaults - 1 (fault_buffer_process.comp): past them the area
		// holds no records of this batch (another batch's, or nothing), and those pages fault again next time.
		const auto count = std::min<uint32_t>(static_cast<uint32_t>(faults[0]), MaxPageFaults - 1);
		LiveCounters::Add(LiveCounters::BdaFaultPages, count);
		for (uint32_t index = 1; index <= count; ++index) {
			fault_ranges.Add(faults[index], BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", faults[index]);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			// Unmapped since the GPU recorded the access (GpuResourceManager::UnmapMemory no
			// longer waits for this processing): no buffer over memory that is gone.
			if (!m_buffer_cache.IsGpuMapped(start, end - start)) return;
			m_buffer_cache.GetBuffer(m_buffer_cache.FindBuffer(start, end - start)).bda_read = true;
		});
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics
