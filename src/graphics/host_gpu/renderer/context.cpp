#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "live-counters.h"
#include "live-trace.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#ifdef KYTY_LOCAL_VULKAN_RECORDING
#include "vulkan-recording.h"
#endif

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	FlushCopyRun();
	if (m_compute_access_pending) {
		ShaderAccessBarrier(m_buffer, vk::PipelineStageFlagBits::eComputeShader);
		m_compute_access_pending = false;
	}
	return m_buffer;
}

vk::CommandBuffer CommandBuffer::ChainHandle() const {
	EXIT_IF(IsInvalid() || m_rendering);
	FlushCopyRun();
	return m_buffer;
}

void CommandBuffer::FlushRunTail() const {
	if (m_run_tail.empty()) return;
	VulkanMemoryBarrier between {};
	between.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	between.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 1, &between,
	                         0, nullptr, 0, nullptr);
	for (const auto& tail: m_run_tail) {
		const vk::BufferCopy copy {tail.source_offset, tail.destination_offset, tail.size};
		m_buffer.copyBuffer(tail.source, tail.destination, 1, &copy);
	}
	m_run_tail.clear();
	// (The host reads the snapshots after the submission completes.)
	VulkanMemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0,
	                         nullptr, 0, nullptr);
}

void CommandBuffer::CopyAfterRun(vk::Buffer source, uint64_t source_offset, vk::Buffer destination,
                                 uint64_t destination_offset, uint64_t size) const {
	EXIT_IF(IsInvalid() || m_rendering || m_copy_run.empty());
	const VkBuffer src = source, dst = destination;
	m_run_tail.push_back({src, dst, source_offset, destination_offset, size});
	// A later copy of the run that writes what this reads, or touches what this writes, starts a new run first.
	m_copy_run.push_back({src, source_offset, source_offset + size, false});
	m_copy_run.push_back({dst, destination_offset, destination_offset + size, true});
}

void CommandBuffer::FlushCopyRun() const {
	if (m_copy_run.empty()) return;
	m_copy_run.clear();
	FlushRunTail();
	VulkanMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	m_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after,
	                         0, nullptr, 0, nullptr);
}

vk::CommandBuffer CommandBuffer::CopyRunHandle(vk::Buffer source, uint64_t source_offset, vk::Buffer destination,
                                               uint64_t destination_offset, uint64_t size) const {
	EXIT_IF(IsInvalid() || m_rendering);
	const auto meets = [](const CopyRunRange& range, VkBuffer buffer, uint64_t begin, uint64_t end) {
		return range.buffer == buffer && begin < range.end && range.begin < end;
	};
	const VkBuffer src = source, dst = destination;
	bool           opens = m_copy_run.empty();
	for (const auto& range: m_copy_run)
		if ((range.written && meets(range, src, source_offset, source_offset + size)) ||
		    meets(range, dst, destination_offset, destination_offset + size)) {
			opens = true;
			break;
		}
	if (opens) {
		// Everything before (the previous run too) ahead of the run's copies.
		const auto          native = Handle();
		VulkanMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
		                       &before, 0, nullptr, 0, nullptr);
	}
	m_copy_run.push_back({src, source_offset, source_offset + size, false});
	m_copy_run.push_back({dst, destination_offset, destination_offset + size, true});
	return m_buffer;
}

void CommandBuffer::ContinueComputeChain() const {
	EXIT_IF(IsInvalid() || m_rendering);
	m_compute_access_pending = true;
}

vk::CommandBuffer CommandBuffer::HandleForFullBarrier() const {
	EXIT_IF(IsInvalid());
	// The caller records a full AllCommands memory dependency immediately (after the run's owed copies).
	m_compute_access_pending = false;
	if (!m_copy_run.empty()) FlushRunTail();
	m_copy_run.clear();
	return m_buffer;
}

#ifdef KYTY_LOCAL_VULKAN_RECORDING
static void ReplayBegin(std::span<const LocalVulkanRecording::Segment> segments,
                        const vk::detail::DispatchLoaderDynamic& dispatch) {
	const auto command = *static_cast<const VkCommandBuffer*>(segments[0].data);
	VkCommandBufferBeginInfo info {};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	if (dispatch.vkBeginCommandBuffer(command, &info) != VK_SUCCESS) {
		EXIT("deferred vkBeginCommandBuffer failed\n");
	}
	const auto slot = *static_cast<const uint32_t*>(segments[1].data);
	if (slot != UINT32_MAX && LiveTrace::g_timestamp_pool != nullptr) {
		const auto pool = static_cast<VkQueryPool>(LiveTrace::g_timestamp_pool);
		dispatch.vkCmdResetQueryPool(command, pool, slot * 2u, 2u);
		dispatch.vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, slot * 2u);
	}
}
#endif

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	InvalidateGraphicsState();
	m_barrier_work = UINT64_MAX;
	m_copy_run.clear();
	m_run_tail.clear();
	// Not Handle(): nothing may be recorded before the buffer begins.
	const vk::CommandBuffer buffer = m_buffer;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
	// The recording worker owns this pool's buffers while it replays them; begin there too.
	{
		const VkCommandBuffer raw = buffer;
		m_timestamp_slot          = LiveTrace::g_on.load(std::memory_order_relaxed) && LiveTrace::g_timestamp_pool
		                                ? LiveTrace::g_timestamp_next.fetch_add(1) % LiveTrace::TimestampSlots
		                                : UINT32_MAX;
		const LocalVulkanRecording::Segment segments[] {{&raw, sizeof(raw)}, {&m_timestamp_slot, sizeof(m_timestamp_slot)}};
		if (LocalVulkanRecording::EnqueueDeferred(ReplayBegin, segments, false)) {
			return;
		}
	}
#endif

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::InvalidateGraphicsState() const noexcept {
	++m_graphics_generation;
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	if (m_rendering && m_render_state == state) {
		return;
	}
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	LiveCounters::Add(LiveCounters::RenderPasses);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
