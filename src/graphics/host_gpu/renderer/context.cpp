#include "common/assert.h"
#include "common/upstreamPaths.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()), m_scheduler(scheduler) {}

vk::CommandBuffer CommandBuffer::Handle(std::source_location where) const {
	EXIT_IF(IsInvalid());
	if (m_scheduler.Threaded()) {
		return m_scheduler.DrainRecording(where.file_name(), where.line());
	}
	return m_buffer;
}

Cmd CommandBuffer::Recorder() const {
	EXIT_IF(IsInvalid());
	return m_scheduler.Recorder();
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid() || m_scheduler.Threaded());
	auto buffer = m_buffer;

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	if (m_scheduler.Threaded()) {
		return; // the recording thread ends it before it submits it
	}
	auto buffer = m_buffer;

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	// Upstream 348789de4 drops these stores; with it named, a device-loss report has no last operation.
	if (KYTY_UPSTREAM_PATH("348789de4")) {
		return;
	}
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF((state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	         state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX) &&
	        !KYTY_UPSTREAM_PATH("e827c92a2"));
	if (m_rendering) {
		EndRendering();
	}

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
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
	Recorder().beginRendering(rendering);
	m_render_state = state;
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	++m_end_rendering_calls;
	if (!m_rendering) {
		return;
	}
	Recorder().endRendering();
	m_rendering = false;
}

} // namespace Libs::Graphics
