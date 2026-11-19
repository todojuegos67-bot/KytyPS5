#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stallWatch.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()), m_scheduler(scheduler) {}

bool CommandBuffer::IsInvalid() const {
	return !m_open;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	// Round 167: what is recorded directly comes after the batched copies' barrier.
	Recorder().FlushCopyBatch();
	// Round 172: and may change any state: the state shadow holds none after it. State recorded
	// through a handle kept past a Recorder() state call is not seen by the shadow, so the handle
	// is taken again (Handle()) after such calls.
	m_scheduler.StateShadow().InvalidateAll();
	if (m_scheduler.Threaded()) {
		// Recording directly: the recording thread must have run everything queued first.
		return m_scheduler.DrainRecording();
	}
	return m_buffer;
}

CommandRecorder CommandBuffer::Recorder() const {
	EXIT_IF(IsInvalid());
	// The scheduler also when recording directly: its batch of copies (CopyBatchState).
	return CommandRecorder(m_scheduler, m_scheduler.Threaded() ? nullptr : m_buffer);
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	if (m_scheduler.Threaded()) {
		// The recording thread ends it before submitting (CommandScheduler::Submit).
		return;
	}
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

bool CommandBuffer::RecordsCheckpoints() const {
	static const bool always = [] {
		const char* value = std::getenv("KYTY_GPU_CHECKPOINTS");
		return value != nullptr && value[0] == '1';
	}();
	return m_graphics.diagnostic_checkpoints_enabled && (always || Bisect::CheckpointsOn());
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
	m_debug_marker    = nullptr;
	if (RecordsCheckpoints() && !IsInvalid()) {
		auto* marker = RecordDiagnosticCheckpoint({.op        = op,
		                                                 .submit_id = submit_id,
		                                                 .arg0      = arg0,
		                                                 .arg1      = arg1,
		                                                 .arg2      = arg2,
		                                                 .arg3      = arg3,
		                                                 .arg4      = arg4});
		m_debug_marker = marker;
		if (marker != nullptr) {
			Recorder().setCheckpointNV(DiagnosticCheckpointMarker(*marker));
		}
	}
}

void CommandBuffer::MarkTransfer(CommandBufferDebugOp op, uint64_t destination, uint64_t source,
                                 uint32_t arg0, uint32_t arg1, uint32_t arg2) const {
	if (!IsInvalid()) {
		// What this command buffer moves, for the stall reports (renderer/stallWatch.h).
		using StallWatch::Transfer;
		auto     kind  = Transfer::Other;
		uint64_t bytes = 0;
		switch (op) {
			case CommandBufferDebugOp::ImageUpload:
				kind  = Transfer::ImageUpload;
				bytes = arg1;
				break;
			case CommandBufferDebugOp::ImageDownload:
				kind  = Transfer::ImageDownload;
				bytes = arg1;
				break;
			case CommandBufferDebugOp::BufferCopy:
				kind  = Transfer::BufferCopy;
				bytes = arg0;
				break;
			case CommandBufferDebugOp::ImageCopy:
			case CommandBufferDebugOp::ImageResolve:
			case CommandBufferDebugOp::ImageCopyThroughBuffer:
			case CommandBufferDebugOp::ImageMipCopy:
			case CommandBufferDebugOp::DepthCopy: kind = Transfer::ImageCopy; break;
			case CommandBufferDebugOp::InternalWork:
				switch (static_cast<InternalWork>(arg0)) {
					case InternalWork::Detile:
						kind  = Transfer::Detile;
						bytes = arg1;
						break;
					case InternalWork::Tile:
						kind  = Transfer::Tile;
						bytes = arg1;
						break;
					case InternalWork::BufferDownload:
						kind  = Transfer::BufferDownload;
						bytes = arg1;
						break;
					case InternalWork::NanScan:
						kind  = Transfer::NanScan;
						bytes = arg1;
						break;
					default: break;
				}
				break;
			default: break;
		}
		m_scheduler.NoteTransfer(kind, bytes);
	}
	if (!RecordsCheckpoints() || IsInvalid()) {
		return;
	}
	auto* marker = RecordDiagnosticCheckpoint({.op        = static_cast<uint32_t>(op),
	                                           .submit_id = m_debug_submit_id,
	                                           .arg0      = arg0,
	                                           .arg1      = arg1,
	                                           .arg2      = arg2,
	                                           .arg4      = destination,
	                                           .arg5      = source});
	if (marker != nullptr) {
		Recorder().setCheckpointNV(DiagnosticCheckpointMarker(*marker));
	}
}

void CommandBuffer::CarryDiagnosticAnnotation() const noexcept {
	AnnotateDiagnosticCheckpoint(m_annotation_vs, m_annotation_ps, m_annotation_color,
	                             m_annotation_depth, m_annotation_writes);
	AnnotateDiagnosticTargets(m_annotation_targets);
}

void CommandBuffer::AnnotateDiagnosticTargets(const DiagnosticTargets& targets) const noexcept {
	m_annotation_targets = targets;
	if (m_debug_marker != nullptr) {
		m_debug_marker->targets = targets;
	}
}

void CommandBuffer::AnnotateDiagnosticCheckpoint(uint64_t vs_hash, uint64_t ps_hash,
                                                 uint32_t color_samples,
                                                 uint32_t depth_samples,
                                                 uint32_t write_flags) const noexcept {
	m_annotation_vs     = vs_hash;
	m_annotation_ps     = ps_hash;
	m_annotation_color  = color_samples;
	m_annotation_depth  = depth_samples;
	m_annotation_writes = write_flags;
	if (m_debug_marker == nullptr) {
		return;
	}
	m_debug_marker->write_flags   = write_flags;
	m_debug_marker->vs_hash       = vs_hash;
	m_debug_marker->ps_hash       = ps_hash;
	m_debug_marker->color_samples = color_samples;
	m_debug_marker->depth_samples = depth_samples;
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
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
		colors[i].resolveMode             = attachment.resolve_mode;
		colors[i].resolveImageView        = attachment.resolve_image_view;
		colors[i].resolveImageLayout      = attachment.resolve_image_layout;
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
	ScrubDeadRenderingViews(rendering, "a draw's rendering");
	Recorder().beginRendering(rendering);
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	for (auto& color: m_render_state.color_attachments) {
		color.is_clear = false;
	}
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	EmitStats::NoteEndRendering(); // gt7-emit-1119
	Recorder().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
