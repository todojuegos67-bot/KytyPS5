#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <vector>
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
	return m_buffer;
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
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

namespace {
struct RecentCommand {
	uint32_t op, arg0, arg1, arg2, arg3;
	uint64_t submit_id, arg4;
};
constexpr uint32_t                       RecentCommandCount = 48;
std::array<RecentCommand, RecentCommandCount> g_recent_commands {};
std::atomic<uint32_t>                    g_recent_command_next {0};
} // namespace

void ReportGpuDeviceLost(GraphicContext& graphics, const char* where) {
	static std::atomic<bool> reported {false};
	if (reported.exchange(true)) return;
	std::printf("DeviceLost: detected in %s\n", where);
	if (graphics.device_fault_enabled && VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT != nullptr) {
		VkDeviceFaultCountsEXT counts {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
		const VkDevice device = static_cast<VkDevice>(graphics.device);
		if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, nullptr) == VK_SUCCESS) {
			std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
			std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
			VkDeviceFaultInfoEXT info {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
			info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
			info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
			counts.vendorBinarySize = 0;
			const auto r = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, &info);
			std::printf("DeviceFault: result=%d description=\"%s\" addresses=%u vendor=%u\n", static_cast<int>(r),
			            info.description, counts.addressInfoCount, counts.vendorInfoCount);
			for (const auto& a: addresses) {
				std::printf("DeviceFault:   address type=%d addr=0x%016llx precision=0x%llx\n",
				            static_cast<int>(a.addressType), static_cast<unsigned long long>(a.reportedAddress),
				            static_cast<unsigned long long>(a.addressPrecision));
			}
			for (const auto& v: vendors) {
				std::printf("DeviceFault:   vendor \"%s\" code=0x%llx data=0x%llx\n", v.description,
				            static_cast<unsigned long long>(v.vendorFaultCode),
				            static_cast<unsigned long long>(v.vendorFaultData));
			}
		} else {
			std::printf("DeviceFault: vkGetDeviceFaultInfoEXT failed\n");
		}
	} else {
		std::printf("DeviceFault: VK_EXT_device_fault not available\n");
	}
	const auto next = g_recent_command_next.load(std::memory_order_relaxed);
	const auto count = std::min(next, RecentCommandCount);
	std::printf("DeviceLost: last %u recorded commands (op 0=dispatch 1=draw 2=draw-auto 9=dispatch-indirect), oldest first:\n", count);
	for (uint32_t i = next - count; i != next; ++i) {
		const auto& e = g_recent_commands[i % RecentCommandCount];
		std::printf("DeviceLost:   submit=%llu op=%u args=%u,%u,%u,%u addr=0x%016llx\n",
		            static_cast<unsigned long long>(e.submit_id), e.op, e.arg0, e.arg1, e.arg2, e.arg3,
		            static_cast<unsigned long long>(e.arg4));
	}
	std::fflush(stdout);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	g_recent_commands[g_recent_command_next.fetch_add(1, std::memory_order_relaxed) % RecentCommandCount] = {
	    op, arg0, arg1, arg2, arg3, submit_id, arg4};
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
