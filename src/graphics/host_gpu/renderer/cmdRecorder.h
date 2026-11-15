#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORDER_H_

#include "common/packetStream.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics {

class CommandScheduler;

// What a recorded packet runs against on the recording thread.
struct CmdBack {
	vk::CommandBuffer command   = nullptr; // the command buffer being recorded
	vk::Device        device    = nullptr;
	CommandScheduler* scheduler = nullptr;
};

// The Vulkan commands of the GPU thread, with vk::CommandBuffer's names and arguments (the forms this code uses).
// Direct, it calls the command buffer at once. On a stream (KYTY_RECORD_THREAD, record-thread design H1) it copies
// the arguments and every array they point to into a packet, and the recording thread makes the same call, in the
// same order, later. Nothing passed by pointer is used after a method returns. Structures with a pNext chain are
// refused on a stream (none of the GPU thread's commands has one).
class Cmd {
public:
	// NOLINTNEXTLINE(google-explicit-constructor): a raw command buffer is a direct recorder.
	Cmd(vk::CommandBuffer direct): m_direct(direct) {}
	explicit Cmd(Common::PacketStream& stream): m_stream(&stream) {}

	[[nodiscard]] bool              Direct() const noexcept { return m_stream == nullptr; }
	// No command buffer to record into (a direct recorder of a null handle).
	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept { return m_stream == nullptr && !m_direct; }
	[[nodiscard]] vk::CommandBuffer DirectHandle() const noexcept { return m_direct; }

	// Barriers
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst, vk::DependencyFlags flags,
	                     uint32_t memory_count, const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const;
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const;

	// Rendering
	void beginRendering(const vk::RenderingInfo& rendering) const;
	void beginRendering(const vk::RenderingInfo* rendering) const { beginRendering(*rendering); }
	void endRendering() const;
	void clearAttachments(uint32_t count, const vk::ClearAttachment* attachments, uint32_t rect_count,
	                      const vk::ClearRect* rects) const;

	// Binding
	void bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) const;
	void bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t first, uint32_t count,
	                        const vk::DescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic) const;
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const;
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset, uint32_t size,
	                   const void* values) const;
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets,
	                        const vk::DeviceSize* sizes, const vk::DeviceSize* strides) const;
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const;

	// Draws and dispatches
	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex, uint32_t first_instance) const;
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index, int32_t vertex_offset,
	                 uint32_t first_instance) const;
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const;
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const;
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const;
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const;
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const;
	void dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t x, uint32_t y, uint32_t z) const;
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const;

	// Transfers
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count, const vk::BufferCopy* regions) const;
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count, const vk::ImageCopy* regions) const;
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination, uint32_t count,
	                       const vk::BufferImageCopy* regions) const;
	void blitImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count, const vk::ImageBlit* regions,
	               vk::Filter filter) const;
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, uint32_t count, const vk::ImageResolve* regions) const;
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t value) const;
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color, uint32_t count,
	                     const vk::ImageSubresourceRange* ranges) const;
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue* value,
	                            uint32_t count, const vk::ImageSubresourceRange* ranges) const;

	// The same with vk::ArrayProxy (one element, an array, a vector), as vk::CommandBuffer has them.
	void pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set,
	                          vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSetKHR(point, layout, set, writes.size(), writes.data());
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
		copyImage(source, source_layout, destination, destination_layout, regions.size(), regions.data());
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
		resolveImage(source, source_layout, destination, destination_layout, regions.size(), regions.data());
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyBufferToImage(source, destination, layout, regions.size(), regions.data());
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyImageToBuffer(source, layout, destination, regions.size(), regions.data());
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
	                     vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
		clearColorImage(image, layout, &color, ranges.size(), ranges.data());
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue& value,
	                            vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
		clearDepthStencilImage(image, layout, &value, ranges.size(), ranges.data());
	}

	// Conditional rendering
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const;
	void endConditionalRenderingEXT() const;

	// Queries and markers
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const;
	void endQuery(vk::QueryPool pool, uint32_t query) const;
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const;
	void writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query) const;
	void setCheckpointNV(const void* marker) const;

	// Dynamic state
	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const;
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const;
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const;
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const;
	void setDepthTestEnable(vk::Bool32 enable) const;
	void setDepthWriteEnable(vk::Bool32 enable) const;
	void setDepthCompareOp(vk::CompareOp op) const;
	void setDepthBoundsTestEnable(vk::Bool32 enable) const;
	void setDepthBounds(float min, float max) const;
	void setDepthBiasEnable(vk::Bool32 enable) const;
	void setDepthBias(float constant, float clamp, float slope) const;
	void setStencilTestEnable(vk::Bool32 enable) const;
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
	                  vk::CompareOp compare) const;
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const;
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const;
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const;
	void setCullMode(vk::CullModeFlags mode) const;
	void setFrontFace(vk::FrontFace face) const;
	void setLineWidth(float width) const;
	void setBlendConstants(const float constants[4]) const;
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const;
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const;

	// Host: vkUpdateDescriptorSets in the command order (update-after-bind sets read at submit, design rule 5).
	void UpdateDescriptorSets(vk::Device device, uint32_t count, const vk::WriteDescriptorSet* writes) const;

	// A command of a fixed payload P that the recording thread runs as F{}(command buffer, payload).
	template <typename P, typename F>
	void Put(const P& payload, F /*run*/) const {
		static_assert(std::is_trivially_copyable_v<P> && std::is_empty_v<F>);
		auto* data = m_stream->Append(&RunPut<P, F>, sizeof(P));
		std::memcpy(data, &payload, sizeof(P));
	}

	// The same, run as F{}(CmdBack&, payload): for the scheduler's own packets (begin, submit, sparse sends).
	template <typename P, typename F>
	void PutBack(const P& payload, F /*run*/) const {
		static_assert(std::is_trivially_copyable_v<P> && std::is_empty_v<F>);
		auto* data = m_stream->Append(&RunPutBack<P, F>, sizeof(P));
		std::memcpy(data, &payload, sizeof(P));
	}

private:
	template <typename P, typename F>
	static void RunPut(const std::byte* data, void* context) {
		P payload;
		std::memcpy(&payload, data, sizeof(P));
		F {}(static_cast<CmdBack*>(context)->command, payload);
	}
	template <typename P, typename F>
	static void RunPutBack(const std::byte* data, void* context) {
		P payload;
		std::memcpy(&payload, data, sizeof(P));
		F {}(*static_cast<CmdBack*>(context), payload);
	}

	Common::PacketStream* m_stream = nullptr;
	vk::CommandBuffer     m_direct = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CMDRECORDER_H_
