#include "graphics/host_gpu/renderer/cmdRecorder.h"

#include "common/assert.h"

#include <array>
#include <new>

namespace Libs::Graphics {

namespace {

constexpr size_t Aligned(size_t bytes) {
	return (bytes + Common::PacketStream::Align - 1u) & ~(Common::PacketStream::Align - 1u);
}

template <typename T>
constexpr size_t ArrayBytes(const T* data, size_t count) {
	return data != nullptr ? Aligned(sizeof(T) * count) : 0u;
}

CmdBack& Back(void* context) {
	return *static_cast<CmdBack*>(context);
}

void NoChain(const void* next, const char* what) {
	if (next != nullptr) {
		EXIT("the recording thread does not copy pNext chains (%s)\n", what);
	}
}

// A packet's payload: the fixed part P, then the arrays its pointers point at.
template <typename P>
struct Packet {
	P*         fixed;
	std::byte* tail;

	template <typename T>
	const T* Copy(const T* data, size_t count) {
		if (data == nullptr || count == 0) return nullptr;
		auto* copy = reinterpret_cast<T*>(tail);
		std::memcpy(static_cast<void*>(copy), data, sizeof(T) * count);
		tail += Aligned(sizeof(T) * count);
		return copy;
	}
};

template <typename P>
Packet<P> Begin(Common::PacketStream& stream, Common::PacketStream::Run run, size_t array_bytes) {
	auto* data  = stream.Append(run, Aligned(sizeof(P)) + array_bytes);
	auto* fixed = new (data) P {};
	return {fixed, data + Aligned(sizeof(P))};
}

template <typename P>
const P& Fixed(const std::byte* data) {
	return *std::launder(reinterpret_cast<const P*>(data));
}

// Where a write's descriptors are, by its type.
enum class InfoKind { Image, Buffer, TexelView };
InfoKind KindOf(vk::DescriptorType type) {
	switch (type) {
		case vk::DescriptorType::eSampler:
		case vk::DescriptorType::eCombinedImageSampler:
		case vk::DescriptorType::eSampledImage:
		case vk::DescriptorType::eStorageImage:
		case vk::DescriptorType::eInputAttachment: return InfoKind::Image;
		case vk::DescriptorType::eUniformBuffer:
		case vk::DescriptorType::eStorageBuffer:
		case vk::DescriptorType::eUniformBufferDynamic:
		case vk::DescriptorType::eStorageBufferDynamic: return InfoKind::Buffer;
		case vk::DescriptorType::eUniformTexelBuffer:
		case vk::DescriptorType::eStorageTexelBuffer: return InfoKind::TexelView;
		default: EXIT("the recording thread does not copy descriptors of type %s\n", vk::to_string(type).c_str());
	}
	return InfoKind::Image;
}

size_t WritesBytes(uint32_t count, const vk::WriteDescriptorSet* writes) {
	size_t bytes = ArrayBytes(writes, count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& write = writes[i];
		NoChain(write.pNext, "VkWriteDescriptorSet");
		switch (KindOf(write.descriptorType)) {
			case InfoKind::Image: bytes += ArrayBytes(write.pImageInfo, write.descriptorCount); break;
			case InfoKind::Buffer: bytes += ArrayBytes(write.pBufferInfo, write.descriptorCount); break;
			case InfoKind::TexelView: bytes += ArrayBytes(write.pTexelBufferView, write.descriptorCount); break;
		}
	}
	return bytes;
}

template <typename P>
const vk::WriteDescriptorSet* CopyWrites(Packet<P>& packet, uint32_t count, const vk::WriteDescriptorSet* writes) {
	auto* copy = const_cast<vk::WriteDescriptorSet*>(packet.Copy(writes, count));
	for (uint32_t i = 0; i < count; i++) {
		auto& write = copy[i];
		switch (KindOf(write.descriptorType)) {
			case InfoKind::Image:
				write.pImageInfo       = packet.Copy(writes[i].pImageInfo, write.descriptorCount);
				write.pBufferInfo      = nullptr;
				write.pTexelBufferView = nullptr;
				break;
			case InfoKind::Buffer:
				write.pBufferInfo      = packet.Copy(writes[i].pBufferInfo, write.descriptorCount);
				write.pImageInfo       = nullptr;
				write.pTexelBufferView = nullptr;
				break;
			case InfoKind::TexelView:
				write.pTexelBufferView = packet.Copy(writes[i].pTexelBufferView, write.descriptorCount);
				write.pImageInfo       = nullptr;
				write.pBufferInfo      = nullptr;
				break;
		}
	}
	return copy;
}

} // namespace

void Cmd::pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst, vk::DependencyFlags flags,
                          uint32_t memory_count, const vk::MemoryBarrier* memory, uint32_t buffer_count,
                          const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
                          const vk::ImageMemoryBarrier* images) const {
	if (Direct()) {
		m_direct.pipelineBarrier(src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
		return;
	}
	for (uint32_t i = 0; i < memory_count; i++) NoChain(memory[i].pNext, "VkMemoryBarrier");
	for (uint32_t i = 0; i < buffer_count; i++) NoChain(buffers[i].pNext, "VkBufferMemoryBarrier");
	for (uint32_t i = 0; i < image_count; i++) NoChain(images[i].pNext, "VkImageMemoryBarrier");
	struct P {
		vk::PipelineStageFlags        src, dst;
		vk::DependencyFlags           flags;
		uint32_t                      memory_count, buffer_count, image_count;
		const vk::MemoryBarrier*      memory;
		const vk::BufferMemoryBarrier* buffers;
		const vk::ImageMemoryBarrier* images;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.pipelineBarrier(p.src, p.dst, p.flags, p.memory_count, p.memory, p.buffer_count,
		                                          p.buffers, p.image_count, p.images);
	    },
	    ArrayBytes(memory, memory_count) + ArrayBytes(buffers, buffer_count) + ArrayBytes(images, image_count));
	*packet.fixed = {src,
	                 dst,
	                 flags,
	                 memory_count,
	                 buffer_count,
	                 image_count,
	                 packet.Copy(memory, memory_count),
	                 packet.Copy(buffers, buffer_count),
	                 packet.Copy(images, image_count)};
}

void Cmd::pipelineBarrier2(const vk::DependencyInfo& dependency) const {
	if (Direct()) {
		m_direct.pipelineBarrier2(dependency);
		return;
	}
	NoChain(dependency.pNext, "VkDependencyInfo");
	for (uint32_t i = 0; i < dependency.memoryBarrierCount; i++)
		NoChain(dependency.pMemoryBarriers[i].pNext, "VkMemoryBarrier2");
	for (uint32_t i = 0; i < dependency.bufferMemoryBarrierCount; i++)
		NoChain(dependency.pBufferMemoryBarriers[i].pNext, "VkBufferMemoryBarrier2");
	for (uint32_t i = 0; i < dependency.imageMemoryBarrierCount; i++)
		NoChain(dependency.pImageMemoryBarriers[i].pNext, "VkImageMemoryBarrier2");
	using P     = vk::DependencyInfo;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) { Back(context).command.pipelineBarrier2(Fixed<P>(data)); },
	    ArrayBytes(dependency.pMemoryBarriers, dependency.memoryBarrierCount) +
	        ArrayBytes(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount) +
	        ArrayBytes(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount));
	auto& copy                 = *packet.fixed;
	copy                       = dependency;
	copy.pMemoryBarriers       = packet.Copy(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
	copy.pBufferMemoryBarriers = packet.Copy(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
	copy.pImageMemoryBarriers  = packet.Copy(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
}

void Cmd::beginRendering(const vk::RenderingInfo& rendering) const {
	if (Direct()) {
		m_direct.beginRendering(rendering);
		return;
	}
	NoChain(rendering.pNext, "VkRenderingInfo");
	for (uint32_t i = 0; i < rendering.colorAttachmentCount; i++)
		NoChain(rendering.pColorAttachments[i].pNext, "VkRenderingAttachmentInfo");
	if (rendering.pDepthAttachment != nullptr) NoChain(rendering.pDepthAttachment->pNext, "VkRenderingAttachmentInfo");
	if (rendering.pStencilAttachment != nullptr)
		NoChain(rendering.pStencilAttachment->pNext, "VkRenderingAttachmentInfo");
	using P     = vk::RenderingInfo;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) { Back(context).command.beginRendering(Fixed<P>(data)); },
	    ArrayBytes(rendering.pColorAttachments, rendering.colorAttachmentCount) +
	        ArrayBytes(rendering.pDepthAttachment, 1) + ArrayBytes(rendering.pStencilAttachment, 1));
	auto& copy              = *packet.fixed;
	copy                    = rendering;
	copy.pColorAttachments  = packet.Copy(rendering.pColorAttachments, rendering.colorAttachmentCount);
	copy.pDepthAttachment   = packet.Copy(rendering.pDepthAttachment, 1);
	copy.pStencilAttachment = packet.Copy(rendering.pStencilAttachment, 1);
}

void Cmd::endRendering() const {
	if (Direct()) {
		m_direct.endRendering();
		return;
	}
	struct P {};
	Put(P {}, [](vk::CommandBuffer command, const P&) { command.endRendering(); });
}

void Cmd::clearAttachments(uint32_t count, const vk::ClearAttachment* attachments, uint32_t rect_count,
                           const vk::ClearRect* rects) const {
	if (Direct()) {
		m_direct.clearAttachments(count, attachments, rect_count, rects);
		return;
	}
	struct P {
		uint32_t                   count, rect_count;
		const vk::ClearAttachment* attachments;
		const vk::ClearRect*       rects;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.clearAttachments(p.count, p.attachments, p.rect_count, p.rects);
	    },
	    ArrayBytes(attachments, count) + ArrayBytes(rects, rect_count));
	*packet.fixed = {count, rect_count, packet.Copy(attachments, count), packet.Copy(rects, rect_count)};
}

void Cmd::bindPipeline(vk::PipelineBindPoint point, vk::Pipeline pipeline) const {
	if (Direct()) {
		m_direct.bindPipeline(point, pipeline);
		return;
	}
	struct P {
		vk::PipelineBindPoint point;
		vk::Pipeline          pipeline;
	};
	Put(P {point, pipeline}, [](vk::CommandBuffer command, const P& p) { command.bindPipeline(p.point, p.pipeline); });
}

void Cmd::bindDescriptorSets(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t first, uint32_t count,
                             const vk::DescriptorSet* sets, uint32_t dynamic_count, const uint32_t* dynamic) const {
	if (Direct()) {
		m_direct.bindDescriptorSets(point, layout, first, count, sets, dynamic_count, dynamic);
		return;
	}
	struct P {
		vk::PipelineBindPoint    point;
		vk::PipelineLayout       layout;
		uint32_t                 first, count, dynamic_count;
		const vk::DescriptorSet* sets;
		const uint32_t*          dynamic;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.bindDescriptorSets(p.point, p.layout, p.first, p.count, p.sets, p.dynamic_count,
		                                             p.dynamic);
	    },
	    ArrayBytes(sets, count) + ArrayBytes(dynamic, dynamic_count));
	*packet.fixed = {point,         layout, first, count, dynamic_count, packet.Copy(sets, count),
	                 packet.Copy(dynamic, dynamic_count)};
}

void Cmd::pushDescriptorSetKHR(vk::PipelineBindPoint point, vk::PipelineLayout layout, uint32_t set, uint32_t count,
                               const vk::WriteDescriptorSet* writes) const {
	if (Direct()) {
		m_direct.pushDescriptorSetKHR(point, layout, set, count, writes);
		return;
	}
	struct P {
		vk::PipelineBindPoint         point;
		vk::PipelineLayout            layout;
		uint32_t                      set, count;
		const vk::WriteDescriptorSet* writes;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.pushDescriptorSetKHR(p.point, p.layout, p.set, p.count, p.writes);
	    },
	    WritesBytes(count, writes));
	*packet.fixed = {point, layout, set, count, nullptr};
	packet.fixed->writes = CopyWrites(packet, count, writes);
}

void Cmd::pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset, uint32_t size,
                        const void* values) const {
	if (Direct()) {
		m_direct.pushConstants(layout, stages, offset, size, values);
		return;
	}
	struct P {
		vk::PipelineLayout   layout;
		vk::ShaderStageFlags stages;
		uint32_t             offset, size;
		const std::byte*     values;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.pushConstants(p.layout, p.stages, p.offset, p.size, p.values);
	    },
	    Aligned(size));
	*packet.fixed = {layout, stages, offset, size, packet.Copy(static_cast<const std::byte*>(values), size)};
}

void Cmd::bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets,
                             const vk::DeviceSize* sizes, const vk::DeviceSize* strides) const {
	if (Direct()) {
		m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		return;
	}
	struct P {
		uint32_t              first, count;
		const vk::Buffer*     buffers;
		const vk::DeviceSize* offsets;
		const vk::DeviceSize* sizes;
		const vk::DeviceSize* strides;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.bindVertexBuffers2(p.first, p.count, p.buffers, p.offsets, p.sizes, p.strides);
	    },
	    ArrayBytes(buffers, count) + ArrayBytes(offsets, count) + ArrayBytes(sizes, count) +
	        ArrayBytes(strides, count));
	*packet.fixed = {first,
	                 count,
	                 packet.Copy(buffers, count),
	                 packet.Copy(offsets, count),
	                 packet.Copy(sizes, count),
	                 packet.Copy(strides, count)};
}

void Cmd::bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
	if (Direct()) {
		m_direct.bindIndexBuffer(buffer, offset, type);
		return;
	}
	struct P {
		vk::Buffer     buffer;
		vk::DeviceSize offset;
		vk::IndexType  type;
	};
	Put(P {buffer, offset, type},
	    [](vk::CommandBuffer command, const P& p) { command.bindIndexBuffer(p.buffer, p.offset, p.type); });
}

void Cmd::draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex, uint32_t first_instance) const {
	if (Direct()) {
		m_direct.draw(vertices, instances, first_vertex, first_instance);
		return;
	}
	struct P {
		uint32_t vertices, instances, first_vertex, first_instance;
	};
	Put(P {vertices, instances, first_vertex, first_instance}, [](vk::CommandBuffer command, const P& p) {
		command.draw(p.vertices, p.instances, p.first_vertex, p.first_instance);
	});
}

void Cmd::drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index, int32_t vertex_offset,
                      uint32_t first_instance) const {
	if (Direct()) {
		m_direct.drawIndexed(indices, instances, first_index, vertex_offset, first_instance);
		return;
	}
	struct P {
		uint32_t indices, instances, first_index;
		int32_t  vertex_offset;
		uint32_t first_instance;
	};
	Put(P {indices, instances, first_index, vertex_offset, first_instance}, [](vk::CommandBuffer command, const P& p) {
		command.drawIndexed(p.indices, p.instances, p.first_index, p.vertex_offset, p.first_instance);
	});
}

namespace {
struct IndirectArgs {
	vk::Buffer     buffer;
	vk::DeviceSize offset;
	uint32_t       count, stride;
};
} // namespace

void Cmd::drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
	if (Direct()) {
		m_direct.drawIndirect(buffer, offset, count, stride);
		return;
	}
	Put(IndirectArgs {buffer, offset, count, stride}, [](vk::CommandBuffer command, const IndirectArgs& p) {
		command.drawIndirect(p.buffer, p.offset, p.count, p.stride);
	});
}

void Cmd::drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
	if (Direct()) {
		m_direct.drawIndexedIndirect(buffer, offset, count, stride);
		return;
	}
	Put(IndirectArgs {buffer, offset, count, stride}, [](vk::CommandBuffer command, const IndirectArgs& p) {
		command.drawIndexedIndirect(p.buffer, p.offset, p.count, p.stride);
	});
}

void Cmd::drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
	if (Direct()) {
		m_direct.drawMeshTasksEXT(x, y, z);
		return;
	}
	struct P {
		uint32_t x, y, z;
	};
	Put(P {x, y, z}, [](vk::CommandBuffer command, const P& p) { command.drawMeshTasksEXT(p.x, p.y, p.z); });
}

void Cmd::drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
	if (Direct()) {
		m_direct.drawMeshTasksIndirectEXT(buffer, offset, count, stride);
		return;
	}
	Put(IndirectArgs {buffer, offset, count, stride}, [](vk::CommandBuffer command, const IndirectArgs& p) {
		command.drawMeshTasksIndirectEXT(p.buffer, p.offset, p.count, p.stride);
	});
}

void Cmd::dispatch(uint32_t x, uint32_t y, uint32_t z) const {
	if (Direct()) {
		m_direct.dispatch(x, y, z);
		return;
	}
	struct P {
		uint32_t x, y, z;
	};
	Put(P {x, y, z}, [](vk::CommandBuffer command, const P& p) { command.dispatch(p.x, p.y, p.z); });
}

void Cmd::dispatchBase(uint32_t base_x, uint32_t base_y, uint32_t base_z, uint32_t x, uint32_t y, uint32_t z) const {
	if (Direct()) {
		m_direct.dispatchBase(base_x, base_y, base_z, x, y, z);
		return;
	}
	struct P {
		uint32_t base_x, base_y, base_z, x, y, z;
	};
	Put(P {base_x, base_y, base_z, x, y, z}, [](vk::CommandBuffer command, const P& p) {
		command.dispatchBase(p.base_x, p.base_y, p.base_z, p.x, p.y, p.z);
	});
}

void Cmd::dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
	if (Direct()) {
		m_direct.dispatchIndirect(buffer, offset);
		return;
	}
	Put(IndirectArgs {buffer, offset, 0, 0},
	    [](vk::CommandBuffer command, const IndirectArgs& p) { command.dispatchIndirect(p.buffer, p.offset); });
}

void Cmd::copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count, const vk::BufferCopy* regions) const {
	if (Direct()) {
		m_direct.copyBuffer(source, destination, count, regions);
		return;
	}
	struct P {
		vk::Buffer          source, destination;
		uint32_t            count;
		const vk::BufferCopy* regions;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.copyBuffer(p.source, p.destination, p.count, p.regions);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {source, destination, count, packet.Copy(regions, count)};
}

namespace {
template <typename Region>
struct ImageToImage {
	vk::Image       source;
	vk::ImageLayout source_layout;
	vk::Image       destination;
	vk::ImageLayout destination_layout;
	uint32_t        count;
	vk::Filter      filter;
	const Region*   regions;
};
} // namespace

void Cmd::copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                    vk::ImageLayout destination_layout, uint32_t count, const vk::ImageCopy* regions) const {
	if (Direct()) {
		m_direct.copyImage(source, source_layout, destination, destination_layout, count, regions);
		return;
	}
	using P     = ImageToImage<vk::ImageCopy>;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.copyImage(p.source, p.source_layout, p.destination, p.destination_layout, p.count,
		                                    p.regions);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {source, source_layout, destination, destination_layout, count, {}, packet.Copy(regions, count)};
}

void Cmd::blitImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                    vk::ImageLayout destination_layout, uint32_t count, const vk::ImageBlit* regions,
                    vk::Filter filter) const {
	if (Direct()) {
		m_direct.blitImage(source, source_layout, destination, destination_layout, count, regions, filter);
		return;
	}
	using P     = ImageToImage<vk::ImageBlit>;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.blitImage(p.source, p.source_layout, p.destination, p.destination_layout, p.count,
		                                    p.regions, p.filter);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {source, source_layout, destination, destination_layout, count, filter,
	                 packet.Copy(regions, count)};
}

void Cmd::resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
                       vk::ImageLayout destination_layout, uint32_t count, const vk::ImageResolve* regions) const {
	if (Direct()) {
		m_direct.resolveImage(source, source_layout, destination, destination_layout, count, regions);
		return;
	}
	using P     = ImageToImage<vk::ImageResolve>;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.resolveImage(p.source, p.source_layout, p.destination, p.destination_layout,
		                                       p.count, p.regions);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {source, source_layout, destination, destination_layout, count, {}, packet.Copy(regions, count)};
}

namespace {
struct BufferImage {
	vk::Buffer                buffer;
	vk::Image                 image;
	vk::ImageLayout           layout;
	uint32_t                  count;
	const vk::BufferImageCopy* regions;
};
} // namespace

void Cmd::copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout, uint32_t count,
                            const vk::BufferImageCopy* regions) const {
	if (Direct()) {
		m_direct.copyBufferToImage(source, destination, layout, count, regions);
		return;
	}
	auto packet = Begin<BufferImage>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<BufferImage>(data);
		    Back(context).command.copyBufferToImage(p.buffer, p.image, p.layout, p.count, p.regions);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {source, destination, layout, count, packet.Copy(regions, count)};
}

void Cmd::copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination, uint32_t count,
                            const vk::BufferImageCopy* regions) const {
	if (Direct()) {
		m_direct.copyImageToBuffer(source, layout, destination, count, regions);
		return;
	}
	auto packet = Begin<BufferImage>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<BufferImage>(data);
		    Back(context).command.copyImageToBuffer(p.image, p.layout, p.buffer, p.count, p.regions);
	    },
	    ArrayBytes(regions, count));
	*packet.fixed = {destination, source, layout, count, packet.Copy(regions, count)};
}

void Cmd::fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t value) const {
	if (Direct()) {
		m_direct.fillBuffer(buffer, offset, size, value);
		return;
	}
	struct P {
		vk::Buffer     buffer;
		vk::DeviceSize offset, size;
		uint32_t       value;
	};
	Put(P {buffer, offset, size, value},
	    [](vk::CommandBuffer command, const P& p) { command.fillBuffer(p.buffer, p.offset, p.size, p.value); });
}

namespace {
template <typename Value>
struct ClearImage {
	vk::Image                         image;
	vk::ImageLayout                   layout;
	uint32_t                          count;
	const Value*                      value;
	const vk::ImageSubresourceRange* ranges;
};
} // namespace

void Cmd::clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color, uint32_t count,
                          const vk::ImageSubresourceRange* ranges) const {
	if (Direct()) {
		m_direct.clearColorImage(image, layout, color, count, ranges);
		return;
	}
	using P     = ClearImage<vk::ClearColorValue>;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.clearColorImage(p.image, p.layout, p.value, p.count, p.ranges);
	    },
	    ArrayBytes(color, 1) + ArrayBytes(ranges, count));
	*packet.fixed = {image, layout, count, packet.Copy(color, 1), packet.Copy(ranges, count)};
}

void Cmd::clearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue* value,
                                 uint32_t count, const vk::ImageSubresourceRange* ranges) const {
	if (Direct()) {
		m_direct.clearDepthStencilImage(image, layout, value, count, ranges);
		return;
	}
	using P     = ClearImage<vk::ClearDepthStencilValue>;
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.clearDepthStencilImage(p.image, p.layout, p.value, p.count, p.ranges);
	    },
	    ArrayBytes(value, 1) + ArrayBytes(ranges, count));
	*packet.fixed = {image, layout, count, packet.Copy(value, 1), packet.Copy(ranges, count)};
}

void Cmd::beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const {
	if (Direct()) {
		m_direct.beginConditionalRenderingEXT(info);
		return;
	}
	NoChain(info->pNext, "VkConditionalRenderingBeginInfoEXT");
	Put(*info, [](vk::CommandBuffer command, const vk::ConditionalRenderingBeginInfoEXT& p) {
		command.beginConditionalRenderingEXT(&p);
	});
}

void Cmd::endConditionalRenderingEXT() const {
	if (Direct()) {
		m_direct.endConditionalRenderingEXT();
		return;
	}
	struct P {};
	Put(P {}, [](vk::CommandBuffer command, const P&) { command.endConditionalRenderingEXT(); });
}

void Cmd::beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
	if (Direct()) {
		m_direct.beginQuery(pool, query, flags);
		return;
	}
	struct P {
		vk::QueryPool         pool;
		uint32_t              query;
		vk::QueryControlFlags flags;
	};
	Put(P {pool, query, flags},
	    [](vk::CommandBuffer command, const P& p) { command.beginQuery(p.pool, p.query, p.flags); });
}

void Cmd::endQuery(vk::QueryPool pool, uint32_t query) const {
	if (Direct()) {
		m_direct.endQuery(pool, query);
		return;
	}
	struct P {
		vk::QueryPool pool;
		uint32_t      query;
	};
	Put(P {pool, query}, [](vk::CommandBuffer command, const P& p) { command.endQuery(p.pool, p.query); });
}

void Cmd::resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
	if (Direct()) {
		m_direct.resetQueryPool(pool, first, count);
		return;
	}
	struct P {
		vk::QueryPool pool;
		uint32_t      first, count;
	};
	Put(P {pool, first, count},
	    [](vk::CommandBuffer command, const P& p) { command.resetQueryPool(p.pool, p.first, p.count); });
}

void Cmd::writeTimestamp2(vk::PipelineStageFlags2 stage, vk::QueryPool pool, uint32_t query) const {
	if (Direct()) {
		m_direct.writeTimestamp2(stage, pool, query);
		return;
	}
	struct P {
		vk::PipelineStageFlags2 stage;
		vk::QueryPool           pool;
		uint32_t                query;
	};
	Put(P {stage, pool, query},
	    [](vk::CommandBuffer command, const P& p) { command.writeTimestamp2(p.stage, p.pool, p.query); });
}

void Cmd::setCheckpointNV(const void* marker) const {
	if (Direct()) {
		m_direct.setCheckpointNV(marker);
		return;
	}
	struct P {
		const void* marker; // an opaque value, never dereferenced
	};
	Put(P {marker}, [](vk::CommandBuffer command, const P& p) { command.setCheckpointNV(p.marker); });
}

void Cmd::setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
	if (Direct()) {
		m_direct.setViewport(first, count, viewports);
		return;
	}
	struct P {
		uint32_t            first, count;
		const vk::Viewport* viewports;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.setViewport(p.first, p.count, p.viewports);
	    },
	    ArrayBytes(viewports, count));
	*packet.fixed = {first, count, packet.Copy(viewports, count)};
}

void Cmd::setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
	if (Direct()) {
		m_direct.setScissor(first, count, scissors);
		return;
	}
	struct P {
		uint32_t          first, count;
		const vk::Rect2D* scissors;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.setScissor(p.first, p.count, p.scissors);
	    },
	    ArrayBytes(scissors, count));
	*packet.fixed = {first, count, packet.Copy(scissors, count)};
}

void Cmd::setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
	if (Direct()) {
		m_direct.setViewportWithCount(count, viewports);
		return;
	}
	struct P {
		uint32_t            count;
		const vk::Viewport* viewports;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.setViewportWithCount(p.count, p.viewports);
	    },
	    ArrayBytes(viewports, count));
	*packet.fixed = {count, packet.Copy(viewports, count)};
}

void Cmd::setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
	if (Direct()) {
		m_direct.setScissorWithCount(count, scissors);
		return;
	}
	struct P {
		uint32_t          count;
		const vk::Rect2D* scissors;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.setScissorWithCount(p.count, p.scissors);
	    },
	    ArrayBytes(scissors, count));
	*packet.fixed = {count, packet.Copy(scissors, count)};
}

#define KYTY_CMD_SET1(name, type)                                                                                     \
	void Cmd::name(type value) const {                                                                                \
		if (Direct()) {                                                                                               \
			m_direct.name(value);                                                                                     \
			return;                                                                                                   \
		}                                                                                                             \
		struct P {                                                                                                    \
			type value;                                                                                               \
		};                                                                                                            \
		Put(P {value}, [](vk::CommandBuffer command, const P& p) { command.name(p.value); });                         \
	}
KYTY_CMD_SET1(setDepthTestEnable, vk::Bool32)
KYTY_CMD_SET1(setDepthWriteEnable, vk::Bool32)
KYTY_CMD_SET1(setDepthCompareOp, vk::CompareOp)
KYTY_CMD_SET1(setDepthBoundsTestEnable, vk::Bool32)
KYTY_CMD_SET1(setDepthBiasEnable, vk::Bool32)
KYTY_CMD_SET1(setStencilTestEnable, vk::Bool32)
KYTY_CMD_SET1(setCullMode, vk::CullModeFlags)
KYTY_CMD_SET1(setFrontFace, vk::FrontFace)
KYTY_CMD_SET1(setLineWidth, float)
KYTY_CMD_SET1(setAttachmentFeedbackLoopEnableEXT, vk::ImageAspectFlags)
#undef KYTY_CMD_SET1

#define KYTY_CMD_FACE(name)                                                                                           \
	void Cmd::name(vk::StencilFaceFlags faces, uint32_t value) const {                                                \
		if (Direct()) {                                                                                               \
			m_direct.name(faces, value);                                                                              \
			return;                                                                                                   \
		}                                                                                                             \
		struct P {                                                                                                    \
			vk::StencilFaceFlags faces;                                                                               \
			uint32_t             value;                                                                               \
		};                                                                                                            \
		Put(P {faces, value}, [](vk::CommandBuffer command, const P& p) { command.name(p.faces, p.value); });         \
	}
KYTY_CMD_FACE(setStencilReference)
KYTY_CMD_FACE(setStencilWriteMask)
KYTY_CMD_FACE(setStencilCompareMask)
#undef KYTY_CMD_FACE

void Cmd::setDepthBounds(float min, float max) const {
	if (Direct()) {
		m_direct.setDepthBounds(min, max);
		return;
	}
	struct P {
		float min, max;
	};
	Put(P {min, max}, [](vk::CommandBuffer command, const P& p) { command.setDepthBounds(p.min, p.max); });
}

void Cmd::setDepthBias(float constant, float clamp, float slope) const {
	if (Direct()) {
		m_direct.setDepthBias(constant, clamp, slope);
		return;
	}
	struct P {
		float constant, clamp, slope;
	};
	Put(P {constant, clamp, slope},
	    [](vk::CommandBuffer command, const P& p) { command.setDepthBias(p.constant, p.clamp, p.slope); });
}

void Cmd::setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
                       vk::CompareOp compare) const {
	if (Direct()) {
		m_direct.setStencilOp(faces, fail, pass, depth_fail, compare);
		return;
	}
	struct P {
		vk::StencilFaceFlags faces;
		vk::StencilOp        fail, pass, depth_fail;
		vk::CompareOp        compare;
	};
	Put(P {faces, fail, pass, depth_fail, compare}, [](vk::CommandBuffer command, const P& p) {
		command.setStencilOp(p.faces, p.fail, p.pass, p.depth_fail, p.compare);
	});
}

void Cmd::setBlendConstants(const float constants[4]) const {
	if (Direct()) {
		m_direct.setBlendConstants(constants);
		return;
	}
	struct P {
		std::array<float, 4> constants;
	};
	P payload {};
	std::memcpy(payload.constants.data(), constants, sizeof(payload.constants));
	Put(payload, [](vk::CommandBuffer command, const P& p) { command.setBlendConstants(p.constants.data()); });
}

void Cmd::setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
	if (Direct()) {
		m_direct.setColorWriteEnableEXT(count, enables);
		return;
	}
	struct P {
		uint32_t          count;
		const vk::Bool32* enables;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).command.setColorWriteEnableEXT(p.count, p.enables);
	    },
	    ArrayBytes(enables, count));
	*packet.fixed = {count, packet.Copy(enables, count)};
}

void Cmd::UpdateDescriptorSets(vk::Device device, uint32_t count, const vk::WriteDescriptorSet* writes) const {
	if (Direct()) {
		device.updateDescriptorSets(count, writes, 0, nullptr);
		return;
	}
	struct P {
		uint32_t                      count;
		const vk::WriteDescriptorSet* writes;
	};
	auto packet = Begin<P>(
	    *m_stream,
	    [](const std::byte* data, void* context) {
		    const auto& p = Fixed<P>(data);
		    Back(context).device.updateDescriptorSets(p.count, p.writes, 0, nullptr);
	    },
	    WritesBytes(count, writes));
	packet.fixed->count  = count;
	packet.fixed->writes = CopyWrites(packet, count, writes);
}

} // namespace Libs::Graphics
