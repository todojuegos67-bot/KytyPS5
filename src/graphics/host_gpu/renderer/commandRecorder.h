#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/bisect.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <source_location>

namespace Libs::Graphics {

// Round 165: what the renderer records besides draws and dispatches. Counted for the 10 s
// "Recorded" lines (GpuPhases), and marked on the gap the next timed command closes while
// GpuTiming times frames, so the GPU time between commands can be told apart by what filled it.
namespace RecordedCommands {

enum Kind : uint32_t {
	Barriers,       // pipelineBarrier and pipelineBarrier2 calls
	MemoryBarriers, // their memory, buffer and image barriers
	BufferBarriers,
	ImageBarriers,
	LayoutChanges, // image barriers that change the layout
	FullBarriers,  // calls that wait for, or block, every stage (ALL_COMMANDS)
	BufferCopies,  // copyBuffer calls and their bytes
	BufferCopyBytes,
	BufferUpdates, // updateBuffer
	BufferUpdateBytes,
	BufferFills, // fillBuffer
	BufferFillBytes,
	ImageUploads,      // buffer to image
	ImageDownloads,    // image to buffer
	ImageCopies,       // image to image, and resolves
	Clears,            // clearColorImage, clearDepthStencilImage
	RenderPasses,      // beginRendering
	LoadedAttachments, // attachments those load (LOAD_OP_LOAD)
	// Round 167 (copies:lean, images:lean).
	BatchedCopies,        // copies made in batches (CopyBatchState)
	BatchBefores,         // the barriers they waited for the work before them with
	BatchAfters,          // the barriers that made them visible, recorded on their own
	BatchMerged,          // and taken into another barrier
	BatchOverlaps,        // barriers between batched copies of the same bytes
	ImageBarriersSkipped, // image uses that made no barrier (Image::GetBarriers)
	// Round 172 (DynamicStateShadow).
	StateCalls,        // dynamic state calls
	StateUnchanged,    // ... that set what the command buffer held already
	PipelineBinds,     // pipeline binds
	PipelineUnchanged, // ... of the pipeline bound already
	StateLeftOut,      // calls of both kinds left out (state:filter)
	Count
};

// The marks of a gap between two timed commands.
enum Gap : uint32_t {
	GapBarrier    = 1u << 0u,
	GapLayout     = 1u << 1u,
	GapCopy       = 1u << 2u, // buffer copies, updates
	GapImageCopy  = 1u << 3u, // image uploads, downloads, copies, resolves
	GapClear      = 1u << 4u,
	GapFill       = 1u << 5u,
	GapRenderPass = 1u << 6u, // a rendering began or ended
};

inline std::array<std::atomic<uint64_t>, Count> g_counts {};
inline std::atomic<uint32_t>                    g_gap {0};

inline void Add(Kind kind, uint64_t n = 1) noexcept {
	g_counts[kind].fetch_add(n, std::memory_order_relaxed);
}
inline void Mark(uint32_t gap) noexcept {
	g_gap.fetch_or(gap, std::memory_order_relaxed);
}

// Round 166: barrier calls by the source line that made them (the caller of the barrier helpers
// that pass theirs on), for the "Recorded" lines' places. A slot is claimed once; a full table
// drops the rest.
struct BarrierSite {
	std::atomic<const char*> file {nullptr};
	std::atomic<uint32_t>    line {0};
	std::atomic<uint64_t>    count {0};
	std::atomic<uint64_t>    full {0};
};
inline std::array<BarrierSite, 256> g_barrier_sites {};

inline void NoteSite(const std::source_location& site, bool full) noexcept {
	const char* file = site.file_name();
	const auto  line = site.line();
	const auto  hash = (reinterpret_cast<uintptr_t>(file) >> 3u) ^ (uintptr_t {line} * 2654435761u);
	for (size_t probe = 0; probe < 16; probe++) {
		auto&       slot    = g_barrier_sites[(hash + probe) % g_barrier_sites.size()];
		const char* claimed = slot.file.load(std::memory_order_acquire);
		if (claimed == nullptr) {
			const char* expected = nullptr;
			if (slot.file.compare_exchange_strong(expected, file, std::memory_order_acq_rel)) {
				slot.line.store(line, std::memory_order_release);
				claimed = file;
			} else {
				claimed = expected;
			}
		}
		if (claimed == file && slot.line.load(std::memory_order_acquire) == line) {
			slot.count.fetch_add(1, std::memory_order_relaxed);
			if (full) {
				slot.full.fetch_add(1, std::memory_order_relaxed);
			}
			return;
		}
	}
}

inline void
NoteBarrier(bool full, uint32_t memory, uint32_t buffers, uint32_t images, uint32_t layout_changes,
            const std::source_location& site = std::source_location::current()) noexcept {
	NoteSite(site, full);
	Add(Barriers);
	Add(MemoryBarriers, memory);
	Add(BufferBarriers, buffers);
	Add(ImageBarriers, images);
	Add(LayoutChanges, layout_changes);
	if (full) {
		Add(FullBarriers);
	}
	Mark(layout_changes != 0 ? (GapBarrier | GapLayout) : GapBarrier);
}

} // namespace RecordedCommands

// The Vulkan commands of the hot recording paths, with vk::CommandBuffer's names and arguments
// (ported from IDXTRI's wolverine-perf branch, d9958e28 + ad1e1396).
// Bound to a raw command buffer it records directly; bound to a threaded scheduler
// (KYTY_RECORD_THREAD) it copies the arguments, and every array they point to, into the
// scheduler's command chunk and queues the call for the recording thread. Nothing the caller
// passes by pointer is used after the method returns.
class CommandRecorder {
public:
	// NOLINTNEXTLINE(google-explicit-constructor): raw command buffers stay usable as recorders.
	CommandRecorder(vk::CommandBuffer direct): m_direct(direct) {}
	CommandRecorder(CommandScheduler& scheduler, vk::CommandBuffer direct)
	    : m_scheduler(&scheduler), m_direct(direct) {}

	[[nodiscard]] bool Threaded() const noexcept { return m_direct == nullptr; }
	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
		return m_direct == nullptr && m_scheduler == nullptr;
	}

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		if (auto* shadow = Shadow(); shadow != nullptr) {
			// Round 172: a graphics pipeline of the emulator's own (blits, clears) may hold any
			// state static, which binding it makes undefined: the shadow holds none after it. Such
			// a bind is never left out (state set since its last bind may be what it holds
			// static). A compute pipeline holds no dynamic state.
			if (bind_point == vk::PipelineBindPoint::eCompute) {
				if (NoteBind(*shadow, shadow->compute_pipeline == pipeline)) {
					return;
				}
				shadow->compute_pipeline = pipeline;
			} else {
				shadow->binds++;
				shadow->InvalidateGraphics();
			}
		}
		Run([=](vk::CommandBuffer command) { command.bindPipeline(bind_point, pipeline); });
	}
	// Round 172: a guest draw's graphics pipeline, made with every dynamic state the draws set
	// (pipeline/shaders.cpp), the color write enables only with color targets. The other
	// optional ones (raster state, feedback loop) are the same for every pipeline of a run. A
	// state a guest pipeline may hold static must be invalidated here.
	void bindGuestPipeline(vk::Pipeline pipeline, bool color_write_dynamic) const {
		auto* shadow = Shadow();
		if (shadow == nullptr) {
			bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			return;
		}
		if (NoteBind(*shadow, shadow->graphics_pipeline == pipeline)) {
			return;
		}
		if (!color_write_dynamic) {
			shadow->Invalidate(DynamicStateShadow::ColorWrite);
		}
		shadow->graphics_pipeline = pipeline;
		Run([=](vk::CommandBuffer command) {
			command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		});
	}
	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex,
	          uint32_t first_instance) const {
		Touch();
		Run([=](vk::CommandBuffer command) {
			command.draw(vertices, instances, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index, int32_t offset,
	                 uint32_t first_instance) const {
		Touch();
		Run([=](vk::CommandBuffer command) {
			command.drawIndexed(indices, instances, first_index, offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		Touch();
		Run([=](vk::CommandBuffer command) {
			command.drawIndirect(buffer, offset, count, stride);
		});
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		Touch();
		Run([=](vk::CommandBuffer command) {
			command.drawIndexedIndirect(buffer, offset, count, stride);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		Touch();
		Run([=](vk::CommandBuffer command) { command.drawMeshTasksEXT(x, y, z); });
	}
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                              uint32_t stride) const {
		Touch();
		Run([=](vk::CommandBuffer command) {
			command.drawMeshTasksIndirectEXT(buffer, offset, count, stride);
		});
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		Touch();
		Run([=](vk::CommandBuffer command) { command.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		Touch();
		Run([=](vk::CommandBuffer command) { command.dispatchIndirect(buffer, offset); });
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Run([=](vk::CommandBuffer command) { command.bindIndexBuffer(buffer, offset, type); });
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (!Threaded()) {
			m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
			return;
		}
		const auto* b = m_scheduler->Stash(buffers, count);
		const auto* o = m_scheduler->Stash(offsets, count);
		const auto* s = m_scheduler->Stash(sizes, sizes != nullptr ? count : 0);
		const auto* d = m_scheduler->Stash(strides, strides != nullptr ? count : 0);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindVertexBuffers2(first, count, b, o, s, d);
		});
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		if (!Threaded()) {
			m_direct.pushConstants(layout, stages, offset, size, values);
			return;
		}
		const auto* data = m_scheduler->Stash(static_cast<const std::byte*>(values), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pushConstants(layout, stages, offset, size, data);
		});
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) const {
		if (!Threaded()) {
			m_direct.bindDescriptorSets(bind_point, layout, first, count, sets, dynamic_count,
			                            dynamic_offsets);
			return;
		}
		const auto* s = m_scheduler->Stash(sets, count);
		const auto* d = m_scheduler->Stash(dynamic_offsets, dynamic_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindDescriptorSets(bind_point, layout, first, count, s, dynamic_count, d);
		});
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (!Threaded()) {
			m_direct.pushDescriptorSetKHR(bind_point, layout, set, count, writes);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			ScrubDeadDescriptorViews(copy, count, "recorded descriptors");
			command.pushDescriptorSetKHR(bind_point, layout, set, count, copy);
		});
	}
	// vkUpdateDescriptorSets of a set allocated for this command buffer, then its bind: both move
	// to the recording thread (the update is driver work too). The set must not be in use yet.
	// `first_set`: the set number it is bound as (KYTY_FASTDRAW_IMAGESET's set 2; 0 otherwise).
	void updateAndBindDescriptorSet(vk::Device device, vk::PipelineBindPoint bind_point,
	                                vk::PipelineLayout layout, vk::DescriptorSet set,
	                                uint32_t count, const vk::WriteDescriptorSet* writes,
	                                uint32_t first_set = 0) const {
		if (!Threaded()) {
			device.updateDescriptorSets(count, writes, 0, nullptr);
			m_direct.bindDescriptorSets(bind_point, layout, first_set, 1, &set, 0, nullptr);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			ScrubDeadDescriptorViews(copy, count, "recorded descriptors");
			device.updateDescriptorSets(count, copy, 0, nullptr);
			command.bindDescriptorSets(bind_point, layout, first_set, 1, &set, 0, nullptr);
		});
	}
	void pipelineBarrier(vk::PipelineStageFlags source, vk::PipelineStageFlags destination,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images,
	                     const std::source_location& site = std::source_location::current()) const {
		Touch();
		{
			uint32_t layouts = 0;
			for (uint32_t i = 0; i < image_count; i++) {
				layouts += images[i].oldLayout != images[i].newLayout ? 1u : 0u;
			}
			RecordedCommands::NoteBarrier(
			    static_cast<bool>((source | destination) & vk::PipelineStageFlagBits::eAllCommands),
			    memory_count, buffer_count, image_count, layouts, site);
		}
		if (!Threaded()) {
			m_direct.pipelineBarrier(source, destination, flags, memory_count, memory, buffer_count,
			                         buffers, image_count, images);
			return;
		}
		RequireNoNext(memory, memory_count);
		RequireNoNext(buffers, buffer_count);
		RequireNoNext(images, image_count);
		const auto* m = m_scheduler->Stash(memory, memory_count);
		const auto* b = m_scheduler->Stash(buffers, buffer_count);
		const auto* i = m_scheduler->Stash(images, image_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pipelineBarrier(source, destination, flags, memory_count, m, buffer_count, b,
			                        image_count, i);
		});
	}
	void
	pipelineBarrier2(const vk::DependencyInfo&   dependency,
	                 const std::source_location& site = std::source_location::current()) const {
		auto*      batch = Batch();
		const bool full  = WaitsForEverything(dependency);
		if (batch != nullptr && batch->after_pending) {
			if (full) {
				// It makes the batch's copies visible as its "after" barrier would.
				EndBatch(*batch);
				RecordedCommands::Add(RecordedCommands::BatchMerged);
			} else if (dependency.memoryBarrierCount < MaxMergedMemoryBarriers) {
				// The batch's "after" barrier goes in with this one.
				std::array<vk::MemoryBarrier2, MaxMergedMemoryBarriers> memory {};
				for (uint32_t i = 0; i < dependency.memoryBarrierCount; i++) {
					memory[i] = dependency.pMemoryBarriers[i];
				}
				memory[dependency.memoryBarrierCount] = AfterBarrier();
				auto merged                           = dependency;
				merged.memoryBarrierCount             = dependency.memoryBarrierCount + 1;
				merged.pMemoryBarriers                = memory.data();
				EndBatch(*batch);
				RecordedCommands::Add(RecordedCommands::BatchMerged);
				RecordBarrier2(merged, site);
				batch->fenced = false;
				return;
			} else {
				RecordAfter();
			}
		}
		RecordBarrier2(dependency, site);
		if (batch != nullptr) {
			batch->fenced = full;
		}
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		Touch();
		RecordCopy(source, destination, count, regions);
	}
	// Round 167 (copies:lean): a copy of the CPU's bytes into a GPU buffer (an upload, a store),
	// in the command buffer's batch of such copies (CopyBatchState). Consecutive ones share the
	// barriers around them: the first that must wait for the GPU work recorded before it
	// (`wait_before`: draws before it may still read the bytes it replaces) records the barrier
	// for that, and the next command that may read what the batch wrote (a draw, a dispatch, a
	// rendering, a copy, a barrier) first makes it visible. A barrier that waited for everything
	// just before serves as the first. Copies of bytes an earlier one of the batch wrote wait for
	// it. The caller has ended the rendering.
	void copyBufferBatched(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                       const vk::BufferCopy* regions, bool wait_before) const {
		auto* batch = Batch();
		EXIT_IF(batch == nullptr || count == 0);
		uint64_t begin = UINT64_MAX;
		uint64_t end   = 0;
		for (uint32_t i = 0; i < count; i++) {
			begin = std::min<uint64_t>(begin, regions[i].dstOffset);
			end   = std::max<uint64_t>(end, regions[i].dstOffset + regions[i].size);
		}
		bool overlap = batch->written.size() >= MaxBatchWrites;
		for (const auto& written: batch->written) {
			if (overlap) {
				break;
			}
			overlap = written.buffer == destination && written.begin < end && begin < written.end;
		}
		const bool before = wait_before && !batch->fenced;
		if (overlap && !before) {
			vk::MemoryBarrier2 barrier {};
			barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
			barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
			RecordMemoryBarrier2(barrier);
			batch->written.clear();
			RecordedCommands::Add(RecordedCommands::BatchOverlaps);
		}
		if (before) {
			// Also orders the batch's earlier copies before this one.
			vk::MemoryBarrier2 barrier {};
			barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
			barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
			RecordMemoryBarrier2(barrier);
			batch->fenced = true;
			batch->written.clear();
			RecordedCommands::Add(RecordedCommands::BatchBefores);
		}
		RecordCopy(source, destination, count, regions);
		batch->written.push_back({destination, begin, end});
		batch->after_pending = true;
		RecordedCommands::Add(RecordedCommands::BatchedCopies);
	}
	// Round 167: records the batch's "after" barrier if it is due (CommandBuffer::Handle, Submit).
	void FlushCopyBatch() const { Touch(); }
	void copyBuffer(vk::Buffer source, vk::Buffer destination, const vk::BufferCopy& region) const {
		copyBuffer(source, destination, 1, &region);
	}
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::BufferUpdates);
		RecordedCommands::Add(RecordedCommands::BufferUpdateBytes, size);
		RecordedCommands::Mark(RecordedCommands::GapCopy);
		if (!Threaded()) {
			m_direct.updateBuffer(buffer, offset, size, data);
			return;
		}
		const auto* copy = m_scheduler->Stash(static_cast<const std::byte*>(data), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.updateBuffer(buffer, offset, size, copy);
		});
	}
	void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.writeTimestamp(stage, pool, query); });
	}
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
		Run([=](vk::CommandBuffer command) { command.resetQueryPool(pool, first, count); });
	}
	void beginQuery(vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.beginQuery(pool, query, {}); });
	}
	void beginQuery(vk::QueryPool pool, uint32_t query, vk::QueryControlFlags flags) const {
		Run([=](vk::CommandBuffer command) { command.beginQuery(pool, query, flags); });
	}
	void endQuery(vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.endQuery(pool, query); });
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::BufferFills);
		RecordedCommands::Add(RecordedCommands::BufferFillBytes, size);
		RecordedCommands::Mark(RecordedCommands::GapFill);
		Run([=](vk::CommandBuffer command) { command.fillBuffer(buffer, offset, size, value); });
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::ImageUploads);
		RecordedCommands::Mark(RecordedCommands::GapImageCopy);
		if (!Threaded()) {
			m_direct.copyBufferToImage(source, destination, layout, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyBufferToImage(source, destination, layout, count, r);
		});
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       const vk::BufferImageCopy& region) const {
		copyBufferToImage(source, destination, layout, 1, &region);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::ImageDownloads);
		RecordedCommands::Mark(RecordedCommands::GapImageCopy);
		if (!Threaded()) {
			m_direct.copyImageToBuffer(source, layout, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImageToBuffer(source, layout, destination, count, r);
		});
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       const vk::BufferImageCopy& region) const {
		copyImageToBuffer(source, layout, destination, 1, &region);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::ImageCopies);
		RecordedCommands::Mark(RecordedCommands::GapImageCopy);
		if (!Threaded()) {
			m_direct.copyImage(source, source_layout, destination, destination_layout, count,
			                   regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImage(source, source_layout, destination, destination_layout, count, r);
		});
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
		copyImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, const vk::ImageResolve& region) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::ImageCopies);
		RecordedCommands::Mark(RecordedCommands::GapImageCopy);
		Run([=](vk::CommandBuffer command) {
			command.resolveImage(source, source_layout, destination, destination_layout, 1,
			                     &region);
		});
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::Clears);
		RecordedCommands::Mark(RecordedCommands::GapClear);
		if (!Threaded()) {
			m_direct.clearColorImage(image, layout, color, count, ranges);
			return;
		}
		const auto  value = *color;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearColorImage(image, layout, &value, count, r);
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) const {
		Touch();
		RecordedCommands::Add(RecordedCommands::Clears);
		RecordedCommands::Mark(RecordedCommands::GapClear);
		if (!Threaded()) {
			m_direct.clearDepthStencilImage(image, layout, value, count, ranges);
			return;
		}
		const auto  clear = *value;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearDepthStencilImage(image, layout, &clear, count, r);
		});
	}
	template <typename Writes>
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, const Writes& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, static_cast<uint32_t>(std::size(writes)),
		                     std::data(writes));
	}
	void beginRendering(const vk::RenderingInfo* info) const { beginRendering(*info); }
	void beginRendering(const vk::RenderingInfo& info) const {
		// A barrier cannot go inside the rendering: the batch's goes before it.
		Touch();
		{
			uint64_t loaded = 0;
			for (uint32_t i = 0; i < info.colorAttachmentCount; i++) {
				loaded += info.pColorAttachments[i].loadOp == vk::AttachmentLoadOp::eLoad ? 1u : 0u;
			}
			if (info.pDepthAttachment != nullptr &&
			    info.pDepthAttachment->loadOp == vk::AttachmentLoadOp::eLoad) {
				loaded++;
			}
			RecordedCommands::Add(RecordedCommands::RenderPasses);
			RecordedCommands::Add(RecordedCommands::LoadedAttachments, loaded);
			RecordedCommands::Mark(RecordedCommands::GapRenderPass);
		}
		if (!Threaded()) {
			m_direct.beginRendering(info);
			return;
		}
		EXIT_IF(info.pNext != nullptr);
		RequireNoNext(info.pColorAttachments, info.colorAttachmentCount);
		RequireNoNext(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		RequireNoNext(info.pStencilAttachment, info.pStencilAttachment != nullptr ? 1u : 0u);
		auto copy = info;
		copy.pColorAttachments =
		    m_scheduler->Stash(info.pColorAttachments, info.colorAttachmentCount);
		copy.pDepthAttachment =
		    m_scheduler->Stash(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		copy.pStencilAttachment = m_scheduler->Stash(info.pStencilAttachment,
		                                             info.pStencilAttachment != nullptr ? 1u : 0u);
		m_scheduler->Record([copy](vk::CommandBuffer command) {
			// The views may have been destroyed since the call was queued.
			ScrubDeadRenderingViews(copy, "a recorded rendering");
			command.beginRendering(copy);
		});
	}
	void endRendering() const {
		RecordedCommands::Mark(RecordedCommands::GapRenderPass);
		Run([](vk::CommandBuffer command) { command.endRendering(); });
	}
	void setCheckpointNV(const void* marker) const {
		Run([=](vk::CommandBuffer command) { command.setCheckpointNV(marker); });
	}
	// Round 124: one viewport or scissor, for pipelines with VK_DYNAMIC_STATE_VIEWPORT/SCISSOR
	// (the blit helper's).
	void setViewport(const vk::Viewport& viewport) const {
		Invalidate(DynamicStateShadow::Viewports);
		Run([viewport](vk::CommandBuffer command) { command.setViewport(0, 1, &viewport); });
	}
	void setScissor(const vk::Rect2D& scissor) const {
		Invalidate(DynamicStateShadow::Scissors);
		Run([scissor](vk::CommandBuffer command) { command.setScissor(0, 1, &scissor); });
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (Unchanged(DynamicStateShadow::Viewports, viewports, count * sizeof(vk::Viewport))) {
			return;
		}
		if (!Threaded()) {
			m_direct.setViewportWithCount(count, viewports);
			return;
		}
		const auto* v = m_scheduler->Stash(viewports, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setViewportWithCount(count, v); });
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (Unchanged(DynamicStateShadow::Scissors, scissors, count * sizeof(vk::Rect2D))) {
			return;
		}
		if (!Threaded()) {
			m_direct.setScissorWithCount(count, scissors);
			return;
		}
		const auto* s = m_scheduler->Stash(scissors, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setScissorWithCount(count, s); });
	}
	void setLineWidth(float width) const {
		if (Unchanged(DynamicStateShadow::LineWidth, &width, sizeof(width))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		const std::array<float, 4> values {constants[0], constants[1], constants[2], constants[3]};
		if (Unchanged(DynamicStateShadow::BlendConstants, values.data(), sizeof(values))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setBlendConstants(values.data()); });
	}
	void setCullMode(vk::CullModeFlags mode) const {
		if (Unchanged(DynamicStateShadow::CullMode, &mode, sizeof(mode))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setCullMode(mode); });
	}
	void setFrontFace(vk::FrontFace face) const {
		if (Unchanged(DynamicStateShadow::FrontFace, &face, sizeof(face))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setFrontFace(face); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		if (Unchanged(DynamicStateShadow::DepthBoundsTest, &enable, sizeof(enable))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		if (const std::array<float, 2> values {min_bounds, max_bounds};
		    Unchanged(DynamicStateShadow::DepthBounds, values.data(), sizeof(values))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBounds(min_bounds, max_bounds); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		if (Unchanged(DynamicStateShadow::DepthTest, &enable, sizeof(enable))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		if (Unchanged(DynamicStateShadow::DepthWrite, &enable, sizeof(enable))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		if (Unchanged(DynamicStateShadow::DepthCompare, &op, sizeof(op))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthCompareOp(op); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		if (Unchanged(DynamicStateShadow::DepthBiasEnable, &enable, sizeof(enable))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		if (const std::array<float, 3> values {constant, clamp, slope};
		    Unchanged(DynamicStateShadow::DepthBias, values.data(), sizeof(values))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBias(constant, clamp, slope); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		if (Unchanged(DynamicStateShadow::StencilTest, &enable, sizeof(enable))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags face, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		if (const std::array<uint32_t, 4> values {
		        static_cast<uint32_t>(fail), static_cast<uint32_t>(pass),
		        static_cast<uint32_t>(depth_fail), static_cast<uint32_t>(compare)};
		    UnchangedFaces(face, DynamicStateShadow::StencilOpFront,
		                   DynamicStateShadow::StencilOpBack, values.data(), sizeof(values))) {
			return;
		}
		Run([=](vk::CommandBuffer command) {
			command.setStencilOp(face, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags face, uint32_t mask) const {
		if (UnchangedFaces(face, DynamicStateShadow::StencilCompareFront,
		                   DynamicStateShadow::StencilCompareBack, &mask, sizeof(mask))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilCompareMask(face, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags face, uint32_t mask) const {
		if (UnchangedFaces(face, DynamicStateShadow::StencilWriteFront,
		                   DynamicStateShadow::StencilWriteBack, &mask, sizeof(mask))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilWriteMask(face, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags face, uint32_t reference) const {
		if (UnchangedFaces(face, DynamicStateShadow::StencilReferenceFront,
		                   DynamicStateShadow::StencilReferenceBack, &reference,
		                   sizeof(reference))) {
			return;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilReference(face, reference); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (Unchanged(DynamicStateShadow::ColorWrite, enables, count * sizeof(vk::Bool32))) {
			return;
		}
		if (!Threaded()) {
			m_direct.setColorWriteEnableEXT(count, enables);
			return;
		}
		const auto* e = m_scheduler->Stash(enables, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setColorWriteEnableEXT(count, e); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		if (Unchanged(DynamicStateShadow::FeedbackLoop, &aspects, sizeof(aspects))) {
			return;
		}
		Run([=](vk::CommandBuffer command) {
			command.setAttachmentFeedbackLoopEnableEXT(aspects);
		});
	}

private:
	static constexpr uint32_t MaxMergedMemoryBarriers = 8;
	static constexpr size_t   MaxBatchWrites          = 256;

	// Round 172 (DynamicStateShadow).
	[[nodiscard]] DynamicStateShadow* Shadow() const noexcept {
		return m_scheduler != nullptr ? &m_scheduler->StateShadow() : nullptr;
	}
	// Whether a state call can be left out: it sets what the command buffer holds already, and
	// state:filter (or KYTY_STATE_FILTER=1) is on. The shadow follows every call either way.
	[[nodiscard]] bool Unchanged(DynamicStateShadow::Kind kind, const void* data,
	                             size_t size) const {
		auto* shadow = Shadow();
		if (shadow == nullptr) {
			return false;
		}
		shadow->calls++;
		if (!shadow->Same(kind, data, static_cast<uint32_t>(size))) {
			return false;
		}
		shadow->unchanged++;
		if (!Bisect::StateFilter()) {
			return false;
		}
		shadow->left_out++;
		return true;
	}
	// A stencil state of the front face, the back face or both.
	[[nodiscard]] bool UnchangedFaces(vk::StencilFaceFlags face, DynamicStateShadow::Kind front,
	                                  DynamicStateShadow::Kind back, const void* data,
	                                  size_t size) const {
		auto* shadow = Shadow();
		if (shadow == nullptr) {
			return false;
		}
		shadow->calls++;
		const auto bytes = static_cast<uint32_t>(size);
		bool       same  = true;
		// Both are updated, not only the first that differs.
		if (face & vk::StencilFaceFlagBits::eFront) {
			same = shadow->Same(front, data, bytes) && same;
		}
		if (face & vk::StencilFaceFlagBits::eBack) {
			same = shadow->Same(back, data, bytes) && same;
		}
		if (!same) {
			return false;
		}
		shadow->unchanged++;
		if (!Bisect::StateFilter()) {
			return false;
		}
		shadow->left_out++;
		return true;
	}
	void Invalidate(DynamicStateShadow::Kind kind) const noexcept {
		if (auto* shadow = Shadow(); shadow != nullptr) {
			shadow->Invalidate(kind);
		}
	}
	// A pipeline bind: whether it binds what is bound already and can be left out.
	[[nodiscard]] static bool NoteBind(DynamicStateShadow& shadow, bool same) {
		shadow.binds++;
		if (!same) {
			return false;
		}
		shadow.binds_unchanged++;
		if (!Bisect::StateFilter()) {
			return false;
		}
		shadow.left_out++;
		return true;
	}
	[[nodiscard]] CopyBatchState* Batch() const noexcept {
		return m_scheduler != nullptr ? &m_scheduler->CopyBatch() : nullptr;
	}
	// The barrier after a batch of copies: what they wrote, for everything after them.
	[[nodiscard]] static vk::MemoryBarrier2 AfterBarrier() noexcept {
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.dstAccessMask =
		    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		return barrier;
	}
	// A memory barrier from all the work before it, its writes made visible, to all after it.
	[[nodiscard]] static bool WaitsForEverything(const vk::DependencyInfo& dependency) noexcept {
		for (uint32_t i = 0; i < dependency.memoryBarrierCount; i++) {
			const auto& b = dependency.pMemoryBarriers[i];
			if (static_cast<bool>(b.srcStageMask & vk::PipelineStageFlagBits2::eAllCommands) &&
			    static_cast<bool>(b.dstStageMask & vk::PipelineStageFlagBits2::eAllCommands) &&
			    static_cast<bool>(b.srcAccessMask & vk::AccessFlagBits2::eMemoryWrite) &&
			    static_cast<bool>(b.dstAccessMask & vk::AccessFlagBits2::eMemoryRead) &&
			    static_cast<bool>(b.dstAccessMask & vk::AccessFlagBits2::eMemoryWrite)) {
				return true;
			}
		}
		return false;
	}
	static void EndBatch(CopyBatchState& batch) noexcept {
		batch.after_pending = false;
		batch.written.clear();
	}
	void
	RecordMemoryBarrier2(const vk::MemoryBarrier2&   barrier,
	                     const std::source_location& site = std::source_location::current()) const {
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		RecordBarrier2(dependency, site);
	}
	void RecordAfter() const {
		auto* batch = Batch();
		RecordMemoryBarrier2(AfterBarrier());
		EndBatch(*batch);
		batch->fenced = false;
		RecordedCommands::Add(RecordedCommands::BatchAfters);
	}
	// What a command that may read or write memory does first: the batch's copies become
	// visible to it.
	void Touch() const {
		auto* batch = Batch();
		if (batch == nullptr) {
			return;
		}
		if (batch->after_pending) {
			RecordAfter();
		}
		batch->fenced = false;
	}
	void RecordBarrier2(const vk::DependencyInfo&   dependency,
	                    const std::source_location& site) const {
		{
			uint32_t layouts = 0;
			bool     full    = false;
			for (uint32_t i = 0; i < dependency.imageMemoryBarrierCount; i++) {
				const auto& b = dependency.pImageMemoryBarriers[i];
				layouts += b.oldLayout != b.newLayout ? 1u : 0u;
				full |= static_cast<bool>((b.srcStageMask | b.dstStageMask) &
				                          vk::PipelineStageFlagBits2::eAllCommands);
			}
			for (uint32_t i = 0; i < dependency.memoryBarrierCount; i++) {
				const auto& b = dependency.pMemoryBarriers[i];
				full |= static_cast<bool>((b.srcStageMask | b.dstStageMask) &
				                          vk::PipelineStageFlagBits2::eAllCommands);
			}
			for (uint32_t i = 0; i < dependency.bufferMemoryBarrierCount; i++) {
				const auto& b = dependency.pBufferMemoryBarriers[i];
				full |= static_cast<bool>((b.srcStageMask | b.dstStageMask) &
				                          vk::PipelineStageFlagBits2::eAllCommands);
			}
			RecordedCommands::NoteBarrier(full, dependency.memoryBarrierCount,
			                              dependency.bufferMemoryBarrierCount,
			                              dependency.imageMemoryBarrierCount, layouts, site);
		}
		if (!Threaded()) {
			m_direct.pipelineBarrier2(dependency);
			return;
		}
		EXIT_IF(dependency.pNext != nullptr);
		RequireNoNext(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		RequireNoNext(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
		RequireNoNext(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		auto copy = dependency;
		copy.pMemoryBarriers =
		    m_scheduler->Stash(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		copy.pBufferMemoryBarriers = m_scheduler->Stash(dependency.pBufferMemoryBarriers,
		                                                dependency.bufferMemoryBarrierCount);
		copy.pImageMemoryBarriers =
		    m_scheduler->Stash(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		m_scheduler->Record([copy](vk::CommandBuffer command) { command.pipelineBarrier2(copy); });
	}
	void RecordCopy(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		{
			uint64_t bytes = 0;
			for (uint32_t i = 0; i < count; i++) {
				bytes += regions[i].size;
			}
			RecordedCommands::Add(RecordedCommands::BufferCopies);
			RecordedCommands::Add(RecordedCommands::BufferCopyBytes, bytes);
			RecordedCommands::Mark(RecordedCommands::GapCopy);
		}
		if (!Threaded()) {
			m_direct.copyBuffer(source, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.copyBuffer(source, destination, count, r); });
	}
	template <typename F>
	void Run(F&& fn) const {
		if (!Threaded()) {
			fn(m_direct);
		} else {
			m_scheduler->Record(std::forward<F>(fn));
		}
	}
	template <typename T>
	static void RequireNoNext(const T* items, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			EXIT_IF(items[i].pNext != nullptr);
		}
	}
	// A deep copy of descriptor writes: the writes and the info arrays they point to.
	const vk::WriteDescriptorSet* StashWrites(const vk::WriteDescriptorSet* writes,
	                                          uint32_t                      count) const {
		RequireNoNext(writes, count);
		auto* copy = const_cast<vk::WriteDescriptorSet*>(m_scheduler->Stash(writes, count));
		for (uint32_t i = 0; i < count; i++) {
			auto& write      = copy[i];
			write.pImageInfo = m_scheduler->Stash(
			    write.pImageInfo, write.pImageInfo != nullptr ? write.descriptorCount : 0);
			write.pBufferInfo = m_scheduler->Stash(
			    write.pBufferInfo, write.pBufferInfo != nullptr ? write.descriptorCount : 0);
			write.pTexelBufferView =
			    m_scheduler->Stash(write.pTexelBufferView,
			                       write.pTexelBufferView != nullptr ? write.descriptorCount : 0);
		}
		return copy;
	}

	CommandScheduler* m_scheduler = nullptr;
	vk::CommandBuffer m_direct    = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
