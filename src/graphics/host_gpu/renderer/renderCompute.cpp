#include "graphics/drawDiff.h"
#include "common/liveSwitches.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

// Research: the rollback below drains the GPU twice per dispatch and almost never fires once a
// shader's pages are resident (one retry in a whole run, against ~2500 protected dispatches a
// frame). A shader is trusted after trust_runs clean attempts and dispatched without it; any
// fault that unprotected work reports re-arms every shader. KYTY_DISPATCH_RECOVERY=always keeps
// it on every dispatch; KYTY_DISPATCH_RECOVERY_TRUST=<n> sets the clean runs needed.
struct DispatchRecoveryPolicy {
	std::unordered_map<uint64_t, uint32_t> clean_runs;
	uint64_t                               seen_faults = 0;
	uint32_t                               trust_runs  = 4;
	bool                                   always      = false;
};

// Dispatches are prepared on the guest GPU thread only.
DispatchRecoveryPolicy& RecoveryPolicy() {
	static DispatchRecoveryPolicy policy = [] {
		DispatchRecoveryPolicy p;
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY")) {
			p.always = std::strcmp(value, "always") == 0;
		}
		if (const char* value = std::getenv("KYTY_DISPATCH_RECOVERY_TRUST")) {
			p.trust_runs = static_cast<uint32_t>(std::strtoul(value, nullptr, 0));
		}
		return p;
	}();
	return policy;
}

// A missing BDA page returns zero during the first attempt. Preserve every
// externally writable buffer, including atomic destinations, so that attempt
// can be rolled back before any dependent dispatch observes its results.
class DispatchBufferRecovery {
public:
	DispatchBufferRecovery(RenderContext& context, const PreparedBindings& bindings,
	                       uint64_t shader_address)
	    : m_context(context), m_shader_address(shader_address) {
		const auto& program = *bindings.runtime->program;
		m_enabled           = program.info.uses_dma &&
		                      std::none_of(program.info.images.begin(), program.info.images.end(),
		                                   [](const auto& image) { return image.written; });
		if (!m_enabled) return;
		auto& cache     = context.GetBufferCache();
		auto& scheduler = context.GetCommandScheduler();
		auto& policy    = RecoveryPolicy();
		if (!policy.always) {
			if (const auto faults = cache.UnattributedFaults(); faults != policy.seen_faults) {
				policy.seen_faults = faults;
				policy.clean_runs.clear();
			}
			if (policy.clean_runs[shader_address] >= policy.trust_runs) {
				m_enabled = false;
				return;
			}
		}
		// Reports from earlier work cannot be attributed to this transaction.
		cache.ProcessFaultBuffer();
		scheduler.Finish();
		RangeSet    ranges;
		const auto& layout = program.bindings;
		// Sources are per bound (live) buffer; the first descriptor group maps them to resources.
		for (uint32_t slot = 0; slot < layout.memory_offset_count; ++slot) {
			const auto  resource = layout.descriptors.front().resources[slot];
			const auto& source   = bindings.buffer_sources[slot];
			if (!program.info.buffers[resource].written || source.size == 0) continue;
			const auto start = source.address & ~uint64_t {3};
			const auto end   = (source.address + source.size + 3) & ~uint64_t {3};
			ranges.Add(start, end - start);
		}
		ranges.ForEach([&](uint64_t start, uint64_t end) {
			const auto id = cache.FindBuffer(start, end - start);
			cache.SynchronizeBuffersInRange(start, end - start);
			const auto& source = cache.GetBuffer(id);
			Save(source, source.Offset(start), start, end - start, false);
		});
		if (std::any_of(program.bindings.descriptors.begin(), program.bindings.descriptors.end(),
		                [](const auto& binding) {
			                return binding.kind == ShaderRecompiler::IR::DescriptorBindingKind::Gds;
		                })) {
			const auto& source = *cache.GetGdsBuffer();
			Save(source, 0, 0, source.Size(), true);
		}
	}

	bool Retry() {
		if (!m_enabled) return false;
		auto&      cache  = m_context.GetBufferCache();
		const auto report = cache.CollectFaults();
		auto&      policy = RecoveryPolicy();
		if (report.page_count == 0) {
			if (report.trap.claimed != 0) {
				const auto hash =
				    (uint64_t {report.trap.shader_hash_high} << 32) | report.trap.shader_hash_low;
				EXIT("GPU shader trap after page residency: hash=0x%016" PRIx64
				     " pc=0x%08x code=0x%02x\n",
				     hash, report.trap.pc, report.trap.code);
			}
			if (m_attempts == 0) {
				policy.clean_runs[m_shader_address]++;
			}
			return false;
		}
		policy.clean_runs[m_shader_address] = 0;
		if (++m_attempts > 128) {
			EXIT("GPU shader paging did not converge: shader=0x%016" PRIx64 " pages=%" PRIu64 "\n",
			     m_shader_address, report.page_count);
		}
		static uint32_t reports = 0;
		if (reports++ < 32) {
			LOGF("GPU dispatch retry: shader=0x%016" PRIx64 " attempt=%u missing_pages=%" PRIu64
			     "\n",
			     m_shader_address, m_attempts, report.page_count);
		}
		for (const auto& saved: m_saved) {
			if (saved.gds) {
				cache.GetGdsBuffer()->CopyFrom(m_context.GetCommandScheduler().Current(),
				                               *saved.buffer, 0, 0, saved.buffer->Size());
			} else {
				// Resolving a fault can merge owners; never retain a native destination
				// handle or BufferId across CollectFaults().
				auto [destination, offset] =
				    cache.ObtainBuffer(saved.address, saved.buffer->Size(), true);
				destination->CopyFrom(m_context.GetCommandScheduler().Current(), *saved.buffer, 0,
				                      offset, saved.buffer->Size());
			}
		}
		m_context.PrepareBda();
		return true;
	}

private:
	void Save(const Buffer& source, uint64_t offset, uint64_t address, uint64_t size, bool gds) {
		auto saved =
		    std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
		                             MemoryUsage::DeviceLocal, 0, AllFlags, size);
		saved->CopyFrom(m_context.GetCommandScheduler().Current(), source, offset, 0, size);
		m_saved.push_back({address, gds, std::move(saved)});
	}
	struct Saved {
		uint64_t                address;
		bool                    gds;
		std::unique_ptr<Buffer> buffer;
	};
	RenderContext&     m_context;
	uint64_t           m_shader_address;
	bool               m_enabled  = false;
	uint32_t           m_attempts = 0;
	std::vector<Saved> m_saved;
};

} // namespace

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                 CommandBuffer& command, uint32_t group_x,
                                                 uint32_t group_y, uint32_t group_z,
                                                 uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX)
				return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1])
			return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue                  clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// Diagnostic: KYTY_DUMP_BUFFERS_SHADER=<16-hex hash> (startup): every buffer that compute program binds is saved
// to _TempData/bufdump/f<frame>_<order>_<before|after>_s<slot>_<r|w>_<addr>_<size>.bin right before and right
// after each of its dispatches (what it read, what it wrote), while KYTY_DUMP_BUFFERS_COUNT (live, default 0)
// dispatches remain: set it to N in the live file to dump the next N dispatches. Buffers over
// KYTY_DUMP_BUFFERS_MAX_MB (live, default 64) are skipped. (Wolverine: CS 021204e9e4fe4393 touches only buffers;
// skipping it ended the flicker.)
static bool DumpBuffersWanted(uint64_t hash) {
	static const uint64_t wanted = [] {
		const char* v = std::getenv("KYTY_DUMP_BUFFERS_SHADER");
		return v != nullptr ? std::strtoull(v, nullptr, 16) : 0ull;
	}();
	if (wanted == 0 || hash != wanted) return false;
	static auto& count = Common::LiveSwitches::Get("KYTY_DUMP_BUFFERS_COUNT", 0);
	return count.load(std::memory_order_relaxed) > 0;
}

// Diagnostic: KYTY_BUFFER_USERS_ADDR=<hex guest address> (startup): while KYTY_BUFFER_USERS_COUNT (live, default
// 0) is above 0, every compute dispatch that binds a buffer covering that address is logged (frame, shader hash,
// slot, read/write, bound range), one line per binding, counting down. Finds who fills a buffer (Wolverine: the
// light list at 0x11a59e8000 that CS 021204e9e4fe4393 sorts).
static void LogBufferUsers(const ShaderRecompiler::IR::CompiledShaderInfo& program, const PreparedBindings& bindings) {
	static const uint64_t address = [] {
		const char* v = std::getenv("KYTY_BUFFER_USERS_ADDR");
		return v != nullptr ? std::strtoull(v, nullptr, 16) : 0ull;
	}();
	if (address == 0) return;
	static auto& count = Common::LiveSwitches::Get("KYTY_BUFFER_USERS_COUNT", 0);
	if (count.load(std::memory_order_relaxed) <= 0) return;
	const auto& layout = program.bindings;
	if (layout.descriptors.empty()) return;
	for (uint32_t slot = 0; slot < layout.memory_offset_count && slot < bindings.buffer_sources.size(); ++slot) {
		const auto& source = bindings.buffer_sources[slot];
		if (source.size == 0 || address < source.address || address >= source.address + source.size) continue;
		const auto resource = layout.descriptors.front().resources[slot];
		const bool written  = resource < program.info.buffers.size() && program.info.buffers[resource].written;
		::printf("Buffer user: frame %llu cs %016llx slot %u %s 0x%010llx+0x%llx\n",
		         static_cast<unsigned long long>(DrawDiff::Frame()),
		         static_cast<unsigned long long>(program.shader_hash), slot, written ? "WRITE" : "read",
		         static_cast<unsigned long long>(source.address), static_cast<unsigned long long>(source.size));
		count.fetch_sub(1, std::memory_order_relaxed);
	}
	std::fflush(stdout);
}

void RenderExecutor::DumpBuffers(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 const PreparedBindings& bindings, const CommandRecorder& vk_buffer,
                                 const char* when) {
	static auto&    count    = Common::LiveSwitches::Get("KYTY_DUMP_BUFFERS_COUNT", 0);
	static auto&    max_mb   = Common::LiveSwitches::Get("KYTY_DUMP_BUFFERS_MAX_MB", 64);
	static uint64_t sequence = 0;
	static bool     folder   = false;
	if (std::strcmp(when, "after") == 0) count.fetch_sub(1, std::memory_order_relaxed);
	if (!folder) {
		std::error_code error;
		std::filesystem::create_directories("_TempData/bufdump", error);
		folder = true;
	}
	auto&       cache  = m_context.GetBufferCache();
	auto&       sched  = m_context.GetCommandScheduler();
	const auto  frame  = DrawDiff::Frame();
	const auto  order  = sequence++;
	const auto& layout = program.bindings;
	const auto  limit =
	    static_cast<uint64_t>(std::max<int64_t>(1, max_mb.load(std::memory_order_relaxed))) << 20u;
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                          vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                          nullptr);
	uint32_t saved = 0;
	for (uint32_t slot = 0;
	     slot < layout.memory_offset_count && slot < bindings.buffer_sources.size(); ++slot) {
		const auto  resource = layout.descriptors.front().resources[slot];
		const auto& source   = bindings.buffer_sources[slot];
		if (source.size == 0 || source.address == 0 || source.size > limit) continue;
		const bool written =
		    resource < program.info.buffers.size() && program.info.buffers[resource].written;
		const auto id = cache.FindBuffer(source.address, source.size);
		if (!id) continue;
		auto&      src    = cache.GetBuffer(id);
		const auto offset = src.Offset(source.address);
		if (offset + source.size > src.Size()) continue;
		auto download = std::make_shared<Buffer>(m_context.GetGraphics(), sched, MemoryUsage::Download,
		                                         0, AllFlags, source.size);
		auto* mapped  = download->Mapped().data();
		if (mapped == nullptr) continue;
		const vk::BufferCopy copy {offset, 0, source.size};
		vk_buffer.copyBuffer(src.Handle(), download->Handle(), 1, &copy);
		char name[256];
		std::snprintf(name, sizeof(name), "_TempData/bufdump/f%06llu_%05llu_%s_s%u_%c_%010llx_%llu.bin",
		              static_cast<unsigned long long>(frame), static_cast<unsigned long long>(order),
		              when, slot, written ? 'w' : 'r', static_cast<unsigned long long>(source.address),
		              static_cast<unsigned long long>(source.size));
		std::string path = name;
		const auto  size = source.size;
		sched.DeferPriorityOperation([download, mapped, size, path] {
			download->Invalidate(0, size);
			if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
				std::fwrite(mapped, 1, size, file);
				std::fclose(file);
			}
		});
		saved++;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eTransferRead;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                          nullptr);
	if (order < 6 || order % 50 == 0) {
		::printf("Buffer dump: frame %llu #%llu %s: %u of %u slots saved, %u dispatches left\n",
		         static_cast<unsigned long long>(frame), static_cast<unsigned long long>(order), when,
		         saved, layout.memory_offset_count,
		         static_cast<unsigned>(std::max<int64_t>(0, count.load(std::memory_order_relaxed))));
		std::fflush(stdout);
	}
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock = m_context.LockMutexProfiled();
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions      = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	{
		const std::array<uint32_t, 3> dispatched {thread_group_x, thread_group_y, thread_group_z};
		const std::array<uint32_t, 3> group_size {cs_regs.cs_regs.num_thread_x,
		                                          cs_regs.cs_regs.num_thread_y,
		                                          cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; axis++) {
			const auto size = std::max(group_size[axis], 1u);
			input_info.dispatch_groups[axis] =
			    use_thread_dimensions ? (dispatched[axis] + size - 1u) / size : dispatched[axis];
		}
	}
	ShaderProgram compute_program;
	{
		KYTY_PROFILER_BLOCK("Dispatch::GetComputeProgram");
		compute_program =
		    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}
	if (!compute_program) {
		// Temporary until RT is implemented.
		ResetBindings();
		return;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0] = thread_group_x;
		input_info.dispatch_threads_num[1] = thread_group_y;
		input_info.dispatch_threads_num[2] = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                 thread_group_z, mode))) {
		ResetBindings();
		return;
	}

	const bool large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	if ((large_workgroup || has_sampler) &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	for (uint32_t axis = 0; axis < 3u; axis++) {
		bindings.dispatch_dimensions[axis] = input_info.dispatch_threads_num[axis];
	}
	FindBuffers(bindings);
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	// KYTY_SINGLE_REBIND=1 (live, A/B): the loop below resolves the images again before recording, so
	// this first pass (the recovery transaction only needs the buffers) only repeats view lookups.
	static auto& single_rebind = Common::LiveSwitches::Get("KYTY_SINGLE_REBIND", 0);
	if (single_rebind.load(std::memory_order_relaxed) == 0) {
		RebindImages(bindings);
	}
	RebindBuffers(bindings);
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	do {
		RebindImages(bindings);
		RebindBuffers(bindings);
		const auto        vk_buffer        = buffer.Recorder();
		PreparedBindings* descriptor_stage = &bindings;
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
		has_storage_writes =
		    std::any_of(program.info.images.begin(), program.info.images.end(),
		                [](const auto& image) {
			                return image.written &&
			                       image.resource_class ==
			                           ShaderRecompiler::IR::ImageResourceClass::Storage;
		                }) ||
		    has_storage_writes;
		if (has_storage_writes) {
			// A host fence used to serialize every dispatch. Preserve its read-before-write
			// ordering while allowing the queue to execute asynchronously.
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		const bool shader_time = CommandRecorder::ShaderTimeEnabled();
		if (shader_time) {
			vk_buffer.shaderTimeBegin(program.shader_hash, true);
		}
		DrawDiff::Note(program.shader_hash, 0, thread_group_x * thread_group_y * thread_group_z, 0, 2);
		LogBufferUsers(program, bindings);
		const bool dump_buffers = DumpBuffersWanted(program.shader_hash);
		if (dump_buffers) DumpBuffers(program, bindings, vk_buffer, "before");
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		if (dump_buffers) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			DumpBuffers(program, bindings, vk_buffer, "after");
		}
		if (shader_time) {
			vk_buffer.shaderTimeEnd();
		}

		// The removed host fence also ordered read-only dispatches before later writers.
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	} while (recovery.Retry());
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
                                      uint32_t mode) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	// The arguments are thread counts: IndirectDispatchGroups converts them on the GPU, and the
	// shader bounds its threads by the counts it reads from the same memory.
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopPendingOperations();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock    = m_context.LockMutexProfiled();
	const auto&       cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions   = use_thread_dimensions;
	input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	if (!compute_program) {
		// Temporary until RT is implemented.
		ResetBindings();
		return;
	}
	if (use_thread_dimensions && m_indirect_groups == nullptr) {
		m_indirect_groups = std::make_unique<IndirectDispatchGroups>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler());
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	FindBuffers(bindings);
	const auto& program = *input_info.stage.program;
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	// KYTY_SINGLE_REBIND=1 (live, A/B): the loop below resolves the images again before recording, so
	// this first pass (the recovery transaction only needs the buffers) only repeats view lookups.
	static auto& single_rebind = Common::LiveSwitches::Get("KYTY_SINGLE_REBIND", 0);
	if (single_rebind.load(std::memory_order_relaxed) == 0) {
		RebindImages(bindings);
	}
	RebindBuffers(bindings);
	DispatchBufferRecovery recovery(m_context, bindings, cs_regs.cs_regs.data_addr);
	do {
		RebindImages(bindings);
		// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
		const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
		    args_addr, sizeof(vk::DispatchIndirectCommand), false, false, {},
		    use_thread_dimensions);
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		vk::Buffer     indirect_buffer = args_buffer->Handle();
		vk::DeviceSize indirect_offset = args_offset;
		if (use_thread_dimensions) {
			EXIT_IF(!args_buffer->HasDeviceAddress());
			CheckBdaLive(args_buffer->BufferDeviceAddress(), "RenderCompute indirect args");
			const std::array<uint32_t, 3> local_size {std::max(cs_regs.cs_regs.num_thread_x, 1u),
			                                          std::max(cs_regs.cs_regs.num_thread_y, 1u),
			                                          std::max(cs_regs.cs_regs.num_thread_z, 1u)};
			const auto                    converted = m_indirect_groups->Convert(
                buffer.Recorder(), args_buffer->BufferDeviceAddress() + args_offset, local_size);
			indirect_buffer              = converted.groups_buffer;
			indirect_offset              = converted.groups_offset;
			bindings.dispatch_dimensions = {static_cast<uint32_t>(converted.threads),
			                                static_cast<uint32_t>(converted.threads >> 32u), 0u};
		}
		RebindBuffers(bindings);
		PreparedBindings* descriptor_stage = &bindings;
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
		const auto vk_buffer = buffer.Recorder();
		const bool has_storage_writes =
		    HasShaderBufferWrites(input_info.stage) ||
		    std::any_of(
		        program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			        return image.written && image.resource_class ==
			                                    ShaderRecompiler::IR::ImageResourceClass::Storage;
		        });
		if (has_storage_writes) {
			ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
		}
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask =
		    vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(
		    vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eComputeShader |
		        vk::PipelineStageFlagBits::eTransfer,
		    vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		const bool shader_time = CommandRecorder::ShaderTimeEnabled();
		if (shader_time) {
			vk_buffer.shaderTimeBegin(program.shader_hash, true);
		}
		DrawDiff::Note(program.shader_hash, 0, 0, 0, 3);
		LogBufferUsers(program, bindings);
		const bool dump_buffers = DumpBuffersWanted(program.shader_hash);
		if (dump_buffers) DumpBuffers(program, bindings, vk_buffer, "before");
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		if (dump_buffers) {
			ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
			DumpBuffers(program, bindings, vk_buffer, "after");
		}
		if (shader_time) {
			vk_buffer.shaderTimeEnd();
		}
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	} while (recovery.Retry());
	ResetBindings();
}

} // namespace Libs::Graphics
