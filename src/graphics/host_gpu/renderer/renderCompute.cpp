#include "local/skip-shader.h"
#include "local/vgeo-nativo.h" // (w11, KYTY_NATIVO_VGEO)
#include "local/pathtrace.h" // (pt1, BB_PATHTRACING)
#include "local/envio1.h"
#include "local/rt-local-lights.h"
#include "local/hwrt-scene.h"
#include "local/sync-stats.h"
#include "common/assert.h"
#include "live-trace-gpu.h"
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
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
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
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "frame-capture.h"
#include "frame-gen.h"
#include "native-preparation-state.h"
#include "live-counters.h"

#include <string_view>
#include <thread>
#include <unordered_map>
#include <xxhash.h>
#include "xpr-capture.h"
#include "device-fault.h"
#include "speculation-state.h"
#include "live-census.h"
#include "live-counters.h"
#include "p7-compat.h" // (p7, KYTY_P7: C1)
#include "local/juego3.h" // (juego3, KYTY_JUEGO3)

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include "local/tiron4-counters.h" // (tiron5)
#include <unordered_map>
#include <vector>

extern volatile std::atomic<uint32_t> kyty_local_fill_clear_mode; // (textureCache.cpp)

#include "local/onda-dump-gpu.inc" // (onda4: KYTY_ONDA_DUMP, F12 image sequence)

namespace Libs::Graphics {

// KYTY_DISPATCH_TIMING=<hash>[,<hash>...] (or ;): the GPU time of those programs' normal-path dispatches, from
// timestamps written around each (the interval also holds the tails of earlier work it overlaps), printed for a
// dispatch of KYTY_DISPATCH_TIMING_MS (default 20) or more, with a summary per program every 5 s. (The culling chain's
// GPU hangs, 10-13: a dispatch looping for seconds - its shader's per-lane loop runs a count read from memory - or one
// that never ends.)
namespace DispatchTiming {
constexpr uint32_t Slots = 8192; // (pairs of timestamps)
struct Totals {
	std::atomic<uint64_t> count {0}, ns {0}, max_ns {0};
};
// A slot's dispatch, for a stall: the one whose start the GPU wrote and whose end it did not is the dispatch it is stuck
// in.
// With it, the first bytes of its buffers: as guest memory held them when it was recorded, and as the GPU read them (a
// copy recorded right before it into `snap`).
constexpr uint32_t SnapBuffers = 8, SnapBytes = 64;
struct Record {
	uint64_t tick = 0, hash = 0, sequence = 0;
	uint32_t x = 0, y = 0, z = 0;
	uint32_t                                                     buffers = 0;
	std::array<uint64_t, SnapBuffers>                            address {};
	std::array<std::array<uint32_t, SnapBytes / 4>, SnapBuffers> guest {};
	uint32_t                                                     srt_count = 0; // (the flattened SRT words it was given)
	std::array<uint32_t, 64>                                     srt {};
};
struct State {
	std::vector<uint64_t>     hashes;
	std::deque<Totals>        totals; // (one per hash)
	double                    threshold_ms = 20.0;
	vk::QueryPool             pool {};
	vk::Device                device {};
	uint64_t                  next = 0;
	std::array<Record, Slots> records {};
	std::atomic<int64_t>      printed {0};
	VkBuffer                  snap        = VK_NULL_HANDLE;
	VmaAllocation             snap_memory = nullptr;
	VmaAllocator              allocator   = nullptr;
	const uint8_t*            snap_mapped = nullptr;
	bool                      all         = false; // (KYTY_DISPATCH_TIMING=*: every normal-path dispatch)
};
static void StallDump(uint64_t reached);
static State& Get() {
	static State      state;
	static const bool parsed = [] {
		if (const char* value = std::getenv("KYTY_DISPATCH_TIMING"); value != nullptr && value[0] == '*') {
			state.all = true;
			DeviceFault::g_stall_dump.store(&StallDump, std::memory_order_release);
		}
		for (const char* at = std::getenv("KYTY_DISPATCH_TIMING"); at != nullptr && *at != '\0' && !state.all;) {
			char* end = nullptr;
			state.hashes.push_back(std::strtoull(at, &end, 16));
			state.totals.emplace_back();
			if (end == nullptr || (*end != ',' && *end != ';')) break;
			at = end + 1;
		}
		if (const char* ms = std::getenv("KYTY_DISPATCH_TIMING_MS"); ms != nullptr && *ms != '\0')
			state.threshold_ms = std::atof(ms);
		if (!state.hashes.empty()) DeviceFault::g_stall_dump.store(&StallDump, std::memory_order_release);
		return true;
	}();
	(void)parsed;
	return state;
}
// Before the dispatch: a slot (UINT32_MAX: not timed).
static uint32_t Begin(GraphicContext& graphics, vk::CommandBuffer command, uint64_t hash,
                      const PreparedBindings* bindings = nullptr, BufferCache* cache = nullptr,
                      std::span<const uint32_t> srt = {}) {
	auto& s = Get();
	if (!s.all && (s.hashes.empty() || std::ranges::find(s.hashes, hash) == s.hashes.end())) return UINT32_MAX;
	if (!s.pool) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = Slots * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &s.pool) != vk::Result::eSuccess) return UINT32_MAX;
		s.device = graphics.device;
		VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		create.size  = uint64_t {Slots} * SnapBuffers * SnapBytes;
		create.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VmaAllocationInfo mapped {};
		if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &s.snap, &s.snap_memory, &mapped) == VK_SUCCESS) {
			s.allocator   = graphics.allocator;
			s.snap_mapped = static_cast<const uint8_t*>(mapped.pMappedData);
		}
	}
	const uint64_t sequence = s.next++;
	const uint32_t slot     = static_cast<uint32_t>(sequence % Slots) * 2;
	auto&          record   = s.records[slot / 2];
	record                  = {.tick = 0, .hash = hash, .sequence = sequence};
	record.srt_count        = static_cast<uint32_t>(std::min<size_t>(srt.size(), record.srt.size()));
	std::copy_n(srt.begin(), record.srt_count, record.srt.begin());
	(void)cache;
	if (bindings != nullptr && s.snap_mapped != nullptr && bindings->buffers.size() == bindings->buffer_sources.size()) {
		// (From the descriptor the dispatch reads: its buffer, or the upload ring's copy of a small range the guest wrote.
		// The data starts the adjustment past the descriptor's offset: its range is the size plus the adjustment.)
		for (size_t i = 0; i < bindings->buffers.size() && record.buffers < SnapBuffers; ++i) {
			const auto&    source     = bindings->buffer_sources[i];
			const auto&    descriptor = bindings->buffers[i];
			const uint64_t bytes      = std::min<uint64_t>(source.size, SnapBytes) & ~uint64_t {3};
			if (source.address == 0 || bytes == 0 || !descriptor.buffer || descriptor.range == VK_WHOLE_SIZE ||
			    descriptor.range < source.size)
				continue;
			const auto index      = record.buffers++;
			record.address[index] = source.address;
			(void)Libs::LibKernel::Memory::TryReadBackingToHost(source.address, record.guest[index].data(), bytes);
			const vk::BufferCopy copy {descriptor.offset + (descriptor.range - source.size),
			                           (uint64_t {slot / 2} * SnapBuffers + index) * SnapBytes, bytes};
			command.copyBuffer(descriptor.buffer, s.snap, 1, &copy);
		}
	}
	command.resetQueryPool(s.pool, slot, 2);
	command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, s.pool, slot);
	return slot;
}
// After it: read once the tick's work completed.
static void End(GraphicContext& graphics, CommandScheduler& scheduler, vk::CommandBuffer command, uint32_t slot,
                uint64_t hash, uint32_t x, uint32_t y, uint32_t z) {
	if (slot == UINT32_MAX) return;
	auto& s = Get();
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, slot + 1);
	const double period = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
	const auto   tick   = scheduler.CurrentTick();
	auto&        record = s.records[slot / 2];
	record.tick         = tick;
	record.x            = x;
	record.y            = y;
	record.z            = z;
	scheduler.DeferOperation([&graphics, slot, hash, x, y, z, period, tick] {
		auto&                   s = Get();
		std::array<uint64_t, 2> stamps {};
		if (graphics.device.getQueryPoolResults(s.pool, slot, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64) != vk::Result::eSuccess ||
		    stamps[1] < stamps[0])
			return;
		const auto ns    = static_cast<uint64_t>(static_cast<double>(stamps[1] - stamps[0]) * period);
		const auto index = static_cast<size_t>(std::ranges::find(s.hashes, hash) - s.hashes.begin());
		if (index >= s.totals.size()) { // (KYTY_DISPATCH_TIMING=*: the slow ones alone)
			if (static_cast<double>(ns) / 1e6 >= s.threshold_ms)
				std::printf("DISPATCH-TIME %016llx %.1f ms tick %llu groups %u x %u x %u\n", static_cast<unsigned long long>(hash),
				            static_cast<double>(ns) / 1e6, static_cast<unsigned long long>(tick), x, y, z);
			return;
		}
		auto&      t     = s.totals[index];
		t.count.fetch_add(1, std::memory_order_relaxed);
		t.ns.fetch_add(ns, std::memory_order_relaxed);
		for (auto old = t.max_ns.load(std::memory_order_relaxed); ns > old && !t.max_ns.compare_exchange_weak(old, ns);) {}
		if (static_cast<double>(ns) / 1e6 >= s.threshold_ms)
			std::printf("DISPATCH-TIME %016llx %.1f ms tick %llu groups %u x %u x %u\n", static_cast<unsigned long long>(hash),
			            static_cast<double>(ns) / 1e6, static_cast<unsigned long long>(tick), x, y, z);
		const auto now  = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		auto       last = s.printed.load(std::memory_order_relaxed);
		if (now - last < 5 || !s.printed.compare_exchange_strong(last, now)) return;
		for (size_t i = 0; i < s.hashes.size(); ++i) {
			auto&      sum   = s.totals[i];
			const auto count = sum.count.exchange(0);
			if (count == 0) continue;
			std::printf("DISPATCH-TIMES %016llx: %llu, mean %.2f ms, max %.1f ms\n",
			            static_cast<unsigned long long>(s.hashes[i]), static_cast<unsigned long long>(count),
			            static_cast<double>(sum.ns.exchange(0)) / 1e6 / static_cast<double>(count),
			            static_cast<double>(sum.max_ns.exchange(0)) / 1e6);
		}
		std::fflush(stdout);
	});
}
// The timed dispatches past `reached` (the first few submissions), in recording order: done, started only (the GPU is
// in it), or not started.
static void StallDump(uint64_t reached) {
	auto& s = Get();
	if (!s.pool) return;
	std::vector<const Record*> stalled;
	for (const auto& record: s.records)
		if (record.tick > reached && record.tick <= reached + 4) stalled.push_back(&record);
	std::ranges::sort(stalled, {}, &Record::sequence);
	uint32_t done = 0, shown = 0;
	for (const auto* record: stalled) {
		const auto              slot = static_cast<uint32_t>(record->sequence % Slots) * 2;
		std::array<uint64_t, 4> values {}; // (value, availability) x 2
		const auto              result =
		    s.device.getQueryPoolResults(s.pool, slot, 2, sizeof(values), values.data(), 2 * sizeof(uint64_t),
		                                 vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		if (result != vk::Result::eSuccess && result != vk::Result::eNotReady) {
			std::printf("STALL-TIME query read failed (%d)\n", static_cast<int>(result));
			break;
		}
		const bool started = values[1] != 0, ended = values[3] != 0;
		if (started && ended) {
			++done;
			continue;
		}
		if (shown++ < 40)
			std::printf("STALL-TIME tick %llu #%llu %016llx groups %u x %u x %u: %s (%u timed dispatches done before it)\n",
			            static_cast<unsigned long long>(record->tick), static_cast<unsigned long long>(record->sequence),
			            static_cast<unsigned long long>(record->hash), record->x, record->y, record->z,
			            started ? "STARTED, NOT ENDED" : "not started", done);
		// What the dispatch the GPU is in read: its buffers' first bytes, as guest memory held them when it was recorded and
		// as the GPU copy right before it read them.
		if (started && record->srt_count != 0) {
			std::string words;
			for (uint32_t w = 0; w < record->srt_count; ++w) {
				char word[16];
				std::snprintf(word, sizeof(word), " %08x", record->srt[w]);
				words += word;
			}
			std::printf("STALL-TIME   flattened SRT (%u words):%s\n", record->srt_count, words.c_str());
		}
		if (started && s.snap_mapped != nullptr) {
			(void)vmaInvalidateAllocation(s.allocator, s.snap_memory, 0, VK_WHOLE_SIZE);
			for (uint32_t i = 0; i < record->buffers; ++i) {
				const auto* gpu = reinterpret_cast<const uint32_t*>(
				    s.snap_mapped + (uint64_t {static_cast<uint32_t>(record->sequence % Slots)} * SnapBuffers + i) * SnapBytes);
				std::string guest_text, gpu_text;
				for (uint32_t w = 0; w < SnapBytes / 4; ++w) {
					char word[16];
					std::snprintf(word, sizeof(word), " %08x", record->guest[i][w]);
					guest_text += word;
					std::snprintf(word, sizeof(word), " %08x", gpu[w]);
					gpu_text += word;
				}
				std::printf("STALL-TIME   buffer %u at 0x%llx guest:%s\n", i, static_cast<unsigned long long>(record->address[i]),
				            guest_text.c_str());
				std::printf("STALL-TIME   buffer %u at 0x%llx   gpu:%s%s\n", i, static_cast<unsigned long long>(record->address[i]),
				            gpu_text.c_str(), std::memcmp(gpu, record->guest[i].data(), SnapBytes) != 0 ? "  DIFFERS" : "");
			}
		}
	}
	std::printf("STALL-TIME %zu timed dispatches past tick %llu, %u done\n", stalled.size(),
	            static_cast<unsigned long long>(reached), done);
	std::fflush(stdout);
}
} // namespace DispatchTiming

namespace {
// KYTY_PROG_TRACE=<hash,...> (diagnostic, issue #52): for these programs' normal-path dispatches, a line per dispatch
// with each buffer binding (guest range, written, whether its bytes were CPU-dirty before it was resolved, bound from
// the shader upload ring or a game buffer) and each image binding (range, format, size, image serial, GPU-written).
// "indirect" in the list adds every indirect dispatch (who reads the arguments a traced program wrote). At most 50 lines a
// program, 20000 in all: the programs asked for are not crowded out by others. Out of DispatchDirect, which calls it
// without a condition: a branch there changes that function's control flow, and the compiler then drops its PGO profile
// (10-18: ~3 ms a frame more render-thread time at 1-1 on the laptop).
class ProgTrace {
public:
	ProgTrace(uint64_t shader_hash, uint64_t indirect_args, const PreparedBindings& bindings, BufferCache& cache) {
		if (!Programs().empty()) Begin(shader_hash, indirect_args, bindings, cache);
	}
	void Print(const PreparedBindings& bindings, RenderContext& context, const Common::SlotVector<Image>& images,
	           uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, uint64_t indirect_args) const {
		if (m_on) PrintLine(bindings, context, images, groups_x, groups_y, groups_z, indirect_args);
	}

private:
	static const std::string& Programs() {
		static const std::string programs = [] {
			const char* value = std::getenv("KYTY_PROG_TRACE");
			std::string list  = value != nullptr ? std::string(value) : std::string();
			// (KYTY_RTLS_DEBUG / KYTY_RT_LOCAL_SHADOWS: every dispatch's bindings go to RtLocal::OnDispatch from here,
			// out of DispatchDirect's own code and its PGO profile)
			if (RtLocal::Wanted()) list += list.empty() ? "rtls" : ",rtls";
			return list;
		}();
		return programs;
	}
	[[gnu::noinline]] void Begin(uint64_t shader_hash, uint64_t indirect_args, const PreparedBindings& bindings,
	                             BufferCache& cache) {
		m_traced = Programs() != "rtls" && (Programs().find(fmt::format("{:016x}", shader_hash)) != std::string::npos ||
		                                    (indirect_args != 0 && Programs().find("indirect") != std::string::npos) ||
		                                    CensusNow());
		m_on = m_traced || RtLocal::Wanted();
		if (!m_on) return;
		m_shader_hash = shader_hash;
		if (!m_traced) return;
		for (const auto& source: bindings.buffer_sources)
			m_dirty_before.push_back(source.size != 0 && cache.IsRegionCpuModified(source.address, source.size));
	}
	// (rtq2) KYTY_PROG_TRACE=* : a census of every compute program of the frame, once, KYTY_PROG_TRACE_AFTER_S (default
	// 90) seconds after the first dispatch (in play, not in the menus), for KYTY_PROG_TRACE_FRAMES_S (default 2) seconds:
	// per program at most 4 lines (its bindings, images with address/format/size, written). Used to find the game's
	// SSR / IBL / fog chain by the addresses one writes and the next reads (kRTT_SSR -> CS 9fb5cc274b5e4f82).
	static bool CensusNow() {
		static const bool star = Programs().find('*') != std::string::npos;
		if (!star) return false;
		using Clock = std::chrono::steady_clock;
		static const Clock::time_point first = Clock::now();
		static const double after = [] {
			const char* v = std::getenv("KYTY_PROG_TRACE_AFTER_S");
			return v != nullptr && *v != 0 ? std::atof(v) : 90.0;
		}();
		static const double span = [] {
			const char* v = std::getenv("KYTY_PROG_TRACE_FRAMES_S");
			return v != nullptr && *v != 0 ? std::atof(v) : 2.0;
		}();
		const double t = std::chrono::duration<double>(Clock::now() - first).count();
		return t >= after && t < after + span;
	}
	[[gnu::noinline]] void PrintLine(const PreparedBindings& bindings, RenderContext& context,
	                                 const Common::SlotVector<Image>& images, uint32_t groups_x, uint32_t groups_y,
	                                 uint32_t groups_z, uint64_t indirect_args) const {
		if (RtLocal::Wanted() && bindings.runtime != nullptr && bindings.runtime->program != nullptr && RtLocal::WantsDispatch(m_shader_hash)) {
			const auto& info = bindings.runtime->program->info;
			std::vector<RtLocal::DispatchBuffer> buffers;
			std::vector<RtLocal::DispatchImage>  bound;
			for (size_t i = 0; i < bindings.buffer_sources.size(); ++i)
				buffers.push_back({bindings.buffer_sources[i].address, bindings.buffer_sources[i].size,
				                   i < info.buffers.size() && info.buffers[i].written});
			for (size_t i = 0; i < bindings.images.size(); ++i) {
				const auto& d = bindings.images[i].desc.info;
				bound.push_back({d.data.address, d.extent.width, d.extent.height, d.resources.layers,
				                 d.IsDepth() || d.pixel_format == vk::Format::eR32Sfloat || d.pixel_format == vk::Format::eR16Unorm,
				                 i < info.images.size() && info.images[i].written});
			}
			RtLocal::OnDispatch(m_shader_hash, buffers, bound, bindings.runtime->resources.flattened_srt);
		}
		if (!m_traced) return;
		static std::mutex                             lines_mutex;
		static std::unordered_map<uint64_t, uint32_t> per_program;
		static uint32_t                               lines = 0;
		{
			std::lock_guard lock(lines_mutex);
			static const bool star = Programs().find('*') != std::string::npos;
			if (lines >= 20000 || per_program[m_shader_hash]++ >= (star ? 4u : 50u)) return;
			++lines;
		}
		const auto& program = *bindings.runtime->program;
		std::string text    = fmt::format("PROG-TRACE cs={:016x} tick={} groups={},{},{} indirect=0x{:x}", m_shader_hash,
		                                  context.GetCommandScheduler().CurrentTick(), groups_x, groups_y, groups_z,
		                                  indirect_args);
		const auto ring = context.GetBufferCache().GetShaderUploadBuffer().Handle();
		for (size_t i = 0; i < bindings.buffer_sources.size() && i < bindings.buffers.size(); ++i) {
			const auto& source = bindings.buffer_sources[i];
			text += fmt::format(" b{}=0x{:x}+0x{:x}{}{}{}", i, source.address, source.size,
			                    i < program.info.buffers.size() && program.info.buffers[i].written ? "w" : "",
			                    m_dirty_before[i] ? ":dirty" : ":clean", bindings.buffers[i].buffer == ring ? ":ring" : ":buf");
		}
		for (size_t i = 0; i < bindings.images.size(); ++i) {
			const auto* image = images.try_get(bindings.images[i].image_id);
			if (image == nullptr) continue;
			text += fmt::format(" i{}=0x{:x}+0x{:x}:{}:{}x{}:s{}{}", i, image->info.data.address, image->info.data.size,
			                    static_cast<uint32_t>(image->info.guest_format), image->info.extent.width,
			                    image->info.extent.height, image->serial, image->IsGpuModified() ? ":gpu" : "");
		}
		std::printf("%s\n", text.c_str());
	}

	bool              m_on          = false;
	bool              m_traced      = false;
	uint64_t          m_shader_hash = 0;
	std::vector<bool> m_dirty_before;
};
} // namespace

static uint64_t BufferDescriptorSize(const ShaderBufferResource& descriptor) {
	const uint64_t records = descriptor.NumRecords();
	const uint64_t stride  = descriptor.Stride();
	return stride == 0 ? records : records * stride;
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = BufferDescriptorSize(source);
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size);

// The shader and user data of a dispatch (a compute clear's: RenderExecutor::m_meta_clears).
static uint64_t MetaClearKey(const HW::ComputeShaderInfo& cs) {
	std::array<uint32_t, 2 + HW::UserSgprInfo::SGPRS_MAX> words {static_cast<uint32_t>(cs.cs_regs.data_addr),
	                                                              static_cast<uint32_t>(cs.cs_regs.data_addr >> 32u)};
	const auto count = std::min<uint32_t>(cs.cs_regs.user_sgpr, HW::UserSgprInfo::SGPRS_MAX);
	std::copy_n(cs.cs_user_sgpr.value, count, words.begin() + 2);
	return XXH3_64bits(words.data(), (2 + count) * sizeof(uint32_t));
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer& buffer, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z,
                                                uint32_t mode, uint64_t& cleared) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	// Only a written buffer can be consumed as a clear: without one this is always false (and
	// it ran a texture-cache lookup per buffer on every dispatch).
	if (std::none_of(program.info.buffers.begin(), program.info.buffers.end(),
	                 [](const auto& resource) { return resource.written; })) {
		return false;
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	// Consuming takes ClearMeta of a written buffer, which holds only for a metadata surface
	// (IsMeta): without one, the read checks and the fill evaluation below cannot change the answer.
	bool written_meta = false;
	for (uint32_t i = 0; i < program.info.buffers.size() && !written_meta; i++)
		written_meta = program.info.buffers[i].written &&
		               cache.IsMeta(DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]).Base48());
	if (!written_meta) {
		return false;
	}
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
		ShaderBufferResource fill_descriptor;
		uint32_t             fill_value   = 0;
		uint64_t             fill_size    = 0;
		const bool           uniform_fill =
		    ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, fill_descriptor,
		                             fill_value, fill_size) ||
		    ResolveComputePatternFill(input, group_x, group_y, group_z, mode, fill_descriptor,
		                              fill_value, fill_size);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				const bool known =
				    uniform_fill && descriptor.Base48() == fill_descriptor.Base48();
				if (known ? cache.ClearMeta(descriptor.Base48(), fill_value)
				          : cache.ClearMeta(descriptor.Base48())) {
					cleared = descriptor.Base48();
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
	const auto& resources = input.stage.resources;
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
	const auto     size        = BufferDescriptorSize(descriptor);
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

static bool ResolveComputePatternFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                      uint32_t group_y, uint32_t group_z, uint32_t mode,
                                      ShaderBufferResource& resolved_descriptor,
                                      uint32_t& resolved_clear, uint64_t& resolved_size) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	const auto& user_data = resources.user_data;
	if (program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.images.empty() || !program.info.samplers.empty() ||
	    program.info.uses_dma || input.dispatch_thread_dimensions || mode != 0x41u ||
	    user_data.size() != 10 || program.user_data_base != 0) {
		return false;
	}
	const auto& resource   = program.info.buffers.front();
	const auto& raw        = resources.buffers.front();
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
	if (!resource.written || resource.read || resource.atomic || resource.scalar ||
	    resource.max_byte_extent != 4 ||
	    (resource.formatted && descriptor.Format() != Prospero::BufferFormat::k32UInt) ||
	    (descriptor.Stride() != 4 && descriptor.Stride() != 0) || descriptor.SwizzleEnabled() ||
	    descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    resource.packed_stride != descriptor.PackedStride() || raw.dword_count != 4 ||
	    descriptor.Base48() == 0) {
		return false;
	}
	for (uint32_t i = 0; i < raw.dword_count; i++) {
		if (raw.dwords[i] != user_data[i]) {
			return false;
		}
	}
	const uint32_t clear  = user_data[4];
	const uint32_t period = user_data[9];
	const uint32_t slots  = period == 0u ? 4u : std::min(period, 4u);
	for (uint32_t slot = 1; slot < slots; slot++) {
		if (user_data[4 + slot] != clear) {
			return false;
		}
	}
	if (input.threads_num[0] != 64 || input.threads_num[1] != 1 || input.threads_num[2] != 1 ||
	    group_x == 0 || group_y != 1 || group_z != 1 || !input.group_id[0] || input.group_id[1] ||
	    input.group_id[2] || input.thread_ids_num != 1 || input.wave_size != 64 ||
	    input.tg_size_en) {
		return false;
	}
	const uint64_t count = user_data[8];
	const auto     size  = BufferDescriptorSize(descriptor);
	if (count == 0 || size == 0 || size > UINT32_MAX || count * sizeof(uint32_t) != size ||
	    group_x != (count + input.threads_num[0] - 1) / input.threads_num[0]) {
		return false;
	}
	resolved_descriptor = descriptor;
	resolved_clear      = clear;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
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
			    groups[axis] * threads > UINT32_MAX) return false;
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
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size) &&
	    !ResolveComputePatternFill(input, group_x, group_y, group_z, mode, descriptor,
	                               packed_clear, size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		// Track deferred DCC state while the original dispatch writes the metadata allocation.
		cache.TrackDccFill(descriptor.Base48(), size, packed_clear);
		static std::atomic<uint32_t> logged_metadata_clears {0};
		if (logged_metadata_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: metadata fill shader=0x%016" PRIx64
			     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
			     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
		}
		// A fill over images whose ranges it does not match (the game clears render targets in 4 MiB pieces and in spans
		// of several): the same fill on the buffer cache's buffer, and the images over it note the value - one all of whose
		// buffer writes were such fills is cleared instead of uploaded (TextureCache::InitializeImage).
		const auto base = descriptor.Base48();
		if (kyty_local_fill_clear_mode.load(std::memory_order_relaxed) != 0 && Spec::Current() == nullptr &&
		    ((base | size) & 3u) == 0 && cache.HasTrackedDataOverlap(base, size)) {
			const auto [buffer, offset] = m_context.GetBufferCache().ObtainBuffer(base, size, true, true);
			if (buffer != nullptr && (offset & 3u) == 0) {
				buffer->Fill(offset, size, packed_clear);
				cache.InvalidateMemoryFromGPU(base, size, "fill", &packed_clear);
				return true;
			}
		}
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

void RenderExecutor::PredictDispatch(const HW::Shader& sh_ctx, uint32_t thread_group_x, uint32_t thread_group_y,
                                     uint32_t thread_group_z, uint32_t mode, Spec::State& record) {
	auto&       learned = Catalog();
	const auto& cs      = sh_ctx.GetCs();
	const auto& regs    = cs.cs_regs;
	if (regs.data_addr == learned.m_linear_copy_shader && ShaderMapGeneration() == learned.m_linear_copy_generation) {
		const DemonsSouls::LinearCopyDispatch dispatch {
		    .user_data          = std::span<const uint32_t> {cs.cs_user_sgpr.value, regs.user_sgpr},
		    .threads            = {regs.num_thread_x, regs.num_thread_y, regs.num_thread_z},
		    .group_id           = {regs.tgid_x_en, regs.tgid_y_en, regs.tgid_z_en},
		    .thread_ids         = regs.tidig_comp_cnt + 1u,
		    .workgroup_register = regs.user_sgpr,
		    .tg_size            = regs.tg_size_en,
		    .thread_dimensions  = (mode & (1u << 5u)) != 0};
		if (const auto copy = DemonsSouls::LinearCopyOf(dispatch, thread_group_x, thread_group_y, thread_group_z, mode)) {
			auto& buffers = m_context.GetBufferCache();
			if (buffers.IsRegionGpuModified(copy->dst, copy->bytes) || buffers.IsRegionGpuModified(copy->src, copy->bytes))
				record.NoteWritten(copy->dst, copy->bytes);
			else
				record.Journal(copy->dst, copy->src, static_cast<uint32_t>(copy->bytes), true);
			return;
		}
	}
	const auto read = CatalogRead();
	if (const auto found = learned.m_meta_clears.find(MetaClearKey(cs)); found != learned.m_meta_clears.end())
		record.PredictClear(found->second);
}

namespace {
// (render2) KYTY_ASYNC_COMPUTE_PIPELINES=1 (off by default): a normal-path dispatch whose compute pipeline is not made
// yet no longer always stops the GPU thread for the driver. Each frame the first KYTY_ASYNC_COMPUTE_SYNC_MS (default
// 2) of such compiles are made here as before, so a frame that brings a few new pipelines dispatches them all; past
// that the pipeline goes to a compile worker (PipelineCache::TryCreateComputePipelineAsync), the GPU thread waits for
// it at most KYTY_ASYNC_COMPUTE_WAIT_MS a frame (default 2) and the dispatch is skipped while it compiles, as a draw
// is with KYTY_ASYNC_DRAW_PIPELINES. Never skipped: the TAA pass (frame generation and the upscalers hang on it), a
// captured frame, and a dispatch whose pipeline has been compiling for KYTY_ASYNC_COMPUTE_HOLD_MS (default 250):
// those wait for the worker. Measured basis (16 GB profile, no pipeline warmup): 14-145 compute pipelines in one
// frame at a zone's load, ~0.6 ms each, 13-132 ms of dispatch.pipeline in that frame.
struct AsyncComputeDispatches {
	bool    on             = false;
	int64_t sync_budget_us = 2000, wait_budget_us = 2000, hold_ms = 250;
	// The GPU thread's: this frame's compiles made here and waits for the workers, and when each program's dispatch
	// was first skipped.
	uint64_t frame   = UINT64_MAX;
	int64_t  sync_us = 0, waited_us = 0;
	std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> skipping;
	// Since the start, and as the last [async-compute] line had them.
	struct Totals {
		uint64_t made_here = 0, made_here_us = 0, queued = 0, waited_us = 0, skipped = 0, held = 0, held_us = 0;
		uint64_t compiled = 0, compiled_us = 0;
	};
	Totals                                total, reported;
	bool                                  active = false;
	std::chrono::steady_clock::time_point reported_at {};
};
AsyncComputeDispatches& AsyncComputeState() {
	static AsyncComputeDispatches state = [] {
		AsyncComputeDispatches made;
		const auto             number = [](const char* name, double fallback, double low, double high) {
			const char* value = std::getenv(name);
			return std::clamp(value != nullptr && *value != '\0' ? std::atof(value) : fallback, low, high);
		};
		const char* value   = std::getenv("KYTY_ASYNC_COMPUTE_PIPELINES");
		made.on             = value != nullptr && *value != '\0' && std::string_view(value) != "0";
		if (Tiron4::PipeOn()) made.on = true; // (tiron5: KYTY_TIRON5_PIPE=1)
		made.sync_budget_us = static_cast<int64_t>(number("KYTY_ASYNC_COMPUTE_SYNC_MS", 2.0, 0.0, 1000.0) * 1000.0);
		made.wait_budget_us = static_cast<int64_t>(number("KYTY_ASYNC_COMPUTE_WAIT_MS", 2.0, 0.0, 1000.0) * 1000.0);
		made.hold_ms        = static_cast<int64_t>(number("KYTY_ASYNC_COMPUTE_HOLD_MS", 250.0, 0.0, 60000.0));
		if (made.on) {
			std::printf("Async compute pipelines: on (KYTY_ASYNC_COMPUTE_PIPELINES=1): %.1f ms of compiles a frame on the "
			            "GPU thread (KYTY_ASYNC_COMPUTE_SYNC_MS), then on the compile workers, waited for %.1f ms a frame "
			            "(KYTY_ASYNC_COMPUTE_WAIT_MS), the dispatch skipped meanwhile for at most %lld ms "
			            "(KYTY_ASYNC_COMPUTE_HOLD_MS)\n",
			            static_cast<double>(made.sync_budget_us) / 1000.0, static_cast<double>(made.wait_budget_us) / 1000.0,
			            static_cast<long long>(made.hold_ms));
			std::fflush(stdout);
		}
		return made;
	}();
	return state;
}
// A line at most every 5 s, only after something happened: what the switch did since the last one.
void AsyncComputeReport(AsyncComputeDispatches& state, const PipelineCache& cache) {
	if (!state.active) return;
	const auto now = std::chrono::steady_clock::now();
	if (now - state.reported_at < std::chrono::seconds(5)) return;
	state.total.compiled    = cache.AsyncComputeCompiled();
	state.total.compiled_us = cache.AsyncComputeCompileUs();
	const auto& t = state.total;
	const auto& r = state.reported;
	const auto  ms = [](uint64_t us) { return static_cast<double>(us) / 1000.0; };
	std::printf("[async-compute] since the last line: %llu pipelines made on the GPU thread (%.1f ms), %llu sent to the "
	            "workers, %llu compiled there (%.1f ms off the GPU thread), waited %.1f ms, %llu dispatches skipped, %llu "
	            "held (%.1f ms); total skipped %llu\n",
	            static_cast<unsigned long long>(t.made_here - r.made_here), ms(t.made_here_us - r.made_here_us),
	            static_cast<unsigned long long>(t.queued - r.queued), static_cast<unsigned long long>(t.compiled - r.compiled),
	            ms(t.compiled_us - r.compiled_us), ms(t.waited_us - r.waited_us),
	            static_cast<unsigned long long>(t.skipped - r.skipped), static_cast<unsigned long long>(t.held - r.held),
	            ms(t.held_us - r.held_us), static_cast<unsigned long long>(t.skipped));
	std::fflush(stdout);
	state.reported    = state.total;
	state.reported_at = now;
	state.active      = false; // (until a dispatch next finds its pipeline not made)
}
} // namespace

// (render4) KYTY_DS_LINEAR_COPY_HOOK: the copy an announcement of the game's linear copy function makes, under the
// terms DispatchDirect's register check below copies under: the copy shader proved by its program (m_linear_copy_shader)
// and the shader map unchanged since, no frame capture, no speculation, no live shader skip; every 64th use is left to
// the game's packets, whose dispatch takes the normal path and proves the program again. The census scopes are the
// dispatch's (the [render] line counts it as a dispatch and its time as linear_copy).
bool RenderExecutor::DsLinearCopy(CommandBuffer& buffer, uint64_t dst, uint64_t src, uint32_t count, uint32_t period,
                                  uint64_t commands, uint64_t commands_bytes) {
	if (buffer.IsInvalid() || Role() != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed))
		return false;
	auto&      learned = Catalog();
	const auto shader  = learned.m_linear_copy_shader.load(std::memory_order_relaxed);
	if (shader == 0 || ShaderMapGeneration() != learned.m_linear_copy_generation) return false;
	if ((++m_linear_copy_hook_uses & 63u) == 0) return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	LiveCensus::Scope census(LiveCensus::Dispatch, shader, 0);
	LiveCensus::SetDispatchShader(LiveCensus::LinearCopyShader);
	Envio1::NoteDispatch();
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard   lock(m_context.GetMutex());
	envio1_lock.reset();
	LiveCensus::Scope census_copy(LiveCensus::DispatchPhase, shader, 11);
	return DemonsSouls::TryLinearCopyRanges(m_context.GetBufferCache(), dst, src, count, period, commands, commands_bytes);
}

// (copia1, KYTY_COPIA1; local/copia1.h) DsLinearCopy for a run of announcements: its terms checked once, then each copy
// as DsLinearCopy makes it (the 64th-use rule per copy), in order, under one lock and one census scope; the first copy
// not made ends the run.
uint32_t RenderExecutor::DsLinearCopyBatch(CommandBuffer& buffer, const DsLinearCopyItem* items, uint32_t n) {
	if (n == 0 || buffer.IsInvalid() || Role() != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed))
		return 0;
	auto&      learned = Catalog();
	const auto shader  = learned.m_linear_copy_shader.load(std::memory_order_relaxed);
	if (shader == 0 || ShaderMapGeneration() != learned.m_linear_copy_generation) return 0;
	m_context.GetCommandScheduler().PopPendingOperations();
	LiveCensus::Scope census(LiveCensus::Dispatch, shader, 0);
	LiveCensus::SetDispatchShader(LiveCensus::LinearCopyShader);
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard   lock(m_context.GetMutex());
	envio1_lock.reset();
	LiveCensus::Scope census_copy(LiveCensus::DispatchPhase, shader, 11);
	uint32_t made = 0;
	for (; made < n; ++made) {
		if ((++m_linear_copy_hook_uses & 63u) == 0) break;
		Envio1::NoteDispatch();
		const auto& item = items[made];
		if (!DemonsSouls::TryLinearCopyRanges(m_context.GetBufferCache(), item.dst, item.src, item.count, item.period,
		                                      item.commands, item.commands_bytes))
			break;
	}
	return made;
}

// (F5 pase 1, KYTY_NATIVO_PASE_SSRCLEAR; local/ds-nativo-pase.h) The SSRClear pass natively: the game's texture clear
// (eboot+0x9ad560: its clear shader over the whole of one 2D texture) as a Vulkan clear of the texture cache's image of
// that T#. Only when the image is exactly the view (one level, one layer, the T#'s size, colour, no DCC): the clear
// covers all of it. Anything else: false, and the game's packets run as always.
bool RenderExecutor::DsNativeClear(CommandBuffer& buffer, const uint32_t* tsharp) {
	if (buffer.IsInvalid() || Role() != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed) || tsharp == nullptr)
		return false;
	// T# fields (shaderBindings.h ShaderTextureResource): Base40, Width5 / Height5, BaseLevel, LastLevel, BaseArray5,
	// Depth, MetaCompress / WriteCompress.
	const uint64_t address    = ((tsharp[0] | (uint64_t {tsharp[1]} << 32u)) & 0xFFFFFFFFFFull) << 8u;
	const uint32_t width      = (((tsharp[1] >> 30u) & 0x3u) | ((tsharp[2] & 0xFFFu) << 2u)) + 1u;
	const uint32_t height     = ((tsharp[2] >> 14u) & 0x3FFFu) + 1u;
	const uint32_t base_level = (tsharp[3] >> 12u) & 0xFu;
	const uint32_t last_level = (tsharp[3] >> 16u) & 0xFu;
	const uint32_t base_array = (tsharp[4] >> 16u) & 0x1FFFu;
	const uint32_t depth      = tsharp[4] & 0x1FFFu;
	const bool     compressed = ((tsharp[6] >> 20u) & 0x3u) != 0;
	if (address == 0 || base_level != 0 || last_level != 0 || base_array != 0 || depth != 0 || compressed) return false;
	m_context.GetCommandScheduler().PopPendingOperations();
	Envio1::NoteDispatch();
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard              lock(m_context.GetMutex());
	envio1_lock.reset();
	auto&         cache = m_context.GetTextureCache();
	const ImageId id    = cache.FindImageFromRange(address, 1, false);
	if (!id) return false;
	uint64_t size = 0;
	{
		std::scoped_lock cache_lock {cache.m_lock};
		const auto*      image = cache.TryGetImage(id);
		if (image == nullptr || image->depth_id || image->info.IsDepth() || image->info.IsVolume() ||
		    image->info.samples != 1 || image->info.data.address != address || image->info.resources.levels != 1 ||
		    image->info.TransferLayers() != 1 || image->info.extent.width != width || image->info.extent.height != height)
			return false;
		size = image->info.data.size;
	}
	return size != 0 && cache.ClearImageFromBuffer(buffer, address, size, 0u);
}

// (F5 pase 2, B1) KYTY_NATIVO_PASE_SOMBRAS=3: ver render.h. Mismo marco que DsNativeClear.
bool RenderExecutor::DsShadowCopyLayer(CommandBuffer& buffer, uint64_t address, uint32_t layer, const char** why) {
	*why = "?";
	if (buffer.IsInvalid() || Role() != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed) || address == 0)
		return *why = "no se puede ahora (especulacion, captura o buffer)", false;
	m_context.GetCommandScheduler().PopPendingOperations();
	Envio1::NoteDispatch();
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard              lock(m_context.GetMutex());
	envio1_lock.reset();
	auto&         cache = m_context.GetTextureCache();
	// (La imagen que uso el draw anterior como profundidad, si es la de esta direccion; si no, por su rango.)
	const ImageId id    = (m_ds_last_depth_id && m_ds_last_depth_addr == address) ? m_ds_last_depth_id
	                                                                               : cache.FindImageFromRange(address, 1, false);
	if (!id) return *why = "la cache no tiene la imagen", false;
	auto&            graphics = m_context.GetGraphics();
	std::scoped_lock cache_lock {cache.m_lock};
	auto*            image = cache.TryGetImage(id);
	if (image == nullptr || image->backing.image == nullptr || image->backing.sparse != nullptr)
		return *why = "imagen sin vk::Image o dispersa", false;
	if (!image->info.IsDepth() || image->info.samples != 1 || layer >= image->backing.layers || image->backing.mip_levels < 1)
		return *why = "no es una imagen de profundidad con esa capa", false;
	// La imagen propia: una vez, del formato y tamano de la de la cache (una capa).
	static VkImage       own_image  = VK_NULL_HANDLE;
	static VmaAllocation own_memory = nullptr;
	static vk::Format    own_format = vk::Format::eUndefined;
	static vk::Extent3D  own_extent {};
	const vk::Extent3D   extent {image->backing.extent.width, image->backing.extent.height, 1};
	if (own_image != VK_NULL_HANDLE && (own_format != image->backing.format || own_extent != extent)) return *why = "otro formato", false;
	if (own_image == VK_NULL_HANDLE) {
		VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		create.imageType     = VK_IMAGE_TYPE_2D;
		create.format        = static_cast<VkFormat>(image->backing.format);
		create.extent        = {extent.width, extent.height, 1};
		create.mipLevels     = 1;
		create.arrayLayers   = 1;
		create.samples       = VK_SAMPLE_COUNT_1_BIT;
		create.tiling        = VK_IMAGE_TILING_OPTIMAL;
		create.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
		create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VmaAllocationCreateInfo allocate {};
		allocate.usage = VMA_MEMORY_USAGE_AUTO;
		if (vmaCreateImage(graphics.allocator, &create, &allocate, &own_image, &own_memory, nullptr) != VK_SUCCESS) {
			own_image = VK_NULL_HANDLE;
			return *why = "no se pudo crear la imagen propia", false;
		}
		own_format = image->backing.format;
		own_extent = extent;
	}
	vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eDepth;
	buffer.EndRendering(); // (como DispatchDirect: una copia no puede ir dentro de un pase de render; el siguiente draw lo reabre)
	const vk::CommandBuffer command = buffer.Handle();
	buffer.InvalidateGraphicsState();
	// La capa de la cache a TransferSrc con su propio Transit; el estado previo, guardado para devolverla.
	const VulkanImageState previous = image->CurrentState();
	ImageSubresourceRange  range {};
	range.base_level  = 0;
	range.level_count = 1;
	range.base_layer  = layer;
	range.layer_count = 1;
	image->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
	// La propia a TransferDst (su contenido anterior no importa).
	vk::ImageMemoryBarrier2 to_dst {};
	to_dst.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	to_dst.srcAccessMask       = vk::AccessFlagBits2::eNone;
	to_dst.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	to_dst.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	to_dst.oldLayout           = vk::ImageLayout::eUndefined;
	to_dst.newLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.image               = vk::Image(own_image);
	to_dst.subresourceRange    = vk::ImageSubresourceRange {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &to_dst;
	command.pipelineBarrier2(dependency);
	vk::ImageCopy region {};
	region.srcSubresource = vk::ImageSubresourceLayers {aspect, 0, layer, 1};
	region.dstSubresource = vk::ImageSubresourceLayers {aspect, 0, 0, 1};
	region.extent         = extent;
	command.copyImage(image->backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(own_image),
	                  vk::ImageLayout::eTransferDstOptimal, 1, &region);
	// La propia, lista para leerse; la de la cache, de vuelta a su estado previo.
	vk::ImageMemoryBarrier2 to_read = to_dst;
	to_read.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	to_read.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	to_read.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	to_read.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead;
	to_read.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_read.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
	dependency.pImageMemoryBarriers = &to_read;
	command.pipelineBarrier2(dependency);
	if (previous.layout != vk::ImageLayout::eUndefined)
		image->Transit(previous.layout, previous.access_mask, range, command);
	*why = "copiada";
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode,
                                    uint64_t indirect_args) {
	FrameCapture::Scope frame_capture("Dispatch", true);
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();
	// (w11, KYTY_NATIVO_VGEO) (w20) Every compute dispatch tries the table way FIRST, before the normal path's prologue
	// (traces, census and counter scopes, Envio1, SkipShader paint, the null/valid shader checks): the normal path did the
	// same TableDispatch right after that prologue, so a dispatch the table takes is recorded the same way, ~1.6 us
	// sooner (measured w11: 0.68 us/dispatch this way). TableDispatch refuses by itself what cannot go the table way (not
	// seen twice by the normal path, the linear copy / meta clear specials, TAA under frame generation, HWRT, path
	// tracing, SkipShader, capture); those continue below without a second table attempt. No per-CS list: before (w18),
	// a learned list of 48 CS (>= 2/f, rows emptied every 300 frames) took 157 of ~1320 dispatches/f.
	struct VgeoTimer {
		std::atomic<uint64_t>* count = nullptr; std::atomic<uint64_t>* ns = nullptr; uint64_t t0 = 0;
		~VgeoTimer() { if (count != nullptr) { count->fetch_add(1, std::memory_order_relaxed); ns->fetch_add(VgeoNativo::Now() - t0, std::memory_order_relaxed); } }
	} vgeo_timer;
	bool table_tried = false;
	if (VgeoNativo::Mode() != 0) {
		const uint64_t vgeo_cs = sh_ctx.GetCs().cs_regs.data_addr;
		const auto&    paint   = SkipShader::Paint::Get();
		vgeo_timer.t0          = VgeoNativo::Now();
		if (VgeoNativo::TableFirst() && vgeo_cs != 0 && kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) != 0 &&
		    Spec::Current() == nullptr && !FrameCapture::Active() && !OndaDump::Recording() && !paint.list && !paint.live &&
		    !Juego3::FixCapable() && (mode & (1u << 5u)) == 0 &&
		    (indirect_args != 0 || (thread_group_x != 0 && thread_group_y != 0 && thread_group_z != 0))) {
			buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id, thread_group_x,
			                    thread_group_y, thread_group_z, mode, vgeo_cs);
			std::optional<RenderLockGuard> vgeo_lock;
			if (Role() == 0) vgeo_lock.emplace(m_context.GetMutex());
			m_dispatch_declined = nullptr;
			if (TableDispatch(buffer, thread_group_x, thread_group_y, thread_group_z, indirect_args, false)) {
				vgeo_timer.count = &VgeoNativo::g_fast; vgeo_timer.ns = &VgeoNativo::g_fast_ns;
				return;
			}
			table_tried      = true; // (its refusal stands: the normal path below does not ask the table again)
			vgeo_timer.count = &VgeoNativo::g_normal; vgeo_timer.ns = &VgeoNativo::g_normal_ns;
		} else {
			vgeo_timer.count = &VgeoNativo::g_val_normal; vgeo_timer.ns = &VgeoNativo::g_val_normal_ns;
		}
	}
	{
		// (Diagnostic, KYTY_DISPATCH_TRACE_ADDR=<cs address hex>: that shader's dispatch sizes and user data.)
		static const uint64_t trace = std::getenv("KYTY_DISPATCH_TRACE_ADDR") != nullptr
		                                  ? std::strtoull(std::getenv("KYTY_DISPATCH_TRACE_ADDR"), nullptr, 16)
		                                  : 0;
		static std::atomic<uint32_t> traced {0};
		if (trace != 0 && sh_ctx.GetCs().cs_regs.data_addr == trace && traced.fetch_add(1) < 400) {
			const auto& cs = sh_ctx.GetCs();
			std::printf("DISPATCH-TRACE cs 0x%llx groups %u %u %u indirect 0x%llx threads %u %u %u user",
			            static_cast<unsigned long long>(trace), thread_group_x, thread_group_y, thread_group_z,
			            static_cast<unsigned long long>(indirect_args), cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
			            cs.cs_regs.num_thread_z);
			for (uint32_t i = 0; i < 16; i++) std::printf(" %08x", cs.cs_user_sgpr.value[i]);
			std::printf("\n");
		}
	}
	if (FrameCapture::Active()) {
		FrameCapture::g_call.cs_address = sh_ctx.GetCs().cs_regs.data_addr;
		FrameCapture::g_call.indirect   = indirect_args;
		FrameCapture::g_call.groups[0]  = thread_group_x;
		FrameCapture::g_call.groups[1]  = thread_group_y;
		FrameCapture::g_call.groups[2]  = thread_group_z;
	}
	LiveCensus::Scope census(LiveCensus::Dispatch, sh_ctx.GetCs().cs_regs.data_addr, indirect_args != 0);
	LiveTrace::MarkAfter gpu_mark {[&] { return m_context.GetCommandScheduler().Current().RawHandle(); },
	                               LiveTrace::MarkDispatch, sh_ctx.GetCs().cs_regs.data_addr};
	if (LiveCounters::g_last_dispatch_shader != 0) {
		LiveCounters::Add(LiveCounters::DispatchAfterDispatch);
		if (LiveCounters::g_last_dispatch_shader == sh_ctx.GetCs().cs_regs.data_addr)
			LiveCounters::Add(LiveCounters::DispatchSameShader);
	}
	LiveCounters::g_last_dispatch_shader = sh_ctx.GetCs().cs_regs.data_addr;
	SyncStats::g_shader = sh_ctx.GetCs().cs_regs.data_addr;
	const Juego3::DispatchScope juego3_cs(sh_ctx.GetCs().cs_regs.data_addr); // (juego3: the writes below are this shader's)

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	// (The GPU thread's: a speculation's thread does not take it.)
	std::optional<RenderLockGuard> lock;
	{
		LiveCensus::Scope census_lock(LiveCensus::DispatchPhase, sh_ctx.GetCs().cs_regs.data_addr, 12);
		Envio1::NoteDispatch();
		const Envio1::Scope envio1_lock(Envio1::Lock); // (envio1)
		if (Role() == 0) lock.emplace(m_context.GetMutex());
	}
	if (const auto* spec = Spec::Current(); spec != nullptr && spec->refused != nullptr) return;
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (!ShaderAddressValid(sh_ctx.GetCs().cs_regs.data_addr)) {
		return;
	}
	SkipShader::Paint::NoteDispatch(sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y, thread_group_z);
	if (SkipShader::Skipped(sh_ctx.GetCs().cs_regs.data_addr) ||
	    SkipShader::Paint::SkipDispatchLive(sh_ctx.GetCs().cs_regs.data_addr))
		return; // live `skipcs` probe; KYTY_PAINT_LIVE

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	// A compute shader the normal path dispatched before (TableDispatchSeen) whose program allows table mode (an
	// indirect dispatch of threads: the normal path reads its counts).
	const bool thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	if (!table_tried) m_dispatch_declined = nullptr; // (w20: a refusal of the table-first attempt above stands)
	if (!table_tried && kyty_local_table_dispatch_mode.load(std::memory_order_relaxed) != 0 && !OndaDump::Recording() && // (onda4)
	    (!thread_dimensions || indirect_args == 0) &&
	    (indirect_args != 0 || (thread_group_x != 0 && thread_group_y != 0 && thread_group_z != 0))) {
		LiveCensus::Scope census_table(LiveCensus::DispatchPhase, sh_ctx.GetCs().cs_regs.data_addr, 10);
		if (TableDispatch(buffer, thread_group_x, thread_group_y, thread_group_z, indirect_args, thread_dimensions)) return;
	}
	// A direct dispatch of no thread groups (with thread dimensions: no threads) runs nothing: skipped before its
	// program is prepared (about 60 a frame at 1-1, each evaluating its resource tables first).
	if (indirect_args == 0 && (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0)) return;
	// A speculative translation dispatches only the table way, or copies the linear copy shader's way
	// (speculation-state.h).
	const bool spec    = Spec::Current() != nullptr;
	auto&      learned = Catalog();
	if (spec && sh_ctx.GetCs().cs_regs.data_addr != learned.m_linear_copy_shader) {
		// (A compute clear's hole clears its surface: the draws after it are holes that apply the clear.)
		if (indirect_args == 0) {
			const auto read = CatalogRead();
			if (const auto found = learned.m_meta_clears.find(MetaClearKey(sh_ctx.GetCs())); found != learned.m_meta_clears.end())
				Spec::Current()->PredictClear(found->second);
		}
		return Spec::Refuse(m_dispatch_declined != nullptr ? m_dispatch_declined
		                    : thread_dimensions            ? "dispatch (thread dimensions)"
		                                                   : "dispatch");
	}
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

	// The linear copy shader, while its code stands: checked from the registers without its program (700+ small
	// copies a frame in Boletaria, each preparing the same program).
	// (Another executor's catalog: its owner proves the program now and then.)
	if (indirect_args == 0 && cs_regs.cs_regs.data_addr == learned.m_linear_copy_shader &&
	    ShaderMapGeneration() == learned.m_linear_copy_generation && ((++m_linear_copy_uses & 63u) != 0 || Role() != 0) &&
	    !FrameCapture::Active()) {
		const auto&                           regs = cs_regs.cs_regs;
		const DemonsSouls::LinearCopyDispatch dispatch {
		    .user_data          = std::span<const uint32_t> {cs_regs.cs_user_sgpr.value, regs.user_sgpr},
		    .threads            = {regs.num_thread_x, regs.num_thread_y, regs.num_thread_z},
		    .group_id           = {regs.tgid_x_en, regs.tgid_y_en, regs.tgid_z_en},
		    .thread_ids         = regs.tidig_comp_cnt + 1u,
		    .workgroup_register = regs.user_sgpr,
		    .tg_size            = regs.tg_size_en,
		    .thread_dimensions  = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0};
		LiveCensus::Scope census_copy(LiveCensus::DispatchPhase, sh_ctx.GetCs().cs_regs.data_addr, 11);
		if (DemonsSouls::TryLinearCopy(dispatch, m_context.GetBufferCache(), thread_group_x, thread_group_y,
		                               thread_group_z, mode)) {
			LiveCensus::SetDispatchShader(LiveCensus::LinearCopyShader); // (render4, KYTY_RENDER_STATS: the "cs" part)
			TableDispatchSeen(sh_ctx.GetCs(), "dispatch: linear copy", thread_dimensions);
			ResetBindings();
			return;
		}
	}
	if (spec) return Spec::Refuse("linear copy");

	NativePreparationScratch<ShaderComputeInputInfo> input_storage;
	auto& input_info = input_storage.Get();
	ResetNativeStageInput(input_info);
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	const auto census_shader = sh_ctx.GetCs().cs_regs.data_addr;
	std::optional<LiveCensus::Scope> phase;
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 0);
	std::string unevaluated;
	const auto  shader_generation = ShaderMapGeneration(); // (before the program reads the code)
	const auto  compute_program   = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, sh_regs, input_info, &unevaluated);
	if (!compute_program && !unevaluated.empty()) {
		// An indirect dispatch whose resource tables do not evaluate (pointers into memory the guest has not
		// mapped, while it loads; a stale descriptor no specialization can take, such as an FMASK surface in a
		// sampled slot, seen in the Nexus): it is skipped. With no thread groups in its arguments the GPU never
		// reads the tables; with some it cannot run as the guest built it, and ending the game helps no one.
		// A direct dispatch the same way: one dispatch missing beats the game ending.
		std::array<uint32_t, 3> groups {thread_group_x, thread_group_y, thread_group_z};
		if (indirect_args != 0) {
			m_context.GetBufferCache().ReadMemory(indirect_args, sizeof(groups));
			std::memcpy(groups.data(), reinterpret_cast<const void*>(indirect_args), sizeof(groups));
		}
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Dispatch: %s dispatch of %ux%ux%u thread groups skipped, its tables did not evaluate: %s",
			            indirect_args != 0 ? "an indirect" : "a", groups[0], groups[1], groups[2], unevaluated.c_str());
		ResetBindings();
		return;
	}
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 1);
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const uint32_t frame_num = static_cast<uint32_t>(m_context.FrameNumber());
	const bool     large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const auto& program   = *input_info.stage.program;
	const auto& resources = input_info.stage.resources;
	LiveCensus::SetDispatchShader(program.shader_hash); // (render4, KYTY_RENDER_STATS: the "cs" part)
	if (LiveCounters::g_dispatch_keys_on.load(std::memory_order_relaxed)) {
		// Local diagnostic: is a dispatch (shader, user data, groups) repeated from last frame, and
		// with the same materialized resources? (What a per-object native dispatch record could reuse.)
		static std::unordered_map<uint64_t, uint64_t> previous, current;
		static uint32_t                               keys_frame = 0;
		if (frame_num != keys_frame) {
			previous.swap(current);
			current.clear();
			keys_frame = frame_num;
		}
		const uint32_t groups[3] {thread_group_x, thread_group_y, thread_group_z};
		uint64_t key = XXH3_64bits_withSeed(resources.user_data.data(), resources.user_data.size() * 4u, program.shader_hash);
		key          = XXH3_64bits_withSeed(groups, sizeof(groups), key ^ indirect_args);
		uint64_t value = XXH3_64bits(resources.flattened_srt.data(), resources.flattened_srt.size() * 4u);
		for (const auto* list: {&resources.buffers, &resources.images, &resources.samplers})
			value = XXH3_64bits_withSeed(list->data(), list->size() * sizeof(ShaderRecompiler::IR::DescriptorValue), value);
		current[key] = value;
		const auto found = previous.find(key);
		LiveCounters::Add(found == previous.end() ? LiveCounters::DispatchKeyNew
		                  : found->second == value ? LiveCounters::DispatchKeySame
		                                           : LiveCounters::DispatchKeyChanged);
	}
	if (FrameCapture::Active()) {
		FrameCapture::g_call.cs = program.shader_hash;
		for (size_t i = 0; i < 3; ++i) FrameCapture::g_call.local[i] = input_info.threads_num[i];
	}
	if (indirect_args == 0 && DemonsSouls::TryLinearCopy(input_info, m_context.GetBufferCache(),
	                                                     thread_group_x, thread_group_y,
	                                                     thread_group_z, mode)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "linear_copy";
		m_linear_copy_shader     = sh_ctx.GetCs().cs_regs.data_addr;
		m_linear_copy_generation = shader_generation;
		TableDispatchSeen(sh_ctx.GetCs(), "dispatch: linear copy", use_thread_dimensions);
		ResetBindings();
		return;
	}
	if (uint64_t cleared = 0; indirect_args == 0 && TryConsumeComputeMetaClear(input_info, buffer, thread_group_x,
	                                                                          thread_group_y, thread_group_z, mode, cleared)) {
		const auto write                            = CatalogWrite();
		m_meta_clears[MetaClearKey(sh_ctx.GetCs())] = cleared;
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "meta_clear";
		TableDispatchSeen(sh_ctx.GetCs(), "dispatch: meta clear", use_thread_dimensions);
		ResetBindings();
		return;
	}
	if (indirect_args == 0 && TryConsumeComputeImageClear(input_info, buffer, thread_group_x,
	                                                      thread_group_y, thread_group_z, mode)) {
		if (FrameCapture::Active()) FrameCapture::g_call.consumed = "image_clear";
		TableDispatchSeen(sh_ctx.GetCs(), "dispatch: image clear", use_thread_dimensions);
		ResetBindings();
		return;
	}
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	// The limit is checked before counting: past it, no dispatch pays a locked increment.
	if ((large_workgroup || has_sampler) && dispatch_log_count.load(std::memory_order_relaxed) < 512 &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
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

	if (use_thread_dimensions && indirect_args != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("GraphicsRenderDispatchDirect: indirect dispatch with thread dimensions at"
			     " 0x%016" PRIx64 " uses host-read counts\n",
			     indirect_args);
		}
		indirect_args = 0;
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

	if (indirect_args == 0 && (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0)) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	phase.emplace(LiveCensus::DispatchPhase, census_shader, 2);
	TableDispatchSeen(sh_ctx.GetCs(), nullptr, use_thread_dimensions);
	// (render2) KYTY_ASYNC_COMPUTE_PIPELINES (AsyncComputeDispatches above): the pipeline before anything of the
	// dispatch is recorded or resolved, so that a skipped dispatch leaves nothing behind.
	PipelineCache::Pipeline* async_pipeline = nullptr;
	if (auto& async = AsyncComputeState(); async.on && Role() == 0 && !FrameCapture::Active()) {
		using How        = PipelineCache::AsyncCompute;
		auto& cache      = m_context.GetPipelineCache();
		bool  compiling  = false;
		const auto time_now = [] { return std::chrono::steady_clock::now(); };
		const auto us    = [](std::chrono::steady_clock::duration time) {
			return std::chrono::duration_cast<std::chrono::microseconds>(time).count();
		};
		if (const auto frame = m_context.FrameNumber(); frame != async.frame) {
			async.frame     = frame;
			async.sync_us   = 0;
			async.waited_us = 0;
			AsyncComputeReport(async, cache);
		}
		async_pipeline = cache.TryCreateComputePipelineAsync(input_info, compute_program, How::Find, &compiling);
		if (async_pipeline == nullptr) {
			async.active     = true;
			const auto begin = time_now();
			// (The TAA pass: one dispatch a frame that frame generation, the upscalers and the scene's notes hang on.)
			const bool keep = FrameGen::IsTaaShader(program.shader_hash);
			if (!compiling && (keep || async.sync_us < async.sync_budget_us)) {
				// Within the frame's budget: made here, as before.
				async_pipeline   = &cache.CreateComputePipeline(input_info, compute_program);
				const auto spent = us(time_now() - begin);
				async.sync_us += spent;
				async.total.made_here++;
				async.total.made_here_us += static_cast<uint64_t>(spent);
			} else {
				if (!compiling) async.total.queued++;
				async_pipeline = cache.TryCreateComputePipelineAsync(input_info, compute_program, How::Queue);
				if (async_pipeline == nullptr && async.waited_us < async.wait_budget_us) {
					const auto waiting  = cache.PipelineWaiting();
					const auto deadline = begin + std::chrono::microseconds(async.wait_budget_us - async.waited_us);
					while (waiting != nullptr && !waiting->load(std::memory_order_acquire) && time_now() < deadline)
						std::this_thread::yield();
					const auto spent = us(time_now() - begin);
					async.waited_us += spent;
					async.total.waited_us += static_cast<uint64_t>(spent);
					async_pipeline = cache.TryCreateComputePipelineAsync(input_info, compute_program, How::Find);
				}
				if (async_pipeline == nullptr) {
					const auto first = async.skipping.try_emplace(compute_program.id, begin).first->second;
					if (keep || begin - first >= std::chrono::milliseconds(async.hold_ms)) {
						const auto held_begin = time_now();
						async_pipeline        = cache.TryCreateComputePipelineAsync(input_info, compute_program, How::Block);
						async.total.held++;
						async.total.held_us += static_cast<uint64_t>(us(time_now() - held_begin));
					}
				}
				if (async_pipeline == nullptr) {
					const auto skipped = ++async.total.skipped;
					if ((skipped & (skipped - 1)) == 0) {
						std::printf("Async compute pipelines: %llu dispatches skipped while their pipelines compile\n",
						            static_cast<unsigned long long>(skipped));
					}
					ResetBindings();
					return;
				}
			}
			async.skipping.erase(compute_program.id);
		}
	}
	buffer.EndRendering();
	auto& pipeline = async_pipeline != nullptr
	                     ? *async_pipeline
	                     : m_context.GetPipelineCache().CreateComputePipeline(input_info, compute_program);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 3);
	NativePreparationScratch<PreparedBindings> binding_storage(
	    kyty_local_binding_scratch_mode.load(std::memory_order_relaxed) != 0);
	auto& bindings = binding_storage.Get();
	PrepareBindingsInto(input_info.stage, bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 4);
	FindBuffers(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 5);
	PrepareBdaBindings(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 6);
	// Materialize the persistent argument owner before final descriptor handles:
	// a tiny CPU upload must not leave an indirect command using a stream slice
	// that can be retired when preparation rotates the command buffer.
	if (indirect_args != 0) {
		m_context.GetBufferCache().EnsureBufferContents(indirect_args, 3u * sizeof(uint32_t));
	}
	auto&    buffer_cache = m_context.GetBufferCache();
	uint64_t unregistered = buffer_cache.UnregisterEpoch();
	// (KYTY_PROG_TRACE: ProgTrace, called here without a condition.)
	const ProgTrace prog_trace(program.shader_hash, indirect_args, bindings, buffer_cache);
	RebindBuffers(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 7);
	RebindImages(bindings);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 8);
	prog_trace.Print(bindings, m_context, m_context.GetTextureCache().m_slot_images, thread_group_x, thread_group_y,
	                 thread_group_z, indirect_args);
	// (A buffer resolved above joined into a new one since - by a later buffer's lookup or an image's refresh - no longer
	// owns its bytes, and the dispatch's writes there would be lost: as PrepareGraphicsBindingsInto.)
	for (uint32_t attempt = 0; attempt < 4 && buffer_cache.UnregisterEpoch() != unregistered; ++attempt) {
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 16)
			std::printf("Bindings: a buffer was unregistered while a dispatch's bindings were made (CS 0x%016llx), resolved "
			            "again\n",
			            static_cast<unsigned long long>(program.shader_hash));
		unregistered = buffer_cache.UnregisterEpoch();
		RebindBuffers(bindings);
	}
	if (FrameGen::IsTaaShader(program.shader_hash)) FrameGen::NoteScene(); // (the game's 3D view)
	FrameGen::NoteShader(program.shader_hash, 'c');
	if (FrameGen::RtParticlesDebug()) {
		// (rtparticles census: the dispatches writing a storage image, matched to the scene color there.)
		for (size_t i = 0; i < bindings.images.size() && i < program.info.images.size(); ++i) {
			const auto& info = bindings.images[i].desc.info;
			if (program.info.images[i].written && !info.data.Empty())
				FrameGen::NoteSceneDispatch(program.shader_hash, info.data.address, info.extent.width, info.extent.height);
		}
	}
	if (FrameGen::SsrWanted() && !bindings.images.empty()) {
		// (KYTY_SSR, logging only: a dispatch reading the G-buffer targets, the deferred lighting.)
		std::array<uint64_t, 32>   addresses {};
		std::array<vk::Format, 32> formats {};
		const size_t               n = std::min<size_t>(bindings.images.size(), addresses.size());
		for (size_t i = 0; i < n; ++i) {
			addresses[i] = bindings.images[i].desc.info.data.address;
			formats[i]   = bindings.images[i].desc.view_info.format;
		}
		FrameGen::OnSsrDispatch(program.shader_hash, std::span<const uint64_t>(addresses.data(), n),
		                        std::span<const vk::Format>(formats.data(), n));
	}
	if (FrameGen::ContactShadowsWanted()) {
		// (KYTY_CONTACT_SHADOWS: a dispatch sampling the sun's shadow-map array carries the cascade matrices.)
		FrameGen::ShadowConsumer consumer {};
		for (const auto& image: bindings.images) {
			const auto& info = image.desc.info;
			if (info.resources.layers >= 4 && info.extent.width == info.extent.height && info.extent.width >= 512 &&
			    !info.IsBlock() &&
			    (info.IsDepth() || info.pixel_format == vk::Format::eR32Sfloat || info.pixel_format == vk::Format::eR16Unorm)) {
				consumer.shadow_address = info.data.address;
				consumer.shadow_size    = info.extent.width;
				consumer.shadow_layers  = info.resources.layers;
				break;
			}
		}
		if (consumer.shadow_address != 0) {
			for (const auto& source: bindings.buffer_sources) {
				if (consumer.buffers == consumer.buffer_address.size()) break;
				if (source.address == 0 || source.size == 0 || source.size > 65536) continue;
				consumer.buffer_address[consumer.buffers] = source.address;
				consumer.buffer_size[consumer.buffers]    = source.size;
				++consumer.buffers;
			}
			FrameGen::OnShadowConsumer(program.shader_hash, consumer, input_info.stage.resources.flattened_srt);
		}
	}
	if (FrameGen::Capturing() && FrameGen::IsTaaShader(program.shader_hash) &&
	    bindings.images.size() > 3) {
		// TAA inputs (docs/RE-DEMONS-SOULS.md): slot 1 depth, slot 3 motion vectors.
		FrameGen::OnDispatch(program.shader_hash, bindings.images[1].image_id,
		                     bindings.images[1].desc.info.data.address, bindings.images[3].image_id,
		                     bindings.images[3].desc.info.data.address);
	}
	// KYTY_TAA_DIAG=1: the TAA dispatch's images (slot, guest address, format, size, storage), the first frames and
	// every 600th (the plan to put DLSS in its place: which image its output is, and what reads it after).
	// (onda3) The TAA's weights (CB from user data s0:s1 -> qword [8]; dwords 7..11 = s20..s23 and the deflicker term,
	// dword 24 = history valid, dword 25 = the CPU's ramp at offset 100), printed whenever they change: to see the static
	// weight against the motion one in the same run as the "onda".
	if (FrameGen::IsTaaShader(program.shader_hash) && std::getenv("KYTY_ONDA_JUEGO") != nullptr) {
		const auto& cs  = sh_ctx.GetCs();
		const uint64_t ud = static_cast<uint64_t>(cs.cs_user_sgpr.value[0]) | (static_cast<uint64_t>(cs.cs_user_sgpr.value[1]) << 32u);
		if (ud != 0) {
			const uint64_t cb = reinterpret_cast<const uint64_t*>(ud)[1];
			if (cb != 0) {
				const auto* w = reinterpret_cast<const float*>(cb);
				static float last[7] {};
				const float  now[7] {w[7], w[8], w[9], w[10], w[11], w[24], w[25]};
				if (std::memcmp(now, last, sizeof(now)) != 0) {
					std::memcpy(last, now, sizeof(now));
					uint32_t valid = 0;
					std::memcpy(&valid, &now[5], 4);
					std::printf("[onda3] TAA %016llx pesos: estatico(s20) %.4f movimiento(s21) %.4f factor(s22) %.4f "
					            "deflicker(s23) %.4f off44 %.4f historia %u rampa(off100) %.4f\n",
					            static_cast<unsigned long long>(program.shader_hash), now[0], now[1], now[2], now[3],
					            now[4], valid, now[6]);
					std::fflush(stdout);
				}
			}
		}
	}
	if (FrameGen::IsTaaShader(program.shader_hash) && FrameGen::TaaDiagFrame()) {
		auto& cache = m_context.GetTextureCache();
		std::printf("[taa] dispatch %ux%ux%u groups, %zu images:", thread_group_x, thread_group_y, thread_group_z,
		            bindings.images.size());
		for (size_t i = 0; i < bindings.images.size(); ++i) {
			const auto& b = bindings.images[i];
			if (!b.image_id) {
				std::printf(" [%zu] none", i);
				continue;
			}
			const auto& image = cache.GetImage(b.image_id);
			std::printf(" [%zu] 0x%" PRIx64 " fmt %d %ux%u%s", i, b.desc.info.data.address,
			            static_cast<int>(image.backing.format), image.backing.extent.width, image.backing.extent.height,
			            b.desc.type == TextureCache::BindingType::Storage ? " storage" : "");
		}
		std::printf("\n");
		std::fflush(stdout);
	}
	if (FrameCapture::Active()) {
		auto& call     = FrameCapture::g_call;
		call.prepared  = true;
		call.indirect  = indirect_args;
		call.groups[0] = thread_group_x;
		call.groups[1] = thread_group_y;
		call.groups[2] = thread_group_z;
		for (uint32_t i = 0; i < bindings.images.size(); ++i) {
			auto image = FrameCapture::FromInfo(bindings.images[i].desc.info, FrameCapture::Texture,
			                                    FrameCapture::StageCs, i);
			if (i < program.info.images.size()) {
				image.storage = program.info.images[i].resource_class ==
				                ShaderRecompiler::IR::ImageResourceClass::Storage;
				image.written = program.info.images[i].written ? 1 : 0;
			}
			call.images.push_back(image);
		}
		for (uint32_t i = 0; i < bindings.buffer_sources.size(); ++i) {
			FrameCapture::Buffer source;
			source.address = bindings.buffer_sources[i].address;
			source.size    = bindings.buffer_sources[i].size;
			source.written = i < program.info.buffers.size() && program.info.buffers[i].written;
			call.buffers.push_back(source);
		}
		if (FrameCapture::WantsData(program.shader_hash)) {
			FrameCapture::AddWords("cs.user", 0, resources.user_data);
			FrameCapture::AddWords("cs.srt", 0, resources.flattened_srt);
			for (uint32_t i = 0; i < bindings.buffer_sources.size(); ++i) {
				FrameCapture::AddGuest("cs.buf" + std::to_string(i), bindings.buffer_sources[i].address,
				                       bindings.buffer_sources[i].size);
			}
		}
	}
	if (XprCapture::Enabled() && XprCapture::IsCullProgram(program.shader_hash)) {
		for (size_t i = 0; i < program.info.buffers.size(); ++i) {
			if (program.info.buffers[i].written) {
				XprCapture::LearnCullOutput(bindings.buffer_sources[i].address,
				                            bindings.buffer_sources[i].size);
			}
		}
	}

	vk::Buffer indirect_buffer = nullptr;
	uint64_t   indirect_offset = 0;
	if (indirect_args != 0) {
		auto& cache     = m_context.GetBufferCache();
		auto& owner     = cache.GetBuffer(cache.FindBuffer(indirect_args, 3u * sizeof(uint32_t)));
		indirect_buffer = owner.Handle();
		indirect_offset = owner.Offset(indirect_args);
	}

	const bool        chain            = DemonsSouls::ComputeChains(program.shader_hash);
	auto              vk_buffer        = chain ? buffer.ChainHandle() : buffer.Handle();
	// (onda4, KYTY_ONDA_DUMP: during an F12 session, a listed shader's bound images copied before it runs)
	uint64_t   onda_written = 0;
	const bool onda = OndaDump::Wanted() && Role() == 0 && Spec::Current() == nullptr &&
	                  OndaDump::BeginDispatch(m_context.GetCommandScheduler(), m_context.FrameNumber());
	if (onda && !bindings.images.empty()) {
		for (size_t i = 0; i < program.info.images.size() && i < 64; ++i)
			if (program.info.images[i].written) onda_written |= uint64_t {1} << i;
		OndaDump::BeforeDispatch(m_context.GetGraphics(), m_context.GetCommandScheduler(), m_context.GetTextureCache(), vk_buffer,
		                         m_context.FrameNumber(), program.shader_hash, std::span<const TextureBinding> {bindings.images},
		                         onda_written);
	}
	// (KYTY_SSR: before a G-buffer consumer, the packed layers copied; before the commit, the bindings intact.)
	if (FrameGen::SsrWanted() && !bindings.images.empty())
		FrameGen::SsrCaptureGbuffer(m_context.GetGraphics(), m_context.GetTextureCache(), vk_buffer, program.shader_hash,
		                            std::span<const TextureBinding> {bindings.images});
	PreparedBindings* descriptor_stage = &bindings;
	const Envio1::Scope envio1_record(Envio1::Record); // (envio1: to the end of the dispatch)
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u}, chain);
	phase.emplace(LiveCensus::DispatchPhase, census_shader, 9);
	const bool continues_chain = chain && buffer.ComputeChainPending();
	bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	if (has_storage_writes && !continues_chain) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	// (p7b, C1 precise: local/p7-compat.h) A chained dispatch's hazards against its chain's earlier dispatches, from
	// the guest ranges of its V#s (what CommitBindings bound); one barrier only where there is one. Counted in any mode
	// but KYTY_P7=0, recorded with KYTY_P7_C1=1.
	if (chain && P7::Tracking()) {
		std::array<P7::ChainAccess, 16> accesses {};
		size_t                          count = 0;
		for (size_t i = 0; i < bindings.buffer_sources.size() && i < program.info.buffers.size(); ++i) {
			const auto& source   = bindings.buffer_sources[i];
			const auto& resource = program.info.buffers[i];
			if (source.size == 0 || source.address == 0) continue;
			if (count == accesses.size()) {
				// (More buffers than room: everything this one touches unknown - counted as reads of anything.)
				count = SIZE_MAX;
				break;
			}
			accesses[count++] = {.begin  = source.address,
			                     .end    = source.address + source.size,
			                     .read   = resource.read,
			                     .write  = resource.written,
			                     .atomic = resource.atomic};
		}
		const bool unknown = count == SIZE_MAX || program.info.uses_dma;
		const auto hazard  = P7::ChainStep(&buffer, continues_chain, indirect_args, accesses.data(),
		                                   count == SIZE_MAX ? accesses.size() : count, unknown);
		if (hazard != 0 && P7::ChainPrecise()) {
			VulkanMemoryBarrier hazard_barrier {}; // (the alias: <windows.h> defines MemoryBarrier)
			hazard_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
			hazard_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
			                               vk::AccessFlagBits::eIndirectCommandRead;
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
			                          vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 1, &hazard_barrier, 0, nullptr, 0, nullptr);
			P7::Add(P7::ChainBarriers);
		}
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, BindPipelineHandle(pipeline));
	if (indirect_args != 0) {
		vk::BufferMemoryBarrier args_barrier {};
		args_barrier.sType         = vk::StructureType::eBufferMemoryBarrier;
		args_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite |
		                             vk::AccessFlagBits::eTransferWrite |
		                             vk::AccessFlagBits::eMemoryWrite;
		args_barrier.dstAccessMask       = vk::AccessFlagBits::eIndirectCommandRead;
		args_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		args_barrier.buffer              = indirect_buffer;
		args_barrier.offset              = indirect_offset;
		args_barrier.size                = 3u * sizeof(uint32_t);
		// (KYTY_INDIRECT_ARGS_BARRIER=1: also within a chain of dispatches, as TableDispatch's.)
		static const bool args_barrier_always = [] {
			const char* value = std::getenv("KYTY_INDIRECT_ARGS_BARRIER");
			return value != nullptr && value[0] == '1';
		}();
		// (p7, C1: also inside a chain, local/p7-compat.h; live.)
		const bool args_barrier_p7 = continues_chain && !args_barrier_always && P7::ArgsBarrierInChain();
		if (args_barrier_p7) P7::Add(P7::ArgsBarriers);
		if (!continues_chain || args_barrier_always || args_barrier_p7) {
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                          vk::PipelineStageFlagBits::eDrawIndirect,
			                          vk::DependencyFlags {}, 0, nullptr, 1, &args_barrier, 0,
			                          nullptr);
		}
		const auto timing = DispatchTiming::Begin(m_context.GetGraphics(), vk_buffer, program.shader_hash);
		vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
		DispatchTiming::End(m_context.GetGraphics(), m_context.GetCommandScheduler(), vk_buffer, timing,
		                    program.shader_hash, 0, 0, 0);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, 0, 0, 0, indirect_args);
	} else {
		const auto timing = DispatchTiming::Begin(m_context.GetGraphics(), vk_buffer, program.shader_hash, &bindings,
		                                          &m_context.GetBufferCache(), input_info.stage.resources.flattened_srt);
		// KYTY_TAA_REPLACE=dlss: DLSS in place of the game's TAA, on the TAA's images.
		const bool taa_replaced = FrameGen::IsTaaShader(program.shader_hash) && FrameGen::TaaReplaceWanted() &&
		                          FrameGen::ReplaceTaa(buffer, vk_buffer, std::span<const TextureBinding> {bindings.images});
		if (!taa_replaced) vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
		if (!taa_replaced && FrameGen::IsTaaShader(program.shader_hash) && FrameGen::TaaReplaceWanted())
			(void)FrameGen::AfterTaa(buffer, vk_buffer, std::span<const TextureBinding> {bindings.images});
		DispatchTiming::End(m_context.GetGraphics(), m_context.GetCommandScheduler(), vk_buffer, timing,
		                    program.shader_hash, thread_group_x, thread_group_y, thread_group_z);
		DeviceFault::NoteDispatch(m_context.GetCommandScheduler().CurrentTick(), program.shader_hash, thread_group_x,
		                          thread_group_y, thread_group_z, 0);
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	if (chain) {
		buffer.ContinueComputeChain();
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	// (onda4) what it wrote, for a listed shader or a listed address
	if (onda && !bindings.images.empty())
		OndaDump::AfterDispatch(m_context.GetGraphics(), m_context.GetCommandScheduler(), m_context.GetTextureCache(), vk_buffer,
		                        m_context.FrameNumber(), program.shader_hash, std::span<const TextureBinding> {bindings.images},
		                        onda_written);
	// (KYTY_GBUF_VIEW / KYTY_GBUF_TAP, refl7 diagnostic: a G-buffer reader's outputs logged, the tapped image copied;
	// refl8: KYTY_RT_REFLECTIONS' guide, the game's own SSR result copied from the deferred lighting's inputs)
	if ((FrameGen::GbufViewWanted() || FrameGen::RtReflGuideWanted()) && !bindings.images.empty()) {
		uint64_t written = 0;
		for (size_t i = 0; i < program.info.images.size() && i < 64; ++i)
			if (program.info.images[i].written) written |= uint64_t {1} << i;
		FrameGen::GbufTapAfterDispatch(m_context.GetGraphics(), m_context.GetTextureCache(), vk_buffer, program.shader_hash,
		                               std::span<const TextureBinding> {bindings.images}, written);
		FrameGen::RtReflGuideAfterDispatch(m_context.GetGraphics(), m_context.GetTextureCache(), vk_buffer, program.shader_hash,
		                                   std::span<const TextureBinding> {bindings.images}, written);
	}
	// (pt1, BB_PATHTRACING: the images of the dispatches inside the scopes it reads: ResolveSunShadows, RenderLights,
	// ResolveLighting, RenderSSR)
	if (PathTrace::Requested() && Role() == 0 && !bindings.images.empty() && PathTrace::WantDispatchHook()) {
		uint64_t written = 0;
		for (size_t i = 0; i < program.info.images.size() && i < 64; ++i)
			if (program.info.images[i].written) written |= uint64_t {1} << i;
		PathTrace::AfterDispatch(m_context.GetTextureCache(), std::span<const TextureBinding> {bindings.images}, written, program.shader_hash);
	}
	// (pt1c, BB_PATHTRACING: the colour the game's TAA reads = the lit HDR image)
	if (PathTrace::Requested() && Role() == 0 && !bindings.images.empty() && FrameGen::IsTaaShader(program.shader_hash)) {
		uint64_t written = 0;
		for (size_t i = 0; i < program.info.images.size() && i < 64; ++i)
			if (program.info.images[i].written) written |= uint64_t {1} << i;
		PathTrace::NoteTaaInput(m_context.GetTextureCache(), std::span<const TextureBinding> {bindings.images}, written, program.shader_hash);
	}
	// (KYTY_RT_SUN: the first dispatch of a flip sampling the sun cascades' array copies their layers)
	if (FrameGen::RtSunWanted() && !bindings.images.empty())
		FrameGen::RtSunAfterDispatch(m_context.GetGraphics(), m_context.GetTextureCache(), vk_buffer, program.shader_hash,
		                             std::span<const TextureBinding> {bindings.images});
	// (KYTY_HWRT: after the game's TAA, the frame's scene acceleration structures; its debug view over the TAA's output)
	if (HwScene::Active() && !chain && FrameGen::IsTaaShader(program.shader_hash))
		HwScene::AfterTaa(m_context.GetGraphics(), m_context.GetTextureCache(), vk_buffer,
		                  std::span<const TextureBinding> {bindings.images});
	ResetBindings();
	m_context.GetCommandScheduler().CompleteDispatch();
	phase.reset();
}

} // namespace Libs::Graphics

// (F5 pase 2, KYTY_NATIVO_PASE_SOMBRAS=3 / =1) Sombras en cascada nativas: RenderExecutor::DsNatMark / DsNatBeforeDraw.
#include "graphics/host_gpu/renderer/dsSombrasNativas.inc"
