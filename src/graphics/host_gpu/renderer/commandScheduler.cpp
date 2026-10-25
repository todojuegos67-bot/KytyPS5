#include <source_location>
#include <cstring>
#include <string>
#include <mutex>
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/threadSampler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit(uint64_t tick) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search = [this, &gpu_tick, tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = tick;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = tick;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

// KYTY_GPU_TIME=1 (a live switch, off by default): every command buffer gets a timestamp at its
// start and end. When its tick completes, the GPU time it covered (overlaps with the previous
// command buffer removed) is added up, and every 5 s the log reports how busy the GPU was.
struct CommandScheduler::GpuTimer {
	static constexpr uint32_t Slots = 4096;

	explicit GpuTimer(GraphicContext& graphics): graphics(graphics) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = Slots * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &pool) != vk::Result::eSuccess) {
			pool = nullptr;
		}
		period_ns = graphics.GetPhysicalDeviceProperties().limits.timestampPeriod;
		info.queryCount = ShaderBlockSize * ShaderBlocks * 2;
		if (graphics.device.createQueryPool(&info, nullptr, &shader_pool) !=
		    vk::Result::eSuccess) {
			shader_pool = nullptr;
		}
	}
	~GpuTimer() {
		if (pool) {
			graphics.device.destroyQueryPool(pool, nullptr);
		}
		if (shader_pool) {
			graphics.device.destroyQueryPool(shader_pool, nullptr);
		}
	}
	KYTY_CLASS_NO_COPY(GpuTimer);

	static bool Enabled() {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_GPU_TIME", 0);
		return enabled.load(std::memory_order_relaxed) != 0;
	}

	// GPU thread, a new command buffer.
	void Begin(vk::CommandBuffer command) {
		ShaderBegin(command);
		open_slot = UINT32_MAX;
		if (!pool || !Enabled() || pending.size() >= Slots) {
			return;
		}
		open_slot = next_slot++ % Slots;
		command.resetQueryPool(pool, open_slot * 2, 2);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, pool, open_slot * 2);
	}

	// GPU thread, before the command buffer ends.
	void End(vk::CommandBuffer command) {
		ShaderEnd(command);
		if (open_slot != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, pool,
			                        open_slot * 2 + 1);
		}
	}

	void Submitted(uint64_t tick) {
		if (shader_block != UINT32_MAX) {
			shader_pending.push_back({tick, shader_block, shader_used, std::move(shader_tags)});
			shader_tags.clear();
			shader_block = UINT32_MAX;
		}
		if (open_slot != UINT32_MAX) {
			pending.push_back({tick, open_slot});
			open_slot = UINT32_MAX;
		}
	}

	// KYTY_SHADER_GPU_TIME: each command buffer takes a block of ShaderBlockSize query pairs,
	// reset at its start (outside any render pass); draws and dispatches past the block, or
	// while every block is in flight, are not timed.
	// KYTY_OP_GPU_TIME (live, default 0) also times the copies, fills, clears, resolves,
	// barriers and dispatches recorded through CommandRecorder outside a shader tag (detiling,
	// uploads, readbacks...), by call site. Pair 0 of a block spans the whole command buffer,
	// so every 5 s the GPU time is split exclusively: each tag gets only the time past the end
	// of the tags recorded before it, and what no tag covers is reported as untagged (binds,
	// render pass loads/stores, raw command-buffer work, gaps).
	static bool OpTimeEnabled() {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_OP_GPU_TIME", 0);
		return enabled.load(std::memory_order_relaxed) != 0;
	}

	void ShaderBegin(vk::CommandBuffer command) {
		shader_block = UINT32_MAX;
		shader_used  = 0;
		shader_open  = UINT32_MAX;
		op_open      = UINT32_MAX;
		shader_tags.clear();
		if (!shader_pool || (!CommandRecorder::ShaderTimeEnabled() && !OpTimeEnabled())) {
			return;
		}
		for (uint32_t block = 0; block < ShaderBlocks; block++) {
			if (!shader_block_busy[block]) {
				shader_block_busy[block] = true;
				shader_block             = block;
				command.resetQueryPool(shader_pool, block * ShaderBlockSize * 2,
				                       ShaderBlockSize * 2);
				command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, shader_pool,
				                        block * ShaderBlockSize * 2);
				shader_used = 1;
				return;
			}
		}
		blockless_buffers++;
	}

	// Before the command buffer ends: the end of pair 0.
	void ShaderEnd(vk::CommandBuffer command) {
		ShaderTimeEnd(command);
		OpTimeEnd(command);
		if (shader_block != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, shader_pool,
			                        shader_block * ShaderBlockSize * 2 + 1);
		}
	}

	void ShaderTimeBegin(vk::CommandBuffer command, uint64_t hash, bool compute) {
		OpTimeEnd(command);
		shader_open = UINT32_MAX;
		if (shader_block == UINT32_MAX || shader_used >= ShaderBlockSize) {
			return;
		}
		shader_open = shader_block * ShaderBlockSize + shader_used++;
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, shader_pool,
		                        shader_open * 2);
		shader_tags.push_back({hash, shader_open, static_cast<uint8_t>(compute ? 1 : 0), 0});
	}

	void ShaderTimeEnd(vk::CommandBuffer command) {
		if (shader_open != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, shader_pool,
			                        shader_open * 2 + 1);
			shader_open = UINT32_MAX;
		}
	}

	// KYTY_OP_GPU_TIME: inside a shader tag (a dispatch's own barriers) the op is part of it.
	void OpTimeBegin(vk::CommandBuffer command, uint32_t site, uint64_t bytes) {
		if (shader_block == UINT32_MAX || shader_open != UINT32_MAX) {
			return;
		}
		OpTimeEnd(command);
		if (shader_used >= ShaderBlockSize) {
			untimed_ops++;
			return;
		}
		op_open = shader_block * ShaderBlockSize + shader_used++;
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, shader_pool, op_open * 2);
		shader_tags.push_back({site, op_open, 2, bytes});
	}

	void OpTimeEnd(vk::CommandBuffer command) {
		if (op_open != UINT32_MAX) {
			command.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, shader_pool,
			                        op_open * 2 + 1);
			op_open = UINT32_MAX;
		}
	}

	template <typename IsFree>
	void CollectShaders(IsFree&& is_free) {
		while (!shader_pending.empty() && is_free(shader_pending.front().tick)) {
			auto& done = shader_pending.front();
			// One read for the block: per query its timestamp and its availability.
			const uint32_t base  = done.block * ShaderBlockSize;
			const uint32_t pairs = std::max<uint32_t>(done.used, 1);
			stamp_scratch.assign(static_cast<size_t>(pairs) * 4, 0);
			const auto result = graphics.device.getQueryPoolResults(
			    shader_pool, base * 2, pairs * 2, stamp_scratch.size() * sizeof(uint64_t),
			    stamp_scratch.data(), 2 * sizeof(uint64_t),
			    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			const bool read_ok = result == vk::Result::eSuccess || result == vk::Result::eNotReady;
			auto read_pair = [&](uint32_t slot, std::array<uint64_t, 2>& stamps) {
				const size_t i = static_cast<size_t>(slot - base) * 4;
				if (!read_ok || slot < base || slot - base >= pairs || stamp_scratch[i + 1] == 0 ||
				    stamp_scratch[i + 3] == 0) {
					return false;
				}
				stamps = {stamp_scratch[i], stamp_scratch[i + 2]};
				return stamps[1] >= stamps[0];
			};
			std::array<uint64_t, 2> span {};
			const bool has_span = read_pair(base, span);
			uint64_t   last     = has_span ? span[0] : 0;
			double     tagged   = 0;
			for (const auto& tag: done.tags) {
				std::array<uint64_t, 2> stamps {};
				if (!read_pair(tag.slot, stamps)) {
					lost_tags++;
					continue;
				}
				const double inclusive = static_cast<double>(stamps[1] - stamps[0]) * period_ns;
				const auto   from      = std::max(stamps[0], last);
				const double exclusive =
				    stamps[1] > from ? static_cast<double>(stamps[1] - from) * period_ns : 0.0;
				last = std::max(last, stamps[1]);
				tagged += exclusive;
				kind_ns[tag.kind] += exclusive;
				auto& total = tag.kind == 2 ? op_totals[tag.key] : shader_totals[tag.key];
				total.ns += inclusive;
				total.exclusive_ns += exclusive;
				total.calls++;
				total.bytes += tag.bytes;
				total.kind = tag.kind;
			}
			if (has_span) {
				const double whole = static_cast<double>(span[1] - span[0]) * period_ns;
				span_ns += whole;
				untagged_ns += std::max(0.0, whole - tagged);
				spanned_buffers++;
			}
			shader_block_busy[done.block] = false;
			shader_pending.pop_front();
		}
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<double>(now - shader_report).count();
		if (elapsed >= 5.0) {
			if (!shader_totals.empty()) {
				std::vector<std::pair<uint64_t, ShaderTotal>> sorted(shader_totals.begin(),
				                                                     shader_totals.end());
				const size_t top = std::min<size_t>(sorted.size(), 20);
				std::partial_sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(top),
				                  sorted.end(), [](const auto& a, const auto& b) {
					                  return a.second.ns > b.second.ns;
				                  });
				::printf("Shader GPU time (5 s): top 20\n");
				for (size_t i = 0; i < top; i++) {
					::printf("  %016" PRIx64 " %s %.3f ms in %" PRIu64 " calls\n", sorted[i].first,
					         sorted[i].second.kind == 1 ? "CS" : "PS", sorted[i].second.ns / 1e6,
					         sorted[i].second.calls);
				}
			}
			if (spanned_buffers != 0) {
				// One list, shaders and ops together, by exclusive time.
				struct Row {
					uint64_t    key;
					ShaderTotal total;
				};
				std::vector<Row> rows;
				rows.reserve(shader_totals.size() + op_totals.size());
				for (const auto& [key, total]: shader_totals) rows.push_back({key, total});
				for (const auto& [key, total]: op_totals) rows.push_back({key, total});
				const size_t top = std::min<size_t>(rows.size(), 40);
				std::partial_sort(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(top),
				                  rows.end(), [](const Row& a, const Row& b) {
					                  return a.total.exclusive_ns > b.total.exclusive_ns;
				                  });
				::printf("GPU work (%.1f s): %" PRIu64 " command buffers timed (%" PRIu64
				         " without a block), span %.1f ms; exclusive: draws %.1f ms, dispatches "
				         "%.1f ms, ops %.1f ms, untagged %.1f ms; %" PRIu64 " ops untimed (block "
				         "full), %" PRIu64 " tags lost\n",
				         elapsed, spanned_buffers, blockless_buffers, span_ns / 1e6,
				         kind_ns[0] / 1e6, kind_ns[1] / 1e6, kind_ns[2] / 1e6, untagged_ns / 1e6,
				         untimed_ops, lost_tags);
				::printf("GPU work (5 s): top %zu by exclusive time (exclusive ms, inclusive ms, "
				         "calls, MiB)\n",
				         top);
				for (size_t i = 0; i < top; i++) {
					const auto& row = rows[i];
					if (row.total.kind == 2) {
						::printf("  OP %-48s %8.3f %8.3f %7" PRIu64 " %8.1f\n",
						         CommandScheduler::OpSiteName(static_cast<uint32_t>(row.key)).c_str(),
						         row.total.exclusive_ns / 1e6, row.total.ns / 1e6, row.total.calls,
						         static_cast<double>(row.total.bytes) / 1048576.0);
					} else {
						::printf("  %s %016" PRIx64 "%33s %8.3f %8.3f %7" PRIu64 "          %s\n",
						         row.total.kind == 1 ? "CS" : "PS", row.key, "",
						         row.total.exclusive_ns / 1e6, row.total.ns / 1e6, row.total.calls,
						         CommandScheduler::ShapeText(row.key).c_str());
					}
				}
			}
			std::fflush(stdout);
			shader_totals.clear();
			op_totals.clear();
			kind_ns           = {};
			span_ns           = 0;
			untagged_ns       = 0;
			spanned_buffers   = 0;
			blockless_buffers = 0;
			untimed_ops       = 0;
			lost_tags         = 0;
			shader_report     = now;
		}
	}

	template <typename IsFree>
	void Collect(IsFree&& is_free) {
		CollectShaders(is_free);
		while (!pending.empty() && is_free(pending.front().tick)) {
			const auto              slot = pending.front().slot;
			std::array<uint64_t, 2> stamps {};
			if (graphics.device.getQueryPoolResults(
			        pool, slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
			        vk::QueryResultFlagBits::e64) == vk::Result::eSuccess &&
			    stamps[1] >= stamps[0]) {
				const auto begin = std::max(stamps[0], last_end);
				if (stamps[1] > begin) {
					busy_ns += static_cast<double>(stamps[1] - begin) * period_ns;
				}
				last_end = std::max(last_end, stamps[1]);
				buffers++;
			}
			pending.pop_front();
		}
		const auto now     = std::chrono::steady_clock::now();
		const auto elapsed = std::chrono::duration<double>(now - report).count();
		if (elapsed >= 5.0) {
			if (buffers != 0) {
				const double busy_ms_per_s = busy_ns / 1e6 / elapsed;
				::printf("GPU time (%.1f s): busy %.1f ms/s (%.0f%%), %" PRIu64
				         " command buffers\n",
				         elapsed, busy_ms_per_s, busy_ms_per_s / 10.0, buffers);
				std::fflush(stdout);
				if (tracy::ProfilerAvailable()) {
					TracyPlot("GPU busy ms/s", busy_ms_per_s);
				}
			}
			busy_ns = 0;
			buffers = 0;
			report  = now;
		}
	}

	struct Pending {
		uint64_t tick = 0;
		uint32_t slot = 0;
	};

	GraphicContext&                       graphics;
	vk::QueryPool                         pool      = nullptr;
	float                                 period_ns = 1.0f;
	uint32_t                              next_slot = 0;
	uint32_t                              open_slot = UINT32_MAX;
	std::deque<Pending>                   pending;
	uint64_t                              last_end = 0;
	double                                busy_ns  = 0;
	uint64_t                              buffers  = 0;
	std::chrono::steady_clock::time_point report   = std::chrono::steady_clock::now();
	static constexpr uint32_t ShaderBlockSize = 1024;
	static constexpr uint32_t ShaderBlocks    = 96;
	struct ShaderTag {
		uint64_t key   = 0; // shader hash, or the op site (kind 2)
		uint32_t slot  = 0;
		uint8_t  kind  = 0; // 0 PS, 1 CS, 2 op
		uint64_t bytes = 0;
	};
	struct ShaderPending {
		uint64_t               tick  = 0;
		uint32_t               block = 0;
		uint32_t               used  = 0;
		std::vector<ShaderTag> tags;
	};
	struct ShaderTotal {
		double   ns           = 0;
		double   exclusive_ns = 0;
		uint64_t calls        = 0;
		uint64_t bytes        = 0;
		uint8_t  kind         = 0;
	};
	vk::QueryPool                             shader_pool  = nullptr;
	std::array<bool, ShaderBlocks>            shader_block_busy {};
	uint32_t                                  shader_block = UINT32_MAX;
	uint32_t                                  shader_used  = 0;
	uint32_t                                  shader_open  = UINT32_MAX;
	std::vector<ShaderTag>                    shader_tags;
	std::deque<ShaderPending>                 shader_pending;
	std::unordered_map<uint64_t, ShaderTotal> shader_totals;
	std::unordered_map<uint64_t, ShaderTotal> op_totals;
	std::array<double, 3>                     kind_ns {};
	double                                    span_ns           = 0;
	double                                    untagged_ns       = 0;
	uint64_t                                  spanned_buffers   = 0;
	uint64_t                                  blockless_buffers = 0;
	uint64_t                                  untimed_ops       = 0;
	uint64_t                                  lost_tags         = 0;
	uint32_t                                  op_open           = UINT32_MAX;
	std::vector<uint64_t>                     stamp_scratch;
	std::chrono::steady_clock::time_point     shader_report = std::chrono::steady_clock::now();
};

void CommandScheduler::ShaderTimeBegin(vk::CommandBuffer command, uint64_t hash, bool compute) {
	m_gpu_timer->ShaderTimeBegin(command, hash, compute);
}

void CommandScheduler::ShaderTimeEnd(vk::CommandBuffer command) {
	m_gpu_timer->ShaderTimeEnd(command);
}

void CommandScheduler::OpTimeBegin(vk::CommandBuffer command, uint32_t site, uint64_t bytes) {
	m_gpu_timer->OpTimeBegin(command, site, bytes);
}

void CommandScheduler::OpTimeEnd(vk::CommandBuffer command) {
	m_gpu_timer->OpTimeEnd(command);
}

bool CommandScheduler::OpTimeEnabled() {
	return GpuTimer::OpTimeEnabled();
}

namespace {
struct OpSites {
	std::mutex                             mutex;
	std::unordered_map<std::string, uint32_t> ids;
	std::vector<std::string>               names;
};
OpSites& Sites() {
	static OpSites sites;
	return sites;
}
} // namespace

uint32_t CommandScheduler::OpSite(const char* label, const std::source_location& where) {
	const char* file = where.file_name();
	for (const char* c = file; *c != '\0'; c++) {
		if (*c == '/' || *c == '\\') {
			file = c + 1;
		}
	}
	std::string name = std::string(label) + " " + file + ":" + std::to_string(where.line());
	auto&       sites = Sites();
	std::lock_guard lock(sites.mutex);
	const auto [it, inserted] = sites.ids.try_emplace(name, static_cast<uint32_t>(sites.names.size()));
	if (inserted) {
		sites.names.push_back(name);
	}
	return it->second;
}

namespace {
struct Shape {
	uint32_t width = 0, height = 0, count = 0, instances = 0;
	bool     pops  = false;
};
struct Shapes {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, Shape>    shapes;
};
Shapes& AllShapes() {
	static Shapes shapes;
	return shapes;
}
} // namespace

void CommandScheduler::NoteShape(uint64_t hash, uint32_t width, uint32_t height, uint32_t count,
                                 uint32_t instances, bool pops) {
	auto&           shapes = AllShapes();
	std::lock_guard lock(shapes.mutex);
	shapes.shapes[hash] = {width, height, count, instances, pops};
}

std::string CommandScheduler::ShapeText(uint64_t hash) {
	auto&           shapes = AllShapes();
	std::lock_guard lock(shapes.mutex);
	const auto      it = shapes.shapes.find(hash);
	if (it == shapes.shapes.end()) {
		return {};
	}
	const auto& s = it->second;
	char        text[96];
	if (s.width == 0 && s.height == 0) {
		std::snprintf(text, sizeof(text), "groups %u", s.count);
	} else {
		std::snprintf(text, sizeof(text), "%ux%u, %u verts x %u%s", s.width, s.height, s.count,
		              s.instances, s.pops ? ", POPS" : "");
	}
	return text;
}

std::string CommandScheduler::OpSiteName(uint32_t site) {
	auto&           sites = Sites();
	std::lock_guard lock(sites.mutex);
	return site < sites.names.size() ? sites.names[site] : std::string("?");
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_gpu_timer(std::make_unique<GpuTimer>(graphics)), m_master(graphics), m_context(context),
      m_graphics(graphics), m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	StopRecordingThread();
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

// Submissions per 5 s by the call site that ended the command buffer (where the work is split).
static void NoteSubmitSite(const char* kind, const std::source_location& where) {
	static std::mutex                                   mutex;
	static std::unordered_map<std::string, uint64_t>    counts;
	static std::chrono::steady_clock::time_point        report = std::chrono::steady_clock::now();
	std::lock_guard lock(mutex);
	const char* file = where.file_name();
	if (const char* slash = std::strrchr(file, '/')) file = slash + 1;
	if (const char* slash = std::strrchr(file, '\\')) file = slash + 1;
	counts[std::string(kind) + " " + file + ":" + std::to_string(where.line())]++;
	const auto now = std::chrono::steady_clock::now();
	if (now - report >= std::chrono::seconds(5)) {
		std::vector<std::pair<std::string, uint64_t>> sorted(counts.begin(), counts.end());
		std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		::printf("Submit sites (5 s):\n");
		for (size_t i = 0; i < sorted.size() && i < 12; i++) ::printf("  %6llu %s\n", static_cast<unsigned long long>(sorted[i].second), sorted[i].first.c_str());
		std::fflush(stdout);
		counts.clear();
		report = now;
	}
}

void CommandScheduler::Flush(std::source_location where) {
	SubmitInfo submit;
	Flush(submit, where);
}

void CommandScheduler::Flush(SubmitInfo& submit, std::source_location where) {
	NoteSubmitSite("flush", where);
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait(std::source_location where) {
	NoteSubmitSite("flush+wait", where);
	KYTY_PROFILER_FUNCTION();
	const auto tick = Submit();
	m_master.Wait(tick, where);
	BeginNext();
}

void CommandScheduler::Finish(std::source_location where) {
	NoteSubmitSite("finish", where);
	KYTY_PROFILER_FUNCTION();
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1, where);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick, std::source_location where) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick, where);
		BeginNext();
	} else {
		m_master.Wait(tick, where);
	}
}

void CommandScheduler::PopPendingOperations() {
	KYTY_PROFILER_FUNCTION();
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Poll();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	static auto& record_thread = Common::LiveSwitches::Get("KYTY_RECORD_THREAD", 1);
	const bool   threaded =
	    m_record_thread.joinable() && record_thread.load(std::memory_order_relaxed) != 0;
	if (!threaded && m_record_thread.joinable()) {
		// The recording thread may still be ending or submitting the previous command buffer, and
		// the pool must not be used by two threads.
		(void)DrainRecording();
	}
	m_threaded           = threaded;
	m_command.m_threaded = threaded;
	m_command.m_open     = true;
	if (threaded) {
		BeginThreadedCommand();
		return m_command;
	}
	m_command.m_buffer = m_command_pool.Commit(m_master.CurrentTick());
	m_command.Begin();
	m_gpu_timer->Collect([this](uint64_t tick) { return IsFree(tick); });
	m_gpu_timer->Begin(m_command.m_buffer);
	return m_command;
}

// The final barrier and the queue submission of a command buffer, for both the GPU thread and
// the recording thread.
struct CommandScheduler::SubmitDebug {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint32_t arg0 = 0, arg1 = 0, arg2 = 0, arg3 = 0;
	uint64_t arg4 = 0;
};

void CommandScheduler::QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick,
                                   const SubmitDebug& debug) {
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);
	vk::Result result;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, debug.op, debug.submit_id, debug.arg0,
		                  debug.arg1, debug.arg2, debug.arg3, debug.arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_gpu_timer->Submitted(tick);
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (AnyMappedDeviceBuffer()) {
		// Buffers in host-visible device memory are read by the host once this submission's
		// tick completes (BufferCache direct readback): make its writes visible to the host.
		m_command.EndRendering();
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		barrier.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
		barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		m_command.Recorder().pipelineBarrier2(dependency);
	}
	m_command.EndRendering();
	const SubmitDebug debug {m_command.m_debug_op,   m_command.m_debug_submit_id,
	                         m_command.m_debug_arg0, m_command.m_debug_arg1,
	                         m_command.m_debug_arg2, m_command.m_debug_arg3,
	                         m_command.m_debug_arg4};
	// The time this thread hands the command buffer over: the gates on it (KYTY_LABEL_FLUSH_US,
	// KYTY_SLICE_FLUSH_US) measure this thread's batching, not when the driver call happens.
	m_last_submit_us.store(std::chrono::duration_cast<std::chrono::microseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count(),
	                       std::memory_order_relaxed);
	if (m_threaded) {
		return SubmitThreaded(submit, debug);
	}
	m_gpu_timer->End(m_command.m_buffer);
	m_command.End();
	uint64_t tick = 0;
	{
		// The tick is taken under the queue lock in this path, as before.
		Common::LockGuard lock(m_graphics.queue_mutex);
		tick = m_master.NextTick();
	}
	QueueSubmit(m_command.m_buffer, submit, tick, debug);
	m_command.m_buffer = nullptr;
	m_command.m_open   = false;
	return tick;
}

// Research: KYTY_RECORD_THREAD (live, per command buffer; default 1, 0 records inline). Thread_Gpu spent ~15-20 %
// of its time inside the Vulkan driver recording and submitting (run 21: 9.4 -> 9.9 fps). With the switch on, the
// renderer's scheduler hands the Vulkan calls of the current command buffer to a recording
// thread: CommandRecorder (Recorder()) and Record() queue them in order, with every array they
// point to copied into the command chunk; the recording thread owns the command pool, begins,
// ends and submits the command buffers. Ticks are still assigned here, in order, when a command
// buffer is submitted, so everything keyed by ticks is unchanged; a host wait on a tick the
// recording thread has not submitted yet just waits longer (timeline semaphores allow it).
// Code that still records through the raw Handle() first waits for the recording thread to
// catch up (DrainRecording), so it stays correct, only slower.
void CommandScheduler::EnableRecordingThread() {
	EXIT_IF(m_record_thread.joinable());
	m_record_thread = std::jthread([this](std::stop_token stop) { RecordingThread(stop); });
}

void CommandScheduler::BeginThreadedCommand() {
	static std::atomic_bool announced = false;
	if (!announced.exchange(true, std::memory_order_relaxed)) {
		::printf("Record thread: first threaded command buffer\n");
		std::fflush(stdout);
	}
	m_command.m_buffer = nullptr;
	m_command.InvalidateDynamicState();
	const auto tick = m_master.CurrentTick();
	Record([this, tick](vk::CommandBuffer) {
		m_worker_buffer = m_command_pool.Commit(tick);
		vk::CommandBufferBeginInfo begin_info {};
		begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		EXIT_NOT_IMPLEMENTED(m_worker_buffer.begin(&begin_info) != vk::Result::eSuccess);
		m_gpu_timer->Collect([this](uint64_t done) { return IsFree(done); });
		m_gpu_timer->Begin(m_worker_buffer);
	});
}

uint64_t CommandScheduler::SubmitThreaded(SubmitInfo submit, const SubmitDebug& debug) {
	const auto tick = m_master.NextTick();
	Record([this, submit, tick, debug](vk::CommandBuffer command) mutable {
		m_gpu_timer->End(command);
		EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);
		QueueSubmit(command, submit, tick, debug);
		m_worker_buffer = nullptr;
	});
	DispatchChunk();
	m_command.m_buffer = nullptr;
	m_command.m_open   = false;

	// Every 5 s: threaded command buffers, chunks handed over, and drains (raw Handle() uses,
	// which make Thread_Gpu wait for the recording thread: the sites still to convert).
	m_record_stats.buffers++;
	const auto now = std::chrono::steady_clock::now();
	if (now - m_record_stats.report >= std::chrono::seconds(5)) {
		uint64_t chunks = 0;
		{
			std::lock_guard lock(m_record_mutex);
			chunks = m_chunks_dispatched;
		}
		::printf("Record thread (5 s): %" PRIu64 " command buffers, %" PRIu64 " chunks, %" PRIu64
		         " drains\n",
		         m_record_stats.buffers, chunks - m_record_stats.chunks, m_record_stats.drains);
		std::fflush(stdout);
		m_record_stats = {.chunks = chunks, .report = now};
	}
	return tick;
}

void* CommandScheduler::Allocate(size_t size, size_t alignment) {
	EXIT_IF(!m_threaded || size > Chunk::Size);
	const auto take = [this] {
		std::lock_guard lock(m_record_mutex);
		if (m_free_chunks.empty()) {
			return std::make_unique<Chunk>();
		}
		auto chunk = std::move(m_free_chunks.back());
		m_free_chunks.pop_back();
		return chunk;
	};
	if (m_chunk == nullptr) {
		m_chunk = take();
	}
	auto offset = (m_chunk->used + alignment - 1) & ~(alignment - 1);
	if (offset + size > Chunk::Size) {
		DispatchChunk();
		m_chunk = take();
		offset  = 0;
	}
	m_chunk->used = offset + size;
	return m_chunk->storage + offset;
}

void CommandScheduler::DispatchChunk() {
	if (m_chunk == nullptr || m_chunk->used == 0) {
		return;
	}
	{
		std::lock_guard lock(m_record_mutex);
		m_record_queue.push_back(std::move(m_chunk));
		m_chunks_dispatched++;
	}
	m_record_available.notify_one();
}

vk::CommandBuffer CommandScheduler::DrainRecording() {
	if (!m_record_thread.joinable()) {
		return m_command.m_buffer;
	}
	KYTY_PROFILER_FUNCTION();
	if (m_threaded) {
		m_record_stats.drains++;
	}
	DispatchChunk();
	std::unique_lock lock(m_record_mutex);
	m_record_executed.wait(lock, [this] { return m_chunks_executed == m_chunks_dispatched; });
	return m_worker_buffer;
}

void CommandScheduler::RecordingThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	ThreadSampler::RegisterCurrentThread("Thread_GpuRecord");
	// A command's stashed arrays may sit in up to two chunks before its own: executed chunks are
	// recycled only after the next two have run.
	std::deque<std::unique_ptr<Chunk>> retained;
	for (;;) {
		std::unique_ptr<Chunk> chunk;
		Common::Thread::ApplyGpuThreadPriority();
		{
			std::unique_lock lock(m_record_mutex);
			if (!m_record_available.wait(lock, stop, [this] { return !m_record_queue.empty(); })) {
				return;
			}
			chunk = std::move(m_record_queue.front());
			m_record_queue.pop_front();
		}
		for (auto* command = chunk->first; command != nullptr;) {
			auto* next = command->next;
			command->Execute(m_worker_buffer);
			command->~RecordedCommand();
			command = next;
		}
		chunk->first = nullptr;
		chunk->last  = nullptr;
		chunk->used  = 0;
		retained.push_back(std::move(chunk));
		std::unique_ptr<Chunk> recycled;
		if (retained.size() > 2) {
			recycled = std::move(retained.front());
			retained.pop_front();
		}
		{
			std::lock_guard lock(m_record_mutex);
			if (recycled != nullptr) {
				m_free_chunks.push_back(std::move(recycled));
			}
			m_chunks_executed++;
		}
		m_record_executed.notify_all();
	}
}

void CommandScheduler::StopRecordingThread() {
	if (!m_record_thread.joinable()) {
		return;
	}
	(void)DrainRecording();
	m_record_thread.request_stop();
	m_record_available.notify_all();
	m_record_thread.join();
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
