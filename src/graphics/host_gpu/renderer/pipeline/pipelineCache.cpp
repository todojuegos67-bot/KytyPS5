#include "graphics/phaseStats.h"
#include <optional>
#include "graphics/drawDiff.h"
#include "graphics/frameCosts.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/liveSwitches.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/pipelinePrewarm.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/SrtNative.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <tuple>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

bool ShaderFailureNonFatal() {
	return true;
}

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", KYTY_GIT_REVISION,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

// KYTY_PIPELINE_CACHE_ANY_REVISION=1 keeps a cache written by another emulator revision, so
// A/B builds of one title don't start cold. The device, driver and UUID must still match, and
// the driver validates every entry itself; entries for pipelines a build no longer creates are
// simply never looked up.
bool DriverCacheSignatureMatches(std::string_view cached, std::string_view current) {
	if (cached == current) {
		return true;
	}
	// On by default in this build (updates and the launcher keep the warm cache);
	// KYTY_PIPELINE_CACHE_ANY_REVISION=0 turns it off.
	static const bool any_revision = [] {
		const char* value = std::getenv("KYTY_PIPELINE_CACHE_ANY_REVISION");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (!any_revision) {
		return false;
	}
	// "KytyPC1:<revision>:<vendor>:..." -> ":<vendor>:..."
	const auto without_revision = [](std::string_view signature) -> std::string_view {
		const auto first = signature.find(':');
		const auto second =
		    first == std::string_view::npos ? first : signature.find(':', first + 1);
		return second == std::string_view::npos ? std::string_view {} : signature.substr(second);
	};
	const auto cached_rest = without_revision(cached);
	return !cached_rest.empty() && cached.starts_with("KytyPC1:") &&
	       cached_rest == without_revision(current);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool UsesShaderClock(const ShaderRecompiler::IR::Program& program) {
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == ShaderRecompiler::IR::ValueOpcode::ReadClockRealtime64) {
				return true;
			}
		}
	}
	return false;
}

// The resource walker reads guest memory one dword per scalar load, and every read pays the
// GPU-ownership checks and the address-space lock. KYTY_SHADER_READ_CHUNKS (a live switch):
// 1 reads each aligned 256-byte chunk once per program lookup and serves its dwords from that
// copy; 2 (default) looks up each 4 KiB page's backing bytes once per lookup (FindGpuCleanBacking,
// after Senaxx 820f72e5) and copies every later read of the page straight from them. A chunk or
// page with any GPU-owned byte, or one that is not all mapped, is refused and its reads take the
// per-read path, so the values are the ones read before. Nothing marks a page GPU-written during
// one synchronous lookup on the GPU thread; the verdicts live for one lookup only.
// The same object can log every read and its result (KYTY_RESOURCE_MEMO, ProgramCache).
class ShaderReadChunks {
public:
	struct LoggedRead {
		uint64_t address = 0;
		uint32_t first   = 0; // index of the first word in ReadLog::words
		uint32_t count   = 0;
		bool     ok      = false;
	};
	struct ReadLog {
		std::vector<LoggedRead> reads;
		std::vector<uint32_t>   words;
	};

	explicit ShaderReadChunks(int64_t mode): m_chunks(mode == 1), m_pages(mode == 2) {}

	// Returns false when the chunk cannot serve the read; the caller reads as before.
	bool Read(uint64_t address, std::span<uint32_t> values) {
		if (m_pages) {
			return ReadPage(address, values);
		}
		if (!m_chunks) {
			return false;
		}
		const auto base = address & ~(ChunkSize - 1);
		if (values.size_bytes() > ChunkSize || address + values.size_bytes() > base + ChunkSize) {
			return false;
		}
		auto* slot = Find(base);
		if (slot == nullptr) {
			slot        = &m_slots[m_next++ % m_slots.size()];
			slot->base  = base;
			slot->clean = Libs::LibKernel::Memory::TryReadGpuCleanBacking(base, slot->words.data(),
			                                                              ChunkSize);
		}
		if (!slot->clean) {
			return false;
		}
		std::memcpy(values.data(), slot->words.data() + (address - base) / sizeof(uint32_t),
		            values.size_bytes());
		return true;
	}

	void SetLog(ReadLog* log) { m_log = log; }

	void Log(uint64_t address, std::span<const uint32_t> values, bool ok) {
		if (m_log == nullptr) {
			return;
		}
		m_log->reads.push_back({.address = address,
		                        .first   = static_cast<uint32_t>(m_log->words.size()),
		                        .count   = static_cast<uint32_t>(values.size()),
		                        .ok      = ok});
		if (ok) {
			m_log->words.insert(m_log->words.end(), values.begin(), values.end());
		}
	}

	static int64_t Mode() {
		// Wolverine run 23: see PROGRESS.md for the 1 vs 2 A/B.
		static auto& mode = Common::LiveSwitches::Get("KYTY_SHADER_READ_CHUNKS", 2);
		return mode.load(std::memory_order_relaxed);
	}

private:
	static constexpr uint64_t ChunkSize = 256;

	struct Slot {
		uint64_t                                           base  = UINT64_MAX;
		bool                                               clean = false;
		std::array<uint32_t, ChunkSize / sizeof(uint32_t)> words;
	};

	struct Page {
		uint64_t       base    = UINT64_MAX;
		const uint8_t* backing = nullptr;
	};

	bool ReadPage(uint64_t address, std::span<uint32_t> values) {
		const auto base = Common::AlignDown(address, TRACKER_PAGE_SIZE);
		if (values.empty() ||
		    Common::AlignDown(address + values.size_bytes() - 1, TRACKER_PAGE_SIZE) != base) {
			return false;
		}
		const Page* page = nullptr;
		for (size_t i = 0; i < m_page_count; i++) {
			if (m_pages_cache[i].base == base) {
				page = &m_pages_cache[i];
				break;
			}
		}
		if (page == nullptr) {
			auto& slot = m_page_count < m_pages_cache.size()
			                 ? m_pages_cache[m_page_count++]
			                 : m_pages_cache[m_next_page++ % m_pages_cache.size()];
			slot       = {.base = base,
			              .backing =
			                  Libs::LibKernel::Memory::FindGpuCleanBacking(base, TRACKER_PAGE_SIZE)};
			page       = &slot;
		}
		if (page->backing == nullptr) {
			return false;
		}
		std::memcpy(values.data(), page->backing + (address - base), values.size_bytes());
		return true;
	}

	Slot* Find(uint64_t base) {
		for (auto& slot: m_slots) {
			if (slot.base == base) {
				return &slot;
			}
		}
		return nullptr;
	}

	bool                 m_chunks = false;
	bool                 m_pages  = false;
	ReadLog*             m_log    = nullptr;
	std::array<Slot, 8>  m_slots;
	size_t               m_next = 0;
	std::array<Page, 16> m_pages_cache;
	size_t               m_page_count = 0;
	size_t               m_next_page  = 0;
};

// Both SRT readers: only memory the guest has committed. Bytes the GPU has not written are
// current in the backing store; reading them there skips the tracked-page fault, which drains
// the GPU to refresh whatever else on the page the GPU wrote (constants that share a page with
// GPU-written arguments cost a drain per dispatch). Bytes that are mapped but GPU-owned are read
// through the guest mapping, so the fault refreshes them and the value is the one the shader
// would see. Without a reader the walker dereferenced whatever address a descriptor chain
// produced, including 0 on a path the shader never takes.
bool ReadShaderGuestMemoryImpl(ShaderReadChunks* chunks, uint64_t address,
                               std::span<uint32_t> values) {
	if (values.empty()) {
		return false;
	}
	if (chunks != nullptr && chunks->Read(address, values)) {
		return true;
	}
	if (Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(),
	                                                    values.size_bytes())) {
		return true;
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes())) {
		return false;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

bool ReadShaderGuestMemoryRaw(void* userdata, uint64_t address, std::span<uint32_t> values) {
	KYTY_PROFILER_BLOCK("ShaderGuestMemoryRead");
	auto*      chunks = static_cast<ShaderReadChunks*>(userdata);
	const bool ok     = ReadShaderGuestMemoryImpl(chunks, address, values);
	if (chunks != nullptr) {
		chunks->Log(address, values, ok);
	}
	return ok;
}

// Research: a subroutine's code for inlining, in 1 KiB steps while the guest has it committed.
std::vector<uint32_t> ReadShaderCode(uint64_t address) {
	constexpr size_t      Chunk = 256;
	constexpr size_t      Limit = 16384;
	std::vector<uint32_t> words;
	while (words.size() < Limit) {
		std::array<uint32_t, Chunk> chunk {};
		if (!ReadShaderGuestMemoryRaw(nullptr, address + words.size() * sizeof(uint32_t), chunk)) {
			break;
		}
		words.insert(words.end(), chunk.begin(), chunk.end());
	}
	return words;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	KYTY_PROFILER_BLOCK("ShaderGuestMemoryRead");
	auto*      chunks = static_cast<ShaderReadChunks*>(userdata);
	const bool ok     = ReadShaderGuestMemoryImpl(chunks, address, values);
	if (chunks != nullptr) {
		chunks->Log(address, values, ok);
	}
	return ok;
}

// KYTY_RESOURCE_MEMO (live; default 1 since wolverine-fast-1027): see ProgramCache::Materialize.
// Safe by construction: a hit re-reads every guest word the walk read and requires them equal.
bool ResourceMemoEnabled() {
	static auto& enabled = Common::LiveSwitches::Get("KYTY_RESOURCE_MEMO", 1);
	return enabled.load(std::memory_order_relaxed) != 0;
}

// --skip-shaders and KYTY_SKIP_SHADER_HASHES="hash,hash,...": skip the draws and dispatches of
// these guest shaders, the same way a shader that fails to compile is skipped. To see what one
// shader contributes, to step past one that loses the device, or to leave out work nothing can
// use (the ray-tracing BVH updates whose only consumer gives up).
// KYTY_LIVE_SKIP_BITS=k / KYTY_LIVE_SKIP_VALUE=v / KYTY_LIVE_SKIP_STAGE=s (live, research):
// skip, as --skip-shaders does, every shader whose hash has v in its low k bits, to find by
// bisection which shader draws a visual fault (stage s: 0 all, 1 pixel, 2 compute, 3 vertex).
// Each skipped hash is printed once per (k, v) as LIVE-SKIP. k = 0: off.
static bool LiveSkipRequested(uint64_t shader_hash, ShaderType stage) {
	static auto& bits_switch  = Common::LiveSwitches::Get("KYTY_LIVE_SKIP_BITS", 0);
	static auto& value_switch = Common::LiveSwitches::Get("KYTY_LIVE_SKIP_VALUE", 0);
	static auto& stage_switch = Common::LiveSwitches::Get("KYTY_LIVE_SKIP_STAGE", 0);
	const auto   bits         = bits_switch.load(std::memory_order_relaxed);
	if (bits <= 0 || bits >= 64) {
		return false;
	}
	switch (stage_switch.load(std::memory_order_relaxed)) {
		case 1:
			if (stage != ShaderType::Pixel) return false;
			break;
		case 2:
			if (stage != ShaderType::Compute) return false;
			break;
		case 3:
			if (stage == ShaderType::Pixel || stage == ShaderType::Compute) return false;
			break;
		default: break;
	}
	const uint64_t mask  = (uint64_t {1} << static_cast<uint32_t>(bits)) - 1u;
	const uint64_t value = static_cast<uint64_t>(value_switch.load(std::memory_order_relaxed));
	if ((shader_hash & mask) != (value & mask)) {
		return false;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> printed;
	static uint64_t                     printed_key = ~uint64_t {0};
	const uint64_t                      key         = (static_cast<uint64_t>(bits) << 56u) ^ (value & mask);
	std::lock_guard                     lock(mutex);
	if (key != printed_key) {
		printed.clear();
		printed_key = key;
	}
	if (printed.insert(shader_hash).second) {
		std::printf("LIVE-SKIP bits=%d value=0x%" PRIx64 " stage=%d hash=%016" PRIx64 "\n", static_cast<int>(bits),
		            value & mask, static_cast<int>(stage), shader_hash);
		std::fflush(stdout);
	}
	return true;
}

bool SkipShaderRequested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const auto parse = [&result](std::string_view list) {
			size_t start = 0;
			while (start < list.size()) {
				const auto end = std::min(list.find(',', start), list.size());
				if (end > start) {
					result.push_back(
					    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16));
				}
				start = end + 1;
			}
		};
		parse(Config::GetSkipShaderHashes());
		if (const char* value = std::getenv("KYTY_SKIP_SHADER_HASHES"); value != nullptr) {
			parse(value);
		}
		for (const auto hash: result) {
			LOGF("ProgramCache: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n",
			     hash);
		}
		return result;
	}();
	const auto found = std::ranges::find(hashes, shader_hash);
	if (found == hashes.end()) {
		return false;
	}
	// KYTY_SKIP_LIST_MASK=<mask> (live, diagnostic; default -1 = every listed shader skipped, as
	// before): bit i clear runs the i-th shader of the skip list (--skip-shaders order, then
	// KYTY_SKIP_SHADER_HASHES) again, to see in a running game whether a skipped shader is what
	// leaves vegetation popping or lighting flickering. Re-enabled shaders compile on first use.
	static auto& mask_switch = Common::LiveSwitches::Get("KYTY_SKIP_LIST_MASK", -1);
	const auto   index       = static_cast<uint64_t>(found - hashes.begin());
	const auto   mask        = static_cast<uint64_t>(mask_switch.load(std::memory_order_relaxed));
	if (index < 64 && ((mask >> index) & 1u) == 0) {
		static std::mutex                   mutex;
		static std::unordered_set<uint64_t> printed;
		std::lock_guard                     lock(mutex);
		if (printed.insert(shader_hash).second) {
			std::printf("SKIP-LIST: shader %016" PRIx64 " (entry %u) runs again (KYTY_SKIP_LIST_MASK)\n",
			            shader_hash, static_cast<unsigned>(index));
			std::fflush(stdout);
		}
		return false;
	}
	return true;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// A/B switch for background pipeline compiles (KYTY_ASYNC_PIPELINES=0 compiles every graphics
// pipeline on the GPU thread, as before) and how long a draw waits for a new one
// (KYTY_PIPELINE_WAIT_MS, default 20) before it is skipped.
static const bool g_async_pipelines = [] {
	const char* value = std::getenv("KYTY_ASYNC_PIPELINES");
	return value == nullptr || value[0] != '0';
}();
static const std::chrono::milliseconds g_pipeline_wait = [] {
	const char* value = std::getenv("KYTY_PIPELINE_WAIT_MS");
	return std::chrono::milliseconds(value != nullptr ? std::strtoul(value, nullptr, 10) : 20u);
}();

struct PendingGraphicsPipeline {
	std::unique_ptr<GraphicsPipelineBuild> build;
	vk::PipelineCache                      driver_cache = nullptr;
	vk::Pipeline                           pipeline     = nullptr;
	vk::Result                             result       = vk::Result::eSuccess;
	std::atomic<bool>                      done {false};
};

// Compiles graphics pipelines on worker threads. The driver compiled on the GPU thread, which
// stopped the whole emulated GPU for seconds per new pipeline (103 slow pipelines, 170 s of a
// 600 s first pass through the prologue).
class PipelineCompiler {
public:
	explicit PipelineCompiler(uint32_t threads) {
		for (uint32_t i = 0; i < threads; i++) {
			m_threads.emplace_back([this] { Prewarm::MarkAsyncThread(); Run(); });
		}
	}
	~PipelineCompiler() { Stop(); }
	KYTY_CLASS_NO_COPY(PipelineCompiler);

	void Submit(std::shared_ptr<PendingGraphicsPipeline> job) {
		{
			std::lock_guard lock(m_mutex);
			m_queue.push_back(std::move(job));
		}
		m_work.notify_one();
	}

	// Waits up to the budget; true when the job is done.
	bool Wait(const PendingGraphicsPipeline& job, std::chrono::milliseconds budget) {
		std::unique_lock lock(m_mutex);
		return m_done.wait_for(lock, budget,
		                       [&job] { return job.done.load(std::memory_order_acquire); });
	}

	// Finishes the compiles in progress; queued ones are left to their owner.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopped) {
				return;
			}
			m_stopped = true;
			m_queue.clear();
		}
		m_work.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
		m_threads.clear();
	}

	[[nodiscard]] bool Stopped() {
		std::lock_guard lock(m_mutex);
		return m_stopped;
	}

private:
	void Run() {
		for (;;) {
			std::shared_ptr<PendingGraphicsPipeline> job;
			{
				std::unique_lock lock(m_mutex);
				m_work.wait(lock, [this] { return m_stopped || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				job = std::move(m_queue.front());
				m_queue.pop_front();
			}
			job->result = CreateGraphicsPipeline(*job->build, job->driver_cache, &job->pipeline);
			{
				std::lock_guard lock(m_mutex);
				job->done.store(true, std::memory_order_release);
			}
			m_done.notify_all();
		}
	}

	std::mutex                                           m_mutex;
	std::condition_variable                              m_work;
	std::condition_variable                              m_done;
	std::deque<std::shared_ptr<PendingGraphicsPipeline>> m_queue;
	std::vector<std::thread>                             m_threads;
	bool                                                 m_stopped = false;
};

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

// wolv-perf-1111b: KYTY_ASYNC_PS (live, default 1; 0 = old behaviour, every new pixel program
// translated inline on Thread_Gpu). Log 2026-11-11: in the slow area Thread_Gpu spent 1-2 s of every
// 5 s in "prog.recompile" (~35 ms per new program, 70 % of them pixel shaders), with 0.6 % idle.
// A brand-new pixel program is translated (RDNA2 -> IR, the expensive part) on worker threads
// (KYTY_ASYNC_PS_THREADS, default 3); its draws are skipped while that runs, only when the draw
// writes no depth/stencil (same visible-critical rule as the background pipeline compile). Any other
// draw that needs the program waits for the job (or runs it inline if no worker took it yet). The
// SPIR-V emission, resource plan and module creation stay on Thread_Gpu, as before. Results with
// subroutine calls or unsupported translations are discarded and redone inline (old path).
// Log: "Async PS translate (5 s, KYTY_ASYNC_PS=N): ..." every 5 s.
thread_local bool t_async_ps_may_skip = false; // set by the draw path (PrepareDrawRenderState)
thread_local bool t_async_ps_pending  = false; // the last pixel lookup was skipped while translating

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;
		// Exact expanded code includes callees and their original return-PC constants.
		// A different target/body must not reuse a program compiled for an earlier call.
		// wolv-draw-1111: shared with the expander's cache and compared by its XXH3-128 and
		// size (was a copy compared word by word, up to 128K words, on every compute lookup).
		std::shared_ptr<const std::vector<uint32_t>> function_code;
		uint64_t                                     function_hash_lo = 0;
		uint64_t                                     function_hash_hi = 0;
		uint32_t                                     function_size    = 0;

		bool operator==(const ProgramKey& other) const {
			return stage == other.stage && hash == other.hash &&
			       user_data_count == other.user_data_count && code_size == other.code_size &&
			       function_size == other.function_size && function_hash_lo == other.function_hash_lo &&
			       function_hash_hi == other.function_hash_hi && static_state == other.static_state;
		}
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	// KYTY_RESOURCE_MEMO: the inputs and results of a recent MaterializeResources call.
	struct MemoSlot {
		std::vector<uint32_t>                        user_data;
		uint64_t                                     shader_base = 0;
		// Compute dispatch size: bounds the written extent of each written buffer.
		std::array<uint32_t, 3>                      workgroup_count {};
		std::array<uint32_t, 3>                      workgroup_size {};
		ShaderReadChunks::ReadLog                    log;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                                     last_use = 0;
		uint64_t                                     input_hash = 0;
	};
	// A snapshot word: kind 0/1/2 = buffers/images/samplers[index].dwords[dword], 3 = flattened_srt[index].
	struct PatchPosition {
		uint8_t  kind  = 0;
		uint8_t  word  = 0; // the user-data word it copies
		uint16_t dword = 0;
		uint32_t index = 0;
		uint32_t offset = 0; // the snapshot word = user-data word + offset (mod 2^32)
		bool     operator==(const PatchPosition&) const = default;
	};
	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque: input_info.stage points at a permutation's program, and a new permutation of
		// the same shader must not move the ones already handed out (a vector reallocated them
		// past 8 permutations).
		std::deque<Permutation>                     permutations;
		std::vector<MemoSlot>                        memo;
		// KYTY_MEMO_USED_KEY: which user-data words the resource plan reads (GetUserData), from
		// a scan of the plan's own instructions; empty until computed, all-true when unsure.
		std::vector<uint8_t>                         used_user_data;
		bool                                         used_computed = false;
		// Memo miss diagnostics: new-input misses this window, and the last missed key.
		uint64_t                                     new_input_misses = 0;
		std::vector<uint32_t>                        last_missed_key;
		std::array<uint64_t, 64>                     varying_word_count {};
		uint64_t                                     base_changes = 0, workgroup_changes = 0;
		uint64_t                                     last_missed_base = 0;
		std::array<uint32_t, 3>                      last_missed_groups {};
		// KYTY_MEMO_BYPASS: lookups and hits in the current 256-lookup window, and how many
		// lookups still go straight to MaterializeResources.
		uint32_t                                     bypass_window = 0, bypass_window_hits = 0;
		uint32_t                                     bypass_left   = 0;
		// KYTY_MEMO_PATCH: the last full evaluation and what it learned (see TryPatch).
		struct Patch {
			bool                                         valid = false;
			bool                                         banned = false;
			std::vector<uint32_t>                        key; // all user-data words
			uint64_t                                     shader_base = 0;
			std::array<uint32_t, 3>                      workgroup_count {};
			std::array<uint32_t, 3>                      workgroup_size {};
			ShaderReadChunks::ReadLog                    log;
			ShaderRecompiler::IR::ResourceSnapshot       resources;
			ShaderRecompiler::IR::ResourceSpecialization specialization;
			// Per user-data word: consistent observations (word changed, every snapshot change
			// it caused was a verbatim copy of it, nothing else changed); 255 = never patch it.
			std::array<uint8_t, 64>                      learned {};
			// Snapshot words that are verbatim copies of a user-data word.
			std::vector<PatchPosition>                   positions;
			// Per read of `log`: -1 at a fixed address, else the user-data word whose 32-bit change
			// moves its address by the same amount (a pointer into a ring of tables).
			std::vector<int8_t>                          read_word;
			uint32_t                                     since_check = 0;
			// Full evaluations learned from and patches applied: a program with 64 of the first
			// and none of the second stops learning (its read logs and copies cost every miss).
			uint32_t                                     learned_from = 0, patches = 0;
		} patch;
	};
	SourceEntry* memo_worst = nullptr;

	// --- KYTY_MEMO_PATCH (wolv-draw-1111) -------------------------------------------------------
	// Log 11-11: ~120K memo misses per 5 s at 9-10 us, most of them programs whose user data
	// changes on every draw (0xc8202066634a4c62: words 8, 10, 20 new on each miss, ring-allocated
	// constants). When such a word only ever lands verbatim in the snapshot (a V# base or size
	// built from user data, a flattened-SRT copy) the snapshot of the new draw is the previous full
	// evaluation's with those words replaced. Learned per program and word: after a full
	// evaluation, the previous one is compared with it; a word that changed counts as consistent
	// when every snapshot word that changed held that user-data word before and after, the read
	// log (addresses, sizes and values), the specialization and every other snapshot field are
	// identical. A word with 2 consistent observations and no inconsistent one is patchable; one
	// inconsistent observation bans the word for good. A lookup whose changed words (among those
	// the plan reads) are all patchable, whose shader base and dispatch size are the previous
	// evaluation's and whose logged guest words all read the same, copies that evaluation and
	// patches the words. Every 32nd patch is also evaluated in full and compared; a mismatch bans
	// the program (then the full result is used). KYTY_MEMO_PATCH=2 compares every patch.
	// 0 = old path. Log: "Memo patch (5 s)".
	static int64_t MemoPatchMode() {
		static auto& mode = Common::LiveSwitches::Get("KYTY_MEMO_PATCH", 1);
		return mode.load(std::memory_order_relaxed);
	}
	static constexpr uint8_t PatchLearnedNeeded = 2;
	static constexpr uint8_t PatchBanned        = 255;
	uint64_t patch_hits = 0, patch_checks = 0, patch_check_mismatch = 0, patch_verify_mismatch = 0;
	uint64_t patch_no_base = 0, patch_unlearned = 0, patch_reads_changed = 0, patch_banned_programs = 0;
	uint64_t patch_obs_ok = 0, patch_obs_log = 0, patch_obs_spec = 0, patch_obs_other = 0, patch_obs_unexplained = 0;
	uint64_t patch_obs_many = 0, patch_obs_memory = 0, patch_obs_ambiguous = 0, patch_obs_shape = 0;
	uint64_t patch_relocated_reads = 0, patch_gave_up = 0;
	std::chrono::steady_clock::time_point patch_report = std::chrono::steady_clock::now();

	static uint32_t* PatchSlot(ShaderRecompiler::IR::ResourceSnapshot& r, const PatchPosition& p) {
		switch (p.kind) {
			case 0: return p.index < r.buffers.size() ? &r.buffers[p.index].dwords[p.dword & 7u] : nullptr;
			case 1: return p.index < r.images.size() ? &r.images[p.index].dwords[p.dword & 7u] : nullptr;
			case 2: return p.index < r.samplers.size() ? &r.samplers[p.index].dwords[p.dword & 7u] : nullptr;
			default: return p.index < r.flattened_srt.size() ? &r.flattened_srt[p.index] : nullptr;
		}
	}
	static bool SnapshotsEqual(const ShaderRecompiler::IR::ResourceSnapshot& a,
	                           const ShaderRecompiler::IR::ResourceSnapshot& b) {
		return a.buffer_write_extents == b.buffer_write_extents && a.bindless_heaps == b.bindless_heaps &&
		       a.bindless_sampler_heaps == b.bindless_sampler_heaps && a.buffers == b.buffers &&
		       a.images == b.images && a.samplers == b.samplers && a.flattened_srt == b.flattened_srt &&
		       a.user_data == b.user_data && a.specialization_reads == b.specialization_reads &&
		       a.uniform_fill == b.uniform_fill;
	}
	static bool LogShapesEqual(const ShaderReadChunks::ReadLog& a, const ShaderReadChunks::ReadLog& b) {
		if (a.reads.size() != b.reads.size() || a.words.size() != b.words.size()) return false;
		for (size_t i = 0; i < a.reads.size(); i++) {
			const auto& x = a.reads[i];
			const auto& y = b.reads[i];
			if (x.address != y.address || x.count != y.count || x.ok != y.ok || x.first != y.first) return false;
		}
		return true;
	}
	static bool UsedWord(const SourceEntry& entry, size_t i) {
		return !entry.used_computed || (i < entry.used_user_data.size() && entry.used_user_data[i] != 0);
	}

	static int64_t WordDelta(uint32_t now, uint32_t before) {
		return static_cast<int64_t>(static_cast<int32_t>(now - before));
	}

	// The patched result into entry.resources/specialization, or false (nothing changed).
	bool TryPatch(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime, ShaderReadChunks& reads) {
		auto& pt = entry.patch;
		if (!pt.valid || pt.banned || pt.key.size() != runtime.user_data.size() ||
		    pt.shader_base != runtime.shader_base || pt.workgroup_count != runtime.workgroup_count ||
		    pt.workgroup_size != runtime.workgroup_size) {
			patch_no_base++;
			return false;
		}
		bool any = false;
		for (size_t i = 0; i < pt.key.size(); i++) {
			if (pt.key[i] == runtime.user_data[i] || !UsedWord(entry, i)) continue;
			if (i >= pt.learned.size() || pt.learned[i] < PatchLearnedNeeded || pt.learned[i] == PatchBanned) {
				patch_unlearned++;
				return false;
			}
			any = true;
		}
		if (!any) {
			patch_no_base++;
			return false; // same inputs: the memo's case
		}
		// The logged reads, those that move with a pointer word moved by its change, must read
		// the same words as in the base evaluation.
		const auto& log = pt.log;
		for (size_t r = 0; r < log.reads.size(); r++) {
			const auto& read    = log.reads[r];
			uint64_t    address = read.address;
			if (r < pt.read_word.size() && pt.read_word[r] >= 0) {
				const auto w = static_cast<size_t>(pt.read_word[r]);
				address += static_cast<uint64_t>(WordDelta(runtime.user_data[w], pt.key[w]));
				patch_relocated_reads++;
			}
			memo_words.resize(read.count);
			const bool ok = ReadShaderGuestMemoryImpl(&reads, address, memo_words);
			if (ok != read.ok || (ok && !std::equal(memo_words.begin(), memo_words.end(),
			                                        log.words.begin() + read.first))) {
				patch_reads_changed++;
				return false;
			}
		}
		entry.resources      = pt.resources;
		entry.specialization = pt.specialization;
		for (const auto& position: pt.positions) {
			if (auto* word = PatchSlot(entry.resources, position); word != nullptr) {
				*word = runtime.user_data[position.word] + position.offset;
			}
		}
		entry.resources.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
		patch_hits++;
		pt.patches++;
		return true;
	}

	// After a full evaluation (entry.resources/specialization hold it): learn from the previous
	// one, then keep this one as the base for the next patch.
	void LearnPatch(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                ShaderReadChunks::ReadLog& log) {
		auto& pt = entry.patch;
		if (pt.banned) return;
		if (++pt.learned_from >= 64u && pt.patches == 0) {
			pt.banned = true;
			patch_gave_up++;
			return;
		}
		const auto& ud = runtime.user_data;
		bool keep_relations = true;
		if (pt.valid && pt.key.size() == ud.size() && pt.shader_base == runtime.shader_base &&
		    pt.workgroup_count == runtime.workgroup_count && pt.workgroup_size == runtime.workgroup_size) {
			std::array<uint32_t, 64> changed {};
			uint32_t                 changed_count = 0;
			uint64_t                 changed_mask  = 0;
			bool                     too_many      = false;
			for (size_t i = 0; i < pt.key.size(); i++) {
				if (pt.key[i] == ud[i] || !UsedWord(entry, i)) continue;
				if (i >= changed.size() || changed_count >= 8) {
					too_many = true;
					break;
				}
				changed[changed_count++] = static_cast<uint32_t>(i);
				changed_mask |= uint64_t {1} << i;
			}
			// The changed word whose delta is `delta` (32-bit wrap or sign-extended), -1 none, -2 several.
			const auto candidate = [&](int64_t delta, bool wrap32) -> int {
				int found = -1;
				for (uint32_t c = 0; c < changed_count; c++) {
					const auto w = changed[c];
					const bool match =
					    wrap32 ? static_cast<uint32_t>(delta) == ud[w] - pt.key[w] : delta == WordDelta(ud[w], pt.key[w]);
					if (match) {
						if (found != -1) return -2;
						found = static_cast<int>(w);
					}
				}
				return found;
			};
			if (too_many) {
				patch_obs_many++;
			} else if (changed_count != 0) {
				const auto& oldr = pt.resources;
				const auto& newr = entry.resources;
				// 0 ok, 1 read shape/address unexplained, 2 spec, 3 other, 4 unexplained snapshot word,
				// 5 guest words changed, 6 ambiguous (learn nothing, ban nothing)
				int  verdict          = 0;
				bool positions_broken = false;
				patch_scratch.clear();
				patch_read_scratch.assign(log.reads.size(), -1);
				bool shape = log.reads.size() == pt.log.reads.size() && log.words.size() == pt.log.words.size();
				for (size_t r = 0; shape && r < log.reads.size(); r++) {
					shape = log.reads[r].count == pt.log.reads[r].count && log.reads[r].ok == pt.log.reads[r].ok &&
					        log.reads[r].first == pt.log.reads[r].first;
				}
				if (!shape) {
					verdict        = 1;
					keep_relations = false; // a new read structure: relations by index mean nothing
				}
				for (size_t r = 0; verdict == 0 && r < log.reads.size(); r++) {
					const auto delta = static_cast<int64_t>(log.reads[r].address - pt.log.reads[r].address);
					const int  known = r < pt.read_word.size() ? pt.read_word[r] : -1;
					if (delta == 0) {
						if (known >= 0 && ((changed_mask >> known) & 1u) != 0) verdict = 1;
						continue;
					}
					const int w = candidate(delta, false);
					if (w == -2) {
						verdict = 6;
					} else if (w == -1 || (known >= 0 && known != w)) {
						verdict = 1;
					} else {
						patch_read_scratch[r] = static_cast<int8_t>(w);
					}
				}
				if (verdict == 0 && pt.log.words != log.words) {
					verdict = 5;
				} else if (verdict == 0 && !(pt.specialization == entry.specialization)) {
					verdict = 2;
				} else if (verdict == 0 &&
				           (!(oldr.buffer_write_extents == newr.buffer_write_extents) ||
				            !(oldr.bindless_heaps == newr.bindless_heaps) ||
				            !(oldr.bindless_sampler_heaps == newr.bindless_sampler_heaps) ||
				            !(oldr.specialization_reads == newr.specialization_reads) ||
				            !(oldr.uniform_fill == newr.uniform_fill) || oldr.buffers.size() != newr.buffers.size() ||
				            oldr.images.size() != newr.images.size() || oldr.samplers.size() != newr.samplers.size() ||
				            oldr.flattened_srt.size() != newr.flattened_srt.size())) {
					verdict = 3;
				}
				const auto explain = [&](uint8_t kind, uint32_t index, uint32_t dword, uint32_t before, uint32_t after) {
					if (before == after) return true;
					const int w = candidate(static_cast<int64_t>(after - before), true);
					if (w == -2) {
						verdict = 6;
						return false;
					}
					if (w == -1) return false;
					patch_scratch.push_back({kind, static_cast<uint8_t>(w), static_cast<uint16_t>(dword), index,
					                         before - pt.key[static_cast<size_t>(w)]});
					return true;
				};
				const auto descriptors = [&](uint8_t kind, const auto& a, const auto& b) {
					for (uint32_t i = 0; i < a.size(); i++) {
						if (a[i].dword_count != b[i].dword_count) return false;
						for (uint32_t d = 0; d < 8; d++) {
							if (!explain(kind, i, d, a[i].dwords[d], b[i].dwords[d])) return false;
						}
					}
					return true;
				};
				if (verdict == 0) {
					bool ok = descriptors(0, oldr.buffers, newr.buffers) && descriptors(1, oldr.images, newr.images) &&
					          descriptors(2, oldr.samplers, newr.samplers);
					for (uint32_t i = 0; ok && i < oldr.flattened_srt.size(); i++) {
						ok = explain(3, i, 0, oldr.flattened_srt[i], newr.flattened_srt[i]);
					}
					if (!ok && verdict == 0) verdict = 4;
				}
				// Every position learned earlier must still hold its word plus offset.
				if (verdict == 0 || verdict == 4) {
					for (const auto& position: pt.positions) {
						const auto* word = PatchSlot(entry.resources, position);
						if (word == nullptr || position.word >= ud.size() || *word != ud[position.word] + position.offset) {
							positions_broken = true;
							break;
						}
					}
					if (positions_broken && verdict == 0) verdict = 4;
				}
				if (verdict == 0) {
					patch_obs_ok++;
					for (const auto& position: patch_scratch) {
						if (std::ranges::find(pt.positions, position) == pt.positions.end()) {
							pt.positions.push_back(position);
						}
					}
					pt.read_word.resize(log.reads.size(), -1);
					for (size_t r = 0; r < log.reads.size(); r++) {
						if (patch_read_scratch[r] >= 0) pt.read_word[r] = patch_read_scratch[r];
					}
					for (uint32_t c = 0; c < changed_count; c++) {
						auto& learned = pt.learned[changed[c]];
						if (learned < PatchLearnedNeeded) learned++;
					}
				} else if (verdict == 5) {
					patch_obs_memory++;
				} else if (verdict == 6) {
					patch_obs_ambiguous++;
				} else if (verdict == 1 && !keep_relations) {
					patch_obs_shape++;
				} else {
					(verdict == 1 ? patch_obs_log
					              : verdict == 2 ? patch_obs_spec : verdict == 3 ? patch_obs_other : patch_obs_unexplained)++;
					for (uint32_t c = 0; c < changed_count; c++) {
						pt.learned[changed[c]] = PatchBanned;
					}
					if (positions_broken) {
						pt.banned = true; // a learned relation stopped holding: trust nothing learned here
						patch_banned_programs++;
					}
				}
			}
		}
		if (!keep_relations) {
			// The read structure changed: what was learned by read index or snapshot index is void.
			pt.read_word.clear();
			pt.positions.clear();
			for (auto& learned: pt.learned) {
				if (learned != PatchBanned) learned = 0;
			}
		}
		pt.valid = true;
		pt.key.assign(ud.begin(), ud.end());
		pt.shader_base     = runtime.shader_base;
		pt.workgroup_count = runtime.workgroup_count;
		pt.workgroup_size  = runtime.workgroup_size;
		pt.log             = log;
		pt.read_word.resize(log.reads.size(), -1);
		pt.resources       = entry.resources;
		pt.specialization  = entry.specialization;
	}
	std::vector<int8_t>        patch_read_scratch;
	std::vector<PatchPosition> patch_scratch;

	// Patch, check or verify; true when entry.resources/specialization hold the result.
	bool MaterializeWithPatch(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                          ShaderReadChunks& reads, bool& patched) {
		patched = false;
		const auto mode = MemoPatchMode();
		if (mode == 0) {
			return false;
		}
		bool tried = false;
		{
			PhaseStats::Scope patch_phase {PhaseStats::PhMemoPatch};
			tried = TryPatch(entry, runtime, reads);
		}
		if (!tried) {
			return false;
		}
		auto& pt = entry.patch;
		if (mode != 2 && ++pt.since_check < 32u) {
			patched = true;
			return true;
		}
		pt.since_check = 0;
		// Check: the full evaluation must give the same.
		patch_checks++;
		patch_check_resources      = entry.resources;
		patch_check_specialization = entry.specialization;
		PhaseStats::Scope miss_phase {PhaseStats::PhMemoMiss};
		if (!ShaderRecompiler::IR::MaterializeResources(entry.resource_plan, runtime, entry.resources,
		                                                entry.specialization)) {
			return false; // let the caller evaluate again and fail as before
		}
		if (!SnapshotsEqual(patch_check_resources, entry.resources) ||
		    !(patch_check_specialization == entry.specialization)) {
			(mode == 2 ? patch_verify_mismatch : patch_check_mismatch)++;
			if (mode != 2) {
				pt.banned = true;
				patch_banned_programs++;
			}
			static std::atomic<uint32_t> reported {0};
			if (reported.fetch_add(1) < 16) {
				::printf("Memo patch MISMATCH: shader 0x%016" PRIx64 " stage %u (program %s)\n",
				         entry.resource_plan.shader_hash, static_cast<uint32_t>(entry.resource_plan.stage),
				         mode == 2 ? "kept, verify mode" : "banned");
			}
		}
		patched = true; // entry holds the full evaluation
		return true;
	}
	ShaderRecompiler::IR::ResourceSnapshot       patch_check_resources;
	ShaderRecompiler::IR::ResourceSpecialization patch_check_specialization;

	void ReportPatch() {
		const auto now = std::chrono::steady_clock::now();
		if (now - patch_report < std::chrono::seconds(5)) {
			return;
		}
		patch_report = now;
		::printf("Memo patch (5 s, KYTY_MEMO_PATCH=%lld): %llu lookups patched instead of evaluated (est. saved %.0f ms "
		         "at the memo.miss us/call); declined: %llu no base/same inputs, %llu a changed word not learned, %llu "
		         "logged reads changed; learning: %llu consistent, inconsistent %llu reads / %llu specialization / %llu "
		         "other fields / %llu unexplained words, %llu too many words changed, %llu guest words changed, %llu ambiguous, "
		         "%llu new read structure; %llu reads relocated with a pointer word; %llu programs gave up (64 evaluations, no patch); checks %llu, mismatches %llu "
		         "(%llu programs banned), verify mismatches %llu\n",
		         static_cast<long long>(MemoPatchMode()), static_cast<unsigned long long>(patch_hits),
		         static_cast<double>(patch_hits) * 9.5 / 1000.0, static_cast<unsigned long long>(patch_no_base),
		         static_cast<unsigned long long>(patch_unlearned), static_cast<unsigned long long>(patch_reads_changed),
		         static_cast<unsigned long long>(patch_obs_ok), static_cast<unsigned long long>(patch_obs_log),
		         static_cast<unsigned long long>(patch_obs_spec), static_cast<unsigned long long>(patch_obs_other),
		         static_cast<unsigned long long>(patch_obs_unexplained), static_cast<unsigned long long>(patch_obs_many),
		         static_cast<unsigned long long>(patch_obs_memory), static_cast<unsigned long long>(patch_obs_ambiguous),
		         static_cast<unsigned long long>(patch_obs_shape), static_cast<unsigned long long>(patch_relocated_reads),
		         static_cast<unsigned long long>(patch_gave_up),
		         static_cast<unsigned long long>(patch_checks), static_cast<unsigned long long>(patch_check_mismatch),
		         static_cast<unsigned long long>(patch_banned_programs), static_cast<unsigned long long>(patch_verify_mismatch));
		patch_hits = patch_checks = patch_check_mismatch = patch_verify_mismatch = 0;
		patch_no_base = patch_unlearned = patch_reads_changed = patch_banned_programs = 0;
		patch_obs_ok = patch_obs_log = patch_obs_spec = patch_obs_other = patch_obs_unexplained = patch_obs_many = patch_obs_memory = 0;
		patch_obs_ambiguous = patch_obs_shape = patch_relocated_reads = patch_gave_up = 0;
		std::fflush(stdout);
	}

	// The plan owns every instruction it evaluates (value_storage), and user data reaches the
	// walk, its native code and the clean walk only through GetUserData. Words no GetUserData
	// names cannot change the snapshot or the specialization, except snapshot.user_data (a plain
	// copy, restored on a hit). An operand that is not a constant scalar register marks all used.
	static void ComputeUsedUserData(SourceEntry& entry) {
		const auto& plan = entry.resource_plan;
		entry.used_user_data.assign(plan.user_data_count, 0);
		bool all = false;
		for (const auto& inst: plan.value_storage) {
			if (inst.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) {
				continue;
			}
			if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != ShaderRecompiler::IR::Type::ScalarReg) {
				all = true;
				break;
			}
			const auto reg = ShaderRecompiler::IR::RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < plan.user_data_base || reg - plan.user_data_base >= entry.used_user_data.size()) {
				continue; // the walker fails on it whatever the data (see SrtWalker)
			}
			entry.used_user_data[reg - plan.user_data_base] = 1;
		}
		if (all) {
			entry.used_user_data.assign(plan.user_data_count, 1);
		}
		entry.used_computed = true;
	}
	std::vector<uint32_t> memo_key_words;
	uint64_t              memo_masked_words = 0, memo_total_words = 0;

	// KYTY_RESOURCE_MEMO_SLOTS (live, default 32; was a fixed 8): input sets kept per program.
	// Owner log 20261027-172055: 21-25% hits with 8; one program drawn for many objects a frame
	// cycles through more input sets than that, evicting each before its next frame.
	static size_t MemoSlots() {
		static auto& slots = Common::LiveSwitches::Get("KYTY_RESOURCE_MEMO_SLOTS", 128);
		return static_cast<size_t>(std::clamp<int64_t>(slots.load(std::memory_order_relaxed), 1, 256));
	}
	static uint64_t MemoInputHash(const ShaderRecompiler::IR::SrtRuntime& runtime,
	                              std::span<const uint32_t>               key_words) {
		uint64_t h = XXH3_64bits(key_words.data(), key_words.size_bytes());
		h ^= runtime.shader_base * 0x9e3779b97f4a7c15ull;
		h ^= XXH3_64bits(runtime.workgroup_count.data(), sizeof(runtime.workgroup_count)) * 31u;
		h ^= XXH3_64bits(runtime.workgroup_size.data(), sizeof(runtime.workgroup_size)) * 131u;
		return h;
	}
	uint64_t memo_miss_inputs = 0, memo_miss_reads = 0, memo_bypassed = 0, memo_bypass_starts = 0;

	// Materialization reads guest memory only through the runtime's two readers, so the same
	// user data and shader base, with every logged read returning the same result and words,
	// give the same snapshot and specialization. A hit re-reads the logged words and copies the
	// stored result instead of evaluating the resource plan again.
	bool Materialize(SourceEntry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                 ShaderReadChunks& reads) {
		if (!ResourceMemoEnabled()) {
			return ShaderRecompiler::IR::MaterializeResources(
			    entry.resource_plan, runtime, entry.resources, entry.specialization);
		}
		memo_clock++;
		// KYTY_MEMO_BYPASS (live, default 1; 0 = old): a program whose last 256 lookups all missed
		// (log 11-11 06:30: 0xc8202066634a4c62, 21K new-input misses per 5 s, 128 slots full, a
		// pointer word new on every draw) evaluates its plan directly for its next 2048 lookups:
		// no key hash, slot scan, read log or snapshot copy into a slot. Same result as a miss.
		static auto& bypass_switch = Common::LiveSwitches::Get("KYTY_MEMO_BYPASS", 1);
		const bool   bypass_on     = bypass_switch.load(std::memory_order_relaxed) != 0;
		if (bypass_on && entry.bypass_left != 0) {
			entry.bypass_left--;
			memo_bypassed++;
			if (bool patched = false; MaterializeWithPatch(entry, runtime, reads, patched)) {
				ReportMemo();
				ReportPatch();
				return true;
			}
			PhaseStats::Scope miss_phase {PhaseStats::PhMemoMiss};
			const bool learn = MemoPatchMode() != 0 && !entry.patch.banned;
			ShaderReadChunks::ReadLog log;
			if (learn) reads.SetLog(&log);
			const bool ok = ShaderRecompiler::IR::MaterializeResources(
			    entry.resource_plan, runtime, entry.resources, entry.specialization);
			reads.SetLog(nullptr);
			if (ok && learn) LearnPatch(entry, runtime, log);
			ReportMemo();
			ReportPatch();
			return ok;
		}
		if (++entry.bypass_window >= 256u) {
			if (bypass_on && entry.bypass_window_hits == 0) {
				entry.bypass_left = 2048;
				memo_bypass_starts++;
			}
			entry.bypass_window = entry.bypass_window_hits = 0;
		}
		// KYTY_MEMO_USED_KEY (live, default 1): the key holds only the user-data words the plan
		// reads; the others (per-object constants and addresses the walk never follows) are
		// zeroed, so objects drawn with the same descriptor tables share a slot.
		static auto& used_key = Common::LiveSwitches::Get("KYTY_MEMO_USED_KEY", 1);
		memo_key_words.assign(runtime.user_data.begin(), runtime.user_data.end());
		if (used_key.load(std::memory_order_relaxed) != 0) {
			if (!entry.used_computed) {
				ComputeUsedUserData(entry);
			}
			for (size_t i = 0; i < memo_key_words.size(); i++) {
				if (i >= entry.used_user_data.size() || entry.used_user_data[i] == 0) {
					memo_key_words[i] = 0;
					memo_masked_words++;
				}
			}
		}
		memo_total_words += memo_key_words.size();
		const std::span<const uint32_t> key_words {memo_key_words};
		const uint64_t input_hash = MemoInputHash(runtime, key_words);
		bool           inputs_seen = false;
		for (auto& slot: entry.memo) {
			if (slot.input_hash != input_hash) {
				continue;
			}
			if (slot.shader_base == runtime.shader_base &&
			    slot.workgroup_count == runtime.workgroup_count &&
			    slot.workgroup_size == runtime.workgroup_size &&
			    std::ranges::equal(slot.user_data, key_words)) {
				inputs_seen = true;
				if (!ReadsUnchanged(slot.log, reads)) {
					continue;
				}
				PhaseStats::Scope hit_phase {PhaseStats::PhMemoHit};
				entry.resources      = slot.resources;
				entry.specialization = slot.specialization;
				// The only verbatim copy of the inputs in the snapshot (shader data words).
				entry.resources.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
				slot.last_use = memo_clock;
				memo_hits++;
				entry.bypass_window_hits++;
				ReportMemo();
				return true;
			}
		}
		if (bool patched = false; MaterializeWithPatch(entry, runtime, reads, patched)) {
			ReportMemo();
			ReportPatch();
			return true;
		}
		memo_misses++;
		(inputs_seen ? memo_miss_reads : memo_miss_inputs)++;
		PhaseStats::Scope miss_phase {PhaseStats::PhMemoMiss};
		if (!inputs_seen) {
			// Which key words differ from the previous new-input miss of this program.
			entry.new_input_misses++;
			if (entry.last_missed_key.size() == key_words.size()) {
				for (size_t i = 0; i < key_words.size() && i < entry.varying_word_count.size(); i++) {
					if (entry.last_missed_key[i] != key_words[i]) {
						entry.varying_word_count[i]++;
					}
				}
			}
			if (entry.last_missed_base != runtime.shader_base) {
				entry.base_changes++;
			}
			if (entry.last_missed_groups != runtime.workgroup_count) {
				entry.workgroup_changes++;
			}
			entry.last_missed_key.assign(key_words.begin(), key_words.end());
			entry.last_missed_base   = runtime.shader_base;
			entry.last_missed_groups = runtime.workgroup_count;
			if (memo_worst == nullptr || entry.new_input_misses > memo_worst->new_input_misses) {
				memo_worst = &entry;
			}
		}
		ShaderReadChunks::ReadLog log;
		reads.SetLog(&log);
		const bool ok = ShaderRecompiler::IR::MaterializeResources(
		    entry.resource_plan, runtime, entry.resources, entry.specialization);
		reads.SetLog(nullptr);
		if (ok && MemoPatchMode() != 0 && !entry.patch.banned) {
			LearnPatch(entry, runtime, log);
		}
		if (ok) {
			MemoSlot* slot = nullptr;
			// Same inputs whose reads changed: refresh that slot instead of taking another.
			for (auto& candidate: entry.memo) {
				if (candidate.input_hash == input_hash && candidate.shader_base == runtime.shader_base &&
				    std::ranges::equal(candidate.user_data, key_words)) {
					slot = &candidate;
					break;
				}
			}
			if (slot == nullptr) {
				if (entry.memo.size() < MemoSlots()) {
					slot = &entry.memo.emplace_back();
				} else {
					slot = &*std::ranges::min_element(entry.memo, {}, &MemoSlot::last_use);
				}
			}
			slot->input_hash = input_hash;
			slot->user_data.assign(key_words.begin(), key_words.end());
			slot->shader_base    = runtime.shader_base;
			slot->workgroup_count = runtime.workgroup_count;
			slot->workgroup_size  = runtime.workgroup_size;
			slot->log            = std::move(log);
			slot->resources      = entry.resources;
			slot->specialization = entry.specialization;
			slot->last_use = memo_clock;
		}
		ReportMemo();
		ReportPatch();
		return ok;
	}

	bool ReadsUnchanged(const ShaderReadChunks::ReadLog& log, ShaderReadChunks& reads) {
		for (const auto& read: log.reads) {
			memo_words.resize(read.count);
			const bool ok = ReadShaderGuestMemoryImpl(&reads, read.address, memo_words);
			if (ok != read.ok || (ok && !std::equal(memo_words.begin(), memo_words.end(),
			                                        log.words.begin() + read.first))) {
				return false;
			}
		}
		return true;
	}

	void ReportMemo() {
		const auto now = std::chrono::steady_clock::now();
		if (now - memo_report < std::chrono::seconds(5)) {
			return;
		}
		const auto total = memo_hits + memo_misses;
		::printf("Resource memo (5 s): %" PRIu64 " hits, %" PRIu64 " misses (%.1f%% hits; misses: %" PRIu64
		         " new inputs, %" PRIu64 " same inputs but guest words changed; %zu slots); lookup cache %" PRIu64
		         " hits, %" PRIu64 " misses\n",
		         memo_hits, memo_misses, total != 0 ? 100.0 * memo_hits / total : 0.0, memo_miss_inputs,
		         memo_miss_reads, MemoSlots(), lookup_hits, lookup_misses);
		::printf("Resource memo bypass (5 s, KYTY_MEMO_BYPASS=%lld): %" PRIu64 " lookups evaluated directly, %" PRIu64
		         " programs entered bypass\n",
		         static_cast<long long>(Common::LiveSwitches::Get("KYTY_MEMO_BYPASS", 1).load()), memo_bypassed,
		         memo_bypass_starts);
		memo_bypassed = memo_bypass_starts = 0;
		::printf("Resource memo key (5 s): KYTY_MEMO_USED_KEY=%lld, %.1f%% of user-data words left out of the key\n",
		         static_cast<long long>(Common::LiveSwitches::Get("KYTY_MEMO_USED_KEY", 1).load()),
		         memo_total_words != 0 ? 100.0 * static_cast<double>(memo_masked_words) / memo_total_words : 0.0);
		if (memo_worst != nullptr && memo_worst->new_input_misses != 0) {
			// The program with most new-input misses: how often each key word changed between
			// consecutive misses (u = the plan reads it, - = left out of the key).
			auto& w = *memo_worst;
			std::string words;
			for (size_t i = 0; i < w.last_missed_key.size() && i < w.varying_word_count.size(); i++) {
				if (w.varying_word_count[i] != 0) {
					words += fmt::format(" [{}{}]{}", i,
					                     i < w.used_user_data.size() && w.used_user_data[i] ? "u" : "-",
					                     w.varying_word_count[i]);
				}
			}
			::printf("Resource memo worst (5 s): shader 0x%016" PRIx64 " stage %u, %" PRIu64
			         " new-input misses, %zu memo slots used, user data %zu words; shader base changed %" PRIu64
			         ", workgroups changed %" PRIu64 "; changing words:%s\n",
			         w.resource_plan.shader_hash, static_cast<uint32_t>(w.resource_plan.stage), w.new_input_misses,
			         w.memo.size(), w.last_missed_key.size(), w.base_changes, w.workgroup_changes, words.c_str());
		}
		for (auto& [key, e]: programs) {
			(void)key;
			e.new_input_misses = 0;
			e.varying_word_count.fill(0);
			e.base_changes = e.workgroup_changes = 0;
		}
		memo_worst = nullptr;
		memo_miss_inputs = memo_miss_reads = 0;
		memo_masked_words = memo_total_words = 0;
		lookup_hits = lookup_misses = 0;
		std::fflush(stdout);
		memo_hits   = 0;
		memo_misses = 0;
		memo_report = now;
	}

	// KYTY_PREDICT_STATS=1 (live, default 0): how often a lookup finds the same program as the
	// previous lookup of its stage (a candidate for evaluating the resources ahead from the
	// upcoming user data), and how often the user data is the same too.
	struct PredictStage {
		const SourceEntry*    last = nullptr;
		std::vector<uint32_t> user_data;
		uint64_t              same_program = 0, same_inputs = 0, other = 0;
	};
	std::array<PredictStage, 3>           predict_stages;
	std::chrono::steady_clock::time_point predict_report = std::chrono::steady_clock::now();

	void NotePredictability(ShaderType stage, const SourceEntry* entry,
	                        std::span<const uint32_t> user_data) {
		static auto& enabled = Common::LiveSwitches::Get("KYTY_PREDICT_STATS", 0);
		if (enabled.load(std::memory_order_relaxed) == 0 || entry == nullptr) {
			return;
		}
		const size_t index = stage == ShaderType::Pixel     ? 1u
		                     : stage == ShaderType::Compute ? 2u
		                                                    : 0u;
		auto&        s     = predict_stages[index];
		if (s.last == entry) {
			const bool same = std::ranges::equal(s.user_data, user_data);
			(same ? s.same_inputs : s.same_program)++;
		} else {
			s.other++;
		}
		s.last = entry;
		s.user_data.assign(user_data.begin(), user_data.end());
		const auto now = std::chrono::steady_clock::now();
		if (now - predict_report < std::chrono::seconds(5)) {
			return;
		}
		predict_report                       = now;
		static constexpr const char* Names[] = {"vertex", "pixel", "compute"};
		::printf("Predictability (5 s):");
		for (size_t i = 0; i < predict_stages.size(); i++) {
			auto&      p     = predict_stages[i];
			const auto total = p.same_program + p.same_inputs + p.other;
			::printf(" %s %" PRIu64 ": same program %.1f%% (same inputs %.1f%%)", Names[i], total,
			         total ? 100.0 * static_cast<double>(p.same_program + p.same_inputs) /
			                     static_cast<double>(total)
			               : 0.0,
			         total ? 100.0 * static_cast<double>(p.same_inputs) / static_cast<double>(total)
			               : 0.0);
			p.same_program = p.same_inputs = p.other = 0;
		}
		::printf("\n");
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			PipelineKeyHash::Mix(hash, key.function_size);
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		FrameCosts::Timer cost_timer {FrameCosts::g_shader_us, FrameCosts::g_shader_count};
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		Prewarm::NoteSpirv(module, result.spirv);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module,
		                       .spirv_words = static_cast<uint64_t>(result.spirv.size())},
		};
	}

	// KYTY_ASYNC_PS: background translation of new pixel programs (see the comment above the struct).
	struct AsyncTranslateJob {
		std::vector<uint32_t>               code;
		std::vector<uint32_t>               user_data;
		ShaderPixelInputInfo                pixel_info {};
		ShaderRecompiler::CompileOptions    options;
		ShaderRecompiler::TranslateResult   result;
		std::atomic<int>                    state {0}; // 0 queued, 1 running, 2 done
		uint64_t                            work_us = 0;
	};
	std::unordered_map<ProgramKey, std::shared_ptr<AsyncTranslateJob>, ProgramKeyHash> async_jobs;
	std::unordered_set<ProgramKey, ProgramKeyHash>              async_sync_only;
	std::mutex                                                  async_mutex;
	std::condition_variable                                     async_wake;
	std::condition_variable                                     async_done;
	std::deque<std::shared_ptr<AsyncTranslateJob>>              async_queue;
	std::vector<std::thread>                                    async_threads;
	bool                                                        async_stop = false;
	std::atomic<uint64_t> async_worker_us {0}, async_completed {0};
	uint64_t async_submitted = 0, async_skipped_draws = 0, async_waits = 0, async_wait_us = 0,
	         async_inline = 0, async_fallbacks = 0, async_used = 0, async_unskippable = 0;
	std::chrono::steady_clock::time_point async_report = std::chrono::steady_clock::now();

	static bool AsyncPsEnabled() {
		static auto& on = Common::LiveSwitches::Get("KYTY_ASYNC_PS", 1);
		return on.load(std::memory_order_relaxed) != 0;
	}
	static void RunAsyncJob(AsyncTranslateJob& job) {
		const auto begin = std::chrono::steady_clock::now();
		job.result       = ShaderRecompiler::TranslateProgram(job.code, job.options);
		job.work_us      = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                         std::chrono::steady_clock::now() - begin)
		                                         .count());
	}
	void AsyncWorker() {
		for (;;) {
			std::shared_ptr<AsyncTranslateJob> job;
			{
				std::unique_lock lock(async_mutex);
				async_wake.wait(lock, [&] { return async_stop || !async_queue.empty(); });
				if (async_stop) {
					return;
				}
				job = std::move(async_queue.front());
				async_queue.pop_front();
			}
			int expected = 0;
			if (!job->state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
				continue; // Thread_Gpu took it inline
			}
			RunAsyncJob(*job);
			async_worker_us.fetch_add(job->work_us, std::memory_order_relaxed);
			async_completed.fetch_add(1, std::memory_order_relaxed);
			{
				std::lock_guard lock(async_mutex);
				job->state.store(2, std::memory_order_release);
			}
			async_done.notify_all();
		}
	}
	void StartAsyncWorkers() {
		if (!async_threads.empty()) {
			return;
		}
		unsigned threads = 3;
		if (const char* value = std::getenv("KYTY_ASYNC_PS_THREADS"); value != nullptr && *value != '\0') {
			threads = std::clamp(static_cast<unsigned>(std::strtoul(value, nullptr, 10)), 1u, 8u);
		}
		for (unsigned i = 0; i < threads; i++) {
			async_threads.emplace_back([this] { AsyncWorker(); });
		}
		::printf("Async PS translate: %u worker threads (KYTY_ASYNC_PS_THREADS)\n", threads);
	}
	void StopAsyncWorkers() {
		{
			std::lock_guard lock(async_mutex);
			async_stop = true;
		}
		async_wake.notify_all();
		for (auto& thread: async_threads) {
			thread.join();
		}
		async_threads.clear();
	}
	// Waits for a job, or runs it here when no worker has started it.
	void FinishAsyncJob(AsyncTranslateJob& job) {
		int expected = 0;
		if (job.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			RunAsyncJob(job);
			job.state.store(2, std::memory_order_release);
			async_inline++;
			return;
		}
		if (job.state.load(std::memory_order_acquire) == 2) {
			return;
		}
		const auto begin = std::chrono::steady_clock::now();
		{
			std::unique_lock lock(async_mutex);
			async_done.wait(lock, [&] { return job.state.load(std::memory_order_acquire) == 2; });
		}
		async_waits++;
		async_wait_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
		                                           std::chrono::steady_clock::now() - begin)
		                                           .count());
	}
	void MaybeReportAsync() {
		const auto now = std::chrono::steady_clock::now();
		if (now - async_report < std::chrono::seconds(5)) {
			return;
		}
		async_report = now;
		if (async_submitted == 0 && async_skipped_draws == 0 && async_waits == 0 && async_unskippable == 0) {
			return;
		}
		::printf("Async PS translate (5 s, KYTY_ASYNC_PS=%d): %llu submitted, %llu translated on workers in %.0f ms "
		         "(off Thread_Gpu), %llu used; %llu draws skipped while translating; %llu waits for a job (%.1f ms), "
		         "%llu run inline, %llu fell back to the inline path; %llu new programs translated inline (draw writes depth/stencil); %zu pending\n",
		         AsyncPsEnabled() ? 1 : 0, static_cast<unsigned long long>(async_submitted),
		         static_cast<unsigned long long>(async_completed.exchange(0, std::memory_order_relaxed)),
		         static_cast<double>(async_worker_us.exchange(0, std::memory_order_relaxed)) / 1000.0,
		         static_cast<unsigned long long>(async_used), static_cast<unsigned long long>(async_skipped_draws),
		         static_cast<unsigned long long>(async_waits), static_cast<double>(async_wait_us) / 1000.0,
		         static_cast<unsigned long long>(async_inline), static_cast<unsigned long long>(async_fallbacks),
		         static_cast<unsigned long long>(async_unskippable), async_jobs.size());
		async_submitted = async_skipped_draws = async_waits = async_wait_us = async_inline = async_fallbacks =
		    async_used = async_unskippable = 0;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		if (SkipShaderRequested(params.hash) || LiveSkipRequested(params.hash, stage)) {
			return ShaderProgram {};
		}
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			MaybeReportAsync();
		}
		BufferCache::SetReadbackStatsShader(params.hash);
		DrawDiff::t_shader = params.hash;
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		std::optional<PhaseStats::Scope> prog_phase {std::in_place, PhaseStats::PhProgKey};
		{
			KYTY_PROFILER_BLOCK("ProgramCache::BuildKey");
			lookup_key.stage           = stage;
			lookup_key.hash            = params.hash;
			lookup_key.user_data_count = params.user_data_count;
			lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
			BuildStageStaticKey(input_info, lookup_key.static_state);
			// Research: a program with inlined calls is valid only for the same call targets.
			if (const auto found = call_targets.find(params.hash); found != call_targets.end()) {
				for (const auto index: found->second) {
					lookup_key.static_state.push_back(index < user_data.size() ? user_data[index]
					                                                           : 0u);
				}
			}
			// Research: ShaderFunctions expands calls first; the calls it refuses reach the
			// recompiler's own inliner.
			lookup_key.function_code.reset();
			lookup_key.function_size    = 0;
			lookup_key.function_hash_lo = lookup_key.function_hash_hi = 0;
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				PhaseStats::Scope expand_phase {PhaseStats::PhKeyExpand};
				ExpandShaderFunctions(params, user_data, input_info.wave_size, lookup_key);
			}
		}
		auto entry = programs.end();
		prog_phase.reset();
		prog_phase.emplace(PhaseStats::PhProgLookup);
		{
			KYTY_PROFILER_BLOCK("ProgramCache::Lookup");
			// KYTY_LOOKUP_CACHE (live, default 1): the last key found per stage and its entry. A
			// hit needs the full key equal (stage, hash, sizes, static register state, expanded
			// code), the same bucket count (iterators survive) and no new unsupported key, so it
			// returns exactly what find() would; it only saves hashing the key.
			static auto& lookup_cache_switch = Common::LiveSwitches::Get("KYTY_LOOKUP_CACHE", 1);
			auto&        last = last_lookup[static_cast<size_t>(stage) % last_lookup.size()];
			const bool   use_cache = lookup_cache_switch.load(std::memory_order_relaxed) != 0;
			if (use_cache && last.valid && last.buckets == programs.bucket_count() &&
			    last.unsupported_size == unsupported.size() && last.key == lookup_key) {
				entry = last.entry;
				lookup_hits++;
			} else {
				if (unsupported.contains(lookup_key)) {
					return ShaderProgram {};
				}
				entry = programs.find(lookup_key);
				lookup_misses++;
				last.valid = use_cache && entry != programs.end();
				if (last.valid) {
					last.key              = lookup_key;
					last.entry            = entry;
					last.buckets          = programs.bucket_count();
					last.unsupported_size = unsupported.size();
				}
			}
		}
		NotePredictability(stage, entry != programs.end() ? &entry->second : nullptr, user_data);
		ShaderReadChunks                 read_chunks(ShaderReadChunks::Mode());
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryRaw,
		    .userdata                   = &read_chunks,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .float_image_atomics        = Config::FloatImageAtomicsEnabled(),
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			for (uint32_t axis = 0; axis < 3u; axis++) {
				runtime.workgroup_count[axis] = input_info.dispatch_groups[axis];
				runtime.workgroup_size[axis]  = input_info.threads_num[axis];
			}
		}
		if (entry != programs.end()) {
			bool materialized = false;
			prog_phase.reset();
			prog_phase.emplace(PhaseStats::PhProgMaterialize);
			{
				KYTY_PROFILER_BLOCK("ProgramCache::MaterializeResources");
				materialized = Materialize(entry->second, runtime, read_chunks);
				if (ShaderRecompiler::IR::SrtNativeStats stats;
				    ShaderRecompiler::IR::TakeSrtNativeReport(stats)) {
					::printf("SRT native: %u plans compiled, %u failed, %" PRIu64 " KB of code, %" PRIu64
					     " instructions, %" PRIu64 " through the interpreter\n",
					     stats.plans, stats.failed, stats.bytes / 1024u, stats.instructions,
					     stats.interpreted);
				}
			}
			if (!materialized) {
				// A descriptor source that cannot be read right now (memory the guest has not
				// mapped or filled yet) skips this draw rather than the session; the next one
				// re-evaluates from scratch.
				if (!ShaderFailureNonFatal()) {
					EXIT("shader resource materialization failed\n");
				}
				DrawDiff::NoteSkip(params.hash, "resource materialization failed");
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 16) {
					LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
					     ": resource materialization failed (materialization line %d)\n",
					     static_cast<uint32_t>(stage), params.hash,
					     ShaderRecompiler::IR::LastIndirectImageFailureLine());
				}
				return ShaderProgram {};
			}
			auto permutation = entry->second.permutations.end();
			prog_phase.reset();
			prog_phase.emplace(PhaseStats::PhProgFind);
			{
				KYTY_PROFILER_BLOCK("ProgramCache::FindPermutation");
				permutation = std::ranges::find_if(
				    entry->second.permutations, [&](const Permutation& candidate) {
					    const auto& layout = candidate.program.bindings;
					    return layout.push_data_start_dword ==
					               ShaderRecompiler::IR::PushData::StartFor(
					                   push_data_cursor, layout.ShaderDataDwords()) &&
					           candidate.specialization == entry->second.specialization;
				    });
			}
			if (permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		// KYTY_ASYNC_PS: a brand-new pixel program is translated on a worker (see the struct comment).
		std::shared_ptr<AsyncTranslateJob> async_job;
		if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			if (entry == programs.end()) {
				auto found = async_jobs.find(lookup_key);
				if (found != async_jobs.end()) {
					async_job = found->second;
				} else if (AsyncPsEnabled() && t_async_ps_may_skip && params.back_code.empty() &&
				           !call_targets.contains(params.hash) && !async_sync_only.contains(lookup_key)) {
					StartAsyncWorkers();
					async_job            = std::make_shared<AsyncTranslateJob>();
					async_job->code.assign(params.code.begin(), params.code.end());
					async_job->user_data.assign(user_data.begin(), user_data.end());
					async_job->pixel_info       = input_info;
					async_job->pixel_info.stage = {};
					auto& o            = async_job->options;
					o.stage            = stage;
					o.shader_hash      = params.hash;
					o.user_data        = async_job->user_data;
					o.wave_size        = input_info.wave_size;
					o.dump_ir          = false;
					o.early_dump       = false;
					o.dump_label       = "ShaderRecompiler PS (async)";
					o.non_fatal        = ShaderFailureNonFatal();
					o.bindless_images  = bindless_images;
					o.input_info.pixel = &async_job->pixel_info;
					// No read_code: subroutine calls stay unsupported here and are redone inline.
					async_jobs.emplace(lookup_key, async_job);
					{
						std::lock_guard lock(async_mutex);
						async_queue.push_back(async_job);
					}
					async_wake.notify_one();
					async_submitted++;
				}
				if (async_job == nullptr && AsyncPsEnabled() && !t_async_ps_may_skip) {
					async_unskippable++;
				}
				if (async_job != nullptr) {
					if (async_job->state.load(std::memory_order_acquire) != 2 && t_async_ps_may_skip &&
					    AsyncPsEnabled()) {
						async_skipped_draws++;
						t_async_ps_pending = true;
						return ShaderProgram {};
					}
					FinishAsyncJob(*async_job);
					async_jobs.erase(lookup_key);
					if (async_job->result.unsupported || !async_job->result.call_target_user_data.empty()) {
						async_fallbacks++;
						async_sync_only.insert(lookup_key);
						async_job.reset();
					}
				}
			}
		}

		// A program (or permutation) this session has not built yet: recompiled on the GPU
		// thread. Wolverine run 54: the 0.3-0.7 s frames while turning were spent here.
		prog_phase.reset();
		prog_phase.emplace(PhaseStats::PhProgRecompile);
		KYTY_PROFILER_BLOCK("ProgramCache::Recompile", profiler::colors::RedA100);
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		options.non_fatal = ShaderFailureNonFatal();
		options.bindless_images = bindless_images;
		options.read_code       = ReadShaderCode;
		const auto compile_code = lookup_key.function_code == nullptr || lookup_key.function_code->empty()
		                              ? params.code
		                              : std::span<const uint32_t>(*lookup_key.function_code);
		auto translated = async_job != nullptr ? std::move(async_job->result)
		                                       : ShaderRecompiler::TranslateProgram(compile_code, options);
		if (async_job != nullptr) {
			async_used++;
		}
		if (!translated.unsupported && !translated.call_target_user_data.empty() &&
		    call_targets.try_emplace(params.hash, translated.call_target_user_data).second) {
			for (const auto index: translated.call_target_user_data) {
				lookup_key.static_state.push_back(index < user_data.size() ? user_data[index] : 0u);
			}
		}
		if (translated.unsupported) {
			// Remember the refusal: a skipped shader is dispatched again every frame, and
			// re-deriving the same answer costs as much as a compile each time.
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if (!shader_clock && UsesShaderClock(translated.program)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("S_MEMREALTIME needs shaderDeviceClock\n");
			}
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 16) {
				LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
				     ": S_MEMREALTIME needs shaderDeviceClock\n",
				     static_cast<uint32_t>(stage), params.hash);
			}
			unsupported.insert(lookup_key);
			return ShaderProgram {};
		}
		if (entry == programs.end()) {
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)).first;
			if (!ShaderRecompiler::IR::MaterializeResources(
			        entry->second.resource_plan, runtime, entry->second.resources,
			        entry->second.specialization)) {
				if (!ShaderFailureNonFatal()) {
					EXIT("shader resource materialization failed\n");
				}
				DrawDiff::NoteSkip(params.hash, "resource materialization failed");
				static std::atomic<uint32_t> reported = 0;
				if (reported.fetch_add(1) < 16) {
					LOGF("ProgramCache: skipping stage %u hash=0x%016" PRIx64
					     ": resource materialization failed on first use (materialization line %d)\n",
					     static_cast<uint32_t>(stage), params.hash,
					     ShaderRecompiler::IR::LastIndirectImageFailureLine());
				}
				return ShaderProgram {};
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor));
		if (entry->second.permutations.size() > 1u) {
			entry->second.permutations.back().handle.repeat_permutation = true;
		}
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	ProgramCache(vk::Device device, bool shader_clock, bool bindless_images)
	    : device(device), shader_clock(shader_clock), bindless_images(bindless_images) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		StopAsyncWorkers();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::chrono::steady_clock::time_point expand_report = std::chrono::steady_clock::now();
	void ReportExpand(bool fast) {
		const auto now = std::chrono::steady_clock::now();
		if (now - expand_report < std::chrono::seconds(5)) {
			return;
		}
		expand_report = now;
		const auto st = function_expander.TakeStats();
		::printf("Shader expand (5 s, KYTY_EXPAND_CACHE=%d): %llu compute lookups, %llu without calls; code compared %llu, "
		         "skipped by hash %llu; call resolutions run %llu, reused %llu (%llu table words re-read); expansions "
		         "reused %llu (%llu callee words re-read), built %llu (time: key.expand in Phases)\n",
		         fast ? 1 : 0, static_cast<unsigned long long>(st.calls), static_cast<unsigned long long>(st.no_calls),
		         static_cast<unsigned long long>(st.code_compares), static_cast<unsigned long long>(st.code_hash_skips),
		         static_cast<unsigned long long>(st.resolve_full), static_cast<unsigned long long>(st.resolve_reused),
		         static_cast<unsigned long long>(st.resolve_words_checked),
		         static_cast<unsigned long long>(st.expansions_reused),
		         static_cast<unsigned long long>(st.callee_words_checked),
		         static_cast<unsigned long long>(st.expansions_built));
		std::fflush(stdout);
	}

	// Shader function expansion decodes the shader and every callee, and a shader with table
	// calls can be dispatched hundreds of times a frame; the expander keeps what it can reuse.
	// wolv-draw-1111: KYTY_EXPAND_CACHE (live, default 1; 0 = old: every dispatch compares the
	// whole shader code with the cached copy and resolves its call targets again, decoding
	// register writes through enum names). 1: the shader's registered hash stands for its code
	// (as the program key already does), and a call-target resolution is reused while the
	// user-data words it consulted and every guest word it read are unchanged. The callee code is
	// still re-read and compared every time. Log: "Shader expand (5 s)".
	void ExpandShaderFunctions(const ShaderParams& params, std::span<const uint32_t> user_data,
	                           uint32_t wave_size, ProgramKey& key) {
		static auto& expand_cache = Common::LiveSwitches::Get("KYTY_EXPAND_CACHE", 1);
		const bool   fast         = expand_cache.load(std::memory_order_relaxed) != 0;
		std::string reason;
		ShaderRecompiler::Decoder::ShaderFunctionExpander::Result result;
		const bool ok = function_expander.ExpandShared(
		    params.code, params.Base(), user_data,
		    [](uint64_t address, std::span<uint32_t> words) {
			    return ReadShaderGuestMemoryRaw(nullptr, address, words);
		    },
		    wave_size, fast ? params.hash : 0, fast, result, reason);
		if (result.code != nullptr && !result.code->empty()) {
			key.function_code    = std::move(result.code);
			key.function_size    = static_cast<uint32_t>(key.function_code->size());
			key.function_hash_lo = result.hash_lo;
			key.function_hash_hi = result.hash_hi;
		}
		ReportExpand(fast);
		if (!ok) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1) < 16) {
				::printf("Shader function expansion hash=%016" PRIx64 ": %s\n", params.hash,
				         reason.c_str());
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	std::unordered_set<ProgramKey, ProgramKeyHash>              unsupported;
	struct LastLookup {
		bool                                                              valid = false;
		ProgramKey                                                        key;
		std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash>::iterator entry;
		size_t                                                            buckets          = 0;
		size_t                                                            unsupported_size = 0;
	};
	std::array<LastLookup, 16> last_lookup {};
public:
	uint64_t lookup_hits = 0, lookup_misses = 0;
	ShaderRecompiler::Decoder::ShaderFunctionExpander          function_expander;
	ProgramKey                                                  lookup_key;
	// Research: per shader hash, the user-data dwords holding its inlined call targets.
	std::unordered_map<uint64_t, std::vector<uint32_t>>         call_targets;
	// KYTY_RESOURCE_MEMO bookkeeping (GPU thread).
	uint64_t                              memo_clock  = 0;
	uint64_t                              memo_hits   = 0;
	uint64_t                              memo_misses = 0;
	std::vector<uint32_t>                 memo_words;
	std::chrono::steady_clock::time_point memo_report = std::chrono::steady_clock::now();
	vk::Device                                                  device;
	bool                                                        shader_clock = false;
	bool                                                        bindless_images = false;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics),
      m_program_cache(std::make_unique<ProgramCache>(
          graphics.device, graphics.shader_device_clock_enabled,
          graphics.bindless_enabled && Config::BindlessImagesEnabled())) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	if (g_async_pipelines) {
		// KYTY_PIPELINE_WORKERS=<n>: pipeline compile threads (1-16); default as before (a quarter of the CPUs, 1-4).
		auto threads = std::clamp(std::thread::hardware_concurrency() / 4u, 1u, 4u);
		if (const char* value = std::getenv("KYTY_PIPELINE_WORKERS"); value != nullptr && *value != '\0')
			threads = std::clamp(static_cast<unsigned>(std::strtoul(value, nullptr, 10)), 1u, 16u);
		m_compiler         = std::make_unique<PipelineCompiler>(threads);
		PipelineCacheLog("Vulkan pipelines: compiled on {} worker threads (wait {} ms)", threads,
		                 g_pipeline_wait.count());
	}
}

void PipelineCache::StartPrewarm() {
	if (!m_prewarm_started) {
		m_prewarm_started = true;
		if (Prewarm::EnsureStarted(m_graphics, PipelineCacheTitleId(), m_driver_cache)) {
			Save();
			Prewarm::Shutdown();
			std::printf("Prewarm: KYTY_PREWARM_ONLY=1: precompile finished, driver cache saved, exiting\n");
			std::fflush(stdout);
			std::_Exit(0);
		}
	}
}

PipelineCache::~PipelineCache() {
	StopAsyncCompute();
	Prewarm::Shutdown();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->pending && pipeline->pending->done.load(std::memory_order_acquire)) {
				m_graphics.device.destroyPipeline(pipeline->pending->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

struct DriverCacheAutosave {
	std::mutex              mutex;
	std::condition_variable wake;
	bool                    stop       = false;
	uint64_t                saved_size = 0;
	std::thread             thread;
};

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	const std::string_view git_hash     = KYTY_GIT_HASH;
	const std::string_view git_revision = KYTY_GIT_REVISION;
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty")) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() ||
			    !DriverCacheSignatureMatches(cached_signature, signature) ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				const bool short_read = signature_read != cached_signature.size() ||
				                        hash_read != sizeof(payload_hash) || payload_read != initial_data.size();
				initial_data.clear();
				// Which part differed (wolverine-turbo3-1025): "KytyPC1:<revision>:<vendor>:<device>:<driver>:<uuid>".
				std::string reason;
				if (short_read) {
					reason = "short read";
				} else if (!DriverCacheSignatureMatches(cached_signature, signature)) {
					reason = "signature differs: cached '" +
					         std::string(cached_signature.substr(0, cached_signature.find('\n'))) +
					         "', this build and driver '" + signature.substr(0, signature.find('\n')) + "'";
				} else {
					reason = "payload hash mismatch (file damaged or written by an interrupted save)";
				}
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch: {})",
				    path, reason);
			} else if (cached_signature != signature) {
				PipelineCacheLog("Vulkan pipeline cache: keeping {} from another emulator revision",
				                 path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	// KYTY_PIPELINE_CACHE_AUTOSAVE_S=N (default 120, 0 = off; wolverine-turbo3-1025): every N s, the driver cache
	// is written in the background when it grew by 16 MiB or more since the last save. Before, it was written only
	// at a clean exit: the 2026-10-25 TURBO run started cold ("invalidating"), was closed from the console and
	// saved nothing, so its slow phase compiled pipelines on the fly (pipeline waits up to 2.8 s per 5 s, against
	// 0.1-0.6 s in the user's warm 10-20 run) and the next run would start cold again.
	const char* autosave_env = std::getenv("KYTY_PIPELINE_CACHE_AUTOSAVE_S");
	const long  autosave_s   = autosave_env != nullptr ? std::strtol(autosave_env, nullptr, 10) : 120;
	if (autosave_s > 0) {
		m_autosave             = std::make_unique<DriverCacheAutosave>();
		m_autosave->saved_size = initial_data.size();
		m_autosave->thread     = std::thread([this, autosave_s] {
            std::unique_lock lock(m_autosave->mutex);
            while (!m_autosave->wake.wait_for(lock, std::chrono::seconds(autosave_s),
                                              [this] { return m_autosave->stop; })) {
                lock.unlock();
                (void)WriteDriverCache(16ull << 20u);
                lock.lock();
            }
		});
	}
}

void PipelineCache::Save() {
	// The autosave thread stops first: it must not read the driver cache once it is destroyed below.
	if (m_autosave != nullptr) {
		{
			std::lock_guard lock(m_autosave->mutex);
			m_autosave->stop = true;
		}
		m_autosave->wake.notify_all();
		if (m_autosave->thread.joinable()) {
			m_autosave->thread.join();
		}
	}
	// No worker may use the driver cache once it is destroyed below; later pipelines are
	// compiled on the GPU thread (FinishPending).
	if (m_compiler != nullptr) {
		m_compiler->Stop();
	}
	if (m_driver_cache == nullptr) {
		return;
	}
	if (WriteDriverCache(0)) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
		m_driver_cache = nullptr;
	}
}

// vkGetPipelineCacheData does not need the cache externally synchronized, so this also runs on the autosave
// thread while workers compile into the cache.
bool PipelineCache::WriteDriverCache(uint64_t min_growth) {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		if (min_growth != 0 && m_autosave != nullptr && size < m_autosave->saved_size + min_growth) {
			return true;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	if (m_autosave != nullptr) {
		m_autosave->saved_size = payload.size();
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}{}", payload.size(),
	                 Common::PathToString(m_driver_cache_path), min_growth != 0 ? " (autosave)" : "");
	return true;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		if (!PrepareTessellationPrograms(vertex_regs, context, vertex_info, vertex_params)) {
			if (!ShaderFailureNonFatal()) {
				EXIT("unsupported tessellation programs\n");
			}
			return {};
		}
	} else {
		KYTY_PROFILER_BLOCK("PipelineCache::PrepareProgram(VS)");
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		// A subgroup with more threads than a mesh workgroup may have (NVIDIA: 128) runs its
		// waves in passes; see EmitMeshEntryPoint.
		mesh.passes = mesh.PassesFor(
		    std::min(limits.maxMeshWorkGroupInvocations, limits.maxMeshWorkGroupSize[0]));
		if (mesh.passes == 0 || mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			// Skipped like the draws of a shader that gives up, and reported once per shader.
			static std::mutex                   logged_mutex;
			static std::unordered_set<uint64_t> logged;
			std::lock_guard                     lock(logged_mutex);
			if (logged.insert(vertex_params[0].hash).second) {
				LOGF("mesh shader 0x%016" PRIx64 " exceeds host limits, draws skipped: wave%u "
				     "threads=%u vertices=%u primitives=%u LDS=%u\n",
				     vertex_params[0].hash, mesh.wave_size, mesh.HostThreads(), mesh.max_vertices,
				     mesh.max_primitives, mesh.lds_size_dwords);
			}
			return {};
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		KYTY_PROFILER_BLOCK("PipelineCache::PrepareProgram(PS)");
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		// SPI_SHADER_COL_FORMAT describes export packing, not the attachment numeric type.
		// In particular, 32-bit exports can carry raw integer material data.
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr == 0 ||
			    render_target_mask_slot(context.GetRenderTargetMask(), slot) == 0) {
				continue;
			}
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << slot;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << slot;
			}
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask =
			    (pixel_info.target_uint_mask & ~2u) | ((pixel_info.target_uint_mask & 1u) << 1u);
			pixel_info.target_sint_mask =
			    (pixel_info.target_sint_mask & ~2u) | ((pixel_info.target_sint_mask & 1u) << 1u);
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; }) &&
		           ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0]) ==
		               BlendMappingSupport::SourceAlpha) {
			// Preserve logical alpha when the export mapping moves it.
			pixel_info.alpha_blend_source_remap = true;
			pixel_info.dual_source_blending     = true;
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = {};
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	t_async_ps_pending = false;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
		if (!result.pixel && t_async_ps_pending) {
			// KYTY_ASYNC_PS: skipped while its pixel program translates; no VS lookup (it would
			// build a permutation for a push-data layout without the pixel stage).
			t_async_ps_pending = false;
			return {};
		}
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		if (!result.vertex[i]) {
			return {};
		}
	}
	// The vertex lookups above ran last and left their hash in t_shader: a draw's bindings and
	// targets are attributed to its pixel shader (KYTY_DUMP_SHADER, DrawDiff), not to its VS.
	if (pixel_active) {
		DrawDiff::t_shader = pixel_params.hash;
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	// Use one effective size for the cache key, LDS declaration, and access bounds.
	const auto max_lds_dwords =
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize / 4u;
	if (input_info.lds_size_dwords > max_lds_dwords) {
		static std::atomic_bool warned = false;
		if (!warned.exchange(true, std::memory_order_relaxed)) {
			PipelineCacheLog("GPU warning: game compute shader requests {} bytes of LDS, but "
			                 "the Vulkan device limit is {} bytes. Clamping LDS; rendering may "
			                 "be incorrect.",
			                 input_info.lds_size_dwords * 4u, max_lds_dwords * 4u);
		}
	}
	input_info.lds_size_dwords = std::min(input_info.lds_size_dwords, max_lds_dwords);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	auto* pipeline = TryGetGraphicsPipeline(colors, depth, vertex_info, command, ps_input_info,
	                                        topology, primitive_restart_enable, programs, false);
	EXIT_IF(pipeline == nullptr);
	return *pipeline;
}

bool PipelineCache::FinishPending(Pipeline& pipeline) {
	auto& job = *pipeline.pending;
	if (!job.done.load(std::memory_order_acquire)) {
		if (m_compiler != nullptr && !m_compiler->Stopped()) {
			return false;
		}
		// The workers stopped before this job started: compile it here.
		job.result = CreateGraphicsPipeline(*job.build, m_driver_cache, &job.pipeline);
		job.done.store(true, std::memory_order_release);
	}
	EXIT_NOT_IMPLEMENTED(job.result != vk::Result::eSuccess || job.pipeline == nullptr);
	pipeline.pipeline = job.pipeline;
	pipeline.pending.reset();
	return true;
}

PipelineCache::Pipeline* PipelineCache::TryGetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs, bool may_defer) {
	StartPrewarm();
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		const bool alpha_remap =
		    slot == 0 && ps_input_info != nullptr && ps_input_info->alpha_blend_source_remap;
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && !alpha_remap &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (alpha_remap) {
			static_params.blend_alpha_source_remap = true;
		}
		if (static_params.blend_enable[slot]) {
			static_params.color_srcblend[slot]       = bc.color_srcblend;
			static_params.color_comb_fcn[slot]       = bc.color_comb_fcn;
			static_params.color_destblend[slot]      = bc.color_destblend;
			static_params.separate_alpha_blend[slot] = bc.separate_alpha_blend;
			if (bc.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = bc.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = bc.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = bc.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	const auto defer = [this] {
		FrameCosts::g_draws_deferred.fetch_add(1, std::memory_order_relaxed);
		if (++m_deferred_draws % 256 == 1) {
			LOGF("PipelineCache: %" PRIu64 " draws skipped while their pipeline compiled in the "
			     "background\n",
			     m_deferred_draws);
		}
		return nullptr;
	};
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		auto& found = *iter->second;
		if (found.pending && !FinishPending(found)) {
			return defer();
		}
		return &found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	auto build = PrepareGraphicsPipeline(m_graphics, *cached, rendering, key.vertex_input,
	                                     vertex_info, ps_input_info, programs, static_params);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);
	if (may_defer && m_compiler != nullptr && !m_compiler->Stopped()) {
		// A pipeline the driver has cached finishes within the wait; a new one is compiled in
		// the background, and its draws are skipped until it is ready.
		auto job          = std::make_shared<PendingGraphicsPipeline>();
		job->build        = std::move(build);
		job->driver_cache = m_driver_cache;
		cached->pending   = job;
		m_compiler->Submit(job);
		{
			FrameCosts::Timer cost_timer {FrameCosts::g_pipeline_wait_us, FrameCosts::g_pipeline_waits};
			(void)m_compiler->Wait(*job, g_pipeline_wait);
		}
		auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
		EXIT_IF(!inserted);
		auto& pipeline = *iter->second;
		if (!FinishPending(pipeline)) {
			return defer();
		}
		LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
		return &pipeline;
	}
	vk::Result result = vk::Result::eSuccess;
	{
		// Draws that may not be skipped (they write memory, depth or stencil): compiled inline.
		FrameCosts::Timer cost_timer {FrameCosts::g_critical_wait_us, FrameCosts::g_critical_waits};
		result = CreateGraphicsPipeline(*build, m_driver_cache, &cached->pipeline);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	StartPrewarm();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	{
		FrameCosts::Timer cost_timer {FrameCosts::g_compute_sync_us, FrameCosts::g_compute_syncs};
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
// Ported from yotei-movie5-1027 (e4ddbca9, bd048103, 5a917b43), Wolverine defaults:
// KYTY_ASYNC_COMPUTE_WORDS=<n> (default 0 = off): compute pipelines whose SPIR-V has more words are
//   created by background threads (KYTY_ASYNC_COMPUTE_THREADS, default 4).
// KYTY_ASYNC_REPEAT_WORDS=<n> (default 20000, 0 = off): the same for a repeated permutation of a
//   guest shader (another permutation already existed) over n words. Ghost of Yotei: one culling CS
//   got a new ~90K-word permutation almost every frame, a ~1.3 s synchronous compile each.
// KYTY_ASYNC_COMPUTE_SKIP=1 (default 0): a dispatch whose pipeline is still compiling is skipped.
//   Off by default because skipped producer passes (vertex/index/matrix data, culling lists) leave
//   garbage buffers: exploded geometry and missing objects in Yotei. With 0 the background compile
//   is waited for, so the dispatch is never lost.
// Background compiles happen only when they can pay off: with KYTY_ASYNC_COMPUTE_SKIP=0 a new
// pipeline compiles inline (waiting for a worker would only add a thread hop).
struct PipelineCache::AsyncCompute {
	struct Job {
		uint64_t                  id = 0;
		std::unique_ptr<Pipeline> pipeline;
		ComputePipelineJob        job;
	};

	std::mutex                                              mutex;
	std::condition_variable                                 cv;
	std::condition_variable                                 done_cv;
	std::deque<Job>                                         queue;
	std::unordered_set<uint64_t>                            pending;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> done;
	std::vector<std::thread>                                threads;
	bool                                                    stop = false;
};

static uint64_t EnvU64(const char* name, uint64_t fallback) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 10) : fallback;
}

static bool ComputeSkipAllowed() {
	static const bool allowed = EnvU64("KYTY_ASYNC_COMPUTE_SKIP", 0) == 1;
	return allowed;
}

void PipelineCache::StopAsyncCompute() {
	if (m_async_compute == nullptr) {
		return;
	}
	auto& async = *m_async_compute;
	{
		std::lock_guard lock(async.mutex);
		async.stop = true;
		for (auto& job: async.queue) {
			async.pending.erase(job.id);
			m_graphics.device.destroyPipelineLayout(job.pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(job.pipeline->descriptor_set_layout, nullptr);
		}
		async.queue.clear();
	}
	async.cv.notify_all();
	for (auto& t: async.threads) {
		t.join();
	}
	for (auto& [id, pipeline]: async.done) {
		m_compute_pipelines.emplace(id, std::move(pipeline));
	}
	async.done.clear();
	m_async_compute.reset();
}

PipelineCache::Pipeline*
PipelineCache::TryGetComputePipeline(const ShaderComputeInputInfo& input_info,
                                     const ShaderProgram&          compute_program) {
	EXIT_IF(!compute_program);
	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return iter->second.get();
	}
	static const uint64_t words        = EnvU64("KYTY_ASYNC_COMPUTE_WORDS", 0);
	static const uint64_t repeat_words = EnvU64("KYTY_ASYNC_REPEAT_WORDS", 20000);

	if (m_async_compute != nullptr) {
		auto&                     async = *m_async_compute;
		std::unique_ptr<Pipeline> ready;
		{
			std::unique_lock lock(async.mutex);
			if (async.pending.contains(compute_program.id)) {
				if (ComputeSkipAllowed()) {
					FrameCosts::g_compute_skipped.fetch_add(1, std::memory_order_relaxed);
					return nullptr;
				}
				// Never skip (default): wait for the background compile.
				FrameCosts::Timer cost_timer {FrameCosts::g_compute_sync_us, FrameCosts::g_compute_syncs};
				async.done_cv.wait(lock, [&] { return !async.pending.contains(compute_program.id); });
			}
			if (auto it = async.done.find(compute_program.id); it != async.done.end()) {
				ready = std::move(it->second);
				async.done.erase(it);
			}
		}
		if (ready != nullptr) {
			auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(ready));
			EXIT_IF(!inserted);
			return iter->second.get();
		}
	}

	const bool background = ComputeSkipAllowed() &&
	                        ((words != 0 && compute_program.spirv_words > words) ||
	                         (repeat_words != 0 && compute_program.repeat_permutation &&
	                          compute_program.spirv_words > repeat_words));
	if (!background) {
		return &GetComputePipeline(input_info, compute_program);
	}

	StartPrewarm();
	auto job      = AsyncCompute::Job {};
	job.id        = compute_program.id;
	job.pipeline  = std::make_unique<Pipeline>();
	// Layouts and the prewarm lookup read the program: done here, only the driver compile moves.
	CreateComputeLayoutInternal(m_graphics, *job.pipeline, input_info, compute_program.module, job.job);
	if (job.job.module == nullptr) {
		auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(job.pipeline));
		EXIT_IF(!inserted);
		return iter->second.get();
	}

	if (m_async_compute == nullptr) {
		m_async_compute = std::make_unique<AsyncCompute>();
		const auto count =
		    static_cast<uint32_t>(std::clamp<uint64_t>(EnvU64("KYTY_ASYNC_COMPUTE_THREADS", 4), 1, 16));
		auto* async = m_async_compute.get();
		for (uint32_t i = 0; i < count; i++) {
			async->threads.emplace_back([this, async] {
				Prewarm::MarkAsyncThread();
				for (;;) {
					AsyncCompute::Job work;
					{
						std::unique_lock lock(async->mutex);
						async->cv.wait(lock, [async] { return async->stop || !async->queue.empty(); });
						if (async->stop) {
							return;
						}
						work = std::move(async->queue.front());
						async->queue.pop_front();
					}
					const auto start = std::chrono::steady_clock::now();
					CreateComputePipelineObject(m_graphics, *work.pipeline, work.job, m_driver_cache);
					const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					                    std::chrono::steady_clock::now() - start)
					                    .count();
					std::printf("Pipeline: async compute %016llx compiled in %lld ms\n",
					            static_cast<unsigned long long>(work.id), static_cast<long long>(ms));
					{
						std::lock_guard lock(async->mutex);
						async->done.emplace(work.id, std::move(work.pipeline));
						async->pending.erase(work.id);
					}
					async->done_cv.notify_all();
				}
			});
		}
		std::printf("Pipeline: async compute enabled, %u threads, SPIR-V > %llu words, repeat permutations > %llu words, skip=%d\n",
		            count, static_cast<unsigned long long>(words),
		            static_cast<unsigned long long>(repeat_words), ComputeSkipAllowed() ? 1 : 0);
	}
	std::printf("Pipeline: async compute %016llx queued (%llu words%s)\n",
	            static_cast<unsigned long long>(job.id),
	            static_cast<unsigned long long>(compute_program.spirv_words),
	            compute_program.repeat_permutation ? ", repeat permutation" : "");
	std::fflush(stdout);
	FrameCosts::g_compute_async_queued.fetch_add(1, std::memory_order_relaxed);
	FrameCosts::g_compute_skipped.fetch_add(1, std::memory_order_relaxed);
	{
		std::lock_guard lock(m_async_compute->mutex);
		m_async_compute->pending.insert(job.id);
		m_async_compute->queue.push_back(std::move(job));
	}
	m_async_compute->cv.notify_one();
	return nullptr;
}
} // namespace Libs::Graphics
