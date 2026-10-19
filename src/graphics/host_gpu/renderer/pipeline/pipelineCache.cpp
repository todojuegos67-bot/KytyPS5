#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/driverCachePolicy.h"
#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBinaries.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCacheBlobs.h"
#include "graphics/host_gpu/renderer/pipeline/shaderReadObserver.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include "resource-plan-serial.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"
#include "native-preparation-scratch.h"
#include "native-resource-state.h"
#include "shader-warmup-cache.h"
#include "scene-tags.h"
#include "live-census.h"
#include "live-counters.h"
#include "device-fault.h"
#include "local-platform.h"
#include "slow-log.h"
#include "startup-progress.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <fstream>
#include <functional>
#include <condition_variable>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

extern "C" {
// 1: find graphics pipelines through a block-hashed index of the full key.
volatile std::atomic<uint32_t> kyty_local_pipeline_index_mode {0};
// (performance-switches.h, KYTY_SPECULATE) Speculative translation records ahead on other threads.
extern volatile std::atomic_uint32_t kyty_local_speculate_mode;
}

namespace Libs::Graphics {

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

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties,
                                std::string_view binary_key) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// A keyed (local) cache spans emulator builds: the driver's own UUID and version below scope
	// its binaries, and it finds a pipeline by the pipeline's whole input.
	const auto revision = binary_key.empty() ? std::string(KYTY_GIT_REVISION) : fmt::format("local:{}", binary_key);
	return fmt::format("KytyPC1:{}:{:08x}:{:08x}:{:08x}:{}\n", revision,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

// Compiler inputs are portable across executable rebuilds; the schema version
// and device capabilities still constrain them. Driver binaries are scoped by the
// launcher's cache key. Every warm program is checked against live source.
std::string ShaderInputDeviceSignature(const vk::PhysicalDeviceProperties& properties) {
	const auto signature = DriverCacheSignature(properties, {});
	return signature.substr(signature.size() - (8 * 3 + VK_UUID_SIZE * 2 + 5));
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
	if (Log::GetDirection() != Log::Direction::Console) {
		std::fwrite(message.data(), 1, message.size(), stdout);
		std::fflush(stdout);
	}
	Log::Write(message);
	Log::Flush();
}

// The name of the caches made from one build of the game (the static precompile's, the warmup recording): its
// title and version, such as PPSA01341_01.007.000, since versions need not share shaders (Demon's Souls
// 01.005.000 and 01.007.000 have 10 of about 21000 in common). The driver's pipeline cache stays the title's:
// the driver finds a pipeline by its whole input. (Caches named by the title alone, from before, are renamed
// by run-windows.ps1 and precompile-windows.ps1.)
const std::string& PipelineCacheGameId() {
	static const std::string id = [] {
		const auto  title = PipelineCacheTitleId();
		std::string version;
		if (title.empty() || !Loader::SystemContentParamSfoGetString("APP_VER", &version) || version.empty() ||
		    !std::ranges::all_of(version, [](unsigned char c) { return std::isalnum(c) != 0 || c == '.'; }))
			return title;
		return title + "_" + version;
	}();
	return id;
}

// Reads through a null pointer (a draw the game leaves unset during loads): no guest memory is mapped
// there, and the GPU reads zeros where the host would fault. Likewise past the guest address space (1 TiB, the
// trackers' range): an in-game world change (archstone travel to Latria) read an SRT pointer of -1.
constexpr uint64_t NullPageEnd     = 0x10000;
constexpr uint64_t GuestAddressEnd = uint64_t {1} << 40u;
[[nodiscard]] constexpr bool NoGuestMemory(uint64_t address, uint64_t size) {
	return address < NullPageEnd || address >= GuestAddressEnd || GuestAddressEnd - address < size;
}

bool ReadShaderGuestMemory(void*, uint64_t address, uint32_t* value) {
	if (value != nullptr && NoGuestMemory(address, sizeof(*value))) {
		*value = 0;
		return true;
	}
	return value != nullptr &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(address, value, sizeof(*value));
}

bool SyncShaderGuestMemory(void*, uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::SyncGpuCleanBacking(address, size);
}

bool ReadShaderRawGuestMemory(void*, uint64_t address, uint32_t* value) {
	// A GPU-written neighbour may protect a clean descriptor on the same page.
	// Reading its checked backing alias avoids an unnecessary GPU drain. Dirty,
	// unmapped and untracked addresses retain the original load/fault behavior.
	if (NoGuestMemory(address, sizeof(*value))) {
		*value = 0;
	} else if (Libs::LibKernel::Memory::TryReadGpuCleanBackingOnWatchedPage(address, value, sizeof(*value))) {
		LiveCounters::Add(LiveCounters::SrtWatchedReads);
	} else {
		std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(*value));
	}
	return true;
}

uint64_t ShaderMappingEnd(void*, uint64_t address) {
	return Libs::LibKernel::Memory::MappingEnd(address);
}

bool ReadShaderMemorySpan(void*, uint64_t address, uint32_t* values, uint32_t count, bool clean) {
	// Clean spans also read whole tables (resource materialization); raw spans are SRT groups. The
	// null page goes word by word (the readers above).
	return count >= 2 && count <= (clean ? 1024u : 16u) && !NoGuestMemory(address, uint64_t {count} * 4u) &&
	       Libs::LibKernel::Memory::TryReadGpuShaderSpan(address, values, count * 4u, clean);
}

} // namespace

void MaterializationReaders(ShaderRecompiler::IR::SrtRuntime& runtime) {
	runtime.read_memory                = ReadShaderRawGuestMemory;
	runtime.read_specialization_memory = ReadShaderGuestMemory;
	runtime.sync_memory                = SyncShaderGuestMemory;
	runtime.try_read_memory_span       = ReadShaderMemorySpan;
	runtime.mapping_end                = ShaderMappingEnd;
}

namespace {

// Native XPR records evaluate their SRT again over memory the guest may have reused since: a stale
// pointer on the way fails the evaluation instead of faulting.
bool ReadMappedRawGuestMemory(void* userdata, uint64_t address, uint32_t* value) {
	return Libs::LibKernel::Memory::IsFullyMapped(address, sizeof(*value)) &&
	       ReadShaderRawGuestMemory(userdata, address, value);
}

bool ReadMappedMemorySpan(void* userdata, uint64_t address, uint32_t* values, uint32_t count, bool clean) {
	return Libs::LibKernel::Memory::IsFullyMapped(address, uint64_t {count} * 4u) &&
	       ReadShaderMemorySpan(userdata, address, values, count, clean);
}

// An indirect dispatch takes a failed evaluation back (GetComputeProgram): with no thread groups in its
// arguments the GPU never reads its tables, which then hold anything.
thread_local std::string* g_unevaluated = nullptr;
// A failed evaluation without a caller to take it back (draws, direct dispatches): the caller skips the work
// (TakeMaterializationFailure) instead of the game ending over one draw whose tables hold a stale pointer.
thread_local bool g_materialization_failed = false;

bool ReportMaterialization(const char* label, ShaderType stage, uint64_t hash,
                           const ShaderRecompiler::IR::MaterializeReport& report, bool ok) {
	if (!ok) {
		auto message = fmt::format("shader resource materialization failed: stage={} hash=0x{:016x} reason={}\n",
		                           static_cast<uint32_t>(stage), hash, report.reason);
		if (g_unevaluated == nullptr) {
			g_materialization_failed = true;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::printf("%s (skipped)\n", message.substr(0, message.size() - 1).c_str());
				std::fflush(stdout);
			}
			return false;
		}
		*g_unevaluated = std::move(message);
		return false;
	}
	if (!report.dropped_summary.empty()) {
		LOGF("%s indirect image tables: hash=0x%016" PRIx64 " dropped=%" PRIu32 " shapes=%" PRIu32
		     "%s\n",
		     label, hash, report.dropped_candidates, report.dropped_shapes,
		     report.dropped_summary.c_str());
	}
	return true;
}

// KYTY_DUMP_HASHES=<hash,hash,...>: the SPIR-V and guest code of these programs, without the whole
// graphics debug dump (tools/local/spirv-stats.py reads the driver's statistics of the modules).
bool DumpWanted(uint64_t hash) {
	static const std::string list = [] {
		const char* value = std::getenv("KYTY_DUMP_HASHES");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return !list.empty() && list.find(fmt::format("{:016x}", hash)) != std::string::npos;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled() && !DumpWanted(shader_hash)) {
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
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!Config::GraphicsDebugDumpEnabled() && !DumpWanted(shader_hash)) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
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
	std::fprintf(stderr, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

// Resource plans out of memory (ProgramCache::SweepPlans): zstd-compressed ResourcePlanSerial bytes in a scratch file,
// as the shader prefetch spills its own. A plan is ~77 KB of IR in memory and ~2.6 KB there (the prefetch's 20,133:
// 1,555 MiB, 52 MiB written); the warmup's (32 GB of RAM: 9,018 programs, ~700 MB) and those of every place played
// stayed in memory all session. KYTY_PLAN_EVICT=0 keeps them.
struct SpilledPlan {
	uint64_t file = 0, offset = 0;
	uint32_t bytes = 0, raw = 0; // (bytes 0: not written)
};

static bool PlanEvictionEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PLAN_EVICT");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return enabled;
}

// A plan's bytes written (any thread: the file's end reserved atomically).
static bool SpillPlanBytes(const ShaderRecompiler::IR::ResourcePlan& plan, SpilledPlan& out) {
	static std::once_flag        once;
	static uint64_t              file = 0;
	static std::atomic<uint64_t> end {0};
	std::call_once(once, [] { file = LocalPlatform::OpenScratchFile(); });
	std::vector<uint8_t> bytes;
	if (file == 0 || !ResourcePlanSerial::Save(plan, bytes) || bytes.size() > UINT32_MAX) return false;
	const auto packed = ZstdPack(bytes.data(), bytes.size());
	if (packed.empty()) return false;
	const auto offset = end.fetch_add(packed.size(), std::memory_order_relaxed);
	if (!LocalPlatform::WriteScratchFile(file, offset, packed.data(), packed.size())) return false;
	out = {file, offset, static_cast<uint32_t>(packed.size()), static_cast<uint32_t>(bytes.size())};
	return true;
}

// A plan SpillPlanBytes or the shader prefetch wrote (its linear SRT plan left to the caller); false when it cannot be
// read.
static bool LoadPlanBytes(const SpilledPlan& in, ShaderRecompiler::IR::ResourcePlan& plan, bool& linear) {
	std::vector<char>    packed(in.bytes);
	std::vector<uint8_t> bytes(in.raw);
	return LocalPlatform::ReadScratchFile(in.file, in.offset, packed.data(), packed.size()) &&
	       ZstdUnpack(packed, bytes.data(), bytes.size()) && ResourcePlanSerial::Load(bytes, plan, &linear);
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
		bool                                         table_mode = false; // an empty handle: refused
	};

	// A permutation the prefetch translated, its SPIR-V in the prefetch's scratch file: zstd-compressed (spirv_bytes,
	// a fifth of it), or as it is (spirv_bytes 0).
	struct PrefetchedModule {
		ShaderRecompiler::IR::ResourceSpecialization specialization; // compiled with
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		uint64_t                                     spirv_offset = 0;
		size_t                                       spirv_words  = 0;
		size_t                                       spirv_bytes  = 0;
		uint64_t                                     spirv_hash   = 0; // XXH3 of the SPIR-V (NotePipelineKeyModule)
	};

	// A table mode permutation a compile worker translates (Get): the next Get of its specialization and push data
	// start publishes it. It owns what the translation reads.
	struct TableJob {
		ShaderRecompiler::IR::ResourceSpecialization   specialization;
		uint32_t                                       push_data_cursor = 0;
		std::vector<uint32_t>                          code, back_code, user_data;
		ShaderVertexInputInfo                          vertex;
		ShaderPixelInputInfo                           pixel;
		ShaderComputeInputInfo                         compute;
		ShaderRecompiler::CompileOptions               options;
		std::optional<ShaderRecompiler::CompileResult> result;
		ShaderRecompiler::IR::CompiledShaderInfo       info;             // the result's, taken by the worker
		vk::ShaderModule                               module = nullptr; // the result's, made by the worker
		std::string                                    refused;
		std::atomic<bool>                              done {false};
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {}

		ShaderRecompiler::IR::ResourcePlan                resource_plan;
		ShaderRecompiler::IR::ResourceSpecializationGuard specialization_guard;
		// ShaderStageRuntime keeps a pointer into a compiled permutation. Later
		// specializations of the same source must not invalidate an earlier draw.
		std::deque<Permutation> permutations;
		// The permutation the specialization guard selected at `guarded_publication`.
		const Permutation* guarded_permutation = nullptr;
		uint64_t           guarded_publication = 0;
		// Verified against live shader bytes on first use, then released. This
		// also rejects warm records from a changed game with a reused shader hash.
		std::vector<uint32_t> warm_code, warm_back_code;
		// SourceDigest of the code it was made of: a warmup running beside the game adds a record's permutation to a
		// program the game made only of the same code (TranslateWarmup).
		uint64_t source_digest = 0;
		// Prefetched permutations not used yet (AdoptPrefetchedModule).
		std::vector<PrefetchedModule> prefetched;
		// Table mode permutations being translated.
		std::vector<std::shared_ptr<TableJob>> table_jobs;
		// The plan out of memory (SweepPlans): its bytes, whether only they hold it, its linear SRT plan meanwhile (made
		// from the plan alone: rebuilding it took ~1 ms), the sweep of its last use.
		SpilledPlan                                                 plan_spilled;
		bool                                                        plan_out = false;
		std::shared_ptr<const ShaderRecompiler::IR::LinearSrtPlan> plan_linear;
		uint32_t                                                    plan_use = 0;
		bool                                                        plan_loading = false; // a worker reads it (PreloadPlans)
		// Its place in plans_out_by_use while out of memory and not asked of a worker (plan_listed).
		std::multimap<uint32_t, SourceEntry*>::iterator             plan_node {};
		bool                                                        plan_listed = false;
		bool                                                        plan_preloaded = false; // back by a worker, not used since
	};
	// Plans: sweeps so far (one a flip, SweepPlans), out of memory now, brought back (by the render thread; by workers).
	uint32_t plan_clock = 0;
	size_t   plans_out = 0;
	uint64_t plan_reloads = 0, plan_preloads = 0, plan_preload_uses = 0;

	// A place's plans come back together: a frame at 1-1 back from other worlds rebuilt ~70 (29 ms of IR, operator new
	// the most of it) on the render thread. The first one it brings back sends the others of its time (last used within
	// PreloadWindow flips of it) to the compile workers, and the render thread takes theirs as they come (EnsurePlan,
	// each flip SweepPlans). The plans out of memory by their last use; the workers' results (by entry, with the count of
	// erased programs when they were asked: an entry erased since may be gone, its result dropped unread).
	std::multimap<uint32_t, SourceEntry*> plans_out_by_use;
	struct LoadedPlan {
		ShaderRecompiler::IR::ResourcePlan plan;
		bool                               linear = false;
		uint64_t                           erased = 0;
	};
	struct PlanLoads {
		std::mutex                                   mutex;
		std::unordered_map<SourceEntry*, LoadedPlan> plans;
	};
	std::shared_ptr<PlanLoads> plan_loads = std::make_shared<PlanLoads>();
	uint64_t                   programs_erased = 0;
	void UnlistPlan(SourceEntry& source) {
		if (!source.plan_listed) return;
		plans_out_by_use.erase(source.plan_node);
		source.plan_listed = false;
	}
	// An entry about to be erased: no queue keeps it (rare: a warm program whose code changed).
	void ForgetPlanLoads(SourceEntry& entry) {
		UnlistPlan(entry);
		programs_erased++;
	}

	// The plan back in the entry (its linear SRT plan kept meanwhile).
	void RestorePlan(SourceEntry& source, ShaderRecompiler::IR::ResourcePlan plan) {
		UnlistPlan(source);
		plan.linear_srt      = std::move(source.plan_linear);
		source.resource_plan = std::move(plan);
		// (The guard knows the plan by its address: the same, but it is checked again from the start.)
		source.specialization_guard = {};
		source.guarded_permutation  = nullptr;
		source.plan_out             = false;
		plans_out--;
	}

	// A worker's result for the entry, taken (false: none, or one not to use).
	bool TakeLoadedPlan(SourceEntry& source, ShaderRecompiler::IR::ResourcePlan& plan) {
		LoadedPlan loaded;
		{
			std::lock_guard lock(plan_loads->mutex);
			const auto      found = plan_loads->plans.find(&source);
			if (found == plan_loads->plans.end()) return false;
			loaded = std::move(found->second);
			plan_loads->plans.erase(found);
		}
		source.plan_loading = false;
		if (loaded.erased != programs_erased || loaded.linear != (source.plan_linear != nullptr)) return false;
		plan = std::move(loaded.plan);
		return true;
	}

	// The plans out of memory last used within PreloadWindow flips of `around` to the compile workers.
	void PreloadPlans(uint32_t around) {
		// (16 GB simulated, 5 worlds x2: 600 flips brought back 14,072 plans beside the render thread, 3,861 of them used
		// before leaving again, and left it 207 of its own; 150 flips missed some: spins at 1-1 and Latria fell to 30-38 fps.)
		static constexpr uint32_t PreloadWindow = 600;
		static constexpr size_t   MaxPlans = 256, PlansPerJob = 4;
		if (!push_job || plans_out_by_use.empty()) return;
		std::vector<std::pair<SourceEntry*, SpilledPlan>> batch;
		for (auto it = plans_out_by_use.lower_bound(around > PreloadWindow ? around - PreloadWindow : 0);
		     it != plans_out_by_use.end() && it->first <= around + PreloadWindow && batch.size() < MaxPlans;) {
			auto* entry        = it->second;
			it                 = plans_out_by_use.erase(it); // (listed again when out again)
			entry->plan_listed = false;
			if (!entry->plan_out || entry->plan_loading) continue;
			entry->plan_loading = true;
			batch.emplace_back(entry, entry->plan_spilled);
		}
		for (size_t first = 0; first < batch.size(); first += PlansPerJob) {
			std::vector<std::pair<SourceEntry*, SpilledPlan>> part(
			    batch.begin() + static_cast<std::ptrdiff_t>(first),
			    batch.begin() + static_cast<std::ptrdiff_t>(std::min(batch.size(), first + PlansPerJob)));
			push_job([loads = plan_loads, part = std::move(part), erased = programs_erased] {
				for (const auto& [entry, spilled]: part) {
					LoadedPlan loaded;
					loaded.erased = erased;
					if (!LoadPlanBytes(spilled, loaded.plan, loaded.linear)) continue; // (the render thread reads it)
					std::lock_guard lock(loads->mutex);
					loads->plans[entry] = std::move(loaded);
				}
			});
		}
	}

	// The workers' results in their entries (each flip): those of entries still out, a use noted (their place is back).
	void DrainLoadedPlans() {
		std::unordered_map<SourceEntry*, LoadedPlan> plans;
		{
			std::lock_guard lock(plan_loads->mutex);
			if (plan_loads->plans.empty()) return;
			plans.swap(plan_loads->plans);
		}
		for (auto& [entry, loaded]: plans) {
			if (loaded.erased != programs_erased) continue; // (the entry may be gone)
			entry->plan_loading = false;
			if (!entry->plan_out || loaded.linear != (entry->plan_linear != nullptr)) continue;
			RestorePlan(*entry, std::move(loaded.plan));
			entry->plan_use       = plan_clock;
			entry->plan_preloaded = true;
			plan_preloads++;
		}
	}

	// The source's plan in memory again (before each use of it), its last use noted.
	void EnsurePlan(SourceEntry& source) {
		const auto last_use = source.plan_use;
		source.plan_use     = plan_clock;
		if (source.plan_preloaded) {
			source.plan_preloaded = false;
			plan_preload_uses++;
		}
		if (!source.plan_out) return;
		ShaderRecompiler::IR::ResourcePlan plan;
		if (!TakeLoadedPlan(source, plan)) {
			PreloadPlans(last_use);
			bool linear = false;
			if (!LoadPlanBytes(source.plan_spilled, plan, linear) || linear != (source.plan_linear != nullptr))
				EXIT("a shader's resource plan could not be read back from the scratch file\n");
		}
		RestorePlan(source, std::move(plan));
		plan_reloads++;
	}

	// Each flip; every PlanSweepFlips, the plans no use needed for PlanIdleFlips leave memory beyond a budget of RAM/128
	// (~77 KB a plan: 1,700 at 16 GB, 3,400 at 32 GB, 13,600 at 128 GB), the least recently used first, written first
	// where no file holds them yet (a few a sweep: programs translated during play; the warmup's and the prefetch's are).
	// They are destroyed on a compile worker (the warmup's thousands at once: their IR nodes freed off the render thread).
	// Without the budget every idle plan left, also on PCs with RAM to spare: a place came back with a frame of reloads
	// (~70 plans, 29 ms of IR rebuilt): a walk at 1-1 back from other worlds peaked at 42 ms each time, 20-24 ms kept;
	// Stonefang's walk 1% low 54-57 fps, 78-81 kept.
	void SweepPlans() {
		// (KYTY_PLAN_IDLE_FLIPS=<n>: another idle time, and a sweep each n/30 flips: a test of plans leaving and coming
		// back all the time.)
		static const uint32_t PlanIdleFlips = [] {
			const char* value = std::getenv("KYTY_PLAN_IDLE_FLIPS");
			return value != nullptr && *value != '\0' ? std::max(30u, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)))
			                                          : 3600u;
		}();
		static constexpr uint32_t PlanWritesPerSweep = 4;
		const uint32_t            PlanSweepFlips     = PlanIdleFlips / 30u;
		DrainLoadedPlans();
		if (!PlanEvictionEnabled() || ++plan_clock % PlanSweepFlips != 0) return;
		// (RAM the system does not report: a budget of 0, every idle plan leaves.)
		static const size_t budget = [] {
			const uint64_t ram = LocalPlatform::PhysicalMemory();
			return static_cast<size_t>(ram / 128u / (77u * 1024u));
		}();
		size_t                    in_memory = 0;
		std::vector<SourceEntry*> idle;
		for (auto& [key, source]: programs) {
			(void)key;
			if (source.plan_out) continue;
			in_memory++;
			if (plan_clock - source.plan_use >= PlanIdleFlips) idle.push_back(&source);
		}
		if (in_memory <= budget || idle.empty()) return;
		std::ranges::sort(idle, {}, [](const SourceEntry* source) { return source->plan_use; });
		idle.resize(std::min(idle.size(), in_memory - budget));
		uint32_t writes = 0;
		auto     freed  = std::make_shared<std::vector<ShaderRecompiler::IR::ResourcePlan>>();
		for (auto* entry: idle) {
			auto& source = *entry;
			if (source.plan_spilled.bytes == 0) {
				if (writes >= PlanWritesPerSweep || !SpillPlanBytes(source.resource_plan, source.plan_spilled)) continue;
				writes++;
			}
			source.plan_linear = std::move(source.resource_plan.linear_srt);
			freed->push_back(std::move(source.resource_plan));
			source.resource_plan = {};
			source.plan_out       = true;
			source.plan_preloaded = false;
			plans_out++;
			source.plan_node   = plans_out_by_use.emplace(source.plan_use, &source);
			source.plan_listed = true;
		}
		if (freed->empty()) return;
		if (freed->size() >= 256)
			PipelineCacheLog("Shader plans: {} out of memory ({} now), {} brought back so far ({} beside the render thread, "
			                 "{} of those used)",
			                 plans_out, freed->size(), plan_reloads + plan_preloads, plan_preloads, plan_preload_uses);
		if (push_job) push_job([freed] { freed->clear(); });
	}
	// Runs a job on the pipeline cache's compile workers (set by PipelineCache).
	std::function<void(std::function<void()>)> push_job;
	// The last table mode Get found its permutation still translating (empty, not refused), and that translation's
	// completion.
	bool                                     table_pending = false;
	std::shared_ptr<const std::atomic<bool>> table_waiting;

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
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	// KYTY_SHADER_PREFETCH (default on; 0 off): the programs of the static precompile's inputs
	// (StaticInputsPath: every shader the game ships, with the resource specialization it most likely
	// gets) are translated on low-priority threads while the game runs, largest first, so a program
	// met for the first time after the warmup needs only its shader module (Get). Their SPIR-V
	// (gigabytes for the whole game) and resource plans wait in a temporary file, the infos in memory.
	// Heap bytes of a prefetched group by part (AddGroupMemory).
	enum PrefetchMemoryPart : size_t { MemPlanValues, MemPlanMemoryInfo, MemPlanOther, MemPlanInfo, MemModuleInfo,
		MemModuleTable, MemModuleBda, MemModuleBindings, MemModuleSpecialization, MemGroup, PrefetchMemoryParts };
	struct Prefetch {
		enum State : uint8_t { Pending, Running, Done, Taken };
		struct Group { // the inputs of one program
			ProgramKey                                        key;
			uint64_t                                          source = 0; // SourceDigest
			size_t                                            code_words = 0;
			uint32_t                                          recorded = UINT32_MAX; // its first record in this PC's play
			uint32_t                                          first_record = UINT32_MAX; // (by_record's index of it)
			std::vector<uint32_t>                             records;
			std::atomic<uint8_t>                              state {Pending};
			std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
			std::vector<PrefetchedModule>                     modules;
			// The plan in the scratch file instead (zstd-compressed ResourcePlanSerial bytes): `plan` stays empty until
			// the game takes the group over (LoadSpilledPlan). Its linear SRT plan stays here: rebuilding it (its code
			// generated, its AOT function looked up) took ~1 ms of the render thread's per plan; the plan's values ~0.2.
			uint64_t                                                          plan_offset = 0;
			uint32_t                                                          plan_bytes = 0, plan_raw = 0;
			std::shared_ptr<const ShaderRecompiler::IR::LinearSrtPlan>       plan_linear;
		};
		~Prefetch() {
			{
				std::lock_guard lock(follow_mutex);
				stop.store(true, std::memory_order_relaxed);
			}
			follow_cv.notify_all();
			if (loader.joinable()) loader.join();
			for (auto& thread: threads) thread.join();
			// (Stored modules read their SPIR-V from it: CreateModuleInPlace.)
			if (!spill_in_store.load(std::memory_order_relaxed)) LocalPlatform::CloseScratchFile(spill);
		}

		LocalShaderWarmup::Cache                               inputs;
		std::vector<std::unique_ptr<Group>>                    groups; // in translation order
		std::unordered_map<ProgramKey, Group*, ProgramKeyHash> index;  // read-only once `ready`
		std::atomic<bool>                                      ready {false}, stop {false};
		std::atomic<size_t>                                    next {0}, finished {0};
		// The groups of recorded programs (the first ones) and how many workers go on after them (StartPrefetch).
		size_t   recorded_groups = 0;
		uint32_t gentle_threads  = 1;
		// Following play (Follow, Followed): each record of this PC's recorded play's group, in recorded order (read-only
		// once `ready`), and windows of records after programs the game met, translated before the rest (the newest
		// window first). Play meets programs in about the order it recorded them: of the 157 programs a 16 GB laptop still
		// translated itself in a boot and prologue walk (1.3 s; the oldest and newest alternately had not reached the middle
		// of its 11,501 records), 84 were within 64 records of the 40 programs met before, often the very next record:
		// runs of 9 and 10 met in one frame (~110 ms), which helper threads (FollowWork) translate beside the render
		// thread. Not followed 120 and 157 (0.9 and 1.3 s), followed 74 (0.56 s), with the helpers 32 and 48 (0.24 and
		// 0.27 s); the walk's fps alike (22.5-25 either way, run to run).
		static constexpr uint32_t FollowWindow = 256, FollowWindows = 8;
		struct Window {
			uint32_t start = UINT32_MAX, next = UINT32_MAX;
		};
		std::vector<Group*>                by_record;
		std::mutex                         follow_mutex;
		std::condition_variable            follow_cv;
		std::array<Window, FollowWindows>  windows {};
		uint32_t                           newest     = 0; // windows[newest % FollowWindows]: the last one opened
		uint64_t                           generation = 0; // windows opened
		std::atomic<size_t>                followed {0};
		// Any thread taking a program over: play met the program whose first record this is. (KYTY_SHADER_FOLLOW=0: not
		// followed.)
		void Follow(uint32_t record) {
			static const bool off = [] {
				const char* value = std::getenv("KYTY_SHADER_FOLLOW");
				return value != nullptr && std::string_view(value) == "0";
			}();
			if (off || record >= by_record.size()) return;
			{
				std::lock_guard lock(follow_mutex);
				for (const auto& window: windows)
					if (window.start != UINT32_MAX && record - window.start < FollowWindow) return; // (followed already)
				windows[++newest % FollowWindows] = {record, record + 1};
				generation++;
			}
			follow_cv.notify_all();
		}
		// The next pending group of the windows, the newest first, or null (`seen`: the windows opened by then).
		Group* Followed(uint64_t* seen = nullptr) {
			std::lock_guard lock(follow_mutex);
			if (seen != nullptr) *seen = generation;
			for (uint32_t age = 0; age < FollowWindows; ++age) {
				auto& window = windows[(newest - age) % FollowWindows];
				while (window.start != UINT32_MAX && window.next - window.start < FollowWindow && window.next < by_record.size()) {
					auto* group = by_record[window.next++];
					if (group != nullptr && group->state.load(std::memory_order_relaxed) == Pending) return group;
				}
			}
			return nullptr;
		}
		std::atomic<uint64_t>                                  spill_end {0};
		std::atomic<uint64_t>                                  spirv_bytes {0}; // before compression
		uint64_t                                               spill = 0;
		std::atomic<bool>                                      spill_in_store {false};
		std::thread                                            loader;
		std::vector<std::thread>                               threads;
		// Render thread: programs and modules taken over, programs translated there while pending; groups of warm programs
		// taken over (TakePrefetchedModules) and found not done yet.
		size_t adopted = 0, adopted_modules = 0, claimed = 0, taken = 0, take_early = 0;
		// Render thread: spilled plans loaded back, and the time it took (LoadSpilledPlan).
		size_t plans_loaded = 0;
		double plan_load_ms = 0.0, plan_load_max_ms = 0.0, plan_read_ms = 0.0, plan_values_ms = 0.0, plan_linear_ms = 0.0;
		// What the translated groups held in memory before their plans were spilled (AddGroupMemory), summed by the
		// workers as they finish them.
		std::array<std::atomic<uint64_t>, PrefetchMemoryParts> memory {};
		// Plans written to the scratch file (SpillPlan) and their bytes there; KYTY_PLAN_SPILL_VERIFY: reloads that saved
		// differently.
		std::atomic<uint64_t> spilled_plans {0}, spilled_plan_bytes {0}, spill_verify_failures {0};
	};
	std::unique_ptr<Prefetch> prefetch;
	// `prefetch` for threads without the pipeline cache's lock (GetPrefetchProgress, the live report): a warmup beside
	// the game starts the prefetch once done.
	std::atomic<Prefetch*> prefetch_live {nullptr};

	// Heap bytes of a prefetched group by part (container capacities, list and string nodes; allocator headers left
	// out): what keeping every program's plan and infos costs.
	template <typename T> static uint64_t Capacity(const std::vector<T>& v) { return v.capacity() * sizeof(T); }
	static uint64_t InfoBytes(const ShaderRecompiler::IR::ShaderInfo& info) {
		uint64_t n = Capacity(info.buffers) + Capacity(info.images) + Capacity(info.samplers) + Capacity(info.sampled_pairs) +
		             Capacity(info.inputs) + Capacity(info.outputs);
		for (const auto& image: info.images) n += Capacity(image.indirect_resources);
		for (const auto& input: info.inputs) n += input.debug_name.capacity() > 15 ? input.debug_name.capacity() + 1 : 0;
		for (const auto& output: info.outputs) n += output.debug_name.capacity() > 15 ? output.debug_name.capacity() + 1 : 0;
		return n;
	}
	static void AddGroupMemory(Prefetch& p, const Prefetch::Group& group) {
		std::array<uint64_t, PrefetchMemoryParts> n {};
		n[MemGroup] = sizeof(group) + Capacity(group.records) + Capacity(group.modules);
		if (group.plan) {
			const auto& plan = *group.plan;
			for (const auto& inst: plan.value_storage)
				n[MemPlanValues] += sizeof(inst) + 2 * sizeof(void*) + inst.NumArgs() * sizeof(ShaderRecompiler::IR::Value) +
				                    inst.NumPhiBlocks() * sizeof(void*) + inst.Uses().capacity() * sizeof(ShaderRecompiler::IR::Use);
			n[MemPlanMemoryInfo] = Capacity(plan.memory_info);
			n[MemPlanOther]      = Capacity(plan.descriptor_sources) + Capacity(plan.control_flow) +
			                       Capacity(plan.materialization_sources) + Capacity(plan.srt_reads) + Capacity(plan.clean_flat_slots);
			for (const auto& block: plan.control_flow) n[MemPlanOther] += Capacity(block.successors) + Capacity(block.sources);
			if (plan.linear_srt) {
				const auto& linear = *plan.linear_srt;
				n[MemPlanOther] += sizeof(linear) + Capacity(linear.nodes) + Capacity(linear.read_groups) +
				                   Capacity(linear.descriptor_words) + Capacity(linear.flat_words) + Capacity(linear.sources) +
				                   Capacity(linear.clean_slots) + Capacity(linear.descriptor_sizes) + Capacity(linear.active_sources);
			}
			n[MemPlanInfo] = InfoBytes(plan.info);
		}
		for (const auto& module: group.modules) {
			const auto& info = module.program;
			n[MemModuleInfo] += InfoBytes(info.info);
			n[MemModuleTable] += Capacity(info.table_plan.slots) + Capacity(info.table_plan.buffers) +
			                     Capacity(info.table_plan.images) + Capacity(info.table_plan.samplers) +
			                     Capacity(info.table_plan.gpu) + Capacity(info.table_plan.cpu) + Capacity(info.table_plan.indirect);
			n[MemModuleBda] += Capacity(info.bda_read_plan.spans);
			n[MemModuleBindings] += Capacity(info.bindings.user_data_registers) + Capacity(info.bindings.descriptors);
			for (const auto& binding: info.bindings.descriptors) n[MemModuleBindings] += Capacity(binding.resources);
			n[MemModuleSpecialization] += Capacity(module.specialization.buffers) + Capacity(module.specialization.images);
		}
		for (size_t i = 0; i < PrefetchMemoryParts; ++i) p.memory[i].fetch_add(n[i], std::memory_order_relaxed);
	}

	static uint64_t SourceDigest(std::span<const uint32_t> code, std::span<const uint32_t> back_code) {
		return XXH3_64bits_withSeed(back_code.data(), back_code.size_bytes(), XXH3_64bits(code.data(), code.size_bytes()));
	}

	// Loads the inputs on a thread of its own, leaves out the programs the warmup holds (`warm`), and
	// starts the translation: first the programs of this PC's recorded play (`recorded`, the warmup's records when it
	// translated none) in the order they were first met, then the rest, the biggest first. A PC without recorded play
	// (`recorded` and `warm` empty: a new install) takes the release's instead (`released`, a recorded seed file; its
	// compute records made for `host_subgroup_size`).
	void StartPrefetch(std::filesystem::path path, std::string identity, std::vector<ProgramKey> warm,
	                   std::vector<std::vector<uint32_t>> recorded, std::filesystem::path released,
	                   uint32_t host_subgroup_size) {
		prefetch = std::make_unique<Prefetch>();
		prefetch_live.store(prefetch.get(), std::memory_order_release);
		prefetch->loader = std::thread([this, path = std::move(path), identity = std::move(identity),
		                                warm = std::move(warm), recorded = std::move(recorded),
		                                released = std::move(released), host_subgroup_size]() mutable {
			LocalPlatform::SetThreadName("Kyty.Prefetch");
			LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
			auto&      p     = *prefetch;
			const auto begin = std::chrono::steady_clock::now();
			// The release's recorded play (other PCs' tours of every world and players' play: the release's
			// specialization hints, which precompile-windows.ps1 completes with this PC's seed file). A new install's prefetch
			// went biggest first and had not reached the level's programs when the game got there: entering 1-1, the render
			// thread translated ~490 programs (~1.1 s: a first frame of 1.1-1.2 s, 16-18 slow frames). Taken over now as
			// from a recording of its own: a first frame of 0.4 s, 9 slow frames, none translated there.
			bool from_release = false;
			if (recorded.empty() && warm.empty() && !released.empty()) {
				std::vector<std::vector<uint32_t>> pipelines;
				from_release = LocalShaderWarmup::Cache::ReadInputs(released, recorded, pipelines) && !recorded.empty();
				PipelineCacheLog("Shader prefetch: no recorded play on this PC: the release's {} {} ({} records)",
				                 from_release ? "from" : "could not be read from", Common::PathToString(released),
				                 recorded.size());
				if (!from_release) recorded.clear();
			}
			if (!std::filesystem::exists(path) || !p.inputs.Open(path, identity) || p.inputs.records.empty()) {
				PipelineCacheLog("Shader prefetch: no inputs for this GPU and driver in {} (precompile-windows.ps1 "
				                 "writes them){}", Common::PathToString(path), recorded.empty() ? "" : ": the recording's only");
				if (recorded.empty()) return;
				p.inputs.records.clear();
			}
			const std::unordered_set<ProgramKey, ProgramKeyHash> skip(warm.begin(), warm.end());
			for (uint32_t i = 0; i < p.inputs.records.size() && !p.stop.load(std::memory_order_relaxed); ++i) {
				LocalShaderWarmup::Record record;
				LocalShaderWarmup::Reader reader {p.inputs.records[i]};
				if (!LocalShaderWarmup::Visit(reader, record) || !LocalShaderWarmup::ValidKey(record)) continue;
				ProgramKey key {record.stage, record.hash, record.user_data_count,
				                static_cast<uint32_t>(record.code.size()), record.static_key};
				if (skip.contains(key)) continue;
				auto [found, inserted] = p.index.try_emplace(key, nullptr);
				if (inserted) {
					auto& group      = *p.groups.emplace_back(std::make_unique<Prefetch::Group>());
					group.key        = std::move(key);
					group.source     = SourceDigest(record.code, record.back_code);
					group.code_words = record.code.size();
					found->second    = &group;
				}
				found->second->records.push_back(i);
			}
			// A PC with too little RAM to translate its recorded play at start-up (16 GB): the programs it needs while the
			// game starts and loads are among the first recorded (1,138 translated on the render thread at the laptop's
			// tutorial start, 13.8 s, 58% of them in the oldest tenth of the recording, all in it), and the biggest-first
			// order reached them minutes later (20K programs on 4 threads). The newest are the place played last, where the
			// save most likely is: oldest first, a 16 GB laptop walking 1-1 a minute after its HUD still translated 201
			// programs itself (2.4 s, up to 121 ms each), 177 of them in the newest tenth of its 11,492 records. So the
			// oldest and the newest alternately (RecordedRank, as the warmup's WarmOrder).
			// The recording's records the static inputs lack (play since the last precompile, or a warmup that stopped at its
			// budget before them: the inputs hold the seeds' guesses and the hints) are inputs too, each record once: those
			// the warmup did not translate, also of its programs (their other specializations: TakePrefetchedModules).
			uint32_t ranked = 0, added = 0;
			p.by_record.resize(recorded.size());
			for (uint32_t i = 0; i < recorded.size() && !p.stop.load(std::memory_order_relaxed); ++i) {
				LocalShaderWarmup::Record record;
				LocalShaderWarmup::Reader reader {recorded[i]};
				if (!LocalShaderWarmup::Visit(reader, record) || !LocalShaderWarmup::ValidKey(record)) continue;
				// The release's compute records hold one subgroup size for every GPU (the hints' default): this GPU's, as
				// the precompile takes them, else none of their programs is the one a dispatch looks up (on NVIDIA 9,974
				// of the release's 17,809 programs).
				if (from_release && record.stage == ShaderType::Compute &&
				    record.compute.host_subgroup_size != host_subgroup_size) {
					record.compute.host_subgroup_size = host_subgroup_size;
					BuildStageStaticKey(record.compute, record.static_key);
					LocalShaderWarmup::Writer writer;
					(void)LocalShaderWarmup::Visit(writer, record);
					recorded[i] = std::move(writer.words);
				}
				ProgramKey key {record.stage, record.hash, record.user_data_count,
				                static_cast<uint32_t>(record.code.size()), record.static_key};
				auto [found, inserted] = p.index.try_emplace(key, nullptr);
				if (inserted) {
					auto& group      = *p.groups.emplace_back(std::make_unique<Prefetch::Group>());
					group.key        = std::move(key);
					group.source     = SourceDigest(record.code, record.back_code);
					group.code_words = record.code.size();
					found->second    = &group;
				}
				auto& group    = *found->second;
				p.by_record[i] = &group;
				if (std::ranges::none_of(group.records, [&](uint32_t index) { return p.inputs.records[index] == recorded[i]; })) {
					group.records.push_back(static_cast<uint32_t>(p.inputs.records.size()));
					p.inputs.records.push_back(std::move(recorded[i]));
					added++;
				}
				if (group.recorded == UINT32_MAX) {
					// (The release's in its own order: its newest records are no place this PC played last.)
					group.recorded     = from_release ? i : RecordedRank(i, static_cast<uint32_t>(recorded.size()));
					group.first_record = i;
					ranked++;
				}
			}
			std::vector<std::vector<uint32_t>>().swap(recorded);
			std::ranges::stable_sort(p.groups, {}, [](const auto& group) {
				return std::pair {group->recorded, ~group->code_words};
			});
			p.spill = LocalPlatform::OpenScratchFile();
			if (p.spill == 0) {
				PipelineCacheLog("Shader prefetch: no temporary file for the SPIR-V");
				return;
			}
			uint32_t threads = SpareThreads();
			// The recording's programs (the place played, what the game soon needs) on all of them, the rest (places not
			// played on this PC yet) on half: they ran into the play, on the other half of the render thread's core too.
			// Standing at 1-1 on an 8-core laptop, the render thread took 24.2-25.2 ms a frame beside 4 prefetch threads,
			// 21.6 ms beside 2 and 19.4 ms once the prefetch was done (all of it: 3.5 minutes on 4 threads, 5.5 on 2).
			// KYTY_SHADER_PREFETCH_THREADS: that many all along.
			p.gentle_threads = std::max(1u, threads / 2);
			if (const char* value = std::getenv("KYTY_SHADER_PREFETCH_THREADS"); value != nullptr && *value != '\0')
				p.gentle_threads = threads = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 64u);
			// (The release's play orders the work alone: it names most of the game's programs, translated on the gentle
			// threads as without it.)
			p.recorded_groups = from_release ? 0
			                                 : static_cast<size_t>(std::ranges::count_if(p.groups, [](const auto& group) {
				                                   return group->recorded != UINT32_MAX;
			                                   }));
			PipelineCacheLog("Shader prefetch: {} programs ({} inputs, {} of them recorded play's; {} recorded first) to "
			                 "translate on {} threads ({} after the recorded ones), loaded in {} ms",
			                 p.groups.size(), p.inputs.records.size(), added, ranked, threads, p.gentle_threads,
			                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count());
			p.ready.store(true, std::memory_order_release);
			for (uint32_t t = 0; t < threads; ++t) p.threads.emplace_back([this, t] { PrefetchWork(t); });
			// (KYTY_SHADER_FOLLOW_THREADS: that many helpers, 0 none.)
			uint32_t helpers = p.by_record.empty() ? 0 : 2;
			if (const char* value = std::getenv("KYTY_SHADER_FOLLOW_THREADS"); value != nullptr && *value != '\0')
				helpers = std::min(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 16u);
			for (uint32_t t = 0; t < helpers; ++t) p.threads.emplace_back([this] { FollowWork(); });
		});
	}

	// The CPUs the game leaves to work beside it (the shader prefetch, a warmup beside the game): its loading keeps
	// about 12 threads busy (guest job threads, the GPU and recording threads), and threads beside them (on the other
	// half of a core) slow them however low their priority. An 8-core laptop (16 CPUs) with 8 prefetch threads took
	// 120+ s instead of 20 s to load 1-1 (start-up to the game 264-300 s; with 4: 107 s), and with a 16-thread warmup
	// showed the intro after 74-100 s instead of 24 s (2 threads); half the CPUs at most (a 32-CPU desktop keeps 16).
	static uint32_t SpareThreads() {
		const uint32_t cpus = std::max(1u, std::thread::hardware_concurrency());
		return std::clamp(cpus > 12u ? cpus - 12u : 2u, 2u, std::max(2u, cpus / 2u));
	}

	// A recorded record's place in the prefetch's order: the oldest and the newest alternately (0, n-1, 1, n-2, ...).
	static uint32_t RecordedRank(uint32_t index, uint32_t count) {
		return index <= (count - 1) / 2 ? 2 * index : 2 * (count - 1 - index) + 1;
	}

	void PrefetchWork(uint32_t worker) {
		LocalPlatform::SetThreadName("Kyty.Prefetch");
		LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
		auto& p = *prefetch;
		while (!p.stop.load(std::memory_order_relaxed)) {
			// The recorded programs after those the game met first.
			if (auto* group = p.Followed()) {
				if (RunGroup(p, *group)) p.followed.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			if (worker >= p.gentle_threads && p.next.load(std::memory_order_relaxed) >= p.recorded_groups) break;
			if (WaitForMemory(p)) continue; // (a program play just met first)
			const size_t n = p.next.fetch_add(1, std::memory_order_relaxed);
			if (n >= p.groups.size()) break;
			(void)RunGroup(p, *p.groups[n]);
			if (p.finished.fetch_add(1, std::memory_order_relaxed) + 1 == p.groups.size()) {
				PipelineCacheLog("Shader prefetch: all {} programs translated, {} MiB of SPIR-V ({} MiB compressed)",
				                 p.groups.size(), p.spirv_bytes.load(std::memory_order_relaxed) >> 20u,
				                 p.spill_end.load(std::memory_order_relaxed) >> 20u);
				std::vector<std::vector<uint32_t>>().swap(p.inputs.records); // no longer read
				const auto mib = [&](PrefetchMemoryPart part) { return p.memory[part].load(std::memory_order_relaxed) >> 20u; };
				PipelineCacheLog("Shader prefetch memory (MiB): plan values {} memory infos {} other {} info {}, modules info {} "
				                 "table plans {} read plans {} bindings {} specializations {}, groups {}",
				                 mib(MemPlanValues), mib(MemPlanMemoryInfo), mib(MemPlanOther), mib(MemPlanInfo), mib(MemModuleInfo),
				                 mib(MemModuleTable), mib(MemModuleBda), mib(MemModuleBindings), mib(MemModuleSpecialization),
				                 mib(MemGroup));
				PipelineCacheLog("Shader prefetch: {} plans in the temporary file ({} MiB compressed), {} reloaded differently",
				                 p.spilled_plans.load(std::memory_order_relaxed),
				                 p.spilled_plan_bytes.load(std::memory_order_relaxed) >> 20u,
				                 p.spill_verify_failures.load(std::memory_order_relaxed));
			}
		}
	}

	// While the PC has less physical memory left than KYTY_PREFETCH_MIN_AVAIL_MB (default 1024 MiB), the prefetch makes
	// none of the programs play has not met (they are for later): its translations, SPIR-V and plan file add to the paging
	// the PC is doing then, the game loading beside it on 16 GB (issue #50: a bug check 0x1A in a protection change of the
	// game's memory, while a level loaded with the prefetch translating 20,000 programs). True when one play met came up.
	static bool WaitForMemory(Prefetch& p) {
		static const uint64_t minimum = [] {
			const char* value = std::getenv("KYTY_PREFETCH_MIN_AVAIL_MB");
			return (value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : uint64_t {1024}) << 20u;
		}();
		static std::atomic<uint32_t> reports {0};
		bool                         waited = false;
		while (!p.stop.load(std::memory_order_relaxed)) {
			const uint64_t available = LocalPlatform::AvailablePhysicalMemory();
			if (available == 0 || available >= minimum) break;
			if (!waited && reports.fetch_add(1, std::memory_order_relaxed) < 8)
				PipelineCacheLog("Shader prefetch: waits, {} MiB of physical memory left (below {} MiB)", available >> 20u,
				                 minimum >> 20u);
			waited = true;
			if (p.Followed() != nullptr) return true;
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
		return false;
	}

	// The group translated here unless another thread took it.
	bool RunGroup(Prefetch& p, Prefetch::Group& group) {
		auto expected = static_cast<uint8_t>(Prefetch::Pending);
		if (!group.state.compare_exchange_strong(expected, Prefetch::Running, std::memory_order_acq_rel)) return false;
		PrefetchTranslate(group);
		AddGroupMemory(p, group);
		SpillPlan(p, group);
		group.state.store(Prefetch::Done, std::memory_order_release);
		group.state.notify_all();
		return true;
	}

	// A helper of the prefetch: idle but for the windows of programs play just met (Follow), which it translates beside
	// the workers (each busy with a program of its own, up to ~100 ms) and the render thread meeting them.
	void FollowWork() {
		LocalPlatform::SetThreadName("Kyty.Follow");
		LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
		auto& p = *prefetch;
		while (!p.stop.load(std::memory_order_relaxed) && p.finished.load(std::memory_order_relaxed) < p.groups.size()) {
			uint64_t seen = 0;
			if (auto* group = p.Followed(&seen)) {
				if (RunGroup(p, *group)) p.followed.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			std::unique_lock lock(p.follow_mutex);
			p.follow_cv.wait_for(lock, std::chrono::milliseconds(250),
			                     [&] { return p.stop.load(std::memory_order_relaxed) || p.generation != seen; });
		}
	}

	// The group's plan to the scratch file (KYTY_PLAN_SPILL=0 keeps it in memory): 20K plans were ~2.5 GB of RAM.
	static bool PlanSpillEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_PLAN_SPILL");
			return value == nullptr || std::string_view(value) != "0";
		}();
		return enabled;
	}
	void SpillPlan(Prefetch& p, Prefetch::Group& group) {
		if (!group.plan || !PlanSpillEnabled()) return;
		std::vector<uint8_t> bytes;
		if (!ResourcePlanSerial::Save(*group.plan, bytes) || bytes.size() > UINT32_MAX) return;
		static const bool verify = std::getenv("KYTY_PLAN_SPILL_VERIFY") != nullptr;
		if (verify) {
			ShaderRecompiler::IR::ResourcePlan loaded;
			std::vector<uint8_t>               again;
			bool                               linear = false;
			const bool                         reloaded = ResourcePlanSerial::Load(bytes, loaded, &linear);
			loaded.linear_srt                           = group.plan->linear_srt;
			if (!reloaded || linear != (group.plan->linear_srt != nullptr) || !ResourcePlanSerial::Save(loaded, again) ||
			    again != bytes) {
				if (p.spill_verify_failures.fetch_add(1, std::memory_order_relaxed) < 20)
					PipelineCacheLog("Shader prefetch: plan of 0x{:016x} reloads differently", group.plan->shader_hash);
				return;
			}
		}
		const auto packed = ZstdPack(bytes.data(), bytes.size());
		if (packed.empty()) return;
		const auto packed_size = packed.size();
		const auto offset      = p.spill_end.fetch_add(packed_size, std::memory_order_relaxed);
		if (!LocalPlatform::WriteScratchFile(p.spill, offset, packed.data(), packed_size)) return;
		group.plan_offset = offset;
		group.plan_bytes  = static_cast<uint32_t>(packed_size);
		group.plan_raw    = static_cast<uint32_t>(bytes.size());
		group.plan_linear = std::move(group.plan->linear_srt); // (made from the plan alone: no reference into it)
		group.plan.reset();
		p.spilled_plans.fetch_add(1, std::memory_order_relaxed);
		p.spilled_plan_bytes.fetch_add(packed_size, std::memory_order_relaxed);
	}
	// The plan SpillPlan wrote, back in the group (render thread, the group taken over); false when it cannot be read.
	bool LoadSpilledPlan(Prefetch::Group& group) {
		if (group.plan || group.plan_bytes == 0) return group.plan.has_value();
		const auto begin = std::chrono::steady_clock::now();
		std::vector<char>    packed(group.plan_bytes);
		std::vector<uint8_t> bytes(group.plan_raw);
		if (!LocalPlatform::ReadScratchFile(prefetch->spill, group.plan_offset, packed.data(), packed.size()) ||
		    !ZstdUnpack(packed, bytes.data(), bytes.size()))
			return false;
		const auto read = std::chrono::steady_clock::now();
		ShaderRecompiler::IR::ResourcePlan plan;
		bool                               linear = false;
		if (!ResourcePlanSerial::Load(bytes, plan, &linear)) return false;
		const auto values = std::chrono::steady_clock::now();
		if (linear != (group.plan_linear != nullptr)) return false;
		plan.linear_srt = std::move(group.plan_linear);
		group.plan.emplace(std::move(plan));
		const auto end = std::chrono::steady_clock::now();
		prefetch->plan_read_ms += std::chrono::duration<double, std::milli>(read - begin).count();
		prefetch->plan_values_ms += std::chrono::duration<double, std::milli>(values - read).count();
		prefetch->plan_linear_ms += std::chrono::duration<double, std::milli>(end - values).count();
		const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
		prefetch->plans_loaded++;
		prefetch->plan_load_ms += ms;
		prefetch->plan_load_max_ms = std::max(prefetch->plan_load_max_ms, ms);
		if (std::has_single_bit(prefetch->plans_loaded) && prefetch->plans_loaded >= 64)
			PipelineCacheLog("Shader prefetch: {} plans loaded back from the temporary file in {:.1f} ms (at most {:.2f} ms; "
			                 "read {:.1f}, values {:.1f}, linear SRT {:.1f})",
			                 prefetch->plans_loaded, prefetch->plan_load_ms, prefetch->plan_load_max_ms, prefetch->plan_read_ms,
			                 prefetch->plan_values_ms, prefetch->plan_linear_ms);
		return true;
	}

	// What PrepareWarmJob does for a recorded input, for each of the program's.
	void PrefetchTranslate(Prefetch::Group& group) {
		auto& p = *prefetch;
		for (const auto index: group.records) {
			LocalShaderWarmup::Record record;
			LocalShaderWarmup::Reader reader {p.inputs.records[index]};
			if (!LocalShaderWarmup::Visit(reader, record)) continue;
			std::vector<uint32_t> user_data(record.user_data_count);
			const auto            options = LocalShaderWarmup::Options(record, user_data);
			try {
				// Inputs the recompiler rejects (code the game never runs) are left out, as the precompile did.
				Common::RecoverableExitScope recoverable;
				auto translated = ShaderRecompiler::TranslateProgram(record.code, options);
				if (!group.plan) group.plan.emplace(ShaderRecompiler::IR::ExtractResourcePlan(translated.program));
				ShaderRecompiler::IR::CanonicalizeSpecialization(translated.program.info, record.specialization);
				auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, record.specialization,
				                                               record.push_cursor);
				// Compressed (a fifth of the bytes): the file of a whole game's SPIR-V (4.5 GB) is kept in memory by the
				// system as a temporary file, which an 8 GB-GPU laptop with 32 GB of RAM did not have to spare.
				const auto  bytes      = result.spirv.size() * sizeof(uint32_t);
				const auto  packed     = ZstdPack(result.spirv.data(), bytes);
				const bool  compressed = !packed.empty() && packed.size() < bytes;
				const void* data       = compressed ? static_cast<const void*>(packed.data()) : result.spirv.data();
				const auto  stored     = compressed ? packed.size() : bytes;
				const auto  offset      = p.spill_end.fetch_add(stored, std::memory_order_relaxed);
				p.spirv_bytes.fetch_add(bytes, std::memory_order_relaxed);
				if (!LocalPlatform::WriteScratchFile(p.spill, offset, data, stored)) continue;
				group.modules.push_back({std::move(record.specialization), std::move(result.program).TakeCompiledInfo(),
				                         offset, result.spirv.size(), compressed ? stored : 0,
				                         XXH3_64bits(result.spirv.data(), bytes)});
				ShaderRecompiler::RecycleSpirv(std::move(result.spirv));
			} catch (const Common::RecoverableExit&) {
			}
		}
	}

	// The prefetched translation of the program `lookup_key` names, met for the first time: taken over
	// once done (waiting for it when a worker translates it now), else null and left to the caller.
	Prefetch::Group* PrefetchedGroup(const ShaderParams& params) {
		if (prefetch == nullptr || !prefetch->ready.load(std::memory_order_acquire)) return nullptr;
		const auto found = prefetch->index.find(lookup_key);
		if (found == prefetch->index.end()) return nullptr;
		auto& group = *found->second;
		prefetch->Follow(group.first_record);
		auto state = group.state.load(std::memory_order_acquire);
		if (state == Prefetch::Pending &&
		    group.state.compare_exchange_strong(state, Prefetch::Taken, std::memory_order_acq_rel)) {
			prefetch->claimed++;
			return nullptr;
		}
		for (; state == Prefetch::Running; state = group.state.load(std::memory_order_acquire))
			group.state.wait(Prefetch::Running, std::memory_order_acquire);
		if (state != Prefetch::Done || group.source != SourceDigest(params.code, params.back_code) || !LoadSpilledPlan(group))
			return nullptr;
		group.state.store(Prefetch::Taken, std::memory_order_relaxed);
		prefetch->adopted++;
		return &group;
	}

	// The prefetched translations of a program the warmup already holds (the recording's records it did not translate:
	// the program's other specializations) join its modules, once the prefetch's worker is done with them (pending or
	// running: left to it, the caller translates); false when none came.
	bool TakePrefetchedModules(SourceEntry& source, const ShaderParams& params) {
		if (prefetch == nullptr || !prefetch->ready.load(std::memory_order_acquire)) return false;
		const auto found = prefetch->index.find(lookup_key);
		if (found == prefetch->index.end()) return false;
		auto& group = *found->second;
		auto  state = static_cast<uint8_t>(Prefetch::Done);
		if (group.source != SourceDigest(params.code, params.back_code)) return false;
		if (!group.state.compare_exchange_strong(state, Prefetch::Taken, std::memory_order_acq_rel)) {
			prefetch->take_early += state != Prefetch::Taken;
			return false;
		}
		prefetch->taken++;
		for (auto& module: group.modules) source.prefetched.push_back(std::move(module));
		group.modules.clear();
		return !source.prefetched.empty();
	}

	// A prefetched module of the source compiled with the specialization a translation would compile
	// here (portable formats, see Get) and a push data start the cursor allows: its permutation.
	const Permutation* AdoptPrefetchedModule(SourceEntry& source, const ShaderRecompiler::CompileOptions& options,
	                                         ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                         uint32_t push_data_cursor) {
		if (source.prefetched.empty() || prefetch == nullptr) return nullptr;
		auto compiled = specialization;
		(void)ShaderRecompiler::IR::PortableFormats(source.resource_plan.info, compiled);
		// The exact specialization first (a module of the recording's records: they hold the exact one, as the warmup
		// compiles them), then the portable one (the static inputs': the precompile's modules).
		const auto with = [&](const ShaderRecompiler::IR::ResourceSpecialization& wanted) {
			return std::ranges::find_if(source.prefetched, [&](const PrefetchedModule& candidate) {
				const auto& layout = candidate.program.bindings;
				return candidate.specialization == wanted &&
				       layout.push_data_start_dword ==
				           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor, layout.ShaderDataDwords());
			});
		};
		auto module = with(specialization);
		if (module == source.prefetched.end()) module = with(compiled);
		if (module == source.prefetched.end()) return nullptr;
		if (const auto stored = CreateModuleInPlace(options, *module)) {
			source.permutations.push_back(
			    ModulePermutation(options, {}, std::move(module->program), std::move(specialization), stored));
			source.prefetched.erase(module);
			prefetch->adopted_modules++;
			return &source.permutations.back();
		}
		std::vector<uint32_t> spirv(module->spirv_words);
		const auto            bytes = spirv.size() * sizeof(uint32_t);
		if (module->spirv_bytes != 0) {
			std::vector<char> packed(module->spirv_bytes);
			if (!LocalPlatform::ReadScratchFile(prefetch->spill, module->spirv_offset, packed.data(), packed.size()) ||
			    !ZstdUnpack(packed, spirv.data(), bytes))
				return nullptr;
		} else if (!LocalPlatform::ReadScratchFile(prefetch->spill, module->spirv_offset, spirv.data(), bytes)) {
			return nullptr;
		}
		source.permutations.push_back(
		    ModulePermutation(options, spirv, std::move(module->program), std::move(specialization)));
		source.prefetched.erase(module);
		prefetch->adopted_modules++;
		return &source.permutations.back();
	}

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		return FinishPermutation(params, options, std::move(result), std::move(specialization));
	}

	// The device half of CompilePermutation.
	Permutation FinishPermutation(const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                              ShaderRecompiler::CompileResult              result,
	                              ShaderRecompiler::IR::ResourceSpecialization specialization) {
		auto info = std::move(result.program).TakeCompiledInfo();
		return FinishPermutation(params, options, result, std::move(info), std::move(specialization));
	}

	// On the worker that compiled `result`: its compiled info, and its IR freed there. Both walk every instruction
	// of the program (the BDA read plan, the destructors), up to ~0.2 s for the largest shaders, which the render
	// thread paid when it published a worker's permutation.
	static ShaderRecompiler::IR::CompiledShaderInfo TakeOnWorker(ShaderRecompiler::CompileResult& result) {
		auto                        info = std::move(result.program).TakeCompiledInfo();
		[[maybe_unused]] const auto ir   = std::move(result.program);
		return info;
	}

	// The device half of CompilePermutation for a result whose info TakeOnWorker took (the SPIR-V and the module
	// may come from a worker).
	Permutation FinishPermutation(const ShaderParams& params, const ShaderRecompiler::CompileOptions& options,
	                              const ShaderRecompiler::CompileResult&       result,
	                              ShaderRecompiler::IR::CompiledShaderInfo     info,
	                              ShaderRecompiler::IR::ResourceSpecialization specialization,
	                              vk::ShaderModule                             module = nullptr) {
		DumpShaderOriginal(StageName(options.stage), options.shader_hash, params.code, result.decoded_dump);
		auto permutation = ModulePermutation(options, result.spirv, std::move(info), std::move(specialization), module);
		if (options.dump_ir) {
			if (!options.early_dump) {
				LOGF("%s decoded RDNA2:\n%s", options.dump_label, result.decoded_dump.c_str());
				LOGF("%s IR:\n%s", options.dump_label, result.ir_dump.c_str());
			}
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return permutation;
	}

	static const char* StageName(ShaderType stage) {
		const char* name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: name = "vs"; break;
			case ShaderType::Mesh: name = "ms"; break;
			case ShaderType::Pixel: name = "ps"; break;
			case ShaderType::Compute: name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		return name;
	}

	// The shader module of a program's SPIR-V, on any thread (a compile worker makes a table mode program's).
	static vk::ShaderModule CreateModule(vk::Device device, const ShaderRecompiler::CompileOptions& options,
	                                     const std::vector<uint32_t>& spirv) {
		const char* stage_name = StageName(options.stage);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, spirv);
		// With the slow-call log: each module's SPIR-V (what the driver and static caches are keyed by).
		if (SlowLog::Threshold() > 0.0) {
			std::printf("MODULE %s hash=0x%016llx spirv=%016llx\n", stage_name,
			            static_cast<unsigned long long>(options.shader_hash),
			            static_cast<unsigned long long>(XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t))));
		}

		// (The store's, where pipelines take SPIR-V inline: moduleStore.h.)
		vk::ShaderModule module = ModuleStore::Add(spirv);
		const bool       stored = module != nullptr;
		if (!stored) {
			vk::ShaderModuleCreateInfo create_info {};
			create_info.codeSize = spirv.size() * sizeof(uint32_t);
			create_info.pCode    = spirv.data();
			RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
			                     "create recompiled shader module");
			EXIT_IF(module == nullptr);
		}
		LiveCounters::g_shader_modules.fetch_add(1, std::memory_order_relaxed);
		if (InlineKeyCheck()) NoteInlineSpirv(module, spirv);
		if (ModuleSpirvHashesRecorded()) {
			NotePipelineKeyModule(module, options.shader_hash, XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t)),
			                      spirv.size() * sizeof(uint32_t));
		}
		if (!stored) SetVulkanObjectNameF(device, module, "Kyty.Shader.{}[0x{:016x}]", stage_name, options.shader_hash);
		return module;
	}

	// CreateModule for a prefetched module: a stored module of its SPIR-V where the prefetch's file has it (no copy), or
	// null (the store is off; the SPIR-V validated, dumped or checked here).
	vk::ShaderModule CreateModuleInPlace(const ShaderRecompiler::CompileOptions& options, const PrefetchedModule& prefetched) {
		if (Config::ShaderValidationEnabled() || Config::GraphicsDebugDumpEnabled() || DumpWanted(options.shader_hash) ||
		    InlineKeyCheck() || prefetched.spirv_words > UINT32_MAX)
			return nullptr;
		const auto bytes  = prefetched.spirv_words * sizeof(uint32_t);
		const auto module = ModuleStore::AddFrom(
		    prefetch->spill, prefetched.spirv_offset,
		    static_cast<uint32_t>(prefetched.spirv_bytes != 0 ? prefetched.spirv_bytes : bytes),
		    static_cast<uint32_t>(prefetched.spirv_words), prefetched.spirv_bytes != 0);
		if (module == nullptr) return nullptr;
		prefetch->spill_in_store.store(true, std::memory_order_relaxed);
		if (SlowLog::Threshold() > 0.0) {
			std::printf("MODULE %s hash=0x%016llx spirv=%016llx\n", StageName(options.stage),
			            static_cast<unsigned long long>(options.shader_hash),
			            static_cast<unsigned long long>(prefetched.spirv_hash));
		}
		LiveCounters::g_shader_modules.fetch_add(1, std::memory_order_relaxed);
		if (ModuleSpirvHashesRecorded()) NotePipelineKeyModule(module, options.shader_hash, prefetched.spirv_hash, bytes);
		return module;
	}

	// A program's permutation and its shader module (`module`: made already, else here); its SPIR-V translated here,
	// by a warmup worker, the prefetch or a compile worker.
	Permutation ModulePermutation(const ShaderRecompiler::CompileOptions& options, const std::vector<uint32_t>& spirv,
	                              ShaderRecompiler::IR::CompiledShaderInfo     program,
	                              ShaderRecompiler::IR::ResourceSpecialization specialization,
	                              vk::ShaderModule                             module = nullptr) {
		if (module == nullptr) module = CreateModule(device, options, spirv);
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(program),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	// `table_mode`: the permutation compiled in table mode (ShaderRecompiler::IR::EnterTableMode), empty
	// when the program does not allow it.
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor, bool table_mode = false, uint32_t table_portable = 0) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.mesh.threads_num[0] != 0 ? ShaderType::Mesh : ShaderType::Vertex;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}

		Libs::LibKernel::Memory::g_srt_shader_hash = params.hash;
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = static_cast<uint32_t>(params.user_data.size());
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		if (entry != programs.end() && !entry->second.warm_code.empty()) {
			if (!std::ranges::equal(params.code, entry->second.warm_code) ||
			    !std::ranges::equal(params.back_code, entry->second.warm_back_code)) {
				std::erase_if(by_program, [&](const auto& item) { return item.second.first == &entry->second; });
				ForgetPlanLoads(entry->second);
				// (A warmup beside the game may be making pipelines of its permutations: kept until it is done.)
				if (warming) {
					retired.push_back(programs.extract(entry));
				} else {
					for (const auto& p : entry->second.permutations) {
						DestroyShaderModule(device, p.handle.module);
						if (p.handle.module) LiveCounters::g_shader_modules.fetch_sub(1, std::memory_order_relaxed);
					}
					programs.erase(entry);
				}
				entry = programs.end();
			} else {
				std::vector<uint32_t>().swap(entry->second.warm_code);
				std::vector<uint32_t>().swap(entry->second.warm_back_code);
			}
		}
		if (entry != programs.end()) EnsurePlan(entry->second);
		ShaderRecompiler::IR::ResourceSnapshot temporary_resources;
		auto& resources = NativePreparationScratchEnabled() ? input_info.stage.resources
		                                                    : temporary_resources;
		NativePreparationScratch<ShaderRecompiler::IR::ResourceSpecialization> specialization_storage;
		auto& specialization = specialization_storage.Get();
		const auto publish_stage = [&](const auto& program) {
			if (&resources != &input_info.stage.resources) {
				input_info.stage.resources = std::move(resources);
			}
			input_info.stage.program = &program;
		};
		const ShaderRecompiler::IR::SrtRuntime input_runtime {
		    .user_data                  = params.user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderRawGuestMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .sync_memory                = SyncShaderGuestMemory,
		    .try_read_memory_span       = ReadShaderMemorySpan,
		    .mapping_end                = ShaderMappingEnd,
		};
		ShaderReadObserver::Runtime observed_runtime(input_runtime);
		const auto& runtime = observed_runtime.Get();
		ShaderRecompiler::IR::MaterializeReport report;
		// Table mode reads every buffer's stride from the renderer (EmitTableMode) and decodes the formats the
		// specialization chose, as the specialized module does (the renderer checks every use's V#s against them,
		// TableBlocks), but for the buffers of `table_portable` (bits by buffer) whose V#s were seen to change format:
		// those decode the buffer word's (IR::PortableFormats). Its permutation is that, whatever the base's low bits.
		ShaderRecompiler::IR::ResourceSpecialization table_specialization;
		const auto table_form = [&](const ShaderRecompiler::IR::ResourceSpecialization& value) {
			table_specialization = value;
			const auto& buffers  = entry->second.resource_plan.info.buffers;
			for (size_t i = 0; i < table_specialization.buffers.size(); ++i) {
				auto& buffer            = table_specialization.buffers[i];
				buffer.byte_base_offset = false;
				if (i < 32 && ((table_portable >> i) & 1u) != 0 && i < buffers.size() &&
				    ShaderRecompiler::IR::RuntimeBufferFormat(buffers[i])) {
					buffer.descriptor_format  = Prospero::BufferFormat::kInvalid;
					buffer.descriptor_swizzle = DstSel(4, 5, 6, 7);
				}
			}
			// An indirect image with the table program's elements (its candidates' count is the draw's or dispatch's:
			// one program takes any), the snapshot it binds from padded alike (TableResolveSet).
			(void)ShaderRecompiler::IR::TableIndirectForm(table_specialization, &resources.images);
		};
		if (entry != programs.end()) {
			const ShaderRecompiler::IR::ResourceSpecialization* borrowed_specialization = nullptr;
			if (!ReportMaterialization(label, stage, params.hash, report,
			                           ShaderRecompiler::IR::MaterializeResources(
			                               entry->second.resource_plan, runtime, resources, specialization,
			                               &report, &entry->second.specialization_guard, &borrowed_specialization)))
				return {};
			if (table_mode) table_form(borrowed_specialization ? *borrowed_specialization : specialization);
			const auto& active_specialization =
			    table_mode ? table_specialization : borrowed_specialization ? *borrowed_specialization : specialization;
			const auto compatible_push_data = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(
				    push_data_cursor, layout.ShaderDataDwords());
			};
			const auto find_permutation = [&] { return std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        return candidate.table_mode == table_mode && compatible_push_data(candidate) &&
				               candidate.specialization == active_specialization;
			        }); };
			// A borrowed specialization comes from the guard: the permutation it
			// selected under the same publication is still the right one.
			auto& source = entry->second;
			if (!table_mode && borrowed_specialization && source.guarded_permutation &&
			    source.guarded_publication == source.specialization_guard.publication &&
			    compatible_push_data(*source.guarded_permutation)) {
				const auto* selected = source.guarded_permutation;
				publish_stage(selected->program);
				selected->program.bindings.AdvancePushData(push_data_cursor);
				return selected->handle;
			}
			if (const auto permutation = find_permutation();
			    permutation != entry->second.permutations.end()) {
				if (!permutation->handle) return {}; // table mode refused
				// Only a borrowed result certifies this exact guard publication.
				// An ineligible reference transaction must not relabel an old guard.
				if (borrowed_specialization && !table_mode) {
					source.guarded_permutation = &*permutation;
					source.guarded_publication = source.specialization_guard.publication;
				}
				publish_stage(permutation->program);
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
			if (borrowed_specialization) specialization = *borrowed_specialization;
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.enable_lod_stats = true;
		options.shader_hash = params.hash;
		options.user_data   = params.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::ShaderLogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;
		options.scratch_dwords = input_info.scratch_size_dwords;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			if (stage == ShaderType::Mesh) {
				options.user_data_base = 0;
				options.wave_size      = input_info.mesh.wave_size;
				options.scratch_dwords = input_info.mesh.scratch_size_dwords;
			}
		} else if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			options.wave_size = input_info.wave_size;
		}
		// Table mode: a program the normal path compiled first, in the table form of its specialization, translated
		// by a compile worker (table_pending until then); a refused one keeps an empty permutation.
		if (table_mode) {
			if (entry == programs.end()) return {};
			auto&      jobs  = entry->second.table_jobs;
			const auto found = std::ranges::find_if(jobs, [&](const auto& job) {
				return job->push_data_cursor == push_data_cursor && job->specialization == table_specialization;
			});
			if (found == jobs.end()) {
				auto job              = std::make_shared<TableJob>();
				job->specialization   = table_specialization;
				job->push_data_cursor = push_data_cursor;
				job->code.assign(params.code.begin(), params.code.end());
				job->back_code.assign(params.back_code.begin(), params.back_code.end());
				job->user_data         = params.user_data;
				job->options           = options;
				job->options.table_mode = true;
				job->options.user_data = job->user_data;
				job->options.back_code = job->back_code;
				job->options.input_info = {};
				if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
					job->vertex                    = input_info;
					job->options.input_info.vertex = &job->vertex;
				} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
					job->pixel                    = input_info;
					job->options.input_info.pixel = &job->pixel;
				} else {
					job->compute                    = input_info;
					job->options.input_info.compute = &job->compute;
				}
				jobs.push_back(job);
				push_job([job, device = device] {
					try {
						Common::RecoverableExitScope recoverable;
						auto translated = ShaderRecompiler::TranslateProgram(job->code, job->options);
						job->result     = ShaderRecompiler::CompileProgram(std::move(translated), job->options,
						                                                   job->specialization, job->push_data_cursor);
						job->module     = CreateModule(device, job->options, job->result->spirv);
						job->info       = TakeOnWorker(*job->result);
					} catch (const Common::RecoverableExit& refused) {
						job->refused = refused.message;
					}
					job->done.store(true, std::memory_order_release);
				});
				table_pending = true;
				table_waiting = std::shared_ptr<const std::atomic<bool>>(job, &job->done);
				return {};
			}
			const auto job = *found;
			if (!job->done.load(std::memory_order_acquire)) {
				table_pending = true;
				table_waiting = std::shared_ptr<const std::atomic<bool>>(job, &job->done);
				return {};
			}
			jobs.erase(found);
			Permutation permutation {.specialization = table_specialization, .table_mode = true};
			if (job->result) {
				const ShaderParams job_params {.code = job->code, .user_data = job->user_data, .hash = params.hash,
				                               .back_code = job->back_code};
				permutation = FinishPermutation(job_params, job->options, *job->result, std::move(job->info),
				                                ShaderRecompiler::IR::ResourceSpecialization {table_specialization},
				                                job->module);
				permutation.table_mode = true;
			} else {
				static std::atomic<uint32_t> reports {0};
				if (reports.fetch_add(1, std::memory_order_relaxed) < 200)
					std::printf("Table: %s 0x%016llx: %s", label, static_cast<unsigned long long>(params.hash),
					            job->refused.c_str());
			}
			const auto& stored = entry->second.permutations.emplace_back(std::move(permutation));
			if (!stored.handle) return {};
			publish_stage(stored.program);
			stored.program.bindings.AdvancePushData(push_data_cursor);
			return stored.handle;
		}
		// A program met for the first time that the prefetch translated: its plan, materialized for
		// this draw, and its modules are taken over.
		if (entry == programs.end()) {
			if (auto* group = PrefetchedGroup(params)) {
				if (!ReportMaterialization(label, stage, params.hash, report,
				                           ShaderRecompiler::IR::MaterializeResources(*group->plan, runtime, resources,
				                                                                      specialization, &report)))
					return {};
				entry = programs.try_emplace(lookup_key, std::move(*group->plan)).first;
				entry->second.source_digest = group->source;
				entry->second.prefetched = std::move(group->modules);
				entry->second.plan_spilled = {prefetch->spill, group->plan_offset, group->plan_bytes, group->plan_raw};
				entry->second.plan_use     = plan_clock;
				group->plan.reset();
			}
		}
		// (KYTY_PIPELINE_SOURCE_LOG: a PROGSRC line per permutation made here - adopted from the prefetch or translated -
		// with its time.)
		const auto source_begin = PipelineSourceLog() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		bool       translated_here = false;
		double     translate_ms    = 0.0;
		const auto adopt = [&] { return AdoptPrefetchedModule(entry->second, options, specialization, push_data_cursor); };
		if (entry == programs.end() ||
		    (adopt() == nullptr && (!TakePrefetchedModules(entry->second, params) || adopt() == nullptr))) {
			translated_here = true;
			// With the slow-call log: the translation's specialization-independent part (front) and
			// whether the program had other permutations already.
			const bool known_program = entry != programs.end();
			const auto front_begin   = std::chrono::steady_clock::now();
			double     front_ms      = 0.0;
			SlowLog::Scope translate_slow([&](double ms) {
				std::printf("SLOW TranslateProgram %.1f ms %s hash=0x%016llx front=%.1f ms %s\n", ms, label,
				            static_cast<unsigned long long>(params.hash), front_ms,
				            known_program ? "specialization" : "program");
			}, SlowLog::HitchThreshold());
			LiveCensus::WaitScope compiling(LiveCensus::WaitCompile);
			// (The render thread's own waits: the slow frame lines' translate/pipeline counts.)
			std::optional<LiveCounters::ScopedUs> translate_timed;
			if (LiveCensus::g_render) {
				LiveCounters::Add(LiveCounters::Translates);
				translate_timed.emplace(LiveCounters::TranslateUs);
			}
			auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
			front_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - front_begin).count();
			translate_ms = front_ms;
			if (entry == programs.end()) {
				auto resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				if (!ReportMaterialization(label, stage, params.hash, report,
				                           ShaderRecompiler::IR::MaterializeResources(
				                               resource_plan, runtime, resources, specialization, &report)))
					return {};
				entry = programs.try_emplace(lookup_key, std::move(resource_plan)).first;
				entry->second.plan_use      = plan_clock;
				entry->second.source_digest = SourceDigest(params.code, params.back_code);
			}
			// A specialization met for the first time whose formatted buffers can decode their formats at
			// run time gets the portable module (IR::PortableFormats) under its own specialization: the
			// static precompile (WarmSeeds) holds that module's pipelines, so no driver compilation holds
			// up this draw. The warmup file records the exact specialization, and the next start compiles
			// its specialized module.
			auto portable = specialization;
			if (ShaderRecompiler::IR::PortableFormats(entry->second.resource_plan.info, portable)) {
				entry->second.permutations.push_back(CompilePermutation(
				    params, options, std::move(translated), std::move(portable), push_data_cursor));
				entry->second.permutations.back().specialization = std::move(specialization);
			} else {
				entry->second.permutations.push_back(CompilePermutation(
				    params, options, std::move(translated), std::move(specialization), push_data_cursor));
			}
			render_translations++;
			render_translate_ms +=
			    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - front_begin).count();
		}
		const auto& permutation = entry->second.permutations.back();
		if (PipelineSourceLog())
			std::printf("PROGSRC %s hash=%016llx translated=%d translate_ms=%.1f ms=%.1f\n", label,
			            static_cast<unsigned long long>(params.hash), translated_here ? 1 : 0, translate_ms,
			            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - source_begin).count());
		if (warmup.Enabled()) {
			LocalShaderWarmup::Record record;
			record.stage = stage;
			record.hash = params.hash;
			record.user_data_count = static_cast<uint32_t>(params.user_data.size());
			record.push_cursor = push_data_cursor;
			record.static_key = lookup_key.static_state;
			record.code.assign(params.code.begin(), params.code.end());
			record.back_code.assign(params.back_code.begin(), params.back_code.end());
			record.specialization = permutation.specialization;
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) record.vertex = input_info;
			else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) record.pixel = input_info;
			else record.compute = input_info;
			const auto index = warmup.Add(record);
			if (index != LocalShaderWarmup::NoShader) recorded_programs[permutation.handle.id] = index;
		}
		publish_stage(permutation.program);
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::Mesh) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)]);
		return permutation.handle;
	}

	// Warmup recompiles on KYTY_SHADER_WARMUP_THREADS workers (default 1: this thread
	// only). Recompiling a record is pure, so workers translate a chunk to SPIR-V and make its module, and
	// TranslateWarmup registers the results in its order; a worker's result for a record it then skips is dropped.
	struct WarmJob {
		LocalShaderWarmup::Record          record;
		std::vector<uint32_t>              user_data;
		ShaderRecompiler::IR::ResourcePlan       plan;
		SpilledPlan                              spilled; // (its bytes, written by the worker: SweepPlans)
		ShaderRecompiler::CompileResult          result;
		ShaderRecompiler::IR::CompiledShaderInfo info;             // the result's, taken by the worker
		vk::ShaderModule                         module = nullptr; // the result's, made by the worker
	};

	// In front of the game KYTY_SHADER_WARMUP_THREADS (default 1); beside it KYTY_SHADER_WARMUP_BACKGROUND_THREADS
	// (default SpareThreads).
	static uint32_t WarmupThreads(bool background) {
		uint32_t threads = background ? SpareThreads() : 1;
		const char* name = background ? "KYTY_SHADER_WARMUP_BACKGROUND_THREADS" : "KYTY_SHADER_WARMUP_THREADS";
		if (const char* value = std::getenv(name); value && *value) {
			const auto end    = value + std::strlen(value);
			const auto parsed = std::from_chars(value, end, threads);
			if (parsed.ec != std::errc {} || parsed.ptr != end || threads == 0 || threads > 64)
				threads = background ? SpareThreads() : 1;
		}
		return threads;
	}

	static std::unique_ptr<WarmJob> PrepareWarmJob(vk::Device device, std::span<const uint32_t> saved) {
		auto                      job = std::make_unique<WarmJob>();
		LocalShaderWarmup::Reader reader {saved};
		if (!LocalShaderWarmup::Visit(reader, job->record) || !LocalShaderWarmup::ValidKey(job->record)) return nullptr;
		job->user_data.resize(job->record.user_data_count);
		const auto options = LocalShaderWarmup::Options(job->record, job->user_data);
		auto translated    = ShaderRecompiler::TranslateProgram(job->record.code, options);
		job->plan          = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (PlanEvictionEnabled()) (void)SpillPlanBytes(job->plan, job->spilled);
		// Recorded before buffer words (or without them): the permutation the game now selects.
		ShaderRecompiler::IR::CanonicalizeSpecialization(translated.program.info, job->record.specialization);
		job->result        = ShaderRecompiler::CompileProgram(std::move(translated), options, job->record.specialization,
		                                                      job->record.push_cursor);
		job->info          = TakeOnWorker(job->result);
		job->module        = CreateModule(device, options, job->result.spirv);
		return job;
	}

	// The module of a job whose record is skipped.
	static void DropWarmJob(vk::Device device, WarmJob& job) {
		DestroyShaderModule(device, job.module);
		if (job.module) LiveCounters::g_shader_modules.fetch_sub(1, std::memory_order_relaxed);
		job.module = nullptr;
	}

	// The order the warmup translates the records in: newest first (in front of the game: a time limit leaves the
	// oldest out), or beside it from both ends of the recording - its oldest records are what the game starts with
	// (the intro, the menus, the loading screens), its newest the place played last (where the save most likely is).
	static size_t WarmOrder(size_t n, size_t count, bool both_ends) {
		return both_ends ? (n % 2 == 0 ? n / 2 : count - 1 - n / 2) : count - 1 - n;
	}

	// Streams the records, in WarmOrder, through the workers; they stay at most Window records ahead because a result
	// holds a whole program. Beside the game (`background`) they run below normal priority.
	class WarmWorkers {
	public:
		WarmWorkers(vk::Device device, std::span<const std::span<const uint32_t>> records, uint32_t threads,
		            bool background)
		    : device(device), records(records), jobs(records.size()), ready(records.size()), background(background) {
			for (uint32_t t = 0; t < threads; ++t) pool.emplace_back([this] { Run(); });
		}
		~WarmWorkers() {
			stop.store(true, std::memory_order_relaxed);
			consumed.fetch_add(1, std::memory_order_release);
			consumed.notify_all();
			for (auto& thread: pool) thread.join();
			for (auto& job: jobs)
				if (job) DropWarmJob(device, *job); // (made, never taken: a stop or a time limit)
		}
		WarmWorkers(const WarmWorkers&)            = delete;
		WarmWorkers& operator=(const WarmWorkers&) = delete;
		// TranslateWarmup takes every index once, in WarmOrder; null for an invalid record.
		std::unique_ptr<WarmJob> Take(size_t index) {
			ready[index].wait(0, std::memory_order_acquire);
			consumed.fetch_add(1, std::memory_order_release);
			consumed.notify_all();
			return std::move(jobs[index]);
		}

	private:
		static constexpr size_t Window = 512;
		void Run() {
			if (background) {
				LocalPlatform::SetThreadName("Kyty.Warmup");
				LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
			}
			for (;;) {
				const size_t n = taken.fetch_add(1, std::memory_order_relaxed);
				if (n >= jobs.size()) return;
				for (size_t c = consumed.load(std::memory_order_acquire); n >= c + Window;
				     c = consumed.load(std::memory_order_acquire)) {
					if (stop.load(std::memory_order_relaxed)) return;
					consumed.wait(c, std::memory_order_acquire);
				}
				if (stop.load(std::memory_order_relaxed)) return;
				const size_t index = WarmOrder(n, jobs.size(), background);
				jobs[index]        = PrepareWarmJob(device, records[index]);
				ready[index].store(1, std::memory_order_release);
				ready[index].notify_one();
			}
		}
		vk::Device                                 device;
		std::span<const std::span<const uint32_t>> records;
		std::vector<std::unique_ptr<WarmJob>>      jobs;
		std::vector<std::atomic<uint8_t>>          ready;
		bool                                       background = false;
		std::atomic<size_t>                        taken {0};
		std::atomic<size_t>                        consumed {0};
		std::atomic<bool>                          stop {false};
		std::vector<std::thread>                   pool;
	};

	// The recording (`path`; or `adopt_from`'s, while this device signature has none): the shader inputs and pipeline
	// recipes the warmup replays (TranslateWarmup, PipelineCache::WarmPipelines). It reads them as warm_records and
	// warm_recipes, since the game's recording appends to the cache's vectors meanwhile (Get, AddPipeline): a record's
	// words stay where they are (growing the vector moves each record's buffer along, not its words).
	void LoadWarmup(const std::filesystem::path& path, const std::string& identity,
	                const std::filesystem::path& adopt_from = {}) {
		if (!warmup.Open(path, identity))
			PipelineCacheLog("Shader warmup: ignoring invalid cache {}", Common::PathToString(path));
		if (!adopt_from.empty() && warmup.records.empty()) {
			if (warmup.Adopt(adopt_from))
				PipelineCacheLog("Shader warmup: adopted {} inputs and {} pipeline recipes from {}", warmup.records.size(),
				                 warmup.pipelines.size(), Common::PathToString(adopt_from));
			else PipelineCacheLog("Shader warmup: could not adopt {}", Common::PathToString(adopt_from));
		}
		PipelineCacheLog("Shader warmup: loaded {} shader inputs and {} pipeline recipes", warmup.records.size(), warmup.pipelines.size());
		warm_records.assign(warmup.records.begin(), warmup.records.end());
		warm_recipes = warmup.pipelines;
	}

	// The recording's records translated (warm_permutations: each record's permutation). In front of the game
	// (KYTY_SHADER_WARMUP_BACKGROUND=0, a -Precompile run): newest first, until KYTY_SHADER_WARMUP_SECONDS or
	// KYTY_SHADER_WARMUP_TRANSLATE_SECONDS. Beside it (`background`): every record in WarmOrder, each registered under
	// `mutex` (the pipeline cache's) by this thread alone - a worker below normal priority holding it while the game's
	// threads kept every CPU would hold up the render thread. The game's Get may have made the program meanwhile, or
	// the permutation: a record then adds what the program lacks, of the same code only.
	// The permutation TranslateWarmup made (or found) for a recorded input, while the cache still has it; null: the
	// program left (its code changed) or never had it. Under the pipeline cache's mutex, `warming` set while it is used.
	Permutation* FindRecorded(const LocalShaderWarmup::Record& record) {
		const ProgramKey key {record.stage, record.hash, record.user_data_count, static_cast<uint32_t>(record.code.size()),
		                      record.static_key};
		const auto entry = programs.find(key);
		if (entry == programs.end() || entry->second.source_digest != SourceDigest(record.code, record.back_code)) return nullptr;
		const auto found = std::ranges::find_if(entry->second.permutations, [&](const auto& p) {
			return !p.table_mode && p.specialization == record.specialization &&
			       p.program.bindings.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(record.push_cursor, p.program.bindings.ShaderDataDwords());
		});
		return found != entry->second.permutations.end() ? &*found : nullptr;
	}

	void TranslateWarmup(Common::Mutex& mutex, bool background, const std::atomic<bool>& stop) {
		const auto begin = std::chrono::steady_clock::now();
		// 0 means all recorded inputs. A caller can explicitly bound startup time.
		uint32_t budget_seconds = 0;
		if (const char* value = std::getenv("KYTY_SHADER_WARMUP_SECONDS"); value && *value) {
			const auto end = value + std::strlen(value);
			const auto parsed = std::from_chars(value, end, budget_seconds);
			if (parsed.ec != std::errc {} || parsed.ptr != end || budget_seconds > 3600) budget_seconds = 0;
		}
		// (It bounds the wait in front of the game; beside it, the warmup goes on: what it does not make, the game makes
		// where first needed.)
		if (budget_seconds && !background) warm_deadline = begin + std::chrono::seconds(budget_seconds);
		// In front of the game, the translation alone (KYTY_SHADER_WARMUP_TRANSLATE_SECONDS; 0, the default: no limit of
		// its own). 5 s of the newest records, the rest left to the shader prefetch and their pipelines to their first
		// use, cost a 16-thread desktop's tour 6 render-thread hitches of 15-127 ms.
		uint32_t translate_seconds = 0;
		if (const char* value = std::getenv("KYTY_SHADER_WARMUP_TRANSLATE_SECONDS"); value && *value) {
			const auto end    = value + std::strlen(value);
			const auto parsed = std::from_chars(value, end, translate_seconds);
			if (parsed.ec != std::errc {} || parsed.ptr != end || translate_seconds > 3600) translate_seconds = 0;
		}
		const char* only               = std::getenv("KYTY_SHADER_WARMUP_ONLY");
		auto        translate_deadline = background ? std::chrono::steady_clock::time_point::max() : warm_deadline;
		if (!background && translate_seconds != 0 && (only == nullptr || std::string_view(only) != "1"))
			translate_deadline = std::min(translate_deadline, begin + std::chrono::seconds(translate_seconds));
		const size_t count = warm_records.size();
		warm_permutations.assign(count, nullptr);
		warm_total.store(count, std::memory_order_relaxed);
		warm_done.store(0, std::memory_order_relaxed);
		if (background) warm_phase.store(1, std::memory_order_release);
		size_t compiled = 0, left = 0;
		const uint32_t threads = WarmupThreads(background);
		std::optional<WarmWorkers> workers;
		if (threads > 1 || background) workers.emplace(device, warm_records, threads, background);
		for (size_t n = 0; n < count; ++n) {
			if (stop.load(std::memory_order_relaxed) || std::chrono::steady_clock::now() >= translate_deadline) {
				left = count - n;
				break;
			}
			if (!background) StartupProgress::Report("Preparing shaders", n, count);
			warm_done.store(n, std::memory_order_relaxed);
			const size_t index = WarmOrder(n, count, background);
			const auto   job   = workers ? workers->Take(index) : PrepareWarmJob(device, warm_records[index]);
			if (!job) continue;
			auto&          record = job->record;
			const uint64_t digest = SourceDigest(record.code, record.back_code);
			// The compiler consumes the user-data width; values are resolved anew by
			// MaterializeResources when the guest actually draws or dispatches.
			std::vector<uint32_t> user_data = std::move(job->user_data);
			auto                  options   = LocalShaderWarmup::Options(record, user_data);
			ProgramKey            key {record.stage, record.hash, record.user_data_count,
			                           static_cast<uint32_t>(record.code.size()), record.static_key};
			Common::LockGuard lock(mutex);
			auto entry = programs.find(key);
			if (entry != programs.end() && entry->second.source_digest != digest) {
				DropWarmJob(device, *job);
				continue;
			}
			if (entry != programs.end()) {
				const auto found = std::ranges::find_if(entry->second.permutations, [&](const auto& p) {
					return !p.table_mode && p.specialization == record.specialization &&
					       p.program.bindings.push_data_start_dword ==
					           ShaderRecompiler::IR::PushData::StartFor(record.push_cursor, p.program.bindings.ShaderDataDwords());
				});
				if (found != entry->second.permutations.end()) {
					warm_permutations[index] = &*found;
					DropWarmJob(device, *job);
					continue;
				}
			} else {
				warm_keys.push_back(key);
				entry = programs.try_emplace(std::move(key), std::move(job->plan)).first;
				entry->second.plan_spilled   = job->spilled;
				entry->second.plan_use       = plan_clock;
				entry->second.warm_code      = record.code;
				entry->second.warm_back_code = record.back_code;
				entry->second.source_digest  = digest;
			}
			ShaderParams params {record.code, std::move(user_data), record.hash, record.back_code};
			entry->second.permutations.push_back(FinishPermutation(params, options, job->result, std::move(job->info),
			                                                       std::move(record.specialization), job->module));
			job->module       = nullptr;
			auto& permutation = entry->second.permutations.back();
			recorded_programs[permutation.handle.id] = uint32_t(index);
			warm_permutations[index] = &permutation;
			if (++compiled % 1000 == 0) PipelineCacheLog("Shader warmup: compiling {}/{}", compiled, count);
		}
		workers.reset();
		PipelineCacheLog("Shader warmup: compiled {}/{} permutations in {} ms ({} threads{}); {} records left to the shader "
		    "prefetch", compiled, count,
		    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
		    threads, background ? ", beside the game" : "", left);
	}
	LocalShaderWarmup::Cache warmup;
	// The warmup's view of the recording (LoadWarmup), each record's permutation (TranslateWarmup) and the keys of the
	// programs it made (the shader prefetch leaves their static inputs out); all freed once it is done.
	std::vector<std::span<const uint32_t>> warm_records;
	std::vector<std::vector<uint32_t>>     warm_recipes;
	std::vector<ProgramKey>                warm_keys;
	// A warmup beside the game, for the HUD (PipelineCache::GetPrefetchProgress): its phase (0: none, 1: shaders, 2:
	// pipelines) and progress.
	std::atomic<uint32_t> warm_phase {0};
	std::atomic<size_t>   warm_done {0}, warm_total {0};
	std::unordered_map<uint64_t, uint32_t> recorded_programs;
	std::vector<Permutation*> warm_permutations;
	std::chrono::steady_clock::time_point warm_deadline = std::chrono::steady_clock::time_point::max();
	uint32_t RecordedIndex(const ShaderProgram& program) const {
		const auto it = recorded_programs.find(program.id);
		return it == recorded_programs.end() ? LocalShaderWarmup::NoShader : it->second;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		ReleaseRetired();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				DestroyShaderModule(device, permutation.handle.module);
				if (permutation.handle.module) LiveCounters::g_shader_modules.fetch_sub(1, std::memory_order_relaxed);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	// While a warmup runs beside the game (PipelineCache::RunWarmup): programs whose recorded code the game no longer
	// has (Get) wait here, their modules alive, until it is done (it may be making their pipelines).
	bool                                       warming = false;
	std::vector<decltype(programs)::node_type> retired;
	void ReleaseRetired() {
		for (auto& node: retired)
			for (const auto& permutation: node.mapped().permutations) {
				DestroyShaderModule(device, permutation.handle.module);
				if (permutation.handle.module) LiveCounters::g_shader_modules.fetch_sub(1, std::memory_order_relaxed);
			}
		retired.clear();
	}
	// Programs Get translated itself (none the warmup or the prefetch had), and the time it took.
	uint64_t render_translations = 0;
	double   render_translate_ms = 0.0;
	// Native XPR re-evaluation: compiled program -> its source and permutation.
	std::unordered_map<const ShaderRecompiler::IR::CompiledShaderInfo*,
	                   std::pair<SourceEntry*, const Permutation*>>
	    by_program;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
};

// A pipeline a CompileWorkers thread creates; `done` publishes the finished object.
struct PipelineCache::PendingComputePipeline {
	std::unique_ptr<Pipeline> pipeline = std::make_unique<Pipeline>();
	ShaderComputeInputInfo    input_info;
	vk::ShaderModule          module = nullptr;
	std::atomic<bool>         done {false};
};

struct PipelineCache::PendingGraphicsPipeline {
	std::unique_ptr<Pipeline> pipeline = std::make_unique<Pipeline>();
	PipelineRenderingState    rendering;
	PipelineVertexInputState  vertex_input;
	PipelineStaticParameters  static_params;
	ShaderVertexInputInfo     vs_input_info;
	ShaderPixelInputInfo      ps_input_info;
	bool                      ps_active = false;
	ShaderProgram             vertex_program;
	ShaderProgram             pixel_program;
	bool                      native_bindings = false;
	std::atomic<bool>         done {false};
};

// Threads that create pipelines off the render thread (vkCreate* calls are thread-safe, and the
// Vulkan recording layer only intercepts the render thread's own calls).
class PipelineCache::CompileWorkers {
public:
	// background: below-normal priority off the render CPUs, and jobs still queued at shutdown are dropped.
	explicit CompileWorkers(uint32_t count, bool background = false): m_background(background) {
		for (uint32_t i = 0; i < count; i++)
			m_threads.emplace_back([this] {
				if (m_background) LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
				Run();
			});
	}
	~CompileWorkers() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) thread.join();
	}
	KYTY_CLASS_NO_COPY(CompileWorkers);
	void Push(std::function<void()> job) {
		{
			std::lock_guard lock(m_mutex);
			m_jobs.push_back(std::move(job));
		}
		m_wake.notify_one();
	}

private:
	void Run() {
		for (;;) {
			std::function<void()> job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
				// Queued jobs still run at shutdown: their pipelines are destroyed with the rest.
				if (m_jobs.empty() || (m_stop && m_background)) return;
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			job();
		}
	}

	std::mutex                        m_mutex;
	std::condition_variable           m_wake;
	std::deque<std::function<void()>> m_jobs;
	bool                              m_stop = false;
	const bool                        m_background;
	std::vector<std::thread>          m_threads;
};

PipelineCache::CompileWorkers& PipelineCache::Workers() {
	if (m_compile_workers == nullptr) {
		// The native XPR variants a new area brings (hundreds, ~100 ms each in no cache, unoptimized or not): two
		// threads queued them for seconds (their draws on the normal path meanwhile). Background threads.
		// Half the CPUs on a CPU of 24 threads or more (a quarter before): Boletaria brought ~850 variants of ~145 ms
		// in 10 minutes on a 32-thread CPU, 173 native stores a slow frame waiting for them, while the background
		// threads left most of the CPU idle. KYTY_COMPILE_WORKERS=<n> sets the count.
		const uint32_t cpus  = std::max(1u, std::thread::hardware_concurrency());
		uint32_t       count = std::max(2u, cpus >= 24 ? cpus / 2u : cpus / 4u);
		if (const char* value = std::getenv("KYTY_COMPILE_WORKERS"); value != nullptr && *value != '\0')
			count = static_cast<uint32_t>(std::clamp(std::atoi(value), 1, 64));
		m_compile_workers = std::make_unique<CompileWorkers>(count, true);
	}
	return *m_compile_workers;
}

PipelineCache::CompileWorkers& PipelineCache::TableWorkers() {
	if (m_table_workers == nullptr) {
		// As many as the shader prefetch's (background threads: they take no time the game's threads want).
		m_table_workers = std::make_unique<CompileWorkers>(std::max(2u, std::thread::hardware_concurrency() / 2u), true);
	}
	return *m_table_workers;
}

PipelineCache::CompileWorkers& PipelineCache::OptimizeWorkers() {
	if (m_optimize_workers == nullptr) {
		m_optimize_workers = std::make_unique<CompileWorkers>(std::max(2u, std::thread::hardware_concurrency() / 4u), true);
	}
	return *m_optimize_workers;
}

// A pipeline no cache holds: compiled unoptimized now (VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) and optimized on a
// worker, or optimized before its first use (up to seconds on NVIDIA). Unoptimized first only on NVIDIA, where it was
// measured (~100x faster to compile) and tested. (An AMD RX 6700 XT's driver crash at the first compute pipelines,
// which kept AMD out, was native wave64 on RDNA2: the unoptimized build was not the cause. Whether AMD's driver
// compiles faster unoptimized is not measured yet: an RX 9070 XT took 18 s for each of four compute pipelines.)
// KYTY_PIPELINE_FAST_BUILD=1 / 0 chooses either on any GPU.
static PipelineBuild FirstBuild(const GraphicContext& graphics) {
	static const int fast = [] {
		const char* value = std::getenv("KYTY_PIPELINE_FAST_BUILD");
		return value == nullptr ? -1 : (std::string_view(value) != "0" ? 1 : 0);
	}();
	constexpr uint32_t NvidiaVendor = 0x10DE;
	const bool         nvidia       = graphics.GetPhysicalDeviceProperties().vendorID == NvidiaVendor;
	return fast == 1 || (fast == -1 && nvidia) ? PipelineBuild::Fast : PipelineBuild::Full;
}

// An unoptimized pipeline (PipelineBuild::Fast): `build` compiles the optimized one on a worker, into
// a copy with the same layouts; PromoteOptimized() puts it in place.
void PipelineCache::BuildOptimized(Pipeline& pipeline, std::function<void(Pipeline&)> build) {
	if (!pipeline.unoptimized) return;
	m_unoptimized_builds++;
	m_optimizing.push_back(&pipeline);
	pipeline.optimized = std::make_shared<OptimizedBuild>();
	auto layouts       = pipeline;
	layouts.pipeline   = nullptr;
	layouts.optimized  = nullptr;
	OptimizeWorkers().Push([this, optimized = pipeline.optimized, layouts, build = std::move(build)]() mutable {
		if (!m_stopping.load(std::memory_order_relaxed)) build(layouts);
		optimized->pipeline = layouts.pipeline;
		optimized->done.store(true, std::memory_order_release);
		m_optimized_builds.fetch_add(1, std::memory_order_release);
		MainPipelineDone();
	});
}

void PipelineCache::PromoteFinished() {
	Common::LockGuard lock(m_mutex);
	PromoteOptimized();
	m_program_cache->SweepPlans();
	EvictIdlePipelines();
}

namespace {
// The flips (EvictIdlePipelines) and what eviction did (ReportSession).
std::atomic<uint32_t> g_pipeline_flip {1};
struct {
	std::atomic<uint64_t> kept {0}, evicted {0}, remade {0}, remake_us {0}, remake_max_us {0}, scene {0};
} g_pipeline_evictions;

// KYTY_PIPELINE_EVICT: 0 keeps every pipeline, 2 (a test) evicts whatever the video memory in use; set, it evicts on any
// GPU, else only on one short of video memory (GraphicContext::small_video_memory, under 12 GiB). KYTY_PIPELINE_EVICT_SECONDS:
// the idle time (default 180 s, at least 5).
struct EvictSettings {
	bool                     set    = false;
	bool                     on     = true;
	bool                     always = false;
	std::chrono::nanoseconds idle   = std::chrono::seconds(180);
};
const EvictSettings& PipelineEvictSettings() {
	static const EvictSettings settings = [] {
		EvictSettings out;
		const char*   evict   = std::getenv("KYTY_PIPELINE_EVICT");
		const char*   seconds = std::getenv("KYTY_PIPELINE_EVICT_SECONDS");
		out.set               = evict != nullptr && *evict != '\0';
		out.on                = !out.set || std::string_view(evict) != "0";
		out.always            = out.set && std::string_view(evict) == "2";
		if (seconds != nullptr && *seconds != '\0')
			out.idle = std::chrono::seconds(std::max<uint64_t>(std::strtoull(seconds, nullptr, 10), 5));
		return out;
	}();
	return settings;
}

// A pipeline made now: the flip it was made at as its last use (not idle before a first binding); and whether it keeps
// how to make it again (made of the static store's, where it may be evicted: only there, for the inputs it keeps).
bool KeepsRemake(GraphicContext& graphics, PipelineCache::Pipeline& pipeline, bool from_store) {
	const auto& settings = PipelineEvictSettings();
	pipeline.last_flip   = g_pipeline_flip.load(std::memory_order_relaxed);
	// (Where the game reports its worlds, on any GPU: the worlds it leaves take their pipelines along.)
	const bool scenes = SceneTags::Generation() != 0;
	if (!settings.on || (!settings.set && !graphics.small_video_memory && !scenes) || !from_store) return false;
	g_pipeline_evictions.kept.fetch_add(1, std::memory_order_relaxed);
	return true;
}
} // namespace

// What a graphics pipeline is made of, as its remake keeps it (~11 KB: the vertex input info's 32 resource slots), without
// the draw's resource values (ShaderStageRuntime::resources: no part of the pipeline). Cold memory once made.
struct PipelineCache::GraphicsMakeInputs {
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;
	ShaderVertexInputInfo    vs_input_info;
	ShaderPixelInputInfo     ps_input_info;
	ShaderProgram            vertex_program;
	ShaderProgram            pixel_program;
	bool                     ps_active       = false;
	bool                     native_bindings = false;
	bool                     table           = false; // the table pipelines' driver cache
};

std::shared_ptr<const std::function<void(PipelineCache::Pipeline&)>>
PipelineCache::GraphicsRemake(GraphicsMakeInputs inputs) {
	inputs.vs_input_info.stage.resources = {};
	inputs.ps_input_info.stage.resources = {};
	return std::make_shared<const std::function<void(Pipeline&)>>([this, in = std::move(inputs)](Pipeline& target) {
		CreatePipelineInternal(m_graphics, target, in.rendering, in.vertex_input, in.vs_input_info, in.vertex_program,
		                       in.ps_active ? &in.ps_input_info : nullptr, in.pixel_program, in.static_params,
		                       in.table ? TablePipelineCache() : m_driver_cache, in.native_bindings, PipelineBuild::Full);
	});
}

vk::Pipeline BindPipelineHandle(const PipelineCache::Pipeline& pipeline) {
	pipeline.last_flip = g_pipeline_flip.load(std::memory_order_relaxed);
	if (pipeline.recipe != 0) {
		if (const auto generation = SceneTags::Generation(); pipeline.scene_noted != generation) {
			pipeline.scene_noted = generation;
			SceneTags::NoteUse(pipeline.recipe);
		}
	}
	if (pipeline.pipeline != nullptr || pipeline.remake == nullptr) return pipeline.pipeline;
	// Made again as it was made (its layouts are the shared ones for its bindings: the same), from the static store.
	const auto              start = std::chrono::steady_clock::now();
	PipelineCache::Pipeline made;
	(*pipeline.remake)(made);
	EXIT_IF(made.pipeline == nullptr || made.pipeline_layout != pipeline.pipeline_layout ||
	        made.descriptor_set_layout != pipeline.descriptor_set_layout);
	const_cast<PipelineCache::Pipeline&>(pipeline).pipeline = made.pipeline;
	const auto us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
	g_pipeline_evictions.remade.fetch_add(1, std::memory_order_relaxed);
	g_pipeline_evictions.remake_us.fetch_add(us, std::memory_order_relaxed);
	if (us > g_pipeline_evictions.remake_max_us.load(std::memory_order_relaxed))
		g_pipeline_evictions.remake_max_us.store(us, std::memory_order_relaxed);
	return pipeline.pipeline;
}

// With the video memory in use within EvictMargin of the GPU's budget (checked every CheckFlips flips), up to MaxEvictions
// a check of the pipelines the static store gave that no binding used for KYTY_PIPELINE_EVICT_SECONDS (default 180, at
// least 5) leave it: their handles destroyed (BindPipelineHandle makes one again from the store, ~1-2 ms). Their shader
// code is video memory: on an 8 GB GPU, a 5-minute 1-1 tour took the driver's share 0.6 -> 1.0 GB while the video memory
// in use went past the budget (7.7 GB of 7.25); evicting every pipeline idle for 5 s there (a test: 9,946 evicted, 6,188
// made again at 1.1 ms each, fps unchanged) kept it at 0.8 GB. Not those an optimized build will replace, and none while
// speculative translation records ahead (KYTY_SPECULATE). The time since a pipeline's last binding is what keeps its
// handle out of commands still to run: GPU work is never minutes behind. Only pipelines in this cache's maps: a worker's
// result enters them on this thread, its remake set before (PipelineEvictSettings, KeepsRemake).
void PipelineCache::EvictIdlePipelines() {
	const auto& settings = PipelineEvictSettings();
	constexpr uint32_t CheckFlips   = 60;
	constexpr size_t   MaxEvictions = 256;
	constexpr uint64_t EvictMargin  = uint64_t {512} << 20u;
	const uint32_t     flip         = g_pipeline_flip.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto         now          = std::chrono::steady_clock::now();
	m_flip_times[flip % m_flip_times.size()] = now.time_since_epoch().count();
	// A world the game left (SceneTags::Enter): once the one it went to was played for SceneSettle, the pipelines no
	// binding used since the change - and none of the new world's recorded ones, which the warmup made for its places not
	// reached yet - leave whatever the video memory, a pass a check until one finds none. Each held ~220 KB of the NVIDIA
	// driver's RAM and video memory for its code, until the session ended. 16 GB policy, 1-1 -> five worlds -> 1-1, 60 s
	// in each: the driver's pools 0.87 -> 2.01 GB kept, 0.84 -> 0.98 GB with those pipelines evicted (private 4.2 -> 2.8
	// GB more); a pipeline evicted and bound again is made from the static store (~0.4 ms there). Where the warmup made
	// every recorded pipeline (a world it did not know yet), those of the other worlds go too.
	constexpr auto SceneSettle = std::chrono::seconds(60);
	if (const auto generation = SceneTags::Generation(); generation != m_scene_generation) {
		m_scene_generation = generation;
		m_scene_flip       = flip;
		m_scene_since      = now;
		m_scene_swept      = false;
	}
	if (!settings.on || flip % CheckFlips != 0 || kyty_local_speculate_mode.load(std::memory_order_relaxed) != 0) return;
	const bool scene_left = !m_scene_swept && now - m_scene_since >= SceneSettle;
	uint64_t video = 0, budget = 0;
	const bool known = m_graphics.QueryLocalVideoMemory(&video, &budget);
	m_evict_pressure = known && (settings.always || video + EvictMargin >= budget);
	if (!m_evict_pressure && !scene_left) return;
	// The newest flip that began by the horizon among those the ring holds; none when even the oldest began after it.
	uint32_t below = 0; // bound in an earlier flip: idle since then
	if (m_evict_pressure) {
		const auto horizon = (now - settings.idle).time_since_epoch().count();
		uint32_t   first   = flip - std::min<uint32_t>(flip - 1, static_cast<uint32_t>(m_flip_times.size() - 1)), last = flip;
		if (m_flip_times[first % m_flip_times.size()] <= horizon) {
			while (first < last) {
				const uint32_t middle = first + (last - first + 1) / 2;
				if (m_flip_times[middle % m_flip_times.size()] <= horizon) first = middle;
				else last = middle - 1;
			}
			below = last;
		}
	}
	std::unordered_set<uint64_t> scene_recipes;
	if (scene_left) scene_recipes = SceneTags::Recipes(SceneTags::Current());
	size_t     evicted = 0, left = 0;
	const auto evict   = [&](Pipeline& pipeline) {
		if (evicted == MaxEvictions || pipeline.pipeline == nullptr || pipeline.remake == nullptr || pipeline.unoptimized ||
		    pipeline.optimized != nullptr)
			return;
		const bool idle  = pipeline.last_flip < below;
		const bool other = scene_left && pipeline.last_flip < m_scene_flip && !scene_recipes.contains(pipeline.recipe);
		if (!idle && !other) return;
		DeviceFault::ForgetPipeline(static_cast<VkPipeline>(pipeline.pipeline));
		m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		pipeline.pipeline = nullptr;
		evicted++;
		left += idle ? 0 : 1;
	};
	for (auto& [key, pipeline]: m_graphics_pipelines) evict(*pipeline);
	for (auto& [id, pipeline]: m_compute_pipelines) evict(*pipeline);
	if (scene_left && evicted < MaxEvictions) m_scene_swept = true;
	g_pipeline_evictions.scene.fetch_add(left, std::memory_order_relaxed);
	if (scene_left && m_scene_swept) {
		std::printf("Pipelines of the worlds left (%s for %lld s): %llu evicted in all, %llu made again (%.0f ms)\n",
		            SceneTags::Name(SceneTags::Current()).c_str(),
		            static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(now - m_scene_since).count()),
		            static_cast<unsigned long long>(g_pipeline_evictions.scene.load(std::memory_order_relaxed)),
		            static_cast<unsigned long long>(g_pipeline_evictions.remade.load(std::memory_order_relaxed)),
		            static_cast<double>(g_pipeline_evictions.remake_us.load(std::memory_order_relaxed)) / 1000.0);
		std::fflush(stdout);
	}
	if (evicted == left) {
		(void)g_pipeline_evictions.evicted.fetch_add(evicted, std::memory_order_relaxed);
		return;
	}
	const auto total = g_pipeline_evictions.evicted.fetch_add(evicted, std::memory_order_relaxed) + evicted;
	// (A line at most every 30 s: the totals so far.)
	static auto logged = std::chrono::steady_clock::time_point {};
	if (now - logged < std::chrono::seconds(30)) return;
	logged = now;
	std::printf("Pipelines out of video memory while idle: %llu evicted, %llu made again (%.0f ms); video memory %llu of %llu "
	            "MiB\n",
	            static_cast<unsigned long long>(total),
	            static_cast<unsigned long long>(g_pipeline_evictions.remade.load(std::memory_order_relaxed)),
	            static_cast<double>(g_pipeline_evictions.remake_us.load(std::memory_order_relaxed)) / 1000.0,
	            static_cast<unsigned long long>(video >> 20u), static_cast<unsigned long long>(budget >> 20u));
	std::fflush(stdout);
}

// Finished optimized builds replace their pipelines in place, for every holder of the Pipeline
// (native XPR records never look it up again). The unoptimized pipeline stays alive until this
// cache is destroyed: commands recorded before may still use it.
void PipelineCache::PromoteOptimized() {
	const auto finished = m_optimized_builds.load(std::memory_order_acquire);
	if (finished == m_promoted_builds) return;
	m_promoted_builds = finished;
	std::erase_if(m_optimizing, [&](Pipeline* pipeline) {
		const auto& build = *pipeline->optimized;
		if (!build.done.load(std::memory_order_acquire)) return false;
		if (build.pipeline != nullptr) {
			m_replaced_pipelines.push_back(pipeline->pipeline);
			pipeline->pipeline    = build.pipeline;
			pipeline->unoptimized = false;
		}
		pipeline->optimized.reset();
		return true;
	});
}

// A table mode pipeline's compile finished (the table cache saver waits for them to settle).
void PipelineCache::TablePipelineDone() {
	m_table_pipeline_last_done.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
	m_table_pipelines_done.fetch_add(1, std::memory_order_release);
	m_table_pipeline_jobs.fetch_sub(1, std::memory_order_release);
}

// A pipeline was created with the main driver cache (the saver saves it once such compiles settle).
void PipelineCache::MainPipelineDone() {
	m_main_pipeline_last_done.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
	m_main_pipelines_done.fetch_add(1, std::memory_order_release);
}

// What the session's shader prefetch, first pipeline builds and shader plans did, once: at the window's close (the
// process may end without this cache's destructor: the lines were never written, issue #20) or in the destructor.
void PipelineCache::ReportSession() {
	Common::LockGuard lock(m_mutex);
	if (m_session_reported) return;
	m_session_reported = true;
	if (const auto& prefetch = m_program_cache->prefetch; prefetch != nullptr && prefetch->ready.load(std::memory_order_acquire)) {
		PipelineCacheLog("Shader prefetch: {} plans loaded back from the temporary file in {:.1f} ms (at most {:.2f} ms)",
		                 prefetch->plans_loaded, prefetch->plan_load_ms, prefetch->plan_load_max_ms);
		PipelineCacheLog("Shader prefetch: {} of {} programs translated ({} of them following play); the game took over {} "
		                 "of them ({} modules) and translated {} itself",
		                 prefetch->finished.load(), prefetch->groups.size(), prefetch->followed.load(), prefetch->adopted,
		                 prefetch->adopted_modules, prefetch->claimed);
	}
	PipelineCacheLog("Pipelines compiled unoptimized first: {}, replaced by their optimized build: {}",
	                 m_unoptimized_builds, m_replaced_pipelines.size());
	PipelineCacheLog("Shader plans: {} out of memory at the end, {} brought back by the render thread, {} beside it ({} of "
	                 "those used)",
	                 m_program_cache->plans_out, m_program_cache->plan_reloads, m_program_cache->plan_preloads,
	                 m_program_cache->plan_preload_uses);
	PipelineCacheLog("Made where first needed: {} programs translated ({:.0f} ms), {} pipelines ({:.0f} ms)",
	                 m_program_cache->render_translations, m_program_cache->render_translate_ms, m_render_pipelines,
	                 m_render_pipeline_ms);
	ReportStaticStore();
	if (const auto kept = g_pipeline_evictions.kept.load(); kept != 0) {
		PipelineCacheLog("Pipelines out of video memory while idle: {} evicted of {} that could be, {} made again ({:.0f} ms, at "
		                 "most {:.1f} ms)",
		                 g_pipeline_evictions.evicted.load(), kept, g_pipeline_evictions.remade.load(),
		                 static_cast<double>(g_pipeline_evictions.remake_us.load()) / 1000.0,
		                 static_cast<double>(g_pipeline_evictions.remake_max_us.load()) / 1000.0);
	}
}

void PipelineCache::StopWarmup() {
	m_warmup_stop.store(true, std::memory_order_relaxed);
	if (m_warmup_thread.joinable()) m_warmup_thread.join();
}

void PipelineCache::FinishCompileWorkers() {
	StopWarmup();
	m_table_cache_saver = {}; // (stops and joins it)
	ReportSession();
	m_program_cache->prefetch_live.store(nullptr, std::memory_order_release);
	m_program_cache->prefetch.reset();
	m_stopping.store(true, std::memory_order_relaxed);
	m_optimize_workers.reset();
	m_table_workers.reset();
	m_compile_workers.reset();
	for (auto& [key, pending]: m_pending_graphics_pipelines) {
		if (pending->done.load(std::memory_order_acquire) && pending->pipeline->pipeline != nullptr) {
			m_graphics_pipelines.emplace(key, std::move(pending->pipeline));
		}
	}
	m_pending_graphics_pipelines.clear();
}

// The compiler inputs the static precompile compiled, as a warmup file: what the shader prefetch
// translates (precompile-windows.ps1 writes it next to the static pipeline cache).
static std::filesystem::path StaticInputsPath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".shaders");
}

// Live "objects" (live-counters.h): container sizes read from the live thread (counts only, no iteration).
static PipelineCache* g_live_pipeline_cache = nullptr;

void PipelineCache::ReportObjects(uint64_t id) const {
	const auto  store    = ModuleStore::Stats();
	const auto* prefetch = m_program_cache->prefetch_live.load(std::memory_order_acquire);
	std::printf("LIVE_PIPELINE_OBJECTS id=%llu graphics_pipelines=%zu compute_pipelines=%zu pending_graphics=%zu pending_compute=%zu "
	            "replaced_unoptimized=%zu optimizing=%zu programs=%zu jit_bytes=%zu stored_modules=%llu stored_spirv_mib=%llu "
	            "store_file_mib=%llu store_loads=%llu store_load_misses=%llu plans_out=%zu plan_reloads=%llu plan_preloads=%llu plan_preload_uses=%llu prefetch_adopted=%zu "
	            "prefetch_claimed=%zu prefetch_taken=%zu prefetch_take_early=%zu prefetch_finished=%zu prefetch_followed=%zu render_translations=%llu "
	            "render_translate_ms=%.0f render_pipelines=%llu render_pipeline_ms=%.0f warmup_phase=%u\n",
	            static_cast<unsigned long long>(id), m_graphics_pipelines.size(), m_compute_pipelines.size(), m_pending_graphics_pipelines.size(),
	            m_pending_compute_pipelines.size(), m_replaced_pipelines.size(), m_optimizing.size(),
	            m_program_cache->programs.size(), JitArena::Used(), static_cast<unsigned long long>(store.modules),
	            static_cast<unsigned long long>(store.spirv_bytes >> 20u), static_cast<unsigned long long>(store.stored_bytes >> 20u),
	            static_cast<unsigned long long>(store.loads), static_cast<unsigned long long>(store.load_misses),
	            m_program_cache->plans_out, static_cast<unsigned long long>(m_program_cache->plan_reloads),
	            static_cast<unsigned long long>(m_program_cache->plan_preloads),
	            static_cast<unsigned long long>(m_program_cache->plan_preload_uses),
	            prefetch ? prefetch->adopted : size_t {0}, prefetch ? prefetch->claimed : size_t {0},
	            prefetch ? prefetch->taken : size_t {0}, prefetch ? prefetch->take_early : size_t {0},
	            prefetch ? prefetch->finished.load() : size_t {0}, prefetch ? prefetch->followed.load() : size_t {0},
	            static_cast<unsigned long long>(m_program_cache->render_translations), m_program_cache->render_translate_ms,
	            static_cast<unsigned long long>(m_render_pipelines), m_render_pipeline_ms,
	            m_program_cache->warm_phase.load(std::memory_order_relaxed));
}

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	g_live_pipeline_cache          = this;
	LiveCounters::g_objects_report = [](uint64_t id) {
		if (g_live_pipeline_cache != nullptr) g_live_pipeline_cache->ReportObjects(id);
	};
	m_program_cache->push_job = [this](std::function<void()> job) { TableWorkers().Push(std::move(job)); };
	// (The static store first: whether it is there sets how much of the driver caches is loaded.)
	StartupProgress::Report("Loading the pipeline cache", 0, 0);
	InitializeStaticCache(false);
	InitializeDriverCache();
	// Not in the static precompile: a shell that ran the game passes its KYTY_SHADER_WARMUP(_ONLY) on.
#ifndef KYTY_STATIC_PRECOMPILE
	PipelineCacheLog("Pipelines in no cache: {}", FirstBuild(m_graphics) == PipelineBuild::Fast
	                                                  ? "made unoptimized first, the optimized build replaces them "
	                                                    "(KYTY_PIPELINE_FAST_BUILD=0: optimized at once)"
	                                                  : "made optimized (KYTY_PIPELINE_FAST_BUILD=1: unoptimized first)");
	const char* warmup             = std::getenv("KYTY_SHADER_WARMUP");
	bool        prefetch_by_warmup = false; // (RunWarmup starts the shader prefetch)
	if (warmup && IsBinaryDriverCacheKey(m_driver_cache_key) && m_driver_cache != nullptr &&
	    (std::string_view(warmup) == "1" || std::string_view(warmup) == "record")) {
		const auto device = ShaderInputDeviceSignature(m_graphics.GetPhysicalDeviceProperties());
		const auto title = PipelineCacheTitleId();
		const auto root = std::filesystem::path("_PipelineCache") / "warmup-v2";
		// (Its file is the game version's, its identity line the title's: PipelineCacheGameId.)
		auto path = root / fmt::format("{:016x}", XXH3_64bits(device.data(), device.size())) / (PipelineCacheGameId() + ".shaders");
		// KYTY_SHADER_WARMUP_FILE: another file (a fresh recording starts empty, nothing adopted).
		const char* file = std::getenv("KYTY_SHADER_WARMUP_FILE");
		if (file != nullptr && *file != 0) path = file;
		// No inputs for this device signature yet (a driver update, or inputs recorded on another
		// OS): start from the newest file of the same game version and GPU (vendor and device id).
		std::filesystem::path adopt_from;
		std::error_code       error;
		if ((file == nullptr || *file == 0) && !std::filesystem::exists(path, error)) {
			const auto gpu = "KytyShaderWarmup3:" + title + device.substr(0, 1 + 8 + 1 + 8 + 1);
			std::filesystem::file_time_type newest {};
			for (const auto& entry: std::filesystem::directory_iterator(root, error)) {
				const auto candidate = entry.path() / (PipelineCacheGameId() + ".shaders");
				if (!std::filesystem::is_regular_file(candidate, error) ||
				    !LocalShaderWarmup::Cache::FileIdentity(candidate).starts_with(gpu))
					continue;
				const auto written = std::filesystem::last_write_time(candidate, error);
				if (!error && (adopt_from.empty() || written > newest)) {
					adopt_from = candidate;
					newest     = written;
				}
			}
		}
		// The recorded pipelines (15K, most of other areas) cost ~1 GiB of video memory and ~3 GiB of RAM (5 GiB
		// committed): a GPU that cannot also hold the game's working set (small_video_memory) makes the graphics ones
		// (almost no video memory) and the compute ones it met most recently up to 1/16 of its video memory (a laptop's
		// 8 GB RTX 3080: making them when first used, 2-3.6 ms each, held up walks for 150-900 ms at a new view), the
		// rest when they are first used (WarmPipelines). A PC with less RAM than the ~20 GB the game takes while it
		// loads makes none at start-up (~0.75 ms each from the static precompile's binaries when first used). Such a
		// PC records the shaders but translates none at start-up either: their
		// SPIR-V and modules took 3.1 GiB of RAM, and the shader prefetch translates every shader of the game in the
		// background anyway (16 GB reported at 1-1: +76 MiB instead of +3181, the HUD 10 s sooner, the same 59-60 fps
		// and one slow shader translation).
		constexpr uint64_t PipelineWarmupMinRam = uint64_t {30} << 30u;
		const uint64_t     ram                  = LocalPlatform::PhysicalMemory();
		const bool         low_ram              = ram != 0 && ram < PipelineWarmupMinRam;
		const bool         compile_shaders      = std::string_view(warmup) == "1" && !low_ram;
		if (low_ram && std::string_view(warmup) == "1")
			PipelineCacheLog("Shader warmup: recording only (RAM {} MiB)", ram >> 20u);
		m_program_cache->LoadWarmup(path, title + device, adopt_from);
		// The warmup by scene (local/scene-tags.h): Demon's Souls reports the map it loads (libAmpr); KYTY_SCENE_WARMUP=0
		// makes every recorded pipeline at start-up, as before.
		const char* by_scene = std::getenv("KYTY_SCENE_WARMUP");
		m_warmup_path        = path;
		m_scene_warmup       = (by_scene == nullptr || std::string_view(by_scene) != "0") && title.starts_with("PPSA0134");
		if (m_scene_warmup) {
			auto scenes = path;
			scenes += ".scenes";
			if (!SceneTags::Load(scenes, title + device))
				PipelineCacheLog("Scene warmup: ignoring invalid {}", Common::PathToString(scenes));
		}
		// KYTY_PIPELINE_WARMUP: 1 makes them all whatever the memory, 0 none; KYTY_PIPELINE_WARMUP_VIDEO_MIB: the compute
		// pipelines' video memory cap instead of 1/16 of a small one's.
		const char*            forced = std::getenv("KYTY_PIPELINE_WARMUP");
		const std::string_view force  = forced != nullptr ? forced : "";
		const bool             warm_pipelines = force == "1" || (force != "0" && !low_ram);
		const char*            cap            = std::getenv("KYTY_PIPELINE_WARMUP_VIDEO_MIB");
		const uint64_t         compute_video  = cap != nullptr && *cap != 0 ? std::strtoull(cap, nullptr, 10) << 20u
		                                        : force != "1" && m_graphics.small_video_memory
		                                            ? m_graphics.GetTotalMemoryBudget() / 16u
		                                            : UINT64_MAX;
		if (!warm_pipelines && std::string_view(warmup) == "1")
			PipelineCacheLog("Pipeline warmup: skipped (video memory budget {} MiB, RAM {} MiB)",
			                 m_graphics.GetTotalMemoryBudget() >> 20u, ram >> 20u);
		// KYTY_SHADER_WARMUP_BACKGROUND (default on; 0: in front of the game, the "Preparing shaders" and "Preparing
		// pipelines" screens): the recorded shaders are translated and their pipelines made while the game starts, its
		// intro and menus (in front: 16 s of each start-up for a 16-thread desktop's 17,028 recorded inputs, growing with
		// every place played). What the game needs before is made where it is needed, as without a warmup. A -Precompile
		// run (KYTY_SHADER_WARMUP_ONLY) warms up in front, then exits.
		const char*    only       = std::getenv("KYTY_SHADER_WARMUP_ONLY");
		const char*    background = std::getenv("KYTY_SHADER_WARMUP_BACKGROUND");
		WarmupSettings settings;
		settings.compile_shaders = compile_shaders;
		settings.warm_pipelines  = std::string_view(warmup) == "1" && warm_pipelines;
		settings.compute_video   = compute_video;
		settings.background      = compile_shaders && !m_program_cache->warm_records.empty() &&
		                      (only == nullptr || std::string_view(only) != "1") &&
		                      (background == nullptr || std::string_view(background) != "0");
		prefetch_by_warmup = true;
		if (settings.background) {
			m_program_cache->warmup.StartWriter();
			m_program_cache->warming = true;
			m_warmup_thread          = std::thread([this, settings] { RunWarmup(settings); });
		} else {
			RunWarmup(settings);
		}
	}
	if (!prefetch_by_warmup) StartShaderPrefetch({});
#endif
}

// The start-up warmup: the recording's shaders translated (TranslateWarmup) and its pipelines made (WarmPipelines), in
// front of the game or beside it (`background`: on m_warmup_thread); then the shader prefetch, which starts with the
// records the warmup did not translate.
void PipelineCache::RunWarmup(WarmupSettings settings) {
	auto& programs = *m_program_cache;
	// (This thread alone takes m_mutex: at normal priority, its workers below it.)
	if (settings.background) LocalPlatform::SetThreadName("Kyty.Warmup");
	const auto begin         = std::chrono::steady_clock::now();
	const auto memory_before = LocalPlatform::ProcessMemory();
	m_graphics.RefreshMemoryBudget();
	const auto video_before = m_graphics.GetDeviceMemoryUsage();
	if (settings.compile_shaders) programs.TranslateWarmup(m_mutex, settings.background, m_warmup_stop);
	// (The recording's records the warmup did not translate: the shader prefetch's first.)
	std::vector<std::vector<uint32_t>> recorded_inputs;
	for (size_t i = 0; i < programs.warm_records.size(); ++i)
		if (i >= programs.warm_permutations.size() || programs.warm_permutations[i] == nullptr)
			recorded_inputs.emplace_back(programs.warm_records[i].begin(), programs.warm_records[i].end());
	const auto memory_shaders = LocalPlatform::ProcessMemory();
	// By scene (local/scene-tags.h): the shader prefetch at once, the pipelines once the game loads its first scene (after
	// the title and menus, whose own it makes where needed), those of later scenes as it loads them (SceneLoop).
	const bool by_scene = settings.background && settings.warm_pipelines && m_scene_warmup;
	if (by_scene && !m_warmup_stop.load(std::memory_order_relaxed)) StartShaderPrefetch(std::move(recorded_inputs));
	if (settings.warm_pipelines && !m_warmup_stop.load(std::memory_order_relaxed)) {
		const auto pipelines_begin = std::chrono::steady_clock::now();
		if (by_scene) {
			// (No scene in SceneWaitSeconds: every recorded pipeline, as without scenes.)
			constexpr int64_t SceneWaitSeconds = 300;
			programs.warm_phase.store(0, std::memory_order_release);
			while (SceneTags::Current() == SceneTags::None && !m_warmup_stop.load(std::memory_order_relaxed) &&
			       std::chrono::steady_clock::now() - pipelines_begin < std::chrono::seconds(SceneWaitSeconds))
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			if (!m_warmup_stop.load(std::memory_order_relaxed))
				WarmScene(SceneTags::Current(), settings.compute_video, true);
		} else {
			WarmPipelines(settings.compute_video, settings.background);
		}
		// What the driver had to compile is kept at once: the cache is otherwise written only
		// at a normal exit, so after a crash the next launch compiled it all again (2+ minutes). (Beside the game: by
		// the cache saver, once compiles settle: MainPipelineDone.)
		if (!settings.background && std::chrono::steady_clock::now() - pipelines_begin > std::chrono::seconds(5)) (void)Save();
	}
	programs.warm_permutations = {};
	programs.warm_recipes      = {};
	programs.warm_records      = {};
	if (settings.background) {
		Common::LockGuard lock(m_mutex);
		programs.warming = false;
		programs.ReleaseRetired();
		programs.warm_phase.store(0, std::memory_order_release);
		PipelineCacheLog("Shader warmup: done beside the game in {} ms; meanwhile the game translated {} programs itself "
		                 "({:.0f} ms) and made {} pipelines ({:.0f} ms)",
		                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
		                 programs.render_translations, programs.render_translate_ms, m_render_pipelines, m_render_pipeline_ms);
	} else {
		const auto memory_after = LocalPlatform::ProcessMemory();
		m_graphics.RefreshMemoryBudget();
		const auto mib = [](uint64_t to, uint64_t from) { return (static_cast<int64_t>(to) - static_cast<int64_t>(from)) >> 20; };
		PipelineCacheLog("Warmup memory: shaders {:+} MiB committed {:+} MiB resident, pipelines {:+} MiB committed {:+} MiB "
		                 "resident, video memory {:+} MiB",
		    mib(memory_shaders.private_bytes, memory_before.private_bytes),
		    mib(memory_shaders.working_set, memory_before.working_set),
		    mib(memory_after.private_bytes, memory_shaders.private_bytes),
		    mib(memory_after.working_set, memory_shaders.working_set),
		    mib(m_graphics.GetDeviceMemoryUsage(), video_before));
		m_graphics.LogVideoMemory("warmup");
		programs.warmup.StartWriter();
		if (const char* only = std::getenv("KYTY_SHADER_WARMUP_ONLY"); only && std::string_view(only) == "1") {
			if (!Save()) {
				PipelineCacheLog("Shader precompile: failed to save cache");
				std::fflush(nullptr);
				Common::RunExitDrain();
				std::_Exit(1);
			}
			PipelineCacheLog("Shader precompile: complete ({} inputs, {} graphics and {} compute pipelines)",
			    programs.warmup.records.size(), m_graphics_pipelines.size(), m_compute_pipelines.size());
			std::fflush(nullptr);
			Common::RunExitDrain();
			std::_Exit(0);
		}
	}
	if (by_scene) {
		SceneLoop(settings.compute_video);
		return;
	}
	if (!m_warmup_stop.load(std::memory_order_relaxed)) StartShaderPrefetch(std::move(recorded_inputs));
}

void PipelineCache::WarmScene(uint32_t scene, uint64_t compute_video, bool in_memory) {
	auto& programs = *m_program_cache;
	m_scenes_warmed.insert(scene);
	bool known = scene != SceneTags::None && SceneTags::Knows(scene);
	std::unordered_set<uint64_t> only;
	if (known) only = SceneTags::Recipes(scene);
	// (Later scenes: the recording read again, records and recipes only appended to it, so the indices hold; each
	// input's permutation looked up: none, the program left or the game made its own, its pipelines made where needed.
	// warm_records views `records` until the pipelines are made.)
	std::vector<std::vector<uint32_t>> records;
	if (!in_memory) {
		std::vector<std::vector<uint32_t>> recipes;
		if (!LocalShaderWarmup::Cache::ReadInputs(m_warmup_path, records, recipes)) {
			PipelineCacheLog("Scene warmup: {}: the recording could not be read", SceneTags::Name(scene));
			return;
		}
		if (known) std::erase_if(recipes, [&](const auto& words) { return !only.contains(LocalShaderWarmup::RecipeKey(words)); });
		std::vector<uint8_t> needed(records.size(), 0);
		for (const auto& words: recipes) {
			LocalShaderWarmup::PipelineRecord recipe;
			LocalShaderWarmup::Reader         reader {words};
			if (!LocalShaderWarmup::VisitPipeline(reader, recipe)) continue;
			for (const auto index: {recipe.vertex, recipe.pixel, recipe.compute})
				if (index < needed.size()) needed[index] = 1;
		}
		std::vector<std::pair<uint32_t, LocalShaderWarmup::Record>> decoded;
		for (uint32_t i = 0; i < records.size(); ++i) {
			if (needed[i] == 0) continue;
			LocalShaderWarmup::Record record;
			LocalShaderWarmup::Reader reader {records[i]};
			if (LocalShaderWarmup::Visit(reader, record)) decoded.emplace_back(i, std::move(record));
		}
		Common::LockGuard lock(m_mutex);
		programs.warming = true;
		programs.warm_permutations.assign(records.size(), nullptr);
		for (const auto& [index, record]: decoded) programs.warm_permutations[index] = programs.FindRecorded(record);
		programs.warm_records.assign(records.begin(), records.end());
		programs.warm_recipes = std::move(recipes);
	}
	// Tags of which fewer than half are among the recording's recipes are another recording's (it started over).
	size_t present = 0;
	if (known) {
		for (const auto& words: programs.warm_recipes) present += only.contains(LocalShaderWarmup::RecipeKey(words)) ? 1 : 0;
		if (present * 2 < only.size()) {
			PipelineCacheLog("Scene warmup: {}: {} of its {} pipelines recorded, the recording's taken", SceneTags::Name(scene),
			                 present, only.size());
			known = false;
		}
	}
	if (known) {
		PipelineCacheLog("Scene warmup: {}: its {} recorded pipelines", SceneTags::Name(scene), present);
	} else {
		PipelineCacheLog("Scene warmup: {}: not known yet, every recorded pipeline",
		                 scene != SceneTags::None ? SceneTags::Name(scene) : std::string("no scene"));
		m_scene_warmed_all = true;
		if (!in_memory)
			for (const auto& words: programs.warm_recipes) only.insert(LocalShaderWarmup::RecipeKey(words));
	}
	WarmPipelines(compute_video, true, known || !in_memory ? &only : nullptr);
	if (!in_memory) {
		Common::LockGuard lock(m_mutex);
		programs.warming = false;
		programs.ReleaseRetired();
		programs.warm_phase.store(0, std::memory_order_release);
		programs.warm_permutations = {};
		programs.warm_records      = {};
		programs.warm_recipes      = {};
	}
}

// The warmup thread after its start-up work: each scene the game loads gets its recorded pipelines (WarmScene; none
// once every recorded one was made), and the scene tags are saved every SaveSeconds, until m_warmup_stop.
void PipelineCache::SceneLoop(uint64_t compute_video) {
	constexpr int64_t SaveSeconds = 15;
	uint64_t          seen        = SceneTags::Generation();
	auto              saved       = std::chrono::steady_clock::now();
	while (!m_warmup_stop.load(std::memory_order_relaxed)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (const auto now = std::chrono::steady_clock::now(); now - saved >= std::chrono::seconds(SaveSeconds)) {
			(void)SceneTags::Save();
			saved = now;
		}
		if (const auto generation = SceneTags::Generation(); generation != seen) {
			seen             = generation;
			const auto scene = SceneTags::Current();
			if (!m_scene_warmed_all && !m_scenes_warmed.contains(scene)) WarmScene(scene, compute_video, false);
		}
	}
}

// KYTY_SHADER_PREFETCH (default on; 0 off): ProgramCache::StartPrefetch, the warmup's programs left out.
void PipelineCache::StartShaderPrefetch(std::vector<std::vector<uint32_t>> recorded_inputs) {
	const char* prefetch = std::getenv("KYTY_SHADER_PREFETCH");
	if ((prefetch != nullptr && std::string_view(prefetch) == "0") || KYTY_BUILD != KYTY_BUILD_RELEASE ||
	    PipelineCacheTitleId().empty())
		return;
	auto identity = PipelineCacheTitleId() + ShaderInputDeviceSignature(m_graphics.GetPhysicalDeviceProperties());
	// The release's recorded play for a PC without its own (KYTY_PREFETCH_RELEASE=0: not): next to the seed file
	// precompile-windows.ps1 used, released-<title>_<version>.seeds (the package's folder, or _Build\static-precompile in a
	// source tree).
	std::filesystem::path released;
	if (const char* value = std::getenv("KYTY_PREFETCH_RELEASE"); value == nullptr || std::string_view(value) != "0") {
		const auto      name = "released-" + PipelineCacheGameId() + ".seeds";
		std::error_code error;
		for (const auto& candidate: {std::filesystem::path(name), std::filesystem::path("_Build") / "static-precompile" / name}) {
			if (std::filesystem::is_regular_file(candidate, error)) {
				released = candidate;
				break;
			}
		}
	}
	Common::LockGuard lock(m_mutex);
	m_program_cache->StartPrefetch(StaticInputsPath(), std::move(identity), std::move(m_program_cache->warm_keys),
	                               std::move(recorded_inputs), std::move(released),
	                               m_graphics.SupportsComputeWave64() ? 64u : 32u);
}

PipelineCache::PrefetchProgress PipelineCache::GetPrefetchProgress() const {
	const auto& programs = *m_program_cache;
	if (const auto phase = programs.warm_phase.load(std::memory_order_acquire); phase != 0) {
		const auto total = programs.warm_total.load(std::memory_order_relaxed);
		return {std::min(programs.warm_done.load(std::memory_order_relaxed), total), total,
		        phase == 1 ? "Preparing recorded shaders" : "Preparing recorded pipelines"};
	}
	const auto* prefetch = programs.prefetch_live.load(std::memory_order_acquire);
	if (prefetch == nullptr || !prefetch->ready.load(std::memory_order_acquire)) return {};
	const auto total = prefetch->groups.size();
	return {std::min(prefetch->finished.load(std::memory_order_relaxed), total), total};
}

void PipelineCache::WarmPipelines(uint64_t compute_video, bool background, const std::unordered_set<uint64_t>* only) {
	auto& programs = *m_program_cache;
	const auto begin = std::chrono::steady_clock::now();
	const auto get = [&](uint32_t index) -> ProgramCache::Permutation* {
		return index < programs.warm_permutations.size() ? programs.warm_permutations[index] : nullptr;
	};
	const auto decode = [&](uint32_t index, LocalShaderWarmup::Record& record) {
		LocalShaderWarmup::Reader reader {programs.warm_records[index]};
		return LocalShaderWarmup::Visit(reader, record);
	};
	// Recipes are resolved and deduplicated here; the driver compilations (the slow part with a cold driver cache) run
	// on KYTY_SHADER_WARMUP_THREADS workers (beside the game below normal priority), and this thread adds what they
	// made under m_mutex: a pipeline the game made meanwhile stays the game's, the warmup's is destroyed unused.
	struct Job {
		LocalShaderWarmup::PipelineRecord recipe;
		const ProgramCache::Permutation*  vertex = nullptr;
		const ProgramCache::Permutation*  pixel = nullptr;
		const ProgramCache::Permutation*  compute = nullptr;
		GraphicsPipelineKey               key;
		uint64_t                          recipe_key = 0;
	};
	std::vector<Job> jobs;
	jobs.reserve(programs.warm_recipes.size());
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> graphics_keys;
	std::unordered_set<uint64_t>                                     compute_ids;
	std::atomic<size_t> skipped {0};
	// The most recently recorded first (the area played last, likely the one the save is in): what a video memory cap
	// for compute pipelines leaves out are the oldest.
	for (auto recorded = programs.warm_recipes.rbegin(); recorded != programs.warm_recipes.rend(); ++recorded) {
		Job job;
		job.recipe_key = LocalShaderWarmup::RecipeKey(*recorded);
		if (only != nullptr && !only->contains(job.recipe_key)) continue;
		LocalShaderWarmup::Reader reader {*recorded};
		if (!LocalShaderWarmup::VisitPipeline(reader, job.recipe)) { ++skipped; continue; }
		if (job.recipe.compute != LocalShaderWarmup::NoShader) {
			job.compute = get(job.recipe.compute);
			if (!job.compute) { ++skipped; continue; }
			if (!compute_ids.insert(job.compute->handle.id).second) continue;
		} else {
			job.vertex = get(job.recipe.vertex);
			job.pixel  = get(job.recipe.pixel);
			if (!job.vertex || (job.recipe.pixel != LocalShaderWarmup::NoShader && !job.pixel)) { ++skipped; continue; }
			job.key = {job.recipe.rendering, job.vertex->handle.id, job.pixel ? job.pixel->handle.id : 0,
			           job.recipe.vertex_input, job.recipe.state};
			if (!graphics_keys.insert(job.key).second) continue;
		}
		jobs.push_back(std::move(job));
	}
	if (only != nullptr) {
		// (A later scene's: what an earlier one made, or the game, stays.)
		Common::LockGuard lock(m_mutex);
		std::erase_if(jobs, [&](const Job& job) {
			return job.compute != nullptr ? m_compute_pipelines.contains(job.compute->handle.id) : m_graphics_pipelines.contains(job.key);
		});
	}
	if (background) {
		programs.warm_total.store(jobs.size(), std::memory_order_relaxed);
		programs.warm_done.store(0, std::memory_order_relaxed);
		programs.warm_phase.store(2, std::memory_order_release);
	}

	std::atomic<size_t> next {0}, computes {0}, left {0};
	// Compute pipelines while the warmup's video memory stays under `compute_video` (checked every 32 of them).
	m_graphics.RefreshMemoryBudget();
	const uint64_t    video_start = m_graphics.GetDeviceMemoryUsage();
	std::atomic<bool> compute_full {false};
	// What the workers made (job index and pipeline), for this thread to add.
	std::mutex                                                made_mutex;
	std::condition_variable                                   made_ready;
	std::vector<std::pair<size_t, std::unique_ptr<Pipeline>>> made;
	uint32_t                                                  workers_done = 0;
	const auto work = [&] {
		for (size_t n; (n = next.fetch_add(1, std::memory_order_relaxed)) < jobs.size();) {
			if (m_warmup_stop.load(std::memory_order_relaxed) || std::chrono::steady_clock::now() >= programs.warm_deadline)
				return;
			auto& job      = jobs[n];
			auto  pipeline = std::make_unique<Pipeline>();
			if (job.compute) {
				if (compute_full.load(std::memory_order_relaxed)) { ++left; continue; }
				LocalShaderWarmup::Record input;
				if (!decode(job.recipe.compute, input)) { ++skipped; continue; }
				input.compute.stage.program = &job.compute->program;
				CreatePipelineInternal(m_graphics, *pipeline, input.compute, job.compute->handle.module, m_driver_cache);
				if (compute_video != UINT64_MAX && (computes.fetch_add(1, std::memory_order_relaxed) + 1) % 32 == 0) {
					m_graphics.RefreshMemoryBudget();
					const auto video = m_graphics.GetDeviceMemoryUsage();
					if (video > video_start && video - video_start >= compute_video) compute_full.store(true);
				}
			} else {
				LocalShaderWarmup::Record vs, ps;
				if (!decode(job.recipe.vertex, vs) || (job.pixel && !decode(job.recipe.pixel, ps))) { ++skipped; continue; }
				vs.vertex.stage.program = &job.vertex->program;
				if (job.pixel) ps.pixel.stage.program = &job.pixel->program;
				CreatePipelineInternal(m_graphics, *pipeline, job.recipe.rendering, job.recipe.vertex_input,
					vs.vertex, job.vertex->handle, job.pixel ? &ps.pixel : nullptr,
					job.pixel ? job.pixel->handle : ShaderProgram {}, job.recipe.state, m_driver_cache);
			}
			if (pipeline->pipeline == nullptr || pipeline->pipeline_layout == nullptr) { ++skipped; continue; }
			MainPipelineDone();
			{
				std::lock_guard lock(made_mutex);
				made.emplace_back(n, std::move(pipeline));
			}
			made_ready.notify_one();
		}
	};
	const uint32_t threads = std::min<uint32_t>(ProgramCache::WarmupThreads(background),
	                                            static_cast<uint32_t>(std::max<size_t>(jobs.size(), 1)));
	size_t compiled = 0, present = 0;
	{
		std::vector<std::jthread> pool;
		for (uint32_t t = 0; t < threads; ++t)
			pool.emplace_back([&] {
				if (background) {
					LocalPlatform::SetThreadName("Kyty.Warmup");
					LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
				}
				work();
				{
					std::lock_guard lock(made_mutex);
					workers_done++;
				}
				made_ready.notify_one();
			});
		for (bool finished = false; !finished;) {
			std::vector<std::pair<size_t, std::unique_ptr<Pipeline>>> batch;
			{
				std::unique_lock lock(made_mutex);
				made_ready.wait(lock, [&] { return !made.empty() || workers_done == threads; });
				batch.swap(made);
				finished = workers_done == threads;
			}
			if (batch.empty()) continue;
			{
				Common::LockGuard lock(m_mutex);
				for (auto& [n, pipeline]: batch) {
					const auto& job   = jobs[n];
					pipeline->recipe  = job.recipe_key;
					const bool  added = job.compute != nullptr
					                        ? m_compute_pipelines.try_emplace(job.compute->handle.id, std::move(pipeline)).second
					                        : m_graphics_pipelines.try_emplace(job.key, std::move(pipeline)).second;
					added ? ++compiled : ++present;
				}
			}
			for (auto& [n, pipeline]: batch)
				if (pipeline != nullptr) m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr); // (the game's came first)
			if (const auto done = compiled + present; done / 1000 != (done - batch.size()) / 1000)
				PipelineCacheLog("Pipeline warmup: compiling {}/{}", done, jobs.size());
			if (background) programs.warm_done.store(compiled + present, std::memory_order_relaxed);
			else StartupProgress::Report("Preparing pipelines", compiled + present, jobs.size()); // (the main thread's)
		}
	}
	PipelineCacheLog("Pipeline warmup: compiled {} of {} recipes ({} the game made first), skipped {}, elapsed {} ms ({} "
	                 "threads{})",
		compiled, programs.warm_recipes.size(), present, skipped.load(),
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
		threads, background ? ", beside the game" : "");
	if (compute_video != UINT64_MAX)
		PipelineCacheLog("Pipeline warmup: compute pipelines up to {} MiB of video memory, {} older ones made when first used",
		                 compute_video >> 20u, left.load());
}

PipelineCache::~PipelineCache() {
	LiveCounters::g_objects_report = nullptr;
	g_live_pipeline_cache          = nullptr;
	FinishCompileWorkers();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->optimized != nullptr) {
				m_graphics.device.destroyPipeline(pipeline->optimized->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	// (The pipelines' layouts are shared: once all of them are gone.)
	DestroySharedLayouts(m_graphics.device);
	m_graphics.pipeline_binaries = nullptr;
	m_graphics.pipeline_blobs    = nullptr;
	for (const auto pipeline: m_replaced_pipelines) {
		m_graphics.device.destroyPipeline(pipeline, nullptr);
	}
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
	if (m_table_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_table_cache, nullptr);
	}
	if (m_static_cache != nullptr) {
		m_graphics.static_pipeline_cache = nullptr;
		m_graphics.device.destroyPipelineCache(m_static_cache, nullptr);
	}
}

// The static pipeline cache: _PipelineCache/static/<title>_<version>.bin, headed by the GPU and driver it was
// compiled on only (its pipelines are keyed by their whole create info, whatever build made them).
static std::string StaticCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	return "KytyPCStatic1" + ShaderInputDeviceSignature(properties);
}

static std::filesystem::path StaticCachePath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".bin");
}

// The translation features the static store's pipelines were made with (ShaderRecompiler::Frontend::TranslatorFeatures),
// then those the last precompile translated with: _PipelineCache/static/<title>_<version>.features. The first are written
// by a precompile whose last merge made the store of its own pipelines alone (--merge --prune), the second by the last
// merge of any precompile (--stamp: with recorded play lost, or -Quick, some pipelines stay as they were: another run
// would not make them, so the store is not out of date for that). None (an older store, none at all): 0.
static std::filesystem::path StaticFeaturesPath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".features");
}

static std::pair<uint32_t, uint32_t> ReadStaticStamp() {
	std::ifstream file(StaticFeaturesPath());
	uint32_t      features = 0, attempted = 0;
	if (!(file >> features)) return {0, 0};
	return {features, file >> attempted ? attempted : features};
}

static uint32_t ReadStaticFeatures() {
	return ReadStaticStamp().first;
}

// `made`: the store holds this precompile's pipelines alone. (The precompile tool's merge: the emulator has no use.)
[[maybe_unused]] static void WriteStaticFeatures(bool made) {
	namespace Translate = ShaderRecompiler::Frontend;
	const auto      used      = Translate::TranslatorFeatureSet();
	const auto      features  = made ? used : ReadStaticFeatures();
	const auto      path      = StaticFeaturesPath();
	auto            temporary = path;
	std::error_code error;
	temporary += ".tmp";
	{
		std::ofstream file(temporary, std::ios::trunc);
		file << features << ' ' << used << '\n';
		if (!file) return;
	}
	std::filesystem::rename(temporary, path, error);
	PipelineCacheLog("Static pipeline store: made with translation features {:#x} (this precompile's: {:#x})", features, used);
}

// Its pipelines as the driver's binaries (VK_KHR_pipeline_binary), read when needed: the driver keeps a
// copy of a whole pipeline cache in memory. Headed by the GPU, the driver and its global pipeline key.
static std::filesystem::path StaticBinariesPath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".binaries");
}

static std::string StaticBinariesSignature(GraphicContext& graphics) {
	const auto& properties = graphics.GetPhysicalDeviceProperties();
	auto        device     = ShaderInputDeviceSignature(properties);
#ifdef _WIN32
	// AMD's Windows driver keys pipelines by the executable's name: none of a store made before the precompile ran under
	// the emulator's name (precompile-windows.ps1, issue #28) matched the game's keys. Marked, such a store is out of
	// date (StaticCacheCurrent), so the launcher offers the precompile again; others' stores are unchanged.
	constexpr uint32_t AmdVendor = 0x1002u;
	if (properties.vendorID == AmdVendor) device += " made-as-kyty_emulator";
#endif
	return PipelineBinaries::Signature(graphics.device, device);
}

// Without pipeline binaries: each pipeline's own pipeline cache data (pipelineCacheBlobs.h), read when it is needed,
// in place of the static pipeline cache, which the driver copies into memory whole. KYTY_STATIC_BLOBS=0: the cache.
static bool StaticBlobsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STATIC_BLOBS");
		return !(value != nullptr && std::string_view(value) == "0");
	}();
	return enabled;
}

static std::filesystem::path StaticBlobsPath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".blobs");
}

static std::string StaticBlobsSignature(const vk::PhysicalDeviceProperties& properties) {
	return "KytyPCBlobs1" + ShaderInputDeviceSignature(properties);
}

// KYTY_STATIC_BLOBS_FROM_CACHE=1 (the precompile): blobs made of the static pipeline cache's pipelines (loaded as a
// source), compiled only where the driver cannot give them otherwise.
static bool StaticBlobsFromCache() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_STATIC_BLOBS_FROM_CACHE");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return on;
}

// A static pipeline cache of this GPU and driver exists (its file begins with their signature).
static bool StaticCacheFileMatches(const vk::PhysicalDeviceProperties& properties) {
	const auto    signature = StaticCacheSignature(properties);
	std::ifstream file(StaticCachePath(), std::ios::binary);
	std::string   saved(signature.size(), '\0');
	return file.read(saved.data(), static_cast<std::streamsize>(saved.size())) && saved == signature;
}

// Whether this PC's static precompile is pipeline cache blobs: without pipeline binaries, unless a static pipeline cache
// of this GPU and driver was made before them (no blobs beside it). That one stays in use, and a precompile adds to it as
// it did: a cache made once is never made again for the blobs, which come with the next cache made from nothing (a new
// install, a driver update). Decided once per process.
static bool StaticBlobsMode(GraphicContext& graphics) {
	if (graphics.pipeline_binaries_enabled || !StaticBlobsEnabled()) return false;
	static const bool mode = [&] {
		std::error_code error;
		return std::filesystem::exists(StaticBlobsPath(), error) || StaticBlobsFromCache() ||
		       !StaticCacheFileMatches(graphics.GetPhysicalDeviceProperties());
	}();
	return mode;
}

// A pipeline cache file (signature, XXH3 of the payload, payload): the payload, or nothing when the
// file is missing, damaged or from another GPU or driver.
static std::vector<uint8_t> ReadPipelineCacheFile(const std::filesystem::path& path, const std::string& signature) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return {};
	const auto size = static_cast<size_t>(file.tellg());
	if (size < signature.size() + sizeof(uint64_t)) return {};
	std::string          saved(signature.size(), '\0');
	uint64_t             hash = 0;
	std::vector<uint8_t> payload(size - signature.size() - sizeof(hash));
	file.seekg(0);
	file.read(saved.data(), static_cast<std::streamsize>(saved.size()));
	file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
	file.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
	if (!file || saved != signature || XXH3_64bits(payload.data(), payload.size()) != hash) return {};
	return payload;
}

void PipelineCache::InitializeStaticCache(bool create) {
	if (PipelineCacheTitleId().empty() || KYTY_BUILD != KYTY_BUILD_RELEASE) return;
#ifndef KYTY_STATIC_PRECOMPILE
	// The game's shaders translated as its static pipeline store's were (before any is translated: the pipeline cache's
	// construction): an older store's pipelines stay in use. No update makes a player compile again: 9935d547's
	// cooperative halves missed an older store on an 8 GB laptop, where the pipelines made at their first use (past the
	// warm-up's video memory) took 4.5-6.6 s each on the render thread. A precompile stamps the store it makes with the
	// features it translated with, and offers itself again until then (StaticCacheCurrent).
	if (!create) {
		namespace Translate = ShaderRecompiler::Frontend;
		Translate::SetTranslatorFeatures(ReadStaticFeatures());
		if (const auto off = Translate::TranslatorDeviceFeatures() & ~Translate::TranslatorFeatureSet(); off != 0)
			PipelineCacheLog("Translation features {:#x} off: the static pipeline store was made without them (precompile "
			                 "again to use them)",
			                 off);
	}
#endif
	// KYTY_STATIC_PIPELINE_CACHE=0: without it (first-encounter comparisons).
	if (const char* value = std::getenv("KYTY_STATIC_PIPELINE_CACHE"); !create && value != nullptr &&
	    std::string_view(value) == "0")
		return;
	const auto begin = std::chrono::steady_clock::now();
	if (m_static_binaries != nullptr || m_static_blobs != nullptr) return;
	if (StaticBlobsMode(m_graphics)) {
		RecordModuleSpirvHashes();
		m_static_blobs = PipelineCacheBlobs::Open(StaticBlobsPath(), StaticBlobsSignature(m_graphics.GetPhysicalDeviceProperties()));
		if (m_static_blobs != nullptr) {
			m_graphics.pipeline_blobs = m_static_blobs.get();
			PipelineCacheLog("Static pipeline blobs: {} pipelines, {} MiB in {} ({} ms)", m_static_blobs->Pipelines(),
			                 m_static_blobs->DataBytes() >> 20u, Common::PathToString(StaticBlobsPath()),
			                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin)
			                     .count());
#ifdef KYTY_STATIC_PRECOMPILE
			// A conversion (StaticBlobsFromCache) takes what the blobs lack from the static pipeline cache too.
			if (!StaticBlobsFromCache()) return;
			create = false;
#else
			return;
#endif
		}
		// None yet: the precompile makes them (and loads no static cache: every shard process would hold one whole,
		// unless it converts one: StaticBlobsFromCache); the emulator takes a static cache from before them.
#ifdef KYTY_STATIC_PRECOMPILE
		if (!StaticBlobsFromCache()) return;
		create = false;
#else
		if (create) return;
#endif
	}
	if (m_graphics.pipeline_binaries_enabled) {
		m_static_binaries = PipelineBinaries::Open(m_graphics.device, StaticBinariesPath(), StaticBinariesSignature(m_graphics));
		if (m_static_binaries != nullptr) {
			m_graphics.pipeline_binaries = m_static_binaries.get();
			PipelineCacheLog("Static pipeline binaries: {} pipelines, {} binaries, {} MiB in {} ({} ms)",
			                 m_static_binaries->Pipelines(), m_static_binaries->Binaries(),
			                 m_static_binaries->DataBytes() >> 20u, Common::PathToString(StaticBinariesPath()),
			                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin)
			                     .count());
			return;
		}
		// None yet (or another GPU's or driver's): a static pipeline cache still serves, and in the
		// precompile it is what the binaries are taken from without compiling.
		create = false;
	}
	const auto path    = StaticCachePath();
	const auto payload = ReadPipelineCacheFile(path, StaticCacheSignature(m_graphics.GetPhysicalDeviceProperties()));
	if (payload.empty() && !create) return;
	vk::PipelineCacheCreateInfo info {};
	info.initialDataSize = payload.size();
	info.pInitialData    = payload.empty() ? nullptr : payload.data();
	if (m_graphics.device.createPipelineCache(&info, nullptr, &m_static_cache) != vk::Result::eSuccess) {
		info.initialDataSize = 0;
		info.pInitialData    = nullptr;
		if (m_graphics.device.createPipelineCache(&info, nullptr, &m_static_cache) != vk::Result::eSuccess) {
			m_static_cache = nullptr;
			return;
		}
	}
	m_graphics.static_pipeline_cache = m_static_cache;
	PipelineCacheLog("Static pipeline cache: {} bytes from {} in {} ms{}", payload.size(), Common::PathToString(path),
	                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
	                 m_graphics.pipeline_cache_control_enabled ? "" : " (unused: no pipeline cache control)");
}

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
	if (const auto* key = std::getenv("KYTY_DRIVER_CACHE_KEY"); key && *key) {
		if (!IsBinaryDriverCacheKey(key)) {
			PipelineCacheLog("Vulkan pipeline cache: disabled (invalid executable fingerprint)");
			return;
		}
		m_driver_cache_key = key;
	}
	if (git_hash == "unknown" || git_revision == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown git revision)");
		return;
	}
	if (git_hash.ends_with("-dirty") && m_driver_cache_key.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (dirty build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	if (!m_driver_cache_key.empty()) {
		m_driver_cache_path = std::filesystem::path("_PipelineCache") / "local" /
		                      m_driver_cache_key / (title_id + ".bin");
	}
	// The launcher keeps one cache across emulator builds: pipelines of changed shaders pile up in it, so past
	// 2 GiB (one build's warmup is ~130 MB) it starts over. The driver copies a loaded cache whole into the process's
	// memory (a 0.94 GB and a 0.6 GB one: 1.5 GB of RAM at 1-1): 512 MiB each below 24 GB of RAM (ordinary play
	// passes 1/64 of 16 GB: 256 MiB, 265 MiB after 30 minutes of 1-1, issue #14; a 16 GB PC is short of RAM as it is),
	// 1/16 of the RAM from 24 GB (a 1.2 GB cache on a 32 GB PC started over at every launch, issue #9): a cache that
	// starts over compiles its pipelines in play again.
	// With the static store (binaries or blobs: the precompile's recorded pipelines, their table and native variants
	// included) a driver cache only holds what the store lacks: ~4 MB a session at 1-1, the table cache nothing. Caches
	// past StoreLimit are pipelines piled up over earlier builds and stores (0.86 + 0.43 GB on a desktop, each copied
	// whole by the driver into the process: 1.5 GB of its pools) and start over once; what they held comes from the store,
	// or compiles once more (1-1: ~170 native XPR variants on the async workers, their draws on the normal path meanwhile;
	// the render thread made the same pipelines in the same time). Such small caches are also saved at least every
	// SmallSaveSeconds while compiles go on: a session that never settles 20 s and does not exit cleanly kept none of them.
	const uint64_t ram       = LocalPlatform::PhysicalMemory();
	const bool     store     = m_static_binaries != nullptr || m_static_blobs != nullptr;
	const auto     ram_limit = [ram, store](uint64_t limit) {
		constexpr uint64_t StoreLimit = uint64_t {256} << 20u;
		if (ram != 0) limit = std::min(limit, ram >= (uint64_t {24} << 30u) ? ram / 16u : uint64_t {512} << 20u);
		return store ? std::min(limit, StoreLimit) : limit;
	};
	m_driver_cache = LoadDriverCache(m_driver_cache_path, ram_limit(uint64_t {2} << 30u), m_driver_cache_saved_size);
	if (m_driver_cache == nullptr) return;
	// Table mode pipelines (src/local/table-xpr.inc) in a cache of their own: the static precompile holds none of
	// them (each compiles in ~0.1-4 s), they are saved once their compiles settle (not only at a clean exit), and
	// they do not push the main cache past its limit.
	m_table_cache_path = m_driver_cache_path;
	m_table_cache_path.replace_extension(".table.bin");
	m_table_cache = LoadDriverCache(m_table_cache_path, ram_limit(uint64_t {2} << 30u), m_table_cache_saved_size);
	if (m_table_cache == nullptr) return;
	m_table_cache_saver = std::jthread([this, store_sized = store](std::stop_token stop) {
		LocalPlatform::SetThreadName("Kyty.TableCache");
		LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
		constexpr int64_t           SmallSaveSeconds = 60;
		std::mutex                  mutex;
		std::condition_variable_any wake;
		uint64_t                    saved      = m_table_pipelines_done.load();
		uint64_t                    main_saved = m_main_pipelines_done.load();
		auto                        table_at   = std::chrono::steady_clock::now();
		auto                        main_at    = table_at;
		const auto settled = [&](const std::atomic<int64_t>& last_done, std::chrono::steady_clock::time_point saved_at) {
			const auto now  = std::chrono::steady_clock::now();
			const auto last = std::chrono::steady_clock::time_point(
			    std::chrono::steady_clock::duration(last_done.load(std::memory_order_acquire)));
			return now - last >= std::chrono::seconds(20) ||
			       (store_sized && now - saved_at >= std::chrono::seconds(SmallSaveSeconds));
		};
		while (!stop.stop_requested()) {
			{
				std::unique_lock lock(mutex);
				wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
			}
			// Each cache once no pipeline has compiled into it for 20 s (a burst of new pairs or pipelines settled).
			if (stop.stop_requested()) break;
			if (const auto done = m_table_pipelines_done.load(std::memory_order_acquire);
			    done != saved && m_table_pipeline_jobs.load(std::memory_order_acquire) == 0 &&
			    settled(m_table_pipeline_last_done, table_at)) {
				std::lock_guard save(m_table_cache_mutex);
				if (SaveDriverCache(m_table_cache, m_table_cache_path, m_table_cache_saved_size)) saved = done;
				table_at = std::chrono::steady_clock::now();
			}
			if (const auto done = m_main_pipelines_done.load(std::memory_order_acquire);
			    done != main_saved && settled(m_main_pipeline_last_done, main_at)) {
				std::lock_guard save(m_main_cache_mutex);
				if (SaveDriverCache(m_driver_cache, m_driver_cache_path, m_driver_cache_saved_size)) main_saved = done;
				main_at = std::chrono::steady_clock::now();
			}
		}
	});
}

// A driver pipeline cache file: the signature (GPU, driver and cache key), the payload's XXH3, the payload; one
// past `limit` bytes starts over. Null: the cache is disabled.
vk::PipelineCache PipelineCache::LoadDriverCache(const std::filesystem::path& file_path, uint64_t limit,
                                                 size_t& loaded_size) {
	const auto path         = Common::PathToString(file_path);
	const bool cache_exists = Common::File::IsFileExisting(file_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(file_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties(), m_driver_cache_key);
		if (file_size > limit) {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: starting {} over ({} bytes, limit {})", path, file_size, limit);
		} else if (file_size >= signature.size() + sizeof(uint64_t) &&
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
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCache           cache = nullptr;
	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		return nullptr;
	}
	loaded_size = initial_data.size();
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	return cache;
}

// Written to a temporary file renamed to `file_path`, unless nothing compiled since it was loaded or saved
// (`saved_size`). The cache stays in use: a save right after the warmup destroyed it, so nothing compiled later
// in that session (native XPR variants, first uses, optimized builds) was cached or saved at the exit.
bool PipelineCache::SaveDriverCache(vk::PipelineCache cache, const std::filesystem::path& file_path,
                                    size_t& saved_size) {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		if (size == saved_size) return true;
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(cache, &size, payload.data());
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
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties(), m_driver_cache_key);
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(file_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = file_path;
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
	    !Common::File::RenameFile(temp_path, file_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}", Common::PathToString(file_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(), Common::PathToString(file_path));
	saved_size = payload.size();
	return true;
}

bool PipelineCache::Save() {
	Common::LockGuard lock(m_mutex);
	if (m_scene_warmup && !SceneTags::Save()) PipelineCacheLog("Scene warmup: saving the scene tags failed");
	const bool inputs_saved = m_program_cache->warmup.Save();
	if (!inputs_saved) PipelineCacheLog("Shader warmup: cache save failed");
	else if (m_program_cache->warmup.Enabled()) PipelineCacheLog("Shader warmup: saved {} inputs and {} pipelines",
	    m_program_cache->warmup.records.size(), m_program_cache->warmup.pipelines.size());
	bool saved = inputs_saved;
	if (m_driver_cache != nullptr) {
		std::lock_guard main(m_main_cache_mutex);
		saved &= SaveDriverCache(m_driver_cache, m_driver_cache_path, m_driver_cache_saved_size);
	}
	if (m_table_cache != nullptr) {
		std::lock_guard table(m_table_cache_mutex);
		saved &= SaveDriverCache(m_table_cache, m_table_cache_path, m_table_cache_saved_size);
	}
	return saved;
}

namespace {
template <typename Cache>
auto FindProgramSource(Cache& cache, const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	auto found = cache.by_program.find(&program);
	if (found == cache.by_program.end()) {
		// Sources are node-stable and permutations live in deques, so a pair stays valid until the source is
		// erased (warm-code mismatch, first use). One pass indexes every permutation: a pass per program
		// (8900 permutations, 0.1-4 ms) ran for each program a record store met first, hundreds in one frame
		// when an area's draws were first stored (a 1.9 s frame at Boletaria 1-1).
		for (auto& [key, source]: cache.programs)
			for (const auto& permutation: source.permutations)
				cache.by_program.try_emplace(&permutation.program, std::pair {&source, &permutation});
		found = cache.by_program.find(&program);
	}
	return found;
}
} // namespace

bool PipelineCache::TraceStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                               std::span<const uint32_t> user_data, uint64_t shader_base,
                               ShaderRecompiler::IR::SrtReadTrace& trace) {
	Common::LockGuard lock(m_mutex);
	auto&             cache = *m_program_cache;
	const auto        found = FindProgramSource(cache, program);
	if (found == cache.by_program.end()) return false;
	const ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = shader_base,
	    .read_memory                = ReadMappedRawGuestMemory,
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .sync_memory                = SyncShaderGuestMemory,
	    .try_read_memory_span       = ReadMappedMemorySpan,
	};
	cache.EnsurePlan(*found->second.first);
	return ShaderRecompiler::IR::TraceLinearSrtReads(found->second.first->resource_plan, runtime, trace);
}

ShaderRecompiler::IR::SrtUserDataUse PipelineCache::UserDataUse(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	Common::LockGuard lock(m_mutex);
	auto&             cache = *m_program_cache;
	const auto        found = FindProgramSource(cache, program);
	if (found == cache.by_program.end()) return {};
	cache.EnsurePlan(*found->second.first);
	return ShaderRecompiler::IR::UserDataUse(found->second.first->resource_plan);
}

bool PipelineCache::RematerializeStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                       std::span<const uint32_t> user_data, uint64_t shader_base,
                                       ShaderRecompiler::IR::ResourceSnapshot& resources) {
	Common::LockGuard lock(m_mutex);
	auto&             cache = *m_program_cache;
	const auto        found = FindProgramSource(cache, program);
	if (found == cache.by_program.end()) return false;
	auto& [source, permutation] = found->second;
	cache.EnsurePlan(*source);
	const ShaderRecompiler::IR::SrtRuntime input_runtime {
	    .user_data                  = user_data,
	    .shader_base                = shader_base,
	    .read_memory                = ReadMappedRawGuestMemory,
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .sync_memory                = SyncShaderGuestMemory,
	    .try_read_memory_span       = ReadMappedMemorySpan,
	};
	ShaderReadObserver::Runtime observed_runtime(input_runtime);
	ShaderRecompiler::IR::ResourceSpecialization        specialization;
	const ShaderRecompiler::IR::ResourceSpecialization* borrowed = nullptr;
	ShaderRecompiler::IR::MaterializeReport             report;
	if (!ShaderRecompiler::IR::MaterializeResources(source->resource_plan, observed_runtime.Get(),
	                                                resources, specialization, &report,
	                                                &source->specialization_guard, &borrowed)) {
		return false;
	}
	return (borrowed != nullptr ? *borrowed : specialization) == permutation->specialization;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    ShaderVertexInputInfo& vertex_info, ShaderPixelInputInfo& pixel_info, bool table_mode,
    std::array<uint32_t, 2> table_portable) {
	LiveCensus::Scope census(LiveCensus::GraphicsPrograms, 0);
	m_program_cache->table_pending = false;
	m_program_cache->table_waiting.reset();
	// Reused per thread: every normal-path draw prepares both (their user data were new vectors).
	thread_local ShaderParams vertex_params, pixel_params;
	PrepareProgramInto(vertex_regs, context, user_config, vertex_info, vertex_params);
	const bool mesh_active   = vertex_info.mesh.threads_num[0] != 0;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info.mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	if (pixel_active) {
		PrepareProgramInto(pixel_regs, sh, target_export_mapping, pixel_info, pixel_params);
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info.clip_space;
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
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (pixel_active) {
		pixel_info.lod_stats_subgroup = m_graphics.fragment_subgroup_reduction;
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, table_mode, table_portable[1]);
	}
	result.vertex = m_program_cache->Get(vertex_params, vertex_info, push_data_cursor, table_mode, table_portable[0]);
	return result;
}

bool PipelineCache::TakeMaterializationFailure() {
	return std::exchange(g_materialization_failed, false);
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               std::string* unevaluated, bool table_mode, uint32_t table_portable) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	// Reused per thread: every dispatch prepares one (its user data was a new vector each time).
	thread_local ShaderParams params;
	PrepareProgramInto(regs, sh, input_info, params);
	Common::LockGuard lock(m_mutex);
	m_program_cache->table_pending = false;
	m_program_cache->table_waiting.reset();
	uint32_t          push_data_cursor = 0;
	g_unevaluated     = unevaluated;
	const auto program = m_program_cache->Get(params, input_info, push_data_cursor, table_mode, table_portable);
	g_unevaluated     = nullptr;
	return program;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

std::size_t PipelineCache::NativeGraphicsPipelineKeyHash::operator()(
    const GraphicsPipelineKey& key) const {
	// These exact 166 bytes already define static-state equality. Do not hash the
	// other structures' padding: their equality compares members instead.
	return XXH3_64bits_withSeed(&key.static_params, sizeof(key.static_params),
	                           GraphicsPipelineKeyHash::Prefix(key));
}

PipelineCache::Pipeline& PipelineCache::CreateGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const ShaderProgram& vertex_program,
    const ShaderProgram& pixel_program, bool native_bindings) {
	return *CreateGraphicsPipelineImpl(colors, depth, vs_input_info, command, ps_input_info, topology,
	                                   primitive_restart_enable, vertex_program, pixel_program,
	                                   native_bindings, false);
}

PipelineCache::Pipeline* PipelineCache::TryCreateGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const ShaderProgram& vertex_program,
    const ShaderProgram& pixel_program, bool native_bindings) {
	return CreateGraphicsPipelineImpl(colors, depth, vs_input_info, command, ps_input_info, topology,
	                                  primitive_restart_enable, vertex_program, pixel_program,
	                                  native_bindings, true);
}

PipelineCache::Pipeline* PipelineCache::CreateGraphicsPipelineImpl(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const ShaderProgram& vertex_program,
    const ShaderProgram& pixel_program, bool native_bindings, bool async) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	PromoteOptimized();
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	key.vs_shader_id            = vs_id;
	key.ps_shader_id            = ps_id;
	key.native_bindings         = native_bindings;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = color_count;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[i] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[i] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
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
	static_params.stencil_test_enable      = depth.stencil_test_enable;
	static_params.stencil_front            = depth.stencil_static_front;
	static_params.stencil_back             = depth.stencil_static_back;
	const bool rect_list     = topology == vk::PrimitiveTopology::ePatchList;
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	for (uint32_t i = 0; i < color_count; i++) {
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[i]       = bc.color_srcblend;
		static_params.color_comb_fcn[i]       = bc.color_comb_fcn;
		static_params.color_destblend[i]      = bc.color_destblend;
		static_params.alpha_srcblend[i]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[i]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[i]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[i] = bc.separate_alpha_blend;
		static_params.blend_enable[i]         = bc.enable;
		static_params.blend_bypass[i]         = rt.info.blend_bypass;
	}
	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	const bool indexed = kyty_local_pipeline_index_mode.load(std::memory_order_relaxed) != 0;
	if (indexed) {
		if (auto found = m_native_graphics_pipelines.find(key);
		    found != m_native_graphics_pipelines.end()) {
			return found->second;
		}
	}
	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		if (indexed) {
			m_native_graphics_pipelines.emplace(key, iter->second.get());
		}
		return iter->second.get();
	}

	if (async) {
		if (auto pending = m_pending_graphics_pipelines.find(key);
		    pending != m_pending_graphics_pipelines.end()) {
			if (!pending->second->done.load(std::memory_order_acquire)) {
				m_pipeline_waiting = std::shared_ptr<const std::atomic<bool>>(pending->second, &pending->second->done);
				return nullptr;
			}
			auto finished = std::move(pending->second->pipeline);
			m_pending_graphics_pipelines.erase(pending);
			EXIT_NOT_IMPLEMENTED(finished->pipeline == nullptr || finished->pipeline_layout == nullptr);
			auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(finished));
			EXIT_IF(!inserted);
			if (indexed) {
				m_native_graphics_pipelines.emplace(iter->first, iter->second.get());
			}
			return iter->second.get();
		}
		auto job             = std::make_shared<PendingGraphicsPipeline>();
		job->rendering       = rendering;
		job->vertex_input    = key.vertex_input;
		job->static_params   = static_params;
		job->vs_input_info   = vs_input_info;
		job->ps_active       = ps_active;
		job->vertex_program  = vertex_program;
		job->pixel_program   = pixel_program;
		job->native_bindings = native_bindings;
		if (ps_active) {
			job->ps_input_info = *ps_input_info;
		}
		m_pending_graphics_pipelines.emplace(key, job);
		const bool table = vs_input_info.stage.program != nullptr && vs_input_info.stage.program->table_mode;
		if (table) m_table_pipeline_jobs.fetch_add(1, std::memory_order_relaxed);
		(table ? TableWorkers() : Workers()).Push([this, job, table] {
			// (KYTY_HITCH_LOG_MS: a pipeline no cache held, compiled while its draws took another path.)
			SlowLog::Scope slow([&](double ms) {
				const auto* pixel = job->ps_active ? job->ps_input_info.stage.program : nullptr;
				std::printf("SLOW AsyncGraphicsPipeline %.1f ms %s VS=0x%016llx PS=0x%016llx\n", ms, table ? "table" : "native",
				            static_cast<unsigned long long>(job->vs_input_info.stage.program->shader_hash),
				            static_cast<unsigned long long>(pixel != nullptr ? pixel->shader_hash : 0));
			}, SlowLog::HitchThreshold());
			CreatePipelineInternal(m_graphics, *job->pipeline, job->rendering, job->vertex_input,
			                       job->vs_input_info, job->vertex_program,
			                       job->ps_active ? &job->ps_input_info : nullptr, job->pixel_program,
			                       job->static_params, table ? TablePipelineCache() : m_driver_cache,
			                       job->native_bindings);
			if (KeepsRemake(m_graphics, *job->pipeline, LastPipelineFromStore())) {
				GraphicsMakeInputs inputs;
				inputs.rendering       = job->rendering;
				inputs.vertex_input    = job->vertex_input;
				inputs.static_params   = job->static_params;
				inputs.vs_input_info   = job->vs_input_info;
				inputs.ps_input_info   = job->ps_input_info;
				inputs.ps_active       = job->ps_active;
				inputs.vertex_program  = job->vertex_program;
				inputs.pixel_program   = job->pixel_program;
				inputs.native_bindings = job->native_bindings;
				inputs.table           = table;
				job->pipeline->remake  = GraphicsRemake(std::move(inputs));
			}
			if (table) TablePipelineDone();
			else MainPipelineDone();
			job->done.store(true, std::memory_order_release);
		});
		LiveCounters::Add(LiveCounters::AsyncPipelines);
		m_pipeline_waiting = std::shared_ptr<const std::atomic<bool>>(job, &job->done);
		return nullptr;
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

	SlowLog::Scope create_slow([&](double ms) {
		const auto* vs = vs_input_info.stage.program;
		const auto* ps = ps_active ? ps_input_info->stage.program : nullptr;
		std::printf("SLOW CreateGraphicsPipeline %.1f ms VS=%llu PS=%llu VS=0x%016llx PS=0x%016llx\n", ms,
		            static_cast<unsigned long long>(vs_id), static_cast<unsigned long long>(ps_id),
		            static_cast<unsigned long long>(vs != nullptr ? vs->shader_hash : 0),
		            static_cast<unsigned long long>(ps != nullptr ? ps->shader_hash : 0));
	}, SlowLog::HitchThreshold());
	LiveCensus::WaitScope compiling(LiveCensus::WaitCompile);
	// (The render thread's own waits: the slow frame lines' translate/pipeline counts.)
	std::optional<LiveCounters::ScopedUs> graphics_timed;
	if (LiveCensus::g_render) {
		LiveCounters::Add(LiveCounters::GraphicsPipelines);
		graphics_timed.emplace(LiveCounters::GraphicsPipelineUs);
	}
	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	const auto create_begin = std::chrono::steady_clock::now();
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vs_input_info,
	                       vertex_program, ps_input_info, pixel_program, static_params,
	                       m_driver_cache, native_bindings, FirstBuild(m_graphics));
	const bool from_store = LastPipelineFromStore();
	m_render_pipelines++;
	m_render_pipeline_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	MainPipelineDone();
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	if (indexed) {
		m_native_graphics_pipelines.emplace(iter->first, iter->second.get());
	}
	std::optional<ShaderPixelInputInfo> pixel_info;
	if (ps_active) pixel_info = *ps_input_info;
	// Its build as `build` says (BuildOptimized: the optimized one).
	const auto make = [this, rendering, vertex_input = iter->first.vertex_input, vs_input_info, pixel_info, vertex_program,
	                   pixel_program, static_params, native_bindings](PipelineBuild build) {
		return [=, this](Pipeline& target) {
			CreatePipelineInternal(m_graphics, target, rendering, vertex_input, vs_input_info, vertex_program,
			                       pixel_info ? &*pixel_info : nullptr, pixel_program, static_params, m_driver_cache,
			                       native_bindings, build);
		};
	};
	if (KeepsRemake(m_graphics, *iter->second, from_store)) {
		GraphicsMakeInputs inputs;
		inputs.rendering       = rendering;
		inputs.vertex_input    = iter->first.vertex_input;
		inputs.static_params   = static_params;
		inputs.vs_input_info   = vs_input_info;
		inputs.ps_active       = ps_active;
		inputs.vertex_program  = vertex_program;
		inputs.pixel_program   = pixel_program;
		inputs.native_bindings = native_bindings;
		if (ps_active) inputs.ps_input_info = *ps_input_info;
		iter->second->remake = GraphicsRemake(std::move(inputs));
	}
	BuildOptimized(*iter->second, make(PipelineBuild::Optimize));
	// Warmup recipes describe the normal variant only.
	if (native_bindings) {
		return iter->second.get();
	}
	LocalShaderWarmup::PipelineRecord recipe;
	recipe.vertex = m_program_cache->RecordedIndex(vertex_program);
	recipe.pixel =
	    ps_active ? m_program_cache->RecordedIndex(pixel_program) : LocalShaderWarmup::NoShader;
	recipe.rendering    = iter->first.rendering;
	recipe.vertex_input = iter->first.vertex_input;
	recipe.state        = iter->first.static_params;
	if (recipe.vertex != LocalShaderWarmup::NoShader &&
	    (!ps_active || recipe.pixel != LocalShaderWarmup::NoShader)) {
		m_program_cache->warmup.AddPipeline(recipe);
		iter->second->recipe = LocalShaderWarmup::RecipeKey(recipe);
	}

	return iter->second.get();
}

PipelineCache::Pipeline&
PipelineCache::CreateComputePipeline(const ShaderComputeInputInfo& input_info,
                                     const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);
	PromoteOptimized();

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	SlowLog::Scope create_slow([&](double ms) {
		const auto* cs = input_info.stage.program;
		std::printf("SLOW CreateComputePipeline %.1f ms CS=%llu CS=0x%016llx\n", ms,
		            static_cast<unsigned long long>(compute_program.id),
		            static_cast<unsigned long long>(cs != nullptr ? cs->shader_hash : 0));
	}, SlowLog::HitchThreshold());
	LiveCensus::WaitScope compiling(LiveCensus::WaitCompile);
	// (The render thread's own waits: the slow frame lines' translate/pipeline counts.)
	std::optional<LiveCounters::ScopedUs> compute_timed;
	if (LiveCensus::g_render) {
		LiveCounters::Add(LiveCounters::ComputePipelines);
		compute_timed.emplace(LiveCounters::ComputePipelineUs);
	}
	auto       cached       = std::make_unique<Pipeline>();
	const auto create_begin = std::chrono::steady_clock::now();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache,
	                       FirstBuild(m_graphics));
	const bool from_store = LastPipelineFromStore();
	m_render_pipelines++;
	m_render_pipeline_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - create_begin).count();
	MainPipelineDone();

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	const auto make = [this, input_info, module = compute_program.module](PipelineBuild build) {
		return [=, this](Pipeline& target) { CreatePipelineInternal(m_graphics, target, input_info, module, m_driver_cache, build); };
	};
	if (KeepsRemake(m_graphics, *iter->second, from_store))
		iter->second->remake = std::make_shared<const std::function<void(Pipeline&)>>(make(PipelineBuild::Full));
	BuildOptimized(*iter->second, make(PipelineBuild::Optimize));
	LocalShaderWarmup::PipelineRecord recipe;
	recipe.compute = m_program_cache->RecordedIndex(compute_program);
	if (recipe.compute != LocalShaderWarmup::NoShader) {
		m_program_cache->warmup.AddPipeline(recipe);
		iter->second->recipe = LocalShaderWarmup::RecipeKey(recipe);
	}

	return *iter->second;
}
bool PipelineCache::TablePending() const {
	return m_program_cache->table_pending;
}

std::shared_ptr<const std::atomic<bool>> PipelineCache::TableWaiting() const {
	return m_program_cache->table_waiting;
}

PipelineCache::Pipeline* PipelineCache::TryCreateComputePipeline(const ShaderComputeInputInfo& input_info,
                                                                 const ShaderProgram&          compute_program) {
	EXIT_IF(!compute_program);
	Common::LockGuard lock(m_mutex);
	PromoteOptimized();
	if (auto iter = m_compute_pipelines.find(compute_program.id); iter != m_compute_pipelines.end())
		return iter->second.get();
	if (auto pending = m_pending_compute_pipelines.find(compute_program.id);
	    pending != m_pending_compute_pipelines.end()) {
		if (!pending->second->done.load(std::memory_order_acquire)) {
			m_pipeline_waiting = std::shared_ptr<const std::atomic<bool>>(pending->second, &pending->second->done);
			return nullptr;
		}
		auto finished = std::move(pending->second->pipeline);
		m_pending_compute_pipelines.erase(pending);
		EXIT_NOT_IMPLEMENTED(finished->pipeline == nullptr || finished->pipeline_layout == nullptr);
		return m_compute_pipelines.emplace(compute_program.id, std::move(finished)).first->second.get();
	}
	auto job        = std::make_shared<PendingComputePipeline>();
	job->input_info = input_info;
	job->module     = compute_program.module;
	m_pending_compute_pipelines.emplace(compute_program.id, job);
	m_table_pipeline_jobs.fetch_add(1, std::memory_order_relaxed);
	TableWorkers().Push([this, job] { // (table dispatches' pipelines)
		SlowLog::Scope slow([&](double ms) {
			std::printf("SLOW AsyncComputePipeline %.1f ms table CS=0x%016llx\n", ms,
			            static_cast<unsigned long long>(job->input_info.stage.program->shader_hash));
		}, SlowLog::HitchThreshold());
		CreatePipelineInternal(m_graphics, *job->pipeline, job->input_info, job->module, TablePipelineCache(),
		                       PipelineBuild::Full, true);
		if (KeepsRemake(m_graphics, *job->pipeline, LastPipelineFromStore())) {
			job->pipeline->remake = std::make_shared<const std::function<void(Pipeline&)>>(
			    [this, input_info = job->input_info, module = job->module](Pipeline& target) {
				    CreatePipelineInternal(m_graphics, target, input_info, module, TablePipelineCache(), PipelineBuild::Full,
				                           true);
			    });
		}
		TablePipelineDone();
		job->done.store(true, std::memory_order_release);
	});
	LiveCounters::Add(LiveCounters::AsyncPipelines);
	m_pipeline_waiting = std::shared_ptr<const std::atomic<bool>>(job, &job->done);
	return nullptr;
}
#ifdef KYTY_STATIC_PRECOMPILE
#include "staticPrecompile.inc"
#endif

} // namespace Libs::Graphics
