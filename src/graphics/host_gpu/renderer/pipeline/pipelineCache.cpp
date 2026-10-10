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
#include "graphics/host_gpu/renderer/pipeline/pipelineBinaries.h"
#include "graphics/host_gpu/renderer/pipeline/shaderReadObserver.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"
#include "native-preparation-scratch.h"
#include "native-resource-state.h"
#include "shader-warmup-cache.h"
#include "live-census.h"
#include "live-counters.h"
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
// there, and the GPU reads zeros where the host would fault.
constexpr uint64_t NullPageEnd = 0x10000;

bool ReadShaderGuestMemory(void*, uint64_t address, uint32_t* value) {
	if (value != nullptr && address < NullPageEnd) {
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
	if (address < NullPageEnd) {
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
	return count >= 2 && count <= (clean ? 1024u : 16u) && address >= NullPageEnd &&
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
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

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

	// A permutation the prefetch translated, its SPIR-V in the prefetch's scratch file.
	struct PrefetchedModule {
		ShaderRecompiler::IR::ResourceSpecialization specialization; // compiled with
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		uint64_t                                     spirv_offset = 0;
		size_t                                       spirv_words  = 0;
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
		// Prefetched permutations not used yet (AdoptPrefetchedModule).
		std::vector<PrefetchedModule> prefetched;
		// Table mode permutations being translated.
		std::vector<std::shared_ptr<TableJob>> table_jobs;
	};
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
	// (gigabytes for the whole game) waits in a temporary file, the plans and infos in memory.
	struct Prefetch {
		enum State : uint8_t { Pending, Running, Done, Taken };
		struct Group { // the inputs of one program
			ProgramKey                                        key;
			uint64_t                                          source = 0; // SourceDigest
			size_t                                            code_words = 0;
			std::vector<uint32_t>                             records;
			std::atomic<uint8_t>                              state {Pending};
			std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
			std::vector<PrefetchedModule>                     modules;
		};
		~Prefetch() {
			stop.store(true, std::memory_order_relaxed);
			if (loader.joinable()) loader.join();
			for (auto& thread: threads) thread.join();
			LocalPlatform::CloseScratchFile(spill);
		}

		LocalShaderWarmup::Cache                               inputs;
		std::vector<std::unique_ptr<Group>>                    groups; // in translation order
		std::unordered_map<ProgramKey, Group*, ProgramKeyHash> index;  // read-only once `ready`
		std::atomic<bool>                                      ready {false}, stop {false};
		std::atomic<size_t>                                    next {0}, finished {0};
		std::atomic<uint64_t>                                  spill_end {0};
		uint64_t                                               spill = 0;
		std::thread                                            loader;
		std::vector<std::thread>                               threads;
		// Render thread: programs and modules taken over, programs translated there while pending.
		size_t adopted = 0, adopted_modules = 0, claimed = 0;
	};
	std::unique_ptr<Prefetch> prefetch;

	static uint64_t SourceDigest(std::span<const uint32_t> code, std::span<const uint32_t> back_code) {
		return XXH3_64bits_withSeed(back_code.data(), back_code.size_bytes(), XXH3_64bits(code.data(), code.size_bytes()));
	}

	// Loads the inputs on a thread of its own, leaves out the programs the warmup holds (`warm`), and
	// starts the translation.
	void StartPrefetch(std::filesystem::path path, std::string identity, std::vector<ProgramKey> warm) {
		prefetch         = std::make_unique<Prefetch>();
		prefetch->loader = std::thread([this, path = std::move(path), identity = std::move(identity),
		                                warm = std::move(warm)] {
			LocalPlatform::SetThreadName("Kyty.Prefetch");
			LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
			auto&      p     = *prefetch;
			const auto begin = std::chrono::steady_clock::now();
			if (!std::filesystem::exists(path) || !p.inputs.Open(path, identity) || p.inputs.records.empty()) {
				PipelineCacheLog("Shader prefetch: no inputs for this GPU and driver in {} (precompile-windows.ps1 "
				                 "writes them)", Common::PathToString(path));
				return;
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
			std::ranges::stable_sort(p.groups, std::ranges::greater {}, [](const auto& group) { return group->code_words; });
			p.spill = LocalPlatform::OpenScratchFile();
			if (p.spill == 0) {
				PipelineCacheLog("Shader prefetch: no temporary file for the SPIR-V");
				return;
			}
			uint32_t threads = std::max(2u, std::thread::hardware_concurrency() / 2u);
			if (const char* value = std::getenv("KYTY_SHADER_PREFETCH_THREADS"); value != nullptr && *value != '\0')
				threads = std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 64u);
			PipelineCacheLog("Shader prefetch: {} programs ({} inputs) to translate on {} threads, loaded in {} ms",
			                 p.groups.size(), p.inputs.records.size(), threads,
			                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count());
			p.ready.store(true, std::memory_order_release);
			for (uint32_t t = 0; t < threads; ++t) p.threads.emplace_back([this] { PrefetchWork(); });
		});
	}

	void PrefetchWork() {
		LocalPlatform::SetThreadName("Kyty.Prefetch");
		LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
		auto& p = *prefetch;
		for (size_t n; !p.stop.load(std::memory_order_relaxed) &&
		               (n = p.next.fetch_add(1, std::memory_order_relaxed)) < p.groups.size();) {
			auto& group    = *p.groups[n];
			auto  expected = static_cast<uint8_t>(Prefetch::Pending);
			if (group.state.compare_exchange_strong(expected, Prefetch::Running, std::memory_order_acq_rel)) {
				PrefetchTranslate(group);
				group.state.store(Prefetch::Done, std::memory_order_release);
				group.state.notify_all();
			}
			if (p.finished.fetch_add(1, std::memory_order_relaxed) + 1 == p.groups.size()) {
				PipelineCacheLog("Shader prefetch: all {} programs translated, {} MiB of SPIR-V", p.groups.size(),
				                 p.spill_end.load(std::memory_order_relaxed) >> 20u);
				std::vector<std::vector<uint32_t>>().swap(p.inputs.records); // no longer read
			}
		}
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
				const auto bytes  = result.spirv.size() * sizeof(uint32_t);
				const auto offset = p.spill_end.fetch_add(bytes, std::memory_order_relaxed);
				if (!LocalPlatform::WriteScratchFile(p.spill, offset, result.spirv.data(), bytes)) continue;
				group.modules.push_back({std::move(record.specialization), std::move(result.program).TakeCompiledInfo(),
				                         offset, result.spirv.size()});
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
		auto  state = group.state.load(std::memory_order_acquire);
		if (state == Prefetch::Pending &&
		    group.state.compare_exchange_strong(state, Prefetch::Taken, std::memory_order_acq_rel)) {
			prefetch->claimed++;
			return nullptr;
		}
		for (; state == Prefetch::Running; state = group.state.load(std::memory_order_acquire))
			group.state.wait(Prefetch::Running, std::memory_order_acquire);
		if (state != Prefetch::Done || !group.plan || group.source != SourceDigest(params.code, params.back_code))
			return nullptr;
		group.state.store(Prefetch::Taken, std::memory_order_relaxed);
		prefetch->adopted++;
		return &group;
	}

	// A prefetched module of the source compiled with the specialization a translation would compile
	// here (portable formats, see Get) and a push data start the cursor allows: its permutation.
	const Permutation* AdoptPrefetchedModule(SourceEntry& source, const ShaderRecompiler::CompileOptions& options,
	                                         ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                         uint32_t push_data_cursor) {
		if (source.prefetched.empty() || prefetch == nullptr) return nullptr;
		auto compiled = specialization;
		(void)ShaderRecompiler::IR::PortableFormats(source.resource_plan.info, compiled);
		const auto module = std::ranges::find_if(source.prefetched, [&](const PrefetchedModule& candidate) {
			const auto& layout = candidate.program.bindings;
			return candidate.specialization == compiled &&
			       layout.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor, layout.ShaderDataDwords());
		});
		if (module == source.prefetched.end()) return nullptr;
		std::vector<uint32_t> spirv(module->spirv_words);
		if (!LocalPlatform::ReadScratchFile(prefetch->spill, module->spirv_offset, spirv.data(),
		                                    spirv.size() * sizeof(uint32_t)))
			return nullptr;
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

		vk::ShaderModuleCreateInfo create_info {};
		create_info.codeSize    = spirv.size() * sizeof(uint32_t);
		create_info.pCode       = spirv.data();
		vk::ShaderModule module = nullptr;
		RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
		                     "create recompiled shader module");
		EXIT_IF(module == nullptr);
		if (PipelineKeyLog()) {
			NotePipelineKeyModule(module, options.shader_hash,
			                      XXH3_64bits(spirv.data(), spirv.size() * sizeof(uint32_t)));
		}
		SetVulkanObjectNameF(device, module, "Kyty.Shader.{}[0x{:016x}]", stage_name,
		                     options.shader_hash);
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
				for (const auto& p : entry->second.permutations) {
					device.destroyShaderModule(p.handle.module, nullptr);
				}
				std::erase_if(by_program, [&](const auto& item) { return item.second.first == &entry->second; });
				programs.erase(entry);
				entry = programs.end();
			} else {
				std::vector<uint32_t>().swap(entry->second.warm_code);
				std::vector<uint32_t>().swap(entry->second.warm_back_code);
			}
		}
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
				entry->second.prefetched = std::move(group->modules);
				group->plan.reset();
			}
		}
		if (entry == programs.end() ||
		    AdoptPrefetchedModule(entry->second, options, specialization, push_data_cursor) == nullptr) {
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
			if (entry == programs.end()) {
				auto resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				if (!ReportMaterialization(label, stage, params.hash, report,
				                           ShaderRecompiler::IR::MaterializeResources(
				                               resource_plan, runtime, resources, specialization, &report)))
					return {};
				entry = programs.try_emplace(lookup_key, std::move(resource_plan)).first;
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
		}
		const auto& permutation = entry->second.permutations.back();
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
	// only). Recompiling a record is pure, so workers translate a chunk to SPIR-V and
	// Warm registers the results in the serial order: the same entries, permutations
	// and handle ids; a worker's result for a record Warm then skips is dropped.
	struct WarmJob {
		LocalShaderWarmup::Record          record;
		std::vector<uint32_t>              user_data;
		ShaderRecompiler::IR::ResourcePlan       plan;
		ShaderRecompiler::CompileResult          result;
		ShaderRecompiler::IR::CompiledShaderInfo info; // the result's, taken by the worker
	};

	static uint32_t WarmupThreads() {
		uint32_t threads = 1;
		if (const char* value = std::getenv("KYTY_SHADER_WARMUP_THREADS"); value && *value) {
			const auto end    = value + std::strlen(value);
			const auto parsed = std::from_chars(value, end, threads);
			if (parsed.ec != std::errc {} || parsed.ptr != end || threads == 0 || threads > 64) threads = 1;
		}
		return threads;
	}

	static std::unique_ptr<WarmJob> PrepareWarmJob(std::span<const uint32_t> saved) {
		auto                      job = std::make_unique<WarmJob>();
		LocalShaderWarmup::Reader reader {saved};
		if (!LocalShaderWarmup::Visit(reader, job->record) || !LocalShaderWarmup::ValidKey(job->record)) return nullptr;
		job->user_data.resize(job->record.user_data_count);
		const auto options = LocalShaderWarmup::Options(job->record, job->user_data);
		auto translated    = ShaderRecompiler::TranslateProgram(job->record.code, options);
		job->plan          = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		// Recorded before buffer words (or without them): the permutation the game now selects.
		ShaderRecompiler::IR::CanonicalizeSpecialization(translated.program.info, job->record.specialization);
		job->result        = ShaderRecompiler::CompileProgram(std::move(translated), options, job->record.specialization,
		                                                      job->record.push_cursor);
		job->info          = TakeOnWorker(job->result);
		return job;
	}

	// Streams the records, in Warm's descending order, through the workers; they stay
	// at most Window records ahead because a result holds a whole program.
	class WarmWorkers {
	public:
		WarmWorkers(const std::vector<std::vector<uint32_t>>& records, uint32_t threads)
		    : records(records), jobs(records.size()), ready(records.size()) {
			for (uint32_t t = 0; t < threads; ++t) pool.emplace_back([this] { Run(); });
		}
		~WarmWorkers() {
			stop.store(true, std::memory_order_relaxed);
			consumed.fetch_add(1, std::memory_order_release);
			consumed.notify_all();
			for (auto& thread: pool) thread.join();
		}
		WarmWorkers(const WarmWorkers&)            = delete;
		WarmWorkers& operator=(const WarmWorkers&) = delete;
		// Warm takes every index once, in descending order; null for an invalid record.
		std::unique_ptr<WarmJob> Take(size_t index) {
			ready[index].wait(0, std::memory_order_acquire);
			consumed.fetch_add(1, std::memory_order_release);
			consumed.notify_all();
			return std::move(jobs[index]);
		}

	private:
		static constexpr size_t Window = 512;
		void Run() {
			for (;;) {
				const size_t n = taken.fetch_add(1, std::memory_order_relaxed);
				if (n >= jobs.size()) return;
				for (size_t c = consumed.load(std::memory_order_acquire); n >= c + Window;
				     c = consumed.load(std::memory_order_acquire)) {
					if (stop.load(std::memory_order_relaxed)) return;
					consumed.wait(c, std::memory_order_acquire);
				}
				if (stop.load(std::memory_order_relaxed)) return;
				const size_t index = jobs.size() - 1 - n;
				jobs[index]        = PrepareWarmJob(records[index]);
				ready[index].store(1, std::memory_order_release);
				ready[index].notify_one();
			}
		}
		const std::vector<std::vector<uint32_t>>& records;
		std::vector<std::unique_ptr<WarmJob>>     jobs;
		std::vector<std::atomic<uint8_t>>         ready;
		std::atomic<size_t>                       taken {0};
		std::atomic<size_t>                       consumed {0};
		std::atomic<bool>                         stop {false};
		std::vector<std::thread>                  pool;
	};

	void Warm(const std::filesystem::path& path, const std::string& identity, bool compile,
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
		if (!compile || warmup.records.empty()) return;
		const auto begin = std::chrono::steady_clock::now();
		// 0 means all recorded inputs. A caller can explicitly bound startup time.
		uint32_t budget_seconds = 0;
		if (const char* value = std::getenv("KYTY_SHADER_WARMUP_SECONDS"); value && *value) {
			const auto end = value + std::strlen(value);
			const auto parsed = std::from_chars(value, end, budget_seconds);
			if (parsed.ec != std::errc {} || parsed.ptr != end || budget_seconds > 3600) budget_seconds = 0;
		}
		if (budget_seconds) warm_deadline = begin + std::chrono::seconds(budget_seconds);
		warm_permutations.resize(warmup.records.size());
		size_t compiled = 0;
		const uint32_t threads = WarmupThreads();
		std::optional<WarmWorkers> workers;
		if (threads > 1) workers.emplace(warmup.records, threads);
		for (size_t remaining = warmup.records.size(); remaining != 0; --remaining) {
			const size_t index = remaining - 1;
			if (std::chrono::steady_clock::now() >= warm_deadline) break;
			StartupProgress::Report("Preparing shaders", warmup.records.size() - remaining, warmup.records.size());
			std::unique_ptr<WarmJob> job;
			LocalShaderWarmup::Record parsed;
			if (workers) {
				job = workers->Take(index);
				if (!job) continue;
			} else {
				LocalShaderWarmup::Reader reader {warmup.records[index]};
				if (!LocalShaderWarmup::Visit(reader, parsed) || !LocalShaderWarmup::ValidKey(parsed)) continue;
			}
			auto& record = job ? job->record : parsed;
			ProgramKey key {record.stage, record.hash, record.user_data_count,
			                static_cast<uint32_t>(record.code.size()), record.static_key};
			auto entry = programs.find(key);
			if (entry != programs.end() && (entry->second.warm_code != record.code ||
			    entry->second.warm_back_code != record.back_code)) continue;
			if (entry != programs.end()) {
				const auto found = std::ranges::find_if(entry->second.permutations, [&](const auto& p) {
					return p.specialization == record.specialization && p.program.bindings.push_data_start_dword ==
					    ShaderRecompiler::IR::PushData::StartFor(record.push_cursor, p.program.bindings.ShaderDataDwords());
				});
				if (found != entry->second.permutations.end()) { warm_permutations[index] = &*found; continue; }
			}
			// The compiler consumes the user-data width; values are resolved anew by
			// MaterializeResources when the guest actually draws or dispatches.
			std::vector<uint32_t> user_data = job ? std::move(job->user_data) : std::vector<uint32_t>(record.user_data_count);
			auto options = LocalShaderWarmup::Options(record, user_data);
			if (job) {
				if (entry == programs.end()) {
					entry = programs.try_emplace(std::move(key), std::move(job->plan)).first;
					entry->second.warm_code = record.code;
					entry->second.warm_back_code = record.back_code;
				}
				ShaderParams params {record.code, std::move(user_data), record.hash, record.back_code};
				entry->second.permutations.push_back(
				    FinishPermutation(params, options, job->result, std::move(job->info), std::move(record.specialization)));
			} else {
				auto translated = ShaderRecompiler::TranslateProgram(record.code, options);
				ShaderRecompiler::IR::CanonicalizeSpecialization(translated.program.info, record.specialization);
				if (entry == programs.end()) {
					auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
					entry = programs.try_emplace(std::move(key), std::move(plan)).first;
					entry->second.warm_code = record.code;
					entry->second.warm_back_code = record.back_code;
				}
				ShaderParams params {record.code, std::move(user_data), record.hash, record.back_code};
				entry->second.permutations.push_back(CompilePermutation(params, options, std::move(translated),
				    std::move(record.specialization), record.push_cursor));
			}
			auto& permutation = entry->second.permutations.back();
			recorded_programs[permutation.handle.id] = uint32_t(index);
			warm_permutations[index] = &permutation;
			if (++compiled % 250 == 0) PipelineCacheLog("Shader warmup: compiling {}/{}", compiled, warmup.records.size());
		}
		workers.reset();
		PipelineCacheLog("Shader warmup: compiled {}/{} permutations in {} ms ({} threads)", compiled,
		    warmup.records.size(),
		    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
		    threads);
	}
	LocalShaderWarmup::Cache warmup;
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
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
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
// measured (~100x faster to compile) and tested. An AMD RX 6700 XT (RDNA2) crashed in its driver at start-up, every
// run, while the first compute pipelines were made (10-09 report); the precompile, which builds them optimized, ran
// on that PC without a crash, so the unoptimized build is the suspect (unconfirmed: no AMD GPU here).
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

void PipelineCache::FinishCompileWorkers() {
	m_table_cache_saver = {}; // (stops and joins it)
	if (auto& prefetch = m_program_cache->prefetch; prefetch != nullptr) {
		if (prefetch->ready.load(std::memory_order_acquire))
			PipelineCacheLog("Shader prefetch: {} of {} programs translated; the game took over {} of them ({} modules) "
			                 "and translated {} itself",
			                 prefetch->finished.load(), prefetch->groups.size(), prefetch->adopted,
			                 prefetch->adopted_modules, prefetch->claimed);
		prefetch.reset();
	}
	m_stopping.store(true, std::memory_order_relaxed);
	m_optimize_workers.reset();
	m_table_workers.reset();
	m_compile_workers.reset();
	if (m_unoptimized_builds != 0) {
		PipelineCacheLog("Pipelines compiled unoptimized first: {}, replaced by their optimized build: {}",
		                 m_unoptimized_builds, m_replaced_pipelines.size());
	}
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

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_program_cache->push_job = [this](std::function<void()> job) { TableWorkers().Push(std::move(job)); };
	InitializeDriverCache();
	StartupProgress::Report("Loading the pipeline cache", 0, 0);
	InitializeStaticCache(false);
	// Not in the static precompile: a shell that ran the game passes its KYTY_SHADER_WARMUP(_ONLY) on.
#ifndef KYTY_STATIC_PRECOMPILE
	const char* warmup = std::getenv("KYTY_SHADER_WARMUP");
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
		// committed): a GPU that cannot also hold the game's working set (small_video_memory), or a PC with less RAM
		// than the ~20 GB the game takes while it loads, makes them when they are first used (~0.75 ms each from the
		// static precompile's binaries). Such a PC records the shaders but translates none at start-up either: their
		// SPIR-V and modules took 3.1 GiB of RAM, and the shader prefetch translates every shader of the game in the
		// background anyway (16 GB reported at 1-1: +76 MiB instead of +3181, the HUD 10 s sooner, the same 59-60 fps
		// and one slow shader translation).
		constexpr uint64_t PipelineWarmupMinRam = uint64_t {24} << 30u;
		const uint64_t     ram                  = LocalPlatform::PhysicalMemory();
		const bool         low_ram              = ram != 0 && ram < PipelineWarmupMinRam;
		const bool         compile_shaders      = std::string_view(warmup) == "1" && !low_ram;
		if (low_ram && std::string_view(warmup) == "1")
			PipelineCacheLog("Shader warmup: recording only (RAM {} MiB)", ram >> 20u);
		const auto memory_before = LocalPlatform::ProcessMemory();
		m_graphics.RefreshMemoryBudget();
		const auto video_before = m_graphics.GetDeviceMemoryUsage();
		m_program_cache->Warm(path, title + device, compile_shaders, adopt_from);
		const auto memory_shaders = LocalPlatform::ProcessMemory();
		const bool warm_pipelines = !m_graphics.small_video_memory && !low_ram;
		if (!warm_pipelines && std::string_view(warmup) == "1")
			PipelineCacheLog("Pipeline warmup: skipped (video memory budget {} MiB, RAM {} MiB)",
			                 m_graphics.GetTotalMemoryBudget() >> 20u, ram >> 20u);
		if (std::string_view(warmup) == "1" && warm_pipelines) {
			const auto begin = std::chrono::steady_clock::now();
			WarmPipelines();
			// What the driver had to compile is kept at once: the cache is otherwise written only
			// at a normal exit, so after a crash the next launch compiled it all again (2+ minutes).
			if (std::chrono::steady_clock::now() - begin > std::chrono::seconds(5)) (void)Save();
		}
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
		m_program_cache->warmup.StartWriter();
		if (const char* only = std::getenv("KYTY_SHADER_WARMUP_ONLY"); only && std::string_view(only) == "1") {
			if (!Save()) {
				PipelineCacheLog("Shader precompile: failed to save cache");
				std::fflush(nullptr);
				Common::RunExitDrain();
				std::_Exit(1);
			}
			PipelineCacheLog("Shader precompile: complete ({} inputs, {} graphics and {} compute pipelines)",
			    m_program_cache->warmup.records.size(), m_graphics_pipelines.size(), m_compute_pipelines.size());
			std::fflush(nullptr);
			Common::RunExitDrain();
			std::_Exit(0);
		}
	}
	if (const char* prefetch = std::getenv("KYTY_SHADER_PREFETCH");
	    (prefetch == nullptr || std::string_view(prefetch) != "0") && KYTY_BUILD == KYTY_BUILD_RELEASE &&
	    !PipelineCacheTitleId().empty()) {
		std::vector<ProgramCache::ProgramKey> warm;
		for (const auto& [key, source]: m_program_cache->programs) warm.push_back(key);
		m_program_cache->StartPrefetch(
		    StaticInputsPath(),
		    PipelineCacheTitleId() + ShaderInputDeviceSignature(m_graphics.GetPhysicalDeviceProperties()), std::move(warm));
	}
#endif
}

PipelineCache::PrefetchProgress PipelineCache::GetPrefetchProgress() const {
	const auto* prefetch = m_program_cache->prefetch.get();
	if (prefetch == nullptr || !prefetch->ready.load(std::memory_order_acquire)) return {};
	const auto total = prefetch->groups.size();
	return {std::min(prefetch->finished.load(std::memory_order_relaxed), total), total};
}

void PipelineCache::WarmPipelines() {
	auto& programs = *m_program_cache;
	const auto begin = std::chrono::steady_clock::now();
	const auto get = [&](uint32_t index) -> ProgramCache::Permutation* {
		return index < programs.warm_permutations.size() ? programs.warm_permutations[index] : nullptr;
	};
	const auto decode = [&](uint32_t index, LocalShaderWarmup::Record& record) {
		LocalShaderWarmup::Reader reader {programs.warmup.records[index]};
		return LocalShaderWarmup::Visit(reader, record);
	};
	// Recipes are resolved and deduplicated here, inserting empty pipelines; the driver
	// compilations (the slow part with a cold driver cache) then fill them in place on
	// KYTY_SHADER_WARMUP_THREADS workers. Nothing else uses the maps before the renderer runs.
	struct Job {
		LocalShaderWarmup::PipelineRecord recipe;
		const ProgramCache::Permutation*  vertex = nullptr;
		const ProgramCache::Permutation*  pixel = nullptr;
		const ProgramCache::Permutation*  compute = nullptr;
		Pipeline*                         pipeline = nullptr;
	};
	std::vector<Job> jobs;
	jobs.reserve(programs.warmup.pipelines.size());
	std::atomic<size_t> skipped {0};
	for (const auto& words: programs.warmup.pipelines) {
		Job job;
		LocalShaderWarmup::Reader reader {words};
		if (!LocalShaderWarmup::VisitPipeline(reader, job.recipe)) { ++skipped; continue; }
		if (job.recipe.compute != LocalShaderWarmup::NoShader) {
			job.compute = get(job.recipe.compute);
			if (!job.compute) { ++skipped; continue; }
			auto [entry, inserted] = m_compute_pipelines.try_emplace(job.compute->handle.id);
			if (!inserted) continue;
			entry->second = std::make_unique<Pipeline>();
			job.pipeline  = entry->second.get();
		} else {
			job.vertex = get(job.recipe.vertex);
			job.pixel  = get(job.recipe.pixel);
			if (!job.vertex || (job.recipe.pixel != LocalShaderWarmup::NoShader && !job.pixel)) { ++skipped; continue; }
			GraphicsPipelineKey key {job.recipe.rendering, job.vertex->handle.id, job.pixel ? job.pixel->handle.id : 0,
			                         job.recipe.vertex_input, job.recipe.state};
			auto [entry, inserted] = m_graphics_pipelines.try_emplace(std::move(key));
			if (!inserted) continue;
			entry->second = std::make_unique<Pipeline>();
			job.pipeline  = entry->second.get();
		}
		jobs.push_back(std::move(job));
	}

	std::atomic<size_t> next {0}, compiled {0};
	const auto work = [&] {
		for (size_t n; (n = next.fetch_add(1, std::memory_order_relaxed)) < jobs.size();) {
			if (std::chrono::steady_clock::now() >= programs.warm_deadline) return;
			auto& job = jobs[n];
			if (job.compute) {
				LocalShaderWarmup::Record input;
				if (!decode(job.recipe.compute, input)) { ++skipped; continue; }
				input.compute.stage.program = &job.compute->program;
				CreatePipelineInternal(m_graphics, *job.pipeline, input.compute, job.compute->handle.module, m_driver_cache);
			} else {
				LocalShaderWarmup::Record vs, ps;
				if (!decode(job.recipe.vertex, vs) || (job.pixel && !decode(job.recipe.pixel, ps))) { ++skipped; continue; }
				vs.vertex.stage.program = &job.vertex->program;
				if (job.pixel) ps.pixel.stage.program = &job.pixel->program;
				CreatePipelineInternal(m_graphics, *job.pipeline, job.recipe.rendering, job.recipe.vertex_input,
					vs.vertex, job.vertex->handle, job.pixel ? &ps.pixel : nullptr,
					job.pixel ? job.pixel->handle : ShaderProgram {}, job.recipe.state, m_driver_cache);
			}
			if (const auto done = compiled.fetch_add(1, std::memory_order_relaxed) + 1; done % 250 == 0)
				PipelineCacheLog("Pipeline warmup: compiling {}/{}", done, jobs.size());
			StartupProgress::Report("Preparing pipelines", n + 1, jobs.size()); // shown from the main thread only
		}
	};
	const uint32_t threads = std::min<uint32_t>(ProgramCache::WarmupThreads(),
	                                            static_cast<uint32_t>(std::max<size_t>(jobs.size(), 1)));
	{
		std::vector<std::jthread> pool;
		for (uint32_t t = 1; t < threads; ++t) pool.emplace_back(work);
		work();
	}
	// Past the deadline, or with an undecodable input, an entry stays empty.
	std::erase_if(m_compute_pipelines, [](const auto& entry) { return entry.second->pipeline == nullptr; });
	std::erase_if(m_graphics_pipelines, [](const auto& entry) { return entry.second->pipeline == nullptr; });
	programs.warm_permutations.clear();
	PipelineCacheLog("Pipeline warmup: compiled {} of {} recipes, skipped {}, elapsed {} ms ({} threads)",
		compiled.load(), programs.warmup.pipelines.size(), skipped.load(),
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(),
		threads);
}

PipelineCache::~PipelineCache() {
	FinishCompileWorkers();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			if (pipeline->optimized != nullptr) {
				m_graphics.device.destroyPipeline(pipeline->optimized->pipeline, nullptr);
			}
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	m_graphics.pipeline_binaries = nullptr;
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

// Its pipelines as the driver's binaries (VK_KHR_pipeline_binary), read when needed: the driver keeps a
// copy of a whole pipeline cache in memory. Headed by the GPU, the driver and its global pipeline key.
static std::filesystem::path StaticBinariesPath() {
	return std::filesystem::path("_PipelineCache") / "static" / (PipelineCacheGameId() + ".binaries");
}

static std::string StaticBinariesSignature(GraphicContext& graphics) {
	return PipelineBinaries::Signature(graphics.device, ShaderInputDeviceSignature(graphics.GetPhysicalDeviceProperties()));
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
	// KYTY_STATIC_PIPELINE_CACHE=0: without it (first-encounter comparisons).
	if (const char* value = std::getenv("KYTY_STATIC_PIPELINE_CACHE"); !create && value != nullptr &&
	    std::string_view(value) == "0")
		return;
	const auto begin = std::chrono::steady_clock::now();
	if (m_static_binaries != nullptr) return;
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
	// 1 GiB (one build's warmup is ~130 MB) it starts over.
	m_driver_cache = LoadDriverCache(m_driver_cache_path, uint64_t {1} << 30u, m_driver_cache_saved_size);
	if (m_driver_cache == nullptr) return;
	// Table mode pipelines (src/local/table-xpr.inc) in a cache of their own: the static precompile holds none of
	// them (each compiles in ~0.1-4 s), they are saved once their compiles settle (not only at a clean exit), and
	// they do not push the main cache past its limit.
	m_table_cache_path = m_driver_cache_path;
	m_table_cache_path.replace_extension(".table.bin");
	m_table_cache = LoadDriverCache(m_table_cache_path, uint64_t {2} << 30u, m_table_cache_saved_size);
	if (m_table_cache == nullptr) return;
	m_table_cache_saver = std::jthread([this](std::stop_token stop) {
		LocalPlatform::SetThreadName("Kyty.TableCache");
		LocalPlatform::MakeBackgroundThread(std::getenv("KYTY_RENDER_CPUS"));
		std::mutex                  mutex;
		std::condition_variable_any wake;
		uint64_t                    saved      = m_table_pipelines_done.load();
		uint64_t                    main_saved = m_main_pipelines_done.load();
		const auto settled = [](const std::atomic<int64_t>& last_done) {
			const auto last = std::chrono::steady_clock::time_point(
			    std::chrono::steady_clock::duration(last_done.load(std::memory_order_acquire)));
			return std::chrono::steady_clock::now() - last >= std::chrono::seconds(20);
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
			    settled(m_table_pipeline_last_done)) {
				std::lock_guard save(m_table_cache_mutex);
				if (SaveDriverCache(m_table_cache, m_table_cache_path, m_table_cache_saved_size)) saved = done;
			}
			if (const auto done = m_main_pipelines_done.load(std::memory_order_acquire);
			    done != main_saved && settled(m_main_pipeline_last_done)) {
				std::lock_guard save(m_main_cache_mutex);
				if (SaveDriverCache(m_driver_cache, m_driver_cache_path, m_driver_cache_saved_size)) main_saved = done;
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
			PipelineCacheLog("Vulkan pipeline cache: starting {} over ({} bytes)", path, file_size);
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
	return ShaderRecompiler::IR::TraceLinearSrtReads(found->second.first->resource_plan, runtime, trace);
}

ShaderRecompiler::IR::SrtUserDataUse PipelineCache::UserDataUse(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	Common::LockGuard lock(m_mutex);
	auto&             cache = *m_program_cache;
	const auto        found = FindProgramSource(cache, program);
	return found == cache.by_program.end() ? ShaderRecompiler::IR::SrtUserDataUse {}
	                                       : ShaderRecompiler::IR::UserDataUse(found->second.first->resource_plan);
}

bool PipelineCache::RematerializeStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                       std::span<const uint32_t> user_data, uint64_t shader_base,
                                       ShaderRecompiler::IR::ResourceSnapshot& resources) {
	Common::LockGuard lock(m_mutex);
	auto&             cache = *m_program_cache;
	const auto        found = FindProgramSource(cache, program);
	if (found == cache.by_program.end()) return false;
	auto& [source, permutation] = found->second;
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
		std::printf("SLOW CreateGraphicsPipeline %.1f ms VS=%llu PS=%llu\n", ms,
		            static_cast<unsigned long long>(vs_id), static_cast<unsigned long long>(ps_id));
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
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vs_input_info,
	                       vertex_program, ps_input_info, pixel_program, static_params,
	                       m_driver_cache, native_bindings, FirstBuild(m_graphics));
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
	BuildOptimized(*iter->second, [this, rendering, vertex_input = iter->first.vertex_input, vs_input_info,
	                               pixel_info, vertex_program, pixel_program, static_params,
	                               native_bindings](Pipeline& layouts) {
		CreatePipelineInternal(m_graphics, layouts, rendering, vertex_input, vs_input_info, vertex_program,
		                       pixel_info ? &*pixel_info : nullptr, pixel_program, static_params, m_driver_cache,
		                       native_bindings, PipelineBuild::Optimize);
	});
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
		std::printf("SLOW CreateComputePipeline %.1f ms CS=%llu\n", ms,
		            static_cast<unsigned long long>(compute_program.id));
	}, SlowLog::HitchThreshold());
	LiveCensus::WaitScope compiling(LiveCensus::WaitCompile);
	// (The render thread's own waits: the slow frame lines' translate/pipeline counts.)
	std::optional<LiveCounters::ScopedUs> compute_timed;
	if (LiveCensus::g_render) {
		LiveCounters::Add(LiveCounters::ComputePipelines);
		compute_timed.emplace(LiveCounters::ComputePipelineUs);
	}
	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache,
	                       FirstBuild(m_graphics));
	MainPipelineDone();

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	BuildOptimized(*iter->second, [this, input_info, module = compute_program.module](Pipeline& layouts) {
		CreatePipelineInternal(m_graphics, layouts, input_info, module, m_driver_cache, PipelineBuild::Optimize);
	});
	LocalShaderWarmup::PipelineRecord recipe;
	recipe.compute = m_program_cache->RecordedIndex(compute_program);
	if (recipe.compute != LocalShaderWarmup::NoShader) {
		m_program_cache->warmup.AddPipeline(recipe);
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
