#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
struct LocalDescriptorPlan;
#endif
class CommandBuffer;
class PipelineBinaries;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       depth_bounds_test_enable = false;
	float                      depth_min_bounds         = 0.0f;
	float                      depth_max_bounds         = 0.0f;
	bool                       stencil_test_enable      = false;
	PipelineStencilStaticState stencil_front;
	PipelineStencilStaticState stencil_back;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
	bool                       blend_bypass[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 166);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	bool Save();

#ifdef KYTY_STATIC_PRECOMPILE
	// The static precompile, a program of its own (src/local/shader-precompile.cpp): a seed file's
	// shaders and pipelines (tools/local/static-precompile) into the static pipeline cache.
	struct PrecompileOptions {
		std::filesystem::path seeds;
		std::filesystem::path out;     // what was compiled, as a warmup file (optional)
		bool                  static_inputs = false; // out: the one the emulator's shader prefetch reads
		std::filesystem::path timings; // each pipeline's compile time (optional)
		uint32_t              shard   = 0;
		uint32_t              shards  = 1;
		uint32_t              threads = 1;
		bool                  pipelines = true;
	};
	// left_out: pipelines whose binaries were left out (PipelineBinaryWriter), for a smaller shard.
	static bool Precompile(GraphicContext& graphics, const PrecompileOptions& options, size_t& left_out);
	// The shader prefetch's inputs (--static-inputs) are this GPU's and driver's, newer than the seeds;
	// the static pipeline cache is this GPU's and driver's.
	static bool StaticInputsCurrent(GraphicContext& graphics, const std::filesystem::path& seeds);
	static bool StaticCacheCurrent(GraphicContext& graphics);
	// The caches the shards of a precompile saved next to the static cache, merged into it. With
	// pipeline binaries and `prune`, the store becomes the shards' pipelines only (a whole run's: what
	// older runs left that no seed makes any more is dropped).
	static bool MergePrecompileShards(GraphicContext& graphics, bool prune);
#endif

	// A pipeline no cache holds, compiled without optimization so its draw or dispatch need not wait:
	// a worker compiles the optimized one, which then replaces it (PromoteOptimized).
	struct OptimizedBuild {
		std::atomic<bool> done {false};
		vk::Pipeline      pipeline = nullptr;
	};

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// Native XPR variant: ordinary set, flattened SRT/shader data dynamic.
		bool                    native_bindings       = false;
		// PipelineBuild::Fast compiled it unoptimized; `optimized` is its optimized build.
		bool                            unoptimized = false;
		std::shared_ptr<OptimizedBuild> optimized;
#ifdef KYTY_LOCAL_VULKAN_RECORDING
		// Immutable encoding plan; queued packets keep their own shared ownership.
		mutable std::shared_ptr<const LocalDescriptorPlan> local_descriptor_plan;
#endif
	};

	// Whether a program of this thread failed to evaluate its resource tables since the last call (the
	// draw or dispatch is then skipped), and clears it.
	[[nodiscard]] static bool TakeMaterializationFailure();

	struct GraphicsPrograms {
		ShaderProgram vertex;
		ShaderProgram pixel;
	};

	// `table_mode`: both programs in table mode (ShaderRecompiler::IR::EnterTableMode), each empty when it does not
	// allow that.
	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, ShaderVertexInputInfo& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, bool table_mode = false,
	                    std::array<uint32_t, 2> table_portable = {});
	// `unevaluated`: a resource table that does not evaluate returns an empty program with the error here
	// (an indirect dispatch decides) instead of stopping the emulator.
	// `table_mode`: the program in table mode (ShaderRecompiler::IR::EnterTableMode), empty when it does not allow
	// it or the normal path has not prepared the shader yet.
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                std::string* unevaluated = nullptr, bool table_mode = false,
	                                uint32_t table_portable = 0);

	Pipeline&
	CreateGraphicsPipeline(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                       const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
	                       const ShaderPixelInputInfo* ps_input_info,
	                       vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                       const ShaderProgram& vertex_program, const ShaderProgram& pixel_program,
	                       bool native_bindings = false);
	// The same pipeline without blocking the caller on the driver compile: null while a
	// background worker creates it (the native XPR variants, whose draws take the normal path
	// until then). A later call with the same state returns the finished pipeline.
	[[nodiscard]] Pipeline*
	TryCreateGraphicsPipeline(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                          const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
	                          const ShaderPixelInputInfo* ps_input_info,
	                          vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                          const ShaderProgram& vertex_program, const ShaderProgram& pixel_program,
	                          bool native_bindings);
	Pipeline& CreateComputePipeline(const ShaderComputeInputInfo& input_info,
	                                const ShaderProgram&          compute_program);
	// The last GetGraphicsPrograms or GetComputeProgram in table mode found a program a worker still translates
	// (its empty program is no refusal: asked again, it is there).
	[[nodiscard]] bool TablePending() const;
	// When TablePending: the completion of that translation (set once it is done).
	[[nodiscard]] std::shared_ptr<const std::atomic<bool>> TableWaiting() const;
	// When TryCreateGraphicsPipeline or TryCreateComputePipeline returned null: the completion of that compile.
	[[nodiscard]] const std::shared_ptr<const std::atomic<bool>>& PipelineWaiting() const { return m_pipeline_waiting; }
	// As TryCreateGraphicsPipeline: null while a worker compiles it; its layout takes allocated descriptor sets
	// (native bindings, for table mode programs).
	[[nodiscard]] Pipeline* TryCreateComputePipeline(const ShaderComputeInputInfo& input_info,
	                                                 const ShaderProgram&          compute_program);
	// The optimized builds finished since replace their unoptimized pipelines (at each flip: the draws of native
	// XPR records hold their pipelines and look nothing up). GPU thread.
	void PromoteFinished();
	// Native XPR records (src/local/native-xpr.inc): the SRT evaluation of a stage
	// whose compiled permutation is already chosen, with the readers the normal
	// path uses. False when the evaluation fails or would select another
	// permutation (a different resource specialization).
	[[nodiscard]] bool RematerializeStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                                      std::span<const uint32_t> user_data, uint64_t shader_base,
	                                      ShaderRecompiler::IR::ResourceSnapshot& resources);
	// The same stage's reads split into data and structural (TraceLinearSrtReads).
	[[nodiscard]] bool TraceStage(const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                              std::span<const uint32_t> user_data, uint64_t shader_base,
	                              ShaderRecompiler::IR::SrtReadTrace& trace);
	// How its SRT evaluation uses user data (ShaderRecompiler::IR::UserDataUse).
	[[nodiscard]] ShaderRecompiler::IR::SrtUserDataUse UserDataUse(const ShaderRecompiler::IR::CompiledShaderInfo& program);

	// The shader prefetch (ProgramCache::Prefetch): programs translated or taken over, of `total`;
	// all zero while it loads its inputs or when it does not run. Any thread.
	struct PrefetchProgress {
		size_t done = 0, total = 0;
	};
	[[nodiscard]] PrefetchProgress GetPrefetchProgress() const;

private:
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		uint64_t                 vs_shader_id = 0;
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;
		bool                     native_bindings = false;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vs_shader_id == other.vs_shader_id &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params &&
			       native_bindings == other.native_bindings;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}

		static void MixStaticParams(std::size_t& hash, const PipelineStaticParameters& params) {
			const auto* bytes = reinterpret_cast<const uint8_t*>(&params);
			for (std::size_t i = 0; i < sizeof(params); i++) {
				Mix(hash, bytes[i]);
			}
		}

		static void MixRendering(std::size_t& hash, const PipelineRenderingState& rendering) {
			Mix(hash, rendering.color_count);
			for (uint32_t i = 0; i < rendering.color_count; i++) {
				Mix(hash, static_cast<uint32_t>(rendering.color_formats[i]));
			}
			Mix(hash, static_cast<uint32_t>(rendering.depth_format));
			Mix(hash, static_cast<uint32_t>(rendering.stencil_format));
		}
	};

	struct GraphicsPipelineKeyHash {
		static std::size_t Prefix(const GraphicsPipelineKey& key) {
			std::size_t hash = 0;
			PipelineKeyHash::MixRendering(hash, key.rendering);
			PipelineKeyHash::Mix(hash, key.vs_shader_id);
			PipelineKeyHash::Mix(hash, key.ps_shader_id);
			PipelineKeyHash::Mix(hash, key.native_bindings);
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
			return hash;
		}
		std::size_t operator()(const GraphicsPipelineKey& key) const {
			auto hash = Prefix(key);
			PipelineKeyHash::MixStaticParams(hash, key.static_params);
			return hash;
		}
	};
	struct NativeGraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	// The static precompile's pipelines (_PipelineCache/static), kept across emulator builds:
	// GraphicContext::static_pipeline_cache while loaded.
	vk::PipelineCache             m_static_cache = nullptr;
	// Or its binaries (pipelineBinaries.h): GraphicContext::pipeline_binaries while open.
	std::unique_ptr<PipelineBinaries> m_static_binaries;
	std::filesystem::path         m_driver_cache_path;
	std::string                   m_driver_cache_key;
	// The size of the driver cache's data when it was loaded or last saved (Save skips an unchanged one).
	size_t                        m_driver_cache_saved_size = 0;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	// Pipelines remain owned by m_graphics_pipelines until this cache is destroyed.
	// Both indices use the complete GraphicsPipelineKey equality predicate.
	std::unordered_map<GraphicsPipelineKey, Pipeline*, NativeGraphicsPipelineKeyHash>
	    m_native_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	Common::Mutex m_mutex;
	// TryCreateGraphicsPipeline: pipelines a worker is still compiling, by key.
	struct PendingGraphicsPipeline;
	class CompileWorkers;
	std::unordered_map<GraphicsPipelineKey, std::shared_ptr<PendingGraphicsPipeline>, GraphicsPipelineKeyHash>
	                                m_pending_graphics_pipelines;
	struct PendingComputePipeline;
	std::unordered_map<uint64_t, std::shared_ptr<PendingComputePipeline>> m_pending_compute_pipelines;
	std::unique_ptr<CompileWorkers> m_compile_workers;
	std::shared_ptr<const std::atomic<bool>> m_pipeline_waiting; // (PipelineWaiting)
	// Table mode programs and pipelines (src/local/table-xpr.inc): a new area brings hundreds, which ahead of the
	// native records' pipelines in one queue kept the draws on the normal path for a minute.
	std::unique_ptr<CompileWorkers> m_table_workers;
	// The optimized builds of pipelines in use unoptimized (BuildOptimized): background threads of their own, so a
	// pipeline a draw waits for is never queued behind them (0.1-4 s each).
	std::unique_ptr<CompileWorkers> m_optimize_workers;
	// Unoptimized pipelines whose optimized build is pending, and those it replaced (recorded commands
	// may still use them).
	std::vector<Pipeline*>    m_optimizing;
	std::vector<vk::Pipeline> m_replaced_pipelines;
	uint32_t                  m_unoptimized_builds = 0;
	std::atomic<uint32_t>     m_optimized_builds {0}; // finished by workers
	uint32_t                  m_promoted_builds = 0;  // of those, seen by PromoteOptimized
	std::atomic<bool>         m_stopping {false};     // optimized builds not started yet are skipped
	// Table mode pipelines' driver cache (InitializeDriverCache), saved by m_table_cache_saver once their compiles
	// settle: the jobs queued or running, those done and when the last one finished (steady clock ticks).
	vk::PipelineCache     m_table_cache = nullptr;
	std::filesystem::path m_table_cache_path;
	size_t                m_table_cache_saved_size = 0;
	std::mutex            m_table_cache_mutex; // (a save)
	std::atomic<uint32_t> m_table_pipeline_jobs {0};
	std::atomic<uint64_t> m_table_pipelines_done {0};
	std::atomic<int64_t>  m_table_pipeline_last_done {0};
	// The main driver cache's compiles, for the same saver: a session that does not end at the window (killed, or
	// a crash) lost every pipeline it compiled, and the next one compiled them again (hundreds of native record
	// variants when entering an area, ~100 ms each).
	std::mutex            m_main_cache_mutex; // (a save)
	std::atomic<uint64_t> m_main_pipelines_done {0};
	std::atomic<int64_t>  m_main_pipeline_last_done {0};
	std::jthread          m_table_cache_saver;

	Pipeline* CreateGraphicsPipelineImpl(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                                     const ShaderVertexInputInfo& vs_input_info, CommandBuffer& command,
	                                     const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
	                                     bool primitive_restart_enable, const ShaderProgram& vertex_program,
	                                     const ShaderProgram& pixel_program, bool native_bindings, bool async);
	void      FinishCompileWorkers();
	CompileWorkers& Workers();
	CompileWorkers& TableWorkers();
	CompileWorkers& OptimizeWorkers();
	void            BuildOptimized(Pipeline& pipeline, std::function<void(Pipeline&)> build);
	void            PromoteOptimized();
	void InitializeDriverCache();
	vk::PipelineCache LoadDriverCache(const std::filesystem::path& file_path, uint64_t limit, size_t& loaded_size);
	bool SaveDriverCache(vk::PipelineCache cache, const std::filesystem::path& file_path, size_t& saved_size);
	vk::PipelineCache TablePipelineCache() const { return m_table_cache != nullptr ? m_table_cache : m_driver_cache; }
	void              TablePipelineDone();
	void              MainPipelineDone();
	void InitializeStaticCache(bool create);
	void WarmPipelines();
#ifdef KYTY_STATIC_PRECOMPILE
	bool SaveStaticCache(vk::PipelineCache cache, const std::filesystem::path& path);
	static bool MergeBinaryShards(GraphicContext& graphics, bool prune);
	bool WarmSeeds(const PrecompileOptions& options, size_t& left_out);
#endif
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// A pipeline the static cache lacks is compiled (Full); or taken from the driver cache, else compiled
// unoptimized (Fast: Pipeline::unoptimized); Optimize compiles the optimized build of a Fast one (its
// layouts given, `pipeline` null).
enum class PipelineBuild { Full, Fast, Optimize };
void CreatePipelineInternal(
    GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
    const PipelineRenderingState& rendering, const PipelineVertexInputState& vertex_input,
    const ShaderVertexInputInfo& vs_input_info, const ShaderProgram& vertex_program,
    const ShaderPixelInputInfo* ps_input_info, const ShaderProgram& pixel_program,
    const PipelineStaticParameters& static_params, vk::PipelineCache driver_cache,
    bool native_bindings = false, PipelineBuild build = PipelineBuild::Full);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache,
                            PipelineBuild build = PipelineBuild::Full, bool native_bindings = false);
// The guest memory readers resource materialization reads through (clean reads: what the GPU wrote is synchronized
// first), for the table path's address probes (TablePlan::IndirectImage).
void MaterializationReaders(ShaderRecompiler::IR::SrtRuntime& runtime);
// KYTY_PIPELINE_KEY_LOG (diagnostic): a PIPEKEY line with the driver key of each pipeline the static store did not
// hold (the precompile: of each it makes) and its modules' shader and SPIR-V hashes.
bool PipelineKeyLog();
void NotePipelineKeyModule(vk::ShaderModule module, uint64_t shader_hash, uint64_t spirv_hash);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
