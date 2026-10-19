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

#include <array>
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
#include <unordered_set>
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
class PipelineCacheBlobs;

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
		// No pipeline created: the timings list what would be (0 ms), each with its key (what makes two the same).
		bool                  dry_run = false;
		// The run's plan: each pipeline key of its seed files with the work item that makes it first. A dry run adds
		// the seed file's keys to it; a shard creates only the pipelines whose first work item is its own.
		std::filesystem::path plan;
		// With a dry run (--covered-by): the seed file of recorded play's specializations (a recording, or the release's
		// hints) planned before this one. This file's work items whose programs it all holds - a compute seed whose
		// shader it has, a recipe both of whose shaders it has - are planned as skipped: their guessed specializations
		// of programs whose real ones are known. (1.07 on a laptop: a fifth of a full run's compile time; two 1-1 walks
		// met 1 such pipeline of 2,227 and 0 of 2,130, every other one recorded play's.) One also needed by a work item
		// it does not cover is made there.
		std::filesystem::path covered_by;
		// Native XPR variants of a seed file without recorded specializations too (its guesses; see WarmSeeds).
		bool                  native_guesses = false;
		// Native XPR variants at all (--native-variants): without it none are made. The emulator builds a missing one
		// in the background (TryCreateGraphicsPipeline, KYTY_ASYNC_XPR_PIPELINES: run-windows.ps1's default), its draws
		// taking the normal path meanwhile; native records draw well under 1% of a frame's draws since table mode.
		// (1.07: 18,935 of the recorded specializations' 89,091 pipelines; a fresh 1-1 start built ~170 of them in the
		// background, with fewer slow frames than before.)
		bool                  native_variants = false;
		// Compute seeds translated for this host subgroup size instead of the one they were listed or recorded with
		// (0: as they were): kyty_shader_precompile passes the GPU's (GetComputeProgram's), --amd what a GPU running
		// wave64 programs natively (AMD RDNA3) gets.
		uint32_t              host_subgroup_size = 0;
	};
	// left_out: pipelines whose binaries were left out (PipelineBinaryWriter), for a smaller shard.
	static bool Precompile(GraphicContext& graphics, const PrecompileOptions& options, size_t& left_out);
	// The shader prefetch's inputs (--static-inputs) are this GPU's and driver's, newer than the seeds;
	// the static pipeline cache is this GPU's and driver's and, unless seeds is empty, newer than the seeds too (a
	// newer precompile program lists more and makes the seeds again: what is new is compiled then).
	static bool StaticInputsCurrent(GraphicContext& graphics, const std::filesystem::path& seeds);
	static bool StaticCacheCurrent(GraphicContext& graphics, const std::filesystem::path& seeds);
	// The caches the shards of a precompile saved next to the static cache, merged into it. With
	// pipeline binaries and `prune`, the store becomes the shards' pipelines only (a whole run's: what
	// older runs left that no seed makes any more is dropped).
	// `keep_shards`: the shard files stay (a merge in the middle of a run, which its final pruning merge counts).
	static bool MergePrecompileShards(GraphicContext& graphics, bool prune, bool keep_shards = false, bool stamp = false);
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
		// Out of video memory while idle (EvictIdlePipelines): the flip of its last binding (BindPipelineHandle), and how
		// to make it again (a pipeline the static store gave: made again from it; null: never evicted).
		mutable uint32_t                                       last_flip = 0;
		std::shared_ptr<const std::function<void(Pipeline&)>> remake;
		// Its warmup recipe (LocalShaderWarmup::RecipeKey; 0: none, a native variant), and the scene generation it was
		// last bound in (local/scene-tags.h: noted once per scene).
		uint64_t         recipe      = 0;
		mutable uint64_t scene_noted = 0;
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
	// Each flip (from the CPU or the GPU's commands): optimized builds in place, shader plans unused for a while out of
	// memory.
	void PromoteFinished();
	// The session's numbers in the log (shader prefetch, unoptimized first builds, shader plans), once: at the window's
	// close, before the process may end without this cache's destructor.
	void ReportSession();
	// Live "objects": what the cache keeps, as counts (stdout).
	void ReportObjects(uint64_t id) const;
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
		size_t      done = 0, total = 0;
		const char* label = nullptr; // a warmup beside the game's phase (null: the prefetch's progress)
	};
	// Stops a warmup still running beside the game, and waits for it (the window's close; not under m_mutex).
	void StopWarmup();
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
	std::unique_ptr<PipelineCacheBlobs> m_static_blobs;
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
	bool      m_session_reported = false; // (ReportSession)
	// The start-up warmup (the constructor's choices): in front of the game, or beside it on m_warmup_thread
	// (KYTY_SHADER_WARMUP_BACKGROUND) until done or m_warmup_stop.
	struct WarmupSettings {
		bool     compile_shaders = false, warm_pipelines = false, background = false;
		uint64_t compute_video = UINT64_MAX;
	};
	void              RunWarmup(WarmupSettings settings);
	void              StartShaderPrefetch(std::vector<std::vector<uint32_t>> recorded_inputs);
	std::thread       m_warmup_thread;
	std::atomic<bool> m_warmup_stop {false};
	// The warmup by scene (local/scene-tags.h, KYTY_SCENE_WARMUP): the recording's file (its .scenes beside it), whether
	// it is on, and what it made. WarmScene: the pipelines of `scene` (every recorded one for a scene not known yet), from
	// the warmup's own records (`in_memory`, the first scene) or the recording read again; SceneLoop: those of each scene
	// the game loads later, until m_warmup_stop.
	std::filesystem::path        m_warmup_path;
	bool                         m_scene_warmup     = false;
	bool                         m_scene_warmed_all = false;
	std::unordered_set<uint32_t> m_scenes_warmed;
	void                         WarmScene(uint32_t scene, uint64_t compute_video, bool in_memory);
	void                         SceneLoop(uint64_t compute_video);
	// Pipelines made where first needed (CreateGraphicsPipeline, CreateComputePipeline), and the time it took.
	uint64_t m_render_pipelines   = 0;
	double   m_render_pipeline_ms = 0.0;
	CompileWorkers& Workers();
	CompileWorkers& TableWorkers();
	CompileWorkers& OptimizeWorkers();
	void            BuildOptimized(Pipeline& pipeline, std::function<void(Pipeline&)> build);
	void            PromoteOptimized();
	// At each flip: with the video memory in use near the GPU's budget, the pipelines no binding used for a while leave
	// it (their handles destroyed; BindPipelineHandle makes one again from the static store when it is bound); and a
	// while after the game changed its world (SceneTags), those of the worlds it left.
	void            EvictIdlePipelines();
	// (EvictIdlePipelines) A graphics pipeline's inputs, as its remake keeps them.
	struct GraphicsMakeInputs;
	std::shared_ptr<const std::function<void(Pipeline&)>> GraphicsRemake(GraphicsMakeInputs inputs);
	// (EvictIdlePipelines) When each recent flip began (steady clock, by flip % size), and the video memory pressure.
	std::array<int64_t, 32768> m_flip_times {};
	bool                      m_evict_pressure = false;
	// (EvictIdlePipelines) The world the game is in: its SceneTags generation, the flip and time it began, and whether
	// the pipelines of the worlds before it went.
	uint64_t                              m_scene_generation = 0;
	uint32_t                              m_scene_flip       = 0;
	std::chrono::steady_clock::time_point m_scene_since {};
	bool                                  m_scene_swept = true;
	void InitializeDriverCache();
	vk::PipelineCache LoadDriverCache(const std::filesystem::path& file_path, uint64_t limit, size_t& loaded_size);
	bool SaveDriverCache(vk::PipelineCache cache, const std::filesystem::path& file_path, size_t& saved_size);
	vk::PipelineCache TablePipelineCache() const { return m_table_cache != nullptr ? m_table_cache : m_driver_cache; }
	void              TablePipelineDone();
	void              MainPipelineDone();
	void InitializeStaticCache(bool create);
	// The recorded pipelines; compute ones, newest first, while they took less than `compute_video` bytes of video memory.
	// `only`: of these recipes alone (LocalShaderWarmup::RecipeKey), none the cache has.
	void WarmPipelines(uint64_t compute_video, bool background, const std::unordered_set<uint64_t>* only = nullptr);
#ifdef KYTY_STATIC_PRECOMPILE
	bool SaveStaticCache(vk::PipelineCache cache, const std::filesystem::path& path);
	static bool MergeBinaryShards(GraphicContext& graphics, bool prune, bool keep_shards);
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
// The session's line of what the game's pipelines got from the static store, its binaries or blobs (shaders.cpp): made
// of what it holds, not among them, refused by the driver.
void ReportStaticStore();
// Whether the static store gave the last pipeline this thread made (shaders.cpp).
bool LastPipelineFromStore();
// The pipeline's handle for a binding now (the GPU thread): the flip noted, and the pipeline made again first when it was
// evicted (PipelineCache::EvictIdlePipelines).
vk::Pipeline BindPipelineHandle(const PipelineCache::Pipeline& pipeline);
// zstd on the calling thread's own contexts (shaders.cpp): ZSTD_compress and ZSTD_decompress make one for every call, the
// compressor's ~0.5 MiB (the shader prefetch's threads allocated 2.9 GB of them a minute, memory the system zero-fills).
// ZstdPack: `data` compressed into the thread's buffer, valid until the thread's next ZstdPack (it only grows, and is not
// zeroed again); empty when zstd fails. ZstdUnpack: `packed` into `out`, whether it gave exactly `bytes` bytes.
std::span<const char> ZstdPack(const void* data, size_t bytes, int level = 1);
bool                  ZstdUnpack(std::span<const char> packed, void* out, size_t bytes);
void NotePipelineKeyModule(vk::ShaderModule module, uint64_t shader_hash, uint64_t spirv_hash, size_t spirv_bytes = 0);
// (KYTY_INLINE_KEY_CHECK) Each module's SPIR-V kept, so that every pipeline key is compared with the key of the same
// pipeline given its shaders' SPIR-V inline (VK_KHR_maintenance5): what a store keyed by the first needs of the second.
bool InlineKeyCheck();
void NoteInlineSpirv(vk::ShaderModule module, const std::vector<uint32_t>& spirv);
// The descriptor set and pipeline layouts the pipelines share (one pair per binding set, shaders.cpp): destroyed after
// every pipeline that uses them.
void DestroySharedLayouts(vk::Device device);
// KYTY_PIPELINE_SOURCE_LOG (diagnostic): a PIPESRC line for each pipeline made (its kind, identity, whether the static
// cache gave it, its build and time; from a cache blob: the blob's size and the microseconds of its read, of the cache
// made of it and of the driver's creation) and a PIPESPV line (its SPIR-V size and the microseconds before its
// identity, of the identity and of the handle), a PROGSRC line per program permutation made (adopted from the
// prefetch or translated, its time): which of the precompile's pipelines a game session meets, and what they cost.
bool PipelineSourceLog();
// Each shader module's SPIR-V hash is noted when made (NotePipelineKeyModule) from now on: pipeline identities need them
// (pipeline cache blobs, the source and key logs).
void RecordModuleSpirvHashes();
bool ModuleSpirvHashesRecorded();
// A pipeline's identity: its modules' SPIR-V, its fixed-function state (as a recipe holds it), its stages' static input
// keys and native bindings; the same for the same pipeline (the precompile's plan keys it so, the source log too).
uint64_t GraphicsPipelineIdentity(uint64_t vertex_spirv, uint64_t pixel_spirv, const PipelineRenderingState& rendering,
                                  const PipelineVertexInputState& vertex_input, const PipelineStaticParameters& state,
                                  const ShaderVertexInputInfo& vs_input_info, const ShaderPixelInputInfo* ps_input_info,
                                  bool native_bindings);
uint64_t ComputePipelineIdentity(uint64_t spirv, const ShaderComputeInputInfo& input_info, bool native_bindings);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
