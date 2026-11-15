#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <atomic>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct ResourceSpecialization;
struct ResourcePlan;
struct Program;
} // namespace ShaderRecompiler::IR

namespace ShaderRecompiler {
struct CompileOptions;
} // namespace ShaderRecompiler

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
struct ShaderParams;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace Precompile {
class JournalWriter;
class Precompiler;
class StageStore;
class WarmList;
struct ImportResult;
struct PipelineRecord;
struct TemplateRecord;
} // namespace Precompile

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
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

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 114);

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
	uint64_t         spirv_hash = 0; // XXH3 of its SPIR-V: graphics pipeline library parts are keyed by it

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The driver's pipeline cache file: the signature, the payload's hash, the payload. Read gives false for a missing,
// short, foreign or damaged file.
bool WriteDriverCacheFile(const std::filesystem::path& path, std::string_view signature,
                          std::span<const uint8_t> payload);
bool ReadDriverCacheFile(const std::filesystem::path& path, std::string_view signature, std::vector<uint8_t>& payload);
// Graphics pipeline library parts (KYTY_GPL, GraphicContext::SplitsFragmentSet), shared by the GPU thread and the
// ahead workers: a part another thread is building is waited for, not built twice.
class PipelineLibraries {
public:
	vk::Pipeline Get(uint64_t key, const std::function<vk::Pipeline()>& build);
	// A part's own layouts: a part may use them until it is linked, so they live as long as the cache.
	void         Keep(vk::PipelineLayout layout, std::initializer_list<vk::DescriptorSetLayout> sets);
	void         Destroy(vk::Device device);

private:
	std::mutex                                 m_mutex;
	std::condition_variable                    m_built;
	std::unordered_map<uint64_t, vk::Pipeline> m_parts;
	std::unordered_set<uint64_t>               m_building;
	std::vector<vk::PipelineLayout>            m_layouts;
	std::vector<vk::DescriptorSetLayout>       m_sets;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		// Upstream abc509d33: the task shader reads an indirect mesh draw's records; its stage is in the push range.
		bool                    task_stage            = false;
		// The pixel shader's own set (IR::FragmentSet) where GraphicContext::SplitsFragmentSet.
		vk::DescriptorSetLayout fragment_set_layout            = nullptr;
		bool                    fragment_uses_push_descriptors = false;
		// The image groups' own sets (IR::GroupSetFor): IR::GroupSet's (vertex or compute stage), IR::GroupSet + 1's (pixel).
		std::array<vk::DescriptorSetLayout, 2> group_set_layouts {};
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, ShaderIndirectOffsets indirect_offsets = {},
	                    bool indirect_draw = false);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);
	// A compute shader GetComputeProgram skips (ray tracing): its dispatch needs nothing read.
	[[nodiscard]] bool SkipsComputeShader(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);
	// Compute ahead (ComputeAheadScanner, its thread): builds a dispatch's pipeline on a worker.
	void BuildComputeAhead(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                       ShaderComputeInputInfo input_info);
	// Graphics ahead (the same thread): builds a draw's pixel shader with the states its vertex
	// shader was built with before.
	void BuildGraphicsAhead(const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	                        const HW::ShaderRegisters& sh, const HW::Context& context,
	                        const HW::UserConfig& user_config,
	                        std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                        ShaderIndirectOffsets indirect_offsets, bool indirect_draw);
	// The submission (GuestGpu's seq) the calling thread is in: the GPU thread's, or the one the scanner reads. A draw
	// the GPU thread found no ahead result for in a submission is not materialized by the scanner in that submission.
	static void AheadSubmission(uint64_t seq);
	// --precompile-only (shaderPrecompile.h): imports or exports hints when asked, compiles the journal into the
	// driver cache and saves it. The process's exit code: 0 done, 2 unavailable, 3 stopped, 4 save failed, 5 export
	// failed, 6 import failed.
	int RunPrecompileOnly();
	// Restores the hints' records from the game files with this GPU's host fields. Its callers stamp the file
	// (StampHints, m_hints_path) once the journal holding them is written, so a start does not import it again.
	Precompile::ImportResult ImportHintsFile(const std::filesystem::path&                         path,
	                                         const std::function<void(uint64_t done, uint64_t total)>& progress = {});

private:
	void StampHints(const std::filesystem::path& path);
	struct ProgramCache;
	class PrecompileCompiler;
	struct Ahead;
	struct AheadJob;
	struct AheadRegisters;
	enum class AheadStep : uint8_t;
	void RunGraphicsAhead(AheadJob& job);
	AheadStep PrepareGraphicsAhead(const AheadRegisters& draw, const HW::Context& context,
	                               const std::shared_ptr<AheadJob>& handed, std::shared_ptr<AheadJob>& build);
	AheadStep PrepareComputeAhead(const AheadRegisters& dispatch, ShaderComputeInputInfo input_info,
	                              const std::shared_ptr<AheadJob>& handed, std::shared_ptr<AheadJob>& build);
	uint32_t PrepareExactVertex(const AheadRegisters& draw, const HW::Context& context,
	                            const std::shared_ptr<AheadJob>& handed, AheadJob& job, AheadStep& step);
	void StartComputeAhead();
	void StopComputeAhead();
	void RunComputeAhead();
	void NoteGpuBuild(uint64_t identity);
	void WaitForComputeAhead(uint64_t shader_hash, uint64_t identity);

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	void NoteReady(ShaderType stage, uint64_t hash, std::vector<uint32_t> static_key,
	               const ShaderRecompiler::IR::ResourceSpecialization& specialization, const std::vector<uint32_t>& spirv,
	               const ShaderRecompiler::IR::CompiledShaderInfo& info);
	void NoteReadyPlan(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
	                   const ShaderRecompiler::IR::Program& translated);
	void NoteReadyPlan(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
	                   std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> plan);
	std::unique_ptr<ShaderRecompiler::IR::ResourcePlan> TakeReadyPlan(ShaderType stage, uint64_t hash,
	                                                                  const std::vector<uint32_t>& static_key);
	// The Ahead workers' takes from the stage store, in place of a translation: a program's resource plan (any record
	// of it), and one permutation's SPIR-V and compiled info, its plan noted for the GPU thread as a translation's is.
	bool StorePlan(ShaderType stage, const ShaderParams& params, const std::vector<uint32_t>& static_key,
	               const ShaderRecompiler::CompileOptions& options, ShaderRecompiler::IR::ResourcePlan& plan);
	bool StoreModule(ShaderType stage, const ShaderParams& params, const std::vector<uint32_t>& static_key,
	                 const ShaderRecompiler::CompileOptions& options, uint32_t push_data_cursor,
	                 const ShaderRecompiler::IR::ResourceSpecialization& specialization, std::vector<uint32_t>& spirv,
	                 ShaderRecompiler::IR::CompiledShaderInfo& info);
	bool TakeReadyModule(ShaderType stage, uint64_t hash, const std::vector<uint32_t>& static_key,
	                     const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t push_data_cursor,
	                     std::vector<uint32_t>& spirv, ShaderRecompiler::IR::CompiledShaderInfo& info, bool take = true);
	void RecordGraphicsTemplate(const GraphicsPipelineKey& key, std::span<const ShaderVertexInputInfo> vertex_info,
	                            const GraphicsPrograms& programs, uint64_t state_signature);
	void WaitForGraphicsAhead(uint64_t pixel_hash, uint64_t identity, const GraphicsPipelineKey& key);
	uint64_t GraphicsIdentity(std::span<const ShaderVertexInputInfo> vertex_info, const ShaderPixelInputInfo& pixel_info,
	                          const GraphicsPrograms& programs, const PipelineVertexInputState& vertex_input,
	                          uint64_t state_signature) const;
	GraphicsPipelineKey MakeGraphicsKey(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	                                    const HW::Context& ctx, bool rect_list, const ShaderPixelInputInfo* ps_input_info,
	                                    vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                    const GraphicsPrograms& programs) const;
	static PipelineVertexInputState MakeVertexInput(const ShaderVertexInputInfo& vs_input_info, bool mesh);
	std::array<ShaderParams, 3> PrepareVertexStages(const HW::VertexShaderInfo& vertex_regs, const HW::ShaderRegisters& sh,
	                                                const HW::Context& context, const HW::UserConfig& user_config,
	                                                bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                                                bool& object_primitive_id, uint32_t& push_data_cursor,
	                                                ShaderIndirectOffsets indirect_offsets = {},
	                                                bool indirect_draw = false,
	                                                bool* exceeds_host = nullptr) const;

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	std::unique_ptr<PipelineLibraries> m_libraries;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	std::atomic<bool>             m_driver_cache_dirty = false;
	std::future<void>             m_driver_cache_write; // the hash and write of the last checkpoint, on a worker
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	// The GPU thread's last pipeline: most draws use the pipeline of the draw before, and comparing costs less than
	// hashing the key. Entries of the map stay where they are, and none is erased while the cache lives.
	const GraphicsPipelineKey* m_last_graphics_key      = nullptr;
	Pipeline*                  m_last_graphics_pipeline = nullptr;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	std::unique_ptr<Ahead>                                  m_ahead;
	std::unique_ptr<PrecompileCompiler>                     m_precompile_compiler;
	std::unique_ptr<Precompile::JournalWriter>              m_journal;
	std::unique_ptr<Precompile::WarmList>                   m_warm;
	std::unique_ptr<Precompile::Precompiler>                m_precompiler;
	std::unique_ptr<Precompile::StageStore>                 m_stage_store; // translated stages (stageStore.h)
	// The journal's stage records without callees by program (StageProgramKey): one record's key each, for the Ahead
	// workers' plan. Written once at start, before m_stage_index_ready; the workers start before it.
	std::map<std::vector<uint32_t>, uint64_t>               m_stage_index;
	std::atomic<bool>                                       m_stage_index_ready {false};
	std::unordered_set<uint64_t>                            m_recorded_pipelines; // pipeline record keys in the journal
	// Pipelines the precompile created in this process, by pipeline record key, with their stages' SPIR-V hashes
	// (vertex stages, then pixel): the GPU thread takes one whose stages are its own instead of creating it again.
	struct ReadyPipeline {
		Pipeline                pipeline;
		std::array<uint64_t, 4> spirv {};
	};
	std::mutex                                              m_ready_mutex;
	std::unordered_map<uint64_t, ReadyPipeline>             m_ready_pipelines;
	std::deque<uint64_t>                                    m_ready_order; // their keys as handed over, oldest first
	bool                                                    m_precompile_started = false;
	uint64_t                                                m_driver_cache_loaded_bytes = 0;
	std::atomic<bool>                                       m_driver_cache_save_failed  = false;
	std::filesystem::path m_journal_path, m_warm_path, m_inflight_path, m_suspects_path, m_hints_path;

	void InitializeDriverCache();
	// The journal (recording), the warm list and the precompile on start (shaderPrecompile.h).
	void InitializePrecompile();
	void StartPrecompile();
	std::optional<Precompile::PipelineRecord> MakePipelineRecord(const GraphicsPipelineKey& key,
	                                                             const GraphicsPrograms& programs, bool ps_active) const;
	Precompile::TemplateRecord MakeTemplate(CommandBuffer& command, std::span<const RenderColorInfo> colors,
	                                        const RenderDepthInfo& depth, vk::PrimitiveTopology topology,
	                                        bool primitive_restart, const Precompile::PipelineRecord& record);
	void CheckpointDriverCache();
	// Takes the driver's data now and returns the hash and write of it; empty when there is nothing to save.
	std::function<void()> SnapshotDriverCache();
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// What a draw's pixel stage exports to (renderDraw.cpp): each target's export mapping, and
// whether the pixel stage runs. The draw path and graphics ahead both use it.
bool DrawPixelExports(const HW::Context& ctx, const HW::Shader& sh_ctx,
                      std::array<Prospero::ColorComponentMapping, 8>& mapping);
// A draw's color outputs: the targets its pixel stage exports to that the target mask lets through
// (renderDraw.cpp).
uint32_t DrawColorOutputMask(const HW::Context& ctx);
// A draw's Vulkan topology (renderDraw.cpp); false for a primitive type whose draws are skipped.
bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology);
// The format and samples of the depth attachment a draw with these registers renders to, as
// ResolveRenderDepthTarget finds them (depthRenderTarget.cpp); false when it binds none.
bool DrawDepthTargetFormat(GraphicContext& graphics, const HW::Context& hw, vk::Format& format, uint32_t& samples);
bool BuildsFromLibraries(const GraphicContext& graphics, std::span<const ShaderVertexInputInfo> vertex_info,
                         const ShaderPixelInputInfo* ps_input_info, const PipelineCache::GraphicsPrograms& programs,
                         const PipelineStaticParameters& static_params);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache,
                            PipelineLibraries*                     libraries = nullptr);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
