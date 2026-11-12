#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineSeeds.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CoverageVariants;
struct ShaderParams;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

namespace DrawPrep {
struct PacketResult;
} // namespace DrawPrep

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
	bool                       alpha_to_coverage_enable = false;
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
	// Round 159: PA_SC_AA_MASK reduced to the draw's samples, when the pixel shader's copy that
	// applies it is used (spirvCoverageMask.h); all ones otherwise.
	uint32_t aa_mask[2] = {0xffffffffu, 0xffffffffu};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 134);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
	// Sample count of the color attachments when it differs from the rasterization count
	// (PipelineStaticParameters::samples); 0 when every attachment matches rasterization.
	uint32_t                                             color_samples  = 0;
	uint32_t                                             depth_samples  = 0;
	// The bound layer count when a vertex-side stage exports the render target array index
	// (ShaderRecompiler::IR::LayerLimitSpecId), else 0.
	uint32_t                                             layer_limit    = 0;

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
	uint32_t         spirv_words = 0;
	uint64_t         hash        = 0;
	// Round 95: the code it was translated from is an expansion that inlined called functions.
	bool inlined_calls = false;
	// A pixel shader that reads its sample index, as one that runs once per pixel with the index
	// reading 0 (SpirvWithoutSampleId); null otherwise. For NV mixed-sample draws.
	vk::ShaderModule pixel_rate_module = nullptr;
	// A hull shader that samples at an explicit LOD, as one that turns its coordinates into
	// normalized ones at that LOD (SpirvNormalizedExplicitLod), with an id of its own; null
	// otherwise. A draw whose samplers there all have unnormalized coordinates runs it (with them
	// made normalized), and sets normalized_samplers on its copy of the program.
	vk::ShaderModule normalized_module   = nullptr;
	uint64_t         normalized_id       = 0;
	bool             normalized_samplers = false;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The SPIR-V of a pixel shader that reads the SampleId built-in, with that input gone and its
// reads giving 0; empty when the shader has no SampleId input or uses it in a way this does not
// rewrite. A SampleId input makes the fragment shader run once per sample, and NV mixed samples
// (fewer color samples than raster samples, PS5 EQAA) forbids per-sample shading unless coverage
// reduction truncates (VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-01415).
[[nodiscard]] std::vector<uint32_t> SpirvWithoutSampleId(std::span<const uint32_t> spirv);
// KYTY_EQAA_PER_SAMPLE=1 (for testing): NV mixed-sample draws keep pixel shaders that read their
// sample index as they are, per-sample shading included, which NVIDIA does not allow there.
[[nodiscard]] bool EqaaPerSampleShading();

// The SPIR-V of a stage before the rasterizer with every float add, subtract, multiply, divide
// and dot product decorated NoContraction, and its Position output Invariant; empty when there
// is nothing to add. The guest GPU rounds a multiply before the add that follows it unless the
// shader asks for a fused one, and a driver free to fuse them may do so in one shader and not in
// another computing the same position: a depth prepass and the pass that tests EQUAL against it
// then disagree in the last bits.
[[nodiscard]] std::vector<uint32_t> SpirvExactFloat(std::span<const uint32_t> spirv);
// The SPIR-V of a stage whose explicit-LOD samples (OpImageSampleExplicitLod with a Lod and
// nothing but offsets) take their coordinates divided by the size of the mip the LOD names, the
// nearest within the image (an array layer as it is); empty when it has none it can rewrite. Run
// with a normalized sampler, it samples as the guest GPU does with FORCE_UNNORMALIZED: at
// coordinates in that mip's texels. Vulkan allows unnormalized coordinates only on a view of one
// mip at LOD 0, and GT7's hull shaders cull hidden patches against a depth pyramid sampled this
// way at a level each picks from the patch's size on screen.
[[nodiscard]] std::vector<uint32_t> SpirvNormalizedExplicitLod(std::span<const uint32_t> spirv);
// KYTY_UNNORMALIZED_FIX (default on; =0, or KYTY_BISECT's unorm:old, off): hull shaders get the
// SpirvNormalizedExplicitLod copy, used by draws whose samplers there are all unnormalized.
[[nodiscard]] bool UnnormalizedSamplingFix();
// Whether every sampler a stage samples through has unnormalized coordinates (this draw's).
[[nodiscard]] bool StageSamplersAllUnnormalized(const ShaderStageRuntime& stage);
// KYTY_EXACT_FLOAT=1 (round 75, a test): every vertex, geometry and tessellation shader's module is
// made from SpirvExactFloat's copy. The shader cache is unchanged; the driver compiles again.
// KYTY_EXACT_FLOAT=<hash>[,<hash>...] (round 102): those shaders alone.
[[nodiscard]] bool ExactFloatPositions();
[[nodiscard]] bool ExactFloatShader(uint64_t shader_hash);

// Everything vkCreateGraphicsPipelines reads for one graphics pipeline, owned in one place so the
// driver compile can run on a worker thread (the structures point into each other: it stays
// where it was allocated).
struct GraphicsPipelineBuild {
	GraphicsPipelineBuild() = default;
	~GraphicsPipelineBuild();
	KYTY_CLASS_NO_COPY(GraphicsPipelineBuild);

	vk::Device                                                            device;
	vk::ShaderModule                                                      tess_control = nullptr;
	vk::ShaderModule                                                      tess_eval    = nullptr;
	std::array<vk::PipelineShaderStageCreateInfo, 4>                      stages {};
	std::array<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo, 4>  subgroup_sizes {};
	std::array<vk::VertexInputAttributeDescription, ShaderVertexInputInfo::RES_MAX> input_attr {};
	std::array<vk::VertexInputBindingDescription, ShaderVertexInputInfo::RES_MAX>   input_desc {};
	vk::PipelineVertexInputStateCreateInfo                                vertex_input_info {};
	vk::PipelineInputAssemblyStateCreateInfo                              input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT                     depth_clip_control {};
	vk::PipelineViewportStateCreateInfo                                   viewport_state {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT                  clip_ext {};
	vk::PipelineRasterizationStateCreateInfo                              rasterizer {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT            provoking_vertex {};
	vk::PipelineMultisampleStateCreateInfo                                multisampling {};
	vk::PipelineCoverageReductionStateCreateInfoNV                        coverage_reduction {};
	std::array<vk::PipelineColorBlendAttachmentState, RENDER_COLOR_ATTACHMENTS_MAX> color_blend {};
	std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX>                  color_write_enable {};
	vk::PipelineColorWriteCreateInfoEXT                                   color_write {};
	vk::PipelineColorBlendStateCreateInfo                                 color_blending {};
	vk::PipelineDepthStencilStateCreateInfo                               depth_stencil {};
	std::vector<vk::DynamicState>                                         dynamic_states;
	vk::PipelineDynamicStateCreateInfo                                    dynamic_state {};
	vk::PipelineTessellationStateCreateInfo                               tessellation {};
	PipelineRenderingState                                                rendering;
	vk::PipelineRenderingCreateInfo                                       rendering_info {};
	std::array<vk::SampleCountFlagBits, RENDER_COLOR_ATTACHMENTS_MAX>     color_attachment_samples {};
	vk::AttachmentSampleCountInfoAMD                                      attachment_sample_count {};
	vk::GraphicsPipelineCreateInfo                                        pipeline_info {};
	vk::SpecializationMapEntry                                            layer_limit_entry {};
	uint32_t                                                              layer_limit_value = 0;
	vk::SpecializationInfo                                                layer_limit_spec {};
	uint64_t                                                              vs_hash = 0;
	uint64_t                                                              ps_hash = 0;
};

// A graphics pipeline a worker thread compiles (see PipelineCache::TryGetGraphicsPipeline).
struct PendingGraphicsPipeline;
class PipelineCompiler;

// The owning renderer serializes access, including saves while the GPU is running.
// Whether every color target the draw writes is at most KYTY_SMALL_TARGET_SYNC pixels wide and
// high (default 1024; 0 = never): such draws are never skipped while their shaders translate or
// their pipeline compiles (pipelineCache.cpp).
bool SmallColorTargetDraw(const HW::Context& context);

// A reflection cube's color target (renderDraw.cpp learns them; round 87): once the game has
// drawn it in two frames, the draws into it may be skipped while their shaders translate or their
// pipelines compile. GPU thread only.
void NoteReflectionCubeTarget(uint64_t address);

// Why a draw's new pipeline may not be compiled in the background (TryGetGraphicsPipeline).
enum class PipelineDefer : uint8_t {
	Allowed,      // the draw may be skipped while a worker compiles it
	WritesMemory, // the draw writes buffers or storage images, which others may read
	SmallTarget,  // a small target the game does not draw every frame: glyphs, cached layers
	Never,        // GetGraphicsPipeline
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	void Save();
	// KYTY_PRECOMPILE=1 (pipelineSeeds.h): every guest seed's program into the shader disk cache
	// and every seed pipeline into the driver cache, on all processors.
	void PrecompileSeeds();
	// Round seedcs: the "Pipelines this run" line (at Save).
	void LogPipelinesMade();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
		// Samples bindless images: descriptor set 1 is the bindless table.
		bool                    uses_bindless         = false;
		// KYTY_FASTDRAW_IMAGESET (IR::FastDrawImageSet): the layout of descriptor set 2, the
		// stages' own images and samplers (null without the switch, or with no image or sampler);
		// written from DescriptorHeap sets by CommitBindings.
		vk::DescriptorSetLayout image_set_layout      = nullptr;
		// Set while a worker compiles `pipeline`; draws skip it until the compile finishes.
		std::shared_ptr<PendingGraphicsPipeline> pending;
		// Round 105: the driver could not build this compute pipeline (`pipeline` is null); its
		// dispatches are skipped (TryGetComputePipeline).
		bool build_failed = false;
		// Round 117: built with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT (KYTY_PARTITION_QUICK
		// for the partitions of the ray tracing material passes), looked up so as well.
		bool disable_optimization = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;
		// The game draws into this draw's target every frame (a reflection cube drawn in two
		// frames or more, or a pass tested against a depth target into a target drawn in three
		// of six frames in a row, and in use since): skipping the draw while its shaders translate
		// or its pipeline compiles loses one frame of it, not a picture drawn once (rounds 87 and
		// 88, RedrawnTargetDraw in pipelineCache.cpp).
		bool redrawn_target = false;
		// Round 159: set for a pipeline build only: the fragment stage's module with the draw's AA
		// masks applied (the pixel shader's copy, or a coverage-only shader for a draw without
		// one), or null.
		vk::ShaderModule coverage_module = nullptr;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	// Draw prep (KYTY_DRAW_PREP, drawPrep.h): the SRT refreshes GetGraphicsPrograms or
	// GetComputeProgram would run for a draw's or dispatch's registers, in two steps so that the
	// walks can run on several threads (round 73): the stage inputs from the registers (the
	// scanner thread, which follows them), then the walks (any of its workers), into `out`.
	struct DrawPrepInputs {
		bool                   compute = false;
		bool                   pixel   = false;
		ShaderParams           pixel_params;
		ShaderPixelInputInfo   pixel_info;
		ShaderParams           vertex_params;
		ShaderVertexInputInfo  vertex_info;
		ShaderParams           compute_params;
		ShaderComputeInputInfo compute_info;
		// A tessellated draw (round 78): its LS, HS and ES stages, in that order (vertex_params
		// and vertex_info are not used then); empty otherwise.
		std::vector<ShaderParams>          tess_params;
		std::vector<ShaderVertexInputInfo> tess_info;
	};
	bool DrawPrepDescribeGraphics(const HW::VertexShaderInfo& vertex_regs,
	                              const HW::PixelShaderInfo&  pixel_regs,
	                              const HW::ShaderRegisters& sh, const HW::Context& context,
	                              const HW::UserConfig&                               user_config,
	                              std::span<const Prospero::ColorComponentMapping, 8> mapping,
	                              bool pixel_active, DrawPrepInputs& inputs);
	bool DrawPrepDescribeCompute(const HW::ComputeShaderInfo& regs, const HW::ShaderRegisters& sh,
	                             const ShaderComputeInputInfo& input_info, DrawPrepInputs& inputs);
	bool DrawPrepWalk(const DrawPrepInputs& inputs, DrawPrep::PacketResult& out);
	// Round 99: `partition`, when given, asks for the program of the partition of
	// partition->target (ShaderCallPartition: 0 for the first one) when the shader's call is
	// partitioned, and receives every partition's target (empty when it is not partitioned); the
	// dispatch then runs once per partition. A partition's program may come back empty while it
	// translates or waits for its turn: that partition is skipped meanwhile.
	using ComputePartition = ShaderRecompiler::Decoder::ShaderCallPartition;
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info,
	                                ComputePartition*            partition = nullptr);
	// Whether the shader at this address is on the skip list (--skip-shaders): its dispatches can
	// be dropped before anything is prepared for them.
	[[nodiscard]] static bool IsSkipListed(uint64_t shader_addr);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	// As GetGraphicsPipeline; with PipelineDefer::Allowed, a pipeline the driver does not have
	// is compiled on a worker thread, and nullptr says to skip this draw.
	Pipeline* TryGetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                                 const RenderDepthInfo&                 depth,
	                                 std::span<const ShaderVertexInputInfo> vertex_info,
	                                 CommandBuffer&                         command,
	                                 const ShaderPixelInputInfo*            ps_input_info,
	                                 vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                 const GraphicsPrograms& programs, PipelineDefer defer);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);
	// Round 94: as GetComputePipeline, but a pipeline of KYTY_COMPUTE_BACKGROUND_WORDS SPIR-V
	// words or more (default 1,000,000; 0: none; round 95: a tenth of that for an expansion that
	// inlined called functions) that no driver cache holds is compiled on a thread of its own, and
	// nullptr comes back until it is done: the dispatch is skipped. The driver took 110 s for one
	// of 1.5 million words (a ray tracing material pass with its looping material functions
	// inlined), with the emulated GPU stopped all along.
	// Round 98: `expansion`, the program's expanded code (GetComputeProgram), goes with a
	// background compile, so that the finished pipeline becomes its shader's ready expansion even
	// when a newer one is asked for by then. At most KYTY_COMPUTE_BACKGROUND_THREADS compile at
	// once (default 3), one per shader.
	Pipeline* TryGetComputePipeline(const ShaderComputeInputInfo& input_info,
	                                const ShaderProgram&          compute_program,
	                                std::span<const uint32_t>     expansion = {});

private:
	struct BackgroundCompute;
	struct WaitingPartition;
	// Waits up to `budget` for the compute pipelines still compiling in the background (Save:
	// they compile into the driver cache); returns how many still compile.
	size_t FinishBackgroundCompute(std::chrono::milliseconds budget);
	// Round 117: starts the background compile of a program no driver cache held (its layouts
	// made by the lookup), for TryGetComputePipeline and StartWaitingPartitions.
	void StartBackgroundCompute(uint64_t id, std::unique_ptr<Pipeline> cached,
	                            const ShaderComputeInputInfo&                   input_info,
	                            const ShaderRecompiler::IR::CompiledShaderInfo& program,
	                            vk::ShaderModule module, uint32_t words, bool partition,
	                            std::span<const uint32_t> expansion, uint64_t skipped, bool ahead);
	// Round 117: partitions waiting for a compile slot take the slots that are free, the ones
	// their dispatches asked for most first, without waiting for their next dispatch: rt116a
	// ended with 63 of 87 waiting and 6 compiling.
	void StartWaitingPartitions();
	// Round 117: while compute pipelines compile in the background or wait to, a line every 10 s
	// on the console and in the log (how many compile, wait and are done, and the memory left),
	// and one when they are all done.
	void ReportBackgroundProgress();
	// Round 98: takes in the background compiles that are done (their pipelines become usable, and
	// an expansion the newest ready one of its shader), and the one still compiling for a shader.
	void                     ReapBackgroundCompute();
	const BackgroundCompute* CompilingCompute(uint64_t shader_hash) const;
	// Round 99: the expanded code of the shader's pipeline compiling in the background, or else of
	// one waiting for a compile slot; null when neither.
	const std::vector<uint32_t>* PendingExpansion(uint64_t shader_hash) const;
	// Save, done with the driver cache: destroyed, or kept while background compiles use it.
	void ReleaseDriverCache(size_t still_compiling);
	// Round 134: the write of the driver cache to its file (Save's, and the saves during play).
	enum class CacheWrite : uint8_t { Saved, Unchanged, TooLarge, Failed };
	CacheWrite WriteDriverCacheFile(bool during_play, size_t& bytes);
	// GPU thread, for each new pipeline taken from a background compile: a save during play when
	// one is due (KYTY_PIPELINE_SAVE_MINUTES).
	void MaybeSaveDuringPlay();
	struct ProgramCache;

	// GetGraphicsPrograms' stage inputs from the registers (shared with DrawPrepDescribeGraphics;
	// `scan`: the scanner's, whose tessellation stages are described without ever ending the
	// emulator, PrepareTessellationProgramsForScan).
	bool DescribeGraphicsStages(
	    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	    const HW::ShaderRegisters& sh, const HW::Context& context,
	    const HW::UserConfig&                               user_config,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	    ShaderPixelInputInfo& pixel_info, std::array<ShaderParams, 3>& vertex_params,
	    ShaderParams& pixel_params, bool scan = false);

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
	void RecordPipelineSeed(const GraphicsPipelineKey& key, const GraphicsPrograms& programs,
	                        uint32_t vertex_stages, bool ps_active);
	static uint64_t SeedFormat();

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	// Round 159: the fragment shaders' copies that apply PA_SC_AA_MASK (destroyed first).
	std::unique_ptr<CoverageVariants> m_coverage;
	// The driver cache new pipelines go to, saved to m_driver_cache_path; round 92: and the older
	// cache files that are full (read only: a pipeline one of them holds is created from it).
	vk::PipelineCache              m_driver_cache = nullptr;
	std::filesystem::path          m_driver_cache_path;
	uint64_t                       m_driver_cache_loaded = 0; // bytes of it loaded (0: none)
	std::vector<vk::PipelineCache> m_driver_older;
	// Round 96: driver caches Save let go of while background compiles still used them.
	std::vector<vk::PipelineCache> m_driver_cache_parked;
	// Unoptimized pipelines replaced by their optimized ones (FinishPending), destroyed at the end.
	std::vector<vk::Pipeline> m_quick_retired;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// Round 97: by shader hash, its last expansion whose compute pipeline is ready (the program id
	// and the expanded code), run while a newer one compiles (GetComputeProgram); and the newer
	// programs whose wait was logged.
	// Round 98: `serial` orders them (a finished background compile replaces an older one only).
	struct ReadyExpansion {
		uint64_t              id     = 0;
		uint64_t              serial = 0;
		std::vector<uint32_t> code;
	};
	std::unordered_map<uint64_t, ReadyExpansion> m_ready_expansions;
	std::unordered_set<uint64_t>                 m_last_expansion_logged;
	uint64_t                                     m_expansion_serial = 0;
	// Round 94: compute pipelines compiling on threads of their own, by program id. Round 98: the
	// programs no driver cache held that wait for a compile slot (with the layouts their lookup
	// made, so they are not looked up again), and those whose wait was logged.
	std::unordered_map<uint64_t, std::unique_ptr<BackgroundCompute>> m_compute_background;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>>          m_compute_waiting;
	std::unordered_set<uint64_t>                                     m_compute_wait_logged;
	// Round 99: the waiting programs' shaders and expanded code (PendingExpansion).
	struct WaitingExpansion {
		uint64_t              hash = 0;
		std::vector<uint32_t> code;
	};
	std::unordered_map<uint64_t, WaitingExpansion> m_compute_waiting_expansions;
	// Round 117: what a waiting partition's compile needs (StartWaitingPartitions), by program id.
	std::unordered_map<uint64_t, std::unique_ptr<WaitingPartition>> m_partition_waiting;
	uint64_t                                                        m_partition_wait_serial = 0;
	// Round 117: ReportBackgroundProgress's counts and times.
	uint64_t                              m_background_started  = 0;
	uint64_t                              m_background_finished = 0;
	uint64_t                              m_backlog_started_at  = 0;
	uint64_t                              m_backlog_finished_at = 0;
	bool                                  m_backlog             = false;
	bool                                  m_backlog_reported    = false;
	std::chrono::steady_clock::time_point m_backlog_begin {};
	std::chrono::steady_clock::time_point m_backlog_last_report {};
	bool                                  m_saved = false;
	// Round 134: the saves of the driver cache during play (MaybeSaveDuringPlay), each on a thread
	// of its own; Save waits for one still writing.
	std::mutex                            m_save_mutex;
	std::thread                           m_save_thread;
	std::atomic<bool>                     m_save_running {false};
	std::chrono::steady_clock::time_point m_last_save {};
	uint64_t                              m_new_since_save = 0;
	// Round 99: the shaders whose call is partitioned (by hash), their partitions' programs (by
	// id: their pipelines compile alongside each other), and when the next new partition's SPIR-V
	// may be made (GetComputeProgram).
	std::unordered_set<uint64_t>                            m_partitioned_shaders;
	std::unordered_set<uint64_t>                            m_partition_programs;
	std::chrono::steady_clock::time_point                   m_partition_next_build {};
	std::unique_ptr<PipelineCompiler>                       m_compiler;
	uint64_t                                                m_deferred_draws = 0;
	// Round 87: whether a new pipeline is first looked up in the driver's caches alone
	// (pipelineCreationCacheControl; KYTY_PIPELINE_PROBE=0 turns it off), and what is left of
	// this frame's time for waits on background compiles (KYTY_PIPELINE_FRAME_WAIT_MS).
	bool                                  m_probe      = false;
	uint64_t                              m_wait_frame = UINT64_MAX;
	std::chrono::microseconds             m_wait_left {0};
	std::chrono::steady_clock::time_point m_wait_renewed {};

	void InitializeDriverCache();
	// Takes a finished background compile into the pipeline; false while it still runs.
	bool FinishPending(Pipeline& pipeline);
	// Waits for a background compile while this frame's budget lasts.
	void WaitForPipeline(const PendingGraphicsPipeline& job);
};

// KYTY_DYNAMIC_RASTER_STATE (default on): cull mode and front face (and, except on macOS, the
// depth-bounds test enable and bounds) are dynamic state of every renderer graphics pipeline,
// recorded per draw by SetGraphicsDynamicParams, and leave the pipeline key; key fields that
// cannot change the pipeline are zeroed. =0 bakes them into each pipeline as before.
[[nodiscard]] bool PipelineDynamicRasterStateEnabled();

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Creates the pipeline's layouts and fills the create info; CreateGraphicsPipeline compiles it.
std::unique_ptr<GraphicsPipelineBuild>
PrepareGraphicsPipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                        const PipelineRenderingState&          rendering,
                        const PipelineVertexInputState&        vertex_input,
                        std::span<const ShaderVertexInputInfo> vertex_info,
                        const ShaderPixelInputInfo*            ps_input_info,
                        const PipelineCache::GraphicsPrograms& programs,
                        const PipelineStaticParameters&        static_params);
// False for an embedded-fetch attribute whose components the shader only replaces by constants:
// it gets no native vertex attribute, and its buffer is not bound for it.
[[nodiscard]] bool VertexAttributeFetched(const ShaderVertexInputInfo& info, int attribute);
// Any thread. With cache_only (pipelineCreationCacheControl), the driver only looks the pipeline
// up in its caches: vk::Result::ePipelineCompileRequired when it would have to compile it.
vk::Result CreateGraphicsPipeline(const GraphicsPipelineBuild& build,
                                  vk::PipelineCache driver_cache, vk::Pipeline* pipeline,
                                  bool cache_only = false, bool no_optimization = false);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
// How a compute pipeline is made (round 94): Full makes its layouts, looks it up in `older` (the
// full driver cache files, round 92) and compiles it into driver_cache when they do not hold it;
// LookUp makes its layouts and looks it up in `older` and in driver_cache, compiling nothing
// (false when none holds it, the layouts kept); Compile compiles it into driver_cache, on any
// thread, after a LookUp made its layouts.
enum class ComputeBuild : uint8_t { Full, LookUp, Compile };
// Round seedcs: pipelines made this run (the log's "Pipelines this run" line at the end), from the
// driver's caches (a lookup that found it, or a create of under CachedMs) or compiled. Counted
// when a pipeline is made, never per draw or dispatch.
struct PipelineMadeCounts {
	static constexpr int64_t CachedMs = 10;
	std::atomic<uint64_t>    graphics_cached {0};
	std::atomic<uint64_t>    graphics_compiled {0};
	std::atomic<uint64_t>    compute_cached {0};
	std::atomic<uint64_t>    compute_compiled {0};
};
PipelineMadeCounts& MadeCounts();
bool CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache,
                            std::span<const vk::PipelineCache> older = {},
                            ComputeBuild                       build = ComputeBuild::Full);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
