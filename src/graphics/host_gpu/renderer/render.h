#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/hashMemo.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cmdRecorder.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/fsr3.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/renderer/upscaler/upscaler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <optional>
#include <source_location>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
	// Nonzero: the guest address of the instance count, which a guest shader wrote (NUM_INSTANCES after an
	// indirect multi-draw); instance_count above is then unknown, and the GPU copies the count into the draw.
	uint64_t         instance_count_address     = 0;
};

// Guest writes and read backs that reached the caches. While it stands, what a rebind reads from them hasn't moved.
extern std::atomic<uint64_t> g_caches_touched;

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
	// Nonzero: the guest address of the four argument dwords, which a guest shader wrote; the
	// counts above are then unknown. A mesh draw reads them on the GPU, any other reads them here.
	uint64_t         gpu_arguments              = 0;
	// With indirect records: the same address, for a mesh draw, which the GPU can't read records for. It then goes
	// on as gpu_arguments would, on the state already prepared: a mesh draw's doesn't depend on the records.
	uint64_t         mesh_gpu_arguments         = 0;
	// As DrawIndexArgs::instance_count_address.
	uint64_t         instance_count_address     = 0;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return !m_open; } // m_buffer is null while the recording thread holds it

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	// Calls to EndRendering, a pass open or not: anything that records outside a pass, or changes
	// a target, makes one first. A draw inside a pass makes none.
	[[nodiscard]] uint64_t EndRenderingCalls() const { return m_end_rendering_calls; }
	[[nodiscard]] bool RenderingWith(const RenderState& state) const { return m_rendering && m_render_state == state; }

	// The Vulkan command buffer, to record into directly. With the recording thread (CommandScheduler::Threaded)
	// this first waits until that thread has recorded everything queued: Recorder() is the way that does not wait.
	[[nodiscard]] vk::CommandBuffer Handle(std::source_location where = std::source_location::current()) const;
	// Records at once, or queues for the recording thread.
	[[nodiscard]] Cmd Recorder() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	CommandScheduler&   m_scheduler;
	vk::CommandBuffer   m_buffer          = nullptr; // direct only
	bool                m_open            = false;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	mutable uint64_t    m_end_rendering_calls = 0;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

// A draw's depth target and its first color_count color targets.
struct DrawTargets {
	RenderDepthInfo depth;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	uint32_t        color_count = 0;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// With `create` false it gives an image that exists or none (TextureCache::FindImage). `outcome` is Fixed for a
	// null from the descriptor alone, Absent for a storage candidate's null for want of an image of its backing (it
	// holds while no TextureCache::ImageChangesSince change covers outcome's bytes), Recheck for any other binding (a
	// draw's targets or an image's state can change it). A storage candidate's outcome also gives the bytes a target
	// of the draw must not overlap (TextureCache::HasTargetIn).
	// A binding made from the descriptor alone or from a plain lookup (while the image found is unchanged) is
	// remembered (the texture memo) and given again for the same resource, descriptor and `create`, with the lookup's
	// side effects repeated. Storage candidates always resolve.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool create = true, Common::GroupResult* outcome = nullptr);

private:
	// The resource fields ResolveTexture can read: all but the resource's place in its program (source,
	// descriptor_index, first_use_pc) and a group's candidate list (indirect_resources). One texture read by two
	// programs is one entry, and keeping a key never allocates.
	struct TextureMemoResource {
		TextureMemoResource() = default;
		explicit TextureMemoResource(const ShaderRecompiler::IR::ImageResource& resource);
		ShaderRecompiler::IR::ImageResourceClass resource_class = ShaderRecompiler::IR::ImageResourceClass::None;
		Prospero::TextureNumericClass            numeric_class  = Prospero::TextureNumericClass::Unsupported;
		ShaderRecompiler::Decoder::ImageDimension dimension     = ShaderRecompiler::Decoder::ImageDimension::Unknown;
		ShaderRecompiler::IR::ImageMipMode       mip_mode       = ShaderRecompiler::IR::ImageMipMode::None;
		uint32_t                                 mip_count      = 1;
		Prospero::BufferFormat                   conversion_format = Prospero::BufferFormat::kInvalid;
		uint32_t                                 shader_swizzle = 0;
		uint32_t                                 indirect_root  = 0;
		uint32_t                                 indirect_mapping_offset    = 0;
		uint32_t                                 indirect_search_iterations = 0;
		uint32_t                                 indirect_capacity          = 0;
		bool read = false, written = false, atomic = false, atomic64 = false, unfiltered_access_only = false,
		     sample_only = false, one_layer_safe = false, base_level_only = false, depth_compare = false, cube = false,
		     r128 = false, heap = false, guest_heap = false;
		bool operator==(const TextureMemoResource&) const = default;
	};
	struct TextureMemoKey {
		TextureMemoResource                   resource;
		ShaderRecompiler::IR::DescriptorValue value;
		bool                                  create = true;
	};
	// What a hit gives (`binding`, `outcome`) and checks or repeats: a heap resource's mapping over `mapped_size` bytes,
	// and the plain lookup that found `lookup` with `stamp` for the binding's desc with its view's base layer then.
	struct TextureMemoValue {
		TextureBinding       binding;
		Common::GroupOutcome outcome = Common::GroupOutcome::Recheck;
		ImageId              lookup;
		uint64_t             stamp               = 0;
		uint32_t             metadata_base_layer = 0;
		uint64_t             mapped_size         = 0;
		bool                 cacheable           = false;
		uint64_t             fill                = 0; // which filling of its entry this is; never repeats
	};
	[[nodiscard]] static uint64_t TextureMemoHash(const ShaderRecompiler::IR::ImageResource&   resource,
	                                              const ShaderRecompiler::IR::DescriptorValue& value, bool create);
	// ResolveTexture without the memo; with `made`, says whether and how the binding can be remembered.
	[[nodiscard]] TextureBinding ResolveTextureFully(const ShaderRecompiler::IR::ImageResource&   resource,
	                                                 const ShaderRecompiler::IR::DescriptorValue& value, bool create,
	                                                 Common::GroupResult* outcome, TextureMemoValue* made);

	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// A later record of a packet read on the CPU, whose patched registers didn't change: its own draw on what the record
	// before it left bound. False: nothing was drawn, and it takes DrawIndex or DrawAuto.
	[[nodiscard]] bool DrawLaterMeshRecord(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args, bool indexed,
	                                       uint32_t draw_index_slot, uint32_t draw_index);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool        Upscale(vk::CommandBuffer command, const Pssr::Request& request,
	                                  Dlss::Inputs& inputs);

	// A null feature is one the upscaler refused or failed.
	struct UpscalerContext {
		uint64_t                           context;
		Dlss::Creation                     creation;
		std::unique_ptr<Upscaler::Feature> feature;
	};
	[[nodiscard]] bool UpscaleFsr(vk::CommandBuffer command, const Pssr::Request& request,
	                              const Fsr3::Inputs& inputs);

	// FSR 3.1 history of a PSSR context; a null fsr is one FSR refused or failed.
	struct FsrContext {
		uint64_t        context;
		Fsr3::Creation  creation;
		Fsr3::Context*  fsr;
	};

	// What ExecutePreparedDraw left bound, when a later record may draw on it (DrawLaterMeshRecord).
	struct LaterRecords {
		bool                   valid       = false;
		uint64_t               tick        = 0;
		RenderState            rendering;
		vk::PipelineLayout     layout      = nullptr;
		vk::ShaderStageFlags   stages;
		uint32_t               draw_dwords = 0;
		uint32_t               index_size  = 0;
		bool                   uses_dma    = false;
		const DrawRenderState* state       = nullptr; // TakeDrawRenderState's, untouched until the next full draw
		uint64_t               caches_touched = 0; // before the draw read them
		PreparedBindings*      stage       = nullptr; // the mesh stage
		vk::DescriptorSetLayout set_layout = nullptr;
		bool                   push_descriptors = false;
	};
	static bool UploadDrawIndex(RenderContext& context, PreparedBindings& stage, uint32_t slot, uint32_t draw_index, uint32_t& dword);
	void       CommitDrawIndex(CommandBuffer& buffer, uint32_t dword, uint32_t draw_index);
	uint64_t             m_draw_caches_touched = 0;
	size_t               m_commit_split = 0; // the last commit's writes of set 0
	vk::ShaderStageFlags m_commit_push_stages;

	RenderContext&                        m_context;
	LaterRecords                         m_later_records;
	// GET_LOD_STATS counters since their last reset. With --lod-stats a bound T# with counting
	// enabled counts as sampled at its BASE_LEVEL, clamped when its MIN_LOD lies above that level.
	struct MipStatsCounter {
		uint8_t finest_level = 15; // no samples
		bool    clamped      = false;
		bool    unclamped    = false;
	};

	bool                                  m_lod_stats = Config::LodStatsEnabled();
	GraphicsBindings                     m_graphics_bindings;
	// The current draw's targets; PrepareDrawRenderState resolves them again for each draw (upstream 7ae328b31).
	DrawTargets                           m_targets;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<ImageId>                  m_force_general_images; // this draw's images sampled and stored at once
	std::vector<ImageId>                  m_target_images; // this draw's BindRenderTarget calls, for group walks
	std::vector<std::pair<uint64_t, uint64_t>> m_target_bytes; // PrepareBindings: what they cover
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	// Ranges of m_descriptor_images a commit does not take from it: the tails of group arrays, left unwritten
	// (partially bound; no source), and group arrays, written from their own image infos (PreparedBindings::groups).
	struct DescriptorRun {
		size_t                         begin  = 0;
		size_t                         end    = 0;
		const vk::DescriptorImageInfo* source = nullptr;
		bool operator<(const DescriptorRun& other) const { return begin < other.begin; }
	};
	std::vector<DescriptorRun> m_descriptor_runs;
	// The image groups' kept sets (IR::GroupSetFor), by program: the set, the image epoch and the groups' generations it
	// was written at (nullptr while the last was a per-frame set).
	struct KeptGroupSet {
		vk::DescriptorSet     set   = nullptr;
		uint64_t              epoch = 0;
		std::vector<uint64_t> generations;
	};
	std::unordered_map<const void*, KeptGroupSet> m_kept_group_sets;
	std::vector<uint64_t>                         m_group_generations;
	std::vector<vk::WriteDescriptorSet>           m_group_writes;
	// CommitBindings' scalar-read check: the written images' ranges by start, and the furthest end up to each.
	std::vector<std::pair<uint64_t, uint64_t>> m_written_images;
	std::vector<uint64_t>                      m_written_ends;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	// The render target memo: each color slot's and the depth target's last resolve, by a copy of the registers it read.
	// While FindImage's plain stamp for the image stands, the same registers resolve to the same target (KYTY_TARGET_MEMO).
	struct ColorTargetMemo {
		HW::RenderTarget rt {};
		uint32_t         mask         = 0;
		uint32_t         slice_offset = 0;
		RenderColorInfo  info;
		uint64_t         stamp = 0; // 0 holds nothing
	};
	struct DepthTargetMemo {
		HW::DepthRenderTarget z {};
		HW::RenderControl     rc {};
		HW::DepthControl      dc {};
		HW::StencilControl    sc {};
		HW::StencilMask       sm {};
		uint32_t              clear_bits    = 0;
		uint32_t              stencil_clear = 0;
		RenderDepthInfo       info;
		uint64_t              stamp = 0; // 0 holds nothing
	};
	std::array<ColorTargetMemo, 8> m_color_memo {};
	DepthTargetMemo                m_depth_memo {};
	// Hits, misses and (mode 2) hits a full resolve proved wrong, logged every 5 s.
	uint64_t m_target_hits = 0, m_target_misses = 0, m_target_wrong = 0, m_target_calls = 0, m_target_stat_start = 0;
	[[nodiscard]] static int TargetMemoMode();
	// ResolveTexture's memo: draws resolve the same textures again and again.
	Common::HashMemo<TextureMemoKey, TextureMemoValue, 1024, 4> m_texture_memo;
	using TextureMemoEntry = decltype(m_texture_memo)::Entry;
	// What a slot of a shader last resolved to, by the address of its resource (PrepareBindings). The entry can have been
	// filled again since: `fill` says.
	struct ImageSlotMemo {
		const ShaderRecompiler::IR::ImageResource* resource = nullptr;
		ShaderRecompiler::IR::DescriptorValue      value;
		TextureMemoEntry*                          entry = nullptr;
		uint64_t                                   fill  = 0;
	};
	std::unique_ptr<std::array<ImageSlotMemo, 8192>> m_image_slots = std::make_unique<std::array<ImageSlotMemo, 8192>>();
	uint64_t                                         m_texture_memo_fills = 0;
	// The memo entry ResolveTexture's last answer came from, if any.
	TextureMemoEntry* m_resolved_entry = nullptr;
	// The compute pass that turns guest draw arguments into mesh-task arguments; made on first use.
	[[nodiscard]] vk::Pipeline MeshArgumentsPipeline();
	vk::PipelineLayout         m_mesh_arguments_layout   = nullptr;
	vk::Pipeline               m_mesh_arguments_pipeline = nullptr;
	// The same for many draws: one dispatch before a pass, whose mesh draws add their records to its list
	// until something that could write their arguments is recorded (BatchMeshArguments).
	[[nodiscard]] vk::Pipeline MeshArgumentsBatchPipeline();
	void                       BatchMeshArguments(CommandBuffer& buffer, const struct MeshArgumentsParameters& parameters);
	vk::PipelineLayout         m_mesh_batch_layout   = nullptr;
	vk::Pipeline               m_mesh_batch_pipeline = nullptr;
	struct MeshArgumentsBatch {
		uint8_t* mapped              = nullptr; // in the stream ring: the dispatch's command, then the records
		uint32_t count               = 0;
		uint64_t tick                = 0;
		uint64_t end_rendering_calls = 0;
	} m_mesh_arguments_batch;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	std::vector<UpscalerContext>          m_upscaler_contexts;
	std::vector<FsrContext>               m_fsr_contexts;
	std::array<MipStatsCounter, 256>      m_mip_stats {};

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

// `fill`: instead of the snapshot's (a Pattern fill's value, read as the dispatch runs).
[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size,
                                            const ShaderRecompiler::IR::UniformFill* fill = nullptr);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
