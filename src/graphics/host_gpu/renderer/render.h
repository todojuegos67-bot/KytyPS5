#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW
namespace Spec {
struct State;
} // namespace Spec

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
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
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
	// Arguments the GPU reads (gpu_args_count DrawIndexedIndirectArgs at gpu_args, gpu_args_stride
	// apart), not read on the CPU: index_count and index_addr then cover the whole index buffer.
	uint64_t gpu_args        = 0;
	uint32_t gpu_args_count  = 0;
	uint32_t gpu_args_stride = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// Arguments the GPU reads (DrawIndirectArgs), as in DrawIndexArgs.
	uint64_t gpu_args        = 0;
	uint32_t gpu_args_count  = 0;
	uint32_t gpu_args_stride = 0;
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

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	// Raw internal graphics operations must invalidate the values recorded by
	// the ordinary draw path. Compute and transfer commands do not alter them.
	void InvalidateGraphicsState() const noexcept;
	// Advances with every InvalidateGraphicsState: a path that binds graphics
	// state itself can tell whether anything else bound state since.
	[[nodiscard]] uint64_t GraphicsGeneration() const noexcept { return m_graphics_generation; }

	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Only the dispatch path can bypass a pending dependency, after preparing
	// resources. Every other host command drains it through Handle().
	[[nodiscard]] vk::CommandBuffer ChainHandle() const;
	[[nodiscard]] bool ComputeChainPending() const noexcept { return m_compute_access_pending; }
	void ContinueComputeChain() const;
	[[nodiscard]] vk::CommandBuffer HandleForFullBarrier() const;
	// Guest buffer copies in a row (Buffer::CopyInRun): one barrier before the run and one after it, recorded before
	// the next other command (Handle, ChainHandle; a full barrier covers it). A copy that reads what an earlier copy of
	// the run writes, or writes what one reads or writes, starts a new run.
	[[nodiscard]] vk::CommandBuffer CopyRunHandle(vk::Buffer source, uint64_t source_offset, vk::Buffer destination,
	                                              uint64_t destination_offset, uint64_t size) const;
	// A run is open: everything recorded since its barrier is its copies (any other command flushes it first).
	[[nodiscard]] bool InCopyRun() const noexcept { return !m_copy_run.empty(); }
	// A copy of what the open run wrote (a copy-feedback snapshot, BufferCache::ScheduleCopyFeedback): recorded when the
	// run ends, after its copies, behind one transfer barrier, then made available to the host. A copy of the run that
	// writes its source range starts a new run, which records it first.
	void CopyAfterRun(vk::Buffer source, uint64_t source_offset, vk::Buffer destination, uint64_t destination_offset,
	                  uint64_t size) const;
	// Local diagnostic (GPU marks): the handle without draining a pending dependency.
	[[nodiscard]] vk::CommandBuffer RawHandle() const noexcept { return m_buffer; }
	// The recorded work count after this buffer's last global barrier (CommandProcessor::EmitGlobalBarrier).
	[[nodiscard]] uint64_t& BarrierWork() const noexcept { return m_barrier_work; }

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
	void FlushCopyRun() const;
	void FlushRunTail() const;

	struct CopyRunRange {
		VkBuffer buffer;
		uint64_t begin, end;
		bool     written;
	};
	struct RunTailCopy {
		VkBuffer source, destination;
		uint64_t source_offset, destination_offset, size;
	};

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	mutable bool        m_compute_access_pending = false;
	mutable std::vector<CopyRunRange> m_copy_run; // a run's ranges (not empty: its after barrier is owed)
	mutable std::vector<RunTailCopy>  m_run_tail; // CopyAfterRun's copies, owed with the run's end
	mutable uint64_t    m_graphics_generation    = 0;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_timestamp_slot  = UINT32_MAX; // live trace GPU timestamps
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable uint64_t    m_barrier_work = UINT64_MAX;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                    uint64_t indirect_args = 0);
	// A state pass's direct dispatch (Spec::t_record): the copy the linear copy shader makes (journaled, as a speculation's
	// CopyBuffer journals it; one on GPU-written memory: its destination written) or the surface a compute clear clears
	// (predicted), as the catalog knows them.
	void PredictDispatch(const HW::Shader& sh_ctx, uint32_t thread_group_x, uint32_t thread_group_y,
	                     uint32_t thread_group_z, uint32_t mode, Spec::State& record);

	[[nodiscard]] PreparedBindings PrepareBindings(const ShaderStageRuntime& runtime);
	void PrepareBindingsInto(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(PreparedBindings& bindings);
	void PrepareBdaBindings(const PreparedBindings& first, const PreparedBindings* second = nullptr);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	[[nodiscard]] bool             AcquireImages(PreparedBindings& bindings);
	[[nodiscard]] bool             AcquirableImage(const TextureBinding& binding);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings, bool compute_chain = false);

private:
	[[nodiscard]] bool ReadOnlyDrawBufferRangeSafe(GuestRange range, const DrawRenderState& state,
	                                               bool sampled_overlaps);
	struct TextureResolveCache;
	std::shared_ptr<TextureResolveCache> m_texture_resolve_cache;

	bool TryDrawIndexRun(uint64_t submit_id, CommandBuffer& buffer, std::span<const DrawIndexArgs> draws,
	                     std::span<const uint64_t> argument_addresses);
	// False only for GPU-read arguments this draw cannot use (a mesh draw): the caller reads
	// them on the CPU and draws again.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args); // as DrawIndex

	struct GraphicsBindings {
		PreparedBindings                vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding ResolveTextureUncached(
	    const ShaderRecompiler::IR::ImageResource& resource,
	    const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] GraphicsBindings PrepareGraphicsBindings(const ShaderStageRuntime& vertex,
	                                                       const ShaderStageRuntime& pixel,
	                                                       bool                      pixel_active);
	void PrepareGraphicsBindingsInto(const ShaderStageRuntime& vertex,
	                                  const ShaderStageRuntime& pixel, bool pixel_active,
	                                  GraphicsBindings& bindings);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          bool log_setup_phases, DrawRenderState& state);
	bool ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, bool log_pipeline_phase,
	                         bool set_bind_debug, bool set_auto_debug);
	void CommitGraphicsState(CommandBuffer& buffer, const ShaderVertexInputInfo& input,
	                         const RenderColorInfo* colors, uint32_t color_count,
	                         const RenderDepthInfo& depth, vk::Pipeline pipeline,
	                         vk::ImageAspectFlags feedback_aspects);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               const std::optional<PreparedBindings>& pixel = std::nullopt);
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      TrackImageBinding(ImageId id);
	void                      ResetBindings();
	// (`cleared`: the metadata address it cleared.)
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer& buffer, uint32_t group_x,
	                                                     uint32_t group_y, uint32_t group_z,
	                                                     uint32_t mode, uint64_t& cleared);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                              CommandBuffer& command, uint32_t group_x,
	                                              uint32_t group_y, uint32_t group_z, uint32_t mode);

	// XPR draw capture for the offline replay tests (xpr-capture.h).
	void CaptureXprDraw(CommandBuffer& buffer, const DrawRenderState& state, const DrawIndexArgs& args);
	void CaptureXprTargets(const DrawRenderState& state, const GraphicsBindings& bindings);

	// Native XPR draws (src/local/native-xpr.inc).
	struct NativeXprDrawState;
	struct NativeXprRecord;
	struct NativeXprCache;
	struct NativeXprDrawArgs {
		uint64_t      index_address = 0, index_bytes = 0;
		vk::IndexType index_type    = vk::IndexType::eUint16;
		uint64_t      args_address  = 0; // GPU-side DrawIndexedIndirectCommand
		bool          keep_clears   = false;
	};
	std::shared_ptr<NativeXprCache> m_native_xpr;
	bool                            m_native_xpr_store = false; // next prepared draw stores
	struct NativeXprVerifyRequest {
		const NativeXprRecord*    record = nullptr;
		const NativeXprDrawState* state  = nullptr;
	} m_native_xpr_verify; // mode 2: the next prepared draw is compared with this record
	void NativeXprVerify(CommandBuffer& buffer, const DrawRenderState& state, const RenderState& rendering,
	                     std::span<PreparedBindings* const> stages, const DrawEmitInfo& emit);
	NativeXprCache& NativeXprState();
	void NativeXprRequestStore();
	void NativeXprKey(CommandBuffer& buffer, std::vector<uint32_t>& key) const;
	[[nodiscard]] uint64_t NativeXprStateKey(CommandBuffer& buffer) const;
	[[nodiscard]] vk::DescriptorSet NativeXprAllocateSet(vk::DescriptorSetLayout layout,
	                                                     vk::DescriptorPool& pool);
	void NativeXprRetire(std::unique_ptr<NativeXprRecord> record, bool keep_for_learning);
	void NativeXprRetireSet(vk::DescriptorPool pool, vk::DescriptorSet set);
	[[nodiscard]] const char* NativeXprBindResources(NativeXprRecord& record,
	                                                 std::span<PreparedBindings* const> stages,
	                                                 vk::DescriptorSetLayout layout, bool runtime);
	[[nodiscard]] bool NativeXprRebind(NativeXprRecord& record,
	                                   std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>& fresh);
	[[nodiscard]] bool NativeXprTexelRangesClear(const NativeXprRecord& record);
	[[nodiscard]] bool NativeXprSwitchVariant(NativeXprRecord& record, uint64_t signature);
	void               NativeXprDropVariants(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprRebindSlots(NativeXprRecord& record,
	                                        std::span<const std::pair<uint32_t, uint32_t>> images,
	                                        std::span<const std::pair<uint32_t, uint32_t>> buffers);
	void NativeXprCollect();
	// A draw state's dynamic state, only what differs from what the emissions recorded in the command buffer.
	static void NativeXprEmitDynamic(NativeXprCache& cache, vk::CommandBuffer vk_buffer, const NativeXprDrawState& draw_state,
	                                 bool same_command);
	void NativeXprLearn(const NativeXprRecord& stale, NativeXprRecord& fresh);
	void NativeXprStore(CommandBuffer& buffer, const DrawRenderState& state,
	                    vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                    const RenderState& rendering, std::span<PreparedBindings* const> stages,
	                    const DrawEmitInfo& emit);
	[[nodiscard]] bool NativeXprValidate(NativeXprRecord& record, uint64_t frame);
	[[nodiscard]] bool NativeXprReevaluate(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprValidateResources(NativeXprRecord& record, uint64_t frame, bool mapping_moved);
	void NativeXprRetrace(NativeXprRecord& record);
	[[nodiscard]] bool NativeXprGather(NativeXprRecord& record);
	enum class NativeXprPatch { Done, Evaluate, Invalid };
	[[nodiscard]] NativeXprPatch NativeXprPatchDescriptors(NativeXprRecord& record);
	struct NativeXprDirect; // one direct draw (public NativeXprDirectDraw) with its record's offsets
	[[nodiscard]] bool NativeXprEmit(CommandBuffer& buffer, NativeXprRecord& record,
	                                 const NativeXprDrawState& draw_state,
	                                 const PipelineCache::Pipeline& pipeline,
	                                 std::span<const uint64_t> commands, uint64_t index_base,
	                                 uint64_t index_bytes, vk::IndexType index_type, bool keep_clears,
	                                 const NativeXprDirect* direct = nullptr);
	[[nodiscard]] bool NativeXprDraw(CommandBuffer& buffer, const NativeXprDrawArgs& args);

public:
	// Runtime entry points for the command processor (kyty_local_native_xpr_mode).
	// A direct indexed draw (DRAW_INDEX_2 / DRAW_INDEX_OFFSET_2): its indices, count and instances; or a direct draw
	// of vertices (DRAW_INDEX_AUTO, not `indexed`: the count is of vertices; table draws only).
	struct NativeXprDirectDraw {
		uint64_t index_address  = 0;
		uint32_t index_count    = 0;
		uint32_t instance_count = 1;
		bool     indexed        = true;
	};
	// `direct`: that draw instead of the DRAW_INDEX_INDIRECT run `commands`.
	[[nodiscard]] bool NativeXprTry(CommandBuffer& buffer, bool clean, std::span<const uint64_t> commands,
	                                uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                                const NativeXprDirectDraw* direct = nullptr);
	// After a refused NativeXprTry: the normal path stores. Returns the log the
	// caller's ShaderReadObserver fills with the SRT evaluation's reads.
	[[nodiscard]] std::vector<std::pair<uint64_t, uint64_t>>* NativeXprReadLog();
	void NativeXprEndPacket();
	// Graphics registers may have changed without the observer seeing it.
	void NativeXprForgetState();
	// A clean draw (only user data and index state changed) after a table draw whose bindings still stand:
	// drawn the same way without the native path's lookups (KYTY_TABLE_XPR=2).
	[[nodiscard]] bool TableContinue(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                                 uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	// Table dispatches (KYTY_TABLE_DISPATCH): a dispatch drawn the table way, and the normal path's dispatches.
	[[nodiscard]] bool TableDispatch(CommandBuffer& buffer, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
	                                 uint64_t indirect_args, bool thread_dimensions);
	// Why the last TableDispatch that declined did (the hole's reason when a speculation's dispatch has no other way).
	const char* m_dispatch_declined = nullptr;
	// (`special`: the special path that took the dispatch, as a speculation's hole reason; null: none.)
	void TableDispatchSeen(const HW::ComputeShaderInfo& cs, const char* special, bool thread_dimensions);
	// What a speculative translation retired with `pending` (descriptor sets; CommandScheduler::PendingTick and up)
	// gets `tick`.
	void ResolvePendingTicks(uint64_t pending, uint64_t tick);
	// A speculation's executor draws with what `owner` (the graphics queue's) learned: its table pairs and dispatches,
	// draw states, linear copy shader and compute clears (RenderExecutor::Catalog); `role` its own proofs' (1..Roles-1:
	// one per speculation thread).
	void ReadCatalogOf(RenderExecutor& owner, uint32_t role) noexcept {
		m_catalog = &owner;
		m_role    = role;
	}
	static constexpr uint32_t Roles = 9;

private:
	// Table draws (src/local/table-xpr.inc).
	struct TableXpr;
	struct TablePair;
	struct TableVariant;
	struct TableImageSet;
	enum class TableResult { Drawn, Native, Store };
	std::shared_ptr<TableXpr> m_table_xpr;
	// The executor whose catalog this one reads (a speculation's: the graphics queue's), null: its own. It changes none
	// of it but its own proofs in it (Role: TablePair::generation, TableImageSet::validated_frame...); what it holds of
	// it across draws is dropped when the owner changed it (TableXpr::epoch).
	RenderExecutor* m_catalog = nullptr;
	uint32_t        m_role    = 0;
	[[nodiscard]] RenderExecutor& Catalog() noexcept { return m_catalog != nullptr ? *m_catalog : *this; }
	// The catalog's owner changes its pairs and dispatches, image sets, draw states and compute clears under its lock
	// exclusively (CatalogWrite); another executor reads them under it shared (CatalogRead, on its own thread).
	mutable std::shared_mutex m_catalog_mutex;
	[[nodiscard]] std::unique_lock<std::shared_mutex> CatalogWrite() const { return std::unique_lock(m_catalog_mutex); }
	[[nodiscard]] std::shared_lock<std::shared_mutex> CatalogRead() { return std::shared_lock(Catalog().m_catalog_mutex); }
	[[nodiscard]] uint32_t        Role() const noexcept { return m_role; }
	// The catalog's table (null: nothing learned yet), and this executor's own table state.
	[[nodiscard]] TableXpr*       CatalogTable() noexcept { return Catalog().m_table_xpr.get(); }
	TableXpr&                     OwnTable();
	void                          DropStaleCatalog();
	// The linear copy shader the normal path last copied with (DemonsSouls::TryLinearCopy), the shader map generation
	// before its program was prepared, and its dispatches since (every 64th takes the normal path again).
	std::atomic<uint64_t> m_linear_copy_shader {0}, m_linear_copy_generation {0};
	uint32_t              m_linear_copy_uses = 0;
	// The metadata surfaces compute dispatches cleared (TryConsumeComputeMetaClear), by their shader and user data: a
	// speculation's hole of such a dispatch clears it (Spec::State::PredictClear). (Not the DCC fills
	// TryConsumeComputeImageClear tracks: the normal path consumes those only where it can clear the image, 10-06.)
	std::unordered_map<uint64_t, uint64_t> m_meta_clears;
	TableResult TableTry(CommandBuffer& buffer, std::span<const uint64_t> commands, uint64_t index_base,
	                     uint64_t index_bytes, vk::IndexType index_type, const NativeXprDirectDraw* direct);
	TableResult TableDraw(CommandBuffer& buffer, TablePair& pair, TableVariant& variant,
	                      const NativeXprDrawState& draw_state, std::span<const uint64_t> commands,
	                      uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
	                      const NativeXprDirectDraw* direct);
	[[nodiscard]] TableImageSet* TableResolveSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                                             const NativeXprDrawState* draw_state, std::span<const uint32_t> words);
	[[nodiscard]] bool           TableValidateSet(TableImageSet& set);
	void                         TableRuns(TablePair& pair);
	void                         TableReset(TablePair& pair);
	[[nodiscard]] bool           TableEvaluate(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	void                         TableGather(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	[[nodiscard]] TableImageSet* TableSet(CommandBuffer& buffer, TablePair& pair, const TableVariant* variant,
	                                      const NativeXprDrawState* draw_state);
	[[nodiscard]] bool           TableBlocks(TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data,
	                                         std::array<vk::DeviceAddress, 2>& blocks, bool* writes);
	void                         TablePrepareBda(const TablePair& pair, std::array<std::span<const uint32_t>, 2> user_data);
	[[nodiscard]] bool           TableTransit(const TableImageSet& set, vk::CommandBuffer vk_buffer,
	                                          const NativeXprDrawState* draw_state);
	void TableStore(CommandBuffer& buffer, const DrawRenderState& state, vk::PrimitiveTopology topology,
	                bool primitive_restart_enable, const RenderState& rendering, bool storable);
	void NativeXprCaptureState(CommandBuffer& buffer, const DrawRenderState& state, const RenderState& rendering,
	                           uint64_t state_key);
	[[nodiscard]] bool NativeXprTargetsCurrent(const NativeXprDrawState& draw_state);
	[[nodiscard]] bool NativeXprEmitTargets(const NativeXprDrawState& draw_state, vk::CommandBuffer vk_buffer);


private:
	RenderContext&                        m_context;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	std::unordered_set<uint64_t> m_unrepresentable_textures;
	std::unordered_set<uint64_t> m_depth_tiled_reports;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
