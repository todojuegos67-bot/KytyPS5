#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/indirectDispatch.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <unordered_set>
#include <array>
#include <memory>
#include <optional>
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
namespace TableTargets {
struct Table;
} // namespace TableTargets
namespace TableBindings {
struct Table;
} // namespace TableBindings
struct DrawCallInfo;
struct DiagnosticCheckpoint;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
class CommandRecorder;
struct RenderExecutorTestAccess;
namespace GpuTiming {
struct DrawDetail;
} // namespace GpuTiming

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
	DispatchIndirect,
	Unknown,
	// Transfers (CommandBuffer::MarkTransfer), for device-loss reports.
	ImageUpload,
	ImageDownload,
	ImageCopy,
	ImageResolve,
	ImageCopyThroughBuffer,
	ImageMipCopy,
	ImageClear,
	BufferCopy,
	DepthCopy,
	// The emulator's own GPU work (shaders and copies a guest command does not name), arg0 its
	// InternalWork kind: a device lost between two guest commands names it.
	InternalWork,
};

// CommandBufferDebugOp::InternalWork's arg0.
enum class InternalWork : uint32_t {
	ColorClearCheck = 1, // dst: the metadata's guest address, src: the image's; args: words, slice
	NanScan,             // dst: the image's guest address; args: bytes, layers
	Detile,              // the tiler (guest layout to linear); args: target bytes, dispatches
	Tile,                // the tiler (linear to guest layout); args: target bytes, dispatches
	ColorTransform,      // the tiler's color transform; args: texels, to the host (1) or not
	DepthConvert,        // the tiler's D16 conversion; args: width, height
	FaultScan,           // the page-fault buffer's scan
	BufferDownload,      // GPU-written bytes to guest memory; dst: the buffer's guest address;
	                     // args: bytes, copies
	ImageDump,           // KYTY_DUMP_IMAGES; dst: the image's guest address; args: bytes, tiles
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
	// DRAW_INDEX_INDIRECT read on the GPU: the guest address of its DrawIndexedIndirectArgs
	// (index count, instance count, start index, base vertex, start instance; the layout of
	// VkDrawIndexedIndirectCommand). index_addr is then the index buffer's base and index_count
	// its INDEX_BUFFER_SIZE, the bound of what the GPU may read.
	uint64_t         gpu_args_address           = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// DRAW_INDIRECT read on the GPU: the guest address of its DrawIndirectArgs (vertex count,
	// instance count, start vertex, start instance). The counts above are then unknown on the
	// CPU; only fast-launch mesh draws come this way.
	uint64_t         gpu_args_address           = 0;
	// Nonzero (vertex count known): the instance count is the dword the GPU reads there (the
	// NUM_INSTANCES a DRAW_INDIRECT read on the GPU left).
	uint64_t         gpu_instance_count_address = 0;
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
	// Whether commands record NVIDIA diagnostic checkpoints: the device has them and
	// KYTY_GPU_CHECKPOINTS=1 (or KYTY_BISECT's checkpoints:on) asks for them. Off by default since
	// round 69: three a draw, each a command the GPU executes, for a device-loss report only.
	[[nodiscard]] bool RecordsCheckpoints() const;
	// A checkpoint before a transfer (an upload, copy, clear, ...): the destination's guest
	// address, the source's (if any) and three numbers. Leaves the running draw's debug info and
	// annotations alone.
	void MarkTransfer(CommandBufferDebugOp op, uint64_t destination, uint64_t source = 0,
	                  uint32_t arg0 = 0, uint32_t arg1 = 0, uint32_t arg2 = 0) const;
	// Adds the running draw's shaders and sample counts to the checkpoint SetDebugInfo recorded
	// last (no-op without NV diagnostic checkpoints).
	// Puts the last annotation on the checkpoint SetDebugInfo recorded last (a later phase of
	// the same draw).
	void CarryDiagnosticAnnotation() const noexcept;
	void AnnotateDiagnosticCheckpoint(uint64_t vs_hash, uint64_t ps_hash, uint32_t color_samples,
	                                  uint32_t depth_samples,
	                                  uint32_t write_flags = 0) const noexcept;
	// Adds the draw's render targets to the checkpoint SetDebugInfo recorded last (carried to
	// later phases with the rest of the annotation).
	void AnnotateDiagnosticTargets(const DiagnosticTargets& targets) const noexcept;
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }

	// The raw command buffer. With the recording thread (CommandScheduler::EnableRecordingThread)
	// this first waits until the thread has run everything queued; hot paths use Recorder().
	[[nodiscard]] vk::CommandBuffer Handle() const;
	// Records directly, or through the recording thread.
	[[nodiscard]] CommandRecorder Recorder() const;
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
	// Begun and not submitted yet; m_buffer is its handle when not recorded on a thread.
	bool                m_open            = false;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	DiagnosticCheckpoint* m_debug_marker  = nullptr;
	// The last AnnotateDiagnosticCheckpoint values, for CarryDiagnosticAnnotation.
	mutable uint64_t      m_annotation_vs      = 0;
	mutable uint64_t      m_annotation_ps      = 0;
	mutable uint32_t      m_annotation_color   = 0;
	mutable uint32_t      m_annotation_depth   = 0;
	mutable uint32_t      m_annotation_writes  = 0;
	mutable DiagnosticTargets m_annotation_targets {};
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode);
	// Round 99: one dispatch of a shader whose call is partitioned runs once per partition
	// (PipelineCache::GetComputeProgram): these run the partition of partition->target (0: the
	// first), and receive every partition's target.
	void DispatchDirectPart(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                        uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
	                        PipelineCache::ComputePartition* partition);
	void DispatchIndirectPart(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                          uint32_t mode, PipelineCache::ComputePartition* partition);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	// Bindless heaps: register the draw's heaps, patch their regions, and once per frame resolve
	// the keys shaders flagged as pending.
	void PrepareBindlessHeaps(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void ResolveBindlessRequests();
	// Engine method (engine-method.md, E2), at the start of a guest submission: the heap entries
	// the guest wrote since the last one are compared and their keys settled again.
	void SyncBindlessHeaps();
	void SyncWrittenBindlessEntries();
	// E3: settle entries no draw has asked for yet, within the per-submission budget.
	void SettleBindlessHeaps();
	bool ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// The key's T# changed: release what it was settled to and settle it again.
	bool ResettleBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Release only: the key is pending again and its image unpinned when nothing else uses it.
	void ReleaseBindlessKey(BindlessTable::Heap& heap, uint32_t key);
	// Settles again, within the budget, the keys of a window of each heap whose T# changed since
	// they were settled; returns how many it resolved.
	uint32_t RevalidateBindlessKeys(uint32_t budget);
	void                           FindBuffers(PreparedBindings& bindings);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// Watch mode (GpuTiming): the pixel stage's first light grid as this draw reads it, and its
	// user-data buffer's words, into `detail.grid` (round 71).
	void DescribeLightGrid(const PreparedBindings& pixel, GpuTiming::DrawDetail& detail);
	// Diagnostics (round 73): a pixel shader's light-grid binding as the shader reads it (the
	// guest bytes of the layers it views), once a guest submission for each place: one with
	// cells of more than 1024 lights is counted and logged with its draw, its T# and the other
	// light-grid views of the submission ("LightGrid view:" lines).
	void CheckLightGridView(const ShaderStageRuntime& runtime, uint32_t index,
	                        const TextureBinding& binding);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	// `memo`: a draw's or a dispatch's binding (PrepareBindings), which may be the one the same
	// descriptor resolved to before (TextureBinding::memo_view) or be kept for that.
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            bool memo = false);
	// RebindImages: drops what the memo kept for a descriptor (its image had changed).
	void ForgetTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
	                       const ShaderRecompiler::IR::DescriptorValue& value);
	// RebindImages: keeps a binding just resolved and given its view for the descriptor's next uses.
	void RecordTextureMemo(const ShaderRecompiler::IR::ImageResource&   resource,
	                       const ShaderRecompiler::IR::DescriptorValue& value,
	                       const TextureBinding&                        binding);
	// `same_backing`: as TextureCache::FindImage's.
	[[nodiscard]] TextureBinding FindResolvedTexture(
	    const ShaderRecompiler::IR::ImageResource& resource, const ShaderTextureResource& descriptor,
	    TextureCache::ImageDesc desc, bool shader_conversion, vk::Format pixel_format,
	    vk::Format view_format, uint32_t size, bool* same_backing = nullptr);
	// KYTY_TABLE (tableBindings.h). The slot (0-2 vertex, 3 pixel) of a graphics stage's bindings.
	[[nodiscard]] uint32_t TableBindingsSlot(const PreparedBindings& prepared) const;
	// Whether the slot's last bindings were prepared from this program and descriptors (counts
	// the misses). KYTY_TABLE=2 keeps a copy of them for TableBindingsVerify.
	[[nodiscard]] bool TableBindingsMatch(PreparedBindings& prepared,
	                                      const ShaderStageRuntime& runtime, bool normalize);
	// Whether they may be used again now; with `replay`, binds their images as PrepareBindings
	// would.
	[[nodiscard]] bool TableBindingsReuse(PreparedBindings& prepared, bool replay, uint32_t slot);
	// A reused stage's per-draw uploads (user data words, flattened SRT, shader data).
	void TableBindingsUpload(PreparedBindings& prepared);
	// After the normal path: keep (or drop) the slot's entry; `verify`: compare with the copy.
	void TableBindingsStore(PreparedBindings& prepared, bool verify);
	// `table_reused`: bit i, stage i's bindings are the slot's last (TableBindingsMatch), not
	// prepared again.
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors, uint32_t table_reused = 0);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
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
	// A draw the guest uses for another operation (a metadata clear, a copy, a resolve) runs as
	// that: what it was, or null for a draw to run.
	[[nodiscard]] const char* ConsumeDrawAsOperation(CommandBuffer& buffer,
	                                                 uint32_t       render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	// `code_address`: the guest code of the dispatch's shader, which may be the SDK's pattern fill
	// (renderCompute.cpp); 0 leaves that out. The meta clear takes a pattern fill's value with the
	// dispatch's shape (group_x 0: not known).
	[[nodiscard]] bool TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                              const CommandBuffer&          buffer,
	                                              uint64_t code_address = 0, uint32_t group_x = 0,
	                                              uint32_t group_y = 0, uint32_t group_z = 0,
	                                              uint32_t mode = 0);
	[[nodiscard]] bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                               CommandBuffer& command, uint32_t group_x,
	                                               uint32_t group_y, uint32_t group_z,
	                                               uint32_t mode, uint64_t code_address = 0);

	// Temporary multisampled colour images for the EQAA fallback (EmulatesColorAtDepthSamples).
	struct MsColorTarget {
		ImageId                image_id   = 0;
		vk::Format             format     = vk::Format::eUndefined;
		uint32_t               width      = 0;
		uint32_t               height     = 0;
		uint32_t               layers     = 0;
		uint32_t               samples    = 0;
		uint32_t               base_level = 0;
		uint32_t               base_layer = 0;
		std::unique_ptr<Image> image;
		vk::ImageView          view = nullptr;
	};
	MsColorTarget& GetMsColorTarget(CommandBuffer& buffer, ImageId image_id, const Image& source,
	                                uint32_t width, uint32_t height, uint32_t layers,
	                                uint32_t samples, uint32_t base_level, uint32_t base_layer,
	                                bool& created);

	RenderContext&                        m_context;
	std::vector<MsColorTarget>            m_ms_color_targets;
	GraphicsBindings                     m_graphics_bindings;
	// KYTY_TABLE (tableTargets.h): the render targets' table.
	std::shared_ptr<TableTargets::Table> m_target_table; // shared: Table is incomplete here
	// KYTY_TABLE (tableBindings.h): the bindings table.
	std::shared_ptr<TableBindings::Table> m_binding_table;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	// Bindless heaps already surveyed (base ^ table offset << 48).
	std::unordered_set<uint64_t>          m_bindless_surveyed;

	void PrepareBindlessSamplers(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
	                    uint32_t index_offset, int32_t base_vertex, uint32_t first_vertex,
	                    int32_t host_vertex_offset, uint32_t host_first_instance, bool indirect);
	uint64_t                              m_bindless_frame = UINT64_MAX;
	std::vector<uint32_t>                 m_bindless_requests;
	std::vector<std::pair<std::array<uint32_t, 8>, uint32_t>> m_runtime_image_requests;
	std::vector<std::array<uint32_t, 8>>  m_bindless_window;
	std::vector<std::pair<uint64_t, uint64_t>> m_bindless_written;
	std::vector<ImageId>                       m_bindless_changed;
	uint64_t                                   m_bindless_eager_frame = UINT64_MAX;
	uint32_t                                   m_bindless_eager_left  = 0;
	bool                                       m_bindless_eager_device_full = false;
	uint64_t                                   m_bindless_eager_bytes = 0;
	// Bindless usage probe (UpdateBindlessUsageProbe): the frame it was armed in (0 = idle), the
	// first frame the next may start, and the frame the pending feedback snapshot was recorded.
	uint64_t                              m_bindless_probe_frame      = 0;
	uint64_t                              m_bindless_next_probe       = 0;
	uint64_t                              m_bindless_snapshot_frame   = 0;
	std::vector<ImageId>                  m_bindless_unused;
	void UpdateBindlessUsageProbe(uint64_t frame, uint64_t snapshot_frame);
	std::vector<uint32_t>                 m_bindless_srt;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	// Created at the first thread-dimension indirect dispatch.
	std::unique_ptr<IndirectDispatchGroups> m_indirect_groups;
	// Ring of VkDrawMeshTasksIndirectCommandEXT for indirect fast-launch mesh draws (16 bytes an
	// entry); an entry is rewritten only after the ring has gone round, behind its last read.
	std::unique_ptr<Buffer>                 m_mesh_indirect;
	uint32_t                                m_mesh_indirect_next = 0;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

// `why` (optional): the check that failed, when it is false.
[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size,
                                            const char** why = nullptr);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
