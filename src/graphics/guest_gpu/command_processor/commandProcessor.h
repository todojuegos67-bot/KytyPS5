#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "graphics/host_gpu/renderer/replay.h"
#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <chrono>
#include <cstdint>
#include <source_location>
#include <span>
#include <vector>

namespace Libs::Graphics {

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// Draw prep (KYTY_DRAW_PREP): the submission number of a graphics command stream (0: not
	// counted), whose draw and dispatch packets are numbered as the scanner numbers them.
	void SetPrepSeq(uint64_t seq, uint32_t queue) noexcept {
		m_prep_seq   = seq;
		m_prep_queue = queue;
	}
	// Round 108: the stream may stop between packets, once it has run some packets and for
	// `microseconds` (0: never), when the count of compute submissions enqueued
	// (GuestGpu::ComputeEnqueued) is past `mark`; Yielded says so, and the next Process goes on
	// from the packet it stopped before.
	void AllowYield(uint32_t microseconds, uint64_t mark) noexcept {
		m_yield_us      = microseconds;
		m_yield_mark    = mark;
		m_yield_packets = 0;
		m_yielded       = false;
		m_yield_start   = (microseconds != 0 ? std::chrono::steady_clock::now()
		                                     : std::chrono::steady_clock::time_point {});
	}
	[[nodiscard]] bool Yielded() const noexcept { return m_yielded; }
	[[nodiscard]] bool YieldTimeReached() const noexcept {
		return std::chrono::steady_clock::now() - m_yield_start >=
		       std::chrono::microseconds(m_yield_us);
	}

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
	uint64_t                  m_prep_seq      = 0;
	uint32_t                  m_prep_queue    = 0;
	uint32_t                  m_prep_ordinal  = 0;
	const uint32_t*           m_prep_packet   = nullptr; // the last numbered packet
	Replay::StreamState       m_replay;                  // gt7-replay-1119 (KYTY_REPLAY)
	uint32_t                              m_yield_us      = 0;
	bool                                  m_yielded       = false;
	std::chrono::steady_clock::time_point m_yield_start {};
	uint64_t                              m_yield_mark    = 0;
	uint32_t                              m_yield_packets = 0;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	void            BufferWait();
	// See CommandScheduler::CompleteReleaseMemWrite -- only safe for a RELEASE_MEM that already
	// wrote its guest-visible value synchronously and scheduled no interrupt callback.
	void            CompleteReleaseMemWrite();
	// See CommandScheduler::CompleteReleaseMemInterrupt -- for a RELEASE_MEM that DOES request a
	// guest interrupt/event; uses a smaller batch bound as a hedge against delaying it.
	void            CompleteReleaseMemInterrupt();
	// See CommandScheduler::CompleteDraw -- called after every DrawIndex/DrawAuto.
	void            CompleteDraw();
	// See CommandScheduler::CompleteDispatch -- called after every dispatch.
	void            CompleteDispatch();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed, uint32_t draw_index_register);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	// Round 135: the full barrier the PM4 cache flushes, RELEASE_MEM and suspend points recorded
	// before (mainline 70668fbd dropped them; the renderer's resource barriers order the work), now
	// only with KYTY_PM4_BARRIERS=global or KYTY_BISECT's barriers:global, to compare
	// (GuestGpu::GlobalPm4Barriers).
	void EmitGlobalBarrier(const std::source_location& site = std::source_location::current());
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	// A REWIND packet: the queue waits at it until its valid bit is set. Round 124: a compute
	// queue waits at most 3 s at one packet, counted while it is polled (a host stall counts
	// 250 ms at most). Once the CPU has released rewinds (the ACB patch function made packets
	// valid, or one this queue waited at became valid), that packet is passed, up to 8 times;
	// otherwise, or after that, the queue no longer waits at REWINDs (as before the ACB's rewinds
	// were written at all). Both are logged.
	void WaitForRewind(bool valid, const uint32_t* packet);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);
	// Before the command processor stores to guest memory itself (a label the CPU may wait on):
	// the page protections and upload reads it deferred are done (round 111).
	void BeforeGuestStore();

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	// A label as the command processor sees it: end-of-pipe writes count once recorded, even
	// before the CPU can see them (GuestGpu::LabelsAtCompletion).
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* addr) const;
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	void SuspendPm4();
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;
	// The DrawIndirectArgs whose instance count NUM_INSTANCES holds after a GPU-read indirect draw;
	// read on the CPU only if a later draw takes its instance count from there.
	uint64_t m_num_instances_args = 0;
	[[nodiscard]] uint32_t NumInstances();

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	// WaitForRewind's watchdog (compute queues): the packet waited at, when it was polled last,
	// the polled time it has waited, whether the CPU released a REWIND this queue waited at, the
	// packets passed after a timeout, and whether the queue no longer waits.
	const uint32_t*                       m_rewind_packet    = nullptr;
	std::chrono::steady_clock::time_point m_rewind_polled    = {};
	std::chrono::steady_clock::duration   m_rewind_waited    = {};
	bool                                  m_rewind_released  = false;
	uint32_t                              m_rewind_timeouts  = 0;
	bool                                  m_rewind_waits_off = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
