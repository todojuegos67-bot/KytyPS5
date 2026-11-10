#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace vk::detail { class DispatchLoaderDynamic; }
struct VkCommandBuffer_T;

// Vulkan command recording on a worker thread (KYTY_VULKAN_RECORDING). The
// renderer remains the sole resource/cache owner; a producer scope transfers
// only fully copied Vulkan call arguments.
namespace LocalVulkanRecording {
void Install();
// The Vulkan entry points without recording: for a queue the recording worker does not own.
const vk::detail::DispatchLoaderDynamic& DirectDispatch();
// Replaces the vkCmdBindPipeline every bind ends in (recorded, packet or direct: the device-loss breadcrumbs of
// device-fault.h) and returns the one replaced. After Install, before any command is recorded.
void* ReplaceBindPipeline(void* entry);
// Complete this producer's CPU recording before handing its Vulkan objects to
// another thread. A receiver's drain cannot flush the sender's thread-local queue.
// This does not wait for GPU execution.
void Drain();
struct Segment {
    const void* data = nullptr;
    size_t size = 0;
};
using ReplayPacket = void (*)(std::span<const Segment>, const vk::detail::DispatchLoaderDynamic&);
// Draws and descriptor encoding are recorded as whole packets.
bool PacketsEnabled();
// Segments are copied transactionally. An optional immutable owner is retained
// until replay finishes; callers never lend stack or mutable-cache pointers.
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner = {});
// Preserve order when a packet cannot fit. Does not retain any argument.
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments);
// A packet's replay that records commands the recording does not see (a library's own: DLSS through Streamline)
// ends the reuse of bound pipelines and dynamic state the draws' replays keep, as a direct call does.
void InvalidateReplayState();
// Command-buffer begin/end and queue submission, queued in order with the recorded
// commands instead of draining the recording queue first (KYTY_DEFERRED_SUBMIT).
// A submit packet is published at once: other threads may wait on its tick.
bool DeferredSubmitEnabled();
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish);
// Deferred queue submissions: counted when queued and when the worker has handed them to
// the driver. WaitDeferredSubmits blocks until every submission queued so far reached the
// driver (not the GPU). A present must not depend on a signal still in the worker's queue:
// the Windows driver then blocks inside vkQueuePresentKHR, holding the queue lock the
// worker needs for that very submission.
void NoteDeferredSubmitQueued();
void NoteDeferredSubmitDone();
void WaitDeferredSubmits();
// Deferred submissions queued and handed to the driver so far (diagnostics).
uint64_t DeferredSubmitsQueued();
uint64_t DeferredSubmitsDone();
uint64_t StateEpoch();
// Commands the calling thread recorded that do GPU work or synchronize (draws, dispatches,
// copies, clears, rendering scopes, barriers): unchanged between two barriers = no work in between.
uint64_t WorkCalls();
// Work calls and packets (draws, descriptors) the calling thread recorded: unchanged = nothing recorded.
uint64_t RecordedWork();
// Commands that change memory or images (draws, dispatches, copies, clears) and packets the calling thread
// recorded: unchanged = nothing but barriers and rendering scopes recorded.
uint64_t RecordedWrites();
class ProducerScope {
public:
    ProducerScope();
    ~ProducerScope();
    ProducerScope(const ProducerScope&) = delete;
    ProducerScope& operator=(const ProducerScope&) = delete;
};

// Worker packets spliced at commit. A passive producer (a speculation worker's) records exactly as a producer
// does, into chunks from its own pool, with no replay thread: nothing it records reaches the driver until the
// render thread splices the chunks into its own stream, where its replay thread executes them in order between
// the render thread's packets. Its commands name the placeholder command buffer, which a spliced chunk's replay
// resolves to the splice's target (the render thread's open command buffer).
struct Chunk;
// A passive producer's closed segment: its chunks in recording order, not replayed. Destroying (or Discard) runs
// each packet's destructor without executing it and returns the chunks to their pool.
class ChunkList {
public:
    ChunkList() = default;
    ChunkList(ChunkList&& other) noexcept;
    ChunkList& operator=(ChunkList&& other) noexcept;
    ~ChunkList();
    ChunkList(const ChunkList&) = delete;
    ChunkList& operator=(const ChunkList&) = delete;
    [[nodiscard]] bool empty() const { return chunks.empty(); }
    std::vector<Chunk*> chunks;
    uint64_t commands = 0;
    // A command of the segment could not be recorded (too large, an unknown extension chain, a direct call on the
    // placeholder): the segment is incomplete and must be discarded, never spliced.
    bool failed = false;
};
VkCommandBuffer_T* PlaceholderCommandBuffer();
// The placeholder resolved to the target of the spliced chunk being replayed; any other handle is itself.
VkCommandBuffer_T* Resolve(VkCommandBuffer_T* command);
// The calling thread records into its passive producer (inside a segment).
bool Passive();
// The calling thread has a passive producer (PassiveProducerScope).
bool HasPassiveProducer();
// The passive producer's segment: Begin discards anything recorded since the last End and arms the producer (the
// thread's calls are recorded from then on); End disarms it, closes the open chunk and hands the segment's chunks over.
// Outside a segment the thread's calls go to the driver as without a producer.
void BeginSegment();
ChunkList EndSegment();
void Discard(ChunkList& list);
// Render thread (a producer's thread): the list's chunks appended to its stream after everything it recorded so
// far, each replayed with the placeholder resolved to `target`. A failed list is discarded instead (false).
bool Splice(ChunkList& list, VkCommandBuffer_T* target);
// Chunks spliced and passive chunks allocated so far (diagnostics).
uint64_t SplicedChunks();
uint64_t PassiveChunksAllocated();
class PassiveProducerScope {
public:
    PassiveProducerScope();
    ~PassiveProducerScope();
    PassiveProducerScope(const PassiveProducerScope&) = delete;
    PassiveProducerScope& operator=(const PassiveProducerScope&) = delete;
};
}
