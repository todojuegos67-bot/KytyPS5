#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace vk::detail { class DispatchLoaderDynamic; }

// Vulkan command recording on a worker thread (KYTY_VULKAN_RECORDING). The
// renderer remains the sole resource/cache owner; a producer scope transfers
// only fully copied Vulkan call arguments.
namespace LocalVulkanRecording {
void Install();
// The Vulkan entry points without recording: for a queue the recording worker does not own.
const vk::detail::DispatchLoaderDynamic& DirectDispatch();
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
}
