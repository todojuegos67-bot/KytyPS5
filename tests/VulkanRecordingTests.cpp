#include "vulkan-recording.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <array>
#include <string>
#include <memory>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
// The recording's thread naming and pinning (local-platform.cpp is not linked into this test).
namespace LocalPlatform {
void SetThreadName(const char*) {}
uint32_t ThreadId() { return 0; }
void PinThreadToCpuList(const char*) {}
bool ProcessCpusIdleTime(uint64_t*, uint64_t*) { return false; }
}
extern "C" volatile std::atomic<uint32_t> kyty_local_vulkan_recording_mode;
namespace {
void Check(bool condition) { if (!condition) std::abort(); }
std::atomic<uint64_t> draws {0};
std::atomic<bool> stalled {false}, release {false};
uint64_t sequence = 0;
bool expect_extension = false;
uint64_t packet_epoch = UINT64_MAX;
// Draws numbered [splice_lo, splice_hi) were recorded by a passive producer and must reach the driver on the splice
// target; every other draw on the null command buffer.
VkCommandBuffer splice_target = nullptr;
uint32_t splice_lo = UINT32_MAX, splice_hi = 0;
struct PacketData { uint32_t sequence; uint32_t same_epoch; };
void Packet(std::span<const LocalVulkanRecording::Segment> segments,
            const vk::detail::DispatchLoaderDynamic& dispatch) {
    const auto& item = *static_cast<const PacketData*>(segments[0].data);
    const auto* values = static_cast<const uint32_t*>(segments[1].data);
    Check(segments[1].size == 4 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 4; ++i) Check(values[i] == 100 + i);
    const auto epoch = LocalVulkanRecording::StateEpoch();
    if (epoch != UINT64_MAX) Check((epoch == packet_epoch) == bool(item.same_epoch));
    packet_epoch = epoch;
    dispatch.vkCmdDraw(nullptr, 3, 1, item.sequence, 7);
}
VKAPI_ATTR void VKAPI_CALL LineWidth(VkCommandBuffer, float value) { Check(value == 1.0f); }

VKAPI_ATTR void VKAPI_CALL Draw(VkCommandBuffer command, uint32_t vertices, uint32_t instances,
                               uint32_t first, uint32_t base) {
    Check(first == sequence++ && vertices == 3 && base == 7);
    Check(command == (first >= splice_lo && first < splice_hi ? splice_target : nullptr));
    if (instances == 999) {
        stalled.store(true);
        while (!release.load()) std::this_thread::yield();
    }
    draws.fetch_add(1);
}
VKAPI_ATTR void VKAPI_CALL Push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                               uint32_t, uint32_t bytes, const void* pointer) {
    Check(bytes == 16);
    const auto* values = static_cast<const uint32_t*>(pointer);
    for (uint32_t i = 0; i < 4; ++i) Check(values[i] == 100 + i);
}
VKAPI_ATTR void VKAPI_CALL Descriptors(VkCommandBuffer, VkPipelineBindPoint,
    VkPipelineLayout, uint32_t, uint32_t count, const VkWriteDescriptorSet* writes) {
    Check(count == 2);
    Check(writes[0].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    Check(writes[0].pBufferInfo[1].range == 1234);
    Check(writes[1].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    Check(writes[1].pImageInfo[0].imageLayout == VK_IMAGE_LAYOUT_GENERAL);
}
VKAPI_ATTR void VKAPI_CALL Barrier(VkCommandBuffer, const VkDependencyInfo* info) {
    Check(info->bufferMemoryBarrierCount == 2);
    Check(info->pBufferMemoryBarriers[1].size == 9876);
    Check(bool(info->pNext) == expect_extension);
}
VKAPI_ATTR void VKAPI_CALL Rendering(VkCommandBuffer, const VkRenderingInfo* info) {
    Check(info->colorAttachmentCount == 1 && info->pColorAttachments[0].clearValue.color.uint32[2] == 42);
    Check(info->pDepthAttachment->clearValue.depthStencil.depth == 0.5f);
}
VKAPI_ATTR VkResult VKAPI_CALL End(VkCommandBuffer) {
    Check(draws.load() == sequence);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL Counter(VkDevice, VkSemaphore, uint64_t* value) {
    *value = 123;
    return VK_SUCCESS;
}
}

int main() {
    auto& api = VULKAN_HPP_DEFAULT_DISPATCHER;
    api.vkCmdDraw = Draw;
    api.vkCmdPushConstants = Push;
    api.vkCmdPushDescriptorSetKHR = Descriptors;
    api.vkCmdPipelineBarrier2 = Barrier;
    api.vkCmdBeginRendering = Rendering;
    api.vkEndCommandBuffer = End;
    api.vkGetSemaphoreCounterValue = Counter;
    api.vkCmdSetLineWidth = LineWidth;
    LocalVulkanRecording::Install();
    {
        LocalVulkanRecording::ProducerScope scope;
        kyty_local_vulkan_recording_mode.store(1);
        // Publish a full block whose first command holds the consumer. The
        // following parameter storage will be destroyed before it is read.
        for (uint32_t i = 0; i < 128; ++i) api.vkCmdDraw(nullptr, 3, i ? 1 : 999, i, 7);
        while (!stalled.load()) std::this_thread::yield();
        uint64_t counter = 0;
        // The queued command is deliberately blocked. Querying already
        // submitted GPU progress must not wait for this CPU recording work.
        Check(api.vkGetSemaphoreCounterValue(nullptr, nullptr, &counter) == VK_SUCCESS && counter == 123);
        {
            uint32_t values[] {100, 101, 102, 103};
            api.vkCmdPushConstants(nullptr, nullptr, 0, 0, sizeof(values), values);
            values[0] = 0;
            VkDescriptorBufferInfo buffers[2] {};
            buffers[1].range = 1234;
            VkDescriptorImageInfo image {.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet writes[2] {
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .descriptorCount = 2,
                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 .pImageInfo = reinterpret_cast<const VkDescriptorImageInfo*>(1), .pBufferInfo = buffers},
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &image,
                 .pBufferInfo = reinterpret_cast<const VkDescriptorBufferInfo*>(1)},
            };
            api.vkCmdPushDescriptorSetKHR(nullptr, VK_PIPELINE_BIND_POINT_GRAPHICS, nullptr, 0, 2, writes);
            buffers[1].range = 0;
            image.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkBufferMemoryBarrier2 barriers[2] {};
            barriers[1].size = 9876;
            VkDependencyInfo dependency {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = 2, .pBufferMemoryBarriers = barriers};
            api.vkCmdPipelineBarrier2(nullptr, &dependency);
            barriers[1].size = 0;
            VkRenderingAttachmentInfo color {}, depth {};
            color.clearValue.color.uint32[2] = 42;
            depth.clearValue.depthStencil.depth = 0.5f;
            VkRenderingInfo rendering {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .colorAttachmentCount = 1, .pColorAttachments = &color, .pDepthAttachment = &depth};
            api.vkCmdBeginRendering(nullptr, &rendering);
            color.clearValue = {};
            depth.clearValue = {};
        }
        release.store(true);
        Check(api.vkEndCommandBuffer(nullptr) == VK_SUCCESS);
        // Multiple ring wraps and mode changes must retain every call in order.
        for (uint32_t i = 128; i < 20000; ++i) api.vkCmdDraw(nullptr, 3, 1, i, 7);
        kyty_local_vulkan_recording_mode.store(0);
        api.vkCmdDraw(nullptr, 3, 1, 20000, 7);
        Check(draws.load() == 20001);
        kyty_local_vulkan_recording_mode.store(1);
        api.vkCmdDraw(nullptr, 3, 1, 20001, 7);
        // Unknown extension graphs are passed synchronously after a drain.
        VkBufferMemoryBarrier2 barriers[2] {};
        barriers[1].size = 9876;
        uint64_t opaque = 0;
        VkDependencyInfo extension {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .pNext = &opaque,
            .bufferMemoryBarrierCount = 2, .pBufferMemoryBarriers = barriers};
        expect_extension = true;
        api.vkCmdPipelineBarrier2(nullptr, &extension);
        Check(draws.load() == 20002);

        // A nontrivial packet keeps its plan alive, owns every input segment,
        // and shares ordering with raw commands. Block the consumer once more.
        stalled.store(false);
        release.store(false);
        for (uint32_t i = 20002; i < 20130; ++i)
            api.vkCmdDraw(nullptr, 3, i == 20002 ? 999 : 1, i, 7);
        while (!stalled.load()) std::this_thread::yield();
        auto owner = std::make_shared<std::array<uint32_t, 4>>();
        std::weak_ptr<const void> weak = owner;
        PacketData item {20130, false};
        uint32_t values[] {100, 101, 102, 103};
        const LocalVulkanRecording::Segment data[] {{&item, sizeof(item)}, {values, sizeof(values)}};
        Check(LocalVulkanRecording::EnqueuePacket(Packet, data, owner));
        owner.reset();
        Check(!weak.expired());
        ++item.sequence;
        item.same_epoch = true;
        Check(LocalVulkanRecording::EnqueuePacket(Packet, data));
        api.vkCmdSetLineWidth(nullptr, 1.0f);
        ++item.sequence;
        item.same_epoch = false;
        Check(LocalVulkanRecording::EnqueuePacket(Packet, data));
        values[0] = 0;
        release.store(true);
        Check(api.vkEndCommandBuffer(nullptr) == VK_SUCCESS);
        Check(weak.expired() && draws.load() == 20133);

        // Synchronous fallback must preserve the order of previously queued work.
        values[0] = 100;
        ++item.sequence;
        Check(LocalVulkanRecording::EnqueuePacket(Packet, data));
        ++item.sequence;
        kyty_local_vulkan_recording_mode.store(0);
        Check(!LocalVulkanRecording::EnqueuePacket(Packet, data));
        LocalVulkanRecording::ReplayInline(Packet, data);
        Check(draws.load() == 20135);
        kyty_local_vulkan_recording_mode.store(1);
        std::vector<std::byte> oversized(65536);
        const LocalVulkanRecording::Segment large[] {{oversized.data(), oversized.size()}};
        Check(!LocalVulkanRecording::EnqueuePacket(Packet, large));
        const LocalVulkanRecording::Segment invalid[] {{nullptr, 1}};
        Check(!LocalVulkanRecording::EnqueuePacket(Packet, invalid));

        // Worker packets spliced at commit: a passive producer records against the placeholder and replays
        // nothing; a spliced segment executes in the render thread's order on the target, a discarded one only
        // destroys its captures, a failed one (a command it could not record) never reaches the driver.
        auto* const placeholder = LocalVulkanRecording::PlaceholderCommandBuffer();
        auto* const target = reinterpret_cast<VkCommandBuffer>(uintptr_t {0x1000});
        for (uint32_t round = 0; round < 2; ++round) {
            const uint32_t first = static_cast<uint32_t>(draws.load()), worker_draws = 300;
            LocalVulkanRecording::ChunkList spliced, discarded, failed;
            std::weak_ptr<const void> discard_owner;
            std::thread worker([&] {
                LocalVulkanRecording::PassiveProducerScope passive;
                Check(!LocalVulkanRecording::Passive() && !LocalVulkanRecording::PacketsEnabled());
                LocalVulkanRecording::BeginSegment();
                Check(LocalVulkanRecording::Passive() && LocalVulkanRecording::PacketsEnabled());
                Check(!LocalVulkanRecording::DeferredSubmitEnabled());
                for (uint32_t i = 0; i < worker_draws; ++i) api.vkCmdDraw(placeholder, 3, 1, first + 10 + i, 7);
                spliced = LocalVulkanRecording::EndSegment();
                LocalVulkanRecording::BeginSegment();
                // (Executed, these would break the draw order and the placeholder check.)
                for (uint32_t i = 0; i < 200; ++i) api.vkCmdDraw(placeholder, 3, 1, 999999, 7);
                auto owner = std::make_shared<std::array<uint32_t, 4>>();
                discard_owner = owner;
                PacketData item {999999, false};
                uint32_t values[] {100, 101, 102, 103};
                const LocalVulkanRecording::Segment data[] {{&item, sizeof(item)}, {values, sizeof(values)}};
                Check(LocalVulkanRecording::EnqueuePacket(Packet, data, owner));
                discarded = LocalVulkanRecording::EndSegment();
                LocalVulkanRecording::BeginSegment();
                api.vkCmdDraw(placeholder, 3, 1, 999999, 7);
                VkBufferMemoryBarrier2 barriers[2] {};
                uint64_t opaque = 0;
                VkDependencyInfo extension {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .pNext = &opaque,
                    .bufferMemoryBarrierCount = 2, .pBufferMemoryBarriers = barriers};
                api.vkCmdPipelineBarrier2(placeholder, &extension);
                // A packet that cannot be recorded is not replayed inline either.
                Check(!LocalVulkanRecording::EnqueuePacket(Packet, large));
                LocalVulkanRecording::ReplayInline(Packet, data);
                failed = LocalVulkanRecording::EndSegment();
                // Outside a segment the calls reach the driver at once, as on a thread without a producer.
                Check(!LocalVulkanRecording::Passive());
                api.vkCmdSetLineWidth(nullptr, 1.0f);
            });
            worker.join();
            Check(!spliced.failed && spliced.commands == worker_draws && spliced.chunks.size() == 3);
            Check(!discarded.failed && discarded.commands == 201 && !discard_owner.expired());
            Check(failed.failed);
            Check(draws.load() == first);
            splice_target = target;
            splice_lo = first + 10;
            splice_hi = first + 10 + worker_draws;
            for (uint32_t i = 0; i < 10; ++i) api.vkCmdDraw(nullptr, 3, 1, first + i, 7);
            Check(LocalVulkanRecording::Splice(spliced, target) && spliced.empty());
            for (uint32_t i = 0; i < 10; ++i) api.vkCmdDraw(nullptr, 3, 1, first + 10 + worker_draws + i, 7);
            Check(!LocalVulkanRecording::Splice(failed, target));
            LocalVulkanRecording::Discard(discarded);
            Check(discard_owner.expired());
            Check(api.vkEndCommandBuffer(nullptr) == VK_SUCCESS);
            Check(draws.load() == first + 20 + worker_draws);
            Check(LocalVulkanRecording::SplicedChunks() == 3 * (round + 1));
            // The second round reuses the first one's pool: every chunk came back (3 spliced, 2 discarded, the
            // failed segment's closed one and its open one).
            Check(LocalVulkanRecording::PassiveChunksAllocated() == 7);
            splice_lo = UINT32_MAX;
            splice_hi = 0;
        }
    }
    std::puts("Vulkan recording: raw/packet ordering, pointer/plan ownership, state invalidation, ring reuse, query, fallback and passive splice / discard passed");
}
