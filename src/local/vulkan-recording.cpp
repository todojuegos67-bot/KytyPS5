#include "vulkan-recording.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "live-census.h"
#include "live-trace-gpu.h"
#include "local-platform.h"
#include "time-census.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <type_traits>

extern "C" {
// 1: record Vulkan commands for the worker thread, draws and descriptor
// encoding as whole packets (KYTY_VULKAN_RECORDING).
volatile std::atomic<uint32_t> kyty_local_vulkan_recording_mode {0};
// 1: queue command-buffer begin/end/submit instead of draining (DeferredSubmitEnabled).
volatile std::atomic<uint32_t> kyty_local_deferred_submit_mode {0};
// Pauses the idle worker spins before it blocks (a blocked worker costs the render thread a
// wake-up system call per publish on Windows).
#if defined(_WIN32)
volatile std::atomic<uint32_t> kyty_local_recording_spin {4000};
#else
volatile std::atomic<uint32_t> kyty_local_recording_spin {0};
#endif
}

namespace LocalVulkanRecording {
namespace {
constexpr size_t ChunkBytes = 64 * 1024;
// Deep enough that a burst of recorded calls never blocks the producer in WaitForRoom.
constexpr size_t ChunkCount = 64;
constexpr uint32_t End = std::numeric_limits<uint32_t>::max();
vk::detail::DispatchLoaderDynamic original;

struct Header {
    void (*execute)(void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    uint32_t payload = 0;
    uint32_t next = End;
};
struct Chunk {
    alignas(64) std::array<std::byte, ChunkBytes> data;
    size_t used = 0;
    uint32_t first = End, last = End, count = 0;
};

// A transaction owns one command plus its parameter graph in a single chunk.
// Failure never publishes any of its partially copied data.
class Writer {
public:
    explicit Writer(Chunk& chunk) : data(chunk.data.data()), used(chunk.used) {}
    bool valid = true, overflow = false;
    std::byte* data;
    size_t used;
    uint32_t command = End;

    void* Allocate(size_t bytes, size_t alignment) {
        if (!valid) return nullptr;
        const size_t offset = (used + alignment - 1) & ~(alignment - 1);
        if (bytes > ChunkBytes || offset > ChunkBytes - bytes) {
            valid = false;
            overflow = true;
            return nullptr;
        }
        used = offset + bytes;
        return data + offset;
    }
    const void* Copy(const void* source, size_t bytes) {
        if (!source || !bytes) return nullptr;
        auto* target = Allocate(bytes, alignof(std::max_align_t));
        if (target) std::memcpy(target, source, bytes);
        return target;
    }
    template<typename T>
    const T* Copy(const T* source, size_t count) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!source || !count) return nullptr;
        if (count > ChunkBytes / sizeof(T)) { valid = false; overflow = true; return nullptr; }
        auto* target = static_cast<T*>(Allocate(count * sizeof(T), alignof(T)));
        if (!target) return nullptr;
        std::memcpy(target, source, count * sizeof(T));
        for (size_t i = 0; i < count && valid; ++i) {
            auto& value = target[i];
            if constexpr (requires { value.pNext; }) {
                // Unknown extension chains retain the exact synchronous call.
                if (value.pNext) { valid = false; break; }
            }
            if constexpr (std::is_same_v<T, VkDependencyInfo>) {
                value.pMemoryBarriers = Copy(value.pMemoryBarriers, value.memoryBarrierCount);
                value.pBufferMemoryBarriers = Copy(value.pBufferMemoryBarriers, value.bufferMemoryBarrierCount);
                value.pImageMemoryBarriers = Copy(value.pImageMemoryBarriers, value.imageMemoryBarrierCount);
            } else if constexpr (std::is_same_v<T, VkRenderingInfo>) {
                value.pColorAttachments = Copy(value.pColorAttachments, value.colorAttachmentCount);
                value.pDepthAttachment = Copy(value.pDepthAttachment, 1);
                value.pStencilAttachment = Copy(value.pStencilAttachment, 1);
            } else if constexpr (std::is_same_v<T, VkWriteDescriptorSet>) {
                // Ignored fields need not point to readable memory. Copy only
                // the payload selected by descriptorType, never all pointers.
                value.pImageInfo = nullptr;
                value.pBufferInfo = nullptr;
                value.pTexelBufferView = nullptr;
                switch (value.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                        value.pImageInfo = Copy(source[i].pImageInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                        value.pBufferInfo = Copy(source[i].pBufferInfo, value.descriptorCount);
                        break;
                    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                        value.pTexelBufferView = Copy(source[i].pTexelBufferView, value.descriptorCount);
                        break;
                    default: valid = false; break;
                }
            }
        }
        return target;
    }
    template<typename Function>
    void Command(Function function) {
        static_assert(std::is_nothrow_move_constructible_v<Function>);
        static_assert(std::is_nothrow_destructible_v<Function>);
        auto* header = static_cast<Header*>(Allocate(sizeof(Header), alignof(Header)));
        void* payload = Allocate(sizeof(Function), alignof(Function));
        if (!valid) return;
        new (payload) Function(std::move(function));
        new (header) Header {
            .execute = [](void* pointer) { (*static_cast<Function*>(pointer))(); },
            .destroy = nullptr,
            .payload = static_cast<uint32_t>(static_cast<std::byte*>(payload) - data),
        };
        if constexpr (!std::is_trivially_destructible_v<Function>)
            header->destroy = [](void* pointer) { static_cast<Function*>(pointer)->~Function(); };
        command = static_cast<uint32_t>(reinterpret_cast<std::byte*>(header) - data);
    }
};

class Stream;
thread_local Stream* executing = nullptr;
class Stream {
public:
    Stream() : worker([this] { Run(); }) {}
    ~Stream() {
        Drain();
        stopping.store(true, std::memory_order_release);
        // atomic::wait requires a changed value, not just notify, to terminate.
        published.fetch_add(1, std::memory_order_release);
        published.notify_one();
        worker.join();
    }
    alignas(64) std::atomic<uint64_t> state_epoch {0};
    alignas(64) bool was_recording = false;
    void BeforeDirect() {
        Drain();
        const bool active = kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
        if (active || was_recording) state_epoch.fetch_add(1, std::memory_order_relaxed);
        was_recording = active;
    }
    template<typename Capture>
    bool Enqueue(Capture capture) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            WaitForRoom();
            auto& chunk = chunks[sequence % ChunkCount];
            Writer writer(chunk);
            capture(writer);
            if (writer.valid && writer.command != End) {
                if (chunk.last != End)
                    reinterpret_cast<Header*>(chunk.data.data() + chunk.last)->next = writer.command;
                else chunk.first = writer.command;
                chunk.last = writer.command;
                chunk.used = writer.used;
                ++chunk.count;
                if (chunk.count >= 128 || chunk.used >= ChunkBytes * 3 / 4) Publish();
                return true;
            }
            if (!writer.overflow || chunk.count == 0) break;
            Publish();
        }
        return false;
    }
    void Flush() { Publish(); }
    // Packets recorded or published and not yet replayed (a direct call would wait for them).
    [[nodiscard]] bool Pending() const {
        return chunks[sequence % ChunkCount].count != 0 || completed.load(std::memory_order_acquire) != sequence;
    }
    void Drain() {
        Publish();
        if (completed.load(std::memory_order_acquire) == sequence) return;
        for (auto done = completed.load(std::memory_order_acquire); done != sequence;
             done = completed.load(std::memory_order_acquire)) completed.wait(done, std::memory_order_acquire);
    }
private:
    // Producer and consumer only touch a slot while they own it. Release /
    // acquire counters also publish resets before the producer reuses a slot.
    std::array<Chunk, ChunkCount> chunks;
    uint64_t sequence = 0;
    alignas(64) std::atomic<uint64_t> published {0};
    alignas(64) std::atomic<uint64_t> completed {0};
    alignas(64) std::atomic<bool> sleeping {false};
    std::atomic<bool> stopping {false};
    std::thread worker;

    void WaitForRoom() {
        for (auto done = completed.load(std::memory_order_acquire); sequence - done >= ChunkCount;
             done = completed.load(std::memory_order_acquire)) completed.wait(done, std::memory_order_acquire);
    }
    void Publish() {
        WaitForRoom();
        auto& chunk = chunks[sequence % ChunkCount];
        if (chunk.count == 0) return;
        published.store(++sequence, std::memory_order_seq_cst);
        // Wake the worker only when it said it blocks (both sides seq_cst): a wake-up is a
        // system call on Windows, and the worker is usually busy or still spinning.
        if (sleeping.load(std::memory_order_seq_cst)) published.notify_one();
    }
    // The replay thread must not land on an efficiency core: every GPU wait includes its
    // latency.  KYTY_RECORDING_CPUS (e.g. "1,2,3,6,7") restricts it; re-applied now and
    // then because the launcher assigns one mask to every game thread at startup.
    static void KeepAffinity() {
        static const char* const list = std::getenv("KYTY_RECORDING_CPUS");
        if (list) LocalPlatform::PinThreadToCpuList(list);
    }
    void Run() {
        executing = this;
        LocalPlatform::SetThreadName("Kyty.Record");
        KeepAffinity();
        uint64_t current = 0;
        for (;;) {
            if ((current & 4095u) == 0) KeepAffinity();
            for (auto ready = published.load(std::memory_order_acquire); current == ready;
                 ready = published.load(std::memory_order_acquire)) {
                const uint32_t spins = kyty_local_recording_spin.load(std::memory_order_relaxed);
                for (uint32_t spin = 0; spin < spins && ready == current; ++spin) {
                    _mm_pause();
                    ready = published.load(std::memory_order_acquire);
                }
                if (ready != current) break;
                sleeping.store(true, std::memory_order_seq_cst);
                if (published.load(std::memory_order_seq_cst) == current)
                    published.wait(current, std::memory_order_seq_cst);
                sleeping.store(false, std::memory_order_relaxed);
            }
            if (stopping.load(std::memory_order_acquire)) return;
            auto& chunk = chunks[current % ChunkCount];
            for (auto offset = chunk.first; offset != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + offset);
                header->execute(chunk.data.data() + header->payload);
                if (header->destroy) header->destroy(chunk.data.data() + header->payload);
                offset = header->next;
            }
            chunk.used = 0;
            chunk.count = 0;
            chunk.first = chunk.last = End;
            completed.store(++current, std::memory_order_release);
            completed.notify_one();
        }
    }
};

thread_local std::unique_ptr<Stream> producer;
Stream* RecordingStream() {
    if (producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) {
        producer->was_recording = true;
        return producer.get();
    }
    return nullptr;
}
void InvalidateRawState() {
    if (executing) executing->state_epoch.fetch_add(1, std::memory_order_relaxed);
}
[[gnu::noinline]] void BeforeDirect() {
    if (producer) {
        // Diagnostic (live timecensus, source DirectDrain): direct calls made while packets are pending.
        if (producer->Pending()) KYTY_TIME_CENSUS(DirectDrain, 0);
        producer->BeforeDirect();
    }
}

// Work calls recorded by this thread (read by the render thread's barrier dedupe, which orders
// only its own command stream): a plain increment, not a locked add on every command.
thread_local uint64_t g_work_calls    = 0;
thread_local uint64_t g_writing_calls = 0;
thread_local uint64_t g_packets       = 0;
#include "local-vulkan-recording.inc"
} // namespace

void Install() { InstallDispatch(); }
const vk::detail::DispatchLoaderDynamic& DirectDispatch() { return original; }
uint64_t WorkCalls() { return g_work_calls; }
uint64_t RecordedWork() { return g_work_calls + g_packets; }
uint64_t RecordedWrites() { return g_writing_calls + g_packets; }
void Drain() { if (producer) producer->Drain(); }
bool PacketsEnabled() {
    return producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
}
uint64_t StateEpoch() {
    return executing ? executing->state_epoch.load(std::memory_order_relaxed) : UINT64_MAX;
}
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner) {
    if (!PacketsEnabled() || !replay || segments.size() > 8) return false;
    for (const auto& segment : segments)
        if (segment.size && !segment.data) return false;
    ++g_packets;
    auto* stream = RecordingStream();
    return stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        writer.Command([owner, copied, count, replay] { replay({copied.data(), count}, original); });
    });
}
bool DeferredSubmitEnabled() {
    return producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0 &&
        kyty_local_deferred_submit_mode.load(std::memory_order_relaxed) != 0;
}
bool EnqueueDeferred(ReplayPacket replay, std::span<const Segment> segments, bool publish) {
    if (!DeferredSubmitEnabled() || !replay || segments.size() > 8) return false;
    auto* stream = RecordingStream();
    if (!stream) return false;
    const bool queued = stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        // A new command buffer may reuse a handle: drop raw-state reuse as a direct call would.
        writer.Command([copied, count, replay] { InvalidateRawState(); replay({copied.data(), count}, original); });
    });
    if (queued && publish) stream->Flush();
    return queued;
}
namespace {
std::atomic<uint64_t> g_deferred_queued {0};
std::atomic<uint64_t> g_deferred_done {0};
} // namespace
void NoteDeferredSubmitQueued() { g_deferred_queued.fetch_add(1, std::memory_order_release); }
uint64_t DeferredSubmitsQueued() { return g_deferred_queued.load(std::memory_order_acquire); }
uint64_t DeferredSubmitsDone() { return g_deferred_done.load(std::memory_order_acquire); }
void NoteDeferredSubmitDone() {
    g_deferred_done.fetch_add(1, std::memory_order_release);
    g_deferred_done.notify_all();
}
void WaitDeferredSubmits() {
    const auto target = g_deferred_queued.load(std::memory_order_acquire);
    for (auto done = g_deferred_done.load(std::memory_order_acquire); done < target;
         done = g_deferred_done.load(std::memory_order_acquire)) g_deferred_done.wait(done, std::memory_order_acquire);
}
void ReplayInline(ReplayPacket replay, std::span<const Segment> segments) {
    if (producer) producer->BeforeDirect();
    replay(segments, original);
}
ProducerScope::ProducerScope() {
    if (producer) std::abort();
    producer = std::make_unique<Stream>();
}
ProducerScope::~ProducerScope() { producer.reset(); }
} // namespace LocalVulkanRecording
