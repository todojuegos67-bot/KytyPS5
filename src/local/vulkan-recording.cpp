#include "vulkan-recording.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "live-census.h"
#include "live-counters.h"
#include "live-trace-gpu.h"
#include "local-platform.h"
#include "par-audit.h"  // (A0b) diagnostic only: KYTY_AUDIT off by default
#include "par-queues.h" // (A0b) ParQueues::t_side: which translating thread encodes (0 DCB, 1 ACB)
#include "time-census.h"
#include "tej-coser-sonda.h" // (tej-coser: sonda de llamadas por hilo, KYTY_TEJ=3)
#include "tej-repro.h"       // (tej-repro: captura y reproduccion en sombra, KYTY_TEJ=3)

#include <array>
#include <atomic>
#include <chrono>
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
        ParAudit::Timed audit_encode(ParQueues::t_side < 0 ? 0 : ParQueues::t_side); // (A0b) counts; times 1 in 64
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

// A producer waiting for the worker (a full stream, a drain): the slow-frame lines' record_wait (render thread) and
// the live counter record_wait_us.
struct RecordWait {
    LiveCensus::WaitScope                       scope {LiveCensus::WaitRecord};
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    RecordWait() = default;
    RecordWait(const RecordWait&) = delete;
    RecordWait& operator=(const RecordWait&) = delete;
    ~RecordWait() {
        LiveCounters::Add(LiveCounters::RecordWaitUs,
                          static_cast<uint64_t>((std::chrono::steady_clock::now() - start).count()) / 1000u);
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
        const RecordWait waiting;
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
        auto done = completed.load(std::memory_order_acquire);
        if (sequence - done < ChunkCount) return;
        const RecordWait waiting;
        for (; sequence - done >= ChunkCount; done = completed.load(std::memory_order_acquire))
            completed.wait(done, std::memory_order_acquire);
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
            {
                ParAudit::Idle audit_idle(published.load(std::memory_order_acquire) == current); // (A0b) worker idle
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
            }
            if (stopping.load(std::memory_order_acquire)) return;
            auto& chunk = chunks[current % ChunkCount];
            for (auto offset = chunk.first; offset != End;) {
                const auto* header = reinterpret_cast<const Header*>(chunk.data.data() + offset);
                {
                    ParAudit::Timed audit_replay(2); // (A0b) the driver call of one recorded command, 1 in 64 timed
                    header->execute(chunk.data.data() + header->payload);
                }
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

thread_local std::unique_ptr<Stream> owned_producer;
// (par1, KYTY_PAR_QUEUES=1) The ACB thread records into the GPU thread's stream (AdoptProducer): one stream, one open
// command buffer, in the order the queue lock gives; only the lock's holder records, so the stream keeps one producer
// at a time.
thread_local Stream* adopted_producer = nullptr;
// This thread's stream: its own, or the one it adopted (null: none).
inline Stream* Producer() { return owned_producer ? owned_producer.get() : adopted_producer; }
Stream* RecordingStream() {
    auto* producer = Producer();
    if (producer && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0) {
        producer->was_recording = true;
        return producer;
    }
    return nullptr;
}
void InvalidateRawState() {
    if (executing) executing->state_epoch.fetch_add(1, std::memory_order_relaxed);
}
[[gnu::noinline]] void BeforeDirect() {
    if (auto* producer = Producer()) {
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
// (tej-repro) vkCmdPushDescriptorSet(KHR): VkWriteDescriptorSet lleva punteros segun el tipo; se capturan sus datos
// (cada write: binding, elemento, cuenta, tipo y por descriptor sampler/vista/layout, buffer/offset/rango o vista texel).
void TejPushDescriptors(bool capture, uint32_t id, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t set,
                        uint32_t count, const VkWriteDescriptorSet* writes) {
    using TejRepro::W;
    TejRepro::Rec r(id, TejRepro::CSet, capture, TejRepro::MSets);
    r.V(W(point), false), r.V(W(layout), true), r.V(W(set), false), r.V(W(count), false);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& w = writes[i];
        if (w.pNext != nullptr) r.bad = true;
        r.V(W(w.dstBinding), false), r.V(W(w.dstArrayElement), false), r.V(W(w.descriptorCount), false);
        r.V(W(w.descriptorType), false);
        for (uint32_t j = 0; j < w.descriptorCount; ++j) {
            switch (w.descriptorType) {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    if (w.pImageInfo == nullptr) { r.bad = true; r.V(0, false), r.V(0, false), r.V(0, false); break; }
                    r.V(W(w.pImageInfo[j].sampler), true), r.V(W(w.pImageInfo[j].imageView), true);
                    r.V(W(w.pImageInfo[j].imageLayout), false);
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    if (w.pBufferInfo == nullptr) { r.bad = true; r.V(0, false), r.V(0, false), r.V(0, false); break; }
                    // (paso A2) sin handles = tambien sin el offset: el trabajador sube a SU posicion del anillo
                    r.V(W(w.pBufferInfo[j].buffer), true), r.V(W(w.pBufferInfo[j].offset), true);
                    r.V(W(w.pBufferInfo[j].range), false);
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    if (w.pTexelBufferView == nullptr) { r.bad = true; r.V(0, false); break; }
                    r.V(W(w.pTexelBufferView[j]), true);
                    break;
                default: r.bad = true; r.V(0, false); break;
            }
        }
    }
    r.Done();
}
bool TejReplayPush(uint32_t id, const uint64_t* p, const uint64_t* e, VkCommandBuffer cb,
                   const vk::detail::DispatchLoaderDynamic& d) {
    struct Scratch {
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorImageInfo> images;
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<VkBufferView> views;
        std::vector<size_t> start;
    };
    static thread_local Scratch s;
    s.writes.clear(), s.images.clear(), s.buffers.clear(), s.views.clear(), s.start.clear();
    if (e - p < 4) return false;
    const auto point = TejRepro::From<VkPipelineBindPoint>(*p++);
    const auto layout = TejRepro::From<VkPipelineLayout>(*p++);
    const auto set = static_cast<uint32_t>(*p++);
    const auto count = static_cast<uint32_t>(*p++);
    for (uint32_t i = 0; i < count; ++i) {
        if (e - p < 4) return false;
        VkWriteDescriptorSet w {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstBinding = static_cast<uint32_t>(*p++);
        w.dstArrayElement = static_cast<uint32_t>(*p++);
        w.descriptorCount = static_cast<uint32_t>(*p++);
        w.descriptorType = TejRepro::From<VkDescriptorType>(*p++);
        size_t at = 0;
        for (uint32_t j = 0; j < w.descriptorCount; ++j) {
            switch (w.descriptorType) {
                case VK_DESCRIPTOR_TYPE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                    if (e - p < 3) return false;
                    if (j == 0) at = s.images.size();
                    s.images.push_back({TejRepro::From<VkSampler>(p[0]), TejRepro::From<VkImageView>(p[1]),
                                        TejRepro::From<VkImageLayout>(p[2])});
                    p += 3;
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                    if (e - p < 3) return false;
                    if (j == 0) at = s.buffers.size();
                    s.buffers.push_back({TejRepro::From<VkBuffer>(p[0]), p[1], p[2]});
                    p += 3;
                    break;
                case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                    if (e - p < 1) return false;
                    if (j == 0) at = s.views.size();
                    s.views.push_back(TejRepro::From<VkBufferView>(*p++));
                    break;
                default: return false;
            }
        }
        s.writes.push_back(w);
        s.start.push_back(at);
    }
    if (p != e) return false;
    // Punteros al final: los vectores ya no crecen.
    for (size_t i = 0; i < s.writes.size(); ++i) {
        auto& w = s.writes[i];
        if (w.descriptorCount == 0) continue;
        switch (w.descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: w.pBufferInfo = s.buffers.data() + s.start[i]; break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: w.pTexelBufferView = s.views.data() + s.start[i]; break;
            default: w.pImageInfo = s.images.data() + s.start[i]; break;
        }
    }
    auto fn = id == TejCoser::Id("vkCmdPushDescriptorSet") ? d.vkCmdPushDescriptorSet : d.vkCmdPushDescriptorSetKHR;
    if (fn == nullptr) return false;
    fn(cb, point, layout, set, count, s.writes.data());
    return true;
}
#include "local-vulkan-recording.inc"
} // namespace

void Install() {
    InstallDispatch();
    TejCoser::g_installed.store(true, std::memory_order_relaxed); // (tej-coser: la sonda ve los vkCmd)
}
const vk::detail::DispatchLoaderDynamic& DirectDispatch() { return original; }
void* ReplaceBindPipeline(void* entry) {
    auto* replaced = reinterpret_cast<void*>(original.vkCmdBindPipeline);
    original.vkCmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(entry);
    return replaced;
}
uint64_t WorkCalls() { return g_work_calls; }
uint64_t RecordedWork() { return g_work_calls + g_packets; }
uint64_t RecordedWrites() { return g_writing_calls + g_packets; }
void Drain() { if (auto* producer = Producer()) producer->Drain(); }
bool PacketsEnabled() {
    return Producer() && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0;
}
uint64_t StateEpoch() {
    return executing ? executing->state_epoch.load(std::memory_order_relaxed) : UINT64_MAX;
}
// (tej-repro) Replays seguros con el dispatcher de descarte (solo llaman vkCmd* por el dispatcher que reciben): los
// registran sus sitios (draws, descriptores). Otro paquete (DLSS, hwrt) = no reproducible, y nunca se llama en sombra.
namespace {
std::atomic<ReplayPacket> g_tej_allowed[4] {};
std::atomic<ReplayPacket> g_tej_owned[4] {}; // sus segmentos apuntan a un dueno: sin dueno no se capturan
thread_local bool         t_tej_noted = false; // EnqueuePacket ya lo anoto y fallo: ReplayInline no lo repite
bool TejAllowed(ReplayPacket replay) {
    for (const auto& a : g_tej_allowed)
        if (a.load(std::memory_order_relaxed) == replay) return true;
    return false;
}
void TejNote(ReplayPacket replay, std::span<const Segment> segments, const std::shared_ptr<const void>& owner) {
    // (paso A2) En el render (sin captura) un paquete que no es de draws/descriptores es trabajo propio del emulador
    // (DLSS, hwrt): lo graba el render por su camino y no es estado de draw; no marca nada (antes: draw fantasma y
    // t_bad del render pegado para siempre). En el trabajador, el paquete no cabe en la lista: no reproducible.
    if (!TejAllowed(replay) || segments.size() > 8) {
        if (TejRepro::t_cap != nullptr)
            TejRepro::BadPacket(segments.size() > 8 ? "paquete de mas de 8 segmentos" : "paquete propio del emulador (DLSS/hwrt)");
        return;
    }
    if (auto* c = TejRepro::t_cap) {
        bool needs_owner = false;
        for (const auto& a : g_tej_owned) needs_owner |= a.load(std::memory_order_relaxed) == replay;
        if (needs_owner && (!owner || TejRepro::t_own == nullptr)) {
            TejRepro::BadPacket("descriptores sin dueno (cola llena: ReplayInline)");
            return;
        }
        const size_t hdr = c->size();
        c->push_back(TejRepro::kPacket);
        c->push_back(0);
        c->push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(replay)));
        c->push_back(segments.size());
        for (const auto& segment : segments) {
            c->push_back(segment.size);
            const size_t at = c->size();
            c->resize(at + (segment.size + 7) / 8, 0);
            if (segment.size != 0) std::memcpy(c->data() + at, segment.data, segment.size);
        }
        (*c)[hdr + 1] = c->size() - hdr - 2;
        if (owner && TejRepro::t_own) TejRepro::t_own->push_back(owner);
    }
    // Seguidor de estado: lo que el paquete grabara, por las funciones de descarte (nunca el driver).
    if (TejRepro::t_acc) replay(segments, TejStubDispatch());
}
} // namespace
void TejAllow(ReplayPacket replay, bool needs_owner) {
    if (needs_owner)
        for (auto& a : g_tej_owned) {
            ReplayPacket none = nullptr;
            if (a.load(std::memory_order_relaxed) == replay || a.compare_exchange_strong(none, replay)) break;
        }
    for (auto& a : g_tej_allowed) {
        ReplayPacket none = nullptr;
        if (a.load(std::memory_order_relaxed) == replay) return;
        if (a.compare_exchange_strong(none, replay)) return;
    }
}
const vk::detail::DispatchLoaderDynamic& TejStubDispatch() {
    static const vk::detail::DispatchLoaderDynamic stub = [] {
        auto d = original;
        InstallTejStub(d);
        return d;
    }();
    return stub;
}
bool TejReplay(const uint64_t* words, size_t count, VkCommandBuffer cb, const vk::detail::DispatchLoaderDynamic& d) {
    const uint64_t* p = words;
    const uint64_t* const e = words + count;
    while (p != e) {
        if (e - p < 2) return false;
        const auto id = static_cast<uint32_t>(p[0]);
        const uint64_t n = p[1];
        const uint64_t* body = p + 2;
        if (static_cast<uint64_t>(e - body) < n) return false;
        const uint64_t* const be = body + n;
        if (id == TejRepro::kPacket) {
            if (n < 2 || body[1] > 8) return false;
            const auto replay = reinterpret_cast<ReplayPacket>(static_cast<uintptr_t>(body[0]));
            if (!TejAllowed(replay)) return false;
            std::array<Segment, 8> segs {};
            const auto nseg = static_cast<size_t>(body[1]);
            const uint64_t* q = body + 2;
            for (size_t i = 0; i < nseg; ++i) {
                if (q >= be) return false;
                const uint64_t bytes = *q++;
                if (static_cast<uint64_t>(be - q) < (bytes + 7) / 8) return false;
                segs[i] = {bytes ? static_cast<const void*>(q) : nullptr, static_cast<size_t>(bytes)};
                q += (bytes + 7) / 8;
            }
            if (q != be) return false;
            replay({segs.data(), nseg}, d);
        } else if (id == TejCoser::Id("vkCmdPushDescriptorSetKHR") || id == TejCoser::Id("vkCmdPushDescriptorSet")) {
            if (!TejReplayPush(id, body, be, cb, d)) return false;
        } else {
            const uint64_t* q = body;
            if (!TejReplayCall(id, q, be, cb, d) || q != be) return false;
        }
        p = be;
    }
    return true;
}
bool EnqueuePacket(ReplayPacket replay, std::span<const Segment> segments,
                   std::shared_ptr<const void> owner) {
    if (!PacketsEnabled() || !replay || segments.size() > 8) return false;
    for (const auto& segment : segments)
        if (segment.size && !segment.data) return false;
    const bool noted = TejRepro::t_rec; // (tej-repro)
    if (noted) TejNote(replay, segments, owner);
    ++g_packets;
    auto* stream = RecordingStream();
    const bool queued = stream->Enqueue([&](Writer& writer) {
        std::array<Segment, 8> copied {};
        for (size_t i = 0; i < segments.size(); ++i)
            copied[i] = {writer.Copy(segments[i].data, segments[i].size), segments[i].size};
        const auto count = segments.size();
        writer.Command([owner, copied, count, replay] { replay({copied.data(), count}, original); });
    });
    if (!queued && noted) t_tej_noted = true; // (tej-repro) el llamador hara ReplayInline: no anotar dos veces
    return queued;
}
bool DeferredSubmitEnabled() {
    return Producer() && kyty_local_vulkan_recording_mode.load(std::memory_order_relaxed) != 0 &&
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
    if (t_tej_noted) t_tej_noted = false;               // (tej-repro) ya anotado por EnqueuePacket
    else if (TejRepro::t_rec) TejNote(replay, segments, {});
    if (auto* producer = Producer()) producer->BeforeDirect();
    replay(segments, original);
}
void InvalidateReplayState() { InvalidateRawState(); }
ProducerScope::ProducerScope() {
    if (owned_producer || adopted_producer) std::abort();
    owned_producer = std::make_unique<Stream>();
}
ProducerScope::~ProducerScope() { owned_producer.reset(); }
void* CurrentProducer() { return owned_producer.get(); }
void AdoptProducer(void* stream) {
    if (owned_producer) std::abort();
    adopted_producer = static_cast<Stream*>(stream);
}
} // namespace LocalVulkanRecording
