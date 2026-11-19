#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EMITSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EMITSTATS_H_

#include "graphics/host_gpu/gpuPhases.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>

// gt7-emit-1119: what a draw's emission costs on the GPU thread, call by call, and
// KYTY_EMIT_FAST (off by default).
//
// Counters: each Vulkan call the recorder queues or records while the GPU thread is inside a draw
// (DrawIndex, DrawAuto), how many a draw makes and the GPU thread's microseconds for them (timed
// in the frames GpuPhases times, one in KYTY_GPU_PHASES_EVERY), the steps around them
// (translation lock waits, image layout transitions, building the descriptor writes, the set's
// push or allocation, the dynamic state, the target views, the diagnostics), the bytes and
// commands queued for the recording thread, and where a rendering was ended. An "Emision" line
// every 10 s (Log::WriteUnmuted), after the GPU phases.
//
// KYTY_EMIT_FAST:
//   0 (default) the emission as before.
//   1 a draw's descriptor set 0 goes through a descriptor update template made for its pipeline
//     (pipeline/descriptors.cpp): one flat array of descriptor infos is queued instead of the
//     writes and their arrays; the same set 0 as the command buffer holds for the same pipeline
//     layout (the draw before pushed or bound it) is not pushed or allocated again, nor the same
//     push constants; index and vertex buffer binds of what is bound already are left out. The
//     pipeline layouts are not changed, so neither are the pipelines nor the driver's cache.
//   2 verifier: the normal path records and is used; what the fast path would have queued (the
//     template's array against the writes) and left out (against the normal path's own record of
//     what it bound) is compared, and the differences are counted (and the first 16 logged).
namespace Libs::Graphics::EmitStats {

[[nodiscard]] inline int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_EMIT_FAST");
		if (value == nullptr || value[0] < '0' || value[0] > '2') {
			return 0;
		}
		return value[0] - '0';
	}();
	return mode;
}

enum Item : uint32_t {
	// Recorder calls (count, and GPU-thread time in timed frames).
	CallPushConstants,
	CallPushDescriptors,   // vkCmdPushDescriptorSetKHR
	CallPushTemplate,      // vkCmdPushDescriptorSetWithTemplateKHR (=1)
	CallUpdateBindSet,     // vkUpdateDescriptorSets + vkCmdBindDescriptorSets of a fresh set
	CallUpdateTemplateSet, // vkUpdateDescriptorSetWithTemplate + bind of a fresh set (=1)
	CallBindSets,          // vkCmdBindDescriptorSets (set 1, the bindless table, among them)
	CallBindPipeline,
	CallBindVertex,
	CallBindIndex,
	CallState, // dynamic state calls (the compare with the shadow)
	CallBarrier,
	CallBeginRendering,
	CallEndRendering,
	CallDraw,
	CallCopy, // buffer copies, updates, fills
	CallQuery,
	// Left out: the command buffer holds it already.
	SkipPipeline,
	SkipState,
	SkipVertex,
	SkipIndex,
	SkipSet,
	SkipConstants,
	// Steps (count, and time in timed frames).
	StepLockWait,    // the translation lock, when another thread held it
	StepTransit,     // Image::Transit
	StepWrites,      // CommitBindings: layouts, descriptor infos and writes, push data
	StepSet,         // CommitBindings: push constants and set 0 (templates, compares)
	StepHeapCommit,  // DescriptorHeap::Commit (a set of a pool)
	StepDynamic,     // SetGraphicsDynamicParams
	StepTargetViews, // FindRenderTarget / FindDepthTarget
	StepAnnotate,    // debug phases and diagnostic annotations
	StepQueries,     // frame graph and occlusion queries, GPU timing, profiler zone
	StepTemplate,    // a descriptor update template made (once per pipeline)
	// Counts only.
	CountTransitBarriers, // transitions that recorded a barrier
	CountPoolAllocations, // vkAllocateDescriptorSets (a batch of 32 sets, or fewer)
	CountWrites,          // descriptor writes
	CountInfos,           // their image and buffer infos
	CountStashBytes,      // bytes copied for the recording thread
	CountRecords,         // commands queued for the recording thread
	CountTemplateBytes,   // template arrays queued (=1)
	EndInCommit,          // a rendering ended in CommitBindings
	EndInRendering,       // ... in AcquireRenderTargets
	EndInEmit,            // ... in the rest of the draw's emission
	EndInPrepare,         // ... in the draw's earlier steps (textures, buffers, images)
	Count
};

namespace Detail {
// Set on the GPU thread inside a draw while GpuPhases counts (DrawScope).
inline thread_local bool t_in_draw = false;
// Written by the GPU thread only (as GpuPhases' totals).
inline std::array<uint64_t, Count> g_count {};
inline std::array<uint64_t, Count> g_cycles {};
inline std::array<uint64_t, Count> g_timed {};
} // namespace Detail

[[nodiscard]] inline bool InDraw() noexcept {
	return Detail::t_in_draw;
}

inline void Add(Item item, uint64_t n = 1) noexcept {
	if (Detail::t_in_draw) {
		Detail::g_count[item] += n;
	}
}

// Counts its block, and times it in GpuPhases' timed frames, inside a draw on the GPU thread.
class Scope {
public:
	explicit Scope(Item item) noexcept: m_item(item) {
		if (!Detail::t_in_draw) {
			return;
		}
		Detail::g_count[item]++;
		m_active = true;
		if (GpuPhases::Detail::t_timed) {
			m_start = GpuPhases::Now();
		}
	}
	~Scope() { End(); }
	void End() noexcept {
		if (m_active && m_start != 0) {
			Detail::g_cycles[m_item] += GpuPhases::Now() - m_start;
			Detail::g_timed[m_item]++;
		}
		m_active = false;
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Item     m_item;
	uint64_t m_start  = 0;
	bool     m_active = false;
};

// Around a draw (DrawIndex, DrawAuto) on the GPU thread.
class DrawScope {
public:
	DrawScope() noexcept: m_previous(Detail::t_in_draw) {
		Detail::t_in_draw = GpuPhases::Detail::t_on;
	}
	~DrawScope() { Detail::t_in_draw = m_previous; }
	DrawScope(const DrawScope&)            = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	bool m_previous;
};

// A lock taken with its wait counted and timed (StepLockWait) when another thread held it.
template <typename Mutex>
class LockScope {
public:
	explicit LockScope(Mutex& mutex): m_mutex(mutex) {
		if (m_mutex.try_lock()) {
			return;
		}
		const Scope wait(StepLockWait);
		m_mutex.lock();
	}
	~LockScope() { m_mutex.unlock(); }
	LockScope(const LockScope&)            = delete;
	LockScope& operator=(const LockScope&) = delete;

private:
	Mutex& m_mutex;
};

// A rendering ended: where, by the GPU phase step it ended in (CommandBuffer::EndRendering).
inline void NoteEndRendering() noexcept {
	if (!Detail::t_in_draw) {
		return;
	}
	using GpuPhases::Phase;
	switch (GpuPhases::Detail::t_step) {
		case Phase::DrawCommit: Add(EndInCommit); break;
		case Phase::DrawRendering: Add(EndInRendering); break;
		case Phase::DrawEmit: Add(EndInEmit); break;
		default: Add(EndInPrepare); break;
	}
}

// =2: a difference between the fast path and the normal one (`what`, with details). Counted by
// kind; the first 16 are logged.
enum class Diff : uint32_t {
	Signature, // the template's bindings, types and counts against the writes
	Template,  // the template's array against the writes' infos
	SetSkip,   // the set would be left out, but the writes differ from the last bound
	SetPush,   // the set would be pushed, but the writes equal the last bound
	Count
};
void NoteDiff(Diff kind, const std::string& what);

// GPU thread, every 10 s after the GPU phases: the "Emision" line; resets the counts.
void Report(double cycles_per_us, double seconds, uint64_t draws);

} // namespace Libs::Graphics::EmitStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EMITSTATS_H_
