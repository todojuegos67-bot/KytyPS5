#include "graphics/host_gpu/renderer/demonsSouls.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"
#include "sync-stats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// KYTY_COMPUTE_CHAIN=0: no chains (10-14, the 8 GB laptop's GPU hangs at 2-2, a TDR in ~half of the 10-minute tours
// with chains: none in 8 tours without them, at ~25% less fps there). Fencing only the culling, list-building or atomic
// programs, a chain length cap of 8, or barriers for every V#, image and BDA read overlap did not stop them, nor did
// fencing the programs that continue chains most (10-18, an RTX 3080 Laptop at the 1-1 arena: with fe938d3a2d22b5c2
// fenced the hangs were rarer, and an RTX 5090 lost 12% of its frame rate).
extern "C" {
[[gnu::used]] volatile std::atomic<uint32_t> kyty_local_compute_chain_mode {[] {
	const char* value = std::getenv("KYTY_COMPUTE_CHAIN");
	return value != nullptr && value[0] == '0' ? 0u : 1u;
}()};
}

namespace Libs::Graphics::DemonsSouls {
bool IsSupportedGame() {
	// Called by the dispatch path after game metadata has been loaded.
	static const bool supported = [] {
		std::string title;
		return Loader::SystemContentParamSfoGetString("TITLE_ID", &title) && IsSupportedTitle(title);
	}();
	return supported;
}

bool ComputeChains() {
	static const bool reported = [] {
		if (kyty_local_compute_chain_mode.load(std::memory_order_relaxed) == 0)
			std::printf("Compute chains: off (KYTY_COMPUTE_CHAIN=0)\n");
		return true;
	}();
	(void)reported;
	return IsSupportedGame() && kyty_local_compute_chain_mode.load(std::memory_order_relaxed) != 0;
}

// The engine's cs_memset32 shader (coredata/enginesupport/shaders/_ps5/cs_memset32) as the builds seen have it:
// 01.007.000's, and 01.005.000's with the period in s12 instead of s8 (the same translated program).
constexpr std::array<uint64_t, 2> LinearCopyShaders {0xeb7456322124ecc7ULL, 0x81bb2b1b9c751ecaULL};

// Verified shader semantics: dst[i] = src[i % period], i < count. Only the
// linear subset is replaced. Other formats, workgroup shapes, overlaps and
// GPU-owned control words stay on the shader path; CopyBuffer owns coherence.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode) {
	const auto& program = *input.stage.program;
	if (!IsSupportedGame() || std::find(LinearCopyShaders.begin(), LinearCopyShaders.end(), program.shader_hash) ==
	                              LinearCopyShaders.end() ||
	    program.user_data_base != 0)
		return false;
	const LinearCopyDispatch dispatch {.user_data          = input.stage.resources.user_data,
	                                   .threads            = {input.threads_num[0], input.threads_num[1],
	                                                          input.threads_num[2]},
	                                   .group_id           = {input.group_id[0], input.group_id[1], input.group_id[2]},
	                                   .thread_ids         = static_cast<uint32_t>(input.thread_ids_num),
	                                   .workgroup_register = static_cast<uint32_t>(input.workgroup_register),
	                                   .tg_size            = input.tg_size_en,
	                                   .thread_dimensions  = input.dispatch_thread_dimensions};
	return TryLinearCopy(dispatch, cache, x, y, z, mode);
}

namespace {
// KYTY_SYNC_STATS: what the linear copy shader's dispatches became over 10 s (render thread; relaxed counters).
struct CopyStats {
	std::atomic<uint32_t> copies {0}, copies_aliased {0}, shader {0};
	std::array<std::pair<const char*, std::atomic<uint32_t>>, 16> reasons {};
	std::chrono::steady_clock::time_point start {};
};
CopyStats g_copy_stats;
void NoteCopy(std::atomic<uint32_t>& counter, const char* reason = nullptr) {
	if (!SyncStats::Enabled()) return;
	counter.fetch_add(1, std::memory_order_relaxed);
	if (reason != nullptr) {
		for (auto& [name, count]: g_copy_stats.reasons) {
			if (name == nullptr) name = reason;
			if (name == reason) {
				count.fetch_add(1, std::memory_order_relaxed);
				break;
			}
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (g_copy_stats.start == std::chrono::steady_clock::time_point {}) g_copy_stats.start = now;
	if (now - g_copy_stats.start < std::chrono::seconds(10)) return;
	g_copy_stats.start = now;
	std::printf("[memset] 10 s: %u copies (%u into aliased memory), %u left to the shader:", g_copy_stats.copies.exchange(0),
	            g_copy_stats.copies_aliased.exchange(0), g_copy_stats.shader.exchange(0));
	for (auto& [name, count]: g_copy_stats.reasons)
		if (name != nullptr) std::printf(" %s %u;", name, count.exchange(0));
	std::printf("\n");
	std::fflush(stdout);
}
} // namespace

bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode) {
	const char* reason  = nullptr;
	bool        aliased = false;
	const auto  copy    = LinearCopyOf(dispatch, x, y, z, mode, &reason, &aliased);
	if (!copy) {
		NoteCopy(g_copy_stats.shader, reason);
		return false;
	}
	// A destination mapped at other addresses too: the copy is made on the CPU only once those are coherent with it
	// (before, the shader ran and its whole destination was taken as GPU-written: the image descriptors the game keeps
	// there were then downloaded, the GPU drained, by every dispatch reading one; ~1000 such copies per 10 s at the
	// dragon's path, 2.5 s of 10 waited).
	if (aliased) {
		if (!cache.PrepareAliasesForCpuWrite(copy->dst, copy->bytes)) {
			NoteCopy(g_copy_stats.shader, "alias GPU-written");
			return false;
		}
		NoteCopy(g_copy_stats.copies_aliased);
	}
	cache.CopyBuffer(copy->dst, copy->src, copy->bytes, false, false);
	cache.ScheduleCopyFeedback(copy->dst, copy->bytes);
	NoteCopy(g_copy_stats.copies);
	return true;
}

// (Diagnostic, KYTY_LINEAR_COPY_TRACE: why a dispatch of the copy shader took the shader path.)
static void LinearCopyRefused(const char* reason, uint64_t dst = 0, uint64_t count = 0, uint64_t period = 0) {
	static const bool           trace = std::getenv("KYTY_LINEAR_COPY_TRACE") != nullptr;
	static std::atomic<uint32_t> printed {0};
	if (trace && printed.fetch_add(1) < 300)
		std::printf("LINEAR-COPY-REFUSED %s dst 0x%llx count %llu period %llu\n", reason,
		            static_cast<unsigned long long>(dst), static_cast<unsigned long long>(count),
		            static_cast<unsigned long long>(period));
}

std::optional<LinearCopy> LinearCopyOf(const LinearCopyDispatch& dispatch, uint32_t x, uint32_t y, uint32_t z, uint32_t mode,
                                       const char** reason, bool* aliased) {
	// (The reason for KYTY_SYNC_STATS' [memset] lines and KYTY_LINEAR_COPY_TRACE's LINEAR-COPY-REFUSED lines.)
	const auto refuse = [&](const char* why, uint64_t dst = 0, uint64_t count = 0, uint64_t period = 0) {
		if (reason != nullptr) *reason = why;
		LinearCopyRefused(why, dst, count, period);
		return std::nullopt;
	};
	const auto& data = dispatch.user_data;
	if (data.size() != 12 || dispatch.thread_dimensions || mode != 0x41 || dispatch.threads[0] != 64 ||
	    dispatch.threads[1] != 1 || dispatch.threads[2] != 1 || !dispatch.group_id[0] || dispatch.group_id[1] ||
	    dispatch.group_id[2] || dispatch.thread_ids != 1 || dispatch.workgroup_register != 12 || dispatch.tg_size ||
	    y != 1 || z != 1 || x == 0)
		return refuse("registers");
	ShaderBufferResource source, destination, parameters;
	std::memcpy(source.fields, data.data(), 16);
	std::memcpy(destination.fields, data.data() + 4, 16);
	std::memcpy(parameters.fields, data.data() + 8, 16);
	for (const auto* d: {&source, &destination}) {
		// Exact control bits observed for idxen format_x UInt32. Reject every
		// unanalysed format/swizzle/OOB/cache/addressing variation.
		if (d->fields[3] != 0x00014004 || (d->fields[1] & 0xffff0000u) != 0x00040000u ||
		    d->Base48() == 0 || (d->Base48() & 3) != 0)
			return refuse("buffer descriptors");
	}
	if (parameters.fields[3] != 0x0004dfac || (parameters.fields[1] & 0xffff0000u) != 0x00100000u ||
	    parameters.NumRecords() != 1 || (parameters.Base48() & 3) != 0)
		return refuse("parameter descriptor");
	std::array<uint32_t, 2> controls {};
	if (!Libs::LibKernel::Memory::TryReadGpuCleanBackingToHost(parameters.Base48(), controls.data(),
	                                                           sizeof(controls)))
		return refuse("parameters GPU-written", destination.Base48());
	const uint64_t count = controls[0], period = controls[1], bytes = count * 4;
	if (count == 0 || period < count || count > source.NumRecords() || count > destination.NumRecords())
		return refuse("counts", destination.Base48(), count, period);
	if (x != (count + 63) / 64 || bytes > 16 * 1024 * 1024) return refuse("groups or size", destination.Base48(), count, period);
	const auto src = source.Base48(), dst = destination.Base48();
	if ((src < dst + bytes && dst < src + bytes) ||
	    (parameters.Base48() < dst + bytes && dst < parameters.Base48() + 16))
		return refuse("overlap", dst, count, period);
	// (The source and the parameters are only read: an alias of theirs changes nothing. The destination's aliases are the
	// caller's to make coherent; without a caller that does, the shader runs.)
	// (Over consecutive mappings: the game's heaps are mapped a few hundred KiB at a time, and a 4 MiB copy of its
	// constants spanned ten, which took the shader path every frame: its destination GPU-written, every table
	// evaluation reading descriptors there waited for the GPU, ~2 drains a frame after the first area. A destination
	// unique over each of its consecutive mappings is not aliased.)
	if (!LibKernel::Memory::IsUniqueGuestBackingRangeAcrossMappings(dst, bytes)) {
		static const bool trace = std::getenv("KYTY_LINEAR_COPY_TRACE") != nullptr;
		if (trace)
			std::printf("LINEAR-COPY-MAPPINGS src 0x%llx end 0x%llx dst 0x%llx end 0x%llx bytes 0x%llx\n",
			            static_cast<unsigned long long>(src), static_cast<unsigned long long>(LibKernel::Memory::MappingEnd(src)),
			            static_cast<unsigned long long>(dst), static_cast<unsigned long long>(LibKernel::Memory::MappingEnd(dst)),
			            static_cast<unsigned long long>(bytes));
		if (aliased == nullptr) return refuse("aliased backing (destination)", dst, count, period);
		*aliased = true;
	} else if (aliased != nullptr) {
		*aliased = false;
	}
	if (!LibKernel::Memory::IsFullyMapped(src, bytes) || !LibKernel::Memory::IsFullyMapped(dst, bytes))
		return refuse("unmapped", dst, count, period);
	return LinearCopy {src, dst, bytes};
}
} // namespace Libs::Graphics::DemonsSouls
