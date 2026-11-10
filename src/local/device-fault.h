#pragma once

// On device loss: what VK_EXT_device_fault reports (the faulting GPU addresses), and the buffers with device addresses
// whose ranges held them, live or destroyed (work recorded against a buffer freed under it faults on its address).
// After a loss the timeline semaphores read UINT64_MAX: everything looks complete and is freed, so the validation
// layer's "destroyed while in use" errors that follow are symptoms; the address here is the cause's.
// A hang (a TDR: 10-13 laptop walk) reports only the running shaders' instruction pointers: the report also lists the
// last submissions with the pipelines each bound (Breadcrumbs, always kept). KYTY_GPU_CHECKPOINTS=1 (diagnostic; NVIDIA
// VK_NV_device_diagnostic_checkpoints: the laptop's 1-1 ran 12.8 instead of 22 fps) puts a checkpoint after each
// pipeline bind: the report then names, per queue, the pipeline whose work the GPU started last and the one whose work
// it finished last.

#include "common/assert.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DeviceFault {

struct BufferSpan {
	uint64_t size = 0, cpu = 0, created_ns = 0, destroyed_ns = 0;
};
struct Registry {
	std::atomic<VkDevice>                  device {VK_NULL_HANDLE};
	PFN_vkGetDeviceFaultInfoEXT            get = nullptr;
	std::mutex                             mutex;
	std::multimap<uint64_t, BufferSpan>    buffers; // by device address (an address reused after a free: both)
	// Past this many the older half of the destroyed ones leave (the limit then twice what is left, at least
	// BufferHistory): a report wants the recent ones, and a world change destroys ~10000 (72 bytes each, kept by the
	// million before).
	static constexpr size_t                BufferHistory = size_t {1} << 16u;
	size_t                                 buffer_limit  = BufferHistory;
	std::atomic<int>                       report_state {0}; // 0 none, 1 being written, 2 written
};
inline Registry g_registry;

// Every pipeline bind goes through BindPipelineHook (installed at device creation): it keeps, per command buffer, the
// pipelines bound in it (a plain append on the binding thread), and NoteSubmit files them under the submission's
// timeline tick. A hang's report lists the last submissions: the first past the tick its timeline reached ("STALL
// graphics timeline: ... at N") is the one the GPU did not finish.
struct Breadcrumbs {
	PFN_vkCmdBindPipeline                                      bind = nullptr; // the bind the hook replaced
	std::mutex                                                 mutex;
	std::unordered_map<VkCommandBuffer, std::vector<uint64_t>> open;      // binds of buffers not yet submitted
	std::unordered_map<uint64_t, std::string>                  pipelines; // VkPipeline -> what it runs
	struct Submission {
		uint64_t              timeline = 0, tick = 0;
		std::vector<uint64_t> pipelines;
		// What it waited for: (semaphore, value) pairs.
		std::vector<std::pair<uint64_t, uint64_t>> waits;
	};
	std::array<Submission, 32> submitted;
	size_t                     next = 0;
};
inline Breadcrumbs                         g_breadcrumbs;
inline thread_local VkCommandBuffer        t_bind_command = VK_NULL_HANDLE;
inline thread_local std::vector<uint64_t>* t_bind_log     = nullptr;

struct Checkpoints {
	PFN_vkCmdSetCheckpointNV                     set = nullptr;
	PFN_vkGetQueueCheckpointDataNV               get = nullptr;
	std::vector<std::pair<VkQueue, const char*>> queues;
};
inline Checkpoints g_checkpoints;

[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// The device was created with the extension and its feature.
inline void Enable(VkDevice device, PFN_vkGetDeviceProcAddr get_proc) {
	g_registry.get = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(get_proc(device, "vkGetDeviceFaultInfoEXT"));
	if (g_registry.get != nullptr) g_registry.device.store(device, std::memory_order_release);
}

[[nodiscard]] inline bool CheckpointsRequested() {
	static const bool requested = std::getenv("KYTY_GPU_CHECKPOINTS") != nullptr;
	return requested;
}
[[nodiscard]] inline bool CheckpointsOn() noexcept {
	return g_checkpoints.set != nullptr;
}
[[nodiscard]] inline bool BreadcrumbsOn() noexcept {
	return g_breadcrumbs.bind != nullptr;
}
[[nodiscard]] inline uint64_t PipelineKey(VkPipeline pipeline) noexcept {
	return (uint64_t)(pipeline); // NOLINT (a pointer or an integer, by platform)
}

inline VKAPI_ATTR void VKAPI_CALL BindPipelineHook(VkCommandBuffer command, VkPipelineBindPoint point,
                                                  VkPipeline pipeline) {
	g_breadcrumbs.bind(command, point, pipeline);
	// (A buffer is recorded by one thread at a time; the map's nodes stay where they are.)
	if (command != t_bind_command) {
		std::lock_guard lock(g_breadcrumbs.mutex);
		t_bind_log     = &g_breadcrumbs.open[command];
		t_bind_command = command;
	}
	auto&      log = *t_bind_log;
	const auto key = PipelineKey(pipeline);
	if (log.empty() || log.back() != key) {
		if (log.size() >= 512) log.erase(log.begin(), log.begin() + 256);
		log.push_back(key);
	}
	if (g_checkpoints.set != nullptr) g_checkpoints.set(command, (void*)(pipeline)); // NOLINT
}

// `bind` is the entry every pipeline bind goes through: replaced by the hook. Before any command is recorded.
inline void InstallBindHook(PFN_vkCmdBindPipeline* bind) {
	if (bind == nullptr || *bind == nullptr || g_breadcrumbs.bind != nullptr) return;
	g_breadcrumbs.bind = *bind;
	*bind              = BindPipelineHook;
}

// The device was created with VK_NV_device_diagnostic_checkpoints: the hook adds a checkpoint after each bind.
inline void EnableCheckpoints(VkDevice device, PFN_vkGetDeviceProcAddr get_proc) {
	const auto set = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(get_proc(device, "vkCmdSetCheckpointNV"));
	const auto get = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(get_proc(device, "vkGetQueueCheckpointDataNV"));
	if (set == nullptr || get == nullptr) return;
	g_checkpoints.get = get;
	g_checkpoints.set = set;
	std::printf("GPU checkpoints: one after each pipeline bind (KYTY_GPU_CHECKPOINTS)\n");
}

// Before any command is recorded.
inline void NoteQueue(VkQueue queue, const char* name) {
	if (CheckpointsOn() && queue != VK_NULL_HANDLE) g_checkpoints.queues.emplace_back(queue, name);
}

inline void NotePipeline(VkPipeline pipeline, std::string what) {
	if (!BreadcrumbsOn() || pipeline == VK_NULL_HANDLE) return;
	std::lock_guard lock(g_breadcrumbs.mutex);
	g_breadcrumbs.pipelines[PipelineKey(pipeline)] = std::move(what);
}

// A pipeline destroyed (its handle's value may come back as another's).
inline void ForgetPipeline(VkPipeline pipeline) {
	if (!BreadcrumbsOn() || pipeline == VK_NULL_HANDLE) return;
	std::lock_guard lock(g_breadcrumbs.mutex);
	g_breadcrumbs.pipelines.erase(PipelineKey(pipeline));
}

// The buffers' binds, filed under the submission (`timeline` signalled to `tick`).
inline void NoteSubmit(uint64_t timeline, uint64_t tick, const VkCommandBuffer* commands, uint32_t count,
                       const VkSemaphore* wait_semaphores = nullptr, const uint64_t* wait_values = nullptr,
                       uint32_t waits = 0) {
	if (!BreadcrumbsOn()) return;
	std::lock_guard lock(g_breadcrumbs.mutex);
	auto&           entry = g_breadcrumbs.submitted[g_breadcrumbs.next++ % g_breadcrumbs.submitted.size()];
	entry.timeline        = timeline;
	entry.tick            = tick;
	entry.pipelines.clear();
	entry.waits.clear();
	for (uint32_t i = 0; i < waits; ++i) entry.waits.emplace_back((uint64_t)(wait_semaphores[i]), wait_values[i]); // NOLINT
	for (uint32_t i = 0; i < count; ++i) {
		const auto found = g_breadcrumbs.open.find(commands[i]);
		if (found == g_breadcrumbs.open.end()) continue;
		entry.pipelines.insert(entry.pipelines.end(), found->second.begin(), found->second.end());
		found->second.clear();
	}
}

// A buffer reset without a submission.
inline void NoteDiscard(VkCommandBuffer command) {
	if (!BreadcrumbsOn()) return;
	std::lock_guard lock(g_breadcrumbs.mutex);
	if (const auto found = g_breadcrumbs.open.find(command); found != g_breadcrumbs.open.end()) found->second.clear();
}

inline void NoteCreated(uint64_t device, uint64_t size, uint64_t cpu) {
	if (device == 0 || g_registry.device.load(std::memory_order_relaxed) == VK_NULL_HANDLE) return;
	std::lock_guard lock(g_registry.mutex);
	if (g_registry.buffers.size() >= g_registry.buffer_limit) {
		// The older half of the destroyed ones leave.
		std::vector<uint64_t> destroyed;
		for (const auto& [address, span]: g_registry.buffers)
			if (span.destroyed_ns != 0) destroyed.push_back(span.destroyed_ns);
		if (!destroyed.empty()) {
			const auto middle = destroyed.begin() + static_cast<std::ptrdiff_t>(destroyed.size() / 2);
			std::nth_element(destroyed.begin(), middle, destroyed.end());
			const uint64_t cutoff = *middle;
			std::erase_if(g_registry.buffers, [&](const auto& item) {
				return item.second.destroyed_ns != 0 && item.second.destroyed_ns <= cutoff;
			});
		}
		g_registry.buffer_limit = std::max(Registry::BufferHistory, g_registry.buffers.size() * 2);
	}
	g_registry.buffers.emplace(device, BufferSpan {size, cpu, NowNs(), 0});
}

inline void NoteDestroyed(uint64_t device) {
	if (device == 0 || g_registry.device.load(std::memory_order_relaxed) == VK_NULL_HANDLE) return;
	std::lock_guard lock(g_registry.mutex);
	const auto [first, last] = g_registry.buffers.equal_range(device);
	for (auto it = first; it != last; ++it)
		if (it->second.destroyed_ns == 0) {
			it->second.destroyed_ns = NowNs();
			break;
		}
}

inline const char* AddressType(VkDeviceFaultAddressTypeEXT type) {
	switch (type) {
		case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "none";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "read invalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "write invalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "execute invalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "instruction pointer unknown";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "instruction pointer invalid";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "instruction pointer fault";
		default: return "?";
	}
}

// NOLINTNEXTLINE(cert-dcl50-cpp)
inline void Append(std::string& text, const char* format, ...) {
	char    line[1024];
	va_list args;
	va_start(args, format);
	const int length = std::vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	if (length > 0) text.append(line, std::min(static_cast<size_t>(length), sizeof(line) - 1));
}

inline void AppendFault(std::string& text, VkDevice device) {
	VkDeviceFaultCountsEXT counts {};
	counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
	if (g_registry.get(device, &counts, nullptr) != VK_SUCCESS) {
		text += "Device fault: no report\n";
		return;
	}
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	VkDeviceFaultInfoEXT info {};
	info.sType         = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendors.data();
	const auto result  = g_registry.get(device, &counts, &info);
	Append(text, "Device fault (%d): %s; %u addresses, %u vendor records\n", static_cast<int>(result), info.description,
	       counts.addressInfoCount, counts.vendorInfoCount);
	addresses.resize(std::min<size_t>(addresses.size(), counts.addressInfoCount));
	// The same address from many warps (a hang lists each running one's instruction pointer): once, with its count.
	std::vector<std::pair<VkDeviceFaultAddressInfoEXT, uint32_t>> unique;
	for (const auto& address: addresses) {
		auto same = std::find_if(unique.begin(), unique.end(), [&](const auto& item) {
			return item.first.addressType == address.addressType && item.first.reportedAddress == address.reportedAddress;
		});
		if (same != unique.end()) ++same->second;
		else unique.emplace_back(address, 1);
	}
	const auto      now = NowNs();
	std::lock_guard lock(g_registry.mutex);
	for (uint32_t i = 0; i < unique.size(); ++i) {
		const auto& [address, count] = unique[i];
		// (The precision is a power of two: the fault is somewhere in the aligned range of that size.)
		const auto precision = address.addressPrecision != 0 ? address.addressPrecision : 1;
		const auto begin     = address.reportedAddress & ~(precision - 1);
		const auto end       = begin + precision;
		Append(text, "Device fault address %u: %s 0x%llx (precision 0x%llx) x%u\n", i, AddressType(address.addressType),
		       static_cast<unsigned long long>(address.reportedAddress),
		       static_cast<unsigned long long>(address.addressPrecision), count);
		uint32_t shown = 0;
		for (const auto& [device_address, span]: g_registry.buffers) {
			if (device_address >= end) break;
			if (device_address + span.size <= begin) continue;
			if (++shown > 16) break;
			Append(text, "  buffer device 0x%llx size 0x%llx guest 0x%llx: created %.1f ms ago, ",
			       static_cast<unsigned long long>(device_address), static_cast<unsigned long long>(span.size),
			       static_cast<unsigned long long>(span.cpu), static_cast<double>(now - span.created_ns) / 1e6);
			if (span.destroyed_ns != 0) Append(text, "destroyed %.1f ms ago\n", static_cast<double>(now - span.destroyed_ns) / 1e6);
			else text += "live\n";
		}
		if (shown == 0) {
			// The nearest buffers below and above.
			auto above = g_registry.buffers.lower_bound(begin);
			if (above != g_registry.buffers.begin()) {
				const auto& [below_address, below] = *std::prev(above);
				Append(text, "  nearest below: device 0x%llx size 0x%llx guest 0x%llx %s\n",
				       static_cast<unsigned long long>(below_address), static_cast<unsigned long long>(below.size),
				       static_cast<unsigned long long>(below.cpu), below.destroyed_ns != 0 ? "destroyed" : "live");
			}
			if (above != g_registry.buffers.end())
				Append(text, "  nearest above: device 0x%llx size 0x%llx guest 0x%llx %s\n",
				       static_cast<unsigned long long>(above->first), static_cast<unsigned long long>(above->second.size),
				       static_cast<unsigned long long>(above->second.cpu),
				       above->second.destroyed_ns != 0 ? "destroyed" : "live");
		}
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount && i < vendors.size(); ++i)
		Append(text, "Device fault vendor %u: %s code 0x%llx data 0x%llx\n", i, vendors[i].description,
		       static_cast<unsigned long long>(vendors[i].vendorFaultCode),
		       static_cast<unsigned long long>(vendors[i].vendorFaultData));
}

[[nodiscard]] inline const char* PipelineName(uint64_t key) {
	const auto found = g_breadcrumbs.pipelines.find(key);
	return found != g_breadcrumbs.pipelines.end() ? found->second.c_str() : "(not a known pipeline)";
}

inline void AppendSubmissions(std::string& text) {
	std::lock_guard lock(g_breadcrumbs.mutex);
	const auto      count = std::min(g_breadcrumbs.next, g_breadcrumbs.submitted.size());
	Append(text, "Last %zu submissions (each: its timeline and tick, then the pipelines it bound, first-bound order):\n", count);
	std::vector<std::pair<uint64_t, uint32_t>> distinct;
	for (size_t i = g_breadcrumbs.next - count; i < g_breadcrumbs.next; ++i) {
		const auto& entry = g_breadcrumbs.submitted[i % g_breadcrumbs.submitted.size()];
		distinct.clear();
		for (const auto key: entry.pipelines) {
			auto same = std::find_if(distinct.begin(), distinct.end(), [&](const auto& item) { return item.first == key; });
			if (same != distinct.end()) ++same->second;
			else distinct.emplace_back(key, 1);
		}
		Append(text, "  timeline 0x%llx tick %llu: %zu binds, %zu pipelines\n", static_cast<unsigned long long>(entry.timeline),
		       static_cast<unsigned long long>(entry.tick), entry.pipelines.size(), distinct.size());
		for (const auto& [key, binds]: distinct)
			Append(text, "    0x%llx %s x%u\n", static_cast<unsigned long long>(key), PipelineName(key), binds);
	}
}

inline void AppendCheckpoints(std::string& text) {
	std::lock_guard lock(g_breadcrumbs.mutex);
	for (const auto& [queue, name]: g_checkpoints.queues) {
		uint32_t count = 0;
		g_checkpoints.get(queue, &count, nullptr);
		std::vector<VkCheckpointDataNV> data(count);
		for (auto& item: data) item.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
		g_checkpoints.get(queue, &count, data.data());
		Append(text, "GPU checkpoints of the %s queue: %u\n", name, count);
		for (uint32_t i = 0; i < count && i < data.size(); ++i) {
			const auto  key   = reinterpret_cast<uint64_t>(data[i].pCheckpointMarker);
			const char* stage = data[i].stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT      ? "started"
			                    : data[i].stage == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT ? "finished"
			                                                                            : "stage";
			Append(text, "  %s (0x%x): pipeline 0x%llx %s\n", stage, static_cast<unsigned>(data[i].stage),
			       static_cast<unsigned long long>(key), PipelineName(key));
		}
	}
}

// A diagnostic's report of its own state for the work past `reached` (KYTY_DISPATCH_TIMING: which of the timed
// dispatches the GPU is in, and what that one read), called once per stall.
inline std::atomic<void (*)(uint64_t reached)> g_stall_dump {nullptr};

// The last dispatches recorded (the render thread's; a stall reads them unlocked, a torn entry is possible): their
// submission tick, program and size, for ReportStall. (The culling chain's TDR at 2-2, 10-13: its inputs were intact.)
struct DispatchNote {
	uint64_t tick = 0, hash = 0, indirect = 0;
	uint32_t x = 0, y = 0, z = 0;
};
inline std::array<DispatchNote, 4096> g_dispatches;
inline std::atomic<size_t>            g_dispatch_next {0};
inline void NoteDispatch(uint64_t tick, uint64_t hash, uint32_t x, uint32_t y, uint32_t z, uint64_t indirect) {
	if (!BreadcrumbsOn()) return;
	// (The render thread's: no locked add. A speculation's thread may note one too, which at worst overwrites an entry
	// of this diagnostic. The add and the entry's store were ~0.6% of the render thread at Latria, ~1250 dispatches a
	// frame.)
	const auto at = g_dispatch_next.load(std::memory_order_relaxed);
	g_dispatch_next.store(at + 1, std::memory_order_relaxed);
	g_dispatches[at % g_dispatches.size()] = {.tick = tick, .hash = hash, .indirect = indirect, .x = x, .y = y, .z = z};
}

// A timeline that stopped (MasterSemaphore::Wait's STALL): the submissions of it the GPU has not finished (the first
// few), each with what it waits for and where those semaphores are now (`counter`: a semaphore's current value).
// The 8 GB simulation froze with everything submitted and no thread in a binding (10-13): is the first unfinished
// submission waiting for a semaphore that never comes, or did the GPU stall?
template <typename Counter>
inline void ReportStall(uint64_t timeline, uint64_t reached, Counter&& counter) {
	if (const auto dump = g_stall_dump.load(std::memory_order_acquire); dump != nullptr) {
		static std::atomic<uint64_t> dumped {UINT64_MAX};
		if (dumped.exchange(reached) != reached) dump(reached);
	}
	if (!BreadcrumbsOn()) return;
	std::string text;
	{
		std::lock_guard lock(g_breadcrumbs.mutex);
		const auto      count = std::min(g_breadcrumbs.next, g_breadcrumbs.submitted.size());
		uint32_t        shown = 0;
		for (size_t i = g_breadcrumbs.next - count; i < g_breadcrumbs.next && shown < 4; ++i) {
			const auto& entry = g_breadcrumbs.submitted[i % g_breadcrumbs.submitted.size()];
			if (entry.timeline != timeline || entry.tick <= reached) continue;
			++shown;
			Append(text, "  STALL submission tick %llu: %zu pipelines, %zu waits\n", static_cast<unsigned long long>(entry.tick),
			       entry.pipelines.size(), entry.waits.size());
			for (const auto& [semaphore, value]: entry.waits)
				Append(text, "    waits for semaphore 0x%llx >= %llu, now %llu%s\n", static_cast<unsigned long long>(semaphore),
				       static_cast<unsigned long long>(value), static_cast<unsigned long long>(counter(semaphore)),
				       semaphore == timeline ? " (this timeline)" : "");
		}
		if (shown == 0) Append(text, "  STALL: no submission after tick %llu in the last %zu\n", static_cast<unsigned long long>(reached), count);
	}
	// The dispatches of the first unfinished submissions, by program: how many, the largest (groups).
	{
		struct Sum {
			uint64_t tick, hash;
			uint32_t count = 0, indirect = 0;
			uint64_t largest = 0;
			uint32_t x = 0, y = 0, z = 0;
		};
		std::vector<Sum> sums;
		const auto       next = g_dispatch_next.load(std::memory_order_relaxed);
		for (size_t i = next > g_dispatches.size() ? next - g_dispatches.size() : 0; i < next; ++i) {
			const auto note = g_dispatches[i % g_dispatches.size()];
			if (note.tick <= reached || note.tick > reached + 4) continue;
			auto it = std::find_if(sums.begin(), sums.end(), [&](const Sum& s) { return s.tick == note.tick && s.hash == note.hash; });
			if (it == sums.end()) it = sums.insert(sums.end(), Sum {.tick = note.tick, .hash = note.hash});
			++it->count;
			if (note.indirect != 0) ++it->indirect;
			const uint64_t groups = uint64_t {note.x} * note.y * note.z;
			if (groups >= it->largest) {
				it->largest = groups;
				it->x = note.x, it->y = note.y, it->z = note.z;
			}
		}
		for (const auto& s: sums)
			Append(text, "    STALL dispatches tick %llu cs %016llx: %u (%u indirect), largest %llu groups (%u x %u x %u)\n",
			       static_cast<unsigned long long>(s.tick), static_cast<unsigned long long>(s.hash), s.count, s.indirect,
			       static_cast<unsigned long long>(s.largest), s.x, s.y, s.z);
	}
	std::fwrite(text.data(), 1, text.size(), stdout);
	std::fflush(stdout);
}

// Once, at the first sign of the loss (any thread); the others wait until it is written.
inline void Report() {
	const auto device = g_registry.device.load(std::memory_order_acquire);
	int        state  = 0;
	if (!g_registry.report_state.compare_exchange_strong(state, 1, std::memory_order_acq_rel)) {
		// (Another thread writes it: a fatal exit here ended a 10-13 hang's report after the first of its 110 addresses.)
		for (int i = 0; i < 500 && g_registry.report_state.load(std::memory_order_acquire) == 1; ++i)
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		return;
	}
	Common::BeginExitBlockingReport();
	std::string text;
	if (device != VK_NULL_HANDLE) AppendFault(text, device);
	if (BreadcrumbsOn()) AppendSubmissions(text);
	if (CheckpointsOn()) AppendCheckpoints(text);
	std::fwrite(text.data(), 1, text.size(), stdout);
	std::fflush(stdout);
	g_registry.report_state.store(2, std::memory_order_release);
	Common::EndExitBlockingReport();
}

} // namespace DeviceFault
