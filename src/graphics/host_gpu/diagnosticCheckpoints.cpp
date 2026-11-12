#include "common/logging/log.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferAddressLog.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t CHECKPOINT_RING_SIZE = 1u << 16u;

std::array<DiagnosticCheckpoint, CHECKPOINT_RING_SIZE> g_checkpoints {};
std::atomic<uint64_t>                                  g_checkpoint_sequence {0};

const char* OpName(uint32_t op) {
	switch (op) {
		case 0: return "DispatchDirect";
		case 1: return "DrawIndex";
		case 2: return "DrawIndexAuto";
		case 3: return "EopWrite";
		case 4: return "EopInterrupt";
		case 5: return "EopWriteBack";
		case 6: return "EopFlip";
		case 7: return "EopWriteBackFlip";
		case 8: return "EopOnlyFlip";
		case 9: return "DispatchIndirect";
		case 11: return "ImageUpload";
		case 12: return "ImageDownload";
		case 13: return "ImageCopy";
		case 14: return "ImageResolve";
		case 15: return "ImageCopyThroughBuffer";
		case 16: return "ImageMipCopy";
		case 17: return "ImageClear";
		case 18: return "BufferCopy";
		case 19: return "DepthCopy";
		case 20: return "InternalWork";
		default: return "Unknown";
	}
}

// CommandBufferDebugOp::InternalWork's kinds (render.h, InternalWork).
const char* InternalWorkName(uint32_t kind) {
	switch (kind) {
		case 1: return "ColorClearCheck";
		case 2: return "NanScan";
		case 3: return "Detile";
		case 4: return "Tile";
		case 5: return "ColorTransform";
		case 6: return "DepthConvert";
		case 7: return "FaultScan";
		case 8: return "BufferDownload";
		case 9: return "ImageDump";
		default: return "?";
	}
}

bool IsTransferOp(uint32_t op) {
	return op >= 11 && op <= 20;
}

void Print(const char* stage, const DiagnosticCheckpoint& checkpoint) {
	if (checkpoint.op == 20) {
		std::printf("  [%s] seq=%" PRIu64 " op=InternalWork(%s) submit=%" PRIu64
		            " dst=0x%016" PRIx64 " src=0x%016" PRIx64 " args=%u,%u\n",
		            stage, checkpoint.sequence, InternalWorkName(checkpoint.arg0),
		            checkpoint.submit_id, checkpoint.arg4, checkpoint.arg5, checkpoint.arg1,
		            checkpoint.arg2);
		LOGF("  [%s] seq=%" PRIu64 " op=InternalWork(%s) submit=%" PRIu64 " dst=0x%016" PRIx64
		     " src=0x%016" PRIx64 " args=%u,%u\n",
		     stage, checkpoint.sequence, InternalWorkName(checkpoint.arg0), checkpoint.submit_id,
		     checkpoint.arg4, checkpoint.arg5, checkpoint.arg1, checkpoint.arg2);
		return;
	}
	if (IsTransferOp(checkpoint.op)) {
		std::printf("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " dst=0x%016" PRIx64
		            " src=0x%016" PRIx64 " args=%u,%u,%u\n",
		            stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
		            checkpoint.arg4, checkpoint.arg5, checkpoint.arg0, checkpoint.arg1,
		            checkpoint.arg2);
		LOGF("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " dst=0x%016" PRIx64
		     " src=0x%016" PRIx64 " args=%u,%u,%u\n",
		     stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
		     checkpoint.arg4, checkpoint.arg5, checkpoint.arg0, checkpoint.arg1, checkpoint.arg2);
		return;
	}
	std::printf("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	            "\n",
	            stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
	            checkpoint.arg0, checkpoint.arg1, checkpoint.arg2, checkpoint.arg3,
	            checkpoint.arg4);
	LOGF("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id, checkpoint.arg0,
	     checkpoint.arg1, checkpoint.arg2, checkpoint.arg3, checkpoint.arg4);
	if ((checkpoint.op == 0 || checkpoint.op == 9) && checkpoint.vs_hash != 0) {
		std::printf("      CS=0x%016" PRIx64 "\n", checkpoint.vs_hash);
		LOGF("      CS=0x%016" PRIx64 "\n", checkpoint.vs_hash);
	} else if (checkpoint.vs_hash != 0 || checkpoint.ps_hash != 0) {
		const auto flags = checkpoint.write_flags;
		std::printf("      VS=0x%016" PRIx64 " PS=0x%016" PRIx64
		            " color_samples=%u depth_samples=%u writes=%u layers=%u colors=%u depth=%u\n",
		            checkpoint.vs_hash, checkpoint.ps_hash, checkpoint.color_samples,
		            checkpoint.depth_samples, flags & 0xffu, (flags >> 8u) & 0xffu,
		            (flags >> 16u) & 0xffu, (flags >> 24u) & 1u);
		LOGF("      VS=0x%016" PRIx64 " PS=0x%016" PRIx64
		     " color_samples=%u depth_samples=%u writes=%u layers=%u colors=%u depth=%u\n",
		     checkpoint.vs_hash, checkpoint.ps_hash, checkpoint.color_samples,
		     checkpoint.depth_samples, flags & 0xffu, (flags >> 8u) & 0xffu, (flags >> 16u) & 0xffu,
		     (flags >> 24u) & 1u);
		const auto& t = checkpoint.targets;
		if (t.area != 0) {
			char line[320];
			std::snprintf(line, sizeof(line),
			              "      targets: color=0x%016" PRIx64 " %ux%u fmt=%u depth=0x%016" PRIx64
			              " %ux%u fmt=%u area=%ux%u attachments=%u clears=%s%s%s%s\n",
			              t.color_address, t.color_extent & 0xffffu, t.color_extent >> 16u,
			              t.color_format, t.depth_address, t.depth_extent & 0xffffu,
			              t.depth_extent >> 16u, t.depth_format, t.area & 0xffffu, t.area >> 16u,
			              t.flags & 0xffu, (t.flags & (1u << 8u)) != 0 ? "color " : "",
			              (t.flags & (1u << 9u)) != 0 ? "depth " : "",
			              (t.flags & (1u << 10u)) != 0 ? "stencil " : "",
			              (t.flags & (1u << 11u)) != 0 ? "(has stencil)" : "");
			std::printf("%s", line);
			LOGF("%s", line);
			for (uint32_t i = 0; i < DiagnosticTargets::MoreColors; i++) {
				if (t.more_addresses[i] == 0 && t.more_formats[i] == 0) {
					continue;
				}
				std::snprintf(line, sizeof(line),
				              "      color slot %u: 0x%016" PRIx64 " %ux%u fmt=%u samples=%u\n",
				              t.more_slots[i] & 0xffu, t.more_addresses[i],
				              t.more_extents[i] & 0xffffu, t.more_extents[i] >> 16u,
				              t.more_formats[i], t.more_slots[i] >> 8u);
				std::printf("%s", line);
				LOGF("%s", line);
			}
		}
	}
}

// The checkpoints the GPU had started and not finished at a device loss, from the oldest
// stage's to the newest stage's, with a few before ("done") and after ("queued"): every command
// in that window is a suspect, not only the two the stages name.
void PrintInFlight(uint64_t oldest, uint64_t newest) {
	constexpr uint64_t Before = 12;
	constexpr uint64_t After  = 6;
	// Past Max, the first Max - Tail (from just before the oldest stage's, the likeliest
	// culprits) and the last Tail.
	constexpr uint64_t Max   = 600;
	constexpr uint64_t Tail  = 100;
	const uint64_t     last  = g_checkpoint_sequence.load(std::memory_order_relaxed);
	const uint64_t     first = oldest > Before ? oldest - Before : 1;
	const uint64_t     end   = std::min(newest + After, last);
	if (end < first) {
		return;
	}
	const uint64_t count = end - first + 1;
	std::printf("--- In flight at the device loss: checkpoints %" PRIu64 "..%" PRIu64
	            " (done < %" PRIu64 ", running %" PRIu64 "..%" PRIu64 ", queued after) ---\n",
	            first, end, oldest, oldest, newest);
	LOGF("--- In flight at the device loss: checkpoints %" PRIu64 "..%" PRIu64 " (done < %" PRIu64
	     ", running %" PRIu64 "..%" PRIu64 ", queued after) ---\n",
	     first, end, oldest, oldest, newest);
	for (uint64_t sequence = first; sequence <= end; sequence++) {
		if (count > Max && sequence == first + (Max - Tail)) {
			const uint64_t skipped = count - Max;
			std::printf("  (in flight: %" PRIu64 " checkpoints left out here)\n", skipped);
			LOGF("  (in flight: %" PRIu64 " checkpoints left out here)\n", skipped);
			sequence += skipped;
		}
		const auto& slot = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
		const char* tag  = sequence < oldest ? "done" : sequence <= newest ? "running" : "queued";
		if (slot.sequence != sequence) {
			std::printf("  [%s] seq=%" PRIu64 " (overwritten)\n", tag, sequence);
			LOGF("  [%s] seq=%" PRIu64 " (overwritten)\n", tag, sequence);
			continue;
		}
		Print(tag, slot);
	}
}

// Recently destroyed image views (NoteImageViewDestroyed), oldest first; a view created again
// with the same handle leaves the set. Never destroyed: images may still be released during
// static destruction.
struct DeadView {
	uint64_t   guest_address = 0;
	vk::Format format        = vk::Format::eUndefined;
	uint64_t   checkpoint    = 0; // the checkpoint sequence when it was destroyed
};
struct DeadViews {
	static constexpr size_t                   Limit = 1u << 16u;
	std::shared_mutex                         mutex;
	std::unordered_map<VkImageView, DeadView> views;
	std::deque<VkImageView>                   order;
};
DeadViews& GetDeadViews() {
	static auto* registry = new DeadViews;
	return *registry;
}
std::atomic<bool> g_any_dead_views {false};

void ReportDeadView(const char* where, const char* what, uint32_t index, VkImageView view) {
	static std::atomic<uint32_t> reported {0};
	const auto                   count = reported.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (count > 32 && (count & (count - 1u)) != 0) {
		return;
	}
	DeadView dead {};
	{
		auto&            registry = GetDeadViews();
		std::shared_lock lock(registry.mutex);
		if (const auto found = registry.views.find(view); found != registry.views.end()) {
			dead = found->second;
		}
	}
	const auto now = g_checkpoint_sequence.load(std::memory_order_relaxed);
	std::printf("DEAD image view in %s: %s %u, view %p of guest 0x%016" PRIx64
	            " format %d, destroyed %" PRIu64 " checkpoints ago; replaced with a null view (%u "
	            "so far)\n",
	            where, what, index, static_cast<void*>(view), dead.guest_address,
	            static_cast<int>(dead.format), now - std::min(now, dead.checkpoint), count);
	LOGF("DEAD image view in %s: %s %u, view %p of guest 0x%016" PRIx64
	     " format %d, destroyed %" PRIu64 " checkpoints ago; replaced with a null view (%u so "
	     "far)\n",
	     where, what, index, static_cast<void*>(view), dead.guest_address,
	     static_cast<int>(dead.format), now - std::min(now, dead.checkpoint), count);
}

} // namespace

void NoteImageViewCreated(vk::ImageView view) {
	if (view == nullptr || !g_any_dead_views.load(std::memory_order_relaxed)) {
		return;
	}
	auto&            registry = GetDeadViews();
	std::unique_lock lock(registry.mutex);
	registry.views.erase(static_cast<VkImageView>(view));
}

namespace {
std::atomic<uint64_t> g_image_view_destroy_generation {0};
std::atomic<uint64_t> g_image_set_layout_generation {0};
} // namespace

void NoteImageSetLayoutDestroyed() {
	g_image_set_layout_generation.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t ImageSetLayoutGeneration() {
	return g_image_set_layout_generation.load(std::memory_order_acquire);
}

uint64_t ImageViewDestroyGeneration() {
	return g_image_view_destroy_generation.load(std::memory_order_acquire);
}

void NoteImageViewDestroyed(vk::ImageView view, uint64_t guest_address, vk::Format format) {
	if (view == nullptr) {
		return;
	}
	g_image_view_destroy_generation.fetch_add(1, std::memory_order_acq_rel);
	const auto       key      = static_cast<VkImageView>(view);
	auto&            registry = GetDeadViews();
	std::unique_lock lock(registry.mutex);
	const auto [entry, inserted] = registry.views.insert_or_assign(
	    key,
	    DeadView {guest_address, format, g_checkpoint_sequence.load(std::memory_order_relaxed)});
	if (inserted) {
		registry.order.push_back(key);
	}
	while (registry.order.size() > DeadViews::Limit) {
		registry.views.erase(registry.order.front());
		registry.order.pop_front();
	}
	g_any_dead_views.store(true, std::memory_order_relaxed);
}

bool DiagnosticViewIsLive(vk::ImageView view) {
	if (view == nullptr || !g_any_dead_views.load(std::memory_order_relaxed)) {
		return true;
	}
	auto&            registry = GetDeadViews();
	std::shared_lock lock(registry.mutex);
	return !registry.views.contains(static_cast<VkImageView>(view));
}

void ScrubDeadRenderingViews(const vk::RenderingInfo& info, const char* where) {
	if (!g_any_dead_views.load(std::memory_order_relaxed)) {
		return;
	}
	const auto scrub = [&](const vk::RenderingAttachmentInfo* attachment, const char* what,
	                       uint32_t index) {
		if (attachment == nullptr || DiagnosticViewIsLive(attachment->imageView)) {
			return;
		}
		ReportDeadView(where, what, index, attachment->imageView);
		auto* writable             = const_cast<vk::RenderingAttachmentInfo*>(attachment);
		writable->imageView        = nullptr;
		writable->resolveImageView = nullptr;
		writable->resolveMode      = vk::ResolveModeFlagBits::eNone;
	};
	for (uint32_t i = 0; i < info.colorAttachmentCount && info.pColorAttachments != nullptr; i++) {
		scrub(info.pColorAttachments + i, "color attachment", i);
	}
	const auto depth_view =
	    info.pDepthAttachment != nullptr ? info.pDepthAttachment->imageView : vk::ImageView {};
	scrub(info.pDepthAttachment, "depth attachment", 0);
	if (info.pStencilAttachment != nullptr && depth_view &&
	    info.pStencilAttachment->imageView == depth_view && !DiagnosticViewIsLive(depth_view)) {
		// The depth attachment's view, reported with it.
		auto* stencil      = const_cast<vk::RenderingAttachmentInfo*>(info.pStencilAttachment);
		stencil->imageView = nullptr;
		stencil->resolveImageView = nullptr;
		stencil->resolveMode      = vk::ResolveModeFlagBits::eNone;
	} else {
		scrub(info.pStencilAttachment, "stencil attachment", 0);
	}
}

void ScrubDeadDescriptorViews(const vk::WriteDescriptorSet* writes, uint32_t count,
                              const char* where) {
	if (!g_any_dead_views.load(std::memory_order_relaxed) || writes == nullptr) {
		return;
	}
	for (uint32_t i = 0; i < count; i++) {
		const auto& write = writes[i];
		// Storage images only: a write through a stale view faults, a read through one samples
		// garbage (and checking every sampled image costs a lookup per texture per draw).
		if (write.pImageInfo == nullptr ||
		    write.descriptorType != vk::DescriptorType::eStorageImage) {
			continue;
		}
		for (uint32_t k = 0; k < write.descriptorCount; k++) {
			const auto& info = write.pImageInfo[k];
			if (DiagnosticViewIsLive(info.imageView)) {
				continue;
			}
			ReportDeadView(where, "descriptor binding", write.dstBinding, info.imageView);
			const_cast<vk::DescriptorImageInfo&>(info).imageView = nullptr;
		}
	}
}

DiagnosticCheckpoint* RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint) {
	const auto sequence = g_checkpoint_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      slot     = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	slot                = checkpoint;
	slot.sequence       = sequence;
	return &slot;
}

static void DumpDeviceFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		std::printf("--- Device fault: query failed: %s ---\n", vk::to_string(result).c_str());
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                       binary(static_cast<size_t>(counts.vendorBinarySize));
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	result = graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	            vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	            counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	LOGF("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	     vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	     counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	if (!binary.empty()) {
		if (auto* file = std::fopen("_device_fault.nv-gpudmp", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, binary.size(), file);
			std::fclose(file);
			std::printf("  vendor binary written to _device_fault.nv-gpudmp\n");
		}
	}
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& address = addresses[i];
		std::printf("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		            vk::to_string(address.addressType).c_str(),
		            static_cast<uint64_t>(address.reportedAddress),
		            static_cast<uint64_t>(address.addressPrecision));
		LOGF("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		     vk::to_string(address.addressType).c_str(),
		     static_cast<uint64_t>(address.reportedAddress),
		     static_cast<uint64_t>(address.addressPrecision));
		// Round 111: which buffer held the address a read or write faulted at.
		if (address.addressType == vk::DeviceFaultAddressTypeEXT::eReadInvalid ||
		    address.addressType == vk::DeviceFaultAddressTypeEXT::eWriteInvalid) {
			const auto held =
			    BufferAddressLog::Describe(static_cast<uint64_t>(address.reportedAddress),
			                               static_cast<uint64_t>(address.addressPrecision));
			std::printf("%s", held.c_str());
			LOGF("%s", held.c_str());
		}
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& vendor = vendors[i];
		std::printf("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		            vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		            static_cast<uint64_t>(vendor.vendorFaultData));
		LOGF("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		     vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		     static_cast<uint64_t>(vendor.vendorFaultData));
	}
	std::fflush(stdout);
}

void DumpDeviceLossDiagnostics(GraphicContext& graphics) {
	Aftermath::WaitForCrashDump();
	DumpDeviceFault(graphics);
	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		return;
	}
	std::vector<vk::CheckpointDataNV> data;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		data = graphics.queue.getCheckpointDataNV();
	}
	std::printf("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n",
	            data.size(), g_checkpoint_sequence.load(std::memory_order_relaxed));
	LOGF("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n", data.size(),
	     g_checkpoint_sequence.load(std::memory_order_relaxed));
	uint64_t oldest = UINT64_MAX;
	uint64_t newest = 0;
	for (const auto& entry: data) {
		// The marker is the checkpoint's sequence (DiagnosticCheckpointMarker).
		const auto sequence =
		    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(entry.pCheckpointMarker));
		const auto& checkpoint = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
		const auto  stage      = vk::to_string(entry.stage);
		const bool  current    = sequence != 0 && checkpoint.sequence == sequence;
		std::printf("  [%s] marker=%" PRIu64 "%s\n", stage.c_str(), sequence,
		            current ? "" : " (its ring slot has been reused)");
		LOGF("  [%s] marker=%" PRIu64 "%s\n", stage.c_str(), sequence,
		     current ? "" : " (its ring slot has been reused)");
		if (sequence == 0) {
			continue;
		}
		if (current) {
			Print(stage.c_str(), checkpoint);
		}
		oldest = std::min(oldest, sequence);
		newest = std::max(newest, sequence);
	}
	if (oldest != UINT64_MAX && newest >= oldest) {
		PrintInFlight(oldest, newest);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics
