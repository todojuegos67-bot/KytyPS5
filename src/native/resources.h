#pragma once
// (f5) Native resource ownership: the interface (design; no implementation in this build).
//
// Today every buffer / image the GPU reads is a view of guest memory kept coherent by page protection (RAIZ): CPU
// writes fault or are swept, ranges are uploaded per draw (td3 + tq3 + uploads 2.9 ms p50 on the GPU thread, 48 MB per
// frame of uploads, ~10.800 pages re-protected per frame; grafico/f0). A natively owned resource is created and kept
// by the renderer; the game's CPU writes that feed it arrive through hooks found in Ghidra (the writer functions),
// not through faults. Its guest range is then not watched at all.
//
// A range may move from Emulated to Owned only with proof that the hooks cover every CPU writer of it:
//   - static: the writers in Ghidra (stores into the allocator's range: frame heap 0x7f9cf0, ring 0x838480, the
//     TaggedArena / GameToGPU writers) and
//   - run time: a write audit over the range (the existing write-audit / f0 PAGE_GUARD sampling) showing zero writes
//     from code that is not a hooked writer, over a multi-world soak.
// Until then the range stays Emulated and the native pass reads it through the emulator's BufferCache (still correct,
// no gain), which is why ownership is per range and not per pass.
#include <cstdint>

namespace NativeRender {

using ResourceId = uint32_t;

enum class Ownership : uint8_t {
	Emulated = 0, // guest memory + page-protection coherence (today)
	Owned    = 1, // the renderer's; CPU writes only through routed hooks; guest range unwatched
	GpuOnly  = 2, // written and read only by the GPU (render targets, GPU-written buffers): never read back unless a
	              // proven CPU reader exists (f0 measured the game's CPU reads per label)
};

struct GuestRange {
	uint64_t address = 0;
	uint64_t size    = 0;
};

struct BufferDesc {
	GuestRange range;           // where the game believes it is
	uint32_t   usage      = 0;  // vertex / index / storage / uniform / indirect bits (renderer-defined)
	Ownership  ownership  = Ownership::Emulated;
	const char* label     = nullptr; // memory-map label (demonsSoulsMemMap) or pass name
};

struct ImageDesc {
	GuestRange range;            // the game's surface (may be tiled)
	uint32_t   width = 0, height = 0, depth = 1, mips = 1, layers = 1;
	uint32_t   guest_format = 0; // the T#/CB format the game uses
	uint32_t   host_format  = 0; // VkFormat chosen once (textures.h)
	Ownership  ownership    = Ownership::Emulated;
	const char* label       = nullptr;
};

// A CPU write the game makes into an owned range, delivered by a hook at the writer (not by a fault).
struct RoutedWrite {
	uint64_t    address = 0;
	uint64_t    size    = 0;
	const void* bytes   = nullptr; // the game's source (valid during the call)
};

class ResourceTable {
public:
	virtual ~ResourceTable() = default;
	virtual ResourceId CreateBuffer(const BufferDesc& desc)  = 0;
	virtual ResourceId CreateImage(const ImageDesc& desc)    = 0;
	virtual void       Destroy(ResourceId id)                = 0;
	// The guest range of an owned resource was written by the game (a routed hook). Queued for the next upload point
	// in stream order (the pass that reads it), never later than the scope that consumes it.
	virtual void       Route(ResourceId id, const RoutedWrite& write) = 0;
	// The resource backing `address` (owned or not); NoResource if the renderer does not know it.
	[[nodiscard]] virtual ResourceId Find(uint64_t address) const = 0;
	[[nodiscard]] virtual Ownership  OwnershipOf(ResourceId id) const = 0;
	// Native handles (VkBuffer / VkImage, VkDeviceAddress) for recording.
	[[nodiscard]] virtual uint64_t   Handle(ResourceId id) const = 0;
	[[nodiscard]] virtual uint64_t   DeviceAddress(ResourceId id) const = 0;
};

} // namespace NativeRender
