#pragma once
// (f5) Native frame graph / pass scheduler: the interface (design; frame-graph.cpp has the scheduling logic only).
//
// What it replaces: the GPU thread walking the game's PM4 (DCB/ACB) for a pass. A converted pass is recorded from data
// the game hands its own pass function (captured by a hook whose bytes are checked, as port2/render1 do), not from
// the packets that function writes.
//
// Hybrid mode (the only mode until the whole frame is native): the game still records every pass as today. For each
// pass the registry says Emulated / Native / Verify (KYTY_NATIVE_PASSES):
//   Emulated : the PM4 between the pass's scope begin and end is walked as today.
//   Native   : at the scope begin the GPU thread runs the native pass, then skips the PM4 up to the scope end and
//              applies the register / user-data state that range leaves (ExitState; the game's next packets expect it:
//              juego1, 1d0c450/1d0c340 copy the whole user-data block to the DCB end per stream).
//   Verify   : both run; the native one writes into shadow targets (VerifyTargets), compared by verify.h; the shown
//              image is the emulated one; "[f5] DISCREPANCIA" lines per pass.
// The scope begin / end packets are the cut points: they are the game's own (0x999220 / 0x999420, proven), they are
// on the queue the pass runs on, and the frame map measures what lies between them.
//
// Pass boundaries the frame graph needs that the scopes do not give (to be proven per pass, never assumed):
//   - the labels the pass signs and waits (acb1 Grep: L0/L1/L2 offsets; frame-map.json labels_signed / labels_waited);
//   - the targets it binds (0x912080 set-render-targets, 7 slots of 0xdc bytes: G, layout to be read per pass);
//   - its GPU-written inputs (V# written by compute: port4 measured 4.334 per frame).
#include <cstdint>
#include <vector>

namespace NativeRender {

using ResourceId = uint32_t; // resources.h
using PassId     = uint32_t; // pass-registry.h id
constexpr ResourceId NoResource = 0xffffffffu;

enum class Access : uint8_t {
	Read        = 1,
	Write       = 2,
	ReadWrite   = 3,
	RenderTarget = 4, // color attachment write
	DepthTarget = 5,  // depth attachment read/write
	IndirectArgs = 6, // read as indirect draw / dispatch arguments
};

enum class QueueKind : uint8_t {
	Graphics = 0, // the game's DCB
	Compute  = 1, // the game's ACB(s)
};

struct ResourceUse {
	ResourceId resource = NoResource;
	Access     access   = Access::Read;
};

// What a converted pass is recorded from: filled by the pass's capture hook on the game thread (that frame's values
// only), consumed on the GPU thread when the stream reaches the pass's scope begin. Owned by the frame.
struct PassCapture {
	PassId   pass        = 0;
	uint64_t frame       = 0;
	uint64_t scope_begin = 0; // the begin packet's address in the command buffer (the cut point)
	uint64_t scope_end   = 0; // the end packet's address
	const void* data     = nullptr; // pass-specific struct (e.g. XPR depth: mask, data+0x40 args, R per stream)
	uint32_t data_size   = 0;
};

// A native pass. One object per pass kind, created at boot, stateless between frames except caches.
class NativePass {
public:
	virtual ~NativePass() = default;
	[[nodiscard]] virtual PassId Id() const = 0;
	[[nodiscard]] virtual QueueKind Queue() const = 0;
	// The resources this frame's instance reads and writes (from the capture); used for ordering and barriers.
	virtual void Declare(const PassCapture& capture, std::vector<ResourceUse>* uses) const = 0;
	// Record the pass. `cmd`: the GPU thread's open command buffer (VkCommandBuffer), `targets`: the images it writes
	// (shadow ones in Verify). False: cannot this frame (missing pipeline, unknown state): the emulated path runs.
	virtual bool Record(const PassCapture& capture, void* cmd, const ResourceId* targets, uint32_t target_count) = 0;
};

// Scheduling of one frame's native passes. In hybrid mode the order is the game's (the scope order in the stream);
// the graph only derives the barriers between native passes and the edges to emulated neighbours.
class FrameGraph {
public:
	struct Node {
		PassId                   pass  = 0;
		QueueKind                queue = QueueKind::Graphics;
		std::vector<ResourceUse> uses;
	};
	struct Barrier {
		uint32_t   before = 0; // node index that wrote / read
		uint32_t   after  = 0; // node index that needs it
		ResourceId resource = NoResource;
		Access     from = Access::Write, to = Access::Read;
		bool       cross_queue = false; // needs a semaphore (sync-model.h), not a pipeline barrier
	};

	void Reset();
	uint32_t Add(const Node& node); // in the game's order
	// The hazards between nodes (RAW, WAR, WAW on the same resource), each the latest earlier access only.
	[[nodiscard]] std::vector<Barrier> Hazards() const;
	[[nodiscard]] const std::vector<Node>& Nodes() const { return m_nodes; }

private:
	std::vector<Node> m_nodes;
};

} // namespace NativeRender
