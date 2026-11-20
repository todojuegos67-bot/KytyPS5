#pragma once
// (f5) Native renderer for Demon's Souls PS5 01.007.000: the switch and the hooks the emulator calls.
//
// KYTY_NATIVE_RENDER (read once, at first use; unset = off):
//   unset / 0 / off : off. Every hook below is one test of a static value; nothing is allocated.
//   map             : frame map (measurement only, nothing the game sees or the GPU draws changes). The game's own GPU
//                     profiler scopes delimit its passes in the command stream; the GPU thread's time and the draws /
//                     dispatches / waits it translates are charged to the pass open on that queue. "[f5]" lines every
//                     5 s. See frame-map-probe.h.
//   hybrid          : the per-pass path (KYTY_NATIVE_PASSES selects passes). In this build the only implementation is
//                     the control pass of the verification framework (pass-verify.cpp): "Name:2" only.
//
// With map or hybrid, two more switches (read once; both need the eboot check that turns the map on):
//   KYTY_NATIVE_TECHO=<pass>,<pass>,...  per-pass ceiling (pass-techo.cpp): in rotating 5 s windows, the draws and
//                     dispatches of ONE listed pass are passed over on the GPU thread's walk (everything else of its
//                     range runs: registers, user data, labels, events, waits). OFF windows in between. Measurement
//                     only, test folder only: the image of the pass is missing in its ON windows.
//   KYTY_NATIVE_PASSES=<pass>:2,...      per-pass verification (pass-verify.cpp): the pass's exit state (the registers
//                     its range sets, the targets it rendered to) is snapshot and compared with what the pass's native
//                     implementation gives ("[f5v]" lines; DISCREPANCIA per pass). Image unchanged.
// Pass names: as the map prints them; case and spaces are ignored ("RenderShadowCascade3" = "RenderShadowCascade 3").
//
// How passes are found (proven on 01.007.000 with Ghidra; f5/mapa/frame-map.json):
//   0x999220 ScopeBegin(scope, ctx, category, name): takes a 0x50-byte record from the profiler pool, stores the
//            category at +0x40 and the name pointer at +0x48, then 0x99a080 -> 0x1d0a920 -> AgcCbReleaseMem(action
//            0x28 BOTTOM_OF_PIPE_TS, data_sel 3 = 64-bit GPU clock, address = record).
//   0x999420 ScopeEnd(scope, ctx): if the record is set, 0x99a0d0 -> 0x1d0a920 -> AgcCbReleaseMem(0x28, data_sel 3,
//            address = record + 0x20).
//   117 call sites in 96 functions, 70 distinct static names (src/native/pass-table.inc) and a few computed ones.
// The bytes of those five functions are checked at load (InstallCheck); other bytes: the frame map stays off.
#include <atomic>
#include <cstdint>

namespace NativeRender {

enum class RunMode : uint8_t {
	Off    = 0,
	FrameMap = 1,
	Hybrid = 2,
};

[[nodiscard]] RunMode ReadRunMode() noexcept; // reads KYTY_NATIVE_RENDER once and prints the mode
[[nodiscard]] inline RunMode GetRunMode() noexcept {
	static const RunMode mode = ReadRunMode();
	return mode;
}
[[nodiscard]] inline bool On() noexcept { return GetRunMode() != RunMode::Off; }

// ---- loader (loader/demonsSoulsCopy.cpp Install): eboot.bin only ----
// `base`/`size`: the module's mapping. Checks the scope functions' bytes and the static names; remembers the base.
void InstallCheckSlow(uint64_t base, uint64_t size, bool is_eboot);
inline void InstallCheck(uint64_t base, uint64_t size, bool is_eboot) {
	if (On()) InstallCheckSlow(base, size, is_eboot);
}

// ---- the game's threads (libs/agc.cpp AgcCbReleaseMem) ----
// A RELEASE_MEM packet `packet` just written by the game; `address` its destination. Only a GPU clock write of event
// 0x28 can be a scope's: the names the record holds (+0x48 for a begin, +0x28 from the address for an end) are read
// here, on the game's thread that wrote them, and kept in a side table by packet address.
void NoteReleaseMemSlow(const uint32_t* packet, uint64_t address, uint32_t action, uint32_t data_sel);
inline void NoteReleaseMem(const uint32_t* packet, uint64_t address, uint32_t action, uint32_t data_sel) {
	if (On() && data_sel == 3u && (action & 0x3fu) == 0x28u) NoteReleaseMemSlow(packet, address, action, data_sel);
}

// ---- the GPU thread ----
// CpOpReleaseMem, a 64-bit clock write (data_sel 3): `packet` the packet's first dword (its header).
void TimestampPacketSlow(const uint32_t* packet, uint64_t address);
inline void TimestampPacket(const uint32_t* packet, uint64_t address) {
	if (On()) TimestampPacketSlow(packet, address);
}
// GuestGpu::ThreadRun around Process: a slice of queue `queue_id` (0 graphics, others compute).
void SliceBeginSlow(uint32_t queue_id);
void SliceEndSlow();
inline void SliceBegin(uint32_t queue_id) {
	if (On()) SliceBeginSlow(queue_id);
}
inline void SliceEnd() {
	if (On()) SliceEndSlow();
}
// CommandProcessor::Flip: the game's frame ends. `idle_ns`: LiveControl::g_render_idle_ns (the GPU thread's waits for
// guest work since start); `sub_cycles`: LiveCensus::g_kind_cycles[Submission] (TSC cycles of its submissions; 0
// without KYTY_RENDER_STATS). Both only read (the per-pass ceiling's windows).
void FlipSlow(int64_t idle_ns, uint64_t sub_cycles);
inline void Flip(int64_t idle_ns, uint64_t sub_cycles) {
	if (On()) FlipSlow(idle_ns, sub_cycles);
}

// ---- the GPU thread's PM4 walk (graphicsRun.cpp, before a packet's handler; never in a state pass or a speculation) ----
// On only with KYTY_NATIVE_TECHO or KYTY_NATIVE_PASSES (and the eboot checked): one static load per packet otherwise.
[[nodiscard]] bool ReadWalkOn() noexcept;
[[nodiscard]] inline bool WalkOn() noexcept {
	static const bool on = On() && ReadWalkOn();
	return on;
}
// The packet `packet` (opcode `opcode`, `remaining_dw` dwords left in the buffer `buffer` of `buffer_dw` dwords) is
// about to run. Returns the dwords to pass over (a draw / dispatch of the pass skipped in a ceiling window), else 0.
[[nodiscard]] uint32_t WalkPacketSlow(uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw, const uint32_t* buffer,
                                      uint32_t buffer_dw);
// The verification's GPU work (copies of a verified pass's targets at its scope begin / end) is due: run it with the
// current command buffer (native/verify-gpu.h RunGpuWork) before this packet.
[[nodiscard]] bool GpuWorkPending() noexcept;

// ---- the renderer (GPU thread): a render target a draw is about to write ----
// renderDraw.cpp AcquireRenderTargets (normal path) and the table path's draw (table-xpr.inc). `index`/`generation`:
// the TextureCache ImageId; level / layer: the view's. Only noted while a verified pass is open (one load otherwise).
inline std::atomic<bool> g_note_targets {false}; // (set by the GPU thread while a verified pass instance is open)
// (pt1) set while BB_PATHTRACING has the GBuffer / ResolveLighting scope open (local/pathtrace.cpp)
inline std::atomic<bool> g_pt_targets {false};
void NoteTargetSlow(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth);
void NotePtTarget(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth); // (pt1, pathtrace.cpp)
[[nodiscard]] inline bool NoteTargetsWanted() noexcept {
	return g_note_targets.load(std::memory_order_relaxed) || g_pt_targets.load(std::memory_order_relaxed);
}
inline void NoteTarget(uint32_t index, uint32_t generation, uint32_t level, uint32_t layer, bool depth) {
	if (g_note_targets.load(std::memory_order_relaxed)) NoteTargetSlow(index, generation, level, layer, depth);
	if (g_pt_targets.load(std::memory_order_relaxed)) NotePtTarget(index, generation, level, layer, depth);
}

} // namespace NativeRender
