#pragma once
// (f5b) The pass cut: what the frame map's scope stacks give the per-pass tools (internal to src/native).
//
// The game's profiler scopes (frame-map-probe.cpp) are walked by the GPU thread in stream order, so at any packet the
// open scopes of the queue being translated ARE the passes that packet belongs to. That is an exact command-stream
// range per pass instance, whatever blocks of the command buffer its packets lie in: the stack is per queue and lives
// across blocks and slices (f5p1 techo's ranges, registered by the game-side hook as one block, missed 18.5% of the
// p0/p1 calls for being split across two blocks; here a split instance is just counted, `instancias partidas`).
//
// Everything here runs on the GPU thread only (the probe checks the thread before calling).
#include "native/pass-registry.h"

#include <cstdint>

namespace NativeRender::Cut {

// The queue translating now (-1: none) and the innermost open pass of it (NoPass: none).
[[nodiscard]] int32_t  Queue() noexcept;
[[nodiscard]] uint32_t Innermost() noexcept;
// Whether pass `id` is open at any depth on the queue translating now.
[[nodiscard]] bool IsOpen(uint32_t id) noexcept;
// Cumulative (never reset) exclusive GPU-thread ns of pass `id`, and of all slices (in passes + outside).
[[nodiscard]] int64_t ExclNs(uint32_t id) noexcept;
[[nodiscard]] int64_t InclNs(uint32_t id) noexcept; // (open at any depth)
[[nodiscard]] int64_t SliceNs() noexcept;
[[nodiscard]] bool    OnGpuThread() noexcept;

} // namespace NativeRender::Cut

namespace NativeRender::Techo { // pass-techo.cpp
[[nodiscard]] bool Configured() noexcept;   // KYTY_NATIVE_TECHO names at least one pass (read once)
// The walk's packet `opcode` of the innermost pass `id` (NoPass: none) on queue `queue`: true = pass it over.
[[nodiscard]] bool Skip(uint32_t opcode, uint32_t id) noexcept;
void Flip(int64_t now_ns, int64_t idle_ns, uint64_t sub_cycles); // GPU thread, every flip
} // namespace NativeRender::Techo

namespace NativeRender::Verify { // pass-verify.cpp
[[nodiscard]] bool Configured() noexcept;   // KYTY_NATIVE_PASSES asks "Name:2" for at least one pass (read once)
// Scope events of the queue translating now (after the stack changed).
void Begin(uint32_t id, int32_t queue);
void End(uint32_t id, int32_t queue);
// Every packet the walk runs while Configured (state packets are what the exit state is made of).
void Packet(uint32_t opcode, const uint32_t* packet, uint32_t len_dw, const uint32_t* buffer);
void Flip();
} // namespace NativeRender::Verify
