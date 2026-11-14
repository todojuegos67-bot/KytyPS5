#pragma once
// KYTY_NATIVE_TEX=1 (off by default), streaming part (agent B): guest-side hooks that tell the host when texture memory
// of the engine's VirtualMemoryGPU allocator (the one allocator of streamed textures and texture-array layers) is given
// back, so a native texture path can drop what it keeps for that range. Notification only: the engine's own code still
// runs unchanged after each hook (nothing is freed, decommitted, trimmed or moved by the emulator).
// Static notes: /home/claude/static/native-tex/B-streaming.md (01.007.000, verified instruction by instruction).
//
// Hook points (each a `call rel32` redirected to a cave that saves the argument registers, calls the host, restores
// them and jumps to the original target; every byte around them is checked first, any mismatch: not installed):
//   0xaa3346  VirtualMemoryGPU::Realloc -> ResizePages 0x801da0, op r8d = 0 commit / 2 decommit of [rdx, rdx + rcx)
//             (page-granular, absolute VA; rsi = the run's PageAllocationInfo)
//   0xaa2bc1  the same, op 1: the whole run resized to rcx (rsi = the run, or 0 for a new one): a shrink releases its tail
//   0xaa353d  VirtualMemoryGPU::Free -> ReleasePages 0x801a10 (rsi = the run: VA [+0x60], size [+0x58])
//   0x900f26  TextureArrayManager's texture -> array-slice copy (memcpy in 0x900e50): rdi = slice, rsi = source texture
//             base, rdx = size -> OnTextureMove (the only CPU move of texture bytes found; the streaming path itself never
//             moves them: runs are resized in place, B-streaming.md section 4)

#include <atomic>
#include <cstdint>

namespace Loader {
struct Program;

// The shared callback interface of the native texture path. Called on the game's thread (often a job fiber with a
// small stack); release/commit inside the engine's allocator lock, BEFORE the engine changes the mapping. Keep them short and
// non-blocking (queue the work). Defaults (demonsSoulsNativeTexStream.cpp): the original behaviour.
namespace NativeTex {
// [addr, addr + size) of texture memory is being given back (decommitted, or the whole run freed). Default: nothing.
void OnTextureRelease(uint64_t addr, uint64_t size) noexcept;
// [addr, addr + size) is being committed for texture data (a mip chunk or an array layer). Default: nothing.
void OnTextureCommit(uint64_t addr, uint64_t size) noexcept;
// size bytes of texture data are copied from src to dst (0x900f26, in place of the game's memcpy). Default: the copy
// in guest memory, told to the trackers once (PrepareHostOverwrite / FinishHostOverwrite, as MipCopy).
void OnTextureMove(uint64_t dst, uint64_t src, uint64_t size) noexcept;
// The emulator's side (graphics/host_gpu/nativeTex.h, set once at start-up): `release` is told of every release; `move`
// returning true did the move (the guest copy is not made). Unset: the defaults above.
using StreamRelease = void (*)(uint64_t addr, uint64_t size) noexcept;
using StreamMove    = bool (*)(uint64_t dst, uint64_t src, uint64_t size) noexcept;
using StreamCommit  = void (*)(uint64_t addr, uint64_t size) noexcept;
// `decommit`: op 2 (pages of a run given back inside the run; the texture keeps the run) - told to `release` when unset.
void SetStreamConsumer(StreamRelease release, StreamMove move, StreamCommit commit = nullptr,
                       StreamCommit decommit = nullptr) noexcept;
} // namespace NativeTex

namespace DemonsSoulsNativeTexStream {

// Bit per installed hook (1 = decommit/commit, 2 = run resize, 4 = run free, 8 = slice copy); 0 = off.
extern std::atomic<uint32_t> g_installed;
[[nodiscard]] inline bool On() noexcept { return g_installed.load(std::memory_order_relaxed) != 0; }

// Load time (the runtime linker's patch list), with KYTY_NATIVE_TEX=1.
void Install(Program* program);

} // namespace DemonsSoulsNativeTexStream
} // namespace Loader
