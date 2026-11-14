#pragma once
// KYTY_ENGINE_TEX_REGISTRY=1 (default off): fills graphics/host_gpu/engineTexRegistry.h from the engine's own texture
// code (Demon's Souls 01.007.000, byte-checked at load; any mismatch: nothing is patched and the registry stays off).
//   - CTexture::FTextureFactory 0x8fe750: epilogue 0x8fed9d -> cave: register the loaded CTexture (rax);
//   - CTexture::Create 0x8ffed0: epilogue 0x900b4f -> cave: register the created CTexture (rbx);
//   - CTexture release: the deferred-free calls 0x900c03 / 0x900cc9 / 0x900fa4 (rsi = [tex+0x98], rbx = tex) -> cave:
//     unregister, then the engine's own 0x903460.
// Cave at eboot+0x80c0000 (0x4000; Idle 0x8000000, SceneGuard 0x8008000, Camera 0x8020000, NativeMem 0x8040000,
// NativeTex 0x8060000, GpuPages 0x8080000..+0x8000, NativeTexStream 0x80b4000).
// Nothing the engine does changes: every hook only reads CTexture fields and returns to the original instructions.

namespace Loader {
struct Program;
namespace DemonsSoulsEngineTex {

void Install(Program* program);
// The CTexture hooks are in (for KYTY_ENGINE_TEX_REGISTRY=1 or KYTY_NATIVE_TEXTURE_PC=1).
[[nodiscard]] bool HooksInstalled() noexcept;

} // namespace DemonsSoulsEngineTex
} // namespace Loader
