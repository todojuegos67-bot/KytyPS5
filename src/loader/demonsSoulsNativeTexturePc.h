#pragma once
// KYTY_NATIVE_TEXTURE_PC=1 (default off): Demon's Souls 01.007.000 engine textures created/filled the PC way
// (graphics/host_gpu/nativeTexturePc.h). This installer wires the byte-checked engine hooks to that registry:
//   - CTexture::Create / FTextureFactory epilogues + deferred-free calls: loader/demonsSoulsEngineTex.cpp (cave 0x80c0000)
//   - FTextureFactory Read 0x8fed17 / pool layer Read 0x8fef15: loader/demonsSoulsNativeTex.cpp (cave 0x8060000)
//   - VirtualMemoryGPU decommit/shrink/free + slice copy 0x900f26: loader/demonsSoulsNativeTexStream.cpp (cave 0x80b4000)
//   - stream-in mip memcpy 0x901459 (r14 = the CTexture): here, cave eboot+0x80d8000 (0x1000).
// It refuses (nothing of it is used) when KYTY_NATIVE_TEX / KYTY_ENGINE_GPU_NATIVE (the emulator-side ring path, same
// hook consumers) or KYTY_NATIVE_MEM=3 (owns 0x901459) is on, or when any of the hooks above is not installed.

namespace Loader {
struct Program;
namespace DemonsSoulsNativeTexturePc {

// After DemonsSoulsEngineTex / NativeTex / DemonsSoulsNativeTexStream::Install (the runtime linker's patch list).
void Install(Program* program);

} // namespace DemonsSoulsNativeTexturePc
} // namespace Loader
