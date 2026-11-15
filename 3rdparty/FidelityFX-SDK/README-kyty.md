# FidelityFX SDK 1.1.4 (subset)

From <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK>, tag `v1.1.4` (commit `c6efa6b`), MIT (`LICENSE.txt`).
Unchanged files, at their paths in the SDK. Kept: what the FSR 3.1 upscaler and FSR 3.1 frame generation need on Vulkan.

- `sdk/include/FidelityFX/host`: the host API headers for FSR 3.1 and the Vulkan backend.
- `sdk/include/FidelityFX/gpu`: the shared GPU headers, `fsr1` (RCAS), `spd` (the luma pyramid) and `fsr3upscaler`.
- `sdk/src/components/fsr3upscaler`: the upscaler's host code.
- `sdk/src/backends/vk`: the Vulkan backend and the FSR 3.1 GLSL passes; `sdk/src/backends/shared`: the blob dispatcher.
- `sdk/src/shared`: asserts, messages, object management.
- Frame generation (`KYTY_FSR3_FG=1`): `sdk/src/components/frameinterpolation` and `opticalflow` with their host and GPU
  headers and GLSL passes, `ffx_fsr3.h` (the frame generation config types), the two blob accessor headers, and
  `sdk/src/backends/vk/FrameInterpolationSwapchain` (the swapchain that presents and paces generated frames; Windows code,
  built on Windows only; its prebuilt SPIR-V headers are the SDK's, its `.glsl`/`.bat` sources are not kept).

Not kept: every other effect, the DX12 backend, the samples, the prebuilt DLLs, and the generated shader permutation headers
with their reflection (the SDK makes those with its binary FidelityFX_SC tool). Kyty compiles the GLSL passes with glslang at
build time and reads the bindings from the SPIR-V instead (`src/graphics/host_gpu/renderer/fsr3Blobs.cpp`).

Kyty's additions and changes:

- `kyty/ffx_vk_fi_loader.h` and `kyty/ffx_vk_fi_functions.h`, force-included into the frame interpolation swapchain's
  sources: the same for its direct Vulkan calls, through pointers `src/graphics/host_gpu/renderer/fsr3FrameGen.cpp` fills.
- `sdk/src/backends/vk/FrameInterpolationSwapchain/FrameInterpolationSwapchainVK.cpp`: two `submit` calls pass a named empty
  `SubmissionSemaphores` instead of a temporary, which clang-cl won't bind to a non-const reference (marked "Kyty:").
- `kyty/ffx_vk_loader.h`, force-included into `ffx_vk.cpp`: renames the Vulkan functions the backend calls directly, which
  `src/graphics/host_gpu/renderer/fsr3.cpp` defines through Kyty's dispatcher (Kyty links no Vulkan import library).
- `sdk/src/backends/vk/ffx_vk.cpp`, `addBarrier`: an image with `FFX_RESOURCE_USAGE_STENCILTARGET` changes layout in its
  stencil aspect too, as Vulkan requires of a depth-stencil image without `separateDepthStencilLayouts` (marked "Kyty:").
- `sdk/src/backends/vk/ffx_vk.cpp`, `CreateBackendContextVK`: loads `vkGetBufferMemoryRequirements2` by its core name. The
  breadcrumbs setup calls it on every backend context, and the KHR name is null on a device that doesn't enable
  `VK_KHR_get_memory_requirements2`, as Kyty's doesn't (marked "Kyty:").
- `kyty/ffx_portable.h`, force-included into every FidelityFX source outside MSVC: the MSVC CRT names the SDK's
  portable code calls (`_countof`, `wcscpy_s`, `strcpy_s`, `sprintf_s`) and the headers it leaves out (`<cstring>`,
  `<cwchar>`, `<locale>`, `<bit>`), so FSR 3.1 builds on Linux and macOS.
- `sdk/include/FidelityFX/host/ffx_types.h`: `FFX_SDK_DEFAULT_CONTEXT_SIZE` is twice as large outside Windows, where
  `wchar_t` is 4 bytes and the upscaler and frame interpolation contexts outgrow 128 Ki `uint32_t` (marked "Kyty:").
