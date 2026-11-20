#pragma once
// (f5b) The verification's GPU side (verify-gpu.cpp): copies of a verified pass's targets, recorded in the walk's
// command buffer where the pass's scope begins / ends. Called by the GPU thread's walk (graphicsRun.cpp) when
// NativeRender::GpuWorkPending() (native-render.h); one load per packet otherwise, and only with KYTY_NATIVE_PASSES=..:2.
namespace Libs::Graphics {
class CommandBuffer;
} // namespace Libs::Graphics

namespace NativeRender {
void RunGpuWork(Libs::Graphics::CommandBuffer& buffer);
} // namespace NativeRender
