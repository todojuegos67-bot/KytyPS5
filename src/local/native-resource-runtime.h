#pragma once
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/passes/LinearSrt.h"
#include "native-preparation-scratch.h"

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace LocalNativeResource {

// The descriptors and flattened SRT a linear plan evaluates to, in reusable storage
// (NativePreparationScratch; EvaluateLinearSrtOriginal in SrtWalker.cpp).
struct OutputWorkspace {
    std::vector<DescriptorValue> evaluated;
    std::vector<uint32_t> flattened;
};
} // namespace LocalNativeResource
} // namespace Libs::Graphics::ShaderRecompiler::IR
