#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_STEPLOOKUP_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_STEPLOOKUP_H_

#include <bit>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// A step function of a key: the value of the last step whose start is at or below the key, or the initial value
// below the first start (starts ascend). A binary search over the starts, so a shader does log2(steps) compares
// instead of one compare and select per step. The SPIR-V emitter and its test run this same search through Ops,
// which gives the arithmetic: shader instructions, or plain integers.
// Ops: Constant(u32), Add, Sub, LessEqual (unsigned), And, Select(condition, a, b), NotZero, Start(index),
// StepValue(index).
template <typename Ops, typename Value>
Value LookupStep(Ops& ops, Value key, uint32_t steps, Value initial) {
	auto found = ops.Constant(0u); // how many steps start at or below the key
	for (uint32_t span = std::bit_floor(steps); span != 0u; span >>= 1u) {
		const auto candidate = ops.Add(found, ops.Constant(span));
		const auto inside    = ops.LessEqual(candidate, ops.Constant(steps));
		const auto index     = ops.Select(inside, ops.Sub(candidate, ops.Constant(1u)), ops.Constant(0u));
		found = ops.Select(ops.And(inside, ops.LessEqual(ops.Start(index), key)), candidate, found);
	}
	const auto any  = ops.NotZero(found);
	const auto last = ops.Select(any, ops.Sub(found, ops.Constant(1u)), ops.Constant(0u));
	return ops.Select(any, ops.StepValue(last), initial);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_STEPLOOKUP_H_
