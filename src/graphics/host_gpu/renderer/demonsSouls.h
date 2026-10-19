#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace Libs::Graphics {
class BufferCache;
struct ShaderComputeInputInfo;
namespace DemonsSouls {
// Demon's Souls in the regions seen (PPSA01341 01.007.000 and 01.005.000, PPSA01340 01.005.000, PPSA01342
// 01.005.000), any version: each code patch finds its code by the signature of the builds seen or checks the bytes
// of an audited build, the periodic copy checks its shader hashes, and the explicit compute boundaries are the
// engine's.
constexpr bool IsSupportedTitle(std::string_view title) {
	return title == "PPSA01340" || title == "PPSA01341" || title == "PPSA01342";
}
bool IsSupportedGame();
// Chains of dispatches with no barrier between them until the guest's own wait or cache action (these titles only).
// KYTY_COMPUTE_CHAIN=0 (live kyty_local_compute_chain_mode): barriers around every dispatch, as for other games.
bool ComputeChains();
// ComputeChains() for one program's dispatches: false for the fenced programs (a default list, or KYTY_CHAIN_FENCE=
// <hash>[;<hash>...] and KYTY_CHAIN_FENCE_RANGE=<lo>-<hi>, hex) and, with KYTY_CHAIN_ONLY=<hash>[;<hash>...], for every
// program not listed there.
bool ComputeChains(uint64_t shader_hash);
// The registers of a dispatch the linear copy shader reads: its user data, workgroup and thread-group inputs.
struct LinearCopyDispatch {
	std::span<const uint32_t> user_data;
	std::array<uint32_t, 3>   threads {};
	std::array<bool, 3>       group_id {};
	uint32_t                  thread_ids = 0, workgroup_register = 0;
	bool                      tg_size = false, thread_dimensions = false;
};
// What a dispatch of the linear copy shader copies (its checks passed), as TryLinearCopy copies it.
struct LinearCopy {
	uint64_t src = 0, dst = 0, bytes = 0;
};
// `reason` (optional): why a dispatch is not one (KYTY_SYNC_STATS' [memset] lines). `aliased` (optional): set when the
// destination's memory is mapped at other addresses too (the caller makes them coherent, else the shader runs).
std::optional<LinearCopy> LinearCopyOf(const LinearCopyDispatch& dispatch, uint32_t x, uint32_t y, uint32_t z, uint32_t mode,
                                       const char** reason = nullptr, bool* aliased = nullptr);
// A dispatch of the linear copy shader (a program of its hash), copied on the CPU.
bool TryLinearCopy(const ShaderComputeInputInfo& input, BufferCache& cache, uint32_t x, uint32_t y,
                   uint32_t z, uint32_t mode);
// The same from the registers: for a shader the program-level checks passed for (its code unchanged since).
bool TryLinearCopy(const LinearCopyDispatch& dispatch, BufferCache& cache, uint32_t x, uint32_t y, uint32_t z,
                   uint32_t mode);
} // namespace DemonsSouls
} // namespace Libs::Graphics
