#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCHECK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCHECK_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

// RenderExecutor::CommitBindings checks every draw's scalar specialization reads (evaluated before the draw) against what
// the draw's images and buffers write, and stops on an overlap. It is an assertion: it changes no draw. At Marvel's
// Wolverine's Trask outpost it cost about 2.3 % of the GPU thread (34-samp3, 2026-11-14), so it runs only with
// KYTY_CHECK_SCALAR_READS=1, for checking runs.
namespace Libs::Graphics::ScalarReadCheck {

[[nodiscard]] inline bool EnabledBy(const char* value) {
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = EnabledBy(std::getenv("KYTY_CHECK_SCALAR_READS"));
	return enabled;
}

// written: [address, size) ranges sorted by address; ends[i]: the furthest end of written[0..i]. True when a range
// overlaps [address, address + size): of the ranges that start before the read ends, the furthest end is past its start.
[[nodiscard]] inline bool OverlapsWritten(const std::vector<std::pair<uint64_t, uint64_t>>& written,
                                         const std::vector<uint64_t>& ends, uint64_t address, uint64_t size) {
	const auto before = std::lower_bound(written.begin(), written.end(), std::pair {address + size, uint64_t {0}}) -
	                    written.begin();
	return before != 0 && ends[static_cast<size_t>(before) - 1u] > address;
}

} // namespace Libs::Graphics::ScalarReadCheck

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SCALARREADCHECK_H_ */
