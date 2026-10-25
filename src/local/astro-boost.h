#ifndef EMULATOR_SRC_LOCAL_ASTRO_BOOST_H_
#define EMULATOR_SRC_LOCAL_ASTRO_BOOST_H_
// astro-perf-1025: brunoKytyPs5's perf/astrobot-rx6600xt-boost changes that trade correctness for speed, each
// behind KYTY_ASTRO_BOOST (default off: this tree's behaviour). KYTY_ASTRO_BOOST=1 (or "all") turns every one on;
// a comma list turns on only the named ones, for A/B:
//   clock    no BDA epoch advance after a reference-clock copy (later draws may see the old clock bytes)
//   flush    an incomplete (waiting) submission is not flushed at once (batched like a complete one)
//   patches  GPU indirect draws even when the draw's offsets go into user SGPRs (those SGPRs keep stale values)
//   count    an indirect count read by the GPU even when the tracker does not know a GPU write of it
//   writes   512 pending writes before a forced rendering break (64 before)
//   fastmath Performance shaders: MAD as one FMA, no NoContraction on FMA (pass-to-pass depth invariance lost)
//   earlyz   Performance shaders: early fragment tests forced on every pixel shader without kill/depth/mask export
// The safe parts of that branch are on unconditionally: multiDrawIndirect, the maxDrawIndirectCount clamp, the
// null-argument guard, required vertex-stage subgroup sizes where the driver supports them (AMD; NVIDIA has a
// fixed 32), and the larger GC thresholds on GPUs with less than 12 GiB.
#include <cstdlib>
#include <string>
#include <string_view>

namespace Kyty::AstroBoost {

[[nodiscard]] inline bool On(std::string_view item) {
	static const std::string value = [] {
		const char* env = std::getenv("KYTY_ASTRO_BOOST");
		return std::string(env != nullptr ? env : "");
	}();
	if (value.empty() || value == "0") {
		return false;
	}
	if (value == "1" || value == "all") {
		return true;
	}
	size_t start = 0;
	while (start <= value.size()) {
		const auto end   = value.find(',', start);
		const auto token = std::string_view(value).substr(
		    start, end == std::string::npos ? std::string::npos : end - start);
		if (token == item) {
			return true;
		}
		if (end == std::string::npos) {
			break;
		}
		start = end + 1;
	}
	return false;
}

} // namespace Kyty::AstroBoost

#endif // EMULATOR_SRC_LOCAL_ASTRO_BOOST_H_
