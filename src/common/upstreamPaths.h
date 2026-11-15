#ifndef EMULATOR_COMMON_UPSTREAM_PATHS_H_
#define EMULATOR_COMMON_UPSTREAM_PATHS_H_

#include <cstdlib>
#include <cstring>

namespace Common {

// Upstream commits whose code is kept next to lean's own path (session 45). Lean's path runs unless
// KYTY_UPSTREAM_PATHS names the commit's short hash (comma-separated) or is "all". The commits in default_on
// run upstream's path unless KYTY_LEAN_PATHS names them (or is "all").
inline bool PathListHas(const char* variable, const char* hash) {
	const char* list = std::getenv(variable);
	return list != nullptr && (std::strcmp(list, "all") == 0 || std::strstr(list, hash) != nullptr);
}

inline bool UpstreamPathOn(const char* hash) {
	// Measured equal or better in Pro (session 65, 2026-11-13): 8810d3869 device creation from structure chains.
	static constexpr const char* default_on = "8810d3869";
	return PathListHas("KYTY_UPSTREAM_PATHS", hash) ||
	       (std::strstr(default_on, hash) != nullptr && !PathListHas("KYTY_LEAN_PATHS", hash));
}

} // namespace Common

// Read once per call site.
#define KYTY_UPSTREAM_PATH(hash) ([] { static const bool on = ::Common::UpstreamPathOn(hash); return on; }())

#endif /* EMULATOR_COMMON_UPSTREAM_PATHS_H_ */
