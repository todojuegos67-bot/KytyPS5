#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_GT7TITLE_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_GT7TITLE_H_

#include "loader/systemContent.h"

#include <string>

namespace Loader {

// Gran Turismo 7 (PPSA01317) runs on the Demon's Souls core with a few
// behaviours of its own port (on-demand PRX loading, etc.). The title id comes from
// sce_sys/param.json, loaded before the runtime linker runs.
inline bool IsGranTurismo7() {
	static const bool is_gt7 = [] {
		std::string title_id;
		if (!SystemContentParamSfoGetString("TITLE_ID", &title_id)) {
			return false;
		}
		return title_id == "PPSA01317";
	}();
	return is_gt7;
}

} // namespace Loader

#endif /* EMULATOR_INCLUDE_EMULATOR_LOADER_GT7TITLE_H_ */
