#include "graphics/host_gpu/renderer/pipeline/scalarReadCheck.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ScalarReadCheckTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

int main() {
	namespace C = Libs::Graphics::ScalarReadCheck;
	// The switch: off unless set to something other than 0.
	Check(!C::EnabledBy(nullptr), "unset is off");
	Check(!C::EnabledBy(""), "empty is off");
	Check(!C::EnabledBy("0"), "0 is off");
	Check(C::EnabledBy("1"), "1 is on");

	// With it on, the search CommitBindings runs still finds an overlap: written ranges sorted, with the furthest end.
	const std::vector<std::pair<uint64_t, uint64_t>> written {{0x1000, 0x100}, {0x1080, 0x1000}, {0x5000, 0x10}};
	const std::vector<uint64_t>                      ends {0x1100, 0x2080, 0x5010};
	Check(C::OverlapsWritten(written, ends, 0x1000, 4), "a read at a range's start");
	Check(C::OverlapsWritten(written, ends, 0x2000, 0x100), "a read inside a range that started earlier");
	Check(C::OverlapsWritten(written, ends, 0x4ff0, 0x20), "a read across a range's start");
	Check(!C::OverlapsWritten(written, ends, 0x2080, 0x100), "a read at a range's end");
	Check(!C::OverlapsWritten(written, ends, 0xff0, 0x10), "a read ending at a range's start");
	Check(!C::OverlapsWritten(written, ends, 0x3000, 0x1000), "a read between ranges");
	Check(!C::OverlapsWritten({}, {}, 0x1000, 4), "nothing written");
	std::puts("ScalarReadCheckTests: ok");
	return 0;
}
