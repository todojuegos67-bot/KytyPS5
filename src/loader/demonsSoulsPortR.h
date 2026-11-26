#pragma once
// (port-r) KYTY_PORT_R=1|2 (0 by default): Demon's Souls' common geometry list (eboot+0xab57d0) on the host
// (local/port-r.h).

namespace Loader {
struct Program;
namespace DemonsSoulsPortR {
// eboot.bin of 01.007.000 only (the function's bytes compared first); anything else: nothing patched, said once.
void Install(Program* program);
} // namespace DemonsSoulsPortR
} // namespace Loader
