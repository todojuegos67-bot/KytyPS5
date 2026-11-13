#pragma once

namespace Loader {
struct Program;
// KYTY_DS_ALLOC_PATCH=1: the game's page allocator (bpe::memory::PlatformPageAllocator::Init) is told at most
// 12 GiB + 2.4 MiB of available direct memory, so it keeps no block above its 12 GiB budget mapped for good and the
// rest of direct memory stays free for later allocations (demonsSoulsAllocClamp.cpp). Off by default.
namespace DemonsSoulsAllocClamp {
void Install(Program* program);
} // namespace DemonsSoulsAllocClamp
} // namespace Loader
