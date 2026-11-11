#pragma once

namespace Loader {
struct Program;
// Demon's Souls' texture streaming loads none of the largest levels the GPU leaves out of large block-compressed textures
// (KYTY_TEXTURE_REDUCE): they no longer take the game's memory (demonsSoulsTextureMips.cpp).
namespace DemonsSoulsTextureMips {
void Install(Program* program);
} // namespace DemonsSoulsTextureMips
} // namespace Loader
