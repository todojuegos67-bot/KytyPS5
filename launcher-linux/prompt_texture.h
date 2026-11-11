#ifndef KYTY_LAUNCHER_LINUX_PROMPT_TEXTURE_H
#define KYTY_LAUNCHER_LINUX_PROMPT_TEXTURE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// The key cap of a prompt icon, drawn here and written into a prompt texture (launcher-windows'
// PromptTexture.cs draws with GDI+ and Segoe UI; this one with stb_truetype and the desktop's
// bold sans font). A texture keeps its own header and metadata; only the pixels of all eight mip
// levels (128x128 down to 1x1) are replaced, through the pixel-offset map of the pack.
namespace PromptTexture {

using Buffer = std::vector<uint8_t>;

// A TrueType file to draw with, before the desktop's bold sans font (tests, and a PC whose
// fonts the program cannot find).
void SetFontFile(const std::string& path);

// The key cap at 512x512, RGBA (not premultiplied): white with a grey border on transparency,
// the label black in it (lines split at '\n').
Buffer Draw(const std::string& label);

// The key cap at width x width (a power of two up to 512), RGBA.
Buffer Scaled(const Buffer& drawn, int width);

// The eight mip levels, 128x128 down to 1x1, RGBA bytes.
std::array<Buffer, 8> Pixels(const std::string& label);

// The texture of the pack's blank template (131149 bytes) with the label's pixels.
Buffer Generate(const std::string& label);
// The same with a template of that size: another icon's texture (its metadata stays).
Buffer GenerateWithTemplate(const std::string& label, const Buffer& icon_template);

} // namespace PromptTexture

#endif
