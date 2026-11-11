#ifndef KYTY_LAUNCHER_LINUX_PROMPT_ASSETS_H
#define KYTY_LAUNCHER_LINUX_PROMPT_ASSETS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// The keyboard prompt pack compiled into the program: launcher-windows/KeyboardPrompts' files
// (the Windows launcher embeds the same ones as resources): manifest.txt, blank.ctxr.gz,
// pixel-offsets.bin.gz and the 22 <icon>.ctxr.gz.
namespace PromptAssets {

struct Asset {
	const char*    name;
	const uint8_t* data;
	size_t         size;
};

// A file of the pack by its name ("ps4_l1.ctxr.gz"); nullptr when there is none.
const Asset* Find(const std::string& name);

} // namespace PromptAssets

#endif
