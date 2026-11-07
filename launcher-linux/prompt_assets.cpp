#include "prompt_assets.h"

#include <cstring>

// The files are linked in as they are (the assembler's .incbin), from the folder the build names:
// no generated source, no tool at build time. KYTY_PROMPT_ASSETS_DIR is that folder.
#ifndef KYTY_PROMPT_ASSETS_DIR
#error "KYTY_PROMPT_ASSETS_DIR must name launcher-windows/KeyboardPrompts"
#endif

#define KYTY_PROMPT_ASSET(id, file)                                             \
	__asm__(".section .rodata\n"                                                \
	        ".balign 16\n"                                                      \
	        ".global kyty_prompt_" #id "_begin\n"                               \
	        "kyty_prompt_" #id "_begin:\n"                                      \
	        ".incbin \"" KYTY_PROMPT_ASSETS_DIR "/" file "\"\n"                  \
	        ".global kyty_prompt_" #id "_end\n"                                 \
	        "kyty_prompt_" #id "_end:\n"                                        \
	        ".byte 0\n"                                                         \
	        ".previous\n");                                                     \
	extern "C" const uint8_t kyty_prompt_##id##_begin[];                        \
	extern "C" const uint8_t kyty_prompt_##id##_end[]

KYTY_PROMPT_ASSET(a0, "manifest.txt");
KYTY_PROMPT_ASSET(a1, "blank.ctxr.gz");
KYTY_PROMPT_ASSET(a2, "pixel-offsets.bin.gz");
KYTY_PROMPT_ASSET(a3, "ps4_button_circle.ctxr.gz");
KYTY_PROMPT_ASSET(a4, "ps4_button_cross.ctxr.gz");
KYTY_PROMPT_ASSET(a5, "ps4_button_square.ctxr.gz");
KYTY_PROMPT_ASSET(a6, "ps4_button_triangle.ctxr.gz");
KYTY_PROMPT_ASSET(a7, "ps4_dpad_d.ctxr.gz");
KYTY_PROMPT_ASSET(a8, "ps4_dpad_l.ctxr.gz");
KYTY_PROMPT_ASSET(a9, "ps4_dpad_lr.ctxr.gz");
KYTY_PROMPT_ASSET(a10, "ps4_dpad_r.ctxr.gz");
KYTY_PROMPT_ASSET(a11, "ps4_dpad_u.ctxr.gz");
KYTY_PROMPT_ASSET(a12, "ps4_dpad_ud.ctxr.gz");
KYTY_PROMPT_ASSET(a13, "ps4_l1.ctxr.gz");
KYTY_PROMPT_ASSET(a14, "ps4_l2.ctxr.gz");
KYTY_PROMPT_ASSET(a15, "ps4_l3.ctxr.gz");
KYTY_PROMPT_ASSET(a16, "ps4_options.ctxr.gz");
KYTY_PROMPT_ASSET(a17, "ps4_r1.ctxr.gz");
KYTY_PROMPT_ASSET(a18, "ps4_r2.ctxr.gz");
KYTY_PROMPT_ASSET(a19, "ps4_r3.ctxr.gz");
KYTY_PROMPT_ASSET(a20, "ps4_stick_l.ctxr.gz");
KYTY_PROMPT_ASSET(a21, "ps4_stick_r.ctxr.gz");
KYTY_PROMPT_ASSET(a22, "ps4_touchpad.ctxr.gz");
KYTY_PROMPT_ASSET(a23, "ps4_touchpad_left.ctxr.gz");
KYTY_PROMPT_ASSET(a24, "ps4_touchpad_right.ctxr.gz");

namespace PromptAssets {

namespace {

#define KYTY_PROMPT_ENTRY(id, file) \
	{file, kyty_prompt_##id##_begin, static_cast<size_t>(kyty_prompt_##id##_end - kyty_prompt_##id##_begin)}

const Asset Assets[] = {
	KYTY_PROMPT_ENTRY(a0, "manifest.txt"),
	KYTY_PROMPT_ENTRY(a1, "blank.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a2, "pixel-offsets.bin.gz"),
	KYTY_PROMPT_ENTRY(a3, "ps4_button_circle.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a4, "ps4_button_cross.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a5, "ps4_button_square.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a6, "ps4_button_triangle.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a7, "ps4_dpad_d.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a8, "ps4_dpad_l.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a9, "ps4_dpad_lr.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a10, "ps4_dpad_r.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a11, "ps4_dpad_u.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a12, "ps4_dpad_ud.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a13, "ps4_l1.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a14, "ps4_l2.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a15, "ps4_l3.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a16, "ps4_options.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a17, "ps4_r1.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a18, "ps4_r2.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a19, "ps4_r3.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a20, "ps4_stick_l.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a21, "ps4_stick_r.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a22, "ps4_touchpad.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a23, "ps4_touchpad_left.ctxr.gz"),
	KYTY_PROMPT_ENTRY(a24, "ps4_touchpad_right.ctxr.gz"),
};

} // namespace

const Asset* Find(const std::string& name) {
	for (const auto& asset: Assets)
		if (name == asset.name) return &asset;
	return nullptr;
}

} // namespace PromptAssets
