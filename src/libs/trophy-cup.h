#ifndef EMULATOR_SRC_LIBS_TROPHY_CUP_H_
#define EMULATOR_SRC_LIBS_TROPHY_CUP_H_

// (10-28) The trophy cup icon, drawn with an ImGui draw list in the grade's colour (the unlock popup and the port's
// Trofeos tab). grade: 1 platinum, 2 gold, 3 silver, 4 bronze (else grey). alpha 0-1.

#include "imgui.h"

namespace Libs::Trophy {

inline ImU32 CupColor(int grade, float alpha = 1.0f) {
	const int a = static_cast<int>(255.0f * alpha);
	switch (grade) {
		case 1: return IM_COL32(170, 200, 235, a);
		case 2: return IM_COL32(236, 196, 72, a);
		case 3: return IM_COL32(200, 205, 212, a);
		case 4: return IM_COL32(205, 127, 70, a);
		default: return IM_COL32(110, 110, 110, a);
	}
}

// A cup in the square (x, y, size): bowl, handles, stem, base, and a light shine.
inline void DrawCup(ImDrawList* dl, float x, float y, float size, ImU32 col, float alpha = 1.0f) {
	const float s = size;
	const float th = s * 0.07f;
	// handles
	dl->AddCircle({x + s * 0.22f, y + s * 0.30f}, s * 0.13f, col, 16, th);
	dl->AddCircle({x + s * 0.78f, y + s * 0.30f}, s * 0.13f, col, 16, th);
	// bowl
	dl->AddRectFilled({x + s * 0.24f, y + s * 0.12f}, {x + s * 0.76f, y + s * 0.52f}, col, s * 0.24f, ImDrawFlags_RoundCornersBottom);
	dl->AddRectFilled({x + s * 0.22f, y + s * 0.10f}, {x + s * 0.78f, y + s * 0.16f}, col, s * 0.02f);
	// stem and base
	dl->AddRectFilled({x + s * 0.45f, y + s * 0.50f}, {x + s * 0.55f, y + s * 0.70f}, col);
	dl->AddRectFilled({x + s * 0.34f, y + s * 0.70f}, {x + s * 0.66f, y + s * 0.78f}, col, s * 0.02f);
	dl->AddRectFilled({x + s * 0.26f, y + s * 0.78f}, {x + s * 0.74f, y + s * 0.88f}, col, s * 0.03f);
	// shine
	dl->AddRectFilled({x + s * 0.32f, y + s * 0.18f}, {x + s * 0.38f, y + s * 0.40f}, IM_COL32(255, 255, 255, static_cast<int>(90 * alpha)), s * 0.03f);
}

} // namespace Libs::Trophy

#endif
