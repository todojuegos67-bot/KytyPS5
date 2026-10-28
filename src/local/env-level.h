#pragma once
// (ds-fx-all-1027) An effect's quality env var: 0..3, or No / Bajo / Medio / Alto (also off / low / medium / high), any
// case; as hwrt-gi.cpp's EnvLevel. Anything else is 0 (off).
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::FrameGen {
inline int ParseEnvLevel(const char* v) {
	if (v == nullptr || *v == '\0') return 0;
	if (*v >= '0' && *v <= '9') return std::clamp(std::atoi(v), 0, 3);
	char w[8] {};
	for (int i = 0; i < 7 && v[i] != '\0'; ++i) w[i] = static_cast<char>(v[i] | 0x20);
	if (std::strncmp(w, "bajo", 4) == 0 || std::strncmp(w, "low", 3) == 0) return 1;
	if (std::strncmp(w, "medi", 4) == 0) return 2;
	if (std::strncmp(w, "alto", 4) == 0 || std::strncmp(w, "high", 4) == 0) return 3;
	return 0;
}
} // namespace Libs::Graphics::FrameGen
