#include "prompt_texture.h"

#include "bytes.h"
#include "keyboard_layout.h"
#include "prompt_assets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <unistd.h>

// (stb_truetype's functions are static here, as in Dear ImGui's copy: nothing clashes at link
// time; the ones not used are not a warning.)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#pragma GCC diagnostic pop

namespace PromptTexture {

namespace {

constexpr int    CanvasSize      = 512;
constexpr int    TextureBytes    = 131149;
constexpr int    PixelCount      = 21845; // 128*128 + 64*64 + ... + 1
constexpr size_t FirstPixelByte  = 65;
constexpr size_t PixelAreaBytes  = 131072;
constexpr int    Scale           = 4;     // the drawing is in the 128 units of the texture

Buffer Resource(const char* name) {
	const auto* asset = PromptAssets::Find(name);
	if (asset == nullptr) throw PromptError(std::string("Missing prompt texture resource: ") + name);
	std::string error;
	auto        bytes = Bytes::Gunzip(asset->data, asset->size, &error);
	if (!error.empty()) throw PromptError(std::string("Damaged prompt texture resource: ") + name);
	return bytes;
}

const Buffer& Blank() {
	static const Buffer blank = Resource("blank.ctxr.gz");
	return blank;
}

const std::vector<int32_t>& Offsets() {
	static const std::vector<int32_t> offsets = [] {
		const auto data = Resource("pixel-offsets.bin.gz");
		if (data.size() != static_cast<size_t>(PixelCount) * 4 || Blank().size() != TextureBytes)
			throw PromptError("Invalid prompt texture layout");
		std::vector<int32_t> result(PixelCount);
		std::memcpy(result.data(), data.data(), data.size()); // (little-endian, as on Windows)
		std::set<int32_t> seen(result.begin(), result.end());
		if (seen.size() != result.size()) throw PromptError("Invalid prompt pixel offset");
		for (const int32_t at: result)
			if (at < static_cast<int32_t>(FirstPixelByte) || at + 4 > 131137)
				throw PromptError("Invalid prompt pixel offset");
		return result;
	}();
	return offsets;
}

// ---------------------------------------------------------------- fonts

struct Font {
	std::string        path;
	std::vector<uint8_t> data;
	stbtt_fontinfo     info {};
	bool Has(uint32_t codepoint) const { return stbtt_FindGlyphIndex(&info, static_cast<int>(codepoint)) != 0; }
};

std::string g_font_file;
std::vector<std::unique_ptr<Font>> g_fonts;
bool g_fonts_loaded = false;

std::vector<std::string> RunLines(const char* command) {
	std::vector<std::string> lines;
	// (A fixed command: nothing of the user's goes into it.)
	FILE* pipe = popen(command, "r"); // NOLINT
	if (pipe == nullptr) return lines;
	std::string line;
	for (int c; (c = std::fgetc(pipe)) != EOF;) {
		if (c == '\n') {
			if (!line.empty()) lines.push_back(line);
			line.clear();
		} else {
			line += static_cast<char>(c);
		}
		if (line.size() > 4096) break;
	}
	if (!line.empty()) lines.push_back(line);
	pclose(pipe);
	return lines;
}

std::unique_ptr<Font> LoadFont(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) return nullptr;
	auto font = std::make_unique<Font>();
	font->path = path;
	std::ostringstream text;
	text << file.rdbuf();
	const auto content = text.str();
	if (content.size() < 12 || content.size() > (64u << 20u)) return nullptr;
	font->data.assign(content.begin(), content.end());
	const int offset = stbtt_GetFontOffsetForIndex(font->data.data(), 0);
	if (offset < 0 || stbtt_InitFont(&font->info, font->data.data(), offset) == 0) return nullptr;
	return font;
}

// The fonts to draw with, the best first: the one set, the desktop's bold sans (fontconfig's
// order), the usual bold sans files.
void LoadFonts() {
	if (g_fonts_loaded) return;
	g_fonts_loaded = true;
	std::set<std::string> tried;
	auto                  add = [&](const std::string& path) {
        if (path.empty() || !tried.insert(path).second || access(path.c_str(), R_OK) != 0) return;
        if (auto font = LoadFont(path)) g_fonts.push_back(std::move(font));
	};
	add(g_font_file);
	int count = 0;
	for (const auto& file: RunLines("fc-match -s -f '%{file}\\n' 'sans-serif:bold' 2>/dev/null")) {
		if (++count > 12) break;
		add(file);
	}
	for (const char* file: {"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
	                        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
	                        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
	                        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
	                        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
	                        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
	                        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
	                        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"})
		add(file);
}

std::vector<uint32_t> Codepoints(const std::string& text) {
	std::vector<uint32_t> result;
	for (size_t i = 0; i < text.size();) {
		const auto c = static_cast<unsigned char>(text[i]);
		uint32_t   cp = c;
		size_t     n  = 1;
		if (c >= 0xf0 && i + 3 < text.size()) {
			cp = (c & 0x07u) << 18u | (static_cast<unsigned char>(text[i + 1]) & 0x3fu) << 12u |
			     (static_cast<unsigned char>(text[i + 2]) & 0x3fu) << 6u |
			     (static_cast<unsigned char>(text[i + 3]) & 0x3fu);
			n = 4;
		} else if (c >= 0xe0 && i + 2 < text.size()) {
			cp = (c & 0x0fu) << 12u | (static_cast<unsigned char>(text[i + 1]) & 0x3fu) << 6u |
			     (static_cast<unsigned char>(text[i + 2]) & 0x3fu);
			n = 3;
		} else if (c >= 0xc0 && i + 1 < text.size()) {
			cp = (c & 0x1fu) << 6u | (static_cast<unsigned char>(text[i + 1]) & 0x3fu);
			n = 2;
		}
		result.push_back(cp);
		i += n;
	}
	return result;
}

const Font& FontFor(const std::vector<uint32_t>& codepoints) {
	LoadFonts();
	if (g_fonts.empty())
		throw PromptError("No bold sans-serif TrueType font was found to draw the key labels. "
		                  "Install one (for example DejaVu Sans or Liberation Sans) and try again.");
	for (const auto& font: g_fonts) {
		bool all = true;
		for (const uint32_t cp: codepoints)
			if (cp != ' ' && cp != '\n' && !font->Has(cp)) {
				all = false;
				break;
			}
		if (all) return *font;
	}
	return *g_fonts.front();
}

struct Line {
	std::vector<uint32_t> codepoints;
};

float LineWidth(const Font& font, const Line& line, float scale) {
	float width = 0;
	for (size_t i = 0; i < line.codepoints.size(); ++i) {
		int advance = 0, bearing = 0;
		stbtt_GetCodepointHMetrics(&font.info, static_cast<int>(line.codepoints[i]), &advance, &bearing);
		width += static_cast<float>(advance) * scale;
		if (i + 1 < line.codepoints.size())
			width += static_cast<float>(stbtt_GetCodepointKernAdvance(
			             &font.info, static_cast<int>(line.codepoints[i]),
			             static_cast<int>(line.codepoints[i + 1]))) *
			         scale;
	}
	return width;
}

} // namespace

void SetFontFile(const std::string& path) {
	g_font_file    = path;
	g_fonts.clear();
	g_fonts_loaded = false;
}

Buffer Draw(const std::string& label) {
	Buffer canvas(static_cast<size_t>(CanvasSize) * CanvasSize * 4, 0);
	auto   pixel = [&](int x, int y) { return &canvas[(static_cast<size_t>(y) * CanvasSize + static_cast<size_t>(x)) * 4]; };
	auto   fill  = [&](int x0, int y0, int x1, int y1, uint8_t value) {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                auto* p = pixel(x, y);
                p[0] = p[1] = p[2] = value;
                p[3] = 255;
            }
	};
	// The key cap as the pack's own icons have it (tools/keyboard-prompts/icons, measured): a dark
	// grey border 4.5 units wide round a white cap, from (8.5, 21.5) to (120.5, 107.75) in the
	// texture's 128 units, so a rebound key sits among the 21 that are not.
	fill(34, 86, 482, 431, 90);
	fill(52, 104, 464, 413, 255);
	// The text: black, centred on the cap by its ink. How tall the capitals are depends on how many
	// characters the longest line has, as in the pack's icons (38, 30, 21, 21, then 17 units);
	// a label too wide for the cap is made smaller until it fits.
	std::vector<Line> lines(1);
	const auto        codepoints = Codepoints(label);
	size_t            longest = 0;
	for (const uint32_t cp: codepoints) {
		if (cp == '\n') {
			lines.emplace_back();
		} else {
			lines.back().codepoints.push_back(cp);
			longest = std::max(longest, lines.back().codepoints.size());
		}
	}
	if (longest == 0) return canvas;
	const Font& font = FontFor(codepoints);
	int         ascent = 0, descent = 0, gap = 0;
	stbtt_GetFontVMetrics(&font.info, &ascent, &descent, &gap);
	int cap_x0 = 0, cap_y0 = 0, cap_x1 = 0, cap_y1 = 0;
	if (stbtt_GetCodepointBox(&font.info, 'H', &cap_x0, &cap_y0, &cap_x1, &cap_y1) == 0 || cap_y1 <= 0)
		cap_y1 = ascent * 7 / 10;
	static const float CapUnits[] = {38.0f, 30.0f, 21.0f, 21.0f};
	float cap = longest <= 4 ? CapUnits[longest - 1] : 17.0f;
	struct Placed {
		int      cp;
		float    x, y; // the pen's position before centring
	};
	std::vector<Placed> placed;
	float               scale = 0;
	float               min_x = 0, max_x = 0, min_y = 0, max_y = 0;
	for (;; cap -= 0.5f) {
		scale = cap * Scale / static_cast<float>(cap_y1);
		placed.clear();
		const float pitch = cap * 1.76f * Scale;
		min_x = min_y = 1e9f;
		max_x = max_y = -1e9f;
		for (size_t n = 0; n < lines.size(); ++n) {
			const auto& line = lines[n];
			float       x    = -LineWidth(font, line, scale) / 2.0f;
			const float y    = static_cast<float>(n) * pitch;
			for (size_t i = 0; i < line.codepoints.size(); ++i) {
				const int cp = static_cast<int>(line.codepoints[i]);
				int       advance = 0, bearing = 0;
				stbtt_GetCodepointHMetrics(&font.info, cp, &advance, &bearing);
				if (cp != ' ') {
					int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
					stbtt_GetCodepointBitmapBox(&font.info, cp, scale, scale, &x0, &y0, &x1, &y1);
					if (x1 > x0 && y1 > y0) {
						min_x = std::min(min_x, x + static_cast<float>(x0));
						max_x = std::max(max_x, x + static_cast<float>(x1));
						min_y = std::min(min_y, y + static_cast<float>(y0));
						max_y = std::max(max_y, y + static_cast<float>(y1));
					}
					placed.push_back({cp, x, y});
				}
				x += static_cast<float>(advance) * scale;
				if (i + 1 < line.codepoints.size())
					x += static_cast<float>(stbtt_GetCodepointKernAdvance(&font.info, cp, static_cast<int>(line.codepoints[i + 1]))) * scale;
			}
		}
		if (max_x <= min_x || max_y <= min_y) return canvas; // (nothing to draw: spaces)
		if ((max_x - min_x <= 90.0f * Scale && max_y - min_y <= 68.0f * Scale) || cap <= 3.0f) break;
	}
	const float shift_x = 64.0f * Scale - (min_x + max_x) / 2.0f;
	const float shift_y = 64.0f * Scale - (min_y + max_y) / 2.0f;
	for (const auto& glyph_at: placed) {
		const float x = glyph_at.x + shift_x, y = glyph_at.y + shift_y;
		const float fx = x - std::floor(x), fy = y - std::floor(y);
		int         x0 = 0, y0 = 0, x1 = 0, y1 = 0;
		stbtt_GetCodepointBitmapBoxSubpixel(&font.info, glyph_at.cp, scale, scale, fx, fy, &x0, &y0, &x1, &y1);
		const int w = x1 - x0, h = y1 - y0;
		if (w <= 0 || h <= 0) continue;
		std::vector<uint8_t> bits(static_cast<size_t>(w) * static_cast<size_t>(h));
		stbtt_MakeCodepointBitmapSubpixel(&font.info, bits.data(), w, h, w, scale, scale, fx, fy, glyph_at.cp);
		for (int gy = 0; gy < h; ++gy)
			for (int gx = 0; gx < w; ++gx) {
				const int px = static_cast<int>(std::floor(x)) + x0 + gx;
				const int py = static_cast<int>(std::floor(y)) + y0 + gy;
				if (px < 0 || py < 0 || px >= CanvasSize || py >= CanvasSize) continue;
				auto* p = pixel(px, py);
				if (p[3] != 255) continue; // (only on the key cap)
				const unsigned coverage = bits[static_cast<size_t>(gy) * static_cast<size_t>(w) + static_cast<size_t>(gx)];
				// Black over what is there.
				for (int c = 0; c < 3; ++c)
					p[c] = static_cast<uint8_t>((p[c] * (255u - coverage) + 127u) / 255u);
			}
	}
	return canvas;
}

Buffer Scaled(const Buffer& drawn, int width) {
	if (width < 1 || width > CanvasSize || (CanvasSize % width) != 0 ||
	    drawn.size() != static_cast<size_t>(CanvasSize) * CanvasSize * 4)
		throw PromptError("Invalid prompt texture size");
	const int block = CanvasSize / width;
	Buffer    out(static_cast<size_t>(width) * static_cast<size_t>(width) * 4);
	for (int y = 0; y < width; ++y)
		for (int x = 0; x < width; ++x) {
			uint64_t r = 0, g = 0, b = 0, a = 0;
			for (int by = 0; by < block; ++by)
				for (int bx = 0; bx < block; ++bx) {
					const auto* p = &drawn[(static_cast<size_t>(y * block + by) * CanvasSize + static_cast<size_t>(x * block + bx)) * 4];
					r += static_cast<uint64_t>(p[0]) * p[3];
					g += static_cast<uint64_t>(p[1]) * p[3];
					b += static_cast<uint64_t>(p[2]) * p[3];
					a += p[3];
				}
			auto* q = &out[(static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4];
			const uint64_t area = static_cast<uint64_t>(block) * static_cast<uint64_t>(block);
			if (a == 0) {
				q[0] = q[1] = q[2] = q[3] = 0;
			} else {
				q[0] = static_cast<uint8_t>((r + a / 2) / a);
				q[1] = static_cast<uint8_t>((g + a / 2) / a);
				q[2] = static_cast<uint8_t>((b + a / 2) / a);
				q[3] = static_cast<uint8_t>((a + area / 2) / area);
			}
		}
	return out;
}

std::array<Buffer, 8> Pixels(const std::string& label) {
	std::array<Buffer, 8> result;
	const auto            drawn = Draw(label);
	for (int level = 0; level < 8; ++level)
		result[static_cast<size_t>(level)] = Scaled(drawn, std::max(1, 128 >> level));
	return result;
}

Buffer Generate(const std::string& label) {
	return GenerateWithTemplate(label, Blank());
}

Buffer GenerateWithTemplate(const std::string& label, const Buffer& icon_template) {
	if (icon_template.size() != Blank().size()) throw PromptError("Invalid prompt template");
	const auto& offsets = Offsets();
	Buffer      result  = icon_template;
	std::fill(result.begin() + static_cast<std::ptrdiff_t>(FirstPixelByte),
	          result.begin() + static_cast<std::ptrdiff_t>(FirstPixelByte + PixelAreaBytes), 0);
	size_t pixel = 0;
	for (const auto& mip: Pixels(label))
		for (size_t at = 0; at < mip.size(); at += 4)
			std::memcpy(&result[static_cast<size_t>(offsets[pixel++])], &mip[at], 4);
	return result;
}

} // namespace PromptTexture
