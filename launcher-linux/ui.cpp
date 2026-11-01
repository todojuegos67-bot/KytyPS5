#include "ui.h"

#include "strings.h"

#include <SDL.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <stb_image.h>
#include <thread>
#include <unistd.h>

using Strings::F;
using Strings::T;

namespace {

// The Windows launcher's colors (launcher-windows/Theme.xaml).
constexpr uint32_t WindowColor = 0x101012, SidebarColor = 0x16161A, BarColor = 0x141417,
                   CardColor = 0x18181C, InputColor = 0x222228, InputHoverColor = 0x2A2A31,
                   BorderColor = 0x2E2E35, BorderHoverColor = 0x4A4A55, TextColor = 0xE8E5DF,
                   TextDimColor = 0x9A968F, TextFaintColor = 0x6B6863, GoldColor = 0xC9A45C,
                   GoodColor = 0x86B36F, WarnColor = 0xD9A441, BadColor = 0xD2604F,
                   PlayColor = 0x8E1B1B, PlayHoverColor = 0xAD2626, PlayPressedColor = 0x6C1414,
                   PlayBorderColor = 0xB8473F;

ImVec4 V(uint32_t rgb, float alpha = 1.0f) {
	return {static_cast<float>((rgb >> 16u) & 0xffu) / 255.0f,
	        static_cast<float>((rgb >> 8u) & 0xffu) / 255.0f,
	        static_cast<float>(rgb & 0xffu) / 255.0f, alpha};
}

ImU32 C(uint32_t rgb, float alpha = 1.0f) {
	return ImGui::GetColorU32(V(rgb, alpha));
}

constexpr float BodySize = 15.0f, SmallSize = 13.5f, SectionSize = 16.0f, PageTitleSize = 27.0f,
                BannerSize = 36.0f, GameTitleSize = 22.0f;

const char* const PageNames[] {"play",    "display",  "graphics",    "game",
                               "shaders", "advanced", "experimental"};
const char* const PageKeys[] {"nav.play",    "nav.display",  "nav.graphics",    "nav.game",
                              "nav.shaders", "nav.advanced", "nav.experimental"};

// The window's sizes for each aspect ratio of the 3D.
const std::vector<std::string>& Resolutions(const std::string& ratio) {
	static const std::vector<std::string> r169 {"1280x720", "1920x1080", "2560x1440", "3840x2160"};
	static const std::vector<std::string> r219 {"2560x1080", "3440x1440", "3840x1600", "5120x2160"};
	static const std::vector<std::string> r329 {"3840x1080", "5120x1440", "7680x2160"};
	return ratio == "21:9" ? r219 : ratio == "32:9" ? r329 : r169;
}

// The height Demon's Souls renders its 3D at; below 720 with the 1080p output only (DLSS's lower
// modes there, the game's TAA down to 540). The upscalers' usual shares among them (issue #129):
// Ultra Quality 77% (1664 of 4K, 832 of 1080p), Quality 66.7%, Balanced 58% (1256), Performance 50%,
// Ultra Performance 33%.
constexpr int RenderHeights[] {2160, 1800, 1664, 1620, 1512, 1440, 1296, 1256, 1152, 1080, 900, 832, 720};
constexpr int LowRenderHeights[] {630, 540, 360};
// The 1440p output's (DLSS or FSR 3, KYTY_OUTPUT_HEIGHT): DLAA, Ultra Quality 1104, Quality 960,
// Balanced 840, Performance 720, Ultra Performance 480 (its third; the emulator raises a height
// between the third and the half to the half).
constexpr int Output1440Heights[] {1440, 1152, 1104, 1080, 960, 840, 720, 480};

int HeightOf(const std::string& resolution) {
	const auto x = resolution.find('x');
	return x == std::string::npos ? 0 : std::atoi(resolution.c_str() + x + 1);
}

std::string Times(const std::string& resolution) {
	std::string out;
	for (const char c: resolution)
		out += c == 'x' ? std::string(" × ") : std::string(1, c);
	return out;
}

std::string Lower(std::string text) {
	for (auto& c: text)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return text;
}

std::string Trim(std::string text) {
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
		text.pop_back();
	size_t start = 0;
	while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0)
		++start;
	return text.substr(start);
}

std::string Paren(const std::string& a, const std::string& b) {
	return Strings::Current == Strings::Language::Chinese ? a + "（" + b + "）"
	                                                      : a + " (" + b + ")";
}

// ------------------------------------------------------------ fonts

// A font file stb_truetype draws: TrueType outlines (a CFF font, "OTTO", it cannot), alone or first
// in a collection.
bool TrueTypeFile(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	unsigned char head[16] {};
	if (!file.read(reinterpret_cast<char*>(head), sizeof(head))) return false;
	auto truetype = [](const unsigned char* tag) {
		return (tag[0] == 0 && tag[1] == 1 && tag[2] == 0 && tag[3] == 0) ||
		       std::memcmp(tag, "true", 4) == 0;
	};
	if (std::memcmp(head, "ttcf", 4) == 0) {
		const uint32_t offset = static_cast<uint32_t>(head[12]) << 24u |
		                        static_cast<uint32_t>(head[13]) << 16u |
		                        static_cast<uint32_t>(head[14]) << 8u | head[15];
		unsigned char tag[4] {};
		file.seekg(offset);
		return static_cast<bool>(file.read(reinterpret_cast<char*>(tag), 4)) && truetype(tag);
	}
	return truetype(head);
}

std::vector<std::string> Lines(const std::string& text) {
	std::vector<std::string> lines;
	size_t                   start = 0;
	while (start < text.size()) {
		const size_t end  = std::min(text.find('\n', start), text.size());
		auto         line = Trim(text.substr(start, end - start));
		if (!line.empty()) lines.push_back(line);
		start = end + 1;
	}
	return lines;
}

// The desktop's font for `pattern` (fontconfig's order), the first stb_truetype draws; else one of
// the usual files.
std::string FindFont(const std::string& pattern, const std::vector<std::string>& fallbacks) {
	std::string output;
	if (Platform::HasProgram("fc-match") &&
	    Platform::Run({"fc-match", "-s", "-f", "%{file}\n", pattern}, &output, 3000) == 0) {
		int tried = 0;
		for (const auto& file: Lines(output)) {
			if (++tried > 40) break;
			if (TrueTypeFile(file)) return file;
		}
	}
	for (const auto& file: fallbacks)
		if (access(file.c_str(), R_OK) == 0 && TrueTypeFile(file)) return file;
	return "";
}

// A font with Chinese characters, merged into the interface's font (the Chinese interface and
// console languages).
std::string FindChineseFont() {
	std::string output;
	if (Platform::HasProgram("fc-list") &&
	    Platform::Run({"fc-list", ":lang=zh-cn", "-f", "%{file}\n"}, &output, 3000) == 0) {
		auto files = Lines(output);
		// (Sans fonts first: a serif or a bitmap one only when nothing else.)
		std::stable_sort(files.begin(), files.end(),
		                 [](const std::string& a, const std::string& b) {
			                 auto rank = [](const std::string& f) {
				                 const auto lower = Lower(f);
				                 if (lower.find("sans") != std::string::npos ||
				                     lower.find("hei") != std::string::npos)
					                 return 0;
				                 return lower.find("serif") != std::string::npos ||
				                                lower.find("ming") != std::string::npos
				                            ? 2
				                            : 1;
			                 };
			                 return rank(a) < rank(b);
		                 });
		for (const auto& file: files)
			if (TrueTypeFile(file)) return file;
	}
	for (const char* file: {"/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
	                        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
	                        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
	                        "/usr/share/fonts/google-droid/DroidSansFallback.ttf"})
		if (access(file, R_OK) == 0 && TrueTypeFile(file)) return file;
	return "";
}

// ------------------------------------------------------------ checks

enum class Mark { Good, Warn, Bad, Busy };

// The check's sign, drawn (no font has to have it).
void DrawMark(Mark mark, float size) {
	ImDrawList*  draw = ImGui::GetWindowDrawList();
	const auto   at   = ImGui::GetCursorScreenPos();
	const float  h    = ImGui::GetTextLineHeight();
	const ImVec2 c(at.x + size * 0.45f, at.y + h * 0.5f);
	const float  r     = size * 0.32f;
	const float  thick = std::max(1.5f, size * 0.11f);
	switch (mark) {
		case Mark::Good: {
			const ImVec2 points[] {{c.x - r, c.y},
			                       {c.x - r * 0.3f, c.y + r * 0.75f},
			                       {c.x + r * 1.05f, c.y - r * 0.8f}};
			draw->AddPolyline(points, 3, C(GoodColor), ImDrawFlags_None, thick);
			break;
		}
		case Mark::Bad:
			draw->AddLine({c.x - r * 0.8f, c.y - r * 0.8f}, {c.x + r * 0.8f, c.y + r * 0.8f},
			              C(BadColor), thick);
			draw->AddLine({c.x - r * 0.8f, c.y + r * 0.8f}, {c.x + r * 0.8f, c.y - r * 0.8f},
			              C(BadColor), thick);
			break;
		case Mark::Warn:
			draw->AddLine({c.x, c.y - r * 1.05f}, {c.x, c.y + r * 0.35f}, C(WarnColor), thick);
			draw->AddCircleFilled({c.x, c.y + r * 0.95f}, thick * 0.65f, C(WarnColor));
			break;
		case Mark::Busy:
			for (int i = -1; i <= 1; ++i)
				draw->AddCircleFilled({c.x + static_cast<float>(i) * r * 0.85f, c.y}, thick * 0.55f,
				                      C(TextDimColor));
			break;
	}
	ImGui::Dummy({size, h});
}

} // namespace

// ------------------------------------------------------------ window

LauncherWindow::LauncherWindow(LauncherContext& context, SDL_Renderer* renderer)
    : context_(context), commands_(context), renderer_(renderer) {
	game_text_      = S().Str("game");
	extra_text_     = S().Str("extra_args");
	wanted_taa_     = S().Int("render") > 0 ? S().Int("render") : 1440;
	wanted_sr_      = S().Int("render_dlss") > 0 ? S().Int("render_dlss") : 1080;
	dlss_available_ = context_.DlssAvailable();
	LoadGame(game_text_);
}

LauncherWindow::~LauncherWindow() {
	if (banner_ != nullptr) SDL_DestroyTexture(banner_);
	S().Save();
}

const char* LauncherWindow::PageName(int index) {
	return PageNames[std::clamp(index, 0, PageCount() - 1)];
}

void LauncherWindow::Wake() {
	SDL_Event event {};
	event.type = SDL_USEREVENT;
	SDL_PushEvent(&event);
}

void LauncherWindow::ApplyScale(float display_scale) {
	display_scale_   = display_scale;
	const int chosen = S().Int("ui_scale");
	scale_ = chosen >= 50 && chosen <= 300 ? static_cast<float>(chosen) / 100.0f : display_scale_;
	scale_changed_ = false;

	ImGuiIO& io = ImGui::GetIO();
	if (!fonts_loaded_) {
		fonts_loaded_   = true;
		const auto sans = FindFont(
		    "sans-serif",
		    {"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
		     "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
		     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
		     "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
		     "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
		     "/usr/share/fonts/noto/NotoSans-Regular.ttf",
		     "/usr/share/fonts/truetype/freefont/FreeSans.ttf"});
		const auto serif =
		    FindFont("serif", {"/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
		                       "/usr/share/fonts/TTF/DejaVuSerif.ttf",
		                       "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
		                       "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
		                       "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
		                       "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
		                       "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"});
		const auto chinese = FindChineseFont();
		auto       add     = [&](const std::string& file) -> ImFont* {
            if (file.empty()) return nullptr;
            ImFont* font = io.Fonts->AddFontFromFileTTF(file.c_str(), BodySize);
            if (font != nullptr && !chinese.empty() && chinese != file) {
                ImFontConfig merge;
                merge.MergeMode = true;
                io.Fonts->AddFontFromFileTTF(chinese.c_str(), BodySize, &merge);
            }
            return font;
		};
		sans_  = add(sans);
		serif_ = add(serif);
		if (sans_ == nullptr) sans_ = io.Fonts->AddFontDefault();
		if (serif_ == nullptr) serif_ = sans_;
		io.FontDefault = sans_;
	}

	ImGuiStyle& style = ImGui::GetStyle();
	style             = ImGuiStyle();
	ImGui::StyleColorsDark(&style);
	auto* colors                          = style.Colors;
	colors[ImGuiCol_WindowBg]             = V(WindowColor);
	colors[ImGuiCol_ChildBg]              = V(0, 0.0f);
	colors[ImGuiCol_PopupBg]              = V(CardColor);
	colors[ImGuiCol_Border]               = V(BorderColor);
	colors[ImGuiCol_FrameBg]              = V(InputColor);
	colors[ImGuiCol_FrameBgHovered]       = V(InputHoverColor);
	colors[ImGuiCol_FrameBgActive]        = V(InputHoverColor);
	colors[ImGuiCol_Button]               = V(InputColor);
	colors[ImGuiCol_ButtonHovered]        = V(InputHoverColor);
	colors[ImGuiCol_ButtonActive]         = V(BorderColor);
	colors[ImGuiCol_Header]               = V(InputHoverColor);
	colors[ImGuiCol_HeaderHovered]        = V(InputHoverColor);
	colors[ImGuiCol_HeaderActive]         = V(BorderHoverColor);
	colors[ImGuiCol_Text]                 = V(TextColor);
	colors[ImGuiCol_TextDisabled]         = V(TextFaintColor);
	colors[ImGuiCol_CheckMark]            = V(GoldColor);
	colors[ImGuiCol_SliderGrab]           = V(GoldColor);
	colors[ImGuiCol_ScrollbarBg]          = V(0, 0.0f);
	colors[ImGuiCol_ScrollbarGrab]        = V(BorderColor);
	colors[ImGuiCol_ScrollbarGrabHovered] = V(BorderHoverColor);
	colors[ImGuiCol_ScrollbarGrabActive]  = V(GoldColor, 0.7f);
	colors[ImGuiCol_NavCursor]            = V(GoldColor);
	colors[ImGuiCol_TitleBg]              = V(SidebarColor);
	colors[ImGuiCol_TitleBgActive]        = V(SidebarColor);
	colors[ImGuiCol_ModalWindowDimBg]     = V(0, 0.6f);
	colors[ImGuiCol_TextSelectedBg]       = V(GoldColor, 0.35f);
	colors[ImGuiCol_Separator]            = V(BorderColor);
	style.WindowRounding                  = 6.0f;
	style.ChildRounding                   = 6.0f;
	style.FrameRounding                   = 4.0f;
	style.PopupRounding                   = 4.0f;
	style.GrabRounding                    = 4.0f;
	style.WindowBorderSize                = 1.0f;
	style.FrameBorderSize                 = 1.0f;
	style.PopupBorderSize                 = 1.0f;
	style.FramePadding                    = {10.0f, 6.0f};
	style.ItemSpacing                     = {8.0f, 7.0f};
	style.WindowPadding                   = {16.0f, 14.0f};
	style.ScrollbarSize                   = 12.0f;
	style.WindowTitleAlign                = {0.5f, 0.5f};
	style.ScaleAllSizes(scale_);
	style.FontSizeBase = BodySize;
	style.FontScaleDpi = scale_;
}

// ------------------------------------------------------------ widgets

void LauncherWindow::PageTitle(const char* key) {
	ImGui::PushFont(serif_, PageTitleSize);
	ImGui::TextUnformatted(T(key));
	ImGui::PopFont();
	ImGui::Dummy({0, 6.0f * scale_});
}

void LauncherWindow::Section(const char* key) {
	ImGui::Dummy({0, 14.0f * scale_});
	ImGui::PushFont(nullptr, SectionSize);
	ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
	ImGui::TextUnformatted(T(key));
	ImGui::PopStyleColor();
	ImGui::PopFont();
	const auto at = ImGui::GetCursorScreenPos();
	ImGui::GetWindowDrawList()->AddLine(
	    {at.x, at.y}, {at.x + ImGui::GetContentRegionAvail().x, at.y}, C(BorderColor));
	ImGui::Dummy({0, 4.0f * scale_});
}

void LauncherWindow::Desc(const std::string& text) {
	if (text.empty()) return;
	ImGui::PushFont(nullptr, SmallSize);
	ImGui::PushStyleColor(ImGuiCol_Text, V(TextDimColor));
	ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x);
	ImGui::TextUnformatted(text.c_str());
	ImGui::PopTextWrapPos();
	ImGui::PopStyleColor();
	ImGui::PopFont();
}

// A setting: its name on the left, the control and what it does on the right.
void LauncherWindow::Row(const char* label_key, const std::function<void()>& control,
                         const std::string& desc) {
	const float label_width = 232.0f * scale_;
	const float x0          = ImGui::GetCursorPosX();
	ImGui::Dummy({0, 3.0f * scale_});
	ImGui::BeginGroup();
	ImGui::AlignTextToFramePadding();
	ImGui::PushTextWrapPos(x0 + label_width - 16.0f * scale_);
	ImGui::TextUnformatted(label_key != nullptr ? T(label_key) : "");
	ImGui::PopTextWrapPos();
	ImGui::EndGroup();
	ImGui::SameLine(x0 + label_width);
	ImGui::BeginGroup();
	ImGui::PushID(label_key != nullptr ? label_key : desc.c_str());
	control();
	ImGui::PopID();
	Desc(desc);
	ImGui::EndGroup();
	ImGui::Dummy({0, 5.0f * scale_});
}

// A toggle switch (the Windows launcher's Switch style).
bool LauncherWindow::Switch(const char* id, bool* value, bool enabled) {
	const float height = ImGui::GetFrameHeight() * 0.82f;
	const float width  = height * 1.85f;
	const auto  at     = ImGui::GetCursorScreenPos();
	const float top    = at.y + (ImGui::GetFrameHeight() - height) * 0.5f;
	ImGui::BeginDisabled(!enabled);
	const bool clicked = ImGui::InvisibleButton(id, {width, ImGui::GetFrameHeight()});
	const bool hovered = ImGui::IsItemHovered();
	ImGui::EndDisabled();
	if (clicked && enabled) *value = !*value;
	ImDrawList* draw  = ImGui::GetWindowDrawList();
	const float alpha = enabled ? 1.0f : 0.35f;
	const bool  on    = *value;
	draw->AddRectFilled({at.x, top}, {at.x + width, top + height},
	                    on ? C(GoldColor, alpha) : C(hovered ? InputHoverColor : InputColor, alpha),
	                    height * 0.5f);
	if (!on)
		draw->AddRect({at.x, top}, {at.x + width, top + height},
		              C(hovered ? BorderHoverColor : BorderColor, alpha), height * 0.5f);
	const float radius = height * 0.5f - 3.0f * scale_;
	const float cx     = on ? at.x + width - height * 0.5f : at.x + height * 0.5f;
	draw->AddCircleFilled({cx, top + height * 0.5f}, radius,
	                      on ? C(WindowColor, alpha) : C(TextDimColor, alpha));
	return clicked && enabled;
}

bool LauncherWindow::SettingSwitch(const char* id, const char* key, bool enabled) {
	bool value = S().Bool(key);
	if (!Switch(id, &value, enabled)) return false;
	S().Set(key, value);
	Changed();
	return true;
}

bool LauncherWindow::Combo(const char* id, int* index, const std::vector<std::string>& labels,
                           float width, bool enabled) {
	if (labels.empty()) return false;
	*index       = std::clamp(*index, 0, static_cast<int>(labels.size()) - 1);
	bool changed = false;
	ImGui::BeginDisabled(!enabled);
	ImGui::SetNextItemWidth(std::min(width * scale_, ImGui::GetContentRegionAvail().x));
	if (ImGui::BeginCombo(id, labels[static_cast<size_t>(*index)].c_str(),
	                      ImGuiComboFlags_HeightLarge)) {
		for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
			const bool selected = i == *index;
			if (ImGui::Selectable(labels[static_cast<size_t>(i)].c_str(), selected)) {
				changed = i != *index;
				*index  = i;
			}
			if (selected) ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::EndDisabled();
	return changed;
}

bool LauncherWindow::FlatButton(const char* label, bool enabled) {
	ImGui::BeginDisabled(!enabled);
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {14.0f * scale_, 7.0f * scale_});
	ImGui::PushStyleColor(ImGuiCol_Border, V(BorderHoverColor));
	const bool clicked = ImGui::Button(label);
	ImGui::PopStyleColor();
	ImGui::PopStyleVar();
	ImGui::EndDisabled();
	return clicked;
}

void LauncherWindow::Feedback(int issue) {
	const auto text = F("feedback", {std::to_string(issue)});
	ImGui::PushFont(nullptr, SmallSize);
	ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
	ImGui::TextUnformatted(text.c_str());
	ImGui::PopStyleColor();
	ImGui::PopFont();
	if (ImGui::IsItemHovered()) {
		ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		const auto min = ImGui::GetItemRectMin();
		const auto max = ImGui::GetItemRectMax();
		ImGui::GetWindowDrawList()->AddLine({min.x, max.y}, max, C(GoldColor));
	}
	if (ImGui::IsItemClicked())
		Platform::Open("https://github.com/chenxiao07/KytyPS5/issues/" + std::to_string(issue));
}

void LauncherWindow::Card(const char* id, const std::function<void()>& content) {
	ImGui::PushStyleColor(ImGuiCol_ChildBg, V(CardColor));
	ImGui::PushStyleColor(ImGuiCol_Border, V(BorderColor));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {18.0f * scale_, 16.0f * scale_});
	ImGui::BeginChild(id, {0, 0},
	                  ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY |
	                      ImGuiChildFlags_AlwaysUseWindowPadding,
	                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	content();
	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopStyleColor(2);
}

// ------------------------------------------------------------ frame

void LauncherWindow::Frame() {
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->Pos);
	ImGui::SetNextWindowSize(viewport->Size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("##launcher", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
	                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImGui::PopStyleVar(3);

	const float sidebar = 232.0f * scale_;
	Sidebar(sidebar);
	ImGui::SameLine(0, 0);
	ImGui::BeginGroup();
	const float bar = 66.0f * scale_;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {44.0f * scale_, 30.0f * scale_});
	ImGui::BeginChild("##page", {0, ImGui::GetContentRegionAvail().y - bar},
	                  page_ == 0 ? ImGuiChildFlags_None : ImGuiChildFlags_AlwaysUseWindowPadding);
	ImGui::PopStyleVar();
	if (page_ != 0) {
		// (A column of at most 880 points, as on Windows.)
		const float most = 880.0f * scale_;
		ImGui::BeginChild("##column", {std::min(most, ImGui::GetContentRegionAvail().x), 0},
		                  ImGuiChildFlags_AutoResizeY,
		                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	}
	switch (page_) {
		case 0: PlayPage(); break;
		case 1: DisplayPage(); break;
		case 2: GraphicsPage(); break;
		case 3: GamePage(); break;
		case 4: ShadersPage(); break;
		case 5: AdvancedPage(); break;
		default: ExperimentalPage(); break;
	}
	if (page_ != 0) {
		ImGui::EndChild();
		ImGui::Dummy({0, 30.0f * scale_});
	}
	ImGui::EndChild();
	BottomBar();
	ImGui::EndGroup();
	Modals();
	ImGui::End();
	S().Save();
}

void LauncherWindow::Sidebar(float width) {
	ImGui::PushStyleColor(ImGuiCol_ChildBg, V(SidebarColor));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {24.0f * scale_, 26.0f * scale_});
	ImGui::BeginChild("##sidebar", {width, 0}, ImGuiChildFlags_AlwaysUseWindowPadding,
	                  ImGuiWindowFlags_NoScrollbar);
	ImGui::PopStyleVar();
	// (The right edge.)
	const auto origin = ImGui::GetWindowPos();
	ImGui::GetWindowDrawList()->AddLine({origin.x + width - 1, origin.y},
	                                    {origin.x + width - 1, origin.y + ImGui::GetWindowHeight()},
	                                    C(BorderColor));
	if (game_.Found()) {
		ImGui::PushFont(serif_, GameTitleSize);
		ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
		ImGui::PushTextWrapPos(width - 16.0f * scale_);
		ImGui::TextUnformatted((game_.name.empty() ? game_.title_id : game_.name).c_str());
		ImGui::PopTextWrapPos();
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImGui::Dummy({0, 2.0f * scale_});
	}
	ImGui::TextUnformatted("KytyPS5");
	ImGui::PushStyleColor(ImGuiCol_Text, V(TextFaintColor));
	ImGui::PushFont(nullptr, SmallSize);
	ImGui::TextUnformatted(T("app.subtitle"));
	ImGui::PopFont();
	ImGui::PopStyleColor();
	ImGui::Dummy({0, 20.0f * scale_});

	// The pages: the whole width, a gold bar at the chosen one.
	const float left = ImGui::GetCursorPosX();
	for (int i = 0; i < PageCount(); ++i) {
		ImGui::PushID(i);
		const bool chosen = i == page_;
		ImGui::SetCursorPosX(0);
		const auto   at   = ImGui::GetCursorScreenPos();
		const ImVec2 size = {width - 1, ImGui::GetTextLineHeight() + 16.0f * scale_};
		if (ImGui::InvisibleButton("##nav", size)) page_ = i;
		const bool  hovered = ImGui::IsItemHovered();
		ImDrawList* draw    = ImGui::GetWindowDrawList();
		if (chosen || hovered)
			draw->AddRectFilled(at, {at.x + size.x, at.y + size.y},
			                    C(chosen ? InputColor : InputHoverColor, chosen ? 1.0f : 0.6f));
		if (chosen) draw->AddRectFilled(at, {at.x + 3.0f * scale_, at.y + size.y}, C(GoldColor));
		draw->AddText({at.x + left, at.y + 8.0f * scale_},
		              C(chosen || hovered ? TextColor : TextDimColor), T(PageKeys[i]));
		ImGui::PopID();
	}

	// The hint at the bottom.
	ImGui::PushFont(nullptr, SmallSize);
	const float wrap = width - left - 18.0f * scale_;
	const auto  hint = ImGui::CalcTextSize(T("nav.hint"), nullptr, false, wrap);
	const float y    = ImGui::GetWindowHeight() - hint.y - 22.0f * scale_;
	if (y > ImGui::GetCursorPosY()) {
		ImGui::SetCursorPos({left, y});
		ImGui::PushStyleColor(ImGuiCol_Text, V(TextFaintColor));
		ImGui::PushTextWrapPos(left + wrap);
		ImGui::TextUnformatted(T("nav.hint"));
		ImGui::PopTextWrapPos();
		ImGui::PopStyleColor();
	}
	ImGui::PopFont();
	ImGui::EndChild();
	ImGui::PopStyleColor();
}

void LauncherWindow::BottomBar() {
	ImGui::PushStyleColor(ImGuiCol_ChildBg, V(BarColor));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {28.0f * scale_, 14.0f * scale_});
	ImGui::BeginChild("##bar", {0, 0}, ImGuiChildFlags_AlwaysUseWindowPadding,
	                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	ImGui::PopStyleVar();
	const auto origin = ImGui::GetWindowPos();
	ImGui::GetWindowDrawList()->AddLine(origin, {origin.x + ImGui::GetWindowWidth(), origin.y},
	                                    C(BorderColor));

	// The buttons on the right; the summary in what is left.
	ImGui::PushFont(nullptr, 16.0f);
	const char* play = T("button.play");
	const float play_width =
	    std::max(150.0f * scale_, ImGui::CalcTextSize(play).x + 56.0f * scale_);
	ImGui::PopFont();
	const char* precompile       = T("button.precompile");
	const float precompile_width = ImGui::CalcTextSize(precompile).x + 32.0f * scale_;
	const float height           = ImGui::GetContentRegionAvail().y;
	const float right            = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
	const float summary_width =
	    right - play_width - precompile_width - 30.0f * scale_ - ImGui::GetCursorPosX();

	const bool recent_notice =
	    !notice_.empty() && std::chrono::steady_clock::now() - notice_at_ < std::chrono::seconds(6);
	std::string summary = recent_notice ? notice_ : Summary(nullptr);
	ImGui::PushFont(nullptr, SmallSize);
	if (ImGui::CalcTextSize(summary.c_str()).x > summary_width) {
		while (!summary.empty() && ImGui::CalcTextSize((summary + "…").c_str()).x > summary_width) {
			summary.pop_back();
			while (!summary.empty() &&
			       (static_cast<unsigned char>(summary.back()) & 0xC0u) == 0x80u)
				summary.pop_back();
			if (!summary.empty() && (static_cast<unsigned char>(summary.back()) & 0x80u) != 0)
				summary.pop_back();
		}
		summary += "…";
	}
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (height - ImGui::GetTextLineHeight()) * 0.5f);
	ImGui::PushStyleColor(ImGuiCol_Text, V(recent_notice ? GoodColor : TextDimColor));
	ImGui::TextUnformatted(summary.c_str());
	ImGui::PopStyleColor();
	ImGui::PopFont();

	ImGui::SameLine(right - play_width - precompile_width - 14.0f * scale_);
	ImGui::SetCursorPosY(14.0f * scale_ +
	                     (height - ImGui::GetFrameHeight() - 6.0f * scale_) * 0.5f);
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {16.0f * scale_, 9.0f * scale_});
	ImGui::PushStyleColor(ImGuiCol_Border, V(BorderHoverColor));
	if (ImGui::Button(precompile, {precompile_width, 0})) StartInTerminal(commands_.Precompile(""));
	ImGui::PopStyleColor();
	ImGui::PopStyleVar();

	ImGui::SameLine(right - play_width);
	ImGui::SetCursorPosY(14.0f * scale_);
	ImGui::PushFont(nullptr, 16.0f);
	ImGui::PushStyleColor(ImGuiCol_Button, V(PlayColor));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, V(PlayHoverColor));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, V(PlayPressedColor));
	ImGui::PushStyleColor(ImGuiCol_Border, V(PlayBorderColor));
	ImGui::PushStyleColor(ImGuiCol_Text, V(0xFFFFFF));
	ImGui::BeginDisabled(context_.emulator.empty());
	if (ImGui::Button(play, {play_width, height})) Play();
	ImGui::EndDisabled();
	ImGui::PopStyleColor(5);
	ImGui::PopFont();
	ImGui::EndChild();
	ImGui::PopStyleColor();
}

// ------------------------------------------------------------ Play

void LauncherWindow::PlayPage() {
	const float width = ImGui::GetContentRegionAvail().x;
	const auto  at    = ImGui::GetCursorScreenPos();
	const float h     = 270.0f * scale_;
	ImDrawList* draw  = ImGui::GetWindowDrawList();
	// The game's own art from its dump (sce_sys/pic0.png), under a fade into the window; a plain
	// gradient without one.
	draw->AddRectFilledMultiColor(at, {at.x + width, at.y + h}, C(0x2A2218), C(0x1A1714),
	                              C(0x101012), C(0x16130F));
	if (banner_ != nullptr && banner_w_ > 0 && banner_h_ > 0) {
		// (Filling the width and height, cropped: UniformToFill.)
		const float scale =
		    std::max(width / static_cast<float>(banner_w_), h / static_cast<float>(banner_h_));
		const float w  = static_cast<float>(banner_w_) * scale;
		const float hh = static_cast<float>(banner_h_) * scale;
		const float u0 = (w - width) * 0.5f / w;
		const float v0 = (hh - h) * 0.5f / hh;
		draw->AddImage(static_cast<ImTextureID>(reinterpret_cast<intptr_t>(banner_)), at,
		               {at.x + width, at.y + h}, {u0, v0}, {1.0f - u0, 1.0f - v0});
	}
	draw->AddRectFilledMultiColor(at, {at.x + width, at.y + h * 0.55f}, C(WindowColor, 0.12f),
	                              C(WindowColor, 0.12f), C(WindowColor, 0.38f),
	                              C(WindowColor, 0.38f));
	draw->AddRectFilledMultiColor({at.x, at.y + h * 0.55f}, {at.x + width, at.y + h},
	                              C(WindowColor, 0.38f), C(WindowColor, 0.38f),
	                              C(WindowColor, 1.0f), C(WindowColor, 1.0f));
	draw->AddRectFilledMultiColor(at, {at.x + width * 0.6f, at.y + h}, C(WindowColor, 0.7f),
	                              C(WindowColor, 0.0f), C(WindowColor, 0.0f), C(WindowColor, 0.7f));
	const std::string title =
	    game_.Found() ? (game_.name.empty() ? game_.title_id : game_.name) : T("play.choose");
	const std::string line = game_.Found() ? game_.title_id + "  ·  " + game_.version : "";
	ImGui::PushFont(serif_, BannerSize);
	const float title_h = ImGui::GetTextLineHeight();
	ImGui::PopFont();
	const float line_h = line.empty() ? 0.0f : ImGui::GetTextLineHeight() + 6.0f * scale_;
	ImGui::SetCursorScreenPos(
	    {at.x + 44.0f * scale_, at.y + h - 22.0f * scale_ - line_h - title_h});
	ImGui::PushFont(serif_, BannerSize);
	ImGui::PushStyleColor(ImGuiCol_Text, V(0xFFFFFF));
	ImGui::TextUnformatted(title.c_str());
	ImGui::PopStyleColor();
	ImGui::PopFont();
	if (!line.empty()) {
		ImGui::SetCursorScreenPos(
		    {at.x + 46.0f * scale_, ImGui::GetCursorScreenPos().y + 4.0f * scale_});
		ImGui::PushStyleColor(ImGuiCol_Text, V(TextDimColor));
		ImGui::TextUnformatted(line.c_str());
		ImGui::PopStyleColor();
	}
	ImGui::SetCursorScreenPos({at.x, at.y + h});

	// Checks on the left; the game and the current settings on the right.
	const float margin = 44.0f * scale_;
	const float gap    = 24.0f * scale_;
	const float inner  = width - 2 * margin;
	const float left_w = (inner - gap) * (1.0f / 2.05f);
	ImGui::SetCursorPos({margin, ImGui::GetCursorPosY() + 22.0f * scale_});
	ImGui::BeginGroup();
	ImGui::BeginChild("##checks-column", {left_w, 0}, ImGuiChildFlags_AutoResizeY,
	                  ImGuiWindowFlags_NoScrollbar);
	Card("##checks", [&] {
		ImGui::PushFont(nullptr, SectionSize);
		ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
		ImGui::TextUnformatted(T("play.checks"));
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImGui::Dummy({0, 4.0f * scale_});
		auto check = [&](Mark mark, const std::string& text) {
			DrawMark(mark, 22.0f * scale_);
			ImGui::SameLine(0, 0);
			ImGui::PushTextWrapPos(0.0f);
			ImGui::TextUnformatted(text.c_str());
			ImGui::PopTextWrapPos();
			ImGui::Dummy({0, 2.0f * scale_});
		};
		if (context_.emulator.empty()) check(Mark::Bad, T("emulator.missing"));
		if (game_job_ != nullptr)
			check(Mark::Busy, T("game.loading"));
		else if (game_.Found())
			check(Mark::Good, F("game.ok", {game_.name.empty() ? game_.title_id : game_.name,
			                                game_.title_id, game_.version}));
		else
			check(Mark::Bad, T(Trim(game_text_).empty() ? "game.none" : "game.bad"));
		if (context_.gpus.empty()) {
			check(Mark::Warn, T("gpu.none"));
		} else {
			const auto& gpu = context_.gpus.front();
			if (gpu.memory > 0)
				check(
				    Mark::Good,
				    F("gpu.ok", {gpu.name, std::to_string(static_cast<int>(std::lround(
				                               static_cast<double>(gpu.memory) / 1073741824.0)))}));
			else
				check(Mark::Good, F("gpu.okNoMemory", {gpu.name}));
		}
		const auto ram_gb =
		    static_cast<int>(std::lround(static_cast<double>(context_.ram) / 1073741824.0));
		check(ram_gb <= 20 ? Mark::Warn : Mark::Good,
		      F(ram_gb <= 20 ? "ram.low" : "ram.ok", {std::to_string(ram_gb)}));
		switch (store_) {
			case StoreState::Checking: check(Mark::Busy, T("shaders.checking")); break;
			case StoreState::Current: check(Mark::Good, T("shaders.current")); break;
			case StoreState::Outdated: check(Mark::Warn, T("shaders.outdated")); break;
			case StoreState::OutdatedAll: check(Mark::Warn, T("shaders.outdatedAll")); break;
			case StoreState::None: check(Mark::Warn, T("shaders.none")); break;
			default:
				if (game_.Found()) check(Mark::Busy, T("shaders.unknown"));
				break;
		}
		if (dlss_available_)
			check(Mark::Good, T("dlss.ok"));
		else
			check(Mark::Busy, T(context_.rtx ? "dlss.noLibrary" : "dlss.noRtx"));
		check(Mark::Good, T("fsr.ok"));
	});
	ImGui::EndChild();
	ImGui::SameLine(0, gap);
	ImGui::BeginChild("##game-column", {inner - left_w - gap, 0}, ImGuiChildFlags_AutoResizeY,
	                  ImGuiWindowFlags_NoScrollbar);
	Card("##game", [&] {
		ImGui::PushFont(nullptr, SectionSize);
		ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
		ImGui::TextUnformatted(T("play.gamebox"));
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImGui::Dummy({0, 4.0f * scale_});
		ImGui::SetNextItemWidth(-1);
		if (ImGui::InputText("##game-path", &game_text_)) {
			game_edit_         = std::chrono::steady_clock::now();
			game_edit_pending_ = true;
		}
		if (FlatButton(T("button.folder"))) PickFolder(false);
		ImGui::SameLine();
		if (FlatButton(T("button.zar"))) PickFolder(true);
	});
	ImGui::Dummy({0, 10.0f * scale_});
	Card("##summary", [&] {
		ImGui::PushFont(nullptr, SectionSize);
		ImGui::PushStyleColor(ImGuiCol_Text, V(GoldColor));
		ImGui::TextUnformatted(T("play.summary"));
		ImGui::PopStyleColor();
		ImGui::PopFont();
		ImGui::Dummy({0, 2.0f * scale_});
		std::vector<std::pair<std::string, std::string>> lines;
		(void)Summary(&lines);
		if (ImGui::BeginTable("##summary-table", 2, ImGuiTableFlags_None)) {
			ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
			ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
			for (const auto& [label, value]: lines) {
				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::PushStyleColor(ImGuiCol_Text, V(TextDimColor));
				ImGui::TextUnformatted(label.c_str());
				ImGui::PopStyleColor();
				ImGui::TableSetColumnIndex(1);
				ImGui::PushTextWrapPos(0.0f);
				ImGui::TextUnformatted(value.c_str());
				ImGui::PopTextWrapPos();
			}
			ImGui::EndTable();
		}
	});
	ImGui::EndChild();
	ImGui::EndGroup();
	ImGui::Dummy({0, 30.0f * scale_});
}

// ------------------------------------------------------------ Display

void LauncherWindow::DisplayPage() {
	PageTitle("display.title");
	Section("display.window");
	const std::vector<std::string> ratios {"16:9", "21:9", "32:9"};
	const auto                     ratio = S().Str("ratio");
	// The ratio's resolutions; the one saved, else the one nearest its height (saved).
	const auto& list = Resolutions(ratio);
	int         current =
	    static_cast<int>(std::find(list.begin(), list.end(), S().Str("resolution")) - list.begin());
	if (current >= static_cast<int>(list.size())) {
		const int height =
		    HeightOf(S().Str("resolution")) > 0 ? HeightOf(S().Str("resolution")) : 1440;
		int best = 1 << 30;
		for (int i = 0; i < static_cast<int>(list.size()); ++i) {
			const int distance = std::abs(HeightOf(list[static_cast<size_t>(i)]) - height);
			if (distance < best) {
				best    = distance;
				current = i;
			}
		}
		S().Set("resolution", list[static_cast<size_t>(current)]);
	}
	Row(
	    "resolution",
	    [&] {
		    std::vector<std::string> labels;
		    for (const auto& r: list)
			    labels.push_back(Times(r));
		    if (Combo("##resolution", &current, labels, 300)) {
			    S().Set("resolution", list[static_cast<size_t>(current)]);
			    Changed();
		    }
	    },
	    T("resolution.desc"));
	Row(
	    "ratio",
	    [&] {
		    int index =
		        static_cast<int>(std::find(ratios.begin(), ratios.end(), ratio) - ratios.begin());
		    if (Combo("##ratio", &index, {T("ratio.169"), "21:9", "32:9"}, 300)) {
			    S().Set("ratio", ratios[static_cast<size_t>(std::clamp(index, 0, 2))]);
			    Changed();
		    }
	    },
	    T("ratio.desc"));
	// (With 21:9 or 32:9 only: the movies and menus at 16:9 between black bars, commands.cpp.)
	Row(
	    "widebars", [&] { SettingSwitch("##widebars", "wide_bars", S().Str("ratio") != "16:9"); },
	    T("widebars.desc"));
	Row("fullscreen", [&] { SettingSwitch("##fullscreen", "fullscreen"); });
	Row(
	    "aspect", [&] { SettingSwitch("##aspect", "aspect", S().Bool("fullscreen")); },
	    T("aspect.desc"));
	Row("hdr", [&] { SettingSwitch("##hdr", "hdr"); }, T("hdr.desc"));
	Section("display.rate");
	Row("fps120", [&] { SettingSwitch("##fps120", "fps120"); }, T("fps120.desc"));
	Row("vsync", [&] { SettingSwitch("##vsync", "vsync"); }, T("vsync.desc"));
	Row("lowlatency", [&] { SettingSwitch("##lowlatency", "lowlatency"); }, T("lowlatency.desc"));
}

// ------------------------------------------------------------ Graphics

bool LauncherWindow::RenderForUpscaler() const {
	return commands_.Dlss() || commands_.FsrUpscaler();
}

bool LauncherWindow::HdOutputChosen() const {
	return commands_.HdOutput();
}

// The output's lines: 1080 (the 1080p output), 1440 (1440p with an upscaler) or 2160.
int LauncherWindow::OutputLines() const {
	if (HdOutputChosen()) return 1080;
	return commands_.Output1440() ? 1440 : 2160;
}

double LauncherWindow::RatioAspect() const {
	const auto ratio = S().Str("ratio");
	return ratio == "21:9" ? 64.0 / 27.0 : ratio == "32:9" ? 32.0 / 9.0 : 16.0 / 9.0;
}

// The highest render height whose 3D at the ratio fits the game's render target memory, made for
// 3840x2160: no more of the GPU's 64 KiB tiles than 3840x2160 (the emulator's FitsRenderTargets).
int LauncherWindow::MaxRenderHeight() const {
	const double aspect = RatioAspect();
	const int    tiles[4][2] {{256, 128}, {128, 128}, {128, 64}, {64, 64}};
	for (int h = 2160; h >= 720; h -= 8) {
		const double w    = std::floor(h * aspect / 8.0 + 0.5) * 8;
		bool         fits = true;
		for (const auto& tile: tiles) {
			if (std::ceil(w / tile[0]) * std::ceil(static_cast<double>(h) / tile[1]) >
			    std::ceil(3840.0 / tile[0]) * std::ceil(2160.0 / tile[1]))
				fits = false;
		}
		if (fits) return h;
	}
	return 720;
}

// The heights the ratio allows (with DLSS or FSR 3 those its output takes: 1080 and up or 720 for
// 4K, up to 1080 for 1080p; below 720 with the 1080p output only).
std::vector<int> LauncherWindow::AllowedRenderHeights() const {
	const int        most  = MaxRenderHeight();
	const bool       sr    = RenderForUpscaler();
	const int        lines = OutputLines();
	const bool       hd    = lines == 1080;
	std::vector<int> allowed;
	if (lines == 1440) {
		for (const int h: Output1440Heights)
			if (h <= most) allowed.push_back(h);
		return allowed;
	}
	for (const int h: RenderHeights)
		if (h <= most && (!sr || (hd ? h <= 1080 : h >= 1080 || h == 720))) allowed.push_back(h);
	if (hd)
		for (const int h: LowRenderHeights)
			if (sr || h >= 540) allowed.push_back(h);
	return allowed;
}

// The one last chosen for this upscaler, a higher one as the highest allowed, an unknown one as the
// upscaler's default (DLSS 1080, the game's TAA 1440).
int LauncherWindow::SelectedRender() const {
	const auto allowed = AllowedRenderHeights();
	if (allowed.empty()) return RenderForUpscaler() ? 1080 : 1440;
	const int wanted = RenderForUpscaler() ? wanted_sr_ : wanted_taa_;
	if (wanted > MaxRenderHeight()) return allowed.front();
	for (const int candidate: {wanted, RenderForUpscaler() ? 1080 : 1440})
		if (std::find(allowed.begin(), allowed.end(), candidate) != allowed.end()) return candidate;
	return allowed.front();
}

// The DLSS mode a render height runs in: the emulator picks it from the height's share of the
// game's output (95% DLAA, 62% Quality, 55% Balanced, 45% Performance, less Ultra Performance);
// from 72% named Ultra Quality (77% the usual share), DLSS's Quality mode with more lines.
std::string LauncherWindow::SuperResolutionMode(int height) const {
	const double share  = static_cast<double>(height) / OutputLines();
	const char*  native = commands_.FsrUpscaler() ? "fsr.native" : "dlss.dlaa";
	return T(share >= 0.95   ? native
	         : share >= 0.72 ? "dlss.ultraquality"
	         : share >= 0.62 ? "dlss.quality"
	         : share >= 0.55 ? "dlss.balanced"
	         : share >= 0.45 ? "dlss.performance"
	                         : "dlss.ultra");
}

std::string LauncherWindow::RenderLabel(int height) const {
	if (RenderForUpscaler()) return Paren(std::to_string(height), SuperResolutionMode(height));
	switch (height) {
		case 2160: return Paren("2160", T("render.native"));
		case 1512: return Paren("1512", T("render.pro"));
		case 1440: return Paren("1440", T("render.ps5"));
		default: return std::to_string(height);
	}
}

std::string LauncherWindow::RenderNote() const {
	int       h     = SelectedRender();
	const int lines = OutputLines();
	// (The 1080p and 1440p outputs' 3D is at most their height.)
	const bool capped = lines < 2160 && h > lines;
	if (capped) h = lines;
	// (The PS5 Pro path's 4K output takes 1080 at least, or 720 for its Ultra Performance: a third
	// of the output; the 1440p output 720 at least, or 480.)
	const int  least  = lines == 1440 ? 720 : 1080;
	const bool raised = RenderForUpscaler() && lines != 1080 && h < least && h != lines / 3;
	if (raised) h = least;
	const int w = static_cast<int>(std::floor(h * RatioAspect() / 8.0 + 0.5) * 8);
	return F(capped   ? (lines == 1440 ? "render.capped1440" : "render.capped")
	         : raised ? (lines == 1440 ? "render.raised1440" : "render.raised")
	                  : "render.desc",
	         {std::to_string(w), std::to_string(h)});
}

void LauncherWindow::GraphicsPage() {
	PageTitle("graphics.title");
	Section("graphics.resolution");
	// (DLSS where it runs - an NVIDIA RTX card with its runtime built -, FSR 3 on any graphics
	// card: built in.)
	std::vector<std::string> upscalers {""};
	std::vector<std::string> upscaler_labels {T("upscaler.game")};
	if (dlss_available_) {
		upscalers.emplace_back("dlss");
		upscaler_labels.emplace_back(T("upscaler.dlss"));
	}
	upscalers.emplace_back("fsr");
	upscaler_labels.emplace_back(T("upscaler.fsr"));
	// (No upscaler and no anti-aliasing: Commands' KYTY_GAME_CONVARS, issue #220.)
	upscalers.emplace_back("off");
	upscaler_labels.emplace_back(T("upscaler.off"));
	const auto  chosen = S().Str("upscaler");
	std::string note;
	if (chosen == "fsr")
		note = T("upscaler.fsrNote");
	else if (chosen == "off")
		note = T("upscaler.offNote");
	else
		note = T(dlss_available_ ? "upscaler.ok"
		         : !context_.rtx ? "upscaler.noRtx"
		                         : "upscaler.noLibrary");
	Row(
	    "upscaler",
	    [&] {
		    int index =
		        std::max(0, static_cast<int>(std::find(upscalers.begin(), upscalers.end(), chosen) -
		                                     upscalers.begin()));
		    if (index >= static_cast<int>(upscalers.size())) index = 0;
		    if (Combo("##upscaler", &index, upscaler_labels, 300)) {
			    S().Set("upscaler", upscalers[static_cast<size_t>(index)]);
			    Changed();
		    }
	    },
	    note);

	const auto allowed  = AllowedRenderHeights();
	const int  selected = SelectedRender();
	// (The effective height is what is saved, as the Windows launcher's form saves it.)
	if (S().Int(RenderForUpscaler() ? "render_dlss" : "render") != selected) {
		S().Set(RenderForUpscaler() ? "render_dlss" : "render", selected);
	}
	Row(
	    "render",
	    [&] {
		    std::vector<std::string> labels;
		    for (const int h: allowed)
			    labels.push_back(RenderLabel(h));
		    int index = static_cast<int>(std::find(allowed.begin(), allowed.end(), selected) -
		                                 allowed.begin());
		    if (Combo("##render", &index, labels, 300)) {
			    const int h = allowed[static_cast<size_t>(index)];
			    (RenderForUpscaler() ? wanted_sr_ : wanted_taa_) = h;
			    S().Set(RenderForUpscaler() ? "render_dlss" : "render", h);
			    Changed();
		    }
	    },
	    RenderNote());

	const std::vector<std::string> outputs {"auto", "4k", "1080p", "1440p"};
	Row(
	    "output",
	    [&] {
		    int index = static_cast<int>(
		        std::find(outputs.begin(), outputs.end(), S().Str("output")) - outputs.begin());
		    if (index >= 4) index = 0;
		    if (Combo("##output", &index,
		              {T("output.auto"), T("output.4k"), T("output.1080"), T("output.1440")},
		              300)) {
			    S().Set("output", outputs[static_cast<size_t>(index)]);
			    Changed();
		    }
	    },
	    std::string(T("output.desc")) +
	        (S().Str("output") == "auto"
	             ? " " + F("output.autoNow", {HdOutputChosen() ? "1080p" : "4K"})
	         : S().Str("output") == "1440p" && !RenderForUpscaler()
	             ? " " + std::string(T("output.1440NoUpscaler"))
	             : ""));

	// The DLSS model (KYTY_DLSS_PRESET): the DLSS version's own choice, its transformer (K; DLSS 4.5's M and L) or
	// its CNN model.
	const std::vector<std::string> presets {"", "K", "M", "L", "E"};
	Row(
	    "dlssmodel",
	    [&] {
		    int index =
		        static_cast<int>(std::find(presets.begin(), presets.end(), S().Str("dlss_preset")) -
		                         presets.begin());
		    if (index >= 5) index = 0;
		    if (Combo("##dlssmodel", &index,
		              {T("dlssmodel.auto"), T("dlssmodel.k"), T("dlssmodel.m"), T("dlssmodel.l"), T("dlssmodel.e")}, 300,
		              commands_.Dlss())) {
			    S().Set("dlss_preset", presets[static_cast<size_t>(index)]);
			    Changed();
		    }
	    },
	    T("dlssmodel.desc"));

	// The upscaler's sharpening (KYTY_DLSS_SHARPNESS, KYTY_FSR_SHARPNESS): one row for the chosen
	// upscaler, each with a value of its own: DLSS none by default (the game's own TAA sharpens,
	// DLSS does not: issue #224), FSR 3 its 0.6 (issue #227).
	const std::vector<std::string> sharpnesses {"0", "0.3", "0.6", "0.9"};
	Row(
	    "sharpen",
	    [&] {
		    const bool        fsr      = commands_.FsrUpscaler();
		    const std::string key      = fsr ? "fsr_sharpness" : "dlss_sharpness";
		    const std::string fallback = fsr ? "0.6" : "0";
		    const auto        stored   = S().Str(key);
		    const auto        position = [&](const std::string& value) {
                return static_cast<int>(std::find(sharpnesses.begin(), sharpnesses.end(), value) - sharpnesses.begin());
		    };
		    int index = position(stored.empty() ? fallback : stored);
		    if (index >= 4) index = position(fallback);
		    if (Combo("##sharpen", &index,
		              {T("sharpen.off"), T("sharpen.low"), T("sharpen.medium"), T("sharpen.high")}, 300,
		              commands_.Dlss() || fsr)) {
			    const auto& chosen = sharpnesses[static_cast<size_t>(index)];
			    S().Set(key, chosen == fallback ? std::string() : chosen);
			    Changed();
		    }
	    },
	    T("sharpen.desc"));

	Section("graphics.detail");
	const std::vector<std::string> qualities {"standard", "low", "lowest"};
	Row(
	    "quality",
	    [&] {
		    int index = static_cast<int>(
		        std::find(qualities.begin(), qualities.end(), Lower(S().Str("quality"))) -
		        qualities.begin());
		    if (index >= 3) index = 0;
		    if (Combo("##quality", &index,
		              {T("quality.standard"), T("quality.low"), T("quality.lowest")}, 300)) {
			    S().Set("quality", qualities[static_cast<size_t>(index)]);
			    Changed();
		    }
	    },
	    T("quality.desc"));

	Section("graphics.fg");
	Row(
	    "framegen",
	    [&] {
		    std::vector<std::string> labels {T("framegen.off")};
		    for (int x = 2; x <= 6; ++x)
			    labels.push_back(std::to_string(x) + "x");
		    const int value = S().Int("framegen");
		    int       index = dlss_available_ && value >= 2 ? value - 1 : 0;
		    // (Kept as it was where DLSS's runtime is missing.)
		    if (Combo("##framegen", &index, labels, 160, dlss_available_)) {
			    S().Set("framegen", index > 0 ? index + 1 : 0);
			    Changed();
		    }
	    },
	    T(dlss_available_ ? "framegen.ok"
	      : !context_.rtx ? "framegen.noRtx"
	                      : "framegen.noLibrary"));
	Row(
	    "dlssruntime",
	    [&] {
		    if (FlatButton(T("button.getdlss"), !context_.dlss_script.empty()))
			    StartInTerminal(commands_.GetDlss());
		    ImGui::SameLine();
		    ImGui::AlignTextToFramePadding();
		    const auto        library = context_.DlssLibrary();
		    const std::string state =
		        context_.dlss_script.empty() && library.empty() ? T("dlssruntime.noScript")
		        : library.empty()                               ? T("dlssruntime.none")
		                          : F("dlssruntime.found", {library.parent_path().string()});
		    ImGui::PushStyleColor(ImGuiCol_Text, V(library.empty() ? TextDimColor : GoodColor));
		    ImGui::TextUnformatted(state.c_str());
		    ImGui::PopStyleColor();
	    },
	    T("dlssruntime.desc"));
	Row(
	    nullptr,
	    [&] {
		    SettingSwitch("##fsr", "fsr");
		    ImGui::SameLine(0, 12.0f * scale_);
		    ImGui::AlignTextToFramePadding();
		    ImGui::PushTextWrapPos(0.0f);
		    ImGui::TextUnformatted(T("fsr"));
		    ImGui::PopTextWrapPos();
	    },
	    T("fsr.desc"));
}

// ------------------------------------------------------------ Game

void LauncherWindow::GamePage() {
	PageTitle("game.title");
	Section("game.console");
	Row(
	    "language",
	    [&] {
		    int index = S().Int("language");
		    if (Combo("##language", &index, Strings::ConsoleLanguages(), 340)) {
			    S().Set("language", index);
			    Changed();
		    }
	    },
	    T("language.desc"));
	// What the game is told its sound goes to (KYTY_AUDIO_OUTPUT, issue #207): it mixes for it.
	const std::vector<std::string> audio_outputs {"", "headphones", "amplifier", "soundbar"};
	Row(
	    "audiooutput",
	    [&] {
		    int index = static_cast<int>(
		        std::find(audio_outputs.begin(), audio_outputs.end(), S().Str("audio_output")) - audio_outputs.begin());
		    if (index >= 4) index = 0;
		    if (Combo("##audiooutput", &index,
		              {T("audiooutput.tv"), T("audiooutput.headphones"), T("audiooutput.amplifier"),
		               T("audiooutput.soundbar")},
		              340)) {
			    S().Set("audio_output", audio_outputs[static_cast<size_t>(index)]);
			    Changed();
		    }
	    },
	    T("audiooutput.desc"));
	Section("game.cheats");
	Row(
	    "cheats",
	    [&] {
		    SettingSwitch("##cheats", "cheats");
		    ImGui::SameLine(0, 16.0f * scale_);
		    if (FlatButton(T("button.cheats"))) OpenCheats();
		    ImGui::SameLine(0, 14.0f * scale_);
		    ImGui::AlignTextToFramePadding();
		    std::string state;
		    if (!game_.Found()) {
			    state = T("cheats.noGame");
		    } else if (const auto file = game_.CheatFile(context_.root); file.empty()) {
			    state = T("cheats.noFile");
		    } else {
			    const auto cheats = CheatFile::Load(file);
			    const auto on     = std::count_if(cheats.mods.begin(), cheats.mods.end(),
			                                      [](const CheatMod& m) { return m.enabled; });
			    state = F("cheats.count", {std::to_string(on), std::to_string(cheats.mods.size())});
		    }
		    ImGui::PushStyleColor(ImGuiCol_Text, V(TextDimColor));
		    ImGui::TextUnformatted(state.c_str());
		    ImGui::PopStyleColor();
	    },
	    T("cheats.desc"));
}

// ------------------------------------------------------------ Shaders

void LauncherWindow::ShadersPage() {
	PageTitle("shaders.title");
	const char* state = nullptr;
	switch (store_) {
		case StoreState::Checking: state = "shaders.checking"; break;
		case StoreState::Current: state = "shaders.current"; break;
		case StoreState::Outdated: state = "shaders.outdated"; break;
		case StoreState::OutdatedAll: state = "shaders.outdatedAll"; break;
		case StoreState::None: state = "shaders.none"; break;
		default: state = game_.Found() ? "shaders.unknown" : "cheats.noGame"; break;
	}
	ImGui::PushTextWrapPos(0.0f);
	ImGui::TextUnformatted(T(state));
	ImGui::PopTextWrapPos();
	Section("shaders.precompile");
	// (precompile-windows.ps1's estimate: 55 minutes on 22 processors.)
	const int minutes =
	    static_cast<int>(std::ceil(55.0 * 22 / std::max(1, Platform::ProcessorCount()) / 10) * 10);
	char hours[16];
	std::snprintf(hours, sizeof(hours), "%.1f", minutes / 60.0);
	const auto duration = minutes < 90 ? F("duration.minutes", {std::to_string(minutes)})
	                                   : F("duration.hours", {hours});
	Row(
	    nullptr,
	    [&] {
		    if (FlatButton(T("button.precompile"), !context_.precompile_script.empty()))
			    StartInTerminal(commands_.Precompile(""));
	    },
	    F("precompile.desc", {duration}));
	Row(nullptr, [&] {
		SettingSwitch("##quick", "precompile_quick");
		ImGui::SameLine(0, 12.0f * scale_);
		ImGui::AlignTextToFramePadding();
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextUnformatted(T("precompile.quick"));
		ImGui::PopTextWrapPos();
	});
	Row(
	    nullptr,
	    [&] {
		    const bool hybrid = !context_.efficiency_cpus.empty();
		    bool       value  = S().Bool("ecores") && hybrid;
		    if (Switch("##ecores", &value, hybrid)) {
			    S().Set("ecores", value);
			    Changed();
		    }
		    ImGui::SameLine(0, 12.0f * scale_);
		    ImGui::AlignTextToFramePadding();
		    ImGui::PushTextWrapPos(0.0f);
		    ImGui::TextUnformatted(T("ecores"));
		    ImGui::PopTextWrapPos();
	    },
	    context_.efficiency_cpus.empty() ? T("ecores.none") : "");
	Section("shaders.share");
	Row(
	    nullptr,
	    [&] {
		    if (FlatButton(T("button.hints"), !context_.precompile_script.empty()))
			    StartInTerminal(commands_.Precompile("--export-hints"));
	    },
	    T("hints.desc"));
}

// ------------------------------------------------------------ Advanced

void LauncherWindow::AdvancedPage() {
	PageTitle("advanced.title");
	Section("advanced.stability");
	Row(
	    "hud",
	    [&] {
		    int index = std::clamp(S().Int("hud"), 0, 2);
		    if (Combo("##hud", &index, {T("hud.off"), T("hud.rate"), T("hud.detailed")}, 300)) {
			    S().Set("hud", index);
			    Changed();
		    }
	    },
	    T("hud.desc"));
	Row(
	    "mangohud",
	    [&] {
		    bool value = S().Bool("mangohud") && context_.HasMangoHud();
		    if (Switch("##mangohud", &value, context_.HasMangoHud())) {
			    S().Set("mangohud", value);
			    Changed();
		    }
	    },
	    T(context_.HasMangoHud() ? "mangohud.desc" : "mangohud.missing"));
	Row("safegpu", [&] { SettingSwitch("##safegpu", "safegpu"); }, T("safegpu.desc"));
	if (context_.gpus.size() > 1) {
		Section("advanced.gpu");
		Row(
		    "gpu",
		    [&] {
			    std::vector<std::string> labels {T("gpu.auto")};
			    int                      index = 0;
			    for (size_t i = 0; i < context_.gpus.size(); ++i) {
				    labels.push_back(context_.gpus[i].name);
				    if (context_.gpus[i].name == S().Str("gpu")) index = static_cast<int>(i) + 1;
			    }
			    if (Combo("##gpu", &index, labels, 360)) {
				    S().Set("gpu", index == 0 ? std::string()
				                              : context_.gpus[static_cast<size_t>(index) - 1].name);
				    Changed();
			    }
		    },
		    T("gpu.desc"));
	}
	Section("advanced.extra");
	Row(
	    "extra",
	    [&] {
		    ImGui::SetNextItemWidth(-1);
		    if (ImGui::InputTextWithHint("##extra", "KYTY_DLSS_PRESET=E --vblank-frequency 60",
		                                 &extra_text_)) {
			    S().Set("extra_args", Trim(extra_text_));
			    Changed();
		    }
	    },
	    T("extra.desc"));
	Section("advanced.launcher");
	Row("uilang", [&] {
		const std::vector<std::string> codes {"", "en", "zh", "pt"};
		int index = static_cast<int>(std::find(codes.begin(), codes.end(), S().Str("ui_language")) -
		                             codes.begin());
		if (index >= 4) index = 0;
		if (Combo("##uilang", &index,
		          {T("uilang.auto"), "English", "中文（简体）", "Português (Brasil)"}, 240)) {
			S().Set("ui_language", codes[static_cast<size_t>(index)]);
			Strings::Current = Strings::FromCode(codes[static_cast<size_t>(index)]);
			Changed();
		}
	});
	Row("uiscale", [&] {
		const std::vector<int> scales {0, 100, 125, 150, 175, 200};
		int index = static_cast<int>(std::find(scales.begin(), scales.end(), S().Int("ui_scale")) -
		                             scales.begin());
		if (index >= static_cast<int>(scales.size())) index = 0;
		if (Combo("##uiscale", &index, {T("uiscale.auto"), "100%", "125%", "150%", "175%", "200%"},
		          240)) {
			S().Set("ui_scale", scales[static_cast<size_t>(index)]);
			Changed();
			scale_changed_ = true;
		}
	});
	Row(
	    "terminal",
	    [&] {
		    std::vector<std::string> labels {T("terminal.auto")};
		    int                      index = 0;
		    for (size_t i = 0; i < context_.terminals.size(); ++i) {
			    labels.push_back(context_.terminals[i].name);
			    if (context_.terminals[i].name == S().Str("terminal"))
				    index = static_cast<int>(i) + 1;
		    }
		    if (Combo("##terminal", &index, labels, 240, !context_.terminals.empty())) {
			    S().Set("terminal", index == 0
			                            ? std::string()
			                            : context_.terminals[static_cast<size_t>(index) - 1].name);
			    Changed();
		    }
	    },
	    T(context_.terminals.empty() ? "terminal.none" : "terminal.desc"));
	Row(nullptr, [&] {
		// (Side by side while they fit, else on the next line.)
		bool first = true;
		auto next  = [&](const char* label) {
            const float width = ImGui::CalcTextSize(label).x + 28.0f * scale_;
            if (!first && ImGui::GetContentRegionAvail().x - ImGui::GetItemRectMax().x +
                                  ImGui::GetCursorScreenPos().x >
                              width + 10.0f * scale_)
                ImGui::SameLine(0, 10.0f * scale_);
            first = false;
		};
		next(T("button.logs"));
		if (FlatButton(T("button.logs"))) {
			std::error_code error;
			fs::create_directories(context_.LogsFolder(), error);
			Platform::Open(context_.LogsFolder().string());
		}
		next(T("button.readme"));
		if (FlatButton(T("button.readme"), !context_.Readme().empty()))
			Platform::Open(context_.Readme().string());
		next(T("button.root"));
		if (FlatButton(T("button.root"))) Platform::Open(context_.root.string());
		next(T("button.desktop"));
		if (FlatButton(T("button.desktop"))) {
			// (The game's own icon from its folder, when it has one.)
			std::error_code error;
			const fs::path  icon = fs::path(game_.path) / "sce_sys" / "icon0.png";
			std::string     reason;
			const auto      file = Platform::WriteDesktopEntry(
                context_.root, !game_.path.empty() && fs::exists(icon, error) ? icon : fs::path(),
                &reason);
			Message(file.empty() ? F("desktop.failed", {reason})
			                     : F("desktop.done", {file.string()}));
		}
	});
	Desc(T("credit"));
}

// ------------------------------------------------------------ Experimental

// Options still being tried out (not defaults yet): a player turns one on, and its issue collects
// how it went.
void LauncherWindow::ExperimentalPage() {
	PageTitle("experimental.title");
	Desc(T("experimental.intro"));
	Section("experimental.framerate");
	Row("fps30", [&] { SettingSwitch("##fps30", "fps30"); }, T("fps30.desc"));
	Row(nullptr, [&] { Feedback(85); });
	Section("experimental.display");
	Row(
	    "presentdirect", [&] { SettingSwitch("##presentdirect", "present_direct"); },
	    T("presentdirect.desc"));
	Row(nullptr, [&] { Feedback(37); });
	Row("vrrflips", [&] { SettingSwitch("##vrrflips", "vrr_flips"); }, T("vrrflips.desc"));
	Row(nullptr, [&] { Feedback(221); });
	Section("experimental.amd");
	Row(
	    "antilag",
	    [&] {
		    bool value = S().Bool("antilag") && context_.amd;
		    if (Switch("##antilag", &value, context_.amd)) {
			    S().Set("antilag", value);
			    Changed();
		    }
	    },
	    T(context_.amd ? "antilag.desc" : "antilag.notAmd"));
	Row(nullptr, [&] { Feedback(80); });
}

// ------------------------------------------------------------ summary

std::string LauncherWindow::Summary(std::vector<std::pair<std::string, std::string>>* lines) const {
	const auto res         = Times(S().Str("resolution"));
	const bool dlss        = commands_.Dlss();
	const auto upscaled_by =
	    std::string(T(dlss                         ? "summary.dlss"
	                : commands_.FsrUpscaler()      ? "summary.fsr3"
	                : S().Str("upscaler") == "off" ? "summary.noAa"
	                                               : "summary.taa"));
	const bool hd          = commands_.HdOutput();
	const bool sr          = dlss || commands_.FsrUpscaler();
	int        height      = sr ? S().Int("render_dlss") : S().Int("render");
	const bool output1440  = !hd && commands_.Output1440();
	if (hd && height > 1080) height = 1080;
	if (output1440 && height > 1440) height = 1440;
	// (The PS5 Pro path's 4K output takes 1080 at least, or 720, as its render height list; the
	// 1440p output 720, or 480.)
	if (sr && !hd && !output1440 && height < 1080 && height != 720) height = 1080;
	if (output1440 && height < 720 && height != 480) height = 720;
	std::string frame_generation;
	if (S().Bool("fsr"))
		frame_generation = T("summary.fsr");
	else if (dlss_available_ && S().Int("framegen") >= 2)
		frame_generation = "DLSS " + std::to_string(S().Int("framegen")) + "x";
	const auto quality_key = "quality." + (S().Str("quality").empty() ? std::string("standard")
	                                                                  : Lower(S().Str("quality")));
	std::vector<std::string> options;
	if (S().Bool("vsync")) options.emplace_back(T("summary.vsync"));
	if (S().Bool("fps30"))
		options.emplace_back(T("summary.fps30"));
	else if (S().Bool("fps120"))
		options.emplace_back(T("summary.fps120"));
	if (S().Bool("lowlatency")) options.emplace_back(T("lowlatency"));
	if (S().Bool("hdr")) options.emplace_back(T("summary.hdr"));
	if (S().Bool("wide_bars") && S().Str("ratio") != "16:9") options.emplace_back(T("summary.wideBars"));
	if (hd) options.emplace_back(T("summary.hd"));
	if (output1440) options.emplace_back(T("summary.1440"));
	if (S().Bool("present_direct")) options.emplace_back(T("summary.presentDirect"));
	if (S().Bool("vrr_flips")) options.emplace_back(T("summary.vrrFlips"));
	if (S().Bool("antilag")) options.emplace_back(T("summary.antilag"));
	if (S().Bool("mangohud") && context_.HasMangoHud()) options.emplace_back(T("summary.mangohud"));
	if (!S().Str("extra_args").empty()) options.emplace_back(T("summary.extra"));
	auto join = [](const std::vector<std::string>& parts, const char* separator) {
		std::string out;
		for (const auto& part: parts)
			out += (out.empty() ? "" : separator) + part;
		return out;
	};
	const auto window = F(S().Bool("fullscreen") ? "summary.fullscreen" : "summary.window", {res});
	const auto render = F("summary.render", {std::to_string(height)});
	if (lines != nullptr) {
		*lines = {
		    {T("nav.display"), window + "  ·  " + S().Str("ratio")},
		    {"3D", render + "  ·  " + upscaled_by},
		    {T("graphics.fg"),
		     frame_generation.empty() ? std::string(T("framegen.off")) : frame_generation},
		    {T("quality"), T(quality_key.c_str())},
		    {T("summary.options"), options.empty() ? std::string("—") : join(options, "  ·  ")},
		};
	}
	std::vector<std::string> parts {window, S().Str("ratio"), render, upscaled_by};
	if (!frame_generation.empty()) parts.push_back(frame_generation);
	parts.insert(parts.end(), options.begin(), options.end());
	return join(parts, "   ·   ");
}

// ------------------------------------------------------------ actions

void LauncherWindow::Changed() {
	// (Saved at the end of the frame.)
}

void LauncherWindow::Message(const std::string& text) {
	message_      = text;
	message_open_ = true;
}

void LauncherWindow::StartInTerminal(const Launch& launch) {
	if (launch.argv.empty() || launch.argv[0].empty()) return;
	S().Save();
	std::string error;
	if (context_.terminals.empty()) {
		Message(T("terminal.none"));
		return;
	}
	if (!commands_.StartInTerminal(launch, &error)) {
		Message(F("error.start", {error}));
		return;
	}
	notice_    = F("run.terminal", {context_.ChosenTerminal()->name});
	notice_at_ = std::chrono::steady_clock::now();
}

void LauncherWindow::Play() {
	S().Set("game", Trim(game_text_));
	S().Save();
	const auto  launch = commands_.Play(game_, false);
	std::string error;
	// The game's log in a terminal window (as run.cmd's console on Windows); without a terminal
	// only in its log file.
	const bool started = !context_.terminals.empty() ? commands_.StartInTerminal(launch, &error)
	                                                 : commands_.RunDirect(launch, false) == 0;
	if (!started) {
		Message(F("error.start", {error.empty() ? std::string("bash") : error}));
		return;
	}
	close_ = true;
}

void LauncherWindow::LoadGame(const std::string& path) {
	const auto trimmed = Trim(path);
	if (trimmed != S().Str("game")) {
		S().Set("game", trimmed);
		Changed();
	}
	if (banner_ != nullptr) {
		SDL_DestroyTexture(banner_);
		banner_ = nullptr;
	}
	store_job_ = nullptr;
	// (A .zar's param.json comes from the precompile program, off this thread; a folder's at once.)
	auto job   = std::make_shared<Job<GameInfo>>();
	game_job_  = job;
	game_      = GameInfo {};
	game_.path = trimmed;
	std::thread([job, trimmed, tool = context_.precompile_tool, root = context_.root] {
		job->value = GameInfo::Read(trimmed, tool, root);
		job->done  = true;
		Wake();
	}).detach();
}

void LauncherWindow::LoadBanner() {
	if (game_.picture.empty() || renderer_ == nullptr) return;
	int  w = 0, h = 0, channels = 0;
	auto pixels = stbi_load(game_.picture.c_str(), &w, &h, &channels, 4);
	if (pixels == nullptr) return;
	SDL_Texture* texture =
	    SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
	if (texture != nullptr) {
		SDL_UpdateTexture(texture, nullptr, pixels, w * 4);
		SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
		SDL_SetTextureScaleMode(texture, SDL_ScaleModeBest);
		banner_   = texture;
		banner_w_ = w;
		banner_h_ = h;
	}
	stbi_image_free(pixels);
}

// The precompiled shaders' state, asked of the precompile program (at most every 30 s: also when
// the window comes back to the front after a precompile).
void LauncherWindow::CheckStore(bool force) {
	if (!game_.Found()) {
		store_ = StoreState::Unknown;
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (!force && (store_job_ != nullptr || now - store_checked_ < std::chrono::seconds(30)))
		return;
	store_checked_ = now;
	if (store_job_ == nullptr && store_ == StoreState::Unknown) store_ = StoreState::Checking;
	auto job   = std::make_shared<Job<StoreState>>();
	store_job_ = job;
	std::thread([job, root = context_.root, tool = context_.precompile_tool, game = game_] {
		job->value = CheckShaderStore(root, tool, game);
		job->done  = true;
		Wake();
	}).detach();
}

void LauncherWindow::Activated() {
	CheckStore(false);
	dlss_available_ = context_.DlssAvailable();
}

void LauncherWindow::Poll() {
	if (game_edit_pending_ &&
	    std::chrono::steady_clock::now() - game_edit_ > std::chrono::milliseconds(450)) {
		game_edit_pending_ = false;
		LoadGame(game_text_);
	}
	if (game_job_ != nullptr && game_job_->done) {
		game_     = game_job_->value;
		game_job_ = nullptr;
		LoadBanner();
		store_ = StoreState::Unknown;
		CheckStore(true);
	}
	if (store_job_ != nullptr && store_job_->done) {
		store_     = store_job_->value;
		store_job_ = nullptr;
	}
	if (picker_job_ != nullptr && picker_job_->done) {
		const auto chosen = Trim(picker_job_->value);
		picker_job_       = nullptr;
		if (!chosen.empty()) {
			game_text_ = chosen;
			LoadGame(chosen);
		}
	}
	// (DLSS's runtime appears while the window is open: get-dlss-linux.sh ran in its terminal.)
	static auto last = std::chrono::steady_clock::now();
	if (std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
		last            = std::chrono::steady_clock::now();
		dlss_available_ = context_.DlssAvailable();
	}
}

// ------------------------------------------------------------ dialogs

// The desktop's own dialog (kdialog on KDE, zenity or qarma elsewhere), off the window's thread;
// the built-in browser without one.
void LauncherWindow::PickFolder(bool zar) {
	if (picker_job_ != nullptr) return;
	const auto      current = Trim(game_text_);
	std::error_code error;
	fs::path        start = current.empty() ? fs::path() : fs::path(current);
	if (!start.empty() && !fs::is_directory(start, error)) start = start.parent_path();
	if (start.empty() || !fs::is_directory(start, error)) {
		const char* home = std::getenv("HOME");
		start            = home != nullptr ? home : "/";
	}
	std::string desktop;
	if (const char* value = std::getenv("XDG_CURRENT_DESKTOP"); value != nullptr)
		desktop = Lower(value);
	const bool               kde   = desktop.find("kde") != std::string::npos;
	const auto               title = std::string(T(zar ? "dialog.zar" : "dialog.game"));
	std::vector<std::string> argv;
	auto                     zenity = [&](const std::string& program) {
        argv = {program, "--file-selection", "--title=" + title,
                "--filename=" + (start / "").string()};
        if (zar)
            argv.emplace_back("--file-filter=ZArchive (*.zar) | *.zar");
        else
            argv.emplace_back("--directory");
	};
	auto kdialog = [&] {
		if (zar)
			argv = {"kdialog",           "--title",      title,
			        "--getopenfilename", start.string(), "*.zar|ZArchive (*.zar)"};
		else
			argv = {"kdialog", "--title", title, "--getexistingdirectory", start.string()};
	};
	if (kde && Platform::HasProgram("kdialog"))
		kdialog();
	else if (Platform::HasProgram("zenity"))
		zenity("zenity");
	else if (Platform::HasProgram("qarma"))
		zenity("qarma");
	else if (Platform::HasProgram("kdialog"))
		kdialog();
	if (argv.empty()) {
		browser_open_     = true;
		browser_zar_      = zar;
		browser_path_     = start.string();
		browser_selected_ = "";
		return;
	}
	auto job    = std::make_shared<Job<std::string>>();
	picker_job_ = job;
	std::thread([job, argv] {
		std::string output;
		if (Platform::Run(argv, &output, 24 * 3600 * 1000) == 0) job->value = output;
		job->done = true;
		Wake();
	}).detach();
}

void LauncherWindow::OpenCheats() {
	const auto file = game_.CheatFile(context_.root);
	if (file.empty()) {
		Message(T(game_.Found() ? "cheats.noFile" : "cheats.noGame"));
		return;
	}
	auto cheats = CheatFile::Load(file);
	if (!cheats.error.empty()) {
		Message(F("cheats.unreadable", {file.string(), cheats.error}));
		return;
	}
	if (cheats.mods.empty()) {
		Message(F("cheats.empty", {file.string()}));
		return;
	}
	cheat_path_  = file;
	cheat_file_  = std::move(cheats);
	cheats_open_ = true;
	cheat_hover_ = -1;
}

void LauncherWindow::Modals() {
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	const ImVec2         center   = viewport->GetCenter();

	if (message_open_) {
		ImGui::OpenPopup("##message");
		message_open_ = false;
	}
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({520.0f * scale_, 0});
	if (ImGui::BeginPopupModal("##message", nullptr,
	                           ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
	                               ImGuiWindowFlags_NoMove)) {
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextUnformatted(message_.c_str());
		ImGui::PopTextWrapPos();
		ImGui::Dummy({0, 8.0f * scale_});
		ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 120.0f * scale_);
		if (ImGui::Button(T("ok"), {100.0f * scale_, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape) ||
		    ImGui::IsKeyPressed(ImGuiKey_Enter))
			ImGui::CloseCurrentPopup();
		ImGui::EndPopup();
	}

	// The cheats of the game's file: their flags.
	if (cheats_open_) {
		ImGui::OpenPopup("##cheats");
		cheats_open_ = false;
	}
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::min(640.0f * scale_, viewport->Size.x - 40),
	                          std::min(560.0f * scale_, viewport->Size.y - 40)});
	if (ImGui::BeginPopupModal("##cheats", nullptr,
	                           ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
	                               ImGuiWindowFlags_NoMove)) {
		ImGui::PushFont(serif_, 22.0f);
		ImGui::TextUnformatted(F("cheatsWindow", {cheat_file_.title}).c_str());
		ImGui::PopFont();
		const float footer =
		    ImGui::GetFrameHeightWithSpacing() * 2 + ImGui::GetTextLineHeight() * 3;
		ImGui::BeginChild("##mods", {0, -footer}, ImGuiChildFlags_Borders);
		for (size_t i = 0; i < cheat_file_.mods.size(); ++i) {
			ImGui::PushID(static_cast<int>(i));
			auto& mod = cheat_file_.mods[i];
			ImGui::Checkbox(mod.name.c_str(), &mod.enabled);
			if (ImGui::IsItemHovered() || ImGui::IsItemFocused())
				cheat_hover_ = static_cast<int>(i);
			ImGui::PopID();
		}
		ImGui::EndChild();
		ImGui::PushStyleColor(ImGuiCol_Text, V(TextDimColor));
		ImGui::PushTextWrapPos(0.0f);
		if (cheat_hover_ >= 0 && cheat_hover_ < static_cast<int>(cheat_file_.mods.size()))
			ImGui::TextUnformatted(
			    cheat_file_.mods[static_cast<size_t>(cheat_hover_)].description.c_str());
		else
			ImGui::TextUnformatted(" ");
		ImGui::PopTextWrapPos();
		ImGui::PopStyleColor();
		ImGui::SetCursorPosY(ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() -
		                     ImGui::GetStyle().WindowPadding.y);
		if (FlatButton(T("cheats.all")))
			for (auto& mod: cheat_file_.mods)
				mod.enabled = true;
		ImGui::SameLine();
		if (FlatButton(T("cheats.none")))
			for (auto& mod: cheat_file_.mods)
				mod.enabled = false;
		ImGui::SameLine(ImGui::GetWindowWidth() - 220.0f * scale_);
		if (ImGui::Button(T("ok"), {95.0f * scale_, 0})) {
			const auto result = cheat_file_.Save(cheat_path_);
			ImGui::CloseCurrentPopup();
			if (!result.empty()) Message(T(result.c_str()));
		}
		ImGui::SameLine();
		if (ImGui::Button(T("cancel"), {95.0f * scale_, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape))
			ImGui::CloseCurrentPopup();
		ImGui::EndPopup();
	}

	// The built-in folder browser (no zenity, qarma or kdialog on this PC).
	if (browser_open_) {
		ImGui::OpenPopup("##browser");
		browser_open_ = false;
	}
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::min(720.0f * scale_, viewport->Size.x - 40),
	                          std::min(560.0f * scale_, viewport->Size.y - 40)});
	if (ImGui::BeginPopupModal("##browser", nullptr,
	                           ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
	                               ImGuiWindowFlags_NoMove)) {
		ImGui::PushFont(serif_, 22.0f);
		ImGui::TextUnformatted(T(browser_zar_ ? "dialog.zar" : "dialog.game"));
		ImGui::PopFont();
		std::error_code error;
		fs::path        dir(browser_path_);
		if (FlatButton(T("picker.up"))) {
			browser_path_     = dir.parent_path().string();
			browser_selected_ = "";
		}
		ImGui::SameLine();
		if (FlatButton(T("picker.home"))) {
			const char* home  = std::getenv("HOME");
			browser_path_     = home != nullptr ? home : "/";
			browser_selected_ = "";
		}
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-1);
		ImGui::InputText("##browser-path", &browser_path_);
		const float footer = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().WindowPadding.y;
		ImGui::BeginChild("##entries", {0, -footer}, ImGuiChildFlags_Borders);
		std::vector<fs::path> dirs, files;
		for (const auto& entry:
		     fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, error)) {
			const auto name = entry.path().filename().string();
			if (name.empty() || name[0] == '.') continue;
			if (entry.is_directory(error))
				dirs.push_back(entry.path());
			else if (browser_zar_ && Lower(entry.path().extension().string()) == ".zar")
				files.push_back(entry.path());
		}
		std::sort(dirs.begin(), dirs.end());
		std::sort(files.begin(), files.end());
		fs::path enter;
		for (const auto& path: dirs) {
			const auto label = path.filename().string() + "/";
			if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) &&
			    ImGui::IsMouseDoubleClicked(0))
				enter = path;
			if (ImGui::IsItemFocused() && ImGui::IsKeyPressed(ImGuiKey_Enter)) enter = path;
		}
		for (const auto& path: files) {
			if (ImGui::Selectable(path.filename().string().c_str(),
			                      browser_selected_ == path.string(),
			                      ImGuiSelectableFlags_AllowDoubleClick)) {
				browser_selected_ = path.string();
				if (ImGui::IsMouseDoubleClicked(0)) {
					game_text_ = browser_selected_;
					LoadGame(game_text_);
					ImGui::CloseCurrentPopup();
				}
			}
		}
		ImGui::EndChild();
		if (!enter.empty()) {
			browser_path_     = enter.string();
			browser_selected_ = "";
		}
		const bool can_choose =
		    browser_zar_ ? !browser_selected_.empty() : fs::is_directory(dir, error);
		ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 330.0f * scale_);
		ImGui::BeginDisabled(!can_choose);
		if (ImGui::Button(T(browser_zar_ ? "picker.open" : "picker.choose"),
		                  {200.0f * scale_, 0})) {
			game_text_ = browser_zar_ ? browser_selected_ : dir.string();
			LoadGame(game_text_);
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndDisabled();
		ImGui::SameLine();
		if (ImGui::Button(T("cancel"), {100.0f * scale_, 0}) ||
		    ImGui::IsKeyPressed(ImGuiKey_Escape))
			ImGui::CloseCurrentPopup();
		ImGui::EndPopup();
	}
}
