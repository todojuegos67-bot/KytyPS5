// (ds-troph) The emulator's F1 menu (Volver al juego, Ver trofeos, Cerrar el juego) and its trophy page, the same as the
// PC port's (src/port/ds-menu.cpp): every trophy with its real icon (one atlas texture, Libs::Trophy::BuildArt), the
// progress, the counts per grade, filters, order and animations. Drawn by the system overlay (ImGui, Vulkan backend).
#include "libs/trophy-page.h"

#include "SDL.h"
#include "common/emulatorConfig.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "libs/trophy-cup.h"
#include "libs/trophy.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::TrophyPage {

namespace {

std::atomic<unsigned long long> g_rev {1};
void BumpRev() {
	g_rev.fetch_add(1, std::memory_order_relaxed);
}

// 0 closed, 1 the menu, 2 the trophy page (window thread writes, present thread reads and writes)
std::atomic<int> g_mode {0};
int              g_tab_delta = 0; // present thread

struct Input {
	enum Kind { Key, Analog, MousePos, MouseButton, Wheel, Tab, Reset } kind;
	int   key  = 0;
	bool  down = false;
	float x = 0, y = 0;
};
std::mutex        g_mutex;
std::deque<Input> g_inputs;
int               g_passthrough = 0; // window thread: zeroed axes sent to the game as the menu opened
void Queue(const Input& in) {
	std::scoped_lock lock(g_mutex);
	g_inputs.push_back(in);
	BumpRev();
}

// The texts: Spanish when the game's language is Spanish, else English.
const char* T(const char* es) {
	static const std::map<std::string, const char*> en = {
	    {"Volver al juego", "Back to the game"},
	    {"Ver trofeos", "View trophies"},
	    {"Cerrar el juego", "Close the game"},
	    {"F1 o Esc: volver al juego", "F1 or Esc: back to the game"},
	    {"TROFEOS", "TROPHIES"},
	    {"%d de %d trofeos obtenidos", "%d of %d trophies earned"},
	    {"Todos", "All"},
	    {"Obtenidos", "Earned"},
	    {"Sin obtener", "Not earned"},
	    {"Ordenar:", "Sort:"},
	    {"Grado", "Grade"},
	    {"Fecha", "Date"},
	    {"Volver", "Back"},
	    {"L1/R1 o Q/E: filtro   \xC2\xB7   Y o Tab: orden   \xC2\xB7   B, Esc o F1: volver",
	     "L1/R1 or Q/E: filter   \xC2\xB7   Y or Tab: sort   \xC2\xB7   B, Esc or F1: back"},
	    {"Cargando los trofeos...", "Loading the trophies..."},
	    {"No se encontraron los trofeos del juego (falta sce_sys/trophy2/trophy00.ucp).",
	     "The game's trophies were not found (sce_sys/trophy2/trophy00.ucp is missing)."},
	    {"A\xC3\xBAn no has obtenido ning\xC3\xBAn trofeo.", "You haven't earned any trophy yet."},
	    {"\xC2\xA1Has obtenido todos los trofeos!", "You have earned every trophy!"},
	    {"Trofeo oculto", "Hidden trophy"},
	    {"Se revelar\xC3\xA1 al obtenerlo.", "Revealed once earned."},
	    {"Obtenido", "Earned"},
	    {"Obtenido el %s", "Earned on %s"},
	    {"Platino", "Platinum"},
	    {"Oro", "Gold"},
	    {"Plata", "Silver"},
	    {"Bronce", "Bronze"},
	};
	const uint32_t lang = Config::GetConsoleLanguage();
	if (lang == 3 || lang == 20) return es;
	const auto it = en.find(es);
	return it == en.end() ? es : it->second;
}

ImFont* FindFont(const char* name) {
	for (ImFont* font: ImGui::GetIO().Fonts->Fonts)
		if (std::strstr(font->GetDebugName(), name) != nullptr) return font;
	return nullptr;
}

namespace Page {

struct ArtJob {
	std::mutex                          mutex;
	std::shared_ptr<Libs::Trophy::Art>  ready;
	std::atomic<bool>                   running {false};
	std::atomic<bool>                   failed {false};
};
ArtJob& Job() {
	static ArtJob job; // (never destroyed before a detached worker ends)
	return job;
}

ImTextureData*                       g_tex     = nullptr;
ImGuiContext*                        g_tex_ctx = nullptr;
std::shared_ptr<Libs::Trophy::Art>   g_art; // the slots (pixels moved into g_tex)
bool                                 g_art_requested = false;

struct Row {
	Libs::Trophy::Info info;
	bool               earned = false;
	int64_t            time   = 0;
};
std::vector<Row>   g_rows;
int                g_status       = -1; // TrySnapshot's
double             g_refreshed_at = -100.0;
std::vector<int>   g_view; // indices into g_rows: the filter and the order
int                g_filter = 0, g_sort = 0; // 0 todos, 1 obtenidos, 2 sin obtener; 0 grado, 1 fecha
int                g_sel_id      = -1;
float              g_scroll = 0, g_scroll_target = 0;
double             g_opened_at = 0, g_list_at = 0;
std::map<int, float> g_hot;
ImVec2             g_last_mouse {-1, -1};
float              g_pill_x = -1, g_pill_w = 0;

void EnsureArt(float unit) {
	auto& job = Job();
	if (g_tex != nullptr && g_tex_ctx != ImGui::GetCurrentContext()) {
		g_tex = nullptr; // (another context: its backend owns the old one)
		g_art.reset();
		g_art_requested = false;
	}
	if (g_tex != nullptr) return;
	if (!g_art_requested && !job.running.load()) {
		g_art_requested = true;
		job.failed      = false;
		job.running     = true;
		// The icons drawn at about 112 units (and up to 6 % larger when hovered): that many pixels, 128 - 256.
		const int cell = std::clamp((static_cast<int>(std::ceil(118.0f * unit)) + 15) / 16 * 16, 128, 256);
		std::thread([cell] {
			auto art = std::make_shared<Libs::Trophy::Art>();
			const bool ok = Libs::Trophy::BuildArt(cell, art.get());
			auto&      j  = Job();
			{
				std::scoped_lock lock(j.mutex);
				if (ok && !art->pixels.empty()) j.ready = std::move(art);
				else j.failed = true;
			}
			j.running = false;
			BumpRev();
		}).detach();
	}
	std::shared_ptr<Libs::Trophy::Art> art;
	{
		std::scoped_lock lock(job.mutex);
		art = std::move(job.ready);
	}
	if (art == nullptr || art->width <= 0 || art->height <= 0) return;
	auto* tex = IM_NEW(ImTextureData)();
	tex->Create(ImTextureFormat_RGBA32, art->width, art->height);
	std::memcpy(tex->GetPixels(), art->pixels.data(), static_cast<size_t>(art->width) * art->height * 4);
	tex->UseColors = true;
	ImGui::RegisterUserTexture(tex);
	art->pixels.clear();
	art->pixels.shrink_to_fit();
	g_tex     = tex;
	g_tex_ctx = ImGui::GetCurrentContext();
	g_art     = std::move(art);
}

// The icon's UV in the atlas (false: no icon for it).
bool IconUv(int slot, ImVec2* uv0, ImVec2* uv1) {
	if (g_tex == nullptr || g_art == nullptr || slot < 0) return false;
	const auto& a = *g_art;
	const float x = static_cast<float>((slot % a.cols) * a.stride + a.pad), y = static_cast<float>((slot / a.cols) * a.stride + a.pad);
	*uv0          = {x / static_cast<float>(a.width), y / static_cast<float>(a.height)};
	*uv1          = {(x + static_cast<float>(a.cell)) / static_cast<float>(a.width), (y + static_cast<float>(a.cell)) / static_cast<float>(a.height)};
	return true;
}
int SlotOf(int id) {
	if (g_art == nullptr) return -1;
	const auto it = g_art->slots.find(id);
	return it == g_art->slots.end() ? -1 : it->second;
}

int GradeRank(int g) {
	return g >= 1 && g <= 4 ? g : 5;
}

void Rebuild() {
	g_view.clear();
	for (int i = 0; i < static_cast<int>(g_rows.size()); i++) {
		const auto& r = g_rows[static_cast<size_t>(i)];
		if ((g_filter == 1 && !r.earned) || (g_filter == 2 && r.earned)) continue;
		g_view.push_back(i);
	}
	std::stable_sort(g_view.begin(), g_view.end(), [](int ia, int ib) {
		const auto& a = g_rows[static_cast<size_t>(ia)];
		const auto& b = g_rows[static_cast<size_t>(ib)];
		if (g_sort == 1) {
			if (a.earned != b.earned) return a.earned;
			if (a.earned && a.time != b.time) return a.time > b.time; // newest first
		}
		if (GradeRank(a.info.grade) != GradeRank(b.info.grade)) return GradeRank(a.info.grade) < GradeRank(b.info.grade);
		return a.info.id < b.info.id;
	});
}

void Refresh(bool force) {
	const double now = ImGui::GetTime();
	if (!force && now - g_refreshed_at < 1.0) return;
	std::vector<Libs::Trophy::Info> all;
	std::map<int, int64_t>          times;
	const int                       status = Libs::Trophy::TrySnapshot(&all, &times);
	if (status == 0) return; // busy: the last copy, again next frame
	g_refreshed_at = now;
	g_status       = status;
	std::vector<Row> rows;
	if (status == 1) {
		rows.reserve(all.size());
		for (auto& i: all) {
			Row r;
			const auto it = times.find(i.id);
			r.earned      = it != times.end();
			r.time        = r.earned ? it->second : 0;
			r.info        = std::move(i);
			rows.push_back(std::move(r));
		}
	}
	bool same = rows.size() == g_rows.size();
	for (size_t i = 0; same && i < rows.size(); i++)
		same = rows[i].info.id == g_rows[i].info.id && rows[i].earned == g_rows[i].earned && rows[i].time == g_rows[i].time;
	if (same && !force) return;
	g_rows = std::move(rows);
	Rebuild();
}

void Open() {
	g_opened_at = g_list_at = ImGui::GetTime();
	g_scroll = g_scroll_target = 0;
	g_sel_id                   = -1;
	g_hot.clear();
	g_pill_x = -1;
	Refresh(true);
}

float Ease(float t) {
	t = std::clamp(t, 0.0f, 1.0f);
	return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}
ImU32 Col(int r, int g, int b, float a) {
	return IM_COL32(r, g, b, static_cast<int>(std::clamp(a, 0.0f, 1.0f) * 255.0f));
}
ImU32 WithAlpha(ImU32 c, float a) {
	const int base = static_cast<int>((c >> IM_COL32_A_SHIFT) & 0xFF);
	return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(std::clamp(static_cast<float>(base) * a, 0.0f, 255.0f)) << IM_COL32_A_SHIFT);
}

ImFont* FontFor(const std::string& text, ImFont* latin) {
	static ImFont* cjk = nullptr;
	static bool    looked = false;
	static ImGuiContext* ctx = nullptr;
	if (!looked || ctx != ImGui::GetCurrentContext()) {
		looked = true;
		ctx    = ImGui::GetCurrentContext();
		cjk    = nullptr;
		for (const char* n: {"msyh", "simhei", "simsun"})
			if (cjk == nullptr) cjk = FindFont(n);
	}
	for (const unsigned char c: text)
		if (c >= 0xd0 && cjk != nullptr) return cjk; // (Cyrillic, CJK ...: the system's CJK face)
	return latin;
}

// The text cut to max_w with "..." (one line).
void TextFit(ImDrawList* dl, ImFont* font, float size, ImVec2 at, ImU32 col, const std::string& text, float max_w) {
	if (max_w <= 0) return;
	if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str()).x <= max_w) {
		dl->AddText(font, size, at, col, text.c_str());
		return;
	}
	const float dots = font->CalcTextSizeA(size, FLT_MAX, 0.0f, "...").x;
	size_t      lo = 0, hi = text.size();
	while (lo < hi) { // the longest prefix (on a UTF-8 boundary) that fits
		size_t mid = (lo + hi + 1) / 2;
		while (mid > 0 && mid < text.size() && (static_cast<unsigned char>(text[mid]) & 0xC0) == 0x80) mid--;
		if (mid <= lo) {
			hi = lo;
			break;
		}
		if (font->CalcTextSizeA(size, FLT_MAX, 0.0f, text.c_str(), text.c_str() + mid).x + dots <= max_w) lo = mid;
		else hi = mid - 1;
	}
	while (lo > 0 && lo < text.size() && (static_cast<unsigned char>(text[lo]) & 0xC0) == 0x80) lo--;
	const std::string cut = text.substr(0, lo) + "...";
	dl->AddText(font, size, at, col, cut.c_str());
}

// A padlock in the square (x, y, s).
void DrawLock(ImDrawList* dl, float x, float y, float s, ImU32 col) {
	dl->PathArcTo({x + s * 0.5f, y + s * 0.42f}, s * 0.22f, IM_PI, 2.0f * IM_PI, 12);
	dl->PathStroke(col, 0, s * 0.11f);
	dl->AddLine({x + s * 0.28f, y + s * 0.42f}, {x + s * 0.28f, y + s * 0.5f}, col, s * 0.11f);
	dl->AddLine({x + s * 0.72f, y + s * 0.42f}, {x + s * 0.72f, y + s * 0.5f}, col, s * 0.11f);
	dl->AddRectFilled({x + s * 0.16f, y + s * 0.48f}, {x + s * 0.84f, y + s * 0.94f}, col, s * 0.1f);
}

// The shimmer: a bright diagonal band sweeping over (p0, p1) once every few seconds.
void Shimmer(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float t, float phase, float alpha) {
	const float period = 4.2f, sweep = 0.95f;
	const float u      = std::fmod(t + phase, period) / sweep;
	if (u >= 1.0f || alpha <= 0.0f) return;
	const float w = p1.x - p0.x, h = p1.y - p0.y;
	const float cx = p0.x - w * 0.6f + u * w * 2.2f, slant = h * 0.45f;
	dl->PushClipRect(p0, p1, true);
	for (int k = 0; k < 3; k++) {
		const float half = w * (k == 0 ? 0.16f : k == 1 ? 0.09f : 0.035f);
		const ImU32 c    = Col(255, 255, 255, alpha * (k == 0 ? 0.08f : k == 1 ? 0.12f : 0.20f));
		dl->AddQuadFilled({cx - half + slant, p0.y}, {cx + half + slant, p0.y}, {cx + half - slant, p1.y}, {cx - half - slant, p1.y}, c);
	}
	dl->PopClipRect();
}

std::string DateText(int64_t t) {
	if (t <= 0) return T("Obtenido");
	const std::time_t tt = static_cast<std::time_t>(t);
	std::tm           tm {};
#if defined(_WIN32)
	if (localtime_s(&tm, &tt) != 0) return T("Obtenido");
#else
	if (localtime_r(&tt, &tm) == nullptr) return T("Obtenido");
#endif
	char date[64], line[160];
	std::snprintf(date, sizeof(date), "%02d/%02d/%04d %02d:%02d", tm.tm_mday, tm.tm_mon + 1, tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
	std::snprintf(line, sizeof(line), T("Obtenido el %s"), date);
	return line;
}

const char* GradeName(int g) {
	switch (g) {
		case 1: return T("Platino");
		case 2: return T("Oro");
		case 3: return T("Plata");
		case 4: return T("Bronce");
		default: return "";
	}
}
int Points(int g) {
	return g == 1 ? 300 : g == 2 ? 90 : g == 3 ? 30 : g == 4 ? 15 : 0;
}

bool Clicked(ImVec2 p0, ImVec2 p1) {
	const auto& io = ImGui::GetIO();
	return ImGui::IsMouseClicked(ImGuiMouseButton_Left) && io.MousePos.x >= p0.x && io.MousePos.x < p1.x && io.MousePos.y >= p0.y &&
	       io.MousePos.y < p1.y;
}
bool Hovered(ImVec2 p0, ImVec2 p1) {
	const auto& io = ImGui::GetIO();
	return io.MousePos.x >= p0.x && io.MousePos.x < p1.x && io.MousePos.y >= p0.y && io.MousePos.y < p1.y;
}
bool Pressed(std::initializer_list<ImGuiKey> keys) {
	for (const ImGuiKey k: keys)
		if (ImGui::IsKeyPressed(k, true)) return true;
	return false;
}

// Returns false when the page asks to go back (its Volver).
bool Draw(float width, float height, float unit) {
	const auto& io  = ImGui::GetIO();
	const float u   = unit;
	const float now = static_cast<float>(ImGui::GetTime());
	const float dt  = std::clamp(io.DeltaTime, 0.0f, 0.1f);
	EnsureArt(unit);
	Refresh(false);
	bool back = false;

	// Input: the filters (L1/R1, Q/E, PageUp/PageDown), the order (Y, Tab).
	if (g_tab_delta != 0) {
		g_filter           = ((g_filter + g_tab_delta) % 3 + 3) % 3;
		g_tab_delta = 0;
		Rebuild();
		g_list_at = ImGui::GetTime();
		g_scroll_target = 0;
	}
	if (Pressed({ImGuiKey_GamepadFaceUp, ImGuiKey_Tab})) {
		g_sort = 1 - g_sort;
		Rebuild();
		g_list_at = ImGui::GetTime();
	}

	// The page: fades and slides in.
	const float in = Ease((now - static_cast<float>(g_opened_at)) / 0.32f);
	ImGui::SetNextWindowPos({0, 0});
	ImGui::SetNextWindowSize({width, height});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("##trophy_page", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImDrawList* dl   = ImGui::GetWindowDrawList();
	ImFont*     body = FindFont("segoeui");
	if (body == nullptr) body = ImGui::GetFont();
	dl->AddRectFilled({0, 0}, {width, height}, Col(0, 0, 0, 0.55f * in));

	const float margin = std::max(24.0f * u, std::min(width, height) * 0.035f);
	const float pw     = std::min(width - margin * 2.0f, 1680.0f * u);
	const float ph     = height - margin * 2.0f;
	const float px0    = (width - pw) * 0.5f;
	const float py0    = margin + (1.0f - in) * 36.0f * u;
	const float pad    = 28.0f * u;
	const float round  = 18.0f * u;
	const ImU32 gold   = Col(222, 196, 140, in);
	// shadow, card, header band, border
	for (int k = 3; k >= 1; k--)
		dl->AddRectFilled({px0 - k * 4.0f * u, py0 - k * 3.0f * u}, {px0 + pw + k * 4.0f * u, py0 + ph + k * 6.0f * u}, Col(0, 0, 0, 0.10f * in), round + k * 4.0f * u);
	dl->AddRectFilled({px0, py0}, {px0 + pw, py0 + ph}, Col(17, 18, 22, 0.97f * in), round);
	const float header_h = 168.0f * u;
	dl->AddRectFilledMultiColor({px0 + round, py0}, {px0 + pw - round, py0 + header_h}, Col(46, 40, 30, 0.55f * in), Col(30, 32, 40, 0.55f * in),
	                            Col(17, 18, 22, 0.0f), Col(17, 18, 22, 0.0f));
	dl->AddRect({px0, py0}, {px0 + pw, py0 + ph}, Col(222, 196, 140, 0.28f * in), round, 0, 1.2f * u);

	// Totals.
	int total = 0, have = 0, earned = 0;
	int grade_total[5] = {}, grade_have[5] = {};
	for (const auto& r: g_rows) {
		const int g = r.info.grade >= 1 && r.info.grade <= 4 ? r.info.grade : 0;
		total += Points(g);
		grade_total[g]++;
		if (r.earned) {
			have += Points(g);
			earned++;
			grade_have[g]++;
		}
	}
	const float frac = total > 0 ? static_cast<float>(have) / static_cast<float>(total) : 0.0f;

	// Header: the game's icon, its title, the progress, the counts per grade.
	{
		const float isz = 116.0f * u;
		const float ix = px0 + pad, iy = py0 + (header_h - isz) * 0.5f;
		ImVec2      uv0, uv1;
		dl->AddRectFilled({ix - 2 * u, iy - 2 * u}, {ix + isz + 2 * u, iy + isz + 2 * u}, Col(222, 196, 140, 0.35f * in), 16.0f * u);
		if (g_art != nullptr && IconUv(g_art->game_slot, &uv0, &uv1))
			dl->AddImageRounded(g_tex->GetTexRef(), {ix, iy}, {ix + isz, iy + isz}, uv0, uv1, Col(255, 255, 255, in), 14.0f * u);
		else {
			dl->AddRectFilled({ix, iy}, {ix + isz, iy + isz}, Col(40, 42, 50, in), 14.0f * u);
			Libs::Trophy::DrawCup(dl, ix + isz * 0.15f, iy + isz * 0.12f, isz * 0.7f, Libs::Trophy::CupColor(2, in), in);
		}
		const float tx = ix + isz + 26.0f * u;
		const float grades_w = std::min(4.0f * 112.0f * u, pw * 0.34f);
		const float gx0      = px0 + pw - pad - grades_w;
		const float text_w   = gx0 - 30.0f * u - tx;
		dl->AddText(body, 16.0f * u, {tx, iy + 2 * u}, gold, T("TROFEOS"));
		std::string title = Libs::Trophy::GameTitle();
		if (title.empty()) title = "Demon's Souls";
		TextFit(dl, FontFor(title, body), 38.0f * u, {tx, iy + 20 * u}, Col(245, 242, 235, in), title, text_w);
		char line[160];
		std::snprintf(line, sizeof(line), T("%d de %d trofeos obtenidos"), earned, static_cast<int>(g_rows.size()));
		TextFit(dl, body, 18.0f * u, {tx, iy + 66 * u}, Col(200, 198, 192, in), line, text_w);
		// The bar: a track, the gold fill (animated in), a light sweeping along it and the percentage.
		char pct[16];
		std::snprintf(pct, sizeof(pct), "%d%%", static_cast<int>(std::floor(frac * 100.0f + 0.0001f)));
		const float pct_w = body->CalcTextSizeA(26.0f * u, FLT_MAX, 0.0f, "100%").x;
		const float bx0 = tx, bx1 = std::max(tx + 40 * u, gx0 - 40.0f * u - pct_w - 14.0f * u), by = iy + isz - 14.0f * u, bh = 12.0f * u;
		dl->AddRectFilled({bx0, by}, {bx1, by + bh}, Col(255, 255, 255, 0.10f * in), bh * 0.5f);
		const float shown = frac * Ease((now - static_cast<float>(g_opened_at) - 0.15f) / 0.9f);
		if (shown > 0.0f) {
			const float fx = bx0 + (bx1 - bx0) * shown;
			dl->AddRectFilledMultiColor({bx0 + bh * 0.5f, by}, {std::max(bx0 + bh * 0.5f, fx - bh * 0.5f), by + bh}, Col(176, 132, 64, in),
			                            Col(236, 204, 120, in), Col(236, 204, 120, in), Col(176, 132, 64, in));
			dl->AddCircleFilled({bx0 + bh * 0.5f, by + bh * 0.5f}, bh * 0.5f, Col(176, 132, 64, in), 16);
			dl->AddCircleFilled({std::max(bx0 + bh * 0.5f, fx - bh * 0.5f), by + bh * 0.5f}, bh * 0.5f, Col(236, 204, 120, in), 16);
			Shimmer(dl, {bx0, by}, {fx, by + bh}, now, 0.0f, in);
		}
		dl->AddText(body, 26.0f * u, {bx1 + 14.0f * u, by + bh * 0.5f - 15.0f * u}, gold, pct);
		// The grades.
		const float cell_w = grades_w / 4.0f;
		for (int g = 1; g <= 4; g++) {
			const float cx  = gx0 + cell_w * static_cast<float>(g - 1);
			const float cup = std::min(44.0f * u, cell_w * 0.5f);
			const float cy  = iy + 8.0f * u;
			if (g == 1 && grade_have[1] > 0) {
				const float p = 0.5f + 0.5f * std::sin(now * 2.4f);
				for (int k = 0; k < 4; k++)
					dl->AddCircleFilled({cx + cell_w * 0.5f, cy + cup * 0.5f}, cup * (0.55f + 0.14f * k + 0.08f * p), Col(170, 200, 235, (0.07f - 0.014f * k) * in), 32);
			}
			Libs::Trophy::DrawCup(dl, cx + (cell_w - cup) * 0.5f, cy, cup, Libs::Trophy::CupColor(g, in), in);
			char n[32];
			std::snprintf(n, sizeof(n), "%d/%d", grade_have[g], grade_total[g]);
			const float nw = body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, n).x;
			dl->AddText(body, 22.0f * u, {cx + (cell_w - nw) * 0.5f, cy + cup + 8.0f * u}, Col(240, 238, 232, in), n);
			const char* gn = GradeName(g);
			const float gw = body->CalcTextSizeA(15.0f * u, FLT_MAX, 0.0f, gn).x;
			dl->AddText(body, 15.0f * u, {cx + (cell_w - gw) * 0.5f, cy + cup + 36.0f * u}, Libs::Trophy::CupColor(g, in), gn);
		}
	}

	// The filters and the order.
	const float fy0 = py0 + header_h, fh = 46.0f * u;
	{
		const char* names[3] = {T("Todos"), T("Obtenidos"), T("Sin obtener")};
		const int   counts[3] = {static_cast<int>(g_rows.size()), earned, static_cast<int>(g_rows.size()) - earned};
		float       x = px0 + pad;
		float       sel_x = x, sel_w = 0;
		struct Pill {
			float x, w;
			std::string text;
		};
		Pill pills[3];
		for (int i = 0; i < 3; i++) {
			pills[i].text = std::string(names[i]) + "  " + std::to_string(counts[i]);
			pills[i].w    = body->CalcTextSizeA(19.0f * u, FLT_MAX, 0.0f, pills[i].text.c_str()).x + 36.0f * u;
			pills[i].x    = x;
			if (i == g_filter) sel_x = x, sel_w = pills[i].w;
			x += pills[i].w + 10.0f * u;
		}
		const float k = 1.0f - std::exp(-dt * 16.0f);
		if (g_pill_x < 0) g_pill_x = sel_x, g_pill_w = sel_w;
		g_pill_x += (sel_x - g_pill_x) * k;
		g_pill_w += (sel_w - g_pill_w) * k;
		dl->AddRectFilled({g_pill_x, fy0 + 4 * u}, {g_pill_x + g_pill_w, fy0 + fh - 4 * u}, Col(222, 196, 140, 0.95f * in), fh);
		for (int i = 0; i < 3; i++) {
			const ImVec2 p0 {pills[i].x, fy0 + 4 * u}, p1 {pills[i].x + pills[i].w, fy0 + fh - 4 * u};
			const bool   hot = Hovered(p0, p1);
			if (i != g_filter) dl->AddRect(p0, p1, Col(255, 255, 255, (hot ? 0.45f : 0.18f) * in), fh, 0, 1.0f * u);
			if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			dl->AddText(body, 19.0f * u, {pills[i].x + 18.0f * u, fy0 + fh * 0.5f - 12.0f * u},
			            i == g_filter ? Col(24, 22, 18, in) : Col(225, 222, 215, in), pills[i].text.c_str());
			if (Clicked(p0, p1) && i != g_filter) {
				g_filter = i;
				Rebuild();
				g_list_at       = ImGui::GetTime();
				g_scroll_target = 0;
			}
		}
		const std::string sort = std::string(T("Ordenar:")) + " " + T(g_sort == 0 ? "Grado" : "Fecha");
		const float sw = body->CalcTextSizeA(19.0f * u, FLT_MAX, 0.0f, sort.c_str()).x + 36.0f * u;
		const ImVec2 s0 {px0 + pw - pad - sw, fy0 + 4 * u}, s1 {px0 + pw - pad, fy0 + fh - 4 * u};
		if (s0.x > x) {
			const bool hot = Hovered(s0, s1);
			dl->AddRect(s0, s1, Col(222, 196, 140, (hot ? 0.8f : 0.4f) * in), fh, 0, 1.0f * u);
			dl->AddText(body, 19.0f * u, {s0.x + 18.0f * u, fy0 + fh * 0.5f - 12.0f * u}, gold, sort.c_str());
			if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			if (Clicked(s0, s1)) {
				g_sort = 1 - g_sort;
				Rebuild();
				g_list_at = ImGui::GetTime();
			}
		}
		dl->AddLine({px0 + pad, fy0 + fh + 10 * u}, {px0 + pw - pad, fy0 + fh + 10 * u}, Col(255, 255, 255, 0.08f * in), 1.0f * u);
	}

	// The footer: the keys, and Volver.
	const float foot_h = 46.0f * u;
	{
		const float y    = py0 + ph - foot_h;
		const char* hint = T("L1/R1 o Q/E: filtro   \xC2\xB7   Y o Tab: orden   \xC2\xB7   B, Esc o F1: volver");
		const char* vb   = T("Volver");
		const float vw   = body->CalcTextSizeA(18.0f * u, FLT_MAX, 0.0f, vb).x + 40.0f * u;
		const ImVec2 v0 {px0 + pad, y + 6 * u}, v1 {px0 + pad + vw, y + foot_h - 10 * u};
		const bool   hot = Hovered(v0, v1);
		dl->AddRect(v0, v1, Col(255, 255, 255, (hot ? 0.6f : 0.25f) * in), 8.0f * u, 0, 1.0f * u);
		dl->AddText(body, 18.0f * u, {v0.x + 20.0f * u, v0.y + (v1.y - v0.y) * 0.5f - 11.0f * u}, Col(230, 228, 222, in), vb);
		if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		if (Clicked(v0, v1)) back = true;
		const float hw = body->CalcTextSizeA(16.0f * u, FLT_MAX, 0.0f, hint).x;
		if (v1.x + 20 * u + hw < px0 + pw - pad)
			dl->AddText(body, 16.0f * u, {px0 + pw - pad - hw, y + foot_h * 0.5f - 12.0f * u}, Col(160, 158, 152, in), hint);
	}

	// The list.
	const ImVec2 l0 {px0 + pad, fy0 + fh + 22.0f * u}, l1 {px0 + pw - pad, py0 + ph - foot_h - 8.0f * u};
	const float  view_h = std::max(0.0f, l1.y - l0.y);
	if (g_status != 1) {
		const char* msg = g_status == -1 ? T("Cargando los trofeos...")
		                                 : T("No se encontraron los trofeos del juego (falta sce_sys/trophy2/trophy00.ucp).");
		const float mw  = std::min(body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, msg).x, l1.x - l0.x);
		dl->AddText(body, 22.0f * u, {(l0.x + l1.x - mw) * 0.5f, l0.y + view_h * 0.4f}, Col(210, 206, 198, in), msg, nullptr, l1.x - l0.x);
	} else if (g_view.empty()) {
		const char* msg = g_filter == 1 ? T("Aún no has obtenido ningún trofeo.") : T("¡Has obtenido todos los trofeos!");
		const float mw  = body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, msg).x;
		dl->AddText(body, 22.0f * u, {(l0.x + l1.x - mw) * 0.5f, l0.y + view_h * 0.4f}, Col(210, 206, 198, in), msg);
	} else {
		const int   cols  = (l1.x - l0.x) >= 1300.0f * u ? 2 : 1;
		const float gap   = 16.0f * u;
		const float cw    = ((l1.x - l0.x) - gap * static_cast<float>(cols - 1)) / static_cast<float>(cols);
		const float ch    = 138.0f * u;
		const int   count = static_cast<int>(g_view.size());
		const int   rows  = (count + cols - 1) / cols;
		const float content = static_cast<float>(rows) * (ch + gap) - gap;
		const float max_scroll = std::max(0.0f, content - view_h + 4.0f * u);

		// The selection: kept by id; arrows, D-pad and the stick move it, the mouse picks what it moves over.
		int sel = -1;
		for (int i = 0; i < count; i++)
			if (g_rows[static_cast<size_t>(g_view[static_cast<size_t>(i)])].info.id == g_sel_id) sel = i;
		bool moved = false;
		const auto step = [&](int d) {
			sel   = sel < 0 ? 0 : std::clamp(sel + d, 0, count - 1);
			moved = true;
		};
		if (Pressed({ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown})) step(cols);
		if (Pressed({ImGuiKey_UpArrow, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp})) step(-cols);
		if (Pressed({ImGuiKey_RightArrow, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight})) step(1);
		if (Pressed({ImGuiKey_LeftArrow, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft})) step(-1);
		if (Pressed({ImGuiKey_Home})) {
			sel   = 0;
			moved = true;
		}
		if (Pressed({ImGuiKey_End})) {
			sel   = count - 1;
			moved = true;
		}
		if (Hovered(l0, l1) && io.MouseWheel != 0.0f) g_scroll_target -= io.MouseWheel * (ch + gap) * 0.8f;
		if (moved && sel >= 0) {
			const float top = static_cast<float>(sel / cols) * (ch + gap);
			if (top < g_scroll_target) g_scroll_target = top - 8.0f * u;
			if (top + ch > g_scroll_target + view_h) g_scroll_target = top + ch - view_h + 8.0f * u;
		}
		g_scroll_target = std::clamp(g_scroll_target, 0.0f, max_scroll);
		g_scroll += (g_scroll_target - g_scroll) * (1.0f - std::exp(-dt * 14.0f));
		if (std::fabs(g_scroll_target - g_scroll) < 0.25f) g_scroll = g_scroll_target;
		const bool mouse_moved = io.MousePos.x != g_last_mouse.x || io.MousePos.y != g_last_mouse.y;
		g_last_mouse           = io.MousePos;
		if (mouse_moved && Hovered(l0, l1)) {
			const float lx = io.MousePos.x - l0.x, ly = io.MousePos.y - l0.y + g_scroll;
			const int   c = static_cast<int>(lx / (cw + gap)), r = static_cast<int>(ly / (ch + gap));
			const bool  inside = lx - static_cast<float>(c) * (cw + gap) < cw && ly - static_cast<float>(r) * (ch + gap) < ch;
			const int   i = r * cols + c;
			if (inside && c < cols && i >= 0 && i < count) sel = i;
		}
		g_sel_id = sel >= 0 ? g_rows[static_cast<size_t>(g_view[static_cast<size_t>(sel)])].info.id : -1;

		dl->PushClipRect(l0, l1, true);
		const float list_t = now - static_cast<float>(g_list_at);
		const int   first  = std::max(0, static_cast<int>(g_scroll / (ch + gap)) - 1) * cols;
		for (int i = first; i < count; i++) {
			const int   r   = i / cols, c = i % cols;
			const float y0  = l0.y + static_cast<float>(r) * (ch + gap) - g_scroll;
			if (y0 > l1.y) break;
			const auto& row = g_rows[static_cast<size_t>(g_view[static_cast<size_t>(i)])];
			const int   id  = row.info.id;
			const int   g   = row.info.grade >= 1 && row.info.grade <= 4 ? row.info.grade : 0;
			const bool  secret = row.info.hidden && !row.earned;
			// Stagger: each card a little after the one before (the first screen only).
			const int   order  = std::max(0, i - first);
			const float appear = Ease((list_t - 0.10f - 0.035f * static_cast<float>(std::min(order, 16))) / 0.35f);
			const float a      = in * appear;
			if (a <= 0.0f) continue;
			float& hot = g_hot[id];
			hot += ((i == sel ? 1.0f : 0.0f) - hot) * (1.0f - std::exp(-dt * 12.0f));
			const float x0 = l0.x + static_cast<float>(c) * (cw + gap);
			const float yy = y0 + (1.0f - appear) * 22.0f * u;
			const float sc = 1.0f + 0.018f * hot;
			const ImVec2 mid {x0 + cw * 0.5f, yy + ch * 0.5f};
			const ImVec2 p0 {mid.x - cw * 0.5f * sc, mid.y - ch * 0.5f * sc}, p1 {mid.x + cw * 0.5f * sc, mid.y + ch * 0.5f * sc};
			const ImU32  gc = Libs::Trophy::CupColor(g, a);
			const float  cr = 14.0f * u;
			// glow (selection), card, border
			for (int k = 3; k >= 1; k--)
				if (hot > 0.01f)
					dl->AddRect({p0.x - k * 2.5f * u, p0.y - k * 2.5f * u}, {p1.x + k * 2.5f * u, p1.y + k * 2.5f * u}, WithAlpha(gc, hot * (0.22f - 0.05f * k)),
					            cr + k * 2.5f * u, 0, 2.5f * u);
			dl->AddRectFilled(p0, p1, row.earned ? Col(31, 32, 38, a) : Col(24, 25, 29, a), cr);
			if (row.earned)
				dl->AddRectFilledMultiColor({p0.x + cr, p0.y}, {p1.x - cr, p0.y + ch * 0.5f}, WithAlpha(gc, 0.06f), Col(0, 0, 0, 0), Col(0, 0, 0, 0),
				                            WithAlpha(gc, 0.02f));
			dl->AddRect(p0, p1, hot > 0.01f ? WithAlpha(gc, 0.35f + 0.6f * hot) : Col(255, 255, 255, 0.07f * a), cr, 0, (1.0f + 1.0f * hot) * u);

			// The icon (larger when focused); a platinum's halo; the shimmer on earned ones; locked: dimmed, a padlock.
			const float isz = 112.0f * u * (1.0f + 0.06f * hot);
			const float icx = p0.x + 13.0f * u + 56.0f * u, icy = (p0.y + p1.y) * 0.5f;
			const ImVec2 i0 {icx - isz * 0.5f, icy - isz * 0.5f}, i1 {icx + isz * 0.5f, icy + isz * 0.5f};
			const float  ir = 10.0f * u;
			if (g == 1 && row.earned) {
				const float p = 0.5f + 0.5f * std::sin(now * 2.2f + static_cast<float>(id));
				for (int k = 1; k <= 5; k++)
					dl->AddRectFilled({i0.x - k * 2.6f * u * (1 + p * 0.4f), i0.y - k * 2.6f * u * (1 + p * 0.4f)},
					                  {i1.x + k * 2.6f * u * (1 + p * 0.4f), i1.y + k * 2.6f * u * (1 + p * 0.4f)}, Col(170, 205, 245, (0.10f - 0.018f * k) * a * (0.6f + 0.4f * p)),
					                  ir + k * 2.6f * u);
			}
			ImVec2 uv0, uv1;
			const bool has_icon = !secret && IconUv(SlotOf(id), &uv0, &uv1);
			dl->AddRectFilled(i0, i1, Col(40, 42, 50, a), ir);
			if (has_icon) {
				const ImU32 tint = row.earned ? Col(255, 255, 255, a) : Col(120, 120, 124, a);
				dl->AddImageRounded(g_tex->GetTexRef(), i0, i1, uv0, uv1, tint, ir);
				if (!row.earned) dl->AddRectFilled(i0, i1, Col(52, 52, 56, 0.55f * a), ir); // (desaturated)
				else Shimmer(dl, i0, i1, now, static_cast<float>(id) * 0.61f, a * (g == 1 ? 1.4f : 1.0f));
			} else if (secret) {
				const char* q  = "?";
				const float qs = isz * 0.55f;
				const ImVec2 qz = body->CalcTextSizeA(qs, FLT_MAX, 0.0f, q);
				dl->AddText(body, qs, {icx - qz.x * 0.5f, icy - qz.y * 0.5f}, Col(120, 120, 126, a), q);
			} else {
				Libs::Trophy::DrawCup(dl, i0.x + isz * 0.15f, i0.y + isz * 0.12f, isz * 0.7f,
				                      row.earned ? gc : Libs::Trophy::CupColor(0, a), a);
			}
			if (!row.earned) {
				const float ls = 30.0f * u;
				dl->AddCircleFilled({i1.x - ls * 0.5f - 2 * u, i1.y - ls * 0.5f - 2 * u}, ls * 0.62f, Col(14, 14, 17, 0.9f * a), 24);
				DrawLock(dl, i1.x - ls - 2 * u + ls * 0.1f, i1.y - ls - 2 * u + ls * 0.04f, ls * 0.8f, Col(200, 200, 205, a));
			}
			dl->AddRect(i0, i1, row.earned ? WithAlpha(gc, 0.55f) : Col(255, 255, 255, 0.06f * a), ir, 0, 1.0f * u);

			// The texts: the name, two lines of description, then the grade and the date (or Sin obtener).
			const float tx = p0.x + 13.0f * u + 112.0f * u + 22.0f * u, tw = p1.x - 18.0f * u - tx;
			const std::string name   = secret ? std::string(T("Trofeo oculto")) : row.info.name;
			const std::string detail = secret ? std::string(T("Se revelará al obtenerlo.")) : row.info.detail;
			TextFit(dl, FontFor(name, body), 23.0f * u, {tx, p0.y + 14.0f * u}, row.earned ? Col(246, 244, 240, a) : Col(170, 170, 176, a), name, tw);
			{
				ImFont*      f  = FontFor(detail, body);
				const float  fs = 17.0f * u;
				const ImVec4 clip {tx, p0.y + 46.0f * u, tx + tw, p0.y + 46.0f * u + fs * 2.0f * 1.18f + 2.0f * u};
				dl->AddText(f, fs, {tx, p0.y + 46.0f * u}, row.earned ? Col(196, 196, 204, a) : Col(128, 128, 136, a), detail.c_str(), nullptr, tw, &clip);
			}
			{
				const float ly  = p1.y - 34.0f * u;
				const float cup = 22.0f * u;
				Libs::Trophy::DrawCup(dl, tx, ly, cup, row.earned ? gc : WithAlpha(gc, 0.55f), a);
				const char* gn = GradeName(g);
				dl->AddText(body, 16.0f * u, {tx + cup + 6.0f * u, ly + 1.0f * u}, row.earned ? gc : WithAlpha(gc, 0.6f), gn);
				const float gw   = body->CalcTextSizeA(16.0f * u, FLT_MAX, 0.0f, gn).x;
				const std::string when = row.earned ? DateText(row.time) : std::string(T("Sin obtener"));
				const float wx = tx + cup + 6.0f * u + gw + 10.0f * u;
				dl->AddText(body, 16.0f * u, {wx, ly + 1.0f * u}, Col(120, 120, 128, a), "\xC2\xB7");
				TextFit(dl, body, 16.0f * u, {wx + 14.0f * u, ly + 1.0f * u}, row.earned ? Col(222, 196, 140, a) : Col(140, 140, 148, a), when,
				        tx + tw - wx - 14.0f * u);
			}
		}
		dl->PopClipRect();
		// The scroll bar (where the view is), and fades at the list's edges.
		if (max_scroll > 0.0f) {
			const float bx = px0 + pw - pad * 0.5f, th = std::max(40.0f * u, view_h * view_h / (content + 4.0f * u));
			const float ty = l0.y + (view_h - th) * (g_scroll / max_scroll);
			dl->AddRectFilled({bx - 2 * u, l0.y}, {bx + 2 * u, l1.y}, Col(255, 255, 255, 0.05f * in), 2 * u);
			dl->AddRectFilled({bx - 2 * u, ty}, {bx + 2 * u, ty + th}, Col(222, 196, 140, 0.6f * in), 2 * u);
			if (g_scroll > 1.0f)
				dl->AddRectFilledMultiColor({l0.x, l0.y}, {l1.x, l0.y + 18 * u}, Col(17, 18, 22, in), Col(17, 18, 22, in), Col(17, 18, 22, 0), Col(17, 18, 22, 0));
			if (g_scroll < max_scroll - 1.0f)
				dl->AddRectFilledMultiColor({l0.x, l1.y - 18 * u}, {l1.x, l1.y}, Col(17, 18, 22, 0), Col(17, 18, 22, 0), Col(17, 18, 22, in), Col(17, 18, 22, in));
		}
	}
	ImGui::End();
	ImGui::PopStyleVar(2);
	return !back;
}

} // namespace Page


// The menu: a centred card with the three buttons; arrows / D-pad / stick choose, Enter / Space / A press, the mouse too.
int    g_menu_sel = 0;
double g_menu_at  = 0;
ImVec2 g_menu_mouse {-1, -1};

void DrawMenu(float width, float height, float u) {
	const auto& io  = ImGui::GetIO();
	const float now = static_cast<float>(ImGui::GetTime());
	const float in  = Page::Ease((now - static_cast<float>(g_menu_at)) / 0.25f);
	ImGui::SetNextWindowPos({0, 0});
	ImGui::SetNextWindowSize({width, height});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::Begin("##emu_f1menu", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	                 ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImDrawList* dl   = ImGui::GetWindowDrawList();
	ImFont*     body = FindFont("segoeui");
	if (body == nullptr) body = ImGui::GetFont();
	dl->AddRectFilled({0, 0}, {width, height}, Page::Col(0, 0, 0, 0.55f * in));
	const char* items[3] = {T("Volver al juego"), T("Ver trofeos"), T("Cerrar el juego")};
	const float bw = 440.0f * u, bh = 56.0f * u, gap = 14.0f * u, pad = 32.0f * u;
	const float cw = bw + pad * 2.0f, chh = pad + 52.0f * u + 3.0f * bh + 2.0f * gap + 20.0f * u + 30.0f * u + pad;
	const float x0 = (width - cw) * 0.5f, y0 = (height - chh) * 0.5f + (1.0f - in) * 24.0f * u;
	dl->AddRectFilled({x0 + 4 * u, y0 + 8 * u}, {x0 + cw + 4 * u, y0 + chh + 8 * u}, Page::Col(0, 0, 0, 0.35f * in), 18.0f * u);
	dl->AddRectFilled({x0, y0}, {x0 + cw, y0 + chh}, Page::Col(17, 18, 22, 0.97f * in), 18.0f * u);
	dl->AddRect({x0, y0}, {x0 + cw, y0 + chh}, Page::Col(222, 196, 140, 0.30f * in), 18.0f * u, 0, 1.2f * u);
	dl->AddText(body, 30.0f * u, {x0 + pad, y0 + pad - 4 * u}, Page::Col(222, 196, 140, in), "Demon's Souls");
	dl->AddLine({x0 + pad, y0 + pad + 40 * u}, {x0 + cw - pad, y0 + pad + 40 * u}, Page::Col(255, 255, 255, 0.10f * in), 1.0f * u);
	if (Page::Pressed({ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown})) g_menu_sel = (g_menu_sel + 1) % 3;
	if (Page::Pressed({ImGuiKey_UpArrow, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp})) g_menu_sel = (g_menu_sel + 2) % 3;
	const bool mouse_moved = io.MousePos.x != g_menu_mouse.x || io.MousePos.y != g_menu_mouse.y;
	g_menu_mouse           = io.MousePos;
	int pressed            = -1;
	if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) || ImGui::IsKeyPressed(ImGuiKey_Space, false) ||
	    ImGui::IsKeyPressed(ImGuiKey_GamepadFaceDown, false))
		pressed = g_menu_sel;
	static float hot[3] = {};
	for (int i = 0; i < 3; i++) {
		const float  by = y0 + pad + 52.0f * u + static_cast<float>(i) * (bh + gap);
		const ImVec2 p0 {x0 + pad, by}, p1 {x0 + pad + bw, by + bh};
		if (mouse_moved && Page::Hovered(p0, p1)) g_menu_sel = i;
		if (Page::Clicked(p0, p1)) pressed = i;
		hot[i] += ((g_menu_sel == i ? 1.0f : 0.0f) - hot[i]) * (1.0f - std::exp(-io.DeltaTime * 14.0f));
		const float  s = 1.0f + 0.02f * hot[i];
		const ImVec2 m {(p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f};
		const ImVec2 q0 {m.x - bw * 0.5f * s, m.y - bh * 0.5f * s}, q1 {m.x + bw * 0.5f * s, m.y + bh * 0.5f * s};
		for (int k = 3; k >= 1; k--)
			if (hot[i] > 0.01f)
				dl->AddRect({q0.x - k * 2.5f * u, q0.y - k * 2.5f * u}, {q1.x + k * 2.5f * u, q1.y + k * 2.5f * u},
				            Page::Col(222, 196, 140, hot[i] * (0.22f - 0.05f * k) * in), 10.0f * u + k * 2.5f * u, 0, 2.5f * u);
		dl->AddRectFilled(q0, q1, Page::Col(static_cast<int>(34 + 30 * hot[i]), static_cast<int>(35 + 24 * hot[i]), static_cast<int>(42 + 8 * hot[i]), in), 10.0f * u);
		dl->AddRect(q0, q1, Page::Col(222, 196, 140, (0.25f + 0.6f * hot[i]) * in), 10.0f * u, 0, 1.0f * u);
		if (i == 1) Libs::Trophy::DrawCup(dl, q0.x + 16 * u, m.y - 14 * u, 28 * u, Libs::Trophy::CupColor(2, in), in);
		const float tw = body->CalcTextSizeA(22.0f * u, FLT_MAX, 0.0f, items[i]).x;
		dl->AddText(body, 22.0f * u, {m.x - tw * 0.5f, m.y - 14.0f * u}, Page::Col(240, 238, 232, in), items[i]);
		if (Page::Hovered(p0, p1)) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
	}
	const char* hint = T("F1 o Esc: volver al juego");
	const float hw   = body->CalcTextSizeA(16.0f * u, FLT_MAX, 0.0f, hint).x;
	dl->AddText(body, 16.0f * u, {x0 + (cw - hw) * 0.5f, y0 + chh - pad - 14.0f * u}, Page::Col(150, 148, 142, in), hint);
	ImGui::End();
	ImGui::PopStyleVar(2);
	if (pressed == 0) {
		g_mode = 0;
		Queue({Input::Reset});
	} else if (pressed == 1) {
		g_tab_delta = 0;
		Page::Open();
		g_mode = 2;
	} else if (pressed == 2) {
		// Just closes: nothing saved, nothing skipped (the game's own saves are already on disk).
#if defined(_WIN32)
		ExitProcess(0);
#else
		std::_Exit(0);
#endif
	}
}

ImGuiKey PadKey(int button) {
	switch (button) {
		case SDL_CONTROLLER_BUTTON_A: return ImGuiKey_GamepadFaceDown;
		case SDL_CONTROLLER_BUTTON_B: return ImGuiKey_GamepadFaceRight;
		case SDL_CONTROLLER_BUTTON_X: return ImGuiKey_GamepadFaceLeft;
		case SDL_CONTROLLER_BUTTON_Y: return ImGuiKey_GamepadFaceUp;
		case SDL_CONTROLLER_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
		case SDL_CONTROLLER_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
		case SDL_CONTROLLER_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
		case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
		default: return ImGuiKey_None;
	}
}
ImGuiKey KeyboardKey(SDL_Keycode key) {
	switch (key) {
		case SDLK_UP: return ImGuiKey_UpArrow;
		case SDLK_DOWN: return ImGuiKey_DownArrow;
		case SDLK_LEFT: return ImGuiKey_LeftArrow;
		case SDLK_RIGHT: return ImGuiKey_RightArrow;
		case SDLK_RETURN: return ImGuiKey_Enter;
		case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
		case SDLK_SPACE: return ImGuiKey_Space;
		case SDLK_TAB: return ImGuiKey_Tab;
		case SDLK_HOME: return ImGuiKey_Home;
		case SDLK_END: return ImGuiKey_End;
		default: return ImGuiKey_None;
	}
}

// Window thread: opens the menu, the game's sticks centred (it gets nothing while the menu is open).
void OpenMenu() {
	g_passthrough = 0;
	for (int i = 0; i < SDL_NumJoysticks(); i++) {
		if (SDL_IsGameController(i) == SDL_FALSE) continue;
		for (int axis = 0; axis < SDL_CONTROLLER_AXIS_MAX; axis++) {
			SDL_Event e {};
			e.type        = SDL_CONTROLLERAXISMOTION;
			e.caxis.which = SDL_JoystickGetDeviceInstanceID(i);
			e.caxis.axis  = static_cast<Uint8>(axis);
			e.caxis.value = 0;
			if (SDL_PushEvent(&e) == 1) g_passthrough++;
		}
	}
	Queue({Input::Reset});
	g_mode = 1;
	BumpRev();
}

// Back one step: the trophy page to the menu, the menu to the game.
void Back() {
	int m = g_mode.load();
	g_mode = m == 2 ? 1 : 0;
	Queue({Input::Reset});
}

} // namespace

bool IsOpen() {
	return g_mode.load(std::memory_order_relaxed) != 0;
}

unsigned long long Revision() {
	return g_rev.load(std::memory_order_relaxed);
}

bool HandleEvent(const SDL_Event& e) {
	const bool open = g_mode.load() != 0;
	switch (e.type) {
		case SDL_KEYDOWN:
			if (e.key.keysym.sym == SDLK_F1) {
				if (e.key.repeat == 0) {
					if (!open) OpenMenu();
					else Back();
				}
				return true;
			}
			if (!open) return false;
			if (e.key.keysym.sym == SDLK_ESCAPE) {
				if (e.key.repeat == 0) Back();
				return true;
			}
			if (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_PAGEUP) Queue({Input::Tab, -1});
			if (e.key.keysym.sym == SDLK_e || e.key.keysym.sym == SDLK_PAGEDOWN) Queue({Input::Tab, +1});
			if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, true});
			return true;
		case SDL_KEYUP:
			if (e.key.keysym.sym == SDLK_F1) return true;
			if (open)
				if (const ImGuiKey k = KeyboardKey(e.key.keysym.sym); k != ImGuiKey_None) Queue({Input::Key, k, false});
			return false; // (releases reach the game: nothing stays held)
		case SDL_TEXTINPUT:
		case SDL_TEXTEDITING: return open;
		case SDL_CONTROLLERBUTTONDOWN: {
			if (!open) return false;
			const auto b = e.cbutton.button;
			if (b == SDL_CONTROLLER_BUTTON_B) {
				Back();
				return true;
			}
			if (b == SDL_CONTROLLER_BUTTON_LEFTSHOULDER) Queue({Input::Tab, -1});
			if (b == SDL_CONTROLLER_BUTTON_RIGHTSHOULDER) Queue({Input::Tab, +1});
			if (const ImGuiKey k = PadKey(b); k != ImGuiKey_None) Queue({Input::Key, k, true});
			return true;
		}
		case SDL_CONTROLLERBUTTONUP:
			if (open)
				if (const ImGuiKey k = PadKey(e.cbutton.button); k != ImGuiKey_None) Queue({Input::Key, k, false});
			return false;
		case SDL_CONTROLLERAXISMOTION: {
			if (g_passthrough > 0 && e.caxis.value == 0) {
				g_passthrough--;
				return false;
			}
			if (!open) return false;
			const float v = e.caxis.value / (e.caxis.value < 0 ? 32768.0f : 32767.0f);
			if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTY) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickUp, v < -0.35f, std::max(-v, 0.0f)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickDown, v > 0.35f, std::max(v, 0.0f)});
			} else if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTX) {
				Queue({Input::Analog, ImGuiKey_GamepadLStickLeft, v < -0.35f, std::max(-v, 0.0f)});
				Queue({Input::Analog, ImGuiKey_GamepadLStickRight, v > 0.35f, std::max(v, 0.0f)});
			}
			return true;
		}
		case SDL_MOUSEMOTION:
			if (!open) return false;
			Queue({Input::MousePos, 0, false, static_cast<float>(e.motion.x), static_cast<float>(e.motion.y)});
			return true;
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
			if (!open) return false;
			Queue({Input::MousePos, 0, false, static_cast<float>(e.button.x), static_cast<float>(e.button.y)});
			Queue({Input::MouseButton, e.button.button == SDL_BUTTON_RIGHT ? 1 : (e.button.button == SDL_BUTTON_MIDDLE ? 2 : 0),
			       e.type == SDL_MOUSEBUTTONDOWN});
			return e.type == SDL_MOUSEBUTTONDOWN;
		case SDL_MOUSEWHEEL:
			if (!open) return false;
			Queue({Input::Wheel, 0, false, static_cast<float>(e.wheel.x), static_cast<float>(e.wheel.y)});
			return true;
		default: return false;
	}
}

void Draw(float width, float height) {
	auto& io = ImGui::GetIO();
	{
		std::deque<Input> inputs;
		{
			std::scoped_lock lock(g_mutex);
			inputs.swap(g_inputs);
		}
		for (const auto& in: inputs) {
			switch (in.kind) {
				case Input::Key: io.AddKeyEvent(static_cast<ImGuiKey>(in.key), in.down); break;
				case Input::Analog: io.AddKeyAnalogEvent(static_cast<ImGuiKey>(in.key), in.down, in.x); break;
				case Input::MousePos: io.AddMousePosEvent(in.x, in.y); break;
				case Input::MouseButton: io.AddMouseButtonEvent(in.key, in.down); break;
				case Input::Wheel: io.AddMouseWheelEvent(in.x, in.y); break;
				case Input::Tab: g_tab_delta += in.key; break;
				case Input::Reset: g_menu_at = ImGui::GetTime(); break;
			}
		}
	}
	const int mode = g_mode.load();
	io.MouseDrawCursor = mode != 0;
	if (mode == 0) return;
	const float unit = std::max(std::min(width / 1920.0f, height / 1080.0f), 0.5f);
	if (mode == 2) {
		if (!Page::Draw(width, height, unit)) {
			int expected = 2;
			g_mode.compare_exchange_strong(expected, 1);
			g_menu_at = ImGui::GetTime();
		}
	} else {
		DrawMenu(width, height, unit);
	}
	BumpRev(); // (animated: a new picture each frame)
}

} // namespace Libs::TrophyPage
