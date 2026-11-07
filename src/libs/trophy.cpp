#include "libs/trophy.h"

#include "SDL.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "kernel/fileSystem.h"
#include "loader/systemContent.h"
#include "stb_image.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

namespace Libs::Trophy {

namespace {

constexpr uint32_t UCP_MAGIC      = 0xb228c60a;
constexpr uint32_t UCP_VERSION    = 1;
constexpr size_t   UCP_HEADER_LEN = 0x40;
constexpr size_t   UCP_TOC_SKIP   = 0x20;
constexpr size_t   UCP_ENTRY_LEN  = 0x40;
constexpr size_t   UCP_NAME_LEN   = 0x20;
constexpr int      ICON_SIZE      = 48; // the popup's icon is drawn from this many cells a side

using Bytes = std::vector<uint8_t>;

uint32_t Be32(const Bytes& d, size_t o) {
	return (uint32_t(d[o]) << 24u) | (uint32_t(d[o + 1]) << 16u) | (uint32_t(d[o + 2]) << 8u) | uint32_t(d[o + 3]);
}
uint64_t Be64(const Bytes& d, size_t o) { return (uint64_t(Be32(d, o)) << 32u) | Be32(d, o + 4); }

std::string Lower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
}

// The package's files by lower-case name (as the launcher's TrophyViewerDialog reads it).
bool ReadUcp(const std::filesystem::path& file_name, std::map<std::string, Bytes>& files) {
	std::ifstream file(file_name, std::ios::binary);
	if (!file) return false;
	const Bytes data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (data.size() < UCP_HEADER_LEN || Be32(data, 0) != UCP_MAGIC || Be32(data, 4) != UCP_VERSION) return false;
	if (Be64(data, 8) > data.size()) return false;
	const uint32_t count = Be32(data, 0x10);
	const uint64_t toc   = Be32(data, 0x14);
	if (toc > data.size() || UCP_TOC_SKIP + uint64_t(count) * UCP_ENTRY_LEN > data.size() - toc) return false;
	for (uint32_t i = 0; i < count; i++) {
		const size_t e = toc + UCP_TOC_SKIP + size_t(i) * UCP_ENTRY_LEN;
		size_t       len = 0;
		while (len < UCP_NAME_LEN && data[e + len] != 0) len++;
		std::string name(reinterpret_cast<const char*>(data.data() + e), len);
		while (!name.empty() && name.back() == ' ') name.pop_back();
		const uint64_t offset = Be64(data, e + 0x20), size = Be64(data, e + 0x28);
		if (name.empty() || offset > data.size() || size > data.size() - offset) continue;
		files[Lower(name)] = Bytes(data.begin() + static_cast<ptrdiff_t>(offset),
		                           data.begin() + static_cast<ptrdiff_t>(offset + size));
	}
	return true;
}

std::string JsonText(const nlohmann::json& v) {
	if (v.is_string()) return v.get<std::string>();
	if (v.is_number_integer()) return std::to_string(v.get<int64_t>());
	return {};
}

// The language tags tried for the console's language (sceSystemServiceParam LANG), best first.
std::vector<std::string> LanguageTags(uint32_t lang) {
	switch (lang) {
		case 0: return {"ja-jp", "ja"};
		case 1: return {"en-us", "en"};
		case 2: return {"fr-fr", "fr"};
		case 3: return {"es-es", "es"};
		case 4: return {"de-de", "de"};
		case 5: return {"it-it", "it"};
		case 6: return {"nl-nl", "nl"};
		case 7: return {"pt-pt", "pt"};
		case 8: return {"ru-ru", "ru"};
		case 9: return {"ko-kr", "ko"};
		case 10: return {"zh-hant", "zh-tw", "zh"};
		case 11: return {"zh-hans", "zh-cn", "zh"};
		case 12: return {"fi-fi", "fi"};
		case 13: return {"sv-se", "sv"};
		case 14: return {"da-dk", "da"};
		case 15: return {"no-no", "nb", "no"};
		case 16: return {"pl-pl", "pl"};
		case 17: return {"pt-br", "pt"};
		case 18: return {"en-gb", "en"};
		case 19: return {"tr-tr", "tr"};
		case 20: return {"es-419", "es-mx", "es"};
		case 21: return {"ar-ae", "ar"};
		case 22: return {"fr-ca", "fr"};
		case 23: return {"cs-cz", "cs"};
		case 24: return {"hu-hu", "hu"};
		case 25: return {"el-gr", "el"};
		case 26: return {"ro-ro", "ro"};
		case 27: return {"th-th", "th"};
		case 28: return {"vi-vn", "vi"};
		case 29: return {"id-id", "id"};
		default: return {"en-us", "en"};
	}
}

const char* Primary(uint32_t lang) {
	switch (lang) {
		case 0: return "ja";
		case 2: case 22: return "fr";
		case 3: case 20: return "es";
		case 4: return "de";
		case 5: return "it";
		case 6: return "nl";
		case 7: case 17: return "pt";
		case 8: return "ru";
		case 9: return "ko";
		case 10: case 11: return "zh";
		case 16: return "pl";
		case 19: return "tr";
		default: return "en";
	}
}

// "Trophy earned" and the grades in the game's language (the main ones; English otherwise).
struct Words {
	const char* earned;
	const char* grades[4]; // platinum, gold, silver, bronze
};
Words LabelFor(const std::string& primary) {
	static const std::map<std::string, Words> words = {
	    {"es", {"Trofeo obtenido", {"Platino", "Oro", "Plata", "Bronce"}}},
	    {"en", {"Trophy earned", {"Platinum", "Gold", "Silver", "Bronze"}}},
	    {"fr", {"Trophée obtenu", {"Platine", "Or", "Argent", "Bronze"}}},
	    {"de", {"Trophäe erhalten", {"Platin", "Gold", "Silber", "Bronze"}}},
	    {"it", {"Trofeo ottenuto", {"Platino", "Oro", "Argento", "Bronzo"}}},
	    {"pt", {"Troféu conquistado", {"Platina", "Ouro", "Prata", "Bronze"}}},
	    {"nl", {"Trofee verdiend", {"Platina", "Goud", "Zilver", "Brons"}}},
	    {"pl", {"Zdobyto trofeum", {"Platyna", "Złoto", "Srebro", "Brąz"}}},
	    {"ru", {"Получен приз", {"Платина", "Золото", "Серебро", "Бронза"}}},
	    {"tr", {"Kupa kazanıldı", {"Platin", "Altın", "Gümüş", "Bronz"}}},
	    {"ja", {"トロフィーを獲得しました", {"プラチナ", "ゴールド", "シルバー", "ブロンズ"}}},
	    {"zh", {"获得奖杯", {"白金", "金", "银", "铜"}}},
	    {"ko", {"트로피 획득", {"플래티넘", "골드", "실버", "브론즈"}}},
	};
	const auto it = words.find(primary);
	return it != words.end() ? it->second : words.at("en");
}

int GradeOf(const std::string& g) {
	if (g == "P" || g == "p") return 1;
	if (g == "G" || g == "g") return 2;
	if (g == "S" || g == "s") return 3;
	if (g == "B" || g == "b") return 4;
	return 0;
}

struct Entry {
	Info                  info;
	int                   icon_w = 0, icon_h = 0;
	std::vector<uint32_t> icon;
};

struct State {
	std::mutex                             mutex;
	bool                                   loaded = false;
	std::string                            title_id;
	std::string                            language; // the metadata file's tag
	Words                                  words {};
	std::map<int, Entry>                   trophies;
	std::map<int, int64_t>                 unlocked; // id -> unix time
	std::filesystem::path                  store;
	std::filesystem::path                  ucp;     // (port34) the package read, for BuildArt
	std::filesystem::path                  sce_sys; // (port34) the game's sce_sys (icon0.png)
	std::string                            title;   // (port34) param.sfo's TITLE
	std::deque<std::shared_ptr<const Toast>> queue;
	std::chrono::steady_clock::time_point  shown_at {};
};

State& S() {
	static State s;
	return s;
}

// The icon box-filtered down to ICON_SIZE (the popup draws it as cells: no texture to upload).
void LoadIcon(const std::map<std::string, Bytes>& files, int id, Entry& e) {
	char a[32], b[32];
	std::snprintf(a, sizeof(a), "trop%d.png", id);
	std::snprintf(b, sizeof(b), "trop%04d.png", id);
	for (const char* n: {a, b}) {
		const auto it = files.find(n);
		if (it == files.end()) continue;
		int  w = 0, h = 0, c = 0;
		auto* px = stbi_load_from_memory(it->second.data(), static_cast<int>(it->second.size()), &w, &h, &c, 4);
		if (px == nullptr || w <= 0 || h <= 0) {
			if (px != nullptr) stbi_image_free(px);
			continue;
		}
		e.icon_w = e.icon_h = ICON_SIZE;
		e.icon.assign(size_t(ICON_SIZE) * ICON_SIZE, 0);
		for (int y = 0; y < ICON_SIZE; y++) {
			for (int x = 0; x < ICON_SIZE; x++) {
				const int x0 = x * w / ICON_SIZE, x1 = std::max(x0 + 1, (x + 1) * w / ICON_SIZE);
				const int y0 = y * h / ICON_SIZE, y1 = std::max(y0 + 1, (y + 1) * h / ICON_SIZE);
				uint32_t sum[4] = {}, n = 0;
				for (int yy = y0; yy < y1; yy++)
					for (int xx = x0; xx < x1; xx++, n++)
						for (int k = 0; k < 4; k++) sum[k] += px[(size_t(yy) * w + xx) * 4 + k];
				e.icon[size_t(y) * ICON_SIZE + x] = (sum[0] / n) | ((sum[1] / n) << 8u) | ((sum[2] / n) << 16u) |
				                                    ((sum[3] / n) << 24u);
			}
		}
		stbi_image_free(px);
		return;
	}
}

void SaveStore(State& s) {
	nlohmann::json j;
	j["title_id"] = s.title_id;
	auto& u       = j["unlocked"];
	u             = nlohmann::json::object();
	for (const auto& [id, t]: s.unlocked) u[std::to_string(id)] = t;
	std::error_code ec;
	std::filesystem::create_directories(s.store.parent_path(), ec);
	const auto tmp = s.store.string() + ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		if (!f) return;
		f << j.dump(1, '\t');
	}
	std::filesystem::rename(tmp, s.store, ec);
	if (ec) {
		std::filesystem::remove(s.store, ec);
		std::filesystem::rename(tmp, s.store, ec);
	}
}

// For the launcher's trophy section: the trophies in the game's language (the launcher does not read the .ucp).
void SaveList(State& s) {
	nlohmann::json j;
	j["title_id"] = s.title_id;
	j["language"] = s.language;
	auto& list    = j["trophies"];
	list          = nlohmann::json::array();
	for (const auto& [id, e]: s.trophies) {
		const int g = e.info.grade;
		list.push_back({{"id", id},
		                {"grade", g == 1 ? "P" : g == 2 ? "G" : g == 3 ? "S" : g == 4 ? "B" : ""},
		                {"grade_text", g >= 1 && g <= 4 ? s.words.grades[g - 1] : ""},
		                {"name", e.info.name},
		                {"detail", e.info.detail},
		                {"hidden", e.info.hidden}});
	}
	std::error_code dir_ec;
	std::filesystem::create_directories(s.store.parent_path(), dir_ec); // (1030d) a new _SaveData: the folder first
	std::ofstream f(s.store.parent_path() / "trophies-list.json", std::ios::binary | std::ios::trunc);
	if (f) f << j.dump(1, '\t');
}

bool SoundWanted();
bool PopupWanted();
void SyncFromSave(State& s);

void Load(State& s) {
	if (s.loaded) return;
	s.loaded = true;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &s.title_id) || s.title_id.empty()) s.title_id = "UNKNOWN";
	s.store = std::filesystem::path("_SaveData") / s.title_id / "trophies.json";
	const uint32_t lang = Config::GetConsoleLanguage();
	std::printf("Trophy: init (title %s, sound %s, popups %s)\n", s.title_id.c_str(), SoundWanted() ? "on" : "off",
	            PopupWanted() ? "on" : "off");
	s.words             = LabelFor(Primary(lang));
	if (!Loader::SystemContentParamSfoGetString("TITLE", &s.title)) s.title.clear();

	// The unlocks so far.
	if (std::ifstream f(s.store, std::ios::binary); f) {
		const auto j = nlohmann::json::parse(f, nullptr, false);
		if (j.is_object() && j.contains("unlocked") && j["unlocked"].is_object()) {
			for (const auto& [k, v]: j["unlocked"].items()) {
				char* end = nullptr;
				const long id = std::strtol(k.c_str(), &end, 10);
				if (end != k.c_str()) s.unlocked[static_cast<int>(id)] = v.is_number() ? v.get<int64_t>() : 0;
			}
		}
	}

	// The package: the first trophy2/*.ucp.
	std::error_code       ec;
	const auto            dir = LibKernel::FileSystem::GetRealFilename("/app0") / "sce_sys" / "trophy2";
	std::filesystem::path ucp;
	for (const auto& f: std::filesystem::directory_iterator(dir, ec)) {
		if (Lower(Common::PathToString(f.path().extension())) == ".ucp" && (ucp.empty() || f.path() < ucp)) ucp = f.path();
	}
	s.sce_sys = LibKernel::FileSystem::GetRealFilename("/app0") / "sce_sys";
	std::map<std::string, Bytes> files;
	if (ucp.empty() || !ReadUcp(ucp, files)) {
		LOGF("Trophy: no trophy package in %s\n", Common::PathToString(dir).c_str());
		std::printf("Trophy: no trophy package in %s (%s)\n", Common::PathToString(dir).c_str(),
		            ucp.empty() ? "no .ucp" : "unreadable .ucp");
		return;
	}
	s.ucp              = ucp;
	const auto conf_it = files.find("tropconf.json");
	if (conf_it == files.end()) {
		std::printf("Trophy: %s has no tropconf.json\n", Common::PathToString(ucp).c_str());
		return;
	}
	const auto conf = nlohmann::json::parse(conf_it->second.begin(), conf_it->second.end(), nullptr, false);
	if (!conf.is_object()) return;

	// The metadata in the game's language, else English, else the package's default, else any.
	std::vector<std::string> tags = LanguageTags(lang);
	for (const char* t: {"en-us", "en-gb", "en"}) tags.emplace_back(t);
	if (conf.contains("defaultLanguage")) tags.push_back(Lower(JsonText(conf["defaultLanguage"])));
	const Bytes* meta = nullptr;
	for (const auto& tag: tags) {
		if (auto it = files.find("tropmeta_" + tag + ".json"); it != files.end()) {
			meta = &it->second, s.language = tag;
			break;
		}
		// (a tag with its region: "es" finds tropmeta_es-es.json)
		if (tag.find('-') == std::string::npos) {
			for (const auto& [name, data]: files) {
				if (name.rfind("tropmeta_" + tag + "-", 0) == 0 && name.size() > 5 && name.substr(name.size() - 5) == ".json") {
					meta = &data, s.language = name.substr(9, name.size() - 14);
					break;
				}
			}
			if (meta != nullptr) break;
		}
	}
	if (meta == nullptr) {
		for (const auto& [name, data]: files) {
			if (name.rfind("tropmeta", 0) == 0) {
				meta = &data, s.language = name;
				break;
			}
		}
	}
	std::map<int, std::pair<std::string, std::string>> texts;
	if (meta != nullptr) {
		const auto m = nlohmann::json::parse(meta->begin(), meta->end(), nullptr, false);
		if (m.is_object() && m.contains("metadata") && m["metadata"].contains("trophyMetadata")) {
			for (const auto& t: m["metadata"]["trophyMetadata"]) {
				const auto id = JsonText(t.value("id", nlohmann::json()));
				if (id.empty()) continue;
				texts[std::atoi(id.c_str())] = {JsonText(t.value("name", nlohmann::json())),
				                                JsonText(t.value("detail", nlohmann::json()))};
			}
		}
	}
	if (conf.contains("trophies") && conf["trophies"].is_array()) {
		for (const auto& t: conf["trophies"]) {
			const auto id_text = JsonText(t.value("id", nlohmann::json()));
			if (id_text.empty()) continue;
			Entry e;
			e.info.id     = std::atoi(id_text.c_str());
			e.info.grade  = GradeOf(JsonText(t.value("grade", nlohmann::json())));
			e.info.hidden = t.value("hidden", false);
			if (auto it = texts.find(e.info.id); it != texts.end()) {
				e.info.name   = it->second.first;
				e.info.detail = it->second.second;
			}
			if (e.info.name.empty()) e.info.name = "Trophy " + id_text;
			LoadIcon(files, e.info.id, e);
			s.trophies[e.info.id] = std::move(e);
		}
	}
	if (!s.trophies.empty() && s.trophies.count(kNexusTrophyId) == 0) {
		Entry e;
		e.info.id    = kNexusTrophyId;
		e.info.grade = 4;
		const bool es = s.language.rfind("es", 0) == 0;
		e.info.name   = es ? "Bienvenido al Nexo" : "Welcome to the Nexus";
		e.info.detail = es ? "Llega al Nexo por primera vez." : "Reach the Nexus for the first time.";
		if (const auto first = s.trophies.find(1); first != s.trophies.end()) {
			e.icon_w = first->second.icon_w, e.icon_h = first->second.icon_h, e.icon = first->second.icon;
		}
		s.trophies[kNexusTrophyId] = std::move(e);
	}
	LOGF("Trophy: %zu trophies (%s), %zu unlocked\n", s.trophies.size(), s.language.c_str(), s.unlocked.size());
	std::printf("Trophy: opened %s: %zu trophies, language %s (console language %u), %zu unlocked (%s)\n",
	            Common::PathToString(ucp.filename()).c_str(), s.trophies.size(), s.language.c_str(), lang,
	            s.unlocked.size(), Common::PathToString(s.store).c_str());
	SaveList(s);
	SyncFromSave(s); // (1109) the trophies the game's save already proves
}

// The chime: our own, synthesized (two bell-like partials rising a fifth, soft attack, exponential decay).
// KYTY_TROPHY_SOUND=0 turns it off.
bool SoundWanted() {
	const char* v = std::getenv("KYTY_TROPHY_SOUND");
	return v == nullptr || (v[0] != '0' && Lower(v) != "off" && Lower(v) != "false");
}
void PlayChime() {
	if (!SoundWanted()) return;
	std::thread([] {
		static std::mutex        m;
		static SDL_AudioDeviceID device = 0;
		std::lock_guard          lock(m);
		if (device == 0) {
			if (SDL_InitSubSystem(SDL_INIT_AUDIO) < 0) return;
			SDL_AudioSpec want {}, have {};
			want.freq     = 48000;
			want.format   = AUDIO_F32SYS;
			want.channels = 2;
			want.samples  = 1024;
			device        = SDL_OpenAudioDevice(nullptr, 0, &want, &have, 0);
			if (device == 0) return;
			SDL_PauseAudioDevice(device, 0);
		}
		constexpr int    rate = 48000;
		constexpr double pi   = 3.14159265358979323846;
		const int        n    = rate * 9 / 10;
		std::vector<float> buf(size_t(n) * 2);
		struct Note {
			double start, freq, amp;
		};
		const Note notes[] = {{0.00, 784.0, 0.22}, {0.11, 1174.7, 0.20}, {0.22, 1568.0, 0.13}}; // G5, D6, G6
		for (int i = 0; i < n; i++) {
			const double t = double(i) / rate;
			double       v = 0;
			for (const auto& nt: notes) {
				const double u = t - nt.start;
				if (u < 0) continue;
				const double env = std::min(1.0, u / 0.006) * std::exp(-u * 5.5);
				v += nt.amp * env * (std::sin(2 * pi * nt.freq * u) + 0.25 * std::sin(2 * pi * nt.freq * 2.01 * u) +
				                     0.08 * std::sin(2 * pi * nt.freq * 3.0 * u));
			}
			const double fade = i > n - 2400 ? double(n - i) / 2400.0 : 1.0;
			buf[size_t(i) * 2] = buf[size_t(i) * 2 + 1] = static_cast<float>(v * fade * 0.6);
		}
		SDL_QueueAudio(device, buf.data(), static_cast<uint32_t>(buf.size() * sizeof(float)));
	}).detach();
}

bool PopupWanted() {
	const char* v = std::getenv("KYTY_TROPHY_POPUP");
	return v == nullptr || v[0] != '0';
}

const char* GradeName(int g) { return g == 1 ? "platinum" : g == 2 ? "gold" : g == 3 ? "silver" : g == 4 ? "bronze" : "unknown"; }

void Record(State& s, int id, float seconds = TOAST_SECONDS) {
	s.unlocked[id] = static_cast<int64_t>(std::time(nullptr));
	const auto it  = s.trophies.find(id);
	auto       t   = std::make_shared<Toast>();
	t->id          = id;
	if (it != s.trophies.end()) {
		t->grade = it->second.info.grade;
		t->name  = it->second.info.name;
		t->detail = it->second.info.detail;
		t->icon_w = it->second.icon_w, t->icon_h = it->second.icon_h, t->icon = it->second.icon;
	} else {
		t->name = "Trophy " + std::to_string(id);
	}
	t->seconds    = seconds;
	t->label      = s.words.earned;
	t->grade_text = t->grade >= 1 && t->grade <= 4 ? s.words.grades[t->grade - 1] : "";
	LOGF("Trophy: unlocked id %d %s (%s)\n", id, t->name.c_str(), GradeName(t->grade));
	std::printf("Trophy: unlocked id %d %s (%s)\n", id, t->name.c_str(), GradeName(t->grade));
	if (PopupWanted()) s.queue.push_back(std::move(t));
}


// (port34) Area-average downscale (bilinear when enlarging), premultiplied so transparent edges do not darken, into
// dst (stride in pixels) at size x size.
struct Tap {
	int   i;
	float w;
};
std::vector<std::vector<Tap>> Taps(int src, int dst) {
	std::vector<std::vector<Tap>> taps(static_cast<size_t>(dst));
	const float scale = static_cast<float>(src) / static_cast<float>(dst);
	for (int x = 0; x < dst; x++) {
		auto& t = taps[static_cast<size_t>(x)];
		if (scale > 1.0f) {
			const float a = static_cast<float>(x) * scale, b = a + scale;
			for (int i = static_cast<int>(std::floor(a)); i < static_cast<int>(std::ceil(b)) && i < src; i++) {
				const float w = std::min(b, static_cast<float>(i + 1)) - std::max(a, static_cast<float>(i));
				if (w > 0.0f) t.push_back({i, w / scale});
			}
		} else {
			const float c  = (static_cast<float>(x) + 0.5f) * scale - 0.5f;
			const int   i0 = static_cast<int>(std::floor(c));
			const float f  = c - static_cast<float>(i0);
			t.push_back({std::clamp(i0, 0, src - 1), 1.0f - f});
			t.push_back({std::clamp(i0 + 1, 0, src - 1), f});
		}
	}
	return taps;
}
void Resample(const uint8_t* px, int w, int h, int size, uint32_t* dst, int stride) {
	const auto tx = Taps(w, size), ty = Taps(h, size);
	std::vector<float> rows(static_cast<size_t>(h) * size * 4); // horizontal pass (premultiplied)
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < size; x++) {
			float acc[4] = {};
			for (const auto& t: tx[static_cast<size_t>(x)]) {
				const uint8_t* p = px + (static_cast<size_t>(y) * w + t.i) * 4;
				const float    a = p[3] / 255.0f;
				acc[0] += p[0] * a * t.w, acc[1] += p[1] * a * t.w, acc[2] += p[2] * a * t.w, acc[3] += p[3] * t.w;
			}
			std::copy(acc, acc + 4, rows.begin() + static_cast<ptrdiff_t>((static_cast<size_t>(y) * size + x) * 4));
		}
	}
	for (int y = 0; y < size; y++) {
		for (int x = 0; x < size; x++) {
			float acc[4] = {};
			for (const auto& t: ty[static_cast<size_t>(y)]) {
				const float* r = rows.data() + (static_cast<size_t>(t.i) * size + x) * 4;
				for (int k = 0; k < 4; k++) acc[k] += r[k] * t.w;
			}
			const float a   = std::clamp(acc[3], 0.0f, 255.0f);
			const float inv = a > 0.0f ? 255.0f / a : 0.0f;
			const auto  c   = [&](float v) { return static_cast<uint32_t>(std::clamp(v * inv + 0.5f, 0.0f, 255.0f)); };
			dst[static_cast<size_t>(y) * stride + x] =
			    c(acc[0]) | (c(acc[1]) << 8u) | (c(acc[2]) << 16u) | (static_cast<uint32_t>(a + 0.5f) << 24u);
		}
	}
}

bool DecodeInto(const Bytes& png, Art& art, int slot) {
	int   w = 0, h = 0, c = 0;
	auto* px = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &w, &h, &c, 4);
	if (px == nullptr || w <= 0 || h <= 0 || w > 8192 || h > 8192) {
		if (px != nullptr) stbi_image_free(px);
		return false;
	}
	const int ox = (slot % art.cols) * art.stride, oy = (slot / art.cols) * art.stride;
	uint32_t* base = art.pixels.data() + static_cast<size_t>(oy) * art.width + ox;
	Resample(px, w, h, art.cell, base + static_cast<size_t>(art.pad) * art.width + art.pad, art.width);
	stbi_image_free(px);
	// The gutter: the cell's edge pixels repeated (no bleeding from the neighbours with linear filtering).
	for (int y = 0; y < art.stride; y++) {
		const int sy = std::clamp(y - art.pad, 0, art.cell - 1) + art.pad;
		for (int x = 0; x < art.stride; x++) {
			const int sx = std::clamp(x - art.pad, 0, art.cell - 1) + art.pad;
			if (sx == x && sy == y) continue;
			base[static_cast<size_t>(y) * art.width + x] = base[static_cast<size_t>(sy) * art.width + sx];
		}
	}
	return true;
}

const Bytes* IconFile(const std::map<std::string, Bytes>& files, int id) {
	char a[32], b[32];
	std::snprintf(a, sizeof(a), "trop%d.png", id);
	std::snprintf(b, sizeof(b), "trop%04d.png", id);
	for (const char* n: {a, b})
		if (const auto it = files.find(n); it != files.end()) return &it->second;
	return nullptr;
}

// (1109) Retroactive trophies for Demon's Souls (PPSA01341): on the first trophy load, every trophy the game's own save
// proves is unlocked through Record (the normal path: trophies.json, the popup queue, the chime), each popup shorter.
// Read only: the save is never written. DES_TROPHY_SYNC=0 (or des_trophy_sync=0 in settings.ini) turns it off.
//
// USR-DATA: a 16-byte header (u32 7, u32 crc?, u32 size - 16, u32 1), then tagged entries: u32 key (FNV-1a 32 of the
// upper-case name, as the eboot's "Equipment_Inventory%02d_EquipId"), u8 type, the value. Types: 2/6/0c 1 byte, 3 2
// bytes, 4/8/0a 4 bytes, 0f 16 bytes, 0d/0e a LEB128 length and that many bytes (0e: a nested block).
namespace DesSync {

constexpr float kToastSeconds = 3.5f;

uint32_t Fnv(const std::string& name) {
	uint32_t h = 0x811c9dc5u;
	for (const char c: name) h = (h ^ static_cast<uint8_t>(std::toupper(static_cast<unsigned char>(c)))) * 0x01000193u;
	return h;
}

struct Value {
	uint8_t type = 0;
	Bytes   data;
};
using Entries = std::map<uint32_t, Value>; // the top level only (the nested blocks are skipped whole)

bool Parse(const Bytes& d, Entries* out) {
	if (d.size() < 16) return false;
	const uint32_t size = uint32_t(d[8]) | (uint32_t(d[9]) << 8u) | (uint32_t(d[10]) << 16u) | (uint32_t(d[11]) << 24u);
	if (size != d.size() - 16) return false;
	size_t o = 16;
	while (o < d.size()) {
		if (o + 5 > d.size()) return false;
		const uint32_t key = uint32_t(d[o]) | (uint32_t(d[o + 1]) << 8u) | (uint32_t(d[o + 2]) << 16u) | (uint32_t(d[o + 3]) << 24u);
		const uint8_t  t   = d[o + 4];
		o += 5;
		size_t n = 0;
		switch (t) {
			case 0x02: case 0x06: case 0x0c: n = 1; break;
			case 0x03: n = 2; break;
			case 0x04: case 0x08: case 0x0a: n = 4; break;
			case 0x0f: n = 16; break;
			case 0x0d: case 0x0e: {
				uint64_t len = 0;
				for (int shift = 0;; shift += 7) {
					if (o >= d.size() || shift > 28) return false;
					const uint8_t b = d[o++];
					len |= uint64_t(b & 0x7fu) << static_cast<unsigned>(shift);
					if (b < 0x80) break;
				}
				n = static_cast<size_t>(len);
				break;
			}
			default: return false;
		}
		if (n > d.size() - o) return false;
		(*out)[key] = Value {t, Bytes(d.begin() + static_cast<ptrdiff_t>(o), d.begin() + static_cast<ptrdiff_t>(o + n))};
		o += n;
	}
	return true;
}

bool Int(const Entries& e, const std::string& name, int64_t* v) {
	const auto it = e.find(Fnv(name));
	if (it == e.end()) return false;
	const auto& b = it->second.data;
	switch (it->second.type) {
		case 0x02: case 0x06: case 0x0c: *v = b[0]; return true;
		case 0x03: *v = b[0] | (b[1] << 8); return true;
		case 0x04: case 0x08:
			*v = static_cast<int32_t>(uint32_t(b[0]) | (uint32_t(b[1]) << 8u) | (uint32_t(b[2]) << 16u) | (uint32_t(b[3]) << 24u));
			return true;
		default: return false;
	}
}

std::string Fmt(const char* f, int n) {
	char b[96];
	std::snprintf(b, sizeof(b), f, n);
	return b;
}

// What one save file holds that matters here.
struct Proofs {
	std::map<uint32_t, std::string> items; // category | id (0 weapon, 1 armor, 2 ring, 4 goods in the top nibble) -> where
	std::map<int, std::string>      rings; // the game's own ring list (TROPHIES_RING%02d): ring id -> where
	int64_t                         clear_count = 0;
};

void Read(const std::filesystem::path& file, const Entries& e, Proofs* p) {
	const std::string where = Common::PathToString(file.parent_path().filename()) + "/" + Common::PathToString(file.filename());
	int64_t           v = 0, q = 0;
	if (Int(e, "CLEARCOUNT", &v) && v > p->clear_count) p->clear_count = v;
	// The inventory (Equipment_Inventory%02d_Type | _EquipId, _Quantity).
	for (int n = 0, missing = 0; n < 4000 && missing < 8; n++) {
		int64_t type = 0, id = 0;
		if (!Int(e, Fmt("EQUIPMENT_INVENTORY%02d_EQUIPID", n), &id) || !Int(e, Fmt("EQUIPMENT_INVENTORY%02d_TYPE", n), &type)) {
			missing++;
			continue;
		}
		missing = 0;
		if (Int(e, Fmt("EQUIPMENT_INVENTORY%02d_QUANTITY", n), &q) && q <= 0) continue;
		const uint32_t code = static_cast<uint32_t>(type) | static_cast<uint32_t>(id);
		p->items.emplace(code, where + " inventory slot " + std::to_string(n));
	}
	// The storage box (Equipment_StorageBox%02d_EquipTypeId, _LowQuantity + _HighQuantity).
	int64_t count = 0;
	if (!Int(e, "EQUIPMENT_STORAGEBOX_COUNT", &count)) count = 0;
	for (int n = 0; n < count && n < 4000; n++) {
		int64_t code = 0, lo = 0, hi = 0;
		if (!Int(e, Fmt("EQUIPMENT_STORAGEBOX%02d_EQUIPTYPEID", n), &code)) continue;
		(void)Int(e, Fmt("EQUIPMENT_STORAGEBOX%02d_LOWQUANTITY", n), &lo);
		(void)Int(e, Fmt("EQUIPMENT_STORAGEBOX%02d_HIGHQUANTITY", n), &hi);
		if (lo + hi <= 0) continue;
		p->items.emplace(static_cast<uint32_t>(code), where + " storage box slot " + std::to_string(n));
	}
	// Every ring ever picked up, as the game keeps it for the ring trophy.
	if (Int(e, "TROPHIES_RING_COUNT", &count)) {
		for (int n = 0; n < count && n < 1000; n++)
			if (Int(e, Fmt("TROPHIES_RING%02d", n), &v)) p->rings.emplace(static_cast<int>(v), where + " TROPHIES_RING" + Fmt("%02d", n));
	}
}

bool Wanted() {
	if (const char* v = std::getenv("DES_TROPHY_SYNC"); v != nullptr) {
		const std::string x = Lower(v);
		return !(x == "0" || x == "off" || x == "false" || x == "no");
	}
	std::ifstream f("settings.ini", std::ios::binary);
	for (std::string line; std::getline(f, line);) {
		std::string l = Lower(line);
		l.erase(std::remove_if(l.begin(), l.end(), [](unsigned char c) { return std::isspace(c) != 0; }), l.end());
		if (l.rfind("des_trophy_sync=", 0) == 0) {
			const std::string x = l.substr(16);
			return !(x == "0" || x == "off" || x == "false" || x == "no");
		}
	}
	return true;
}

constexpr uint32_t kGoods = 0x40000000u, kRing = 0x20000000u;

} // namespace DesSync

void SyncFromSave(State& s) {
	using namespace DesSync;
	if (s.title_id != "PPSA01341" && s.title_id != "PPSA01342" && s.title_id != "PPSA01340") return;
	if (!Wanted()) {
		std::printf("[trophy-sync] off (DES_TROPHY_SYNC=0)\n");
		return;
	}
	// Every character save of the profile (SAVEDATA0PlayerProfile*/USR-DATA*): read only.
	Proofs                p;
	std::error_code       ec;
	const auto            root  = s.store.parent_path();
	int                   files = 0;
	for (const auto& dir: std::filesystem::directory_iterator(root, ec)) {
		const std::string dn = Common::PathToString(dir.path().filename());
		if (!dir.is_directory(ec) || dn.find("PlayerProfile") == std::string::npos) continue;
		std::error_code ec2;
		for (const auto& f: std::filesystem::directory_iterator(dir.path(), ec2)) {
			const std::string fn = Common::PathToString(f.path().filename());
			if (fn.rfind("USR-DATA", 0) != 0 || !f.is_regular_file(ec2)) continue;
			std::ifstream in(f.path(), std::ios::binary);
			const Bytes   data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
			Entries       e;
			if (!Parse(data, &e)) {
				std::printf("[trophy-sync] %s/%s: not a save this reads (skipped)\n", dn.c_str(), fn.c_str());
				continue;
			}
			files++;
			Read(f.path(), e, &p);
		}
	}
	std::printf("[trophy-sync] read %d save file(s): %zu items, %zu rings in the ring list, clear count %lld\n", files,
	            p.items.size(), p.rings.size(), static_cast<long long>(p.clear_count));
	if (files == 0) return;

	// trophy -> proof (empty: not proven)
	std::vector<std::pair<int, std::string>> proven;
	const auto has = [&](uint32_t code, std::string* where) {
		const auto it = p.items.find(code);
		if (it == p.items.end()) return false;
		*where = it->second;
		return true;
	};
	// 1-16: the boss's demon soul held (the game's goods_caption: "The soul of the <boss> Demon"), or the game cleared
	// (every one of these is mandatory before the Old One).
	struct Boss {
		int         trophy;
		uint32_t    soul;
		const char* soul_name;
	};
	static const Boss bosses[] = {
	    {1, 15, "Lead Demon Soul"},      {2, 16, "Iron Demon Soul"},    {3, 17, "Silver Demon Soul"},
	    {4, 7, "False King Demon Soul"}, {5, 27, "Hard Demon Soul"},    {6, 28, "Searing Demon Soul"},
	    {7, 29, "Dragon Demon Soul"},    {8, 21, "Doll Demon Soul"},    {9, 22, "Mixed Demon Soul"},
	    {10, 23, "Golden Demon Soul"},   {11, 18, "Swollen Demon Soul"}, {12, 19, "Hero Demon Soul"},
	    {13, 20, "Storm Demon Soul"},    {14, 24, "Writhing Demon Soul"}, {15, 25, "Eroded Demon Soul"},
	    {16, 26, "Pureblood Demon Soul"},
	};
	for (const auto& b: bosses) {
		std::string where;
		if (has(kGoods | b.soul, &where)) proven.emplace_back(b.trophy, std::string(b.soul_name) + " (goods " + std::to_string(b.soul) + ") in " + where);
		else if (p.clear_count >= 1)
			proven.emplace_back(b.trophy, "game cleared (CLEARCOUNT=" + std::to_string(p.clear_count) + "): the boss is mandatory before the Old One");
	}
	// 26: the Maiden in Black's demon soul (only from slaying her: the soul-power ending).
	if (std::string where; has(kGoods | 30, &where)) proven.emplace_back(26, "Maiden in Black Demon Soul (goods 30) in " + where);
	// 31: Istarelle (weapons 70300-70305); 33: Northern Regalia (21500-21501).
	for (uint32_t id = 70300; id <= 70305; id++)
		if (std::string where; has(id, &where)) {
			proven.emplace_back(31, "Istarelle (weapon " + std::to_string(id) + ") in " + where);
			break;
		}
	for (uint32_t id = 21500; id <= 21501; id++)
		if (std::string where; has(id, &where)) {
			proven.emplace_back(33, "Northern Regalia (weapon " + std::to_string(id) + ") in " + where);
			break;
		}
	// 36: every ring (accessory_name: 100-126 but 122, the four 996x; either Ring of Longevity) in the game's ring list or held.
	{
		std::vector<std::vector<int>> need;
		for (int r = 100; r <= 126; r++)
			if (r != 122) need.push_back({r});
		need.push_back({9960});
		need.push_back({9961, 9965});
		need.push_back({9962});
		need.push_back({9963});
		int missing = 0;
		for (const auto& any: need) {
			bool ok = false;
			for (const int r: any) {
				std::string where;
				ok = ok || p.rings.count(r) != 0 || has(kRing | static_cast<uint32_t>(r), &where);
			}
			if (!ok) missing++;
		}
		if (missing == 0) proven.emplace_back(36, "all " + std::to_string(need.size()) + " rings in the ring list / inventory");
		else std::printf("[trophy-sync] rings: %d of %zu still missing (no trophy 36)\n", missing, need.size());
	}

	int added = 0;
	for (const auto& [id, proof]: proven) {
		if (s.unlocked.count(id) != 0 || s.trophies.count(id) == 0) continue;
		if (added == 0) {
			// The trophy state before the first sync, once.
			const auto backup = s.store.parent_path() / "trophies.json.pre-sync";
			if (!std::filesystem::exists(backup, ec) && std::filesystem::exists(s.store, ec)) {
				std::filesystem::copy_file(s.store, backup, ec);
				std::printf("[trophy-sync] backup %s%s\n", Common::PathToString(backup).c_str(), ec ? " FAILED" : "");
			}
		}
		Record(s, id, kToastSeconds);
		added++;
		std::printf("[trophy-sync] unlocked %d %s: %s\n", id, s.trophies[id].info.name.c_str(), proof.c_str());
		LOGF("[trophy-sync] unlocked %d %s: %s\n", id, s.trophies[id].info.name.c_str(), proof.c_str());
	}
	// The platinum, as Unlock gives it, when the sync completed the set.
	int  platinum = -1;
	bool rest     = !s.trophies.empty();
	for (const auto& [id, e]: s.trophies) {
		if (e.info.grade == 1) platinum = id;
		else if (id != kNexusTrophyId && s.unlocked.count(id) == 0) rest = false;
	}
	if (platinum >= 0 && rest && s.unlocked.count(platinum) == 0) {
		Record(s, platinum, kToastSeconds);
		added++;
		std::printf("[trophy-sync] unlocked %d %s: every other trophy held\n", platinum, s.trophies[platinum].info.name.c_str());
	}
	if (added > 0) SaveStore(s);
	std::printf("[trophy-sync] %d new, %zu proven in all\n", added, proven.size());
}

} // namespace

bool Unlock(int trophy_id, int* platinum_id) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	Load(s);
	if (platinum_id != nullptr) *platinum_id = -1;
	if (s.unlocked.count(trophy_id) != 0) return false;
	Record(s, trophy_id);
	// The platinum: when every other trophy is earned.
	int  platinum = -1;
	bool rest     = !s.trophies.empty();
	for (const auto& [id, e]: s.trophies) {
		if (e.info.grade == 1) platinum = id;
		else if (id != kNexusTrophyId && s.unlocked.count(id) == 0) rest = false;
	}
	if (platinum >= 0 && rest && s.unlocked.count(platinum) == 0) {
		Record(s, platinum);
		if (platinum_id != nullptr) *platinum_id = platinum;
	}
	SaveStore(s);
	return true;
}

bool IsUnlocked(int trophy_id) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	Load(s);
	return s.unlocked.count(trophy_id) != 0;
}

bool GetInfo(int trophy_id, Info* info) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	Load(s);
	const auto it = s.trophies.find(trophy_id);
	if (it == s.trophies.end()) return false;
	if (info != nullptr) *info = it->second.info;
	return true;
}

int Count() {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	Load(s);
	return static_cast<int>(s.trophies.size());
}

bool Snapshot(std::vector<Info>* all, std::vector<int>* unlocked) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (!s.loaded || s.trophies.empty()) return false;
	if (all != nullptr) {
		all->clear();
		for (const auto& [id, e]: s.trophies) all->push_back(e.info);
	}
	if (unlocked != nullptr) {
		unlocked->clear();
		for (const auto& [id, t]: s.unlocked) unlocked->push_back(id);
	}
	return true;
}

int TrySnapshot(std::vector<Info>* all, std::map<int, int64_t>* unlock_times) {
	auto&            s = S();
	std::unique_lock lock(s.mutex, std::try_to_lock);
	if (!lock.owns_lock()) return 0;
	if (!s.loaded) return -1;
	if (s.trophies.empty()) return -2;
	if (all != nullptr) {
		all->clear();
		for (const auto& [id, e]: s.trophies) all->push_back(e.info);
	}
	if (unlock_times != nullptr) *unlock_times = s.unlocked;
	return 1;
}

bool UnlockTimes(std::map<int, int64_t>* unlock_times) {
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (!s.loaded) return false;
	if (unlock_times != nullptr) *unlock_times = s.unlocked;
	return true;
}

std::string GameTitle() {
	auto&            s = S();
	std::unique_lock lock(s.mutex, std::try_to_lock);
	return lock.owns_lock() && s.loaded ? s.title : std::string();
}

bool BuildArt(int cell, Art* art) {
	if (art == nullptr) return false;
	*art = Art {};
	std::filesystem::path ucp, sce_sys;
	std::vector<int>      ids;
	{
		auto&           s = S();
		std::lock_guard lock(s.mutex);
		Load(s);
		ucp     = s.ucp;
		sce_sys = s.sce_sys;
		for (const auto& [id, e]: s.trophies) ids.push_back(id);
	}
	if (ucp.empty() || ids.empty()) return false;
	std::map<std::string, Bytes> files;
	if (!ReadUcp(ucp, files)) return false;
	art->cell   = std::clamp(cell, 32, 512);
	art->pad    = 2;
	art->stride = art->cell + art->pad * 2;
	const int slots = static_cast<int>(ids.size()) + 1;
	art->cols   = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(slots)))));
	const int rows = (slots + art->cols - 1) / art->cols;
	art->width  = art->cols * art->stride;
	art->height = rows * art->stride;
	art->pixels.assign(static_cast<size_t>(art->width) * art->height, 0);
	int next = 0;
	for (const int id: ids) {
		const Bytes* png = IconFile(files, id);
		if (png == nullptr && id == kNexusTrophyId) png = IconFile(files, 1); // (the port's own: trophy 1's icon, as the popup)
		if (png != nullptr && DecodeInto(*png, *art, next)) art->slots[id] = next++;
	}
	{
		std::error_code ec;
		const auto      icon0 = sce_sys / "icon0.png";
		if (std::filesystem::exists(icon0, ec)) {
			std::ifstream f(icon0, std::ios::binary);
			const Bytes   png((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
			if (!png.empty() && DecodeInto(png, *art, next)) art->game_slot = next++;
		}
	}
	std::printf("Trophy: icon atlas %dx%d (%d px icons, %zu trophies%s)\n", art->width, art->height, art->cell, art->slots.size(),
	            art->game_slot >= 0 ? ", game icon" : "");
	return true;
}

void Preload() {
	// (At the start, on its own thread: the init and package lines in every log, and no reading at the first unlock.)
	std::thread([] {
		auto&           s = S();
		std::lock_guard lock(s.mutex);
		Load(s);
	}).detach();
}

std::shared_ptr<const Toast> CurrentToast(float* seconds) {
	auto& s = S();
	// (Nearly every frame: nothing queued. Read without waiting on an unlock in progress.)
	std::unique_lock lock(s.mutex, std::try_to_lock);
	if (!lock.owns_lock() || s.queue.empty()) return nullptr;
	const auto now = std::chrono::steady_clock::now();
	if (s.shown_at == std::chrono::steady_clock::time_point {}) {
		s.shown_at = now;
		PlayChime();
	}
	float t = std::chrono::duration<float>(now - s.shown_at).count();
	if (t >= s.queue.front()->seconds) {
		s.queue.pop_front();
		s.shown_at = {};
		if (s.queue.empty()) return nullptr;
		s.shown_at = now;
		PlayChime();
		t = 0.0f;
	}
	if (seconds != nullptr) *seconds = t;
	return s.queue.front();
}

} // namespace Libs::Trophy
