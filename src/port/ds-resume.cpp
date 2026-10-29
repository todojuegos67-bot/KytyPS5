// "Demon's Souls" PC port (ds-pc-port8): F1 = save and quit; JUGAR = back where F1 left (ds-resume.h).

#include "port/ds-resume.h"

#include "port/ds-port.h"

#include "kernel/fileSystem.h"
#include "libs/controller.h"
#include "libs/saveDataInfo.h"
#include "loader/demonsSoulsWarp.h"
#include "local/port-live.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace Port::Resume {

namespace fs = std::filesystem;

namespace {

std::string U8(const fs::path& p) {
	const auto s = p.u8string();
	return std::string(s.begin(), s.end());
}

fs::path P(const std::string& s) {
	return fs::path(std::u8string(s.begin(), s.end()));
}

uint32_t Fnv1a(const std::string& data) {
	uint32_t hash = 0x811c9dc5u;
	for (const char c: data) hash = (hash ^ static_cast<uint8_t>(c)) * 0x01000193u;
	return hash;
}

std::string ReadAll(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int64_t MTime(const fs::path& path) {
	std::error_code ec;
	const auto      t = fs::last_write_time(path, ec);
	return ec ? 0 : static_cast<int64_t>(t.time_since_epoch().count());
}

std::string Stamp(const char* format) {
	const std::time_t now = std::time(nullptr);
	std::tm           local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[64] {};
	std::strftime(text, sizeof(text), format, &local);
	return text;
}

fs::path SaveDir(const fs::path& root, const std::string& title) {
	return root / "_SaveData" / P(title);
}

// The character being played: the PlayerProfile slot whose USR-DATA was written last.
std::string NewestProfile(const fs::path& root, const std::string& title) {
	std::string     best;
	int64_t         best_time = 0;
	std::error_code ec;
	for (const auto& entry: fs::directory_iterator(SaveDir(root, title), ec)) {
		const std::string name = U8(entry.path().filename());
		if (!entry.is_directory(ec) || name.find("PlayerProfile") == std::string::npos) continue;
		const auto file = entry.path() / "USR-DATA";
		if (!fs::is_regular_file(file, ec)) continue;
		const int64_t t = MTime(file);
		if (best.empty() || t > best_time) {
			best      = name;
			best_time = t;
		}
	}
	return best;
}

bool Describe(const fs::path& save, Info* info) {
	std::error_code ec;
	if (!fs::is_regular_file(save, ec)) return false;
	const std::string data = ReadAll(save);
	if (data.empty()) return false;
	info->size    = data.size();
	info->mtime   = MTime(save);
	info->hash    = Fnv1a(data);
	info->map_uid = Loader::DemonsSoulsWarp::SaveMapUid(save);
	return true;
}

bool Write(const fs::path& root, const Info& info) {
	std::ostringstream o;
	char               place[160] {};
	std::snprintf(place, sizeof(place), "x=%.4f\ny=%.4f\nz=%.4f\ngiro=%.4f\n", info.x, info.y, info.z, info.yaw);
	o << "; Demon's Souls PC: lo que F1 (guardar y salir) dejo para JUGAR. Borrarlo: JUGAR no continua solo.\n"
	  << "perfil=" << info.profile << "\n"
	  << "tamano=" << info.size << "\n"
	  << "fecha=" << info.mtime << "\n"
	  << "suma=" << info.hash << "\n"
	  << "mapa=" << info.map_uid << "\n"
	  << "lugar=" << (info.place ? 1 : 0) << "\n"
	  << place << "cuando=" << info.when << "\n";
	const fs::path  file = File(root);
	fs::path        tmp  = file;
	tmp += ".tmp";
	std::error_code ec;
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		out << o.str();
		out.flush();
		if (!out) return false;
	}
	fs::rename(tmp, file, ec);
	if (ec) fs::remove(tmp, ec);
	return !ec;
}

// _SaveData to _SaveData-copias\F1-<date-time>: copied into a .tmp folder, renamed when complete; the last 3 kept.
bool BackupF1(const fs::path& root, std::string* where) {
	std::error_code ec;
	const fs::path  src = root / "_SaveData";
	if (!fs::is_directory(src, ec)) return false;
	const fs::path dir  = root / "_SaveData-copias";
	const std::string name = "F1-" + Stamp("%Y%m%d-%H%M%S");
	const fs::path tmp  = dir / (name + ".tmp");
	const fs::path done = dir / name;
	fs::create_directories(dir, ec);
	fs::remove_all(tmp, ec);
	ec.clear();
	fs::copy(src, tmp, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
	if (ec) {
		std::printf("[port] F1: the copy of _SaveData failed (%s); nothing kept of it\n", ec.message().c_str());
		fs::remove_all(tmp, ec);
		return false;
	}
	fs::remove_all(done, ec);
	fs::rename(tmp, done, ec);
	if (ec) {
		fs::remove_all(tmp, ec);
		return false;
	}
	*where = U8(done);
	// The F1 copies: the newest 3 (names sort by time); unfinished .tmp ones go.
	std::vector<fs::path> copies;
	for (const auto& entry: fs::directory_iterator(dir, ec)) {
		const std::string n = U8(entry.path().filename());
		if (n.rfind("F1-", 0) != 0 || !entry.is_directory(ec)) continue;
		if (n.size() > 4 && n.substr(n.size() - 4) == ".tmp") {
			fs::remove_all(entry.path(), ec);
			continue;
		}
		copies.push_back(entry.path());
	}
	std::sort(copies.begin(), copies.end());
	while (copies.size() > 3) {
		fs::remove_all(copies.front(), ec);
		copies.erase(copies.begin());
	}
	return true;
}

std::atomic<bool> g_quit_started {false};
std::mutex        g_text_mutex;
std::string       g_text;

void SetText(const char* text) {
	std::scoped_lock lock(g_text_mutex);
	g_text = text;
}

void QuitWorker(fs::path root) {
	using namespace std::chrono_literals;
	// The player's place now (before anything waits).
	float      xyz[3] {}, yaw = 0;
	const bool place = Loader::DemonsSoulsWarp::PlayerPlace(xyz, &yaw);
	std::printf("[port] F1: save and quit (%s)\n", place ? "the player's place read" : "no player now: no place kept");
	std::fflush(stdout);
	SetText("Esperando a que el juego termine de guardar...");
	const bool idle = Libs::SaveData::BlockForExit(12000, 800);
	Libs::SaveData::FlushAll();
	SetText("Copia de seguridad de las partidas...");
	std::string where;
	if (BackupF1(root, &where)) std::printf("[port] F1: _SaveData copied to %s\n", where.c_str());
	const std::string title   = Libs::SaveData::TitleId();
	const std::string profile = NewestProfile(root, title);
	if (!profile.empty()) {
		Info info;
		info.profile = profile;
		if (Describe(SaveDir(root, title) / P(profile) / "USR-DATA", &info)) {
			info.place = place && info.map_uid != 0;
			info.x     = xyz[0];
			info.y     = xyz[1];
			info.z     = xyz[2];
			info.yaw   = static_cast<float>(yaw * 180.0 / 3.14159265358979);
			info.when  = Stamp("%Y-%m-%d %H:%M");
			const bool ok = Write(root, info);
			std::printf("[port] F1: %s %s (map 0x%08x, place %s %.2f %.2f %.2f)%s\n", ok ? "resume written for" : "resume NOT written for",
			            profile.c_str(), info.map_uid, info.place ? "kept" : "unknown", info.x, info.y, info.z,
			            idle ? "" : " after waiting for a busy save");
		}
	} else {
		std::printf("[port] F1: no character save yet: nothing to continue\n");
	}
	std::fflush(stdout);
	SetText("Partida guardada. Cerrando el juego...");
	std::this_thread::sleep_for(700ms);
	RequestQuit();
}

// ---- The start (JUGAR). ----

struct Start {
	bool     armed = false;  // Prepare found a resume
	bool     place = false;  // and its place may be used
	Info     info;
	fs::path root, file;
	std::atomic<bool> started {false};
} g_start;

// The character is in the world: its place readable twice in a row (01.007.000), or the game wrote its save (what it
// does right after loading a character; the title screen only reads).
bool Loaded(uint64_t writes_before, int* place_hits) {
	float xyz[3] {}, yaw = 0;
	*place_hits = Loader::DemonsSoulsWarp::PlayerPlace(xyz, &yaw) ? *place_hits + 1 : 0;
	return *place_hits >= 2 || Libs::LibKernel::FileSystem::OpenSaveWrites() > 0 ||
	       Libs::LibKernel::FileSystem::SaveWriteOpens() > writes_before;
}

void AutoContinue() {
	using namespace std::chrono_literals;
	std::this_thread::sleep_for(2500ms);
	if (g_start.place) {
		if (Loader::DemonsSoulsWarp::IsPlaceBuild()) {
			const auto& i = g_start.info;
			if (!Loader::DemonsSoulsWarp::ArmResume(i.profile, i.map_uid, i.x, i.y, i.z, i.yaw))
				std::printf("[port] continue: the place could not be armed: the game's own saved place\n");
		} else {
			std::printf("[port] continue: not 01.007.000: the game's own saved place\n");
		}
	}
	const uint64_t writes = Libs::LibKernel::FileSystem::SaveWriteOpens();
	int            hits   = 0;
	int            presses = 0;
	const auto     until  = std::chrono::steady_clock::now() + 90s;
	bool           loaded = false;
	while (std::chrono::steady_clock::now() < until) {
		if (Loaded(writes, &hits)) {
			loaded = true;
			break;
		}
		// Cross: skips the intro, passes "press a button" and picks Continue (the title's first item with a save).
		if (hits == 0) {
			Libs::Controller::SetScriptedInput(-1, -1, -1, -1, Libs::Controller::PAD_BUTTON_CROSS, 0.12);
			presses++;
		}
		std::this_thread::sleep_for(1000ms);
	}
	std::printf("[port] continue: %s after %d Cross press(es)%s\n", loaded ? "the character is loaded" : "not loaded within 90 s, stopped",
	            presses, Loader::DemonsSoulsWarp::ResumeServed() ? "; its save read at the F1 place" : "");
	std::fflush(stdout);
	if (loaded) {
		std::error_code ec;
		fs::path used = g_start.file;
		used += ".used";
		fs::rename(g_start.file, used, ec);
		if (ec) fs::remove(g_start.file, ec);
	}
}

} // namespace

fs::path File(const fs::path& root) {
	return root / "_SaveData" / "port-resume.ini";
}

bool Read(const fs::path& root, Info* info) {
	std::ifstream in(File(root), std::ios::binary);
	if (!in) return false;
	std::map<std::string, std::string> v;
	std::string                        line;
	while (std::getline(in, line)) {
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
		if (line.empty() || line[0] == ';') continue;
		const auto eq = line.find('=');
		if (eq != std::string::npos) v[line.substr(0, eq)] = line.substr(eq + 1);
	}
	if (v["perfil"].empty()) return false;
	info->profile = v["perfil"];
	info->size    = std::strtoull(v["tamano"].c_str(), nullptr, 10);
	info->mtime   = std::strtoll(v["fecha"].c_str(), nullptr, 10);
	info->hash    = static_cast<uint32_t>(std::strtoul(v["suma"].c_str(), nullptr, 10));
	info->map_uid = static_cast<uint32_t>(std::strtoul(v["mapa"].c_str(), nullptr, 10));
	info->place   = v["lugar"] == "1";
	info->x       = std::strtof(v["x"].c_str(), nullptr);
	info->y       = std::strtof(v["y"].c_str(), nullptr);
	info->z       = std::strtof(v["z"].c_str(), nullptr);
	info->yaw     = std::strtof(v["giro"].c_str(), nullptr);
	info->when    = v["cuando"];
	return true;
}

std::string Summary(const fs::path& root) {
	Info info;
	if (!Read(root, &info)) return {};
	return info.when;
}

void StartSaveAndQuit(const fs::path& root) {
	if (g_quit_started.exchange(true)) return;
	SetText("Guardando la partida...");
	PortLive::menu_open.store(true, std::memory_order_relaxed); // (no generated frames, the game gets no input)
	std::thread(QuitWorker, root).detach();
}

bool SaveAndQuitActive() {
	return g_quit_started.load();
}

std::string SaveAndQuitText() {
	std::scoped_lock lock(g_text_mutex);
	return g_text;
}

void Prepare(const fs::path& root, const std::string& title_id, bool enabled) {
	Info info;
	if (!enabled || title_id.empty() || !Read(root, &info)) return;
	const fs::path save = SaveDir(root, title_id) / P(info.profile) / "USR-DATA";
	Info           now;
	if (!Describe(save, &now)) {
		std::printf("[port] continue: %s has no save now: the title screen as usual\n", info.profile.c_str());
		return;
	}
	const bool same = now.size == info.size && now.mtime == info.mtime && now.hash == info.hash;
	g_start.info  = info;
	g_start.place = same && info.place && info.map_uid != 0 && info.map_uid == now.map_uid;
	g_start.root  = root;
	g_start.file  = File(root);
	g_start.armed = true;
	std::printf("[port] continue: F1 of %s (%s): Continue by itself%s\n", info.when.c_str(), info.profile.c_str(),
	            g_start.place ? ", at the place of F1" : (same ? "" : "; the save changed since F1: its own place"));
	std::fflush(stdout);
}

void Presented() {
	if (!g_start.armed || g_start.started.exchange(true)) return;
	std::thread(AutoContinue).detach();
}

} // namespace Port::Resume
