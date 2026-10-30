#include "port/ds-update.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <nlohmann/json.hpp>

#include <windows.h>
#include <shellapi.h>
#include <winhttp.h>
#endif

#ifndef KYTY_DS_PORT_VERSION
#define KYTY_DS_PORT_VERSION "DemonsSoulsPC-dev"
#endif

namespace Port::Update {

namespace {

std::string U8(const fs::path& p) {
	const auto s = p.generic_u8string();
	return std::string(s.begin(), s.end());
}

fs::path P(const std::string& s) {
	return fs::path(std::u8string(s.begin(), s.end()));
}

std::string Lower(std::string s) {
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return s;
}

bool EndsWith(const std::string& s, const std::string& end) {
	return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}

std::string ReadFile(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool WriteFile(const fs::path& path, const std::string& text) {
	const fs::path tmp = path.parent_path() / (path.filename().native() + fs::path(".tmp").native());
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return false;
		out << text;
		if (!out) return false;
	}
	std::error_code ec;
	fs::rename(tmp, path, ec);
	if (ec) fs::remove(tmp, ec);
	return !ec;
}

std::string Trim(std::string s) {
	const auto sp = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
	while (!s.empty() && sp(static_cast<unsigned char>(s.back()))) s.pop_back();
	size_t i = 0;
	while (i < s.size() && sp(static_cast<unsigned char>(s[i]))) i++;
	return s.substr(i);
}

// A version as a folder name.
std::string Safe(std::string v) {
	for (char& c: v)
		if (std::string("\\/:*?\"<>|").find(c) != std::string::npos || static_cast<unsigned char>(c) < 32) c = '_';
	v = Trim(v);
	if (v.empty() || v == "." || v == "..") v = "sin-version";
	return v;
}

int64_t Stamp(const fs::path& p) {
	std::error_code ec;
	const auto t = fs::last_write_time(p, ec);
	return ec ? 0 : static_cast<int64_t>(t.time_since_epoch().count());
}

// The user's folders and files (the launcher's $ExcludeDirs / $ExcludeFiles, for the port).
const char* const kUserDirs[]  = {"_PipelineCache", "_SaveData", "_SaveData-copias", "_backup_update", "logs", "config", "profiles"};
// (10-28) amd_fidelityfx_vk.dll: the patched AMD DLL (FSR 3 frame generation x3 / x4) is never replaced on update or restore.
const char* const kUserFiles[] = {"game-path.txt", "ram16.txt", "kbm_curve.txt", "version.txt", "amd_fidelityfx_vk.dll"};

} // namespace

const char* CurrentVersion() {
	return KYTY_DS_PORT_VERSION;
}

std::string BuildIso() {
	const std::string d = __DATE__, t = __TIME__;
	static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
	int m = 0;
	for (int i = 0; i < 12; i++) if (d.compare(0, 3, months[i]) == 0) m = i + 1;
	char buf[32];
	std::snprintf(buf, sizeof(buf), "%s-%02d-%02dT%s", d.substr(7, 4).c_str(), m, std::atoi(d.substr(4, 2).c_str()), t.c_str());
	return buf;
}

const char* BuildStamp() {
	static std::string stamp = [] {
		// __DATE__ "Oct 30 2026", __TIME__ "19:40:12"
		const std::string d = __DATE__, t = __TIME__;
		static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
		int m = 0;
		for (int i = 0; i < 12; i++) if (d.compare(0, 3, months[i]) == 0) m = i + 1;
		char buf[64];
		std::snprintf(buf, sizeof(buf), "%02d/%02d/%s %s UTC", std::atoi(d.substr(4, 2).c_str()), m, d.substr(7, 4).c_str(), t.substr(0, 5).c_str());
		return std::string(buf);
	}();
	return stamp.c_str();
}

std::string TagDate(const std::string& tag) {
	// The first 8 digits in a row that start with 20 (YYYYMMDD), as the launcher's '(20\d{6})'.
	for (size_t i = 0; i + 8 <= tag.size(); i++) {
		if (tag[i] != '2' || tag[i + 1] != '0') continue;
		bool ok = true;
		for (size_t k = 0; k < 8 && ok; k++) ok = std::isdigit(static_cast<unsigned char>(tag[i + k])) != 0;
		if (ok) return tag.substr(i, 8);
	}
	return {};
}

bool IsExcluded(const fs::path& rel, const fs::path& game_rel) {
	const fs::path norm = rel.lexically_normal();
	if (norm.empty()) return true;
	const std::string first = Lower(U8(*norm.begin()));
	if (first == ".." || norm.is_absolute()) return true; // (outside the folder: never)
	for (const char* d: kUserDirs)
		if (first == Lower(d)) return true;
	if (!game_rel.empty()) {
		// The game's folder inside the port's folder: neither it nor anything under it.
		const auto g = game_rel.lexically_normal();
		auto       a = norm.begin();
		auto       b = g.begin();
		for (; a != norm.end() && b != g.end(); ++a, ++b)
			if (Lower(U8(*a)) != Lower(U8(*b))) break;
		if (b == g.end()) return true;
	}
	const std::string leaf = Lower(U8(norm.filename()));
	for (const char* f: kUserFiles)
		if (leaf == f) return true;
	for (const char* end: {".ini", ".seeds", ".seeds.program", ".seeds.new", ".prewarm2", ".old"})
		if (EndsWith(leaf, end)) return true;
	return false;
}

fs::path FindReleaseRoot(const fs::path& extracted) {
	std::error_code ec;
	if (fs::is_regular_file(extracted / P(kExeName), ec)) return extracted;
	fs::path only;
	int      dirs = 0;
	for (const auto& e: fs::directory_iterator(extracted, ec)) {
		if (e.is_directory(ec)) {
			dirs++;
			only = e.path();
		}
	}
	if (dirs == 1 && fs::is_regular_file(only / P(kExeName), ec)) return only;
	return {};
}

std::string MergeIni(const std::string& user, const std::string& shipped) {
	const auto lines = [](const std::string& text) {
		std::vector<std::string> out;
		std::stringstream        in(text);
		std::string              line;
		while (std::getline(in, line)) {
			if (!line.empty() && line.back() == '\r') line.pop_back();
			out.push_back(line);
		}
		return out;
	};
	const auto key_of = [](const std::string& raw) -> std::string {
		std::string l = Trim(raw);
		if (l.size() >= 3 && static_cast<unsigned char>(l[0]) == 0xEF) l = l.substr(3);
		if (l.empty() || l[0] == ';' || l[0] == '#' || l[0] == '[') return {};
		const auto eq = l.find('=');
		return eq == std::string::npos ? std::string() : Trim(l.substr(0, eq));
	};
	const auto section_of = [](const std::string& raw) -> std::string {
		const std::string l = Trim(raw);
		return l.size() >= 2 && l.front() == '[' && l.back() == ']' ? l : std::string();
	};
	std::vector<std::string> u = lines(user);
	std::set<std::string>    have;
	for (const auto& l: u)
		if (const auto k = key_of(l); !k.empty()) have.insert(k);
	// What the user lacks: per section of the shipped file, each key with its comment lines.
	struct Add {
		std::string              section;
		std::vector<std::string> text;
	};
	std::vector<Add>         adds;
	std::string              section;
	std::vector<std::string> comments;
	for (const auto& l: lines(shipped)) {
		if (const auto s = section_of(l); !s.empty()) {
			section = s;
			comments.clear();
			continue;
		}
		const std::string t = Trim(l);
		if (!t.empty() && (t[0] == ';' || t[0] == '#')) {
			comments.push_back(l);
			continue;
		}
		const auto k = key_of(l);
		if (!k.empty() && have.count(k) == 0) {
			if (adds.empty() || adds.back().section != section) adds.push_back({section, {}});
			adds.back().text.insert(adds.back().text.end(), comments.begin(), comments.end());
			adds.back().text.push_back(l);
			have.insert(k);
		}
		comments.clear();
	}
	if (adds.empty()) return user;
	for (const auto& add: adds) {
		// After the last line of that section in the user's file, else a new section at the end.
		int at = -1;
		for (int i = 0, in = 0; i < static_cast<int>(u.size()); i++) {
			if (const auto s = section_of(u[i]); !s.empty()) in = s == add.section ? 1 : 0;
			else if (in != 0 && !Trim(u[i]).empty()) at = i;
		}
		if (add.section.empty() && at < 0) {
			for (int i = 0; i < static_cast<int>(u.size()) && section_of(u[i]).empty(); i++)
				if (!Trim(u[i]).empty()) at = i;
		}
		if (at >= 0) {
			u.insert(u.begin() + at + 1, add.text.begin(), add.text.end());
		} else {
			if (!u.empty() && !Trim(u.back()).empty()) u.emplace_back();
			if (!add.section.empty()) u.push_back(add.section);
			u.insert(u.end(), add.text.begin(), add.text.end());
		}
	}
	std::string out;
	for (const auto& l: u) out += l + "\n";
	return out;
}

std::vector<Backup> ListBackups(const fs::path& root) {
	std::vector<Backup> list;
	std::error_code     ec;
	for (const auto& e: fs::directory_iterator(root / kBackupDir, ec)) {
		if (!e.is_directory(ec) || !fs::is_regular_file(e.path() / P(kExeName), ec)) continue;
		Backup b;
		b.name    = U8(e.path().filename());
		b.version = Trim(ReadFile(e.path() / ".version"));
		if (b.version.empty()) b.version = b.name;
		b.time = std::max(Stamp(e.path() / ".version"), Stamp(e.path()));
		list.push_back(b);
	}
	std::sort(list.begin(), list.end(), [](const Backup& a, const Backup& b) { return a.time != b.time ? a.time > b.time : a.name > b.name; });
	return list;
}

void PruneBackups(const fs::path& root, int keep, const std::vector<std::string>& protect) {
	std::error_code ec;
	std::vector<std::pair<int64_t, fs::path>> dirs;
	int                                       protected_found = 0;
	for (const auto& e: fs::directory_iterator(root / kBackupDir, ec)) {
		if (!e.is_directory(ec)) continue;
		if (std::find(protect.begin(), protect.end(), U8(e.path().filename())) != protect.end()) {
			protected_found++;
			continue;
		}
		dirs.emplace_back(std::max(Stamp(e.path() / ".version"), Stamp(e.path())), e.path());
	}
	std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	keep = std::max(0, keep - protected_found);
	for (size_t i = static_cast<size_t>(keep); i < dirs.size(); i++) fs::remove_all(dirs[i].second, ec);
}

namespace {

Result DoInstall(const fs::path& root, const fs::path& src, const std::string& old_version, const std::string& new_version,
                 const Options& options, const std::string& keep_also) {
	Result          r;
	std::error_code ec;
	if (!fs::is_regular_file(src / P(kExeName), ec)) {
		r.error = std::string("La version nueva no trae ") + kExeName + ". No se ha cambiado nada.";
		return r;
	}
	// What the new version brings (not the user's).
	std::vector<fs::path> files;
	for (auto it = fs::recursive_directory_iterator(src, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
		const fs::path rel = fs::relative(it->path(), src, ec);
		if (it->is_directory(ec)) {
			if (IsExcluded(rel, options.game_rel)) it.disable_recursion_pending();
			continue;
		}
		if (!it->is_regular_file(ec) || rel == ".version" || IsExcluded(rel, options.game_rel)) continue;
		files.push_back(rel);
	}
	if (ec) {
		r.error = "No se pudo leer la version nueva (" + ec.message() + "). No se ha cambiado nada.";
		return r;
	}
	std::sort(files.begin(), files.end());

	// 1) The backup: the exe and DLLs of the folder and every file the new version replaces.
	const fs::path base = root / kBackupDir;
	std::string    name = Safe(old_version);
	for (int n = 2; fs::exists(base / P(name), ec); n++) name = Safe(old_version) + "-" + std::to_string(n);
	const fs::path bdir = base / P(name);
	try {
		fs::create_directories(bdir);
		std::set<fs::path> saved;
		const auto save = [&](const fs::path& rel) {
			if (saved.count(rel) != 0) return;
			fs::create_directories((bdir / rel).parent_path());
			fs::copy_file(root / rel, bdir / rel, fs::copy_options::overwrite_existing);
			saved.insert(rel);
		};
		for (const auto& e: fs::directory_iterator(root)) {
			if (!e.is_regular_file()) continue;
			const std::string leaf = Lower(U8(e.path().filename()));
			if ((EndsWith(leaf, ".exe") || EndsWith(leaf, ".dll")) && !IsExcluded(e.path().filename(), options.game_rel))
				save(e.path().filename());
		}
		for (const auto& rel: files)
			if (fs::is_regular_file(root / rel)) save(rel);
		if (!WriteFile(bdir / ".version", old_version)) throw std::runtime_error("no se pudo escribir la copia de seguridad");
	} catch (const std::exception& e) {
		fs::remove_all(bdir, ec);
		r.error = std::string("No se pudo hacer la copia de seguridad en ") + kBackupDir + " (" + e.what() + "). No se ha cambiado nada.";
		return r;
	}

	// 2) The new files: one in use (the running exe, its DLLs) is renamed to .old first.
	struct Step {
		fs::path rel;
		fs::path old; // empty: created
	};
	std::vector<Step> done;
	const fs::path    ini      = root / "settings.ini";
	const bool        had_ini  = fs::exists(ini, ec);
	const std::string user_ini = had_ini ? ReadFile(ini) : std::string();
	bool              ini_written = false;
	try {
		for (const auto& rel: files) {
			const fs::path dst = root / rel;
			fs::create_directories(dst.parent_path());
			Step step {rel, {}};
			if (fs::exists(dst)) {
				fs::path old = dst.parent_path() / (dst.filename().native() + fs::path(".old").native());
				for (int n = 2; fs::exists(old); n++) {
					std::error_code rm;
					if (fs::remove(old, rm) && !rm) break; // (a .old of an earlier update no longer in use)
					old = dst.parent_path() / (dst.filename().native() + fs::path(".old" + std::to_string(n)).native());
				}
				fs::rename(dst, old);
				step.old = old;
			}
			done.push_back(step);
			if (!options.fail_on.empty() && U8(rel) == options.fail_on) throw std::runtime_error("fallo de prueba en " + options.fail_on);
			fs::copy_file(src / rel, dst, fs::copy_options::overwrite_existing);
		}
		// settings.ini: the user's values, the new keys merged in.
		if (fs::is_regular_file(src / "settings.ini")) {
			const std::string shipped = ReadFile(src / "settings.ini");
			const std::string merged  = had_ini ? MergeIni(user_ini, shipped) : shipped;
			if (!had_ini || merged != user_ini) {
				if (!WriteFile(ini, merged)) throw std::runtime_error("no se pudo escribir settings.ini");
				ini_written = true;
			}
		}
	} catch (const std::exception& e) {
		// Rollback: the new files out, the old ones back.
		int errors = 0;
		for (auto it = done.rbegin(); it != done.rend(); ++it) {
			std::error_code rm;
			const fs::path  dst = root / it->rel;
			fs::remove(dst, rm);
			if (!it->old.empty()) {
				fs::rename(it->old, dst, rm);
				if (rm) {
					fs::copy_file(bdir / it->rel, dst, fs::copy_options::overwrite_existing, rm);
					if (rm) errors++;
				}
			}
		}
		if (ini_written) {
			if (had_ini ? !WriteFile(ini, user_ini) : !fs::remove(ini, ec)) errors++;
		}
		if (errors == 0) {
			fs::remove_all(bdir, ec);
			r.error = std::string("No se pudo actualizar (") + e.what() + "). Se ha deshecho todo: la carpeta sigue como estaba.";
		} else {
			r.error = std::string("No se pudo actualizar (") + e.what() + ") y no se pudo deshacer todo: la version anterior esta en " +
			          kBackupDir + "\\" + name + ".";
		}
		return r;
	}
	// 3) The .old files: deleted now when they can be (the ones in use, at the next start), the old backups pruned.
	for (const auto& step: done)
		if (!step.old.empty()) fs::remove(step.old, ec);
	if (keep_also.empty()) PruneBackups(root, kKeepBackups, {name});
	else PruneBackups(root, kKeepBackups, {name, keep_also}); // (restore: the restored one stays too)
	r.ok      = true;
	r.backup  = name;
	r.version = new_version;
	return r;
}

} // namespace

Result Install(const fs::path& root, const fs::path& src, const std::string& old_version, const std::string& new_version,
               const Options& options) {
	try {
		return DoInstall(root, src, old_version, new_version, options, {});
	} catch (const std::exception& e) {
		Result r;
		r.error = std::string("No se pudo actualizar: ") + e.what();
		return r;
	}
}

Result Restore(const fs::path& root, const std::string& name, const std::string& current_version, const Options& options) {
	Result r;
	if (name.empty() || name.find_first_of("\\/") != std::string::npos || name == "." || name == "..") {
		r.error = "Copia no valida.";
		return r;
	}
	const fs::path src = root / kBackupDir / P(name);
	std::error_code ec;
	if (!fs::is_regular_file(src / P(kExeName), ec)) {
		r.error = std::string("La copia ") + name + " no tiene " + kExeName + ".";
		return r;
	}
	std::string version = Trim(ReadFile(src / ".version"));
	if (version.empty()) version = name;
	try {
		r = DoInstall(root, src, current_version, version, options, name);
	} catch (const std::exception& e) {
		r.ok    = false;
		r.error = std::string("No se pudo restaurar: ") + e.what();
	}
	if (!r.ok && r.error.rfind("No se pudo actualizar", 0) == 0) r.error.replace(0, 21, "No se pudo restaurar");
	return r;
}

int RemoveOldFiles(const fs::path& root) {
	int             removed = 0;
	std::error_code ec;
	std::vector<fs::path> olds;
	for (auto it = fs::recursive_directory_iterator(root, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
		const fs::path rel = fs::relative(it->path(), root, ec);
		if (it->is_directory(ec)) {
			// (Not into the user's folders: _PipelineCache, the saves, the backups.)
			const std::string first = Lower(U8(*rel.begin()));
			for (const char* d: kUserDirs)
				if (first == Lower(d)) it.disable_recursion_pending();
			continue;
		}
		const std::string leaf = Lower(U8(rel.filename()));
		const auto        dot  = leaf.rfind(".old");
		if (dot == std::string::npos || dot == 0) continue;
		const std::string tail = leaf.substr(dot + 4);
		if (!std::all_of(tail.begin(), tail.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
		olds.push_back(it->path());
	}
	for (const auto& p: olds) {
		std::error_code rm;
		if (fs::remove(p, rm) && !rm) removed++;
	}
	return removed;
}

#if defined(_WIN32)

namespace {

std::wstring W(const std::string& s) {
	if (s.empty()) return {};
	const int    n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
	std::wstring w(static_cast<size_t>(n), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
	return w;
}

std::string N(const std::wstring& w) {
	if (w.empty()) return {};
	const int   n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
	std::string s(static_cast<size_t>(n), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
	return s;
}

struct Handle {
	HINTERNET h = nullptr;
	explicit Handle(HINTERNET v): h(v) {}
	~Handle() {
		if (h != nullptr) WinHttpCloseHandle(h);
	}
	Handle(const Handle&)            = delete;
	Handle& operator=(const Handle&) = delete;
};

// GET over HTTPS (redirects followed): the body to `sink`. false: `error`.
bool HttpGet(const std::string& url, const std::function<bool(const char*, DWORD, int64_t)>& sink, std::string* error) {
	const std::wstring wurl = W(url);
	URL_COMPONENTS     parts {};
	parts.dwStructSize      = sizeof(parts);
	wchar_t host[256]       = {};
	wchar_t path[4096]      = {};
	parts.lpszHostName      = host;
	parts.dwHostNameLength  = 256;
	parts.lpszUrlPath       = path;
	parts.dwUrlPathLength   = 4096;
	wchar_t extra[4096]     = {};
	parts.lpszExtraInfo     = extra;
	parts.dwExtraInfoLength = 4096;
	if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &parts) || parts.nScheme != INTERNET_SCHEME_HTTPS) {
		*error = "direccion no valida";
		return false;
	}
	Handle session(WinHttpOpen(L"DemonsSoulsPC-Update", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
	if (session.h == nullptr) session.h = WinHttpOpen(L"DemonsSoulsPC-Update", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	if (session.h == nullptr) {
		*error = "sin conexion (WinHTTP)";
		return false;
	}
	WinHttpSetTimeouts(session.h, 15000, 15000, 30000, 60000);
	Handle connect(WinHttpConnect(session.h, host, parts.nPort, 0));
	const std::wstring object = std::wstring(path) + extra;
	Handle request(connect.h != nullptr ? WinHttpOpenRequest(connect.h, L"GET", object.c_str(), nullptr, WINHTTP_NO_REFERER,
	                                                         WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE)
	                                    : nullptr);
	if (request.h == nullptr) {
		*error = "no se pudo conectar con " + N(host);
		return false;
	}
	const wchar_t* headers = L"Accept: application/vnd.github+json, application/octet-stream\r\n";
	if (!WinHttpSendRequest(request.h, headers, static_cast<DWORD>(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
	    !WinHttpReceiveResponse(request.h, nullptr)) {
		*error = "no se pudo conectar con " + N(host) + " (error " + std::to_string(GetLastError()) + ")";
		return false;
	}
	DWORD status = 0, size = sizeof(status);
	WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
	                    WINHTTP_NO_HEADER_INDEX);
	if (status != 200) {
		*error = "respuesta " + std::to_string(status) + " de " + N(host);
		return false;
	}
	wchar_t length_text[32] = {};
	DWORD   length_size     = sizeof(length_text);
	int64_t total           = -1;
	if (WinHttpQueryHeaders(request.h, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, length_text, &length_size,
	                        WINHTTP_NO_HEADER_INDEX))
		total = _wtoi64(length_text);
	std::vector<char> buffer(1 << 16);
	for (;;) {
		DWORD read = 0;
		if (!WinHttpReadData(request.h, buffer.data(), static_cast<DWORD>(buffer.size()), &read)) {
			*error = "la descarga se corto (error " + std::to_string(GetLastError()) + ")";
			return false;
		}
		if (read == 0) break;
		if (!sink(buffer.data(), read, total)) {
			*error = "cancelado";
			return false;
		}
	}
	return true;
}

fs::path ResultFile() {
	wchar_t temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	return fs::path(temp) / L"ds-update-result.txt";
}

void WriteResult(const Result& r) {
	std::ofstream out(ResultFile(), std::ios::binary | std::ios::trunc);
	out << (r.ok ? "ok" : "error") << "\n" << (r.ok ? r.version : r.error) << "\n" << r.backup << "\n";
}

Result ReadResult() {
	Result            r;
	std::stringstream in(ReadFile(ResultFile()));
	std::string       state;
	std::getline(in, state);
	std::string text, backup;
	std::getline(in, text);
	std::getline(in, backup);
	r.ok = Trim(state) == "ok";
	if (r.ok) r.version = Trim(text);
	else r.error = text.empty() ? "El actualizador no termino." : Trim(text);
	r.backup = Trim(backup);
	return r;
}

} // namespace

bool CheckLatest(Release* release, std::string* error) {
	std::string body;
	if (!HttpGet("https://api.github.com/repos/todojuegos67-bot/KytyPS5/releases?per_page=50",
	             [&](const char* data, DWORD n, int64_t) {
		             body.append(data, n);
		             return body.size() < (8u << 20);
	             },
	             error)) {
		*error = "No se pudo consultar GitHub: " + *error;
		return false;
	}
	try {
		// (10-28c) Only the port's own releases (DemonsSoulsPC-*): releases/latest is the emulator's on purpose. The list
		// comes newest first; the first published (not draft / prerelease) one with a DemonsSoulsPC-*.zip wins.
		const auto list = nlohmann::json::parse(body);
		if (!list.is_array()) {
			*error = "Respuesta de GitHub no valida.";
			return false;
		}
		for (const auto& json: list) {
		if (json.value("draft", false) || json.value("prerelease", false)) continue;
		const std::string tag = json.value("tag_name", "");
		if (tag.rfind("DemonsSoulsPC-", 0) != 0) continue;
		release->tag    = tag;
		release->name   = json.value("name", "");
		release->published = json.value("published_at", "");
		release->url.clear();
		int64_t best    = -1;
		for (const auto& a: json.value("assets", nlohmann::json::array())) {
			const std::string name = a.value("name", "");
			const std::string url  = a.value("browser_download_url", "");
			if (name.rfind("DemonsSoulsPC", 0) != 0 || !EndsWith(Lower(name), ".zip") || url.rfind("https://", 0) != 0) continue;
			const int64_t size = a.value("size", static_cast<int64_t>(0));
			if (size > best) {
				best           = size;
				release->asset = name;
				release->url   = url;
				release->size  = size;
			}
		}
		if (!release->url.empty()) return true;
		}
		*error = "No hay ninguna version del port (DemonsSoulsPC-*) publicada en GitHub.";
		return false;
	} catch (...) {
		*error = "Respuesta de GitHub no valida.";
		return false;
	}
	return true;
}

bool Download(const Release& release, const fs::path& file, const std::function<void(int64_t, int64_t)>& progress, const bool* cancel,
              std::string* error) {
	std::ofstream out(file, std::ios::binary | std::ios::trunc);
	if (!out) {
		*error = "No se pudo crear " + U8(file);
		return false;
	}
	int64_t    done = 0;
	const bool ok   = HttpGet(release.url,
	                          [&](const char* data, DWORD n, int64_t total) {
                              out.write(data, n);
                              done += n;
                              if (progress) progress(done, total > 0 ? total : release.size);
                              return out.good() && (cancel == nullptr || !*cancel);
	                          },
	                          error);
	out.close();
	if (ok && release.size > 0 && done != release.size) {
		*error = "la descarga esta incompleta";
		std::error_code ec;
		fs::remove(file, ec);
		return false;
	}
	if (!ok) {
		*error = "No se pudo descargar " + release.asset + ": " + *error;
		std::error_code ec;
		fs::remove(file, ec);
	}
	return ok;
}

bool Extract(const fs::path& zip, const fs::path& dir, std::string* error) {
	std::error_code ec;
	fs::remove_all(dir, ec);
	fs::create_directories(dir, ec);
	wchar_t system[MAX_PATH + 1] = {};
	GetSystemDirectoryW(system, MAX_PATH);
	const fs::path tar = fs::path(system) / L"tar.exe";
	if (!fs::exists(tar, ec)) {
		*error = "Windows no tiene tar.exe (hace falta Windows 10 1803 o mas nuevo).";
		return false;
	}
	std::wstring        command = L"\"" + tar.wstring() + L"\" -xf \"" + zip.wstring() + L"\" -C \"" + dir.wstring() + L"\"";
	STARTUPINFOW        si {};
	PROCESS_INFORMATION pi {};
	si.cb = sizeof(si);
	if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, dir.c_str(), &si, &pi)) {
		*error = "No se pudo descomprimir (tar.exe no arranca).";
		return false;
	}
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);
	if (code != 0) {
		*error = "No se pudo descomprimir el zip (codigo " + std::to_string(code) + ").";
		return false;
	}
	return true;
}

bool Writable(const fs::path& root) {
	const fs::path test = root / (L".ds-write-test-" + std::to_wstring(GetCurrentProcessId()));
	HANDLE         h    = CreateFileW(test.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_FLAG_DELETE_ON_CLOSE, nullptr);
	if (h == INVALID_HANDLE_VALUE) return false;
	CloseHandle(h);
	return true;
}

Result InstallZip(const fs::path& root, const fs::path& zip, const std::string& tag, const fs::path& game_rel) {
	Result         r;
	wchar_t        temp[MAX_PATH + 1] = {};
	GetTempPathW(MAX_PATH, temp);
	const fs::path dir = fs::path(temp) / (L"ds-update-" + std::to_wstring(GetTickCount64()));
	std::string    error;
	if (!Extract(zip, dir, &error)) {
		r.error = error + " No se ha cambiado nada.";
	} else if (const fs::path src = FindReleaseRoot(dir); src.empty()) {
		r.error = std::string("El zip descargado no trae ") + kExeName + ". No se ha cambiado nada.";
	} else {
		Options options;
		options.game_rel = game_rel;
		r                = Install(root, src, CurrentVersion(), tag, options);
	}
	std::error_code ec;
	fs::remove_all(dir, ec);
	return r;
}

Result RunElevated(const fs::path& exe, const std::vector<std::wstring>& args) {
	Result       r;
	std::wstring params;
	for (const auto& a: args) {
		std::wstring q = a;
		q.erase(std::remove(q.begin(), q.end(), L'"'), q.end());
		if (!q.empty() && q.back() == L'\\') q += L'\\'; // (a trailing backslash would escape the quote)
		params += (params.empty() ? L"" : L" ") + (L"\"" + q + L"\"");
	}
	std::error_code ec;
	fs::remove(ResultFile(), ec);
	SHELLEXECUTEINFOW info {};
	info.cbSize       = sizeof(info);
	info.fMask        = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC;
	info.lpVerb       = L"runas";
	const std::wstring file = exe.wstring();
	info.lpFile       = file.c_str();
	info.lpParameters = params.c_str();
	info.nShow        = SW_HIDE;
	if (!ShellExecuteExW(&info) || info.hProcess == nullptr) {
		r.error = "Sin permiso de administrador no se puede escribir en esta carpeta. No se ha cambiado nada.";
		return r;
	}
	WaitForSingleObject(info.hProcess, INFINITE);
	CloseHandle(info.hProcess);
	return ReadResult();
}

int HelperMain(int /*argc*/, wchar_t** /*argv*/) {
	int     n    = 0;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &n);
	if (argv == nullptr || n < 2) return -1;
	const std::wstring mode = argv[1];
	int                code = -1;
	if (mode == L"--update-helper" && n >= 6) {
		// root zip tag game_rel
		const Result r = InstallZip(argv[2], argv[3], N(argv[4]), argv[5]);
		WriteResult(r);
		code = r.ok ? 0 : 1;
	} else if (mode == L"--restore-helper" && n >= 6) {
		// root name current_version game_rel
		Options options;
		options.game_rel = argv[5];
		const Result r   = Restore(argv[2], N(argv[3]), N(argv[4]), options);
		WriteResult(r);
		code = r.ok ? 0 : 1;
	}
	LocalFree(argv);
	return code;
}

#endif

} // namespace Port::Update
