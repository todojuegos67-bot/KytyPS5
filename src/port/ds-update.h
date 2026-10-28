#ifndef EMULATOR_SRC_PORT_DS_UPDATE_H_
#define EMULATOR_SRC_PORT_DS_UPDATE_H_

// "Demon's Souls" PC port: the in-place update ("Actualizar version") and its undo ("Volver a la version anterior"),
// as Launcher-souls.ps1's updater (10-27) does them, for the port's folder:
//   - the new release is copied over this folder; what the user made is never touched: _PipelineCache, _SaveData,
//     _SaveData-copias, logs, the seeds (*.seeds, *.seeds.program), kbm_curve.txt, the game's folder if it is inside;
//     settings.ini keeps every value, only the keys the new version adds are merged in;
//   - the engine files it replaces (the exe, the DLLs, streamline\, tools\...) are copied first to
//     _backup_update\<old version>\ (the last two kept);
//   - a file in use (the running exe and its DLLs: Windows lets them be renamed, not overwritten) is renamed to
//     <name>.old before the new one is copied; RemoveOldFiles deletes those at the next start;
//   - on any failure everything is put back as it was.
// The core is plain std::filesystem (tests/PortUpdateTests.cpp runs it on Linux); the download, the unzip and the
// elevation are Windows only (ds-update.cpp, _WIN32).

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace Port::Update {

namespace fs = std::filesystem;

inline constexpr const char* kExeName   = "Demon's Souls.exe";
inline constexpr const char* kBackupDir = "_backup_update";
inline constexpr int         kKeepBackups = 2;

// The version compiled into the exe (-DKYTY_DS_PORT_VERSION, the CI's package name = its release tag).
const char* CurrentVersion();

// `rel`: a path relative to the port's folder. True: the user's (never copied over, never backed up).
// `game_rel`: the game's folder relative to the port's folder when it is inside it (empty: outside).
bool IsExcluded(const fs::path& rel, const fs::path& game_rel = {});

// The folder of an extracted release that holds "Demon's Souls.exe" (itself or its only subfolder, as the CI's zip
// "Demon's Souls PC\..."); empty when there is none.
fs::path FindReleaseRoot(const fs::path& extracted);

// settings.ini: `user` with the keys of `shipped` it lacks added (in their section, with the comment lines above them);
// every value of `user` kept.
std::string MergeIni(const std::string& user, const std::string& shipped);

struct Result {
	bool        ok = false;
	std::string error;      // Spanish, for the menu
	std::string backup;     // the backup folder made (its name)
	std::string version;    // the version now installed
};

struct Options {
	fs::path    game_rel;          // see IsExcluded
	// Tests: fail the copy of this file (relative path), to see the rollback.
	std::string fail_on;
};

// Copies the release in `src` over `root`. `old_version` names the backup, `new_version` is the result's.
Result Install(const fs::path& root, const fs::path& src, const std::string& old_version, const std::string& new_version,
               const Options& options = {});

struct Backup {
	std::string name;     // the folder under _backup_update
	std::string version;  // its .version (or the name)
	int64_t     time = 0; // when it was made (for the order)
};
// Newest first; only the ones that hold "Demon's Souls.exe".
std::vector<Backup> ListBackups(const fs::path& root);

// Puts back the backup `name` (the current files are backed up first, so it can be undone).
Result Restore(const fs::path& root, const std::string& name, const std::string& current_version, const Options& options = {});

// At start: deletes the *.old files a previous update left (the exe and DLLs that were running).
int RemoveOldFiles(const fs::path& root);

// Keeps `keep` backups: the `protect` ones and the newest others.
void PruneBackups(const fs::path& root, int keep, const std::vector<std::string>& protect = {});

#if defined(_WIN32)
// ---- Windows: GitHub, the download, the unzip, the elevation. ----
struct Release {
	std::string tag, name, asset, url;
	int64_t     size = 0;
};
// releases/latest of todojuegos67-bot/KytyPS5 and its port zip (DemonsSoulsPC-*.zip). false: `error` says why.
// (10-28) The 8-digit YYYYMMDD date in a tag (the first "20" + 6 digits), or "" without one.
std::string TagDate(const std::string& tag);
bool CheckLatest(Release* release, std::string* error);
// Downloads to `file`; `progress(done, total)`; `cancel` is polled.
bool Download(const Release& release, const fs::path& file, const std::function<void(int64_t, int64_t)>& progress,
              const bool* cancel, std::string* error);
// Unzips with Windows' own tar.exe into `dir` (emptied first).
bool Extract(const fs::path& zip, const fs::path& dir, std::string* error);
// True when `root` can be written without elevation.
bool Writable(const fs::path& root);
// The whole update from a downloaded zip: extract, install, clean the temporary files.
Result InstallZip(const fs::path& root, const fs::path& zip, const std::string& tag, const fs::path& game_rel);
// Runs this exe elevated ("runas") in its helper mode (--update-helper / --restore-helper) and waits for it.
Result RunElevated(const fs::path& exe, const std::vector<std::wstring>& args);
// The helper mode (ds-main.cpp's main, before anything else): 0 ok, 1 failed (the result file says why), -1 not a helper.
int HelperMain(int argc, wchar_t** argv);
#endif

} // namespace Port::Update

#endif // EMULATOR_SRC_PORT_DS_UPDATE_H_
