#ifndef KYTY_LAUNCHER_LINUX_SETTINGS_H
#define KYTY_LAUNCHER_LINUX_SETTINGS_H

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

// launcher-settings.json: the same file and keys as the Windows launcher
// (launcher-windows/Settings.cs), so the choices read the same; keys this launcher does not know
// are kept as they were, in their order. A few keys are Linux's own (mangohud, terminal,
// precompile_quick, ui_scale).
class Settings {
public:
	explicit Settings(std::filesystem::path path);

	[[nodiscard]] std::string Str(const std::string& key) const;
	[[nodiscard]] bool        Bool(const std::string& key) const;
	[[nodiscard]] int         Int(const std::string& key) const;

	void Set(const std::string& key, const std::string& value);
	void Set(const std::string& key, const char* value) { Set(key, std::string(value)); }
	void Set(const std::string& key, bool value);
	void Set(const std::string& key, int value);

	// Written when something changed since the last save (or the read); false when the file could
	// not be written (a read-only folder: the choices last until the window closes).
	bool Save();

	[[nodiscard]] const std::filesystem::path& Path() const { return path_; }

	bool read_only = false; // (--dry-run, --screenshot: the file stays as it was)

private:
	std::filesystem::path  path_;
	nlohmann::ordered_json values_;
	std::string            saved_;
};

#endif
