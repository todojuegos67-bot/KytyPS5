#include "settings.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

namespace {

// The Windows launcher's defaults, in its order (the file's order); then this launcher's own.
const nlohmann::ordered_json& Defaults() {
	static const nlohmann::ordered_json defaults = [] {
		nlohmann::ordered_json d = nlohmann::ordered_json::object();
		d["game"]                = "";
		d["resolution"]          = "2560x1440";
		d["fullscreen"]          = false;
		d["aspect"]              = true;
		d["language"]            = 1;
		d["redzone"]             = true;
		d["cheats"]              = false;
		d["ecores"]              = false;
		d["fps120"]              = false;
		d["render"]              = 1440;
		d["ratio"]               = "16:9";
		d["quality"]             = "standard";
		d["vsync"]               = false;
		d["hdr"]                 = false;
		d["framegen"]            = 0;
		d["hud"]                 = 0;
		d["streamline"]          = "";
		d["safegpu"]             = false;
		d["upscaler"]            = "";
		d["render_dlss"]         = 1080;
		d["fsr"]                 = false;
		d["output"]              = "auto";
		d["lowlatency"]          = false;
		d["ui_language"]         = "";
		d["present_direct"]      = false;
		d["vrr_flips"]           = false;
		d["antilag"]             = false;
		d["keep_resident"]       = false;
		d["fps30"]               = false;
		d["extra_args"]          = "";
		d["gpu"]                 = "";
		d["dlss_preset"]         = "";
		d["dlss_sharpness"]      = ""; // (Graphics page: KYTY_DLSS_SHARPNESS's value, "" none)
		d["fsr_sharpness"]       = ""; // (Graphics page: KYTY_FSR_SHARPNESS's value, "" its 0.6)
		d["audio_output"]        = ""; // (Game page: "" a TV)
		d["output_auto"]         = true;
		d["wide_bars"]           = false; // (Display page: 21:9 or 32:9, movies and menus at 16:9)
		// (Linux.)
		d["mangohud"]         = true; // run_ds.sh's MANGO=1 default
		d["terminal"]         = "";
		d["precompile_quick"] = false;
		d["ui_scale"]         = 0; // percent; 0: automatic
		return d;
	}();
	return defaults;
}

} // namespace

Settings::Settings(std::filesystem::path path): path_(std::move(path)) {
	values_ = Defaults();
	std::ifstream file(path_, std::ios::binary);
	if (file) {
		std::stringstream text;
		text << file.rdbuf();
		std::string content = text.str();
		// (The Windows launcher writes a byte order mark.)
		if (content.size() >= 3 && content.compare(0, 3, "\xEF\xBB\xBF") == 0) content.erase(0, 3);
		auto parsed = nlohmann::ordered_json::parse(content, nullptr, false);
		if (parsed.is_object()) {
			// 4K was the game output's default, written by every save: a file from before Automatic
			// has it as Automatic.
			const bool before_auto = !parsed.contains("output_auto");
			for (auto& [key, value]: parsed.items())
				values_[key] = value;
			if (before_auto && Str("output") == "4k") values_["output"] = "auto";
		}
	}
	// (The Windows launcher's predecessor kept the game folder in game-path.txt.)
	if (Str("game").empty()) {
		std::ifstream legacy(path_.parent_path() / "game-path.txt");
		std::string   line;
		if (legacy && std::getline(legacy, line)) {
			while (!line.empty() &&
			       (line.back() == '\r' || line.back() == ' ' || line.back() == '\n'))
				line.pop_back();
			values_["game"] = line;
		}
	}
	saved_ = values_.dump(4);
}

std::string Settings::Str(const std::string& key) const {
	const auto it = values_.find(key);
	if (it == values_.end() || it->is_null()) return "";
	if (it->is_string()) return it->get<std::string>();
	if (it->is_boolean()) return it->get<bool>() ? "true" : "false";
	if (it->is_number_integer()) return std::to_string(it->get<int64_t>());
	return it->dump();
}

bool Settings::Bool(const std::string& key) const {
	const auto it = values_.find(key);
	if (it == values_.end() || it->is_null()) return false;
	if (it->is_boolean()) return it->get<bool>();
	if (it->is_string()) {
		std::string text = it->get<std::string>();
		for (auto& c: text)
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return text == "true";
	}
	if (it->is_number()) return it->get<double>() != 0;
	return false;
}

// (An older launcher saved true for the overlay's frame rate: true is 1.)
int Settings::Int(const std::string& key) const {
	const auto it = values_.find(key);
	if (it == values_.end() || it->is_null()) return 0;
	if (it->is_boolean()) return it->get<bool>() ? 1 : 0;
	if (it->is_number()) return static_cast<int>(it->get<double>());
	if (it->is_string()) {
		const std::string text = it->get<std::string>();
		char*             end  = nullptr;
		const long        n    = std::strtol(text.c_str(), &end, 10);
		return end != text.c_str() && *end == 0 ? static_cast<int>(n) : 0;
	}
	return 0;
}

void Settings::Set(const std::string& key, const std::string& value) {
	values_[key] = value;
}

void Settings::Set(const std::string& key, bool value) {
	values_[key] = value;
}

void Settings::Set(const std::string& key, int value) {
	values_[key] = value;
}

void Settings::Save() {
	if (read_only) return;
	const std::string text = values_.dump(4);
	if (text == saved_) return;
	// A new file next to the old one, then renamed over it: a crash mid-write leaves the old file
	// whole.
	std::error_code error;
	std::filesystem::create_directories(path_.parent_path(), error);
	const auto temporary = std::filesystem::path(path_.string() + ".tmp");
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		if (!file) return; // a read-only folder: the choices last until the window closes
		file << text << "\n";
		if (!file) return;
	}
	std::filesystem::rename(temporary, path_, error);
	if (!error) saved_ = text;
}
