#include "game.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unistd.h>

namespace {

std::string ReadText(const fs::path& file) {
	std::ifstream     in(file, std::ios::binary);
	std::stringstream text;
	text << in.rdbuf();
	std::string content = text.str();
	if (content.size() >= 3 && content.compare(0, 3, "\xEF\xBB\xBF") == 0) content.erase(0, 3);
	return content;
}

bool Exists(const fs::path& path) {
	std::error_code error;
	return fs::exists(path, error);
}

std::string Text(const nlohmann::ordered_json& map, const char* key) {
	if (!map.is_object()) return "";
	const auto it = map.find(key);
	if (it == map.end() || it->is_null()) return "";
	return it->is_string() ? it->get<std::string>() : it->dump();
}

bool EndsWithZar(const std::string& path) {
	if (path.size() < 4) return false;
	std::string tail = path.substr(path.size() - 4);
	for (auto& c: tail)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return tail == ".zar";
}

} // namespace

bool GameInfo::DemonsSouls() const {
	return title_id == "PPSA01340" || title_id == "PPSA01341" || title_id == "PPSA01342";
}

fs::path GameInfo::CheatFile(const fs::path& root) const {
	if (!Found()) return {};
	const auto file = root / "_Patches" / (title_id + ".json");
	return Exists(file) ? file : fs::path();
}

GameInfo GameInfo::Read(const std::string& path, const fs::path& tool, const fs::path& root) {
	GameInfo info;
	info.path = path;
	if (path.empty()) return info;
	nlohmann::ordered_json param;
	std::error_code        error;
	const fs::path         folder(path);
	if (fs::is_directory(folder, error) && Exists(folder / "sce_sys" / "param.json")) {
		param = nlohmann::ordered_json::parse(ReadText(folder / "sce_sys" / "param.json"), nullptr,
		                                      false);
		for (const char* name: {"pic0.png", "pic1.png"}) {
			if (Exists(folder / "sce_sys" / name)) {
				info.picture = folder / "sce_sys" / name;
				break;
			}
		}
	} else if (EndsWithZar(path) && fs::is_regular_file(folder, error) && !tool.empty()) {
		// The precompile program copies param.json out of the archive.
		char      temporary[] = "/tmp/kyty-param-XXXXXX";
		const int fd          = mkstemp(temporary);
		if (fd >= 0) {
			close(fd);
			if (Platform::Run({tool.string(), "--game", path, "--param", temporary}, nullptr, 15000,
			                  root) == 0)
				param = nlohmann::ordered_json::parse(ReadText(temporary), nullptr, false);
			unlink(temporary);
		}
	}
	if (!param.is_object()) return info;
	info.title_id = Text(param, "titleId");
	info.version  = Text(param, "contentVersion");
	if (const auto it = param.find("localizedParameters"); it != param.end() && it->is_object()) {
		const auto language = Text(*it, "defaultLanguage");
		if (!language.empty()) {
			if (const auto names = it->find(language); names != it->end())
				info.name = Text(*names, "titleName");
		}
	}
	return info;
}

StoreState CheckShaderStore(const fs::path& root, const fs::path& precompile_tool,
                            const GameInfo& game) {
	if (!game.Found()) return StoreState::Unknown;
	const auto store = root / "_PipelineCache" / "static";
	bool       made  = false;
	for (const char* ext: {".binaries", ".bin", ".blobs"})
		made = made || Exists(store / (game.GameId() + ext));
	if (!made) return StoreState::None;
	const auto seed_name = "seeds-" + game.GameId() + ".seeds";
	fs::path   seeds;
	for (const auto& candidate:
	     {root / seed_name, root / "_Build" / "static-precompile" / seed_name}) {
		if (Exists(candidate)) {
			seeds = candidate;
			break;
		}
	}
	if (seeds.empty() || precompile_tool.empty()) return StoreState::Current;
	std::string output;
	const int   code = Platform::Run(
        {precompile_tool.string(), "--game", game.path, "--seeds", seeds.string(), "--status"},
        &output, 60000, root);
	if (code != 0) return StoreState::Unknown;
	// (OutdatedAll: the store lacks a translation that changes nearly every shader, so the precompile redoes it all,
	// issue #246.)
	if (output.find("static cache outdated") != std::string::npos)
		return output.find("next precompile all") != std::string::npos ? StoreState::OutdatedAll : StoreState::Outdated;
	if (output.find("static cache current") == std::string::npos) return StoreState::None;
	// (A precompile that stopped before its last merge leaves large shard checkpoints. One older
	// than its store is an earlier run's - another graphics driver's, which no merge takes - not an
	// unfinished one.)
	std::error_code error;
	for (const auto& entry: fs::directory_iterator(store, error)) {
		const auto name = entry.path().filename().string();
		const auto at   = name.find(".shard");
		if (at == std::string::npos || !entry.is_regular_file(error) ||
		    entry.file_size(error) <= (1u << 20u))
			continue;
		const auto own = store / name.substr(0, at);
		if (!(Exists(own) && fs::last_write_time(own, error) > entry.last_write_time(error)))
			return StoreState::Outdated;
	}
	return StoreState::Current;
}

CheatFile CheatFile::Load(const fs::path& file) {
	CheatFile result;
	result.before = ReadText(file);
	auto data     = nlohmann::ordered_json::parse(result.before, nullptr, false);
	if (!data.is_object()) {
		result.error = data.is_discarded() ? "not JSON" : "not a JSON object";
		return result;
	}
	result.title = Text(data, "name");
	if (result.title.empty()) result.title = Text(data, "id");
	if (const auto it = data.find("mods"); it != data.end() && it->is_array()) {
		for (const auto& mod: *it) {
			CheatMod entry;
			entry.name        = Text(mod, "name");
			entry.description = Text(mod, "description");
			if (mod.is_object()) {
				const auto enabled = mod.find("enabled");
				entry.enabled =
				    enabled == mod.end() || !enabled->is_boolean() || enabled->get<bool>();
			}
			result.mods.push_back(entry);
		}
	}
	return result;
}

std::string CheatFile::Save(const fs::path& file) const {
	// Not over a file that changed while the window was open.
	if (ReadText(file) != before) return "cheats.changed";
	auto data = nlohmann::ordered_json::parse(before, nullptr, false);
	if (!data.is_object() || !data.contains("mods") || !data["mods"].is_array() ||
	    data["mods"].size() != mods.size())
		return "cheats.countChanged";
	for (size_t i = 0; i < mods.size(); ++i)
		if (data["mods"][i].is_object()) data["mods"][i]["enabled"] = mods[i].enabled;
	std::error_code error;
	fs::copy_file(file, fs::path(file.string() + ".bak"), fs::copy_options::overwrite_existing,
	              error);
	if (error) return "cheats.verifyFailed";
	{
		std::ofstream out(file, std::ios::binary | std::ios::trunc);
		out << data.dump(2) << "\n";
	}
	// Read back: the same names and flags, or the backup goes back.
	const auto check = Load(file);
	bool       good  = check.error.empty() && check.mods.size() == mods.size();
	for (size_t i = 0; good && i < mods.size(); ++i)
		good = check.mods[i].name == mods[i].name && check.mods[i].enabled == mods[i].enabled;
	if (!good) {
		fs::copy_file(fs::path(file.string() + ".bak"), file, fs::copy_options::overwrite_existing,
		              error);
		return "cheats.verifyFailed";
	}
	return "";
}
