#include "loader/gameArgs.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"

#include "common/file.h"
#include "graphics/host_gpu/hostMemory.h"
#include "loader/demonsSoulsSuperResolution.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <set>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>

namespace Loader::GameArgs {
namespace {

// One "+name=value" line per entry of KYTY_GAME_CONVARS (';' or ',' between entries), then of
// KYTY_GAME_CONVARS_EXTRA: the launcher sets KYTY_GAME_CONVARS itself, the extra list (a user environment variable) is
// read after it, so its values win (A/B tests of single engine settings, e.g. r_enableSpotShadows=false).
const std::string& ExtraLines() {
	static const std::string lines = [] {
		std::string out;
		const char* value = std::getenv("KYTY_GAME_CONVARS");
		const char* extra = std::getenv("KYTY_GAME_CONVARS_EXTRA");
		std::string list  = value != nullptr ? value : "";
		if (extra != nullptr && *extra != '\0') {
			list += std::string(";") + extra;
			std::printf("Game convars: KYTY_GAME_CONVARS_EXTRA=%s (after the launcher's)\n", extra);
			std::fflush(stdout);
		}
		for (size_t at = 0; at < list.size();) {
			auto end = list.find_first_of(";,", at);
			if (end == std::string::npos) end = list.size();
			const auto first = list.find_first_not_of(' ', at);
			const auto last  = list.find_last_not_of(' ', end - 1);
			if (first != std::string::npos && first < end && last >= first) out += "+" + list.substr(first, last - first + 1) + "\n";
			at = end + 1;
		}
		return out;
	}();
	return lines;
}

enum class Kind { None, Args, PerfConfig, HighConfig };

// KYTY_UPSCALE_RATIO without KYTY_RENDER_HEIGHT (DemonsSoulsRenderSize::CinematicUpscaled).
bool CinematicByRatio() {
	const char* ratio  = std::getenv("KYTY_UPSCALE_RATIO");
	const char* height = std::getenv("KYTY_RENDER_HEIGHT");
	return ratio != nullptr && *ratio != 0 && (height == nullptr || *height == 0);
}

// The engine opens them as "$/packagecmdlineargs.txt" and "$/misc/renderconfig_ps5_perf.txt" (lower case): the
// names are compared without case.
Kind KindOf(const std::filesystem::path& real) {
	// (Demon's Souls' files: another title's file of one of these names is its own.)
	static const bool ds = Libs::Graphics::DemonsSouls::IsSupportedGame();
	if (!ds) return Kind::None;
	auto name = real.filename().string();
	for (auto& c: name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (name == "packagecmdlineargs.txt" && !ExtraLines().empty()) return Kind::Args;
	if (name == "renderconfig_ps5_perf.txt" && DemonsSoulsSuperResolution::Active()) return Kind::PerfConfig;
	// The Cinematic mode (RenderConfig_PS5_High.txt: display mode 4, 3840x2160 natively, no upscaler) with the launcher's
	// upscaler preset: display mode 5 too, so it renders at the ratio's size (the performance mode's mode 5 size,
	// DemonsSoulsRenderSize) and goes through the upscaler; the rest of its config (its quality) kept.
	if (name == "renderconfig_ps5_high.txt" && DemonsSoulsSuperResolution::Active() && CinematicByRatio())
		return Kind::HighConfig;
	return Kind::None;
}

std::string Contents(const std::filesystem::path& real, Kind kind) {
	std::string  text;
	Common::File in;
	if (in.Open(real, Common::File::Mode::Read)) { // (also a game archive's member)
		text.resize(static_cast<size_t>(in.Size()));
		uint32_t read = 0;
		in.Read(text.data(), static_cast<uint32_t>(text.size()), &read);
		text.resize(read);
	}
	if (!text.empty() && text.back() != '\n') text += '\n';
	if (kind == Kind::Args) return text + ExtraLines();
	// The performance mode's display mode: 5 (the PS5 Pro's: 2688x1512 and super resolution), the rest of it kept.
	std::string out;
	bool        replaced = false;
	for (size_t at = 0; at < text.size();) {
		const auto end  = text.find('\n', at);
		const auto line = text.substr(at, end - at + 1);
		if (line.find("r_Wants4K") != std::string::npos) {
			out += "+r_Wants4K = WantsTrinity\n";
			replaced = true;
		} else {
			out += line;
		}
		at = end + 1;
	}
	return replaced ? out : out + "+r_Wants4K = WantsTrinity\n";
}

} // namespace

std::filesystem::path RedirectRead(const std::filesystem::path& real) {
	const auto kind = KindOf(real);
	if (kind == Kind::None) return {};
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	// The copy goes next to the temporary files; one that cannot be written leaves the file as it is.
	std::error_code error;
	const auto      copy = std::filesystem::temp_directory_path(error) /
	                  (kind == Kind::Args         ? "kyty-PackageCmdLineArgs.txt"
	                   : kind == Kind::HighConfig ? "kyty-RenderConfig_PS5_High.txt"
	                                              : "kyty-RenderConfig_PS5_Perf.txt");
	if (error) return {};
	const auto    text = Contents(real, kind);
	std::ofstream out(copy, std::ios::binary | std::ios::trunc);
	out.write(text.data(), static_cast<std::streamsize>(text.size()));
	out.close();
	if (!out) return {};
	if (kind == Kind::Args) {
		std::printf("Game args: %s read with\n%s", real.string().c_str(), ExtraLines().c_str());
	} else {
		std::printf("Game args: %s read with r_Wants4K = WantsTrinity\n", real.string().c_str());
	}
	std::fflush(stdout);
	return copy;
}

std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real) {
	const auto kind = KindOf(real);
	if (kind == Kind::None) return std::nullopt;
	return Contents(real, kind).size();
}

namespace {
bool IdentChar(unsigned char c) { return std::isalnum(c) != 0 || c == '_'; }

bool Interesting(const std::string& name, int mode) {
	if (name.size() < 6 || name.size() > 80 || std::isalpha(static_cast<unsigned char>(name[0])) == 0) return false;
	std::string lower = name;
	for (auto& c: lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (lower.rfind("r_", 0) == 0) return true;
	static const char* const keys[] = {"shadow", "texture", "stream", "lod", "fog", "volumetric", "ssr", "ambientocc",
	                                   "decal", "particle", "fur", "hair", "cloth", "probe", "reflection", "resolution",
	                                   "budget", "pool", "mip", "cascade", "pixelsize", "drawdistance", "viewdistance",
	                                   "cvar", "convar"};
	bool key = false;
	for (const auto* k: keys) key = key || lower.find(k) != std::string::npos;
	bool has_upper = false;
	for (char c: name) has_upper = has_upper || std::isupper(static_cast<unsigned char>(c)) != 0;
	// An identifier, not a sentence or a path: camelCase or snake_case, no spaces (guaranteed by the scan).
	const bool shaped = has_upper || name.find('_') != std::string::npos;
	if (key && shaped) return true;
	if (mode >= 2) {
		if (lower.rfind("g_", 0) == 0 || lower.rfind("s_", 0) == 0) return true;
		return name.size() >= 8 && has_upper && std::islower(static_cast<unsigned char>(name[0])) != 0;
	}
	return false;
}
} // namespace

void DumpConvarNames(const Program* program) {
	const char* value = std::getenv("KYTY_GAME_CONVARS_DUMP");
	const int   mode  = value != nullptr ? std::atoi(value) : 0;
	if (mode <= 0 || program == nullptr || program->file_name.filename() != "eboot.bin") return;
	std::set<std::string> names;
	constexpr uint64_t    Page = 0x4000;
	const uint64_t        base = program->base_vaddr, end = base + program->mapped_size;
	std::string           run;
	for (uint64_t page = base; page < end; page += Page) {
		const uint64_t size = std::min(Page, end - page);
		if (!Libs::Graphics::HostMemoryRangeIsReadable(page, size)) {
			run.clear();
			continue;
		}
		const auto* bytes = reinterpret_cast<const unsigned char*>(page);
		for (uint64_t i = 0; i < size; i++) {
			if (IdentChar(bytes[i])) {
				run += static_cast<char>(bytes[i]);
				if (run.size() > 96) run.clear();
				continue;
			}
			// A whole NUL-terminated identifier string (the name as the game registers it).
			if (bytes[i] == 0 && Interesting(run, mode)) names.insert(run);
			run.clear();
		}
	}
	std::string text;
	for (const auto& name: names) text += name + "\n";
	std::error_code error;
	const auto      path = std::filesystem::temp_directory_path(error) / "kyty-convar-names.txt";
	if (!error) {
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		out.write(text.data(), static_cast<std::streamsize>(text.size()));
	}
	std::printf("Game convars: %zu candidate names in the eboot (KYTY_GAME_CONVARS_DUMP=%d; list also in %s)\n%s"
	            "Game convars: overrides in effect (KYTY_GAME_CONVARS):\n%s",
	            names.size(), mode, error ? "(no temp folder)" : path.string().c_str(), text.c_str(),
	            ExtraLines().empty() ? "(none)\n" : ExtraLines().c_str());
	std::fflush(stdout);
}

} // namespace Loader::GameArgs
