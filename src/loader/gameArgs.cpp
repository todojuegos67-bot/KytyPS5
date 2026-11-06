#include "loader/gameArgs.h"

#include "common/emulatorConfig.h"
#include "common/file.h"
#include "graphics/host_gpu/renderer/demonsSouls.h"
#include "loader/demonsSoulsSuperResolution.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Loader::GameArgs {
namespace {

// One "+name=value" line per console variable: Demon's Souls' defaults, then those of KYTY_GAME_CONVARS (';' or ','
// between entries), whose names replace the defaults'. The default: no XPR index-buffer culling, a GPU pass that culls
// every object's triangles (~2800 compute dispatches a frame at Boletaria 1-1) before drawing them; the PC GPU draws
// them all instead, the image unchanged (Boletaria 1-1 standing 34.30 -> 36.48 fps; doIndexBufferCulling=true: back).
// sound_enableAmbisonics=false: the game's own mix for speakers. Left alone it gives a console's system a fifth-order ambisonics
// scene to render, of which the emulator plays the first order (libAudio2.cpp): music and ambience close to mono (left and right
// correlate 0.74 in the Nexus and 0.93 at Latria, here 0.27 and 0.42; issue #294) and speech 7 dB lower against the ambience.
// The third-order voice files play here too, and the frames cost the same (=true: back).
// Above a 60 Hz vblank also gameMinDeltaTime=0.004 (as run_ds.sh gives it): the game steps by at least a frame of its
// target rate each frame, so every frame past 60 fps played it that much faster (whatever started the emulator).
// KYTY_SKIP_LOGOS=1: the game's own cp11_debug_skipSplashScreens=true, no start-up logos (Sony Interactive Entertainment
// presents, PlayStation Studios, Bluepoint: ~16 s that no button skips; issue #263). The opening cinematic still plays.
// A test switch only: the launchers no longer offer it, as a player with it on crashed twice after ~50 minutes and
// played over two hours without it (issue #266; in the game's code it only keeps the logo movie from starting).
bool SkipLogos() {
	const char* value = std::getenv("KYTY_SKIP_LOGOS");
	return value != nullptr && value[0] == '1';
}

const std::string& ExtraLines() {
	static const std::string lines = [] {
		std::vector<std::pair<std::string, std::string>> entries; // (name, "name=value")
		const auto add = [&](const std::string& list) {
			for (size_t at = 0; at < list.size();) {
				auto end = list.find_first_of(";,", at);
				if (end == std::string::npos) end = list.size();
				const auto first = list.find_first_not_of(' ', at);
				const auto last  = list.find_last_not_of(' ', end - 1);
				if (first != std::string::npos && first < end && last >= first) {
					auto       entry = list.substr(first, last - first + 1);
					const auto name  = entry.substr(0, entry.find('='));
					std::erase_if(entries, [&](const auto& held) { return held.first == name; });
					entries.emplace_back(name, std::move(entry));
				}
				at = end + 1;
			}
		};
		const char* value = std::getenv("KYTY_GAME_CONVARS"); // (0: none, not even the defaults)
		if (value != nullptr && std::string_view(value) == "0") return std::string();
		if (Libs::Graphics::DemonsSouls::IsSupportedGame()) {
			add("doIndexBufferCulling=false");
			add("sound_enableAmbisonics=false");
			if (Config::GetVblankFrequency() > 60) add("gameMinDeltaTime=0.004");
			if (SkipLogos()) add("cp11_debug_skipSplashScreens=true");
		}
		if (value != nullptr) add(value);
		std::string out;
		for (const auto& [name, entry]: entries) out += "+" + entry + "\n";
		return out;
	}();
	return lines;
}

enum class Kind { None, Args, PerfConfig };

// The engine opens them as "$/packagecmdlineargs.txt" and "$/misc/renderconfig_ps5_perf.txt" (lower case): the
// names are compared without case.
Kind KindOf(const std::filesystem::path& real) {
	auto name = real.filename().string();
	for (auto& c: name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	if (name == "packagecmdlineargs.txt" && !ExtraLines().empty()) return Kind::Args;
	if (name == "renderconfig_ps5_perf.txt" && DemonsSoulsSuperResolution::Active()) return Kind::PerfConfig;
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
	                  (kind == Kind::Args ? "kyty-PackageCmdLineArgs.txt" : "kyty-RenderConfig_PS5_Perf.txt");
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

} // namespace Loader::GameArgs
