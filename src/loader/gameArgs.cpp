#include "loader/gameArgs.h"

#include "common/file.h"
#include "loader/demonsSoulsSuperResolution.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>

namespace Loader::GameArgs {
namespace {

// One "+name=value" line per entry of KYTY_GAME_CONVARS (';' or ',' between entries).
const std::string& ExtraLines() {
	static const std::string lines = [] {
		std::string out;
		const char* value = std::getenv("KYTY_GAME_CONVARS");
		const std::string list = value != nullptr ? value : "";
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
