#include "loader/gameArgs.h"

#include "common/file.h"

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

// The engine opens it as "$/packagecmdlineargs.txt" (lower case): the name is compared without case.
bool IsArgsFile(const std::filesystem::path& real) {
	if (ExtraLines().empty()) return false;
	auto name = real.filename().string();
	for (auto& c: name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return name == "packagecmdlineargs.txt";
}

std::string Contents(const std::filesystem::path& real) {
	std::string  text;
	Common::File in;
	if (in.Open(real, Common::File::Mode::Read)) { // (also a game archive's member)
		text.resize(static_cast<size_t>(in.Size()));
		uint32_t read = 0;
		in.Read(text.data(), static_cast<uint32_t>(text.size()), &read);
		text.resize(read);
	}
	if (!text.empty() && text.back() != '\n') text += '\n';
	return text + ExtraLines();
}

} // namespace

std::filesystem::path RedirectRead(const std::filesystem::path& real) {
	if (!IsArgsFile(real)) return {};
	static std::mutex mutex;
	std::lock_guard   lock(mutex);
	// The copy goes next to the temporary files; one that cannot be written leaves the file as it is.
	std::error_code error;
	const auto      copy = std::filesystem::temp_directory_path(error) / "kyty-PackageCmdLineArgs.txt";
	if (error) return {};
	const auto    text = Contents(real);
	std::ofstream out(copy, std::ios::binary | std::ios::trunc);
	out.write(text.data(), static_cast<std::streamsize>(text.size()));
	out.close();
	if (!out) return {};
	std::printf("Game args: %s read with\n%s", real.string().c_str(), ExtraLines().c_str());
	std::fflush(stdout);
	return copy;
}

std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real) {
	if (!IsArgsFile(real)) return std::nullopt;
	return Contents(real).size();
}

} // namespace Loader::GameArgs
