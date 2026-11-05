#include "graphics/shader/recompiler/backend/spirv/HelperInlinePolicy.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::HelperInlinePolicy {
namespace {

struct State {
	std::mutex                   mutex;
	std::unordered_set<uint64_t> blacklist;
	State() {
		if (std::FILE* f = std::fopen(kBlacklistPath, "r"); f != nullptr) {
			char line[128];
			while (std::fgets(line, sizeof(line), f) != nullptr) {
				uint64_t hash = 0;
				if (std::sscanf(line, "guest %" SCNx64, &hash) == 1 && hash != 0) blacklist.insert(hash);
			}
			std::fclose(f);
			if (!blacklist.empty()) {
				std::printf("HelperInline: %zu guest shader(s) blacklisted for DontInline (%s)\n",
				            blacklist.size(), kBlacklistPath);
			}
		}
	}
};

State& Get() {
	static State s;
	return s;
}

} // namespace

uint64_t ThresholdWords() {
	static const uint64_t words = [] {
		const char* v = std::getenv("KYTY_SPIRV_DONTINLINE_WORDS");
		return v != nullptr ? std::strtoull(v, nullptr, 10) : 200000ull;
	}();
	return words;
}

bool GuestBlacklisted(uint64_t guest_hash) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	return s.blacklist.contains(guest_hash);
}

namespace {
bool EnvOn(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] == '1';
}
} // namespace

bool SwitchGroup() {
	static const bool on = EnvOn("KYTY_SPIRV_HELPER_SWITCHGROUP");
	return on;
}
bool GeomGroup() {
	static const bool on = EnvOn("KYTY_SPIRV_HELPER_GEOMGROUP");
	return on;
}
bool BoundsHoist() {
	static const bool on = EnvOn("KYTY_SPIRV_HELPER_BOUNDSHOIST");
	return on;
}

uint64_t CacheKey(uint64_t guest_hash) {
	return (ThresholdWords() << 4u) | (SwitchGroup() ? 8u : 0u) | (GeomGroup() ? 4u : 0u) |
	       (BoundsHoist() ? 2u : 0u) | (GuestBlacklisted(guest_hash) ? 1u : 0u);
}

} // namespace Libs::Graphics::ShaderRecompiler::HelperInlinePolicy
