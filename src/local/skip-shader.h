#pragma once
// Live probe `skipcs <guest code address hex>`: dispatches of that compute program, or draws of that pixel shader, are not run (toggles; `skipcs 0`
// clears all). For finding which pass draws an artifact; the image keeps its previous contents. Up to 8 programs.
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace SkipShader {
inline std::atomic<uint64_t> g_cs[8] {};
inline std::atomic<bool>     g_any {false};

inline bool Skipped(uint64_t address) noexcept {
	if (!g_any.load(std::memory_order_relaxed)) return false;
	for (auto& slot : g_cs)
		if (slot.load(std::memory_order_relaxed) == address) return true;
	return false;
}

inline void Toggle(uint64_t address) noexcept {
	if (address == 0) {
		for (auto& slot : g_cs) slot.store(0);
		g_any.store(false);
		return;
	}
	for (auto& slot : g_cs)
		if (slot.load() == address) {
			slot.store(0);
			return;
		}
	for (auto& slot : g_cs)
		if (slot.load() == 0) {
			slot.store(address);
			g_any.store(true);
			return;
		}
}

// KYTY_SKIP_SHADERS (or KYTY_SKIP_SHADER_HASHES, or --skip-shaders, which sets it) = "hash,hash,..."
// (16-hex shader hashes as logged, optional 0x): dispatches and draws whose compute/vertex/pixel shader
// hash matches are not run, as the Wolverine fork's --skip-shaders. Each skipped hash is printed once.
inline const std::vector<uint64_t>& HashList() {
	static const std::vector<uint64_t> list = [] {
		std::vector<uint64_t> r;
		for (const char* name : {"KYTY_SKIP_SHADERS", "KYTY_SKIP_SHADER_HASHES"}) {
			const char* v = std::getenv(name);
			if (v == nullptr) continue;
			std::string s(v);
			size_t      start = 0;
			while (start <= s.size()) {
				size_t end = s.find(',', start);
				if (end == std::string::npos) end = s.size();
				std::string tok = s.substr(start, end - start);
				while (!tok.empty() && (tok.front() == ' ')) tok.erase(0, 1);
				while (!tok.empty() && (tok.back() == ' ')) tok.pop_back();
				if (!tok.empty()) {
					const uint64_t h = std::strtoull(tok.c_str(), nullptr, 16);
					if (h != 0) {
						r.push_back(h);
						std::printf("SkipShader: skipping the draws and dispatches of shader 0x%016" PRIx64 "\n", h);
					}
				}
				start = end + 1;
			}
		}
		return r;
	}();
	return list;
}

inline bool HashSkipped(uint64_t hash, const char* what) {
	const auto& list = HashList();
	if (list.empty() || hash == 0) return false;
	bool hit = false;
	for (auto h : list)
		if (h == hash) hit = true;
	if (!hit) return false;
	static std::mutex                   m;
	static std::unordered_set<uint64_t> warned;
	bool                                first;
	{
		std::lock_guard lock(m);
		first = warned.insert(hash).second;
	}
	if (first) std::printf("Warning: skipping %s of shader 0x%016" PRIx64 " (KYTY_SKIP_SHADERS)\n", what, hash);
	return true;
}
} // namespace SkipShader
