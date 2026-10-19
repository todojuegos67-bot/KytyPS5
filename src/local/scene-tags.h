#pragma once
// Pipelines by scene (KYTY_SCENE_WARMUP, default on where the game reports its scenes): which recorded pipeline recipes
// (LocalShaderWarmup::RecipeKey) each scene of the game bound - a place the game loads as a whole, Demon's Souls' maps
// (libAmpr: the map's .arealoadlist read) - kept beside the warmup recording (<recording>.scenes). The warmup then makes
// the pipelines of the scene the game loads instead of every recorded one: each live pipeline holds ~220 KB of the NVIDIA
// driver's RAM (1-1 after six teleports: 20.7K pipelines and 5.9 GB of driver pools, of them 4.7K bound there).
//
// A scene is known once it was played for KnownSeconds in a session (`visits`); an unknown one gets every recorded
// recipe, as before, while its own are noted (a pipeline at its first binding in each scene, BindPipelineHandle).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <xxhash.h>

namespace SceneTags {

inline constexpr uint32_t None         = 0;
inline constexpr int64_t  KnownSeconds = 60;

struct State {
	std::mutex                                                 mutex;
	std::unordered_map<uint32_t, std::unordered_set<uint64_t>> tags;
	std::unordered_map<uint32_t, uint32_t>                     visits;
	std::unordered_map<uint32_t, std::chrono::steady_clock::duration> played; // this session
	std::unordered_set<uint32_t>                               counted;        // visits counted this session
	std::chrono::steady_clock::time_point                      entered_at {};
	std::filesystem::path                                      path;
	std::string                                                identity;
	bool                                                       dirty = false;
	std::atomic<uint32_t>                                      current {None};
	std::atomic<uint64_t>                                      generation {0};
};
inline State& Get() {
	static State* const state = new State(); // (never destroyed: binding threads may outlive static destruction)
	return *state;
}

// A Demon's Souls map id as the game names it: m04_01_00_00 -> 0x04010000.
inline std::string Name(uint32_t scene) {
	char text[16];
	std::snprintf(text, sizeof(text), "m%02x_%02x_%02x_%02x", scene >> 24u, (scene >> 16u) & 0xffu, (scene >> 8u) & 0xffu,
	              scene & 0xffu);
	return text;
}

inline uint32_t Current() { return Get().current.load(std::memory_order_acquire); }
inline uint64_t Generation() { return Get().generation.load(std::memory_order_acquire); }

// Any thread: the game loads `scene`.
inline void Enter(uint32_t scene) {
	auto& s = Get();
	if (scene == None || s.current.load(std::memory_order_acquire) == scene) return;
	{
		std::lock_guard lock(s.mutex);
		const auto now = std::chrono::steady_clock::now();
		if (const auto was = s.current.load(std::memory_order_relaxed); was != None) s.played[was] += now - s.entered_at;
		s.entered_at = now;
		s.current.store(scene, std::memory_order_release);
		s.generation.fetch_add(1, std::memory_order_acq_rel);
	}
	std::printf("Scene: %s\n", Name(scene).c_str());
	std::fflush(stdout);
}

// A pipeline of recipe `recipe` bound in the current scene (BindPipelineHandle: once per scene).
inline void NoteUse(uint64_t recipe) {
	auto&      s     = Get();
	const auto scene = s.current.load(std::memory_order_acquire);
	if (scene == None || recipe == 0) return;
	std::lock_guard lock(s.mutex);
	if (s.tags[scene].insert(recipe).second) s.dirty = true;
}

// Whether `scene` was played long enough in an earlier session for its recipes to stand for it.
inline bool Knows(uint32_t scene) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	const auto      it = s.visits.find(scene);
	return it != s.visits.end() && it->second != 0 && s.tags.contains(scene);
}

inline std::unordered_set<uint64_t> Recipes(uint32_t scene) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	const auto      it = s.tags.find(scene);
	return it != s.tags.end() ? it->second : std::unordered_set<uint64_t> {};
}

// The file beside a warmup recording, tied to its identity line (another recording's recipe indices mean other recipes).
// Layout: "KytySceneTags1:" identity, XXH3 of the payload, payload = scene count, per scene: id, visits, recipe count,
// recipes. A missing, foreign or damaged file starts empty.
inline bool Load(const std::filesystem::path& path, const std::string& identity) {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	s.path     = path;
	s.identity = "KytySceneTags1:" + identity;
	s.tags.clear();
	s.visits.clear();
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return true;
	const auto size = static_cast<uint64_t>(file.tellg());
	if (size < s.identity.size() + 8 || (size - s.identity.size() - 8) % 4 != 0) return false;
	std::string saved(s.identity.size(), '\0');
	uint64_t    checksum = 0;
	std::vector<uint32_t> words((size - s.identity.size() - 8) / 4);
	file.seekg(0);
	file.read(saved.data(), static_cast<std::streamsize>(saved.size()));
	file.read(reinterpret_cast<char*>(&checksum), 8);
	file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
	if (!file || saved != s.identity || XXH3_64bits(words.data(), words.size() * 4) != checksum) return false;
	size_t     at   = 0;
	const auto take = [&](uint32_t& out) { return at < words.size() ? (out = words[at++], true) : false; };
	uint32_t   scenes = 0;
	if (!take(scenes)) return false;
	for (uint32_t i = 0; i < scenes; ++i) {
		uint32_t scene = 0, visits = 0, count = 0;
		if (!take(scene) || !take(visits) || !take(count) || words.size() - at < uint64_t {count} * 2) {
			s.tags.clear();
			s.visits.clear();
			return false;
		}
		auto& set = s.tags[scene];
		for (uint32_t r = 0; r < count; ++r, at += 2) set.insert(uint64_t {words[at]} | uint64_t {words[at + 1]} << 32u);
		s.visits[scene] = visits;
	}
	return at == words.size();
}

// Written to a temporary file renamed over the old one when something changed: the scenes played KnownSeconds this
// session count one visit (once).
inline bool Save() {
	auto&           s = Get();
	std::lock_guard lock(s.mutex);
	if (s.path.empty()) return true;
	auto played = s.played;
	if (const auto now = s.current.load(std::memory_order_relaxed); now != None)
		played[now] += std::chrono::steady_clock::now() - s.entered_at;
	for (const auto& [scene, time]: played) {
		if (time >= std::chrono::seconds(KnownSeconds) && s.tags.contains(scene) && s.counted.insert(scene).second) {
			s.visits[scene]++;
			s.dirty = true;
		}
	}
	if (!s.dirty) return true;
	std::vector<uint32_t> words {static_cast<uint32_t>(s.tags.size())};
	for (const auto& [scene, recipes]: s.tags) {
		words.push_back(scene);
		words.push_back(s.visits[scene]);
		words.push_back(static_cast<uint32_t>(recipes.size()));
		for (const auto recipe: recipes) {
			words.push_back(static_cast<uint32_t>(recipe));
			words.push_back(static_cast<uint32_t>(recipe >> 32u));
		}
	}
	const uint64_t  checksum = XXH3_64bits(words.data(), words.size() * 4);
	auto            temporary = s.path;
	temporary += ".tmp";
	std::ofstream   file(temporary, std::ios::binary | std::ios::trunc);
	file.write(s.identity.data(), static_cast<std::streamsize>(s.identity.size()));
	file.write(reinterpret_cast<const char*>(&checksum), 8);
	file.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
	file.close();
	std::error_code error;
	if (!file) return false;
	std::filesystem::rename(temporary, s.path, error);
	if (error) return false;
	s.dirty = false;
	return true;
}

} // namespace SceneTags
