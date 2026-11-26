#pragma once
// (tiron6, KYTY_TIRON6=1) Compute pipeline seeds: the recipe key (LocalShaderWarmup::RecipeKey, the recording's compute
// recipe) of every compute pipeline the game made at run time, kept in _PipelineCache/local/tiron6-compute.seeds across
// sessions. PipelineCache::Tiron1WarmCompute makes them first at start-up (below normal priority, without waiting for the
// shader prefetch and without the 1500 cap), then every other compute recipe of the recording. A pipeline still not made
// when the game needs it is made where it is needed, as before: no dispatch is ever skipped.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Tiron6 {
inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON6");
		return value == nullptr || value[0] != '0'; // (picos) on by default
	}();
	return on;
}

struct Seeds {
	std::mutex                   mutex;
	bool                         loaded = false;
	std::vector<uint64_t>        order;   // (the file's, then this session's new ones)
	std::unordered_set<uint64_t> known;
	size_t                       written = 0; // (order[0, written) is on the disk)
	size_t                       from_disk = 0;
};
inline Seeds& State() {
	static Seeds seeds;
	return seeds;
}
inline const std::filesystem::path& Path() {
	static const std::filesystem::path path = std::filesystem::path("_PipelineCache") / "local" / "tiron6-compute.seeds";
	return path;
}
constexpr uint64_t Magic = 0x3153363654594b54ull; // "TKYT66S1"

// (Under State().mutex.)
inline void LoadLocked(Seeds& s) {
	if (s.loaded) return;
	s.loaded = true;
	std::FILE* file = std::fopen(Path().string().c_str(), "rb");
	if (file == nullptr) return;
	uint64_t magic = 0;
	if (std::fread(&magic, sizeof(magic), 1, file) == 1 && magic == Magic) {
		uint64_t key = 0;
		while (std::fread(&key, sizeof(key), 1, file) == 1)
			if (key != 0 && s.known.insert(key).second) s.order.push_back(key);
	} else {
		s.order.clear();
		s.known.clear();
	}
	std::fclose(file);
	s.written   = s.order.size();
	s.from_disk = s.order.size();
}

// (tiron10, KYTY_TIRON10=1) The graphics pipelines the game made at run time are seeds too (same file), and the start-up
// warm makes them with the compute ones (PipelineCache::Tiron1WarmCompute): 20261120-004624, 15 graphics pipelines made on
// the render thread in one frame (graphics_pipeline_us 91636, the 576.7 ms frame; shadow-layer draws keep the blocking
// compile so that no shadow is skipped).
inline bool GraphicsOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON10");
		const bool  set   = value == nullptr || value[0] != '0'; // (picos) on by default
		std::printf("[tiron10] pipelines graficos en la precarga (semillas de sesiones anteriores): %s\n",
		            set ? "ON (KYTY_TIRON10=1)" : "off");
		std::fflush(stdout);
		return set;
	}();
	return on;
}

inline bool IsSeed(uint64_t key) {
	auto&           s = State();
	std::lock_guard lock(s.mutex);
	LoadLocked(s);
	return s.known.contains(key);
}

inline size_t SeedCount() {
	auto&           s = State();
	std::lock_guard lock(s.mutex);
	LoadLocked(s);
	return s.from_disk;
}

// GPU thread, when the game made a compute pipeline (CreateComputePipeline, TryCreateComputePipelineAsync): no disk here.
inline void Note(uint64_t key) {
	if (!On() || key == 0) return;
	auto&           s = State();
	std::lock_guard lock(s.mutex);
	LoadLocked(s);
	if (s.known.insert(key).second) s.order.push_back(key);
}

// The warmup thread every 5 s and PipelineCache::Save: the new seeds appended to the file (written first if missing).
inline void Flush() {
	if (!On()) return;
	auto&                 s = State();
	std::vector<uint64_t> fresh;
	bool                  create = false;
	{
		std::lock_guard lock(s.mutex);
		LoadLocked(s);
		if (s.written >= s.order.size()) return;
		std::error_code error;
		create = !std::filesystem::exists(Path(), error) || s.written == 0;
		fresh.assign(s.order.begin() + static_cast<std::ptrdiff_t>(create ? 0 : s.written), s.order.end());
		s.written = s.order.size();
	}
	std::error_code error;
	std::filesystem::create_directories(Path().parent_path(), error);
	std::FILE* file = std::fopen(Path().string().c_str(), create ? "wb" : "ab");
	if (file == nullptr) return;
	if (create) std::fwrite(&Magic, sizeof(Magic), 1, file);
	if (!fresh.empty()) std::fwrite(fresh.data(), sizeof(uint64_t), fresh.size(), file);
	std::fclose(file);
}

// The recording's recipes reordered for the warm (taken from the back, newest first): the seeds at the back, in the
// order the game first used them (the first used last, so taken first); the others before, as recorded.
template <typename Recipe, typename KeyOf>
inline void OrderBySeeds(std::vector<Recipe>& recipes, KeyOf key_of) {
	std::unordered_map<uint64_t, size_t> rank;
	{
		auto&           s = State();
		std::lock_guard lock(s.mutex);
		LoadLocked(s);
		for (size_t i = 0; i < s.order.size(); ++i) rank.emplace(s.order[i], i);
	}
	if (rank.empty()) return;
	std::vector<std::pair<size_t, size_t>> seeded; // (rank, index)
	std::vector<Recipe>                    others;
	others.reserve(recipes.size());
	for (size_t i = 0; i < recipes.size(); ++i) {
		const auto found = rank.find(key_of(recipes[i]));
		if (found != rank.end()) seeded.emplace_back(found->second, i);
	}
	std::vector<uint8_t> is_seed(recipes.size(), 0);
	for (const auto& [r, i]: seeded) is_seed[i] = 1;
	// Highest rank (used last) nearest the front of the seeds' block, rank 0 at the very back.
	std::sort(seeded.begin(), seeded.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < recipes.size(); ++i)
		if (!is_seed[i]) others.push_back(std::move(recipes[i]));
	for (const auto& [r, i]: seeded) others.push_back(std::move(recipes[i]));
	recipes = std::move(others);
}

// The compute pipelines' video memory cap: KYTY_TIRON6_VIDEO_MIB, else 1/8 of the budget (Tiron1: 1/16), never below
// the caller's.
inline uint64_t VideoCap(uint64_t current, uint64_t budget) {
	if (const char* value = std::getenv("KYTY_TIRON6_VIDEO_MIB"); value != nullptr && *value != 0)
		return std::strtoull(value, nullptr, 10) << 20u;
	return current == UINT64_MAX ? current : std::max(current, budget / 8u);
}
// (tiron6 v2) Tiron1WarmCompute finished (or gave up): the start-up's wait (PipelineCache::Tiron6WaitInFront) ends.
inline std::atomic<bool> g_done {false};
// (tiron6 v2) The splash's bar and status line (port/ds-main.cpp sets it; null without the port's splash).
inline std::atomic<void (*)(const char*, float)> g_splash {nullptr};
inline void Report(const char* text, float bar) {
	if (auto* report = g_splash.load(std::memory_order_acquire); report != nullptr) report(text, bar);
}
// (ram16-t6, 11-23) KYTY_TIRON6_SEEDS_ONLY=1 (on by default in the 16 GB profile, ds-settings): the start-up warm makes
// only the compute pipelines the game itself made in earlier sessions (the seeds), not every other compute recipe of the
// recording. Log 20261123-082215 (16 GB): the warm made 4025 of 4722 recipes, +3935 MiB resident (the emulator 4.3 -> 8.2
// GB), and the PC played with 150-440 MiB available and Windows compressing up to 2.4 GB: the 321 ms stall had all game
// threads 300-650 ms in neither CPU nor a wait (hard faults / decompression). A seed-less recipe still needed is made on
// use, as before (and becomes a seed). Fewer than KYTY_TIRON6_SEEDS_ONLY_MIN seeds (300; a new install): everything.
inline bool SeedsOnly() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON6_SEEDS_ONLY");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}
inline size_t SeedsOnlyMin() {
	static const size_t minimum = [] {
		const char* value = std::getenv("KYTY_TIRON6_SEEDS_ONLY_MIN");
		return value != nullptr && *value != 0 ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t {300};
	}();
	return minimum;
}
// KYTY_TIRON6_MAX_S: how long the start-up waits for them (default 60 s; 0: not at all, beside the game as v1).
inline double MaxSeconds() {
	static const double seconds = [] {
		const char* value = std::getenv("KYTY_TIRON6_MAX_S");
		if (value == nullptr || *value == 0) return 60.0;
		const double asked = std::strtod(value, nullptr);
		return asked < 0.0 ? 0.0 : std::min(asked, 3600.0);
	}();
	return seconds;
}
// The warm's worker threads: KYTY_TIRON6_THREADS, else every CPU (`cpus`) but at most one per 256 MiB of free RAM
// above a 1.5 GiB reserve (a translation worker's peak; 16 GB PC with ~6 GiB free at start-up: 18 threads), at least 4.
inline uint32_t Threads(uint32_t cpus, uint64_t available_bytes) {
	if (const char* value = std::getenv("KYTY_TIRON6_THREADS"); value != nullptr && *value != 0) {
		const long asked = std::strtol(value, nullptr, 10);
		if (asked > 0) return static_cast<uint32_t>(std::min<long>(asked, 64));
	}
	if (available_bytes == 0) return cpus;
	const uint64_t mib     = available_bytes >> 20u;
	const uint64_t by_ram  = mib > 1536 ? (mib - 1536) / 256 : 0;
	return static_cast<uint32_t>(std::max<uint64_t>(4, std::min<uint64_t>(cpus, by_ram)));
}
// (solo-usados, 11-26, KYTY_TIRON6_SOLO_USADOS=1, default off) Log 20261126-222050: the warm of every seed made 4041
// pipelines, +3868 MiB resident in the NVIDIA driver, and evicting them later gives none of it back (the driver keeps its
// heap). So fewer are made: each session notes the seeds it really BOUND (BindPipelineHandle, once per pipeline) in
// _PipelineCache/local/tiron6-usados.seeds (the last Sessions() sessions, each a block); the warm then makes only the
// seeds bound in any of them. The others are made on use from the driver's VkPipelineCache (0.4-6.5 ms each). No usage
// recorded yet (first session with it on) or too few: every seed, as before. A session that bound fewer than
// UsedSessionMin() pipelines (a crash at the menu) is not kept.
inline bool SoloUsados() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TIRON6_SOLO_USADOS");
		const bool  set   = value == nullptr || value[0] != '0'; // (por defecto ON desde 11-26 23:20; =0 lo apaga)
		std::printf("[solo-usados] precarga solo de pipelines usados en las ultimas sesiones: %s\n",
		            set ? "ON (KYTY_TIRON6_SOLO_USADOS=1)" : "off");
		std::fflush(stdout);
		return set;
	}();
	return on;
}
inline size_t EnvSize(const char* name, size_t fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && *value != 0 ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}
inline size_t Sessions() {
	static const size_t n = std::max<size_t>(1, std::min<size_t>(16, EnvSize("KYTY_TIRON6_USADOS_SESIONES", 3)));
	return n;
}
inline size_t UsedSessionMin() {
	static const size_t n = EnvSize("KYTY_TIRON6_USADOS_MIN", 200);
	return n;
}
struct Used {
	std::mutex                          mutex;
	bool                                loaded = false;
	std::vector<std::vector<uint64_t>>  past;   // (oldest first, at most Sessions())
	std::unordered_set<uint64_t>        warm;   // (the union of `past`: what the warm makes)
	std::vector<uint64_t>               now;    // (this session's, in binding order)
	std::unordered_set<uint64_t>        now_set;
	size_t                              written = 0; // (now[0, written) on the disk)
};
inline Used& UsedState() {
	static Used used;
	return used;
}
inline const std::filesystem::path& UsedPath() {
	static const std::filesystem::path path = std::filesystem::path("_PipelineCache") / "local" / "tiron6-usados.seeds";
	return path;
}
constexpr uint64_t UsedMagic = 0x3155363654594b54ull; // "TKYT66U1"
inline void UsedLoadLocked(Used& u) {
	if (u.loaded) return;
	u.loaded = true;
	std::FILE* file = std::fopen(UsedPath().string().c_str(), "rb");
	if (file == nullptr) return;
	uint64_t magic = 0, count = 0;
	if (std::fread(&magic, sizeof(magic), 1, file) == 1 && magic == UsedMagic) {
		while (std::fread(&count, sizeof(count), 1, file) == 1 && count <= (1u << 24u)) {
			std::vector<uint64_t> keys(static_cast<size_t>(count));
			if (count != 0 && std::fread(keys.data(), sizeof(uint64_t), keys.size(), file) != keys.size()) break;
			u.past.push_back(std::move(keys));
		}
	}
	std::fclose(file);
	while (u.past.size() > Sessions()) u.past.erase(u.past.begin());
	for (const auto& session: u.past) u.warm.insert(session.begin(), session.end());
}
// The warm's filter: true when usage is known (solo-usados on, and the last sessions bound at least SeedsOnlyMin()).
inline bool UsedKnown() {
	if (!On() || !SoloUsados()) return false;
	auto&           u = UsedState();
	std::lock_guard lock(u.mutex);
	UsedLoadLocked(u);
	return u.warm.size() >= std::min<size_t>(SeedsOnlyMin(), 300);
}
inline bool WasUsed(uint64_t key) {
	auto&           u = UsedState();
	std::lock_guard lock(u.mutex);
	UsedLoadLocked(u);
	return u.warm.contains(key);
}
inline size_t UsedCount() {
	auto&           u = UsedState();
	std::lock_guard lock(u.mutex);
	UsedLoadLocked(u);
	return u.warm.size();
}
inline size_t UsedSessionsKnown() {
	auto&           u = UsedState();
	std::lock_guard lock(u.mutex);
	UsedLoadLocked(u);
	return u.past.size();
}
// Render/GPU threads (BindPipelineHandle), once per pipeline object: no disk here.
inline void NoteBound(uint64_t key) {
	if (key == 0 || !On() || !SoloUsados()) return;
	auto&           u = UsedState();
	std::lock_guard lock(u.mutex);
	if (u.now_set.insert(key).second) u.now.push_back(key);
}
// Warmup thread every 5 s and PipelineCache::Save: the file rewritten (past sessions but the oldest, then this one).
inline void UsedFlush() {
	if (!On() || !SoloUsados()) return;
	auto&                              u = UsedState();
	std::vector<std::vector<uint64_t>> keep;
	std::vector<uint64_t>              current;
	{
		std::lock_guard lock(u.mutex);
		UsedLoadLocked(u);
		if (u.written >= u.now.size() || u.now.size() < UsedSessionMin()) return;
		u.written = u.now.size();
		const size_t skip = u.past.size() >= Sessions() ? u.past.size() - (Sessions() - 1) : 0;
		keep.assign(u.past.begin() + static_cast<std::ptrdiff_t>(skip), u.past.end());
		current = u.now;
	}
	std::error_code error;
	std::filesystem::create_directories(UsedPath().parent_path(), error);
	auto       temporary = UsedPath();
	temporary += ".tmp";
	std::FILE* file      = std::fopen(temporary.string().c_str(), "wb");
	if (file == nullptr) return;
	bool ok = std::fwrite(&UsedMagic, sizeof(UsedMagic), 1, file) == 1;
	keep.push_back(std::move(current));
	for (const auto& session: keep) {
		const uint64_t count = session.size();
		ok = ok && std::fwrite(&count, sizeof(count), 1, file) == 1;
		if (count != 0) ok = ok && std::fwrite(session.data(), sizeof(uint64_t), session.size(), file) == session.size();
	}
	ok = std::fclose(file) == 0 && ok;
	if (ok) std::filesystem::rename(temporary, UsedPath(), error);
}
} // namespace Tiron6
