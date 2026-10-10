#include "loader/demonsSoulsWarp.h"

#include "common/file.h"
#include "common/stringUtils.h"
#include "kernel/fileSystem.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <optional>

namespace Loader::DemonsSoulsWarp {

namespace {

// USR-DATA: u32 version, u32 FNV-1a 32 of the payload, u32 payload size, u32, then entries of key
// (FNV-1a 32 of an upper-case name), type byte, value (tools/local/demons-souls-save.py).
constexpr size_t   HeaderSize  = 16;
constexpr uint32_t PositionKey = 0xfb0faeb2; // 16-byte blob x y z 1 (name unknown)
constexpr uint32_t RotationKey = 0x739273b8; // 16-byte blob 0 yaw(radians) 0 0 (name unknown)
// Byte, 1 when the position and rotation entries follow it (name unknown). A character still in the
// tutorial has 0 and no such entries; every save the game wrote with 1 has them right after it.
constexpr uint32_t PlacedKey   = 0x4819bb27;
constexpr size_t   BlobEntry   = 6 + 16; // key, type 0x0e, length 16, value

uint32_t Fnv1a(std::span<const uint8_t> data) {
	uint32_t hash = 0x811c9dc5u;
	for (const auto byte: data) hash = (hash ^ byte) * 0x01000193u;
	return hash;
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::vector<uint8_t> data;
	Common::File         in;
	if (in.Open(path, Common::File::Mode::Read)) { // (also a game archive's member)
		data.resize(static_cast<size_t>(in.Size()));
		uint32_t read = 0;
		in.Read(data.data(), static_cast<uint32_t>(data.size()), &read);
		data.resize(read);
	}
	return data;
}

// The value offset of the one entry starting with `prefix` (key and type bytes), 0 if none or several.
size_t FindValue(std::span<const uint8_t> data, std::span<const uint8_t> prefix) {
	size_t found = 0;
	for (auto it = std::search(data.begin(), data.end(), prefix.begin(), prefix.end()); it != data.end();
	     it      = std::search(it + 1, data.end(), prefix.begin(), prefix.end())) {
		if (found != 0) return 0;
		found = static_cast<size_t>(it - data.begin()) + prefix.size();
	}
	return found;
}

template <size_t N>
std::array<uint8_t, N> Prefix(uint32_t key, std::initializer_list<uint8_t> type) {
	std::array<uint8_t, N> prefix {};
	std::memcpy(prefix.data(), &key, sizeof(key));
	std::copy(type.begin(), type.end(), prefix.begin() + sizeof(key));
	return prefix;
}

uint32_t MapUidOf(std::span<const uint8_t> data) {
	const auto prefix = Prefix<5>(Fnv1a({reinterpret_cast<const uint8_t*>("MAPUID"), 6}), {0x04});
	const auto at     = FindValue(data, prefix);
	uint32_t   uid    = 0;
	if (at != 0 && at + sizeof(uid) <= data.size()) std::memcpy(&uid, data.data() + at, sizeof(uid));
	return uid;
}

// Player parts (type 4) of a remake MSB (little-endian, 64-bit offsets): params of name offset (+8),
// count (+16) and offsets (+24), the last one the next param's.
std::vector<Spawn> ReadMsb(const std::filesystem::path& path) {
	const auto data = ReadFile(path);
	const auto read = [&](uint64_t at, auto* out) {
		if (at > data.size() || sizeof(*out) > data.size() - at) return false;
		std::memcpy(out, data.data() + at, sizeof(*out));
		return true;
	};
	const auto text = [&](uint64_t at) {
		const auto end = at < data.size() ? std::find(data.begin() + static_cast<int64_t>(at), data.end(), 0) : data.end();
		return at < data.size() ? std::string(data.begin() + static_cast<int64_t>(at), end) : std::string();
	};
	const auto map = path.stem().string();
	uint32_t   uid = 0;
	for (const char c: map) {
		if (c >= '0' && c <= '9') uid = uid << 4u | static_cast<uint32_t>(c - '0'); // m04_01_00_00 -> 0x04010000
	}
	std::vector<Spawn> out;
	uint64_t           at = 0;
	for (int param = 0; param < 64; ++param) {
		int64_t name_at = 0;
		int32_t count   = 0;
		if (!read(at + 8, &name_at) || !read(at + 16, &count) || count <= 0 || count > 1 << 20) break;
		std::vector<int64_t> offsets(static_cast<size_t>(count));
		for (size_t i = 0; i < offsets.size(); ++i) {
			if (!read(at + 24 + 8 * i, &offsets[i])) return out;
		}
		if (text(static_cast<uint64_t>(name_at)) == "PARTS_PARAM_ST") {
			for (size_t i = 0; i + 1 < offsets.size(); ++i) {
				const auto entry     = static_cast<uint64_t>(offsets[i]);
				int64_t    part_name = 0;
				int32_t    type      = 0;
				float      xyz[3] {}, yaw = 0;
				if (read(entry, &part_name) && read(entry + 8, &type) && type == 4 && read(entry + 0x20, &xyz) &&
				    read(entry + 0x30, &yaw)) {
					out.push_back({map, text(entry + static_cast<uint64_t>(part_name)), uid, xyz[0], xyz[1], xyz[2], yaw});
				}
			}
		}
		if (offsets.back() <= 0) break;
		at = static_cast<uint64_t>(offsets.back());
	}
	return out;
}

std::mutex            g_mutex;
std::atomic_bool      g_active {false};
std::optional<Spawn>  g_armed;
size_t                g_selected = 0;
bool                  g_served   = false; // a patched copy was read since arming
std::filesystem::path g_save;             // the save it was made from
std::filesystem::file_time_type       g_save_time {};
std::chrono::steady_clock::time_point g_shown_until {}, g_checked {};
std::string                           g_message;
bool                                  g_picking = false; // the shown message is the spawn list

void Show(std::string message, int seconds) {
	g_message     = std::move(message);
	g_shown_until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
	g_picking     = false;
}

std::string Describe(const Spawn& spawn) {
	static constexpr const char* Worlds[] = {"", "Nexus", "Boletarian Palace", "Shrine of Storms", "Tower of Latria",
	                                         "Valley of Defilement", "Stonefang Tunnel", "", "Tutorial"};
	const auto world = spawn.map_uid >> 24u;
	const auto name  = world < std::size(Worlds) ? Worlds[world] : "";
	return std::string(name) + (*name != '\0' ? "  " : "") + spawn.map + "  " + spawn.name;
}

// The warp ends once the game saved on the target map after reading the patched copy.
void CheckDoneLocked() {
	if (!g_armed || !g_served || g_save.empty()) return;
	std::error_code error;
	const auto      time = std::filesystem::last_write_time(g_save, error);
	if (error || time <= g_save_time || MapUidOf(ReadFile(g_save)) != g_armed->map_uid) return;
	std::printf("Warp: the game saved on %s, done\n", g_armed->map.c_str());
	std::fflush(stdout);
	Show("Warp done: " + Describe(*g_armed), 4);
	g_armed.reset();
	g_active = false;
}

} // namespace

const std::vector<Spawn>& Spawns() {
	static const std::vector<Spawn> spawns = [] {
		std::vector<Spawn> all;
		const auto         dir = Libs::LibKernel::FileSystem::GetRealFilename("/app0/cp11demonssouls/dvdroot/map/mapstudio");
		std::vector<std::filesystem::path> files;
		for (const auto& entry: Common::File::GetDirEntries(dir)) { // (also a game archive's directory)
			const auto& name = entry.name;
			if (entry.is_file && name.size() == 16 && name.starts_with("m0") && name.ends_with(".msb")) files.push_back(dir / Common::PathFromUtf8(name));
		}
		std::ranges::sort(files);
		for (const auto& file: files) {
			// A character starts at c0000_<n>; the other player parts (m03_00's c1000, the m07 maps',
			// m03_01_00_99's) are no start (m03_00 c1000 crashed the game while loading).
			for (auto& spawn: ReadMsb(file)) {
				if (spawn.name.starts_with("c0000_")) all.push_back(std::move(spawn));
			}
		}
		return all;
	}();
	return spawns;
}

bool Arm(std::string_view map, std::string_view spawn) {
	const auto& spawns = Spawns();
	const auto  it     = std::ranges::find_if(spawns, [&](const Spawn& s) { return s.map == map && s.name == spawn; });
	if (it == spawns.end()) return false;
	std::lock_guard lock(g_mutex);
	g_selected = static_cast<size_t>(it - spawns.begin());
	g_armed    = *it;
	g_served   = false;
	g_save.clear();
	g_active = true;
	Show({}, 0); // HudText shows the armed warp
	std::printf("Warp: armed %s %s\n", it->map.c_str(), it->name.c_str());
	std::fflush(stdout);
	return true;
}

void Disarm() {
	std::lock_guard lock(g_mutex);
	if (g_armed) Show("Warp cancelled", 3);
	g_armed.reset();
	g_active = false;
}

void Select(int step) {
	const auto& spawns = Spawns();
	std::lock_guard lock(g_mutex);
	if (spawns.empty()) {
		Show("Warp: no map files found", 3);
		return;
	}
	// The first press shows the list, the next ones move in it.
	const auto count = static_cast<int64_t>(spawns.size());
	if (g_picking && std::chrono::steady_clock::now() < g_shown_until) {
		g_selected = static_cast<size_t>(((static_cast<int64_t>(g_selected) + step) % count + count) % count);
	}
	std::string list = "Warp: F9/F10 pick, F8 arm (" + std::to_string(g_selected + 1) + "/" + std::to_string(count) + ")";
	for (int64_t offset = -2; offset <= 2; ++offset) {
		const auto index = ((static_cast<int64_t>(g_selected) + offset) % count + count) % count;
		list += (offset == 0 ? "\n> " : "\n   ") + Describe(spawns[static_cast<size_t>(index)]);
	}
	Show(std::move(list), 8);
	g_picking = true;
}

void ToggleSelected() {
	const auto& spawns = Spawns();
	if (g_active) {
		Disarm();
		return;
	}
	size_t selected = 0;
	{
		std::lock_guard lock(g_mutex);
		if (spawns.empty() || !g_picking || std::chrono::steady_clock::now() >= g_shown_until) {
			Show(spawns.empty() ? "Warp: no map files found" : "Warp: pick a spawn point with F9/F10 first", 4);
			return;
		}
		selected = g_selected;
	}
	Arm(spawns[selected].map, spawns[selected].name);
}

std::string HudText() {
	std::lock_guard lock(g_mutex);
	const auto      now = std::chrono::steady_clock::now();
	if (g_active && now - g_checked >= std::chrono::seconds(1)) {
		g_checked = now;
		CheckDoneLocked();
	}
	if (now < g_shown_until) return g_message;
	if (!g_armed) return {};
	return g_served ? "Warp: loading " + Describe(*g_armed)
	                : "Warp armed: " + Describe(*g_armed) +
	                      "\nOptions > Settings > Exit Game, then Continue on the title; F8 cancels";
}

bool PatchSave(std::vector<uint8_t>& data, const Spawn& spawn) {
	if (data.size() < HeaderSize) return false;
	uint32_t checksum = 0, size = 0;
	std::memcpy(&checksum, data.data() + 4, sizeof(checksum));
	std::memcpy(&size, data.data() + 8, sizeof(size));
	if (size != data.size() - HeaderSize || Fnv1a({data.data() + HeaderSize, size}) != checksum) return false;
	// Value offsets in `data`, 0 when the entry is missing or not unique.
	const auto find = [&](std::span<const uint8_t> prefix) {
		const auto at = FindValue({data.data() + HeaderSize, data.size() - HeaderSize}, prefix);
		return at == 0 ? 0 : HeaderSize + at;
	};
	const auto position_prefix = Prefix<6>(PositionKey, {0x0e, 0x10});
	const auto rotation_prefix = Prefix<6>(RotationKey, {0x0e, 0x10});
	auto       position        = find(position_prefix);
	auto       rotation        = find(rotation_prefix);
	if (position == 0 && rotation == 0) {
		const auto placed = find(Prefix<5>(PlacedKey, {0x0c}));
		if (placed == 0 || data[placed] != 0) return false;
		std::array<uint8_t, 2 * BlobEntry> entries {};
		std::copy(position_prefix.begin(), position_prefix.end(), entries.begin());
		std::copy(rotation_prefix.begin(), rotation_prefix.end(), entries.begin() + BlobEntry);
		data[placed] = 1;
		data.insert(data.begin() + static_cast<int64_t>(placed) + 1, entries.begin(), entries.end());
		position = placed + 1 + position_prefix.size();
		rotation = placed + 1 + BlobEntry + rotation_prefix.size();
		size     = static_cast<uint32_t>(data.size() - HeaderSize);
		std::memcpy(data.data() + 8, &size, sizeof(size));
	}
	const auto map = find(Prefix<5>(Fnv1a({reinterpret_cast<const uint8_t*>("MAPUID"), 6}), {0x04}));
	if (map == 0 || position == 0 || rotation == 0 || map + 4 > data.size() || position + 16 > data.size() ||
	    rotation + 16 > data.size()) {
		return false;
	}
	const float place[4] {spawn.x, spawn.y, spawn.z, 1.0f};
	const float turn[4] {0.0f, spawn.yaw * 3.14159265358979f / 180.0f, 0.0f, 0.0f};
	std::memcpy(data.data() + map, &spawn.map_uid, sizeof(spawn.map_uid));
	std::memcpy(data.data() + position, place, sizeof(place));
	std::memcpy(data.data() + rotation, turn, sizeof(turn));
	checksum = Fnv1a({data.data() + HeaderSize, data.size() - HeaderSize});
	std::memcpy(data.data() + 4, &checksum, sizeof(checksum));
	return true;
}

// A character's save while a warp is armed: SAVEDATA0PlayerProfile<n>/USR-DATA (the options are
// SAVEDATA0OptionsProfile0's).
static bool IsCharacterSave(const std::filesystem::path& real) {
	return g_active.load(std::memory_order_relaxed) && real.filename() == "USR-DATA" &&
	       real.parent_path().filename().string().find("PlayerProfile") != std::string::npos;
}

std::optional<uint64_t> RedirectedSize(const std::filesystem::path& real) {
	if (!IsCharacterSave(real)) return std::nullopt;
	std::lock_guard lock(g_mutex);
	CheckDoneLocked();
	if (!g_armed) return std::nullopt;
	auto data = ReadFile(real);
	if (!PatchSave(data, *g_armed)) return std::nullopt;
	return data.size();
}

std::filesystem::path RedirectRead(const std::filesystem::path& real) {
	if (!IsCharacterSave(real)) return {};
	std::lock_guard lock(g_mutex);
	CheckDoneLocked();
	if (!g_armed) return {};
	auto data = ReadFile(real);
	if (!PatchSave(data, *g_armed)) {
		Show("Warp: the save's layout is not the expected one, it is read as it is", 6);
		std::printf("Warp: %s has an unexpected layout, read as it is\n", real.string().c_str());
		std::fflush(stdout);
		return {};
	}
	// The copy goes next to the temporary files; a copy that cannot be written leaves the save as it is
	// (a failing open would look like a missing save to the game).
	std::error_code error;
	const auto      copy = std::filesystem::temp_directory_path(error) / "kyty-warp-USR-DATA";
	std::ofstream   out(copy, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
	out.close();
	if (error || !out) {
		Show("Warp: no temporary file for the save copy", 6);
		return {};
	}
	if (!g_served) {
		g_save_time = std::filesystem::last_write_time(real, error);
		g_save      = real;
	}
	g_served = true;
	std::printf("Warp: %s read as %s %s\n", real.string().c_str(), g_armed->map.c_str(), g_armed->name.c_str());
	std::fflush(stdout);
	return copy;
}

void NoteWrite(const std::filesystem::path& real) {
	if (!g_active.load(std::memory_order_relaxed) || real.filename() != "USR-DATA") return;
	std::printf("Warp: the game writes %s\n", real.string().c_str());
	std::fflush(stdout);
}

} // namespace Loader::DemonsSoulsWarp
