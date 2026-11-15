#include "graphics/host_gpu/renderer/pipeline/stageStore.h"

#include "common/config.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <system_error>
#include <xxhash.h>
#include <zstd.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace Libs::Graphics::Precompile {

namespace {

constexpr std::array<char, 8> Magic {'K', 'Y', 'S', 'T', 'A', 'G', 'E', '1'};

} // namespace

uint64_t StageStore::BuildStamp() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	const auto length = GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	if (length == 0 || length >= name.size()) return 0;
	const std::filesystem::path path = name.data();
#else
	const std::filesystem::path path = "/proc/self/exe";
#endif
	std::error_code error;
	const auto      time = std::filesystem::last_write_time(path, error).time_since_epoch().count();
	return error ? 0 : static_cast<uint64_t>(time);
}

bool StageStore::Open(const std::filesystem::path& path, std::span<const uint8_t> header, bool writing) {
	std::lock_guard lock(m_mutex);
	m_index.clear();
	m_writing = false;
	std::error_code error;
	const auto      size = std::filesystem::file_size(path, error);
	bool            same = false;
	m_end                = Magic.size() + sizeof(uint32_t) + header.size();
	if (!error && size >= m_end) {
		std::ifstream file(path, std::ios::binary);
		std::array<char, Magic.size()> magic {};
		uint32_t                       length = 0;
		std::vector<uint8_t>           stored(header.size());
		file.read(magic.data(), magic.size());
		file.read(reinterpret_cast<char*>(&length), sizeof(length));
		file.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(stored.size()));
		same = file && magic == Magic && length == header.size() && std::ranges::equal(stored, header);
		// The index: the frames up to the first one cut short.
		for (Frame frame; same && file.read(reinterpret_cast<char*>(&frame), sizeof(frame));) {
			const auto offset = m_end + sizeof(frame);
			if (offset + frame.compressed > size) break;
			m_index.try_emplace(frame.key, Entry {.offset = offset, .compressed = frame.compressed, .raw = frame.raw,
			                                      .checksum = frame.checksum});
			m_end = offset + frame.compressed;
			file.seekg(static_cast<std::streamoff>(m_end));
		}
	}
	if (!writing) {
		if (!same) return false;
		m_file.open(path, std::ios::binary | std::ios::in);
		return m_file.is_open();
	}
	std::filesystem::create_directories(path.parent_path(), error);
	if (!same) {
		m_index.clear();
		m_end = Magic.size() + sizeof(uint32_t) + header.size();
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		const auto    length = static_cast<uint32_t>(header.size());
		file.write(Magic.data(), Magic.size());
		file.write(reinterpret_cast<const char*>(&length), sizeof(length));
		file.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
		if (!file) return false;
	} else if (m_end != size) {
		std::filesystem::resize_file(path, m_end, error); // a torn tail
		if (error) return false;
	}
	m_file.open(path, std::ios::binary | std::ios::in | std::ios::out);
	m_writing = m_file.is_open();
	return m_writing;
}

std::optional<std::vector<uint8_t>> StageStore::Find(uint64_t key) {
	std::vector<uint8_t> compressed;
	Entry                entry;
	{
		std::lock_guard lock(m_mutex);
		const auto      found = m_index.find(key);
		if (found == m_index.end() || !m_file.is_open()) return std::nullopt;
		entry = found->second;
		compressed.resize(entry.compressed);
		m_file.clear();
		m_file.seekg(static_cast<std::streamoff>(entry.offset));
		m_file.read(reinterpret_cast<char*>(compressed.data()), static_cast<std::streamsize>(compressed.size()));
		if (!m_file) return std::nullopt;
	}
	if (XXH3_64bits(compressed.data(), compressed.size()) != entry.checksum) return std::nullopt;
	std::vector<uint8_t> raw(entry.raw);
	const auto size = ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size());
	if (ZSTD_isError(size) || size != raw.size()) return std::nullopt;
	return raw;
}

void StageStore::Append(uint64_t key, std::span<const uint8_t> record) {
	if (!m_writing || record.empty()) return;
	{
		std::lock_guard lock(m_mutex);
		if (m_index.contains(key)) return;
	}
	std::vector<uint8_t> bytes(ZSTD_compressBound(record.size()));
	const auto size = ZSTD_compress(bytes.data(), bytes.size(), record.data(), record.size(), 3);
	if (ZSTD_isError(size)) return;
	bytes.resize(size);
	const Frame frame {.compressed = static_cast<uint32_t>(size), .raw = static_cast<uint32_t>(record.size()), .key = key,
	                   .checksum = XXH3_64bits(bytes.data(), bytes.size())};
	std::lock_guard lock(m_mutex);
	if (m_index.contains(key)) return;
	m_file.clear();
	m_file.seekp(static_cast<std::streamoff>(m_end));
	m_file.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
	m_file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	m_file.flush();
	if (!m_file) {
		m_writing = false; // a full disk: the store stops growing, its whole records stay
		return;
	}
	m_index.emplace(key, Entry {.offset = m_end + sizeof(frame), .compressed = frame.compressed, .raw = frame.raw,
	                            .checksum = frame.checksum});
	m_end += sizeof(frame) + bytes.size();
}

size_t StageStore::Size() {
	std::lock_guard lock(m_mutex);
	return m_index.size();
}

uint64_t StageStore::Bytes() {
	std::lock_guard lock(m_mutex);
	return m_end;
}

} // namespace Libs::Graphics::Precompile
