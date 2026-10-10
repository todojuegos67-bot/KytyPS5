// Common::File on ZArchive members ("game.zar!/dir/file"): after upstream's ArchiveFileTests (KytyPS5 #724), adapted
// to this tree's File API, plus reads that span the parallel lanes (archive.cpp ArchiveIo).
#include "ArchiveTestFixture.h"
#include "common/archive.h"
#include "common/file.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <zarchive/zarchivecommon.h>

namespace {

void Check(bool value, const char* text) {
	if (std::getenv("KYTY_TEST_TRACE") != nullptr) {
		std::printf("check: %s\n", text);
		std::fflush(stdout);
	}
	if (!value) {
		std::fprintf(stderr, "ArchiveFileTests: failed: %s\n", text);
		std::abort();
	}
}

std::vector<uint8_t> ReadAll(Common::File& file) {
	std::vector<uint8_t> data(static_cast<size_t>(file.Size()));
	uint32_t             read = 0;
	file.Read(data.data(), static_cast<uint32_t>(data.size()), &read);
	data.resize(read);
	return data;
}

class TempDirectory {
public:
	TempDirectory() {
		const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
		m_path            = std::filesystem::temp_directory_path() / ("kyty_zarchive_test_!_" + std::to_string(unique));
		Check(std::filesystem::create_directories(m_path), "create temporary directory");
	}

	~TempDirectory() {
		std::error_code error;
		std::filesystem::remove_all(m_path, error);
	}

	[[nodiscard]] const std::filesystem::path& Path() const { return m_path; }

	KYTY_CLASS_NO_COPY(TempDirectory);

private:
	std::filesystem::path m_path;
};

template <typename T>
void SetValue(std::vector<uint8_t>& bytes, size_t offset, T value) {
	value = _ZARCHIVE::_store(value);
	Check(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset, "patch lies inside fixture");
	std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void SetMemberNameOffset(std::vector<uint8_t>& bytes, const _ZARCHIVE::Footer& footer, uint32_t name_offset) {
	const auto entry_offset = footer.sectionFileTree.offset + sizeof(_ZARCHIVE::FileDirectoryEntry);
	uint32_t   original     = 0;
	std::memcpy(&original, bytes.data() + entry_offset, sizeof(original));
	SetValue(bytes, entry_offset, (_ZARCHIVE::_load(original) & 0x80000000u) | name_offset);
}

template <typename Edit>
std::filesystem::path MutateArchive(const std::filesystem::path& source, const char* name, Edit edit) {
	Common::File input(source, Common::File::Mode::Read);
	auto         bytes = ReadAll(input);
	_ZARCHIVE::Footer footer {};
	Check(bytes.size() >= sizeof(footer), "fixture has a footer");
	std::memcpy(&footer, bytes.data() + bytes.size() - sizeof(footer), sizeof(footer));
	_ZARCHIVE::Footer::Deserialize(&footer, &footer);
	edit(bytes, footer);
	_ZARCHIVE::Footer::Serialize(&footer, &footer);
	std::memcpy(bytes.data() + bytes.size() - sizeof(footer), &footer, sizeof(footer));
	const auto   destination = source.parent_path() / name;
	Common::File output;
	Check(output.Create(destination), "create malformed archive fixture");
	uint32_t written = 0;
	output.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &written);
	Check(written == bytes.size(), "write malformed archive fixture");
	return destination;
}

void CheckMalformedArchives(const std::filesystem::path& source, const std::vector<uint8_t>& payload) {
	const auto bad_name = MutateArchive(source, "bad-name.zar", [](auto& bytes, auto& footer) {
		SetMemberNameOffset(bytes, footer, static_cast<uint32_t>(footer.sectionNames.size));
	});
	Check(!Common::OpenArchive(bad_name), "reject name offset at the end of name table");

	const auto bad_length = MutateArchive(source, "bad-length.zar", [](auto& bytes, auto& footer) {
		SetMemberNameOffset(bytes, footer, static_cast<uint32_t>(footer.sectionNames.size - 1));
		bytes[footer.sectionNames.offset + footer.sectionNames.size - 1] = 0x80;
	});
	Check(!Common::OpenArchive(bad_length), "reject truncated extended name length");

	const auto bad_children = MutateArchive(source, "bad-children.zar", [](auto& bytes, auto& footer) {
		SetValue(bytes, footer.sectionFileTree.offset + sizeof(uint32_t), UINT32_MAX);
	});
	Check(!Common::OpenArchive(bad_children), "reject invalid directory children without throwing");

	const auto bad_range = MutateArchive(source, "bad-range.zar", [](auto&, auto& footer) {
		footer.sectionCompressedData.offset = std::numeric_limits<uint64_t>::max();
	});
	Check(!Common::OpenArchive(bad_range), "reject overflowing footer section range");

	const auto truncated = source.parent_path() / "truncated.zar";
	Check(std::filesystem::copy_file(source, truncated), "copy truncated archive fixture");
	std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
	Check(!Common::OpenArchive(truncated), "reject missing footer byte");

	const auto corrupted = MutateArchive(source, "corrupt-data.zar", [](auto& bytes, auto& footer) {
		_ZARCHIVE::CompressionOffsetRecord record {};
		std::memcpy(&record, bytes.data() + footer.sectionOffsetRecords.offset, sizeof(record));
		_ZARCHIVE::CompressionOffsetRecord::Deserialize(&record, 1, &record);
		const auto second_block = footer.sectionCompressedData.offset + record.baseOffset + record.size[0] + 1;
		std::fill_n(bytes.data() + second_block, record.size[1] + 1, uint8_t {0});
	});
	Common::File damaged(Common::MakeArchivePath(corrupted, "assets/subdir/data.bin"), Common::File::Mode::Read);
	Check(!damaged.IsInvalid(), "open archive with damaged compressed data");
	std::vector<uint8_t> bytes(payload.size());
	uint32_t             read = 0;
	damaged.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
	Check(read > 0 && read < 64 * 1024 && damaged.Tell() == read &&
	          std::equal(bytes.begin(), bytes.begin() + read, payload.begin()),
	      "report the valid prefix before a corrupt compressed block");
	Check(damaged.Seek(2 * 64 * 1024), "seek past corrupt compressed block");
	damaged.Read(bytes.data(), 97, &read);
	Check(read == 97 && std::equal(bytes.begin(), bytes.begin() + read, payload.begin() + 2 * 64 * 1024),
	      "recover by reading an unaffected compressed block");
}

} // namespace

int main() {
	TempDirectory temporary;
	const auto    archive_path = temporary.Path() / Common::PathFromUtf8("game-\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88.zar");

	// Bigger than the chunks a read keeps in flight on the lanes (ArchiveIo: 8 x 1 MiB), not a multiple of a chunk.
	std::vector<uint8_t> payload(19 * 1024 * 1024 + 37);
	for (size_t index = 0; index < payload.size(); index++) {
		payload[index] = static_cast<uint8_t>((index * 37 + 11 + (index >> 16) * 101) & 0xff);
	}
	Check(ArchiveTests::CreateArchive(archive_path, payload), "create archive fixture");

	const auto native_file      = Common::File::GetInfo(archive_path);
	const auto native_directory = Common::File::GetInfo(temporary.Path() / "");
	Check(native_file && native_file->is_file && native_file->size == std::filesystem::file_size(archive_path),
	      "query native file type and size together");
	Check(native_directory && !native_directory->is_file && native_directory->size == 0,
	      "query native directory with trailing separator");
	Check(!Common::File::GetInfo(archive_path / "") && Common::File::Size(archive_path / "") == 0 &&
	          !Common::File::GetInfo(temporary.Path() / "missing.bin"),
	      "reject missing native paths and files with a trailing separator");

	const auto root       = Common::MakeArchivePath(archive_path);
	auto       reader_pin = Common::OpenArchive(archive_path);
	Check(reader_pin && Common::OpenArchive(root) == reader_pin, "mounted archive pin reuses the same reader and index");
	const auto archive_file      = Common::File::GetInfo(root / "assets/subdir/data.bin");
	const auto archive_directory = Common::File::GetInfo(root / "assets/");
	Check(archive_file && archive_file->is_file && archive_file->size == payload.size(),
	      "query archive member type and size together");
	Check(archive_directory && !archive_directory->is_file && archive_directory->size == 0 &&
	          !Common::File::GetInfo(root / "missing.bin"),
	      "distinguish archive directories from missing members");
	Check(Common::IsSupportedArchive(archive_path) && Common::IsSupportedArchive("GAME.ZAR") &&
	          !Common::IsSupportedArchive("game.zip"),
	      "select an archive backend by supported extension");
	Check(!Common::IsArchivePath(temporary.Path() / "ordinary!" / "asset.bin"),
	      "ordinary host paths with exclamation marks stay host paths");
	Check(Common::IsArchivePath(root), "recognize virtual archive root");
	Check(Common::GetArchiveHostPath(root) == archive_path, "recover host archive path");
	Check(Common::File::IsDirectoryExisting(root), "find archive root directory");
	Check(Common::File::IsFileExisting(root / "EBOOT.BIN"), "lookup is case insensitive");
	Check(!Common::File::IsFileExisting(root / "missing.bin"), "reject missing member");
	Check(Common::IsArchivePath(root / "../outside.bin"), "retain archive classification for invalid traversal");
	Check(!Common::File::IsFileExisting(root / "../outside.bin"), "reject leading parent traversal");
	Check(Common::File::IsFileExisting(root / "assets/../eboot.bin"), "normalize internal parent traversal");
	Check(Common::File::IsFileExisting(root / ArchiveTests::LongFilename), "decode extended-length archive names");
	Check(Common::File::IsFileExisting(root / Common::PathFromUtf8(ArchiveTests::UnicodeFilename)), "lookup Unicode archive member");
	// The emulator's mount point form: generic slashes and a trailing one (kernel/fileSystem.cpp MountPoints).
	Check(Common::File::IsFileExisting(Common::PathFromUtf8(Common::PathToGenericString(root) + "/") / "sce_sys" / "param.json"),
	      "resolve a member below a mounted root");

	const auto entries = Common::File::GetDirEntries(root);
	Check(std::ranges::any_of(entries, [](const auto& entry) { return entry.is_file && entry.name == "eboot.bin"; }),
	      "enumerate root file");
	Check(std::ranges::any_of(entries, [](const auto& entry) { return !entry.is_file && entry.name == "assets"; }),
	      "enumerate root directory");
	Check(std::ranges::any_of(entries,
	                          [](const auto& entry) { return entry.is_file && entry.name == ArchiveTests::UnicodeFilename; }),
	      "enumerate Unicode member names as UTF-8");

	const auto data_path = root / "assets/subdir/data.bin";
	Check(Common::File::Size(data_path) == payload.size(), "report member size");
	Common::File file(data_path, Common::File::Mode::Read);
	Check(!file.IsInvalid(), "open archive member");
	Check(file.Remaining() == payload.size(), "report initial remaining bytes");
	Check(file.Seek(64 * 1024 - 19), "seek before compression block boundary");
	std::vector<uint8_t> slice(97);
	uint32_t             bytes_read = 0;
	file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
	Check(bytes_read == slice.size(), "read across compression block boundary");
	Check(std::equal(slice.begin(), slice.end(), payload.begin() + 64 * 1024 - 19),
	      "preserve data across compression block boundary");

	Check(file.Seek(payload.size() - 9), "seek near member end");
	file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
	Check(bytes_read == 9, "truncate read at member end");
	Check(file.IsEOF(), "report member end of file");
	Check(file.Seek(payload.size() + 100), "allow seek past member end");
	Check(file.Remaining() == 0, "report no bytes remaining past member end");
	file.Read(slice.data(), static_cast<uint32_t>(slice.size()), &bytes_read);
	Check(bytes_read == 0, "read past member end returns no data");

	Check(file.Seek(0), "seek to member start");
	std::vector<uint8_t> full(payload.size());
	file.Read(full.data(), static_cast<uint32_t>(full.size()), &bytes_read);
	Check(bytes_read == full.size() && file.Tell() == full.size(), "read a member of many chunks");
	Check(full == payload, "preserve a member of many chunks");

	// Reads that start and end inside chunks, on the lanes, and positional reads of a shared file.
	Common::ArchiveFile* shared = nullptr;
	auto                 kept   = Common::OpenArchiveFile(data_path);
	Check(kept != nullptr, "open a member to share");
	shared = kept.get();
	for (const auto& [offset, size]: {std::pair<uint64_t, uint32_t> {(1u << 20u) - 5, (9u << 20u) + 11},
	                                  {123457, 1}, {(3u << 20u) + 1, (1u << 20u) - 2}, {0, 1u << 20u}}) {
		std::vector<uint8_t> part(size);
		Check(shared->ReadAt(offset, part.data(), size) == size &&
		          std::equal(part.begin(), part.end(), payload.begin() + static_cast<std::ptrdiff_t>(offset)),
		      "read a range that starts and ends inside chunks");
	}

	std::atomic_bool         concurrent_reads_ok = true;
	std::vector<std::thread> readers;
	for (uint32_t worker = 0; worker < 8; worker++) {
		readers.emplace_back([&, worker] {
			Common::File                concurrent_file(data_path, Common::File::Mode::Read);
			std::array<uint8_t, 4096>   concurrent_data {};
			std::vector<uint8_t>        big(3u << 20u);
			for (uint32_t iteration = 0; iteration < 16; iteration++) {
				const auto offset = static_cast<size_t>(worker * 7919 + iteration * 12347) % (payload.size() - concurrent_data.size());
				uint32_t   concurrent_bytes_read = 0;
				if (concurrent_file.IsInvalid() || !concurrent_file.Seek(offset)) {
					concurrent_reads_ok = false;
					return;
				}
				concurrent_file.Read(concurrent_data.data(), concurrent_data.size(), &concurrent_bytes_read);
				if (concurrent_bytes_read != concurrent_data.size() ||
				    !std::equal(concurrent_data.begin(), concurrent_data.end(),
				                payload.begin() + static_cast<std::ptrdiff_t>(offset))) {
					concurrent_reads_ok = false;
					return;
				}
				// A read of several chunks next to the small ones, on the shared file (positional).
				const auto big_offset = (static_cast<uint64_t>(worker) * 1000003u + iteration * 777777u) % (payload.size() - big.size());
				if (shared->ReadAt(big_offset, big.data(), static_cast<uint32_t>(big.size())) != big.size() ||
				    !std::equal(big.begin(), big.end(), payload.begin() + static_cast<std::ptrdiff_t>(big_offset))) {
					concurrent_reads_ok = false;
					return;
				}
			}
		});
	}
	for (auto& reader: readers) {
		reader.join();
	}
	Check(concurrent_reads_ok, "read concurrently through independent and shared file handles");

	Common::File metadata(root / "sce_sys/param.json", Common::File::Mode::Read);
	const auto   metadata_data = ReadAll(metadata);
	const std::string metadata_text(reinterpret_cast<const char*>(metadata_data.data()), metadata_data.size());
	Check(metadata_text.find("Archive Test") != std::string::npos, "read metadata member");

	Common::File writable;
	Check(!writable.Open(data_path, Common::File::Mode::ReadWrite), "reject writable member open");
	Check(!Common::File::DeleteFile(data_path), "reject member deletion");
	Check(!Common::File::CreateDirectory(root / "new-dir"), "reject member directory creation");
	Check(!file.Truncate(0) && !file.Unlink(), "reject descriptor mutations");
	const auto   copied = temporary.Path() / "copied.bin";
	Common::File host_fixture;
	Check(host_fixture.Create(copied), "create host mutation fixture");
	host_fixture.Close();
	Check(!Common::File::CopyFile(data_path, copied) && !Common::File::CopyFile(copied, data_path) &&
	          !Common::File::RenameFile(data_path, copied) && !Common::File::RenameFile(copied, data_path),
	      "reject unsupported copy and rename operations involving archives");

	CheckMalformedArchives(archive_path, payload);

	file.Close();
	metadata.Close();
	kept.reset();
	Check(Common::OpenArchive(archive_path) == reader_pin, "mount pin retains the archive reader after the last member closes");
	reader_pin.reset();
	std::error_code remove_error;
	Check(std::filesystem::remove(archive_path, remove_error) && !remove_error,
	      "release archive handles after the final reader closes");
	const std::array<uint8_t, 3> replacement {42, 7, 19};
	Check(ArchiveTests::CreateArchive(archive_path, replacement), "replace a closed archive");
	Common::File replaced(data_path, Common::File::Mode::Read);
	const auto   replaced_data = ReadAll(replaced);
	Check(replaced_data.size() == replacement.size() &&
	          std::memcmp(replaced_data.data(), replacement.data(), replacement.size()) == 0,
	      "reopening after the final reader closes sees replacement contents");

	std::printf("ArchiveFileTests: all cases passed\n");
	return 0;
}
