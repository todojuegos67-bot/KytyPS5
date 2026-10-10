#include "common/archive.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <zarchive/zarchivereader.h>

// POSIX hosts and MinGW's pthread-backed libstdc++ have cancellable condition waits.
// MSVC/ClangCL and MinGW's native Win32 threading backend do not use pthread waits.
#if !defined(_WIN32) || defined(_GLIBCXX_GCC_GTHR_POSIX_H)
#include <pthread.h>
#define KYTY_ARCHIVE_PTHREAD_CANCELLATION
#endif

namespace Common {

// Backends expose immutable metadata and offset-based reads. Cursor management, path routing,
// reader lifetime and the host-stack worker are shared by every archive format.
class ArchiveReader {
public:
	struct Entry {
		uint64_t id;
		uint64_t size;
		bool     is_file;
	};

	// Reads on different lanes run in parallel (each lane its own stream and block cache), those on one lane in turn.
	static constexpr uint32_t Lanes = 4;

	virtual ~ArchiveReader()                                          = default;
	virtual std::optional<Entry>        Find(std::string_view member) = 0;
	virtual std::vector<File::DirEntry> List(std::string_view member) = 0;
	virtual uint64_t Read(uint32_t lane, uint64_t id, uint64_t offset, uint32_t size, void* data) = 0;
};

namespace {

class ZArchiveReaderBackend final: public ArchiveReader {
public:
	ZArchiveReaderBackend(std::filesystem::path path, std::unique_ptr<ZArchiveReader> reader)
	    : m_path(std::move(path)), m_reader(std::move(reader)) {}

	std::optional<Entry> Find(std::string_view member) override {
		const auto node = m_reader->LookUp(member);
		if (node == ZARCHIVE_INVALID_NODE) {
			return {};
		}
		return Entry {node, m_reader->GetFileSize(node), m_reader->IsFile(node)};
	}

	std::vector<File::DirEntry> List(std::string_view member) override {
		const auto node = m_reader->LookUp(member);
		if (!m_reader->IsDirectory(node)) {
			return {};
		}
		const auto count = m_reader->GetDirEntryCount(node);
		if (count > (1u << 20u)) {
			return {};
		}
		std::vector<File::DirEntry> result;
		result.reserve(count);
		for (uint32_t index = 0; index < count; ++index) {
			ZArchiveReader::DirEntry entry {};
			if (m_reader->GetDirEntry(node, index, entry)) {
				result.push_back({std::string(entry.name), entry.isFile});
			}
		}
		return result;
	}

	uint64_t Read(uint32_t lane, uint64_t id, uint64_t offset, uint32_t size, void* data) override {
		// ZArchiveReader serializes the calls on its stream and block cache: the lanes after the first read with
		// readers of their own (opened on a lane's first read; the first one's if that fails).
		auto* reader = m_reader.get();
		if (lane != 0 && lane < Lanes) {
			std::call_once(m_lane_opened[lane], [&] {
				m_lane_readers[lane].reset(ZArchiveReader::OpenFromFile(m_path));
			});
			if (m_lane_readers[lane] != nullptr) {
				reader = m_lane_readers[lane].get();
			}
		}
		return reader->ReadFromFile(static_cast<ZArchiveNodeHandle>(id), offset, size, data);
	}

private:
	std::filesystem::path           m_path;
	std::unique_ptr<ZArchiveReader> m_reader; // lane 0's, and the look-ups'
	std::unique_ptr<ZArchiveReader> m_lane_readers[Lanes];
	std::once_flag                  m_lane_opened[Lanes];
};

std::shared_ptr<ArchiveReader> OpenZArchive(const std::filesystem::path& path) {
	auto reader = std::unique_ptr<ZArchiveReader>(ZArchiveReader::OpenFromFile(path));
	return reader ? std::make_shared<ZArchiveReaderBackend>(path, std::move(reader)) : nullptr;
}

struct ArchiveFormat {
	std::string_view extension;
	std::shared_ptr<ArchiveReader> (*open)(const std::filesystem::path&);
};

constexpr ArchiveFormat Formats[] = {{".zar", OpenZArchive}};
using NativeView                  = std::basic_string_view<std::filesystem::path::value_type>;

const ArchiveFormat* FindFormat(NativeView path) {
	for (const auto& format: Formats) {
		if (path.size() < format.extension.size()) {
			continue;
		}
		const auto extension = path.substr(path.size() - format.extension.size());
		if (std::equal(
		        extension.begin(), extension.end(), format.extension.begin(),
		        [](auto a, char b) { return (a >= 'A' && a <= 'Z' ? a + ('a' - 'A') : a) == b; })) {
			return &format;
		}
	}
	return nullptr;
}

size_t FindMarker(NativeView path) {
	// Native files use this check on every operation. The usual path has no marker and allocates
	// nothing.
	for (auto marker = path.find('!'); marker != NativeView::npos;
	     marker      = path.find('!', marker + 1)) {
		if ((marker + 1 == path.size() || path[marker + 1] == '/' || path[marker + 1] == '\\') &&
		    FindFormat(path.substr(0, marker)) != nullptr) {
			return marker;
		}
	}
	return NativeView::npos;
}

struct ParsedPath {
	std::filesystem::path archive;
	std::string           member;
};

std::optional<ParsedPath> ParsePath(const std::filesystem::path& path) {
	const auto& native = path.native();
	const auto  marker = FindMarker(native);
	if (marker == NativeView::npos) {
		return {};
	}
	ParsedPath parsed {.archive = std::filesystem::path(native.substr(0, marker))};
	auto       member = PathToGenericString(std::filesystem::path(native.substr(marker + 1)));
	std::replace(member.begin(), member.end(), '\\', '/');
	std::string_view remaining = member;
	while (!remaining.empty()) {
		const auto separator = remaining.find('/');
		const auto component = remaining.substr(0, separator);
		remaining.remove_prefix(separator == std::string_view::npos ? remaining.size() : separator + 1);
		if (component.empty() || component == ".") {
			continue;
		}
		if (component == "..") {
			if (parsed.member.empty()) {
				return {};
			}
			const auto separator = parsed.member.rfind('/');
			parsed.member.resize(separator == std::string::npos ? 0 : separator);
		} else {
			if (!parsed.member.empty()) {
				parsed.member += '/';
			}
			parsed.member += component;
		}
	}
	return parsed;
}

std::mutex                                                              g_readers_mutex;
std::unordered_map<std::filesystem::path, std::weak_ptr<ArchiveReader>> g_readers;

// Guest stacks can be only 64 KiB. Decompress on ordinary host stacks, then copy into guest memory on the caller so
// GPU-tracked writes use its fault handler. A thread a lane (ArchiveReader::Lanes): a read is split into chunks, which
// decompress on the lanes in parallel (one lane decompressed about 1 GB/s; a 320 MiB read took 323 ms, and
// smaller reads waited behind it); a chunk's lane follows its place in the file, so small reads of one region
// meet the block cache of the lane that read it before. Requests live on waiting callers.
class ArchiveIo {
public:
	static ArchiveIo& Instance() {
		// Guest exit can run static destructors while other guest threads are still reading.
		// Keep the workers alive until process termination, as with the guest threads themselves.
		static auto* instance = new ArchiveIo;
		return *instance;
	}

	// Reads [offset, offset + size) of the member into `data` (copied on this thread): the bytes read.
	uint32_t Read(ArchiveReader& reader, uint64_t id, uint64_t offset, uint32_t size, uint8_t* data) {
		constexpr uint32_t                chunk_size = 1u << 20u;
		constexpr uint32_t                window     = ArchiveReader::Lanes * 2; // chunks in flight
		thread_local std::vector<uint8_t> buffers;
		const uint32_t                    chunks = static_cast<uint32_t>((uint64_t {size} + chunk_size - 1) / chunk_size);
		const uint32_t                    slots  = std::min(window, chunks);
		const size_t                      needed = slots <= 1 ? size : size_t {slots} * chunk_size;
		if (buffers.size() < needed) {
			buffers.resize(needed);
		}
		Batch    batch;
		Request  requests[window];
		uint32_t submitted = 0;
		const auto submit  = [&](uint32_t index) {
			auto& request  = requests[index % window];
			request.reader = &reader;
			request.batch  = &batch;
			request.id     = id;
			request.offset = offset + uint64_t {index} * chunk_size;
			request.size   = std::min(chunk_size, size - index * chunk_size);
			request.data   = buffers.data() + size_t {index % window} * chunk_size;
			request.next   = nullptr;
			request.result = 0;
			request.done   = false;
			Submit(LaneOf(id, request.offset), &request);
		};
		while (submitted < slots) {
			submit(submitted++);
		}
		uint32_t read  = 0;
		bool     ended = false;
		for (uint32_t index = 0; index < submitted; index++) {
			auto& request = requests[index % window];
			{
				std::unique_lock lock(batch.mutex);
				batch.ready.wait(lock, [&] { return request.done; });
			}
			if (!ended) {
				const auto got = std::min<uint64_t>(request.result, request.size);
				std::memcpy(data + read, request.data, got);
				read += static_cast<uint32_t>(got);
				// A short chunk ends the read (the file's end, or a block that cannot be read).
				ended = got < request.size;
			}
			if (!ended && submitted < chunks) {
				submit(submitted++);
			}
		}
		return read;
	}

	KYTY_CLASS_NO_COPY(ArchiveIo);

private:
	struct Batch {
		std::mutex              mutex;
		std::condition_variable ready;
	};

	struct Request {
		ArchiveReader* reader = nullptr;
		Batch*         batch  = nullptr;
		uint64_t       id     = 0;
		uint64_t       offset = 0;
		uint32_t       size   = 0;
		uint8_t*       data   = nullptr;
		Request*       next   = nullptr;
		uint64_t       result = 0;
		bool           done   = false;
	};

	struct Lane {
		std::mutex              mutex;
		std::condition_variable wake;
		Request*                first = nullptr;
		Request*                last  = nullptr;
	};

	ArchiveIo() {
		for (uint32_t lane = 0; lane < ArchiveReader::Lanes; lane++) {
			std::thread([this, lane] { Loop(lane); }).detach();
		}
	}

	// The lane of a member's 1 MiB chunk: a file's consecutive chunks on consecutive lanes.
	static uint32_t LaneOf(uint64_t id, uint64_t offset) {
		return static_cast<uint32_t>(((id * 0x9E3779B97F4A7C15ull) >> 40u) + (offset >> 20u)) % ArchiveReader::Lanes;
	}

	void Submit(uint32_t lane, Request* request) {
		auto&           queue = m_lanes[lane];
		std::lock_guard lock(queue.mutex);
		if (queue.last != nullptr) {
			queue.last->next = request;
		} else {
			queue.first = request;
		}
		queue.last = request;
		queue.wake.notify_one();
	}

	void Loop(uint32_t lane) {
		auto& queue = m_lanes[lane];
		for (;;) {
			Request* request;
			{
				std::unique_lock lock(queue.mutex);
				queue.wake.wait(lock, [&] { return queue.first != nullptr; });
				request     = queue.first;
				queue.first = request->next;
				if (queue.first == nullptr) {
					queue.last = nullptr;
				}
			}
			const auto result = request->reader->Read(lane, request->id, request->offset, request->size, request->data);
			auto*      batch  = request->batch;
			std::lock_guard lock(batch->mutex);
			request->result = result;
			request->done   = true;
			// Notify before releasing the mutex: the waiting caller owns the condition variable.
			batch->ready.notify_all();
		}
	}

	Lane m_lanes[ArchiveReader::Lanes];
};

} // namespace

struct ArchiveFile::Private {
	std::shared_ptr<ArchiveReader> reader;
	ArchiveReader::Entry           entry;
	uint64_t                       position = 0;
};

ArchiveFile::ArchiveFile(std::unique_ptr<Private> p): m_p(std::move(p)) {}
ArchiveFile::~ArchiveFile() = default;

uint64_t ArchiveFile::Size() const {
	return m_p->entry.size;
}
uint64_t ArchiveFile::Tell() const {
	return m_p->position;
}
bool ArchiveFile::Seek(uint64_t offset) {
	m_p->position = offset;
	return true;
}

void ArchiveFile::Read(void* data, uint32_t size, uint32_t* bytes_read) {
	const auto read = ReadAt(m_p->position, data, size);
	m_p->position += read;
	if (bytes_read != nullptr) {
		*bytes_read = read;
	}
}

uint32_t ArchiveFile::ReadAt(uint64_t offset, void* data, uint32_t size) const {
#if defined(KYTY_ARCHIVE_PTHREAD_CANCELLATION)
	// A pending cancellation must not destroy the waiting request or its TLS buffer while the
	// worker still uses them. Restore the caller's state only after the copy is complete.
	int cancellation_state = PTHREAD_CANCEL_ENABLE;
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &cancellation_state);
#endif
	uint32_t read = 0;
	if (data != nullptr && size != 0 && offset < Size()) {
		const auto requested = static_cast<uint32_t>(std::min<uint64_t>(size, Size() - offset));
		read = ArchiveIo::Instance().Read(*m_p->reader, m_p->entry.id, offset, requested, static_cast<uint8_t*>(data));
	}
#if defined(KYTY_ARCHIVE_PTHREAD_CANCELLATION)
	pthread_setcancelstate(cancellation_state, nullptr);
#endif
	return read;
}

std::filesystem::path MakeArchivePath(const std::filesystem::path& archive,
                                      const std::filesystem::path& member) {
	auto root = archive;
	root += "!";
	return member.empty() ? root : root / member.relative_path();
}

bool IsSupportedArchive(const std::filesystem::path& path) {
	return FindFormat(path.native()) != nullptr;
}

bool IsArchivePath(const std::filesystem::path& path) {
	return FindMarker(path.native()) != NativeView::npos;
}

std::filesystem::path GetArchiveHostPath(const std::filesystem::path& path) {
	const auto marker = FindMarker(path.native());
	return marker == NativeView::npos ? std::filesystem::path {}
	                                  : std::filesystem::path(path.native().substr(0, marker));
}

std::shared_ptr<ArchiveReader> OpenArchive(const std::filesystem::path& path) {
	const auto  archive = IsArchivePath(path) ? GetArchiveHostPath(path) : path;
	const auto* format  = FindFormat(archive.native());
	if (format == nullptr) {
		return {};
	}
	std::error_code error;
	auto            key = std::filesystem::absolute(archive, error);
	if (error) {
		return {};
	}
	{
		std::lock_guard lock(g_readers_mutex);
		if (auto it = g_readers.find(key); it != g_readers.end()) {
			if (auto reader = it->second.lock()) {
				return reader;
			}
		}
	}
	// Do not hold the cache mutex while opening an unrelated archive from disk.
	auto reader = format->open(archive);
	if (reader == nullptr) {
		return {};
	}
	std::lock_guard lock(g_readers_mutex);
	std::erase_if(g_readers, [](const auto& entry) { return entry.second.expired(); });
	auto& cached = g_readers[key];
	if (auto existing = cached.lock()) {
		return existing;
	}
	cached = reader;
	return reader;
}

std::optional<File::Info> GetArchiveInfo(const std::filesystem::path& path) {
	const auto parsed = ParsePath(path);
	if (parsed) {
		if (auto reader = OpenArchive(parsed->archive)) {
			if (auto entry = reader->Find(parsed->member)) {
				return File::Info {entry->is_file, entry->size};
			}
		}
	}
	return {};
}

std::vector<File::DirEntry> GetArchiveDirEntries(const std::filesystem::path& path) {
	const auto parsed = ParsePath(path);
	if (parsed) {
		if (auto reader = OpenArchive(parsed->archive)) {
			return reader->List(parsed->member);
		}
	}
	return {};
}

std::unique_ptr<ArchiveFile> OpenArchiveFile(const std::filesystem::path& path) {
	auto parsed = ParsePath(path);
	if (!parsed) {
		return {};
	}
	auto       reader = OpenArchive(parsed->archive);
	const auto entry  = reader ? reader->Find(parsed->member) : std::nullopt;
	if (!entry || !entry->is_file) {
		return {};
	}
	auto p     = std::make_unique<ArchiveFile::Private>();
	p->reader  = std::move(reader);
	p->entry   = *entry;
	return std::unique_ptr<ArchiveFile>(new ArchiveFile(std::move(p)));
}

} // namespace Common
