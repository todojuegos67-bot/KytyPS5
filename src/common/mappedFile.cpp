#include "common/mappedFile.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace Common {

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

bool MappedFile::Create(const std::filesystem::path& name, uint64_t size) {
	Close(0);
	if (size == 0) {
		return false;
	}
	HANDLE file = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
	                          nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		return false;
	}
	HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32u),
	                                    static_cast<DWORD>(size & 0xffffffffu), nullptr);
	void*  data    = mapping != nullptr ? MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, size) : nullptr;
	if (mapping != nullptr) {
		CloseHandle(mapping); // the view keeps the mapping until it is unmapped
	}
	if (data == nullptr) {
		CloseHandle(file);
		return false;
	}
	m_file = reinterpret_cast<intptr_t>(file);
	m_data = static_cast<uint8_t*>(data);
	m_size = size;
	return true;
}

bool MappedFile::Close(uint64_t final_size) {
	if (m_data == nullptr) {
		return false;
	}
	auto* file = reinterpret_cast<HANDLE>(m_file);
	bool  ok   = FlushViewOfFile(m_data, 0) != 0;
	ok         = UnmapViewOfFile(m_data) != 0 && ok;
	if (final_size < m_size) {
		LARGE_INTEGER end {};
		end.QuadPart = static_cast<LONGLONG>(final_size);
		ok           = SetFilePointerEx(file, end, nullptr, FILE_BEGIN) != 0 && SetEndOfFile(file) != 0 && ok;
	}
	ok     = FlushFileBuffers(file) != 0 && ok;
	ok     = CloseHandle(file) != 0 && ok;
	m_file = 0;
	m_data = nullptr;
	m_size = 0;
	return ok;
}

#else

bool MappedFile::Create(const std::filesystem::path& name, uint64_t size) {
	Close(0);
	if (size == 0) {
		return false;
	}
	const int file = open(name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (file < 0) {
		return false;
	}
	void* data = ftruncate(file, static_cast<off_t>(size)) == 0
	                 ? mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0)
	                 : MAP_FAILED;
	if (data == MAP_FAILED) {
		close(file);
		return false;
	}
	m_file = file;
	m_data = static_cast<uint8_t*>(data);
	m_size = size;
	return true;
}

bool MappedFile::Close(uint64_t final_size) {
	if (m_data == nullptr) {
		return false;
	}
	const int file = static_cast<int>(m_file);
	bool      ok   = msync(m_data, m_size, MS_SYNC) == 0;
	ok             = munmap(m_data, m_size) == 0 && ok;
	if (final_size < m_size) {
		ok = ftruncate(file, static_cast<off_t>(final_size)) == 0 && ok;
	}
	ok     = fsync(file) == 0 && ok;
	ok     = close(file) == 0 && ok;
	m_file = 0;
	m_data = nullptr;
	m_size = 0;
	return ok;
}

#endif

} // namespace Common
