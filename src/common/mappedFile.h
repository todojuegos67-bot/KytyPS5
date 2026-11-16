#ifndef KYTY_COMMON_MAPPEDFILE_H_
#define KYTY_COMMON_MAPPEDFILE_H_

#include "common/common.h"

#include <cstdint>
#include <filesystem>

namespace Common {

// A new file of a fixed size, written through a shared mapping. Its pages belong to the file, so they are not charged
// to the commit limit as the same bytes on the heap would be.
class MappedFile final {
public:
	MappedFile() = default;
	~MappedFile() { Close(0); }
	KYTY_CLASS_NO_COPY(MappedFile);

	// Creates or replaces the file with size bytes and maps it. An open file is closed empty first.
	bool Create(const std::filesystem::path& name, uint64_t size);
	// Flushes, unmaps, cuts the file to final_size bytes and closes it. False when not open or when a step failed.
	bool Close(uint64_t final_size);

	[[nodiscard]] uint8_t* Data() const { return m_data; }
	[[nodiscard]] uint64_t Size() const { return m_size; }

private:
	intptr_t m_file = 0;
	uint8_t* m_data = nullptr;
	uint64_t m_size = 0;
};

} // namespace Common

#endif /* KYTY_COMMON_MAPPEDFILE_H_ */
