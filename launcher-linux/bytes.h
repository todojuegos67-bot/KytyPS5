#ifndef KYTY_LAUNCHER_LINUX_BYTES_H
#define KYTY_LAUNCHER_LINUX_BYTES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Hashes and gzip for the keyboard and mouse prompts (launcher-windows' KeyboardPrompts.cs uses
// .NET's SHA256 and GZipStream): no library (the inflate is a small one of its own).
namespace Bytes {

using Buffer = std::vector<uint8_t>;

// SHA-256 as 64 lower-case hex digits.
std::string Sha256Hex(const uint8_t* data, size_t size);
inline std::string Sha256Hex(const Buffer& data) {
	return Sha256Hex(data.data(), data.size());
}

uint32_t Crc32(const uint8_t* data, size_t size);

// The bytes of a gzip file (one member; its CRC and length are checked); empty `error` on success.
Buffer Gunzip(const uint8_t* data, size_t size, std::string* error);
inline Buffer Gunzip(const Buffer& data, std::string* error) {
	return Gunzip(data.data(), data.size(), error);
}

// A gzip file holding `data` in stored (uncompressed) deflate blocks: any gzip reader, .NET's
// GZipStream included, reads it. The files it makes are small (a prompt texture is 128 KiB).
Buffer GzipStored(const Buffer& data);

} // namespace Bytes

#endif
