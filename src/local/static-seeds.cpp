// kyty_shader_precompile --make-seeds: the static precompile's seed file, made from the game's files. This is
// the port of `tools/local/static-precompile/precompile.py seeds` (with what it uses of agc.py, keys.py,
// materials.py and warmfile.py), which needs Python 3 with numpy, and it writes the same file byte for byte:
// the same records and pipeline recipes, deduplicated and ordered the same way, under the same identity line
// and checksum (xxh3.py is XXH3_64bits). Where precompile.py stops with an exception (a game file it does not
// expect) this stops too (Fatal: nothing is written); where it skips what it was reading (a ValueError it
// catches) this skips it too (Refused). The Python tools stay the reference and do the analyses.
#include "static-seeds.h"

#include <nlohmann/json.hpp>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace StaticSeeds {

namespace {

namespace fs = std::filesystem;
using Bytes  = std::span<const uint8_t>;

static_assert(std::endian::native == std::endian::little, "the game's files and the seed file are little endian");

// Where precompile.py stops with an exception (a file it does not expect): no seed file.
struct Fatal: std::runtime_error {
	using std::runtime_error::runtime_error;
};

// A ValueError precompile.py catches (agc.AgcError and the like): what was being read is skipped.
struct Refused: std::runtime_error {
	using std::runtime_error::runtime_error;
};

constexpr uint32_t NoShader = 0xffffffffu;
// warmfile.ST_*: the stage of a record.
constexpr uint32_t StageVertex = 1, StagePixel = 2, StageCompute = 4;
// Prospero::ShaderBinaryType of an AGC header.
constexpr uint8_t TypeCs = 0, TypePs = 1, TypeGs = 2;
// The host GPU as precompile.py assumes it: no wave64 compute, fragment subgroup reduction.
constexpr uint32_t HostSubgroupSize = 32, LodStatsSubgroup = 1;
// precompile.py SEED_IDENTITY: Demon's Souls' title, whatever the game.
constexpr std::string_view SeedIdentity = "KytyShaderSeeds2:PPSA01341\n";

#ifdef _WIN32
// pathlib's Windows paths: '\' separates too, and paths compare, hash and match patterns case-insensitively.
constexpr bool WindowsPaths = true;
#else
constexpr bool WindowsPaths = false;
#endif

// ---------------------------------------------------------------------------------------------------------
// Reading, with Python's struct and slice semantics
// ---------------------------------------------------------------------------------------------------------

// The `size` bytes at `offset` exist: struct.unpack_from fails past the end (struct.error).
void Need(Bytes data, uint64_t offset, uint64_t size) {
	if (offset > data.size() || data.size() - offset < size) {
		throw Fatal("the data ends inside a field at offset " + std::to_string(offset));
	}
}

// A little-endian field.
template <class T>
T Field(Bytes data, uint64_t offset) {
	Need(data, offset, sizeof(T));
	T value {};
	std::memcpy(&value, data.data() + offset, sizeof(T));
	return value;
}

// Sums Python's integers make without overflowing: beyond 64 bits only in a malformed eboot.bin.
uint64_t Add(uint64_t a, uint64_t b) {
	if (a > UINT64_MAX - b) throw Fatal("an offset beyond 64 bits");
	return a + b;
}

int64_t AddSigned(int64_t a, int64_t b) {
	if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) throw Fatal("an address beyond 64 bits");
	return a + b;
}

int64_t Signed(uint64_t value) {
	if (value > static_cast<uint64_t>(INT64_MAX)) throw Fatal("an address beyond 63 bits");
	return static_cast<int64_t>(value);
}

// A slice bound: Python clamps it to the data, so saturating is the same.
uint64_t SaturatingAdd(uint64_t a, uint64_t b) {
	return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

// data[start:stop].
Bytes Slice(Bytes data, uint64_t start, uint64_t stop) {
	start = std::min<uint64_t>(start, data.size());
	stop  = std::clamp<uint64_t>(stop, start, data.size());
	return data.subspan(start, stop - start);
}

// data[start:stop] with Python's negative bounds, counted from the end.
Bytes SliceSigned(Bytes data, int64_t start, int64_t stop) {
	const auto size  = static_cast<int64_t>(data.size());
	const auto bound = [&](int64_t i) { return i < 0 ? std::max<int64_t>(i + size, 0) : std::min(i, size); };
	const auto begin = bound(start);
	const auto end   = std::max(bound(stop), begin);
	return data.subspan(static_cast<size_t>(begin), static_cast<size_t>(end - begin));
}

// image[start:stop] = value, as a bytearray takes it: a value of another length resizes the image.
void Assign(std::vector<uint8_t>& image, uint64_t start, uint64_t stop, Bytes value) {
	start = std::min<uint64_t>(start, image.size());
	stop  = std::clamp<uint64_t>(stop, start, image.size());
	if (stop - start == value.size()) {
		std::copy(value.begin(), value.end(), image.begin() + static_cast<std::ptrdiff_t>(start));
		return;
	}
	image.erase(image.begin() + static_cast<std::ptrdiff_t>(start), image.begin() + static_cast<std::ptrdiff_t>(stop));
	image.insert(image.begin() + static_cast<std::ptrdiff_t>(start), value.begin(), value.end());
}

bool StartsWith(Bytes data, Bytes prefix) {
	return data.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), data.begin());
}

std::string Utf8(const fs::path& path) {
	const auto text = path.u8string();
	return {reinterpret_cast<const char*>(text.data()), text.size()};
}

// Path.read_bytes(): a file it cannot read stops precompile.py.
std::vector<uint8_t> ReadAll(const fs::path& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) throw Fatal("cannot read " + Utf8(path));
	const auto size = static_cast<std::streamoff>(file.tellg());
	if (size < 0) throw Fatal("cannot read " + Utf8(path));
	std::vector<uint8_t> data(static_cast<size_t>(size));
	file.seekg(0);
	if (size != 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) throw Fatal("cannot read " + Utf8(path));
	return data;
}

// str.lower() of the ASCII letters.
std::string Lower(std::string_view text) {
	std::string out(text);
	for (auto& c: out) {
		if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
	}
	return out;
}

// A path component as pathlib compares it.
std::string Compared(std::string_view part) {
	return WindowsPaths ? Lower(part) : std::string(part);
}

// The key a path below the game directory compares by (pathlib's equality).
std::string PathKey(const std::vector<std::string>& parts) {
	std::string key;
	for (const auto& part: parts) {
		key += Compared(part);
		key += '/';
	}
	return key;
}

// A file below the game directory: its path, its components there, and those as pathlib compares them.
struct GameFile {
	fs::path                 path;
	std::vector<std::string> parts;
	std::vector<std::string> compared;
};

// sorted(Path(game).rglob('*' + suffix)): the entries named *suffix (case-insensitively on Windows) of the game
// directory and of its subdirectories that are not symbolic links (one that cannot be listed is left out), in
// the order pathlib sorts paths: component by component.
std::vector<GameFile> FindGameFiles(const fs::path& game, std::string_view suffix) {
	std::vector<GameFile> found;
	std::vector<GameFile> directories {GameFile {game, {}, {}}};
	while (!directories.empty()) {
		const auto directory = std::move(directories.back());
		directories.pop_back();
		std::vector<fs::directory_entry> entries;
		std::error_code                  error;
		for (fs::directory_iterator it(directory.path, error), end; !error && it != end; it.increment(error)) {
			entries.push_back(*it);
		}
		if (error) continue;
		for (const auto& entry: entries) {
			GameFile   file {entry.path(), directory.parts, directory.compared};
			const auto name = Utf8(entry.path().filename());
			file.parts.push_back(name);
			file.compared.push_back(Compared(name));
			const auto ending = std::string_view(name).substr(name.size() - std::min(name.size(), suffix.size()));
			if (name.size() >= suffix.size() && Compared(ending) == suffix) found.push_back(file);
			std::error_code status;
			if (!entry.is_symlink(status) && entry.is_directory(status)) directories.push_back(std::move(file));
		}
	}
	std::stable_sort(found.begin(), found.end(),
	                 [](const GameFile& a, const GameFile& b) { return a.compared < b.compared; });
	return found;
}

// ---------------------------------------------------------------------------------------------------------
// agc.py: AGC headers and CSDR bundles
// ---------------------------------------------------------------------------------------------------------

constexpr uint8_t AgcMagic[]     = {'1', '2', '3', '4'};
constexpr uint8_t BundleFooter[] = {'R', 'D', 'S', 'C'};

// An AGC header as agc.parse_agc reads it: every field, compared as its dicts compare.
struct Agc {
	uint32_t                                   header_size = 0, shader_size = 0, num_input_semantics = 0;
	uint16_t                                   scratch_size_dw_per_thread = 0, num_output_semantics = 0;
	uint16_t                                   special_sizes_bytes = 0;
	uint8_t                                    type = 0, num_cx_registers = 0, num_sh_registers = 0;
	std::vector<std::pair<uint32_t, uint32_t>> cx_registers, sh_registers; // (offset, value)
	bool                                       has_specials = false;
	std::array<uint32_t, 2>                    vgt_shader_stages_en {};
	uint32_t                                   dispatch_modifier = 0;
	std::array<uint16_t, 2>                    user_data_range {};
	std::vector<uint32_t>                      input_semantics, output_semantics; // the semantic words
	bool                                       has_user_data = false;
	uint16_t                                   srt_size_dw = 0, direct_resource_count = 0;
	std::vector<uint16_t>                      direct_resource_offset;

	bool operator==(const Agc&) const = default;
};

// agc._rel_ptr: the 64-bit self-relative pointer at `field` (0: null) to `size` bytes inside the header.
std::optional<uint64_t> RelPtr(Bytes h, uint64_t field, uint64_t size, const char* what) {
	const auto rel = Field<int64_t>(h, field);
	if (rel == 0) return std::nullopt;
	// field + rel as Python computes it (the field lies inside the header).
	const bool    beyond = rel > 0 && static_cast<uint64_t>(rel) > static_cast<uint64_t>(INT64_MAX) - field;
	const int64_t ptr    = beyond ? INT64_MAX : static_cast<int64_t>(field) + rel;
	if (beyond || ptr < 96 || static_cast<uint64_t>(ptr) > h.size() || size > h.size() - static_cast<uint64_t>(ptr)) {
		throw Refused(std::string(what) + ": pointer " + std::to_string(ptr) + " (+" + std::to_string(size) +
		              ") outside a " + std::to_string(h.size()) + "-byte header");
	}
	return static_cast<uint64_t>(ptr);
}

// agc.parse_agc: one AGC header blob (96 bytes and the tables its pointers lead to).
Agc ParseAgc(Bytes h) {
	if (h.size() < 96) throw Refused("header shorter than 96 bytes");
	if (Field<uint32_t>(h, 0) != 0x34333231u || Field<uint32_t>(h, 4) != 0x18u) throw Refused("bad magic or version");
	Agc a;
	a.header_size                = Field<uint32_t>(h, 64);
	a.shader_size                = Field<uint32_t>(h, 68);
	a.num_input_semantics        = Field<uint32_t>(h, 80);
	a.scratch_size_dw_per_thread = Field<uint16_t>(h, 84);
	a.num_output_semantics       = Field<uint16_t>(h, 86);
	a.special_sizes_bytes        = Field<uint16_t>(h, 88);
	a.type                       = h[90];
	a.num_cx_registers           = h[91];
	a.num_sh_registers           = h[92];
	if (a.header_size != h.size()) {
		throw Refused("header_size " + std::to_string(a.header_size) + " != blob length " + std::to_string(h.size()));
	}
	const auto registers = [&](uint64_t field, uint32_t count, const char* what) {
		std::vector<std::pair<uint32_t, uint32_t>> out;
		const auto                                 at = RelPtr(h, field, uint64_t {count} * 8, what);
		if (!at) {
			if (count != 0) throw Refused(std::string(what) + ": null with " + std::to_string(count) + " registers");
			return out;
		}
		for (uint64_t i = 0; i < count; i++) {
			out.emplace_back(Field<uint32_t>(h, *at + 8 * i), Field<uint32_t>(h, *at + 8 * i + 4));
		}
		return out;
	};
	a.cx_registers = registers(24, a.num_cx_registers, "cx_registers");
	a.sh_registers = registers(32, a.num_sh_registers, "sh_registers");
	if (const auto at = RelPtr(h, 40, a.special_sizes_bytes, "specials"); at && a.special_sizes_bytes >= 48) {
		const auto raw         = h.subspan(*at, a.special_sizes_bytes);
		a.has_specials         = true;
		a.vgt_shader_stages_en = {Field<uint32_t>(raw, 8), Field<uint32_t>(raw, 12)};
		a.dispatch_modifier    = Field<uint32_t>(raw, 16);
		a.user_data_range      = {Field<uint16_t>(raw, 20), Field<uint16_t>(raw, 22)};
	}
	const auto semantics = [&](uint64_t field, uint32_t count, const char* what) {
		std::vector<uint32_t> out;
		if (const auto at = RelPtr(h, field, uint64_t {count} * 4, what)) {
			for (uint64_t i = 0; i < count; i++) out.push_back(Field<uint32_t>(h, *at + 4 * i));
		}
		return out;
	};
	a.input_semantics  = semantics(48, a.num_input_semantics, "input_semantics");
	a.output_semantics = semantics(56, a.num_output_semantics, "output_semantics");
	if (const auto at = RelPtr(h, 8, 56, "user_data")) {
		const auto count        = Field<uint16_t>(h, *at + 44);
		const auto offsets      = RelPtr(h, *at, uint64_t {count} * 2, "direct_resource_offset");
		a.has_user_data         = true;
		a.srt_size_dw           = Field<uint16_t>(h, *at + 42);
		a.direct_resource_count = count;
		for (uint64_t i = 0; offsets && i < count; i++) {
			a.direct_resource_offset.push_back(Field<uint16_t>(h, *offsets + 2 * i));
		}
		if (count != 0 && !offsets) throw Refused("direct_resource_offset null with a nonzero count");
	}
	return a;
}

// A shader of a bundle or of eboot.bin: its code (a view of the file's bytes) and its parsed header.
struct Shader {
	Bytes code;
	Agc   agc;
};

// agc.parse_bundle: the (code, header) of each entry of a CSDR bundle.
std::vector<std::pair<Bytes, Bytes>> ParseBundle(Bytes data) {
	const uint64_t size = data.size();
	if (size < 20) throw Refused("truncated bundle header");
	if (!StartsWith(data.subspan(size - 12), BundleFooter) || Field<uint32_t>(data, size - 4) != size - 12) {
		throw Refused("invalid CSDR footer or payload length");
	}
	const uint64_t count = Field<uint32_t>(data, 4);
	if (count == 0 || count > 16384 || 8 + count * 20 > size) throw Refused("unsupported or truncated entry table");
	const uint64_t table_end = 8 + count * 20;
	uint64_t       cursor    = (table_end + 7) & ~uint64_t {7};
	for (const auto byte: Slice(data, table_end, cursor)) {
		if (byte != 0xff) throw Refused("invalid table alignment padding");
	}
	std::vector<std::pair<Bytes, Bytes>> entries;
	for (uint64_t i = 0; i < count; i++) {
		// {u32 stage, u32 flags, u32 code bytes, u32 header bytes, u32 code offset from this field}
		const uint64_t entry        = 8 + i * 20;
		const uint64_t code_bytes   = Field<uint32_t>(data, entry + 8);
		const uint64_t header_bytes = Field<uint32_t>(data, entry + 12);
		const uint64_t begin        = entry + 16 + Field<uint32_t>(data, entry + 16);
		const uint64_t header       = begin + code_bytes;
		const uint64_t end          = header + header_bytes;
		const auto     name         = "entry " + std::to_string(i) + ": ";
		if (begin != cursor || code_bytes == 0 || code_bytes % 4 != 0 || header_bytes < 96 || end > size - 12) {
			throw Refused(name + "non-contiguous or out-of-bounds payload");
		}
		if (!StartsWith(data.subspan(header), AgcMagic)) throw Refused(name + "missing AGC header");
		if (Field<uint32_t>(data, header + 68) != code_bytes || Field<uint32_t>(data, header + 64) != header_bytes ||
		    data[header + 90] > 8) {
			throw Refused(name + "inconsistent AGC sizes or type");
		}
		entries.emplace_back(data.subspan(begin, code_bytes), data.subspan(header, header_bytes));
		cursor = end;
	}
	if (cursor != size - 12) throw Refused("unrecognized trailing bundle data");
	return entries;
}

// ---------------------------------------------------------------------------------------------------------
// agc.py: the shaders embedded in eboot.bin
// ---------------------------------------------------------------------------------------------------------

constexpr uint8_t ElfMagic[]    = {0x7f, 'E', 'L', 'F'};
constexpr uint8_t SelfMagicA[]  = {0x4f, 0x15, 0x3d, 0x1d};
constexpr uint8_t SelfMagicB[]  = {0x54, 0x14, 0xf5, 0xee};
constexpr uint8_t HeaderMagic[] = {'1', '2', '3', '4', 0x18, 0, 0, 0};

struct ElfLoad {
	uint32_t type;
	uint64_t offset, vaddr, filesz;
};

// agc.plain_elf: the ELF image of eboot.bin: the file when it is one, else the ELF of a SELF with plaintext
// segments (a fake-signed one), each segment flagged 0x800 put back at the file offset of program header
// (type >> 20) & 0xfff.
std::vector<uint8_t> PlainElf(std::vector<uint8_t> data) {
	if (StartsWith(data, ElfMagic)) return data;
	const Bytes self = data;
	if (self.size() < 32 || (!StartsWith(self, SelfMagicA) && !StartsWith(self, SelfMagicB))) {
		throw Refused("not an ELF file nor a SELF");
	}
	const auto     file_size = Field<uint64_t>(self, 16);
	const uint64_t count     = Field<uint16_t>(self, 24);
	const uint64_t ehdr      = 32 + 32 * count;
	if (self.size() < ehdr + 64 || !StartsWith(self.subspan(ehdr), ElfMagic)) {
		throw Refused("SELF without an ELF header");
	}
	const auto     phoff     = Field<uint64_t>(self, ehdr + 32);
	const uint64_t phentsize = Field<uint16_t>(self, ehdr + 54);
	const uint64_t phnum     = Field<uint16_t>(self, ehdr + 56);
	const auto     table_end = Add(phoff, phnum * phentsize);
	std::vector<std::pair<uint64_t, uint64_t>> headers; // (p_offset, p_filesz)
	for (uint64_t i = 0; i < phnum; i++) {
		const auto at = Add(Add(ehdr, phoff), i * phentsize);
		Need(self, at, 56);
		headers.emplace_back(Field<uint64_t>(self, at + 8), Field<uint64_t>(self, at + 32));
	}
	uint64_t size = table_end;
	for (const auto& [offset, filesz]: headers) size = std::max(size, Add(offset, filesz));
	std::vector<uint8_t> image(size);
	Assign(image, 0, table_end, Slice(self, ehdr, SaturatingAdd(ehdr, table_end)));
	for (uint64_t i = 0; i < count; i++) {
		const auto kind   = Field<uint64_t>(self, 32 + 32 * i);
		const auto offset = Field<uint64_t>(self, 40 + 32 * i);
		const auto stored = Field<uint64_t>(self, 48 + 32 * i);
		const auto length = Field<uint64_t>(self, 56 + 32 * i);
		const auto index  = (kind >> 20u) & 0xfffu;
		if ((kind & 0x800u) == 0 || index >= phnum) continue;
		if ((kind & 0x2u) != 0 || stored != length || length != headers[index].second || offset > self.size() ||
		    length > self.size() - offset) {
			throw Refused("an encrypted or compressed SELF: the emulator needs the decrypted game (its plain eboot.bin "
			              "or a fake-signed SELF)");
		}
		Assign(image, headers[index].first, SaturatingAdd(headers[index].first, length), self.subspan(offset, length));
	}
	// A header whose bytes no segment holds: what follows the SELF's own bytes, when they are its size.
	for (const auto& [offset, filesz]: headers) {
		if (filesz == 0 || file_size > self.size() || filesz != self.size() - file_size) continue;
		const auto current = Slice(image, offset, SaturatingAdd(offset, filesz));
		if (std::any_of(current.begin(), current.end(), [](uint8_t byte) { return byte != 0; })) continue;
		Assign(image, offset, SaturatingAdd(offset, filesz), self.subspan(file_size));
	}
	return image;
}

// agc._elf_loads: (type, offset, vaddr, filesz) of each program header.
std::vector<ElfLoad> ElfLoads(Bytes elf) {
	if (!StartsWith(elf, ElfMagic)) throw Refused("not an ELF file");
	const auto           phoff     = Field<uint64_t>(elf, 32);
	const uint64_t       phentsize = Field<uint16_t>(elf, 54);
	const uint64_t       phnum     = Field<uint16_t>(elf, 56);
	std::vector<ElfLoad> loads;
	for (uint64_t i = 0; i < phnum; i++) {
		const auto at = Add(phoff, i * phentsize);
		Need(elf, at, 56);
		loads.push_back({Field<uint32_t>(elf, at), Field<uint64_t>(elf, at + 8), Field<uint64_t>(elf, at + 16),
		                 Field<uint64_t>(elf, at + 32)});
	}
	return loads;
}

// R_X86_64_RELATIVE relocations, target -> addend, in a Python dict's order: by first appearance, a later
// relocation of the same target replacing the addend in place.
struct Relocations {
	std::vector<std::pair<uint64_t, int64_t>> entries;
	std::unordered_map<uint64_t, size_t>      index;

	void Set(uint64_t where, int64_t addend) {
		const auto [found, added] = index.try_emplace(where, entries.size());
		if (added) {
			entries.emplace_back(where, addend);
		} else {
			entries[found->second].second = addend;
		}
	}
	const int64_t* Find(uint64_t where) const {
		const auto found = index.find(where);
		return found == index.end() ? nullptr : &entries[found->second].second;
	}
};

// agc._relative_relocations: the table PT_DYNAMIC's first DT_RELA (7) and DT_RELASZ (8) name.
Relocations RelativeRelocations(Bytes elf, const std::vector<ElfLoad>& loads) {
	Relocations out;
	const auto  dynamic = std::find_if(loads.begin(), loads.end(), [](const ElfLoad& load) { return load.type == 2; });
	if (dynamic == loads.end()) return out;
	std::map<int64_t, uint64_t> tags;
	for (uint64_t i = 0; i < dynamic->filesz / 16; i++) {
		const auto at = Add(dynamic->offset, 16 * i);
		Need(elf, at, 16);
		const auto tag = Field<int64_t>(elf, at);
		if (tag == 0) break;
		tags.try_emplace(tag, Field<uint64_t>(elf, at + 8));
	}
	const auto rela = tags.find(7), rela_size = tags.find(8);
	if (rela == tags.end() || rela_size == tags.end()) return out;
	// va_to_off: the table's file offset, through the PT_LOAD segment that maps its address.
	std::optional<uint64_t> start;
	bool                    beyond = false;
	for (const auto& load: loads) {
		if (load.type == 1 && load.vaddr <= rela->second && rela->second - load.vaddr < load.filesz) {
			const auto inside = rela->second - load.vaddr;
			if (load.offset > UINT64_MAX - inside) {
				beyond = true;
			} else {
				start = load.offset + inside;
			}
			break;
		}
	}
	for (uint64_t i = 0; i < rela_size->second / 24; i++) {
		if (!start) {
			throw Fatal(beyond ? "the relocation table lies beyond 64 bits" : "no segment maps the relocation table");
		}
		const auto at = Add(*start, 24 * i);
		Need(elf, at, 24);
		const auto where  = Field<uint64_t>(elf, at);
		const auto info   = Field<uint64_t>(elf, at + 8);
		const auto addend = Field<int64_t>(elf, at + 16);
		if ((info & 0xffffffffu) == 8) out.Set(where, addend);
	}
	return out;
}

// agc.embedded_shaders: the AGC headers in eboot.bin's data segment with their code (decrypted/eboot.bin when
// the game has one, else the eboot.bin the emulator runs), none without either. `image` keeps the ELF image the
// code views.
std::vector<Shader> EmbeddedShaders(const fs::path& game, std::vector<uint8_t>& image) {
	std::error_code error;
	auto            path = game / "decrypted" / "eboot.bin";
	if (!fs::is_regular_file(path, error)) path = game / "eboot.bin";
	if (!fs::is_regular_file(path, error)) return {};
	image             = PlainElf(ReadAll(path));
	const Bytes elf   = image;
	const auto  loads = ElfLoads(elf);
	const auto  found = std::search(elf.begin(), elf.end(), std::begin(HeaderMagic), std::end(HeaderMagic));
	if (found == elf.end()) return {};
	const auto     first   = static_cast<uint64_t>(found - elf.begin());
	const ElfLoad* segment = nullptr;
	for (const auto& load: loads) {
		if (load.type == 1 && load.offset <= first && first - load.offset < load.filesz) {
			segment = &load;
			break;
		}
	}
	if (segment == nullptr) throw Fatal("no segment holds the embedded shader headers");
	// The array of headers: each one's header_size leads to the next, while that starts with the magic.
	std::vector<std::pair<uint64_t, Bytes>> headers; // (file offset, header)
	for (uint64_t at = first; at - segment->offset < segment->filesz;) {
		const uint64_t size = Field<uint32_t>(elf, at + 64);
		headers.emplace_back(at, Slice(elf, at, at + size));
		const auto next = at + size;
		if (!StartsWith(Slice(elf, next, SaturatingAdd(next, 8)), HeaderMagic)) break;
		if (size == 0) throw Fatal("an embedded header of size 0 (precompile.py would not end)");
		at = next;
	}
	const auto            relocations = RelativeRelocations(elf, loads);
	std::vector<uint64_t> sizes; // shader_size of each header
	for (const auto& header: headers) sizes.push_back(Field<uint32_t>(header.second, 68));
	// Header address -> index (one beyond 64 bits is no relocation's addend).
	std::unordered_map<uint64_t, size_t> header_at;
	for (size_t k = 0; k < headers.size(); k++) {
		const auto inside = headers[k].first - segment->offset;
		if (segment->vaddr <= UINT64_MAX - inside) header_at[segment->vaddr + inside] = k;
	}
	// The code of header k starts where header k-1's (256-byte aligned) code ends; a relocated (header, code)
	// pointer pair fixes the start of the sequence.
	std::optional<int64_t> starts;
	for (const auto& [where, addend]: relocations.entries) {
		const auto k = addend >= 0 ? header_at.find(static_cast<uint64_t>(addend)) : header_at.end();
		if (k == header_at.end() || where > UINT64_MAX - 8) continue;
		const auto* code = relocations.Find(where + 8);
		if (code == nullptr) continue;
		int64_t code_va = *code;
		for (size_t j = 0; j < k->second; j++) {
			code_va = AddSigned(code_va, -static_cast<int64_t>((sizes[j] + 0xffu) & ~uint64_t {0xff}));
		}
		starts = code_va;
		break;
	}
	if (!starts) throw Refused("no relocation pairs an embedded header with its code");
	const auto          delta   = AddSigned(Signed(segment->offset), -Signed(segment->vaddr)); // file offset - address
	int64_t             code_va = *starts;
	std::vector<Shader> out;
	for (const auto& header: headers) {
		auto       agc    = ParseAgc(header.second);
		const auto size   = static_cast<int64_t>(agc.shader_size);
		const auto offset = AddSigned(code_va, delta);
		out.push_back({SliceSigned(elf, offset, AddSigned(offset, size)), std::move(agc)});
		code_va = AddSigned(AddSigned(code_va, size), 0xff) & ~int64_t {0xff};
	}
	return out;
}

// ---------------------------------------------------------------------------------------------------------
// materials.py: '.cmat' materials
// ---------------------------------------------------------------------------------------------------------

constexpr uint32_t KindPair = 0x11, KindCompute = 0x20;
constexpr uint8_t  MaterialFooter[] = {'S', 'T', 'A', 'M'};

// materials.parse: a material's bundle path and techniques (pass id, kind). precompile.py does not catch its
// errors: a material it cannot read stops it.
std::pair<std::string, std::vector<std::pair<uint32_t, uint32_t>>> ParseMaterial(Bytes b) {
	const uint64_t size = b.size();
	if (size < 0x30 || !StartsWith(b.subspan(size - 12), MaterialFooter) || Field<uint32_t>(b, size - 4) != size - 12) {
		throw Fatal("missing STAM footer");
	}
	// The bundle path: a LEB128 length, then '$/...'.
	uint64_t p = 0x20, length = 0;
	for (unsigned shift = 0;;) {
		if (p >= size) throw Fatal("the bundle path's length runs past the end");
		const uint8_t c = b[p++];
		length |= uint64_t {c & 0x7fu} << shift;
		shift += 7;
		if ((c & 0x80) == 0) break;
		if (shift > 28) throw Fatal("bad length");
	}
	const auto path = Slice(b, p, SaturatingAdd(p, length));
	if (path.size() != length || (length != 0 && path[0] != '$')) throw Fatal("bad path at " + std::to_string(p));
	std::string bundle;
	for (size_t i = 1; i < path.size(); i++) {
		if (path[i] >= 0x80) throw Fatal("a bundle path that is not ASCII");
		bundle.push_back(static_cast<char>(path[i]));
	}
	p += length;
	const auto count = Field<uint32_t>(b, p);
	p += 4;
	if (count > 64) throw Fatal("implausible technique count");
	std::vector<std::pair<uint32_t, uint32_t>> techniques;
	for (uint64_t i = 0; i < count; i++) {
		techniques.emplace_back(Field<uint32_t>(b, p + 8 * i), Field<uint32_t>(b, p + 8 * i + 4));
	}
	return {std::move(bundle), std::move(techniques)};
}

// materials.bundle_file: the components below the game directory of a material's bundle path
// ('/materials/.../****/x.csdr': the leading slashes stripped, '****' the PS5 folder), as pathlib splits them.
std::vector<std::string> BundleParts(std::string path) {
	path.erase(0, path.find_first_not_of('/'));
	std::string replaced;
	for (size_t at = 0;;) {
		const auto found = path.find("****", at);
		replaced.append(path, at, found == std::string::npos ? std::string::npos : found - at);
		if (found == std::string::npos) break;
		replaced += "_ps5";
		at = found + 4;
	}
	std::vector<std::string> parts;
	std::string              part;
	const auto               flush = [&] {
		if (!part.empty() && part != ".") parts.push_back(part);
		part.clear();
	};
	for (const char c: replaced) {
		if (c == '/' || (WindowsPaths && c == '\\')) {
			flush();
		} else {
			part.push_back(c);
		}
	}
	flush();
	return parts;
}

// ---------------------------------------------------------------------------------------------------------
// keys.py: compile inputs from the AGC header, as records (warmfile.encode_record)
// ---------------------------------------------------------------------------------------------------------

// SH/CX register offsets (src/graphics/guest_gpu/pm4.h).
constexpr uint32_t ComputeNumThreadX = 0x207, ComputeNumThreadY = 0x208, ComputeNumThreadZ = 0x209;
constexpr uint32_t ComputePgmRsrc2 = 0x213, SpiShaderPgmRsrc2Ps = 0xb, SpiShaderPgmRsrc2Gs = 0x8b;
constexpr uint32_t SpiPsInputEna = 0x1b3, SpiPsInputAddr = 0x1b4, SpiPsInControl = 0x1b6;
constexpr uint32_t SpiShaderColFormat = 0x1c5, DbShaderControl = 0x203, PaClVsOutCntl = 0x207;
constexpr uint32_t IdentityExportMapping = 0xe4;

// agc.reg_first: the value of the first register at `offset`.
std::optional<uint32_t> RegFirst(const std::vector<std::pair<uint32_t, uint32_t>>& registers, uint32_t offset) {
	for (const auto& [at, value]: registers) {
		if (at == offset) return value;
	}
	return std::nullopt;
}

uint32_t UserDataCount(uint32_t rsrc2) {
	return ((rsrc2 >> 1u) & 0x1fu) | (((rsrc2 >> 27u) & 1u) << 5u);
}

// warmfile.encode_record of a record keys.py makes: its code's XXH3-64 (ShaderParams.hash), no push data and no
// back code, the resource specialization left to the compiler (no buffers, no images).
std::vector<uint32_t> RecordWords(uint32_t stage, Bytes code, uint32_t user_data_count, std::span<const uint32_t> key,
                                  std::span<const uint32_t> info) {
	if (code.size() % 4 != 0) throw Fatal("shader code of " + std::to_string(code.size()) + " bytes, not whole words");
	const auto            hash  = XXH3_64bits(code.data(), code.size());
	const auto            count = static_cast<uint32_t>(code.size() / 4);
	std::vector<uint32_t> words {stage, static_cast<uint32_t>(hash), static_cast<uint32_t>(hash >> 32u),
	                             user_data_count, 0, count};
	words.resize(words.size() + count);
	if (count != 0) std::memcpy(words.data() + 6, code.data(), code.size());
	words.push_back(0); // back code
	words.push_back(static_cast<uint32_t>(key.size()));
	words.insert(words.end(), key.begin(), key.end());
	words.insert(words.end(), info.begin(), info.end());
	words.push_back(0); // buffers
	words.push_back(0); // images
	return words;
}

// keys.compute_record: every input but the resource specialization.
std::vector<uint32_t> ComputeRecord(Bytes code, const Agc& agc) {
	const auto rsrc2 = RegFirst(agc.sh_registers, ComputePgmRsrc2);
	const auto x     = RegFirst(agc.sh_registers, ComputeNumThreadX);
	const auto y     = RegFirst(agc.sh_registers, ComputeNumThreadY);
	const auto z     = RegFirst(agc.sh_registers, ComputeNumThreadZ);
	if (!rsrc2 || !x || !y || !z) throw Refused("compute header without its static registers");
	const uint32_t r = *rsrc2;
	// The game dispatches with the header's modifier (AgcCbDispatch): wave size, thread dimensions.
	const uint32_t modifier   = agc.has_specials ? agc.dispatch_modifier : 0;
	const uint32_t lds        = ((r >> 15u) & 0x1ffu) * 128;
	const uint32_t scratch    = agc.scratch_size_dw_per_thread;
	const uint32_t wave       = ((modifier >> 15u) & 1u) != 0 ? 32 : 64;
	const uint32_t dimensions = (modifier >> 5u) & 1u;
	const uint32_t group_x = (r >> 7u) & 1u, group_y = (r >> 8u) & 1u, group_z = (r >> 9u) & 1u;
	const uint32_t thread_ids = ((r >> 11u) & 3u) + 1, workgroup = (r >> 1u) & 0x1fu, tg_size = (r >> 10u) & 1u;
	const uint32_t key[]  = {workgroup, wave,    HostSubgroupSize, thread_ids, lds,     scratch, dimensions,
	                         *x,        group_x, *y,               group_y,    *z,      group_z, tg_size};
	const uint32_t info[] = {*x,      *y,      *z,         lds,        scratch,   HostSubgroupSize, wave, 0, 0, 0,
	                         group_x, group_y, group_z, dimensions, thread_ids, workgroup,        tg_size};
	return RecordWords(StageCompute, code, workgroup, key, info);
}

// keys.interpolant_mapping (AgcCreateInterpolantMapping): SPI_PS_INPUT_CNTL_i of each pixel shader input from
// the vertex shader's outputs.
std::array<uint32_t, 32> InterpolantMapping(const Agc& vertex, const Agc& pixel) {
	std::array<uint32_t, 32> regs {};
	for (uint32_t i = 0; i < 32; i++) regs[i] = i;
	if (pixel.num_input_semantics == 0) return regs;
	for (size_t i = 0; i < pixel.input_semantics.size(); i++) {
		const uint32_t  ps_word = pixel.input_semantics[i];
		const uint32_t* vs_word = nullptr;
		for (const auto& output: vertex.output_semantics) {
			if ((output & 0xffu) == (ps_word & 0xffu)) {
				vs_word = &output;
				break;
			}
		}
		uint32_t value = 0;
		if ((ps_word & 0x00300000u) != 0) {
			value = (ps_word << 4u) & 0x03000000u;
			if (vs_word == nullptr) {
				value |= 0x00180020u;
			} else {
				const uint32_t common = ps_word & *vs_word;
				value &= 0xfff7ffdfu;
				value |= (common >> 15u) & 0x20u;
				value ^= 0x00080020u;
				value &= ~0x00100000u;
				value |= (~common >> 1u) & 0x00100000u;
			}
			value = (value & ~0x00600000u) | (((ps_word >> 30u) & 3u) << 21u);
		} else {
			value = (ps_word & 0x01000000u) != 0 || vs_word == nullptr ? 0x20u : 0u;
		}
		if (vs_word == nullptr) {
			value &= ~0x41fu;
		} else {
			const uint32_t flat = (ps_word & 0x00400000u) != 0 || (ps_word & 0x01000000u) != 0 ? 0x400u : 0u;
			value               = (value & ~0x41fu) | ((*vs_word >> 8u) & 0x1fu) | flat;
		}
		// (Python's list of 32 settings takes no more: an IndexError.)
		if (i >= regs.size()) throw Fatal("a pixel shader with more than 32 input semantics");
		regs[i] = (value & ~0x300u) | (((ps_word >> 28u) & 3u) << 8u);
	}
	return regs;
}

// keys._ps_system_input_base: the VGPRs the enabled system inputs (SPI_PS_INPUT_ADDR) take first.
uint32_t SystemInputBase(uint32_t addr) {
	constexpr std::pair<uint32_t, uint32_t> inputs[] = {{0x1, 2},  {0x2, 2},  {0x4, 2},  {0x8, 3},
	                                                    {0x10, 2}, {0x20, 2}, {0x40, 2}, {0x80, 1}};
	uint32_t                                base     = 0;
	for (const auto& [bit, count]: inputs) {
		if ((addr & bit) != 0) base += count;
	}
	return base;
}

// keys.pixel_record: a pixel shader drawn after `vertex` (whose outputs feed its interpolators).
std::vector<uint32_t> PixelRecord(Bytes code, const Agc& agc, const Agc& vertex) {
	const auto     reg         = [&](uint32_t offset) { return RegFirst(agc.cx_registers, offset).value_or(0); };
	const uint32_t addr        = reg(SpiPsInputAddr);
	const uint32_t active      = reg(SpiPsInputEna) & addr;
	const uint32_t control     = reg(DbShaderControl);
	const uint32_t z_export    = control & 1u;
	const uint32_t z_order     = (control >> 4u) & 3u;
	const uint32_t kill        = (control >> 6u) & 1u;
	const uint32_t mask_export = (control >> 8u) & 1u;
	const uint32_t col         = reg(SpiShaderColFormat);
	const uint32_t input_num   = reg(SpiPsInControl) & 0x3fu;
	uint32_t       custom      = 0;
	for (size_t i = 0; i < std::min<size_t>(std::min<uint32_t>(input_num, 32), agc.input_semantics.size()); i++) {
		const auto word = agc.input_semantics[i];
		if (((word >> 24u) & 1u) != 0 && ((word >> 20u) & 3u) == 0) custom |= 1u << i;
	}
	const auto interpolators = InterpolantMapping(vertex, agc);
	// warmfile.valid_key reads a setting of the 32 per input: more inputs stop precompile.py (an IndexError).
	if (input_num > 32) throw Fatal("a pixel shader with " + std::to_string(input_num) + " inputs");
	const uint32_t base            = SystemInputBase(addr);
	const uint32_t center          = (active & 2u) != 0 ? ((active & 1u) != 0 ? 2u : 0u) : 0xffffffffu;
	const uint32_t scratch         = agc.scratch_size_dw_per_thread;
	const uint32_t pos_x           = (active & 0x100u) != 0;
	const uint32_t pos_y           = (active & 0x200u) != 0;
	const uint32_t pos_z           = (active & 0x400u) != 0;
	const uint32_t pos_w           = (active & 0x800u) != 0;
	const uint32_t front_face      = (active & 0x1000u) != 0;
	const uint32_t ancillary       = (active & 0x2000u) != 0;
	const uint32_t no_perspective  = (active & 0x20u) != 0;
	const uint32_t sample_shading  = (active & 0x11u) != 0;
	const uint32_t early_z         = z_order == 1 && kill == 0 && z_export == 0 && mask_export == 0;
	const uint32_t execute_on_noop = (control >> 10u) & 1u;
	std::vector<uint32_t> key {scratch, input_num, base,      custom,         center, pos_x,    pos_y,       pos_z,
	                           pos_w,   front_face, ancillary, no_perspective, kill,   z_export, mask_export, early_z};
	for (uint32_t i = 0; i < 8; i++) key.push_back((col >> (4 * i)) & 0xfu); // target_output_mode
	// target_export_mapping: the identity channel order of the render targets the game binds, 4 a word.
	key.push_back(IdentityExportMapping * 0x01010101u);
	key.push_back(IdentityExportMapping * 0x01010101u);
	key.insert(key.end(), interpolators.begin(), interpolators.begin() + input_num);
	std::vector<uint32_t> info {LodStatsSubgroup, input_num,      base,           custom, center,   scratch,
	                            pos_x,            pos_y,          pos_z,          pos_w,  front_face, ancillary,
	                            no_perspective,   kill,           z_export,       mask_export, sample_shading, early_z,
	                            execute_on_noop};
	info.insert(info.end(), interpolators.begin(), interpolators.begin() + input_num);
	info.resize(info.size() + (32 - input_num), 0);
	for (uint32_t i = 0; i < 8; i++) info.push_back((col >> (4 * i)) & 0xfu);
	info.resize(info.size() + 8, IdentityExportMapping);
	const auto rsrc2 = RegFirst(agc.sh_registers, SpiShaderPgmRsrc2Ps).value_or(0);
	return RecordWords(StagePixel, code, UserDataCount(rsrc2), key, info);
}

// keys.vertex_record: this game's vertex shaders pull their vertices through the SRT (no fetch tables, no clip
// transform, no mesh path), so the key holds header fields only; Refused with fetch tables.
std::vector<uint32_t> VertexRecord(Bytes code, const Agc& agc) {
	const auto& offsets = agc.direct_resource_offset;
	if ((offsets.size() > 10 && offsets[10] != 0xffff) || (offsets.size() > 8 && offsets[8] != 0xffff)) {
		throw Refused("vertex shader with fetch tables: its key needs draw-time descriptors");
	}
	const uint32_t scratch  = agc.scratch_size_dw_per_thread;
	const uint32_t out_cntl = RegFirst(agc.cx_registers, PaClVsOutCntl).value_or(0);
	// start_instance_sgpr -1: the direct draws' variant.
	const uint32_t key[]  = {0, 0, 0, 0, scratch, out_cntl, 0xffffffffu, 0, 0};
	const uint32_t info[] = {0, 0, 0,                         // resources_num, fetch_attrib_reg, fetch_buffer_reg
	                         scratch, out_cntl, 0xffffffffu, // scratch, PA_CL_VS_OUT_CNTL, start_instance_sgpr
	                         0, 0, 0,                         // fetch_external, fetch_embedded, clip enabled
	                         0, 0, 0, 0, 0, 0,                // clip scale, offset, half extent
	                         0, 0, 0, 0, 0, 64, 64,           // mesh: threads, LDS, scratch, subgroup, wave
	                         0, 0, 0, 0, 0, 0};               // mesh: primitive and vertex counts, provoking vertex
	const auto     rsrc2  = RegFirst(agc.sh_registers, SpiShaderPgmRsrc2Gs).value_or(0);
	return RecordWords(StageVertex, code, UserDataCount(rsrc2), key, info);
}

// ---------------------------------------------------------------------------------------------------------
// Pipeline recipes (warmfile.encode_pipeline, decode_pipeline) and pass-states.json
// ---------------------------------------------------------------------------------------------------------

// The state words of a graphics recipe, after its vertex, pixel and compute indices.
constexpr size_t RecipeWords = 244, RecipeColorCount = 0, RecipeColorFormats = 3, RecipeBindingCount = 11;
constexpr size_t RecipeAttributeCount = 12, RecipeBindings = 13, RecipeAttributes = 77, RecipeTopology = 143;
constexpr size_t RecipeSamples = 145, RecipeStencil = 151, RecipeCullFront = 159, RecipeCullBack = 160;
constexpr size_t RecipePolygonMode = 163, RecipeColorMask = 164, RecipeColorSrcBlend = 172, RecipeColorCombFcn = 180;
constexpr size_t RecipeColorDestBlend = 188, RecipeAlphaSrcBlend = 196, RecipeAlphaCombFcn = 204;
constexpr size_t RecipeAlphaDestBlend = 212, RecipeBlendFlags = 220;
// Read as booleans besides the bindings' instance flags and the blend flags: negative_one_to_one,
// depth_clip_enable, primitive_restart_enable, sample_shading_enable, depth_bounds_test_enable,
// stencil_test_enable, cull_front, cull_back, face, provoking_vtx_last.
constexpr size_t RecipeFlags[] = {141, 142, 144, 146, 147, 150, 159, 160, 161, 162};

// warmfile.decode_pipeline of a learned recipe with shaders 0 and 0: precompile.py stops on one it rejects.
void CheckRecipe(const std::vector<uint32_t>& w) {
	const auto fail = [](const char* what) {
		throw Fatal(std::string("pass-states.json: a recipe that does not decode (") + what + ")");
	};
	if (w.size() != RecipeWords) fail("not 244 words");
	for (const auto at: RecipeFlags) {
		if (w[at] > 1) fail("a flag above 1");
	}
	for (size_t i = 0; i < 32; i++) {
		if (w[RecipeBindings + 2 * i + 1] > 1) fail("an instance flag above 1");
	}
	for (size_t at = RecipeBlendFlags; at < RecipeWords; at++) {
		if (w[at] > 1) fail("a blend flag above 1");
	}
	for (size_t at = RecipeStencil; at < RecipeStencil + 8; at++) {
		if (w[at] > 7) fail("a stencil op or compare above 7");
	}
	for (size_t at = RecipeColorSrcBlend; at < RecipeBlendFlags; at++) {
		if (w[at] > 255) fail("a blend factor or function wider than 8 bits");
	}
	const uint32_t color_count = w[RecipeColorCount], binding_count = w[RecipeBindingCount];
	const uint32_t attribute_count = w[RecipeAttributeCount], samples = w[RecipeSamples];
	if (color_count > 8 || binding_count > 32 || attribute_count > 32) fail("count out of range");
	if (w[RecipeTopology] > 10 || w[RecipePolygonMode] > 2) fail("topology/polygon_mode out of range");
	if (samples == 0 || samples > 64 || (samples & (samples - 1)) != 0) fail("samples not a power of two <= 64");
	for (size_t k = 0; k < attribute_count; k++) {
		if (w[RecipeAttributes + 2 * k + 1] >= binding_count) fail("attribute binding >= binding_count");
	}
	for (size_t k = 0; k < color_count; k++) {
		if (w[RecipeColorFormats + k] == 0 || w[RecipeColorMask + k] > 15 || w[RecipeColorSrcBlend + k] > 20 ||
		    w[RecipeColorDestBlend + k] > 20 || w[RecipeColorCombFcn + k] > 4 || w[RecipeAlphaSrcBlend + k] > 20 ||
		    w[RecipeAlphaDestBlend + k] > 20 || w[RecipeAlphaCombFcn + k] > 4) {
			fail("per-target state out of range");
		}
	}
}

// precompile._culling_variants: the recipe as learned and with culling toggled (a material's two-sidedness
// decides it, and its flag matches the recorded culling of only four draws in five).
std::array<std::vector<uint32_t>, 2> CullingVariants(const std::vector<uint32_t>& words) {
	CheckRecipe(words);
	auto toggled = words;
	if (toggled[RecipeCullFront] != 0 || toggled[RecipeCullBack] != 0) {
		toggled[RecipeCullFront] = 0;
		toggled[RecipeCullBack]  = 0;
	} else {
		toggled[RecipeCullBack] = 1;
	}
	return {words, std::move(toggled)};
}

// precompile._with_shaders.
std::vector<uint32_t> WithShaders(uint32_t vertex, uint32_t pixel, const std::vector<uint32_t>& state) {
	std::vector<uint32_t> words {vertex, pixel, NoShader};
	words.insert(words.end(), state.begin(), state.end());
	return words;
}

using Palette = std::vector<std::vector<uint32_t>>;

// pass-states.json's groups: the state words of each material pass's recipes, of the engine's own passes
// ("engine") and of eboot.bin's vertex-only draws ("vertex-only"). A null group is kept as such: states.get
// gives None for it as for a missing one, but iterating it stops precompile.py.
struct States {
	std::map<std::string, std::optional<Palette>> groups;

	// states.get(group): null when the group is missing or null.
	const Palette* Find(const std::string& group) const {
		const auto found = groups.find(group);
		return found == groups.end() || !found->second ? nullptr : &*found->second;
	}
	// states.get('vertex-only', []).
	const Palette& VertexOnly() const {
		static const Palette none;
		const auto           found = groups.find("vertex-only");
		if (found == groups.end()) return none;
		if (!found->second) throw Fatal("pass-states.json: the vertex-only group is null");
		return *found->second;
	}
};

// json.loads(pass-states.json)['groups'], whose recipes precompile.py packs as 32-bit unsigned words
// (array('I'): booleans are 0 and 1, anything else stops it).
States LoadStates(const fs::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) throw Fatal("cannot read " + Utf8(path));
	const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	const auto        root = nlohmann::json::parse(text, nullptr, false);
	const auto        bad  = [&](const std::string& what) { return Fatal(Utf8(path) + ": " + what); };
	if (root.is_discarded() || !root.is_object()) throw bad("not a JSON object");
	const auto groups = root.find("groups");
	if (groups == root.end() || !groups->is_object()) throw bad("no \"groups\" object");
	States states;
	for (auto group = groups->begin(); group != groups->end(); ++group) {
		const auto& value = group.value();
		if (value.is_null()) {
			states.groups.emplace(group.key(), std::nullopt);
			continue;
		}
		if (!value.is_array()) throw bad("group " + group.key() + " is not a list");
		Palette palette;
		for (const auto& recipe: value) {
			if (!recipe.is_array()) throw bad("group " + group.key() + " holds a recipe that is not a list");
			std::vector<uint32_t> words;
			for (const auto& word: recipe) {
				if (word.is_boolean()) {
					words.push_back(word.get<bool>() ? 1 : 0);
				} else if (word.is_number_unsigned() && word.get<uint64_t>() <= UINT32_MAX) {
					words.push_back(static_cast<uint32_t>(word.get<uint64_t>()));
				} else if (word.is_number_integer() && word.get<int64_t>() == 0) {
					words.push_back(0); // -0
				} else {
					throw bad("group " + group.key() + " holds a word that is not a 32-bit unsigned integer");
				}
			}
			palette.push_back(std::move(words));
		}
		states.groups.emplace(group.key(), std::move(palette));
	}
	return states;
}

// precompile.py SeedFile: the records and the pipeline recipes, each kept once, in the order first added.
class SeedFile {
public:
	std::vector<std::vector<uint32_t>> records, pipelines;

	uint32_t Record(std::vector<uint32_t> words) {
		return Add(records, record_index, std::move(words));
	}
	void Pipeline(std::vector<uint32_t> words) {
		Add(pipelines, pipeline_index, std::move(words));
	}

	// SeedFile.write: the identity line, the XXH3-64 of the body, the body (each list: its count, then each
	// entry's word count and words); the body's size. A file is written whole or not at all.
	uint64_t Write(const fs::path& path) const {
		std::vector<uint32_t> body;
		for (const auto* entries: {&records, &pipelines}) {
			body.push_back(static_cast<uint32_t>(entries->size()));
			for (const auto& words: *entries) {
				body.push_back(static_cast<uint32_t>(words.size()));
				body.insert(body.end(), words.begin(), words.end());
			}
		}
		const uint64_t  bytes    = body.size() * sizeof(uint32_t);
		const uint64_t  checksum = XXH3_64bits(body.data(), bytes);
		std::error_code error;
		if (path.has_parent_path()) fs::create_directories(path.parent_path(), error);
		auto temporary = path;
		temporary += ".tmp";
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		file.write(SeedIdentity.data(), static_cast<std::streamsize>(SeedIdentity.size()));
		file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
		file.write(reinterpret_cast<const char*>(body.data()), static_cast<std::streamsize>(bytes));
		file.close();
		if (!file) throw Fatal("cannot write " + Utf8(temporary));
		fs::rename(temporary, path, error);
		if (error) throw Fatal("cannot write " + Utf8(path) + ": " + error.message());
		return bytes;
	}

private:
	using Index = std::unordered_multimap<uint64_t, uint32_t>;
	Index record_index, pipeline_index;

	static uint32_t Add(std::vector<std::vector<uint32_t>>& entries, Index& index, std::vector<uint32_t> words) {
		const auto hash         = XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
		const auto [begin, end] = index.equal_range(hash);
		for (auto it = begin; it != end; ++it) {
			if (entries[it->second] == words) return it->second;
		}
		const auto id = static_cast<uint32_t>(entries.size());
		index.emplace(hash, id);
		entries.push_back(std::move(words));
		return id;
	}
};

// ---------------------------------------------------------------------------------------------------------
// precompile.py Inventory
// ---------------------------------------------------------------------------------------------------------

struct Bundle {
	std::vector<std::string> parts; // below the game directory
	std::vector<Shader>      entries;
};

struct Material {
	std::string                                key;        // its bundle's PathKey
	std::vector<std::pair<uint32_t, uint32_t>> techniques; // (pass id, kind)
};

// Every shader of the game, by code bytes, with the headers it comes with.
struct Inventory {
	std::deque<std::vector<uint8_t>> files;     // the bundles' bytes, which their shaders' code views
	std::vector<uint8_t>             image;     // eboot.bin's ELF image, which the embedded shaders' code views
	std::vector<Bundle>              bundles;   // the bundles that parse, in the order sorted() gives their paths
	std::vector<Shader>              embedded;  // the shaders embedded in eboot.bin
	std::vector<Material>            materials; // the first material of each bundle, in .cmat path order

	// Inventory.headers: each distinct code with its distinct headers, in order of first appearance.
	struct Program {
		Bytes                   code;
		std::vector<const Agc*> headers;
	};
	std::vector<Program>                           programs;
	std::unordered_map<std::string_view, uint32_t> program_of;

	explicit Inventory(const fs::path& game);

	static std::string_view View(Bytes code) {
		return {reinterpret_cast<const char*>(code.data()), code.size()};
	}
	uint32_t ProgramOf(Bytes code) const {
		return program_of.at(View(code));
	}
};

Inventory::Inventory(const fs::path& game) {
	for (auto& file: FindGameFiles(game, ".csdr")) {
		// agc.csdr_files: the '_trinity' twins are the PS5 Pro variants.
		const std::string_view name = file.parts.back();
		if (name.size() > 5 && name.substr(0, name.size() - 5).ends_with("_trinity")) continue;
		const Bytes data = files.emplace_back(ReadAll(file.path));
		Bundle      bundle {std::move(file.parts), {}};
		try {
			for (const auto& [code, header]: ParseBundle(data)) bundle.entries.push_back({code, ParseAgc(header)});
		} catch (const Refused&) {
			files.pop_back();
			continue;
		}
		bundles.push_back(std::move(bundle));
	}
	try {
		embedded = EmbeddedShaders(game, image);
	} catch (const Refused& error) {
		// (The bundles' shaders still precompile: eboot.bin's own are left to the first frames that use them.)
		embedded.clear();
		std::fprintf(stderr, "eboot.bin: %s; its embedded shaders are not precompiled\n", error.what());
	} catch (const Fatal& error) {
		throw Fatal(std::string("eboot.bin: ") + error.what());
	}
	// materials.techniques: materials sharing a bundle share its list (the first one's).
	std::unordered_map<std::string, size_t> material_of;
	for (const auto& file: FindGameFiles(game, ".cmat")) {
		try {
			auto [bundle, techniques] = ParseMaterial(ReadAll(file.path));
			auto key                  = PathKey(BundleParts(std::move(bundle)));
			if (material_of.emplace(key, materials.size()).second) {
				materials.push_back({std::move(key), std::move(techniques)});
			}
		} catch (const Fatal& error) {
			throw Fatal(Utf8(file.path) + ": " + error.what());
		}
	}
	const auto add = [&](const Shader& shader) {
		const auto [found, added] = program_of.try_emplace(View(shader.code), static_cast<uint32_t>(programs.size()));
		if (added) programs.push_back({shader.code, {}});
		auto& headers = programs[found->second].headers;
		if (std::none_of(headers.begin(), headers.end(), [&](const Agc* known) { return *known == shader.agc; })) {
			headers.push_back(&shader.agc);
		}
	};
	for (const auto& bundle: bundles) {
		for (const auto& shader: bundle.entries) add(shader);
	}
	for (const auto& shader: embedded) add(shader);
}

// A vertex and pixel shader drawn together, with the material pass (none: not a material's).
struct Drawn {
	const Shader*           vertex;
	const Shader*           pixel;
	std::optional<uint32_t> pass;
};

// Inventory.graphics_pairs: the (Gs, Ps) entries of every technique of every material, then the adjacent
// (Gs, Ps) entries of the other bundles.
std::vector<Drawn> GraphicsPairs(const Inventory& inventory) {
	std::unordered_map<std::string, size_t> bundle_of;
	for (size_t i = 0; i < inventory.bundles.size(); i++) bundle_of.emplace(PathKey(inventory.bundles[i].parts), i);
	std::vector<Drawn> out;
	std::vector<bool>  covered(inventory.bundles.size());
	for (const auto& material: inventory.materials) {
		const auto found = bundle_of.find(material.key);
		if (found == bundle_of.end()) continue;
		const auto& entries = inventory.bundles[found->second].entries;
		// materials.technique_entries: the techniques take the bundle's entries in order, two for a (VS, PS)
		// pair, one for a compute shader, and all of them.
		std::vector<std::pair<uint32_t, size_t>> pairs; // (pass id, first entry)
		size_t                                   cursor = 0;
		bool                                     fits   = true;
		for (const auto& [pass, kind]: material.techniques) {
			const size_t take = kind == KindPair ? 2 : kind == KindCompute ? 1 : 0;
			if (take == 0 || cursor + take > entries.size()) {
				fits = false;
				break;
			}
			if (kind == KindPair) pairs.emplace_back(pass, cursor);
			cursor += take;
		}
		if (!fits || cursor != entries.size()) continue;
		covered[found->second] = true;
		for (const auto& [pass, first]: pairs) {
			const auto& vertex = entries[first];
			const auto& pixel  = entries[first + 1];
			if (vertex.agc.type == TypeGs && pixel.agc.type == TypePs) out.push_back({&vertex, &pixel, pass});
		}
	}
	for (size_t i = 0; i < inventory.bundles.size(); i++) {
		const auto& entries = inventory.bundles[i].entries;
		if (covered[i]) continue;
		for (size_t e = 0; e + 1 < entries.size(); e++) {
			if (entries[e].agc.type == TypeGs && entries[e + 1].agc.type == TypePs) {
				out.push_back({&entries[e], &entries[e + 1], std::nullopt});
			}
		}
	}
	return out;
}

// Inventory.engine_pairs: the engine's own passes (post-processing, UI, particles:
// coredata/enginesupport/shaders/<effect>/, one shader a bundle), paired by the engine's code: every vertex
// shader with every pixel shader of the same effect directory, the full-screen vertex shaders with the pixel
// shaders of directories without one, and the shaders embedded in eboot.bin with each other.
std::vector<Drawn> EnginePairs(const Inventory& inventory) {
	constexpr std::string_view engine[] = {"coredata", "enginesupport", "shaders"};
	// Effect directory (lowercase) -> its vertex shaders [0] and pixel shaders [1], ordered by name.
	std::map<std::string, std::array<std::vector<const Shader*>, 2>> by_directory;
	for (const auto& bundle: inventory.bundles) {
		const auto& parts = bundle.parts;
		if (parts.size() < 3 || Lower(parts[0]) != engine[0] || Lower(parts[1]) != engine[1] ||
		    Lower(parts[2]) != engine[2]) {
			continue;
		}
		for (const auto& shader: bundle.entries) {
			if (shader.agc.type == TypeGs || shader.agc.type == TypePs) {
				by_directory[Lower(parts[parts.size() - 2])][shader.agc.type == TypePs ? 1 : 0].push_back(&shader);
			}
		}
	}
	const std::vector<const Shader*> none;
	const auto                       fullscreen_found = by_directory.find("vs_fullscreen");
	const auto& fullscreen = fullscreen_found != by_directory.end() ? fullscreen_found->second[0] : none;
	std::vector<Drawn> out;
	for (const auto& [name, shaders]: by_directory) {
		for (const auto* vertex: shaders[0].empty() ? fullscreen : shaders[0]) {
			for (const auto* pixel: shaders[1]) out.push_back({vertex, pixel, std::nullopt});
		}
	}
	std::vector<const Shader*> embedded_vertex, embedded_pixel;
	for (const auto& shader: inventory.embedded) {
		if (shader.agc.type == TypeGs) embedded_vertex.push_back(&shader);
		if (shader.agc.type == TypePs) embedded_pixel.push_back(&shader);
	}
	embedded_vertex.insert(embedded_vertex.end(), fullscreen.begin(), fullscreen.end());
	for (const auto* vertex: embedded_vertex) {
		for (const auto* pixel: embedded_pixel) out.push_back({vertex, pixel, std::nullopt});
	}
	return out;
}

const char* StageName(uint32_t stage) {
	switch (stage) {
		case 1: return "Vertex";
		case 2: return "Pixel";
		case 3: return "Fetch";
		case 4: return "Compute";
		case 5: return "Mesh";
		default: return "Unknown";
	}
}

// The directory of this program, or empty.
fs::path ProgramDirectory() {
#ifdef _WIN32
	std::wstring path(32768, L'\0');
	const auto   length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
	if (length == 0 || length >= path.size()) return {};
	path.resize(length);
	return fs::path(path).parent_path();
#else
	std::error_code error;
	const auto      program = fs::read_symlink("/proc/self/exe", error);
	return error ? fs::path() : program.parent_path();
#endif
}

} // namespace

std::filesystem::path FindStates() {
	const auto            name = fs::path("tools") / "local" / "static-precompile" / "pass-states.json";
	std::vector<fs::path> candidates;
	if (const auto program = ProgramDirectory(); !program.empty()) {
		candidates.push_back(program / name);               // a release package: the program at its root
		candidates.push_back(program / ".." / ".." / name); // a build tree: _Build/<build>/
		candidates.push_back(program / "pass-states.json");
	}
	candidates.push_back(name); // the scripts run the program from the repository's or the package's root
	for (const auto& candidate: candidates) {
		std::error_code error;
		if (fs::is_regular_file(candidate, error)) return candidate;
	}
	return {};
}

// precompile.py cmd_seeds.
int Make(const Options& options) {
	try {
		const auto begin       = std::chrono::steady_clock::now();
		const auto states_file = options.states.empty() ? FindStates() : options.states;
		if (states_file.empty()) throw Fatal("no tools/local/static-precompile/pass-states.json (--states <file>)");
		const auto      states = LoadStates(states_file);
		const Inventory inventory(options.game);
		if (inventory.bundles.empty()) {
			std::fprintf(stderr, "no shader bundles (*.csdr) under %s\n", Utf8(options.game).c_str());
		}
		SeedFile                        seeds;
		std::map<std::string, uint64_t> counts;
		// set(args.stages.split(',')).
		bool compute = false, graphics = false;
		for (size_t at = 0;;) {
			const auto comma = options.stages.find(',', at);
			const auto stage = options.stages.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
			compute          = compute || stage == "cs";
			graphics         = graphics || stage == "gfx";
			if (comma == std::string::npos) break;
			at = comma + 1;
		}

		if (compute) {
			for (const auto& program: inventory.programs) {
				for (const auto* header: program.headers) {
					if (header->type != TypeCs) continue;
					try {
						seeds.Record(ComputeRecord(program.code, *header));
					} catch (const Refused& error) {
						throw Fatal(std::string("a compute shader: ") + error.what()); // not caught there
					}
					counts["compute programs"]++;
				}
			}
		}

		// The pipelines of a (VS, PS) pair with each state of its group; a pair whose vertex shader is refused
		// has none (the vertex shader's record is kept when only the pixel shader's is refused, as in Python).
		const auto add_pipelines = [&](const Shader& vertex, const Shader& pixel, const std::string& group,
		                               bool culling) {
			const auto* palette = states.Find(group);
			if (palette == nullptr) {
				counts["pairs without states (" + group + ")"]++;
				return;
			}
			uint32_t vs = 0, ps = 0;
			try {
				vs = seeds.Record(VertexRecord(vertex.code, vertex.agc));
				ps = seeds.Record(PixelRecord(pixel.code, pixel.agc, vertex.agc));
			} catch (const Refused&) {
				counts["pairs with unsupported inputs"]++;
				return;
			}
			for (const auto& words: *palette) {
				if (!culling) {
					seeds.Pipeline(WithShaders(vs, ps, words));
					continue;
				}
				for (const auto& variant: CullingVariants(words)) seeds.Pipeline(WithShaders(vs, ps, variant));
			}
		};

		if (graphics) {
			std::set<std::tuple<uint32_t, uint32_t, int64_t>> done; // (VS code, PS code, pass id or -1)
			for (const auto& drawn: GraphicsPairs(inventory)) {
				const auto vs = inventory.ProgramOf(drawn.vertex->code), ps = inventory.ProgramOf(drawn.pixel->code);
				if (!done.emplace(vs, ps, drawn.pass ? int64_t {*drawn.pass} : int64_t {-1}).second) continue;
				const auto group = drawn.pass ? std::to_string(*drawn.pass) : std::string("None"); // str(pass_id)
				add_pipelines(*drawn.vertex, *drawn.pixel, group, true);
			}
			for (const auto& drawn: EnginePairs(inventory)) add_pipelines(*drawn.vertex, *drawn.pixel, "engine", false);
			// Inventory.vertex_only: eboot.bin's vertex shaders, which the engine also draws without a pixel shader.
			for (const auto& shader: inventory.embedded) {
				if (shader.agc.type != TypeGs) continue;
				uint32_t vs = 0;
				try {
					vs = seeds.Record(VertexRecord(shader.code, shader.agc));
				} catch (const Refused& error) {
					throw Fatal(std::string("eboot.bin: ") + error.what()); // not caught there
				}
				for (const auto& words: states.VertexOnly()) seeds.Pipeline(WithShaders(vs, NoShader, words));
			}
		}

		if (options.limit != 0) {
			// A smoke test: the first programs (records[:limit]) and the pipelines that only use them.
			const int64_t keep  = options.limit;
			const auto    count = static_cast<int64_t>(seeds.records.size());
			const auto    kept  = keep >= 0 ? std::min(keep, count) : std::max<int64_t>(count + keep, 0);
			seeds.records.resize(static_cast<size_t>(kept));
			std::erase_if(seeds.pipelines, [&](const std::vector<uint32_t>& words) {
				return (words[0] != NoShader && words[0] >= keep) || (words[1] != NoShader && words[1] >= keep);
			});
		}

		const auto size    = seeds.Write(options.out);
		const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
		// The programs by stage, in order of first appearance (a Counter).
		std::vector<std::pair<std::string, uint64_t>> by_stage;
		for (const auto& words: seeds.records) {
			const std::string name  = StageName(words[0]);
			const auto        found = std::find_if(by_stage.begin(), by_stage.end(),
			                                       [&](const auto& stage) { return stage.first == name; });
			if (found == by_stage.end()) {
				by_stage.emplace_back(name, 1);
			} else {
				found->second++;
			}
		}
		std::string stage_list;
		for (const auto& [name, count]: by_stage) {
			stage_list += (stage_list.empty() ? "'" : ", '") + name + "': " + std::to_string(count);
		}
		std::printf("%s: %zu programs {%s}, %zu graphics pipelines, %.1f MiB, %.0f s\n", Utf8(options.out).c_str(),
		            seeds.records.size(), stage_list.c_str(), seeds.pipelines.size(),
		            static_cast<double>(size) / (1024.0 * 1024.0), seconds);
		for (const auto& [what, count]: counts) {
			std::printf("  %s: %llu\n", what.c_str(), static_cast<unsigned long long>(count));
		}
		std::fflush(stdout);
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_shader_precompile --make-seeds: %s\n", error.what());
		return 1;
	}
}

} // namespace StaticSeeds
