#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WOLVERINESTORES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WOLVERINESTORES_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

// Marvel's Wolverine's particle programs each have one image store, through a key read from a table the whole heap could
// be named by, so its candidates are every storage image in the heap: about 13,000 a stage, resolved and bound at every
// effect draw. What the key can be is in guest memory, though. From the disassembly of VS 311b7ad8a3a9382c (file
// fc94abf5b9e0e8e4) and PS b13fb3b0503c4b1b (file 2464e975d88b7f15):
//
//   list record, dword +8 >> 18  = material
//   registry (table +1600)       material * 96 + 28 = place
//   manager (table +1376)        place * 16 + 0x94  = slot
//   heap                         slot * 32          = the T# stored to
//
// The record's index comes from the draw (VS) or a flat input (PS). A hardware load past a buffer's size returns 0, so
// reading all of the list, and material 0, covers every index.
namespace Libs::Graphics::WolverineStores {

inline constexpr uint64_t VertexProgram = 0x311b7ad8a3a9382cull;
inline constexpr uint64_t PixelProgram  = 0xb13fb3b0503c4b1bull;
inline constexpr uint32_t MaxRecords    = 16;

[[nodiscard]] constexpr bool IsStoreProgram(uint64_t hash) {
	return hash == VertexProgram || hash == PixelProgram;
}

// The images a draw's store can reach, by address.
struct Reach {
	std::array<uint64_t, MaxRecords + 1u> addresses {};
	uint32_t                              count = 0;

	[[nodiscard]] bool Holds(uint64_t address) const {
		return std::find(addresses.begin(), addresses.begin() + count, address) != addresses.begin() + count;
	}
	void Add(uint64_t address) {
		if (address != 0u && !Holds(address)) addresses[count++] = address;
	}
};

[[nodiscard]] constexpr uint64_t BufferBase(const uint32_t* v) {
	return uint64_t {v[0]} | (uint64_t {v[1] & 0xffffu} << 32u);
}
[[nodiscard]] constexpr uint32_t BufferStride(const uint32_t* v) {
	return (v[1] >> 16u) & 0x3fffu;
}
[[nodiscard]] constexpr uint64_t ImageAddress(uint32_t dword0, uint32_t dword1) {
	return (uint64_t {dword0} | (uint64_t {dword1 & 0xffu} << 32u)) << 8u;
}

// False when it can't be known: a read failed (`read(address, data, bytes)` returns false), the list is longer than
// MaxRecords or it has no stride. The caller then binds every candidate. `read` must hand back the bytes as the GPU left
// them, not what guest memory holds of bytes the GPU may have written.
template <typename Read>
[[nodiscard]] bool FindReach(uint64_t program, std::span<const uint32_t> user_data, uint32_t user_data_base, Read&& read, Reach& reach) {
	const bool vertex = program == VertexProgram;
	reach.count       = 0;
	if (user_data.size() + user_data_base < (vertex ? 39u : 32u) || user_data_base > (vertex ? 8u : 0u)) return false;
	const auto reg = [&](uint32_t r) { return user_data.data() + (r - user_data_base); };
	uint32_t   global[4] {};
	if (vertex) {
		if (!read(uint64_t {*reg(36)} | (uint64_t {*reg(37) & 0xffffu} << 32u), global, sizeof(global))) return false;
	} else {
		std::copy_n(reg(28), 4, global);
	}
	uint32_t registry[4] {}, manager[4] {};
	if (!read(BufferBase(global) + 1600u, registry, sizeof(registry)) || !read(BufferBase(global) + 1376u, manager, sizeof(manager))) return false;
	const uint32_t* list = reg(vertex ? 8u : 0u);
	const uint32_t* heap = reg(vertex ? 12u : 4u);
	const auto      add  = [&](uint32_t material) {
        uint32_t place = 0, slot = 0, image[2] {};
        if (material < registry[2] && !read(BufferBase(registry) + material * 96ull + 28u, &place, sizeof(place))) return false;
        if (place * 16ull + 0x98u <= uint64_t {manager[2]} * std::max(BufferStride(manager), 1u) &&
            !read(BufferBase(manager) + place * 16ull + 0x94u, &slot, sizeof(slot))) {
            return false;
        }
        if (!read(BufferBase(heap) + slot * 32ull, image, sizeof(image))) return false;
        reach.Add(ImageAddress(image[0], image[1]));
        return true;
	};
	const auto stride = BufferStride(list);
	if (stride < 12u || list[2] > MaxRecords) return false;
	for (uint32_t record = 0; record < list[2]; ++record) {
		uint32_t material = 0;
		if (!read(BufferBase(list) + uint64_t {record} * stride + 8u, &material, sizeof(material)) || !add(material >> 18u)) return false;
	}
	return add(0u);
}

// `narrowed` becomes `values` with every candidate over another image a null descriptor. `kept` lists the ranks that aren't
// null, so that the next call only touches those and the candidates that are in reach then.
template <typename Value>
void Narrow(std::span<const Value> values, const Reach& reach, std::vector<Value>& narrowed, std::vector<uint32_t>& kept) {
	Value null {};
	if (!values.empty()) null.dword_count = values[0].dword_count;
	if (narrowed.size() != values.size()) {
		narrowed.assign(values.size(), null);
	} else {
		for (const auto rank: kept) narrowed[rank] = null;
	}
	kept.clear();
	for (uint32_t rank = 0; rank < values.size(); ++rank) {
		const auto& value = values[rank];
		if (!reach.Holds(ImageAddress(value.dwords[0], value.dwords[1]))) continue;
		narrowed[rank] = value;
		kept.push_back(rank);
	}
}

} // namespace Libs::Graphics::WolverineStores

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WOLVERINESTORES_H_
