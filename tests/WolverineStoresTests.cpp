#include "graphics/host_gpu/renderer/pipeline/wolverineStores.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

namespace {

using namespace Libs::Graphics::WolverineStores;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WolverineStoresTests: failed: %s\n", message);
		std::abort();
	}
}

constexpr uint64_t TableBase    = 0x100000;
constexpr uint64_t RegistryBase = 0x200000;
constexpr uint64_t ManagerBase  = 0x300000;
constexpr uint64_t ListBase     = 0x400000;
constexpr uint64_t HeapBase     = 0x500000;

// Guest memory in miniature, with the chain's tables.
struct Memory {
	std::map<uint64_t, uint32_t> dwords;
	std::set<uint64_t>           gpu_written;

	void Put(uint64_t address, uint32_t value) { dwords[address] = value; }
	bool Read(uint64_t address, void* data, uint64_t bytes) const {
		auto* out = static_cast<uint32_t*>(data);
		for (uint64_t i = 0; i < bytes / 4u; ++i) {
			const auto at = address + i * 4u;
			if (gpu_written.contains(at)) return false;
			const auto found = dwords.find(at);
			out[i]           = found != dwords.end() ? found->second : 0u;
		}
		return true;
	}
	void PutBuffer(uint64_t address, uint64_t base, uint32_t stride, uint32_t count) {
		Put(address, static_cast<uint32_t>(base));
		Put(address + 4u, static_cast<uint32_t>(base >> 32u) | (stride << 16u));
		Put(address + 8u, count);
	}
	// A material's place, its slot and the image in the slot.
	void PutMaterial(uint32_t material, uint32_t place, uint32_t slot, uint64_t image, uint32_t slot_field = 0x94u) {
		Put(RegistryBase + material * 96ull + 28u, place);
		Put(ManagerBase + place * 16ull + slot_field, slot);
		Put(HeapBase + slot * 32ull, static_cast<uint32_t>(image >> 8u));
		Put(HeapBase + slot * 32ull + 4u, static_cast<uint32_t>(image >> 40u));
	}
	void PutRecord(uint32_t record, uint32_t stride, uint32_t material) {
		Put(ListBase + uint64_t {record} * stride + 8u, (material << 18u) | 0x123u);
	}
};

Memory MakeMemory() {
	Memory memory;
	memory.PutBuffer(TableBase + 1600u, RegistryBase, 96u, 1000u);
	memory.PutBuffer(TableBase + 1376u, ManagerBase, 16u, 4096u);
	memory.PutMaterial(0u, 7u, 70u, 0x700000000ull);
	memory.PutMaterial(591u, 9u, 18596u, 0x900000000ull);
	memory.PutMaterial(170u, 11u, 18579u, 0xb00000000ull);
	memory.PutRecord(0u, 16u, 591u);
	return memory;
}

// The registers the vertex program reads (user data from register 8), with its table's pointer in guest memory.
std::vector<uint32_t> VertexUser(Memory& memory, uint32_t records, uint32_t stride = 16u) {
	std::vector<uint32_t> user(31u);
	const auto            put = [&](uint32_t reg, uint64_t base, uint32_t buffer_stride, uint32_t count) {
        user[reg - 8u]      = static_cast<uint32_t>(base);
        user[reg - 8u + 1u] = static_cast<uint32_t>(base >> 32u) | (buffer_stride << 16u);
        user[reg - 8u + 2u] = count;
	};
	put(8u, ListBase, stride, records);
	put(12u, HeapBase, 0u, 0x100000u);
	const uint64_t pointer = 0x600000;
	user[36u - 8u]         = static_cast<uint32_t>(pointer);
	user[37u - 8u]         = 0u;
	memory.PutBuffer(pointer, TableBase, 0u, 0x10000u);
	return user;
}

std::vector<uint32_t> PixelUser(uint32_t records, uint32_t stride = 16u, uint32_t table = 28u) {
	std::vector<uint32_t> user(table + 4u);
	const auto            put = [&](uint32_t reg, uint64_t base, uint32_t buffer_stride, uint32_t count) {
        user[reg]      = static_cast<uint32_t>(base);
        user[reg + 1u] = static_cast<uint32_t>(base >> 32u) | (buffer_stride << 16u);
        user[reg + 2u] = count;
	};
	put(0u, ListBase, stride, records);
	put(4u, HeapBase, 0u, 0x100000u);
	put(table, TableBase, 0u, 0x10000u);
	return user;
}

bool Find(uint64_t program, const std::vector<uint32_t>& user, uint32_t base, const Memory& memory, Reach& reach) {
	return FindReach(program, std::span<const uint32_t>(user), base,
	                 [&](uint64_t address, void* data, uint64_t bytes) { return memory.Read(address, data, bytes); }, reach);
}

void TestReach() {
	auto  memory = MakeMemory();
	Reach reach;

	// The list's material and material 0, in both stages.
	const auto vertex = VertexUser(memory, 1u);
	Check(Find(VertexProgram, vertex, 8u, memory, reach), "the vertex stage's reach is known");
	Check(reach.count == 2u && reach.Holds(0x900000000ull) && reach.Holds(0x700000000ull), "the vertex stage reaches its material and material 0");
	const auto pixel = PixelUser(1u);
	Check(Find(PixelProgram, pixel, 0u, memory, reach), "the pixel stage's reach is known");
	Check(reach.count == 2u && reach.Holds(0x900000000ull) && reach.Holds(0x700000000ull), "the pixel stage reaches its material and material 0");

	// Every record counts, whatever the draw's first instance: the shader's index isn't read.
	memory.PutRecord(1u, 16u, 170u);
	Check(Find(PixelProgram, PixelUser(2u), 0u, memory, reach), "two records are known");
	Check(reach.count == 3u && reach.Holds(0xb00000000ull), "the second record counts");

	// The same image twice is one address, and a null T# none.
	memory.PutRecord(1u, 16u, 591u);
	Check(Find(PixelProgram, PixelUser(2u), 0u, memory, reach) && reach.count == 2u, "the same image counts once");
	memory.PutMaterial(0u, 7u, 71u, 0u);
	Check(Find(PixelProgram, PixelUser(1u), 0u, memory, reach) && reach.count == 1u, "a null T# is no image");
	memory.PutMaterial(0u, 7u, 70u, 0x700000000ull);

	// A material past the registry reads place 0, and a place past the manager reads slot 0, as a hardware load does.
	memory.PutMaterial(0u, 0u, 70u, 0x700000000ull);
	memory.PutRecord(0u, 16u, 5000u);
	Check(Find(PixelProgram, PixelUser(1u), 0u, memory, reach) && reach.count == 1u && reach.Holds(0x700000000ull), "a material past the registry reads place 0");
	memory = MakeMemory();
	memory.PutMaterial(592u, 900000u, 18596u, 0xc00000000ull);
	memory.Put(HeapBase, 0xd000000u);
	memory.PutRecord(0u, 16u, 592u);
	Check(Find(PixelProgram, PixelUser(1u), 0u, memory, reach) && reach.Holds(0xd00000000ull) && !reach.Holds(0xc00000000ull), "a place past the manager reads slot 0");

	// Not known: too long a list, no stride, too little user data, or a byte the GPU may have written.
	memory = MakeMemory();
	Check(!Find(PixelProgram, PixelUser(MaxRecords + 1u), 0u, memory, reach), "a long list isn't known");
	Check(Find(PixelProgram, PixelUser(MaxRecords), 0u, memory, reach), "the longest list that is known");
	Check(!Find(PixelProgram, PixelUser(1u, 0u), 0u, memory, reach), "a list with no stride isn't known");
	Check(!Find(PixelProgram, std::vector<uint32_t>(31u), 0u, memory, reach), "too little pixel user data");
	auto small = VertexUser(memory, 1u);
	small.resize(30u);
	Check(!Find(VertexProgram, small, 8u, memory, reach), "too little vertex user data");
	const auto check_written = [&](uint64_t address, const char* message) {
		auto written = MakeMemory();
		written.gpu_written.insert(address);
		Check(!Find(PixelProgram, PixelUser(1u), 0u, written, reach), message);
	};
	check_written(TableBase + 1600u, "the registry's descriptor");
	check_written(TableBase + 1376u, "the manager's descriptor");
	check_written(ListBase + 8u, "the record");
	check_written(RegistryBase + 591u * 96ull + 28u, "the place");
	check_written(ManagerBase + 9u * 16ull + 0x94u, "the slot");
	check_written(HeapBase + 18596u * 32ull, "the T#");
	check_written(RegistryBase + 28u, "material 0's place");
}

struct Value {
	std::array<uint32_t, 8> dwords {};
	uint32_t                dword_count = 0;
	bool                    operator==(const Value&) const = default;
};

Value MakeValue(uint64_t address, uint32_t format = 0u) {
	Value value;
	value.dwords[0]   = static_cast<uint32_t>(address >> 8u);
	value.dwords[1]   = static_cast<uint32_t>(address >> 40u) | (format << 20u);
	value.dwords[7]   = 0xabcdu;
	value.dword_count = 8u;
	return value;
}

void TestNarrow() {
	const std::vector<Value> values = {MakeValue(0x100000000ull), MakeValue(0x200000000ull), MakeValue(0x200000000ull, 5u), MakeValue(0x300000000ull), {}};
	Reach                    reach;
	reach.Add(0x200000000ull);
	std::vector<Value>    narrowed;
	std::vector<uint32_t> kept;
	Narrow(std::span<const Value>(values), reach, narrowed, kept);
	Value null;
	null.dword_count = 8u;

	// Every candidate over the image stays, in its own format, and the others are null.
	Check(narrowed.size() == values.size(), "one value for each candidate");
	Check(narrowed[1] == values[1] && narrowed[2] == values[2], "both candidates over the image stay");
	Check(narrowed[0] == null && narrowed[3] == null && narrowed[4] == null, "the others are null");
	Check(kept == std::vector<uint32_t>({1u, 2u}), "the ranks kept");

	// Next draw, another image: the old ranks go null and the new one stays.
	reach = {};
	reach.Add(0x300000000ull);
	Narrow(std::span<const Value>(values), reach, narrowed, kept);
	Check(narrowed[1] == null && narrowed[2] == null && narrowed[3] == values[3], "the next draw's image replaces the last");
	Check(kept == std::vector<uint32_t>({3u}), "the ranks kept then");

	// A kept candidate's new descriptor shows.
	auto changed = values;
	changed[3].dwords[7] = 0x1234u;
	Narrow(std::span<const Value>(changed), reach, narrowed, kept);
	Check(narrowed[3] == changed[3], "a kept candidate's descriptor follows the table");

	// Another size starts over.
	const std::vector<Value> fewer = {MakeValue(0x300000000ull)};
	Narrow(std::span<const Value>(fewer), reach, narrowed, kept);
	Check(narrowed.size() == 1u && narrowed[0] == fewer[0] && kept == std::vector<uint32_t>({0u}), "another size starts over");
	Narrow(std::span<const Value>(), reach, narrowed, kept);
	Check(narrowed.empty() && kept.empty(), "no values");
}

// Place 103's smoke-trail pixel programs: their table in register 20; A's slot at +0x84, B's at +0x94.
void TestTrailReach() {
	auto memory = MakeMemory();
	// Material 591's place 9 holds another slot at +0x84, so A and B reach different images.
	memory.PutMaterial(591u, 9u, 18600u, 0xa00000000ull, 0x84u);
	memory.PutMaterial(0u, 7u, 71u, 0x710000000ull, 0x84u);
	Reach reach;
	const auto user = PixelUser(1u, 16u, 20u);
	Check(Find(TrailPixelB, user, 0u, memory, reach) && reach.count == 2u && reach.Holds(0x900000000ull) &&
	          reach.Holds(0x700000000ull),
	      "trail B reads its slot at +0x94 through the table in register 20");
	Check(Find(TrailPixelA, user, 0u, memory, reach) && reach.count == 2u && reach.Holds(0xa00000000ull) &&
	          reach.Holds(0x710000000ull),
	      "trail A reads its slot at +0x84");
	Check(!Find(TrailPixelA, std::vector<uint32_t>(23u), 0u, memory, reach), "too little trail user data");
	Check(!Find(PixelProgram, user, 0u, memory, reach), "the particle program needs user data up to its table in register 28");
	Check(!Find(0x3de16049649d9ad6ull, user, 0u, memory, reach), "no chain for another program");
	// The particle program still reads register 28 and +0x94.
	Check(Find(PixelProgram, PixelUser(1u), 0u, memory, reach) && reach.Holds(0x900000000ull), "the particle chain is unchanged");
}

} // namespace

int main() {
	Check(IsStoreProgram(VertexProgram) && IsStoreProgram(PixelProgram) && IsStoreProgram(TrailPixelA) &&
	          IsStoreProgram(TrailPixelB) && !IsStoreProgram(0x3de16049649d9ad6ull),
	      "the store programs");
	TestReach();
	TestTrailReach();
	TestNarrow();
	std::puts("WolverineStoresTests: all passed");
	return 0;
}
