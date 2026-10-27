#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/superResolution.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace Libs {

namespace LibPsml {

LIB_VERSION("Psml", 1, "Psml", 1, 1);

namespace Psml {

// Sony's PS5 Pro super resolution (PSSR, "Mfsr"). Demon's Souls 01.007.000 calls it from SuperResolution.cpp
// (made only where sceKernelIsTrinityMode says PS5 Pro; loader/demonsSoulsSuperResolution.h makes it here):
// InitSystem = initialize, main memory requirements, shared resources; InitContext(3840, 2160) = context memory
// requirements, create context; Apply each frame in place of the TAA = the game's wrapper asks the dispatch size,
// reserves that many dwords in its draw command buffer and calls dispatch with {cursor, dwords}. The objects the game
// keeps are 64-bit handles; here pointers to Object.

namespace Pm4 = Graphics::Pm4; // (KYTY_PM4)

constexpr int Error(uint32_t value) {
	return static_cast<int>(value);
}

constexpr int PSML_ERROR_NOT_INITIALIZED = Error(0x8a810001u);
constexpr int PSML_ERROR_INVALID_OBJECT  = Error(0x8a810005u);
constexpr int PSML_ERROR_OUT_OF_SPACE    = Error(0x8a810006u); // (the game's wrapper's own, for too few dwords)
constexpr int PSML_ERROR_INVALID_POINTER = Error(0x8a810009u);
constexpr int PSML_ERROR_INVALID_VALUE   = Error(0x8a81000du);
constexpr int PSML_ERROR_NULL_OBJECT     = Error(0x8a810014u);

constexpr uint32_t SHARED_RESOURCES_MAGIC = 0xa9c4;
constexpr uint32_t CONTEXT_MAGIC          = 0x9231;

constexpr uint64_t MAIN_MEMORY_BLOCK_SIZE = 0x200000;
constexpr uint64_t MAIN_MEMORY_ALIGNMENT  = 0x200000;
constexpr uint64_t EXTRA_VA_BYTES         = 0x600000;

struct MainMemoryRequirements {
	uint64_t block_size;
	uint64_t alignment;
	uint64_t block_count;
};

struct MainMemoryParameters {
	uint32_t type;
	uint32_t reserved;
};

struct DirectMemoryBlock {
	uint64_t address;
	uint64_t size;
};

struct SharedResourcesInitParameters {
	uint32_t                 type;
	uint32_t                 reserved;
	const DirectMemoryBlock* blocks;
	uint64_t                 block_count;
	uint64_t                 virtual_address_start;
};

// The game's (InitContext): +0x08 blocks, +0x10 block count, +0x18/+0x1a the output width and height (16 bits),
// +0x20 and +0x28 a {4, width, height} triple, +0x38 the shared resources.
struct ContextInitParameters {
	uint32_t                 type;
	uint32_t                 reserved;
	const DirectMemoryBlock* blocks;
	uint64_t                 block_count;
	uint16_t                 width;
	uint16_t                 height;
	uint32_t                 reserved2;
	const uint32_t*          size_a;
	const uint32_t*          size_b;
	uint64_t                 reserved3;
	uint64_t                 shared_resources;
};
static_assert(offsetof(ContextInitParameters, width) == 0x18 && offsetof(ContextInitParameters, shared_resources) == 0x38);

// What the dispatch writes into: the game's reservation in its draw command buffer.
struct CommandSpan {
	uint32_t* commands;
	uint32_t  dwords;
};

struct Object {
	uint32_t magic  = 0;
	uint32_t number = 0;
	uint32_t width = 0, height = 0; // a context's output
};

static_assert(sizeof(MainMemoryRequirements) == 24);
static_assert(sizeof(DirectMemoryBlock) == 16);

static bool                  g_initialized      = false;
static uint32_t              g_supported_modes  = 0x3;
static std::atomic<int32_t>  g_shared_ref_count = 0;
static std::atomic<uint32_t> g_contexts         = 0;

static uint64_t BaseVaBytes(uint32_t type) {
	// The native implementation indexes this table before adding one 6 MiB range.
	switch (type) {
		case 0: return 0x06200000;
		case 1: return 0x18200000;
		case 2: return 0x12200000;
		default: return 0;
	}
}

static bool IsSupportedType(uint32_t type) {
	switch (type) {
		case 0: return true;
		case 1: return (g_supported_modes & 0x1u) != 0;
		case 2: return (g_supported_modes & 0x2u) != 0;
		default: return false;
	}
}

static uint64_t RequiredBlockCount(uint32_t type) {
	const auto base = BaseVaBytes(type);
	if (base == 0) {
		return 0;
	}
	return (base + EXTRA_VA_BYTES) / MAIN_MEMORY_BLOCK_SIZE;
}

static int CheckInitialized() {
	return (g_initialized ? OK : PSML_ERROR_NOT_INITIALIZED);
}

static const Object* AsObject(uint64_t handle, uint32_t magic) {
	const auto* object = reinterpret_cast<const Object*>(handle);
	return object != nullptr && object->magic == magic ? object : nullptr;
}

static bool IsObject(uint64_t handle) {
	return AsObject(handle, SHARED_RESOURCES_MAGIC) != nullptr || AsObject(handle, CONTEXT_MAGIC) != nullptr;
}

static int KYTY_SYSV_ABI PsmlInitialize() {
	PRINT_NAME();

	g_initialized      = true;
	g_supported_modes  = 0x3;
	g_shared_ref_count = 0;

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetMainMemoryRequirements(MainMemoryRequirements*     out,
                                                       const MainMemoryParameters* params) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr || params == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (!IsSupportedType(params->type)) {
		return PSML_ERROR_INVALID_VALUE;
	}

	out->block_size  = MAIN_MEMORY_BLOCK_SIZE;
	out->alignment   = MAIN_MEMORY_ALIGNMENT;
	out->block_count = RequiredBlockCount(params->type);

	return OK;
}

static int KYTY_SYSV_ABI PsmlCreateSharedResources(uint64_t* out, const SharedResourcesInitParameters* params) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr || params == nullptr || params->blocks == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (params->reserved != 0 || !IsSupportedType(params->type) ||
	    params->block_count < RequiredBlockCount(params->type)) {
		return PSML_ERROR_INVALID_VALUE;
	}

	auto* object  = new Object;
	object->magic = SHARED_RESOURCES_MAGIC;
	*out          = reinterpret_cast<uint64_t>(object);
	g_shared_ref_count.fetch_add(1, std::memory_order_relaxed);

	return OK;
}

static int KYTY_SYSV_ABI PsmlDestroySharedResources(uint64_t handle) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (handle == 0) {
		return PSML_ERROR_NULL_OBJECT;
	}
	if (AsObject(handle, SHARED_RESOURCES_MAGIC) == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	g_shared_ref_count.fetch_sub(1, std::memory_order_relaxed);

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetContextMemoryRequirements(MainMemoryRequirements* out, const void* /*params*/) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	out->block_size  = MAIN_MEMORY_BLOCK_SIZE;
	out->alignment   = MAIN_MEMORY_ALIGNMENT;
	out->block_count = 1;

	return OK;
}

static int KYTY_SYSV_ABI PsmlCreateContext(uint64_t* out, const ContextInitParameters* params) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr || params == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (AsObject(params->shared_resources, SHARED_RESOURCES_MAGIC) == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	auto* object   = new Object;
	object->magic  = CONTEXT_MAGIC;
	object->number = g_contexts.fetch_add(1, std::memory_order_relaxed);
	object->width  = params->width;
	object->height = params->height;
	*out           = reinterpret_cast<uint64_t>(object);
	std::printf("Super resolution: context %u for a %ux%u output\n", object->number, object->width, object->height);
	std::fflush(stdout);

	return OK;
}

static int KYTY_SYSV_ABI PsmlDestroyContext(uint64_t handle) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (AsObject(handle, CONTEXT_MAGIC) == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	return OK;
}

// The dwords a dispatch writes: one R_SUPER_RESOLUTION packet.
static int KYTY_SYSV_ABI PsmlGetDispatchSize(uint64_t handle, uint32_t* out_dwords) {
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (AsObject(handle, CONTEXT_MAGIC) == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (out_dwords == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	*out_dwords = Graphics::SuperResolution::PacketDwords;

	return OK;
}

// The game's parameter block (SuperResolution::Apply) copied into the packet, with the T#s its first fields point to.
static int KYTY_SYSV_ABI PsmlDispatch(uint64_t handle, const CommandSpan* span, const uint8_t* params) {
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	const auto* context = AsObject(handle, CONTEXT_MAGIC);
	if (context == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (span == nullptr || span->commands == nullptr || params == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (span->dwords < Graphics::SuperResolution::PacketDwords) {
		return PSML_ERROR_OUT_OF_SPACE;
	}

	Graphics::SuperResolution::Packet packet {};
	packet.context       = context->number;
	packet.output_width  = context->width;
	packet.output_height = context->height;
	std::memcpy(packet.parameters, params, sizeof(packet.parameters));
	for (uint32_t slot = 0; slot < Graphics::SuperResolution::TextureSlots; slot++) {
		const uint32_t* descriptor = nullptr;
		std::memcpy(&descriptor, params + slot * sizeof(uint64_t), sizeof(descriptor));
		if (descriptor == nullptr) continue;
		std::memcpy(packet.textures[slot], descriptor, sizeof(packet.textures[slot]));
		packet.textures_given |= 1u << slot;
	}
	auto* commands = span->commands;
	// (The game's reservation is exactly the dispatch size.)
	commands[0] = KYTY_PM4(Graphics::SuperResolution::PacketDwords, Graphics::Pm4::IT_NOP, Graphics::Pm4::R_SUPER_RESOLUTION);
	std::memcpy(commands + 1, &packet, sizeof(packet));
	const auto rest = span->dwords - Graphics::SuperResolution::PacketDwords;
	if (rest == 1) {
		commands[Graphics::SuperResolution::PacketDwords] = 0x80000000u; // (a type-2 filler)
	} else if (rest >= 2) {
		commands[Graphics::SuperResolution::PacketDwords] = KYTY_PM4(rest, Graphics::Pm4::IT_NOP, Graphics::Pm4::R_ZERO);
	}

	return OK;
}

// 01.005.000's dispatch (RUNLFro+qok) takes an older parameter block (by @Al3cko, issue #87; the native wrapper at
// libScePsml+0x2c240): the descriptors and the projection where 1.07 has them, the jitter, motion scale and reset
// eight bytes earlier, and the block ends at 0x148. Copied into 1.07's layout, the packet is the same.
static int KYTY_SYSV_ABI PsmlDispatchLegacy(uint64_t handle, const CommandSpan* span, const uint8_t* params) {
	if (params == nullptr) return PSML_ERROR_INVALID_POINTER;
	uint8_t normalized[Graphics::SuperResolution::ParameterBytes] {};
	const auto copy = [&](size_t to, size_t from, size_t bytes) { std::memcpy(normalized + to, params + from, bytes); };
	copy(0x000, 0x000, 0x40);
	copy(0x048, 0x040, 0x10);
	copy(0x058, 0x058, 0xa0);
	copy(0x100, 0x0f8, 0x1d);
	uint32_t flags = 0;
	std::memcpy(&flags, params + 0x118, sizeof(flags));
	flags &= 0x80008000u;
	std::memcpy(normalized + 0x120, &flags, sizeof(flags));
	copy(0x124, 0x11c, 4);
	copy(0x130, 0x128, 4);
	const float one = 1.0f;
	std::memcpy(normalized + 0x134, &one, sizeof(one));
	copy(0x138, 0x12c, 8);
	copy(0x140, 0x134, 4);
	copy(0x144, 0x13c, 4);
	copy(0x14c, 0x13c, 12);
	return PsmlDispatch(handle, span, normalized);
}

// The game's texture LOD bias for the input size `width` x `height`: it adds 0.5 and keeps it at most 0, so this
// gives it the size's ratio to the output in powers of two (DLSS' advice: -0.49 at 1440 for 2160).
static int KYTY_SYSV_ABI PsmlGetMipBias(uint64_t handle, float* out_bias, uint32_t width, uint32_t height) {
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	const auto* context = AsObject(handle, CONTEXT_MAGIC);
	if (context == nullptr) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (out_bias == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	(void)width;
	*out_bias = height != 0 && context->height != 0
	                ? static_cast<float>(std::log2(static_cast<double>(height) / static_cast<double>(context->height)) - 0.5)
	                : 0.0f;

	return OK;
}

static int KYTY_SYSV_ABI PsmlRequestCapture(uint64_t handle) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!IsObject(handle)) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	return OK;
}

static int KYTY_SYSV_ABI PsmlValidateObject(uint64_t handle) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!IsObject(handle)) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	return OK;
}

} // namespace Psml

LIB_DEFINE(InitPsml_1) {
	LIB_FUNC("3WVD91e12ZQ", Psml::PsmlInitialize);
	LIB_FUNC("+2KpvixvL6E", Psml::PsmlGetMainMemoryRequirements);
	LIB_FUNC("eWoKNeB6V-k", Psml::PsmlCreateSharedResources);
	LIB_FUNC("jEevBXmagOQ", Psml::PsmlDestroySharedResources);
	LIB_FUNC("VGjrQa-WqdU", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("EO9YQXmJEN8", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("IMlj247LdTo", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("sjTyYheKTrU", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("ArakEpzsZo0", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("fccGInHrj8A", Psml::PsmlCreateContext);
	// 01.005.000's context creation: the same blocks, sizes and shared-resource offsets (its native wrapper at 0x2b770).
	LIB_FUNC("gxv3i+MTEzU", Psml::PsmlCreateContext);
	LIB_FUNC("JaLBe0P3jSU", Psml::PsmlDestroyContext);
	LIB_FUNC("AHalTX9wFZY", Psml::PsmlGetDispatchSize);
	LIB_FUNC("94iBp3KvIuI", Psml::PsmlDispatch);
	LIB_FUNC("RUNLFro+qok", Psml::PsmlDispatchLegacy);
	LIB_FUNC("GHna9-DvnUk", Psml::PsmlGetMipBias);
	LIB_FUNC("GJY0MvuTcs8", Psml::PsmlRequestCapture);
	LIB_FUNC("LXq+6mIxpCw", Psml::PsmlValidateObject);
	LIB_FUNC("FSGaTQze0UY", Psml::PsmlValidateObject);
}

} // namespace LibPsml

} // namespace Libs
