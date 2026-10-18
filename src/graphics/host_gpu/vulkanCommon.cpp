#include "graphics/host_gpu/vulkanCommon.h"

#include "graphics/host_gpu/spirvLocalArrays.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "common/assert.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <array>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {

struct FormatMapping {
	Prospero::BufferFormat guest;
	vk::Format             host;
};

constexpr FormatMapping kFormatMappings[] = {
    {Prospero::BufferFormat::k8UNorm, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8SNorm, vk::Format::eR8Snorm},
    {Prospero::BufferFormat::k8UScaled, vk::Format::eR8Uscaled},
    {Prospero::BufferFormat::k8SScaled, vk::Format::eR8Sscaled},
    {Prospero::BufferFormat::k8UInt, vk::Format::eR8Uint},
    {Prospero::BufferFormat::k8SInt, vk::Format::eR8Sint},
    {Prospero::BufferFormat::k16UNorm, vk::Format::eR16Unorm},
    {Prospero::BufferFormat::k16SNorm, vk::Format::eR16Snorm},
    {Prospero::BufferFormat::k16UScaled, vk::Format::eR16Uscaled},
    {Prospero::BufferFormat::k16SScaled, vk::Format::eR16Sscaled},
    {Prospero::BufferFormat::k16UInt, vk::Format::eR16Uint},
    {Prospero::BufferFormat::k16SInt, vk::Format::eR16Sint},
    {Prospero::BufferFormat::k16Float, vk::Format::eR16Sfloat},
    {Prospero::BufferFormat::k8_8UNorm, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8SNorm, vk::Format::eR8G8Snorm},
    {Prospero::BufferFormat::k8_8UScaled, vk::Format::eR8G8Uscaled},
    {Prospero::BufferFormat::k8_8SScaled, vk::Format::eR8G8Sscaled},
    {Prospero::BufferFormat::k8_8UInt, vk::Format::eR8G8Uint},
    {Prospero::BufferFormat::k8_8SInt, vk::Format::eR8G8Sint},
    {Prospero::BufferFormat::k32UInt, vk::Format::eR32Uint},
    {Prospero::BufferFormat::k32SInt, vk::Format::eR32Sint},
    {Prospero::BufferFormat::k32Float, vk::Format::eR32Sfloat},
    {Prospero::BufferFormat::k16_16UNorm, vk::Format::eR16G16Unorm},
    {Prospero::BufferFormat::k16_16SNorm, vk::Format::eR16G16Snorm},
    {Prospero::BufferFormat::k16_16UScaled, vk::Format::eR16G16Uscaled},
    {Prospero::BufferFormat::k16_16SScaled, vk::Format::eR16G16Sscaled},
    {Prospero::BufferFormat::k16_16UInt, vk::Format::eR16G16Uint},
    {Prospero::BufferFormat::k16_16SInt, vk::Format::eR16G16Sint},
    {Prospero::BufferFormat::k16_16Float, vk::Format::eR16G16Sfloat},
    {Prospero::BufferFormat::k11_11_10Float, vk::Format::eB10G11R11UfloatPack32},
    {Prospero::BufferFormat::k10_10_10_2UNorm, vk::Format::eA2B10G10R10UnormPack32},
    {Prospero::BufferFormat::k10_10_10_2SNorm, vk::Format::eA2B10G10R10SnormPack32},
    {Prospero::BufferFormat::k10_10_10_2UScaled, vk::Format::eA2B10G10R10UscaledPack32},
    {Prospero::BufferFormat::k10_10_10_2UInt, vk::Format::eA2B10G10R10UintPack32},
    {Prospero::BufferFormat::k8_8_8_8UNorm, vk::Format::eR8G8B8A8Unorm},
    {Prospero::BufferFormat::k8_8_8_8SNorm, vk::Format::eR8G8B8A8Snorm},
    {Prospero::BufferFormat::k8_8_8_8UScaled, vk::Format::eR8G8B8A8Uscaled},
    {Prospero::BufferFormat::k8_8_8_8SScaled, vk::Format::eR8G8B8A8Sscaled},
    {Prospero::BufferFormat::k8_8_8_8UInt, vk::Format::eR8G8B8A8Uint},
    {Prospero::BufferFormat::k8_8_8_8SInt, vk::Format::eR8G8B8A8Sint},
    {Prospero::BufferFormat::k32_32UInt, vk::Format::eR32G32Uint},
    {Prospero::BufferFormat::k32_32SInt, vk::Format::eR32G32Sint},
    {Prospero::BufferFormat::k32_32Float, vk::Format::eR32G32Sfloat},
    {Prospero::BufferFormat::k16_16_16_16UNorm, vk::Format::eR16G16B16A16Unorm},
    {Prospero::BufferFormat::k16_16_16_16SNorm, vk::Format::eR16G16B16A16Snorm},
    {Prospero::BufferFormat::k16_16_16_16UScaled, vk::Format::eR16G16B16A16Uscaled},
    {Prospero::BufferFormat::k16_16_16_16SScaled, vk::Format::eR16G16B16A16Sscaled},
    {Prospero::BufferFormat::k16_16_16_16UInt, vk::Format::eR16G16B16A16Uint},
    {Prospero::BufferFormat::k16_16_16_16SInt, vk::Format::eR16G16B16A16Sint},
    {Prospero::BufferFormat::k16_16_16_16Float, vk::Format::eR16G16B16A16Sfloat},
    {Prospero::BufferFormat::k32_32_32UInt, vk::Format::eR32G32B32Uint},
    {Prospero::BufferFormat::k32_32_32SInt, vk::Format::eR32G32B32Sint},
    {Prospero::BufferFormat::k32_32_32Float, vk::Format::eR32G32B32Sfloat},
    {Prospero::BufferFormat::k32_32_32_32UInt, vk::Format::eR32G32B32A32Uint},
    {Prospero::BufferFormat::k32_32_32_32SInt, vk::Format::eR32G32B32A32Sint},
    {Prospero::BufferFormat::k32_32_32_32Float, vk::Format::eR32G32B32A32Sfloat},
    // Narrow-channel sRGB formats are optional in Vulkan. Keep a same-width fallback until
    // sampler-aware sRGB emulation is available.
    {Prospero::BufferFormat::k8Srgb, vk::Format::eR8Unorm},
    {Prospero::BufferFormat::k8_8Srgb, vk::Format::eR8G8Unorm},
    {Prospero::BufferFormat::k8_8_8_8Srgb, vk::Format::eR8G8B8A8Srgb},
    {Prospero::BufferFormat::k9_9_9_5Float, vk::Format::eE5B9G9R9UfloatPack32},
    {Prospero::BufferFormat::k5_6_5UNorm, vk::Format::eB5G6R5UnormPack16},
    {Prospero::BufferFormat::k5_5_5_1UNorm, vk::Format::eA1R5G5B5UnormPack16},
    {Prospero::BufferFormat::k1_5_5_5UNorm, vk::Format::eR5G5B5A1UnormPack16},
    {Prospero::BufferFormat::k4_4_4_4UNorm, vk::Format::eR4G4B4A4UnormPack16},
    {Prospero::BufferFormat::kBc1UNorm, vk::Format::eBc1RgbaUnormBlock},
    {Prospero::BufferFormat::kBc1Srgb, vk::Format::eBc1RgbaSrgbBlock},
    {Prospero::BufferFormat::kBc2UNorm, vk::Format::eBc2UnormBlock},
    {Prospero::BufferFormat::kBc2Srgb, vk::Format::eBc2SrgbBlock},
    {Prospero::BufferFormat::kBc3UNorm, vk::Format::eBc3UnormBlock},
    {Prospero::BufferFormat::kBc3Srgb, vk::Format::eBc3SrgbBlock},
    {Prospero::BufferFormat::kBc4UNorm, vk::Format::eBc4UnormBlock},
    {Prospero::BufferFormat::kBc4SNorm, vk::Format::eBc4SnormBlock},
    {Prospero::BufferFormat::kBc5UNorm, vk::Format::eBc5UnormBlock},
    {Prospero::BufferFormat::kBc5SNorm, vk::Format::eBc5SnormBlock},
    {Prospero::BufferFormat::kBc6UFloat, vk::Format::eBc6HUfloatBlock},
    {Prospero::BufferFormat::kBc6SFloat, vk::Format::eBc6HSfloatBlock},
    {Prospero::BufferFormat::kBc7UNorm, vk::Format::eBc7UnormBlock},
    {Prospero::BufferFormat::kBc7Srgb, vk::Format::eBc7SrgbBlock},
};

constexpr auto MakeFormatLookup() {
	constexpr auto kMaxFormat = static_cast<size_t>(Prospero::BufferFormat::kBc7Srgb);
	std::array<vk::Format, kMaxFormat + 1> lookup {};
	lookup.fill(vk::Format::eUndefined);
	for (const auto& mapping: kFormatMappings) {
		lookup[static_cast<size_t>(mapping.guest)] = mapping.host;
	}
	return lookup;
}

constexpr auto kFormatLookup = MakeFormatLookup();

} // namespace

vk::Format VulkanFormat(Prospero::BufferFormat guest_format) {
	const auto index = static_cast<size_t>(guest_format);
	return index < kFormatLookup.size() ? kFormatLookup[index] : vk::Format::eUndefined;
}

void RequireVulkanSuccess(vk::Result result, const char* operation) {
	if (result != vk::Result::eSuccess) {
		EXIT("%s failed: %s (%d)\n", operation, vk::to_string(result).c_str(),
		     static_cast<int>(result));
	}
}

namespace {

// KYTY_FUNCTION_ARRAY_SHRINK=1 (default 0: off): shader modules get their per-invocation
// (Function storage) arrays shrunk to the indices they can reach (spirvLocalArrays.h). The driver
// reserves local memory for the largest per-thread footprint of any pipeline times every thread the
// GPU keeps resident and never returns it; the recompiler's 8192-dword LDS emulation in vertex and
// pixel shaders made that 3.9 GiB on an RTX 3090 (Astro Bot's galaxy map), where the pixel shaders
// only reach 96 dwords. A rewritten module that fails spirv-val is not used (logged once).
// =zero also zero-fills the shrunk arrays at function entry (OpConstantNull initializer), =poison
// fills them with float NaNs (diagnostic: makes reads of elements no path wrote visible).
struct ShrinkMode {
	bool                   enabled = false;
	SpirvLocalArrays::Init init    = SpirvLocalArrays::Init::None;
};

const ShrinkMode& FunctionArrayShrinkMode() {
	static const ShrinkMode mode = [] {
		ShrinkMode  result;
		const auto* value = std::getenv("KYTY_FUNCTION_ARRAY_SHRINK");
		result.enabled    = value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
		if (result.enabled) {
			if (std::strcmp(value, "zero") == 0) {
				result.init = SpirvLocalArrays::Init::Zero;
			} else if (std::strcmp(value, "poison") == 0) {
				result.init = SpirvLocalArrays::Init::Poison;
			}
			std::printf("Kyty Function-storage arrays: shrunk to their proven index bound%s "
			            "(KYTY_FUNCTION_ARRAY_SHRINK=%s)\n",
			            result.init == SpirvLocalArrays::Init::Zero     ? ", zero-filled"
			            : result.init == SpirvLocalArrays::Init::Poison ? ", filled with NaNs (diagnostic)"
			                                                            : "",
			            value);
			std::fflush(stdout);
		}
		return result;
	}();
	return mode;
}

// Returns the rewritten module, or nothing (unchanged or invalid). `declared` / `created`: the
// module's Function-storage bytes per invocation before and as given to the driver.
std::vector<uint32_t> ShrinkFunctionArrays(std::span<const uint32_t> code, uint64_t& declared,
                                           uint64_t& created) {
	std::vector<uint32_t> shrunk;
	const auto            result = SpirvLocalArrays::Shrink(code, shrunk, FunctionArrayShrinkMode().init);
	declared                     = result.bytes_before;
	created                      = result.bytes_before;
	if (!result.changed) {
		return {};
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
	                                     const char* message) {
		if (messages.size() < 512) {
			messages += message;
			messages += "; ";
		}
	});
	const bool valid = tools.Validate(shrunk);
	static std::mutex                   log_mutex;
	static std::set<std::string>        logged;
	std::string                         arrays;
	for (const auto& array: result.arrays) {
		char part[160];
		std::snprintf(part, sizeof(part), "%s%s %u -> %u", arrays.empty() ? "" : ", ",
		              array.name.empty() ? "array" : array.name.c_str(), array.old_length, array.new_length);
		arrays += part;
	}
	{
		std::scoped_lock lock(log_mutex);
		if (logged.insert(arrays + (valid ? "" : " invalid")).second) {
			std::printf("Kyty Function-storage arrays (KYTY_FUNCTION_ARRAY_SHRINK): %s elements, %llu -> %llu "
			            "bytes per invocation%s%s\n",
			            arrays.c_str(), static_cast<unsigned long long>(result.bytes_before),
			            static_cast<unsigned long long>(result.bytes_after),
			            valid ? "" : "; the rewritten module failed validation, the original is used: ",
			            valid ? "" : messages.c_str());
			std::fflush(stdout);
		}
	}
	if (valid) {
		created = result.bytes_after;
	}
	return valid ? shrunk : std::vector<uint32_t> {};
}

} // namespace

vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device) {
	// Ported from Jetsku/KytyPS5 (u59 int4): only the module given to the driver changes.
	std::vector<uint32_t> shrunk;
	if (FunctionArrayShrinkMode().enabled) {
		uint64_t declared = 0;
		uint64_t created  = 0;
		shrunk            = ShrinkFunctionArrays(code, declared, created);
		if (!shrunk.empty()) {
			code = shrunk;
		}
	}
	vk::ShaderModuleCreateInfo create_info {};
	create_info.codeSize    = code.size_bytes();
	create_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	RequireVulkanSuccess(device.createShaderModule(&create_info, nullptr, &module),
	                     "create SPIR-V shader module");
	return module;
}

} // namespace Libs::Graphics
