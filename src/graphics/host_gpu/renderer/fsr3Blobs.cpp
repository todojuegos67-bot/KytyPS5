// FSR 3.1's shader blobs (upscaler, frame interpolation, optical flow) for the FidelityFX Vulkan backend: Kyty's glslang build of the SDK's GLSL
// passes (CMakeLists.txt, fsr3_shaders/), with the bindings the backend needs read from the SPIR-V
// itself. The SDK's own blob headers carry them from its shader compiler, which Kyty doesn't use.
#include "fsr3_shaders/fsr3_fg_permutations.h"

#include <bit> // FidelityFX's ffx_util.h calls std::popcount without it
#include <blob_accessors/ffx_frameinterpolation_shaderblobs.h>
#include <blob_accessors/ffx_fsr3upscaler_shaderblobs.h>
#include <blob_accessors/ffx_opticalflow_shaderblobs.h>

#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// FORCE_WAVE64 and ALLOW_FP16 pick no blob of their own: wave64 is never forced (below) and the
// passes are built without 16-bit math. Frame interpolation's: low-res MVs, jitter, inverted depth;
// optical flow's: HDR colour input.
constexpr uint32_t BlobBits = 0x3fu, FiBits = 0x7u, OfBits = 0x4u;

struct Binding {
	std::string name;
	uint32_t    binding;
	uint32_t    count;
	uint32_t    set;
};

// One kind of binding, with the arrays FfxShaderBlob points into.
struct Bindings {
	std::vector<Binding>     list;
	std::vector<const char*> names;
	std::vector<uint32_t>    bindings, counts, sets;

	void Finish() {
		for (const auto& entry: list) {
			names.push_back(entry.name.c_str());
			bindings.push_back(entry.binding);
			counts.push_back(entry.count);
			sets.push_back(entry.set);
		}
	}
};

struct Reflection {
	Bindings constant_buffers, srv_textures, uav_textures, srv_buffers, uav_buffers;
};

// The resource variables of a compute module: sampled images (SRVs), storage images (UAVs) and
// uniform blocks (constant buffers), storage buffers (SRV when read-only, else UAV). Samplers are
// skipped: the backend makes them from the pipeline description. Anything else fails, as FSR 3.1's
// passes have nothing else.
bool Reflect(std::span<const uint32_t> code, Reflection& out) {
	constexpr uint32_t OpName = 5, OpTypeImage = 25, OpTypeSampler = 26, OpTypeArray = 28,
	                   OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpVariable = 59,
	                   OpDecorate = 71, OpMemberDecorate = 72;
	constexpr uint32_t DecorationBinding = 33, DecorationSet = 34, DecorationNonWritable = 24,
	                   StorageUniform = 2, StorageBuffer = 12, DimBuffer = 5;
	if (code.size() < 5 || code[0] != 0x07230203u) {
		return false;
	}
	std::unordered_map<uint32_t, std::string>                names;
	std::unordered_map<uint32_t, uint32_t>                   bindings, sets, constants;
	std::unordered_set<uint32_t>                             read_only; // variables or block types
	std::unordered_map<uint32_t, std::span<const uint32_t>> types;
	std::vector<std::span<const uint32_t>>                   variables;
	for (size_t i = 5; i < code.size();) {
		const uint32_t count = code[i] >> 16u;
		if (count == 0 || i + count > code.size()) {
			return false;
		}
		const auto words = code.subspan(i, count);
		switch (code[i] & 0xffffu) {
			case OpName:
				if (count > 2) {
					const auto* text = reinterpret_cast<const char*>(&words[2]);
					names[words[1]] = std::string(text, strnlen(text, (count - 2) * 4));
				}
				break;
			case OpDecorate:
				if (count >= 4 && words[2] == DecorationBinding) {
					bindings[words[1]] = words[3];
				} else if (count >= 4 && words[2] == DecorationSet) {
					sets[words[1]] = words[3];
				} else if (count >= 3 && words[2] == DecorationNonWritable) {
					read_only.insert(words[1]);
				}
				break;
			case OpMemberDecorate: // glslang marks a readonly block's members
				if (count >= 4 && words[3] == DecorationNonWritable) {
					read_only.insert(words[1]);
				}
				break;
			case OpConstant:
				if (count >= 4) {
					constants[words[2]] = words[3];
				}
				break;
			case OpTypeImage:
			case OpTypeSampler:
			case OpTypeArray:
			case OpTypeStruct:
			case OpTypePointer:
				if (count >= 2) {
					types[words[1]] = words;
				}
				break;
			case OpVariable:
				if (count >= 4) {
					variables.push_back(words);
				}
				break;
			default: break;
		}
		i += count;
	}
	for (const auto& variable: variables) {
		const uint32_t id = variable[2];
		if (!bindings.contains(id)) {
			continue; // not a descriptor
		}
		const auto pointer = types.find(variable[1]);
		if (pointer == types.end() || (pointer->second[0] & 0xffffu) != OpTypePointer) {
			return false;
		}
		auto     type  = types.find(pointer->second[3]);
		uint32_t array = 1;
		if (type != types.end() && (type->second[0] & 0xffffu) == OpTypeArray) {
			if (!constants.contains(type->second[3])) {
				return false;
			}
			array = constants[type->second[3]];
			type  = types.find(type->second[2]);
		}
		if (type == types.end()) {
			return false;
		}
		const Binding entry {names[id], bindings[id], array, sets[id]};
		switch (type->second[0] & 0xffffu) {
			case OpTypeSampler: break;
			case OpTypeImage:
				if (type->second.size() < 9 || type->second[3] == DimBuffer) {
					return false;
				}
				if (type->second[7] == 1) {
					out.srv_textures.list.push_back(entry);
				} else if (type->second[7] == 2) {
					out.uav_textures.list.push_back(entry);
				} else {
					return false;
				}
				break;
			case OpTypeStruct:
				if (variable[3] == StorageUniform) {
					out.constant_buffers.list.push_back(entry);
				} else if (variable[3] == StorageBuffer) {
					const bool srv = read_only.contains(id) || read_only.contains(type->second[1]);
					(srv ? out.srv_buffers : out.uav_buffers).list.push_back(entry);
				} else {
					return false;
				}
				break;
			default: return false;
		}
	}
	out.constant_buffers.Finish();
	out.srv_textures.Finish();
	out.uav_textures.Finish();
	out.srv_buffers.Finish();
	out.uav_buffers.Finish();
	return true;
}

std::mutex                                   g_mutex;
std::map<const Fsr3Permutation*, Reflection> g_reflections;
std::unordered_set<const Fsr3Permutation*>   g_failed;

FfxErrorCode GetBlob(std::span<const Fsr3Permutation> table, uint32_t pass, uint32_t bits, FfxShaderBlob* outBlob) {
	std::memset(static_cast<void*>(outBlob), 0, sizeof(*outBlob));
	for (const auto& permutation: table) {
		if (permutation.pass != pass || permutation.bits != bits) {
			continue;
		}
		std::lock_guard lock(g_mutex);
		auto found = g_reflections.find(&permutation);
		if (found == g_reflections.end()) {
			if (g_failed.contains(&permutation)) {
				return FFX_ERROR_INVALID_ARGUMENT;
			}
			Reflection reflection;
			if (!Reflect({permutation.code, permutation.size / sizeof(uint32_t)}, reflection)) {
				g_failed.insert(&permutation);
				return FFX_ERROR_INVALID_ARGUMENT;
			}
			found = g_reflections.emplace(&permutation, std::move(reflection)).first;
		}
		auto&               r = found->second; // FfxShaderBlob takes non-const name arrays
		const FfxShaderBlob blob {
		    .data                      = reinterpret_cast<const uint8_t*>(permutation.code),
		    .size                      = static_cast<uint32_t>(permutation.size),
		    .cbvCount                  = static_cast<uint32_t>(r.constant_buffers.list.size()),
		    .srvTextureCount           = static_cast<uint32_t>(r.srv_textures.list.size()),
		    .uavTextureCount           = static_cast<uint32_t>(r.uav_textures.list.size()),
		    .srvBufferCount            = static_cast<uint32_t>(r.srv_buffers.list.size()),
		    .uavBufferCount            = static_cast<uint32_t>(r.uav_buffers.list.size()),
		    .boundConstantBufferNames  = r.constant_buffers.names.data(),
		    .boundConstantBuffers      = r.constant_buffers.bindings.data(),
		    .boundConstantBufferCounts = r.constant_buffers.counts.data(),
		    .boundConstantBufferSpaces = r.constant_buffers.sets.data(),
		    .boundSRVTextureNames      = r.srv_textures.names.data(),
		    .boundSRVTextures          = r.srv_textures.bindings.data(),
		    .boundSRVTextureCounts     = r.srv_textures.counts.data(),
		    .boundSRVTextureSpaces     = r.srv_textures.sets.data(),
		    .boundUAVTextureNames      = r.uav_textures.names.data(),
		    .boundUAVTextures          = r.uav_textures.bindings.data(),
		    .boundUAVTextureCounts     = r.uav_textures.counts.data(),
		    .boundUAVTextureSpaces     = r.uav_textures.sets.data(),
		    .boundSRVBufferNames       = r.srv_buffers.names.data(),
		    .boundSRVBuffers           = r.srv_buffers.bindings.data(),
		    .boundSRVBufferCounts      = r.srv_buffers.counts.data(),
		    .boundSRVBufferSpaces      = r.srv_buffers.sets.data(),
		    .boundUAVBufferNames       = r.uav_buffers.names.data(),
		    .boundUAVBuffers           = r.uav_buffers.bindings.data(),
		    .boundUAVBufferCounts      = r.uav_buffers.counts.data(),
		    .boundUAVBufferSpaces      = r.uav_buffers.sets.data(),
		};
		std::memcpy(static_cast<void*>(outBlob), &blob, sizeof(blob)); // its members are const
		return FFX_OK;
	}
	// shortcut: only the permutations Kyty's contexts ask for are built (CMakeLists.txt); add
	// others there when a context needs them.
	return FFX_ERROR_INVALID_ARGUMENT;
}

} // namespace

FfxErrorCode fsr3UpscalerGetPermutationBlobByIndex(FfxFsr3UpscalerPass passId,
                                                   uint32_t permutationOptions,
                                                   FfxShaderBlob* outBlob) {
	return GetBlob(FSR3_PERMUTATIONS, static_cast<uint32_t>(passId), permutationOptions & BlobBits, outBlob);
}

FfxErrorCode frameInterpolationGetPermutationBlobByIndex(FfxFrameInterpolationPass passId, FfxBindStage /*stageId*/,
                                                         uint32_t permutationOptions, FfxShaderBlob* outBlob) {
	return GetBlob(FSR3_FI_PERMUTATIONS, static_cast<uint32_t>(passId), permutationOptions & FiBits, outBlob);
}

FfxErrorCode opticalflowGetPermutationBlobByIndex(FfxOpticalflowPass passId, uint32_t permutationOptions,
                                                  FfxShaderBlob* outBlob) {
	return GetBlob(FSR3_OF_PERMUTATIONS, static_cast<uint32_t>(passId), permutationOptions & OfBits, outBlob);
}

FfxErrorCode frameInterpolationIsWave64(uint32_t /*permutationOptions*/, bool& isWave64) {
	isWave64 = false;
	return FFX_OK;
}

FfxErrorCode opticalflowIsWave64(uint32_t /*permutationOptions*/, bool& isWave64) {
	isWave64 = false;
	return FFX_OK;
}

FfxErrorCode fsr3UpscalerIsWave64(uint32_t /*permutationOptions*/, bool& isWave64) {
	isWave64 = false;
	return FFX_OK;
}
