#include "local/hwrt-scene.h"

#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"
#include "local/frame-gen.h"
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
#include "local/vulkan-recording.h"
#endif
#include "gpu_tiler_shaders/hwrt_gather_spv.h"
#include "gpu_tiler_shaders/hwrt_debug_spv.h"
#include "gpu_tiler_shaders/hwrt_debug11_spv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(KYTY_LOCAL_VULKAN_RECORDING)
extern "C" {
extern volatile std::atomic_uint32_t kyty_local_vulkan_recording_mode;
}
#endif

// KYTY_HWRT (hwrt-scene.h). Everything below runs only with the switch on and a device that has the features.
namespace Libs::Graphics::HwScene {
namespace {

constexpr uint32_t Slots            = 4;           // frame slots of the CPU-written rings
constexpr uint32_t MaxRecords       = 16384;       // per slot (records 0..63 of a slot capture nothing)
constexpr uint32_t NullRecords      = 64;
constexpr uint32_t MaxCapture       = 6u << 20u;   // captured vertices per frame (16 B each: 96 MiB)
constexpr uint32_t MaxMeshVertices  = 1u << 20u;   // window * instances of one mesh
constexpr uint32_t MaxMeshIndices   = 3u << 20u;   // index count * instances of one mesh
constexpr uint32_t MaxEntries       = 24576;
constexpr uint32_t MaxInstances     = 32768;       // TLAS
constexpr uint32_t MaxChecks        = 256;         // compare results per slot

bool g_supported = false;

uint32_t EnvU32(const char* name, uint32_t fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') return fallback;
	return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}
// KYTY_HWRT_BUILDS (48): new meshes built per frame at most; KYTY_HWRT_BUILD_TRIS (400000): their triangles at most.
uint32_t BuildsPerFrame() { static const uint32_t v = std::max(1u, EnvU32("KYTY_HWRT_BUILDS", 48)); return v; }
uint32_t BuildTrisPerFrame() { static const uint32_t v = std::max(1000u, EnvU32("KYTY_HWRT_BUILD_TRIS", 400000)); return v; }
// KYTY_HWRT_KEEP (600): frames a mesh not drawn stays in the TLAS (out of view); 4x that: its BLAS is freed.
uint32_t KeepFrames() { static const uint32_t v = std::max(1u, EnvU32("KYTY_HWRT_KEEP", 600)); return v; }
// KYTY_HWRT_DEBUG=1: instance colours shaded by distance in place of the TAA's output; 2: blended 50 %. Add 4: y flipped.
uint32_t RefitInterval() { static const uint32_t v = std::max(1u, EnvU32("KYTY_HWRT_REFIT_EVERY", 4)); return v; }
uint32_t RefitsPerFrame() { static const uint32_t v = EnvU32("KYTY_HWRT_REFITS", 64); return v; }
uint32_t DebugMode() { static const uint32_t v = EnvU32("KYTY_HWRT_DEBUG", 0); return v; }

struct Buffer {
	VkBuffer          buffer  = VK_NULL_HANDLE;
	VmaAllocation     memory  = nullptr;
	void*             mapped  = nullptr;
	vk::DeviceAddress address = 0;
	vk::DeviceSize    size    = 0;
};

bool g_host_readback = false; // (MakeBuffer: the next host buffer is read back by the CPU)

bool MakeBuffer(GraphicContext& graphics, Buffer& out, vk::DeviceSize size, VkBufferUsageFlags usage, bool host) {
	VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	create.size  = std::max<vk::DeviceSize>(size, 16);
	create.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (host) {
		// (10-26: the records were in system memory and every G-buffer vertex read them over PCIe - 6 fps. Written by the
		// CPU, read by the GPU: device-local host-visible memory (ReBAR) first, system memory only without it.)
		allocate.usage         = g_host_readback ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		allocate.flags         = (g_host_readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
		                                          : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) |
		                         VMA_ALLOCATION_CREATE_MAPPED_BIT;
		allocate.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}
	VmaAllocationInfo info {};
	if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &out.buffer, &out.memory, &info) != VK_SUCCESS) {
		out = {};
		return false;
	}
	out.mapped = info.pMappedData;
	out.size   = create.size;
	vk::BufferDeviceAddressInfo address {};
	address.buffer = vk::Buffer(out.buffer);
	out.address    = graphics.device.getBufferAddress(address);
	if (host && out.mapped == nullptr) {
		vmaDestroyBuffer(graphics.allocator, out.buffer, out.memory);
		out = {};
		return false;
	}
	return true;
}
void FreeBuffer(GraphicContext& graphics, Buffer& b) {
	if (b.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(graphics.allocator, b.buffer, b.memory);
	b = {};
}

struct Record {
	uint32_t base = 0, window = 0, instances = 0, pad = 0;
};

enum class Mode : uint32_t { Write = 0, Compare = 1 };

struct Entry {
	uint64_t key = 0;
	uint32_t id = 0;
	uint64_t vs = 0, ps = 0;
	uint32_t window = 0, instances = 0, indices = 0; // indices: per instance
	uint32_t tris = 0;
	Buffer   vb, ib, as_buf;
	vk::AccelerationStructureKHR blas = nullptr;
	vk::DeviceAddress            blas_address = 0;
	bool     created = false; // its buffers exist
	bool     built = false, dynamic = false, updatable = false;
	uint64_t last_seen = 0, next_check = 0, capture_frame = UINT64_MAX;
	uint32_t built_count = 0;
};

struct Job {
	Entry*   entry = nullptr;
	uint32_t src   = 0; // first captured vertex
	Mode     mode  = Mode::Write;
	uint32_t result = 0;
	bool     build = false, refit = false;
};

// (hwrt2) What the ray-traced effects read of the frame's scene (std430, hwrt2 shaders' binding).
struct RayInfo {
	float    inv[16];       // clip -> world rows
	float    vp[16];        // world -> clip rows (clip = M * world)
	float    cam[4];        // the camera's position
	float    info[4];       // 1 valid, TLAS instances
	uint32_t materials[4];  // the materials' address (8 words per TLAS instance: PS, VS, id, vertices, indices)
};

struct Pending {
	vk::AccelerationStructureKHR as = nullptr;
	Buffer                       a, b, c;
	uint64_t                     frame = 0;
};

struct State {
	std::mutex mutex;
	bool       init_tried = false, ready = false, disabled = false;
	Buffer     capture, records, instances[Slots], materials[Slots], results[Slots], scratch, tlas_buf;
	Buffer     rayinfo[Slots]; // (hwrt2: the frame's scene for the ray-traced effects, RayInfo)
	int        ray_slot = -1;  // the last slot whose TLAS and RayInfo are complete
	vk::AccelerationStructureKHR tlas = nullptr;
	vk::DeviceAddress            tlas_address = 0;
	bool                         tlas_built = false;
	vk::DeviceSize scratch_align = 256;
	vk::Pipeline       gather = nullptr, debug = nullptr, debug11 = nullptr;
	vk::PipelineLayout gather_layout = nullptr, debug_layout = nullptr;
	vk::DescriptorSetLayout debug_set_layout = nullptr;
	vk::QueryPool pool = nullptr;
	double        period_ns = 1.0;
	bool          pending_query[Slots] {};
	std::unordered_map<VkShaderModule, VkShaderModule> modules;
	// The frame being recorded.
	uint64_t frame = 1;
	uint32_t refits_frame = 0;
	uint32_t slot = 0, next_record = NullRecords, captured = 0, builds = 0, build_tris = 0, checks = 0;
	std::vector<Job> jobs;
	std::vector<std::pair<uint64_t, uint32_t>> check_keys[Slots]; // (key, result index) per slot
	bool          have_camera = false;
	std::array<float, 16> view_proj {};
	// Meshes.
	std::unordered_map<uint64_t, Entry> entries;
	uint32_t next_id = 1;
	std::deque<Pending> graveyard;
	// The 10 s log.
	uint64_t draws_seen = 0, draws_captured = 0, draws_reused = 0, frames = 0, frames_no_camera = 0;
	uint64_t skips[static_cast<uint32_t>(Skip::Count)] {};
	uint64_t scene_states = 0, scene_states_capturable = 0, new_meshes = 0, rebuilt = 0, refits = 0, compares = 0, became_dynamic = 0;
	uint64_t build_calls = 0, debug_calls = 0, notes = 0;
	uint64_t evicted = 0, captured_vertices = 0, tlas_instances_sum = 0, patched = 0, patch_failed = 0;
	double   build_ms = 0.0, cpu_note_ms = 0.0, cpu_after_ms = 0.0, gpu_ms_max = 0.0;
	uint32_t timed = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g;

void Say(const char* text) {
	std::printf("HWRT: %s\n", text);
	std::fflush(stdout);
}

bool HasExtension(const std::vector<vk::ExtensionProperties>& available, const char* name) {
	return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}
bool HasName(const std::vector<const char*>& list, const char* name) {
	return std::any_of(list.begin(), list.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
}

bool MakeCompute(GraphicContext& graphics, const uint32_t* code, size_t bytes, vk::PipelineLayout layout, vk::Pipeline& out) {
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = bytes;
	module_info.pCode    = code;
	vk::ShaderModule module = nullptr;
	if (graphics.device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo info {};
	info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	info.stage.module = module;
	info.stage.pName  = "main";
	info.layout       = layout;
	const auto created = graphics.device.createComputePipelines(nullptr, 1, &info, nullptr, &out);
	graphics.device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) out = nullptr;
	return out != nullptr;
}

bool Init(GraphicContext& graphics) {
	if (g.init_tried) return g.ready;
	g.init_tried = true;
	if (!g_supported) return false;
	const VkBufferUsageFlags as_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	bool ok = MakeBuffer(graphics, g.capture, vk::DeviceSize {MaxCapture} * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
	ok = ok && MakeBuffer(graphics, g.records, vk::DeviceSize {MaxRecords} * Slots * sizeof(Record), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
	for (uint32_t s = 0; s < Slots && ok; ++s) {
		ok = ok && MakeBuffer(graphics, g.instances[s], vk::DeviceSize {MaxInstances} * sizeof(VkAccelerationStructureInstanceKHR), as_input, true);
		ok = ok && MakeBuffer(graphics, g.materials[s], vk::DeviceSize {MaxInstances} * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
		ok = ok && MakeBuffer(graphics, g.rayinfo[s], sizeof(RayInfo), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
		g_host_readback = true;
		ok = ok && MakeBuffer(graphics, g.results[s], vk::DeviceSize {MaxChecks} * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
		g_host_readback = false;
	}
	if (ok) {
		std::memset(g.records.mapped, 0, static_cast<size_t>(g.records.size));
		vk::PhysicalDeviceAccelerationStructurePropertiesKHR as_props {};
		vk::PhysicalDeviceProperties2                        props {};
		props.pNext = &as_props;
		graphics.physical_device.getProperties2(&props);
		g.scratch_align = std::max<vk::DeviceSize>(as_props.minAccelerationStructureScratchOffsetAlignment, 256);
		g.period_ns     = props.properties.limits.timestampPeriod;
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = Slots * 2;
		if (graphics.device.createQueryPool(&query, nullptr, &g.pool) != vk::Result::eSuccess) g.pool = nullptr;
		// The gather (push constants only: buffer device addresses).
		const vk::PushConstantRange gather_push {vk::ShaderStageFlagBits::eCompute, 0, 128};
		vk::PipelineLayoutCreateInfo gather_info {};
		gather_info.pushConstantRangeCount = 1;
		gather_info.pPushConstantRanges    = &gather_push;
		ok = graphics.device.createPipelineLayout(&gather_info, nullptr, &g.gather_layout) == vk::Result::eSuccess &&
		     MakeCompute(graphics, HWRT_GATHER_SPV, sizeof(HWRT_GATHER_SPV), g.gather_layout, g.gather);
		// The debug view: the TLAS and the TAA's output (push descriptors).
		if (ok) {
			std::array<vk::DescriptorSetLayoutBinding, 2> bindings {};
			bindings[0].binding         = 0;
			bindings[0].descriptorType  = vk::DescriptorType::eAccelerationStructureKHR;
			bindings[0].descriptorCount = 1;
			bindings[0].stageFlags      = vk::ShaderStageFlagBits::eCompute;
			bindings[1].binding         = 1;
			bindings[1].descriptorType  = vk::DescriptorType::eStorageImage;
			bindings[1].descriptorCount = 1;
			bindings[1].stageFlags      = vk::ShaderStageFlagBits::eCompute;
			vk::DescriptorSetLayoutCreateInfo set_info {};
			set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
			set_info.bindingCount = static_cast<uint32_t>(bindings.size());
			set_info.pBindings    = bindings.data();
			vk::PipelineLayoutCreateInfo debug_info {};
			debug_info.setLayoutCount         = 1;
			debug_info.pSetLayouts            = &g.debug_set_layout;
			debug_info.pushConstantRangeCount = 1;
			debug_info.pPushConstantRanges    = &gather_push;
			if (DebugMode() != 0 &&
			    (graphics.device.createDescriptorSetLayout(&set_info, nullptr, &g.debug_set_layout) != vk::Result::eSuccess ||
			     graphics.device.createPipelineLayout(&debug_info, nullptr, &g.debug_layout) != vk::Result::eSuccess ||
			     !MakeCompute(graphics, HWRT_DEBUG_SPV, sizeof(HWRT_DEBUG_SPV), g.debug_layout, g.debug))) {
				Say("the debug view's pipeline could not be made: no debug view");
				g.debug = nullptr;
			}
			if (g.debug != nullptr && !MakeCompute(graphics, HWRT_DEBUG11_SPV, sizeof(HWRT_DEBUG11_SPV), g.debug_layout, g.debug11))
				g.debug11 = nullptr;
		}
	}
	if (!ok) {
		Say("could not allocate the capture buffers or make the gather pipeline: off");
		return false;
	}
	g.ready = true;
	std::printf("HWRT: ready (capture %u vertices at 0x%llx, records at 0x%llx; builds/frame %u, build tris/frame %u, keep %u "
	            "frames, debug %u)\n",
	            MaxCapture, static_cast<unsigned long long>(g.capture.address), static_cast<unsigned long long>(g.records.address),
	            BuildsPerFrame(), BuildTrisPerFrame(), KeepFrames(), DebugMode());
	std::fflush(stdout);
	return true;
}

// ---- The SPIR-V patch -------------------------------------------------------------------------------------------
// Before each OpReturn of the vertex entry point:
//   d = pc.dwords[31] + DrawIndex; rec = records[d] (by address); local = VertexIndex - BaseVertex;
//   inst = InstanceIndex - BaseInstance; if (d < cap && local < rec.window && inst < rec.instances)
//       capture[rec.base + inst * rec.window + local] = gl_Position;
struct SpirvPatcher {
	const std::vector<uint32_t>& in;
	uint32_t                     bound = 0;
	struct Inst {
		size_t   at;
		uint32_t op, count;
	};
	std::vector<Inst>     insts;
	std::vector<uint32_t> new_types, new_decorations, new_interface;
	std::string           error;

	explicit SpirvPatcher(const std::vector<uint32_t>& spirv): in(spirv) {}
	uint32_t Id() { return bound++; }
	static void Emit(std::vector<uint32_t>& out, uint32_t op, std::initializer_list<uint32_t> operands) {
		out.push_back((static_cast<uint32_t>(operands.size() + 1) << 16u) | op);
		out.insert(out.end(), operands.begin(), operands.end());
	}
	const uint32_t* W(const Inst& i) const { return in.data() + i.at; }

	uint32_t FindType(uint32_t op, std::initializer_list<uint32_t> operands) {
		for (const auto& i: insts) {
			if (i.op != op || i.count != operands.size() + 2) continue;
			const uint32_t* w = W(i);
			if (std::equal(operands.begin(), operands.end(), w + 2)) return w[1];
		}
		for (size_t k = 0; k < new_types.size();) {
			const uint32_t n = new_types[k] >> 16u, o = new_types[k] & 0xffffu;
			if (o == op && n == operands.size() + 2 && std::equal(operands.begin(), operands.end(), new_types.begin() + static_cast<long>(k) + 2))
				return new_types[k + 1];
			k += n;
		}
		const uint32_t        id = Id();
		std::vector<uint32_t> ops {id};
		ops.insert(ops.end(), operands.begin(), operands.end());
		new_types.push_back((static_cast<uint32_t>(ops.size() + 1) << 16u) | op);
		new_types.insert(new_types.end(), ops.begin(), ops.end());
		return id;
	}
	uint32_t Const32(uint32_t type, uint32_t value) {
		const uint32_t id = Id();
		Emit(new_types, 43, {type, id, value});
		return id;
	}
	uint32_t Const64(uint32_t type, uint64_t value) {
		const uint32_t id = Id();
		Emit(new_types, 43, {type, id, static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32u)});
		return id;
	}

	bool Run(std::vector<uint32_t>& out, uint64_t capture_address, uint64_t records_address, uint32_t record_cap, uint32_t capture_cap) {
		if (in.size() < 5 || in[0] != 0x07230203u) return error = "not SPIR-V", false;
		const uint32_t version = in[1];
		bound                  = in[3];
		for (size_t at = 5; at < in.size();) {
			const uint32_t count = in[at] >> 16u, op = in[at] & 0xffffu;
			if (count == 0 || at + count > in.size()) return error = "malformed", false;
			insts.push_back({at, op, count});
			at += count;
		}
		uint32_t entry = 0;
		size_t   entry_inst = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k)
			if (insts[k].op == 15 && W(insts[k])[1] == 0) {
				if (entry != 0) return error = "several vertex entry points", false;
				entry = W(insts[k])[2], entry_inst = k;
			}
		if (entry == 0) return error = "no vertex entry point", false;
		bool     has_int64 = false, has_psb = false, has_ext_psb = false, has_dp = false, has_ext_dp = false;
		uint32_t pos_var = 0, pos_struct = 0, pos_member = 0;
		// Built-in input variables by built-in (VertexIndex 42, InstanceIndex 43, BaseVertex 4424, BaseInstance 4425,
		// DrawIndex 4426).
		std::unordered_map<uint32_t, uint32_t> builtin_var;
		for (const auto& i: insts) {
			const uint32_t* w = W(i);
			if (i.op == 17 && w[1] == 11) has_int64 = true;
			if (i.op == 17 && w[1] == 5347) has_psb = true;
			if (i.op == 17 && w[1] == 4427) has_dp = true;
			if (i.op == 10) {
				const auto* name = reinterpret_cast<const char*>(w + 1);
				const auto  len  = static_cast<size_t>(i.count - 1) * 4;
				if (std::strncmp(name, "SPV_KHR_physical_storage_buffer", len) == 0) has_ext_psb = true;
				if (std::strncmp(name, "SPV_KHR_shader_draw_parameters", len) == 0) has_ext_dp = true;
			}
			if (i.op == 71 && i.count >= 4 && w[2] == 11) {
				if (w[3] == 0) pos_var = w[1];
				else builtin_var[w[3]] = w[1];
			}
			if (i.op == 72 && i.count >= 5 && w[3] == 11 && w[4] == 0) pos_struct = w[1], pos_member = w[2];
		}
		std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointers; // id -> (storage, pointee)
		std::unordered_map<uint32_t, uint32_t>                      var_type; // var id -> pointer type
		std::unordered_map<uint32_t, uint32_t>                      var_storage;
		std::unordered_map<uint32_t, const uint32_t*>               types;
		for (const auto& i: insts) {
			const uint32_t* w = W(i);
			if (i.op == 32) pointers[w[1]] = {w[2], w[3]};
			if (i.op == 59) var_type[w[2]] = w[1], var_storage[w[2]] = w[3];
			if ((i.op >= 19 && i.op <= 30) || i.op == 32) types[w[1]] = w;
		}
		uint32_t vec4 = 0, pos_struct_var = 0;
		if (pos_var != 0) {
			const auto p = pointers.find(var_type[pos_var]);
			if (p == pointers.end() || p->second.first != 3) return error = "Position is not an Output variable", false;
			vec4 = p->second.second;
		} else if (pos_struct != 0) {
			for (const auto& [var, ptype]: var_type) {
				const auto p = pointers.find(ptype);
				if (p != pointers.end() && p->second.first == 3 && p->second.second == pos_struct) pos_struct_var = var;
			}
			const auto s = types.find(pos_struct);
			if (pos_struct_var == 0 || s == types.end() || (s->second[0] >> 16u) < 3 + pos_member)
				return error = "no gl_PerVertex output variable", false;
			vec4 = s->second[2 + pos_member];
		} else {
			return error = "no Position output", false;
		}
		const auto v4 = types.find(vec4);
		if (v4 == types.end() || (v4->second[0] & 0xffffu) != 23 || v4->second[3] != 4) return error = "Position is not a vec4", false;
		// The push constant block (the translator's: struct { uint dwords[32]; }), else a new one.
		uint32_t push_var = 0, push_struct = 0;
		for (const auto& [var, storage]: var_storage)
			if (storage == 9) {
				if (push_var != 0) return error = "several push constant variables", false;
				push_var = var;
			}
		const uint32_t t_bool = FindType(20, {});
		const uint32_t t_u32  = FindType(21, {32, 0});
		const uint32_t t_u64  = FindType(21, {64, 0});
		const uint32_t t_pu32 = FindType(32, {5349, t_u32});
		const uint32_t t_pv4  = FindType(32, {5349, vec4});
		const uint32_t c_0    = Const32(t_u32, 0);
		const uint32_t c_31   = Const32(t_u32, CursorDword);
		bool           push_new = false;
		if (push_var != 0) {
			const auto p = pointers.find(var_type[push_var]);
			if (p == pointers.end()) return error = "push constant pointer", false;
			push_struct      = p->second.second;
			const auto s     = types.find(push_struct);
			if (s == types.end() || (s->second[0] & 0xffffu) != 30 || (s->second[0] >> 16u) != 3)
				return error = "push constant block is not the translator's", false;
			const auto a = types.find(s->second[2]);
			if (a == types.end() || (a->second[0] & 0xffffu) != 28) return error = "push constant block is not an array", false;
			const auto elem = types.find(a->second[2]);
			if (elem == types.end() || (elem->second[0] & 0xffffu) != 21 || elem->second[2] != 32)
				return error = "push constant array is not of 32-bit integers", false;
		} else {
			const uint32_t c_32  = Const32(t_u32, 32);
			const uint32_t arr   = Id();
			Emit(new_types, 28, {arr, t_u32, c_32});
			push_struct = Id();
			Emit(new_types, 30, {push_struct, arr});
			Emit(new_decorations, 71, {arr, 6, 4});              // ArrayStride 4
			Emit(new_decorations, 72, {push_struct, 0, 35, 0});  // member 0 Offset 0
			Emit(new_decorations, 71, {push_struct, 2});         // Block
			const uint32_t t_ppush = FindType(32, {9, push_struct});
			push_var               = Id();
			Emit(new_types, 59, {t_ppush, push_var, 9});
			push_new = true;
			if (version >= 0x00010400u) new_interface.push_back(push_var);
		}
		// The push array's element type is the module's (an existing u32: FindType returned it, or the one made above).
		const uint32_t t_ppc = FindType(32, {9, t_u32});
		// Built-ins.
		struct Builtin {
			uint32_t builtin, var = 0, type = 0;
			bool     is_signed = false;
		};
		std::array<Builtin, 5> bi {{{42}, {43}, {4424}, {4425}, {4426}}};
		for (auto& b: bi) {
			const auto found = builtin_var.find(b.builtin);
			if (found != builtin_var.end()) {
				b.var  = found->second;
				const auto p = pointers.find(var_type[b.var]);
				if (p == pointers.end() || p->second.first != 1) return error = "a built-in is not an Input variable", false;
				b.type = p->second.second;
				const auto t = types.find(b.type);
				if (t == types.end() || (t->second[0] & 0xffffu) != 21 || t->second[2] != 32) return error = "a built-in's type", false;
				b.is_signed = t->second[3] != 0;
			} else {
				const uint32_t t_pin = FindType(32, {1, t_u32});
				b.var  = Id();
				b.type = t_u32;
				Emit(new_types, 59, {t_pin, b.var, 1});
				Emit(new_decorations, 71, {b.var, 11, b.builtin});
				new_interface.push_back(b.var);
			}
		}
		uint32_t t_pout = 0, c_member = 0;
		if (pos_var == 0) {
			t_pout   = FindType(32, {3, vec4});
			c_member = Const32(t_u32, pos_member);
		}
		const uint32_t c_rec  = Const64(t_u64, records_address);
		const uint32_t c_cap  = Const64(t_u64, capture_address);
		const uint32_t c_16   = Const64(t_u64, 16);
		const uint32_t c_4    = Const64(t_u64, 4);
		const uint32_t c_8    = Const64(t_u64, 8);
		const uint32_t c_rcap = Const32(t_u32, record_cap);
		const uint32_t c_ccap = Const32(t_u32, capture_cap);

		size_t first_function = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k)
			if (insts[k].op == 54) {
				first_function = k;
				break;
			}
		size_t types_start = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k) {
			const uint32_t op = insts[k].op;
			if ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59 || op == 1 || op == 54) {
				types_start = k;
				break;
			}
		}
		if (first_function == SIZE_MAX || types_start == SIZE_MAX) return error = "no function", false;
		(void)push_new;

		out.clear();
		out.reserve(in.size() + 512);
		out.insert(out.end(), in.begin(), in.begin() + 5);
		bool   in_entry = false, caps_done = false;
		size_t returns  = 0;
		for (size_t k = 0; k < insts.size(); ++k) {
			const auto&     i = insts[k];
			const uint32_t* w = W(i);
			if (!caps_done && i.op != 17) {
				caps_done = true;
				if (!has_int64) Emit(out, 17, {11});
				if (!has_psb) Emit(out, 17, {5347});
				if (!has_dp) Emit(out, 17, {4427});
				const auto extension = [&](const char* name) {
					std::array<uint32_t, 16> words {};
					const size_t             len = std::strlen(name);
					std::memcpy(words.data(), name, len);
					const uint32_t n = static_cast<uint32_t>(len / 4 + 1);
					out.push_back(((n + 1) << 16u) | 10u);
					out.insert(out.end(), words.begin(), words.begin() + n);
				};
				if (!has_ext_psb && version < 0x00010500u) extension("SPV_KHR_physical_storage_buffer");
				if (!has_ext_dp && version < 0x00010300u) extension("SPV_KHR_shader_draw_parameters");
			}
			if (k == types_start) out.insert(out.end(), new_decorations.begin(), new_decorations.end());
			if (k == first_function) out.insert(out.end(), new_types.begin(), new_types.end());
			if (i.op == 14) { // OpMemoryModel: PhysicalStorageBuffer64 addressing
				out.push_back(w[0]);
				out.push_back(5348);
				out.insert(out.end(), w + 2, w + i.count);
				continue;
			}
			if (k == entry_inst && !new_interface.empty()) {
				out.push_back(((i.count + static_cast<uint32_t>(new_interface.size())) << 16u) | 15u);
				out.insert(out.end(), w + 1, w + i.count);
				out.insert(out.end(), new_interface.begin(), new_interface.end());
				continue;
			}
			if (i.op == 54) in_entry = w[2] == entry;
			if (i.op == 56) in_entry = false;
			if (in_entry && i.op == 253) {
				++returns;
				uint32_t ptr = pos_var;
				if (ptr == 0) {
					ptr = Id();
					Emit(out, 65, {t_pout, ptr, pos_struct_var, c_member});
				}
				// The built-ins as u32.
				std::array<uint32_t, 5> v {};
				for (size_t b = 0; b < bi.size(); ++b) {
					const uint32_t loaded = Id();
					Emit(out, 61, {bi[b].type, loaded, bi[b].var});
					if (bi[b].is_signed) {
						v[b] = Id();
						Emit(out, 124, {t_u32, v[b], loaded}); // OpBitcast
					} else {
						v[b] = loaded;
					}
				}
				const uint32_t pc_p = Id(), pc = Id(), d = Id(), d_ok = Id();
				Emit(out, 65, {t_ppc, pc_p, push_var, c_0, c_31});
				Emit(out, 61, {t_u32, pc, pc_p});
				Emit(out, 128, {t_u32, d, pc, v[4]});
				Emit(out, 176, {t_bool, d_ok, d, c_rcap}); // ULessThan
				const uint32_t then1 = Id(), merge1 = Id();
				Emit(out, 247, {merge1, 0});
				Emit(out, 250, {d_ok, then1, merge1});
				Emit(out, 248, {then1});
				// The record.
				const uint32_t d64 = Id(), roff = Id(), raddr = Id();
				Emit(out, 113, {t_u64, d64, d});
				Emit(out, 132, {t_u64, roff, d64, c_16});
				Emit(out, 128, {t_u64, raddr, c_rec, roff});
				const auto load_u32 = [&](uint32_t address) {
					const uint32_t p = Id(), r = Id();
					Emit(out, 120, {t_pu32, p, address}); // OpConvertUToPtr
					Emit(out, 61, {t_u32, r, p, 2, 4});   // Aligned 4
					return r;
				};
				const uint32_t a1 = Id(), a2 = Id();
				const uint32_t base = load_u32(raddr);
				Emit(out, 128, {t_u64, a1, raddr, c_4});
				const uint32_t window = load_u32(a1);
				Emit(out, 128, {t_u64, a2, raddr, c_8});
				const uint32_t count = load_u32(a2);
				const uint32_t local = Id(), inst = Id(), ok_l = Id(), ok_i = Id(), ok = Id();
				Emit(out, 130, {t_u32, local, v[0], v[2]}); // OpISub
				Emit(out, 130, {t_u32, inst, v[1], v[3]});
				Emit(out, 176, {t_bool, ok_l, local, window});
				Emit(out, 176, {t_bool, ok_i, inst, count});
				Emit(out, 167, {t_bool, ok, ok_l, ok_i}); // OpLogicalAnd
				const uint32_t then2 = Id(), merge2 = Id();
				Emit(out, 247, {merge2, 0});
				Emit(out, 250, {ok, then2, merge2});
				Emit(out, 248, {then2});
				const uint32_t m = Id(), e1 = Id(), e = Id(), ok_e = Id();
				Emit(out, 132, {t_u32, m, inst, window});   // OpIMul
				Emit(out, 128, {t_u32, e1, m, local});
				Emit(out, 128, {t_u32, e, e1, base});
				Emit(out, 176, {t_bool, ok_e, e, c_ccap});
				const uint32_t then3 = Id(), merge3 = Id();
				Emit(out, 247, {merge3, 0});
				Emit(out, 250, {ok_e, then3, merge3});
				Emit(out, 248, {then3});
				const uint32_t pos = Id(), e64 = Id(), off = Id(), addr = Id(), dst = Id();
				Emit(out, 61, {vec4, pos, ptr});
				Emit(out, 113, {t_u64, e64, e});
				Emit(out, 132, {t_u64, off, e64, c_16});
				Emit(out, 128, {t_u64, addr, c_cap, off});
				Emit(out, 120, {t_pv4, dst, addr});
				Emit(out, 62, {dst, pos, 2, 16});
				Emit(out, 249, {merge3});
				Emit(out, 248, {merge3});
				Emit(out, 249, {merge2});
				Emit(out, 248, {merge2});
				Emit(out, 249, {merge1});
				Emit(out, 248, {merge1});
				Emit(out, 253, {});
				continue;
			}
			out.insert(out.end(), w, w + i.count);
		}
		if (returns == 0) return error = "no return in the entry point", false;
		out[3] = bound;
		if (out[1] < 0x00010300u) out[1] = 0x00010300u;
		return true;
	}
};

bool Invert4(const std::array<double, 16>& m, std::array<double, 16>& r) {
	std::array<double, 16> inv {};
	inv[0]  = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8]  = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5]  = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9]  = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2]  = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6]  = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3]  = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7]  = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
	const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (!(std::abs(det) > 1e-30) || !std::isfinite(det)) return false;
	for (int i = 0; i < 16; ++i) r[i] = inv[i] / det;
	return true;
}

void Barrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
             vk::AccessFlags2 dst) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src;
	barrier.dstStageMask  = dst_stage;
	barrier.dstAccessMask = dst;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

bool MakeAs(GraphicContext& graphics, Buffer& buf, vk::AccelerationStructureKHR& as, vk::DeviceSize size,
            vk::AccelerationStructureTypeKHR type) {
	if (!MakeBuffer(graphics, buf, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) return false;
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = vk::Buffer(buf.buffer);
	create.size   = size;
	create.type   = type;
	if (graphics.device.createAccelerationStructureKHR(&create, nullptr, &as) != vk::Result::eSuccess) {
		as = nullptr;
		FreeBuffer(graphics, buf);
		return false;
	}
	return true;
}

void Bury(Entry& e) {
	Pending p;
	p.as    = e.blas;
	p.a     = e.vb;
	p.b     = e.ib;
	p.c     = e.as_buf;
	p.frame = g.frame;
	g.graveyard.push_back(p);
	e.blas = nullptr;
	e.vb = e.ib = e.as_buf = {};
	e.built = e.created = false;
}

void Collect(GraphicContext& graphics, bool all) {
	while (!g.graveyard.empty() && (all || g.graveyard.front().frame + Slots + 2 <= g.frame)) {
		auto& p = g.graveyard.front();
		if (p.as != nullptr) graphics.device.destroyAccelerationStructureKHR(p.as);
		FreeBuffer(graphics, p.a);
		FreeBuffer(graphics, p.b);
		FreeBuffer(graphics, p.c);
		g.graveyard.pop_front();
	}
}

uint64_t Mix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
	h *= 0xff51afd7ed558ccdULL;
	return h ^ (h >> 33u);
}

double Ms(std::chrono::steady_clock::time_point since) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

void ReadTiming(GraphicContext& graphics) {
	if (g.pool == nullptr) return;
	for (uint32_t s = 0; s < Slots; ++s) {
		if (!g.pending_query[s] || s == g.slot) continue;
		std::array<uint64_t, 4> r {};
		if (graphics.device.getQueryPoolResults(g.pool, s * 2, 2, sizeof(r), r.data(), 2 * sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability) !=
		    vk::Result::eSuccess)
			continue;
		if (r[1] == 0 || r[3] == 0) continue;
		g.pending_query[s] = false;
		const double ms = static_cast<double>(r[2] - r[0]) * g.period_ns / 1e6;
		g.build_ms += ms;
		g.gpu_ms_max = std::max(g.gpu_ms_max, ms);
		++g.timed;
	}
}

void MaybeLog() {
	const auto now = std::chrono::steady_clock::now();
	if (now - g.last_log < std::chrono::seconds(10)) return;
	const double frames = static_cast<double>(std::max<uint64_t>(g.frames, 1));
	uint64_t     blas_bytes = 0, vertex_bytes = 0, built = 0, dynamic = 0, tris = 0;
	for (const auto& [key, e]: g.entries) {
		if (!e.built) continue;
		++built;
		dynamic += e.dynamic ? 1 : 0;
		tris += e.tris;
		blas_bytes += e.as_buf.size;
		vertex_bytes += e.vb.size + e.ib.size;
	}
	const auto sk = [&](Skip s) { return static_cast<unsigned long long>(g.skips[static_cast<uint32_t>(s)]); };
	std::printf("HWRT: 10 s: %llu frames (%llu without a camera); draws seen %llu (%.0f/frame), captured %llu, reused (cached, "
	            "nothing captured) %llu; skipped: no variant %llu, speculation %llu, push data %llu, records full %llu, topology %llu, "
	            "budget %llu, table full %llu, GPU-written %llu; G-buffer states %llu (capturable %llu); variants patched %llu, failed %llu\n",
	            static_cast<unsigned long long>(g.frames), static_cast<unsigned long long>(g.frames_no_camera),
	            static_cast<unsigned long long>(g.draws_seen), static_cast<double>(g.draws_seen) / frames,
	            static_cast<unsigned long long>(g.draws_captured), static_cast<unsigned long long>(g.draws_reused), sk(Skip::NoVariant),
	            sk(Skip::Speculation), sk(Skip::PushData), sk(Skip::Records), sk(Skip::Topology), sk(Skip::Budget), sk(Skip::Entries), sk(Skip::GpuWritten),
	            static_cast<unsigned long long>(g.scene_states), static_cast<unsigned long long>(g.scene_states_capturable),
	            static_cast<unsigned long long>(g.patched), static_cast<unsigned long long>(g.patch_failed));
	std::printf("HWRT: 10 s: unique BLAS %llu (%llu dynamic, %llu meshes known), %.2f M triangles, BLAS memory %.1f MiB, vertex+index "
	            "%.1f MiB; new %llu, rebuilt %llu, refits %llu, compares %llu (became dynamic %llu), evicted %llu; TLAS instances %.0f/frame; "
	            "captured vertices %.0f/frame; GPU (gather+build) %.3f ms avg, %.3f max (%u timed); CPU render thread %.3f ms/frame "
	            "(draw notes %.3f + after-TAA %.3f)\n",
	            static_cast<unsigned long long>(built), static_cast<unsigned long long>(dynamic),
	            static_cast<unsigned long long>(g.entries.size()), static_cast<double>(tris) / 1e6,
	            static_cast<double>(blas_bytes) / 1048576.0, static_cast<double>(vertex_bytes) / 1048576.0,
	            static_cast<unsigned long long>(g.new_meshes), static_cast<unsigned long long>(g.rebuilt),
	            static_cast<unsigned long long>(g.refits), static_cast<unsigned long long>(g.compares),
	            static_cast<unsigned long long>(g.became_dynamic), static_cast<unsigned long long>(g.evicted),
	            static_cast<double>(g.tlas_instances_sum) / frames, static_cast<double>(g.captured_vertices) / frames,
	            g.timed > 0 ? g.build_ms / g.timed : 0.0, g.gpu_ms_max, g.timed, (g.cpu_note_ms + g.cpu_after_ms) / frames,
	            g.cpu_note_ms / frames, g.cpu_after_ms / frames);
	std::printf("HWRT: 10 s: scene builds %llu (one a frame: at the TAA dispatch or the PS5 Pro upscale packet), debug views %llu%s\n",
	            static_cast<unsigned long long>(g.build_calls), static_cast<unsigned long long>(g.debug_calls),
	            g.build_calls == 0 ? " - NO BUILD HOOK RAN: no TAA dispatch / super-resolution packet reached the hook" : "");
	std::fflush(stdout);
	g.draws_seen = g.draws_captured = g.draws_reused = g.frames = g.frames_no_camera = 0;
	std::fill(std::begin(g.skips), std::end(g.skips), 0);
	g.scene_states = g.scene_states_capturable = g.new_meshes = g.rebuilt = g.refits = g.compares = g.became_dynamic = 0;
	g.evicted = g.captured_vertices = g.tlas_instances_sum = 0;
	g.build_calls = g.debug_calls = 0;
	g.build_ms = g.cpu_note_ms = g.cpu_after_ms = g.gpu_ms_max = 0.0;
	g.timed    = 0;
	g.last_log = now;
}

// A new mesh's buffers: its indices (from guest memory) for every instance, its world-space vertices (written by the
// gather), on the render thread. False: not capturable (logged as topology).
bool CreateEntry(GraphicContext& graphics, Entry& e, const Draw& d) {
	const uint32_t count = d.index_count - d.index_count % 3;
	uint32_t       hi    = 0;
	if (d.index_size == 2) {
		const auto* src = reinterpret_cast<const uint16_t*>(d.index_address);
		for (uint32_t i = 0; i < count; ++i) hi = std::max<uint32_t>(hi, src[i]);
	} else {
		const auto* src = reinterpret_cast<const uint32_t*>(d.index_address);
		for (uint32_t i = 0; i < count; ++i) hi = std::max<uint32_t>(hi, src[i]);
	}
	const uint64_t window = uint64_t {hi} + 1;
	if (window * d.instance_count > MaxMeshVertices || uint64_t {count} * d.instance_count > MaxMeshIndices) return false;
	e.window    = static_cast<uint32_t>(window);
	e.instances = d.instance_count;
	e.indices   = count;
	e.tris      = count / 3 * d.instance_count;
	const VkBufferUsageFlags as_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	if (!MakeBuffer(graphics, e.ib, vk::DeviceSize {count} * d.instance_count * 4, as_input, true)) return false;
	if (!MakeBuffer(graphics, e.vb, vk::DeviceSize {e.window} * d.instance_count * 16, as_input | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false)) {
		FreeBuffer(graphics, e.ib);
		return false;
	}
	auto* dst = static_cast<uint32_t*>(e.ib.mapped);
	for (uint32_t inst = 0; inst < d.instance_count; ++inst) {
		const uint32_t add = inst * e.window;
		if (d.index_size == 2) {
			const auto* src = reinterpret_cast<const uint16_t*>(d.index_address);
			for (uint32_t i = 0; i < count; ++i) *dst++ = uint32_t {src[i]} + add;
		} else {
			const auto* src = reinterpret_cast<const uint32_t*>(d.index_address);
			for (uint32_t i = 0; i < count; ++i) *dst++ = src[i] + add;
		}
	}
	vmaFlushAllocation(graphics.allocator, e.ib.memory, 0, VK_WHOLE_SIZE);
	e.created = true;
	return true;
}

GraphicContext* g_graphics = nullptr;

} // namespace

bool Requested() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_HWRT");
		return value != nullptr && *value == '1';
	}();
	return on;
}

void AddDeviceExtensions(const std::vector<vk::ExtensionProperties>& available, std::vector<const char*>& device_extensions) {
	if (!Requested()) return;
	const char* needed[] {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	                      VK_KHR_RAY_QUERY_EXTENSION_NAME};
	for (const char* name: needed) {
		if (!HasExtension(available, name)) {
			std::printf("HWRT: the device has no %s: hardware ray tracing off\n", name);
			std::fflush(stdout);
			return;
		}
	}
	for (const char* name: needed)
		if (!HasName(device_extensions, name)) device_extensions.push_back(name);
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& device_extensions, void* next) {
	if (!Requested()) return next;
	if (!HasName(device_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) || !HasName(device_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME))
		return next;
	vk::PhysicalDeviceAccelerationStructureFeaturesKHR as {};
	vk::PhysicalDeviceRayQueryFeaturesKHR              rq {};
	as.pNext = &rq;
	vk::PhysicalDeviceFeatures2 features {};
	features.pNext = &as;
	physical.getFeatures2(&features);
	if (!as.accelerationStructure || !rq.rayQuery) {
		Say(!as.accelerationStructure ? "no accelerationStructure feature: off" : "no rayQuery feature: off");
		return next;
	}
	// (Static: they must outlive vkCreateDevice.)
	static vk::PhysicalDeviceAccelerationStructureFeaturesKHR s_as {};
	static vk::PhysicalDeviceRayQueryFeaturesKHR              s_rq {};
	s_as                       = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {};
	s_rq                       = vk::PhysicalDeviceRayQueryFeaturesKHR {};
	s_as.accelerationStructure = VK_TRUE;
	s_rq.rayQuery              = VK_TRUE;
	s_rq.pNext                 = next;
	s_as.pNext                 = &s_rq;
	g_supported                = true;
	Say("VK_KHR_acceleration_structure + VK_KHR_ray_query enabled (KYTY_HWRT=1)");
	return &s_as;
}

bool Active() { return g_supported && Requested() && !g.disabled; }

vk::ShaderModule CaptureModule(GraphicContext& graphics, vk::ShaderModule original) {
	if (!Active() || original == nullptr) return nullptr;
	std::lock_guard lock(g.mutex);
	g_graphics = &graphics;
	if (!Init(graphics)) return nullptr;
	const auto key = static_cast<VkShaderModule>(original);
	if (auto found = g.modules.find(key); found != g.modules.end()) return vk::ShaderModule(found->second);
	VkShaderModule made  = VK_NULL_HANDLE;
	const auto     spirv = ModuleStore::Holds(original) ? ModuleStore::Load(original) : nullptr;
	if (spirv == nullptr) {
		++g.patch_failed;
		static bool said = false;
		if (!said) said = true, Say("a G-buffer vertex shader's SPIR-V is not readable (module store off): not captured");
	} else {
		std::vector<uint32_t> patched;
		SpirvPatcher          patcher(*spirv);
		if (!patcher.Run(patched, g.capture.address, g.records.address, MaxRecords * Slots, MaxCapture)) {
			++g.patch_failed;
			static int said = 0;
			if (said++ < 8) {
				std::printf("HWRT: a G-buffer vertex shader could not be patched (%s): not captured\n", patcher.error.c_str());
				std::fflush(stdout);
			}
		} else {
			vk::ShaderModule module = ModuleStore::Enabled() ? ModuleStore::Add(patched) : vk::ShaderModule {};
			if (module == nullptr) {
				vk::ShaderModuleCreateInfo create {};
				create.codeSize = patched.size() * sizeof(uint32_t);
				create.pCode    = patched.data();
				if (graphics.device.createShaderModule(&create, nullptr, &module) != vk::Result::eSuccess) module = nullptr;
			}
			made = static_cast<VkShaderModule>(module);
			if (made != VK_NULL_HANDLE) ++g.patched;
			static bool said = false;
			if (!said && made != VK_NULL_HANDLE) said = true, Say("first G-buffer vertex shader patched (gl_Position capture)");
		}
	}
	g.modules[key] = made;
	return vk::ShaderModule(made);
}

uint64_t VariantId(uint64_t program_id) {
	uint64_t h = Mix(program_id ^ 0x48575254ULL, g.capture.address);
	h          = Mix(h, g.records.address);
	h          = Mix(h, uint64_t {MaxRecords * Slots} << 32u | MaxCapture);
	return h | (uint64_t {1} << 63u);
}

void NoteSkip(Skip why, uint32_t draws) {
	if (!Active()) return;
	if ((++g.notes & 255u) == 0) MaybeLog(); // (the stats print even when no frame is built)
	g.skips[static_cast<uint32_t>(why)] += draws;
	g.draws_seen += draws;
}

void NoteSceneState(bool capturable) {
	if (!Active()) return;
	++g.scene_states;
	MaybeLog();
	if (capturable) ++g.scene_states_capturable;
}

uint32_t Reserve(uint32_t count) {
	// (Failed: a cursor past every record - the patched shader's bound check - so nothing of the call is captured.)
	if (!g.ready || count == 0 || g.next_record + count > MaxRecords) return MaxRecords * Slots;
	const uint32_t slot_base = g.slot * MaxRecords;
	const uint32_t first = g.next_record;
	g.next_record += count;
	auto* recs = static_cast<Record*>(g.records.mapped) + slot_base;
	for (uint32_t i = 0; i < count; ++i) recs[first + i] = {};
	return slot_base + first;
}

bool Note(uint32_t record, const Draw& d) {
	if (!g.ready || g_graphics == nullptr) return false;
	const auto start = std::chrono::steady_clock::now();
	if ((++g.notes & 255u) == 0) MaybeLog();
	++g.draws_seen;
	const uint32_t slot_base = g.slot * MaxRecords;
	if (record >= MaxRecords * Slots || record < slot_base + NullRecords) {
		++g.skips[static_cast<uint32_t>(Skip::Records)];
		g.cpu_note_ms += Ms(start);
		return false;
	}
	auto&          rec = static_cast<Record*>(g.records.mapped)[record];
	const uint64_t key = Mix(Mix(Mix(Mix(Mix(d.vs, d.index_address), uint64_t {d.index_count} << 32u | d.instance_count),
	                                 static_cast<uint32_t>(d.vertex_offset)),
	                             d.index_size),
	                         0x5eedULL);
	if (d.index_count < 3 || d.instance_count == 0 || d.index_address == 0) {
		++g.skips[static_cast<uint32_t>(Skip::Topology)];
		g.cpu_note_ms += Ms(start);
		return false;
	}
	auto found = g.entries.find(key);
	if (found == g.entries.end()) {
		if (g.entries.size() >= MaxEntries) {
			++g.skips[static_cast<uint32_t>(Skip::Entries)];
			g.cpu_note_ms += Ms(start);
			return false;
		}
		found = g.entries.emplace(key, Entry {}).first;
		auto& e = found->second;
		e.key = key, e.id = g.next_id++ & 0xffffffu, e.vs = d.vs, e.ps = d.ps;
		e.next_check = g.frame + 32 + (e.id & 31u);
	}
	auto& e     = found->second;
	e.last_seen = g.frame;
	if (e.capture_frame == g.frame) { // (drawn twice this frame: the first draw's capture serves)
		++g.draws_reused;
		g.cpu_note_ms += Ms(start);
		return false;
	}
	Job job;
	job.entry = &e;
	if (!e.built) {
		// A new mesh: within the frame's build budget only.
		const uint32_t tris = d.index_count / 3 * d.instance_count;
		if (g.builds >= BuildsPerFrame() || g.build_tris + tris > BuildTrisPerFrame()) {
			++g.skips[static_cast<uint32_t>(Skip::Budget)];
			g.cpu_note_ms += Ms(start);
			return false;
		}
		if (!e.created && !CreateEntry(*g_graphics, e, d)) {
			++g.skips[static_cast<uint32_t>(Skip::Topology)];
			g.entries.erase(found);
			g.cpu_note_ms += Ms(start);
			return false;
		}
		++g.builds;
		g.build_tris += e.tris;
		job.build = true;
	} else if (e.dynamic && ((g.frame + e.id) % RefitInterval()) == 0 && g.refits_frame < RefitsPerFrame()) {
		// (A moving mesh: refit every KYTY_HWRT_REFIT_EVERY frames (4), at most KYTY_HWRT_REFITS (64) a frame.)
		job.refit = true;
		++g.refits_frame;
	} else if (g.frame >= e.next_check && g.checks < MaxChecks) {
		job.mode   = Mode::Compare;
		job.result = g.checks++;
		e.next_check = g.frame + 32 + (e.id & 31u);
	} else {
		++g.draws_reused;
		g.cpu_note_ms += Ms(start);
		return false;
	}
	const uint32_t need = e.window * e.instances;
	if (g.captured + need > MaxCapture) {
		if (job.build) --g.builds, g.build_tris -= e.tris;
		if (job.mode == Mode::Compare) --g.checks;
		++g.skips[static_cast<uint32_t>(Skip::Records)];
		g.cpu_note_ms += Ms(start);
		return false;
	}
	rec.base      = g.captured;
	rec.window    = e.window;
	rec.instances = e.instances;
	job.src       = g.captured;
	g.captured += need;
	e.capture_frame = g.frame;
	g.jobs.push_back(job);
	++g.draws_captured;
	g.cpu_note_ms += Ms(start);
	return true;
}

void NoteCamera(uint64_t vs_hash, std::span<const uint32_t> srt) {
	if (!g.ready || g.have_camera || !FrameGen::IsGbufferVertexShader(vs_hash) || srt.size() < FrameGen::SrtViewProj + 16) return;
	for (uint32_t i = 0; i < 16; ++i) std::memcpy(&g.view_proj[i], &srt[FrameGen::SrtViewProj + i], 4);
	g.have_camera = std::all_of(g.view_proj.begin(), g.view_proj.end(), [](float f) { return std::isfinite(f); });
}

namespace {

using Dispatch = vk::detail::DispatchLoaderDynamic;

// Commands recorded in order with the renderer's own (KYTY_VULKAN_RECORDING: a packet the recording worker replays;
// without it, now). The payload owns everything the commands point at.
struct Work {
	vk::CommandBuffer                                      command;
	std::function<void(vk::CommandBuffer, const Dispatch&)> record;
};
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
void ReplayWork(std::span<const LocalVulkanRecording::Segment> segments, const Dispatch& dispatch) {
	const Work* work = nullptr;
	std::memcpy(static_cast<void*>(&work), segments[0].data, sizeof(work));
	work->record(work->command, dispatch);
	LocalVulkanRecording::InvalidateReplayState(); // (our pipelines and descriptors)
}
#endif
void Submit(vk::CommandBuffer command, std::function<void(vk::CommandBuffer, const Dispatch&)> record) {
	auto work     = std::make_shared<Work>();
	work->command = command;
	work->record  = std::move(record);
#if defined(KYTY_LOCAL_VULKAN_RECORDING)
	const Work*                          pointer = work.get();
	const LocalVulkanRecording::Segment segment {&pointer, sizeof(pointer)};
	if (!LocalVulkanRecording::EnqueuePacket(ReplayWork, {&segment, 1}, work)) LocalVulkanRecording::ReplayInline(ReplayWork, {&segment, 1});
#else
	work->record(work->command, VULKAN_HPP_DEFAULT_DISPATCHER);
#endif
}

void BarrierD(vk::CommandBuffer command, const Dispatch& d, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src,
              vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stage;
	barrier.srcAccessMask = src;
	barrier.dstStageMask  = dst_stage;
	barrier.dstAccessMask = dst;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(&dependency, d);
}

struct GatherPush {
	float    inv[16];
	float    cam[4];
	uint32_t src_base, count, mode, result;
	uint64_t src, dst, results;
};
static_assert(sizeof(GatherPush) <= 128);
struct DebugPush {
	float    inv[16];
	float    cam[4];
	uint32_t width, height, mode, pad;
};

// What one frame's build records (owned by its packet).
struct FramePayload {
	std::vector<GatherPush>                                        gathers;
	std::vector<vk::AccelerationStructureGeometryKHR>              geometries;
	std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>     infos;
	std::vector<vk::AccelerationStructureBuildRangeInfoKHR>        ranges;
	std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_ptrs;
	vk::AccelerationStructureGeometryKHR                           tlas_geometry {};
	vk::AccelerationStructureBuildGeometryInfoKHR                  tlas_info {};
	vk::AccelerationStructureBuildRangeInfoKHR                     tlas_range {};
	bool                                                           tlas = false;
	uint32_t                                                       slot = 0;
};

std::array<float, 16> g_inv {};
std::array<float, 4>  g_cam {};
bool                  g_frame_camera = false;

} // namespace

void EndFrame(GraphicContext& graphics, vk::CommandBuffer command, const char* where) {
	if (!Active()) return;
	std::lock_guard lock(g.mutex);
	g_graphics = &graphics;
	if (!Init(graphics)) return;
	const auto start = std::chrono::steady_clock::now();
	++g.build_calls;
	static bool said_where = false;
	if (!said_where) {
		said_where = true;
		std::printf("HWRT: the scene is built at %s\n", where);
		std::fflush(stdout);
	}
	ReadTiming(graphics);
	Collect(graphics, false);
	const uint32_t slot = g.slot;
	// The compare results of the frame that used the next slot last (Slots - 1 frames ago).
	for (const auto& [key, index]: g.check_keys[(slot + 1) % Slots]) {
		const auto found = g.entries.find(key);
		if (found == g.entries.end()) continue;
		float delta = 0.0f;
		std::memcpy(&delta, static_cast<const uint32_t*>(g.results[(slot + 1) % Slots].mapped) + index, 4);
		if (delta > 1.0f && !found->second.dynamic) {
			found->second.dynamic = true;
			++g.became_dynamic;
		}
	}
	g.check_keys[(slot + 1) % Slots].clear();
	++g.frames;
	std::array<double, 16> vp {}, inv {};
	for (int i = 0; i < 16; ++i) vp[i] = g.view_proj[i];
	const bool camera = g.have_camera && Invert4(vp, inv);
	for (int i = 0; i < 16; ++i) g_inv[i] = static_cast<float>(inv[i]);
	g_cam = {};
	if (camera && std::abs(inv[14]) > 1e-20)
		for (int i = 0; i < 3; ++i) g_cam[i] = static_cast<float>(inv[i * 4 + 2] / inv[14]);
	g_frame_camera = camera;
	if (!camera) ++g.frames_no_camera;

	auto             payload = std::make_shared<FramePayload>();
	payload->slot            = slot;
	std::vector<Job> jobs;
	jobs.swap(g.jobs);
	if (camera && !jobs.empty()) {
		auto* results = static_cast<uint32_t*>(g.results[slot].mapped);
		GatherPush push {};
		std::memcpy(push.inv, g_inv.data(), sizeof(push.inv));
		std::memcpy(push.cam, g_cam.data(), sizeof(push.cam));
		push.src     = g.capture.address;
		push.results = g.results[slot].address;
		for (const auto& job: jobs) {
			auto& e = *job.entry;
			if (!e.created) continue;
			push.src_base = job.src;
			push.count    = e.window * e.instances;
			push.mode     = static_cast<uint32_t>(job.mode);
			push.result   = job.result;
			push.dst      = e.vb.address;
			if (job.mode == Mode::Compare) {
				results[job.result] = 0;
				g.check_keys[slot].emplace_back(e.key, job.result);
				++g.compares;
			}
			payload->gathers.push_back(push);
			g.captured_vertices += push.count;
		}
		payload->geometries.reserve(jobs.size());
		std::vector<vk::DeviceSize> scratch_sizes;
		vk::DeviceSize              scratch_total = 0;
		for (const auto& job: jobs) {
			auto& e = *job.entry;
			if (!e.created || job.mode == Mode::Compare || !(job.build || job.refit)) continue;
			const bool update = job.refit && e.built && e.updatable;
			vk::AccelerationStructureGeometryTrianglesDataKHR tri {};
			tri.vertexFormat             = vk::Format::eR32G32B32Sfloat;
			tri.vertexData.deviceAddress = e.vb.address;
			tri.vertexStride             = 16;
			tri.maxVertex                = e.window * e.instances - 1;
			tri.indexType                = vk::IndexType::eUint32;
			tri.indexData.deviceAddress  = e.ib.address;
			vk::AccelerationStructureGeometryKHR geometry {};
			geometry.geometryType       = vk::GeometryTypeKHR::eTriangles;
			geometry.geometry.triangles = tri;
			geometry.flags              = vk::GeometryFlagBitsKHR::eOpaque;
			payload->geometries.push_back(geometry);
			vk::AccelerationStructureBuildGeometryInfoKHR info {};
			info.type  = vk::AccelerationStructureTypeKHR::eBottomLevel;
			info.flags = e.dynamic ? vk::BuildAccelerationStructureFlagsKHR(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
			                                                                vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
			                       : vk::BuildAccelerationStructureFlagsKHR(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
			info.mode          = update ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild;
			info.geometryCount = 1;
			info.pGeometries   = &payload->geometries.back();
			const uint32_t prims = e.tris;
			vk::AccelerationStructureBuildSizesInfoKHR sizes {};
			graphics.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info, &prims, &sizes);
			if (!update) {
				if (e.blas != nullptr) {
					Pending p;
					p.as    = e.blas;
					p.c     = e.as_buf;
					p.frame = g.frame;
					g.graveyard.push_back(p);
					e.blas   = nullptr;
					e.as_buf = {};
					++g.rebuilt;
				}
				if (!MakeAs(graphics, e.as_buf, e.blas, sizes.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel)) {
					payload->geometries.pop_back();
					e.built = false;
					continue;
				}
				vk::AccelerationStructureDeviceAddressInfoKHR address {};
				address.accelerationStructure = e.blas;
				e.blas_address                = graphics.device.getAccelerationStructureAddressKHR(address);
				e.updatable                   = e.dynamic;
				if (job.build && !e.dynamic) ++g.new_meshes;
			} else {
				++g.refits;
			}
			info.srcAccelerationStructure = update ? e.blas : nullptr;
			info.dstAccelerationStructure = e.blas;
			const auto need = (std::max(update ? sizes.updateScratchSize : sizes.buildScratchSize, vk::DeviceSize {1}) + g.scratch_align - 1) /
			                  g.scratch_align * g.scratch_align;
			scratch_sizes.push_back(need);
			scratch_total += need;
			payload->infos.push_back(info);
			vk::AccelerationStructureBuildRangeInfoKHR range {};
			range.primitiveCount = prims;
			payload->ranges.push_back(range);
			e.built = true;
			++e.built_count;
		}
		if (!payload->infos.empty()) {
			if (g.scratch.size < scratch_total + g.scratch_align) {
				if (g.scratch.buffer != VK_NULL_HANDLE) {
					Pending p;
					p.a     = g.scratch;
					p.frame = g.frame;
					g.graveyard.push_back(p);
					g.scratch = {};
				}
				if (!MakeBuffer(graphics, g.scratch, (scratch_total + g.scratch_align) * 3 / 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false)) {
					for (auto& info: payload->infos) {
						(void)info;
					}
					payload->infos.clear();
					payload->ranges.clear();
				}
			}
			vk::DeviceAddress at = (g.scratch.address + g.scratch_align - 1) / g.scratch_align * g.scratch_align;
			for (size_t i = 0; i < payload->infos.size(); ++i) {
				payload->infos[i].scratchData.deviceAddress = at;
				at += scratch_sizes[i];
			}
			for (size_t i = 0; i < payload->ranges.size(); ++i) payload->range_ptrs.push_back(&payload->ranges[i]);
		}
	}
	// The TLAS: every built mesh seen in the last KeepFrames() frames; meshes unseen for 4x that are freed.
	uint32_t instance_count = 0;
	{
		auto* instances = static_cast<VkAccelerationStructureInstanceKHR*>(g.instances[slot].mapped);
		auto* materials = static_cast<uint32_t*>(g.materials[slot].mapped);
		for (auto it = g.entries.begin(); it != g.entries.end();) {
			auto& e = it->second;
			if (e.last_seen + uint64_t {KeepFrames()} * 4 < g.frame) {
				Bury(e);
				++g.evicted;
				it = g.entries.erase(it);
				continue;
			}
			if (e.built && e.blas_address != 0 && e.last_seen + KeepFrames() >= g.frame && instance_count < MaxInstances) {
				VkAccelerationStructureInstanceKHR instance {};
				instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
				instance.instanceCustomIndex                    = e.id;
				instance.mask                                   = 0xff;
				instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
				instance.accelerationStructureReference         = e.blas_address;
				instances[instance_count]                       = instance;
				// (Phase 2's material hook: the draw's pixel and vertex shader, by TLAS instance index.)
				// (hwrt2: + its world-space vertices and indices, for a hit's geometric normal)
				materials[instance_count * 8 + 0] = static_cast<uint32_t>(e.ps);
				materials[instance_count * 8 + 1] = static_cast<uint32_t>(e.ps >> 32u);
				materials[instance_count * 8 + 2] = static_cast<uint32_t>(e.vs);
				materials[instance_count * 8 + 3] = e.id;
				materials[instance_count * 8 + 4] = static_cast<uint32_t>(e.vb.address);
				materials[instance_count * 8 + 5] = static_cast<uint32_t>(e.vb.address >> 32u);
				materials[instance_count * 8 + 6] = static_cast<uint32_t>(e.ib.address);
				materials[instance_count * 8 + 7] = static_cast<uint32_t>(e.ib.address >> 32u);
				++instance_count;
			}
			++it;
		}
	}
	g.tlas_instances_sum += instance_count;
	if (instance_count > 0) {
		vk::AccelerationStructureGeometryInstancesDataKHR instances_data {};
		instances_data.arrayOfPointers         = VK_FALSE;
		instances_data.data.deviceAddress      = g.instances[slot].address;
		payload->tlas_geometry.geometryType       = vk::GeometryTypeKHR::eInstances;
		payload->tlas_geometry.geometry.instances = instances_data;
		auto& tlas_info         = payload->tlas_info;
		tlas_info.type          = vk::AccelerationStructureTypeKHR::eTopLevel;
		tlas_info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
		tlas_info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
		tlas_info.geometryCount = 1;
		tlas_info.pGeometries   = &payload->tlas_geometry;
		const uint32_t max_instances = MaxInstances;
		vk::AccelerationStructureBuildSizesInfoKHR tlas_sizes {};
		graphics.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &tlas_info, &max_instances,
		                                                      &tlas_sizes);
		bool ok = g.tlas != nullptr || MakeAs(graphics, g.tlas_buf, g.tlas, tlas_sizes.accelerationStructureSize,
		                                      vk::AccelerationStructureTypeKHR::eTopLevel);
		static Buffer tlas_scratch;
		if (ok && tlas_scratch.buffer == VK_NULL_HANDLE)
			ok = MakeBuffer(graphics, tlas_scratch, tlas_sizes.buildScratchSize + g.scratch_align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
		if (ok) {
			tlas_info.dstAccelerationStructure  = g.tlas;
			tlas_info.scratchData.deviceAddress = (tlas_scratch.address + g.scratch_align - 1) / g.scratch_align * g.scratch_align;
			payload->tlas_range.primitiveCount  = instance_count;
			payload->tlas                       = true;
		}
	}
	g.tlas_built = payload->tlas;
	// (hwrt2) The frame's scene for the ray-traced effects: clip <-> world, the camera, the materials.
	g.ray_slot = -1;
	if (g.tlas_built && camera) {
		auto& ri = *static_cast<RayInfo*>(g.rayinfo[slot].mapped);
		std::memcpy(ri.inv, g_inv.data(), sizeof(ri.inv));
		for (int i = 0; i < 16; ++i) ri.vp[i] = g.view_proj[i];
		std::memcpy(ri.cam, g_cam.data(), sizeof(ri.cam));
		ri.info[0] = 1.0f;
		ri.info[1] = static_cast<float>(instance_count);
		ri.info[2] = ri.info[3] = 0.0f;
		ri.materials[0] = static_cast<uint32_t>(g.materials[slot].address);
		ri.materials[1] = static_cast<uint32_t>(g.materials[slot].address >> 32u);
		ri.materials[2] = ri.materials[3] = 0;
		g.ray_slot = static_cast<int>(slot);
	}
	// The GPU side, in order with the frame's commands.
	const bool       timed  = g.pool != nullptr;
	const auto       pool   = g.pool;
	const auto       gather = g.gather;
	const auto       layout = g.gather_layout;
	if (timed) g.pending_query[slot] = true;
	Submit(command, [payload, timed, pool, gather, layout](vk::CommandBuffer cmd, const Dispatch& d) {
		const uint32_t s = payload->slot;
		if (timed) {
			cmd.resetQueryPool(pool, s * 2, 2, d);
			cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, s * 2, d);
		}
		if (!payload->gathers.empty()) {
			// The draws' vertex-shader stores -> the gather's reads.
			BarrierD(cmd, d, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
			         vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, gather, d);
			for (const auto& push: payload->gathers) {
				cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push, d);
				cmd.dispatch((push.count + 255) / 256, 1, 1, d);
			}
			BarrierD(cmd, d, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
			         vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eVertexShader |
			             vk::PipelineStageFlagBits2::eHost,
			         vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eHostRead);
		}
		if (!payload->infos.empty())
			cmd.buildAccelerationStructuresKHR(static_cast<uint32_t>(payload->infos.size()), payload->infos.data(),
			                                   payload->range_ptrs.data(), d);
		if (payload->tlas) {
			BarrierD(cmd, d, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eComputeShader,
			         vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
			         vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
			         vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
			const vk::AccelerationStructureBuildRangeInfoKHR* tlas_ranges = &payload->tlas_range;
			cmd.buildAccelerationStructuresKHR(1, &payload->tlas_info, &tlas_ranges, d);
			BarrierD(cmd, d, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
			         vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
			         vk::AccessFlagBits2::eAccelerationStructureReadKHR);
		}
		if (timed) cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, s * 2 + 1, d);
	});
	static bool said = false;
	if (!said && g.tlas_built) {
		said = true;
		std::printf("HWRT: first TLAS: %u instances (%zu meshes known)\n", instance_count, g.entries.size());
		std::fflush(stdout);
	}
	++g.frame;
	g.slot        = (g.slot + 1) % Slots;
	g.next_record = NullRecords;
	g.captured = g.builds = g.build_tris = g.checks = g.refits_frame = 0;
	g.have_camera = false;
	g.cpu_after_ms += Ms(start);
	MaybeLog();
}

void DebugView(GraphicContext& graphics, vk::CommandBuffer command, vk::ImageView view, vk::Format format, vk::Extent2D extent) {
	if (!Active() || DebugMode() == 0) return;
	std::lock_guard lock(g.mutex);
	if (!g.ready || g.debug == nullptr || !g.tlas_built || !g_frame_camera || view == nullptr) return;
	const bool r11 = format == vk::Format::eB10G11R11UfloatPack32;
	if (format != vk::Format::eR16G16B16A16Sfloat && !(r11 && g.debug11 != nullptr)) {
		static bool said = false;
		if (!said) {
			said = true;
			std::printf("HWRT: debug view target format %d is neither RGBA16F nor B10G11R11: no debug view\n", static_cast<int>(format));
			std::fflush(stdout);
		}
		return;
	}
	++g.debug_calls;
	DebugPush push {};
	std::memcpy(push.inv, g_inv.data(), sizeof(push.inv));
	std::memcpy(push.cam, g_cam.data(), sizeof(push.cam));
	push.width  = extent.width;
	push.height = extent.height;
	push.mode   = DebugMode();
	const vk::Pipeline       pipeline = r11 ? g.debug11 : g.debug;
	const vk::PipelineLayout layout   = g.debug_layout;
	const auto tlas     = std::make_shared<vk::AccelerationStructureKHR>(g.tlas);
	(void)graphics;
	Submit(command, [push, pipeline, layout, tlas, view](vk::CommandBuffer cmd, const Dispatch& d) {
		BarrierD(cmd, d, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
		         vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead);
		vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
		as_write.accelerationStructureCount = 1;
		as_write.pAccelerationStructures    = tlas.get();
		vk::DescriptorImageInfo image_info {};
		image_info.imageView   = view;
		image_info.imageLayout = vk::ImageLayout::eGeneral;
		std::array<vk::WriteDescriptorSet, 2> writes {};
		writes[0].pNext           = &as_write;
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = vk::DescriptorType::eAccelerationStructureKHR;
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = vk::DescriptorType::eStorageImage;
		writes[1].pImageInfo      = &image_info;
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline, d);
		cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, static_cast<uint32_t>(writes.size()), writes.data(), d);
		cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push, d);
		cmd.dispatch((push.width + 7) / 8, (push.height + 7) / 8, 1, d);
		BarrierD(cmd, d, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
		         vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	});
}

bool CurrentRayScene(RayScene& out) {
	out = {};
	if (!Active()) return false;
	std::lock_guard lock(g.mutex);
	if (!g.ready || g.ray_slot < 0 || g.tlas == nullptr) return false;
	out.tlas   = reinterpret_cast<uint64_t>(static_cast<VkAccelerationStructureKHR>(g.tlas));
	out.info   = vk::Buffer(g.rayinfo[g.ray_slot].buffer);
	out.size   = sizeof(RayInfo);
	return true;
}

vk::AccelerationStructureKHR SceneTlas(bool* built_now) {
	if (built_now != nullptr) *built_now = false;
	if (!Active()) return nullptr;
	std::lock_guard lock(g.mutex);
	if (!g.ready || g.tlas == nullptr) return nullptr;
	if (built_now != nullptr) *built_now = g.tlas_built;
	return g.tlas;
}

void AfterTaa(GraphicContext& graphics, TextureCache& cache, vk::CommandBuffer command, std::span<const TextureBinding> images) {
	if (!Active()) return;
	EndFrame(graphics, command, "the TAA dispatch");
	if (DebugMode() != 0 && images.size() >= 7 && images[6].image_id && images[6].image_view != nullptr) {
		auto& output = cache.GetImage(images[6].image_id);
		DebugView(graphics, command, images[6].image_view, output.backing.format,
		          vk::Extent2D {output.backing.extent.width, output.backing.extent.height});
	}
}

} // namespace Libs::Graphics::HwScene
