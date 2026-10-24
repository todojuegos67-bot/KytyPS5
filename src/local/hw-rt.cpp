#include "local/hw-rt.h"

#include "graphics/host_gpu/renderer/pipeline/moduleStore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

// KYTY_RT_SHADOWS_HW (hw-rt.h). Everything below is reached only with the switch on and a device that has the features.
namespace Libs::Graphics::HwRt {
namespace {

// Sizes per frame slot (3 slots: the CPU fills one while the GPU may still read the others).
constexpr uint32_t Slots         = 3;
constexpr uint32_t MaxVertices   = 2u << 20u;  // per slot (16 B each: 32 MiB)
constexpr uint32_t MaxIndices    = 6u << 20u;  // per slot (4 B each: 24 MiB)
constexpr uint32_t MaxDraws      = 8192;       // per slot
constexpr uint32_t MaxDrawIndices = 1u << 20u; // a larger draw is not captured

bool g_supported = false;

struct Draw {
	uint32_t              vbase = 0, vcount = 0, ibase = 0, icount = 0;
	std::array<double, 12> light_to_world {}; // 3x4 rows
};

struct Buffer {
	VkBuffer        buffer = VK_NULL_HANDLE;
	VmaAllocation   memory = nullptr;
	void*           mapped = nullptr;
	vk::DeviceAddress address = 0;
	vk::DeviceSize  size = 0;
};

struct State {
	std::mutex mutex;
	bool       init_tried = false, ready = false;
	Buffer     capture, cursor, indices[Slots], transforms[Slots], instances[Slots], scratch, blas_buf, tlas_buf;
	vk::AccelerationStructureKHR blas = nullptr, tlas = nullptr;
	vk::DeviceSize scratch_align = 256;
	uint32_t   slot = 0;
	std::vector<Draw> draws[Slots];
	uint32_t   vused = 0, iused = 0;
	bool       overflow = false;
	// Refit: the last build's primitive counts per geometry.
	std::vector<uint32_t> last_counts;
	bool       have_blas = false;
	// Variant modules by original.
	std::unordered_map<VkShaderModule, VkShaderModule> modules;
	// Timing.
	vk::QueryPool pool = nullptr;
	double     period_ns = 1.0;
	uint64_t   frames_built = 0, frames_read = 0;
	bool       pending[Slots] {};
	uint32_t   trace_slot = 0;
	// The 10 s log.
	double     sum_tris = 0.0, sum_build = 0.0, sum_trace = 0.0, shadowed = -1.0;
	uint32_t   n_frames = 0, n_timed = 0, refits = 0, draws_seen = 0, draws_skipped = 0, distinct_matrices = 0;
	std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
} g;

void Say(const char* fmt, const char* what) {
	std::printf(fmt, what);
	std::fflush(stdout);
}

bool HasExtension(const std::vector<vk::ExtensionProperties>& available, const char* name) {
	return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
}
bool HasName(const std::vector<const char*>& list, const char* name) {
	return std::any_of(list.begin(), list.end(), [&](const char* e) { return std::strcmp(e, name) == 0; });
}

bool MakeBuffer(GraphicContext& graphics, Buffer& out, vk::DeviceSize size, VkBufferUsageFlags usage, bool host) {
	VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	create.size  = size;
	create.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (host) allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo info {};
	if (vmaCreateBuffer(graphics.allocator, &create, &allocate, &out.buffer, &out.memory, &info) != VK_SUCCESS) {
		out = {};
		return false;
	}
	out.mapped = info.pMappedData;
	out.size   = size;
	vk::BufferDeviceAddressInfo address {};
	address.buffer = vk::Buffer(out.buffer);
	out.address    = graphics.device.getBufferAddress(address);
	if (host && out.mapped == nullptr) return false;
	return true;
}
void FreeBuffer(GraphicContext& graphics, Buffer& b) {
	if (b.buffer != VK_NULL_HANDLE) vmaDestroyBuffer(graphics.allocator, b.buffer, b.memory);
	b = {};
}

bool Init(GraphicContext& graphics) {
	if (g.init_tried) return g.ready;
	g.init_tried = true;
	if (!g_supported) return false;
	const VkBufferUsageFlags as_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	bool ok = MakeBuffer(graphics, g.capture, vk::DeviceSize {MaxVertices} * Slots * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | as_input, false);
	ok = ok && MakeBuffer(graphics, g.cursor, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
	for (uint32_t s = 0; s < Slots && ok; ++s) {
		ok = ok && MakeBuffer(graphics, g.indices[s], vk::DeviceSize {MaxIndices} * 4, as_input, true);
		ok = ok && MakeBuffer(graphics, g.transforms[s], vk::DeviceSize {MaxDraws} * sizeof(VkTransformMatrixKHR), as_input, true);
		ok = ok && MakeBuffer(graphics, g.instances[s], sizeof(VkAccelerationStructureInstanceKHR), as_input, true);
	}
	if (ok) {
		vk::PhysicalDeviceAccelerationStructurePropertiesKHR as_props {};
		vk::PhysicalDeviceProperties2                        props {};
		props.pNext = &as_props;
		graphics.physical_device.getProperties2(&props);
		g.scratch_align = std::max<vk::DeviceSize>(as_props.minAccelerationStructureScratchOffsetAlignment, 1);
		g.period_ns     = props.properties.limits.timestampPeriod;
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = Slots * 4;
		if (graphics.device.createQueryPool(&query, nullptr, &g.pool) != vk::Result::eSuccess) g.pool = nullptr;
	}
	if (!ok) {
		std::printf("RTSHADOW HW: could not allocate the capture / AS buffers: screen-space only\n");
		std::fflush(stdout);
		return false;
	}
	g.ready = true;
	std::printf("RTSHADOW HW: ready (capture %u vertices x %u slots at 0x%llx, cursor at 0x%llx)\n", MaxVertices, Slots,
	            static_cast<unsigned long long>(g.capture.address), static_cast<unsigned long long>(g.cursor.address));
	std::fflush(stdout);
	return true;
}

// ---- The SPIR-V patch: store gl_Position to capture[cursor + gl_VertexIndex] before each return of the entry point.
struct SpirvPatcher {
	const std::vector<uint32_t>& in;
	uint32_t                     bound = 0;
	struct Inst {
		size_t   at;
		uint32_t op, count;
	};
	std::vector<Inst> insts;
	std::vector<uint32_t> new_types, new_decorations;
	std::string error;

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
		// Types made by this patch.
		for (size_t k = 0; k < new_types.size();) {
			const uint32_t n = new_types[k] >> 16u, o = new_types[k] & 0xffffu;
			if (o == op && n == operands.size() + 2 && std::equal(operands.begin(), operands.end(), new_types.begin() + k + 2))
				return new_types[k + 1];
			k += n;
		}
		const uint32_t id = Id();
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

	bool Run(std::vector<uint32_t>& out, uint64_t capture_address, uint64_t cursor_address, uint32_t capacity) {
		if (in.size() < 5 || in[0] != 0x07230203u) return error = "not SPIR-V", false;
		const uint32_t version = in[1];
		bound = in[3];
		for (size_t at = 5; at < in.size();) {
			const uint32_t count = in[at] >> 16u, op = in[at] & 0xffffu;
			if (count == 0 || at + count > in.size()) return error = "malformed", false;
			insts.push_back({at, op, count});
			at += count;
		}
		// The vertex entry point.
		uint32_t entry = 0;
		size_t   entry_inst = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k)
			if (insts[k].op == 15 && W(insts[k])[1] == 0) {
				if (entry != 0) return error = "several vertex entry points", false;
				entry = W(insts[k])[2], entry_inst = k;
			}
		if (entry == 0) return error = "no vertex entry point", false;
		bool has_int64 = false, has_psb = false, has_ext = false;
		uint32_t pos_var = 0, pos_struct = 0, pos_member = 0, vi_var = 0;
		for (const auto& i: insts) {
			const uint32_t* w = W(i);
			if (i.op == 17 && w[1] == 11) has_int64 = true;
			if (i.op == 17 && w[1] == 5347) has_psb = true;
			if (i.op == 10) {
				const char* name = reinterpret_cast<const char*>(w + 1);
				if (std::strncmp(name, "SPV_KHR_physical_storage_buffer", (i.count - 1) * 4) == 0) has_ext = true;
			}
			if (i.op == 71 && i.count >= 4 && w[2] == 11 && w[3] == 0) pos_var = w[1];
			if (i.op == 71 && i.count >= 4 && w[2] == 11 && w[3] == 42) vi_var = w[1];
			if (i.op == 72 && i.count >= 5 && w[3] == 11 && w[4] == 0) pos_struct = w[1], pos_member = w[2];
		}
		// Types of the globals: var -> pointer type -> pointee.
		std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointers; // id -> (storage, pointee)
		std::unordered_map<uint32_t, uint32_t>                      var_type; // var id -> pointer type
		std::unordered_map<uint32_t, const uint32_t*>               types;
		for (const auto& i: insts) {
			const uint32_t* w = W(i);
			if (i.op == 32) pointers[w[1]] = {w[2], w[3]};
			if (i.op == 59) var_type[w[2]] = w[1];
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
		// Types and constants.
		const uint32_t t_bool = FindType(20, {});
		const uint32_t t_u32  = FindType(21, {32, 0});
		const uint32_t t_u64  = FindType(21, {64, 0});
		const uint32_t t_pu32 = FindType(32, {5349, t_u32});
		const uint32_t t_pv4  = FindType(32, {5349, vec4});
		uint32_t       t_vi   = t_u32;
		bool           vi_signed = false, vi_new = false;
		if (vi_var != 0) {
			const auto p = pointers.find(var_type[vi_var]);
			if (p == pointers.end() || p->second.first != 1) return error = "VertexIndex is not an Input variable", false;
			t_vi = p->second.second;
			const auto t = types.find(t_vi);
			if (t == types.end() || (t->second[0] & 0xffffu) != 21 || t->second[2] != 32) return error = "VertexIndex type", false;
			vi_signed = t->second[3] != 0;
		} else {
			const uint32_t t_pin = FindType(32, {1, t_u32});
			vi_var = Id(), vi_new = true;
			Emit(new_types, 59, {t_pin, vi_var, 1});
			Emit(new_decorations, 71, {vi_var, 11, 42});
		}
		uint32_t t_pout = 0, c_member = 0;
		if (pos_var == 0) {
			t_pout   = FindType(32, {3, vec4});
			c_member = Const32(t_u32, pos_member);
		}
		const uint32_t c_cursor = Const64(t_u64, cursor_address);
		const uint32_t c_cap    = Const64(t_u64, capture_address);
		const uint32_t c_16     = Const64(t_u64, 16);
		const uint32_t c_limit  = Const32(t_u32, capacity);

		// The entry point's function: each OpReturn.
		size_t first_function = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k)
			if (insts[k].op == 54) { first_function = k; break; }
		size_t types_start = SIZE_MAX;
		for (size_t k = 0; k < insts.size(); ++k) {
			const uint32_t op = insts[k].op;
			if ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59 || op == 1 || op == 54) { types_start = k; break; }
		}
		if (first_function == SIZE_MAX || types_start == SIZE_MAX) return error = "no function", false;

		out.clear();
		out.reserve(in.size() + 256);
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
				if (!has_ext && version < 0x00010500u) {
					const char   name[] = "SPV_KHR_physical_storage_buffer"; // (31 chars + nul: 8 words)
					uint32_t     words[8] {};
					std::memcpy(words, name, sizeof(name));
					out.push_back((9u << 16u) | 10u);
					out.insert(out.end(), words, words + 8);
				}
			}
			if (k == types_start) out.insert(out.end(), new_decorations.begin(), new_decorations.end());
			if (k == first_function) out.insert(out.end(), new_types.begin(), new_types.end());
			if (i.op == 14) { // OpMemoryModel: PhysicalStorageBuffer64 addressing
				out.push_back(w[0]);
				out.push_back(5348);
				out.insert(out.end(), w + 2, w + i.count);
				continue;
			}
			if (k == entry_inst && vi_new) {
				out.push_back(((i.count + 1) << 16u) | 15u);
				out.insert(out.end(), w + 1, w + i.count);
				out.push_back(vi_var);
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
				const uint32_t pos = Id(), vi = Id(), vu = vi_signed ? Id() : vi, cur_p = Id(), cur = Id(), e = Id(), ok = Id();
				const uint32_t then = Id(), merge = Id(), e64 = Id(), off = Id(), addr = Id(), dst = Id();
				Emit(out, 61, {vec4, pos, ptr});
				Emit(out, 61, {t_vi, vi, vi_var});
				if (vi_signed) Emit(out, 124, {t_u32, vu, vi});
				Emit(out, 120, {t_pu32, cur_p, c_cursor});
				Emit(out, 61, {t_u32, cur, cur_p, 2, 4});
				Emit(out, 128, {t_u32, e, cur, vu});
				Emit(out, 176, {t_bool, ok, e, c_limit});
				Emit(out, 247, {merge, 0});
				Emit(out, 250, {ok, then, merge});
				Emit(out, 248, {then});
				Emit(out, 113, {t_u64, e64, e});
				Emit(out, 132, {t_u64, off, e64, c_16});
				Emit(out, 128, {t_u64, addr, c_cap, off});
				Emit(out, 120, {t_pv4, dst, addr});
				Emit(out, 62, {dst, pos, 2, 16});
				Emit(out, 249, {merge});
				Emit(out, 248, {merge});
				Emit(out, 253, {});
				continue;
			}
			out.insert(out.end(), w, w + i.count);
		}
		if (returns == 0) return error = "no return in the entry point", false;
		out[3] = bound;
		if (out[1] < 0x00010300u) out[1] = 0x00010300u; // (PhysicalStorageBuffer needs SPIR-V 1.3 with the extension)
		return true;
	}
};

uint32_t LightMatrix(std::span<const uint32_t> srt, std::array<double, 12>& m) {
	// The first orthographic matrix in the SRT (as FindOrthoAxes in frame-gen.cpp): rows clip = M * [p, 1].
	std::vector<float> f(std::min<size_t>(srt.size(), 160));
	std::memcpy(f.data(), srt.data(), f.size() * 4u);
	const auto ortho = [](const float* r0, const float* r1, const float* r2) {
		const double l0 = std::sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
		const double l1 = std::sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
		const double l2 = std::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
		if (!(l0 > 1e-8 && l1 > 1e-8 && l2 > 1e-8)) return false;
		const auto cosv = [](const float* a, const float* b, double la, double lb) {
			return std::abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (la * lb);
		};
		return cosv(r0, r1, l0, l1) < 0.01 && cosv(r0, r2, l0, l2) < 0.01 && cosv(r1, r2, l1, l2) < 0.01 &&
		       std::abs(l0 - l1) < 0.01 * l0 && std::abs(l2 - l0) > 0.01 * l0;
	};
	const auto finite = [&](size_t o, size_t n) {
		for (size_t i = 0; i < n; ++i)
			if (!std::isfinite(f[o + i])) return false;
		return true;
	};
	for (size_t o = 0; o + 12 <= f.size(); ++o) {
		if (!finite(o, 12)) continue;
		const float* a = &f[o];
		if (o + 16 <= f.size() && finite(o, 16)) {
			if (a[12] == 0.0f && a[13] == 0.0f && a[14] == 0.0f && a[15] == 1.0f && ortho(a, a + 4, a + 8)) {
				for (int i = 0; i < 12; ++i) m[i] = a[i];
				return 1;
			}
			if (a[3] == 0.0f && a[7] == 0.0f && a[11] == 0.0f && a[15] == 1.0f) {
				const float c0[3] {a[0], a[4], a[8]}, c1[3] {a[1], a[5], a[9]}, c2[3] {a[2], a[6], a[10]};
				if (ortho(c0, c1, c2)) {
					for (int i = 0; i < 3; ++i)
						for (int j = 0; j < 4; ++j) m[i * 4 + j] = a[j * 4 + i];
					return 2;
				}
			}
		}
		if ((o & 3u) == 0 && ortho(a, a + 4, a + 8)) {
			for (int i = 0; i < 12; ++i) m[i] = a[i];
			return 3;
		}
	}
	return 0;
}

bool InvertAffine(const std::array<double, 12>& m, std::array<double, 12>& r) {
	const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g2 = m[8], h = m[9], i = m[10];
	const double det = a * (e * i - f * h) - b * (d * i - f * g2) + c * (d * h - e * g2);
	if (!(std::abs(det) > 1e-30) || !std::isfinite(det)) return false;
	const double inv[9] {(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
	                     (f * g2 - d * i) / det, (a * i - c * g2) / det, (c * d - a * f) / det,
	                     (d * h - e * g2) / det, (b * g2 - a * h) / det, (a * e - b * d) / det};
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) r[row * 4 + col] = inv[row * 3 + col];
		r[row * 4 + 3] = -(inv[row * 3] * m[3] + inv[row * 3 + 1] * m[7] + inv[row * 3 + 2] * m[11]);
	}
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
	if (as != nullptr && buf.size >= size) return true;
	if (as != nullptr || buf.buffer != VK_NULL_HANDLE) {
		graphics.device.waitIdle(); // (rare: grows by 1.5x)
		if (as != nullptr) graphics.device.destroyAccelerationStructureKHR(as);
		as = nullptr;
		FreeBuffer(graphics, buf);
	}
	size = size + size / 2;
	if (!MakeBuffer(graphics, buf, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) return false;
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = vk::Buffer(buf.buffer);
	create.size   = size;
	create.type   = type;
	if (graphics.device.createAccelerationStructureKHR(&create, nullptr, &as) != vk::Result::eSuccess) {
		as = nullptr;
		return false;
	}
	return true;
}

void ReadTiming(GraphicContext& graphics) {
	if (g.pool == nullptr) return;
	for (uint32_t s = 0; s < Slots; ++s) {
		if (!g.pending[s] || s == g.slot) continue;
		std::array<uint64_t, 8> r {};
		if (graphics.device.getQueryPoolResults(g.pool, s * 4, 4, sizeof(r), r.data(), 2 * sizeof(uint64_t),
		                                        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability) !=
		    vk::Result::eSuccess)
			continue;
		if (r[1] == 0 || r[3] == 0 || r[5] == 0 || r[7] == 0) continue;
		g.pending[s] = false;
		g.sum_build += static_cast<double>(r[2] - r[0]) * g.period_ns / 1e6;
		g.sum_trace += static_cast<double>(r[6] - r[4]) * g.period_ns / 1e6;
		++g.n_timed;
	}
}

void MaybeLog() {
	const auto now = std::chrono::steady_clock::now();
	if (now - g.last_log < std::chrono::seconds(10)) return;
	if (g.n_frames > 0) {
		std::printf("RTSHADOW HW: stats: %.0f triangles captured per frame (%u draws captured, %u skipped in 10 s; %u distinct light "
		            "matrices last frame), AS build %.3f ms, trace %.3f ms (%u timed), %u refits of %u builds, %.1f %% shadowed%s\n",
		            g.sum_tris / g.n_frames, g.draws_seen, g.draws_skipped, g.distinct_matrices,
		            g.n_timed > 0 ? g.sum_build / g.n_timed : 0.0, g.n_timed > 0 ? g.sum_trace / g.n_timed : 0.0, g.n_timed, g.refits,
		            g.n_frames, g.shadowed, g.overflow ? " (capture buffers full: some draws left out)" : "");
		std::fflush(stdout);
	}
	g.sum_tris = g.sum_build = g.sum_trace = 0.0;
	g.n_frames = g.n_timed = g.refits = g.draws_seen = g.draws_skipped = 0;
	g.overflow = false;
	g.last_log = now;
}

} // namespace

bool Requested() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RT_SHADOWS_HW");
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
			Say("RTSHADOW HW: the device has no %s: hardware ray tracing off (screen-space shadows)\n", name);
			return;
		}
	}
	for (const char* name: needed)
		if (!HasName(device_extensions, name)) device_extensions.push_back(name);
}

void* ChainDeviceFeatures(vk::PhysicalDevice physical, const std::vector<const char*>& device_extensions, void* next) {
	if (!Requested()) return next;
	if (!HasName(device_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
	    !HasName(device_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME))
		return next;
	vk::PhysicalDeviceAccelerationStructureFeaturesKHR as {};
	vk::PhysicalDeviceRayQueryFeaturesKHR              rq {};
	as.pNext = &rq;
	vk::PhysicalDeviceFeatures2 features {};
	features.pNext = &as;
	physical.getFeatures2(&features);
	if (!as.accelerationStructure || !rq.rayQuery) {
		Say("RTSHADOW HW: %s: hardware ray tracing off (screen-space shadows)\n",
		    !as.accelerationStructure ? "no accelerationStructure feature" : "no rayQuery feature");
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
	std::printf("RTSHADOW HW: VK_KHR_acceleration_structure + VK_KHR_ray_query enabled\n");
	std::fflush(stdout);
	return &s_as;
}

bool Supported() { return g_supported; }

bool CaptureWanted() { return g_supported && Requested(); }

namespace {
std::array<std::atomic<uint64_t>, 64> g_shadow_vs {};
} // namespace
void NoteShadowVs(uint64_t vs_hash) {
	if (vs_hash == 0 || IsShadowVs(vs_hash)) return;
	for (auto& slot: g_shadow_vs) {
		uint64_t expected = 0;
		if (slot.compare_exchange_strong(expected, vs_hash, std::memory_order_relaxed)) {
			std::printf("RTSHADOW HW: shadow-map vertex shader %016llx (its draws take the capturing path)\n",
			            static_cast<unsigned long long>(vs_hash));
			std::fflush(stdout);
			return;
		}
		if (expected == vs_hash) return;
	}
}
bool IsShadowVs(uint64_t vs_hash) {
	if (!CaptureWanted()) return false;
	if (vs_hash == 0xdaa84716d49bf71dULL) return true;
	for (const auto& slot: g_shadow_vs) {
		const uint64_t v = slot.load(std::memory_order_relaxed);
		if (v == vs_hash) return true;
		if (v == 0) return false;
	}
	return false;
}

vk::ShaderModule CaptureModule(GraphicContext& graphics, vk::ShaderModule original) {
	if (!CaptureWanted() || original == nullptr) return nullptr;
	std::lock_guard lock(g.mutex);
	if (!Init(graphics)) return nullptr;
	const auto key = static_cast<VkShaderModule>(original);
	if (auto found = g.modules.find(key); found != g.modules.end()) return vk::ShaderModule(found->second);
	VkShaderModule made = VK_NULL_HANDLE;
	const auto     spirv = ModuleStore::Holds(original) ? ModuleStore::Load(original) : nullptr;
	if (spirv == nullptr) {
		static bool said = false;
		if (!said) said = true, Say("RTSHADOW HW: %s\n", "a shadow vertex shader's SPIR-V is not readable (module store off): not captured");
	} else {
		std::vector<uint32_t> patched;
		SpirvPatcher          patcher(*spirv);
		if (!patcher.Run(patched, g.capture.address, g.cursor.address, MaxVertices * Slots)) {
			static int said = 0;
			if (said++ < 8) Say("RTSHADOW HW: a shadow vertex shader could not be patched (%s): not captured\n", patcher.error.c_str());
		} else {
			vk::ShaderModule module = ModuleStore::Enabled() ? ModuleStore::Add(patched) : vk::ShaderModule {};
			if (module == nullptr) {
				vk::ShaderModuleCreateInfo create {};
				create.codeSize = patched.size() * sizeof(uint32_t);
				create.pCode    = patched.data();
				if (graphics.device.createShaderModule(&create, nullptr, &module) != vk::Result::eSuccess) module = nullptr;
			}
			made = static_cast<VkShaderModule>(module);
			static bool said = false;
			if (!said && made != VK_NULL_HANDLE)
				said = true, Say("RTSHADOW HW: %s\n", "first shadow vertex shader patched (gl_Position capture)");
		}
	}
	g.modules[key] = made;
	return vk::ShaderModule(made);
}

uint64_t VariantId(uint64_t program_id) {
	uint64_t h = program_id ^ 0x9e3779b97f4a7c15ULL;
	for (const uint64_t v: {g.capture.address, g.cursor.address, uint64_t {MaxVertices * Slots}}) {
		h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
		h *= 0xff51afd7ed558ccdULL;
		h ^= h >> 33u;
	}
	return h | (uint64_t {1} << 63u);
}

bool PrepareDraw(GraphicContext& graphics, std::span<const uint32_t> srt, const void* indices, uint32_t element_size, uint32_t count,
                 int32_t vertex_offset, uint32_t first_vertex, uint32_t* cursor) {
	if (!CaptureWanted() || count < 3 || count > MaxDrawIndices) return false;
	std::array<double, 12> light {}, inv {};
	if (LightMatrix(srt, light) == 0 || !InvertAffine(light, inv)) return false;
	std::lock_guard lock(g.mutex);
	if (!Init(graphics)) return false;
	auto& list = g.draws[g.slot];
	const uint32_t tri_count = count - count % 3;
	uint32_t       lo = UINT32_MAX, hi = 0;
	if (indices != nullptr) {
		for (uint32_t i = 0; i < tri_count; ++i) {
			const uint32_t v = element_size == 2 ? static_cast<const uint16_t*>(indices)[i] : static_cast<const uint32_t*>(indices)[i];
			lo = std::min(lo, v), hi = std::max(hi, v);
		}
	} else {
		lo = 0, hi = tri_count - 1;
	}
	const uint32_t vcount = hi - lo + 1;
	if (list.size() >= MaxDraws || g.vused + vcount > MaxVertices || g.iused + tri_count > MaxIndices || vcount > MaxDrawIndices) {
		g.overflow = true;
		++g.draws_skipped;
		return false;
	}
	Draw d {};
	d.vbase = g.vused, d.vcount = vcount, d.ibase = g.iused, d.icount = tri_count, d.light_to_world = inv;
	auto* dst = static_cast<uint32_t*>(g.indices[g.slot].mapped) + g.iused;
	if (indices != nullptr) {
		for (uint32_t i = 0; i < tri_count; ++i)
			dst[i] = (element_size == 2 ? static_cast<const uint16_t*>(indices)[i] : static_cast<const uint32_t*>(indices)[i]) - lo;
	} else {
		for (uint32_t i = 0; i < tri_count; ++i) dst[i] = i;
	}
	g.vused += vcount, g.iused += tri_count;
	list.push_back(d);
	++g.draws_seen;
	// gl_VertexIndex = index + vertex offset (indexed) or first_vertex + i: element = slot base + vbase + (index - lo).
	const uint32_t first = indices != nullptr ? static_cast<uint32_t>(vertex_offset) + lo : first_vertex;
	*cursor = g.slot * MaxVertices + d.vbase - first;
	return true;
}

void RecordCursor(vk::CommandBuffer command, uint32_t cursor) {
	Barrier(command, vk::PipelineStageFlagBits2::eVertexShader, vk::AccessFlagBits2::eShaderStorageRead, vk::PipelineStageFlagBits2::eTransfer,
	        vk::AccessFlagBits2::eTransferWrite);
	command.updateBuffer(vk::Buffer(g.cursor.buffer), 0, sizeof(cursor), &cursor);
	Barrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eVertexShader,
	        vk::AccessFlagBits2::eShaderStorageRead);
}

vk::AccelerationStructureKHR Build(GraphicContext& graphics, vk::CommandBuffer command, const std::array<double, 12>& w2v) {
	if (!CaptureWanted()) return nullptr;
	std::lock_guard lock(g.mutex);
	if (!Init(graphics)) return nullptr;
	ReadTiming(graphics);
	const uint32_t slot = g.slot;
	auto&          list = g.draws[slot];
	// The next frame's capture goes to the next slot.
	g.slot  = (g.slot + 1) % Slots;
	g.draws[g.slot].clear();
	g.vused = g.iused = 0;
	MaybeLog();
	if (list.empty()) return nullptr;
	// Geometries: one per draw; transforms = world -> view * light clip -> world.
	auto* xf = static_cast<VkTransformMatrixKHR*>(g.transforms[slot].mapped);
	std::vector<vk::AccelerationStructureGeometryKHR>       geometries(list.size());
	std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(list.size());
	std::vector<uint32_t>                                   counts(list.size());
	double                                                  tris = 0.0;
	std::vector<std::array<double, 12>>                     distinct;
	for (size_t k = 0; k < list.size(); ++k) {
		const auto& d = list[k];
		const auto& l = d.light_to_world;
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 4; ++j) {
				double v = w2v[i * 4 + 0] * l[0 * 4 + j] + w2v[i * 4 + 1] * l[1 * 4 + j] + w2v[i * 4 + 2] * l[2 * 4 + j];
				if (j == 3) v += w2v[i * 4 + 3];
				xf[k].matrix[i][j] = static_cast<float>(v);
			}
		if (distinct.size() < 64 &&
		    std::none_of(distinct.begin(), distinct.end(), [&](const auto& m) { return std::equal(m.begin(), m.end(), l.begin()); }))
			distinct.push_back(l);
		vk::AccelerationStructureGeometryTrianglesDataKHR tri {};
		tri.vertexFormat             = vk::Format::eR32G32B32Sfloat;
		tri.vertexData.deviceAddress = g.capture.address + (vk::DeviceSize {slot} * MaxVertices + d.vbase) * 16;
		tri.vertexStride             = 16;
		tri.maxVertex                = d.vcount - 1;
		tri.indexType                = vk::IndexType::eUint32;
		tri.indexData.deviceAddress  = g.indices[slot].address + vk::DeviceSize {d.ibase} * 4;
		tri.transformData.deviceAddress = g.transforms[slot].address + k * sizeof(VkTransformMatrixKHR);
		geometries[k].geometryType       = vk::GeometryTypeKHR::eTriangles;
		geometries[k].geometry.triangles = tri;
		geometries[k].flags              = vk::GeometryFlagBitsKHR::eOpaque;
		counts[k]                        = d.icount / 3;
		ranges[k].primitiveCount         = counts[k];
		tris += counts[k];
	}
	g.distinct_matrices = static_cast<uint32_t>(distinct.size());
	const bool refit = g.have_blas && counts == g.last_counts;
	vk::AccelerationStructureBuildGeometryInfoKHR blas_info {};
	blas_info.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
	blas_info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
	                          vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
	blas_info.mode          = refit ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild;
	blas_info.geometryCount = static_cast<uint32_t>(geometries.size());
	blas_info.pGeometries   = geometries.data();
	vk::AccelerationStructureBuildSizesInfoKHR blas_sizes {};
	graphics.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &blas_info, counts.data(),
	                                                      &blas_sizes);
	// The TLAS: one instance (identity: the geometry is in view space already).
	vk::AccelerationStructureGeometryKHR tlas_geometry {};
	vk::AccelerationStructureGeometryInstancesDataKHR instances_data {};
	instances_data.arrayOfPointers    = VK_FALSE;
	instances_data.data.deviceAddress = g.instances[slot].address;
	tlas_geometry.geometryType        = vk::GeometryTypeKHR::eInstances;
	tlas_geometry.geometry.instances  = instances_data;
	vk::AccelerationStructureBuildGeometryInfoKHR tlas_info {};
	tlas_info.type          = vk::AccelerationStructureTypeKHR::eTopLevel;
	tlas_info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	tlas_info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	tlas_info.geometryCount = 1;
	tlas_info.pGeometries   = &tlas_geometry;
	const uint32_t one      = 1;
	vk::AccelerationStructureBuildSizesInfoKHR tlas_sizes {};
	graphics.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &tlas_info, &one, &tlas_sizes);

	const bool grew_blas = g.blas == nullptr || g.blas_buf.size < blas_sizes.accelerationStructureSize;
	if (!MakeAs(graphics, g.blas_buf, g.blas, blas_sizes.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel) ||
	    !MakeAs(graphics, g.tlas_buf, g.tlas, tlas_sizes.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eTopLevel))
		return nullptr;
	if (grew_blas && refit) {
		blas_info.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	}
	const bool     do_refit = blas_info.mode == vk::BuildAccelerationStructureModeKHR::eUpdate;
	const auto     align    = [](vk::DeviceSize v, vk::DeviceSize a) { return (v + a - 1) / a * a; };
	const vk::DeviceSize blas_scratch = align(std::max(blas_sizes.buildScratchSize, blas_sizes.updateScratchSize), g.scratch_align);
	const vk::DeviceSize need         = blas_scratch + align(tlas_sizes.buildScratchSize, g.scratch_align) + g.scratch_align;
	if (g.scratch.size < need) {
		if (g.scratch.buffer != VK_NULL_HANDLE) graphics.device.waitIdle();
		FreeBuffer(graphics, g.scratch);
		if (!MakeBuffer(graphics, g.scratch, need + need / 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false)) return nullptr;
	}
	const vk::DeviceAddress scratch0 = align(g.scratch.address, g.scratch_align);

	vk::AccelerationStructureDeviceAddressInfoKHR blas_address {};
	blas_address.accelerationStructure = g.blas;
	VkAccelerationStructureInstanceKHR instance {};
	instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
	instance.mask                           = 0xff;
	instance.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instance.accelerationStructureReference = graphics.device.getAccelerationStructureAddressKHR(blas_address);
	std::memcpy(g.instances[slot].mapped, &instance, sizeof(instance));

	if (g.pool != nullptr) {
		command.resetQueryPool(g.pool, slot * 4, 4);
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, slot * 4);
	}
	// The shadow draws' vertex-shader stores (and anything before) -> the build's reads.
	Barrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
	        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eAccelerationStructureWriteKHR |
	            vk::AccessFlagBits2::eShaderRead);
	blas_info.srcAccelerationStructure  = do_refit ? g.blas : nullptr;
	blas_info.dstAccelerationStructure  = g.blas;
	blas_info.scratchData.deviceAddress = scratch0;
	const vk::AccelerationStructureBuildRangeInfoKHR* blas_ranges = ranges.data();
	command.buildAccelerationStructuresKHR(1, &blas_info, &blas_ranges);
	Barrier(command, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
	        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, vk::AccessFlagBits2::eAccelerationStructureReadKHR);
	tlas_info.dstAccelerationStructure  = g.tlas;
	tlas_info.scratchData.deviceAddress = scratch0 + blas_scratch;
	vk::AccelerationStructureBuildRangeInfoKHR        tlas_range {};
	tlas_range.primitiveCount                                   = 1;
	const vk::AccelerationStructureBuildRangeInfoKHR* tlas_ranges = &tlas_range;
	command.buildAccelerationStructuresKHR(1, &tlas_info, &tlas_ranges);
	// The build -> the trace (compute) and the next frame's capture stores (vertex shaders) after its reads.
	Barrier(command, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
	        vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eAccelerationStructureReadKHR,
	        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllCommands,
	        vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderWrite);
	if (g.pool != nullptr) command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, slot * 4 + 1);
	g.trace_slot  = slot;
	g.have_blas   = true;
	g.last_counts = std::move(counts);
	g.sum_tris += tris;
	++g.n_frames;
	if (do_refit) ++g.refits;
	static bool said = false;
	if (!said) {
		said = true;
		std::printf("RTSHADOW HW: first acceleration structure: %zu draws, %.0f triangles (%u distinct light matrices)\n", list.size(),
		            tris, g.distinct_matrices);
		std::fflush(stdout);
	}
	return g.tlas;
}

void TraceBegin(vk::CommandBuffer command) {
	if (g.pool != nullptr) command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, g.trace_slot * 4 + 2);
}
void TraceEnd(vk::CommandBuffer command) {
	if (g.pool == nullptr) return;
	command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, g.pool, g.trace_slot * 4 + 3);
	g.pending[g.trace_slot] = true;
}
void NoteShadowed(double percent) { g.shadowed = percent; }

} // namespace Libs::Graphics::HwRt
