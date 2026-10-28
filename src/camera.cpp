#include "camera.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inj::camera {
namespace {

struct UboEntry { size_t offset, stride; uint32_t count; };
struct BufferMem { VkDeviceMemory memory = VK_NULL_HANDLE; VkDeviceSize offset = 0, size = 0; };

std::mutex g_mutex;
std::unordered_map<VkDescriptorUpdateTemplate, std::vector<UboEntry>> g_templates;
std::unordered_map<VkBuffer, BufferMem>       g_buffers;
std::unordered_map<VkDeviceMemory, uint8_t*>  g_mapped;  // the pointer to the allocation's offset 0

thread_local bool t_scene = false;

constexpr uint32_t kBudget      = 384;      // uniform ranges scanned per frame
constexpr size_t   kMaxRange    = 64 * 1024;
uint32_t g_scanned = 0;                     // this frame (guarded by g_mutex)
std::unordered_set<const void*> g_seen;     // ranges scanned this frame
std::unordered_set<float*>      g_written;  // jitter terms we wrote this frame

struct Cand {
	char  form;     // 'T' / 'R'
	float m[16];
	float* jx_ptr; float* jy_ptr;  // into mapped memory (null: a vkCmdUpdateBuffer copy)
	uint32_t count;
};
std::vector<Cand> g_cands;  // this frame's distinct matrices

// What the last frames settled on.
State    g_state;
uint32_t g_missing = 0;
struct JitterSample { float x, y; };
JitterSample g_hist[32] {};
uint32_t g_hist_n = 0, g_hist_at = 0;
int64_t  g_key_a = 0, g_key_b = 0;
char     g_key_form = '-';
float    g_render_aspect = 16.0f / 9.0f;
VkExtent2D g_render {};

bool     g_inject = false;
float    g_inject_ratio = 1.0f;
uint32_t g_inject_index = 0;
float    g_inject_px[2] {0, 0};  // this frame's offset (pixels)
uint64_t g_frames = 0;
uint32_t g_logged_found = 0;

bool Near0(float v, float eps) { return std::fabs(v) <= eps; }

float Halton(uint32_t index, uint32_t base) {
	float f = 1.0f, r = 0.0f;
	while (index > 0) {
		f /= static_cast<float>(base);
		r += f * static_cast<float>(index % base);
		index /= base;
	}
	return r;
}

// A perspective matrix at p (16 floats)? Fills the candidate's form.
bool LooksLikeProjection(const float* m, char& form) {
	const float a = m[0], b = m[5];
	if (!(std::fabs(a) > 0.05f && std::fabs(a) < 50.0f && std::fabs(b) > 0.05f && std::fabs(b) < 50.0f)) return false;
	if (!Near0(m[1], 1e-4f) || !Near0(m[4], 1e-4f) || !Near0(m[3], 1e-4f) || !Near0(m[7], 1e-4f)) return false;
	if (!Near0(m[12], 1e-4f) || !Near0(m[13], 1e-4f) || !Near0(m[15], 1e-4f)) return false;
	const float aspect = std::fabs(b / a);
	if (std::fabs(aspect - g_render_aspect) > 0.03f * g_render_aspect) return false;
	// form T: m14 = +-1, m11 = d, jitter at m2 / m6, m8 = m9 = 0.
	if (std::fabs(std::fabs(m[14]) - 1.0f) < 1e-4f && Near0(m[8], 1e-4f) && Near0(m[9], 1e-4f) && std::fabs(m[2]) < 0.05f &&
	    std::fabs(m[6]) < 0.05f && !Near0(m[11], 1e-9f)) {
		form = 'T';
		return true;
	}
	// form R: m11 = +-1, m14 = d, jitter at m8 / m9, m2 = m6 = 0.
	if (std::fabs(std::fabs(m[11]) - 1.0f) < 1e-4f && Near0(m[2], 1e-4f) && Near0(m[6], 1e-4f) && std::fabs(m[8]) < 0.05f &&
	    std::fabs(m[9]) < 0.05f && !Near0(m[14], 1e-9f)) {
		form = 'R';
		return true;
	}
	return false;
}

// Row-vector convention (D3D: clip = v * P): P[r][c] at out[4r + c].
void ToRowVector(const Cand& c, float* out) {
	if (c.form == 'T') {  // memory holds columns of P
		for (int r = 0; r < 4; ++r)
			for (int k = 0; k < 4; ++k) out[r * 4 + k] = c.m[k * 4 + r];
	} else {
		std::memcpy(out, c.m, sizeof(c.m));
	}
}

void Scan(uint8_t* base, size_t size, bool writable) {
	if (size < 64) return;
	size = std::min(size, kMaxRange);
	for (size_t at = 0; at + 64 <= size; at += 16) {
		auto* m = reinterpret_cast<float*>(base + at);
		char form = '-';
		if (!LooksLikeProjection(m, form)) continue;
		float* jx = form == 'T' ? &m[2] : &m[8];
		float* jy = form == 'T' ? &m[6] : &m[9];
		const float w = form == 'T' ? m[14] : m[11];
		// Halton injection: once per range per frame, only into a matrix that carries no jitter of its own.
		bool injected = false;
		if (g_inject && writable && Near0(*jx, 1e-7f) && Near0(*jy, 1e-7f) && g_render.width > 0 && g_written.count(jx) == 0) {
			*jx = (2.0f * g_inject_px[0] / static_cast<float>(g_render.width)) / w;
			*jy = (-2.0f * g_inject_px[1] / static_cast<float>(g_render.height)) / w;
			g_written.insert(jx);
			injected = true;
		}
		Cand c {form, {}, writable ? jx : nullptr, writable ? jy : nullptr, 1};
		std::memcpy(c.m, m, sizeof(c.m));
		if (injected) {  // the detection sees the matrix as the game wrote it
			if (form == 'T') c.m[2] = c.m[6] = 0;
			else c.m[8] = c.m[9] = 0;
		}
		bool merged = false;
		for (auto& e: g_cands) {
			if (e.form == c.form && std::memcmp(e.m, c.m, sizeof(c.m)) == 0) { ++e.count; merged = true; break; }
		}
		if (!merged && g_cands.size() < 64) g_cands.push_back(c);
		at += 48;  // (the next 48 bytes belong to this matrix)
	}
}

void ScanRange(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
	if (g_scanned >= kBudget) return;
	auto b = g_buffers.find(buffer);
	if (b == g_buffers.end()) return;
	auto m = g_mapped.find(b->second.memory);
	if (m == g_mapped.end() || m->second == nullptr) return;
	if (range == VK_WHOLE_SIZE) range = b->second.size > offset ? b->second.size - offset : 0;
	if (b->second.size != 0 && offset + range > b->second.size) return;
	uint8_t* p = m->second + b->second.offset + offset;
	if (!g_seen.insert(p).second) return;
	++g_scanned;
	Scan(p, static_cast<size_t>(range), true);
}

} // namespace

void OnCreateTemplate(VkDescriptorUpdateTemplate tmpl, const VkDescriptorUpdateTemplateCreateInfo& info) {
	if (info.templateType != VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET) return;
	std::vector<UboEntry> entries;
	for (uint32_t i = 0; i < info.descriptorUpdateEntryCount; ++i) {
		const auto& e = info.pDescriptorUpdateEntries[i];
		if (e.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || e.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
			entries.push_back({e.offset, e.stride, e.descriptorCount});
	}
	if (entries.empty()) return;
	std::lock_guard lock(g_mutex);
	g_templates[tmpl] = std::move(entries);
}
void OnDestroyTemplate(VkDescriptorUpdateTemplate tmpl) {
	std::lock_guard lock(g_mutex);
	g_templates.erase(tmpl);
}
void OnCreateBuffer(VkBuffer buffer, VkDeviceSize size) {
	std::lock_guard lock(g_mutex);
	g_buffers[buffer] = {VK_NULL_HANDLE, 0, size};
}
void OnBindBufferMemory(VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
	std::lock_guard lock(g_mutex);
	auto it = g_buffers.find(buffer);  // (uniform buffers only: OnCreateBuffer)
	if (it == g_buffers.end()) return;
	it->second.memory = memory;
	it->second.offset = offset;
}
void OnDestroyBuffer(VkBuffer buffer) {
	std::lock_guard lock(g_mutex);
	g_buffers.erase(buffer);
}
void OnMapMemory(VkDeviceMemory memory, VkDeviceSize offset, void* data) {
	if (data == nullptr) return;
	std::lock_guard lock(g_mutex);
	g_mapped[memory] = static_cast<uint8_t*>(data) - offset;
}
void OnUnmapMemory(VkDeviceMemory memory) {
	std::lock_guard lock(g_mutex);
	g_mapped.erase(memory);
}

void SetScenePass(bool scene) {
	t_scene = scene;
}

void OnUpdateWithTemplate(VkDescriptorUpdateTemplate tmpl, const void* data) {
	if (!t_scene || data == nullptr || Cfg().up_backend == 0) return;
	std::lock_guard lock(g_mutex);
	if (g_scanned >= kBudget) return;
	auto t = g_templates.find(tmpl);
	if (t == g_templates.end()) return;
	const auto* bytes = static_cast<const uint8_t*>(data);
	for (const auto& e: t->second) {
		for (uint32_t i = 0; i < e.count; ++i) {
			VkDescriptorBufferInfo info {};
			std::memcpy(&info, bytes + e.offset + i * e.stride, sizeof(info));
			if (info.buffer != VK_NULL_HANDLE) ScanRange(info.buffer, info.offset, info.range);
		}
	}
}

void OnUpdateDescriptorSets(uint32_t count, const VkWriteDescriptorSet* writes) {
	if (!t_scene || writes == nullptr || Cfg().up_backend == 0) return;
	std::lock_guard lock(g_mutex);
	for (uint32_t i = 0; i < count && g_scanned < kBudget; ++i) {
		const auto& w = writes[i];
		if ((w.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && w.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) ||
		    w.pBufferInfo == nullptr)
			continue;
		for (uint32_t j = 0; j < w.descriptorCount; ++j)
			if (w.pBufferInfo[j].buffer != VK_NULL_HANDLE) ScanRange(w.pBufferInfo[j].buffer, w.pBufferInfo[j].offset, w.pBufferInfo[j].range);
	}
}

void OnCmdUpdateBuffer(const void* data, VkDeviceSize size) {
	if (data == nullptr || size < 64 || Cfg().up_backend == 0) return;
	std::lock_guard lock(g_mutex);
	if (g_scanned >= kBudget) return;
	++g_scanned;
	// (A copy the GPU makes later: read-only for us, no injection.)
	std::vector<uint8_t> copy(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + std::min<size_t>(size, kMaxRange));
	Scan(copy.data(), copy.size(), false);
}

void EndFrame(VkExtent2D render) {
	std::lock_guard lock(g_mutex);
	++g_frames;
	if (render.width > 0 && render.height > 0) {
		g_render = render;
		g_render_aspect = static_cast<float>(render.width) / static_cast<float>(render.height);
	}
	g_state.scanned    = g_scanned;
	g_state.candidates = static_cast<uint32_t>(g_cands.size());
	// The camera: the matrix bound by the most draws (the main view), ties to one carrying a jitter.
	const Cand* best = nullptr;
	for (const auto& c: g_cands) {
		const float jx = c.form == 'T' ? c.m[2] : c.m[8], jy = c.form == 'T' ? c.m[6] : c.m[9];
		const bool jit = jx != 0 || jy != 0;
		if (best == nullptr || c.count > best->count ||
		    (c.count == best->count && jit && (best->form == 'T' ? best->m[2] : best->m[8]) == 0)) best = &c;
	}
	const bool injected_now = g_inject && !g_written.empty();
	if (best != nullptr) {
		g_missing = 0;
		const float w  = best->form == 'T' ? best->m[14] : best->m[11];
		const float jx = best->form == 'T' ? best->m[2] : best->m[8];
		const float jy = best->form == 'T' ? best->m[6] : best->m[9];
		// NDC offset = jitter term * w (w = +-1); pixels = ndc * size / 2, y flipped (FidelityFX's convention).
		const float px = jx * w * static_cast<float>(g_render.width) * 0.5f;
		const float py = -jy * w * static_cast<float>(g_render.height) * 0.5f;
		const int64_t ka = std::llround(std::fabs(best->m[0]) * 1e4), kb = std::llround(std::fabs(best->m[5]) * 1e4);
		if (ka != g_key_a || kb != g_key_b || best->form != g_key_form) {
			g_key_a = ka; g_key_b = kb; g_key_form = best->form;
			g_hist_n = g_hist_at = 0;  // a new camera (FOV change, another view): its own history
		}
		g_hist[g_hist_at] = {px, py};
		g_hist_at = (g_hist_at + 1) % 32;
		g_hist_n = std::min<uint32_t>(g_hist_n + 1, 32);
		std::vector<std::pair<int, int>> distinct;
		float maxabs = 0;
		for (uint32_t i = 0; i < g_hist_n; ++i) {
			const std::pair<int, int> k {static_cast<int>(std::lround(g_hist[i].x * 1000)), static_cast<int>(std::lround(g_hist[i].y * 1000))};
			if (std::find(distinct.begin(), distinct.end(), k) == distinct.end()) distinct.push_back(k);
			maxabs = std::max({maxabs, std::fabs(g_hist[i].x), std::fabs(g_hist[i].y)});
		}
		const bool nonzero = !(distinct.size() == 1 && distinct[0] == std::make_pair(0, 0));
		g_state.found = true;
		g_state.distinct = static_cast<uint32_t>(distinct.size());
		g_state.jitter_detected = distinct.size() >= 4 && nonzero && maxabs <= 1.0f;
		g_state.form = best->form;
		ToRowVector(*best, g_state.proj);
		// Camera: FOV from b; near / far from the depth terms (z_ndc = e*c + d / t, t the view distance, e = w).
		const float b = std::fabs(best->m[5]);
		g_state.fov = 2.0f * std::atan(1.0f / b);
		const float c = best->m[10], d = best->form == 'T' ? best->m[11] : best->m[14];
		const auto dist = [&](float target) { const float den = target - w * c; return den != 0 ? std::fabs(d / den) : INFINITY; };
		const float t0 = dist(0.0f), t1 = dist(1.0f);  // where depth reads 0 / 1
		g_state.reversed   = t1 < t0;
		g_state.near_plane = std::min(t0, t1);
		g_state.far_plane  = std::max(t0, t1);
		if (!std::isfinite(g_state.near_plane) || g_state.near_plane <= 0) g_state.near_plane = 0.1f;
		if (g_logged_found++ == 0 || g_frames % 1800 == 0)
			Log("[camera] projection found (form %c, %u draws, %u ranges scanned): fov %.1f deg, near %.3f far %s%.1f, %s depth, "
			    "jitter %s (%.3f, %.3f px, %u distinct values in %u frames)",
			    best->form, best->count, g_scanned, g_state.fov * 57.2958f, g_state.near_plane,
			    std::isfinite(g_state.far_plane) ? "" : "inf ", std::isfinite(g_state.far_plane) ? g_state.far_plane : 0.0f,
			    g_state.reversed ? "reversed" : "forward", g_state.jitter_detected ? "detected" : (injected_now ? "injected (Halton)" : "none"),
			    px, py, g_state.distinct, g_hist_n);
		g_state.jitter_px[0] = g_state.jitter_detected ? px : 0.0f;
		g_state.jitter_px[1] = g_state.jitter_detected ? py : 0.0f;
	} else if (++g_missing > 30) {
		if (g_state.found) Log("[camera] projection no longer found (%u ranges scanned)", g_scanned);
		g_state.found = g_state.jitter_detected = false;
		g_state.form = '-';
		g_state.jitter_px[0] = g_state.jitter_px[1] = 0;
		g_hist_n = g_hist_at = 0;
	}
	// The injected offset that was written this frame (what the upscaler must be told).
	g_state.injecting = injected_now;
	if (injected_now && !g_state.jitter_detected) {
		g_state.jitter_px[0] = g_inject_px[0];
		g_state.jitter_px[1] = g_inject_px[1];
	}
	// Next frame's Halton offset: phases = 8 * ratio^2 (AMD's recommendation).
	const uint32_t phases = std::max<uint32_t>(8, static_cast<uint32_t>(std::ceil(8.0f * g_inject_ratio * g_inject_ratio)));
	g_inject_index = (g_inject_index + 1) % phases;
	g_inject_px[0] = Halton(g_inject_index + 1, 2) - 0.5f;
	g_inject_px[1] = Halton(g_inject_index + 1, 3) - 0.5f;
	g_cands.clear();
	g_seen.clear();
	g_written.clear();
	g_scanned = 0;
}

State Current() {
	std::lock_guard lock(g_mutex);
	return g_state;
}

void SetInjection(bool on, float ratio) {
	std::lock_guard lock(g_mutex);
	// Only into a projection that has no jitter of its own (a game's TAA sequence is never overwritten).
	g_inject = on && !g_state.jitter_detected;
	g_inject_ratio = std::max(1.0f, ratio);
}

void DrawMenu() {
	const State s = Current();
	if (!s.found) {
		ImGui::TextDisabled("Camara: sin matriz de proyeccion (%u rangos UBO leidos)", s.scanned);
		return;
	}
	ImGui::Text("Camara: proyeccion forma %c, FOV %.1f, near %.3f, far %s", s.form, s.fov * 57.2958f, s.near_plane,
	            std::isfinite(s.far_plane) ? "finito" : "infinito");
	ImGui::Text("Jitter: %s (%.3f, %.3f px, %u valores distintos)",
	            s.jitter_detected ? "detectado" : s.injecting ? "inyectado (Halton)" : "ninguno", s.jitter_px[0], s.jitter_px[1], s.distinct);
}

void Shutdown() {
	std::lock_guard lock(g_mutex);
	g_templates.clear();
	g_buffers.clear();
	g_mapped.clear();
	g_cands.clear();
	g_seen.clear();
	g_written.clear();
	g_state = {};
	g_hist_n = g_hist_at = 0;
	g_scanned = 0;
}

} // namespace inj::camera
