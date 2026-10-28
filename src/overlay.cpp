#include "overlay.h"

#include "analyzer.h"
#include "framegen.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace inj::overlay {
namespace {

bool g_menu = false;
bool g_init = false, g_init_failed = false;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkFormat   g_format = VK_FORMAT_UNDEFINED;
VkExtent2D g_extent {};
uint32_t   g_image_count = 0;
VkRenderPass     g_render_pass = VK_NULL_HANDLE;
VkDescriptorPool g_pool = VK_NULL_HANDLE;
VkSampler        g_sampler = VK_NULL_HANDLE;
std::vector<VkImageView>   g_views;
std::vector<VkFramebuffer> g_framebuffers;
std::unordered_map<VkImageView, VkDescriptorSet> g_textures;
bool g_key_f1 = false, g_key_f2 = false;
std::chrono::steady_clock::time_point g_last {};

PFN_vkVoidFunction Loader(const char* name, void*) {
	return RealDeviceProc(name);
}

void DestroyTargets() {
	for (auto fb: g_framebuffers) R().vkDestroyFramebuffer(R().device, fb, nullptr);
	for (auto v: g_views) R().vkDestroyImageView(R().device, v, nullptr);
	g_framebuffers.clear();
	g_views.clear();
}

bool InitImGui() {
	if (g_init) return true;
	if (g_init_failed) return false;
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = nullptr;
	io.BackendPlatformName = "injector-polled";
	io.Fonts->AddFontDefault();
	if (!ImGui_ImplVulkan_LoadFunctions(&Loader, nullptr)) {
		Log("[overlay] ImGui Vulkan functions not loaded: overlay off");
		g_init_failed = true;
		return false;
	}
	VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64}};
	VkDescriptorPoolCreateInfo pi {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pi.maxSets = 64;
	pi.poolSizeCount = 1;
	pi.pPoolSizes = sizes;
	if (R().vkCreateDescriptorPool(R().device, &pi, nullptr, &g_pool) != VK_SUCCESS) { g_init_failed = true; return false; }
	VkSamplerCreateInfo si {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	si.magFilter = si.minFilter = VK_FILTER_NEAREST;
	si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	si.maxLod = 1.0f;
	R().vkCreateSampler(R().device, &si, nullptr, &g_sampler);
	ImGui_ImplVulkan_InitInfo init {};
	init.Instance       = R().instance;
	init.PhysicalDevice = R().physical;
	init.Device         = R().device;
	init.QueueFamily    = Q().game_family;
	init.Queue          = Q().game;
	init.DescriptorPool = g_pool;
	init.RenderPass     = g_render_pass;
	init.MinImageCount  = 2;
	init.ImageCount     = std::max(2u, g_image_count);
	init.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
	if (!ImGui_ImplVulkan_Init(&init)) { g_init_failed = true; Log("[overlay] ImGui_ImplVulkan_Init failed"); return false; }
	ImGui_ImplVulkan_CreateFontsTexture();
	g_init = true;
	Log("[overlay] ImGui ready (F1 menu, F2 frame generation on/off)");
	return true;
}

bool EnsureRenderPass(VkFormat format) {
	if (g_render_pass != VK_NULL_HANDLE && format == g_format) return true;
	if (g_render_pass != VK_NULL_HANDLE) {
		if (g_init) {  // the pipeline is tied to the render pass's format: start ImGui's backend again
			for (auto& [view, set]: g_textures) ImGui_ImplVulkan_RemoveTexture(set);
			g_textures.clear();
			ImGui_ImplVulkan_Shutdown();
			ImGui::DestroyContext();
			g_init = false;
		}
		R().vkDestroyRenderPass(R().device, g_render_pass, nullptr);
		g_render_pass = VK_NULL_HANDLE;
	}
	VkAttachmentDescription a {};
	a.format = format;
	a.samples = VK_SAMPLE_COUNT_1_BIT;
	a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	a.initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	a.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	VkAttachmentReference ref {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub {};
	sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	sub.colorAttachmentCount = 1;
	sub.pColorAttachments = &ref;
	VkSubpassDependency dep {};
	dep.srcSubpass = VK_SUBPASS_EXTERNAL;
	dep.dstSubpass = 0;
	dep.srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dep.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	VkRenderPassCreateInfo ci {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	ci.attachmentCount = 1;
	ci.pAttachments = &a;
	ci.subpassCount = 1;
	ci.pSubpasses = &sub;
	ci.dependencyCount = 1;
	ci.pDependencies = &dep;
	if (R().vkCreateRenderPass(R().device, &ci, nullptr, &g_render_pass) != VK_SUCCESS) {
		g_render_pass = VK_NULL_HANDLE;
		return false;
	}
	g_format = format;
	return true;
}

bool Down(int vk) {
	return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

void DrawUi() {
	auto& s = Cfg();
	const auto st = framegen::LastStats();
	if (s.osd) {
		char line[160];
		std::snprintf(line, sizeof line, "FG %s | real %.0f -> shown %.0f fps (x%.2f) | F1 menu", st.state, st.real_fps,
		              st.shown_fps, st.ratio);
		ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(4, 4), ImVec2(12 + 7.0f * std::strlen(line), 24), IM_COL32(0, 0, 0, 150));
		ImGui::GetForegroundDrawList()->AddText(ImVec2(8, 7), IM_COL32(255, 255, 255, 255), line);
	}
	if (!g_menu) return;
	ImGui::SetNextWindowPos(ImVec2(40, 40), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(520, 640), ImGuiCond_FirstUseEver);
	ImGui::Begin("Injector (F1)");
	ImGui::Text("Juego: %s.exe", ExeName().c_str());
	if (ImGui::CollapsingHeader("Frame generation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (!framegen::Available()) ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "No disponible (ver injector.log)");
		ImGui::Checkbox("FSR 3 frame generation x2 (F2)", &s.fg_enabled);
		ImGui::SliderInt("Tope VRR (fps, -1 auto, 0 off)", &s.fg_cap_fps, -1, 360);
		ImGui::Text("Estado: %s", st.state);
		ImGui::Text("Real %.1f fps -> mostrados %.1f fps (x%.2f), generados %u/s", st.real_fps, st.shown_fps, st.ratio, st.generated);
		ImGui::Text("Entradas: %s, tope %.0f fps", st.game_inputs ? "depth + MV del juego" : "neutras (solo optical flow)", st.cap_fps);
	}
	if (ImGui::CollapsingHeader("Analizar vectores", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("Analisis en vivo", &s.analyze);
		analyzer::DrawMenu();
	}
	if (ImGui::CollapsingHeader("Upscaler (fase 2)")) {
		ImGui::TextWrapped("FSR / XeSS / DLSS (Streamline) usaran los mismos datos del perfil. No implementado en fase 1.");
	}
	ImGui::Checkbox("OSD", &s.osd);
	ImGui::End();
}

} // namespace

void OnSwapchain(VkSwapchainKHR swapchain, const VkImage* images, uint32_t count, VkFormat format, VkExtent2D extent) {
	std::lock_guard lock(QueueLock());
	DestroyTargets();
	if (!EnsureRenderPass(format)) return;
	g_swapchain = swapchain;
	g_extent = extent;
	g_image_count = count;
	for (uint32_t i = 0; i < count; ++i) {
		VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		vi.image = images[i];
		vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
		vi.format = format;
		vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		VkImageView view = VK_NULL_HANDLE;
		R().vkCreateImageView(R().device, &vi, nullptr, &view);
		VkFramebufferCreateInfo fi {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
		fi.renderPass = g_render_pass;
		fi.attachmentCount = 1;
		fi.pAttachments = &view;
		fi.width = extent.width;
		fi.height = extent.height;
		fi.layers = 1;
		VkFramebuffer fb = VK_NULL_HANDLE;
		R().vkCreateFramebuffer(R().device, &fi, nullptr, &fb);
		g_views.push_back(view);
		g_framebuffers.push_back(fb);
	}
}

void OnSwapchainDestroyed(VkSwapchainKHR swapchain) {
	std::lock_guard lock(QueueLock());
	if (swapchain != g_swapchain) return;
	DestroyTargets();
	g_swapchain = VK_NULL_HANDLE;
}

void PollInput() {
	const bool f1 = Down(VK_F1), f2 = Down(VK_F2);
	if (f1 && !g_key_f1) {
		g_menu = !g_menu;
		Log("[overlay] menu %s", g_menu ? "open" : "closed");
		if (!g_menu) SaveSettings();
	}
	if (f2 && !g_key_f2) {
		Cfg().fg_enabled = !Cfg().fg_enabled;
		Log("[overlay] F2: frame generation %s", Cfg().fg_enabled ? "on" : "off");
		SaveSettings();
	}
	g_key_f1 = f1;
	g_key_f2 = f2;
}

bool WantsDraw() {
	return !g_init_failed && (g_menu || Cfg().osd);
}
bool MenuVisible() {
	return g_menu;
}

void Record(VkCommandBuffer cb, uint32_t image_index) {
	if (!WantsDraw() || image_index >= g_framebuffers.size() || g_framebuffers[image_index] == VK_NULL_HANDLE) return;
	if (!InitImGui()) return;
	ImGuiIO& io = ImGui::GetIO();
	const auto now = std::chrono::steady_clock::now();
	io.DeltaTime = g_last == decltype(g_last) {} ? 1.0f / 60 : std::max(1e-4f, std::chrono::duration<float>(now - g_last).count());
	g_last = now;
	io.DisplaySize = ImVec2(static_cast<float>(g_extent.width), static_cast<float>(g_extent.height));
	io.MouseDrawCursor = g_menu;
	if (HWND hwnd = GameWindow()) {
		POINT p {};
		RECT rc {};
		if (GetCursorPos(&p) && ScreenToClient(hwnd, &p) && GetClientRect(hwnd, &rc) && rc.right > 0 && rc.bottom > 0) {
			io.AddMousePosEvent(p.x * io.DisplaySize.x / rc.right, p.y * io.DisplaySize.y / rc.bottom);
		}
	}
	io.AddMouseButtonEvent(0, g_menu && Down(VK_LBUTTON));
	io.AddMouseButtonEvent(1, g_menu && Down(VK_RBUTTON));
	ImGui_ImplVulkan_NewFrame();
	ImGui::NewFrame();
	DrawUi();
	ImGui::Render();
	VkRenderPassBeginInfo rp {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
	rp.renderPass = g_render_pass;
	rp.framebuffer = g_framebuffers[image_index];
	rp.renderArea = {{0, 0}, g_extent};
	R().vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
	R().vkCmdEndRenderPass(cb);
}

void* Texture(VkImageView view) {
	if (!g_init || view == VK_NULL_HANDLE) return nullptr;
	auto it = g_textures.find(view);
	if (it != g_textures.end()) return reinterpret_cast<void*>(it->second);
	VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(g_sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	g_textures[view] = set;
	return reinterpret_cast<void*>(set);
}

void ForgetTexture(VkImageView view) {
	auto it = g_textures.find(view);
	if (it == g_textures.end()) return;
	if (g_init) ImGui_ImplVulkan_RemoveTexture(it->second);
	g_textures.erase(it);
}

} // namespace inj::overlay
