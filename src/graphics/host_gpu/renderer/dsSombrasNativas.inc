// (F5 pase 2, KYTY_NATIVO_PASE_SOMBRAS=3 / =1; local/ds-nativo-sombras.h, notas-sombras-nativas.md) Sombras en cascada
// NATIVAS de Demon's Souls. Incluido al final de renderCompute.cpp (fuera de su espacio de nombres).
//
// Por cada cubo (0xab57d0) de cada cascada que el gancho dejo en una ranura (g_nat_slots), en el hilo de la GPU:
//  - marca de INICIO: toma la ranura; en el primer draw del cubo (DsNatBeforeDraw) copia la capa emulada (antes de sus
//    draws) a la imagen propia A y obtiene de MALLA-NATIVA (Lookup) el VkBuffer de los vertices e indices de cada draw;
//  - marca de FIN (DsNatMark): =3: copia la capa emulada (despues) a B, dibuja los draws en A con el VS del juego en GLSL
//    (gpu_ds_sombras.vert) y el estado dinamico que el emulador resolvio para esos draws, y compara A y B con un compute
//    (gpu_ds_sombras_cmp.comp) hacia un bufer visible por la CPU, leido cuando la GPU termino ese tick (un fotograma
//    despues). =1, en una cascada que ya dio >= 99,9 %: los draws del juego no se hacen (DsNatBeforeDraw devuelve
//    true) y A se copia a la capa del juego.
// Nunca lee memoria del juego (solo la ranura) ni toma el candado de MALLA fuera de Lookup. Cualquier fallo: ese cubo se
// queda emulado (nada cambia) y se cuenta el motivo. Linea cada 10 s por cascada.

#include "local/ds-nativo-sombras.h"
#include "local/ds-sombras-ab.h" // (sombras-ab, KYTY_NATIVO_SOMBRAS_AB)
#include "graphics/host_gpu/renderer/depthRenderTarget.h" // (htile_fill_*: el valor del clear HTILE, como renderDraw.cpp)
#include "gpu_blit_shaders/gpu_ds_sombras_spv.h"
#include "gpu_blit_shaders/gpu_ds_sombras_skin_spv.h"
#include "gpu_blit_shaders/gpu_ds_sombras_cmp_spv.h"

namespace Libs::Graphics {
namespace {

constexpr uint32_t DsNatRing = 32, DsNatCamRing = 256, DsNatCamStride = 0x800, DsNatResultStride = 32;

struct DsNatImage {
	VkImage         image  = VK_NULL_HANDLE;
	VmaAllocation   memory = nullptr;
	vk::ImageView   view   = nullptr;
	vk::ImageLayout layout = vk::ImageLayout::eUndefined;
};

struct DsNatPending {
	uint64_t tick = 0;
	uint32_t pass = 0, drawn = 0;
	uint64_t pixels = 0;
	bool     busy = false, full = false; // (full: el cubo entero en nativo, sin entradas omitidas)
	bool     cleared = false; // (la capa tenia un clear HTILE pendiente al empezar: A = valor de clear)
	bool     skin = false;    // (con entradas de skinning dibujadas en nativo)
	// (DIAG cascada 0, 11-27) De que esta hecho el cubo: cubo del juego (0 Occluders, 1 Normal, 3 Fadeout), mezcla de VS
	// (bit0 estatico, bit1 VS de skinning con skinning, bit2 VS de skinning sin skinning), entradas por tipo, las de s10
	// heredado y hasta 4 ES distintos.
	uint32_t bucket = 0, mix = 0, n_static = 0, n_skin = 0, n_skinvs_static = 0, n_s10 = 0;
	uint64_t es[4] {};
};

struct DsNatState {
	bool                    ready = false, broken = false;
	vk::Format              format = vk::Format::eUndefined;
	vk::Extent2D            extent {};
	DsNatImage              a, b;
	vk::DescriptorSetLayout draw_set = nullptr, cmp_set = nullptr;
	vk::PipelineLayout      draw_layout = nullptr, cmp_layout = nullptr;
	vk::Pipeline            draw_pipeline = nullptr, cmp_pipeline = nullptr;
	vk::Pipeline            draw_pipeline_skin = nullptr; // (gpu_ds_sombras_skin.vert: draws con NatDraw::vs == 1)
	vk::Sampler             sampler = nullptr;
	VkBuffer                cam = VK_NULL_HANDLE, res = VK_NULL_HANDLE;
	VmaAllocation           cam_mem = nullptr, res_mem = nullptr;
	uint8_t*                cam_map = nullptr;
	uint8_t*                res_map = nullptr;
	DsNatPending            pending[DsNatRing];
	uint32_t                next = 0;
	// El cubo en curso.
	enum Phase { Idle, WaitFirst, Active } phase = Idle;
	DsNativoSombras::NatSlot* slot = nullptr;
	uint32_t                  pass = 0, layer = 0;
	uint64_t                  addr = 0;
	bool                      replace = false, partial = false, cleared = false;
	uint64_t                  depth_seq = 0; // (m_ds_last_depth_seq al empezar el cubo)
	bool                      native_done = false; // (=1: el nativo ya dibujo este cubo; desde ahi se saltan draws)
	uint32_t                  native_drawn = 0;
	uint64_t                  replace_undone[4] {}; // (=1: cubos sin dibujo nativo: un solo draw del juego o fallo)
	uint32_t                  next_entry = 0, skipped_dcb = 0; // (=1 parcial: emparejado de draws)
	bool                      keep_first = false, first_seen = false; // (=1 con clear HTILE: el 1er draw del juego se hace)
	DsNativoSombras::NatDcbScan scan {};
	uint32_t                  cube_seen = 0;
	uint64_t                  part_ok_run[4] {};
	uint64_t                  skin_ok_run[4] {};
	uint64_t                  diag_es_static[4] {}, diag_es_skin[4] {}; // (diagnostico: VS vistos, estaticas / skinning)
	VkBuffer                  diag_buf = VK_NULL_HANDLE;
	VmaAllocation             diag_mem = nullptr;
	uint8_t*                  diag_map = nullptr;
	uint32_t                  diag_used = 0;
	uint64_t                  diag_tick[8] {};
	uint64_t                  diag_vbase[8] {};
	uint32_t                  diag_info[8][3] {};
	uint64_t                  inpass_yes[4] {}, inpass_no[4] {}; // (b: dibujos dentro del render pass del emulador / no) // (parciales seguidos con nativo_mas_cerca = 0, entre ventanas)
	uint64_t                  dcb_direct[4] {}, dcb_ib[4] {}, dcb_noend[4] {}, seen_all[4] {};
	uint64_t                  entries_seen[4] {}, game_draws[4] {}, game_skipped[4] {}, doubled[4] {}, quiet_clear[4] {}, quiet_order[4] {};
	uint32_t                  sample[4] {}; // (=3: se compara 1 de cada N cubos)
	uint32_t                  cam_next = 0;
	uint64_t                  cam_tick[DsNatCamRing] {};
	std::vector<std::pair<vk::Buffer, uint64_t>> vb, ib;
	std::vector<uint8_t> native; // (1: la dibuja el nativo; sus vertices de skinning se obtienen en la marca de fin)
	// (=1 + KYTY_NATIVO_SOMBRAS_CASCADA) Los cubos de la cascada en curso: [0] es el primero (s.slot al empezar). Sus
	// entradas nativas se dibujan todas de una vez, en el segundo draw del juego del primero; los demas solo saltan.
	struct GCube {
		DsNativoSombras::NatSlot*                    slot = nullptr;
		uint32_t                                     seq  = 0;
		std::vector<std::pair<vk::Buffer, uint64_t>> vb, ib;
		std::vector<uint8_t>                         native;
		bool                                         has_skin = false, released = false;
		uint32_t                                     drawn    = 0;
	};
	std::vector<GCube> group;
	uint32_t           group_cur = 0;
	bool               group_on = false, group_drawn = false, group_noskip = false;
	uint64_t           groups[4] {}, group_cubes[4] {}, group_abort[4] {}, group_clear[4] {};
	bool                 has_skin = false;
	// (SALTO) El cubo en curso se salto entero en ProcessPm4 (NOP sobre sus paquetes): ni un draw del juego traducido.
	bool     whole_skip = false;
	ImageId  last_native_id {}; // (la imagen en la que dibujo el ultimo nativo: si el emulador resuelve otra, se cuenta)
	uint64_t whole_cubes[4] {}, whole_no[4] {}, img_changed[4] {}, native_clears[4] {};
	const char* whole_why[4] {};
	// La imagen de profundidad de las cascadas (la misma para las 4; capa = cascada).
	ImageId  shadow_id {};
	uint64_t shadow_addr = 0;
	// El estado dinamico del emulador por cascada (el ultimo valido: =1 lo usa sin draws del juego).
	DsNativoSombras::NatDyn dyn[4] {};
	bool                    dyn_ok[4] {};
	bool                    logged_regs[4] {};
	// Ventana de 10 s.
	// (completos: cubos sin ninguna entrada omitida, comparacion exacta; parciales: las omitidas, p. ej. con skinning, las
	// dibuja solo el juego: un pixel donde el emulado queda MAS CERCA puede ser suyo; donde el nativo queda mas cerca, no.)
	uint64_t    eq[4] {}, total[4] {}, drawn[4] {}, events[4] {}, compared[4] {}, fails[4] {}, replaced[4] {};
	uint64_t    full_cubes[4] {}, full_eq[4] {}, full_total[4] {};
	uint64_t    nat_small[4] {}; uint32_t nat_max[4] {}; // (T36: nativo mas cerca por <= KYTY_NATIVO_SOMBRAS_TOL / mayor error de esos)
	uint64_t    clr_cubes[4] {}, clr_eq[4] {}, clr_total[4] {}, clr_native_near[4] {}; // (empezaron con clear HTILE)
	uint64_t    skin_cubes[4] {}, skin_eq[4] {}, skin_total[4] {}, skin_native_near[4] {}, skin_drawn[4] {}, skin_missing[4] {};
	uint64_t    sampled_out[4] {};
	uint64_t    part_cubes[4] {}, part_eq[4] {}, part_total[4] {}, part_native_near[4] {}, part_emu_near[4] {};
	uint32_t    max_err[4] {}, full_max_err[4] {}, part_native_max[4] {};
	const char* why[4] {};
	std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};
DsNatState g_ds_nat;

// (CASCADA) Suelta las ranuras de los cubos de la cascada aun no terminados (menos la del cubo en curso, s.slot, que la
// suelta quien llama) y deja de agrupar. Nunca deja draws saltados sin dibujo: solo se salta despues de dibujar todos.
void DsNatGroupRelease() {
	auto& s = g_ds_nat;
	for (auto& c: s.group)
		if (!c.released && c.slot != s.slot) {
			DsNativoSombras::NatRelease(c.slot);
			c.released = true;
		}
	s.group.clear();
	s.group_on    = false;
	s.group_drawn = false;
	s.group_cur   = 0;
}

void DsNatFail(const char* why) {
	auto& s = g_ds_nat;
	if (s.group_on) {
		if (s.pass < 4 && !s.group_drawn) ++s.group_abort[s.pass];
		DsNatGroupRelease();
	}
	if (s.pass < 4) {
		++s.fails[s.pass];
		s.why[s.pass] = why;
	}
	DsNativoSombras::NatRelease(s.slot);
	s.slot    = nullptr;
	s.phase   = DsNatState::Idle;
	s.replace = false;
	DsNativoSombras::g_nat_dyn_target.store(0xFFu, std::memory_order_relaxed);
}

bool DsNatCreateImage(GraphicContext& g, DsNatImage& out, vk::Format format, vk::Extent2D extent) {
	VkImageCreateInfo create {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	create.imageType     = VK_IMAGE_TYPE_2D;
	create.format        = static_cast<VkFormat>(format);
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.samples       = VK_SAMPLE_COUNT_1_BIT;
	create.tiling        = VK_IMAGE_TILING_OPTIMAL;
	create.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
	                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	create.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(g.allocator, &create, &allocate, &out.image, &out.memory, nullptr) != VK_SUCCESS) {
		out.image = VK_NULL_HANDLE;
		return false;
	}
	vk::ImageViewCreateInfo view {};
	view.image            = vk::Image(out.image);
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = format;
	view.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
	return g.device.createImageView(&view, nullptr, &out.view) == vk::Result::eSuccess;
}

bool DsNatCreateBuffer(GraphicContext& g, VkDeviceSize bytes, VkBufferUsageFlags usage, bool readback, VkBuffer& buffer,
                       VmaAllocation& memory, uint8_t*& mapped) {
	VkBufferCreateInfo create {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	create.size        = bytes;
	create.usage       = usage;
	create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO;
	allocate.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
	                                                              : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
	VmaAllocationInfo info {};
	if (vmaCreateBuffer(g.allocator, &create, &allocate, &buffer, &memory, &info) != VK_SUCCESS || info.pMappedData == nullptr)
		return false;
	mapped = static_cast<uint8_t*>(info.pMappedData);
	return true;
}

vk::ShaderModule DsNatModule(GraphicContext& g, const uint32_t* code, size_t words) {
	vk::ShaderModuleCreateInfo create {};
	create.codeSize         = words * sizeof(uint32_t);
	create.pCode            = code;
	vk::ShaderModule module = nullptr;
	return g.device.createShaderModule(&create, nullptr, &module) == vk::Result::eSuccess ? module : vk::ShaderModule {};
}

// Una vez: imagenes A y B (formato y tamano de la de la cache), anillos de camara y resultados, y los dos pipelines.
bool DsNatInit(GraphicContext& g, vk::Format format, vk::Extent2D extent) {
	auto& s = g_ds_nat;
	if (s.ready) return s.format == format && s.extent == extent;
	if (s.broken) return false;
	s.broken = true; // (si algo falla aqui, no se reintenta)
	s.format = format;
	s.extent = extent;
	if (!DsNatCreateImage(g, s.a, format, extent) || !DsNatCreateImage(g, s.b, format, extent)) return false;
	if (!DsNatCreateBuffer(g, VkDeviceSize {DsNatCamRing} * DsNatCamStride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false, s.cam,
	                       s.cam_mem, s.cam_map) ||
	    !DsNatCreateBuffer(g, VkDeviceSize {DsNatRing} * DsNatResultStride,
	                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, s.res, s.res_mem, s.res_map))
		return false;
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	if (g.device.createSampler(&sampler, nullptr, &s.sampler) != vk::Result::eSuccess) return false;
	// Dibujo: UBO de camara (empuje de descriptores) + 64 bytes por draw.
	{
		vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr};
		vk::DescriptorSetLayoutCreateInfo set {};
		set.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set.bindingCount = 1;
		set.pBindings    = &binding;
		if (g.device.createDescriptorSetLayout(&set, nullptr, &s.draw_set) != vk::Result::eSuccess) return false;
		const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eVertex, 0, 64};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &s.draw_set;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &push;
		if (g.device.createPipelineLayout(&layout, nullptr, &s.draw_layout) != vk::Result::eSuccess) return false;
		const vk::ShaderModule vs      = DsNatModule(g, GPU_DS_SOMBRAS_SPV, std::size(GPU_DS_SOMBRAS_SPV));
		const vk::ShaderModule vs_skin = DsNatModule(g, GPU_DS_SOMBRAS_SKIN_SPV, std::size(GPU_DS_SOMBRAS_SKIN_SPV));
		if (!vs || !vs_skin) {
			if (vs) g.device.destroyShaderModule(vs, nullptr);
			if (vs_skin) g.device.destroyShaderModule(vs_skin, nullptr);
			return false;
		}
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eVertex;
		stage.module = vs;
		stage.pName  = "main";
		const vk::VertexInputBindingDescription   vbind {0, 12, vk::VertexInputRate::eVertex};
		const vk::VertexInputAttributeDescription vattr {0, 0, vk::Format::eR32G32B32Sfloat, 0};
		vk::PipelineVertexInputStateCreateInfo    vertex_input {};
		vertex_input.vertexBindingDescriptionCount   = 1;
		vertex_input.pVertexBindingDescriptions      = &vbind;
		vertex_input.vertexAttributeDescriptionCount = 1;
		vertex_input.pVertexAttributeDescriptions    = &vattr;
		vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
		input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
		vk::PipelineViewportStateCreateInfo viewport {};
		viewport.viewportCount = 1;
		viewport.scissorCount  = 1;
		vk::PipelineRasterizationStateCreateInfo rasterization {};
		rasterization.polygonMode = vk::PolygonMode::eFill;
		rasterization.lineWidth   = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisample {};
		multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
		vk::PipelineDepthStencilStateCreateInfo depth {};
		vk::PipelineColorBlendStateCreateInfo   color_blend {};
		const std::array dynamic_states {vk::DynamicState::eViewport,         vk::DynamicState::eScissor,
		                                 vk::DynamicState::eDepthBias,        vk::DynamicState::eDepthBiasEnable,
		                                 vk::DynamicState::eDepthTestEnable,  vk::DynamicState::eDepthWriteEnable,
		                                 vk::DynamicState::eDepthCompareOp,   vk::DynamicState::eCullMode,
		                                 vk::DynamicState::eFrontFace};
		vk::PipelineDynamicStateCreateInfo dynamic {};
		dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic.pDynamicStates    = dynamic_states.data();
		vk::PipelineRenderingCreateInfo rendering {};
		rendering.depthAttachmentFormat = format;
		vk::GraphicsPipelineCreateInfo create {};
		create.pNext               = &rendering;
		create.stageCount          = 1;
		create.pStages             = &stage;
		create.pVertexInputState   = &vertex_input;
		create.pInputAssemblyState = &input_assembly;
		create.pViewportState      = &viewport;
		create.pRasterizationState = &rasterization;
		create.pMultisampleState   = &multisample;
		create.pDepthStencilState  = &depth;
		create.pColorBlendState    = &color_blend;
		create.pDynamicState       = &dynamic;
		create.layout              = s.draw_layout;
		const vk::Result made = g.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &s.draw_pipeline);
		// (SKIN) El mismo estado con el VS de skinning (mismo layout: UBO de camara + 64 bytes).
		stage.module               = vs_skin;
		const vk::Result made_skin = made == vk::Result::eSuccess
		                                 ? g.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &s.draw_pipeline_skin)
		                                 : made;
		g.device.destroyShaderModule(vs, nullptr);
		g.device.destroyShaderModule(vs_skin, nullptr);
		if (made != vk::Result::eSuccess || made_skin != vk::Result::eSuccess) return false;
	}
	// Comparacion: A, B (muestreadas) y el resultado.
	{
		std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
		bindings[0] = {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
		bindings[1] = {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
		bindings[2] = {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
		vk::DescriptorSetLayoutCreateInfo set {};
		set.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		set.bindingCount = static_cast<uint32_t>(bindings.size());
		set.pBindings    = bindings.data();
		if (g.device.createDescriptorSetLayout(&set, nullptr, &s.cmp_set) != vk::Result::eSuccess) return false;
		const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		vk::PipelineLayoutCreateInfo layout {};
		layout.setLayoutCount         = 1;
		layout.pSetLayouts            = &s.cmp_set;
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &push;
		if (g.device.createPipelineLayout(&layout, nullptr, &s.cmp_layout) != vk::Result::eSuccess) return false;
		const vk::ShaderModule cs = DsNatModule(g, GPU_DS_SOMBRAS_CMP_SPV, std::size(GPU_DS_SOMBRAS_CMP_SPV));
		if (!cs) return false;
		vk::ComputePipelineCreateInfo pipeline {};
		pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		pipeline.stage.module = cs;
		pipeline.stage.pName  = "main";
		pipeline.layout       = s.cmp_layout;
		const vk::Result made = g.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &s.cmp_pipeline);
		g.device.destroyShaderModule(cs, nullptr);
		if (made != vk::Result::eSuccess) return false;
	}
	// (diagnostico, KYTY_NATIVO_SOMBRAS_DIAG=1) 8 muestras de 64 bytes de vertices de skinning, leidas por la CPU.
	if (!DsNatCreateBuffer(g, 8 * 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, s.diag_buf, s.diag_mem, s.diag_map)) s.diag_buf = VK_NULL_HANDLE;
	s.broken = false;
	s.ready  = true;
	return true;
}

void DsNatBarrier(vk::CommandBuffer command, DsNatImage& image, vk::ImageLayout to, vk::PipelineStageFlags2 src_stage,
                  vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = src_stage;
	b.srcAccessMask       = src_access;
	b.dstStageMask        = dst_stage;
	b.dstAccessMask       = dst_access;
	b.oldLayout           = image.layout;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = vk::Image(image.image);
	b.subresourceRange    = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
	image.layout = to;
}

// La capa `layer` de la imagen de la cache <-> la imagen propia (to_game: propia -> juego). La de la cache vuelve a su
// estado previo con su propio Transit.
void DsNatCopyLayer(vk::CommandBuffer command, Image* image, uint32_t layer, DsNatImage& own, bool to_game) {
	constexpr auto  all     = vk::PipelineStageFlagBits2::eAllCommands;
	const auto      aspect  = vk::ImageAspectFlagBits::eDepth;
	const VulkanImageState previous = image->CurrentState();
	ImageSubresourceRange  range {};
	range.base_level  = 0;
	range.level_count = 1;
	range.base_layer  = layer;
	range.layer_count = 1;
	if (to_game) {
		image->Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, range, command);
		DsNatBarrier(command, own, vk::ImageLayout::eTransferSrcOptimal, all, vk::AccessFlagBits2::eMemoryWrite,
		             vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	} else {
		image->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, command);
		DsNatBarrier(command, own, vk::ImageLayout::eTransferDstOptimal, all, vk::AccessFlagBits2::eMemoryRead,
		             vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	}
	vk::ImageCopy region {};
	region.srcSubresource = vk::ImageSubresourceLayers {aspect, 0, to_game ? 0u : layer, 1};
	region.dstSubresource = vk::ImageSubresourceLayers {aspect, 0, to_game ? layer : 0u, 1};
	region.extent         = vk::Extent3D {g_ds_nat.extent.width, g_ds_nat.extent.height, 1};
	if (to_game)
		command.copyImage(vk::Image(own.image), vk::ImageLayout::eTransferSrcOptimal, image->backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region);
	else
		command.copyImage(image->backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(own.image),
		                  vk::ImageLayout::eTransferDstOptimal, 1, &region);
	if (previous.layout != vk::ImageLayout::eUndefined) image->Transit(previous.layout, previous.access_mask, range, command);
}

// (DIAG) [cascada][cubo][mezcla de VS] = {pixeles nativo mas cerca, cubos con error, cubos sin error}.
uint64_t g_ds_nat_err[4][4][8][3] {};
void DsNatErrReport() {
	struct Row { uint64_t px, bad, good; uint32_t c, b, m; };
	Row rows[128];
	uint32_t n = 0;
	for (uint32_t c = 0; c < 4; ++c)
		for (uint32_t b = 0; b < 4; ++b)
			for (uint32_t m = 0; m < 8; ++m)
				if (g_ds_nat_err[c][b][m][1] + g_ds_nat_err[c][b][m][2] != 0)
					rows[n++] = {g_ds_nat_err[c][b][m][0], g_ds_nat_err[c][b][m][1], g_ds_nat_err[c][b][m][2], c, b, m};
	std::sort(rows, rows + n, [](const Row& x, const Row& y) { return x.px > y.px; });
	for (uint32_t i = 0; i < n && i < 10; ++i)
		std::printf("[nativo-sombras] DIAG error top%u: cascada %u cubo %u mezcla %s%s%s: nativo_mas_cerca=%llu px en %llu cubos (sin error %llu)\n",
		            i + 1, rows[i].c, rows[i].b, (rows[i].m & 1u) ? "estatico " : "", (rows[i].m & 2u) ? "skin " : "",
		            (rows[i].m & 4u) ? "vs_skin_sin_skin" : "", static_cast<unsigned long long>(rows[i].px),
		            static_cast<unsigned long long>(rows[i].bad), static_cast<unsigned long long>(rows[i].good));
}
// Los resultados de las comparaciones cuyo tick ya termino la GPU; cada 10 s, la linea por cascada.
void DsNatPoll(CommandScheduler& scheduler, GraphicContext& g) {
	namespace H = DsNativoSombras;
	auto& s = g_ds_nat;
	if (s.ready) {
		for (uint32_t i = 0; i < DsNatRing; ++i) {
			auto& p = s.pending[i];
			if (!p.busy || !scheduler.IsFree(p.tick)) continue;
			vmaInvalidateAllocation(g.allocator, s.res_mem, VkDeviceSize {i} * DsNatResultStride, DsNatResultStride);
			uint32_t r[6];
			std::memcpy(r, s.res_map + uint64_t {i} * DsNatResultStride, sizeof(r));
			// r: iguales, error maximo (bits), pixeles con el nativo mas cerca, con el emulado mas cerca.
			const uint32_t c = p.pass;
			s.eq[c] += r[0];
			s.total[c] += p.pixels;
			s.drawn[c] += p.drawn;
			++s.compared[c];
			if (r[1] > s.max_err[c]) s.max_err[c] = r[1];
			s.nat_small[c] += r[4];
			if (r[5] > s.nat_max[c]) s.nat_max[c] = r[5];
			if (r[2] != 0 && H::NatDiag()) {
				// (DIAG) Cubos con pixeles donde el nativo queda MAS CERCA que el juego (= error del nativo): agregado por
				// cascada x cubo x mezcla de VS (linea cada 10 s, 10 mas grandes) y los 30 primeros cubos con su detalle.
				g_ds_nat_err[c & 3u][p.bucket & 3u][p.mix & 7u][0] += r[2];
				g_ds_nat_err[c & 3u][p.bucket & 3u][p.mix & 7u][1] += 1;
				static uint32_t shown = 0;
				if (shown < 30) {
					++shown;
					std::printf("[nativo-sombras] DIAG error cascada %u cubo %u: nativo_mas_cerca=%u px err_max=%g | entradas: estaticas=%u "
					            "skinning=%u vs_skin_sin_skin=%u s10_heredado=%u completo=%u | ES %llx %llx %llx %llx\n",
					            c, p.bucket, r[2], [&] { float f = 0; std::memcpy(&f, &r[1], 4); return double(f); }(), p.n_static, p.n_skin, p.n_skinvs_static, p.n_s10,
					            p.full ? 1u : 0u, static_cast<unsigned long long>(p.es[0]), static_cast<unsigned long long>(p.es[1]),
					            static_cast<unsigned long long>(p.es[2]), static_cast<unsigned long long>(p.es[3]));
				}
			}
			if (r[2] == 0 && H::NatDiag()) g_ds_nat_err[c & 3u][p.bucket & 3u][p.mix & 7u][2] += 1;
			if (p.skin) {
				++s.skin_cubes[c];
				s.skin_eq[c] += r[0];
				s.skin_total[c] += p.pixels;
				s.skin_native_near[c] += r[2];
			}
			if (p.cleared) {
				++s.clr_cubes[c];
				s.clr_eq[c] += r[0];
				s.clr_total[c] += p.pixels;
				s.clr_native_near[c] += r[2];
			}
			if (p.full) {
				++s.full_cubes[c];
				s.full_eq[c] += r[0];
				s.full_total[c] += p.pixels;
				if (r[1] > s.full_max_err[c]) s.full_max_err[c] = r[1];
			} else {
				++s.part_cubes[c];
				s.part_eq[c] += r[0];
				s.part_total[c] += p.pixels;
				s.part_native_near[c] += r[2];
				s.part_emu_near[c] += r[3];
			}
			p.busy = false;
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last_report < std::chrono::seconds(10)) return;
	s.last_report = now;
	if (H::NatDiag()) DsNatErrReport();
	const auto pc = [](uint64_t a, uint64_t b) { return b != 0 ? 100.0 * double(a) / double(b) : 0.0; };
	const auto fl = [](uint32_t bits) {
		float f = 0;
		std::memcpy(&f, &bits, sizeof(f));
		return double(f);
	};
	bool any = false;
	if (H::NatDiag() && s.ready) {
		static uint32_t printed = 0;
		for (; printed < s.diag_used && scheduler.IsFree(s.diag_tick[printed]); ++printed) {
			vmaInvalidateAllocation(g.allocator, s.diag_mem, VkDeviceSize {printed} * 64, 64);
			float f[16];
			std::memcpy(f, s.diag_map + uint64_t {printed} * 64, sizeof(f));
			std::printf("[nativo-sombras] DIAG skinning cascada %u V# base=%llx stride=%u registros=%u: primeros vertices %g %g %g | %g %g %g "
			            "| %g %g %g | %g %g %g\n",
			            s.diag_info[printed][2], static_cast<unsigned long long>(s.diag_vbase[printed]), s.diag_info[printed][0],
			            s.diag_info[printed][1], double(f[0]), double(f[1]), double(f[2]), double(f[3]), double(f[4]), double(f[5]),
			            double(f[6]), double(f[7]), double(f[8]), double(f[9]), double(f[10]), double(f[11]));
		}
		std::printf("[nativo-sombras] DIAG VS (ES) de las entradas estaticas: %llx %llx %llx %llx | con skinning: %llx %llx %llx %llx\n",
		            static_cast<unsigned long long>(s.diag_es_static[0]), static_cast<unsigned long long>(s.diag_es_static[1]),
		            static_cast<unsigned long long>(s.diag_es_static[2]), static_cast<unsigned long long>(s.diag_es_static[3]),
		            static_cast<unsigned long long>(s.diag_es_skin[0]), static_cast<unsigned long long>(s.diag_es_skin[1]),
		            static_cast<unsigned long long>(s.diag_es_skin[2]), static_cast<unsigned long long>(s.diag_es_skin[3]));
	}
	for (uint32_t c = 0; c < 4; ++c) {
		if (s.events[c] == 0 && s.fails[c] == 0) continue;
		any = true;
		// (=1 solo con cubos completos: >= 99,9 % de pixeles iguales en ellos)
		// (Validada: >= 99,9 % iguales en sus cubos completos, o, sin completos, >= 50 cubos parciales comparados con
		// "nativo mas cerca" = 0 (el nativo nunca pinta donde el juego no) y ninguno completo por debajo.)
		const bool full_ok = s.full_total[c] != 0 && pc(s.full_eq[c], s.full_total[c]) >= 99.9;
		// (Log 20261125-030831-604: la cascada 0 no tiene cubos completos y con =3 muestreado (1 de cada 16) solo compara
		// ~43 parciales por ventana: nunca llegaba a 50. Ahora se acumulan entre ventanas: >= 50 parciales seguidos con
		// "nativo mas cerca" = 0; un solo pixel nativo mas cerca reinicia la cuenta.)
		if (s.part_native_near[c] != 0) s.part_ok_run[c] = 0;
		else s.part_ok_run[c] += s.part_cubes[c];
		const bool part_ok = s.full_total[c] == 0 && s.part_ok_run[c] >= 50;
		if (full_ok || part_ok) H::g_nat_proven[c].store(true, std::memory_order_relaxed);
		// (SKIN, log 20261125-045715-006: la cascada 2 se valido sin comparar ningun cubo con skinning y reemplazaba esas
		// entradas.) Validacion propia: >= 20 cubos con skinning comparados seguidos, cada ventana >= 99,9 % iguales y
		// "nativo mas cerca" = 0; una ventana peor la retira. Sin ella no se reemplaza nada en la cascada (sigue comparando).
		if (s.skin_cubes[c] != 0) {
			const bool skin_ok = pc(s.skin_eq[c], s.skin_total[c]) >= 99.9 && s.skin_native_near[c] == 0;
			if (skin_ok) s.skin_ok_run[c] += s.skin_cubes[c];
			else {
				s.skin_ok_run[c] = 0;
				H::g_nat_skin_proven[c].store(false, std::memory_order_relaxed);
			}
			if (s.skin_ok_run[c] >= 20) H::g_nat_skin_proven[c].store(true, std::memory_order_relaxed);
		}
		std::printf("[nativo-sombras] =%u cascada %u: iguales %.3f%% err_max %g dibujados %llu | completos: cubos=%llu iguales=%.3f%% "
		            "err_max=%g | parciales: cubos=%llu iguales=%.3f%% nativo_mas_cerca=%.3f%% (error) emulado_mas_cerca=%.3f%% "
		            "(omitidas) | con clear HTILE al empezar: cubos=%llu iguales=%.3f%% nativo_mas_cerca=%.3f%% | cubos=%llu "
		            "comparados=%llu sin_comparar(1 de cada N)=%llu reemplazados=%llu fallos_gpu=%llu (ultimo: %s)%s\n",
		            H::Mode(), c, pc(s.eq[c], s.total[c]), fl(s.max_err[c]), static_cast<unsigned long long>(s.drawn[c]),
		            static_cast<unsigned long long>(s.full_cubes[c]), pc(s.full_eq[c], s.full_total[c]), fl(s.full_max_err[c]),
		            static_cast<unsigned long long>(s.part_cubes[c]), pc(s.part_eq[c], s.part_total[c]),
		            pc(s.part_native_near[c], s.part_total[c]), pc(s.part_emu_near[c], s.part_total[c]),
		            static_cast<unsigned long long>(s.clr_cubes[c]), pc(s.clr_eq[c], s.clr_total[c]),
		            pc(s.clr_native_near[c], s.clr_total[c]),
		            static_cast<unsigned long long>(s.events[c]), static_cast<unsigned long long>(s.compared[c]),
		            static_cast<unsigned long long>(s.sampled_out[c]),
		            static_cast<unsigned long long>(s.replaced[c]), static_cast<unsigned long long>(s.fails[c]),
		            s.why[c] != nullptr ? s.why[c] : "-",
		            H::g_nat_proven[c].load(std::memory_order_relaxed) ? " [validada: apta para =1]" : "");
		// (Rendimiento: de las entradas de sombra de esta cascada que llegaron a la GPU con datos, cuantos draws del juego
		// se saltaron (los hace el nativo), y cuantas entradas nativas se dibujaron dos veces.)
		std::printf("[nativo-sombras] =%u cascada %u: draws del juego saltados %.1f%% (%llu de %llu entradas con datos; en cubos "
		            "reemplazados %llu de %llu draws) | dibujados dos veces=%llu | cubos reemplazados con clear HTILE (1er draw "
		            "del juego conservado)=%llu | no reemplazados por test no min/max=%llu | DCB entre marcas: draws directos=%llu "
		            "INDIRECT_BUFFER=%llu sin marca de fin en el bufer=%llu; vistos por DsNatBeforeDraw=%llu | cubos =1 sin dibujo "
		            "nativo (un solo draw del juego o fallo antes de saltar)=%llu | dentro del render pass del emulador=%llu, "
		            "no posible=%llu | CASCADA: pases=%llu cubos=%llu no agrupados (ranura/camara/fallo)=%llu con clear a mitad=%llu\n",
		            H::Mode(), c, pc(s.game_skipped[c], s.entries_seen[c]), static_cast<unsigned long long>(s.game_skipped[c]),
		            static_cast<unsigned long long>(s.entries_seen[c]), static_cast<unsigned long long>(s.game_skipped[c]),
		            static_cast<unsigned long long>(s.game_draws[c]), static_cast<unsigned long long>(s.doubled[c]),
		            static_cast<unsigned long long>(s.quiet_clear[c]), static_cast<unsigned long long>(s.quiet_order[c]),
		            static_cast<unsigned long long>(s.dcb_direct[c]), static_cast<unsigned long long>(s.dcb_ib[c]),
		            static_cast<unsigned long long>(s.dcb_noend[c]), static_cast<unsigned long long>(s.seen_all[c]),
		            static_cast<unsigned long long>(s.replace_undone[c]), static_cast<unsigned long long>(s.inpass_yes[c]),
		            static_cast<unsigned long long>(s.inpass_no[c]), static_cast<unsigned long long>(s.groups[c]),
		            static_cast<unsigned long long>(s.group_cubes[c]), static_cast<unsigned long long>(s.group_abort[c]),
		            static_cast<unsigned long long>(s.group_clear[c]));
		std::printf("[nativo-sombras] =%u cascada %u: SALTO cubos saltados enteros en ProcessPm4=%llu no saltados=%llu (ultimo: %s) | "
			            "clears HTILE hechos en nativo=%llu | imagen resuelta distinta tras el nativo=%llu\n",
			            H::Mode(), c, static_cast<unsigned long long>(s.whole_cubes[c]), static_cast<unsigned long long>(s.whole_no[c]),
			            s.whole_why[c] != nullptr ? s.whole_why[c] : "-", static_cast<unsigned long long>(s.native_clears[c]),
			            static_cast<unsigned long long>(s.img_changed[c]));
		s.whole_cubes[c] = s.whole_no[c] = s.native_clears[c] = s.img_changed[c] = 0;
		s.whole_why[c]   = nullptr;
		s.replace_undone[c] = s.inpass_yes[c] = s.inpass_no[c] = 0;
		s.groups[c] = s.group_cubes[c] = s.group_abort[c] = s.group_clear[c] = 0;
		s.dcb_direct[c] = s.dcb_ib[c] = s.dcb_noend[c] = s.seen_all[c] = 0;
		s.game_draws[c] = s.game_skipped[c] = s.doubled[c] = s.quiet_clear[c] = s.quiet_order[c] = s.entries_seen[c] = 0;
		s.eq[c] = s.total[c] = s.drawn[c] = s.events[c] = s.compared[c] = s.fails[c] = s.replaced[c] = 0;
		s.full_cubes[c] = s.full_eq[c] = s.full_total[c] = 0;
		if (H::NatSkinOn())
			std::printf("[nativo-sombras] =%u cascada %u: SKIN cubos con skinning nativo comparados=%llu iguales=%.3f%% "
			            "nativo_mas_cerca=%.3f%% | entradas con skinning dibujadas=%llu sin bufer en la cache=%llu | validada con "
			            "skinning=%s (seguidos %llu)\n",
			            H::Mode(), c, static_cast<unsigned long long>(s.skin_cubes[c]), pc(s.skin_eq[c], s.skin_total[c]),
			            pc(s.skin_native_near[c], s.skin_total[c]), static_cast<unsigned long long>(s.skin_drawn[c]),
			            static_cast<unsigned long long>(s.skin_missing[c]), H::g_nat_skin_proven[c].load() ? "si" : "no",
			            static_cast<unsigned long long>(s.skin_ok_run[c]));
		{
			float f = 0;
			std::memcpy(&f, &s.nat_max[c], 4);
			std::printf("[nativo-sombras] =%u cascada %u: TOL nativo mas cerca dentro de tolerancia=%llu px (no cuentan como error), "
			            "mayor |nativo-juego| con el nativo mas cerca=%g (KYTY_NATIVO_SOMBRAS_TOL)\n",
			            H::Mode(), c, static_cast<unsigned long long>(s.nat_small[c]), double(f));
			s.nat_small[c] = 0;
			s.nat_max[c]   = 0;
		}
		s.skin_cubes[c] = s.skin_eq[c] = s.skin_total[c] = s.skin_native_near[c] = s.skin_drawn[c] = s.skin_missing[c] = 0;
		s.clr_cubes[c] = s.clr_eq[c] = s.clr_total[c] = s.clr_native_near[c] = s.sampled_out[c] = 0;
		s.part_cubes[c] = s.part_eq[c] = s.part_total[c] = s.part_native_near[c] = s.part_emu_near[c] = 0;
		s.max_err[c] = s.full_max_err[c] = 0;
		s.why[c]     = nullptr;
	}
	if (any) {
		char cubes[1024], entries[1024], info[512];
		H::NatFormatWhy(H::g_nat_cube_why, cubes, sizeof(cubes));
		H::NatFormatWhy(H::g_nat_entry_why, entries, sizeof(entries));
		H::NatFormatWhy(H::g_nat_info_why, info, sizeof(info));
		std::printf("[nativo-sombras] captura (acumulado): cubos capturados=%llu sin ninguna entrada omitida=%llu | cubos sin "
		            "captura por motivo: %s | entradas omitidas por motivo: %s | info: %s | draws directos del recorrido fuera del "
		            "hilo de la GPU normal (acumulado): pasada de estado=%llu especulacion=%llu otro hilo=%llu\n",
		            static_cast<unsigned long long>(H::g_nat_cubes.load()), static_cast<unsigned long long>(H::g_nat_cubes_full.load()),
		            cubes[0] != '\0' ? cubes : "-", entries[0] != '\0' ? entries : "-", info[0] != '\0' ? info : "-",
		            static_cast<unsigned long long>(H::g_nat_draws_elsewhere[0].load()),
		            static_cast<unsigned long long>(H::g_nat_draws_elsewhere[1].load()),
		            static_cast<unsigned long long>(H::g_nat_draws_elsewhere[2].load()));
	}
	std::fflush(stdout);
}

bool DsNatStart(RenderContext& context, CommandBuffer& buffer, int role, uint64_t addr, uint32_t layer);
void DsNatQuiet();

} // namespace

namespace {
// La capa `layer` de la imagen de las cascadas tiene un clear HTILE pendiente (IsMetaCleared toma el candado de la cache:
// la direccion del HTILE con el candado, la consulta sin el).
bool DsNatClearPending(RenderContext& context, uint32_t layer) {
	auto&    cache        = context.GetTextureCache();
	uint64_t meta_address = 0;
	bool     meta_htile   = false;
	{
		std::scoped_lock meta_lock {cache.DsNatLock()};
		if (const auto* im = cache.TryGetImage(g_ds_nat.shadow_id); im != nullptr) {
			meta_htile   = im->info.metadata.kind == ImageMetadataKind::Htile;
			meta_address = im->info.metadata.range.address;
		}
	}
	return meta_htile && cache.IsMetaCleared(meta_address, layer);
}

// (Clear nativo) Si la capa tiene un clear HTILE pendiente, lo que haria el emulador en su render pass (renderDraw.cpp:
// IsMetaCleared -> loadOp clear con DB_DEPTH_CLEAR o el valor uniforme del relleno; TouchMeta(.., false) lo consume).
struct DsNatClearInfo {
	bool     pending = false, load_clear = false;
	float    value = 0.0f;
	uint64_t meta  = 0;
};
DsNatClearInfo DsNatClearQuery(RenderContext& context, CommandBuffer& buffer, uint32_t layer) {
	auto&          cache = context.GetTextureCache();
	DsNatClearInfo r {};
	bool           htile = false, stencil_compressed = false;
	{
		std::scoped_lock meta_lock {cache.DsNatLock()};
		if (const auto* im = cache.TryGetImage(g_ds_nat.shadow_id); im != nullptr) {
			htile              = im->info.metadata.kind == ImageMetadataKind::Htile;
			r.meta             = im->info.metadata.range.address;
			stencil_compressed = im->info.metadata.stencil_compressed;
		}
	}
	uint32_t fill  = 0;
	bool     known = false;
	r.pending      = htile && cache.IsMetaCleared(r.meta, layer, &fill, &known);
	if (!r.pending) return r;
	const bool uniform = known && !htile_fill_clears_depth(fill) && htile_fill_depth_uniform(fill, stencil_compressed);
	r.load_clear       = !known || htile_fill_clears_depth(fill) || uniform;
	r.value            = uniform ? htile_fill_depth_value(fill, stencil_compressed) : buffer.GetRegisters().GetDepthClearValue();
	return r;
}

// Vertices e indices de cada draw de una ranura: MALLA-NATIVA (Lookup; su candado, tal como esta disenado). Devuelve
// las entradas dibujables; `empty`: las vacias. Una entrada sin VkBuffer o sin alinear se omite (el cubo pasa a parcial).
uint32_t DsNatLookups(BufferCache& buffers, const DsNativoSombras::NatSlot* slot, std::vector<std::pair<vk::Buffer, uint64_t>>& vb,
                      std::vector<std::pair<vk::Buffer, uint64_t>>& ib, std::vector<uint8_t>& native, bool& has_skin,
                      uint32_t& empty) {
	namespace H         = DsNativoSombras;
	const uint32_t count = slot->count;
	vb.resize(count);
	ib.resize(count);
	native.assign(count, 0);
	has_skin          = false;
	uint32_t drawable = 0;
	empty             = 0;
	using Got = decltype(buffers.ObtainMallaNativa(0, 0));
	struct Hit {
		uint64_t addr = 0, size = 0;
		Got      got {};
	};
	std::array<Hit, 16> hits {};
	uint32_t            hit_next = 0;
	const auto cached = [&](uint64_t addr, uint64_t size) {
		for (const auto& h: hits)
			if (h.addr == addr && h.size == size && h.addr != 0) return h.got;
		auto& h = hits[hit_next++ % hits.size()];
		h.addr  = addr;
		h.size  = size;
		h.got   = buffers.ObtainMallaNativa(addr, size);
		return h.got;
	};
	for (uint32_t k = 0; k < count; ++k) {
		const auto& dr = slot->draws[k];
		vb[k]        = {vk::Buffer {}, 0};
		ib[k]        = {vk::Buffer {}, 0};
		if (dr.skip == 2u) {
			++empty; // (vacia: tampoco la dibuja el juego)
			continue;
		}
		if (dr.skip != 0) continue;
		const uint64_t ibyte = (uint64_t {dr.istart} + dr.icount) * (dr.index32 != 0 ? 4u : 2u);
		if (dr.skin != 0) {
			// (Skinning: los indices son del CMesh (MALLA-NATIVA); los vertices, de la cache en la marca de fin.)
			const auto [ibuf, ioff] = cached(dr.ibase, ibyte);
			if (ibuf != nullptr && ioff % (dr.index32 != 0 ? 4u : 2u) != 0) {
				H::NatCount(H::g_nat_entry_why, "skinning: indices sin alinear");
				continue;
			}
			// (11-27) Sin VkBuffer de MALLA-NATIVA (KYTY_MALLA_NATIVA=0 o aun sin preparar): el indice se toma con el candado
			// de render de la cache de buferes del emulador (DsNatResolveCache), el mismo VkBuffer que usa el draw emulado.
			ib[k]     = {ibuf != nullptr ? ibuf->Handle() : vk::Buffer {}, ibuf != nullptr ? ioff : 0};
			native[k] = 1;
			has_skin  = true;
			++drawable;
			continue;
		}
		// (Cache por llamada: la misma malla se repite en el cubo; un Lookup (candado de MALLA) por bloque distinto.)
		const auto [vbuf, voff] = cached(dr.vbase, uint64_t {dr.vrecords} * dr.vstride);
		const auto [ibuf, ioff] = cached(dr.ibase, ibyte);
		if ((ibuf != nullptr && ioff % (dr.index32 != 0 ? 4u : 2u) != 0) || (vbuf != nullptr && voff % 4u != 0)) {
			H::NatCount(H::g_nat_entry_why, "indice o vertices sin alinear");
			continue;
		}
		// (11-27, log 20261127-033027-959: con KYTY_MALLA_NATIVA=0 ninguna entrada tenia VkBuffer -> 0 dibujados) Lo que
		// MALLA-NATIVA no tenga se resuelve luego, con el candado de render, en la cache de buferes (DsNatResolveCache).
		vb[k]     = {vbuf != nullptr ? vbuf->Handle() : vk::Buffer {}, vbuf != nullptr ? voff : 0};
		ib[k]     = {ibuf != nullptr ? ibuf->Handle() : vk::Buffer {}, ibuf != nullptr ? ioff : 0};
		native[k] = 1;
		++drawable;
	}
	return drawable;
}

// (=1, cascada validada sin skinning validado) Las entradas con skinning no las dibuja el nativo: native[k] = 0, su draw
// del juego se hace (DsNatBeforeDraw las empareja igual, en su orden). El resto del cubo sigue en nativo. Devuelve
// cuantas se quitaron.
uint32_t DsNatDropSkin(const DsNativoSombras::NatSlot* slot, std::vector<uint8_t>& native, bool& has_skin) {
	uint32_t n = 0;
	for (uint32_t k = 0; k < slot->count && k < native.size(); ++k)
		if (native[k] != 0 && slot->draws[k].skin != 0) {
			native[k] = 0;
			++n;
		}
	has_skin = false;
	return n;
}
// (s10 heredado, 11-25; ver NatDraw::s10_reg) Las entradas cuyo s10 no estaba en el bloque de la DCB: la GPU del
// emulador usa el valor que ya tiene el registro USER_DATA_GS (gs_user_sgpr), el mismo en todo el bloque (el gancho
// comprobo que nada lo escribe desde el inicio del bloque hasta esos draws). Se lee aqui (hilo de la GPU, estado del
// emulador, sin memoria del juego) y solo se dibuja en nativo si (s10 & 0xffff) == 0 (el gancho copio la fila 0).
// `first`: el cubo en cuyo primer draw se lee el registro (otro bloque -> no se sabe: al juego). Devuelve las quitadas.
uint32_t DsNatS10Gate(CommandBuffer& buffer, const DsNativoSombras::NatSlot* slot, const DsNativoSombras::NatSlot* first,
                      std::vector<uint8_t>& native, bool& has_skin) {
	namespace H = DsNativoSombras;
	const auto& user = buffer.GetShaders().GetVs().gs_user_sgpr;
	uint32_t    dropped = 0, used = 0, other_block = 0, nonzero = 0, bad_reg = 0;
	has_skin = false;
	for (uint32_t k = 0; k < slot->count && k < native.size(); ++k) {
		const auto& dr = slot->draws[k];
		if (native[k] == 0) continue;
		if (dr.s10_reg != 0) {
			const uint32_t id = dr.s10_reg - 0x8Cu;
			const char*    why = nullptr;
			if (id >= 32u) why = "x", ++bad_reg;
			else if (first != nullptr && (first->block != slot->block || slot->sh_full == 0)) why = "x", ++other_block;
			else if ((user.value[id] & 0xFFFFu) != 0) {
				why = "x";
				++nonzero;
				static std::atomic<uint32_t> shown {0};
				if (shown.fetch_add(1, std::memory_order_relaxed) < 8)
					std::printf("[nativo-sombras] s10 heredado != 0: s%u=%08x (reg 0x%x) cascada %u: entrada al juego\n", id + 8u,
					            user.value[id], dr.s10_reg, slot->pass);
			}
			if (why != nullptr) {
				native[k] = 0;
				++dropped;
				continue;
			}
			++used;
		}
		if (dr.skin != 0) has_skin = true;
	}
	if (used != 0) H::NatCountN(H::g_nat_info_why, "s10 heredado = 0 en el registro del emulador: dibujada en nativo (fila 0)", used);
	if (nonzero != 0) H::NatCountN(H::g_nat_entry_why, "skinning: s10 heredado != 0 (la dibuja el juego)", nonzero);
	if (other_block != 0) H::NatCountN(H::g_nat_entry_why, "skinning: s10 heredado de otro bloque de la DCB o bloque no entendido (la dibuja el juego)", other_block);
	if (bad_reg != 0) H::NatCountN(H::g_nat_entry_why, "skinning: s10 heredado fuera de USER_DATA_GS", bad_reg);
	return dropped;
}
bool DsNatSkinProven(uint32_t pass) {
	namespace H = DsNativoSombras;
	return !H::NatSkinOn() || pass > 3u || H::g_nat_skin_proven[pass].load(std::memory_order_relaxed);
}

// (El dibujo nativo de un cubo: =3, en A con la comparacion; =1, en la capa del juego.) Falso: fallo contado, nada
// dibujado. =1 lo llama ANTES de saltar ningun draw del juego (DsNatBeforeDraw, en el segundo draw del cubo): un fallo
// deja el cubo entero al juego (log 20261125-030831-604: zonas sin sombra; antes se dibujaba en la marca de fin, despues
// de haber saltado draws, y un fallo ahi los perdia).
bool DsNatWork(RenderContext& context, CommandBuffer& buffer, int role, uint32_t pass) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost(DsSombrasAb::CWork);
	auto& s = g_ds_nat;
	// (=1, clear nativo) Un clear HTILE pendiente en la capa: lo hace el propio render pass nativo (loadOp clear con el
	// valor que usaria el emulador) y se consume en la cache al terminar (TouchMeta), como renderDraw.cpp. Ya no se
	// conserva ningun draw del juego para aplicarlo.
	const DsNatClearInfo clr = s.replace ? DsNatClearQuery(context, buffer, s.layer) : DsNatClearInfo {};
	if ((H::g_nat_dyn_mask.load(std::memory_order_acquire) & (1u << pass)) != 0) {
		s.dyn[pass]    = H::g_nat_dyn_pass[pass]; // (el ultimo resuelto para esta cascada: este cubo o uno anterior)
		s.dyn_ok[pass] = true;
	}
	if (!s.dyn_ok[pass]) return DsNatFail("sin estado dinamico de la cascada aun"), false;
	{
		const auto& regs = buffer.GetRegisters();
		if (regs.GetDepthRenderTarget().z_write_base_addr != s.addr || regs.GetDepthDepthView().slice_start != s.layer)
			return DsNatFail("la imagen o la capa del cubo no son las de la marca"), false;
	}
	if (buffer.IsInvalid() || role != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed))
		return DsNatFail("no se puede ahora (especulacion, captura o buffer)"), false;
	// (CASCADA) Los demas cubos de la cascada se dibujan en este mismo pase: sus VkBuffer de malla, ahora (antes de los
	// candados, como al empezar un cubo).
	const bool group_draw = s.replace && s.group_on && !s.group_drawn;
	if (group_draw) {
		auto& buffers = context.GetBufferCache();
		for (size_t c = 1; c < s.group.size(); ++c) {
			auto&    g = s.group[c];
			uint32_t e = 0;
			(void)DsNatLookups(buffers, g.slot, g.vb, g.ib, g.native, g.has_skin, e);
			(void)DsNatS10Gate(buffer, g.slot, s.slot, g.native, g.has_skin);
			if (g.has_skin && !DsNatSkinProven(pass)) (void)DsNatDropSkin(g.slot, g.native, g.has_skin); // (las dibuja el juego)
			if (g.has_skin) s.has_skin = true;
		}
	}
	auto&          scheduler = context.GetCommandScheduler();
	auto&          graphics  = context.GetGraphics();
	// Los anillos: resultados (solo al comparar) y camara (cada cubo; libre cuando la GPU termino su tick).
	const uint32_t ring = s.next % DsNatRing;
	if (!s.replace && s.pending[ring].busy) return DsNatFail("anillo de resultados lleno"), false;
	const uint32_t cam_slot = s.cam_next % DsNatCamRing;
	if (s.cam_tick[cam_slot] != 0 && !scheduler.IsFree(s.cam_tick[cam_slot])) return DsNatFail("anillo de camaras lleno"), false;
	scheduler.PopPendingOperations();
	Envio1::NoteDispatch();
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard              lock(context.GetMutex());
	envio1_lock.reset();
	// (Cache por llamada (la cascada entera con CASCADA): un ObtainBuffer por bufer de skinning distinto, no por entrada.)
	struct SkinHit {
		uint64_t   addr = 0, size = 0;
		vk::Buffer buf {};
		uint64_t   off = 0;
		bool       ok  = false;
	};
	std::vector<SkinHit> skin_hits;
	const auto obtain_skin = [&](BufferCache& buffers, uint64_t addr, uint64_t size, vk::Buffer& out, uint64_t& off) {
		for (const auto& h: skin_hits)
			if (h.addr == addr && h.size == size) return out = h.buf, off = h.off, h.ok;
		const auto [vbuf, voff] = buffers.ObtainBuffer(addr, size, false);
		SkinHit h {addr, size, vbuf != nullptr ? vbuf->Handle() : vk::Buffer {}, voff, vbuf != nullptr && voff % 4u == 0};
		skin_hits.push_back(h);
		return out = h.buf, off = h.off, h.ok;
	};
	// (SKIN) Los vertices de skinning, ahora (con el candado de render, antes del de la cache): la cache de buferes
	// del emulador, como el draw emulado de esa entrada (ObtainBuffer: su VkBuffer y sus esperas entre colas). Aqui
	// y no al empezar el cubo: los draws emulados del cubo pueden rehacer ese bufer entre medias.
	// (11-27) Vertices (sin skinning) e indices que MALLA-NATIVA no dio: la cache de buferes del emulador (ObtainBuffer,
	// como el draw emulado de la entrada), aqui con el candado de render. Lo que no se obtenga lo dibuja el juego.
	{
		auto&      buffers  = context.GetBufferCache();
		const auto resolve  = [&](const DsNativoSombras::NatSlot* sl, std::vector<std::pair<vk::Buffer, uint64_t>>& vbs,
                                 std::vector<std::pair<vk::Buffer, uint64_t>>& ibs, std::vector<uint8_t>& nat) {
			for (uint32_t k = 0; k < sl->count && k < nat.size(); ++k) {
				if (nat[k] == 0) continue;
				const auto& dr = sl->draws[k];
				const uint32_t isz = dr.index32 != 0 ? 4u : 2u;
				if (!ibs[k].first) {
					vk::Buffer b {};
					uint64_t   o = 0;
					if (!obtain_skin(buffers, dr.ibase, (uint64_t {dr.istart} + dr.icount) * isz, b, o) || o % isz != 0) {
						nat[k] = 0;
						vbs[k] = {vk::Buffer {}, 0};
						H::NatCount(H::g_nat_entry_why, "sin VkBuffer de indices en la cache (la dibuja el juego)");
						continue;
					}
					ibs[k] = {b, o};
					H::NatCount(H::g_nat_info_why, "indices de la cache de buferes (sin MALLA-NATIVA)");
				}
				if (dr.skin == 0 && !vbs[k].first) {
					vk::Buffer b {};
					uint64_t   o = 0;
					if (!obtain_skin(buffers, dr.vbase, uint64_t {dr.vrecords} * dr.vstride, b, o)) {
						nat[k] = 0;
						vbs[k] = {vk::Buffer {}, 0};
						H::NatCount(H::g_nat_entry_why, "sin VkBuffer de vertices en la cache (la dibuja el juego)");
						continue;
					}
					vbs[k] = {b, o};
					H::NatCount(H::g_nat_info_why, "vertices de la cache de buferes (sin MALLA-NATIVA)");
				}
			}
		};
		resolve(s.slot, s.vb, s.ib, s.native);
		for (size_t c = 1; group_draw && c < s.group.size(); ++c) resolve(s.group[c].slot, s.group[c].vb, s.group[c].ib, s.group[c].native);
	}
	if (s.has_skin) {
		auto& buffers = context.GetBufferCache();
		for (uint32_t k = 0; k < s.slot->count; ++k) {
			const auto& dr = s.slot->draws[k];
			if (s.native[k] == 0 || dr.skin == 0) continue;
			vk::Buffer vbuf {};
			uint64_t   voff = 0;
			if (!obtain_skin(buffers, dr.vbase, uint64_t {dr.vrecords} * dr.vstride, vbuf, voff)) {
				++s.skin_missing[pass];
				s.native[k] = 0; // (sin dibujo nativo: su draw del juego no se saltara; =1 dibuja antes de saltar)
				continue;
			}
			s.vb[k] = {vbuf, voff};
			++s.skin_drawn[pass];
		}
		for (size_t c = 1; group_draw && c < s.group.size(); ++c) {
			auto& g = s.group[c];
			for (uint32_t k = 0; k < g.slot->count; ++k) {
				const auto& dr = g.slot->draws[k];
				if (g.native[k] == 0 || dr.skin == 0) continue;
				vk::Buffer vbuf {};
				uint64_t   voff = 0;
				if (!obtain_skin(buffers, dr.vbase, uint64_t {dr.vrecords} * dr.vstride, vbuf, voff)) {
					++s.skin_missing[pass];
					g.native[k] = 0; // (sin dibujo nativo: su draw del juego no se saltara)
					continue;
				}
				g.vb[k] = {vbuf, voff};
				++s.skin_drawn[pass];
			}
		}
	}
	auto&                                 cache = context.GetTextureCache();
	std::unique_lock<TrackingSpinLock>    cache_lock {cache.DsNatLock()};
	auto*                                 image = cache.TryGetImage(s.shadow_id);
	if (image == nullptr || image->backing.image == nullptr || image->backing.sparse != nullptr || !image->Overlaps(s.addr, 1) ||
	    s.layer >= image->backing.layers)
		return DsNatFail("la imagen de la cascada ya no esta"), false;
	const auto& d = s.dyn[pass];
	// (b, KYTY_NATIVO_SOMBRAS_INPASS=1, apagado por defecto) =1: si el render pass abierto del emulador es exactamente esta
	// capa (sin color, sin stencil, una capa, misma vista que la de la cache para esa capa), el nativo se dibuja dentro, sin
	// cerrarlo ni reabrirlo ni transiciones. Con skinning no (su barrera no puede ir dentro de un render pass).
	bool          inpass      = false;
	vk::ImageView inpass_view = nullptr;
	if (s.replace && H::NatInPass() && !s.has_skin && !clr.pending && buffer.IsRendering()) {
		ImageViewInfo view_info {};
		view_info.format      = image->backing.format;
		view_info.type        = vk::ImageViewType::e2D;
		view_info.aspect      = vk::ImageAspectFlagBits::eDepth;
		view_info.base_layer  = s.layer;
		view_info.layer_count = 1;
		view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		inpass_view           = image->FindView(view_info);
		const auto& rs        = buffer.CurrentRenderState();
		inpass = inpass_view && rs.num_color_attachments == 0 && rs.num_layers == 1 && rs.depth_stencil_attachment.has_depth &&
		         !rs.depth_stencil_attachment.has_stencil && rs.depth_stencil_attachment.image_view == inpass_view;
		++(inpass ? s.inpass_yes[pass] : s.inpass_no[pass]);
	}
	if (!inpass) buffer.EndRendering();
	const vk::CommandBuffer command = buffer.Handle();
	buffer.InvalidateGraphicsState();
	if (!s.replace) {
		DsNatCopyLayer(command, image, s.layer, s.b, false); // (la capa emulada, despues de los draws del juego)
		DsNatBarrier(command, s.b, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
		             vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eComputeShader,
		             vk::AccessFlagBits2::eShaderRead);
	}
	// La camara de la cascada al anillo (memoria propia, visible por la CPU).
	std::memcpy(s.cam_map + uint64_t {cam_slot} * DsNatCamStride, s.slot->view, H::NatViewBytes);
	vmaFlushAllocation(graphics.allocator, s.cam_mem, VkDeviceSize {cam_slot} * DsNatCamStride, H::NatViewBytes);
	++s.cam_next;
	s.cam_tick[cam_slot] = scheduler.CurrentTick();
	// El destino: A (=3, con la capa de antes de los draws) o, =1, la capa del juego misma (sin copias).
	vk::ImageView         target = s.a.view;
	VulkanImageState      game_previous {};
	ImageSubresourceRange game_range {};
	if (s.replace && inpass) {
		target = inpass_view;
	} else if (s.replace) {
		ImageViewInfo view_info {};
		view_info.format      = image->backing.format;
		view_info.type        = vk::ImageViewType::e2D;
		view_info.aspect      = vk::ImageAspectFlagBits::eDepth;
		view_info.base_layer  = s.layer;
		view_info.layer_count = 1;
		view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		target                = image->FindView(view_info);
		if (!target) return DsNatFail("sin vista de la capa del juego"), false;
		game_previous          = image->CurrentState();
		game_range.base_level  = 0;
		game_range.level_count = 1;
		game_range.base_layer  = s.layer;
		game_range.layer_count = 1;
		image->Transit(vk::ImageLayout::eDepthAttachmentOptimal,
		               vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		               game_range, command);
	}
	if (s.has_skin) {
		// (La salida del compute de skinning (o su subida) antes de leerla como vertices.)
		vk::MemoryBarrier2 skin {};
		skin.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		skin.srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eMemoryWrite;
		skin.dstStageMask  = vk::PipelineStageFlagBits2::eVertexAttributeInput;
		skin.dstAccessMask = vk::AccessFlagBits2::eVertexAttributeRead;
		vk::DependencyInfo skin_dependency {};
		skin_dependency.memoryBarrierCount = 1;
		skin_dependency.pMemoryBarriers    = &skin;
		command.pipelineBarrier2(skin_dependency);
		// (diagnostico) Los primeros 64 bytes de los vertices de una entrada con skinning, tal como los va a leer el nativo.
		if (H::NatDiag() && s.diag_buf != VK_NULL_HANDLE && s.diag_used < 8) {
			for (uint32_t k = 0; k < s.slot->count; ++k) {
				const auto& dr = s.slot->draws[k];
				if (dr.skin == 0 || s.native[k] == 0 || !s.vb[k].first) continue;
				vk::MemoryBarrier2 to_copy {};
				to_copy.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
				to_copy.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
				to_copy.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				to_copy.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
				vk::DependencyInfo dep {};
				dep.memoryBarrierCount = 1;
				dep.pMemoryBarriers    = &to_copy;
				command.pipelineBarrier2(dep);
				const vk::BufferCopy region {s.vb[k].second, VkDeviceSize {s.diag_used} * 64, 64};
				command.copyBuffer(s.vb[k].first, vk::Buffer(s.diag_buf), 1, &region);
				vk::MemoryBarrier2 to_host {};
				to_host.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				to_host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
				to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
				to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
				dep.pMemoryBarriers   = &to_host;
				command.pipelineBarrier2(dep);
				s.diag_tick[s.diag_used]    = scheduler.CurrentTick();
				s.diag_vbase[s.diag_used]   = dr.vbase;
				s.diag_info[s.diag_used][0] = dr.vstride;
				s.diag_info[s.diag_used][1] = dr.vrecords;
				s.diag_info[s.diag_used][2] = pass;
				++s.diag_used;
				break;
			}
		}
	}
	vk::RenderingAttachmentInfo depth_attachment {};
	depth_attachment.imageView   = target;
	depth_attachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
	depth_attachment.loadOp      = clr.pending && clr.load_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth_attachment.clearValue.depthStencil = vk::ClearDepthStencilValue {clr.value, 0};
	depth_attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent = s.extent;
	rendering.layerCount        = 1;
	rendering.pDepthAttachment  = &depth_attachment;
	if (!inpass) command.beginRendering(&rendering);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, s.draw_pipeline);
	const vk::DescriptorBufferInfo camera {vk::Buffer(s.cam), VkDeviceSize {cam_slot} * DsNatCamStride, H::NatViewBytes};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eUniformBuffer;
	write.pBufferInfo     = &camera;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, s.draw_layout, 0, 1, &write);
	const vk::Viewport viewport {d.vp[0], d.vp[1], d.vp[2], d.vp[3], d.vp[4], d.vp[5]};
	const vk::Rect2D   scissor {{d.sc[0], d.sc[1]}, {static_cast<uint32_t>(d.sc[2]), static_cast<uint32_t>(d.sc[3])}};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.setDepthTestEnable(d.depth_test);
	command.setDepthWriteEnable(d.depth_write);
	command.setDepthCompareOp(static_cast<vk::CompareOp>(d.depth_compare));
	command.setDepthBiasEnable(d.bias_enable);
	command.setDepthBias(d.bias_constant, d.bias_clamp, d.bias_slope);
	// Cara y recorte de caras: de los registros, como el emulador (face 1 = horario).
	const auto& mc = buffer.GetRegisters().GetModeControl();
	vk::CullModeFlags cull {};
	if (mc.cull_front) cull |= vk::CullModeFlagBits::eFront;
	if (mc.cull_back) cull |= vk::CullModeFlagBits::eBack;
	command.setCullMode(cull);
	command.setFrontFace(mc.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);
	const uint32_t count = s.slot->count;
	uint32_t drawn = 0;
	uint32_t bound_vs = 0; // (0: draw_pipeline, ya enlazado; 1: draw_pipeline_skin. El descriptor empujado sigue valido:
	                       // mismo layout)
	const auto bind_vs = [&](uint32_t vs) {
		if (vs == bound_vs) return;
		bound_vs = vs;
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, vs == 1 ? s.draw_pipeline_skin : s.draw_pipeline);
	};
	for (uint32_t k = 0; k < count; ++k) {
		const auto& dr = s.slot->draws[k];
		if (!s.vb[k].first || !s.ib[k].first || s.native[k] == 0) continue; // (omitida: la dibuja solo el juego)
		++drawn;
		bind_vs(dr.vs);
		command.bindVertexBuffers(0, 1, &s.vb[k].first, &s.vb[k].second);
		command.bindIndexBuffer(s.ib[k].first, s.ib[k].second, dr.index32 != 0 ? vk::IndexType::eUint32 : vk::IndexType::eUint16);
		float push[16];
		std::memcpy(push, dr.obj, sizeof(dr.obj));
		std::memcpy(push + 12, dr.t3, sizeof(dr.t3));
		command.pushConstants(s.draw_layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(push), push);
		command.drawIndexed(dr.icount, 1, dr.istart, 0, 0);
	}
	// (CASCADA) Los demas cubos de la cascada, en el mismo pase (misma camara: la de la cascada; se comprobo al agrupar).
	uint32_t group_extra = 0;
	for (size_t c = 1; group_draw && c < s.group.size(); ++c) {
		auto& g = s.group[c];
		for (uint32_t k = 0; k < g.slot->count; ++k) {
			const auto& dr = g.slot->draws[k];
			if (g.native[k] == 0 || !g.vb[k].first || !g.ib[k].first) {
				if (k < g.native.size()) g.native[k] = 0; // (no dibujada: su draw del juego se hace)
				continue;
			}
			++g.drawn;
			bind_vs(dr.vs);
			command.bindVertexBuffers(0, 1, &g.vb[k].first, &g.vb[k].second);
			command.bindIndexBuffer(g.ib[k].first, g.ib[k].second, dr.index32 != 0 ? vk::IndexType::eUint32 : vk::IndexType::eUint16);
			float push[16];
			std::memcpy(push, dr.obj, sizeof(dr.obj));
			std::memcpy(push + 12, dr.t3, sizeof(dr.t3));
			command.pushConstants(s.draw_layout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(push), push);
			command.drawIndexed(dr.icount, 1, dr.istart, 0, 0);
		}
		group_extra += g.drawn;
	}
	if (!inpass) command.endRendering();
	if (!s.logged_regs[pass]) {
		s.logged_regs[pass] = true;
		std::printf("[nativo-sombras] =%u cascada %u: primer dibujo nativo: %u draws, capa %u de %llx, viewport (%g,%g %gx%g z %g..%g) "
		            "prof test=%u escritura=%u comparacion=%u sesgo=%u (%g, clamp %g, pendiente %g) recorte caras=%u/%u cara=%u "
		            "ps=%llx\n",
		            H::Mode(), pass, count, s.layer, static_cast<unsigned long long>(s.addr), double(d.vp[0]), double(d.vp[1]),
		            double(d.vp[2]), double(d.vp[3]), double(d.vp[4]), double(d.vp[5]), d.depth_test, d.depth_write,
		            d.depth_compare, d.bias_enable, double(d.bias_constant), double(d.bias_clamp), double(d.bias_slope),
		            mc.cull_front ? 1u : 0u, mc.cull_back ? 1u : 0u, mc.face ? 1u : 0u,
		            static_cast<unsigned long long>(buffer.GetShaders().GetPs().ps_regs.data_addr));
		std::fflush(stdout);
	}
	if (s.replace) {
		// =1: ya esta en la capa del juego; vuelve a su estado previo.
		if (!inpass && game_previous.layout != vk::ImageLayout::eUndefined)
			image->Transit(game_previous.layout, game_previous.access_mask, game_range, command);
		++s.replaced[pass];
		s.drawn[pass] += drawn + group_extra;
		s.native_drawn   = drawn; // (los dibujados dos veces se cuentan en la marca de fin, con los saltados ya sabidos)
		s.last_native_id = s.shadow_id;
		if (clr.pending) {
			// (El clear ya esta en el render pass nativo: la cache deja de tenerlo pendiente, como tras el del emulador.)
			cache_lock.unlock();
			if (!cache.TouchMeta(clr.meta, s.layer, false)) H::NatCount(H::g_nat_info_why, "clear nativo: TouchMeta fallo");
			++s.native_clears[pass];
		}
	} else {
		// =3: comparar A (nativa) con B (emulada).
		DsNatBarrier(command, s.a, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eLateFragmentTests,
		             vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eComputeShader,
		             vk::AccessFlagBits2::eShaderRead);
		const VkDeviceSize at = VkDeviceSize {ring} * DsNatResultStride;
		command.fillBuffer(vk::Buffer(s.res), at, DsNatResultStride, 0);
		vk::MemoryBarrier2 fill {};
		fill.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		fill.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		fill.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
		fill.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &fill;
		command.pipelineBarrier2(dependency);
		const vk::DescriptorImageInfo  native {s.sampler, s.a.view, vk::ImageLayout::eShaderReadOnlyOptimal};
		const vk::DescriptorImageInfo  emulated {s.sampler, s.b.view, vk::ImageLayout::eShaderReadOnlyOptimal};
		const vk::DescriptorBufferInfo result {vk::Buffer(s.res), at, DsNatResultStride};
		std::array<vk::WriteDescriptorSet, 3> writes {};
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		writes[0].pImageInfo      = &native;
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		writes[1].pImageInfo      = &emulated;
		writes[2].dstBinding      = 2;
		writes[2].descriptorCount = 1;
		writes[2].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[2].pBufferInfo     = &result;
		// (less: el test de profundidad del juego guarda el menor; "mas cerca" = menor. Si no, el mayor.)
		const vk::CompareOp op = static_cast<vk::CompareOp>(d.depth_compare);
		static const float tol = [] {
			const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_TOL");
			const float v    = text != nullptr ? std::strtof(text, nullptr) : 0.0f;
			return v > 0.0f && v < 0.01f ? v : 0.0f;
		}();
		uint32_t tol_bits = 0;
		std::memcpy(&tol_bits, &tol, 4);
		const uint32_t size[4] {s.extent.width, s.extent.height,
		                        op == vk::CompareOp::eLess || op == vk::CompareOp::eLessOrEqual ? 1u : 0u, tol_bits};
		command.bindPipeline(vk::PipelineBindPoint::eCompute, s.cmp_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, s.cmp_layout, 0, static_cast<uint32_t>(writes.size()),
		                             writes.data());
		command.pushConstants(s.cmp_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(size), size);
		command.dispatch((s.extent.width + 15u) / 16u, (s.extent.height + 15u) / 16u, 1);
		vk::MemoryBarrier2 host {};
		host.srcStageMask               = vk::PipelineStageFlagBits2::eComputeShader;
		host.srcAccessMask              = vk::AccessFlagBits2::eShaderWrite;
		host.dstStageMask               = vk::PipelineStageFlagBits2::eHost;
		host.dstAccessMask              = vk::AccessFlagBits2::eHostRead;
		dependency.pMemoryBarriers      = &host;
		command.pipelineBarrier2(dependency);
		auto& p  = s.pending[ring];
		p.tick   = scheduler.CurrentTick();
		p.pass   = pass;
		p.drawn  = drawn;
		p.full   = !s.partial;
		p.cleared = s.cleared;
		p.skin    = s.has_skin;
		p.bucket  = s.slot->bucket;
		p.mix = p.n_static = p.n_skin = p.n_skinvs_static = p.n_s10 = 0;
		for (auto& e: p.es) e = 0;
		for (uint32_t k = 0; k < s.slot->count && k < s.native.size(); ++k) {
			const auto& dr = s.slot->draws[k];
			if (s.native[k] == 0) continue;
			if (dr.vs == 0) ++p.n_static, p.mix |= 1u;
			else if (dr.skin != 0) ++p.n_skin, p.mix |= 2u;
			else ++p.n_skinvs_static, p.mix |= 4u;
			if (dr.s10_reg != 0) ++p.n_s10;
			for (auto& e: p.es) {
				if (e == dr.es) break;
				if (e == 0) { e = dr.es; break; }
			}
		}
		p.pixels = uint64_t {s.extent.width} * s.extent.height;
		p.busy   = true;
		++s.next;
	}
	return true;
}
} // namespace

namespace {
// (Log 20261125-092605-970: SALTO 0/0 y reemplazados=0 en las 4 cascadas, todas "validada: apta para =1". Con
// KYTY_NATIVO_SOMBRAS_SKIN=1 el reemplazo exigia ademas la validacion con skinning, que nunca llega: ninguna entrada con
// skinning se captura ("USER_DATA s10 no esta en la DCB" 1,4 M, "otro VS" 0,6 M; SKIN comparados=0). Esas entradas ya
// van omitidas (las dibuja el juego); la validacion con skinning solo hace falta si el cubo lleva alguna entrada con
// skinning que dibujaria el nativo.)
bool DsNatSlotSkin(const DsNativoSombras::NatSlot* slot) {
	if (slot == nullptr) return false;
	for (uint32_t k = 0; k < slot->count; ++k)
		if (slot->draws[k].skip == 0 && slot->draws[k].skin != 0) return true;
	return false;
}
bool DsNatSkinReady(const DsNativoSombras::NatSlot* slot, uint32_t pass) {
	namespace H = DsNativoSombras;
	return !H::NatSkinOn() || pass > 3u || H::g_nat_skin_proven[pass].load(std::memory_order_relaxed) || !DsNatSlotSkin(slot);
}
// (SALTO, =1) Todas las entradas no vacias del cubo las dibujo el nativo: se escribe la NOP sobre sus paquetes (el dword
// que reservo el gancho, aun relleno) y ProcessPm4, al volver de la marca, salta el cubo entero. Falso: se queda draw a
// draw (las entradas nativas se saltan una a una en DsNatBeforeDraw, las demas las dibuja el juego).
bool DsNatFlip(const DsNativoSombras::NatSlot* slot, const std::vector<uint8_t>& native, const char** why) {
	namespace H = DsNativoSombras;
	if (slot->skip_word == 0 || slot->skip_dw == 0 || slot->skip_dw > H::NatSkipMaxDw) return *why = "sin estado final del gancho", false;
	for (uint32_t k = 0; k < slot->count; ++k)
		if (slot->draws[k].skip != 2u && (k >= native.size() || native[k] == 0)) return *why = "entradas sin dibujo nativo", false;
	auto* word = reinterpret_cast<volatile uint32_t*>(slot->skip_word);
	if (*word != H::NatSkipFiller) return *why = "el dword reservado no es el relleno", false;
	*word = H::NatSkipNop(slot->skip_dw + 1u);
	return true;
}
} // namespace

void RenderExecutor::DsNatMark(CommandBuffer& buffer, uint32_t kind_pass, uint32_t seq) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost(DsSombrasAb::CMarca);
	auto&          s    = g_ds_nat;
	const uint32_t kind = kind_pass & 0xFFu, pass = (kind_pass >> 8u) & 0xFFu;
	DsNatPoll(m_context.GetCommandScheduler(), m_context.GetGraphics());
	// (El emulador resolvio otra imagen en la direccion de la cascada que la del ultimo dibujo nativo: se cuenta.)
	if (kind == H::NatBegin && pass < 4 && s.last_native_id && m_ds_last_depth_addr == s.shadow_addr &&
	    m_ds_last_depth_id != s.last_native_id) {
		++s.img_changed[pass];
		s.last_native_id = {};
	}
	const auto count_whole = [&](uint32_t drawn, bool ok, const char* why) {
		if (ok) {
			++s.whole_cubes[pass];
			s.whole_skip  = true;
			s.skipped_dcb = drawn;
			s.game_skipped[pass] += drawn;
		} else {
			++s.whole_no[pass];
			s.whole_why[pass] = why;
		}
	};
	if (kind == H::NatBegin) {
		if (s.phase != DsNatState::Idle) DsNatFail("cubo anterior sin marca de fin");
		if (pass > 3u) return;
		s.pass = pass;
		++s.events[pass];
		s.scan      = H::g_nat_scan; // (diagnostico: lo que hay en la DCB hasta la marca de fin)
		s.cube_seen = 0;
		s.dcb_direct[pass] += s.scan.direct;
		s.dcb_ib[pass] += s.scan.ib;
		if (H::NatDiag() && !s.scan.end_found) ++s.dcb_noend[pass]; // (el recuento de la DCB, solo con KYTY_NATIVO_SOMBRAS_DIAG=1)
		// (CASCADA) El cubo siguiente de la cascada en curso: su ranura ya es nuestra.
		DsNativoSombras::NatSlot* grouped = nullptr;
		if (s.group_on) {
			const uint32_t next = s.group_cur + 1;
			if (next < s.group.size() && s.group[next].seq == seq && s.group[next].slot->pass == pass) {
				grouped     = s.group[next].slot;
				s.group_cur = next;
			} else {
				DsNatGroupRelease(); // (otro orden: se deja de agrupar; lo ya dibujado solo se repite, no falta)
			}
		}
		if (grouped != nullptr && s.group_drawn) {
			// Continuacion: su dibujo nativo ya esta hecho (con el del primer cubo). Desde su primer draw se saltan los de
			// sus entradas nativas, salvo que la capa tenga un clear HTILE pendiente (entonces, y para el resto de la
			// cascada, nada se salta: el clear borra igual lo nativo y lo emulado anterior).
			auto& g       = s.group[s.group_cur];
			s.slot        = grouped;
			s.entries_seen[pass] += s.slot->count;
			if (!s.group_noskip && DsNatClearPending(m_context, s.layer)) {
				s.group_noskip = true;
				++s.group_clear[pass];
			}
			s.native        = g.native;
			if (s.group_noskip) s.native.assign(s.native.size(), 0); // (nada se salta en este cubo)
			s.phase         = DsNatState::Active;
			s.replace       = true; // (su dibujo ya esta hecho: la marca de fin no dibuja ni compara)
			s.native_done   = true;
			s.native_drawn  = g.drawn;
			s.first_seen    = true; // (sin conservar el primero: el clear, si lo habia, era del primer cubo)
			s.next_entry    = 0;
			s.skipped_dcb   = 0;
			s.whole_skip    = false;
			if (H::Mode() == 1 && !s.group_noskip) {
				const char* why = nullptr;
				const bool  ok  = DsNatFlip(s.slot, s.native, &why);
				count_whole(g.drawn, ok, why);
			}
			H::g_nat_dyn_target.store(pass, std::memory_order_relaxed);
			return;
		}
		if (grouped != nullptr) {
			// Agrupados pero sin dibujo hecho (el primer cubo no llego a dibujar): se deja de agrupar y este cubo va solo.
			for (auto& c: s.group)
				if (c.slot == grouped) c.released = true; // (su ranura la usa este cubo)
			DsNatGroupRelease();
			s.slot = grouped;
		} else {
			s.slot = H::NatTake(seq);
		}
		if (s.slot != nullptr && s.slot->pass == pass) s.entries_seen[pass] += s.slot->count;
		if (s.slot == nullptr) return DsNatFail("sin datos del gancho (captura fallida: ver motivos)");
		if (s.slot->pass != pass || s.slot->count == 0) return DsNatFail("ranura de otro cubo");
		// (11-27, log 20261127-045004-895: solo la cascada 0 (la cercana) da nativo_mas_cerca 0,048 %, err_max 0,14.) El cubo 3
		// es kRS_Depth_Fadeout (0x25b3a10; 0xa13e60/0xa12de0 ponen ctx[0x27] = 0xc solo para el): objetos que se desvanecen
		// con descarte en el PS, que el nativo (sin PS) dibuja enteros. Ese cubo lo dibuja el juego.
		// KYTY_NATIVO_SOMBRAS_FADEOUT=1: dibujarlo tambien en nativo (como antes).
		static const bool fade_on = [] {
			const char* t = std::getenv("KYTY_NATIVO_SOMBRAS_FADEOUT");
			return t != nullptr && t[0] == '1' && t[1] == '\0';
		}();
		if (s.slot->bucket == 3u && !fade_on) return DsNatFail("cubo Fadeout (descarte en el PS: lo dibuja el juego)");
		s.depth_seq = m_ds_last_depth_seq; // (=1: el dibujo exige una resolucion de profundidad posterior, de este cubo)
		// (CASCADA) Primer cubo de una cascada validada: se toman ya las ranuras de los demas cubos de la cascada (las marcas
		// que siguen en este bufer). Si alguna no esta lista, es de otra cascada o tiene otra camara, no se agrupa (este cubo
		// sigue solo, como sin CASCADA).
		if (H::Mode() == 1 && H::NatCascada() && !s.group_on && H::g_nat_proven[pass].load(std::memory_order_relaxed) &&
		    s.dyn_ok[pass] && H::g_nat_group.count >= 2 && H::g_nat_group.seq[0] == seq && H::g_nat_group.pass == pass) {
			s.group.clear();
			s.group.push_back({});
			s.group[0].slot = s.slot;
			s.group[0].seq  = seq;
			bool ok         = true;
			for (uint32_t i = 1; i < H::g_nat_group.count && ok; ++i) {
				auto* t = H::NatTake(H::g_nat_group.seq[i]);
				if (t == nullptr) { ok = false; break; }
				s.group.push_back({});
				s.group.back().slot = t;
				s.group.back().seq  = H::g_nat_group.seq[i];
				// (Sin skinning validado, sus entradas con skinning las dibuja el juego: DsNatDropSkin.)
				ok = t->pass == pass && t->count != 0 && std::memcmp(t->view, s.slot->view, H::NatViewBytes) == 0;
			}
			if (ok) {
				s.group_on     = true;
				s.group_cur    = 0;
				s.group_drawn  = false;
				s.group_noskip = false;
			} else {
				++s.group_abort[pass];
				for (size_t i = 1; i < s.group.size(); ++i) H::NatRelease(s.group[i].slot);
				s.group.clear();
			}
		}
		s.phase = DsNatState::WaitFirst;
		H::g_nat_dyn_target.store(pass, std::memory_order_relaxed);
		// (Imagen ya conocida: la capa = el pase (DB_DEPTH_VIEW medido: slice = pase). Se copia aqui, antes de cualquier
		// draw del cubo; asi no depende de que el primer draw pase por CommandProcessor::DrawIndex.)
		s.whole_skip = false;
		if (s.shadow_id && s.shadow_addr != 0 && !buffer.IsInvalid()) (void)DsNatStart(m_context, buffer, Role(), s.shadow_addr, pass);
		// (SALTO, =1) Cubo de una cascada validada: el dibujo nativo aqui mismo, en la marca de inicio, y si cubre todas sus
		// entradas, el cubo entero fuera de ProcessPm4. Solo si los registros ya son los de esta cascada (imagen y capa: si
		// el cubo los trae dentro, como el primero de cada cascada, va por su primer draw) y la ultima profundidad resuelta
		// es la de la cascada.
		if (H::Mode() == 1 && s.phase == DsNatState::Active && s.replace && !s.native_done) {
			const auto& regs = buffer.GetRegisters();
			const char* why  = nullptr;
			if (regs.GetDepthRenderTarget().z_write_base_addr != s.addr || regs.GetDepthDepthView().slice_start != s.layer)
				why = "registros de otra cascada en la marca (1er cubo: va por su 1er draw)";
			else {
				// (11-27, log 20261127-060057-463: cascada 3, 453 cubos sin salto por esto.) Los registros ya son los de la
				// cascada (imagen y capa) pero aun no los resolvio ningun draw: se resuelve aqui, como el primer draw del cubo
				// (ResolveRenderDepthTarget, con el candado de render, igual que renderDraw.cpp).
				if (m_ds_last_depth_addr != s.addr || !m_ds_last_depth_id) {
					RenderLockGuard resolve_lock(m_context.GetMutex());
					RenderDepthInfo tmp {};
					ResolveRenderDepthTarget(buffer, tmp);
					H::NatCount(H::g_nat_info_why, "=1: profundidad de la cascada resuelta en la marca");
				}
				if (m_ds_last_depth_addr != s.addr || !m_ds_last_depth_id)
					why = "la ultima profundidad resuelta no es la de la cascada";
				else if (s.partial)
					why = "cubo parcial";
			}
			if (why != nullptr) {
				count_whole(0, false, why);
				return;
			}
			s.shadow_id = m_ds_last_depth_id;
			if (!DsNatWork(m_context, buffer, Role(), pass)) return; // (DsNatFail: el cubo entero al juego)
			s.native_done = true;
			s.first_seen  = true;
			if (s.group_on) {
				s.group_drawn = true;
				++s.groups[pass];
				s.group_cubes[pass] += s.group.size();
			}
			const bool ok = DsNatFlip(s.slot, s.native, &why);
			count_whole(s.native_drawn, ok, why);
		}
		return;
	}
	// Fin.
	if (s.phase == DsNatState::Idle) return;
	{
		// (diagnostico) Los 10 primeros cubos: draws directos de la DCB entre las marcas frente a los que vio
		// DsNatBeforeDraw (todos, no solo los saltados) y las entradas del gancho.
		static std::atomic<uint32_t> logged {0};
		if (H::NatDiag() && s.slot != nullptr && logged.fetch_add(1, std::memory_order_relaxed) < 10) {
			std::printf("[nativo-sombras] cubo cascada %u: entradas %u | DCB entre marcas: draws directos %u, otros draws %u, "
			            "INDIRECT_BUFFER %u, paquetes %u, marca de fin %s | vistos por DsNatBeforeDraw %u | saltados %u | %s\n",
			            pass, s.slot->count, s.scan.direct, s.scan.other_draws, s.scan.ib, s.scan.packets,
			            s.scan.end_found ? "en el mismo bufer" : "NO en el mismo bufer", s.cube_seen, s.skipped_dcb,
			            s.replace ? "reemplazado" : "comparado");
			std::fflush(stdout);
		}
	}
	if (s.slot == nullptr || s.slot->seq != seq || s.pass != pass) return DsNatFail("marca de fin de otro cubo");
	if (s.phase != DsNatState::Active) return DsNatFail("cubo sin draws");
	H::g_nat_dyn_target.store(0xFFu, std::memory_order_relaxed);
	if (s.replace) {
		// =1: el dibujo nativo ya se hizo antes de saltar nada (DsNatBeforeDraw). Si no llego a hacerse (cubo de un solo
		// draw del juego, o fallo), no se salto ningun draw: el juego lo dibujo entero.
		if (!s.native_done) ++s.replace_undone[pass];
		else if (s.skipped_dcb < s.native_drawn) {
			s.doubled[pass] += s.native_drawn - s.skipped_dcb; // (el primero, conservado, cuenta aqui: 1 por cubo)
			if (s.skipped_dcb + 1 < s.native_drawn)
				H::NatCount(H::g_nat_info_why, "=1: cubos con entradas nativas cuyo draw del juego tambien se hizo (sin contar el primero)");
		}
	} else if (!DsNatWork(m_context, buffer, Role(), pass)) {
		return;
	}
	// (Lo que este cubo uso de la cache queda como estaba: los Transit de DsNatCopyLayer.)
	if (s.group_on) {
		for (auto& c: s.group)
			if (c.slot == s.slot) c.released = true;
		if (s.group_cur + 1 >= s.group.size()) {
			s.group.clear();
			s.group_on = false, s.group_drawn = false, s.group_cur = 0;
		}
	}
	H::NatRelease(s.slot);
	s.slot    = nullptr;
	s.phase   = DsNatState::Idle;
	s.replace = false;
}

namespace {
// (Comienzo de un cubo) La capa emulada `layer` de la imagen de las cascadas (antes de los draws del cubo) a A, y el
// VkBuffer de MALLA-NATIVA de cada entrada dibujable. Falso: fallo contado, el cubo se queda emulado.
// (Sin fallo: este cubo no se hace en nativo (=3 fuera de la muestra; =1 cubo no completo en cascada ya validada).)
void DsNatQuiet() {
	auto& s = g_ds_nat;
	if (s.group_on) DsNatGroupRelease();
	DsNativoSombras::NatRelease(s.slot);
	s.slot    = nullptr;
	s.phase   = DsNatState::Idle;
	s.replace = false;
	DsNativoSombras::g_nat_dyn_target.store(0xFFu, std::memory_order_relaxed);
}

bool DsNatStart(RenderContext& context, CommandBuffer& buffer, int role, uint64_t addr, uint32_t layer) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost(DsSombrasAb::CStart);
	auto& s = g_ds_nat;
	if (buffer.IsInvalid() || role != 0 || Spec::Current() != nullptr || FrameCapture::Active() ||
	    SkipShader::g_any.load(std::memory_order_relaxed))
		return DsNatFail("no se puede ahora (especulacion, captura o buffer)"), false;
	// (sombras-ab, KYTY_NATIVO_SOMBRAS_AB) Tramo "juego": no se toma el cubo, el juego lo dibuja todo (el modo cambia solo
	// en el flip; un cubo ya empezado sigue con lo que eligio aqui).
	if (DsSombrasAb::Seconds() != 0 && DsSombrasAb::Juego()) return DsNatQuiet(), false;
	// Que se hace con este cubo: =1 en una cascada ya validada: reemplazar si es completo (si no, nada: el juego);
	// si no: comparar 1 de cada N (KYTY_NATIVO_SOMBRAS_CADA, 16 por defecto; 1 = todos).
	static const uint32_t every = [] {
		const char* text = std::getenv("KYTY_NATIVO_SOMBRAS_CADA");
		const long  n    = text != nullptr ? std::strtol(text, nullptr, 10) : 16;
		return n >= 1 && n <= 100000 ? static_cast<uint32_t>(n) : 16u;
	}();
	// (Log 20261125-094604-438: un cubo con alguna entrada con skinning sin validar se quedaba entero al juego salvo 1 de
	// cada N. Ahora, en una cascada validada: 1 de cada N se compara (asi se valida el skinning); los demas se reemplazan
	// con las entradas con skinning al juego y el resto en nativo.)
	const bool static_ok  = H::Mode() == 1 && H::g_nat_proven[s.pass].load(std::memory_order_relaxed) && s.dyn_ok[s.pass];
	const bool skin_ready = DsNatSkinReady(s.slot, s.pass);
	bool       proven     = static_ok && skin_ready;
	bool       drop_skin  = false;
	if (static_ok && !skin_ready && (s.sample[s.pass]++ % every) != 0) {
		proven    = true;
		drop_skin = true;
	} else if (!proven && !static_ok && (s.sample[s.pass]++ % every) != 0) {
		// (T34 11-27: comparar cada cubo sin validar (bc3999ba) costaba copias y lecturas en cada cubo de las cascadas 0 y 3,
		// que no se validan nunca (c0 skinning nativo_mas_cerca 4,49 %): nativo 56,3 fps vs juego 65,6. Vuelve 1 de cada N.)
		++s.sampled_out[s.pass];
		return DsNatQuiet(), false;
	}
	// Vertices e indices de cada draw: MALLA-NATIVA (Lookup; su candado, tal como esta disenado). Antes de nada.
	uint32_t       empty    = 0;
	const uint32_t count    = s.slot->count;
	uint32_t       drawable = DsNatLookups(context.GetBufferCache(), s.slot, s.vb, s.ib, s.native, s.has_skin, empty);
	drawable -= DsNatS10Gate(buffer, s.slot, nullptr, s.native, s.has_skin);
	if (drop_skin) {
		drawable -= DsNatDropSkin(s.slot, s.native, s.has_skin);
		if (drawable == 0) return DsNatQuiet(), false; // (solo entradas con skinning: el cubo entero al juego)
	}
	if (drawable == 0) return DsNatFail("ninguna entrada con VkBuffer"), false;
	s.partial = drawable + empty != count;
	context.GetCommandScheduler().PopPendingOperations();
	Envio1::NoteDispatch();
	std::optional<Envio1::Scope> envio1_lock(std::in_place, Envio1::Lock);
	RenderLockGuard              lock(context.GetMutex());
	envio1_lock.reset();
	auto& cache = context.GetTextureCache();
	// (IsMetaCleared toma el candado de la cache: antes de tomarlo aqui. La direccion del HTILE, con el candado.)
	uint64_t meta_address = 0;
	bool     meta_htile   = false;
	{
		std::scoped_lock meta_lock {cache.DsNatLock()};
		if (const auto* im = cache.TryGetImage(s.shadow_id); im != nullptr) {
			meta_htile   = im->info.metadata.kind == ImageMetadataKind::Htile;
			meta_address = im->info.metadata.range.address;
		}
	}
	uint32_t         fill    = 0;
	bool             known   = false;
	const bool       cleared = meta_htile && cache.IsMetaCleared(meta_address, layer, &fill, &known);
	std::scoped_lock cache_lock {cache.DsNatLock()};
	auto*            image = cache.TryGetImage(s.shadow_id);
	if (image == nullptr || image->backing.image == nullptr || image->backing.sparse != nullptr || !image->Overlaps(addr, 1))
		return DsNatFail("imagen sin vk::Image, dispersa u otra"), false;
	if (!image->info.IsDepth() || image->info.samples != 1 || layer >= image->backing.layers || image->backing.mip_levels < 1)
		return DsNatFail("no es una imagen de profundidad con esa capa"), false;
	if (image->backing.format != vk::Format::eD32Sfloat) return DsNatFail("la imagen no es D32"), false;
	const vk::Extent2D extent {image->backing.extent.width, image->backing.extent.height};
	if (!DsNatInit(context.GetGraphics(), image->backing.format, extent)) return DsNatFail("no se pudieron crear los recursos"), false;
	// (Log 20261125-020819-021: en los parciales "nativo mas cerca" 38-88 %.) Un clear HTILE pendiente en esta capa
	// (renderDraw.cpp: IsMetaCleared -> loadOp clear en el primer draw del pase) no esta aun en la imagen: copiarla daba
	// la capa del fotograma ANTERIOR. Entonces A empieza con el valor de clear (DB_DEPTH_CLEAR, como el emulador).
	s.addr    = addr;
	s.layer   = layer;
	s.cleared = cleared;
	s.phase   = DsNatState::Active;
	if (proven) {
		// =1: se dibuja directamente en la capa del juego al final, sin copias. En un cubo parcial, solo se saltan los
		// draws del juego de las entradas nativas (DsNatBeforeDraw los empareja por indice); las omitidas las sigue
		// dibujando el juego en su orden. Con un test de profundidad min/max (LESS/LEQUAL/GREATER/GEQUAL) y solo
		// profundidad, el resultado no depende del orden. Con clear HTILE pendiente no: el primer draw del juego que
		// quede aplicaria el clear despues (o nunca, si se saltan todos) -> ese cubo, emulado.
		const auto op = static_cast<vk::CompareOp>(s.dyn[s.pass].depth_compare);
		const bool order_free = op == vk::CompareOp::eLess || op == vk::CompareOp::eLessOrEqual ||
		                        op == vk::CompareOp::eGreater || op == vk::CompareOp::eGreaterOrEqual;
		// (Log 20261125-023042-999: las cascadas 0, 1 y 3 tenian clear HTILE pendiente en todos sus cubos y se quedaban
		// emuladas. Con clear pendiente se deja hacer el PRIMER draw del juego del cubo: aplica el clear (loadOp clear) antes
		// que todo lo demas; el resto se salta como siempre y el nativo dibuja al final, despues de ese clear.)
		if (s.partial && !order_free) {
			++s.quiet_order[s.pass];
			return DsNatQuiet(), false;
		}
		s.replace     = true;
		s.keep_first  = false; // (el clear HTILE lo hace el nativo: DsNatWork)
		s.whole_skip  = false;
		s.native_done  = false;
		s.native_drawn = 0;
		s.first_seen  = false;
		s.next_entry  = 0;
		s.skipped_dcb = 0;
		return true;
	}
	s.replace = false;
	buffer.EndRendering();
	const vk::CommandBuffer command = buffer.Handle();
	buffer.InvalidateGraphicsState();
	if (cleared && (!known || htile_fill_clears_depth(fill) ||
	                htile_fill_depth_uniform(fill, image->info.metadata.stencil_compressed))) {
		if (known) H::NatCount(H::g_nat_info_why, "clear HTILE con relleno conocido (A = DB_DEPTH_CLEAR)");
		DsNatBarrier(command, s.a, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
		             vk::AccessFlagBits2::eMemoryRead, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
		// (Como renderDraw.cpp: con relleno conocido y uniforme, su valor; si no, DB_DEPTH_CLEAR. Log 20261125-061906-403:
		// cascadas 0/1 solo 37 % / 17 % iguales, todos sus cubos con clear HTILE y "A = DB_DEPTH_CLEAR".)
		const bool  sc      = image->info.metadata.stencil_compressed;
		const bool  uniform = known && !htile_fill_clears_depth(fill) && htile_fill_depth_uniform(fill, sc);
		const vk::ClearDepthStencilValue value {uniform ? htile_fill_depth_value(fill, sc) : buffer.GetRegisters().GetDepthClearValue(), 0};
		const vk::ImageSubresourceRange  range {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
		command.clearDepthStencilImage(vk::Image(s.a.image), vk::ImageLayout::eTransferDstOptimal, &value, 1, &range);
	} else {
		DsNatCopyLayer(command, image, layer, s.a, false); // (la capa emulada, antes de los draws de este cubo)
	}
	DsNatBarrier(command, s.a, vk::ImageLayout::eDepthAttachmentOptimal, vk::PipelineStageFlagBits2::eTransfer,
	             vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
	             vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
	return true;
}

} // namespace

bool RenderExecutor::DsNatBeforeDraw(CommandBuffer& buffer, uint64_t index_addr, uint32_t index_count) {
	namespace H = DsNativoSombras;
	const DsSombrasAb::Timer cost(DsSombrasAb::CAntesDraw);
	auto&          s     = g_ds_nat;
	const auto&    regs  = buffer.GetRegisters();
	const uint64_t addr  = regs.GetDepthRenderTarget().z_write_base_addr;
	const uint32_t layer = regs.GetDepthDepthView().slice_start;
	// La imagen de las cascadas: la que resolvio el ultimo draw de un cubo marcado (la misma para las 4; capa = pase).
	// Recordada, la marca de inicio de los cubos siguientes ya copia la capa sin esperar a un draw.
	if (s.phase != DsNatState::Idle && m_ds_last_depth_id && m_ds_last_depth_addr == addr) {
		s.shadow_id   = m_ds_last_depth_id;
		s.shadow_addr = addr;
	}
	if (s.phase != DsNatState::Idle) {
		++s.cube_seen;
		++s.seen_all[s.pass];
	}
	if (s.phase == DsNatState::WaitFirst) {
		if (addr == 0 || s.shadow_addr != addr || !s.shadow_id) return DsNatFail("imagen de la cascada aun no resuelta"), false;
		if (!DsNatStart(m_context, buffer, Role(), addr, layer)) return false;
	}
	if (s.phase != DsNatState::Active || !s.replace) return false;
	++s.game_draws[s.pass];
	s.first_seen = true;
	// (=1) El dibujo nativo, en el PRIMER draw del juego del cubo (el clear HTILE pendiente lo hace el nativo: ya no se
	// conserva ningun draw del juego). Si falla, no se salta nada: el cubo entero lo dibuja el juego. (Solo llegan aqui
	// los cubos que no se pudieron saltar enteros en la marca de inicio.)
	if (!s.native_done) {
		// (Prueba larga 20261125-033425-864: con la VRAM en sobrepresupuesto la cache expulsa y rehace imagenes, y en la
		// misma direccion pueden quedar dos (log: "duplicadas"); la imagen recordada de cubos anteriores podia no ser ya
		// la que dibuja el emulador, y el nativo escribia en otra: sombras que faltan, cada vez mas. Ahora solo se dibuja
		// si el primer draw del juego de ESTE cubo resolvio su profundidad (ResolveRenderDepthTarget) en esta direccion,
		// y con esa imagen.)
		// (Log 20261125-061906-403: 297/312 fallos "no se confirmo": el 1er draw del cubo no vuelve a resolver la
		// profundidad si el destino no cambio (m_ds_last_depth_seq igual). La imagen que el emulador usa ahora es la del
		// ULTIMO ResolveRenderDepthTarget: si se expulsa o se rehace, ese draw vuelve a resolver (sube el seq y cambia el
		// id). Basta con que el ultimo resuelto sea esta direccion y siga en la cache (DsNatWork: TryGetImage).)
		// (Log 20261125-094604-438: cascada 3 con 0 % saltado, todos sus cubos "la ultima profundidad resuelta no es la
		// de la cascada": su primer draw es el primero en esa capa tras otro destino, y este gancho va ANTES de que ese
		// draw resuelva su profundidad. Antes se abandonaba el cubo entero. Ahora este draw lo hace el juego (resuelve la
		// profundidad de la cascada) y el nativo dibuja en el siguiente; con test min/max (GEQUAL medido) dibujar dos veces
		// esa entrada da lo mismo.)
		const auto op1 = static_cast<vk::CompareOp>(s.dyn[s.pass].depth_compare);
		const bool twice_ok = op1 == vk::CompareOp::eLess || op1 == vk::CompareOp::eLessOrEqual ||
		                      op1 == vk::CompareOp::eGreater || op1 == vk::CompareOp::eGreaterOrEqual;
		if ((m_ds_last_depth_addr != s.addr || !m_ds_last_depth_id) && !twice_ok)
			return DsNatFail("=1: la ultima profundidad resuelta no es la de la cascada"), false;
		if (m_ds_last_depth_addr != s.addr || !m_ds_last_depth_id) {
			H::NatCount(H::g_nat_info_why, "=1: 1er draw del cubo al juego (resuelve la profundidad de la cascada), nativo en el siguiente");
		} else {
			s.shadow_id = m_ds_last_depth_id;
			if (!DsNatWork(m_context, buffer, Role(), s.pass)) return false; // (DsNatFail ya dejo el cubo al juego)
			s.native_done = true;
			if (s.group_on) {
				s.group_drawn = true;
				++s.groups[s.pass];
				s.group_cubes[s.pass] += s.group.size();
			}
		}
	}
	// =1: este draw del juego es el de una entrada que dibuja el nativo? Se empareja por (direccion de indices, n) con
	// la siguiente entrada en orden (la lista y la DCB van en el mismo orden; un draw que no pase por aqui, p. ej. uno
	// indirecto, solo hace saltar entradas). Sin pareja: el draw del juego se hace.
	const uint32_t count = s.slot->count;
	for (uint32_t k = s.next_entry; k < count; ++k) {
		const auto&    dr   = s.slot->draws[k];
		const uint64_t addr_k = dr.ibase + uint64_t {dr.istart} * (dr.index32 != 0 ? 4u : 2u);
		// (Tambien las omitidas, en su orden: la misma malla puede salir varias veces en el cubo, unas omitidas y otras
		// no; emparejar solo con las nativas saltaria el draw del juego equivocado.)
		if (addr_k != index_addr || dr.icount != index_count) continue;
		s.next_entry = k + 1;
		// (verificacion del arreglo del ES: el que el gancho arrastro de la DCB frente al real de este draw)
		if (DsSombrasAb::Seconds() != 0 && dr.skip == 0)
			(dr.es == 0 ? DsSombrasAb::g_es_sin : dr.es == H::g_nat_cur_es ? DsSombrasAb::g_es_igual : DsSombrasAb::g_es_distinto)
			    .fetch_add(1, std::memory_order_relaxed);
		if (H::NatDiag()) { // (diagnostico: el VS de las entradas estaticas y de las de skinning)
			auto* seen = dr.skin != 0 ? s.diag_es_skin : s.diag_es_static;
			for (uint32_t i = 0; i < 4; ++i) {
				if (seen[i] == H::g_nat_cur_es) break;
				if (seen[i] == 0) {
					seen[i] = H::g_nat_cur_es;
					// (SKIN, prueba 26: el VS de skinning (hash 0x3866090d8246e47e, 33 slots, 3 buferes) no es el estatico
					// (0x7833ddbaf675c1c3) que reproduce gpu_ds_sombras.vert: su codigo al log, lo lee el gancho.)
					if (dr.skin != 0 && H::g_nat_cur_es != 0 && H::g_req_ready.load(std::memory_order_acquire) == 0) {
						for (auto& p: H::g_req_ptr) p.store(0, std::memory_order_relaxed);
						H::g_req_code[0].store(H::g_nat_cur_es, std::memory_order_relaxed);
						for (uint32_t j = 1; j < 4; ++j) H::g_req_code[j].store(0, std::memory_order_relaxed);
						H::g_req_ready.store(s.pass + 1u, std::memory_order_release);
					}
					break;
				}
			}
		}
		if (s.native[k] != 0 && s.native_done) {
			++s.skipped_dcb;
			++s.game_skipped[s.pass];
			return true;
		}
		return false;
	}
	// (Sin pareja.) Las 5 primeras, al log: lo que llega del paquete y la siguiente entrada nativa que se esperaba.
	// (Un draw de una entrada omitida, p. ej. con skinning, no es discrepancia: tampoco se vuelca.)
	bool known = false;
	for (uint32_t k = s.next_entry; k < count && !known; ++k) {
		const auto& dk = s.slot->draws[k];
		known = dk.icount == index_count && dk.ibase + uint64_t {dk.istart} * (dk.index32 != 0 ? 4u : 2u) == index_addr;
	}
	static std::atomic<uint32_t> dumped {0};
	if (H::NatDiag() && !known && dumped.load(std::memory_order_relaxed) < 5) {
		uint32_t want = s.next_entry;
		while (want < count && (s.slot->draws[want].skip != 0 || s.native[want] == 0)) ++want;
		if (dumped.fetch_add(1, std::memory_order_relaxed) < 5) {
			if (want < count) {
				const auto& dw = s.slot->draws[want];
				std::printf("[nativo-sombras] =1 sin pareja: cascada %u draw del juego indices=%llx n=%u | siguiente entrada nativa k=%u "
				            "(de %u): base=%llx inicio=%u n=%u %s -> indices=%llx%s\n",
				            s.pass, static_cast<unsigned long long>(index_addr), index_count, want, count,
				            static_cast<unsigned long long>(dw.ibase), dw.istart, dw.icount, dw.index32 != 0 ? "32 bits" : "16 bits",
				            static_cast<unsigned long long>(dw.ibase + uint64_t {dw.istart} * (dw.index32 != 0 ? 4u : 2u)),
				            dw.skin != 0 ? " (skinning)" : "");
			} else {
				std::printf("[nativo-sombras] =1 sin pareja: cascada %u draw del juego indices=%llx n=%u | no quedan entradas nativas "
				            "(siguiente=%u de %u)\n",
				            s.pass, static_cast<unsigned long long>(index_addr), index_count, s.next_entry, count);
			}
			std::fflush(stdout);
		}
	}
	return false;
}

bool RenderExecutor::DsNatWants() const noexcept {
	return g_ds_nat.phase != DsNatState::Idle;
}

} // namespace Libs::Graphics
