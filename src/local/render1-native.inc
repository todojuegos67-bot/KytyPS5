// (render1 F2) KYTY_RENDER1=2 (local/render1.h): the depth passes p0 and p1 drawn a second time, the native way, into a
// depth image of their own, and compared with the normal depth at the end of each batch. The image shown is always the
// normal path's: nothing here writes an image or buffer the game or the normal path reads.
//
// A batch: the draws of p0/p1 (paired with the hook's markers, depth only) that the table / native record emissions
// draw into one depth target, with nothing but state packets between them (the command processor ends it before any
// other packet: local/render1.h KeepsBatch). At its first draw the normal depth is copied into the shadow image (the
// pass starts from the same depth); each draw adds an item: the pipeline prepared for its (programs, state) - the same
// translated shaders, from the cache -, its descriptor set and dynamic offsets, its push constants (user data and the
// table block's device address: constants by BDA), its index buffer and its arguments, groups of consecutive i*0x14
// that the game wrote, as one multi-draw indirect each. At the end the items are drawn into the shadow image, in one
// rendering, binding only what changes, and a compute (render1-cmp.comp) counts the texels whose depth differs by more
// than an epsilon and the largest difference; the slot is read 4 frames later.
// A draw that cannot be reproduced safely is not added and is counted by reason; the batch it was in is not compared
// ("contaminado": the normal depth has a draw the shadow has not).
// Called on the graphics queue's command processor thread only (Role 0, no speculation).

namespace {

struct Render1Item {
	const void*                    state    = nullptr; // NativeXprDrawState
	const PipelineCache::Pipeline* pipeline = nullptr;
	vk::DescriptorSet              set      = nullptr;
	std::array<uint32_t, 4>        offsets {};
	uint32_t                       offset_count = 0;
	vk::ShaderStageFlags           push_stages {};
	uint32_t                       push_first = 0, push_dwords = 0;
	vk::Buffer                     index        = nullptr;
	uint64_t                       index_offset = 0;
	vk::IndexType                  index_type   = vk::IndexType::eUint16;
	uint32_t                       args_first = 0, args_count = 0;
};
struct Render1Args {
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
	uint32_t   count  = 0;
};
struct Render1Shadow {
	VkImage       image  = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	vk::ImageView attach = nullptr, sample = nullptr;
	vk::Format    format = vk::Format::eUndefined;
	vk::Extent3D  extent {};
	uint64_t      used_frame = 0;
};
struct Render1State {
	// The open batch.
	bool                  open = false, tainted = false;
	vk::CommandBuffer     command = nullptr;
	ImageId               target {};
	uint64_t              serial = 0;
	vk::ImageLayout       layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2      access {};
	ImageSubresourceRange range {};
	vk::ImageAspectFlags  aspects {};
	uint32_t              width = 0, height = 0;
	// (F2b) The rendering's color attachments (eUndefined: a null view) and its depth / stencil attachments, as the
	// draws' pipelines were made for them: the shadow rendering has the same, the colors in images of its own.
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	uint32_t                                             color_count = 0;
	bool                                                 has_depth = true, has_stencil = false;
	Render1Shadow*        shadow = nullptr;
	std::vector<Render1Item> items;
	std::vector<Render1Args> args;
	std::vector<uint32_t>    push;
	uint32_t                 draws = 0;
	bool                     index_known = false;
	uint64_t                 index_base = 0, index_bytes = 0;
	vk::IndexType            index_type = vk::IndexType::eUint16;
	vk::Buffer               index      = nullptr;
	uint64_t                 index_offset = 0;
	// Per frame.
	uint64_t frame = UINT64_MAX;
	uint32_t batches_this_frame = 0;
	// Resources.
	std::vector<std::unique_ptr<Render1Shadow>>     shadows;
	std::vector<std::unique_ptr<Render1Shadow>>     dummies; // (F2b) color images nobody reads (never the game's)
	std::vector<std::pair<vk::ImageView, uint64_t>> retired_views; // destroyed 8 frames after their batch
	vk::DescriptorSetLayout set_layout = nullptr;
	vk::PipelineLayout      layout_cmp = nullptr;
	vk::Pipeline            compare    = nullptr;
	vk::Sampler             sampler    = nullptr;
	VkBuffer                results    = VK_NULL_HANDLE;
	VmaAllocation           results_memory = nullptr;
	const uint32_t*         mapped = nullptr;
	bool                    failed = false;
	static constexpr uint32_t Slots = 256, SlotUints = 4;
	std::array<uint64_t, Slots> slot_frame {}; // UINT64_MAX: free
	uint32_t                    next_slot = 0;
	bool                        slots_ready = false;
};
Render1State g_r1;

constexpr uint32_t Render1MaxItems = 4096, Render1MaxArgs = 16384, Render1MaxBatchesPerFrame = 96;
// Depth is compared as stored (0..1): an epsilon of one 24-bit step leaves rounding of the same draw alone.
constexpr float Render1Epsilon = 1.0f / 16777216.0f;

[[nodiscard]] bool Render1HasStencil(vk::Format format) {
	return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
	       format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eS8Uint;
}
[[nodiscard]] bool Render1HasDepth(vk::Format format) {
	return format == vk::Format::eD16Unorm || format == vk::Format::eX8D24UnormPack32 || format == vk::Format::eD32Sfloat ||
	       format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
	       format == vk::Format::eD32SfloatS8Uint;
}

uint64_t Render1Cycles() { return __rdtsc(); }

void Render1ImageBarrier(vk::CommandBuffer command, vk::Image image, vk::ImageAspectFlags aspects, vk::ImageLayout from,
                         vk::ImageLayout to, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src,
                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst) {
	vk::ImageMemoryBarrier2 b {};
	b.srcStageMask        = src_stage;
	b.srcAccessMask       = src;
	b.dstStageMask        = dst_stage;
	b.dstAccessMask       = dst;
	b.oldLayout           = from;
	b.newLayout           = to;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image               = image;
	b.subresourceRange    = {aspects, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &b;
	command.pipelineBarrier2(dependency);
}

void Render1MemoryBarrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src,
                          vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst) {
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

// The compare pipeline (push descriptors) and the host-visible results buffer, once.
bool Render1EnsureCompare(GraphicContext& graphics) {
	auto& r = g_r1;
	if (r.compare != nullptr) return true;
	if (r.failed) return false;
	r.failed          = true;
	const auto device = graphics.device;
	if (graphics.max_push_descriptors < 3) return false;
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t i = 0; i < 3; ++i) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageBuffer;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	}
	vk::DescriptorSetLayoutCreateInfo set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_info.pBindings    = bindings.data();
	if (device.createDescriptorSetLayout(&set_info, nullptr, &r.set_layout) != vk::Result::eSuccess) return false;
	vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &r.set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	if (device.createPipelineLayout(&layout_info, nullptr, &r.layout_cmp) != vk::Result::eSuccess) return false;
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize    = sizeof(RENDER1_CMP_SPV);
	module_info.pCode       = RENDER1_CMP_SPV;
	vk::ShaderModule module = nullptr;
	if (device.createShaderModule(&module_info, nullptr, &module) != vk::Result::eSuccess) return false;
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = r.layout_cmp;
	const auto created = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &r.compare);
	device.destroyShaderModule(module, nullptr);
	if (created != vk::Result::eSuccess) {
		r.compare = nullptr;
		return false;
	}
	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eNearest;
	sampler.minFilter    = vk::Filter::eNearest;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	if (device.createSampler(&sampler, nullptr, &r.sampler) != vk::Result::eSuccess) {
		r.sampler = nullptr;
		device.destroyPipeline(r.compare, nullptr);
		r.compare = nullptr;
		return false;
	}
	VkBufferCreateInfo buffer {};
	buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer.size  = Render1State::Slots * Render1State::SlotUints * sizeof(uint32_t);
	buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo host {};
	host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	host.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, &buffer, &host, &r.results, &r.results_memory, &mapped) != VK_SUCCESS) {
		r.results = VK_NULL_HANDLE;
		device.destroyPipeline(r.compare, nullptr);
		r.compare = nullptr;
		return false;
	}
	r.mapped = static_cast<const uint32_t*>(mapped.pMappedData);
	r.slot_frame.fill(UINT64_MAX);
	r.slots_ready = true;
	r.failed      = false;
	return true;
}

// The shadow depth image for (format, extent): kept while used; one unused for 600 frames is destroyed.
Render1Shadow* Render1EnsureShadow(GraphicContext& graphics, vk::Format format, vk::Extent3D extent, uint64_t frame) {
	auto& r = g_r1;
	for (auto it = r.shadows.begin(); it != r.shadows.end();) {
		auto& s = **it;
		if (s.format == format && s.extent == extent) {
			s.used_frame = frame;
			return &s;
		}
		if (frame > s.used_frame + 600) {
			graphics.device.destroyImageView(s.attach, nullptr);
			graphics.device.destroyImageView(s.sample, nullptr);
			vmaDestroyImage(graphics.allocator, s.image, s.memory);
			it = r.shadows.erase(it);
		} else {
			++it;
		}
	}
	if (r.shadows.size() >= 6) return nullptr;
	auto              s = std::make_unique<Render1Shadow>();
	VkImageCreateInfo create {};
	create.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {extent.width, extent.height, 1};
	create.mipLevels   = 1;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &s->image, &s->memory, nullptr) != VK_SUCCESS) return nullptr;
	const vk::ImageAspectFlags all =
	    vk::ImageAspectFlagBits::eDepth | (Render1HasStencil(format) ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlags {});
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(s->image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {all, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &s->attach) != vk::Result::eSuccess) {
		vmaDestroyImage(graphics.allocator, s->image, s->memory);
		return nullptr;
	}
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &s->sample) != vk::Result::eSuccess) {
		graphics.device.destroyImageView(s->attach, nullptr);
		vmaDestroyImage(graphics.allocator, s->image, s->memory);
		return nullptr;
	}
	s->format     = format;
	s->extent     = extent;
	s->used_frame = frame;
	r.shadows.push_back(std::move(s));
	return r.shadows.back().get();
}

// (F2b) A color image of its own for a color attachment of the shadow rendering (format, extent): the native draws of
// the depth passes write no color (their targets' masks are 0), but the pipelines were made with the attachments'
// formats, so the rendering has them; never the game's images. One unused for 600 frames is destroyed.
Render1Shadow* Render1EnsureColor(GraphicContext& graphics, vk::Format format, vk::Extent3D extent, uint64_t frame) {
	auto& r = g_r1;
	for (auto it = r.dummies.begin(); it != r.dummies.end();) {
		auto& s = **it;
		if (s.format == format && s.extent.width >= extent.width && s.extent.height >= extent.height) {
			s.used_frame = frame;
			return &s;
		}
		if (frame > s.used_frame + 600) {
			graphics.device.destroyImageView(s.attach, nullptr);
			vmaDestroyImage(graphics.allocator, s.image, s.memory);
			it = r.dummies.erase(it);
		} else {
			++it;
		}
	}
	if (r.dummies.size() >= 8) return nullptr;
	auto              s = std::make_unique<Render1Shadow>();
	VkImageCreateInfo create {};
	create.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	create.imageType   = VK_IMAGE_TYPE_2D;
	create.format      = static_cast<VkFormat>(format);
	create.extent      = {extent.width, extent.height, 1};
	create.mipLevels   = 1;
	create.arrayLayers = 1;
	create.samples     = VK_SAMPLE_COUNT_1_BIT;
	create.tiling      = VK_IMAGE_TILING_OPTIMAL;
	create.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(graphics.allocator, &create, &allocate, &s->image, &s->memory, nullptr) != VK_SUCCESS) return nullptr;
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = vk::Image(s->image);
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	if (graphics.device.createImageView(&view_info, nullptr, &s->attach) != vk::Result::eSuccess) {
		vmaDestroyImage(graphics.allocator, s->image, s->memory);
		return nullptr;
	}
	s->format     = format;
	s->extent     = extent;
	s->used_frame = frame;
	r.dummies.push_back(std::move(s));
	return r.dummies.back().get();
}

// Results of batches compared 4 frames ago or more (the GPU is done with them: fewer frames are in flight).
void Render1Collect(GraphicContext& graphics, uint64_t frame) {
	auto& r = g_r1;
	if (!r.slots_ready || r.mapped == nullptr) return;
	bool invalidated = false;
	for (uint32_t i = 0; i < Render1State::Slots; ++i) {
		if (r.slot_frame[i] == UINT64_MAX || frame < r.slot_frame[i] + 4) continue;
		if (!invalidated) {
			(void)vmaInvalidateAllocation(graphics.allocator, r.results_memory, 0, VK_WHOLE_SIZE);
			invalidated = true;
		}
		const uint32_t* s = r.mapped + i * Render1State::SlotUints;
		auto&           g = Render1::g_f2;
		g.diff_pixels.fetch_add(uint64_t {s[0]} + s[3], std::memory_order_relaxed); // (a NaN on either side differs)
		g.Max(s[1]);
		g.pixels.fetch_add(s[2], std::memory_order_relaxed);
		g.results.fetch_add(1, std::memory_order_relaxed);
		r.slot_frame[i] = UINT64_MAX;
	}
	for (size_t i = 0; i < r.retired_views.size();) {
		if (frame >= r.retired_views[i].second + 8) {
			graphics.device.destroyImageView(r.retired_views[i].first, nullptr);
			r.retired_views[i] = r.retired_views.back();
			r.retired_views.pop_back();
		} else {
			++i;
		}
	}
}

void Render1Clear() {
	auto& r = g_r1;
	r.open = false, r.tainted = false;
	r.items.clear(), r.args.clear(), r.push.clear();
	r.draws = 0, r.index_known = false, r.shadow = nullptr;
}

} // namespace

bool RenderExecutor::Render1BatchOpen() const noexcept {
	return g_r1.open && Role() == 0;
}

void RenderExecutor::Render1Taint(uint32_t reason) {
	if (Role() != 0 || Spec::Current() != nullptr) return;
	if (reason < Render1::Reasons && reason != Render1::RViaNormal) Render1::Note(static_cast<Render1::Reason>(reason));
	if (g_r1.open) g_r1.tainted = true;
}

bool RenderExecutor::Render1Begin(CommandBuffer& buffer, const NativeXprDrawState& draw_state, uint32_t refuse) {
	auto& current = Render1::t_current;
	if (!current.native || current.consumed || Role() != 0 || Spec::Current() != nullptr) return false;
	current.consumed = true;
	const uint64_t t0 = Render1Cycles();
	auto&          r  = g_r1;
	const auto     no = [&](Render1::Reason why) {
		Render1::Note(why, std::max<uint32_t>(1, Render1::t_step_draws));
		current.noted = true;
		if (r.open) r.tainted = true;
		Render1::t_native_cycles += Render1Cycles() - t0;
		return false;
	};
	// (The checks of a draw taken are the native path's own work.)
	const auto took = [&] {
		const uint64_t cycles = Render1Cycles() - t0;
		Render1::t_native_cycles += cycles;
		Render1::g_f2.native_cycles.fetch_add(cycles, std::memory_order_relaxed);
	};
	if (refuse < Render1::Reasons) return no(static_cast<Render1::Reason>(refuse));
	// (F2b) The depth target: the one target that is depth; the others are the rendering's color attachments, in order
	// (NativeXprCaptureState: colors first, then the depth). Each refusal "objetivo" is counted by its cause and its
	// target's signature (render1.h NoteTarget).
	auto&          textures = m_context.GetTextureCache();
	const auto&    rendering = draw_state.rendering;
	uint32_t       depth_index = UINT32_MAX, depths = 0;
	for (uint32_t i = 0; i < draw_state.targets.size(); ++i)
		if (draw_state.targets[i].requested.IsDepth()) depth_index = i, ++depths;
	Image*         image = depth_index != UINT32_MAX ? textures.TryGetImage(draw_state.targets[depth_index].id) : nullptr;
	const auto     no_target = [&](Render1::TargetWhy why) {
		Render1::TargetSig sig {};
		sig.why    = why;
		sig.colors = rendering.num_color_attachments;
		sig.targets = static_cast<uint32_t>(draw_state.targets.size());
		if (image != nullptr) {
			sig.format  = static_cast<uint32_t>(image->backing.format);
			sig.width   = image->backing.extent.width;
			sig.height  = image->backing.extent.height;
			sig.layers  = image->backing.layers;
			sig.samples = image->backing.samples;
			sig.type    = static_cast<uint32_t>(image->backing.image_type);
		}
		sig.stencil = rendering.depth_stencil_attachment.has_stencil ? 1u : 0u;
		Render1::NoteTarget(sig, std::max<uint32_t>(1, Render1::t_step_draws));
		return no(Render1::RTarget);
	};
	if (depths == 0) return no_target(Render1::TNoDepth);
	if (depths > 1) return no_target(Render1::TTwoDepths);
	// (F2c) `dynamic.feedback_enabled` only says the device has the feedback-loop extension (LocalDrawRecording::Capture
	// stores the feature for every draw): F2/F2b refused every draw on it. A real feedback loop is the depth attachment in
	// the feedback layout (its aspects non-empty: the shader samples the attachment it writes): that one stays out.
	const bool real_feedback =
	    (draw_state.dynamic.feedback_enabled && static_cast<bool>(draw_state.dynamic.feedback_aspects)) ||
	    rendering.depth_stencil_attachment.image_layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT;
	const auto note_read = [&](bool taken) {
		const auto& rd = Render1::t_read;
		Render1::ReadSig sig {};
		if (image != nullptr) {
			sig.format       = static_cast<uint32_t>(image->backing.format);
			sig.width        = image->backing.extent.width;
			sig.height       = image->backing.extent.height;
			sig.image_layers = image->backing.layers;
		}
		sig.write_layer   = depth_index != UINT32_MAX ? draw_state.targets[depth_index].range.base_layer : 0;
		sig.reads         = rd.known && rd.reads_depth ? 1u : 0u;
		sig.read_layer    = rd.base_layer;
		sig.read_layers   = rd.layers;
		sig.real_feedback = real_feedback ? 1u : 0u;
		sig.taken         = taken ? 1u : 0u;
		Render1::NoteRead(sig, std::max<uint32_t>(1, Render1::t_step_draws));
	};
	if (real_feedback) {
		note_read(false);
		return no_target(Render1::TFeedback);
	}
	// (Reading its own depth image is no feedback for the native draws: they write the shadow image; the game's image
	// stays in the layout the normal path left it in until the batch is drawn. Reading the layer it writes without the
	// feedback layout would be undefined on the normal path too: counted in the read line, not refused.)
	note_read(true);
	if (rendering.num_color_attachments != draw_state.targets.size() - 1 || depth_index != draw_state.targets.size() - 1 ||
	    rendering.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX)
		return no_target(Render1::TColorOrder);
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < rendering.num_color_attachments; ++i) {
		if (rendering.color_attachments[i].image_view == nullptr) continue; // (eUndefined: the pipeline's too)
		const auto* color = textures.TryGetImage(draw_state.targets[i].id);
		if (color == nullptr || color->serial != draw_state.targets[i].serial || color->backing.image == nullptr)
			return no_target(Render1::TColorImage);
		if (color->backing.samples != 1) return no(Render1::RMsaa);
		colors[i] = color->backing.format;
	}
	const auto& target = draw_state.targets[depth_index];
	if (rendering.num_layers != 1 || target.range.layer_count != 1 || target.range.base_level != 0 ||
	    target.range.level_count != 1)
		return no(Render1::RLayers);
	if (image == nullptr) return no_target(Render1::TNoImage);
	if (image->serial != target.serial || image->backing.image == nullptr) return no_target(Render1::TSerial);
	if (!Render1HasDepth(image->backing.format)) return no_target(Render1::TFormat);
	if (image->backing.image_type != vk::ImageType::e2D) return no_target(Render1::TType);
	if (!rendering.depth_stencil_attachment.has_depth || rendering.depth_stencil_attachment.image_view == nullptr)
		return no_target(Render1::TNoDepthAttachment);
	if (image->backing.samples != 1) return no(Render1::RMsaa);
	if (!(image->backing.usage & vk::ImageUsageFlagBits::eSampled) ||
	    !(image->backing.usage & vk::ImageUsageFlagBits::eTransferSrc))
		return no(Render1::RNoSampled);
	if (buffer.IsInvalid()) return no(Render1::RCbChanged);
	const auto vk_buffer = buffer.Handle();
	// Another target, or another command buffer: the open batch ends first (a command buffer already submitted is not
	// recorded into: that batch is dropped).
	if (r.open && r.command != vk_buffer) {
		Render1::Note(Render1::RCbChanged);
		Render1::g_f2.batches_dropped.fetch_add(1, std::memory_order_relaxed);
		Render1Clear();
	}
	const bool same_attachments = r.color_count == rendering.num_color_attachments && r.colors == colors &&
	                              r.has_stencil == rendering.depth_stencil_attachment.has_stencil;
	if (r.open && (r.target != target.id || r.serial != target.serial || r.range.base_layer != target.range.base_layer ||
	               !same_attachments)) {
		took();
		Render1EndBatch(buffer, Render1::ETarget);
		return Render1Begin2(buffer, draw_state, *image, depth_index);
	}
	took();
	if (r.open) return true;
	return Render1Begin2(buffer, draw_state, *image, depth_index);
}

// A new batch at this draw: the normal depth copied into the shadow image as it is now (the draws before it in place).
bool RenderExecutor::Render1Begin2(CommandBuffer& buffer, const NativeXprDrawState& draw_state, Image& image,
                                   uint32_t depth_index) {
	const uint64_t t0     = Render1Cycles();
	auto&          r      = g_r1;
	const auto&    target = draw_state.targets[depth_index];
	auto&          graphics = m_context.GetGraphics();
	const uint64_t frame  = m_context.FrameNumber();
	const auto     no     = [&](Render1::Reason why) {
		Render1::Note(why, std::max<uint32_t>(1, Render1::t_step_draws));
		Render1::t_current.noted = true;
		const uint64_t cycles = Render1Cycles() - t0;
		Render1::g_f2.check_cycles.fetch_add(cycles, std::memory_order_relaxed);
		Render1::t_native_cycles += cycles;
		return false;
	};
	if (r.frame != frame) {
		r.frame              = frame;
		r.batches_this_frame = 0;
		Render1Collect(graphics, frame);
	}
	if (r.batches_this_frame >= Render1MaxBatchesPerFrame) return no(Render1::RCap);
	if (target.metadata != 0 && m_context.GetTextureCache().IsMetaCleared(target.metadata, target.slice))
		return no(Render1::RClear);
	const auto format = image.backing.format;
	const vk::Extent3D extent {image.backing.extent.width, image.backing.extent.height, 1};
	auto* shadow = Render1EnsureShadow(graphics, format, extent, frame);
	if (shadow == nullptr) return no(Render1::RNoResources);
	(void)Render1EnsureCompare(graphics); // (without it the batches are drawn, not compared: counted at the end)
	buffer.EndRendering();
	const auto vk_buffer = buffer.Handle();
	const vk::ImageAspectFlags aspects =
	    vk::ImageAspectFlagBits::eDepth | (Render1HasStencil(format) ? vk::ImageAspectFlagBits::eStencil : vk::ImageAspectFlags {});
	ImageSubresourceRange range = target.range;
	// The shadow image's earlier contents are not needed (overwritten whole): from undefined.
	Render1ImageBarrier(vk_buffer, vk::Image(shadow->image), aspects, vk::ImageLayout::eUndefined,
	                    vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllCommands,
	                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
	                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, range, vk_buffer);
	vk::ImageCopy copy {};
	copy.srcSubresource = vk::ImageSubresourceLayers {aspects, 0, range.base_layer, 1};
	copy.dstSubresource = vk::ImageSubresourceLayers {aspects, 0, 0, 1};
	copy.extent         = extent;
	vk_buffer.copyImage(image.backing.image, vk::ImageLayout::eTransferSrcOptimal, vk::Image(shadow->image),
	                    vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	// The normal depth back where its draw state has it.
	image.Transit(target.layout, target.access, range, vk_buffer);
	Render1ImageBarrier(vk_buffer, vk::Image(shadow->image), aspects, vk::ImageLayout::eTransferDstOptimal,
	                    vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits2::eTransfer,
	                    vk::AccessFlagBits2::eTransferWrite,
	                    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
	                    vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
	// What the emission recorded as bound is no more (its index buffer is bound again: known to the batch).
	buffer.InvalidateGraphicsState();
	Render1Clear();
	r.open    = true;
	r.command = vk_buffer;
	r.target  = target.id;
	r.serial  = target.serial;
	r.layout  = target.layout;
	r.access  = target.access;
	r.range   = range;
	r.aspects = aspects;
	r.width   = std::min(draw_state.rendering.width, extent.width);
	r.height  = std::min(draw_state.rendering.height, extent.height);
	r.shadow  = shadow;
	r.color_count = draw_state.rendering.num_color_attachments;
	r.colors      = {};
	for (uint32_t i = 0; i < r.color_count; ++i)
		if (draw_state.rendering.color_attachments[i].image_view != nullptr)
			if (const auto* color = m_context.GetTextureCache().TryGetImage(draw_state.targets[i].id))
				r.colors[i] = color->backing.format;
	r.has_stencil = draw_state.rendering.depth_stencil_attachment.has_stencil;
	++r.batches_this_frame;
	Render1::g_f2.batches.fetch_add(1, std::memory_order_relaxed);
	const uint64_t cycles = Render1Cycles() - t0;
	Render1::g_f2.check_cycles.fetch_add(cycles, std::memory_order_relaxed);
	Render1::t_native_cycles += cycles; // (validation: not the normal path's time either)
	return true;
}

void RenderExecutor::Render1Add(CommandBuffer& buffer, const NativeXprDrawState& draw_state,
                                const PipelineCache::Pipeline& pipeline, vk::DescriptorSet set,
                                std::span<const uint32_t> offsets, vk::ShaderStageFlags push_stages,
                                std::span<const uint32_t> push, bool index_bound, vk::Buffer index_buffer,
                                uint64_t index_offset, uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type,
                                std::span<const std::pair<Buffer*, uint64_t>> arguments,
                                std::span<const uint32_t> group_sizes, std::span<const uint64_t> commands) {
	const uint64_t t0      = Render1Cycles();
	auto&          r       = g_r1;
	auto&          current = Render1::t_current;
	const auto     no      = [&](Render1::Reason why) {
		Render1::Note(why, std::max<size_t>(1, commands.size()));
		current.noted = true;
		r.tainted     = true;
		Render1::t_native_cycles += Render1Cycles() - t0;
	};
	if (!r.open || buffer.Handle() != r.command) return no(Render1::RCbChanged);
	if (commands.empty() || commands[0] != current.args) return no(Render1::RArgs);
	if (index_bound) {
		r.index_known  = true;
		r.index_base   = index_base;
		r.index_bytes  = index_bytes;
		r.index_type   = index_type;
		r.index        = index_buffer;
		r.index_offset = index_offset;
	} else if (!r.index_known || r.index_base != index_base || r.index_bytes != index_bytes || r.index_type != index_type) {
		return no(Render1::RNoIndex);
	}
	if (r.items.size() >= Render1MaxItems || r.args.size() + arguments.size() > Render1MaxArgs ||
	    offsets.size() > 4)
		return no(Render1::RTooMany);
	Render1Item item;
	item.state        = &draw_state;
	item.pipeline     = &pipeline;
	item.set          = set;
	item.offset_count = static_cast<uint32_t>(offsets.size());
	std::copy(offsets.begin(), offsets.end(), item.offsets.begin());
	item.push_stages  = push_stages;
	item.push_first   = static_cast<uint32_t>(r.push.size());
	item.push_dwords  = static_cast<uint32_t>(push.size());
	r.push.insert(r.push.end(), push.begin(), push.end());
	item.index        = r.index;
	item.index_offset = r.index_offset;
	item.index_type   = r.index_type;
	item.args_first   = static_cast<uint32_t>(r.args.size());
	for (size_t g = 0; g < arguments.size() && g < group_sizes.size(); ++g)
		r.args.push_back({arguments[g].first->Handle(), arguments[g].second, group_sizes[g]});
	item.args_count = static_cast<uint32_t>(r.args.size()) - item.args_first;
	r.items.push_back(item);
	r.draws += static_cast<uint32_t>(commands.size());
	current.captured = true;
	Render1::g_f2.native_draws.fetch_add(commands.size(), std::memory_order_relaxed);
	Render1::g_f2.native_items.fetch_add(1, std::memory_order_relaxed);
	Render1::t_native_cycles += Render1Cycles() - t0;
	Render1::g_f2.native_cycles.fetch_add(Render1Cycles() - t0, std::memory_order_relaxed);
}

// The batch drawn natively into the shadow image, then compared with the normal depth.
void RenderExecutor::Render1EndBatch(CommandBuffer& buffer, uint32_t why) {
	auto& r = g_r1;
	if (!r.open || Role() != 0 || Spec::Current() != nullptr) return;
	const uint64_t t0 = Render1Cycles();
	if (why < Render1::Ends) Render1::g_f2.ends[why].fetch_add(1, std::memory_order_relaxed);
	std::optional<RenderLockGuard> lock;
	lock.emplace(m_context.GetMutex());
	auto& f2 = Render1::g_f2;
	if (buffer.IsInvalid() || buffer.Handle() != r.command) {
		Render1::Note(Render1::RCbChanged);
		f2.batches_dropped.fetch_add(1, std::memory_order_relaxed);
		Render1Clear();
		return;
	}
	auto& textures = m_context.GetTextureCache();
	auto* image    = textures.TryGetImage(r.target);
	if (image == nullptr || image->serial != r.serial || r.shadow == nullptr) {
		Render1::Note(Render1::RTarget);
		f2.batches_dropped.fetch_add(1, std::memory_order_relaxed);
		Render1Clear();
		return;
	}
	buffer.EndRendering();
	const auto vk_buffer = buffer.Handle();
	auto&      shadow    = *r.shadow;
	// 1. The native draws.
	if (!r.items.empty()) {
		vk::RenderingAttachmentInfo depth {};
		depth.imageView   = shadow.attach;
		depth.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		depth.loadOp      = vk::AttachmentLoadOp::eLoad;
		depth.storeOp     = vk::AttachmentStoreOp::eStore;
		// (F2b) The color attachments in images of our own (nothing is written into them: the depth passes' color
		// masks are 0; never the game's images).
		std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
		bool colors_ok = true;
		for (uint32_t i = 0; i < r.color_count; ++i) {
			colors[i].imageView   = nullptr;
			colors[i].imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
			colors[i].loadOp      = vk::AttachmentLoadOp::eDontCare;
			colors[i].storeOp     = vk::AttachmentStoreOp::eDontCare;
			if (r.colors[i] == vk::Format::eUndefined) continue;
			auto* dummy = Render1EnsureColor(m_context.GetGraphics(), r.colors[i], {r.width, r.height, 1}, m_context.FrameNumber());
			if (dummy == nullptr) {
				colors_ok = false;
				break;
			}
			Render1ImageBarrier(vk_buffer, vk::Image(dummy->image), vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
			                    vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			                    vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			                    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead);
			colors[i].imageView = dummy->attach;
		}
		if (!colors_ok) {
			Render1::Note(Render1::RNoResources);
			f2.batches_dropped.fetch_add(1, std::memory_order_relaxed);
			Render1Clear();
			return;
		}
		vk::RenderingInfo info {};
		info.renderArea           = vk::Rect2D {{0, 0}, {r.width, r.height}};
		info.layerCount           = 1;
		info.colorAttachmentCount = r.color_count;
		info.pColorAttachments    = r.color_count != 0 ? colors.data() : nullptr;
		info.pDepthAttachment     = &depth;
		info.pStencilAttachment   = r.has_stencil && Render1HasStencil(shadow.format) ? &depth : nullptr;
		vk_buffer.beginRendering(info);
		const PipelineCache::Pipeline* bound_pipeline = nullptr;
		const void*                    bound_state    = nullptr;
		vk::Buffer                     bound_index    = nullptr;
		uint64_t                       bound_offset   = UINT64_MAX;
		vk::IndexType                  bound_type     = vk::IndexType::eUint16;
		for (const auto& item: r.items) {
			if (item.pipeline != bound_pipeline) {
				vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, BindPipelineHandle(*item.pipeline));
				bound_pipeline = item.pipeline;
			}
			if (item.state != bound_state) {
				NativeXprEmitDynamic(*m_native_xpr, vk_buffer, *static_cast<const NativeXprDrawState*>(item.state), false);
				bound_state = item.state;
			}
			vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, item.pipeline->pipeline_layout, 0, 1, &item.set,
			                             item.offset_count, item.offsets.data());
			if (item.push_dwords != 0)
				vk_buffer.pushConstants(item.pipeline->pipeline_layout, item.push_stages, 0,
				                        item.push_dwords * sizeof(uint32_t), r.push.data() + item.push_first);
			if (item.index != bound_index || item.index_offset != bound_offset || item.index_type != bound_type) {
				vk_buffer.bindIndexBuffer(item.index, item.index_offset, item.index_type);
				bound_index = item.index, bound_offset = item.index_offset, bound_type = item.index_type;
			}
			for (uint32_t a = item.args_first; a < item.args_first + item.args_count; ++a)
				vk_buffer.drawIndexedIndirect(r.args[a].buffer, r.args[a].offset, r.args[a].count,
				                              static_cast<uint32_t>(sizeof(vk::DrawIndexedIndirectCommand)));
		}
		vk_buffer.endRendering();
		// What the emissions recorded as bound is no more.
		buffer.InvalidateGraphicsState();
		if (m_native_xpr) m_native_xpr->emit = {};
	}
	const uint64_t t1 = Render1Cycles();
	f2.native_cycles.fetch_add(t1 - t0, std::memory_order_relaxed);
	// 2. The comparison (not for a batch with a draw the shadow has not).
	auto& graphics = m_context.GetGraphics();
	if (r.tainted) {
		f2.batches_tainted.fetch_add(1, std::memory_order_relaxed);
	} else if (r.compare == nullptr || !r.slots_ready) {
		Render1::Note(Render1::RNoResources);
	} else if (r.slot_frame[r.next_slot] != UINT64_MAX) {
		Render1::Note(Render1::RRingFull);
	} else {
		const uint32_t slot = r.next_slot;
		r.next_slot         = (r.next_slot + 1) % Render1State::Slots;
		vk::ImageViewCreateInfo view_info {};
		view_info.image            = image->backing.image;
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = image->backing.format;
		view_info.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, r.range.base_layer, 1};
		vk::ImageView normal_view  = nullptr;
		if (graphics.device.createImageView(&view_info, nullptr, &normal_view) != vk::Result::eSuccess) {
			Render1::Note(Render1::RNoResources);
		} else {
			r.retired_views.emplace_back(normal_view, m_context.FrameNumber());
			Render1ImageBarrier(vk_buffer, vk::Image(shadow.image), r.aspects, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			                    vk::ImageLayout::eDepthStencilReadOnlyOptimal,
			                    vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eComputeShader,
			                    vk::AccessFlagBits2::eShaderSampledRead);
			image->Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::AccessFlagBits2::eShaderSampledRead, r.range,
			               vk_buffer);
			const uint64_t at = uint64_t {slot} * Render1State::SlotUints * sizeof(uint32_t);
			vk_buffer.fillBuffer(vk::Buffer(r.results), at, Render1State::SlotUints * sizeof(uint32_t), 0);
			Render1MemoryBarrier(vk_buffer, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			                     vk::PipelineStageFlagBits2::eComputeShader,
			                     vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
			vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, r.compare);
			const vk::DescriptorImageInfo  normal_info {r.sampler, normal_view, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
			const vk::DescriptorImageInfo  native_info {r.sampler, shadow.sample, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
			const vk::DescriptorBufferInfo results_info {vk::Buffer(r.results), 0, VK_WHOLE_SIZE};
			std::array<vk::WriteDescriptorSet, 3> writes {};
			for (uint32_t i = 0; i < 3; ++i) {
				writes[i].dstBinding      = i;
				writes[i].descriptorCount = 1;
				writes[i].descriptorType  = i < 2 ? vk::DescriptorType::eCombinedImageSampler : vk::DescriptorType::eStorageBuffer;
			}
			writes[0].pImageInfo  = &normal_info;
			writes[1].pImageInfo  = &native_info;
			writes[2].pBufferInfo = &results_info;
			vk_buffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, r.layout_cmp, 0,
			                               static_cast<uint32_t>(writes.size()), writes.data());
			struct {
				uint32_t width, height, base;
				float    epsilon;
			} push {r.width, r.height, slot * Render1State::SlotUints, Render1Epsilon};
			static_assert(sizeof(push) == 16);
			vk_buffer.pushConstants(r.layout_cmp, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			vk_buffer.dispatch((r.width + 15) / 16, (r.height + 15) / 16, 1);
			Render1MemoryBarrier(vk_buffer, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
			                     vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
			// The normal depth back where the batch's draw state had it.
			image->Transit(r.layout, r.access, r.range, vk_buffer);
			r.slot_frame[slot] = m_context.FrameNumber();
			f2.batches_compared.fetch_add(1, std::memory_order_relaxed);
		}
	}
	const uint64_t t2 = Render1Cycles();
	f2.check_cycles.fetch_add(t2 - t1, std::memory_order_relaxed);
	Render1::t_native_cycles += t2 - t0; // (inside a draw's step: not the normal path's time)
	Render1Clear();
}

// --- render1 F3 (KYTY_RENDER1=3) ------------------------------------------------------------------------------------
// The p0/p1 depth draws F2 took are drawn ONLY the native way, into the game's own depth image, at their packet's place
// in the walk (no batch, nothing deferred: barriers, target changes, clears, labels and every other packet keep their
// order with them). The normal path neither translates nor records them: TryNativeXprDraws calls Render1Draw before
// TableContinue / NativeXprTry and returns when it drew. What the native way does on its own, the minimum a draw needs:
//   1. the pair, variant and draw state lookup (TableTry's caches; nothing is created or stored: a miss, a pair whose
//      shaders changed or whose pipeline is not ready, is the normal path's, which stores it);
//   2. F2's conditions (one depth target of a 2D depth image after the color attachments, no real feedback, one layer
//      and level, single sample, sampled and copyable, no fast clear pending; no buffer writes, no KYTY_HWRT capture);
//   3. the table evaluation of the slots and T#/S# words from the user data (R) as the guest memory is now, its image
//      set, the BDA page table ranges and the block (the same functions as the normal path: the words change every
//      frame - reuso1 [M] - so they are evaluated per call, never kept);
//   4. the arguments' and indices' buffers (the held entries, else ObtainBuffer, as TableDraw's);
//   5. the emission (targets and set images placed, rendering, dynamic state, pipeline, set, push data, indices,
//      one multi-draw indirect per group of consecutive arguments) and what follows a table draw (the frame generation
//      hooks, the emission cache, the last table draw for TableContinue).
// What it leaves out of the normal path: TableContinue / NativeXprTry / TableTry's store requests and native record
// keys, trust1 / draw1 / reuso1 bookkeeping and the census scopes. Any refusal before the emission returns false and
// the normal path draws in its place (as a refused TableDraw: nothing of the draw was recorded yet).

namespace {

// The R values the native calls of the frame had (direct-mapped; a measurement: a collision only undercounts).
struct Render1RSeen {
	uint64_t                  frame = UINT64_MAX;
	std::array<uint64_t, 4096> r {};
};
Render1RSeen g_r1_seen;

} // namespace

bool RenderExecutor::Render1Draw(CommandBuffer& buffer, bool clean, std::span<const uint64_t> commands,
                                 uint64_t index_base, uint64_t index_bytes, vk::IndexType index_type) {
	auto& current = Render1::t_current;
	if (!current.native || Role() != 0 || Spec::Current() != nullptr) return false;
	auto&    f3   = Render1::g_f3;
	uint64_t mark = __rdtsc();
	const auto phase = [&](Render1::F3Phase p) {
		const uint64_t now = __rdtsc();
		f3.phase_cycles[p].fetch_add(now - mark, std::memory_order_relaxed);
		mark = now;
	};
	const auto no = [&](Render1::F3Reason why) {
		phase(Render1::F3PRefused);
		Render1::F3Note(why, std::max<uint32_t>(1, Render1::t_step_draws));
		current.f3_noted = true;
		return false;
	};
	if (kyty_local_table_xpr_mode.load(std::memory_order_relaxed) == 0 ||
	    (kyty_local_native_xpr_diag.load(std::memory_order_relaxed) & 8u) != 0)
		return no(Render1::FTableOff);
	if (buffer.IsInvalid() || commands.empty() || commands.size() > 64 || index_base == 0 || index_bytes == 0)
		return no(Render1::FBuffer);
	// 1. As NativeXprTry before its table try: pending operations, the context lock, retired sets, the state's key.
	auto& cache = NativeXprState();
	if (!clean) cache.state_key_valid = false;
	m_context.GetCommandScheduler().PopPendingOperations();
	RenderLockGuard lock(m_context.GetMutex());
	NativeXprCollect();
	if (!clean || !cache.state_key_valid) {
		cache.state_key       = NativeXprStateKey(buffer);
		cache.state_key_valid = true;
	}
	// The pair, its variant and the draw state, as TableTry finds them (only found: the normal path makes them).
	auto& table = OwnTable();
	TableSweepPairs();
	const auto& sh = buffer.GetShaders();
	const auto& vs = sh.GetVs();
	if (vs.es_regs.data_addr == 0 || vs.ls_regs.data_addr != 0 || vs.hs_regs.data_addr != 0) return no(Render1::FNoPair);
	const auto                    frame      = m_context.FrameNumber();
	const auto                    vs_address = vs.es_regs.data_addr, ps_address = sh.GetPs().ps_regs.data_addr;
	const std::array<uint64_t, 3> pair_key {vs_address, ps_address, static_cast<uint32_t>(vs.start_instance_user_sgpr)};
	TablePair*                    known = table.found_key == pair_key ? table.found_pair : nullptr;
	if (known == nullptr) {
		const auto pair_hash = XXH3_64bits(pair_key.data(), sizeof(pair_key));
		const auto entries   = (kyty_local_table_mru.load(std::memory_order_relaxed) & 1u) != 0 ? TableXpr::PairCacheEntries : 256;
		auto&      cached    = table.pair_cache[pair_hash & (entries - 1)];
		known                = cached.first == pair_hash ? cached.second : nullptr;
		if (known == nullptr) {
			const auto it = table.pairs.find(pair_hash);
			if (it == table.pairs.end()) return no(Render1::FNoPair);
			known  = &it->second;
			cached = {pair_hash, known};
		}
		table.found_key  = pair_key;
		table.found_pair = known;
	}
	auto& pair = *known;
	TablePrefetchPair(pair);
	pair.used_frame = frame;
	if (pair.vs != vs_address || pair.ps != ps_address || pair.start_instance != vs.start_instance_user_sgpr ||
	    pair.generation[0] != ShaderMapGeneration() || pair.respecialize || pair.refused || pair.programs[0] == nullptr)
		return no(Render1::FNoPair);
	if (FrameGen::Enabled() && FrameGen::IsGbufferVertexShader(pair.programs[0]->shader_hash) && FrameGen::WantsCamera())
		return no(Render1::FCamera);
	const bool hint    = (kyty_local_table_mru.load(std::memory_order_relaxed) & 2u) != 0 &&
	                  pair.variant_hint < pair.variants.size() && pair.variants[pair.variant_hint].state == cache.state_key;
	const auto variant = hint ? pair.variants.begin() + pair.variant_hint
	                          : std::ranges::find(pair.variants, cache.state_key, &TableXpr::Variant::state);
	if (variant == pair.variants.end() || variant->refused || variant->pipeline == nullptr) return no(Render1::FNoPair);
	pair.variant_hint = static_cast<uint32_t>(variant - pair.variants.begin());
	if (cache.last_state == nullptr || cache.last_state_key != cache.state_key) {
		const auto& states   = Catalog().NativeXprState().states;
		const auto  state_it = states.find(cache.state_key);
		if (state_it == states.end()) return no(Render1::FNoPair);
		cache.last_state_key = cache.state_key;
		cache.last_state     = state_it->second.get();
	}
	const auto& draw_state = *cache.last_state;
	TablePrefetchArrays(pair, pair.last_set[0]);
	if (!NativeXprTargetsCurrent(draw_state)) return no(Render1::FNoPair);
	// 2. F2's conditions.
	if (variant->hwrt_scene && HwScene::Active()) return no(Render1::FHwrt);
	const auto stages = pair.programs[1] != nullptr ? 2u : 1u;
	for (uint32_t st = 0; st < stages; ++st)
		for (const auto& resource: pair.programs[st]->info.buffers)
			if (resource.written) return no(Render1::FWrites);
	// F2's target conditions (Render1Begin / Render1Begin2), without the batch.
	const auto f3_target = [](TextureCache& textures, const NativeXprDrawState& draw_state) -> Render1::F3Reason {
		const auto& rendering   = draw_state.rendering;
		uint32_t    depth_index = UINT32_MAX, depths = 0;
		for (uint32_t i = 0; i < draw_state.targets.size(); ++i)
			if (draw_state.targets[i].requested.IsDepth()) depth_index = i, ++depths;
		if (depths != 1) return Render1::FTarget;
		if ((draw_state.dynamic.feedback_enabled && static_cast<bool>(draw_state.dynamic.feedback_aspects)) ||
		    rendering.depth_stencil_attachment.image_layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT)
			return Render1::FFeedback;
		if (rendering.num_color_attachments != draw_state.targets.size() - 1 || depth_index != draw_state.targets.size() - 1 ||
		    rendering.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX)
			return Render1::FTarget;
		for (uint32_t i = 0; i < rendering.num_color_attachments; ++i) {
			if (rendering.color_attachments[i].image_view == nullptr) continue;
			const auto* color = textures.TryGetImage(draw_state.targets[i].id);
			if (color == nullptr || color->serial != draw_state.targets[i].serial || color->backing.image == nullptr)
				return Render1::FTarget;
			if (color->backing.samples != 1) return Render1::FMsaa;
		}
		const auto& target = draw_state.targets[depth_index];
		if (rendering.num_layers != 1 || target.range.layer_count != 1 || target.range.base_level != 0 ||
		    target.range.level_count != 1)
			return Render1::FLayers;
		const auto* image = textures.TryGetImage(target.id);
		if (image == nullptr || image->serial != target.serial || image->backing.image == nullptr ||
		    !Render1HasDepth(image->backing.format) || image->backing.image_type != vk::ImageType::e2D ||
		    !rendering.depth_stencil_attachment.has_depth || rendering.depth_stencil_attachment.image_view == nullptr)
			return Render1::FTarget;
		if (image->backing.samples != 1) return Render1::FMsaa;
		if (!(image->backing.usage & vk::ImageUsageFlagBits::eSampled) ||
		    !(image->backing.usage & vk::ImageUsageFlagBits::eTransferSrc))
			return Render1::FNoSampled;
		if (target.metadata != 0 && textures.IsMetaCleared(target.metadata, target.slice)) return Render1::FClear;
		return Render1::F3Reasons;
	};
	auto& texture_cache = m_context.GetTextureCache();
	if (const auto why = f3_target(texture_cache, draw_state); why != Render1::F3Reasons) return no(why);
	phase(Render1::F3PLookup);
	// 3. The evaluation, the set, the BDA ranges and the blocks.
	const std::array<std::span<const uint32_t>, 2> user_data {std::span<const uint32_t> {sh.GetVs().gs_user_sgpr.value},
	                                                          std::span<const uint32_t> {sh.GetPs().ps_user_sgpr.value}};
	{
		const auto* held = &table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u];
		__builtin_prefetch(held);
		__builtin_prefetch(reinterpret_cast<const char*>(held + 1) - 1);
	}
	if (!TableEvaluate(pair, user_data)) return no(Render1::FEvaluate);
	TableGather(pair, user_data);
	phase(Render1::F3PEvaluate);
	auto* set = TableSet(buffer, pair, &*variant, &draw_state);
	if (set == nullptr) return no(Render1::FSet);
	// (Measurement: a draw whose set samples its own depth image, F2c's case.)
	for (const auto& target: draw_state.targets) {
		if (!target.requested.IsDepth()) continue;
		for (const auto& image: set->images)
			if (image.id == target.id) {
				f3.reads_own_depth.fetch_add(1, std::memory_order_relaxed);
				break;
			}
		break;
	}
	phase(Render1::F3PSet);
	auto&                            buffer_cache = m_context.GetBufferCache();
	std::array<vk::DeviceAddress, 2> blocks {};
	bool                             writes = false;
	TablePrepareBda(pair, user_data);
	const auto registration = buffer_cache.RegistrationEpoch();
	if (!TableBlocks(pair, user_data, blocks, &writes)) return no(Render1::FBlocks);
	// What TableDraw does with the evaluated slots for every table draw (each one checks its own switch).
	const std::span<const uint32_t> slots0(table.blocks[0].data(), pair.programs[0]->table_plan.slots.size());
	if (HwScene::Active()) HwScene::NoteCamera(pair.programs[0]->shader_hash, slots0);
	if (PathTrace::Requested()) PathTrace::OnDraw(pair.programs[0]->shader_hash, slots0); // (pt1: the sun)
	if (Render5Flags::Capturing()) FrameGen::OnDraw(pair.programs[0]->shader_hash, slots0, 't');
	if (Render5Flags::Wanted(Render5Flags::Ssr)) FrameGen::OnSsrDraw(pair.programs[0]->shader_hash, slots0);
	if (Render5Flags::Wanted(Render5Flags::Contact)) FrameGen::OnShadowDraw(pair.programs[0]->shader_hash, slots0, false);
	if (Render5Flags::Wanted(Render5Flags::RtSun) && draw_state.targets.size() == 1) {
		const auto& t = draw_state.targets[0];
		const auto& r = t.requested;
		if (r.IsDepth() && r.resources.layers >= 4 && r.extent.width == r.extent.height && r.extent.width >= 512)
			FrameGen::OnSunCascadeDraw(pair.programs[0]->shader_hash, slots0, t.range.base_layer, r.data.address);
	}
	phase(Render1::F3PBlocks);
	// 4. The arguments' and the indices' buffers (TableDraw's held entries and ObtainBuffer).
	constexpr uint64_t stride = sizeof(vk::DrawIndexedIndirectCommand);
	std::array<std::pair<uint64_t, uint64_t>, 4> region_epochs;
	region_epochs.fill({UINT64_MAX, 0});
	const auto cpu_epoch = [&](uint64_t address, uint64_t size) {
		const auto region = address / TRACKER_REGION_SIZE;
		if (region != (address + size - 1) / TRACKER_REGION_SIZE) return buffer_cache.CpuModificationEpoch(address, size);
		auto& known_epoch = region_epochs[region & 3u];
		if (known_epoch.first != region) known_epoch = {region, buffer_cache.CpuModificationEpoch(address, size)};
		return known_epoch.second;
	};
	const auto obtain = [&](TableXpr::HeldBuffer& held, uint64_t address, uint64_t size,
	                        auto&& prepare) -> std::pair<Buffer*, uint64_t> {
		if (held.buffer != nullptr && address >= held.begin && address + size <= held.end &&
		    held.registration == buffer_cache.RegistrationEpoch() && held.cpu_epoch != 0 &&
		    cpu_epoch(held.begin, held.end - held.begin) == held.cpu_epoch)
			return {held.buffer, held.buffer->Offset(address)};
		held = {};
		prepare();
		const auto result = buffer_cache.ObtainBuffer(address, size, false);
		region_epochs.fill({UINT64_MAX, 0});
		const auto owner = buffer_cache.FindBuffer(address, size);
		if (result.first != nullptr && buffer_cache.GetBufferIfLive(owner) == result.first) {
			const auto region = address & ~(TRACKER_REGION_SIZE - 1);
			auto       begin  = std::max(result.first->CpuAddress(), region);
			auto       end    = std::min(result.first->CpuAddress() + result.first->Size(), region + TRACKER_REGION_SIZE);
			if (address + size <= end) {
				const auto epoch = buffer_cache.CpuModificationEpoch(begin, end - begin);
				if (buffer_cache.IsRegionCpuModified(begin, end - begin)) {
					begin = address;
					end   = address + size;
				}
				if (epoch != 0 && !buffer_cache.IsRegionCpuDirty(begin, end - begin))
					held = {begin, end, epoch, buffer_cache.RegistrationEpoch(), result.first};
			}
		}
		return result;
	};
	std::array<std::pair<Buffer*, uint64_t>, 64> arguments;
	std::array<uint32_t, 64>                     group_sizes;
	uint32_t                                     groups = 0;
	for (size_t first = 0; first < commands.size();) {
		size_t count = 1;
		while (first + count < commands.size() && commands[first + count] == commands[first] + count * stride) ++count;
		const auto [owner, offset] = obtain(table.args, commands[first], count * stride, [] {});
		if (owner == nullptr) return no(Render1::FArgs);
		arguments[groups]   = {owner, offset};
		group_sizes[groups] = static_cast<uint32_t>(count);
		++groups;
		first += count;
	}
	auto&      emit       = cache.emit;
	const bool same_index = emit.command == buffer.Handle() && emit.generation == buffer.GraphicsGeneration() &&
	                        emit.index_base == index_base && emit.index_bytes == index_bytes && emit.index_type == index_type;
	std::pair<Buffer*, uint64_t> indices {};
	if (!same_index) {
		indices = obtain(table.indices[(index_base * 0x9e3779b97f4a7c15ull) >> 51u], index_base, index_bytes, [&] {
			if (index_bytes <= BufferCache::CACHING_PAGESIZE && buffer_cache.IsRegionCpuModified(index_base, index_bytes) &&
			    !buffer_cache.IsRegionGpuModified(index_base, index_bytes))
				buffer_cache.EnsureBufferContents(index_base, index_bytes);
		});
		if (indices.first == nullptr) return no(Render1::FIndex);
	}
	if (buffer_cache.RegistrationEpoch() != registration) return no(Render1::FRegistration);
	if (buffer.IsInvalid()) return no(Render1::FBuffer);
	phase(Render1::F3PBuffers);
	// 5. The emission, as TableDraw's (into the game's own targets: the draw is recorded here and nowhere else).
	const auto vk_buffer     = buffer.Handle();
	const bool same_command  = emit.command == vk_buffer && emit.generation == buffer.GraphicsGeneration();
	const auto images_epoch  = texture_cache.ImagesEpoch();
	const bool in_place      = same_command && emit.state == &draw_state && table.last_set == set && table.images_epoch == images_epoch;
	if (!in_place) {
		if (draw_state.emitted_epoch != images_epoch && !NativeXprEmitTargets(draw_state, vk_buffer)) return no(Render1::FTargets);
		if ((set->transit_epoch != images_epoch || set->transit_state != &draw_state) && !TableTransit(*set, vk_buffer, &draw_state))
			return no(Render1::FTargets);
		draw_state.emitted_epoch = set->transit_epoch = texture_cache.ImagesEpoch();
		set->transit_state       = &draw_state;
	}
	m_context.GetCommandScheduler().BeginRendering(draw_state.rendering);
	if (same_command && emit.state != &draw_state && emit.state != nullptr && emit.state->dynamic_hash == draw_state.dynamic_hash &&
	    std::memcmp(&emit.state->dynamic, &draw_state.dynamic, sizeof(draw_state.dynamic)) == 0)
		emit.state = &draw_state;
	if (!same_command || emit.state != &draw_state) NativeXprEmitDynamic(cache, vk_buffer, draw_state, same_command);
	const auto& pipeline = *variant->pipeline;
	if (!same_command || emit.pipeline != &pipeline)
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, BindPipelineHandle(pipeline));
	if (!same_command || emit.pipeline != &pipeline || table.last_set != set)
		vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 0, 1, &set->set, 0, nullptr);
	ShaderRecompiler::IR::PushData push_data;
	uint32_t                       push_dwords = 0;
	for (uint32_t st = 0; st < stages; ++st) {
		const auto& program  = *pair.programs[st];
		const auto& bindings = program.bindings;
		if (!bindings.UsesPushData()) continue;
		push_dwords = std::max(push_dwords, bindings.push_data_start_dword + bindings.ShaderDataDwords());
		for (size_t i = 0; i < bindings.user_data_registers.size(); ++i) {
			const auto index = bindings.user_data_registers[i] - program.user_data_base;
			push_data.dwords[bindings.push_data_start_dword + i] = index < user_data[st].size() ? user_data[st][index] : 0;
		}
		if (program.stage == ShaderType::Pixel)
			for (size_t i = 0; i < table.lod_stats.size(); ++i)
				push_data.dwords[bindings.push_data_start_dword + bindings.LodStatsDword() + i] = table.lod_stats[i];
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword()]     = static_cast<uint32_t>(blocks[st]);
		push_data.dwords[bindings.push_data_start_dword + bindings.TableBlockDword() + 1] = static_cast<uint32_t>(blocks[st] >> 32u);
	}
	if (push_dwords != 0)
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
		                        push_dwords * sizeof(uint32_t), push_data.dwords.data());
	if (!same_index) vk_buffer.bindIndexBuffer(indices.first->Handle(), indices.second, index_type);
	for (uint32_t group = 0; group < groups; ++group)
		vk_buffer.drawIndexedIndirect(arguments[group].first->Handle(), arguments[group].second, group_sizes[group],
		                              static_cast<uint32_t>(stride));
	if (Render5Flags::Wanted(Render5Flags::ParticlesDebug) && !draw_state.targets.empty())
		FrameGen::NoteOtherPathDraw(draw_state.targets[0].requested.data.address, 0, false, 1);
	if (Render5Flags::Capturing()) FrameGen::AfterDraw(buffer);
	if (writes) { // (not reached: such draws are refused above; kept as TableDraw's)
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader);
	}
	emit.command       = vk_buffer;
	emit.generation    = buffer.GraphicsGeneration();
	emit.state         = &draw_state;
	emit.pipeline      = &pipeline;
	emit.index_base    = index_base;
	emit.index_bytes   = index_bytes;
	emit.index_type    = index_type;
	draw_state.Use(frame);
	table.last_pair    = &pair;
	table.last_variant = static_cast<size_t>(&*variant - pair.variants.data());
	table.last_state   = &draw_state;
	table.last_set     = set;
	table.images_epoch = texture_cache.ImagesEpoch();
	current.drawn      = true;
	// (Measurement: the call's R seen earlier in the frame by another native call.)
	if (current.r != 0) {
		auto& seen = g_r1_seen;
		if (seen.frame != frame) {
			seen.frame = frame;
			seen.r.fill(0);
		}
		auto& slot = seen.r[(current.r * 0x9e3779b97f4a7c15ull) >> 52u];
		f3.r_calls.fetch_add(1, std::memory_order_relaxed);
		if (slot == current.r) f3.r_repeated.fetch_add(1, std::memory_order_relaxed);
		slot = current.r;
	}
	phase(Render1::F3PEmit);
	return true;
}
