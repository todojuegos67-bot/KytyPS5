#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IMAGEFORMS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IMAGEFORMS_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::IR {

// One module for a shader whose sampled images change form from dispatch to dispatch, where the specialization would
// build one module for every combination. Marvel's Wolverine (PPSA03671) 1.05, compute 42ac4a0da0bcdf3a (8116197e60a810a7
// before the code-only hashes): images 1-7 each hold one of the forms below; the fight and the white transition
// dispatched 88 combinations, which the GPU thread compiled as they came (~850 ms each), and the shipped hints v9 hold
// 818 of them. In the forms module each of those images is the root of its forms: one
// binding a form, the dispatch's own descriptor on its form's binding and null ones on the others, and the shader data's
// form dword (Bindings::ImageFormDword, ImageFormBits a slot) picks the binding at each access. A dispatch with an image
// outside these forms, or with anything else this does not check, keeps the specialized module (ApplyImageForms).
// KYTY_IMAGE_FORMS=0 turns it off.
struct ImageForm {
	Prospero::TextureNumericClass numeric_class;
	Decoder::ImageDimension       dimension;
	bool                          cube;
};

constexpr uint64_t ImageFormsShader    = 0x42ac4a0da0bcdf3aull;
constexpr uint32_t ImageFormFirstSlot  = 1u;
constexpr uint32_t ImageFormSlots      = 7u;
// The forms the recorded dispatches hold (journals of 2026-10-30, and every record of the shipped hints v9); the first
// is the root's own. A cube is a 2D array.
constexpr std::array<ImageForm, 7> ImageForms {{
    {Prospero::TextureNumericClass::Float, Decoder::ImageDimension::Dim2DArray, false},
    {Prospero::TextureNumericClass::Uint, Decoder::ImageDimension::Dim2DArray, false},
    {Prospero::TextureNumericClass::Float, Decoder::ImageDimension::Dim3D, false},
    {Prospero::TextureNumericClass::Uint, Decoder::ImageDimension::Dim3D, false},
    {Prospero::TextureNumericClass::Float, Decoder::ImageDimension::Dim1D, false},
    {Prospero::TextureNumericClass::Float, Decoder::ImageDimension::Dim2DArray, true},
    {Prospero::TextureNumericClass::Uint, Decoder::ImageDimension::Dim2DArray, true},
}};
static_assert(ImageForms.size() <= (1u << ImageFormBits) && ImageFormSlots * ImageFormBits <= 32u);

inline bool ImageFormsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_FORMS");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// After BuildResourceSpecialization: makes the dispatch's specialization the forms module's when the shader is one with
// form slots and every slot holds one of the forms, and sets the snapshot's form dword. False: the specialization and
// snapshot are as they were (form dword 0).
inline bool ApplyImageForms(const ResourcePlan& program, ResourceSnapshot& snapshot,
                            ResourceSpecialization& specialization) {
	snapshot.image_forms = 0;
	if (program.shader_hash != ImageFormsShader || !ImageFormsEnabled()) return false;
	auto&      images = specialization.images;
	const auto count  = static_cast<uint32_t>(program.info.images.size());
	// Direct images only: no candidates, groups or FMASK images, so a snapshot's image is the specialization's.
	if (images.size() != count || snapshot.images.size() != count || count < ImageFormFirstSlot + ImageFormSlots ||
	    std::ranges::any_of(images, [](const ResourceSpecialization::Image& image) {
		    return image.indirect_root != ImageResource::NoIndirectImage || image.fmask || image.heap ||
		           image.guest_heap || image.indirect_capacity != 0u;
	    }))
		return false;
	constexpr uint32_t unsupported_flags = Decoder::ImageSampleFlagDerivative | Decoder::ImageSampleFlagCompare |
	                                       Decoder::ImageSampleFlagCd | Decoder::ImageSampleFlagGatherHorizontal;
	std::array<uint32_t, ImageFormSlots> forms {};
	for (uint32_t slot = 0; slot < ImageFormSlots; slot++) {
		const auto  index = ImageFormFirstSlot + slot;
		const auto& base  = program.info.images[index];
		const auto& image = images[index];
		// Sampled at level zero or an explicit level: a form's coordinates are its own, its operands the root's.
		if (base.resource_class != ImageResourceClass::Sampled || base.written || base.atomic || !base.sample_only ||
		    base.depth_compare || base.r128 || base.heap || base.guest_heap || base.mip_mode != ImageMipMode::None ||
		    base.indirect_root != ImageResource::NoIndirectImage || image.mip_count != 1u ||
		    image.conversion_format != Prospero::BufferFormat::kInvalid ||
		    image.shader_swizzle != ShaderImageIdentitySwizzle ||
		    std::ranges::any_of(program.memory_info, [&](const MemoryInfo& memory) {
			    return memory.kind == ResourceKind::Image && memory.resource == index &&
			           (memory.image_has_mip || (memory.image_sample_flags & unsupported_flags) != 0u);
		    }))
			return false;
		const auto form = std::ranges::find_if(ImageForms, [&](const ImageForm& candidate) {
			return candidate.numeric_class == image.numeric_class && candidate.dimension == image.dimension &&
			       candidate.cube == image.cube;
		});
		if (form == ImageForms.end()) return false;
		forms[slot] = static_cast<uint32_t>(form - ImageForms.begin());
	}
	const auto as_form = [](uint32_t form, uint32_t root, uint32_t slot) {
		return ResourceSpecialization::Image {.numeric_class              = ImageForms[form].numeric_class,
		                                      .dimension                  = ImageForms[form].dimension,
		                                      .indirect_root              = root,
		                                      .indirect_mapping_offset    = ImageFormMapping,
		                                      .indirect_search_iterations = slot,
		                                      .cube                       = ImageForms[form].cube};
	};
	const DescriptorValue null_image {.dword_count = 8u};
	for (uint32_t slot = 0; slot < ImageFormSlots; slot++) {
		const auto index      = ImageFormFirstSlot + slot;
		const auto descriptor = snapshot.images[index];
		images[index]         = as_form(0u, index, slot);
		if (forms[slot] != 0u) snapshot.images[index] = null_image;
		for (uint32_t form = 1; form < ImageForms.size(); form++) {
			images.push_back(as_form(form, index, slot));
			snapshot.images.push_back(form == forms[slot] ? descriptor : null_image);
		}
		snapshot.image_forms |= forms[slot] << (ImageFormBits * slot);
	}
	if (!snapshot.image_groups.empty()) snapshot.image_groups.resize(images.size());
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IMAGEFORMS_H_
