// FSR 3.1 upscaling through AMD's FidelityFX API (amd_fidelityfx_vk.dll, the signed DLL of the FidelityFX SDK v1.1.4,
// MIT), the same DLL frame generation uses. Inputs and conventions as KytyPS5's FSR 3.1 path (src/local/frame-gen.cpp):
// motion vectors scaled to render pixels toward the previous frame, jitter in pixels, depth flags from the analyzer.
#include "framegen.h"
#include "upscale_backend.h"

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <vector>

namespace inj::upscale::fsr3 {
namespace {

void Message(uint32_t type, const wchar_t* message) {
	static std::atomic<uint32_t> count {0};
	if (count.fetch_add(1) < 100) Log("[fsr3.1] FidelityFX message (%u): %ls", type, message);
}

FfxApiResource Res(const OwnImage& img, uint32_t state, uint32_t usage) {
	FfxApiResource r {};
	r.resource             = reinterpret_cast<void*>(img.image);
	r.state                = state;
	r.description.type     = FFX_API_RESOURCE_TYPE_TEXTURE2D;
	r.description.format   = ffxApiGetSurfaceFormatVK(img.format);
	r.description.width    = img.extent.width;
	r.description.height   = img.extent.height;
	r.description.depth    = 1;
	r.description.mipCount = 1;
	r.description.flags    = FFX_API_RESOURCE_FLAGS_NONE;
	r.description.usage    = usage;
	return r;
}

class Fsr3 final : public Upscaler {
public:
	explicit Fsr3(const ffxFunctions* ffx) : ffx_(ffx) {}
	~Fsr3() override { Destroy(); }
	const char* Name() const override { return "FSR 3.1"; }
	std::string Detail() const override { return detail_; }

	bool Dispatch(const Frame& f) override {
		uint32_t flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
		if (f.hdr) flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
		if (f.reversed) flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
		if (ctx_ == nullptr || flags != flags_ || f.display.width != display_.width || f.display.height != display_.height) {
			if (!Create(f.display, flags)) return false;
		}
		ffxDispatchDescUpscale d {};
		d.header.type        = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
		d.commandList        = f.cb;
		d.color              = Res(*f.color, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
		d.depth              = Res(*f.depth, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_DEPTHTARGET);
		d.motionVectors      = Res(*f.motion, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
		d.output             = Res(*f.output, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);
		d.jitterOffset       = {f.jitter[0], f.jitter[1]};
		d.motionVectorScale  = {f.mv_scale[0], f.mv_scale[1]};
		d.renderSize         = {f.render.width, f.render.height};
		d.upscaleSize        = {f.display.width, f.display.height};
		d.enableSharpening   = false;  // (our RCAS at the display size, as for every upscaler)
		d.sharpness          = 0.0f;
		d.frameTimeDelta     = f.dt_ms;
		d.preExposure        = 1.0f;
		d.reset              = f.reset;
		d.cameraNear         = f.cam.found ? f.cam.near_plane : 0.1f;
		d.cameraFar          = f.cam.found && std::isfinite(f.cam.far_plane) ? f.cam.far_plane : (f.reversed ? FLT_MAX : 1000.0f);
		d.cameraFovAngleVertical  = f.cam.found ? f.cam.fov : 1.0f;
		d.viewSpaceToMetersFactor = 1.0f;
		d.flags              = 0;
		const auto r = ffx_->Dispatch(&ctx_, &d.header);
		if (r != FFX_API_RETURN_OK) {
			if (errors_++ < 8) Log("[fsr3.1] dispatch failed (%u)", r);
			return false;
		}
		if (frames_++ == 0)
			Log("[fsr3.1] first dispatch: %ux%u -> %ux%u, flags 0x%x, mv scale (%.0f, %.0f), jitter (%.3f, %.3f)", f.render.width,
			    f.render.height, f.display.width, f.display.height, flags, f.mv_scale[0], f.mv_scale[1], f.jitter[0], f.jitter[1]);
		return true;
	}

private:
	bool Create(VkExtent2D display, uint32_t flags) {
		if (ctx_ != nullptr) old_.push_back(ctx_);  // (frames already recorded may still use it: destroyed with this upscaler)
		ctx_ = nullptr;
		ffxCreateBackendVKDesc backend {};
		backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
		backend.vkDevice         = R().device;
		backend.vkPhysicalDevice = R().physical;
		backend.vkDeviceProcAddr = R().gdpa;  // the driver's, never our hooks
		ffxCreateContextDescUpscale create {};
		create.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		create.header.pNext   = &backend.header;
		create.flags          = flags;
		create.maxRenderSize  = {display.width, display.height};  // any render size up to the display (the game's own scale)
		create.maxUpscaleSize = {display.width, display.height};
		create.fpMessage      = &Message;
		const auto r = ffx_->CreateContext(&ctx_, &create.header, nullptr);
		if (r != FFX_API_RETURN_OK) {
			Log("[fsr3.1] upscale context not created (%u)", r);
			ctx_ = nullptr;
			return false;
		}
		// The version the DLL picked.
		uint64_t version = 0;
		ffxQueryDescGetVersions versions {};
		versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
		versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
		versions.device = R().device;
		uint64_t count = 1;
		const char* name = nullptr;
		versions.outputCount = &count;
		versions.versionIds = &version;
		versions.versionNames = &name;
		ffx_->Query(nullptr, &versions.header);
		detail_  = std::string("FidelityFX upscaler ") + (name != nullptr ? name : "?");
		display_ = display;
		flags_   = flags;
		Log("[fsr3.1] upscale context: up to %ux%u, flags 0x%x (%s%s%s), %s", display.width, display.height, flags,
		    (flags & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE) ? "HDR " : "LDR ", (flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) ? "reversed depth " : "",
		    "auto exposure", detail_.c_str());
		return true;
	}
	void Destroy() {
		if (ctx_ != nullptr) ffx_->DestroyContext(&ctx_, nullptr);
		ctx_ = nullptr;
		for (auto& c: old_) ffx_->DestroyContext(&c, nullptr);
		old_.clear();
	}

	const ffxFunctions* ffx_;
	ffxContext ctx_ = nullptr;
	std::vector<ffxContext> old_;
	VkExtent2D display_ {};
	uint32_t   flags_ = 0, errors_ = 0;
	uint64_t   frames_ = 0;
	std::string detail_;
};

} // namespace

bool Ready(std::string& why) {
	if (framegen::FfxApi() == nullptr) {
		why = "amd_fidelityfx_vk.dll no esta junto al exe";
		return false;
	}
	return true;
}

std::unique_ptr<Upscaler> Create(std::string& why) {
	const ffxFunctions* ffx = framegen::FfxApi();
	if (ffx == nullptr) {
		why = "amd_fidelityfx_vk.dll not next to the exe";
		return nullptr;
	}
	return std::make_unique<Fsr3>(ffx);
}

} // namespace inj::upscale::fsr3
