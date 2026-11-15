#include "fsr3_shaders/fsr3_fg_permutations.h"

#include <bit> // FidelityFX's ffx_util.h calls std::popcount without it
#include <blob_accessors/ffx_frameinterpolation_shaderblobs.h>
#include <blob_accessors/ffx_fsr3upscaler_shaderblobs.h>
#include <blob_accessors/ffx_opticalflow_shaderblobs.h>

#include <gtest/gtest.h>
#include <iterator>
#include <span>
#include <string_view>

namespace {

struct Found {
	uint32_t binding;
	uint32_t count;
	uint32_t set;
};

Found Find(const char* const* names, const uint32_t* bindings, const uint32_t* counts,
                          const uint32_t* sets, uint32_t size, std::string_view name) {
	for (uint32_t i = 0; i < size; ++i) {
		if (name == names[i]) {
			return Found {bindings[i], counts[i], sets[i]};
		}
	}
	return {~0u, 0, ~0u}; // absent
}

Found Srv(const FfxShaderBlob& blob, std::string_view name) {
	return Find(blob.boundSRVTextureNames, blob.boundSRVTextures, blob.boundSRVTextureCounts,
	            blob.boundSRVTextureSpaces, blob.srvTextureCount, name);
}
Found Uav(const FfxShaderBlob& blob, std::string_view name) {
	return Find(blob.boundUAVTextureNames, blob.boundUAVTextures, blob.boundUAVTextureCounts,
	            blob.boundUAVTextureSpaces, blob.uavTextureCount, name);
}
Found SrvBuffer(const FfxShaderBlob& blob, std::string_view name) {
	return Find(blob.boundSRVBufferNames, blob.boundSRVBuffers, blob.boundSRVBufferCounts,
	            blob.boundSRVBufferSpaces, blob.srvBufferCount, name);
}
Found UavBuffer(const FfxShaderBlob& blob, std::string_view name) {
	return Find(blob.boundUAVBufferNames, blob.boundUAVBuffers, blob.boundUAVBufferCounts,
	            blob.boundUAVBufferSpaces, blob.uavBufferCount, name);
}
Found Cb(const FfxShaderBlob& blob, std::string_view name) {
	return Find(blob.boundConstantBufferNames, blob.boundConstantBuffers,
	            blob.boundConstantBufferCounts, blob.boundConstantBufferSpaces, blob.cbvCount, name);
}

FfxShaderBlob Get(uint32_t pass, uint32_t bits, FfxErrorCode expected = FFX_OK) {
	FfxShaderBlob blob {};
	EXPECT_EQ(fsr3UpscalerGetPermutationBlobByIndex(static_cast<FfxFsr3UpscalerPass>(pass), bits,
	                                                &blob),
	          expected);
	return blob;
}

} // namespace

TEST(Fsr3Blobs, EveryBuiltPermutationReflects) {
	EXPECT_EQ(std::size(FSR3_PERMUTATIONS), 44u);
	for (const auto& permutation: FSR3_PERMUTATIONS) {
		SCOPED_TRACE(testing::Message() << "pass " << permutation.pass << " bits " << permutation.bits);
		// FORCE_WAVE64 (64) and ALLOW_FP16 (128) pick the same blob.
		const auto blob = Get(permutation.pass, permutation.bits | 64u | 128u);
		ASSERT_NE(blob.data, nullptr);
		EXPECT_EQ(blob.size, permutation.size);
		EXPECT_EQ(reinterpret_cast<const uint32_t*>(blob.data)[0], 0x07230203u);
		EXPECT_GT(blob.srvTextureCount + blob.uavTextureCount, 0u);
		EXPECT_EQ(blob.srvBufferCount + blob.uavBufferCount + blob.samplerCount, 0u);
		const auto cb = Cb(blob, "cbFSR3Upscaler");
		const auto rcas = Cb(blob, "cbRCAS");
		const auto spd = Cb(blob, "cbSPD");
		const auto reactive = Cb(blob, "cbGenerateReactive");
		EXPECT_GT(cb.count + rcas.count + spd.count + reactive.count, 0u);
	}
}

TEST(Fsr3Blobs, BindingsMatchThePassSources) {
	// ffx_fsr3upscaler_prepare_inputs_pass.glsl and ffx_fsr3upscaler_accumulate_pass.glsl.
	const auto prepare = Get(FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS, 2 | 4 | 16);
	EXPECT_EQ(Srv(prepare, "r_input_motion_vectors").binding, 0u);
	EXPECT_EQ(Srv(prepare, "r_input_depth").binding, 1u);
	EXPECT_EQ(Srv(prepare, "r_input_color_jittered").binding, 2u);
	EXPECT_EQ(Uav(prepare, "rw_dilated_depth").binding, 4u);
	EXPECT_EQ(Cb(prepare, "cbFSR3Upscaler").binding, 8u);
	EXPECT_EQ(Cb(prepare, "cbFSR3Upscaler").set, 0u);

	const auto accumulate = Get(FFX_FSR3UPSCALER_PASS_ACCUMULATE, 2 | 4 | 16);
	EXPECT_EQ(Uav(accumulate, "rw_upscaled_output").binding, 10u);
	EXPECT_EQ(Uav(accumulate, "rw_upscaled_output").count, 1u);
	EXPECT_EQ(Srv(accumulate, "r_input_exposure").binding, 0u);
	EXPECT_EQ(Cb(accumulate, "cbFSR3Upscaler").binding, 12u);
	// Samplers come from the pipeline description, not the blob.
	EXPECT_EQ(Srv(accumulate, "s_LinearClamp").count, 0u);

	EXPECT_EQ(Uav(Get(FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN, 2 | 4 | 32), "rw_internal_upscaled_color")
	              .count,
	          1u);
}

TEST(Fsr3Blobs, UnbuiltPermutationsFail) {
	// SDR colour, display-size motion vectors and jittered motion vectors aren't built.
	EXPECT_EQ(Get(FFX_FSR3UPSCALER_PASS_ACCUMULATE, 4, FFX_ERROR_INVALID_ARGUMENT).data, nullptr);
	EXPECT_EQ(Get(FFX_FSR3UPSCALER_PASS_ACCUMULATE, 2, FFX_ERROR_INVALID_ARGUMENT).data, nullptr);
	EXPECT_EQ(Get(FFX_FSR3UPSCALER_PASS_ACCUMULATE, 2 | 4 | 8, FFX_ERROR_INVALID_ARGUMENT).data,
	          nullptr);
	EXPECT_EQ(Get(FFX_FSR3UPSCALER_PASS_TCR_AUTOGENERATE, 2 | 4, FFX_ERROR_INVALID_ARGUMENT).data,
	          nullptr);
	bool wave64 = true;
	EXPECT_EQ(fsr3UpscalerIsWave64(64u, wave64), FFX_OK);
	EXPECT_FALSE(wave64);
}

TEST(Fsr3Blobs, FrameGenerationPassesReflect) {
	// 11 interpolation passes for normal and inverted depth, 7 optical flow passes.
	EXPECT_EQ(std::size(FSR3_FI_PERMUTATIONS), 22u);
	EXPECT_EQ(std::size(FSR3_OF_PERMUTATIONS), 7u);
	for (const auto& permutation: FSR3_FI_PERMUTATIONS) {
		SCOPED_TRACE(testing::Message() << "FI pass " << permutation.pass << " bits " << permutation.bits);
		FfxShaderBlob blob {};
		// FORCE_WAVE64 (8) and ALLOW_FP16 (16) pick the same blob.
		ASSERT_EQ(frameInterpolationGetPermutationBlobByIndex(static_cast<FfxFrameInterpolationPass>(permutation.pass),
		                                                      FFX_BIND_COMPUTE_SHADER_STAGE, permutation.bits | 8u | 16u,
		                                                      &blob),
		          FFX_OK);
		EXPECT_EQ(blob.size, permutation.size);
		EXPECT_GT(blob.srvTextureCount + blob.uavTextureCount, 0u);
	}
	for (const auto& permutation: FSR3_OF_PERMUTATIONS) {
		SCOPED_TRACE(testing::Message() << "OF pass " << permutation.pass);
		FfxShaderBlob blob {};
		ASSERT_EQ(opticalflowGetPermutationBlobByIndex(static_cast<FfxOpticalflowPass>(permutation.pass), 1u | 2u, &blob),
		          FFX_OK);
		EXPECT_EQ(blob.size, permutation.size);
		EXPECT_GT(blob.srvTextureCount + blob.uavTextureCount, 0u);
	}
	// ffx_frameinterpolation_callbacks_glsl.h: counters are a read-only and a read-write buffer.
	FfxShaderBlob setup {};
	ASSERT_EQ(frameInterpolationGetPermutationBlobByIndex(FFX_FRAMEINTERPOLATION_PASS_SETUP, FFX_BIND_COMPUTE_SHADER_STAGE, 1u,
	                                                      &setup),
	          FFX_OK);
	EXPECT_EQ(UavBuffer(setup, "rw_counters").count, 1u);
	FfxShaderBlob interpolate {};
	ASSERT_EQ(frameInterpolationGetPermutationBlobByIndex(FFX_FRAMEINTERPOLATION_PASS_INTERPOLATION,
	                                                      FFX_BIND_COMPUTE_SHADER_STAGE, 5u, &interpolate),
	          FFX_OK);
	EXPECT_EQ(SrvBuffer(interpolate, "r_counters").count, 1u);
	EXPECT_EQ(UavBuffer(interpolate, "r_counters").count, 0u);
	// Jittered motion vectors aren't built; HDR optical flow isn't.
	FfxShaderBlob none {};
	EXPECT_EQ(frameInterpolationGetPermutationBlobByIndex(FFX_FRAMEINTERPOLATION_PASS_SETUP, FFX_BIND_COMPUTE_SHADER_STAGE, 3u,
	                                                      &none),
	          FFX_ERROR_INVALID_ARGUMENT);
	EXPECT_EQ(opticalflowGetPermutationBlobByIndex(FFX_OPTICALFLOW_PASS_PREPARE_LUMA, 4u, &none), FFX_ERROR_INVALID_ARGUMENT);
}

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
