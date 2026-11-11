# AMD FidelityFX SDK 1.1.4 (MIT): frame interpolation, optical flow and the Vulkan backend, for
# FSR 3 frame generation. Shader blobs are compiled at build time by gen_shaderblobs.py, which
# replaces the SDK's Windows-only shader compiler.
find_program(KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(ffx_dir "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB ffx_shader_sources CONFIGURE_DEPENDS
	"${ffx_dir}/src/backends/vk/shaders/frameinterpolation/*.glsl"
	"${ffx_dir}/src/backends/vk/shaders/opticalflow/*.glsl"
	"${ffx_dir}/src/backends/vk/shaders/fsr3upscaler/*.glsl"
	"${ffx_dir}/include/FidelityFX/gpu/*.h"
	"${ffx_dir}/include/FidelityFX/gpu/frameinterpolation/*.h"
	"${ffx_dir}/include/FidelityFX/gpu/opticalflow/*.h"
	"${ffx_dir}/include/FidelityFX/gpu/fsr3upscaler/*.h")
set(ffx_blobs "${CMAKE_CURRENT_BINARY_DIR}/ffx_shaderblobs_vk.cpp")
add_custom_command(
	OUTPUT "${ffx_blobs}"
	COMMAND "${Python3_EXECUTABLE}" -I "${ffx_dir}/gen_shaderblobs.py" "${KYTY_GLSLANG_VALIDATOR}"
	        "${ffx_dir}" "${ffx_blobs}"
	DEPENDS "${ffx_dir}/gen_shaderblobs.py" ${ffx_shader_sources}
	COMMENT "Compiling FidelityFX frame interpolation and FSR 3 upscaler shaders"
	VERBATIM)

add_library(kyty_fidelityfx STATIC
	"${ffx_dir}/src/backends/vk/ffx_vk.cpp"
	"${ffx_dir}/src/backends/vk/kyty_swapchain_stub.cpp"
	"${ffx_dir}/src/backends/shared/ffx_shader_blobs.cpp"
	"${ffx_dir}/src/components/frameinterpolation/ffx_frameinterpolation.cpp"
	"${ffx_dir}/src/components/opticalflow/ffx_opticalflow.cpp"
	"${ffx_dir}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
	"${ffx_dir}/src/shared/ffx_assert.cpp"
	"${ffx_dir}/src/shared/ffx_breadcrumbs_list.cpp"
	"${ffx_dir}/src/shared/ffx_message.cpp"
	"${ffx_dir}/src/shared/ffx_object_management.cpp"
	"${ffx_blobs}")
target_include_directories(kyty_fidelityfx
	PUBLIC "${ffx_dir}/include"
	PRIVATE "${ffx_dir}/src" "${ffx_dir}/src/backends/shared" "${ffx_dir}/src/components"
	        "${ffx_dir}/src/shared")
target_compile_definitions(kyty_fidelityfx PRIVATE FFX_FI FFX_OF FFX_FSR3UPSCALER)
target_link_libraries(kyty_fidelityfx PUBLIC Vulkan::Headers)
# (ffx_linux_compat.h: the Vulkan calls routed to the emulator's dispatcher, and on Linux the MSVC library extensions.)
if(WIN32)
	target_compile_options(kyty_fidelityfx PRIVATE "/FI${ffx_dir}/src/ffx_linux_compat.h" /w)
else()
	target_compile_options(kyty_fidelityfx PRIVATE
		-include "${ffx_dir}/src/ffx_linux_compat.h" -w)
endif()
install(FILES "${ffx_dir}/LICENSE.txt" DESTINATION "licenses/fidelityfx")
