cmake_minimum_required(VERSION 3.18...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
	set(LINUX TRUE)
endif()

if (NOT (WIN32 OR LINUX OR APPLE))
    message(FATAL_ERROR "only Windows, Linux, and macOS builds are supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include("${KYTY_SOURCE_DIR}/utils.cmake")
include(CTest)

option(KYTY_ENABLE_CLANG_TIDY "Run clang-tidy checks during builds" OFF)

set(KYTY_BUILD_ORIGIN "Source" CACHE STRING "Build provenance shown in diagnostics")
set_property(CACHE KYTY_BUILD_ORIGIN PROPERTY STRINGS Source Official Fork)
if(NOT KYTY_BUILD_ORIGIN STREQUAL "Source" AND NOT KYTY_BUILD_ORIGIN STREQUAL "Official" AND
   NOT KYTY_BUILD_ORIGIN STREQUAL "Fork")
	message(FATAL_ERROR "KYTY_BUILD_ORIGIN must be Source, Official, or Fork")
endif()

set(KYTY_RELEASE_TAG "" CACHE STRING "Release tag embedded in official builds")
set(KYTY_BUILD_REPOSITORY "" CACHE STRING "GitHub repository embedded in fork builds")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
if(KYTY_BUILD_ORIGIN STREQUAL "Official")
	if(KYTY_RELEASE_TAG STREQUAL "")
		message(FATAL_ERROR "Official builds require KYTY_RELEASE_TAG")
	endif()
	set(KYTY_OFFICIAL_BUILD ON)
elseif(KYTY_BUILD_ORIGIN STREQUAL "Fork")
	if(KYTY_BUILD_REPOSITORY STREQUAL "")
		message(FATAL_ERROR "Fork builds require KYTY_BUILD_REPOSITORY")
	endif()
	set(KYTY_FORK_BUILD ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(WIN32)
	# xbyak.h includes <windows.h> itself; keep its min/max macros away from std::min/std::max.
	add_compile_definitions(NOMINMAX)
endif()

include(TestBigEndian)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
endif()

test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
	set(KYTY_ENDIAN KYTY_ENDIAN_BIG)
else()
	set(KYTY_ENDIAN KYTY_ENDIAN_LITTLE)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
	set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()

if(LINUX OR APPLE)
	# macOS rides the POSIX/Linux code paths until it gets a dedicated platform
	set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
else()
	set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
endif()

set(CLANG FALSE)
set(GCC FALSE)
set(KYTY_CLANG_CL FALSE)
get_filename_component(KYTY_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
string(TOLOWER "${KYTY_CXX_COMPILER_NAME}" KYTY_CXX_COMPILER_NAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
	set(CLANG TRUE)
	set(KYTY_COMPILER CLANG)
	if(WIN32 AND KYTY_CXX_COMPILER_NAME STREQUAL "clang-cl")
		set(KYTY_CLANG_CL TRUE)
		set(KYTY_LINKER LLD_LINK)
	elseif(APPLE)
		set(KYTY_LINKER LD64) # Apple's default linker; lld flags don't apply
	else()
		set(KYTY_LINKER LLD)
	endif()
	if ((CMAKE_AR MATCHES "llvm") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0.0) AND (CMAKE_CXX_COMPILER MATCHES "winlibs"))
		string(REPLACE "llvm-ar.exe" "ar.exe" CMAKE_AR ${CMAKE_AR}) # llvm-ar.exe 11.0.0 from winlibs is broken
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC TRUE)
	set(KYTY_COMPILER GCC)
	set(KYTY_LINKER LD)
else()
	message(FATAL_ERROR "only Clang and GCC are supported; found ${CMAKE_CXX_COMPILER_ID}")
endif()

set(KYTY_PROJECT_NAME "Emulator" CACHE STRING "Project name suffix")

string(TOLOWER ${KYTY_COMPILER} KYTY_COMPILER_ID)
string(TOLOWER ${KYTY_LINKER} KYTY_LINKER_ID)

if (CLANG AND (KYTY_LINKER STREQUAL LLD))
	set(KYTY_LD_OPTIONS "-fuse-ld=lld")
endif()

if (KYTY_LINKER STREQUAL LD AND NOT LINUX)
	set(KYTY_LD_OPTIONS "-Wl,--image-base=0x100000000000")
endif()

project(Kyty${KYTY_PROJECT_NAME}${CMAKE_BUILD_TYPE}${KYTY_COMPILER} VERSION 0.3.0)

include_directories(
	${KYTY_THIRD_PARTY_DIR}/gtest/include
	${KYTY_THIRD_PARTY_DIR}/gtest
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum
	${KYTY_THIRD_PARTY_DIR}/ffx_api/include
	${PROJECT_BINARY_DIR}
	${KYTY_SOURCE_DIR}
)

set(KYTY_VERSION "${PROJECT_VERSION}")

configure_file(
  ${KYTY_SOURCE_DIR}/cmake_config.h.in
  ${PROJECT_BINARY_DIR}/cmake_config.h
  )

find_package(Git)

add_custom_target(KytyGitVersion
	COMMAND ${CMAKE_COMMAND}
		-D INPUT_FILE=${KYTY_SOURCE_DIR}/kytyGitVersion.h.in
		-D OUTPUT_FILE=${PROJECT_BINARY_DIR}/kytyGitVersion.h
		-D GIT_EXECUTABLE=${GIT_EXECUTABLE}
		-D GIT_WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
		-P ${KYTY_SOURCE_DIR}/generate_version.cmake
	BYPRODUCTS ${PROJECT_BINARY_DIR}/kytyGitVersion.h
	COMMENT "Generate kytyGitVersion.h"
)

add_library(kyty_git_version INTERFACE)
add_dependencies(kyty_git_version KytyGitVersion)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0)
	list(APPEND KYTY_IWYU
		kyty_emulator
		common
		#launcher
	)
	list(APPEND KYTY_CLANG_TIDY
		kyty_emulator
		#common
		launcher
	)
endif()

option(KYTY_BUILD_LAUNCHER "Build Qt launcher" ON)

config_compiler_and_linker()

if(KYTY_CLANG_CL)
	# The Linux release is built with -O3 -march=native, ThinLTO and a PGO profile; the same for
	# Windows Release builds (clang-cl + lld-link + llvm-lib).
	option(KYTY_NATIVE_OPT "Release: -O3 -march=KYTY_MARCH" ON)
	# A build for other PCs: x86-64-v3 (AVX2, FMA, BMI2). native also takes this CPU's newer
	# extensions (GFNI, AVX-VNNI), which crash CPUs without them at the first use.
	set(KYTY_MARCH "native" CACHE STRING "Release: the -march of KYTY_NATIVE_OPT (x86-64-v3 for other PCs)")
	option(KYTY_THIN_LTO "Release: ThinLTO" ON)
	option(KYTY_PGO_GENERATE "Instrumented build; the live command `pgo <file>` writes the profile" OFF)
	set(KYTY_PGO_USE "" CACHE FILEPATH "llvm-profdata output to optimize with (empty: none)")
	if(KYTY_NATIVE_OPT)
		add_compile_options("$<$<CONFIG:Release>:/clang:-O3>" "$<$<CONFIG:Release>:/clang:-march=${KYTY_MARCH}>")
	endif()
	if(KYTY_THIN_LTO)
		add_compile_options("$<$<CONFIG:Release>:-flto=thin>")
		add_link_options("$<$<CONFIG:Release>:/opt:lldltojobs=all>")
	endif()
	if(KYTY_PGO_GENERATE)
		cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH kyty_llvm_bin)
		file(GLOB kyty_profile_runtime "${kyty_llvm_bin}/../lib/clang/*/lib/windows/clang_rt.profile-x86_64.lib")
		if(NOT kyty_profile_runtime)
			message(FATAL_ERROR "KYTY_PGO_GENERATE: clang_rt.profile-x86_64.lib not found next to ${CMAKE_CXX_COMPILER}")
		endif()
		list(GET kyty_profile_runtime 0 kyty_profile_runtime)
		add_compile_options(-fprofile-instr-generate)
		add_compile_definitions(KYTY_PGO_GENERATE=1)
		link_libraries("${kyty_profile_runtime}")
	elseif(KYTY_PGO_USE)
		add_compile_options("-fprofile-instr-use=${KYTY_PGO_USE}" -Wno-profile-instr-unprofiled
		                    -Wno-profile-instr-out-of-date -Wno-backend-plugin)
	endif()
endif()

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")
add_subdirectory("${KYTY_SOURCE_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")

file(GLOB kyty_emulator_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/libs/*.cpp"
	"${KYTY_SOURCE_DIR}/libs/*.h"
	"${KYTY_SOURCE_DIR}/graphics/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.inc"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.h"
	"${KYTY_SOURCE_DIR}/kernel/*.cpp"
	"${KYTY_SOURCE_DIR}/kernel/*.h"
	"${KYTY_SOURCE_DIR}/loader/*.cpp"
	"${KYTY_SOURCE_DIR}/loader/*.h"
)

find_program(KYTY_GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
# A SPIR-V module of gpu_tiler_<source>.comp compiled with the defines that follow, embedded as GPU_TILER_<NAME>_SPV.
function(kyty_gpu_tiler_shader name source)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${source}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${name}_spv.h")
	string(TOUPPER "GPU_TILER_${name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${ARGN}
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	set(gpu_tiler_shader_headers ${gpu_tiler_shader_headers} "${shader_header}" PARENT_SCOPE)
endfunction()
# The tile/detile modules: one per direction, block family (TileBlockFamily's order) and element size, listed in
# gpu_tiler_variants.h in TileManager's pipeline slot order. Constants, not specialization constants (issue #43).
set(gpu_tiler_families standard256 standard4 standard4_3d standard64 standard64_3d prt prt_3d render_target depth)
set(gpu_tiler_variant_includes "")
set(gpu_tiler_variant_entries "")
foreach(tile 0 1)
	foreach(family IN LISTS gpu_tiler_families)
		foreach(element 1 2 4 8 16)
			set(variant "${family}_e${element}_t${tile}")
			kyty_gpu_tiler_shader(${variant} ${family} -DTILER_ELEMENT_BYTES=${element}u -DTILER_TILE=${tile}u)
			string(TOUPPER "GPU_TILER_${variant}_SPV" variant_symbol)
			string(APPEND gpu_tiler_variant_includes "#include \"gpu_tiler_shaders/gpu_tiler_${variant}_spv.h\"\n")
			string(APPEND gpu_tiler_variant_entries "\t{${variant_symbol}, std::size(${variant_symbol})},\n")
		endforeach()
	endforeach()
endforeach()
file(MAKE_DIRECTORY "${gpu_tiler_generated_dir}")
file(CONFIGURE OUTPUT "${gpu_tiler_generated_dir}/gpu_tiler_variants.h" @ONLY CONTENT
"#pragma once
// Generated by CMakeLists.txt: the tiler's modules in TileManager's pipeline slot order.
#include <cstddef>
#include <cstdint>
#include <iterator>
${gpu_tiler_variant_includes}
struct GpuTilerModule {
	const uint32_t* code;
	size_t          words;
};
inline constexpr GpuTilerModule GPU_TILER_VARIANTS[] = {
${gpu_tiler_variant_entries}};
")
kyty_gpu_tiler_shader(promote_d16_to_d24 promote_d16 -DTILER_D32=0u)
kyty_gpu_tiler_shader(promote_d16_to_d32 promote_d16 -DTILER_D32=1u)
kyty_gpu_tiler_shader(demote_d24_to_d16 demote_d16 -DTILER_D32=0u)
kyty_gpu_tiler_shader(demote_d32_to_d16 demote_d16 -DTILER_D32=1u)
kyty_gpu_tiler_shader(swap_bgra16 swap_bgra16)

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

# Packed LOD report shader used by the buffer cache's report path.
set(lod_stats_pack_source "${KYTY_SOURCE_DIR}/local/lod-stats-pack.comp")
set(lod_stats_pack_spv "${gpu_tiler_generated_dir}/lod_stats_pack.spv")
set(lod_stats_pack_header "${gpu_tiler_generated_dir}/lod_stats_pack_spv.h")
add_custom_command(
	OUTPUT "${lod_stats_pack_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${lod_stats_pack_spv}" "${lod_stats_pack_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${lod_stats_pack_spv} -DOUTPUT=${lod_stats_pack_header}
	        -DSYMBOL=LOD_STATS_PACK_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${lod_stats_pack_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${lod_stats_pack_header}")

# KYTY_UPSCALE_SHARPNESS: sharpening after DLSS or XeSS (src/local/frame-gen.cpp).
set(sharpen_source "${KYTY_SOURCE_DIR}/local/sharpen.comp")
set(sharpen_spv "${gpu_tiler_generated_dir}/sharpen.spv")
set(sharpen_header "${gpu_tiler_generated_dir}/sharpen_spv.h")
add_custom_command(
	OUTPUT "${sharpen_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${sharpen_spv}" "${sharpen_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${sharpen_spv} -DOUTPUT=${sharpen_header}
	        -DSYMBOL=SHARPEN_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${sharpen_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${sharpen_header}")

# KYTY_TAA_REPLACE=upscaler: the raw frame into the TAA's output, its alpha kept (src/local/frame-gen.cpp).
set(taa_bypass_source "${KYTY_SOURCE_DIR}/local/taa-bypass.comp")
set(taa_bypass_spv "${gpu_tiler_generated_dir}/taa_bypass.spv")
set(taa_bypass_header "${gpu_tiler_generated_dir}/taa_bypass_spv.h")
add_custom_command(
	OUTPUT "${taa_bypass_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${taa_bypass_spv}" "${taa_bypass_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${taa_bypass_spv} -DOUTPUT=${taa_bypass_header}
	        -DSYMBOL=TAA_BYPASS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${taa_bypass_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${taa_bypass_header}")

# KYTY_SSGI: screen-space GI before the PS5 Pro path's upscaler (src/local/frame-gen.cpp).
set(ssgi_source "${KYTY_SOURCE_DIR}/local/ssgi.comp")
set(ssgi_spv "${gpu_tiler_generated_dir}/ssgi.spv")
set(ssgi_header "${gpu_tiler_generated_dir}/ssgi_spv.h")
add_custom_command(
	OUTPUT "${ssgi_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${ssgi_spv}" "${ssgi_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${ssgi_spv} -DOUTPUT=${ssgi_header}
	        -DSYMBOL=SSGI_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${ssgi_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${ssgi_header}")

# KYTY_GTAO: GTAO ambient occlusion before the PS5 Pro path's upscaler (src/local/frame-gen.cpp).
set(gtao_source "${KYTY_SOURCE_DIR}/local/gtao.comp")
set(gtao_spv "${gpu_tiler_generated_dir}/gtao.spv")
set(gtao_header "${gpu_tiler_generated_dir}/gtao_spv.h")
add_custom_command(
	OUTPUT "${gtao_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gtao_spv}" "${gtao_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gtao_spv} -DOUTPUT=${gtao_header}
	        -DSYMBOL=GTAO_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gtao_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${gtao_header}")

# KYTY_SSR: screen-space reflections before the PS5 Pro path's upscaler (src/local/frame-gen.cpp).
set(ssr_source "${KYTY_SOURCE_DIR}/local/ssr.comp")
set(ssr_spv "${gpu_tiler_generated_dir}/ssr.spv")
set(ssr_header "${gpu_tiler_generated_dir}/ssr_spv.h")
add_custom_command(
	OUTPUT "${ssr_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${ssr_spv}" "${ssr_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${ssr_spv} -DOUTPUT=${ssr_header}
	        -DSYMBOL=SSR_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${ssr_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${ssr_header}")

# KYTY_RT_REFLECTIONS: screen-space ray-traced reflections with denoising before the upscaler (src/local/rt-reflections.cpp).
set(rt_reflections_source "${KYTY_SOURCE_DIR}/local/rt-reflections.comp")
set(rt_reflections_spv "${gpu_tiler_generated_dir}/rt_reflections.spv")
set(rt_reflections_header "${gpu_tiler_generated_dir}/rt_reflections_spv.h")
add_custom_command(
	OUTPUT "${rt_reflections_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${rt_reflections_spv}" "${rt_reflections_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rt_reflections_spv} -DOUTPUT=${rt_reflections_header}
	        -DSYMBOL=RT_REFLECTIONS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rt_reflections_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rt_reflections_header}")
# KYTY_HWRT_REFL (hwrt2): the same pass with hardware ray queries into the scene TLAS (src/local/hwrt-scene.cpp) where
# the screen-space march misses.
set(rt_reflections_hw_spv "${gpu_tiler_generated_dir}/rt_reflections_hw.spv")
set(rt_reflections_hw_header "${gpu_tiler_generated_dir}/rt_reflections_hw_spv.h")
add_custom_command(
	OUTPUT "${rt_reflections_hw_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os -DKYTY_HWRT
	        -o "${rt_reflections_hw_spv}" "${rt_reflections_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rt_reflections_hw_spv} -DOUTPUT=${rt_reflections_hw_header}
	        -DSYMBOL=RT_REFLECTIONS_HW_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rt_reflections_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rt_reflections_hw_header}")
# KYTY_HWRT_REFL_MATERIAL (hwrt3): + the off-screen hits' albedo (binding 24, the albedo array of src/local/hwrt-scene.cpp).
set(rt_reflections_hwm_spv "${gpu_tiler_generated_dir}/rt_reflections_hwm.spv")
set(rt_reflections_hwm_header "${gpu_tiler_generated_dir}/rt_reflections_hwm_spv.h")
add_custom_command(
	OUTPUT "${rt_reflections_hwm_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os -DKYTY_HWRT -DKYTY_HWRT_MATERIAL
	        -o "${rt_reflections_hwm_spv}" "${rt_reflections_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rt_reflections_hwm_spv} -DOUTPUT=${rt_reflections_hwm_header}
	        -DSYMBOL=RT_REFLECTIONS_HWM_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rt_reflections_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rt_reflections_hwm_header}")

# KYTY_GBUF_VIEW / KYTY_GBUF_TAP: the G-buffer diagnostic view and statistics before the upscaler (src/local/gbuf-view.cpp).
set(gbuf_view_source "${KYTY_SOURCE_DIR}/local/gbuf-view.comp")
set(gbuf_view_spv "${gpu_tiler_generated_dir}/gbuf_view.spv")
set(gbuf_view_header "${gpu_tiler_generated_dir}/gbuf_view_spv.h")
add_custom_command(
	OUTPUT "${gbuf_view_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gbuf_view_spv}" "${gbuf_view_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gbuf_view_spv} -DOUTPUT=${gbuf_view_header}
	        -DSYMBOL=GBUF_VIEW_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gbuf_view_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${gbuf_view_header}")

# KYTY_HWRT_LIGHTS: ray-traced local-light shadows through the scene TLAS (src/local/hw-lights.cpp).
set(hwlights_source "${KYTY_SOURCE_DIR}/local/hw-lights.comp")
set(hwlights_spv "${gpu_tiler_generated_dir}/hwlights.spv")
set(hwlights_header "${gpu_tiler_generated_dir}/hwlights_spv.h")
add_custom_command(
	OUTPUT "${hwlights_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${hwlights_spv}" "${hwlights_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${hwlights_spv} -DOUTPUT=${hwlights_header}
	        -DSYMBOL=HWLIGHTS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${hwlights_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${hwlights_header}")
# KYTY_RT_FLICKER: the RT passes' flicker self-check (src/local/rt-flicker.cpp).
set(rtflicker_source "${KYTY_SOURCE_DIR}/local/rt-flicker.comp")
set(rtflicker_spv "${gpu_tiler_generated_dir}/rtflicker.spv")
set(rtflicker_header "${gpu_tiler_generated_dir}/rtflicker_spv.h")
add_custom_command(
	OUTPUT "${rtflicker_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${rtflicker_spv}" "${rtflicker_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rtflicker_spv} -DOUTPUT=${rtflicker_header}
	        -DSYMBOL=RTFLICKER_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rtflicker_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rtflicker_header}")
# KYTY_SHINE: specular highlights + material-aware sharpening before the upscaler (src/local/shine.cpp).
set(shine_source "${KYTY_SOURCE_DIR}/local/shine.comp")
set(shine_spv "${gpu_tiler_generated_dir}/shine.spv")
set(shine_header "${gpu_tiler_generated_dir}/shine_spv.h")
add_custom_command(
	OUTPUT "${shine_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${shine_spv}" "${shine_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${shine_spv} -DOUTPUT=${shine_header}
	        -DSYMBOL=SHINE_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${shine_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${shine_header}")

# KYTY_GODRAYS: sun shafts + light halos before the upscaler (src/local/godrays.cpp).
set(godrays_source "${KYTY_SOURCE_DIR}/local/godrays.comp")
set(godrays_spv "${gpu_tiler_generated_dir}/godrays.spv")
set(godrays_header "${gpu_tiler_generated_dir}/godrays_spv.h")
add_custom_command(
	OUTPUT "${godrays_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${godrays_spv}" "${godrays_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${godrays_spv} -DOUTPUT=${godrays_header}
	        -DSYMBOL=GODRAYS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${godrays_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${godrays_header}")

# KYTY_HWRT_PARTICLES (hwrt3p): emissive particle light with ray-query shadows into the KYTY_HWRT TLAS (src/local/hwrt-particles.cpp).
set(hwrt_particles_source "${KYTY_SOURCE_DIR}/local/hwrt-particles.comp")
set(hwrt_particles_spv "${gpu_tiler_generated_dir}/hwrt_particles.spv")
set(hwrt_particles_header "${gpu_tiler_generated_dir}/hwrt_particles_spv.h")
add_custom_command(
	OUTPUT "${hwrt_particles_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${hwrt_particles_spv}" "${hwrt_particles_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${hwrt_particles_spv} -DOUTPUT=${hwrt_particles_header}
	        -DSYMBOL=HWRT_PARTICLES_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${hwrt_particles_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${hwrt_particles_header}")

# KYTY_HWRT_CONTACT: ray-traced contact shadows with ray queries into the KYTY_HWRT TLAS (src/local/rt-contact.cpp).
set(rt_contact_source "${KYTY_SOURCE_DIR}/local/rt-contact.comp")
set(rt_contact_spv "${gpu_tiler_generated_dir}/rt_contact.spv")
set(rt_contact_header "${gpu_tiler_generated_dir}/rt_contact_spv.h")
add_custom_command(
	OUTPUT "${rt_contact_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${rt_contact_spv}" "${rt_contact_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rt_contact_spv} -DOUTPUT=${rt_contact_header}
	        -DSYMBOL=RT_CONTACT_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rt_contact_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rt_contact_header}")

# KYTY_RT_SHADOWS: screen-space ray-traced sun shadows before the upscaler (src/local/frame-gen.cpp).
set(cshadow_source "${KYTY_SOURCE_DIR}/local/cshadow.comp")
set(cshadow_spv "${gpu_tiler_generated_dir}/cshadow.spv")
set(cshadow_header "${gpu_tiler_generated_dir}/cshadow_spv.h")
add_custom_command(
	OUTPUT "${cshadow_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${cshadow_spv}" "${cshadow_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${cshadow_spv} -DOUTPUT=${cshadow_header}
	        -DSYMBOL=CSHADOW_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${cshadow_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${cshadow_header}")
# KYTY_RT_SUN (src/local/rt-sun.comp): the sun's visibility from the game's cascades, the CShadowApply pass's other shader.
set(rtsun_source "${KYTY_SOURCE_DIR}/local/rt-sun.comp")
set(rtsun_spv "${gpu_tiler_generated_dir}/rtsun.spv")
set(rtsun_header "${gpu_tiler_generated_dir}/rtsun_spv.h")
add_custom_command(
	OUTPUT "${rtsun_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${rtsun_spv}" "${rtsun_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rtsun_spv} -DOUTPUT=${rtsun_header}
	        -DSYMBOL=RTSUN_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rtsun_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rtsun_header}")
# KYTY_HWRT_SHADOW: the same shader with the scene TLAS's sun rays (-DKYTY_HWRT_SHADOW).
set(rtsun_hw_spv "${gpu_tiler_generated_dir}/rtsun_hw.spv")
set(rtsun_hw_header "${gpu_tiler_generated_dir}/rtsun_hw_spv.h")
add_custom_command(
	OUTPUT "${rtsun_hw_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os -DKYTY_HWRT_SHADOW
	        -o "${rtsun_hw_spv}" "${rtsun_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${rtsun_hw_spv} -DOUTPUT=${rtsun_hw_header}
	        -DSYMBOL=RTSUN_HW_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${rtsun_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${rtsun_hw_header}")
# KYTY_HWRT (src/local/hwrt-scene.cpp): the scene capture's world-space gather and the ray query debug view.
# (hwrt3, KYTY_HWRT_REFL_MATERIAL: albedo, albedo_array: a base-colour texture downsampled into the albedo array)
foreach(hwrt_shader gather debug debug11 gi albedo albedo_array)
	string(TOUPPER "${hwrt_shader}" hwrt_upper)
	set(hwrt_source "${KYTY_SOURCE_DIR}/local/hwrt-${hwrt_shader}.comp")
	set(hwrt_defines "")
	if(hwrt_shader STREQUAL "debug11")
		# (the same view into a B10G11R11 target: the PS5 Pro path's FSR output)
		set(hwrt_source "${KYTY_SOURCE_DIR}/local/hwrt-debug.comp")
		set(hwrt_defines "-DKYTY_FMT_R11")
	endif()
	if(hwrt_shader STREQUAL "albedo_array")
		set(hwrt_source "${KYTY_SOURCE_DIR}/local/hwrt-albedo.comp")
		set(hwrt_defines "-DKYTY_ARRAY")
	endif()
	set(hwrt_spv "${gpu_tiler_generated_dir}/hwrt_${hwrt_shader}.spv")
	set(hwrt_header "${gpu_tiler_generated_dir}/hwrt_${hwrt_shader}_spv.h")
	add_custom_command(
		OUTPUT "${hwrt_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${hwrt_defines}
		        -o "${hwrt_spv}" "${hwrt_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${hwrt_spv} -DOUTPUT=${hwrt_header}
		        -DSYMBOL=HWRT_${hwrt_upper}_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${hwrt_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		VERBATIM
	)
	list(APPEND kyty_emulator_src "${hwrt_header}")
endforeach()

# FSR 3 frame generation's HUD layer (src/local/fsr3-fg.cpp).
set(hud_split_source "${KYTY_SOURCE_DIR}/local/hud-split.comp")
set(hud_split_spv "${gpu_tiler_generated_dir}/hud_split.spv")
set(hud_split_header "${gpu_tiler_generated_dir}/hud_split_spv.h")
add_custom_command(
	OUTPUT "${hud_split_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${hud_split_spv}" "${hud_split_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${hud_split_spv} -DOUTPUT=${hud_split_header}
	        -DSYMBOL=HUD_SPLIT_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${hud_split_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${hud_split_header}")
# KYTY_KBM_WARP's late reprojection under a still HUD (src/local/kbm-warp.cpp).
set(kbm_warp_source "${KYTY_SOURCE_DIR}/local/kbm-warp.comp")
set(kbm_warp_spv "${gpu_tiler_generated_dir}/kbm_warp.spv")
set(kbm_warp_header "${gpu_tiler_generated_dir}/kbm_warp_spv.h")
add_custom_command(
	OUTPUT "${kbm_warp_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${kbm_warp_spv}" "${kbm_warp_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${kbm_warp_spv} -DOUTPUT=${kbm_warp_header}
	        -DSYMBOL=KBM_WARP_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${kbm_warp_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${kbm_warp_header}")
# The game's exposure on the upscaled image of its super resolution path (renderSuperResolution.cpp).
set(sr_exposure_source "${KYTY_SOURCE_DIR}/local/sr-exposure.comp")
set(sr_exposure_spv "${gpu_tiler_generated_dir}/sr_exposure.spv")
set(sr_exposure_header "${gpu_tiler_generated_dir}/sr_exposure_spv.h")
add_custom_command(
	OUTPUT "${sr_exposure_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${sr_exposure_spv}" "${sr_exposure_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${sr_exposure_spv} -DOUTPUT=${sr_exposure_header}
	        -DSYMBOL=SR_EXPOSURE_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${sr_exposure_source}" "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	VERBATIM
)
list(APPEND kyty_emulator_src "${sr_exposure_header}")

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
set(gpu_blit_shader_sources
	"${gpu_tiler_shader_dir}/gpu_blit_fs_triangle.vert"
	"${gpu_tiler_shader_dir}/gpu_blit_color_to_ms_depth.frag"
	"${gpu_tiler_shader_dir}/gpu_imgui_hdr.frag"
)
foreach(shader_source IN LISTS gpu_blit_shader_sources)
	get_filename_component(shader_name "${shader_source}" NAME_WE)
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}"
		VERBATIM
	)
	list(APPEND gpu_blit_shader_headers "${shader_header}")
endforeach()
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})

list(APPEND kyty_emulator_src
	"${KYTY_SOURCE_DIR}/emulator.h"
	"${KYTY_SOURCE_DIR}/emulator.cpp"
	"${KYTY_SOURCE_DIR}/local/local-platform.h"
	"${KYTY_SOURCE_DIR}/local/local-platform.cpp"
	"${KYTY_SOURCE_DIR}/local/dyn-multiplier.h"
	"${KYTY_SOURCE_DIR}/local/frame-gen.h"
	"${KYTY_SOURCE_DIR}/local/frame-gen.cpp"
	"${KYTY_SOURCE_DIR}/local/rt-reflections.h"
	"${KYTY_SOURCE_DIR}/local/gbuf-view.h"
	"${KYTY_SOURCE_DIR}/local/gbuf-view.cpp"
	"${KYTY_SOURCE_DIR}/local/rt-reflections.cpp"
	"${KYTY_SOURCE_DIR}/local/hwrt-scene.h"
	"${KYTY_SOURCE_DIR}/local/hwrt-scene.cpp"
	"${KYTY_SOURCE_DIR}/local/hwrt-gi.h"
	"${KYTY_SOURCE_DIR}/local/hwrt-gi.cpp"
	"${KYTY_SOURCE_DIR}/local/rt-local-lights.h"
	"${KYTY_SOURCE_DIR}/local/rt-local-lights.cpp"
	"${KYTY_SOURCE_DIR}/local/hw-lights.h"
	"${KYTY_SOURCE_DIR}/local/hw-lights.cpp"
	"${KYTY_SOURCE_DIR}/local/rt-contact.h"
	"${KYTY_SOURCE_DIR}/local/rt-contact.cpp"
	"${KYTY_SOURCE_DIR}/local/rt-flicker.h"
	"${KYTY_SOURCE_DIR}/local/rt-flicker.cpp"
	"${KYTY_SOURCE_DIR}/local/shine.h"
	"${KYTY_SOURCE_DIR}/local/shine.cpp"
	"${KYTY_SOURCE_DIR}/local/godrays.h"
	"${KYTY_SOURCE_DIR}/local/godrays.cpp"
	"${KYTY_SOURCE_DIR}/local/hwrt-particles.h"
	"${KYTY_SOURCE_DIR}/local/hwrt-particles.cpp"
	"${KYTY_SOURCE_DIR}/local/fsr3-fg.h"
	"${KYTY_SOURCE_DIR}/local/fsr3-fg.cpp"
	"${KYTY_SOURCE_DIR}/local/kbm-warp.h"
	"${KYTY_SOURCE_DIR}/local/kbm-warp.cpp"
	"${KYTY_SOURCE_DIR}/local/anti-lag.h"
	"${KYTY_SOURCE_DIR}/local/anti-lag.cpp"
	"${KYTY_SOURCE_DIR}/local/xess-sr.cpp"
	"${KYTY_SOURCE_DIR}/local/sync-stats.h"
	"${KYTY_SOURCE_DIR}/local/sync-stats.cpp"
	"${KYTY_SOURCE_DIR}/local/debug-snapshot.h"
	"${KYTY_SOURCE_DIR}/local/debug-snapshot.cpp"
	"${KYTY_SOURCE_DIR}/local/perf-overlay.h"
	"${KYTY_SOURCE_DIR}/local/perf-overlay.cpp"
	"${KYTY_SOURCE_DIR}/local/soft-release.h"
	"${KYTY_SOURCE_DIR}/local/soft-release.cpp"
	"${KYTY_SOURCE_DIR}/local/ram-guard.h"
	"${KYTY_SOURCE_DIR}/local/ram-guard.cpp"
	"${KYTY_SOURCE_DIR}/local/resource-plan-serial.h"
	"${KYTY_SOURCE_DIR}/local/resource-plan-serial.cpp"
	"${KYTY_SOURCE_DIR}/local/jit-arena.h"
	"${KYTY_SOURCE_DIR}/local/jit-arena.cpp"
)
list(REMOVE_DUPLICATES kyty_emulator_src)

if (KYTY_CLANG_CL)
	set_source_files_properties(${kyty_emulator_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

set(kyty_emulator_link_libraries common Vulkan::Headers spirv-tools-opt spirv-tools SDL2-static imgui xxhash FFmpeg::ffmpeg fmt::fmt nlohmann_json::nlohmann_json LibAtrac9 kyty_git_version Zydis::Zydis xbyak::xbyak libzstd_static)

# Linux system libraries required by the static FFmpeg archive.
if(LINUX)
	find_package(Threads REQUIRED)
	list(APPEND kyty_emulator_link_libraries m ${CMAKE_DL_LIBS} Threads::Threads)

	# Optional FFmpeg dependencies.
	find_package(ZLIB)
	if(ZLIB_FOUND)
		list(APPEND kyty_emulator_link_libraries ZLIB::ZLIB)
	endif()
	find_library(KYTY_BZ2_LIBRARY bz2)
	if(KYTY_BZ2_LIBRARY)
		list(APPEND kyty_emulator_link_libraries ${KYTY_BZ2_LIBRARY})
	endif()
	find_library(KYTY_LZMA_LIBRARY lzma)
	if(KYTY_LZMA_LIBRARY)
		list(APPEND kyty_emulator_link_libraries ${KYTY_LZMA_LIBRARY})
	endif()
endif()

set(inc_headers
	${KYTY_SOURCE_DIR}
	${KYTY_SOURCE_DIR}/local
	${KYTY_THIRD_PARTY_DIR}/SDL2/include
	${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxhash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
	${KYTY_THIRD_PARTY_DIR}/renderdoc
	${KYTY_THIRD_PARTY_DIR}/stb
)

if (KYTY_CLANG_CL)
	list(APPEND kyty_emulator_link_libraries winpthread)
	list(APPEND inc_headers
		${KYTY_THIRD_PARTY_DIR}/winpthread/include
	)
endif()

list(APPEND check_headers
	${KYTY_SOURCE_DIR}
)

function(add_kyty_full_emulator_test target source)
	add_executable(${target} EXCLUDE_FROM_ALL ${source} ${kyty_emulator_src})
	target_link_libraries(${target} ${kyty_emulator_link_libraries})
	target_include_directories(${target} PRIVATE ${inc_headers})
	# These targets share the emulator's sources: give them its linker too, or ThinLTO objects
	# fail to resolve on a stock toolchain (GNU ld has no ThinLTO without the LLVM plugin).
	if(KYTY_LD_OPTIONS)
		target_link_options(${target} PRIVATE ${KYTY_LD_OPTIONS})
	endif()
	if (WIN32)
		target_link_libraries(${target} iphlpapi)
	endif()
	if (KYTY_CLANG_CL)
		target_link_libraries(${target} onecore)
		# The threads' stacks as the emulator's (below): the driver compiles pipelines on the calling thread.
		target_link_options(${target} PRIVATE "/STACK:16777216")
		add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:${target}>/libwinpthread-1.dll)
	endif()
	# The macOS x86_64 guest address space needs its .zerofill segments anchored
	# by linker flags, or the kernel kills the binary on load (posix_spawn EIO).
	configure_macos_guest_address_space(${target})
endfunction()

function(configure_macos_guest_address_space target)
	if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
			(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
		target_sources(${target} PRIVATE "${KYTY_SOURCE_DIR}/kernel/macosGuestAddressSpace.cpp")
		target_compile_definitions(${target} PRIVATE KYTY_LINKED_GUEST_ADDRESS_SPACE=1)
		target_link_options(${target} PRIVATE
			-Wl,-ld_classic,-no_pie,-no_fixup_chains,-no_huge,-pagezero_size,0x40000,-segaddr,SYSTEM_MANAGED,0x40000,-segaddr,SYSTEM_RESERVED,0x7ffffc000,-segaddr,USER_AREA,0x7000000000,-image_base,0x700000000000)
	endif()
endfunction()

add_kyty_full_emulator_test(shader_cfg_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")
add_kyty_full_emulator_test(xpr_replay_tests "${KYTY_TESTS_DIR}/XprReplayTests.cpp")
# The static shader precompile, a program of its own (src/local/shader-precompile.cpp): the emulator's
# sources, built like the emulator (the same SPIR-V), with the precompile code of pipelineCache.cpp.
add_kyty_full_emulator_test(kyty_shader_precompile "${KYTY_SOURCE_DIR}/local/shader-precompile.cpp")
target_compile_definitions(kyty_shader_precompile PRIVATE KYTY_STATIC_PRECOMPILE=1)
# --make-seeds: the seed file from the game's files (the port of tools/local/static-precompile/precompile.py
# seeds); in the precompile program only, the emulator does not read the game's shader bundles.
target_sources(kyty_shader_precompile PRIVATE "${KYTY_SOURCE_DIR}/local/static-seeds.h"
	"${KYTY_SOURCE_DIR}/local/static-seeds.cpp")
# --make-hints and --apply-hints: the specialization hints a release ships (zstd-compressed, as .zar games are).
target_sources(kyty_shader_precompile PRIVATE "${KYTY_SOURCE_DIR}/local/shader-hints.h"
	"${KYTY_SOURCE_DIR}/local/shader-hints.cpp")
target_link_libraries(kyty_shader_precompile libzstd_static)
if (WIN32)
	target_link_libraries(kyty_shader_precompile opengl32 setupapi ws2_32)
endif()

add_executable(lane_local_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LaneLocalTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/ShaderCFG.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/ExportOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/ImageOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/MemoryOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/ScalarAluOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/ShaderDecoder.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/VectorAluOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/LaneLocal.cpp"
)
target_link_libraries(lane_local_tests fmt::fmt)
target_include_directories(lane_local_tests PRIVATE ${inc_headers})

add_executable(scalar_provenance_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarProvenanceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/BdaReadPlan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ReadLaneElimination.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/local/jit-arena.cpp"
)
target_link_libraries(scalar_provenance_tests fmt::fmt xbyak::xbyak)
target_include_directories(scalar_provenance_tests PRIVATE ${inc_headers})

add_executable(page_manager_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PageManagerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
)
target_link_libraries(page_manager_tests common)
target_include_directories(page_manager_tests PRIVATE ${inc_headers})

add_executable(archive_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ArchiveFileTests.cpp"
)
target_link_libraries(archive_file_tests common ZArchive::zarchive)
target_include_directories(archive_file_tests PRIVATE ${inc_headers})

add_executable(demons_souls_copy_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DemonsSoulsCopyTests.cpp")
target_link_libraries(demons_souls_copy_tests xbyak::xbyak)
target_include_directories(demons_souls_copy_tests PRIVATE ${inc_headers})

add_executable(demons_souls_idle_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DemonsSoulsIdleTests.cpp")
target_link_libraries(demons_souls_idle_tests xbyak::xbyak)
target_include_directories(demons_souls_idle_tests PRIVATE ${inc_headers})

add_executable(page_protection_table_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PageProtectionTableTests.cpp")
target_link_libraries(page_protection_table_tests fmt::fmt common)
target_include_directories(page_protection_table_tests PRIVATE ${inc_headers})

add_executable(bit_array_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BitArrayTests.cpp"
)
target_include_directories(bit_array_tests PRIVATE ${inc_headers})

add_executable(ime_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImeDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeCommon.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeDialog.cpp"
)
target_include_directories(ime_dialog_tests PRIVATE ${inc_headers})

add_executable(memory_tracker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/MemoryTrackerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/memoryTracker.cpp"
)
target_link_libraries(memory_tracker_tests fmt::fmt common)
target_include_directories(memory_tracker_tests PRIVATE ${inc_headers})

add_executable(shader_vertex_metadata_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderVertexMetadataTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderVertexMetadata.cpp"
)
target_include_directories(shader_vertex_metadata_tests PRIVATE ${inc_headers})

add_executable(resource_materialization_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceMaterializationTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/BdaReadPlan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/local/jit-arena.cpp"
)
target_link_libraries(resource_materialization_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_materialization_tests PRIVATE ${inc_headers})

add_executable(resource_tracking_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/BdaReadPlan.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/local/jit-arena.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/WaterfallDescriptor.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/BindingLayout.cpp"
)
target_link_libraries(resource_tracking_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_tracking_tests PRIVATE ${inc_headers})

add_executable(audio_out2_port_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AudioOut2PortTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libAudio2.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(audio_out2_port_tests common fmt::fmt)
target_include_directories(audio_out2_port_tests PRIVATE ${inc_headers})

add_executable(event_queue_lifetime_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueLifetimeTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_lifetime_tests common fmt::fmt)
target_include_directories(event_queue_lifetime_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(kernel_file_system_tests
	"${KYTY_TESTS_DIR}/KernelFileSystemTests.cpp")
add_kyty_full_emulator_test(pad_input_tests "${KYTY_TESTS_DIR}/PadInputTests.cpp")
add_executable(save_data_capacity_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataCapacityTests.cpp")
target_include_directories(save_data_capacity_tests PRIVATE ${inc_headers})
add_executable(save_data_backup_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataBackupTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/saveDataBackup.cpp")
target_include_directories(save_data_backup_tests PRIVATE ${inc_headers})
add_kyty_full_emulator_test(fiber_migration_tests
	"${KYTY_TESTS_DIR}/FiberMigrationTests.cpp")

add_executable(sync_on_address_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SyncOnAddressTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/syncOnAddress.cpp"
)
target_link_libraries(sync_on_address_tests common)
target_include_directories(sync_on_address_tests PRIVATE ${inc_headers})

add_executable(image_page_table_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImagePageTableTests.cpp"
)
target_link_libraries(image_page_table_tests common fmt::fmt)
target_include_directories(image_page_table_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(shader_recompiler_compute_tests "${KYTY_TESTS_DIR}/ShaderRecompilerComputeTests.cpp")

set(gpu_test_generated_dir "${PROJECT_BINARY_DIR}/gpu_test_shaders")
set(gpu_test_ms_depth_source
	"${gpu_tiler_shader_dir}/gpu_test_ms_depth.comp")
set(gpu_test_ms_depth_spv
	"${gpu_test_generated_dir}/gpu_test_ms_depth.spv")
set(gpu_test_ms_depth_header
	"${gpu_test_generated_dir}/gpu_test_ms_depth_spv.h")
add_custom_command(
	OUTPUT "${gpu_test_ms_depth_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_test_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_test_ms_depth_spv}" "${gpu_test_ms_depth_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_test_ms_depth_spv}
	        -DOUTPUT=${gpu_test_ms_depth_header} -DSYMBOL=GPU_TEST_MS_DEPTH_SPV
	        -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_test_ms_depth_source}"
	VERBATIM
)
target_sources(shader_recompiler_compute_tests PRIVATE
	"${gpu_test_ms_depth_header}")

add_kyty_full_emulator_test(virtual_memory_allocation_tests "${KYTY_TESTS_DIR}/VirtualMemoryAllocationTests.cpp")
target_compile_definitions(virtual_memory_allocation_tests PRIVATE
	KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS=1)

# These tests use exceptions.
if(NOT KYTY_CLANG_CL)
	foreach(kyty_exception_test scalar_provenance_tests resource_tracking_tests
			virtual_memory_allocation_tests)
		target_compile_options(${kyty_exception_test} PRIVATE -fexceptions)
	endforeach()
endif()

if(BUILD_TESTING)
	find_package(Threads REQUIRED)
	add_executable(atrac9_initialization_tests EXCLUDE_FROM_ALL tests/Atrac9InitializationTests.cpp)
	target_link_libraries(atrac9_initialization_tests PRIVATE LibAtrac9 Threads::Threads)
	add_test(NAME atrac9_initialization COMMAND $<TARGET_FILE:atrac9_initialization_tests>)
	add_test(NAME emulator_user_name_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 1234567890123456)
	add_test(NAME emulator_user_name_too_long
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 12345678901234567)
	set_tests_properties(emulator_user_name_too_long PROPERTIES WILL_FAIL TRUE)
	add_test(NAME emulator_present_mode_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Immediate)
	add_test(NAME emulator_present_mode_invalid
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Invalid)
	set_tests_properties(emulator_present_mode_invalid PROPERTIES WILL_FAIL TRUE)
	add_test(NAME ime_dialog COMMAND $<TARGET_FILE:ime_dialog_tests>)
	add_test(NAME shader_cfg COMMAND $<TARGET_FILE:shader_cfg_tests>)
	add_test(NAME scalar_provenance COMMAND $<TARGET_FILE:scalar_provenance_tests>)
	add_test(NAME image_page_table COMMAND $<TARGET_FILE:image_page_table_tests>)
	add_test(NAME memory_tracker COMMAND $<TARGET_FILE:memory_tracker_tests>)
	add_test(NAME page_manager COMMAND $<TARGET_FILE:page_manager_tests>)
	add_test(NAME archive_file COMMAND $<TARGET_FILE:archive_file_tests>)
	add_test(NAME demons_souls_copy COMMAND $<TARGET_FILE:demons_souls_copy_tests>)
	add_test(NAME demons_souls_idle COMMAND $<TARGET_FILE:demons_souls_idle_tests>)
	add_test(NAME page_protection_table COMMAND $<TARGET_FILE:page_protection_table_tests>)
	add_test(NAME bit_array COMMAND $<TARGET_FILE:bit_array_tests>)
	add_test(NAME shader_vertex_metadata COMMAND $<TARGET_FILE:shader_vertex_metadata_tests>)
	add_test(NAME resource_materialization COMMAND $<TARGET_FILE:resource_materialization_tests>)
	add_test(NAME resource_tracking COMMAND $<TARGET_FILE:resource_tracking_tests>)
	add_test(NAME event_queue_lifetime COMMAND $<TARGET_FILE:event_queue_lifetime_tests>)
	add_test(NAME kernel_file_system COMMAND $<TARGET_FILE:kernel_file_system_tests>)
	add_test(NAME pad_input COMMAND $<TARGET_FILE:pad_input_tests>)
	add_test(NAME save_data_capacity COMMAND $<TARGET_FILE:save_data_capacity_tests>)
	add_test(NAME save_data_backup COMMAND $<TARGET_FILE:save_data_backup_tests>)
	add_test(NAME fiber_migration COMMAND $<TARGET_FILE:fiber_migration_tests>)
	add_test(NAME sync_on_address COMMAND $<TARGET_FILE:sync_on_address_tests>)
	add_test(NAME audio_out2_port COMMAND $<TARGET_FILE:audio_out2_port_tests>)
	add_test(NAME shader_recompiler_compute COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	add_test(NAME native_dispatch_indirect
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --native-dispatch-indirect-only)
	add_test(NAME cpu_write_window COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cpu-write-window-only)
	add_test(NAME shader_recompiler_alignbyte
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alignbyte-only)
	add_test(NAME guest_stack_exit
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --guest-stack-exit-only)
	add_test(NAME virtual_memory_allocation
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	add_test(NAME guest_red_zone_patcher
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --red-zone-patcher-only)
	add_test(NAME shader_lod_feedback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --lodstats-gpu-only)
	add_test(NAME demons_souls_compute_chain
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --demons-compute-chain-only)
	add_test(NAME compute_stream_submission
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --streaming-compute-only)
	add_test(NAME command_scheduler_timeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	add_test(NAME stream_buffer_ring
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --stream-buffer-only)
	add_test(NAME gpu_command_lane
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	add_test(NAME pm4_context_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --context-state-only)
	add_test(NAME gpu_tiler
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-tiler-only)

	add_test(NAME texture_resolution_cache
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texture-resolve-cache-only)
	add_test(NAME texture_cache_layered_image
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --layered-image-only)
	add_test(NAME host_image_allocation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --host-image-allocation-only)
	add_test(NAME texture_cache_image_views
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-view-cache-only)
	add_test(NAME texture_cache_storage_sampled
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --storage-sampled-only)
	add_test(NAME texture_cache_depth_readback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-readback-only)
	add_test(NAME compute_meta_clear_classification
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --compute-meta-clear-only)
	add_test(NAME buffer_cache_dirty_gc
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-gc-only)

	if(WIN32)
		# These tests still depend on the Windows multisample-depth path.
		add_test(NAME texture_cache_image_overlap
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-overlap-only)
		add_test(NAME texture_cache_htile_clear
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --htile-clear-only)
		add_test(NAME buffer_cache_ranges
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)
	endif()

	add_custom_target(kyty_tests DEPENDS
		ime_dialog_tests
		shader_cfg_tests
		scalar_provenance_tests
		image_page_table_tests
		memory_tracker_tests
		page_manager_tests
		archive_file_tests
		demons_souls_copy_tests
		demons_souls_idle_tests
		page_protection_table_tests
		atrac9_initialization_tests
		bit_array_tests
		shader_vertex_metadata_tests
		resource_materialization_tests
		resource_tracking_tests
		event_queue_lifetime_tests
		kernel_file_system_tests
		pad_input_tests
		fiber_migration_tests
		save_data_capacity_tests
		save_data_backup_tests
		sync_on_address_tests
		shader_recompiler_compute_tests
		virtual_memory_allocation_tests
		audio_out2_port_tests
	)
endif()


add_executable(kyty_emulator "${KYTY_SOURCE_DIR}/main.cpp" ${kyty_emulator_src})
configure_macos_guest_address_space(kyty_emulator)

if(LINUX)
	set_property(SOURCE
		"${KYTY_SOURCE_DIR}/graphics/guest_gpu/graphicsRun.cpp"
		"${KYTY_SOURCE_DIR}/graphics/guest_gpu/speculation.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/commandScheduler.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/pipelineCache.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/descriptors.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/textureCache.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/tiler.cpp"
		"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
		"${KYTY_SOURCE_DIR}/local/jit-arena.cpp"
		"${KYTY_SOURCE_DIR}/graphics/shader/shader.cpp"
		"${KYTY_SOURCE_DIR}/loader/redZonePatcher.cpp"
		"${KYTY_SOURCE_DIR}/local/local-platform.cpp"
		"${KYTY_SOURCE_DIR}/local/static-seeds.cpp"
		"${KYTY_SOURCE_DIR}/libs/libAudio2.cpp"
		"${KYTY_SOURCE_DIR}/libs/libKernel.cpp"
		"${KYTY_SOURCE_DIR}/kernel/syncOnAddress.cpp"
		APPEND PROPERTY COMPILE_OPTIONS -fexceptions)
endif()

target_link_libraries(kyty_emulator ${kyty_emulator_link_libraries})
if (WIN32)
	target_link_libraries(kyty_emulator opengl32)
	target_link_libraries(kyty_emulator iphlpapi)
	target_link_libraries(kyty_emulator setupapi)
	target_link_libraries(kyty_emulator ws2_32)
endif()
if (KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator onecore)
endif()
if (CLANG AND NOT KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator pthread)
endif()
# dlopen/dlsym/dladdr for RenderDoc.
target_link_libraries(kyty_emulator ${CMAKE_DL_LIBS})
target_include_directories(kyty_emulator PRIVATE ${inc_headers})

# Intel XeSS super resolution (src/local/xess-sr.cpp): the XeSS SDK's headers when they were fetched beside the
# build (_Build/deps/xess, the CI's upscale builds); libxess.dll is loaded at run time.
set(KYTY_XESS_SDK "${CMAKE_CURRENT_SOURCE_DIR}/_Build/deps/xess" CACHE PATH "Intel XeSS SDK (inc/xess/xess_vk.h)")
if (WIN32 AND EXISTS "${KYTY_XESS_SDK}/inc/xess/xess_vk.h")
	set_property(SOURCE "${KYTY_SOURCE_DIR}/local/xess-sr.cpp" APPEND PROPERTY INCLUDE_DIRECTORIES "${KYTY_XESS_SDK}/inc")
	set_property(SOURCE "${KYTY_SOURCE_DIR}/local/xess-sr.cpp" APPEND PROPERTY COMPILE_DEFINITIONS KYTY_XESS=1)
	message(STATUS "XeSS SDK: ${KYTY_XESS_SDK}")
endif()

# Local frame generation experiments (src/local/frame-gen.cpp): the NVIDIA Streamline SDK when it
# was unpacked under _Build/deps (never part of the repository). Its DLLs are loaded at run time.
if (WIN32)
	set(KYTY_STREAMLINE_SDK "${CMAKE_CURRENT_SOURCE_DIR}/_Build/deps/streamline/sdk" CACHE PATH
	    "NVIDIA Streamline SDK for KYTY_FRAMEGEN (local only)")
	if (EXISTS "${KYTY_STREAMLINE_SDK}/include/sl.h")
		set_property(SOURCE "${KYTY_SOURCE_DIR}/local/frame-gen.cpp" APPEND PROPERTY
		             INCLUDE_DIRECTORIES "${KYTY_STREAMLINE_SDK}/include")
		set_property(SOURCE "${KYTY_SOURCE_DIR}/local/frame-gen.cpp" APPEND PROPERTY
		             COMPILE_DEFINITIONS KYTY_FRAMEGEN_STREAMLINE=1)
	endif()
endif()

clang_tidy_check(kyty_emulator "" "${check_headers}" "${inc_headers}")

include_what_you_use(kyty_emulator "${inc_headers}")

set(KYTY_EMULATOR_MAP_NAME "kyty_emulator_${KYTY_COMPILER_ID}_${KYTY_LINKER_ID}.map")

set(KYTY_EMULATOR_MAP_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KYTY_EMULATOR_MAP_NAME}")
set(KYTY_EMULATOR_PDB_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/kyty_emulator.pdb")

if(KYTY_CLANG_CL)
	# /STACK: 16 MiB reserved for each thread that does not ask for a size (committed only as it is used), not
	# Windows' 1 MiB. The driver compiles a pipeline on the thread that creates it: NVIDIA 566.07's compiler ran a
	# wave64 compute shader's cooperative halves (9935d547) past 1 MiB (STATUS_STACK_OVERFLOW in nvoglv64.dll, a
	# precompile shard on an RTX 3080 Laptop) and through it at 16 MiB.
	target_link_options(kyty_emulator PRIVATE
		"/STACK:16777216"
		"/DYNAMICBASE:NO"
		"/DEBUG:FULL"
		"/PDB:${KYTY_EMULATOR_PDB_LINK_PATH}"
		"/lldmap:${KYTY_EMULATOR_MAP_LINK_PATH}"
	)
	add_custom_command(TARGET kyty_emulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:kyty_emulator>/libwinpthread-1.dll)
elseif(WIN32 OR LINUX)
	set_target_properties(kyty_emulator PROPERTIES LINK_FLAGS "${KYTY_LD_OPTIONS} -Wl,-Map=${KYTY_EMULATOR_MAP_LINK_PATH}")
endif()

if(APPLE)
	# The emulator writes x86-64 trampolines/PLT stubs into memory and executes them, and
	# runs guest code under Rosetta. That requires the JIT / unsigned-executable-memory
	# entitlements, so re-sign the binary after every link (an unsigned/relinked binary
	# reverts to the hardened defaults and aborts when it first executes written code).
	add_custom_command(TARGET kyty_emulator POST_BUILD
		COMMAND codesign -s - --force --options runtime
			--entitlements "${KYTY_SOURCE_DIR}/macos_jit.entitlements"
			$<TARGET_FILE:kyty_emulator>
		COMMENT "Codesign kyty_emulator with JIT entitlements (macOS)")
endif()

install(TARGETS kyty_emulator DESTINATION .)

# x86_64 Linux and Windows: the Vulkan recording worker (KYTY_VULKAN_RECORDING, with the
# native XPR draws built on it) and the native resource preparation tests (the
# ahead-of-time SRT library is loaded with dlopen on Linux, LoadLibrary on Windows).
if((LINUX OR WIN32) AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
	find_package(Python3 REQUIRED COMPONENTS Interpreter)
	set(recording_generated "${CMAKE_CURRENT_BINARY_DIR}/local-vulkan-recording.inc")
	set(recording_generator "${KYTY_SOURCE_DIR}/local/generate-vulkan-recording.py")
	set(vulkan_registry "${KYTY_THIRD_PARTY_DIR}/Vulkan-Headers/registry/vk.xml")
	set(vulkan_hpp "${KYTY_THIRD_PARTY_DIR}/Vulkan-Headers/include/vulkan/vulkan.hpp")
	add_custom_command(OUTPUT "${recording_generated}"
		COMMAND "${Python3_EXECUTABLE}" "${recording_generator}" "${vulkan_registry}" "${vulkan_hpp}"
			"${recording_generated}"
		DEPENDS "${recording_generator}" "${vulkan_registry}" "${vulkan_hpp}"
		VERBATIM)
	foreach(recording_target kyty_emulator kyty_shader_precompile shader_recompiler_compute_tests xpr_replay_tests
			virtual_memory_allocation_tests pad_input_tests)
		target_sources(${recording_target} PRIVATE
			"${KYTY_SOURCE_DIR}/local/vulkan-recording.cpp" "${recording_generated}")
		target_include_directories(${recording_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
		target_compile_definitions(${recording_target} PRIVATE KYTY_LOCAL_VULKAN_RECORDING=1)
	endforeach()
	add_executable(vulkan_recording_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/VulkanRecordingTests.cpp"
		"${KYTY_SOURCE_DIR}/local/vulkan-recording.cpp" "${recording_generated}")
	target_include_directories(vulkan_recording_tests PRIVATE ${inc_headers} "${CMAKE_CURRENT_BINARY_DIR}")
	target_link_libraries(vulkan_recording_tests PRIVATE Threads::Threads Vulkan::Headers)

	foreach(native_resource_target kyty_emulator kyty_shader_precompile shader_recompiler_compute_tests shader_cfg_tests
			resource_tracking_tests resource_materialization_tests virtual_memory_allocation_tests pad_input_tests)
		target_sources(${native_resource_target} PRIVATE "${lod_stats_pack_header}")
		target_compile_definitions(${native_resource_target} PRIVATE KYTY_LOCAL_NATIVE_RESOURCES=1)
		target_link_libraries(${native_resource_target} ${CMAKE_DL_LIBS})
	endforeach()
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_LINKER MATCHES "ld.lld$")
		# Clang otherwise selected GNU ld/LLVMgold despite CMAKE_LINKER, which
		# crashed linking these large ThinLTO test executables. LLVM 18's
		# optimized ThinLTO backend also crashes selecting instructions for
		# GetGraphicsPrograms in the GPU fixture: tests use the smaller pass set.
		target_link_options(shader_recompiler_compute_tests PRIVATE "-fuse-ld=${CMAKE_LINKER}" "-Wl,--lto-O1")
		target_link_options(shader_cfg_tests PRIVATE "-fuse-ld=${CMAKE_LINKER}")
	endif()

	if(BUILD_TESTING)
		add_dependencies(kyty_tests vulkan_recording_tests)
		add_test(NAME vulkan_recording COMMAND vulkan_recording_tests)
		add_test(NAME specialization_guard
			COMMAND shader_recompiler_compute_tests --specialization-guard-only)
		add_test(NAME native_pipeline_index
			COMMAND shader_recompiler_compute_tests --native-pipeline-index-only)
		add_test(NAME draw_run_ranges COMMAND shader_recompiler_compute_tests --draw-run-ranges-only)
		add_test(NAME binding_scratch COMMAND shader_recompiler_compute_tests --binding-scratch-only)
		add_test(NAME portable_formats_emission COMMAND shader_recompiler_compute_tests --portable-formats-emission-only)
		add_test(NAME portable_formats COMMAND shader_recompiler_compute_tests --portable-formats-only)
		add_test(NAME shader_upload_ring COMMAND shader_recompiler_compute_tests --shader-upload-ring-only)
		add_test(NAME host_backing_read COMMAND shader_recompiler_compute_tests --host-backing-read-only)
		add_test(NAME buffer_cache_guest_readback
			COMMAND shader_recompiler_compute_tests --guest-readback-only)
		add_test(NAME copy_feedback COMMAND shader_recompiler_compute_tests --copy-feedback-only)
		add_test(NAME write_window_handoff
			COMMAND shader_recompiler_compute_tests --write-window-handoff-only)
		add_test(NAME async_write_readback
			COMMAND shader_recompiler_compute_tests --async-write-readback-only)
	endif()
endif()

if(KYTY_BUILD_LAUNCHER)
	add_subdirectory("${KYTY_SOURCE_DIR}/launcher" "${CMAKE_CURRENT_BINARY_DIR}/launcher")
endif()
if(KYTY_CLANG_CL)
	install(FILES "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" DESTINATION .)
endif()
