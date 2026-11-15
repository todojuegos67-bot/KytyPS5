cmake_minimum_required(VERSION 3.19...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)

foreach(lang IN ITEMS C CXX)
    # An explicitly empty launcher disables caching for this language.
    if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        if(SCCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "sccache enabled for ${lang}: ${SCCACHE_PROGRAM}")
        elseif(CCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "ccache enabled for ${lang}: ${CCACHE_PROGRAM}")
        endif()
    endif()
endforeach()

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
	set(LINUX TRUE)
endif()

if (NOT (WIN32 OR LINUX OR APPLE))
    message(FATAL_ERROR "only Windows, Linux, and macOS builds are supported")
endif()

if(WIN32)
	# Prevent Windows headers from defining min/max macros in all targets.
	add_compile_definitions(NOMINMAX)
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

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
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

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")

# Profile-guided optimization (clang/clang-cl; as in Jetsku's fork), a separate build folder each:
# 1. KYTY_PGO_GENERATE=ON: an instrumented emulator; run it with KYTY_PGO_FILE=<x.profraw>
#    (main.cpp writes the counters every KYTY_PGO_DUMP_SECONDS, default 20).
# 2. llvm-profdata merge -o <x.profdata> <dumps> (the llvm-profdata next to the compiler).
# 3. KYTY_PGO_USE=<x.profdata>: the optimized emulator.
# Both builds must be configured from the same source path: clang-cl names functions in anonymous
# namespaces after a hash of the source file's path. The stored profile (pgo/README.md) is made and
# used from K:/, whichever folder the checkout is in. Only Kyty's own targets (below) get the flags.
option(KYTY_PGO_GENERATE "Instrumented build for profile-guided optimization (clang)" OFF)
set(KYTY_PGO_SOURCE_ROOT "K:/" CACHE PATH "The source path the stored profile belongs to")
set(KYTY_PGO_USE "" CACHE FILEPATH "llvm-profdata output to optimize with (clang; empty: none)")
if(KYTY_PGO_GENERATE AND KYTY_PGO_USE)
	message(FATAL_ERROR "KYTY_PGO_GENERATE and KYTY_PGO_USE are separate builds")
endif()
if((KYTY_PGO_GENERATE OR KYTY_PGO_USE) AND NOT CLANG)
	message(FATAL_ERROR "KYTY_PGO_GENERATE/KYTY_PGO_USE need clang or clang-cl")
endif()
if(KYTY_CLANG_CL AND (KYTY_PGO_GENERATE OR KYTY_PGO_USE) AND NOT CMAKE_SOURCE_DIR STREQUAL KYTY_PGO_SOURCE_ROOT)
	message(WARNING "PGO from ${CMAKE_SOURCE_DIR}, not ${KYTY_PGO_SOURCE_ROOT}: functions in anonymous namespaces will not match the profile")
endif()
if(KYTY_PGO_GENERATE)
	add_compile_options(-fprofile-instr-generate)
	add_compile_definitions(KYTY_PGO_GENERATE=1)
	if(KYTY_CLANG_CL)
		# lld-link is called directly, so the profile runtime has to be named; KYTY_PGO_RUNTIME names
		# it when the compiler has none beside it (the same LLVM major version).
		set(KYTY_PGO_RUNTIME "" CACHE FILEPATH "clang_rt.profile-x86_64.lib (empty: next to the compiler)")
		cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH kyty_llvm_bin)
		file(GLOB kyty_profile_runtime "${kyty_llvm_bin}/../lib/clang/*/lib/windows/clang_rt.profile-x86_64.lib")
		if(KYTY_PGO_RUNTIME)
			set(kyty_profile_runtime "${KYTY_PGO_RUNTIME}")
		endif()
		if(NOT kyty_profile_runtime)
			message(FATAL_ERROR "KYTY_PGO_GENERATE: clang_rt.profile-x86_64.lib not found next to ${CMAKE_CXX_COMPILER}; set KYTY_PGO_RUNTIME")
		endif()
		list(GET kyty_profile_runtime 0 kyty_profile_runtime)
		link_libraries("${kyty_profile_runtime}")
		# The objects ask for clang_rt.profile.lib by its unsuffixed name; the runtime named above is it.
		add_link_options(/nodefaultlib:clang_rt.profile.lib)
	else()
		add_link_options(-fprofile-instr-generate)
	endif()
elseif(KYTY_PGO_USE)
	if(NOT EXISTS "${KYTY_PGO_USE}")
		message(FATAL_ERROR "KYTY_PGO_USE: ${KYTY_PGO_USE} does not exist")
	endif()
	add_compile_options("-fprofile-instr-use=${KYTY_PGO_USE}" -Wno-profile-instr-unprofiled -Wno-backend-plugin)
	# The compiler caches (sccache in CI, ccache here) hash this flag's path and not the file, so a retrained
	# profile has to change the command line, or they hand back objects optimized with the old one.
	file(SHA256 "${KYTY_PGO_USE}" kyty_pgo_hash)
	add_compile_definitions(KYTY_PGO_PROFILE=${kyty_pgo_hash})
	set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${KYTY_PGO_USE}")
endif()

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
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/upscaler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/upscaler/*.h"
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

find_program(KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
set(gpu_tiler_shader_names
	standard256
	standard4
	standard4_3d
	standard64
	standard64_3d
	prt
	prt_3d
	render_target
	depth
	promote_d16
	demote_d16
	color_transform
	color_clear
)
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
foreach(shader_name IN LISTS gpu_tiler_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}_spv.h")
	string(TOUPPER "GPU_TILER_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	list(APPEND gpu_tiler_shader_headers "${shader_header}")
endforeach()

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")

set(mesh_indirect_shader_source "${gpu_tiler_shader_dir}/mesh_indirect_args.comp")
set(mesh_indirect_shader_spv "${gpu_tiler_generated_dir}/mesh_indirect_args.spv")
set(mesh_indirect_shader_header "${gpu_tiler_generated_dir}/mesh_indirect_args_spv.h")
add_custom_command(
	OUTPUT "${mesh_indirect_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${mesh_indirect_shader_spv}" "${mesh_indirect_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${mesh_indirect_shader_spv} -DOUTPUT=${mesh_indirect_shader_header}
	        -DSYMBOL=MESH_INDIRECT_ARGS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${mesh_indirect_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${mesh_indirect_shader_header}")
set(mesh_batch_shader_source "${gpu_tiler_shader_dir}/mesh_indirect_args_batch.comp")
set(mesh_batch_shader_spv "${gpu_tiler_generated_dir}/mesh_indirect_args_batch.spv")
set(mesh_batch_shader_header "${gpu_tiler_generated_dir}/mesh_indirect_args_batch_spv.h")
add_custom_command(
	OUTPUT "${mesh_batch_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${mesh_batch_shader_spv}" "${mesh_batch_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${mesh_batch_shader_spv} -DOUTPUT=${mesh_batch_shader_header}
	        -DSYMBOL=MESH_INDIRECT_ARGS_BATCH_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${mesh_batch_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${mesh_batch_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
function(add_gpu_blit_shader shader_name shader_file)
	set(shader_source "${gpu_tiler_shader_dir}/${shader_file}")
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${ARGN}
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" "${gpu_tiler_shader_dir}/gpu_video_out_hdr10_to_sdr.inc"
		VERBATIM
	)
	set(gpu_blit_shader_headers ${gpu_blit_shader_headers} "${shader_header}" PARENT_SCOPE)
endfunction()
add_gpu_blit_shader(gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
add_gpu_blit_shader(gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
add_gpu_blit_shader(gpu_video_out_overlay gpu_video_out_overlay.frag)
add_gpu_blit_shader(gpu_imgui_hdr gpu_video_out_overlay.frag -DIMGUI_HDR=1)
add_gpu_blit_shader(gpu_pssr_linear gpu_pssr_linear.comp)
add_gpu_blit_shader(gpu_bvh_mirror gpu_bvh_mirror.comp)
add_gpu_blit_shader(gpu_bvh_walk gpu_bvh_mirror.comp -DWALK=1)
# At SPIR-V 1.6 glslang declares a task shader's workgroup size with LocalSizeId, which needs maintenance4.
add_gpu_blit_shader(gpu_mesh_indirect gpu_mesh_indirect.task --target-env spirv1.5)
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})

# FSR 3.1's passes (3rdparty/FidelityFX-SDK), one SPIR-V per permutation Kyty's contexts ask for:
# HDR colour and render-size motion vectors without jitter; inverted depth and the Lanczos LUT vary.
set(fsr3_shader_dir "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/src/backends/vk/shaders/fsr3upscaler")
set(fsr3_gpu_dir "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/include/FidelityFX/gpu")
set(fsr3_generated_dir "${PROJECT_BINARY_DIR}/fsr3_shaders")
file(GLOB_RECURSE fsr3_shader_includes CONFIGURE_DEPENDS "${fsr3_gpu_dir}/*.h")
set(fsr3_headers "")
set(fsr3_includes "")
set(fsr3_table "")
# FfxFsr3UpscalerPass ids and their shaders; 7 is the accumulate pass with sharpening.
foreach(entry "0:prepare_inputs" "1:luma_pyramid" "2:shading_change_pyramid" "3:shading_change"
              "4:prepare_reactivity" "5:luma_instability" "6:accumulate" "7:accumulate" "8:rcas"
              "9:debug_view" "10:autogen_reactive")
	string(REPLACE ":" ";" entry "${entry}")
	list(GET entry 0 pass)
	list(GET entry 1 pass_file)
	set(sharpen 0)
	if(pass EQUAL 7)
		set(sharpen 1)
	endif()
	foreach(inverted 0 1)
		foreach(lut 0 1)
			# FSR3UPSCALER_SHADER_PERMUTATION_*: LUT 1, HDR 2, low-res motion 4, inverted depth 16, sharpening 32.
			math(EXPR bits "${lut} + 2 + 4 + ${inverted} * 16 + ${sharpen} * 32")
			set(symbol "FSR3_${pass}_${bits}")
			set(header "${fsr3_generated_dir}/fsr3_${pass}_${bits}.h")
			set(source "${fsr3_shader_dir}/ffx_fsr3upscaler_${pass_file}_pass.glsl")
			# The defines the SDK's own build passes (CMakeCompileFSR3UpscalerShaders.txt), then the permutation's.
			add_custom_command(
				OUTPUT "${header}"
				COMMAND ${CMAKE_COMMAND} -E make_directory "${fsr3_generated_dir}"
				COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -S comp -e main -Os
				        -DFFX_GLSL=1 -DFFX_GPU=1
				        -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
				        -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
				        -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
				        -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
				        -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
				        -DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE=${lut}
				        -DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT=1
				        -DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS=1
				        -DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS=0
				        -DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH=${inverted}
				        -DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING=${sharpen}
				        "-I${fsr3_gpu_dir}" "-I${fsr3_gpu_dir}/fsr3upscaler" --vn ${symbol}
				        -o "${header}" "${source}"
				DEPENDS "${source}" ${fsr3_shader_includes}
				VERBATIM
			)
			list(APPEND fsr3_headers "${header}")
			string(APPEND fsr3_includes "#include \"fsr3_${pass}_${bits}.h\"\n")
			string(APPEND fsr3_table "\t{${pass}, ${bits}, ${symbol}, sizeof(${symbol})},\n")
		endforeach()
	endforeach()
endforeach()
file(CONFIGURE OUTPUT "${fsr3_generated_dir}/fsr3_permutations.h" @ONLY CONTENT
"#pragma once\n#include <cstddef>\n#include <cstdint>\n${fsr3_includes}
struct Fsr3Permutation {\n\tuint32_t pass;\n\tuint32_t bits;\n\tconst uint32_t* code;\n\tsize_t size;\n};
static const Fsr3Permutation FSR3_PERMUTATIONS[] = {\n${fsr3_table}};\n")
list(APPEND kyty_emulator_src ${fsr3_headers})

# Frame generation's passes (3rdparty/FidelityFX-SDK, FSR 3.1): frame interpolation with
# render-size motion vectors without jitter, inverted depth varying; optical flow with SDR input
# bits (its colour is read through a transfer function at run time).
set(fsr3_fg_headers "")
set(fsr3_fg_includes "")
set(fsr3_fi_table "")
set(fsr3_of_table "")
# FfxFrameInterpolationPass ids and their shaders.
foreach(entry "0:reconstruct_and_dilate" "1:setup" "2:reconstruct_previous_depth" "3:game_motion_vector_field"
              "4:optical_flow_vector_field" "5:disocclusion_mask" "6:-" "7:compute_inpainting_pyramid" "8:inpainting"
              "9:compute_game_vector_field_inpainting_pyramid" "10:debug_view")
	string(REPLACE ":" ";" entry "${entry}")
	list(GET entry 0 pass)
	list(GET entry 1 pass_file)
	if(NOT pass_file STREQUAL "-") # pass 6, interpolation itself, is ffx_frameinterpolation_pass.glsl
		set(source "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/src/backends/vk/shaders/frameinterpolation/ffx_frameinterpolation_${pass_file}_pass.glsl")
	else()
		set(source "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/src/backends/vk/shaders/frameinterpolation/ffx_frameinterpolation_pass.glsl")
	endif()
	foreach(inverted 0 1)
		# FRAMEINTERPOLATION_SHADER_PERMUTATION_*: low-res motion 1, jittered motion 2, inverted depth 4.
		math(EXPR bits "1 + ${inverted} * 4")
		set(symbol "FSR3_FI_${pass}_${bits}")
		set(header "${fsr3_generated_dir}/fsr3_fi_${pass}_${bits}.h")
		# The defines of the SDK's build (CMakeCompileFrameinterpolationShaders.txt), then the permutation's.
		add_custom_command(
			OUTPUT "${header}"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${fsr3_generated_dir}"
			COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -S comp -e main -Os
			        -DFFX_GLSL=1 -DFFX_GPU=1
			        -DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
			        -DFFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
			        -DFFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
			        -DFFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
			        -DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2
			        -DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=1
			        -DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0
			        -DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=${inverted}
			        "-I${fsr3_gpu_dir}" "-I${fsr3_gpu_dir}/frameinterpolation" --vn ${symbol}
			        -o "${header}" "${source}"
			DEPENDS "${source}" ${fsr3_shader_includes}
			VERBATIM
		)
		list(APPEND fsr3_fg_headers "${header}")
		string(APPEND fsr3_fg_includes "#include \"fsr3_fi_${pass}_${bits}.h\"\n")
		string(APPEND fsr3_fi_table "\t{${pass}, ${bits}, ${symbol}, sizeof(${symbol})},\n")
	endforeach()
endforeach()
# FfxOpticalflowPass ids and their shaders.
foreach(entry "0:prepare_luma" "1:compute_luminance_pyramid" "2:generate_scd_histogram" "3:compute_scd_divergence"
              "4:compute_optical_flow_advanced_pass_v5" "5:filter_optical_flow_pass_v5" "6:scale_optical_flow_advanced_pass_v5")
	string(REPLACE ":" ";" entry "${entry}")
	list(GET entry 0 pass)
	list(GET entry 1 pass_file)
	string(REGEX REPLACE "_pass_v5$" "" pass_stem "${pass_file}")
	if(pass_stem STREQUAL pass_file)
		set(source "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/src/backends/vk/shaders/opticalflow/ffx_opticalflow_${pass_file}_pass.glsl")
	else()
		set(source "${KYTY_THIRD_PARTY_DIR}/FidelityFX-SDK/sdk/src/backends/vk/shaders/opticalflow/ffx_opticalflow_${pass_file}.glsl")
	endif()
	set(symbol "FSR3_OF_${pass}_0")
	set(header "${fsr3_generated_dir}/fsr3_of_${pass}_0.h")
	add_custom_command(
		OUTPUT "${header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${fsr3_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -S comp -e main -Os
		        -DFFX_GLSL=1 -DFFX_GPU=1 -DFFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT=0
		        "-I${fsr3_gpu_dir}" "-I${fsr3_gpu_dir}/opticalflow" --vn ${symbol}
		        -o "${header}" "${source}"
		DEPENDS "${source}" ${fsr3_shader_includes}
		VERBATIM
	)
	list(APPEND fsr3_fg_headers "${header}")
	string(APPEND fsr3_fg_includes "#include \"fsr3_of_${pass}_0.h\"\n")
	string(APPEND fsr3_of_table "\t{${pass}, 0, ${symbol}, sizeof(${symbol})},\n")
endforeach()
file(CONFIGURE OUTPUT "${fsr3_generated_dir}/fsr3_fg_permutations.h" @ONLY CONTENT
"#pragma once\n#include \"fsr3_permutations.h\"\n${fsr3_fg_includes}
static const Fsr3Permutation FSR3_FI_PERMUTATIONS[] = {\n${fsr3_fi_table}};
static const Fsr3Permutation FSR3_OF_PERMUTATIONS[] = {\n${fsr3_of_table}};\n")
list(APPEND kyty_emulator_src ${fsr3_fg_headers})

list(APPEND kyty_emulator_src
	"${KYTY_SOURCE_DIR}/emulator.h"
	"${KYTY_SOURCE_DIR}/emulator.cpp"
)
list(REMOVE_DUPLICATES kyty_emulator_src)

if (KYTY_CLANG_CL)
	set_source_files_properties(${kyty_emulator_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

set(kyty_emulator_link_libraries common Vulkan::Headers fidelityfx_fsr3 spirv-tools-opt spirv-tools SDL3::SDL3-static imgui xxhash FFmpeg::ffmpeg fmt::fmt nlohmann_json::nlohmann_json LibAtrac9 kyty_git_version Zydis::Zydis xbyak::xbyak opus OpenXR::openxr_loader libzstd_static)

# Linux system libraries.
if(LINUX)
	find_package(Threads REQUIRED)
	list(APPEND kyty_emulator_link_libraries m ${CMAKE_DL_LIBS} Threads::Threads)
endif()

set(inc_headers
	${KYTY_SOURCE_DIR}
	${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxHash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
	${KYTY_THIRD_PARTY_DIR}/renderdoc
	${KYTY_THIRD_PARTY_DIR}/stb
	${KYTY_THIRD_PARTY_DIR}/ffx-api
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
	if (WIN32)
		target_link_libraries(${target} iphlpapi dxgi)
	endif()
	if (KYTY_CLANG_CL)
		target_link_libraries(${target} onecore)
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

add_executable(scalar_provenance_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarProvenanceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ReadLaneElimination.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(scalar_provenance_tests fmt::fmt xbyak::xbyak)
target_include_directories(scalar_provenance_tests PRIVATE ${inc_headers})

add_executable(build_flags_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BuildFlagsTests.cpp")

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

add_executable(trophy_system_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/TrophySystemTests.cpp"
)
target_link_libraries(trophy_system_tests common Vulkan::Headers)
target_include_directories(trophy_system_tests PRIVATE ${inc_headers})

add_executable(avplayer_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AvPlayerFileTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(avplayer_file_tests common FFmpeg::ffmpeg ZArchive::zarchive)
target_include_directories(avplayer_file_tests PRIVATE ${inc_headers})

add_executable(vdecsw_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/VdecswTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/videoDec2Decoder.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(vdecsw_tests common FFmpeg::ffmpeg)
target_include_directories(vdecsw_tests PRIVATE ${inc_headers})

add_executable(bit_array_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BitArrayTests.cpp"
)
target_include_directories(bit_array_tests PRIVATE ${inc_headers})

add_executable(wolverine_render_step_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/WolverineRenderStepTests.cpp"
)
target_include_directories(wolverine_render_step_tests PRIVATE ${inc_headers})

add_executable(wolverine_physics_step_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/WolverinePhysicsStepTests.cpp"
)
target_include_directories(wolverine_physics_step_tests PRIVATE ${inc_headers})
target_link_libraries(wolverine_physics_step_tests xbyak::xbyak)

add_executable(lru_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LruCacheTests.cpp"
)
target_include_directories(lru_cache_tests PRIVATE ${inc_headers})

add_executable(wolverine_stores_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/WolverineStoresTests.cpp"
)
target_include_directories(wolverine_stores_tests PRIVATE ${inc_headers})
add_executable(group_reuse_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GroupReuseTests.cpp"
)
target_include_directories(group_reuse_tests PRIVATE ${inc_headers})
add_executable(group_notices_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GroupNoticesTests.cpp"
)
target_include_directories(group_notices_tests PRIVATE ${inc_headers})
add_executable(scalar_read_check_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarReadCheckTests.cpp"
)
target_include_directories(scalar_read_check_tests PRIVATE ${inc_headers})

add_executable(hash_memo_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/HashMemoTests.cpp"
)
target_include_directories(hash_memo_tests PRIVATE ${inc_headers})

add_executable(packet_stream_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PacketStreamTests.cpp"
)
target_include_directories(packet_stream_tests PRIVATE ${inc_headers})

add_executable(stage_store_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/StageStoreTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/stageStore.cpp"
)
target_include_directories(stage_store_tests PRIVATE ${inc_headers})
target_link_libraries(stage_store_tests xxhash libzstd_static)

add_executable(pipeline_ahead_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PipelineAheadTests.cpp"
)
target_include_directories(pipeline_ahead_tests PRIVATE ${inc_headers})
add_executable(slot_vector_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SlotVectorTests.cpp"
)
target_include_directories(slot_vector_tests PRIVATE ${inc_headers})

add_executable(common_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CommonTests.cpp")
target_link_libraries(common_tests common kyty_test_gtest)

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
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(resource_materialization_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_materialization_tests PRIVATE ${inc_headers})

add_executable(resource_tracking_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
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

add_executable(audio_out_timing_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AudioOutTimingTests.cpp")
target_link_libraries(audio_out_timing_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(audio_out_timing_tests PRIVATE ${inc_headers})

add_executable(ngs2_sampler_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/Ngs2SamplerTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ngs2_sampler_tests common fmt::fmt LibAtrac9)
target_include_directories(ngs2_sampler_tests PRIVATE ${inc_headers})

add_executable(ajm_resample_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AjmResampleTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ajm_resample_tests common fmt::fmt FFmpeg::ffmpeg LibAtrac9 opus)
target_include_directories(ajm_resample_tests PRIVATE ${inc_headers})

add_executable(pad_haptics_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PadHapticsTests.cpp"
)
target_link_libraries(pad_haptics_tests common fmt::fmt SDL3::SDL3-static opus)
target_include_directories(pad_haptics_tests PRIVATE ${inc_headers})

add_executable(controller_settings_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ControllerSettingsTests.cpp"
)
target_link_libraries(controller_settings_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(controller_settings_tests PRIVATE ${inc_headers})

add_executable(host_input_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/HostInputTests.cpp")
target_link_libraries(host_input_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(host_input_tests PRIVATE ${inc_headers})

add_executable(save_data_memory_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataMemoryTests.cpp"
)
target_link_libraries(save_data_memory_tests common fmt::fmt)
target_include_directories(save_data_memory_tests PRIVATE ${inc_headers})

add_executable(png_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PngEncTests.cpp")
target_link_libraries(png_enc_tests common fmt::fmt)
target_include_directories(png_enc_tests PRIVATE ${inc_headers})

add_executable(jpeg_dec_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/JpegDecTests.cpp")
target_link_libraries(jpeg_dec_tests common fmt::fmt)
target_include_directories(jpeg_dec_tests PRIVATE ${inc_headers})

add_executable(jpeg_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/JpegEncTests.cpp")
target_link_libraries(jpeg_enc_tests common fmt::fmt)
target_include_directories(jpeg_enc_tests PRIVATE ${inc_headers})

add_executable(occlusion_queries_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/OcclusionQueriesTests.cpp")
target_link_libraries(occlusion_queries_tests common fmt::fmt Vulkan::Headers)
target_include_directories(occlusion_queries_tests PRIVATE ${inc_headers})

add_executable(gpl_per_vertex_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/GplPerVertexTests.cpp")
target_link_libraries(gpl_per_vertex_tests Vulkan::Headers ${CMAKE_DL_LIBS})

add_executable(ces_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CesTests.cpp")
target_link_libraries(ces_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(ces_tests PRIVATE ${inc_headers})

add_executable(save_data_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/dialog.cpp"
)
target_link_libraries(save_data_dialog_tests common fmt::fmt)
target_include_directories(save_data_dialog_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(http_uri_parse_tests
	"${KYTY_TESTS_DIR}/HttpUriParseTests.cpp")

add_kyty_full_emulator_test(shader_precompile_tests
	"${KYTY_TESTS_DIR}/ShaderPrecompileTests.cpp")
add_executable(rtc_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/RtcTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libRtc.cpp"
	"${KYTY_SOURCE_DIR}/loader/symbolDatabase.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp")
target_link_libraries(rtc_tests common kyty_test_gtest)
target_include_directories(rtc_tests PRIVATE ${inc_headers})

add_executable(event_queue_lifetime_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueLifetimeTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_lifetime_tests common fmt::fmt)
target_include_directories(event_queue_lifetime_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(kernel_file_system_tests
	"${KYTY_TESTS_DIR}/KernelFileSystemTests.cpp")
target_link_libraries(kernel_file_system_tests ZArchive::zarchive)

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
foreach(test_shader gpu_test_ms_depth gpu_test_ray_query)
	set(test_shader_spv "${gpu_test_generated_dir}/${test_shader}.spv")
	set(test_shader_header "${gpu_test_generated_dir}/${test_shader}_spv.h")
	string(TOUPPER "${test_shader}_SPV" test_shader_symbol)
	add_custom_command(
		OUTPUT "${test_shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_test_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        -o "${test_shader_spv}" "${gpu_tiler_shader_dir}/${test_shader}.comp"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${test_shader_spv}
		        -DOUTPUT=${test_shader_header} -DSYMBOL=${test_shader_symbol}
		        -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${gpu_tiler_shader_dir}/${test_shader}.comp"
		VERBATIM
	)
	target_sources(shader_recompiler_compute_tests PRIVATE "${test_shader_header}")
endforeach()

add_kyty_full_emulator_test(virtual_memory_allocation_tests "${KYTY_TESTS_DIR}/VirtualMemoryAllocationTests.cpp")
add_library(kyty_test_gtest STATIC EXCLUDE_FROM_ALL
	"${KYTY_THIRD_PARTY_DIR}/fmt/test/gtest/gmock-gtest-all.cc")
target_include_directories(kyty_test_gtest PUBLIC "${KYTY_THIRD_PARTY_DIR}/fmt/test/gtest")
target_sources(virtual_memory_allocation_tests PRIVATE "${KYTY_TESTS_DIR}/TrinityTests.cpp")
target_link_libraries(virtual_memory_allocation_tests kyty_test_gtest)

add_kyty_full_emulator_test(net_resolver_tests "${KYTY_TESTS_DIR}/NetResolverTests.cpp")
target_link_libraries(net_resolver_tests kyty_test_gtest)

add_executable(dlss_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DlssTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/upscaler/dlss.cpp")
target_link_libraries(dlss_tests common Vulkan::Headers SDL3::SDL3-static kyty_test_gtest)
target_include_directories(dlss_tests PRIVATE ${inc_headers})

add_executable(fsr3_blob_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/Fsr3BlobTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/fsr3Blobs.cpp"
	${fsr3_headers} ${fsr3_fg_headers})
target_link_libraries(fsr3_blob_tests fidelityfx_fsr3 kyty_test_gtest)
target_include_directories(fsr3_blob_tests PRIVATE ${inc_headers})
if (WIN32)
	add_executable(fsr_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/FsrTests.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/upscaler/dlss.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/upscaler/fsr.cpp")
	target_link_libraries(fsr_tests common Vulkan::Headers SDL3::SDL3-static kyty_test_gtest)
	target_include_directories(fsr_tests PRIVATE ${inc_headers})
endif()

add_executable(psml_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PsmlTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libPsml.cpp"
	"${KYTY_SOURCE_DIR}/loader/symbolDatabase.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp")
target_link_libraries(psml_tests common kyty_test_gtest)
target_include_directories(psml_tests PRIVATE ${inc_headers})

add_executable(gpu_blit_shader_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GpuBlitShaderTests.cpp"
	"${gpu_blit_generated_dir}/gpu_mesh_indirect_spv.h")
target_link_libraries(gpu_blit_shader_tests kyty_test_gtest)
target_include_directories(gpu_blit_shader_tests PRIVATE ${inc_headers})
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
	add_test(NAME build_flags COMMAND $<TARGET_FILE:build_flags_tests>)
	add_test(NAME archive_file COMMAND $<TARGET_FILE:archive_file_tests>)
	add_test(NAME trophy_system COMMAND $<TARGET_FILE:trophy_system_tests>)
	add_test(NAME avplayer_file COMMAND $<TARGET_FILE:avplayer_file_tests>)
	add_test(NAME vdecsw COMMAND $<TARGET_FILE:vdecsw_tests>)
	add_test(NAME bit_array COMMAND $<TARGET_FILE:bit_array_tests>)
	add_test(NAME lru_cache COMMAND $<TARGET_FILE:lru_cache_tests>)
	add_test(NAME wolverine_render_step COMMAND $<TARGET_FILE:wolverine_render_step_tests>)
	add_test(NAME wolverine_physics_step COMMAND $<TARGET_FILE:wolverine_physics_step_tests>)
	add_test(NAME group_reuse COMMAND $<TARGET_FILE:group_reuse_tests>)
	add_test(NAME wolverine_stores COMMAND $<TARGET_FILE:wolverine_stores_tests>)
	add_test(NAME group_notices COMMAND $<TARGET_FILE:group_notices_tests>)
	add_test(NAME scalar_read_check COMMAND $<TARGET_FILE:scalar_read_check_tests>)
	add_test(NAME hash_memo COMMAND $<TARGET_FILE:hash_memo_tests>)
	add_test(NAME packet_stream COMMAND $<TARGET_FILE:packet_stream_tests>)
	add_test(NAME stage_store COMMAND $<TARGET_FILE:stage_store_tests>)
	add_test(NAME pipeline_ahead COMMAND $<TARGET_FILE:pipeline_ahead_tests>)
	add_test(NAME common COMMAND $<TARGET_FILE:common_tests>)
	add_test(NAME shader_vertex_metadata COMMAND $<TARGET_FILE:shader_vertex_metadata_tests>)
	add_test(NAME resource_materialization COMMAND $<TARGET_FILE:resource_materialization_tests>)
	add_test(NAME resource_tracking COMMAND $<TARGET_FILE:resource_tracking_tests>)
	add_test(NAME event_queue_lifetime COMMAND $<TARGET_FILE:event_queue_lifetime_tests>)
	add_test(NAME kernel_file_system COMMAND $<TARGET_FILE:kernel_file_system_tests>)
	add_test(NAME kernel_fs_core COMMAND $<TARGET_FILE:kernel_file_system_tests> --core-cases)
	add_test(NAME sync_on_address COMMAND $<TARGET_FILE:sync_on_address_tests>)
	add_test(NAME audio_out2_port COMMAND $<TARGET_FILE:audio_out2_port_tests>)
	add_test(NAME audio_out_timing COMMAND $<TARGET_FILE:audio_out_timing_tests>)
	add_test(NAME ngs2_sampler COMMAND $<TARGET_FILE:ngs2_sampler_tests>)
	add_test(NAME ajm_resample COMMAND $<TARGET_FILE:ajm_resample_tests>)
	add_test(NAME pad_haptics COMMAND $<TARGET_FILE:pad_haptics_tests>)
	add_test(NAME controller_settings COMMAND $<TARGET_FILE:controller_settings_tests>)
	add_test(NAME host_input COMMAND $<TARGET_FILE:host_input_tests>)
	add_test(NAME save_data_memory COMMAND $<TARGET_FILE:save_data_memory_tests>)
	add_test(NAME png_enc COMMAND $<TARGET_FILE:png_enc_tests>)
	add_test(NAME jpeg_dec COMMAND $<TARGET_FILE:jpeg_dec_tests>)
	add_test(NAME jpeg_enc COMMAND $<TARGET_FILE:jpeg_enc_tests>)
	add_test(NAME occlusion_queries COMMAND $<TARGET_FILE:occlusion_queries_tests>)
	add_test(NAME gpl_per_vertex COMMAND $<TARGET_FILE:gpl_per_vertex_tests>)
	add_test(NAME ces COMMAND $<TARGET_FILE:ces_tests>)
	add_test(NAME save_data_dialog COMMAND $<TARGET_FILE:save_data_dialog_tests>)
	add_test(NAME http_uri_parse COMMAND $<TARGET_FILE:http_uri_parse_tests>)
	add_test(NAME shader_precompile COMMAND $<TARGET_FILE:shader_precompile_tests>)
	add_test(NAME precompile_env_guard
		COMMAND ${CMAKE_COMMAND} -DSRC=${KYTY_SOURCE_DIR} -P "${KYTY_TESTS_DIR}/PrecompileEnvGuard.cmake")
	add_test(NAME rtc COMMAND $<TARGET_FILE:rtc_tests>)
	add_test(NAME shader_recompiler_compute COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	add_test(NAME shader_recompiler_tessellation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --tessellation-only)
	add_test(NAME shader_recompiler_float_min_max
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --float-min-max-only)
	add_test(NAME shader_recompiler_relaxed_float
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --relaxed-float-only)
	add_test(NAME shader_recompiler_vop3_conversions
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --vop3-conversions-only)
	add_test(NAME shader_recompiler_quadmask
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --quadmask-only)
	add_test(NAME shader_recompiler_integer16_ternary
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --integer16-ternary-only)
	add_test(NAME shader_recompiler_mad_u16
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mad-u16-only)
	add_test(NAME shader_recompiler_mul_hi24
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mul-hi24-only)
	add_test(NAME shader_recompiler_alignbyte
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alignbyte-only)
	add_test(NAME graphics_pipeline_rasterization
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --polygon-mode-only)
	add_test(NAME graphics_draw_offsets
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-offset-only)
	add_test(NAME virtual_memory_allocation
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	add_test(NAME trinity_mode
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --trinity-only)
	add_test(NAME psml COMMAND $<TARGET_FILE:psml_tests>)
	add_test(NAME fsr3_blobs COMMAND $<TARGET_FILE:fsr3_blob_tests>)
	add_test(NAME gpu_blit_shaders COMMAND $<TARGET_FILE:gpu_blit_shader_tests>)
	add_test(NAME pssr COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --pssr-only)
	add_test(NAME guest_red_zone_patcher
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --red-zone-patcher-only)
	add_test(NAME command_scheduler_timeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	add_test(NAME stream_buffer_ring
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --stream-buffer-only)
	add_test(NAME gpu_command_lane
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	add_test(NAME pm4_flip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --flip-only)
	add_test(NAME device_creation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --device-creation-only)
	# The same with lean's own device creation instead of upstream 8810d3869's structure chains (the default).
	add_test(NAME device_creation_lean
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --device-creation-only)
	set_tests_properties(device_creation_lean PROPERTIES ENVIRONMENT "KYTY_LEAN_PATHS=8810d3869")
	add_test(NAME video_out_flip_rate
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --flip-rate-only)
	add_test(NAME video_out_hdr10_to_sdr
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --video-out-only)
	add_test(NAME depth_bits
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-bits-only)
	add_test(NAME srgb_texture
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --srgb-texture-only)
	add_test(NAME msaa_storage
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --msaa-storage-only)
	add_test(NAME pm4_context_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --context-state-only)
	add_test(NAME gpu_tiler
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-tiler-only)
	add_test(NAME image_transfer_layout
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-layout-only)

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
	add_test(NAME indirect_write_tables
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --indirect-write-tables-only)
	add_test(NAME color_render_target_1d
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --rt1d-only)

	add_test(NAME texture_cache_image_overlap
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-overlap-only)
	add_test(NAME texture_cache_htile_clear
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --htile-clear-only)
	add_test(NAME buffer_cache_ranges
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)

	add_custom_target(kyty_tests DEPENDS
		ime_dialog_tests
		shader_cfg_tests
		scalar_provenance_tests
		image_page_table_tests
		memory_tracker_tests
		page_manager_tests
		build_flags_tests
		archive_file_tests
		trophy_system_tests
		avplayer_file_tests
		vdecsw_tests
		bit_array_tests
		wolverine_render_step_tests
		wolverine_physics_step_tests
		lru_cache_tests
		group_reuse_tests
		wolverine_stores_tests
		group_notices_tests
		scalar_read_check_tests
		hash_memo_tests
		packet_stream_tests
		stage_store_tests
		pipeline_ahead_tests
		common_tests
		shader_vertex_metadata_tests
		resource_materialization_tests
		resource_tracking_tests
		event_queue_lifetime_tests
		kernel_file_system_tests
		sync_on_address_tests
		shader_recompiler_compute_tests
		virtual_memory_allocation_tests
		audio_out2_port_tests
		audio_out_timing_tests
		ngs2_sampler_tests
		ajm_resample_tests
		psml_tests
		fsr3_blob_tests
		gpu_blit_shader_tests
		pad_haptics_tests
		controller_settings_tests
		host_input_tests
		save_data_memory_tests
		png_enc_tests
		jpeg_dec_tests
		jpeg_enc_tests
		occlusion_queries_tests
		gpl_per_vertex_tests
		ces_tests
		save_data_dialog_tests
		http_uri_parse_tests
		shader_precompile_tests
		rtc_tests
	)
endif()

# Test Darwin signal-context restoration for SSE4a instructions.
if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
		(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
	add_executable(macos_sse4a_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosSse4aTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(macos_sse4a_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(macos_sse4a_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME macos_sse4a COMMAND $<TARGET_FILE:macos_sse4a_tests>)
		set_tests_properties(macos_sse4a PROPERTIES TIMEOUT 15)
		add_dependencies(kyty_tests macos_sse4a_tests)
	endif()
endif()

add_executable(kyty_emulator "${KYTY_SOURCE_DIR}/main.cpp" ${kyty_emulator_src})
configure_macos_guest_address_space(kyty_emulator)

set(KYTY_TROPHY_UNLOCK_SOUND "${CMAKE_CURRENT_SOURCE_DIR}/assets/sounds/trophy-unlock.wav")
add_custom_target(kyty_trophy_sound
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_TROPHY_UNLOCK_SOUND}"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds/trophy-unlock.wav"
	VERBATIM)
add_dependencies(kyty_emulator kyty_trophy_sound)
install(FILES "${KYTY_TROPHY_UNLOCK_SOUND}" DESTINATION assets/sounds)

add_custom_target(kyty_touch_cursors
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/cursors"
	VERBATIM)
add_dependencies(kyty_emulator kyty_touch_cursors)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors" DESTINATION assets)

set(KYTY_NOTO_SANS_FONTS
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Regular.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Bold.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-Italic.ttf"
	"${KYTY_NOTO_SANS_DIR}/NotoSans-BoldItalic.ttf")
add_custom_target(kyty_noto_sans_fonts
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kyty_emulator>/assets/fonts"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${KYTY_NOTO_SANS_FONTS}
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/fonts"
	VERBATIM)
add_dependencies(kyty_emulator kyty_noto_sans_fonts)
install(FILES ${KYTY_NOTO_SANS_FONTS} DESTINATION assets/fonts)

target_link_libraries(kyty_emulator ${kyty_emulator_link_libraries})
if (WIN32)
	target_link_libraries(kyty_emulator opengl32)
	target_link_libraries(kyty_emulator iphlpapi)
	target_link_libraries(kyty_emulator setupapi)
	target_link_libraries(kyty_emulator ws2_32)
	target_link_libraries(kyty_emulator dxgi)
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

clang_tidy_check(kyty_emulator "" "${check_headers}" "${inc_headers}")

include_what_you_use(kyty_emulator "${inc_headers}")

set(KYTY_EMULATOR_MAP_NAME "kyty_emulator_${KYTY_COMPILER_ID}_${KYTY_LINKER_ID}.map")

set(KYTY_EMULATOR_MAP_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KYTY_EMULATOR_MAP_NAME}")
set(KYTY_EMULATOR_PDB_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/kyty_emulator.pdb")

if(KYTY_CLANG_CL)
	# 16 MiB stacks (reserved, committed as used): the driver compiles a pipeline on the thread that
	# creates it, and NVIDIA 566.07 overflowed Windows' default 1 MiB on a big compute shader.
	target_link_options(kyty_emulator PRIVATE
		"/STACK:16777216"
		"/DYNAMICBASE:NO"
		"/DEBUG:FULL"
		"/PDB:${KYTY_EMULATOR_PDB_LINK_PATH}"
		"/lldmap:${KYTY_EMULATOR_MAP_LINK_PATH}"
	)
	add_custom_command(TARGET kyty_emulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:kyty_emulator>/libwinpthread-1.dll)
elseif(WIN32 OR LINUX)
	set_target_properties(kyty_emulator PROPERTIES LINK_FLAGS "${KYTY_LD_OPTIONS}")
	# LINK_FLAGS is a space-delimited string: a -Map path containing spaces would
	# be split into bogus arguments, so pass it as a discrete link option instead.
	target_link_options(kyty_emulator PRIVATE "-Wl,-Map=${KYTY_EMULATOR_MAP_LINK_PATH}")
endif()

if(LINUX)
    target_link_options(kyty_emulator PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
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
if(KYTY_BUILD_LAUNCHER)
	add_subdirectory("${KYTY_SOURCE_DIR}/launcher" "${CMAKE_CURRENT_BINARY_DIR}/launcher")
	# Marvel's Wolverine's own start window with the shader precompile (session 12); the launcher above is upstream's.
	add_subdirectory("${KYTY_SOURCE_DIR}/wolverine_launcher" "${CMAKE_CURRENT_BINARY_DIR}/wolverine_launcher")

	if(LINUX)
		target_link_options(launcher PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
		target_link_options(kyty_updater PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
	endif()
endif()
if(KYTY_CLANG_CL)
	install(FILES "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" DESTINATION .)
endif()
