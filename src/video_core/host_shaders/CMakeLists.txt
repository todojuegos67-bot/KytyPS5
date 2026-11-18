# SPDX-FileCopyrightText: Copyright 2023 Citra Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

if (TARGET glslang-standalone)
    set(GLSLANG_COMPILER glslang-standalone)
elseif (TARGET glslang::glslang-standalone)
    set(GLSLANG_COMPILER glslang::glslang-standalone)
endif()

set(SHADER_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include)
set(SHADER_DIR ${SHADER_INCLUDE}/video_core/host_shaders)
set(HOST_SHADERS_INCLUDE ${SHADER_INCLUDE} PARENT_SCOPE)

set(SHADER_FILES "")

function(add_host_shader FILENAME SHADER_NAME)
    set(SOURCE_HEADER_FILE ${SHADER_DIR}/${SHADER_NAME}.h)
    set(DEPFILE ${SOURCE_HEADER_FILE}.d)
    string(TOUPPER ${SHADER_NAME} SYMBOL)
    set(SHADER_DEFINES ${ARGN})
    list(TRANSFORM SHADER_DEFINES PREPEND "-D")
    add_custom_command(
        OUTPUT ${SOURCE_HEADER_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DIR}
        COMMAND ${GLSLANG_COMPILER} --target-env vulkan1.3
                -o ${SOURCE_HEADER_FILE} --vn ${SYMBOL} --depfile ${DEPFILE}
                ${SHADER_DEFINES} ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${SOURCE_HEADER_FILE}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/AddShaderHeaderInclude.cmake
        DEPENDS ${GLSLANG_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                ${CMAKE_CURRENT_SOURCE_DIR}/AddShaderHeaderInclude.cmake
        DEPFILE ${DEPFILE}
        BYPRODUCTS ${DEPFILE}
        COMMENT "Building host shader ${FILENAME} -> ${SOURCE_HEADER_FILE}"
        VERBATIM
    )
    set(SHADER_HEADERS ${SHADER_HEADERS} ${SOURCE_HEADER_FILE} PARENT_SCOPE)
    set(SHADER_FILES ${SHADER_FILES} ${FILENAME} PARENT_SCOPE)
endfunction()

# Path tracing (rt/vk_ray_tracing.cpp): ray tracing pipeline stages from one source.
function(add_host_shader_stage FILENAME SHADER_NAME STAGE)
    set(SOURCE_HEADER_FILE ${SHADER_DIR}/${SHADER_NAME}.h)
    set(DEPFILE ${SOURCE_HEADER_FILE}.d)
    string(TOUPPER ${SHADER_NAME} SYMBOL)
    set(SHADER_DEFINES ${ARGN})
    list(TRANSFORM SHADER_DEFINES PREPEND "-D")
    add_custom_command(
        OUTPUT ${SOURCE_HEADER_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_DIR}
        COMMAND ${GLSLANG_COMPILER} --target-env vulkan1.3 -S ${STAGE}
                -o ${SOURCE_HEADER_FILE} --vn ${SYMBOL} --depfile ${DEPFILE}
                ${SHADER_DEFINES} ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
        COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${SOURCE_HEADER_FILE}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/AddShaderHeaderInclude.cmake
        DEPENDS ${GLSLANG_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                ${CMAKE_CURRENT_SOURCE_DIR}/AddShaderHeaderInclude.cmake
        DEPFILE ${DEPFILE}
        BYPRODUCTS ${DEPFILE}
        COMMENT "Building host shader ${FILENAME} (${STAGE}) -> ${SOURCE_HEADER_FILE}"
        VERBATIM
    )
    set(SHADER_HEADERS ${SHADER_HEADERS} ${SOURCE_HEADER_FILE} PARENT_SCOPE)
    set(SHADER_FILES ${SHADER_FILES} ${FILENAME} PARENT_SCOPE)
endfunction()

add_host_shader(color_to_ms_depth.frag color_to_ms_depth_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_frag)
add_host_shader(ms_image_blit.frag ms_image_blit_msaa_frag SRC_MSAA=1)
add_host_shader(fault_buffer_process.comp fault_buffer_process_comp)
add_host_shader(fs_tri.vert fs_tri_vert)
add_host_shader(fsr.comp fsr_easu_comp SAMPLE_EASU=1)
add_host_shader(fsr.comp fsr_rcas_comp SAMPLE_RCAS=1)
add_host_shader(post_process.frag post_process_frag)
add_host_shader(bb_mv.comp bb_mv_comp)
add_host_shader(bb_hud_mask.comp bb_hud_mask_comp)
add_host_shader(bb_hud_over.comp bb_hud_over_comp)
add_host_shader(bb_hudless_fix.comp bb_hudless_fix_comp)
add_host_shader(bb_hud_layer.comp bb_hud_layer_comp)
add_host_shader(bb_objmv2.comp bb_objmv2_comp)
add_host_shader(bb_dlss5.comp bb_dlss5_comp)
add_host_shader(bb_taa.comp bb_taa_comp)
add_host_shader(bb_taa_sharpen.comp bb_taa_sharpen_comp)
add_host_shader(bb_fsr4_prep.comp bb_fsr4_prep_comp)
add_host_shader(bb_rt_mesh.comp bb_rt_mesh_comp)
add_host_shader(bb_rt_reflect.comp bb_rt_reflect_comp)
add_host_shader(bb_rt_post.comp bb_rt_post_comp)
add_host_shader(bb_rt_vol.comp bb_rt_vol_comp)
add_host_shader(bb_rt_shadow.comp bb_rt_shadow_comp)
add_host_shader(bb_rt_depthcvt.comp bb_rt_depthcvt_comp)

# Path tracing mod by Joao Vitor Alves (GPL-2.0): the same variants as its gpu/CMakeLists.txt.
if (ENABLE_PATHTRACING)
    add_host_shader(rt_gather.comp rt_gather_comp)
    add_host_shader(rt_omm_bake.comp rt_omm_bake_comp)
    add_host_shader(rt_debug.comp rt_debug_comp)
    add_host_shader(rt_probe.comp rt_probe_comp)
    add_host_shader(rt_gi.comp rt_gi_comp)
    add_host_shader(rt_gi_apply.comp rt_gi_apply_comp)
    add_host_shader(rt_sharc.comp rt_sharc_comp)
    add_host_shader(rt_gi.comp rt_reflect_comp RT_REFLECT=1)
    add_host_shader(rt_gi.comp rt_restir_gi_init_comp RT_RESTIR_GI=1 RT_RESTIR_INIT=1)
    add_host_shader(rt_gi.comp rt_restir_gi_resolve_comp RT_RESTIR_GI=1 RT_RESTIR_RESOLVE=1)
    add_host_shader_stage(rt_gi.comp rt_restir_gi_init_rgen rgen RT_RESTIR_GI=1 RT_RESTIR_INIT=1 RT_GI_RGEN=1)
    add_host_shader_stage(rt_gi.comp rt_restir_gi_init_ser_rgen rgen RT_RESTIR_GI=1 RT_RESTIR_INIT=1 RT_GI_RGEN=1 RT_GI_SER=1)
    add_host_shader_stage(rt_gi.comp rt_gi_ahit rahit RT_GI_AHIT=1)
    add_host_shader_stage(rt_gi.comp rt_gi_chit rchit RT_GI_CHIT=1)
    add_host_shader_stage(rt_gi.comp rt_gi_miss rmiss RT_GI_MISS=1)
    add_host_shader(rt_gi.comp rt_light_vis_comp RT_LIGHT_VIS=1)
    add_host_shader(rt_gi.comp rt_sky_layer_comp RT_SKY_LAYER=1)
    add_host_shader(rt_restir_gi.comp rt_restir_gi_temporal_comp)
    add_host_shader(rt_restir_gi.comp rt_restir_gi_spatial_comp RT_RESTIR_SPATIAL=1)
    add_host_shader(rt_shadow.comp rt_shadow_comp)
    add_host_shader(rt_fog.comp rt_fog_comp)
    add_host_shader(rt_fog.comp rt_fog_apply_comp RT_FOG_APPLY=1)
    add_host_shader(rt_fog.comp rt_fog_opaque_comp RT_FOG_OPAQUE=1)
    add_host_shader(rt_sky.comp rt_sky_comp)
    add_host_shader(rt_exposure.comp rt_exposure_comp)
    add_host_shader(rt_world_set.comp rt_world_set_comp)
    add_host_shader(rt_moon_dir.comp rt_moon_dir_comp)
    add_host_shader(rt_direct.comp rt_direct_comp)
    add_host_shader(rt_ao.comp rt_ao_comp)
    add_host_shader(rt_direct.comp rt_restir_temporal_comp RT_RESTIR_PASS=1)
    add_host_shader(rt_direct.comp rt_restir_spatial_comp RT_RESTIR_PASS=2)
    add_host_shader(dlss_rr_guides.comp dlss_rr_guides_comp)
    add_host_shader(dlss_rr_tonemap.comp dlss_rr_linearize_comp)
    add_host_shader(dlss_rr_tonemap.comp dlss_rr_tonemap_comp RR_TONEMAP_FORWARD=1)
endif()

foreach(BPP IN ITEMS 8 16 32 64 96 128)
    add_host_shader(tiling.comp tiling_micro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=0)
    add_host_shader(tiling.comp tiling_macro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=1)
endforeach()

list(REMOVE_DUPLICATES SHADER_FILES)
add_custom_target(host_shaders DEPENDS ${SHADER_HEADERS} SOURCES ${SHADER_FILES})
