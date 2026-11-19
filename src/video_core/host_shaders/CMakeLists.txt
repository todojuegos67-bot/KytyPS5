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
add_host_shader(bb_rcas.comp bb_rcas_comp)
add_host_shader(bb_fsr4_prep.comp bb_fsr4_prep_comp)
add_host_shader(bb_rt_mesh.comp bb_rt_mesh_comp)
add_host_shader(bb_rt_reflect.comp bb_rt_reflect_comp)
add_host_shader(bb_rt_post.comp bb_rt_post_comp)
add_host_shader(bb_rt_post.comp bb_rt_post_r11_comp HDR_R11=1)
add_host_shader(bb_rt_vol.comp bb_rt_vol_comp)
add_host_shader(bb_rt_shadow.comp bb_rt_shadow_comp)
add_host_shader(bb_rt_depthcvt.comp bb_rt_depthcvt_comp)
add_host_shader(bb_rt_gsnap.comp bb_rt_gsnap_comp)
add_host_shader(bb_volfog.comp bb_volfog_inject_comp VF_PASS=0)
add_host_shader(bb_volfog.comp bb_volfog_light_comp VF_PASS=1)
add_host_shader(bb_volfog.comp bb_volfog_integrate_comp VF_PASS=2)
add_host_shader(bb_volfog.comp bb_volfog_apply_comp VF_PASS=3)
add_host_shader(bb_volfog.comp bb_volfog_apply_r11_comp VF_PASS=3 HDR_R11=1)
add_host_shader(bb_volfog.comp bb_volfog_measure_comp VF_PASS=4)
add_host_shader(bb_volfog.comp bb_volfog_measure_r11_comp VF_PASS=4 HDR_R11=1)
add_host_shader(bb_pt_gather.comp bb_pt_gather_comp)
add_host_shader(bb_pt_view.comp bb_pt_view_comp)
add_host_shader(bb_pt_gi_trace.comp bb_pt_gi_trace_comp)
add_host_shader(bb_pt_gi_accum.comp bb_pt_gi_accum_comp)

foreach(BPP IN ITEMS 8 16 32 64 96 128)
    add_host_shader(tiling.comp tiling_micro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=0)
    add_host_shader(tiling.comp tiling_macro_${BPP}_comp BITS_PER_PIXEL=${BPP} IS_MACRO_TILED=1)
endforeach()

list(REMOVE_DUPLICATES SHADER_FILES)
add_custom_target(host_shaders DEPENDS ${SHADER_HEADERS} SOURCES ${SHADER_FILES})
