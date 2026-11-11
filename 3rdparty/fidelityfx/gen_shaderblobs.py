#!/usr/bin/env python3
"""Compiles the FidelityFX frame interpolation and optical flow GLSL passes for Vulkan and writes
the shader blob accessors the SDK's Vulkan backend expects (replaces the SDK's Windows-only
FidelityFX-SC tool). Bindings are read from the SPIR-V itself.

usage: gen_shaderblobs.py GLSLANG_VALIDATOR SDK_DIR OUTPUT_CPP
Only wave32, 32-bit permutations are generated; wave64 and FP16 requests return them as well."""
import itertools
import os
import struct
import subprocess
import sys
import tempfile

FI_PASSES = [
    ("FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_AND_DILATE", "ffx_frameinterpolation_reconstruct_and_dilate_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_SETUP", "ffx_frameinterpolation_setup_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_PREV_DEPTH", "ffx_frameinterpolation_reconstruct_previous_depth_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_GAME_MOTION_VECTOR_FIELD", "ffx_frameinterpolation_game_motion_vector_field_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_OPTICAL_FLOW_VECTOR_FIELD", "ffx_frameinterpolation_optical_flow_vector_field_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_DISOCCLUSION_MASK", "ffx_frameinterpolation_disocclusion_mask_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_INTERPOLATION", "ffx_frameinterpolation_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_INPAINTING_PYRAMID", "ffx_frameinterpolation_compute_inpainting_pyramid_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_INPAINTING", "ffx_frameinterpolation_inpainting_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_GAME_VECTOR_FIELD_INPAINTING_PYRAMID",
     "ffx_frameinterpolation_compute_game_vector_field_inpainting_pyramid_pass"),
    ("FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW", "ffx_frameinterpolation_debug_view_pass"),
]
# Permutation option bits (FRAMEINTERPOLATION_SHADER_PERMUTATION_*) and their defines.
FI_OPTIONS = [(1 << 0, "FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS"),
              (1 << 1, "FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS"),
              (1 << 2, "FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH")]
FI_BASE = ["-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
           "-DFFX_FRAMEINTERPOLATION_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
           "-DFFX_FRAMEINTERPOLATION_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
           "-DFFX_FRAMEINTERPOLATION_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
           "-DFFX_FRAMEINTERPOLATION_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2"]

OF_PASSES = [
    ("FFX_OPTICALFLOW_PASS_PREPARE_LUMA", "ffx_opticalflow_prepare_luma_pass"),
    ("FFX_OPTICALFLOW_PASS_GENERATE_OPTICAL_FLOW_INPUT_PYRAMID", "ffx_opticalflow_compute_luminance_pyramid_pass"),
    ("FFX_OPTICALFLOW_PASS_GENERATE_SCD_HISTOGRAM", "ffx_opticalflow_generate_scd_histogram_pass"),
    ("FFX_OPTICALFLOW_PASS_COMPUTE_SCD_DIVERGENCE", "ffx_opticalflow_compute_scd_divergence_pass"),
    ("FFX_OPTICALFLOW_PASS_COMPUTE_OPTICAL_FLOW_ADVANCED_V5", "ffx_opticalflow_compute_optical_flow_advanced_pass_v5"),
    ("FFX_OPTICALFLOW_PASS_FILTER_OPTICAL_FLOW_V5", "ffx_opticalflow_filter_optical_flow_pass_v5"),
    ("FFX_OPTICALFLOW_PASS_SCALE_OPTICAL_FLOW_ADVANCED_V5", "ffx_opticalflow_scale_optical_flow_advanced_pass_v5"),
]
OF_OPTIONS = [(1 << 2, "FFX_OPTICALFLOW_OPTION_HDR_COLOR_INPUT")]

FSR3_PASSES = [
    ("FFX_FSR3UPSCALER_PASS_PREPARE_INPUTS", "ffx_fsr3upscaler_prepare_inputs_pass"),
    ("FFX_FSR3UPSCALER_PASS_LUMA_PYRAMID", "ffx_fsr3upscaler_luma_pyramid_pass"),
    ("FFX_FSR3UPSCALER_PASS_SHADING_CHANGE_PYRAMID", "ffx_fsr3upscaler_shading_change_pyramid_pass"),
    ("FFX_FSR3UPSCALER_PASS_SHADING_CHANGE", "ffx_fsr3upscaler_shading_change_pass"),
    ("FFX_FSR3UPSCALER_PASS_PREPARE_REACTIVITY", "ffx_fsr3upscaler_prepare_reactivity_pass"),
    ("FFX_FSR3UPSCALER_PASS_LUMA_INSTABILITY", "ffx_fsr3upscaler_luma_instability_pass"),
    ("FFX_FSR3UPSCALER_PASS_ACCUMULATE", "ffx_fsr3upscaler_accumulate_pass"),
    ("FFX_FSR3UPSCALER_PASS_ACCUMULATE_SHARPEN", "ffx_fsr3upscaler_accumulate_pass"),
    ("FFX_FSR3UPSCALER_PASS_RCAS", "ffx_fsr3upscaler_rcas_pass"),
    ("FFX_FSR3UPSCALER_PASS_DEBUG_VIEW", "ffx_fsr3upscaler_debug_view_pass"),
    ("FFX_FSR3UPSCALER_PASS_GENERATE_REACTIVE", "ffx_fsr3upscaler_autogen_reactive_pass"),
]

FSR3_OPTIONS = [
    (1 << 0, "FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE"),
    (1 << 1, "FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT"),
    (1 << 2, "FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS"),
    (1 << 3, "FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS"),
    (1 << 4, "FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH"),
    (1 << 5, "FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING"),
]

FSR3_BASE = [
    "-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1",
    "-DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0",
    "-DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2",
]

# SPIR-V opcodes, decorations and storage classes used for reflection.
OP_NAME, OP_DECORATE, OP_MEMBER_DECORATE = 5, 71, 72
OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE = 25, 26, 27
OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER = 28, 29, 30, 32
OP_CONSTANT, OP_VARIABLE = 43, 59
DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_NON_WRITABLE, DEC_BINDING, DEC_SET = 2, 3, 24, 33, 34
SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE_BUFFER = 0, 2, 12
CATEGORIES = ["cbv", "srv_texture", "uav_texture", "srv_buffer", "uav_buffer", "sampler", "rt"]


def reflect(spirv):
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    names, decorations, member_nonwritable, types, constants, variables = {}, {}, {}, {}, {}, []
    i = 5
    while i < len(words):
        count, opcode = words[i] >> 16, words[i] & 0xFFFF
        operands = words[i + 1:i + count]
        if opcode == OP_NAME:
            raw = struct.pack(f"<{len(operands) - 1}I", *operands[1:])
            names[operands[0]] = raw.split(b"\0", 1)[0].decode()
        elif opcode == OP_DECORATE:
            decorations.setdefault(operands[0], {})[operands[1]] = operands[2] if len(operands) > 2 else True
        elif opcode == OP_MEMBER_DECORATE and operands[2] == DEC_NON_WRITABLE:
            member_nonwritable.setdefault(operands[0], set()).add(operands[1])
        elif opcode == OP_TYPE_IMAGE:
            types[operands[0]] = ("image", operands[2], operands[6])  # dim, sampled
        elif opcode == OP_TYPE_SAMPLER:
            types[operands[0]] = ("sampler",)
        elif opcode == OP_TYPE_SAMPLED_IMAGE:
            types[operands[0]] = ("sampled_image", operands[1])
        elif opcode == OP_TYPE_ARRAY:
            types[operands[0]] = ("array", operands[1], operands[2])
        elif opcode == OP_TYPE_RUNTIME_ARRAY:
            types[operands[0]] = ("array", operands[1], None)
        elif opcode == OP_TYPE_STRUCT:
            types[operands[0]] = ("struct", len(operands) - 1)
        elif opcode == OP_TYPE_POINTER:
            types[operands[0]] = ("pointer", operands[1], operands[2])
        elif opcode == OP_CONSTANT:
            constants[operands[1]] = operands[2]
        elif opcode == OP_VARIABLE:
            variables.append((operands[1], operands[0], operands[2]))
        i += count

    bindings = {category: [] for category in CATEGORIES}
    for var_id, pointer, storage in variables:
        decoration = decorations.get(var_id, {})
        if DEC_BINDING not in decoration:
            continue
        kind = types[pointer]
        base, count = kind[2], 1
        while types.get(base, ("",))[0] == "array":
            _, element, length = types[base]
            count *= constants.get(length, 1) if length is not None else 1
            base = element
        base_type = types.get(base, ("",))
        if storage == SC_UNIFORM_CONSTANT:
            if base_type[0] == "sampler":
                category = "sampler"
            elif base_type[0] == "sampled_image":
                category = "srv_texture"
            elif base_type[0] == "image":
                category = "uav_texture" if base_type[2] == 2 else "srv_texture"
            else:
                raise ValueError(f"unsupported uniform constant {names.get(var_id)}")
        elif storage in (SC_UNIFORM, SC_STORAGE_BUFFER):
            block = decorations.get(base, {})
            if storage == SC_UNIFORM and DEC_BLOCK in block:
                category = "cbv"
            else:
                readonly = DEC_NON_WRITABLE in decoration or \
                    len(member_nonwritable.get(base, ())) == base_type[1]
                category = "srv_buffer" if readonly else "uav_buffer"
        else:
            continue
        bindings[category].append((names.get(var_id, f"unnamed_{var_id}"), decoration[DEC_BINDING],
                                   count, decoration.get(DEC_SET, 0)))
    for category in bindings:
        bindings[category].sort(key=lambda entry: entry[1])
    return bindings


COMPILE_CACHE = {}
BLOB_CACHE = {}


def compile_permutation(glslang, sdk, effect_dir, source, defines):
    key = (effect_dir, source, tuple(defines))
    if key in COMPILE_CACHE:
        return COMPILE_CACHE[key]
    gpu = os.path.join(sdk, "include", "FidelityFX", "gpu")
    # (A file in a directory of its own: on Windows glslang cannot write a NamedTemporaryFile this process holds open.)
    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "shader.spv")
        command = [glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-Os",
                   "-DFFX_GLSL=1", "-DFFX_GPU=1", "-DFFX_HALF=0", *defines,
                   f"-I{gpu}", f"-I{os.path.join(gpu, effect_dir)}",
                   os.path.join(sdk, "src", "backends", "vk", "shaders", effect_dir, source + ".glsl"),
                   "-o", output]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"glslang failed for {source} {defines}:\n{result.stdout}{result.stderr}")
        with open(output, "rb") as spirv:
            data = spirv.read()
        COMPILE_CACHE[key] = data
        return data


def emit_blob(out, symbol, spirv, bindings):
    out.append(f"static const uint32_t {symbol}_code[] = {{")
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    for start in range(0, len(words), 12):
        out.append("    " + ", ".join(f"0x{w:08x}u" for w in words[start:start + 12]) + ",")
    out.append("};")
    fields = []
    for category in CATEGORIES:
        entries = bindings[category]
        if entries:
            out.append(f"static const char* {symbol}_{category}_names[] = {{"
                       + ", ".join(f'"{name}"' for name, _, _, _ in entries) + "};")
            for index, field in ((1, "bindings"), (2, "counts"), (3, "spaces")):
                out.append(f"static const uint32_t {symbol}_{category}_{field}[] = {{"
                           + ", ".join(str(entry[index]) for entry in entries) + "};")
            fields.append([f"{symbol}_{category}_names", f"{symbol}_{category}_bindings",
                           f"{symbol}_{category}_counts", f"{symbol}_{category}_spaces"])
        else:
            fields.append(["nullptr"] * 4)
    counts = ", ".join(str(len(bindings[category])) for category in CATEGORIES)
    pointers = ",\n    ".join(", ".join(group) for group in fields)
    out.append(f"static const FfxShaderBlob {symbol} = {{\n"
               f"    reinterpret_cast<const uint8_t*>({symbol}_code), sizeof({symbol}_code),\n"
               f"    {counts},\n    {pointers}}};")


def emit_effect(out, glslang, sdk, effect_dir, prefix, passes, options, base_defines):
    tables = []
    for pass_enum, source in passes:
        symbols = []
        for mask in range(1 << len(options)):
            defines = list(base_defines)
            for bit, (_, define) in enumerate(options):
                defines.append(f"-D{define}={(mask >> bit) & 1}")
            spirv = compile_permutation(glslang, sdk, effect_dir, source, defines)
            if spirv not in BLOB_CACHE:
                symbol = f"{prefix}_{source}_{len(BLOB_CACHE)}"
                emit_blob(out, symbol, spirv, reflect(spirv))
                BLOB_CACHE[spirv] = symbol
            symbols.append(BLOB_CACHE[spirv])
        tables.append((pass_enum, symbols))
    return tables


def emit_lookup(out, function, pass_type, tables, options, extra_parameter):
    out.append(f"static uint32_t {function}Index(uint32_t permutationOptions) {{")
    out.append("    uint32_t index = 0;")
    for bit, (flag, _) in enumerate(options):
        out.append(f"    if ((permutationOptions & 0x{flag:x}u) != 0) index |= {1 << bit}u;")
    out.append("    return index;\n}")
    out.append(f"FfxErrorCode {function}({pass_type} passId,{extra_parameter} uint32_t permutationOptions, "
               "FfxShaderBlob* outBlob) {")
    out.append(f"    const uint32_t index = {function}Index(permutationOptions);")
    out.append("    switch (passId) {")
    for pass_enum, symbols in tables:
        out.append(f"        case {pass_enum}: {{")
        out.append("            static const FfxShaderBlob* blobs[] = {" + ", ".join(f"&{s}" for s in symbols) + "};")
        out.append("            memcpy(outBlob, blobs[index], sizeof(FfxShaderBlob));")
        out.append("            return FFX_OK;\n        }")
    out.append("        default: return FFX_ERROR_INVALID_ENUM;\n    }\n}")


def main():
    glslang, sdk, output = sys.argv[1:4]
    out = ["// Generated by gen_shaderblobs.py from the FidelityFX SDK GLSL passes. Do not edit.",
           "#define FFX_CPU", "#include <FidelityFX/host/ffx_util.h>",
           '#include "blob_accessors/ffx_frameinterpolation_shaderblobs.h"',
           '#include "blob_accessors/ffx_opticalflow_shaderblobs.h"',
           '#include "blob_accessors/ffx_fsr3upscaler_shaderblobs.h"',
           "#include <cstring>", ""]
    fi = emit_effect(out, glslang, sdk, "frameinterpolation", "fi", FI_PASSES, FI_OPTIONS, FI_BASE)
    of = emit_effect(out, glslang, sdk, "opticalflow", "of", OF_PASSES, OF_OPTIONS, [])
    fsr3 = emit_effect(out, glslang, sdk, "fsr3upscaler", "fsr3", FSR3_PASSES, FSR3_OPTIONS, FSR3_BASE)
    out.append('extern "C" {')
    emit_lookup(out, "frameInterpolationGetPermutationBlobByIndex", "FfxFrameInterpolationPass", fi,
                FI_OPTIONS, " FfxBindStage,")
    out.append("FfxErrorCode frameInterpolationIsWave64(uint32_t, bool& isWave64) {\n"
               "    isWave64 = false;\n    return FFX_OK;\n}")
    emit_lookup(out, "opticalflowGetPermutationBlobByIndex", "FfxOpticalflowPass", of, OF_OPTIONS, "")
    out.append("FfxErrorCode opticalflowIsWave64(uint32_t, bool& isWave64) {\n"
               "    isWave64 = false;\n    return FFX_OK;\n}")
    emit_lookup(out, "fsr3UpscalerGetPermutationBlobByIndex", "FfxFsr3UpscalerPass", fsr3, FSR3_OPTIONS, "")
    out.append("FfxErrorCode fsr3UpscalerIsWave64(uint32_t, bool& isWave64) {\n"
               "    isWave64 = false;\n    return FFX_OK;\n}")
    out.append("}")
    tmp = output + ".tmp"
    open(tmp, "w").write("\n".join(out) + "\n")
    os.replace(tmp, output)


main()
