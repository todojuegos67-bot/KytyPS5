"""Compile inputs of a shader from its AGC header: the emulator's register -> input info derivation.

Ports of ShaderGetStaticInputInfoCS/PS/VS + BuildStageStaticKey (src/graphics/shader/shader.cpp),
the register handlers that load the header's register lists (pm4Handlers.cpp) and
AgcCreateInterpolantMapping (src/libs/agc.cpp), which fills SPI_PS_INPUT_CNTL from a (VS, PS) pair.
Checked against every recorded warmup record (tools/local/static-precompile/precompile.py check).
"""
import warmfile
import xxh3
from agc import reg_first

# SH/CX register offsets (src/graphics/guest_gpu/pm4.h)
COMPUTE_NUM_THREAD_X, COMPUTE_NUM_THREAD_Y, COMPUTE_NUM_THREAD_Z, COMPUTE_PGM_RSRC2 = 0x207, 0x208, 0x209, 0x213
SPI_SHADER_PGM_RSRC2_PS, SPI_SHADER_PGM_RSRC2_GS = 0xb, 0x8b
SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR, SPI_PS_IN_CONTROL = 0x1b3, 0x1b4, 0x1b6
SPI_SHADER_COL_FORMAT, DB_SHADER_CONTROL, PA_CL_VS_OUT_CNTL = 0x1c5, 0x203, 0x207
IDENTITY_EXPORT_MAPPING = 0xe4
# The SGPR the game's indirect draws have the CP write the start instance into (START_INST_LOC): the third GS user SGPR,
# s8 + 2, of a vertex shader with three (its table pointer, then the instance offset; every recorded indirect draw's).
INDIRECT_START_INSTANCE_SGPR = 8 + 2


def _words(code):
    return len(code) // 4


def _user_data_count(rsrc2):
    return ((rsrc2 >> 1) & 0x1f) | (((rsrc2 >> 27) & 1) << 5)


def compute_record(code, agc, host_subgroup_size):
    """The compute record of a shader: every input but the resource specialization.
    `host_subgroup_size` is the host GPU's (64 with wave64 compute support, else 32)."""
    sh = agc['sh_registers']
    rsrc2 = reg_first(sh, COMPUTE_PGM_RSRC2)
    threads = (reg_first(sh, COMPUTE_NUM_THREAD_X), reg_first(sh, COMPUTE_NUM_THREAD_Y),
               reg_first(sh, COMPUTE_NUM_THREAD_Z))
    if rsrc2 is None or None in threads:
        raise ValueError('compute header without its static registers')
    modifier = agc['specials']['dispatch_modifier'] if agc['specials'] else 0
    info = {
        'threads_num': threads, 'lds_size_dwords': ((rsrc2 >> 15) & 0x1ff) * 128,
        'scratch_size_dwords': agc['scratch_size_dw_per_thread'], 'host_subgroup_size': host_subgroup_size,
        # The game dispatches with the header's modifier (AgcCbDispatch): wave size, thread dimensions.
        'wave_size': 32 if (modifier >> 15) & 1 else 64, 'dispatch_threads_num': (0, 0, 0),
        'group_id': ((rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1),
        'dispatch_thread_dimensions': (modifier >> 5) & 1, 'thread_ids_num': ((rsrc2 >> 11) & 3) + 1,
        'workgroup_register': (rsrc2 >> 1) & 0x1f, 'tg_size_en': (rsrc2 >> 10) & 1, '_kind': 'cs',
    }
    key = (info['workgroup_register'], info['wave_size'], host_subgroup_size, info['thread_ids_num'],
           info['lds_size_dwords'], info['scratch_size_dwords'], info['dispatch_thread_dimensions'],
           threads[0], info['group_id'][0], threads[1], info['group_id'][1], threads[2], info['group_id'][2],
           info['tg_size_en'])
    return _record(warmfile.ST_COMPUTE, code, info['workgroup_register'], key, info)


# AgcCreateInterpolantMapping (agc.cpp): SPI_PS_INPUT_CNTL_i for each PS input from the VS outputs.
def _apply_default(value, ps_word):
    return (value & ~0x300 & 0xffffffff) | (((ps_word >> 28) & 3) << 8)


def _apply_default_hi(value, ps_word):
    return (value & ~0x00600000 & 0xffffffff) | (((ps_word >> 30) & 3) << 21)


def interpolant_mapping(vs_agc, ps_agc):
    regs = list(range(32))
    if ps_agc is None or ps_agc['num_input_semantics'] == 0:
        return regs
    outputs = vs_agc['output_semantics'] if vs_agc else []
    for i, semantic in enumerate(ps_agc['input_semantics']):
        vs_semantic = next((o for o in outputs if o['semantic'] == semantic['semantic']), None)
        ps_word = semantic['word']
        if ps_word & 0x00300000:
            value = (ps_word << 4) & 0x03000000
            if vs_semantic is None:
                value |= 0x00180020
            else:
                common = ps_word & vs_semantic['word']
                value &= 0xfff7ffdf
                value |= (common >> 15) & 0x20
                value ^= 0x00080020
                value &= ~0x00100000 & 0xffffffff
                value |= ((~common & 0xffffffff) >> 1) & 0x00100000
            value = _apply_default_hi(value, ps_word)
        else:
            value = 0x20 if (ps_word & 0x01000000) or vs_semantic is None else 0
        if vs_semantic is None:
            value &= ~0x41f & 0xffffffff
        else:
            flat = 0x400 if (ps_word & 0x00400000) or (ps_word & 0x01000000) else 0
            value = (value & ~0x41f & 0xffffffff) | ((vs_semantic['word'] >> 8) & 0x1f) | flat
        regs[i] = _apply_default(value, ps_word) & 0xffffffff
    return regs


def _ps_system_input_base(addr):
    return sum(n for bit, n in ((0x1, 2), (0x2, 2), (0x4, 2), (0x8, 3), (0x10, 2), (0x20, 2), (0x40, 2), (0x80, 1))
               if addr & bit)


def pixel_record(code, agc, vs_agc, lod_stats_subgroup):
    """The pixel record of a shader drawn after `vs_agc` (its outputs feed the interpolators).
    `lod_stats_subgroup` is the host device's fragment subgroup reduction support."""
    cx = agc['cx_registers']
    reg = lambda offset: reg_first(cx, offset, 0)
    ena, addr = reg(SPI_PS_INPUT_ENA), reg(SPI_PS_INPUT_ADDR)
    active = ena & addr
    dbc = reg(DB_SHADER_CONTROL)
    z_export, z_order, kill, mask_export = dbc & 1, (dbc >> 4) & 3, (dbc >> 6) & 1, (dbc >> 8) & 1
    col = reg(SPI_SHADER_COL_FORMAT)
    modes = tuple((col >> (4 * i)) & 0xf for i in range(8))
    input_num = reg(SPI_PS_IN_CONTROL) & 0x3f
    custom = 0
    for i, semantic in enumerate(agc['input_semantics'][:min(input_num, 32)]):
        if semantic['is_custom'] and not semantic['is_f16']:
            custom |= 1 << i
    interpolators = interpolant_mapping(vs_agc, agc)
    info = {
        'lod_stats_subgroup': lod_stats_subgroup, 'input_num': input_num,
        'ps_system_input_base': _ps_system_input_base(addr), 'custom_interpolation_mask': custom,
        'ps_perspective_center_vgpr': (2 if active & 1 else 0) if active & 2 else 0xffffffff,
        'scratch_size_dwords': agc['scratch_size_dw_per_thread'],
        'ps_pos_x': int(bool(active & 0x100)), 'ps_pos_y': int(bool(active & 0x200)),
        'ps_pos_z': int(bool(active & 0x400)), 'ps_pos_w': int(bool(active & 0x800)),
        'ps_front_face': int(bool(active & 0x1000)), 'ps_ancillary': int(bool(active & 0x2000)),
        'ps_no_perspective': int(bool(active & 0x20)), 'ps_pixel_kill_enable': kill,
        'ps_depth_export_enable': z_export, 'ps_sample_mask_export_enable': mask_export,
        'ps_sample_shading': int(bool(active & 0x11)),
        'ps_early_z': int(z_order == 1 and not kill and not z_export and not mask_export),
        'ps_execute_on_noop': (dbc >> 10) & 1,
        'interpolator_settings': tuple(interpolators[:input_num]) + (0,) * (32 - input_num),
        'target_output_mode': modes,
        # The render targets the game binds use the identity channel order (every recorded PS).
        'target_export_mapping': (IDENTITY_EXPORT_MAPPING,) * 8, '_kind': 'ps',
    }
    key = [info['scratch_size_dwords'], input_num, info['ps_system_input_base'], custom,
           info['ps_perspective_center_vgpr'], info['ps_pos_x'], info['ps_pos_y'], info['ps_pos_z'], info['ps_pos_w'],
           info['ps_front_face'], info['ps_ancillary'], info['ps_no_perspective'], kill, z_export, mask_export,
           info['ps_early_z']] + list(modes)
    for base in (0, 4):
        key.append(sum((IDENTITY_EXPORT_MAPPING & 0xff) << (8 * i) for i in range(4)))
    key += interpolators[:input_num]
    rsrc2 = reg_first(agc['sh_registers'], SPI_SHADER_PGM_RSRC2_PS, 0)
    return _record(warmfile.ST_PIXEL, code, _user_data_count(rsrc2), tuple(key), info)


def vertex_record(code, agc, start_instance_sgpr=-1):
    """The vertex (NGG 'Gs') record: this game's vertex shaders pull their vertices through the SRT
    (no fetch tables, no clip transform, no mesh path), so the key holds header fields only; start_instance_sgpr -1
    for direct draws, the SGPR the start instance is written to for indirect ones."""
    offsets = agc['user_data']['direct_resource_offset'] if agc['user_data'] else []
    if (len(offsets) > 10 and offsets[10] != 0xffff) or (len(offsets) > 8 and offsets[8] != 0xffff):
        raise ValueError('vertex shader with fetch tables: its key needs draw-time descriptors')
    scratch = agc['scratch_size_dw_per_thread']
    out_cntl = reg_first(agc['cx_registers'], PA_CL_VS_OUT_CNTL, 0)
    info = {
        'resources_num': 0, 'fetch_attrib_reg': 0, 'fetch_buffer_reg': 0, 'scratch_size_dwords': scratch,
        'pa_cl_vs_out_cntl': out_cntl, 'start_instance_sgpr': start_instance_sgpr, 'fetch_external': 0, 'fetch_embedded': 0, 'clip_enabled': 0,
        'clip_scale': (0, 0), 'clip_offset': (0, 0), 'clip_half_extent': (0, 0),
        'mesh': {'threads_num': (0, 0, 0), 'lds_size_dwords': 0, 'scratch_size_dwords': 0, 'host_subgroup_size': 64,
                 'wave_size': 64, 'input_primitive': 0, 'primitives_per_group': 0, 'vertices_per_group': 0,
                 'max_vertices': 0, 'max_primitives': 0, 'provoking_vertex': 0},
        'res_fields': [], 'res_dst': [], '_kind': 'vs',
    }
    key = (0, 0, 0, 0, scratch, out_cntl, start_instance_sgpr & 0xffffffff, 0, 0)
    rsrc2 = reg_first(agc['sh_registers'], SPI_SHADER_PGM_RSRC2_GS, 0)
    return _record(warmfile.ST_VERTEX, code, _user_data_count(rsrc2), key, info)


def vertex_records(code, agc):
    """The records of a vertex shader: its direct draws', and its indirect draws' where it has the three user SGPRs
    whose last the CP writes the start instance into (a pipeline of each: the start instance is read from Vulkan's
    base instance there)."""
    direct = vertex_record(code, agc)
    if direct.udc != 3:
        return [direct]
    return [direct, vertex_record(code, agc, INDIRECT_START_INSTANCE_SGPR)]


def _record(stage, code, udc, key, info):
    rec = warmfile.Record()
    rec.index, rec.stage, rec.hash, rec.udc, rec.push = -1, stage, xxh3.xxh3_64(code), udc, 0
    rec.code, rec.back, rec.key, rec.info = bytes(code), b'', tuple(int(k) for k in key), info
    rec.code_words, rec.buffers, rec.images, rec.cid, rec.raw, rec.nwords = _words(code), (), (), None, None, 0
    return rec
