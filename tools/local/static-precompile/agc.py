"""AGC shader headers, CSDR shader bundles and the shaders embedded in eboot.bin (read-only parsing).

AGC header (`struct Shader` in src/graphics/shader/shader.h; AgcCreateShader in src/libs/agc.cpp),
96 bytes, little endian, 64-bit self-relative pointers (stored value + the field's own offset; 0 = null):
   0 u32 file_header ('1234')   4 u32 version (0x18)   8 user_data   16 code (0 in files)
  24 cx_registers   32 sh_registers   40 specials   48 input_semantics   56 output_semantics
  64 u32 header_size   68 u32 shader_size   72 u32 embedded_cb_dqw   76 u32 target
  80 u32 num_input_semantics   84 u16 scratch_dw_per_thread   86 u16 num_output_semantics
  88 u16 special_sizes_bytes   90 u8 type (Prospero::ShaderBinaryType)   91 u8 n_cx   92 u8 n_sh

A CSDR bundle is a table of entries, each the shader code followed by its AGC header. eboot.bin carries a further array of AGC headers
in its data segment with the code laid out after it, one blob per header in header order, each
256-byte aligned (the relocations that pair a header with its code confirm the rule).
"""
import struct
from pathlib import Path

import gamefs

MAGIC = 0x34333231
VERSION = 0x18
HEADER_MAGIC = b'1234\x18\x00\x00\x00'

# Prospero::ShaderBinaryType (src/graphics/guest_gpu/gpu_defs.h)
CS, PS, GS = 0, 1, 2
BINARY_TYPE_NAMES = {0: 'Cs', 1: 'Ps', 2: 'Gs', 3: 'Hs', 4: 'GsFront', 5: 'HsFront', 6: 'GsBack', 7: 'HsBack', 8: 'Fs'}


class AgcError(ValueError):
    pass


def _u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def _u16(b, o):
    return struct.unpack_from('<H', b, o)[0]


def _i64(b, o):
    return struct.unpack_from('<q', b, o)[0]


def _rel_ptr(h, field, size, what):
    rel = _i64(h, field)
    if rel == 0:
        return None
    ptr = field + rel
    if ptr < 96 or ptr + size > len(h):
        raise AgcError(f'{what}: pointer {ptr} (+{size}) outside a {len(h)}-byte header')
    return ptr


def _semantic(word):
    return {'semantic': word & 0xff, 'hardware_mapping': (word >> 8) & 0xff,
            'size_in_elements': (word >> 16) & 0xf, 'is_f16': (word >> 20) & 3,
            'is_flat_shaded': (word >> 22) & 1, 'is_linear': (word >> 23) & 1,
            'is_custom': (word >> 24) & 1, 'default_value': (word >> 28) & 3,
            'default_value_hi': (word >> 30) & 3, 'word': word}


def parse_agc(h):
    """Parses one AGC header blob. Raises AgcError for a structurally invalid header."""
    if len(h) < 96:
        raise AgcError('header shorter than 96 bytes')
    if _u32(h, 0) != MAGIC or _u32(h, 4) != VERSION:
        raise AgcError('bad magic or version')
    out = {'header_size': _u32(h, 64), 'shader_size': _u32(h, 68), 'num_input_semantics': _u32(h, 80),
           'scratch_size_dw_per_thread': _u16(h, 84), 'num_output_semantics': _u16(h, 86),
           'special_sizes_bytes': _u16(h, 88), 'type': h[90], 'num_cx_registers': h[91], 'num_sh_registers': h[92]}
    if out['header_size'] != len(h):
        raise AgcError(f'header_size {out["header_size"]} != blob length {len(h)}')

    def registers(field, n, what):
        p = _rel_ptr(h, field, n * 8, what)
        if p is None:
            if n:
                raise AgcError(f'{what}: null with {n} registers')
            return []
        return [struct.unpack_from('<II', h, p + 8 * i) for i in range(n)]

    out['cx_registers'] = registers(24, out['num_cx_registers'], 'cx_registers')
    out['sh_registers'] = registers(32, out['num_sh_registers'], 'sh_registers')
    specials = None
    sp = _rel_ptr(h, 40, out['special_sizes_bytes'], 'specials')
    if sp is not None and out['special_sizes_bytes'] >= 48:
        raw = h[sp:sp + out['special_sizes_bytes']]
        specials = {'vgt_shader_stages_en': struct.unpack_from('<II', raw, 8), 'dispatch_modifier': _u32(raw, 16),
                    'user_data_range': struct.unpack_from('<HH', raw, 20)}
    out['specials'] = specials

    def semantics(field, n, what):
        p = _rel_ptr(h, field, n * 4, what)
        return [] if p is None else [_semantic(_u32(h, p + 4 * i)) for i in range(n)]

    out['input_semantics'] = semantics(48, out['num_input_semantics'], 'input_semantics')
    out['output_semantics'] = semantics(56, out['num_output_semantics'], 'output_semantics')
    user_data = None
    up = _rel_ptr(h, 8, 56, 'user_data')
    if up is not None:
        count = _u16(h, up + 44)
        dp = _rel_ptr(h, up, 2 * count, 'direct_resource_offset')
        user_data = {'srt_size_dw': _u16(h, up + 42), 'direct_resource_count': count,
                     'direct_resource_offset': list(struct.unpack_from(f'<{count}H', h, dp)) if dp is not None else []}
        if count and dp is None:
            raise AgcError('direct_resource_offset null with a nonzero count')
    out['user_data'] = user_data
    return out


def reg_first(registers, offset, default=None):
    for o, v in registers:
        if o == offset:
            return v
    return default


def parse_bundle(data):
    """Entries of one CSDR bundle: dicts with index, bundle_stage, flags, binary_type, code, header."""
    if len(data) < 20:
        raise ValueError('truncated bundle header')
    if data[-12:-8] != b'RDSC' or struct.unpack_from('<I', data, len(data) - 4)[0] != len(data) - 12:
        raise ValueError('invalid CSDR footer or payload length')
    _, count = struct.unpack_from('<2I', data)
    if not 0 < count <= 16384 or 8 + count * 20 > len(data):
        raise ValueError('unsupported or truncated entry table')
    table_end = 8 + count * 20
    cursor = (table_end + 7) & ~7
    if any(b != 0xff for b in data[table_end:cursor]):
        raise ValueError('invalid table alignment padding')
    entries = []
    for i in range(count):
        entry = 8 + i * 20
        stage, flags, code_bytes, header_bytes, relative = struct.unpack_from('<5I', data, entry)
        begin = entry + 16 + relative
        header = begin + code_bytes
        end = header + header_bytes
        if begin != cursor or code_bytes == 0 or code_bytes % 4 or header_bytes < 96 or end > len(data) - 12:
            raise ValueError(f'entry {i}: non-contiguous or out-of-bounds payload')
        if data[header:header + 4] != b'1234':
            raise ValueError(f'entry {i}: missing AGC header')
        saved_header, saved_code = struct.unpack_from('<2I', data, header + 64)
        binary_type = data[header + 90]
        if saved_code != code_bytes or saved_header != header_bytes or binary_type > 8:
            raise ValueError(f'entry {i}: inconsistent AGC sizes or type')
        entries.append({'index': i, 'bundle_stage': stage, 'flags': flags, 'binary_type': binary_type,
                        'code': bytes(data[begin:header]), 'header': bytes(data[header:end])})
        cursor = end
    if cursor != len(data) - 12:
        raise ValueError('unrecognized trailing bundle data')
    return entries


def csdr_files(game):
    """The PS5 bundles the emulator loads (the '_trinity' twins are the PS5 Pro variants)."""
    return [f for f in sorted(gamefs.game_path(game).rglob('*.csdr')) if not f.stem.endswith('_trinity')]


SELF_MAGICS = (b'\x4f\x15\x3d\x1d', b'\x54\x14\xf5\xee')


def plain_elf(data):
    """The ELF image of an eboot.bin: the file when it is one, else the ELF of its SELF with plaintext segments (a
    fake-signed one, as the emulator's loader reads it: src/loader/elf.cpp). Each segment flagged 0x800 holds the
    file bytes of program header (type >> 20) & 0xfff; the image puts them back at that header's file offset."""
    if data[:4] == b'\x7fELF':
        return data
    if len(data) < 32 or data[:4] not in SELF_MAGICS:
        raise AgcError('not an ELF file nor a SELF')
    file_size = struct.unpack_from('<Q', data, 16)[0]
    count = struct.unpack_from('<H', data, 24)[0]
    ehdr = 32 + 32 * count
    if len(data) < ehdr + 64 or data[ehdr:ehdr + 4] != b'\x7fELF':
        raise AgcError('SELF without an ELF header')
    phoff = struct.unpack_from('<Q', data, ehdr + 32)[0]
    phentsize, phnum = struct.unpack_from('<HH', data, ehdr + 54)
    table_end = phoff + phnum * phentsize
    phdrs = [struct.unpack_from('<IIQQQQQQ', data, ehdr + phoff + i * phentsize) for i in range(phnum)]
    image = bytearray(max([table_end] + [p[2] + p[5] for p in phdrs]))
    image[:table_end] = data[ehdr:ehdr + table_end]
    for i in range(count):
        kind, offset, stored, size = struct.unpack_from('<4Q', data, 32 + 32 * i)
        index = (kind >> 20) & 0xfff
        if not kind & 0x800 or index >= phnum:
            continue
        if kind & 0x2 or stored != size or size != phdrs[index][5] or offset + size > len(data):
            raise AgcError('an encrypted or compressed SELF: the emulator needs the decrypted game (its plain '
                           'eboot.bin or a fake-signed SELF)')
        image[phdrs[index][2]:phdrs[index][2] + size] = data[offset:offset + size]
    # (A header whose bytes no segment holds: what follows the SELF's own bytes, when they are its size.)
    for p in phdrs:
        if p[5] != 0 and p[5] == len(data) - file_size and not any(image[p[2]:p[2] + p[5]]):
            image[p[2]:p[2] + p[5]] = data[file_size:]
    return bytes(image)


def _elf_loads(elf):
    if elf[:4] != b'\x7fELF':
        raise AgcError('not an ELF file')
    phoff = struct.unpack_from('<Q', elf, 32)[0]
    phentsize, phnum = struct.unpack_from('<HH', elf, 54)
    loads = []
    for i in range(phnum):
        kind, _, off, va, _, filesz, _, _ = struct.unpack_from('<IIQQQQQQ', elf, phoff + i * phentsize)
        loads.append((kind, off, va, filesz))
    return loads


def _relative_relocations(elf, loads):
    """R_X86_64_RELATIVE relocations: target VA -> addend."""
    dynamic = next((l for l in loads if l[0] == 2), None)
    if dynamic is None:
        return {}
    tags = {}
    for i in range(dynamic[3] // 16):
        tag, value = struct.unpack_from('<qQ', elf, dynamic[1] + 16 * i)
        if tag == 0:
            break
        tags.setdefault(tag, value)
    if 7 not in tags or 8 not in tags:
        return {}

    def va_to_off(va):
        for kind, off, base, size in loads:
            if kind == 1 and base <= va < base + size:
                return off + va - base
        return None

    start = va_to_off(tags[7])
    relocations = {}
    for i in range(tags[8] // 24):
        where, info, addend = struct.unpack_from('<QQq', elf, start + 24 * i)
        if info & 0xffffffff == 8:
            relocations[where] = addend
    return relocations


def embedded_shaders(game):
    """The AGC headers in eboot.bin's data segment with their code: list of dicts (index, agc, header, code).
    From decrypted/eboot.bin when the game has one, else the eboot.bin the emulator runs (a plain ELF or a SELF
    with plaintext segments, plain_elf); [] without either."""
    path = gamefs.game_path(game) / 'decrypted' / 'eboot.bin'
    if not path.is_file():
        path = gamefs.game_path(game) / 'eboot.bin'
    if not path.is_file():
        return []
    elf = plain_elf(path.read_bytes())
    loads = _elf_loads(elf)
    first = elf.find(HEADER_MAGIC)
    if first < 0:
        return []
    segment = next(l for l in loads if l[0] == 1 and l[1] <= first < l[1] + l[3])
    delta = segment[1] - segment[2]  # file offset - VA in the data segment
    headers = []
    pos = first
    while 0 <= pos < segment[1] + segment[3]:
        size = struct.unpack_from('<I', elf, pos + 64)[0]
        headers.append((pos, elf[pos:pos + size]))
        nxt = pos + size
        if elf[nxt:nxt + 8] != HEADER_MAGIC:
            break
        pos = nxt
    # The code of header k starts where header k-1's (256-byte aligned) code ends; a relocated
    # (header, code) pointer pair fixes the start of the sequence.
    relocations = _relative_relocations(elf, loads)
    sizes = [struct.unpack_from('<I', h, 68)[0] for _, h in headers]
    starts = None
    header_va = {pos - delta: k for k, (pos, _) in enumerate(headers)}
    for where, addend in relocations.items():
        k = header_va.get(addend)
        if k is not None and where + 8 in relocations:
            code_va = relocations[where + 8]
            for j in range(k):
                code_va -= (sizes[j] + 0xff) & ~0xff
            starts = code_va
            break
    if starts is None:
        raise AgcError('no relocation pairs an embedded header with its code')
    out = []
    code_va = starts
    for k, (pos, header) in enumerate(headers):
        agc = parse_agc(header)
        off = code_va + delta
        out.append({'index': k, 'agc': agc, 'header': header, 'code': elf[off:off + agc['shader_size']]})
        code_va = (code_va + agc['shader_size'] + 0xff) & ~0xff
    return out
