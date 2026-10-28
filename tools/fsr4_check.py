#!/usr/bin/env python3
# FSR 4 asset check (fsr4_shaders/): reflects every SPIR-V module of each preset's shader_manifest.json with a small
# self-contained SPIR-V parser and checks it against the manifest: size, sha256, entry point, local size,
# capabilities, extensions and each descriptor (set, binding, type). The emulator's runtime (src/local/fsr4-sr.cpp)
# (and the injector's port, src/up_fsr4.cpp) builds its descriptor set layouts from the manifest, so a match here means its layouts match the shaders.
# Usage: spv_reflect.py <fsr4_shaders dir> [--dump <file.spv>]   (exit 1 on any mismatch)
import hashlib
import json
import os
import struct
import sys

OP_NAME, OP_MEMBER_NAME, OP_EXTENSION, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_CAPABILITY = 5, 6, 10, 15, 16, 17
OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE = 21, 22, 23, 25, 26, 27
OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER, OP_CONSTANT, OP_VARIABLE = 28, 29, 30, 32, 43, 59
OP_DECORATE, OP_MEMBER_DECORATE = 71, 72
DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_OFFSET, DEC_BINDING, DEC_SET = 2, 3, 35, 33, 34
SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_PUSH_CONSTANT, SC_STORAGE_BUFFER = 0, 2, 9, 12
CAPS = {1: 'Shader', 9: 'Float16', 22: 'Int16', 39: 'Int8', 49: 'StorageImageExtendedFormats', 61: 'GroupNonUniform',
        68: 'GroupNonUniformQuad', 5350: 'ComputeDerivativeGroupLinearKHR', 6016: 'DotProductInputAll',
        6017: 'DotProductInput4x8Bit', 6018: 'DotProductInput4x8BitPacked', 6019: 'DotProduct',
        4433: 'StorageBuffer16BitAccess', 4434: 'UniformAndStorageBuffer16BitAccess', 4448: 'StorageBuffer8BitAccess',
        5288: 'ComputeDerivativeGroupQuadsKHR'}
FORMATS = {0: 'Unknown', 1: 'Rgba32f', 2: 'Rgba16f', 3: 'R32f', 4: 'Rgba8', 6: 'Rg32f', 7: 'Rg16f', 9: 'R16f',
           15: 'R8', 24: 'Rg8', 33: 'R32ui', 21: 'Rgba32ui', 30: 'Rgba8ui'}


def words_str(words):
    raw = struct.pack('<%dI' % len(words), *words)
    return raw.split(b'\0', 1)[0].decode()


def reflect(data):
    words = struct.unpack('<%dI' % (len(data) // 4), data)
    if words[0] != 0x07230203:
        raise ValueError('not SPIR-V')
    names, member_names, decorations, member_offsets, types, variables = {}, {}, {}, {}, {}, []
    caps, exts, entry, local_size = [], [], None, None
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        a = words[i + 1:i + count]
        if op == OP_CAPABILITY:
            caps.append(CAPS.get(a[0], str(a[0])))
        elif op == OP_EXTENSION:
            exts.append(words_str(a))
        elif op == OP_ENTRY_POINT:
            entry = words_str(a[2:])
        elif op == OP_EXECUTION_MODE and a[1] == 17:  # LocalSize
            local_size = list(a[2:5])
        elif op == OP_NAME:
            names[a[0]] = words_str(a[1:])
        elif op == OP_MEMBER_NAME:
            member_names[(a[0], a[1])] = words_str(a[2:])
        elif op == OP_DECORATE:
            decorations.setdefault(a[0], {})[a[1]] = a[2] if len(a) > 2 else True
        elif op == OP_MEMBER_DECORATE and a[2] == DEC_OFFSET:
            member_offsets[(a[0], a[1])] = a[3]
        elif op in (OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR, OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE,
                    OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_POINTER):
            types[a[0]] = (op, a[1:])
        elif op == OP_VARIABLE:
            variables.append((a[1], a[0], a[2]))  # id, pointer type, storage class
        i += count
    resources, push = [], None

    def strip(t):
        while types.get(t, (None,))[0] in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            t = types[t][1][0]
        return t

    for var, ptr, sc in variables:
        pointee = strip(types[ptr][1][1])
        kind, args = types.get(pointee, (None, ()))
        if sc == SC_PUSH_CONSTANT:
            push = [(member_names.get((pointee, m), str(m)), member_offsets.get((pointee, m))) for m in range(len(args))]
            continue
        if sc not in (SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE_BUFFER):
            continue
        d = decorations.get(var, {})
        r = {'set': d.get(DEC_SET, 0), 'binding': d.get(DEC_BINDING), 'name': names.get(var, '')}
        if kind == OP_TYPE_IMAGE:
            sampled = args[5]
            r['type'] = 'storage_image' if sampled == 2 else 'sampled_image'
            r['format'] = FORMATS.get(args[6], str(args[6]))
        elif kind == OP_TYPE_SAMPLER:
            r['type'] = 'sampler'
        elif kind == OP_TYPE_SAMPLED_IMAGE:
            r['type'] = 'combined_image_sampler'
        elif kind == OP_TYPE_STRUCT:
            block = decorations.get(pointee, {})
            if sc == SC_STORAGE_BUFFER or DEC_BUFFER_BLOCK in block:
                r['type'] = 'storage_buffer'
            else:
                r['type'] = 'uniform_buffer'
                r['members'] = [(member_names.get((pointee, m), str(m)), member_offsets.get((pointee, m)))
                                for m in range(len(args))]
        else:
            continue
        resources.append(r)
    return {'entry_point': entry, 'local_size': local_size, 'capabilities': sorted(caps), 'extensions': sorted(exts),
            'resources': sorted(resources, key=lambda r: (r['set'], r['binding'])), 'push_constants': push}


def check(root):
    errors, checked, skipped = 0, 0, 0
    manifests = sorted(f for f in os.listdir(root) if f.endswith('_shader_manifest.json'))
    for mf in manifests:
        manifest = json.load(open(os.path.join(root, mf)))
        for name, art in sorted(manifest['artifacts'].items()):
            path = os.path.join(root, name)
            if '_4320_' in name and not os.path.exists(path):
                skipped += 1  # (8K variants: not shipped, the runtime picks 1080 or 2160 only)
                continue
            data = open(path, 'rb').read()
            problems = []
            if len(data) != art['size']:
                problems.append('size %d != %d' % (len(data), art['size']))
            if hashlib.sha256(data).hexdigest() != art['sha256']:
                problems.append('sha256')
            if name.endswith('.spv'):
                r = reflect(data)
                if r['entry_point'] != art['entry_point']:
                    problems.append('entry %s' % r['entry_point'])
                if r['local_size'] != art['local_size']:
                    problems.append('local size %s' % r['local_size'])
                if r['capabilities'] != sorted(art['capabilities']):
                    problems.append('caps %s' % r['capabilities'])
                if r['extensions'] != sorted(art['extensions']):
                    problems.append('exts %s' % r['extensions'])
                got = [(x['set'], x['binding'], x['type']) for x in r['resources']]
                want = sorted((x['set'], x['binding'], x['type']) for x in art['resources'])
                if got != want:
                    problems.append('descriptors %s != %s' % (got, want))
                if r['push_constants']:
                    problems.append('push constants %s (the runtime has none)' % r['push_constants'])
                # The variant in opt/ must keep the same interface (the runtime may load either).
                opt = os.path.join(root, 'opt', name)
                if os.path.exists(opt):
                    o = reflect(open(opt, 'rb').read())
                    if [(x['set'], x['binding'], x['type']) for x in o['resources']] != got or \
                            o['local_size'] != r['local_size']:
                        problems.append('opt/ variant interface differs')
            checked += 1
            if problems:
                errors += 1
                print('MISMATCH %s: %s' % (name, '; '.join(problems)))
    for extra in ('rcas.spv', 'spd_auto_exposure.spv'):
        r = reflect(open(os.path.join(root, extra), 'rb').read())
        checked += 1
        print('%s: entry %s local %s %s' % (extra, r['entry_point'], r['local_size'],
                                             [(x['binding'], x['type'], x['name']) for x in r['resources']]))
    print('%d manifests, %d files checked, %d mismatches (%d 4320 entries not shipped)' % (len(manifests), checked, errors, skipped))
    return errors == 0 and len(manifests) == 6


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--dump':
        print(json.dumps(reflect(open(sys.argv[3], 'rb').read()), indent=1))
        sys.exit(0)
    sys.exit(0 if check(sys.argv[1]) else 1)
