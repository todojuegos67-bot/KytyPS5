#!/usr/bin/env python3
"""Static shader and pipeline precompilation for Demon's Souls (PPSA01341).

Collects every shader the game ships (CSDR bundles and the shaders embedded in eboot.bin), derives
their compile inputs from the AGC headers and the materials, writes them as a seed file for the
emulator's precompile mode, and measures how much of a recorded warmup cache the static set covers.

  precompile.py inventory               what the game ships
  precompile.py check                   static compile keys against the recorded warmup cache
  precompile.py learn-states            the render-pass states (targets, blend, depth) of each material
                                        pass, from a recorded warmup cache, into pass-states.json
  precompile.py seeds OUT               seed file for kyty_shader_precompile (precompile-windows.ps1)
  precompile.py coverage COMPILED       the warmup file a seed run wrote against a recorded one
  precompile.py recorded-seeds OUT      a recorded cache as seeds (its own specializations), whose run
                                        gives `coverage COMPILED --recorded-compiled` the SPIR-V to compare
  precompile.py runtime-coverage LOG COMPILED
                                        the modules a game run compiled (its MODULE lines, KYTY_SLOW_LOG_MS
                                        set) found among a precompile's (static-cache hits)
"""
import argparse
import array
import collections
import json
import re
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agc  # noqa: E402
import keys  # noqa: E402
import materials  # noqa: E402
import warmfile  # noqa: E402
import xxh3  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
GAME = Path.home() / 'Documents' / 'PPSA01341-app0'  # precompile-windows.ps1's default too
STATES = Path(__file__).with_name('pass-states.json')
HOST_SUBGROUP_SIZE = 32  # SupportsComputeWave64() is false on the NVIDIA GPU
LOD_STATS_SUBGROUP = 1   # fragment_subgroup_reduction on the same GPU
SEED_IDENTITY = b'KytyShaderSeeds2:PPSA01341\n'
ENGINE_SHADERS = ('coredata', 'enginesupport', 'shaders')
FULLSCREEN_VS = 'vs_fullscreen'


class Inventory:
    """Every shader of the game, by code bytes, with the headers it comes with."""

    def __init__(self, game):
        self.game = Path(game)
        self.headers = collections.defaultdict(list)  # code -> [agc]
        self.bundles = {}                              # bundle file -> [(code, agc)]
        self.failures = []
        for f in agc.csdr_files(self.game):
            try:
                entries = agc.parse_bundle(f.read_bytes())
                parsed = [(e['code'], agc.parse_agc(e['header'])) for e in entries]
            except (ValueError, agc.AgcError) as error:
                self.failures.append((str(f), str(error)))
                continue
            self.bundles[f] = parsed
            for code, header in parsed:
                self._add(code, header)
        try:
            self.embedded = agc.embedded_shaders(self.game)
        except (ValueError, agc.AgcError) as error:
            # (The bundles' shaders still precompile: eboot.bin's own are left to the first frames that use them.)
            self.embedded = []
            self.failures.append(('eboot.bin', str(error)))
            print(f'eboot.bin: {error}; its embedded shaders are not precompiled', file=sys.stderr)
        for shader in self.embedded:
            self._add(shader['code'], shader['agc'])
        self.materials = materials.techniques(self.game)

    def _add(self, code, header):
        known = self.headers[code]
        if not any(h['header_size'] == header['header_size'] and h == header for h in known):
            known.append(header)

    def graphics_pairs(self):
        """(VS code, VS header, PS code, PS header, pass id or None, material flags or None) for every
        technique of every material, then the adjacent (Gs, Ps) entries of the other bundles."""
        out, covered = [], set()
        for bundle, (flags, technique_list) in self.materials.items():
            entries = self.bundles.get(bundle)
            if entries is None:
                continue
            layout = materials.technique_entries(entries, technique_list)
            if layout is None:
                continue
            covered.add(bundle)
            for pass_id, kind, indices in layout:
                if kind != materials.KIND_PAIR:
                    continue
                (vs_code, vs_agc), (ps_code, ps_agc) = entries[indices[0]], entries[indices[1]]
                if vs_agc['type'] == agc.GS and ps_agc['type'] == agc.PS:
                    out.append((vs_code, vs_agc, ps_code, ps_agc, pass_id, flags))
        for bundle, entries in self.bundles.items():
            if bundle in covered:
                continue
            for (vs_code, vs_agc), (ps_code, ps_agc) in zip(entries, entries[1:]):
                if vs_agc['type'] == agc.GS and ps_agc['type'] == agc.PS:
                    out.append((vs_code, vs_agc, ps_code, ps_agc, None, None))
        return out

    def engine_pairs(self):
        """The engine's own passes (post-processing, UI, particles: coredata/enginesupport/shaders/<effect>/),
        one shader per bundle, paired by the engine's code: every vertex shader with every pixel shader of
        the same effect directory, the full-screen vertex shader with the pixel shaders of directories
        without one, and the shaders embedded in eboot.bin with each other."""
        by_dir = collections.defaultdict(lambda: ([], []))
        for bundle, entries in self.bundles.items():
            parts = [p.lower() for p in bundle.relative_to(self.game).parts]
            if tuple(parts[:3]) != ENGINE_SHADERS:
                continue
            for code, header in entries:
                if header['type'] in (agc.GS, agc.PS):
                    by_dir[bundle.parent.name.lower()][header['type'] == agc.PS].append((code, header))
        out = []
        fullscreen = by_dir.get(FULLSCREEN_VS, ([], []))[0]
        for name, (vertex, pixel) in sorted(by_dir.items()):
            for vs in vertex or fullscreen:
                out += [(vs[0], vs[1], ps[0], ps[1]) for ps in pixel]
        embedded_vs = [(s['code'], s['agc']) for s in self.embedded if s['agc']['type'] == agc.GS]
        embedded_ps = [(s['code'], s['agc']) for s in self.embedded if s['agc']['type'] == agc.PS]
        out += [(vs[0], vs[1], ps[0], ps[1]) for vs in embedded_vs + fullscreen for ps in embedded_ps]
        return out

    def vertex_only(self):
        """Vertex shaders the engine draws without a pixel shader (the eboot's own)."""
        return [(s['code'], s['agc']) for s in self.embedded if s['agc']['type'] == agc.GS]


def game_id(game):
    """The name of the game version's caches (PipelineCacheGameId in pipelineCache.cpp): <title>_<version>."""
    param = json.loads((Path(game) / 'sce_sys' / 'param.json').read_text(encoding='utf-8'))
    return f"{param['titleId']}_{param['contentVersion']}"


def recorded_file(path=None, game=GAME):
    if path:
        return Path(path)
    name = f'{game_id(game)}.shaders'
    files = sorted((REPO / '_PipelineCache' / 'warmup-v2').glob(f'*/{name}'), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f'no recorded warmup cache {name} under _PipelineCache/warmup-v2')
    return files[-1]


def cmd_inventory(args):
    inv = Inventory(args.game)
    types = collections.Counter()
    for code, headers in inv.headers.items():
        for t in {h['type'] for h in headers}:
            types[agc.BINARY_TYPE_NAMES.get(t, t)] += 1
    print(f'bundles: {len(inv.bundles)} (unparsed {len(inv.failures)}), embedded in eboot: {len(inv.embedded)}')
    print(f'distinct shader code: {len(inv.headers)}  by type: {dict(types)}')
    pairs = inv.graphics_pairs()
    print(f'materials: {len(inv.materials)} bundles; graphics pairs: {len(pairs)} '
          f'({len({(p[0], p[2]) for p in pairs})} distinct VS/PS code pairs); engine pairs: {len(inv.engine_pairs())}')


def cmd_check(args):
    """Static compile keys against every recorded record: the program-entry level of coverage."""
    inv = Inventory(args.game)
    wf = warmfile.WarmFile(recorded_file(args.recorded, args.game)).load(verify_checksum=False)
    records = wf.ok_records()
    vs_for_ps = collections.defaultdict(set)
    for p in wf.ok_pipelines():
        if p.compute == warmfile.NO_SHADER and p.pixel != warmfile.NO_SHADER:
            vs_for_ps[p.pixel].add(p.vertex)
    result = collections.Counter()
    for rec in records:
        name = warmfile.STAGE_NAME[rec.stage]
        headers = inv.headers.get(rec.code)
        if not headers:
            result[(name, 'code not shipped')] += 1
            continue
        want = rec.entry_key()
        candidates = []
        for header in headers:
            try:
                if rec.stage == warmfile.ST_COMPUTE:
                    candidates.append(keys.compute_record(rec.code, header, HOST_SUBGROUP_SIZE))
                elif rec.stage == warmfile.ST_VERTEX:
                    candidates += keys.vertex_records(rec.code, header)
                elif rec.stage == warmfile.ST_PIXEL:
                    for vs_index in vs_for_ps.get(rec.index, ()) or (None,):
                        vs_code = records[vs_index].code if vs_index is not None else None
                        for vs_header in inv.headers.get(vs_code, [None]):
                            candidates.append(keys.pixel_record(rec.code, header, vs_header, LOD_STATS_SUBGROUP))
            except ValueError:
                continue
        hit = any(c.entry_key() == want for c in candidates)
        result[(name, 'entry key derived' if hit else 'entry key differs')] += 1
    for (stage, outcome), n in sorted(result.items()):
        print(f'{stage:8s} {outcome:20s} {n}')


# ---------------------------------------------------------------------------------------------------
# Render-pass states
# ---------------------------------------------------------------------------------------------------
def _state_words(p):
    """A graphics pipeline recipe without its shaders: rendering, vertex input and static state."""
    return warmfile.encode_pipeline(p)[3:]


def _with_shaders(words, vertex, pixel):
    return [vertex, pixel, warmfile.NO_SHADER] + list(words)


def _culling_variants(words):
    """The recipe as recorded and with culling toggled: a material's two-sidedness decides it, and the
    material flag that should say so matches the recorded culling of only four draws in five."""
    p = warmfile.decode_pipeline(_with_shaders(words, 0, 0))
    out = [list(words)]
    s = p.state
    if s['cull_front'] or s['cull_back']:
        s['cull_front'], s['cull_back'] = 0, 0
    else:
        s['cull_back'] = 1
    out.append(_state_words(p))
    return out


def cmd_learn_states(args):
    """The states each material pass draws with (the engine's render-pass setup: targets, blending,
    depth and stencil), and those of the engine's own passes, learned from a recorded cache."""
    inv = Inventory(args.game)
    wf = warmfile.WarmFile(recorded_file(args.recorded, args.game)).load(verify_checksum=False)
    records = wf.records
    pass_of = collections.defaultdict(set)
    for vs_code, _, ps_code, _, pass_id, _ in inv.graphics_pairs():
        pass_of[(vs_code, ps_code)].add(pass_id)
    groups = collections.defaultdict(list)
    for p in wf.ok_pipelines():
        if p.is_compute:
            continue
        if p.pixel == warmfile.NO_SHADER:
            group = 'vertex-only'
        else:
            passes = pass_of.get((records[p.vertex].code, records[p.pixel].code))
            if passes is None:
                group = 'engine'
            elif len(passes) == 1:
                group = str(next(iter(passes)))
            else:
                continue  # a pair drawn in several passes: which one this was is unknown
        words = _state_words(p)
        if words not in groups[group]:
            groups[group].append(words)
    out = {'source': str(recorded_file(args.recorded, args.game)), 'groups': dict(sorted(groups.items()))}
    STATES.write_text(json.dumps(out, separators=(',', ':')))
    print(f'{STATES.name}: ' + ', '.join(f'{g} {len(v)}' for g, v in out['groups'].items()))


# ---------------------------------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------------------------------
class SeedFile:
    """KytyShaderSeeds2: the warmup file layout (records, then pipeline recipes) under its own identity."""

    def __init__(self):
        self.records, self.record_index = [], {}
        self.pipelines, self.pipeline_index = [], set()

    def record(self, rec):
        if not warmfile.valid_key(rec):
            raise ValueError('static key does not match the input info')
        words = warmfile.encode_record(rec)
        blob = array.array('I', words).tobytes()
        index = self.record_index.get(blob)
        if index is None:
            index = self.record_index[blob] = len(self.records)
            self.records.append(blob)
        return index

    def pipeline(self, words):
        blob = array.array('I', words).tobytes()
        if blob not in self.pipeline_index:
            self.pipeline_index.add(blob)
            self.pipelines.append(blob)

    def write(self, path):
        parts = [struct.pack('<I', len(self.records))]
        for blob in self.records:
            parts += [struct.pack('<I', len(blob) // 4), blob]
        parts.append(struct.pack('<I', len(self.pipelines)))
        for blob in self.pipelines:
            parts += [struct.pack('<I', len(blob) // 4), blob]
        body = b''.join(parts)
        Path(path).write_bytes(SEED_IDENTITY + struct.pack('<Q', xxh3.xxh3_64(body)) + body)
        return len(body)


def cmd_seeds(args):
    begin = time.time()
    states = json.loads(STATES.read_text())['groups']
    inv = Inventory(args.game)
    seeds = SeedFile()
    counts = collections.Counter()
    stages = set(args.stages.split(','))

    if 'cs' in stages:
        for code, headers in inv.headers.items():
            for header in headers:
                if header['type'] == agc.CS:
                    seeds.record(keys.compute_record(code, header, HOST_SUBGROUP_SIZE))
                    counts['compute programs'] += 1

    def graphics(vs_code, vs_agc, ps_code, ps_agc, group, culling=True):
        palette = states.get(group)
        if palette is None:
            counts[f'pairs without states ({group})'] += 1
            return
        try:
            vertex = [seeds.record(record) for record in keys.vertex_records(vs_code, vs_agc)]
            ps = seeds.record(keys.pixel_record(ps_code, ps_agc, vs_agc, LOD_STATS_SUBGROUP))
        except ValueError:
            counts['pairs with unsupported inputs'] += 1
            return
        for vs in vertex:
            for words in palette:
                for variant in _culling_variants(words) if culling else [words]:
                    seeds.pipeline(_with_shaders(variant, vs, ps))

    if 'gfx' in stages:
        done = set()
        for vs_code, vs_agc, ps_code, ps_agc, pass_id, _ in inv.graphics_pairs():
            key = (vs_code, ps_code, pass_id)
            if key not in done:
                done.add(key)
                graphics(vs_code, vs_agc, ps_code, ps_agc, str(pass_id))
        for vs_code, vs_agc, ps_code, ps_agc in inv.engine_pairs():
            graphics(vs_code, vs_agc, ps_code, ps_agc, 'engine', culling=False)
        for vs_code, vs_agc in inv.vertex_only():
            for record in keys.vertex_records(vs_code, vs_agc):
                vs = seeds.record(record)
                for words in states.get('vertex-only', []):
                    seeds.pipeline(_with_shaders(words, vs, warmfile.NO_SHADER))

    if args.limit:
        # A smoke test: the first programs and the pipelines that only use them.
        keep = args.limit
        seeds.records = seeds.records[:keep]
        seeds.pipelines = [b for b in seeds.pipelines
                           if all(i == warmfile.NO_SHADER or i < keep for i in struct.unpack_from('<2I', b))]
    size = seeds.write(args.out)
    by_stage = collections.Counter(warmfile.STAGE_NAME[struct.unpack_from('<I', b)[0]] for b in seeds.records)
    print(f'{args.out}: {len(seeds.records)} programs {dict(by_stage)}, {len(seeds.pipelines)} graphics pipelines, '
          f'{size / 2**20:.1f} MiB, {time.time() - begin:.0f} s')
    for what, n in sorted(counts.items()):
        print(f'  {what}: {n}')


# ---------------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------------
def _identity(rec):
    """What selects a compiled permutation at run time: the program entry, the resource specialization
    and the push-data cursor."""
    return rec.entry_key() + (rec.buffers, rec.images, rec.push)


def _pipeline_identity(p, ident):
    if p.is_compute:
        return ('cs', ident[p.compute])
    return ('gfx', ident[p.vertex], ident[p.pixel] if p.pixel != warmfile.NO_SHADER else None,
            p.rendering_norm(), p.vertex_input_norm(), warmfile.state_norm(p))


def _differences(rec, other):
    out = set()
    if rec.push != other.push:
        out.add('push cursor')
    if len(rec.buffers) != len(other.buffers):
        out.add('buffer count')
    else:
        for a, b in zip(rec.buffers, other.buffers):
            out.update(f'buffer {name}' for name, x, y in zip(warmfile.BUFFER_FIELDS, a, b) if x != y)
    if len(rec.images) != len(other.images):
        out.add('image count (indirect table)' if len(rec.images) > len(other.images) else 'image count')
    for a, b in zip(rec.images, other.images):
        out.update(f'image {name}' for name, x, y in zip(warmfile.IMAGE_FIELDS, a, b) if x != y)
    return frozenset(out)


def cmd_recorded_seeds(args):
    """A recorded warmup cache as a seed file (records keep their specialization): compiled the same way,
    it gives the SPIR-V the game's own permutations translate to, for `coverage --recorded-compiled`."""
    raw = recorded_file(args.recorded, args.game).read_bytes()
    body = raw[raw.index(b'\n') + 1:]  # the checksum covers the body only
    Path(args.out).write_bytes(SEED_IDENTITY + body)
    print(f'{args.out}: {len(body) / 2**20:.1f} MiB')


def _spirv(compiled_path):
    """Output record -> SPIR-V hash, from the list a seed run writes next to its warmup file."""
    out = {}
    for line in Path(str(compiled_path) + '.spirv.txt').read_text().split('\n'):
        if line:
            index, digest = line.split()
            out[int(index)] = int(digest, 16)
    return out


def _spirv_coverage(seeded, recorded_compiled):
    """Driver-cache level: a recorded permutation whose SPIR-V equals a seeded one's (and a pipeline whose
    stages' SPIR-V and state equal a seeded pipeline's) is a driver cache hit, whatever its inputs."""
    seeded_files = warmfile.WarmFile(Path(seeded)).load(verify_checksum=False)
    rec_files = warmfile.WarmFile(Path(recorded_compiled)).load(verify_checksum=False)
    seeded_spirv, rec_spirv = _spirv(seeded), _spirv(recorded_compiled)
    have = {(seeded_files.records[i].stage, h) for i, h in seeded_spirv.items()}
    table = collections.defaultdict(collections.Counter)
    for i, rec in enumerate(rec_files.records):
        stage = warmfile.STAGE_NAME[rec.stage]
        table[stage]['same SPIR-V' if (rec.stage, rec_spirv.get(i)) in have else 'other SPIR-V'] += 1
    print(f'\nSPIR-V ({recorded_compiled}: the recorded permutations recompiled)')
    for stage in sorted(table):
        total = sum(table[stage].values())
        print(f'  {stage:8s} {total:5d}: ' + ', '.join(f'{what} {n} ({100 * n / total:.0f}%)'
                                                     for what, n in sorted(table[stage].items())))

    def key(p, spirv):
        if p.is_compute:
            return ('cs', spirv.get(p.compute))
        return ('gfx', spirv.get(p.vertex), spirv.get(p.pixel) if p.pixel != warmfile.NO_SHADER else None,
                p.rendering_norm(), p.vertex_input_norm(), warmfile.state_norm(p))
    seeded_pipelines = {key(p, seeded_spirv) for p in seeded_files.ok_pipelines()}
    result = collections.Counter()
    for p in rec_files.ok_pipelines():
        result[(key(p, rec_spirv)[0], key(p, rec_spirv) in seeded_pipelines)] += 1
    for kind in ('cs', 'gfx'):
        hit, miss = result[(kind, True)], result[(kind, False)]
        print(f'  pipelines {kind:4s} {hit + miss:5d}: driver cache hit {hit} ({100 * hit / max(hit + miss, 1):.0f}%), '
              f'miss {miss}')


def cmd_coverage(args):
    if args.recorded_compiled:
        _spirv_coverage(args.compiled, args.recorded_compiled)
        return
    recorded = warmfile.WarmFile(recorded_file(args.recorded, args.game)).load(verify_checksum=False)
    compiled = warmfile.WarmFile(Path(args.compiled)).load(verify_checksum=False)
    by_entry = collections.defaultdict(list)
    exact = set()
    for rec in compiled.ok_records():
        by_entry[rec.entry_key()].append(rec)
        exact.add(_identity(rec))
    table = collections.defaultdict(collections.Counter)
    reasons = collections.defaultdict(collections.Counter)
    fields = collections.defaultdict(collections.Counter)
    for rec in recorded.ok_records():
        stage = warmfile.STAGE_NAME[rec.stage]
        candidates = by_entry.get(rec.entry_key())
        if not candidates:
            table[stage]['program not seeded'] += 1
        elif _identity(rec) in exact:
            table[stage]['compiled exactly'] += 1
        else:
            table[stage]['other specialization'] += 1
            diff = min((_differences(rec, c) for c in candidates), key=len)
            reasons[stage][diff] += 1
            for d in diff:
                fields[stage][d] += 1
    print(f'recorded: {recorded.path} ({len(recorded.records)} programs, {len(recorded.pipelines)} pipelines)')
    print(f'compiled: {compiled.path} ({len(compiled.records)} programs, {len(compiled.pipelines)} pipelines)\n')
    print('programs')
    for stage in sorted(table):
        total = sum(table[stage].values())
        print(f'  {stage:8s} {total:5d}: ' + ', '.join(f'{what} {n} ({100 * n / total:.0f}%)'
                                                     for what, n in sorted(table[stage].items())))
    print('\nwhat differs from the closest compiled permutation (programs that were seeded)')
    for stage in sorted(fields):
        print(f'  {stage}: ' + ', '.join(f'{d} {n}' for d, n in fields[stage].most_common()))
        for diff, n in reasons[stage].most_common(args.top):
            print(f'      {n:5d}  ' + ' + '.join(sorted(diff)))

    ident = [_identity(r) if r else None for r in compiled.records]
    compiled_pipelines = {_pipeline_identity(p, ident) for p in compiled.ok_pipelines()}
    rec_ident = [_identity(r) if r else None for r in recorded.records]
    # The same recipe with each shader's program entry only: covered once the specializations match.
    entry_only = lambda key: key if key is None else key[:5]  # noqa: E731
    compiled_shapes = {(k[0],) + tuple(entry_only(x) for x in k[1:3]) + k[3:] if k[0] == 'gfx' else ('cs', entry_only(k[1]))
                       for k in compiled_pipelines}
    result = collections.Counter()
    for p in recorded.ok_pipelines():
        key = _pipeline_identity(p, rec_ident)
        kind = key[0]
        if key in compiled_pipelines:
            result[(kind, 'compiled exactly')] += 1
            continue
        shape = (kind,) + tuple(entry_only(x) for x in key[1:3]) + key[3:] if kind == 'gfx' else ('cs', entry_only(key[1]))
        result[(kind, 'same programs and state, other specialization' if shape in compiled_shapes
                else 'not seeded')] += 1
    print('\npipelines')
    for kind in ('cs', 'gfx'):
        total = sum(n for (k, _), n in result.items() if k == kind)
        print(f'  {kind:8s} {total:5d}: ' + ', '.join(f'{what} {n} ({100 * n / max(total, 1):.0f}%)'
                                                     for (k, what), n in sorted(result.items()) if k == kind))


def cmd_runtime_coverage(args):
    """The modules a game run compiled against a precompile's --out: found (its pipelines are in the
    static cache when their state was seeded too), the program seeded with another specialization, or
    the program not seeded."""
    stages = {'vs': 1, 'ms': 1, 'ps': 2, 'cs': 4}
    pattern = re.compile(r'MODULE (\w+) hash=0x([0-9a-f]+) spirv=([0-9a-f]+)')
    run = set()
    for line in Path(args.log).read_text(errors='replace').split('\n'):
        match = pattern.search(line)
        if match:
            run.add((stages[match.group(1)], int(match.group(2), 16), int(match.group(3), 16)))
    compiled = warmfile.WarmFile(Path(args.compiled)).load(verify_checksum=False)
    spirv = _spirv(args.compiled)
    programs, found = set(), set()
    for i, rec in enumerate(compiled.records):
        if rec is not None and i in spirv:
            programs.add((rec.stage, rec.hash))
            found.add((rec.stage, spirv[i]))
    table = collections.defaultdict(collections.Counter)
    for stage, program, module in run:
        table[warmfile.STAGE_NAME[stage]]['found' if (stage, module) in found else
                                          'seeded, other specialization' if (stage, program) in programs else
                                          'not seeded'] += 1
    print(f'{len(run)} modules compiled at run time')
    for stage in sorted(table):
        total = sum(table[stage].values())
        print(f'  {stage:8s} {total:5d}: ' + ', '.join(f'{what} {n} ({100 * n / total:.0f}%)'
                                                     for what, n in sorted(table[stage].items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', default=str(GAME))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inventory').set_defaults(run=cmd_inventory)
    check = sub.add_parser('check')
    check.add_argument('--recorded')
    check.set_defaults(run=cmd_check)
    learn = sub.add_parser('learn-states')
    learn.add_argument('--recorded')
    learn.set_defaults(run=cmd_learn_states)
    seeds = sub.add_parser('seeds')
    seeds.add_argument('out')
    seeds.add_argument('--stages', default='cs,gfx', help='cs, gfx or both (default)')
    seeds.add_argument('--limit', type=int, default=0, help='keep the first N programs (a smoke test)')
    seeds.set_defaults(run=cmd_seeds)
    recorded_seeds = sub.add_parser('recorded-seeds')
    recorded_seeds.add_argument('out')
    recorded_seeds.add_argument('--recorded')
    recorded_seeds.set_defaults(run=cmd_recorded_seeds)
    coverage = sub.add_parser('coverage')
    coverage.add_argument('compiled')
    coverage.add_argument('--recorded')
    coverage.add_argument('--recorded-compiled', help='the recorded cache compiled as seeds: SPIR-V level')
    coverage.add_argument('--top', type=int, default=12)
    coverage.set_defaults(run=cmd_coverage)
    runtime = sub.add_parser('runtime-coverage')
    runtime.add_argument('log', help='run log of a game run with KYTY_SLOW_LOG_MS set (MODULE lines)')
    runtime.add_argument('compiled', help='a precompile --out file (precompile-windows.ps1 -Coverage)')
    runtime.set_defaults(run=cmd_runtime_coverage)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
