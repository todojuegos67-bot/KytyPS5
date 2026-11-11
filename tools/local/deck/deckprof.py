#!/usr/bin/env python3
"""deckprof.py <samples.prof> <kyty_emulator> [--top 40]: a live `profp` capture (process CPU time, 4000 samples/s;
deckbench.py run --prof) by thread, module and function. The .maps file next to the samples places each pc: the
emulator (symbolized with `nm -C` of the same build), the game's code (the guest's own mappings), the GPU driver and
other libraries. <samples>.threads (thread id and name per line, deckbench.py writes it) names the threads."""
import argparse
import bisect
import collections
import os
import struct
import subprocess

WORDS = 16


def load_maps(path):
    maps = []
    with open(path) as f:
        for line in f:
            parts = line.split(maxsplit=5)
            start, end = (int(x, 16) for x in parts[0].split('-'))
            maps.append((start, end, int(parts[2], 16), parts[5].strip() if len(parts) > 5 else ''))
    maps.sort()
    return maps


def load_symbols(exe, dynamic=False):
    command = ['nm', '-C', '--defined-only', '-n'] + (['-D'] if dynamic else []) + [exe]
    out = subprocess.run(command, capture_output=True, text=True).stdout
    addresses, names = [], []
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in 'tTwW':
            addresses.append(int(parts[0], 16))
            names.append(parts[2])
    return addresses, names


def module_of(path, pc):
    if not path:
        return 'guest/anonymous' if 0x800000000 <= pc < 0x10000000000 else 'anonymous'
    base = os.path.basename(path)
    for key, label in (('kyty_emulator', 'emulator'), ('libvulkan_radeon', 'RADV driver'), ('libc.so', 'libc'),
                       ('libstdc++', 'libstdc++'), ('libm.so', 'libm'), ('libpipewire', 'pipewire'), ('libpulse', 'pulse')):
        if key in base:
            return label
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples')
    parser.add_argument('exe')
    parser.add_argument('--top', type=int, default=40)
    parser.add_argument('--libs', default='', help='a folder with copies of the libraries the maps name (by file name)')
    args = parser.parse_args()
    maps = load_maps(args.samples + '.maps')
    starts = [m[0] for m in maps]
    exe_base = min((m[0] - m[2] for m in maps if m[3].endswith('kyty_emulator')), default=0)
    addresses, names = load_symbols(args.exe)
    # Libraries with a local copy (--libs): their exported symbols, at the base each was loaded at.
    libraries = {}
    if args.libs:
        for start, end, offset, path in maps:
            base_name = os.path.basename(path)
            copy = os.path.join(args.libs, base_name)
            if path and base_name not in libraries and os.path.exists(copy) and 'kyty_emulator' not in base_name:
                base = min(m[0] - m[2] for m in maps if m[3] == path)
                libraries[base_name] = (base, *load_symbols(copy, dynamic=True))
    threads = {}
    if os.path.exists(args.samples + '.threads'):
        for line in open(args.samples + '.threads'):
            tid, _, name = line.strip().partition(' ')
            threads[int(tid)] = name
    data = open(args.samples, 'rb').read()
    total = len(data) // (8 * WORDS)
    by_thread, by_module, by_function = collections.Counter(), collections.Counter(), collections.Counter()
    by_thread_function = collections.defaultdict(collections.Counter)
    for i in range(total):
        pc, word = struct.unpack_from('<QQ', data, i * 8 * WORDS)
        tid = word & ~(1 << 63)
        thread = f'{threads.get(tid, "?")} ({tid})'
        at = bisect.bisect_right(starts, pc) - 1
        path = maps[at][3] if at >= 0 and pc < maps[at][1] else ''
        module = module_of(path, pc)
        if module == 'emulator':
            j = bisect.bisect_right(addresses, pc - exe_base) - 1
            function = names[j] if j >= 0 else '?'
        elif module.startswith('guest'):
            function = f'guest {pc >> 12 << 12:#x}' if pc < 0x900000000 or pc >= 0xa00000000 else f'eboot+{(pc - 0x900000000) >> 8 << 8:#x}'
        elif os.path.basename(path) in libraries:
            base, lib_addresses, lib_names = libraries[os.path.basename(path)]
            j = bisect.bisect_right(lib_addresses, pc - base) - 1
            function = lib_names[j] if j >= 0 else module
        else:
            function = module
        by_thread[thread] += 1
        by_module[module] += 1
        by_function[(module, function)] += 1
        by_thread_function[thread][(module, function)] += 1
    seconds = total / 4000
    print(f'{total} samples = {seconds:.2f} CPU-seconds')
    print('\nthreads:')
    for thread, n in by_thread.most_common(25):
        print(f'  {100 * n / total:5.1f}%  {thread}')
    print('\nmodules:')
    for module, n in by_module.most_common(12):
        print(f'  {100 * n / total:5.1f}%  {module}')
    print('\nfunctions:')
    for (module, function), n in by_function.most_common(args.top):
        print(f'  {100 * n / total:5.1f}%  [{module}] {function[:150]}')
    for thread, n in by_thread.most_common(6):
        print(f'\n{thread} ({100 * n / total:.1f}%):')
        for (module, function), m in by_thread_function[thread].most_common(12):
            print(f'  {100 * m / n:5.1f}%  [{module}] {function[:140]}')


if __name__ == '__main__':
    main()
