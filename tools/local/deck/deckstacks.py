#!/usr/bin/env python3
"""deckstacks.py <samples.prof> <kyty_emulator> [--libs dir] [--top 40] [--depth 3]: a live `prof`/`profw` capture of
the render thread (16 words a sample: pc, the word at rsp, up to 14 frame-pointer return addresses) by leaf function,
and by its first callers in the emulator: libc and driver leaves (syscalls, waits) named after the emulator code that
reached them. The .maps file next to the samples (a copy of /proc/<pid>/maps) places each address."""
import argparse
import bisect
import collections
import os
import struct
import subprocess

WORDS = 16


def symbols(path, dynamic=False):
    command = ['nm', '-C', '--defined-only', '-n'] + (['-D'] if dynamic else []) + [path]
    out = subprocess.run(command, capture_output=True, text=True).stdout
    addresses, names = [], []
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in 'tTwW':
            addresses.append(int(parts[0], 16))
            names.append(parts[2])
    return addresses, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples')
    parser.add_argument('exe')
    parser.add_argument('--libs', default='')
    parser.add_argument('--top', type=int, default=40)
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()
    maps = []
    for line in open(args.samples + '.maps'):
        parts = line.split(maxsplit=5)
        start, end = (int(x, 16) for x in parts[0].split('-'))
        maps.append((start, end, int(parts[2], 16), parts[5].strip() if len(parts) > 5 else ''))
    maps.sort()
    starts = [m[0] for m in maps]
    bases = {}
    for start, end, offset, path in maps:
        if path:
            bases[path] = min(bases.get(path, start - offset), start - offset)
    exe_symbols = symbols(args.exe)
    library_symbols = {}

    def name(address):
        at = bisect.bisect_right(starts, address) - 1
        if at < 0 or address >= maps[at][1]:
            return None, 'guest' if 0x800000000 <= address < 0x10000000000 else '?'
        path = maps[at][3]
        if not path:
            return None, 'guest' if 0x800000000 <= address < 0x10000000000 else 'anon'
        offset = address - bases[path]
        if path.endswith('kyty_emulator'):
            addresses, names = exe_symbols
            j = bisect.bisect_right(addresses, offset) - 1
            return 'emulator', names[j] if j >= 0 else '?'
        base_name = os.path.basename(path)
        if base_name not in library_symbols:
            copy = os.path.join(args.libs, base_name) if args.libs else ''
            library_symbols[base_name] = symbols(copy, True) if copy and os.path.exists(copy) else ([], [])
        addresses, names = library_symbols[base_name]
        j = bisect.bisect_right(addresses, offset) - 1
        label = names[j] if j >= 0 else ''
        return base_name, f'{base_name} {label}+{offset:#x}' if label else f'{base_name}+{offset:#x}'

    data = open(args.samples, 'rb').read()
    total = len(data) // (8 * WORDS)
    leaves, chains = collections.Counter(), collections.Counter()
    for i in range(total):
        words = struct.unpack_from(f'<{WORDS}Q', data, i * 8 * WORDS)
        module, leaf = name(words[0])
        if module not in (None, 'emulator'):
            leaf = leaf.split('+')[0]
        leaves[leaf[:120]] += 1
        frames = [leaf[:60]]
        for word in words[1:]:
            if word == 0 or len(frames) > args.depth:
                break
            m, n = name(word)
            if m == 'emulator':
                frames.append(n[:70])
        chains[' <- '.join(frames)] += 1
    print(f'{total} samples')
    print('leaves:')
    for leaf, n in leaves.most_common(args.top):
        print(f'  {100 * n / total:5.1f}%  {leaf}')
    print('chains:')
    for chain, n in chains.most_common(args.top):
        print(f'  {100 * n / total:5.1f}%  {chain}')


if __name__ == '__main__':
    main()
