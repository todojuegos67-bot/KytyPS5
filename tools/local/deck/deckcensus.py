#!/usr/bin/env python3
"""deckcensus.py <run.census> <kyty_emulator> [--kind 13] [--top 30]: a live `census` capture (deckbench.py run --census)
with its callers symbolized by `nm -C` of the same build. Kind 13 = the code that recorded each vkCmdPipelineBarrier(2)
(src/local/live-census.h has the kinds); without --kind, every kind's total per frame."""
import argparse
import bisect
import collections
import subprocess

KINDS = {0: 'Dispatch', 1: 'Draw', 3: 'DispatchPhase', 4: 'GpuWait', 5: 'ReadbackWait', 6: 'SyncDownload',
         7: 'GraphicsPrograms', 8: 'NativeGather', 9: 'QueueRun', 10: 'SrtInterpreter', 11: 'CommandSync',
         12: 'DrawPhase', 13: 'Barrier'}
CALLER_KINDS = {4, 5, 13}


def symbols(exe):
    out = subprocess.run(['nm', '-C', '--defined-only', '-n', exe], capture_output=True, text=True, check=True).stdout
    addresses, names = [], []
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in 'tTwW':
            addresses.append(int(parts[0], 16))
            names.append(parts[2])
    return addresses, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('census')
    parser.add_argument('exe')
    parser.add_argument('--kind', type=int, default=-1)
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()
    base, frames, entries = 0, 1, []
    for line in open(args.census):
        fields = dict(item.split('=', 1) for item in line.split()[1:] if '=' in item)
        if line.startswith('LIVE_CENSUS '):
            base, frames = int(fields['base'], 16), int(fields['frames'])
        elif line.startswith('LIVE_CENSUS_ENTRY'):
            entries.append((int(fields['kind']), int(fields['a'], 16), int(fields['b'], 16),
                            float(fields['calls_per_frame']), float(fields['ms_per_frame'])))
    print(f'{frames} frames')
    if args.kind < 0:
        totals = collections.defaultdict(lambda: [0.0, 0.0])
        for kind, a, b, calls, ms in entries:
            totals[kind][0] += calls
            totals[kind][1] += ms
        for kind, (calls, ms) in sorted(totals.items(), key=lambda item: -item[1][1]):
            print(f'  {str(KINDS.get(kind, kind)):18s} {calls:9.1f} calls/frame {ms:8.3f} ms/frame')
        return
    addresses, names = symbols(args.exe)
    rows = collections.Counter()
    times = collections.Counter()
    for kind, a, b, calls, ms in entries:
        if kind != args.kind:
            continue
        if kind in CALLER_KINDS:
            i = bisect.bisect_right(addresses, a - base) - 1
            key = names[i] if i >= 0 else hex(a)
            if kind == 13:
                key += ' (2)' if b == 1 else ''
        else:
            key = f'{a:#x} {b:#x}'
        rows[key] += calls
        times[key] += ms
    for key, calls in rows.most_common(args.top):
        print(f'  {calls:8.1f}/frame {times[key]:7.3f} ms  {key[:160]}')


if __name__ == '__main__':
    main()
