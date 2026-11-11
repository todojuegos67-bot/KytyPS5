#!/usr/bin/env python3
"""deckpasses.py [--seconds 9] [--speed 2] x,y,z x,y,z ...: on a game left running by `deckbench.py run --keep`, the
player glides to each point in turn (live glide), each leg measured (live measure): one line a leg, then the counters
whose per-frame values differ by more than a quarter between the first leg and the last (first-time work in a fresh
boot against the same route again)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live  # noqa: E402


def numbers(line):
    out = {}
    for item in line.split()[1:]:
        key, _, value = item.partition('=')
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('points', nargs='+')
    parser.add_argument('--seconds', type=int, default=9)
    parser.add_argument('--speed', type=float, default=2)
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    legs = []
    for index, point in enumerate(args.points):
        x, y, z = point.split(',')
        live.send([f'glide {x} {y} {z} {args.speed}'])
        time.sleep(0.3)
        fps, counters = {}, {}
        for line in live.send([f'measure {args.seconds} leg{index}'], timeout=args.seconds + 60):
            if line.startswith('LIVE_MEASURE'):
                fps = numbers(line)
            elif line.startswith('LIVE_COUNTERS'):
                counters = numbers(line)
        legs.append(counters)
        print(f'leg {index} to {point}: ' + ' '.join(f'{k}={v:g}' for k, v in fps.items() if k in
                                                     ('fps', 'frames', 'render_ms', 'max_ms', 'p99_ms')), flush=True)
        time.sleep(1)
    first, last = legs[0], legs[-1]
    print(f'{"counter":30s} {"first leg":>12s} {"last leg":>12s}')
    for key in first:
        a, b = first.get(key, 0.0), last.get(key, 0.0)
        if max(a, b) > 0.5 and abs(a - b) > 0.25 * max(1.0, b):
            print(f'{key:30s} {a:12.1f} {b:12.1f}')


if __name__ == '__main__':
    main()
