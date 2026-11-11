#!/usr/bin/env python3
"""deckclockab.py A B --clocks 1300,1450 [--rounds 2] [--seconds 9]: on a game left running by `deckbench.py run
--keep`, the player glides from A to B and back, the GPU clock set before each leg (deck-perf.sh gpu <MHz>, sudo -n),
the clocks alternating leg by leg (the first leg, the route's first pass, left out); one line a leg, then each clock's
means. The clock is left at the first one."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live  # noqa: E402
from deckpasses import numbers  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--clocks', default='1300,1450')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=9)
    parser.add_argument('--speed', type=float, default=2)
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    clocks = args.clocks.split(',')
    legs = [(args.b, None)] + [(args.a if i % 2 == 0 else args.b, i % len(clocks)) for i in range(args.rounds * len(clocks) * 2)]
    results = {}
    for index, (point, clock) in enumerate(legs):
        mhz = clocks[clock if clock is not None else 0]
        subprocess.run(['sudo', '-n', 'bash', str(folder / 'deck-perf.sh'), 'gpu', mhz], stdout=subprocess.DEVNULL)
        x, y, z = point.split(',')
        live.send([f'glide {x} {y} {z} {args.speed}'])
        time.sleep(0.3)
        fps = {}
        for line in live.send([f'measure {args.seconds} clk{index}'], timeout=args.seconds + 60):
            if line.startswith('LIVE_MEASURE'):
                fps = numbers(line)
        label = 'first' if clock is None else f'{mhz} MHz'
        print(f'leg {index:2d} {label:9s} fps={fps.get("fps", 0):.2f} render={fps.get("render_ms", 0):.2f} '
              f'p99={fps.get("p99_ms", 0):.1f} over50={fps.get("over50", 0):g}', flush=True)
        if clock is not None:
            results.setdefault(mhz, []).append(fps)
        time.sleep(1)
    subprocess.run(['sudo', '-n', 'bash', str(folder / 'deck-perf.sh'), 'gpu', clocks[0]], stdout=subprocess.DEVNULL)
    for mhz, runs in results.items():
        mean = lambda key: sum(r.get(key, 0) for r in runs) / len(runs)  # noqa: E731
        print(f'{mhz} MHz legs={len(runs)} fps={mean("fps"):.2f} render={mean("render_ms"):.2f} p99={mean("p99_ms"):.1f}')


if __name__ == '__main__':
    main()
