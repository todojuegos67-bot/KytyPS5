#!/usr/bin/env python3
"""deckglideab.py [--seconds 9] [--speed 2] [--rounds 3] A B "name=a|name=b": on a game left running by
`deckbench.py run --keep`, the player glides from point A to point B and back, each leg measured (live measure) with
the live switches of its variant (live `set`), the variants alternating leg by leg (the first leg is the route's
first pass: left out). One line a leg, then each variant's mean fps, render ms, p99 and slow frames."""
import argparse
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
    parser.add_argument('variants', help='switches of each variant: "name=1;other=2|name=0"')
    parser.add_argument('--seconds', type=int, default=9)
    parser.add_argument('--speed', type=float, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    variants = [[item for item in variant.split(';') if item] for variant in args.variants.split('|')]
    legs = [(args.b, None)] + [(args.a if i % 2 == 0 else args.b, i % len(variants)) for i in range(args.rounds * len(variants) * 2)]
    results = {}
    for index, (point, variant) in enumerate(legs):
        for item in variants[variant if variant is not None else 0]:
            name, _, value = item.partition('=')
            live.send([f'set {name} {value}'])
        x, y, z = point.split(',')
        live.send([f'glide {x} {y} {z} {args.speed}'])
        time.sleep(0.3)
        fps = {}
        for line in live.send([f'measure {args.seconds} leg{index}'], timeout=args.seconds + 60):
            if line.startswith('LIVE_MEASURE'):
                fps = numbers(line)
        label = 'first' if variant is None else ';'.join(variants[variant])
        print(f'leg {index:2d} {label:24s} fps={fps.get("fps", 0):.2f} render={fps.get("render_ms", 0):.2f} '
              f'p99={fps.get("p99_ms", 0):.1f} max={fps.get("max_ms", 0):.1f} over50={fps.get("over50", 0):g}', flush=True)
        if variant is not None:
            results.setdefault(variant, []).append(fps)
        time.sleep(1)
    for variant, runs in sorted(results.items()):
        mean = lambda key: sum(r.get(key, 0) for r in runs) / len(runs)  # noqa: E731
        print(f'{";".join(variants[variant]):24s} legs={len(runs)} fps={mean("fps"):.2f} render={mean("render_ms"):.2f} '
              f'p99={mean("p99_ms"):.1f} over50={sum(r.get("over50", 0) for r in runs):g}')


if __name__ == '__main__':
    main()
