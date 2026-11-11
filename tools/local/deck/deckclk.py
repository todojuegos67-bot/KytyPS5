#!/usr/bin/env python3
"""deckclk.py <MHz|auto> ... [--measure 8] [--spin 8]: GPU clock steps on a game left running by `deckbench.py run --keep`
(sudo -n deck-perf.sh gpu <MHz>, which needs passwordless sudo): at each, standing and camera-spin fps (deckbench's
measure), one line a step."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live, measure  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clocks', nargs='+')
    parser.add_argument('--measure', type=int, default=8)
    parser.add_argument('--spin', type=int, default=8)
    parser.add_argument('--settle', type=float, default=4)
    parser.add_argument('--tdp', type=int, default=0, help='also the TDP limit (W) for every step')
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    perf = folder / 'deck-perf.sh'
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    if args.tdp:
        subprocess.run(['sudo', '-n', 'bash', str(perf), 'tdp', str(args.tdp)], check=True, capture_output=True)
    for clock in args.clocks:
        subprocess.run(['sudo', '-n', 'bash', str(perf), 'gpu', clock], check=True, capture_output=True)
        time.sleep(args.settle)
        stand = measure(live, args.measure, f'gpu{clock}-stand')
        line = (f'gpu {clock:>4s}  stand {stand["fps"]:5.1f} (low1 {stand["low1"]:5.1f}, render {stand["render_ms"]:5.1f}, '
                f'gpu {stand["gpu_busy"]:5.1f}% {stand["gpu_mhz"]:.0f} MHz, cpu {stand["cpu_mhz"]:.0f} MHz)')
        if args.spin > 0:
            live.send([f'pad -1 -1 255 128 0 {args.spin + 2}'])
            time.sleep(1)
            spin = measure(live, args.spin, f'gpu{clock}-spin')
            line += f'  spin {spin["fps"]:5.1f} (low1 {spin["low1"]:5.1f}, over50 {spin["over50"]:.0f})'
            time.sleep(1.5)
        print(line, flush=True)


if __name__ == '__main__':
    main()
