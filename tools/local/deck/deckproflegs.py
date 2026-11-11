#!/usr/bin/env python3
"""deckproflegs.py TID x,y,z x,y,z ... [--seconds 9] [--speed 2] [--label legs]: on a game left running by
`deckbench.py run --keep`, the player glides to each point in turn, each leg measured (live measure) while `perf
record` samples thread TID (the render thread: cpu-clock 1 kHz and context switches, frame-pointer call chains; sudo)
into prof/<label>-<leg>.data. Compare the first leg's profile with a later leg over the same ground (first-visit work)."""
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
    parser.add_argument('tid')
    parser.add_argument('points', nargs='+')
    parser.add_argument('--seconds', type=int, default=9)
    parser.add_argument('--speed', type=float, default=2)
    parser.add_argument('--label', default='legs')
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    (folder / 'prof').mkdir(exist_ok=True)
    for index, point in enumerate(args.points):
        data = folder / 'prof' / f'{args.label}-{index}.data'
        perf = subprocess.Popen(['sudo', '-n', 'timeout', '-s', 'INT', str(args.seconds + 1), 'perf', 'record', '-t',
                                 args.tid, '-e', 'cpu-clock', '-F', '1000', '-e', 'sched:sched_switch', '--call-graph',
                                 'fp', '-o', str(data)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        x, y, z = point.split(',')
        live.send([f'glide {x} {y} {z} {args.speed}'])
        fps = {}
        for line in live.send([f'measure {args.seconds} {args.label}{index}'], timeout=args.seconds + 60):
            if line.startswith('LIVE_MEASURE'):
                fps = numbers(line)
        perf.wait()
        subprocess.run(['sudo', '-n', 'chown', 'deck', str(data)])
        print(f'leg {index} to {point}: ' + ' '.join(f'{k}={v:g}' for k, v in fps.items() if k in
                                                     ('fps', 'frames', 'render_ms', 'max_ms', 'p99_ms', 'over50')),
              flush=True)
        time.sleep(1)


if __name__ == '__main__':
    main()
