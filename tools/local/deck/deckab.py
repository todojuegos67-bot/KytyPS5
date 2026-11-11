#!/usr/bin/env python3
"""deckab.py "label|+name=value;+name2=value2;!switch=1" ...: same-boot steps on a game left running by
`deckbench.py run --keep`.

Each step sends its engine commands (live `gamecmd`, as KYTY_GAME_CONVARS lines: the steps add up) and sets its
emulator live switches (!name=value, live `set`), waits --settle,
measures --measure seconds (fps, render thread) and traces --tracem seconds of GPU time per draw and dispatch, then
prints one line: fps, the GPU's busy ms per frame (command buffer spans) and its change from the first step. The game
keeps running afterwards (deckbench.py stop ends it)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live, measure  # noqa: E402


def gpu_ms(trace, tools):
    out = subprocess.run([sys.executable, str(tools / 'live-trace.py'), 'marks', str(trace), '--top', '1'],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if 'GPU busy (command buffer spans):' in line:
            return float(line.split(':')[1].split()[0])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('steps', nargs='+')
    parser.add_argument('--dir', default='~/KytyPS5')
    parser.add_argument('--settle', type=float, default=6)
    parser.add_argument('--measure', type=int, default=10)
    parser.add_argument('--tracem', type=float, default=2)
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    live.offset = log.stat().st_size  # (answers from the end of the log on)
    first = None
    for step in args.steps:
        label, _, commands = step.partition('|')
        for command in filter(None, (c.strip() for c in commands.split(';'))):
            # !name=value: an emulator live switch (live-switch.h); else an engine command.
            if command.startswith('!'):
                name, _, value = command[1:].partition('=')
                live.send([f'set {name} {value}'])
            else:
                live.send([f'gamecmd {command}'])
        time.sleep(args.settle)
        result = measure(live, args.measure, label)
        trace = folder / 'prof' / f'ab-{label}.trace'
        trace.parent.mkdir(exist_ok=True)
        for line in live.send([f'tracem {args.tracem} {trace}'], timeout=args.tracem + 120):
            if line.startswith('LIVE_TRACE') and 'tsc_hz=' in line:
                Path(f'{trace}.hz').write_text(line.split('tsc_hz=')[1].split()[0])
                if 'base=' in line:
                    Path(f'{trace}.base').write_text(line.split('base=')[1].split()[0])
        busy = gpu_ms(trace, folder)
        first = busy if first is None else first
        delta = f'{busy - first:+6.2f}' if busy is not None and first is not None else '   ?'
        print(f'{label:24s} {result.get("fps", 0):5.1f} fps  render {result.get("render_ms", 0):5.1f} ms  '
              f'GPU {busy if busy is not None else 0:6.2f} ms/frame ({delta})  gpu {result.get("gpu_busy")}% '
              f'{result.get("gpu_mhz")} MHz {result.get("power_w")} W', flush=True)
        print('STEP ' + json.dumps({**result, 'gpu_ms': busy, 'commands': commands}), file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
