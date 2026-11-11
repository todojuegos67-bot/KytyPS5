#!/usr/bin/env python3
"""deckpokeab.py SYMBOL v1,v2,... [--rounds 2] [--measure 8] [--settle 3]: on a game left running by `deckbench.py run
--keep`, a 32-bit emulator global (a [[gnu::used]] kyty_local_* switch) set live to each value in turn (live `poke32` at
its host address: the executable's load base from /proc/<pid>/maps plus the symbol's offset, from `nm` in the build
container kytybuild), each measured standing (live `measure`), the values alternating for --rounds; one line a step
and each value's means. The first value is left set."""
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
    parser.add_argument('symbol')
    parser.add_argument('values')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--measure', type=int, default=8)
    parser.add_argument('--settle', type=float, default=3)
    parser.add_argument('--exe', default='/src/kyty/_Build/deck/kyty_emulator')
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    nm = subprocess.run(['podman', 'exec', 'kytybuild', 'nm', args.exe], capture_output=True, text=True).stdout
    offset = next(int(line.split()[0], 16) for line in nm.splitlines() if line.split()[-1] == args.symbol)
    pid = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()[0]
    base = None
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[5].endswith('/kyty_emulator') and int(fields[2], 16) == 0:
            base = int(fields[0].split('-')[0], 16)
            break
    address = base + offset
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    values = args.values.split(',')
    results = {}
    for round_index in range(args.rounds):
        for value in values:
            live.send([f'poke32 {address:x} {int(value):x}'])
            time.sleep(args.settle)
            fps = {}
            for line in live.send([f'measure {args.measure} poke{value}'], timeout=args.measure + 60):
                if line.startswith('LIVE_MEASURE'):
                    fps = numbers(line)
            print(f'{args.symbol}={value:>6s} fps={fps.get("fps", 0):.2f} render={fps.get("render_ms", 0):.2f} '
                  f'busy={fps.get("render_busy", 0)} p99={fps.get("p99_ms", 0):.1f}', flush=True)
            results.setdefault(value, []).append(fps)
    live.send([f'poke32 {address:x} {int(values[0]):x}'])
    for value, runs in results.items():
        mean = lambda key: sum(r.get(key, 0) for r in runs) / len(runs)  # noqa: E731
        print(f'{args.symbol}={value:>6s} mean fps={mean("fps"):.2f} render={mean("render_ms"):.2f} p99={mean("p99_ms"):.1f}')


if __name__ == '__main__':
    main()
