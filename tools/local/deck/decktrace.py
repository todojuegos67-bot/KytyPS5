#!/usr/bin/env python3
"""decktrace.py <label> [--seconds 3] [--writes]: the scheduling timeline (live `trace`, `tracew` with --writes) of a
game left running by `deckbench.py run --keep`, as deckbench.py run --trace writes it (prof/<label>.timeline with its
.hz, .base and .threads): `deck.sh trace <label>` reads it."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--writes', action='store_true')
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    timeline = folder / 'prof' / f'{args.label}.timeline'
    timeline.parent.mkdir(exist_ok=True)
    command = 'tracew' if args.writes else 'trace'
    for line in live.send([f'{command} {args.seconds} {timeline}'], timeout=args.seconds + 120):
        if line.startswith('LIVE_TRACE') and 'tsc_hz=' in line:
            Path(f'{timeline}.hz').write_text(line.split('tsc_hz=')[1].split()[0])
            if 'base=' in line:
                Path(f'{timeline}.base').write_text(line.split('base=')[1].split()[0])
    pid = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()[0]
    names = []
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            names.append(f'{task.name} {(task / "comm").read_text().strip()}')
        except OSError:
            pass
    Path(f'{timeline}.threads').write_text(''.join(name + chr(10) for name in names))
    print(timeline)


if __name__ == '__main__':
    main()
