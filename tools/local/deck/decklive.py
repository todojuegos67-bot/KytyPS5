#!/usr/bin/env python3
"""decklive.py "command" ... [--timeout S]: live commands (KYTY_LIVE_FILE) to the emulator running on the Steam Deck
(`deckbench.py run --keep`), its answers printed: e.g. "capture /home/deck/KytyPS5/prof/cap 1", "set name 1",
"gamecmd +r_bloomEnable=false", "measure 5"."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('commands', nargs='+')
    parser.add_argument('--dir', default='~/KytyPS5')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    live.offset = log.stat().st_size
    for line in live.send(args.commands, timeout=args.timeout):
        print(line)


if __name__ == '__main__':
    main()
