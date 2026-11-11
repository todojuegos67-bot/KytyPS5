#!/usr/bin/env python3
"""decksoak.py [--cycles 2] [--worlds "2 1,3 1,4 1,5 1,0 1,1 1"] [--settle 12] [--stay 25]: on a game left running by
`deckbench.py run --keep`, the game's own world changes (live `gamecmd +cp11_loadLevel W L`, as archstone travel) in
turn, each world measured for --stay seconds from --settle seconds after its load; one line a world (fps, p99, slow frames, the emulator's
write-window faults a frame: a fault loop shows there), a screenshot of each in shots/ (soak-<n>.ppm). Stops at the
first world whose measurement fails (a hang or an exit)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deckbench import Live  # noqa: E402
from deckpasses import numbers  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--worlds', default='2 1,3 1,4 1,5 1,0 1,1 1')
    parser.add_argument('--settle', type=int, default=12)
    parser.add_argument('--stay', type=int, default=25)
    parser.add_argument('--dir', default='~/KytyPS5')
    args = parser.parse_args()
    folder = Path(args.dir).expanduser()
    log = max((folder / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    live = Live(folder / 'live-commands.txt', log)
    n = 0
    for cycle in range(args.cycles):
        for world in args.worlds.split(','):
            live.send([f'gamecmd +cp11_loadLevel {world} forceReload=true unlockArchstones=false'])
            time.sleep(args.settle)
            fps, counters = {}, {}
            for line in live.send([f'measure {args.stay} soak{n}'], timeout=args.stay + 90):
                if line.startswith('LIVE_MEASURE'):
                    fps = numbers(line)
                elif line.startswith('LIVE_COUNTERS'):
                    counters = numbers(line)
            live.send([f'screenshot {folder}/shots/soak-{n}.ppm'], timeout=30)
            print(f'cycle {cycle} world {world}: fps={fps.get("fps", 0):.1f} p99={fps.get("p99_ms", 0):.0f} '
                  f'over50={fps.get("over50", 0):g} over100={fps.get("over100", 0):g} '
                  f'window_faults={counters.get("window_faults", 0):.0f} read_faults={counters.get("read_faults", 0):.0f}',
                  flush=True)
            if not fps.get('frames'):
                print('no frames: stopped')
                return
            n += 1


if __name__ == '__main__':
    main()
