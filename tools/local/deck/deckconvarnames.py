#!/usr/bin/env python3
"""sudo -n python3 deckconvarnames.py REGEX: the identifier-like strings in the running game's executable image (guest
0x900000000-0x904000000 read from /proc/<pid>/mem: the eboot's convar names among them) that match REGEX (case
ignored), sorted, one a line. E.g. "^(r_|GI|fog_)" lists the engine's render settings for KYTY_GAME_CONVARS."""
import re
import subprocess
import sys


def main():
    pid = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()[0]
    names = set()
    with open(f'/proc/{pid}/maps') as maps, open(f'/proc/{pid}/mem', 'rb', 0) as mem:
        for line in maps:
            fields = line.split()
            low, high = (int(x, 16) for x in fields[0].split('-'))
            if low < 0x900000000 or high > 0x904000000 or 'r' not in fields[1]:
                continue
            try:
                mem.seek(low)
                data = mem.read(high - low)
            except OSError:
                continue
            for match in re.finditer(rb'[A-Za-z_][A-Za-z0-9_]{5,60}\x00', data):
                names.add(match.group()[:-1].decode())
    pattern = re.compile(sys.argv[1], re.I)
    for name in sorted(names):
        if pattern.search(name):
            print(name)


if __name__ == '__main__':
    main()
