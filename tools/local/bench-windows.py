#!/usr/bin/env python3
"""Windows benchmark helper: swap in the fixed baseline save, talk to the live command file.

    python tools/local/bench-windows.py prepare            # user save -> backup, baseline save in
    python tools/local/bench-windows.py reset              # fresh baseline before each run
    python tools/local/bench-windows.py live measure 20 ref # run live commands, print LIVE_* lines
    python tools/local/bench-windows.py restore            # baseline out, user save back (verified)

The user's save is only ever moved (renamed), never deleted; `restore` refuses to run while an
emulator process exists and checks every file's SHA-256 against the backup manifest.
KYTY_BENCH_BASELINE=<dir> (with PPSA01341 and .kyty-capacity): another baseline for prepare/reset,
such as a copy of a save elsewhere in the game.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAVES = ROOT / '_SaveData'
BASELINE = Path(os.environ.get('KYTY_BENCH_BASELINE') or ROOT / '_Build/walk-fps-20260914/save-baseline')
STATE = ROOT / '_Build/windows-bench'
LIVE_FILE = STATE / 'live-commands.txt'
# The save and the emulator's bookkeeping for it: the slots' capacity, texts (.kyty-param) and backups (.kyty-backup,
# libs/saveDataBackup.h): a run's backups of the bench save must not stay among the user's, where a broken save
# would get the bench save back.
ENTRIES = ('PPSA01341', '.kyty-capacity', '.kyty-param', '.kyty-backup')


def digest(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def emulator_running():
    # A process stuck in driver teardown after a crash stays listed with status Unknown.
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq kyty_emulator.exe', '/FI', 'STATUS eq RUNNING', '/NH'],
                         capture_output=True, text=True).stdout
    return 'kyty_emulator.exe' in out


def prepare():
    if emulator_running():
        sys.exit('an emulator process is running; close it first')
    active = STATE / 'active.json'
    if active.exists():
        sys.exit(f'a baseline save is already installed ({active}); run restore first')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    backup = STATE / f'user-save-{stamp}'
    backup.mkdir(parents=True)
    manifest = {'stamp': stamp, 'backup': str(backup), 'entries': {}}
    moved = []
    try:
        for name in ENTRIES:
            source = SAVES / name
            if source.exists():
                manifest['entries'][name] = digest(source) if source.is_dir() else None
                source.rename(backup / name)  # a move on the same volume: nothing is deleted
                moved.append(name)
    except OSError:
        # (A folder another program holds, e.g. 10-20 .kyty-capacity: the ones moved go back, the user's save in place.)
        for name in reversed(moved):
            (backup / name).rename(SAVES / name)
        backup.rmdir()
        raise
    for name in ENTRIES:
        if (BASELINE / name).exists():
            shutil.copytree(BASELINE / name, SAVES / name)
    manifest['baseline'] = digest(SAVES / 'PPSA01341')
    active.write_text(json.dumps(manifest, indent=1))
    print(f'user save moved to {backup}; baseline save installed')


def reset():
    """Fresh baseline save for the next run (the game autosaves over it while playing)."""
    if emulator_running():
        sys.exit('an emulator process is running; close it first')
    if not (STATE / 'active.json').exists():
        sys.exit('no baseline save is installed; run prepare first')
    used = STATE / f'baseline-used-{time.strftime("%Y%m%d-%H%M%S")}'
    used.mkdir(parents=True)
    for name in ENTRIES:
        if (SAVES / name).exists():
            (SAVES / name).rename(used / name)
        if (BASELINE / name).exists():
            shutil.copytree(BASELINE / name, SAVES / name)
    print('baseline save reinstalled')


def restore():
    if emulator_running():
        sys.exit('an emulator process is running; close it first')
    active = STATE / 'active.json'
    if not active.exists():
        sys.exit('no baseline save is installed')
    manifest = json.loads(active.read_text())
    backup = Path(manifest['backup'])
    used = STATE / f'baseline-used-{manifest["stamp"]}'
    # A reset in the same second as prepare took that name already: keep both copies.
    if any((used / name).exists() for name in ENTRIES):
        used = STATE / f'baseline-used-{manifest["stamp"]}-restore-{time.strftime("%Y%m%d-%H%M%S")}'
    used.mkdir(parents=True, exist_ok=True)
    for name in ENTRIES:
        if (SAVES / name).exists():
            (SAVES / name).rename(used / name)  # the benchmark's copy is kept, not deleted
    for name, files in manifest['entries'].items():
        (backup / name).rename(SAVES / name)
        if files is not None and digest(SAVES / name) != files:
            sys.exit(f'restored {name} does not match its manifest')
    active.rename(STATE / f'restored-{manifest["stamp"]}.json')
    print(f'user save restored and verified ({sum(len(f or {}) for f in manifest["entries"].values())} files)')


IMAGE_BASE = 0x140000000  # kyty_emulator.exe links with /DYNAMICBASE:NO


def symbol_address(name):
    """Runtime address of a global from the lld map (RVA + fixed image base)."""
    map_file = next((ROOT / '_Build/windows').glob('kyty_emulator_*.map'))
    for line in map_file.open(errors='replace'):
        fields = line.split()
        if len(fields) >= 4 and fields[-1] == name:
            return IMAGE_BASE + int(fields[0], 16)
    sys.exit(f'{name} not found in {map_file}')


def expand(line):
    # "set NAME VALUE" -> "poke32 <address> <hex value>" (a 32-bit switch such as kyty_local_*).
    parts = line.split()
    if len(parts) == 3 and parts[0] == 'set':
        return f'poke32 {symbol_address(parts[1]):x} {int(parts[2], 0):x}'
    return line


def live(command_lines, timeout=900):
    command_lines = [expand(line) for line in command_lines]
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = max((ROOT / '_Build/run-logs').glob('*.out.log'), key=lambda p: p.stat().st_mtime)
    identifier = int(time.time() * 1000)
    LIVE_FILE.write_text(f'id {identifier}\n' + '\n'.join(command_lines) + '\n')
    deadline = time.monotonic() + timeout
    next_check = time.monotonic() + 5
    while time.monotonic() < deadline:
        text = log.read_text(errors='replace')
        if f'LIVE_DONE id={identifier}' in text:
            for line in text.splitlines():
                if f'id={identifier}' in line and not line.startswith('LIVE_DONE'):
                    print(line)
            return
        # An emulator that died (a crash) never answers: give up at once instead of at the timeout.
        if time.monotonic() >= next_check:
            next_check = time.monotonic() + 5
            tasks = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq kyty_emulator.exe', '/NH'],
                                   capture_output=True, text=True).stdout
            if 'kyty_emulator.exe' not in tasks:
                sys.exit(f'no LIVE_DONE for id {identifier}: the emulator is not running ({log})')
        time.sleep(0.5)
    sys.exit(f'no LIVE_DONE for id {identifier} in {log}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'restore':
        restore()
    elif sys.argv[1] == 'reset':
        reset()
    elif sys.argv[1] == 'live':
        # One command per argument group separated by ';' (e.g. "measure 20 a ; measure 20 b").
        live([part.strip() for part in ' '.join(sys.argv[2:]).split(';') if part.strip()])
    else:
        sys.exit(__doc__)
