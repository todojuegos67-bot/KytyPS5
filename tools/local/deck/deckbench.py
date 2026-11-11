#!/usr/bin/env python3
"""deckbench.py: one measured run of Demon's Souls 01.007.000 on the Steam Deck, for tuning over SSH (deck.sh run).

The folder (--dir, default ~/KytyPS5: the Deck's install, where builds go directly, USER 10-28) holds the emulator
under test and bench-save/ (the PC's 1-1 bench save), copied into the folder's _SaveData before each run: the Deck's
saves there are replaced (USER 10-28: they do not matter). The emulator starts in the desktop session (systemd-run --user:
the session's DISPLAY and WAYLAND_DISPLAY), enters the game with the scripted pad (live `pad`, padboot.py's way),
settles, measures with the Deck's telemetry (GPU busy/clock/power, CPU clocks, memory, swap) and stops.

  deckbench.py run [--set NAME=VALUE ...] [--preset lowest|none] [--measure 20] [--spin 0] [--shot] [--label x]
  deckbench.py stop
"""
import argparse
import collections
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpumetrics  # noqa: E402

HOME = Path.home()
GAME = HOME / 'Games' / 'PPSA01341-app0.zar'
UNIT = 'kyty-bench'
CHR_INS_MAN = 0x9027cf378  # 01.007.000: the player chain (padboot.py)
OPTIONS, CROSS = 0x8, 0x4000
# The launcher's Lowest preset at 540p with the 1080p output (run-deck.sh).
PRESETS = {
    'lowest': {
        'KYTY_RENDER_HEIGHT': '540',
        'KYTY_HD_OUTPUT': '1',
        'KYTY_TEXTURE_REDUCE': '2',
        'KYTY_GAME_CONVARS': 'r_forceViewportHeightForPixelSizeCalc=480;sunShadowMaxDistance=60;'
                             'r_enableAmbientShadowing=false;r_enableSpotShadows=false',
        'KYTY_PRESENT_ASPECT': 'fit',
    },
    'none': {},
}
# 'deck': 'lowest' and the effects whose GPU time the Steam Deck cannot spare (same-boot A/B, deckab.py).
DECK_EFFECTS_OFF = ('r_motionBlurEnable=false;r_dofEnable=false;r_vignetteEnable=false;r_ssrMethod=0;'
                    'r_contactShadowsEnabled=false;r_tessellationEnable=false;r_bloomEnable=false')
PRESETS['deck'] = {**PRESETS['lowest'],
                   'KYTY_GAME_CONVARS': PRESETS['lowest']['KYTY_GAME_CONVARS'] + ';' + DECK_EFFECTS_OFF}
SESSION = {'XDG_RUNTIME_DIR': f'/run/user/{os.getuid()}',
           'DBUS_SESSION_BUS_ADDRESS': f'unix:path=/run/user/{os.getuid()}/bus'}


def plausible(address):
    return 0x100000000 <= address < 0x8000000000


class Live:
    """The emulator's live command file (KYTY_LIVE_FILE) and its answers in the log."""

    def __init__(self, path, log):
        self.path, self.log, self.offset, self.rest, self.ident = Path(path), Path(log), 0, b'', int(time.time())

    def lines(self):
        try:
            with open(self.log, 'rb') as f:
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        data = self.rest + data
        cut = data.rfind(b'\n') + 1
        self.rest = data[cut:]
        return data[:cut].decode('utf-8', errors='replace').splitlines()

    def send(self, commands, timeout=30):
        self.ident += 1
        ident = self.ident
        temp = self.path.with_suffix('.tmp')
        temp.write_text(f'id {ident}\n' + '\n'.join(commands) + '\n')
        os.replace(temp, self.path)
        out, deadline = [], time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in self.lines():
                if f'id={ident}' in line:
                    if line.startswith('LIVE_DONE'):
                        return out
                    out.append(line)
            time.sleep(0.05)
        return out

    def peek(self, address):
        if not plausible(address):
            return 0
        for line in self.send([f'peek {address:x} 8']):
            if line.startswith('LIVE_PEEK') and 'bytes=' in line:
                data = bytes.fromhex(line.split('bytes=')[1].strip())
                return struct.unpack('<Q', data)[0] if len(data) == 8 else 0
        return 0

    def in_game(self):
        p = self.peek(CHR_INS_MAN)
        c = self.peek(p + 0x258) if plausible(p) else 0
        k = self.peek(c + 0x38) if plausible(c) else 0
        return plausible(self.peek(k + 0x20)) if plausible(k) else False

    def position(self):
        # The player's position (H+0x20, three floats), None outside the game.
        p = self.peek(CHR_INS_MAN)
        c = self.peek(p + 0x258) if plausible(p) else 0
        k = self.peek(c + 0x38) if plausible(c) else 0
        h = self.peek(k + 0x20) if plausible(k) else 0
        if not plausible(h):
            return None
        for line in self.send([f'peek {h + 0x20:x} 12']):
            if line.startswith('LIVE_PEEK') and 'bytes=' in line:
                data = bytes.fromhex(line.split('bytes=')[1].strip())
                return struct.unpack('<3f', data) if len(data) == 12 else None
        return None

    def press(self, buttons, held=0.15, after=3.0):
        self.send([f'pad -1 -1 -1 -1 {buttons:x} {held}'])
        time.sleep(held + after)


class Telemetry(threading.Thread):
    """GPU busy %, GPU clock, package power, CPU clocks, memory and swap, the GPU's memory (GTT: the RAM it took, VRAM: the
    1 GiB carved out of it) every half second; with gpu_metrics
    (gpumetrics.py) the power by rail (cpu_w, gfx_w) and the share of samples each SMU limit held the clocks."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()
        device = Path('/sys/class/drm/card0/device')
        self.busy = device / 'gpu_busy_percent'
        self.gtt, self.vram = device / 'mem_info_gtt_used', device / 'mem_info_vram_used'
        self.hwmon = next(device.glob('hwmon/hwmon*'), None)

    @staticmethod
    def read(path, scale=1.0):
        try:
            return int(Path(path).read_text().split()[0]) * scale
        except (OSError, ValueError, IndexError):
            return None

    def run(self):
        while not self.stop.wait(0.5):
            meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            kb = lambda key: int(meminfo[key].split()[0]) / 1024
            cpu = [float(line.split(':')[1]) for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('cpu MHz')]
            self.samples.append({
                'gpu_busy': self.read(self.busy),
                'gpu_mhz': self.read(self.hwmon / 'freq1_input', 1e-6) if self.hwmon else None,
                'power_w': (self.read(self.hwmon / 'power1_average', 1e-6) or self.read(self.hwmon / 'power1_input', 1e-6))
                if self.hwmon else None,
                'cpu_mhz': sum(cpu) / len(cpu) if cpu else None,
                'mem_avail_mb': kb('MemAvailable'),
                'swap_used_mb': kb('SwapTotal') - kb('SwapFree'),
                'gtt_mb': self.read(self.gtt, 1 / 2**20),
                'vram_mb': self.read(self.vram, 1 / 2**20),
                **self.metrics(),
            })

    @staticmethod
    def metrics():
        try:
            m = gpumetrics.read()
            return {'cpu_w': m['cpu_w'], 'gfx_w': m['gfx_w'], 'limits': m['limits']}
        except (OSError, KeyError, struct.error):
            return {'cpu_w': None, 'gfx_w': None, 'limits': []}

    def summary(self):
        def avg(key):
            values = [s[key] for s in self.samples if s[key] is not None]
            return round(sum(values) / len(values), 1) if values else None
        limits = collections.Counter(name for s in self.samples for name in s['limits'])
        return {'gpu_busy': avg('gpu_busy'), 'gpu_mhz': avg('gpu_mhz'), 'power_w': avg('power_w'),
                'cpu_w': avg('cpu_w'), 'gfx_w': avg('gfx_w'),
                'limits': {name: round(count / len(self.samples), 2) for name, count in limits.most_common()},
                'cpu_mhz': avg('cpu_mhz'),
                'mem_avail_min_mb': round(min((s['mem_avail_mb'] for s in self.samples), default=0)),
                'swap_used_max_mb': round(max((s['swap_used_mb'] for s in self.samples), default=0)),
                'gtt_max_mb': round(max((s['gtt_mb'] or 0 for s in self.samples), default=0)),
                'vram_max_mb': round(max((s['vram_mb'] or 0 for s in self.samples), default=0))}


def emulator_running():
    return subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0


def stop():
    subprocess.run(['systemctl', '--user', 'stop', UNIT], env={**os.environ, **SESSION}, capture_output=True)
    for _ in range(60):
        if not emulator_running():
            return True
        time.sleep(0.5)
    return False


def measure(live, seconds, label):
    telemetry = Telemetry()
    telemetry.start()
    lines = live.send([f'measure {seconds} {label}'], timeout=seconds + 60)
    telemetry.stop.set()
    telemetry.join()
    result = {'label': label}
    for line in lines:
        if line.startswith('LIVE_MEASURE'):
            for item in line.split()[2:]:
                if '=' in item:
                    key, value = item.split('=', 1)
                    try:
                        result[key] = float(value.rstrip('%'))
                    except ValueError:
                        result[key] = value
    result.update(telemetry.summary())
    return result


def run(args):
    folder = Path(args.dir).expanduser()
    exe = Path(args.exe).expanduser() if args.exe else folder / 'kyty_emulator'
    if emulator_running():
        if not args.kill_running:
            sys.exit('an emulator is running (close it, or --kill-running)')
        subprocess.run(['pkill', '-TERM', '-x', 'kyty_emulator'])
        time.sleep(5)
    # The bench save into this folder's own _SaveData.
    saves = folder / '_SaveData'
    for name in ('PPSA01341', '.kyty-capacity', '.kyty-param', '.kyty-backup'):
        shutil.rmtree(saves / name, ignore_errors=True)
    for entry in (folder / 'bench-save').iterdir():
        shutil.copytree(entry, saves / entry.name)
    logs = folder / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / f'{time.strftime("%Y%m%d-%H%M%S")}-{args.label}.log'
    live_file = folder / 'live-commands.txt'
    live_file.write_text('id 0\n')
    environment = dict(PRESETS[args.preset])
    environment.update({'KYTY_GAME': str(GAME), 'KYTY_LIVE_FILE': str(live_file), 'KYTY_RENDER_CPUS': '1,2,3,4,5,6,7',
                        'KYTY_FPS_HUD': '1'})
    # A development build's translator identity (build-deck.sh): its translated shaders stay cached across builds.
    translator = exe.parent / 'translator-id'
    if translator.exists():
        environment['KYTY_PREFETCH_CACHE_BUILD'] = translator.read_text().strip()
    for item in args.set:
        key, value = item.split('=', 1)
        if key == 'KYTY_GAME_CONVARS' and environment.get(key):
            value = environment[key] + ';' + value
        environment[key] = value
    command = ['systemd-run', '--user', f'--unit={UNIT}', '--collect', f'--working-directory={folder}',
               f'--property=StandardOutput=file:{log}', '--property=StandardError=inherit']
    command += [f'--setenv={key}={value}' for key, value in environment.items()]
    command += [str(exe), '--game', str(GAME), '--screen-width', '1280', '--screen-height', '800', '--fullscreen',
                '--console-language', '1', '--shader-validation', 'false', '--shader-log-direction', 'Silent',
                '--printf-direction', 'Silent', '--vblank-frequency', '60', '--present-mode', 'Mailbox']
    subprocess.run(command, env={**os.environ, **SESSION}, check=True, capture_output=True)
    started = time.monotonic()
    print(f'started {exe.name} ({args.label}), log {log}', flush=True)
    live = Live(live_file, log)
    result = {'label': args.label, 'log': str(log), 'environment': environment}
    try:
        presses = 0
        while not live.in_game():
            if time.monotonic() - started > args.boot_timeout or not emulator_running():
                result['error'] = 'not in game' if emulator_running() else 'the emulator exited'
                print(json.dumps(result), flush=True)
                return 1
            presses += 1
            if presses % 4 == 0:
                live.press(OPTIONS, 0.15, 1.0)
                live.press(CROSS, 1.6, 2.0)
            else:
                live.press(CROSS)
        result['boot_s'] = round(time.monotonic() - started)
        print(f'in game after {result["boot_s"]} s', flush=True)
        time.sleep(args.settle)
        if args.where:
            print(json.dumps({'position': live.position()}), flush=True)
        result['stand'] = measure(live, args.measure, f'{args.label}-stand')
        print(json.dumps(result['stand']), flush=True)
        if args.spin > 0:
            live.send([f'pad -1 -1 255 128 0 {args.spin + 2}'])
            time.sleep(1)
            result['spin'] = measure(live, args.spin, f'{args.label}-spin')
            print(json.dumps(result['spin']), flush=True)
        if args.glide:
            # The player walks to a fixed point (live `glide`, no input) at --glide-speed m/s, measured on the way: the
            # same route every run, where a stick walk's path varies.
            start = live.position()
            target = [float(v) for v in args.glide.split(',')]
            live.send([f'glide {target[0]} {target[1]} {target[2]} {args.glide_speed}'])
            time.sleep(1)
            result['glide'] = measure(live, args.glide_seconds, f'{args.label}-glide')
            result['glide']['from'] = start
            result['glide']['to'] = live.position()
            print(json.dumps(result['glide']), flush=True)
        if args.walk > 0:
            # Forward from the bench save's spot (1-1: down the first street, the camera looking ahead).
            live.send([f'pad 128 0 -1 -1 0 {args.walk + 2}'])
            time.sleep(1)
            result['walk'] = measure(live, args.walk, f'{args.label}-walk')
            print(json.dumps(result['walk']), flush=True)
        if args.shot:
            shot = folder / 'shots' / f'{args.label}.ppm'
            shot.parent.mkdir(exist_ok=True)
            live.send([f'screenshot {shot}'])
            result['shot'] = str(shot)
        if args.prof > 0:
            # Process CPU samples (live `profp`) with the threads' names (deckprof.py reads both).
            prof = folder / 'prof' / f'{args.label}.prof'
            prof.parent.mkdir(exist_ok=True)
            pid = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()[0]
            live.send([f'profp {args.prof} {prof}'], timeout=args.prof + 60)
            names = []
            for task in Path(f'/proc/{pid}/task').iterdir():
                try:
                    names.append(f'{task.name} {(task / "comm").read_text().strip()}')
                except OSError:
                    pass
            Path(f'{prof}.threads').write_text(''.join(name + chr(10) for name in names))
            result['prof'] = str(prof)
        if args.census > 0:
            # Render-thread time per call kind, and the code that recorded each barrier (deckcensus.py).
            census = folder / 'prof' / f'{args.label}.census'
            census.parent.mkdir(exist_ok=True)
            lines = live.send([f'census {args.census}'], timeout=args.census + 60)
            census.write_text(''.join(line + chr(10) for line in lines if line.startswith('LIVE_CENSUS')))
            result['census'] = str(census)
        if args.trace > 0 or args.tracew > 0:
            # The scheduling timeline (live `trace`: guest submits and waits, flips, GPU ticks; live-trace.py frames,
            # live-trace-report.py); `tracew` adds every GPU write into guest buffers (live-trace.py producers).
            timeline = folder / 'prof' / f'{args.label}.timeline'
            timeline.parent.mkdir(exist_ok=True)
            command, seconds = ('tracew', args.tracew) if args.tracew > 0 else ('trace', args.trace)
            for line in live.send([f'{command} {seconds} {timeline}'], timeout=seconds + 120):
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
            result['timeline'] = str(timeline)
        if args.tracem > 0:
            # GPU time per shader (live `tracem`: a GPU timestamp after every draw and dispatch; live-trace.py marks).
            trace = folder / 'prof' / f'{args.label}.trace'
            trace.parent.mkdir(exist_ok=True)
            for line in live.send([f'tracem {args.tracem} {trace}'], timeout=args.tracem + 120):
                if line.startswith('LIVE_TRACE') and 'tsc_hz=' in line:
                    Path(f'{trace}.hz').write_text(line.split('tsc_hz=')[1].split()[0])
                    if 'base=' in line:
                        Path(f'{trace}.base').write_text(line.split('base=')[1].split()[0])
            result['trace'] = str(trace)
    finally:
        if not args.keep:
            stop()
    print('RESULT ' + json.dumps(result), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--dir', default='~/KytyPS5')
    r.add_argument('--exe', default='')
    r.add_argument('--set', action='append', default=[], help='NAME=VALUE for the emulator (KYTY_GAME_CONVARS adds)')
    r.add_argument('--preset', choices=sorted(PRESETS), default='lowest')
    r.add_argument('--label', default='run')
    r.add_argument('--settle', type=float, default=20)
    r.add_argument('--measure', type=int, default=20)
    r.add_argument('--spin', type=int, default=0)
    r.add_argument('--walk', type=int, default=0, help='seconds walking forward, measured (after --spin)')
    r.add_argument('--glide', default='', help='x,y,z: the player walks there (live glide), measured (after --spin)')
    r.add_argument('--glide-speed', type=float, default=4.0)
    r.add_argument('--glide-seconds', type=int, default=10)
    r.add_argument('--where', action='store_true', help='print the player position after settling')
    r.add_argument('--shot', action='store_true')
    r.add_argument('--prof', type=float, default=0, help='seconds of process CPU samples after measuring')
    r.add_argument('--tracem', type=float, default=0, help='seconds of GPU time per draw and dispatch after measuring')
    r.add_argument('--trace', type=float, default=0, help='seconds of the scheduling timeline after measuring')
    r.add_argument('--tracew', type=float, default=0, help='--trace with every GPU write into guest buffers')
    r.add_argument('--census', type=float, default=0, help='seconds of render-thread census (barrier callers) after measuring')
    r.add_argument('--boot-timeout', type=float, default=600)
    r.add_argument('--keep', action='store_true', help='leave the emulator running')
    r.add_argument('--kill-running', action='store_true')
    sub.add_parser('stop')
    args = parser.parse_args()
    if args.action == 'stop':
        return 0 if stop() else 1
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
