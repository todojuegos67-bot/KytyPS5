#!/usr/bin/env python3
"""Measure a fixed saved entry, optionally walking forward before stopping.

Continue through the menus, settle once, measure, and close normally. Restore the
user's original saves only after the emulator has stopped; retain test saves in
the evidence directory. For manual play, use run-demons-souls.sh instead.
The optional forward setup happens once after the HUD appears; all measurement
windows remain stationary and send no gameplay or camera input.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from debug_cpu_policy import atomic_json, start_ticks

ROOT = Path(__file__).resolve().parents[2]
# (With the slots' texts and backups: a test save's backups must not stay among the user's, libs/saveDataBackup.h.)
SAVE_SCOPES = (Path('PPSA01341'), Path('.kyty-capacity/PPSA01341'), Path('.kyty-param/PPSA01341'),
               Path('.kyty-backup/PPSA01341'))


def restore_save_snapshot(snapshot, save_root):
    """Restore this title's exact file set, preserving saves for other titles."""
    wanted = {}
    current = {}
    for scope in SAVE_SCOPES:
        for root, files in ((snapshot, wanted), (save_root, current)):
            directory = root / scope
            if directory.is_symlink():
                raise ValueError('Save scope must not be a symlink')
            for path in directory.rglob('*'):
                if path.is_symlink():
                    raise ValueError('Save snapshot must not contain symlinks')
                if path.is_file():
                    files[path.relative_to(root)] = path
    if not any(path.parts[0] == 'PPSA01341' for path in wanted):
        raise ValueError('Snapshot contains no game save files')
    expected = {str(name): hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in wanted.items()}
    for name, path in current.items():
        if name not in wanted:
            path.unlink()
    for name, source in wanted.items():
        target = save_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for scope in SAVE_SCOPES:
        for directory in sorted((save_root / scope).rglob('*'),
                                key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    actual = {str(path.relative_to(save_root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for scope in SAVE_SCOPES for path in (save_root / scope).rglob('*') if path.is_file()}
    if actual != expected:
        raise RuntimeError('Restored save file set or contents differ from the snapshot')
    return actual


def cpu_snapshot(pid):
    threads = {}
    for task in (Path('/proc')/str(pid)/'task').iterdir():
        try:
            raw = (task/'stat').read_text()
            fields = raw.rsplit(')', 1)[1].split()
            threads[task.name] = {
                'name': raw[raw.index('(')+1:raw.rindex(')')],
                'start_ticks': fields[19], 'user_ticks': int(fields[11]),
                'system_ticks': int(fields[12]), 'minor_faults': int(fields[7]),
                'major_faults': int(fields[9]),
            }
        except FileNotFoundError:
            pass  # A thread can exit while /proc is being enumerated.
    return {'monotonic_ns': time.monotonic_ns(), 'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'),
            'threads': threads}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / '_Build/profiles/manual-current-launch.json')
    p.add_argument('--save-baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=20)
    p.add_argument('--walk-forward-seconds', type=float, default=0,
                   help='After the HUD settles, hold W once, release, then settle before measuring (0–10)')
    p.add_argument('--settle-seconds', type=float, default=10,
                   help='Settle after the HUD and after the forward setup (default 10; iteration runs use less)')
    p.add_argument('--keep-seconds', type=int, default=0,
                   help='Keep this owned stationary process for live A/B; create OUT/stop to close and restore saves')
    p.add_argument('--shader-warmup-seconds', type=int, default=0,
                   help='Forwarded to the launcher: bound the startup precompile budget (0 = all)')
    p.add_argument('--diagnose-stutter', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-callgraph', action='store_true')
    p.add_argument('--profile-kernel', action='store_true', help='Include kernel cycles in the profile')
    p.add_argument('--profile-all-threads', action='store_true', help='Profile the process instead of only its renderer')
    p.add_argument('--profile-stat', action='store_true', help='Also collect renderer hardware counters')
    p.add_argument('--cpu-times', action='store_true', help='Read per-thread CPU counters around the measurement')
    p.add_argument('--verify-native-resources', action='store_true')
    p.add_argument('--output-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                   help='Diagnostic only: measure a different render size instead of 2560x1440')
    a = p.parse_args()
    if (a.profile_callgraph or a.profile_kernel or a.profile_stat or a.profile_all_threads) and not a.profile:
        p.error('Profile modifiers require --profile')
    if not 5 <= a.seconds <= 30:
        p.error('Measurement must be 5–30 seconds')
    if not 0 <= a.keep_seconds <= 7200:
        p.error('Retention must be 0–7200 seconds')
    if not math.isfinite(a.walk_forward_seconds) or not 0 <= a.walk_forward_seconds <= 10:
        p.error('Forward setup must be 0–10 seconds')
    a.out = a.out.resolve(); a.save_baseline = a.save_baseline.resolve();a.config=a.config.resolve()
    if not (a.save_baseline/'PPSA01341').is_dir():
        p.error('Expected a save baseline containing PPSA01341')
    if subprocess.run(['pgrep','-x','kyty_emulator'],capture_output=True).returncode == 0:
        p.error('Close the existing emulator first')
    a.out.mkdir(parents=True,exist_ok=False)
    result = {'status':'running','config':str(a.config),'save_baseline':str(a.save_baseline),
              'protocol':'continue-stationary-v4-exact-save-no-loading-input',
              'seconds':a.seconds,'diagnose_stutter':a.diagnose_stutter,'profile':a.profile,
              'verify_native_resources':a.verify_native_resources,
              'profile_callgraph':a.profile_callgraph,
              'profile_kernel':a.profile_kernel, 'profile_stat':a.profile_stat, 'cpu_times':a.cpu_times,
              'profile_all_threads':a.profile_all_threads,
              'input':'After Continue, only confirm an observed offline prompt; no loading skips or gameplay/camera input',
              'started_ns':time.monotonic_ns()}
    result.update(stationary=True, keep_seconds=a.keep_seconds)
    if a.walk_forward_seconds:
        result.update(protocol='continue-forward-stop-v1-exact-save',
                      walk_forward_seconds=a.walk_forward_seconds,
                      input='After observed gameplay HUD and 10s settle, hold W once for the requested wall time; release and settle 10s; no input during measurement')
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def save(): atomic_json(a.out/'result.json',result)
    def restore(snapshot):
        return restore_save_snapshot(snapshot, ROOT/'_SaveData')
    shutil.copytree(ROOT/'_SaveData',a.out/'saves-before')
    save()
    launcher = None; pid = None; identity = None; perf = None; perf_stat = None
    try:
        result['baseline_save_sha256'] = restore(a.save_baseline)
        save()
        command=['python3',str(ROOT/'tools/local/play-demons-souls.py'),'--checkpoint-config',str(a.config)]
        command.extend(['--output-size',*map(str,a.output_size)] if a.output_size else ['--2k'])
        result['output_size']=a.output_size or [2560,1440]
        if a.diagnose_stutter: command.append('--diagnose-stutter')
        if a.verify_native_resources: command.append('--verify-native-resources')
        if a.shader_warmup_seconds:
            command.extend(['--shader-warmup-seconds', str(a.shader_warmup_seconds)])
        with (a.out/'launcher.log').open('w') as log:
            launcher=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            content=(a.out/'launcher.log').read_text()
            if launcher.poll() is not None:raise RuntimeError('Launcher exited before readiness')
            if '8 项优化已在启动阶段启用，现可正常操作。' in content:
                directory=Path(re.search(r'日志：(.+)',content)[1]);driver=(directory/'driver.log').read_text()
                match=re.search(r'^PID (\d+); evidence: (.+)$',driver,re.M)
                pid=int(match[1]);evidence=Path(match[2]);identity=start_ticks(pid)
                result.update(pid=pid,pid_start_ticks=identity,launch_directory=str(directory),evidence=str(evidence),binary_sha256=json.loads((evidence/'run.json').read_text())['executable_sha256']);save();break
            time.sleep(.2)
        else:raise RuntimeError('Startup timed out')
        print(f'Continue: PID {pid}',flush=True)
        with (a.out/'continue.log').open('w') as log:
            subprocess.run(['taskset','-c','8-15','python3',str(ROOT/'tools/local/demons-souls-smoke.py'),'--pid',str(pid),'--resume','--stop-at','gameplay','--fast-skip','--timeout','180'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=200,check=True)
        continue_directory = Path(re.search(r'^Evidence: (.+)$', (a.out/'continue.log').read_text(), re.M)[1])
        input_events = [json.loads(line) for line in (continue_directory/'input.jsonl').read_text().splitlines()]
        if any(event['after_continue'] and not event['resume_prompt'] for event in input_events):
            raise RuntimeError('Unobserved input after Continue invalidates the entry measurement')
        atomic_json(a.out/'continue-input.json', input_events)
        result['continue_evidence'] = str(continue_directory)
        result['hud_ns']=time.monotonic_ns();save()
        child_environment = dict(item.split('=', 1) for item in
            (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\0') if '=' in item)
        result['runtime_environment'] = {k:v for k,v in child_environment.items() if k.startswith('KYTY_') or k == 'LD_PRELOAD'}
        # Streaming and temporal history settle before the measurement.
        time.sleep(a.settle_seconds)
        if a.walk_forward_seconds:
            print(f'Forward setup: W for {a.walk_forward_seconds:g}s, then stop and settle', flush=True)
            setup = subprocess.run(['taskset', '-c', '8-15', 'python3',
                str(ROOT/'tools/local/quick-benchmark-game.py'), '--pid', str(pid),
                '--forward-seconds', str(a.walk_forward_seconds), '--settle-seconds', str(a.settle_seconds), '--seconds', '2',
                '--camera-frames', '0', '--screenshots', '--label', 'forward-stop-setup'],
                cwd=ROOT, timeout=55, check=True, capture_output=True, text=True)
            setup_dir = Path(json.loads(setup.stdout)['evidence'])
            shutil.copytree(setup_dir, a.out/'forward-setup')
            setup_result = json.loads((a.out/'forward-setup/result.json').read_text())
            moves = [phase for phase in setup_result['phases'] if phase['key'] is not None]
            if (setup_result['status'] != 'completed_unverified' or len(moves) != 1 or
                    moves[0]['key'] != 'w' or
                    abs(moves[0]['held_seconds'] - a.walk_forward_seconds) > .1):
                raise RuntimeError('Forward setup did not complete the requested bounded W input')
            result['forward_setup'] = {'evidence': str(a.out/'forward-setup'),
                'keydown_ns': moves[0]['keydown_ns'], 'keyup_ns': moves[0]['keyup_ns'],
                'held_seconds': moves[0]['held_seconds'], 'settle_seconds': a.settle_seconds}
            save()
        if a.profile:
            renderer=json.loads((directory/'startup-cpu-policy.json').read_text())
            tids=[int(tid) for tid,name in renderer['thread_names'].items() if name=='Kyty.Gpu']
            if len(tids)!=1:raise RuntimeError('Renderer TID is ambiguous')
            with (a.out/'perf.log').open('w') as log:
                command=['sudo','-n','perf','record','--no-buildid','--sample-cpu','-F','99' if a.profile_callgraph else '199',
                         '-e','cycles' if a.profile_kernel else 'cycles:u',
                         '-p' if a.profile_all_threads else '-t',
                         str(pid if a.profile_all_threads else tids[0]),'-o',str(a.out/'perf.data')]
                if a.profile_callgraph:command.extend(['--call-graph','dwarf,8192'])
                perf=subprocess.Popen(command+['--','sleep',str(a.seconds+3)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if a.profile_stat:
                with (a.out/'perf-stat.log').open('w') as log:
                    events = 'cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults'
                    perf_stat=subprocess.Popen(['sudo','-n','perf','stat','-x',';',
                        '-e',events,'-t',str(tids[0]),'--','sleep',str(a.seconds+3)],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Stationary entry: no input, {a.seconds}s',flush=True)
        if a.cpu_times:
            atomic_json(a.out/'cpu-before.json',cpu_snapshot(pid))
        measurement = subprocess.run(['taskset','-c','8-15','python3',str(ROOT/'tools/local/quick-benchmark-game.py'),
            '--pid',str(pid),'--seconds',str(a.seconds),'--camera-frames','0','--screenshots','--label','entry'],
            cwd=ROOT,timeout=a.seconds+35,check=True,capture_output=True,text=True)
        if a.cpu_times:
            atomic_json(a.out/'cpu-after.json',cpu_snapshot(pid))
        measurement_dir = Path(json.loads(measurement.stdout)['evidence'])
        shutil.copytree(measurement_dir, a.out/'measurement')
        phases=json.loads((a.out/'measurement/result.json').read_text())['phases']
        assert len(phases) == 1 and phases[0]['key'] is None
        result['measurement'] = phases[0] | {'fps': phases[0]['measured_fps']}
        samples=result['measurement']['samples'];windows=[]
        for i,sample in enumerate(samples):
            before=next((q for q in samples[:i] if sample['monotonic_ns']-q['monotonic_ns']<=1_100_000_000),None)
            if before and sample['monotonic_ns']-before['monotonic_ns']>=900_000_000:
                windows.append((sample['frame']-before['frame'])*1e9/(sample['monotonic_ns']-before['monotonic_ns']))
        result['min_approximately_1s_fps']=min(windows) if windows else None
        if a.verify_native_resources:
            command = ['sudo','-n','python3',str(ROOT/'tools/local/read-native-counters.py'),
                       str(a.out/'result.json'),'--out',str(a.out/'native-counters.json')]
            subprocess.run(command, check=True)
            counters=json.loads((a.out/'native-counters.json').read_text())['counters']
        if a.verify_native_resources:
            srt=counters['kyty_local_srt_native_stats']
            if counters['kyty_local_srt_native_mode'] != [2] or not srt[3] or srt[1] or srt[4]:
                raise RuntimeError('Native resource verification did not complete cleanly')
            result['native_verification']=counters
        if a.diagnose_stutter:
            subprocess.run(['python3',str(ROOT/'tools/local/analyze-stutter.py'),str(evidence/'stdout.log'),'--from-ns',str(result['measurement']['start_ns']),'--to-ns',str(result['measurement']['end_ns']),'--out',str(a.out/'stutter.json')],check=True)
        result['status']='measured';save()
        if a.keep_seconds:
            result['diagnostics_deadline_monotonic'] = time.monotonic() + a.keep_seconds
            result['retained'] = True
            save()
            print(f'STATIONARY_SESSION_READY={a.out / "result.json"}', flush=True)
            while time.monotonic() < result['diagnostics_deadline_monotonic'] and not (a.out/'stop').exists():
                if launcher.poll() is not None or start_ticks(pid) != identity:
                    raise RuntimeError('Retained stationary process exited')
                time.sleep(.2)
            result['retained'] = False
            result['retention_end'] = 'stop_requested' if (a.out/'stop').exists() else 'deadline'
            save()
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Retain the supervisor evidence even when shader warmup fails before
        # the launcher prints readiness and before a live PID can be sampled.
        if 'evidence' not in result:
            content = (a.out/'launcher.log').read_text() if (a.out/'launcher.log').exists() else ''
            launch_match = re.search(r'日志：(.+)', content)
            if launch_match:
                launch_path = Path(launch_match[1])
                driver_path = launch_path/'driver.log'
                driver_text = driver_path.read_text() if driver_path.exists() else ''
                process_match = re.search(r'^PID (\d+); evidence: (.+)$', driver_text, re.M)
                if process_match:
                    evidence_path = Path(process_match[2]).resolve()
                    metadata_path = evidence_path/'run.json'
                    if evidence_path.is_relative_to(ROOT/'_Build/experiments') and metadata_path.is_file():
                        metadata = json.loads(metadata_path.read_text())
                        result.update(evidence=str(evidence_path), launch_directory=str(launch_path),
                                      binary_sha256=metadata['executable_sha256'],
                                      pid=int(process_match[1]), pid_start_ticks=str(metadata['pid_start_ticks']))
        if pid is not None:
            try:
                if start_ticks(pid)==identity:
                    subprocess.run(['python3',str(ROOT/'tools/local/close-game-window.py'),'--pid',str(pid)],cwd=ROOT,timeout=15,check=False)
            except FileNotFoundError:pass
        if launcher is not None:
            try:launcher.wait(timeout=20)
            except subprocess.TimeoutExpired:
                launcher.terminate();launcher.wait(timeout=20)
        for profiler, name in [(perf, 'perf'), (perf_stat, 'perf_stat')]:
            if profiler is None:
                continue
            try:
                try:profiler.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    subprocess.run(['sudo','-n','kill','-TERM','--',str(-profiler.pid)],timeout=5,check=False)
                    profiler.wait(timeout=10)
            except subprocess.SubprocessError as exc:
                result[name+'_cleanup_error']=str(exc)
            result[name+'_returncode']=profiler.returncode
        if subprocess.run(['pgrep','-x','kyty_emulator'],capture_output=True).returncode!=0:
            shutil.copytree(ROOT/'_SaveData',a.out/'saves-after')
            result['restored_save_sha256'] = restore(a.out/'saves-before')
            result['original_save_restored']=True
        result['finished_ns']=time.monotonic_ns();save()
        print(json.dumps({'status':result['status'],'fps':result.get('measurement',{}).get('fps'),'min_1s_fps':result.get('min_approximately_1s_fps'),'out':str(a.out)},indent=2),flush=True)


if __name__=='__main__':main()
