#!/usr/bin/env python3
"""Timeline report of a live `trace SECONDS PATH` capture (src/local/live-trace.h).

    live-trace-report.py TRACE [--hz TSC_HZ] [--gaps 0.3] [--frames 6]

Prints per-frame intervals, thread roles, the render (back) end's idle gaps and, for each gap,
the submission that ended it and what that guest thread waited for before submitting.
"""
import argparse
import bisect
import collections
import struct
from pathlib import Path

NAMES = {1: 'Submit', 2: 'GuestDone', 3: 'FrontIdle', 4: 'BackIdle', 5: 'BackFrame', 6: 'Interrupt',
         7: 'EqueueWait', 8: 'FlipSubmit', 9: 'FlipComplete', 10: 'FrontSuspend', 11: 'GpuSubmit',
         12: 'GpuDone', 13: 'GuestReadback', 14: 'RenderIdle', 15: 'BackItem', 16: 'FrontItem',
         17: 'SemaWait', 18: 'CondWait'}


def load(path):
    data = Path(path).read_bytes()
    records = [struct.unpack_from('<QIIQQ', data, i * 32) for i in range(len(data) // 32)]
    records.sort()
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace')
    p.add_argument('--hz', type=float, default=3187153320.0)
    p.add_argument('--gaps', type=float, default=0.3, help='report back-end idle gaps longer than this (ms)')
    p.add_argument('--frames', type=int, default=6)
    args = p.parse_args()
    records = load(args.trace)
    t0 = records[0][0]
    ms = lambda tsc: (tsc - t0) * 1000.0 / args.hz

    by_type = collections.defaultdict(list)
    tids = collections.defaultdict(collections.Counter)
    for tsc, tid, kind, a, b in records:
        by_type[kind].append((ms(tsc), tid, a, b))
        tids[tid][NAMES.get(kind, kind)] += 1
    print(f'{len(records)} records, {ms(records[-1][0]):.1f} ms')
    print('threads:')
    for tid, kinds in sorted(tids.items(), key=lambda kv: -sum(kv[1].values()))[:14]:
        print(f'  {tid}: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common(6)))

    # Idle intervals of the executing thread (combined: RenderIdle; split: BackIdle).
    def intervals(kind, flag_field=2):
        out, open_at = [], {}
        for t, tid, a, b in by_type[kind]:
            if a == 1:
                open_at[tid] = (t, b)
            elif tid in open_at:
                begin, why = open_at.pop(tid)
                out.append((begin, t, tid, why))
        return out

    back_idle = intervals(4) or intervals(14)
    front_idle = intervals(3)
    # (BackFrame, 5, is no longer recorded: the flips' completions, FlipComplete, mark the frames then.)
    frames = [t for t, _, _, _ in by_type[5]] or [t for t, _, _, _ in by_type[9]]
    if len(frames) < 2:
        print('fewer than two frames in the trace')
        return
    flips = [t for t, _, _, _ in by_type[9]]
    print(f'frames {len(frames)}: ' + (f'mean {(frames[-1] - frames[0]) / (len(frames) - 1):.2f} ms' if len(frames) > 1 else ''))
    total = (frames[-1] - frames[0]) if len(frames) > 1 else 1
    idle_in = sum(min(e, frames[-1]) - max(b, frames[0]) for b, e, _, _ in back_idle if e > frames[0] and b < frames[-1])
    print(f'back idle {idle_in / (len(frames) - 1):.2f} ms/frame; front idle '
          f'{sum(e - b for b, e, _, _ in front_idle) / max(1, len(frames) - 1):.2f} ms/frame')

    # Guest waits per thread (equeue waits and readbacks).
    waits = collections.defaultdict(list)  # tid -> [(begin, end, kind, info)]
    for kind, label in ((7, 'equeue'), (13, 'readback')):
        open_at = {}
        for t, tid, a, b in by_type[kind]:
            if a == 1:
                open_at[tid] = (t, b)
            elif tid in open_at:
                begin, info = open_at.pop(tid)
                waits[tid].append((begin, t, label, b if kind == 7 else info))
    for tid in waits:
        waits[tid].sort()
    submits = sorted(by_type[1])
    submit_times = [s[0] for s in submits]
    interrupts = sorted(by_type[6])
    interrupt_times = [i[0] for i in interrupts]
    gpu_done = {a: t for t, _, a, _ in by_type[12]}
    gpu_submit = {}
    for t, _, a, b in by_type[11]:
        gpu_submit.setdefault((a, b), t)

    print(f'\nback idle gaps > {args.gaps} ms (first {args.frames} frames):')
    shown = 0
    for begin, end, tid, why in back_idle:
        if end - begin < args.gaps or not frames or begin < frames[0]:
            continue
        if shown >= args.frames * 6:
            break
        shown += 1
        frame = bisect.bisect_right(frames, begin)
        line = f'  f{frame} {begin:9.2f}-{end:9.2f} ({end - begin:5.2f} ms)'
        i = bisect.bisect_left(submit_times, begin)
        if i < len(submits) and submits[i][0] <= end + 0.05:
            st, stid, sa, sb = submits[i]
            line += f'  ended by submit q{sa & 0xff} type{sa >> 8} ({sb} dw) tid {stid} at {st:.2f}'
            prior = [w for w in waits.get(stid, []) if w[1] <= st + 0.001]
            if prior:
                wb, we, wl, wi = prior[-1]
                line += f'; that thread {wl} wait {wb:.2f}-{we:.2f} ({we - wb:.2f} ms)'
                if wl == 'equeue':
                    j = bisect.bisect_right(interrupt_times, we) - 1
                    if j >= 0:
                        it, _, ia, ib = interrupts[j]
                        line += f' woken by interrupt ev{ia} ctx{ib} at {it:.2f}'
        print(line)

    print('\nper-frame summary (ms from frame start):')
    for k in range(min(args.frames, len(frames) - 1)):
        f0, f1 = frames[k], frames[k + 1]
        idle = sum(min(e, f1) - max(b, f0) for b, e, _, _ in back_idle if e > f0 and b < f1)
        subs = [s for s in submits if f0 <= s[0] < f1]
        irq = [i for i in interrupts if f0 <= i[0] < f1]
        fl = [t for t in flips if f0 <= t < f1]
        print(f'  frame {k}: {f1 - f0:6.2f} ms, back idle {idle:5.2f}, submits {len(subs)}, interrupts {len(irq)}, '
              f'flip completes at ' + ', '.join(f'{t - f0:.1f}' for t in fl))


if __name__ == '__main__':
    main()
