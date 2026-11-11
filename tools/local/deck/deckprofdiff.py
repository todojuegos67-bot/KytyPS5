#!/usr/bin/env python3
"""deckprofdiff.py A.data B.data [--top 40]: two `perf record` captures of one thread (deckproflegs.py: cpu-clock 1 kHz +
sched_switch, frame-pointer chains) compared by inclusive time per function (a sample's ms counts once for every
function on its stack; a context switch's off-CPU time, up to the thread's next event, counts the same way, marked
off-CPU): the functions whose time grew most from B to A, in ms per second of capture (first-visit work when A is a
route's first pass and B the same ground again)."""
import argparse
import collections
import subprocess


def load(path):
    out = subprocess.run(['sudo', '-n', 'perf', 'script', '-f', '-i', path, '-F', 'time,event,ip,sym,dso'],
                         capture_output=True, text=True).stdout
    events, cur = [], None
    for line in out.splitlines():
        if not line.strip():
            if cur:
                events.append(cur)
            cur = None
            continue
        if not line.startswith((' ', '\t')):
            if cur:
                events.append(cur)
            t = float(line.split(':')[0].split()[-1])
            cur = [t, 'sw' if 'sched_switch' in line else 'cpu', []]
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        sym, _, dso = parts[1].rpartition(' (')
        sym = sym.strip()
        if 'kernel' in dso:
            sym = '[k] ' + sym
        cur[2].append(sym.split('(')[0][:70])
    if cur:
        events.append(cur)
    span = events[-1][0] - events[0][0] if events else 1.0
    incl, offcpu = collections.Counter(), collections.Counter()
    for i, (t, kind, stack) in enumerate(events):
        if kind == 'cpu':
            ms = 1.0
        else:
            ms = (events[i + 1][0] - t) * 1000 if i + 1 < len(events) else 0.0
        for name in set(stack):
            if name == '[unknown]':
                continue
            incl[name] += ms
            if kind == 'sw':
                offcpu[name] += ms
    return incl, offcpu, span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--top', type=int, default=40)
    args = parser.parse_args()
    a, a_off, a_span = load(args.a)
    b, b_off, b_span = load(args.b)
    print(f'A {a_span:.1f} s, B {b_span:.1f} s; ms per second of capture (A, B, A-B; off-CPU share of A)')
    rows = []
    for name in set(a) | set(b):
        ra, rb = a[name] / a_span, b[name] / b_span
        rows.append((ra - rb, ra, rb, a_off[name] / a_span, name))
    rows.sort(reverse=True)
    for diff, ra, rb, off, name in rows[:args.top]:
        print(f'{ra:7.1f} {rb:7.1f} {diff:+7.1f}  off {off:5.1f}  {name}')


if __name__ == '__main__':
    main()
