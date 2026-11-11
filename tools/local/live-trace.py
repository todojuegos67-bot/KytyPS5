#!/usr/bin/env python3
"""Reports on a live `trace` / `tracew` capture (src/local/live-trace.h).

    live-trace.py frames TRACE [--detail N]      per-frame GPU busy, render idle, GPU idle gaps
    live-trace.py frame TRACE INDEX [--span MS]  one frame as an event timeline
    live-trace.py producers TRACE [--frames A:B] async guest readbacks and their GPU producers
                                                 (needs `tracew`)
    live-trace.py gpu TRACE INDEX [--span MS]    GPU command buffers of one frame: when they ran,
                                                 when the render thread submitted them, from which
                                                 guest queue slice
    live-trace.py sites TRACE [--range A:B]      guest instructions that fault on tracked pages
                                                 (reads: the readback wait that follows each one)
    live-trace.py marks TRACE [--top N]          GPU time per shader from a `tracem` capture (a GPU
                                                 timestamp after every draw, dispatch and native XPR
                                                 run), all frames and the part after each flip

The TSC rate is read from TRACE.hz when present (the LIVE_TRACE line prints it), else --hz.
Memory stays bounded: ranges are matched with sorted intervals, never expanded per page.
"""
import argparse
import collections
try:
    import resource
except ImportError:  # Windows
    resource = None
import statistics
import struct
from pathlib import Path

SUBMIT, GUEST_DONE, EQUEUE_WAIT, FLIP_SUBMIT, FLIP_COMPLETE, FRONT_SUSPEND = 1, 2, 7, 8, 9, 10
GPU_SUBMIT, GPU_DONE, GUEST_READBACK, RENDER_IDLE, READBACK_TICKS = 11, 12, 13, 14, 19
TICK_DONE, RENDER_SLICE, GPU_SPAN, GPU_SPAN_NS, GPU_WRITE, SYNC_READBACK = 20, 21, 22, 23, 26, 28
FAULT_SITE, FAULT_CALLER = 32, 33


class Trace:
    def __init__(self, path, hz):
        data = Path(path).read_bytes()
        hz_file = Path(str(path) + '.hz')
        self.hz = float(hz_file.read_text()) if hz_file.exists() else hz
        records = sorted(struct.iter_unpack('<QIIQQ', data))
        self.t0 = records[0][0]
        self.records = records
        self.done = {}
        spans, last = {}, {}
        for tsc, tid, kind, a, b in records:
            if kind == TICK_DONE:
                self.done.setdefault(a, self.ms(tsc))
            elif kind == GPU_SPAN:
                last[tid] = a
            elif kind == GPU_SPAN_NS and tid in last:
                spans[last.pop(tid)] = (a, b)
        # GPU clock -> TSC: the completion a monitor sees is never earlier than the GPU end.
        offsets = [self.done[t] - e / 1e6 for t, (bg, e) in spans.items() if t in self.done]
        offset = min(offsets) if offsets else 0.0
        self.gpu_offset = offset
        self.gpu = sorted((bg / 1e6 + offset, e / 1e6 + offset, t) for t, (bg, e) in spans.items())
        self.render = collections.Counter(r[1] for r in records if r[2] == RENDER_SLICE).most_common(1)[0][0]
        self.flips = [self.ms(r[0]) for r in records if r[2] == FLIP_SUBMIT and r[1] == self.render]
        self.render_idle = []
        start = None
        for tsc, tid, kind, a, b in records:
            if tid == self.render and kind == RENDER_IDLE:
                if a == 1:
                    start = (self.ms(tsc), b)
                elif start is not None:
                    self.render_idle.append((start[0], self.ms(tsc), start[1]))
                    start = None
        self.readback_waits = []  # (begin, end, tid, address)
        open_waits = {}
        for tsc, tid, kind, a, b in records:
            if kind == GUEST_READBACK:
                if a == 1:
                    open_waits[tid] = (self.ms(tsc), b)
                elif tid in open_waits:
                    begin, address = open_waits.pop(tid)
                    self.readback_waits.append((begin, self.ms(tsc), tid, address))

    def ms(self, tsc):
        return (tsc - self.t0) / self.hz * 1e3


def union(intervals, lo, hi):
    total, current = 0.0, None
    for s, e in sorted(intervals):
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if current is None or s > current[1]:
            if current:
                total += current[1] - current[0]
            current = [s, e]
        else:
            current[1] = max(current[1], e)
    if current:
        total += current[1] - current[0]
    return total


def merged(intervals, lo, hi, join=0.02):
    out = []
    for s, e in sorted(intervals):
        if e < lo or s > hi:
            continue
        if out and s <= out[-1][1] + join:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def cmd_frames(t, args):
    gpu = [(s, e) for s, e, _ in t.gpu]
    idle = [(s, e) for s, e, _ in t.render_idle]
    frames, busy, ridle = [], [], []
    for i in range(1, len(t.flips) - 1):
        lo, hi = t.flips[i], t.flips[i + 1]
        frames.append(hi - lo)
        busy.append(union(gpu, lo, hi))
        ridle.append(union(idle, lo, hi))
        if i <= args.detail:
            print(f'frame {i}: {hi - lo:6.2f} ms  GPU busy {busy[-1]:6.2f}  render idle {ridle[-1]:5.2f}')
            cursor = lo
            for s, e in merged(gpu, lo, hi):
                if s - cursor > 0.4:
                    print(f'   GPU idle {cursor - lo:6.2f}-{s - lo:6.2f} ({s - cursor:5.2f} ms), render idle '
                          f'{union(idle, cursor, s):5.2f}')
                cursor = max(cursor, e)
    print(f'{len(frames)} frames: frame {statistics.mean(frames):.2f} ms, GPU busy {statistics.mean(busy):.2f} ms, '
          f'render idle {statistics.mean(ridle):.2f} ms')


def cmd_frame(t, args):
    lo = t.flips[args.index]
    hi = lo + args.span
    events = []
    for s, e, blocked in t.render_idle:
        if s < hi and e > lo and e - s > 0.05:
            events.append((s, f'RENDER idle {e - s:.2f} ms ({"blocked" if blocked else "empty"})'))
    for tsc, tid, kind, a, b in t.records:
        x = t.ms(tsc)
        if not lo <= x < hi:
            continue
        if kind == SUBMIT:
            events.append((x, f'guest submit tid {tid} q{a & 0xff} type{(a >> 8) & 0xff} {b} dw'))
        elif kind == FRONT_SUSPEND and tid == t.render:
            events.append((x, f'render WAIT_REG_MEM {a:#x} == {b:#x}'))
        elif kind == FLIP_SUBMIT and tid == t.render:
            events.append((x, 'FLIP submit (render)'))
    for s, e, tid, address in t.readback_waits:
        if s < hi and e > lo and e - s > 0.05:
            events.append((s, f'guest {tid} readback wait {e - s:.2f} ms {address:#x}'))
    for s, e in merged([(s, e) for s, e, _ in t.gpu], lo, hi):
        if e - s > 0.1:
            events.append((s, f'GPU busy {e - s:.2f} ms (until {e - lo:.2f})'))
    for x, text in sorted(events):
        print(f'{x - lo:7.2f}  {text}')


def cmd_producers(t, args):
    import numpy as np
    reads = [(t.ms(tsc), a, b & 0xffffffff, b >> 32, 'sync' if kind == SYNC_READBACK else 'async')
             for tsc, tid, kind, a, b in t.records if kind in (READBACK_TICKS, SYNC_READBACK)]
    writes = [(tsc, a, b) for tsc, tid, kind, a, b in t.records if kind == GPU_WRITE]
    w_time = np.array([(tsc - t.t0) / t.hz * 1e3 for tsc, _, _ in writes])
    w_lo = np.array([a for _, a, _ in writes], dtype=np.uint64)
    w_hi = w_lo + np.array([b & 0xffffffff for _, _, b in writes], dtype=np.uint64)
    w_tick = np.array([b >> 32 for _, _, b in writes], dtype=np.uint64)
    # Per distinct read range: the writes that overlap it (in time order), then per read the last
    # of them recorded before the read.
    overlapping = {}
    producer = []
    for x, a, size, _, _ in reads:
        key = (a, size)
        if key not in overlapping:
            overlapping[key] = np.nonzero((w_lo < np.uint64(a + size)) & (np.uint64(a) < w_hi))[0]
        hits = overlapping[key]
        before = np.searchsorted(w_time[hits], x) - 1
        producer.append((float(w_time[hits[before]]), int(w_tick[hits[before]])) if before >= 0 else None)
    ready = sum(1 for (x, _, _, _, _), p in zip(reads, producer) if p and t.done.get(p[1], 1e18) <= x)
    print(f'{len(reads)} readbacks, {sum(1 for p in producer if p)} with a traced producer, '
          f'{ready} of them complete at request time; {len(t.flips)} flips')
    first_frame, last_frame = (int(v) for v in args.frames.split(':'))
    for k in range(first_frame, min(last_frame, len(t.flips) - 1)):
        lo, hi = t.flips[k], t.flips[k + 1]
        print(f'\nframe {k}: {hi - lo:.2f} ms')
        for (x, a, size, tick, tid), p in zip(reads, producer):
            if not lo <= x < hi:
                continue
            copy_done = t.done.get(tick, float('nan')) - lo
            if p:
                done = t.done.get(p[1], float('nan')) - lo
                text = f'producer tick {p[1]} recorded {p[0] - lo:7.2f} done {done:7.2f} ({tick - p[1]} ticks older)'
            else:
                text = 'producer not traced'
            print(f'  {tid:5s} {x - lo:6.2f} {a:#x}+{size:#x} copy tick {tick} done {copy_done:6.2f}  {text}')


def cmd_gpu(t, args):
    lo = t.flips[args.index]
    hi = lo + args.span
    submitted, replayed, queue = {}, {}, None
    for tsc, tid, kind, a, b in t.records:
        # The recording worker's replay (b == 1) is when the command buffer reached the queue,
        # after the upload copies it reads (KYTY_DEFERRED_SUBMIT).
        if kind == GPU_SUBMIT and b == 1:
            replayed[a] = t.ms(tsc)
        if tid != t.render:
            continue
        if kind == RENDER_SLICE:
            queue = f'q{a & 0xff}' if (a >> 17) & 1 else None
        elif kind == GPU_SUBMIT and b == 0:
            submitted[a] = (t.ms(tsc), queue)
    busy_by_queue = collections.Counter()
    for s, e, tick in t.gpu:
        if e < lo or s > hi:
            continue
        at, q = submitted.get(tick, (None, None))
        busy_by_queue[q] += min(e, hi) - max(s, lo)
        if e - s >= args.min:
            queued = replayed.get(tick)
            print(f'GPU {s - lo:7.2f}-{e - lo:7.2f} ({e - s:5.2f} ms) tick {tick} from {q} submitted '
                  f'{"?" if at is None else f"{at - lo:.2f}"} queued {"?" if queued is None else f"{queued - lo:.2f}"}')
    print('busy by submitting slice:', {k: round(v, 2) for k, v in busy_by_queue.most_common()})


# Tags of kind-4 marks: emulator-side Vulkan commands (MARK_IDS in generate-vulkan-recording.py).
VULKAN_MARKS = {1: 'Dispatch', 2: 'DispatchIndirect', 3: 'DispatchBase', 10: 'CopyBuffer', 11: 'CopyBuffer2',
                12: 'CopyImage', 13: 'CopyImage2', 14: 'CopyBufferToImage', 15: 'CopyBufferToImage2',
                16: 'CopyImageToBuffer', 17: 'CopyImageToBuffer2', 20: 'FillBuffer', 21: 'UpdateBuffer',
                30: 'ClearColorImage', 31: 'ClearDepthStencilImage', 32: 'ClearAttachments', 40: 'BlitImage',
                41: 'BlitImage2', 42: 'ResolveImage', 43: 'ResolveImage2', 50: 'PipelineBarrier',
                51: 'PipelineBarrier2'}
# Bits 16-31 of a kind-4 mark: the emulator's own work it belongs to (MarkSiteId in src/local/live-trace-gpu.h).
MARK_SITES = {1: 'tiler-detile', 2: 'tiler-tile', 3: 'tiler-image', 4: 'tiler-d16', 5: 'tiler-bgra16',
              6: 'fault-process', 7: 'lod-stats'}


def code_names(exe, trace):
    """An address -> 'function' namer for code of `exe` loaded at TRACE.base (nm -C)."""
    import bisect
    import subprocess
    base_file = Path(str(trace) + '.base')
    base = int(base_file.read_text(), 16) if base_file.exists() else 0
    out = subprocess.run(['nm', '-C', '--defined-only', '-n', exe], capture_output=True, text=True, check=True).stdout
    addresses, names = [], []
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in 'tTwW':
            addresses.append(int(parts[0], 16))
            names.append(parts[2].split('(')[0])
    def name(address):
        i = bisect.bisect_right(addresses, address - base) - 1
        return names[i] if i >= 0 else f'{address:#x}'
    return name


def cmd_marks(t, args):
    kinds = {1: 'draw', 2: 'dispatch', 3: 'native', 4: 'vk'}
    recorded = {}
    for tsc, tid, kind, a, b in t.records:
        if kind == 24:
            recorded[a] = (t.ms(tsc), b >> 60, b & 0x0fffffffffffffff)
    done = {a: b / 1e6 + t.gpu_offset for tsc, tid, kind, a, b in t.records if kind == 25}
    # A barrier's mark: the code that recorded it (GpuMarkCaller), named with --exe and TRACE.base.
    callers = {a: b for tsc, tid, kind, a, b in t.records if kind == 40}
    name_of = code_names(args.exe, args.trace) if args.exe and callers else None
    flush_times, flush_ticks = [], []
    for tsc, tid, kind, a, b in t.records:
        if tid == t.render and kind == GPU_SUBMIT and b == 0:
            flush_times.append(t.ms(tsc))
            flush_ticks.append(a)
    begin = {tick: s for s, e, tick in t.gpu}
    import bisect
    # The guest queue whose slice the render thread recorded each mark in ('-': outside slices).
    slices, opened = [], None
    for tsc, tid, kind, a, b in t.records:
        if tid == t.render and kind == RENDER_SLICE:
            if (a >> 17) & 1:
                opened = (t.ms(tsc), a & 0xff)
            elif opened is not None:
                slices.append((opened[0], t.ms(tsc), opened[1]))
                opened = None
    slice_begins = [b for b, _, _ in slices]

    def queue_of(x):
        i = bisect.bisect_right(slice_begins, x) - 1
        return f'q{slices[i][2]}' if i >= 0 and x <= slices[i][1] else '-'
    by_tick = collections.defaultdict(list)
    for slot, (x, kind, shader) in recorded.items():
        i = bisect.bisect_left(flush_times, x)
        if i < len(flush_ticks) and slot in done:
            by_tick[flush_ticks[i]].append((slot, kind, shader, queue_of(x)))
    total, after_flip = collections.Counter(), collections.Counter()
    count, by_queue = collections.Counter(), collections.Counter()
    frames = max(1, len(t.flips) - 1)
    for tick, marks in by_tick.items():
        if tick not in begin:
            continue
        previous = begin[tick]
        for slot, kind, shader, queue in sorted(marks):
            end = done[slot]
            cost = max(0.0, end - previous)
            previous = max(previous, end)
            by_queue[queue] += cost
            if args.queue is not None and queue != args.queue:
                continue
            key = (kinds.get(kind, kind), shader)
            if slot in callers:
                key += (name_of(callers[slot]) if name_of else f'{callers[slot]:#x}',)
            total[key] += cost
            count[key] += 1
            i = bisect.bisect_right(t.flips, end) - 1
            if 0 <= i < len(t.flips) - 1 and end - t.flips[i] < args.after:
                after_flip[key] += cost
    print(f'{frames} frames; GPU ms per frame by shader (whole frame | first {args.after} ms after each flip)')
    for key, value in total.most_common(args.top):
        if key[0] == 'vk':
            name = VULKAN_MARKS.get(key[1] & 0xffff, str(key[1] & 0xffff))
            if key[1] >> 16:
                name += '@' + MARK_SITES.get(key[1] >> 16, str(key[1] >> 16))
            if len(key) > 2:
                name += ' ' + key[2]
        else:
            name = f'{key[1]:#014x}'
        print(f'  {key[0]:8s} {name:>28s}  {value / frames:6.3f} | {after_flip[key] / frames:6.3f}  '
              f'({count[key] / frames:.1f}/frame)')
    print(f'  all marks: {sum(total.values()) / frames:.2f} | {sum(after_flip.values()) / frames:.2f}')
    print('  by the guest queue that recorded them: ' +
          ', '.join(f'{q} {v / frames:.2f}' for q, v in by_queue.most_common()))
    busy = sum(e - s for s, e, tick in t.gpu) / frames
    print(f'  GPU busy (command buffer spans): {busy:.2f} ms/frame; by kind:')
    for kind in sorted(set(key[0] for key in total)):
        print(f'    {kind:8s} {sum(v for key, v in total.items() if key[0] == kind) / frames:6.2f} | '
              f'{sum(v for key, v in after_flip.items() if key[0] == kind) / frames:6.2f}')


def cmd_sites(t, args):
    lo, hi = (float(v) * 1000.0 for v in args.range.split(':')) if args.range else (float('-inf'), float('inf'))
    # Per thread: the readback waits in time order, to charge each read fault its wait.
    waits = collections.defaultdict(list)
    for begin, end, tid, address in t.readback_waits:
        waits[tid].append((begin, end))
    for items in waits.values():
        items.sort()
    import bisect
    reads = collections.Counter()
    read_ms = collections.Counter()
    writes = collections.Counter()
    example = {}
    for tsc, tid, kind, a, b in t.records:
        if kind != FAULT_SITE:
            continue
        ms = t.ms(tsc)
        if not lo <= ms <= hi:
            continue
        address = b & ((1 << 63) - 1)
        if b >> 63:
            writes[a] += 1
            continue
        reads[a] += 1
        example.setdefault(a, address)
        items = waits.get(tid, [])
        at = bisect.bisect_left(items, (ms, -1.0))
        if at < len(items) and items[at][0] - ms < 1.0:
            read_ms[a] += items[at][1] - items[at][0]
    print(f'read-fault sites: {sum(reads.values())} faults, {sum(read_ms.values()):.1f} ms of readback waits')
    for site, n in sorted(reads.items(), key=lambda item: -read_ms[item[0]])[:args.top]:
        print(f'  {site:#014x}  {n:6d} faults  {read_ms[site]:8.2f} ms  e.g. {example[site]:#x}')
    print(f'write-fault sites: {sum(writes.values())} faults')
    for site, n in writes.most_common(args.top):
        print(f'  {site:#014x}  {n:6d} faults')
    # Host code faults (FaultCaller): the qword at the stack top, the caller of a leaf copy routine.
    callers = collections.Counter((b, a) for tsc, tid, kind, a, b in t.records
                                  if kind == FAULT_CALLER and lo <= t.ms(tsc) <= hi)
    if callers:
        print(f'host fault sites by stack-top caller: {sum(callers.values())} faults')
        for (site, caller), n in callers.most_common(args.top):
            print(f'  {site:#014x} <- {caller:#014x}  {n:6d}')


def main():
    # A runaway report must never take the machine down with the game running.
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hz', type=float, default=3187145000.0)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('frames')
    p.add_argument('trace')
    p.add_argument('--detail', type=int, default=3)
    p.set_defaults(func=cmd_frames)
    p = sub.add_parser('frame')
    p.add_argument('trace')
    p.add_argument('index', type=int)
    p.add_argument('--span', type=float, default=34)
    p.set_defaults(func=cmd_frame)
    p = sub.add_parser('gpu')
    p.add_argument('trace')
    p.add_argument('index', type=int)
    p.add_argument('--span', type=float, default=34)
    p.add_argument('--min', type=float, default=0.1)
    p.set_defaults(func=cmd_gpu)
    p = sub.add_parser('marks')
    p.add_argument('--queue', help="only marks recorded in this guest queue's slices (q0, q49, ...; - outside)")
    p.add_argument('trace')
    p.add_argument('--top', type=int, default=30)
    p.add_argument('--after', type=float, default=7.0)
    p.add_argument('--exe', help='the build that made the capture: barriers by the function that recorded them')
    p.set_defaults(func=cmd_marks)
    p = sub.add_parser('sites')
    p.add_argument('trace')
    p.add_argument('--range', default='')
    p.add_argument('--top', type=int, default=25)
    p.set_defaults(func=cmd_sites)
    p = sub.add_parser('producers')
    p.add_argument('trace')
    p.add_argument('--frames', default='5:7')
    p.set_defaults(func=cmd_producers)
    args = parser.parse_args()
    args.func(Trace(args.trace, args.hz), args)


if __name__ == '__main__':
    main()
