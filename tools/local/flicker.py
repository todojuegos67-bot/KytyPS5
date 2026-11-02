"""flicker.py <series.kser> [out.png]: wrong frames in a fixed-camera frame series.

The series comes from the live command `screenshots <path> <count>` (live-control.h, screenshot.h RequestSeries),
e.g. python tools/local/bench-windows.py live 'screenshots E:/out/arch.kser 5400' with the camera still.

Each frame is compared with the median of its 4 nearest neighbours (2 before, 2 after). A frame with objects missing
differs from all of them; motion (fog, cloth, particles) differs a little from each. Prints the score distribution and
the frames whose score is far above the run's typical score; writes a strip of the worst frames beside a neighbour.
"""
import sys

import numpy as np

path = sys.argv[1]
with open(path, 'rb') as f:
    header = f.readline().split()
    w, h = int(header[1]), int(header[2])
    data = np.frombuffer(f.read(), dtype=np.uint8)
n = data.size // (w * h * 3)
frames = data[: n * w * h * 3].reshape(n, h, w, 3).astype(np.int16)
scores = np.zeros(n)
area = np.zeros(n)
for i in range(n):
    near = [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < n]
    med = np.median(frames[near], axis=0)
    diff = np.abs(frames[i] - med).max(axis=2)
    scores[i] = diff.mean()
    area[i] = (diff > 40).mean() * 100  # % of pixels changed a lot
typical = np.median(scores)
mad = np.median(np.abs(scores - typical)) + 1e-6
bad = [i for i in range(n) if area[i] > 1.0 and scores[i] > typical + 12 * mad]
# Exact repeats (the presenter's idle refreshes) are not distinct frames.
repeats = sum(1 for i in range(1, n) if not np.any(frames[i] != frames[i - 1]))
print(f'{path}: {n} frames ({repeats} repeats), {w}x{h}; score median {typical:.2f} p99 {np.percentile(scores, 99):.2f} '
      f'max {scores.max():.2f}; area p99 {np.percentile(area, 99):.2f}% max {area.max():.2f}%')
print(f'bad frames: {len(bad)}' + (': ' + ' '.join(f'{i}({scores[i]:.1f},{area[i]:.1f}%)' for i in bad[:40]) if bad else ''))
if len(sys.argv) > 2:
    from PIL import Image
    worst = list(np.argsort(-area)[:6])
    rows = []
    for i in sorted(worst):
        j = i - 1 if i > 0 else i + 1
        rows.append(np.concatenate([frames[i], frames[j]], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0).astype(np.uint8)).save(sys.argv[2])
    print('worst frames (left) beside the previous one:', ' '.join(f'{i}({area[i]:.1f}%)' for i in sorted(worst)), '->', sys.argv[2])
