#!/usr/bin/env python3
"""gpumetrics.py [SECONDS] [INTERVAL]: the Steam Deck APU's gpu_metrics (v2.x, amdgpu) every INTERVAL seconds:
GPU activity, clocks, power by rail, temperatures and what limits the clocks (the SMU's throttle bits).
Also a module: read() returns one sample as a dict."""
import struct
import sys
import time
from pathlib import Path

PATH = Path('/sys/class/drm/card0/device/gpu_metrics')
# indep_throttle_status bits (SMU_THROTTLER_*, kgd_pp_interface.h)
THROTTLERS = {0: 'PPT0', 1: 'PPT1', 2: 'PPT2', 3: 'PPT3', 4: 'SPL', 5: 'FPPT', 6: 'SPPT', 7: 'SPPT_APU',
              16: 'TDC_GFX', 17: 'TDC_SOC', 18: 'TDC_MEM', 19: 'TDC_VDD', 20: 'TDC_CVIP', 21: 'EDC_CPU',
              22: 'EDC_GFX', 23: 'APCC', 32: 'TEMP_GPU', 33: 'TEMP_CORE', 34: 'TEMP_MEM', 35: 'TEMP_EDGE',
              36: 'TEMP_HOTSPOT', 37: 'TEMP_SOC', 46: 'PROCHOT_CPU', 47: 'PROCHOT_GFX', 56: 'PPM', 57: 'FIT'}


def read():
    data = PATH.read_bytes()
    size, major, minor = struct.unpack_from('<HBB', data, 0)
    if major != 2:
        return {'format': f'{major}.{minor}'}
    temp_gfx, temp_soc = struct.unpack_from('<HH', data, 4)
    temp_core = [t for t in struct.unpack_from('<8H', data, 8) if t != 0xffff]  # (4 cores: the rest unused)
    gfx_activity, = struct.unpack_from('<H', data, 28)
    socket_mw, cpu_mw, soc_mw, gfx_mw = struct.unpack_from('<4H', data, 40)
    avg_gfxclk, = struct.unpack_from('<H', data, 64)
    cur_gfxclk, = struct.unpack_from('<H', data, 76)
    core_clk = [c for c in struct.unpack_from('<8H', data, 88) if c != 0xffff]
    throttle, = struct.unpack_from('<I', data, 108)
    indep, = struct.unpack_from('<Q', data, 120) if minor >= 2 and size >= 128 else (0,)
    return {'format': f'{major}.{minor}', 'gfx_activity': gfx_activity / 100, 'gfxclk_avg': avg_gfxclk,
            'gfxclk': cur_gfxclk, 'socket_w': socket_mw / 1000, 'cpu_w': cpu_mw / 1000, 'soc_w': soc_mw / 1000,
            'gfx_w': gfx_mw / 1000, 'temp_gfx': temp_gfx / 100, 'temp_soc': temp_soc / 100,
            'temp_core_max': max(temp_core, default=0) / 100, 'core_mhz': sum(core_clk) / max(len(core_clk), 1),
            'throttle': throttle, 'limits': [name for bit, name in THROTTLERS.items() if indep >> bit & 1]}


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    end = time.time() + seconds
    while time.time() < end:
        m = read()
        print(f"gfx {m['gfx_activity']:5.1f}% {m['gfxclk_avg']:4d}/{m['gfxclk']:4d} MHz  socket {m['socket_w']:5.2f} W "
              f"(cpu {m['cpu_w']:5.2f} soc {m['soc_w']:5.2f} gfx {m['gfx_w']:5.2f})  cores {m['core_mhz']:4.0f} MHz  "
              f"temp gfx {m['temp_gfx']:4.1f} soc {m['temp_soc']:4.1f} core {m['temp_core_max']:4.1f}  "
              f"throttle {m['throttle']:#x} {','.join(m['limits']) or '-'}", flush=True)
        time.sleep(interval)


if __name__ == '__main__':
    main()
