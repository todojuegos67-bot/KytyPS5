#!/usr/bin/env python3
"""Disassemble guest x86-64 code out of the running emulator.

Guest modules are identity-mapped in kyty_emulator.exe, so a guest address is also an address
in the emulator process. The bytes are read with ReadProcessMemory, wrapped as .byte lines,
assembled by clang and disassembled by llvm-objdump at the guest address.

    guest-disasm.py 0x9002dd700 [--size 0x100] [--mark 0x9002dd73b] [--pid N]
    guest-disasm.py 0x9002dd700 --dump 64       hex words instead (data)
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import tempfile

LLVM = r'C:\Program Files\LLVM\bin'
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400


def emulator_pid():
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq kyty_emulator.exe', '/FO', 'CSV', '/NH'],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        fields = [f.strip('"') for f in line.split('","')]
        if len(fields) > 1 and fields[0].lower().startswith('kyty_emulator'):
            return int(fields[1])
    sys.exit('kyty_emulator.exe is not running')


def read(pid, address, size):
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_size_t)]
    handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not handle:
        sys.exit(f'OpenProcess failed: {ctypes.get_last_error()}')
    buffer = ctypes.create_string_buffer(size)
    got = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(got))
    kernel32.CloseHandle(handle)
    if not ok:
        sys.exit(f'ReadProcessMemory({address:#x}, {size:#x}) failed: {ctypes.get_last_error()}')
    return buffer.raw[:got.value]


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [('BaseAddress', ctypes.c_void_p), ('AllocationBase', ctypes.c_void_p),
                ('AllocationProtect', wt.DWORD), ('PartitionId', wt.WORD), ('RegionSize', ctypes.c_size_t),
                ('State', wt.DWORD), ('Protect', wt.DWORD), ('Type', wt.DWORD)]


def regions(pid, low, high, executable_only):
    """Committed regions of the guest range (execute permission only, if asked)."""
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MEMORY_BASIC_INFORMATION),
                                        ctypes.c_size_t]
    kernel32.VirtualQueryEx.restype = ctypes.c_size_t
    handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    info = MEMORY_BASIC_INFORMATION()
    address, found = low, []
    while address < high and kernel32.VirtualQueryEx(handle, ctypes.c_void_p(address), ctypes.byref(info),
                                                     ctypes.sizeof(info)):
        base, size = info.BaseAddress or 0, info.RegionSize
        executable = info.Protect & 0xF0  # PAGE_EXECUTE*
        if info.State == 0x1000 and (executable or not executable_only) and not info.Protect & 0x100:
            found.append((base, size))
        address = base + size
    kernel32.CloseHandle(handle)
    return found


def find(pid, pattern, low, high, executable_only):
    hits, step = [], 1 << 24
    for base, size in regions(pid, low, high, executable_only):
        for offset in range(0, size, step):
            # Chunks overlap by the pattern length so a match across a boundary is seen.
            chunk = read(pid, base + offset, min(step + len(pattern) - 1, size - offset))
            at = chunk.find(pattern)
            while at >= 0:
                hits.append(base + offset + at)
                at = chunk.find(pattern, at + 1)
    return sorted(set(hits))


def xrefs(pid, target, low, high):
    """Positions of rip-relative displacements that resolve to target (the displacement ends
    the instruction, or is followed by an 8- or 32-bit immediate)."""
    import numpy as np
    hits = []
    for base, size in regions(pid, low, high, True):
        code = np.frombuffer(read(pid, base, size), dtype=np.uint8)
        if code.size < 8:
            continue
        disp = (code[:-3].astype(np.int64) | code[1:-2].astype(np.int64) << 8 | code[2:-1].astype(np.int64) << 16 |
                code[3:].astype(np.int64) << 24)
        disp = np.where(disp >= 1 << 31, disp - (1 << 32), disp)
        position = np.arange(disp.size, dtype=np.int64) + base
        for tail in (4, 5, 8):
            for at in np.nonzero(position + tail + disp == target)[0]:
                hits.append((int(base + at), tail))
    return sorted(hits)


def disassemble(code, address, marks):
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, 'code.s')
        obj = os.path.join(tmp, 'code.o')
        with open(source, 'w') as f:
            f.write('.text\n')
            for i in range(0, len(code), 16):
                f.write('.byte ' + ','.join(str(b) for b in code[i:i + 16]) + '\n')
        subprocess.run([os.path.join(LLVM, 'clang.exe'), '--target=x86_64-pc-linux-gnu', '-c', source, '-o', obj],
                       check=True)
        out = subprocess.run([os.path.join(LLVM, 'llvm-objdump.exe'), '-d', '--no-show-raw-insn',
                              '-M', 'intel', f'--adjust-vma={address:#x}', obj],
                             capture_output=True, text=True, check=True).stdout
    lines = []
    for line in out.splitlines():
        head = line.split(':', 1)[0].strip()
        try:
            marked = int(head, 16) in marks
        except ValueError:
            marked = False
        if head and all(c in '0123456789abcdef' for c in head):
            lines.append(('>>> ' if marked else '    ') + line.strip())
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('address')
    p.add_argument('--size', default='0x100')
    p.add_argument('--mark', action='append', default=[])
    p.add_argument('--pid', type=int)
    p.add_argument('--dump', type=int, help='print this many 64-bit words instead of disassembling')
    p.add_argument('--find', help='hex bytes to search for in guest memory (address = range low:high)')
    p.add_argument('--data', action='store_true', help='--find: search data regions too')
    p.add_argument('--regions', action='store_true', help='list the executable regions of the range')
    p.add_argument('--xref', help='rip-relative references to this address (address = range low:high)')
    a = p.parse_args()
    pid = a.pid or emulator_pid()
    if a.xref:
        low, high = (int(v, 0) for v in a.address.split(':'))
        for at, tail in xrefs(pid, int(a.xref, 0), low, high):
            print(f'{at:012x} (displacement +{tail})')
        return
    if a.find or a.regions:
        low, high = (int(v, 0) for v in a.address.split(':'))
        if a.regions:
            for base, size in regions(pid, low, high, not a.data):
                print(f'{base:012x}-{base + size:012x} {size >> 10} KiB')
            return
        for hit in find(pid, bytes.fromhex(a.find), low, high, not a.data):
            print(f'{hit:012x}')
        return
    address = int(a.address, 0)
    if a.dump:
        data = read(pid, address, a.dump * 8)
        for i in range(0, len(data), 8):
            word = int.from_bytes(data[i:i + 8], 'little')
            print(f'{address + i:012x}: {word:016x}')
        return
    size = int(a.size, 0)
    print(disassemble(read(pid, address, size), address, {int(m, 0) for m in a.mark}))


if __name__ == '__main__':
    main()
