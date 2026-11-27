#!/usr/bin/env python3
"""perfil-simbolos.py: symbolizes a logs/perfil-<stamp>.txt (KYTY_PERFIL=1, src/local/perfil.h) with the same build's
link map (_Build/windows/kyty_emulator_clang_lld_link.map, lld-link /lldmap).

  python3 tools/perfil-simbolos.py perfil-XXXX.txt kyty_emulator_clang_lld_link.map [--base 0x140000000] [--top 60]

Prints: modules, self time per function, inclusive time per function (each function once per stack), top stacks.
exe RVAs are mapped with the map (addresses are VAs; the exe is linked /DYNAMICBASE:NO at 0x140000000).
Names are demangled with llvm-undname when it is on PATH.
"""
import argparse
import bisect
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

HEX = re.compile(r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+(\d+)\s+(.*)$")


def load_map(path):
    syms = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = HEX.match(line)
            if not m:
                continue
            rest = m.group(4).strip()
            if not rest or rest.startswith(".") or ":(" in rest or rest.endswith(".obj") or rest.endswith(".lib"):
                continue
            addr = int(m.group(1), 16)
            if addr == 0:
                continue
            syms.append((addr, int(m.group(2), 16), rest))
    syms.sort()
    return syms


class Symbolizer:
    def __init__(self, syms, base):
        self.syms = syms
        self.addrs = [s[0] for s in syms]
        self.base = base
        self.cache = {}

    def func(self, token):
        """'exe+0x1234' -> function name; other modules stay 'module' (+RVA for self listings)."""
        if not token.startswith("exe+0x"):
            return token.split("+")[0]
        if token in self.cache:
            return self.cache[token]
        va = self.base + int(token[6:], 16)
        i = bisect.bisect_right(self.addrs, va) - 1
        name = "exe+" + token[6:]
        if i >= 0:
            a, size, sym = self.syms[i]
            if size == 0 or va < a + size or (i + 1 < len(self.syms) and va < self.syms[i + 1][0]):
                name = sym
        self.cache[token] = name
        return name


def demangle_all(names):
    tool = shutil.which("llvm-undname") or shutil.which("undname")
    mangled = [n for n in names if n.startswith("?")]
    if not tool or not mangled:
        return {}
    try:
        out = subprocess.run([tool], input="\n".join(mangled) + "\n", capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    res = {}
    lines = [l for l in out.splitlines() if l.strip()]
    # llvm-undname echoes each input then its result
    for i in range(0, len(lines) - 1, 2):
        res[lines[i].strip()] = lines[i + 1].strip()
    return res


def parse(path):
    sec = None
    total = 0
    data = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("samples "):
                total = int(line.split()[1])
            elif line.startswith("["):
                sec = line.split("]")[0][1:]
            elif line and sec and not line.startswith("#"):
                data[sec].append(line.split())
    return total, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perfil")
    ap.add_argument("map")
    ap.add_argument("--base", default="0x140000000")
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()
    total, data = parse(a.perfil)
    if total == 0:
        sys.exit("sin muestras")
    sym = Symbolizer(load_map(a.map), int(a.base, 16))

    self_f = Counter()
    for c, tok in data.get("raw self", []):
        self_f[sym.func(tok)] += int(c)
    # Inclusive per function: from the stacks (each function once per stack), which cover the top 200 only, so also
    # report inclusive per call site summed per function from [raw inclusive] (may double count recursion).
    incl_f = Counter()
    for row in data.get("raw inclusive", []):
        incl_f[sym.func(row[1])] += int(row[0])
    stacks = []
    for row in data.get("stacks top 200", []):
        stacks.append((int(row[0]), row[2:]))
    names = set(self_f) | set(incl_f) | {sym.func(t) for _, fr in stacks for t in fr}
    dm = demangle_all(names)
    d = lambda n: dm.get(n, n)

    pct = lambda c: 100.0 * c / total
    print(f"muestras {total}")
    print("\n== modulos (hoja) ==")
    for row in data.get("modules", []):
        print("  ", " ".join(row))
    print(f"\n== self por funcion (top {a.top}) ==")
    for n, c in self_f.most_common(a.top):
        print(f"{c:8d} {pct(c):6.2f}%  {d(n)}")
    print("\n== inclusivo por funcion (top 40) ==")
    for n, c in incl_f.most_common(40):
        print(f"{c:8d} {pct(c):6.2f}%  {d(n)}")
    print("\n== pilas (top 25, hoja -> llamadores) ==")
    for c, fr in stacks[:25]:
        chain, last = [], None
        for t in fr:
            n = d(sym.func(t))
            if n != last:
                chain.append(n)
            last = n
        print(f"{c:8d} {pct(c):6.2f}%  " + "  <-  ".join(x[:90] for x in chain[:8]))


if __name__ == "__main__":
    main()
