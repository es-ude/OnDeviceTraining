#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Float census for the BFP forward kernels (Cortex-M0+ objects).

Disassembles every object of an arm_cross_m0plus build, builds the call graph
and, for each root in forward_roots.txt, counts the call sites of float /
double / libm / division / int64 / memcpy / clz helpers that the root can
reach. On a core without FPU or divider each such call is real cost, so the
goal is "no hits outside the allow-listed boundaries".

  uv run tools/bfp-census/census.py [BUILD_DIR] [--strict] [-v]

Calls through function pointers (vtables, kernel callbacks) are not followed;
list such targets as roots instead.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

CATEGORIES = [
    ("float", re.compile(
        r"^__aeabi_(f(add|sub|rsub|mul|div|neg|cmp\w*|cmpun)|cfcmp\w*|cfrcmple|"
        r"i2f|ui2f|l2f|ul2f|f2i\w*|f2u\w*|f2l\w*|f2ul\w*)$")),
    ("double", re.compile(r"^__aeabi_(d\w+|cdcmp\w*|cdrcmple|i2d|ui2d|l2d|ul2d|f2d)$")),
    ("libm", re.compile(
        r"^(ldexp|frexp|pow|round|fabs|exp|exp2|log|log2|log10|sqrt|cbrt|floor|ceil|fmod|lround|"
        r"lrint|rint|nearbyint|trunc|fmin|fmax|fma|expm1|log1p|modf|scalbn|copysign|hypot|"
        r"sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh)f?$")),
    ("div", re.compile(
        r"^__aeabi_(ui|i|ul|l)?div(mod)?$|^__(u)?div(si|di)3$|^__(u)?mod(si|di)3$")),
    ("int64", re.compile(r"^__aeabi_(lmul|llsl|llsr|lasr|lcmp|ulcmp)$")),
    ("mem", re.compile(r"^(memcpy|memmove|memset|__aeabi_mem(cpy|move|set)\d*)$")),
    ("clz", re.compile(r"^__(clz|ctz|popcount)(si|di)2$")),
]
CALL_RELOC = re.compile(r"R_ARM_THM_(CALL|JUMP24|JUMP11|JUMP19)\s+(\S+)")
FUNC_HDR = re.compile(r"^[0-9a-f]+ <(.+)>:$")
DIRECT_BRANCH = re.compile(r"\t(?:bl|b|b\.w|b\.n|bl\.w)\w*\s+[0-9a-f]+ <([^>+]+)(?:\+0x[0-9a-f]+)?>")


def category(sym):
    for name, rx in CATEGORIES:
        if rx.match(sym):
            return name
    return None


def self_test():
    expect = {
        "__aeabi_uidiv": "div", "__aeabi_uidivmod": "div", "__aeabi_idiv": "div",
        "__aeabi_idivmod": "div", "__aeabi_ldivmod": "div", "__aeabi_uldivmod": "div",
        "ldexpf": "libm", "cosf": "libm", "fmaf": "libm", "sqrtf": "libm",
        "__aeabi_fadd": "float", "__aeabi_i2f": "float", "__aeabi_f2d": "double",
        "__aeabi_lmul": "int64", "memcpy": "mem", "__clzsi2": "clz",
        "validateBfpQConfigShape": None, "exit": None,
    }
    bad = {k: (category(k), v) for k, v in expect.items() if category(k) != v}
    for k, (got, want) in bad.items():
        print(f"self-test FAIL {k}: got {got}, want {want}", file=sys.stderr)
    print("self-test " + ("FAILED" if bad else "ok"))
    return 1 if bad else 0


def find_objdump():
    for cand in (os.environ.get("ARM_OBJDUMP"), shutil.which("arm-none-eabi-objdump")):
        if cand:
            return cand
    sys.exit("census: arm-none-eabi-objdump not found (put the Arm toolchain on PATH or set ARM_OBJDUMP)")


def read_pairs(path):
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line.split())
    return out


def parse_object(objdump, obj):
    """Return {function: [callee symbols]} for one object."""
    text = subprocess.run([objdump, "-dr", str(obj)], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    pending, pending_name = False, None
    for line in text.splitlines():
        m = FUNC_HDR.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        r = CALL_RELOC.search(line)
        if r:
            # The relocation names the real target; the preceding instruction
            # line only shows a placeholder, so drop the edge it recorded.
            if pending:
                funcs[cur].pop()
                pending = False
            sym = r.group(2).split("+")[0]
            if not sym.startswith("."):  # section symbol: keep the direct form's name
                funcs[cur].append(sym)
            elif pending_name:
                funcs[cur].append(pending_name)
            pending_name = None
            continue
        pending, pending_name = False, None
        d = DIRECT_BRANCH.search(line)
        if d and d.group(1) != cur:
            funcs[cur].append(d.group(1))
            pending, pending_name = True, d.group(1)
    return funcs


def base_name(sym):
    return re.sub(r"(\.(constprop|isra|part|cold|lto_priv)\.?\d*)+$", "", sym)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_dir", nargs="?", default="build/arm_cross_m0plus")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any non-allow-listed hit")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every hit function per root")
    ap.add_argument("--self-test", action="store_true", help="check the symbol classifier and exit")
    ap.add_argument("--roots", default=str(HERE / "forward_roots.txt"))
    ap.add_argument("--allow", default=str(HERE / "allowlist.txt"))
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    objdump = find_objdump()
    objs = sorted(Path(args.build_dir).rglob("*.c.obj")) + sorted(Path(args.build_dir).rglob("*.c.o"))
    if not objs:
        sys.exit(f"census: no *.c.obj under {args.build_dir} (build the arm_cross_m0plus preset first)")

    local = {}      # (source, func) -> callees
    defined = defaultdict(list)  # func -> [(source, func)] for cross-object lookup
    for o in objs:
        source = o.name.rsplit(".obj", 1)[0].rsplit(".o", 1)[0]  # Foo.c
        for fn, callees in parse_object(objdump, o).items():
            local[(source, fn)] = callees
            defined[fn].append((source, fn))

    unresolved = {}  # callee -> reason; printed once so silent drops are visible

    def resolve(src, sym):
        if (src, sym) in local:
            return (src, sym)
        cands = defined.get(sym, [])
        if len(cands) == 1:
            return cands[0]
        unresolved.setdefault(sym, "ambiguous" if cands else "undefined")
        return None

    allow_stop, allow_sym = set(), set()
    for parts in read_pairs(Path(args.allow)):
        if len(parts) == 1:
            allow_stop.add(parts[0])
        else:
            allow_sym.add((parts[0], parts[1]))

    rows, strict_fail = [], False
    for parts in read_pairs(Path(args.roots)):
        src, name = parts[0], parts[1]
        seeds = [k for k in local if k[0] == src and base_name(k[1]) == name]
        if not seeds:
            print(f"census: root {src}:{name} not found in {args.build_dir}", file=sys.stderr)
            strict_fail = True
            continue
        seen, stack, hits = set(), list(seeds), defaultdict(lambda: defaultdict(int))
        while stack:
            node = stack.pop()
            if node in seen or base_name(node[1]) in allow_stop:
                continue
            seen.add(node)
            for callee in local[node]:
                cat = category(callee)
                if cat:
                    if (base_name(node[1]), callee) not in allow_sym:
                        hits[node][(cat, callee)] += 1
                    continue
                nxt = resolve(node[0], callee)
                if nxt:
                    stack.append(nxt)
        totals = defaultdict(int)
        for fnhits in hits.values():
            for (cat, _), n in fnhits.items():
                totals[cat] += n
        rows.append((f"{src}:{name}", len(seen), totals, hits))

    cats = [c for c, _ in CATEGORIES]
    w = max([len(r[0]) for r in rows] + [10])
    print(f"{'root':<{w}} {'fns':>4} " + " ".join(f"{c:>6}" for c in cats))
    grand = defaultdict(int)
    for label, nfn, totals, _ in rows:
        print(f"{label:<{w}} {nfn:>4} " + " ".join(f"{totals.get(c, 0):>6}" for c in cats))
        for c, n in totals.items():
            grand[c] += n
    print(f"{'(sum over roots; shared helpers counted per root)':<{w}}      " + " ".join(f"{grand.get(c, 0):>6}" for c in cats))

    unique = defaultdict(lambda: defaultdict(int))
    reached_by = defaultdict(set)
    for label, _, _, hits in rows:
        for node, fnhits in hits.items():
            reached_by[node].add(label)
            for k, n in fnhits.items():
                unique[node][k] = n
    print("\nhit functions (each counted once):")
    for node in sorted(unique, key=lambda n: -sum(unique[n].values())):
        detail = ", ".join(f"{s} x{n}" for (_, s), n in sorted(unique[node].items()))
        line = f"  {node[0]}:{node[1]}  {detail}"
        if args.verbose:
            line += f"\n      reached from: {', '.join(sorted(reached_by[node]))}"
        print(line)
    for sym, why in sorted(unresolved.items()):
        print(f"census: warning: callee {sym} is {why}; not followed", file=sys.stderr)
    n_hit = sum(sum(h.values()) for h in unique.values())
    print(f"\ntotal non-allow-listed hit call sites (unique): {n_hit}")

    if args.strict and (n_hit or strict_fail):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
