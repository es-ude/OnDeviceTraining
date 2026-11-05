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

  uv run tools/bfp-census/census.py [BUILD_DIR] [--roots FILE] [--strict] [-v]

Calls through function pointers (vtables, kernel callbacks) are not followed;
list such targets as roots instead. Every allow-list entry is scoped to ONE
root, so an entry for one root never hides another root's hits.
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
# --strict gates these; mem and clz are reported only (per-call copies/zeroing).
GATE_CATS = {"float", "double", "libm", "div", "int64"}
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
    # Scoped allow-list: an entry for root A must not touch root B's sites, an
    # excess site over the count stays a hit, and a missing site is stale.
    allow = {"A.c:a": {"sym": {("f", "__aeabi_uidiv"): 2}, "stop": set()}}
    sites = {("x.c", "f"): {("div", "__aeabi_uidiv"): 3}}
    hits_a, stale_a = apply_allow(sites, allow.get("A.c:a"), set())
    hits_b, _ = apply_allow(sites, allow.get("B.c:b"), set())
    _, stale_c = apply_allow({}, allow.get("A.c:a"), set())
    scoped_ok = (hits_a == {("x.c", "f"): {("div", "__aeabi_uidiv"): 1}} and not stale_a
                 and hits_b == sites and stale_c == ["f __aeabi_uidiv: allowed 2, found 0"])
    if not scoped_ok:
        print(f"self-test FAIL scoped allow-list: {hits_a} {stale_a} {hits_b} {stale_c}",
              file=sys.stderr)
    # Self-only root: its own float call is a hit, its callees' are not.
    graph = {("d.c", "disp"): ["__aeabi_fmul", "armF"], ("d.c", "armF"): ["__aeabi_fadd"]}
    res = lambda src, sym: (src, sym) if (src, sym) in graph else None
    own, _, _ = walk_root(graph, [("d.c", "disp")], set(), True, res)
    full, _, _ = walk_root(graph, [("d.c", "disp")], set(), False, res)
    own_hits, _ = apply_allow(own, None, set())
    self_ok = (own_hits == {("d.c", "disp"): {("float", "__aeabi_fmul"): 1}}
               and ("d.c", "armF") in full)
    if not self_ok:
        print(f"self-test FAIL self-only root: {own_hits} {dict(full)}", file=sys.stderr)
    failed = bool(bad) or not scoped_ok or not self_ok
    print("self-test " + ("FAILED" if failed else "ok"))
    return 1 if failed else 0


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


def read_roots(path, stack=()):
    """Root lines of a roots file, `include <file>` lines expanded in place
    (relative to the including file); a root line seen before is dropped, so a
    union of phase files lists each root once."""
    if path.resolve() in stack:
        sys.exit(f"census: include cycle at {path}")
    out, seen = [], set()
    for parts in read_pairs(path):
        lines = (read_roots(path.parent / parts[1], stack + (path.resolve(),))
                 if parts[0] == "include" and len(parts) == 2 else [parts])
        for line in lines:
            if tuple(line) not in seen:
                seen.add(tuple(line))
                out.append(line)
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


def read_allow(path):
    """{root: {"stop": {function}, "sym": {(function, symbol): count}}}.

    Line forms (root = <source.c>:<function>, as the roots file names it):
      <root> <function>                  traversal stops at <function> under <root>
      <root> <function> <symbol> <count> <count> call sites of <symbol> inside
                                         <function> are allowed under <root>
    """
    allow = defaultdict(lambda: {"stop": set(), "sym": {}})
    for parts in read_pairs(path):
        if ":" not in parts[0] or len(parts) not in (2, 4):
            sys.exit(f"census: bad allow-list line (want '<src.c>:<root> <function> "
                     f"[<symbol> <count>]'): {' '.join(parts)}")
        if len(parts) == 2:
            allow[parts[0]]["stop"].add(parts[1])
        else:
            allow[parts[0]]["sym"][(parts[1], parts[2])] = int(parts[3])
    return allow


def walk_root(local, seeds, stops, self_only, resolve):
    """Collect the call sites a root reaches: {node: {(cat, sym): n}}, plus the
    nodes entered and the base names reached (stops included).

    self_only: count the seeds' own call sites and follow nothing -- for the
    dtype dispatchers, whose callees span every dtype's arm.
    """
    seen, reached, stack = set(), set(), list(seeds)
    sites = defaultdict(lambda: defaultdict(int))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        if base_name(node[1]) in stops:
            reached.add(base_name(node[1]))
            continue
        seen.add(node)
        reached.add(base_name(node[1]))
        for callee in local[node]:
            cat = category(callee)
            if cat:
                sites[node][(cat, callee)] += 1
                continue
            if self_only:
                continue
            nxt = resolve(node[0], callee)
            if nxt:
                stack.append(nxt)
    return sites, seen, reached


def apply_allow(sites, entry, reached):
    """Split one root's call sites into hits and stale entries.

    sites: {node: {(cat, sym): n}}; entry: that root's allow-list (or None);
    reached: base names of the functions the walk entered or stopped at.
    """
    entry = entry or {"stop": set(), "sym": {}}
    per_key = defaultdict(int)
    for node, fnhits in sites.items():
        for (_, sym), n in fnhits.items():
            per_key[(base_name(node[1]), sym)] += n
    stale = []
    for (fn, sym), allowed in sorted(entry["sym"].items()):
        if per_key.get((fn, sym), 0) < allowed:
            stale.append(f"{fn} {sym}: allowed {allowed}, found {per_key.get((fn, sym), 0)}")
    for fn in sorted(entry["stop"] - reached):
        stale.append(f"{fn}: stop entry never reached")
    budget = dict(entry["sym"])
    hits = defaultdict(lambda: defaultdict(int))
    for node in sorted(sites):
        for (cat, sym), n in sorted(sites[node].items()):
            key = (base_name(node[1]), sym)
            take = min(n, budget.get(key, 0))
            if key in budget:
                budget[key] -= take
            if n - take:
                hits[node][(cat, sym)] += n - take
    return {k: dict(v) for k, v in hits.items()}, stale


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_dir", nargs="?", default="build/arm_cross_m0plus")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on a gated hit, a missing root or a stale allow-list entry")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every hit function per root")
    ap.add_argument("--self-test", action="store_true", help="check the symbol classifier and exit")
    ap.add_argument("--roots", default=str(HERE / "har_forward_roots.txt"),
                    help="roots file (default: the HAR forward gate set; "
                         "har_train_step_roots.txt = the whole HAR training step; "
                         "forward_roots.txt = every BFP forward kernel)")
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

    allow = read_allow(Path(args.allow))

    rows, strict_fail, stale_all = [], False, []
    for parts in read_roots(Path(args.roots)):
        src, name = parts[0], parts[1]
        self_only = parts[2:] == ["self"]
        if parts[2:] and not self_only:
            sys.exit(f"census: bad root line (want '<src.c> <function> [self]'): {' '.join(parts)}")
        label = f"{src}:{name}" + (" (self)" if self_only else "")
        entry = allow.get(f"{src}:{name}")
        stops = entry["stop"] if entry else set()
        seeds = [k for k in local if k[0] == src and base_name(k[1]) == name]
        if not seeds:
            print(f"census: root {label} not found in {args.build_dir}", file=sys.stderr)
            strict_fail = True
            continue
        sites, seen, reached = walk_root(local, seeds, stops, self_only, resolve)
        hits, stale = apply_allow(sites, entry, reached)
        stale_all += [f"{label}: {s}" for s in stale]
        totals = defaultdict(int)
        for fnhits in hits.values():
            for (cat, _), n in fnhits.items():
                totals[cat] += n
        rows.append((label, len(seen), totals, hits))

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
    n_gate = sum(n for h in unique.values() for (c, _), n in h.items() if c in GATE_CATS)
    print(f"\ntotal non-allow-listed hit call sites (unique): {n_hit} "
          f"(gated {'/'.join(c for c, _ in CATEGORIES if c in GATE_CATS)}: {n_gate})")
    for line in stale_all:
        print(f"census: stale allow-list entry: {line}", file=sys.stderr)

    if args.strict and (n_gate or strict_fail or stale_all):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
