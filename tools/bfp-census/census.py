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
list such targets as roots instead. A roots file with an `indirect <file>` line
is closed over them: every function whose address the build takes must be
covered by the roots or excluded there with a reason, and every indirect call
site the roots reach must be declared there. Every allow-list entry is scoped
to ONE root, so an entry for one root never hides another root's hits.
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
# A call (blx) or tail call (bx) through a register. A `bx rN` whose rN a
# `pop` loaded within the last three instructions is a return (Thumb-1 pops
# the saved lr into a low register when the frame has stack arguments).
INDIRECT_CALL = re.compile(r"\t(blx|bx)\s+(r\d+|sb|sl|fp|ip)\b")
POP_REGS = re.compile(r"\tpop\s+\{([^}]*)\}")
# A function's address stored in data or loaded from a literal pool. The
# assembler keeps the function symbol on these (Thumb interworking needs its
# type), so a section-symbol ABS32 is a jump-table entry, not a function.
ADDR_RELOC = re.compile(r"^[0-9a-f]+\s+R_ARM_ABS32\s+(\S+)$")
RELOC_HDR = re.compile(r"^RELOCATION RECORDS FOR \[(.+)\]:$")
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
    # Pointer coverage: a target neither covered nor excluded is a problem, and
    # so is a covered function whose indirect call count is not the declared one;
    # an exclusion of a covered target or of a vanished one is stale.
    addr = {("v.c", "fwd"): {"t.c[.data]"}, ("v.c", "cold"): {"t.c[.data]"},
            ("v.c", "gone.part.0"): {"t.c[.data]"}}
    spec = {"exclude": {("v.c", "cold"): "never runs", ("v.c", "old"): "removed"},
            "site": {("d.c", "disp"): (1, "vtable")}}
    probs, stale = check_indirect(spec, {("v.c", "fwd"), ("d.c", "disp")}, addr,
                                  {("d.c", "disp"): 2})
    ind_ok = (probs == ["indirect target v.c:gone (address taken in t.c[.data]) is neither "
                        "covered by a root nor excluded",
                        "d.c:disp makes 2 indirect call(s), declared 1"]
              and stale == ["exclude v.c old: not an address-taken function of this build"])
    spec["exclude"][("v.c", "fwd")] = "wrong"
    _, stale2 = check_indirect(spec, {("v.c", "fwd"), ("d.c", "disp")}, addr, {("d.c", "disp"): 1})
    ind_ok = ind_ok and "exclude v.c fwd: covered by the roots" in stale2
    if not ind_ok:
        print(f"self-test FAIL indirect coverage: {probs} {stale} {stale2}", file=sys.stderr)
    failed = bool(bad) or not scoped_ok or not self_ok or not ind_ok
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
    union of phase files lists each root once. An `indirect <file>` line comes
    back as ["indirect", <absolute path>]."""
    if path.resolve() in stack:
        sys.exit(f"census: include cycle at {path}")
    out, seen = [], set()
    for parts in read_pairs(path):
        if parts[0] == "include" and len(parts) == 2:
            lines = read_roots(path.parent / parts[1], stack + (path.resolve(),))
        elif parts[0] == "indirect" and len(parts) == 2:
            lines = [["indirect", str((path.parent / parts[1]).resolve())]]
        else:
            lines = [parts]
        for line in lines:
            if tuple(line) not in seen:
                seen.add(tuple(line))
                out.append(line)
    return out


def parse_object(objdump, obj):
    """Return ({function: [callee symbols]}, {function: indirect call sites}) for one object."""
    text = subprocess.run([objdump, "-dr", str(obj)], capture_output=True, text=True, check=True).stdout
    funcs, indirect, cur = {}, defaultdict(int), None
    popped = []  # registers popped by each of the last three instructions
    pending, pending_name = False, None
    for line in text.splitlines():
        m = FUNC_HDR.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        if line.startswith(" ") and ":\t" in line:  # an instruction
            ind = INDIRECT_CALL.search(line)
            if ind and not (ind.group(1) == "bx" and any(ind.group(2) in regs for regs in popped)):
                indirect[cur] += 1
            pop = POP_REGS.search(line)
            popped = (popped + [{r.strip() for r in pop.group(1).split(",")} if pop else set()])[-3:]
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
    return funcs, dict(indirect)


def parse_addr_refs(objdump, obj):
    """Return [(section, symbol)] for every ABS32 relocation of one object: the
    address loads and data words that can make a symbol an indirect-call target
    (the caller keeps those that name a function)."""
    text = subprocess.run([objdump, "-r", str(obj)], capture_output=True, text=True, check=True).stdout
    refs, sec = [], None
    for line in text.splitlines():
        h = RELOC_HDR.match(line)
        if h:
            sec = h.group(1)
            continue
        r = ADDR_RELOC.match(line.strip())
        if r and not r.group(1).startswith("."):
            refs.append((sec, r.group(1).split("+")[0]))
    return refs


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


def read_indirect(path):
    """{"exclude": {(src, fn): reason}, "site": {(src, fn): (count, reason)}}.

    Line forms:
      exclude <source.c> <function> <reason ...>       an address-taken function
                                                       that does not run on the step
      site <source.c> <function> <count> <reason ...>  a covered function makes
                                                       exactly <count> indirect calls
    """
    spec = {"exclude": {}, "site": {}}
    for parts in read_pairs(path):
        if parts[0] == "exclude" and len(parts) >= 4:
            spec["exclude"][(parts[1], parts[2])] = " ".join(parts[3:])
        elif parts[0] == "site" and len(parts) >= 5 and parts[3].isdigit():
            spec["site"][(parts[1], parts[2])] = (int(parts[3]), " ".join(parts[4:]))
        else:
            sys.exit(f"census: bad indirect line (want 'exclude <src.c> <function> <reason>' or "
                     f"'site <src.c> <function> <count> <reason>'): {' '.join(parts)}")
    return spec


def check_indirect(spec, covered, addr_taken, indirect_sites):
    """Close the roots over calls through pointers.

    covered: (source, base name) of every function a root walked or names;
    addr_taken: {(source, function): {where the address is taken}};
    indirect_sites: {(source, function): indirect call sites}, whole build.
    Returns (problems, stale): an address-taken function neither covered nor
    excluded, or a covered function whose indirect call count differs from its
    `site` line, is a problem; an entry that no longer applies is stale.
    """
    norm = lambda d: {(s, base_name(f)): v for (s, f), v in d.items()}
    targets = defaultdict(set)
    for (src, fn), where in addr_taken.items():
        targets[(src, base_name(fn))] |= where
    sites = defaultdict(int)
    for (src, fn), n in indirect_sites.items():
        sites[(src, base_name(fn))] += n
    exclude, declared = norm(spec["exclude"]), norm(spec["site"])
    problems, stale = [], []
    for node in sorted(targets):
        if node in covered and node in exclude:
            stale.append(f"exclude {node[0]} {node[1]}: covered by the roots")
        elif node not in covered and node not in exclude:
            problems.append(f"indirect target {node[0]}:{node[1]} (address taken in "
                            f"{', '.join(sorted(targets[node]))}) is neither covered by a root "
                            f"nor excluded")
    for node in sorted(set(exclude) - set(targets)):
        stale.append(f"exclude {node[0]} {node[1]}: not an address-taken function of this build")
    for node in sorted(covered):
        want = declared.get(node, (0, ""))[0]
        if sites.get(node, 0) != want:
            problems.append(f"{node[0]}:{node[1]} makes {sites.get(node, 0)} indirect call(s), "
                            f"declared {want}")
    for node in sorted(set(declared) - covered):
        stale.append(f"site {node[0]} {node[1]}: not covered by the roots")
    return problems, stale


def load_build(build_dir, objdump):
    """Parse every object once: {"local": {(src, fn): callees}, "defined":
    {fn: [(src, fn)]}, "indirect": {(src, fn): n}, "addr": {(src, fn): {where}}}."""
    objs = sorted(Path(build_dir).rglob("*.c.obj")) + sorted(Path(build_dir).rglob("*.c.o"))
    if not objs:
        sys.exit(f"census: no *.c.obj under {build_dir} (build the arm_cross_m0plus preset first)")
    local, defined, indirect = {}, defaultdict(list), {}
    refs = []
    for o in objs:
        source = o.name.rsplit(".obj", 1)[0].rsplit(".o", 1)[0]  # Foo.c
        funcs, ind = parse_object(objdump, o)
        for fn, callees in funcs.items():
            local[(source, fn)] = callees
            defined[fn].append((source, fn))
        for fn, n in ind.items():
            indirect[(source, fn)] = n
        refs += [(source, sec, sym) for sec, sym in parse_addr_refs(objdump, o)]
    addr = defaultdict(set)
    for source, sec, sym in refs:
        # A static function is referenced from its own object; a global one
        # resolves by name. Data symbols and library functions are not in local.
        node = (source, sym) if (source, sym) in local else (
            defined[sym][0] if len(defined.get(sym, [])) == 1 else None)
        if node is not None:
            addr[node].add(f"{source}[{sec}]")
    return {"local": local, "defined": defined, "indirect": indirect, "addr": dict(addr)}


def run_census(build, roots, allow, out, err, verbose=False):
    """Walk every root of `roots` (read_roots lines), print the report to `out`
    and the failures to `err`. Returns the number of --strict failures (0 = the
    gate passes)."""
    local, defined = build["local"], build["defined"]
    unresolved = {}  # callee -> reason; printed once so silent drops are visible

    def resolve(src, sym):
        if (src, sym) in local:
            return (src, sym)
        cands = defined.get(sym, [])
        if len(cands) == 1:
            return cands[0]
        unresolved.setdefault(sym, "ambiguous" if cands else "undefined")
        return None

    indirect_files = [p[1] for p in roots if p[0] == "indirect"]
    if len(indirect_files) > 1:
        sys.exit(f"census: more than one indirect file: {indirect_files}")
    rows, missing, stale_all, covered = [], 0, [], set()
    for parts in roots:
        if parts[0] == "indirect":
            continue
        src, name = parts[0], parts[1]
        self_only = parts[2:] == ["self"]
        if parts[2:] and not self_only:
            sys.exit(f"census: bad root line (want '<src.c> <function> [self]'): {' '.join(parts)}")
        label = f"{src}:{name}" + (" (self)" if self_only else "")
        entry = allow.get(f"{src}:{name}")
        stops = entry["stop"] if entry else set()
        seeds = [k for k in local if k[0] == src and base_name(k[1]) == name]
        if not seeds:
            print(f"census: root {label} not found", file=err)
            missing += 1
            continue
        sites, seen, reached = walk_root(local, seeds, stops, self_only, resolve)
        covered |= {(s, base_name(f)) for s, f in list(seen) + seeds}
        hits, stale = apply_allow(sites, entry, reached)
        stale_all += [f"{label}: {s}" for s in stale]
        totals = defaultdict(int)
        for fnhits in hits.values():
            for (cat, _), n in fnhits.items():
                totals[cat] += n
        rows.append((label, len(seen), totals, hits))

    cats = [c for c, _ in CATEGORIES]
    w = max([len(r[0]) for r in rows] + [10])
    print(f"{'root':<{w}} {'fns':>4} " + " ".join(f"{c:>6}" for c in cats), file=out)
    grand = defaultdict(int)
    for label, nfn, totals, _ in rows:
        print(f"{label:<{w}} {nfn:>4} " + " ".join(f"{totals.get(c, 0):>6}" for c in cats), file=out)
        for c, n in totals.items():
            grand[c] += n
    print(f"{'(sum over roots; shared helpers counted per root)':<{w}}      "
          + " ".join(f"{grand.get(c, 0):>6}" for c in cats), file=out)

    unique = defaultdict(lambda: defaultdict(int))
    reached_by = defaultdict(set)
    for label, _, _, hits in rows:
        for node, fnhits in hits.items():
            reached_by[node].add(label)
            for k, n in fnhits.items():
                unique[node][k] = n
    print("\nhit functions (each counted once):", file=out)
    for node in sorted(unique, key=lambda n: -sum(unique[n].values())):
        detail = ", ".join(f"{s} x{n}" for (_, s), n in sorted(unique[node].items()))
        line = f"  {node[0]}:{node[1]}  {detail}"
        if verbose:
            line += f"\n      reached from: {', '.join(sorted(reached_by[node]))}"
        print(line, file=out)
    for sym, why in sorted(unresolved.items()):
        print(f"census: warning: callee {sym} is {why}; not followed", file=err)
    n_hit = sum(sum(h.values()) for h in unique.values())
    n_gate = sum(n for h in unique.values() for (c, _), n in h.items() if c in GATE_CATS)
    print(f"\ntotal non-allow-listed hit call sites (unique): {n_hit} "
          f"(gated {'/'.join(c for c, _ in CATEGORIES if c in GATE_CATS)}: {n_gate})", file=out)
    for line in stale_all:
        print(f"census: stale allow-list entry: {line}", file=err)

    n_ind = 0
    if indirect_files:
        spec = read_indirect(Path(indirect_files[0]))
        problems, stale = check_indirect(spec, covered, build["addr"], build["indirect"])
        n_targets = len({(s, base_name(f)) for s, f in build["addr"]})
        n_sites = len(covered & {(s, base_name(f)) for s, f in build["indirect"]})
        print(f"\nindirect calls ({Path(indirect_files[0]).name}): {n_targets} address-taken "
              f"functions, {len(spec['exclude'])} excluded; {n_sites} covered functions call "
              f"through a pointer; problems {len(problems)}, stale {len(stale)}", file=out)
        for line in problems:
            print(f"census: {line}", file=err)
        for line in stale:
            print(f"census: stale indirect entry: {line}", file=err)
        n_ind = len(problems) + len(stale)
    return (1 if n_gate else 0) + missing + len(stale_all) + n_ind


# Each control breaks the step gate one way; --strict must fail on every one.
# Forbidden helpers injected (in memory) into the indirect targets the step
# reaches only through a pointer: the remat HEAP scheduler (s->fns), the
# layers' calcOutputShape (the scheduler's wire table), the ReLU forward
# dispatcher and the loader's getBatch.
STEP_INJECTIONS = (
    [("RematHeap.c", f, "__aeabi_fmul") for f in
     ("rematHeapBegin", "rematHeapNext", "rematHeapDone", "rematHeapEnd", "rematHeapDeinit")]
    + [("Relu.c", "reluForward", "__aeabi_uidiv"), ("Relu.c", "reluForward", "__aeabi_lmul"),
       ("DataLoaderApi.c", "getBatch", "__aeabi_fadd")]
    + [(f"{src}.c", f"{fn}CalcOutputShape", "__aeabi_lmul") for src, fn in
       (("QuantizationLayer", "quantization"), ("Conv1d", "conv1d"), ("Relu", "relu"),
        ("MaxPool1d", "maxPool1d"), ("AvgPool1d", "avgPool1d"), ("Flatten", "flatten"),
        ("Linear", "linear"), ("Softmax", "softmax"))])


def negative_controls(build, roots, allow):
    """Run the step gate unmodified (must pass) and under each control (must
    fail). Returns the number of controls that did not behave."""
    import copy
    import io
    bad = 0

    def gate(b, r, what, want_fail, needle=None):
        """needle: text the report must contain, so a control fails for its
        own reason and not for an unrelated one."""
        nonlocal bad
        out, err = io.StringIO(), io.StringIO()
        failures = run_census(b, r, allow, out, err)
        text = out.getvalue() + err.getvalue()
        found = (needle is None or (needle.search(text) if hasattr(needle, "search")
                                    else needle in text))
        ok = (failures > 0) == want_fail and bool(found)
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {what}: {failures} strict failure(s)")

    gate(build, roots, "unmodified step gate passes", False)
    for src, fn, sym in STEP_INJECTIONS:
        b = dict(build, local=copy.copy(build["local"]))
        nodes = [k for k in b["local"] if k[0] == src and base_name(k[1]) == fn]
        if not nodes:
            print(f"  FAIL {src}:{fn} not in the build")
            bad += 1
            continue
        for k in nodes:
            b["local"][k] = b["local"][k] + [sym]
        # The hit line itself: "  <src>:<fn>[.clone]  ... <sym> x<n> ...".
        gate(b, roots, f"{sym} injected into {src}:{fn}", True,
             re.compile(rf"^  {re.escape(src)}:{re.escape(fn)}\S*  .*\b{sym} x\d", re.M))
    # The pointer coverage itself: an unrooted target and an undeclared site.
    dropped = [p for p in roots if p[:2] != ["RematHeap.c", "rematHeapNext"]]
    gate(build, dropped, "root RematHeap.c:rematHeapNext dropped", True,
         "indirect target RematHeap.c:rematHeapNext")
    node = next(k for k in build["indirect"] if k[0] == "RematScheduler.c"
                and base_name(k[1]) == "rematNext")
    b = dict(build, indirect={**build["indirect"], node: build["indirect"][node] + 1})
    gate(b, roots, "extra indirect call in RematScheduler.c:rematNext", True,
         "RematScheduler.c:rematNext makes 2 indirect call(s), declared 1")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build_dir", nargs="?", default="build/arm_cross_m0plus")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on a gated hit, a missing root, a stale allow-list entry or "
                         "an uncovered indirect call")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every hit function per root")
    ap.add_argument("--self-test", action="store_true",
                    help="check the classifier and the gate logic; with a built BUILD_DIR also "
                         "run the step gate's negative controls")
    ap.add_argument("--roots", default=str(HERE / "har_forward_roots.txt"),
                    help="roots file (default: the HAR forward gate set; "
                         "har_train_step_roots.txt = the whole HAR training step; "
                         "forward_roots.txt = every BFP forward kernel)")
    ap.add_argument("--allow", default=str(HERE / "allowlist.txt"))
    args = ap.parse_args()

    if args.self_test:
        failed = self_test()
        has_build = any(Path(args.build_dir).rglob("*.c.obj"))
        if not has_build:
            print(f"negative controls SKIPPED: no objects under {args.build_dir}")
            return failed
        print("negative controls (har_train_step_roots.txt):")
        bad = negative_controls(load_build(args.build_dir, find_objdump()),
                                read_roots(HERE / "har_train_step_roots.txt"),
                                read_allow(Path(args.allow)))
        print("negative controls " + ("FAILED" if bad else "ok"))
        return 1 if failed or bad else 0
    build = load_build(args.build_dir, find_objdump())
    failures = run_census(build, read_roots(Path(args.roots)), read_allow(Path(args.allow)),
                          sys.stdout, sys.stderr, args.verbose)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    sys.exit(main())
