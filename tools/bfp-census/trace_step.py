#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Record the functions a bfp8full HAR training step executes (both heads).

Builds the HAR BFP trainer on the host with -finstrument-functions and
step_trace_hook.c, runs bfp8full_hflt and bfp8full_hbfp for three steps each
and writes the union of the executed function names to
har_train_step_trace.txt. census.py checks every name there that is a
function of the arm objects against the step roots (its `trace` line), so
re-run this after changing what the step calls.

  uv run tools/bfp-census/trace_step.py [--build-dir DIR] [--out FILE]
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONFIGS = ("bfp8full_hflt", "bfp8full_hbfp")
TARGET = "train_c_har_classifier_bfp"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"trace_step: {' '.join(map(str, cmd))} failed ({r.returncode}):\n"
                 f"{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r.stdout


def read_link_map(path):
    """{link-time address: (translation unit, function)} from an ld64 link map
    (-Wl,-map): the object table names each symbol's translation unit, so two
    static functions of one name stay apart."""
    files, syms, section = {}, {}, None
    for line in path.read_text(errors="replace").splitlines():
        head = re.match(r"^# ([A-Za-z ]+):", line)
        if head:
            section = head.group(1)
            continue
        if section == "Object files":
            m = re.match(r"^\[\s*(\d+)\]\s+(.*)$", line)
            if m:
                obj = re.search(r"\(([^()]+)\)$", m.group(2))  # archive member
                name = obj.group(1) if obj else Path(m.group(2)).name
                files[int(m.group(1))] = re.sub(r"\.(o|obj)$", "", name)
        elif section == "Symbols":
            m = re.match(r"^0x([0-9A-Fa-f]+)\s+0x[0-9A-Fa-f]+\s+\[\s*(\d+)\]\s+_(\S+)$", line)
            if m:
                syms[int(m.group(1), 16)] = (files.get(int(m.group(2)), "?"), m.group(3))
    return syms


def symbolize(link_map, addr_file):
    """(translation unit, function) of the recorded addresses (relative to main)."""
    syms = read_link_map(link_map)
    main_addr = next(a for a, (_, n) in syms.items() if n == "main")
    names = set()
    for x in addr_file.read_text().split():
        a = main_addr + int(x, 16)
        if a not in syms:
            sys.exit(f"trace_step: address {a:#x} is not a function start in {link_map}")
        names.add(syms[a])
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-dir", default=str(REPO / "build" / "step_trace"))
    ap.add_argument("--out", default=str(HERE / "har_train_step_trace.txt"))
    args = ap.parse_args()
    build = Path(args.build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC") or shutil.which("cc")
    hook = build / "step_trace_hook.o"
    run([cc, "-O2", "-c", str(HERE / "step_trace_hook.c"), "-o", str(hook)])
    if platform.system() != "Darwin":
        sys.exit("trace_step: only the macOS (ld64) link map is parsed so far")
    link_map = build / "link.map"
    # Weak references must be allowed to stay undefined (CMake's compiler check).
    link = [str(hook), "-Wl,-U,_trainingEpochDefault", "-Wl,-U,_trainingBatchDefault",
            f"-Wl,-map,{link_map}"]
    run(["cmake", "-S", str(REPO), "-B", str(build), "-G", "Ninja", f"-DCMAKE_C_COMPILER={cc}",
         "-DBUILD_EXAMPLES=ON", "-DCMAKE_C_FLAGS=-O2 -ffp-contract=off -finstrument-functions",
         f"-DCMAKE_EXE_LINKER_FLAGS={' '.join(link)}"])
    run(["cmake", "--build", str(build), "--target", TARGET])
    binary = build / "examples" / "har_classifier" / TARGET

    sys.path.insert(0, str(REPO / "examples" / "har_classifier"))
    from run_matrix import CONFIGS as MATRIX, write_bfp_data  # noqa: E402

    # bfp8full trains on pre-converted BFP data: write it with this build.
    write_bfp_data(CONFIGS, binary.parent)

    names = set()
    for cfg in CONFIGS:
        _, env = MATRIX[cfg]
        addr = build / f"{cfg}.addr"
        addr.unlink(missing_ok=True)
        run([str(binary)], cwd=REPO, env={**os.environ, **env, "EPOCHS": "1", "SEED": "1",
                                           "LOG_PATH": str(build / f"{cfg}.json"),
                                           "STEP_TRACE_OUT": str(addr)})
        if not addr.exists():
            sys.exit(f"trace_step: {cfg} wrote no trace (did it reach a fourth batch?)")
        got = symbolize(link_map, addr)
        print(f"{cfg}: {len(got)} functions")
        names |= got
    sys.path.insert(0, str(HERE))
    from census import source_fingerprint  # noqa: E402

    w = max(len(src) for src, _ in names)
    Path(args.out).write_text(
        "# Generated by trace_step.py: the functions a bfp8full HAR training step\n"
        "# executes (host, -finstrument-functions, both heads, three steps each), as\n"
        "# <translation unit> <function>. Do not edit; census.py --strict fails when\n"
        "# the sources no longer match the fingerprint.\n"
        f"# fingerprint {source_fingerprint(REPO)}\n"
        + "".join(f"{src:<{w}} {fn}\n" for src, fn in sorted(names)))
    print(f"wrote {len(names)} functions to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
