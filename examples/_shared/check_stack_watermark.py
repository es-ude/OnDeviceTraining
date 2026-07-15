"""Stack-watermark report/gate for instrumented examples (hardening item 6).

Reads RunLog JSONs produced by ODT_MEM_PROFILE builds (mem_instrument) and
compares memory.stack_peak_b against a per-platform budget. Stage 1 is
REPORT-ONLY: over-budget prints WARN and exits 0; pass --enforce to gate.

Budget maintenance rule (F_STACK_SLACK precedent): budgets are measured
peak + 8 KiB slack, with provenance recorded below. NEVER raise a budget in
the same PR that raised the consumption — a raise is its own reviewed PR
with a fresh measurement as evidence. If this check fires, either fix the
regression or justify the new budget in a dedicated PR.
"""
from __future__ import annotations

import argparse
import json
import sys

SLACK_B = 8192

# Measured peaks are bit-exact across seeds AND sym widths (scratch is held
# unpacked, width-independent). Provenance: #296 Stage 2 (bounded-scratch
# PR B), 2026-07-10, devenv build, macOS arm64, EPOCHS=1 SEED=1 — float
# 27768 B (conv3 weight-grad OUT_ACC rawData 24576 B + frames incl. the
# fixed conversion chunk buffers, +440 B vs Stage 1), sym 51968 B
# (converted-operand row 24576 + rawData 24576 + frames; the 49152 B
# repack chain is gone — conversions stream in O(chunk)). History:
# pre-#296 float 99560 / sym 149008 (2026-07-07, 60-run sweep); Stage 1
# float 27328 / sym 100080 (2026-07-09).
# linux: deliberately uncalibrated until the CI job has produced peaks;
# calibrate from the c-stack-watermark job logs, then set + --enforce in a
# dedicated PR.
#
# "float" covers BOTH float HAR trainers (train_c_har_classifier and
# train_c_har_classifier_adamw, #328): same model + microbatch, optimizer
# state is heap-side. The darwin number was calibrated on the SGD binary;
# re-key per-binary if calibration ever shows the two diverging.
BUDGETS_B: dict[str, dict[str, int | None]] = {
    "darwin": {"float": 27768 + SLACK_B, "sym": 51968 + SLACK_B},
    "linux": {"float": None, "sym": None},
}


def _config_of(memory: dict) -> str:
    return "float" if memory.get("sym_bits", -1) < 0 else "sym"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="RunLog JSON path(s)")
    parser.add_argument("--enforce", action="store_true",
                        help="exit 1 on over-budget (default: warn only)")
    parser.add_argument("--platform", choices=sorted(BUDGETS_B),
                        default=("darwin" if sys.platform == "darwin" else "linux"))
    args = parser.parse_args(argv)

    rc = 0
    for path in args.logs:
        with open(path) as f:
            memory = json.load(f).get("memory", {})
        peak = memory.get("stack_peak_b", 0)
        if not peak:
            print(f"ERROR {path}: no stack_peak_b — not an ODT_MEM_PROFILE build?")
            return 2
        config = _config_of(memory)
        budget = BUDGETS_B[args.platform][config]
        if budget is None:
            print(f"UNCALIBRATED {path}: {config} peak {peak} B on {args.platform} "
                  f"(no budget yet — record this value to calibrate)")
        elif peak <= budget:
            print(f"OK   {path}: {config} peak {peak} B <= budget {budget} B")
        else:
            print(f"WARN {path}: {config} peak {peak} B EXCEEDS budget {budget} B "
                  f"(+{peak - budget} B) — fix the regression or raise the budget "
                  f"in a dedicated PR (see module docstring)")
            if args.enforce:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
