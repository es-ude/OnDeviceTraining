# `odt.planner` — the remat budget planner

For a model, a micro-batch `m` and an arena budget in bytes, the planner returns the
retain plan that recomputes least and whose ARENA arena, as C will lay it out, fits the
budget (remat D12 in `docs/conventions/remat-scheduler.md`). It runs on the host only;
C stays the truth (remat D18): the planner answers with the two variables the example
trainers already read, `REMAT_PLAN=retain:<j>,...,<n>[+inplace]` and `REMAT_BUDGET_B`,
and the trainer checks the arena against the budget itself.

## Use

```bash
B=./build/examples/examples/har_classifier/train_c_har_classifier
DUMP_PROBLEM=1 $B > har.problem.json            # the trainer's model; the hash goes to stderr
uv run -m odt.planner solve har.problem.json --m 1 --budget 32768
# # problem fnv1a64=26b4a7babc5f10bf
# REMAT_PLAN=retain:4,6,7,10,12+inplace REMAT_BUDGET_B=32768
uv run -m odt.planner frontier har.problem.json --m 1
# # problem fnv1a64=26b4a7babc5f10bf
# budget_lo_b=41216 plan=retain:1,3,4,6,7,10,12+inplace arena_b=41216 wires_peak_b=41216 placed_peak_b=41216 recompute_cost=0 refwd_steps=0
# ... one line per budget at which the best plan changes, largest first
```

- `solve` prints the problem's hash as a comment and the line to run the plan with.
  `--no-inplace` keeps to plans without in-place dx; otherwise the planner chooses.
  `--m` has no default: plans hold for one micro-batch.
- `frontier` prints every breakpoint: the plan of a line answers every budget from its
  `budget_lo_b` (its arena) up to the previous line's minus one; below the last line no
  plan fits.
- If nothing fits, `solve` exits 1 with `odt.planner: nothing fits REMAT_BUDGET_B=<B>: no
  retain plan's arena is at most <B> B (the smallest placed peak of any plan is <F> B;
  the frontier's last point is the smallest arena)`. When the budget is at least that
  smallest placed peak but C's placement of every plan whose peak fits needs more, the
  parenthesis reads instead `(the smallest placed peak of any plan, <F> B, is within the
  budget, but a placement gap lifts the arena of every such plan above it; the
  frontier's last point is the smallest arena)`. A problem it cannot read or does not
  accept exits 1 with `odt.planner: <the reason>`; a budget outside 1 … 2^64 − 1 exits 2.
- In an environment that does not install the project, prefix the commands with
  `PYTHONPATH=python` (from the repository root).

`examples/_shared/remat_frontier.py` writes an example's frontier as a CSV in which C
measured every row (the planner's numbers sit next to C's and must match);
`examples/_shared/remat_conformance.py` trains every frontier plan against STORE_ALL.

## The problem file (schema v0)

One JSON object, as `DUMP_PROBLEM=1` writes it (one line, `\n` at the end). The dump
writes the layer types and dtypes the example trainers use; everything else is refused
by name. The schema itself is wider (below), so a later dump can grow into it.

```text
{"schema_version": 0,
 "input": {"shape": [1, ...], "dtype": {"kind": ...}},
 "loss": "CROSS_ENTROPY" | "MSE",
 "layers": [{"type", "frozen", "trackRunningStats", "outShape", "outputDtype",
             "propLossDtype", "cost"?}, ...]}
```

- Shapes carry the leading batch dimension, which is 1; `m` is a planner input, so a wire
  holds `m × prod(shape[1:])` elements. A Flatten's `outShape` has rank 2.
- `layers` is positional, one entry per C layer, Softmax included under CROSS_ENTROPY;
  `type` is C's `layerType_t` name.
- A dtype is an object: `{"kind"}` for `FLOAT32`, `INT32`, `SYM_INT32`, `BOOL`; `SYM` and
  `ASYM` add `qBits`; `BFP` adds `mantissaBits`, `exponentBits`, `groupSize`. Any other
  key is refused.
- `cost` holds the operand C's multiply-accumulate count (`layerForwardCost`) needs
  beyond element counts: `{"macPerElement"}` for CONV1D and LINEAR per output element
  (Cin/groups · K, inF) and for CONV1D_TRANSPOSED per input element (Cout/groups · K);
  `{"kernel"}` for MAXPOOL1D and AVGPOOL1D; no `cost` on any other type.
- The hash is FNV-1a 64 over the file's bytes, the final newline included (C prints it
  as `PROBLEM_HASH fnv1a64=<16 hex digits>`). It only makes a plan made for another model
  visible; nothing checks it.

`load_problem` refuses a malformed file with `ProblemError`, naming the field.

## What the planner accepts

The schema is wider than the planner on purpose. The planner accepts only what a C
reference run covers — the three example trainers: every dtype FLOAT32, `m` = 1, loss
CROSS_ENTROPY, layer types CONV1D, RELU, MAXPOOL1D, AVGPOOL1D, FLATTEN, LINEAR, SOFTMAX,
frozen or not, at most 64 layers (the harness's `REMAT_SELECT_MAX_LAYERS`), an input of
rank at most 8 with the batch axis counted (C can neither dump nor train a higher one;
`accept.py`), an ACT wire of at most `SIZE_MAX` elements and at most `SIZE_MAX` bytes and
wires that total at most `SIZE_MAX` bytes, as C's wire table does, and a layer forward cost
of at most `SIZE_MAX`, as C's `layerForwardCost` does, with the layers' forward costs
totalling at most `SIZE_MAX`, as C's plan build does (these four limits are in
`cmodel/table.py`). Everything else raises `UnsupportedProblem` naming the field, before
any plan is built. Widening it is one rule in `accept.py`, the matching rule in the
copy of C (`cmodel/`), a C reference case and its goldens — never a schema change.

## How it is exact, in short

Every plan keeps the last output, ACT n; a plan is a mask over ACT 1 … n − 1 and an
in-place setting. Recompute cost and the count of recomputed forwards add up over the
runs of dropped outputs between kept ones, and a plan's placed peak is the largest of
those runs' own peaks plus the kept outputs living through them. A dynamic program over
that structure bounds every completion; a best-first search pops whole plans in order of
cost, places each with the copy of C's placer, and stops at the first cost tier that holds
a plan whose arena fits. An arena is never below its placed peak, so the search skips no
plan that fits. The full argument is in `solve.py`.

The solver sits behind a `Strategy` interface, but that interface takes the retain-list
plan space (`PlanSpace`) and returns a retain-only `Plan`. A second plan family, such as a
multi-level recompute strategy, needs a plan-space object for its plans and a payload
field on the plan: a medium change, not one more class.

## The copy of C and its goldens

`cmodel/` is the planner's only copy of C rules: the wire table, the retain steps and
buffer ranges (with and without in-place dx), the costs, the peaks and ARENA's five
placement orders. `python/tests/test_planner_cmodel.py` checks it against committed C
output in `python/tests/planner_goldens/`: the three trainers' problem files and their
`DRY_PLAN` lines for a fixed sample of plans and for every frontier plan. Those files are
written only by `python/tests/planner_goldens/regenerate.py` (it runs the built trainers;
see its docstring); after a C change to plans or placement, rerun it and review the diff.
