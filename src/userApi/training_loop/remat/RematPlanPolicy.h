#ifndef ODT_REMAT_PLAN_POLICY_H
#define ODT_REMAT_PLAN_POLICY_H

#include <stddef.h>

#include "RematPlan.h"
#include "RematScheduler.h"

/* The program generators, private to the RematPlan library. */

/* The one owner of the gradient-wire decision, made once at table init (and repeated at every
 * bind from the same key, so it gives the same values). Numbers the seed (only when a backward
 * runs), then, walking the runsBackward nodes in descending order and each node's inputs in input
 * order, a GRAD for every input buffer that needs a gradient, at that buffer's first
 * contribution; ids start at n + 1. Fills dxWire/dxMode for every node's MODEL_MAX_ARITY slots.
 * With facts == NULL it only counts, so the table can be sized first; dxWire/dxMode may then be
 * NULL too. Returns the number of GRAD wires. */
size_t rematDeriveGradWires(const model_t *m, lossFuncType_t lt, const uint8_t *needsGrad,
                            const uint8_t *runsBackward, rematWireFact_t *facts, uint16_t *dxWire,
                            uint8_t *dxMode);

/* What a program needs before its block exists. */
typedef struct rematDeriveSize {
    size_t numSteps;
    size_t numOperands;
    size_t numRanges;
} rematDeriveSize_t;

/* The TRAIN program's sizes for this spec. The operand pool does not depend on the policy:
 * every node's FORWARD slice, the LOSS_FORWARD slice, then, when a backward runs, the
 * LOSS_BACKWARD slice and the BACKWARD slice of every runsBackward node, descending. numRanges
 * counts the writes rematScanRanges opens a range for, with spec->inPlaceDx deciding the
 * members. Reads the caller's sequence or retain mask, so the spec's own checks must have run
 * first. */
rematDeriveSize_t rematDeriveTrainSize(const rematWireTable_t *t, const rematPlanSpec_t *spec);
/* The EVAL program's sizes: n + 1 steps, the pool's FORWARD and LOSS_FORWARD slices, and the n
 * ranges of ACT 1..n. */
rematDeriveSize_t rematDeriveEvalSize(const rematWireTable_t *t);
/* The one operand writer. Fills the steps (kind and layer from the policy generator, or the
 * caller's sequence), the pool and nodeOps, and points each step at its node's slice by kind;
 * a REFORWARD shares its FORWARD's slice. The caller places p's arrays and counts from
 * rematDeriveTrainSize. A step the grammar will reject (an unknown kind, a layer outside its
 * kind's range, a node whose slice is REMAT_NONE) gets the empty slice, so the deriver is safe
 * on any caller's step list. Reads the table only, never a layer. */
void rematDeriveTrain(const rematWireTable_t *t, const rematPlanSpec_t *spec, rematProgram_t *p);
/* The EVAL program: the TRAIN program's first n + 1 steps and their slices, operands included.
 * The caller places p's arrays and counts from rematDeriveEvalSize. */
void rematDeriveEval(const rematWireTable_t *t, rematProgram_t *p);

/* One range per opening write, in step order, read from the stored operands only; the same rule
 * for every step kind. A READ operand extends the newest range of its chain root (META extends
 * nothing; ACT 0 has no range); a write operand's wire, unless REMAT_NONE, opens {wire, s, s} if it
 * is its own root and its mode is WRITE, and otherwise extends its root's newest range. STORE_ALL
 * then ends every ACT range at numSteps - 1. A read before any write of its wire extends nothing
 * (the grammar rejects it). Ranges come out begin-sorted, ties by wire id: a chain step opens at
 * most one range. The marks must be filled first (NULL: no members); ranges holds the sizing
 * pass's numRanges. O(S * R), no scratch. */
void rematScanRanges(const rematWireTable_t *t, const rematProgram_t *p, rematPlanPolicy_t policy,
                     const uint8_t *inPlace, rematRange_t *ranges);

/* FORWARD 0..n-1, LOSS_FORWARD; then, when a backward runs, LOSS_BACKWARD and a BACKWARD for
 * every runsBackward node, descending. */
void rematFillTrainSteps(const rematWireTable_t *t, rematStep_t *steps);
/* Is wire w an in-place member: the dx GRAD l of a ReLU, Dropout or Flatten whose wire and wire
 * w - 1, the gradient its BACKWARD reads, have one dtype, FLOAT32 or SYM_INT32, and equal bytes?
 * The one membership predicate: the marks and the sizing pass's range count both use it. */
bool rematInPlaceMember(const rematWireTable_t *t, size_t w);
/* inPlace[w] = 1 for every in-place member w, 0 for every other wire; returns
 * the number of members. With inPlace == NULL it only counts. */
size_t rematFillInPlaceMarks(const rematWireTable_t *t, uint8_t *inPlace);
/* The wire whose range holds this one's bytes: an in-place member walks down to its chain's root;
 * every other wire, and every wire under NULL marks, is its own. */
uint16_t rematChainRoot(const uint8_t *inPlace, uint16_t wire);
/* The RETAIN_LIST generator: FORWARD 0..n-1 and LOSS_FORWARD; then, before
 * LOSS_BACKWARD and before every BACKWARD(l) that reads its input, the
 * REFORWARD run k..j-1 that recomputes a dropped ACT j (j = n, or l) from the
 * nearest present ACT k < j: ACT 0, a kept one, or one an earlier run
 * recomputed. The needs only descend, so each ACT is recomputed at most once
 * and a program holds at most 3n + 2 steps. So the program is the default
 * TRAIN program plus REFORWARD steps. With steps == NULL it only counts (the
 * plan sizes its block first). Returns the step count. */
size_t rematFillRetainSteps(const rematWireTable_t *t, const bool *retain, rematStep_t *steps);

#endif // ODT_REMAT_PLAN_POLICY_H
