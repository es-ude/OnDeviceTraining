#ifndef ODT_REMAT_PLAN_POLICY_H
#define ODT_REMAT_PLAN_POLICY_H

#include <stddef.h>

#include "RematPlan.h"
#include "RematScheduler.h"

/* The program generators, private to the RematPlan library. */

size_t rematTrainStepCount(const rematWireTable_t *t);
/* BACKWARD(l) = n + 2 + (top - l), for deepest <= l <= top. */
size_t rematBackwardStep(const rematWireTable_t *t, size_t l);
void rematFillTrainSteps(const rematWireTable_t *t, rematStep_t *steps);
/* ranges[id - 1] for every slab wire id, in wire-id order (= begin order).
 * LIVENESS reads the model for layerBackwardReadsInput. */
void rematFillTrainRanges(rematPlanPolicy_t policy, const rematWireTable_t *t, layer_t **model,
                          size_t numSteps, rematRange_t *ranges);
/* The scan generator (SEQUENCE; RETAIN_LIST on its own step list): one range
 * per write, so the count is the number of writes; each range opens at its
 * write and ends at the last read of its wire before the wire's next write
 * (at the write itself if nothing reads it). Ranges come out begin-sorted.
 * Safe on any step list: a step the grammar will reject (an unknown kind, a
 * layer outside its kind's range) opens and extends nothing. numSteps must be
 * at most UINT16_MAX (step indexes are stored as uint16_t). */
size_t rematScanRangeCount(const rematWireTable_t *t, const rematStep_t *steps, size_t numSteps);
void rematScanRanges(const rematWireTable_t *t, layer_t **model, const rematStep_t *steps,
                     size_t numSteps, rematRange_t *ranges);
/* The RETAIN_LIST generator: FORWARD 0..n-1 and LOSS_FORWARD; then, before
 * LOSS_BACKWARD and before every BACKWARD(l) that reads its input, the
 * REFORWARD run k..j-1 that recomputes a dropped ACT j (j = n, or l) from the
 * nearest present ACT k < j: ACT 0, a kept one, or one an earlier run
 * recomputed. The needs only descend, so each ACT is recomputed at most once
 * and a program holds at most 3n + 2 steps. With steps == NULL it only counts
 * (the plan sizes its block first). Returns the step count; *numWrites (if
 * not NULL) gets the number of steps that write a wire, which is
 * rematScanRangeCount of the same list. */
size_t rematFillRetainSteps(const rematWireTable_t *t, layer_t **model, const bool *retain,
                            rematStep_t *steps, size_t *numWrites);
/* The EVAL generator: n + 1 steps and n ranges (ACT 1..n, in wire-id order),
 * whatever the policy. */
void rematFillEvalProgram(const rematWireTable_t *t, rematStep_t *steps, rematRange_t *ranges);

#endif // ODT_REMAT_PLAN_POLICY_H
