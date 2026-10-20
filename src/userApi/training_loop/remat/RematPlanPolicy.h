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
/* The EVAL generator: n + 1 steps and n ranges (ACT 1..n, in wire-id order),
 * whatever the policy. */
void rematFillEvalProgram(const rematWireTable_t *t, rematStep_t *steps, rematRange_t *ranges);

#endif // ODT_REMAT_PLAN_POLICY_H
