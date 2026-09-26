#ifndef ODT_REMAT_PLAN_POLICY_H
#define ODT_REMAT_PLAN_POLICY_H

#include <stddef.h>

#include "RematPlan.h"
#include "RematScheduler.h"

/* The TRAIN generators (spec §4.3-§4.4), private to the RematPlan library. */

size_t rematTrainStepCount(const rematWireTable_t *t);
/* BACKWARD(l) = n + 2 + (top - l), for deepest <= l <= top. */
size_t rematBackwardStep(const rematWireTable_t *t, size_t l);
void rematFillTrainSteps(const rematWireTable_t *t, rematStep_t *steps);
/* ranges[id - 1] for every slab wire id, in wire-id order (= begin order).
 * LIVENESS reads the model for layerBackwardReadsInput. */
void rematFillTrainRanges(rematPlanPolicy_t policy, const rematWireTable_t *t, layer_t **model,
                          size_t numSteps, rematRange_t *ranges);

#endif // ODT_REMAT_PLAN_POLICY_H
