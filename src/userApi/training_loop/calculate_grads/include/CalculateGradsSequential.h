#ifndef CALCULATE_GRADS_SEQUENTIAL_H
#define CALCULATE_GRADS_SEQUENTIAL_H

#include "TrainingLoopApi.h"

/* Defined in RematScheduler.h; a caller that reads the default includes it.
 * C11 allows the repeated typedef, so this header needs no scheduler header. */
typedef struct rematPlanSpec rematPlanSpec_t;

/*! Precondition: modelSize >= 1. An empty model exits naming it
 *  ("rematWireTableInit: modelSize == 0: nothing to schedule"). The wire
 *  table also caps the size: its wire ids are uint16_t, and a model with a
 *  full backward has 2 * modelSize + 1 wires, so about 32,000 layers is the
 *  limit; a larger model exits naming it ("wire ids are uint16_t").
 *
 *  call is NULLable. NULL, or a NULL call->remat, builds and tears down an
 *  ephemeral HEAP scheduler on calculateGradsDefaultPlanSpec() inside the
 *  call (remat D15, D30). In builds with the memory counter (ODT_MEM_PROFILE)
 *  that init, like any rematHeapInit, checks that it reserved exactly what its
 *  report claims, so the rule of RematScheduler.h applies to the call: no
 *  other thread may reserve or free memory while it sets the scheduler up.
 *
 *  A non-NULL call->remat is the caller's: initialised for this model and
 *  for this input's shape (rematHeapInit / rematArenaInit),
 *  used by the call and left initialised for the next one; the caller deinits
 *  it. A zeroed or deinitialised scheduler, or one whose key does not match,
 *  exits naming it. */
trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label,
                                          const trainingCall_t *call);

/*! The plan a training call without a scheduler runs on: the ephemeral HEAP
 *  scheduler of a NULL call (or a NULL call->remat) is built from it. A rule
 *  of the training call, not of the scheduler library, whose own NULL spec
 *  keeps meaning STORE_ALL. Never NULL; its fields are read through
 *  RematScheduler.h. */
const rematPlanSpec_t *calculateGradsDefaultPlanSpec(void);

#endif // CALCULATE_GRADS_SEQUENTIAL_H
