#ifndef CALCULATE_GRADS_SEQUENTIAL_H
#define CALCULATE_GRADS_SEQUENTIAL_H

#include "TrainingLoopApi.h"

/*! Precondition: modelSize >= 1. An empty model exits naming it
 *  ("rematWireTableInit: modelSize == 0: nothing to schedule"). The wire
 *  table also caps the size: its wire ids are uint16_t, and a model with a
 *  full backward has 2 * modelSize + 1 wires, so about 32,000 layers is the
 *  limit; a larger model exits naming it ("wire ids are uint16_t").
 *
 *  call is NULLable. NULL, or a NULL call->remat, builds and tears down an
 *  ephemeral HEAP + STORE_ALL scheduler inside the call (remat D30). A
 *  non-NULL call->remat is the caller's: initialised for this model and for
 *  this input's shape (rematHeapInit / rematArenaInit), used by the call and
 *  left initialised for the next one; the caller deinits it. A zeroed or
 *  deinitialised scheduler, or one whose key does not match, exits naming it. */
trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label,
                                          const trainingCall_t *call);

#endif // CALCULATE_GRADS_SEQUENTIAL_H
