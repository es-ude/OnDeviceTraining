#ifndef CALCULATE_GRADS_SEQUENTIAL_H
#define CALCULATE_GRADS_SEQUENTIAL_H

#include "TrainingLoopApi.h"

/*! Precondition: modelSize >= 1. An empty model exits naming it
 *  ("rematWireTableInit: modelSize == 0: nothing to schedule"). The wire
 *  table also caps the size: its wire ids are uint16_t, and a model with a
 *  full backward has 2 * modelSize + 1 wires, so about 32,000 layers is the
 *  limit; a larger model exits naming it ("wire ids are uint16_t"). */
trainingStats_t *calculateGradsSequential(layer_t **model, size_t modelSize,
                                          lossConfig_t lossConfig, reduction_t forwardReduction,
                                          tensor_t *input, tensor_t *label);

#endif // CALCULATE_GRADS_SEQUENTIAL_H
