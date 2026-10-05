#ifndef ODT_TEST_LEGACY_CALCULATE_GRADS_H
#define ODT_TEST_LEGACY_CALCULATE_GRADS_H

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"
#include "TraceApi.h"
#include "TrainingLoopApi.h"

/*! The pre-remat calculateGradsImpl: the bit-identity oracle (remat D42).
 *  Same contract as tracedGrads with a NULL call (sink may be NULL; the
 *  oracle keeps its pre-#4 signature until #4 PR5b retires it), the same
 *  four OdtHook events, its own per-call allocators. Test builds only. */
trainingStats_t *legacyCalculateGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                                      reduction_t forwardReduction, tensor_t *input,
                                      tensor_t *label, traceSink_t sink, void *sinkCtx);

#endif // ODT_TEST_LEGACY_CALCULATE_GRADS_H
