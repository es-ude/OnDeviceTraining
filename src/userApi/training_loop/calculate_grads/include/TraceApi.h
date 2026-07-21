#ifndef TRACE_API_H
#define TRACE_API_H

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Tensor.h"
#include "TrainingLoopApi.h"

/*! Fired at every probe point of one traced training step. The framework hands
 *  a tensor to the sink and never opens a file; the sink (above the src/
 *  boundary) decides what to do with it.
 *
 *  - layerIdx:   model index of the layer; for the loss gradient, == modelSize.
 *  - layerType:  the layer's type (for naming / dtype decisions).
 *  - phase:      "fwd" | "agrad" | "lossgrad" for tracedGrads (Task 2);
 *                "<tag>.weight" / "<tag>.bias" for traceModelWeights/Grads (Task 3).
 *  - tensor:     borrowed; valid only for the duration of the call. */
typedef void (*traceSink_t)(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                            tensor_t *tensor);

/*! Same forward+backward as calculateGradsSequential, but fires `sink` after
 *  each layer's forward ("fwd"), after the loss backward ("lossgrad",
 *  layerIdx == modelSize), and after each layer's backward ("agrad").
 *
 *  Backward truncates at the deepest (closest-to-input) trainable layer
 *  (#380 PR2): "lossgrad"/"agrad" fire only for that layer and everything
 *  above it, never for a layer below it (no dx is consumed there). If no
 *  layer in the model trains, backward is skipped entirely -- NO "lossgrad"
 *  and NO "agrad" events fire (the loss value is still computed and "fwd"
 *  still fires for every layer). */
trainingStats_t *tracedGrads(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx);

/*! Fire `sink` for each trainable layer's weight and bias PARAM tensors, with
 *  phase "<tag>.weight" / "<tag>.bias". Param-less layers and NULL bias are
 *  skipped; a frozen layer's weight/bias GRAD is also skipped (#380 --
 *  frozen layers carry grad == NULL, and the sink contract promises a
 *  borrowed VALID tensor, never NULL). (Trainable: LINEAR, CONV1D,
 *  CONV1D_TRANSPOSED, LAYERNORM, GROUPNORM.) */
void traceModelWeights(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                       void *ctx);

/*! Same, for the GRAD tensor of each parameter_t. */
void traceModelGrads(layer_t **model, size_t modelSize, const char *tag, traceSink_t sink,
                     void *ctx);

#endif /* TRACE_API_H */
