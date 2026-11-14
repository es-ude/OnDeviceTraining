#ifndef TRACE_API_H
#define TRACE_API_H

#include <stddef.h>

#include "Layer.h"
#include "LossFunction.h"
#include "Model.h"
#include "Tensor.h"
#include "TrainingLoopApi.h"

/*! Fired at every probe point of one traced training step. The framework hands
 *  a tensor to the sink and never opens a file; the sink (above the src/
 *  boundary) decides what to do with it.
 *
 *  - layerIdx:   model index of the layer; for the loss gradient, == m->numLayers.
 *  - layerType:  the layer's type (for naming / dtype decisions).
 *  - phase:      "fwd" | "refwd" | "agrad" | "lossgrad" for tracedGrads;
 *                "<tag>.weight" / "<tag>.bias" for traceModelWeights/Grads.
 *  - tensor:     borrowed; valid only for the duration of the call. */
typedef void (*traceSink_t)(void *ctx, size_t layerIdx, layerType_t layerType, const char *phase,
                            tensor_t *tensor);

/*! Same forward+backward as calculateGradsSequential, but fires `sink` after
 *  each layer's forward ("fwd"), after each recompute of a layer's forward
 *  that the scheduler's plan asks for during the backward phase ("refwd",
 *  with the recomputed output; a plan may recompute a layer more than once),
 *  after the loss backward ("lossgrad", layerIdx == m->numLayers), and before
 *  each layer's backward ("agrad", with the gradient entering it: the
 *  gradient w.r.t. that layer's output).
 *
 *  A layer's backward, and its "agrad", runs exactly where modelGradFlags
 *  sets runsBackward: the layer has trainable params, or an input that
 *  needs a gradient. A frozen layer whose inputs need no gradient fires no
 *  "agrad", even when a trainable layer has a lower node index (a DAG
 *  branch it does not read). On a chain
 *  this is the range from the deepest trainable layer to the top. If no
 *  layer in the model trains, backward is skipped entirely -- NO "lossgrad"
 *  and NO "agrad" events fire (the loss value is still computed and "fwd"
 *  still fires for every layer). Preconditions: as for
 *  calculateGradsSequential (m->numLayers >= 1, and the uint16_t wire cap);
 *  call as for calculateGradsSequential. */
trainingStats_t *tracedGrads(const model_t *m, lossConfig_t lossConfig,
                             reduction_t forwardReduction, tensor_t *input, tensor_t *label,
                             traceSink_t sink, void *ctx, const trainingCall_t *call);

/*! Trace parameter values, including frozen parameters, with phase
 *  "<tag>.weight" / "<tag>.bias"; omit absent biases and non-parameter
 *  buffers (BatchNorm1d running statistics). Gradient tracing also omits
 *  frozen parameters. */
void traceModelWeights(const model_t *m, const char *tag, traceSink_t sink, void *ctx);

/*! Same, for the GRAD tensor of each parameter_t. */
void traceModelGrads(const model_t *m, const char *tag, traceSink_t sink, void *ctx);

#endif /* TRACE_API_H */
