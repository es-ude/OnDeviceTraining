#ifndef INFERENCE_H
#define INFERENCE_H

#include "DataLoader.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Model.h"
#include "Tensor.h"
#include "TrainingCall.h"

typedef struct inferenceStats {
    tensor_t *output;
    float loss;
} inferenceStats_t;

void freeInferenceStats(inferenceStats_t *inferenceStats);

/*! Key-less: every layer gets a NULL context, so it serves forwards that draw
 *  nothing. A forward that draws (stochastic rounding, a BFP_SHIFT_SR softmax,
 *  a Dropout layer left in training mode) needs a key: run it through
 *  inferenceWithLoss with call->key (a keyed inference() is #491); here it
 *  stops with a named error at its first draw. A non-chain model (an ADD merge)
 *  runs on an eval-only scheduler built and freed inside the call. */
tensor_t *inference(const model_t *m, tensor_t *input);

/*! Wraps each sample with batchViewOf before calling inference() -- see
 *  docs/conventions/data-shape.md, "Who adds the batch axis". Key-less like
 *  inference(). A non-chain model runs every sample on one eval-only scheduler
 *  keyed on samples[0], so each sample must have its shape and dtype or the
 *  call exits at bind; a chain takes mixed shapes. */
tensor_t **inferenceBatched(const model_t *m, batch_t *batch);

/*! call is NULLable: a NULL call or a NULL call->remat runs the forward on
 *  per-call buffers (a non-chain model: on a per-call eval-only scheduler, as
 *  for inference()). A non-NULL call->remat runs it on that
 *  scheduler's EVAL program inside its memory (remat D19): the scheduler must
 *  be keyed to this model and to input's shape, or the call exits at bind
 *  naming the field. input is borrowed, never copied or written. Output
 *  values, shape, dtype, dynamic quantization state and the loss equal the
 *  NULL path's bit for bit; only an input's sparsity marker is dropped: a
 *  scheduler's output is unmarked, where a chain's NULL path marks it (a
 *  non-chain model's NULL path runs on a scheduler, so its output is unmarked
 *  too). A non-NULL call->key is this call's key: layer i gets it folded with
 *  (LAYER, i) and the loss with (LAYER, m->numLayers), on either path; without
 *  it every layer gets a NULL context. label is required: a NULL label exits
 *  (inference() is the forward without a loss). */
inferenceStats_t *inferenceWithLoss(const model_t *m, tensor_t *input, tensor_t *label,
                                    lossFuncType_t funcType, reduction_t forwardReduction,
                                    const trainingCall_t *call);

#endif // INFERENCE_H
