#ifndef ODT_BATCH_VIEW_H
#define ODT_BATCH_VIEW_H

#include <stddef.h>

#include "Tensor.h"

#define BATCH_VIEW_MAX_RANK 8 /* same cap as the deserializer's SKIP_TENSOR_MAX_DIMS */

/*! Caller-owned storage for a [1, ...sample] view of ONE natural-shape
 *  dataset sample (docs/conventions/data-shape.md: the loop owns the batch
 *  axis). Lives on the caller's stack: reentrant, no heap, nothing to free.
 *
 *  A filled view is self-referential (tensor.shape -> shape, shape.dimensions
 *  -> dimensions[], ...): never copy it by value -- use &view->tensor. */
typedef struct batchView {
    tensor_t tensor;
    shape_t shape;
    size_t dimensions[BATCH_VIEW_MAX_RANK];
    size_t orderOfDimensions[BATCH_VIEW_MAX_RANK];
} batchView_t;

/*! Fills the caller-owned view with shape [1, ...sample->shape] and returns
 *  &view->tensor. data, quantization and sparsity are shared with `sample`
 *  (no copy, no heap). order[0] = 0, order[i+1] = sample order[i] + 1.
 *  The view is valid while `sample` is alive and the view has not been
 *  refilled. Fails fast if sample rank + 1 > BATCH_VIEW_MAX_RANK.
 *
 *  Use when: handing one dataset sample to a tensor-level entry point
 *  (inference, inferenceWithLoss, calculateGradsSequential, tracedGrads),
 *  which all take batched [B, ...] tensors. The batch_t consumers
 *  (trainingBatchDefault, trainingEpochDefault, evaluation*, inferenceBatched)
 *  already do this themselves. */
tensor_t *batchViewOf(batchView_t *view, tensor_t *sample);

#endif // ODT_BATCH_VIEW_H
