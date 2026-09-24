#ifndef TRAINING_BATCH_DEFAULT_H
#define TRAINING_BATCH_DEFAULT_H

#include <stddef.h>

#include "TrainingLoopApi.h"

/*! Trains one macro batch: walks batch->samples in order in batch->size / m
 *  chunks of exactly m rows (m = microBatchSize; 0 means 1) and calls
 *  calculateGradsFn once per chunk -- grads accumulate across chunks, the
 *  optimizer step is the caller's (trainingEpochDefault).
 *  - m == 1: every sample is wrapped by batchViewOf, no copy, no heap.
 *  - m > 1: each chunk's items and labels are gathered into two [m, ...]
 *    buffers, reserved once per call and freed before returning. FLOAT32
 *    only: every layer must pass layerIsFloat32Only (checked once per call,
 *    before anything runs) and every sample's item and label must be
 *    FLOAT32, sparsity-free and match sample 0 in rank, dimensions and
 *    order -- fail fast otherwise.
 *  Fails fast unless batch->size % m == 0 (a replay loader must keep its
 *  appended sample count divisible by m), and at m > 1 on an empty batch.
 *  Frees every sample_t it consumes.
 *  Returns the per-sample mean loss for REDUCTION_MEAN (chunk losses weighted
 *  by their m rows, divided by batch->size) and the plain sum for SUM.
 *  See docs/conventions/data-shape.md, "Who adds the batch axis". */
float trainingBatchDefault(layer_t **model, size_t modelSize, lossConfig_t lossConfig,
                           batch_t *batch, calculateGradsFn_t calculateGradsFn,
                           reduction_t forwardReduction, size_t microBatchSize);

#endif // TRAINING_BATCH_DEFAULT_H
