#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "Tensor.h"

typedef enum lossFuncType { MSE, CROSS_ENTROPY } lossFuncType_t;

typedef enum reduction { REDUCTION_SUM, REDUCTION_MEAN } reduction_t;

typedef struct lossConfig {
    lossFuncType_t funcType;
    reduction_t backwardReduction;
    tensor_t *classWeights;
} lossConfig_t;

/* Convenience initializer. Returns {funcType, REDUCTION_MEAN, REDUCTION_MEAN, NULL}.
 * forwardReduction is intentionally NOT a config field — it is a per-call parameter
 * on aggregators. trainingRun hardcodes REDUCTION_MEAN to keep train/eval comparability. */
lossConfig_t defaultLossConfig(lossFuncType_t funcType);

/*! Per-microbatch forward.
 *
 * Reduction-aware; PyTorch-parity for both MEAN and SUM.
 *
 * \param modelOutput  Tensor of shape [B, ...], rank >= 2 with at least one
 *                     element (B = microbatch dim). Always explicit: the
 *                     training loop passes [1, ...] views (microBatchSize 1)
 *                     or [m, ...] stacks (#152); see docs/conventions/loss.md,
 *                     "Microbatch shape".
 * \param label        Same rank and dimensions as modelOutput (enforced, #153).
 * \param reduction    REDUCTION_MEAN ⇒ per-microbatch mean over own elements;
 *                     REDUCTION_SUM  ⇒ per-microbatch raw sum.
 * \return Per-microbatch scalar loss value.
 *
 * Contract, enforced by every dispatcher (#153): modelOutput has rank >= 2
 * and at least one element, and the label has modelOutput's rank and dimensions
 * (orderOfDimensions is not compared). All microbatches in one macro batch
 * must have equal B (uniform microbatch size assumption — see
 * docs/CONVENTIONS.md §"Loss API: microbatch contracts"). */
typedef float (*lossFwdFn_t)(tensor_t *modelOutput, tensor_t *label, reduction_t reduction);

/*! Per-microbatch backward.
 *
 * Writes raw per-element gradient (no batchSize/reduction divisor).
 * Macro-batch scaling is applied at the optimizer step via
 * scaleOptimizerGradients(optimizer, computeMeanScale(...)) in
 * trainingEpochDefault when backwardReduction == REDUCTION_MEAN.
 *
 * \param modelOutput  Same shape contract as forward.
 * \param label        Same rank and dimensions as modelOutput (enforced, #153).
 * \param result       Output buffer (same shape) for the raw per-element grad. */
typedef void (*lossBwdFn_t)(tensor_t *modelOutput, tensor_t *label, tensor_t *result);

/* Per-loss MEAN-reduction scale factor (PyTorch parity).
 * Only called when backwardReduction == REDUCTION_MEAN.
 * Each loss family derives numFeaturesPerSample from the model output
 * shape itself (B = dimensions[0], F = numElements / B), so the caller
 * does not need to know about microbatch-vs-feature dimensions:
 *   MSE: 1 / (totalSamples × F)
 *   CE:  1 / totalSamples (modelOutput unused)
 *
 * Contract on modelOutput: a `[B, ...]` tensor (rank >= 2, at least one
 * element) whose dims[0] is the batch axis; computeMeanScaleMSE fails fast
 * otherwise (#153), computeMeanScaleCE never reads it. The only caller
 * (trainingEpochDefault) does not pass the model output itself -- it passes
 * the batchViewOf view of sample 0's label (#152 PR3a), so for MSE this is
 * dims[0] of the LABEL's [1, ...] view, not the model's output tensor. */
typedef float (*computeMeanScaleFn_t)(size_t totalSamples, tensor_t *modelOutput);

typedef struct lossFunctions {
    lossFwdFn_t forward;
    lossBwdFn_t backward;
    computeMeanScaleFn_t computeMeanScale;
} lossFunctions_t;

extern lossFunctions_t lossFunctions[];

#endif // LOSSFUNCTION_H
