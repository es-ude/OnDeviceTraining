#ifndef ODT_BATCHNORM1D_H
#define ODT_BATCHNORM1D_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

/* DEFAULT (zero-init) = 0.1; VALUE = `momentum` literally (0 = running stats
 * never move); CUMULATIVE = PyTorch momentum=None, factor 1/num_batches_tracked.
 * Configs only ever hold VALUE or CUMULATIVE (init resolves DEFAULT). The values
 * are the serialized bytes: never renumber. */
typedef enum {
    BN_MOMENTUM_DEFAULT = 0,
    BN_MOMENTUM_VALUE = 1,
    BN_MOMENTUM_CUMULATIVE = 2,
} bnMomentumMode_t;

/* BatchNorm1d over [m, C] or [m, C, T] (identity order, FLOAT32 only), per-
 * channel statistics over the m (and T) values: Ghost-BN over the micro-
 * batch m. PyTorch nn.BatchNorm1d semantics: normalization uses
 * the BIASED batch variance, the running update the UNBIASED one.
 *   batch statistics iff !trackRunningStats || (training && !frozen)
 *   running update   iff trackRunningStats && training && !frozen
 * A batch-statistics forward/backward needs n = m*T >= 2 (fails fast).
 * Evaluation stacks consecutive samples into chunks of up to m rows (m = 1
 * by default), so an untracked (!trackRunningStats) BatchNorm1d normalizes over
 * its chunk: its eval output depends on the chunk mates (as in PyTorch) and needs
 * n = rows * T >= 2 for EVERY chunk, including the ragged last one; trainingRun
 * and the evaluationEpoch* entry points check this up front
 * (batchNorm1dRequireEvaluable).
 *
 * CONCURRENCY INVARIANT: the training forward writes runningMean/
 * runningVar/numBatchesTracked. One instance must never run two forwards
 * concurrently, nor an eval forward concurrently with a training forward
 * (no locks by design).
 *
 * A non-finite input in a batch-statistics forward writes NaN into
 * runningMean/runningVar (PyTorch parity); serializeModel will write it, but
 * the v6 reader and modelLoadStateDictBuffers reject non-finite running
 * statistics at load. */
typedef struct batchNorm1dConfig {
    parameter_t *gamma;         /* [C]; NULL iff !affine */
    parameter_t *beta;          /* [C]; NULL iff !affine */
    tensor_t *runningMean;      /* [C] FLOAT32 buffer (not a parameter); NULL iff !track */
    tensor_t *runningVar;       /* [C] FLOAT32 buffer (not a parameter); NULL iff !track */
    uint64_t numBatchesTracked; /* saturating; unused iff !track */
    size_t numChannels;
    float eps;
    bnMomentumMode_t momentumMode; /* VALUE or CUMULATIVE */
    float momentum;                /* read only for VALUE */
    bool affine;
    bool trackRunningStats;
    bool training; /* loop-owned (setLayersTrainingMode); false = eval */
    bool frozen;   /* create-time: behaves as eval-mode BN */
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;
    quantization_t *propLossQ;
    bool ownsQuantizations;
} batchNorm1dConfig_t;

/*! gamma/beta are both set or both NULL, and so are runningMean/runningVar: each
 *  pair is detected from one pointer. */
void initBatchNorm1dConfig(batchNorm1dConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                           tensor_t *runningMean, tensor_t *runningVar, size_t numChannels,
                           float eps, bnMomentumMode_t momentumMode, float momentum,
                           quantization_t *forwardQ, quantization_t *backwardQ);

void batchNorm1dForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);
/*! propLoss == NULL is a grads-only call (no input of this layer needs a gradient): the
 *  gamma/beta grads are computed and no dx memory is touched. Grads
 *  accumulate (+=); dx is overwritten. Never touches the running stats. */
void batchNorm1dBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                         const layerExec_t *exec);
void batchNorm1dCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

/*! Evaluation pre-flight: exits with the untracked-evaluation message
 *  if `layer` (a BATCHNORM1D) tracks no running statistics and an evaluation
 *  forward on `evalInputShape` ([rows, C] or [rows, C, T]) would see
 *  n = rows * prod(dims[2:]) < 2 values per channel. No-op for a tracked BN.
 *  `what` names the caller in the message. */
void batchNorm1dRequireEvaluable(const layer_t *layer, const shape_t *evalInputShape,
                                 const char *what);

#endif // ODT_BATCHNORM1D_H
