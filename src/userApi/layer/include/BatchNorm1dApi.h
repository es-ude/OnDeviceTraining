#ifndef ODT_BATCHNORM1D_API_H
#define ODT_BATCHNORM1D_API_H

#include <stdbool.h>
#include <stddef.h>

#include "BatchNorm1d.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"

/* Zero-init = PyTorch nn.BatchNorm1d defaults (affine, tracking, momentum
 * 0.1, eps 1e-5, trainable). eps = 0 is NOT representable (0 -> 1e-5, the
 * repo idiom); momentum 0 needs BN_MOMENTUM_VALUE. */
typedef struct batchNorm1dInit {
    size_t numChannels;            /* REQUIRED, C > 0 */
    float eps;                     /* 0 -> 1e-5; < 0 or non-finite rejected */
    bnMomentumMode_t momentumMode; /* DEFAULT (0.1) / VALUE / CUMULATIVE (None) */
    float momentum;                /* VALUE only: finite, in [0, 1]; 0 = stats never move */
    bool noAffine;                 /* true -> no gamma/beta (affine=False) */
    bool noRunningStats;           /* true -> track_running_stats=False */
    trainable_t trainable;         /* TRAINABLE_FALSE -> frozen = eval-mode BN; noAffine +
                                       TRAINABLE_TRUE is rejected (nothing to train) */
} batchNorm1dInit_t;

/*! Borrowing factory: stores lq->outputQ/propLossQ verbatim (caller keeps
 *  ownership; they must outlive the layer). FLOAT32 only: every declared
 *  math and storage slot must be FLOAT32 (weight/bias storage and grad
 *  storage only when affine). Allocates gamma = 1 / beta = 0 [C] (+ FLOAT32
 *  grads unless frozen) iff affine, running_mean = 0 / running_var = 1 [C]
 *  iff tracking. Use when: the quantization configs are shared/long-lived. */
layer_t *batchNorm1dLayerInit(batchNorm1dInit_t *init, layerQuant_t *lq);

/*! Owning factory: deep-copies outputQ/propLossQ (freeBatchNorm1dLayer tears
 *  them down). Use when: the configs are stack-locals or one-off. */
layer_t *batchNorm1dLayerInitOwning(batchNorm1dInit_t *init, layerQuant_t *lq);

/*! Frees gamma/beta (+ grads), the running buffers and the wrappers; the
 *  outputQ/propLossQ copies too iff the Owning factory built the layer. */
void freeBatchNorm1dLayer(layer_t *layer);

#endif // ODT_BATCHNORM1D_API_H
