#ifndef CONV1DAPI_H
#define CONV1DAPI_H

#include "Kernel.h"
#include "Layer.h"

#include "LayerCommon.h"
#include "LayerQuant.h"

_Static_assert(VALID == 0,
               "paddingType_t::VALID must be enum value 0 so .padding zero-init defaults to VALID");

/*! Conv1d factory configuration. Build via designated initializer:
 *
 *      conv1dLayerInit(&(conv1dInit_t){
 *          .inChannels = 3, .outChannels = 16, .kernelSize = 5,
 *          .padding = SAME, .stride = 1,
 *      }, lq);
 *
 *  REQUIRED fields fire PRINT_ERROR + exit(1) if zero. Defaults below are
 *  applied when the field is zero-init (compound-literal omission). */
typedef struct conv1dInit {
    /* REQUIRED */
    size_t inChannels;
    size_t outChannels;
    size_t kernelSize;
    /* OPTIONAL — zero-init defaults */
    size_t stride;           /* 0 → 1 */
    paddingType_t padding;   /* 0 → VALID (enum value 0); SAME or EXPLICIT also valid */
    size_t paddingAmount;    /* symmetric pad per side; used ONLY when padding == EXPLICIT */
    size_t dilation;         /* 0 → 1 */
    size_t groups;           /* 0 → 1 */
    bias_t bias;             /* BIAS_DEFAULT (0) → resolves to true (PyTorch parity) */
    weightInit_t weightInit; /* zero-init → INIT_DEFAULT (PyTorch kaiming a=√5) */
    trainable_t trainable;   /* zero-init → TRAINABLE_DEFAULT (trainable) */
} conv1dInit_t;

/*! Borrowing variant — factory allocates weights/bias/kernel internally
 *  and stores the four math `quantization_t*` from `lq` verbatim. Caller
 *  retains ownership of `lq` and the quantizations; `lq` may be a
 *  compound literal.
 *  Use when: the quantizations are shared/long-lived (e.g. reused across
 *  several layers) and the caller manages their lifetime — they must
 *  outlive the layer. */
layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq);

/*! Owning variant — same as `conv1dLayerInit`, but additionally
 *  `deepCopyQuantization`s each of the four math quantizations. Caller
 *  can drop `lq` and the quantization_t's immediately.
 *  Use when: the quantizations are stack-locals or one-off configs and you
 *  want fire-and-forget teardown (freeConv1dLayer tears them down too). */
layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated. Reads
 *  `config->ownsQuantizations` to decide whether to also free the four
 *  math quantizations and their qConfigs. */
void freeConv1dLayer(layer_t *conv1dLayer);

#endif // CONV1DAPI_H
