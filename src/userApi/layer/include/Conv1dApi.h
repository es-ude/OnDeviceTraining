#ifndef CONV1DAPI_H
#define CONV1DAPI_H

#include "Kernel.h"
#include "Layer.h"

/* Legacy (pre-2026-05-15 factory API) — retained during PR 1/2 coexistence window.
 * New code should use the conv1dInit_t-based factories declared in PR 2. */

/*! Legacy Conv1d factory.
 *
 * @param weights Weights with gradients
 * @param bias Optional bias parameter with gradients
 * @param kernel Kernel to be used for convolution
 * @param forwardQ Quantization for forward pass
 * @param weightGradQ Quantization for weight gradient calculation
 * @param biasGradQ Quantization for bias gradient calculation
 * @param propLossQ Quantization for prop loss calculation
 *
 * @returns Pointer to initialized layer_t
 */
layer_t *conv1dLayerInitLegacy(parameter_t *weights, parameter_t *bias, kernel_t *kernel,
                               quantization_t *forwardQ, quantization_t *weightGradQ,
                               quantization_t *biasGradQ, quantization_t *propLossQ);

/*! Frees a Conv1d layer built via the legacy factory. */
void freeConv1dLayerLegacy(layer_t *conv1dLayer);

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
} conv1dInit_t;

/*! Borrowing variant — factory allocates weights/bias/kernel internally
 *  and stores the four math `quantization_t*` from `lq` verbatim. Caller
 *  retains ownership of `lq` and the quantizations; `lq` may be a
 *  compound literal. */
layer_t *conv1dLayerInit(conv1dInit_t *init, layerQuant_t *lq);

/*! Owning variant — same as `conv1dLayerInit`, but additionally
 *  `deepCopyQuantization`s each of the four math quantizations. Caller
 *  can drop `lq` and the quantization_t's immediately. */
layer_t *conv1dLayerInitOwning(conv1dInit_t *init, layerQuant_t *lq);

/*! Tears down everything the factory allocated. Reads
 *  `config->ownsQuantizations` to decide whether to also free the four
 *  math quantizations and their qConfigs. */
void freeConv1dLayer(layer_t *conv1dLayer);

#endif // CONV1DAPI_H
