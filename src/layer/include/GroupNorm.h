#ifndef ODT_GROUPNORM_H
#define ODT_GROUPNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Tensor.h"

typedef struct layer layer_t;

/* GroupNorm over rank-3 [B, C, T] inputs (identity order, fail-fast guarded):
 * per-(b,g) statistics over N = C/G * T elements, per-CHANNEL affine — the
 * key delta vs LayerNorm, whose gamma/beta are per-ELEMENT over the
 * normalized dims. PyTorch nn.GroupNorm parity (biased variance, eps inside
 * sqrt). */
typedef struct groupNormConfig {
    parameter_t *gamma; /* [C]; factory init 1 */
    parameter_t *beta;  /* [C]; factory init 0 */
    size_t numGroups;   /* G; must divide numChannels (factory-validated) */
    size_t numChannels; /* C */
    float eps;          /* default 1e-5 */
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;        /* produced forward-wire storage */
    quantization_t *propLossQ;      /* produced dx-wire storage */
    outputMode_t weightGradAccMode; /* dgamma executeOp accumulate mode (PR3 spec D1) */
    outputMode_t biasGradAccMode;   /* dbeta executeOp accumulate mode (PR3 spec D1) */
    bool ownsQuantizations;         /* true → freeGroupNormLayer tears down outputQ/propLossQ
                                     * (Owning factory); false → caller owns them (Borrowing). */
} groupNormConfig_t;

void initGroupNormConfig(groupNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t numGroups, size_t numChannels, float eps, quantization_t *forwardQ,
                         quantization_t *backwardQ);

void groupNormForward(layer_t *layer, tensor_t *input, tensor_t *output);
void groupNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss);
void groupNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_GROUPNORM_H
