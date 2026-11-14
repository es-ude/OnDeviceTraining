#ifndef ODT_GROUPNORM_H
#define ODT_GROUPNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

/* GroupNorm over rank-3 identity-order [B, C, T] inputs: statistics per (b, g) over
 * N = C/G * T elements, affine per CHANNEL (LayerNorm's is per element). PyTorch
 * nn.GroupNorm semantics: biased variance, eps inside the sqrt. */
typedef struct groupNormConfig {
    parameter_t *gamma; /* [C]; factory init 1 */
    parameter_t *beta;  /* [C]; factory init 0 */
    size_t numGroups;   /* G; divides numChannels */
    size_t numChannels; /* C */
    float eps;          /* default 1e-5 */
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;        /* produced forward-wire storage */
    quantization_t *propLossQ;      /* produced dx-wire storage */
    outputMode_t weightGradAccMode; /* dgamma accumulate mode (an OUT_ACC_* mode) */
    outputMode_t biasGradAccMode;   /* dbeta accumulate mode (an OUT_ACC_* mode) */
    bool ownsQuantizations;         /* true → freeGroupNormLayer tears down outputQ/propLossQ
                                     * (Owning factory); false → caller owns them (Borrowing). */
    bool frozen;                    /* create-time freeze: no grad buffers; optimizer collection,
                                       state allocation and backward weight/bias-grad ops skip this layer */
} groupNormConfig_t;

void initGroupNormConfig(groupNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t numGroups, size_t numChannels, float eps, quantization_t *forwardQ,
                         quantization_t *backwardQ);

/*! The config contract forward/backward rely on: group geometry, supported arithmetic,
 *  arithmetic/storage coherence and, for a trainable layer, grad modes and grad storage.
 *  Call once where a config enters -- the factory and the deserializer -- never per step.
 *  `who` prefixes the message. */
void groupNormValidateConfig(const groupNormConfig_t *cfg, const char *who);

void groupNormForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);
void groupNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                       const layerExec_t *exec);
void groupNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_GROUPNORM_H
