#ifndef ODT_LAYERNORM_H
#define ODT_LAYERNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

typedef struct layerNormConfig {
    parameter_t *gamma;      /* shape normalizedShape; init 1 */
    parameter_t *beta;       /* shape normalizedShape; init 0 */
    size_t *normalizedShape; /* last-D logical dims */
    size_t numNormDims;      /* D */
    float eps;               /* default 1e-5 */
    arithmetic_t forwardMath;
    arithmetic_t propLossMath;
    quantization_t *outputQ;        /* produced forward-wire storage */
    quantization_t *propLossQ;      /* produced dx-wire storage */
    outputMode_t weightGradAccMode; /* dgamma accumulate mode (an OUT_ACC_* mode) */
    outputMode_t biasGradAccMode;   /* dbeta accumulate mode (an OUT_ACC_* mode) */
    bool ownsQuantizations;         /* true → freeLayerNormLayer tears down outputQ/propLossQ
                                     * (Owning factory); false → caller owns them (Borrowing). */
    bool frozen;                    /* create-time freeze: no grad buffers; optimizer collection,
                                       state allocation and backward weight/bias-grad ops skip this layer */
} layerNormConfig_t;

void initLayerNormConfig(layerNormConfig_t *cfg, parameter_t *gamma, parameter_t *beta,
                         size_t *normalizedShape, size_t numNormDims, float eps,
                         quantization_t *forwardQ, quantization_t *backwardQ);

/*! The config contract forward/backward rely on: normalizedShape matching gamma/beta,
 *  supported arithmetic, arithmetic/storage coherence and, for a trainable layer, grad modes
 *  and grad storage. Call once where a config enters -- the factory and the deserializer --
 *  never per step. `who` prefixes the message. */
void layerNormValidateConfig(const layerNormConfig_t *cfg, const char *who);

void layerNormForward(layer_t *layer, tensor_t *input, tensor_t *output, const layerExec_t *exec);
void layerNormBackward(layer_t *layer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                       const layerExec_t *exec);
void layerNormCalcOutputShape(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

#endif // ODT_LAYERNORM_H
