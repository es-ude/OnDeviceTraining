#ifndef ODT_LAYER_H
#define ODT_LAYER_H

#include <stdbool.h>

#include "Tensor.h"

typedef struct linearConfig linearConfig_t;
typedef struct reluConfig reluConfig_t;
typedef struct softmaxConfig softmaxConfig_t;
typedef struct conv1dConfig conv1dConfig_t;
typedef struct conv1dTransposedConfig conv1dTransposedConfig_t;
typedef struct maxPool1dConfig maxPool1dConfig_t;
typedef struct avgPool1dConfig avgPool1dConfig_t;
typedef struct adaptiveAvgPool1dConfig adaptiveAvgPool1dConfig_t;
typedef struct dropoutConfig dropoutConfig_t;
typedef struct layerNormConfig layerNormConfig_t;
typedef struct groupNormConfig groupNormConfig_t;
typedef struct quantizationConfig quantizationConfig_t;

/* WIRE FORMAT -- append-only. The serialized layer record's uint8 tag is this
 * enum's POSITION (src/serial/Serialize.c). Never insert or reorder members:
 * that renumbers the tail and silently corrupts every previously serialized
 * model. Append new members at the end and extend the wire-tag pins in
 * test/unit/serial/UnitTestSerialize.c. */
typedef enum layerType {
    LINEAR,
    RELU,
    CONV1D,
    CONV1D_TRANSPOSED,
    MAXPOOL1D,
    AVGPOOL1D,
    SOFTMAX,
    FLATTEN,
    QUANTIZATION,
    ADAPTIVE_AVGPOOL1D,
    DROPOUT,
    LAYERNORM,
    GROUPNORM
} layerType_t;

typedef enum layerQType { FLOAT_LAYER, ASYM_LAYER } layerQType_t;

typedef union layerConfig {
    linearConfig_t *linear;
    reluConfig_t *relu;
    softmaxConfig_t *softmax;
    conv1dConfig_t *conv1d;
    conv1dTransposedConfig_t *conv1dTransposed;
    maxPool1dConfig_t *maxPool1d;
    avgPool1dConfig_t *avgPool1d;
    adaptiveAvgPool1dConfig_t *adaptiveAvgPool1d;
    dropoutConfig_t *dropout;
    layerNormConfig_t *layerNorm;
    groupNormConfig_t *groupNorm;
    quantizationConfig_t *quantization;
} layerConfig_t;

typedef struct layer {
    layerType_t type;
    layerConfig_t *config;
} layer_t;

typedef void (*forwardFn_t)(layer_t *layer, tensor_t *inputTensor, tensor_t *outputTensor);
typedef void (*backwardFn_t)(layer_t *layer, tensor_t *forwardInput, tensor_t *loss,
                             tensor_t *propLoss);
typedef void (*calcOutputShapeFn_t)(layer_t *layer, shape_t *inputShape, shape_t *outputShape);

typedef struct layerFunctions {
    forwardFn_t forward;
    backwardFn_t backward;
    calcOutputShapeFn_t calcOutputShape;
} layerFunctions_t;

extern layerFunctions_t layerFunctions[];

void initLayer(layer_t *layer, layerType_t layerType, layerConfig_t *config);

/*! Whether `layer` was frozen at create time (#380): true for LINEAR/CONV1D/
 *  CONV1D_TRANSPOSED/LAYERNORM/GROUPNORM configs with `frozen == true`,
 *  false for every other layer type (no trainable params, or not yet wired). */
bool layerIsFrozen(const layer_t *layer);

#endif
