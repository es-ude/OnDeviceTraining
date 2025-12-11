#ifndef ODT_LAYER_H
#define ODT_LAYER_H

#include "Tensor.h"
#include "Linear.h"

typedef struct linearConfig linearConfig_t;

typedef enum layerType
{
    LINEAR,
    RELU,
    CONV1D,
    SOFTMAX,
    SEQUENTIAL
} layerType_t;

typedef enum layerQType
{
    FLOAT_LAYER,
    ASYM_LAYER
} layerQType_t;

typedef union layerConfig
{
    linearConfig_t* linear;
} layerConfig_t;

typedef struct layer
{
    layerType_t type;
    layerConfig_t* config;
    layerQType_t qType;
    qtype_t inputQType;
    quantization_t* outputQ;
} layer_t;

typedef void (*forwardFn_t)(layer_t* layer, tensor_t* inputTensor, tensor_t* outputTensor);
typedef void (*backwardFn_t)(layer_t* layer, tensor_t* forwardInput, tensor_t* loss, tensor_t* propLoss);
typedef void (*calcOutputShapeFn_t)(layer_t* layer, shape_t* inputShape, shape_t* outputShape);

typedef struct layerFunctions
{
    forwardFn_t forward;
    backwardFn_t backward;
    calcOutputShapeFn_t calcOutputShape;
} layerFunctions_t;


extern layerFunctions_t layerFunctions[];

void initLayer(layer_t* layer, layerType_t layerType, layerConfig_t* config, layerQType_t qType, qtype_t inputQType,
               quantization_t* outputQ);


size_t calcBytesOutputData(quantization_t* outputQ, size_t numberOfOutputs);

#endif
