#ifndef ENV5_RUNTIME_LINEAR_H
#define ENV5_RUNTIME_LINEAR_H
#include "Tensor.h"

typedef struct layer layer_t;

typedef struct linearConfig
{
    parameter_t* weights;
    parameter_t* bias;

    quantization_t* forwardQ;
    quantization_t* weightGradQ;
    quantization_t* biasGradQ;
    quantization_t* propLossQ;
} linearConfig_t;

void linearInitConfig(linearConfig_t *linearConfig, parameter_t *weights, parameter_t *bias,
                      quantization_t *forwardQ, quantization_t *weightGradQ,
                      quantization_t *biasGradQ, quantization_t *propLossQ);

// IMPORTANT: Assumes all tensors have FLOAT32 quantization
void linearForwardFloat(tensor_t* w, tensor_t* b, tensor_t* input, tensor_t* output);
// IMPORTANT: Assumes all tensors have SYM_INT32 quantization
void linearForwardSymInt32(tensor_t* w, tensor_t* b, tensor_t* input, tensor_t* output);
// IMPORTANT: Used for mismatched quantizations
void linearForward(layer_t* linearLayer, tensor_t* input, tensor_t* output);

// IMPORTANT: Assumes all tensors have FLOAT32 quantization
void backwardFloat(linearConfig_t* linearConfig, tensor_t* forwardInput, tensor_t* loss,
                   tensor_t* propLossTensor);
// IMPORTANT: Assumes all tensors have SYM_INT32 quantization
void backwardSymInt32(linearConfig_t *linearConfig, tensor_t *forwardInput, tensor_t *loss,
                         tensor_t *propLossTensor);
// IMPORTANT: Used for mismatched quantizations
void linearBackward(layer_t* linearLayer, tensor_t* forwardInput, tensor_t* loss, tensor_t* propLossTensor);


void linearCalcWeightGradsFloat32(tensor_t* loss, tensor_t* forwardInput, tensor_t* weightGrads);
void linearCalcBiasGradsFloat32(tensor_t* biasGrads, tensor_t* loss);
void linearCalcPropLossFloat32(tensor_t* weights, tensor_t* loss, tensor_t* propLoss);

void linearCalcWeightGradsSymInt32(tensor_t* loss, tensor_t* forwardInput, tensor_t* weightGrads);
void linearCalcBiasGradsSymInt32(tensor_t* biasGrads, tensor_t* loss);
void linearCalcPropLossSymInt32(tensor_t* weights, tensor_t* loss, tensor_t* propLoss);

void linearCalcOutputShape(layer_t* linearLayer, shape_t* inputShape, shape_t* outputShape);

#endif // ENV5_RUNTIME_LINEAR_H
