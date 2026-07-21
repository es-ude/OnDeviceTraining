#ifndef ENV5_RUNTIME_LINEAR_H
#define ENV5_RUNTIME_LINEAR_H
#include <stdbool.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Tensor.h"

typedef struct layer layer_t;

typedef struct linearConfig {
    parameter_t *weights;
    parameter_t *bias;

    arithmetic_t forwardMath;    /* declared forward compute representation */
    arithmetic_t weightGradMath; /* declared weight-grad ARITHMETIC */
    arithmetic_t biasGradMath;   /* declared bias-grad ARITHMETIC */
    arithmetic_t propLossMath;   /* declared dx-wire ARITHMETIC (kernel selection) */

    quantization_t *outputQ;   /* produced forward-wire storage config */
    quantization_t *propLossQ; /* storage config of the produced dx-wire buffer */

    outputMode_t weightGradAccMode; /* weight-grad executeOp accumulate mode (PR3 spec D1) */
    outputMode_t biasGradAccMode;   /* bias-grad executeOp accumulate mode (PR3 spec D1) */

    bool ownsQuantizations; /* true → free* will tear down outputQ/propLossQ and their
                               qConfigs */

    bool frozen; /* create-time freeze (#380): no grad buffers; optimizer collection,
                    state allocation and backward weight/bias-grad ops skip this layer */
} linearConfig_t;

void linearInitConfig(linearConfig_t *linearConfig, parameter_t *weights, parameter_t *bias,
                      quantization_t *forwardQ, quantization_t *backwardMath,
                      quantization_t *propLossQ);

// IMPORTANT: Assumes all tensors have FLOAT32 quantization
void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output);
// IMPORTANT: Assumes all tensors have SYM_INT32 quantization
void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output);
// IMPORTANT: Used for mismatched quantizations
void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output);

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLossTensor);

/* Raw-emit kernels: each writes into the passed tensor without accumulating;
 * accumulation and width restoration are handled by the executeOp epilogue. */
void linearCalcWeightGradsFloat32(tensor_t *forwardInput, tensor_t *loss, tensor_t *weightGrads);
void linearCalcBiasGradsFloat32(tensor_t *loss, tensor_t *biasGrad);
void linearCalcPropLossFloat32(tensor_t *loss, tensor_t *weights, tensor_t *propLoss);

void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads);
void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss);
void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss);

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_LINEAR_H
