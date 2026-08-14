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
/* Group-quant PR2 (Task 3): `w` is the executeOp prologue's unpacked grouped-
 * SYM scratch (SYM_INT32 dtype, poisoned scale); `weightGroups` carries the
 * real per-group scales/qBits/groupSize. Routed to by linearForwardKernelSym
 * when the stored weight is grouped SYM — not a direct public entry point. */
void linearForwardSymInt32Grouped(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                                  const symQConfig_t *weightGroups);
/* BFP epic PR2 (Task 7): the BFP twin of linearForwardSymInt32Grouped's role.
 * All operands are the executeOp prologue's unpacked-BFP scratch (int32
 * mantissa codes under a live bfpQConfig_t; `b` NULL-able) — same transpose
 * dance around matmulBfpTensors. Routed to by the funnel adapter — not a
 * direct public entry point. */
void linearForwardBfp(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output);
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
/* Group-quant PR3 (Task 1): dx sibling of linearForwardSymInt32Grouped —
 * `weights` is the executeOp prologue's unpacked grouped-SYM scratch,
 * `weightGroups` the real per-group scales/qBits/groupSize. NO transpose:
 * dx reduces over weight dim-0 (outFeatures), storage-strided by inFeatures;
 * the unified matmul core binds groups per visited storage element. Routed
 * to by propLossKernelSym when the stored weight is grouped SYM. */
void linearCalcPropLossSymInt32Grouped(tensor_t *weights, tensor_t *loss, tensor_t *propLoss,
                                       const symQConfig_t *weightGroups);

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_LINEAR_H
