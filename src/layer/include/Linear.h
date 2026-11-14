#ifndef ENV5_RUNTIME_LINEAR_H
#define ENV5_RUNTIME_LINEAR_H
#include <stdbool.h>

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"

typedef struct layer layer_t;
typedef struct layerExec layerExec_t;

typedef struct linearConfig {
    parameter_t *weights;
    parameter_t *bias;

    arithmetic_t forwardMath;    /* declared forward compute representation */
    arithmetic_t weightGradMath; /* declared weight-grad ARITHMETIC */
    arithmetic_t biasGradMath;   /* declared bias-grad ARITHMETIC */
    arithmetic_t propLossMath;   /* declared dx-wire ARITHMETIC (kernel selection) */

    quantization_t *outputQ;   /* produced forward-wire storage config */
    quantization_t *propLossQ; /* storage config of the produced dx-wire buffer */

    outputMode_t weightGradAccMode; /* weight-grad executeOp accumulate mode */
    outputMode_t biasGradAccMode;   /* bias-grad executeOp accumulate mode */

    bool ownsQuantizations; /* true → free* will tear down outputQ/propLossQ and their
                               qConfigs */

    bool frozen; /* create-time freeze: no grad buffers; optimizer collection, state
                    allocation and backward weight/bias-grad ops skip this layer */
} linearConfig_t;

// IMPORTANT: Assumes all tensors have FLOAT32 quantization
void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output);
// IMPORTANT: Assumes all tensors have SYM_INT32 quantization
void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                           rngStream_t *rs);
/* `w` is the executeOp prologue's unpacked grouped scratch (SYM_INT32, placeholder scale);
 * `groupedWeight` carries the real per-group scales and the code domain. Reached through
 * linearForward for a grouped SYM or ASYM weight. */
void linearForwardSymInt32Grouped(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output,
                                  const symGroupedWeight_t *groupedWeight, rngStream_t *rs);
/* All operands are the executeOp prologue's unpacked BFP scratch (int32 codes under a live
 * bfpQConfig_t); `b` may be NULL; `output` is the funnel's BFP_RAW raw (arithmetic-bfp.md §13).
 * Reached through linearForward under ARITH_BFP. */
void linearForwardBfpRaw(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output);
void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output,
                   const layerExec_t *exec);

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLossTensor, const layerExec_t *exec);

/* Raw-emit kernels: each overwrites the passed tensor; the executeOp epilogue owns
 * accumulation and width restoration. The SYM_INT32 ones stamp the output's scale and
 * qMaxBits with the static width of the emitted codes, up to 32 (Matmul.h has the
 * requant-before-clone rule). The loss must be [batch, features] with features equal to the
 * bias length, as linearBackward's wire gate establishes. */
void linearCalcWeightGradsFloat32(tensor_t *forwardInput, tensor_t *loss, tensor_t *weightGrads);
void linearCalcBiasGradsFloat32(tensor_t *loss, tensor_t *biasGrad);
void linearCalcPropLossFloat32(tensor_t *loss, tensor_t *weights, tensor_t *propLoss);

void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads);
void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss);
void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss);
/* dx twin of linearForwardSymInt32Grouped, without the transpose: groups bind per visited
 * storage element. Reached through linearBackward for a grouped SYM or ASYM weight. */
void linearCalcPropLossSymInt32Grouped(tensor_t *weights, tensor_t *loss, tensor_t *propLoss,
                                       const symGroupedWeight_t *groupedWeight, rngStream_t *rs);

/* BFP backward cores, reached through linearBackward under ARITH_BFP. Every operand is the
 * executeOp prologue's unpacked BFP scratch (int32 codes under a live bfpQConfig_t); the
 * epilogue owns accumulation and re-blocking. weightGrad = loss^T @ x, propLoss = loss @ W with
 * W as stored [outF, inF], biasGrad = per-feature batch sum of the loss codes (int32 partial
 * per same-group segment; fails fast without int32 sum headroom).
 *
 * FLOAT32 raw (the float bridge, BFP_RAW_NONE): FLOAT32 grads and the fixed grid
 * (OUT_ACC_FIXED_SCALE) -- matmulBfpTensors' float fold / the float segment fold. */
void linearCalcWeightGradsBfp(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads);
void linearCalcBiasGradsBfp(tensor_t *loss, tensor_t *biasGrad);
/* Integer, BFP_RAW out (docs/conventions/arithmetic-bfp.md §14):
 *  - propLoss: every ARITH_BFP dx (BFP_RAW_BACKWARD_DX), the raw on the dx
 *    wire's geometry (or {1,0} for a FLOAT32 dx) via matmulBfpRawTensors;
 *  - weightGrad / biasGrad ...Raw: BFP grads under OUT_ACC_DYNAMIC_RESCALE
 *    (BFP_RAW_BACKWARD_ACC), the raw on the grad's geometry. weightGrad via
 *    matmulBfpRawTensors; biasGrad folds each segment into an int32 sum per
 *    feature on its own E_acc and emits with the data-derived raw exponent
 *    (bfpRawEmitFromRows). */
void linearCalcPropLossBfp(tensor_t *loss, tensor_t *weights, tensor_t *propLoss);
void linearCalcWeightGradsBfpRaw(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads);
void linearCalcBiasGradsBfpRaw(tensor_t *loss, tensor_t *biasGrad);

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape);

#endif // ENV5_RUNTIME_LINEAR_H
