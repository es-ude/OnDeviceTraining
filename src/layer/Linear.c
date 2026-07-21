#define SOURCE_FILE "LINEAR"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "Add.h"
#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Layer.h"
#include "Linear.h"
#include "Matmul.h"
#include "Rounding.h"
#include "TensorConversion.h"

void linearInitConfig(linearConfig_t *linearConfig, parameter_t *weights, parameter_t *bias,
                      quantization_t *forwardQ, quantization_t *backwardMath,
                      quantization_t *propLossQ) {
    linearConfig->weights = weights;
    linearConfig->bias = bias;
    linearConfig->forwardMath = arithmeticFromQuantizationOrDefault(forwardQ);
    linearConfig->weightGradMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->biasGradMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->propLossMath = arithmeticFromQuantizationOrDefault(backwardMath);
    linearConfig->outputQ = forwardQ;
    linearConfig->propLossQ = propLossQ;

    /* Today's per-callsite hardcodes (linearBackward, below), now carried on
     * the config so every caller of this init function -- factory-built or
     * hand-wired directly (test/unit/layer/UnitTestLinear.c) -- gets the
     * historical behavior without having to know about the PR3 knob. A
     * layerQuant_t-driven factory overrides these right after this call
     * (LinearApi.c) if the caller opted into a different mode. */
    linearConfig->weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE;
    linearConfig->biasGradAccMode = OUT_ACC_FIXED_SCALE;
    linearConfig->frozen = false;
}

void linearForwardFloat(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulFloat32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

void linearForwardSymInt32(tensor_t *w, tensor_t *b, tensor_t *input, tensor_t *output) {
    transposeTensor(w, 0, 1);
    matmulSymInt32TensorsWithBias(input, w, output, b);
    transposeTensor(w, 0, 1);
}

/* executeOp forward kernel adapters — operands are {input, weights} or
 * {input, weights, bias} (bias omitted, not NULL-padded, when the layer has
 * no bias); ctx unused (matmul infers geometry from the tensors themselves). */
static void linearForwardKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                     const void *ctx) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardFloat(ops[1], bias, ops[0], rawOut);
}
static void linearForwardKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                   const void *ctx) {
    (void)auxOut;
    (void)ctx;
    tensor_t *bias = (n > 2) ? ops[2] : NULL;
    linearForwardSymInt32(ops[1], bias, ops[0], rawOut);
}

void linearForward(layer_t *linearLayer, tensor_t *input, tensor_t *output) {
    linearConfig_t *linearConfig = linearLayer->config->linear;

    tensor_t *weights = getParamFromParameter(linearConfig->weights);
    tensor_t *bias = linearConfig->bias != NULL ? getParamFromParameter(linearConfig->bias) : NULL;

    executeOp(
        &(opSpec_t){
            .kernel = linearConfig->forwardMath.type == ARITH_SYM_INT32 ? linearForwardKernelSym
                                                                        : linearForwardKernelFloat,
            .inputs = bias != NULL ? (tensor_t *[]){input, weights, bias}
                                   : (tensor_t *[]){input, weights},
            .nInputs = bias != NULL ? 3 : 2,
            .arithmetic = linearConfig->forwardMath,
            .mode = OUT_WRITE,
        },
        output);
}

void linearCalcWeightGradsFloat32(tensor_t *forwardInput, tensor_t *loss, tensor_t *weightGrads) {
    transposeTensor(loss, 0, 1);
    matmulFloat32Tensors(loss, forwardInput, weightGrads);
    transposeTensor(loss, 0, 1);
}

void linearCalcBiasGradsFloat32(tensor_t *loss, tensor_t *biasGrad) {
    /* Raw emit: per-feature batch sums of loss values; the executeOp epilogue adds. */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrad);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    float *bg = (float *)biasGrad->data;
    float *l = (float *)loss->data;
    for (size_t f = 0; f < numFeatures; f++) {
        float sum = 0.0f;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] = sum;
    }
}

void linearCalcPropLossFloat32(tensor_t *loss, tensor_t *weights, tensor_t *propLoss) {
    matmulFloat32Tensors(loss, weights, propLoss);
}

void linearCalcWeightGradsSymInt32(tensor_t *loss, tensor_t *forwardInput, tensor_t *weightGrads) {
    transposeTensor(loss, 1, 0);
    matmulSymInt32Tensors(loss, forwardInput, weightGrads);
    transposeTensor(loss, 1, 0);
}

void linearCalcBiasGradsSymInt32(tensor_t *biasGrads, tensor_t *loss) {
    /* Raw emit: per-feature batch sums of loss mantissas, at the LOSS scale.
     * The executeOp OUT_ACC_FIXED_SCALE epilogue rescales into the persistent
     * grad's existing scale and integer-adds (Deutel-adjacent ODT scheme). */
    size_t numFeatures = calcNumberOfElementsByTensor(biasGrads);
    size_t numLoss = calcNumberOfElementsByTensor(loss);
    size_t batch = (numFeatures == 0) ? 0 : numLoss / numFeatures;
    int32_t *bg = (int32_t *)biasGrads->data;
    int32_t *l = (int32_t *)loss->data;
    for (size_t f = 0; f < numFeatures; f++) {
        /* int32 accumulator (NO int64 in SYM paths): loss mantissas are
         * int16-range per the qMaxBits<=16 contract, so the batch sum stays
         * within int32 for any batch <= 65536 - far beyond any real batch. */
        int32_t sum = 0;
        for (size_t n = 0; n < batch; n++) {
            sum += l[n * numFeatures + f];
        }
        bg[f] = sum;
    }
    ((symInt32QConfig_t *)biasGrads->quantization->qConfig)->scale =
        ((symInt32QConfig_t *)loss->quantization->qConfig)->scale;
}

void linearCalcPropLossSymInt32(tensor_t *weights, tensor_t *loss, tensor_t *propLoss) {
    matmulSymInt32Tensors(loss, weights, propLoss);
}

/* executeOp kernel adapters — ops convention: weight-grad {loss, fwdIn},
 * bias-grad {loss}, propLoss {loss, weightsParam}. auxOut/ctx unused here. */
static void weightGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                  const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsFloat32(ops[1], ops[0], rawOut);
}
static void weightGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcWeightGradsSymInt32(ops[0], ops[1], rawOut);
}
static void biasGradKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsFloat32(ops[0], rawOut);
}
static void biasGradKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcBiasGradsSymInt32(rawOut, ops[0]);
}
static void propLossKernelFloat(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                                const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossFloat32(ops[0], ops[1], rawOut);
}
static void propLossKernelSym(tensor_t **ops, size_t n, tensor_t *rawOut, tensor_t *auxOut,
                              const void *ctx) {
    (void)n;
    (void)auxOut;
    (void)ctx;
    linearCalcPropLossSymInt32(ops[1], ops[0], rawOut);
}

void linearBackward(layer_t *linearLayer, tensor_t *forwardInput, tensor_t *loss,
                    tensor_t *propLoss) {
    linearConfig_t *cfg = linearLayer->config->linear;

    if (!cfg->frozen) {
        executeOpValidateAccMode(cfg->weightGradAccMode, "Linear weightGradAccMode");
        executeOp(
            &(opSpec_t){
                .kernel = cfg->weightGradMath.type == ARITH_SYM_INT32 ? weightGradKernelSym
                                                                      : weightGradKernelFloat,
                .inputs = (tensor_t *[]){loss, forwardInput},
                .nInputs = 2,
                .arithmetic = cfg->weightGradMath,
                .mode = cfg->weightGradAccMode,
            },
            getGradFromParameter(cfg->weights));

        if (cfg->bias != NULL) {
            executeOpValidateAccMode(cfg->biasGradAccMode, "Linear biasGradAccMode");
            executeOp(
                &(opSpec_t){
                    .kernel = cfg->biasGradMath.type == ARITH_SYM_INT32 ? biasGradKernelSym
                                                                        : biasGradKernelFloat,
                    .inputs = (tensor_t *[]){loss},
                    .nInputs = 1,
                    .arithmetic = cfg->biasGradMath,
                    .mode = cfg->biasGradAccMode,
                },
                getGradFromParameter(cfg->bias));
        }
    }

    executeOp(
        &(opSpec_t){
            .kernel =
                cfg->propLossMath.type == ARITH_SYM_INT32 ? propLossKernelSym : propLossKernelFloat,
            .inputs = (tensor_t *[]){loss, getParamFromParameter(cfg->weights)},
            .nInputs = 2,
            .arithmetic = cfg->propLossMath,
            .mode = OUT_WRITE,
        },
        propLoss);
}

void linearCalcOutputShape(layer_t *linearLayer, shape_t *inputShape, shape_t *outputShape) {
    if (inputShape->numberOfDimensions != 2) {
        PRINT_ERROR("Linear layer expects 2D input, got %luD\n", inputShape->numberOfDimensions);
    }

    size_t batchSize = inputShape->dimensions[0];

    linearConfig_t *cfg = linearLayer->config->linear;
    shape_t *weightShape = cfg->weights->param->shape;
    size_t outFeatures = weightShape->dimensions[0];

    outputShape->dimensions[0] = batchSize;
    outputShape->dimensions[1] = outFeatures;

    outputShape->numberOfDimensions = inputShape->numberOfDimensions;

    setOrderOfDimsForNewTensor(inputShape->numberOfDimensions, outputShape->orderOfDimensions);
}
