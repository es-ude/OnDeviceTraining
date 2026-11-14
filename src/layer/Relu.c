#define SOURCE_FILE "RELU"

#include <stdlib.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "Common.h"
#include "Comparison.h"
#include "DTypes.h"
#include "Layer.h"
#include "Quantization.h"
#include "Relu.h"
#include "ScaleCanon.h"
#include "Tensor.h"
#include "TensorConversion.h"

static bool reluIsArithmetic(arithmeticType_t type) {
    return type == ARITH_FLOAT32 || type == ARITH_SYM_INT32 || type == ARITH_BFP;
}

void reluValidateConfig(const reluConfig_t *cfg, const char *who) {
    if (!reluIsArithmetic(cfg->forwardMath.type) || !reluIsArithmetic(cfg->propLossMath.type)) {
        PRINT_ERROR("%s: forwardMath and propLossMath must be ARITH_FLOAT32, ARITH_SYM_INT32 or "
                    "ARITH_BFP (got %d, %d)",
                    who, (int)cfg->forwardMath.type, (int)cfg->propLossMath.type);
        exit(1);
    }
}

/* This path operates directly on storage; admit compatible wire dtypes before applying it. Each
 * arithmetic reads its own storage dtype: FLOAT32 floats, SYM_INT32 mantissas, BFP packed codes. */
static void reluRequireWireTypes(tensor_t *const *wires, size_t n, arithmeticType_t math,
                                 const char *where) {
    static const qtype_t wireType[] = {
        [ARITH_FLOAT32] = FLOAT32, [ARITH_SYM_INT32] = SYM_INT32, [ARITH_BFP] = BFP};
    for (size_t w = 0; w < n; w++) {
        if (wires[w]->quantization->type != wireType[math]) {
            PRINT_ERROR("%s: arithmetic %d requires dtype %d wires, wire %zu has dtype %d", where,
                        (int)math, (int)wireType[math], w, (int)wires[w]->quantization->type);
            exit(1);
        }
    }
}

void reluForward(layer_t *reluLayer, tensor_t *input, tensor_t *output, const layerExec_t *exec) {
    reluConfig_t *reluConfig = reluLayer->config->relu;
    arithmeticType_t math = reluConfig->forwardMath.type;
    reluRequireWireTypes((tensor_t *[]){input, output}, 2, math, "ReLU forward");

    switch (math) {
    case ARITH_FLOAT32:
        gteFloatValue(input, 0, 0, output);
        break;
    case ARITH_SYM_INT32: {
        symInt32QConfig_t *inputQC = input->quantization->qConfig;
        symInt32QConfig_t *outputQC = output->quantization->qConfig;
        gteSymInt32Zero(input, 0, output);
        storeCanonScale(&outputQC->scale, loadCanonScale(&inputQC->scale));
        break;
    }
    case ARITH_BFP:
        bfpRequireSameGeometry(input->quantization->qConfig, calcNumberOfElementsByTensor(input),
                               output->quantization->qConfig, calcNumberOfElementsByTensor(output),
                               "ReLU forward BFP");
        gteBfpZero(input, output);
        break;
    }
}

/* Alias-capable: propLoss may share loss's bytes (same-index read-then-write),
 * as an in-place dx plan binds them. */
static void reluBackwardFloat(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    float *inputArray = (float *)forwardInput->data;
    float *gradOutArray = (float *)loss->data;
    float *gradInArray = (float *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInArray[i] = 0;
        } else {
            gradInArray[i] = gradOutArray[i];
        }
    }
}

/* Alias-capable: propLoss may share loss's bytes (same-index read-then-write),
 * as an in-place dx plan binds them; the two wires keep their own qConfigs, so
 * the scale copy below goes from one config to the other. */
static void reluBackwardSymInt32(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);

    int32_t *inputArray = (int32_t *)forwardInput->data;
    int32_t *gradOutputArray = (int32_t *)loss->data;
    int32_t *gradInputArray = (int32_t *)propLoss->data;

    for (size_t i = 0; i < numberOfElements; i++) {
        if (inputArray[i] <= 0) {
            gradInputArray[i] = 0;
        } else {
            gradInputArray[i] = gradOutputArray[i];
        }
    }

    symInt32QConfig_t *lossQC = loss->quantization->qConfig;
    symInt32QConfig_t *propLossQC = propLoss->quantization->qConfig;
    storeCanonScale(&propLossQC->scale, loadCanonScale(&lossQC->scale));
}

/* Uses forward-input signs only (code <= 0 iff value <= 0); loss and destination must share BFP
 * geometry and disjoint storage. forwardInput may carry any grid of the same length. Kept codes
 * and loss's exponents are carried verbatim: zeroing codes only shrinks a block's absmax. */
static void reluBackwardBfp(tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss) {
    size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);
    const bfpQConfig_t *inQC = forwardInput->quantization->qConfig;
    const bfpQConfig_t *lossQC = loss->quantization->qConfig;
    bfpQConfig_t *propLossQC = propLoss->quantization->qConfig;

    int32_t inCodes[ODT_CONVERSION_CHUNK_ELEMS];
    int32_t lossCodes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < numberOfElements; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = numberOfElements - off < ODT_CONVERSION_CHUNK_ELEMS
                           ? numberOfElements - off
                           : ODT_CONVERSION_CHUNK_ELEMS;
        unpackSignExtend((const uint8_t *)forwardInput->data + off * inQC->mantissaBits / 8,
                         inQC->mantissaBits, 0, inCodes, count);
        unpackSignExtend((const uint8_t *)loss->data + off * lossQC->mantissaBits / 8,
                         lossQC->mantissaBits, 0, lossCodes, count);
        for (size_t i = 0; i < count; i++) {
            if (inCodes[i] <= 0) {
                lossCodes[i] = 0;
            }
        }
        byteConversion((uint8_t *)lossCodes, 32,
                       (uint8_t *)propLoss->data + off * propLossQC->mantissaBits / 8,
                       propLossQC->mantissaBits, count);
    }
    memcpy(propLossQC->exponents, lossQC->exponents, lossQC->numGroups);
}

void reluBackward(layer_t *reluLayer, tensor_t *forwardInput, tensor_t *loss, tensor_t *propLoss,
                  const layerExec_t *exec) {
    reluConfig_t *reluConfig = reluLayer->config->relu;
    arithmeticType_t math = reluConfig->propLossMath.type;
    reluRequireWireTypes((tensor_t *[]){forwardInput, loss, propLoss}, 3, math, "ReLU backward");

    switch (math) {
    case ARITH_FLOAT32:
        reluBackwardFloat(forwardInput, loss, propLoss);
        break;
    case ARITH_SYM_INT32:
        reluBackwardSymInt32(forwardInput, loss, propLoss);
        break;
    case ARITH_BFP: {
        size_t numberOfElements = calcNumberOfElementsByTensor(forwardInput);
        bfpRequireElementCount(calcNumberOfElementsByTensor(loss), numberOfElements,
                               "ReLU backward BFP (loss vs forwardInput)");
        bfpRequireSameGeometry(
            loss->quantization->qConfig, numberOfElements, propLoss->quantization->qConfig,
            calcNumberOfElementsByTensor(propLoss), "ReLU backward BFP (loss -> propLoss)");
        reluBackwardBfp(forwardInput, loss, propLoss);
        break;
    }
    }
}

void reluCalcOutputShape(layer_t *reluLayer, shape_t *inputShape, shape_t *outputShape) {
    memcpy(outputShape->dimensions, inputShape->dimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    memcpy(outputShape->orderOfDimensions, inputShape->orderOfDimensions,
           inputShape->numberOfDimensions * sizeof(size_t));
    outputShape->numberOfDimensions = inputShape->numberOfDimensions;
}
