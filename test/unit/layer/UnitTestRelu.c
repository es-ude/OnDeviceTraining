#include <string.h>

#include "ArithmeticType.h"
#include "BfpKernelSupport.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "LayerQuant.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testReluForwardFloat() {
    size_t numberOfElements = 6;

    /* 1. Build heap input tensor (shape 2x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap output tensor (shape 2x3). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    /* 3. Build shared float quantization for the layer. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *reluLayer = reluLayerInit(&lq);

    /* 4. Exercise. */
    reluForward(reluLayer, input, output);

    /* 5. CAPTURE assertion values. */
    float captured[6];
    readBytesAsFloatArray(numberOfElements, output->data, captured);

    /* 6. FREE in reverse-init order. freeReluLayer releases only the layer
     *    config wrapper; the shared floatQ is freed exactly once at the end. */
    freeReluLayer(reluLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    /* 7. ASSERT on captured. */
    float expected[] = {0.f, 0.f, 1.f, 2.f, 5.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, numberOfElements);
}

void testReluForwardSymInt32() {
    size_t numberOfValues = 6;

    /* 1. Build heap input tensor (SymInt32, shape 2x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap output tensor (SymInt32, shape 2x3). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 3. Shared SymInt32 quantization for the layer. */
    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, symIntQ);
    layer_t *reluLayer = reluLayerInit(&lq);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.forward(reluLayer, input, output);

    /* 4. Convert SymInt32 output back to Float for comparison; output buffer
     *    is heap-allocated to keep us in the heap-tier idiom. */
    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 2;
    outFloatDims[1] = 3;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

    /* 5. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    /* 6. FREE. */
    freeTensor(outputFloat);
    freeReluLayer(reluLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(symIntQ);

    /* 7. ASSERT. */
    float expected[] = {0, 0, 1, 2, 5, 0};
    for (size_t i = 0; i < numberOfValues; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testReluBackwardFloat() {
    size_t numberOfElements = 6;

    /* 1. Build heap forwardInput tensor. */
    size_t *fwdDims = reserveMemory(1 * sizeof(size_t));
    fwdDims[0] = numberOfElements;
    size_t *fwdOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 1, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor. */
    size_t *lossDims = reserveMemory(1 * sizeof(size_t));
    lossDims[0] = numberOfElements;
    size_t *lossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 1, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

    /* 3. Build heap propLoss tensor (output of backward). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfElements;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    /* 4. Build the layer with shared float quantization. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *reluLayer = reluLayerInit(&lq);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    /* 5. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < numberOfElements; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    /* 6. FREE. */
    freeReluLayer(reluLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* 7. ASSERT. */
    float expected[] = {0.f, 0.f, -4.f, 6.f, 3.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, numberOfElements);
}

void testReluBackwardSymInt32() {
    size_t numberOfValues = 6;

    /* 1. Build heap forwardInput tensor (SymInt32). */
    size_t *fwdDims = reserveMemory(1 * sizeof(size_t));
    fwdDims[0] = numberOfValues;
    size_t *fwdOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 1, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){-1.f, 0.f, 1.f, 2.f, 5.f, -6.f}, 6);

    /* 2. Build heap loss tensor (SymInt32). */
    size_t *lossDims = reserveMemory(1 * sizeof(size_t));
    lossDims[0] = numberOfValues;
    size_t *lossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 1, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){0.f, 2.f, -4.f, 6.f, 3.f, 2.f}, 6);

    /* 3. Build heap propLoss tensor (SymInt32). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfValues;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 4. Build layer with shared SymInt32 quantization. */
    quantization_t *symIntQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, symIntQ);
    layer_t *reluLayer = reluLayerInit(&lq);
    layerFunctions_t reluFns = layerFunctions[RELU];
    reluFns.backward(reluLayer, forwardInput, loss, propLoss);

    /* 5. Convert SymInt32 propLoss back to Float for comparison. */
    size_t *propLossFloatDims = reserveMemory(1 * sizeof(size_t));
    propLossFloatDims[0] = numberOfValues;
    size_t *propLossFloatOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossFloatOrder);
    shape_t *propLossFloatShape = reserveMemory(sizeof(shape_t));
    setShape(propLossFloatShape, propLossFloatDims, 1, propLossFloatOrder);
    tensor_t *propLossFloat = initTensor(propLossFloatShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat);

    /* 6. CAPTURE. */
    float captured[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        captured[i] = ((float *)propLossFloat->data)[i];
    }

    /* 7. FREE. */
    freeTensor(propLossFloat);
    freeReluLayer(reluLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(symIntQ);

    /* 8. ASSERT. */
    float expected[] = {0, 0, -4, 6, 3, 0};
    for (size_t i = 0; i < numberOfValues; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testReluLayerInitAndFreeRoundTrip(void) {
    /* Roundtrip: reluLayerInit allocates layer + outer layerConfig +
     * inner reluConfig (3 reserveMemory calls). freeReluLayer must
     * release all three. reluLayerInit requires non-NULL outputQ/propLossQ
     * (validateLayerQuantForRelu), so this uses a minimal real profile
     * instead of the old Legacy ctor's NULL-tolerant borrow. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *reluLayer = reluLayerInit(&lq);
    TEST_ASSERT_NOT_NULL(reluLayer);
    TEST_ASSERT_EQUAL_INT(RELU, reluLayer->type);
    TEST_ASSERT_NOT_NULL(reluLayer->config);
    TEST_ASSERT_NOT_NULL(reluLayer->config->relu);

    freeReluLayer(reluLayer);
    freeQuantization(q);
}

/* ============================================================================
 * Tests for the new layerQuant_t-based factory API (PR 1).
 * ========================================================================== */

void testReluLayerInitBorrowingStoresLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qFwd),
        .propLossMath = arithmeticFromQuantization(qBwd),
        .outputQ = qFwd,
        .propLossQ = qBwd,
        /* weightStorage / biasStorage ignored by ReLU */
    };

    layer_t *layer = reluLayerInit(&lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(RELU, layer->type);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_EQUAL_PTR(qFwd, cfg->outputQ);
    TEST_ASSERT_EQUAL_PTR(qBwd, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->propLossMath.type);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    freeReluLayer(layer);
}

void testReluLayerInitOwningDeepCopiesLqPointers(void) {
    quantization_t *qFwd = quantizationInitFloat();
    quantization_t *qBwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(qFwd),
        .propLossMath = arithmeticFromQuantization(qBwd),
        .outputQ = qFwd,
        .propLossQ = qBwd,
    };

    layer_t *layer = reluLayerInitOwning(&lq);

    reluConfig_t *cfg = layer->config->relu;
    TEST_ASSERT_NOT_EQUAL(qFwd, cfg->outputQ);
    TEST_ASSERT_NOT_EQUAL(qBwd, cfg->propLossQ);
    TEST_ASSERT_EQUAL_INT(qFwd->type, cfg->outputQ->type);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->forwardMath.type);
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeReluLayer(layer);
}

void setUp() {}
void tearDown() {}

/* BFP epic PR2 Task 8: heap BFP wire, per-tensor {1,0}, 8-bit mantissas. The
 * guards under test read only ->quantization->type, so the payload content is
 * irrelevant — what matters is that the buffer is BFP-SIZED (n bytes, not
 * n floats): an unguarded float* read/write would run past it. */
static tensor_t *buildBfpTensor1D(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBfp(8, 8, HALF_AWAY), NULL);
}

/* BFP epic PR4: build a BFP wire with EXACT codes and per-group exponents.
 * Writing the packed payload directly (byteConversion) instead of quantizing
 * keeps the fixture independent of the quantizer and lets the test pin the
 * verbatim exponent carry with two DIFFERENT group exponents. */
static tensor_t *buildBfpWireWithCodes(size_t const *dims, size_t numDims, uint8_t mantissaBits,
                                       uint8_t exponentBits, size_t numGroups, size_t groupSize,
                                       int32_t *codes, uint8_t const *exponents) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    quantization_t *q = numGroups > 1 ? quantizationInitBfpGrouped(mantissaBits, exponentBits,
                                                                   HALF_AWAY, numGroups, groupSize)
                                      : quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY);
    tensor_t *t = initTensor(shape, q, NULL);
    size_t n = calcNumberOfElementsByTensor(t);
    if (codes != NULL) {
        byteConversion((uint8_t *)codes, 32, t->data, mantissaBits, n);
    }
    bfpQConfig_t *qc = q->qConfig;
    if (exponents != NULL) {
        memcpy(qc->exponents, exponents, numGroups);
    }
    return t;
}

static tensor_t *buildFloatTensor1D(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

/* BFP epic PR4 (R-P2, spec §5 Relu row + deviation 6): ReLU on packed BFP is a
 * pure code clamp — negatives to code 0 — with the group exponents copied
 * VERBATIM. Two groups with DIFFERENT exponents (130 / 126) so a dropped or
 * zero-state exponent copy is observable; the output codes are pre-filled with
 * a -9 sentinel so a kernel that skips positives is observable too. */
void testReluForwardBfpClampsCodesAndCarriesExponents(void) {
    size_t dims[] = {1, 8};
    int32_t inCodes[8] = {12, -7, 0, 31, -32, 5, -1, 20};
    uint8_t inExps[2] = {130, 126};
    int32_t outCodes[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t outExps[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, inCodes, inExps);
    tensor_t *output = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, outCodes, outExps);

    reluForwardBfp(input, output);

    int32_t got[8];
    unpackSignExtend(output->data, 6, 0, got, 8);
    int32_t expected[8] = {12, 0, 0, 31, 0, 5, 0, 20};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 8);

    bfpQConfig_t *outQC = output->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(130, outQC->exponents[0],
                                    "group 0 exponent must be carried verbatim");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(126, outQC->exponents[1],
                                    "group 1 exponent must be carried verbatim");

    int32_t inAfter[8];
    unpackSignExtend(input->data, 6, 0, inAfter, 8);
    int32_t inUnchanged[8] = {12, -7, 0, 31, -32, 5, -1, 20};
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(inUnchanged, inAfter, 8,
                                          "the input wire must not be modified");
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4: the POSITIVE half of the narrowed guard — reluForward must
 * DISPATCH an ARITH_BFP config to reluForwardBfp instead of rejecting the wire.
 * The three death tests below are all satisfied by the old, broader
 * requireNoBfpWire (it exits on every BFP wire), so none of them can tell a
 * present ARITH_BFP arm from a deleted one; this one can. */
void testReluForwardBfpArmDispatchesThroughReluForward(void) {
    reluConfig_t cfg = {0};
    cfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t dims[] = {1, 8};
    int32_t inCodes[8] = {12, -7, 0, 31, -32, 5, -1, 20};
    uint8_t inExps[2] = {130, 126};
    int32_t outCodes[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t outExps[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, inCodes, inExps);
    tensor_t *output = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, outCodes, outExps);

    reluForward(&layer, input, output);

    int32_t got[8];
    unpackSignExtend(output->data, 6, 0, got, 8);
    int32_t expected[8] = {12, 0, 0, 31, 0, 5, 0, 20};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 8);
    bfpQConfig_t *outQC = output->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(130, outQC->exponents[0],
                                    "group 0 exponent must be carried verbatim");
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4 (R-P2 backward): the mask is the SIGN of forwardInput's packed
 * codes — exponents are unsigned scale factors, so the sign bit alone decides
 * kept/dropped and no dequant is needed. Kept positions copy the loss code
 * verbatim, dropped positions write code 0, and loss's group exponents are
 * carried verbatim onto propLoss. forwardInput deliberately carries a
 * DIFFERENT grid (per-tensor, m=8) from the loss/propLoss pair: its geometry
 * is not gated because it is read sign-only. */
void testReluBackwardBfpMasksBySignAndCarriesExponents(void) {
    size_t dims[] = {1, 8};
    int32_t fwdCodes[8] = {5, -3, 0, 40, -1, 17, 9, -80};
    uint8_t fwdExps[1] = {129};
    int32_t lossCodes[8] = {11, -6, 30, -12, 4, -20, 7, 25};
    uint8_t lossExps[2] = {131, 124};
    int32_t propCodes[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t propExps[2] = {127, 127};
    tensor_t *forwardInput = buildBfpWireWithCodes(dims, 2, 8, 8, 1, 0, fwdCodes, fwdExps);
    tensor_t *loss = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, lossCodes, lossExps);
    tensor_t *propLoss = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, propCodes, propExps);

    reluBackwardBfp(forwardInput, loss, propLoss);

    int32_t got[8];
    unpackSignExtend(propLoss->data, 6, 0, got, 8);
    /* mask: fwd code <= 0 drops (indices 1, 2, 4, 7) — <= 0 matches the
     * FLOAT32/SYM arms exactly, and code <= 0 iff value <= 0 (scale > 0). */
    int32_t expected[8] = {11, 0, 0, -12, 0, -20, 7, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 8);

    bfpQConfig_t *propQC = propLoss->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(131, propQC->exponents[0],
                                    "loss group 0 exponent must be carried verbatim");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(124, propQC->exponents[1],
                                    "loss group 1 exponent must be carried verbatim");
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
}

/* BFP epic PR4: the POSITIVE half of the backward narrowing — reluBackward must
 * DISPATCH an ARITH_BFP config to reluBackwardBfp. Same argument as the
 * forward twin: every death test here is already satisfied by the old blanket
 * requireNoBfpWire, so only this one can tell a present arm from a deleted one. */
void testReluBackwardBfpArmDispatchesThroughReluBackward(void) {
    reluConfig_t cfg = {0};
    cfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t dims[] = {1, 8};
    int32_t fwdCodes[8] = {5, -3, 0, 40, -1, 17, 9, -80};
    uint8_t fwdExps[1] = {129};
    int32_t lossCodes[8] = {11, -6, 30, -12, 4, -20, 7, 25};
    uint8_t lossExps[2] = {131, 124};
    int32_t propCodes[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t propExps[2] = {127, 127};
    tensor_t *forwardInput = buildBfpWireWithCodes(dims, 2, 8, 8, 1, 0, fwdCodes, fwdExps);
    tensor_t *loss = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, lossCodes, lossExps);
    tensor_t *propLoss = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, propCodes, propExps);

    reluBackward(&layer, forwardInput, loss, propLoss);

    int32_t got[8];
    unpackSignExtend(propLoss->data, 6, 0, got, 8);
    int32_t expected[8] = {11, 0, 0, -12, 0, -20, 7, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 8);
    bfpQConfig_t *propQC = propLoss->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(131, propQC->exponents[0],
                                    "loss group 0 exponent must be carried verbatim");
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
}

/* R-P7 asks for a "BFP wire under a pinned ARITH_FLOAT32" fake-quant test
 * mirroring testAvgPool1dForwardWithSymInt32Input. For the POOLS that test
 * exists (Task 5's testAvgPool1dForwardBfpWireUnderPinnedFloat32) because
 * their forwards run INSIDE executeOp, whose ARITH_FLOAT32 prologue dequants
 * any storage dtype. ReLU has no such bridge BY CONSTRUCTION: it never enters
 * the funnel and its FLOAT32 arm raw-casts ->data, so a BFP wire under a
 * pinned ARITH_FLOAT32 is not fake-quant, it is heap corruption. The honest
 * mirror is therefore this death test — the same fixture, the opposite
 * expectation. */

/* BFP epic PR4 (R-P7d): the narrowed guards. A BFP wire under the FLOAT32 arm
 * must still die (#315 parity — the raw float* view would read packed bytes),
 * and the ARITH_BFP arm must reject a non-BFP wire and a geometry mismatch. */
void testReluForwardFloat32ArmRejectsBfpWire(void) {
    reluConfig_t cfg = {0};
    cfg.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t dims[] = {1, 8};
    tensor_t *bfpWire = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, NULL, NULL);
    tensor_t *floatWire = buildFloatTensor1D(8);

    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, bfpWire, floatWire));
    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, floatWire, bfpWire));

    freeTensor(floatWire);
    freeTensor(bfpWire);
}

void testReluForwardBfpArmRejectsNonBfpWireAndGeometryMismatch(void) {
    reluConfig_t cfg = {0};
    cfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t dims[] = {1, 8};
    tensor_t *bfpWire = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, NULL, NULL);
    tensor_t *floatWire = buildFloatTensor1D(8);
    tensor_t *otherGrid = buildBfpWireWithCodes(dims, 2, 6, 8, 4, 2, NULL, NULL);

    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, floatWire, bfpWire));
    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, bfpWire, floatWire));
    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, bfpWire, otherGrid));

    freeTensor(otherGrid);
    freeTensor(floatWire);
    freeTensor(bfpWire);
}

/* BFP epic PR4: the length half of the gate, and the ONLY case the grid check
 * cannot see. Both wires carry the per-tensor sentinel {numGroups=1,
 * groupSize=0}, which validateBfpQConfigShape accepts for ANY element count
 * and which compares field-for-field EQUAL between the two — so before the
 * count check this pair sailed through and gteBfpZero unpacked 8 elements out
 * of a 4-element buffer and packed 8 into it. Both directions are asserted:
 * a short destination over-writes, a short source over-reads. */
void testReluForwardBfpArmRejectsUnequalElementCounts(void) {
    reluConfig_t cfg = {0};
    cfg.forwardMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t longDims[] = {1, 8};
    size_t shortDims[] = {1, 4};
    tensor_t *longWire = buildBfpWireWithCodes(longDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *shortWire = buildBfpWireWithCodes(shortDims, 2, 6, 8, 1, 0, NULL, NULL);

    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, longWire, shortWire));
    ASSERT_EXITS_WITH_FAILURE(reluForward(&layer, shortWire, longWire));

    freeTensor(shortWire);
    freeTensor(longWire);
}

/* BFP epic PR4 (R-P7d): FLOAT32 arm still rejects a BFP wire (#315 parity, now
 * via the per-arm dtype guard); the ARITH_BFP arm rejects a non-BFP wire. */
void testReluBackwardBfpArmRejectsNonBfpWire(void) {
    reluConfig_t cfg = {0};
    cfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t dims[] = {1, 8};
    tensor_t *bfpWire = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, NULL, NULL);
    tensor_t *otherBfp = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, NULL, NULL);
    tensor_t *floatWire = buildFloatTensor1D(8);

    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, floatWire, bfpWire, otherBfp));
    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, bfpWire, floatWire, otherBfp));
    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, bfpWire, otherBfp, floatWire));

    freeTensor(floatWire);
    freeTensor(otherBfp);
    freeTensor(bfpWire);
}

/* BFP epic PR4: the length gate on ALL THREE wires. Every tensor carries the
 * per-tensor sentinel {1, 0} and identical widths, so the grid comparison is
 * field-for-field equal and only the explicit count check separates them. One
 * assertion per wire, so a gate that covers just the loss/propLoss pair (and
 * lets a short forwardInput through) still fails this test. */
void testReluBackwardBfpArmRejectsUnequalElementCounts(void) {
    reluConfig_t cfg = {0};
    cfg.propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerConfig_t lc = {.relu = &cfg};
    layer_t layer = {.type = RELU, .config = &lc};
    size_t longDims[] = {1, 8};
    size_t shortDims[] = {1, 4};
    tensor_t *longA = buildBfpWireWithCodes(longDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *longB = buildBfpWireWithCodes(longDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *longC = buildBfpWireWithCodes(longDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *shortWire = buildBfpWireWithCodes(shortDims, 2, 6, 8, 1, 0, NULL, NULL);

    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, shortWire, longB, longC));
    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, longA, shortWire, longC));
    ASSERT_EXITS_WITH_FAILURE(reluBackward(&layer, longA, longB, shortWire));

    freeTensor(shortWire);
    freeTensor(longC);
    freeTensor(longB);
    freeTensor(longA);
}

/* #315: reluBackward dispatches on the layer's DECLARED propLossMath and
 * raw-casts the wire data pointers without checking the wires' ACTUAL dtype. A
 * FLOAT32 arm fed SYM_INT32 wires reads int mantissa codes as floats — silent
 * garbage grads that propagate with no diagnostic (the SYM arm on FLOAT32 wires
 * NULL-derefs qConfig instead). Guard the wire dtypes and fail fast, mirroring
 * the LayerNorm/GroupNorm backward guards. */
void testReluBackwardExitsOnDtypeMismatch() {
    symInt32QConfig_t qc;
    initSymInt32QConfig(HALF_AWAY, &qc);
    qc.scale = 1.f;
    quantization_t symQ;
    initSymInt32Quantization(&qc, &symQ);

    size_t dims[] = {6};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t fwdData[] = {-1, 0, 1, 2, 5, -6};
    tensor_t forwardInput;
    setTensorValues(&forwardInput, (uint8_t *)fwdData, &shape, &symQ, NULL);

    int32_t lossData[] = {0, 2, -4, 6, 3, 2};
    tensor_t loss;
    setTensorValues(&loss, (uint8_t *)lossData, &shape, &symQ, NULL);

    int32_t propLossData[6] = {0};
    tensor_t propLoss;
    setTensorValues(&propLoss, (uint8_t *)propLossData, &shape, &symQ, NULL);

    /* FLOAT32-declared layer (propLossMath = ARITH_FLOAT32) fed SYM_INT32 wires. */
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *reluLayer = reluLayerInit(&lq);
    layerFunctions_t reluFns = layerFunctions[RELU];

    ASSERT_EXITS_WITH_FAILURE(reluFns.backward(reluLayer, &forwardInput, &loss, &propLoss));

    /* BFP epic PR4: the same FLOAT32-declared arm fed a BFP-STORED wire. The
     * blanket pre-dispatch reject is gone, so this now exercises the per-arm
     * #315 guard on a packed wire. */
    tensor_t *bfpWire = buildBfpTensor1D(6);
    tensor_t *floatWire = buildFloatTensor1D(6);
    ASSERT_EXITS_WITH_FAILURE(reluFns.backward(reluLayer, bfpWire, floatWire, floatWire));

    freeTensor(floatWire);
    freeTensor(bfpWire);
    freeReluLayer(reluLayer);
    freeQuantization(floatQ);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testReluForwardFloat);
    RUN_TEST(testReluForwardSymInt32);

    RUN_TEST(testReluBackwardFloat);
    RUN_TEST(testReluBackwardSymInt32);
    RUN_TEST(testReluBackwardExitsOnDtypeMismatch);
    RUN_TEST(testReluForwardBfpClampsCodesAndCarriesExponents);
    RUN_TEST(testReluForwardBfpArmDispatchesThroughReluForward);
    RUN_TEST(testReluForwardFloat32ArmRejectsBfpWire);
    RUN_TEST(testReluForwardBfpArmRejectsNonBfpWireAndGeometryMismatch);
    RUN_TEST(testReluForwardBfpArmRejectsUnequalElementCounts);
    RUN_TEST(testReluBackwardBfpMasksBySignAndCarriesExponents);
    RUN_TEST(testReluBackwardBfpArmDispatchesThroughReluBackward);
    RUN_TEST(testReluBackwardBfpArmRejectsNonBfpWire);
    RUN_TEST(testReluBackwardBfpArmRejectsUnequalElementCounts);

    RUN_TEST(testReluLayerInitAndFreeRoundTrip);

    RUN_TEST(testReluLayerInitBorrowingStoresLqPointers);
    RUN_TEST(testReluLayerInitOwningDeepCopiesLqPointers);
    return UNITY_END();
}
