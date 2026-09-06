#define SOURCE_FILE "UNIT_TEST_DROPOUT"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "Bernoulli.h"
#include "DeathTest.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_bfp_dropout.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- shared builders ---- */

static tensor_t *buildFloatTensor(size_t n, const float *vals) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, n);
    }
    return t;
}

static tensor_t *buildBoolMask(size_t n) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

/* Wraps a dropoutConfig in a stack layer_t for direct forward/backward calls. */
static layer_t makeDropoutLayer(dropoutConfig_t *dcfg, layerConfig_t *lcfg) {
    lcfg->dropout = dcfg;
    layer_t layer = {.type = DROPOUT, .config = lcfg};
    return layer;
}

/* BFP epic PR4: build a BFP wire with EXACT codes and per-group exponents —
 * writing the packed payload directly keeps the fixture independent of the
 * quantizer (local copy of Task 1's helper, the per-file test-helper idiom). */
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

void testForwardEvalIdentityFloat(void) {
    size_t n = 4;
    float in[] = {1.f, -2.f, 3.f, -4.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq); // training defaults to false
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutForward(&layer, input, output);

    float captured[4];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(in, captured, n);
}

void testForwardEvalIdentitySymInt32(void) {
    size_t n = 4;

    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *symIn = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    int32_t inInts[] = {10, -20, 30, -40};
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)symIn->data)[i] = inInts[i];
    }
    ((symInt32QConfig_t *)symIn->quantization->qConfig)->scale = 0.1f;

    size_t *odims = reserveMemory(sizeof(size_t));
    odims[0] = n;
    size_t *oorder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, oorder);
    shape_t *oshape = reserveMemory(sizeof(shape_t));
    setShape(oshape, odims, 1, oorder);
    tensor_t *symOut = initTensor(oshape, quantizationInitSymInt32(HALF_AWAY), NULL);

    tensor_t *mask = buildBoolMask(n);
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq); // eval mode
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutForward(&layer, symIn, symOut);

    int32_t capturedInts[4];
    for (size_t i = 0; i < n; i++) {
        capturedInts[i] = ((int32_t *)symOut->data)[i];
    }
    float outScale = ((symInt32QConfig_t *)symOut->quantization->qConfig)->scale;

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(symOut);
    freeTensor(symIn);

    TEST_ASSERT_EQUAL_INT32_ARRAY(inInts, capturedInts, n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.1f, outScale);
}

/* Stub sampler: keep even indices (bit 1), drop odd (bit 0). Ignores probTrue. */
static void stubKeepEven(tensor_t *mask, float probTrue) {
    (void)probTrue;
    size_t n = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < n; i++) {
        tensorBoolSet(mask, i, (i % 2) == 0);
    }
}

void testForwardTrainingFloatScalesAndDrops(void) {
    size_t n = 4;
    float in[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    dropoutForward(&layer, input, output);
    bernoulliSetFillMaskFn(saved);

    float captured[4];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    // s = 1/(1-0.5) = 2; keep even idx (×2), drop odd (→0).
    float expected[] = {2.f, 0.f, 6.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, n);
}

void testForwardTrainingSymInt32ScaleFold(void) {
    size_t n = 4;
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *symIn = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    int32_t inInts[] = {10, -20, 30, 40};
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)symIn->data)[i] = inInts[i];
    }
    ((symInt32QConfig_t *)symIn->quantization->qConfig)->scale = 0.1f;

    size_t *odims = reserveMemory(sizeof(size_t));
    odims[0] = n;
    size_t *oorder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, oorder);
    shape_t *oshape = reserveMemory(sizeof(shape_t));
    setShape(oshape, odims, 1, oorder);
    tensor_t *symOut = initTensor(oshape, quantizationInitSymInt32(HALF_AWAY), NULL);

    tensor_t *mask = buildBoolMask(n);
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    dropoutForward(&layer, symIn, symOut);
    bernoulliSetFillMaskFn(saved);

    int32_t capturedInts[4];
    for (size_t i = 0; i < n; i++) {
        capturedInts[i] = ((int32_t *)symOut->data)[i];
    }
    float outScale = ((symInt32QConfig_t *)symOut->quantization->qConfig)->scale;

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(symOut);
    freeTensor(symIn);

    // keep even idx → int copied, drop odd → 0; scale folded ×2 (s=1/(1-0.5)).
    int32_t expectedInts[] = {10, 0, 30, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedInts, capturedInts, n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.2f, outScale);
}

/* BFP epic PR4 (R-P3, spec D4 + deviation 5): Dropout is NON-NATIVE by
 * decision -- 1/(1-p) is not a power of two, so unlike SYM there is no single
 * scale to fold it into. The BFP arm is a float bridge that re-derives every
 * group's exponent. Gold from generate_expected_bfp_dropout.py; the mask is
 * the deterministic stubKeepEven, so no RNG enters the assertion. */
void testDropoutForwardTrainingBfpBridgeRepacksWithFreshExponents(void) {
    size_t dims[] = {1, kBfpDropoutInCodes_len};
    int32_t inCodes[8];
    for (size_t i = 0; i < kBfpDropoutInCodes_len; i++) {
        inCodes[i] = kBfpDropoutInCodes[i];
    }
    int32_t sentinel[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t zeroState[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, inCodes, kBfpDropoutInExps);
    tensor_t *output = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, sentinel, zeroState);
    tensor_t *mask = buildBoolMask(kBfpDropoutInCodes_len);

    quantization_t *fq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    quantization_t *bq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    dropoutForward(&layer, input, output);
    bernoulliSetFillMaskFn(saved);

    int32_t got[8];
    unpackSignExtend(output->data, (uint8_t)kBfpDropoutMantissaBits, 0, got,
                     kBfpDropoutOutCodes_len);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpDropoutOutCodes, got, kBfpDropoutOutCodes_len,
                                          "masked+scaled codes must match the goldgen");
    bfpQConfig_t *outQC = output->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutOutExps[0], outQC->exponents[0],
                                    "group 0 exponent must be RE-DERIVED, not copied");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutOutExps[1], outQC->exponents[1],
                                    "group 1 exponent must be RE-DERIVED, not copied");

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);
}

/* Eval mode is the exponent-verbatim copy of R-P2's Relu forward: no mask, no
 * factor, so nothing is re-derived. */
void testDropoutForwardEvalIdentityBfp(void) {
    size_t dims[] = {1, kBfpDropoutInCodes_len};
    int32_t inCodes[8];
    for (size_t i = 0; i < kBfpDropoutInCodes_len; i++) {
        inCodes[i] = kBfpDropoutInCodes[i];
    }
    int32_t sentinel[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t zeroState[2] = {127, 127};
    tensor_t *input = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, inCodes, kBfpDropoutInExps);
    tensor_t *output = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, sentinel, zeroState);
    tensor_t *mask = buildBoolMask(kBfpDropoutInCodes_len);

    quantization_t *fq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    quantization_t *bq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = false;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutForward(&layer, input, output);

    int32_t got[8];
    unpackSignExtend(output->data, (uint8_t)kBfpDropoutMantissaBits, 0, got,
                     kBfpDropoutInCodes_len);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpDropoutInCodes, got, kBfpDropoutInCodes_len,
                                          "eval mode must copy the codes verbatim");
    bfpQConfig_t *outQC = output->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutInExps[0], outQC->exponents[0],
                                    "eval mode must copy the exponents verbatim");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutInExps[1], outQC->exponents[1],
                                    "eval mode must copy the exponents verbatim");

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);
}

/* BFP epic PR4 (wave-1 review M4): eval-mode Dropout IN PLACE — the same wire
 * passed as input and output. It is constructible through the public entry
 * (requireBfpPairForArm compares a wire's geometry with its own and admits it),
 * so the verbatim carry must handle it: identity, no diagnostic, and above all
 * no memcpy(p, p, n), whose restrict-qualified parameters make the self-copy
 * formally undefined. This pins the CONTRACT (admitted + identity), not the UB:
 * a future implementer who copies the bridge's alias REJECT into the verbatim
 * path would turn a legal call into an exit(1), and this test is what stops
 * that. */
void testDropoutForwardEvalInPlaceBfpIsIdentity(void) {
    size_t dims[] = {1, kBfpDropoutInCodes_len};
    int32_t inCodes[8];
    for (size_t i = 0; i < kBfpDropoutInCodes_len; i++) {
        inCodes[i] = kBfpDropoutInCodes[i];
    }
    tensor_t *wire = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, inCodes, kBfpDropoutInExps);
    tensor_t *mask = buildBoolMask(kBfpDropoutInCodes_len);

    quantization_t *fq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, fq);
    dcfg.training = false;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutForward(&layer, wire, wire);

    int32_t got[8];
    unpackSignExtend(wire->data, (uint8_t)kBfpDropoutMantissaBits, 0, got, kBfpDropoutInCodes_len);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpDropoutInCodes, got, kBfpDropoutInCodes_len,
                                          "in-place eval must leave the codes untouched");
    bfpQConfig_t *qc = wire->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutInExps[0], qc->exponents[0],
                                    "in-place eval must leave the exponents untouched");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutInExps[1], qc->exponents[1],
                                    "in-place eval must leave the exponents untouched");

    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(wire);
}

static void fillMaskKeepEven(tensor_t *mask) {
    size_t n = calcNumberOfElementsByTensor(mask);
    for (size_t i = 0; i < n; i++) {
        tensorBoolSet(mask, i, (i % 2) == 0);
    }
}

void testBackwardFloatUsesMaskAndScale(void) {
    size_t n = 4;
    tensor_t *forwardInput = buildFloatTensor(n, (float[]){1.f, 1.f, 1.f, 1.f});
    float grad[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *loss = buildFloatTensor(n, grad);
    tensor_t *propLoss = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);
    fillMaskKeepEven(mask); // simulate the mask the forward pass produced

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutBackward(&layer, forwardInput, loss, propLoss);

    float captured[4];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);

    // s=2; kept idx: grad×2, dropped idx: 0.
    float expected[] = {2.f, 0.f, 6.f, 0.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, n);
}

void testBackwardSymInt32UsesMaskAndScaleFold(void) {
    size_t n = 4;

    size_t *ldims = reserveMemory(sizeof(size_t));
    ldims[0] = n;
    size_t *lorder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lorder);
    shape_t *lshape = reserveMemory(sizeof(shape_t));
    setShape(lshape, ldims, 1, lorder);
    tensor_t *loss = initTensor(lshape, quantizationInitSymInt32(HALF_AWAY), NULL);
    int32_t gradInts[] = {5, 6, 7, 8};
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)loss->data)[i] = gradInts[i];
    }
    ((symInt32QConfig_t *)loss->quantization->qConfig)->scale = 0.2f;

    size_t *pdims = reserveMemory(sizeof(size_t));
    pdims[0] = n;
    size_t *porder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, porder);
    shape_t *pshape = reserveMemory(sizeof(shape_t));
    setShape(pshape, pdims, 1, porder);
    tensor_t *propLoss = initTensor(pshape, quantizationInitSymInt32(HALF_AWAY), NULL);

    tensor_t *mask = buildBoolMask(n);
    fillMaskKeepEven(mask);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutBackward(&layer, NULL, loss, propLoss);

    int32_t capturedInts[4];
    for (size_t i = 0; i < n; i++) {
        capturedInts[i] = ((int32_t *)propLoss->data)[i];
    }
    float outScale = ((symInt32QConfig_t *)propLoss->quantization->qConfig)->scale;

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(propLoss);
    freeTensor(loss);

    int32_t expectedInts[] = {5, 0, 7, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedInts, capturedInts, n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.4f, outScale); // 0.2 × 2
}

void testVtableForwardIdentityFloat(void) {
    size_t n = 3;
    float in[] = {7.f, 8.f, 9.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq); // eval
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    layerFunctions_t fns = layerFunctions[DROPOUT];
    fns.forward(&layer, input, output);

    float captured[3];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(in, captured, n);
}

void testCalcOutputShapeIsIdentity(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 5;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *inShape = reserveMemory(sizeof(shape_t));
    setShape(inShape, dims, 2, order);

    size_t *odims = reserveMemory(2 * sizeof(size_t));
    size_t *oorder = reserveMemory(2 * sizeof(size_t));
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    outShape->dimensions = odims;
    outShape->orderOfDimensions = oorder;
    outShape->numberOfDimensions = 0;

    tensor_t *mask = buildBoolMask(10);
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutCalcOutputShape(&layer, inShape, outShape);

    size_t nd = outShape->numberOfDimensions;
    size_t d0 = outShape->dimensions[0];
    size_t d1 = outShape->dimensions[1];

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeReservedMemory(outShape);
    freeReservedMemory(oorder);
    freeReservedMemory(odims);
    freeReservedMemory(inShape);
    freeReservedMemory(order);
    freeReservedMemory(dims);

    TEST_ASSERT_EQUAL_UINT(2, nd);
    TEST_ASSERT_EQUAL_UINT(2, d0);
    TEST_ASSERT_EQUAL_UINT(5, d1);
}

void testFactoryBuildsAndForwards(void) {
    size_t n = 3;
    float in[] = {4.f, 5.f, 6.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    layer_t *layer = dropoutLayerInit(0.5f, mask, fq, bq);
    bool typeOk = (layer->type == DROPOUT);
    float pOk = layer->config->dropout->p;
    bool trainingDefaultFalse = (layer->config->dropout->training == false);

    layerFunctions[DROPOUT].forward(layer, input, output); // eval → identity
    float captured[3];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeDropoutLayer(layer);
    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    TEST_ASSERT_TRUE(typeOk);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, pOk);
    TEST_ASSERT_TRUE(trainingDefaultFalse);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(in, captured, n);
}

void testForwardTrainingP0KeepsAllNoScale(void) {
    size_t n = 4;
    float in[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.0f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    // p=0 → reference sampler fills probTrue=1.0 → all keep; scale=1/(1-0)=1 → identity.
    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(bernoulliFillMaskReference);
    dropoutForward(&layer, input, output);
    bernoulliSetFillMaskFn(saved);

    float captured[4];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }
    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(in, captured, n);
}

void testForwardTrainingFloatP025Scale(void) {
    size_t n = 4;
    float in[] = {4.f, 8.f, 12.f, 16.f};
    tensor_t *input = buildFloatTensor(n, in);
    tensor_t *output = buildFloatTensor(n, NULL);
    tensor_t *mask = buildBoolMask(n);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.25f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    bernoulliFillMaskFn_t saved = bernoulliGetFillMaskFn();
    bernoulliSetFillMaskFn(stubKeepEven);
    dropoutForward(&layer, input, output);
    bernoulliSetFillMaskFn(saved);

    float captured[4];
    for (size_t i = 0; i < n; i++) {
        captured[i] = ((float *)output->data)[i];
    }
    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(output);
    freeTensor(input);

    // s = 1/(1-0.25) = 1.33333; keep even idx (×s), drop odd → 0.
    float expected[] = {5.33333f, 0.f, 16.0f, 0.f};
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected[i], captured[i]);
    }
}

/* #315: dropoutBackward dispatches on the layer's DECLARED propLossMath and
 * raw-casts loss/propLoss data pointers (forwardInput is unused). A FLOAT32 arm
 * fed SYM_INT32 wires reads int mantissa codes as floats — silent garbage grads
 * (the SYM arm on FLOAT32 wires NULL-derefs qConfig). Guard the dereferenced
 * wire dtypes and fail fast, mirroring the LayerNorm/GroupNorm backward guards. */
void testDropoutBackwardExitsOnDtypeMismatch(void) {
    size_t n = 4;

    size_t *ldims = reserveMemory(sizeof(size_t));
    ldims[0] = n;
    size_t *lorder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, lorder);
    shape_t *lshape = reserveMemory(sizeof(shape_t));
    setShape(lshape, ldims, 1, lorder);
    tensor_t *loss = initTensor(lshape, quantizationInitSymInt32(HALF_AWAY), NULL);

    size_t *pdims = reserveMemory(sizeof(size_t));
    pdims[0] = n;
    size_t *porder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, porder);
    shape_t *pshape = reserveMemory(sizeof(shape_t));
    setShape(pshape, pdims, 1, porder);
    tensor_t *propLoss = initTensor(pshape, quantizationInitSymInt32(HALF_AWAY), NULL);

    tensor_t *mask = buildBoolMask(n);
    fillMaskKeepEven(mask);

    /* FLOAT32-declared dropout (propLossMath = ARITH_FLOAT32) fed SYM_INT32 wires. */
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(dropoutBackward(&layer, NULL, loss, propLoss));

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(propLoss);
    freeTensor(loss);
}

/* BFP epic PR4: the POSITIVE pin on dropoutBackward's ARITH_BFP arm. Both
 * backward death tests below expect exit(1), so a DELETED `case ARITH_BFP:`
 * (which falls through to `default: PRINT_ERROR(...); exit(1)`) satisfies them
 * just as well as a present one — only an assertion on the produced codes can
 * tell the two apart. The backward is dropoutMaskScaleBfp on the loss wire, so
 * it reuses the forward's gold unchanged: same mask (pre-filled here, since the
 * backward never draws), same factor, same fresh per-group derive. */
void testDropoutBackwardBfpArmDispatchesThroughDropoutBackward(void) {
    size_t dims[] = {1, kBfpDropoutInCodes_len};
    int32_t lossCodes[8];
    for (size_t i = 0; i < kBfpDropoutInCodes_len; i++) {
        lossCodes[i] = kBfpDropoutInCodes[i];
    }
    int32_t sentinel[8] = {-9, -9, -9, -9, -9, -9, -9, -9};
    uint8_t zeroState[2] = {127, 127};
    tensor_t *loss = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, lossCodes, kBfpDropoutInExps);
    tensor_t *propLoss = buildBfpWireWithCodes(
        dims, 2, (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize, sentinel, zeroState);
    tensor_t *mask = buildBoolMask(kBfpDropoutInCodes_len);
    fillMaskKeepEven(mask); /* the pattern the forward would have produced */

    quantization_t *fq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    quantization_t *bq = quantizationInitBfpGrouped(
        (uint8_t)kBfpDropoutMantissaBits, (uint8_t)kBfpDropoutExponentBits, HALF_AWAY,
        (size_t)kBfpDropoutNumGroups, (size_t)kBfpDropoutGroupSize);
    dropoutConfig_t dcfg;
    initDropoutConfig(&dcfg, 0.5f, mask, fq, bq);
    dcfg.training = true;
    layerConfig_t lcfg;
    layer_t layer = makeDropoutLayer(&dcfg, &lcfg);

    dropoutBackward(&layer, NULL, loss, propLoss);

    int32_t got[8];
    unpackSignExtend(propLoss->data, (uint8_t)kBfpDropoutMantissaBits, 0, got,
                     kBfpDropoutOutCodes_len);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(kBfpDropoutOutCodes, got, kBfpDropoutOutCodes_len,
                                          "masked+scaled grad codes must match the goldgen");
    bfpQConfig_t *propQC = propLoss->quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutOutExps[0], propQC->exponents[0],
                                    "group 0 exponent must be RE-DERIVED, not copied");
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(kBfpDropoutOutExps[1], propQC->exponents[1],
                                    "group 1 exponent must be RE-DERIVED, not copied");

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(propLoss);
    freeTensor(loss);
}

/* BFP epic PR4 (R-P7d): FLOAT32/SYM arms still reject a BFP wire (#315 parity);
 * the ARITH_BFP arm rejects a non-BFP wire and a differing block grid. */
void testDropoutBfpGuardsNarrowedNotRemoved(void) {
    size_t dims[] = {1, 8};
    tensor_t *bfpA = buildBfpWireWithCodes(dims, 2, 6, 8, 2, 4, NULL, NULL);
    tensor_t *otherGrid = buildBfpWireWithCodes(dims, 2, 6, 8, 4, 2, NULL, NULL);
    tensor_t *floatWire = buildFloatTensor(8, NULL);
    tensor_t *mask = buildBoolMask(8);
    quantization_t *fq = quantizationInitFloat();
    dropoutConfig_t floatCfg;
    initDropoutConfig(&floatCfg, 0.5f, mask, fq, fq);
    layerConfig_t floatLc;
    layer_t floatLayer = makeDropoutLayer(&floatCfg, &floatLc);

    ASSERT_EXITS_WITH_FAILURE(dropoutForward(&floatLayer, bfpA, floatWire));
    ASSERT_EXITS_WITH_FAILURE(dropoutBackward(&floatLayer, NULL, bfpA, floatWire));

    quantization_t *bq = quantizationInitBfpGrouped(6, 8, HALF_AWAY, 2, 4);
    dropoutConfig_t bfpCfg;
    initDropoutConfig(&bfpCfg, 0.5f, mask, bq, bq);
    layerConfig_t bfpLc;
    layer_t bfpLayer = makeDropoutLayer(&bfpCfg, &bfpLc);

    ASSERT_EXITS_WITH_FAILURE(dropoutForward(&bfpLayer, floatWire, bfpA));
    ASSERT_EXITS_WITH_FAILURE(dropoutForward(&bfpLayer, bfpA, otherGrid));
    ASSERT_EXITS_WITH_FAILURE(dropoutBackward(&bfpLayer, NULL, bfpA, otherGrid));

    freeQuantization(bq);
    freeQuantization(fq);
    freeTensor(mask);
    freeTensor(floatWire);
    freeTensor(otherGrid);
    freeTensor(bfpA);
}

/* BFP epic PR4: the two hazards the grid comparison cannot see. (a) Unequal
 * element counts under two per-tensor {1, 0} grids, which compare equal
 * field-for-field. (b) A SHARED exponent array between two distinct payloads:
 * pass 1 writes the fresh stored exponents into it, so pass 2's decode of the
 * source at its "original" grid reads pass 1's output instead. Constructing
 * (b) needs the two tensors to share ONE quantization_t, which is exactly what
 * a caller that reuses the config for both wires produces. */
void testDropoutBfpRejectsUnequalCountsAndAliasedExponents(void) {
    size_t longDims[] = {1, 8};
    size_t shortDims[] = {1, 4};
    tensor_t *longWire = buildBfpWireWithCodes(longDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *shortWire = buildBfpWireWithCodes(shortDims, 2, 6, 8, 1, 0, NULL, NULL);
    tensor_t *mask = buildBoolMask(8);
    quantization_t *perTensorQ = quantizationInitBfp(6, 8, HALF_AWAY);
    dropoutConfig_t cfg;
    initDropoutConfig(&cfg, 0.5f, mask, perTensorQ, perTensorQ);
    layerConfig_t lc;
    layer_t layer = makeDropoutLayer(&cfg, &lc);

    ASSERT_EXITS_WITH_FAILURE(dropoutForward(&layer, longWire, shortWire));
    ASSERT_EXITS_WITH_FAILURE(dropoutBackward(&layer, NULL, longWire, shortWire));

    /* Two distinct payload buffers, ONE shared quantization_t (hence one
     * exponents array): counts and grid both match, only the alias is wrong. */
    size_t *sharedDims = reserveMemory(2 * sizeof(size_t));
    sharedDims[0] = 1;
    sharedDims[1] = 8;
    size_t *sharedOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, sharedOrder);
    shape_t *sharedShape = reserveMemory(sizeof(shape_t));
    setShape(sharedShape, sharedDims, 2, sharedOrder);
    quantization_t *aliasQ = quantizationInitBfp(6, 8, HALF_AWAY);
    tensor_t *aliasSrc = initTensor(sharedShape, aliasQ, NULL);
    size_t *dstDims = reserveMemory(2 * sizeof(size_t));
    dstDims[0] = 1;
    dstDims[1] = 8;
    size_t *dstOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, dstOrder);
    shape_t *dstShape = reserveMemory(sizeof(shape_t));
    setShape(dstShape, dstDims, 2, dstOrder);
    tensor_t *aliasDst = initTensor(dstShape, aliasQ, NULL); /* SAME quantization_t */
    dropoutConfig_t aliasCfg;
    initDropoutConfig(&aliasCfg, 0.5f, mask, aliasQ, aliasQ);
    aliasCfg.training = true; /* eval mode would take the verbatim carry, which
                               * has no alias check and no two-pass hazard */
    layerConfig_t aliasLc;
    layer_t aliasLayer = makeDropoutLayer(&aliasCfg, &aliasLc);

    TEST_ASSERT_NOT_EQUAL_MESSAGE(aliasSrc->data, aliasDst->data,
                                  "the alias fixture must differ in payload, not in qConfig");
    ASSERT_EXITS_WITH_FAILURE(dropoutForward(&aliasLayer, aliasSrc, aliasDst));

    /* UnitTestDropout.c has no shell-only free helper and the plan forbids
     * inventing one: free aliasDst's own blocks by hand and let
     * freeTensor(aliasSrc) take the shared aliasQ exactly once. */
    freeReservedMemory(aliasDst->data);
    freeShape(aliasDst->shape);
    freeReservedMemory(aliasDst);
    freeTensor(aliasSrc);
    freeQuantization(perTensorQ);
    freeTensor(mask);
    freeTensor(shortWire);
    freeTensor(longWire);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testForwardEvalIdentityFloat);
    RUN_TEST(testForwardEvalIdentitySymInt32);
    RUN_TEST(testForwardTrainingFloatScalesAndDrops);
    RUN_TEST(testForwardTrainingSymInt32ScaleFold);
    RUN_TEST(testBackwardFloatUsesMaskAndScale);
    RUN_TEST(testBackwardSymInt32UsesMaskAndScaleFold);
    RUN_TEST(testDropoutBackwardExitsOnDtypeMismatch);
    RUN_TEST(testDropoutForwardTrainingBfpBridgeRepacksWithFreshExponents);
    RUN_TEST(testDropoutForwardEvalIdentityBfp);
    RUN_TEST(testDropoutForwardEvalInPlaceBfpIsIdentity);
    RUN_TEST(testDropoutBackwardBfpArmDispatchesThroughDropoutBackward);
    RUN_TEST(testDropoutBfpGuardsNarrowedNotRemoved);
    RUN_TEST(testDropoutBfpRejectsUnequalCountsAndAliasedExponents);
    RUN_TEST(testVtableForwardIdentityFloat);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testFactoryBuildsAndForwards);
    RUN_TEST(testForwardTrainingP0KeepsAllNoScale);
    RUN_TEST(testForwardTrainingFloatP025Scale);
    return UNITY_END();
}
