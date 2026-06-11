#define SOURCE_FILE "UNIT_TEST_DROPOUT"

#include <stdbool.h>
#include <stdlib.h>

#include "Bernoulli.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "Layer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testForwardEvalIdentityFloat);
    RUN_TEST(testForwardEvalIdentitySymInt32);
    RUN_TEST(testForwardTrainingFloatScalesAndDrops);
    RUN_TEST(testForwardTrainingSymInt32ScaleFold);
    RUN_TEST(testBackwardFloatUsesMaskAndScale);
    RUN_TEST(testBackwardSymInt32UsesMaskAndScaleFold);
    RUN_TEST(testVtableForwardIdentityFloat);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testFactoryBuildsAndForwards);
    RUN_TEST(testForwardTrainingP0KeepsAllNoScale);
    RUN_TEST(testForwardTrainingFloatP025Scale);
    return UNITY_END();
}
