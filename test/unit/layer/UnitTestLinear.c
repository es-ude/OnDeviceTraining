#include <math.h>
#include <stddef.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BfpAccumulateRef.h"
#include "BfpRawFixture.h"
#include "BfpScatterRef.h"
#include "BorrowedLayer.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerExec.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Rounding.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "SymAccumGuard.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_bfp_layer_forward.h"
#include "expected_linear_grouped_asym.h"
#include "unity.h"

/* The init recipe carries the layer's init key, appended last so that a
 * zero-initialised recipe leaves it NULL; the bias takes the same key. */
_Static_assert(_Generic(((weightInit_t){0}).key, const rngKey_t *: 1, default: 0),
               "weightInit_t.key must be a const rngKey_t *");
_Static_assert(offsetof(weightInit_t, key) > offsetof(weightInit_t, gain),
               "weightInit_t.key must be the last field");
_Static_assert(_Generic((&initBiasTensor),
                   void (*)(tensor_t *, size_t, const rngKey_t *): 1,
                   default: 0),
               "initBiasTensor must take (tensor_t *, size_t, const rngKey_t *)");

void testLinearForwardFloatRank1BiasRank2Output() {
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 2;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});

    linearForward(linearLayer, input, output, NULL);

    float captured[2];
    captured[0] = ((float *)output->data)[0];
    captured[1] = ((float *)output->data)[1];

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(testQ);

    float expected[] = {-5.f, -4.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, 2);
}

void testLinearForwardSymInt32Rank1BiasRank2Output() {
    size_t numberOfOutputs = 2;

    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    parameter_t *weights = parameterInit(weightsParam, NULL);

    /* RANK-1 sym bias [2]. */
    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 2;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearForward(linearLayer, input, output, NULL);

    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 1;
    outFloatDims[1] = 2;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat, NULL);

    float captured[2];
    for (size_t i = 0; i < numberOfOutputs; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    freeTensor(outputFloat);
    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(test);

    float expected[] = {-5.f, -4.f};
    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testLinearBackwardFloatRank1Bias() {
    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    size_t numWeightElements = calcNumberOfElementsByShape(cfg->weights->param->shape);
    size_t numBiasElements = calcNumberOfElementsByShape(cfg->bias->param->shape);
    size_t numPropLossElements = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < numWeightElements; i++) {
        capturedWeightGrad[i] = ((float *)cfg->weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numBiasElements; i++) {
        capturedBiasGrad[i] = ((float *)cfg->bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numPropLossElements; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(testQ);

    float expected_weight_grad[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float expected_bias_grad[] = {-4.f, -3.f};
    float expected_propagated_loss[] = {-8.f, -23.f, 30.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_weight_grad, capturedWeightGrad, numWeightElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_propagated_loss, capturedPropLoss, numPropLossElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_bias_grad, capturedBiasGrad, numBiasElements);
}

void testLinearBackwardSymInt32Rank1Bias() {
    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfForwardInputs = 3;

    /* 1. Build heap weights parameter (SymInt32, shape 2x3) with grad. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitSymInt32(weightsParam, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* 2. Build heap bias parameter (SymInt32, RANK-1 shape [2]) with grad. */
    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitSymInt32(biasParam, HALF_AWAY, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    /* 3. Build heap forwardInput tensor (SymInt32, shape 1x3). */
    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap loss tensor (SymInt32, shape 1x2). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    /* 5. Build heap propLoss tensor (SymInt32, shape 1x3). */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = numberOfForwardInputs;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 6. Build layer (shared SymInt32 quantization). */
    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    /* 7. Convert SymInt32 grads back to Float for comparison. */
    size_t *wgDims = reserveMemory(2 * sizeof(size_t));
    wgDims[0] = 2;
    wgDims[1] = 3;
    size_t *wgOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wgOrder);
    shape_t *wgShape = reserveMemory(sizeof(shape_t));
    setShape(wgShape, wgDims, 2, wgOrder);
    tensor_t *weightGradFloat = initTensor(wgShape, quantizationInitFloat(), NULL);
    convertTensor(weights->grad, weightGradFloat, NULL);

    /* RANK-1 bias-grad convert-back block. */
    size_t *bgDims = reserveMemory(1 * sizeof(size_t));
    bgDims[0] = 2;
    size_t *bgOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bgOrder);
    shape_t *bgShape = reserveMemory(sizeof(shape_t));
    setShape(bgShape, bgDims, 1, bgOrder);
    tensor_t *biasGradFloat = initTensor(bgShape, quantizationInitFloat(), NULL);
    convertTensor(bias->grad, biasGradFloat, NULL);

    size_t *plDims = reserveMemory(2 * sizeof(size_t));
    plDims[0] = 1;
    plDims[1] = numberOfForwardInputs;
    size_t *plOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, plOrder);
    shape_t *plShape = reserveMemory(sizeof(shape_t));
    setShape(plShape, plDims, 2, plOrder);
    tensor_t *propLossFloat = initTensor(plShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat, NULL);

    /* 8. CAPTURE. */
    float capturedWeightGrad[6];
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedWeightGrad[i] = ((float *)weightGradFloat->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedBiasGrad[i] = ((float *)biasGradFloat->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numberOfForwardInputs; i++) {
        capturedPropLoss[i] = ((float *)propLossFloat->data)[i];
    }

    /* 9. FREE. */
    freeTensor(propLossFloat);
    freeTensor(biasGradFloat);
    freeTensor(weightGradFloat);
    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(test);

    /* 10. ASSERT. */
    float expectedWeightGrads[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    for (size_t i = 0; i < numberOfWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrads[i], capturedWeightGrad[i]);
    }

    float expectedBiasGrads[] = {-4.f, -3.f};
    for (size_t i = 0; i < numberOfBiases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrads[i], capturedBiasGrad[i]);
    }

    float expectedPropagatedLoss[] = {-8.f, -23.f, 30.f};
    for (size_t i = 0; i < numberOfForwardInputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(.2f, expectedPropagatedLoss[i], capturedPropLoss[i]);
    }
}

/* #380 PR1 Task 1: create-time trainable knob (trainable_t). */
static layer_t *buildFloatLinearWithTrainable(trainable_t trainable) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .trainable = trainable, .weightInit.key = &initKey},
        &lq);
    freeQuantization(q);
    return layer;
}

void testLinearFactoryFrozenElidesGrads(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    linearConfig_t *cfg = layer->config->linear;
    bool weightsGradNull = cfg->weights->grad == NULL;
    bool biasGradNull = cfg->bias->grad == NULL;
    bool frozen = layerIsFrozen(layer);
    freeLinearLayer(layer);
    TEST_ASSERT_TRUE(weightsGradNull);
    TEST_ASSERT_TRUE(biasGradNull);
    TEST_ASSERT_TRUE(frozen);
}

void testLinearFactoryDefaultAllocatesGrads(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT);
    linearConfig_t *cfg = layer->config->linear;
    bool weightsGradPresent = cfg->weights->grad != NULL;
    bool biasGradPresent = cfg->bias->grad != NULL;
    bool frozen = layerIsFrozen(layer);
    freeLinearLayer(layer);
    TEST_ASSERT_TRUE(weightsGradPresent);
    TEST_ASSERT_TRUE(biasGradPresent);
    TEST_ASSERT_FALSE(frozen);
}

void setUp() {}
void tearDown() {}

void testLinearForwardFloat() {
    /* 3. Build heap input tensor (shape 1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap output tensor (shape 1x2). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 2;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    /* 5. Build the layer with shared float quantization. */
    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});

    linearForward(linearLayer, input, output, NULL);

    /* 6. CAPTURE. */
    float captured[2];
    captured[0] = ((float *)output->data)[0];
    captured[1] = ((float *)output->data)[1];

    /* 7. FREE. freeLinearLayer releases everything the factory allocated. */
    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(testQ);

    /* 8. ASSERT. */
    float expected[] = {-5.f, -4.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, 2);
}

void testLinearBackwardFloat() {
    /* 3. Build heap forwardInput tensor, shape 1x3. */
    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap loss tensor, shape 1x2. */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    /* 5. Build heap propLoss tensor, shape 1x3. */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    /* 6. Build the layer. */
    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    /* 7. CAPTURE. */
    size_t numWeightElements = calcNumberOfElementsByShape(cfg->weights->param->shape);
    size_t numBiasElements = calcNumberOfElementsByShape(cfg->bias->param->shape);
    size_t numPropLossElements = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < numWeightElements; i++) {
        capturedWeightGrad[i] = ((float *)cfg->weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numBiasElements; i++) {
        capturedBiasGrad[i] = ((float *)cfg->bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numPropLossElements; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    /* 8. FREE. */
    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(testQ);

    /* 9. ASSERT. */
    float expected_weight_grad[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    float expected_bias_grad[] = {-4.f, -3.f};
    float expected_propagated_loss[] = {-8.f, -23.f, 30.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_weight_grad, capturedWeightGrad, numWeightElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_propagated_loss, capturedPropLoss, numPropLossElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected_bias_grad, capturedBiasGrad, numBiasElements);
}

void testLinearForwardSymInt32() {
    size_t numberOfOutputs = 2;

    /* 1. Build heap weights parameter (SymInt32, shape 2x3) with grad. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitSymInt32(weightsParam, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* 2. Build heap bias parameter (SymInt32, shape 1x2) with grad. */
    size_t *biasDims = reserveMemory(2 * sizeof(size_t));
    biasDims[0] = 1;
    biasDims[1] = 2;
    size_t *biasOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 2, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitSymInt32(biasParam, HALF_AWAY, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    /* 3. Build heap input tensor (SymInt32, shape 1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap output tensor (SymInt32, shape 1x2). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 2;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 5. Build layer (shared SymInt32 quantization). */
    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearForward(linearLayer, input, output, NULL);

    /* 6. Convert SymInt32 output back to Float for comparison. */
    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 1;
    outFloatDims[1] = 2;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat, NULL);

    /* 7. CAPTURE. */
    float captured[2];
    for (size_t i = 0; i < numberOfOutputs; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    /* 8. FREE. */
    freeTensor(outputFloat);
    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(test);

    /* 9. ASSERT. */
    float expected[] = {-5, -4};
    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testLinearBackwardSymInt32() {
    size_t numberOfWeights = 6;
    size_t numberOfBiases = 2;
    size_t numberOfForwardInputs = 3;

    /* 1. Build heap weights parameter (SymInt32, shape 2x3) with grad. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitSymInt32(weightsParam, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* 2. Build heap bias parameter (SymInt32, shape 1x2) with grad. */
    size_t *biasDims = reserveMemory(2 * sizeof(size_t));
    biasDims[0] = 1;
    biasDims[1] = 2;
    size_t *biasOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 2, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitSymInt32(biasParam, HALF_AWAY, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    /* 3. Build heap forwardInput tensor (SymInt32, shape 1x3). */
    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap loss tensor (SymInt32, shape 1x2). */
    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    /* 5. Build heap propLoss tensor (SymInt32, shape 1x3). */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = numberOfForwardInputs;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 6. Build layer (shared SymInt32 quantization). */
    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    /* 7. Convert SymInt32 grads back to Float for comparison. The convert-back
     *    output buffers are heap-allocated to keep us in the heap-tier idiom. */
    size_t *wgDims = reserveMemory(2 * sizeof(size_t));
    wgDims[0] = 2;
    wgDims[1] = 3;
    size_t *wgOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wgOrder);
    shape_t *wgShape = reserveMemory(sizeof(shape_t));
    setShape(wgShape, wgDims, 2, wgOrder);
    tensor_t *weightGradFloat = initTensor(wgShape, quantizationInitFloat(), NULL);
    convertTensor(weights->grad, weightGradFloat, NULL);

    size_t *bgDims = reserveMemory(2 * sizeof(size_t));
    bgDims[0] = 1;
    bgDims[1] = 2;
    size_t *bgOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, bgOrder);
    shape_t *bgShape = reserveMemory(sizeof(shape_t));
    setShape(bgShape, bgDims, 2, bgOrder);
    tensor_t *biasGradFloat = initTensor(bgShape, quantizationInitFloat(), NULL);
    convertTensor(bias->grad, biasGradFloat, NULL);

    size_t *plDims = reserveMemory(2 * sizeof(size_t));
    plDims[0] = 1;
    plDims[1] = numberOfForwardInputs;
    size_t *plOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, plOrder);
    shape_t *plShape = reserveMemory(sizeof(shape_t));
    setShape(plShape, plDims, 2, plOrder);
    tensor_t *propLossFloat = initTensor(plShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat, NULL);

    /* 8. CAPTURE. */
    float capturedWeightGrad[6];
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedWeightGrad[i] = ((float *)weightGradFloat->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedBiasGrad[i] = ((float *)biasGradFloat->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numberOfForwardInputs; i++) {
        capturedPropLoss[i] = ((float *)propLossFloat->data)[i];
    }

    /* 9. FREE. */
    freeTensor(propLossFloat);
    freeTensor(biasGradFloat);
    freeTensor(weightGradFloat);
    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(test);

    /* 10. ASSERT. */
    float expectedWeightGrads[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    for (size_t i = 0; i < numberOfWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrads[i], capturedWeightGrad[i]);
    }

    float expectedBiasGrads[] = {-4.f, -3.f};
    for (size_t i = 0; i < numberOfBiases; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrads[i], capturedBiasGrad[i]);
    }

    float expectedPropagatedLoss[] = {-8.f, -23.f, 30.f};
    for (size_t i = 0; i < numberOfForwardInputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(.2f, expectedPropagatedLoss[i], capturedPropLoss[i]);
    }
}

/* Sign-extends packed SYM mantissas for in-test readback (byteConversion
 * zero-fills on widen); mirrors UnitTestExecuteOp.c's helper of the same
 * name. */
static void symTestUnpackSignExtend(const uint8_t *packed, size_t qBits, int32_t *out, size_t n) {
    byteConversion((uint8_t *)packed, qBits, (uint8_t *)out, 32, n);
    const int32_t signBit = (int32_t)1 << (qBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << qBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = out[i] & mask;
        out[i] = (v ^ signBit) - signBit;
    }
}

/* PR3 Task 4 (packed-grad D1): weightGradAccMode=OUT_ACC_FIXED_SCALE routed to a packed
 * SYM@8 weight-grad target. The freshly-allocated grad (gradInitSym) starts
 * all-zero mantissas at scale=1.0 -- one backward call is therefore the
 * "first store" path (packed-grad D1): the grid is derived from the
 * increment instead of carried. Gate-level asserts only (no pinned floats):
 * SYM@8, nonzero mantissas, a scale that moved off the untouched-default 1.0. */
void testLinearBackwardPackedSymWeightGradFixedScaleFirstStore(void) {
    size_t numberOfWeights = 6;

    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitSym(weightsParam, 8, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    quantization_t *test = quantizationInitFloat();
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);
    /* weightGradAccMode/biasGradAccMode deliberately DISTINCT (mutation
     * check: cross-wiring the weight-grad executeOp call to read
     * biasGradAccMode instead would go undetected if the two fields held the
     * same value). */
    linearLayer->config->linear->weightGradAccMode = OUT_ACC_FIXED_SCALE;
    linearLayer->config->linear->biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE;

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    symQConfig_t *gradQC = (symQConfig_t *)weightsGrad->quantization->qConfig;
    bool gradTypeIsSym = (weightsGrad->quantization->type == SYM);
    uint8_t gradQBits = gradQC->qBits;
    float scaleAfterCall1 = gradQC->scales[0];

    int32_t mant1[6];
    symTestUnpackSignExtend(weightsGrad->data, gradQC->qBits, mant1, numberOfWeights);
    bool anyNonzeroAfterCall1 = false;
    for (size_t i = 0; i < numberOfWeights; i++) {
        if (mant1[i] != 0) {
            anyNonzeroAfterCall1 = true;
        }
    }

    /* Second backward call with the NEGATED loss: FIXED_SCALE must CARRY the
     * grid established by call 1 (packed-grad D1 -- no re-derivation, no renorm),
     * and the exactly-opposite increment drives every mantissa back toward
     * (near-)zero -- safely within the established grid, no overflow risk.
     * If the weight-grad call site were cross-wired to read biasGradAccMode
     * (DYNAMIC_RESCALE) instead, the near-zero recomputed values would force
     * a fresh, much smaller (or absMax==0 -> 1.0) absmax-derived scale --
     * clearly different from the carried scaleAfterCall1. */
    tensor_t *loss2 = initTensor(getShapeLike(loss->shape), quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss2, (float[]){4.f, 3.f}, 2);
    linearBackward(linearLayer, forwardInput, loss2, propLoss, NULL);
    float scaleAfterCall2 = gradQC->scales[0];

    freeTensor(loss2);
    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(test);

    TEST_ASSERT_TRUE_MESSAGE(gradTypeIsSym, "weight grad must stay SYM (packed) after backward");
    TEST_ASSERT_EQUAL_UINT8(8, gradQBits);
    TEST_ASSERT_TRUE_MESSAGE(anyNonzeroAfterCall1,
                             "first-store accumulate must write nonzero mantissas");
    TEST_ASSERT_TRUE_MESSAGE(scaleAfterCall1 > 0.0f, "derived scale must be positive");
    TEST_ASSERT_TRUE_MESSAGE(scaleAfterCall1 != 1.0f,
                             "first-store must derive the grid, not keep the untouched scale=1.0");
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(
        scaleAfterCall1, scaleAfterCall2,
        "FIXED_SCALE must carry the grid across calls (packed-grad D1) -- a scale change here "
        "means the weight-grad call site is reading the wrong accMode field");
}

/* PR3 Task 4 (packed-grad D1) hazard guard: the same fixture as above, but
 * weightGradAccMode is (deliberately) left at its zero-init value -- OUT_WRITE
 * happens to be 0, so a hand-wired config that forgets to set the new field
 * would otherwise silently overwrite instead of accumulate (packed-grad
 * D1). linearBackward must fail fast instead. */
void testLinearBackwardZeroInitAccModeDies(void) {
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitSym(weightsParam, 8, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 1;
    lossDims[1] = 2;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss, (float[]){-4.f, -3.f}, 2);

    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    quantization_t *test = quantizationInitFloat();
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);
    linearLayer->config->linear->weightGradAccMode =
        (outputMode_t)0; /* == OUT_WRITE, forgotten knob */

    ASSERT_EXITS_WITH_FAILURE(linearBackward(linearLayer, forwardInput, loss, propLoss, NULL));

    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(test);
}

void testLinearBackwardFloatWithMismatchedQuantizations() {
    /* Mismatched-quantization variant of testLinearBackwardFloat: the loss
     * arrives in ASYM form, while the layer's parameters and propLoss are
     * Float. Validates that linearBackward routes the loss through a
     * conversion before applying it. */

    /* 3. Build heap forwardInput tensor (Float, shape 1x3). */
    size_t *fwdDims = reserveMemory(2 * sizeof(size_t));
    fwdDims[0] = 1;
    fwdDims[1] = 3;
    size_t *fwdOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, fwdOrder);
    shape_t *fwdShape = reserveMemory(sizeof(shape_t));
    setShape(fwdShape, fwdDims, 2, fwdOrder);
    tensor_t *forwardInput = initTensor(fwdShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(forwardInput, (float[]){0.f, 1.f, 2.f}, 3);

    /* 4. Build heap ASYM loss tensor directly via tensorFillFromFloatBuffer
     *    (the fill helper does the Float->ASYM conversion internally). Converters
     *    write only data + qconfig and no longer touch output->shape (#247), so
     *    an intermediate Float tensor would work too; the direct fill is simply
     *    fewer allocations. */
    size_t *lossAsymDims = reserveMemory(2 * sizeof(size_t));
    lossAsymDims[0] = 1;
    lossAsymDims[1] = 2;
    size_t *lossAsymOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossAsymOrder);
    shape_t *lossAsymShape = reserveMemory(sizeof(shape_t));
    setShape(lossAsymShape, lossAsymDims, 2, lossAsymOrder);
    tensor_t *lossAsym = initTensor(lossAsymShape, quantizationInitAsym(8, HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(lossAsym, (float[]){-4.f, -3.f}, 2);

    /* 5. Build heap propLoss tensor (Float, shape 1x3). */
    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 1;
    propLossDims[1] = 3;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    /* 6. Build the layer with shared float quantization. */
    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, lossAsym, propLoss, NULL);

    /* 7. CAPTURE. */
    size_t sizeWeights = calcNumberOfElementsByParameter(cfg->weights);
    size_t sizeBias = calcNumberOfElementsByParameter(cfg->bias);
    size_t sizePropLoss = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < sizeWeights; i++) {
        capturedWeightGrad[i] = ((float *)cfg->weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < sizeBias; i++) {
        capturedBiasGrad[i] = ((float *)cfg->bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < sizePropLoss; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    /* 8. FREE. */
    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(lossAsym);
    freeTensor(forwardInput);
    freeQuantization(testQ);

    /* 9. ASSERT. */
    float expectedWeightGrad[] = {0.f, -4.f, -8.f, 0.f, -3.f, -6.f};
    for (size_t i = 0; i < sizeWeights; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedWeightGrad[i], capturedWeightGrad[i]);
    }

    float expectedBiasGrad[] = {-4.f, -3.f};
    for (size_t i = 0; i < sizeBias; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedBiasGrad[i], capturedBiasGrad[i]);
    }

    float expectedPropagatedLoss[] = {-8.f, -23.f, 30.f};
    for (size_t i = 0; i < sizePropLoss; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expectedPropagatedLoss[i], capturedPropLoss[i]);
    }
}

void testLinearLayerInitNonTrainable(void) {
    /* 1. Build heap weights tensor. linearLayerInitNonTrainable wraps it
     *    in parameter_t internally with grad=NULL. The post-#106 NULL-guard
     *    in freeParameter makes that wrapper safe to free. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weights = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weights, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);

    /* 2. Build heap bias tensor. */
    size_t *biasDims = reserveMemory(2 * sizeof(size_t));
    biasDims[0] = 1;
    biasDims[1] = 2;
    size_t *biasOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 2, biasOrder);
    tensor_t *bias = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bias, (float[]){-1.f, 3.f}, 2);

    /* 3. Build the non-trainable layer. linearLayerInitNonTrainableLegacy was
     * deleted (Task 9); no factory allocates grad-optional Linear params, so
     * this hand-builds the config the same way that ctor used to (wrap the
     * caller's tensors via parameterInit(t, NULL), store only forwardMath /
     * outputQ) — every other linearConfig_t field stays calloc-zeroed
     * (reserveMemory), which is safe since linearForward only reads
     * forwardMath.type. */
    quantization_t *forwardQ = quantizationInitFloat();
    linearConfig_t *nonTrainableCfg = reserveMemory(sizeof(linearConfig_t));
    nonTrainableCfg->weights = parameterInit(weights, NULL);
    nonTrainableCfg->bias = parameterInit(bias, NULL);
    nonTrainableCfg->forwardMath = arithmeticFromQuantization(forwardQ);
    nonTrainableCfg->outputQ = forwardQ;
    nonTrainableCfg->ownsQuantizations = false;
    layerConfig_t *nonTrainableLayerCfg = reserveMemory(sizeof(layerConfig_t));
    nonTrainableLayerCfg->linear = nonTrainableCfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, LINEAR, nonTrainableLayerCfg);

    /* Wiring asserts (read into stack locals before any free). */
    int capturedLayerNotNull = (layer != NULL);
    int capturedTypeOk = (layer != NULL) && (layer->type == LINEAR);
    linearConfig_t *config = layer->config->linear;
    int capturedWeightGradNull = (config->weights->grad == NULL);
    int capturedBiasGradNull = (config->bias->grad == NULL);

    /* 4. Build heap input tensor (shape 1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* 5. Build heap output tensor (shape 1x2). */
    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 1;
    outputDims[1] = 2;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    linearForward(layer, input, output, NULL);

    /* 6. CAPTURE. */
    float capturedOutput[2];
    capturedOutput[0] = ((float *)output->data)[0];
    capturedOutput[1] = ((float *)output->data)[1];

    /* 7. FREE. freeLinearLayer cascades into freeParameter(weights/bias)
     *    (NULL-grad-safe, post-#106 H3), which cascades into freeTensor. */
    freeLinearLayer(layer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(forwardQ);

    /* 8. ASSERT. */
    TEST_ASSERT_TRUE(capturedLayerNotNull);
    TEST_ASSERT_TRUE(capturedTypeOk);
    TEST_ASSERT_TRUE(capturedWeightGradNull);
    TEST_ASSERT_TRUE(capturedBiasGradNull);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -5.f, capturedOutput[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -4.f, capturedOutput[1]);
}

void testLinearLayerInitAndFreeRoundTrip(void) {
    /* Roundtrip: linearLayerInit allocates layer + outer layerConfig +
     * inner linearConfig (+ weights/bias parameters). freeLinearLayer must
     * release all of it without crashing. linearLayerInit validates
     * lq->outputQ/propLossQ/weightStorage (unlike the deleted Legacy ctor,
     * which tolerated NULL), so this uses a minimal real profile instead. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *linearLayer = linearLayerInit(
        &(linearInit_t){.inFeatures = 1, .outFeatures = 1, .weightInit.key = &initKey}, &lq);
    TEST_ASSERT_NOT_NULL(linearLayer);
    TEST_ASSERT_EQUAL_INT(LINEAR, linearLayer->type);
    TEST_ASSERT_NOT_NULL(linearLayer->config);
    TEST_ASSERT_NOT_NULL(linearLayer->config->linear);

    freeLinearLayer(linearLayer);
    freeQuantization(q);
}

/* ============================================================================
 * Tests for the new layerQuant_t / linearInit_t factory API (PR 1).
 * ========================================================================== */

void testLinearLayerInitBorrowingBuildsLayerWithCorrectShapeAndStoresQuantPointers(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
            .weightInit.key = &initKey,
        },
        &lq);

    TEST_ASSERT_NOT_NULL(layer);
    TEST_ASSERT_EQUAL_INT(LINEAR, layer->type);

    linearConfig_t *cfg = layer->config->linear;
    TEST_ASSERT_NOT_NULL(cfg);
    TEST_ASSERT_FALSE(cfg->ownsQuantizations);

    /* Borrowing variant stores the storage pointer verbatim; the arithmetic
     * slots are by-value derivations of q's type. */
    TEST_ASSERT_EQUAL_PTR(q, cfg->outputQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->weightGradMath.type);
    TEST_ASSERT_EQUAL_PTR(q, cfg->propLossQ);

    /* Weights allocated with shape [outFeatures, inFeatures] */
    TEST_ASSERT_NOT_NULL(cfg->weights);
    tensor_t *weightTensor = cfg->weights->param;
    TEST_ASSERT_NOT_NULL(weightTensor);
    TEST_ASSERT_EQUAL_UINT(2, weightTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(2, weightTensor->shape->dimensions[0]); /* outFeatures */
    TEST_ASSERT_EQUAL_UINT(3, weightTensor->shape->dimensions[1]); /* inFeatures */

    /* Bias allocated with shape [outFeatures] */
    TEST_ASSERT_NOT_NULL(cfg->bias);
    tensor_t *biasTensor = cfg->bias->param;
    TEST_ASSERT_NOT_NULL(biasTensor);
    TEST_ASSERT_EQUAL_UINT(1, biasTensor->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(2, biasTensor->shape->dimensions[0]);

    freeLinearLayer(layer);
}

void testLinearLayerInitBorrowingZeroInChannelsAbortsViaPrintError(void) {
    /* Factory abort on missing required field — covered by design contract;
     * cannot assert PRINT_ERROR + exit from Unity. Marker test. */
    TEST_IGNORE_MESSAGE("Factory abort on missing required field — covered by design contract; "
                        "cannot assert PRINT_ERROR + exit from Unity.");
}

void testLinearLayerInitBorrowingBiasDefaultResolvesToTrue(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4,
            .outFeatures = 1,
            /* .bias omitted -> BIAS_DEFAULT (0) -> resolves to true */
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    TEST_ASSERT_NOT_NULL(cfg->bias); /* bias parameter was allocated */

    freeLinearLayer(layer);
}

void testLinearLayerInitBorrowingBiasFalseLeavesBiasNull(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4,
            .outFeatures = 1,
            .bias = BIAS_FALSE,
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    TEST_ASSERT_NULL(cfg->bias); /* bias parameter not allocated */

    freeLinearLayer(layer);
}

void testLinearLayerInitDefaultGradStorageIsFloat32DespiteSymPropLossQ(void) {
    /* PR1c: default grads are FLOAT32; SYM via knob. A SYM propLossQ with the
     * grad-storage knob left NULL must NOT fall back to SYM_INT32 anymore —
     * the factory default is a hard-pinned FLOAT32, independent of propLossQ. */
    quantization_t *fwd = quantizationInitFloat();             /* FLOAT32 forward + storage */
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY); /* SYM_INT32 backward */
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(fwd),
        .weightGradMath = arithmeticFromQuantization(bwd),
        .biasGradMath = arithmeticFromQuantization(bwd),
        .propLossMath = arithmeticFromQuantization(bwd),
        .outputQ = fwd,
        .propLossQ = bwd,
        .weightStorage = fwd, /* KAIMING init requires FLOAT32 weight storage */
        .biasStorage = fwd,
        /* weightGradStorage / biasGradStorage deliberately left NULL. */
    };

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    int weightGradType = cfg->weights->grad->quantization->type;
    int biasGradType = cfg->bias->grad->quantization->type;

    freeLinearLayer(layer);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_EQUAL_INT_MESSAGE(FLOAT32, weightGradType,
                                  "PR1c: default (NULL knob) weight grad storage must be FLOAT32");
    TEST_ASSERT_EQUAL_INT_MESSAGE(FLOAT32, biasGradType,
                                  "PR1c: default (NULL knob) bias grad storage must be FLOAT32");
}

void testLinearLayerInitSymInt32BackwardMathYieldsSymInt32Grad(void) {
    /* Regression for the "config lies" bug: a Linear built with a SYM_INT32
     * backwardMath must store SYM_INT32 parameter gradients when the caller
     * opts in via the grad-storage knob. PR1c: default grads are FLOAT32; SYM
     * via knob — the explicit weightGradStorage/biasGradStorage below is what
     * keeps this test exercising the SYM path post-flip. */
    quantization_t *fwd = quantizationInitFloat();             /* FLOAT32 forward + storage */
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY); /* SYM_INT32 backward */
    layerQuant_t lq = {
        .forwardMath = arithmeticFromQuantization(fwd),
        .weightGradMath = arithmeticFromQuantization(bwd),
        .biasGradMath = arithmeticFromQuantization(bwd),
        .propLossMath = arithmeticFromQuantization(bwd),
        .outputQ = fwd,
        .propLossQ = bwd,
        .weightStorage = fwd, /* KAIMING init requires FLOAT32 weight storage */
        .biasStorage = fwd,
        .weightGradStorage = bwd,
        .biasGradStorage = bwd,
    };

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    int weightGradType = cfg->weights->grad->quantization->type;
    int biasGradType = cfg->bias->grad->quantization->type;

    freeLinearLayer(layer);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_EQUAL_INT_MESSAGE(SYM_INT32, weightGradType,
                                  "weight grad must be SYM_INT32 when backwardMath is SYM_INT32");
    TEST_ASSERT_EQUAL_INT_MESSAGE(SYM_INT32, biasGradType,
                                  "bias grad must be SYM_INT32 when backwardMath is SYM_INT32");
}

void testLinearLayerInitOwningDeepCopiesQuantizations(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;

    /* Owning variant: cfg->outputQ is a fresh allocation, NOT the original q */
    TEST_ASSERT_NOT_EQUAL(q, cfg->outputQ);
    TEST_ASSERT_EQUAL_INT(ARITH_FLOAT32, cfg->weightGradMath.type);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);

    /* But the copy has equal type to the original */
    TEST_ASSERT_EQUAL_INT(q->type, cfg->outputQ->type);

    /* ownsQuantizations flag is set */
    TEST_ASSERT_TRUE(cfg->ownsQuantizations);

    freeLinearLayer(layer);
}

void testLinearLayerInitOwningFreesAllAllocationsWithoutLeak(void) {
    /* Build + free 5 layers — if anything leaks, valgrind will catch it
     * during CI (not asserted here, just exercise the path). */
    for (int i = 0; i < 5; i++) {
        quantization_t *q = quantizationInitFloat();
        layerQuant_t lq;
        layerQuantInitUniform(&lq, q);

        const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
        layer_t *layer = linearLayerInitOwning(
            &(linearInit_t){
                .inFeatures = 8,
                .outFeatures = 4,
                .bias = BIAS_TRUE,
                .weightInit.key = &initKey,
            },
            &lq);

        freeLinearLayer(layer);
        /* Note: caller-side q deliberately not freed — it's caller-owned and
         * the Owning factory has its own copies. q leaks but that's the
         * existing pattern in this codebase (quantizationInit* returns heap,
         * never freed). */
    }
    TEST_PASS();
}

/* ============================================================================
 * Grad-storage knob (#261 / layerQuant_t restructure, Task 8 step 3).
 * ========================================================================== */

void testLinearLayerInitOwningWeightGradStorageKnobOverridesPropLossQDefault(void) {
    /* PR1c: default (knob == NULL) grad storage is a hard-pinned FLOAT32; a
     * non-NULL weightGradStorage config must override that default end-to-end,
     * through the same getQLike path gradInit already uses. (This test's own
     * propLossQ is already FLOAT32, so the flip doesn't change its outcome —
     * only the comment's claim about what NULL falls back to.) */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *gradKnob = quantizationInitSymInt32WithBits(HALF_AWAY, 16);
    lq.weightGradStorage = gradKnob;

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    tensor_t *wGrad = getGradFromParameter(cfg->weights);
    int gradType = wGrad->quantization->type;
    /* Guard the qConfig dereference: a wrong-type grad (e.g. the knob silently
     * ignored) has qConfig == NULL for FLOAT32 — asserting gradType first
     * keeps that failure a clean assertion, not a NULL-deref crash. */
    uint8_t gradBits =
        (gradType == SYM_INT32) ? ((symInt32QConfig_t *)wGrad->quantization->qConfig)->qMaxBits : 0;

    freeLinearLayer(layer);
    freeQuantization(gradKnob);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_INT_MESSAGE(SYM_INT32, gradType,
                                  "weightGradStorage knob must override the propLossQ default");
    TEST_ASSERT_EQUAL_UINT8(16, gradBits);
}

void testLinearLayerInitOwningBoolWeightGradStorageKnobAborts(void) {
    /* getQLike (gradInit's clone path) supports SYM (packed grads, #269) but
     * deliberately has no BOOL arm — a knob naming an unsupported dtype must
     * fail fast at the factory, the earliest gate. */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *boolGradQ = quantizationInitBool();
    lq.weightGradStorage = boolGradQ;

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    ASSERT_EXITS_WITH_FAILURE(linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq));

    freeQuantization(boolGradQ);
    freeQuantization(q);
}

/* BFP epic PR3 Task 6: BFP twin of the SYM_INT32 knob test above -- a
 * per-tensor BFP weightGradStorage template must land BFP weight grad
 * storage end-to-end, through the same gradInit/getQLike path (Step 1's
 * grouped-only gate). Written FIRST in Task 6 alongside the load-bearing e2e
 * (UnitTestMultiLayerTraining.c): RED at gradInit's then-unconditional BFP
 * reject before Step 1 lands, GREEN after. */
void testLinearLayerInitOwningBfpWeightGradStorageKnob(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *gradKnob = quantizationInitBfp(8, 8, HALF_AWAY);
    lq.weightGradStorage = gradKnob;

    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    tensor_t *wGrad = getGradFromParameter(cfg->weights);
    int gradType = wGrad->quantization->type;
    /* Guard the qConfig dereference (same pattern as the SYM_INT32 twin). */
    size_t gradNumGroups =
        (gradType == BFP) ? ((bfpQConfig_t *)wGrad->quantization->qConfig)->numGroups : 0;

    freeLinearLayer(layer);
    freeQuantization(gradKnob);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, gradType,
                                  "weightGradStorage knob must land BFP weight grad storage");
    TEST_ASSERT_EQUAL_size_t_MESSAGE(1, gradNumGroups, "grads are per-tensor-only (#300 axis)");
}

/* Helper: build a 2-D FLOAT32 tensor on the heap with the given values. */
static tensor_t *buildFloatTensor2d(size_t rows, size_t cols, const float *data) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = rows;
    d[1] = cols;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    tensor_t *t = initTensor(s, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, data, rows * cols);
    return t;
}

/* Helper: build a 1x3 FLOAT32 input tensor with the given values (NULL => zeros). */
static tensor_t *e2eMakeInput(const float *vals) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = 1;
    d[1] = 3;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    tensor_t *t = initTensor(s, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, 3);
    }
    return t;
}
/* Helper: build a 1x2 FLOAT32 loss tensor (NULL => zeros). */
static tensor_t *e2eMake1x2(const float *vals) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = 1;
    d[1] = 2;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    tensor_t *t = initTensor(s, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, 2);
    }
    return t;
}

/* Helper: build a 1x3 SYM_INT32 propLoss buffer (the SYM_INT32 backward writes the
 * propLoss result + scale in place, so its dtype must match propLossQ = SYM_INT32). */
static tensor_t *e2eMakeSym1x3(void) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = 1;
    d[1] = 3;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    return initTensor(s, quantizationInitSymInt32(HALF_AWAY), NULL);
}

/* #380 PR1 Task 5: backward guard -- a frozen twin must skip the weight/bias
 * grad writes entirely (buffers stay all-zero) while still producing a
 * propLoss byte-identical to its trainable twin. Hand-seeded FLOAT32
 * fixtures via buildBorrowedLinearLayer (deterministic, no RNG) so the two
 * twins start out bit-identical; only `frozen` differs. */
void testLinearBackwardFrozenTwinPropLossIdenticalGradsZero(void) {
    quantization_t *q = quantizationInitFloat();

    tensor_t *weightsParamA = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *weightsGradA = gradInitFloat(weightsParamA, NULL);
    parameter_t *weightsA = parameterInit(weightsParamA, weightsGradA);
    tensor_t *biasParamA = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    tensor_t *biasGradA = gradInitFloat(biasParamA, NULL);
    parameter_t *biasA = parameterInit(biasParamA, biasGradA);
    layer_t *trainableTwin = buildBorrowedLinearLayer(weightsA, biasA, q);

    tensor_t *weightsParamB = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *weightsGradB = gradInitFloat(weightsParamB, NULL);
    parameter_t *weightsB = parameterInit(weightsParamB, weightsGradB);
    tensor_t *biasParamB = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    tensor_t *biasGradB = gradInitFloat(biasParamB, NULL);
    parameter_t *biasB = parameterInit(biasParamB, biasGradB);
    layer_t *frozenTwin = buildBorrowedLinearLayer(weightsB, biasB, q);
    frozenTwin->config->linear->frozen = true;

    tensor_t *forwardInput = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){-4.f, -3.f});
    tensor_t *propLossTrainable = buildFloatTensor2d(1, 3, (float[]){0.f, 0.f, 0.f});
    tensor_t *propLossFrozen = buildFloatTensor2d(1, 3, (float[]){0.f, 0.f, 0.f});

    linearBackward(trainableTwin, forwardInput, loss, propLossTrainable, NULL);
    linearBackward(frozenTwin, forwardInput, loss, propLossFrozen, NULL);

    size_t numWeights = calcNumberOfElementsByTensor(weightsParamA);
    size_t numBias = calcNumberOfElementsByTensor(biasParamA);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossTrainable);

    bool trainableWeightGradNonzero = false;
    bool frozenWeightGradAllZero = true;
    for (size_t i = 0; i < numWeights; i++) {
        if (((float *)weightsGradA->data)[i] != 0.0f) {
            trainableWeightGradNonzero = true;
        }
        if (((float *)weightsGradB->data)[i] != 0.0f) {
            frozenWeightGradAllZero = false;
        }
    }
    bool trainableBiasGradNonzero = false;
    bool frozenBiasGradAllZero = true;
    for (size_t i = 0; i < numBias; i++) {
        if (((float *)biasGradA->data)[i] != 0.0f) {
            trainableBiasGradNonzero = true;
        }
        if (((float *)biasGradB->data)[i] != 0.0f) {
            frozenBiasGradAllZero = false;
        }
    }
    bool propLossIdentical =
        memcmp(propLossTrainable->data, propLossFrozen->data,
               calcNumberOfBytesForData(propLossTrainable->quantization, numPropLoss)) == 0;

    freeLinearLayer(trainableTwin);
    freeLinearLayer(frozenTwin);
    freeTensor(propLossFrozen);
    freeTensor(propLossTrainable);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(trainableWeightGradNonzero,
                             "trainable twin weight grad must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenWeightGradAllZero,
                             "frozen twin weight grad must stay untouched (all-zero)");
    TEST_ASSERT_TRUE_MESSAGE(trainableBiasGradNonzero,
                             "trainable twin bias grad must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenBiasGradAllZero,
                             "frozen twin bias grad must stay untouched (all-zero)");
    TEST_ASSERT_TRUE_MESSAGE(propLossIdentical, "propLoss must be byte-identical between twins");
}

/* Factory-frozen layer (grads == NULL, Task 1): linearBackward must complete
 * without dereferencing the (absent) grad buffers -- the ASan gate catches
 * any NULL/OOB deref if the guard is missing or misplaced. */
void testLinearBackwardFrozenFactoryLayerRunsWithoutGradBuffers(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    tensor_t *forwardInput = e2eMakeInput((float[]){1.f, 2.f, 3.f});
    tensor_t *loss = e2eMake1x2((float[]){-4.f, -3.f});
    tensor_t *propLoss = e2eMakeInput(NULL);

    linearBackward(layer, forwardInput, loss, propLoss, NULL);

    bool gradStillNull = layer->config->linear->weights->grad == NULL;

    freeLinearLayer(layer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);

    TEST_ASSERT_TRUE(gradStillNull);
}

/* ---- each funnel op of the layer draws from its own op key ---- */

/* The expected stream states come from the Python mirror, tags as numbers
 * (FORWARD 4, BACKWARD 5, OP 7, OUT 10; Linear's forward is OP 0, its dx OP 3):
 *   uv run python -c "from examples._shared.keyed_rng import (fold, key_from_seed as kfs,
 *   open_stream as o); k = kfs(0x1a7e); f = fold(k, 4, 0); b = fold(k, 5, 0); print([
 *   hex(o(fold(fold(p, 7, i), 10, 0))) for (p, i) in ((f, 0), (b, 3))])"
 * Each check also compares with the explicit rngFold chain. */
#define LINEAR_EXEC_TEST_SEED 0x1a7eu
#define LINEAR_MAX_CELL_RECORDS 4

typedef struct {
    bool hasStream;
    uint32_t state;
} linearCellRecord_t;

static linearCellRecord_t linearCellRecords[LINEAR_MAX_CELL_RECORDS];
static size_t linearCellRecordCount;
static conversionFunction_t linearSavedCell;

/* Records the stream, then runs the real [FLOAT32][SYM_INT32] cell. */
static void linearRecordingCell(tensor_t *in, tensor_t *out, rngStream_t *rs) {
    if (linearCellRecordCount < LINEAR_MAX_CELL_RECORDS) {
        linearCellRecords[linearCellRecordCount].hasStream = rs != NULL;
        linearCellRecords[linearCellRecordCount].state = (rs != NULL) ? rs->state : 0u;
    }
    linearCellRecordCount++;
    linearSavedCell(in, out, rs);
}

static void linearInstallRecordingCell(void) {
    linearSavedCell = conversionMatrix[FLOAT32][SYM_INT32];
    conversionMatrix[FLOAT32][SYM_INT32] = linearRecordingCell;
    linearCellRecordCount = 0;
}

static void linearRestoreCell(void) {
    conversionMatrix[FLOAT32][SYM_INT32] = linearSavedCell;
}

static tensor_t *linearMakeSym1x2(void) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = 1;
    d[1] = 2;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    return initTensor(s, quantizationInitSymInt32(HALF_AWAY), NULL);
}

static void assertLinearOutStream(size_t record, uint32_t phaseTag, uint32_t opIndex,
                                  uint32_t pythonState) {
    rngKey_t layerKey = rngKeyFromSeed(LINEAR_EXEC_TEST_SEED);
    rngKey_t opKey = rngFold(rngFold(layerKey, phaseTag, 0u), RNG_TAG_OP, opIndex);
    TEST_ASSERT_TRUE(linearCellRecords[record].hasStream);
    TEST_ASSERT_EQUAL_HEX32(rngOpen(rngFold(opKey, RNG_TAG_OUT, 0u)).state,
                            linearCellRecords[record].state);
    TEST_ASSERT_EQUAL_HEX32(pythonState, linearCellRecords[record].state);
}

/* A FLOAT32 forward into a SYM_INT32 output rounds in the OUT_WRITE epilogue
 * of the forward op: OP 0 under the FORWARD phase of the layer key. */
void testLinearForwardOpDrawsFromItsForwardOpKey(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    tensor_t *input = e2eMakeInput((float[]){0.5f, -1.f, 2.f});
    tensor_t *output = linearMakeSym1x2();
    layerExec_t exec = {.key = rngKeyFromSeed(LINEAR_EXEC_TEST_SEED)};
    linearInstallRecordingCell();

    linearForward(layer, input, output, &exec);

    linearRestoreCell();
    freeTensor(output);
    freeTensor(input);
    freeLinearLayer(layer);
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)linearCellRecordCount);
    assertLinearOutStream(0, RNG_TAG_FORWARD, 0u, 0x02960061u);
}

/* A frozen layer's backward runs only the dx op (OP 3 under BACKWARD); a
 * SYM_INT32 propLoss rounds in its OUT_WRITE epilogue. */
void testLinearBackwardDxOpDrawsFromItsBackwardOpKey(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    tensor_t *forwardInput = e2eMakeInput((float[]){0.5f, -1.f, 2.f});
    tensor_t *loss = e2eMake1x2((float[]){0.25f, -0.75f});
    tensor_t *propLoss = e2eMakeSym1x3();
    layerExec_t exec = {.key = rngKeyFromSeed(LINEAR_EXEC_TEST_SEED)};
    linearInstallRecordingCell();

    linearBackward(layer, forwardInput, loss, propLoss, &exec);

    linearRestoreCell();
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeLinearLayer(layer);
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)linearCellRecordCount);
    assertLinearOutStream(0, RNG_TAG_BACKWARD, 3u, 0xeb73a7ebu);
}

/* Without a context no funnel op gets a key, so the epilogue gets no stream. */
void testLinearForwardWithoutAContextHandsTheFunnelNoStream(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    tensor_t *input = e2eMakeInput((float[]){0.5f, -1.f, 2.f});
    tensor_t *output = linearMakeSym1x2();
    linearInstallRecordingCell();

    linearForward(layer, input, output, NULL);

    linearRestoreCell();
    freeTensor(output);
    freeTensor(input);
    freeLinearLayer(layer);
    TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)linearCellRecordCount);
    TEST_ASSERT_FALSE(linearCellRecords[0].hasStream);
}

static void linearSrForwardWithoutAContext(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE);
    layer->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    linearForward(layer, e2eMakeInput((float[]){0.5f, -1.f, 2.f}), linearMakeSym1x2(), NULL);
}

/* No key, no randomness: a Linear forward that rounds stochastically into a
 * SYM_INT32 output stops at its first draw when the layer has no context
 * (the HALF_AWAY forward above takes NULL and never stops). */
void testLinearSrForwardWithoutAContextExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "roundByMode: SR rounding without a stream",
                             linearSrForwardWithoutAContext());
}

#define LINEAR_PURITY_OUT 16

/* One SR forward of a Linear(3 -> 16) into a fresh 8-bit SYM_INT32 output
 * under the key of `seed`: every output element is rounded by its own draw. */
static void linearSrForwardUnder(layer_t *layer, tensor_t *input, uint64_t seed,
                                 int32_t codes[LINEAR_PURITY_OUT], float *scale) {
    size_t *d = reserveMemory(2 * sizeof(size_t));
    d[0] = 1;
    d[1] = LINEAR_PURITY_OUT;
    size_t *o = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, 2, o);
    tensor_t *output = initTensor(s, quantizationInitSymInt32WithBits(HALF_AWAY, 8), NULL);
    const layerExec_t exec = {.key = rngKeyFromSeed(seed)};

    linearForward(layer, input, output, &exec);

    memcpy(codes, output->data, LINEAR_PURITY_OUT * sizeof(int32_t));
    *scale = ((symInt32QConfig_t *)output->quantization->qConfig)->scale;
    freeTensor(output);
}

/* The same key gives the same codes and scale; another key gives other codes. */
void testLinearSrForwardIsAPureFunctionOfItsKey(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    lq.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInitOwning(&(linearInit_t){.inFeatures = 3,
                                                           .outFeatures = LINEAR_PURITY_OUT,
                                                           .trainable = TRAINABLE_FALSE,
                                                           .weightInit.key = &initKey},
                                           &lq);
    freeQuantization(q);
    tensor_t *input = e2eMakeInput((float[]){0.5f, -1.f, 2.f});
    int32_t first[LINEAR_PURITY_OUT];
    int32_t again[LINEAR_PURITY_OUT];
    int32_t other[LINEAR_PURITY_OUT];
    float firstScale;
    float againScale;
    float otherScale;

    linearSrForwardUnder(layer, input, 0x1eafu, first, &firstScale);
    linearSrForwardUnder(layer, input, 0x1eafu, again, &againScale);
    linearSrForwardUnder(layer, input, 0x1eb0u, other, &otherScale);

    freeTensor(input);
    freeLinearLayer(layer);
    TEST_ASSERT_EQUAL_INT32_ARRAY_MESSAGE(first, again, LINEAR_PURITY_OUT,
                                          "the same key repeats the codes");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(&firstScale, &againScale, sizeof(float),
                                     "the same key repeats the scale");
    TEST_ASSERT_FALSE_MESSAGE(memcmp(first, other, sizeof(first)) == 0,
                              "another key rounds otherwise");
}

/* #380 PR2 Task 1: propLoss == NULL is a grads-only call -- weight/bias grads
 * must be computed exactly as with a real propLoss, and no dx memory may be
 * touched. Twin fixture (both TRAINABLE, hand-seeded, bit-identical) mirrors
 * the PR1 frozen-twin idiom: twin A gets a real propLoss buffer, twin B gets
 * a literal NULL. Pre-guard, twin B's call dereferences the NULL propLoss and
 * crashes (RED); post-guard, weight/bias grads match twin A's byte-for-byte
 * and twin A's propLoss is non-degenerate (proving the NULL round skipped
 * ONLY dx, not the grad computation). */
void testLinearBackwardNullPropLossComputesGradsOnly(void) {
    quantization_t *q = quantizationInitFloat();

    tensor_t *weightsParamA = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *weightsGradA = gradInitFloat(weightsParamA, NULL);
    parameter_t *weightsA = parameterInit(weightsParamA, weightsGradA);
    tensor_t *biasParamA = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    tensor_t *biasGradA = gradInitFloat(biasParamA, NULL);
    parameter_t *biasA = parameterInit(biasParamA, biasGradA);
    layer_t *twinA = buildBorrowedLinearLayer(weightsA, biasA, q);

    tensor_t *weightsParamB = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *weightsGradB = gradInitFloat(weightsParamB, NULL);
    parameter_t *weightsB = parameterInit(weightsParamB, weightsGradB);
    tensor_t *biasParamB = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    tensor_t *biasGradB = gradInitFloat(biasParamB, NULL);
    parameter_t *biasB = parameterInit(biasParamB, biasGradB);
    layer_t *twinB = buildBorrowedLinearLayer(weightsB, biasB, q);

    tensor_t *forwardInput = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){-4.f, -3.f});
    tensor_t *propLossA = buildFloatTensor2d(1, 3, (float[]){0.f, 0.f, 0.f});

    linearBackward(twinA, forwardInput, loss, propLossA, NULL);
    linearBackward(twinB, forwardInput, loss, NULL, NULL);

    size_t numWeights = calcNumberOfElementsByTensor(weightsParamA);
    size_t numBias = calcNumberOfElementsByTensor(biasParamA);
    size_t numPropLoss = calcNumberOfElementsByTensor(propLossA);

    bool weightGradsIdentical =
        memcmp(weightsGradA->data, weightsGradB->data,
               calcNumberOfBytesForData(weightsGradA->quantization, numWeights)) == 0;
    bool biasGradsIdentical =
        memcmp(biasGradA->data, biasGradB->data,
               calcNumberOfBytesForData(biasGradA->quantization, numBias)) == 0;
    bool propLossANonDegenerate = false;
    for (size_t i = 0; i < numPropLoss; i++) {
        if (((float *)propLossA->data)[i] != 0.0f) {
            propLossANonDegenerate = true;
        }
    }

    freeLinearLayer(twinA);
    freeLinearLayer(twinB);
    freeTensor(propLossA);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(
        weightGradsIdentical,
        "weight grads must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(
        biasGradsIdentical,
        "bias grads must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(propLossANonDegenerate,
                             "twin A's propLoss must be non-degenerate (nonzero), proving the "
                             "NULL round only skipped dx");
}

void testLinearSymInt32GradAccumulatesOverTwoMicrobatchesAndSteps(void) {
    /* outFeatures=2, inFeatures=3. forward input [1,2,3], loss [0.5, -0.25].
     * Two identical microbatches => accumulated weight grad ~= 2 * (loss^T @ input).
     * PR1c: default grads are FLOAT32; SYM via knob — weightGradStorage/
     * biasGradStorage below opt this layer back into the SYM_INT32 grad path
     * so the test keeps exercising SYM accumulation parity, not float-vs-float. */
    const float inputVals[3] = {1.0f, 2.0f, 3.0f};
    const float lossVals[2] = {0.5f, -0.25f};

    /* ---- SYM_INT32-backward layer (under test) ---- */
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym = {.forwardMath = arithmeticFromQuantization(fwd),
                          .weightGradMath = arithmeticFromQuantization(bwd),
                          .biasGradMath = arithmeticFromQuantization(bwd),
                          .propLossMath = arithmeticFromQuantization(bwd),
                          .outputQ = fwd,
                          .propLossQ = bwd,
                          .weightStorage = fwd,
                          .biasStorage = fwd,
                          .weightGradStorage = bwd,
                          .biasGradStorage = bwd,
                          .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                          .biasGradAccMode = OUT_ACC_FIXED_SCALE};
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *symLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lqSym);

    tensor_t *symWGrad = symLayer->config->linear->weights->grad;
    TEST_ASSERT_EQUAL_INT(SYM_INT32, symWGrad->quantization->type); /* guard */

    tensor_t *symIn1 = e2eMakeInput(inputVals);
    tensor_t *symLoss1 = e2eMake1x2(lossVals);
    tensor_t *symProp1 = e2eMakeSym1x3(); /* propLoss [1,3], SYM_INT32 (matches propLossQ) */
    linearBackward(symLayer, symIn1, symLoss1, symProp1, NULL);

    tensor_t *symIn2 = e2eMakeInput(inputVals);
    tensor_t *symLoss2 = e2eMake1x2(lossVals);
    tensor_t *symProp2 = e2eMakeSym1x3();
    linearBackward(symLayer, symIn2, symLoss2, symProp2, NULL);

    /* Capture accumulated SYM_INT32 weight grad as float (dequantized). */
    size_t nW = calcNumberOfElementsByTensor(symWGrad);
    float symGradFloat[6];
    {
        tensor_t gf;
        quantization_t gfQ;
        initFloat32Quantization(&gfQ);
        uint8_t gfData[6 * sizeof(float)];
        setTensorValuesForConversion(gfData, &gfQ, symWGrad, &gf);
        convertTensor(symWGrad, &gf, NULL);
        for (size_t i = 0; i < nW; i++) {
            symGradFloat[i] = ((float *)gf.data)[i];
        }
    }

    /* ---- Optimizer step on the SYM_INT32 layer ("updates the param without crashing"). ---- */
    layer_t *symModel[] = {symLayer};
    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *symOptim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, symModel, 1, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    optimizerFunctions[symOptim->type].step(symOptim, NULL);
    tensor_t *symWParam = symLayer->config->linear->weights->param;
    int paramFinite = 1;
    for (size_t i = 0; i < nW; i++) {
        if (!isfinite(((float *)symWParam->data)[i])) {
            paramFinite = 0;
        }
    }

    /* ---- FLOAT32-backward layer (reference) ---- */
    quantization_t *fwd2 = quantizationInitFloat();
    quantization_t *bwd2 = quantizationInitFloat();
    layerQuant_t lqF = {.forwardMath = arithmeticFromQuantization(fwd2),
                        .weightGradMath = arithmeticFromQuantization(bwd2),
                        .biasGradMath = arithmeticFromQuantization(bwd2),
                        .propLossMath = arithmeticFromQuantization(bwd2),
                        .outputQ = fwd2,
                        .propLossQ = bwd2,
                        .weightStorage = fwd2,
                        .biasStorage = fwd2,
                        .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                        .biasGradAccMode = OUT_ACC_FIXED_SCALE};
    layer_t *fLayer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE, .weightInit.key = &initKey},
        &lqF);
    tensor_t *fWGrad = fLayer->config->linear->weights->grad;

    tensor_t *fIn1 = e2eMakeInput(inputVals);
    tensor_t *fLoss1 = e2eMake1x2(lossVals);
    tensor_t *fProp1 = e2eMakeInput(NULL);
    linearBackward(fLayer, fIn1, fLoss1, fProp1, NULL);
    tensor_t *fIn2 = e2eMakeInput(inputVals);
    tensor_t *fLoss2 = e2eMake1x2(lossVals);
    tensor_t *fProp2 = e2eMakeInput(NULL);
    linearBackward(fLayer, fIn2, fLoss2, fProp2, NULL);

    float refGradFloat[6];
    for (size_t i = 0; i < nW; i++) {
        refGradFloat[i] = ((float *)fWGrad->data)[i];
    }

    /* ---- Compare accumulated grads within SYM_INT32 tolerance ---- */
    bool gradsClose = true;
    for (size_t i = 0; i < nW; i++) {
        if (fabsf(symGradFloat[i] - refGradFloat[i]) > 5e-3f) {
            gradsClose = false;
        }
    }

    freeLinearLayer(fLayer);
    freeTensor(fProp2);
    freeTensor(fLoss2);
    freeTensor(fIn2);
    freeTensor(fProp1);
    freeTensor(fLoss1);
    freeTensor(fIn1);
    freeQuantization(bwd2);
    freeQuantization(fwd2);

    /* freeOptim frees the SYM layer's parameters; do NOT also freeLinearLayer(symLayer)
     * (double-free). Free the layer/config shell manually (borrowing factory: caller owns
     * the quantizations, freed separately below). */
    freeOptim(symOptim);
    freeReservedMemory(symLayer->config->linear);
    freeReservedMemory(symLayer->config);
    freeReservedMemory(symLayer);
    freeTensor(symProp2);
    freeTensor(symLoss2);
    freeTensor(symIn2);
    freeTensor(symProp1);
    freeTensor(symLoss1);
    freeTensor(symIn1);
    freeQuantization(momentumQ);
    freeQuantization(bwd);
    freeQuantization(fwd);

    TEST_ASSERT_TRUE_MESSAGE(gradsClose,
                             "SYM_INT32 accumulated weight grad diverged from FLOAT32 reference");
    TEST_ASSERT_TRUE_MESSAGE(paramFinite,
                             "SYM_INT32 optimizer step left a non-finite weight param");
}

/*! Returns the max |value| over a FLOAT32 tensor's data buffer. */
static float linearMaxAbsFloat(const tensor_t *t) {
    const float *vals = (const float *)t->data;
    size_t n = t->shape->dimensions[0];
    for (size_t d = 1; d < t->shape->numberOfDimensions; d++) {
        n *= t->shape->dimensions[d];
    }
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(vals[i]);
        if (a > m) {
            m = a;
        }
    }
    return m;
}

void testLinearLayerInitDefaultWeightsWithinPyTorchBound(void) {
    /* PyTorch default Linear init: weight ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)),
     * bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in)); fan_in = inFeatures. */
    const size_t inFeatures = 256, outFeatures = 64;
    const float bound = 1.0f / sqrtf((float)inFeatures);

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(7u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = inFeatures,
            .outFeatures = outFeatures,
            .bias = BIAS_TRUE,
            .weightInit.key = &initKey,
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    float weightMaxAbs = linearMaxAbsFloat(cfg->weights->param);
    float biasMaxAbs = linearMaxAbsFloat(cfg->bias->param);

    freeLinearLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs <= bound * 1.001f,
                             "Linear default weights exceed PyTorch bound 1/sqrt(fan_in)");
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs >= bound * 0.85f,
                             "Linear default weights far below PyTorch bound -> wrong scale");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs > 0.0f,
                             "Linear default bias is zero (PyTorch draws it from a uniform)");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs <= bound * 1.001f,
                             "Linear default bias exceeds PyTorch bound 1/sqrt(fan_in)");
}

void testLinearLayerInitXavierUniformOverrideUsesGlorotBound(void) {
    /* Explicit weightInit = {INIT_XAVIER_UNIFORM} -> Glorot, default gain 1:
     * xavierUniform(1, fan_in, fan_out) = uniform(+/- sqrt(6/(fan_in+fan_out))).
     * Distinct from the default bound 1/sqrt(fan_in). Bias stays PyTorch
     * default uniform(+/- 1/sqrt(fan_in)). */
    const size_t inFeatures = 256, outFeatures = 64;
    const float defaultBound = 1.0f / sqrtf((float)inFeatures);
    const float xavierBound = sqrtf(6.0f / (float)(inFeatures + outFeatures));

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    const rngKey_t initKey = rngFold(rngKeyFromSeed(7u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = inFeatures,
            .outFeatures = outFeatures,
            .bias = BIAS_TRUE,
            .weightInit = {INIT_XAVIER_UNIFORM, .key = &initKey},
        },
        &lq);

    linearConfig_t *cfg = layer->config->linear;
    float weightMaxAbs = linearMaxAbsFloat(cfg->weights->param);
    float biasMaxAbs = linearMaxAbsFloat(cfg->bias->param);

    freeLinearLayer(layer);
    freeQuantization(q);

    /* Xavier bound here (~0.137) is wider than the default bound (~0.0625):
     * confirms the override changed the scale. */
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs > defaultBound,
                             "Xavier override did not change weights away from the default bound");
    TEST_ASSERT_TRUE_MESSAGE(weightMaxAbs <= xavierBound * 1.001f,
                             "Xavier weights exceed the sqrt(6/(fan_in+fan_out)) bound");
    TEST_ASSERT_TRUE_MESSAGE(biasMaxAbs <= defaultBound * 1.001f,
                             "Bias must stay PyTorch default uniform regardless of weight scheme");
}

/* A Linear(4 -> 3) built from `initKey` with a Xavier(gain 1) recipe; its
 * weights and bias are copied out before the layer is freed. */
static void linearInitValuesOf(const rngKey_t *initKey, float weights[12], float bias[3]) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){.inFeatures = 4,
                        .outFeatures = 3,
                        .bias = BIAS_TRUE,
                        .weightInit = {INIT_XAVIER_UNIFORM, .gain = 1.0f, .key = initKey}},
        &lq);
    memcpy(weights, layer->config->linear->weights->param->data, 12 * sizeof(float));
    memcpy(bias, layer->config->linear->bias->param->data, 3 * sizeof(float));
    freeLinearLayer(layer);
    freeQuantization(q);
}

/* The weight draws from the (PARAM, 0) stream of the layer's init key, the
 * bias from (PARAM, 1). The bias always uses PyTorch's default,
 * kaimingUniform(sqrt(1/3), fan_in): 0.57735026919f is that gain, as
 * LayerCommon.c spells it. */
void testLinearInitDrawsWeightAndBiasFromTheParamStreamsOfItsKey(void) {
    const rngKey_t initKey = rngFold(rngKeyFromSeed(5u), RNG_TAG_INIT, 0u);
    float weights[12];
    float bias[3];
    linearInitValuesOf(&initKey, weights, bias);
    float wantWeights[12];
    float wantBias[3];
    rngStream_t weightStream = rngOpen(rngFold(initKey, RNG_TAG_PARAM, 0u));
    for (size_t i = 0; i < 12; i++) {
        wantWeights[i] = xavierUniform(1.0f, 4, 3, &weightStream);
    }
    rngStream_t biasStream = rngOpen(rngFold(initKey, RNG_TAG_PARAM, 1u));
    for (size_t i = 0; i < 3; i++) {
        wantBias[i] = kaimingUniform(0.57735026919f, 4, &biasStream);
    }
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wantWeights, weights, 12);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(wantBias, bias, 3);
    /* Independent golds, the first three of each, from the Python mirror (tag
     * values spelled as numbers: INIT 22, PARAM 19), from the repository root:
     *   uv run python -c "from examples._shared.keyed_rng import (key_from_seed, fold,
     *   open_stream, next_unit_float as nuf); import numpy as np; f = np.float32; k = fold(
     *   key_from_seed(5), 22, 0); lw = np.sqrt(f(6) / f(7)); lb = f(0.57735026919) * np.sqrt(
     *   f(3) / f(4)); t = (open_stream(fold(k, 19, 0)), 0.0); w = [(t := nuf(t[0]))[1] for _ in (
     *   0, 1, 2)]; t = (open_stream(fold(k, 19, 1)), 0.0); b = [(t := nuf(t[0]))[1] for _ in (
     *   0, 1, 2)]; print([repr(float(-lw + f(u) * (lw + lw))) for u in w], [repr(float(
     *   -lb + f(u) * (lb + lb))) for u in b])"
     */
    const float goldWeights[3] = {-0.36196327209472656f, -0.7439754009246826f,
                                  0.49748659133911133f};
    const float goldBias[3] = {0.12533333897590637f, -0.012614428997039795f, -0.02550959587097168f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(goldWeights, weights, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(goldBias, bias, 3);
}

/* Initial values depend on the key alone: the same key gives the same values
 * whatever was built in between, the next layer position other values. */
void testLinearInitIsAFunctionOfItsKeyAlone(void) {
    const rngKey_t root = rngKeyFromSeed(5u);
    const rngKey_t key0 = rngFold(root, RNG_TAG_INIT, 0u);
    const rngKey_t key1 = rngFold(root, RNG_TAG_INIT, 1u);
    float first[12], firstBias[3], other[12], otherBias[3], again[12], againBias[3];
    linearInitValuesOf(&key0, first, firstBias);
    linearInitValuesOf(&key1, other, otherBias);
    linearInitValuesOf(&key0, again, againBias);
    TEST_ASSERT_EQUAL_MEMORY(first, again, sizeof first);
    TEST_ASSERT_EQUAL_MEMORY(firstBias, againBias, sizeof firstBias);
    TEST_ASSERT_FALSE(memcmp(first, other, sizeof first) == 0);
    TEST_ASSERT_FALSE(memcmp(firstBias, otherBias, sizeof firstBias) == 0);
}

static void linearInitWithoutAKey(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    (void)linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
}

/* No default key: a factory without one stops at its first draw. */
void testLinearInitWithoutAKeyExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "initDistribution: random distribution without a stream",
                             linearInitWithoutAKey());
}

void testLinearBackwardWithoutBiasDoesNotCrash(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_FALSE, .weightInit.key = &initKey},
        &lq);

    tensor_t *fwdIn = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){0.5f, -0.5f});
    tensor_t *propLoss = buildFloatTensor2d(1, 3, (float[]){0.f, 0.f, 0.f});

    layerFunctions[LINEAR].backward(layer, fwdIn, loss, propLoss, NULL);

    float wg00 = ((float *)layer->config->linear->weights->grad->data)[0];
    bool biasIsNull = (layer->config->linear->bias == NULL);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);
    freeLinearLayer(layer);
    freeQuantization(q);
    TEST_ASSERT_TRUE(biasIsNull);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, wg00); /* loss[0,0]*fwdIn[0,0] = 0.5*1 */
}

/* Regression net for the bias-NULL-deref fix bundled into the forward funnel
 * migration (Task 3 of PR1b.2): pre-migration, linearForward unconditionally
 * called getParamFromParameter(cfg->bias) even when bias == NULL, crashing on
 * any bias-disabled layer's forward call — untested until the funnel's
 * operand-array construction forced the NULL check to be added. Mirrors
 * testLinearBackwardWithoutBiasDoesNotCrash's fixture shape. */
void testLinearForwardWithoutBiasDoesNotCrash(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    const rngKey_t initKey = rngFold(rngKeyFromSeed(1u), RNG_TAG_INIT, 0u);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3, .outFeatures = 2, .bias = BIAS_FALSE, .weightInit.key = &initKey},
        &lq);
    layerLoadWeights(layer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, NULL);

    tensor_t *input = buildFloatTensor2d(1, 3, (float[]){0.f, 1.f, 2.f});
    tensor_t *output = buildFloatTensor2d(1, 2, (float[]){0.f, 0.f});

    layerFunctions[LINEAR].forward(layer, input, output, NULL);

    bool biasIsNull = (layer->config->linear->bias == NULL);
    float out0 = ((float *)output->data)[0];
    float out1 = ((float *)output->data)[1];
    freeTensor(output);
    freeTensor(input);
    freeLinearLayer(layer);
    freeQuantization(q);
    TEST_ASSERT_TRUE(biasIsNull);
    TEST_ASSERT_EQUAL_FLOAT(-4.f, out0); /* -1*0 + 2*1 + -3*2 = -4 */
    TEST_ASSERT_EQUAL_FLOAT(-7.f, out1); /*  4*0 + 5*1 + -6*2 = -7 */
}

/* ---- Group-quant PR2 (Task 3): Linear forward with a grouped SYM weight -
 *
 * Same fixture as UnitTestMatmul.c's testMatmulGroupedWeightPerChannelMatchesGold
 * (test/unit/arithmetic/generate_expected_group_matmul.py's "perChannel" case:
 * 2x6 input, 3x6 weight, groupSize=6 == the full reduction length per output
 * channel -- exactly ONE combine per output element). The numbers are
 * duplicated here rather than sharing the generated header across two
 * unrelated test binaries/directories (no existing precedent for that in
 * this tree, see the per-directory CMakeLists.txt files) — kept in sync
 * manually; if the shared fixture ever changes, update both call sites. */
static const int32_t kGroupedAMantissas[] = {1, -2, 3, -1, 2, -3, 2, 1, -1, 3, -2, 1};
static const float kGroupedAScale = 0.5f;
static const int32_t kGroupedWMantissas[] = {4, -3, 2, -1, 5, -2, 1,  2, -4,
                                             3, -1, 2, -2, 3, 1,  -5, 4, -3};
static const float kGroupedWScales[] = {0.019999999552965164f, 0.05000000074505806f,
                                        0.009999999776482582f};
static const int32_t kGroupedBiasMantissas[] = {10, -5, 3};
static const float kGroupedBiasScale = 0.1f;
static const int32_t kGroupedOutMantissas[] = {53, -46, 15, 35, 1, 6};
static const float kGroupedOutScale = 0.02500000037252903f;

/*! Builds the shared grouped-SYM weight/bias/input parameters (borrowed,
 *  caller frees via freeLinearLayer + freeTensor(input) as usual). `q` is
 *  the layer's forward arithmetic (SYM_INT32 or FLOAT32 — independent of the
 *  weight tensor's OWN grouped-SYM storage, exactly like every other SYM
 *  Linear fixture in this file: `q` only derives forwardMath/outputQ). */
static layer_t *buildGroupedFixtureLayer(quantization_t *q, tensor_t **inputOut) {
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 3;
    weightDims[1] = 6;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam =
        initTensor(weightShape, quantizationInitSymGrouped(8, HALF_AWAY, 3, 6), NULL);
    byteConversion((uint8_t *)kGroupedWMantissas, 32, weightsParam->data, 8, 18);
    symQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < 3; g++) {
        weightQC->scales[g] = kGroupedWScales[g];
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t *biasDims = reserveMemory(sizeof(size_t));
    biasDims[0] = 3;
    size_t *biasOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 3; i++) {
        ((int32_t *)biasParam->data)[i] = kGroupedBiasMantissas[i];
    }
    ((symInt32QConfig_t *)biasParam->quantization->qConfig)->scale = kGroupedBiasScale;
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 6;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 12; i++) {
        ((int32_t *)input->data)[i] = kGroupedAMantissas[i];
    }
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = kGroupedAScale;
    *inputOut = input;

    return buildBorrowedLinearLayer(weights, bias, q);
}

/* Compares DEQUANTIZED values, not raw mantissas/scale: the funnel's
 * OUT_WRITE epilogue for a SYM_INT32 target ALWAYS requants through the
 * conversionMatrix diagonal (requantSymInt32Tensor, TensorConversion.c) —
 * same-dtype SYM_INT32->SYM_INT32 is never a raw memmove (ExecuteOp.c's
 * writeOutConversion doc). That cell derives a FRESH scale from a
 * whole-tensor absmax over the kernel's raw output and re-encodes at int12
 * width, so the mantissas/scale captured here differ from
 * matmulSymInt32TensorsGroupedWeight's OWN raw output (kGroupedOutMantissas
 * at kGroupedOutScale, pinned directly in UnitTestMatmul.c's
 * testMatmulGroupedWeightPerChannelMatchesGold) by design — exactly like
 * every OTHER SYM Linear forward test in this file (e.g.
 * testLinearForwardSymInt32 above), which all compare via convertTensor to
 * FLOAT32 + a tolerance rather than raw mantissas. Tolerance: 1.0 *
 * kGroupedOutScale (the kernel's own rescale-combine rounding, see the
 * float-path test below) + one more HALF_AWAY rounding at the requant's
 * fresh scale. requantSymInt32Tensor derives that fresh scale from the
 * absmax over DEQUANTIZED values (mantissa * inScale), not raw mantissas
 * (TensorConversion.c), so requantFreshScale = max(|kGroupedOutMantissas|) *
 * kGroupedOutScale / int12 qMax = 53 * kGroupedOutScale / 2047. */
void testLinearForwardGroupedSymWeights(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedFixtureLayer(testQ, &input);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    linearForward(linearLayer, input, output, NULL);

    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 2;
    outFloatDims[1] = 3;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat, NULL);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)outputFloat->data)[i];
    }

    freeTensor(outputFloat);
    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(testQ);

    const float requantFreshScale = 53.0f / 2047.0f * kGroupedOutScale;
    const float tolerance = 1.0f * kGroupedOutScale + 0.5f * requantFreshScale + 1e-6f;
    for (size_t i = 0; i < 6; i++) {
        float expected = (float)kGroupedOutMantissas[i] * kGroupedOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* FLOAT32 forward path on the SAME grouped-SYM weight/bias/input: the
 * executeOp prologue dequantizes every mismatched operand via
 * convertTensor (SYM(grouped)->FLOAT32 is convertSymTensorToFloat32Tensor,
 * group-aware since Task 2), so this exercises Task 2's grouped dequant
 * against Task 3's grouped SYM_INT32 forward with NO quantization noise of
 * its own (both paths read the identical mantissas/scales — there is no
 * separate "true float" source upstream of either path here).
 *
 * Tolerance derivation (per-channel fixture: exactly 2 rescale-combines per
 * output element -- one weight-group combine, one bias combine):
 *   Each rescaleIntoAccumulatorScale call rounds HALF_AWAY, so its output
 *   (an integer count of sAcc-quanta) differs from the exact real-valued
 *   contribution/sAcc by at most 0.5 (strictly less, except exactly at a
 *   tie). Two independent combines contribute per output element, so the
 *   SYM_INT32 path's integer accumulator differs from the exact real sum by
 *   at most 0.5 + 0.5 = 1.0, i.e. the float output (acc * sAcc) differs from
 *   the true value by at most 1.0 * sAcc = 1.0 * kGroupedOutScale. The
 *   FLOAT32 path computes that true value directly (float32 dot product, no
 *   combine rounding at all) up to float32 arithmetic noise, which is
 *   ~1e-7 relative here and negligible next to the quantization-rounding
 *   term above. Bound used: 1.0 * kGroupedOutScale (a small explicit
 *   headroom -- 1e-6 absolute -- covers that residual float32 noise without
 *   loosening the combine-rounding argument itself). */
void testLinearForwardGroupedFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedFixtureLayer(floatQ, &input);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    linearForward(linearLayer, input, output, NULL);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    const float tolerance = 1.0f * kGroupedOutScale + 1e-6f;
    for (size_t i = 0; i < 6; i++) {
        float expected = (float)kGroupedOutMantissas[i] * kGroupedOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* ---- Group-quant PR3 (Task 1): Linear backward dx with a grouped SYM
 * weight. Hand-duplicated from generate_expected_group_matmul.py's
 * DxPerChannel fixture (same weight mantissas/scales as kGroupedW* above;
 * seeded row-distinct pseudo-random loss) — same manual-sync rule as the
 * kGrouped* literals, see the comment there. */
static const int32_t kGroupedDxLossMantissas[] = {33, -12, -2, -29, 13, 40};
static const float kGroupedDxLossScale = 0.5f;
static const int32_t kGroupedDxOutMantissas[] = {42,  -65, 74,  -47, 76,  -49,
                                                 -49, 85,  -67, 11,  -39, 25};
static const float kGroupedDxOutScale = 0.02500000037252903f;

/*! Shared dx-fixture plumbing for the two backward tests below: builds the
 *  grouped layer (via buildGroupedFixtureLayer), the SYM_INT32 loss and the
 *  FLOAT32 propLoss wire, then runs linearBackward. The layer is FROZEN: the
 *  borrowed fixture parameters carry no grad buffers, and dx is the ONLY
 *  backward op that consumes the (grouped) weight tensor — the weight/bias
 *  grad ops take {loss, fwdIn}/{loss} and are pinned by the existing SYM
 *  backward tests — so freezing scopes these tests to the dx wire (#380:
 *  frozen layers still propagate loss). `propLossQ` is overridden to the
 *  FLOAT32 wire config, mirroring linearInitConfig's backwardMath/propLossQ
 *  split (SYM compute, FLOAT32-stored propLoss wire). */
static void runGroupedDxBackward(quantization_t *mathQ, quantization_t *wireQ,
                                 float capturedOut[12]) {
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedFixtureLayer(mathQ, &input);
    linearLayer->config->linear->propLossQ = wireQ;
    linearLayer->config->linear->frozen = true;

    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 6; i++) {
        ((int32_t *)loss->data)[i] = kGroupedDxLossMantissas[i];
    }
    ((symInt32QConfig_t *)loss->quantization->qConfig)->scale = kGroupedDxLossScale;

    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 2;
    propLossDims[1] = 6;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    linearBackward(linearLayer, input, loss, propLoss, NULL);

    for (size_t i = 0; i < 12; i++) {
        capturedOut[i] = ((float *)propLoss->data)[i];
    }

    freeTensor(propLoss);
    freeTensor(loss);
    freeLinearLayer(linearLayer);
    freeTensor(input);
}

/* Exact FLOAT32-wire compare against the RAW dx gold (the Conv1d grouped
 * forward tests' PR2-ratified design, see
 * testConv1dForwardGroupedPerChannelMatchesGold's comment): propLossMath
 * stays SYM_INT32, but the propLoss WIRE is FLOAT32, so the executeOp
 * OUT_WRITE epilogue takes the SYM_INT32->FLOAT32 conversionMatrix cell —
 * one exact `(float)mantissa * scale` per element, bit-for-bit the same
 * float32 formula this expected-value loop computes — instead of the
 * absmax-fresh-scale SYM requant whose generous tolerance could mask a
 * wrong-but-sound s_acc. Any divergence in the kernel's internal arithmetic
 * (wrong group binding, dropped combine, wrong s_acc) changes the compared
 * value measurably. */
void testLinearBackwardGroupedSymWeightsDxMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *wireQ = quantizationInitFloat();
    float captured[12];
    runGroupedDxBackward(testQ, wireQ, captured);
    freeQuantization(wireQ);
    freeQuantization(testQ);

    for (size_t i = 0; i < 12; i++) {
        float expected = (float)kGroupedDxOutMantissas[i] * kGroupedDxOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* FLOAT32 dx path on the SAME grouped-SYM weight and loss: the executeOp
 * prologue dequantizes both operands via convertTensor (grouped weight
 * through the group-aware SYM->FLOAT32 cell), then the float matmul computes
 * the reference value directly with NO combine rounding.
 *
 * Tolerance derivation (per-channel dx fixture): the dx reduction visits
 * outFeatures = 3 weight rows per output element, each row its OWN group
 * (per-channel), so the SYM_INT32 path folds C = 3 rescale-combines per
 * element (no bias in dx). Each combine rounds HALF_AWAY once: <= 0.5 quanta
 * of s_acc error, so |int path - true| <= 0.5 * C * s_acc = 1.5 *
 * kGroupedDxOutScale = 1.5 * 0.025 = 0.0375. The float path here IS that
 * true value up to ~1e-7 relative float32 noise — covered by the 1e-6
 * absolute headroom. */
void testLinearBackwardGroupedDxFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    float captured[12];
    runGroupedDxBackward(floatQ, floatQ, captured);
    freeQuantization(floatQ);

    const float tolerance = 1.5f * kGroupedDxOutScale + 1e-6f;
    for (size_t i = 0; i < 12; i++) {
        float expected = (float)kGroupedDxOutMantissas[i] * kGroupedDxOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* ---- Group-quant PR4 (Task 3): Linear with a grouped ASYM weight ---------
 *
 * group-quant D5 made executable: the grouped ASYM forward/dx IS the grouped SYM
 * forward/dx on SHIFTED mantissas. The funnel's ASYM grouped-unpack arm
 * (ExecuteOp.c) zero-extends the packed codes and subtracts each element's
 * group zeroPoint (code - zp[g]) -- from there on the compute path is
 * IDENTICAL to the grouped SYM one: the layer passes a symQConfig-shaped
 * VIEW of the asym config (scales BORROWED, numGroups/groupSize copied) as
 * weightGroups, and matmulSymInt32TensorsGroupedWeight runs unchanged. The
 * golds in expected_linear_grouped_asym.h are therefore the EXISTING
 * symmetric references (matmul_grouped_ref / matmul_grouped_dx_ref) fed with
 * mantissas = codes - zps and the asym scales -- the generator asserts that
 * shifted-mantissa equivalence in-place (self-check iii). */

/*! Grouped-ASYM twin of buildGroupedFixtureLayer: 3x6 weight with packed
 *  qBits=8 codes + per-group scales AND code-domain zeroPoints from the
 *  generated fixture; SYM_INT32 bias and input unchanged in kind. */
static layer_t *buildGroupedAsymFixtureLayer(quantization_t *q, tensor_t **inputOut) {
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 3;
    weightDims[1] = 6;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(
        weightShape,
        quantizationInitAsymGrouped((uint8_t)kLinAsymQBits, HALF_AWAY, (size_t)kLinAsymNumGroups,
                                    (size_t)kLinAsymGroupSize),
        NULL);
    byteConversion((uint8_t *)kLinAsymWCodes, 32, weightsParam->data, (size_t)kLinAsymQBits,
                   kLinAsymWCodes_len);
    asymQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < (size_t)kLinAsymNumGroups; g++) {
        weightQC->scales[g] = kLinAsymWScales[g];
        weightQC->zeroPoints[g] = (uint16_t)kLinAsymWZps[g];
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t *biasDims = reserveMemory(sizeof(size_t));
    biasDims[0] = 3;
    size_t *biasOrder = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 3; i++) {
        ((int32_t *)biasParam->data)[i] = kLinAsymBiasMantissas[i];
    }
    ((symInt32QConfig_t *)biasParam->quantization->qConfig)->scale = kLinAsymBiasScale;
    parameter_t *bias = parameterInit(biasParam, NULL);

    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 2;
    inputDims[1] = 6;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 12; i++) {
        ((int32_t *)input->data)[i] = kLinAsymAMantissas[i];
    }
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = kLinAsymAScale;
    *inputOut = input;

    return buildBorrowedLinearLayer(weights, bias, q);
}

/* Exact FLOAT32-wire compare against the RAW grouped gold (the PR2 Conv1d
 * grouped-test design, see testConv1dForwardGroupedPerChannelMatchesGold's
 * comment in UnitTestConv1d.c): forwardMath stays SYM_INT32 but the output
 * wire is FLOAT32, so the OUT_WRITE epilogue is one exact (float)mantissa *
 * scale per element -- bit-for-bit the formula this expected loop computes --
 * and ANY divergence (wrong zp shift, wrong per-group scale, wrong s_acc)
 * changes the compared value measurably. */
void testLinearForwardGroupedAsymWeightsMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedAsymFixtureLayer(testQ, &input);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    linearForward(linearLayer, input, output, NULL);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(testQ);

    for (size_t i = 0; i < 6; i++) {
        float expected = (float)kLinAsymOutMantissas[i] * kLinAsymOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* FLOAT32 forward path on the SAME grouped-ASYM weight: the executeOp
 * prologue dequantizes the declared operand via the group-aware
 * ASYM->FLOAT32 cell (convertAsymTensorToFloatTensor, Task 2), gated by the
 * FLOAT32-arm carrier check exactly like grouped SYM. Tolerance derivation is
 * testLinearForwardGroupedFloatPathAgreesWithinTolerance's verbatim
 * (per-channel fixture, 2 rescale-combines per output element -> 1.0 quanta
 * of the accumulator scale, + 1e-6 float32-noise headroom). */
void testLinearForwardGroupedAsymFloatPathAgreesWithinTolerance(void) {
    quantization_t *floatQ = quantizationInitFloat();
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedAsymFixtureLayer(floatQ, &input);

    size_t *outputDims = reserveMemory(2 * sizeof(size_t));
    outputDims[0] = 2;
    outputDims[1] = 3;
    size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 2, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    linearForward(linearLayer, input, output, NULL);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)output->data)[i];
    }

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    const float tolerance = 1.0f * kLinAsymOutScale + 1e-6f;
    for (size_t i = 0; i < 6; i++) {
        float expected = (float)kLinAsymOutMantissas[i] * kLinAsymOutScale;
        TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, captured[i]);
    }
}

/* Equal-scales-equal-zp bit-identity twin: a grouped ASYM weight whose groups
 * all share ONE scale AND ONE zeroPoint represents exactly the same affine
 * grid as the per-tensor ASYM spelling of it -- so the grouped path (funnel
 * unpack-shift + grouped kernel) and the scalar path (per-tensor
 * ASYM->SYM_INT32 conversion cell + scalar kernel) must produce BIT-IDENTICAL
 * output. Exactness argument (the Conv1d equal-scales twin's, adapted): the
 * common scale 0.25f and the input scale 0.5f are powers of two, so s_acc =
 * 0.125f and every grouped combine's paramScale are the SAME float32 value
 * and rescaleIntoAccumulatorScale(partial, s, s) is an exact round trip
 * (pure exponent shifts) -- the grouped kernel's raw output equals the
 * scalar kernel's raw MAC; the bias seed rescale is the identical
 * rescaleIntoAccumulatorScale(bias, 0.1f, 0.125f, HALF_AWAY) call on both
 * sides; both FLOAT32 wires then dequantize identical (mantissa, scale)
 * pairs. Codes are the PR2 kGroupedWMantissas shifted by the common zp
 * (mantissa + 100, all within [95, 105] of the 8-bit code domain). */
void testLinearForwardGroupedAsymEqualScalesEqualZpBitIdenticalToPerTensor(void) {
    enum { kCommonZp = 100 };
    const float commonScale = 0.25f;
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);

    int32_t codes[18];
    for (size_t i = 0; i < 18; i++) {
        codes[i] = kGroupedWMantissas[i] + kCommonZp;
    }

    /* Grouped side: 3 groups of 6, every group {0.25f, zp 100}. */
    size_t *gwDims = reserveMemory(2 * sizeof(size_t));
    gwDims[0] = 3;
    gwDims[1] = 6;
    size_t *gwOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, gwOrder);
    shape_t *gwShape = reserveMemory(sizeof(shape_t));
    setShape(gwShape, gwDims, 2, gwOrder);
    tensor_t *groupedWParam =
        initTensor(gwShape, quantizationInitAsymGrouped(8, HALF_AWAY, 3, 6), NULL);
    byteConversion((uint8_t *)codes, 32, groupedWParam->data, 8, 18);
    asymQConfig_t *groupedWQC = groupedWParam->quantization->qConfig;
    for (size_t g = 0; g < 3; g++) {
        groupedWQC->scales[g] = commonScale;
        groupedWQC->zeroPoints[g] = kCommonZp;
    }
    parameter_t *groupedWeights = parameterInit(groupedWParam, NULL);

    /* Per-tensor side: the SAME codes under one {0.25f, zp 100} grid. */
    size_t *swDims = reserveMemory(2 * sizeof(size_t));
    swDims[0] = 3;
    swDims[1] = 6;
    size_t *swOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, swOrder);
    shape_t *swShape = reserveMemory(sizeof(shape_t));
    setShape(swShape, swDims, 2, swOrder);
    tensor_t *scalarWParam = initTensor(swShape, quantizationInitAsym(8, HALF_AWAY), NULL);
    byteConversion((uint8_t *)codes, 32, scalarWParam->data, 8, 18);
    asymQConfig_t *scalarWQC = scalarWParam->quantization->qConfig;
    scalarWQC->scales[0] = commonScale;
    scalarWQC->zeroPoints[0] = kCommonZp;
    parameter_t *scalarWeights = parameterInit(scalarWParam, NULL);

    float groupedOut[6];
    float scalarOut[6];
    parameter_t *weightsPerRun[2] = {groupedWeights, scalarWeights};
    float *outPerRun[2] = {groupedOut, scalarOut};
    for (size_t run = 0; run < 2; run++) {
        size_t *biasDims = reserveMemory(sizeof(size_t));
        biasDims[0] = 3;
        size_t *biasOrder = reserveMemory(sizeof(size_t));
        setOrderOfDimsForNewTensor(1, biasOrder);
        shape_t *biasShape = reserveMemory(sizeof(shape_t));
        setShape(biasShape, biasDims, 1, biasOrder);
        tensor_t *biasParam = initTensor(biasShape, quantizationInitSymInt32(HALF_AWAY), NULL);
        for (size_t i = 0; i < 3; i++) {
            ((int32_t *)biasParam->data)[i] = kGroupedBiasMantissas[i];
        }
        ((symInt32QConfig_t *)biasParam->quantization->qConfig)->scale = kGroupedBiasScale;
        parameter_t *bias = parameterInit(biasParam, NULL);

        size_t *inputDims = reserveMemory(2 * sizeof(size_t));
        inputDims[0] = 2;
        inputDims[1] = 6;
        size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, inputOrder);
        shape_t *inputShape = reserveMemory(sizeof(shape_t));
        setShape(inputShape, inputDims, 2, inputOrder);
        tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
        for (size_t i = 0; i < 12; i++) {
            ((int32_t *)input->data)[i] = kGroupedAMantissas[i];
        }
        ((symInt32QConfig_t *)input->quantization->qConfig)->scale = kGroupedAScale;

        layer_t *linearLayer = buildBorrowedLinearLayer(weightsPerRun[run], bias, testQ);

        size_t *outputDims = reserveMemory(2 * sizeof(size_t));
        outputDims[0] = 2;
        outputDims[1] = 3;
        size_t *outputOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, outputOrder);
        shape_t *outputShape = reserveMemory(sizeof(shape_t));
        setShape(outputShape, outputDims, 2, outputOrder);
        tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

        linearForward(linearLayer, input, output, NULL);

        for (size_t i = 0; i < 6; i++) {
            outPerRun[run][i] = ((float *)output->data)[i];
        }

        freeLinearLayer(linearLayer); /* frees this run's weights + bias */
        freeTensor(output);
        freeTensor(input);
    }
    freeQuantization(testQ);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(scalarOut, groupedOut, 6);
}

/* dx twin of testLinearBackwardGroupedSymWeightsDxMatchesGold on the grouped
 * ASYM weight: propLossMath SYM_INT32, FLOAT32 propLoss wire -> exact
 * compare against matmul_grouped_dx_ref fed with the SHIFTED mantissas (the
 * strided grouped core: the dx reduction visits weight storage strided by
 * inFeatures, per-element group binding). Frozen layer scopes the test to
 * the dx wire (borrowed params carry no grads). */
void testLinearBackwardGroupedAsymDxMatchesGold(void) {
    quantization_t *testQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *wireQ = quantizationInitFloat();
    tensor_t *input = NULL;
    layer_t *linearLayer = buildGroupedAsymFixtureLayer(testQ, &input);
    linearLayer->config->linear->propLossQ = wireQ;
    linearLayer->config->linear->frozen = true;

    size_t *lossDims = reserveMemory(2 * sizeof(size_t));
    lossDims[0] = 2;
    lossDims[1] = 3;
    size_t *lossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, lossOrder);
    shape_t *lossShape = reserveMemory(sizeof(shape_t));
    setShape(lossShape, lossDims, 2, lossOrder);
    tensor_t *loss = initTensor(lossShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < 6; i++) {
        ((int32_t *)loss->data)[i] = kLinAsymDxLossMantissas[i];
    }
    ((symInt32QConfig_t *)loss->quantization->qConfig)->scale = kLinAsymDxLossScale;

    size_t *propLossDims = reserveMemory(2 * sizeof(size_t));
    propLossDims[0] = 2;
    propLossDims[1] = 6;
    size_t *propLossOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 2, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitFloat(), NULL);

    linearBackward(linearLayer, input, loss, propLoss, NULL);

    float captured[12];
    for (size_t i = 0; i < 12; i++) {
        captured[i] = ((float *)propLoss->data)[i];
    }

    freeTensor(propLoss);
    freeTensor(loss);
    freeLinearLayer(linearLayer);
    freeTensor(input);
    freeQuantization(wireQ);
    freeQuantization(testQ);

    for (size_t i = 0; i < 12; i++) {
        float expected = (float)kLinAsymDxOutMantissas[i] * kLinAsymDxOutScale;
        TEST_ASSERT_EQUAL_FLOAT(expected, captured[i]);
    }
}

/* ---- BFP epic PR2 (Task 7): Linear ARITH_BFP forward arm -----------------
 *
 * Every test here hand-sets forwardMath to {ARITH_BFP, HALF_AWAY} POST-build:
 * the layer builders take a single FLOAT32 `q`, and pinning the slot keeps the
 * arm selection independent of the derivation (which since Task 9 yields
 * ARITH_BFP for a BFP config anyway). Weights reach BFP storage the documented
 * user way: FLOAT32 init + requantizeTensorInPlace
 * (docs/conventions/arithmetic-bfp.md). */

static tensor_t *makeFloatTensor(size_t const *dims, size_t numDims, float const *data) {
    size_t *ownedDims = reserveMemory(numDims * sizeof(size_t));
    memcpy(ownedDims, dims, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, ownedDims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (data != NULL) {
        tensorFillFromFloatBuffer(t, data, calcNumberOfElementsByTensor(t));
    }
    return t;
}

/*! Native-fixture layer: FLOAT32 weights [3,6] from the generated fixture,
 *  requantized IN PLACE to the fixture's grouped BFP config; FLOAT32 `q`
 *  covers the wires; forwardMath hand-set post-build (pinned, see above). */
static layer_t *buildBfpLinearLayer(bool withBias, quantization_t *floatQ) {
    size_t weightDims[] = {(size_t)kLinBfpOutFeatures, (size_t)kLinBfpInFeatures};
    tensor_t *weightsParam = makeFloatTensor(weightDims, 2, kLinBfpWValues);
    quantization_t *bfpTemplate =
        quantizationInitBfpGrouped((uint8_t)kLinBfpWMantissaBits, (uint8_t)kLinBfpWExponentBits,
                                   HALF_AWAY, (size_t)kLinBfpWNumGroups, (size_t)kLinBfpWGroupSize);
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    freeQuantization(bfpTemplate);
    parameter_t *weights = parameterInit(weightsParam, NULL);

    parameter_t *bias = NULL;
    if (withBias) {
        size_t biasDims[] = {(size_t)kLinBfpOutFeatures};
        bias = parameterInit(makeFloatTensor(biasDims, 1, kLinBfpBiasValues), NULL);
    }

    layer_t *layer = buildBorrowedLinearLayer(weights, bias, floatQ);
    layer->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    return layer;
}

/* Native ARITH_BFP forward against the staged-input gold: the funnel borrows
 * the BFP-stored weights (unpack only) and stages the FLOAT32 input
 * PER-TENSOR at the WEIGHTS' widths (PR2-Decision 1; m=6 here, deliberately != 8
 * and LOSSY for the fixture's X values so a hardcoded staging width diverges
 * observably). generate_expected_bfp_layer_forward.py mirrors exactly that
 * staging. The FLOAT32 wire is bit-exact (ARITH_BFP raw is FLOAT32, BFP D7; the
 * OUT_WRITE FLOAT32->FLOAT32 epilogue is a memmove), so the compare is
 * EQUAL_MEMORY, not a tolerance. Output prefilled with a sentinel: every
 * element must be kernel-written. */
void testLinearForwardBfpNativeMatchesKernelGold(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *linearLayer = buildBfpLinearLayer(false, floatQ);

    size_t inputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpInFeatures};
    tensor_t *input = makeFloatTensor(inputDims, 2, kLinBfpXValues);
    size_t outputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpOutFeatures};
    tensor_t *output = makeFloatTensor(outputDims, 2, NULL);
    for (size_t i = 0; i < kLinBfpExpectedNoBias_len; i++) {
        ((float *)output->data)[i] = 777.0f;
    }

    linearForward(linearLayer, input, output, NULL);

    float captured[6]; /* kLinBfpBatch * kLinBfpOutFeatures */
    memcpy(captured, output->data, sizeof(captured));

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_MEMORY(kLinBfpExpectedNoBias, captured, sizeof(captured));
}

/* PR2-Decision 2: a FLOAT32-stored bias under ARITH_BFP is staged through the
 * SAME per-tensor-at-weights'-widths template as the input (uniform staging
 * rule) -- the layer must not reject it and the result must match the gold
 * whose bias was quantized per-tensor at m=6. */
void testLinearForwardBfpStagesFloat32Bias(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *linearLayer = buildBfpLinearLayer(true, floatQ);

    size_t inputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpInFeatures};
    tensor_t *input = makeFloatTensor(inputDims, 2, kLinBfpXValues);
    size_t outputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpOutFeatures};
    tensor_t *output = makeFloatTensor(outputDims, 2, NULL);
    for (size_t i = 0; i < kLinBfpExpectedWithBias_len; i++) {
        ((float *)output->data)[i] = 777.0f;
    }

    linearForward(linearLayer, input, output, NULL);

    float captured[6]; /* kLinBfpBatch * kLinBfpOutFeatures */
    memcpy(captured, output->data, sizeof(captured));

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_MEMORY(kLinBfpExpectedWithBias, captured, sizeof(captured));
}

/* Layer-level power-of-two twin (precedent: the equal-scales-equal-zp twin
 * above; kernel-level sibling UnitTestMatmul.c's
 * testMatmulBfpPowerOfTwoBitIdenticalToGroupedSym): the SAME mantissas under
 * value-identical power-of-two grids must produce BIT-IDENTICAL FLOAT32
 * wires through BOTH forward arms. SYM side: grouped SYM weights (qBits 8,
 * scales all 0.25f) + SYM_INT32 input (scale 0.5f) under ARITH_SYM_INT32 --
 * per output element ONE tail combine, rescaleIntoAccumulatorScale(partial,
 * 0.125f, 0.125f) is an exact round trip, and the FLOAT32 wire dequant is
 * one exact (float)mantissa * 0.125f. BFP side: the same values as FLOAT32
 * params/input; requantize derives per-group absmax <= 1.25 -> stored
 * exponent 2^-6, weight codes w_m*16 (exact); per-tensor staging at m=8
 * derives absmax 1.5 -> 2^-6, input codes a_m*32 (exact); the single tail
 * fold is ldexpf(512*partial, -12) = partial*0.125f bit-for-bit (partials
 * < 2^24). NO bias: the SYM bias seed would rescale through the
 * non-power-of-two 0.1f grid and break exactness. */
void testLinearForwardBfpPowerOfTwoBitIdenticalToGroupedSymLayer(void) {
    float symOut[6];
    float bfpOut[6];

    { /* SYM side: grouped-SYM weights, SYM_INT32 input, no bias. */
        quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
        size_t *weightDims = reserveMemory(2 * sizeof(size_t));
        weightDims[0] = 3;
        weightDims[1] = 6;
        size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, weightOrder);
        shape_t *weightShape = reserveMemory(sizeof(shape_t));
        setShape(weightShape, weightDims, 2, weightOrder);
        tensor_t *weightsParam =
            initTensor(weightShape, quantizationInitSymGrouped(8, HALF_AWAY, 3, 6), NULL);
        byteConversion((uint8_t *)kGroupedWMantissas, 32, weightsParam->data, 8, 18);
        symQConfig_t *weightQC = weightsParam->quantization->qConfig;
        for (size_t g = 0; g < 3; g++) {
            weightQC->scales[g] = 0.25f;
        }
        parameter_t *weights = parameterInit(weightsParam, NULL);

        size_t *inputDims = reserveMemory(2 * sizeof(size_t));
        inputDims[0] = 2;
        inputDims[1] = 6;
        size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
        setOrderOfDimsForNewTensor(2, inputOrder);
        shape_t *inputShape = reserveMemory(sizeof(shape_t));
        setShape(inputShape, inputDims, 2, inputOrder);
        tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
        for (size_t i = 0; i < 12; i++) {
            ((int32_t *)input->data)[i] = kGroupedAMantissas[i];
        }
        ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 0.5f;

        layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, symQ);

        size_t outputDims[] = {2, 3};
        tensor_t *output = makeFloatTensor(outputDims, 2, NULL);

        linearForward(linearLayer, input, output, NULL);
        memcpy(symOut, output->data, sizeof(symOut));

        freeLinearLayer(linearLayer);
        freeTensor(output);
        freeTensor(input);
        freeQuantization(symQ);
    }

    { /* BFP side: the SAME values as FLOAT32, weights requantized to grouped
       * BFP {3,6} m=8, forwardMath hand-set (pinned). */
        float wVals[18];
        for (size_t i = 0; i < 18; i++) {
            wVals[i] = (float)kGroupedWMantissas[i] * 0.25f;
        }
        float xVals[12];
        for (size_t i = 0; i < 12; i++) {
            xVals[i] = (float)kGroupedAMantissas[i] * 0.5f;
        }

        quantization_t *floatQ = quantizationInitFloat();
        size_t weightDims[] = {3, 6};
        tensor_t *weightsParam = makeFloatTensor(weightDims, 2, wVals);
        quantization_t *bfpTemplate = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 3, 6);
        requantizeTensorInPlace(weightsParam, bfpTemplate);
        freeQuantization(bfpTemplate);
        parameter_t *weights = parameterInit(weightsParam, NULL);

        layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, floatQ);
        linearLayer->config->linear->forwardMath =
            (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

        size_t inputDims[] = {2, 6};
        tensor_t *input = makeFloatTensor(inputDims, 2, xVals);
        size_t outputDims[] = {2, 3};
        tensor_t *output = makeFloatTensor(outputDims, 2, NULL);

        linearForward(linearLayer, input, output, NULL);
        memcpy(bfpOut, output->data, sizeof(bfpOut));

        freeLinearLayer(linearLayer);
        freeTensor(output);
        freeTensor(input);
        freeQuantization(floatQ);
    }

    TEST_ASSERT_EQUAL_MEMORY(symOut, bfpOut, sizeof(symOut));
}

/* Rule 1 fail-fast: ARITH_BFP forward with weights left FLOAT32 dies with
 * the guided requantizeTensorInPlace message (exit(1)) in the LAYER, before
 * the funnel. Harness note: DeathTest.h asserts the exit CODE only; the
 * dropped-check mutation is still caught because an unguarded arm derefs the
 * FLOAT32 weights' NULL qConfig (child dies by SIGNAL -> WIFEXITED false ->
 * this assert fails) -- verified in the Task 7 mutation pass. */
void testLinearForwardBfpRejectsFloat32Weights(void) {
    quantization_t *floatQ = quantizationInitFloat();
    size_t weightDims[] = {(size_t)kLinBfpOutFeatures, (size_t)kLinBfpInFeatures};
    tensor_t *weightsParam = makeFloatTensor(weightDims, 2, kLinBfpWValues);
    parameter_t *weights = parameterInit(weightsParam, NULL);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, floatQ);
    linearLayer->config->linear->forwardMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t inputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpInFeatures};
    tensor_t *input = makeFloatTensor(inputDims, 2, kLinBfpXValues);
    size_t outputDims[] = {(size_t)kLinBfpBatch, (size_t)kLinBfpOutFeatures};
    tensor_t *output = makeFloatTensor(outputDims, 2, NULL);

    ASSERT_EXITS_WITH_FAILURE(linearForward(linearLayer, input, output, NULL));

    freeLinearLayer(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(floatQ);
}

/* Native ARITH_BFP weightGrad on all-BFP-stored operands (the funnel's borrow
 * arm end-to-end): weights BFP-stored per the backward rule-1 mirror (the
 * width anchor), loss + forwardInput requantized per-tensor m=8/e=8 from
 * exactly-representable values, grad target FLOAT32 (#261). Batch 1, so
 * dW[o][i] = loss[o] * x[i] with every value exact at m=8 -- the kernel's
 * single-segment fold is exact float, asserted against the literal products.
 * Also pins the kernel's transpose-restore: the funnel scratch ALIASES the
 * caller's shape_t (setTensorValuesForConversion), so a dropped restore would
 * leave the caller's loss transposed after linearBackward. */
void testLinearBackwardBfpWeightGradNativeMatchesFloatReference(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpTemplate = quantizationInitBfp(8, 8, HALF_AWAY);
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam =
        makeFloatTensor(weightDims, 2, (float[]){0.5f, -1.f, 2.f, 1.f, 0.25f, -0.75f});
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    parameter_t *weights = parameterInit(weightsParam, gradInitFloat(weightsParam, NULL));
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, floatQ);
    linearLayer->config->linear->weightGradMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t fwdDims[] = {1, 3};
    tensor_t *forwardInput = makeFloatTensor(fwdDims, 2, (float[]){1.f, 2.f, 3.f});
    requantizeTensorInPlace(forwardInput, bfpTemplate);
    size_t lossDims[] = {1, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f});
    requantizeTensorInPlace(loss, bfpTemplate);
    freeQuantization(bfpTemplate);

    linearBackward(linearLayer, forwardInput, loss, NULL, NULL);

    float captured[6];
    memcpy(captured, weights->grad->data, sizeof(captured));
    size_t lossOrder0 = loss->shape->orderOfDimensions[0];
    size_t lossOrder1 = loss->shape->orderOfDimensions[1];

    freeLinearLayer(linearLayer);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* dW = loss^T @ x, batch 1: row o is loss[o] * {1, 2, 3}. */
    TEST_ASSERT_EQUAL_FLOAT(-4.f * 1.f, captured[0]);
    TEST_ASSERT_EQUAL_FLOAT(-4.f * 2.f, captured[1]);
    TEST_ASSERT_EQUAL_FLOAT(-4.f * 3.f, captured[2]);
    TEST_ASSERT_EQUAL_FLOAT(3.f * 1.f, captured[3]);
    TEST_ASSERT_EQUAL_FLOAT(3.f * 2.f, captured[4]);
    TEST_ASSERT_EQUAL_FLOAT(3.f * 3.f, captured[5]);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(
        0, lossOrder0,
        "the weightGrad kernel must restore the caller's loss view (transpose "
        "dance on funnel scratch that aliases the caller's shape_t)");
    TEST_ASSERT_EQUAL_size_t(1, lossOrder1);
}

/* Native ARITH_BFP biasGrad in the mixed-slot scenario from the PR2 trio:
 * weightGrad pinned ARITH_FLOAT32 runs first (fake-quant dequant of the BFP
 * loss), then the ARITH_BFP biasGrad consumes the BFP-stored loss through the
 * borrow arm. Batch 2 so the per-feature sum is real (trio-shape deviation:
 * forwardInput grows to [2,3] to keep the FLOAT32 weightGrad op's loss^T @ x
 * shape-consistent with the [2,2] loss); bGrad[f] = sum_n loss[n][f], every
 * value exact at m=8. Weights are BFP-stored per the backward rule-1 mirror. */
void testLinearBackwardBfpBiasGradNativeSumsLoss(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpTemplate = quantizationInitBfp(8, 8, HALF_AWAY);
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam =
        makeFloatTensor(weightDims, 2, (float[]){0.5f, -1.f, 2.f, 1.f, 0.25f, -0.75f});
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    parameter_t *weights = parameterInit(weightsParam, gradInitFloat(weightsParam, NULL));
    size_t biasDims[] = {2};
    tensor_t *biasParam = makeFloatTensor(biasDims, 1, (float[]){-1.f, 3.f});
    parameter_t *bias = parameterInit(biasParam, gradInitFloat(biasParam, NULL));
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, floatQ);
    linearLayer->config->linear->weightGradMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    linearLayer->config->linear->biasGradMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t fwdDims[] = {2, 3};
    tensor_t *forwardInput = makeFloatTensor(fwdDims, 2, (float[]){1.f, 2.f, 3.f, -1.f, 0.5f, 2.f});
    size_t lossDims[] = {2, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f, 2.f, -1.f});
    requantizeTensorInPlace(loss, bfpTemplate);
    freeQuantization(bfpTemplate);

    linearBackward(linearLayer, forwardInput, loss, NULL, NULL);

    float captured[2];
    memcpy(captured, bias->grad->data, sizeof(captured));

    freeLinearLayer(linearLayer);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* bGrad[f] = batch sum over loss column f: {-4 + 2, 3 + (-1)}. */
    TEST_ASSERT_EQUAL_FLOAT(-4.f + 2.f, captured[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f + -1.f, captured[1]);
}

/* Grouped-loss twin of the test above -- the ONLY fixture that reaches the
 * biasGrad core's group-crossing fold branch (the per-tensor twin never
 * leaves group 0). Same loss values, but stored grouped {numGroups=2,
 * groupSize=2}: group 0 = rows {-4, 3} (absmax 4), group 1 = rows {2, -1}
 * (absmax 2) -> DISTINCT derived exponents, and each feature's stride-2 walk
 * crosses group 0 -> 1 between n=0 and n=1, so the fold binds
 * exponents[currentGroup] per segment and resets the partial. All values
 * m=8-exact under their own group exponent, so the expected sums are the
 * plain float sums: db[0] = -4 + 2 = -2, db[1] = 3 + (-1) = 2. */
void testLinearBackwardBfpBiasGradGroupedLossFoldsPerGroup(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpTemplate = quantizationInitBfp(8, 8, HALF_AWAY);
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam =
        makeFloatTensor(weightDims, 2, (float[]){0.5f, -1.f, 2.f, 1.f, 0.25f, -0.75f});
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    freeQuantization(bfpTemplate);
    parameter_t *weights = parameterInit(weightsParam, gradInitFloat(weightsParam, NULL));
    size_t biasDims[] = {2};
    tensor_t *biasParam = makeFloatTensor(biasDims, 1, (float[]){-1.f, 3.f});
    parameter_t *bias = parameterInit(biasParam, gradInitFloat(biasParam, NULL));
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, floatQ);
    linearLayer->config->linear->weightGradMath =
        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    linearLayer->config->linear->biasGradMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t fwdDims[] = {2, 3};
    tensor_t *forwardInput = makeFloatTensor(fwdDims, 2, (float[]){1.f, 2.f, 3.f, -1.f, 0.5f, 2.f});
    quantization_t *groupedBfpTemplate = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    size_t lossDims[] = {2, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f, 2.f, -1.f});
    requantizeTensorInPlace(loss, groupedBfpTemplate);
    freeQuantization(groupedBfpTemplate);

    linearBackward(linearLayer, forwardInput, loss, NULL, NULL);

    float captured[2];
    memcpy(captured, bias->grad->data, sizeof(captured));

    freeLinearLayer(linearLayer);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* db[f] = fold(group 0 partial) + fold(group 1 partial), per feature. */
    TEST_ASSERT_EQUAL_FLOAT(-4.f + 2.f, captured[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.f + -1.f, captured[1]);
}

/* linearCalcBiasGradsBfp is a PUBLIC symbol: without a dtype guard a
 * FLOAT32-stored loss would be reinterpreted as int32 mantissa codes (silent
 * wrong arithmetic). Mirrors matmulBfpTensors' operand fail-fast
 * ("unpacked scratch form"). */
void testLinearCalcBiasGradsBfpRejectsFloat32Loss(void) {
    size_t lossDims[] = {1, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f});
    size_t biasDims[] = {2};
    tensor_t *biasGrad = makeFloatTensor(biasDims, 1, NULL);

    ASSERT_EXITS_WITH_FAILURE(linearCalcBiasGradsBfp(loss, biasGrad));

    freeTensor(biasGrad);
    freeTensor(loss);
}

/* Native ARITH_BFP propLoss in the frozen-layer scenario from the PR2 trio:
 * frozen skips both grad ops, so backward reaches the propLoss op directly
 * with {loss, weights} BOTH BFP-stored (borrow arm). dx = loss @ W with W in
 * RAW [outF, inF] storage (Task 1's strided-walk kernel pin) into a FLOAT32
 * dx wire; every value exact at m=8, asserted against the literal float
 * expressions. */
void testLinearBackwardBfpPropLossNativeMatchesFloatReference(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpTemplate = quantizationInitBfp(8, 8, HALF_AWAY);
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam =
        makeFloatTensor(weightDims, 2, (float[]){0.5f, -1.f, 2.f, 1.f, 0.25f, -0.75f});
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    parameter_t *weights = parameterInit(weightsParam, NULL);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, floatQ);
    linearLayer->config->linear->frozen = true;
    linearLayer->config->linear->propLossMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t fwdDims[] = {1, 3};
    tensor_t *forwardInput = makeFloatTensor(fwdDims, 2, (float[]){1.f, 2.f, 3.f});
    size_t lossDims[] = {1, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f});
    requantizeTensorInPlace(loss, bfpTemplate);
    freeQuantization(bfpTemplate);
    size_t propLossDims[] = {1, 3};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 2, NULL);

    linearBackward(linearLayer, forwardInput, loss, propLoss, NULL);

    float captured[3];
    memcpy(captured, propLoss->data, sizeof(captured));

    freeLinearLayer(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);

    /* dx[i] = sum_o loss[o] * W[o][i], W rows {0.5, -1, 2} / {1, 0.25, -0.75}. */
    TEST_ASSERT_EQUAL_FLOAT(-4.f * 0.5f + 3.f * 1.f, captured[0]);
    TEST_ASSERT_EQUAL_FLOAT(-4.f * -1.f + 3.f * 0.25f, captured[1]);
    TEST_ASSERT_EQUAL_FLOAT(-4.f * 2.f + 3.f * -0.75f, captured[2]);
}

/* #152 PR3a (ruling R1): Linear takes [batch, features] only and fails fast on
 * any other rank, like the Conv1d and pool rank guards. The guard used to print
 * and carry on: a rank-1 input wrote outputShape->dimensions[1] past the
 * one-slot array that inference() sizes by the INPUT rank, and a rank-3 input
 * left dimensions[2] at its calloc 0, a silent 0-element output. Once the loop
 * owns the batch axis, a natural-shape sample that missed batchViewOf reaches a
 * Linear first layer exactly this way. The two direct calls give the output
 * shape room for every write, so an unguarded call returns normally (child
 * exit code 0: a clean RED) instead of writing past a caller array. */
void testLinearCalcOutputShapeRejectsRank1Input(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    size_t inDims[1] = {3};
    size_t inOrder[1] = {0};
    shape_t inShape = {.numberOfDimensions = 1, .dimensions = inDims, .orderOfDimensions = inOrder};
    size_t outDims[2] = {0};
    size_t outOrder[2] = {0};
    shape_t outShape = {
        .numberOfDimensions = 1, .dimensions = outDims, .orderOfDimensions = outOrder};

    ASSERT_EXITS_WITH_FAILURE(linearCalcOutputShape(layer, &inShape, &outShape));

    freeLinearLayer(layer);
}

void testLinearCalcOutputShapeRejectsRank3Input(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    size_t inDims[3] = {1, 1, 3};
    size_t inOrder[3] = {0, 1, 2};
    shape_t inShape = {.numberOfDimensions = 3, .dimensions = inDims, .orderOfDimensions = inOrder};
    size_t outDims[3] = {0};
    size_t outOrder[3] = {0};
    shape_t outShape = {
        .numberOfDimensions = 3, .dimensions = outDims, .orderOfDimensions = outOrder};

    ASSERT_EXITS_WITH_FAILURE(linearCalcOutputShape(layer, &inShape, &outShape));

    freeLinearLayer(layer);
}

/* The user-visible failure: inference() on a one-Linear model handed a natural
 * [3] sample (a missed batchViewOf). Unguarded, it returns a garbage [3]-element
 * output. Rank 3 is not repeated at this level: on the FLOAT32 path the
 * forward's matmul already exits on a >2D operand ("Matmul only supports up to
 * 2D Tensors"), so that test would pass without the guard.
 *
 * Without the guard the write past the one-slot dims array is undefined
 * behaviour: in the plain presets it returns a garbage output (exit 0 -> RED),
 * and under the ASan preset (abort_on_error=1) it aborts with SIGABRT, which
 * DeathTest.h reports as "terminated by signal" (also RED). The two direct
 * linearCalcOutputShape tests above are the deterministic pins in every
 * preset. */
void testInferenceOneLinearModelRejectsRank1Sample(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    layer_t *model[] = {layer};
    size_t sampleDims[] = {3};
    tensor_t *sample = makeFloatTensor(sampleDims, 1, (float[]){1.f, 2.f, 3.f});

    ASSERT_EXITS_WITH_FAILURE(freeTensor(inference(model, 1, sample)));

    freeTensor(sample);
    freeLinearLayer(layer);
}

/* #440: a hand-wired config must not inherit a stale ownsQuantizations, or
 * freeLinearLayer would tear down caller-owned quantizations. Read through
 * unsigned char because loading a bool that holds 0xFF is UB. */
void testLinearInitConfigClearsOwnsQuantizations(void) {
    linearConfig_t cfg;
    memset(&cfg, 0xFF, sizeof cfg);
    linearInitConfig(&cfg, NULL, NULL, NULL, NULL, NULL);
    TEST_ASSERT_EQUAL_UINT8(0, *(const unsigned char *)&cfg.ownsQuantizations);
}

/* #436: Matmul writes the output and dx flat in physical order, so a produced
 * wire must be dense [batch, features] in identity order, and the consumed loss
 * must carry the forward batch. Fixtures perturb the batch dim only: a wider
 * produced wire is under-filled and a narrower loss on a frozen layer is
 * half-read, so an ungated call returns normally (clean RED, child exit 0). */
void testLinearForwardFloatRejectsWrongOutputShape(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    tensor_t *input = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *output = buildFloatTensor2d(2, 2, (float[]){0.f, 0.f, 0.f, 0.f});

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear forward: output",
                             linearForward(layer, input, output, NULL));

    freeTensor(output);
    freeTensor(input);
    freeLinearLayer(layer);
}

/* SYM_INT32 forward math: the gate must sit before the arithmetic dispatch. */
void testLinearForwardSymRejectsWrongOutputShape(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *weightsParam = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    layer_t *layer = buildBorrowedLinearLayer(parameterInit(weightsParam, NULL), NULL, symQ);
    tensor_t *input = buildFloatTensor2d(1, 3, (float[]){0.f, 1.f, 2.f});
    tensor_t *output = buildFloatTensor2d(2, 2, (float[]){0.f, 0.f, 0.f, 0.f});

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear forward: output",
                             linearForward(layer, input, output, NULL));

    freeLinearLayer(layer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(symQ);
}

/* Square [2, 2] output transposed: the physical dims match, so only the
 * identity-order clause can catch the mis-placed flat write. */
void testLinearForwardFloatRejectsTransposedOutput(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    tensor_t *input = buildFloatTensor2d(2, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    tensor_t *output = buildFloatTensor2d(2, 2, (float[]){0.f, 0.f, 0.f, 0.f});
    transposeTensor(output, 0, 1);

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear forward: output order",
                             linearForward(layer, input, output, NULL));

    freeTensor(output);
    freeTensor(input);
    freeLinearLayer(layer);
}

void testLinearBackwardFloatRejectsWrongPropLossShape(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE); /* 3 -> 2 */
    tensor_t *forwardInput = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){-4.f, 3.f});
    tensor_t *propLoss = buildFloatTensor2d(2, 3, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear backward: propLoss",
                             linearBackward(layer, forwardInput, loss, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeLinearLayer(layer);
}

/* ARITH_BFP dx: the gate must sit before the BFP anchor check and the arms. */
void testLinearBackwardBfpRejectsWrongPropLossShape(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpTemplate = quantizationInitBfp(8, 8, HALF_AWAY);
    size_t weightDims[] = {2, 3};
    tensor_t *weightsParam =
        makeFloatTensor(weightDims, 2, (float[]){0.5f, -1.f, 2.f, 1.f, 0.25f, -0.75f});
    requantizeTensorInPlace(weightsParam, bfpTemplate);
    layer_t *layer = buildBorrowedLinearLayer(parameterInit(weightsParam, NULL), NULL, floatQ);
    layer->config->linear->frozen = true;
    layer->config->linear->propLossMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};

    size_t fwdDims[] = {1, 3};
    tensor_t *forwardInput = makeFloatTensor(fwdDims, 2, (float[]){1.f, 2.f, 3.f});
    size_t lossDims[] = {1, 2};
    tensor_t *loss = makeFloatTensor(lossDims, 2, (float[]){-4.f, 3.f});
    requantizeTensorInPlace(loss, bfpTemplate);
    freeQuantization(bfpTemplate);
    size_t propLossDims[] = {2, 3};
    tensor_t *propLoss = makeFloatTensor(propLossDims, 2, NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear backward: propLoss",
                             linearBackward(layer, forwardInput, loss, propLoss, NULL));

    freeLinearLayer(layer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(floatQ);
}

/* Frozen, so the weight-grad op (which would catch the mismatch on its own)
 * never runs and the dx op half-reads a two-row forward from a one-row loss. */
void testLinearBackwardFloatRejectsWrongLossShape(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_FALSE); /* 3 -> 2 */
    tensor_t *forwardInput = buildFloatTensor2d(2, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){-4.f, 3.f});
    tensor_t *propLoss = buildFloatTensor2d(2, 3, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear backward: loss",
                             linearBackward(layer, forwardInput, loss, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeLinearLayer(layer);
}

/* The bias-grad kernels (FLOAT32, SYM_INT32, BFP) sum the loss by raw flat
 * index, so a transposed loss view with the right logical dims gives wrong bias
 * grads while weight-grad and dx (order-aware Matmul) stay right. Square [2, 2]
 * so the physical dims match and only the identity-order clause can catch it. */
void testLinearBackwardFloatRejectsTransposedLoss(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    tensor_t *forwardInput = buildFloatTensor2d(2, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    tensor_t *loss = buildFloatTensor2d(2, 2, (float[]){1.f, 2.f, 3.f, 4.f});
    transposeTensor(loss, 0, 1);
    tensor_t *propLoss = buildFloatTensor2d(2, 3, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    ASSERT_EXITS_WITH_OUTPUT(1, "Linear backward: loss order",
                             linearBackward(layer, forwardInput, loss, propLoss, NULL));

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeLinearLayer(layer);
}

/* The dense twin of the case above (same logical loss [[1, 3], [2, 4]]) must
 * still pass the gate and sum the bias over the batch per feature. */
void testLinearBackwardFloatDenseLossBatch2BiasGradSumsBatch(void) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *weightsParam = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    tensor_t *biasParam = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    layer_t *layer = buildBorrowedLinearLayer(parameterInit(weightsParam, weightsGrad),
                                              parameterInit(biasParam, biasGrad), q);
    tensor_t *forwardInput = buildFloatTensor2d(2, 3, (float[]){1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    tensor_t *loss = buildFloatTensor2d(2, 2, (float[]){1.f, 3.f, 2.f, 4.f});
    tensor_t *propLoss = buildFloatTensor2d(2, 3, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    linearBackward(layer, forwardInput, loss, propLoss, NULL);

    float gotWeightGrad[6];
    float gotBiasGrad[2];
    float gotPropLoss[6];
    memcpy(gotWeightGrad, weightsGrad->data, sizeof gotWeightGrad);
    memcpy(gotBiasGrad, biasGrad->data, sizeof gotBiasGrad);
    memcpy(gotPropLoss, propLoss->data, sizeof gotPropLoss);

    freeLinearLayer(layer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){3.f, 7.f}), gotBiasGrad, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){9.f, 12.f, 15.f, 19.f, 26.f, 33.f}), gotWeightGrad, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){11.f, 17.f, -21.f, 14.f, 24.f, -30.f}), gotPropLoss,
                                  6);
}

/* Matmul reads the input order-aware, so the output batch is the input's
 * LOGICAL dim 0: physical [3, 2] transposed is a logical [2, 3] input. */
void testLinearCalcOutputShapeUsesLogicalBatchOfTransposedInput(void) {
    layer_t *layer = buildFloatLinearWithTrainable(TRAINABLE_DEFAULT); /* 3 -> 2 */
    tensor_t *input = buildFloatTensor2d(3, 2, (float[]){1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    transposeTensor(input, 0, 1);
    size_t outDims[2] = {0};
    size_t outOrder[2] = {0};
    shape_t outShape = {.dimensions = outDims, .orderOfDimensions = outOrder};

    linearCalcOutputShape(layer, input->shape, &outShape);

    freeTensor(input);
    freeLinearLayer(layer);

    TEST_ASSERT_EQUAL_size_t(2, outShape.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t_ARRAY(((size_t[]){2, 2}), outDims, 2);
    TEST_ASSERT_EQUAL_size_t_ARRAY(((size_t[]){0, 1}), outOrder, 2);
}

/* User-visible consequence: the forward output gate takes its expected shape
 * from linearCalcOutputShape, so a transposed input must reach the kernel. */
void testLinearForwardFloatTransposedInputMatchesDense(void) {
    quantization_t *q = quantizationInitFloat();
    tensor_t *weightsParam = buildFloatTensor2d(2, 3, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f});
    tensor_t *biasParam = buildFloatTensor2d(1, 2, (float[]){-1.f, 3.f});
    layer_t *layer = buildBorrowedLinearLayer(parameterInit(weightsParam, NULL),
                                              parameterInit(biasParam, NULL), q);
    tensor_t *input = buildFloatTensor2d(3, 2, (float[]){1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    transposeTensor(input, 0, 1); /* logical [[1, 2, 3], [4, 5, 6]] */
    tensor_t *output = buildFloatTensor2d(2, 2, (float[]){0.f, 0.f, 0.f, 0.f});

    linearForward(layer, input, output, NULL);

    float got[4];
    memcpy(got, output->data, sizeof got);

    freeLinearLayer(layer);
    freeTensor(output);
    freeTensor(input);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){-7.f, -1.f, -13.f, 8.f}), got, 4);
}

/* Grouped ASYM weight whose zero point puts code - zp at -(2^qBits - 1): the
 * funnel unpacks it to 13-bit mantissas under a declared qBits of 12. 511
 * products of 4095 x 2047 exceed INT32_MAX though the declared-width bound
 * (511 int12 terms) would have run them unchecked. */
void testLinearForwardGroupedAsymWideMantissaAbortsInsteadOfWrapping(void) {
    enum { IN = 511, GROUPS = 7, GROUP_SIZE = 73 };
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 1;
    weightDims[1] = IN;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(
        weightShape, quantizationInitAsymGrouped(12, HALF_AWAY, GROUPS, GROUP_SIZE), NULL);
    memset(weightsParam->data, 0, (IN * 12 + 7) / 8); /* all codes 0 */
    asymQConfig_t *weightQC = weightsParam->quantization->qConfig;
    for (size_t g = 0; g < GROUPS; g++) {
        weightQC->scales[g] = 1.f;
        weightQC->zeroPoints[g] = 4095; /* mantissa = 0 - 4095 */
    }
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = IN;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < IN; i++) {
        ((int32_t *)input->data)[i] = -2047;
    }
    ((symInt32QConfig_t *)input->quantization->qConfig)->scale = 1.f;

    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, NULL, q);
    size_t *outDims = reserveMemory(2 * sizeof(size_t));
    outDims[0] = 1;
    outDims[1] = 1;
    size_t *outOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outOrder);
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    setShape(outShape, outDims, 2, outOrder);
    tensor_t *output = initTensor(outShape, quantizationInitFloat(), NULL);

    ASSERT_EXITS_WITH_OUTPUT(1, "matmul SYM_INT32 grouped",
                             linearForward(linearLayer, input, output, NULL));
}

/* ---- Staging routes must declare the true operand width (F-A / F-B) ----
 * The SYM accumulator guard proves a reduction safe from the declared widths of
 * the staged SYM_INT32 operands, so every funnel conversion into SYM_INT32 has
 * to declare the width of the codes it actually writes. */
static shape_t *shape2(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return shape;
}

static tensor_t *symInt32RowOf(size_t n, int32_t code) {
    tensor_t *t = initTensor(shape2(1, n), quantizationInitSymInt32(HALF_AWAY), NULL);
    for (size_t i = 0; i < n; i++) {
        ((int32_t *)t->data)[i] = code;
    }
    ((symInt32QConfig_t *)t->quantization->qConfig)->scale = 1.f;
    return t;
}

/* Per-tensor ASYM [1, n] row, every code 0, zero point zp: each mantissa is -zp. */
static tensor_t *asymRowOfZeroCodes(size_t n, uint8_t qBits, uint16_t zp) {
    tensor_t *t = initTensor(shape2(1, n), quantizationInitAsym(qBits, HALF_AWAY), NULL);
    memset(t->data, 0, (n * qBits + 7) / 8);
    asymQConfig_t *qc = t->quantization->qConfig;
    qc->scales[0] = 1.f;
    qc->zeroPoints[0] = zp;
    return t;
}

/* y = weights[1, n] . input[1, n] through the public linearForward under
 * ARITH_SYM_INT32; takes ownership of both operands and frees everything. */
static float linearForwardDot(tensor_t *weightsTensor, tensor_t *input) {
    parameter_t *weights = parameterInit(weightsTensor, NULL);
    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    layer_t *layer = buildBorrowedLinearLayer(weights, NULL, q);
    tensor_t *output = initTensor(shape2(1, 1), quantizationInitFloat(), NULL);
    linearForward(layer, input, output, NULL);
    float y = ((float *)output->data)[0];
    freeTensor(output);
    freeLinearLayerShellOnly(layer);
    freeQuantization(q);
    freeParameter(weights);
    freeTensor(input);
    return y;
}

/* qBits 12, zp 4095 -> mantissa -4095 (13 signed bits). 511 terms of
 * 4095 x 2047 wrap int32; the declared-int12 bound would have admitted N = 511. */
void testLinearForwardPerTensorAsym12WeightWrapsAborts(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "accumulator overflow",
        linearForwardDot(asymRowOfZeroCodes(511, 12, 4095), symInt32RowOf(511, 2047)));
}

/* qBits 16, zp 65535 -> mantissa -65535 (17 bits); the product alone is wide. */
void testLinearForwardPerTensorAsym16WeightWrapsAborts(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "accumulator overflow",
        linearForwardDot(asymRowOfZeroCodes(17, 16, 65535), symInt32RowOf(17, 2047)));
}

/* Same gap on the input side of the product. */
void testLinearForwardPerTensorAsym12InputWrapsAborts(void) {
    ASSERT_EXITS_WITH_OUTPUT(
        1, "accumulator overflow",
        linearForwardDot(symInt32RowOf(511, 2047), asymRowOfZeroCodes(511, 12, 4095)));
}

/* Controls: the staged width is exact where it fits. qBits 8 (9-bit mantissa)
 * at N = 511 is proven safe; qBits 12 at N = 256 is beyond the proof but
 * 256 * 4095 * 2047 = 2145911040 still fits int32, so the checked path
 * returns it exactly. */
void testLinearForwardPerTensorAsymWidthsThatFitAreExact(void) {
    float y8 = linearForwardDot(asymRowOfZeroCodes(511, 8, 255), symInt32RowOf(511, 2047));
    TEST_ASSERT_FLOAT_WITHIN(32.f, -266734335.f, y8);
    float y12 = linearForwardDot(asymRowOfZeroCodes(256, 12, 4095), symInt32RowOf(256, 2047));
    TEST_ASSERT_FLOAT_WITHIN(256.f, -2145911040.f, y12);
}

/* A raw INT32-stored operand has no width bound below 32 bits. */
void testLinearForwardInt32InputWrapsAborts(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "accumulator overflow", ({
                                 tensor_t *x =
                                     initTensor(shape2(1, 511), quantizationInitInt32(), NULL);
                                 for (size_t i = 0; i < 511; i++) {
                                     ((int32_t *)x->data)[i] = -4095;
                                 }
                                 linearForwardDot(symInt32RowOf(511, 2047), x);
                             }));
}

void testLinearForwardInt32InputThatFitsIsExact(void) {
    tensor_t *x = initTensor(shape2(1, 511), quantizationInitInt32(), NULL);
    for (size_t i = 0; i < 511; i++) {
        ((int32_t *)x->data)[i] = -100;
    }
    float y = linearForwardDot(symInt32RowOf(511, 2047), x);
    TEST_ASSERT_FLOAT_WITHIN(16.f, -104601700.f, y);
}

/* INT32 operands wider than the 12-bit default grid go through the funnel's
 * own staging (32-bit scratch), not the public cell that rejects them:
 * 100 * 4095 * 2047 = 838246500 fits int32, so the checked path is exact. */
void testLinearForwardInt32InputWiderThanDefaultGridIsExact(void) {
    tensor_t *x = initTensor(shape2(1, 100), quantizationInitInt32(), NULL);
    for (size_t i = 0; i < 100; i++) {
        ((int32_t *)x->data)[i] = 4095;
    }
    float y = linearForwardDot(symInt32RowOf(100, 2047), x);
    TEST_ASSERT_FLOAT_WITHIN(64.f, 838246500.f, y);
}

/* The public INT32 -> SYM_INT32 conversion cell must leave a user tensor that
 * getTensorLike (getQLike -> initSymInt32QConfigWithQMaxBits, which rejects
 * widths > 31) can still clone. The 32-bit width is declared on the funnel's
 * op-local scratch only. */
void testConvertInt32ToSymInt32UserTensorStaysCloneable(void) {
    tensor_t *src = initTensor(shape2(1, 4), quantizationInitInt32(), NULL);
    for (size_t i = 0; i < 4; i++) {
        ((int32_t *)src->data)[i] = (int32_t)i - 2;
    }
    tensor_t *dst = initTensor(shape2(1, 4), quantizationInitSymInt32(HALF_AWAY), NULL);
    convertTensor(src, dst, NULL);
    tensor_t *clone = getTensorLike(dst);
    TEST_ASSERT_NOT_NULL(clone);
    TEST_ASSERT_TRUE(((symInt32QConfig_t *)dst->quantization->qConfig)->qMaxBits <= 31);
    freeTensor(clone);
    freeTensor(dst);
    freeTensor(src);
}

/* SYM bias-grad value sum (F9 follow-up): a [batch, 1] loss wire of `batch` codes
 * at declared width `bits`, scale 1; returns the raw per-feature sum the kernel emits. */
static tensor_t *symSumLoss(size_t batch, int32_t code, uint8_t bits) {
    tensor_t *t =
        initTensor(shape2(batch, 1), quantizationInitSymInt32WithBits(HALF_AWAY, bits), NULL);
    for (size_t i = 0; i < batch; i++) {
        ((int32_t *)t->data)[i] = code;
    }
    ((symInt32QConfig_t *)t->quantization->qConfig)->scale = 1.f;
    return t;
}

static int32_t linearSymBiasGradSum(size_t batch, int32_t code, uint8_t bits) {
    tensor_t *loss = symSumLoss(batch, code, bits);
    tensor_t *bg = initTensor(shape2(1, 1), quantizationInitSymInt32(HALF_AWAY), NULL);
    linearCalcBiasGradsSymInt32(bg, loss);
    int32_t sum = ((int32_t *)bg->data)[0];
    freeTensor(bg);
    freeTensor(loss);
    return sum;
}

/* A 16-bit loss wire: N * 2^15 <= INT32_MAX holds up to N = 65535 (proven path);
 * N = 65536 is beyond the proof and takes the checked path. 65536 codes of
 * -32768 sum to exactly INT32_MIN, so the checked path must not false-abort. */
void testLinearBiasGradSymSumProofBoundary16Bit(void) {
    TEST_ASSERT_EQUAL_INT32(32767 * 65535, linearSymBiasGradSum(65535, 32767, 16));
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, linearSymBiasGradSum(65536, -32768, 16));
}

/* A 31-bit raw-int wire: two codes of 2^30 - 1 fit (2^31 - 2), a third wraps. */
void testLinearBiasGradSymSumWideWireExactThenAborts(void) {
    TEST_ASSERT_EQUAL_INT32(INT32_MAX - 1, linearSymBiasGradSum(2, (1 << 30) - 1, 31));
    ASSERT_EXITS_WITH_OUTPUT(1, "value sum", (void)linearSymBiasGradSum(3, (1 << 30) - 1, 31));
}

/* The banner of a value sum must not blame the operand-width knob: the loss wire
 * is stored, not an operand clamp. */
void testLinearBiasGradSymSumBannerNamesWireWidth(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "31-bit", (void)linearSymBiasGradSum(3, (1 << 30) - 1, 31));
}

/* An intermediate overflow aborts even when the final sum would fit: a, a, a, -a, -a
 * ends at a but leaves int32 on the third term. */
void testLinearBiasGradSymSumIntermediateOverflowAborts(void) {
    tensor_t *loss = symSumLoss(5, (1 << 30) - 1, 31);
    ((int32_t *)loss->data)[3] = -((1 << 30) - 1);
    ((int32_t *)loss->data)[4] = -((1 << 30) - 1);
    tensor_t *bg = initTensor(shape2(1, 1), quantizationInitSymInt32(HALF_AWAY), NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "accumulator overflow", linearCalcBiasGradsSymInt32(bg, loss));
}

/* Loss elements must split evenly into bias features, like the BFP twin. */
void testLinearBiasGradSymRejectsIndivisibleLoss(void) {
    tensor_t *loss = initTensor(shape2(5, 1), quantizationInitSymInt32(HALF_AWAY), NULL);
    tensor_t *bg = initTensor(shape2(1, 2), quantizationInitSymInt32(HALF_AWAY), NULL);
    ASSERT_EXITS_WITH_OUTPUT(1, "not divisible", linearCalcBiasGradsSymInt32(bg, loss));
}

/* ---- Raw grad outputs declare their width (#516) ----
 * linearCalcBiasGradsSymInt32 is a public raw producer: called on a user tensor
 * (not the funnel scratch) its output can re-enter a SYM op as an operand, so
 * it must carry the static width of the sum, not the 12-bit init default. */
static uint8_t linearSymBiasGradWidth(size_t batch, int32_t code, uint8_t bits) {
    tensor_t *loss = symSumLoss(batch, code, bits);
    tensor_t *bg = initTensor(shape2(1, 1), quantizationInitSymInt32(HALF_AWAY), NULL);
    linearCalcBiasGradsSymInt32(bg, loss);
    uint8_t width = ((symInt32QConfig_t *)bg->quantization->qConfig)->qMaxBits;
    freeTensor(bg);
    freeTensor(loss);
    return width;
}

/* A value sum of batch 12-bit codes spans [-batch * 2^11, batch * (2^11 - 1)]:
 * batch 1 is the wire's own range (12: the sum of one code adds no bit),
 * batch 2 -> [-4096, 4094] -> 13, batch 3 -> [-6144, 6141] -> 14. */
void testLinearBiasGradSymStampsTheSumWidth(void) {
    symAccumSite_t one = symAccumSiteInitSum("t", 1, 12);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&one), linearSymBiasGradWidth(1, 7, 12));
    TEST_ASSERT_EQUAL_UINT8(12, linearSymBiasGradWidth(1, 7, 12));
    TEST_ASSERT_EQUAL_UINT8(13, linearSymBiasGradWidth(2, 7, 12));
    symAccumSite_t three = symAccumSiteInitSum("t", 3, 12);
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&three), linearSymBiasGradWidth(3, 7, 12));
    TEST_ASSERT_EQUAL_UINT8(14, linearSymBiasGradWidth(3, 7, 12));
}

/* Two 31-bit codes are beyond the proof (term limit 1) but their sum fits, so
 * the checked twin returns; the output declares the raw int32 range. */
void testLinearBiasGradSymStampIs32WhenTheSumIsUnprovable(void) {
    TEST_ASSERT_EQUAL_UINT8(32, linearSymBiasGradWidth(2, (1 << 30) - 1, 31));
}

/* linearCalcWeightGradsSymInt32 emits through the stamped SYM matmul, so its
 * output already carries the product width (12 x 12 codes, K = batch = 1:
 * 2^22 -> 24) -- pinned so the layer never grows a second stamp. */
void testLinearWeightGradSymIsStampedByTheMatmul(void) {
    tensor_t *loss = symSumLoss(1, 3, 12);
    tensor_t *input = symSumLoss(1, 5, 12);
    tensor_t *wg = initTensor(shape2(1, 1), quantizationInitSymInt32WithBits(HALF_AWAY, 5), NULL);
    linearCalcWeightGradsSymInt32(loss, input, wg);
    TEST_ASSERT_EQUAL_INT32(15, ((int32_t *)wg->data)[0]);
    symAccumSite_t site = symAccumSiteInit("t", 1, 12, 12, 0);
    uint8_t width = ((symInt32QConfig_t *)wg->quantization->qConfig)->qMaxBits;
    TEST_ASSERT_EQUAL_UINT8(symAccumResultBits(&site), width);
    TEST_ASSERT_EQUAL_UINT8(24, width);
    freeTensor(wg);
    freeTensor(input);
    freeTensor(loss);
}

/* ---- Linear backward under ARITH_BFP: integer end to end -----------------
 * dx: the BFP_RAW of the backward GEMM (BfpScatterRef.h's
 * bfpRefLinearBackwardRaw, the section-13 integer fold along the Linear walk)
 * re-blocked into the BFP dx wire by the OP's mode, SR drawing from the op's
 * RNG_TAG_OUT stream (bfpRefReblockRaw). weightGrad / biasGrad into BFP grads
 * under DYNAMIC_RESCALE: the GEMM raw / the bias segment fold with the
 * data-derived raw exponent (bfpRefConv1dBiasGradSums on the [batch, outF, 1]
 * view + bfpRefRawFromRows), accumulated by the integer engine
 * (BfpAccumulateRef.h). Several micro-batch calls into one grad; codes AND
 * stored exponents compared after every call. */

/*! BFP-STORED tensor from int32 codes (packed to mantissaBits) and stored
 *  exponents: what the funnel prologue unpacks for a BFP-stored operand. */
static tensor_t *lbStored(size_t numDims, const size_t *dimsIn, const int32_t *codes,
                          const uint8_t *exponents, uint8_t m, size_t groupSize) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    memcpy(dims, dimsIn, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    size_t n = calcNumberOfElementsByShape(shape);
    size_t numGroups = groupSize == 0 ? 1 : n / groupSize;
    tensor_t *t =
        initTensor(shape, quantizationInitBfpGrouped(m, 8, HALF_AWAY, numGroups, groupSize), NULL);
    byteConversion((uint8_t *)codes, 32, t->data, m, n);
    bfpQConfig_t *qc = t->quantization->qConfig;
    for (size_t g = 0; g < numGroups; g++) {
        qc->exponents[g] = exponents[g];
    }
    return t;
}

static uint32_t lbLcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static size_t lbGroups(size_t n, size_t gsz) {
    return gsz == 0 ? 1 : n / gsz;
}

/* The op's stream of Linear op `op` under RNG_TAG_BACKWARD: rngFold(op key,
 * tag, 0), op key = rngFold(rngFold(exec key, RNG_TAG_BACKWARD, 0),
 * RNG_TAG_OP, op) -- LINEAR_OP_WEIGHT_GRAD 1, _BIAS_GRAD 2, _PROP_LOSS 3,
 * part of the key contract. */
static rngStream_t lbOpStream(rngKey_t execKey, uint32_t op, uint32_t tag) {
    return rngOpen(
        rngFold(rngFold(rngFold(execKey, RNG_TAG_BACKWARD, 0u), RNG_TAG_OP, op), tag, 0u));
}

typedef struct lbCase {
    size_t batch, inF, outF;
    uint8_t m;                /* loss and input mantissa bits */
    size_t lossGs, xGs;       /* 0 = per tensor */
    uint8_t gm, ge;           /* grad widths */
    roundingMode_t grm;       /* grad storage rounding */
    size_t wgGs, bgGs;        /* grad groups (0 = per tensor) */
    uint8_t initWgE, initBgE; /* stored exponent of the fresh (zero-code) grads */
} lbCase_t;

#define LB_MAX 256
#define LB_MAX_CALLS 4

typedef struct lbOperands {
    int32_t x[LB_MAX], loss[LB_MAX];
    uint8_t xE[LB_MAX], lossE[LB_MAX];
} lbOperands_t;

typedef struct lbResult {
    size_t wN, bN, wGroups, bGroups;
    int32_t wCodes[LB_MAX], bCodes[LB_MAX], wRef[LB_MAX], bRef[LB_MAX];
    uint8_t wE[LB_MAX], bE[LB_MAX], wRefE[LB_MAX], bRefE[LB_MAX];
    bfpRefStats_t wSt, bSt; /* summed over the calls */
} lbResult_t;

static tensor_t *lbGrad(tensor_t *param, uint8_t m, uint8_t e, roundingMode_t rm, size_t gsz,
                        uint8_t storedE) {
    size_t n = calcNumberOfElementsByTensor(param);
    quantization_t *q = gsz == 0 ? quantizationInitBfp(m, e, rm)
                                 : quantizationInitBfpGrouped(m, e, rm, n / gsz, gsz);
    tensor_t *g = initTensor(getShapeLike(param->shape), q, NULL);
    bfpQConfig_t *qc = g->quantization->qConfig;
    for (size_t i = 0; i < qc->numGroups; i++) {
        qc->exponents[i] = storedE;
    }
    return g;
}

static void lbReadGrad(tensor_t *g, int32_t *codes, uint8_t *stored) {
    bfpQConfig_t *qc = g->quantization->qConfig;
    unpackSignExtend(g->data, qc->mantissaBits, 0, codes, calcNumberOfElementsByTensor(g));
    memcpy(stored, qc->exponents, qc->numGroups);
}

/* A Linear layer [outF, inF] with BFP-stored weights (the width anchor; the
 * grad ops never read their values) and BFP grads, both grad slots ARITH_BFP
 * under `mode`. */
static layer_t *lbLayer(const lbCase_t *c, outputMode_t mode, quantization_t *floatQ) {
    size_t wDims[] = {c->outF, c->inF};
    tensor_t *wParam = makeFloatTensor(wDims, 2, NULL);
    quantization_t *wq = quantizationInitBfp(8, 8, HALF_AWAY);
    requantizeTensorInPlace(wParam, wq);
    freeQuantization(wq);
    parameter_t *weights =
        parameterInit(wParam, lbGrad(wParam, c->gm, c->ge, c->grm, c->wgGs, c->initWgE));
    size_t bDims[] = {c->outF};
    tensor_t *bParam = makeFloatTensor(bDims, 1, NULL);
    parameter_t *bias =
        parameterInit(bParam, lbGrad(bParam, c->gm, c->ge, c->grm, c->bgGs, c->initBgE));
    layer_t *layer = buildBorrowedLinearLayer(weights, bias, floatQ);
    linearConfig_t *cfg = layer->config->linear;
    cfg->weightGradMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    cfg->biasGradMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    cfg->weightGradAccMode = mode;
    cfg->biasGradAccMode = mode;
    return layer;
}

/* Runs `calls` micro-batches (ops[j]) through linearBackward (grads only)
 * into one weight / bias grad pair and the reference chain next to it;
 * asserts equality after every call. seed 0 = no exec (HALF_AWAY grads). */
static void lbRun(const lbCase_t *c, const lbOperands_t *ops, size_t calls, uint64_t seed,
                  lbResult_t *r) {
    size_t xN = c->batch * c->inF;
    size_t lossN = c->batch * c->outF;
    r->wN = c->outF * c->inF;
    r->bN = c->outF;
    r->wGroups = lbGroups(r->wN, c->wgGs);
    r->bGroups = lbGroups(r->bN, c->bgGs);
    TEST_ASSERT_TRUE(xN <= LB_MAX && lossN <= LB_MAX && r->wN <= LB_MAX);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *layer = lbLayer(c, OUT_ACC_DYNAMIC_RESCALE, floatQ);
    linearConfig_t *cfg = layer->config->linear;
    tensor_t *wGrad = getGradFromParameter(cfg->weights);
    tensor_t *bGrad = getGradFromParameter(cfg->bias);
    lbReadGrad(wGrad, r->wRef, r->wRefE);
    lbReadGrad(bGrad, r->bRef, r->bRefE);
    r->wSt = (bfpRefStats_t){.fitsInt32 = true};
    r->bSt = (bfpRefStats_t){.fitsInt32 = true};
    layerExec_t exec = {.key = rngKeyFromSeed(seed)};
    bfpRefGeom_t G = {.batch = c->batch, .cout = c->outF, .outLength = 1};
    size_t xDims[] = {c->batch, c->inF};
    size_t lossDims[] = {c->batch, c->outF};
    for (size_t j = 0; j < calls; j++) {
        const lbOperands_t *o = &ops[j];
        tensor_t *x = lbStored(2, xDims, o->x, o->xE, c->m, c->xGs);
        tensor_t *loss = lbStored(2, lossDims, o->loss, o->lossE, c->m, c->lossGs);
        linearBackward(layer, x, loss, NULL, seed != 0 ? &exec : NULL);
        freeTensor(loss);
        freeTensor(x);

        bfpRefOperand_t xRef = {o->x, o->xE, c->xGs, c->m, 8};
        bfpRefOperand_t lossRef = {o->loss, o->lossE, c->lossGs, c->m, 8};
        int32_t mant[LB_MAX];
        int16_t rawE[LB_MAX];
        bfpRefLinearBackwardRaw(BFP_REF_LIN_WEIGHT_GRAD, c->batch, c->inF, c->outF, &lossRef, &xRef,
                                c->wgGs, mant, rawE, &r->wSt);
        rngStream_t s = lbOpStream(exec.key, 1u, RNG_TAG_ACC);
        bfpAccRefAccumulate(r->wRef, r->wRefE, mant, rawE, r->wN, c->wgGs, c->gm, c->ge, c->grm,
                            seed != 0 ? &s : NULL);
        int64_t sum[LB_MAX];
        int eAcc[LB_MAX];
        bfpRefConv1dBiasGradSums(&G, &lossRef, sum, eAcc, &r->bSt);
        bfpRefRawFromRows(sum, eAcc, r->bN, c->bgGs, mant, rawE, &r->bSt);
        s = lbOpStream(exec.key, 2u, RNG_TAG_ACC);
        bfpAccRefAccumulate(r->bRef, r->bRefE, mant, rawE, r->bN, c->bgGs, c->gm, c->ge, c->grm,
                            seed != 0 ? &s : NULL);

        lbReadGrad(wGrad, r->wCodes, r->wE);
        lbReadGrad(bGrad, r->bCodes, r->bE);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(r->wRefE, r->wE, r->wGroups);
        TEST_ASSERT_EQUAL_INT32_ARRAY(r->wRef, r->wCodes, r->wN);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(r->bRefE, r->bE, r->bGroups);
        TEST_ASSERT_EQUAL_INT32_ARRAY(r->bRef, r->bCodes, r->bN);
    }
    TEST_ASSERT_TRUE(r->wSt.fitsInt32);
    TEST_ASSERT_TRUE(r->bSt.fitsInt32);
    TEST_ASSERT_EQUAL_INT(c->grm, ((bfpQConfig_t *)wGrad->quantization->qConfig)->roundingMode);

    freeLinearLayer(layer);
    freeQuantization(floatQ);
}

/* Random codes in the full m-bit range; stored exponents uniform in bias +
 * [lo, hi]. */
static void lbFill(lbOperands_t *o, uint32_t seed, uint8_t m, size_t xN, size_t xGroups, int xLo,
                   int xHi, size_t lossN, size_t lossGroups, int lossLo, int lossHi) {
    uint32_t s = seed;
    uint32_t span = 1u << m;
    for (size_t i = 0; i < xN; i++) {
        o->x[i] = (int32_t)(lbLcg(&s) % span) - (int32_t)(span / 2u);
    }
    for (size_t i = 0; i < lossN; i++) {
        o->loss[i] = (int32_t)(lbLcg(&s) % span) - (int32_t)(span / 2u);
    }
    for (size_t i = 0; i < xGroups; i++) {
        o->xE[i] = (uint8_t)(127 + xLo + (int)(lbLcg(&s) % (uint32_t)(xHi - xLo + 1)));
    }
    for (size_t i = 0; i < lossGroups; i++) {
        o->lossE[i] = (uint8_t)(127 + lossLo + (int)(lbLcg(&s) % (uint32_t)(lossHi - lossLo + 1)));
    }
}

/* batch 3, [6, 8]: loss groups of 9 straddle the loss rows, input groups of
 * 6 straddle the input rows, so every reduction step of a weightGrad output
 * can change either operand's group (one segment per product); a 36-binade
 * product spread against g = -14 makes some segments exact left shifts and
 * rounds others. Per-tensor m8 grads, four micro-batches: fresh, carried, a
 * much smaller loss (the old grid dominates), a much larger one (the
 * increment does). */
static const lbCase_t kLbMixed = {.batch = 3,
                                  .inF = 8,
                                  .outF = 6,
                                  .m = 8,
                                  .lossGs = 9,
                                  .xGs = 6,
                                  .gm = 8,
                                  .ge = 8,
                                  .grm = HALF_AWAY,
                                  .wgGs = 0,
                                  .bgGs = 0,
                                  .initWgE = 127,
                                  .initBgE = 127};

static void lbMixedOperands(lbOperands_t *ops, uint32_t seed) {
    static const int lossShift[LB_MAX_CALLS] = {0, 3, -14, 12};
    for (size_t j = 0; j < LB_MAX_CALLS; j++) {
        lbFill(&ops[j], seed + 101u * (uint32_t)j, 8, 24, 4, -10, 10, 18, 2, -8 + lossShift[j],
               8 + lossShift[j]);
    }
}

void testLinearBfpGradsIntoBfpStorageMatchIntegerReferenceOverMicroBatches(void) {
    static lbOperands_t ops[LB_MAX_CALLS];
    static lbResult_t r;
    lbMixedOperands(ops, 777u);
    lbRun(&kLbMixed, ops, LB_MAX_CALLS, 0, &r);
    TEST_ASSERT_TRUE(r.wSt.rounded > 0);   /* in-loop alignment rounded */
    TEST_ASSERT_TRUE(r.wSt.leftShift > 0); /* ... and shifted left exactly */
    TEST_ASSERT_TRUE(r.bSt.leftShift > 0);
}

/* Grouped grads whose groups straddle the weight rows (rows of 8, groups of
 * 6; bias groups of 2) from zero codes under a STALE exponent (the fresh-
 * accumulator rule: an all-zero old side must not set the working grid), at
 * m6 grads. */
void testLinearBfpGradsIntoGroupedBfpStorageStraddlingRowsMatchReference(void) {
    static lbOperands_t ops[LB_MAX_CALLS];
    static lbResult_t r;
    lbCase_t c = kLbMixed;
    c.wgGs = 6;
    c.bgGs = 2;
    c.initWgE = 127 + 40;
    c.initBgE = 127 - 40;
    c.gm = 6;
    lbMixedOperands(ops, 4242u);
    lbRun(&c, ops, LB_MAX_CALLS, 0, &r);
    TEST_ASSERT_TRUE(r.wSt.rounded > 0);
}

/* SR grads (the HAR template): the engine rounds by the grad's own mode,
 * one uniform per element in element order from each grad op's ACC(0)
 * stream (op keys LINEAR_OP_WEIGHT_GRAD / _BIAS_GRAD). Non-vacuous: the
 * HALF_AWAY grads differ. */
void testLinearBfpGradsIntoSrBfpStorageDrawFromTheAccStream(void) {
    static lbOperands_t ops[LB_MAX_CALLS];
    static lbResult_t sr;
    static lbResult_t ha;
    lbMixedOperands(ops, 99u);
    lbCase_t c = kLbMixed;
    c.grm = SR_HALF_AWAY;
    lbRun(&c, ops, LB_MAX_CALLS, 2027u, &sr);
    lbRun(&kLbMixed, ops, LB_MAX_CALLS, 0, &ha);
    TEST_ASSERT_FALSE_MESSAGE(memcmp(sr.wCodes, ha.wCodes, sr.wN * sizeof(int32_t)) == 0,
                              "SR and HALF_AWAY weight grads agree: the SR test is vacuous");
    TEST_ASSERT_FALSE_MESSAGE(memcmp(sr.bCodes, ha.bCodes, sr.bN * sizeof(int32_t)) == 0,
                              "SR and HALF_AWAY bias grads agree: the SR test is vacuous");
}

/* Cancellation to the exact zero state: one product per grad element, codes
 * that fit the grad exactly, then the same micro-batch with the loss negated
 * -- every sum is exactly zero, so codes 0 and stored = bias. */
void testLinearBfpGradsCancelToTheZeroState(void) {
    lbCase_t c = {.batch = 1,
                  .inF = 1,
                  .outF = 2,
                  .m = 8,
                  .lossGs = 0,
                  .xGs = 0,
                  .gm = 8,
                  .ge = 8,
                  .grm = HALF_AWAY,
                  .wgGs = 0,
                  .bgGs = 0,
                  .initWgE = 127,
                  .initBgE = 127};
    static lbOperands_t ops[2];
    static lbResult_t r;
    ops[0].x[0] = 3;
    ops[0].xE[0] = 127 - 4;
    ops[0].loss[0] = 5;
    ops[0].loss[1] = -7;
    ops[0].lossE[0] = 127 - 9;
    ops[1] = ops[0];
    ops[1].loss[0] = -5;
    ops[1].loss[1] = 7;
    lbRun(&c, ops, 1, 0, &r);
    /* {15, -21} * 2^-13: absmax 21 <= 127 * 2^(E + 13) first at E = -15 */
    TEST_ASSERT_EQUAL_INT32_ARRAY(((int32_t[]){60, -84}), r.wCodes, 2);
    TEST_ASSERT_EQUAL_UINT8(127 - 15, r.wE[0]);
    lbRun(&c, ops, 2, 0, &r);
    TEST_ASSERT_EQUAL_INT32_ARRAY(((int32_t[]){0, 0}), r.wCodes, 2);
    TEST_ASSERT_EQUAL_UINT8(127, r.wE[0]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(((int32_t[]){0, 0}), r.bCodes, 2);
    TEST_ASSERT_EQUAL_UINT8(127, r.bE[0]);
}

/* Exponent extremes from fresh grads: the m16 worst case (every code -2^15,
 * every product 2^30: the segment limit is ONE product and the guard is
 * positive, g = 1 at batch 2) into m16 grads, exact over two calls; operands
 * at the top of e = 8 (the grad saturates at the cap) and at its floor. */
void testLinearBfpGradsExponentExtremes(void) {
    lbCase_t c = {.batch = 2,
                  .inF = 2,
                  .outF = 2,
                  .m = 16,
                  .lossGs = 0,
                  .xGs = 0,
                  .gm = 16,
                  .ge = 8,
                  .grm = HALF_AWAY,
                  .wgGs = 0,
                  .bgGs = 0,
                  .initWgE = 127,
                  .initBgE = 127};
    static lbOperands_t ops[2];
    static lbResult_t r;
    for (size_t i = 0; i < 4; i++) {
        ops[0].x[i] = -32768;
        ops[0].loss[i] = -32768;
    }
    ops[0].xE[0] = 127 - 4;
    ops[0].lossE[0] = 127 + 1;
    ops[1] = ops[0];
    lbRun(&c, ops, 2, 0, &r);
    /* per call 2 products of 2^30 * 2^-3 = 2^28, two calls 2^29 = 16384 * 2^15 */
    TEST_ASSERT_EQUAL_INT32(16384, r.wCodes[0]);
    TEST_ASSERT_EQUAL_UINT8(127 + 15, r.wE[0]);
    /* bias: per call 2 * -2^15 * 2^1, two calls -2^18 = -16384 * 2^4 */
    TEST_ASSERT_EQUAL_INT32(-16384, r.bCodes[0]);
    TEST_ASSERT_EQUAL_UINT8(127 + 4, r.bE[0]);

    c.m = 8;
    c.gm = 8;
    for (size_t i = 0; i < 4; i++) {
        ops[0].x[i] = (int32_t)(i * 71 % 256) - 128;
        ops[0].loss[i] = 127 - (int32_t)(i * 45 % 256);
    }
    ops[0].xE[0] = 254;
    ops[0].lossE[0] = 254;
    lbRun(&c, ops, 1, 0, &r);
    TEST_ASSERT_EQUAL_UINT8(254, r.wE[0]); /* capped: bias + 127 */
    TEST_ASSERT_EQUAL_UINT8(254, r.bE[0]);
    ops[0].xE[0] = 1;
    ops[0].lossE[0] = 3;
    lbRun(&c, ops, 1, 0, &r);
    TEST_ASSERT_EQUAL_UINT8(0, r.wE[0]); /* floored (the bias sums stay on the loss's grid) */
}

/* The bias sum's guard is a pure-SUM guard (N 2^(m-1), not the product bound
 * N 2^(2m-2)): one loss group per code (every code its own segment). Each
 * feature's batch-0 code is a 1 at E = 0 (the feature's top; a ZERO code
 * there would set nothing), the others sit 15-20 binades lower. With batch 10
 * the sum guard (-20) keeps every segment exact; the product guard (-13)
 * would round the lowest ones. */
void testLinearBfpBiasGradSegmentsFoldExactlyUnderTheSumGuard(void) {
    lbCase_t c = kLbMixed;
    c.batch = 10;
    c.outF = 4;
    c.lossGs = 1;
    c.xGs = 0;
    c.gm = 16;
    static lbOperands_t ops[2];
    static lbResult_t r;
    for (size_t j = 0; j < 2; j++) {
        lbFill(&ops[j], 515u + (uint32_t)j, 8, 80, 1, -2, 2, 40, 40, -20, -15);
        for (size_t f = 0; f < 4; f++) {
            ops[j].loss[f] = 1; /* batch 0 of feature f */
            ops[j].lossE[f] = 127;
        }
    }
    lbRun(&c, ops, 2, 0, &r);
    TEST_ASSERT_EQUAL_size_t(0, r.bSt.rounded);
}

/* The fixed grid is a separate mode (Linear's default bias mode): BFP grads
 * under OUT_ACC_FIXED_SCALE keep the FLOAT32 raw of the float cores and the
 * float fixed-grid engine -- the result equals that engine fed the float
 * cores' output on the same operands. */
void testLinearBfpGradsFixedScaleKeepTheFloatBridge(void) {
    static lbOperands_t ops[LB_MAX_CALLS];
    lbMixedOperands(ops, 31u);
    const lbCase_t *c = &kLbMixed;
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *layer = lbLayer(c, OUT_ACC_FIXED_SCALE, floatQ);
    linearConfig_t *cfg = layer->config->linear;
    tensor_t *wGrad = getGradFromParameter(cfg->weights);
    tensor_t *bGrad = getGradFromParameter(cfg->bias);
    tensor_t *wTwin = lbGrad(cfg->weights->param, 8, 8, HALF_AWAY, 0, 127);
    tensor_t *bTwin = lbGrad(cfg->bias->param, 8, 8, HALF_AWAY, 0, 127);
    size_t xDims[] = {c->batch, c->inF};
    size_t lossDims[] = {c->batch, c->outF};
    tensor_t *x = lbStored(2, xDims, ops[0].x, ops[0].xE, 8, c->xGs);
    tensor_t *loss = lbStored(2, lossDims, ops[0].loss, ops[0].lossE, 8, c->lossGs);
    linearBackward(layer, x, loss, NULL, NULL);
    freeTensor(loss);
    freeTensor(x);

    bfpScratchFixture_t xF, lossF;
    bfpScratchInit(&xF, 2, xDims, ops[0].x, ops[0].xE, 4, c->xGs, 8, 8);
    bfpScratchInit(&lossF, 2, lossDims, ops[0].loss, ops[0].lossE, 2, c->lossGs, 8, 8);
    size_t wgDims[] = {c->outF, c->inF};
    size_t bgDims[] = {c->outF};
    tensor_t *wRaw = makeFloatTensor(wgDims, 2, NULL);
    tensor_t *bRaw = makeFloatTensor(bgDims, 1, NULL);
    linearCalcWeightGradsBfp(&lossF.tensor, &xF.tensor, wRaw);
    linearCalcBiasGradsBfp(&lossF.tensor, bRaw);
    accumulateFloatIntoBfpTensorFixedGrid(wTwin, (const float *)wRaw->data, 48, NULL);
    accumulateFloatIntoBfpTensorFixedGrid(bTwin, (const float *)bRaw->data, 6, NULL);

    int32_t got[48], want[48];
    uint8_t gotE[1], wantE[1];
    lbReadGrad(wGrad, got, gotE);
    lbReadGrad(wTwin, want, wantE);
    TEST_ASSERT_EQUAL_UINT8(wantE[0], gotE[0]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 48);
    lbReadGrad(bGrad, got, gotE);
    lbReadGrad(bTwin, want, wantE);
    TEST_ASSERT_EQUAL_UINT8(wantE[0], gotE[0]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(want, got, 6);

    freeTensor(bRaw);
    freeTensor(wRaw);
    freeTensor(bTwin);
    freeTensor(wTwin);
    freeLinearLayer(layer);
    freeQuantization(floatQ);
}

/* The integer bias core is a public symbol: a non-raw output would be
 * written as int32 mantissas into a float buffer. */
void testLinearCalcBiasGradsBfpRawRejectsANonRawOutput(void) {
    static const int32_t codes[] = {5, -7};
    static const uint8_t exps[] = {127};
    bfpScratchFixture_t lossF;
    size_t lossDims[] = {1, 2};
    bfpScratchInit(&lossF, 2, lossDims, codes, exps, 1, 0, 8, 8);
    size_t bgDims[] = {2};
    tensor_t *bRaw = makeFloatTensor(bgDims, 1, NULL);

    ASSERT_EXITS_WITH_FAILURE(linearCalcBiasGradsBfpRaw(&lossF.tensor, bRaw));

    freeTensor(bRaw);
}

/* ---- The integer backward cores (kernel tier, unpacked-BFP scratch in,
 * BFP_RAW out) against the int64 references, mantissas AND raw exponents. */

static void lbFillCodes(uint32_t *s, uint8_t m, int32_t *codes, size_t n, uint8_t *exps,
                        size_t groups, int lo, int hi) {
    uint32_t span = 1u << m;
    for (size_t i = 0; i < n; i++) {
        codes[i] = (int32_t)(lbLcg(s) % span) - (int32_t)(span / 2u);
    }
    for (size_t g = 0; g < groups; g++) {
        exps[g] = (uint8_t)(127 + lo + (int)(lbLcg(s) % (uint32_t)(hi - lo + 1)));
    }
}

/* dW = loss^T x at batch 3: loss [3, 6] in groups of 9, x [3, 8] in groups
 * of 6, a 30-binade spread against g = -14 (some products rounded, some sums
 * wider than 24 bits -- where a float bridge would round); raw per tensor and
 * in groups of 6 straddling the [6, 8] rows. The core restores the caller's
 * loss view (its transpose dance runs on scratch that aliases the shape). */
void testLinearCalcWeightGradsBfpRawMatchesTheGemmReference(void) {
    static const size_t rawGs[] = {0, 6};
    bool wide = false;
    for (size_t v = 0; v < 2; v++) {
        uint32_t s = 2718u;
        int32_t loss[18], x[24];
        uint8_t lossE[2], xE[4];
        lbFillCodes(&s, 8, loss, 18, lossE, 2, -15, 15);
        lbFillCodes(&s, 8, x, 24, xE, 4, -6, 6);
        /* both operands' top groups hold batch row 0, so the top products
         * meet the bound (sums beyond 24 bits); the others sit 30 binades lower */
        lossE[0] = 127 + 15;
        lossE[1] = 127 - 15;
        xE[0] = 127 + 6;
        bfpScratchFixture_t lossF, xF;
        size_t lossDims[] = {3, 6};
        size_t xDims[] = {3, 8};
        bfpScratchInit(&lossF, 2, lossDims, loss, lossE, 2, 9, 8, 8);
        bfpScratchInit(&xF, 2, xDims, x, xE, 4, 6, 8, 8);
        bfpRawFixture_t out;
        size_t outDims[] = {6, 8};
        size_t groups = rawGs[v] == 0 ? 1 : 48 / rawGs[v];
        bfpRawInit(&out, 2, outDims, groups, rawGs[v]);

        linearCalcWeightGradsBfpRaw(&lossF.tensor, &xF.tensor, &out.tensor);

        bfpRefOperand_t lossRef = {loss, lossE, 9, 8, 8};
        bfpRefOperand_t xRef = {x, xE, 6, 8, 8};
        int32_t mant[48];
        int16_t rawE[8];
        bfpRefStats_t st = {.fitsInt32 = true};
        bfpRefLinearBackwardRaw(BFP_REF_LIN_WEIGHT_GRAD, 3, 8, 6, &lossRef, &xRef, rawGs[v], mant,
                                rawE, &st);
        TEST_ASSERT_TRUE(st.fitsInt32);
        TEST_ASSERT_TRUE(st.rounded > 0);
        assertRawEquals(&out, mant, 48, rawE, groups);
        TEST_ASSERT_EQUAL_size_t(0, lossF.order[0]);
        TEST_ASSERT_EQUAL_size_t(1, lossF.order[1]);
        wide = wide || st.absMaxSum >= ((int64_t)1 << 24);
    }
    TEST_ASSERT_TRUE_MESSAGE(wide, "no sum beyond 24 bits: the test cannot tell a float bridge");
}

/* dx = loss W at batch 2 (the weight as stored, [6, 8], groups of 4: a weight
 * column sees half the groups, E_top takes the operand-wide max there): raw
 * groups of 4 and per tensor. */
void testLinearCalcPropLossBfpEmitsTheGemmRaw(void) {
    static const size_t rawGs[] = {4, 0};
    for (size_t v = 0; v < 2; v++) {
        uint32_t s = 1618u;
        int32_t loss[12], w[48];
        uint8_t lossE[3], wE[12];
        lbFillCodes(&s, 8, loss, 12, lossE, 3, -12, 12);
        lbFillCodes(&s, 8, w, 48, wE, 12, -6, 6);
        bfpScratchFixture_t lossF, wF;
        size_t lossDims[] = {2, 6};
        size_t wDims[] = {6, 8};
        bfpScratchInit(&lossF, 2, lossDims, loss, lossE, 3, 4, 8, 8);
        bfpScratchInit(&wF, 2, wDims, w, wE, 12, 4, 8, 8);
        bfpRawFixture_t out;
        size_t outDims[] = {2, 8};
        size_t groups = rawGs[v] == 0 ? 1 : 16 / rawGs[v];
        bfpRawInit(&out, 2, outDims, groups, rawGs[v]);

        linearCalcPropLossBfp(&lossF.tensor, &wF.tensor, &out.tensor);

        bfpRefOperand_t lossRef = {loss, lossE, 4, 8, 8};
        bfpRefOperand_t wRef = {w, wE, 4, 8, 8};
        int32_t mant[16];
        int16_t rawE[4];
        bfpRefStats_t st = {.fitsInt32 = true};
        bfpRefLinearBackwardRaw(BFP_REF_LIN_DX, 2, 8, 6, &lossRef, &wRef, rawGs[v], mant, rawE,
                                &st);
        TEST_ASSERT_TRUE(st.fitsInt32);
        assertRawEquals(&out, mant, 16, rawE, groups);
    }
}

/* Zero groups never set E_acc: loss [2, 4] in groups of 4, batch row 0 all
 * zero in the zero state (E = 0), row 1 live at E = -28. Bias raw: every
 * feature's sum is row 1's code, exactly; dW = loss^T x (x per tensor at
 * E = 0; the transposed loss is the strided operand-wide max): exact too. */
void testLinearBfpRawGradsZeroLossGroupDoesNotSetTheAccumulator(void) {
    int32_t loss[8] = {0, 0, 0, 0, 113, -77, 91, -59};
    uint8_t lossE[] = {127, 127 - 28};
    bfpScratchFixture_t lossF;
    size_t lossDims[] = {2, 4};
    bfpScratchInit(&lossF, 2, lossDims, loss, lossE, 2, 4, 8, 8);

    bfpRawFixture_t bias;
    size_t biasDims[] = {4};
    bfpRawInit(&bias, 1, biasDims, 1, 0);
    linearCalcBiasGradsBfpRaw(&lossF.tensor, &bias.tensor);
    for (size_t f = 0; f < 4; f++) {
        assertBfpValueExact(bias.mant[f], bias.exps[0], loss[4 + f], -28, f);
    }

    int32_t x[6] = {101, -67, 45, 13, -99, 87};
    uint8_t xE[] = {127};
    bfpScratchFixture_t xF;
    size_t xDims[] = {2, 3};
    bfpScratchInit(&xF, 2, xDims, x, xE, 1, 0, 8, 8);
    bfpRawFixture_t dw;
    size_t dwDims[] = {4, 3};
    bfpRawInit(&dw, 2, dwDims, 1, 0);
    linearCalcWeightGradsBfpRaw(&lossF.tensor, &xF.tensor, &dw.tensor);
    for (size_t f = 0; f < 4; f++) {
        for (size_t i = 0; i < 3; i++) {
            int64_t exact = (int64_t)loss[f] * x[i] + (int64_t)loss[4 + f] * x[3 + i];
            assertBfpValueExact(dw.mant[f * 3 + i], dw.exps[0], exact, -28, f * 3 + i);
        }
    }
}

/* Bias raw from the data: feature 0's loss codes are zeros in the zero state
 * (one group per code, E = 0, the bound's top), features 1-3 hold codes 25-30
 * binades lower. By the bound feature 0 would place the per-tensor raw grid
 * and the real sums would be re-gridded to a few bits; from the data they set
 * it and keep every bit (nothing re-gridded, nothing rounded). */
void testLinearCalcBiasGradsBfpRawExponentFollowsTheDataNotTheBound(void) {
    uint32_t s = 1414u;
    int32_t loss[12];
    uint8_t lossE[12];
    lbFillCodes(&s, 8, loss, 12, lossE, 12, -30, -25);
    for (size_t n = 0; n < 3; n++) {
        loss[n * 4] = 0;
        lossE[n * 4] = 127;
    }
    bfpScratchFixture_t lossF;
    size_t lossDims[] = {3, 4};
    bfpScratchInit(&lossF, 2, lossDims, loss, lossE, 12, 1, 8, 8);
    bfpRawFixture_t out;
    size_t outDims[] = {4};
    bfpRawInit(&out, 1, outDims, 1, 0);

    linearCalcBiasGradsBfpRaw(&lossF.tensor, &out.tensor);

    bfpRefOperand_t lossRef = {loss, lossE, 1, 8, 8};
    bfpRefGeom_t G = {.batch = 3, .cout = 4, .outLength = 1};
    int64_t sum[4];
    int eAcc[4];
    int32_t mant[4];
    int16_t rawE[1];
    bfpRefStats_t st = {.fitsInt32 = true};
    bfpRefConv1dBiasGradSums(&G, &lossRef, sum, eAcc, &st);
    bfpRefRawFromRows(sum, eAcc, 4, 0, mant, rawE, &st);
    assertRawEquals(&out, mant, 4, rawE, 1);
    TEST_ASSERT_EQUAL_size_t(0, st.rounded);
    TEST_ASSERT_EQUAL_size_t(0, st.realigned);
    TEST_ASSERT_EQUAL_INT32(0, out.mant[0]);
    TEST_ASSERT_TRUE(out.exps[0] < -40); /* the low features' grid, not the zero state's */
}

/* ---- dx into a BFP wire ---- */

typedef struct lbDxCase {
    size_t batch, inF, outF;
    uint8_t m;          /* loss and weight mantissa bits */
    size_t lossGs, wGs; /* 0 = per tensor */
    uint8_t wireM;
    size_t wireGs; /* 0 = per tensor */
} lbDxCase_t;

typedef struct lbDxResult {
    int32_t codes[LB_MAX], refCodes[LB_MAX];
    uint8_t stored[LB_MAX], refStored[LB_MAX];
    size_t n, numGroups;
    bfpRefStats_t st;
    roundingMode_t wireModeAfter;
} lbDxResult_t;

/* linearBackward (frozen: dx only) with BFP-stored loss and weights into a
 * BFP dx wire whose template rounds HALF_AWAY; the op rounds by `opMode`,
 * exec keyed by `seed` (the SR stream), NULL for HALF_AWAY. */
static void lbDxRun(const lbDxCase_t *c, const int32_t *lossCodes, const uint8_t *lossExps,
                    const int32_t *wCodes, const uint8_t *wExps, roundingMode_t opMode,
                    uint64_t seed, lbDxResult_t *r) {
    r->n = c->batch * c->inF;
    r->numGroups = lbGroups(r->n, c->wireGs);
    TEST_ASSERT_TRUE(r->n <= LB_MAX);
    size_t wDims[] = {c->outF, c->inF};
    parameter_t *weights = parameterInit(lbStored(2, wDims, wCodes, wExps, c->m, c->wGs), NULL);
    quantization_t *floatQ = quantizationInitFloat();
    layer_t *layer = buildBorrowedLinearLayer(weights, NULL, floatQ);
    layer->config->linear->frozen = true;
    layer->config->linear->propLossMath = (arithmetic_t){.type = ARITH_BFP, .roundingMode = opMode};

    size_t xDims[] = {c->batch, c->inF};
    size_t lossDims[] = {c->batch, c->outF};
    tensor_t *loss = lbStored(2, lossDims, lossCodes, lossExps, c->m, c->lossGs);
    tensor_t *x = makeFloatTensor(xDims, 2, NULL); /* frozen: dx reads only the shape */
    int32_t zeros[LB_MAX] = {0};
    uint8_t zeroExps[LB_MAX] = {0};
    tensor_t *dx = lbStored(2, xDims, zeros, zeroExps, c->wireM, c->wireGs);

    layerExec_t exec = {.key = rngKeyFromSeed(seed)};
    linearBackward(layer, x, loss, dx, opMode == SR_HALF_AWAY ? &exec : NULL);

    bfpQConfig_t *dxQC = dx->quantization->qConfig;
    unpackSignExtend(dx->data, c->wireM, 0, r->codes, r->n);
    memcpy(r->stored, dxQC->exponents, r->numGroups);
    r->wireModeAfter = dxQC->roundingMode;
    freeLinearLayer(layer);
    freeTensor(dx);
    freeTensor(x);
    freeTensor(loss);
    freeQuantization(floatQ);

    bfpRefOperand_t lossRef = {lossCodes, lossExps, c->lossGs, c->m, 8};
    bfpRefOperand_t wRef = {wCodes, wExps, c->wGs, c->m, 8};
    int32_t rawMant[LB_MAX];
    int16_t rawE[LB_MAX];
    r->st = (bfpRefStats_t){.fitsInt32 = true};
    bfpRefLinearBackwardRaw(BFP_REF_LIN_DX, c->batch, c->inF, c->outF, &lossRef, &wRef, c->wireGs,
                            rawMant, rawE, &r->st);
    rngStream_t out = lbOpStream(exec.key, 3u, RNG_TAG_OUT);
    bfpRefReblockRaw(rawMant, rawE, r->n, c->wireGs, c->wireM, 8,
                     opMode == SR_HALF_AWAY ? &out : NULL, r->refCodes, r->refStored);
}

static void lbDxOperands(uint32_t seed, size_t lossN, size_t lossGroups, size_t wN, size_t wGroups,
                         uint8_t m, int32_t *lossCodes, uint8_t *lossExps, int32_t *wCodes,
                         uint8_t *wExps) {
    uint32_t s = seed;
    uint32_t span = 1u << m;
    for (size_t i = 0; i < lossN; i++) {
        lossCodes[i] = (int32_t)(lbLcg(&s) % span) - (int32_t)(span / 2);
    }
    for (size_t i = 0; i < wN; i++) {
        wCodes[i] = (int32_t)(lbLcg(&s) % span) - (int32_t)(span / 2);
    }
    for (size_t i = 0; i < lossGroups; i++) {
        lossExps[i] = (uint8_t)(127 - 12 + (int)(lbLcg(&s) % 25u));
    }
    for (size_t i = 0; i < wGroups; i++) {
        wExps[i] = (uint8_t)(127 - 6 + (int)(lbLcg(&s) % 13u));
    }
}

/* batch 2, [6, 8], m8: loss groups of 4 straddle the loss rows (24-binade
 * spread), weight groups of 4 split every weight row in halves, so a weight
 * COLUMN (dx's strided reduction line) sees only half the groups -- the
 * documented E_top takes the weight's operand-wide max there, the loss row's
 * own max. Into an m6 wire: raw groups of 4 (inside a row), 8 (one row) and
 * per tensor. Codes and stored exponents exact against the int64 GEMM + the
 * section-13 re-block. */
void testLinearBackwardBfpDxIntoBfpWireMatchesIntegerReference(void) {
    static const size_t wireGroupSizes[] = {4, 8, 0};
    for (size_t v = 0; v < 3; v++) {
        lbDxCase_t c = {.batch = 2,
                        .inF = 8,
                        .outF = 6,
                        .m = 8,
                        .lossGs = 4,
                        .wGs = 4,
                        .wireM = 6,
                        .wireGs = wireGroupSizes[v]};
        int32_t lossCodes[12];
        int32_t wCodes[48];
        uint8_t lossExps[3];
        uint8_t wExps[12];
        lbDxOperands(31337u, 12, 3, 48, 12, 8, lossCodes, lossExps, wCodes, wExps);
        lbDxResult_t r;
        lbDxRun(&c, lossCodes, lossExps, wCodes, wExps, HALF_AWAY, 0, &r);

        TEST_ASSERT_TRUE(r.st.fitsInt32);
        TEST_ASSERT_TRUE(r.st.rounded > 0);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(r.refStored, r.stored, r.numGroups);
        TEST_ASSERT_EQUAL_INT32_ARRAY(r.refCodes, r.codes, r.n);
    }
}

/* Worst case at m16 into an m16 wire: every code -2^15 (products 2^30, the
 * segment limit is one product, g = 1 at outF = 3). Exact against the
 * reference: 3 products of 2^30 * 2^-2 = 3 * 2^28 -> m16 code 24576 under
 * stored E = 15. */
void testLinearBackwardBfpDxWorstCaseM16IntoBfpWireIsExact(void) {
    lbDxCase_t c = {
        .batch = 2, .inF = 4, .outF = 3, .m = 16, .lossGs = 0, .wGs = 0, .wireM = 16, .wireGs = 4};
    int32_t lossCodes[6];
    int32_t wCodes[12];
    for (size_t i = 0; i < 6; i++) {
        lossCodes[i] = -32768;
    }
    for (size_t i = 0; i < 12; i++) {
        wCodes[i] = -32768;
    }
    uint8_t lossExps[] = {127 + 3};
    uint8_t wExps[] = {127 - 5};
    lbDxResult_t r;
    lbDxRun(&c, lossCodes, lossExps, wCodes, wExps, HALF_AWAY, 0, &r);

    TEST_ASSERT_TRUE(r.st.fitsInt32);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(r.refStored, r.stored, r.numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(r.refCodes, r.codes, r.n);
    TEST_ASSERT_EQUAL_INT32(24576, r.codes[0]);
    TEST_ASSERT_EQUAL_UINT8(127 + 15, r.stored[0]);
}

/* The training-side seam: the dx wire rounds by the OP's mode (SR), not by
 * the wire template's (HALF_AWAY, restored afterwards), drawing from the op's
 * RNG_TAG_OUT stream one draw per element in element order. Exact against
 * the reference fed the same stream; non-vacuous: SR differs from HALF_AWAY. */
void testLinearBackwardBfpDxIntoBfpWireSrFollowsTheOpModeAndTheOutStream(void) {
    lbDxCase_t c = {
        .batch = 2, .inF = 8, .outF = 6, .m = 8, .lossGs = 4, .wGs = 4, .wireM = 6, .wireGs = 8};
    int32_t lossCodes[12];
    int32_t wCodes[48];
    uint8_t lossExps[3];
    uint8_t wExps[12];
    lbDxOperands(4711u, 12, 3, 48, 12, 8, lossCodes, lossExps, wCodes, wExps);
    lbDxResult_t sr;
    lbDxRun(&c, lossCodes, lossExps, wCodes, wExps, SR_HALF_AWAY, 2026u, &sr);
    lbDxResult_t ha;
    lbDxRun(&c, lossCodes, lossExps, wCodes, wExps, HALF_AWAY, 0, &ha);

    TEST_ASSERT_EQUAL_INT(HALF_AWAY, sr.wireModeAfter);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(sr.refStored, sr.stored, sr.numGroups);
    TEST_ASSERT_EQUAL_INT32_ARRAY(sr.refCodes, sr.codes, sr.n);
    TEST_ASSERT_FALSE_MESSAGE(memcmp(sr.codes, ha.codes, sr.n * sizeof(int32_t)) == 0,
                              "SR and HALF_AWAY agree: the SR test is vacuous");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConvertInt32ToSymInt32UserTensorStaysCloneable);
    RUN_TEST(testLinearForwardPerTensorAsym12WeightWrapsAborts);
    RUN_TEST(testLinearForwardPerTensorAsym16WeightWrapsAborts);
    RUN_TEST(testLinearForwardPerTensorAsym12InputWrapsAborts);
    RUN_TEST(testLinearForwardPerTensorAsymWidthsThatFitAreExact);
    RUN_TEST(testLinearForwardInt32InputWrapsAborts);
    RUN_TEST(testLinearForwardInt32InputThatFitsIsExact);
    RUN_TEST(testLinearForwardInt32InputWiderThanDefaultGridIsExact);
    RUN_TEST(testLinearForwardGroupedAsymWideMantissaAbortsInsteadOfWrapping);
    RUN_TEST(testLinearForwardFloat);
    RUN_TEST(testLinearForwardFloatRank1BiasRank2Output);
    RUN_TEST(testLinearBackwardFloat);
    RUN_TEST(testLinearBackwardFloatRank1Bias);

    RUN_TEST(testLinearForwardSymInt32);
    RUN_TEST(testLinearForwardSymInt32Rank1BiasRank2Output);
    RUN_TEST(testLinearBackwardSymInt32);
    RUN_TEST(testLinearBackwardSymInt32Rank1Bias);
    RUN_TEST(testLinearLayerInitAndFreeRoundTrip);

    RUN_TEST(testLinearBackwardPackedSymWeightGradFixedScaleFirstStore);
    RUN_TEST(testLinearBackwardZeroInitAccModeDies);

    RUN_TEST(testLinearBackwardFloatWithMismatchedQuantizations);
    RUN_TEST(testLinearLayerInitNonTrainable);

    RUN_TEST(testLinearLayerInitBorrowingBuildsLayerWithCorrectShapeAndStoresQuantPointers);
    RUN_TEST(testLinearLayerInitBorrowingZeroInChannelsAbortsViaPrintError);
    RUN_TEST(testLinearLayerInitBorrowingBiasDefaultResolvesToTrue);
    RUN_TEST(testLinearLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testLinearLayerInitDefaultGradStorageIsFloat32DespiteSymPropLossQ);
    RUN_TEST(testLinearLayerInitSymInt32BackwardMathYieldsSymInt32Grad);
    RUN_TEST(testLinearSymInt32GradAccumulatesOverTwoMicrobatchesAndSteps);

    RUN_TEST(testLinearLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testLinearLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testLinearLayerInitOwningWeightGradStorageKnobOverridesPropLossQDefault);
    RUN_TEST(testLinearLayerInitOwningBoolWeightGradStorageKnobAborts);
    RUN_TEST(testLinearLayerInitOwningBfpWeightGradStorageKnob);
    RUN_TEST(testLinearLayerInitDefaultWeightsWithinPyTorchBound);
    RUN_TEST(testLinearLayerInitXavierUniformOverrideUsesGlorotBound);
    RUN_TEST(testLinearInitDrawsWeightAndBiasFromTheParamStreamsOfItsKey);
    RUN_TEST(testLinearInitIsAFunctionOfItsKeyAlone);
    RUN_TEST(testLinearInitWithoutAKeyExits);
    RUN_TEST(testLinearBackwardWithoutBiasDoesNotCrash);
    RUN_TEST(testLinearForwardWithoutBiasDoesNotCrash);

    RUN_TEST(testLinearFactoryFrozenElidesGrads);
    RUN_TEST(testLinearFactoryDefaultAllocatesGrads);
    RUN_TEST(testLinearBackwardFrozenTwinPropLossIdenticalGradsZero);
    RUN_TEST(testLinearBackwardFrozenFactoryLayerRunsWithoutGradBuffers);
    RUN_TEST(testLinearForwardOpDrawsFromItsForwardOpKey);
    RUN_TEST(testLinearBackwardDxOpDrawsFromItsBackwardOpKey);
    RUN_TEST(testLinearForwardWithoutAContextHandsTheFunnelNoStream);
    RUN_TEST(testLinearSrForwardWithoutAContextExits);
    RUN_TEST(testLinearSrForwardIsAPureFunctionOfItsKey);
    RUN_TEST(testLinearBackwardNullPropLossComputesGradsOnly);
    RUN_TEST(testLinearForwardGroupedSymWeights);
    RUN_TEST(testLinearForwardGroupedFloatPathAgreesWithinTolerance);
    RUN_TEST(testLinearBackwardGroupedSymWeightsDxMatchesGold);
    RUN_TEST(testLinearBackwardGroupedDxFloatPathAgreesWithinTolerance);
    RUN_TEST(testLinearForwardGroupedAsymWeightsMatchesGold);
    RUN_TEST(testLinearForwardGroupedAsymFloatPathAgreesWithinTolerance);
    RUN_TEST(testLinearForwardGroupedAsymEqualScalesEqualZpBitIdenticalToPerTensor);
    RUN_TEST(testLinearBackwardGroupedAsymDxMatchesGold);
    RUN_TEST(testLinearForwardBfpNativeMatchesKernelGold);
    RUN_TEST(testLinearForwardBfpStagesFloat32Bias);
    RUN_TEST(testLinearForwardBfpPowerOfTwoBitIdenticalToGroupedSymLayer);
    RUN_TEST(testLinearForwardBfpRejectsFloat32Weights);
    RUN_TEST(testLinearBackwardBfpWeightGradNativeMatchesFloatReference);
    RUN_TEST(testLinearBackwardBfpBiasGradNativeSumsLoss);
    RUN_TEST(testLinearBackwardBfpBiasGradGroupedLossFoldsPerGroup);
    RUN_TEST(testLinearCalcBiasGradsBfpRejectsFloat32Loss);
    RUN_TEST(testLinearBackwardBfpPropLossNativeMatchesFloatReference);
    RUN_TEST(testLinearCalcOutputShapeRejectsRank1Input);
    RUN_TEST(testLinearCalcOutputShapeRejectsRank3Input);
    RUN_TEST(testInferenceOneLinearModelRejectsRank1Sample);
    RUN_TEST(testLinearInitConfigClearsOwnsQuantizations);
    RUN_TEST(testLinearForwardFloatRejectsWrongOutputShape);
    RUN_TEST(testLinearForwardSymRejectsWrongOutputShape);
    RUN_TEST(testLinearForwardFloatRejectsTransposedOutput);
    RUN_TEST(testLinearBackwardFloatRejectsWrongPropLossShape);
    RUN_TEST(testLinearBackwardBfpRejectsWrongPropLossShape);
    RUN_TEST(testLinearBackwardFloatRejectsWrongLossShape);
    RUN_TEST(testLinearBackwardFloatRejectsTransposedLoss);
    RUN_TEST(testLinearBackwardFloatDenseLossBatch2BiasGradSumsBatch);
    RUN_TEST(testLinearCalcOutputShapeUsesLogicalBatchOfTransposedInput);
    RUN_TEST(testLinearForwardFloatTransposedInputMatchesDense);
    RUN_TEST(testLinearBiasGradSymSumProofBoundary16Bit);
    RUN_TEST(testLinearBiasGradSymSumWideWireExactThenAborts);
    RUN_TEST(testLinearBiasGradSymSumBannerNamesWireWidth);
    RUN_TEST(testLinearBiasGradSymSumIntermediateOverflowAborts);
    RUN_TEST(testLinearBiasGradSymRejectsIndivisibleLoss);
    RUN_TEST(testLinearBiasGradSymStampsTheSumWidth);
    RUN_TEST(testLinearBiasGradSymStampIs32WhenTheSumIsUnprovable);
    RUN_TEST(testLinearWeightGradSymIsStampedByTheMatmul);
    RUN_TEST(testLinearBfpGradsIntoBfpStorageMatchIntegerReferenceOverMicroBatches);
    RUN_TEST(testLinearBfpGradsIntoGroupedBfpStorageStraddlingRowsMatchReference);
    RUN_TEST(testLinearBfpGradsIntoSrBfpStorageDrawFromTheAccStream);
    RUN_TEST(testLinearBfpGradsCancelToTheZeroState);
    RUN_TEST(testLinearBfpGradsExponentExtremes);
    RUN_TEST(testLinearBfpBiasGradSegmentsFoldExactlyUnderTheSumGuard);
    RUN_TEST(testLinearBfpGradsFixedScaleKeepTheFloatBridge);
    RUN_TEST(testLinearCalcBiasGradsBfpRawRejectsANonRawOutput);
    RUN_TEST(testLinearCalcWeightGradsBfpRawMatchesTheGemmReference);
    RUN_TEST(testLinearCalcPropLossBfpEmitsTheGemmRaw);
    RUN_TEST(testLinearCalcBiasGradsBfpRawExponentFollowsTheDataNotTheBound);
    RUN_TEST(testLinearBfpRawGradsZeroLossGroupDoesNotSetTheAccumulator);
    RUN_TEST(testLinearBackwardBfpDxIntoBfpWireMatchesIntegerReference);
    RUN_TEST(testLinearBackwardBfpDxWorstCaseM16IntoBfpWireIsExact);
    RUN_TEST(testLinearBackwardBfpDxIntoBfpWireSrFollowsTheOpModeAndTheOutStream);
    return UNITY_END();
}
