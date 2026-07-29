#include <math.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DTypes.h"
#include "DeathTest.h"
#include "Layer.h"
#include "LayerCommon.h"
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
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

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
    layer_t *linearLayer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});

    linearForward(linearLayer, input, output);

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

    linearForward(linearLayer, input, output);

    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 1;
    outFloatDims[1] = 2;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

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
    layer_t *linearLayer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, loss, propLoss);

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

    /* 5. Build heap propLoss tensor (SymInt32, shape (3,)). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfForwardInputs;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 6. Build layer (shared SymInt32 quantization). */
    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearBackward(linearLayer, forwardInput, loss, propLoss);

    /* 7. Convert SymInt32 grads back to Float for comparison. */
    size_t *wgDims = reserveMemory(2 * sizeof(size_t));
    wgDims[0] = 2;
    wgDims[1] = 3;
    size_t *wgOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wgOrder);
    shape_t *wgShape = reserveMemory(sizeof(shape_t));
    setShape(wgShape, wgDims, 2, wgOrder);
    tensor_t *weightGradFloat = initTensor(wgShape, quantizationInitFloat(), NULL);
    convertTensor(weights->grad, weightGradFloat);

    /* RANK-1 bias-grad convert-back block. */
    size_t *bgDims = reserveMemory(1 * sizeof(size_t));
    bgDims[0] = 2;
    size_t *bgOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bgOrder);
    shape_t *bgShape = reserveMemory(sizeof(shape_t));
    setShape(bgShape, bgDims, 1, bgOrder);
    tensor_t *biasGradFloat = initTensor(bgShape, quantizationInitFloat(), NULL);
    convertTensor(bias->grad, biasGradFloat);

    size_t *plDims = reserveMemory(1 * sizeof(size_t));
    plDims[0] = numberOfForwardInputs;
    size_t *plOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, plOrder);
    shape_t *plShape = reserveMemory(sizeof(shape_t));
    setShape(plShape, plDims, 1, plOrder);
    tensor_t *propLossFloat = initTensor(plShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat);

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
    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .trainable = trainable}, &lq);
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

    /* 4. Build heap output tensor (shape 2,). */
    size_t *outputDims = reserveMemory(1 * sizeof(size_t));
    outputDims[0] = 2;
    size_t *outputOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, outputOrder);
    shape_t *outputShape = reserveMemory(sizeof(shape_t));
    setShape(outputShape, outputDims, 1, outputOrder);
    tensor_t *output = initTensor(outputShape, quantizationInitFloat(), NULL);

    /* 5. Build the layer with shared float quantization. */
    quantization_t *testQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, testQ);
    layer_t *linearLayer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});

    linearForward(linearLayer, input, output);

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
    layer_t *linearLayer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, loss, propLoss);

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

    linearForward(linearLayer, input, output);

    /* 6. Convert SymInt32 output back to Float for comparison. */
    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 1;
    outFloatDims[1] = 2;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

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

    /* 5. Build heap propLoss tensor (SymInt32, shape (3,)). */
    size_t *propLossDims = reserveMemory(1 * sizeof(size_t));
    propLossDims[0] = numberOfForwardInputs;
    size_t *propLossOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, propLossOrder);
    shape_t *propLossShape = reserveMemory(sizeof(shape_t));
    setShape(propLossShape, propLossDims, 1, propLossOrder);
    tensor_t *propLoss = initTensor(propLossShape, quantizationInitSymInt32(HALF_AWAY), NULL);

    /* 6. Build layer (shared SymInt32 quantization). */
    quantization_t *test = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linearLayer = buildBorrowedLinearLayer(weights, bias, test);

    linearBackward(linearLayer, forwardInput, loss, propLoss);

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
    convertTensor(weights->grad, weightGradFloat);

    size_t *bgDims = reserveMemory(2 * sizeof(size_t));
    bgDims[0] = 1;
    bgDims[1] = 2;
    size_t *bgOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, bgOrder);
    shape_t *bgShape = reserveMemory(sizeof(shape_t));
    setShape(bgShape, bgDims, 2, bgOrder);
    tensor_t *biasGradFloat = initTensor(bgShape, quantizationInitFloat(), NULL);
    convertTensor(bias->grad, biasGradFloat);

    size_t *plDims = reserveMemory(1 * sizeof(size_t));
    plDims[0] = numberOfForwardInputs;
    size_t *plOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, plOrder);
    shape_t *plShape = reserveMemory(sizeof(shape_t));
    setShape(plShape, plDims, 1, plOrder);
    tensor_t *propLossFloat = initTensor(plShape, quantizationInitFloat(), NULL);
    convertTensor(propLoss, propLossFloat);

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

/* PR3 Task 4 (D1): weightGradAccMode=OUT_ACC_FIXED_SCALE routed to a packed
 * SYM@8 weight-grad target. The freshly-allocated grad (gradInitSym) starts
 * all-zero mantissas at scale=1.0 -- one backward call is therefore the
 * "first store" path (spec 2026-07-03 PR3 §4.1): the grid is derived from the
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

    linearBackward(linearLayer, forwardInput, loss, propLoss);

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
     * grid established by call 1 (spec D1 -- no re-derivation, no renorm),
     * and the exactly-opposite increment drives every mantissa back toward
     * (near-)zero -- safely within the established grid, no overflow risk.
     * If the weight-grad call site were cross-wired to read biasGradAccMode
     * (DYNAMIC_RESCALE) instead, the near-zero recomputed values would force
     * a fresh, much smaller (or absMax==0 -> 1.0) absmax-derived scale --
     * clearly different from the carried scaleAfterCall1. */
    tensor_t *loss2 = initTensor(getShapeLike(loss->shape), quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(loss2, (float[]){4.f, 3.f}, 2);
    linearBackward(linearLayer, forwardInput, loss2, propLoss);
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
        "FIXED_SCALE must carry the grid across calls (D1) -- a scale change here "
        "means the weight-grad call site is reading the wrong accMode field");
}

/* PR3 Task 4 (D1) hazard guard: the same fixture as above, but
 * weightGradAccMode is (deliberately) left at its zero-init value -- OUT_WRITE
 * happens to be 0, so a hand-wired config that forgets to set the new field
 * would otherwise silently overwrite instead of accumulate (spec 2026-07-03
 * PR3 §3). linearBackward must fail fast instead. */
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

    ASSERT_EXITS_WITH_FAILURE(linearBackward(linearLayer, forwardInput, loss, propLoss));

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
    layer_t *linearLayer =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linearLayer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, (float[]){-1.f, 3.f});
    linearConfig_t *cfg = linearLayer->config->linear;

    linearBackward(linearLayer, forwardInput, lossAsym, propLoss);

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

    linearForward(layer, input, output);

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
    layer_t *linearLayer = linearLayerInit(&(linearInit_t){.inFeatures = 1, .outFeatures = 1}, &lq);
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

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
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

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4,
            .outFeatures = 1,
            /* .bias omitted -> BIAS_DEFAULT (0) -> resolves to true */
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

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 4,
            .outFeatures = 1,
            .bias = BIAS_FALSE,
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

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
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

    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
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

    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){
            .inFeatures = 3,
            .outFeatures = 2,
            .bias = BIAS_TRUE,
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

        layer_t *layer = linearLayerInitOwning(
            &(linearInit_t){
                .inFeatures = 8,
                .outFeatures = 4,
                .bias = BIAS_TRUE,
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

    layer_t *layer = linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);

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

    ASSERT_EXITS_WITH_FAILURE(linearLayerInitOwning(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq));

    freeQuantization(boolGradQ);
    freeQuantization(q);
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

    linearBackward(trainableTwin, forwardInput, loss, propLossTrainable);
    linearBackward(frozenTwin, forwardInput, loss, propLossFrozen);

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

    linearBackward(layer, forwardInput, loss, propLoss);

    bool gradStillNull = layer->config->linear->weights->grad == NULL;

    freeLinearLayer(layer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);

    TEST_ASSERT_TRUE(gradStillNull);
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

    linearBackward(twinA, forwardInput, loss, propLossA);
    linearBackward(twinB, forwardInput, loss, NULL);

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
    layer_t *symLayer = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lqSym);

    tensor_t *symWGrad = symLayer->config->linear->weights->grad;
    TEST_ASSERT_EQUAL_INT(SYM_INT32, symWGrad->quantization->type); /* guard */

    tensor_t *symIn1 = e2eMakeInput(inputVals);
    tensor_t *symLoss1 = e2eMake1x2(lossVals);
    tensor_t *symProp1 = e2eMakeSym1x3(); /* propLoss [1,3], SYM_INT32 (matches propLossQ) */
    linearBackward(symLayer, symIn1, symLoss1, symProp1);

    tensor_t *symIn2 = e2eMakeInput(inputVals);
    tensor_t *symLoss2 = e2eMake1x2(lossVals);
    tensor_t *symProp2 = e2eMakeSym1x3();
    linearBackward(symLayer, symIn2, symLoss2, symProp2);

    /* Capture accumulated SYM_INT32 weight grad as float (dequantized). */
    size_t nW = calcNumberOfElementsByTensor(symWGrad);
    float symGradFloat[6];
    {
        tensor_t gf;
        quantization_t gfQ;
        initFloat32Quantization(&gfQ);
        uint8_t gfData[6 * sizeof(float)];
        setTensorValuesForConversion(gfData, &gfQ, symWGrad, &gf);
        convertTensor(symWGrad, &gf);
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
    optimizerFunctions[symOptim->type].step(symOptim);
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
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lqF);
    tensor_t *fWGrad = fLayer->config->linear->weights->grad;

    tensor_t *fIn1 = e2eMakeInput(inputVals);
    tensor_t *fLoss1 = e2eMake1x2(lossVals);
    tensor_t *fProp1 = e2eMakeInput(NULL);
    linearBackward(fLayer, fIn1, fLoss1, fProp1);
    tensor_t *fIn2 = e2eMakeInput(inputVals);
    tensor_t *fLoss2 = e2eMake1x2(lossVals);
    tensor_t *fProp2 = e2eMakeInput(NULL);
    linearBackward(fLayer, fIn2, fLoss2, fProp2);

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

    rngSetSeed(7);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = inFeatures,
            .outFeatures = outFeatures,
            .bias = BIAS_TRUE,
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

    rngSetSeed(7);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){
            .inFeatures = inFeatures,
            .outFeatures = outFeatures,
            .bias = BIAS_TRUE,
            .weightInit = {INIT_XAVIER_UNIFORM},
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

void testLinearBackwardWithoutBiasDoesNotCrash(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *layer = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_FALSE}, &lq);

    tensor_t *fwdIn = buildFloatTensor2d(1, 3, (float[]){1.f, 2.f, 3.f});
    tensor_t *loss = buildFloatTensor2d(1, 2, (float[]){0.5f, -0.5f});
    tensor_t *propLoss = buildFloatTensor2d(1, 3, (float[]){0.f, 0.f, 0.f});

    layerFunctions[LINEAR].backward(layer, fwdIn, loss, propLoss);

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
    layer_t *layer = linearLayerInit(
        &(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_FALSE}, &lq);
    layerLoadWeights(layer, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, NULL);

    tensor_t *input = buildFloatTensor2d(1, 3, (float[]){0.f, 1.f, 2.f});
    tensor_t *output = buildFloatTensor2d(1, 2, (float[]){0.f, 0.f});

    layerFunctions[LINEAR].forward(layer, input, output);

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

    linearForward(linearLayer, input, output);

    size_t *outFloatDims = reserveMemory(2 * sizeof(size_t));
    outFloatDims[0] = 2;
    outFloatDims[1] = 3;
    size_t *outFloatOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outFloatOrder);
    shape_t *outFloatShape = reserveMemory(sizeof(shape_t));
    setShape(outFloatShape, outFloatDims, 2, outFloatOrder);
    tensor_t *outputFloat = initTensor(outFloatShape, quantizationInitFloat(), NULL);
    convertTensor(output, outputFloat);

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

    linearForward(linearLayer, input, output);

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

    linearBackward(linearLayer, input, loss, propLoss);

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

int main(void) {
    UNITY_BEGIN();
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
    RUN_TEST(testLinearLayerInitDefaultWeightsWithinPyTorchBound);
    RUN_TEST(testLinearLayerInitXavierUniformOverrideUsesGlorotBound);
    RUN_TEST(testLinearBackwardWithoutBiasDoesNotCrash);
    RUN_TEST(testLinearForwardWithoutBiasDoesNotCrash);

    RUN_TEST(testLinearFactoryFrozenElidesGrads);
    RUN_TEST(testLinearFactoryDefaultAllocatesGrads);
    RUN_TEST(testLinearBackwardFrozenTwinPropLossIdenticalGradsZero);
    RUN_TEST(testLinearBackwardFrozenFactoryLayerRunsWithoutGradBuffers);
    RUN_TEST(testLinearBackwardNullPropLossComputesGradsOnly);
    RUN_TEST(testLinearForwardGroupedSymWeights);
    RUN_TEST(testLinearForwardGroupedFloatPathAgreesWithinTolerance);
    RUN_TEST(testLinearBackwardGroupedSymWeightsDxMatchesGold);
    RUN_TEST(testLinearBackwardGroupedDxFloatPathAgreesWithinTolerance);
    return UNITY_END();
}
