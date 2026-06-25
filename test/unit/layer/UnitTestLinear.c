#include <math.h>
#include <string.h>

#include "DTypes.h"
#include "Layer.h"
#include "LayerCommon.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "Optimizer.h"
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
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    parameter_t *weights = parameterInit(weightsParam, NULL);

    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    parameter_t *bias = parameterInit(biasParam, NULL);

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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, testQ, testQ, testQ, testQ);

    linearForward(linearLayer, input, output);

    float captured[2];
    captured[0] = ((float *)output->data)[0];
    captured[1] = ((float *)output->data)[1];

    freeLinearLayerLegacy(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, test, test, test, test);

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
    freeLinearLayerLegacy(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
    freeQuantization(test);

    float expected[] = {-5.f, -4.f};
    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], captured[i]);
    }
}

void testLinearBackwardFloatRank1Bias() {
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* RANK-1 bias [2] -> rank-1 bias grad. */
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

    quantization_t *testQ = quantizationInitFloat();
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, testQ, testQ, testQ, testQ);

    linearBackward(linearLayer, forwardInput, loss, propLoss);

    size_t numWeightElements = calcNumberOfElementsByShape(weights->param->shape);
    size_t numBiasElements = calcNumberOfElementsByShape(bias->param->shape);
    size_t numPropLossElements = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < numWeightElements; i++) {
        capturedWeightGrad[i] = ((float *)weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numBiasElements; i++) {
        capturedBiasGrad[i] = ((float *)bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numPropLossElements; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    freeLinearLayerLegacy(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeParameter(bias);
    freeParameter(weights);
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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, test, test, test, test);

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
    freeLinearLayerLegacy(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeParameter(bias);
    freeParameter(weights);
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

void setUp() {}
void tearDown() {}

void testLinearForwardFloat() {
    /* 1. Build heap weights tensor (shape 2x3) wrapped in a parameter (no grad). */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    parameter_t *weights = parameterInit(weightsParam, NULL);

    /* 2. Build heap bias tensor (shape 2,). */
    size_t *biasDims = reserveMemory(1 * sizeof(size_t));
    biasDims[0] = 2;
    size_t *biasOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 1, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    parameter_t *bias = parameterInit(biasParam, NULL);

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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, testQ, testQ, testQ, testQ);

    linearForward(linearLayer, input, output);

    /* 6. CAPTURE. */
    float captured[2];
    captured[0] = ((float *)output->data)[0];
    captured[1] = ((float *)output->data)[1];

    /* 7. FREE. freeLinearLayer releases only the layer config wrapper. */
    freeLinearLayerLegacy(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
    freeQuantization(testQ);

    /* 8. ASSERT. */
    float expected[] = {-5.f, -4.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, captured, 2);
}

void testLinearBackwardFloat() {
    /* 1. Build heap weights parameter (param + grad), shape 2x3. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* 2. Build heap bias parameter (param + grad), shape 1x2. */
    size_t *biasDims = reserveMemory(2 * sizeof(size_t));
    biasDims[0] = 1;
    biasDims[1] = 2;
    size_t *biasOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 2, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, testQ, testQ, testQ, testQ);

    linearBackward(linearLayer, forwardInput, loss, propLoss);

    /* 7. CAPTURE. */
    size_t numWeightElements = calcNumberOfElementsByShape(weights->param->shape);
    size_t numBiasElements = calcNumberOfElementsByShape(bias->param->shape);
    size_t numPropLossElements = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < numWeightElements; i++) {
        capturedWeightGrad[i] = ((float *)weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < numBiasElements; i++) {
        capturedBiasGrad[i] = ((float *)bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < numPropLossElements; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    /* 8. FREE. */
    freeLinearLayerLegacy(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeParameter(bias);
    freeParameter(weights);
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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, test, test, test, test);

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
    freeLinearLayerLegacy(linearLayer);
    freeTensor(output);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, test, test, test, test);

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
    freeLinearLayerLegacy(linearLayer);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(forwardInput);
    freeParameter(bias);
    freeParameter(weights);
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

void testLinearBackwardFloatWithMismatchedQuantizations() {
    /* Mismatched-quantization variant of testLinearBackwardFloat: the loss
     * arrives in ASYM form, while the layer's parameters and propLoss are
     * Float. Validates that linearBackward routes the loss through a
     * conversion before applying it. */

    /* 1. Build heap weights parameter (Float, shape 2x3) with grad. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, -6.f}, 6);
    tensor_t *weightsGrad = gradInitFloat(weightsParam, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    /* 2. Build heap bias parameter (Float, shape 1x2) with grad. */
    size_t *biasDims = reserveMemory(2 * sizeof(size_t));
    biasDims[0] = 1;
    biasDims[1] = 2;
    size_t *biasOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, biasOrder);
    shape_t *biasShape = reserveMemory(sizeof(shape_t));
    setShape(biasShape, biasDims, 2, biasOrder);
    tensor_t *biasParam = initTensor(biasShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(biasParam, (float[]){-1.f, 3.f}, 2);
    tensor_t *biasGrad = gradInitFloat(biasParam, NULL);
    parameter_t *bias = parameterInit(biasParam, biasGrad);

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
    layer_t *linearLayer = linearLayerInitLegacy(weights, bias, testQ, testQ, testQ, testQ);

    linearBackward(linearLayer, forwardInput, lossAsym, propLoss);

    /* 7. CAPTURE. */
    size_t sizeWeights = calcNumberOfElementsByParameter(weights);
    size_t sizeBias = calcNumberOfElementsByParameter(bias);
    size_t sizePropLoss = calcNumberOfElementsByTensor(propLoss);

    float capturedWeightGrad[6];
    for (size_t i = 0; i < sizeWeights; i++) {
        capturedWeightGrad[i] = ((float *)weights->grad->data)[i];
    }
    float capturedBiasGrad[2];
    for (size_t i = 0; i < sizeBias; i++) {
        capturedBiasGrad[i] = ((float *)bias->grad->data)[i];
    }
    float capturedPropLoss[3];
    for (size_t i = 0; i < sizePropLoss; i++) {
        capturedPropLoss[i] = ((float *)propLoss->data)[i];
    }

    /* 8. FREE. */
    freeLinearLayerLegacy(linearLayer);
    freeTensor(propLoss);
    freeTensor(lossAsym);
    freeTensor(forwardInput);
    freeParameter(bias);
    freeParameter(weights);
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

    /* 3. Build the non-trainable layer. */
    quantization_t *forwardQ = quantizationInitFloat();
    layer_t *layer = linearLayerInitNonTrainableLegacy(weights, bias, forwardQ);

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

    /* 7. FREE. linearLayerInitNonTrainable wrapped weights/bias into
     *    parameter_t* via parameterInit; freeing those parameters
     *    cascades to freeTensor(weights) and freeTensor(bias).
     *    freeParameter is NULL-grad-safe (post-#106, H3). */
    parameter_t *weightsParam = config->weights;
    parameter_t *biasParam = config->bias;
    freeLinearLayerLegacy(layer);
    freeTensor(output);
    freeTensor(input);
    freeParameter(biasParam);
    freeParameter(weightsParam);
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
     * inner linearConfig (3 reserveMemory calls). freeLinearLayer must
     * release all three. Pre-fix this test runs to completion but leaks
     * the outer layerConfig wrapper; post-fix it is leak-clean (verified
     * via the LSan sweep).
     *
     * linearLayerInit only stores the parameter and quantization pointers
     * without dereferencing them, and freeLinearLayer does not touch
     * parameters/quantization (those are externally owned). So NULL is a
     * valid stand-in here — keeps the test focused on the StorageApi
     * lifecycle, not on tensor ownership. */
    layer_t *linearLayer = linearLayerInitLegacy(NULL, NULL, NULL, NULL, NULL, NULL);
    TEST_ASSERT_NOT_NULL(linearLayer);
    TEST_ASSERT_EQUAL_INT(LINEAR, linearLayer->type);
    TEST_ASSERT_NOT_NULL(linearLayer->config);
    TEST_ASSERT_NOT_NULL(linearLayer->config->linear);

    freeLinearLayerLegacy(linearLayer);
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

    /* Borrowing variant stores pointers verbatim */
    TEST_ASSERT_EQUAL_PTR(q, cfg->forwardQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->weightGradQ);
    TEST_ASSERT_EQUAL_PTR(q, cfg->biasGradQ);
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

void testLinearLayerInitSymInt32BackwardMathYieldsSymInt32Grad(void) {
    /* Regression for the "config lies" bug: a Linear built with a SYM_INT32
     * backwardMath must store SYM_INT32 parameter gradients, not FLOAT32. */
    quantization_t *fwd = quantizationInitFloat();             /* FLOAT32 forward + storage */
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY); /* SYM_INT32 backward */
    layerQuant_t lq = {
        .forwardMath = fwd,
        .backwardMath = bwd,
        .weightStorage = fwd, /* KAIMING init requires FLOAT32 weight storage */
        .biasStorage = fwd,
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

    /* Owning variant: cfg->forwardQ is a fresh allocation, NOT the original q */
    TEST_ASSERT_NOT_EQUAL(q, cfg->forwardQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->weightGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->biasGradQ);
    TEST_ASSERT_NOT_EQUAL(q, cfg->propLossQ);

    /* But the copy has equal type to the original */
    TEST_ASSERT_EQUAL_INT(q->type, cfg->forwardQ->type);

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

void testLinearSymInt32GradAccumulatesOverTwoMicrobatchesAndSteps(void) {
    /* outFeatures=2, inFeatures=3. forward input [1,2,3], loss [0.5, -0.25].
     * Two identical microbatches => accumulated weight grad ~= 2 * (loss^T @ input). */
    const float inputVals[3] = {1.0f, 2.0f, 3.0f};
    const float lossVals[2] = {0.5f, -0.25f};

    /* ---- SYM_INT32-backward layer (under test) ---- */
    quantization_t *fwd = quantizationInitFloat();
    quantization_t *bwd = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym = {
        .forwardMath = fwd, .backwardMath = bwd, .weightStorage = fwd, .biasStorage = fwd};
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
    optimizer_t *symOptim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, symModel, 1, SYM_INT32);
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
    layerQuant_t lqF = {
        .forwardMath = fwd2, .backwardMath = bwd2, .weightStorage = fwd2, .biasStorage = fwd2};
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

    /* freeOptimSgdM frees the SYM layer's parameters; do NOT also freeLinearLayer(symLayer)
     * (double-free). Free the layer/config shell manually (borrowing factory: caller owns
     * the quantizations, freed separately below). */
    freeOptimSgdM(symOptim);
    freeReservedMemory(symLayer->config->linear);
    freeReservedMemory(symLayer->config);
    freeReservedMemory(symLayer);
    freeTensor(symProp2);
    freeTensor(symLoss2);
    freeTensor(symIn2);
    freeTensor(symProp1);
    freeTensor(symLoss1);
    freeTensor(symIn1);
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

    RUN_TEST(testLinearBackwardFloatWithMismatchedQuantizations);
    RUN_TEST(testLinearLayerInitNonTrainable);

    RUN_TEST(testLinearLayerInitBorrowingBuildsLayerWithCorrectShapeAndStoresQuantPointers);
    RUN_TEST(testLinearLayerInitBorrowingZeroInChannelsAbortsViaPrintError);
    RUN_TEST(testLinearLayerInitBorrowingBiasDefaultResolvesToTrue);
    RUN_TEST(testLinearLayerInitBorrowingBiasFalseLeavesBiasNull);
    RUN_TEST(testLinearLayerInitSymInt32BackwardMathYieldsSymInt32Grad);
    RUN_TEST(testLinearSymInt32GradAccumulatesOverTwoMicrobatchesAndSteps);

    RUN_TEST(testLinearLayerInitOwningDeepCopiesQuantizations);
    RUN_TEST(testLinearLayerInitOwningFreesAllAllocationsWithoutLeak);
    RUN_TEST(testLinearLayerInitDefaultWeightsWithinPyTorchBound);
    RUN_TEST(testLinearLayerInitXavierUniformOverrideUsesGlorotBound);
    return UNITY_END();
}
