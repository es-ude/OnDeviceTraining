#include <stdbool.h>
#include <stdint.h>

#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testInferenceLinearReluFloat() {
    /* 1. Build heap-allocated quantization (shared across both layers). */
    quantization_t *q = quantizationInitFloat();

    /* 2. Build weights tensor (shape 2x3) and its grad. */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, 6);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    /* 3. Build bias tensor (shape 1x2) and its grad. */
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

    /* 4. Build input tensor (shape 1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* 5. Build layers; both layers share the same q (no ownership transfer). */
    layer_t *linear = linearLayerInitLegacy(weights, bias, q, q, q, q);
    layer_t *relu = reluLayerInitLegacy(q, q);
    layer_t *model[] = {linear, relu};

    /* 6. Exercise the system. */
    tensor_t *output = inference(model, 2, input);

    /* 7. CAPTURE assertion values into stack locals BEFORE frees. */
    float capturedOutput[2];
    capturedOutput[0] = ((float *)output->data)[0];
    capturedOutput[1] = ((float *)output->data)[1];

    /* 8. FREE in reverse-init order.
     *    - inference() returns a heap tensor (via getTensorLike) — caller owns it.
     *    - freeReluLayer / freeLinearLayer release config wrappers but NOT shared q
     *      and NOT the parameters.
     *    - freeParameter cascades to param + grad via freeTensor (TensorApi.c:389).
     *    - The shared q is freed exactly once at the end. */
    freeTensor(output);
    freeReluLayerLegacy(relu);
    freeLinearLayerLegacy(linear);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
    freeQuantization(q);

    /* 9. ASSERT on captured locals. */
    float expected[] = {0.f, 20.f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, capturedOutput, 2);
}

void testInferenceLinearReluSymInt32() {
    size_t numberOfOutputs = 2;

    /* Shared SymInt32 quantization for layers. */
    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);

    /* Weights (2x3). */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightsParam = initTensor(weightShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(weightsParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, 6);
    tensor_t *weightGrad = gradInitSymInt32(weightsParam, HALF_AWAY, NULL);
    parameter_t *weights = parameterInit(weightsParam, weightGrad);

    /* Bias (1x2). */
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

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* Layers. */
    layer_t *linear = linearLayerInitLegacy(weights, bias, q, q, q, q);
    layer_t *relu = reluLayerInitLegacy(q, q);
    layer_t *model[] = {linear, relu};

    /* Run inference (returns a heap tensor in SymInt32 form). */
    tensor_t *outputSymInt32 = inference(model, 2, input);

    /* Convert SymInt32 → Float32 view for comparison. The output buffer is
     * heap-allocated to keep us in the heap-tier idiom. */
    size_t *outDims = reserveMemory(2 * sizeof(size_t));
    outDims[0] = 1;
    outDims[1] = numberOfOutputs;
    size_t *outOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, outOrder);
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    setShape(outShape, outDims, 2, outOrder);
    tensor_t *outputFloat = initTensor(outShape, quantizationInitFloat(), NULL);
    convertTensor(outputSymInt32, outputFloat);

    /* CAPTURE assertion values. */
    float capturedActual[2];
    capturedActual[0] = ((float *)outputFloat->data)[0];
    capturedActual[1] = ((float *)outputFloat->data)[1];

    /* FREE in reverse-init order. */
    freeTensor(outputFloat);
    freeTensor(outputSymInt32);
    freeReluLayerLegacy(relu);
    freeLinearLayerLegacy(linear);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
    freeQuantization(q);

    /* ASSERT on captured. */
    float expected[] = {0.f, 20.f};
    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], capturedActual[i]);
    }
}

void testInferenceWithLossLinearReluFloat() {
    quantization_t *q = quantizationInitFloat();

    /* Weights (2x3). */
    size_t *weightDims = reserveMemory(2 * sizeof(size_t));
    weightDims[0] = 2;
    weightDims[1] = 3;
    size_t *weightOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, weightOrder);
    shape_t *weightShape = reserveMemory(sizeof(shape_t));
    setShape(weightShape, weightDims, 2, weightOrder);
    tensor_t *weightParam = initTensor(weightShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(weightParam, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, 6);
    tensor_t *weightGrad = gradInitFloat(weightParam, NULL);
    parameter_t *weights = parameterInit(weightParam, weightGrad);

    /* Bias (1x2). */
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

    /* Input (1x3). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 3;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f, 2.f}, 3);

    /* Label0 (1x2). */
    size_t *label0Dims = reserveMemory(2 * sizeof(size_t));
    label0Dims[0] = 1;
    label0Dims[1] = 2;
    size_t *label0Order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, label0Order);
    shape_t *label0Shape = reserveMemory(sizeof(shape_t));
    setShape(label0Shape, label0Dims, 2, label0Order);
    tensor_t *label0 = initTensor(label0Shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label0, (float[]){59.f, -23.f}, 2);

    /* Layers. */
    layer_t *linear = linearLayerInitLegacy(weights, bias, q, q, q, q);
    layer_t *relu = reluLayerInitLegacy(q, q);
    layer_t *model[] = {linear, relu};

    /* Run inferenceWithLoss. inferenceStats owns its `output` tensor; its
     * matching free is freeInferenceStats. */
    inferenceStats_t *inferenceStats =
        inferenceWithLoss(model, 2, input, label0, MSE, REDUCTION_MEAN);

    /* CAPTURE. */
    float capturedLoss = inferenceStats->loss;
    float capturedOutput[2];
    capturedOutput[0] = ((float *)inferenceStats->output->data)[0];
    capturedOutput[1] = ((float *)inferenceStats->output->data)[1];

    /* FREE in reverse-init order. */
    freeInferenceStats(inferenceStats);
    freeReluLayerLegacy(relu);
    freeLinearLayerLegacy(linear);
    freeTensor(label0);
    freeTensor(input);
    freeParameter(bias);
    freeParameter(weights);
    freeQuantization(q);

    /* ASSERT on captured. */
    float expectedLoss = 2665.f;
    float expectedOutput[] = {0.f, 20.f};
    TEST_ASSERT_EQUAL_FLOAT(expectedLoss, capturedLoss);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOutput, capturedOutput, 2);
}

void testInferenceLayerNormSymInt32(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwd = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = symQ, .backwardMath = bwd, .weightStorage = symQ, .biasStorage = symQ};
    layer_t *ln = layerNormLayerInitOwning(&init, &lq);
    freeQuantization(bwd);
    freeQuantization(symQ);
    layer_t *model[] = {ln};

    /* [2,4] SYM input — int16-range mantissas by construction (NOT a Linear
     * output; Linear SYM outputs are accumulator-range and would violate the
     * LayerNorm input contract — open inter-layer requantization question
     * for the quantized-training epic, #137). */
    size_t *inDims = reserveMemory(2 * sizeof(size_t));
    inDims[0] = 2;
    inDims[1] = 4;
    size_t *inOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inOrder);
    shape_t *inShape = reserveMemory(sizeof(shape_t));
    setShape(inShape, inDims, 2, inOrder);
    tensor_t *input = initTensor(inShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f}, 8);

    tensor_t *output = inference(model, 1, input);

    bool typeOk = (output->quantization->type == SYM_INT32);
    float scale = ((symInt32QConfig_t *)output->quantization->qConfig)->scale;
    float deqMean = 0.f;
    for (size_t i = 0; i < 8; i++) {
        deqMean += (float)((int32_t *)output->data)[i] * scale;
    }
    deqMean /= 8.f;

    freeTensor(output); /* inference() returns a heap tensor — caller owns. */
    freeTensor(input);
    freeLayerNormLayer(ln);

    TEST_ASSERT_TRUE(typeOk);
    TEST_ASSERT_TRUE(scale > 0.0f && scale < 1e-3f);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.f, deqMean);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInferenceLinearReluFloat);
    RUN_TEST(testInferenceLinearReluSymInt32);
    RUN_TEST(testInferenceLayerNormSymInt32);

    RUN_TEST(testInferenceWithLossLinearReluFloat);
    return UNITY_END();
}
