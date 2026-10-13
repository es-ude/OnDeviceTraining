#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
#include "DropoutApi.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "ReluApi.h"
#include "RematScheduler.h"
#include "StorageApi.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void testInferenceLinearReluFloat() {
    /* 1. Build heap-allocated quantization (shared across both layers). */
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

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
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, (float[]){-1.f, 3.f});
    layer_t *relu = reluLayerInit(&lq);
    layer_t *model[] = {linear, relu};

    /* 6. Exercise the system. */
    tensor_t *output = inference(model, 2, input);

    /* 7. CAPTURE assertion values into stack locals BEFORE frees. */
    float capturedOutput[2];
    capturedOutput[0] = ((float *)output->data)[0];
    capturedOutput[1] = ((float *)output->data)[1];

    /* 8. FREE in reverse-init order.
     *    - inference() returns a heap tensor (via getTensorLike) — caller owns it.
     *    - freeReluLayer / freeLinearLayer cascade into their factory-owned
     *      parameters (weights/bias) but NOT the shared, caller-owned q.
     *    - The shared q is freed exactly once at the end. */
    freeTensor(output);
    freeReluLayer(relu);
    freeLinearLayer(linear);
    freeTensor(input);
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
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *linear = buildBorrowedLinearLayer(weights, bias, q);
    layer_t *relu = reluLayerInit(&lq);
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
    convertTensor(outputSymInt32, outputFloat, NULL);

    /* CAPTURE assertion values. */
    float capturedActual[2];
    capturedActual[0] = ((float *)outputFloat->data)[0];
    capturedActual[1] = ((float *)outputFloat->data)[1];

    /* FREE in reverse-init order. */
    freeTensor(outputFloat);
    freeTensor(outputSymInt32);
    freeReluLayer(relu);
    freeLinearLayer(linear);
    freeTensor(input);
    freeQuantization(q);

    /* ASSERT on captured. */
    float expected[] = {0.f, 20.f};
    for (size_t i = 0; i < numberOfOutputs; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], capturedActual[i]);
    }
}

void testInferenceWithLossLinearReluFloat() {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

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
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, (float[]){-1.f, 3.f});
    layer_t *relu = reluLayerInit(&lq);
    layer_t *model[] = {linear, relu};

    /* Run inferenceWithLoss. inferenceStats owns its `output` tensor; its
     * matching free is freeInferenceStats. */
    inferenceStats_t *inferenceStats =
        inferenceWithLoss(model, 2, input, label0, MSE, REDUCTION_MEAN, NULL);

    /* CAPTURE. */
    float capturedLoss = inferenceStats->loss;
    float capturedOutput[2];
    capturedOutput[0] = ((float *)inferenceStats->output->data)[0];
    capturedOutput[1] = ((float *)inferenceStats->output->data)[1];

    /* FREE in reverse-init order. */
    freeInferenceStats(inferenceStats);
    freeReluLayer(relu);
    freeLinearLayer(linear);
    freeTensor(label0);
    freeTensor(input);
    freeQuantization(q);

    /* ASSERT on captured. */
    float expectedLoss = 2665.f;
    float expectedOutput[] = {0.f, 20.f};
    TEST_ASSERT_EQUAL_FLOAT(expectedLoss, capturedLoss);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedOutput, capturedOutput, 2);
}

/* #153: labels carry the batch axis like every tensor-level input. A rank-1
 * [C] label against the model's [1, C] output (the case that once overflowed
 * reserveInferenceStats when it sized from the label) is rejected by the
 * loss before inferenceWithLoss allocates its stats. */
void testInferenceWithLossRejectsRank1LabelAgainstBatchedOutput(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* Input (1x2). */
    size_t *inputDims = reserveMemory(2 * sizeof(size_t));
    inputDims[0] = 1;
    inputDims[1] = 2;
    size_t *inputOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, inputOrder);
    shape_t *inputShape = reserveMemory(sizeof(shape_t));
    setShape(inputShape, inputDims, 2, inputOrder);
    tensor_t *input = initTensor(inputShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(input, (float[]){0.f, 1.f}, 2);

    /* Label -- RANK-1 [3] (label rank < model output rank [1,3]). */
    size_t *labelDims = reserveMemory(1 * sizeof(size_t));
    labelDims[0] = 3;
    size_t *labelOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, labelOrder);
    shape_t *labelShape = reserveMemory(sizeof(shape_t));
    setShape(labelShape, labelDims, 1, labelOrder);
    tensor_t *label = initTensor(labelShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(label, (float[]){59.f, -23.f, 4.f}, 3);

    /* Layers: Linear(2->3) + Relu -> rank-2 [1,3] output. */
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 3, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, (float[]){-1.f, 3.f, -2.f});
    layer_t *relu = reluLayerInit(&lq);
    layer_t *model[] = {linear, relu};
    size_t modelSize = 2;

    ASSERT_EXITS_WITH_FAILURE(
        (void)inferenceWithLoss(model, modelSize, input, label, MSE, REDUCTION_MEAN, NULL));

    freeReluLayer(relu);
    freeLinearLayer(linear);
    freeTensor(label);
    freeTensor(input);
    freeQuantization(q);
}

void testInferenceLayerNormSymInt32(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwd = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(symQ),
                       .propLossMath = arithmeticFromQuantization(bwd),
                       .outputQ = symQ,
                       .propLossQ = bwd,
                       .weightStorage = symQ,
                       .biasStorage = symQ};
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

/* Task 10 regression: initBufferOutput's SYM arm must honor the layer's declared
 * outputQ qMaxBits, not re-default to the int12 operand width (ODT_SYM_OPERAND_QMAXBITS).
 * Pre-fix: the returned tensor always carries qMaxBits=12 regardless of the
 * declared width. Post-fix: it carries the declared 8. */
void testInferenceOutputWireHonorsDeclaredQMaxBits(void) {
    quantization_t *q = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *outputQ8 = quantizationInitSymInt32WithBits(HALF_AWAY, 8);

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

    layer_t *linear = buildBorrowedLinearLayer(weights, bias, q);
    /* Override just the forward-wire storage config to a declared width of 8
     * (not the operand default 12) — exercises initBufferOutput's SYM arm. */
    linear->config->linear->outputQ = outputQ8;
    layer_t *model[] = {linear};

    tensor_t *output = inference(model, 1, input);

    uint8_t qMaxBits = ((symInt32QConfig_t *)output->quantization->qConfig)->qMaxBits;

    freeTensor(output);      /* inference() returns a heap tensor — caller owns. */
    freeLinearLayer(linear); /* ownsQuantizations=false: leaves q/outputQ8 untouched. */
    freeTensor(input);
    freeQuantization(q);
    freeQuantization(outputQ8);

    TEST_ASSERT_EQUAL_UINT8_MESSAGE(8, qMaxBits,
                                    "inference() output wire must carry the layer's declared "
                                    "outputQ qMaxBits, not the re-defaulted int12 operand width");
}

/* BFP epic PR2 Task 8 --------------------------------------------------------
 * The inference path has its OWN pair of wire allocators (initBufferOutput /
 * initBufferInput), separate from the training path's wire table. Both
 * need the BFP arm, and both derive the group geometry from the wire's own
 * element count (BFP PR2-Decision 5) rather than trusting the template. */

static tensor_t *buildFloatTensor2DInf(size_t d0, size_t d1, const float *values) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, d0 * d1);
    return t;
}

/*! initBufferOutput BFP arm: a Linear whose declared outputQ is a grouped BFP
 *  template produces a BFP inference wire, with numGroups DERIVED from the
 *  output's 4 elements / groupSize 2 = 2 -- the template's numGroups=5 is
 *  ignored. The exponents come out of the forward's OUT_WRITE epilogue, so at
 *  least one must have left the zero state (bias 127). */
void testInferenceBufferOutputCarriesBfpWire(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 5, 2);
    lq.outputQ = bfpWireQ;

    tensor_t *input = buildFloatTensor2DInf(1, 2, (float[]){1.f, 2.f});
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 4, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 2.f, 0.f},
                     (float[]){0.f, 0.f, 0.f, 0.f});
    layer_t *model[] = {linear};

    tensor_t *output = inference(model, 1, input);

    /* CAPTURE. */
    int capturedType = (int)output->quantization->type;
    size_t capturedNumGroups = 0;
    size_t capturedGroupSize = 0;
    uint8_t capturedMantissaBits = 0;
    bool anyExponentMoved = false;
    if (output->quantization->type == BFP) {
        bfpQConfig_t *qc = output->quantization->qConfig;
        capturedNumGroups = qc->numGroups;
        capturedGroupSize = qc->groupSize;
        capturedMantissaBits = qc->mantissaBits;
        for (size_t g = 0; g < qc->numGroups; g++) {
            if (qc->exponents[g] != 127) {
                anyExponentMoved = true;
            }
        }
    }

    /* FREE. */
    freeTensor(output);
    freeLinearLayer(linear);
    freeTensor(input);
    freeQuantization(bfpWireQ);
    freeQuantization(q);

    /* ASSERT. */
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, capturedType, "inference wire must carry the declared BFP");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, capturedNumGroups,
                                   "numGroups must be DERIVED (4 elements / groupSize 2), not the "
                                   "template's numGroups=5");
    TEST_ASSERT_EQUAL_UINT(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8(8, capturedMantissaBits);
    TEST_ASSERT_TRUE_MESSAGE(anyExponentMoved,
                             "the forward OUT_WRITE must derive the wire's exponents");
}

/*! initBufferOutput twin of the training-side normalization: a template
 *  groupSize EQUAL to the wire's element count (4 output elements, groupSize 4)
 *  derives numGroups == 1 -- per-tensor blocking, canonical {1,0}. Must
 *  allocate (the divisibility guard passed), not die on the {1,N} grammar. */
void testInferenceBufferOutputBfpGroupSizeEqualToWireNormalizesToPerTensor(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 5, 4);
    lq.outputQ = bfpWireQ;

    tensor_t *input = buildFloatTensor2DInf(1, 2, (float[]){1.f, 2.f});
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 4, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 2.f, 0.f},
                     (float[]){0.f, 0.f, 0.f, 0.f});
    layer_t *model[] = {linear};

    tensor_t *output = inference(model, 1, input);

    /* CAPTURE. */
    int capturedType = (int)output->quantization->type;
    size_t capturedNumGroups = 0;
    size_t capturedGroupSize = 99;
    bool anyExponentMoved = false;
    if (output->quantization->type == BFP) {
        bfpQConfig_t *qc = output->quantization->qConfig;
        capturedNumGroups = qc->numGroups;
        capturedGroupSize = qc->groupSize;
        for (size_t g = 0; g < qc->numGroups; g++) {
            if (qc->exponents[g] != 127) {
                anyExponentMoved = true;
            }
        }
    }

    /* FREE. */
    freeTensor(output);
    freeLinearLayer(linear);
    freeTensor(input);
    freeQuantization(bfpWireQ);
    freeQuantization(q);

    /* ASSERT. */
    TEST_ASSERT_EQUAL_INT_MESSAGE(BFP, capturedType, "inference wire must carry the declared BFP");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(1, capturedNumGroups,
                                   "groupSize == wire elements must derive ONE group");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(
        0, capturedGroupSize, "one whole-tensor group is per-tensor blocking -- canonical {1,0}");
    TEST_ASSERT_TRUE_MESSAGE(anyExponentMoved,
                             "the forward OUT_WRITE must derive the wire's exponents");
}

/*! initBufferInput BFP arm: a BFP-STORED input tensor must survive the entry
 *  buffer with its geometry AND its exponent VALUES intact -- otherwise the
 *  first layer reads the mantissas against a zero-state grid, i.e. every value
 *  silently rescaled by a power of two. Identity Linear, so the FLOAT32 output
 *  is exactly the dequantized input. */
void testInferenceBufferInputCarriesBfpExponents(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);

    /* BFP input: two groups of one, values 1.5 and 24.0 -- far enough apart
     * that a dropped exponent could not go unnoticed. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *input = initTensor(shape, quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 1), NULL);
    tensorFillFromFloatBuffer(input, (float[]){1.5f, 24.f}, 2);

    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(linear, (float[]){1.f, 0.f, 0.f, 1.f}, (float[]){0.f, 0.f});
    layer_t *model[] = {linear};

    tensor_t *output = inference(model, 1, input);

    /* CAPTURE. */
    float capturedOutput[2] = {((float *)output->data)[0], ((float *)output->data)[1]};

    /* FREE. */
    freeTensor(output);
    freeLinearLayer(linear);
    freeTensor(input);
    freeQuantization(q);

    /* ASSERT: 1.5 and 24.0 are both exactly representable at 8 mantissa bits. */
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(1.5f, capturedOutput[0],
                                    "BFP input buffer must carry group 0's exponent");
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(24.f, capturedOutput[1],
                                    "BFP input buffer must carry group 1's exponent");
}

/*! initBufferOutput's divisibility fail-fast (the inference-path twin of
 *  testBfpActWireGroupSizeMismatchDiesNamingTheWire). Discriminating fixture: a
 *  5-element output wire with groupSize 2, so floor division yields the
 *  CONSTRUCTIBLE shape {2, 2} -- initBfpQConfigGrouped's own guard does not
 *  fire, and without this check the packer would index exponents[2] past a
 *  2-entry array. (A groupSize that floors to {1, n} would be rejected by
 *  initBfpQConfigGrouped anyway and would make this test vacuous.) */
void testInitBufferOutputBfpGroupSizeMismatchDies(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    quantization_t *bfpWireQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    lq.outputQ = bfpWireQ;

    tensor_t *input = buildFloatTensor2DInf(1, 2, (float[]){1.f, 2.f});
    layer_t *linear =
        linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 5, .bias = BIAS_TRUE}, &lq);
    layer_t *model[] = {linear};

    ASSERT_EXITS_WITH_FAILURE(freeTensor(inference(model, 1, input)));

    freeLinearLayer(linear);
    freeTensor(input);
    freeQuantization(bfpWireQ);
    freeQuantization(q);
}

/* #152 PR1/PR3b -------------------------------------------------------------
 * maxPool1dLayerInit sizes the argmax [1, C, Lout], and both inference entry
 * points hand that config-owned buffer to the forward as auxOut. PR 1 made a
 * batch-2 input die in the argmax shape guard; PR 3b grows the argmax on
 * demand instead. Each test runs the SAME model at batch 1 first,
 * then at batch 2, so the batch-2 result comes from a grown argmax. */

static tensor_t *buildFloatTensor3DInf(size_t d0, size_t d1, size_t d2, const float *values) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)values, d0 * d1 * d2);
    return t;
}

static layer_t *buildFactoryMaxPoolK2C2L4(layerQuant_t *lq) {
    return maxPool1dLayerInit(
        &(maxPool1dInit_t){.kernelSize = 2, .stride = 2, .inputChannels = 2, .inputLength = 4}, lq);
}

static const float kMaxPoolBatch1[8] = {0.5f, -1.f, 2.f, 0.25f, -0.75f, 1.5f, 0.f, -2.f};
static const float kMaxPoolBatch2[16] = {0.5f, -1.f, 2.f,   0.25f,  -0.75f, 1.5f, 0.f,   -2.f,
                                         3.f,  1.f,  -0.5f, -0.25f, 0.75f,  0.5f, -1.5f, 1.25f};

void testInferenceFactoryMaxPoolGrowsToBatch2(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *pool = buildFactoryMaxPoolK2C2L4(&lq);
    layer_t *model[] = {pool};
    tensor_t *batch1 = buildFloatTensor3DInf(1, 2, 4, kMaxPoolBatch1);
    tensor_t *batch2 = buildFloatTensor3DInf(2, 2, 4, kMaxPoolBatch2);

    tensor_t *output = inference(model, 1, batch1);

    /* CAPTURE. */
    size_t capturedDims[3] = {output->shape->dimensions[0], output->shape->dimensions[1],
                              output->shape->dimensions[2]};
    float capturedValues[4];
    for (size_t i = 0; i < 4; i++) {
        capturedValues[i] = ((float *)output->data)[i];
    }
    freeTensor(output);

    /* #152 PR3b: the forward grows the factory's [1, C, Lout] argmax to
     * batch 2 instead of dying. */
    tensor_t *output2 = inference(model, 1, batch2);
    size_t capturedBatch2Dims[3] = {output2->shape->dimensions[0], output2->shape->dimensions[1],
                                    output2->shape->dimensions[2]};
    float capturedBatch2Values[8];
    for (size_t i = 0; i < 8; i++) {
        capturedBatch2Values[i] = ((float *)output2->data)[i];
    }
    freeTensor(output2);

    /* FREE. */
    freeTensor(batch2);
    freeTensor(batch1);
    freeMaxPool1dLayer(pool);
    freeQuantization(q);

    /* ASSERT: K=2/S=2 windows {0.5,-1} {2,0.25} | {-0.75,1.5} {0,-2}. */
    float expectedValues[4] = {0.5f, 2.f, 1.5f, 0.f};
    TEST_ASSERT_EQUAL_size_t(1, capturedDims[0]);
    TEST_ASSERT_EQUAL_size_t(2, capturedDims[1]);
    TEST_ASSERT_EQUAL_size_t(2, capturedDims[2]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedValues, capturedValues, 4);
    /* Row 1 windows {3,1} {-0.5,-0.25} | {0.75,0.5} {-1.5,1.25}. */
    float expectedBatch2Values[8] = {0.5f, 2.f, 1.5f, 0.f, 3.f, -0.25f, 0.75f, 1.25f};
    TEST_ASSERT_EQUAL_size_t(2, capturedBatch2Dims[0]);
    TEST_ASSERT_EQUAL_size_t(2, capturedBatch2Dims[1]);
    TEST_ASSERT_EQUAL_size_t(2, capturedBatch2Dims[2]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedBatch2Values, capturedBatch2Values, 8);
}

void testInferenceWithLossFactoryMaxPoolGrowsToBatch2(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *pool = buildFactoryMaxPoolK2C2L4(&lq);
    layer_t *model[] = {pool};
    tensor_t *batch1 = buildFloatTensor3DInf(1, 2, 4, kMaxPoolBatch1);
    tensor_t *batch2 = buildFloatTensor3DInf(2, 2, 4, kMaxPoolBatch2);
    tensor_t *label1 = buildFloatTensor3DInf(1, 2, 2, (float[]){0.f, 0.f, 0.f, 0.f});
    tensor_t *label2 =
        buildFloatTensor3DInf(2, 2, 2, (float[]){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    inferenceStats_t *stats =
        inferenceWithLoss(model, 1, batch1, label1, MSE, REDUCTION_MEAN, NULL);

    /* CAPTURE. */
    float capturedLoss = stats->loss;
    freeInferenceStats(stats);

    /* #152 PR3b: batch 2 grows the argmax and scores both rows. */
    inferenceStats_t *stats2 =
        inferenceWithLoss(model, 1, batch2, label2, MSE, REDUCTION_MEAN, NULL);
    float capturedLoss2 = stats2->loss;
    freeInferenceStats(stats2);

    /* FREE. */
    freeTensor(label2);
    freeTensor(label1);
    freeTensor(batch2);
    freeTensor(batch1);
    freeMaxPool1dLayer(pool);
    freeQuantization(q);

    /* ASSERT: pooled {0.5, 2, 1.5, 0} vs zeros -> (0.25 + 4 + 2.25 + 0) / 4. */
    TEST_ASSERT_EQUAL_FLOAT(1.625f, capturedLoss);
    /* Batch 2 adds row 1 {3, -0.25, 0.75, 1.25}: (6.5 + 11.1875) / 8. */
    TEST_ASSERT_EQUAL_FLOAT(2.2109375f, capturedLoss2);
}

/* #152 PR3a: inferenceBatched is a batch_t consumer -- its samples arrive in
 * their natural shape and it adds the batch axis itself, unconditionally. A
 * Relu-only model is rank-agnostic (its output copies the input shape), so the
 * output rank shows exactly what inference() was handed. item1 starts with a 1
 * on purpose (data-shape.md: nothing auto-detects an existing batch axis) -- it must
 * still gain the axis. */
static tensor_t *buildFloatTensor1DInf(size_t n, const float *values) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, values, n);
    return t;
}

void testInferenceBatchedAddsBatchAxisToNaturalSamples(void) {
    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *relu = reluLayerInit(&lq);
    layer_t *model[] = {relu};

    tensor_t *item0 = buildFloatTensor1DInf(3, (float[]){-1.f, 2.f, -3.f});   /* natural [3] */
    tensor_t *item1 = buildFloatTensor2DInf(1, 3, (float[]){4.f, -5.f, 6.f}); /* natural [1, 3] */
    sample_t *s0 = reserveMemory(sizeof(sample_t));
    s0->item = item0;
    s0->label = item0;
    sample_t *s1 = reserveMemory(sizeof(sample_t));
    s1->item = item1;
    s1->label = item1;
    sample_t *samples[] = {s0, s1};
    batch_t batch = {.samples = samples, .size = 2};

    tensor_t **outputs = inferenceBatched(model, 1, &batch);

    size_t rank[2];
    size_t dims[2][3] = {{0}};
    float values[2][3];
    for (size_t i = 0; i < 2; i++) {
        rank[i] = outputs[i]->shape->numberOfDimensions;
        for (size_t d = 0; d < rank[i] && d < 3; d++) {
            dims[i][d] = outputs[i]->shape->dimensions[d];
        }
        for (size_t j = 0; j < 3; j++) {
            values[i][j] = ((float *)outputs[i]->data)[j];
        }
    }

    for (size_t i = 0; i < 2; i++) {
        freeTensor(outputs[i]);
    }
    freeReservedMemory(outputs);
    /* inferenceBatched does not consume the samples (no freeSample inside); the
     * sample_t shells are plain reserveMemory blocks. */
    freeReservedMemory(s1);
    freeReservedMemory(s0);
    freeTensor(item1);
    freeTensor(item0);
    freeReluLayer(relu);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_size_t_MESSAGE(2, rank[0], "output 0 must be [1, 3]");
    TEST_ASSERT_EQUAL_size_t(1, dims[0][0]);
    TEST_ASSERT_EQUAL_size_t(3, dims[0][1]);
    TEST_ASSERT_EQUAL_size_t_MESSAGE(3, rank[1], "output 1 must be [1, 1, 3] (no auto-detection)");
    TEST_ASSERT_EQUAL_size_t(1, dims[1][0]);
    TEST_ASSERT_EQUAL_size_t(1, dims[1][1]);
    TEST_ASSERT_EQUAL_size_t(3, dims[1][2]);
    float expected[2][3] = {{0.f, 2.f, 0.f}, {4.f, 0.f, 6.f}};
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected[i], values[i], 3);
    }
}

void setUp() {}
void tearDown() {}

/* ---- the call struct on the eval level (#4 PR2, remat D19) --------------- */

typedef struct evalFixture {
    quantization_t *q;
    layer_t *model[2];
    tensor_t *input;
    tensor_t *label;
} evalFixture_t;

/* testInferenceWithLossLinearReluFloat's model and sample: loss 2665. */
static void buildEvalFixture(evalFixture_t *f) {
    f->q = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, f->q);
    f->model[0] =
        linearLayerInit(&(linearInit_t){.inFeatures = 3, .outFeatures = 2, .bias = BIAS_TRUE}, &lq);
    layerLoadWeights(f->model[0], (float[]){-1.f, 2.f, -3.f, 4.f, 5.f, 6.f}, (float[]){-1.f, 3.f});
    f->model[1] = reluLayerInit(&lq);
    f->input = buildFloatTensor2DInf(1, 3, (float[]){0.f, 1.f, 2.f});
    f->label = buildFloatTensor2DInf(1, 2, (float[]){59.f, -23.f});
}

static void freeEvalFixture(evalFixture_t *f) {
    freeTensor(f->label);
    freeTensor(f->input);
    freeReluLayer(f->model[1]);
    freeLinearLayer(f->model[0]);
    freeQuantization(f->q);
}

/* Evaluation on a caller's scheduler (remat D19): output and loss equal the
 * NULL path's bit for bit, on either row, on two calls of one instance. The
 * death-test wrapper keeps a wrong exit from ending the binary. */
void testInferenceWithLossRunsOnACallersScheduler(void) {
    evalFixture_t f;
    buildEvalFixture(&f);
    inferenceStats_t *ref =
        inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, NULL);
    rematScheduler_t heap;
    rematScheduler_t arena;
    TEST_ASSERT_TRUE(rematHeapInit(&heap, f.model, 2, defaultLossConfig(MSE), f.input, NULL));
    TEST_ASSERT_TRUE(rematArenaInit(&arena, f.model, 2, defaultLossConfig(MSE), f.input, NULL));
    rematScheduler_t *rows[2] = {&heap, &arena};
    for (size_t k = 0; k < 2u; k++) {
        const trainingCall_t call = {.remat = rows[k]};
        ASSERT_EXITS_WITH_OUTPUT(0, "",
                                 freeInferenceStats(inferenceWithLoss(f.model, 2, f.input, f.label,
                                                                      MSE, REDUCTION_MEAN, &call)));
        for (int rep = 0; rep < 2; rep++) {
            inferenceStats_t *got =
                inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, &call);
            TEST_ASSERT_EQUAL_MEMORY(&ref->loss, &got->loss, sizeof(float));
            TEST_ASSERT_EQUAL_size_t(2, got->output->shape->numberOfDimensions);
            TEST_ASSERT_EQUAL_size_t(2, got->output->shape->dimensions[1]);
            TEST_ASSERT_EQUAL_MEMORY(ref->output->data, got->output->data, 2 * sizeof(float));
            freeInferenceStats(got);
            /* The call ran on the scheduler: its accounting saw the eval peak,
             * ACT 1 + ACT 2 = two [1, 2] float wires = 16 B. */
            rematReport_t report;
            rematSchedulerReport(rows[k], &report);
            TEST_ASSERT_EQUAL_size_t(16, report.observedPeakLiveBytes);
        }
    }
    rematSchedulerDeinit(&arena);
    rematSchedulerDeinit(&heap);
    freeInferenceStats(ref);
    freeEvalFixture(&f);
}

/* A call without a scheduler is not an error: today's body, today's values.
 * The death-test wrapper keeps a wrong exit from ending the binary. */
void testInferenceWithLossTakesACallWithoutAScheduler(void) {
    evalFixture_t f;
    buildEvalFixture(&f);
    ASSERT_EXITS_WITH_OUTPUT(
        0, "",
        freeInferenceStats(inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN,
                                             &(trainingCall_t){0})));
    inferenceStats_t *viaNull =
        inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, NULL);
    inferenceStats_t *viaCall =
        inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, &(trainingCall_t){0});
    float loss[2] = {viaNull->loss, viaCall->loss};
    float out[2][2];
    memcpy(out[0], viaNull->output->data, sizeof out[0]);
    memcpy(out[1], viaCall->output->data, sizeof out[1]);
    freeInferenceStats(viaCall);
    freeInferenceStats(viaNull);
    freeEvalFixture(&f);
    TEST_ASSERT_EQUAL_FLOAT(2665.f, loss[0]);
    TEST_ASSERT_EQUAL_MEMORY(&loss[0], &loss[1], sizeof loss[0]);
    TEST_ASSERT_EQUAL_MEMORY(out[0], out[1], sizeof out[0]);
}

/* ---- the layer keys inferenceWithLoss hands out (#4) ---- */

/* Expected words from the Python mirror (LAYER spelled 3), repository root:
 *   uv run python -c "from examples._shared.keyed_rng import (fold,
 *   key_from_seed as kfs); c = kfs(0xd21f); print([[hex(w) for w in fold(c, 3, i)]
 *   for i in (0, 1, 2)])"
 * Each check also compares with the explicit rngFold chain. */
#define EVAL_CALL_KEY_SEED 0xd21fu
#define EVAL_EXEC_RECORDS 12

typedef struct {
    bool hasExec;
    rngKey_t key;
} execRecord_t;

static execRecord_t g_execRecords[EVAL_EXEC_RECORDS];
static size_t g_execRecordCount;
static layerFunctions_t g_savedLinear;
static layerFunctions_t g_savedRelu;
static lossFunctions_t g_savedMse;

static void recordExec(const layerExec_t *exec) {
    if (g_execRecordCount < EVAL_EXEC_RECORDS) {
        g_execRecords[g_execRecordCount].hasExec = exec != NULL;
        if (exec != NULL) {
            g_execRecords[g_execRecordCount].key = exec->key;
        }
    }
    g_execRecordCount++;
}

static void spyForward(layer_t *layer, tensor_t *in, tensor_t *out, const layerExec_t *exec) {
    recordExec(exec);
    (layer->type == LINEAR ? g_savedLinear : g_savedRelu).forward(layer, in, out, exec);
}

static float spyLossForward(tensor_t *out, tensor_t *label, reduction_t reduction,
                            const layerExec_t *exec) {
    recordExec(exec);
    return g_savedMse.forward(out, label, reduction, exec);
}

static void installExecSpies(void) {
    g_savedLinear = layerFunctions[LINEAR];
    g_savedRelu = layerFunctions[RELU];
    g_savedMse = lossFunctions[MSE];
    layerFunctions[LINEAR].forward = spyForward;
    layerFunctions[RELU].forward = spyForward;
    lossFunctions[MSE].forward = spyLossForward;
    g_execRecordCount = 0;
}

static void restoreExecSpies(void) {
    layerFunctions[LINEAR] = g_savedLinear;
    layerFunctions[RELU] = g_savedRelu;
    lossFunctions[MSE] = g_savedMse;
}

static void assertThreeLayerKeys(rngKey_t callKey) {
    const uint32_t python[3][2] = {
        {0x6842c650u, 0xaf1109a3u}, {0x47477cedu, 0xab1bacc3u}, {0x626020c6u, 0x288f25e1u}};
    TEST_ASSERT_EQUAL_size_t(3, g_execRecordCount);
    for (uint32_t layer = 0; layer < 3u; layer++) {
        rngKey_t expected = rngFold(callKey, RNG_TAG_LAYER, layer);
        TEST_ASSERT_TRUE(g_execRecords[layer].hasExec);
        TEST_ASSERT_EQUAL_HEX32(expected.w[0], g_execRecords[layer].key.w[0]);
        TEST_ASSERT_EQUAL_HEX32(expected.w[1], g_execRecords[layer].key.w[1]);
        TEST_ASSERT_EQUAL_HEX32(python[layer][0], g_execRecords[layer].key.w[0]);
        TEST_ASSERT_EQUAL_HEX32(python[layer][1], g_execRecords[layer].key.w[1]);
    }
}

/* A keyed call hands layer i the call key folded with (LAYER, i) and the loss
 * that of index n, without a scheduler and on a caller's scheduler alike. */
void testInferenceWithLossHandsEachLayerItsLayerKey(void) {
    evalFixture_t f;
    buildEvalFixture(&f);
    const rngKey_t callKey = rngKeyFromSeed(EVAL_CALL_KEY_SEED);
    rematScheduler_t heap;
    TEST_ASSERT_TRUE(rematHeapInit(&heap, f.model, 2, defaultLossConfig(MSE), f.input, NULL));

    installExecSpies();
    freeInferenceStats(inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN,
                                         &(trainingCall_t){.key = &callKey}));
    restoreExecSpies();
    assertThreeLayerKeys(callKey);

    installExecSpies();
    freeInferenceStats(inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN,
                                         &(trainingCall_t){.remat = &heap, .key = &callKey}));
    restoreExecSpies();
    assertThreeLayerKeys(callKey);

    rematSchedulerDeinit(&heap);
    freeEvalFixture(&f);
}

/* Without a key no layer gets a context, and inference() never hands one. */
void testInferenceWithoutAKeyHandsNoLayerContext(void) {
    evalFixture_t f;
    buildEvalFixture(&f);
    installExecSpies();

    freeInferenceStats(inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, NULL));
    freeInferenceStats(
        inferenceWithLoss(f.model, 2, f.input, f.label, MSE, REDUCTION_MEAN, &(trainingCall_t){0}));
    freeTensor(inference(f.model, 2, f.input));

    restoreExecSpies();
    freeEvalFixture(&f);
    TEST_ASSERT_EQUAL_size_t(8, g_execRecordCount);
    for (size_t r = 0; r < 8; r++) {
        TEST_ASSERT_FALSE(g_execRecords[r].hasExec);
    }
}

/* inference() does not switch layer modes: a Dropout layer left in training
 * mode draws its mask, here with no key. */
static void inferThroughADropoutLeftInTrainingMode(void) {
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = 4;
    size_t *order = reserveMemory(sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *mask = initTensor(shape, quantizationInitBool(), NULL);
    quantization_t *q = quantizationInitFloat();
    layer_t *dropout = dropoutLayerInit(0.5f, mask, q, q);
    dropout->config->dropout->training = true;
    layer_t *model[1] = {dropout};
    (void)inference(model, 1, buildFloatTensor2DInf(1, 4, (float[]){1.f, 2.f, 3.f, 4.f}));
}

/* No key, no randomness: inference() is key-less, so a forward that draws
 * stops at its first draw. */
void testInferenceOnADrawingForwardExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "bernoulliFillMaskReference: Bernoulli draw without a stream",
                             inferThroughADropoutLeftInTrainingMode());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInferenceLinearReluFloat);
    RUN_TEST(testInferenceLinearReluSymInt32);
    RUN_TEST(testInferenceLayerNormSymInt32);
    RUN_TEST(testInferenceOutputWireHonorsDeclaredQMaxBits);

    RUN_TEST(testInferenceWithLossLinearReluFloat);
    RUN_TEST(testInferenceWithLossRejectsRank1LabelAgainstBatchedOutput);

    RUN_TEST(testInferenceBufferOutputCarriesBfpWire);
    RUN_TEST(testInferenceBufferInputCarriesBfpExponents);
    RUN_TEST(testInitBufferOutputBfpGroupSizeMismatchDies);
    RUN_TEST(testInferenceBufferOutputBfpGroupSizeEqualToWireNormalizesToPerTensor);

    RUN_TEST(testInferenceFactoryMaxPoolGrowsToBatch2);
    RUN_TEST(testInferenceWithLossFactoryMaxPoolGrowsToBatch2);

    RUN_TEST(testInferenceBatchedAddsBatchAxisToNaturalSamples);
    RUN_TEST(testInferenceWithLossRunsOnACallersScheduler);
    RUN_TEST(testInferenceWithLossTakesACallWithoutAScheduler);
    RUN_TEST(testInferenceWithLossHandsEachLayerItsLayerKey);
    RUN_TEST(testInferenceWithoutAKeyHandsNoLayerContext);
    RUN_TEST(testInferenceOnADrawingForwardExits);
    return UNITY_END();
}
