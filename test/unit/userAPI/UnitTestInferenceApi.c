#include <stdbool.h>
#include <stdint.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "LayerWeightsApi.h"
#include "Linear.h"
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
    convertTensor(outputSymInt32, outputFloat);

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
        inferenceWithLoss(model, 2, input, label0, MSE, REDUCTION_MEAN);

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

/* Hardening regression: reserveInferenceStats sized stats->output from the
 * LABEL's shape, but inferenceWithLoss then copies the PRODUCED model output
 * into it. A classifier's one-hot label is rank-1 [C] while the model output
 * carries an explicit batch dim [1, C] (the HAR/validation constellation) --
 * the label-sized shape copy overflowed past the dimensions array whenever
 * output rank > label rank (ASan-verified, see root-cause report). */
void testInferenceWithLossOutputShapeMatchesProducedOutputNotLabel(void) {
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

    /* Run inferenceWithLoss. inferenceStats owns its `output` tensor; its
     * matching free is freeInferenceStats. */
    inferenceStats_t *inferenceStats =
        inferenceWithLoss(model, modelSize, input, label, MSE, REDUCTION_MEAN);
    /* Plain inference() is the known-correct idiom (getTensorLike + convertTensor);
     * inferenceStats->output must match it element-for-element. */
    tensor_t *ref = inference(model, modelSize, input);

    /* CAPTURE (before free -- pre-fix, stats->output's shape is rank 1, so
     * dimensions[1] would be an out-of-bounds read; guard it). */
    size_t capturedRank = inferenceStats->output->shape->numberOfDimensions;
    size_t capturedDim0 = inferenceStats->output->shape->dimensions[0];
    size_t capturedDim1 =
        (capturedRank >= 2) ? inferenceStats->output->shape->dimensions[1] : SIZE_MAX;
    float capturedOutput[3];
    float capturedRef[3];
    for (size_t i = 0; i < 3; i++) {
        capturedOutput[i] = ((float *)inferenceStats->output->data)[i];
        capturedRef[i] = ((float *)ref->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(ref);
    freeInferenceStats(inferenceStats);
    freeReluLayer(relu);
    freeLinearLayer(linear);
    freeTensor(label);
    freeTensor(input);
    freeQuantization(q);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(2, capturedRank);
    TEST_ASSERT_EQUAL_size_t(1, capturedDim0);
    TEST_ASSERT_EQUAL_size_t(3, capturedDim1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedRef, capturedOutput, 3);
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
 * initBufferInput), separate from the training path's initLayerOutputs. Both
 * need the BFP arm, and both derive the group geometry from the wire's own
 * element count (plan Decision 5) rather than trusting the template. */

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
 *  testInitLayerOutputsBfpGroupSizeMismatchDies). Discriminating fixture: a
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

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testInferenceLinearReluFloat);
    RUN_TEST(testInferenceLinearReluSymInt32);
    RUN_TEST(testInferenceLayerNormSymInt32);
    RUN_TEST(testInferenceOutputWireHonorsDeclaredQMaxBits);

    RUN_TEST(testInferenceWithLossLinearReluFloat);
    RUN_TEST(testInferenceWithLossOutputShapeMatchesProducedOutputNotLabel);

    RUN_TEST(testInferenceBufferOutputCarriesBfpWire);
    RUN_TEST(testInferenceBufferInputCarriesBfpExponents);
    RUN_TEST(testInitBufferOutputBfpGroupSizeMismatchDies);
    RUN_TEST(testInferenceBufferOutputBfpGroupSizeEqualToWireNormalizesToPerTensor);
    return UNITY_END();
}
