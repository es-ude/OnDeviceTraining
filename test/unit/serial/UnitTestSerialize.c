#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "Deserialize.h"
#include "Dropout.h"
#include "DropoutApi.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "MaxPool1d.h"
#include "Pool1dApi.h"
#include "QuantLayerApi.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Serialize.h"
#include "Softmax.h"
#include "SoftmaxApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* SERIALIZE_TEST_FILE_PATH is injected by the CMake target_compile_definitions
 * in test/unit/serial/CMakeLists.txt as an absolute path so the test does not
 * depend on the working directory (which differs between host runs and
 * Docker LSan runs). */
#define FILE_PATH SERIALIZE_TEST_FILE_PATH

void setUp(void) {}
void tearDown(void) {}

/*! LINEAR round trip. `lq` is reused verbatim to construct both the serial
 *  and the deserial mirror layer (both Owning, so each gets its own
 *  parameter tensors + outputQ/propLossQ copies) — the same pattern the
 *  design spec's "designated override" example assumes. `biasGradMath` is
 *  bumped away from the uniform baseline so the four arithmetic fields are
 *  NOT all identical: a field-order bug in serializeLayer/deserializeLayer
 *  (e.g. weightGradMath and biasGradMath swapped) corrupts the file at that
 *  byte offset, which the read-back overwrite on the deserial side then
 *  exposes as a mismatch against the untouched serial-side value. */
static void testRoundTripLinear(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.biasGradMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    linearInit_t init = {.inFeatures = 4, .outFeatures = 3};

    layer_t *serialLayer = linearLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = linearLayerInitOwning(&init, &lq);

    linearConfig_t *serialCfg = serialLayer->config->linear;

    /* Grads are zero-initialized at construction; seed non-zero values on
     * the serial side only so a round trip that actually moves bytes is
     * distinguishable from one that leaves the (still-zero) deserial mirror
     * untouched. */
    size_t numberOfWeights = calcNumberOfElementsByTensor(serialCfg->weights->param);
    size_t numberOfBiases = calcNumberOfElementsByTensor(serialCfg->bias->param);

    float *weightGradSeed = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        weightGradSeed[i] = (float)i + 0.5f;
    }
    tensorFillFromFloatBuffer(serialCfg->weights->grad, weightGradSeed, numberOfWeights);

    float *biasGradSeed = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        biasGradSeed[i] = (float)i - 1.5f;
    }
    tensorFillFromFloatBuffer(serialCfg->bias->grad, biasGradSeed, numberOfBiases);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    linearConfig_t *deserialCfg = deserialLayer->config->linear;

    /* CAPTURE every assertion value before any free. */
    float *capturedSerialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedSerialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedSerialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedSerialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedSerialW[i] = ((float *)serialCfg->weights->param->data)[i];
        capturedDeserialW[i] = ((float *)deserialCfg->weights->param->data)[i];
        capturedSerialWGrad[i] = ((float *)serialCfg->weights->grad->data)[i];
        capturedDeserialWGrad[i] = ((float *)deserialCfg->weights->grad->data)[i];
    }
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedSerialB[i] = ((float *)serialCfg->bias->param->data)[i];
        capturedDeserialB[i] = ((float *)deserialCfg->bias->param->data)[i];
        capturedSerialBGrad[i] = ((float *)serialCfg->bias->grad->data)[i];
        capturedDeserialBGrad[i] = ((float *)deserialCfg->bias->grad->data)[i];
    }

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialWeightGrad = serialCfg->weightGradMath;
    arithmetic_t capturedDeserialWeightGrad = deserialCfg->weightGradMath;
    arithmetic_t capturedSerialBiasGrad = serialCfg->biasGradMath;
    arithmetic_t capturedDeserialBiasGrad = deserialCfg->biasGradMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;

    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    /* FREE in reverse-init order. */
    freeReservedMemory(biasGradSeed);
    freeReservedMemory(weightGradSeed);
    freeLinearLayer(deserialLayer);
    freeLinearLayer(serialLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW, capturedDeserialW, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialWGrad, capturedDeserialWGrad, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB, capturedDeserialB, numberOfBiases);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBGrad, capturedDeserialBGrad, numberOfBiases);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.type, capturedDeserialWeightGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.roundingMode,
                      capturedDeserialWeightGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.type, capturedDeserialBiasGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.roundingMode, capturedDeserialBiasGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);

    /* Sanity: the override actually makes the four fields non-uniform. */
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, capturedSerialForward.type);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialBiasGrad.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);

    freeReservedMemory(capturedSerialW);
    freeReservedMemory(capturedDeserialW);
    freeReservedMemory(capturedSerialWGrad);
    freeReservedMemory(capturedDeserialWGrad);
    freeReservedMemory(capturedSerialB);
    freeReservedMemory(capturedDeserialB);
    freeReservedMemory(capturedSerialBGrad);
    freeReservedMemory(capturedDeserialBGrad);
}

/*! RELU round trip. `outputQ`/`propLossQ` are pure quantization_t configs
 *  (Relu owns no tensor keyed on them at construction time), so this is a
 *  safe place to exercise the two NEW quantization-payload fixes directly:
 *  SYM now round-trips `roundingMode` (previously dropped), and BOOL now
 *  round-trips as a type byte only (previously `exit(1)`). */
static void testRoundTripRelu(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(4, SR_HALF_AWAY);
    quantization_t *boolPropLossQ = quantizationInitBool();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};
    lq.outputQ = symOutputQ;
    lq.propLossQ = boolPropLossQ;

    layer_t *serialLayer = reluLayerInitOwning(&lq);
    layer_t *deserialLayer = reluLayerInitOwning(&lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    reluConfig_t *serialCfg = serialLayer->config->relu;
    reluConfig_t *deserialCfg = deserialLayer->config->relu;

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;

    qtype_t capturedSerialOutputQType = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQType = deserialCfg->outputQ->type;
    symQConfig_t *serialOutputCfg = serialCfg->outputQ->qConfig;
    symQConfig_t *deserialOutputCfg = deserialCfg->outputQ->qConfig;
    float capturedSerialOutputScale = serialOutputCfg->scale;
    float capturedDeserialOutputScale = deserialOutputCfg->scale;
    uint8_t capturedSerialOutputQBits = serialOutputCfg->qBits;
    uint8_t capturedDeserialOutputQBits = deserialOutputCfg->qBits;
    roundingMode_t capturedSerialOutputRM = serialOutputCfg->roundingMode;
    roundingMode_t capturedDeserialOutputRM = deserialOutputCfg->roundingMode;

    qtype_t capturedSerialPropLossQType = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQType = deserialCfg->propLossQ->type;

    freeReluLayer(deserialLayer);
    freeReluLayer(serialLayer);
    freeQuantization(boolPropLossQ);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    /* Sanity: the override applied. */
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialPropLoss.type);

    TEST_ASSERT_EQUAL(SYM, capturedSerialOutputQType);
    TEST_ASSERT_EQUAL(capturedSerialOutputQType, capturedDeserialOutputQType);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialOutputScale, capturedDeserialOutputScale);
    TEST_ASSERT_EQUAL(capturedSerialOutputQBits, capturedDeserialOutputQBits);
    TEST_ASSERT_EQUAL(capturedSerialOutputRM, capturedDeserialOutputRM);
    /* Sanity: not the SYM default (HALF_AWAY) — the roundingMode fix is live. */
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, capturedSerialOutputRM);

    TEST_ASSERT_EQUAL(BOOL, capturedSerialPropLossQType);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQType, capturedDeserialPropLossQType);
}

/*! SOFTMAX round trip; outputQ carries a SYM_INT32 storage config (scale +
 *  roundingMode + qMaxBits), exercising that payload's field order once more
 *  under the new u8-tagged quantization record. */
static void testRoundTripSoftmax(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symIntOutputQ = quantizationInitSymInt32(SR_HALF_AWAY);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.forwardMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};
    lq.outputQ = symIntOutputQ;

    layer_t *serialLayer = softmaxLayerInitOwning(&lq);
    layer_t *deserialLayer = softmaxLayerInitOwning(&lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    softmaxConfig_t *serialCfg = serialLayer->config->softmax;
    softmaxConfig_t *deserialCfg = deserialLayer->config->softmax;

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;

    qtype_t capturedSerialOutputQType = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQType = deserialCfg->outputQ->type;
    symInt32QConfig_t *serialOutputCfg = serialCfg->outputQ->qConfig;
    symInt32QConfig_t *deserialOutputCfg = deserialCfg->outputQ->qConfig;
    float capturedSerialOutputScale = serialOutputCfg->scale;
    float capturedDeserialOutputScale = deserialOutputCfg->scale;
    roundingMode_t capturedSerialOutputRM = serialOutputCfg->roundingMode;
    roundingMode_t capturedDeserialOutputRM = deserialOutputCfg->roundingMode;
    uint8_t capturedSerialOutputQMaxBits = serialOutputCfg->qMaxBits;
    uint8_t capturedDeserialOutputQMaxBits = deserialOutputCfg->qMaxBits;

    qtype_t capturedSerialPropLossQType = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQType = deserialCfg->propLossQ->type;

    freeSoftmaxLayer(deserialLayer);
    freeSoftmaxLayer(serialLayer);
    freeQuantization(symIntOutputQ);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    /* Sanity: the override applied. */
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialForward.type);

    TEST_ASSERT_EQUAL(SYM_INT32, capturedSerialOutputQType);
    TEST_ASSERT_EQUAL(capturedSerialOutputQType, capturedDeserialOutputQType);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialOutputScale, capturedDeserialOutputScale);
    TEST_ASSERT_EQUAL(capturedSerialOutputRM, capturedDeserialOutputRM);
    TEST_ASSERT_EQUAL(capturedSerialOutputQMaxBits, capturedDeserialOutputQMaxBits);

    TEST_ASSERT_EQUAL(capturedSerialPropLossQType, capturedDeserialPropLossQType);
}

/*! FLATTEN carries no payload (tag only). Embedding it before a Relu layer
 *  proves the zero-byte record does not shift the next record's alignment. */
static void testRoundTripFlatten(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.forwardMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    layer_t *serialFlatten = flattenLayerInit();
    layer_t *serialRelu = reluLayerInitOwning(&lq);
    layer_t *serialModel[] = {serialFlatten, serialRelu};

    layer_t *deserialFlatten = flattenLayerInit();
    layer_t *deserialRelu = reluLayerInitOwning(&lq);
    layer_t *deserialModel[] = {deserialFlatten, deserialRelu};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 2, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 2, f);
    fclose(f);

    reluConfig_t *serialCfg = serialRelu->config->relu;
    reluConfig_t *deserialCfg = deserialRelu->config->relu;

    layerType_t capturedSerialFlattenType = serialFlatten->type;
    layerType_t capturedDeserialFlattenType = deserialFlatten->type;
    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQType = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQType = deserialCfg->outputQ->type;

    freeReluLayer(deserialRelu);
    freeFlattenLayer(deserialFlatten);
    freeReluLayer(serialRelu);
    freeFlattenLayer(serialFlatten);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(FLATTEN, capturedSerialFlattenType);
    TEST_ASSERT_EQUAL(FLATTEN, capturedDeserialFlattenType);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    /* Sanity: the override applied — proves Flatten's zero-byte record did
     * not desync the following Relu record. */
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialOutputQType, capturedDeserialOutputQType);
}

/*! CONV1D round trip. Non-trivial geometry (kernel size 3, stride 2, EXPLICIT
 *  padding=1, groups=1) exercises every kernel_t field + the groups scalar;
 *  a second, no-bias layer in the same 2-layer model exercises the
 *  bias-flag=0 path (weights round-trip correctly right after a record whose
 *  own bias sub-record was skipped) without desyncing. `biasGradMath` is
 *  bumped to SYM_INT32 so the four arithmetic fields are not all identical
 *  (field-order-swap detector, same pattern as testRoundTripLinear). */
static void testRoundTripConv1d(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.biasGradMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    conv1dInit_t initBias = {
        .inChannels = 4,
        .outChannels = 6,
        .kernelSize = 3,
        .stride = 2,
        .padding = EXPLICIT,
        .paddingAmount = 1,
        .groups = 1,
        .bias = BIAS_TRUE,
    };
    conv1dInit_t initNoBias = initBias;
    initNoBias.bias = BIAS_FALSE;

    layer_t *serialBias = conv1dLayerInitOwning(&initBias, &lq);
    layer_t *deserialBias = conv1dLayerInitOwning(&initBias, &lq);
    layer_t *serialNoBias = conv1dLayerInitOwning(&initNoBias, &lq);
    layer_t *deserialNoBias = conv1dLayerInitOwning(&initNoBias, &lq);

    conv1dConfig_t *serialBiasCfg = serialBias->config->conv1d;
    conv1dConfig_t *serialNoBiasCfg = serialNoBias->config->conv1d;

    /* Grads are zero at construction; Conv1d backward is FLOAT32-only
     * regardless of profile (Conv1dApi.c). Seed non-zero so the round trip
     * is distinguishable from a no-op. */
    size_t numberOfWeights = calcNumberOfElementsByTensor(serialBiasCfg->weights->param);
    float *weightGradSeed = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        weightGradSeed[i] = (float)i + 0.25f;
    }
    tensorFillFromFloatBuffer(serialBiasCfg->weights->grad, weightGradSeed, numberOfWeights);
    tensorFillFromFloatBuffer(serialNoBiasCfg->weights->grad, weightGradSeed, numberOfWeights);

    size_t numberOfBiases = calcNumberOfElementsByTensor(serialBiasCfg->bias->param);
    float *biasGradSeed = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        biasGradSeed[i] = (float)i - 0.75f;
    }
    tensorFillFromFloatBuffer(serialBiasCfg->bias->grad, biasGradSeed, numberOfBiases);

    layer_t *serialModel[] = {serialBias, serialNoBias};
    layer_t *deserialModel[] = {deserialBias, deserialNoBias};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 2, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 2, f);
    fclose(f);

    conv1dConfig_t *deserialBiasCfg = deserialBias->config->conv1d;
    conv1dConfig_t *deserialNoBiasCfg = deserialNoBias->config->conv1d;

    /* CAPTURE every assertion value before any free. */
    kernel_t capturedSerialKernel = *serialBiasCfg->kernel;
    kernel_t capturedDeserialKernel = *deserialBiasCfg->kernel;
    size_t capturedSerialGroups = serialBiasCfg->groups;
    size_t capturedDeserialGroups = deserialBiasCfg->groups;

    float *capturedSerialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedSerialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedSerialW[i] = ((float *)serialBiasCfg->weights->param->data)[i];
        capturedDeserialW[i] = ((float *)deserialBiasCfg->weights->param->data)[i];
        capturedSerialWGrad[i] = ((float *)serialBiasCfg->weights->grad->data)[i];
        capturedDeserialWGrad[i] = ((float *)deserialBiasCfg->weights->grad->data)[i];
    }

    float *capturedSerialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedSerialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedSerialB[i] = ((float *)serialBiasCfg->bias->param->data)[i];
        capturedDeserialB[i] = ((float *)deserialBiasCfg->bias->param->data)[i];
        capturedSerialBGrad[i] = ((float *)serialBiasCfg->bias->grad->data)[i];
        capturedDeserialBGrad[i] = ((float *)deserialBiasCfg->bias->grad->data)[i];
    }

    arithmetic_t capturedSerialForward = serialBiasCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialBiasCfg->forwardMath;
    arithmetic_t capturedSerialWeightGrad = serialBiasCfg->weightGradMath;
    arithmetic_t capturedDeserialWeightGrad = deserialBiasCfg->weightGradMath;
    arithmetic_t capturedSerialBiasGrad = serialBiasCfg->biasGradMath;
    arithmetic_t capturedDeserialBiasGrad = deserialBiasCfg->biasGradMath;
    arithmetic_t capturedSerialPropLoss = serialBiasCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialBiasCfg->propLossMath;

    qtype_t capturedSerialOutputQ = serialBiasCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialBiasCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialBiasCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialBiasCfg->propLossQ->type;

    bool capturedDeserialNoBiasIsNull = (deserialNoBiasCfg->bias == NULL);
    size_t numberOfNoBiasWeights = calcNumberOfElementsByTensor(serialNoBiasCfg->weights->param);
    float *capturedSerialNoBiasW = reserveMemory(numberOfNoBiasWeights * sizeof(float));
    float *capturedDeserialNoBiasW = reserveMemory(numberOfNoBiasWeights * sizeof(float));
    for (size_t i = 0; i < numberOfNoBiasWeights; i++) {
        capturedSerialNoBiasW[i] = ((float *)serialNoBiasCfg->weights->param->data)[i];
        capturedDeserialNoBiasW[i] = ((float *)deserialNoBiasCfg->weights->param->data)[i];
    }

    /* FREE in reverse-init order. */
    freeReservedMemory(biasGradSeed);
    freeReservedMemory(weightGradSeed);
    freeConv1dLayer(deserialNoBias);
    freeConv1dLayer(serialNoBias);
    freeConv1dLayer(deserialBias);
    freeConv1dLayer(serialBias);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL(3, capturedSerialKernel.size);
    TEST_ASSERT_EQUAL(capturedSerialKernel.size, capturedDeserialKernel.size);
    TEST_ASSERT_EQUAL(EXPLICIT, capturedSerialKernel.paddingType);
    TEST_ASSERT_EQUAL(capturedSerialKernel.paddingType, capturedDeserialKernel.paddingType);
    TEST_ASSERT_EQUAL(2, capturedSerialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.stride, capturedDeserialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.dilation, capturedDeserialKernel.dilation);
    TEST_ASSERT_EQUAL(1, capturedSerialKernel.padding);
    TEST_ASSERT_EQUAL(capturedSerialKernel.padding, capturedDeserialKernel.padding);

    TEST_ASSERT_EQUAL(1, capturedSerialGroups);
    TEST_ASSERT_EQUAL(capturedSerialGroups, capturedDeserialGroups);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW, capturedDeserialW, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialWGrad, capturedDeserialWGrad, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB, capturedDeserialB, numberOfBiases);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBGrad, capturedDeserialBGrad, numberOfBiases);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.type, capturedDeserialWeightGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.roundingMode,
                      capturedDeserialWeightGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.type, capturedDeserialBiasGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.roundingMode, capturedDeserialBiasGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    /* Sanity: the override actually makes the four fields non-uniform. */
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, capturedSerialForward.type);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialBiasGrad.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);

    /* No-bias variant: bias stays NULL on both sides (deserializeLayer never
     * reads into a NULL bias pointer when hasBias=0); its weights (the record
     * right after the with-bias layer's own record) round-trip correctly,
     * proving no desync between the two CONV1D records. */
    TEST_ASSERT_TRUE(capturedDeserialNoBiasIsNull);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialNoBiasW, capturedDeserialNoBiasW,
                                  numberOfNoBiasWeights);

    freeReservedMemory(capturedSerialNoBiasW);
    freeReservedMemory(capturedDeserialNoBiasW);
    freeReservedMemory(capturedSerialW);
    freeReservedMemory(capturedDeserialW);
    freeReservedMemory(capturedSerialWGrad);
    freeReservedMemory(capturedDeserialWGrad);
    freeReservedMemory(capturedSerialB);
    freeReservedMemory(capturedDeserialB);
    freeReservedMemory(capturedSerialBGrad);
    freeReservedMemory(capturedDeserialBGrad);
}

/*! CONV1D_TRANSPOSED round trip: CONV1D layout + outputPadding. Non-trivial
 *  geometry (kernel size 3, stride 2, outputPadding 1 < max(stride,
 *  dilation)=2, groups 1); Phase-1 only supports VALID padding for this
 *  layer. `weightGradMath` bumped to SYM_INT32 for the field-order-swap
 *  detector. */
static void testRoundTripConv1dTransposed(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.weightGradMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    conv1dTransposedInit_t init = {
        .inChannels = 4,
        .outChannels = 6,
        .kernelSize = 3,
        .stride = 2,
        .groups = 1,
        .outputPadding = 1,
        .bias = BIAS_TRUE,
    };

    layer_t *serialLayer = conv1dTransposedLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = conv1dTransposedLayerInitOwning(&init, &lq);

    conv1dTransposedConfig_t *serialCfg = serialLayer->config->conv1dTransposed;

    size_t numberOfWeights = calcNumberOfElementsByTensor(serialCfg->weights->param);
    float *weightGradSeed = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        weightGradSeed[i] = (float)i + 0.5f;
    }
    tensorFillFromFloatBuffer(serialCfg->weights->grad, weightGradSeed, numberOfWeights);

    size_t numberOfBiases = calcNumberOfElementsByTensor(serialCfg->bias->param);
    float *biasGradSeed = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        biasGradSeed[i] = (float)i - 1.0f;
    }
    tensorFillFromFloatBuffer(serialCfg->bias->grad, biasGradSeed, numberOfBiases);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    conv1dTransposedConfig_t *deserialCfg = deserialLayer->config->conv1dTransposed;

    kernel_t capturedSerialKernel = *serialCfg->kernel;
    kernel_t capturedDeserialKernel = *deserialCfg->kernel;
    size_t capturedSerialGroups = serialCfg->groups;
    size_t capturedDeserialGroups = deserialCfg->groups;
    size_t capturedSerialOutputPadding = serialCfg->outputPadding;
    size_t capturedDeserialOutputPadding = deserialCfg->outputPadding;

    float *capturedSerialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedSerialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialWGrad = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedSerialW[i] = ((float *)serialCfg->weights->param->data)[i];
        capturedDeserialW[i] = ((float *)deserialCfg->weights->param->data)[i];
        capturedSerialWGrad[i] = ((float *)serialCfg->weights->grad->data)[i];
        capturedDeserialWGrad[i] = ((float *)deserialCfg->weights->grad->data)[i];
    }
    float *capturedSerialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedSerialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialBGrad = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedSerialB[i] = ((float *)serialCfg->bias->param->data)[i];
        capturedDeserialB[i] = ((float *)deserialCfg->bias->param->data)[i];
        capturedSerialBGrad[i] = ((float *)serialCfg->bias->grad->data)[i];
        capturedDeserialBGrad[i] = ((float *)deserialCfg->bias->grad->data)[i];
    }

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialWeightGrad = serialCfg->weightGradMath;
    arithmetic_t capturedDeserialWeightGrad = deserialCfg->weightGradMath;
    arithmetic_t capturedSerialBiasGrad = serialCfg->biasGradMath;
    arithmetic_t capturedDeserialBiasGrad = deserialCfg->biasGradMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;

    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeConv1dTransposedLayer(deserialLayer);
    freeConv1dTransposedLayer(serialLayer);
    freeReservedMemory(biasGradSeed);
    freeReservedMemory(weightGradSeed);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(3, capturedSerialKernel.size);
    TEST_ASSERT_EQUAL(capturedSerialKernel.size, capturedDeserialKernel.size);
    TEST_ASSERT_EQUAL(capturedSerialKernel.paddingType, capturedDeserialKernel.paddingType);
    TEST_ASSERT_EQUAL(2, capturedSerialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.stride, capturedDeserialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.dilation, capturedDeserialKernel.dilation);
    TEST_ASSERT_EQUAL(capturedSerialKernel.padding, capturedDeserialKernel.padding);

    TEST_ASSERT_EQUAL(1, capturedSerialGroups);
    TEST_ASSERT_EQUAL(capturedSerialGroups, capturedDeserialGroups);
    TEST_ASSERT_EQUAL(1, capturedSerialOutputPadding);
    TEST_ASSERT_EQUAL(capturedSerialOutputPadding, capturedDeserialOutputPadding);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW, capturedDeserialW, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialWGrad, capturedDeserialWGrad, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB, capturedDeserialB, numberOfBiases);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBGrad, capturedDeserialBGrad, numberOfBiases);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.type, capturedDeserialWeightGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialWeightGrad.roundingMode,
                      capturedDeserialWeightGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.type, capturedDeserialBiasGrad.type);
    TEST_ASSERT_EQUAL(capturedSerialBiasGrad.roundingMode, capturedDeserialBiasGrad.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialWeightGrad.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);

    freeReservedMemory(capturedSerialW);
    freeReservedMemory(capturedDeserialW);
    freeReservedMemory(capturedSerialWGrad);
    freeReservedMemory(capturedDeserialWGrad);
    freeReservedMemory(capturedSerialB);
    freeReservedMemory(capturedDeserialB);
    freeReservedMemory(capturedSerialBGrad);
    freeReservedMemory(capturedDeserialBGrad);
}

/*! MAXPOOL1D round trip: non-trivial kernel (size 3, stride 2, VALID);
 *  argmaxIndices is runtime state and is deliberately NOT asserted (brief:
 *  not serialized). `propLossMath` bumped to SYM_INT32. */
static void testRoundTripMaxPool1d(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    maxPool1dInit_t init = {
        .kernelSize = 3,
        .inputChannels = 4,
        .inputLength = 10,
        .stride = 2,
    };

    layer_t *serialLayer = maxPool1dLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = maxPool1dLayerInitOwning(&init, &lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    maxPool1dConfig_t *serialCfg = serialLayer->config->maxPool1d;
    maxPool1dConfig_t *deserialCfg = deserialLayer->config->maxPool1d;

    kernel_t capturedSerialKernel = *serialCfg->kernel;
    kernel_t capturedDeserialKernel = *deserialCfg->kernel;
    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeMaxPool1dLayer(deserialLayer);
    freeMaxPool1dLayer(serialLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(3, capturedSerialKernel.size);
    TEST_ASSERT_EQUAL(capturedSerialKernel.size, capturedDeserialKernel.size);
    TEST_ASSERT_EQUAL(VALID, capturedSerialKernel.paddingType);
    TEST_ASSERT_EQUAL(capturedSerialKernel.paddingType, capturedDeserialKernel.paddingType);
    TEST_ASSERT_EQUAL(2, capturedSerialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.stride, capturedDeserialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.dilation, capturedDeserialKernel.dilation);
    TEST_ASSERT_EQUAL(capturedSerialKernel.padding, capturedDeserialKernel.padding);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialPropLoss.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);
}

/*! AVGPOOL1D round trip: non-trivial kernel (size 3, stride 2, VALID).
 *  `forwardMath` bumped to SYM_INT32. */
static void testRoundTripAvgPool1d(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.forwardMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    avgPool1dInit_t init = {
        .kernelSize = 3,
        .stride = 2,
    };

    layer_t *serialLayer = avgPool1dLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = avgPool1dLayerInitOwning(&init, &lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    avgPool1dConfig_t *serialCfg = serialLayer->config->avgPool1d;
    avgPool1dConfig_t *deserialCfg = deserialLayer->config->avgPool1d;

    kernel_t capturedSerialKernel = *serialCfg->kernel;
    kernel_t capturedDeserialKernel = *deserialCfg->kernel;
    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeAvgPool1dLayer(deserialLayer);
    freeAvgPool1dLayer(serialLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(3, capturedSerialKernel.size);
    TEST_ASSERT_EQUAL(capturedSerialKernel.size, capturedDeserialKernel.size);
    TEST_ASSERT_EQUAL(VALID, capturedSerialKernel.paddingType);
    TEST_ASSERT_EQUAL(capturedSerialKernel.paddingType, capturedDeserialKernel.paddingType);
    TEST_ASSERT_EQUAL(2, capturedSerialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.stride, capturedDeserialKernel.stride);
    TEST_ASSERT_EQUAL(capturedSerialKernel.dilation, capturedDeserialKernel.dilation);
    TEST_ASSERT_EQUAL(capturedSerialKernel.padding, capturedDeserialKernel.padding);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialForward.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);
}

/*! ADAPTIVE_AVGPOOL1D round trip: non-trivial outputSize (5, not the trivial
 *  default of 1). `propLossMath` bumped to SYM_INT32. */
static void testRoundTripAdaptiveAvgPool1d(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = SR_HALF_AWAY};

    adaptiveAvgPool1dInit_t init = {.outputSize = 5};

    layer_t *serialLayer = adaptiveAvgPool1dLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = adaptiveAvgPool1dLayerInitOwning(&init, &lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    adaptiveAvgPool1dConfig_t *serialCfg = serialLayer->config->adaptiveAvgPool1d;
    adaptiveAvgPool1dConfig_t *deserialCfg = deserialLayer->config->adaptiveAvgPool1d;

    size_t capturedSerialOutputSize = serialCfg->outputSize;
    size_t capturedDeserialOutputSize = deserialCfg->outputSize;
    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeAdaptiveAvgPool1dLayer(deserialLayer);
    freeAdaptiveAvgPool1dLayer(serialLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(5, capturedSerialOutputSize);
    TEST_ASSERT_EQUAL(capturedSerialOutputSize, capturedDeserialOutputSize);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialPropLoss.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);
}

/*! DROPOUT round trip. `mask`/`training` are runtime state and are
 *  deliberately NOT asserted (brief: not serialized). `forwardQ`/`backwardQ`
 *  are DIFFERENT quantization types (FLOAT32 vs SYM_INT32) so the two
 *  derived arithmetic fields are non-uniform (field-order-swap detector) —
 *  Dropout has no Owning factory, so both layers borrow the SAME
 *  forwardQ/backwardQ pointers (freed once, manually, at the end). */
static void testRoundTripDropout(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symIntQ = quantizationInitSymInt32(SR_HALF_AWAY);

    size_t maskElements = 8;
    size_t *serialDims = reserveMemory(1 * sizeof(size_t));
    serialDims[0] = maskElements;
    size_t *serialOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, serialOrder);
    shape_t *serialMaskShape = reserveMemory(sizeof(shape_t));
    setShape(serialMaskShape, serialDims, 1, serialOrder);
    tensor_t *serialMask = initTensor(serialMaskShape, quantizationInitBool(), NULL);

    size_t *deserialDims = reserveMemory(1 * sizeof(size_t));
    deserialDims[0] = maskElements;
    size_t *deserialOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, deserialOrder);
    shape_t *deserialMaskShape = reserveMemory(sizeof(shape_t));
    setShape(deserialMaskShape, deserialDims, 1, deserialOrder);
    tensor_t *deserialMask = initTensor(deserialMaskShape, quantizationInitBool(), NULL);

    layer_t *serialLayer = dropoutLayerInit(0.3f, serialMask, floatQ, symIntQ);
    layer_t *deserialLayer = dropoutLayerInit(0.3f, deserialMask, floatQ, symIntQ);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    dropoutConfig_t *serialCfg = serialLayer->config->dropout;
    dropoutConfig_t *deserialCfg = deserialLayer->config->dropout;

    float capturedSerialP = serialCfg->p;
    float capturedDeserialP = deserialCfg->p;
    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeDropoutLayer(deserialLayer);
    freeDropoutLayer(serialLayer);
    freeTensor(deserialMask);
    freeTensor(serialMask);
    freeQuantization(symIntQ);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL_FLOAT(0.3f, capturedSerialP);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialP, capturedDeserialP);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(ARITH_FLOAT32, capturedSerialForward.type);
    TEST_ASSERT_EQUAL(ARITH_SYM_INT32, capturedSerialPropLoss.type);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);
}

/*! LAYERNORM round trip. Non-trivial normalizedShape (rank 2: {4, 5}) and a
 *  non-default eps (1e-3f, catching a bug where eps is silently treated as
 *  unset-zero-init) exercise the numNormDims-driven shape-array read count.
 *  gamma/beta are constant-initialized (all-1 / all-0) by the factory, so
 *  BOTH layers would start identical without an explicit seed — gamma/beta
 *  PARAM values are seeded with distinct non-constant floats (grads are
 *  seeded too, same pattern as testRoundTripLinear) so the round trip is
 *  genuinely distinguishable from a no-op. `forwardMath`/`propLossMath` are
 *  both FLOAT32 but differ only in roundingMode (HALF_AWAY vs SR_HALF_AWAY)
 *  — sufficient to catch a field-order swap without tripping LayerNorm's
 *  SYM_INT32 forward/backward coupling validation. */
static void testRoundTripLayerNorm(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    lq.propLossMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};

    size_t normalizedShapeValues[] = {4, 5};
    layerNormInit_t init = {
        .normalizedShape = normalizedShapeValues,
        .numNormDims = 2,
        .eps = 1e-3f,
    };

    layer_t *serialLayer = layerNormLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = layerNormLayerInitOwning(&init, &lq);

    layerNormConfig_t *serialCfg = serialLayer->config->layerNorm;

    size_t numberOfGammaElements = calcNumberOfElementsByTensor(serialCfg->gamma->param);
    float *gammaSeed = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        gammaSeed[i] = 2.0f + (float)i * 0.1f;
    }
    tensorFillFromFloatBuffer(serialCfg->gamma->param, gammaSeed, numberOfGammaElements);

    size_t numberOfBetaElements = calcNumberOfElementsByTensor(serialCfg->beta->param);
    float *betaSeed = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        betaSeed[i] = -1.0f - (float)i * 0.2f;
    }
    tensorFillFromFloatBuffer(serialCfg->beta->param, betaSeed, numberOfBetaElements);

    /* Grads are zero at construction; seed non-zero so the round trip is
     * distinguishable from a no-op. */
    float *gammaGradSeed = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        gammaGradSeed[i] = (float)i + 0.3f;
    }
    tensorFillFromFloatBuffer(serialCfg->gamma->grad, gammaGradSeed, numberOfGammaElements);

    float *betaGradSeed = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        betaGradSeed[i] = (float)i - 0.4f;
    }
    tensorFillFromFloatBuffer(serialCfg->beta->grad, betaGradSeed, numberOfBetaElements);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    layerNormConfig_t *deserialCfg = deserialLayer->config->layerNorm;

    size_t capturedSerialNumNormDims = serialCfg->numNormDims;
    size_t capturedDeserialNumNormDims = deserialCfg->numNormDims;
    size_t capturedSerialShape[2];
    size_t capturedDeserialShape[2];
    for (size_t d = 0; d < 2; d++) {
        capturedSerialShape[d] = serialCfg->normalizedShape[d];
        capturedDeserialShape[d] = deserialCfg->normalizedShape[d];
    }
    float capturedSerialEps = serialCfg->eps;
    float capturedDeserialEps = deserialCfg->eps;

    float *capturedSerialGamma = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedDeserialGamma = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedSerialGammaGrad = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedDeserialGammaGrad = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        capturedSerialGamma[i] = ((float *)serialCfg->gamma->param->data)[i];
        capturedDeserialGamma[i] = ((float *)deserialCfg->gamma->param->data)[i];
        capturedSerialGammaGrad[i] = ((float *)serialCfg->gamma->grad->data)[i];
        capturedDeserialGammaGrad[i] = ((float *)deserialCfg->gamma->grad->data)[i];
    }
    float *capturedSerialBeta = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedDeserialBeta = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedSerialBetaGrad = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedDeserialBetaGrad = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        capturedSerialBeta[i] = ((float *)serialCfg->beta->param->data)[i];
        capturedDeserialBeta[i] = ((float *)deserialCfg->beta->param->data)[i];
        capturedSerialBetaGrad[i] = ((float *)serialCfg->beta->grad->data)[i];
        capturedDeserialBetaGrad[i] = ((float *)deserialCfg->beta->grad->data)[i];
    }

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeReservedMemory(betaGradSeed);
    freeReservedMemory(gammaGradSeed);
    freeReservedMemory(betaSeed);
    freeReservedMemory(gammaSeed);
    freeLayerNormLayer(deserialLayer);
    freeLayerNormLayer(serialLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(2, capturedSerialNumNormDims);
    TEST_ASSERT_EQUAL(capturedSerialNumNormDims, capturedDeserialNumNormDims);
    TEST_ASSERT_EQUAL_size_t_ARRAY(capturedSerialShape, capturedDeserialShape, 2);
    TEST_ASSERT_EQUAL(4, capturedSerialShape[0]);
    TEST_ASSERT_EQUAL(5, capturedSerialShape[1]);
    TEST_ASSERT_EQUAL_FLOAT(1e-3f, capturedSerialEps);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialEps, capturedDeserialEps);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialGamma, capturedDeserialGamma,
                                  numberOfGammaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialGammaGrad, capturedDeserialGammaGrad,
                                  numberOfGammaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBeta, capturedDeserialBeta, numberOfBetaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBetaGrad, capturedDeserialBetaGrad,
                                  numberOfBetaElements);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(HALF_AWAY, capturedSerialForward.roundingMode);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, capturedSerialPropLoss.roundingMode);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);

    freeReservedMemory(capturedSerialGamma);
    freeReservedMemory(capturedDeserialGamma);
    freeReservedMemory(capturedSerialGammaGrad);
    freeReservedMemory(capturedDeserialGammaGrad);
    freeReservedMemory(capturedSerialBeta);
    freeReservedMemory(capturedDeserialBeta);
    freeReservedMemory(capturedSerialBetaGrad);
    freeReservedMemory(capturedDeserialBetaGrad);
}

/*! GROUPNORM round trip. numGroups=2 / numChannels=4 (channels-per-group 2) and
 *  a non-default eps (1e-3f) exercise the two uint32 geometry fields distinctly:
 *  a numGroups<->numChannels swap still divides evenly but mis-maps, and a
 *  dropped field misaligns every later byte. gamma/beta are constant-initialized
 *  (all-1 / all-0) by the factory, so BOTH layers would start identical without
 *  an explicit seed — gamma/beta PARAM (and grad) values are seeded with
 *  distinct non-constant floats (same pattern as testRoundTripLayerNorm) so the
 *  round trip is genuinely distinguishable from a no-op. `forwardMath`/
 *  `propLossMath` are both FLOAT32 but differ only in roundingMode (HALF_AWAY vs
 *  SR_HALF_AWAY) — sufficient to catch a field-order swap without tripping
 *  GroupNorm's SYM_INT32 forward/backward coupling validation. */
static void testRoundTripGroupNorm(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    lq.propLossMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = SR_HALF_AWAY};

    groupNormInit_t init = {
        .numGroups = 2,
        .numChannels = 4,
        .eps = 1e-3f,
    };

    layer_t *serialLayer = groupNormLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = groupNormLayerInitOwning(&init, &lq);

    groupNormConfig_t *serialCfg = serialLayer->config->groupNorm;

    size_t numberOfGammaElements = calcNumberOfElementsByTensor(serialCfg->gamma->param);
    float *gammaSeed = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        gammaSeed[i] = 2.0f + (float)i * 0.1f;
    }
    tensorFillFromFloatBuffer(serialCfg->gamma->param, gammaSeed, numberOfGammaElements);

    size_t numberOfBetaElements = calcNumberOfElementsByTensor(serialCfg->beta->param);
    float *betaSeed = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        betaSeed[i] = -1.0f - (float)i * 0.2f;
    }
    tensorFillFromFloatBuffer(serialCfg->beta->param, betaSeed, numberOfBetaElements);

    /* Grads are zero at construction; seed non-zero so the round trip is
     * distinguishable from a no-op. */
    float *gammaGradSeed = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        gammaGradSeed[i] = (float)i + 0.3f;
    }
    tensorFillFromFloatBuffer(serialCfg->gamma->grad, gammaGradSeed, numberOfGammaElements);

    float *betaGradSeed = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        betaGradSeed[i] = (float)i - 0.4f;
    }
    tensorFillFromFloatBuffer(serialCfg->beta->grad, betaGradSeed, numberOfBetaElements);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    groupNormConfig_t *deserialCfg = deserialLayer->config->groupNorm;

    size_t capturedSerialNumGroups = serialCfg->numGroups;
    size_t capturedDeserialNumGroups = deserialCfg->numGroups;
    size_t capturedSerialNumChannels = serialCfg->numChannels;
    size_t capturedDeserialNumChannels = deserialCfg->numChannels;
    float capturedSerialEps = serialCfg->eps;
    float capturedDeserialEps = deserialCfg->eps;

    float *capturedSerialGamma = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedDeserialGamma = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedSerialGammaGrad = reserveMemory(numberOfGammaElements * sizeof(float));
    float *capturedDeserialGammaGrad = reserveMemory(numberOfGammaElements * sizeof(float));
    for (size_t i = 0; i < numberOfGammaElements; i++) {
        capturedSerialGamma[i] = ((float *)serialCfg->gamma->param->data)[i];
        capturedDeserialGamma[i] = ((float *)deserialCfg->gamma->param->data)[i];
        capturedSerialGammaGrad[i] = ((float *)serialCfg->gamma->grad->data)[i];
        capturedDeserialGammaGrad[i] = ((float *)deserialCfg->gamma->grad->data)[i];
    }
    float *capturedSerialBeta = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedDeserialBeta = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedSerialBetaGrad = reserveMemory(numberOfBetaElements * sizeof(float));
    float *capturedDeserialBetaGrad = reserveMemory(numberOfBetaElements * sizeof(float));
    for (size_t i = 0; i < numberOfBetaElements; i++) {
        capturedSerialBeta[i] = ((float *)serialCfg->beta->param->data)[i];
        capturedDeserialBeta[i] = ((float *)deserialCfg->beta->param->data)[i];
        capturedSerialBetaGrad[i] = ((float *)serialCfg->beta->grad->data)[i];
        capturedDeserialBetaGrad[i] = ((float *)deserialCfg->beta->grad->data)[i];
    }

    arithmetic_t capturedSerialForward = serialCfg->forwardMath;
    arithmetic_t capturedDeserialForward = deserialCfg->forwardMath;
    arithmetic_t capturedSerialPropLoss = serialCfg->propLossMath;
    arithmetic_t capturedDeserialPropLoss = deserialCfg->propLossMath;
    qtype_t capturedSerialOutputQ = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQ = deserialCfg->outputQ->type;
    qtype_t capturedSerialPropLossQ = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQ = deserialCfg->propLossQ->type;

    freeReservedMemory(betaGradSeed);
    freeReservedMemory(gammaGradSeed);
    freeReservedMemory(betaSeed);
    freeReservedMemory(gammaSeed);
    freeGroupNormLayer(deserialLayer);
    freeGroupNormLayer(serialLayer);
    freeQuantization(floatQ);

    TEST_ASSERT_EQUAL(2, capturedSerialNumGroups);
    TEST_ASSERT_EQUAL(capturedSerialNumGroups, capturedDeserialNumGroups);
    TEST_ASSERT_EQUAL(4, capturedSerialNumChannels);
    TEST_ASSERT_EQUAL(capturedSerialNumChannels, capturedDeserialNumChannels);
    TEST_ASSERT_EQUAL_FLOAT(1e-3f, capturedSerialEps);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialEps, capturedDeserialEps);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialGamma, capturedDeserialGamma,
                                  numberOfGammaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialGammaGrad, capturedDeserialGammaGrad,
                                  numberOfGammaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBeta, capturedDeserialBeta, numberOfBetaElements);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBetaGrad, capturedDeserialBetaGrad,
                                  numberOfBetaElements);

    TEST_ASSERT_EQUAL(capturedSerialForward.type, capturedDeserialForward.type);
    TEST_ASSERT_EQUAL(capturedSerialForward.roundingMode, capturedDeserialForward.roundingMode);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.type, capturedDeserialPropLoss.type);
    TEST_ASSERT_EQUAL(capturedSerialPropLoss.roundingMode, capturedDeserialPropLoss.roundingMode);
    TEST_ASSERT_EQUAL(HALF_AWAY, capturedSerialForward.roundingMode);
    TEST_ASSERT_EQUAL(SR_HALF_AWAY, capturedSerialPropLoss.roundingMode);

    TEST_ASSERT_EQUAL(capturedSerialOutputQ, capturedDeserialOutputQ);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQ, capturedDeserialPropLossQ);

    freeReservedMemory(capturedSerialGamma);
    freeReservedMemory(capturedDeserialGamma);
    freeReservedMemory(capturedSerialGammaGrad);
    freeReservedMemory(capturedDeserialGammaGrad);
    freeReservedMemory(capturedSerialBeta);
    freeReservedMemory(capturedDeserialBeta);
    freeReservedMemory(capturedSerialBetaGrad);
    freeReservedMemory(capturedDeserialBetaGrad);
}

/*! QUANTIZATION round trip. `outputQ`/`propLossQ` are set to DIFFERENT
 *  storage dtypes (SYM / ASYM) so the record exercises BOTH the SYM
 *  roundingMode fix and ASYM's full qConfig (scale, qBits, roundingMode,
 *  zeroPoint) — neither type is uniform, catching an outputQ/propLossQ
 *  field-order swap. This layer has no arithmetic fields at all. */
static void testRoundTripQuantizationLayer(void) {
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);
    quantization_t *asymPropLossQ = quantizationInitAsym(8, SR_HALF_AWAY);

    layerQuant_t lq = {0};
    lq.outputQ = symOutputQ;
    lq.propLossQ = asymPropLossQ;

    layer_t *serialLayer = quantLayerInitOwning(&lq);
    layer_t *deserialLayer = quantLayerInitOwning(&lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    quantizationConfig_t *serialCfg = serialLayer->config->quantization;
    quantizationConfig_t *deserialCfg = deserialLayer->config->quantization;

    qtype_t capturedSerialOutputQType = serialCfg->outputQ->type;
    qtype_t capturedDeserialOutputQType = deserialCfg->outputQ->type;
    symQConfig_t *serialOutputCfg = serialCfg->outputQ->qConfig;
    symQConfig_t *deserialOutputCfg = deserialCfg->outputQ->qConfig;
    float capturedSerialOutputScale = serialOutputCfg->scale;
    float capturedDeserialOutputScale = deserialOutputCfg->scale;
    uint8_t capturedSerialOutputQBits = serialOutputCfg->qBits;
    uint8_t capturedDeserialOutputQBits = deserialOutputCfg->qBits;
    roundingMode_t capturedSerialOutputRM = serialOutputCfg->roundingMode;
    roundingMode_t capturedDeserialOutputRM = deserialOutputCfg->roundingMode;

    qtype_t capturedSerialPropLossQType = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQType = deserialCfg->propLossQ->type;
    asymQConfig_t *serialPropLossCfg = serialCfg->propLossQ->qConfig;
    asymQConfig_t *deserialPropLossCfg = deserialCfg->propLossQ->qConfig;
    float capturedSerialPropLossScale = serialPropLossCfg->scale;
    float capturedDeserialPropLossScale = deserialPropLossCfg->scale;
    uint8_t capturedSerialPropLossQBits = serialPropLossCfg->qBits;
    uint8_t capturedDeserialPropLossQBits = deserialPropLossCfg->qBits;
    roundingMode_t capturedSerialPropLossRM = serialPropLossCfg->roundingMode;
    roundingMode_t capturedDeserialPropLossRM = deserialPropLossCfg->roundingMode;
    int16_t capturedSerialPropLossZP = serialPropLossCfg->zeroPoint;
    int16_t capturedDeserialPropLossZP = deserialPropLossCfg->zeroPoint;

    freeQuantLayer(deserialLayer);
    freeQuantLayer(serialLayer);
    freeQuantization(asymPropLossQ);
    freeQuantization(symOutputQ);

    TEST_ASSERT_EQUAL(SYM, capturedSerialOutputQType);
    TEST_ASSERT_EQUAL(capturedSerialOutputQType, capturedDeserialOutputQType);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialOutputScale, capturedDeserialOutputScale);
    TEST_ASSERT_EQUAL(capturedSerialOutputQBits, capturedDeserialOutputQBits);
    TEST_ASSERT_EQUAL(capturedSerialOutputRM, capturedDeserialOutputRM);

    TEST_ASSERT_EQUAL(ASYM, capturedSerialPropLossQType);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQType, capturedDeserialPropLossQType);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialPropLossScale, capturedDeserialPropLossScale);
    TEST_ASSERT_EQUAL(capturedSerialPropLossQBits, capturedDeserialPropLossQBits);
    TEST_ASSERT_EQUAL(capturedSerialPropLossRM, capturedDeserialPropLossRM);
    TEST_ASSERT_EQUAL(capturedSerialPropLossZP, capturedDeserialPropLossZP);
}

/*! SYM sub-byte tensor DATA record: qBits=4, N=10 -> exactly ceil(40/8) = 5
 *  packed data bytes on the wire. The u32 sentinel written right after the
 *  record catches any sizing skew between writer and reader (one-sided
 *  mutation -> misaligned sentinel).
 *  Mutation guard: reverting BOTH sides to N * calcBytesPerElement makes
 *  serialize fwrite 10 bytes from the 5-byte heap buffer and deserialize
 *  fread 10 into it -> ASan RED (over-read + OOB write); reverting ONE side
 *  misaligns the sentinel -> RED in plain debug too. */
static void testSerializeTensorSymSubByteRoundTripsPackedData(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 5;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitSym(4, HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(src, (float[]){1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -6.f, 5.f, -4.f},
                              10);
    float srcScale = ((symQConfig_t *)src->quantization->qConfig)->scale;
    uint8_t srcBytes[5];
    memcpy(srcBytes, src->data, 5);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    uint32_t sentinelOut = 0x53454E54u;
    fwrite(&sentinelOut, sizeof(sentinelOut), 1, f);
    fclose(f);

    size_t *dDims = reserveMemory(2 * sizeof(size_t));
    dDims[0] = 2;
    dDims[1] = 5;
    size_t *dOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, dOrder);
    shape_t *dShape = reserveMemory(sizeof(shape_t));
    setShape(dShape, dDims, 2, dOrder);
    tensor_t *dst = initTensor(dShape, quantizationInitSym(4, HALF_AWAY), NULL);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    uint32_t sentinelIn = 0;
    fread(&sentinelIn, sizeof(sentinelIn), 1, f);
    fclose(f);

    /* CAPTURE -> free -> assert (file convention). */
    float dstScale = ((symQConfig_t *)dst->quantization->qConfig)->scale;
    uint8_t dstBytes[5];
    memcpy(dstBytes, dst->data, 5);
    freeTensor(src);
    freeTensor(dst);

    TEST_ASSERT_EQUAL_HEX32(0x53454E54u, sentinelIn);
    TEST_ASSERT_EQUAL_FLOAT(srcScale, dstScale);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(srcBytes, dstBytes, 5);
}

/*! BOOL tensor DATA record: N=12 bools -> 2 packed bytes on the wire
 *  (pre-fix: 12). Same sentinel discipline as the SYM round trip. */
static void testSerializeTensorBoolRoundTripsPackedData(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitBool(), NULL);
    bool pattern[12] = {true, false, true,  true, false, false,
                        true, false, false, true, true,  false};
    tensorFillFromBoolBuffer(src, pattern, 12);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    uint32_t sentinelOut = 0x53454E54u;
    fwrite(&sentinelOut, sizeof(sentinelOut), 1, f);
    fclose(f);

    size_t *dDims = reserveMemory(2 * sizeof(size_t));
    dDims[0] = 3;
    dDims[1] = 4;
    size_t *dOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, dOrder);
    shape_t *dShape = reserveMemory(sizeof(shape_t));
    setShape(dShape, dDims, 2, dOrder);
    tensor_t *dst = initTensor(dShape, quantizationInitBool(), NULL);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    uint32_t sentinelIn = 0;
    fread(&sentinelIn, sizeof(sentinelIn), 1, f);
    fclose(f);

    bool got[12];
    for (size_t i = 0; i < 12; i++) {
        got[i] = tensorBoolGet(dst, i);
    }
    freeTensor(src);
    freeTensor(dst);

    TEST_ASSERT_EQUAL_HEX32(0x53454E54u, sentinelIn);
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_EQUAL_INT(pattern[i], got[i]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRoundTripLinear);
    RUN_TEST(testRoundTripRelu);
    RUN_TEST(testRoundTripSoftmax);
    RUN_TEST(testRoundTripFlatten);
    RUN_TEST(testRoundTripConv1d);
    RUN_TEST(testRoundTripConv1dTransposed);
    RUN_TEST(testRoundTripMaxPool1d);
    RUN_TEST(testRoundTripAvgPool1d);
    RUN_TEST(testRoundTripAdaptiveAvgPool1d);
    RUN_TEST(testRoundTripDropout);
    RUN_TEST(testRoundTripLayerNorm);
    RUN_TEST(testRoundTripGroupNorm);
    RUN_TEST(testRoundTripQuantizationLayer);
    RUN_TEST(testSerializeTensorSymSubByteRoundTripsPackedData);
    RUN_TEST(testSerializeTensorBoolRoundTripsPackedData);
    return UNITY_END();
}
