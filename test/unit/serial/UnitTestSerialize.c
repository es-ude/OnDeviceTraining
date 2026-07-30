#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AdaptivePool1dApi.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "BorrowedLayer.h"
#include "Conv1d.h"
#include "Conv1dApi.h"
#include "Conv1dTransposed.h"
#include "Conv1dTransposedApi.h"
#include "DeathTest.h"
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

/* WIRE-FORMAT PINS (#serial): the serialized layer record's uint8 tag IS the
 * layerType_t enum position (Serialize.c serializeLayer). The enum is an
 * append-only wire contract -- inserting or reordering members silently
 * corrupts every previously serialized model. If one of these asserts fires,
 * you reordered/inserted; move your new member to the END instead. When
 * appending, add its pin line here and bump the count pin. */
_Static_assert(LINEAR == 0, "layerType_t wire tag: LINEAR must stay 0 (append-only enum)");
_Static_assert(RELU == 1, "layerType_t wire tag: RELU must stay 1 (append-only enum)");
_Static_assert(CONV1D == 2, "layerType_t wire tag: CONV1D must stay 2 (append-only enum)");
_Static_assert(CONV1D_TRANSPOSED == 3,
               "layerType_t wire tag: CONV1D_TRANSPOSED must stay 3 (append-only enum)");
_Static_assert(MAXPOOL1D == 4, "layerType_t wire tag: MAXPOOL1D must stay 4 (append-only enum)");
_Static_assert(AVGPOOL1D == 5, "layerType_t wire tag: AVGPOOL1D must stay 5 (append-only enum)");
_Static_assert(SOFTMAX == 6, "layerType_t wire tag: SOFTMAX must stay 6 (append-only enum)");
_Static_assert(FLATTEN == 7, "layerType_t wire tag: FLATTEN must stay 7 (append-only enum)");
_Static_assert(QUANTIZATION == 8,
               "layerType_t wire tag: QUANTIZATION must stay 8 (append-only enum)");
_Static_assert(ADAPTIVE_AVGPOOL1D == 9,
               "layerType_t wire tag: ADAPTIVE_AVGPOOL1D must stay 9 (append-only enum)");
_Static_assert(DROPOUT == 10, "layerType_t wire tag: DROPOUT must stay 10 (append-only enum)");
_Static_assert(LAYERNORM == 11, "layerType_t wire tag: LAYERNORM must stay 11 (append-only enum)");
_Static_assert(GROUPNORM == 12, "layerType_t wire tag: GROUPNORM must stay 12 (append-only enum)");
_Static_assert(GROUPNORM + 1 == 13,
               "new layerType_t member: append at the END and add its wire-tag pin above");

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

/*! LINEAR frozen round trip (#380 ODTS v3): both mirrors are built via the
 *  TRAINABLE_FALSE factory knob, so parameter_t.grad is NULL for weights AND
 *  bias on both sides. Pre-v3, serializeParameter/deserializeParameter always
 *  touched parameter->grad unconditionally -- a NULL-deref crash for a frozen
 *  model. The presence byte lets the wire skip the grad tensor entirely: param
 *  values round-trip byte-identical, no crash, and grad stays NULL. */
static void testRoundTripLinearFrozen(void) {
    quantization_t *floatQ = quantizationInitFloat();

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    linearInit_t init = {.inFeatures = 4, .outFeatures = 3, .trainable = TRAINABLE_FALSE};

    layer_t *serialLayer = linearLayerInitOwning(&init, &lq);
    layer_t *deserialLayer = linearLayerInitOwning(&init, &lq);

    linearConfig_t *serialCfg = serialLayer->config->linear;
    linearConfig_t *deserialCfg = deserialLayer->config->linear;

    size_t numberOfWeights = calcNumberOfElementsByTensor(serialCfg->weights->param);
    size_t numberOfBiases = calcNumberOfElementsByTensor(serialCfg->bias->param);

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    bool capturedSerialWeightsGradNull = (serialCfg->weights->grad == NULL);
    bool capturedDeserialWeightsGradNull = (deserialCfg->weights->grad == NULL);
    bool capturedSerialBiasGradNull = (serialCfg->bias->grad == NULL);
    bool capturedDeserialBiasGradNull = (deserialCfg->bias->grad == NULL);

    float *capturedSerialW = reserveMemory(numberOfWeights * sizeof(float));
    float *capturedDeserialW = reserveMemory(numberOfWeights * sizeof(float));
    for (size_t i = 0; i < numberOfWeights; i++) {
        capturedSerialW[i] = ((float *)serialCfg->weights->param->data)[i];
        capturedDeserialW[i] = ((float *)deserialCfg->weights->param->data)[i];
    }
    float *capturedSerialB = reserveMemory(numberOfBiases * sizeof(float));
    float *capturedDeserialB = reserveMemory(numberOfBiases * sizeof(float));
    for (size_t i = 0; i < numberOfBiases; i++) {
        capturedSerialB[i] = ((float *)serialCfg->bias->param->data)[i];
        capturedDeserialB[i] = ((float *)deserialCfg->bias->param->data)[i];
    }

    /* FREE in reverse-init order. */
    freeLinearLayer(deserialLayer);
    freeLinearLayer(serialLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(capturedSerialWeightsGradNull);
    TEST_ASSERT_TRUE(capturedDeserialWeightsGradNull);
    TEST_ASSERT_TRUE(capturedSerialBiasGradNull);
    TEST_ASSERT_TRUE(capturedDeserialBiasGradNull);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW, capturedDeserialW, numberOfWeights);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB, capturedDeserialB, numberOfBiases);

    freeReservedMemory(capturedSerialW);
    freeReservedMemory(capturedDeserialW);
    freeReservedMemory(capturedSerialB);
    freeReservedMemory(capturedDeserialB);
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
    float capturedSerialOutputScale = serialOutputCfg->scales[0];
    float capturedDeserialOutputScale = deserialOutputCfg->scales[0];
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
    float capturedSerialOutputScale = serialOutputCfg->scales[0];
    float capturedDeserialOutputScale = deserialOutputCfg->scales[0];
    uint8_t capturedSerialOutputQBits = serialOutputCfg->qBits;
    uint8_t capturedDeserialOutputQBits = deserialOutputCfg->qBits;
    roundingMode_t capturedSerialOutputRM = serialOutputCfg->roundingMode;
    roundingMode_t capturedDeserialOutputRM = deserialOutputCfg->roundingMode;

    qtype_t capturedSerialPropLossQType = serialCfg->propLossQ->type;
    qtype_t capturedDeserialPropLossQType = deserialCfg->propLossQ->type;
    asymQConfig_t *serialPropLossCfg = serialCfg->propLossQ->qConfig;
    asymQConfig_t *deserialPropLossCfg = deserialCfg->propLossQ->qConfig;
    float capturedSerialPropLossScale = serialPropLossCfg->scales[0];
    float capturedDeserialPropLossScale = deserialPropLossCfg->scales[0];
    uint8_t capturedSerialPropLossQBits = serialPropLossCfg->qBits;
    uint8_t capturedDeserialPropLossQBits = deserialPropLossCfg->qBits;
    roundingMode_t capturedSerialPropLossRM = serialPropLossCfg->roundingMode;
    roundingMode_t capturedDeserialPropLossRM = deserialPropLossCfg->roundingMode;
    uint16_t capturedSerialPropLossZP = serialPropLossCfg->zeroPoints[0];
    uint16_t capturedDeserialPropLossZP = deserialPropLossCfg->zeroPoints[0];

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

/*! Model-level round trip (group-quant PR1): a Linear layer carries PACKED
 *  SYM-native weights (qBits=4, sub-byte) through a full
 *  serializeModel/deserializeModel cycle -- the public Linear factories
 *  cannot build this fixture (requireFloat32 gate on weight storage, #270),
 *  so weights/bias are hand-built and wired via BorrowedLayer.h's
 *  buildBorrowedLinearLayer (test-only, mirrors UnitTestSgd.c's packed-grad
 *  fixtures). Exercises the v4 SYM qConfig record (numGroups/groupSize/
 *  scales[]) inside a real parameter record, not just a bare tensor. */
static void testRoundTripLinearSymPackedWeights(void) {
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = 2;
    wDims[1] = 3;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *serialWeightsParam = initTensor(wShape, quantizationInitSym(4, HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(serialWeightsParam, (float[]){-1.f, -0.5f, 0.f, 0.25f, 0.5f, 0.75f},
                              6);
    parameter_t *serialWeights = parameterInit(serialWeightsParam, NULL);

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = 2;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *serialBiasParam = initTensor(bShape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(serialBiasParam, (float[]){0.1f, -0.2f}, 2);
    parameter_t *serialBias = parameterInit(serialBiasParam, NULL);

    quantization_t *floatQ = quantizationInitFloat();
    layer_t *serialLayer = buildBorrowedLinearLayer(serialWeights, serialBias, floatQ);
    layer_t *serialModel[] = {serialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    tensor_t *deserialWeightsParam = initTensor(getShapeLike(serialWeightsParam->shape),
                                                quantizationInitSym(4, HALF_AWAY), NULL);
    parameter_t *deserialWeights = parameterInit(deserialWeightsParam, NULL);
    tensor_t *deserialBiasParam =
        initTensor(getShapeLike(serialBiasParam->shape), quantizationInitFloat(), NULL);
    parameter_t *deserialBias = parameterInit(deserialBiasParam, NULL);
    layer_t *deserialLayer = buildBorrowedLinearLayer(deserialWeights, deserialBias, floatQ);
    layer_t *deserialModel[] = {deserialLayer};

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    /* CAPTURE before frees. */
    uint8_t serialWeightBytes[3]; /* qBits=4, N=6 -> 24 bits = exactly 3 packed bytes */
    uint8_t deserialWeightBytes[3];
    memcpy(serialWeightBytes, serialWeightsParam->data, 3);
    memcpy(deserialWeightBytes, deserialWeightsParam->data, 3);
    symQConfig_t *serialWeightsQC = serialWeightsParam->quantization->qConfig;
    symQConfig_t *deserialWeightsQC = deserialWeightsParam->quantization->qConfig;
    float capturedSerialScale = serialWeightsQC->scales[0];
    float capturedDeserialScale = deserialWeightsQC->scales[0];
    size_t capturedSerialNumGroups = serialWeightsQC->numGroups;
    size_t capturedDeserialNumGroups = deserialWeightsQC->numGroups;
    size_t capturedSerialGroupSize = serialWeightsQC->groupSize;
    size_t capturedDeserialGroupSize = deserialWeightsQC->groupSize;
    float capturedSerialBias[2];
    float capturedDeserialBias[2];
    for (size_t i = 0; i < 2; i++) {
        capturedSerialBias[i] = ((float *)serialBiasParam->data)[i];
        capturedDeserialBias[i] = ((float *)deserialBiasParam->data)[i];
    }

    /* FREE in reverse-init order (BorrowedLayer.h contract: shell-only free,
     * parameters freed explicitly). */
    freeParameter(deserialBias);
    freeParameter(deserialWeights);
    freeLinearLayerShellOnly(deserialLayer);
    freeParameter(serialBias);
    freeParameter(serialWeights);
    freeLinearLayerShellOnly(serialLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT8_ARRAY(serialWeightBytes, deserialWeightBytes, 3);
    TEST_ASSERT_EQUAL_FLOAT(capturedSerialScale, capturedDeserialScale);
    TEST_ASSERT_EQUAL_size_t(1, capturedSerialNumGroups);
    TEST_ASSERT_EQUAL_size_t(capturedSerialNumGroups, capturedDeserialNumGroups);
    TEST_ASSERT_EQUAL_size_t(0, capturedSerialGroupSize);
    TEST_ASSERT_EQUAL_size_t(capturedSerialGroupSize, capturedDeserialGroupSize);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBias, capturedDeserialBias, 2);
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
    float srcScale = ((symQConfig_t *)src->quantization->qConfig)->scales[0];
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
    float dstScale = ((symQConfig_t *)dst->quantization->qConfig)->scales[0];
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

/*! GOLDEN BYTES (#370, wire format v2): serializeTensor's exact encoding,
 *  pinned byte-for-byte against a hand-computed fixture. Shape fields are u32
 *  little-endian regardless of the host's size_t width and byte order — any
 *  drift back to host-native widths changes these bytes and fails here. */
static void testGoldenBytesTensorFloat32V2(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(src, (float[]){1.0f, -2.0f, 0.5f, -0.25f, 3.0f, -1.5f}, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);
    freeTensor(src);

    static const uint8_t expected[] = {/* numberOfDimensions u32 LE */ 0x02,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* dimensions[2] u32 LE */ 0x02,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x03,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* orderOfDimensions[2] u32 LE */ 0x00,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* qtype FLOAT32 */ 0x01,
                                       /* data payload: 6x f32 LE */ 0x00,
                                       0x00,
                                       0x80,
                                       0x3F,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0xC0,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x3F,
                                       0x00,
                                       0x00,
                                       0x80,
                                       0xBE,
                                       0x00,
                                       0x00,
                                       0x40,
                                       0x40,
                                       0x00,
                                       0x00,
                                       0xC0,
                                       0xBF};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (#370/#380, wire format v5): full-model header (magic +
 *  version 5 + layerCount, all u32 LE) plus a RELU record whose
 *  outputQ/propLossQ pin the SYM_INT32 and ASYM qConfig payload encodings.
 *  SYM_INT32 is untouched by group-quant. ASYM (group-quant PR4, Task 4) now
 *  carries the full v5 record: `u32 numGroups`, `u32 groupSize`,
 *  `f32 scales[numGroups]`, `u16 zeroPoints[numGroups]` (LE), THEN the
 *  pre-existing `u8 qBits`/`u8 rounding` tail — replacing Task 1's v4 bridge
 *  (f32 scale, u8 qBits, u8 rounding, i32 zeroPoint). This is the minimal
 *  numGroups=1 case; see testGoldenBytesModelReluAsymGroupedPropLossV5 below
 *  for the numGroups>1 golden. RELU carries no parameters, so the
 *  v3-introduced grad-presence byte does not appear in this record (see
 *  testGoldenBytesModelLinearFrozenV5 for that). */
static void testGoldenBytesModelReluV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symIntOutputQ = quantizationInitSymInt32WithBits(SR_HALF_AWAY, 12);
    quantization_t *asymPropLossQ = quantizationInitAsym(8, HALF_AWAY);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symIntOutputQ;
    lq.propLossQ = asymPropLossQ;

    layer_t *layer = reluLayerInitOwning(&lq);
    reluConfig_t *cfg = layer->config->relu;
    symInt32QConfig_t *outputCfg = cfg->outputQ->qConfig;
    outputCfg->scale = 0.5f;
    asymQConfig_t *propLossCfg = cfg->propLossQ->qConfig;
    /* code-domain re-pin (group-quant PR4, D6): the in-memory zp is a
     * uint16 CODE-domain value; +3 replaces the old value-domain -3. */
    propLossCfg->scales[0] = 0.25f;
    propLossCfg->zeroPoints[0] = 3;

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeReluLayer(layer);
    freeQuantization(asymPropLossQ);
    freeQuantization(symIntOutputQ);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {
        /* magic */ 'O', 'D', 'T', 'S',
        /* version u32 LE */ 0x05, 0x00, 0x00, 0x00,
        /* layerCount u32 LE */ 0x01, 0x00, 0x00, 0x00,
        /* tag RELU */ 0x01,
        /* forwardMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
        /* propLossMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
        /* outputQ: SYM_INT32, scale 0.5f f32 LE, SR_HALF_AWAY, qMaxBits 12 */
        0x02, 0x00, 0x00, 0x00, 0x3F, 0x01, 0x0C,
        /* propLossQ: ASYM, numGroups 1 u32 LE, groupSize 0 u32 LE, scales[0]
         * 0.25f f32 LE, zeroPoints[0] 3 u16 LE, qBits 8, HALF_AWAY */
        0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3E, 0x03, 0x00,
        0x08, 0x00};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (group-quant PR4, Task 4, wire format v5): the numGroups>1
 *  sibling of testGoldenBytesModelReluV5 above -- pins that a GROUPED ASYM
 *  record (numGroups=2, groupSize=3) serializes with every group's scale
 *  AND zeroPoint in order (scales[] array fully, THEN zeroPoints[] array
 *  fully -- not interleaved), not just element 0. Mirrors
 *  testGoldenBytesModelReluSymGroupedOutputV5's role for SYM. zeroPoints are
 *  chosen non-trivial and byte-order-sensitive (3 and 0x1234) so an
 *  endianness or field-order bug in the u16 write is caught byte-exact. */
static void testGoldenBytesModelReluAsymGroupedPropLossV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *asymGroupedPropLossQ = quantizationInitAsymGrouped(8, HALF_AWAY, 2, 3);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossQ = asymGroupedPropLossQ;

    layer_t *layer = reluLayerInitOwning(&lq);
    reluConfig_t *cfg = layer->config->relu;
    asymQConfig_t *propLossCfg = cfg->propLossQ->qConfig;
    propLossCfg->scales[0] = 0.5f;
    propLossCfg->scales[1] = 0.25f;
    propLossCfg->zeroPoints[0] = 3;
    propLossCfg->zeroPoints[1] = 0x1234;

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeReluLayer(layer);
    freeQuantization(asymGroupedPropLossQ);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {/* magic */ 'O', 'D', 'T', 'S',
                                       /* version u32 LE */ 0x05, 0x00, 0x00, 0x00,
                                       /* layerCount u32 LE */ 0x01, 0x00, 0x00, 0x00,
                                       /* tag RELU */ 0x01,
                                       /* forwardMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* propLossMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* outputQ FLOAT32 */ 0x01,
                                       /* propLossQ: ASYM, numGroups 2 u32 LE, groupSize 3 u32 LE,
                                        * scales[0] 0.5f / scales[1] 0.25f f32 LE,
                                        * zeroPoints[0] 3 / zeroPoints[1] 0x1234 u16 LE,
                                        * qBits 8, HALF_AWAY */
                                       0x04, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x3F, 0x00, 0x00, 0x80, 0x3E, 0x03, 0x00, 0x34,
                                       0x12, 0x08, 0x00};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (group-quant PR1, wire format v5 -- record layout unchanged
 *  since v4, only the header version byte bumped by Task 4's ASYM re-layout,
 *  spec docs/superpowers/specs/2026-07-28-group-quantization-design.md §6):
 *  the SYM qConfig record itself — `u32 numGroups`, `u32 groupSize`, then
 *  `f32 scales[numGroups]`, THEN the pre-existing `u8 qBits`/`u8 rounding`
 *  tail (unchanged order/width from v3). PR1 is always numGroups=1,
 *  groupSize=0 (the "whole tensor" sentinel — Quantization.h), so this pins
 *  the minimal one-scale case; see testGoldenBytesModelReluSymGroupedOutputV5
 *  below for the numGroups>1 golden (group-quant PR2, Task 5). propLossQ
 *  (FLOAT32, a single 0x01 tag byte) sits immediately after the SYM record,
 *  so its offset is the drift alarm for the record's new +8-byte width
 *  (numGroups u32 + groupSize u32) versus the old v3 layout.
 *  Mutation guard: swapping the numGroups/groupSize write order in
 *  Serialize.c's SYM arm flips bytes 18-25 below and fails this pin. */
static void testGoldenBytesModelReluSymOutputV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;

    layer_t *layer = reluLayerInitOwning(&lq);
    reluConfig_t *cfg = layer->config->relu;
    symQConfig_t *outputCfg = cfg->outputQ->qConfig;
    outputCfg->scales[0] = 0.5f;

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {/* magic */ 'O', 'D', 'T', 'S',
                                       /* version u32 LE */ 0x05, 0x00, 0x00, 0x00,
                                       /* layerCount u32 LE */ 0x01, 0x00, 0x00, 0x00,
                                       /* tag RELU */ 0x01,
                                       /* forwardMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* propLossMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* outputQ: SYM, numGroups 1 u32 LE, groupSize 0 u32 LE,
                                        * scales[0] 0.5f f32 LE, qBits 6, HALF_AWAY */
                                       0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x3F, 0x06, 0x00,
                                       /* propLossQ FLOAT32 */ 0x01};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (group-quant PR2, Task 5, wire format v5 -- record layout
 *  unchanged since v4): the numGroups>1 sibling of
 *  testGoldenBytesModelReluSymOutputV5 above -- pins that a GROUPED SYM
 *  record (numGroups=2, groupSize=3) serializes with every group's scale in
 *  order, not just scales[0]. Serialize.c's SYM arm needed no changes for
 *  this (group-general since PR1 — it already loops `symQC->numGroups`
 *  writes); this golden is the writer-side regression net Task 5 owes per
 *  the comment above. */
static void testGoldenBytesModelReluSymGroupedOutputV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symGroupedOutputQ = quantizationInitSymGrouped(6, HALF_AWAY, 2, 3);

    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symGroupedOutputQ;

    layer_t *layer = reluLayerInitOwning(&lq);
    reluConfig_t *cfg = layer->config->relu;
    symQConfig_t *outputCfg = cfg->outputQ->qConfig;
    outputCfg->scales[0] = 0.5f;
    outputCfg->scales[1] = 0.25f;

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeReluLayer(layer);
    freeQuantization(symGroupedOutputQ);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {/* magic */ 'O', 'D', 'T', 'S',
                                       /* version u32 LE */ 0x05, 0x00, 0x00, 0x00,
                                       /* layerCount u32 LE */ 0x01, 0x00, 0x00, 0x00,
                                       /* tag RELU */ 0x01,
                                       /* forwardMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* propLossMath: ARITH_FLOAT32, HALF_AWAY */ 0x00, 0x00,
                                       /* outputQ: SYM, numGroups 2 u32 LE, groupSize 3 u32 LE,
                                        * scales[0] 0.5f / scales[1] 0.25f f32 LE, qBits 6,
                                        * HALF_AWAY */
                                       0x03, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x3F, 0x00, 0x00, 0x80, 0x3E, 0x06, 0x00,
                                       /* propLossQ FLOAT32 */ 0x01};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (#370/#380, wire format v5): MAXPOOL1D record pinning the
 *  kernel geometry encoding (size/stride/dilation/padding as u32 LE +
 *  paddingType u8) — the fields that were raw host size_t in v1. MAXPOOL1D
 *  carries no parameters, so (like RELU above) no grad-presence byte appears
 *  in this record. FLOAT32-only (no SYM/ASYM qConfig), so only the header
 *  version byte changes for Task 4's v5 bump. */
static void testGoldenBytesModelMaxPool1dV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    maxPool1dInit_t init = {.kernelSize = 3, .inputChannels = 4, .inputLength = 10, .stride = 2};
    layer_t *layer = maxPool1dLayerInitOwning(&init, &lq);

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeMaxPool1dLayer(layer);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {/* magic + version 5 + layerCount 1 */ 'O',
                                       'D',
                                       'T',
                                       'S',
                                       0x05,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* tag MAXPOOL1D */ 0x04,
                                       /* kernel size 3 u32 LE */ 0x03,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* paddingType VALID */ 0x00,
                                       /* stride 2 u32 LE */ 0x02,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* dilation 1 u32 LE */ 0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* padding 0 u32 LE */ 0x00,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* forwardMath + propLossMath */ 0x00,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* outputQ FLOAT32, propLossQ FLOAT32 */ 0x01,
                                       0x01};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (#380, wire format v5): a FROZEN Linear layer's weight AND
 *  bias parameter records each lead with a hasGrad=0x00 presence byte
 *  followed directly by the param tensor — no grad tensor on the wire at
 *  all. Pins the exact byte layout the v3-introduced grad-presence byte
 *  introduces (the counterpart to the trainable case, which round-trips a
 *  hasGrad=0x01 byte followed by param THEN grad in testRoundTripLinear).
 *  Both weight and bias here are FLOAT32, so neither the group-quant v4 SYM
 *  bump nor Task 4's v5 ASYM bump touches this record's bytes beyond the
 *  header version. Values are overwritten post-construction (random Kaiming
 *  init is not pin-stable). */
static void testGoldenBytesModelLinearFrozenV5(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    linearInit_t init = {.inFeatures = 1, .outFeatures = 1, .trainable = TRAINABLE_FALSE};
    layer_t *layer = linearLayerInitOwning(&init, &lq);
    linearConfig_t *cfg = layer->config->linear;

    tensorFillFromFloatBuffer(cfg->weights->param, (float[]){2.0f}, 1);
    tensorFillFromFloatBuffer(cfg->bias->param, (float[]){-1.0f}, 1);

    layer_t *model[] = {layer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(model, 1, f);
    fclose(f);
    freeLinearLayer(layer);
    freeQuantization(floatQ);

    static const uint8_t expected[] = {/* magic */ 'O',
                                       'D',
                                       'T',
                                       'S',
                                       /* version u32 LE */ 0x05,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* layerCount u32 LE */ 0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* tag LINEAR */ 0x00,
                                       /* weights: hasGrad=0 (v3 grad-presence byte) */ 0x00,
                                       /* weights param shape: rank 2 u32 LE */ 0x02,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* dims [1,1] u32 LE */ 0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* orderOfDimensions [0,1] u32 LE */ 0x00,
                                       0x00,
                                       0x00,
                                       0x00,
                                       0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* qtype FLOAT32 */ 0x01,
                                       /* data payload: 1x f32 LE = 2.0f */ 0x00,
                                       0x00,
                                       0x00,
                                       0x40,
                                       /* bias: hasGrad=0 (v3 grad-presence byte) */ 0x00,
                                       /* bias param shape: rank 1 u32 LE */ 0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* dims [1] u32 LE */ 0x01,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* orderOfDimensions [0] u32 LE */ 0x00,
                                       0x00,
                                       0x00,
                                       0x00,
                                       /* qtype FLOAT32 */ 0x01,
                                       /* data payload: 1x f32 LE = -1.0f */ 0x00,
                                       0x00,
                                       0x80,
                                       0xBF,
                                       /* forwardMath: ARITH_FLOAT32, HALF_AWAY */ 0x00,
                                       0x00,
                                       /* weightGradMath: ARITH_FLOAT32, HALF_AWAY */ 0x00,
                                       0x00,
                                       /* biasGradMath: ARITH_FLOAT32, HALF_AWAY */ 0x00,
                                       0x00,
                                       /* propLossMath: ARITH_FLOAT32, HALF_AWAY */ 0x00,
                                       0x00,
                                       /* outputQ FLOAT32, propLossQ FLOAT32 */ 0x01,
                                       0x01};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! GOLDEN BYTES (BFP epic PR1, Task 7, wire format v5, spec
 *  docs/superpowers/specs/2026-07-29-block-floating-point-design.md §6): the
 *  BFP qConfig record -- `u32 numGroups`, `u32 groupSize`, `u8
 *  exponents[numGroups]`, then `u8 mantissaBits`, `u8 exponentBits`, `u8
 *  roundingMode` -- followed by the packed mantissa payload
 *  (calcNumberOfBytesForData, mirroring every other dtype). Shape {2,4}
 *  grouped per-row (numGroups=2, groupSize=4): the packed nibbles are
 *  hand-computed from mantissa codes {1,-1,2,-2,3,-3,4,-4} (4-bit two's
 *  complement, low nibble = even-indexed element, high nibble = odd-indexed
 *  element, byte-for-byte per packChunkGuarded/byteConversion's bit order)
 *  and written directly into the tensor's data buffer -- no conversion
 *  routine involved, isolating this test from the separate quantize/pack-path
 *  coverage (epic PR1 Tasks 2-4). Exponents are set to arbitrary non-bias
 *  values (init's zero-state is the bias, 127 for exponentBits=8) to pin
 *  that the writer emits the CURRENT config, not a derived one.
 *  Mutation guard: swapping the mantissaBits/exponentBits write order in
 *  Serialize.c's BFP arm flips bytes 31-32 below and fails this pin. */
static void testGoldenBytesBfpQConfigRecordV5(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitBfpGrouped(4, 8, HALF_AWAY, 2, 4), NULL);

    bfpQConfig_t *qc = src->quantization->qConfig;
    qc->exponents[0] = 5;
    qc->exponents[1] = 250;
    uint8_t payload[] = {0xF1, 0xE2, 0xD3, 0xC4}; /* codes {1,-1,2,-2,3,-3,4,-4}, 4-bit packed */
    memcpy(src->data, payload, sizeof(payload));

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);
    freeTensor(src);

    static const uint8_t expected[] = {
        /* numberOfDimensions u32 LE */ 0x02, 0x00, 0x00, 0x00,
        /* dimensions[2] u32 LE */ 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
        /* orderOfDimensions[2] u32 LE */ 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        /* qtype BFP */ 0x06,
        /* numGroups 2 u32 LE */ 0x02, 0x00, 0x00, 0x00,
        /* groupSize 4 u32 LE */ 0x04, 0x00, 0x00, 0x00,
        /* exponents[2] */ 0x05, 0xFA,
        /* mantissaBits 4, exponentBits 8, roundingMode HALF_AWAY */ 0x04, 0x08, 0x00,
        /* payload: packed 4-bit mantissas, codes {1,-1,2,-2,3,-3,4,-4} */
        0xF1, 0xE2, 0xD3, 0xC4};

    uint8_t got[sizeof(expected) + 8] = {0};
    f = fopen(FILE_PATH, "rb");
    size_t fileBytes = fread(got, 1, sizeof(got), f);
    fclose(f);

    TEST_ASSERT_EQUAL_size_t(sizeof(expected), fileBytes);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(expected, got, sizeof(expected));
}

/*! #370: every fwrite is length-checked — a stream that accepts no bytes (here:
 *  a read-only FILE*) must fail fast instead of silently producing a partial
 *  file. */
static void testSerializeFailsFastOnUnwritableStream(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitFloat(), NULL);

    FILE *f = fopen(FILE_PATH, "wb");
    fclose(f);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(serializeTensor(src, f));
    fclose(f);

    freeTensor(src);
}

#if SIZE_MAX > UINT32_MAX
/*! #370: the wire pins counts/dims to u32 — a size_t dimension that cannot fit
 *  must fail fast on serialize instead of silently truncating on the wire. */
static void testSerializeFailsFastOnDimensionBeyondU32(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *src = initTensor(shape, quantizationInitFloat(), NULL);

    /* Lie about the first dimension AFTER allocation; the guard fires in
     * serializeShape, before any data-payload write could read past the real
     * 2x3 allocation. */
    src->shape->dimensions[0] = ((size_t)UINT32_MAX) + 2u;

    FILE *f = fopen(FILE_PATH, "wb");
    ASSERT_EXITS_WITH_FAILURE(serializeTensor(src, f));
    fclose(f);

    src->shape->dimensions[0] = 2;
    freeTensor(src);
}
#endif

/*! #370: LayerNorm's numNormDims drives the normalizedShape read count; a
 *  mismatched record would write file entries past the skeleton's array (and
 *  pre-v2 did so silently — both fixtures carry 20 gamma/beta elements, so the
 *  tensor payload checks alone cannot see it). */
static void testDeserializeLayerNormRejectsNumNormDimsMismatch(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    size_t serialShapeValues[] = {4, 5};
    layerNormInit_t serialInit = {
        .normalizedShape = serialShapeValues, .numNormDims = 2, .eps = 1e-3f};
    layer_t *serialLayer = layerNormLayerInitOwning(&serialInit, &lq);

    size_t skeletonShapeValues[] = {20};
    layerNormInit_t skeletonInit = {
        .normalizedShape = skeletonShapeValues, .numNormDims = 1, .eps = 1e-3f};
    layer_t *skeletonLayer = layerNormLayerInitOwning(&skeletonInit, &lq);

    layer_t *serialModel[] = {serialLayer};
    layer_t *skeletonModel[] = {skeletonLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(skeletonModel, 1, f));
    fclose(f);

    freeLayerNormLayer(skeletonLayer);
    freeLayerNormLayer(serialLayer);
    freeQuantization(floatQ);
}

/*! #380 PR3: relaxes the old PR1 STRICT mismatch contract pinned above
 *  (Leo approved 2026-07-22) — a TRAINABLE-serialized model's grad-presence
 *  byte (1) deserialized into a FROZEN-built skeleton (parameter->grad ==
 *  NULL) must now be TOLERATED: the grad record is parsed and discarded
 *  (skipSerializedTensor), not treated as file corruption. Two stacked
 *  Linear layers with DIFFERENT in/out feature counts (so a stream desync
 *  produces garbage rather than a coincidental pass): layer 1 deserializes
 *  into a frozen skeleton (its weights+bias grad records get skipped, twice
 *  in a row), layer 2 into a trainable one — layer 2's param AND grad
 *  values parsing correctly right after those two skips is the stream-sync
 *  proof. */
static void testDeserializeSkipsGradIntoFrozenSkeleton(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    linearInit_t init1 = {.inFeatures = 4, .outFeatures = 3};
    linearInit_t init2 = {.inFeatures = 3, .outFeatures = 5};

    layer_t *serialLayer1 = linearLayerInitOwning(&init1, &lq);
    layer_t *serialLayer2 = linearLayerInitOwning(&init2, &lq);

    linearConfig_t *serialCfg1 = serialLayer1->config->linear;
    linearConfig_t *serialCfg2 = serialLayer2->config->linear;

    size_t numWeights1 = calcNumberOfElementsByTensor(serialCfg1->weights->param);
    size_t numBiases1 = calcNumberOfElementsByTensor(serialCfg1->bias->param);
    size_t numWeights2 = calcNumberOfElementsByTensor(serialCfg2->weights->param);
    size_t numBiases2 = calcNumberOfElementsByTensor(serialCfg2->bias->param);

    /* Seed layer2's grads non-zero so its round trip is distinguishable from
     * a no-op. Layer1's grads are irrelevant on the serial side: its
     * skeleton is frozen, so they never get read regardless of their value. */
    float *weightGradSeed2 = reserveMemory(numWeights2 * sizeof(float));
    for (size_t i = 0; i < numWeights2; i++) {
        weightGradSeed2[i] = (float)i + 0.5f;
    }
    tensorFillFromFloatBuffer(serialCfg2->weights->grad, weightGradSeed2, numWeights2);

    float *biasGradSeed2 = reserveMemory(numBiases2 * sizeof(float));
    for (size_t i = 0; i < numBiases2; i++) {
        biasGradSeed2[i] = (float)i - 1.5f;
    }
    tensorFillFromFloatBuffer(serialCfg2->bias->grad, biasGradSeed2, numBiases2);

    layer_t *serialModel[] = {serialLayer1, serialLayer2};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 2, f);
    fclose(f);

    linearInit_t frozenInit1 = init1;
    frozenInit1.trainable = TRAINABLE_FALSE;
    layer_t *skeletonLayer1 = linearLayerInitOwning(&frozenInit1, &lq);
    layer_t *skeletonLayer2 = linearLayerInitOwning(&init2, &lq);
    layer_t *skeletonModel[] = {skeletonLayer1, skeletonLayer2};

    f = fopen(FILE_PATH, "rb");
    deserializeModel(skeletonModel, 2, f);
    fclose(f);

    linearConfig_t *skeletonCfg1 = skeletonLayer1->config->linear;
    linearConfig_t *skeletonCfg2 = skeletonLayer2->config->linear;

    /* CAPTURE every assertion value before any free. */
    bool capturedSkeletonWeights1GradNull = (skeletonCfg1->weights->grad == NULL);
    bool capturedSkeletonBias1GradNull = (skeletonCfg1->bias->grad == NULL);

    float *capturedSerialW1 = reserveMemory(numWeights1 * sizeof(float));
    float *capturedSkeletonW1 = reserveMemory(numWeights1 * sizeof(float));
    for (size_t i = 0; i < numWeights1; i++) {
        capturedSerialW1[i] = ((float *)serialCfg1->weights->param->data)[i];
        capturedSkeletonW1[i] = ((float *)skeletonCfg1->weights->param->data)[i];
    }
    float *capturedSerialB1 = reserveMemory(numBiases1 * sizeof(float));
    float *capturedSkeletonB1 = reserveMemory(numBiases1 * sizeof(float));
    for (size_t i = 0; i < numBiases1; i++) {
        capturedSerialB1[i] = ((float *)serialCfg1->bias->param->data)[i];
        capturedSkeletonB1[i] = ((float *)skeletonCfg1->bias->param->data)[i];
    }

    float *capturedSerialW2 = reserveMemory(numWeights2 * sizeof(float));
    float *capturedSkeletonW2 = reserveMemory(numWeights2 * sizeof(float));
    float *capturedSerialWGrad2 = reserveMemory(numWeights2 * sizeof(float));
    float *capturedSkeletonWGrad2 = reserveMemory(numWeights2 * sizeof(float));
    for (size_t i = 0; i < numWeights2; i++) {
        capturedSerialW2[i] = ((float *)serialCfg2->weights->param->data)[i];
        capturedSkeletonW2[i] = ((float *)skeletonCfg2->weights->param->data)[i];
        capturedSerialWGrad2[i] = ((float *)serialCfg2->weights->grad->data)[i];
        capturedSkeletonWGrad2[i] = ((float *)skeletonCfg2->weights->grad->data)[i];
    }
    float *capturedSerialB2 = reserveMemory(numBiases2 * sizeof(float));
    float *capturedSkeletonB2 = reserveMemory(numBiases2 * sizeof(float));
    float *capturedSerialBGrad2 = reserveMemory(numBiases2 * sizeof(float));
    float *capturedSkeletonBGrad2 = reserveMemory(numBiases2 * sizeof(float));
    for (size_t i = 0; i < numBiases2; i++) {
        capturedSerialB2[i] = ((float *)serialCfg2->bias->param->data)[i];
        capturedSkeletonB2[i] = ((float *)skeletonCfg2->bias->param->data)[i];
        capturedSerialBGrad2[i] = ((float *)serialCfg2->bias->grad->data)[i];
        capturedSkeletonBGrad2[i] = ((float *)skeletonCfg2->bias->grad->data)[i];
    }

    /* FREE in reverse-init order. */
    freeReservedMemory(biasGradSeed2);
    freeReservedMemory(weightGradSeed2);
    freeLinearLayer(skeletonLayer2);
    freeLinearLayer(skeletonLayer1);
    freeLinearLayer(serialLayer2);
    freeLinearLayer(serialLayer1);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(capturedSkeletonWeights1GradNull);
    TEST_ASSERT_TRUE(capturedSkeletonBias1GradNull);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW1, capturedSkeletonW1, numWeights1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB1, capturedSkeletonB1, numBiases1);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialW2, capturedSkeletonW2, numWeights2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialWGrad2, capturedSkeletonWGrad2, numWeights2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialB2, capturedSkeletonB2, numBiases2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBGrad2, capturedSkeletonBGrad2, numBiases2);

    freeReservedMemory(capturedSerialW1);
    freeReservedMemory(capturedSkeletonW1);
    freeReservedMemory(capturedSerialB1);
    freeReservedMemory(capturedSkeletonB1);
    freeReservedMemory(capturedSerialW2);
    freeReservedMemory(capturedSkeletonW2);
    freeReservedMemory(capturedSerialWGrad2);
    freeReservedMemory(capturedSkeletonWGrad2);
    freeReservedMemory(capturedSerialB2);
    freeReservedMemory(capturedSkeletonB2);
    freeReservedMemory(capturedSerialBGrad2);
    freeReservedMemory(capturedSkeletonBGrad2);
}

/*! group-quant PR1 mutation guard: skipSerializedTensor's SYM branch must
 *  parse the v4 record (numGroups/groupSize before the scales array) or the
 *  stream desyncs after the skip. Mirrors
 *  testDeserializeSkipsGradIntoFrozenSkeleton (#380 PR3) but gives the
 *  trainable serial-side layer a packed SYM weight-grad (instead of the
 *  FLOAT32 default) so the skipped record actually exercises the SYM
 *  qConfig arm. serializeLayer's LINEAR arm writes weights (param THEN
 *  grad) immediately followed by bias -- a byte-count drift in the SYM grad
 *  skip corrupts the bias read that follows, so asserting bias round-trips
 *  correctly proves (or, pre-fix, disproves) the skip stayed in sync. */
static void testDeserializeSkipsSymGradIntoFrozenSkeleton(void) {
    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symGradQ = quantizationInitSym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.weightGradStorage = symGradQ;

    linearInit_t init = {.inFeatures = 4, .outFeatures = 3};
    layer_t *serialLayer = linearLayerInitOwning(&init, &lq);
    linearConfig_t *serialCfg = serialLayer->config->linear;

    size_t numWeights = calcNumberOfElementsByTensor(serialCfg->weights->param);
    float *weightGradSeed = reserveMemory(numWeights * sizeof(float));
    for (size_t i = 0; i < numWeights; i++) {
        weightGradSeed[i] = (float)i - 3.5f;
    }
    tensorFillFromFloatBuffer(serialCfg->weights->grad, weightGradSeed, numWeights);

    size_t numBiases = calcNumberOfElementsByTensor(serialCfg->bias->param);
    float *biasSeed = reserveMemory(numBiases * sizeof(float));
    for (size_t i = 0; i < numBiases; i++) {
        biasSeed[i] = (float)i + 0.25f;
    }
    tensorFillFromFloatBuffer(serialCfg->bias->param, biasSeed, numBiases);

    layer_t *serialModel[] = {serialLayer};
    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    linearInit_t frozenInit = init;
    frozenInit.trainable = TRAINABLE_FALSE;
    layer_t *skeletonLayer = linearLayerInitOwning(&frozenInit, &lq);
    layer_t *skeletonModel[] = {skeletonLayer};

    f = fopen(FILE_PATH, "rb");
    deserializeModel(skeletonModel, 1, f);
    fclose(f);

    linearConfig_t *skeletonCfg = skeletonLayer->config->linear;

    /* CAPTURE before frees. */
    bool capturedSkeletonGradNull = (skeletonCfg->weights->grad == NULL);
    float capturedSerialBias[3];
    float capturedSkeletonBias[3];
    for (size_t i = 0; i < numBiases; i++) {
        capturedSerialBias[i] = ((float *)serialCfg->bias->param->data)[i];
        capturedSkeletonBias[i] = ((float *)skeletonCfg->bias->param->data)[i];
    }

    /* FREE in reverse-init order. */
    freeReservedMemory(biasSeed);
    freeReservedMemory(weightGradSeed);
    freeLinearLayer(skeletonLayer);
    freeLinearLayer(serialLayer);
    freeQuantization(symGradQ);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(capturedSkeletonGradNull);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(capturedSerialBias, capturedSkeletonBias, numBiases);
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
    RUN_TEST(testRoundTripLinearSymPackedWeights);
    RUN_TEST(testSerializeTensorSymSubByteRoundTripsPackedData);
    RUN_TEST(testSerializeTensorBoolRoundTripsPackedData);
    RUN_TEST(testGoldenBytesTensorFloat32V2);
    RUN_TEST(testGoldenBytesModelReluV5);
    RUN_TEST(testGoldenBytesModelReluAsymGroupedPropLossV5);
    RUN_TEST(testGoldenBytesModelReluSymOutputV5);
    RUN_TEST(testGoldenBytesModelReluSymGroupedOutputV5);
    RUN_TEST(testGoldenBytesModelMaxPool1dV5);
    RUN_TEST(testGoldenBytesModelLinearFrozenV5);
    RUN_TEST(testGoldenBytesBfpQConfigRecordV5);
    RUN_TEST(testSerializeFailsFastOnUnwritableStream);
#if SIZE_MAX > UINT32_MAX
    RUN_TEST(testSerializeFailsFastOnDimensionBeyondU32);
#endif
    RUN_TEST(testDeserializeLayerNormRejectsNumNormDimsMismatch);
    RUN_TEST(testDeserializeSkipsGradIntoFrozenSkeleton);
    RUN_TEST(testDeserializeSkipsSymGradIntoFrozenSkeleton);
    RUN_TEST(testRoundTripLinearFrozen);
    return UNITY_END();
}
