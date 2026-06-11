#define SOURCE_FILE "UNIT_TEST_QUANT_LAYER_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "CalculateGradsSequential.h"
#include "InferenceApi.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Deterministic fixtures shared by the micro tests (D.4-D.6) and chain test (D.9). */
static const float W0_VALS[12] = {0.5f,  -0.3f, 0.8f, 0.1f,  -0.7f, 0.4f,
                                  0.2f,  -0.9f, 0.6f, -0.2f, 0.3f,  0.7f};
static const float B0_VALS[3] = {0.1f, -0.2f, 0.3f};
static const float INPUT_VALS[8] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f};

static tensor_t *build2DSym(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

/* Trainable Linear with manually built parameters: the new-factory KAIMING
 * init requires FLOAT32 weight storage (LinearApi.c guard), so SYM Linear
 * layers use the Legacy factory + explicit parameter_t (UnitTestLinear SYM
 * precedent), extended with grad tensors for sgdMCreateOptim / sgdStepM. */
static layer_t *buildTrainableLinear(size_t inF, size_t outF, const float *w, const float *b,
                                     quantization_t *mathQ, bool sym) {
    size_t *wDims = reserveMemory(2 * sizeof(size_t));
    wDims[0] = outF;
    wDims[1] = inF;
    size_t *wOrder = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, wOrder);
    shape_t *wShape = reserveMemory(sizeof(shape_t));
    setShape(wShape, wDims, 2, wOrder);
    tensor_t *wParam = initTensor(
        wShape, sym ? quantizationInitSymInt32(HALF_AWAY) : quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(wParam, (float *)w, outF * inF);
    parameter_t *weights = parameterInit(wParam, gradInit(wParam, mathQ, NULL));

    size_t *bDims = reserveMemory(1 * sizeof(size_t));
    bDims[0] = outF;
    size_t *bOrder = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, bOrder);
    shape_t *bShape = reserveMemory(sizeof(shape_t));
    setShape(bShape, bDims, 1, bOrder);
    tensor_t *bParam = initTensor(
        bShape, sym ? quantizationInitSymInt32(HALF_AWAY) : quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(bParam, (float *)b, outF);
    parameter_t *bias = parameterInit(bParam, gradInit(bParam, mathQ, NULL));

    return linearLayerInitLegacy(weights, bias, mathQ, mathQ, mathQ, mathQ);
}

/* Manual Quant-layer builder: documents the quantizationConfig_t contract
 * without depending on the factory (Task D.7). Borrowed quantizations. */
static layer_t *buildQuantLayer(quantization_t *forwardQ, quantization_t *backwardQ) {
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    cfg->forwardQ = forwardQ;
    cfg->backwardQ = backwardQ;
    cfg->ownsQuantizations = false;
    layerConfig_t *lc = reserveMemory(sizeof(layerConfig_t));
    lc->quantization = cfg;
    layer_t *layer = reserveMemory(sizeof(layer_t));
    initLayer(layer, QUANTIZATION, lc);
    return layer;
}

static void freeQuantLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->quantization);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

/* freeOptimSgdM cascades into freeParameter for every registered parameter_t —
 * the SAME objects the layers reference; layers are torn down shell-only
 * afterwards (UnitTestLayerNormIntegration pattern). */
static void freeLinearLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->linear);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testSgdMCreateOptimSkipsQuantizationLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    optimizer_t *optim = sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, SYM_INT32);
    size_t sizeStates = optim->sizeStates;

    freeOptimSgdM(optim); /* frees the Linear weights/bias parameter_t */
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_EQUAL_UINT(2, sizeStates); /* Linear w+b; QUANTIZATION contributes 0 states */
}

void testTrainingStepLinearSymThenQuantRequantsOutputAndFiniteLoss(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    tensor_t *input = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *label = build2DSym(2, 3, (float[]){0.5f, -1.f, 2.f, 1.5f, -0.5f, 1.f}, 6);

    trainingStats_t *stats = calculateGradsSequential(model, 2, defaultLossConfig(MSE),
                                                      REDUCTION_MEAN, input, label);
    bool statsNotNull = (stats != NULL);
    float loss = stats ? stats->loss : NAN;

    int32_t maxAbsMantissa = 0;
    for (size_t i = 0; statsNotNull && i < 6; i++) {
        int32_t m = ((int32_t *)stats->output->data)[i];
        if (m < 0) {
            m = -m;
        }
        if (m > maxAbsMantissa) {
            maxAbsMantissa = m;
        }
    }

    freeTrainingStats(stats);
    freeTensor(label);
    freeTensor(input);
    freeParameter(linear->config->linear->weights); /* no optimizer in this test */
    freeParameter(linear->config->linear->bias);
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(statsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(loss), "loss must be finite after the training step");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa > 0, "quant output must be non-trivial");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa <= 32767,
                             "quant layer output must obey the int16 inter-layer contract");
}

void testInferenceLinearSymThenQuantOutputsInt16RangeMantissas(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    tensor_t *input = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *predicted = inference(model, 2, input);

    bool predictedNotNull = (predicted != NULL);
    bool dequantFinite = true;
    int32_t maxAbsMantissa = 0;
    float scale =
        predictedNotNull ? ((symInt32QConfig_t *)predicted->quantization->qConfig)->scale : NAN;
    for (size_t i = 0; predictedNotNull && i < 6; i++) {
        int32_t m = ((int32_t *)predicted->data)[i];
        dequantFinite = dequantFinite && isfinite((float)m * scale);
        if (m < 0) {
            m = -m;
        }
        if (m > maxAbsMantissa) {
            maxAbsMantissa = m;
        }
    }

    freeTensor(predicted);
    freeTensor(input);
    freeParameter(linear->config->linear->weights);
    freeParameter(linear->config->linear->bias);
    freeQuantLayerShell(quant);
    freeLinearLayerShell(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(predictedNotNull);
    TEST_ASSERT_TRUE_MESSAGE(dequantFinite, "dequantized inference output must be finite");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa > 0, "inference output must be non-trivial");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa <= 32767,
                             "quant layer output must obey the int16 inter-layer contract");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSgdMCreateOptimSkipsQuantizationLayer);
    RUN_TEST(testTrainingStepLinearSymThenQuantRequantsOutputAndFiniteLoss);
    RUN_TEST(testInferenceLinearSymThenQuantOutputsInt16RangeMantissas);
    return UNITY_END();
}
