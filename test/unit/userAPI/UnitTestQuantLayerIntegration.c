#define SOURCE_FILE "UNIT_TEST_QUANT_LAYER_INTEGRATION"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "BorrowedLayer.h"
#include "CalculateGradsSequential.h"
#include "Conv1dTransposed.h"
#include "DeathTest.h"
#include "InferenceApi.h"
#include "Kernel.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "ModelValidationApi.h"
#include "Optimizer.h"
#include "OptimizerApi.h"
#include "QuantLayerApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "QuantizationLayer.h"
#include "SgdApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Deterministic fixtures shared by the micro tests (D.4-D.6) and chain test (D.9). */
static const float W0_VALS[12] = {0.5f, -0.3f, 0.8f, 0.1f,  -0.7f, 0.4f,
                                  0.2f, -0.9f, 0.6f, -0.2f, 0.3f,  0.7f};
static const float B0_VALS[3] = {0.1f, -0.2f, 0.3f};
static const float INPUT_VALS[8] = {1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f};

static tensor_t *build2DFloat(size_t b, size_t f, const float *data, size_t n) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = b;
    dims[1] = f;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

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

/* 3D [B,C,L] SYM int12 tensor (operands), mirroring build2DSym. */
static tensor_t *build3DSymI12(size_t b, size_t c, size_t l, const float *data, size_t n) {
    size_t *dims = reserveMemory(3 * sizeof(size_t));
    dims[0] = b;
    dims[1] = c;
    dims[2] = l;
    size_t *order = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 3, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    tensorFillFromFloatBuffer(t, (float *)data, n);
    return t;
}

/* A trainable SYM ConvT param: int12 operand storage, SYM grad accumulator. */
static parameter_t *buildConvTSymParam(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *p = initTensor(shape, quantizationInitSymInt32WithBits(HALF_AWAY, 12), NULL);
    tensorFillFromFloatBuffer(p, (float *)vals, calcNumberOfElementsByShape(shape));
    tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
    return parameterInit(p, g);
}

/* Trainable Linear with manually built parameters: the factory's KAIMING
 * init requires FLOAT32 weight storage (LayerCommon.c requireFloat32, by
 * design — #270), so SYM Linear layers go through the shared
 * buildBorrowedLinearLayer around explicit parameter_t, extended with grad
 * tensors for sgdMCreateOptim / sgdStepM. */
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

    return buildBorrowedLinearLayer(weights, bias, mathQ);
}

/* Manual Quant-layer builder: documents the quantizationConfig_t contract
 * without depending on the factory (Task D.7). Borrowed quantizations. */
static layer_t *buildQuantLayer(quantization_t *forwardQ, quantization_t *backwardQ) {
    quantizationConfig_t *cfg = reserveMemory(sizeof(quantizationConfig_t));
    cfg->outputQ = forwardQ;
    cfg->propLossQ = backwardQ;
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

/* freeOptim cascades into freeParameter for every registered parameter_t —
 * the SAME objects the layers reference; layers are torn down shell-only
 * afterwards (UnitTestLayerNormIntegration pattern). Linear shells go
 * through the shared freeLinearLayerShellOnly (BorrowedLayer.h). */
static void freeLayerNormLayerShell(layer_t *layer) {
    freeReservedMemory(layer->config->layerNorm->normalizedShape);
    freeReservedMemory(layer->config->layerNorm);
    freeReservedMemory(layer->config);
    freeReservedMemory(layer);
}

void testSgdMCreateOptimSkipsQuantizationLayer(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layer_t *linear = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {linear, quant};

    quantization_t *momentumQ = quantizationInitFloat();
    optimizer_t *optim =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, model, 2, momentumQ,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    size_t sizeStates = optim->sizeStates;

    freeOptim(optim); /* frees the Linear weights/bias parameter_t */
    freeQuantLayerShell(quant);
    freeLinearLayerShellOnly(linear);
    freeQuantization(momentumQ);
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

    trainingStats_t *stats =
        calculateGradsSequential(model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, input, label);
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
    freeLinearLayerShellOnly(linear);
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
    freeLinearLayerShellOnly(linear);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(predictedNotNull);
    TEST_ASSERT_TRUE_MESSAGE(dequantFinite, "dequantized inference output must be finite");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa > 0, "inference output must be non-trivial");
    TEST_ASSERT_TRUE_MESSAGE(maxAbsMantissa <= 32767,
                             "quant layer output must obey the int16 inter-layer contract");
}

static const float W1_VALS[6] = {0.6f, -0.4f, 0.2f, -0.5f, 0.3f, 0.9f};
static const float B1_VALS[2] = {0.05f, -0.1f};
static const float LABEL_VALS[4] = {1.f, -1.f, 0.5f, 2.f};

/* Acceptance criterion of #192 (spec D3): one full-SYM training step through
 * Linear(4->3) -> Quant -> LayerNorm([3]) -> Quant -> Linear(3->2) + MSE +
 * SGD-M, vs a FLOAT32 twin (Linear -> LayerNorm -> Linear; a FLOAT->FLOAT
 * Quant layer is a config error by design, so the twin omits them) trained
 * on identical data.
 *
 * Tolerance: the single-layer SYM precedent is 5e-3 absolute
 * (UnitTestLayerNormIntegration PR-0/PR-3). Here error compounds: weight/
 * input/label quantization (<= absMax/2/32767 ~ 1.5e-5 * absMax per tensor),
 * TWO dynamic requant hops in forward AND backward (<= scale/2 = 0.5 LSB per
 * element per hop), LayerNorm strategy-A grads, and two Linear grad paths
 * consuming requantized dy. Treating each of the ~4 noisy stages as one
 * 5e-3-class error source in series gives 4 * 5e-3 = 2e-2 as the chain
 * bound. */
#define CHAIN_TOL 2e-2f

/* Linear SYM bias grads accumulate at fixed scale (init 1.0); one SGD step
 * moves them in whole LSBs of size lr, so the worst-case deviation from the
 * float twin is lr * 0.5 = 0.1 * 0.5 = 0.05. This is the documented Deutel-
 * style fixed-scale scheme (docs/CONVENTIONS.md "Two accumulation schemes";
 * scheme choice tracked in #218), NOT a Quantization-layer error — all
 * weight/gamma/beta paths through both requant hops stay <= CHAIN_TOL. */
#define BIAS_TOL 5e-2f

static void dequantParam(parameter_t *p, float *out, size_t n, bool sym) {
    if (sym) {
        float scale = ((symInt32QConfig_t *)p->param->quantization->qConfig)->scale;
        for (size_t i = 0; i < n; i++) {
            out[i] = (float)((int32_t *)p->param->data)[i] * scale;
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            out[i] = ((float *)p->param->data)[i];
        }
    }
}

void testFullSymChainTrainingStepMatchesFloatTwin(void) {
    size_t normShape[1] = {3};

    /* ---- SYM model (under test) ---- */
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym;
    layerQuantInitUniform(&lqSym, symQ);
    layer_t *lin0S = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *quant1 = quantLayerInit(&lqSym);
    layer_t *lnS = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqSym);
    layer_t *quant2 = quantLayerInit(&lqSym);
    layer_t *lin1S = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, symQ, true);
    layer_t *modelSym[5] = {lin0S, quant1, lnS, quant2, lin1S};

    /* Setup assert: the chain satisfies the int16 inter-layer contract. */
    TEST_ASSERT_TRUE_MESSAGE(validateModelQuantization(modelSym, 5),
                             "chain with Quant layers must pass the validator");

    tensor_t *inS = build2DSym(2, 4, INPUT_VALS, 8);
    tensor_t *labelS = build2DSym(2, 2, LABEL_VALS, 4);

    quantization_t *momentumQS = quantizationInitFloat();
    optimizer_t *optimS =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelSym, 5, momentumQS,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingStats_t *statsS =
        calculateGradsSequential(modelSym, 5, defaultLossConfig(MSE), REDUCTION_MEAN, inS, labelS);
    sgdStepM(optimS);

    bool symStatsNotNull = (statsS != NULL);
    float symLoss = statsS ? statsS->loss : NAN;

    float w0S[12], b0S[3], gammaS[3], betaS[3], w1S[6], b1S[2];
    dequantParam(lin0S->config->linear->weights, w0S, 12, true);
    dequantParam(lin0S->config->linear->bias, b0S, 3, true);
    dequantParam(lnS->config->layerNorm->gamma, gammaS, 3, true);
    dequantParam(lnS->config->layerNorm->beta, betaS, 3, true);
    dequantParam(lin1S->config->linear->weights, w1S, 6, true);
    dequantParam(lin1S->config->linear->bias, b1S, 2, true);

    /* ---- FLOAT32 twin (reference) ---- */
    quantization_t *fQ = quantizationInitFloat();
    layerQuant_t lqF;
    layerQuantInitUniform(&lqF, fQ);
    layer_t *lin0F = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, fQ, false);
    layer_t *lnF = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqF);
    layer_t *lin1F = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, fQ, false);
    layer_t *modelF[3] = {lin0F, lnF, lin1F};

    TEST_ASSERT_TRUE_MESSAGE(validateModelQuantization(modelF, 3),
                             "FLOAT-only twin must pass the validator");

    tensor_t *inF = build2DFloat(2, 4, INPUT_VALS, 8);
    tensor_t *labelF = build2DFloat(2, 2, LABEL_VALS, 4);

    quantization_t *momentumQF = quantizationInitFloat();
    optimizer_t *optimF =
        sgdMCreateOptim(0.1f, 0.0f, 0.0f, modelF, 3, momentumQF,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});
    trainingStats_t *statsF =
        calculateGradsSequential(modelF, 3, defaultLossConfig(MSE), REDUCTION_MEAN, inF, labelF);
    sgdStepM(optimF);

    float w0F[12], b0F[3], gammaF[3], betaF[3], w1F[6], b1F[2];
    dequantParam(lin0F->config->linear->weights, w0F, 12, false);
    dequantParam(lin0F->config->linear->bias, b0F, 3, false);
    dequantParam(lnF->config->layerNorm->gamma, gammaF, 3, false);
    dequantParam(lnF->config->layerNorm->beta, betaF, 3, false);
    dequantParam(lin1F->config->linear->weights, w1F, 6, false);
    dequantParam(lin1F->config->linear->bias, b1F, 2, false);

    freeTrainingStats(statsF);
    freeTrainingStats(statsS);
    freeTensor(labelF);
    freeTensor(inF);
    freeTensor(labelS);
    freeTensor(inS);
    freeOptim(optimF); /* frees w0F/b0F, gammaF/betaF, w1F/b1F parameter_t */
    freeOptim(optimS);
    freeLinearLayerShellOnly(lin1F);
    freeLayerNormLayerShell(lnF);
    freeLinearLayerShellOnly(lin0F);
    freeLinearLayerShellOnly(lin1S);
    freeQuantLayer(quant2);
    freeLayerNormLayerShell(lnS);
    freeQuantLayer(quant1);
    freeLinearLayerShellOnly(lin0S);
    freeQuantization(momentumQF);
    freeQuantization(momentumQS);
    freeQuantization(fQ);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(symStatsNotNull);
    TEST_ASSERT_TRUE_MESSAGE(isfinite(symLoss), "SYM chain loss must be finite");
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_TRUE_MESSAGE(fabsf(gammaF[i] - 1.0f) > 1e-6f,
                                 "reference gamma must actually train");
    }
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, w0F[i], w0S[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(BIAS_TOL, b0F[i], b0S[i]);
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, gammaF[i], gammaS[i]);
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, betaF[i], betaS[i]);
    }
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(CHAIN_TOL, w1F[i], w1S[i]);
    }
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(BIAS_TOL, b1F[i], b1S[i]);
    }
}

/* The Quant layer in this chain (ConvT -> Quant -> MSE) is deliberately
 * RETAINED, not required: ConvT1d's forward already restores its output wire
 * to `symQ`'s width at the producer (executeOp's OUT_WRITE epilogue,
 * PR1b.2), so a same-config Quant node consuming it straight after is the
 * documented double-requant anti-pattern (docs/conventions/arithmetic-sym.md
 * — "a Quant node with an IDENTICAL config directly consuming a
 * funnel-restored producer wire"), i.e. REDUNDANT on this wire post-PR1b.2.
 * It is kept here to exercise the Quantization layer inside a real training
 * loop (loop-integration coverage), not as a model an author should copy. */
void testConv1dTransposedSymChainTrains(void) {
    /* Tiny uniform-SYM chain: ConvT(1->1, K=2) -> Quant -> MSE. int12 operands keep
     * int12*int12 products inside int32; the loop forces int16 grad accumulators.
     * Calibrated: lr=0.05, STEPS=40 gives firstLoss=0.100313 -> lastLoss=0.006788
     * (~14.8x decrease, monotone). Verified deterministic over 3 consecutive runs. */
    quantization_t *symQ = quantizationInitSymInt32WithBits(HALF_AWAY, 12);

    size_t weightDims[3] = {1, 1, 2}; /* [Cin, Cout/groups, K] */
    size_t biasDims[1] = {1};
    static const float W_VALS[2] = {0.30f, -0.20f};
    static const float B_VALS[1] = {0.05f};
    parameter_t *cw = buildConvTSymParam(3, weightDims, W_VALS);
    parameter_t *cb = buildConvTSymParam(1, biasDims, B_VALS);

    static kernel_t kernel;
    initKernel(&kernel, 2, VALID, 1, 1);
    static conv1dTransposedConfig_t cfg;
    initConv1dTransposedConfigWithWeightsAndBias(&cfg, &kernel, cw, cb, 1, 0, symQ, symQ, symQ,
                                                 symQ);
    static layerConfig_t convLc;
    static layer_t convLayer;
    convLc.conv1dTransposed = &cfg;
    initLayer(&convLayer, CONV1D_TRANSPOSED, &convLc);

    layer_t *quant = buildQuantLayer(symQ, symQ);
    layer_t *model[2] = {&convLayer, quant};

    TEST_ASSERT_TRUE_MESSAGE(validateModelQuantization(model, 2),
                             "ConvT-SYM -> Quant must pass the validator");

    /* Fixed learnable sample: input [1,1,3] -> ConvT output [1,1,4]; label [1,1,4]. */
    static const float IN_VALS[3] = {0.5f, -0.3f, 0.8f};
    static const float LABEL_VALS[4] = {0.10f, 0.20f, -0.15f, 0.05f};

    quantization_t *momentumQ2 = quantizationInitFloat();
    optimizer_t *opt =
        sgdMCreateOptim(0.05f, 0.0f, 0.0f, model, 2, momentumQ2,
                        (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY});

    size_t STEPS = 40;
    float firstLoss = NAN;
    float lastLoss = NAN;
    for (size_t s = 0; s < STEPS; s++) {
        tensor_t *in = build3DSymI12(1, 1, 3, IN_VALS, 3);
        tensor_t *label = build3DSymI12(1, 1, 4, LABEL_VALS, 4);
        trainingStats_t *st =
            calculateGradsSequential(model, 2, defaultLossConfig(MSE), REDUCTION_MEAN, in, label);
        if (s == 0) {
            firstLoss = st->loss;
        }
        lastLoss = st->loss;
        sgdStepM(opt);
        optimizerZeroGrad(opt);
        freeTrainingStats(st);
        freeTensor(label);
        freeTensor(in);
    }

    bool decreased = lastLoss < firstLoss;
    bool finite = isfinite(firstLoss) && isfinite(lastLoss);

    freeOptim(opt); /* frees cw/cb parameter_t */
    freeQuantLayerShell(quant);
    freeQuantization(momentumQ2);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(finite, "SYM ConvT chain losses must be finite");
    TEST_ASSERT_TRUE_MESSAGE(decreased, "SYM ConvT chain must train (loss must decrease)");
}

/* Retired-rule contract update (PR1b.2, spec D3): a SYM-producer chain with
 * no Quant layers between producers used to be REJECTED by
 * validateModelQuantization; the forward funnel now restores width at each
 * producer's own wire, so this exact chain is a perfectly ordinary, ACCEPTED
 * model — a Quant layer here would be a redundant identical-config requant,
 * not a requirement. */
void testValidatorAcceptsChainWithoutQuantLayers(void) {
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lqSym;
    layerQuantInitUniform(&lqSym, symQ);
    size_t normShape[1] = {3};
    layer_t *lin0 = buildTrainableLinear(4, 3, W0_VALS, B0_VALS, symQ, true);
    layer_t *ln = layerNormLayerInit(
        &(layerNormInit_t){.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f}, &lqSym);
    layer_t *lin1 = buildTrainableLinear(3, 2, W1_VALS, B1_VALS, symQ, true);
    layer_t *model[3] = {lin0, ln, lin1};

    /* Linear-SYM's forward already restores width before feeding LayerNorm-SYM
     * directly — no Quant layer needed between them. */
    bool valid = validateModelQuantization(model, 3);

    freeParameter(lin1->config->linear->weights);
    freeParameter(lin1->config->linear->bias);
    freeLinearLayerShellOnly(lin1);
    freeLayerNormLayer(ln); /* factory free: gamma/beta not registered with any optimizer */
    freeParameter(lin0->config->linear->weights);
    freeParameter(lin0->config->linear->bias);
    freeLinearLayerShellOnly(lin0);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE_MESSAGE(valid, "a SYM-producer chain without Quant layers between "
                                    "producers is accepted: the forward funnel already "
                                    "restored width at each producer's wire");
}

/* BFP epic PR1: a same-dtype BFP pair is a LEGAL Quantization-layer config
 * beside SYM_INT32 -- it is the documented re-block/width-change point
 * (conversionMatrix[BFP][BFP]). Config-acceptance level: the layer's forward
 * runs once over stack tensors, NOT because heap BFP tensors would leak --
 * freeQuantization's BFP arm (TensorApi.c) frees exponents[] cleanly through
 * freeTensor/freeQuantization, same as SYM. The real reason is that the WIRE
 * ALLOCATORS (initLayerOutputs/initGradTensor, CalculateGradsSequential.c;
 * InferenceApi.c) have no BFP arm yet -- they default to "Unknown QType!"
 * until epic PR2 -- so no model training/inference loop can allocate a live
 * BFP output/grad tensor today; stack fixtures sidestep that gap entirely.
 * Fixture = testRequantBfpWidthChange's hand-derived gold: m=8 values
 * {100,-50,25,0} -> m=4 stored exponent 131 (E=+4, scale 16), mantissas
 * {6,-3,2,0} -- FRESH exponent proves the forward reached the diagonal. */
void testQuantLayerAcceptsSameDtypeBfpPairForward(void) {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[4] = {100, -50, 25, 0};
    uint8_t inExponents[1] = {127}; /* E=0, scale 1 */
    bfpQConfig_t inQC = {.exponents = inExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 8,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, inData, 8, n);
    tensor_t input;
    setTensorValues(&input, inData, &shape, &inQ, NULL);

    uint8_t outExponents[1] = {9}; /* sentinel != expected 131 */
    bfpQConfig_t outQC = {.exponents = outExponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t outQ;
    initBfpQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t output;
    setTensorValues(&output, outData, &shape, &outQ, NULL);

    layer_t *quant = buildQuantLayer(&outQ, &inQ); /* documents the config contract */
    quantizationForward(quant, &input, &output);
    freeQuantLayerShell(quant);

    TEST_ASSERT_EQUAL_UINT8(131, outQC.exponents[0]);
    int32_t mant[4];
    unpackSignExtend(output.data, 4, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(6, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-3, mant[1]);
    TEST_ASSERT_EQUAL_INT32(2, mant[2]);
    TEST_ASSERT_EQUAL_INT32(0, mant[3]);
}

/* Counter-pin for the widened allowance: same-dtype pairs OTHER than
 * SYM_INT32/BFP (here FLOAT32->FLOAT32) stay a config error -- a Quantization
 * layer that neither changes dtype nor requantizes is a misconfigured no-op. */
void testQuantLayerStillRejectsSameDtypeFloatPair(void) {
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inVals[2] = {1.f, -2.f};
    quantization_t inQ;
    initFloat32Quantization(&inQ);
    tensor_t input;
    setTensorValues(&input, (uint8_t *)inVals, &shape, &inQ, NULL);

    float outVals[2] = {0.f, 0.f};
    quantization_t outQ;
    initFloat32Quantization(&outQ);
    tensor_t output;
    setTensorValues(&output, (uint8_t *)outVals, &shape, &outQ, NULL);

    layer_t qLayer = {.type = QUANTIZATION, .config = NULL}; /* dispatch reads only tensors */
    ASSERT_EXITS_WITH_FAILURE(quantizationForward(&qLayer, &input, &output));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSgdMCreateOptimSkipsQuantizationLayer);
    RUN_TEST(testTrainingStepLinearSymThenQuantRequantsOutputAndFiniteLoss);
    RUN_TEST(testInferenceLinearSymThenQuantOutputsInt16RangeMantissas);
    RUN_TEST(testFullSymChainTrainingStepMatchesFloatTwin);
    RUN_TEST(testValidatorAcceptsChainWithoutQuantLayers);
    RUN_TEST(testConv1dTransposedSymChainTrains);
    RUN_TEST(testQuantLayerAcceptsSameDtypeBfpPairForward);
    RUN_TEST(testQuantLayerStillRejectsSameDtypeFloatPair);
    return UNITY_END();
}
