#define SOURCE_FILE "UNIT_TEST_GROUPNORM"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "Arithmetic.h"
#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "GroupNorm.h"
#include "GroupNormApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

#include "DeathTest.h"
#include "expected_groupnorm.h"

void setUp(void) {}
void tearDown(void) {}

/* Build a FLOAT32 tensor of the given rank with the given dims and (optional)
 * row-major data. Caller frees via freeTensor. */
static tensor_t *buildFloatTensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

/* Build a gamma or beta parameter_t of shape [C], filled from `vals` (NULL ->
 * calloc-zero). Grad is FLOAT32 zero. Caller frees via freeParameter. */
static parameter_t *buildFloatParam(size_t numChannels, const float *vals) {
    tensor_t *p = buildFloatTensorND(1, (size_t[]){numChannels}, vals);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=ODT_SYM_OPERAND_QMAXBITS=12) tensor; float vals are
 * quantized via tensorFillFromFloatBuffer -> convertFloatTensorToSymInt32Tensor (absmax -> scale,
 * round-clamp; absmax==0 -> scale 1.0). NULL vals -> zero mantissas, default scale 1.0. Caller
 * frees via freeTensor. */
static tensor_t *buildSymInt32TensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

/* SYM_INT32 [C] parameter with FLOAT32 grad — the repo-default grad storage
 * (#261: SYM_INT32 is a compute format, grads default FLOAT32). */
static parameter_t *buildSymParamFloatGrad(size_t numChannels, const float *vals) {
    tensor_t *p = buildSymInt32TensorND(1, (size_t[]){numChannels}, vals);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

static float symScaleOf(tensor_t *t) {
    return ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
}

/* Wrap a groupNormConfig in a stack layer_t. */
static layer_t makeGroupNormLayer(groupNormConfig_t *cfg, layerConfig_t *lcfg) {
    lcfg->groupNorm = cfg;
    layer_t layer = {.type = GROUPNORM, .config = lcfg};
    return layer;
}

void testConfigStructIsPopulated(void) {
    parameter_t *gamma = buildFloatParam(4, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildFloatParam(4, NULL);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, fq, bq);

    size_t numGroups = cfg.numGroups;
    size_t numChannels = cfg.numChannels;
    float eps = cfg.eps;
    bool gammaOk = (cfg.gamma == gamma);
    bool betaOk = (cfg.beta == beta);
    bool fqOk = (cfg.outputQ == fq);
    bool bqOk = (cfg.propLossQ == bq);
    bool forwardMathOk = (cfg.forwardMath.type == ARITH_FLOAT32);
    bool propLossMathOk = (cfg.propLossMath.type == ARITH_FLOAT32);
    bool weightAccOk = (cfg.weightGradAccMode == OUT_ACC_DYNAMIC_RESCALE);
    bool biasAccOk = (cfg.biasGradAccMode == OUT_ACC_DYNAMIC_RESCALE);
    bool ownsOk = (cfg.ownsQuantizations == false);

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);

    TEST_ASSERT_EQUAL_UINT(2, numGroups);
    TEST_ASSERT_EQUAL_UINT(4, numChannels);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, eps);
    TEST_ASSERT_TRUE(gammaOk);
    TEST_ASSERT_TRUE(betaOk);
    TEST_ASSERT_TRUE(fqOk);
    TEST_ASSERT_TRUE(bqOk);
    TEST_ASSERT_TRUE(forwardMathOk);
    TEST_ASSERT_TRUE(propLossMathOk);
    TEST_ASSERT_TRUE(weightAccOk);
    TEST_ASSERT_TRUE(biasAccOk);
    TEST_ASSERT_TRUE(ownsOk);
}

void testCalcOutputShapeIsIdentity(void) {
    size_t inDims[] = {2, 6, 4};
    tensor_t *in = buildFloatTensorND(3, inDims, NULL);

    size_t *odims = reserveMemory(3 * sizeof(size_t));
    size_t *oorder = reserveMemory(3 * sizeof(size_t));
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    outShape->dimensions = odims;
    outShape->orderOfDimensions = oorder;
    outShape->numberOfDimensions = 0;

    layer_t layer = {.type = GROUPNORM, .config = NULL};
    groupNormCalcOutputShape(&layer, in->shape, outShape);

    size_t nd = outShape->numberOfDimensions;
    size_t d0 = outShape->dimensions[0];
    size_t d1 = outShape->dimensions[1];
    size_t d2 = outShape->dimensions[2];
    size_t o0 = outShape->orderOfDimensions[0];
    size_t o1 = outShape->orderOfDimensions[1];
    size_t o2 = outShape->orderOfDimensions[2];

    freeReservedMemory(outShape);
    freeReservedMemory(oorder);
    freeReservedMemory(odims);
    freeTensor(in);

    TEST_ASSERT_EQUAL_UINT(3, nd);
    TEST_ASSERT_EQUAL_UINT(2, d0);
    TEST_ASSERT_EQUAL_UINT(6, d1);
    TEST_ASSERT_EQUAL_UINT(4, d2);
    TEST_ASSERT_EQUAL_UINT(0, o0);
    TEST_ASSERT_EQUAL_UINT(1, o1);
    TEST_ASSERT_EQUAL_UINT(2, o2);
}

/* Run forward over a [B,C,T]/G fixture and compare to the expected y.
 * viaVtable exercises the layerFunctions[GROUPNORM] registry row instead of
 * the direct call (both must dispatch to the same forward). */
static void runGoldForward(bool viaVtable, size_t B, size_t C, size_t T, size_t G,
                           const float *xVals, const float *gammaVals, const float *betaVals,
                           const float *expectedY, size_t count) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(B * C * T, count, "fixture geometry vs array length");
    size_t dims[] = {B, C, T};
    tensor_t *in = buildFloatTensorND(3, dims, xVals);
    tensor_t *out = buildFloatTensorND(3, dims, NULL);
    parameter_t *gamma = buildFloatParam(C, gammaVals);
    parameter_t *beta = buildFloatParam(C, betaVals);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, G, C, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    if (viaVtable) {
        layerFunctions[GROUPNORM].forward(&layer, in, out);
    } else {
        groupNormForward(&layer, in, out);
    }

    float captured[64];
    for (size_t i = 0; i < count; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedY[i], captured[i]);
    }
}

void testGoldForwardSingleGroup(void) {
    runGoldForward(false, 1, 4, 5, 1, input_groupNorm_singleGroup, gamma_groupNorm_singleGroup,
                   beta_groupNorm_singleGroup, expectedForward_groupNorm_singleGroup,
                   expectedForward_groupNorm_singleGroup_len);
}

void testGoldForwardTwoGroups(void) {
    runGoldForward(false, 1, 8, 3, 2, input_groupNorm_twoGroups, gamma_groupNorm_twoGroups,
                   beta_groupNorm_twoGroups, expectedForward_groupNorm_twoGroups,
                   expectedForward_groupNorm_twoGroups_len);
}

void testGoldForwardBatch2ThreeGroups(void) {
    runGoldForward(false, 2, 6, 4, 3, input_groupNorm_batch2ThreeGroups,
                   gamma_groupNorm_batch2ThreeGroups, beta_groupNorm_batch2ThreeGroups,
                   expectedForward_groupNorm_batch2ThreeGroups,
                   expectedForward_groupNorm_batch2ThreeGroups_len);
}

void testGoldForwardGroupEqualsChannels(void) {
    runGoldForward(false, 1, 4, 3, 4, input_groupNorm_groupEqualsChannels,
                   gamma_groupNorm_groupEqualsChannels, beta_groupNorm_groupEqualsChannels,
                   expectedForward_groupNorm_groupEqualsChannels,
                   expectedForward_groupNorm_groupEqualsChannels_len);
}

void testVtableGoldForwardTwoGroups(void) {
    runGoldForward(true, 1, 8, 3, 2, input_groupNorm_twoGroups, gamma_groupNorm_twoGroups,
                   beta_groupNorm_twoGroups, expectedForward_groupNorm_twoGroups,
                   expectedForward_groupNorm_twoGroups_len);
}

/* Small-variance fixture: var (1.25e-6) is comparable to eps (1e-5), so this
 * test distinguishes sqrt(var+eps) from the wrong sqrt(var)+eps — the O(1)-
 * variance gold fixtures cannot (relative difference ~5e-6, below tolerance).
 *   B=1, C=2, T=2, G=1 -> one group over all 4 elements:
 *   x = [0.001, 0.002, 0.003, 0.004]: mean=0.0025, biased var=1.25e-6,
 *   invSigma = 1/sqrt(1.125e-5) = 298.142, gamma=1, beta=0
 *   y = n = [-0.447213, -0.149071, +0.149071, +0.447213]
 *   (sqrt(var)+eps would give y0 ~ -1.3298 instead). */
void testForwardFloatSmallVarianceEpsInsideSqrt(void) {
    float x[] = {0.001f, 0.002f, 0.003f, 0.004f};
    float gammaOnes[] = {1.f, 1.f};
    float betaZeros[] = {0.f, 0.f};
    float expected[] = {-0.447213f, -0.149071f, 0.149071f, 0.447213f};
    runGoldForward(false, 1, 2, 2, 1, x, gammaOnes, betaZeros, expected, 4);
}

/* Run backward (via the vtable row) over a [B,C,T]/G fixture with fresh zero
 * grads; compare dx (overwritten into propLoss), dgamma and dbeta (accumulated
 * into param->grad->data). doublePass runs backward TWICE and expects 2x the
 * param grads (accumulate semantics) while dx must stay 1x (overwrite
 * semantics). */
static void runGoldBackward(bool doublePass, size_t B, size_t C, size_t T, size_t G,
                            const float *xVals, const float *gammaVals, const float *betaVals,
                            const float *lossGradVals, const float *expDx, const float *expDgamma,
                            const float *expDbeta, size_t count) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    TEST_ASSERT_TRUE_MESSAGE(C <= 16, "fixture exceeds grad capture buffer");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(B * C * T, count, "fixture geometry vs array length");
    size_t dims[] = {B, C, T};
    tensor_t *fwdIn = buildFloatTensorND(3, dims, xVals);
    tensor_t *loss = buildFloatTensorND(3, dims, lossGradVals);
    tensor_t *propLoss = buildFloatTensorND(3, dims, NULL);
    parameter_t *gamma = buildFloatParam(C, gammaVals);
    parameter_t *beta = buildFloatParam(C, betaVals);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, G, C, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    layerFunctions[GROUPNORM].backward(&layer, fwdIn, loss, propLoss);
    if (doublePass) {
        layerFunctions[GROUPNORM].backward(&layer, fwdIn, loss, propLoss);
    }

    float dx[64], dg[16], db[16];
    for (size_t i = 0; i < count; i++) {
        dx[i] = ((float *)propLoss->data)[i];
    }
    for (size_t i = 0; i < C; i++) {
        dg[i] = ((float *)gamma->grad->data)[i];
        db[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float gradFactor = doublePass ? 2.0f : 1.0f;
    float gradTol = doublePass ? 2e-4f : 1e-4f;
    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]); /* overwrite: 1x even on 2nd pass */
    }
    for (size_t i = 0; i < C; i++) {
        TEST_ASSERT_FLOAT_WITHIN(gradTol, gradFactor * expDgamma[i], dg[i]);
        TEST_ASSERT_FLOAT_WITHIN(gradTol, gradFactor * expDbeta[i], db[i]);
    }
}

void testGoldBackwardSingleGroup(void) {
    runGoldBackward(false, 1, 4, 5, 1, input_groupNorm_singleGroup, gamma_groupNorm_singleGroup,
                    beta_groupNorm_singleGroup, lossGrad_groupNorm_singleGroup,
                    expectedPropLoss_groupNorm_singleGroup, expectedDgamma_groupNorm_singleGroup,
                    expectedDbeta_groupNorm_singleGroup, expectedForward_groupNorm_singleGroup_len);
}

void testGoldBackwardTwoGroups(void) {
    runGoldBackward(false, 1, 8, 3, 2, input_groupNorm_twoGroups, gamma_groupNorm_twoGroups,
                    beta_groupNorm_twoGroups, lossGrad_groupNorm_twoGroups,
                    expectedPropLoss_groupNorm_twoGroups, expectedDgamma_groupNorm_twoGroups,
                    expectedDbeta_groupNorm_twoGroups, expectedForward_groupNorm_twoGroups_len);
}

void testGoldBackwardBatch2ThreeGroups(void) {
    runGoldBackward(
        false, 2, 6, 4, 3, input_groupNorm_batch2ThreeGroups, gamma_groupNorm_batch2ThreeGroups,
        beta_groupNorm_batch2ThreeGroups, lossGrad_groupNorm_batch2ThreeGroups,
        expectedPropLoss_groupNorm_batch2ThreeGroups, expectedDgamma_groupNorm_batch2ThreeGroups,
        expectedDbeta_groupNorm_batch2ThreeGroups, expectedForward_groupNorm_batch2ThreeGroups_len);
}

void testGoldBackwardGroupEqualsChannels(void) {
    runGoldBackward(
        false, 1, 4, 3, 4, input_groupNorm_groupEqualsChannels, gamma_groupNorm_groupEqualsChannels,
        beta_groupNorm_groupEqualsChannels, lossGrad_groupNorm_groupEqualsChannels,
        expectedPropLoss_groupNorm_groupEqualsChannels,
        expectedDgamma_groupNorm_groupEqualsChannels, expectedDbeta_groupNorm_groupEqualsChannels,
        expectedForward_groupNorm_groupEqualsChannels_len);
}

/* dgamma/dbeta must ACCUMULATE across calls (+=); dx must OVERWRITE. */
void testBackwardAccumulatesGradsOverwritesDx(void) {
    runGoldBackward(true, 1, 8, 3, 2, input_groupNorm_twoGroups, gamma_groupNorm_twoGroups,
                    beta_groupNorm_twoGroups, lossGrad_groupNorm_twoGroups,
                    expectedPropLoss_groupNorm_twoGroups, expectedDgamma_groupNorm_twoGroups,
                    expectedDbeta_groupNorm_twoGroups, expectedForward_groupNorm_twoGroups_len);
}

/* #380 PR1 Task 6: a frozen twin's backward must skip the dgamma/dbeta
 * accumulate lines entirely (buffers stay all-zero) while still producing a
 * dx byte-identical to its trainable twin. Duplicates the twoGroups gold
 * fixture (dgamma/dbeta both gold-verified nonzero) into two independent
 * twins that differ only in cfg.frozen -- forwardInput/loss are read-only in
 * backward, so a single shared instance of each is safe to reuse across both
 * calls. */
void testBackwardFloatFrozenTwinDxIdenticalGradsZero(void) {
    size_t B = 1;
    size_t C = 8;
    size_t T = 3;
    size_t G = 2;
    size_t dims[] = {B, C, T};
    tensor_t *fwdIn = buildFloatTensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildFloatTensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLossTrainable = buildFloatTensorND(3, dims, NULL);
    tensor_t *propLossFrozen = buildFloatTensorND(3, dims, NULL);

    parameter_t *gammaA = buildFloatParam(C, gamma_groupNorm_twoGroups);
    parameter_t *betaA = buildFloatParam(C, beta_groupNorm_twoGroups);
    parameter_t *gammaB = buildFloatParam(C, gamma_groupNorm_twoGroups);
    parameter_t *betaB = buildFloatParam(C, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    groupNormConfig_t cfgTrainable;
    initGroupNormConfig(&cfgTrainable, gammaA, betaA, G, C, 1e-5f, fq, bq);
    layerConfig_t lcfgTrainable;
    layer_t layerTrainable = makeGroupNormLayer(&cfgTrainable, &lcfgTrainable);

    groupNormConfig_t cfgFrozen;
    initGroupNormConfig(&cfgFrozen, gammaB, betaB, G, C, 1e-5f, fq, bq);
    cfgFrozen.frozen = true;
    layerConfig_t lcfgFrozen;
    layer_t layerFrozen = makeGroupNormLayer(&cfgFrozen, &lcfgFrozen);

    groupNormBackward(&layerTrainable, fwdIn, loss, propLossTrainable);
    groupNormBackward(&layerFrozen, fwdIn, loss, propLossFrozen);

    bool trainableGradNonzero = false;
    bool frozenGradAllZero = true;
    for (size_t i = 0; i < C; i++) {
        if (((float *)gammaA->grad->data)[i] != 0.0f || ((float *)betaA->grad->data)[i] != 0.0f) {
            trainableGradNonzero = true;
        }
        if (((float *)gammaB->grad->data)[i] != 0.0f || ((float *)betaB->grad->data)[i] != 0.0f) {
            frozenGradAllZero = false;
        }
    }
    bool dxIdentical =
        memcmp(propLossTrainable->data, propLossFrozen->data,
               calcNumberOfBytesForData(propLossTrainable->quantization, B * C * T)) == 0;

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(betaB);
    freeParameter(gammaB);
    freeParameter(betaA);
    freeParameter(gammaA);
    freeTensor(propLossFrozen);
    freeTensor(propLossTrainable);
    freeTensor(loss);
    freeTensor(fwdIn);

    TEST_ASSERT_TRUE_MESSAGE(trainableGradNonzero,
                             "trainable twin dgamma/dbeta must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenGradAllZero,
                             "frozen twin dgamma/dbeta must stay untouched (all-zero)");
    TEST_ASSERT_TRUE_MESSAGE(dxIdentical, "dx must be byte-identical between twins");
}

/* FLOAT32-backward guard set (spec §5.4): FLOAT32 forwardInput/loss/gamma
 * dtypes AND FLOAT32 gamma/beta grad storage AND FLOAT32 propLoss (dx) storage —
 * each violation exits(1). whichSym selects the tensor built as SYM_INT32:
 *   0 forwardInput, 1 loss, 2 gamma param, 3 gamma grad, 4 beta grad,
 *   5 propLoss (dx wire; groupNormBackwardFloat writes it via a raw float*). */
static void runBackwardFloatGuard(int whichSym) {
    size_t dims[] = {1, 4, 2};
    float xVals[8] = {1.f, -1.f, 1.f, -1.f, 2.f, -2.f, 2.f, -2.f};
    tensor_t *fwdIn = (whichSym == 0) ? buildSymInt32TensorND(3, dims, xVals)
                                      : buildFloatTensorND(3, dims, xVals);
    tensor_t *loss = (whichSym == 1) ? buildSymInt32TensorND(3, dims, xVals)
                                     : buildFloatTensorND(3, dims, xVals);
    tensor_t *propLoss =
        (whichSym == 5) ? buildSymInt32TensorND(3, dims, NULL) : buildFloatTensorND(3, dims, NULL);

    float ones[4] = {1.f, 1.f, 1.f, 1.f};
    parameter_t *gamma;
    if (whichSym == 2) {
        gamma = buildSymParamFloatGrad(4, ones);
    } else if (whichSym == 3) {
        tensor_t *p = buildFloatTensorND(1, (size_t[]){4}, ones);
        tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
        gamma = parameterInit(p, g);
    } else {
        gamma = buildFloatParam(4, ones);
    }
    parameter_t *beta;
    if (whichSym == 4) {
        tensor_t *p = buildFloatTensorND(1, (size_t[]){4}, NULL);
        tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
        beta = parameterInit(p, g);
    } else {
        beta = buildFloatParam(4, NULL);
    }

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat(); /* propLossMath = ARITH_FLOAT32 */
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, fwdIn, loss, propLoss));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);
}

void testBackwardFloatGuardsSymForwardInput(void) {
    runBackwardFloatGuard(0);
}
void testBackwardFloatGuardsSymLoss(void) {
    runBackwardFloatGuard(1);
}
void testBackwardFloatGuardsSymGammaParam(void) {
    runBackwardFloatGuard(2);
}
void testBackwardFloatGuardsSymGammaGrad(void) {
    runBackwardFloatGuard(3);
}
void testBackwardFloatGuardsSymBetaGrad(void) {
    runBackwardFloatGuard(4);
}
void testBackwardFloatGuardsSymPropLoss(void) {
    runBackwardFloatGuard(5);
}

/* Build a minimal valid FLOAT32 layer around cfgOut (C=4, G=2) for the
 * runtime-guard death tests; the INPUT under test is supplied per test. */
static layer_t buildGuardLayer(groupNormConfig_t *cfg, layerConfig_t *lcfg, parameter_t **gammaOut,
                               parameter_t **betaOut, quantization_t **fqOut,
                               quantization_t **bqOut) {
    *gammaOut = buildFloatParam(4, (float[]){1.f, 1.f, 1.f, 1.f});
    *betaOut = buildFloatParam(4, NULL);
    *fqOut = quantizationInitFloat();
    *bqOut = quantizationInitFloat();
    initGroupNormConfig(cfg, *gammaOut, *betaOut, 2, 4, 1e-5f, *fqOut, *bqOut);
    return makeGroupNormLayer(cfg, lcfg);
}

void testForwardRejectsWrongRank(void) {
    size_t dims[] = {4, 5}; /* rank-2: violates the rank-3 [B,C,T] contract */
    tensor_t *in = buildFloatTensorND(2, dims, NULL);
    tensor_t *out = buildFloatTensorND(2, dims, NULL);

    parameter_t *gamma;
    parameter_t *beta;
    quantization_t *fq;
    quantization_t *bq;
    groupNormConfig_t cfg;
    layerConfig_t lcfg;
    layer_t layer = buildGuardLayer(&cfg, &lcfg, &gamma, &beta, &fq, &bq);

    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);
}

void testForwardRejectsWrongChannelDim(void) {
    size_t dims[] = {1, 3, 5}; /* dims[1]=3 but cfg numChannels=4 */
    tensor_t *in = buildFloatTensorND(3, dims, NULL);
    tensor_t *out = buildFloatTensorND(3, dims, NULL);

    parameter_t *gamma;
    parameter_t *beta;
    quantization_t *fq;
    quantization_t *bq;
    groupNormConfig_t cfg;
    layerConfig_t lcfg;
    layer_t layer = buildGuardLayer(&cfg, &lcfg, &gamma, &beta, &fq, &bq);

    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);
}

void testForwardRejectsTransposedInput(void) {
    size_t dims[] = {1, 5, 4}; /* transposed below -> logical [1,4,5], order non-identity */
    tensor_t *in = buildFloatTensorND(3, dims, NULL);
    transposeTensor(in, 1, 2);
    size_t outDims[] = {1, 4, 5};
    tensor_t *out = buildFloatTensorND(3, outDims, NULL);

    parameter_t *gamma;
    parameter_t *beta;
    quantization_t *fq;
    quantization_t *bq;
    groupNormConfig_t cfg;
    layerConfig_t lcfg;
    layer_t layer = buildGuardLayer(&cfg, &lcfg, &gamma, &beta, &fq, &bq);

    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);
}

/* SYM forward twin-sanity: the SYM path (int-domain stats, global absmax
 * stretch, requant, per-channel integer affine) must stay within a LOOSE
 * tolerance of the FLOAT32 gold on the same data — the SYM opportunity is
 * correctness-imperfect by design (spec R3), so this is sanity, not gold.
 * Data are the twoGroups PyTorch fixtures (randn, O(1) spread), so int12
 * quantization noise is ~1e-3 per stage; 5e-2 gives >10x headroom while an
 * indexing/scale bug shifts values by O(1). */
void testSymForwardTwinSanityTwoGroups(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *in = buildSymInt32TensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *out = buildSymInt32TensorND(3, dims, NULL);
    parameter_t *gamma = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *beta = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 8, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    layerFunctions[GROUPNORM].forward(&layer, in, out);

    float deq[24];
    int32_t *m = (int32_t *)out->data;
    float scale = symScaleOf(out);
    for (size_t i = 0; i < 24; i++) {
        deq[i] = (float)m[i] * scale;
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_TRUE(scale > 0.f);
    for (size_t i = 0; i < 24; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedForward_groupNorm_twoGroups[i], deq[i]);
    }
}

/* groupNormValidateSymTensor is static; exercise it through the forward. The
 * INPUT is built with qMaxBits=13, exceeding the int12 operand contract
 * (ODT_SYM_OPERAND_QMAXBITS=12) that bounds the affine product q*gammaQ. */
void testSymForwardRejectsOperandWiderThanInt12(void) {
    size_t *inDims = reserveMemory(3 * sizeof(size_t));
    inDims[0] = 1;
    inDims[1] = 4;
    inDims[2] = 2;
    size_t *inOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, inOrder);
    shape_t *inShape = reserveMemory(sizeof(shape_t));
    setShape(inShape, inDims, 3, inOrder);
    quantization_t *wideQ = quantizationInitSymInt32WithBits(HALF_AWAY, 13);
    tensor_t *in = initTensor(inShape, wideQ, NULL);

    size_t dims[] = {1, 4, 2};
    tensor_t *out = buildSymInt32TensorND(3, dims, NULL);
    parameter_t *gamma = buildSymParamFloatGrad(4, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParamFloatGrad(4, NULL);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);
}

/* SYM backward twin-sanity (loose, spec R3 — sanity, not gold): dequantized
 * dx and the FLOAT32-default dgamma/dbeta must track the FLOAT32 gold on the
 * same data. Second call: dgamma/dbeta ACCUMULATE (2x, via the identity-
 * kernel executeOp + OUT_ACC_DYNAMIC_RESCALE route) while dx OVERWRITES (1x,
 * scale refreshed). */
void testSymBackwardTwinSanityTwoGroups(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *fwdIn = buildSymInt32TensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildSymInt32TensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLoss = buildSymInt32TensorND(3, dims, NULL);
    parameter_t *gamma = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *beta = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 8, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    layerFunctions[GROUPNORM].backward(&layer, fwdIn, loss, propLoss);

    float dxDeq[24], dg1[8], db1[8];
    float dxScale = symScaleOf(propLoss);
    for (size_t i = 0; i < 24; i++) {
        dxDeq[i] = (float)((int32_t *)propLoss->data)[i] * dxScale;
    }
    for (size_t i = 0; i < 8; i++) {
        dg1[i] = ((float *)gamma->grad->data)[i];
        db1[i] = ((float *)beta->grad->data)[i];
    }

    layerFunctions[GROUPNORM].backward(&layer, fwdIn, loss, propLoss);

    float dxDeq2[24], dg2[8], db2[8];
    float dxScale2 = symScaleOf(propLoss);
    for (size_t i = 0; i < 24; i++) {
        dxDeq2[i] = (float)((int32_t *)propLoss->data)[i] * dxScale2;
    }
    for (size_t i = 0; i < 8; i++) {
        dg2[i] = ((float *)gamma->grad->data)[i];
        db2[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    TEST_ASSERT_TRUE(dxScale > 0.f);
    for (size_t i = 0; i < 24; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedPropLoss_groupNorm_twoGroups[i], dxDeq[i]);
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedPropLoss_groupNorm_twoGroups[i], dxDeq2[i]);
    }
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedDgamma_groupNorm_twoGroups[i], dg1[i]);
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedDbeta_groupNorm_twoGroups[i], db1[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-1f, 2.0f * expectedDgamma_groupNorm_twoGroups[i], dg2[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-1f, 2.0f * expectedDbeta_groupNorm_twoGroups[i], db2[i]);
    }
}

/* #380 final-review Fix 2: SYM twin of the frozen-backward guard (mirrors
 * UnitTestLayerNorm.c's testSymBackwardFrozenTwinDxIdenticalGradsUntouched).
 * Duplicates the testSymBackwardTwinSanityTwoGroups fixture (dgamma/dbeta
 * gold-verified nonzero) into two independent twins that differ only in
 * cfg.frozen -- forwardInput/loss are read-only in backward, so a single
 * shared instance of each is safe to reuse across both calls. Pass B (dx
 * requant + propLoss scale refresh) stays unconditional per spec, so dx
 * mantissas AND the refreshed propLoss scale must be IDENTICAL between
 * twins; the frozen twin's dgamma/dbeta (FLOAT32 grad storage, the #261
 * repo default) must stay at their fresh zero init -- the `if (!cfg->frozen)`
 * emission block (incQ -> setTensorValues -> executeOp) never runs for it. */
void testSymBackwardFrozenTwinDxIdenticalGradsUntouched(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *fwdIn = buildSymInt32TensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildSymInt32TensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLossTrainable = buildSymInt32TensorND(3, dims, NULL);
    tensor_t *propLossFrozen = buildSymInt32TensorND(3, dims, NULL);

    parameter_t *gammaA = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *betaA = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);
    parameter_t *gammaB = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *betaB = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);

    groupNormConfig_t cfgTrainable;
    initGroupNormConfig(&cfgTrainable, gammaA, betaA, 2, 8, 1e-5f, fq, bq);
    layerConfig_t lcfgTrainable;
    layer_t layerTrainable = makeGroupNormLayer(&cfgTrainable, &lcfgTrainable);

    groupNormConfig_t cfgFrozen;
    initGroupNormConfig(&cfgFrozen, gammaB, betaB, 2, 8, 1e-5f, fq, bq);
    cfgFrozen.frozen = true;
    layerConfig_t lcfgFrozen;
    layer_t layerFrozen = makeGroupNormLayer(&cfgFrozen, &lcfgFrozen);

    layerFunctions[GROUPNORM].backward(&layerTrainable, fwdIn, loss, propLossTrainable);
    layerFunctions[GROUPNORM].backward(&layerFrozen, fwdIn, loss, propLossFrozen);

    int32_t dxTrainable[24], dxFrozen[24];
    for (size_t i = 0; i < 24; i++) {
        dxTrainable[i] = ((int32_t *)propLossTrainable->data)[i];
        dxFrozen[i] = ((int32_t *)propLossFrozen->data)[i];
    }
    float scaleTrainable = symScaleOf(propLossTrainable);
    float scaleFrozen = symScaleOf(propLossFrozen);

    bool trainableGradNonzero = false;
    bool frozenGradAllZero = true;
    for (size_t i = 0; i < 8; i++) {
        if (((float *)gammaA->grad->data)[i] != 0.0f || ((float *)betaA->grad->data)[i] != 0.0f) {
            trainableGradNonzero = true;
        }
        if (((float *)gammaB->grad->data)[i] != 0.0f || ((float *)betaB->grad->data)[i] != 0.0f) {
            frozenGradAllZero = false;
        }
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(betaB);
    freeParameter(gammaB);
    freeParameter(betaA);
    freeParameter(gammaA);
    freeTensor(propLossFrozen);
    freeTensor(propLossTrainable);
    freeTensor(loss);
    freeTensor(fwdIn);

    for (size_t i = 0; i < 24; i++) {
        TEST_ASSERT_EQUAL_INT32_MESSAGE(dxTrainable[i], dxFrozen[i],
                                        "dx mantissas must be identical between twins");
    }
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(scaleTrainable, scaleFrozen,
                                    "refreshed propLoss scale must be identical between twins");
    TEST_ASSERT_TRUE_MESSAGE(trainableGradNonzero,
                             "trainable twin dgamma/dbeta must be written (nonzero)");
    TEST_ASSERT_TRUE_MESSAGE(frozenGradAllZero,
                             "frozen twin dgamma/dbeta must stay untouched (all-zero)");
}

/* The SYM backward must validate ITS operands too: a loss tensor wider than
 * the int12 operand contract exits(1). */
void testSymBackwardRejectsOperandWiderThanInt12(void) {
    size_t dims[] = {1, 4, 2};
    tensor_t *fwdIn =
        buildSymInt32TensorND(3, dims, (float[]){1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, -4.f});

    size_t *lDims = reserveMemory(3 * sizeof(size_t));
    lDims[0] = 1;
    lDims[1] = 4;
    lDims[2] = 2;
    size_t *lOrder = reserveMemory(3 * sizeof(size_t));
    setOrderOfDimsForNewTensor(3, lOrder);
    shape_t *lShape = reserveMemory(sizeof(shape_t));
    setShape(lShape, lDims, 3, lOrder);
    quantization_t *wideQ = quantizationInitSymInt32WithBits(HALF_AWAY, 13);
    tensor_t *loss = initTensor(lShape, wideQ, NULL);

    tensor_t *propLoss = buildSymInt32TensorND(3, dims, NULL);
    parameter_t *gamma = buildSymParamFloatGrad(4, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParamFloatGrad(4, NULL);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, fwdIn, loss, propLoss));

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);
}

void testFactoryBuildsGammaOnesBetaZerosAndForwards(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 0.0f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(fwdMath),
                       .propLossMath = arithmeticFromQuantization(bwdMath),
                       .outputQ = fwdMath,
                       .propLossQ = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = groupNormLayerInit(&init, &lq);

    bool typeOk = (layer->type == GROUPNORM);
    groupNormConfig_t *cfg = layer->config->groupNorm;
    /* eps==0 -> factory substitutes default 1e-5 */
    float capturedEps = cfg->eps;
    /* gamma all ones, beta all zeros */
    float g0 = ((float *)cfg->gamma->param->data)[0];
    float g3 = ((float *)cfg->gamma->param->data)[3];
    float b0 = ((float *)cfg->beta->param->data)[0];
    bool gammaGradFloat = (cfg->gamma->grad->quantization->type == FLOAT32);
    bool betaGradFloat = (cfg->beta->grad->quantization->type == FLOAT32);
    bool fwdMapped = (cfg->outputQ == fwdMath);
    bool bwdMapped = (cfg->propLossQ == bwdMath);
    bool fwdMathOk = (cfg->forwardMath.type == ARITH_FLOAT32);
    bool propLossMathOk = (cfg->propLossMath.type == ARITH_FLOAT32);

    /* Forward smoke: B=1,C=4,T=2,G=2 -> sane output (not NaN/inf; exact gold
     * values are the hand-wired UnitTestGroupNorm.c coverage above). */
    size_t dims[] = {1, 4, 2};
    tensor_t *in =
        buildFloatTensorND(3, dims, (float[]){1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, -4.f});
    tensor_t *out = buildFloatTensorND(3, dims, NULL);
    layerFunctions[GROUPNORM].forward(layer, in, out);
    float y0 = ((float *)out->data)[0];

    freeTensor(out);
    freeTensor(in);
    freeGroupNormLayer(layer);
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    TEST_ASSERT_TRUE(typeOk);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, capturedEps);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.f, g0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.f, g3);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.f, b0);
    TEST_ASSERT_TRUE(gammaGradFloat);
    TEST_ASSERT_TRUE(betaGradFloat);
    TEST_ASSERT_TRUE(fwdMapped);
    TEST_ASSERT_TRUE(bwdMapped);
    TEST_ASSERT_TRUE(fwdMathOk);
    TEST_ASSERT_TRUE(propLossMathOk);
    TEST_ASSERT_TRUE(y0 == y0); /* not NaN */
}

void testFactoryAppliesDefaultEpsWhenZero(void) {
    groupNormInit_t init = {.numGroups = 1, .numChannels = 4, .eps = 0.0f};

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(q),
                       .propLossMath = arithmeticFromQuantization(q),
                       .outputQ = q,
                       .propLossQ = q,
                       .weightStorage = q,
                       .biasStorage = q};

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;
    float capturedEps = cfg->eps;

    freeGroupNormLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, capturedEps);
}

void testFactoryOwningDeepCopiesQuantizations(void) {
    groupNormInit_t init = {.numGroups = 1, .numChannels = 3, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(fwdMath),
                       .propLossMath = arithmeticFromQuantization(bwdMath),
                       .outputQ = fwdMath,
                       .propLossQ = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = groupNormLayerInitOwning(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    /* Owning: outputQ/propLossQ are fresh allocations, NOT the caller's. */
    bool fwdIsCopy = (cfg->outputQ != fwdMath);
    bool bwdIsCopy = (cfg->propLossQ != bwdMath);
    bool fwdTypeOk = (cfg->outputQ->type == fwdMath->type);
    bool owns = cfg->ownsQuantizations;

    /* Caller drops its math quant configs IMMEDIATELY — the layer holds copies. */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    /* Now tear down the layer — frees gamma/beta + the OWNED outputQ/propLossQ
     * copies. No double-free (the caller's originals are already gone and were
     * never aliased). */
    freeGroupNormLayer(layer);

    TEST_ASSERT_TRUE(fwdIsCopy);
    TEST_ASSERT_TRUE(bwdIsCopy);
    TEST_ASSERT_TRUE(fwdTypeOk);
    TEST_ASSERT_TRUE(owns);
}

void testFactoryBorrowingDoesNotFreeCallerQuantizations(void) {
    groupNormInit_t init = {.numGroups = 1, .numChannels = 3, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(fwdMath),
                       .propLossMath = arithmeticFromQuantization(bwdMath),
                       .outputQ = fwdMath,
                       .propLossQ = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    /* Borrowing: verbatim pointers, no ownership. */
    bool fwdVerbatim = (cfg->outputQ == fwdMath);
    bool bwdVerbatim = (cfg->propLossQ == bwdMath);
    bool owns = cfg->ownsQuantizations;

    /* Free the layer FIRST — it must NOT touch the borrowed math quantizations. */
    freeGroupNormLayer(layer);

    /* Caller frees its own quant configs AFTER. If freeGroupNormLayer had freed
     * outputQ/propLossQ, these would be double-frees (ASan/valgrind catch them). */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    TEST_ASSERT_TRUE(fwdVerbatim);
    TEST_ASSERT_TRUE(bwdVerbatim);
    TEST_ASSERT_FALSE(owns);
}

void testFactoryRejectsNonDivisibleGroups(void) {
    groupNormInit_t init = {.numGroups = 3, .numChannels = 4, .eps = 1e-5f};

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(q),
                       .propLossMath = arithmeticFromQuantization(q),
                       .outputQ = q,
                       .propLossQ = q,
                       .weightStorage = q,
                       .biasStorage = q};

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lq));

    freeQuantization(q);
}

void testFactorySymInt32StorageQuantizesGammaBeta(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 0.0f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwdMath = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(symQ),
                       .propLossMath = arithmeticFromQuantization(bwdMath),
                       .outputQ = symQ,
                       .propLossQ = bwdMath,
                       .weightStorage = symQ,
                       .biasStorage = symQ};

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    int32_t g[4];
    int32_t b[4];
    for (size_t i = 0; i < 4; i++) {
        g[i] = ((int32_t *)cfg->gamma->param->data)[i];
        b[i] = ((int32_t *)cfg->beta->param->data)[i];
    }
    float gScale = symScaleOf(cfg->gamma->param);
    float bScale = symScaleOf(cfg->beta->param);
    bool gammaGradFloat = (cfg->gamma->grad->quantization->type == FLOAT32);
    bool betaGradFloat = (cfg->beta->grad->quantization->type == FLOAT32);

    /* Forward smoke through the vtable on SYM input (factory-built params). */
    size_t dims[] = {1, 4, 2};
    tensor_t *in =
        buildSymInt32TensorND(3, dims, (float[]){1.f, -1.f, 2.f, -2.f, 3.f, -3.f, 4.f, -4.f});
    tensor_t *out = buildSymInt32TensorND(3, dims, NULL);
    layerFunctions[GROUPNORM].forward(layer, in, out);
    float outScale = symScaleOf(out);

    freeTensor(out);
    freeTensor(in);
    freeGroupNormLayer(layer);
    freeQuantization(bwdMath);
    freeQuantization(symQ);

    for (size_t i = 0; i < 4; i++) {
        /* gamma=ones is OPERAND storage (default int12, #227): absmax=1 ->
         * every mantissa = qMax = 2047, scale = 1/2047. beta=zeros -> 0, scale 1. */
        TEST_ASSERT_EQUAL_INT(2047, g[i]);
        TEST_ASSERT_EQUAL_INT(0, b[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f / 2047.0f, gScale);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, bScale);
    TEST_ASSERT_TRUE(gammaGradFloat);
    TEST_ASSERT_TRUE(betaGradFloat);
    TEST_ASSERT_TRUE(outScale > 0.f);
}

void testFactoryOwningSymInt32DeepCopies(void) {
    groupNormInit_t init = {.numGroups = 1, .numChannels = 3, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwdMath = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(symQ),
                       .propLossMath = arithmeticFromQuantization(bwdMath),
                       .outputQ = symQ,
                       .propLossQ = bwdMath,
                       .weightStorage = symQ,
                       .biasStorage = symQ};

    layer_t *layer = groupNormLayerInitOwning(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    bool fwdIsCopy = (cfg->outputQ != symQ);
    bool fwdIsSym = (cfg->outputQ->type == SYM_INT32);
    bool fwdCfgIsCopy = (cfg->outputQ->qConfig != symQ->qConfig);
    bool owns = cfg->ownsQuantizations;

    /* Caller drops its quants immediately — the layer holds deep copies
     * (incl. the symInt32QConfig_t; double-free/UAF surfaces under CI ASan). */
    freeQuantization(bwdMath);
    freeQuantization(symQ);
    freeGroupNormLayer(layer);

    TEST_ASSERT_TRUE(fwdIsCopy);
    TEST_ASSERT_TRUE(fwdIsSym);
    TEST_ASSERT_TRUE(fwdCfgIsCopy);
    TEST_ASSERT_TRUE(owns);
}

/* #380 PR1 Task 3: create-time trainable knob (trainable_t). */
static layer_t *buildFloatGroupNormWithTrainable(trainable_t trainable) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 0.0f, .trainable = trainable};

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = arithmeticFromQuantization(q),
                       .propLossMath = arithmeticFromQuantization(q),
                       .outputQ = q,
                       .propLossQ = q,
                       .weightStorage = q,
                       .biasStorage = q};

    layer_t *layer = groupNormLayerInitOwning(&init, &lq);
    freeQuantization(q);
    return layer;
}

void testGroupNormFactoryFrozenElidesGrads(void) {
    layer_t *layer = buildFloatGroupNormWithTrainable(TRAINABLE_FALSE);
    groupNormConfig_t *cfg = layer->config->groupNorm;
    bool gammaGradNull = cfg->gamma->grad == NULL;
    bool betaGradNull = cfg->beta->grad == NULL;
    bool frozen = layerIsFrozen(layer);
    freeGroupNormLayer(layer);
    TEST_ASSERT_TRUE(gammaGradNull);
    TEST_ASSERT_TRUE(betaGradNull);
    TEST_ASSERT_TRUE(frozen);
}

void testGroupNormFactoryDefaultAllocatesGrads(void) {
    layer_t *layer = buildFloatGroupNormWithTrainable(TRAINABLE_DEFAULT);
    groupNormConfig_t *cfg = layer->config->groupNorm;
    bool gammaGradPresent = cfg->gamma->grad != NULL;
    bool betaGradPresent = cfg->beta->grad != NULL;
    bool frozen = layerIsFrozen(layer);
    freeGroupNormLayer(layer);
    TEST_ASSERT_TRUE(gammaGradPresent);
    TEST_ASSERT_TRUE(betaGradPresent);
    TEST_ASSERT_FALSE(frozen);
}

/* #380 PR1 Task 6: factory-frozen layer (grad == NULL, Task 1) --
 * groupNormBackward must complete without dereferencing the (absent) grad
 * buffers. Forked via the death-test harness (DeathTest.h) so a
 * missing/misplaced guard's SIGSEGV fails only this test instead of taking
 * down the whole suite. */
void testGroupNormBackwardFrozenFactoryLayerRunsWithoutGradBuffers(void) {
    layer_t *layer = buildFloatGroupNormWithTrainable(TRAINABLE_FALSE);
    bool gradStillNull = layer->config->groupNorm->gamma->grad == NULL &&
                         layer->config->groupNorm->beta->grad == NULL;

    size_t dims[] = {1, 4, 2};
    float xVals[8] = {1.f, -1.f, 1.f, -1.f, 2.f, -2.f, 2.f, -2.f};
    tensor_t *fwdIn = buildFloatTensorND(3, dims, xVals);
    tensor_t *loss = buildFloatTensorND(3, dims, xVals);
    tensor_t *propLoss = buildFloatTensorND(3, dims, NULL);

    ASSERT_EXITS_WITH(0, groupNormBackward(layer, fwdIn, loss, propLoss));

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);
    freeGroupNormLayer(layer);

    TEST_ASSERT_TRUE(gradStillNull);
}

/* #380 PR2 Task 1: propLoss == NULL is a grads-only call -- dgamma/dbeta must
 * be computed exactly as with a real propLoss, and no dx memory may be
 * touched. Duplicates the testBackwardFloatFrozenTwinDxIdenticalGradsZero
 * fixture (dgamma/dbeta both gold-verified nonzero) into two independent
 * twins that are BOTH trainable; only the propLoss argument differs (twin A:
 * real buffer, twin B: literal NULL). Pre-guard, twin B's call
 * dereferences the NULL propLoss and crashes (RED); post-guard,
 * dgamma/dbeta match twin A's byte-for-byte and twin A's dx is
 * non-degenerate. */
void testBackwardFloatNullPropLossComputesGradsOnly(void) {
    size_t B = 1;
    size_t C = 8;
    size_t T = 3;
    size_t G = 2;
    size_t dims[] = {B, C, T};
    tensor_t *fwdIn = buildFloatTensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildFloatTensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLossA = buildFloatTensorND(3, dims, NULL);

    parameter_t *gammaA = buildFloatParam(C, gamma_groupNorm_twoGroups);
    parameter_t *betaA = buildFloatParam(C, beta_groupNorm_twoGroups);
    parameter_t *gammaB = buildFloatParam(C, gamma_groupNorm_twoGroups);
    parameter_t *betaB = buildFloatParam(C, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    groupNormConfig_t cfgA;
    initGroupNormConfig(&cfgA, gammaA, betaA, G, C, 1e-5f, fq, bq);
    layerConfig_t lcfgA;
    layer_t twinA = makeGroupNormLayer(&cfgA, &lcfgA);

    groupNormConfig_t cfgB;
    initGroupNormConfig(&cfgB, gammaB, betaB, G, C, 1e-5f, fq, bq);
    layerConfig_t lcfgB;
    layer_t twinB = makeGroupNormLayer(&cfgB, &lcfgB);

    groupNormBackward(&twinA, fwdIn, loss, propLossA);
    groupNormBackward(&twinB, fwdIn, loss, NULL);

    bool gammaGradIdentical = memcmp(gammaA->grad->data, gammaB->grad->data,
                                     calcNumberOfBytesForData(gammaA->grad->quantization, C)) == 0;
    bool betaGradIdentical = memcmp(betaA->grad->data, betaB->grad->data,
                                    calcNumberOfBytesForData(betaA->grad->quantization, C)) == 0;
    bool propLossANonDegenerate = false;
    for (size_t i = 0; i < B * C * T; i++) {
        if (((float *)propLossA->data)[i] != 0.0f) {
            propLossANonDegenerate = true;
        }
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(betaB);
    freeParameter(gammaB);
    freeParameter(betaA);
    freeParameter(gammaA);
    freeTensor(propLossA);
    freeTensor(loss);
    freeTensor(fwdIn);

    TEST_ASSERT_TRUE_MESSAGE(
        gammaGradIdentical,
        "dgamma must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(
        betaGradIdentical,
        "dbeta must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(propLossANonDegenerate,
                             "twin A's dx must be non-degenerate (nonzero), proving the NULL "
                             "round only skipped dx");
}

/* #380 PR2 Task 1: SYM variant (mirrors
 * testSymBackwardFrozenTwinDxIdenticalGradsUntouched). Duplicates the
 * testSymBackwardTwinSanityTwoGroups fixture into two independent twins that
 * are BOTH trainable; only the propLoss argument differs. No propLoss-scale
 * assertion here: when propLoss is NULL, pass B (dx requant + the propLoss
 * scale refresh) is skipped entirely for twin B -- there is nothing to
 * compare against twin A's refreshed scale. */
void testSymBackwardNullPropLossComputesGradsOnly(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *fwdIn = buildSymInt32TensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildSymInt32TensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLossA = buildSymInt32TensorND(3, dims, NULL);

    parameter_t *gammaA = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *betaA = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);
    parameter_t *gammaB = buildSymParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *betaB = buildSymParamFloatGrad(8, beta_groupNorm_twoGroups);

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);

    groupNormConfig_t cfgA;
    initGroupNormConfig(&cfgA, gammaA, betaA, 2, 8, 1e-5f, fq, bq);
    layerConfig_t lcfgA;
    layer_t twinA = makeGroupNormLayer(&cfgA, &lcfgA);

    groupNormConfig_t cfgB;
    initGroupNormConfig(&cfgB, gammaB, betaB, 2, 8, 1e-5f, fq, bq);
    layerConfig_t lcfgB;
    layer_t twinB = makeGroupNormLayer(&cfgB, &lcfgB);

    layerFunctions[GROUPNORM].backward(&twinA, fwdIn, loss, propLossA);
    layerFunctions[GROUPNORM].backward(&twinB, fwdIn, loss, NULL);

    bool gammaGradIdentical = memcmp(gammaA->grad->data, gammaB->grad->data,
                                     calcNumberOfBytesForData(gammaA->grad->quantization, 8)) == 0;
    bool betaGradIdentical = memcmp(betaA->grad->data, betaB->grad->data,
                                    calcNumberOfBytesForData(betaA->grad->quantization, 8)) == 0;
    bool propLossANonDegenerate = false;
    for (size_t i = 0; i < 24; i++) {
        if (((int32_t *)propLossA->data)[i] != 0) {
            propLossANonDegenerate = true;
        }
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(betaB);
    freeParameter(gammaB);
    freeParameter(betaA);
    freeParameter(gammaA);
    freeTensor(propLossA);
    freeTensor(loss);
    freeTensor(fwdIn);

    TEST_ASSERT_TRUE_MESSAGE(
        gammaGradIdentical,
        "dgamma must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(
        betaGradIdentical,
        "dbeta must be byte-identical between the real-propLoss and NULL-propLoss twins");
    TEST_ASSERT_TRUE_MESSAGE(propLossANonDegenerate,
                             "twin A's dx must be non-degenerate (nonzero), proving the NULL "
                             "round only skipped dx");
}

/* ---- BFP epic PR5 Task 4 (R-N1/R-N2/R-N3): native ARITH_BFP forward ----
 *
 * These replace the PR2 Task 9 guard test (forward rejects ARITH_BFP): the
 * dispatch no longer fails fast on ARITH_BFP, it takes the native arm. That
 * test's sibling assertion -- a BFP forwardQ DERIVES ARITH_BFP through the
 * ordinary config path -- is inherited here: every test below asserts it
 * before running, so the arm cannot be reached by hand-setting alone.
 *
 * GroupNorm has NO gold generator (its SYM coverage is twin-sanity against the
 * FLOAT32 gold, testSymForwardTwinSanityTwoGroups above), and the BFP coverage
 * keeps exactly that shape: the primary oracle is a fake-quant config-pin twin
 * on a GRID-EXACT fixture (bit-equality, not a tolerance), backed by a
 * dequant-vs-FLOAT32-gold sanity bound and the anchor death tests. */

/* Quantizing BFP builders (per-tensor m=8/e=8): the float values go through
 * conversionMatrix[FLOAT32][BFP], so these are for the LOOSE twin-sanity test,
 * where the quantizer is part of what is being sanity-checked. The pin twin
 * uses the codes-taking builders below instead. */
static tensor_t *buildBfpTensorND(size_t numDims, const size_t *dimsIn, const float *vals) {
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    for (size_t i = 0; i < numDims; i++) {
        dims[i] = dimsIn[i];
    }
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, quantizationInitBfp(8, 8, HALF_AWAY), NULL);
    if (vals != NULL) {
        tensorFillFromFloatBuffer(t, (float *)vals, calcNumberOfElementsByShape(shape));
    }
    return t;
}

static parameter_t *buildBfpParamFloatGrad(size_t numChannels, const float *vals) {
    tensor_t *p = buildBfpTensorND(1, (size_t[]){numChannels}, vals);
    return parameterInit(p, gradInitFloat(p, NULL));
}

/* Packed BFP wire from explicit codes + per-group exponents (the sanctioned
 * fixture route: byteConversion pack + exponent memcpy, arithmetic-bfp.md
 * §5.7 inventory). Writing the payload directly instead of quantizing keeps
 * the fixture independent of the quantizer and pins the exponents the kernel
 * borrows. initTensor is right HERE because these are PACKED wires, not the
 * funnel's unpacked scratch form. */
static tensor_t *buildBfpWireWithCodesGn(const size_t *dimsIn, size_t numDims, uint8_t mantissaBits,
                                         uint8_t exponentBits, size_t numGroups, size_t groupSize,
                                         const int32_t *codes, const uint8_t *exponents) {
    quantization_t *q = (groupSize == 0)
                            ? quantizationInitBfp(mantissaBits, exponentBits, HALF_AWAY)
                            : quantizationInitBfpGrouped(mantissaBits, exponentBits, HALF_AWAY,
                                                         numGroups, groupSize);
    size_t *dims = reserveMemory(numDims * sizeof(size_t));
    memcpy(dims, dimsIn, numDims * sizeof(size_t));
    size_t *order = reserveMemory(numDims * sizeof(size_t));
    setOrderOfDimsForNewTensor(numDims, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, numDims, order);
    tensor_t *t = initTensor(shape, q, NULL);
    size_t n = calcNumberOfElementsByTensor(t);
    int32_t packSrc[n]; /* byteConversion takes a mutable source */
    memcpy(packSrc, codes, n * sizeof(int32_t));
    byteConversion((uint8_t *)packSrc, 32, t->data, mantissaBits, n);
    bfpQConfig_t *qc = t->quantization->qConfig;
    memcpy(qc->exponents, exponents, qc->numGroups);
    return t;
}

/* buildBfpParamFloatGrad's codes-taking twin: a BFP-stored gamma/beta with a
 * FLOAT32 grad (the forward never touches grads). */
static parameter_t *buildBfpParamWithCodesGn(size_t numChannels, uint8_t mantissaBits,
                                             uint8_t exponentBits, size_t numGroups,
                                             size_t groupSize, const int32_t *codes,
                                             const uint8_t *exponents) {
    size_t dims[1] = {numChannels};
    tensor_t *p = buildBfpWireWithCodesGn(dims, 1, mantissaBits, exponentBits, numGroups, groupSize,
                                          codes, exponents);
    return parameterInit(p, gradInitFloat(p, NULL));
}

/* GN-A: the GRID-EXACT fixture behind the fake-quant pin twin.
 * Geometry [B=2, C=4, T=2] with numGroups=2 -> cpg=2, N=4 elements per block,
 * K=4 blocks. The input wire is m=8/e=8 GROUPED {8, 2}, i.e. one BFP block per
 * (b, c) channel, so every norm block spans TWO stored exponents and the BFP
 * mean's segment fold has to close a partial at the boundary.
 *
 * Why bit-equality against the dequant-everything FLOAT32 twin is the right
 * assertion here and not a tolerance: with e=8 the bias is 127, so the stored
 * exponents {126, 127, 128} give scales {0.5, 1, 2} and every dequant
 * code * 2^(E-127) is a small multiple of 0.5. Per block the four dequants sum
 * EXACTLY (no rounding, so the fold order cannot matter), mean = sum/4 is an
 * exact division by a power of two, each deviation is an exact multiple of
 * 0.125, each square an exact multiple of 1/64, their sum exact and /4 exact.
 * The BFP segment-fold mean/variance and the FLOAT32 sequential-sum
 * mean/variance are therefore BIT-IDENTICAL, and invSigma, nval and the affine
 * are then the same float32 op sequence over the same bits in both kernels.
 *
 * Per block (x = code * 2^(E-127)):
 *   k0 = (b0, grp0), off 0-3:    1,   3,   4,  -2  -> mean  1.50, var  5.2500
 *   k1 = (b0, grp1), off 4-7:  2.5,-1.5,   4,   2  -> mean  1.75, var  4.0625
 *   k2 = (b1, grp0), off 8-11:  -4,   6,   7,   1  -> mean  2.50, var 19.2500
 *   k3 = (b1, grp1), off 12-15: -5,   2,   3,  -2  -> mean -0.50, var 10.2500
 * The two blocks sharing a grp differ across b (1.50 vs 2.50, 1.75 vs -0.50),
 * which is what makes a mean[grp]-instead-of-mean[k] mutant observable; the
 * gamma/beta codes are non-uniform across channels, which is what makes a
 * gamma[j]-instead-of-gamma[c] mutant observable. */
static const int32_t kGnBfpAXCodes[16] = {1, 3, 2, -1, 5, -3, 4, 2, -2, 3, 7, 1, -5, 2, 6, -4};
static const uint8_t kGnBfpAXExponents[8] = {127, 128, 126, 127, 128, 127, 127, 126};
static const int32_t kGnBfpAGammaCodes[4] = {1, 2, -1, 3};
static const uint8_t kGnBfpAGammaExponents[1] = {127};
static const int32_t kGnBfpABetaCodes[4] = {0, 1, 2, -1};
static const uint8_t kGnBfpABetaExponents[1] = {127};
/* A freshly zero-seeded produced wire (all-zero codes, all-bias exponents), so
 * every emitted code and exponent comes from the OUT_WRITE epilogue and not
 * from leftover fixture state. */
static const int32_t kGnBfpAOutZeroCodes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t kGnBfpAOutZeroExponents[8] = {127, 127, 127, 127, 127, 127, 127, 127};

static tensor_t *buildGnBfpAInput(const size_t *dims) {
    return buildBfpWireWithCodesGn(dims, 3, 8, 8, 8, 2, kGnBfpAXCodes, kGnBfpAXExponents);
}

static tensor_t *buildGnBfpAOutputWire(const size_t *dims) {
    return buildBfpWireWithCodesGn(dims, 3, 8, 8, 8, 2, kGnBfpAOutZeroCodes,
                                   kGnBfpAOutZeroExponents);
}

static parameter_t *buildGnBfpAGamma(void) {
    return buildBfpParamWithCodesGn(4, 8, 8, 1, 0, kGnBfpAGammaCodes, kGnBfpAGammaExponents);
}

static parameter_t *buildGnBfpABeta(void) {
    return buildBfpParamWithCodesGn(4, 8, 8, 1, 0, kGnBfpABetaCodes, kGnBfpABetaExponents);
}

/* The primary oracle: the native ARITH_BFP forward and the fake-quant
 * ARITH_FLOAT32 forward (the funnel dequantizes every BFP operand into float
 * scratch, then groupNormForwardFloat runs) must emit the BYTE-IDENTICAL wire
 * on GN-A. All operands are BFP-STORED, so the native run borrows the
 * fixture's own grids zero-copy and folds on its stored exponents. This pins
 * the per-(b,grp) block stats indexed by k, the per-CHANNEL affine indexed by
 * c = grp*cpg + j/T, the exact dequants and the OUT_WRITE exponent
 * re-derivation -- only the arithmetic slot differs between the two runs. */
void testGroupNormForwardBfpFakeQuantPinTwin(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *outNative = buildGnBfpAOutputWire(dims);
    tensor_t *outFake = buildGnBfpAOutputWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    /* Derived through the ordinary config path -- pins that the flip holds. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.forwardMath.type);
    cfg.forwardMath.roundingMode = HALF_AWAY;
    cfg.outputQ = outNative->quantization;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    groupNormForward(&layer, in, outNative);

    cfg.forwardMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    groupNormForward(&layer, in, outFake);

    size_t payloadBytes = calcNumberOfBytesForData(outNative->quantization, 16);
    bfpQConfig_t *nativeQC = outNative->quantization->qConfig;
    bfpQConfig_t *fakeQC = outFake->quantization->qConfig;
    bool payloadIdentical = memcmp(outNative->data, outFake->data, payloadBytes) == 0;
    bool exponentsIdentical =
        memcmp(nativeQC->exponents, fakeQC->exponents, nativeQC->numGroups) == 0;
    /* Guard against a vacuous pass: an all-zero wire would satisfy both. */
    bool nativeNonDegenerate = false;
    for (size_t i = 0; i < payloadBytes; i++) {
        if (((uint8_t *)outNative->data)[i] != 0) {
            nativeNonDegenerate = true;
        }
    }

    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(outFake);
    freeTensor(outNative);
    freeTensor(in);

    TEST_ASSERT_TRUE_MESSAGE(nativeNonDegenerate,
                             "the native wire must be non-degenerate for the twin to mean "
                             "anything");
    TEST_ASSERT_TRUE_MESSAGE(payloadIdentical,
                             "native ARITH_BFP and fake-quant ARITH_FLOAT32 must emit the same "
                             "packed payload on a grid-exact fixture");
    TEST_ASSERT_TRUE_MESSAGE(exponentsIdentical,
                             "native ARITH_BFP and fake-quant ARITH_FLOAT32 must derive the same "
                             "wire exponents on a grid-exact fixture");
}

/* GN-A's exact dequants (code * 2^(E-127)) as plain floats -- the staged twin
 * feeds these as FLOAT32-stored operands so the funnel has to quantize them
 * into per-tensor BFP scratch at the ANCHOR widths. */
static const float kGnBfpAXValues[16] = {1.f,  3.f, 4.f, -2.f, 2.5f, -1.5f, 4.f, 2.f,
                                         -4.f, 6.f, 7.f, 1.f,  -5.f, 2.f,   3.f, -2.f};
static const float kGnBfpAGammaValues[4] = {1.f, 2.f, -1.f, 3.f};
static const float kGnBfpABetaValues[4] = {0.f, 1.f, 2.f, -1.f};

/* R-N1 staging: FLOAT32-stored operands must be quantized into per-tensor BFP
 * scratch at the layer's own produced-wire widths (outputQ; m=8/e=8 here), for
 * ALL THREE operands. Oracle without a generator: GN-A is chosen so the
 * per-tensor staging grid is EXACT for every operand -- x's absmax is 7, so
 * m=8 pins the stage scale at 2^-4 and every value (all multiples of 0.5) has
 * an integer code in [-128, 127]; likewise gamma (absmax 3, scale 2^-5) and
 * beta (absmax 2, scale 2^-5). Staging therefore reproduces exactly the same
 * dequantized values the all-BFP-stored run folds on, so the two runs must
 * emit the BYTE-IDENTICAL wire. A missing .bfpStage entry makes the funnel
 * fail fast; a wrong staging width (a hardcoded m instead of the anchor's)
 * rounds 2.5 and shifts the payload. */
void testGroupNormForwardBfpStagedFloat32OperandsTwin(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *inBfp = buildGnBfpAInput(dims);
    tensor_t *inFloat = buildFloatTensorND(3, dims, kGnBfpAXValues);
    tensor_t *outBorrowed = buildGnBfpAOutputWire(dims);
    tensor_t *outStaged = buildGnBfpAOutputWire(dims);
    parameter_t *gammaBfp = buildGnBfpAGamma();
    parameter_t *betaBfp = buildGnBfpABeta();
    parameter_t *gammaFloat = buildFloatParam(4, kGnBfpAGammaValues);
    parameter_t *betaFloat = buildFloatParam(4, kGnBfpABetaValues);

    groupNormConfig_t cfgBorrowed;
    initGroupNormConfig(&cfgBorrowed, gammaBfp, betaBfp, 2, 4, 1e-5f, outBorrowed->quantization,
                        outBorrowed->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfgBorrowed.forwardMath.type);
    layerConfig_t lcfgBorrowed;
    layer_t layerBorrowed = makeGroupNormLayer(&cfgBorrowed, &lcfgBorrowed);
    groupNormForward(&layerBorrowed, inBfp, outBorrowed);

    groupNormConfig_t cfgStaged;
    initGroupNormConfig(&cfgStaged, gammaFloat, betaFloat, 2, 4, 1e-5f, outStaged->quantization,
                        outStaged->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfgStaged.forwardMath.type);
    layerConfig_t lcfgStaged;
    layer_t layerStaged = makeGroupNormLayer(&cfgStaged, &lcfgStaged);
    groupNormForward(&layerStaged, inFloat, outStaged);

    size_t payloadBytes = calcNumberOfBytesForData(outBorrowed->quantization, 16);
    bfpQConfig_t *borrowedQC = outBorrowed->quantization->qConfig;
    bfpQConfig_t *stagedQC = outStaged->quantization->qConfig;
    bool payloadIdentical = memcmp(outBorrowed->data, outStaged->data, payloadBytes) == 0;
    bool exponentsIdentical =
        memcmp(borrowedQC->exponents, stagedQC->exponents, borrowedQC->numGroups) == 0;
    bool nonDegenerate = false;
    for (size_t i = 0; i < payloadBytes; i++) {
        if (((uint8_t *)outStaged->data)[i] != 0) {
            nonDegenerate = true;
        }
    }

    freeParameter(betaFloat);
    freeParameter(gammaFloat);
    freeParameter(betaBfp);
    freeParameter(gammaBfp);
    freeTensor(outStaged);
    freeTensor(outBorrowed);
    freeTensor(inFloat);
    freeTensor(inBfp);

    TEST_ASSERT_TRUE_MESSAGE(nonDegenerate, "the staged wire must be non-degenerate");
    TEST_ASSERT_TRUE_MESSAGE(payloadIdentical,
                             "staged FLOAT32 operands must emit the same packed payload as the "
                             "BFP-stored operands they exactly represent");
    TEST_ASSERT_TRUE_MESSAGE(exponentsIdentical,
                             "staged FLOAT32 operands must derive the same wire exponents as the "
                             "BFP-stored operands they exactly represent");
}

/* BFP forward twin-sanity (the testSymForwardTwinSanityTwoGroups idiom
 * verbatim): the native BFP path (exact-dequant stats through the Reduce BFP
 * arms, float normalize + per-channel affine, OUT_WRITE re-pack) must stay
 * within a LOOSE tolerance of the FLOAT32 gold on the same data -- sanity, not
 * gold. Operands are the twoGroups PyTorch fixtures (randn, O(1) spread)
 * quantized per-tensor at m=8, so the round-trip noise is ~2e-2 end to end
 * while an indexing/stats bug shifts values by O(1); 5e-2 keeps >2x headroom.
 * Complements the pin twin: that one is bit-exact but on hand-built codes,
 * this one runs the real quantizer over real data. */
void testGroupNormForwardBfpTwinSanityTwoGroups(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *in = buildBfpTensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *out = buildBfpTensorND(3, dims, NULL);
    parameter_t *gamma = buildBfpParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *beta = buildBfpParamFloatGrad(8, beta_groupNorm_twoGroups);

    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 8, 1e-5f, out->quantization, bq);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.forwardMath.type);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    layerFunctions[GROUPNORM].forward(&layer, in, out);

    /* Per-tensor wire ({1, 0}), so one shared exponent dequantizes everything. */
    int32_t codes[24];
    bfpQConfig_t *outQC = out->quantization->qConfig;
    unpackSignExtend(out->data, outQC->mantissaBits, 0, codes, 24);
    float wireScale = ldexpf(1.0f, (int)outQC->exponents[0] - bfpExponentBias(outQC));
    float deq[24];
    bool nonDegenerate = false;
    for (size_t i = 0; i < 24; i++) {
        deq[i] = (float)codes[i] * wireScale;
        if (codes[i] != 0) {
            nonDegenerate = true;
        }
    }

    freeQuantization(bq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    TEST_ASSERT_TRUE_MESSAGE(nonDegenerate, "the produced wire must be non-degenerate");
    for (size_t i = 0; i < 24; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedForward_groupNorm_twoGroups[i], deq[i]);
    }
}

/* R-N1: norms have no reduction-weight operand, so the staging width anchor is
 * the layer's OWN produced-wire config. A NULL or non-BFP outputQ leaves the
 * ARITH_BFP arm with no width source at all -- fail fast at op entry, not a
 * silent fallback width. Reachable because the userApi factories copy
 * layerQuant_t slots by value, so a pinned ARITH_BFP slot can arrive next to a
 * NULL/FLOAT32 wire config. */
void testGroupNormForwardBfpMissingOutputQAnchorDies(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *out = buildGnBfpAOutputWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.forwardMath.type);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    cfg.outputQ = NULL;
    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    quantization_t *floatQ = quantizationInitFloat();
    cfg.outputQ = floatQ;
    ASSERT_EXITS_WITH_FAILURE(groupNormForward(&layer, in, out));

    freeQuantization(floatQ);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);
}

/* ---- BFP epic PR5 Task 5 (R-N1/R-N4): native ARITH_BFP backward ---- */

/* GN-C: GN-A's x/gamma/beta plus a grid-exact dy for the backward cross-check.
 * dy shares x's wire geometry (m=8/e=8, grouped {8, 2} -- one BFP block per
 * (b, c) channel row) with its own codes/exponents. Every channel's dbeta
 * folds TWO segments (the b=0 and b=1 rows) with DIFFERENT stored exponents,
 * so a wrong-exponent fold mutant shifts the result; every norm block's dy
 * values are non-uniform, so block-reduction mutants stay observable (the
 * uniform-lossGrad vacuity lesson). All dequants code * 2^(E-127) are small
 * multiples of 0.5: the per-channel dy sums and the block sums are EXACT in
 * float32, which is what makes the test-side FLOAT32-twin cross-check below
 * bit-exact (the same exactness argument as GN-A's). */
static const int32_t kGnBfpCDyCodes[16] = {1, -2, 3, 1, -3, 5, 2, -1, -4, 2, 3, -1, 1, 4, -2, 3};
static const uint8_t kGnBfpCDyExponents[8] = {128, 127, 126, 127, 127, 128, 127, 126};
/* The exact dequants, for the FLOAT32-twin expectation run. */
static const float kGnBfpCDyValues[16] = {2.f,  -4.f, 3.f, 1.f,  -1.5f, 2.5f, 2.f,  -1.f,
                                          -4.f, 2.f,  6.f, -2.f, 1.f,   4.f,  -1.f, 1.5f};
/* Native propLoss wires are seeded with exponents NO derivation can produce
 * here (100 => scale 2^-27; dx is O(1)); the expectation twin seeds at 127.
 * DIFFERENT seeds on the two compared wires keep the exponent assertion
 * non-vacuous: if OUT_WRITE never derived exponents, 100 != 127 fails (Task 4
 * review lesson). */
static const uint8_t kGnBfpCPlSeedExponents[8] = {100, 100, 100, 100, 100, 100, 100, 100};

static tensor_t *buildGnBfpCDy(const size_t *dims) {
    return buildBfpWireWithCodesGn(dims, 3, 8, 8, 8, 2, kGnBfpCDyCodes, kGnBfpCDyExponents);
}

static tensor_t *buildGnBfpCPropLossWire(const size_t *dims) {
    return buildBfpWireWithCodesGn(dims, 3, 8, 8, 8, 2, kGnBfpAOutZeroCodes,
                                   kGnBfpCPlSeedExponents);
}

/* The test-side backward twin (proof-ladder deviation, epic PR5 brief):
 * GroupNorm has no gold generator, and the forward's config-pin idiom cannot
 * provide a backward oracle -- the FLOAT32 backward arm hard-rejects
 * BFP-stored wires (its raw-cast guards) instead of funneling them, so
 * flipping only the arithmetic pin cannot run the SAME tensors through both
 * arms. The expectation is therefore computed HERE: the FLOAT32 backward over
 * exact-dequant FLOAT32 twins of GN-A's x/gamma/beta and GN-C's dy. On this
 * grid-exact fixture the native BFP kernels and the FLOAT32 kernel perform
 * the same float32 op sequence over the same bits (stats bit-identity is
 * pinned by the forward pin twin; the dbeta segment fold differs only in
 * summation order, which cannot matter because every partial sum is an exact
 * small dyadic), so dgamma/dbeta must be memory-equal and the dx raw
 * bit-equal BEFORE the OUT_WRITE pack -- and that pack is the same
 * conversionMatrix FLOAT32->BFP diagonal convertTensor applies to the
 * expectation (both under HALF_AWAY). expDxFloat is a caller-owned [2,4,2]
 * FLOAT32 tensor the twin's dx lands in. */
static void gnBfpCExpectedBackward(float *expDgamma, float *expDbeta, tensor_t *expDxFloat) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *x = buildFloatTensorND(3, dims, kGnBfpAXValues);
    tensor_t *dy = buildFloatTensorND(3, dims, kGnBfpCDyValues);
    parameter_t *gamma = buildFloatParam(4, kGnBfpAGammaValues);
    parameter_t *beta = buildFloatParam(4, kGnBfpABetaValues);
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);
    groupNormBackward(&layer, x, dy, expDxFloat);
    memcpy(expDgamma, gamma->grad->data, 4 * sizeof(float));
    memcpy(expDbeta, beta->grad->data, 4 * sizeof(float));
    freeQuantization(bq);
    freeQuantization(fq);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(dy);
    freeTensor(x);
}

/* BFP backward twin-sanity (the testSymBackwardTwinSanityTwoGroups idiom):
 * quantize the float gold fixture's x/dy/gamma/beta per-tensor at m=8, run
 * the native ARITH_BFP backward, and require the dequantized dx and the
 * FLOAT32-default grads to track the FLOAT32 gold within a LOOSE 5e-2 --
 * sanity over the real quantizer on real data; the bit-exact oracle is the
 * cross-check below. */
void testGroupNormBackwardBfpTwinSanityTwoGroups(void) {
    size_t dims[] = {1, 8, 3};
    tensor_t *fwdIn = buildBfpTensorND(3, dims, input_groupNorm_twoGroups);
    tensor_t *loss = buildBfpTensorND(3, dims, lossGrad_groupNorm_twoGroups);
    tensor_t *propLoss = buildBfpTensorND(3, dims, NULL);
    parameter_t *gamma = buildBfpParamFloatGrad(8, gamma_groupNorm_twoGroups);
    parameter_t *beta = buildBfpParamFloatGrad(8, beta_groupNorm_twoGroups);

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 8, 1e-5f, propLoss->quantization,
                        propLoss->quantization);
    /* Derived through the ordinary config path -- pins that the flip holds. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    layerFunctions[GROUPNORM].backward(&layer, fwdIn, loss, propLoss);

    /* Per-tensor wire ({1, 0}), so one shared exponent dequantizes everything. */
    int32_t codes[24];
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    unpackSignExtend(propLoss->data, plQC->mantissaBits, 0, codes, 24);
    float wireScale = ldexpf(1.0f, (int)plQC->exponents[0] - bfpExponentBias(plQC));
    float dxDeq[24];
    bool nonDegenerate = false;
    for (size_t i = 0; i < 24; i++) {
        dxDeq[i] = (float)codes[i] * wireScale;
        if (codes[i] != 0) {
            nonDegenerate = true;
        }
    }
    float dg[8];
    float db[8];
    memcpy(dg, gamma->grad->data, sizeof(dg));
    memcpy(db, beta->grad->data, sizeof(db));

    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    TEST_ASSERT_TRUE_MESSAGE(nonDegenerate, "the produced dx wire must be non-degenerate");
    for (size_t i = 0; i < 24; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedPropLoss_groupNorm_twoGroups[i], dxDeq[i]);
    }
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedDgamma_groupNorm_twoGroups[i], dg[i]);
        TEST_ASSERT_FLOAT_WITHIN(5e-2f, expectedDbeta_groupNorm_twoGroups[i], db[i]);
    }
}

/* The primary backward oracle: native ARITH_BFP backward on GN-A + GN-C vs
 * the test-side FLOAT32 twin (gnBfpCExpectedBackward). dgamma/dbeta land in
 * FLOAT32 grads via the ACC epilogue (zero-init grad + float raw = plain
 * float32 addition) and are asserted memory-equal; dx goes through the
 * OUT_WRITE pack at the propLoss geometry and is asserted payload+exponent
 * equal against convertTensor over the twin's float dx. */
void testGroupNormBackwardBfpFakeQuantPinCrossCheck(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    /* Derived through the ordinary config path -- pins that the flip holds. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    cfg.propLossMath.roundingMode = HALF_AWAY;
    cfg.propLossQ = propLoss->quantization;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    groupNormBackward(&layer, in, loss, propLoss);

    float expDgamma[4];
    float expDbeta[4];
    tensor_t *expDxFloat = buildFloatTensorND(3, dims, NULL);
    gnBfpCExpectedBackward(expDgamma, expDbeta, expDxFloat);
    tensor_t *expDxWire = buildGnBfpAOutputWire(dims);
    convertTensor(expDxFloat, expDxWire);

    size_t payloadBytes = calcNumberOfBytesForData(propLoss->quantization, 16);
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    bfpQConfig_t *expQC = expDxWire->quantization->qConfig;
    bool payloadIdentical = memcmp(propLoss->data, expDxWire->data, payloadBytes) == 0;
    bool exponentsIdentical = memcmp(plQC->exponents, expQC->exponents, plQC->numGroups) == 0;
    bool exponentsRederived = memcmp(plQC->exponents, kGnBfpCPlSeedExponents, plQC->numGroups) != 0;
    bool nonDegenerate = false;
    for (size_t i = 0; i < payloadBytes; i++) {
        if (((uint8_t *)propLoss->data)[i] != 0) {
            nonDegenerate = true;
        }
    }
    float gotDgamma[4];
    float gotDbeta[4];
    memcpy(gotDgamma, gamma->grad->data, sizeof(gotDgamma));
    memcpy(gotDbeta, beta->grad->data, sizeof(gotDbeta));

    freeTensor(expDxWire);
    freeTensor(expDxFloat);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_TRUE_MESSAGE(nonDegenerate,
                             "the native dx wire must be non-degenerate for the cross-check to "
                             "mean anything");
    TEST_ASSERT_TRUE_MESSAGE(exponentsRederived,
                             "the OUT_WRITE epilogue must rewrite the seeded propLoss exponents");
    TEST_ASSERT_TRUE_MESSAGE(payloadIdentical,
                             "native ARITH_BFP dx must match the FLOAT32-twin expectation packed "
                             "through convertTensor on a grid-exact fixture");
    TEST_ASSERT_TRUE_MESSAGE(exponentsIdentical,
                             "native ARITH_BFP dx exponents must match the FLOAT32-twin "
                             "expectation's derived exponents");
    TEST_ASSERT_EQUAL_MEMORY(expDgamma, gotDgamma, 4 * sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(expDbeta, gotDbeta, 4 * sizeof(float));
}

/* ACC semantics across microbatch calls: a second identical backward must ADD
 * the same float increment again. inc + inc is exact in float32 (an exponent
 * bump, no rounding), so the doubled expectation is asserted byte-exact. Also
 * the sensitized probe for a dropped dgamma/dbeta raw memset: the second
 * call's Phase-2 raw region has just been scribbled by the first call's op
 * sequence. */
void testGroupNormBackwardBfpGradsAccumulateAcrossCalls(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    cfg.propLossMath.roundingMode = HALF_AWAY;
    cfg.propLossQ = propLoss->quantization;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    groupNormBackward(&layer, in, loss, propLoss);
    groupNormBackward(&layer, in, loss, propLoss);

    float gotDgamma[4];
    float gotDbeta[4];
    memcpy(gotDgamma, gamma->grad->data, sizeof(gotDgamma));
    memcpy(gotDbeta, beta->grad->data, sizeof(gotDbeta));

    float expDgamma[4];
    float expDbeta[4];
    tensor_t *expDxFloat = buildFloatTensorND(3, dims, NULL);
    gnBfpCExpectedBackward(expDgamma, expDbeta, expDxFloat);

    freeTensor(expDxFloat);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    float expDgamma2[4];
    float expDbeta2[4];
    for (size_t i = 0; i < 4; i++) {
        expDgamma2[i] = expDgamma[i] + expDgamma[i];
        expDbeta2[i] = expDbeta[i] + expDbeta[i];
    }
    TEST_ASSERT_EQUAL_MEMORY(expDgamma2, gotDgamma, 4 * sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(expDbeta2, gotDbeta, 4 * sizeof(float));
}

/* propLoss == NULL (#380 PR2): grads-only call -- the dx op is skipped, the
 * two grad ops still run and land the exact twin expectation. propLossQ stays
 * the init-derived BFP config: with all-BFP operands the anchor is only
 * type-checked (nothing stages), but R-N1 still requires it. */
void testGroupNormBackwardBfpNullPropLossComputesGradsOnly(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    cfg.propLossMath.roundingMode = HALF_AWAY;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    groupNormBackward(&layer, in, loss, NULL);

    float gotDgamma[4];
    float gotDbeta[4];
    memcpy(gotDgamma, gamma->grad->data, sizeof(gotDgamma));
    memcpy(gotDbeta, beta->grad->data, sizeof(gotDbeta));

    float expDgamma[4];
    float expDbeta[4];
    tensor_t *expDxFloat = buildFloatTensorND(3, dims, NULL);
    gnBfpCExpectedBackward(expDgamma, expDbeta, expDxFloat);

    freeTensor(expDxFloat);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_EQUAL_MEMORY(expDgamma, gotDgamma, 4 * sizeof(float));
    TEST_ASSERT_EQUAL_MEMORY(expDbeta, gotDbeta, 4 * sizeof(float));
}

/* frozen (#380): the two grad ops are skipped entirely -- gamma/beta carry NO
 * grad tensors here (parameterInit(p, NULL), the factory-elision shape), so
 * any grad-op dispatch would dereference NULL. The dx op still runs and must
 * hit the same cross-check expectation. */
void testGroupNormBackwardBfpFrozenSkipsGrads(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    size_t gdims[1] = {4};
    tensor_t *gammaT =
        buildBfpWireWithCodesGn(gdims, 1, 8, 8, 1, 0, kGnBfpAGammaCodes, kGnBfpAGammaExponents);
    parameter_t *gamma = parameterInit(gammaT, NULL);
    tensor_t *betaT =
        buildBfpWireWithCodesGn(gdims, 1, 8, 8, 1, 0, kGnBfpABetaCodes, kGnBfpABetaExponents);
    parameter_t *beta = parameterInit(betaT, NULL);

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    cfg.propLossMath.roundingMode = HALF_AWAY;
    cfg.propLossQ = propLoss->quantization;
    cfg.frozen = true;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    groupNormBackward(&layer, in, loss, propLoss);

    float expDgamma[4];
    float expDbeta[4];
    tensor_t *expDxFloat = buildFloatTensorND(3, dims, NULL);
    gnBfpCExpectedBackward(expDgamma, expDbeta, expDxFloat);
    tensor_t *expDxWire = buildGnBfpAOutputWire(dims);
    convertTensor(expDxFloat, expDxWire);

    size_t payloadBytes = calcNumberOfBytesForData(propLoss->quantization, 16);
    bfpQConfig_t *plQC = propLoss->quantization->qConfig;
    bfpQConfig_t *expQC = expDxWire->quantization->qConfig;
    bool payloadIdentical = memcmp(propLoss->data, expDxWire->data, payloadBytes) == 0;
    bool exponentsIdentical = memcmp(plQC->exponents, expQC->exponents, plQC->numGroups) == 0;

    freeTensor(expDxWire);
    freeTensor(expDxFloat);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);

    TEST_ASSERT_TRUE_MESSAGE(payloadIdentical,
                             "frozen backward's dx must still match the cross-check expectation");
    TEST_ASSERT_TRUE_MESSAGE(exponentsIdentical,
                             "frozen backward's dx exponents must still match the cross-check "
                             "expectation");
}

/* The dx kernel derives its walk from forwardInput's geometry ((b, grp) base +
 * j) but indexes loss at those offsets, so a loss wire shorter than the
 * forward input reads outside the funnel's unpacked scratch. Unfrozen the
 * dgamma kernel's own loss-count gate runs first and covers it incidentally;
 * FROZEN skips dgamma, which makes the dx gate the SOLE catcher -- that is the
 * configuration pinned here. The short wire is per-tensor {1, 0}, so its own
 * grid validates at its own count and only the cross-count gate can reject
 * it. */
void testGroupNormBackwardBfpFrozenShortLossDies(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    /* Half the elements of the forward input: one batch row instead of two. */
    size_t shortDims[3] = {1, 4, 2};
    tensor_t *shortLoss = buildBfpWireWithCodesGn(
        shortDims, 3, 8, 8, 1, 0, (int32_t[]){1, -2, 3, 1, -3, 5, 2, -1}, (uint8_t[]){127});
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    size_t gdims[1] = {4};
    tensor_t *gammaT =
        buildBfpWireWithCodesGn(gdims, 1, 8, 8, 1, 0, kGnBfpAGammaCodes, kGnBfpAGammaExponents);
    parameter_t *gamma = parameterInit(gammaT, NULL);
    tensor_t *betaT =
        buildBfpWireWithCodesGn(gdims, 1, 8, 8, 1, 0, kGnBfpABetaCodes, kGnBfpABetaExponents);
    parameter_t *beta = parameterInit(betaT, NULL);

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    cfg.propLossMath.roundingMode = HALF_AWAY;
    cfg.propLossQ = propLoss->quantization;
    cfg.frozen = true;
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, in, shortLoss, propLoss));

    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(shortLoss);
    freeTensor(in);
}

/* R-N1's backward half: propLossQ anchors ALL THREE backward ops' staging, so
 * a NULL anchor under ARITH_BFP dies at arm entry -- and it must die even
 * when the propLoss TENSOR is NULL (the grad ops still stage at it). */
void testGroupNormBackwardBfpMissingPropLossQAnchorDies(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, cfg.propLossMath.type);
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    cfg.propLossQ = NULL;
    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, in, loss, propLoss));
    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, in, loss, NULL));

    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);
}

/* R-P7d "narrowed, not removed": pinning propLossMath to ARITH_FLOAT32 over
 * BFP-stored wires still dies in the float arm's raw-cast guards -- the BFP
 * arm did not open a silent fall-through for mismatched storage. */
void testGroupNormBackwardFloat32PinnedStillRejectsBfpWires(void) {
    size_t dims[3] = {2, 4, 2};
    tensor_t *in = buildGnBfpAInput(dims);
    tensor_t *loss = buildGnBfpCDy(dims);
    tensor_t *propLoss = buildGnBfpCPropLossWire(dims);
    parameter_t *gamma = buildGnBfpAGamma();
    parameter_t *beta = buildGnBfpABeta();

    groupNormConfig_t cfg;
    initGroupNormConfig(&cfg, gamma, beta, 2, 4, 1e-5f, in->quantization, in->quantization);
    cfg.propLossMath = (arithmetic_t){.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    layerConfig_t lcfg;
    layer_t layer = makeGroupNormLayer(&cfg, &lcfg);

    ASSERT_EXITS_WITH_FAILURE(groupNormBackward(&layer, in, loss, propLoss));

    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(in);
}

/* ---- BFP epic PR5 Task 6: factory gates (coherence rules + param geometry) ---- */

/* Dequant a packed BFP tensor through its OWN grid (code * 2^(E - bias)). The
 * assertion form for FACTORY-built params, whose codes and exponents come out
 * of the quantizer rather than a fixture, so nothing may be hardcoded. */
static void dequantBfpTensorGn(tensor_t *t, float *out) {
    bfpQConfig_t *qc = t->quantization->qConfig;
    size_t n = calcNumberOfElementsByTensor(t);
    int32_t codes[n];
    unpackSignExtend(t->data, qc->mantissaBits, 0, codes, n);
    size_t gsz = (qc->groupSize == 0) ? n : qc->groupSize;
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)codes[i] * bfpGroupScale(qc, i / gsz);
    }
}

/* Reference GroupNorm over already-dequantized floats with gamma=1 / beta=0
 * (the factory's own init constants, so this doubles as an end-to-end check
 * that the constant fills landed on the BFP grid). Every norm block is
 * CONTIGUOUS in the [B][C][T] layout, so K blocks of blockLen elements is the
 * whole blocking. */
static void referenceGroupNormOnesZeros(const float *xs, size_t K, size_t blockLen, float eps,
                                        float *out) {
    for (size_t k = 0; k < K; k++) {
        float mean = 0.f;
        for (size_t j = 0; j < blockLen; j++) {
            mean += xs[k * blockLen + j];
        }
        mean /= (float)blockLen;
        float var = 0.f;
        for (size_t j = 0; j < blockLen; j++) {
            float d = xs[k * blockLen + j] - mean;
            var += d * d;
        }
        var /= (float)blockLen;
        float inv = 1.f / sqrtf(var + eps);
        for (size_t j = 0; j < blockLen; j++) {
            out[k * blockLen + j] = (xs[k * blockLen + j] - mean) * inv;
        }
    }
}

/* The capstone shape Task 7 depends on: ONE grouped BFP config through
 * layerQuantInitUniform derives all four math slots ARITH_BFP and aliases every
 * storage slot, and the factory must accept it end to end. Pins the derived
 * param geometry ({2, 2} from 4 channels / groupSize 2), the EXACT all-ones
 * gamma (1.0 = 64 * 2^-6 at m = 8 -- a grid point, so the dequant is exact),
 * beta's absMax == 0 zero-state (codes 0, every stored exponent = bias), and a
 * native forward over a BFP wire. */
void testFactoryUniformBfpProfileBuildsAndForwards(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 0.0f};

    quantization_t *bfpQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpQ);

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    tensor_t *gammaT = cfg->gamma->param;
    tensor_t *betaT = cfg->beta->param;
    bool gammaBfp = (gammaT->quantization->type == BFP);
    bool betaBfp = (betaT->quantization->type == BFP);
    bfpQConfig_t *gQC = gammaT->quantization->qConfig;
    bfpQConfig_t *bQC = betaT->quantization->qConfig;
    size_t gammaGroups = gQC->numGroups;
    size_t gammaGroupSize = gQC->groupSize;

    float gammaVals[4];
    float betaVals[4];
    dequantBfpTensorGn(gammaT, gammaVals);
    dequantBfpTensorGn(betaT, betaVals);
    bool betaZeroState = true;
    for (size_t g = 0; g < bQC->numGroups; g++) {
        if (bQC->exponents[g] != (uint8_t)bfpExponentBias(bQC)) {
            betaZeroState = false;
        }
    }

    /* Forward over BFP wires: [B=1, C=4, T=2] with numGroups=2 -> 2 contiguous
     * blocks of 4. */
    size_t dims[3] = {1, 4, 2};
    tensor_t *in = buildBfpWireWithCodesGn(dims, 3, 8, 8, 1, 0,
                                           (int32_t[]){1, 3, 2, -1, 5, -3, 4, 2}, (uint8_t[]){127});
    tensor_t *out = buildBfpWireWithCodesGn(dims, 3, 8, 8, 1, 0,
                                            (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, (uint8_t[]){127});
    groupNormForward(layer, in, out);

    bool outBfp = (out->quantization->type == BFP);
    float xs[8];
    float ys[8];
    dequantBfpTensorGn(in, xs);
    dequantBfpTensorGn(out, ys);
    float expected[8];
    referenceGroupNormOnesZeros(xs, 2, 4, 1e-5f, expected);
    bool outNonDegenerate = false;
    for (size_t i = 0; i < 8; i++) {
        if (ys[i] != 0.f) {
            outNonDegenerate = true;
        }
    }

    freeTensor(out);
    freeTensor(in);
    freeGroupNormLayer(layer);
    freeQuantization(bfpQ);

    TEST_ASSERT_TRUE_MESSAGE(gammaBfp, "uniform BFP profile must give gamma BFP storage");
    TEST_ASSERT_TRUE_MESSAGE(betaBfp, "uniform BFP profile must give beta BFP storage");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, gammaGroups, "gamma numGroups is DERIVED from 4 / 2");
    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, gammaGroupSize, "gamma groupSize comes from the template");
    for (size_t i = 0; i < 4; i++) {
        /* BIT-exact, not a tolerance: 1.0 is a BFP grid point at m = 8
         * (code 64, stored exponent bias - 6) and 0.0 is the zero-state. */
        TEST_ASSERT_TRUE_MESSAGE(gammaVals[i] == 1.f, "gamma must dequant to EXACTLY 1.0");
        TEST_ASSERT_TRUE_MESSAGE(betaVals[i] == 0.f, "beta must dequant to EXACTLY 0.0");
    }
    TEST_ASSERT_TRUE_MESSAGE(betaZeroState, "all-zero beta must hit the absMax == 0 zero-state");
    TEST_ASSERT_TRUE(outBfp);
    TEST_ASSERT_TRUE_MESSAGE(outNonDegenerate, "the produced wire must not be all-zero");
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.02f, expected[i], ys[i]);
    }
}

/* R-N6: a grouped BFP storage template's numGroups is a shape-agnostic guess
 * (one layerQuant_t is shared across a whole model), so only its groupSize is
 * honored and numGroups is derived from THIS parameter's channel count.
 * Passing the template's numGroups through would die at initTensor's attach
 * validation instead. */
void testFactoryBfpGammaGeometryDerivedFromParamLength(void) {
    /* (a) a {4, 3} template (a 12-element guess) against a 6-channel param. */
    groupNormInit_t init6 = {.numGroups = 2, .numChannels = 6, .eps = 1e-5f};
    quantization_t *tmplA = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 4, 3);
    layerQuant_t lqA;
    layerQuantInitUniform(&lqA, tmplA);
    layer_t *layerA = groupNormLayerInit(&init6, &lqA);
    bfpQConfig_t *gA = layerA->config->groupNorm->gamma->param->quantization->qConfig;
    size_t groupsA = gA->numGroups;
    size_t sizeA = gA->groupSize;
    freeGroupNormLayer(layerA);
    freeQuantization(tmplA);

    /* (b) groupSize == the param's channel count -> the {1, N} shape is
     * normalized to the per-tensor sentinel {1, 0} (initBfpQConfigGrouped
     * rejects {1, N} outright). */
    groupNormInit_t init4 = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};
    quantization_t *tmplB = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 3, 4);
    layerQuant_t lqB;
    layerQuantInitUniform(&lqB, tmplB);
    layer_t *layerB = groupNormLayerInit(&init4, &lqB);
    bfpQConfig_t *gB = layerB->config->groupNorm->gamma->param->quantization->qConfig;
    size_t groupsB = gB->numGroups;
    size_t sizeB = gB->groupSize;
    freeGroupNormLayer(layerB);
    freeQuantization(tmplB);

    /* (c) a groupSize that does not divide the channel count has no derivable
     * geometry at all -- guided death in the factory. */
    quantization_t *tmplC = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 5);
    layerQuant_t lqC;
    layerQuantInitUniform(&lqC, tmplC);
    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init6, &lqC));
    freeQuantization(tmplC);

    TEST_ASSERT_EQUAL_UINT_MESSAGE(2, groupsA, "numGroups must be derived (6 / 3), not the guess");
    TEST_ASSERT_EQUAL_UINT(3, sizeA);
    TEST_ASSERT_EQUAL_UINT_MESSAGE(1, groupsB, "groupSize == N must normalize to per-tensor");
    TEST_ASSERT_EQUAL_UINT(0, sizeB);
}

/* Rule 3/6's FLOAT32 half: BFP wires with FLOAT32-stored gamma/beta is a legal
 * profile -- the params stage into BFP scratch at the wire anchor. Forward AND
 * backward must both run through the factory-built layer. */
void testFactoryWiresOnlyBfpProfileBuilds(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    quantization_t *floatQ = quantizationInitFloat();
    arithmetic_t bfpMath = {.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerQuant_t lq = {.forwardMath = bfpMath,
                       .weightGradMath = bfpMath,
                       .biasGradMath = bfpMath,
                       .propLossMath = bfpMath,
                       .outputQ = bfpQ,
                       .propLossQ = bfpQ,
                       .weightStorage = floatQ,
                       .biasStorage = floatQ,
                       .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                       .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE};

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;
    bool gammaFloat = (cfg->gamma->param->quantization->type == FLOAT32);
    bool betaFloat = (cfg->beta->param->quantization->type == FLOAT32);

    size_t dims[3] = {1, 4, 2};
    tensor_t *in = buildBfpWireWithCodesGn(dims, 3, 8, 8, 1, 0,
                                           (int32_t[]){1, 3, 2, -1, 5, -3, 4, 2}, (uint8_t[]){127});
    tensor_t *out = buildBfpWireWithCodesGn(dims, 3, 8, 8, 1, 0,
                                            (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0}, (uint8_t[]){127});
    tensor_t *dy = buildBfpWireWithCodesGn(
        dims, 3, 8, 8, 1, 0, (int32_t[]){1, -2, 3, 1, -3, 5, 2, -1}, (uint8_t[]){127});
    tensor_t *dx = buildBfpWireWithCodesGn(dims, 3, 8, 8, 1, 0, (int32_t[]){0, 0, 0, 0, 0, 0, 0, 0},
                                           (uint8_t[]){127});

    groupNormForward(layer, in, out);
    groupNormBackward(layer, in, dy, dx);

    float xs[8];
    float ys[8];
    float dxs[8];
    dequantBfpTensorGn(in, xs);
    dequantBfpTensorGn(out, ys);
    dequantBfpTensorGn(dx, dxs);
    float expected[8];
    referenceGroupNormOnesZeros(xs, 2, 4, 1e-5f, expected);
    bool dxNonDegenerate = false;
    bool gradsNonZero = false;
    for (size_t i = 0; i < 8; i++) {
        if (dxs[i] != 0.f) {
            dxNonDegenerate = true;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        if (((float *)cfg->gamma->grad->data)[i] != 0.f ||
            ((float *)cfg->beta->grad->data)[i] != 0.f) {
            gradsNonZero = true;
        }
    }

    freeTensor(dx);
    freeTensor(dy);
    freeTensor(out);
    freeTensor(in);
    freeGroupNormLayer(layer);
    freeQuantization(floatQ);
    freeQuantization(bfpQ);

    TEST_ASSERT_TRUE(gammaFloat);
    TEST_ASSERT_TRUE(betaFloat);
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.02f, expected[i], ys[i]);
    }
    TEST_ASSERT_TRUE_MESSAGE(dxNonDegenerate, "the BFP backward must write a non-zero dx wire");
    TEST_ASSERT_TRUE_MESSAGE(gradsNonZero, "the BFP backward must accumulate non-zero grads");
}

/* Rule 3: the BFP forward reads gamma/beta as BFP scratch (borrowed or
 * staged from FLOAT32); SYM_INT32 mantissas have no route into it. */
void testFactoryRejectsSymGammaUnderBfpForward(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    arithmetic_t bfpMath = {.type = ARITH_BFP, .roundingMode = HALF_AWAY};
    layerQuant_t lq = {.forwardMath = bfpMath,
                       .propLossMath = bfpMath,
                       .outputQ = bfpQ,
                       .propLossQ = bfpQ,
                       .weightStorage = symQ,
                       .biasStorage = bfpQ,
                       .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                       .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE};

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lq));

    freeQuantization(symQ);
    freeQuantization(bfpQ);
}

/* Rule 4 (the DELIBERATE asymmetry against the GEMM family): "FLOAT32 math over
 * BFP params" is a trap for the norms -- their FLOAT32 backward raw-casts gamma
 * and rejects non-FLOAT32, so the profile would forward fine and die at the
 * first backward. Unconstructible by design. */
void testFactoryRejectsFloat32MathOverBfpParams(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    quantization_t *floatQ = quantizationInitFloat();
    arithmetic_t floatMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY};
    layerQuant_t lq = {.forwardMath = floatMath,
                       .propLossMath = floatMath,
                       .outputQ = floatQ,
                       .propLossQ = floatQ,
                       .weightStorage = bfpQ,
                       .biasStorage = bfpQ};

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lq));

    /* The variant ONLY rule 4 catches: a native ARITH_BFP backward over BFP
     * wires satisfies rules 6 and 7, so nothing but rule 4 stops the FLOAT32
     * FORWARD from raw-casting the same BFP gamma. */
    layerQuant_t lqBfpBackward = {.forwardMath = floatMath,
                                  .propLossMath = {.type = ARITH_BFP, .roundingMode = HALF_AWAY},
                                  .outputQ = bfpQ,
                                  .propLossQ = bfpQ,
                                  .weightStorage = bfpQ,
                                  .biasStorage = bfpQ,
                                  .weightGradAccMode = OUT_ACC_DYNAMIC_RESCALE,
                                  .biasGradAccMode = OUT_ACC_DYNAMIC_RESCALE};

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lqBfpBackward));

    freeQuantization(floatQ);
    freeQuantization(bfpQ);
}

/* Rule 7 (the ONLY rule that catches this one): FLOAT32-stored params under a
 * native BFP forward is fine, but pinning the BACKWARD to ARITH_FLOAT32 over
 * BFP wires hands the raw-casting float backward a packed dx wire. Rules 3, 4
 * and 6 all pass here -- deleting rule 7 makes this test fail. */
void testFactoryRejectsFloat32BackwardOverBfpWires(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = {.type = ARITH_BFP, .roundingMode = HALF_AWAY},
                       .propLossMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
                       .outputQ = bfpQ,
                       .propLossQ = bfpQ,
                       .weightStorage = floatQ,
                       .biasStorage = floatQ};

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lq));

    freeQuantization(floatQ);
    freeQuantization(bfpQ);
}

/* Rule 8: the factory adds NO grad gate of its own -- a grouped BFP grad
 * template dies in gradInit's existing per-tensor-only carrier gate. This test
 * pins that gate THROUGH the factory path. */
void testFactoryRejectsGroupedBfpGradTemplate(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpQ);
    quantization_t *groupedGradQ = quantizationInitBfpGrouped(8, 8, SR_HALF_AWAY, 2, 2);
    lq.weightGradStorage = groupedGradQ;

    ASSERT_EXITS_WITH_FAILURE(groupNormLayerInit(&init, &lq));

    freeQuantization(groupedGradQ);
    freeQuantization(bfpQ);
}

/* Rule 8's positive half: a PER-TENSOR BFP grad template flows straight
 * through, keeping its OWN widths (m = 6 here, not the param's 8) and landing
 * in the fresh zero state (all-zero codes, every exponent = bias). */
void testFactoryBfpGradStorageBuildsPerTensor(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpQ);
    quantization_t *gradQ = quantizationInitBfp(6, 8, HALF_AWAY);
    lq.weightGradStorage = gradQ;
    lq.biasGradStorage = gradQ;

    layer_t *layer = groupNormLayerInit(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    bool gammaGradBfp = (cfg->gamma->grad->quantization->type == BFP);
    bool betaGradBfp = (cfg->beta->grad->quantization->type == BFP);
    bfpQConfig_t *ggQC = cfg->gamma->grad->quantization->qConfig;
    size_t gradGroups = ggQC->numGroups;
    size_t gradGroupSize = ggQC->groupSize;
    uint8_t gradMantissaBits = ggQC->mantissaBits;
    bool gradZeroState = (ggQC->exponents[0] == (uint8_t)bfpExponentBias(ggQC));
    float gradVals[4];
    dequantBfpTensorGn(cfg->gamma->grad, gradVals);

    freeGroupNormLayer(layer);
    freeQuantization(gradQ);
    freeQuantization(bfpQ);

    TEST_ASSERT_TRUE(gammaGradBfp);
    TEST_ASSERT_TRUE(betaGradBfp);
    TEST_ASSERT_EQUAL_UINT(1, gradGroups);
    TEST_ASSERT_EQUAL_UINT(0, gradGroupSize);
    TEST_ASSERT_EQUAL_UINT8_MESSAGE(6, gradMantissaBits,
                                    "the grad template's OWN width must survive the clone");
    TEST_ASSERT_TRUE_MESSAGE(gradZeroState, "a fresh BFP grad must carry the bias exponent");
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.f, gradVals[i]);
    }
}

/* The Owning twin over a BFP profile: deepCopyQuantization's BFP arm must
 * deep-copy the heap exponents array, so dropping the caller's config right
 * after init leaves the layer intact (and freeing it later is not a
 * double-free). */
void testFactoryOwningBfpDeepCopiesQuantizations(void) {
    groupNormInit_t init = {.numGroups = 2, .numChannels = 4, .eps = 1e-5f};

    quantization_t *bfpQ = quantizationInitBfpGrouped(8, 8, HALF_AWAY, 2, 2);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, bfpQ);

    layer_t *layer = groupNormLayerInitOwning(&init, &lq);
    groupNormConfig_t *cfg = layer->config->groupNorm;

    bool outIsCopy = (cfg->outputQ != bfpQ);
    bool plIsCopy = (cfg->propLossQ != bfpQ);
    bool outBfp = (cfg->outputQ->type == BFP);
    bool owns = cfg->ownsQuantizations;
    bfpQConfig_t *srcQC = bfpQ->qConfig;
    bfpQConfig_t *outQC = cfg->outputQ->qConfig;
    bool exponentsDeepCopied = (outQC->exponents != srcQC->exponents);
    size_t copiedGroups = outQC->numGroups;

    /* Caller drops its config IMMEDIATELY -- the layer holds copies. */
    freeQuantization(bfpQ);

    /* Params still readable afterwards: their storage quant came through
     * groupNormParamQLike, not from the caller's allocation. */
    float gammaVals[4];
    dequantBfpTensorGn(cfg->gamma->param, gammaVals);

    freeGroupNormLayer(layer);

    TEST_ASSERT_TRUE(outIsCopy);
    TEST_ASSERT_TRUE(plIsCopy);
    TEST_ASSERT_TRUE(outBfp);
    TEST_ASSERT_TRUE(owns);
    TEST_ASSERT_TRUE_MESSAGE(exponentsDeepCopied, "the BFP exponents array must not be aliased");
    TEST_ASSERT_EQUAL_UINT(2, copiedGroups);
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(1.f, gammaVals[i]);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConfigStructIsPopulated);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testGoldForwardSingleGroup);
    RUN_TEST(testGoldForwardTwoGroups);
    RUN_TEST(testGoldForwardBatch2ThreeGroups);
    RUN_TEST(testGoldForwardGroupEqualsChannels);
    RUN_TEST(testVtableGoldForwardTwoGroups);
    RUN_TEST(testForwardFloatSmallVarianceEpsInsideSqrt);
    RUN_TEST(testForwardRejectsWrongRank);
    RUN_TEST(testForwardRejectsWrongChannelDim);
    RUN_TEST(testForwardRejectsTransposedInput);
    RUN_TEST(testGoldBackwardSingleGroup);
    RUN_TEST(testGoldBackwardTwoGroups);
    RUN_TEST(testGoldBackwardBatch2ThreeGroups);
    RUN_TEST(testGoldBackwardGroupEqualsChannels);
    RUN_TEST(testBackwardAccumulatesGradsOverwritesDx);
    RUN_TEST(testBackwardFloatFrozenTwinDxIdenticalGradsZero);
    RUN_TEST(testBackwardFloatGuardsSymForwardInput);
    RUN_TEST(testBackwardFloatGuardsSymLoss);
    RUN_TEST(testBackwardFloatGuardsSymGammaParam);
    RUN_TEST(testBackwardFloatGuardsSymGammaGrad);
    RUN_TEST(testBackwardFloatGuardsSymBetaGrad);
    RUN_TEST(testBackwardFloatGuardsSymPropLoss);
    RUN_TEST(testSymForwardTwinSanityTwoGroups);
    RUN_TEST(testSymForwardRejectsOperandWiderThanInt12);
    RUN_TEST(testSymBackwardTwinSanityTwoGroups);
    RUN_TEST(testSymBackwardFrozenTwinDxIdenticalGradsUntouched);
    RUN_TEST(testSymBackwardRejectsOperandWiderThanInt12);
    RUN_TEST(testFactoryBuildsGammaOnesBetaZerosAndForwards);
    RUN_TEST(testFactoryAppliesDefaultEpsWhenZero);
    RUN_TEST(testFactoryOwningDeepCopiesQuantizations);
    RUN_TEST(testFactoryBorrowingDoesNotFreeCallerQuantizations);
    RUN_TEST(testFactoryRejectsNonDivisibleGroups);
    RUN_TEST(testFactorySymInt32StorageQuantizesGammaBeta);
    RUN_TEST(testFactoryOwningSymInt32DeepCopies);
    RUN_TEST(testGroupNormFactoryFrozenElidesGrads);
    RUN_TEST(testGroupNormFactoryDefaultAllocatesGrads);
    RUN_TEST(testGroupNormBackwardFrozenFactoryLayerRunsWithoutGradBuffers);
    RUN_TEST(testBackwardFloatNullPropLossComputesGradsOnly);
    RUN_TEST(testSymBackwardNullPropLossComputesGradsOnly);
    RUN_TEST(testGroupNormForwardBfpFakeQuantPinTwin);
    RUN_TEST(testGroupNormForwardBfpStagedFloat32OperandsTwin);
    RUN_TEST(testGroupNormForwardBfpTwinSanityTwoGroups);
    RUN_TEST(testGroupNormForwardBfpMissingOutputQAnchorDies);
    RUN_TEST(testGroupNormBackwardBfpTwinSanityTwoGroups);
    RUN_TEST(testGroupNormBackwardBfpFakeQuantPinCrossCheck);
    RUN_TEST(testGroupNormBackwardBfpGradsAccumulateAcrossCalls);
    RUN_TEST(testGroupNormBackwardBfpNullPropLossComputesGradsOnly);
    RUN_TEST(testGroupNormBackwardBfpFrozenSkipsGrads);
    RUN_TEST(testGroupNormBackwardBfpFrozenShortLossDies);
    RUN_TEST(testGroupNormBackwardBfpMissingPropLossQAnchorDies);
    RUN_TEST(testGroupNormBackwardFloat32PinnedStillRejectsBfpWires);
    RUN_TEST(testFactoryUniformBfpProfileBuildsAndForwards);
    RUN_TEST(testFactoryBfpGammaGeometryDerivedFromParamLength);
    RUN_TEST(testFactoryWiresOnlyBfpProfileBuilds);
    RUN_TEST(testFactoryRejectsSymGammaUnderBfpForward);
    RUN_TEST(testFactoryRejectsFloat32MathOverBfpParams);
    RUN_TEST(testFactoryRejectsFloat32BackwardOverBfpWires);
    RUN_TEST(testFactoryRejectsGroupedBfpGradTemplate);
    RUN_TEST(testFactoryBfpGradStorageBuildsPerTensor);
    RUN_TEST(testFactoryOwningBfpDeepCopiesQuantizations);
    return UNITY_END();
}
