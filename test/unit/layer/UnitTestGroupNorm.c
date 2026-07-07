#define SOURCE_FILE "UNIT_TEST_GROUPNORM"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
    RUN_TEST(testBackwardFloatGuardsSymForwardInput);
    RUN_TEST(testBackwardFloatGuardsSymLoss);
    RUN_TEST(testBackwardFloatGuardsSymGammaParam);
    RUN_TEST(testBackwardFloatGuardsSymGammaGrad);
    RUN_TEST(testBackwardFloatGuardsSymBetaGrad);
    RUN_TEST(testBackwardFloatGuardsSymPropLoss);
    RUN_TEST(testSymForwardTwinSanityTwoGroups);
    RUN_TEST(testSymForwardRejectsOperandWiderThanInt12);
    RUN_TEST(testSymBackwardTwinSanityTwoGroups);
    RUN_TEST(testSymBackwardRejectsOperandWiderThanInt12);
    RUN_TEST(testFactoryBuildsGammaOnesBetaZerosAndForwards);
    RUN_TEST(testFactoryAppliesDefaultEpsWhenZero);
    RUN_TEST(testFactoryOwningDeepCopiesQuantizations);
    RUN_TEST(testFactoryBorrowingDoesNotFreeCallerQuantizations);
    RUN_TEST(testFactoryRejectsNonDivisibleGroups);
    RUN_TEST(testFactorySymInt32StorageQuantizesGammaBeta);
    RUN_TEST(testFactoryOwningSymInt32DeepCopies);
    return UNITY_END();
}
