#define SOURCE_FILE "UNIT_TEST_LAYERNORM"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "LayerNormApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include "expected_layernorm.h"
#include "expected_layernorm_sym.h"
#include "expected_layernorm_sym_bwd.h"

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

/* Build a γ or β parameter_t of shape normalizedShape, filled from `vals`
 * (NULL → leave at calloc-zero). Grad is FLOAT32 zero. Caller frees via
 * freeParameter. */
static parameter_t *buildFloatParam(size_t numDims, const size_t *dimsIn, const float *vals) {
    tensor_t *p = buildFloatTensorND(numDims, dimsIn, vals);
    tensor_t *g = gradInitFloat(p, NULL);
    return parameterInit(p, g);
}

/* Build a SYM_INT32 (HALF_AWAY, qMaxBits=16) tensor; float vals are quantized via
 * tensorFillFromFloatBuffer -> convertFloatTensorToSymInt32Tensor (absmax ->
 * scale, round-clamp; absmax==0 -> scale 1.0). NULL vals -> zero mantissas,
 * default scale 1.0. Caller frees via freeTensor. */
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

/* SYM_INT32 parameter with SYM_INT32 grad (grads unused in forward-only tests). */
static parameter_t *buildSymParam(size_t numDims, const size_t *dimsIn, const float *vals) {
    tensor_t *p = buildSymInt32TensorND(numDims, dimsIn, vals);
    tensor_t *g = gradInitSymInt32(p, HALF_AWAY, NULL);
    return parameterInit(p, g);
}

static float symScaleOf(tensor_t *t) {
    return ((symInt32QConfig_t *)t->quantization->qConfig)->scale;
}

void testConfigStructIsPopulated(void) {
    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);

    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();

    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);

    size_t numNormDims = cfg.numNormDims;
    float eps = cfg.eps;
    bool gammaOk = (cfg.gamma == gamma);
    bool betaOk = (cfg.beta == beta);
    bool fqOk = (cfg.forwardQ == fq);
    bool bqOk = (cfg.backwardQ == bq);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);

    TEST_ASSERT_EQUAL_UINT(1, numNormDims);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1e-5f, eps);
    TEST_ASSERT_TRUE(gammaOk);
    TEST_ASSERT_TRUE(betaOk);
    TEST_ASSERT_TRUE(fqOk);
    TEST_ASSERT_TRUE(bqOk);
}

void testCalcOutputShapeIsIdentity(void) {
    size_t inDims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, inDims, NULL);

    size_t *odims = reserveMemory(2 * sizeof(size_t));
    size_t *oorder = reserveMemory(2 * sizeof(size_t));
    shape_t *outShape = reserveMemory(sizeof(shape_t));
    outShape->dimensions = odims;
    outShape->orderOfDimensions = oorder;
    outShape->numberOfDimensions = 0;

    layer_t layer = {.type = LAYERNORM, .config = NULL};
    layerNormCalcOutputShape(&layer, in->shape, outShape);

    size_t nd = outShape->numberOfDimensions;
    size_t d0 = outShape->dimensions[0];
    size_t d1 = outShape->dimensions[1];

    freeReservedMemory(outShape);
    freeReservedMemory(oorder);
    freeReservedMemory(odims);
    freeTensor(in);

    TEST_ASSERT_EQUAL_UINT(2, nd);
    TEST_ASSERT_EQUAL_UINT(2, d0);
    TEST_ASSERT_EQUAL_UINT(3, d1);
}

/* Wrap a layerNormConfig in a stack layer_t. */
static layer_t makeLayerNormLayer(layerNormConfig_t *cfg, layerConfig_t *lcfg) {
    lcfg->layerNorm = cfg;
    layer_t layer = {.type = LAYERNORM, .config = lcfg};
    return layer;
}

/* Dequant-domain invariants (gamma=1, beta=0): per-group dequantized mean ~ 0,
 * per-group dequantized variance ~ var/(var+eps) (PyTorch eps behavior).
 * Affine-stable: with gamma=ones the dequantized values are unchanged by the
 * Task-3 affine stage (mantissa*32767 and scale/32767 cancel). */
void testSymForwardDequantInvariantsMultiGroup(void) {
    size_t dims[] = {2, 4};
    tensor_t *in =
        buildSymInt32TensorND(2, dims, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f});
    tensor_t *out = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParam(1, ns, (float[]){0.f, 0.f, 0.f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float scale = symScaleOf(out);
    int32_t *m = (int32_t *)out->data;
    float groupMean[2];
    float groupVar[2];
    for (size_t g = 0; g < 2; g++) {
        float mean = 0.f;
        for (size_t j = 0; j < 4; j++) {
            mean += (float)m[g * 4 + j] * scale;
        }
        mean /= 4.f;
        float var = 0.f;
        for (size_t j = 0; j < 4; j++) {
            float d = (float)m[g * 4 + j] * scale - mean;
            var += d * d;
        }
        groupMean[g] = mean;
        groupVar[g] = var / 4.f;
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    /* var/(var+eps): row0 1.25/(1.25+1e-5)=0.999992; row1 125/(125+1e-5)~1.0.
     * Tolerances absorb input quantization noise (s_x = 40/32767 ~ 1.2e-3). */
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.f, groupMean[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.f, groupMean[1]);
    TEST_ASSERT_FLOAT_WITHIN(5e-3f, 0.999992f, groupVar[0]);
    TEST_ASSERT_FLOAT_WITHIN(5e-3f, 1.0f, groupVar[1]);
}

/* Derivation for testForwardFloatSingleGroup:
 *   input  = [1, 2, 3, 4]
 *   mean   = 2.5
 *   biased var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
 *   invSigma = 1/sqrt(1.25 + 1e-5) ≈ 0.8944236
 *   n      = [-1.3416354, -0.4472118, +0.4472118, +1.3416354]
 *   gamma  = [2, 3, 4, 5],  beta = [1, -1, 0.5, 0]
 *   y      = gamma*n + beta = [-1.6832708, -2.3416354, +2.2888472, +6.7081771]
 */
void testForwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){2.f, 3.f, 4.f, 5.f});
    parameter_t *beta = buildFloatParam(1, ns, (float[]){1.f, -1.f, 0.5f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float captured[4];
    for (size_t i = 0; i < 4; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.6832708f, -2.3416354f, 2.2888472f, 6.7081771f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], captured[i]);
    }
}

/* Derivation for testForwardFloatMultiGroup:
 *   rank-2 [2,3], normalizedShape=[3], D=1 → G=2 rows.
 *   Row0=[0,1,2]: mu=1, var=2/3, sigma≈0.81652,
 *     n=[-1.22472, 0, +1.22472]; gamma=[1,2,3], beta=0
 *     → [-1.22472, 0, +3.67417]
 *   Row1=[10,10,10]: mu=10, var=0 → n=[0,0,0] → [0,0,0]
 *   Also exercises shared gamma across groups.
 */
void testForwardFloatMultiGroup(void) {
    size_t dims[] = {2, 3};
    tensor_t *in = buildFloatTensorND(2, dims, (float[]){0.f, 1.f, 2.f, 10.f, 10.f, 10.f});
    tensor_t *out = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float captured[6];
    for (size_t i = 0; i < 6; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.22472f, 0.f, 3.67417f, 0.f, 0.f, 0.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected[i], captured[i]);
    }
}

void testForwardFloatTransposedInput(void) {
    /* Physical [3,2] buffer; after transpose logical shape is [2,3] with
     * logical row0=[0,1,2], row1=[10,10,10] (matches the multi-group test). */
    size_t pdims[] = {3, 2};
    tensor_t *in = buildFloatTensorND(2, pdims, (float[]){0.f, 10.f, 1.f, 10.f, 2.f, 10.f});
    transposeTensor(in, 0, 1); /* logical [2,3], physical unchanged */

    /* Output: same logical [2,3], same transposed layout, so dx/y scatter back
     * to matching physical offsets. */
    size_t pdimsOut[] = {3, 2};
    tensor_t *out = buildFloatTensorND(2, pdimsOut, NULL);
    transposeTensor(out, 0, 1);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    /* Read back logically via the same offset map the kernel used:
     * logical row0 -> [-1.22472, 0, 3.67417], row1 -> [0,0,0]. */
    float captured[6];
    float *od = (float *)out->data;
    /* physical offsets: logical (r,c) -> phys c*2 + r */
    captured[0] = od[0 * 2 + 0]; /* (0,0) */
    captured[1] = od[1 * 2 + 0]; /* (0,1) */
    captured[2] = od[2 * 2 + 0]; /* (0,2) */
    captured[3] = od[0 * 2 + 1]; /* (1,0) */
    captured[4] = od[1 * 2 + 1]; /* (1,1) */
    captured[5] = od[2 * 2 + 1]; /* (1,2) */

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {-1.22472f, 0.f, 3.67417f, 0.f, 0.f, 0.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, expected[i], captured[i]);
    }
}

void testVtableForwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){2.f, 2.f, 2.f, 2.f});
    parameter_t *beta = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerFunctions[LAYERNORM].forward(&layer, in, out);

    float captured[4];
    for (size_t i = 0; i < 4; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expected[] = {2.99999f, -0.99999f, 2.99999f, -0.99999f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expected[i], captured[i]);
    }
}

/* x=[1,-1,1,-1]: mu=0, var=1, invSigma=1/sqrt(1+1e-5)=0.9999950, n=±0.9999950.
 * dy=[1,2,3,4]: dbeta=dy; dgamma=dy*n; dn=dy (gamma=1), meanDn=2.5,
 * meanDnN=-0.4999975, dx=invSigma*(dn-meanDn-n*meanDnN). */
void testBackwardFloatSingleGroup(void) {
    size_t dims[] = {4};
    tensor_t *fwdIn = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *loss = buildFloatTensorND(1, dims, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *propLoss = buildFloatTensorND(1, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL); /* grads start at zero */
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerFunctions[LAYERNORM].backward(&layer, fwdIn, loss, propLoss);

    float dx[4], dgamma[4], dbeta[4];
    for (size_t i = 0; i < 4; i++) {
        dx[i] = ((float *)propLoss->data)[i];
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDx[] = {-1.0f, -0.99999f, 0.99999f, 1.0f};
    float expDgamma[] = {0.999995f, -1.99999f, 2.999985f, -3.99998f};
    float expDbeta[] = {1.f, 2.f, 3.f, 4.f};
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
}

/* Two groups, uniform dy: dbeta sums over groups ([2,2]); dgamma sums dy*n over
 * groups (N=2 saturation: n=[-0.999995,+0.999995] in both rows regardless of
 * the row values -> [-1.99999,+1.99999]).
 * dx = 0 exactly: uniform dy with gamma=1 makes dn constant per group, so
 * dn-meanDn = 0 and meanDnN = dn*mean(n) = 0 (holds for any N; LayerNorm
 * backward annihilates uniform upstream gradients). */
void testBackwardFloatMultiGroupAccumulatesGradients(void) {
    size_t dims[] = {2, 2};
    tensor_t *fwdIn = buildFloatTensorND(2, dims, (float[]){-1.f, 1.f, 2.f, 4.f});
    tensor_t *loss = buildFloatTensorND(2, dims, (float[]){1.f, 1.f, 1.f, 1.f});
    tensor_t *propLoss = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {2};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 1.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 2;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    float dgamma[2], dbeta[2], dx[4];
    for (size_t i = 0; i < 2; i++) {
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }
    for (size_t i = 0; i < 4; i++) {
        dx[i] = ((float *)propLoss->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDgamma[] = {-1.99999f, 1.99999f};
    float expDbeta[] = {2.f, 2.f};
    for (size_t i = 0; i < 2; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.f, dx[i]);
    }
}

/* Divergent layouts: forwardInput is TRANSPOSED (physical [3,2] buffer
 * [0,1,1,3,2,8], logical [2,3] with (r,c) at buf[c*2+r]), while loss/propLoss
 * are natural [2,3]. Catches routing any offset through the wrong tensor map.
 * Logical fixture: x=[[0,1,2],[1,3,8]], dy=[[1,2,3],[-1,0.5,2]], gamma=[1,2,3],
 * beta=0, normalizedShape=[3], eps=1e-5.
 * Expected values derived one-off via PyTorch float64 autograd:
 *   uv run python -c "... F.layer_norm(x,[3],weight=g,bias=b,eps=1e-5);
 *                     y.backward(dy)"
 *   dx     = [0.4081717, -0.8164905, 0.4083187,
 *             -1.175824e-6, -3.919414e-7, 1.567765e-6]
 *   dgamma = [-0.2056869, -0.1698415, 6.391670]
 *   dbeta  = [0, 2.5, 5]  (= sum_g dy, exact) */
void testBackwardFloatDivergentLayouts(void) {
    size_t pdims[] = {3, 2};
    tensor_t *fwdIn = buildFloatTensorND(2, pdims, (float[]){0.f, 1.f, 1.f, 3.f, 2.f, 8.f});
    transposeTensor(fwdIn, 0, 1); /* logical [2,3], physical unchanged */

    size_t dims[] = {2, 3};
    tensor_t *loss = buildFloatTensorND(2, dims, (float[]){1.f, 2.f, 3.f, -1.f, 0.5f, 2.f});
    tensor_t *propLoss = buildFloatTensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildFloatParam(1, ns, (float[]){1.f, 2.f, 3.f});
    parameter_t *beta = buildFloatParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    float dx[6], dgamma[3], dbeta[3];
    for (size_t i = 0; i < 6; i++) {
        dx[i] = ((float *)propLoss->data)[i]; /* natural flat indices */
    }
    for (size_t i = 0; i < 3; i++) {
        dgamma[i] = ((float *)gamma->grad->data)[i];
        dbeta[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    float expDx[] = {0.4081717f,    -0.8164905f,   0.4083187f,
                     -1.175824e-6f, -3.919414e-7f, 1.567765e-6f};
    float expDgamma[] = {-0.2056869f, -0.1698415f, 6.391670f};
    float expDbeta[] = {0.f, 2.5f, 5.f};
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dgamma[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], dbeta[i]);
    }
}

void testFactoryBuildsGammaOnesBetaZerosAndForwards(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 0.0f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInit(&init, &lq);

    bool typeOk = (layer->type == LAYERNORM);
    layerNormConfig_t *cfg = layer->config->layerNorm;
    /* eps==0 → factory substitutes default 1e-5 */
    float capturedEps = cfg->eps;
    /* gamma all ones, beta all zeros */
    float g0 = ((float *)cfg->gamma->param->data)[0];
    float g3 = ((float *)cfg->gamma->param->data)[3];
    float b0 = ((float *)cfg->beta->param->data)[0];
    bool gammaGradFloat = (cfg->gamma->grad->quantization->type == FLOAT32);
    bool betaGradFloat = (cfg->beta->grad->quantization->type == FLOAT32);
    bool fwdMapped = (cfg->forwardQ == fwdMath);
    bool bwdMapped = (cfg->backwardQ == bwdMath);

    /* Forward: x=[1,-1,1,-1], gamma=1, beta=0 -> n ~ [+1,-1,+1,-1] (eps=1e-5). */
    size_t dims[] = {4};
    tensor_t *in = buildFloatTensorND(1, dims, (float[]){1.f, -1.f, 1.f, -1.f});
    tensor_t *out = buildFloatTensorND(1, dims, NULL);
    layerFunctions[LAYERNORM].forward(layer, in, out);
    float y0 = ((float *)out->data)[0];
    float y1 = ((float *)out->data)[1];

    freeTensor(out);
    freeTensor(in);
    freeLayerNormLayer(layer);
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
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.999995f, y0);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.999995f, y1);
}

void testFactoryOwningDeepCopiesQuantizations(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInitOwning(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* Owning: forwardQ/backwardQ are fresh allocations, NOT the caller's. */
    bool fwdIsCopy = (cfg->forwardQ != fwdMath);
    bool bwdIsCopy = (cfg->backwardQ != bwdMath);
    bool fwdTypeOk = (cfg->forwardQ->type == fwdMath->type);
    bool owns = cfg->ownsQuantizations;

    /* Caller drops its math quant configs IMMEDIATELY — the layer holds copies.
     * Storage quant is cloned into the param tensors via getQLike, so freeing the
     * caller's wStore/bStore here is also safe. */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    /* Now tear down the layer — frees gamma/beta + the OWNED forwardQ/backwardQ
     * copies. No double-free (the caller's originals are already gone and were
     * never aliased). */
    freeLayerNormLayer(layer);

    TEST_ASSERT_TRUE(fwdIsCopy);
    TEST_ASSERT_TRUE(bwdIsCopy);
    TEST_ASSERT_TRUE(fwdTypeOk);
    TEST_ASSERT_TRUE(owns);
}

void testFactoryBorrowingDoesNotFreeCallerQuantizations(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *fwdMath = quantizationInitFloat();
    quantization_t *bwdMath = quantizationInitFloat();
    quantization_t *wStore = quantizationInitFloat();
    quantization_t *bStore = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = fwdMath,
                       .backwardMath = bwdMath,
                       .weightStorage = wStore,
                       .biasStorage = bStore};

    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    /* Borrowing: verbatim pointers, no ownership. */
    bool fwdVerbatim = (cfg->forwardQ == fwdMath);
    bool bwdVerbatim = (cfg->backwardQ == bwdMath);
    bool owns = cfg->ownsQuantizations;

    /* Free the layer FIRST — it must NOT touch the borrowed math quantizations. */
    freeLayerNormLayer(layer);

    /* Caller frees its own quant configs AFTER. If freeLayerNormLayer had freed
     * forwardQ/backwardQ, these would be double-frees (ASan/valgrind catch them). */
    freeQuantization(bStore);
    freeQuantization(wStore);
    freeQuantization(bwdMath);
    freeQuantization(fwdMath);

    TEST_ASSERT_TRUE(fwdVerbatim);
    TEST_ASSERT_TRUE(bwdVerbatim);
    TEST_ASSERT_FALSE(owns);
}

void testFactoryRank2NormShapeAndExplicitEps(void) {
    size_t normShape[] = {2, 3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 2, .eps = 0.1f};

    quantization_t *q = quantizationInitFloat();
    layerQuant_t lq = {.forwardMath = q, .backwardMath = q, .weightStorage = q, .biasStorage = q};

    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    size_t numNormDims = cfg->numNormDims;
    float capturedEps = cfg->eps;
    size_t ns0 = cfg->normalizedShape[0];
    size_t ns1 = cfg->normalizedShape[1];
    size_t gammaCount = calcNumberOfElementsByTensor(cfg->gamma->param);

    freeLayerNormLayer(layer);
    freeQuantization(q);

    TEST_ASSERT_EQUAL_UINT(2, numNormDims);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.1f, capturedEps);
    TEST_ASSERT_EQUAL_UINT(2, ns0);
    TEST_ASSERT_EQUAL_UINT(3, ns1);
    TEST_ASSERT_EQUAL_UINT(6, gammaCount);
}

/* Run forward with explicit gamma/beta gold arrays and compare to expected y. */
static void runGoldForward(size_t numDims, const size_t *dims, const float *xVals,
                           size_t numNormDims, const size_t *normShapeIn, const float *gammaVals,
                           const float *betaVals, const float *expectedY, size_t count) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    tensor_t *in = buildFloatTensorND(numDims, dims, xVals);
    tensor_t *out = buildFloatTensorND(numDims, dims, NULL);
    parameter_t *gamma = buildFloatParam(numNormDims, normShapeIn, gammaVals);
    parameter_t *beta = buildFloatParam(numNormDims, normShapeIn, betaVals);
    size_t *normShape = reserveMemory(numNormDims * sizeof(size_t));
    for (size_t i = 0; i < numNormDims; i++) {
        normShape[i] = normShapeIn[i];
    }
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, numNormDims, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float captured[64];
    for (size_t i = 0; i < count; i++) {
        captured[i] = ((float *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedY[i], captured[i]);
    }
}

/* Run backward with gold gamma/beta + gold lossGrad; compare dx, dgamma, dbeta. */
static void runGoldBackward(size_t numDims, const size_t *dims, const float *xVals,
                            size_t numNormDims, const size_t *normShapeIn, const float *gammaVals,
                            const float *betaVals, const float *lossGradVals, const float *expDx,
                            const float *expDgamma, const float *expDbeta, size_t count,
                            size_t normCount) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    TEST_ASSERT_TRUE_MESSAGE(normCount > 0 && normCount <= 64, "fixture exceeds capture buffer");
    tensor_t *fwdIn = buildFloatTensorND(numDims, dims, xVals);
    tensor_t *loss = buildFloatTensorND(numDims, dims, lossGradVals);
    tensor_t *propLoss = buildFloatTensorND(numDims, dims, NULL);
    parameter_t *gamma = buildFloatParam(numNormDims, normShapeIn, gammaVals);
    parameter_t *beta = buildFloatParam(numNormDims, normShapeIn, betaVals);
    size_t *normShape = reserveMemory(numNormDims * sizeof(size_t));
    for (size_t i = 0; i < numNormDims; i++) {
        normShape[i] = normShapeIn[i];
    }
    quantization_t *fq = quantizationInitFloat();
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, numNormDims, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    float dx[64], dg[64], db[64];
    for (size_t i = 0; i < count; i++) {
        dx[i] = ((float *)propLoss->data)[i];
    }
    for (size_t i = 0; i < normCount; i++) {
        dg[i] = ((float *)gamma->grad->data)[i];
        db[i] = ((float *)beta->grad->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDx[i], dx[i]);
    }
    for (size_t i = 0; i < normCount; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDgamma[i], dg[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, expDbeta[i], db[i]);
    }
}

/* SYM gold runner: forward over generated fixtures, then assert
 * (a) mantissas within +-mantissaTol of the float64 emulation (float32 stats
 *     noise can flip a rounding boundary; tol = 2*max|gamma_q|+1),
 * (b) output scale within 1e-4 relative of the emulated s_y,
 * (c) dequantized output within dequantTol of F.layer_norm on the same
 *     dequantized inputs,
 * (d) for gamma=1/beta=0 fixtures: dequant within 3e-5 of PyTorch xhat (the
 *     spec-verified tolerance; generator guarantees absmax(xhat) <= 1.9), and
 * (e) per-group dequant variance within 5e-3 of var/(var+eps). */
static void runSymGoldForward(size_t numDims, const size_t *dims, size_t numNormDims,
                              const size_t *normShapeIn, const float *xVals, const float *gammaVals,
                              const float *betaVals, const int32_t *expMantissas, float expScale,
                              int32_t mantissaTol, const float *expDequant, float dequantTol,
                              const float *expXhat, const float *expGroupVarRatio, size_t groups,
                              size_t count) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    size_t normCount = count / groups;

    tensor_t *in = buildSymInt32TensorND(numDims, dims, xVals);
    tensor_t *out = buildSymInt32TensorND(numDims, dims, NULL);
    parameter_t *gamma = buildSymParam(numNormDims, normShapeIn, gammaVals);
    parameter_t *beta = buildSymParam(numNormDims, normShapeIn, betaVals);
    size_t *normShape = reserveMemory(numNormDims * sizeof(size_t));
    for (size_t i = 0; i < numNormDims; i++) {
        normShape[i] = normShapeIn[i];
    }
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, numNormDims, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    int32_t m[64];
    for (size_t i = 0; i < count; i++) {
        m[i] = ((int32_t *)out->data)[i];
    }
    float scale = symScaleOf(out);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    for (size_t i = 0; i < count; i++) {
        TEST_ASSERT_INT_WITHIN(mantissaTol, expMantissas[i], m[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(expScale * 1e-4f, expScale, scale);
    for (size_t i = 0; i < count; i++) {
        float deq = (float)m[i] * scale;
        TEST_ASSERT_FLOAT_WITHIN(dequantTol, expDequant[i], deq);
        if (expXhat != NULL) {
            TEST_ASSERT_FLOAT_WITHIN(3e-5f, expXhat[i], deq);
        }
    }
    if (expGroupVarRatio != NULL) {
        for (size_t g = 0; g < groups; g++) {
            float mean = 0.f;
            for (size_t j = 0; j < normCount; j++) {
                mean += (float)m[g * normCount + j] * scale;
            }
            mean /= (float)normCount;
            float var = 0.f;
            for (size_t j = 0; j < normCount; j++) {
                float d = (float)m[g * normCount + j] * scale - mean;
                var += d * d;
            }
            var /= (float)normCount;
            TEST_ASSERT_FLOAT_WITHIN(5e-3f, expGroupVarRatio[g], var);
        }
    }
}

/* SYM backward gold runner: hand-built layer with backwardQ = SYM_INT32, fresh
 * (zero, scale-1.0) grads, backward through the vtable. Asserts:
 * (a) dgamma/dbeta grad mantissas within gradMantissaTol of the float64
 *     emulation (quantize-increment + addSymInt32TensorsInplace),
 * (b) grad scales within 1e-4 relative of the emulated scales,
 * (c) dequantized grads within gradDequantTol of float64 torch-autograd,
 * (d) when expDx != NULL: dx mantissas/scale/dequant likewise (dequant asserts
 *     are the scale-blindness catchers: a dropped s_dy/s_gamma/s_x leaves
 *     mantissas IDENTICAL and shows up only in the scale).
 * Fixtures are contiguous default-order: physical offset == g*N + j. */
static void runSymGoldBackward(
    size_t numDims, const size_t *dims, size_t numNormDims, const size_t *normShapeIn,
    const float *xVals, const float *gammaVals, const float *lossGradVals, const int32_t *expDx,
    float expDxScale, int32_t dxMantissaTol, const float *expDxDequant, float dxDequantTol,
    const int32_t *expDgamma, float expDgammaScale, const int32_t *expDbeta, float expDbetaScale,
    int32_t gradMantissaTol, const float *expDgammaDequant, const float *expDbetaDequant,
    float gradDequantTol, size_t count, size_t normCount) {
    TEST_ASSERT_TRUE_MESSAGE(count > 0 && count <= 64, "fixture exceeds capture buffer");
    TEST_ASSERT_TRUE_MESSAGE(normCount > 0 && normCount <= 64, "fixture exceeds capture buffer");

    tensor_t *fwdIn = buildSymInt32TensorND(numDims, dims, xVals);
    tensor_t *loss = buildSymInt32TensorND(numDims, dims, lossGradVals);
    tensor_t *propLoss = buildSymInt32TensorND(numDims, dims, NULL);
    parameter_t *gamma = buildSymParam(numNormDims, normShapeIn, gammaVals);
    parameter_t *beta = buildSymParam(numNormDims, normShapeIn, NULL); /* zeros; unused in bwd */
    size_t *normShape = reserveMemory(numNormDims * sizeof(size_t));
    for (size_t i = 0; i < numNormDims; i++) {
        normShape[i] = normShapeIn[i];
    }
    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, numNormDims, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerFunctions[LAYERNORM].backward(&layer, fwdIn, loss, propLoss);

    int32_t dgm[64], dbm[64], dxm[64];
    for (size_t i = 0; i < normCount; i++) {
        dgm[i] = ((int32_t *)gamma->grad->data)[i];
        dbm[i] = ((int32_t *)beta->grad->data)[i];
    }
    for (size_t i = 0; i < count; i++) {
        dxm[i] = ((int32_t *)propLoss->data)[i];
    }
    float dgScale = symScaleOf(gamma->grad);
    float dbScale = symScaleOf(beta->grad);
    float dxScale = symScaleOf(propLoss);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    for (size_t i = 0; i < normCount; i++) {
        TEST_ASSERT_INT_WITHIN(gradMantissaTol, expDgamma[i], dgm[i]);
        TEST_ASSERT_INT_WITHIN(gradMantissaTol, expDbeta[i], dbm[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(expDgammaScale * 1e-4f, expDgammaScale, dgScale);
    TEST_ASSERT_FLOAT_WITHIN(expDbetaScale * 1e-4f, expDbetaScale, dbScale);
    for (size_t i = 0; i < normCount; i++) {
        TEST_ASSERT_FLOAT_WITHIN(gradDequantTol, expDgammaDequant[i], (float)dgm[i] * dgScale);
        TEST_ASSERT_FLOAT_WITHIN(gradDequantTol, expDbetaDequant[i], (float)dbm[i] * dbScale);
    }
    if (expDx != NULL) {
        for (size_t i = 0; i < count; i++) {
            TEST_ASSERT_INT_WITHIN(dxMantissaTol, expDx[i], dxm[i]);
        }
        TEST_ASSERT_FLOAT_WITHIN(expDxScale * 1e-4f, expDxScale, dxScale);
        for (size_t i = 0; i < count; i++) {
            TEST_ASSERT_FLOAT_WITHIN(dxDequantTol, expDxDequant[i], (float)dxm[i] * dxScale);
        }
    }
}

void testSymBackwardGoldMultiGroup(void) {
    size_t dims[] = {3, 4};
    size_t ns[] = {4};
    runSymGoldBackward(
        2, dims, 1, ns, input_layerNormSymBwd_bwdBase, gamma_layerNormSymBwd_bwdBase,
        lossGrad_layerNormSymBwd_bwdBase, expectedDx_layerNormSymBwd_bwdBase,
        expectedDxScale_layerNormSymBwd_bwdBase, dxMantissaTol_layerNormSymBwd_bwdBase,
        expectedDxDequant_layerNormSymBwd_bwdBase, dxDequantTol_layerNormSymBwd_bwdBase,
        expectedDgamma_layerNormSymBwd_bwdBase, expectedDgammaScale_layerNormSymBwd_bwdBase,
        expectedDbeta_layerNormSymBwd_bwdBase, expectedDbetaScale_layerNormSymBwd_bwdBase,
        gradMantissaTol_layerNormSymBwd_bwdBase, expectedDgammaDequant_layerNormSymBwd_bwdBase,
        expectedDbetaDequant_layerNormSymBwd_bwdBase, gradDequantTol_layerNormSymBwd_bwdBase, 12,
        4);
}

/* dy == 0: dx must be all-zero mantissas with the NEUTRAL scale 1.0 (the
 * absmax==0 idiom), and the fresh zero grads must stay EXACTLY zero with
 * scale 1.0 (zero increments quantize to zeros/scale-1.0 and 0+0 == 0 through
 * the strategy-A add). Zero propagates exactly through float products/sums
 * even under -ffp-contract=fast — no integer pre-check needed (unlike the
 * forward's constant-input cancellation). */
void testSymBackwardZeroLossGradEmitsZerosNeutralScale(void) {
    size_t dims[] = {2, 4};
    tensor_t *fwdIn =
        buildSymInt32TensorND(2, dims, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f});
    tensor_t *loss = buildSymInt32TensorND(2, dims, NULL); /* zero mantissas, scale 1.0 */
    tensor_t *propLoss = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){2.f, 1.f, -1.f, 0.5f});
    parameter_t *beta = buildSymParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss, propLoss);

    int32_t dxm[8], dgm[4], dbm[4];
    for (size_t i = 0; i < 8; i++) {
        dxm[i] = ((int32_t *)propLoss->data)[i];
    }
    for (size_t i = 0; i < 4; i++) {
        dgm[i] = ((int32_t *)gamma->grad->data)[i];
        dbm[i] = ((int32_t *)beta->grad->data)[i];
    }
    float dxScale = symScaleOf(propLoss);
    float dgScale = symScaleOf(gamma->grad);
    float dbScale = symScaleOf(beta->grad);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);

    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_INT(0, dxm[i]);
    }
    TEST_ASSERT_EQUAL_FLOAT(1.0f, dxScale);
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_INT(0, dgm[i]);
        TEST_ASSERT_EQUAL_INT(0, dbm[i]);
    }
    TEST_ASSERT_EQUAL_FLOAT(1.0f, dgScale);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, dbScale);
}

void testSymGoldParityMultiGroup(void) {
    size_t dims[] = {3, 4};
    size_t ns[] = {4};
    runSymGoldForward(2, dims, 1, ns, input_layerNormSym_symParity, gamma_layerNormSym_symParity,
                      beta_layerNormSym_symParity, expectedMantissas_layerNormSym_symParity,
                      expectedScale_layerNormSym_symParity, mantissaTol_layerNormSym_symParity,
                      expectedDequant_layerNormSym_symParity, dequantTol_layerNormSym_symParity,
                      expectedXhat_layerNormSym_symParity,
                      expectedGroupVarRatio_layerNormSym_symParity, 3,
                      expectedMantissas_layerNormSym_symParity_len);
}

void testSymGoldAffine(void) {
    size_t dims[] = {2, 4};
    size_t ns[] = {4};
    runSymGoldForward(2, dims, 1, ns, input_layerNormSym_symAffine, gamma_layerNormSym_symAffine,
                      beta_layerNormSym_symAffine, expectedMantissas_layerNormSym_symAffine,
                      expectedScale_layerNormSym_symAffine, mantissaTol_layerNormSym_symAffine,
                      expectedDequant_layerNormSym_symAffine, dequantTol_layerNormSym_symAffine,
                      NULL, NULL, 2, expectedMantissas_layerNormSym_symAffine_len);
}

void testSymGoldSigmaRatio10x(void) {
    size_t dims[] = {2, 4};
    size_t ns[] = {4};
    runSymGoldForward(
        2, dims, 1, ns, input_layerNormSym_symSigmaRatio, gamma_layerNormSym_symSigmaRatio,
        beta_layerNormSym_symSigmaRatio, expectedMantissas_layerNormSym_symSigmaRatio,
        expectedScale_layerNormSym_symSigmaRatio, mantissaTol_layerNormSym_symSigmaRatio,
        expectedDequant_layerNormSym_symSigmaRatio, dequantTol_layerNormSym_symSigmaRatio,
        expectedXhat_layerNormSym_symSigmaRatio, expectedGroupVarRatio_layerNormSym_symSigmaRatio,
        2, expectedMantissas_layerNormSym_symSigmaRatio_len);
}

/* Small-variance fixture: var (1.25e-6) is comparable to eps (1e-5), so this
 * test distinguishes sqrt(var+eps) from the wrong sqrt(var)+eps — the O(1)-
 * variance fixtures cannot (relative difference ~5e-6, below tolerance).
 *   x = [0.001, 0.002, 0.003, 0.004]: mean=0.0025, biased var=1.25e-6,
 *   invSigma = 1/sqrt(1.125e-5) = 298.142, gamma=1, beta=0
 *   y = n = [-0.447213, -0.149071, +0.149071, +0.447213]
 *   (sqrt(var)+eps would give y0 ~ -1.3298 instead). */
void testForwardFloatSmallVarianceEpsInsideSqrt(void) {
    size_t dims[] = {4};
    size_t ns[] = {4};
    float x[] = {0.001f, 0.002f, 0.003f, 0.004f};
    float gammaOnes[] = {1.f, 1.f, 1.f, 1.f};
    float betaZeros[] = {0.f, 0.f, 0.f, 0.f};
    float expected[] = {-0.447213f, -0.149071f, 0.149071f, 0.447213f};
    runGoldForward(1, dims, x, 1, ns, gammaOnes, betaZeros, expected, 4);
}

void testGoldForwardD1SingleGroup(void) {
    size_t dims[] = {5};
    size_t ns[] = {5};
    runGoldForward(1, dims, input_layerNorm_d1SingleGroup, 1, ns, gamma_layerNorm_d1SingleGroup,
                   beta_layerNorm_d1SingleGroup, expectedForward_layerNorm_d1SingleGroup,
                   expectedForward_layerNorm_d1SingleGroup_len);
}

void testGoldBackwardD1SingleGroup(void) {
    size_t dims[] = {5};
    size_t ns[] = {5};
    runGoldBackward(1, dims, input_layerNorm_d1SingleGroup, 1, ns, gamma_layerNorm_d1SingleGroup,
                    beta_layerNorm_d1SingleGroup, lossGrad_layerNorm_d1SingleGroup,
                    expectedPropLoss_layerNorm_d1SingleGroup,
                    expectedDgamma_layerNorm_d1SingleGroup, expectedDbeta_layerNorm_d1SingleGroup,
                    expectedForward_layerNorm_d1SingleGroup_len, gamma_layerNorm_d1SingleGroup_len);
}

void testGoldForwardD1MultiGroup(void) {
    size_t dims[] = {3, 4};
    size_t ns[] = {4};
    runGoldForward(2, dims, input_layerNorm_d1MultiGroup, 1, ns, gamma_layerNorm_d1MultiGroup,
                   beta_layerNorm_d1MultiGroup, expectedForward_layerNorm_d1MultiGroup,
                   expectedForward_layerNorm_d1MultiGroup_len);
}

void testGoldBackwardD1MultiGroup(void) {
    size_t dims[] = {3, 4};
    size_t ns[] = {4};
    runGoldBackward(2, dims, input_layerNorm_d1MultiGroup, 1, ns, gamma_layerNorm_d1MultiGroup,
                    beta_layerNorm_d1MultiGroup, lossGrad_layerNorm_d1MultiGroup,
                    expectedPropLoss_layerNorm_d1MultiGroup, expectedDgamma_layerNorm_d1MultiGroup,
                    expectedDbeta_layerNorm_d1MultiGroup,
                    expectedForward_layerNorm_d1MultiGroup_len, gamma_layerNorm_d1MultiGroup_len);
}

void testGoldForwardD2MultiGroup(void) {
    size_t dims[] = {2, 3, 4};
    size_t ns[] = {3, 4};
    runGoldForward(3, dims, input_layerNorm_d2MultiGroup, 2, ns, gamma_layerNorm_d2MultiGroup,
                   beta_layerNorm_d2MultiGroup, expectedForward_layerNorm_d2MultiGroup,
                   expectedForward_layerNorm_d2MultiGroup_len);
}

void testGoldBackwardD2MultiGroup(void) {
    size_t dims[] = {2, 3, 4};
    size_t ns[] = {3, 4};
    runGoldBackward(3, dims, input_layerNorm_d2MultiGroup, 2, ns, gamma_layerNorm_d2MultiGroup,
                    beta_layerNorm_d2MultiGroup, lossGrad_layerNorm_d2MultiGroup,
                    expectedPropLoss_layerNorm_d2MultiGroup, expectedDgamma_layerNorm_d2MultiGroup,
                    expectedDbeta_layerNorm_d2MultiGroup,
                    expectedForward_layerNorm_d2MultiGroup_len, gamma_layerNorm_d2MultiGroup_len);
}

void testGoldForwardFullRank(void) {
    size_t dims[] = {2, 3};
    size_t ns[] = {2, 3};
    runGoldForward(2, dims, input_layerNorm_fullRank, 2, ns, gamma_layerNorm_fullRank,
                   beta_layerNorm_fullRank, expectedForward_layerNorm_fullRank,
                   expectedForward_layerNorm_fullRank_len);
}

void testGoldBackwardFullRank(void) {
    size_t dims[] = {2, 3};
    size_t ns[] = {2, 3};
    runGoldBackward(2, dims, input_layerNorm_fullRank, 2, ns, gamma_layerNorm_fullRank,
                    beta_layerNorm_fullRank, lossGrad_layerNorm_fullRank,
                    expectedPropLoss_layerNorm_fullRank, expectedDgamma_layerNorm_fullRank,
                    expectedDbeta_layerNorm_fullRank, expectedForward_layerNorm_fullRank_len,
                    gamma_layerNorm_fullRank_len);
}

/* Constant input -> yhat == 0 exactly in every group (identical mantissas), so
 * the GLOBAL absmax is 0. Guard: all-zero mantissas, normalization-stage scale
 * 1.0 (convertFloatTensorToSymInt32Tensor absMax==0 idiom). */
void testSymForwardConstantInputEmitsZeros(void) {
    size_t dims[] = {2, 3};
    tensor_t *in = buildSymInt32TensorND(2, dims, (float[]){7.f, 7.f, 7.f, 7.f, 7.f, 7.f});
    tensor_t *out = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParam(1, ns, (float[]){0.f, 0.f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    int32_t mantissas[6];
    for (size_t i = 0; i < 6; i++) {
        mantissas[i] = ((int32_t *)out->data)[i];
    }
    float scale = symScaleOf(out);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_INT(0, mantissas[i]);
    }
    TEST_ASSERT_TRUE(scale > 0.0f);
}

/* The affine with gamma=ones multiplies every normalized mantissa by exactly
 * 32767. Recover qNorm = y_q/32767 (remainder must be 0 — pins that the gamma
 * multiply actually happened) and assert the normalization-stage invariants:
 * full-range stretch (max |qNorm| == 32767) and near-zero per-group mantissa
 * sums (Σ n == 0 exactly in real arithmetic; rounding adds <= 0.5/element). */
void testSymForwardMantissaInvariantsViaOnesGamma(void) {
    size_t dims[] = {2, 4};
    tensor_t *in =
        buildSymInt32TensorND(2, dims, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f});
    tensor_t *out = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){1.f, 1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParam(1, ns, (float[]){0.f, 0.f, 0.f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    int32_t m[8];
    for (size_t i = 0; i < 8; i++) {
        m[i] = ((int32_t *)out->data)[i];
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    int32_t qNorm[8];
    int32_t maxAbs = 0;
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_INT(0, m[i] % 32767);
        qNorm[i] = m[i] / 32767;
        int32_t a = (qNorm[i] < 0) ? -qNorm[i] : qNorm[i];
        if (a > maxAbs) {
            maxAbs = a;
        }
    }
    TEST_ASSERT_EQUAL_INT(32767, maxAbs);
    for (size_t g = 0; g < 2; g++) {
        int32_t sum = 0;
        for (size_t j = 0; j < 4; j++) {
            sum += qNorm[g * 4 + j];
        }
        TEST_ASSERT_TRUE(sum >= -4 && sum <= 4);
    }
}

/* Constant input + nonzero beta: y = gamma*0 + beta. Pins (a) the affine runs
 * AFTER the constant guard, (b) the beta-rescale idiom (raw beta_q would
 * dequantize to 2x the right value here since s_beta = 0.5/32767 != s_y), and
 * (c) the post-affine output scale s_y = s_norm * s_gamma = 1/32767. */
void testSymForwardConstantInputAppliesBeta(void) {
    size_t dims[] = {2, 3};
    tensor_t *in = buildSymInt32TensorND(2, dims, (float[]){7.f, 7.f, 7.f, 7.f, 7.f, 7.f});
    tensor_t *out = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParam(1, ns, (float[]){0.5f, -0.25f, 0.125f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float deq[6];
    float scale = symScaleOf(out);
    for (size_t i = 0; i < 6; i++) {
        deq[i] = (float)((int32_t *)out->data)[i] * scale;
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float expectedScale = 1.0f / 32767.0f;
    TEST_ASSERT_FLOAT_WITHIN(expectedScale * 1e-4f, expectedScale, scale);
    float expBeta[] = {0.5f, -0.25f, 0.125f};
    for (size_t g = 0; g < 2; g++) {
        for (size_t j = 0; j < 3; j++) {
            /* total error <= s_beta/2 (beta storage) + s_y/2 (seed rounding) */
            TEST_ASSERT_FLOAT_WITHIN(2.0f * expectedScale, expBeta[j], deq[g * 3 + j]);
        }
    }
}

/* var ~ eps fixture (codebase_eps_mutation_vacuity): with var == eps == 1e-5
 * the reconstructed output variance is var/(var+eps) == 0.5 — the ONLY regime
 * where eps placement and the input scale are visible:
 *   sqrt(var)+eps   -> ratio ~ 0.994 (not 0.5)
 *   eps omitted     -> ratio ~ 1.0
 *   s_x dropped     -> standardization is scale-invariant, so this mutation is
 *                      invisible to every O(1)-variance test; here the
 *                      mantissa-domain variance (~1.1e4) dwarfs eps -> ~1.0.
 * d = sqrt(1.5e-5) so var = 2d^2/3 = 1e-5 exactly; N=3 (N=2 would saturate
 * xhat to +-1 regardless of data). */
void testSymForwardVarNearEpsReconstructsHalf(void) {
    float d = 0.0038729833f; /* sqrt(1.5e-5) */
    size_t dims[] = {3};
    float vals[] = {1.0f - d, 1.0f, 1.0f + d};
    tensor_t *in = buildSymInt32TensorND(1, dims, vals);
    tensor_t *out = buildSymInt32TensorND(1, dims, NULL);

    size_t ns[] = {3};
    parameter_t *gamma = buildSymParam(1, ns, (float[]){1.f, 1.f, 1.f});
    parameter_t *beta = buildSymParam(1, ns, (float[]){0.f, 0.f, 0.f});
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 3;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, in, out);

    float scale = symScaleOf(out);
    float deq[3];
    for (size_t i = 0; i < 3; i++) {
        deq[i] = (float)((int32_t *)out->data)[i] * scale;
    }

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(out);
    freeTensor(in);

    float mean = (deq[0] + deq[1] + deq[2]) / 3.f;
    float var = 0.f;
    for (size_t i = 0; i < 3; i++) {
        float dd = deq[i] - mean;
        var += dd * dd;
    }
    var /= 3.f;

    /* Input quantization perturbs d by <= s_x/2 ~ 1.5e-5 absolute (~0.4%
     * of d) -> var ratio lands at 0.5 +- ~0.02; 0.05 tolerance is ample. */
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.5f, var);
}

/* Layout-agnosticism through quantization: same logical [2,4] content,
 * contiguous vs physically transposed ([4,2] buffer + transposeTensor), must
 * yield bit-identical mantissas at corresponding LOGICAL positions and the
 * identical output scale (same float ops in the same order). Physical [4,2]
 * buffer for logical [2,4]: buf[c*2 + r] = logical(r,c). */
void testSymForwardTransposedMatchesContiguous(void) {
    float gammaVals[] = {1.f, -0.5f, 2.f, 0.25f};
    float betaVals[] = {0.1f, -0.2f, 0.3f, -0.4f};

    size_t dims[] = {2, 4};
    tensor_t *inC =
        buildSymInt32TensorND(2, dims, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f});
    tensor_t *outC = buildSymInt32TensorND(2, dims, NULL);

    size_t pdims[] = {4, 2};
    tensor_t *inT =
        buildSymInt32TensorND(2, pdims, (float[]){1.f, 10.f, 2.f, 20.f, 3.f, 30.f, 4.f, 40.f});
    transposeTensor(inT, 0, 1); /* logical [2,4], physical unchanged */
    tensor_t *outT = buildSymInt32TensorND(2, pdims, NULL);
    transposeTensor(outT, 0, 1);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, gammaVals);
    parameter_t *beta = buildSymParam(1, ns, betaVals);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitFloat();
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormForward(&layer, inC, outC);
    layerNormForward(&layer, inT, outT);

    int32_t mC[8];
    int32_t mT[8];
    int32_t *ocd = (int32_t *)outC->data;
    int32_t *otd = (int32_t *)outT->data;
    for (size_t r = 0; r < 2; r++) {
        for (size_t c = 0; c < 4; c++) {
            mC[r * 4 + c] = ocd[r * 4 + c]; /* contiguous: phys == logical */
            mT[r * 4 + c] = otd[c * 2 + r]; /* transposed: logical (r,c) -> phys c*2+r */
        }
    }
    float scaleC = symScaleOf(outC);
    float scaleT = symScaleOf(outT);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(outT);
    freeTensor(inT);
    freeTensor(outC);
    freeTensor(inC);

    TEST_ASSERT_EQUAL_FLOAT(scaleC, scaleT);
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_INT(mC[i], mT[i]);
    }
}

void testFactorySymInt32StorageQuantizesGammaBeta(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 0.0f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwdMath = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = symQ, .backwardMath = bwdMath, .weightStorage = symQ, .biasStorage = symQ};

    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

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
    size_t dims[] = {4};
    tensor_t *in = buildSymInt32TensorND(1, dims, (float[]){1.f, 2.f, 3.f, 4.f});
    tensor_t *out = buildSymInt32TensorND(1, dims, NULL);
    layerFunctions[LAYERNORM].forward(layer, in, out);
    float outScale = symScaleOf(out);
    float deqMean = 0.f;
    for (size_t i = 0; i < 4; i++) {
        deqMean += (float)((int32_t *)out->data)[i] * outScale;
    }
    deqMean /= 4.f;

    freeTensor(out);
    freeTensor(in);
    freeLayerNormLayer(layer);
    freeQuantization(bwdMath);
    freeQuantization(symQ);

    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_INT(32767, g[i]);
        TEST_ASSERT_EQUAL_INT(0, b[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f / 32767.0f, gScale);
    TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, bScale);
    TEST_ASSERT_TRUE(gammaGradFloat);
    TEST_ASSERT_TRUE(betaGradFloat);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 0.f, deqMean);
}

void testFactoryOwningSymInt32DeepCopiesQuantizations(void) {
    size_t normShape[] = {3};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bwdMath = quantizationInitFloat();
    layerQuant_t lq = {
        .forwardMath = symQ, .backwardMath = bwdMath, .weightStorage = symQ, .biasStorage = symQ};

    layer_t *layer = layerNormLayerInitOwning(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    bool fwdIsCopy = (cfg->forwardQ != symQ);
    bool fwdIsSym = (cfg->forwardQ->type == SYM_INT32);
    bool fwdCfgIsCopy = (cfg->forwardQ->qConfig != symQ->qConfig);
    bool owns = cfg->ownsQuantizations;

    /* Caller drops its quants immediately — the layer holds deep copies
     * (incl. the symInt32QConfig_t; double-free/UAF surfaces under CI ASan). */
    freeQuantization(bwdMath);
    freeQuantization(symQ);
    freeLayerNormLayer(layer);

    TEST_ASSERT_TRUE(fwdIsCopy);
    TEST_ASSERT_TRUE(fwdIsSym);
    TEST_ASSERT_TRUE(fwdCfgIsCopy);
    TEST_ASSERT_TRUE(owns);
}

/* Spec-mandated freeze-the-scale test: the propLoss scale is data-dependent
 * and must be RE-derived on every call. Two backwards with dy2 = 10x dy1 on
 * the same layer/propLoss; each call must match its own golds. */
void testSymBackwardPropLossScaleRefreshedEveryCall(void) {
    size_t dims[] = {2, 4};
    tensor_t *fwdIn = buildSymInt32TensorND(2, dims, input_layerNormSymBwd_bwdTwoCalls);
    tensor_t *loss1 = buildSymInt32TensorND(2, dims, lossGrad_layerNormSymBwd_bwdTwoCalls);
    tensor_t *loss2 = buildSymInt32TensorND(2, dims, lossGrad2_layerNormSymBwd_bwdTwoCalls);
    tensor_t *propLoss = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, gamma_layerNormSymBwd_bwdTwoCalls);
    parameter_t *beta = buildSymParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss1, propLoss);
    int32_t dx1[8];
    for (size_t i = 0; i < 8; i++) {
        dx1[i] = ((int32_t *)propLoss->data)[i];
    }
    float s1 = symScaleOf(propLoss);

    layerNormBackward(&layer, fwdIn, loss2, propLoss);
    int32_t dx2[8];
    for (size_t i = 0; i < 8; i++) {
        dx2[i] = ((int32_t *)propLoss->data)[i];
    }
    float s2 = symScaleOf(propLoss);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss2);
    freeTensor(loss1);
    freeTensor(fwdIn);

    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_INT_WITHIN(dxMantissaTol_layerNormSymBwd_bwdTwoCalls,
                               expectedDx_layerNormSymBwd_bwdTwoCalls[i], dx1[i]);
        TEST_ASSERT_INT_WITHIN(dxMantissaTol_layerNormSymBwd_bwdTwoCalls,
                               expectedDx2_layerNormSymBwd_bwdTwoCalls[i], dx2[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(expectedDxScale_layerNormSymBwd_bwdTwoCalls * 1e-4f,
                             expectedDxScale_layerNormSymBwd_bwdTwoCalls, s1);
    TEST_ASSERT_FLOAT_WITHIN(expectedDxScale2_layerNormSymBwd_bwdTwoCalls * 1e-4f,
                             expectedDxScale2_layerNormSymBwd_bwdTwoCalls, s2);
}

/* Strategy-A accumulation across microbatch calls: grads after two backwards
 * must match the emulated two-call accumulation (incl. the intermediate
 * requant) and stay within the analytic dequant bound of autograd's sum. */
void testSymBackwardGradsAccumulateAcrossCalls(void) {
    size_t dims[] = {2, 4};
    tensor_t *fwdIn = buildSymInt32TensorND(2, dims, input_layerNormSymBwd_bwdTwoCalls);
    tensor_t *loss1 = buildSymInt32TensorND(2, dims, lossGrad_layerNormSymBwd_bwdTwoCalls);
    tensor_t *loss2 = buildSymInt32TensorND(2, dims, lossGrad2_layerNormSymBwd_bwdTwoCalls);
    tensor_t *propLoss = buildSymInt32TensorND(2, dims, NULL);

    size_t ns[] = {4};
    parameter_t *gamma = buildSymParam(1, ns, gamma_layerNormSymBwd_bwdTwoCalls);
    parameter_t *beta = buildSymParam(1, ns, NULL);
    size_t *normShape = reserveMemory(sizeof(size_t));
    normShape[0] = 4;

    quantization_t *fq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bq = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfg;
    initLayerNormConfig(&cfg, gamma, beta, normShape, 1, 1e-5f, fq, bq);
    layerConfig_t lcfg;
    layer_t layer = makeLayerNormLayer(&cfg, &lcfg);

    layerNormBackward(&layer, fwdIn, loss1, propLoss);
    layerNormBackward(&layer, fwdIn, loss2, propLoss);

    int32_t dgm[4], dbm[4];
    for (size_t i = 0; i < 4; i++) {
        dgm[i] = ((int32_t *)gamma->grad->data)[i];
        dbm[i] = ((int32_t *)beta->grad->data)[i];
    }
    float dgScale = symScaleOf(gamma->grad);
    float dbScale = symScaleOf(beta->grad);

    freeQuantization(bq);
    freeQuantization(fq);
    freeReservedMemory(normShape);
    freeParameter(beta);
    freeParameter(gamma);
    freeTensor(propLoss);
    freeTensor(loss2);
    freeTensor(loss1);
    freeTensor(fwdIn);

    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_INT_WITHIN(gradMantissaTol_layerNormSymBwd_bwdTwoCalls,
                               expectedDgammaAfter2_layerNormSymBwd_bwdTwoCalls[i], dgm[i]);
        TEST_ASSERT_INT_WITHIN(gradMantissaTol_layerNormSymBwd_bwdTwoCalls,
                               expectedDbetaAfter2_layerNormSymBwd_bwdTwoCalls[i], dbm[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(expectedDgammaScaleAfter2_layerNormSymBwd_bwdTwoCalls * 1e-4f,
                             expectedDgammaScaleAfter2_layerNormSymBwd_bwdTwoCalls, dgScale);
    TEST_ASSERT_FLOAT_WITHIN(expectedDbetaScaleAfter2_layerNormSymBwd_bwdTwoCalls * 1e-4f,
                             expectedDbetaScaleAfter2_layerNormSymBwd_bwdTwoCalls, dbScale);
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(gradDequantTol2_layerNormSymBwd_bwdTwoCalls,
                                 expectedDgammaDequant2_layerNormSymBwd_bwdTwoCalls[i],
                                 (float)dgm[i] * dgScale);
        TEST_ASSERT_FLOAT_WITHIN(gradDequantTol2_layerNormSymBwd_bwdTwoCalls,
                                 expectedDbetaDequant2_layerNormSymBwd_bwdTwoCalls[i],
                                 (float)dbm[i] * dbScale);
    }
}

/* var == eps == 1e-5 (codebase memory eps-mutation vacuity): the ONLY regime
 * where eps placement inside the backward's recomputed invSigma is visible —
 * O(1)-variance fixtures see a ~5e-6 relative shift (far below 1 dx-LSB).
 * Here sqrt(var)+eps vs sqrt(var+eps) shifts invSigma ~41%. */
void testSymBackwardVarNearEpsGold(void) {
    size_t dims[] = {3};
    size_t ns[] = {3};
    runSymGoldBackward(
        1, dims, 1, ns, input_layerNormSymBwd_bwdVarEps, gamma_layerNormSymBwd_bwdVarEps,
        lossGrad_layerNormSymBwd_bwdVarEps, expectedDx_layerNormSymBwd_bwdVarEps,
        expectedDxScale_layerNormSymBwd_bwdVarEps, dxMantissaTol_layerNormSymBwd_bwdVarEps,
        expectedDxDequant_layerNormSymBwd_bwdVarEps, dxDequantTol_layerNormSymBwd_bwdVarEps,
        expectedDgamma_layerNormSymBwd_bwdVarEps, expectedDgammaScale_layerNormSymBwd_bwdVarEps,
        expectedDbeta_layerNormSymBwd_bwdVarEps, expectedDbetaScale_layerNormSymBwd_bwdVarEps,
        gradMantissaTol_layerNormSymBwd_bwdVarEps, expectedDgammaDequant_layerNormSymBwd_bwdVarEps,
        expectedDbetaDequant_layerNormSymBwd_bwdVarEps, gradDequantTol_layerNormSymBwd_bwdVarEps, 3,
        3);
}

/* Layout-agnosticism: run bwdBase contiguous, then with forwardInput AND
 * propLoss physically transposed ([4,3] buffers + transposeTensor) while dy
 * stays natural — three DIVERGENT maps. Same logical iteration order => same
 * float ops => mantissas (at logical positions), scales, and grads must be
 * BIT-identical. Catches: x read through the wrong map, dy read through x's
 * map, dx scattered row-major instead of through propLoss's map.
 * (PR-1's testBackwardFloatDivergentLayouts is the float precedent.) */
void testSymBackwardDivergentLayoutsBitIdentical(void) {
    size_t dims[] = {3, 4};
    size_t ns[] = {4};

    /* --- contiguous run --- */
    tensor_t *fwdC = buildSymInt32TensorND(2, dims, input_layerNormSymBwd_bwdBase);
    tensor_t *lossC = buildSymInt32TensorND(2, dims, lossGrad_layerNormSymBwd_bwdBase);
    tensor_t *propC = buildSymInt32TensorND(2, dims, NULL);
    parameter_t *gammaC = buildSymParam(1, ns, gamma_layerNormSymBwd_bwdBase);
    parameter_t *betaC = buildSymParam(1, ns, NULL);
    size_t *normShapeC = reserveMemory(sizeof(size_t));
    normShapeC[0] = 4;
    quantization_t *fqC = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bqC = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfgC;
    initLayerNormConfig(&cfgC, gammaC, betaC, normShapeC, 1, 1e-5f, fqC, bqC);
    layerConfig_t lcfgC;
    layer_t layerC = makeLayerNormLayer(&cfgC, &lcfgC);
    layerNormBackward(&layerC, fwdC, lossC, propC);

    /* --- divergent run: x and propLoss physical [4,3], logical [3,4] --- */
    float xT[12];
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 4; c++) {
            xT[c * 3 + r] = input_layerNormSymBwd_bwdBase[r * 4 + c];
        }
    }
    size_t pdims[] = {4, 3};
    tensor_t *fwdT = buildSymInt32TensorND(2, pdims, xT);
    transposeTensor(fwdT, 0, 1); /* logical [3,4], physical unchanged */
    tensor_t *lossT = buildSymInt32TensorND(2, dims, lossGrad_layerNormSymBwd_bwdBase);
    tensor_t *propT = buildSymInt32TensorND(2, pdims, NULL);
    transposeTensor(propT, 0, 1);
    parameter_t *gammaT = buildSymParam(1, ns, gamma_layerNormSymBwd_bwdBase);
    parameter_t *betaT = buildSymParam(1, ns, NULL);
    size_t *normShapeT = reserveMemory(sizeof(size_t));
    normShapeT[0] = 4;
    quantization_t *fqT = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *bqT = quantizationInitSymInt32(HALF_AWAY);
    layerNormConfig_t cfgT;
    initLayerNormConfig(&cfgT, gammaT, betaT, normShapeT, 1, 1e-5f, fqT, bqT);
    layerConfig_t lcfgT;
    layer_t layerT = makeLayerNormLayer(&cfgT, &lcfgT);
    layerNormBackward(&layerT, fwdT, lossT, propT);

    int32_t dxC[12], dxT[12], dgC[4], dgT[4], dbC[4], dbT[4];
    int32_t *pcd = (int32_t *)propC->data;
    int32_t *ptd = (int32_t *)propT->data;
    for (size_t r = 0; r < 3; r++) {
        for (size_t c = 0; c < 4; c++) {
            dxC[r * 4 + c] = pcd[r * 4 + c]; /* contiguous: phys == logical */
            dxT[r * 4 + c] = ptd[c * 3 + r]; /* transposed: logical (r,c) -> phys c*3+r */
        }
    }
    for (size_t i = 0; i < 4; i++) {
        dgC[i] = ((int32_t *)gammaC->grad->data)[i];
        dgT[i] = ((int32_t *)gammaT->grad->data)[i];
        dbC[i] = ((int32_t *)betaC->grad->data)[i];
        dbT[i] = ((int32_t *)betaT->grad->data)[i];
    }
    float sC = symScaleOf(propC);
    float sT = symScaleOf(propT);
    float dgsC = symScaleOf(gammaC->grad);
    float dgsT = symScaleOf(gammaT->grad);

    freeQuantization(bqT);
    freeQuantization(fqT);
    freeReservedMemory(normShapeT);
    freeParameter(betaT);
    freeParameter(gammaT);
    freeTensor(propT);
    freeTensor(lossT);
    freeTensor(fwdT);
    freeQuantization(bqC);
    freeQuantization(fqC);
    freeReservedMemory(normShapeC);
    freeParameter(betaC);
    freeParameter(gammaC);
    freeTensor(propC);
    freeTensor(lossC);
    freeTensor(fwdC);

    TEST_ASSERT_EQUAL_FLOAT(sC, sT);
    TEST_ASSERT_EQUAL_FLOAT(dgsC, dgsT);
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_EQUAL_INT(dxC[i], dxT[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_INT(dgC[i], dgT[i]);
        TEST_ASSERT_EQUAL_INT(dbC[i], dbT[i]);
    }
}

/* Full-SYM profile (forwardMath = backwardMath = storage = SYM_INT32) through
 * the Borrowing factory: gradInit must yield SYM_INT32 grads (PR-0 plumbing),
 * and a vtable backward on factory-built params must accumulate into them.
 * (The new validation rules are death paths — liveness-checked in this task's
 * inversion step, not Unity-testable.) */
void testFactoryFullSymProfileTrainsSymGrads(void) {
    size_t normShape[] = {4};
    layerNormInit_t init = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};

    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq = {
        .forwardMath = symQ, .backwardMath = symQ, .weightStorage = symQ, .biasStorage = symQ};
    layer_t *layer = layerNormLayerInit(&init, &lq);
    layerNormConfig_t *cfg = layer->config->layerNorm;

    bool gradsSym = (cfg->gamma->grad->quantization->type == SYM_INT32) &&
                    (cfg->beta->grad->quantization->type == SYM_INT32);

    size_t dims[] = {2, 4};
    tensor_t *fwdIn =
        buildSymInt32TensorND(2, dims, (float[]){1.f, 2.f, 3.f, 4.f, 10.f, 20.f, 30.f, 40.f});
    tensor_t *loss = buildSymInt32TensorND(
        2, dims, (float[]){0.5f, -1.f, 0.25f, 0.75f, -0.5f, 1.5f, -0.25f, 1.f});
    tensor_t *propLoss = buildSymInt32TensorND(2, dims, NULL);

    layerFunctions[LAYERNORM].backward(layer, fwdIn, loss, propLoss);

    int32_t dbSum = 0;
    for (size_t i = 0; i < 4; i++) {
        int32_t v = ((int32_t *)cfg->beta->grad->data)[i];
        dbSum += (v < 0) ? -v : v;
    }
    float dxScale = symScaleOf(propLoss);

    freeTensor(propLoss);
    freeTensor(loss);
    freeTensor(fwdIn);
    freeLayerNormLayer(layer);
    freeQuantization(symQ);

    TEST_ASSERT_TRUE(gradsSym);
    TEST_ASSERT_TRUE_MESSAGE(dbSum > 0, "backward must accumulate into the factory SYM grads");
    TEST_ASSERT_TRUE(dxScale > 0.0f && dxScale != 1.0f);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testConfigStructIsPopulated);
    RUN_TEST(testCalcOutputShapeIsIdentity);
    RUN_TEST(testForwardFloatSingleGroup);
    RUN_TEST(testForwardFloatMultiGroup);
    RUN_TEST(testForwardFloatTransposedInput);
    RUN_TEST(testVtableForwardFloatSingleGroup);
    RUN_TEST(testBackwardFloatSingleGroup);
    RUN_TEST(testBackwardFloatMultiGroupAccumulatesGradients);
    RUN_TEST(testBackwardFloatDivergentLayouts);
    RUN_TEST(testFactoryBuildsGammaOnesBetaZerosAndForwards);
    RUN_TEST(testFactoryOwningDeepCopiesQuantizations);
    RUN_TEST(testFactoryBorrowingDoesNotFreeCallerQuantizations);
    RUN_TEST(testFactoryRank2NormShapeAndExplicitEps);
    RUN_TEST(testForwardFloatSmallVarianceEpsInsideSqrt);
    RUN_TEST(testGoldForwardD1SingleGroup);
    RUN_TEST(testGoldBackwardD1SingleGroup);
    RUN_TEST(testGoldForwardD1MultiGroup);
    RUN_TEST(testGoldBackwardD1MultiGroup);
    RUN_TEST(testGoldForwardD2MultiGroup);
    RUN_TEST(testGoldBackwardD2MultiGroup);
    RUN_TEST(testGoldForwardFullRank);
    RUN_TEST(testGoldBackwardFullRank);
    RUN_TEST(testSymForwardDequantInvariantsMultiGroup);
    RUN_TEST(testSymForwardConstantInputEmitsZeros);
    RUN_TEST(testSymForwardMantissaInvariantsViaOnesGamma);
    RUN_TEST(testSymForwardConstantInputAppliesBeta);
    RUN_TEST(testSymGoldParityMultiGroup);
    RUN_TEST(testSymGoldAffine);
    RUN_TEST(testSymGoldSigmaRatio10x);
    RUN_TEST(testSymForwardVarNearEpsReconstructsHalf);
    RUN_TEST(testSymForwardTransposedMatchesContiguous);
    RUN_TEST(testFactorySymInt32StorageQuantizesGammaBeta);
    RUN_TEST(testFactoryOwningSymInt32DeepCopiesQuantizations);
    RUN_TEST(testSymBackwardGoldMultiGroup);
    RUN_TEST(testSymBackwardZeroLossGradEmitsZerosNeutralScale);
    RUN_TEST(testSymBackwardPropLossScaleRefreshedEveryCall);
    RUN_TEST(testSymBackwardGradsAccumulateAcrossCalls);
    RUN_TEST(testSymBackwardVarNearEpsGold);
    RUN_TEST(testSymBackwardDivergentLayoutsBitIdentical);
    RUN_TEST(testFactoryFullSymProfileTrainsSymGrads);
    return UNITY_END();
}
