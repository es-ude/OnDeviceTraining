#define SOURCE_FILE "UNIT_TEST_PPCA_REPLAY"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "DeathTest.h"
#include "ExecuteOp.h"
#include "PpcaReplay.h"
#include "PpcaReplayApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildFloat32TensorND(size_t rank, const size_t *srcDims, const float *fill) {
    size_t *dims = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        dims[i] = srcDims[i];
        order[i] = i;
    }
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = rank;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (fill != NULL) {
        float *d = (float *)t->data;
        size_t n = calcNumberOfElementsByTensor(t);
        for (size_t i = 0; i < n; i++) {
            d[i] = fill[i];
        }
    }
    return t;
}

static ppcaReplayConfig_t floatConfig(size_t dim, size_t rank, size_t maxM) {
    static quantization_t floatQ; /* static: outlives the call; cloned by create */
    initFloat32Quantization(&floatQ);
    ppcaReplayConfig_t cfg = {
        .dim = dim,
        .rank = rank,
        .maxSessionSamples = maxM,
        .mergeMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .streamMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .sampleMath = {.type = ARITH_FLOAT32, .roundingMode = HALF_AWAY},
        .meanQ = &floatQ,
        .basisQ = &floatQ,
        .eigvalsQ = &floatQ,
        .sigma2Floor = 1e-6f,
        .shrinkageGamma = 0.0f,
    };
    return cfg;
}

void testCreateInitializesState(void) {
    ppcaReplayConfig_t cfg = floatConfig(16, 4, 32);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_EQUAL_size_t(16, g->dim);
    TEST_ASSERT_EQUAL_size_t(4, g->rank);
    TEST_ASSERT_EQUAL_UINT32(0, g->count);
    TEST_ASSERT_EQUAL_FLOAT(1e-6f, g->sigma2);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, g->totalVar);
    TEST_ASSERT_EQUAL_size_t(16, g->mean->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(4, g->basis->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(16, g->basis->shape->dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(4, g->eigvals->shape->dimensions[0]);
    /* zero-initialized state */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, ((float *)g->basis->data)[0]);
    freePpcaReplay(g);
}

void testWorkspaceBytesFormula(void) {
    /* p = k+m+1. Inventory (floats): bT p*d + gram p^2 + eigvecs p^2 +
     * theta p + lambdaOut k + rowScales p + meanBatch d + muOld d + u d. */
    size_t d = 16, k = 4, m = 32, p = k + m + 1;
    size_t floats = p * d + p * p + p * p + p + k + p + d + d + d;
    TEST_ASSERT_EQUAL_size_t(floats * sizeof(float), ppcaWorkspaceBytes(d, k, m));
}

void testWorkspaceCreateMatchesBytes(void) {
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(16, 4, 32);
    TEST_ASSERT_NOT_NULL(ws);
    TEST_ASSERT_EQUAL_size_t(16, ws->dim);
    TEST_ASSERT_NOT_NULL(ws->bT);
    TEST_ASSERT_NOT_NULL(ws->u);
    freePpcaWorkspace(ws);
}

void testReplayBytesFloat32(void) {
    ppcaReplayConfig_t cfg = floatConfig(16, 4, 32);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    /* FLOAT32: (d + k*d + k) * 4 + struct overhead. */
    size_t expected = (16 + 4 * 16 + 4) * sizeof(float) + sizeof(ppcaReplay_t);
    TEST_ASSERT_EQUAL_size_t(expected, ppcaReplayBytes(g));
    freePpcaReplay(g);
}

void testIsoExemplarCount(void) {
    ppcaReplayConfig_t cfg = floatConfig(16, 4, 32);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    size_t bytes = ppcaReplayBytes(g);
    TEST_ASSERT_EQUAL_size_t(bytes / (16 * sizeof(float)),
                             ppcaReplayIsoExemplarCount(g, 16 * sizeof(float)));
    freePpcaReplay(g);
}

void testSetCreate(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    ppcaReplaySet_t *set = ppcaReplaySetCreate(3, &cfg);
    TEST_ASSERT_EQUAL_size_t(3, set->numClasses);
    TEST_ASSERT_NOT_NULL(set->generators[2]);
    TEST_ASSERT_NOT_NULL(set->workspace);
    freePpcaReplaySet(set);
}

void testCreateRejectsZeroDim(void) {
    ppcaReplayConfig_t cfg = floatConfig(0, 2, 16);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsRankGeDim(void) {
    ppcaReplayConfig_t cfg = floatConfig(4, 4, 16);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsNanSigma2Floor(void) {
    /* PR #366 hardening: sigma2Floor seeds sigma2 AND floors every merge --
     * a NaN (or non-positive) floor silently collapses generated samples
     * toward the class mean (sigma-noise term dies), poisoning replay
     * research results with no crash. Fail fast at create, NaN-robust
     * (adamWInit betas idiom). */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    cfg.sigma2Floor = NAN;
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsNonPositiveSigma2Floor(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    cfg.sigma2Floor = 0.0f;
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsSymInt32Storage(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    symInt32QConfig_t qc;
    initSymInt32QConfig(HALF_AWAY, &qc);
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    cfg.basisQ = &q;
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsInt32Storage(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    quantization_t q;
    initInt32Quantization(&q);
    cfg.meanQ = &q;
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateRejectsBoolStorage(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    quantization_t q;
    initBoolQuantization(&q);
    cfg.eigvalsQ = &q;
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testCreateAcceptsBfpStorage(void) {
    /* BFP epic PR3 Task 6: the whitelist gains per-tensor BFP -- the
     * "arrives with BFP epic PR3" promise in validateStateStorage's comment
     * has now landed. getQLike (already BFP-aware since Task 1/5) clones the
     * template into each state tensor's own quantization. */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
    cfg.basisQ = bfpQ;
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    freeQuantization(bfpQ); /* getQLike clones -- template unused after */

    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_EQUAL_INT(BFP, g->basis->quantization->type);
    freePpcaReplay(g);
}

void testCreateRejectsSymInt32Arithmetic(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    cfg.mergeMath = (arithmetic_t){.type = ARITH_SYM_INT32, .roundingMode = HALF_AWAY};
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayCreate(&cfg));
}

void testWorkspaceCreateRejectsZeroSessionBound(void) {
    ASSERT_EXITS_WITH_FAILURE(ppcaWorkspaceCreate(8, 2, 0));
}

/* Hand-assemble a FLOAT32 generator in a known state (test seam: state
 * tensors are plain FLOAT32 buffers we can write directly). */
static ppcaReplay_t *buildKnownGenerator(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    float *basis = (float *)g->basis->data;
    float s = 1.0f / sqrtf(2.0f);
    /* row0 = (s,s,0,0,0,0), row1 = (0,0,s,-s,0,0) — orthonormal. */
    basis[0] = s;
    basis[1] = s;
    basis[1 * 6 + 2] = s;
    basis[1 * 6 + 3] = -s;
    float *lam = (float *)g->eigvals->data;
    lam[0] = 4.0f;
    lam[1] = 1.0f;
    float *mu = (float *)g->mean->data;
    for (size_t j = 0; j < 6; j++) {
        mu[j] = (float)j;
    }
    g->sigma2 = 0.25f;
    g->count = 100;
    return g;
}

void testSampleDeterministicAndGlobalStreamUntouched(void) {
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *o1 = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    tensor_t *o2 = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rngSetSeed(4711);
    uint32_t globalBefore = rngGetSeed();
    rng32_t a = {.state = 99};
    rng32_t b = {.state = 99};
    ppcaReplaySample(g, &a, o1);
    ppcaReplaySample(g, &b, o2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)o1->data, (float *)o2->data, 6);
    TEST_ASSERT_EQUAL_UINT32(globalBefore, rngGetSeed());
    freeTensor(o2);
    freeTensor(o1);
    freePpcaReplay(g);
}

void testSampleGoldenConstants(void) {
    /* Host-vs-host determinism (spec §2.6): pinned constants. AFTER the
     * implementation first passes the other tests, run once with the
     * printf below, paste the values, delete the printf. */
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 424242};
    ppcaReplaySample(g, &stream, out);
    float *v = (float *)out->data;
    /* for (size_t j = 0; j < 6; j++) printf("%.9gf,\n", (double)v[j]); */
    float golden[6] = {2.6510756f,  2.71321058f, 3.19746828f,
                       3.02661514f, 3.7248354f,  4.89183283f}; /* pinned 2026-07-12 */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(golden, v, 6);
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleCovarianceMatchesModel(void) {
    /* T3: empirical stats over N=20000 draws. Variance along basis row i
     * must approach lambda_i; along an orthogonal direction sigma2; the
     * mean must approach mu. */
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 20260712};
    enum { N = 20000 };
    float s = 1.0f / sqrtf(2.0f);
    float dir0[6] = {s, s, 0, 0, 0, 0};
    float dir1[6] = {0, 0, s, -s, 0, 0};
    float orth[6] = {0, 0, 0, 0, 1.0f, 0};
    double sum0 = 0, sq0 = 0, sum1 = 0, sq1 = 0, sumO = 0, sqO = 0, sumMean = 0;
    for (size_t n = 0; n < N; n++) {
        ppcaReplaySample(g, &stream, out);
        float *v = (float *)out->data;
        float p0 = 0, p1 = 0, pO = 0;
        for (size_t j = 0; j < 6; j++) {
            float centered = v[j] - (float)j; /* mu[j] = j */
            p0 += centered * dir0[j];
            p1 += centered * dir1[j];
            pO += centered * orth[j];
        }
        sum0 += p0;
        sq0 += (double)p0 * p0;
        sum1 += p1;
        sq1 += (double)p1 * p1;
        sumO += pO;
        sqO += (double)pO * pO;
        sumMean += v[0];
    }
    /* double accumulators are TEST-side only (framework code stays float). */
    float var0 = (float)(sq0 / N - (sum0 / N) * (sum0 / N));
    float var1 = (float)(sq1 / N - (sum1 / N) * (sum1 / N));
    float varO = (float)(sqO / N - (sumO / N) * (sumO / N));
    TEST_ASSERT_FLOAT_WITHIN(0.15f, 4.0f, var0);
    TEST_ASSERT_FLOAT_WITHIN(0.06f, 1.0f, var1);
    TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.25f, varO);
    TEST_ASSERT_FLOAT_WITHIN(0.03f, 0.0f, (float)(sumMean / N)); /* mu[0] = 0 */
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleStreamPositionIndependentOfKeff(void) {
    /* Always k+d draws: two generators differing only in count (k_eff 1 vs 2)
     * must leave an identical stream position. */
    ppcaReplay_t *g1 = buildKnownGenerator();
    ppcaReplay_t *g2 = buildKnownGenerator();
    g2->count = 2; /* k_eff = min(2, 1) = 1 */
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t a = {.state = 5};
    rng32_t b = {.state = 5};
    ppcaReplaySample(g1, &a, out);
    ppcaReplaySample(g2, &b, out);
    TEST_ASSERT_EQUAL_UINT32(a.state, b.state);
    freeTensor(out);
    freePpcaReplay(g2);
    freePpcaReplay(g1);
}

void testSampleCountOneIsMeanPlusNoise(void) {
    /* k_eff = 0: out = mu + sigma*eps only — variance sigma2 everywhere. */
    ppcaReplay_t *g = buildKnownGenerator();
    g->count = 1;
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 7};
    double sq = 0;
    enum { N = 5000 };
    for (size_t n = 0; n < N; n++) {
        ppcaReplaySample(g, &stream, out);
        float centered = ((float *)out->data)[0] - 0.0f;
        sq += (double)centered * centered;
    }
    TEST_ASSERT_FLOAT_WITHIN(0.03f, 0.25f, (float)(sq / N));
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleZeroSteadyStateAllocation(void) {
    /* T6: no reserveMemory growth across repeated samples. Under builds
     * without ODT_MEM_PROFILE the counters are no-op zeros (vacuously
     * passing); the memprofile examples preset makes it a real assertion. */
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 13};
    ppcaReplaySample(g, &stream, out); /* warm-up */
    size_t before = memProfileMark();
    for (size_t n = 0; n < 100; n++) {
        ppcaReplaySample(g, &stream, out);
    }
    TEST_ASSERT_EQUAL_size_t(before, memProfileMark());
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleRejectsCountZero(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 1};
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySample(g, &stream, out));
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleRejectsElementCountMismatch(void) {
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){5}, NULL);
    rng32_t stream = {.state = 1};
    ASSERT_EXITS_WITH_FAILURE(ppcaReplaySample(g, &stream, out));
    freeTensor(out);
    freePpcaReplay(g);
}

void testSampleAcceptsReshapedOut(void) {
    /* [1,2,3] has 6 elements == dim: allowed (natural-shape pool tensors). */
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(3, (size_t[]){1, 2, 3}, NULL);
    rng32_t stream = {.state = 11};
    ppcaReplaySample(g, &stream, out);
    freeTensor(out);
    freePpcaReplay(g);
}

void testMeanReplayIsExactCentroid(void) {
    /* Mean-replay baseline (#326 fair comparison): out == mu bit-for-bit;
     * deterministic by construction — no RNG parameter exists to consume. */
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    ppcaReplayMean(g, out);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)g->mean->data, (float *)out->data, 6);
    freeTensor(out);
    freePpcaReplay(g);
}

void testMeanReplayRejectsCountZero(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayMean(g, out));
    freeTensor(out);
    freePpcaReplay(g);
}

void testMeanReplayRejectsElementCountMismatch(void) {
    ppcaReplay_t *g = buildKnownGenerator();
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){5}, NULL);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayMean(g, out));
    freeTensor(out);
    freePpcaReplay(g);
}

/* Draw m samples x = mu + sum_i sqrt(lamTotal_i - sig2)*z_i*u_i + sig*eps
 * into a [m,d] FLOAT32 tensor. basisRows: kTrue x d orthonormal. */
static tensor_t *drawSyntheticSamples(size_t m, size_t d, const float *mu, const float *basisRows,
                                      size_t kTrue, const float *lamTotal, float sigma2,
                                      rng32_t *rng) {
    tensor_t *t = buildFloat32TensorND(2, (size_t[]){m, d}, NULL);
    float *x = (float *)t->data;
    float sigma = sqrtf(sigma2);
    for (size_t s = 0; s < m; s++) {
        for (size_t j = 0; j < d; j++) {
            x[s * d + j] = mu[j];
        }
        for (size_t i = 0; i < kTrue; i++) {
            float z = randomNormalCtx(rng, 0.0f, 1.0f);
            float coeff = sqrtf(lamTotal[i] - sigma2) * z;
            for (size_t j = 0; j < d; j++) {
                x[s * d + j] += coeff * basisRows[i * d + j];
            }
        }
        for (size_t j = 0; j < d; j++) {
            x[s * d + j] += sigma * randomNormalCtx(rng, 0.0f, 1.0f);
        }
    }
    return t;
}

/* Projection of unit vector dir onto the span of the generator's kept rows. */
static float subspaceProjection(const ppcaReplay_t *g, const float *dir) {
    const float *b = (const float *)g->basis->data;
    float acc = 0.0f;
    for (size_t i = 0; i < g->rank; i++) {
        float dot = 0.0f;
        for (size_t j = 0; j < g->dim; j++) {
            dot += b[i * g->dim + j] * dir[j];
        }
        acc += dot * dot;
    }
    return acc;
}

static const float kTrueMu8[8] = {1.0f, -1.0f, 0.5f, 0.0f, 2.0f, 0.0f, -0.5f, 1.5f};
static const float kTrueBasis8[16] = {
    /* two orthonormal rows in R^8 */
    0.70710678f, 0.70710678f, 0.0f,        0.0f,         0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,        0.0f,        0.70710678f, -0.70710678f, 0.0f, 0.0f, 0.0f, 0.0f};
static const float kTrueLam8[2] = {9.0f, 4.0f};

void testUpdateBatchFitRecoversModel(void) { /* T1: n=0 bootstrap == batch fit */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 256);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 256);
    rng32_t rng = {.state = 20260712};
    tensor_t *samples =
        drawSyntheticSamples(200, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);

    ppcaReplayUpdate(g, samples, ws);

    TEST_ASSERT_EQUAL_UINT32(200, g->count);
    TEST_ASSERT_TRUE(subspaceProjection(g, &kTrueBasis8[0]) > 0.99f);
    TEST_ASSERT_TRUE(subspaceProjection(g, &kTrueBasis8[8]) > 0.99f);
    float *lam = (float *)g->eigvals->data;
    TEST_ASSERT_FLOAT_WITHIN(2.0f, 9.0f, lam[0]);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 4.0f, lam[1]);
    TEST_ASSERT_TRUE(g->sigma2 > 0.01f && g->sigma2 < 0.12f);
    float *mu = (float *)g->mean->data;
    for (size_t j = 0; j < 8; j++) {
        TEST_ASSERT_FLOAT_WITHIN(0.35f, kTrueMu8[j], mu[j]);
    }
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testUpdateIncrementalEqualsPooled(void) { /* T2: the Gram-conditioning gate */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 256);
    ppcaReplay_t *gInc = ppcaReplayCreate(&cfg);
    ppcaReplay_t *gPool = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 256);
    rng32_t rng = {.state = 777};
    tensor_t *all = drawSyntheticSamples(240, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);

    ppcaReplayUpdate(gPool, all, ws);

    /* Same 240 rows as three 80-row sessions (views into all's buffer). */
    for (size_t s = 0; s < 3; s++) {
        tensor_t *part = buildFloat32TensorND(2, (size_t[]){80, 8}, NULL);
        float *src = (float *)all->data;
        float *dst = (float *)part->data;
        for (size_t i = 0; i < 80 * 8; i++) {
            dst[i] = src[s * 80 * 8 + i];
        }
        ppcaReplayUpdate(gInc, part, ws);
        freeTensor(part);
    }

    TEST_ASSERT_EQUAL_UINT32(240, gInc->count);
    /* Mutual subspace agreement between pooled and incremental bases. */
    float *bp = (float *)gPool->basis->data;
    TEST_ASSERT_TRUE(subspaceProjection(gInc, &bp[0]) > 0.98f);
    TEST_ASSERT_TRUE(subspaceProjection(gInc, &bp[8]) > 0.98f);
    float *li = (float *)gInc->eigvals->data;
    float *lp = (float *)gPool->eigvals->data;
    TEST_ASSERT_FLOAT_WITHIN(0.08f * lp[0], lp[0], li[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.12f * lp[1], lp[1], li[1]);
    float *mi = (float *)gInc->mean->data;
    float *mp = (float *)gPool->mean->data;
    for (size_t j = 0; j < 8; j++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, mp[j], mi[j]); /* mean merge is exact */
    }
    freeTensor(all);
    freePpcaWorkspace(ws);
    freePpcaReplay(gPool);
    freePpcaReplay(gInc);
}

void testUpdateMeanShiftedIncrementalEqualsPooled(void) {
    /* T2b: sessions with DIFFERENT true means (large deliberate domain
     * shift, +3.0 per session on coordinates 4..7 — orthogonal to both
     * kTrueBasis8 rows, so the spectrum stays cleanly separated:
     * shift-direction ~24, u0 9, u1 4; an every-coordinate shift would
     * overlap u0 by 0.5 and make rank-2 truncation itself diverge between
     * incremental and pooled). The between-session mean scatter enters the
     * incremental path ONLY through the correction column — dropping it
     * (corrFactor=0) loses the dominant eigenvalue entirely, so this test
     * is the mutation kill T2's same-mean sessions cannot provide. */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 256);
    ppcaReplay_t *gInc = ppcaReplayCreate(&cfg);
    ppcaReplay_t *gPool = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 256);
    rng32_t rng = {.state = 20260713};
    tensor_t *sessions[3];
    for (size_t s = 0; s < 3; s++) {
        float muS[8];
        for (size_t j = 0; j < 8; j++) {
            muS[j] = kTrueMu8[j] + ((j >= 4) ? 3.0f * (float)s : 0.0f);
        }
        sessions[s] = drawSyntheticSamples(80, 8, muS, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
    }
    /* Pooled fit sees the SAME 240 rows, concatenated. */
    tensor_t *all = buildFloat32TensorND(2, (size_t[]){240, 8}, NULL);
    for (size_t s = 0; s < 3; s++) {
        float *src = (float *)sessions[s]->data;
        float *dst = (float *)all->data;
        for (size_t i = 0; i < 80 * 8; i++) {
            dst[s * 80 * 8 + i] = src[i];
        }
    }

    ppcaReplayUpdate(gPool, all, ws);
    for (size_t s = 0; s < 3; s++) {
        ppcaReplayUpdate(gInc, sessions[s], ws);
        freeTensor(sessions[s]);
    }

    TEST_ASSERT_EQUAL_UINT32(240, gInc->count);
    float *bp = (float *)gPool->basis->data;
    TEST_ASSERT_TRUE(subspaceProjection(gInc, &bp[0]) > 0.99f);
    TEST_ASSERT_TRUE(subspaceProjection(gInc, &bp[8]) > 0.99f);
    float *li = (float *)gInc->eigvals->data;
    float *lp = (float *)gPool->eigvals->data;
    TEST_ASSERT_FLOAT_WITHIN(0.02f * lp[0], lp[0], li[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.02f * lp[1], lp[1], li[1]);
    float *mi = (float *)gInc->mean->data;
    float *mp = (float *)gPool->mean->data;
    for (size_t j = 0; j < 8; j++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, mp[j], mi[j]); /* mean merge is exact */
    }
    freeTensor(all);
    freePpcaWorkspace(ws);
    freePpcaReplay(gPool);
    freePpcaReplay(gInc);
}

void testUpdateRenormalizesIllConditionedSpectrum(void) {
    /* Renorm mutation kill: theta_2/theta_1 = 3e-6 (above the 1e-6 rank
     * guard with 3x margin). JacobiEig reads eigvals off the in-place
     * rotated diagonal while V accumulates separately, so a tiny theta
     * produced by cancellation of O(||G||) entries carries an absolute
     * error floor of ~eps*||G|| — after the rotate-back's 1/sqrt(theta_hat)
     * scaling the kept row's norm deviates by ~eps*||G||/theta_2
     * (percent-scale here). The renormalization pass (spec §5.2.4) is what
     * pins it back to 1. */
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 16);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(6, 2, 16);
    /* m=3, zero-mean by construction: scatter has exactly two directions,
     * e1 with theta_1 = 2 and e2 with theta_2 = 6e-6 (epsilon = 1e-3). */
    float e = 1e-3f;
    tensor_t *samples = buildFloat32TensorND(
        2, (size_t[]){3, 6},
        (float[]){1.0f, e, 0, 0, 0, 0, -1.0f, e, 0, 0, 0, 0, 0.0f, -2.0f * e, 0, 0, 0, 0});
    ppcaReplayUpdate(g, samples, ws);

    float *lam = (float *)g->eigvals->data;
    TEST_ASSERT_TRUE(lam[1] > 0.0f); /* theta_2 above the rank guard: row 1 kept */
    float *b = (float *)g->basis->data;
    for (size_t i = 0; i < 2; i++) {
        float normSq = 0.0f;
        for (size_t j = 0; j < 6; j++) {
            normSq += b[i * 6 + j] * b[i * 6 + j];
        }
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1.0f, normSq);
    }
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testUpdateOrthonormalAcrossTenMerges(void) { /* T4 strict (path A) */
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 64);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 64);
    rng32_t rng = {.state = 31337};
    for (size_t s = 0; s < 10; s++) {
        tensor_t *samples =
            drawSyntheticSamples(40, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
        ppcaReplayUpdate(g, samples, ws);
        freeTensor(samples);
        float *b = (float *)g->basis->data;
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
                float dot = 0.0f;
                for (size_t t = 0; t < 8; t++) {
                    dot += b[i * 8 + t] * b[j * 8 + t];
                }
                TEST_ASSERT_FLOAT_WITHIN(1e-3f, (i == j) ? 1.0f : 0.0f, dot);
            }
        }
    }
    TEST_ASSERT_EQUAL_UINT32(400, g->count);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testUpdateRankDeficientSessionNoNaN(void) {
    /* d=6, k=4, first session m=3 -> centered rank <= 2: rows 2,3 zeroed,
     * eigvals 2,3 == 0, nothing NaN; sampling still works (k_eff = 2). */
    ppcaReplayConfig_t cfg = floatConfig(6, 4, 16);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(6, 4, 16);
    tensor_t *samples = buildFloat32TensorND(
        2, (size_t[]){3, 6}, (float[]){1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
    ppcaReplayUpdate(g, samples, ws);
    float *b = (float *)g->basis->data;
    float *lam = (float *)g->eigvals->data;
    for (size_t i = 0; i < 4 * 6; i++) {
        TEST_ASSERT_TRUE(b[i] == b[i]); /* not NaN */
    }
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lam[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, lam[3]);
    for (size_t j = 0; j < 6; j++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, b[2 * 6 + j]);
        TEST_ASSERT_EQUAL_FLOAT(0.0f, b[3 * 6 + j]);
    }
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){6}, NULL);
    rng32_t stream = {.state = 3};
    ppcaReplaySample(g, &stream, out); /* must not crash */
    freeTensor(out);
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testUpdateShrinkagePinnedFormula(void) {
    ppcaReplayConfig_t cfg0 = floatConfig(8, 2, 256);
    ppcaReplayConfig_t cfgG = floatConfig(8, 2, 256);
    cfgG.shrinkageGamma = 0.5f;
    ppcaReplay_t *g0 = ppcaReplayCreate(&cfg0);
    ppcaReplay_t *gG = ppcaReplayCreate(&cfgG);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 256);
    rng32_t rng = {.state = 1234};
    tensor_t *samples =
        drawSyntheticSamples(100, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
    ppcaReplayUpdate(g0, samples, ws);
    ppcaReplayUpdate(gG, samples, ws);
    float *l0 = (float *)g0->eigvals->data;
    float *lg = (float *)gG->eigvals->data;
    float mean0 = 0.5f * (l0[0] + l0[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f * l0[0] + 0.5f * mean0, lg[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f * l0[1] + 0.5f * mean0, lg[1]);
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(gG);
    freePpcaReplay(g0);
}

void testUpdateRejectsTooManySamples(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 4);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 4);
    tensor_t *samples = buildFloat32TensorND(2, (size_t[]){5, 8}, NULL);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayUpdate(g, samples, ws));
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testUpdateRejectsBoolSamples(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 16);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 16);
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 8;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initBoolQuantization(q);
    tensor_t *samples = initTensor(shape, q, NULL);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayUpdate(g, samples, ws));
    freeTensor(samples);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testStreamingFirstSampleOnlyMean(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(6, 2, 8);
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){6}, (float[]){1, 2, 3, 4, 5, 6});
    ppcaReplayUpdateStreaming(g, x, ws);
    TEST_ASSERT_EQUAL_UINT32(1, g->count);
    float expected[] = {1, 2, 3, 4, 5, 6};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)g->mean->data, 6);
    /* no component live yet: basis stays zero */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, ((float *)g->basis->data)[0]);
    freeTensor(x);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testStreamingConvergesToTopComponents(void) {
    ppcaReplayConfig_t cfg = floatConfig(8, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 8);
    rng32_t rng = {.state = 90210};
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){8}, NULL);
    /* seed 90210 is a slow-mixing case for amnesic-free CCIPCA: the d01
     * cross-term only crosses under its bound near ~6k samples; 8000 buys
     * ~13% margin against cross-platform libm drift (10-seed sweep +
     * calibration trace in the task-10 report; budget raised over
     * seed-swap deliberately). */
    for (size_t s = 0; s < 8000; s++) {
        tensor_t *one =
            drawSyntheticSamples(1, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
        float *src = (float *)one->data;
        float *dst = (float *)x->data;
        for (size_t j = 0; j < 8; j++) {
            dst[j] = src[j];
        }
        freeTensor(one);
        ppcaReplayUpdateStreaming(g, x, ws);
    }
    TEST_ASSERT_EQUAL_UINT32(8000, g->count);
    /* looser than path A (documented): top component well aligned, second decent */
    TEST_ASSERT_TRUE(subspaceProjection(g, &kTrueBasis8[0]) > 0.90f);
    TEST_ASSERT_TRUE(subspaceProjection(g, &kTrueBasis8[8]) > 0.80f);
    float *mu = (float *)g->mean->data;
    for (size_t j = 0; j < 8; j++) {
        TEST_ASSERT_FLOAT_WITHIN(0.3f, kTrueMu8[j], mu[j]);
    }
    /* T4-loose: near-orthonormal rows */
    float *b = (float *)g->basis->data;
    float d01 = 0.0f, n0 = 0.0f, n1 = 0.0f;
    for (size_t t = 0; t < 8; t++) {
        d01 += b[t] * b[8 + t];
        n0 += b[t] * b[t];
        n1 += b[8 + t] * b[8 + t];
    }
    TEST_ASSERT_FLOAT_WITHIN(5e-2f, 1.0f, n0);
    TEST_ASSERT_FLOAT_WITHIN(5e-2f, 1.0f, n1);
    TEST_ASSERT_FLOAT_WITHIN(5e-2f, 0.0f, d01);
    TEST_ASSERT_TRUE(g->sigma2 > 0.0f);
    freeTensor(x);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

void testStreamingRejectsBoolInput(void) {
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 8);
    ppcaReplay_t *g = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(6, 2, 8);
    size_t *dims = reserveMemory(sizeof(size_t));
    size_t *order = reserveMemory(sizeof(size_t));
    dims[0] = 6;
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initBoolQuantization(q);
    tensor_t *x = initTensor(shape, q, NULL);
    ASSERT_EXITS_WITH_FAILURE(ppcaReplayUpdateStreaming(g, x, ws));
    freeTensor(x);
    freePpcaWorkspace(ws);
    freePpcaReplay(g);
}

static ppcaReplayConfig_t packedConfig(size_t dim, size_t rank, size_t maxM, qtype_t basisType) {
    ppcaReplayConfig_t cfg = floatConfig(dim, rank, maxM);
    /* Stack/static-fixture idiom (group-quant PR1): this helper is called
     * from multiple tests in the same binary, and `static` storage means a
     * heap-allocating initSymQConfig call here would leak its previous
     * scales array on every call after the first. Build the config directly
     * instead (same values initSymQConfig(8, HALF_AWAY, ...) would produce);
     * never freed, per the stack-fixture convention. */
    static float symScale[1] = {1.f};
    static symQConfig_t symQc = {
        .scales = symScale, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 8};
    static quantization_t symQ;
    /* PR4: same idiom for ASYM's TWO arrays (initAsymQConfig would leak two
     * heap blocks per call after the first). */
    static float asymScale[1] = {1.f};
    static uint16_t asymZp[1] = {0};
    static asymQConfig_t asymQc = {.scales = asymScale,
                                   .zeroPoints = asymZp,
                                   .numGroups = 1,
                                   .groupSize = 0,
                                   .roundingMode = HALF_AWAY,
                                   .qBits = 8};
    static quantization_t asymQ;
    initSymQuantization(&symQc, &symQ);
    initAsymQuantization(&asymQc, &asymQ);
    if (basisType == SYM) {
        cfg.basisQ = &symQ;
        cfg.meanQ = &asymQ; /* mean has an offset -> ASYM */
    } else {
        cfg.basisQ = &asymQ;
        cfg.meanQ = &asymQ;
    }
    return cfg;
}

void testPackedStateSamplingDegradationBounded(void) { /* T5 */
    /* Train a FLOAT32 generator, snapshot into a packed-state twin via
     * executeConvert, sample both with the same stream, compare stats. */
    ppcaReplayConfig_t cfgF = floatConfig(8, 2, 256);
    ppcaReplay_t *gF = ppcaReplayCreate(&cfgF);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 256);
    rng32_t rng = {.state = 555};
    tensor_t *samples =
        drawSyntheticSamples(200, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
    ppcaReplayUpdate(gF, samples, ws);
    freeTensor(samples);

    ppcaReplayConfig_t cfgP = packedConfig(8, 2, 256, SYM);
    ppcaReplay_t *gP = ppcaReplayCreate(&cfgP);
    executeConvert(gF->mean, gP->mean);
    executeConvert(gF->basis, gP->basis);
    executeConvert(gF->eigvals, gP->eigvals);
    gP->sigma2 = gF->sigma2;
    gP->totalVar = gF->totalVar;
    gP->count = gF->count;

    tensor_t *out = buildFloat32TensorND(1, (size_t[]){8}, NULL);
    enum { N = 8000 };
    double sumF[8] = {0}, sumP[8] = {0}, sqF = 0, sqP = 0;
    rng32_t sf = {.state = 42}, sp = {.state = 42};
    for (size_t n = 0; n < N; n++) {
        ppcaReplaySample(gF, &sf, out);
        for (size_t j = 0; j < 8; j++) {
            float v = ((float *)out->data)[j];
            sumF[j] += v;
            sqF += (double)v * v;
        }
        ppcaReplaySample(gP, &sp, out);
        for (size_t j = 0; j < 8; j++) {
            float v = ((float *)out->data)[j];
            TEST_ASSERT_TRUE(v == v); /* no NaN from packed state */
            sumP[j] += v;
            sqP += (double)v * v;
        }
    }
    for (size_t j = 0; j < 8; j++) {
        /* int8 grid over ~[-3,3]-ish values: per-dim mean shift stays small */
        TEST_ASSERT_FLOAT_WITHIN(0.15f, (float)(sumF[j] / N), (float)(sumP[j] / N));
    }
    /* total second moment within 10% */
    TEST_ASSERT_FLOAT_WITHIN((float)(0.10 * sqF / N), (float)(sqF / N), (float)(sqP / N));
    freeTensor(out);
    freePpcaReplay(gP);
    freePpcaWorkspace(ws);
    freePpcaReplay(gF);
}

void testPackedStateStreamingGridBounded(void) { /* T7 */
    ppcaReplayConfig_t cfgF = floatConfig(8, 2, 8);
    ppcaReplayConfig_t cfgP = packedConfig(8, 2, 8, ASYM);
    ppcaReplay_t *gF = ppcaReplayCreate(&cfgF);
    ppcaReplay_t *gP = ppcaReplayCreate(&cfgP);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(8, 2, 8);
    rng32_t rng = {.state = 2222};
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){8}, NULL);
    for (size_t s = 0; s < 500; s++) {
        tensor_t *one =
            drawSyntheticSamples(1, 8, kTrueMu8, kTrueBasis8, 2, kTrueLam8, 0.04f, &rng);
        for (size_t j = 0; j < 8; j++) {
            ((float *)x->data)[j] = ((float *)one->data)[j];
        }
        freeTensor(one);
        ppcaReplayUpdateStreaming(gF, x, ws);
        ppcaReplayUpdateStreaming(gP, x, ws);
    }
    /* Packed streaming is GRID-bounded, not data-bounded (spec §5.3): it
     * must stay sane (no NaN, top component still informative), while the
     * FLOAT32 twin is the accuracy reference. subspaceProjection reads raw
     * float data, so dequant gP's packed basis into a float twin first. */
    ppcaReplay_t *gPF = ppcaReplayCreate(&cfgF);
    executeConvert(gP->basis, gPF->basis);
    gPF->count = gP->count;
    float projF = subspaceProjection(gF, &kTrueBasis8[0]);
    float projP = subspaceProjection(gPF, &kTrueBasis8[0]);
    TEST_ASSERT_TRUE(projF > 0.85f);
    TEST_ASSERT_TRUE(projP > 0.60f);
    freePpcaReplay(gPF);
    float *lamP = (float *)gP->eigvals->data; /* eigvalsQ stayed FLOAT32 in packedConfig */
    TEST_ASSERT_TRUE(lamP[0] == lamP[0]);
    freeTensor(x);
    freePpcaWorkspace(ws);
    freePpcaReplay(gP);
    freePpcaReplay(gF);
}

void testUpdateIngestsInt32Samples(void) {
    /* INT32 inputs value-cast to float via the conversion matrix — for
     * integer-valued data this is exact, so results must match a float
     * ingest of the same values. */
    ppcaReplayConfig_t cfg = floatConfig(6, 2, 16);
    ppcaReplay_t *gI = ppcaReplayCreate(&cfg);
    ppcaReplay_t *gJ = ppcaReplayCreate(&cfg);
    ppcaWorkspace_t *ws = ppcaWorkspaceCreate(6, 2, 16);

    size_t *dims = reserveMemory(2 * sizeof(size_t));
    size_t *order = reserveMemory(2 * sizeof(size_t));
    dims[0] = 4;
    dims[1] = 6;
    order[0] = 0;
    order[1] = 1;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 2;
    quantization_t *q = reserveMemory(sizeof(quantization_t));
    initInt32Quantization(q);
    tensor_t *intSamples = initTensor(shape, q, NULL);
    int32_t *iv = (int32_t *)intSamples->data;
    float fv[24];
    for (size_t i = 0; i < 24; i++) {
        iv[i] = (int32_t)(i % 7) - 3;
        fv[i] = (float)iv[i];
    }
    tensor_t *floatSamples = buildFloat32TensorND(2, (size_t[]){4, 6}, fv);

    ppcaReplayUpdate(gI, intSamples, ws);
    ppcaReplayUpdate(gJ, floatSamples, ws);

    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)gJ->mean->data, (float *)gI->mean->data, 6);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY((float *)gJ->basis->data, (float *)gI->basis->data, 12);
    freeTensor(floatSamples);
    freeTensor(intSamples);
    freePpcaWorkspace(ws);
    freePpcaReplay(gJ);
    freePpcaReplay(gI);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testCreateInitializesState);
    RUN_TEST(testWorkspaceBytesFormula);
    RUN_TEST(testWorkspaceCreateMatchesBytes);
    RUN_TEST(testReplayBytesFloat32);
    RUN_TEST(testIsoExemplarCount);
    RUN_TEST(testSetCreate);
    RUN_TEST(testCreateRejectsZeroDim);
    RUN_TEST(testCreateRejectsRankGeDim);
    RUN_TEST(testCreateRejectsNanSigma2Floor);
    RUN_TEST(testCreateRejectsNonPositiveSigma2Floor);
    RUN_TEST(testCreateRejectsSymInt32Storage);
    RUN_TEST(testCreateRejectsInt32Storage);
    RUN_TEST(testCreateRejectsBoolStorage);
    RUN_TEST(testCreateAcceptsBfpStorage);
    RUN_TEST(testCreateRejectsSymInt32Arithmetic);
    RUN_TEST(testWorkspaceCreateRejectsZeroSessionBound);
    RUN_TEST(testSampleDeterministicAndGlobalStreamUntouched);
    RUN_TEST(testSampleGoldenConstants);
    RUN_TEST(testSampleCovarianceMatchesModel);
    RUN_TEST(testSampleStreamPositionIndependentOfKeff);
    RUN_TEST(testSampleCountOneIsMeanPlusNoise);
    RUN_TEST(testSampleZeroSteadyStateAllocation);
    RUN_TEST(testSampleRejectsCountZero);
    RUN_TEST(testSampleRejectsElementCountMismatch);
    RUN_TEST(testSampleAcceptsReshapedOut);
    RUN_TEST(testMeanReplayIsExactCentroid);
    RUN_TEST(testMeanReplayRejectsCountZero);
    RUN_TEST(testMeanReplayRejectsElementCountMismatch);
    RUN_TEST(testUpdateBatchFitRecoversModel);
    RUN_TEST(testUpdateIncrementalEqualsPooled);
    RUN_TEST(testUpdateMeanShiftedIncrementalEqualsPooled);
    RUN_TEST(testUpdateRenormalizesIllConditionedSpectrum);
    RUN_TEST(testUpdateOrthonormalAcrossTenMerges);
    RUN_TEST(testUpdateRankDeficientSessionNoNaN);
    RUN_TEST(testUpdateShrinkagePinnedFormula);
    RUN_TEST(testUpdateRejectsTooManySamples);
    RUN_TEST(testUpdateRejectsBoolSamples);
    RUN_TEST(testStreamingFirstSampleOnlyMean);
    RUN_TEST(testStreamingConvergesToTopComponents);
    RUN_TEST(testStreamingRejectsBoolInput);
    RUN_TEST(testPackedStateSamplingDegradationBounded);
    RUN_TEST(testPackedStateStreamingGridBounded);
    RUN_TEST(testUpdateIngestsInt32Samples);
    return UNITY_END();
}
