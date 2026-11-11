#include <stdio.h>

#include "Deserialize.h"
#include "Div.h"
#include "DropoutApi.h"
#include "Layer.h"
#include "LayerExec.h"
#include "LayerNormApi.h"
#include "LayerQuant.h"
#include "Matmul.h"
#include "Mul.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "Serialize.h"
#include "StorageApi.h"
#include "SymGroupedWeight.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TestScale.h"
#include "unity.h"

#include <stdint.h>
#include <string.h>

/* S3 (integer dyadic scale storage) must be bit-identical to S2 (float storage
 * of canonical dyadic values). This file runs one pass over the S2 boundary
 * cases -- every fitted-grid route (float CEIL, integer-codes CEIL, ASYM range),
 * the fixed-target requant, the NEAREST product, the exact quotient, the
 * accumulate engines, the v7 float deserialize, the neutral 1.0 grids, the
 * bias-seeded and grouped (CEIL sAcc) Matmul, the Dropout coefficient and the
 * LayerNorm forward/backward grids -- and records codes, the
 * stored scale's float bits and the RNG state after each SR case. The golden
 * below was captured from the S2 build (linq/s2) BEFORE the storage type
 * changed; S3 may only change how this file reads and writes a scale (the
 * accessors directly below), never the golden.
 *
 * Build with -DODT_SCALE_IDENTITY_CAPTURE to print the record instead of
 * comparing it. */

static float si32Scale(const symInt32QConfig_t *qc) {
    return odtScaleToFloat(odtScaleLoad(&qc->scale));
}

static void setSi32Scale(symInt32QConfig_t *qc, float s) {
    odtScaleStore(&qc->scale, odtTestScale(s));
}

static float symScale(const symQConfig_t *qc, size_t g) {
    return odtScaleToFloat(odtScaleLoad(&qc->scales[g]));
}

static float asymScale(const asymQConfig_t *qc, size_t g) {
    return odtScaleToFloat(odtScaleLoad(&qc->scales[g]));
}

#define N 8
#define REC_CAP 512

static uint32_t rec[REC_CAP];
static size_t recLen;

static void put(uint32_t v) {
    TEST_ASSERT_TRUE_MESSAGE(recLen < REC_CAP, "record capacity");
    rec[recLen++] = v;
}

static void putCase(uint32_t k) {
    put(0xCA5E0000u | k);
}

static void putFloat(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    put(b);
}

static void putInt32s(const int32_t *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        put((uint32_t)v[i]);
    }
}

static void putBytes(const uint8_t *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        put(v[i]);
    }
}

static const float VA[N] = {3.3f, -1.7f, 0.21f, -2.95f, 1.05f, 0.0f, -0.333f, 2.6f};
static const float VB[N] = {0.9f, -0.45f, 1.6f, 0.07f, -1.25f, 0.6f, -0.8f, 0.33f};

static size_t dims1[1] = {N};
static size_t order1[1] = {0};
static shape_t shape1 = {.dimensions = dims1, .orderOfDimensions = order1, .numberOfDimensions = 1};

static void floatView(tensor_t *t, float *data, quantization_t *q) {
    initFloat32Quantization(q);
    setTensorValues(t, (uint8_t *)data, &shape1, q, NULL);
}

/* Captured from linq/s2 (02408f84). */
static const uint32_t S2_GOLDEN[] = {
    0xca5e0001u, 0x000007ffu, 0xfffffbe2u, 0x00000083u, 0xfffff8dau, 0x0000028bu, 0x00000000u,
    0xffffff31u, 0x0000064cu, 0x3ad34e00u, 0xcc29ed04u, 0xca5e0002u, 0x0000047fu, 0xfffffdc0u,
    0x000007ffu, 0x0000005au, 0xfffff9c1u, 0x00000300u, 0xfffffc01u, 0x000001a6u, 0x3a4ce800u,
    0xca5e0003u, 0x0000003fu, 0xffffffdfu, 0x00000004u, 0xffffffc7u, 0x00000014u, 0x00000000u,
    0xfffffffau, 0x00000032u, 0x3d568e00u, 0x6f31019cu, 0xca5e0004u, 0x00000119u, 0xffffff6fu,
    0x00000012u, 0xffffff05u, 0x00000059u, 0x00000000u, 0xffffffe3u, 0x000000ddu, 0x3c400000u,
    0xecdeb8b9u, 0xca5e0005u, 0x0000001fu, 0x0000002cu, 0x00000090u, 0x0000000du, 0x000000c0u,
    0x0000007fu, 0x00000000u, 0x00000000u, 0x3dda0400u, 0x3dabc600u, 0xca5e0006u, 0x000000c1u,
    0x00000048u, 0x000000ffu, 0x00000077u, 0x00000000u, 0x000000a5u, 0x00000028u, 0x0000008eu,
    0x3c371e00u, 0x00000070u, 0x1e6de7ceu, 0xca5e0007u, 0x0000000fu, 0x00000087u, 0x00000049u,
    0x00000080u, 0x00000067u, 0x00000000u, 0x00000000u, 0x00000000u, 0x3e614a00u, 0x31b51e38u,
    0xca5e0008u, 0x000000d2u, 0x000000fdu, 0x00000009u, 0x000000e8u, 0x00000002u, 0x0000001fu,
    0x00000000u, 0x00000000u, 0x3d532e00u, 0xca5e0009u, 0x001c663cu, 0xffe25010u, 0xfff75232u,
    0x00023b88u, 0x35a92200u, 0xca5e000au, 0x00011b41u, 0x00004a40u, 0x00001ffcu, 0xffffebf6u,
    0xffff8314u, 0x00000000u, 0x000017fau, 0x0000526cu, 0x382bbc00u, 0xca5e000bu, 0x00000001u,
    0x00000001u, 0x00000000u, 0xffffffecu, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000003u,
    0x40040000u, 0xca5e000cu, 0x000007ffu, 0xfffffbe8u, 0x00000373u, 0xfffffa85u, 0xffffff9fu,
    0x00000124u, 0xfffffdd7u, 0x00000593u, 0x3b067600u, 0x76a4acb1u, 0xca5e000du, 0x0000000fu,
    0x0000001fu, 0x000000ebu, 0x000000c5u, 0x0000005eu, 0x00000000u, 0x00000000u, 0x00000000u,
    0x3e8f5e00u, 0x65a544f4u, 0xca5e000eu, 0x000000ffu, 0x0000001bu, 0x000000a9u, 0x00000001u,
    0x00000061u, 0x0000007eu, 0x0000003fu, 0x000000d1u, 0x3ce35c00u, 0x00000068u, 0xce8ebae6u,
    0xca5e000fu, 0x000007ffu, 0xfffffbe2u, 0x00000083u, 0xfffff8dau, 0x0000028bu, 0x00000000u,
    0xffffff31u, 0x0000064cu, 0x3ad34e00u, 0xca5e0010u, 0x3f800000u, 0x00000005u, 0xfffffff9u,
    0x00000064u, 0x00000000u, 0xfffff801u, 0x00000003u, 0x00000009u, 0xffffffffu, 0x3f800000u,
    0xca5e0011u, 0x0026e4c1u, 0xffc880aeu, 0x0001d0b7u, 0xffe86c26u, 0x35a92200u, 0xce8ebae6u,
    0xca5e0012u, 0x00013091u, 0xffff6fbau, 0x00002b13u, 0xffffd829u, 0x3933f400u, 0x78a23d34u,
    0xca5e0013u, 0x000007ffu, 0xfffffbe2u, 0x00000000u, 0xfffff8dau, 0x0000028bu, 0x00000000u,
    0xffffff31u, 0x0000064cu, 0x3b16f000u, 0x0000047fu, 0xfffffdc0u, 0x00000000u, 0x0000005au,
    0xfffff9c1u, 0x00000000u, 0xfffffc01u, 0x000001a6u, 0x3a925e00u, 0xca5e0014u, 0x000007dau,
    0xfffffce7u, 0x00000116u, 0xfffffa29u, 0x000000ffu, 0xfffffc42u, 0xfffffac0u, 0x000007feu,
    0x3a467600u, 0xfffffea4u, 0xfffffd4eu, 0x000003c2u, 0x0000004cu, 0xfffff801u, 0x000007f7u,
    0xfffffd2au, 0x000002ddu, 0x39e08800u,
};

void testS3MatchesS2OnBoundaryCases(void) {
    recLen = 0;
    float va[N];
    float vb[N];
    memcpy(va, VA, sizeof va);
    memcpy(vb, VB, sizeof vb);
    quantization_t faQ, fbQ;
    tensor_t fa, fb;
    floatView(&fa, va, &faQ);
    floatView(&fb, vb, &fbQ);

    /* 1: FLOAT32 -> SYM_INT32, fitted grid CEIL15(absMax / qMax), SR. */
    rngStream_t rs = rngOpen(rngKeyFromSeed(11u));
    int32_t aCodes[N] = {0};
    symInt32QConfig_t aQC;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &aQC, 12);
    quantization_t aQ;
    initSymInt32Quantization(&aQC, &aQ);
    tensor_t ta;
    setTensorValues(&ta, (uint8_t *)aCodes, &shape1, &aQ, NULL);
    convertTensor(&fa, &ta, &rs);
    putCase(1);
    putInt32s(aCodes, N);
    putFloat(si32Scale(&aQC));
    put(rs.state);

    /* 2: FLOAT32 -> SYM_INT32, HALF_AWAY (operand b). */
    int32_t bCodes[N] = {0};
    symInt32QConfig_t bQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &bQC, 12);
    quantization_t bQ;
    initSymInt32Quantization(&bQC, &bQ);
    tensor_t tb;
    setTensorValues(&tb, (uint8_t *)bCodes, &shape1, &bQ, NULL);
    convertTensor(&fb, &tb, NULL);
    putCase(2);
    putInt32s(bCodes, N);
    putFloat(si32Scale(&bQC));

    /* 3: requant diagonal, CEIL15(s_in * max|code| / qMax) from the codes, SR. */
    int32_t rCodes[N] = {0};
    symInt32QConfig_t rQC;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &rQC, 7);
    quantization_t rQ;
    initSymInt32Quantization(&rQC, &rQ);
    tensor_t tr;
    setTensorValues(&tr, (uint8_t *)rCodes, &shape1, &rQ, NULL);
    requantSymInt32Tensor(&ta, &tr, &rs);
    putCase(3);
    putInt32s(rCodes, N);
    putFloat(si32Scale(&rQC));
    put(rs.state);

    /* 4: fixed-target requant onto a preset canonical scale 3 * 2^-8, SR. */
    int32_t fCodes[N] = {0};
    symInt32QConfig_t fQC;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &fQC, 10);
    setSi32Scale(&fQC, 0.01171875f);
    quantization_t fQ;
    initSymInt32Quantization(&fQC, &fQ);
    tensor_t tf;
    setTensorValues(&tf, (uint8_t *)fCodes, &shape1, &fQ, NULL);
    requantSymInt32TensorToScale(&ta, &tf, &rs);
    putCase(4);
    putInt32s(fCodes, N);
    putFloat(si32Scale(&fQC));
    put(rs.state);

    /* 5: FLOAT32 -> packed SYM, two groups, CEIL15 per group. */
    uint8_t gBytes[N] = {0};
    odtScale_t gScales[2] = {ODT_SCALE_ONE, ODT_SCALE_ONE};
    symQConfig_t gQC = {
        .scales = gScales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t gQ;
    initSymQuantization(&gQC, &gQ);
    tensor_t tg;
    setTensorValues(&tg, gBytes, &shape1, &gQ, NULL);
    convertTensor(&fa, &tg, NULL);
    putCase(5);
    putBytes(gBytes, N);
    putFloat(symScale(&gQC, 0));
    putFloat(symScale(&gQC, 1));

    /* 6: FLOAT32 -> ASYM, range fit (CEIL scale, then zp), SR. */
    uint8_t asBytes[N] = {0};
    odtScale_t asScales[1] = {ODT_SCALE_ONE};
    uint16_t asZps[1] = {0};
    asymQConfig_t asQC = {.scales = asScales,
                          .zeroPoints = asZps,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 8,
                          .roundingMode = SR_HALF_AWAY};
    quantization_t asQ;
    initAsymQuantization(&asQC, &asQ);
    tensor_t tas;
    setTensorValues(&tas, asBytes, &shape1, &asQ, NULL);
    convertTensor(&fb, &tas, &rs);
    putCase(6);
    putBytes(asBytes, N);
    putFloat(asymScale(&asQC, 0));
    put(asZps[0]);
    put(rs.state);

    /* 7: SYM_INT32 -> packed SYM, CEIL15 from the codes, SR. */
    uint8_t sBytes[N] = {0};
    odtScale_t sScales[1] = {ODT_SCALE_ONE};
    symQConfig_t sQC = {.scales = sScales,
                        .numGroups = 1,
                        .groupSize = 0,
                        .roundingMode = SR_HALF_AWAY,
                        .qBits = 5};
    quantization_t sQ;
    initSymQuantization(&sQC, &sQ);
    tensor_t ts;
    setTensorValues(&ts, sBytes, &shape1, &sQ, NULL);
    convertTensor(&ta, &ts, &rs);
    putCase(7);
    putBytes(sBytes, N);
    putFloat(symScale(&sQC, 0));
    put(rs.state);

    /* 8: ASYM -> packed SYM, CEIL15 from the codes. */
    uint8_t s2Bytes[N] = {0};
    odtScale_t s2Scales[1] = {ODT_SCALE_ONE};
    symQConfig_t s2QC = {
        .scales = s2Scales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t s2Q;
    initSymQuantization(&s2QC, &s2Q);
    tensor_t ts2;
    setTensorValues(&ts2, s2Bytes, &shape1, &s2Q, NULL);
    convertTensor(&tas, &ts2, NULL);
    putCase(8);
    putBytes(s2Bytes, N);
    putFloat(symScale(&s2QC, 0));

    /* 9: Matmul [2,4] x [4,2]: stored product NEAREST15(s_a * s_b). */
    size_t aDims[2] = {2, 4};
    size_t bDims[2] = {4, 2};
    size_t oDims[2] = {2, 2};
    size_t order2[2] = {0, 1};
    shape_t aShape = {.dimensions = aDims, .orderOfDimensions = order2, .numberOfDimensions = 2};
    shape_t bShape = {.dimensions = bDims, .orderOfDimensions = order2, .numberOfDimensions = 2};
    shape_t oShape = {.dimensions = oDims, .orderOfDimensions = order2, .numberOfDimensions = 2};
    tensor_t ma, mb, mo;
    setTensorValues(&ma, (uint8_t *)aCodes, &aShape, &aQ, NULL);
    setTensorValues(&mb, (uint8_t *)bCodes, &bShape, &bQ, NULL);
    int32_t moCodes[4] = {0};
    symInt32QConfig_t moQC;
    initSymInt32QConfig(HALF_AWAY, &moQC);
    quantization_t moQ;
    initSymInt32Quantization(&moQC, &moQ);
    setTensorValues(&mo, (uint8_t *)moCodes, &oShape, &moQ, NULL);
    matmulSymInt32Tensors(&ma, &mb, &mo);
    putCase(9);
    putInt32s(moCodes, 4);
    putFloat(si32Scale(&moQC));

    /* 10: elementwise Mul, stored product NEAREST15(s_a * s_b). */
    int32_t muCodes[N] = {0};
    symInt32QConfig_t muQC;
    initSymInt32QConfig(HALF_AWAY, &muQC);
    quantization_t muQ;
    initSymInt32Quantization(&muQC, &muQ);
    tensor_t tmu;
    setTensorValues(&tmu, (uint8_t *)muCodes, &shape1, &muQ, NULL);
    mulSymInt32Tensors(&tr, &tb, &tmu);
    putCase(10);
    putInt32s(muCodes, N);
    putFloat(si32Scale(&muQC));

    /* 11: elementwise Div, exact quotient NEAREST15(s_a / s_b). */
    int32_t dvCodes[N] = {0};
    symInt32QConfig_t dvQC;
    initSymInt32QConfig(HALF_AWAY, &dvQC);
    quantization_t dvQ;
    initSymInt32Quantization(&dvQC, &dvQ);
    tensor_t tdv;
    setTensorValues(&tdv, (uint8_t *)dvCodes, &shape1, &dvQ, NULL);
    divSymInt32Tensors(&ta, &tb, &tdv);
    putCase(11);
    putInt32s(dvCodes, N);
    putFloat(si32Scale(&dvQC));

    /* 12: SYM_INT32 + SYM_INT32 accumulate (fresh absmax), SR. */
    int32_t acCodes[N];
    memcpy(acCodes, bCodes, sizeof acCodes);
    symInt32QConfig_t acQC;
    initSymInt32QConfigWithQMaxBits(SR_HALF_AWAY, &acQC, 12);
    setSi32Scale(&acQC, si32Scale(&bQC));
    quantization_t acQ;
    initSymInt32Quantization(&acQC, &acQ);
    tensor_t tac;
    setTensorValues(&tac, (uint8_t *)acCodes, &shape1, &acQ, NULL);
    accumulateSymInt32IntoSymInt32Rescale(&tac, &ta, &rs);
    putCase(12);
    putInt32s(acCodes, N);
    putFloat(si32Scale(&acQC));
    put(rs.state);

    /* 13: float increment into a packed SYM target, rescale engine, SR. */
    accumulateFloatIntoSymTensorRescale(&ts, vb, N, &rs);
    putCase(13);
    putBytes(sBytes, N);
    putFloat(symScale(&sQC, 0));
    put(rs.state);

    /* 14: float increment into an ASYM target, rescale engine, SR. */
    accumulateFloatIntoAsymTensorRescale(&tas, va, N, &rs);
    putCase(14);
    putBytes(asBytes, N);
    putFloat(asymScale(&asQC, 0));
    put(asZps[0]);
    put(rs.state);

    /* 15: v7 float serialize -> deserialize, NEAREST15 (identity on canonical). */
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    serializeTensor(&ta, f);
    rewind(f);
    size_t *dd = reserveMemory(sizeof(size_t));
    dd[0] = N;
    size_t *od = reserveMemory(sizeof(size_t));
    od[0] = 0;
    shape_t *dShape = reserveMemory(sizeof(shape_t));
    setShape(dShape, dd, 1, od);
    tensor_t *td = initTensor(dShape, quantizationInitSymInt32(HALF_AWAY), NULL);
    deserializeTensor(td, f);
    fclose(f);
    putCase(15);
    putInt32s((const int32_t *)td->data, N);
    putFloat(si32Scale(td->quantization->qConfig));
    freeTensor(td);

    /* 16: raw-int and neutral grids, exactly 1.0: all-zero FLOAT32 -> SYM_INT32
     * and INT32 -> SYM_INT32. */
    float zeros[N] = {0};
    quantization_t fzQ;
    tensor_t fz;
    floatView(&fz, zeros, &fzQ);
    int32_t zCodes[N] = {0};
    symInt32QConfig_t zQC;
    initSymInt32QConfig(HALF_AWAY, &zQC);
    setSi32Scale(&zQC, 0.01171875f);
    quantization_t zQ;
    initSymInt32Quantization(&zQC, &zQ);
    tensor_t tz;
    setTensorValues(&tz, (uint8_t *)zCodes, &shape1, &zQ, NULL);
    convertTensor(&fz, &tz, NULL);
    putCase(16);
    putFloat(si32Scale(&zQC));
    int32_t ints[N] = {5, -7, 100, 0, -2047, 3, 9, -1};
    quantization_t iQ;
    initInt32Quantization(&iQ);
    tensor_t ti;
    setTensorValues(&ti, (uint8_t *)ints, &shape1, &iQ, NULL);
    setSi32Scale(&zQC, 0.01171875f);
    convertTensor(&ti, &tz, NULL);
    putInt32s(zCodes, N);
    putFloat(si32Scale(&zQC));

    /* 17: Matmul with a bias seed on the stored product scale, SR. */
    size_t biasDims[1] = {2};
    shape_t biasShape = {
        .dimensions = biasDims, .orderOfDimensions = order1, .numberOfDimensions = 1};
    int32_t biasCodes[2] = {37, -91};
    symInt32QConfig_t biasQC;
    initSymInt32QConfig(HALF_AWAY, &biasQC);
    setSi32Scale(&biasQC, 24561.0f / 1048576.0f);
    quantization_t biasQ;
    initSymInt32Quantization(&biasQC, &biasQ);
    tensor_t tbias;
    setTensorValues(&tbias, (uint8_t *)biasCodes, &biasShape, &biasQ, NULL);
    int32_t mbCodes[4] = {0};
    symInt32QConfig_t mbQC;
    initSymInt32QConfig(SR_HALF_AWAY, &mbQC);
    quantization_t mbQ;
    initSymInt32Quantization(&mbQC, &mbQ);
    tensor_t mbo;
    setTensorValues(&mbo, (uint8_t *)mbCodes, &oShape, &mbQ, NULL);
    matmulSymInt32TensorsWithBias(&ma, &mb, &mbo, &tbias, &rs);
    putCase(17);
    putInt32s(mbCodes, 4);
    putFloat(si32Scale(&mbQC));
    put(rs.state);

    /* 18: grouped-weight Matmul: sAcc = CEIL15(s_a * max_g s_g), SR. */
    int32_t gwCodes[N] = {31, -20, 5, -32, 17, 0, -9, 12};
    symInt32QConfig_t gwQC;
    initSymInt32QConfig(SR_HALF_AWAY, &gwQC);
    quantization_t gwQ;
    initSymInt32Quantization(&gwQC, &gwQ);
    tensor_t gw;
    setTensorValues(&gw, (uint8_t *)gwCodes, &bShape, &gwQ, NULL);
    const symGroupedWeight_t gwCtx = {.groups = &gQC, .domain = SYM_CODES_TWOS_COMPLEMENT};
    int32_t goCodes[4] = {0};
    symInt32QConfig_t goQC;
    initSymInt32QConfig(SR_HALF_AWAY, &goQC);
    quantization_t goQ;
    initSymInt32Quantization(&goQC, &goQ);
    tensor_t go;
    setTensorValues(&go, (uint8_t *)goCodes, &oShape, &goQ, NULL);
    matmulSymInt32TensorsGroupedWeight(&ma, &gw, &tbias, &go, &gwCtx, &rs);
    putCase(18);
    putInt32s(goCodes, 4);
    putFloat(si32Scale(&goQC));
    put(rs.state);

    /* 19: Dropout coefficient NEAREST15(s * NEAREST15(1 / (1 - p))), forward
     * and backward. */
    size_t *mdims = reserveMemory(sizeof(size_t));
    mdims[0] = N;
    size_t *morder = reserveMemory(sizeof(size_t));
    morder[0] = 0;
    shape_t *mshape = reserveMemory(sizeof(shape_t));
    setShape(mshape, mdims, 1, morder);
    tensor_t *mask = initTensor(mshape, quantizationInitBool(), NULL);
    quantization_t *dfq = quantizationInitSymInt32(HALF_AWAY);
    quantization_t *dbq = quantizationInitSymInt32(HALF_AWAY);
    layer_t *drop = dropoutLayerInit(0.3f, mask, dfq, dbq);
    drop->config->dropout->training = true;
    int32_t doCodes[N] = {0};
    symInt32QConfig_t doQC;
    initSymInt32QConfig(HALF_AWAY, &doQC);
    quantization_t doQ;
    initSymInt32Quantization(&doQC, &doQ);
    tensor_t tdo;
    setTensorValues(&tdo, (uint8_t *)doCodes, &shape1, &doQ, NULL);
    const layerExec_t dexec = {.key = rngKeyFromSeed(5u)};
    layerFunctions[DROPOUT].forward(drop, &ta, &tdo, &dexec);
    putCase(19);
    putInt32s(doCodes, N);
    putFloat(si32Scale(&doQC));
    int32_t dpCodes[N] = {0};
    symInt32QConfig_t dpQC;
    initSymInt32QConfig(HALF_AWAY, &dpQC);
    quantization_t dpQ;
    initSymInt32Quantization(&dpQC, &dpQ);
    tensor_t tdp;
    setTensorValues(&tdp, (uint8_t *)dpCodes, &shape1, &dpQ, NULL);
    layerFunctions[DROPOUT].backward(drop, &ta, &tb, &tdp, &dexec);
    putInt32s(dpCodes, N);
    putFloat(si32Scale(&dpQC));
    freeDropoutLayer(drop);
    freeQuantization(dbq);
    freeQuantization(dfq);
    freeTensor(mask);

    /* 20: LayerNorm forward (sNorm CEIL15, sY = NEAREST15(sNorm * s_gamma))
     * and backward (dxScale CEIL15), SYM_INT32 SR wires. */
    size_t normShape[1] = {4};
    layerNormInit_t lnInit = {.normalizedShape = normShape, .numNormDims = 1, .eps = 1e-5f};
    quantization_t *lnQ = quantizationInitSymInt32(SR_HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, lnQ);
    quantization_t *lnParamQ = quantizationInitSymInt32(HALF_AWAY);
    lq.weightStorage = lnParamQ;
    lq.biasStorage = lnParamQ;
    layer_t *ln = layerNormLayerInit(&lnInit, &lq);
    int32_t lnCodes[N] = {0};
    symInt32QConfig_t lnQC;
    initSymInt32QConfig(SR_HALF_AWAY, &lnQC);
    quantization_t lnOutQ;
    initSymInt32Quantization(&lnQC, &lnOutQ);
    tensor_t lnOut;
    setTensorValues(&lnOut, (uint8_t *)lnCodes, &aShape, &lnOutQ, NULL);
    const layerExec_t lexec = {.key = rngKeyFromSeed(9u)};
    layerFunctions[LAYERNORM].forward(ln, &ma, &lnOut, &lexec);
    putCase(20);
    putInt32s(lnCodes, N);
    putFloat(si32Scale(&lnQC));
    tensor_t lnLoss;
    setTensorValues(&lnLoss, (uint8_t *)bCodes, &aShape, &bQ, NULL);
    int32_t lnDx[N] = {0};
    symInt32QConfig_t lnDxQC;
    initSymInt32QConfig(SR_HALF_AWAY, &lnDxQC);
    quantization_t lnDxQ;
    initSymInt32Quantization(&lnDxQC, &lnDxQ);
    tensor_t lnDxT;
    setTensorValues(&lnDxT, (uint8_t *)lnDx, &aShape, &lnDxQ, NULL);
    layerFunctions[LAYERNORM].backward(ln, &ma, &lnLoss, &lnDxT, &lexec);
    putInt32s(lnDx, N);
    putFloat(si32Scale(&lnDxQC));
    freeLayerNormLayer(ln);
    freeQuantization(lnParamQ);
    freeQuantization(lnQ);

#ifdef ODT_SCALE_IDENTITY_CAPTURE
    printf("static const uint32_t S2_GOLDEN[] = {\n");
    for (size_t i = 0; i < recLen; i++) {
        printf("0x%08xu,%s", rec[i], (i % 6 == 5) ? "\n" : " ");
    }
    printf("};\n");
#else
    TEST_ASSERT_EQUAL_size_t(sizeof S2_GOLDEN / sizeof S2_GOLDEN[0], recLen);
    TEST_ASSERT_EQUAL_HEX32_ARRAY(S2_GOLDEN, rec, recLen);
#endif
}

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testS3MatchesS2OnBoundaryCases);
    return UNITY_END();
}
