#include "DTypes.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "RNG.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "expected_asym_nudged.h"
#include "expected_group_quant.h"
#include "expected_requant.h"
#include "unity.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* File-internal helper of TensorConversion.c (external linkage, no header entry);
 * declared locally to characterize its packed-size memset. */
void zeroTensorData(tensor_t *tensor);

/* PR-C pack tests verify the packed SYM output by unpacking it here in-test:
 * byteConversion zero-fills on widen, so sign-extend each value from qBits.
 * (The SYM->* unpack cells live on the parallel PR-B branch, not here.) */
static void symTestUnpackSignExtend(const uint8_t *packed, size_t qBits, int32_t *out, size_t n) {
    byteConversion((uint8_t *)packed, qBits, (uint8_t *)out, 32, n);
    const int32_t signBit = (int32_t)1 << (qBits - 1);
    const int32_t mask = (int32_t)(((uint32_t)1 << qBits) - 1u);
    for (size_t i = 0; i < n; i++) {
        int32_t v = out[i] & mask;
        out[i] = (v ^ signBit) - signBit;
    }
}

void testZeroTensorDataSymSubByteZeroesOnlyPackedBytes() {
    /* SYM qBits=3, N=10 -> packed 4 bytes; the guard byte behind them must survive.
     * Mutation guard: the pre-fix N * calcBytesPerElement memset (10 bytes) clobbers
     * the canary -> RED (and stack-buffer-overflow under ASan). */
    uint8_t data[5];
    memset(data, 0xFF, sizeof(data));
    size_t dims[] = {1, 10};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    float cfgScale[1] = {1.0f};
    symQConfig_t cfg = {
        .scales = cfgScale, .numGroups = 1, .groupSize = 0, .qBits = 3, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&cfg, &q);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    zeroTensorData(&t);

    for (size_t i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_UINT8(0, data[i]);
    }
    TEST_ASSERT_EQUAL_UINT8(0xFF, data[4]);
}

void testConversionIntFloat() {
    uint8_t numValues = 6;

    size_t dims[] = {6};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {1, 2, 3, 4, -1, -2};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&intTensor, &floatTensor);
    float actual[numValues];
    readBytesAsFloatArray(numValues, (uint8_t *)floatData, actual);

    float expected[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, actual, numValues);
}

void testConversionIntSymInt32() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {1, 2, 3, 4, -1, -2};

    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[numValues];

    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    convertTensor(&intTensor, &symInt32Tensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(intTensor.data, symInt32Tensor.data, numValues);
}

void testConversionIntAsym() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};
    int32_t intData[] = {1, 2, 3, 4, -1, -2};

    quantization_t intQ;
    initInt32Quantization(&intQ);

    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    /* Stack-fixture idiom (group-quant PR4): local backing arrays, never
     * freed -- the converter derives the grid into them. */
    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);
    uint8_t asymData[numValues * calcBytesPerElement(&asymQ)];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);
    convertTensor(&intTensor, &asymTensor);

    uint8_t flattenedAsymData[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, flattenedAsymData, 8, numValues);

    /* Re-pinned to the nudged code-domain grid (group-quant PR4, D6): zp is
     * now the CODE-domain +10 (old value-domain zp was -10); the codes
     * happen to coincide with the old grid for this zero-spanning band
     * (verified by the goldgen, not assumed). Gold: expected_asym_nudged.h
     * f1SpanZero (same {1,2,3,4,-1,-2} values as this int fixture). */
    TEST_ASSERT_EQUAL_size_t(numValues, codes_asym_nudged_f1SpanZero_len);
    for (size_t i = 0; i < numValues; i++) {
        TEST_ASSERT_EQUAL_UINT8((uint8_t)codes_asym_nudged_f1SpanZero[i], flattenedAsymData[i]);
    }
    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f1SpanZero, asymQConfig.zeroPoints[0]);
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f1SpanZero, asymQConfig.scales[0]);
}

void testConversionFloatInt() {
    uint8_t numValues = 6;

    float floatData[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    quantization_t intQ;
    initInt32Quantization(&intQ);
    int32_t intData[numValues];
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);
    convertTensor(&floatTensor, &intTensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(6, (uint8_t *)intData, actual);

    int32_t expected[] = {1, 2, 3, 4, -1, -2};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, actual, numValues);
}

void testConversionFloatSymInt32() {
    uint8_t numValues = 6;

    float floatData[] = {1.5f, 2.9f, 3.2f, 4.5f, -1.2f, -6.7f};
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[numValues];
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);
    convertTensor(&floatTensor, &symInt32Tensor);

    /* absmax = 6.7; int12 scale = 6.7/2047 ≈ 0.003273083.
     * Quantized values: round(v / scale): 458, 886, 978, 1375, -367, -2047. */
    float expectedScale = 6.7f / 2047.0f;
    int32_t expectedData[] = {458, 886, 978, 1375, -367, -2047};

    symInt32QConfig_t *outputSymInt32QC = symInt32Tensor.quantization->qConfig;
    TEST_ASSERT_FLOAT_WITHIN(0.000001f, expectedScale, outputSymInt32QC->scale);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, symInt32Tensor.data, numValues);

    convertTensor(&symInt32Tensor, &floatTensor);
    float expectedFloat[] = {1.5f, 2.9f, 3.2f, 4.5f, -1.2f, -6.7f};
    float *actualFloat = (float *)floatTensor.data;
    /* int12 quantisation step = scale ≈ 0.00327; worst-case round-trip
     * error is scale/2 ≈ 0.00164, so tolerance 0.002 is sufficient. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.002f, expectedFloat[i], actualFloat[i]);
    }
}

void testConversionFloatAsym() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)input_asym_nudged_f1SpanZero, &shape, &floatQ, NULL);

    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[numValues * calcBytesPerElement(&asymQ)];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    uint8_t flattenedAsymData[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, flattenedAsymData, 8, numValues);

    /* Re-pinned to the nudged code-domain grid (group-quant PR4, D6): the
     * band {1..4, -1, -2} spans zero so the nudge is a no-op and the scale
     * stays 6/31; zp becomes the code-domain +10 (old value-domain -10) and
     * the codes coincide with the old grid here (goldgen-verified). */
    TEST_ASSERT_EQUAL_size_t(numValues, input_asym_nudged_f1SpanZero_len);
    for (size_t i = 0; i < numValues; i++) {
        TEST_ASSERT_EQUAL_UINT8((uint8_t)codes_asym_nudged_f1SpanZero[i], flattenedAsymData[i]);
    }
    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f1SpanZero, asymQConfig.zeroPoints[0]);
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f1SpanZero, asymQConfig.scales[0]);
}

void testConversionFloatAsymQBits16NegativeBandZeroPointAtCodeCeiling() {
    /* Re-pinned to the nudged code-domain grid (group-quant PR4, D6). This
     * used to be the -72817 int32-zeroPoint width pin (#246); the nudge
     * extends the all-negative band [-10, -1] to [-10, 0], so the code-domain
     * zp lands EXACTLY on the uint16 ceiling 2^16-1 = 65535 -- the D6
     * boundary this fixture now pins (zpReal = -mn/scale = qMax by
     * construction whenever mx nudges to 0). The old max value -1 no longer
     * maps to code 65535 (that code now represents 0.0); the min still maps
     * to code 0. Gold: expected_asym_nudged.h f2NegBand16.
     * Mutation guard: dropping the nudge re-derives the old [-10,-1] band
     * whose zpReal -72817 is outside the uint16 clamp -> zp pins RED. */
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)input_asym_nudged_f2NegBand16, &shape, &floatQ, NULL);

    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 16,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);
    uint8_t asymData[numValues * 2];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f2NegBand16, asymQConfig.zeroPoints[0]);
    TEST_ASSERT_EQUAL_UINT16(65535, asymQConfig.zeroPoints[0]); /* the uint16 ceiling itself */
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f2NegBand16, asymQConfig.scales[0]);

    int32_t codes[6];
    byteConversion(asymTensor.data, 16, (uint8_t *)codes, 32, numValues);
    TEST_ASSERT_EQUAL_INT32_ARRAY(codes_asym_nudged_f2NegBand16, codes, numValues);

    float decoded[6];
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&asymTensor, &floatOut);

    float tol = asymQConfig.scales[0] * 0.5f + 1e-4f;
    for (size_t i = 0; i < numValues; i++) {
        TEST_ASSERT_FLOAT_WITHIN(tol, input_asym_nudged_f2NegBand16[i], decoded[i]);
    }
}

void testInitAsymQConfigRejectsQBitsAbove16(void) {
    /* D6 ceiling (was [1, 30], #246): the code-domain zeroPoint is uint16,
     * so qBits=17 has codes/zp up to 2^17-1 with no uint16 representation.
     * Mutation guard: removing the initAsymQConfig guard lets the child exit
     * 0 -> RED. */
    asymQConfig_t qc;
    ASSERT_EXITS_WITH_FAILURE(initAsymQConfig(17, HALF_AWAY, &qc));
}

void testConversionFloatAsymFarNegativeBandDerivesNudgedGrid(void) {
    /* Replaces testConversionFloatAsymZeroPointBeyondInt32Dies (group-quant
     * PR4, D6): under the zero-inclusion nudge this data's zpReal is BOUNDED
     * into [0, 2^b-1] by construction -- the old int32-overflow death regime
     * (un-nudged zpReal ~ -2.55e9) is unreachable, so the death test is
     * obsolete. The property that replaces it: the far-negative band
     * [-5e6, -4999999.5] nudges to [-5e6, 0], derives zp = 255 (the qBits=8
     * code ceiling) and a VALID grid whose code zp decodes to EXACTLY 0.0f
     * (zero-representability, the nudge's raison d'etre). Gold:
     * expected_asym_nudged.h f3NegFar.
     * Mutation guard (i): dropping the nudge derives the old [-5e6,
     * -4999999.5] band -> scale collapses to 0.5/255 (vs 19607.84) and the
     * codes change -> the scale pin and code pins go RED. (The exact-zero
     * decode alone cannot catch (i): code==zp decodes to 0 on ANY affine
     * grid -- what the nudge buys is that zp is a VALID grid anchor, which
     * the scale/code pins witness.) */
    size_t numValues = 2;
    size_t dims[] = {numValues};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)input_asym_nudged_f3NegFar, &shape, &floatQ, NULL);

    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 8,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);
    uint8_t asymData[2];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f3NegFar, asymQConfig.zeroPoints[0]);
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f3NegFar, asymQConfig.scales[0]);
    uint8_t codes[2];
    byteConversion(asymTensor.data, 8, codes, 8, numValues);
    for (size_t i = 0; i < numValues; i++) {
        TEST_ASSERT_EQUAL_UINT8((uint8_t)codes_asym_nudged_f3NegFar[i], codes[i]);
    }
    /* zero-representability: the code equal to zp must decode to EXACTLY
     * 0.0f THROUGH THE REAL DECODE FUNNEL (not a hand-inlined formula, which
     * would be tautological): pack code == zp into the tensor and convert.
     * Mutation guard: a (code + zp)*scale decode regression yields
     * 2*zp*scale = 1e7, not 0. */
    int32_t zpCodes[2] = {(int32_t)asymQConfig.zeroPoints[0], (int32_t)asymQConfig.zeroPoints[0]};
    byteConversion((uint8_t *)zpCodes, 32, asymTensor.data, 8, numValues);
    float decoded[2];
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&asymTensor, &floatOut);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, decoded[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, decoded[1]);
}

void testConversionFloatAsymEncodeClampAtBandEdgeTie(void) {
    /* Encode-clamp pin (group-quant PR4, mutation guard (ii)): with scale
     * exactly 1.0 and zpReal = 1.5 (an exact HALF_AWAY tie -> zp = 2), the
     * band-max 5.5 ALSO rounds up (round(5.5) = 6), so the un-clamped code
     * 6 + 2 = 8 exceeds qMax = 2^3-1 = 7 -- the ONE fixture whose emitted
     * code depends on emitAsymChunk's clamp. Removing the clamp packs 8 into
     * 3 bits (wraps to 0) -> the code pin goes RED. Also pins
     * zero-representability on a 0.0-CONTAINING buffer: 0.0 encodes to code
     * == zp and decodes to exactly 0.0f. Gold: expected_asym_nudged.h
     * f4ClampTie. */
    size_t numValues = 4;
    size_t dims[] = {numValues};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)input_asym_nudged_f4ClampTie, &shape, &floatQ, NULL);

    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 3,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);
    uint8_t asymData[calcNumberOfBytesForData(&asymQ, numValues)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    int32_t codes[4];
    byteConversion(asymTensor.data, 3, (uint8_t *)codes, 32, numValues);

    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f4ClampTie, asymQConfig.zeroPoints[0]);
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f4ClampTie, asymQConfig.scales[0]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(codes_asym_nudged_f4ClampTie, codes, numValues);

    /* the 0.0 element (index 2) decodes back to EXACTLY 0.0f */
    float decoded[4];
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&asymTensor, &floatOut);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, decoded[2]);
}

void testConversionSymInt32Int() {
    size_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    int32_t intData[numValues];
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    convertTensor(&symInt32Tensor, &intTensor);

    int32_t expected[] = {1, 2, 3, 4, -1, -2};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, intTensor.data, numValues);
}

void testConversionSymInt32Float() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    symInt32QConfig.scale = 1.f;
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&symInt32Tensor, &floatTensor);

    float expected[] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};

    /*float actual[numValues];
    readBytesAsFloatArray(numValues, floatTensor.data, actual);
    for(size_t i = 0; i < numValues; i++) {
        printf("%f\n", actual[i]);
    }*/

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, floatTensor.data, numValues);
}

void testConversionSymInt32Asym() {
    uint8_t numValues = 6;

    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    symInt32QConfig.scale = 1.f;
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);

    int32_t symInt32Data[] = {1, 2, 3, 4, -1, -2};
    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    float asymScales[1] = {1.f};
    uint16_t asymZps[1] = {0};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    size_t outputBitsPerElement = calcBitsPerElement(&asymQ);
    size_t outputTotalNumberOfBits = outputBitsPerElement * numValues;
    size_t numberOfRequiredBytes = ceil((double)outputTotalNumberOfBits / (double)8);
    uint8_t asymData[numberOfRequiredBytes];

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    convertTensor(&symInt32Tensor, &asymTensor);

    uint32_t output[numValues];
    byteConversion(asymTensor.data, asymQConfig.qBits, (uint8_t *)output, 32, numValues);

    /* Re-pinned to the nudged code-domain grid (group-quant PR4, D6): the
     * scale=1 mantissas dequantize to the same {1..4,-1,-2} values as
     * f1SpanZero -- zp +10 (code domain), codes goldgen-verified. */
    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f1SpanZero, asymQConfig.scales[0]);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f1SpanZero, asymQConfig.zeroPoints[0]);
    for (size_t i = 0; i < numValues; i++) {
        TEST_ASSERT_EQUAL_UINT32((uint32_t)codes_asym_nudged_f1SpanZero[i], output[i]);
    }
}

void testConversionAsymInt() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    /* Code-domain re-pin (group-quant PR4, D6): the old value-domain zp -11
     * becomes code-domain +11; the mantissa image code - zp reproduces the
     * exact same integers the old code + (-11) did. */
    float asymScales[1] = {0.1875f};
    uint16_t asymZps[1] = {11};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};

    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t intQ;
    initInt32Quantization(&intQ);
    int32_t intData[numValues];
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    convertTensor(&asymTensor, &intTensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(numValues, intTensor.data, actual);
    int32_t expectedData[] = {5, 11, 16, 20, -5, -11};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, actual, numValues);
}

void testConversionAsymFloat() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    /* Code-domain re-pin (group-quant PR4, D6): zp -11 -> +11, decode
     * (code - 11)*scale reproduces the old (code + (-11))*scale exactly. */
    float asymScales[1] = {0.1875f};
    uint16_t asymZps[1] = {11};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[numValues];

    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&asymTensor, &floatTensor);

    float actual[numValues];
    readBytesAsFloatArray(numValues, floatTensor.data, actual);
    float expectedData[] = {0.9375f, 2.0625f, 3.f, 3.75f, -0.9375f, -2.0625f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expectedData, actual, numValues);
}

void testConversionAsymSymInt32() {
    size_t numValues = 6;
    size_t dims[] = {numValues};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    /* Code-domain re-pin (group-quant PR4, D6): zp -11 -> +11; the ASYM ->
     * SYM_INT32 mantissa image is now code - zp (was code + zeroPoint),
     * producing the same integers. */
    float asymScales[1] = {0.1875f};
    uint16_t asymZps[1] = {11};
    asymQConfig_t asymQConfig = {.scales = asymScales,
                                 .zeroPoints = asymZps,
                                 .numGroups = 1,
                                 .groupSize = 0,
                                 .qBits = 5,
                                 .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQConfig, &asymQ);

    uint8_t asymData[] = {0b11010000, 0b11101110, 0b01101111, 0b00000000};

    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    symInt32QConfig_t symInt32QConfig;
    initSymInt32QConfig(HALF_AWAY, &symInt32QConfig);
    quantization_t symInt32Q;
    initSymInt32Quantization(&symInt32QConfig, &symInt32Q);
    int32_t symInt32Data[numValues];

    tensor_t symInt32Tensor;
    setTensorValues(&symInt32Tensor, (uint8_t *)symInt32Data, &shape, &symInt32Q, NULL);

    convertTensor(&asymTensor, &symInt32Tensor);

    int32_t actual[numValues];
    readBytesAsInt32Array(numValues, symInt32Tensor.data, actual);

    int32_t expectedData[] = {5, 11, 16, 20, -5, -11};

    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedData, actual, numValues);
}

void testConversionBoolBoolCopiesOnlyPackedBytes() {
    /* N=9 BOOL elements occupy (9+7)/8 = 2 packed bytes; the same-type copy
     * must move exactly 2 bytes. Canary: the output payload sits at the start
     * of a 16-byte guard allocation whose bytes 2..15 hold sentinel 0xAA.
     * Before the fix, convertTensor memmoves N * calcBytesPerElement(BOOL)
     * = 9 bytes and clobbers the sentinels with the input buffer's 0x55
     * filler. Both buffers are oversized on purpose so the buggy 9-byte
     * memmove stays inside owned allocations and the RED run is
     * well-defined. initTensor is not used here because it allocates the
     * exact packed size (2 bytes), which would make the buggy copy run out
     * of bounds. */
    enum { N = 9, GUARD_BYTES = 16, PAYLOAD_BYTES = 2 };

    size_t dims[] = {N};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    quantization_t inQ;
    initBoolQuantization(&inQ);
    uint8_t *inBuffer = reserveMemory(GUARD_BYTES);
    memset(inBuffer, 0x55, GUARD_BYTES);
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuffer, &shape, &inQ, NULL);

    const bool pattern[N] = {true, false, true, true, false, false, true, false, true};
    tensorFillFromBoolBuffer(&inTensor, pattern, N);

    quantization_t outQ;
    initBoolQuantization(&outQ);
    uint8_t *outBuffer = reserveMemory(GUARD_BYTES);
    memset(outBuffer, 0xAA, GUARD_BYTES);
    tensor_t outTensor;
    setTensorValues(&outTensor, outBuffer, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    /* Under-copy guard: all 9 bits must arrive. */
    for (size_t i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL(pattern[i], tensorBoolGet(&outTensor, i));
    }
    /* Over-copy guard: every byte after the packed payload is untouched. */
    for (size_t i = PAYLOAD_BYTES; i < GUARD_BYTES; i++) {
        TEST_ASSERT_EQUAL_UINT8(0xAA, outBuffer[i]);
    }

    freeReservedMemory(inBuffer);
    freeReservedMemory(outBuffer);
}

void testQuantTypeToStringBool() {
    TEST_ASSERT_EQUAL_STRING("BOOL", quantTypeToString(BOOL));
}

void testConversionSymInt32SameTypeCopyPropagatesScale() {
    /* Pins the same-type copy semantics PR C builds on: mantissas memmoved,
     * input scale overwrites any pre-set output scale, NO rescale happens.
     * symInt32QConfig_t is zero-initialized instead of using
     * initSymInt32QConfig so this test references no roundingMode_t
     * enumerator (PR A renames them in parallel); the same-type copy path
     * never reads roundingMode. */
    size_t numValues = 4;
    size_t dims[] = {4};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC = {0};
    inQC.scale = 0.03125f;
    inQC.qMaxBits = 16;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {100, -200, 300, -400};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQC = {0};
    outQC.scale = 999.0f; /* pre-set garbage; the copy must overwrite it */
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[4] = {0, 0, 0, 0};
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(inData, outData, numValues);
    TEST_ASSERT_EQUAL_FLOAT(0.03125f, outQC.scale);
}

void testConversionSymSameTypeCopyPropagatesScale() {
    /* SYM -> SYM same width is a verbatim packed copy; the output's stale default
     * scale (1.f) must be replaced by the input's, or every later dequant is wrong.
     * Mutation guard: dropping the new `case SYM:` leaves scale == 1.f -> RED. */
    int32_t mantissas[] = {3, -3, 31, -32};
    size_t dims[] = {1, 4};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};

    float inQCScale[1] = {0.25f};
    symQConfig_t inQC = {
        .scales = inQCScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inData[3];
    byteConversion((uint8_t *)mantissas, 32, inData, 6, 4);
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 6,
                          .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[4];
    memset(outData, 0xAA,
           sizeof(outData)); /* canary in byte 3: only 3 packed bytes may be written */
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    convertTensor(&in, &out);

    TEST_ASSERT_EQUAL_FLOAT(0.25f, outQC.scales[0]);
    TEST_ASSERT_EQUAL_UINT8(6, outQC.qBits);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(inData, outData, 3);
    TEST_ASSERT_EQUAL_UINT8(0xAA, outData[3]);
}

void testConversionSymSameTypeWidthMismatchDies() {
    /* qBits 6 -> 4: a verbatim byte copy would reinterpret the packing (and for
     * wider inputs overflow the output buffer). Width-changing SYM rewrites are
     * real conversions (repack policy: PR3). Mutation guard: removing the width
     * guard lets the child exit 0 -> RED. */
    int32_t mantissas[] = {3, -3};
    size_t dims[] = {1, 2};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};

    float inQCScale[1] = {0.25f};
    symQConfig_t inQC = {
        .scales = inQCScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inData[2];
    byteConversion((uint8_t *)mantissas, 32, inData, 6, 2);
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 4,
                          .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[1] = {0};
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&in, &out));
}

void testConversionAsymSameTypeWidthMismatchDies() {
    /* ASYM variant of the width guard (qBits 5 -> 3). */
    int32_t codes[] = {0, 10};
    size_t dims[] = {1, 2};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};

    float inScales[1] = {0.5f};
    uint16_t inZps[1] = {7};
    asymQConfig_t inQC = {.scales = inScales,
                          .zeroPoints = inZps,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[2];
    byteConversion((uint8_t *)codes, 32, inData, 5, 2);
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outScales[1] = {1.f};
    uint16_t outZps[1] = {0};
    asymQConfig_t outQC = {.scales = outScales,
                           .zeroPoints = outZps,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 3,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[1] = {0};
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&in, &out));
}

void testRequantDynamicAccumulatorRangeMatchesGold() {
    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f1AccumRange;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f1AccumRange_len];
    memcpy(inData, input_requant_f1AccumRange, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f1AccumRange_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, outData,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             outQc.scale);
    // out-of-place: the input tensor must be untouched (pass A reads only)
    TEST_ASSERT_EQUAL_INT32_ARRAY(input_requant_f1AccumRange, inData,
                                  input_requant_f1AccumRange_len);
    TEST_ASSERT_EQUAL_FLOAT(inputScale_requant_f1AccumRange, inQc.scale);
}

void testRequantDynamicAbsmaxZeroGivesZerosScaleOne() {
    size_t dims[] = {input_requant_f2AbsmaxZero_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f2AbsmaxZero;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f2AbsmaxZero_len];
    memcpy(inData, input_requant_f2AbsmaxZero, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f2AbsmaxZero_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f2AbsmaxZero, outData,
                                  expected_requant_f2AbsmaxZero_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f2AbsmaxZero, expectedScale_requant_f2AbsmaxZero,
                             outQc.scale);
}

void testRequantDynamicScaleTracksInputRescale() {
    // ONE output tensor + qConfig reused across BOTH calls (no re-init):
    // a kernel that fails to recompute/write the scale per call keeps the
    // stale value and fails the second assert (freeze-the-scale class).
    size_t dims[] = {input_requant_f3Rescale_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScaleA_requant_f3Rescale;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f3Rescale_len];
    memcpy(inData, input_requant_f3Rescale, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f3Rescale_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedA_requant_f3Rescale, outData,
                                  expectedA_requant_f3Rescale_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTolA_requant_f3Rescale, expectedScaleA_requant_f3Rescale,
                             outQc.scale);

    // same mantissas, input scale x10 -> fresh scale must track ~x10
    inQc.scale = inputScaleB_requant_f3Rescale;
    requantSymInt32Tensor(&inTensor, &outTensor);
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedB_requant_f3Rescale, outData,
                                  expectedB_requant_f3Rescale_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTolB_requant_f3Rescale, expectedScaleB_requant_f3Rescale,
                             outQc.scale);
}

void testRequantDynamicTieRoundsHalfAwayFromZero() {
    // quotients land on exact .5 ties (gold construction: scale == 4.0f);
    // half-to-even AND floor/trunc casts produce different mantissas.
    size_t dims[] = {input_requant_f4Tie_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f4Tie;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f4Tie_len];
    memcpy(inData, input_requant_f4Tie, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f4Tie_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32Tensor(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f4Tie, outData, expected_requant_f4Tie_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f4Tie, expectedScale_requant_f4Tie, outQc.scale);
}

void testRequantDynamicInPlaceAliasMatchesGold() {
    // In-place contract: ONE tensor_t passed as input AND output. Its single
    // qConfig is read in the input role (scale on entry) and written in the
    // output role (fresh scale on exit); qMaxBits/roundingMode of that same
    // config define the target. Pass B rewrites the mantissas index-by-index.
    // Result must be bit-identical to the out-of-place gold.
    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t qc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &qc, (uint8_t)qMaxBits_requant);
    qc.scale = inputScale_requant_f1AccumRange;
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    int32_t data[input_requant_f1AccumRange_len];
    memcpy(data, input_requant_f1AccumRange, sizeof(data));
    tensor_t tensor;
    setTensorValues(&tensor, (uint8_t *)data, &shape, &q, NULL);

    requantSymInt32Tensor(&tensor, &tensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, data,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             qc.scale);
}

void testRequantDynamicViaConversionMatrixDiagonal() {
    // The Quant layer (PR D) dispatches directly over the matrix; pin the
    // diagonal wiring and that it behaves identically to a direct call.
    conversionFunction_t conversionFn = conversionMatrix[SYM_INT32][SYM_INT32];
    TEST_ASSERT_NOT_NULL(conversionFn);

    size_t dims[] = {input_requant_f1AccumRange_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f1AccumRange;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f1AccumRange_len];
    memcpy(inData, input_requant_f1AccumRange, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f1AccumRange_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    conversionFn(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f1AccumRange, outData,
                                  expected_requant_f1AccumRange_len);
    TEST_ASSERT_FLOAT_WITHIN(scaleTol_requant_f1AccumRange, expectedScale_requant_f1AccumRange,
                             outQc.scale);
}

void testConvertTensorSymInt32SameTypeKeepsCopySemantics() {
    // Pins the spec-D1 invariant the PR-D Quant layer relies on:
    // convertTensor's same-type branch short-circuits BEFORE the matrix
    // lookup and stays memmove + scale copy — wiring the diagonal must NOT
    // change it. A requant here would yield {10922, -21845, 32767} with a
    // fresh scale 150/32767 instead of the copied mantissas + scale 0.5f.
    size_t dims[] = {3};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, 16);
    inQc.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[] = {100, -200, 300};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, 16);
    outQc.scale = 999.f;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[3];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expectedCopy[] = {100, -200, 300};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCopy, outData, 3);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQc.scale);
}

void testRequantToScaleNonSaturatingMatchesGold() {
    size_t dims[] = {input_requant_f5ToScaleFit_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f5ToScaleFit;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f5ToScaleFit_len];
    memcpy(inData, input_requant_f5ToScaleFit, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f5ToScaleFit; // pre-set target (fixed-scale contract)
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f5ToScaleFit_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f5ToScaleFit, outData,
                                  expected_requant_f5ToScaleFit_len);
    // fixed-scale contract: the pre-set target scale is NEVER modified
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f5ToScaleFit, outQc.scale);
}

void testRequantToScaleSaturatesAtQMinQMax() {
    // target scale deliberately too small: quotients overshoot BOTH bounds;
    // clamping at qMin/qMax is the documented Deutel-Eq.4 semantics.
    size_t dims[] = {input_requant_f6ToScaleSat_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f6ToScaleSat;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    int32_t inData[input_requant_f6ToScaleSat_len];
    memcpy(inData, input_requant_f6ToScaleSat, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f6ToScaleSat;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    int32_t outData[input_requant_f6ToScaleSat_len];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inTensor, &outTensor);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f6ToScaleSat, outData,
                                  expected_requant_f6ToScaleSat_len);
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f6ToScaleSat, outQc.scale);
}

void testRequantToScaleSharedBufferAliasMatchesGold() {
    // In-place for the fixed-scale variant = SHARED DATA BUFFER with two
    // tensor_t views, each with its OWN qConfig (input scale vs pre-set
    // target). Passing one tensor_t twice would force inScale == targetScale
    // (both roles share a single scale field) — a no-op requant — so the
    // two-view setup is the realistic aliasing mode. Pins the single
    // same-index read-then-write pass.
    size_t dims[] = {input_requant_f6ToScaleSat_len};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t data[input_requant_f6ToScaleSat_len];
    memcpy(data, input_requant_f6ToScaleSat, sizeof(data));

    symInt32QConfig_t inQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQc, (uint8_t)qMaxBits_requant);
    inQc.scale = inputScale_requant_f6ToScaleSat;
    quantization_t inQ;
    initSymInt32Quantization(&inQc, &inQ);
    tensor_t inView;
    setTensorValues(&inView, (uint8_t *)data, &shape, &inQ, NULL);

    symInt32QConfig_t outQc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &outQc, (uint8_t)qMaxBits_requant);
    outQc.scale = targetScale_requant_f6ToScaleSat;
    quantization_t outQ;
    initSymInt32Quantization(&outQc, &outQ);
    tensor_t outView;
    setTensorValues(&outView, (uint8_t *)data, &shape, &outQ, NULL);

    requantSymInt32TensorToScale(&inView, &outView);

    TEST_ASSERT_EQUAL_INT32_ARRAY(expected_requant_f6ToScaleSat, data,
                                  expected_requant_f6ToScaleSat_len);
    TEST_ASSERT_EQUAL_FLOAT(targetScale_requant_f6ToScaleSat, outQc.scale);
}

void testConversionSymSymInt32SignExtends() {
    size_t numValues = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    /* SYM source, qBits=6, scale 0.5; mantissas {3,-3,31,-32} packed */
    float inQCScaleArr[1] = {0.f};
    symQConfig_t inQC = {.scales = inQCScaleArr, .numGroups = 1, .groupSize = 0};
    inQC.scales[0] = 0.5f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t srcMant[] = {3, -3, 31, -32};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, numValues));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    /* pack the signed mantissas into the SYM bitstream */
    byteConversion((uint8_t *)srcMant, 32, inTensor.data, 6, numValues);

    symInt32QConfig_t outQC = {0};
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[4];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expectedMant[] = {3, -3, 31, -32};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedMant, outData, numValues); /* sign preserved */
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scale);                      /* scale carried */
    TEST_ASSERT_EQUAL_UINT8(6, outQC.qMaxBits);                      /* width recorded */
    freeReservedMemory(inBuf);
}

void testConversionSymFloat32Dequantizes() {
    size_t n = 3;
    size_t dims[] = {3};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    float inQCScaleArr[1] = {0.f};
    symQConfig_t inQC = {.scales = inQCScaleArr, .numGroups = 1, .groupSize = 0};
    inQC.scales[0] = 0.25f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t mant[] = {4, -4, 2};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)mant, 32, inTensor.data, 6, n);

    quantization_t outQ;
    initFloat32Quantization(&outQ);
    float outData[3];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    float expected[] = {1.0f, -1.0f, 0.5f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, outData, n);
    freeReservedMemory(inBuf);
}

void testConversionSymInt32CodesDropScale() {
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    float inQCScaleArr[1] = {0.f};
    symQConfig_t inQC = {.scales = inQCScaleArr, .numGroups = 1, .groupSize = 0};
    inQC.scales[0] = 7.5f;
    inQC.qBits = 6; /* scale must be IGNORED */
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t mant[] = {5, -5, 1, -32};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)mant, 32, inTensor.data, 6, n);

    quantization_t outQ;
    initInt32Quantization(&outQ);
    int32_t outData[4];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    int32_t expected[] = {5, -5, 1, -32};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, outData, n);
    freeReservedMemory(inBuf);
}

void testConversionSymAsymRescaleRoundTrips() {
    /* Round-trip: SYM -> ASYM -> FLOAT32 recovers dequantized SYM values.
     *
     * Fixture: n=6, SYM qBits=6, scale=0.5, mantissas {10,-8,4,-2,6,-10}.
     * Dequantized SYM: deq[i] = mant[i] * 0.5 => {5.0, -4.0, 2.0, -1.0, 3.0, -5.0}.
     *
     * ASYM qBits=5: range=10.0, qMax=2^5-1=31, asym scale=10/31≈0.32258.
     * zeroPoint=round(-5.0/(10/31))=-16; codes={31,4,22,13,25,1} (clamped to [0,31]).
     * Codes + scale + zeroPoint are pinned exactly below; the round-trip leg is a
     * secondary sanity check (worst-case error ≈0.16, tolerance 0.35 covers clipping).
     */
    size_t n = 6;
    size_t dims[] = {6};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    /* Build SYM input */
    float inQCScaleArr[1] = {0.f};
    symQConfig_t inQC = {.scales = inQCScaleArr, .numGroups = 1, .groupSize = 0};
    inQC.scales[0] = 0.5f;
    inQC.qBits = 6;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t srcMant[] = {10, -8, 4, -2, 6, -10};
    uint8_t *inBuf = reserveMemory(calcNumberOfBytesForData(&inQ, n));
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);
    byteConversion((uint8_t *)srcMant, 32, inTensor.data, 6, n);

    /* ASYM output tensor */
    float asymScalesArr[1] = {1.f};
    uint16_t asymZpsArr[1] = {0};
    asymQConfig_t asymQC = {.scales = asymScalesArr,
                            .zeroPoints = asymZpsArr,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 5,
                            .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymData[n * calcBytesPerElement(&asymQ)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    /* FLOAT32 output tensor for round-trip verification */
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float outF[6];
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)outF, &shape, &floatQ, NULL);

    /* Convert SYM -> ASYM -> FLOAT32 */
    convertTensor(&inTensor, &asymTensor);
    /* Re-pinned to the nudged code-domain grid (group-quant PR4, D6): the
     * dequantized band {5,-4,2,-1,3,-5} spans zero (nudge no-op), scale stays
     * 10/31, zp becomes code-domain +16 -- and unlike f1SpanZero the CODES
     * genuinely change here: the old single-round encode gave code 1 for
     * -5.0 (round(-15.5 + 16) = round(0.5) = 1) while the new
     * round-then-shift encode gives 0 (round(-15.5) = -16, +16 = 0):
     * HALF_AWAY's "away" direction flips with the shift. Gold:
     * expected_asym_nudged.h f5SymDeq. */
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, scale_asym_nudged_f5SymDeq, asymQC.scales[0]);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f5SymDeq, asymQC.zeroPoints[0]);
    uint8_t flattenedAsymData[n];
    byteConversion(asymTensor.data, asymQC.qBits, flattenedAsymData, 8, n);
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_UINT8((uint8_t)codes_asym_nudged_f5SymDeq[i], flattenedAsymData[i]);
    }
    convertTensor(&asymTensor, &floatTensor);

    /* Expected: dequantized SYM values */
    float expected[] = {5.0f, -4.0f, 2.0f, -1.0f, 3.0f, -5.0f};

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(0.35f, expected[i], outF[i]);
    }

    freeReservedMemory(inBuf);
}

void testConversionSymInt32ToSymRescaleRoundTrips() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t numberOfDims = 1;
    size_t orderOfDims[] = {0};
    shape_t shape = {
        .dimensions = dims, .numberOfDimensions = numberOfDims, .orderOfDimensions = orderOfDims};

    /* Input: SYM_INT32 with scale=0.25, mantissas span [-40, 40].
     * Dequantized values: mantissa * 0.25 = {10, -8, 4, -2, 6, -10}; absmax = 10.0. */
    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.25f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {40, -32, 16, -8, 24, -40};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    /* Output: SYM with qBits=6.
     * Expected fresh scale = absmax / (2^(6-1) - 1) = 10.0 / 31 ≈ 0.322580645. */
    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &symTensor);

    /* Assert fresh output scale. */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 10.0f / 31.0f, outQC.scales[0]);

    /* Manually unpack codes and dequantize; verify within one quant step.
     * One quant step = scale ≈ 0.323; tolerance = 0.33f. */
    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    float expectedVal[] = {10.f, -8.f, 4.f, -2.f, 6.f, -10.f};
    for (size_t i = 0; i < n; i++) {
        float rec = (float)codes[i] * outQC.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(0.33f, expectedVal[i], rec);
    }

    /* Representative codes: 10.0 / (10.0/31) = 31; -10.0 / (10.0/31) = -31. */
    TEST_ASSERT_INT32_WITHIN(1, 31, codes[0]);
    TEST_ASSERT_INT32_WITHIN(1, -31, codes[5]);
}

void testRepackSymInt32ToSymNoRescaleFittingCarriesScale() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {5, -5, 31, -32, 0, 12};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    outQC.scales[0] = 999.0f;
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    repackSymInt32ToSymNoRescale(&inTensor, &symTensor);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    TEST_ASSERT_EQUAL_INT32_ARRAY(inData, codes, 6);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scales[0]);
}

void testRepackSymInt32ToSymNoRescaleRejectsOverflow() {
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {5, 40, -5, 0, 0, 0};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    outQC.scales[0] = 999.0f;
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(repackSymInt32ToSymNoRescale(&inTensor, &symTensor));
}

void testConversionFloatToSymRoundTripsSymmetric() {
    /* n=6, absMax=3.5 => scale = 3.5 / (2^(6-1) - 1) = 3.5/31 ≈ 0.112903226.
     * One quant step = scale ≈ 0.113; worst-case round-trip error = scale/2 ≈ 0.056;
     * tolerance 0.12 is > one full step to cover HALF_AWAY rounding at the boundary. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    float floatData[] = {1.5f, -2.5f, 3.0f, -1.0f, 0.5f, -3.5f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &symTensor);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.5f / 31.0f, outQC.scales[0]);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    for (size_t i = 0; i < n; i++) {
        float rec = (float)codes[i] * outQC.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(0.12f, floatData[i], rec);
    }

    /* Prove symmetric range: the OLD buggy code clamped to [0, qMax-1] so
     * negative inputs became 0; a non-zero negative code proves correct range. */
    TEST_ASSERT_TRUE(codes[1] < 0); /* floatData[1] = -2.5 */
    TEST_ASSERT_TRUE(codes[5] < 0); /* floatData[5] = -3.5 */
}

void testConversionInt32ToSymNoRescaleScale1() {
    /* INT32 codes fitting qBits=6 range [-32, 31] pack with scale=1. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {5, -5, 31, -32, 0, 12};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    outQC.scales[0] = 999.0f; /* garbage — proves scale=1 is written by the cell */
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&intTensor, &symTensor);

    int32_t codes[6];
    symTestUnpackSignExtend(symTensor.data, 6, codes, 6);
    TEST_ASSERT_EQUAL_INT32_ARRAY(intData, codes, 6);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, outQC.scales[0]);
}

void testConversionInt32ToSymRejectsOutOfRange() {
    /* An INT32 code outside [-32, 31] for qBits=6 must exit(1).
     * Mutation guard: if packChunkGuarded's range check is removed, the out-of-range
     * code 40 truncates silently and the child exits 0, failing this test. */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    int32_t intData[] = {5, 40, -5, 0, 0, 0}; /* 40 > 31: out of range for qBits=6 */
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&intTensor, &symTensor));
}

void testChunkedFloatToSymRoundTripsAtChunkBoundary(void) {
    /* Characterization: FLOAT32 -> SYM (packFloatBufferAsSym) at n=517, qBits=3,
     * straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries (chunks [0,256),
     * [256,512), [512,517)) must round-trip within one half quantization step
     * per element. Pinned GREEN on the old whole-tensor VLA before the #296
     * Stage 2 chunk-loop rewrite; must stay GREEN after. */
    size_t n = 517;
    uint8_t qBits = 3;
    float *vals = (float *)reserveMemory(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        vals[i] = ((float)((i * 2654435761u) % 1000) - 500.f) / 25.f;
    }
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatInQ;
    initFloat32Quantization(&floatInQ);
    tensor_t floatIn;
    setTensorValues(&floatIn, (uint8_t *)vals, &shape, &floatInQ, NULL);

    float symQCScale[1] = {1.f};
    symQConfig_t symQC = {.scales = symQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = qBits};
    quantization_t symQ;
    initSymQuantization(&symQC, &symQ);
    uint8_t *symData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&symQ, n));
    tensor_t symOut;
    setTensorValues(&symOut, symData, &shape, &symQ, NULL);

    convertTensor(&floatIn, &symOut);

    float *decoded = (float *)reserveMemory(n * sizeof(float));
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&symOut, &floatOut);

    float tol = symQC.scales[0] * 0.5f + 1e-3f;
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(tol, vals[i], decoded[i]);
    }

    freeReservedMemory(vals);
    freeReservedMemory(symData);
    freeReservedMemory(decoded);
}

void testConversionSymInt32AsymConstantTensorNoDivByZero() {
    /* Constant tensor: min==max. The quantizeFloatToAsym degenerate branch must avoid
     * divide-by-zero and recover the constant. Before the dedup,
     * convertSymInt32TensorToAsymTensor had no min==max guard: scale=(max-min)/qMax=0,
     * so value/scale was inf and the result was garbage (UB on the float->int cast). */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.5f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {8, 8, 8, 8}; /* dequantized = 4.0 each (constant) */
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float asymScalesArr[1] = {1.f};
    uint16_t asymZpsArr[1] = {0};
    asymQConfig_t asymQC = {.scales = asymScalesArr,
                            .zeroPoints = asymZpsArr,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 5,
                            .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymData[n * calcBytesPerElement(&asymQ)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &asymQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float outF[4];
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)outF, &shape, &floatQ, NULL);

    convertTensor(&inTensor, &asymTensor);    /* SYM_INT32 -> ASYM (constant) */
    convertTensor(&asymTensor, &floatTensor); /* ASYM -> FLOAT32 round-trip */

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4f, 4.0f, outF[i]);
    }

    /* Group-quant PR4: under the nudge a NONZERO constant no longer reaches
     * the mn==mx fallback (the band [0, 4] is non-degenerate); only the
     * all-zero tensor does. Pin the adapted fallback: scale 1.f, zp 0,
     * codes 0 -- the value-zero state code 0 decodes from exactly. */
    int32_t zeroData[] = {0, 0, 0, 0};
    tensor_t zeroTensor;
    setTensorValues(&zeroTensor, (uint8_t *)zeroData, &shape, &inQ, NULL);
    convertTensor(&zeroTensor, &asymTensor);
    TEST_ASSERT_EQUAL_FLOAT(1.f, asymQC.scales[0]);
    TEST_ASSERT_EQUAL_UINT16(0, asymQC.zeroPoints[0]);
    uint8_t zeroCodes[4];
    byteConversion(asymTensor.data, 5, zeroCodes, 8, n);
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_UINT8(0, zeroCodes[i]);
    }
}

void testConversionAsymToSymRescaleOffCenterRoundTrips() {
    /* Strategy: build an ASYM input representing an off-center ALL-POSITIVE band [2, 6],
     * convert ASYM -> SYM, manually unpack the SYM output (sign-extend), dequantize with
     * the fresh SYM scale, and assert recovery within tolerance — proving the cell RESCALED
     * (a symmetric grid holds the [2,6] band only because the scale was recomputed). */
    size_t n = 6;
    size_t dims[] = {6};
    size_t orderOfDims[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = orderOfDims};

    /* Build ASYM input: qBits=5, scale=0.25, code-domain zeroPoint=+4 (PR4).
     * asym codes {12,16,20,24,28,14} -> reals = (code - zeroPoint)*scale = (code-4)*0.25
     * = {2.0, 3.0, 4.0, 5.0, 6.0, 2.5} — off-center all-positive band [2, 6]. */
    float inScalesArr[1] = {0.25f};
    uint16_t inZpsArr[1] = {4};
    asymQConfig_t inQC = {.scales = inScalesArr,
                          .zeroPoints = inZpsArr,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);

    int32_t asymCodes[] = {12, 16, 20, 24, 28, 14};
    uint8_t asymData[calcNumberOfBytesForData(&inQ, 6)];
    byteConversion((uint8_t *)asymCodes, 32, asymData, 5, 6);
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &inQ, NULL);

    float reference[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 2.5f};

    /* SYM output qBits=6. */
    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, 6)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&asymTensor, &symTensor);

    /* Assert FRESH symmetric scale proves rescale (NOT the carried asym 0.25):
     * absMax = 6.0; scale = 6.0 / (2^(6-1) - 1) = 6.0/31. */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f / 31.0f, outQC.scales[0]);

    /* Manual unpack + dequant + compare.
     * asym codes are exact integers so the only error is the SYM requantization step
     * ≈ scale/2 = (6/31)/2 ≈ 0.097; tolerance 0.2 is conservative (< one full step). */
    int32_t symCodes[6];
    symTestUnpackSignExtend(symTensor.data, 6, symCodes, 6);
    for (size_t i = 0; i < 6; i++) {
        float rec = (float)symCodes[i] * outQC.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(0.2f, reference[i], rec);
    }

    /* All codes positive: off-center [2,6] band maps onto the positive half of the
     * symmetric grid after rescaling. */
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_TRUE(symCodes[i] > 0);
    }
}

void testConvertSymToInt32RejectsZeroQBits() {
    /* qBits==0 makes unpackSignExtend compute 1 << (srcBits - 1); srcBits is
     * size_t so 0 - 1 wraps to SIZE_MAX and the shift is UB (#247). The guard
     * must reject it before the shift. */
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inQCScaleArr[1] = {0.f};
    symQConfig_t inQC = {.scales = inQCScaleArr, .numGroups = 1, .groupSize = 0};
    inQC.scales[0] = 0.5f;
    inQC.qBits = 0; /* degenerate */
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inBuf[4] = {0}; /* never read: the guard fires first */
    tensor_t inTensor;
    setTensorValues(&inTensor, inBuf, &shape, &inQ, NULL);

    quantization_t outQ;
    initInt32Quantization(&outQ);
    int32_t outData[4];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testConvertInt32ToSymRejectsZeroQBits() {
    /* qBits==0 makes packChunkGuarded compute 1 << (dstBits - 1) with dstBits as
     * size_t (0 - 1 -> SIZE_MAX): UB shift, plus a 0-width byteConversion whose
     * memset size underflows (#247). The guard must reject it up front. */
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t intData[] = {1, -1, 2, -2};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t intTensor;
    setTensorValues(&intTensor, (uint8_t *)intData, &shape, &intQ, NULL);

    float outQCScaleArr[1] = {0.f};
    symQConfig_t outQC = {.scales = outQCScaleArr, .numGroups = 1, .groupSize = 0};
    outQC.qBits = 0; /* degenerate */
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symBuf[4] = {0};
    tensor_t symTensor;
    setTensorValues(&symTensor, symBuf, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&intTensor, &symTensor));
}

void testConvertersPreserveCallerOutputShape() {
    /* Contract (#247): converters write data + qconfig only; the CALLER owns the
     * output tensor's shape, so the converter must NOT repoint output->shape.
     * Pins the removal of the copyDimsAndSparsityToTensor shape-pointer steal,
     * which double-freed heap-owned-shape outputs (see UnitTestLinear /
     * UnitTestAdd). Distinct in/out shape structs make the steal observable as a
     * changed pointer. */
    size_t inDims[] = {6};
    size_t inOrder[] = {0};
    shape_t inShape = {.dimensions = inDims, .numberOfDimensions = 1, .orderOfDimensions = inOrder};
    size_t outDims[] = {6};
    size_t outOrder[] = {0};
    shape_t outShape = {
        .dimensions = outDims, .numberOfDimensions = 1, .orderOfDimensions = outOrder};

    /* FLOAT32 -> ASYM (the converter UnitTestLinear documents as the double-free) */
    float floatData[6] = {1.f, 2.f, 3.f, 4.f, -1.f, -2.f};
    quantization_t floatInQ;
    initFloat32Quantization(&floatInQ);
    tensor_t floatIn;
    setTensorValues(&floatIn, (uint8_t *)floatData, &inShape, &floatInQ, NULL);
    float asymScalesArr[1] = {1.f};
    uint16_t asymZpsArr[1] = {0};
    asymQConfig_t asymQC = {.scales = asymScalesArr,
                            .zeroPoints = asymZpsArr,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 5,
                            .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymData[calcNumberOfBytesForData(&asymQ, 6)];
    tensor_t asymOut;
    setTensorValues(&asymOut, asymData, &outShape, &asymQ, NULL);
    convertTensor(&floatIn, &asymOut);
    TEST_ASSERT_EQUAL_PTR(&outShape, asymOut.shape);

    /* INT32 -> FLOAT32 */
    int32_t intData[6] = {1, 2, 3, 4, -1, -2};
    quantization_t intInQ;
    initInt32Quantization(&intInQ);
    tensor_t intIn;
    setTensorValues(&intIn, (uint8_t *)intData, &inShape, &intInQ, NULL);
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    float floatOutData[6];
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)floatOutData, &outShape, &floatOutQ, NULL);
    convertTensor(&intIn, &floatOut);
    TEST_ASSERT_EQUAL_PTR(&outShape, floatOut.shape);

    /* ASYM -> FLOAT32 */
    float asymInScales[1] = {0.25f};
    uint16_t asymInZps[1] = {4};
    asymQConfig_t asymInQC = {.scales = asymInScales,
                              .zeroPoints = asymInZps,
                              .numGroups = 1,
                              .groupSize = 0,
                              .qBits = 5,
                              .roundingMode = HALF_AWAY};
    quantization_t asymInQ;
    initAsymQuantization(&asymInQC, &asymInQ);
    int32_t asymCodes[6] = {12, 16, 20, 8, 24, 14};
    uint8_t asymInData[calcNumberOfBytesForData(&asymInQ, 6)];
    byteConversion((uint8_t *)asymCodes, 32, asymInData, 5, 6);
    tensor_t asymIn;
    setTensorValues(&asymIn, asymInData, &inShape, &asymInQ, NULL);
    quantization_t floatOut2Q;
    initFloat32Quantization(&floatOut2Q);
    float floatOut2Data[6];
    tensor_t floatOut2;
    setTensorValues(&floatOut2, (uint8_t *)floatOut2Data, &outShape, &floatOut2Q, NULL);
    convertTensor(&asymIn, &floatOut2);
    TEST_ASSERT_EQUAL_PTR(&outShape, floatOut2.shape);
}

void testAccumulateSymFixedGridFirstStoreDerivesGridThenCarries(void) {
    /* Fresh (all-zero) 4-elem SYM@6 target: first accumulate derives
     * scale = absmax(inc)/31 from the increment; second accumulate must CARRY
     * that scale verbatim (fit-preserving). Mutation guard: swapping the
     * fixed-grid store for packFloatBufferAsSym re-derives scale on the 2nd
     * call -> scale assert RED.
     *
     * Derivation (HALF_AWAY; cross-checked with a throwaway float32 harness
     * mirroring the exact arithmetic below -- house style, recon-pack §2):
     * call1: absMax(inc1) = 3.1 -> scale = 3.1/31 ~= 0.1.
     *   codes1[i] = round(inc1[i]/scale): round(31)=31, round(-15.5)=-16
     *   (HALF_AWAY, ties away from zero), round(7.75)=8, round(3.875)=4.
     * call2 (carried scale): codes2[i] = round(codes1[i] + inc2[i]/scale):
     *   round(31 - 1) = 30, round(-16 + 1) = -15, round(8+0) = 8, round(4+0) = 4.
     * inc2[0] is -0.1, not +0.1: +0.1 would push element 0 to 32, overflowing
     * the 6-bit grid (exercised separately by the overflow-aborts test). */
    uint8_t data[3] = {0};
    size_t dims[] = {1, 4};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    float qcScale[1] = {1.f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc1[] = {3.1f, -1.55f, 0.775f, 0.3875f};
    accumulateFloatIntoSymTensorFixedGrid(&target, inc1, 4);
    float scaleAfter1 = qc.scales[0];
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.1f / 31.0f, scaleAfter1);

    int32_t codes1[4];
    symTestUnpackSignExtend(data, 6, codes1, 4);
    int32_t expectedCodes1[] = {31, -16, 8, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCodes1, codes1, 4);

    float inc2[] = {-0.1f, 0.1f, 0.f, 0.f};
    accumulateFloatIntoSymTensorFixedGrid(&target, inc2, 4);

    TEST_ASSERT_EQUAL_FLOAT(scaleAfter1, qc.scales[0]); /* carried, not re-derived */

    int32_t codes2[4];
    symTestUnpackSignExtend(data, 6, codes2, 4);
    int32_t expectedCodes2[] = {30, -15, 8, 4};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCodes2, codes2, 4);
}

void testAccumulateSymFixedGridZeroIncrementIsBitExact(void) {
    /* Carried-grid exactness (HALF_AWAY): accumulating an all-zero increment
     * must leave packed bytes AND scale bit-identical (on-grid values survive
     * re-round exactly; recon-pack §2 proof: mant*scale/scale round-trips to
     * mant exactly for |mant| well under 2^15, which every 6-bit mantissa is).
     * Mutation guard (verified by deliberately breaking the primitive and
     * confirming this test goes RED, per house mutation-testing convention):
     * the rescale variant re-derives scale from the (nonzero) dequantized
     * values and re-rounds every element -> byte/scale assert RED. The seed
     * is packed directly at a DELIBERATELY LOOSE grid (mantissas far from
     * the +-31/-32 boundary): a first mutation attempt seeded via the
     * derive path instead (absmax mapped onto the full +-31 range) and the
     * rescale mutant's fresh-absmax re-derivation accidentally reproduced
     * the same tight grid, making the test pass even under the mutation --
     * this loose-grid seed is required for real discriminating power.
     *
     * Seed directly: scale=0.1, mant={5,-2,3,-4} (not touching the range
     * boundary) -> dequant={0.5,-0.2,0.3,-0.4}. Carried-grid zero-increment
     * codes stay exactly {5,-2,3,-4} (recon-pack §2). A rescale mutant would
     * instead re-derive from absMax(dequant)=0.5: scale'=0.5/31~=0.016129,
     * codes'=round(dequant/scale')={31,-12,19,-25} -- a different scale AND
     * different bytes (verified via a throwaway float32 harness). */
    int32_t seedMant[] = {5, -2, 3, -4};
    uint8_t data[3];
    byteConversion((uint8_t *)seedMant, 32, data, 6, 4);
    size_t dims[] = {1, 4};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    float qcScale[1] = {0.1f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    uint8_t snapshotBytes[3];
    memcpy(snapshotBytes, data, 3);
    float snapshotScale = qc.scales[0];

    float zeroInc[] = {0.f, 0.f, 0.f, 0.f};
    accumulateFloatIntoSymTensorFixedGrid(&target, zeroInc, 4);

    TEST_ASSERT_EQUAL_UINT8_ARRAY(snapshotBytes, data, 3);
    TEST_ASSERT_EQUAL_FLOAT(snapshotScale, qc.scales[0]);
}

void testAccumulateSymFixedGridOverflowAborts(void) {
    /* D2: growing past the 6-bit grid must exit(1) (#227 message), never
     * clamp. Seed mantissa 31 (grid max) directly via byteConversion -- so
     * the CARRIED scale (not a derived one) is under test, since the target
     * is not all-zero -- and add +1 grid step (inc == scale): 31 -> 32,
     * outside the 6-bit range [-32, 31].
     * Mutation guard: clamping instead of fit-guarding lets the child exit
     * 0 (no abort) -> RED. */
    size_t n = 1;
    size_t dims[] = {1};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScale[1] = {0.1f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    int32_t mant[] = {31};
    byteConversion((uint8_t *)mant, 32, data, 6, n);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc[] = {0.1f}; /* +1 grid step: 31 -> 32, overflow */
    ASSERT_EXITS_WITH_FAILURE(accumulateFloatIntoSymTensorFixedGrid(&target, inc, n));
}

void testAccumulateSymRescaleRederivesGridEachCall(void) {
    /* DYNAMIC semantics: after a second accumulate with larger values, scale
     * reflects the NEW absmax (fresh grid every store), not the carried one.
     * Also asserts dequant equivalence within one grid-step tolerance.
     * Mutation guard: the fixed-grid variant would either abort (the carried
     * scale=0.5 grid cannot hold mant+100) or, on a fresh all-zero target,
     * keep an unrelated scale -> the scale1/codes1 asserts RED either way.
     *
     * Seed directly: scale=0.5, mant={2,-1,0,0} -> dequant={1.0,-0.5,0,0}
     * (bit-exact: both values are exact powers of two at this scale).
     * call1 (inc=0): absMax=1.0 -> scale1=1.0/31; codes1=round(dequant/scale1)
     *   = round(31)=31, round(-15.5)=-16 (HALF_AWAY), round(0)=0, round(0)=0.
     * call2 (inc=50 each): reference2[i] = codes1[i]*scale1 + 50 (computed at
     *   runtime from the SUT's own call1 output -- no hand-rounded literal
     *   needed); absMax(reference2) ~= 51 -> scale2 ~= 51/31 ~= 1.645, over
     *   50x scale1 (rederived, not carried). Reconstruction must land within
     *   one grid step (scale2) of reference2. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t seedMant[] = {2, -1, 0, 0};
    uint8_t data[3];
    byteConversion((uint8_t *)seedMant, 32, data, 6, 4);
    float qcScale[1] = {0.5f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc1[] = {0.f, 0.f, 0.f, 0.f};
    accumulateFloatIntoSymTensorRescale(&target, inc1, n);
    float scale1 = qc.scales[0];
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f / 31.0f, scale1);

    int32_t codes1[4];
    symTestUnpackSignExtend(data, 6, codes1, 4);
    int32_t expectedCodes1[] = {31, -16, 0, 0};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expectedCodes1, codes1, 4);

    float reference2[4];
    for (size_t i = 0; i < n; i++) {
        reference2[i] = (float)codes1[i] * scale1 + 50.f;
    }

    float inc2[] = {50.f, 50.f, 50.f, 50.f};
    accumulateFloatIntoSymTensorRescale(&target, inc2, n);
    float scale2 = qc.scales[0];

    /* Rederivation, not carry: scale2 reflects the new (much larger) absmax. */
    TEST_ASSERT_TRUE(scale2 > scale1 * 10.f);

    float absMax2 = reference2[0];
    for (size_t i = 1; i < n; i++) {
        if (reference2[i] > absMax2) {
            absMax2 = reference2[i];
        }
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, absMax2 / 31.0f, scale2);

    int32_t codes2[4];
    symTestUnpackSignExtend(data, 6, codes2, 4);
    for (size_t i = 0; i < n; i++) {
        float recon = (float)codes2[i] * scale2;
        TEST_ASSERT_FLOAT_WITHIN(scale2, reference2[i], recon);
    }
}

void testAccumulateAsymRescaleMatchesFloatReference(void) {
    /* ASYM: decode+add+requant equals numpy-style float reference within one
     * affine grid step; zeroPoint/scale re-derived per store (D4).
     * Mutation guard: carrying the OLD scale/zeroPoint (0.25/+4) instead of
     * rederiving would leave qc.scales[0]/qc.zeroPoints[0] unchanged -> the
     * exact asserts below RED.
     *
     * Re-pinned to the nudged code-domain grid (group-quant PR4, D6) -- the
     * nudge is LIVE here, this is a real semantics change: seed ASYM@5 codes
     * {12,16,20,24} at scale=0.25, zp=+4 -> dequant = (code-zp)*scale =
     * {2, 3, 4, 5} (exact integers). inc = {1.0, -0.5, 2.0, 0.25} ->
     * reference = {3, 2.5, 6, 5.25}: an ALL-POSITIVE band, which the nudge
     * extends to [0, 6], so scale' = 6/31 (old un-nudged: 3.5/31) and
     * zp' = round(0/scale') = 0 (old: 22). Gold: expected_asym_nudged.h
     * f6AccumRef (same reference values). */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScales[1] = {0.25f};
    uint16_t qcZps[1] = {4};
    asymQConfig_t qc = {.scales = qcScales,
                        .zeroPoints = qcZps,
                        .numGroups = 1,
                        .groupSize = 0,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    int32_t seedCodes[] = {12, 16, 20, 24};
    byteConversion((uint8_t *)seedCodes, 32, data, 5, n);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc[] = {1.0f, -0.5f, 2.0f, 0.25f};
    float reference[] = {3.0f, 2.5f, 6.0f, 5.25f};

    accumulateFloatIntoAsymTensorRescale(&target, inc, n);

    TEST_ASSERT_EQUAL_FLOAT(scale_asym_nudged_f6AccumRef, qc.scales[0]);
    TEST_ASSERT_EQUAL_UINT16((uint16_t)zp_asym_nudged_f6AccumRef, qc.zeroPoints[0]);

    int32_t codes[4];
    byteConversion(data, 5, (uint8_t *)codes, 32, n); /* asym codes: non-negative, no sign-extend */
    TEST_ASSERT_EQUAL_INT32_ARRAY(codes_asym_nudged_f6AccumRef, codes, n);
    for (size_t i = 0; i < n; i++) {
        float recon = (float)(codes[i] - (int32_t)qc.zeroPoints[0]) * qc.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(qc.scales[0], reference[i], recon);
    }
}

void testAccumulateAsymValueZeroAfterConfigReset(void) {
    /* With scale=1, zeroPoint=0 and zero codes (the optimizerZeroGrad reset state,
     * spec §5.3), decoded values are exactly 0 -> first accumulate equals the
     * increment quantized fresh (0.0f + inc[i] == inc[i] exactly, no
     * rounding at the add). Reference: convertFloatTensorToAsymTensor(inc)
     * calls the identical quantizeFloatToAsym helper on the identical float
     * array, so a match here is bit-for-bit, not tolerance-based.
     * Mutation guard: skipping the config reset (a stale scale/zeroPoint
     * left over from a prior store) would decode a nonzero baseline and
     * diverge from this fresh-quantize reference. */
    size_t n = 3;
    size_t dims[] = {3};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScales[1] = {1.f};
    uint16_t qcZps[1] = {0};
    asymQConfig_t qc = {.scales = qcScales,
                        .zeroPoints = qcZps,
                        .numGroups = 1,
                        .groupSize = 0,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    memset(data, 0, sizeof(data)); /* zero codes: the optimizerZeroGrad reset state */
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc[] = {2.0f, -3.5f, 0.0f};
    accumulateFloatIntoAsymTensorRescale(&target, inc, n);

    float refScales[1] = {1.f};
    uint16_t refZps[1] = {0};
    asymQConfig_t refQC = {.scales = refScales,
                           .zeroPoints = refZps,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t refQ;
    initAsymQuantization(&refQC, &refQ);
    uint8_t refData[calcNumberOfBytesForData(&refQ, n)];
    tensor_t refTensor;
    setTensorValues(&refTensor, refData, &shape, &refQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)inc, &shape, &floatQ, NULL);
    convertTensor(&floatTensor, &refTensor);

    TEST_ASSERT_EQUAL_FLOAT(refQC.scales[0], qc.scales[0]);
    TEST_ASSERT_EQUAL_UINT16(refQC.zeroPoints[0], qc.zeroPoints[0]);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(refData, data, calcNumberOfBytesForData(&refQ, n));
}

void testAccumulateSymFixedGridMatchesReferenceAtChunkBoundary(void) {
    /* Pin (#296 Stage 2): n=517 straddles three ODT_CONVERSION_CHUNK_ELEMS=256
     * chunks ([0,256), [256,512), [512,517)). Target starts fresh (all-zero),
     * so BOTH chunked phase-A scans -- the all-zero check and the grid-
     * deriving absmax-of-increment -- must see the WHOLE array: the dominant
     * increment value is seeded at index 400, inside the SECOND chunk. A
     * mutation that only scans chunk 0 for either scan would derive a
     * far-too-small scale from a truncated absmax, blowing the exact-scale
     * assertion below. Pinned GREEN on the old whole-tensor VLA implementation
     * before the chunked rewrite; must stay GREEN after (bit-identical: one
     * round per element, in element order). */
    size_t n = 517;
    uint8_t qBits = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScale[1] = {1.f};
    symQConfig_t qc = {.scales = qcScale,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .qBits = qBits};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t *data = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&q, n));
    memset(data, 0, calcNumberOfBytesForData(&q, n)); /* fresh (post-initTensor) accumulator */
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float *inc = (float *)reserveMemory(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        inc[i] = ((float)((i * 2654435761u) % 100u) - 50.f) / 50.f; /* [-1, ~1) */
    }
    inc[400] = 500.f; /* dominant absmax; lands in the SECOND chunk */

    accumulateFloatIntoSymTensorFixedGrid(&target, inc, n);

    const float qMax = powf(2, (float)qBits - 1) - 1; /* 127 */
    float expectedScale = 500.f / qMax;
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expectedScale, qc.scales[0]);

    int32_t *codes = (int32_t *)reserveMemory(n * sizeof(int32_t));
    symTestUnpackSignExtend(data, qBits, codes, n);
    for (size_t i = 0; i < n; i++) {
        float recon = (float)codes[i] * qc.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(qc.scales[0], inc[i], recon);
    }

    freeReservedMemory(codes);
    freeReservedMemory(inc);
    freeReservedMemory(data);
}

void testAccumulateSymRescaleMatchesReferenceAtChunkBoundary(void) {
    /* Pin (#296 Stage 2): n=517 straddles three ODT_CONVERSION_CHUNK_ELEMS=256
     * chunks. DYNAMIC semantics rederive the grid from the decoded-plus-
     * increment values on EVERY call; the dominant contribution comes from
     * the increment alone at index 400 (inside the SECOND chunk), so a
     * phase-A mutation that only scans chunk 0 misses it and derives a
     * far-too-small scale -- blowing the exact-scale assertion below. Pinned
     * GREEN on the old whole-tensor VLA implementation before the chunked
     * rewrite; must stay GREEN after. */
    size_t n = 517;
    uint8_t qBits = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScale[1] = {1.f};
    symQConfig_t qc = {.scales = qcScale,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .qBits = qBits};
    qc.scales[0] = 0.01f;
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t *data = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&q, n));
    int32_t *seedMant = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        seedMant[i] = (int32_t)((i * 2654435761u) % 5u) - 2; /* [-2, 2] */
    }
    byteConversion((uint8_t *)seedMant, 32, data, qBits, n);
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float *inc = (float *)reserveMemory(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        inc[i] = 0.f;
    }
    inc[400] = 500.f; /* dominant; forces the fresh absmax scan into the 2nd chunk */

    float *reference = (float *)reserveMemory(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        reference[i] = (float)seedMant[i] * 0.01f + inc[i];
    }

    accumulateFloatIntoSymTensorRescale(&target, inc, n);

    float absMax = reference[0] < 0.f ? -reference[0] : reference[0];
    for (size_t i = 1; i < n; i++) {
        float av = reference[i] < 0.f ? -reference[i] : reference[i];
        if (av > absMax) {
            absMax = av;
        }
    }
    const float qMax = powf(2, (float)qBits - 1) - 1;
    float expectedScale = absMax / qMax;
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expectedScale, qc.scales[0]);

    int32_t *codes = (int32_t *)reserveMemory(n * sizeof(int32_t));
    symTestUnpackSignExtend(data, qBits, codes, n);
    for (size_t i = 0; i < n; i++) {
        float recon = (float)codes[i] * qc.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(qc.scales[0], reference[i], recon);
    }

    freeReservedMemory(codes);
    freeReservedMemory(reference);
    freeReservedMemory(inc);
    freeReservedMemory(seedMant);
    freeReservedMemory(data);
}

void testAccumulateTensorIntoSymRescaleStreamsIncrementAcrossChunks(void) {
    /* Regression (#296 Stage 2 review): incSrcChunk's TENSOR branch
     * (dequantChunkToFloat(src->tens, off, count, out)) has no test crossing a
     * chunk boundary with a tensor-typed increment -- mutating its `off` arg
     * to a hardcoded 0 survives the whole suite (every chunk re-dequantizes
     * source elements [0, count) instead of [off, off+count)). Pin: target =
     * zero-filled packed SYM (qBits 8), n=517 (three ODT_CONVERSION_CHUNK_ELEMS
     * =256 chunks: [0,256), [256,512), [512,517)); increment = SYM_INT32
     * tensor with a dominant element (100.0, vs. 0.05 elsewhere) at index 400
     * -- inside the SECOND chunk. A hardcoded-offset-0 mutant reads chunk 0's
     * dequantized values for chunk 1 and chunk 2 alike, so index 400's true
     * value never reaches either the absmax scan or the accumulate: the
     * derived scale and decoded codes are both wrong -> RED. */
    size_t n = 517;
    uint8_t qBits = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScale[1] = {1.f};
    symQConfig_t qc = {.scales = qcScale,
                       .numGroups = 1,
                       .groupSize = 0,
                       .roundingMode = HALF_AWAY,
                       .qBits = qBits};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t *data = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&q, n));
    memset(data, 0, calcNumberOfBytesForData(&q, n)); /* fresh (post-initTensor) accumulator */
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    symInt32QConfig_t incQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &incQC, 16);
    incQC.scale = 0.001f;
    quantization_t incQ;
    initSymInt32Quantization(&incQC, &incQ);
    int32_t *incData = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        incData[i] = 50; /* 50 * 0.001 = 0.05 */
    }
    incData[400] = 100000; /* 100000 * 0.001 = 100.0 -- dominant; SECOND chunk */
    tensor_t incTensor;
    setTensorValues(&incTensor, (uint8_t *)incData, &shape, &incQ, NULL);

    accumulateTensorIntoSymRescale(&target, &incTensor);

    const float qMax = powf(2, (float)qBits - 1) - 1; /* 127 */
    float expectedScale = 100.f / qMax; /* target started zero, so absmax == the dominant value */
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expectedScale, qc.scales[0]);

    int32_t *codes = (int32_t *)reserveMemory(n * sizeof(int32_t));
    symTestUnpackSignExtend(data, qBits, codes, n);
    float tol = qc.scales[0] * 0.5f + 1e-4f;
    size_t spotIdx[] = {0, 255, 256, 400, 516};
    for (size_t s = 0; s < sizeof(spotIdx) / sizeof(spotIdx[0]); s++) {
        size_t i = spotIdx[s];
        float expectedVal = (float)incData[i] * incQC.scale;
        float recon = (float)codes[i] * qc.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(tol, expectedVal, recon);
    }

    freeReservedMemory(codes);
    freeReservedMemory(incData);
    freeReservedMemory(data);
}

void testChunkedFloatToAsymMatchesReferenceAcrossChunkBoundaries(void) {
    /* Characterization: FLOAT32 -> ASYM through convertTensor (quantizeFloatToAsym,
     * now grid-derivation + ODT_CONVERSION_CHUNK_ELEMS=256 chunked emit, #296 Stage 2)
     * must decode within one half quantization step at every chunk boundary shape --
     * below/at/above the 256-element stride -- including sub-byte qBits=3 packing,
     * where a wrong chunk byte offset would silently write into the wrong bit range
     * and blow the tolerance below (this is the mutation guard for packedByteOffset). */
    size_t sizes[] = {1, 8, 255, 256, 257, 517, 520};
    uint8_t qBitsValues[] = {3, 8, 12};
    for (size_t qi = 0; qi < sizeof(qBitsValues) / sizeof(qBitsValues[0]); qi++) {
        uint8_t qBits = qBitsValues[qi];
        for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
            size_t n = sizes[s];
            /* deterministic pseudo-random fill, seed fixed */
            float *vals = (float *)reserveMemory(n * sizeof(float));
            for (size_t i = 0; i < n; i++) {
                vals[i] = ((float)((i * 2654435761u) % 1000) - 500.f) / 25.f;
            }
            size_t dims[] = {n};
            size_t order[] = {0};
            shape_t shape = {
                .dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

            quantization_t floatInQ;
            initFloat32Quantization(&floatInQ);
            tensor_t floatIn;
            setTensorValues(&floatIn, (uint8_t *)vals, &shape, &floatInQ, NULL);

            float asymScalesArr[1] = {1.f};
            uint16_t asymZpsArr[1] = {0};
            asymQConfig_t asymQC = {.scales = asymScalesArr,
                                    .zeroPoints = asymZpsArr,
                                    .numGroups = 1,
                                    .groupSize = 0,
                                    .qBits = qBits,
                                    .roundingMode = HALF_AWAY};
            quantization_t asymQ;
            initAsymQuantization(&asymQC, &asymQ);
            uint8_t *asymData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&asymQ, n));
            tensor_t asymOut;
            setTensorValues(&asymOut, asymData, &shape, &asymQ, NULL);

            convertTensor(&floatIn, &asymOut);

            float *decoded = (float *)reserveMemory(n * sizeof(float));
            quantization_t floatOutQ;
            initFloat32Quantization(&floatOutQ);
            tensor_t floatOut;
            setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
            convertTensor(&asymOut, &floatOut);

            float tol = asymQC.scales[0] * 0.5f + 1e-3f;
            for (size_t i = 0; i < n; i++) {
                TEST_ASSERT_FLOAT_WITHIN(tol, vals[i], decoded[i]);
            }

            freeReservedMemory(vals);
            freeReservedMemory(asymData);
            freeReservedMemory(decoded);
        }
    }
}

void testChunkedSymInt32ToSymRoundTripsAtChunkBoundary(void) {
    /* Characterization: SYM_INT32 -> SYM (convertSymInt32TensorToSymTensor) at n=517,
     * qBits=6, straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries (chunks
     * [0,256), [256,512), [512,517)). The dominant (absmax) element sits at index
     * 400 -- inside the SECOND chunk -- so a pass-1 mutation that only scans the
     * first chunk derives a far-too-small scale, which blows both the exact-scale
     * assertion and the round-trip tolerance below (mutation guard, #296 Stage 2).
     * Pinned GREEN on the old whole-tensor VLA before the chunked rewrite; must
     * stay GREEN after. */
    size_t n = 517;
    uint8_t qBits = 6;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.05f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t *inData = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        inData[i] = (int32_t)((i * 2654435761u) % 1000u) - 500; /* [-500, 499] */
    }
    inData[400] = 100000; /* dominates absmax; lands in the second chunk */
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = qBits};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t *symData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&outQ, n));
    tensor_t symOut;
    setTensorValues(&symOut, symData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &symOut);

    float expectedScale = (100000.f * inQC.scale) / 31.f; /* qMax = 2^(6-1) - 1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, expectedScale, outQC.scales[0]);

    float *decoded = (float *)reserveMemory(n * sizeof(float));
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&symOut, &floatOut);

    float tol = outQC.scales[0] * 0.5f + 1e-3f;
    for (size_t i = 0; i < n; i++) {
        float expectedVal = (float)inData[i] * inQC.scale;
        TEST_ASSERT_FLOAT_WITHIN(tol, expectedVal, decoded[i]);
    }

    freeReservedMemory(inData);
    freeReservedMemory(symData);
    freeReservedMemory(decoded);
}

void testChunkedAsymToSymRoundTripsAtChunkBoundary(void) {
    /* Characterization: ASYM -> SYM (convertAsymTensorToSymTensor) at n=517, ASYM
     * qBits=8 (byte-aligned so the fixture can write raw code bytes directly),
     * straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries. The dominant
     * (absmax) code sits at index 400 -- inside the second chunk [256,512) -- so a
     * pass-1 mutation that only scans the first chunk misses it (mutation guard,
     * #296 Stage 2). Pinned GREEN on the old whole-tensor VLA before the chunked
     * rewrite; must stay GREEN after. */
    size_t n = 517;
    uint8_t qBits = 6; /* SYM output */
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScalesArr[1] = {1.0f};
    uint16_t inZpsArr[1] = {128}; /* code-domain (PR4): dequant = code - 128 */
    asymQConfig_t inQC = {.scales = inScalesArr,
                          .zeroPoints = inZpsArr,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 8,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t *asymData = (uint8_t *)reserveMemory(n);
    for (size_t i = 0; i < n; i++) {
        asymData[i] = (uint8_t)(118u + (i * 2654435761u) % 21u); /* codes [118,138] */
    }
    asymData[400] = 255; /* dequant = 255 - 128 = 127, dominates absmax */
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &inQ, NULL);

    float outQCScale[1] = {1.f};
    symQConfig_t outQC = {.scales = outQCScale,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .qBits = qBits};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t *symData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&outQ, n));
    tensor_t symOut;
    setTensorValues(&symOut, symData, &shape, &outQ, NULL);

    convertTensor(&asymTensor, &symOut);

    float expectedScale = 127.0f / 31.0f; /* qMax = 2^(6-1) - 1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, expectedScale, outQC.scales[0]);

    float *decoded = (float *)reserveMemory(n * sizeof(float));
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&symOut, &floatOut);

    float tol = outQC.scales[0] * 0.5f + 1e-3f;
    for (size_t i = 0; i < n; i++) {
        float expectedVal =
            (float)((int32_t)asymData[i] - (int32_t)inQC.zeroPoints[0]) * inQC.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(tol, expectedVal, decoded[i]);
    }

    freeReservedMemory(asymData);
    freeReservedMemory(symData);
    freeReservedMemory(decoded);
}

void testChunkedSymToAsymRoundTripsAtChunkBoundary(void) {
    /* Characterization: SYM -> ASYM (convertSymTensorToAsymTensor) at n=517, SYM
     * qBits=6 (sub-byte packed), straddling two ODT_CONVERSION_CHUNK_ELEMS=256
     * boundaries. The extreme mantissa sits at index 400 -- inside the second
     * chunk [256,512) -- so a pass-1 mutation that only scans the first chunk
     * derives a too-narrow grid, clipping that element on encode and blowing the
     * round-trip tolerance below (mutation guard, #296 Stage 2). Pinned GREEN on
     * the old whole-tensor VLA before the chunked rewrite; must stay GREEN
     * after. */
    size_t n = 517;
    uint8_t qBits = 6;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inQCScale[1] = {1.f};
    symQConfig_t inQC = {.scales = inQCScale,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .qBits = qBits};
    inQC.scales[0] = 0.2f;
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t *mant = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        mant[i] = (int32_t)((i * 2654435761u) % 20u) - 10; /* [-10, 9], well inside qBits=6 */
    }
    mant[400] = 31; /* max representable mantissa; dominates the grid */
    uint8_t *symData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&inQ, n));
    byteConversion((uint8_t *)mant, 32, symData, qBits, n);
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &inQ, NULL);

    float outScalesArr[1] = {1.f};
    uint16_t outZpsArr[1] = {0};
    asymQConfig_t outQC = {.scales = outScalesArr,
                           .zeroPoints = outZpsArr,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t *asymData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&outQ, n));
    tensor_t asymOut;
    setTensorValues(&asymOut, asymData, &shape, &outQ, NULL);

    convertTensor(&symTensor, &asymOut);

    float *decoded = (float *)reserveMemory(n * sizeof(float));
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&asymOut, &floatOut);

    float tol = outQC.scales[0] * 0.5f + 1e-3f;
    for (size_t i = 0; i < n; i++) {
        float expectedVal = (float)mant[i] * inQC.scales[0];
        TEST_ASSERT_FLOAT_WITHIN(tol, expectedVal, decoded[i]);
    }

    freeReservedMemory(mant);
    freeReservedMemory(symData);
    freeReservedMemory(asymData);
    freeReservedMemory(decoded);
}

void testChunkedSymInt32ToAsymRoundTripsAtChunkBoundary(void) {
    /* Characterization: SYM_INT32 -> ASYM (convertSymInt32TensorToAsymTensor) at
     * n=517, straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries in the
     * chunked emit pass. The extreme mantissa sits at index 400 (second chunk
     * [256,512)) so any future chunking bug in the pass-1 min/max scan is caught
     * here too. Pinned GREEN on the old whole-tensor VLA before the chunked
     * rewrite; must stay GREEN after. */
    size_t n = 517;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &inQC, 16);
    inQC.scale = 0.2f;
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t *inData = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        inData[i] = (int32_t)((i * 2654435761u) % 20u) - 10;
    }
    inData[400] = 500; /* dominates min/max; lands in the second chunk */
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float outScalesArr[1] = {1.f};
    uint16_t outZpsArr[1] = {0};
    asymQConfig_t outQC = {.scales = outScalesArr,
                           .zeroPoints = outZpsArr,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t *asymData = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&outQ, n));
    tensor_t asymOut;
    setTensorValues(&asymOut, asymData, &shape, &outQ, NULL);

    convertTensor(&inTensor, &asymOut);

    float *decoded = (float *)reserveMemory(n * sizeof(float));
    quantization_t floatOutQ;
    initFloat32Quantization(&floatOutQ);
    tensor_t floatOut;
    setTensorValues(&floatOut, (uint8_t *)decoded, &shape, &floatOutQ, NULL);
    convertTensor(&asymOut, &floatOut);

    float tol = outQC.scales[0] * 0.5f + 1e-3f;
    for (size_t i = 0; i < n; i++) {
        float expectedVal = (float)inData[i] * inQC.scale;
        TEST_ASSERT_FLOAT_WITHIN(tol, expectedVal, decoded[i]);
    }

    freeReservedMemory(inData);
    freeReservedMemory(asymData);
    freeReservedMemory(decoded);
}

void testChunkedSymToFloat32DequantizesAtChunkBoundary(void) {
    /* Characterization: SYM -> FLOAT32 (convertSymTensorToFloat32Tensor) at n=517,
     * qBits=3, straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries (chunks
     * [0,256), [256,512), [512,517)). Unpack+dequant is exact (no rounding), so
     * every element is asserted bit-for-bit -- a mutated output index (e.g.
     * out[i] instead of out[off+i]) corrupts elements once off>0. Pinned GREEN
     * on the old whole-tensor VLA before the #296 Stage 2 chunked rewrite; must
     * stay GREEN after. */
    size_t n = 517;
    uint8_t qBits = 3;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inQCScale[1] = {0.5f};
    symQConfig_t inQC = {.scales = inQCScale,
                         .numGroups = 1,
                         .groupSize = 0,
                         .qBits = qBits,
                         .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    int32_t *mant = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        mant[i] = (int32_t)(i % 8) - 4; /* spans [-4, 3]: exactly qBits=3 range */
    }
    uint8_t *packed = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&inQ, n));
    byteConversion((uint8_t *)mant, 32, packed, qBits, n);
    tensor_t inTensor;
    setTensorValues(&inTensor, packed, &shape, &inQ, NULL);

    quantization_t outQ;
    initFloat32Quantization(&outQ);
    float *decoded = (float *)reserveMemory(n * sizeof(float));
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)decoded, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    for (size_t i = 0; i < n; i++) {
        float expected = (float)mant[i] * inQC.scales[0];
        TEST_ASSERT_EQUAL_FLOAT(expected, decoded[i]);
    }

    freeReservedMemory(mant);
    freeReservedMemory(packed);
    freeReservedMemory(decoded);
}

void testChunkedAsymToFloat32DequantizesAtChunkBoundary(void) {
    /* Characterization: ASYM -> FLOAT32 (convertAsymTensorToFloatTensor) at n=517,
     * qBits=5, straddling two ODT_CONVERSION_CHUNK_ELEMS=256 boundaries. Unpack is
     * zero-extend (ASYM codes are unsigned) + affine decode, exact (no rounding):
     * every element asserted bit-for-bit. Pinned GREEN on the old whole-tensor VLA
     * before the #296 Stage 2 chunked rewrite; must stay GREEN after. */
    size_t n = 517;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScalesArr[1] = {0.25f};
    uint16_t inZpsArr[1] = {4}; /* code-domain (PR4): decode = (code - 4)*0.25 */
    asymQConfig_t inQC = {.scales = inScalesArr,
                          .zeroPoints = inZpsArr,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    int32_t *codes = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        codes[i] = (int32_t)(i % 32); /* spans [0, 31]: exactly qBits=5 range */
    }
    uint8_t *packed = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&inQ, n));
    byteConversion((uint8_t *)codes, 32, packed, inQC.qBits, n);
    tensor_t inTensor;
    setTensorValues(&inTensor, packed, &shape, &inQ, NULL);

    quantization_t outQ;
    initFloat32Quantization(&outQ);
    float *decoded = (float *)reserveMemory(n * sizeof(float));
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)decoded, &shape, &outQ, NULL);

    convertTensor(&inTensor, &outTensor);

    for (size_t i = 0; i < n; i++) {
        float expected = (float)(codes[i] - (int32_t)inQC.zeroPoints[0]) * inQC.scales[0];
        TEST_ASSERT_EQUAL_FLOAT(expected, decoded[i]);
    }

    freeReservedMemory(codes);
    freeReservedMemory(packed);
    freeReservedMemory(decoded);
}

void testDequantChunkToFloatFloat32MatchesSourceAtOffsets(void) {
    /* dequantChunkToFloat direct test, FLOAT32 cell: plain memcpy at each offset. */
    size_t n = 264; /* > ODT_CONVERSION_CHUNK_ELEMS + 8 so offset 256 has headroom */
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float *vals = (float *)reserveMemory(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        vals[i] = (float)i * 0.5f - 10.f;
    }
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t t;
    setTensorValues(&t, (uint8_t *)vals, &shape, &q, NULL);

    size_t offsets[] = {0, 8, 256};
    size_t count = 8;
    for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); oi++) {
        size_t offset = offsets[oi];
        float out[8];
        dequantChunkToFloat(&t, offset, count, out);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(vals + offset, out, count);
    }
    freeReservedMemory(vals);
}

void testDequantChunkToFloatSymInt32MatchesScaleAtOffsets(void) {
    /* dequantChunkToFloat direct test, SYM_INT32 cell: mantissa*scale, no unpack. */
    size_t n = 264;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t *mant = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        mant[i] = (int32_t)i - 100; /* spans negative/positive */
    }
    symInt32QConfig_t qc;
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &qc, 16);
    qc.scale = 0.25f;
    quantization_t q;
    initSymInt32Quantization(&qc, &q);
    tensor_t t;
    setTensorValues(&t, (uint8_t *)mant, &shape, &q, NULL);

    size_t offsets[] = {0, 8, 256};
    size_t count = 8;
    for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); oi++) {
        size_t offset = offsets[oi];
        float out[8];
        dequantChunkToFloat(&t, offset, count, out);
        for (size_t i = 0; i < count; i++) {
            float expected = (float)mant[offset + i] * qc.scale;
            TEST_ASSERT_EQUAL_FLOAT(expected, out[i]);
        }
    }
    freeReservedMemory(mant);
}

void testDequantChunkToFloatSymUnpacksSignExtendedAtOffsets(void) {
    /* dequantChunkToFloat direct test, SYM cell: sign-extend unpack via
     * unpackSignExtendChunk, then mantissa*scale. Round-trips packed sub-byte
     * data at offsets that straddle the ODT_CONVERSION_CHUNK_ELEMS boundary. */
    size_t n = 264;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t *mant = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        mant[i] = (int32_t)(i % 64) - 32; /* spans [-32, 31]: exactly qBits=6 range */
    }
    float qcScale[1] = {0.5f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t *packed = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&q, n));
    byteConversion((uint8_t *)mant, 32, packed, 6, n);
    tensor_t t;
    setTensorValues(&t, packed, &shape, &q, NULL);

    size_t offsets[] = {0, 8, 256};
    size_t count = 8;
    for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); oi++) {
        size_t offset = offsets[oi];
        float out[8];
        dequantChunkToFloat(&t, offset, count, out);
        for (size_t i = 0; i < count; i++) {
            float expected = (float)mant[offset + i] * qc.scales[0];
            TEST_ASSERT_EQUAL_FLOAT(expected, out[i]);
        }
    }
    freeReservedMemory(mant);
    freeReservedMemory(packed);
}

void testDequantChunkToFloatAsymUnpacksZeroExtendedAtOffsets(void) {
    /* dequantChunkToFloat direct test, ASYM cell: zero-extend unpack via
     * unpackZeroExtendChunk (byteConversion only, no sign bit), then the
     * code-domain decode (code - zp)*scale (PR4). */
    size_t n = 264;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t *codes = (int32_t *)reserveMemory(n * sizeof(int32_t));
    for (size_t i = 0; i < n; i++) {
        codes[i] = (int32_t)(i % 32); /* spans [0, 31]: exactly qBits=5 range */
    }
    float qcScales[1] = {0.25f};
    uint16_t qcZps[1] = {4}; /* code-domain (PR4): decode = (code - 4)*0.25 */
    asymQConfig_t qc = {.scales = qcScales,
                        .zeroPoints = qcZps,
                        .numGroups = 1,
                        .groupSize = 0,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t *packed = (uint8_t *)reserveMemory(calcNumberOfBytesForData(&q, n));
    byteConversion((uint8_t *)codes, 32, packed, 5, n);
    tensor_t t;
    setTensorValues(&t, packed, &shape, &q, NULL);

    size_t offsets[] = {0, 8, 256};
    size_t count = 8;
    for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); oi++) {
        size_t offset = offsets[oi];
        float out[8];
        dequantChunkToFloat(&t, offset, count, out);
        for (size_t i = 0; i < count; i++) {
            float expected = (float)(codes[offset + i] - (int32_t)qc.zeroPoints[0]) * qc.scales[0];
            TEST_ASSERT_EQUAL_FLOAT(expected, out[i]);
        }
    }
    freeReservedMemory(codes);
    freeReservedMemory(packed);
}

void testDequantChunkToFloatRejectsCountAboveChunk(void) {
    /* Fail-fast guard 1/2: count > ODT_CONVERSION_CHUNK_ELEMS must exit(1),
     * never silently read past the fixed-size chunk buffers. */
    size_t n = ODT_CONVERSION_CHUNK_ELEMS + 1;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float *vals = (float *)reserveMemory(n * sizeof(float));
    memset(vals, 0, n * sizeof(float));
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t t;
    setTensorValues(&t, (uint8_t *)vals, &shape, &q, NULL);

    float out[ODT_CONVERSION_CHUNK_ELEMS + 1];
    ASSERT_EXITS_WITH_FAILURE(dequantChunkToFloat(&t, 0, ODT_CONVERSION_CHUNK_ELEMS + 1, out));

    freeReservedMemory(vals);
}

void testDequantChunkToFloatRejectsMisalignedOffset(void) {
    /* Fail-fast guard 2/2: elemOffset % 8 != 0 must exit(1) -- the packed-width
     * byte-alignment contract that lets packedByteOffset skip a byte-boundary
     * check per call. */
    size_t n = 16;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[16] = {0};
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t t;
    setTensorValues(&t, (uint8_t *)vals, &shape, &q, NULL);

    float out[4];
    ASSERT_EXITS_WITH_FAILURE(dequantChunkToFloat(&t, 3, 4, out));
}

void testDequantChunkToFloatRejectsOutOfRangeOffset(void) {
    /* Fix 1 (release-review, PR #324): [elemOffset, elemOffset+count) must not
     * exceed the source tensor's own element count. Before the fix, only
     * count > ODT_CONVERSION_CHUNK_ELEMS and elemOffset % 8 != 0 were guarded --
     * an offset that starts exactly AT the tensor's end (still 8-aligned, still
     * <= chunk size) sailed through both and fell into an out-of-bounds
     * FLOAT32 memcpy. Mutation guard: removing the new range guard lets the
     * child run to completion (or crash outside the exit(1) path) -> RED
     * either way. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    quantization_t q;
    initFloat32Quantization(&q);
    tensor_t t;
    setTensorValues(&t, (uint8_t *)vals, &shape, &q, NULL);

    float out[1];
    ASSERT_EXITS_WITH_FAILURE(dequantChunkToFloat(&t, 8, 1, out));
}

/* unpackSignExtend is public with a srcStartBit so DeltaSym-style decoders
 * can sign-extend a segment that starts mid-byte: -3 at 6 bits packed at bit
 * offset 5 is bytes {0xA0, 0x07} (bit 5..7 = 1,0,1; bit 8..10 = 1,1,1). */
void testUnpackSignExtendReadsSignedCodeAtBitOffset(void) {
    uint8_t src[2] = {0xA0, 0x07};
    int32_t got[1] = {12345};
    unpackSignExtend(src, 6, 5, got, 1);
    TEST_ASSERT_EQUAL_INT32(-3, got[0]);
}

/* srcStartBit == 0 keeps the legacy behavior: the full signed 3-bit range
 * {-4..3} packed LSB-first ({0xAC, 0x8F, 0x68}) sign-extends back exactly. */
void testUnpackSignExtendOffsetZeroCoversFullSignedRange(void) {
    uint8_t src[3] = {0xAC, 0x8F, 0x68};
    int32_t got[8];
    unpackSignExtend(src, 3, 0, got, 8);
    int32_t expected[8] = {-4, -3, -2, -1, 0, 1, 2, 3};
    TEST_ASSERT_EQUAL_INT32_ARRAY(expected, got, 8);
}

void testQuantizeFloatToAsymNoOpOnEmptyTensor(void) {
    /* Fix 2 (release-review, PR #324): n==0 must no-op, never read values[0].
     * Before the fix, quantizeFloatToAsym's findMinFloat/findMaxFloat both
     * unconditionally dereference values[0] regardless of n -- UB for an
     * empty payload. Since both reads land on the same element, mn==mx
     * unconditionally at n==0, so deriveAsymGridFromMinMax's degenerate
     * branch fires and overwrites the output qConfig's scale/zeroPoint from
     * whatever sits at values[0] -- even though there is nothing to
     * quantize. Pin: pre-set sentinel scale/zeroPoint on the output qConfig
     * and a deterministic non-zero value at the input's element 0; both must
     * survive the n==0 call untouched.
     * Mutation guard: dropping the n==0 guard lets deriveAsymGridFromMinMax
     * derive from the seeded 42.f (post-nudge band [0, 42]: scale 42/31,
     * zeroPoints[0] = 0), failing both sentinel asserts below -> RED. */
    size_t dims[] = {0};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatDummy = 42.f; /* n==0: logically unreachable, but real backing
                              * storage so a would-be values[0] read stays
                              * inside owned memory (well-defined RED). */
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)&floatDummy, &shape, &floatQ, NULL);

    float asymScalesArr[1] = {123.f}; /* sentinel: must survive untouched */
    uint16_t asymZpsArr[1] = {7};
    asymQConfig_t asymQC = {.scales = asymScalesArr,
                            .zeroPoints = asymZpsArr,
                            .numGroups = 1,
                            .groupSize = 0,
                            .qBits = 5,
                            .roundingMode = HALF_AWAY};
    quantization_t asymQ;
    initAsymQuantization(&asymQC, &asymQ);
    uint8_t asymDummy;
    tensor_t asymTensor;
    setTensorValues(&asymTensor, &asymDummy, &shape, &asymQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    TEST_ASSERT_EQUAL_FLOAT(123.f, asymQC.scales[0]);
    TEST_ASSERT_EQUAL_UINT16(7, asymQC.zeroPoints[0]);
}

void testAccumulateTensorIntoSymRescaleRejectsSelfAliasedIncrement(void) {
    /* Fix 3 (release-review, PR #324): the rescale engine rewrites the
     * target's qConfig scale between phase A (fresh-grid derivation, reads
     * only) and phase B (chunked decode+requant+pack). If increment aliases
     * target, phase B's incSrcChunk dequantizes the (not-yet-repacked) shared
     * bytes using the ALREADY-overwritten fresh scale instead of the old one
     * phase A used -- silently wrong output, no crash, no exit. The funnel
     * epilogue always passes a distinct intermediate; this guards a caller
     * that violates that contract directly.
     * Mutation guard: removing the alias check lets the child run to
     * completion and exit 0 (no death) -> RED. */
    size_t n = 4;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float qcScale[1] = {0.1f};
    symQConfig_t qc = {
        .scales = qcScale, .numGroups = 1, .groupSize = 0, .qBits = 6, .roundingMode = HALF_AWAY};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    int32_t mant[] = {5, -2, 3, -4};
    byteConversion((uint8_t *)mant, 32, data, 6, n);
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    ASSERT_EXITS_WITH_FAILURE(accumulateTensorIntoSymRescale(&t, &t));
}

/* Group-quant PR2 (Task 2): group-aware FLOAT32<->SYM conversion cells, the
 * scalar-only-cell fail-fast net, group-faithful same-type copy, and the
 * promoted requantizeTensorInPlace API. Fixtures from expected_group_quant.h
 * (goldgen, cross-checked against torch.quantize_per_channel). */

void testFloatToSymGroupedDerivesPerGroupScales(void) {
    /* FLOAT32 -> SYM with a grouped target derives PER-GROUP absmax scales
     * (group of element i = i / groupSize), not one whole-tensor scale. */
    size_t n = kGroupQuantInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)kGroupQuantInput, &shape, &floatQ, NULL);

    float scales[3] = {1.f, 1.f, 1.f};
    symQConfig_t outQC = {.scales = scales,
                          .numGroups = (size_t)kGroupQuantNumGroups,
                          .groupSize = (size_t)kGroupQuantGroupSize,
                          .roundingMode = HALF_AWAY,
                          .qBits = (uint8_t)kGroupQuantQBits};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &symTensor);

    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[0], outQC.scales[0]);
    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[1], outQC.scales[1]);
    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[2], outQC.scales[2]);

    int32_t codes[12];
    symTestUnpackSignExtend(symTensor.data, (size_t)kGroupQuantQBits, codes, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kGroupQuantCodes, codes, n);
}

void testFloatToSymGroupedMatchesTorchPerChannel(void) {
    /* Cross-checked in the generator against torch.quantize_per_channel
     * (axis=0, per-out-channel groups of 4) over an 8x4 GEMM-weight-shaped
     * fixture at qBits=8 -- the exact grouping a linear/conv weight would
     * use for per-out-channel quantization. */
    size_t n = kTorchXCheckInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)kTorchXCheckInput, &shape, &floatQ, NULL);

    float scales[8];
    for (size_t i = 0; i < 8; i++) {
        scales[i] = 1.f;
    }
    symQConfig_t outQC = {.scales = scales,
                          .numGroups = (size_t)kTorchXCheckNumGroups,
                          .groupSize = (size_t)kTorchXCheckGroupSize,
                          .roundingMode = HALF_AWAY,
                          .qBits = (uint8_t)kTorchXCheckQBits};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t symData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &symTensor);

    for (size_t g = 0; g < 8; g++) {
        TEST_ASSERT_EQUAL_FLOAT(kTorchXCheckScales[g], outQC.scales[g]);
    }
    int32_t codes[32];
    symTestUnpackSignExtend(symTensor.data, (size_t)kTorchXCheckQBits, codes, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kTorchXCheckCodes, codes, n);
}

void testSymGroupedToFloatDequantsPerGroup(void) {
    /* SYM(grouped) -> FLOAT32: build the packed tensor directly from the gold
     * codes+scales (no conversion involved on the way in) and check the
     * dequant matches the round-trip-stable float fixture (the SAME array
     * FLOAT32->SYM quantizes FROM in the sibling test above). */
    size_t n = kGroupQuantInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[3];
    memcpy(scales, kGroupQuantScales, sizeof(scales));
    symQConfig_t inQC = {.scales = scales,
                         .numGroups = (size_t)kGroupQuantNumGroups,
                         .groupSize = (size_t)kGroupQuantGroupSize,
                         .roundingMode = HALF_AWAY,
                         .qBits = (uint8_t)kGroupQuantQBits};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t symData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)kGroupQuantCodes, 32, symData, (size_t)kGroupQuantQBits, n);
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &inQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[12] = {0};
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&symTensor, &floatTensor);

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_FLOAT(kGroupQuantInput[i], floatData[i]);
    }
}

void testGroupedSymToSymInt32Dies(void) {
    /* SYM(grouped) -> SYM_INT32 via convertTensor must fail-fast, not
     * silently fold scales[0] onto the whole tensor (the pre-PR2 hazard: a
     * scalar-only cell reading only scales[0] would quietly misinterpret
     * every group but the first). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    symQConfig_t inQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t symData[calcNumberOfBytesForData(&inQ, n)];
    memset(symData, 0, sizeof(symData));
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &inQ, NULL);

    symInt32QConfig_t outQC = {0};
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[8] = {0};
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&symTensor, &outTensor));
}

void testGroupedSymInt32ToSymTargetDies(void) {
    /* SYM_INT32 -> SYM(grouped target): the no-rescale-into-groups hazard's
     * rescale-route sibling; a single dynamically-derived scale has nowhere
     * group-shaped to land. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC = {.scale = 0.5f, .qMaxBits = 16};
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    symQConfig_t outQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedSymToAsymDies(void) {
    /* SYM(grouped) -> ASYM: the affine-grid derivation reads scales[0] only
     * today; a grouped source must fail-fast rather than dequantize every
     * group but the first at the wrong scale. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    symQConfig_t inQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t symData[calcNumberOfBytesForData(&inQ, n)];
    memset(symData, 0, sizeof(symData));
    tensor_t symTensor;
    setTensorValues(&symTensor, symData, &shape, &inQ, NULL);

    float outScalesArr[1] = {1.f};
    uint16_t outZpsArr[1] = {0};
    asymQConfig_t outQC = {.scales = outScalesArr,
                           .zeroPoints = outZpsArr,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&symTensor, &outTensor));
}

void testGroupedAsymToSymTargetDies(void) {
    /* ASYM -> SYM(grouped target): the absmax-derivation cell writes
     * scales[0] only today; a grouped target must fail-fast. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScalesArr[1] = {0.5f};
    uint16_t inZpsArr[1] = {3};
    asymQConfig_t inQC = {.scales = inScalesArr,
                          .zeroPoints = inZpsArr,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    symQConfig_t outQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedInt32ToSymTargetDies(void) {
    /* INT32 -> SYM(grouped target): this cell packs raw codes with scale
     * fixed at 1.0 (scales[0] only) -- a grouped target must fail-fast. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t inQ;
    initInt32Quantization(&inQ);
    int32_t inData[] = {1, 2, 3, 4, 5, 6, 7, 8};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    symQConfig_t outQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedRepackSymInt32ToSymNoRescaleTargetDies(void) {
    /* repackSymInt32ToSymNoRescale into a grouped SYM target: a single
     * carried scale cannot fan out into per-group scales -- must fail-fast
     * (direct-call helper, not a conversionMatrix cell). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC = {.scale = 0.5f, .qMaxBits = 16};
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {1, -2, 3, -4, 5, -6, 7, -8};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    symQConfig_t outQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(repackSymInt32ToSymNoRescale(&inTensor, &outTensor));
}

void testDequantChunkToFloatRejectsGroupedSymSource(void) {
    /* dequantChunkToFloat's SYM cell is a grad-accumulate-only helper
     * (gradInit rejects grouped SYM templates); a grouped source reaching it
     * anyway must fail-fast rather than read scales[0] for every group. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    symQConfig_t qc = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t q;
    initSymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    memset(data, 0, sizeof(data));
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    float out[8];
    ASSERT_EXITS_WITH_FAILURE(dequantChunkToFloat(&t, 0, 8, out));
}

void testSameTypeSymCopyCarriesGroupArrays(void) {
    /* SYM(grouped) -> SYM(grouped) same-width same-type copy must carry the
     * FULL scales array + groupSize, not just scales[0] (the pre-PR2 hazard:
     * a partial copy would leave group 1's scale at the output's stale
     * default and drop groupSize entirely). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t mantissas[] = {3, -3, 31, -32, 10, -10, 20, -20};
    float inScales[2] = {0.25f, 0.5f};
    symQConfig_t inQC = {
        .scales = inScales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)mantissas, 32, inData, 6, n);
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outScales[2] = {1.f, 1.f};
    symQConfig_t outQC = {.scales = outScales,
                          .numGroups = 2,
                          .groupSize = 999, /* garbage; must be overwritten from src */
                          .roundingMode = HALF_AWAY,
                          .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    convertTensor(&in, &out);

    TEST_ASSERT_EQUAL_FLOAT(0.25f, outQC.scales[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scales[1]);
    TEST_ASSERT_EQUAL_size_t(4, outQC.groupSize);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(inData, outData, calcNumberOfBytesForData(&inQ, n));
}

void testSameTypeSymCopyRejectsGroupShapeMismatch(void) {
    /* SYM(grouped, numGroups=2) -> SYM(per-tensor, numGroups=1): dest's
     * 1-element scales array cannot hold src's 2 group scales -- must
     * fail-fast rather than overrun dest->scales. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScales[2] = {0.25f, 0.5f};
    symQConfig_t inQC = {
        .scales = inScales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outScales[1] = {1.f};
    symQConfig_t outQC = {
        .scales = outScales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&in, &out));
}

void testRequantizeTensorInPlaceGrouped(void) {
    /* FLOAT32 12-elem -> grouped SYM target via the PROMOTED public API
     * (requantizeTensorInPlace, TensorApi.h); then back to FLOAT32; compare
     * vs the round-trip-stable gold fixture. Exercises the promotion (the
     * two example-local copies are gone, this is the single source now) AND
     * the grouped conversion together. */
    size_t n = kGroupQuantInput_len;
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;

    quantization_t *floatQ = reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(floatQ);
    tensor_t *t = initTensor(shape, floatQ, NULL);
    tensorFillFromFloatBuffer(t, kGroupQuantInput, n);

    float targetScales[3] = {1.f, 1.f, 1.f};
    symQConfig_t targetQC = {.scales = targetScales,
                             .numGroups = (size_t)kGroupQuantNumGroups,
                             .groupSize = (size_t)kGroupQuantGroupSize,
                             .roundingMode = HALF_AWAY,
                             .qBits = (uint8_t)kGroupQuantQBits};
    quantization_t targetQ;
    initSymQuantization(&targetQC, &targetQ);

    requantizeTensorInPlace(t, &targetQ);

    TEST_ASSERT_EQUAL_INT32(SYM, t->quantization->type);
    symQConfig_t *gotQC = t->quantization->qConfig;
    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[0], gotQC->scales[0]);
    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[1], gotQC->scales[1]);
    TEST_ASSERT_EQUAL_FLOAT(kGroupQuantScales[2], gotQC->scales[2]);
    int32_t codes[12];
    symTestUnpackSignExtend(t->data, (size_t)kGroupQuantQBits, codes, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kGroupQuantCodes, codes, n);

    quantization_t backQ;
    initFloat32Quantization(&backQ);
    requantizeTensorInPlace(t, &backQ);

    TEST_ASSERT_EQUAL_INT32(FLOAT32, t->quantization->type);
    float *gotFloat = (float *)t->data;
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_FLOAT(kGroupQuantInput[i], gotFloat[i]);
    }

    freeTensor(t);
}

/* ---- Group-quant PR4 Task 2: the two PRIMARY ASYM cells (FLOAT32->ASYM,
 * ASYM->FLOAT32) are group-aware now; every OTHER ASYM cell keeps its
 * per-tensor fail-fast (scalar-only guards below). Gold fixtures:
 * expected_asym_nudged.h kAsymGrouped* / kAsymTorchXCheck* (per-group nudged
 * code-domain grids; generator asserts the zps are pairwise distinct and
 * every group scale differs from the whole-tensor derivation -- the collapse
 * discriminators for mutations (i)/(ii)). ---- */

void testFloatToAsymGroupedDerivesPerGroupGrids(void) {
    /* FLOAT32 -> ASYM with a grouped target derives PER-GROUP nudged grids
     * (group of element i = i / groupSize): per-group min/max -> scales[g]/
     * zeroPoints[g], run-based sequential emit against each group's grid. */
    size_t n = kAsymGroupedInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)kAsymGroupedInput, &shape, &floatQ, NULL);

    float scales[3] = {1.f, 1.f, 1.f};
    uint16_t zps[3] = {0, 0, 0};
    asymQConfig_t outQC = {.scales = scales,
                           .zeroPoints = zps,
                           .numGroups = (size_t)kAsymGroupedNumGroups,
                           .groupSize = (size_t)kAsymGroupedGroupSize,
                           .qBits = (uint8_t)kAsymGroupedQBits,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t asymData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    for (size_t g = 0; g < (size_t)kAsymGroupedNumGroups; g++) {
        TEST_ASSERT_EQUAL_FLOAT(kAsymGroupedScales[g], outQC.scales[g]);
        TEST_ASSERT_EQUAL_INT32(kAsymGroupedZps[g], (int32_t)outQC.zeroPoints[g]);
    }
    int32_t codes[12];
    byteConversion(asymTensor.data, (size_t)kAsymGroupedQBits, (uint8_t *)codes, 32, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kAsymGroupedCodes, codes, n);
}

void testFloatToAsymGroupedMatchesTorchPerChannel(void) {
    /* Cross-checked in the generator against torch.quantize_per_channel
     * (quint8, axis=0, per-out-channel groups of 4, OUR nudged grid passed
     * TO torch) over an 8x4 GEMM-weight-shaped fixture at qBits=8 -- see the
     * generator docstring for the zp-convention mapping. */
    size_t n = kAsymTorchXCheckInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)kAsymTorchXCheckInput, &shape, &floatQ, NULL);

    float scales[8];
    uint16_t zps[8];
    for (size_t g = 0; g < 8; g++) {
        scales[g] = 1.f;
        zps[g] = 0;
    }
    asymQConfig_t outQC = {.scales = scales,
                           .zeroPoints = zps,
                           .numGroups = (size_t)kAsymTorchXCheckNumGroups,
                           .groupSize = (size_t)kAsymTorchXCheckGroupSize,
                           .qBits = (uint8_t)kAsymTorchXCheckQBits,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t asymData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &outQ, NULL);

    convertTensor(&floatTensor, &asymTensor);

    for (size_t g = 0; g < 8; g++) {
        TEST_ASSERT_EQUAL_FLOAT(kAsymTorchXCheckScales[g], outQC.scales[g]);
        TEST_ASSERT_EQUAL_INT32(kAsymTorchXCheckZps[g], (int32_t)outQC.zeroPoints[g]);
    }
    int32_t codes[32];
    byteConversion(asymTensor.data, (size_t)kAsymTorchXCheckQBits, (uint8_t *)codes, 32, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kAsymTorchXCheckCodes, codes, n);
}

void testAsymGroupedToFloatDequantsPerGroup(void) {
    /* ASYM(grouped) -> FLOAT32: build the packed tensor directly from the
     * gold codes+grids (no conversion on the way in) and check the per-run
     * (code - zp[g]) * scales[g] dequant against the gold dequant array.
     * The generator's pairwise-distinct-zps assert makes a zeroPoints[0]
     * lookup here diverge on groups 1..k-1 (mutation (ii) guard). */
    size_t n = kAsymGroupedInput_len;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[3];
    uint16_t zps[3];
    for (size_t g = 0; g < (size_t)kAsymGroupedNumGroups; g++) {
        scales[g] = kAsymGroupedScales[g];
        zps[g] = (uint16_t)kAsymGroupedZps[g];
    }
    asymQConfig_t inQC = {.scales = scales,
                          .zeroPoints = zps,
                          .numGroups = (size_t)kAsymGroupedNumGroups,
                          .groupSize = (size_t)kAsymGroupedGroupSize,
                          .qBits = (uint8_t)kAsymGroupedQBits,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t asymData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)kAsymGroupedCodes, 32, asymData, (size_t)kAsymGroupedQBits, n);
    tensor_t asymTensor;
    setTensorValues(&asymTensor, asymData, &shape, &inQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[12] = {0};
    tensor_t floatTensor;
    setTensorValues(&floatTensor, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&asymTensor, &floatTensor);

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_FLOAT(kAsymGroupedDequant[i], floatData[i]);
    }
}

void testRequantizeTensorInPlaceGroupedAsym(void) {
    /* FLOAT32 12-elem -> grouped ASYM target built via the Task-1 public API
     * (quantizationInitAsymGrouped), through requantizeTensorInPlace; then
     * back to FLOAT32; compare vs the gold codes/grids/dequant. End-to-end
     * over BOTH new grouped cells plus the attach-time shape validation. */
    size_t n = kAsymGroupedInput_len;
    size_t *dims = reserveMemory(sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(sizeof(size_t));
    order[0] = 0;
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = 1;

    quantization_t *floatQ = reserveMemory(sizeof(quantization_t));
    initFloat32Quantization(floatQ);
    tensor_t *t = initTensor(shape, floatQ, NULL);
    tensorFillFromFloatBuffer(t, kAsymGroupedInput, n);

    quantization_t *targetQ =
        quantizationInitAsymGrouped((uint8_t)kAsymGroupedQBits, HALF_AWAY,
                                    (size_t)kAsymGroupedNumGroups, (size_t)kAsymGroupedGroupSize);
    requantizeTensorInPlace(t, targetQ);

    TEST_ASSERT_EQUAL_INT32(ASYM, t->quantization->type);
    asymQConfig_t *gotQC = t->quantization->qConfig;
    for (size_t g = 0; g < (size_t)kAsymGroupedNumGroups; g++) {
        TEST_ASSERT_EQUAL_FLOAT(kAsymGroupedScales[g], gotQC->scales[g]);
        TEST_ASSERT_EQUAL_INT32(kAsymGroupedZps[g], (int32_t)gotQC->zeroPoints[g]);
    }
    int32_t codes[12];
    byteConversion(t->data, (size_t)kAsymGroupedQBits, (uint8_t *)codes, 32, n);
    TEST_ASSERT_EQUAL_INT32_ARRAY(kAsymGroupedCodes, codes, n);

    quantization_t backQ;
    initFloat32Quantization(&backQ);
    requantizeTensorInPlace(t, &backQ);

    TEST_ASSERT_EQUAL_INT32(FLOAT32, t->quantization->type);
    float *gotFloat = (float *)t->data;
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_FLOAT(kAsymGroupedDequant[i], gotFloat[i]);
    }

    freeQuantization(targetQ);
    freeTensor(t);
}

/* ---- Scalar-only ASYM cells: their per-tensor fail-fasts REMAIN after the
 * two primary cells went grouped -- a grouped ASYM config reaching any of
 * them must die rather than compute on group 0's grid alone. Route through
 * FLOAT32 (the two grouped cells) instead. ---- */

void testGroupedAsymTargetFromInt32Dies(void) {
    /* INT32 -> ASYM(grouped target): the grid derivation is per-tensor only
     * (deriveAsymGridFromMinMax choke point). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    quantization_t inQ;
    initInt32Quantization(&inQ);
    int32_t inData[] = {1, 2, 3, 4, -1, -2, -3, -4};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    uint16_t zps[2] = {0, 0};
    asymQConfig_t outQC = {.scales = scales,
                           .zeroPoints = zps,
                           .numGroups = 2,
                           .groupSize = 4,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedAsymTargetFromSymInt32Dies(void) {
    /* SYM_INT32 -> ASYM(grouped target): same per-tensor-only grid
     * derivation choke point. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    symInt32QConfig_t inQC = {.scale = 0.5f, .qMaxBits = 16};
    quantization_t inQ;
    initSymInt32Quantization(&inQC, &inQ);
    int32_t inData[] = {1, 2, 3, 4, -1, -2, -3, -4};
    tensor_t inTensor;
    setTensorValues(&inTensor, (uint8_t *)inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    uint16_t zps[2] = {0, 0};
    asymQConfig_t outQC = {.scales = scales,
                           .zeroPoints = zps,
                           .numGroups = 2,
                           .groupSize = 4,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedAsymTargetFromSymDies(void) {
    /* SYM(per-tensor) -> ASYM(grouped target): the SYM source passes its own
     * gate; the ASYM-target side must still die at the grid derivation. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScales[1] = {0.5f};
    symQConfig_t inQC = {
        .scales = inScales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, inData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    uint16_t zps[2] = {0, 0};
    asymQConfig_t outQC = {.scales = scales,
                           .zeroPoints = zps,
                           .numGroups = 2,
                           .groupSize = 4,
                           .qBits = 5,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedAsymSourceToInt32Dies(void) {
    /* ASYM(grouped) -> INT32: the mantissa-image cell reads zeroPoints[0]
     * only -- must die on a grouped source. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t inQC = {.scales = scales,
                          .zeroPoints = zps,
                          .numGroups = 2,
                          .groupSize = 4,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, inData, &shape, &inQ, NULL);

    quantization_t outQ;
    initInt32Quantization(&outQ);
    int32_t outData[8];
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedAsymSourceToSymInt32Dies(void) {
    /* ASYM(grouped) -> SYM_INT32: reads scales[0]/zeroPoints[0] only -- must
     * die on a grouped source (mutation (iii) guard: dropping this cell's
     * fail-fast makes THIS test fail). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t inQC = {.scales = scales,
                          .zeroPoints = zps,
                          .numGroups = 2,
                          .groupSize = 4,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, inData, &shape, &inQ, NULL);

    symInt32QConfig_t outQC = {0};
    outQC.qMaxBits = 16;
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[8] = {0};
    tensor_t outTensor;
    setTensorValues(&outTensor, (uint8_t *)outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testGroupedAsymSourceToSymDies(void) {
    /* ASYM(grouped) -> SYM(per-tensor): the ASYM-source side of the rescale
     * cell decodes against scales[0]/zeroPoints[0] only -- must die. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t inQC = {.scales = scales,
                          .zeroPoints = zps,
                          .numGroups = 2,
                          .groupSize = 4,
                          .qBits = 5,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t inTensor;
    setTensorValues(&inTensor, inData, &shape, &inQ, NULL);

    float outScales[1] = {1.f};
    symQConfig_t outQC = {
        .scales = outScales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t outTensor;
    setTensorValues(&outTensor, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&inTensor, &outTensor));
}

void testDequantChunkToFloatRejectsGroupedAsymSource(void) {
    /* dequantChunkToFloat's ASYM arm is a grad-accumulate-only helper
     * (gradInit rejects grouped ASYM templates, the #300 carrier gate); a
     * grouped source reaching it anyway must fail-fast rather than decode
     * every group on group 0's grid -- the exact SYM-arm twin below it. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t qc = {.scales = scales,
                        .zeroPoints = zps,
                        .numGroups = 2,
                        .groupSize = 4,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    memset(data, 0, sizeof(data));
    tensor_t t;
    setTensorValues(&t, data, &shape, &q, NULL);

    float out[8];
    ASSERT_EXITS_WITH_FAILURE(dequantChunkToFloat(&t, 0, 8, out));
}

void testAccumulateFloatIntoAsymRescaleRejectsGroupedTarget(void) {
    /* accumulate-into-ASYM engine (float-increment entry): grads are
     * per-tensor unconditionally (gradInit carrier gate) -- a grouped target
     * is a caller contract violation and must die. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t qc = {.scales = scales,
                        .zeroPoints = zps,
                        .numGroups = 2,
                        .groupSize = 4,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    memset(data, 0, sizeof(data));
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    float inc[8] = {0};
    ASSERT_EXITS_WITH_FAILURE(accumulateFloatIntoAsymTensorRescale(&target, inc, n));
}

void testAccumulateTensorIntoAsymRescaleRejectsGroupedTarget(void) {
    /* accumulate-into-ASYM engine (tensor-increment entry): same carrier-gate
     * rationale as the float-increment twin above. */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float scales[2] = {0.1f, 0.2f};
    uint16_t zps[2] = {1, 2};
    asymQConfig_t qc = {.scales = scales,
                        .zeroPoints = zps,
                        .numGroups = 2,
                        .groupSize = 4,
                        .qBits = 5,
                        .roundingMode = HALF_AWAY};
    quantization_t q;
    initAsymQuantization(&qc, &q);
    uint8_t data[calcNumberOfBytesForData(&q, n)];
    memset(data, 0, sizeof(data));
    tensor_t target;
    setTensorValues(&target, data, &shape, &q, NULL);

    quantization_t incQ;
    initFloat32Quantization(&incQ);
    float incData[8] = {0};
    tensor_t inc;
    setTensorValues(&inc, (uint8_t *)incData, &shape, &incQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(accumulateTensorIntoAsymRescale(&target, &inc));
}

void testSameTypeAsymCopyCarriesGroupArrays(void) {
    /* ASYM(grouped) -> ASYM(grouped) same-width same-type copy must carry
     * BOTH full per-group arrays (scales AND zeroPoints) + groupSize -- the
     * exact ASYM twin of the SYM carry test above (Task-1 landed the copy;
     * this pins it). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t codes[] = {3, 60, 31, 0, 10, 53, 20, 43};
    float inScales[2] = {0.25f, 0.5f};
    uint16_t inZps[2] = {3, 9};
    asymQConfig_t inQC = {.scales = inScales,
                          .zeroPoints = inZps,
                          .numGroups = 2,
                          .groupSize = 4,
                          .qBits = 6,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)codes, 32, inData, 6, n);
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outScales[2] = {1.f, 1.f};
    uint16_t outZps[2] = {0, 0};
    asymQConfig_t outQC = {.scales = outScales,
                           .zeroPoints = outZps,
                           .numGroups = 2,
                           .groupSize = 999, /* garbage; must be overwritten from src */
                           .qBits = 6,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    convertTensor(&in, &out);

    TEST_ASSERT_EQUAL_FLOAT(0.25f, outQC.scales[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, outQC.scales[1]);
    TEST_ASSERT_EQUAL_UINT16(3, outQC.zeroPoints[0]);
    TEST_ASSERT_EQUAL_UINT16(9, outQC.zeroPoints[1]);
    TEST_ASSERT_EQUAL_size_t(4, outQC.groupSize);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(inData, outData, calcNumberOfBytesForData(&inQ, n));
}

void testSameTypeAsymCopyRejectsGroupShapeMismatch(void) {
    /* ASYM(grouped, numGroups=2) -> ASYM(per-tensor, numGroups=1): dest's
     * 1-element arrays cannot hold src's 2 group entries -- must fail-fast
     * rather than overrun dest->scales/zeroPoints (SYM twin above). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float inScales[2] = {0.25f, 0.5f};
    uint16_t inZps[2] = {3, 9};
    asymQConfig_t inQC = {.scales = inScales,
                          .zeroPoints = inZps,
                          .numGroups = 2,
                          .groupSize = 4,
                          .qBits = 6,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t inData[calcNumberOfBytesForData(&inQ, n)];
    memset(inData, 0, sizeof(inData));
    tensor_t in;
    setTensorValues(&in, inData, &shape, &inQ, NULL);

    float outScales[1] = {1.f};
    uint16_t outZps[1] = {0};
    asymQConfig_t outQC = {.scales = outScales,
                           .zeroPoints = outZps,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 6,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t out;
    setTensorValues(&out, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&in, &out));
}

void testFloatToBfpPerTensorBasic(void) {
    /* absMax 6.0, m=4 -> qMax 7, ratio 6/7=0.857 -> E=0, scale 1.0
     * values quantize to round(v): {6, -3, 1, 0} */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[4] = {6.f, -3.25f, 1.f, 0.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    /* sentinel != expected: the converter must WRITE the exponent */
    uint8_t exponents[1] = {0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    bfpQConfig_t *qc = dst.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(127, qc->exponents[0]); /* E=0 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 4, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(6, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-3, mant[1]); /* -3.25 HALF_AWAY -> -3 */
    TEST_ASSERT_EQUAL_INT32(1, mant[2]);
    TEST_ASSERT_EQUAL_INT32(0, mant[3]);
}

void testFloatToBfpSnapsUpAndPowerOfTwoIsExact(void) {
    /* absMax 8.0, m=4 -> qMax 7, ratio 8/7 in (1,2) -> E=1, scale 2.0: 8 -> mantissa 4 */
    /* also pins the snap-UP direction: E=0 would need |m|=8 > qMax */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[2] = {8.f, -2.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t exponents[1] = {0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    bfpQConfig_t *qc = dst.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(128, qc->exponents[0]); /* E=+1 */
    int32_t mant[2];
    unpackSignExtend(dst.data, 4, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(4, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-1, mant[1]);
}

void testFloatToBfpGroupedIndependentExponents(void) {
    /* groups of 4, m=4 (qMax 7): group0 absMax 6 -> E=0; group1 absMax 28 -> E=2 (28/7=4=2^2,
     * frac==0.5 branch: E = e-1 exercises the exact-power boundary) */
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[8] = {6.f, 1.f, -2.f, 0.f, 28.f, -7.f, 3.f, 14.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t exponents[2] = {0, 0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 2,
                          .groupSize = 4,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dstGrouped2x4;
    setTensorValues(&dstGrouped2x4, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dstGrouped2x4);

    bfpQConfig_t *qc = dstGrouped2x4.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(127, qc->exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(129, qc->exponents[1]); /* E=+2, scale 4 */
    int32_t mant[8];
    unpackSignExtend(dstGrouped2x4.data, 4, 0, mant, 8);
    TEST_ASSERT_EQUAL_INT32(7, mant[4]);  /* 28/4 */
    TEST_ASSERT_EQUAL_INT32(-2, mant[5]); /* -7/4 = -1.75 HALF_AWAY -> -2 */
}

void testFloatToBfpSaturatesNarrowExponentHigh(void) {
    /* e=2 -> bias 1, stored range [0,3], E range [-1, 2], max scale 4, m=4:
     * representable |v| <= 7*4 = 28. absMax 1000 needs E=8 -> clamps to stored 3 (E=2),
     * mantissas clamp to +-7 (D6 saturation, no abort, no pack-guard trip). */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[2] = {1000.f, -5.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t exponents[1] = {0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 2};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dstE2;
    setTensorValues(&dstE2, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dstE2);

    bfpQConfig_t *qc = dstE2.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(3, qc->exponents[0]);
    int32_t mant[2];
    unpackSignExtend(dstE2.data, 4, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(7, mant[0]);  /* saturated */
    TEST_ASSERT_EQUAL_INT32(-1, mant[1]); /* -5/4 -> -1.25 -> -1 */
}

void testFloatToBfpFlushesNarrowExponentLow(void) {
    /* e=2 -> E range [-1,2], min scale 0.5. absMax 0.001 needs E<<-1 -> clamps to stored 0
     * (E=-1): 0.001/0.5 rounds to 0 -> flush toward zero. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[2] = {0.001f, -0.0005f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    /* sentinel at the OTHER end of the stored range: expected result is 0 */
    uint8_t exponents[1] = {3};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 2};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dstE2;
    setTensorValues(&dstE2, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dstE2);

    bfpQConfig_t *qc = dstE2.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(0, qc->exponents[0]);
    int32_t mant[2];
    unpackSignExtend(dstE2.data, 4, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(0, mant[0]);
    TEST_ASSERT_EQUAL_INT32(0, mant[1]);
}

void testFloatToBfpAllZeroGroupKeepsZeroStateExponent(void) {
    /* absMax == 0 -> stored = bias (scale 1), all-zero mantissas (SYM scale-1 parity) */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[4] = {0.f, 0.f, 0.f, 0.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t exponents[1] = {0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    bfpQConfig_t *qc = dst.quantization->qConfig;
    TEST_ASSERT_EQUAL_UINT8(127, qc->exponents[0]); /* = bias, scale 1 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 4, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(0, mant[0]);
    TEST_ASSERT_EQUAL_INT32(0, mant[1]);
    TEST_ASSERT_EQUAL_INT32(0, mant[2]);
    TEST_ASSERT_EQUAL_INT32(0, mant[3]);
}

void testBfpToFloatRoundTripExactForRepresentable(void) {
    /* mantissa * 2^E values round-trip exactly: pack {6,-3,1,0} then convert back */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[4] = {6.f, -3.f, 1.f, 0.f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t exponents[1] = {0};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dstBfp;
    setTensorValues(&dstBfp, bfpData, &shape, &bfpQ, NULL);

    quantization_t backQ;
    initFloat32Quantization(&backQ);
    float backData[4] = {0};
    tensor_t backToFloat;
    setTensorValues(&backToFloat, (uint8_t *)backData, &shape, &backQ, NULL);

    convertTensor(&src, &dstBfp);
    convertTensor(&dstBfp, &backToFloat);

    TEST_ASSERT_EQUAL_FLOAT(6.f, ((float *)backToFloat.data)[0]);
    TEST_ASSERT_EQUAL_FLOAT(-3.f, ((float *)backToFloat.data)[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.f, ((float *)backToFloat.data)[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.f, ((float *)backToFloat.data)[3]);
}

void testBfpGroupedToFloatDequantsPerGroup(void) {
    /* BFP(grouped) -> FLOAT32: packed tensor built directly from gold
     * mantissas+exponents (no conversion on the way in, mirroring
     * testSymGroupedToFloatDequantsPerGroup); dequant must apply each
     * group's OWN 2^E scale (group0 E=0 scale 1, group1 E=+2 scale 4). */
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[8] = {6, 1, -2, 0, 7, -2, 1, 4};
    uint8_t exponents[2] = {127, 129};
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 2,
                         .groupSize = 4,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float floatData[8] = {0};
    tensor_t dst;
    setTensorValues(&dst, (uint8_t *)floatData, &shape, &floatQ, NULL);

    convertTensor(&src, &dst);

    float expected[8] = {6.f, 1.f, -2.f, 0.f, 28.f, -8.f, 4.f, 16.f};
    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_EQUAL_FLOAT(expected[i], floatData[i]);
    }
}

void testFloatToBfpSrRoundingIsSeededDeterministic(void) {
    /* Same fixture through HALF_AWAY vs seeded SR_HALF_AWAY: the exponent
     * derivation is rounding-free (absMax 14 -> ratio 2 -> frac==0.5 -> E=+1,
     * stored 128, identical in every run); the SR mantissa stream must be
     * deterministic under rngSetSeed(42) and differ from the HALF_AWAY result
     * somewhere (7 of 8 quotients are fractional at scale 2). */
    size_t n = 8;
    size_t dims[] = {8};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    float vals[8] = {14.f, 3.5f, -1.75f, 2.625f, 7.f, -10.5f, 0.875f, 5.25f};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)vals, &shape, &floatQ, NULL);

    uint8_t refExponents[1] = {0};
    bfpQConfig_t refQC = {.exponents = refExponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t refQ;
    initBfpQuantization(&refQC, &refQ);
    uint8_t refData[calcNumberOfBytesForData(&refQ, n)];
    tensor_t refDst;
    setTensorValues(&refDst, refData, &shape, &refQ, NULL);
    convertTensor(&src, &refDst);
    int32_t refMant[8];
    unpackSignExtend(refDst.data, 4, 0, refMant, 8);

    uint8_t srExponents[1] = {0};
    bfpQConfig_t srQC = {.exponents = srExponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = SR_HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t srQ;
    initBfpQuantization(&srQC, &srQ);
    uint8_t srData[calcNumberOfBytesForData(&srQ, n)];
    tensor_t srDst;
    setTensorValues(&srDst, srData, &shape, &srQ, NULL);

    rngSetSeed(42);
    convertTensor(&src, &srDst);
    int32_t srMant1[8];
    unpackSignExtend(srDst.data, 4, 0, srMant1, 8);

    rngSetSeed(42);
    convertTensor(&src, &srDst);
    int32_t srMant2[8];
    unpackSignExtend(srDst.data, 4, 0, srMant2, 8);

    TEST_ASSERT_EQUAL_UINT8(128, refQC.exponents[0]);
    TEST_ASSERT_EQUAL_UINT8(128, srQC.exponents[0]);
    TEST_ASSERT_EQUAL_INT32_ARRAY(srMant1, srMant2, 8);
    bool anyDiffers = false;
    for (size_t i = 0; i < n; i++) {
        if (srMant1[i] != refMant[i]) {
            anyDiffers = true;
        }
    }
    TEST_ASSERT_TRUE(anyDiffers);
}

void testInt32ToBfpIsCodesInWithZeroStateExponent(void) {
    /* INT32 -> BFP is codes-in (#227 code domain): int32 {5, -3} fits the m=4
     * code range [-8, 7] verbatim. The exponent is WRITTEN to the zero state
     * (= bias 127 for e=8, scale 1.0), mirroring convertInt32TensorToSymTensor's
     * scales[0] = 1.f reset -- sentinel 9 proves the converter wrote it. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t intData[2] = {5, -3};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)intData, &shape, &intQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 127 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(127, outQC.exponents[0]); /* bias: E=0, scale 1 */
    int32_t mant[2];
    unpackSignExtend(dst.data, 4, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(5, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-3, mant[1]);
}

void testInt32ToBfpOverflowAborts(void) {
    /* 200 > 7 = m=4 code max: codes-in must ABORT via packChunkGuarded (#227
     * code-domain discipline) -- D6 saturation covers value-domain quantization
     * only, never raw code stuffing. */
    size_t n = 1;
    size_t dims[] = {1};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t intData[1] = {200};
    quantization_t intQ;
    initInt32Quantization(&intQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)intData, &shape, &intQ, NULL);

    uint8_t exponents[1] = {9};
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&src, &dst));
}

void testBfpToInt32DropsExponent(void) {
    /* int_repr: mantissas {6, -3} sign-extend to int32 verbatim; the E=+2
     * exponent (stored 129) is DROPPED, not applied (dequantized values would
     * be {24, -12}). Sentinel -777 proves the converter wrote the payload. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[2] = {6, -3};
    uint8_t exponents[1] = {129}; /* E=+2, scale 4 */
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    quantization_t intQ;
    initInt32Quantization(&intQ);
    int32_t outData[2] = {-777, -777}; /* sentinel != expected */
    tensor_t dst;
    setTensorValues(&dst, (uint8_t *)outData, &shape, &intQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_INT32(6, ((int32_t *)dst.data)[0]);
    TEST_ASSERT_EQUAL_INT32(-3, ((int32_t *)dst.data)[1]);
}

void testSymInt32ToBfpPreservesValues(void) {
    /* value-preserving: mantissas {10, -4} * scale 0.5 = values {5, -2}.
     * BFP m=4 -> qMax 7; absMax 5 -> ratio 5/7 in (0.5, 1) -> E=0 (stored
     * 127, scale 1); mantissas round(5/1)=5, round(-2/1)=-2 -- NOT the raw
     * input mantissas {10, -4}. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t symData[2] = {10, -4};
    symInt32QConfig_t inQC = {.scale = 0.5f, .roundingMode = HALF_AWAY, .qMaxBits = 12};
    quantization_t symQ;
    initSymInt32Quantization(&inQC, &symQ);
    tensor_t src;
    setTensorValues(&src, (uint8_t *)symData, &shape, &symQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 127 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(127, outQC.exponents[0]); /* E=0, scale 1 */
    int32_t mant[2];
    unpackSignExtend(dst.data, 4, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(5, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-2, mant[1]);
}

void testBfpToSymInt32FreshAbsmaxGrid(void) {
    /* mantissas {6, -3} @ E=+1 (stored 128, scale 2) = values {12, -6};
     * target qMaxBits=12 -> qMax 2047, fresh grid: scale = 12/2047.
     * codes = roundByMode(v/scale, target HALF_AWAY), float32-exact:
     *   12.f / (12.f/2047.f)  = 2047.0f  -> 2047
     *   -6.f / (12.f/2047.f)  = -1023.5f (exact float tie) -> -1024. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[2] = {6, -3};
    uint8_t exponents[1] = {128}; /* E=+1, scale 2 */
    /* SOURCE roundingMode deliberately SR: dequant is rounding-free, so the
     * converter must never read it -- rounding with the source's mode instead
     * of the TARGET's HALF_AWAY would turn the exact -1023.5 tie (and the
     * exact asserts below) stochastic. */
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = SR_HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    /* scale sentinel -1: the converter must derive and write the fresh grid */
    symInt32QConfig_t outQC = {.scale = -1.f, .roundingMode = HALF_AWAY, .qMaxBits = 12};
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[2] = {-777, -777}; /* sentinel != expected */
    tensor_t dst;
    setTensorValues(&dst, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_FLOAT(12.f / 2047.f, outQC.scale);
    TEST_ASSERT_EQUAL_INT32(2047, ((int32_t *)dst.data)[0]);
    TEST_ASSERT_EQUAL_INT32(-1024, ((int32_t *)dst.data)[1]);
}

void testBfpGroupedToSymInt32UsesPerGroupScales(void) {
    /* grouped source dequants per group BEFORE the fresh scalar grid:
     * group0 E=+2 (stored 129, scale 4) mantissas {6, -5} = values {24, -20};
     * group1 E=0  (stored 127, scale 1) mantissas {2, -3} = values {2, -3}.
     * absMax 24, qMaxBits=8 -> qMax 127, scale = 24/127. codes (float32-checked,
     * no half ties): 24 -> 127; -20 -> -105.8333 -> -106; 2 -> 10.5833 -> 11;
     * -3 -> -15.875 -> -16. A whole-tensor scale-4 misread would give {2, -3}
     * the codes of {8, -12} instead. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[4] = {6, -5, 2, -3};
    uint8_t exponents[2] = {129, 127};
    /* SOURCE roundingMode SR for the same never-read reason as
     * testBfpToSymInt32FreshAbsmaxGrid (three fractional quotients here). */
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 2,
                         .groupSize = 2,
                         .roundingMode = SR_HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    symInt32QConfig_t outQC = {.scale = -1.f, .roundingMode = HALF_AWAY, .qMaxBits = 8};
    quantization_t outQ;
    initSymInt32Quantization(&outQC, &outQ);
    int32_t outData[4] = {-777, -777, -777, -777};
    tensor_t dst;
    setTensorValues(&dst, (uint8_t *)outData, &shape, &outQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_FLOAT(24.f / 127.f, outQC.scale);
    TEST_ASSERT_EQUAL_INT32(127, ((int32_t *)dst.data)[0]);
    TEST_ASSERT_EQUAL_INT32(-106, ((int32_t *)dst.data)[1]);
    TEST_ASSERT_EQUAL_INT32(11, ((int32_t *)dst.data)[2]);
    TEST_ASSERT_EQUAL_INT32(-16, ((int32_t *)dst.data)[3]);
}

void testSymPerTensorToBfpPreservesValues(void) {
    /* value-preserving: SYM qBits=4 mantissas {6, -3, 1, 0} * scale 0.5 =
     * values {3, -1.5, 0.5, 0}. BFP m=6 -> qMax 31; absMax 3 -> ratio 3/31 in
     * (2^-4, 2^-3) -> E=-3 (stored 124, scale 0.125); mantissas
     * {24, -12, 4, 0} -- exact ints, NOT the raw input mantissas. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[4] = {6, -3, 1, 0};
    float scales[1] = {0.5f};
    symQConfig_t inQC = {
        .scales = scales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 4};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t symData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, symData, 4, n);
    tensor_t src;
    setTensorValues(&src, symData, &shape, &inQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 124 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 6,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(124, outQC.exponents[0]); /* E=-3, scale 0.125 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 6, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(24, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-12, mant[1]);
    TEST_ASSERT_EQUAL_INT32(4, mant[2]);
    TEST_ASSERT_EQUAL_INT32(0, mant[3]);
}

void testSymGroupedSourceToBfpUsesPerGroupScales(void) {
    /* grouped SYM source (2 groups, DIFFERENT scales) -> per-tensor BFP:
     * group0 mantissas {6, -3} * 0.5 = {3, -1.5}; group1 {5, -2} * 2 =
     * {10, -4}. absMax 10 over BOTH grids -> m=6 (qMax 31) ratio 10/31 ->
     * E=-1 (stored 126, scale 0.5); mantissas {6, -3, 20, -8}. A
     * scales[0]-only misread dequants group1 as {2.5, -1} -> absMax 3,
     * stored 124, group0 codes {24, -12} -- exponent AND codes both
     * discriminate, in EITHER pass alone. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[4] = {6, -3, 5, -2};
    float scales[2] = {0.5f, 2.f};
    symQConfig_t inQC = {
        .scales = scales, .numGroups = 2, .groupSize = 2, .roundingMode = HALF_AWAY, .qBits = 4};
    quantization_t inQ;
    initSymQuantization(&inQC, &inQ);
    uint8_t symData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, symData, 4, n);
    tensor_t src;
    setTensorValues(&src, symData, &shape, &inQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 126 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 6,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(126, outQC.exponents[0]); /* E=-1, scale 0.5 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 6, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(6, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-3, mant[1]);
    TEST_ASSERT_EQUAL_INT32(20, mant[2]);
    TEST_ASSERT_EQUAL_INT32(-8, mant[3]);
}

void testBfpToSymPerTensorFreshAbsmax(void) {
    /* mantissas {6, -3} @ E=+1 (stored 128, scale 2) = values {12, -6};
     * target qBits=3 -> qMax 3, fresh grid: scales[0] = 12/3 = 4 (exact).
     * codes = roundByMode(v/4, target HALF_AWAY): 12 -> 3; -6 -> -1.5 -> -2. */
    size_t n = 2;
    size_t dims[] = {2};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[2] = {6, -3};
    uint8_t exponents[1] = {128}; /* E=+1, scale 2 */
    /* SOURCE roundingMode deliberately SR: dequant is rounding-free, so the
     * converter must never read it -- the fractional -1.5 quotient would turn
     * stochastic under the source's mode (testBfpToSymInt32FreshAbsmaxGrid
     * precedent). */
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = SR_HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    /* scale sentinel -1: the converter must derive and write the fresh grid */
    float outScales[1] = {-1.f};
    symQConfig_t outQC = {
        .scales = outScales, .numGroups = 1, .groupSize = 0, .roundingMode = HALF_AWAY, .qBits = 3};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_FLOAT(4.f, outQC.scales[0]);
    int32_t mant[2];
    unpackSignExtend(dst.data, 3, 0, mant, 2);
    TEST_ASSERT_EQUAL_INT32(3, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-2, mant[1]);
}

void testBfpToSymGroupedTargetDenies(void) {
    /* BFP -> SYM(grouped target): the absmax-derivation cell writes scales[0]
     * only -- a grouped target must fail-fast before any qconfig/payload
     * writes (mirrors testGroupedAsymToSymTargetDies). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    uint8_t exponents[1] = {127};
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    memset(bfpData, 0, sizeof(bfpData));
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    float scales[2] = {1.f, 1.f};
    symQConfig_t outQC = {
        .scales = scales, .numGroups = 2, .groupSize = 4, .roundingMode = HALF_AWAY, .qBits = 6};
    quantization_t outQ;
    initSymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&src, &dst));
}

void testAsymToBfpPreservesValues(void) {
    /* value-preserving under the PR4 code-domain decode (code - zp)*scale:
     * ASYM qBits=4 codes {15, 1, 7, 0}, zeroPoints[0]=7, scales[0]=0.5 ->
     * dequant {(15-7)*0.5, (1-7)*0.5, (7-7)*0.5, (0-7)*0.5} =
     * {4, -3, 0, -3.5}. BFP m=4 -> qMax 7; absMax 4 -> ratio 4/7 in (0.5, 1)
     * -> E=0 (stored 127, scale 1); mantissas {4, -3, 0, -4} (-3.5 HALF_AWAY
     * -> -4). Code 15 pins the zero-extended unsigned read: a sign-extending
     * misread (-1) dequants to (-1-7)*0.5 = -4 and flips mantissa[0] to -4.
     * The stale value-domain decode (code + zp)*scale is also killed: it
     * gives {11, 4, 7, 3.5} -> absMax 11 -> stored 128, not 127. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldCodes[4] = {15, 1, 7, 0};
    float inScales[1] = {0.5f};
    uint16_t inZps[1] = {7};
    asymQConfig_t inQC = {.scales = inScales,
                          .zeroPoints = inZps,
                          .numGroups = 1,
                          .groupSize = 0,
                          .qBits = 4,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t asymData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldCodes, 32, asymData, 4, n);
    tensor_t src;
    setTensorValues(&src, asymData, &shape, &inQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 127 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 4,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(127, outQC.exponents[0]); /* E=0, scale 1 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 4, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(4, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-3, mant[1]);
    TEST_ASSERT_EQUAL_INT32(0, mant[2]);
    TEST_ASSERT_EQUAL_INT32(-4, mant[3]);
}

void testAsymGroupedSourceToBfpUsesPerGroupGrids(void) {
    /* grouped ASYM source (2 groups, DIFFERENT scales AND zeroPoints) ->
     * per-tensor BFP: group0 codes {15, 1} @ (scale 0.5, zp 7) ->
     * {(15-7)*0.5, (1-7)*0.5} = {4, -3}; group1 codes {8, 1} @ (scale 2,
     * zp 3) -> {(8-3)*2, (1-3)*2} = {10, -4}. absMax 10 over BOTH grids ->
     * m=6 (qMax 31) ratio 10/31 -> E=-1 (stored 126, scale 0.5); mantissas
     * {8, -6, 20, -8}. A grid[0]-only misread dequants group1 as
     * {0.5, -3} -> absMax 4, stored 125, pass-2 codes {8, -6, 1, -6} --
     * exponent AND codes both discriminate, in EITHER pass alone. A
     * zeroPoints[0]-only misread (scales correct) keeps the exponent
     * (absMax 12 -> still E=-1) but shifts group1 codes to {4, -24}. */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldCodes[4] = {15, 1, 8, 1};
    float inScales[2] = {0.5f, 2.f};
    uint16_t inZps[2] = {7, 3};
    asymQConfig_t inQC = {.scales = inScales,
                          .zeroPoints = inZps,
                          .numGroups = 2,
                          .groupSize = 2,
                          .qBits = 4,
                          .roundingMode = HALF_AWAY};
    quantization_t inQ;
    initAsymQuantization(&inQC, &inQ);
    uint8_t asymData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldCodes, 32, asymData, 4, n);
    tensor_t src;
    setTensorValues(&src, asymData, &shape, &inQ, NULL);

    uint8_t exponents[1] = {9}; /* sentinel != expected 126 */
    bfpQConfig_t outQC = {.exponents = exponents,
                          .numGroups = 1,
                          .groupSize = 0,
                          .roundingMode = HALF_AWAY,
                          .mantissaBits = 6,
                          .exponentBits = 8};
    quantization_t bfpQ;
    initBfpQuantization(&outQC, &bfpQ);
    uint8_t bfpData[calcNumberOfBytesForData(&bfpQ, n)];
    tensor_t dst;
    setTensorValues(&dst, bfpData, &shape, &bfpQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_UINT8(126, outQC.exponents[0]); /* E=-1, scale 0.5 */
    int32_t mant[4];
    unpackSignExtend(dst.data, 6, 0, mant, 4);
    TEST_ASSERT_EQUAL_INT32(8, mant[0]);
    TEST_ASSERT_EQUAL_INT32(-6, mant[1]);
    TEST_ASSERT_EQUAL_INT32(20, mant[2]);
    TEST_ASSERT_EQUAL_INT32(-8, mant[3]);
}

void testBfpToAsymUsesCanonicalGrid(void) {
    /* mantissas {-1, 0, 1, 3} @ E=+2 (stored 129, scale 4) = values
     * {-4, 0, 4, 12} -- min/max run over DEQUANTIZED values incl. the
     * negative -4. Canonical grid (deriveAsymGridFromMinMax, PR4 D6 nudged
     * code-domain form; the band already contains 0, so the nudge is inert)
     * at qBits=4 (qMax 15): scale = (12 - -4)/15 = 16/15, zpReal =
     * -mn/scale = 4/(16/15) = 3.74999976f (genuinely fractional: the float
     * quotient does NOT snap to an integer) -> roundByMode HALF_AWAY -> 4;
     * a truncating hand grid gives 3 instead. emitAsymChunk codes =
     * clamp(round(v/scale) + zp, 0, 15): quotients {-3.74999976, 0,
     * 3.74999976, 11.2499993} round to {-4, 0, 4, 11} -> +4 ->
     * {0, 4, 8, 15} (quotients sit ~0.25 from the next integer -- no ties,
     * rounding-robust). */
    size_t n = 4;
    size_t dims[] = {4};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    int32_t goldMant[4] = {-1, 0, 1, 3};
    uint8_t exponents[1] = {129}; /* E=+2, scale 4 */
    /* SOURCE roundingMode SR for the never-read reason of
     * testBfpToSymInt32FreshAbsmaxGrid (dequant is rounding-free). */
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = SR_HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    byteConversion((uint8_t *)goldMant, 32, bfpData, 4, n);
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    /* sentinels: the converter must derive and write the fresh affine grid */
    float outScales[1] = {-1.f};
    uint16_t outZps[1] = {777};
    asymQConfig_t outQC = {.scales = outScales,
                           .zeroPoints = outZps,
                           .numGroups = 1,
                           .groupSize = 0,
                           .qBits = 4,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    convertTensor(&src, &dst);

    TEST_ASSERT_EQUAL_FLOAT(16.f / 15.f, outQC.scales[0]);
    TEST_ASSERT_EQUAL_UINT16(4, outQC.zeroPoints[0]);
    int32_t codes[4];
    byteConversion(dst.data, 4, (uint8_t *)codes, 32, n);
    TEST_ASSERT_EQUAL_INT32(0, codes[0]);
    TEST_ASSERT_EQUAL_INT32(4, codes[1]);
    TEST_ASSERT_EQUAL_INT32(8, codes[2]);
    TEST_ASSERT_EQUAL_INT32(15, codes[3]);
}

void testBfpToAsymGroupedTargetDenies(void) {
    /* BFP -> ASYM(grouped target): the cell derives ONE per-tensor grid
     * (deriveAsymGridFromMinMax writes scales[0]/zeroPoints[0]) -- a grouped
     * target must fail-fast before any qconfig/payload writes
     * (requirePerTensorAsym inside the grid helper; mirrors
     * testBfpToSymGroupedTargetDenies and the sibling *ToAsymTensor cells). */
    size_t n = 8;
    size_t dims[] = {n};
    size_t order[] = {0};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};

    uint8_t exponents[1] = {127};
    bfpQConfig_t inQC = {.exponents = exponents,
                         .numGroups = 1,
                         .groupSize = 0,
                         .roundingMode = HALF_AWAY,
                         .mantissaBits = 4,
                         .exponentBits = 8};
    quantization_t inQ;
    initBfpQuantization(&inQC, &inQ);
    uint8_t bfpData[calcNumberOfBytesForData(&inQ, n)];
    memset(bfpData, 0, sizeof(bfpData));
    tensor_t src;
    setTensorValues(&src, bfpData, &shape, &inQ, NULL);

    float outScales[2] = {1.f, 1.f};
    uint16_t outZps[2] = {0, 0};
    asymQConfig_t outQC = {.scales = outScales,
                           .zeroPoints = outZps,
                           .numGroups = 2,
                           .groupSize = 4,
                           .qBits = 6,
                           .roundingMode = HALF_AWAY};
    quantization_t outQ;
    initAsymQuantization(&outQC, &outQ);
    uint8_t outData[calcNumberOfBytesForData(&outQ, n)];
    tensor_t dst;
    setTensorValues(&dst, outData, &shape, &outQ, NULL);

    ASSERT_EXITS_WITH_FAILURE(convertTensor(&src, &dst));
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(testZeroTensorDataSymSubByteZeroesOnlyPackedBytes);
    RUN_TEST(testConversionIntFloat);
    RUN_TEST(testConversionIntSymInt32);
    RUN_TEST(testConversionIntAsym);

    RUN_TEST(testConversionFloatInt);
    RUN_TEST(testConversionFloatSymInt32);
    RUN_TEST(testConversionFloatAsym);
    RUN_TEST(testConversionFloatAsymQBits16NegativeBandZeroPointAtCodeCeiling);
    RUN_TEST(testInitAsymQConfigRejectsQBitsAbove16);
    RUN_TEST(testConversionFloatAsymFarNegativeBandDerivesNudgedGrid);
    RUN_TEST(testConversionFloatAsymEncodeClampAtBandEdgeTie);

    RUN_TEST(testConversionSymInt32Int);
    RUN_TEST(testConversionSymInt32Float);
    RUN_TEST(testConversionSymInt32Asym);
    RUN_TEST(testConversionSymInt32AsymConstantTensorNoDivByZero);

    RUN_TEST(testConversionAsymInt);
    RUN_TEST(testConversionAsymFloat);
    RUN_TEST(testConversionAsymSymInt32);
    RUN_TEST(testRequantDynamicAccumulatorRangeMatchesGold);
    RUN_TEST(testRequantDynamicAbsmaxZeroGivesZerosScaleOne);
    RUN_TEST(testRequantDynamicScaleTracksInputRescale);
    RUN_TEST(testRequantDynamicTieRoundsHalfAwayFromZero);
    RUN_TEST(testRequantDynamicInPlaceAliasMatchesGold);
    RUN_TEST(testRequantDynamicViaConversionMatrixDiagonal);
    RUN_TEST(testConvertTensorSymInt32SameTypeKeepsCopySemantics);
    RUN_TEST(testRequantToScaleNonSaturatingMatchesGold);
    RUN_TEST(testRequantToScaleSaturatesAtQMinQMax);
    RUN_TEST(testRequantToScaleSharedBufferAliasMatchesGold);
    RUN_TEST(testConversionBoolBoolCopiesOnlyPackedBytes);
    RUN_TEST(testConversionSymInt32SameTypeCopyPropagatesScale);
    RUN_TEST(testConversionSymSameTypeCopyPropagatesScale);
    RUN_TEST(testConversionSymSameTypeWidthMismatchDies);
    RUN_TEST(testConversionAsymSameTypeWidthMismatchDies);
    RUN_TEST(testQuantTypeToStringBool);
    RUN_TEST(testConversionSymSymInt32SignExtends);
    RUN_TEST(testConversionSymFloat32Dequantizes);
    RUN_TEST(testConversionSymInt32CodesDropScale);
    RUN_TEST(testConversionSymAsymRescaleRoundTrips);
    RUN_TEST(testConversionSymInt32ToSymRescaleRoundTrips);
    RUN_TEST(testRepackSymInt32ToSymNoRescaleFittingCarriesScale);
    RUN_TEST(testRepackSymInt32ToSymNoRescaleRejectsOverflow);
    RUN_TEST(testConversionFloatToSymRoundTripsSymmetric);
    RUN_TEST(testConversionInt32ToSymNoRescaleScale1);
    RUN_TEST(testConversionInt32ToSymRejectsOutOfRange);
    RUN_TEST(testChunkedFloatToSymRoundTripsAtChunkBoundary);
    RUN_TEST(testConversionAsymToSymRescaleOffCenterRoundTrips);
    RUN_TEST(testConvertSymToInt32RejectsZeroQBits);
    RUN_TEST(testConvertInt32ToSymRejectsZeroQBits);
    RUN_TEST(testConvertersPreserveCallerOutputShape);

    RUN_TEST(testAccumulateSymFixedGridFirstStoreDerivesGridThenCarries);
    RUN_TEST(testAccumulateSymFixedGridZeroIncrementIsBitExact);
    RUN_TEST(testAccumulateSymFixedGridOverflowAborts);
    RUN_TEST(testAccumulateSymRescaleRederivesGridEachCall);
    RUN_TEST(testAccumulateAsymRescaleMatchesFloatReference);
    RUN_TEST(testAccumulateAsymValueZeroAfterConfigReset);
    RUN_TEST(testAccumulateSymFixedGridMatchesReferenceAtChunkBoundary);
    RUN_TEST(testAccumulateSymRescaleMatchesReferenceAtChunkBoundary);
    RUN_TEST(testAccumulateTensorIntoSymRescaleStreamsIncrementAcrossChunks);

    RUN_TEST(testChunkedFloatToAsymMatchesReferenceAcrossChunkBoundaries);
    RUN_TEST(testChunkedSymInt32ToSymRoundTripsAtChunkBoundary);
    RUN_TEST(testChunkedAsymToSymRoundTripsAtChunkBoundary);
    RUN_TEST(testChunkedSymToAsymRoundTripsAtChunkBoundary);
    RUN_TEST(testChunkedSymInt32ToAsymRoundTripsAtChunkBoundary);
    RUN_TEST(testChunkedSymToFloat32DequantizesAtChunkBoundary);
    RUN_TEST(testChunkedAsymToFloat32DequantizesAtChunkBoundary);
    RUN_TEST(testDequantChunkToFloatFloat32MatchesSourceAtOffsets);
    RUN_TEST(testDequantChunkToFloatSymInt32MatchesScaleAtOffsets);
    RUN_TEST(testDequantChunkToFloatSymUnpacksSignExtendedAtOffsets);
    RUN_TEST(testDequantChunkToFloatAsymUnpacksZeroExtendedAtOffsets);
    RUN_TEST(testDequantChunkToFloatRejectsCountAboveChunk);
    RUN_TEST(testDequantChunkToFloatRejectsMisalignedOffset);
    RUN_TEST(testDequantChunkToFloatRejectsOutOfRangeOffset);
    RUN_TEST(testUnpackSignExtendReadsSignedCodeAtBitOffset);
    RUN_TEST(testUnpackSignExtendOffsetZeroCoversFullSignedRange);
    RUN_TEST(testQuantizeFloatToAsymNoOpOnEmptyTensor);
    RUN_TEST(testAccumulateTensorIntoSymRescaleRejectsSelfAliasedIncrement);

    RUN_TEST(testFloatToSymGroupedDerivesPerGroupScales);
    RUN_TEST(testFloatToSymGroupedMatchesTorchPerChannel);
    RUN_TEST(testSymGroupedToFloatDequantsPerGroup);
    RUN_TEST(testGroupedSymToSymInt32Dies);
    RUN_TEST(testGroupedSymInt32ToSymTargetDies);
    RUN_TEST(testGroupedSymToAsymDies);
    RUN_TEST(testGroupedAsymToSymTargetDies);
    RUN_TEST(testGroupedInt32ToSymTargetDies);
    RUN_TEST(testGroupedRepackSymInt32ToSymNoRescaleTargetDies);
    RUN_TEST(testDequantChunkToFloatRejectsGroupedSymSource);
    RUN_TEST(testSameTypeSymCopyCarriesGroupArrays);
    RUN_TEST(testSameTypeSymCopyRejectsGroupShapeMismatch);
    RUN_TEST(testRequantizeTensorInPlaceGrouped);
    RUN_TEST(testFloatToAsymGroupedDerivesPerGroupGrids);
    RUN_TEST(testFloatToAsymGroupedMatchesTorchPerChannel);
    RUN_TEST(testAsymGroupedToFloatDequantsPerGroup);
    RUN_TEST(testRequantizeTensorInPlaceGroupedAsym);
    RUN_TEST(testGroupedAsymTargetFromInt32Dies);
    RUN_TEST(testGroupedAsymTargetFromSymInt32Dies);
    RUN_TEST(testGroupedAsymTargetFromSymDies);
    RUN_TEST(testGroupedAsymSourceToInt32Dies);
    RUN_TEST(testGroupedAsymSourceToSymInt32Dies);
    RUN_TEST(testGroupedAsymSourceToSymDies);
    RUN_TEST(testDequantChunkToFloatRejectsGroupedAsymSource);
    RUN_TEST(testAccumulateFloatIntoAsymRescaleRejectsGroupedTarget);
    RUN_TEST(testAccumulateTensorIntoAsymRescaleRejectsGroupedTarget);
    RUN_TEST(testSameTypeAsymCopyCarriesGroupArrays);
    RUN_TEST(testSameTypeAsymCopyRejectsGroupShapeMismatch);

    RUN_TEST(testFloatToBfpPerTensorBasic);
    RUN_TEST(testFloatToBfpSnapsUpAndPowerOfTwoIsExact);
    RUN_TEST(testFloatToBfpGroupedIndependentExponents);
    RUN_TEST(testFloatToBfpSaturatesNarrowExponentHigh);
    RUN_TEST(testFloatToBfpFlushesNarrowExponentLow);
    RUN_TEST(testFloatToBfpAllZeroGroupKeepsZeroStateExponent);
    RUN_TEST(testBfpToFloatRoundTripExactForRepresentable);
    RUN_TEST(testBfpGroupedToFloatDequantsPerGroup);
    RUN_TEST(testFloatToBfpSrRoundingIsSeededDeterministic);

    RUN_TEST(testInt32ToBfpIsCodesInWithZeroStateExponent);
    RUN_TEST(testInt32ToBfpOverflowAborts);
    RUN_TEST(testBfpToInt32DropsExponent);
    RUN_TEST(testSymInt32ToBfpPreservesValues);
    RUN_TEST(testBfpToSymInt32FreshAbsmaxGrid);
    RUN_TEST(testBfpGroupedToSymInt32UsesPerGroupScales);

    RUN_TEST(testSymPerTensorToBfpPreservesValues);
    RUN_TEST(testSymGroupedSourceToBfpUsesPerGroupScales);
    RUN_TEST(testBfpToSymPerTensorFreshAbsmax);
    RUN_TEST(testBfpToSymGroupedTargetDenies);
    RUN_TEST(testAsymToBfpPreservesValues);
    RUN_TEST(testAsymGroupedSourceToBfpUsesPerGroupGrids);
    RUN_TEST(testBfpToAsymUsesCanonicalGrid);
    RUN_TEST(testBfpToAsymGroupedTargetDenies);

    return UNITY_END();
}
