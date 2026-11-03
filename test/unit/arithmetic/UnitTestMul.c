#include "Arithmetic.h"
#include "Mul.h"
#include "Quantization.h"
#include "Tensor.h"
#include "unity.h"

void testMulFloatTensors() {
    size_t numberOfElements = 6;

    float aData[] = {2.2f, 3.5f, 4.1f, 5.2f, 6.8f, 7.3f};
    size_t aNumberOfDims = 2;
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};
    quantization_t aQ = {.type = FLOAT32};

    tensor_t aTensor = {
        .data = (uint8_t *)aData,
        .shape = &aShape,
        .quantization = &aQ,
        .sparsity = NULL,
    };

    float bData[] = {1.5f, 3.3f, 8.2f, 0.2f, 5.5f, 0.9f};
    size_t bNumberOfDims = 2;
    size_t bDims[] = {2, 3};
    size_t bOrderOfDims[] = {1, 0};
    shape_t bShape = {.dimensions = bDims,
                      .orderOfDimensions = bOrderOfDims,
                      .numberOfDimensions = bNumberOfDims};
    quantization_t bQ = {.type = FLOAT32};
    tensor_t bTensor = {
        .data = (uint8_t *)bData,
        .shape = &bShape,
        .quantization = &bQ,
        .sparsity = NULL,
    };

    transposeTensor(&bTensor, 0, 1);

    floatElementArithmeticFunc_t mul = mulFloat32s;
    floatPointWiseArithmeticInplace(&aTensor, &bTensor, mul);

    float expected[] = {3.3f, 11.55f, 33.62f, 1.04f, 37.4f, 6.57f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, aTensor.data, numberOfElements);
}

void testMulFloatElementWithTensor() {
    size_t numberOfElements = 6;

    float aData[] = {2.2f, 3.5f, 4.1f, 5.2f, 6.8f, 7.3f};
    size_t aNumberOfDims = 2;
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape = {.dimensions = aDims,
                      .orderOfDimensions = aOrderOfDims,
                      .numberOfDimensions = aNumberOfDims};
    quantization_t aQ = {.type = FLOAT32};

    tensor_t aTensor = {
        .data = (uint8_t *)aData,
        .shape = &aShape,
        .quantization = &aQ,
        .sparsity = NULL,
    };

    float x = 0.5;

    floatElementArithmeticFunc_t mul = mulFloat32s;
    floatElementWithTensorArithmeticInplace(&aTensor, x, mul);

    float expected[] = {1.1f, 1.75f, 2.05f, 2.6f, 3.4f, 3.65f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, aTensor.data, numberOfElements);
}

void testMulInt32sMatchesNativeProduct() {
    TEST_ASSERT_EQUAL_INT32(-42, mulInt32s(-7, 6));
    TEST_ASSERT_EQUAL_INT32(2147395600, mulInt32s(46340, 46340));
    TEST_ASSERT_EQUAL_INT32(-1073709056, mulInt32s(-32768, 32767));
    TEST_ASSERT_EQUAL_INT32(0, mulInt32s(0, INT32_MIN));
}

#ifdef TRACK_INSTRUCTIONS
void testMulInt32sCountsUnderTrackInstructions() {
    size_t before = getMulInstructionCounter();
    for (int32_t i = 0; i < 5; i++) {
        (void)mulInt32s(i, 3);
    }
    TEST_ASSERT_EQUAL_size_t(5, getMulInstructionCounter() - before);
}
#endif

/* A 1-D SYM_INT32 wire over caller-owned codes. The struct owns the shape and
 * qConfig storage that setShape / setTensorValues keep raw pointers to, so it
 * is initialised in place and never copied. */
typedef struct symWire {
    tensor_t t;
    shape_t shape;
    size_t dims[1];
    size_t order[1];
    symInt32QConfig_t qc;
    quantization_t q;
} symWire_t;

static void symWireInit(symWire_t *w, int32_t *codes, size_t n, uint8_t qMaxBits, float scale) {
    w->dims[0] = n;
    w->order[0] = 0;
    setShape(&w->shape, w->dims, 1, w->order);
    initSymInt32QConfigWithQMaxBits(HALF_AWAY, &w->qc, qMaxBits);
    w->qc.scale = scale;
    initSymInt32Quantization(&w->qc, &w->q);
    setTensorValues(&w->t, (uint8_t *)codes, &w->shape, &w->q, NULL);
}

/* Output wires start at a width no stamp test expects, so a missing stamp can
 * never pass as the right one. */
#define UNSTAMPED_WIDTH 5

/* The raw product of two int12 grid corners is 2^22: the output declares the
 * static bound 2^11 * 2^11 = 2^22 -> strict -> 24 (#516), whatever the data. */
void testMulSymInt32TensorsStampsProductWidth(void) {
    int32_t aCodes[2] = {-2048, 3}, bCodes[2] = {-2048, 5}, outCodes[2] = {0, 0};
    symWire_t a, b, out;
    symWireInit(&a, aCodes, 2, 12, 0.5f);
    symWireInit(&b, bCodes, 2, 12, 0.25f);
    symWireInit(&out, outCodes, 2, UNSTAMPED_WIDTH, 1.f);
    mulSymInt32Tensors(&a.t, &b.t, &out.t);
    TEST_ASSERT_EQUAL_INT32(4194304, outCodes[0]);
    TEST_ASSERT_EQUAL_INT32(15, outCodes[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.125f, out.qc.scale);
    TEST_ASSERT_EQUAL_UINT8(24, out.qc.qMaxBits);
    /* operands untouched */
    TEST_ASSERT_EQUAL_UINT8(12, a.qc.qMaxBits);
    TEST_ASSERT_EQUAL_UINT8(12, b.qc.qMaxBits);
}

/* In place the first operand IS the output: its declared width widens from
 * the operand width 12 to the product width 24 together with its scale. */
void testMulSymInt32TensorsInplaceStampsProductWidth(void) {
    int32_t aCodes[2] = {-2048, 3}, bCodes[2] = {-2048, 5};
    symWire_t a, b;
    symWireInit(&a, aCodes, 2, 12, 0.5f);
    symWireInit(&b, bCodes, 2, 12, 0.25f);
    mulSymInt32TensorsInplace(&a.t, &b.t);
    TEST_ASSERT_EQUAL_INT32(4194304, aCodes[0]);
    TEST_ASSERT_EQUAL_INT32(15, aCodes[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.125f, a.qc.scale);
    TEST_ASSERT_EQUAL_UINT8(24, a.qc.qMaxBits);
    TEST_ASSERT_EQUAL_UINT8(12, b.qc.qMaxBits);
}

/* The stamp is static in the operand widths: int4 x int4 -> 2^3 * 2^3 = 64
 * -> 8; int16 x int16 -> 2^30 -> 32 (the raw int32 range; 2^30 is not < 2^30,
 * so 31 would understate the corner product -2^15 * -2^15). The codes are 1. */
static uint8_t mulStampForWidths(uint8_t qa, uint8_t qb) {
    int32_t aCode = 1, bCode = 1, outCode = 0;
    symWire_t a, b, out;
    symWireInit(&a, &aCode, 1, qa, 1.f);
    symWireInit(&b, &bCode, 1, qb, 1.f);
    symWireInit(&out, &outCode, 1, UNSTAMPED_WIDTH, 1.f);
    mulSymInt32Tensors(&a.t, &b.t, &out.t);
    TEST_ASSERT_EQUAL_INT32(1, outCode);
    return out.qc.qMaxBits;
}

void testMulSymInt32TensorsStampIsStaticInTheWidths(void) {
    TEST_ASSERT_EQUAL_UINT8(8, mulStampForWidths(4, 4));
    TEST_ASSERT_EQUAL_UINT8(13, mulStampForWidths(12, 1)); /* 2^11 * 2^0 = 2^11 -> 13 */
    TEST_ASSERT_EQUAL_UINT8(32, mulStampForWidths(16, 16));
    TEST_ASSERT_EQUAL_UINT8(32, mulStampForWidths(31, 31)); /* 2^60 caps at 32 */
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMulFloatTensors);
    RUN_TEST(testMulFloatElementWithTensor);
    RUN_TEST(testMulInt32sMatchesNativeProduct);
#ifdef TRACK_INSTRUCTIONS
    RUN_TEST(testMulInt32sCountsUnderTrackInstructions);
#endif
    RUN_TEST(testMulSymInt32TensorsStampsProductWidth);
    RUN_TEST(testMulSymInt32TensorsInplaceStampsProductWidth);
    RUN_TEST(testMulSymInt32TensorsStampIsStaticInTheWidths);

    return UNITY_END();
}
