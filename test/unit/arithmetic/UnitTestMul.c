#include "Mul.h"
#include "Quantization.h"
#include "Tensor.h"
#include "TestScale.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

void testMulFloat32ElementWithFloat32TensorInplace() {
    float aData[] = {2.2f, 3.5f, 4.1f, 5.2f, 6.8f, 7.3f};
    size_t aDims[] = {2, 3};
    size_t aOrderOfDims[] = {0, 1};
    shape_t aShape = {
        .dimensions = aDims, .orderOfDimensions = aOrderOfDims, .numberOfDimensions = 2};
    quantization_t aQ = {.type = FLOAT32};
    tensor_t aTensor = {
        .data = (uint8_t *)aData, .shape = &aShape, .quantization = &aQ, .sparsity = NULL};

    mulFloat32ElementWithFloat32TensorInplace(&aTensor, 0.5f);

    float expected[] = {1.1f, 1.75f, 2.05f, 2.6f, 3.4f, 3.65f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, aTensor.data, 6);
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
    w->qc.scale = odtTestScale(scale);
    initSymInt32Quantization(&w->qc, &w->q);
    setTensorValues(&w->t, (uint8_t *)codes, &w->shape, &w->q, NULL);
}

/* Output wires start at a width no stamp test expects, so a missing stamp can
 * never pass as the right one. */
#define UNSTAMPED_WIDTH 5

/* The raw product of two int12 grid corners is 2^22: the output declares the
 * static range [-2^11 * (2^11 - 1), 2^22] -> 24 (#516), whatever the data. */
void testMulSymInt32TensorsStampsProductWidth(void) {
    int32_t aCodes[2] = {-2048, 3}, bCodes[2] = {-2048, 5}, outCodes[2] = {0, 0};
    symWire_t a, b, out;
    symWireInit(&a, aCodes, 2, 12, 0.5f);
    symWireInit(&b, bCodes, 2, 12, 0.25f);
    symWireInit(&out, outCodes, 2, UNSTAMPED_WIDTH, 1.f);
    mulSymInt32Tensors(&a.t, &b.t, &out.t);
    TEST_ASSERT_EQUAL_INT32(4194304, outCodes[0]);
    TEST_ASSERT_EQUAL_INT32(15, outCodes[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.125f, odtScaleToFloat(out.qc.scale));
    TEST_ASSERT_EQUAL_UINT8(24, out.qc.qMaxBits);
    /* operands untouched */
    TEST_ASSERT_EQUAL_UINT8(12, a.qc.qMaxBits);
    TEST_ASSERT_EQUAL_UINT8(12, b.qc.qMaxBits);
}

/* The stamp is static in the operand widths: int4 x int4 -> the corner 2^3 *
 * 2^3 = 64 -> 8; int16 x int16 -> 2^30 -> 32 (the raw int32 range: 31 bits
 * end at 2^30 - 1 and would understate the corner product -2^15 * -2^15).
 * The codes are 1. */
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testMulFloat32ElementWithFloat32TensorInplace);
#ifdef TRACK_INSTRUCTIONS
    RUN_TEST(testMulInt32sCountsUnderTrackInstructions);
#endif
    RUN_TEST(testMulSymInt32TensorsStampsProductWidth);
    RUN_TEST(testMulSymInt32TensorsStampIsStaticInTheWidths);
    return UNITY_END();
}
