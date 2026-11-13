#define SOURCE_FILE "POINTWISE-FUSED-UTEST"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "PointwiseFused.h"
#include "Tensor.h"
#include "unity.h"

#include "expected_pointwise_fused.h"

/* #328 signature contracts (compile-time pins). */
_Static_assert(_Generic(&lerpFloat32TensorsInplace,
                   void (*)(tensor_t *, tensor_t *, float): 1,
                   default: 0),
               "#328: lerpFloat32TensorsInplace must be (a, b, weight)");
_Static_assert(_Generic(&addcmulFloat32TensorsInplace,
                   void (*)(tensor_t *, tensor_t *, tensor_t *, float): 1,
                   default: 0),
               "#328: addcmulFloat32TensorsInplace must be (a, b, c, s)");
_Static_assert(_Generic(&addcdivDenomFloat32TensorsInplace,
                   void (*)(tensor_t *, tensor_t *, tensor_t *, float, float, float): 1,
                   default: 0),
               "#328: addcdivDenomFloat32TensorsInplace must be (a, b, v, dScale, eps, s)");

void setUp() {}
void tearDown() {}

/* Raw-stack fixture idiom (UnitTestMul.c precedent), 1-D identity order. */
typedef struct {
    float data[64];
    size_t dims[1];
    size_t order[1];
    shape_t shape;
    quantization_t q;
    tensor_t tensor;
} pfFixture_t;

static tensor_t *pfInit(pfFixture_t *f, const float *src, size_t n) {
    memcpy(f->data, src, n * sizeof(float));
    f->dims[0] = n;
    f->order[0] = 0;
    f->shape =
        (shape_t){.dimensions = f->dims, .orderOfDimensions = f->order, .numberOfDimensions = 1};
    f->q = (quantization_t){.type = FLOAT32};
    f->tensor = (tensor_t){
        .data = (uint8_t *)f->data, .shape = &f->shape, .quantization = &f->q, .sparsity = NULL};
    return &f->tensor;
}

/* Gold header generated at build time on this host; C runs the same
 * per-element rounding order -> any difference is a semantics bug. */
static void assertBitsEqualAt(const float *expected, const tensor_t *actual, size_t n,
                              const char *what) {
    const float *act = (const float *)actual->data;
    for (size_t i = 0; i < n; i++) {
        uint32_t e, a;
        memcpy(&e, &expected[i], sizeof e);
        memcpy(&a, &act[i], sizeof a);
        char msg[96];
        snprintf(msg, sizeof msg, "%s mismatch at element %zu", what, i);
        TEST_ASSERT_EQUAL_HEX32_MESSAGE(e, a, msg);
    }
}

static void runLerpGold(const float *aGold, const float *bGold, const float *expGold, size_t n) {
    pfFixture_t fa, fb;
    tensor_t *a = pfInit(&fa, aGold, n);
    tensor_t *b = pfInit(&fb, bGold, n);
    lerpFloat32TensorsInplace(a, b, pf_lerp_weight);
    assertBitsEqualAt(expGold, a, n, "lerp");
    /* b must be untouched */
    assertBitsEqualAt(bGold, b, n, "lerp b-operand");
}

void testLerpMatchesTorchGoldN32(void) {
    runLerpGold(pf_a_n32, pf_b_n32, pf_expected_lerp_n32, pf_a_n32_len);
}

void testLerpMatchesTorchGoldN64(void) {
    runLerpGold(pf_a_n64, pf_b_n64, pf_expected_lerp_n64, pf_a_n64_len);
}

void testLerpSelfAliasIsIdentity(void) {
    /* b aliasing a: b[i]-a[i] == 0 -> fmaf(w, 0, a[i]) == a[i] exactly. */
    pfFixture_t fa;
    tensor_t *a = pfInit(&fa, pf_a_n32, 32);
    lerpFloat32TensorsInplace(a, a, pf_lerp_weight);
    assertBitsEqualAt(pf_a_n32, a, 32, "lerp self-alias");
}

#ifdef TRACK_INSTRUCTIONS
void testLerpCounterDeltaIsElementCount(void) {
    /* Counters are cumulative process-globals: assert the DELTA. */
    pfFixture_t fa, fb;
    tensor_t *a = pfInit(&fa, pf_a_n32, 32);
    tensor_t *b = pfInit(&fb, pf_b_n32, 32);
    size_t before = getLerpInstructionCounter();
    lerpFloat32TensorsInplace(a, b, pf_lerp_weight);
    TEST_ASSERT_EQUAL_size_t(32, getLerpInstructionCounter() - before);
}
#endif

static void runAddcmulGold(const float *aGold, const float *bGold, const float *cGold,
                           const float *expGold, size_t n) {
    pfFixture_t fa, fb, fc;
    tensor_t *a = pfInit(&fa, aGold, n);
    tensor_t *b = pfInit(&fb, bGold, n);
    tensor_t *c = pfInit(&fc, cGold, n);
    addcmulFloat32TensorsInplace(a, b, c, pf_addcmul_s);
    assertBitsEqualAt(expGold, a, n, "addcmul");
}

void testAddcmulMatchesTorchGoldN32(void) {
    runAddcmulGold(pf_a_n32, pf_b_n32, pf_c_n32, pf_expected_addcmul_n32, pf_a_n32_len);
}

void testAddcmulMatchesTorchGoldN64(void) {
    runAddcmulGold(pf_a_n64, pf_b_n64, pf_c_n64, pf_expected_addcmul_n64, pf_a_n64_len);
}

void testAddcmulFromZeroMatchesTorchGold(void) {
    /* AdamW's FIRST second-moment step: v starts at zero, so the addcmul term
     * IS the result (0 + x is exact) and the product's reassociation bits
     * fully survive — this fixture is grouping-mutation-sensitive where the
     * O(1)-a golds are not (s~1e-3 term absorbed below a[i]'s ULP). */
    static const float pfZeros32[32] = {0};
    static const float pfZeros64[64] = {0};
    runAddcmulGold(pfZeros32, pf_b_n32, pf_c_n32, pf_expected_addcmul_azero_n32, 32);
    runAddcmulGold(pfZeros64, pf_b_n64, pf_c_n64, pf_expected_addcmul_azero_n64, 64);
}

void testAddcmulSameOperandTwiceMatchesTorchGold(void) {
    /* b == c aliasing (AdamW K2: addcmul(v, g, g, 1-beta2)). */
    pfFixture_t fa32, fb32;
    tensor_t *a32 = pfInit(&fa32, pf_a_n32, 32);
    tensor_t *b32 = pfInit(&fb32, pf_b_n32, 32);
    addcmulFloat32TensorsInplace(a32, b32, b32, pf_addcmul_s);
    assertBitsEqualAt(pf_expected_addcmul_bb_n32, a32, 32, "addcmul b==c n32");

    pfFixture_t fa64, fb64;
    tensor_t *a64 = pfInit(&fa64, pf_a_n64, 64);
    tensor_t *b64 = pfInit(&fb64, pf_b_n64, 64);
    addcmulFloat32TensorsInplace(a64, b64, b64, pf_addcmul_s);
    assertBitsEqualAt(pf_expected_addcmul_bb_n64, a64, 64, "addcmul b==c n64");
}

#ifdef TRACK_INSTRUCTIONS
void testAddcmulCounterDeltaIsElementCount(void) {
    pfFixture_t fa, fb, fc;
    tensor_t *a = pfInit(&fa, pf_a_n32, 32);
    tensor_t *b = pfInit(&fb, pf_b_n32, 32);
    tensor_t *c = pfInit(&fc, pf_c_n32, 32);
    size_t before = getAddcmulInstructionCounter();
    addcmulFloat32TensorsInplace(a, b, c, pf_addcmul_s);
    TEST_ASSERT_EQUAL_size_t(32, getAddcmulInstructionCounter() - before);
}
#endif

static void runAddcdivGold(const float *aGold, const float *bGold, const float *vGold,
                           const float *expGold, size_t n) {
    pfFixture_t fa, fb, fv;
    tensor_t *a = pfInit(&fa, aGold, n);
    tensor_t *b = pfInit(&fb, bGold, n);
    tensor_t *v = pfInit(&fv, vGold, n);
    addcdivDenomFloat32TensorsInplace(a, b, v, pf_addcdiv_dscale, pf_addcdiv_eps, pf_addcdiv_s);
    assertBitsEqualAt(expGold, a, n, "addcdivDenom");
}

void testAddcdivDenomMatchesTorchGoldN32(void) {
    runAddcdivGold(pf_a_n32, pf_b_n32, pf_v_n32, pf_expected_addcdiv_n32, pf_a_n32_len);
}

void testAddcdivDenomMatchesTorchGoldN64(void) {
    runAddcdivGold(pf_a_n64, pf_b_n64, pf_v_n64, pf_expected_addcdiv_n64, pf_a_n64_len);
}

void testAddcdivDenomFromZeroMatchesTorchGold(void) {
    /* Mutation-sensitivity fixture (Task 2 lesson applied proactively): the
     * O(1)-a golds above are grouping-mutation BLIND for this op (measured
     * 0/32, 0/64 final-bit differences under the s*(b/d) swap — quotient
     * sits ~3-4 orders of magnitude below a's ULP-relevant scale). With
     * a=0 the addition is exact (0+x==x), so the quotient's own rounding
     * bits survive to the gold comparison (measured 17/32, 21/64 differ). */
    static const float pfZeros32[32] = {0};
    static const float pfZeros64[64] = {0};
    runAddcdivGold(pfZeros32, pf_b_n32, pf_v_n32, pf_expected_addcdiv_azero_n32, 32);
    runAddcdivGold(pfZeros64, pf_b_n64, pf_v_n64, pf_expected_addcdiv_azero_n64, 64);
}

#ifdef TRACK_INSTRUCTIONS
void testAddcdivDenomCounterDeltaIsElementCount(void) {
    pfFixture_t fa, fb, fv;
    tensor_t *a = pfInit(&fa, pf_a_n32, 32);
    tensor_t *b = pfInit(&fb, pf_b_n32, 32);
    tensor_t *v = pfInit(&fv, pf_v_n32, 32);
    size_t before = getAddcdivDenomInstructionCounter();
    addcdivDenomFloat32TensorsInplace(a, b, v, pf_addcdiv_dscale, pf_addcdiv_eps, pf_addcdiv_s);
    TEST_ASSERT_EQUAL_size_t(32, getAddcdivDenomInstructionCounter() - before);
}
#endif

int main() {
    UNITY_BEGIN();
    RUN_TEST(testLerpMatchesTorchGoldN32);
    RUN_TEST(testLerpMatchesTorchGoldN64);
    RUN_TEST(testLerpSelfAliasIsIdentity);
#ifdef TRACK_INSTRUCTIONS
    RUN_TEST(testLerpCounterDeltaIsElementCount);
#endif
    RUN_TEST(testAddcmulMatchesTorchGoldN32);
    RUN_TEST(testAddcmulMatchesTorchGoldN64);
    RUN_TEST(testAddcmulFromZeroMatchesTorchGold);
    RUN_TEST(testAddcmulSameOperandTwiceMatchesTorchGold);
#ifdef TRACK_INSTRUCTIONS
    RUN_TEST(testAddcmulCounterDeltaIsElementCount);
#endif
    RUN_TEST(testAddcdivDenomMatchesTorchGoldN32);
    RUN_TEST(testAddcdivDenomMatchesTorchGoldN64);
    RUN_TEST(testAddcdivDenomFromZeroMatchesTorchGold);
#ifdef TRACK_INSTRUCTIONS
    RUN_TEST(testAddcdivDenomCounterDeltaIsElementCount);
#endif
    return UNITY_END();
}
