#include "param_gate.h"

#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include <stdbool.h>
#include <string.h>

/* Compile-time signature contracts (fastest RED: the header does not exist yet). */
_Static_assert(_Generic((&resolveGroupShape),
                   groupShape_t (*)(size_t, size_t, groupModeSweep_t, int): 1,
                   default: 0),
               "resolveGroupShape must take (size_t N, size_t outCh, groupModeSweep_t, int)");
_Static_assert(_Generic((&viewQShape), qShapeView_t (*)(const quantization_t *): 1, default: 0),
               "viewQShape must take (const quantization_t *)");
_Static_assert(
    _Generic((&paramGateCheck),
        bool (*)(const tensor_t *, const paramGateExpect_t *, char *, size_t): 1,
        default: 0),
    "paramGateCheck must take (const tensor_t *, const paramGateExpect_t *, char *, size_t)");

void setUp() {}
void tearDown() {}

/* Post-#106 heap fixture: [4,3] tensor (N = 12, outCh = 4) owning `q`. Data
 * stays zero -- the gate reads quantization metadata only. */
static tensor_t *makeTensor4x3(quantization_t *q) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 4;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, dims, 2, order);
    return initTensor(s, q, NULL);
}

static bool checkOwned(quantization_t *q, paramGateExpect_t expect, char *msg, size_t msgLen) {
    tensor_t *t = makeTensor4x3(q);
    bool ok = paramGateCheck(t, &expect, msg, msgLen);
    freeTensor(t);
    return ok;
}

/* ---- resolveGroupShape: the HAR table from train_c_sym.c, pinned ---------- */

void testResolveGroupShapeTensorModeIsPerTensor(void) {
    groupShape_t gs = resolveGroupShape(2560, 32, GROUP_MODE_TENSOR, 64);
    TEST_ASSERT_EQUAL_size_t(1, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, gs.groupSize);
}

void testResolveGroupShapeChannelModeOneGroupPerOutCh(void) {
    groupShape_t gs = resolveGroupShape(384, 6, GROUP_MODE_CHANNEL, 0); /* HAR linear */
    TEST_ASSERT_EQUAL_size_t(6, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(64, gs.groupSize);
}

void testResolveGroupShapeSizeModeDivides(void) {
    groupShape_t gs = resolveGroupShape(2560, 32, GROUP_MODE_SIZE, 64); /* HAR conv2 G64 */
    TEST_ASSERT_EQUAL_size_t(40, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(64, gs.groupSize);
}

void testResolveGroupShapeSizeModeFallsBackToChannel(void) {
    groupShape_t gs = resolveGroupShape(1008, 16, GROUP_MODE_SIZE, 64); /* HAR conv1: 64 !| 1008 */
    TEST_ASSERT_EQUAL_size_t(16, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(63, gs.groupSize);
}

void testResolveGroupShapeCollapsesSingleGroupToPerTensor(void) {
    groupShape_t gs = resolveGroupShape(64, 1, GROUP_MODE_CHANNEL, 0); /* {1,64} is never valid */
    TEST_ASSERT_EQUAL_size_t(1, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, gs.groupSize);
}

/* ---- viewQShape: three arms + fail-fast default ---------------------------- */

void testViewQShapeSymGrouped(void) {
    quantization_t *q = quantizationInitSymGrouped(4, HALF_AWAY, 4, 3);
    qShapeView_t v = viewQShape(q);
    freeQuantization(q);
    TEST_ASSERT_EQUAL_UINT8(4, v.qBits);
    TEST_ASSERT_EQUAL_UINT8(0, v.exponentBits);
    TEST_ASSERT_EQUAL_size_t(4, v.numGroups);
    TEST_ASSERT_EQUAL_size_t(3, v.groupSize);
}

void testViewQShapeAsymPerTensor(void) {
    quantization_t *q = quantizationInitAsym(6, HALF_AWAY);
    qShapeView_t v = viewQShape(q);
    freeQuantization(q);
    TEST_ASSERT_EQUAL_UINT8(6, v.qBits);
    TEST_ASSERT_EQUAL_UINT8(0, v.exponentBits);
    TEST_ASSERT_EQUAL_size_t(1, v.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, v.groupSize);
}

void testViewQShapeBfpGroupedReportsMantissaAndExponentBits(void) {
    quantization_t *q = quantizationInitBfpGrouped(5, 6, HALF_AWAY, 4, 3);
    qShapeView_t v = viewQShape(q);
    freeQuantization(q);
    TEST_ASSERT_EQUAL_UINT8(5, v.qBits);
    TEST_ASSERT_EQUAL_UINT8(6, v.exponentBits);
    TEST_ASSERT_EQUAL_size_t(4, v.numGroups);
    TEST_ASSERT_EQUAL_size_t(3, v.groupSize);
}

void testViewQShapeFloat32FailsFast(void) {
    quantization_t *q = quantizationInitFloat();
    ASSERT_EXITS_WITH_FAILURE(viewQShape(q));
    freeQuantization(q);
}

/* ---- paramGateCheck: BFP arm ------------------------------------------------ */

void testGateBfpPerTensorMatches(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitBfp(8, 8, HALF_AWAY),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 1, .groupSize = 0}},
        msg, sizeof(msg));
    TEST_ASSERT_TRUE(ok);
}

void testGateBfpGroupedMatches(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitBfpGrouped(8, 8, HALF_AWAY, 4, 3),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 4, .groupSize = 3}},
        msg, sizeof(msg));
    TEST_ASSERT_TRUE(ok);
}

void testGateBfpWrongMantissaBitsFails(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitBfp(4, 8, HALF_AWAY),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 1, .groupSize = 0}},
        msg, sizeof(msg));
    TEST_ASSERT_FALSE(ok);
    TEST_ASSERT_NOT_NULL(strstr(msg, "mantissaBits"));
}

void testGateBfpWrongExponentBitsFails(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitBfp(8, 4, HALF_AWAY),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 1, .groupSize = 0}},
        msg, sizeof(msg));
    TEST_ASSERT_FALSE(ok);
    TEST_ASSERT_NOT_NULL(strstr(msg, "exponentBits"));
}

void testGateBfpWrongGroupShapeFails(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitBfpGrouped(8, 8, HALF_AWAY, 4, 3),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 1, .groupSize = 0}},
        msg, sizeof(msg));
    TEST_ASSERT_FALSE(ok);
    TEST_ASSERT_NOT_NULL(strstr(msg, "group shape"));
}

void testGateBfpExpectedButSymStoredFails(void) {
    char msg[160] = "";
    bool ok = checkOwned(
        quantizationInitSym(8, HALF_AWAY),
        (paramGateExpect_t){
            .type = BFP, .bits = 8, .exponentBits = 8, .shape = {.numGroups = 1, .groupSize = 0}},
        msg, sizeof(msg));
    TEST_ASSERT_FALSE(ok);
    TEST_ASSERT_NOT_NULL(strstr(msg, "expected BFP"));
    TEST_ASSERT_NOT_NULL(strstr(msg, "got SYM"));
}

/* ---- paramGateCheck: SYM / ASYM arms (the HAR trainer's existing contract) -- */

void testGateSymPerTensorMatchesAndWrongBitsFails(void) {
    char msg[160] = "";
    bool okMatch = checkOwned(
        quantizationInitSym(4, HALF_AWAY),
        (paramGateExpect_t){.type = SYM, .bits = 4, .shape = {.numGroups = 1, .groupSize = 0}}, msg,
        sizeof(msg));
    bool okBits = checkOwned(
        quantizationInitSym(4, HALF_AWAY),
        (paramGateExpect_t){.type = SYM, .bits = 8, .shape = {.numGroups = 1, .groupSize = 0}}, msg,
        sizeof(msg));
    TEST_ASSERT_TRUE(okMatch);
    TEST_ASSERT_FALSE(okBits);
    TEST_ASSERT_NOT_NULL(strstr(msg, "qBits"));
}

void testGateAsymGroupedMatchesAndWrongShapeFails(void) {
    char msg[160] = "";
    bool okMatch = checkOwned(
        quantizationInitAsymGrouped(4, HALF_AWAY, 4, 3),
        (paramGateExpect_t){.type = ASYM, .bits = 4, .shape = {.numGroups = 4, .groupSize = 3}},
        msg, sizeof(msg));
    bool okShape = checkOwned(
        quantizationInitAsymGrouped(4, HALF_AWAY, 4, 3),
        (paramGateExpect_t){.type = ASYM, .bits = 4, .shape = {.numGroups = 2, .groupSize = 6}},
        msg, sizeof(msg));
    TEST_ASSERT_TRUE(okMatch);
    TEST_ASSERT_FALSE(okShape);
    TEST_ASSERT_NOT_NULL(strstr(msg, "group shape"));
}

/* ---- paramGateCheck: FLOAT32 expectation is a type-only check (grad gate) --- */

void testGateFloat32ExpectationIsTypeOnly(void) {
    char msg[160] = "";
    bool okFloat =
        checkOwned(quantizationInitFloat(), (paramGateExpect_t){.type = FLOAT32}, msg, sizeof(msg));
    bool okBfp = checkOwned(quantizationInitBfp(8, 8, HALF_AWAY),
                            (paramGateExpect_t){.type = FLOAT32}, msg, sizeof(msg));
    TEST_ASSERT_TRUE(okFloat);
    TEST_ASSERT_FALSE(okBfp);
    TEST_ASSERT_NOT_NULL(strstr(msg, "expected FLOAT32"));
}

/* An expectation dtype the gate has no arm for (SYM_INT32 is compute
 * format, never sweep storage) must fail fast, not silently pass. */
void testGateUnsupportedExpectationFailsFast(void) {
    tensor_t *t = makeTensor4x3(quantizationInitSymInt32(HALF_AWAY));
    paramGateExpect_t expect = {.type = SYM_INT32, .bits = 8};
    char msg[160] = "";
    ASSERT_EXITS_WITH_FAILURE(paramGateCheck(t, &expect, msg, sizeof(msg)));
    freeTensor(t);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testResolveGroupShapeTensorModeIsPerTensor);
    RUN_TEST(testResolveGroupShapeChannelModeOneGroupPerOutCh);
    RUN_TEST(testResolveGroupShapeSizeModeDivides);
    RUN_TEST(testResolveGroupShapeSizeModeFallsBackToChannel);
    RUN_TEST(testResolveGroupShapeCollapsesSingleGroupToPerTensor);
    RUN_TEST(testViewQShapeSymGrouped);
    RUN_TEST(testViewQShapeAsymPerTensor);
    RUN_TEST(testViewQShapeBfpGroupedReportsMantissaAndExponentBits);
    RUN_TEST(testViewQShapeFloat32FailsFast);
    RUN_TEST(testGateBfpPerTensorMatches);
    RUN_TEST(testGateBfpGroupedMatches);
    RUN_TEST(testGateBfpWrongMantissaBitsFails);
    RUN_TEST(testGateBfpWrongExponentBitsFails);
    RUN_TEST(testGateBfpWrongGroupShapeFails);
    RUN_TEST(testGateBfpExpectedButSymStoredFails);
    RUN_TEST(testGateSymPerTensorMatchesAndWrongBitsFails);
    RUN_TEST(testGateAsymGroupedMatchesAndWrongShapeFails);
    RUN_TEST(testGateFloat32ExpectationIsTypeOnly);
    RUN_TEST(testGateUnsupportedExpectationFailsFast);
    return UNITY_END();
}
