#include "param_gate.h"

#include "BfpKernelSupport.h"
#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

#include <stdbool.h>
#include <stdlib.h>
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
_Static_assert(_Generic((&resolveWireShape),
                   groupShape_t (*)(size_t, wireBlockSweep_t, int): 1,
                   default: 0),
               "resolveWireShape must take (size_t N, wireBlockSweep_t, int)");
_Static_assert(_Generic((&packedPayloadBytes), size_t (*)(qtype_t, uint8_t, size_t): 1, default: 0),
               "packedPayloadBytes must take (qtype_t, uint8_t bits, size_t N)");
_Static_assert(_Generic((&packedMetadataBytes), size_t (*)(qtype_t, size_t): 1, default: 0),
               "packedMetadataBytes must take (qtype_t, size_t numGroups)");
_Static_assert(_Generic((&bfpSweepConfigFromEnv),
                   const char *(*)(bfpSweepConfig_t *): 1,
                   default: 0),
               "bfpSweepConfigFromEnv must take (bfpSweepConfig_t *) and return const char *");

_Static_assert(_Generic((&bfpBlockHeadroomFits),
                   bool (*)(uint8_t, uint8_t, size_t, size_t, size_t): 1,
                   default: 0),
               "bfpBlockHeadroomFits must take (ma, mb, runA, runB, reductionLen)");
_Static_assert(_Generic((&bfpSumHeadroomFits), bool (*)(uint8_t, size_t, size_t): 1, default: 0),
               "bfpSumHeadroomFits must take (m, run, reductionLen)");

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

void testResolveGroupShapeRejectsZeroOutCh(void) {
    ASSERT_EXITS_WITH_FAILURE(resolveGroupShape(64, 0, GROUP_MODE_CHANNEL, 0));
}

void testResolveGroupShapeRejectsEmptyTensor(void) {
    ASSERT_EXITS_WITH_FAILURE(resolveGroupShape(0, 1, GROUP_MODE_CHANNEL, 0));
}

void testResolveGroupShapeRejectsNonDividingOutCh(void) {
    ASSERT_EXITS_WITH_FAILURE(resolveGroupShape(10, 3, GROUP_MODE_CHANNEL, 0)); /* {3,3} != 10 */
}

void testResolveGroupShapeGuardIsModeIndependent(void) {
    /* per-tensor mode divides nothing, but the precondition is the function's,
     * not the grouped branches' -- an empty tensor is rejected here too. */
    ASSERT_EXITS_WITH_FAILURE(resolveGroupShape(0, 1, GROUP_MODE_TENSOR, 0));
}

/* ---- resolveWireShape: the HAR wire table from spec §5, pinned --------- */

void testResolveWireShapeDivisorGroups(void) {
    groupShape_t gs = resolveWireShape(2048, WIRE_BLOCK_SIZE, 16); /* conv1.out @ ab16 */
    TEST_ASSERT_EQUAL_size_t(128, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(16, gs.groupSize);
}

void testResolveWireShapeHeadWireFallsBackToPerTensor(void) {
    groupShape_t gs = resolveWireShape(6, WIRE_BLOCK_SIZE, 16); /* linear.out: 16 !| 6 */
    TEST_ASSERT_EQUAL_size_t(1, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, gs.groupSize);
}

void testResolveWireShapeSizeEqualToWireNormalizesToPerTensor(void) {
    groupShape_t gs = resolveWireShape(64, WIRE_BLOCK_SIZE, 64); /* flatten.out @ ab64 */
    TEST_ASSERT_EQUAL_size_t(1, gs.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, gs.groupSize);
}

void testResolveWireShapeTensorAndFloatModesArePerTensor(void) {
    groupShape_t t = resolveWireShape(2048, WIRE_BLOCK_TENSOR, 0);
    groupShape_t f = resolveWireShape(2048, WIRE_BLOCK_FLOAT, 0);
    TEST_ASSERT_EQUAL_size_t(1, t.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, t.groupSize);
    TEST_ASSERT_EQUAL_size_t(1, f.numGroups);
    TEST_ASSERT_EQUAL_size_t(0, f.groupSize);
}

void testResolveWireShapeForwardAndDxResolveIndependently(void) {
    /* conv2: out = 2048 elements, dx = its INPUT = 1024 elements. */
    groupShape_t out = resolveWireShape(2048, WIRE_BLOCK_SIZE, 32);
    groupShape_t dx = resolveWireShape(1024, WIRE_BLOCK_SIZE, 32);
    TEST_ASSERT_EQUAL_size_t(64, out.numGroups);
    TEST_ASSERT_EQUAL_size_t(32, dx.numGroups);
    TEST_ASSERT_EQUAL_size_t(32, out.groupSize);
    TEST_ASSERT_EQUAL_size_t(32, dx.groupSize);
}

void testResolveWireShapeRejectsEmptyWire(void) {
    ASSERT_EXITS_WITH_FAILURE(resolveWireShape(0, WIRE_BLOCK_TENSOR, 0));
}

void testResolveWireShapeRejectsNonPositiveSize(void) {
    ASSERT_EXITS_WITH_FAILURE(resolveWireShape(2048, WIRE_BLOCK_SIZE, 0));
    ASSERT_EXITS_WITH_FAILURE(resolveWireShape(2048, WIRE_BLOCK_SIZE, -8));
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

/* Type-only means type-only: deliberately wrong bits/shape on a FLOAT32
 * expectation over a FLOAT32 tensor must still pass. */
void testGateFloat32ExpectationIgnoresBitsAndShape(void) {
    char msg[160] = "";
    bool ok = checkOwned(quantizationInitFloat(),
                         (paramGateExpect_t){.type = FLOAT32,
                                             .bits = 99,
                                             .exponentBits = 99,
                                             .shape = {.numGroups = 4, .groupSize = 3}},
                         msg, sizeof(msg));
    TEST_ASSERT_TRUE(ok);
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

/* The unsupported-expectation guard must fire regardless of the tensor's
 * actual type -- including (especially) when actual != expected, which is
 * the common case for a programmer error the guard exists to catch. */
void testGateUnsupportedExpectationFailsFastOnTypeMismatch(void) {
    tensor_t *t = makeTensor4x3(quantizationInitFloat());      /* actual FLOAT32 */
    paramGateExpect_t expect = {.type = SYM_INT32, .bits = 8}; /* no gate arm */
    char msg[160] = "";
    ASSERT_EXITS_WITH_FAILURE(paramGateCheck(t, &expect, msg, sizeof(msg)));
    freeTensor(t);
}

/* ---- packed byte accounting (spec §7.1) ---------------------------------- */

void testPackedPayloadBytesRoundsUpPerTensor(void) {
    TEST_ASSERT_EQUAL_size_t(5, packedPayloadBytes(BFP, 6, 6)); /* 36 bits -> 5 B */
    TEST_ASSERT_EQUAL_size_t(1536, packedPayloadBytes(BFP, 6, 2048));
    TEST_ASSERT_EQUAL_size_t(24, packedPayloadBytes(FLOAT32, 32, 6));
    TEST_ASSERT_EQUAL_size_t(3, packedPayloadBytes(SYM, 4, 6));
    TEST_ASSERT_EQUAL_size_t(3, packedPayloadBytes(ASYM, 4, 6));
}

void testPackedPayloadBytesMatchesFrameworkAccounting(void) {
    quantization_t *bfpQ = quantizationInitBfp(6, 8, HALF_AWAY);
    quantization_t *symQ = quantizationInitSym(4, HALF_AWAY);
    quantization_t *floatQ = quantizationInitFloat();
    /* [4,3] fixture = 12 elements (makeTensor4x3 owns q, freeTensor frees it). */
    tensor_t *tb = makeTensor4x3(bfpQ);
    tensor_t *ts = makeTensor4x3(symQ);
    tensor_t *tf = makeTensor4x3(floatQ);
    size_t b = calcBytesPerTensor(tb), s = calcBytesPerTensor(ts), f = calcBytesPerTensor(tf);
    freeTensor(tb);
    freeTensor(ts);
    freeTensor(tf);
    TEST_ASSERT_EQUAL_size_t(b, packedPayloadBytes(BFP, 6, 12));
    TEST_ASSERT_EQUAL_size_t(s, packedPayloadBytes(SYM, 4, 12));
    TEST_ASSERT_EQUAL_size_t(f, packedPayloadBytes(FLOAT32, 32, 12));
}

void testPackedMetadataBytesPerDtype(void) {
    TEST_ASSERT_EQUAL_size_t(128, packedMetadataBytes(BFP, 128));  /* 1 B exponent per group */
    TEST_ASSERT_EQUAL_size_t(512, packedMetadataBytes(SYM, 128));  /* 4 B float scale */
    TEST_ASSERT_EQUAL_size_t(768, packedMetadataBytes(ASYM, 128)); /* 4 B scale + 2 B zero-point */
    TEST_ASSERT_EQUAL_size_t(0, packedMetadataBytes(FLOAT32, 1));
}

void testPackedBytesRejectUnsupportedDtype(void) {
    ASSERT_EXITS_WITH_FAILURE(packedPayloadBytes(BOOL, 1, 8));
    ASSERT_EXITS_WITH_FAILURE(packedMetadataBytes(SYM_INT32, 1));
}

/* ---- bfpSweepConfigFromEnv (spec §3.1) ---------------------------------- */

static const char *const kBfpKnobs[] = {
    "BFP_MANTISSA_BITS", "BFP_EXPONENT_BITS", "BFP_WEIGHT_BLOCK", "BFP_WIRE_BLOCK",
    "BFP_MATH",          "BFP_GRADS",         "BFP_STATE",        "BFP_ROUNDING"};
static const char *const kLegacyKnobs[] = {"SYM_BITS",      "SYM_WIRES",  "WEIGHT_DTYPE",
                                           "GROUP_MODE",    "GROUP_SIZE", "SYM_ROUNDING",
                                           "ODTS_ROUNDTRIP"};

static void clearSweepEnv(void) {
    for (size_t i = 0; i < sizeof(kBfpKnobs) / sizeof(kBfpKnobs[0]); i++) {
        unsetenv(kBfpKnobs[i]);
    }
    for (size_t i = 0; i < sizeof(kLegacyKnobs) / sizeof(kLegacyKnobs[0]); i++) {
        unsetenv(kLegacyKnobs[i]);
    }
}

void testSweepConfigDefaults(void) {
    clearSweepEnv();
    bfpSweepConfig_t c;
    TEST_ASSERT_NULL(bfpSweepConfigFromEnv(&c));
    TEST_ASSERT_EQUAL_UINT8(8, c.mantissaBits);
    TEST_ASSERT_EQUAL_UINT8(8, c.exponentBits);
    TEST_ASSERT_EQUAL_INT(GROUP_MODE_TENSOR, c.weightMode);
    TEST_ASSERT_EQUAL_INT(WIRE_BLOCK_FLOAT, c.wireMode);
    TEST_ASSERT_EQUAL_INT(BFP_MATH_NATIVE, c.math);
    TEST_ASSERT_FALSE(c.bfpGrads);
    TEST_ASSERT_FALSE(c.bfpState);
    TEST_ASSERT_EQUAL_INT(BFP_ROUNDING_SR, c.rounding);
    TEST_ASSERT_EQUAL_UINT(0u, c.ignoredLegacyKnobs);
    TEST_ASSERT_EQUAL_STRING("tensor", c.weightBlockStr);
    TEST_ASSERT_EQUAL_STRING("float", c.wireBlockStr);
}

void testSweepConfigParsesTheAnchorConfig(void) {
    clearSweepEnv();
    setenv("BFP_WEIGHT_BLOCK", "32", 1);
    setenv("BFP_WIRE_BLOCK", "16", 1);
    setenv("BFP_MANTISSA_BITS", "6", 1);
    setenv("BFP_EXPONENT_BITS", "8", 1);
    setenv("BFP_MATH", "fq", 1);
    setenv("BFP_GRADS", "1", 1);
    setenv("BFP_STATE", "1", 1);
    setenv("BFP_ROUNDING", "det", 1);
    bfpSweepConfig_t c;
    TEST_ASSERT_NULL(bfpSweepConfigFromEnv(&c));
    TEST_ASSERT_EQUAL_INT(GROUP_MODE_SIZE, c.weightMode);
    TEST_ASSERT_EQUAL_INT(32, c.weightSize);
    TEST_ASSERT_EQUAL_INT(WIRE_BLOCK_SIZE, c.wireMode);
    TEST_ASSERT_EQUAL_INT(16, c.wireSize);
    TEST_ASSERT_EQUAL_UINT8(6, c.mantissaBits);
    TEST_ASSERT_EQUAL_INT(BFP_MATH_FQ, c.math);
    TEST_ASSERT_TRUE(c.bfpGrads);
    TEST_ASSERT_TRUE(c.bfpState);
    TEST_ASSERT_EQUAL_INT(BFP_ROUNDING_DET, c.rounding);
    TEST_ASSERT_EQUAL_STRING("32", c.weightBlockStr);
    TEST_ASSERT_EQUAL_STRING("16", c.wireBlockStr);
    setenv("BFP_WEIGHT_BLOCK", "channel", 1);
    setenv("BFP_WIRE_BLOCK", "tensor", 1);
    TEST_ASSERT_NULL(bfpSweepConfigFromEnv(&c));
    TEST_ASSERT_EQUAL_INT(GROUP_MODE_CHANNEL, c.weightMode);
    TEST_ASSERT_EQUAL_INT(WIRE_BLOCK_TENSOR, c.wireMode);
    clearSweepEnv();
}

/* Every invalid value names ITS knob in the message (the trainer prints it). */
void testSweepConfigRejectsEachInvalidValue(void) {
    static const char *const bad[][2] = {
        {"BFP_MANTISSA_BITS", "1"}, {"BFP_MANTISSA_BITS", "17"},  {"BFP_MANTISSA_BITS", "x"},
        {"BFP_EXPONENT_BITS", "1"}, {"BFP_EXPONENT_BITS", "9"},   {"BFP_WEIGHT_BLOCK", "0"},
        {"BFP_WEIGHT_BLOCK", "-4"}, {"BFP_WEIGHT_BLOCK", "chan"}, {"BFP_WIRE_BLOCK", "0"},
        {"BFP_WIRE_BLOCK", "fp32"}, {"BFP_MATH", "fake"},         {"BFP_GRADS", "2"},
        {"BFP_STATE", "yes"},       {"BFP_ROUNDING", "sr_half"},
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        clearSweepEnv();
        setenv(bad[i][0], bad[i][1], 1);
        bfpSweepConfig_t c;
        const char *err = bfpSweepConfigFromEnv(&c);
        TEST_ASSERT_NOT_NULL_MESSAGE(err, bad[i][0]);
        TEST_ASSERT_NOT_NULL_MESSAGE(strstr(err, bad[i][0]), err);
    }
    clearSweepEnv();
}

void testSweepConfigStateRequiresGrads(void) {
    clearSweepEnv();
    setenv("BFP_STATE", "1", 1);
    bfpSweepConfig_t c;
    const char *err = bfpSweepConfigFromEnv(&c);
    TEST_ASSERT_NOT_NULL(err);
    TEST_ASSERT_NOT_NULL(strstr(err, "BFP_STATE"));
    TEST_ASSERT_NOT_NULL(strstr(err, "BFP_GRADS"));
    clearSweepEnv();
}

void testSweepConfigFlagsEachLegacyKnobWithoutFailing(void) {
    static const unsigned bits[] = {LEGACY_KNOB_SYM_BITS,      LEGACY_KNOB_SYM_WIRES,
                                    LEGACY_KNOB_WEIGHT_DTYPE,  LEGACY_KNOB_GROUP_MODE,
                                    LEGACY_KNOB_GROUP_SIZE,    LEGACY_KNOB_SYM_ROUNDING,
                                    LEGACY_KNOB_ODTS_ROUNDTRIP};
    for (size_t i = 0; i < sizeof(kLegacyKnobs) / sizeof(kLegacyKnobs[0]); i++) {
        clearSweepEnv();
        setenv(kLegacyKnobs[i], "1", 1);
        bfpSweepConfig_t c;
        TEST_ASSERT_NULL_MESSAGE(bfpSweepConfigFromEnv(&c), kLegacyKnobs[i]);
        TEST_ASSERT_EQUAL_UINT_MESSAGE(bits[i], c.ignoredLegacyKnobs, kLegacyKnobs[i]);
        TEST_ASSERT_EQUAL_UINT8(8, c.mantissaBits); /* nothing else changed */
    }
    clearSweepEnv();
}

/* ---- headroom predicates == the kernel guards (spec §10.4 pin) ------------ */

void testBlockHeadroomFitsMatchesShippedFormula(void) {
    /* m=12 equal widths: limit = INT32_MAX >> 22 = 511 products. */
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(12, 12, 32, 0, 63)); /* seg = min(32, 63) = 32 */
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(12, 12, 0, 0, 511)); /* per-tensor, K = 511 */
    TEST_ASSERT_FALSE(bfpBlockHeadroomFits(12, 12, 0, 0, 512));
    /* m=14: limit = INT32_MAX >> 26 = 31 -> a 32-block over a 63-run trips. */
    TEST_ASSERT_FALSE(bfpBlockHeadroomFits(14, 14, 32, 0, 63));
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(14, 14, 16, 0, 63));
    /* m=16: limit 1 -- only a 1-element segment fits. */
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(16, 16, 1, 0, 63));
    TEST_ASSERT_FALSE(bfpBlockHeadroomFits(16, 16, 2, 0, 63));
    /* The whole stage-1 grid (m <= 8) is safe at every HAR reduction. */
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(8, 8, 64, 64, 96));
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(12, 12, 600, 0, 511)); /* run > K: clamped to K */
    TEST_ASSERT_FALSE(bfpBlockHeadroomFits(12, 12, 600, 0, 512));
    /* runB == 0 substitutes reductionLen for b, so maxSeg = min(a, b) can never
     * exceed reductionLen there -- the clamp is dead code unless BOTH runs are
     * nonzero and exceed reductionLen. Only this shape actually exercises it. */
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(12, 12, 600, 600, 511)); /* both runs > K: clamped to K */
}

void testSumHeadroomFitsMatchesShippedFormula(void) {
    size_t limit8 = bfpSumSegmentLimit(8);
    TEST_ASSERT_TRUE(bfpSumHeadroomFits(8, 0, 32)); /* AvgPool K=32 */
    TEST_ASSERT_TRUE(bfpSumHeadroomFits(8, 0, limit8));
    TEST_ASSERT_FALSE(bfpSumHeadroomFits(8, 0, limit8 + 1));
    TEST_ASSERT_TRUE(bfpSumHeadroomFits(16, 16, 1u << 20)); /* run caps the segment */
}

/* Predicate and guard must agree at the boundary from both sides. */
void testBlockHeadroomPredicateAgreesWithKernelGuard(void) {
    uint8_t expA[2] = {0, 0}, expB[1] = {0};
    bfpQConfig_t a = {.exponents = expA,
                      .numGroups = 2,
                      .groupSize = 32,
                      .roundingMode = HALF_AWAY,
                      .mantissaBits = 14,
                      .exponentBits = 8};
    bfpQConfig_t b = {.exponents = expB,
                      .numGroups = 1,
                      .groupSize = 0,
                      .roundingMode = HALF_AWAY,
                      .mantissaBits = 14,
                      .exponentBits = 8};
    TEST_ASSERT_FALSE(bfpBlockHeadroomFits(14, 14, 32, 0, 63));
    ASSERT_EXITS_WITH_FAILURE(bfpValidateBlockHeadroom(&a, &b, 63, "pin"));
    a.mantissaBits = 12;
    b.mantissaBits = 12;
    TEST_ASSERT_TRUE(bfpBlockHeadroomFits(12, 12, 32, 0, 63));
    bfpValidateBlockHeadroom(&a, &b, 63, "pin"); /* must NOT exit */
}

/* Sum twin, same discipline: predicate and guard must agree at the boundary. */
void testSumHeadroomPredicateAgreesWithKernelGuard(void) {
    uint8_t exps[1] = {0};
    size_t limit8 = bfpSumSegmentLimit(8);
    bfpQConfig_t q = {.exponents = exps,
                      .numGroups = 1,
                      .groupSize = 0,
                      .roundingMode = HALF_AWAY,
                      .mantissaBits = 8,
                      .exponentBits = 8};
    TEST_ASSERT_FALSE(bfpSumHeadroomFits(8, 0, limit8 + 1));
    ASSERT_EXITS_WITH_FAILURE(bfpValidateSumHeadroom(&q, limit8 + 1, "pin"));
    TEST_ASSERT_TRUE(bfpSumHeadroomFits(8, 0, limit8));
    bfpValidateSumHeadroom(&q, limit8, "pin"); /* must NOT exit */
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testResolveGroupShapeTensorModeIsPerTensor);
    RUN_TEST(testResolveGroupShapeChannelModeOneGroupPerOutCh);
    RUN_TEST(testResolveGroupShapeSizeModeDivides);
    RUN_TEST(testResolveGroupShapeSizeModeFallsBackToChannel);
    RUN_TEST(testResolveGroupShapeCollapsesSingleGroupToPerTensor);
    RUN_TEST(testResolveGroupShapeRejectsZeroOutCh);
    RUN_TEST(testResolveGroupShapeRejectsEmptyTensor);
    RUN_TEST(testResolveGroupShapeRejectsNonDividingOutCh);
    RUN_TEST(testResolveGroupShapeGuardIsModeIndependent);
    RUN_TEST(testResolveWireShapeDivisorGroups);
    RUN_TEST(testResolveWireShapeHeadWireFallsBackToPerTensor);
    RUN_TEST(testResolveWireShapeSizeEqualToWireNormalizesToPerTensor);
    RUN_TEST(testResolveWireShapeTensorAndFloatModesArePerTensor);
    RUN_TEST(testResolveWireShapeForwardAndDxResolveIndependently);
    RUN_TEST(testResolveWireShapeRejectsEmptyWire);
    RUN_TEST(testResolveWireShapeRejectsNonPositiveSize);
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
    RUN_TEST(testGateFloat32ExpectationIgnoresBitsAndShape);
    RUN_TEST(testGateUnsupportedExpectationFailsFast);
    RUN_TEST(testGateUnsupportedExpectationFailsFastOnTypeMismatch);
    RUN_TEST(testPackedPayloadBytesRoundsUpPerTensor);
    RUN_TEST(testPackedPayloadBytesMatchesFrameworkAccounting);
    RUN_TEST(testPackedMetadataBytesPerDtype);
    RUN_TEST(testPackedBytesRejectUnsupportedDtype);
    RUN_TEST(testSweepConfigDefaults);
    RUN_TEST(testSweepConfigParsesTheAnchorConfig);
    RUN_TEST(testSweepConfigRejectsEachInvalidValue);
    RUN_TEST(testSweepConfigStateRequiresGrads);
    RUN_TEST(testSweepConfigFlagsEachLegacyKnobWithoutFailing);
    RUN_TEST(testBlockHeadroomFitsMatchesShippedFormula);
    RUN_TEST(testSumHeadroomFitsMatchesShippedFormula);
    RUN_TEST(testBlockHeadroomPredicateAgreesWithKernelGuard);
    RUN_TEST(testSumHeadroomPredicateAgreesWithKernelGuard);
    return UNITY_END();
}
