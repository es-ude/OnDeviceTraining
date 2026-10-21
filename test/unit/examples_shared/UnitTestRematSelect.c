#define SOURCE_FILE "UNIT_TEST_REMAT_SELECT"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "DeathTest.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Quantization.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "remat_select.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

#define COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

/* HAR's layer types (examples/har_classifier/train_c.c buildModel): the
 * presets read nothing else. */
#define HAR_N 12
static const layerType_t g_harTypes[HAR_N] = {CONV1D, RELU, MAXPOOL1D, CONV1D,  RELU,   MAXPOOL1D,
                                              CONV1D, RELU, AVGPOOL1D, FLATTEN, LINEAR, SOFTMAX};

static void setOnlyThePlan(const char *plan) {
    unsetenv("REMAT_STORAGE");
    unsetenv("DRY_PLAN");
    setenv("REMAT_PLAN", plan, 1);
}

static rematSelection_t selectFor(const char *plan, size_t modelSize) {
    setOnlyThePlan(plan);
    rematSelection_t sel;
    TEST_ASSERT_TRUE_MESSAGE(rematSelectFromEnv(&sel, modelSize), plan);
    return sel;
}

/* What a print writes, read back through a temporary file. */
static char g_printed[512];

static const char *readBack(FILE *f) {
    rewind(f);
    size_t got = fread(g_printed, 1u, sizeof g_printed - 1u, f);
    g_printed[got] = '\0';
    fclose(f);
    return g_printed;
}

static const char *configKeysOf(const rematSelection_t *sel) {
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    rematSelectPrintConfigKeys(f, sel);
    return readBack(f);
}

static void assertKeeps(const bool *mask, size_t n, const size_t *kept, size_t numKept,
                        const char *what) {
    /* From ACT 0: the input is always present, so no plan keeps it. */
    for (size_t j = 0; j <= n; j++) {
        bool expected = false;
        for (size_t i = 0; i < numKept; i++) {
            expected = expected || kept[i] == j;
        }
        char message[64];
        (void)snprintf(message, sizeof message, "%s: ACT %zu", what, j);
        TEST_ASSERT_EQUAL_INT_MESSAGE(expected, mask[j], message);
    }
}

/* retain:3,7 on HAR keeps ACT 3 and ACT 7, and ACT 12, which the loss reads
 * right after the last forward; the log names all three. */
void testARetainListKeepsItsActsAndTheLastOne(void) {
    rematSelection_t sel = selectFor("retain:3,7", HAR_N);
    TEST_ASSERT_EQUAL_INT(REMAT_PLAN_RETAIN_LIST, sel.plan.policy);
    TEST_ASSERT_NULL(sel.plan.retain);
    bool mask[HAR_N + 1];
    rematSelectRetainMask(&sel, NULL, HAR_N, mask);
    assertKeeps(mask, HAR_N, (const size_t[]){3, 7, 12}, 3, "retain:3,7");
    TEST_ASSERT_EQUAL_STRING(", \"remat_storage\": \"heap\", \"remat_plan\": \"retain:3,7,12\"",
                             configKeysOf(&sel));
}

/* retain:sqrt on 12 layers: k = ceil(sqrt(12)) = 4 keeps ACT 4 and ACT 8 (a
 * floor would give 3: ACT 3, 6, 9); on 16 layers k = 4 keeps ACT 4, 8, 12.
 * retain:speed keeps HAR's three Conv1d outputs (ACT 1, 4, 7) and its Linear
 * output (ACT 11), and a Conv1dTransposed output too (HAR has none). ACT n
 * always. */
void testTheRetainPresetsResolveAgainstTheModel(void) {
    layer_t layers[HAR_N];
    layer_t *model[HAR_N];
    for (size_t l = 0; l < HAR_N; l++) {
        layers[l] = (layer_t){.type = g_harTypes[l], .config = NULL};
        model[l] = &layers[l];
    }
    bool mask[17];
    rematSelection_t sqrt12 = selectFor("retain:sqrt", HAR_N);
    rematSelectRetainMask(&sqrt12, model, HAR_N, mask);
    assertKeeps(mask, HAR_N, (const size_t[]){4, 8, 12}, 3, "retain:sqrt, n = 12");
    rematSelection_t sqrt16 = selectFor("retain:sqrt", 16);
    rematSelectRetainMask(&sqrt16, NULL, 16, mask);
    assertKeeps(mask, 16, (const size_t[]){4, 8, 12, 16}, 4, "retain:sqrt, n = 16");
    rematSelection_t speed = selectFor("retain:speed", HAR_N);
    rematSelectRetainMask(&speed, model, HAR_N, mask);
    assertKeeps(mask, HAR_N, (const size_t[]){1, 4, 7, 11, 12}, 5, "retain:speed");
    static const layerType_t convTTypes[4] = {CONV1D, RELU, CONV1D_TRANSPOSED, RELU};
    for (size_t l = 0; l < COUNT_OF(convTTypes); l++) {
        layers[l] = (layer_t){.type = convTTypes[l], .config = NULL};
    }
    rematSelection_t speedConvT = selectFor("retain:speed", COUNT_OF(convTTypes));
    rematSelectRetainMask(&speedConvT, model, COUNT_OF(convTTypes), mask);
    assertKeeps(mask, COUNT_OF(convTTypes), (const size_t[]){1, 3, 4}, 3,
                "retain:speed, Conv1dTransposed");
    TEST_ASSERT_EQUAL_STRING(", \"remat_storage\": \"heap\", \"remat_plan\": \"retain:sqrt\"",
                             configKeysOf(&sqrt12));
    TEST_ASSERT_EQUAL_STRING(", \"remat_storage\": \"heap\", \"remat_plan\": \"retain:speed\"",
                             configKeysOf(&speed));
}

void testTheOtherPlansKeepTheirNames(void) {
    rematSelection_t storeAll = selectFor("store_all", HAR_N);
    rematSelection_t liveness = selectFor("liveness", HAR_N);
    TEST_ASSERT_EQUAL_STRING(", \"remat_storage\": \"heap\", \"remat_plan\": \"store_all\"",
                             configKeysOf(&storeAll));
    TEST_ASSERT_EQUAL_STRING(", \"remat_storage\": \"heap\", \"remat_plan\": \"liveness\"",
                             configKeysOf(&liveness));
}

/* The child reports through stderr; the death harness reads stdout. */
static void exitWithTheVerdictOn(const char *plan, size_t modelSize) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    setOnlyThePlan(plan);
    rematSelection_t sel;
    exit(rematSelectFromEnv(&sel, modelSize) ? 0 : 1);
}

static void assertStopsNamingTheAllowedValues(const char *plan) {
    char expected[224];
    (void)snprintf(expected, sizeof expected,
                   "ERROR: REMAT_PLAN=%s (expected store_all|liveness|retain:<j>[,<j>...]|"
                   "retain:sqrt|retain:speed, optionally followed by +inplace, 1 <= j <= 12 "
                   "ascending)",
                   plan);
    ASSERT_EXITS_WITH_OUTPUT(1, expected, exitWithTheVerdictOn(plan, HAR_N));
}

/* A mistyped list stops before any data is loaded, naming the value and the
 * allowed forms; it never selects a different plan. */
void testAMistypedRetainListStopsBeforeTraining(void) {
    static const char *const mistyped[] = {
        "retain:7,3",  "retain:0",   "retain:13",     "retain:",
        "retain:3,,7", "retain:3,3", "retain:3,7,",   "retain:x",
        "retain:-1",   "retain: 3",  "retain:sqrt,3", "retain:99999999999999999999999",
        "retain"};
    for (size_t i = 0; i < COUNT_OF(mistyped); i++) {
        assertStopsNamingTheAllowedValues(mistyped[i]);
    }
}

/* "+inplace" after any plan asks for in-place gradients on top of it: the
 * plan and its mask stay as without the suffix, and the run log names both. */
void testTheInPlaceSuffixParsesOnEveryPlan(void) {
    static const char *const plans[] = {"store_all", "liveness", "retain:3,7", "retain:sqrt",
                                        "retain:speed"};
    static const char *const logged[] = {"store_all", "liveness", "retain:3,7,12", "retain:sqrt",
                                         "retain:speed"};
    for (size_t i = 0; i < COUNT_OF(plans); i++) {
        char value[32];
        (void)snprintf(value, sizeof value, "%s+inplace", plans[i]);
        rematSelection_t without = selectFor(plans[i], HAR_N);
        rematSelection_t with = selectFor(value, HAR_N);
        TEST_ASSERT_FALSE_MESSAGE(without.plan.inPlaceDx, plans[i]);
        TEST_ASSERT_TRUE_MESSAGE(with.plan.inPlaceDx, value);
        TEST_ASSERT_EQUAL_INT_MESSAGE(without.plan.policy, with.plan.policy, value);
        TEST_ASSERT_EQUAL_INT_MESSAGE(without.retainForm, with.retainForm, value);
        TEST_ASSERT_EQUAL_MEMORY_MESSAGE(without.retain, with.retain, sizeof with.retain, value);
        char expected[96];
        (void)snprintf(expected, sizeof expected,
                       ", \"remat_storage\": \"heap\", \"remat_plan\": \"%s+inplace\"", logged[i]);
        TEST_ASSERT_EQUAL_STRING_MESSAGE(expected, configKeysOf(&with), value);
    }
}

/* A mistyped suffix stops like a mistyped plan: never a plan without the
 * sharing that was asked for, nor one with sharing that was not. */
void testAMistypedInPlaceSuffixStopsBeforeTraining(void) {
    static const char *const mistyped[] = {
        "+inplace",         "liveness+inplace+inplace", "liveness+in",          "LIVENESS+INPLACE",
        "liveness+INPLACE", "retain:7,3+inplace",       "retain:+inplace",      "liveness+",
        "inplace",          "liveness +inplace",        "retain:sqrt+inplace,3"};
    for (size_t i = 0; i < COUNT_OF(mistyped); i++) {
        assertStopsNamingTheAllowedValues(mistyped[i]);
    }
}

void testARetainPlanNeedsAModelOfAtMost64Layers(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "ERROR: REMAT_PLAN=retain:3 needs a model of at most 64 layers, this "
                             "one has 65",
                             exitWithTheVerdictOn("retain:3", 65u));
    ASSERT_EXITS_WITH(0, exitWithTheVerdictOn("retain:3", 64u));
}

/* A zeroed scheduler reports nothing, so the line shows the format alone:
 * the plan as selected, and the recompute numbers at the end. */
void testTheDryPlanLineNamesThePlanAndEndsWithTheRecomputeNumbers(void) {
    rematSelection_t sel = selectFor("retain:3,7", HAR_N);
    rematScheduler_t s = {0};
    FILE *f = tmpfile();
    TEST_ASSERT_NOT_NULL(f);
    rematSelectPrintPlan(f, "har_classifier", &sel, &s);
    TEST_ASSERT_EQUAL_STRING(
        "DRY_PLAN example=har_classifier storage=arena plan=retain:3,7,12 planned=0 placed=0 "
        "data_reserved=0 steps=0 wires_peak_b=0 activations_peak_b=0 arena_b=0 arena_pad_b=0 "
        "arena_gap_b=0 wire_metadata_b=0 refwd_steps=0 forward_cost=0 recompute_cost=0\n",
        readBack(f));
}

static void exitWithTheInitVerdictForElevenLayers(void) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    rematSelection_t sel = selectFor("retain:3", HAR_N);
    rematScheduler_t s = {0};
    exit(rematSelectInit(&s, &sel, "har_classifier", NULL, 11u, (lossConfig_t){0}, NULL) ? 0 : 1);
}

/* A list checked against 12 layers must not build a plan for 11. */
void testInitRefusesARetainPlanReadForAnotherLayerCount(void) {
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "ERROR: rematSelectInit: REMAT_PLAN was read for 12 layers, the model "
                             "has 11",
                             exitWithTheInitVerdictForElevenLayers());
}

/* rematSelectInit hands the selection's plan to the scheduler: "+inplace"
 * reaches the plan as its mark array, the plain value as none. One Flatten
 * (it has no config) over a stack input is enough: the plan reserves its
 * marks whenever the spec asks for them, pair or not. */
static void assertInitMarksExactlyWithTheSuffix(const char *plan, bool withSuffix) {
    rematSelection_t sel = selectFor(plan, 1u);
    layer_t flatten = {.type = FLATTEN, .config = NULL};
    layer_t *model[1] = {&flatten};
    quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    float x[4] = {0};
    size_t dims[] = {1, 4};
    size_t order[] = {0, 1};
    shape_t shape = {.numberOfDimensions = 2, .dimensions = dims, .orderOfDimensions = order};
    tensor_t input;
    setTensorValues(&input, (uint8_t *)x, &shape, &floatQ, NULL);
    rematScheduler_t s;
    TEST_ASSERT_TRUE_MESSAGE(
        rematSelectInit(&s, &sel, "unit", model, 1u, defaultLossConfig(MSE), &input), plan);
    TEST_ASSERT_EQUAL_MESSAGE(withSuffix, s.plan->inPlace != NULL, plan);
    rematSchedulerDeinit(&s);
}

void testInitPassesTheInPlaceSuffixToThePlan(void) {
    assertInitMarksExactlyWithTheSuffix("liveness", false);
    assertInitMarksExactlyWithTheSuffix("liveness+inplace", true);
    assertInitMarksExactlyWithTheSuffix("store_all+inplace", true);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testARetainListKeepsItsActsAndTheLastOne);
    RUN_TEST(testTheRetainPresetsResolveAgainstTheModel);
    RUN_TEST(testTheOtherPlansKeepTheirNames);
    RUN_TEST(testAMistypedRetainListStopsBeforeTraining);
    RUN_TEST(testTheInPlaceSuffixParsesOnEveryPlan);
    RUN_TEST(testAMistypedInPlaceSuffixStopsBeforeTraining);
    RUN_TEST(testInitPassesTheInPlaceSuffixToThePlan);
    RUN_TEST(testARetainPlanNeedsAModelOfAtMost64Layers);
    RUN_TEST(testTheDryPlanLineNamesThePlanAndEndsWithTheRecomputeNumbers);
    RUN_TEST(testInitRefusesARetainPlanReadForAnotherLayerCount);
    return UNITY_END();
}
