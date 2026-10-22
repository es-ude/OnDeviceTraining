#define SOURCE_FILE "UNIT_TEST_REMAT_SELECT"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "Conv1dApi.h"
#include "DeathTest.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "LinearApi.h"
#include "LossFunction.h"
#include "Pool1dApi.h"
#include "Quantization.h"
#include "RNG.h"
#include "ReluApi.h"
#include "RematPlan.h"
#include "RematScheduler.h"
#include "SoftmaxApi.h"
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

/* Sets exactly the given variables; NULL unsets one. */
static void setTheVariables(const char *storage, const char *plan, const char *dry,
                            const char *dump, const char *budget) {
    static const char *const names[] = {"REMAT_STORAGE", "REMAT_PLAN", "DRY_PLAN", "DUMP_PROBLEM",
                                        "REMAT_BUDGET_B"};
    const char *const values[] = {storage, plan, dry, dump, budget};
    for (size_t i = 0; i < COUNT_OF(names); i++) {
        if (values[i] == NULL) {
            unsetenv(names[i]);
        } else {
            setenv(names[i], values[i], 1);
        }
    }
}

static void setOnlyThePlan(const char *plan) {
    setTheVariables(NULL, plan, NULL, NULL, NULL);
}

static rematSelection_t selectFor(const char *plan, size_t modelSize) {
    setOnlyThePlan(plan);
    rematSelection_t sel;
    TEST_ASSERT_TRUE_MESSAGE(rematSelectFromEnv(&sel, modelSize), plan);
    return sel;
}

/* What a print writes, read back through a temporary file. */
static char g_printed[4096];
static char g_hashLine[64];

static const char *readInto(FILE *f, char *text, size_t size) {
    rewind(f);
    size_t got = fread(text, 1u, size - 1u, f);
    text[got] = '\0';
    fclose(f);
    return text;
}

static const char *readBack(FILE *f) {
    return readInto(f, g_printed, sizeof g_printed);
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

static rematSelection_t selectWith(const char *storage, const char *plan, const char *dry,
                                   const char *dump, const char *budget) {
    setTheVariables(storage, plan, dry, dump, budget);
    rematSelection_t sel;
    TEST_ASSERT_TRUE(rematSelectFromEnv(&sel, HAR_N));
    return sel;
}

static void exitWithTheVerdictOnTheVariables(const char *storage, const char *plan, const char *dry,
                                             const char *dump, const char *budget) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    setTheVariables(storage, plan, dry, dump, budget);
    rematSelection_t sel;
    exit(rematSelectFromEnv(&sel, HAR_N) ? 0 : 1);
}

/* DUMP_PROBLEM=1 alone asks for the problem file: the trainer reaches its
 * remat branch (chosen), and no plan is printed or checked. */
void testDumpProblemAloneAsksForTheProblem(void) {
    rematSelection_t sel = selectWith(NULL, NULL, NULL, "1", NULL);
    TEST_ASSERT_TRUE(sel.dumpProblem);
    TEST_ASSERT_TRUE(sel.chosen);
    TEST_ASSERT_FALSE(sel.dryPlan);
    TEST_ASSERT_EQUAL_size_t(0, sel.budgetBytes);
    rematSelection_t none = selectWith(NULL, NULL, NULL, NULL, NULL);
    TEST_ASSERT_FALSE(none.dumpProblem);
    TEST_ASSERT_FALSE(none.chosen);
    TEST_ASSERT_EQUAL_size_t(0, none.budgetBytes);
}

/* The budget is a byte count up to SIZE_MAX, read digit by digit. */
void testABudgetIsReadInBytes(void) {
    rematSelection_t sel = selectWith("arena", "retain:4,7,10", NULL, NULL, "41216");
    TEST_ASSERT_EQUAL_size_t(41216, sel.budgetBytes);
    TEST_ASSERT_TRUE(sel.chosen);
    char largest[32];
    (void)snprintf(largest, sizeof largest, "%zu", (size_t)SIZE_MAX);
    TEST_ASSERT_EQUAL_size_t(SIZE_MAX, selectWith("arena", NULL, NULL, NULL, largest).budgetBytes);
}

/* A run with a budget logs it after the plan, with the unit of its cost
 * numbers ("mac": the multiply-accumulate count of layerForwardCost); a run
 * without one logs neither key (the pins above). */
void testABudgetIsLoggedWithItsCostModel(void) {
    rematSelection_t sel = selectWith("arena", "retain:4,7,10", NULL, NULL, "41216");
    TEST_ASSERT_EQUAL_STRING(
        ", \"remat_storage\": \"arena\", \"remat_plan\": \"retain:4,7,10,12\", "
        "\"wire_budget_b\": 41216, \"wire_cost_model\": \"mac\"",
        configKeysOf(&sel));
}

/* Not a byte count in 1..SIZE_MAX: the run stops before any data is loaded,
 * naming the value and the allowed range; it never runs without the check. */
static void assertTheBudgetStops(const char *budget) {
    char expected[160];
    (void)snprintf(expected, sizeof expected,
                   "ERROR: REMAT_BUDGET_B=%s (expected the arena budget in bytes: decimal "
                   "digits, 1 <= value <= %zu)",
                   budget, (size_t)SIZE_MAX);
    ASSERT_EXITS_WITH_OUTPUT(1, expected,
                             exitWithTheVerdictOnTheVariables("arena", NULL, NULL, NULL, budget));
}

void testAMalformedBudgetStopsBeforeTraining(void) {
    static const char *const malformed[] = {"0",  "00",  "-1",  "+5",   " 5",
                                            "5 ", "12a", "abc", "0x10", "99999999999999999999999"};
    for (size_t i = 0; i < COUNT_OF(malformed); i++) {
        assertTheBudgetStops(malformed[i]);
    }
    char tenTimesLargest[32];
    (void)snprintf(tenTimesLargest, sizeof tenTimesLargest, "%zu0", (size_t)SIZE_MAX);
    assertTheBudgetStops(tenTimesLargest);
}

/* Heap storage has no arena (it reports 0 B), so a budget would pass for
 * nothing: it needs REMAT_STORAGE=arena, also when the storage is left at
 * its heap default. */
void testABudgetNeedsArenaStorage(void) {
    static const char message[] = "ERROR: REMAT_BUDGET_B needs REMAT_STORAGE=arena (the budget "
                                  "bounds the arena; heap storage has none)";
    ASSERT_EXITS_WITH_OUTPUT(1, message,
                             exitWithTheVerdictOnTheVariables(NULL, NULL, NULL, NULL, "4096"));
    ASSERT_EXITS_WITH_OUTPUT(1, message,
                             exitWithTheVerdictOnTheVariables("heap", NULL, NULL, NULL, "4096"));
    ASSERT_EXITS_WITH(0, exitWithTheVerdictOnTheVariables("arena", NULL, NULL, NULL, "4096"));
}

/* The problem does not depend on the storage, the plan or the budget, and
 * the dump prints no plan: any of them next to DUMP_PROBLEM=1 is a mistake,
 * named, never ignored. A value other than 1 is named too. */
void testDumpProblemStandsAlone(void) {
    static const char alone[] = "ERROR: DUMP_PROBLEM=1 stands alone: it writes the problem, "
                                "which no storage, plan or budget changes, and stops (unset "
                                "REMAT_STORAGE, REMAT_PLAN, DRY_PLAN and REMAT_BUDGET_B)";
    ASSERT_EXITS_WITH_OUTPUT(1, alone,
                             exitWithTheVerdictOnTheVariables("arena", NULL, NULL, "1", NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, alone,
                             exitWithTheVerdictOnTheVariables(NULL, "liveness", NULL, "1", NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, alone,
                             exitWithTheVerdictOnTheVariables(NULL, NULL, "1", "1", NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, alone,
                             exitWithTheVerdictOnTheVariables("arena", NULL, NULL, "1", "4096"));
    ASSERT_EXITS_WITH_OUTPUT(1, "ERROR: DUMP_PROBLEM=2 (expected 1)",
                             exitWithTheVerdictOnTheVariables(NULL, NULL, NULL, "2", NULL));
    ASSERT_EXITS_WITH_OUTPUT(1, "ERROR: DUMP_PROBLEM=yes (expected 1)",
                             exitWithTheVerdictOnTheVariables(NULL, NULL, NULL, "yes", NULL));
}

static uint64_t hashOf(const char *text) {
    return rematSelectProblemHash(REMAT_PROBLEM_HASH_BASIS, text, strlen(text));
}

/* FNV-1a 64 (offset basis 0xcbf29ce484222325, prime 0x100000001b3): the
 * published test vectors, and a hash continued over a second part equals the
 * hash of the whole. */
void testTheProblemHashIsFnv1a64(void) {
    TEST_ASSERT_EQUAL_HEX64(0xcbf29ce484222325u, hashOf(""));
    TEST_ASSERT_EQUAL_HEX64(0xaf63dc4c8601ec8cu, hashOf("a"));
    TEST_ASSERT_EQUAL_HEX64(0x85944171f73967e8u, hashOf("foobar"));
    TEST_ASSERT_EQUAL_HEX64(hashOf("foobar"), rematSelectProblemHash(hashOf("foo"), "bar", 3u));
}

/* examples/har_classifier/train_c.c buildModel; freezeConvs gives the
 * stage-2 model of train_c_finetune.c. The dump reads no weight value, so one
 * init key serves every layer. */
#define HAR_IN_LENGTH 128u
static void buildHar(layer_t **model, bool freezeConvs) {
    static quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &floatQ);
    const rngKey_t key = rngKeyFromSeed(1u);
    const trainable_t conv = freezeConvs ? TRAINABLE_FALSE : TRAINABLE_DEFAULT;
    static const size_t channels[4] = {9, 16, 32, 64};
    static const size_t kernels[3] = {7, 5, 3};
    for (size_t b = 0; b < 3u; b++) {
        model[3u * b] = conv1dLayerInit(&(conv1dInit_t){.inChannels = channels[b],
                                                        .outChannels = channels[b + 1u],
                                                        .kernelSize = kernels[b],
                                                        .padding = SAME,
                                                        .trainable = conv,
                                                        .weightInit.key = &key},
                                        &lq);
        model[3u * b + 1u] = reluLayerInit(&lq);
    }
    model[2] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = 16, .inputLength = HAR_IN_LENGTH},
        &lq);
    model[5] = maxPool1dLayerInit(
        &(maxPool1dInit_t){
            .kernelSize = 2, .stride = 2, .inputChannels = 32, .inputLength = HAR_IN_LENGTH / 2u},
        &lq);
    model[8] = avgPool1dLayerInit(
        &(avgPool1dInit_t){.kernelSize = HAR_IN_LENGTH / 4u, .stride = HAR_IN_LENGTH / 4u}, &lq);
    model[9] = flattenLayerInit();
    model[10] = linearLayerInit(
        &(linearInit_t){.inFeatures = 64, .outFeatures = 6, .weightInit.key = &key}, &lq);
    model[11] = softmaxLayerInit(&lq);
}

static void freeHar(layer_t **model) {
    for (size_t b = 0; b < 3u; b++) {
        freeConv1dLayer(model[3u * b]);
        freeReluLayer(model[3u * b + 1u]);
    }
    freeMaxPool1dLayer(model[2]);
    freeMaxPool1dLayer(model[5]);
    freeAvgPool1dLayer(model[8]);
    freeFlattenLayer(model[9]);
    freeLinearLayer(model[10]);
    freeSoftmaxLayer(model[11]);
}

/* A [1, ...] input header; the dump reads its shape and dtype, never data. */
typedef struct inputHeader {
    size_t dims[9];
    size_t order[9];
    shape_t shape;
    tensor_t tensor;
} inputHeader_t;

static const tensor_t *makeInputHeader(inputHeader_t *h, const size_t *dims, size_t rank,
                                       quantization_t *q) {
    for (size_t d = 0; d < rank; d++) {
        h->dims[d] = dims[d];
        h->order[d] = d;
    }
    h->shape =
        (shape_t){.numberOfDimensions = rank, .dimensions = h->dims, .orderOfDimensions = h->order};
    h->tensor = (tensor_t){.data = NULL, .shape = &h->shape, .quantization = q, .sparsity = NULL};
    return &h->tensor;
}

static const lossConfig_t g_crossEntropy = {
    .funcType = CROSS_ENTROPY, .backwardReduction = REDUCTION_MEAN, .classWeights = NULL};

/* Dumps into g_printed (the JSON) and g_hashLine. */
static bool dumpOf(layer_t **model, size_t n, lossConfig_t loss, const tensor_t *input) {
    FILE *json = tmpfile();
    FILE *hashLine = tmpfile();
    TEST_ASSERT_NOT_NULL(json);
    TEST_ASSERT_NOT_NULL(hashLine);
    bool dumped = rematSelectDumpProblem(json, hashLine, model, n, loss, input);
    (void)readInto(json, g_printed, sizeof g_printed);
    (void)readInto(hashLine, g_hashLine, sizeof g_hashLine);
    return dumped;
}

/* HAR's problem line, spelled out by layer (one JSON line, its keys in a
 * fixed order). Shapes at batch 1: the SAME convs keep the length, each
 * MaxPool halves it, the AvgPool window 32 leaves 1, Flatten gives [1, 64].
 * macPerElement = Cin * K (9 * 7, 16 * 5, 32 * 3) and inF (64); kernel = the
 * pool window. Derived a second time, with both hashes (this line, then the
 * same model with its convs frozen), by
 * uv run python -c "import json, functools; [print(s, end=''
 *     ) or print(format(functools.reduce(lambda h, b: ((h ^ b) * 0x100000001b3) % 2**64,
 *     s.encode(), 0xcbf29ce484222325), '016x')) for s in [json.dumps({'schema_version': 0,
 *     'input': {'shape': [1, 9, 128], 'dtype': {'kind': 'FLOAT32'}}, 'loss': 'CROSS_ENTROPY',
 *     'layers': [dict({'type': t, 'frozen': frozen and t == 'CONV1D', 'trackRunningStats': False,
 *     'outShape': [1] + o, 'outputDtype': {'kind': 'FLOAT32'}, 'propLossDtype': {'kind':
 *     'FLOAT32'}}, **({'cost': {c[0]: c[1]}} if c else {})) for t, o, *c in [('CONV1D', [16,
 *     128], 'macPerElement', 63), ('RELU', [16, 128]), ('MAXPOOL1D', [16, 64], 'kernel', 2), (
 *     'CONV1D', [32, 64], 'macPerElement', 80), ('RELU', [32, 64]), ('MAXPOOL1D', [32, 32],
 *     'kernel', 2), ('CONV1D', [64, 32], 'macPerElement', 96), ('RELU', [64, 32]), ('AVGPOOL1D',
 *     [64, 1], 'kernel', 32), ('FLATTEN', [64]), ('LINEAR', [6], 'macPerElement', 64), (
 *     'SOFTMAX', [6])]]}, separators=(',', ':')) + '\n' for frozen in (False, True)]]"
 */
static const char g_harProblem[] =
    "{\"schema_version\":0,\"input\":{\"shape\":[1,9,128],\"dtype\":{\"kind\":\"FLOAT32\"}},"
    "\"loss\":\"CROSS_ENTROPY\",\"layers\":["
    "{\"type\":\"CONV1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,16,128],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"macPerElement\":63}},"
    "{\"type\":\"RELU\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,16,128],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}},"
    "{\"type\":\"MAXPOOL1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,16,64],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"kernel\":2}},"
    "{\"type\":\"CONV1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,32,64],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"macPerElement\":80}},"
    "{\"type\":\"RELU\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,32,64],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}},"
    "{\"type\":\"MAXPOOL1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,32,32],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"kernel\":2}},"
    "{\"type\":\"CONV1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,64,32],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"macPerElement\":96}},"
    "{\"type\":\"RELU\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,64,32],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}},"
    "{\"type\":\"AVGPOOL1D\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,64,1],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"kernel\":32}},"
    "{\"type\":\"FLATTEN\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,64],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}},"
    "{\"type\":\"LINEAR\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,6],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"},"
    "\"cost\":{\"macPerElement\":64}},"
    "{\"type\":\"SOFTMAX\",\"frozen\":false,\"trackRunningStats\":false,\"outShape\":[1,6],"
    "\"outputDtype\":{\"kind\":\"FLOAT32\"},\"propLossDtype\":{\"kind\":\"FLOAT32\"}}"
    "]}\n";

/* The input every HAR trainer keys its plan to: one [1, 9, 128] sample. */
static const tensor_t *harInput(inputHeader_t *in) {
    static quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    return makeInputHeader(in, (const size_t[]){1, 9, HAR_IN_LENGTH}, 3u, &floatQ);
}

/* The whole problem line and its hash; the HAR trainer's DUMP_PROBLEM=1
 * writes the same bytes. */
void testTheProblemOfHarIsItsWholeModel(void) {
    layer_t *model[HAR_N];
    buildHar(model, false);
    inputHeader_t in;
    TEST_ASSERT_TRUE(dumpOf(model, HAR_N, g_crossEntropy, harInput(&in)));
    TEST_ASSERT_EQUAL_STRING(g_harProblem, g_printed);
    TEST_ASSERT_EQUAL_STRING("PROBLEM_HASH fnv1a64=26b4a7babc5f10bf\n", g_hashLine);
    freeHar(model);
}

/* Stage 2 of the finetune trainer: HAR's line with its three convs marked
 * frozen ("true" is one byte shorter than "false"); the hash, derived from
 * that line, covers every other byte. */
void testTheProblemOfTheFinetuneModelMarksItsFrozenConvs(void) {
    layer_t *model[HAR_N];
    buildHar(model, true);
    inputHeader_t in;
    TEST_ASSERT_TRUE(dumpOf(model, HAR_N, g_crossEntropy, harInput(&in)));
    TEST_ASSERT_EQUAL_STRING("PROBLEM_HASH fnv1a64=2eda621ed208752a\n", g_hashLine);
    TEST_ASSERT_EQUAL_size_t(sizeof g_harProblem - 1u - 3u, strlen(g_printed));
    size_t frozenConvs = 0;
    for (const char *p = g_printed; (p = strstr(p, "\"CONV1D\",\"frozen\":true")) != NULL; p++) {
        frozenConvs++;
    }
    TEST_ASSERT_EQUAL_size_t(3, frozenConvs);
    freeHar(model);
}

/* The reason goes to stderr, here moved onto the harness's stdout. Exit 0 =
 * dumped, 1 = refused with nothing written, 2 = refused after writing. */
static void exitWithTheDumpVerdict(FILE *json, layer_t **model, size_t n, lossConfig_t loss,
                                   const tensor_t *input) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    bool dumped = rematSelectDumpProblem(json, stdout, model, n, loss, input);
    exit(dumped ? 0 : (ftell(json) == 0 ? 1 : 2));
}

/* What the problem cannot hold is refused before a byte is written: the
 * input's rank, batch and dtype, the loss, a layer type the dump does not
 * write, and a wire that is not FLOAT32. */
void testTheProblemDumpRefusesWhatTheProblemCannotHold(void) {
    static quantization_t floatQ;
    static quantization_t symQ;
    static symInt32QConfig_t symConfig;
    initFloat32Quantization(&floatQ);
    initSymInt32QConfig(HALF_AWAY, &symConfig);
    initSymInt32Quantization(&symConfig, &symQ);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &floatQ);
    layerQuant_t symOut = lq;
    symOut.outputQ = &symQ;
    layerQuant_t symDx = lq;
    symDx.propLossQ = &symQ;
    layer_t dropout = {.type = DROPOUT, .config = NULL};
    layer_t unknown = {.type = (layerType_t)99, .config = NULL};
    layer_t *relu[2] = {reluLayerInit(&lq), &dropout};
    layer_t *bad[2] = {&unknown, NULL};
    layer_t *reluSymOut[1] = {reluLayerInit(&symOut)};
    layer_t *reluSymDx[1] = {reluLayerInit(&symDx)};
    inputHeader_t in;
    const tensor_t *x = makeInputHeader(&in, (const size_t[]){1, 4}, 2u, &floatQ);
    inputHeader_t in9;
    const tensor_t *x9 =
        makeInputHeader(&in9, (const size_t[]){1, 1, 1, 1, 1, 1, 1, 1, 4}, 9u, &floatQ);
    inputHeader_t inBatch2;
    const tensor_t *xBatch2 = makeInputHeader(&inBatch2, (const size_t[]){2, 4}, 2u, &floatQ);
    inputHeader_t inSym;
    const tensor_t *xSym = makeInputHeader(&inSym, (const size_t[]){1, 4}, 2u, &symQ);
    const lossConfig_t unknownLoss = {.funcType = (lossFuncType_t)7};
    ASSERT_EXITS_WITH(0, exitWithTheDumpVerdict(tmpfile(), relu, 1u, g_crossEntropy, x));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "ERROR: DUMP_PROBLEM: the input has rank 9 (the problem dump writes ranks 1 to 8)",
        exitWithTheDumpVerdict(tmpfile(), relu, 1u, g_crossEntropy, x9));
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "ERROR: DUMP_PROBLEM: the input's first dimension is 2 (the problem is written at batch 1)",
        exitWithTheDumpVerdict(tmpfile(), relu, 1u, g_crossEntropy, xBatch2));
    ASSERT_EXITS_WITH_OUTPUT(
        1, "ERROR: DUMP_PROBLEM: the input has qtype_t 2 (the problem dump writes FLOAT32 only)",
        exitWithTheDumpVerdict(tmpfile(), relu, 1u, g_crossEntropy, xSym));
    ASSERT_EXITS_WITH_OUTPUT(1, "ERROR: DUMP_PROBLEM: unknown lossFuncType_t 7",
                             exitWithTheDumpVerdict(tmpfile(), relu, 1u, unknownLoss, x));
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "ERROR: DUMP_PROBLEM: layer 1 has layerType_t 10, which the problem "
                             "dump does not write (it writes CONV1D, RELU, MAXPOOL1D, AVGPOOL1D, "
                             "FLATTEN, LINEAR and SOFTMAX)",
                             exitWithTheDumpVerdict(tmpfile(), relu, 2u, g_crossEntropy, x));
    ASSERT_EXITS_WITH_OUTPUT(1,
                             "ERROR: DUMP_PROBLEM: layer 0 has layerType_t 99, which the problem "
                             "dump does not write (it writes CONV1D, RELU, MAXPOOL1D, AVGPOOL1D, "
                             "FLATTEN, LINEAR and SOFTMAX)",
                             exitWithTheDumpVerdict(tmpfile(), bad, 1u, g_crossEntropy, x));
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "ERROR: DUMP_PROBLEM: layer 0 has outputDtype qtype_t 2 (the problem dump writes FLOAT32 "
        "only)",
        exitWithTheDumpVerdict(tmpfile(), reluSymOut, 1u, g_crossEntropy, x));
    ASSERT_EXITS_WITH_OUTPUT(
        1,
        "ERROR: DUMP_PROBLEM: layer 0 has propLossDtype qtype_t 2 (the problem dump writes "
        "FLOAT32 only)",
        exitWithTheDumpVerdict(tmpfile(), reluSymDx, 1u, g_crossEntropy, x));
    freeReluLayer(relu[0]);
    freeReluLayer(reluSymOut[0]);
    freeReluLayer(reluSymDx[0]);
}

/* The trainers write the hash line to stderr, so a failure there has no
 * stream left to be named on: the dump returns false and the trainer exits 1. */
void testAFailedHashLineFailsTheDump(void) {
    static quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &floatQ);
    layer_t *relu[1] = {reluLayerInit(&lq)};
    inputHeader_t in;
    const tensor_t *x = makeInputHeader(&in, (const size_t[]){1, 4}, 2u, &floatQ);
    FILE *json = tmpfile();
    FILE *hashLine = fopen("/dev/null", "r");
    TEST_ASSERT_NOT_NULL(json);
    TEST_ASSERT_NOT_NULL(hashLine);
    TEST_ASSERT_FALSE(rematSelectDumpProblem(json, hashLine, relu, 1u, g_crossEntropy, x));
    (void)fclose(hashLine);
    (void)fclose(json);
    freeReluLayer(relu[0]);
}

/* A read-only stream stands for one that takes no bytes (a full disk): the
 * failure is named, and the dump returns false before its hash line. */
static void exitWithTheWriteVerdict(layer_t **model, const tensor_t *input) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    FILE *json = fopen("/dev/null", "r");
    exit(rematSelectDumpProblem(json, stdout, model, 1u, g_crossEntropy, input) ? 0 : 1);
}

void testAFailedProblemWriteIsNamed(void) {
    static quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, &floatQ);
    layer_t *relu[1] = {reluLayerInit(&lq)};
    inputHeader_t in;
    const tensor_t *x = makeInputHeader(&in, (const size_t[]){1, 4}, 2u, &floatQ);
    ASSERT_EXITS_WITH_OUTPUT(1, "ERROR: DUMP_PROBLEM: writing the problem failed",
                             exitWithTheWriteVerdict(relu, x));
    freeReluLayer(relu[0]);
}

/* One Flatten under MSE on a [1, 4] FLOAT32 input: no layer trains, so the
 * plan has no gradient wire, and ARENA places only the Flatten's output, 4
 * floats = 16 B (8-aligned already). */
static bool initFlattenWithBudget(rematScheduler_t *s, const char *budget) {
    rematSelection_t sel = selectWith("arena", NULL, NULL, NULL, budget);
    static layer_t flatten = {.type = FLATTEN, .config = NULL};
    static layer_t *model[1] = {&flatten};
    static quantization_t floatQ;
    initFloat32Quantization(&floatQ);
    static inputHeader_t in;
    const tensor_t *x = makeInputHeader(&in, (const size_t[]){1, 4}, 2u, &floatQ);
    return rematSelectInit(s, &sel, "unit", model, 1u, defaultLossConfig(MSE), x);
}

static void exitWithTheBudgetVerdict(const char *budget) {
    (void)dup2(STDOUT_FILENO, fileno(stderr));
    rematScheduler_t s;
    exit(initFlattenWithBudget(&s, budget) ? 0 : 1);
}

/* The budget bounds ARENA's arena only (the metadata is not in it), and an
 * arena equal to the budget fits. */
void testAnArenaWithinItsBudgetRuns(void) {
    rematScheduler_t s;
    TEST_ASSERT_TRUE(initFlattenWithBudget(&s, "16"));
    rematReport_t r;
    rematSchedulerReport(&s, &r);
    TEST_ASSERT_EQUAL_size_t(16, r.arenaBytes);
    TEST_ASSERT_TRUE(r.metadataBytes > 0u);
    rematSchedulerDeinit(&s);
    ASSERT_EXITS_WITH(0, exitWithTheBudgetVerdict("16"));
}

/* One byte less stops the run before it starts, naming both numbers; the
 * scheduler is deinitialised, as on a failed reservation. */
void testAnArenaOverItsBudgetStopsNamingBoth(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "ERROR: remat arena 16 B exceeds REMAT_BUDGET_B=15 B",
                             exitWithTheBudgetVerdict("15"));
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
    RUN_TEST(testDumpProblemAloneAsksForTheProblem);
    RUN_TEST(testABudgetIsReadInBytes);
    RUN_TEST(testABudgetIsLoggedWithItsCostModel);
    RUN_TEST(testAMalformedBudgetStopsBeforeTraining);
    RUN_TEST(testABudgetNeedsArenaStorage);
    RUN_TEST(testDumpProblemStandsAlone);
    RUN_TEST(testTheProblemHashIsFnv1a64);
    RUN_TEST(testTheProblemOfHarIsItsWholeModel);
    RUN_TEST(testTheProblemOfTheFinetuneModelMarksItsFrozenConvs);
    RUN_TEST(testTheProblemDumpRefusesWhatTheProblemCannotHold);
    RUN_TEST(testAFailedHashLineFailsTheDump);
    RUN_TEST(testAFailedProblemWriteIsNamed);
    RUN_TEST(testAnArenaWithinItsBudgetRuns);
    RUN_TEST(testAnArenaOverItsBudgetStopsNamingBoth);
    return UNITY_END();
}
