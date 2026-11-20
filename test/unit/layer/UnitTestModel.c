#define SOURCE_FILE "UNIT_TEST_MODEL"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "BatchNorm1d.h"
#include "DeathTest.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "LossType.h"
#include "Model.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* Nothing reads config: the accessors only walk the wiring. */
static layer_t g_relu[3] = {{.type = RELU}, {.type = RELU}, {.type = RELU}};

void testAChainNodeReadsItsPredecessor(void) {
    layer_t *layers[3] = {&g_relu[0], &g_relu[1], &g_relu[2]};
    uint16_t ids[MODEL_MAX_ARITY] = {0xBEEF, 0xBEEF};
    TEST_ASSERT_EQUAL_UINT8(1, modelNodeInputs(&MODEL_CHAIN(layers, 3), 2, ids));
    TEST_ASSERT_EQUAL_UINT16(2, ids[0]);
}

/* The accessor reads the CSR slice; it does not validate (the entry does). */
void testACsrNodeReadsItsSliceInInputOrder(void) {
    layer_t *layers[3] = {&g_relu[0], &g_relu[1], &g_relu[2]};
    static const uint16_t start[4] = {0, 1, 3, 4};
    static const uint16_t in[4] = {0, 1, 0, 2};
    model_t m = {layers, 3, start, in};
    uint16_t ids[MODEL_MAX_ARITY];
    TEST_ASSERT_EQUAL_UINT8(2, modelNodeInputs(&m, 1, ids));
    TEST_ASSERT_EQUAL_UINT16(1, ids[0]);
    TEST_ASSERT_EQUAL_UINT16(0, ids[1]);
    TEST_ASSERT_EQUAL_UINT8(1, modelNodeInputs(&m, 2, ids));
    TEST_ASSERT_EQUAL_UINT16(2, ids[0]);
}

void testBothChainEncodingsAreAChain(void) {
    layer_t *layers[3] = {&g_relu[0], &g_relu[1], &g_relu[2]};
    static const uint16_t start[4] = {0, 1, 2, 3};
    static const uint16_t chain[3] = {0, 1, 2};
    static const uint16_t skip[3] = {0, 1, 1};
    TEST_ASSERT_TRUE(modelIsChain(&MODEL_CHAIN(layers, 3)));
    TEST_ASSERT_TRUE(modelIsChain(&(model_t){layers, 3, start, chain}));
    TEST_ASSERT_FALSE(modelIsChain(&(model_t){layers, 3, start, skip}));
    static const uint16_t wideStart[4] = {0, 1, 3, 4};
    static const uint16_t wide[4] = {0, 1, 0, 2};
    TEST_ASSERT_FALSE(modelIsChain(&(model_t){layers, 3, wideStart, wide}));
}

enum { MAX_RANK = 3 };

static void bindShapes(shape_t *s, size_t (*d)[MAX_RANK], size_t (*o)[MAX_RANK], size_t count) {
    for (size_t j = 0; j < count; j++) {
        s[j] = (shape_t){.dimensions = d[j], .orderOfDimensions = o[j]};
    }
}

void testShapePassWritesEveryBufferShape(void) {
    layer_t relu0 = {.type = RELU}, flat = {.type = FLATTEN}, relu1 = {.type = RELU};
    layer_t *layers[3] = {&relu0, &flat, &relu1};
    size_t inDims[3] = {2, 3, 4}, inOrder[3] = {0, 1, 2};
    shape_t input = {.numberOfDimensions = 3, .dimensions = inDims, .orderOfDimensions = inOrder};
    size_t d[4][MAX_RANK], o[4][MAX_RANK];
    shape_t shapes[4];
    bindShapes(shapes, d, o, 4);
    modelShapePass(&MODEL_CHAIN(layers, 3), &input, shapes, MAX_RANK);
    const size_t rank[4] = {3, 3, 2, 2};
    const size_t dims[4][3] = {{2, 3, 4}, {2, 3, 4}, {2, 12, 0}, {2, 12, 0}};
    for (size_t j = 0; j < 4; j++) {
        TEST_ASSERT_EQUAL_size_t(rank[j], shapes[j].numberOfDimensions);
        for (size_t k = 0; k < rank[j]; k++) {
            TEST_ASSERT_EQUAL_size_t(dims[j][k], shapes[j].dimensions[k]);
            TEST_ASSERT_EQUAL_size_t(k, shapes[j].orderOfDimensions[k]);
        }
    }
    TEST_ASSERT_TRUE(shapes[0].dimensions != input.dimensions); /* copied, not aliased */
}

/* The pass follows the wiring, not the position: node 1 reads buffer 0. The pass trusts the
 * entry's validation, so this unvalidated CSR is a fair probe. */
void testShapePassReadsTheWiredInput(void) {
    layer_t flat = {.type = FLATTEN}, relu = {.type = RELU};
    layer_t *layers[2] = {&flat, &relu};
    static const uint16_t start[3] = {0, 1, 2};
    static const uint16_t in[2] = {0, 0};
    size_t inDims[3] = {2, 3, 4}, inOrder[3] = {0, 1, 2};
    shape_t input = {.numberOfDimensions = 3, .dimensions = inDims, .orderOfDimensions = inOrder};
    size_t d[3][MAX_RANK], o[3][MAX_RANK];
    shape_t shapes[3];
    bindShapes(shapes, d, o, 3);
    modelShapePass(&(model_t){layers, 2, start, in}, &input, shapes, MAX_RANK);
    TEST_ASSERT_EQUAL_size_t(3, shapes[2].numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(4, shapes[2].dimensions[2]);
}

static layer_t *g_three[3] = {&g_relu[0], &g_relu[1], &g_relu[2]};

void testValidChainsPass(void) {
    static const uint16_t start[4] = {0, 1, 2, 3}, in[3] = {0, 1, 2};
    layer_t *reused[3] = {&g_relu[0], &g_relu[0], &g_relu[0]}; /* chains may reuse a layer */
    modelRequireValid(&MODEL_CHAIN(g_three, 3), "t");
    modelRequireValid(&(model_t){g_three, 3, start, in}, "t");
    modelRequireValid(&MODEL_CHAIN(reused, 3), "t");
}

void testAnEmptyModelExitsNamingTheCaller(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "someEntry: modelSize == 0: nothing to schedule",
                             modelRequireValid(&MODEL_CHAIN(g_three, 0), "someEntry"));
}

/* Rule 1 reads no layer: 0xFFFF nodes over a 3-entry array is safe. */
void testAModelAtTheBufferIdLimitExits(void) {
    ASSERT_EXITS_WITH_OUTPUT(1, "t: modelSize 65535 reaches 0xFFFF: buffer ids are uint16_t",
                             modelRequireValid(&(model_t){g_three, 0xFFFFu, NULL, NULL}, "t"));
}

void testTheCsrStartsAtZero(void) {
    static const uint16_t start[3] = {1, 1, 2}, in[2] = {0, 1};
    ASSERT_EXITS_WITH_OUTPUT(1, "t: wiring: inputStart[0] is 1, not 0",
                             modelRequireValid(&(model_t){g_three, 2, start, in}, "t"));
}

void testTheCsrNeverDecreases(void) {
    static const uint16_t start[3] = {0, 1, 0}, in[1] = {0};
    ASSERT_EXITS_WITH_OUTPUT(1, "t: wiring: inputStart[2] = 0 is below inputStart[1] = 1",
                             modelRequireValid(&(model_t){g_three, 2, start, in}, "t"));
}

/* The out-of-range second id is never read: the arity rule fires first. */
void testANodeTakesExactlyItsArity(void) {
    static const uint16_t start[3] = {0, 1, 3}, in[3] = {0, 1, 0x7777};
    ASSERT_EXITS_WITH_OUTPUT(1, "t: wiring: node 1 has 2 inputs, layer type 1 takes 1",
                             modelRequireValid(&(model_t){g_three, 2, start, in}, "t"));
}

void testAnInputComesFromAnEarlierBuffer(void) {
    static const uint16_t start[3] = {0, 1, 2}, in[2] = {1, 1};
    ASSERT_EXITS_WITH_OUTPUT(1, "t: wiring: node 0 reads buffer 1, which is not produced before it",
                             modelRequireValid(&(model_t){g_three, 2, start, in}, "t"));
}

void testEveryBufferButTheLastHasAConsumer(void) {
    static const uint16_t start[3] = {0, 1, 2}, in[2] = {0, 0};
    ASSERT_EXITS_WITH_OUTPUT(1, "t: wiring: buffer 1 has no consumer",
                             modelRequireValid(&(model_t){g_three, 2, start, in}, "t"));
}

static void assertFlags(layer_t **layers, size_t n, lossFuncType_t lt, const uint8_t *needs,
                        const uint8_t *runs) {
    uint8_t g[8], r[8];
    memset(g, 0xA5, sizeof g);
    memset(r, 0xA5, sizeof r);
    modelGradFlags(&MODEL_CHAIN(layers, n), lt, g, r);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(needs, g, n + 1);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(runs, r, n);
}

static parameter_t g_p[8];
static linearConfig_t g_linear = {.weights = &g_p[0], .bias = &g_p[1]};
static linearConfig_t g_frozenLinear = {.weights = &g_p[2], .bias = &g_p[3], .frozen = true};
static layerNormConfig_t g_norm = {.gamma = &g_p[4], .beta = &g_p[5]};
static layerNormConfig_t g_frozenNorm = {.gamma = &g_p[4], .beta = &g_p[5], .frozen = true};
static groupNormConfig_t g_frozenGroupNorm = {.gamma = &g_p[6], .beta = &g_p[7], .frozen = true};
static batchNorm1dConfig_t g_nonAffineBn = {.gamma = NULL, .beta = NULL, .affine = false};

static layer_t g_top = {.type = LINEAR, .config = &(layerConfig_t){.linear = &g_linear}};
static layer_t g_frozen = {.type = LINEAR, .config = &(layerConfig_t){.linear = &g_frozenLinear}};
static layer_t g_trainNorm = {.type = LAYERNORM, .config = &(layerConfig_t){.layerNorm = &g_norm}};
static layer_t g_frozenNormLayer = {.type = LAYERNORM,
                                    .config = &(layerConfig_t){.layerNorm = &g_frozenNorm}};
static layer_t g_frozenGn = {.type = GROUPNORM,
                             .config = &(layerConfig_t){.groupNorm = &g_frozenGroupNorm}};
static layer_t g_bn = {.type = BATCHNORM1D,
                       .config = &(layerConfig_t){.batchNorm1d = &g_nonAffineBn}};
static layer_t g_reluLayer = {.type = RELU, .config = NULL};
static layer_t g_flatten = {.type = FLATTEN, .config = NULL};
static layer_t g_softmax = {.type = SOFTMAX, .config = NULL};

void testGradFlagsSkipFrozenAndParamFreeLayersBelowTheFirstTrainable(void) {
    assertFlags((layer_t *[]){&g_reluLayer, &g_frozen, &g_trainNorm, &g_top}, 4, MSE,
                (uint8_t[]){0, 0, 0, 1, 1}, (uint8_t[]){0, 0, 1, 1});
    assertFlags((layer_t *[]){&g_top, &g_reluLayer}, 2, MSE, (uint8_t[]){0, 1, 1},
                (uint8_t[]){1, 1});
}

void testGradFlagsAreAllZeroWhenNothingTrains(void) {
    assertFlags((layer_t *[]){&g_frozen, &g_reluLayer, &g_frozenGn}, 3, MSE,
                (uint8_t[]){0, 0, 0, 0}, (uint8_t[]){0, 0, 0});
    assertFlags((layer_t *[]){&g_reluLayer, &g_flatten}, 2, MSE, (uint8_t[]){0, 0, 0},
                (uint8_t[]){0, 0});
}

/* A non-affine BatchNorm1d carries a config but no parameters: it trains nothing. */
void testGradFlagsSkipANonAffineBatchNorm(void) {
    assertFlags((layer_t *[]){&g_bn, &g_top}, 2, MSE, (uint8_t[]){0, 0, 1}, (uint8_t[]){0, 1});
}

/* Under CROSS_ENTROPY the sink Softmax never runs backward: the combined p - y gradient covers
 * it, while its input buffer still needs a gradient. */
void testGradFlagsNeverRunTheSinkSoftmaxUnderCrossEntropy(void) {
    assertFlags((layer_t *[]){&g_top, &g_reluLayer, &g_softmax}, 3, CROSS_ENTROPY,
                (uint8_t[]){0, 1, 1, 1}, (uint8_t[]){1, 1, 0});
    assertFlags((layer_t *[]){&g_top, &g_reluLayer, &g_softmax}, 3, MSE, (uint8_t[]){0, 1, 1, 1},
                (uint8_t[]){1, 1, 1});
}

static uint32_t nextRandom(uint32_t *state) { /* xorshift32, test-local */
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

/* On a chain, needs-grad is the positional range from the deepest trainable node to the top:
 * node j runs backward iff deepest <= j, and buffer j needs a gradient iff j > deepest. The
 * oracle computes deepest independently of the graph walk under test. */
void testGradFlagsOfAChainAreTheDeepestToTopRange(void) {
    layer_t *pool[] = {&g_top, &g_frozen, &g_reluLayer, &g_trainNorm, &g_frozenNormLayer};
    uint32_t state = 0x9E3779B9u;
    bool sawTraining = false, sawFrozenOnly = false;
    for (int trial = 0; trial < 200; trial++) {
        size_t n = 1 + nextRandom(&state) % 7u;
        layer_t *layers[7];
        for (size_t j = 0; j < n; j++) {
            layers[j] = pool[nextRandom(&state) % 5u];
        }
        size_t deepest = n;
        for (size_t j = 0; j < n && deepest == n; j++) {
            parameter_t *w = NULL, *b = NULL;
            if (layerParameters(layers[j], &w, &b) && !layerIsFrozen(layers[j])) {
                deepest = j;
            }
        }
        sawTraining |= deepest < n;
        sawFrozenOnly |= deepest == n;

        uint8_t needs[8], runs[8];
        memset(needs, 0xA5, sizeof needs);
        memset(runs, 0xA5, sizeof runs);
        modelGradFlags(&MODEL_CHAIN(layers, n), MSE, needs, runs);
        TEST_ASSERT_EQUAL_UINT8(0, needs[0]);
        for (size_t j = 0; j < n; j++) {
            TEST_ASSERT_EQUAL_UINT8(deepest <= j, runs[j]);
        }
        for (size_t j = 1; j <= n; j++) {
            TEST_ASSERT_EQUAL_UINT8(j > deepest, needs[j]);
        }
    }
    TEST_ASSERT_TRUE(sawTraining && sawFrozenOnly);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAChainNodeReadsItsPredecessor);
    RUN_TEST(testACsrNodeReadsItsSliceInInputOrder);
    RUN_TEST(testBothChainEncodingsAreAChain);
    RUN_TEST(testShapePassWritesEveryBufferShape);
    RUN_TEST(testShapePassReadsTheWiredInput);
    RUN_TEST(testValidChainsPass);
    RUN_TEST(testAnEmptyModelExitsNamingTheCaller);
    RUN_TEST(testAModelAtTheBufferIdLimitExits);
    RUN_TEST(testTheCsrStartsAtZero);
    RUN_TEST(testTheCsrNeverDecreases);
    RUN_TEST(testANodeTakesExactlyItsArity);
    RUN_TEST(testAnInputComesFromAnEarlierBuffer);
    RUN_TEST(testEveryBufferButTheLastHasAConsumer);
    RUN_TEST(testGradFlagsSkipFrozenAndParamFreeLayersBelowTheFirstTrainable);
    RUN_TEST(testGradFlagsAreAllZeroWhenNothingTrains);
    RUN_TEST(testGradFlagsSkipANonAffineBatchNorm);
    RUN_TEST(testGradFlagsNeverRunTheSinkSoftmaxUnderCrossEntropy);
    RUN_TEST(testGradFlagsOfAChainAreTheDeepestToTopRange);
    return UNITY_END();
}
