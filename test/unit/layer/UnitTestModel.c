#define SOURCE_FILE "UNIT_TEST_MODEL"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAChainNodeReadsItsPredecessor);
    RUN_TEST(testACsrNodeReadsItsSliceInInputOrder);
    RUN_TEST(testBothChainEncodingsAreAChain);
    return UNITY_END();
}
