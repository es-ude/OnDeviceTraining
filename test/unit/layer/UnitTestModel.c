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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAChainNodeReadsItsPredecessor);
    RUN_TEST(testACsrNodeReadsItsSliceInInputOrder);
    RUN_TEST(testBothChainEncodingsAreAChain);
    RUN_TEST(testShapePassWritesEveryBufferShape);
    RUN_TEST(testShapePassReadsTheWiredInput);
    return UNITY_END();
}
