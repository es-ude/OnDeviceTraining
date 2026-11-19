#define SOURCE_FILE "UNIT_TEST_OP_TABLE"

#include <stddef.h>
#include <stdint.h>

#include "Layer.h"
#include "OpTable.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static int g_calls;
static size_t g_rankSeen;
static layer_t *g_layerSeen;

static void spyShape(layer_t *l, shape_t *in, shape_t *out) {
    (void)in;
    g_calls++;
    g_rankSeen = out->numberOfDimensions;
    g_layerSeen = l;
}

/* Patches the live layerFunctions slot so the spy sees what the adapter hands
 * the layer's shape function. */
static void callShapeThroughASpy(layerType_t t, size_t inRank) {
    calcOutputShapeFn_t saved = layerFunctions[t].calcOutputShape;
    layerFunctions[t].calcOutputShape = spyShape;
    layer_t layer = {.type = t};
    size_t inDims[3] = {2, 3, 4}, inOrder[3] = {0, 1, 2}, d[3], o[3];
    shape_t inShape = {
        .numberOfDimensions = inRank, .dimensions = inDims, .orderOfDimensions = inOrder};
    shape_t out = {.numberOfDimensions = 0, .dimensions = d, .orderOfDimensions = o};
    const shape_t *in[1] = {&inShape};
    g_calls = 0;
    opFunctions[t].shape(&layer, in, 1, &out);
    layerFunctions[t].calcOutputShape = saved;
    TEST_ASSERT_EQUAL_INT(1, g_calls);
    TEST_ASSERT_EQUAL_PTR(&layer, g_layerSeen);
}

void testEveryLayerTypeHasAUnaryRow(void) {
    for (int t = LINEAR; t <= BATCHNORM1D; t++) {
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(1, opFunctions[t].arity, "arity");
        TEST_ASSERT_NOT_NULL(opFunctions[t].shape);
    }
}

/* The adapter reaches the live layerFunctions slot and presets the rank. */
void testUnaryShapePresetsTheInputRank(void) {
    callShapeThroughASpy(RELU, 3);
    TEST_ASSERT_EQUAL_size_t(3, g_rankSeen);
}

void testUnaryShapePresetsRankTwoForFlatten(void) {
    callShapeThroughASpy(FLATTEN, 3);
    TEST_ASSERT_EQUAL_size_t(2, g_rankSeen);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testEveryLayerTypeHasAUnaryRow);
    RUN_TEST(testUnaryShapePresetsTheInputRank);
    RUN_TEST(testUnaryShapePresetsRankTwoForFlatten);
    return UNITY_END();
}
