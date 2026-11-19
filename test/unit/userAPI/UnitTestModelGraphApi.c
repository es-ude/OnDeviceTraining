#define SOURCE_FILE "UNIT_TEST_MODEL_GRAPH_API"

#include <stdint.h>

#include "DeathTest.h"
#include "Layer.h"
#include "Model.h"
#include "ModelGraphApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void testTheBuilderWritesTheCsrOfAChain(void) {
    layer_t a = {.type = RELU}, b = {.type = FLATTEN}, c = {.type = RELU};
    layer_t *layers[3];
    uint16_t start[4];
    uint16_t ids[3];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 3, 3);
    uint16_t y1 = graphNode(&g, &a, GRAPH_INPUT);
    uint16_t y2 = graphNode(&g, &b, y1);
    uint16_t y3 = graphNode(&g, &c, y2);
    model_t m = graphFinish(&g);
    TEST_ASSERT_EQUAL_UINT16(1, y1);
    TEST_ASSERT_EQUAL_UINT16(2, y2);
    TEST_ASSERT_EQUAL_UINT16(3, y3);
    const uint16_t expStart[4] = {0, 1, 2, 3}, expIds[3] = {0, 1, 2};
    TEST_ASSERT_EQUAL_UINT16_ARRAY(expStart, start, 4);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(expIds, ids, 3);
    TEST_ASSERT_EQUAL_PTR(layers, m.layers);
    TEST_ASSERT_EQUAL_size_t(3, m.numLayers);
    TEST_ASSERT_EQUAL_PTR(start, m.inputStart);
    TEST_ASSERT_EQUAL_PTR(ids, m.inputIds);
    TEST_ASSERT_EQUAL_PTR(&b, layers[1]);
    TEST_ASSERT_TRUE(modelIsChain(&m));
}

void testMergeAppendsBothInputsInOrder(void) {
    layer_t a = {.type = RELU}, add = {.type = RELU};
    layer_t *layers[2];
    uint16_t start[3];
    uint16_t ids[3];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 2, 3);
    uint16_t y1 = graphNode(&g, &a, GRAPH_INPUT);
    TEST_ASSERT_EQUAL_UINT16(2, graphMerge(&g, &add, y1, GRAPH_INPUT));
    const uint16_t expStart[3] = {0, 1, 3}, expIds[3] = {0, 1, 0};
    TEST_ASSERT_EQUAL_UINT16_ARRAY(expStart, start, 3);
    TEST_ASSERT_EQUAL_UINT16_ARRAY(expIds, ids, 3);
    TEST_ASSERT_EQUAL_PTR(&add, layers[1]);
}

/* graphFinish is the validation entry: a merge through a unary layer fails on arity. */
void testFinishValidatesTheWiring(void) {
    layer_t a = {.type = RELU}, add = {.type = RELU};
    layer_t *layers[2];
    uint16_t start[3];
    uint16_t ids[3];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 2, 3);
    graphMerge(&g, &add, graphNode(&g, &a, GRAPH_INPUT), GRAPH_INPUT);
    ASSERT_EXITS_WITH_OUTPUT(1, "graphFinish: wiring: node 1 has 2 inputs, layer type 1 takes 1",
                             (void)graphFinish(&g));
}

void testANodeBeyondTheCapacityExits(void) {
    layer_t a = {.type = RELU};
    layer_t *layers[1];
    uint16_t start[2];
    uint16_t ids[2];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 1, 2);
    graphNode(&g, &a, GRAPH_INPUT);
    ASSERT_EXITS_WITH_OUTPUT(1, "graphNode: node capacity 1 exceeded", (void)graphNode(&g, &a, 1));
}

void testANodeBeyondTheEdgeCapacityExits(void) {
    layer_t a = {.type = RELU};
    layer_t *layers[2];
    uint16_t start[3];
    uint16_t ids[1];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 2, 1);
    graphNode(&g, &a, GRAPH_INPUT);
    ASSERT_EXITS_WITH_OUTPUT(1, "graphNode: edge capacity 1 exceeded", (void)graphNode(&g, &a, 1));
}

void testAMergeBeyondTheNodeCapacityExits(void) {
    layer_t a = {.type = RELU};
    layer_t *layers[1];
    uint16_t start[2];
    uint16_t ids[3];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 1, 3);
    graphNode(&g, &a, GRAPH_INPUT);
    ASSERT_EXITS_WITH_OUTPUT(1, "graphMerge: node capacity 1 exceeded",
                             (void)graphMerge(&g, &a, 1, 0));
}

void testAMergeBeyondTheEdgeCapacityExits(void) {
    layer_t a = {.type = RELU};
    layer_t *layers[2];
    uint16_t start[3];
    uint16_t ids[2];
    graphBuilder_t g;
    graphBegin(&g, layers, start, ids, 2, 2);
    graphNode(&g, &a, GRAPH_INPUT);
    ASSERT_EXITS_WITH_OUTPUT(1, "graphMerge: edge capacity 2 exceeded",
                             (void)graphMerge(&g, &a, 1, 0));
}

void testBeginRefusesEdgeOffsetsBeyondUint16(void) {
    layer_t *layers[1];
    uint16_t start[2];
    uint16_t ids[1];
    graphBuilder_t g;
    ASSERT_EXITS_WITH_OUTPUT(1, "graphBegin: edge capacity 65536 exceeds the uint16_t CSR offsets",
                             graphBegin(&g, layers, start, ids, 1, 0x10000u));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTheBuilderWritesTheCsrOfAChain);
    RUN_TEST(testMergeAppendsBothInputsInOrder);
    RUN_TEST(testFinishValidatesTheWiring);
    RUN_TEST(testANodeBeyondTheCapacityExits);
    RUN_TEST(testANodeBeyondTheEdgeCapacityExits);
    RUN_TEST(testAMergeBeyondTheNodeCapacityExits);
    RUN_TEST(testAMergeBeyondTheEdgeCapacityExits);
    RUN_TEST(testBeginRefusesEdgeOffsetsBeyondUint16);
    return UNITY_END();
}
