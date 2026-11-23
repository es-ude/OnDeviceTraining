#define SOURCE_FILE "UNIT_TEST_ADD_LAYER"

#include <stddef.h>

#include "AddLayer.h"
#include "DeathTest.h"
#include "Layer.h"
#include "OpTable.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* The op table stores these without casts. */
static const opForwardFn_t addForwardAsOp = addForward;
static const opShapeFn_t addShapeAsOp = addShape;
static const opBackwardFn_t addBackwardAsOp = addBackward;

static tensor_t *floatTensor(const float *data) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(t, data, 3);
    return t;
}

void testAddForwardSumsElementwise(void) {
    addConfig_t cfg = {.forwardMath = {ARITH_FLOAT32, HALF_AWAY}};
    layerConfig_t lc = {.add = &cfg};
    layer_t add = {.type = ADD, .config = &lc};
    tensor_t *a = floatTensor((float[]){1, 2, 3});
    tensor_t *b = floatTensor((float[]){10, 20, 30});
    tensor_t *out = floatTensor((float[]){-1, -1, -1});
    addForwardAsOp(&add, (tensor_t *const[]){a, b}, 2, out, NULL);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){11, 22, 33}), (float *)out->data, 3);
}

void testAddShapeCopiesTheCommonShape(void) {
    layer_t add = {.type = ADD};
    size_t aDims[] = {1, 3}, bDims[] = {1, 3}, order[] = {0, 1};
    size_t outDims[2] = {0}, outOrder[2] = {0};
    shape_t a = {2, aDims, order}, b = {2, bDims, order};
    shape_t out = {0, outDims, outOrder};
    addShapeAsOp(&add, (const shape_t *const[]){&a, &b}, 2, &out);
    TEST_ASSERT_EQUAL_size_t(2, out.numberOfDimensions);
    TEST_ASSERT_EQUAL_size_t(1, out.dimensions[0]);
    TEST_ASSERT_EQUAL_size_t(3, out.dimensions[1]);
    TEST_ASSERT_EQUAL_size_t(0, out.orderOfDimensions[0]);
    TEST_ASSERT_EQUAL_size_t(1, out.orderOfDimensions[1]);
}

void testAddShapeRefusesToBroadcast(void) {
    layer_t add = {.type = ADD};
    size_t aDims[] = {1, 3}, bDims[] = {1, 4}, order[] = {0, 1};
    size_t outDims[2] = {0}, outOrder[2] = {0};
    shape_t a = {2, aDims, order}, b = {2, bDims, order};
    shape_t out = {0, outDims, outOrder};
    const shape_t *const ins[] = {&a, &b};
    ASSERT_EXITS_WITH_OUTPUT(1, "ADD does not broadcast", addShape(&add, ins, 2, &out));

    /* Equal on the first input's rank: only the rank compare tells them apart. */
    size_t rowDims[] = {3}, colDims[] = {3, 1};
    shape_t row = {1, rowDims, order}, col = {2, colDims, order};
    const shape_t *const rankIns[] = {&row, &col};
    ASSERT_EXITS_WITH_OUTPUT(1, "[3] vs [3, 1]; ADD does not broadcast",
                             addShape(&add, rankIns, 2, &out));
}

void testAddBackwardWritesAccumulatesAndSkips(void) {
    layer_t add = {.type = ADD};
    tensor_t *dy = floatTensor((float[]){1, 2, 3});
    tensor_t *d0 = floatTensor((float[]){9, 9, 9});
    tensor_t *d1 = floatTensor((float[]){10, 10, 10});
    addBackwardAsOp(&add, NULL, 2, dy, (tensor_t *const[]){d0, d1},
                    (uint8_t[]){OP_MODE_WRITE, OP_MODE_ACC}, NULL);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){1, 2, 3}), (float *)d0->data, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){11, 12, 13}), (float *)d1->data, 3);

    tensor_t *d2 = floatTensor((float[]){5, 5, 5});
    addBackwardAsOp(&add, NULL, 2, dy, (tensor_t *const[]){NULL, d2},
                    (uint8_t[]){OP_MODE_WRITE, OP_MODE_WRITE}, NULL);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){1, 2, 3}), (float *)d2->data, 3);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){1, 2, 3}), (float *)dy->data, 3);
}

/* ADD(y, y): both slots route to one buffer, opened by slot 0 and added into by slot 1. */
void testAddBackwardOnTheSameBufferTwiceDoublesDy(void) {
    layer_t add = {.type = ADD};
    tensor_t *dy = floatTensor((float[]){1, 2, 3});
    tensor_t *g = floatTensor((float[]){-7, -7, -7});
    addBackwardAsOp(&add, NULL, 2, dy, (tensor_t *const[]){g, g},
                    (uint8_t[]){OP_MODE_WRITE, OP_MODE_ACC}, NULL);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(((float[]){2, 4, 6}), (float *)g->data, 3);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAddForwardSumsElementwise);
    RUN_TEST(testAddShapeCopiesTheCommonShape);
    RUN_TEST(testAddShapeRefusesToBroadcast);
    RUN_TEST(testAddBackwardWritesAccumulatesAndSkips);
    RUN_TEST(testAddBackwardOnTheSameBufferTwiceDoublesDy);
    return UNITY_END();
}
