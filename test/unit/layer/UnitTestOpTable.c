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

/* EXEC is a sentinel the adapters only forward; nothing dereferences it. */
static int g_execMarker;
#define EXEC ((const layerExec_t *)&g_execMarker)

static struct {
    int calls;
    layer_t *layer;
    tensor_t *in, *out;
    const layerExec_t *exec;
} g_fwd;

static struct {
    int calls;
    layer_t *layer;
    tensor_t *in, *dy, *dx;
    const layerExec_t *exec;
} g_bwd;

static void resetSpies(void) {
    g_fwd.calls = 0;
    g_fwd.layer = NULL;
    g_fwd.in = g_fwd.out = NULL;
    g_fwd.exec = NULL;
    g_bwd.calls = 0;
    g_bwd.layer = NULL;
    g_bwd.in = g_bwd.dy = g_bwd.dx = NULL;
    g_bwd.exec = NULL;
}

static void spyForward(layer_t *layer, tensor_t *in, tensor_t *out, const layerExec_t *exec) {
    g_fwd.calls++;
    g_fwd.layer = layer;
    g_fwd.in = in;
    g_fwd.out = out;
    g_fwd.exec = exec;
}

static void spyBackward(layer_t *layer, tensor_t *in, tensor_t *dy, tensor_t *dx,
                        const layerExec_t *exec) {
    g_bwd.calls++;
    g_bwd.layer = layer;
    g_bwd.in = in;
    g_bwd.dy = dy;
    g_bwd.dx = dx;
    g_bwd.exec = exec;
}

/* The 14 layer rows. A row left out of the designated initialiser reads as zeros: arity 0
 * makes modelRequireValid's wiring arity check reject every wired node of that type, and a
 * NULL slot crashes the driver. accumulatesDx on a unary row
 * would let the deriver ask a layer for an ACC it cannot do. */
void testEveryLayerRowIsAUnaryRowWithAllThreeFunctions(void) {
    for (size_t t = LINEAR; t <= BATCHNORM1D; t++) {
        TEST_ASSERT_NOT_NULL(opFunctions[t].forward);
        TEST_ASSERT_NOT_NULL(opFunctions[t].backward);
        TEST_ASSERT_NOT_NULL(opFunctions[t].shape);
        TEST_ASSERT_EQUAL_UINT8(1u, opFunctions[t].arity);
        TEST_ASSERT_FALSE(opFunctions[t].readsOutput);
        TEST_ASSERT_FALSE(opFunctions[t].accumulatesDx);
    }
}

/* Tests and tools patch the mutable layerFunctions[]; the const table must reach the patch. */
void testEveryRowDispatchesThroughLayerFunctionsAtCallTime(void) {
    for (size_t t = LINEAR; t <= BATCHNORM1D; t++) {
        layerFunctions_t saved = layerFunctions[t];
        layerFunctions[t].forward = spyForward;
        layerFunctions[t].backward = spyBackward;
        layer_t layer = {.type = (layerType_t)t, .config = NULL};
        tensor_t x, y, dy, dx;
        const uint8_t write[1] = {OP_MODE_WRITE};
        resetSpies();
        opFunctions[t].forward(&layer, (tensor_t *const[]){&x}, 1, &y, EXEC);
        opFunctions[t].backward(&layer, (tensor_t *const[]){&x}, 1, &dy, (tensor_t *const[]){&dx},
                                write, EXEC);
        layerFunctions[t] = saved; /* restore before asserting: Unity longjmps on failure */
        TEST_ASSERT_EQUAL_INT(1, g_fwd.calls);
        TEST_ASSERT_EQUAL_PTR(&layer, g_fwd.layer);
        TEST_ASSERT_EQUAL_PTR(&x, g_fwd.in);
        TEST_ASSERT_EQUAL_PTR(&y, g_fwd.out);
        TEST_ASSERT_EQUAL_PTR(EXEC, g_fwd.exec);
        TEST_ASSERT_EQUAL_INT(1, g_bwd.calls);
        TEST_ASSERT_EQUAL_PTR(&x, g_bwd.in);
        TEST_ASSERT_EQUAL_PTR(&dy, g_bwd.dy);
        TEST_ASSERT_EQUAL_PTR(&dx, g_bwd.dx);
        TEST_ASSERT_EQUAL_PTR(EXEC, g_bwd.exec);
    }
}

/* dx[0] == NULL is today's grads-only call (backwardFn_t in Layer.h), passed through as
 * propLoss NULL. */
void testUnaryBackwardHandsANullDxThroughAsAGradsOnlyCall(void) {
    layerFunctions_t saved = layerFunctions[LINEAR];
    layerFunctions[LINEAR].backward = spyBackward;
    layer_t layer = {.type = LINEAR, .config = NULL};
    tensor_t x, dy;
    const uint8_t write[1] = {OP_MODE_WRITE};
    resetSpies();
    g_bwd.dx = (tensor_t *)&x; /* a non-NULL start, so a skipped call cannot pass */
    opFunctions[LINEAR].backward(&layer, (tensor_t *const[]){&x}, 1, &dy, (tensor_t *const[]){NULL},
                                 write, EXEC);
    layerFunctions[LINEAR] = saved;
    TEST_ASSERT_EQUAL_INT(1, g_bwd.calls);
    TEST_ASSERT_NULL(g_bwd.dx);
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
    RUN_TEST(testEveryLayerRowIsAUnaryRowWithAllThreeFunctions);
    RUN_TEST(testEveryRowDispatchesThroughLayerFunctionsAtCallTime);
    RUN_TEST(testUnaryBackwardHandsANullDxThroughAsAGradsOnlyCall);
    RUN_TEST(testUnaryShapePresetsTheInputRank);
    RUN_TEST(testUnaryShapePresetsRankTwoForFlatten);
    return UNITY_END();
}
