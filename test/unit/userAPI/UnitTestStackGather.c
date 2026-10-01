#define SOURCE_FILE "UNIT_TEST_STACK_GATHER"

#include <stddef.h>
#include <stdint.h>

#include "DeathTest.h"
#include "LayerConfigAccess.h"
#include "LayerQuant.h"
#include "Linear.h"
#include "LinearApi.h"
#include "QuantizationApi.h"
#include "StackGather.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

_Static_assert(_Generic(&stackGatherRequireStackable,
                   void (*)(const char *, const char *, tensor_t *, tensor_t *, const char *,
                            size_t, size_t): 1,
                   default: 0),
               "stackGatherRequireStackable signature (#468)");
_Static_assert(_Generic(&stackGatherRequireFloat32Model,
                   void (*)(const char *, const char *, layer_t **, size_t, size_t,
                            nonFloat32FieldFn_t): 1,
                   default: 0),
               "stackGatherRequireFloat32Model signature (#468)");
_Static_assert(_Generic(&stackGatherReserveBuffer,
                   uint8_t *(*)(const char *, const char *, size_t, size_t, const char *): 1,
                   default: 0),
               "stackGatherReserveBuffer signature (#468)");

void setUp(void) {}
void tearDown(void) {}

static tensor_t *floatTensor(const size_t *dims, size_t rank) {
    size_t *d = reserveMemory(rank * sizeof(size_t));
    size_t *o = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        d[i] = dims[i];
    }
    setOrderOfDimsForNewTensor(rank, o);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, d, rank, o);
    return initTensor(s, quantizationInitFloat(), NULL);
}

void testRequireStackableAcceptsIdenticalAndRejectsMismatch(void) {
    tensor_t *ref = floatTensor((size_t[]){2, 3}, 2);
    tensor_t *same = floatTensor((size_t[]){2, 3}, 2);
    tensor_t *otherDims = floatTensor((size_t[]){3, 2}, 2);
    tensor_t *otherRank = floatTensor((size_t[]){6}, 1);
    ASSERT_EXITS_WITH(0, stackGatherRequireStackable("t", "k", ref, same, "item", 1, 2));
    ASSERT_EXITS_WITH_FAILURE(stackGatherRequireStackable("t", "k", ref, otherDims, "item", 1, 2));
    ASSERT_EXITS_WITH_FAILURE(stackGatherRequireStackable("t", "k", ref, otherRank, "item", 1, 2));
    same->quantization->type = SYM_INT32; /* dtype tag only; nothing reads qConfig */
    ASSERT_EXITS_WITH_FAILURE(stackGatherRequireStackable("t", "k", ref, same, "item", 1, 2));
    same->quantization->type = FLOAT32;
    freeTensor(otherRank);
    freeTensor(otherDims);
    freeTensor(same);
    freeTensor(ref);
}

void testRequireFloat32ModelUsesTheGivenAccessor(void) {
    quantization_t *q = quantizationInitFloat();
    quantization_t *symQ = quantizationInitSymInt32(HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, q);
    layer_t *lin = linearLayerInit(&(linearInit_t){.inFeatures = 2, .outFeatures = 2}, &lq);
    layer_t *model[1] = {lin};
    quantization_t *saved = lin->config->linear->propLossQ;
    lin->config->linear->propLossQ = symQ; /* backward-only field */
    ASSERT_EXITS_WITH_FAILURE(
        stackGatherRequireFloat32Model("t", "k", model, 1, 2, layerNonFloat32Field));
    ASSERT_EXITS_WITH(
        0, stackGatherRequireFloat32Model("t", "k", model, 1, 2, layerForwardNonFloat32Field));
    lin->config->linear->propLossQ = saved;
    freeLinearLayer(lin);
    freeQuantization(symQ);
    freeQuantization(q);
}

void testReserveBufferRejectsOverflow(void) {
    ASSERT_EXITS_WITH_FAILURE(stackGatherReserveBuffer("t", "k", SIZE_MAX / 2, 4, "item"));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testRequireStackableAcceptsIdenticalAndRejectsMismatch);
    RUN_TEST(testRequireFloat32ModelUsesTheGivenAccessor);
    RUN_TEST(testReserveBufferRejectsOverflow);
    return UNITY_END();
}
