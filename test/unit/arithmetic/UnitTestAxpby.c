#define SOURCE_FILE "UNIT_TEST_AXPBY"

#include <stddef.h>
#include <stdlib.h>

#include "Axpby.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

static tensor_t *buildFloat32TensorND(size_t rank, const size_t *srcDims, const float *fill) {
    size_t *dims = reserveMemory(rank * sizeof(size_t));
    size_t *order = reserveMemory(rank * sizeof(size_t));
    for (size_t i = 0; i < rank; i++) {
        dims[i] = srcDims[i];
        order[i] = i;
    }
    shape_t *shape = reserveMemory(sizeof(shape_t));
    shape->dimensions = dims;
    shape->orderOfDimensions = order;
    shape->numberOfDimensions = rank;
    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (fill != NULL) {
        float *d = (float *)t->data;
        size_t n = calcNumberOfElementsByTensor(t);
        for (size_t i = 0; i < n; i++) {
            d[i] = fill[i];
        }
    }
    return t;
}

void testAxpbyBasic(void) {
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){4}, (float[]){1.0f, 2.0f, 3.0f, 4.0f});
    tensor_t *y = buildFloat32TensorND(1, (size_t[]){4}, (float[]){10.0f, 20.0f, 30.0f, 40.0f});
    tensor_t *out = buildFloat32TensorND(1, (size_t[]){4}, NULL);

    axpbyFloat32Tensors(2.0f, x, 0.5f, y, out);

    float expected[] = {7.0f, 14.0f, 21.0f, 28.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 4);
    freeTensor(out);
    freeTensor(y);
    freeTensor(x);
}

void testAxpbyAliasedOutput(void) {
    /* out == x must work: same-index read-then-write. */
    tensor_t *x = buildFloat32TensorND(1, (size_t[]){3}, (float[]){1.0f, 2.0f, 3.0f});
    tensor_t *y = buildFloat32TensorND(1, (size_t[]){3}, (float[]){4.0f, 5.0f, 6.0f});

    axpbyFloat32Tensors(1.0f, x, 1.0f, y, x);

    float expected[] = {5.0f, 7.0f, 9.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)x->data, 3);
    freeTensor(y);
    freeTensor(x);
}

void testAxpbyPermutationAwareRead(void) {
    /* x is a transposed VIEW [3,2] over a [2,3] row-major buffer:
     * storage {1,2,3,4,5,6} viewed transposed reads logically
     * {{1,4},{2,5},{3,6}}. y zeros, a=1, b=0 -> out must be the
     * TRANSPOSED element order, flat. */
    tensor_t *xBase =
        buildFloat32TensorND(2, (size_t[]){2, 3}, (float[]){1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    /* Build the transposed view: logical dims [3,2] over the same data.
     * Copy the transpose idiom from src/layer/Linear.c (transposeTensor);
     * if transposeTensor mutates in place, apply it to xBase directly. */
    transposeTensor(xBase, 0, 1);

    tensor_t *y = buildFloat32TensorND(2, (size_t[]){3, 2}, NULL);
    tensor_t *out = buildFloat32TensorND(2, (size_t[]){3, 2}, NULL);

    axpbyFloat32Tensors(1.0f, xBase, 0.0f, y, out);

    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 6);
    freeTensor(out);
    freeTensor(y);
    transposeTensor(xBase, 0, 1); /* untranspose before freeing (Linear.c idiom) */
    freeTensor(xBase);
}

void testAxpbyThreeCyclePermutationRead(void) {
    /* Two composed transposes turn orderOfDimensions into a 3-cycle
     * ([0,1,2] -> [1,0,2] -> [1,2,0]). Post-#344 the shared index algebra
     * (calcElementIndexByIndices) inverts arbitrary permutations, so the read
     * must be CORRECT -- the old involution-only fail-fast guard is gone.
     * Storage is filled with its own flat index, so each expected value spells
     * out the logical->physical map storage[l1*12 + l2*4 + l0] over x's
     * logical dims [4,2,3]. y/out use those logical dims; a=1, b=0 makes out
     * a pure gather of x through the 3-cycle view. */
    tensor_t *x = buildFloat32TensorND(
        3, (size_t[]){2, 3, 4}, (float[]){0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                                          8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                          16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});
    transposeTensor(x, 0, 1);
    transposeTensor(x, 1, 2);

    tensor_t *y = buildFloat32TensorND(3, (size_t[]){4, 2, 3}, NULL);
    tensor_t *out = buildFloat32TensorND(3, (size_t[]){4, 2, 3}, NULL);

    axpbyFloat32Tensors(1.0f, x, 0.0f, y, out);

    float expected[] = {0.0f,  4.0f,  8.0f,  12.0f, 16.0f, 20.0f, 1.0f,  5.0f,
                        9.0f,  13.0f, 17.0f, 21.0f, 2.0f,  6.0f,  10.0f, 14.0f,
                        18.0f, 22.0f, 3.0f,  7.0f,  11.0f, 15.0f, 19.0f, 23.0f};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, (float *)out->data, 24);

    freeTensor(out);
    freeTensor(y);
    transposeTensor(x, 1, 2); /* unwind the cycle before freeing (Linear.c idiom) */
    transposeTensor(x, 0, 1);
    freeTensor(x);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testAxpbyBasic);
    RUN_TEST(testAxpbyAliasedOutput);
    RUN_TEST(testAxpbyPermutationAwareRead);
    RUN_TEST(testAxpbyThreeCyclePermutationRead);
    return UNITY_END();
}
