#define SOURCE_FILE "UNIT_TEST_JACOBI_EIG"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "JacobiEig.h"
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

void testJacobi2x2Analytic(void) {
    /* [[2,1],[1,2]] -> eigvals {3,1}, eigvecs (1,1)/sqrt2 and (1,-1)/sqrt2. */
    tensor_t *a = buildFloat32TensorND(2, (size_t[]){2, 2}, (float[]){2.0f, 1.0f, 1.0f, 2.0f});
    tensor_t *vals = buildFloat32TensorND(1, (size_t[]){2}, NULL);
    tensor_t *vecs = buildFloat32TensorND(2, (size_t[]){2, 2}, NULL);

    jacobiEigSymFloat32(a, vals, vecs, 30, 1e-6f);

    float *lam = (float *)vals->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.0f, lam[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, lam[1]);

    float *v = (float *)vecs->data;
    float s = 1.0f / sqrtf(2.0f);
    /* Sign-agnostic row checks: |v_0 . (s,s)| == 1, |v_1 . (s,-s)| == 1. */
    float d0 = v[0] * s + v[1] * s;
    float d1 = v[2] * s - v[3] * s;
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, fabsf(d0));
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, fabsf(d1));

    freeTensor(vecs);
    freeTensor(vals);
    freeTensor(a);
}

void testJacobiDiagonalPassthrough(void) {
    /* Already-diagonal input: eigvals = sorted diagonal, vecs = permuted identity. */
    tensor_t *a = buildFloat32TensorND(
        2, (size_t[]){3, 3}, (float[]){1.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    tensor_t *vals = buildFloat32TensorND(1, (size_t[]){3}, NULL);
    tensor_t *vecs = buildFloat32TensorND(2, (size_t[]){3, 3}, NULL);

    jacobiEigSymFloat32(a, vals, vecs, 30, 1e-6f);

    float *lam = (float *)vals->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, lam[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, lam[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, lam[2]);

    float *v = (float *)vecs->data;
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, fabsf(v[0 * 3 + 1])); /* row0 = e1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, fabsf(v[1 * 3 + 2])); /* row1 = e2 */
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, fabsf(v[2 * 3 + 0])); /* row2 = e0 */

    freeTensor(vecs);
    freeTensor(vals);
    freeTensor(a);
}

/* Property test: random symmetric A (fixed constants), reconstruct
 * A ?= V^T diag(lam) V from the ROW-eigvec convention, and check row
 * orthonormality. Self-validating without gold files. */
void testJacobiReconstructionAndOrthonormality(void) {
    /* Fixed 4x4 symmetric fixture (hand-picked, well-conditioned). */
    float src[16] = {4.0f,  1.0f, -2.0f, 0.5f,  1.0f, 3.0f, 0.0f,  1.5f,
                     -2.0f, 0.0f, 5.0f,  -1.0f, 0.5f, 1.5f, -1.0f, 2.0f};
    tensor_t *a = buildFloat32TensorND(2, (size_t[]){4, 4}, src);
    tensor_t *vals = buildFloat32TensorND(1, (size_t[]){4}, NULL);
    tensor_t *vecs = buildFloat32TensorND(2, (size_t[]){4, 4}, NULL);

    jacobiEigSymFloat32(a, vals, vecs, 30, 1e-6f);

    float *lam = (float *)vals->data;
    float *v = (float *)vecs->data;

    /* Descending order. */
    for (size_t i = 0; i + 1 < 4; i++) {
        TEST_ASSERT_TRUE(lam[i] >= lam[i + 1]);
    }
    /* Rows orthonormal. */
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            float dot = 0.0f;
            for (size_t t = 0; t < 4; t++) {
                dot += v[i * 4 + t] * v[j * 4 + t];
            }
            TEST_ASSERT_FLOAT_WITHIN(1e-4f, (i == j) ? 1.0f : 0.0f, dot);
        }
    }
    /* Reconstruction: A[r][c] == sum_i lam[i] * v[i][r] * v[i][c]. */
    for (size_t r = 0; r < 4; r++) {
        for (size_t c = 0; c < 4; c++) {
            float acc = 0.0f;
            for (size_t i = 0; i < 4; i++) {
                acc += lam[i] * v[i * 4 + r] * v[i * 4 + c];
            }
            TEST_ASSERT_FLOAT_WITHIN(1e-3f, src[r * 4 + c], acc);
        }
    }

    freeTensor(vecs);
    freeTensor(vals);
    freeTensor(a);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testJacobi2x2Analytic);
    RUN_TEST(testJacobiDiagonalPassthrough);
    RUN_TEST(testJacobiReconstructionAndOrthonormality);
    return UNITY_END();
}
