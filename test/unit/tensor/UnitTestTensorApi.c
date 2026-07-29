#define SOURCE_FILE "UNIT_TEST_TENSOR_API"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "DeathTest.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* Compile-time contract: initTensor takes shape_t *, quantization_t *,
 * sparsity_t * and returns tensor_t *. No data buffer parameter. */
_Static_assert(_Generic((&initTensor),
                   tensor_t *(*)(shape_t *, quantization_t *, sparsity_t *): 1,
                   default: 0),
               "initTensor must take (shape_t *, quantization_t *, sparsity_t *)");

/* Compile-time contract: initDistribution takes tensor_t * and const distribution_t *. */
_Static_assert(_Generic((&initDistribution),
                   void (*)(tensor_t *, const distribution_t *): 1,
                   default: 0),
               "initDistribution must take (tensor_t *, const distribution_t *)");

/* Compile-time contract: tensorFillFromFloatBuffer takes (tensor_t *, const float *, size_t). */
_Static_assert(_Generic((&tensorFillFromFloatBuffer),
                   void (*)(tensor_t *, const float *, size_t): 1,
                   default: 0),
               "tensorFillFromFloatBuffer must take (tensor_t *, const float *, size_t)");

/* Compile-time contract: quantizationInitBool returns quantization_t *. */
_Static_assert(_Generic((&quantizationInitBool), quantization_t *(*)(void): 1, default: 0),
               "quantizationInitBool must take () and return quantization_t *");

/* Compile-time contract: tensorFillFromBoolBuffer takes (tensor_t *, const bool *, size_t). */
_Static_assert(_Generic((&tensorFillFromBoolBuffer),
                   void (*)(tensor_t *, const bool *, size_t): 1,
                   default: 0),
               "tensorFillFromBoolBuffer must take (tensor_t *, const bool *, size_t)");

/* Compile-time contract: gradInit takes (tensor_t *, quantization_t *, sparsity_t *)
 * and returns tensor_t *. Config-respecting grad-init for PR-0. */
_Static_assert(_Generic((&gradInit),
                   tensor_t *(*)(tensor_t *, quantization_t *, sparsity_t *): 1,
                   default: 0),
               "gradInit must take (tensor_t *, quantization_t *, sparsity_t *)");

/* Compile-time contract: gradInitSym takes (tensor_t *, uint8_t, roundingMode_t,
 * sparsity_t *) and returns tensor_t *, mirroring gradInitAsym. Packed grad
 * storage (#269, PR3). */
_Static_assert(_Generic(&gradInitSym,
                   tensor_t *(*)(tensor_t *, uint8_t, roundingMode_t, sparsity_t *): 1,
                   default: 0),
               "gradInitSym signature contract");

void setUp() {}
void tearDown() {}

/* Forward decl for the file-local factory (definition further down). */
static tensor_t *makeFloatTensorForDistTest(size_t d0, size_t d1);

/* Group-quant PR2 (Task 1) file-local Rule-1 factories: a 1-D shape and a
 * FLOAT32 tensor built over it. */
static shape_t *makeShape1d(size_t n);
static tensor_t *makeFloatTensor1d(size_t n);

void testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues() {
    /* dims = {2, 5} -> product = 10, sum = 7. Pre-fill with sentinel 42.0f,
     * then ZEROS should overwrite exactly 10 values (loop bound = product,
     * not sum). */
    tensor_t *t = makeFloatTensorForDistTest(2, 5);
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 10; i++) {
        vals[i] = 42.0f;
    }

    distribution_t d = {.type = ZEROS};
    initDistribution(t, &d);

    /* CAPTURE before free. */
    float captured[10];
    for (size_t i = 0; i < 10; i++) {
        captured[i] = vals[i];
    }
    freeTensor(t);

    /* ASSERT on captured. */
    for (size_t i = 0; i < 10; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, captured[i]);
    }
}

void testTensorInitWithDistribution_Ones_InitializesAllValues() {
    /* dims = {3, 4} -> product = 12. Pre-fill with 0.0f, then ONES sets all
     * 12 values to 1.0f. */
    tensor_t *t = makeFloatTensorForDistTest(3, 4);
    float *vals = (float *)t->data;
    /* initTensor zero-initializes data; explicit pre-fill kept for clarity. */
    for (size_t i = 0; i < 12; i++) {
        vals[i] = 0.0f;
    }

    distribution_t d = {.type = ONES};
    initDistribution(t, &d);

    /* CAPTURE. */
    float captured[12];
    for (size_t i = 0; i < 12; i++) {
        captured[i] = vals[i];
    }
    freeTensor(t);

    /* ASSERT. */
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, captured[i]);
    }
}

void testTensorInitWithDistribution_Normal_InitializesAllValues() {
    /* dims = {4, 5} -> product = 20, sum = 9. If the loop runs sum-many
     * iterations only, the trailing 11 values stay at sentinel. */
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    float *vals = (float *)t->data;
    float sentinel = -999.0f;
    for (size_t i = 0; i < 20; i++) {
        vals[i] = sentinel;
    }

    distribution_t d = {.type = NORMAL, .params.normal = {0.0f, 0.01f}};
    initDistribution(t, &d);

    /* CAPTURE the derived sentinel count. */
    size_t sentinelCount = 0;
    for (size_t i = 0; i < 20; i++) {
        if (vals[i] == sentinel) {
            sentinelCount++;
        }
    }
    freeTensor(t);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT(0, sentinelCount);
}

void testTensorInitWithDistribution_ShapeIsCorrect() {
    /* Verify the resulting tensor has the correct shape dimensions after
     * initDistribution runs. */
    tensor_t *t = makeFloatTensorForDistTest(2, 3);
    distribution_t d = {.type = ZEROS};
    initDistribution(t, &d);

    /* CAPTURE shape data before free. */
    size_t capturedNumDims = t->shape->numberOfDimensions;
    size_t capturedNumElements = calcNumberOfElementsByTensor(t);
    freeTensor(t);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT(2, capturedNumDims);
    TEST_ASSERT_EQUAL_UINT(6, capturedNumElements);
}

static tensor_t *makeFloatTensorForDistTest(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitFloat(), NULL);
}

static shape_t *makeShape1d(size_t n) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return shape;
}

static tensor_t *makeFloatTensor1d(size_t n) {
    return initTensor(makeShape1d(n), quantizationInitFloat(), NULL);
}

static tensor_t *makeBoolTensorN(size_t n) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = n;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBool(), NULL);
}

void testInitDistribution_Zeros_AllValuesAreZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(3, 4);
    /* Pre-write a sentinel so we can prove ZEROS overwrites. */
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 12; ++i) {
        vals[i] = 42.0f;
    }

    distribution_t d = {.type = ZEROS};
    initDistribution(t, &d);

    /* CAPTURE values before free. */
    float captured[12];
    for (size_t i = 0; i < 12; ++i) {
        captured[i] = vals[i];
    }
    freeTensor(t);

    /* ASSERT on captured. */
    for (size_t i = 0; i < 12; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 0.0f, captured[i]);
    }
}

void testInitDistribution_Ones_AllValuesAreOne(void) {
    tensor_t *t = makeFloatTensorForDistTest(3, 4);
    distribution_t d = {.type = ONES};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE. */
    float captured[12];
    for (size_t i = 0; i < 12; ++i) {
        captured[i] = vals[i];
    }
    freeTensor(t);

    /* ASSERT. */
    for (size_t i = 0; i < 12; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, 1.0f, captured[i]);
    }
}

void testInitDistribution_Uniform_AllValuesInRange(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = UNIFORM, .params.uniform = {-0.5f, 0.5f}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE values + derived bool flag. */
    float captured[20];
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        captured[i] = vals[i];
        if (vals[i] != 0.0f) {
            any_nonzero = true;
        }
    }
    freeTensor(t);

    /* ASSERT on captured. */
    for (size_t i = 0; i < 20; ++i) {
        TEST_ASSERT_TRUE(captured[i] >= -0.5f && captured[i] <= 0.5f);
    }
    TEST_ASSERT_TRUE(any_nonzero);
}

void testInitDistribution_Normal_NotAllSentinel(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    float *vals = (float *)t->data;
    for (size_t i = 0; i < 20; ++i) {
        vals[i] = -999.0f;
    }
    distribution_t d = {.type = NORMAL, .params.normal = {0.0f, 0.01f}};
    initDistribution(t, &d);

    /* CAPTURE the derived sentinel count. */
    size_t sentinelCount = 0;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] == -999.0f) {
            sentinelCount++;
        }
    }
    freeTensor(t);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_UINT(0, sentinelCount);
}

void testInitDistribution_XavierUniform_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = XAVIER_UNIFORM,
                        .params.xavier = {.gain = 1.0f, .fanIn = 4, .fanOut = 5}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE the derived bool flag. */
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    freeTensor(t);

    /* ASSERT on captured. */
    TEST_ASSERT_TRUE(any_nonzero);
}

void testInitDistribution_XavierNormal_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = XAVIER_NORMAL,
                        .params.xavier = {.gain = 1.0f, .fanIn = 4, .fanOut = 5}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE. */
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    freeTensor(t);

    /* ASSERT. */
    TEST_ASSERT_TRUE(any_nonzero);
}

void testInitDistribution_KaimingUniform_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = KAIMING_UNIFORM,
                        .params.kaiming = {.gain = sqrtf(2.0f), .fanMode = 4}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE. */
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    freeTensor(t);

    /* ASSERT. */
    TEST_ASSERT_TRUE(any_nonzero);
}

void testInitDistribution_KaimingNormal_NotAllZero(void) {
    tensor_t *t = makeFloatTensorForDistTest(4, 5);
    distribution_t d = {.type = KAIMING_NORMAL,
                        .params.kaiming = {.gain = sqrtf(2.0f), .fanMode = 4}};
    initDistribution(t, &d);
    float *vals = (float *)t->data;

    /* CAPTURE. */
    bool any_nonzero = false;
    for (size_t i = 0; i < 20; ++i) {
        if (vals[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    freeTensor(t);

    /* ASSERT. */
    TEST_ASSERT_TRUE(any_nonzero);
}

void testTensorFillFromBoolBuffer_RoundTrip_N12(void) {
    /* N=12 → 2 bytes; mixed pattern across byte boundary. */
    const bool source[12] = {true,  false, false, true,  true, false,
                             false, true,  true,  false, true, true};

    tensor_t *t = makeBoolTensorN(12);
    tensorFillFromBoolBuffer(t, source, 12);

    /* CAPTURE before free. */
    bool captured[12];
    for (size_t i = 0; i < 12; i++) {
        captured[i] = tensorBoolGet(t, i);
    }
    freeTensor(t);

    /* ASSERT on captured. */
    for (size_t i = 0; i < 12; i++) {
        TEST_ASSERT_EQUAL(source[i], captured[i]);
    }
}

static void fillTensorFromStackArrayThatGoesOutOfScope(tensor_t *t) {
    /* Local stack array — exits scope when this function returns. */
    float src[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensorFillFromFloatBuffer(t, src, 6);
    /* `src` goes out of scope on return; tensor must keep its own copy. */
}

void testTensorFillFromFloatBuffer_CopiesValues_SourceCanGoOutOfScope(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);

    fillTensorFromStackArrayThatGoesOutOfScope(t);

    /* Force stack reuse — analogous to issue #93's regression pattern. */
    for (int i = 0; i < 100; ++i) {
        volatile float junk[6] = {(float)i, (float)~i, 0, 0, 0, 0};
        (void)junk;
    }

    float *vals = (float *)t->data;
    float expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < 6; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-9f, expected[i], vals[i]);
    }

    freeTensor(t);
}

void testFreeParameter_NullGrad_DoesNotSegfault(void) {
    /* H3 regression: a grad-optional Linear (weights wrapped via parameterInit
     * with NULL grad, formerly built by the deleted linearLayerInitNonTrainableLegacy)
     * used to crash freeParameter, which dereferenced the NULL grad. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 1;
    dims[1] = 2;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);
    parameter_t *p = parameterInit(param, NULL);

    freeParameter(p);
    /* If we reach here, the H3 fix worked. */
    TEST_PASS();
}

void testGradInitFloat_DoesNotAliasParentShape(void) {
    /* H2 regression: gradInit* must allocate a fresh shape instead of aliasing
     * the parent tensor's shape. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);

    tensor_t *grad = gradInitFloat(param, NULL);

    TEST_ASSERT_TRUE_MESSAGE(grad->shape != param->shape,
                             "gradInitFloat aliases parent shape (H2 hazard)");

    /* Free grad first — must not corrupt parent. */
    freeTensor(grad);

    /* Parent's shape must still be readable. */
    TEST_ASSERT_EQUAL_UINT(2, param->shape->numberOfDimensions);
    TEST_ASSERT_EQUAL_UINT(3, param->shape->dimensions[0]);
    TEST_ASSERT_EQUAL_UINT(4, param->shape->dimensions[1]);

    freeTensor(param);
}

void testGradInitInt32_DoesNotAliasParentShape(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitInt32(), NULL);
    tensor_t *grad = gradInitInt32(param, NULL);

    TEST_ASSERT_TRUE_MESSAGE(grad->shape != param->shape,
                             "gradInitInt32 aliases parent shape (H2 hazard)");
    freeTensor(grad);
    TEST_ASSERT_EQUAL_UINT(3, param->shape->dimensions[0]);
    freeTensor(param);
}

void testGradInitSymInt32_DoesNotAliasParentShape(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
    tensor_t *grad = gradInitSymInt32(param, HALF_AWAY, NULL);

    TEST_ASSERT_TRUE_MESSAGE(grad->shape != param->shape,
                             "gradInitSymInt32 aliases parent shape (H2 hazard)");
    freeTensor(grad);
    TEST_ASSERT_EQUAL_UINT(3, param->shape->dimensions[0]);
    freeTensor(param);
}

void testGradInitAsym_DoesNotAliasParentShape(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitAsym(8, HALF_AWAY), NULL);
    tensor_t *grad = gradInitAsym(param, 8, HALF_AWAY, NULL);

    TEST_ASSERT_TRUE_MESSAGE(grad->shape != param->shape,
                             "gradInitAsym aliases parent shape (H2 hazard)");
    freeTensor(grad);
    TEST_ASSERT_EQUAL_UINT(3, param->shape->dimensions[0]);
    freeTensor(param);
}

/* Regression for #108. calcNumberOfBytesForData on the ASYM arm did
 * (bitsPerElement * numberOfElements / 8) with integer arithmetic, then ceilf
 * on the already-truncated result — under-allocating by one byte whenever the
 * total bit count was not a multiple of 8. */
void testCalcNumberOfBytesForData_AsymSubByte_RoundsUpInsteadOfTruncating(void) {
    quantization_t *q = quantizationInitAsym(3, HALF_AWAY);
    /* 10 elements * 3 bits = 30 bits => ceil(30/8) = 4 bytes. */
    size_t bytes = calcNumberOfBytesForData(q, 10);
    freeQuantization(q);
    TEST_ASSERT_EQUAL_UINT(4, bytes);
}

/* Companion to the test above for #108: getDataLike has the same
 * integer-div-before-ceilf shape and under-allocates the data buffer.
 * Under ASan, writing the full 4 expected bytes into a 3-byte allocation
 * trips heap-buffer-overflow; the value-assertion test above catches the
 * arithmetic itself. */
void testGetDataLike_AsymSubByte_AllocatesCeilingOfBits(void) {
    quantization_t *q = quantizationInitAsym(3, HALF_AWAY);
    uint8_t *data = getDataLike(q, 10);
    /* 30 bits => 4 bytes; pre-fix only 3 are allocated. */
    for (size_t i = 0; i < 4; ++i) {
        data[i] = 0xFF;
    }
    freeReservedMemory(data);
    freeQuantization(q);
}

void testInitTensor_Int32_AllocatesFourBytesPerElement(void) {
    /* Closes the calcNumberOfBytesForData gap surfaced by code-review on Task A:
     * the INT32 arm was missing, which would have made gradInitInt32 (Task D)
     * exit(1) on its first call. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    quantization_t *q = quantizationInitInt32();
    tensor_t *t = initTensor(shape, q, NULL);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_NOT_NULL(t->data);
    /* 6 elements × 4 bytes = 24 bytes; all zero. */
    for (size_t i = 0; i < 24; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, t->data[i]);
    }
    freeTensor(t);
}

void testInitTensor_AllocatesOwnZeroDataBuffer_FreeTensorIsSafe(void) {
    /* Build shape via reserveMemory so caller doesn't bypass the locality rule. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    quantization_t *q = quantizationInitFloat();

    tensor_t *t = initTensor(shape, q, NULL);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_NOT_NULL(t->data);
    TEST_ASSERT_EQUAL_PTR(shape, t->shape);
    TEST_ASSERT_EQUAL_PTR(q, t->quantization);
    TEST_ASSERT_NULL(t->sparsity);

    /* All bytes of the data buffer must be zero (calloc semantics). */
    size_t bytes = calcBytesPerTensor(t);
    for (size_t i = 0; i < bytes; ++i) {
        TEST_ASSERT_EQUAL_UINT8(0, t->data[i]);
    }

    /* freeTensor must release everything cleanly without external buffers. */
    freeTensor(t);
    /* If we reach here without an abort/segfault, freeTensor is unconditional-safe
     * for an initTensor-allocated tensor. */
}

void testGradInit_Float32_MatchesParamShapeOwnsOwnQuant(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 3;
    dims[1] = 4;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);

    quantization_t *gradQ = quantizationInitFloat();
    tensor_t *grad = gradInit(param, gradQ, NULL);

    /* CAPTURE before frees. */
    int gradTypeMatches = (grad->quantization->type == FLOAT32);
    int ownsOwnQuant = (grad->quantization != gradQ); /* getQLike deep-clones */
    int ownsOwnShape = (grad->shape != param->shape); /* getShapeLike clones */
    size_t nDims = grad->shape->numberOfDimensions;
    size_t d0 = grad->shape->dimensions[0];
    size_t d1 = grad->shape->dimensions[1];

    freeTensor(grad);
    freeQuantization(gradQ);
    freeTensor(param);

    TEST_ASSERT_TRUE_MESSAGE(gradTypeMatches, "gradInit FLOAT32 grad dtype mismatch");
    TEST_ASSERT_TRUE_MESSAGE(ownsOwnQuant, "gradInit must clone quantization (own it)");
    TEST_ASSERT_TRUE_MESSAGE(ownsOwnShape, "gradInit must clone shape (own it)");
    TEST_ASSERT_EQUAL_UINT(2, nDims);
    TEST_ASSERT_EQUAL_UINT(3, d0);
    TEST_ASSERT_EQUAL_UINT(4, d1);
}

void testGradInit_SymInt32_MatchesParamShapeAndDtype(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 5;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    /* Param can stay FLOAT32; grad dtype is driven solely by gradQ. */
    tensor_t *param = initTensor(shape, quantizationInitFloat(), NULL);

    quantization_t *gradQ = quantizationInitSymInt32(HALF_AWAY);
    tensor_t *grad = gradInit(param, gradQ, NULL);

    int gradTypeMatches = (grad->quantization->type == SYM_INT32);
    int ownsOwnQuant = (grad->quantization != gradQ);
    size_t nDims = grad->shape->numberOfDimensions;
    size_t d0 = grad->shape->dimensions[0];
    size_t d1 = grad->shape->dimensions[1];

    freeTensor(grad);
    freeQuantization(gradQ);
    freeTensor(param);

    TEST_ASSERT_TRUE_MESSAGE(gradTypeMatches, "gradInit SYM_INT32 grad dtype mismatch");
    TEST_ASSERT_TRUE_MESSAGE(ownsOwnQuant, "gradInit must clone SYM_INT32 quantization");
    TEST_ASSERT_EQUAL_UINT(2, nDims);
    TEST_ASSERT_EQUAL_UINT(2, d0);
    TEST_ASSERT_EQUAL_UINT(5, d1);
}

void testGradInitSymInt32StaysInt16WhileDefaultIsInt12() {
    /* default operand config is int12 after the #227 flip */
    symInt32QConfig_t opQC;
    initSymInt32QConfig(HALF_AWAY, &opQC);
    TEST_ASSERT_EQUAL_UINT8(12, opQC.qMaxBits);

    /* a grad accumulator built from a param stays int16 (#45 contract) */
    size_t dims[] = {2, 3};
    size_t order[] = {0, 1};
    shape_t shape = {.dimensions = dims, .numberOfDimensions = 2, .orderOfDimensions = order};
    tensor_t *param = initTensor(getShapeLike(&shape), quantizationInitSymInt32(HALF_AWAY), NULL);
    tensor_t *grad = gradInitSymInt32(param, HALF_AWAY, NULL);
    TEST_ASSERT_EQUAL_UINT8(16, ((symInt32QConfig_t *)grad->quantization->qConfig)->qMaxBits);
    freeTensor(grad);
    freeTensor(param);
}

void testGetQLikeSymPreservesWidthAndRoundingResetsScale(void) {
    /* Precedent A (matches SYM_INT32/ASYM arms): qBits+roundingMode carried, scale
     * reset to 1.f — a fresh clone is an ungridded zero-state. Mutation guard:
     * re-removing the SYM arm exits the run ("Unknown QType"). */
    quantization_t *src = quantizationInitSym(6, SR_HALF_AWAY);
    ((symQConfig_t *)src->qConfig)->scales[0] = 0.25f; /* carried scale must NOT be cloned */
    quantization_t *like = getQLike(src);

    qtype_t likeType = like->type;
    /* Group-quant PR1: capture the scalar VALUES here, not a whole-struct copy
     * -- symQConfig_t now carries a heap `scales` pointer, so `*(symQConfig_t
     * *)like->qConfig` would copy that pointer into a stack local that then
     * dangles once freeQuantization(like) below frees the array (assert-last
     * per Rule 3 still requires reading AFTER the free otherwise). */
    symQConfig_t *likeQC = (symQConfig_t *)like->qConfig;
    uint8_t likeQBits = likeQC->qBits;
    roundingMode_t likeRoundingMode = likeQC->roundingMode;
    float likeScale = likeQC->scales[0];
    freeQuantization(src);
    freeQuantization(like);

    TEST_ASSERT_EQUAL_INT(SYM, likeType);
    TEST_ASSERT_EQUAL_UINT8(6, likeQBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, likeRoundingMode);
    TEST_ASSERT_EQUAL_FLOAT(1.f, likeScale);
}

void testGetQLikeSymDeepCopiesScalesArray(void) {
    /* Group-quant PR1 new mechanics: scales is now a heap array owned by
     * each qconfig independently. Mutate the SOURCE's scales[0] AFTER
     * getQLike -- if getQLike aliased the array pointer instead of
     * allocating a fresh one (via initSymQConfig), the clone would observe
     * the mutation and/or share the same pointer. Mutation guard: making
     * getQLike copy the pointer (`likeQC->scales = srcQC->scales;`) instead
     * of calling initSymQConfig makes this FAIL both assertions. */
    quantization_t *src = quantizationInitSym(6, SR_HALF_AWAY);
    quantization_t *like = getQLike(src);

    float *srcScales = ((symQConfig_t *)src->qConfig)->scales;
    float *likeScales = ((symQConfig_t *)like->qConfig)->scales;

    ((symQConfig_t *)src->qConfig)->scales[0] = 0.75f; /* mutate AFTER clone */

    float likeScaleAfterSrcMutation = likeScales[0];

    freeQuantization(src);
    freeQuantization(like);

    TEST_ASSERT_NOT_EQUAL(srcScales, likeScales);
    TEST_ASSERT_EQUAL_FLOAT(1.f, likeScaleAfterSrcMutation);
}

void testFreeQuantizationSymFreesScalesArrayWithoutLeak(void) {
    /* Group-quant PR1 new mechanics: the SYM qconfig now owns a second heap
     * block (the scales array) beyond the qConfig struct itself and the
     * quantization_t wrapper. freeQuantization must release all of it.
     * Verified leak-free via the LSan opt-in recipe (docs/conventions/
     * testing.md) run against this focused binary -- see the PR1 report.
     * Mutation guard: dropping the array free in freeQuantization leaves
     * this test still "passing" under the default ASan build (no crash),
     * but LSan flags the leak (see report). */
    quantization_t *q = quantizationInitSym(4, HALF_AWAY);
    float *scales = ((symQConfig_t *)q->qConfig)->scales;

    freeQuantization(q);

    TEST_ASSERT_NOT_NULL(scales);
}

void testGetDataLikeSymAllocatesPackedCeiling(void) {
    /* qBits=3, N=10 -> 4 packed bytes; ASan companion: write all 4, free safely.
     * Mutation guard: sizing via numberOfValues * calcBytesPerElement would give 10 —
     * assert allocation is usable exactly like calcNumberOfBytesForData says. */
    quantization_t *q = quantizationInitSym(3, HALF_AWAY);
    uint8_t *data = getDataLike(q, 10);
    for (size_t i = 0; i < 4; i++) {
        data[i] = 0xFF;
    }

    /* Scoped second quantization to derive the expected byte count without
     * re-touching `q` after `data` is freed (assert-last discipline). */
    quantization_t *q2 = quantizationInitSym(3, HALF_AWAY);
    size_t expectedBytes = calcNumberOfBytesForData(q2, 10);

    freeReservedMemory(data);
    freeQuantization(q);
    freeQuantization(q2);

    TEST_ASSERT_EQUAL_size_t(4, expectedBytes);
}

void testGradInitSymAllocatesPackedZeroGrad(void) {
    /* Grad = SYM(qBits) clone of the param's SHAPE (not dtype), zero-filled,
     * packed-size buffer. Mutation guard: routing through getQLike-of-param
     * would yield FLOAT32; asserting SYM+qBits catches it. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 5;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *p = initTensor(shape, quantizationInitFloat(), NULL);
    tensor_t *g = gradInitSym(p, 4, SR_HALF_AWAY, NULL);

    qtype_t gType = g->quantization->type;
    symQConfig_t gQC = *(symQConfig_t *)g->quantization->qConfig;
    size_t gBytes = calcBytesPerTensor(g);
    uint8_t byte0 = g->data[0];
    freeTensor(g);
    freeTensor(p);

    TEST_ASSERT_EQUAL_INT(SYM, gType);
    TEST_ASSERT_EQUAL_UINT8(4, gQC.qBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, gQC.roundingMode);
    TEST_ASSERT_EQUAL_size_t(5, gBytes); /* ceil(40/8) */
    TEST_ASSERT_EQUAL_UINT8(0, byte0);   /* calloc zero-fill */
}

/* Group-quant PR2 (Task 1): attach-time validation, group-faithful getQLike,
 * and the grad-carrier gate. */

void testInitTensorValidatesGroupedSymShape(void) {
    /* 10-element tensor, groupSize 4 -> 2*4 != 10 must fail-fast at attach */
    ASSERT_EXITS_WITH(1, {
        quantization_t *q = quantizationInitSymGrouped(4, HALF_AWAY, 2, 4);
        initTensor(makeShape1d(10), q, NULL);
    });
}

void testGetQLikeSymPreservesGroups(void) {
    quantization_t *src = quantizationInitSymGrouped(4, HALF_AWAY, 2, 5);
    ((symQConfig_t *)src->qConfig)->scales[1] = 0.25f;
    quantization_t *like = getQLike(src);
    symQConfig_t *likeQC = like->qConfig;
    size_t ng = likeQC->numGroups, gs = likeQC->groupSize;
    float s1 = likeQC->scales[1];
    int distinct = likeQC->scales != ((symQConfig_t *)src->qConfig)->scales;
    freeQuantization(like);
    freeQuantization(src);
    TEST_ASSERT_EQUAL_size_t(2, ng);
    TEST_ASSERT_EQUAL_size_t(5, gs);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, s1); /* scales VALUES carried over */
    TEST_ASSERT_TRUE(distinct);         /* but deep-copied, not aliased */
}

void testGradInitRejectsGroupedSymTemplate(void) {
    ASSERT_EXITS_WITH(1, {
        quantization_t *gq = quantizationInitSymGrouped(8, HALF_AWAY, 2, 5);
        tensor_t *p = makeFloatTensor1d(10); /* file-local Rule-1 factory */
        gradInit(p, gq, NULL);
    });
}

/* Final-review Fix 1 (CRITICAL, heap-OOB): requantizeTensorInPlace builds its
 * destination view directly (getQLike + getDataLike), bypassing initTensor's
 * validateSymQConfigShape choke point entirely. A grouped SYM target whose
 * numGroups*groupSize does not equal the SOURCE tensor's actual element
 * count sizes the data buffer/view off the SOURCE's count while the group
 * shape (and therefore the number of scales the SYM pack/unpack path will
 * index) come from the TARGET template -- a 12-element FLOAT32 source
 * requantized against a {numGroups=2, groupSize=4} target (implies 8
 * elements) walks group indices up to 12/4-1=2, reading scales[2] out of a
 * 2-element scales[] array (ASan-confirmed heap-OOB read in
 * packFloatBufferAsSym's Phase 2). Must fail-fast before either buffer is
 * touched. */
void testRequantizeTensorInPlaceRejectsMismatchedGroupShape(void) {
    ASSERT_EXITS_WITH(1, {
        tensor_t *t = makeFloatTensor1d(12); /* file-local Rule-1 factory */
        quantization_t *targetQ = quantizationInitSymGrouped(4, HALF_AWAY, 2, 4);
        requantizeTensorInPlace(t, targetQ);
    });
}

/* Group-quant PR4 (Task 1): ASYM always-array siblings of the SYM tests
 * above -- attach-time validation, the getQLike per-tensor-fresh-reset /
 * grouped-deep-copy split, and the grad-carrier gate. */

void testGetQLikeAsymPreservesWidthAndRoundingResetsGrid(void) {
    /* Precedent A (per-tensor): qBits + roundingMode carried, the affine grid
     * (scales[0]/zeroPoints[0]) reset to 1.f/0 -- a fresh clone is an
     * ungridded zero-state (first store derives the grid; zp==0 also makes
     * code 0 decode to exactly 0.0, the zero-grad reset state). */
    quantization_t *src = quantizationInitAsym(5, SR_HALF_AWAY);
    asymQConfig_t *srcQC = src->qConfig;
    srcQC->scales[0] = 0.25f; /* carried grid must NOT be cloned */
    srcQC->zeroPoints[0] = 7;
    quantization_t *like = getQLike(src);

    qtype_t likeType = like->type;
    asymQConfig_t *likeQC = like->qConfig;
    uint8_t likeQBits = likeQC->qBits;
    roundingMode_t likeRoundingMode = likeQC->roundingMode;
    size_t likeNumGroups = likeQC->numGroups;
    size_t likeGroupSize = likeQC->groupSize;
    float likeScale = likeQC->scales[0];
    uint16_t likeZp = likeQC->zeroPoints[0];
    freeQuantization(src);
    freeQuantization(like);

    TEST_ASSERT_EQUAL_INT(ASYM, likeType);
    TEST_ASSERT_EQUAL_UINT8(5, likeQBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, likeRoundingMode);
    TEST_ASSERT_EQUAL_size_t(1, likeNumGroups);
    TEST_ASSERT_EQUAL_size_t(0, likeGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(1.f, likeScale);
    TEST_ASSERT_EQUAL_UINT16(0, likeZp);
}

void testGetQLikeAsymGroupedDeepCopiesGridValues(void) {
    /* Grouped clone: the group grid is an attach-time fact the clone must
     * retain (mirrors the SYM grouped arm) -- numGroups/groupSize AND both
     * per-group arrays' VALUES carried, into FRESH arrays. Mutation guard:
     * aliasing either array pointer instead of deep-copying makes the
     * distinctness asserts (and the mutate-after-clone probe) FAIL. */
    quantization_t *src = quantizationInitAsymGrouped(4, HALF_AWAY, 2, 5);
    asymQConfig_t *srcQC = src->qConfig;
    srcQC->scales[1] = 0.25f;
    srcQC->zeroPoints[1] = 9;
    quantization_t *like = getQLike(src);
    asymQConfig_t *likeQC = like->qConfig;
    size_t ng = likeQC->numGroups, gs = likeQC->groupSize;
    float s1 = likeQC->scales[1];
    uint16_t z1 = likeQC->zeroPoints[1];
    int scalesDistinct = likeQC->scales != srcQC->scales;
    int zpsDistinct = likeQC->zeroPoints != srcQC->zeroPoints;
    srcQC->scales[1] = 0.75f; /* mutate AFTER clone */
    srcQC->zeroPoints[1] = 3;
    float s1AfterSrcMutation = likeQC->scales[1];
    uint16_t z1AfterSrcMutation = likeQC->zeroPoints[1];
    freeQuantization(like);
    freeQuantization(src);
    TEST_ASSERT_EQUAL_size_t(2, ng);
    TEST_ASSERT_EQUAL_size_t(5, gs);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, s1);
    TEST_ASSERT_EQUAL_UINT16(9, z1);
    TEST_ASSERT_TRUE(scalesDistinct);
    TEST_ASSERT_TRUE(zpsDistinct);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, s1AfterSrcMutation);
    TEST_ASSERT_EQUAL_UINT16(9, z1AfterSrcMutation);
}

void testInitTensorValidatesGroupedAsymShape(void) {
    /* 10-element tensor, groupSize 4 -> 2*4 != 10 must fail-fast at attach
     * (initTensor's validation branch now covers ASYM, not just SYM). */
    ASSERT_EXITS_WITH(1, {
        quantization_t *q = quantizationInitAsymGrouped(4, HALF_AWAY, 2, 4);
        initTensor(makeShape1d(10), q, NULL);
    });
}

void testGradInitRejectsGroupedAsymTemplate(void) {
    /* Carrier gate: grads stay per-tensor unconditionally (#300 axis), for
     * ASYM exactly as for SYM. */
    ASSERT_EXITS_WITH(1, {
        quantization_t *gq = quantizationInitAsymGrouped(8, HALF_AWAY, 2, 5);
        tensor_t *p = makeFloatTensor1d(10); /* file-local Rule-1 factory */
        gradInit(p, gq, NULL);
    });
}

void testRequantizeTensorInPlaceRejectsMismatchedAsymGroupShape(void) {
    /* ASYM twin of the SYM bypass-hazard death test above (group-quant PR4
     * Task 2 verify): requantizeTensorInPlace's hand-built view skips
     * initTensor's validateAsymQConfigShape choke point, so its own validate
     * branch must catch a grouped ASYM target whose numGroups*groupSize (2*4)
     * does not equal the SOURCE tensor's element count (12) BEFORE the data
     * buffer is sized -- otherwise the grouped pack path indexes
     * scales[]/zeroPoints[] past their numGroups-sized arrays. */
    ASSERT_EXITS_WITH(1, {
        tensor_t *t = makeFloatTensor1d(12); /* file-local Rule-1 factory */
        quantization_t *targetQ = quantizationInitAsymGrouped(4, HALF_AWAY, 2, 4);
        requantizeTensorInPlace(t, targetQ);
    });
}

/* BFP epic PR1 Task 6: owner-chain arms (getQLike/getDataLike/
 * freeQuantization), the two userApi factories, and the requantize pin. */

void testGetQLikeBfpPerTensorResetsExponents(void) {
    /* BFP twin of testGetQLikeSymPreservesWidthAndRoundingResetsScale:
     * widths + rounding carried, exponent reset to the fresh zero-state
     * (bias) -- a fresh per-tensor clone is an ungridded zero-state.
     * Mutation guard: re-removing the BFP arm exits the run ("Unknown
     * QType"). */
    quantization_t *src = quantizationInitBfp(8, 8, SR_HALF_AWAY);
    ((bfpQConfig_t *)src->qConfig)->exponents[0] = 200; /* non-bias sentinel */
    quantization_t *like = getQLike(src);

    qtype_t likeType = like->type;
    bfpQConfig_t *likeQC = (bfpQConfig_t *)like->qConfig;
    uint8_t likeMantissaBits = likeQC->mantissaBits;
    uint8_t likeExponentBits = likeQC->exponentBits;
    roundingMode_t likeRoundingMode = likeQC->roundingMode;
    uint8_t likeExponent0 = likeQC->exponents[0];
    int32_t bias = bfpExponentBias(likeQC);
    int notAliased = likeQC->exponents != ((bfpQConfig_t *)src->qConfig)->exponents;
    freeQuantization(src);
    freeQuantization(like);

    TEST_ASSERT_EQUAL_INT(BFP, likeType);
    TEST_ASSERT_EQUAL_UINT8(8, likeMantissaBits);
    TEST_ASSERT_EQUAL_UINT8(8, likeExponentBits);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, likeRoundingMode);
    TEST_ASSERT_EQUAL_UINT8((uint8_t)bias, likeExponent0);
    TEST_ASSERT_TRUE(notAliased);
}

void testGetQLikeBfpGroupedDeepCopiesGrid(void) {
    /* BFP twin of testGetQLikeSymPreservesGroups: a grouped source's group
     * SHAPE is an attach-time fact -- the clone must preserve
     * numGroups/groupSize AND deep-copy the exponent VALUES, not alias the
     * array. Mutation guard: aliasing (`likeQC->exponents = qc->exponents`)
     * makes the post-clone-mutation assertion below FAIL. */
    quantization_t *src = quantizationInitBfpGrouped(3, 8, HALF_AWAY, 3, 8);
    bfpQConfig_t *srcQC = (bfpQConfig_t *)src->qConfig;
    srcQC->exponents[0] = 10;
    srcQC->exponents[1] = 20;
    srcQC->exponents[2] = 30;
    quantization_t *like = getQLike(src);
    bfpQConfig_t *likeQC = (bfpQConfig_t *)like->qConfig;

    size_t ng = likeQC->numGroups, gs = likeQC->groupSize;
    uint8_t e0 = likeQC->exponents[0];
    uint8_t e1 = likeQC->exponents[1];
    uint8_t e2 = likeQC->exponents[2];
    int notAliased = likeQC->exponents != srcQC->exponents;

    likeQC->exponents[0] = 99; /* mutate the CLONE after capture */
    uint8_t srcExponent0AfterCloneMutation = srcQC->exponents[0];

    freeQuantization(like);
    freeQuantization(src);

    TEST_ASSERT_EQUAL_size_t(3, ng);
    TEST_ASSERT_EQUAL_size_t(8, gs);
    TEST_ASSERT_EQUAL_UINT8(10, e0);
    TEST_ASSERT_EQUAL_UINT8(20, e1);
    TEST_ASSERT_EQUAL_UINT8(30, e2);
    TEST_ASSERT_TRUE(notAliased);
    TEST_ASSERT_EQUAL_UINT8(10, srcExponent0AfterCloneMutation); /* source untouched */
}

void testGetDataLikeBfpSizesPacked(void) {
    /* mantissaBits=6, N=10 -> ceil(60/8) = 8 packed bytes via the single
     * ceiling authority (calcNumberOfBytesForData), same idiom as
     * testGetDataLikeSymAllocatesPackedCeiling. */
    quantization_t *q = quantizationInitBfp(6, 8, HALF_AWAY);
    uint8_t *data = getDataLike(q, 10);
    for (size_t i = 0; i < 8; i++) {
        data[i] = 0xFF;
    }

    quantization_t *q2 = quantizationInitBfp(6, 8, HALF_AWAY);
    size_t expectedBytes = calcNumberOfBytesForData(q2, 10);

    freeReservedMemory(data);
    freeQuantization(q);
    freeQuantization(q2);

    TEST_ASSERT_EQUAL_size_t(8, expectedBytes);
}

void testFreeQuantizationBfpFreesExponents(void) {
    /* BFP twin of testFreeQuantizationSymFreesScalesArrayWithoutLeak: the
     * BFP qconfig owns a second heap block (the exponents array) beyond the
     * qConfig struct and the quantization_t wrapper -- freeQuantization must
     * release all of it. Leak-freedom is enforced by CI's Linux ASan/LSan
     * job on this binary (macOS ASan init hangs locally, known issue); this
     * test's own pass/fail signal is that the free completes without a
     * double-free/invalid-free abort. */
    quantization_t *q = quantizationInitBfp(8, 8, HALF_AWAY);
    uint8_t *exponents = ((bfpQConfig_t *)q->qConfig)->exponents;

    freeQuantization(q);

    TEST_ASSERT_NOT_NULL(exponents);
}

void testRequantizeTensorInPlaceFloatToBfp(void) {
    /* Task 6 Step 1 pin: requantizeTensorInPlace needs NO code change to
     * support a BFP target -- it is already dtype-generic (getQLike +
     * getDataLike + convertTensor). Round-trip FLOAT32 -> grouped BFP ->
     * FLOAT32 via the promoted public API twice (mirrors
     * testRequantizeTensorInPlaceGrouped's SYM idiom, UnitTestTensorConversion.c).
     * Values chosen so both groups round-trip EXACTLY: group0 {8,-2,4,0}
     * absMax 8 -> derived E=+1 (stored 128, scale 2, matches
     * testFloatToBfpSnapsUpAndPowerOfTwoIsExact's known-good derivation);
     * group1 {28,-8,4,16} absMax 28 -> derived E=+2 (stored 129, scale 4,
     * matches testFloatToBfpGroupedIndependentExponents' absMax). Neither
     * stored exponent equals the zero-state bias (127), so a mutation that
     * silently resets exponents to bias (instead of deriving them from the
     * pack) is caught even though getQLike's OWN grouped clone deep-copies
     * -- the derivation under test happens inside convertTensor, downstream
     * of getQLike's zero-state-valued clone of the target template. */
    tensor_t *t = makeFloatTensor1d(8);
    tensorFillFromFloatBuffer(t, (float[]){8.f, -2.f, 4.f, 0.f, 28.f, -8.f, 4.f, 16.f}, 8);

    quantization_t *targetQ = quantizationInitBfpGrouped(4, 8, HALF_AWAY, 2, 4);
    requantizeTensorInPlace(t, targetQ);
    freeQuantization(targetQ); /* getQLike deep-clones -- template unused after */

    TEST_ASSERT_EQUAL_INT(BFP, t->quantization->type);
    bfpQConfig_t *qc = t->quantization->qConfig;
    TEST_ASSERT_EQUAL_size_t(2, qc->numGroups);
    TEST_ASSERT_EQUAL_size_t(4, qc->groupSize);
    TEST_ASSERT_EQUAL_UINT8(128, qc->exponents[0]);     /* derived E=+1 */
    TEST_ASSERT_EQUAL_UINT8(129, qc->exponents[1]);     /* derived E=+2 */
    TEST_ASSERT_EQUAL_size_t(4, calcBytesPerTensor(t)); /* ceil(4bits*8/8) packed */

    quantization_t *backQ = quantizationInitFloat();
    requantizeTensorInPlace(t, backQ);
    freeQuantization(backQ);

    TEST_ASSERT_EQUAL_INT(FLOAT32, t->quantization->type);
    float expected[8] = {8.f, -2.f, 4.f, 0.f, 28.f, -8.f, 4.f, 16.f};
    float *got = (float *)t->data;
    for (size_t i = 0; i < 8; i++) {
        TEST_ASSERT_EQUAL_FLOAT(expected[i], got[i]);
    }

    freeTensor(t);
}

void testGradInitRejectsBfpTemplate(void) {
    /* BFP twin of testGradInitRejectsGroupedSymTemplate: BFP grad/state
     * storage is out of scope for this epic PR (lands with BFP epic PR3) --
     * gradInit must reject ANY BFP template, not just a grouped one. */
    ASSERT_EXITS_WITH(1, {
        quantization_t *bfpQ = quantizationInitBfp(8, 8, HALF_AWAY);
        tensor_t *p = makeFloatTensor1d(4); /* file-local Rule-1 factory */
        gradInit(p, bfpQ, NULL);
    });
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testTensorInitWithDistribution_Zeros_InitializesProductOfDimsValues);
    RUN_TEST(testTensorInitWithDistribution_Ones_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_Normal_InitializesAllValues);
    RUN_TEST(testTensorInitWithDistribution_ShapeIsCorrect);
    RUN_TEST(testInitTensor_AllocatesOwnZeroDataBuffer_FreeTensorIsSafe);
    RUN_TEST(testInitTensor_Int32_AllocatesFourBytesPerElement);
    RUN_TEST(testCalcNumberOfBytesForData_AsymSubByte_RoundsUpInsteadOfTruncating);
    RUN_TEST(testGetDataLike_AsymSubByte_AllocatesCeilingOfBits);
    RUN_TEST(testTensorFillFromFloatBuffer_CopiesValues_SourceCanGoOutOfScope);
    RUN_TEST(testFreeParameter_NullGrad_DoesNotSegfault);
    RUN_TEST(testGradInitFloat_DoesNotAliasParentShape);
    RUN_TEST(testGradInitInt32_DoesNotAliasParentShape);
    RUN_TEST(testGradInitSymInt32_DoesNotAliasParentShape);
    RUN_TEST(testGradInitAsym_DoesNotAliasParentShape);
    RUN_TEST(testGradInit_Float32_MatchesParamShapeOwnsOwnQuant);
    RUN_TEST(testGradInit_SymInt32_MatchesParamShapeAndDtype);
    RUN_TEST(testInitDistribution_Zeros_AllValuesAreZero);
    RUN_TEST(testInitDistribution_Ones_AllValuesAreOne);
    RUN_TEST(testInitDistribution_Uniform_AllValuesInRange);
    RUN_TEST(testInitDistribution_Normal_NotAllSentinel);
    RUN_TEST(testInitDistribution_XavierUniform_NotAllZero);
    RUN_TEST(testInitDistribution_XavierNormal_NotAllZero);
    RUN_TEST(testInitDistribution_KaimingUniform_NotAllZero);
    RUN_TEST(testInitDistribution_KaimingNormal_NotAllZero);
    RUN_TEST(testTensorFillFromBoolBuffer_RoundTrip_N12);
    RUN_TEST(testGradInitSymInt32StaysInt16WhileDefaultIsInt12);
    RUN_TEST(testGetQLikeSymPreservesWidthAndRoundingResetsScale);
    RUN_TEST(testGetQLikeSymDeepCopiesScalesArray);
    RUN_TEST(testFreeQuantizationSymFreesScalesArrayWithoutLeak);
    RUN_TEST(testGetDataLikeSymAllocatesPackedCeiling);
    RUN_TEST(testGradInitSymAllocatesPackedZeroGrad);
    RUN_TEST(testInitTensorValidatesGroupedSymShape);
    RUN_TEST(testGetQLikeSymPreservesGroups);
    RUN_TEST(testGradInitRejectsGroupedSymTemplate);
    RUN_TEST(testRequantizeTensorInPlaceRejectsMismatchedGroupShape);
    RUN_TEST(testGetQLikeAsymPreservesWidthAndRoundingResetsGrid);
    RUN_TEST(testGetQLikeAsymGroupedDeepCopiesGridValues);
    RUN_TEST(testInitTensorValidatesGroupedAsymShape);
    RUN_TEST(testGradInitRejectsGroupedAsymTemplate);
    RUN_TEST(testRequantizeTensorInPlaceRejectsMismatchedAsymGroupShape);
    RUN_TEST(testGetQLikeBfpPerTensorResetsExponents);
    RUN_TEST(testGetQLikeBfpGroupedDeepCopiesGrid);
    RUN_TEST(testGetDataLikeBfpSizesPacked);
    RUN_TEST(testFreeQuantizationBfpFreesExponents);
    RUN_TEST(testRequantizeTensorInPlaceFloatToBfp);
    RUN_TEST(testGradInitRejectsBfpTemplate);
    return UNITY_END();
}
