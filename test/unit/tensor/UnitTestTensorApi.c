#define SOURCE_FILE "UNIT_TEST_TENSOR_API"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

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

void setUp() {}
void tearDown() {}

/* Forward decl for the file-local factory (definition further down). */
static tensor_t *makeFloatTensorForDistTest(size_t d0, size_t d1);

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
    /* H3 regression: linearLayerInitNonTrainableLegacy passes NULL for grad into
     * parameterInit; today freeParameter dereferences it and crashes. */
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
    return UNITY_END();
}
