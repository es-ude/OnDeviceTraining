#include "npy_dump_sink.h"

#include "DeathTest.h"
#include "NPYLoaderApi.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void setUp() {}
void tearDown() {}

/* Six values, absMax 1.0. Chosen so that every packed grid below is visibly
 * lossy on at least one element (0.3 and -0.125 are off-grid for a 4-bit
 * mantissa/code at any scale derived from absMax 1.0): a dump that merely
 * copied source floats, or raw packed bytes, cannot reproduce the expected
 * arrays. */
static const float kSource[6] = {0.5f, -1.0f, 0.3f, 0.75f, -0.125f, 0.0f};
static const char *kProbes[1] = {"w"};

/* Post-#106 heap fixture: [2,3] tensor owning `q`. */
static tensor_t *makeTensor2x3(quantization_t *q) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *s = reserveMemory(sizeof(shape_t));
    setShape(s, dims, 2, order);
    return initTensor(s, q, NULL);
}

/* Fires the sink as the trace facility would (probe 0 == "w", batch-level
 * naming), then reads <dir>/w.<phase>.npy back through the framework's own
 * .npy reader. Caller owns the returned tensor and removes `pathOut`. */
static tensor_t *dumpAndReload(tensor_t *t, const char *phase, char *pathOut, size_t pathLen) {
    npyDumpCtx_t ctx = {.dir = NPY_DUMP_SINK_TEST_DIR,
                        .probeNames = kProbes,
                        .numProbes = 1,
                        .sampleIdx = NPY_DUMP_NO_SAMPLE};
    npyDumpSink(&ctx, 0, LINEAR, phase, t);
    snprintf(pathOut, pathLen, "%s/w.%s.npy", NPY_DUMP_SINK_TEST_DIR, phase);
    return npyLoadFlat(pathOut);
}

/* Characterization of today's behaviour: a FLOAT32 tensor is written
 * verbatim with its [2,3] shape (this test is GREEN before the change and
 * guards the refactor). */
void testDumpFloat32WritesValuesVerbatim(void) {
    tensor_t *t = makeTensor2x3(quantizationInitFloat());
    tensorFillFromFloatBuffer(t, kSource, 6);
    char path[512];
    tensor_t *back = dumpAndReload(t, "f32", path, sizeof(path));

    float got[6];
    memcpy(got, back->data, sizeof(got));
    qtype_t backType = back->quantization->type;
    size_t ndim = back->shape->numberOfDimensions;
    size_t d0 = back->shape->dimensions[0];
    size_t d1 = back->shape->dimensions[1];

    freeTensor(back);
    freeTensor(t);
    remove(path);

    TEST_ASSERT_EQUAL_INT(FLOAT32, backType);
    TEST_ASSERT_EQUAL_size_t(2, ndim);
    TEST_ASSERT_EQUAL_size_t(2, d0);
    TEST_ASSERT_EQUAL_size_t(3, d1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(kSource, got, 6);
}

/* BFP per-tensor, mantissaBits 4 (qMax 7), exponentBits 8 (bias 127):
 * absMax 1.0 -> E = smallest E with 1.0/2^E <= 7 -> E = -2, scale 0.25
 * (deriveBfpStoredExponent: frexpf(1/7) = 0.5714 * 2^-2). HALF_AWAY
 * mantissas: 2, -4, 1 (0.3/0.25 = 1.2), 3, -1 (-0.125/0.25 = -0.5 -> away
 * from zero), 0. Dequant = mantissa * 0.25. The dump must show THESE grid
 * values (0.25 and -0.25 differ from the 0.3 / -0.125 sources). */
void testDumpBfpWritesDequantizedGridValues(void) {
    static const float expected[6] = {0.5f, -1.0f, 0.25f, 0.75f, -0.25f, 0.0f};
    tensor_t *t = makeTensor2x3(quantizationInitBfp(4, 8, HALF_AWAY));
    tensorFillFromFloatBuffer(t, kSource, 6);
    char path[512];
    tensor_t *back = dumpAndReload(t, "bfp", path, sizeof(path));

    float got[6];
    memcpy(got, back->data, sizeof(got));
    qtype_t backType = back->quantization->type;
    size_t ndim = back->shape->numberOfDimensions;
    size_t d0 = back->shape->dimensions[0];
    size_t d1 = back->shape->dimensions[1];

    freeTensor(back);
    freeTensor(t);
    remove(path);

    TEST_ASSERT_EQUAL_INT(FLOAT32, backType);
    TEST_ASSERT_EQUAL_size_t(2, ndim);
    TEST_ASSERT_EQUAL_size_t(2, d0);
    TEST_ASSERT_EQUAL_size_t(3, d1);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, got, 6);
}

/* Packed SYM (4-bit) rides the SAME conversion-matrix path: the dump equals
 * convertTensor's own SYM->FLOAT32 dequant, and that dequant is visibly
 * lossy on element 2 (0.3 is off every 4-bit grid), so a sink that wrote
 * source floats or packed bytes would fail here too. */
void testDumpSymMatchesConvertTensorDequant(void) {
    tensor_t *t = makeTensor2x3(quantizationInitSym(4, HALF_AWAY));
    tensorFillFromFloatBuffer(t, kSource, 6);
    tensor_t *ref = makeTensor2x3(quantizationInitFloat());
    convertTensor(t, ref);
    float expected[6];
    memcpy(expected, ref->data, sizeof(expected));
    char path[512];
    tensor_t *back = dumpAndReload(t, "sym", path, sizeof(path));

    float got[6];
    memcpy(got, back->data, sizeof(got));

    freeTensor(back);
    freeTensor(ref);
    freeTensor(t);
    remove(path);

    TEST_ASSERT_TRUE_MESSAGE(fabsf(expected[2] - kSource[2]) > 1e-3f,
                             "vacuity guard: 4-bit SYM dequant of 0.3 must be off-grid");
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, got, 6);
}

/* BOOL has no float dequant (no conversionMatrix cell, Dropout-mask dtype):
 * the sink must fail fast (exit 1) instead of writing garbage. */
void testDumpBoolFailsFast(void) {
    tensor_t *t = makeTensor2x3(quantizationInitBool());
    npyDumpCtx_t ctx = {.dir = NPY_DUMP_SINK_TEST_DIR,
                        .probeNames = kProbes,
                        .numProbes = 1,
                        .sampleIdx = NPY_DUMP_NO_SAMPLE};
    ASSERT_EXITS_WITH_FAILURE(npyDumpSink(&ctx, 0, LINEAR, "bool", t));
    freeTensor(t);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testDumpFloat32WritesValuesVerbatim);
    RUN_TEST(testDumpBfpWritesDequantizedGridValues);
    RUN_TEST(testDumpSymMatchesConvertTensorDequant);
    RUN_TEST(testDumpBoolFailsFast);
    return UNITY_END();
}
