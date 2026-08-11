#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ArithmeticType.h"
#include "DeathTest.h"
#include "Deserialize.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "LayerQuant.h"
#include "QuantizationApi.h"
#include "Relu.h"
#include "ReluApi.h"
#include "Serialize.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "unity.h"

/* SERIALIZE_TEST_FILE_PATH is injected by the CMake target_compile_definitions
 * in test/unit/serial/CMakeLists.txt as an absolute path so the test does not
 * depend on the working directory (which differs between host runs and Docker
 * LSan runs). */
#define FILE_PATH SERIALIZE_TEST_FILE_PATH

/* Fixture writer for hand-crafted v2 files: explicit little-endian bytes, so
 * the fixtures stay valid even on a big-endian test host. */
static void writeU32LE(FILE *f, uint32_t value) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16),
                        (uint8_t)(value >> 24)};
    fwrite(bytes, 1, 4, f);
}

/* Companion to writeU32LE for fixtures that need to write more than a
 * handful of scale floats (e.g. an over-cap group count) without hand-typing
 * every byte. */
static void writeF32LE(FILE *f, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    writeU32LE(f, bits);
}

/* group-quant PR4 (Task 4): companion for the v5 ASYM record's
 * zeroPoints[numGroups] u16 LE array. */
static void writeU16LE(FILE *f, uint16_t value) {
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    fwrite(bytes, 1, 2, f);
}

static tensor_t *makeFloatTensor2D(size_t d0, size_t d1, const float *src, size_t count) {
    /* Heap-tier construction per CONVENTIONS Rule 1. */
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);

    tensor_t *t = initTensor(shape, quantizationInitFloat(), NULL);
    if (src != NULL) {
        tensorFillFromFloatBuffer(t, src, count);
    }
    return t;
}

void testSerializeAndDeserializeTensor() {
    size_t numberOfValues = 6;
    float data[] = {9, 9, 9, 4.5f, 2.1112f, 999.123f};

    tensor_t *serialTensor = makeFloatTensor2D(2, 3, data, numberOfValues);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(serialTensor, f);
    fclose(f);

    /* Heap-allocated zero-init buffer destination via initTensor. */
    tensor_t *deserialTensor = makeFloatTensor2D(2, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(deserialTensor, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedDeserialData[6];
    for (size_t i = 0; i < numberOfValues; i++) {
        capturedDeserialData[i] = ((float *)deserialTensor->data)[i];
    }
    qtype_t capturedSerialQType = serialTensor->quantization->type;
    qtype_t capturedDeserialQType = deserialTensor->quantization->type;
    size_t capturedSerialNumDims = serialTensor->shape->numberOfDimensions;
    size_t capturedDeserialNumDims = deserialTensor->shape->numberOfDimensions;

    size_t capturedSerialDims[2];
    size_t capturedDeserialDims[2];
    size_t capturedSerialOrder[2];
    size_t capturedDeserialOrder[2];
    for (size_t i = 0; i < 2; i++) {
        capturedSerialDims[i] = serialTensor->shape->dimensions[i];
        capturedDeserialDims[i] = deserialTensor->shape->dimensions[i];
        capturedSerialOrder[i] = serialTensor->shape->orderOfDimensions[i];
        capturedDeserialOrder[i] = deserialTensor->shape->orderOfDimensions[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(deserialTensor);
    freeTensor(serialTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, capturedDeserialData, numberOfValues);
    TEST_ASSERT_EQUAL(capturedSerialQType, capturedDeserialQType);
    TEST_ASSERT_EQUAL(capturedSerialNumDims, capturedDeserialNumDims);
    TEST_ASSERT_EQUAL_size_t_ARRAY(capturedSerialDims, capturedDeserialDims, 2);
    TEST_ASSERT_EQUAL_size_t_ARRAY(capturedSerialOrder, capturedDeserialOrder, 2);
}

/* Hand-crafted malformed files exercising deserializeModel's NEW validation
 * (bad magic / wrong version / layerCount mismatch / tag mismatch). A single
 * Flatten layer is the minimal pre-built mirror model — Flatten needs no
 * quantization setup, so these tests isolate the header/tag validation from
 * any per-layer-type record decoding. */

static void testDeserializeRejectsBadMagic(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("XXXX", 1, 4, f);
    writeU32LE(f, 5); /* version (dead value: bad magic short-circuits first) */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsWrongVersion(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 1); /* v1 files are host-local artifacts; no back-compat shim */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

/*! group-quant PR1: v3 was a REAL shipped format (pre-group-quant SYM record:
 *  scale/qBits/rounding only, no numGroups/groupSize) -- distinct from the
 *  arbitrary-old-version case above (v1). A v3 file must fail cleanly at the
 *  version check now that SERIALIZE_FORMAT_VERSION is 5, exactly like any
 *  other stale version (no back-compat shim, established policy). */
static void testDeserializeRejectsV3Version(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 3); /* v3: pre-group-quant SYM record layout */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

/*! group-quant PR4, Task 4: v4 was ALSO a REAL shipped format (the interim
 *  state Tasks 1-3 produced: SYM record group-general, but ASYM left on the
 *  intra-branch bridge -- old per-tensor shape, code-domain zp smuggled
 *  through the old i32 slot). Task 4's v5 bump offers NO migration path from
 *  it: a v4 ASYM record's zp decoded WRONG under the code-domain grid by
 *  design of that interim bridge (see Serialize.c/Deserialize.c's v4/v5
 *  history comments), so a v4 file -- including ones this same branch's own
 *  Tasks 1-3 could have produced -- must now fail cleanly at the version
 *  check, exactly like testDeserializeRejectsV3Version above. This closes
 *  Task 1's disclosed mis-decode window: a v4 file can no longer be silently
 *  read into a code-domain config at all, correct or not. */
static void testDeserializeRejectsV4Version(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 4); /* v4: SYM group-general, ASYM still on the v4 bridge */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsLayerCountMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 2); /* layerCount; caller below passes sizeModel = 1 */
    uint8_t tag = (uint8_t)FLATTEN;
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static void testDeserializeRejectsTagMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5);              /* version */
    writeU32LE(f, 1);              /* layerCount */
    uint8_t tag = (uint8_t)LINEAR; /* pre-built mirror layer below is FLATTEN */
    fwrite(&tag, sizeof(uint8_t), 1, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

static tensor_t *makeSymInt32Tensor2D(size_t d0, size_t d1) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = d0;
    dims[1] = d1;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    return initTensor(shape, quantizationInitSymInt32(HALF_AWAY), NULL);
}

/* #316: a checkpoint whose per-tensor dtype differs from the pre-built skeleton
 * must be rejected before deserializeQConfig writes qConfig fields through a
 * mismatched (or NULL, for FLOAT32/INT32/BOOL) pointer, and before the payload
 * fread can overflow the skeleton's allocation. Here a FLOAT32 record is loaded
 * into a SYM_INT32-built skeleton — pre-fix it silently overwrites the dtype;
 * the reverse (SYM into a FLOAT32 skeleton) NULL-derefs. */
void testDeserializeTensorRejectsDtypeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *floatTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(floatTensor, f);
    fclose(f);

    tensor_t *symSkeleton = makeSymInt32Tensor2D(2, 3);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(symSkeleton, f));
    fclose(f);

    freeTensor(symSkeleton);
    freeTensor(floatTensor);
}

/* #316: a same-dtype record whose element count differs from the skeleton would
 * fread past tensor->data, which initTensor sized from the build-time shape. The
 * payload-size check catches size changes (shape- or packed-qBits-driven) that
 * the dtype check alone misses. */
void testDeserializeTensorRejectsPayloadSizeMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *bigTensor = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(bigTensor, f);
    fclose(f);

    /* Same dtype (FLOAT32) and rank, but a smaller allocation (2x2 = 4 elems). */
    tensor_t *smallSkeleton = makeFloatTensor2D(2, 2, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(smallSkeleton, f));
    fclose(f);

    freeTensor(smallSkeleton);
    freeTensor(bigTensor);
}

/* #370: the issue's named dtype-mismatch direction — a SYM record loaded into a
 * FLOAT32-built skeleton (whose qConfig is NULL) must fail fast; pre-#316 this
 * NULL-derefed in deserializeQConfig. */
static void testDeserializeTensorRejectsSymRecordIntoFloatSkeleton(void) {
    size_t *dims = reserveMemory(2 * sizeof(size_t));
    dims[0] = 2;
    dims[1] = 3;
    size_t *order = reserveMemory(2 * sizeof(size_t));
    setOrderOfDimsForNewTensor(2, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 2, order);
    tensor_t *symTensor = initTensor(shape, quantizationInitSym(4, HALF_AWAY), NULL);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(symTensor, f);
    fclose(f);

    tensor_t *floatSkeleton = makeFloatTensor2D(2, 3, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(floatSkeleton, f));
    fclose(f);

    freeTensor(floatSkeleton);
    freeTensor(symTensor);
}

/* #370: a header cut mid-field (magic + half the version u32) must fail fast —
 * pre-v2 the unchecked fread left the version uninitialized/garbage. */
static void testDeserializeModelFailsFastOnTruncatedHeader(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    uint8_t partialVersion[2] = {0x02, 0x00};
    fwrite(partialVersion, 1, 2, f);
    fclose(f);

    layer_t *layer = flattenLayerInit();
    layer_t *model[] = {layer};
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeFlattenLayer(layer);
}

/* #370: a stream cut inside the DATA payload must fail fast at the payload
 * read — pre-v2 the unchecked fread deserialized the truncation as silent
 * garbage (the trailing elements simply kept their zero-init). */
static void testDeserializeTensorFailsFastOnTruncatedPayload(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *src = makeFloatTensor2D(2, 3, data, 6);
    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    long full = ftell(f);
    fclose(f);

    FILE *in = fopen(FILE_PATH, "rb");
    uint8_t *buf = reserveMemory((size_t)full);
    fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full - 2, f);
    fclose(f);
    freeReservedMemory(buf);

    tensor_t *skeleton = makeFloatTensor2D(2, 3, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
    freeTensor(src);
}

/* #370: file rank 1 x [6] into a rank-2 [6,1] skeleton keeps the element count
 * equal, so the #316 payload-size check alone cannot see it — pre-v2 the file
 * rank silently overwrote the skeleton's (and a LARGER file rank wrote dims
 * past the skeleton's arrays). The v2 rank guard must reject it. */
static void testDeserializeTensorRejectsRankMismatch(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = 6;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    tensor_t *src = initTensor(shape, quantizationInitFloat(), NULL);
    tensorFillFromFloatBuffer(src, data, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeFloatTensor2D(6, 1, NULL, 0);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
    freeTensor(src);
}

static tensor_t *makeAsymTensor1D(size_t d0) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitAsym(8, HALF_AWAY), NULL);
}

/* #370, re-pinned for group-quant PR4 (D6): the zeroPoint slot is still i32
 * LE on the wire but now carries the code-domain uint16 zp; 40000 (> INT16_MAX,
 * < 2^16) is the widest-band class that must round-trip losslessly. The old
 * -72817 value-domain pin has no uint16 representation -- it re-derives as the
 * 65535 ceiling under the nudge (see UnitTestTensorConversion's
 * f2NegBand16 pin). */
static void testDeserializeTensorRoundTripsAsymZeroPoint(void) {
    tensor_t *src = makeAsymTensor1D(4);
    asymQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scales[0] = 0.5f;
    srcQc->zeroPoints[0] = 40000;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *dst = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    fclose(f);

    asymQConfig_t *dstQc = dst->quantization->qConfig;
    float capturedScale = dstQc->scales[0];
    uint16_t capturedZeroPoint = dstQc->zeroPoints[0];
    freeTensor(dst);
    freeTensor(src);

    TEST_ASSERT_EQUAL_FLOAT(0.5f, capturedScale);
    TEST_ASSERT_EQUAL_UINT16(40000, capturedZeroPoint);
}

/* #370/PR4, re-pinned for v5 (group-quant PR4, Task 4): hand-crafted record
 * pins the u16 LE zeroPoints[] wire slot -- a value near the top of the
 * uint16 range (40000, > INT16_MAX) must round-trip losslessly through the
 * now-native u16 field. Pre-v5 this test exercised the i32-slot-to-uint16
 * clamp (clampInt32); v5's wire field IS u16, so there is no clamp left to
 * exercise -- this test now simply pins that a wide-but-valid u16 value
 * survives the read intact. */
static void testDeserializeQConfigAcceptsWideZeroPoint(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 1);                                 /* numGroups */
    writeU32LE(f, 0);                                 /* groupSize */
    uint8_t scaleBytes[4] = {0x00, 0x00, 0x00, 0x3F}; /* scales[0] 0.5f LE */
    fwrite(scaleBytes, 1, 4, f);
    writeU16LE(f, 40000u); /* zeroPoints[0] u16 LE, > INT16_MAX */
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[4] = {0, 0, 0, 0};
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    asymQConfig_t *skelQc = skeleton->quantization->qConfig;
    uint16_t capturedZeroPoint = skelQc->zeroPoints[0];
    freeTensor(skeleton);

    TEST_ASSERT_EQUAL_UINT16(40000, capturedZeroPoint);
}

static tensor_t *makeSymTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitSym(qBits, roundingMode), NULL);
}

/*! Group-quant PR2 (Task 5): grouped-SYM twin of makeSymTensor1D above --
 *  numGroups/groupSize must divide d0 (validateSymQConfigShape, called by
 *  initTensor, enforces it at construction). */
static tensor_t *makeSymGroupedTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode,
                                        size_t numGroups, size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitSymGrouped(qBits, roundingMode, numGroups, groupSize),
                      NULL);
}

/*! Group-quant PR2 (Task 5): PR1's fail-fast on ANY file-vs-skeleton
 *  numGroups mismatch is now LEGAL, provided the file's group shape divides
 *  the skeleton's element count -- deserializeQConfig REALLOCATES the
 *  skeleton's scales[] to the file's numGroups instead of dying
 *  (Deserialize.c's PR1-era comment explicitly named this the PR2 relax).
 *  This is the OLD testDeserializeQConfigRejectsNumGroupsMismatch fixture
 *  with its expectation INVERTED: same hand-crafted-bytes numGroups=2
 *  mismatch (skeleton built PER-TENSOR, numGroups=1), but groupSize is now 2
 *  (not the old fixture's 3) so 2*2 == 4 == the skeleton's element count --
 *  a mismatch this well-formed no longer has any guard left to fail at. See
 *  testDeserializeGroupedSymIntoPerTensorSkeleton below for the same relax
 *  exercised through the real serializeTensor/quantizationInitSymGrouped API
 *  instead of hand-assembled bytes. */
static void testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 2); /* numGroups = 2, mismatches the skeleton's 1 */
    writeU32LE(f, 2); /* groupSize = 2; 2*2 == 4 == the skeleton's element count */
    uint8_t scaleBytes[8] = {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x80, 0x3E}; /* 0.5f, 0.25f */
    fwrite(scaleBytes, 1, 8, f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* qBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    symQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    float capturedScale0 = dstQc->scales[0];
    float capturedScale1 = dstQc->scales[1];
    uint8_t capturedPayload[2];
    memcpy(capturedPayload, skeleton->data, 2);

    freeTensor(skeleton);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(2, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, capturedScale0);
    TEST_ASSERT_EQUAL_FLOAT(0.25f, capturedScale1);
    TEST_ASSERT_EQUAL_UINT8(0xAB, capturedPayload[0]);
    TEST_ASSERT_EQUAL_UINT8(0xCD, capturedPayload[1]);
}

/*! Group-quant PR2 (Task 5): a hand-crafted v4 record whose SYM group shape
 *  passes the numGroups==1<=>groupSize==0 sentinel (numGroups=3, groupSize=2
 *  is a well-formed GROUPED shape) but whose numGroups*groupSize (6) does
 *  not equal the skeleton's element count (4) must still die --
 *  validateSymQConfigShape's divisibility check, not the (now relaxed)
 *  numGroups-mismatch guard above. Golden-bytes style (hand-assembled bytes,
 *  UnitTestSerialize.c's testGoldenBytesModelReluSymOutputV4 pattern) rather
 *  than a real serializeTensor call, since no producer in this tree can be
 *  coaxed into emitting a shape this invalid. */
static void testDeserializeGroupedSymRejectsBadDivisibility(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] = 4 elements */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 2); /* groupSize = 2 -> 3*2 == 6 != 4 elements */
    uint8_t scaleBytes[12] = {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00,
                              0x00, 0x3F, 0x00, 0x00, 0x00, 0x3F}; /* three 0.5f scales */
    fwrite(scaleBytes, 1, 12, f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* never reached: the validate dies first */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! Group-quant PR2 (Task 5): a GROUPED file record (numGroups=3, groupSize=2,
 *  general -- not the per-channel special case) deserialized into a
 *  freshly-built PER-TENSOR skeleton (numGroups=1, groupSize=0) must
 *  reallocate the skeleton's scales[] and load every group's scale, and the
 *  packed tensor DATA must round-trip too -- proving the payload read (sized
 *  by qBits x element count only, independent of numGroups) is unaffected by
 *  the group-shape relax. Built through the real API (quantizationInitSymGrouped
 *  + serializeTensor), unlike the hand-crafted-bytes twin above. */
static void testDeserializeGroupedSymIntoPerTensorSkeleton(void) {
    tensor_t *src = makeSymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6};
    byteConversion((uint8_t *)mantissas, 32, src->data, 8, 6);
    symQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scales[0] = 0.1f;
    srcQc->scales[1] = 0.2f;
    srcQc->scales[2] = 0.3f;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(6, 8, HALF_AWAY); /* per-tensor: {1,0} */

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    symQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    float capturedScales[3];
    for (size_t g = 0; g < 3; g++) {
        capturedScales[g] = dstQc->scales[g];
    }
    size_t dataBytes = calcNumberOfBytesForData(skeleton->quantization, 6);
    uint8_t capturedSrcData[8];
    uint8_t capturedDstData[8];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, skeleton->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(skeleton);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(0.1f, capturedScales[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.2f, capturedScales[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.3f, capturedScales[2]);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! group-quant PR1: numGroups==1 but groupSize!=0 violates the sentinel
 *  invariant documented in Quantization.h (numGroups==1 <=> groupSize==0).
 *  Checked on the FILE values, independent of the numGroups-mismatch
 *  relax above -- a file this corrupt (or written by a future format this
 *  build cannot fully interpret) must fail fast rather than silently
 *  accepting a nonsensical per-tensor "group size". STAYS a death test
 *  post-PR2 (Task 5): the sentinel check is untouched by the reallocation
 *  relax.
 *
 *  Well-formed-record rationale and the groupSize=3 choice: mirrors
 *  testSkipSerializedTensorRejectsRankAboveCap's rationale above -- an
 *  otherwise valid record isolates THIS guard from the unrelated #316
 *  payload-size check a stale parser's misaligned read would otherwise
 *  trip. */
static void testDeserializeQConfigRejectsSentinelViolation(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups = 1, matches the skeleton */
    writeU32LE(f, 3); /* groupSize = 3, violates numGroups==1 <=> groupSize==0 */
    uint8_t scaleBytes[4] = {0x00, 0x00, 0x00, 0x3F};
    fwrite(scaleBytes, 1, 4, f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* qBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeSymTensor1D(4, 4, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! BFP epic PR1 (Task 7): per-tensor BFP tensor builder, mirroring
 *  makeSymTensor1D above. */
static tensor_t *makeBfpTensor1D(size_t d0, uint8_t mantissaBits, uint8_t exponentBits,
                                 roundingMode_t roundingMode) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitBfp(mantissaBits, exponentBits, roundingMode), NULL);
}

/*! BFP epic PR1 (Task 7): grouped-BFP twin of makeBfpTensor1D above --
 *  numGroups/groupSize must divide d0 (validateBfpQConfigShape, called by
 *  initTensor, enforces it at construction), mirroring
 *  makeSymGroupedTensor1D. */
static tensor_t *makeBfpGroupedTensor1D(size_t d0, uint8_t mantissaBits, uint8_t exponentBits,
                                        roundingMode_t roundingMode, size_t numGroups,
                                        size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(
        shape,
        quantizationInitBfpGrouped(mantissaBits, exponentBits, roundingMode, numGroups, groupSize),
        NULL);
}

/*! BFP epic PR1 (Task 7): grouped-BFP round trip, mirroring
 *  testDeserializeGroupedSymIntoPerTensorSkeleton's style but into a
 *  SAME-SHAPE skeleton (numGroups/groupSize match on both sides) -- proves
 *  every qConfig field (exponents, mantissaBits, exponentBits, roundingMode)
 *  and the packed payload round-trip intact. Mantissas are written via
 *  byteConversion directly (raw codes, no quantize/pack-path dependency),
 *  mirroring testDeserializeGroupedSymIntoPerTensorSkeleton's own mantissa
 *  setup. */
static void testBfpRoundTripGrouped(void) {
    tensor_t *src = makeBfpGroupedTensor1D(6, 4, 8, HALF_AWAY, 3, 2);
    bfpQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->exponents[0] = 120;
    srcQc->exponents[1] = 130;
    srcQc->exponents[2] = 140;
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6};
    byteConversion((uint8_t *)mantissas, 32, src->data, 4, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *dst = makeBfpGroupedTensor1D(6, 4, 8, HALF_AWAY, 3, 2);

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    fclose(f);

    bfpQConfig_t *dstQc = dst->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    uint8_t capturedExponents[3];
    for (size_t g = 0; g < 3; g++) {
        capturedExponents[g] = dstQc->exponents[g];
    }
    uint8_t capturedMantissaBits = dstQc->mantissaBits;
    uint8_t capturedExponentBits = dstQc->exponentBits;
    roundingMode_t capturedRoundingMode = dstQc->roundingMode;
    size_t dataBytes = calcNumberOfBytesForData(dst->quantization, 6);
    uint8_t capturedSrcData[3];
    uint8_t capturedDstData[3];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, dst->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(dst);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8(120, capturedExponents[0]);
    TEST_ASSERT_EQUAL_UINT8(130, capturedExponents[1]);
    TEST_ASSERT_EQUAL_UINT8(140, capturedExponents[2]);
    TEST_ASSERT_EQUAL_UINT8(4, capturedMantissaBits);
    TEST_ASSERT_EQUAL_UINT8(8, capturedExponentBits);
    TEST_ASSERT_EQUAL_INT(HALF_AWAY, capturedRoundingMode);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! BFP epic PR1 (Task 7): PR2-style relax sibling of
 *  testDeserializeGroupedSymIntoPerTensorSkeleton -- a GROUPED file record
 *  (numGroups=4, groupSize=2) deserialized into a freshly-built PER-TENSOR
 *  skeleton (numGroups=1, groupSize=0) must REALLOCATE the skeleton's
 *  exponents[] and load every group's exponent, and the packed tensor DATA
 *  must round-trip too. */
static void testBfpDeserializeReallocatesExponentsOnShapeChange(void) {
    tensor_t *src = makeBfpGroupedTensor1D(8, 4, 8, HALF_AWAY, 4, 2);
    bfpQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->exponents[0] = 100;
    srcQc->exponents[1] = 110;
    srcQc->exponents[2] = 120;
    srcQc->exponents[3] = 130;
    int32_t mantissas[] = {1, -2, 3, -4, 5, -6, 7, -8};
    byteConversion((uint8_t *)mantissas, 32, src->data, 4, 8);

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeBfpTensor1D(8, 4, 8, HALF_AWAY); /* per-tensor: {1,0} */

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    bfpQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    uint8_t capturedExponents[4];
    for (size_t g = 0; g < 4; g++) {
        capturedExponents[g] = dstQc->exponents[g];
    }
    size_t dataBytes = calcNumberOfBytesForData(skeleton->quantization, 8);
    uint8_t capturedSrcData[4];
    uint8_t capturedDstData[4];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, skeleton->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(skeleton);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(4, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_UINT8(100, capturedExponents[0]);
    TEST_ASSERT_EQUAL_UINT8(110, capturedExponents[1]);
    TEST_ASSERT_EQUAL_UINT8(120, capturedExponents[2]);
    TEST_ASSERT_EQUAL_UINT8(130, capturedExponents[3]);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! BFP epic PR1 (Task 7): sibling of testDeserializeSymRejectsZeroNumGroupsInWireConfig
 *  -- fileNumGroups == 0 must be rejected by the shared
 *  SERIAL_MAX_QCONFIG_GROUPS guard
 *  DIRECTLY, not merely happen to survive as an always-false branch further
 *  down (validateBfpQConfigShape also rejects numGroups==0, but never runs
 *  at a numberOfElements==0 wire-config call site, so it cannot be the thing
 *  catching this here -- a live-tensor fixture like
 *  testDeserializeQConfigRejectsSentinelViolation's would let
 *  validateBfpQConfigShape backstop it instead, verified empirically while
 *  designing this test: a bare-tensor fixture with fileNumGroups=0 still
 *  dies with the explicit guard removed, because numGroups==0 is never a
 *  valid shape for validateBfpQConfigShape either -- so ONLY the layer-level,
 *  numberOfElements==0 call site isolates this guard's own necessity, per
 *  the SYM sibling's own comment making the identical point). groupSize is
 *  written nonzero (1) so the sentinel check does not fire first either. */
static void testBfpDeserializeRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 0);          /* fileNumGroups = 0 */
    writeU32LE(f, 1);          /* groupSize: nonzero, satisfies the sentinel */
    uint8_t mantissaBits = 4;
    fwrite(&mantissaBits, 1, 1, f);
    uint8_t exponentBits = 8;
    fwrite(&exponentBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! BFP epic PR1 (Task 7): sibling of testDeserializeQConfigRejectsSentinelViolation
 *  -- numGroups==1 but groupSize!=0 violates the sentinel invariant
 *  (Quantization.h), checked on the FILE values. Well-formed-record
 *  rationale mirrors the SYM sibling: an otherwise valid record isolates
 *  THIS guard from the unrelated #316 payload-size check a stale parser's
 *  misaligned read would otherwise trip. */
static void testBfpDeserializeRejectsSentinelViolation(void) {
    uint8_t bfpType = (uint8_t)BFP;
    uint8_t mantissaBits = 4;
    uint8_t exponentBits = 8;
    uint8_t roundingMode = 0;    /* HALF_AWAY */
    uint8_t payload[2] = {0, 0}; /* mantissaBits=4, 4 elems -> ceil(16/8) = 2 packed bytes */

    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    fwrite(&bfpType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups = 1, matches the skeleton */
    writeU32LE(f, 3); /* groupSize = 3, violates numGroups==1 <=> groupSize==0 */
    uint8_t exponentByte = 130;
    fwrite(&exponentByte, 1, 1, f);
    fwrite(&mantissaBits, 1, 1, f);
    fwrite(&exponentBits, 1, 1, f);
    fwrite(&roundingMode, 1, 1, f);
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeBfpTensor1D(4, 4, 8, HALF_AWAY);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! Final-review fix (BFP epic PR1): the BFP arm's mantissaBits/exponentBits
 *  wire bytes were overwritten into the skeleton's qConfig VERBATIM, with no
 *  range check -- unlike every other BFP wire field (numGroups, the sentinel
 *  invariant), which the arm already validates. A corrupt v5 record
 *  announcing a mantissaBits outside initBfpQConfigGrouped's own construction
 *  cap ([2,16], Quantization.c) reaches downstream UB the moment anything
 *  quantizes against it (mantissaBits=1 -> qMax=0, div-by-zero in
 *  packFloatBufferAsBfp) with no guard at the trust boundary where the value
 *  first arrives off the wire.
 *
 *  Isolation: exercised at a layer's outputQ wire-config call site
 *  (numberOfElements == 0, RELU's outputQ, mirroring
 *  testBfpDeserializeRejectsZeroNumGroupsInWireConfig) rather than a live
 *  tensor -- a live-tensor fixture risks the #316 payload-size check firing
 *  first for an unrelated reason (mantissaBits also drives
 *  calcNumberOfBytesForData), which would make this test pass even with the
 *  new guard removed. At a bare wire config there is no payload read to
 *  incidentally catch it: verified empirically pre-fix (TDD RED) -- this
 *  record parsed to completion (deserializeModel returned normally, no
 *  exit(1)) because deserializeLayer simply moves on to propLossQ next,
 *  which matches the skeleton's default FLOAT32 tag. The record is otherwise
 *  complete and well-formed (matching numGroups/groupSize, a real exponent
 *  byte, valid roundingMode, a matching propLossQ tag) for that same reason. */
static void testBfpDeserializeRejectsMantissaBitsOutOfRange(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 1);          /* fileNumGroups = 1, matches the skeleton */
    writeU32LE(f, 0);          /* groupSize = 0, matches the sentinel */
    uint8_t exponentByte = 127;
    fwrite(&exponentByte, 1, 1, f);
    uint8_t mantissaBitsTooNarrow = 1; /* < 2, initBfpQConfigGrouped's own cap */
    fwrite(&mantissaBitsTooNarrow, 1, 1, f);
    uint8_t exponentBits = 8;
    fwrite(&exponentBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! Sibling of testBfpDeserializeRejectsMantissaBitsOutOfRange above, same
 *  isolation rationale (bare wire config, no payload read to incidentally
 *  catch it) -- exponentBits outside initBfpQConfigGrouped's own [2,8] cap.
 *  An unvalidated exponentBits=0 would make bfpExponentBias compute
 *  `1 << -1` (UB) the moment anything derives a scale from this config;
 *  exponentBits > 31 would make `1u << exponentBits` UB in
 *  packFloatBufferAsBfp. Unlike mantissaBits, exponentBits never factors into
 *  calcNumberOfBytesForData at all, so no live-tensor fixture could ever
 *  isolate this guard from the payload-size check by construction -- the
 *  bare wire-config call site is the ONLY place this guard's necessity is
 *  observable, mirroring the SYM SERIAL_MAX_QCONFIG_GROUPS sibling tests' own
 *  reasoning for the same call-site choice. */
static void testBfpDeserializeRejectsExponentBitsOutOfRange(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t bfpType = (uint8_t)BFP;
    fwrite(&bfpType, 1, 1, f); /* outputQ dtype */
    writeU32LE(f, 1);          /* fileNumGroups = 1, matches the skeleton */
    writeU32LE(f, 0);          /* groupSize = 0, matches the sentinel */
    uint8_t exponentByte = 127;
    fwrite(&exponentByte, 1, 1, f);
    uint8_t mantissaBits = 4;
    fwrite(&mantissaBits, 1, 1, f);
    uint8_t exponentBitsTooWide = 9; /* > 8, initBfpQConfigGrouped's own cap */
    fwrite(&exponentBitsTooWide, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *bfpOutputQ = quantizationInitBfp(4, 8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = bfpOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(bfpOutputQ);
    freeQuantization(floatQ);
}

/*! #380 PR3: a FROZEN-serialized parameter (hasGrad=0, no grad tensor in the
 *  file) deserialized into a TRAINABLE skeleton (parameter->grad != NULL)
 *  must load the param and leave the skeleton's grad untouched — it was
 *  zero-initialized at construction (reserveMemory is calloc-backed), and
 *  optimizerZeroGrad re-zeros it before every batch regardless, so
 *  "untouched" and "all-zero" coincide here. Exercises deserializeParameter
 *  directly, mirroring this file's tensor-level style. */
static void testDeserializeWeightsOnlyIntoTrainableSkeleton(void) {
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *srcParamTensor = makeFloatTensor2D(2, 3, data, 6);
    parameter_t srcParameter = {.param = srcParamTensor, .grad = NULL};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeParameter(&srcParameter, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    tensor_t *skeletonGradTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t skeletonParameter = {.param = skeletonParamTensor, .grad = skeletonGradTensor};

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&skeletonParameter, f);
    fclose(f);

    /* CAPTURE every assertion value before any free. */
    float capturedParam[6];
    float capturedGrad[6];
    for (size_t i = 0; i < 6; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
        capturedGrad[i] = ((float *)skeletonGradTensor->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonGradTensor);
    freeTensor(skeletonParamTensor);
    freeTensor(srcParamTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(data, capturedParam, 6);
    for (size_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, capturedGrad[i]);
    }
}

/*! #380 PR3 review: skipSerializedTensor's rank-cap guard (SKIP_TENSOR_MAX_DIMS
 *  == 8) must reject a grad record announcing a rank above that bound BEFORE
 *  reading a single dim value. The grad record is hand-crafted as an
 *  otherwise-well-formed rank-9 FLOAT32/1-element tensor (9 dims of 1, 9
 *  positional order values, FLOAT32 type byte, 4-byte payload) rather than a
 *  truncated stub: a record that is merely too-short-to-be-rank-9 would still
 *  fail downstream via the unrelated short-read guard in serialReadBytes even
 *  with the rank cap disabled, silently passing the death-test assertion
 *  (exit code 1 either way) without ever exercising the rank-cap PRINT_ERROR
 *  itself. Only a well-formed rank-9 record makes the two guards
 *  distinguishable: guarded code exits(1) immediately on the oversized rank;
 *  with the guard removed (verified by temporary mutation, see PR3
 *  final-review report) the record parses to completion and the function
 *  returns normally. Only a frozen skeleton (parameter->grad == NULL) reaches
 *  skipSerializedTensor. */
static void testSkipSerializedTensorRejectsRankAboveCap(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    const uint32_t oversizedRank = 9; /* > SKIP_TENSOR_MAX_DIMS (8) */
    writeU32LE(f, oversizedRank);
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, 1); /* dims[d] = 1 -> 1 element total, well-formed */
    }
    for (uint32_t d = 0; d < oversizedRank; d++) {
        writeU32LE(f, d); /* orderOfDimensions: positional only */
    }
    uint8_t floatType = (uint8_t)FLOAT32;
    fwrite(&floatType, 1, 1, f);       /* FLOAT32 qConfig is empty (no bytes) */
    uint8_t payload[4] = {0, 0, 0, 0}; /* 1 FLOAT32 element */
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! #380 PR3 review: skipSerializedTensor's post-skip truncation guard (the
 *  ftell/SEEK_END comparison after the payload fseek) is the only thing that
 *  catches a payload cut short -- fseek past a genuinely truncated file's real
 *  end succeeds silently (POSIX), so the earlier fseek-past-payload call
 *  cannot fail on its own. A genuine hasGrad=1 parameter record has nothing
 *  after the grad tensor (serializeSparsity is a zero-byte stub), so
 *  shortening the whole file trims the grad tensor's payload tail; mirrors
 *  testDeserializeTensorFailsFastOnTruncatedPayload's file-shortening idiom. */
static void testSkipSerializedTensorRejectsTruncatedPayload(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 3, paramData, 6);
    tensor_t *gradTensor = makeFloatTensor2D(2, 3, paramData, 6);
    parameter_t srcParameter = {.param = paramTensor, .grad = gradTensor};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeParameter(&srcParameter, f);
    long full = ftell(f);
    fclose(f);

    FILE *in = fopen(FILE_PATH, "rb");
    uint8_t *buf = reserveMemory((size_t)full);
    fread(buf, 1, (size_t)full, in);
    fclose(in);
    f = fopen(FILE_PATH, "wb");
    fwrite(buf, 1, (size_t)full - 2, f); /* cuts into the grad tensor's payload tail */
    fclose(f);
    freeReservedMemory(buf);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 3, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);
}

/*! Group-quant PR2 (Task 5) mutation guard: skipSerializedTensor's SYM branch
 *  must parse a GROUPED v4 record (numGroups=3, groupSize=2) exactly -- the
 *  payload byte count is independent of numGroups (sized by qBits x element
 *  count only), but the scratch qConfig's scales[] must grow past its
 *  1-element stack array (symScratchScale) to fileNumGroups and be freed
 *  again afterwards (the mutation this guards: dropping that free leaks the
 *  reallocated array, see the report for the ASan/LSan evidence) -- or the
 *  stream desyncs and the sibling record right after misparses.
 *
 *  Built directly via deserializeParameter/serializeTensor rather than
 *  through a real layer's gradInit: Quantization.h's "Carrier gate" note
 *  fail-fasts a grad TEMPLATE with numGroups > 1 (grouped grads are a future
 *  #300 axis), but skipSerializedTensor itself has no such restriction --
 *  it only consumes wire bytes, never attaches the qConfig to a live
 *  gradient tensor. The "grad" here is a real grouped-SYM tensor_t (v4's SYM
 *  arm has been group-general in Serialize.c since PR1) that happens to be
 *  discarded because the skeleton is frozen (parameter->grad == NULL); the
 *  trailing sibling FLOAT32 tensor's correct round-trip is the stream-sync
 *  proof, mirroring testDeserializeSkipsSymGradIntoFrozenSkeleton
 *  (UnitTestSerialize.c)'s bias-round-trips-after-the-skip idiom. */
static void testSkipSerializedGroupedSymGrad(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    tensor_t *gradTensor = makeSymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t gradMantissas[] = {1, -2, 3, -4, 5, -6};
    byteConversion((uint8_t *)gradMantissas, 32, gradTensor->data, 8, 6);
    symQConfig_t *gradQc = gradTensor->quantization->qConfig;
    gradQc->scales[0] = 0.1f;
    gradQc->scales[1] = 0.2f;
    gradQc->scales[2] = 0.3f;

    float siblingData[] = {9.f, 8.f, 7.f};
    tensor_t *siblingTensor = makeFloatTensor2D(1, 3, siblingData, 3);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    serializeTensor(gradTensor, f);
    serializeTensor(siblingTensor, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};
    tensor_t *skeletonSibling = makeFloatTensor2D(1, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&frozenSkeleton, f);
    deserializeTensor(skeletonSibling, f);
    fclose(f);

    /* CAPTURE before any free. */
    float capturedParam[4];
    for (size_t i = 0; i < 4; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
    }
    float capturedSibling[3];
    for (size_t i = 0; i < 3; i++) {
        capturedSibling[i] = ((float *)skeletonSibling->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonSibling);
    freeTensor(skeletonParamTensor);
    freeTensor(siblingTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(paramData, capturedParam, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(siblingData, capturedSibling, 3);
}

/*! Task-5 review fix (Finding 1, CRITICAL): fileNumGroups is untrusted wire
 *  input read directly into an allocation size (fileNumGroups *
 *  sizeof(float)) with no bound before this fix -- on a 32-bit size_t (MCU
 *  target) a value like 0x40000001 makes that multiplication wrap to 4
 *  bytes (a 1-float buffer) while the scales-read loop still iterates the
 *  file's full unwrapped count, a heap overflow; on a 64-bit host the same
 *  value survives the multiply intact but makes reserveMemory's calloc
 *  return NULL, and the write loop dereferences it unconditionally.
 *  SERIAL_MAX_QCONFIG_GROUPS (Deserialize.c) rejects it before any allocation.
 *
 *  Exercised through a RELU layer's outputQ -- a layer wire-config call site
 *  (deserializeLayer passes numberOfElements=0 there, no live tensor backs
 *  it) -- rather than deserializeTensor, specifically to isolate
 *  SERIAL_MAX_QCONFIG_GROUPS from the sibling elements-bound guard (Finding
 *  1.2): that guard is gated on numberOfElements != 0 and would otherwise
 *  ALSO reject any oversized fileNumGroups against a small skeleton tensor,
 *  making a deserializeTensor-based test pass even with
 *  SERIAL_MAX_QCONFIG_GROUPS itself removed.
 *
 *  The record is otherwise COMPLETE and well-formed (real scale floats,
 *  qBits/rounding, a matching propLossQ tag) all the way through, on
 *  purpose: numberOfElements==0 here means no divisibility validate ever
 *  runs regardless, so an incomplete record would die at the next truncated
 *  read instead -- exit(1) for an unrelated reason that would make the
 *  mutation check below a false positive (verified empirically while
 *  designing this test, see the report). fileNumGroups is
 *  SERIAL_MAX_QCONFIG_GROUPS+1 -- the smallest value that is both over the
 *  cap and cheap to fully materialize on the wire. */
static void testDeserializeSymRejectsOversizedNumGroupsInWireConfig(void) {
    const uint32_t oversizedNumGroups = 65537u; /* SERIAL_MAX_QCONFIG_GROUPS (65536) + 1 */

    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);         /* outputQ dtype */
    writeU32LE(f, oversizedNumGroups); /* fileNumGroups: past the cap */
    writeU32LE(f, 1);                  /* groupSize: nonzero, satisfies the sentinel */
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeF32LE(f, 0.5f);
    }
    uint8_t qBits = 6;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default propLossQ */
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (Task 4): the ASYM twin of the oversized-numGroups test
 *  above -- same isolation rationale (RELU's propLossQ wire-config call
 *  site, numberOfElements=0, SERIAL_MAX_QCONFIG_GROUPS is the only guard
 *  reachable there), same record-completeness discipline (real scale/
 *  zeroPoint values, qBits/rounding, a matching outputQ tag all present, so
 *  removing the cap would let the whole model parse instead of dying for an
 *  unrelated truncated-read reason). */
static void testDeserializeAsymRejectsOversizedNumGroupsInWireConfig(void) {
    const uint32_t oversizedNumGroups = 65537u; /* SERIAL_MAX_QCONFIG_GROUPS (65536) + 1 */

    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0; /* ARITH_FLOAT32, HALF_AWAY -- forwardMath */
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f); /* propLossMath */
    fwrite(&arithByte, 1, 1, f);
    uint8_t outputFloatTag = (uint8_t)FLOAT32; /* matches the skeleton's default outputQ */
    fwrite(&outputFloatTag, 1, 1, f);
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);        /* propLossQ dtype */
    writeU32LE(f, oversizedNumGroups); /* fileNumGroups: past the cap */
    writeU32LE(f, 1);                  /* groupSize: nonzero, satisfies the sentinel */
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeF32LE(f, 0.5f); /* scales[g] */
    }
    for (uint32_t g = 0; g < oversizedNumGroups; g++) {
        writeU16LE(f, 0); /* zeroPoints[g] */
    }
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *asymPropLossQ = quantizationInitAsym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossQ = asymPropLossQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(asymPropLossQ);
    freeQuantization(floatQ);
}

/*! Companion to the above, same isolation rationale: fileNumGroups == 0 is
 *  equally invalid (numGroups == 1 is the per-tensor floor; 0 groups
 *  describes no tensor at all) and must be rejected by
 *  SERIAL_MAX_QCONFIG_GROUPS's guard directly, not merely happen to survive
 *  as an always-false branch further down (validateSymQConfigShape also
 *  rejects numGroups==0, but never runs at a numberOfElements==0 wire-config
 *  call site, so it cannot be the thing catching this here). Record is
 *  likewise complete (0 scale floats follow, since fileNumGroups == 0) so
 *  removing the guard would let the whole model parse successfully instead
 *  of dying for an unrelated reason. */
static void testDeserializeSymRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0;
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 0); /* fileNumGroups = 0 */
    writeU32LE(f, 1); /* groupSize: nonzero, satisfies the sentinel (numGroups
                       * != 1 requires groupSize != 0) */
    uint8_t qBits = 6;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0;
    fwrite(&roundingMode, 1, 1, f);
    uint8_t propLossFloatTag = (uint8_t)FLOAT32;
    fwrite(&propLossFloatTag, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *symOutputQ = quantizationInitSym(6, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.outputQ = symOutputQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(symOutputQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (Task 4): the ASYM twin of the zero-numGroups test above
 *  -- same isolation rationale, exercised through propLossQ instead of
 *  outputQ (mirrors testDeserializeAsymRejectsOversizedNumGroupsInWireConfig
 *  above). Record is complete (0 scale/zeroPoint entries follow, since
 *  fileNumGroups == 0) so removing the guard would let the whole model
 *  parse instead of dying for an unrelated reason. */
static void testDeserializeAsymRejectsZeroNumGroupsInWireConfig(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t arithByte = 0;
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    fwrite(&arithByte, 1, 1, f);
    uint8_t outputFloatTag = (uint8_t)FLOAT32;
    fwrite(&outputFloatTag, 1, 1, f);
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 0); /* fileNumGroups = 0 */
    writeU32LE(f, 1); /* groupSize: nonzero, satisfies the sentinel (numGroups
                       * != 1 requires groupSize != 0) */
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0;
    fwrite(&roundingMode, 1, 1, f);
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    quantization_t *asymPropLossQ = quantizationInitAsym(8, HALF_AWAY);
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    lq.propLossQ = asymPropLossQ;
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(asymPropLossQ);
    freeQuantization(floatQ);
}

/*! group-quant PR4 (Task 4, D6): a wire qBits outside [1, 16] must fail fast
 *  regardless of numGroups/numberOfElements -- the code-domain zp requires
 *  qBits <= 16 to fit u16 (see Quantization.h). Exercised through a real
 *  per-tensor ASYM tensor record (deserializeTensor, numberOfElements != 0)
 *  with an otherwise well-formed v5 record: numGroups=1, groupSize=0, a real
 *  scale/zeroPoint, THEN qBits=17. The check is unconditional (independent
 *  of numberOfElements), so this also pins the wire-config call site would
 *  reject it identically. */
static void testDeserializeAsymRejectsWireQBitsAboveSixteen(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 1); /* numGroups */
    writeU32LE(f, 0); /* groupSize */
    writeF32LE(f, 0.5f);
    writeU16LE(f, 3);
    uint8_t qBits = 17; /* > 16, D6 ceiling */
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[4] = {0, 0, 0, 0}; /* never reached: the guard dies first */
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

static tensor_t *makeAsymGroupedTensor1D(size_t d0, uint8_t qBits, roundingMode_t roundingMode,
                                         size_t numGroups, size_t groupSize) {
    size_t *dims = reserveMemory(1 * sizeof(size_t));
    dims[0] = d0;
    size_t *order = reserveMemory(1 * sizeof(size_t));
    setOrderOfDimsForNewTensor(1, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, 1, order);
    return initTensor(shape, quantizationInitAsymGrouped(qBits, roundingMode, numGroups, groupSize),
                      NULL);
}

/*! group-quant PR4 (Task 4): a hand-crafted v5 ASYM record whose group shape
 *  passes the numGroups==1<=>groupSize==0 sentinel (numGroups=3, groupSize=2
 *  is a well-formed GROUPED shape) but whose numGroups*groupSize (6) does
 *  not equal the skeleton's element count (4) must still die --
 *  validateAsymQConfigShape's divisibility check, mirroring
 *  testDeserializeGroupedSymRejectsBadDivisibility exactly, plus the second
 *  (zeroPoints) array. */
static void testDeserializeGroupedAsymRejectsBadDivisibility(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] = 4 elements */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 2); /* groupSize = 2 -> 3*2 == 6 != 4 elements */
    writeF32LE(f, 0.5f);
    writeF32LE(f, 0.5f);
    writeF32LE(f, 0.5f); /* three scales */
    writeU16LE(f, 1);
    writeU16LE(f, 2);
    writeU16LE(f, 3); /* three zeroPoints */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[2] = {0xAB, 0xCD}; /* never reached: the validate dies first */
    fwrite(payload, 1, 2, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! group-quant PR4 (Task 4): a GROUPED file record (numGroups=3, groupSize=2)
 *  deserialized into a freshly-built PER-TENSOR skeleton (numGroups=1,
 *  groupSize=0) must reallocate BOTH the skeleton's scales[] AND
 *  zeroPoints[] arrays and load every group's scale/zeroPoint, and the
 *  packed tensor DATA must round-trip too -- proving the payload read
 *  (sized by qBits x element count only, independent of numGroups) is
 *  unaffected. Mirrors testDeserializeGroupedSymIntoPerTensorSkeleton
 *  exactly, built through the real API (quantizationInitAsymGrouped +
 *  serializeTensor), not hand-crafted bytes. Run under ASan (mutation ii:
 *  sizing only scales[] on realloc leaves zeroPoints[] stale -- an
 *  ASan-visible heap-buffer-overflow the moment the zeroPoints loop writes
 *  past it). */
static void testDeserializeGroupedAsymIntoPerTensorSkeleton(void) {
    tensor_t *src = makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t codes[] = {10, 20, 30, 40, 50, 60};
    byteConversion((uint8_t *)codes, 32, src->data, 8, 6);
    asymQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scales[0] = 0.1f;
    srcQc->scales[1] = 0.2f;
    srcQc->scales[2] = 0.3f;
    srcQc->zeroPoints[0] = 5;
    srcQc->zeroPoints[1] = 120;
    srcQc->zeroPoints[2] = 250;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(6); /* per-tensor: {1,0} */

    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    asymQConfig_t *dstQc = skeleton->quantization->qConfig;
    /* CAPTURE before any free. */
    size_t capturedNumGroups = dstQc->numGroups;
    size_t capturedGroupSize = dstQc->groupSize;
    float capturedScales[3];
    uint16_t capturedZps[3];
    for (size_t g = 0; g < 3; g++) {
        capturedScales[g] = dstQc->scales[g];
        capturedZps[g] = dstQc->zeroPoints[g];
    }
    size_t dataBytes = calcNumberOfBytesForData(skeleton->quantization, 6);
    uint8_t capturedSrcData[8];
    uint8_t capturedDstData[8];
    memcpy(capturedSrcData, src->data, dataBytes);
    memcpy(capturedDstData, skeleton->data, dataBytes);

    /* FREE in reverse-init order. */
    freeTensor(skeleton);
    freeTensor(src);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_size_t(3, capturedNumGroups);
    TEST_ASSERT_EQUAL_size_t(2, capturedGroupSize);
    TEST_ASSERT_EQUAL_FLOAT(0.1f, capturedScales[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.2f, capturedScales[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.3f, capturedScales[2]);
    TEST_ASSERT_EQUAL_UINT16(5, capturedZps[0]);
    TEST_ASSERT_EQUAL_UINT16(120, capturedZps[1]);
    TEST_ASSERT_EQUAL_UINT16(250, capturedZps[2]);
    TEST_ASSERT_EQUAL_HEX8_ARRAY(capturedSrcData, capturedDstData, dataBytes);
}

/*! group-quant PR4 (Task 4) mutation guard: skipSerializedTensor's ASYM
 *  branch must parse a GROUPED v5 record (numGroups=3, groupSize=2) exactly
 *  -- the payload byte count is independent of numGroups, but the scratch
 *  qConfig's scales[]/zeroPoints[] must both grow past their 1-element
 *  initial arrays to fileNumGroups and be freed again afterwards (mutation
 *  iii: dropping the zeroPoints free leaks the reallocated array -- LSan/
 *  ASan-visible, see the report) -- or the stream desyncs and the sibling
 *  record right after misparses. Mirrors testSkipSerializedGroupedSymGrad
 *  exactly. */
static void testSkipSerializedGroupedAsymGrad(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    tensor_t *gradTensor = makeAsymGroupedTensor1D(6, 8, HALF_AWAY, 3, 2);
    int32_t gradCodes[] = {10, 20, 30, 40, 50, 60};
    byteConversion((uint8_t *)gradCodes, 32, gradTensor->data, 8, 6);
    asymQConfig_t *gradQc = gradTensor->quantization->qConfig;
    gradQc->scales[0] = 0.1f;
    gradQc->scales[1] = 0.2f;
    gradQc->scales[2] = 0.3f;
    gradQc->zeroPoints[0] = 5;
    gradQc->zeroPoints[1] = 120;
    gradQc->zeroPoints[2] = 250;

    float siblingData[] = {9.f, 8.f, 7.f};
    tensor_t *siblingTensor = makeFloatTensor2D(1, 3, siblingData, 3);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);
    serializeTensor(gradTensor, f);
    serializeTensor(siblingTensor, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};
    tensor_t *skeletonSibling = makeFloatTensor2D(1, 3, NULL, 0);

    f = fopen(FILE_PATH, "rb");
    deserializeParameter(&frozenSkeleton, f);
    deserializeTensor(skeletonSibling, f);
    fclose(f);

    /* CAPTURE before any free. */
    float capturedParam[4];
    for (size_t i = 0; i < 4; i++) {
        capturedParam[i] = ((float *)skeletonParamTensor->data)[i];
    }
    float capturedSibling[3];
    for (size_t i = 0; i < 3; i++) {
        capturedSibling[i] = ((float *)skeletonSibling->data)[i];
    }

    /* FREE in reverse-init order. */
    freeTensor(skeletonSibling);
    freeTensor(skeletonParamTensor);
    freeTensor(siblingTensor);
    freeTensor(gradTensor);
    freeTensor(paramTensor);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(paramData, capturedParam, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(siblingData, capturedSibling, 3);
}

/*! group-quant PR4 (Task 4): the ASYM twin of
 *  testSkipSerializedGroupedSymGradRejectsBadDivisibility -- a GROUPED grad
 *  record on the SKIP path whose numGroups*groupSize (9) does not equal its
 *  own element count (6) must still fail fast: the ASYM arm's
 *  validateAsymQConfigShape call is reached via the real element count
 *  skipSerializedTensor threads through, exactly like the SYM arm's. */
static void testSkipSerializedGroupedAsymGradRejectsBadDivisibility(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    writeU32LE(f, 1); /* grad numberOfDimensions */
    writeU32LE(f, 6); /* grad dimensions[0] = 6 elements */
    writeU32LE(f, 0); /* grad orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 3); /* groupSize = 3 -> 3*3 == 9 != 6 elements */
    writeF32LE(f, 0.5f);
    writeF32LE(f, 0.5f);
    writeF32LE(f, 0.5f); /* three scales */
    writeU16LE(f, 1);
    writeU16LE(f, 2);
    writeU16LE(f, 3); /* three zeroPoints */
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[3] = {0xAB, 0xCD, 0xEF}; /* never reached: the validate dies first */
    fwrite(payload, 1, 3, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! Task-5 review fix (Finding 1, point 3): skipSerializedTensor now threads
 *  the real element count it just parsed from the record's own dims into
 *  deserializeQConfig instead of a hardcoded 0 -- proven here by a GROUPED
 *  grad record on the SKIP path (frozen skeleton, deserializeParameter)
 *  whose numGroups*groupSize (9) does not equal its own element count (6):
 *  before this fix numberOfElements==0 suppressed validateSymQConfigShape
 *  unconditionally on the skip path, so this record would have silently
 *  "succeeded" past skip parsing despite being corrupt. Hand-crafted bytes,
 *  mirroring testDeserializeGroupedSymRejectsBadDivisibility -- no producer
 *  in this tree emits an invalid-divisibility record (initSymQConfigGrouped
 *  enforces the shape at construction). */
static void testSkipSerializedGroupedSymGradRejectsBadDivisibility(void) {
    float paramData[] = {1.f, 2.f, 3.f, 4.f};
    tensor_t *paramTensor = makeFloatTensor2D(2, 2, paramData, 4);

    FILE *f = fopen(FILE_PATH, "wb");
    uint8_t hasGrad = 1;
    fwrite(&hasGrad, sizeof(uint8_t), 1, f);
    serializeTensor(paramTensor, f);

    writeU32LE(f, 1); /* grad numberOfDimensions */
    writeU32LE(f, 6); /* grad dimensions[0] = 6 elements */
    writeU32LE(f, 0); /* grad orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 3); /* numGroups = 3 */
    writeU32LE(f, 3); /* groupSize = 3 -> 3*3 == 9 != 6 elements */
    uint8_t scaleBytes[12] = {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00,
                              0x00, 0x3F, 0x00, 0x00, 0x00, 0x3F}; /* three 0.5f scales */
    fwrite(scaleBytes, 1, 12, f);
    uint8_t qBits = 4;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    uint8_t payload[3] = {0xAB, 0xCD, 0xEF}; /* never reached: the validate dies first */
    fwrite(payload, 1, 3, f);
    fclose(f);

    tensor_t *skeletonParamTensor = makeFloatTensor2D(2, 2, NULL, 0);
    parameter_t frozenSkeleton = {.param = skeletonParamTensor, .grad = NULL};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeParameter(&frozenSkeleton, f));
    fclose(f);

    freeTensor(skeletonParamTensor);
    freeTensor(paramTensor);
}

/*! BFP epic PR2 (Task 1): arithmetic wire-tag round trip -- a layer's
 *  forwardMath hand-set to {ARITH_BFP, SR_HALF_AWAY} on the SERIAL side only
 *  (mirroring testRoundTripLinear's grad-seed rationale in
 *  UnitTestSerialize.c: a round trip that actually moves bytes is
 *  distinguishable from one that leaves the deserial mirror's uniform
 *  FLOAT32/HALF_AWAY default untouched) survives serializeModel /
 *  deserializeModel intact. RELU is the minimal layer with a forwardMath
 *  field. The derivation flip (arithmeticFromQuantization producing
 *  ARITH_BFP) is Task 9 -- this test only proves the wire tag itself
 *  round-trips once a caller hand-sets it. */
static void testDeserializeArithmeticRoundTripsBfp(void) {
    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);

    layer_t *serialLayer = reluLayerInit(&lq);
    layer_t *deserialLayer = reluLayerInit(&lq);

    serialLayer->config->relu->forwardMath =
        (arithmetic_t){.type = ARITH_BFP, .roundingMode = SR_HALF_AWAY};

    layer_t *serialModel[] = {serialLayer};
    layer_t *deserialModel[] = {deserialLayer};

    FILE *f = fopen(FILE_PATH, "wb");
    serializeModel(serialModel, 1, f);
    fclose(f);

    f = fopen(FILE_PATH, "rb");
    deserializeModel(deserialModel, 1, f);
    fclose(f);

    /* CAPTURE before any free. */
    arithmetic_t loadedForwardMath = deserialLayer->config->relu->forwardMath;

    freeReluLayer(deserialLayer);
    freeReluLayer(serialLayer);
    freeQuantization(floatQ);

    /* ASSERT on captured. */
    TEST_ASSERT_EQUAL_INT(ARITH_BFP, loadedForwardMath.type);
    TEST_ASSERT_EQUAL_INT(SR_HALF_AWAY, loadedForwardMath.roundingMode);
}

/*! BFP epic PR2 (Task 1): corruption guard sibling -- an arithmetic-type wire
 *  tag one past ARITH_BFP (3) has no matching enum member yet. Hand-crafted
 *  ODTS v5 bytes, mirroring the neighboring BFP width-rejection tests'
 *  well-formed-except-one-byte idiom (testBfpDeserializeRejectsMantissaBitsOutOfRange
 *  et al. above): every other byte in the record is what a real serialize of
 *  the RELU skeleton below would produce, EXCEPT the forwardMath type byte,
 *  which is patched to the out-of-range value 3. deserializeArithmetic reads
 *  and range-checks the type byte before the roundingMode byte, so this dies
 *  before the rest of the record is even read -- an unvalidated read here
 *  would instead alias to whatever future 4th arithmeticType_t member gets
 *  appended, silently misinterpreting the compute representation. */
static void testDeserializeArithmeticRejectsUnknownTypeTag(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t badArithType = 3;       /* one past ARITH_BFP (2) -- unknown wire tag */
    fwrite(&badArithType, 1, 1, f); /* forwardMath.type */
    uint8_t halfAway = (uint8_t)HALF_AWAY;
    fwrite(&halfAway, 1, 1, f); /* forwardMath.roundingMode */
    uint8_t floatType = (uint8_t)ARITH_FLOAT32;
    fwrite(&floatType, 1, 1, f); /* propLossMath.type */
    fwrite(&halfAway, 1, 1, f);  /* propLossMath.roundingMode */
    uint8_t floatQType = (uint8_t)FLOAT32;
    fwrite(&floatQType, 1, 1, f); /* outputQ dtype */
    fwrite(&floatQType, 1, 1, f); /* propLossQ dtype */
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(floatQ);
}

/*! BFP epic PR2 (Task 1): roundingMode-tag sibling of
 *  testDeserializeArithmeticRejectsUnknownTypeTag above -- an unvalidated
 *  roundingMode byte one past SR_HALF_AWAY (2) would alias to whatever future
 *  3rd roundingMode_t member gets appended, silently misinterpreting the
 *  op's rounding policy. The type tag is left VALID (ARITH_FLOAT32) so this
 *  isolates the second guard from the first -- if only the type-tag guard
 *  existed, this record would parse the type byte fine and only misbehave
 *  on the untested roundingMode byte. */
static void testDeserializeArithmeticRejectsUnknownRoundingModeTag(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 5); /* version */
    writeU32LE(f, 1); /* layerCount */
    uint8_t tag = (uint8_t)RELU;
    fwrite(&tag, 1, 1, f);
    uint8_t floatType = (uint8_t)ARITH_FLOAT32;
    fwrite(&floatType, 1, 1, f);       /* forwardMath.type -- valid */
    uint8_t badRoundingMode = 2;       /* one past SR_HALF_AWAY (1) -- unknown wire tag */
    fwrite(&badRoundingMode, 1, 1, f); /* forwardMath.roundingMode */
    fwrite(&floatType, 1, 1, f);       /* propLossMath.type */
    uint8_t halfAway = (uint8_t)HALF_AWAY;
    fwrite(&halfAway, 1, 1, f); /* propLossMath.roundingMode */
    uint8_t floatQType = (uint8_t)FLOAT32;
    fwrite(&floatQType, 1, 1, f); /* outputQ dtype */
    fwrite(&floatQType, 1, 1, f); /* propLossQ dtype */
    fclose(f);

    quantization_t *floatQ = quantizationInitFloat();
    layerQuant_t lq;
    layerQuantInitUniform(&lq, floatQ);
    layer_t *layer = reluLayerInit(&lq);
    layer_t *model[] = {layer};

    f = fopen(FILE_PATH, "rb");
    ASSERT_EXITS_WITH_FAILURE(deserializeModel(model, 1, f));
    fclose(f);

    freeReluLayer(layer);
    freeQuantization(floatQ);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testDeserializeRejectsBadMagic);
    RUN_TEST(testDeserializeRejectsWrongVersion);
    RUN_TEST(testDeserializeRejectsV3Version);
    RUN_TEST(testDeserializeRejectsV4Version);
    RUN_TEST(testDeserializeRejectsLayerCountMismatch);
    RUN_TEST(testDeserializeRejectsTagMismatch);
    RUN_TEST(testDeserializeTensorRejectsDtypeMismatch);
    RUN_TEST(testDeserializeTensorRejectsPayloadSizeMismatch);
    RUN_TEST(testDeserializeTensorRejectsSymRecordIntoFloatSkeleton);
    RUN_TEST(testDeserializeModelFailsFastOnTruncatedHeader);
    RUN_TEST(testDeserializeTensorFailsFastOnTruncatedPayload);
    RUN_TEST(testDeserializeTensorRejectsRankMismatch);
    RUN_TEST(testDeserializeTensorRoundTripsAsymZeroPoint);
    RUN_TEST(testDeserializeQConfigAcceptsWideZeroPoint);
    RUN_TEST(testDeserializeQConfigAcceptsNumGroupsMismatchViaReallocation);
    RUN_TEST(testDeserializeGroupedSymRejectsBadDivisibility);
    RUN_TEST(testDeserializeGroupedSymIntoPerTensorSkeleton);
    RUN_TEST(testDeserializeQConfigRejectsSentinelViolation);
    RUN_TEST(testBfpRoundTripGrouped);
    RUN_TEST(testBfpDeserializeReallocatesExponentsOnShapeChange);
    RUN_TEST(testBfpDeserializeRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testBfpDeserializeRejectsSentinelViolation);
    RUN_TEST(testBfpDeserializeRejectsMantissaBitsOutOfRange);
    RUN_TEST(testBfpDeserializeRejectsExponentBitsOutOfRange);
    RUN_TEST(testDeserializeWeightsOnlyIntoTrainableSkeleton);
    RUN_TEST(testSkipSerializedTensorRejectsRankAboveCap);
    RUN_TEST(testSkipSerializedTensorRejectsTruncatedPayload);
    RUN_TEST(testSkipSerializedGroupedSymGrad);
    RUN_TEST(testDeserializeSymRejectsOversizedNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsOversizedNumGroupsInWireConfig);
    RUN_TEST(testDeserializeSymRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsZeroNumGroupsInWireConfig);
    RUN_TEST(testDeserializeAsymRejectsWireQBitsAboveSixteen);
    RUN_TEST(testDeserializeGroupedAsymRejectsBadDivisibility);
    RUN_TEST(testDeserializeGroupedAsymIntoPerTensorSkeleton);
    RUN_TEST(testSkipSerializedGroupedAsymGrad);
    RUN_TEST(testSkipSerializedGroupedAsymGradRejectsBadDivisibility);
    RUN_TEST(testSkipSerializedGroupedSymGradRejectsBadDivisibility);
    RUN_TEST(testDeserializeArithmeticRoundTripsBfp);
    RUN_TEST(testDeserializeArithmeticRejectsUnknownTypeTag);
    RUN_TEST(testDeserializeArithmeticRejectsUnknownRoundingModeTag);
    return UNITY_END();
}
