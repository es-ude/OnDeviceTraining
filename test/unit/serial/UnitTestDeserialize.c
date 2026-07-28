#include <stdint.h>
#include <stdio.h>

#include "DeathTest.h"
#include "Deserialize.h"
#include "Flatten.h"
#include "FlattenApi.h"
#include "Layer.h"
#include "QuantizationApi.h"
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
    writeU32LE(f, 4); /* version (dead value: bad magic short-circuits first) */
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
 *  version check now that SERIALIZE_FORMAT_VERSION is 4, exactly like any
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

static void testDeserializeRejectsLayerCountMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    fwrite("ODTS", 1, 4, f);
    writeU32LE(f, 4); /* version */
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
    writeU32LE(f, 4);              /* version */
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

/* #370/#246: ASYM zeroPoint travels as i32 LE and must round-trip losslessly;
 * -72817 is the qBits=16 worst case that used to wrap in the int16 field. */
static void testDeserializeTensorRoundTripsAsymZeroPoint(void) {
    tensor_t *src = makeAsymTensor1D(4);
    asymQConfig_t *srcQc = src->quantization->qConfig;
    srcQc->scale = 0.5f;
    srcQc->zeroPoint = -72817;

    FILE *f = fopen(FILE_PATH, "wb");
    serializeTensor(src, f);
    fclose(f);

    tensor_t *dst = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(dst, f);
    fclose(f);

    asymQConfig_t *dstQc = dst->quantization->qConfig;
    float capturedScale = dstQc->scale;
    int32_t capturedZeroPoint = dstQc->zeroPoint;
    freeTensor(dst);
    freeTensor(src);

    TEST_ASSERT_EQUAL_FLOAT(0.5f, capturedScale);
    TEST_ASSERT_EQUAL_INT32(-72817, capturedZeroPoint);
}

/* #370/#246: hand-crafted record pins the i32 LE zeroPoint wire slot — a value
 * outside the old int16 range must arrive intact in the widened field. */
static void testDeserializeQConfigAcceptsWideZeroPoint(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t asymType = (uint8_t)ASYM;
    fwrite(&asymType, 1, 1, f);
    uint8_t scaleBytes[4] = {0x00, 0x00, 0x00, 0x3F}; /* 0.5f LE */
    fwrite(scaleBytes, 1, 4, f);
    uint8_t qBits = 8;
    fwrite(&qBits, 1, 1, f);
    uint8_t roundingMode = 0; /* HALF_AWAY */
    fwrite(&roundingMode, 1, 1, f);
    writeU32LE(f, 40000u); /* zeroPoint i32 LE, > INT16_MAX */
    uint8_t payload[4] = {0, 0, 0, 0};
    fwrite(payload, 1, 4, f);
    fclose(f);

    tensor_t *skeleton = makeAsymTensor1D(4);
    f = fopen(FILE_PATH, "rb");
    deserializeTensor(skeleton, f);
    fclose(f);

    asymQConfig_t *skelQc = skeleton->quantization->qConfig;
    int32_t capturedZeroPoint = skelQc->zeroPoint;
    freeTensor(skeleton);

    TEST_ASSERT_EQUAL_INT32(40000, capturedZeroPoint);
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

/*! group-quant PR1: a v4 SYM qConfig record whose file numGroups (2) does not
 *  match the skeleton's numGroups (always 1 in PR1 -- initSymQConfig's only
 *  output, see quantizationInitSym) must fail fast rather than silently
 *  misparsing into the fixed-size 1-element scales[] array. Group-aware
 *  deserialize (reallocating the skeleton's scales[] to match the file) is a
 *  PR2 concern; PR1 keeps the #316 no-silent-misparse discipline instead.
 *
 *  The record is otherwise WELL-FORMED (real qBits, a correctly-sized
 *  trailing payload) rather than truncated/garbage -- mirrors
 *  testSkipSerializedTensorRejectsRankAboveCap's rationale above: a record
 *  that is merely malformed would still trip some OTHER, unrelated guard
 *  (e.g. the #316 payload-size check misreading a stale parser's
 *  garbage-aligned qBits), silently passing this assertion without ever
 *  exercising the numGroups-mismatch guard itself. groupSize=3 here is
 *  deliberately chosen so a pre-v4 (scale/qBits/rounding-only) parser's
 *  misaligned "qBits" byte (groupSize's low byte) still computes the SAME
 *  packed byte count as the real qBits=4 at 4 elements (ceil(4*3/8) ==
 *  ceil(4*4/8) == 2) -- i.e. this fixture is the drift alarm for THIS guard
 *  specifically, not a stand-in for any parser that merely notices
 *  something is wrong. */
static void testDeserializeQConfigRejectsNumGroupsMismatch(void) {
    FILE *f = fopen(FILE_PATH, "wb");
    writeU32LE(f, 1); /* numberOfDimensions */
    writeU32LE(f, 4); /* dimensions[0] */
    writeU32LE(f, 0); /* orderOfDimensions[0] */
    uint8_t symType = (uint8_t)SYM;
    fwrite(&symType, 1, 1, f);
    writeU32LE(f, 2); /* numGroups = 2, mismatches the skeleton's 1 */
    writeU32LE(f, 3); /* groupSize (consistent w/ numGroups=2; see comment re: 3 vs 4) */
    uint8_t scaleBytes[8] = {0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x3F}; /* two 0.5f scales */
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
    ASSERT_EXITS_WITH_FAILURE(deserializeTensor(skeleton, f));
    fclose(f);

    freeTensor(skeleton);
}

/*! group-quant PR1: numGroups==1 but groupSize!=0 violates the sentinel
 *  invariant documented in Quantization.h (numGroups==1 <=> groupSize==0).
 *  Checked on the FILE values, independent of the numGroups-matches-skeleton
 *  check above -- a file this corrupt (or written by a future format this
 *  build cannot fully interpret) must fail fast rather than silently
 *  accepting a nonsensical per-tensor "group size".
 *
 *  Well-formed-record rationale and the groupSize=3 choice: same as
 *  testDeserializeQConfigRejectsNumGroupsMismatch above -- an otherwise
 *  valid record isolates THIS guard from the unrelated #316 payload-size
 *  check a stale parser's misaligned read would otherwise trip. */
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

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testSerializeAndDeserializeTensor);
    RUN_TEST(testDeserializeRejectsBadMagic);
    RUN_TEST(testDeserializeRejectsWrongVersion);
    RUN_TEST(testDeserializeRejectsV3Version);
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
    RUN_TEST(testDeserializeQConfigRejectsNumGroupsMismatch);
    RUN_TEST(testDeserializeQConfigRejectsSentinelViolation);
    RUN_TEST(testDeserializeWeightsOnlyIntoTrainableSkeleton);
    RUN_TEST(testSkipSerializedTensorRejectsRankAboveCap);
    RUN_TEST(testSkipSerializedTensorRejectsTruncatedPayload);
    return UNITY_END();
}
