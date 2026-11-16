#define SOURCE_FILE "DESERIALIZE"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "AddLayer.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Deserialize.h"
#include "DeserializeInternal.h"
#include "Dropout.h"
#include "DyadicScale.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "Model.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "SerialWire.h"
#include "Softmax.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorRecord.h"

/* Reads the ODTS layout documented in Serialize.c into a factory-built
 * skeleton. Only SERIALIZE_FORMAT_VERSION (TensorRecord.h) is accepted. Every
 * file field that sizes an access or must agree with the skeleton is checked
 * as it arrives, so a rejected file can leave the skeleton partially
 * overwritten. A scale is accepted only in canonical form (deserializeScale);
 * a kind-1 wiring must equal the skeleton's CSR, compared as it streams in. */
#define SERIALIZE_MAGIC "ODTS"

void deserializeTensor(tensor_t *tensor, FILE *f) {
    /* Captured before the shape/quantization overwrite: tensor->data was sized
     * from it, and a record with more elements or a wider packing (which the
     * dtype check cannot see) would otherwise read past that allocation. */
    size_t expectedBytes =
        calcNumberOfBytesForData(tensor->quantization, calcNumberOfElementsByShape(tensor->shape));

    deserializeShape(tensor->shape, f);
    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    deserializeQuantization(tensor->quantization, f, numberOfValues);

    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    if (dataBytes != expectedBytes) {
        PRINT_ERROR("deserializeTensor: file payload %zu bytes does not match the skeleton's "
                    "allocated %zu bytes (shape/qBits mismatch)",
                    dataBytes, expectedBytes);
        exit(1);
    }

    serialReadBytes(tensor->data, dataBytes, f);
}

/* Gradients are per-tensor (gradInit's rule), but a SYM/ASYM record may regroup the skeleton's
 * grid, and the optimizer folds a gradient scale into group 0 only. */
static void requirePerTensorGrad(const tensor_t *grad) {
    const quantization_t *q = grad->quantization;
    size_t numGroups = 1;
    if (q->type == SYM) {
        numGroups = ((const symQConfig_t *)q->qConfig)->numGroups;
    } else if (q->type == ASYM) {
        numGroups = ((const asymQConfig_t *)q->qConfig)->numGroups;
    }
    if (numGroups > 1) {
        PRINT_ERROR("deserializeParameter: grouped %s gradient record (numGroups=%zu); "
                    "gradients must be per-tensor",
                    q->type == SYM ? "SYM" : "ASYM", numGroups);
        exit(1);
    }
}

void deserializeParameter(parameter_t *parameter, FILE *f) {
    uint8_t hasGrad = serialReadU8(f);
    if (hasGrad > 1u) {
        PRINT_ERROR("deserializeParameter: hasGrad byte %u is neither 0 nor 1", (unsigned)hasGrad);
        exit(1);
    }
    deserializeTensor(parameter->param, f);
    if (hasGrad && parameter->grad != NULL) {
        deserializeTensor(parameter->grad, f);
        requirePerTensorGrad(parameter->grad);
    } else if (hasGrad) {
        /* Frozen skeleton: it holds no grad, and grads are transient anyway.
         * Parse past the record so the stream stays in sync. */
        skipSerializedTensor(f);
    }
    /* hasGrad == 0 into a trainable skeleton leaves its grad unchanged. */
}

static void requireWiringEntry(const char *field, size_t index, uint16_t file, uint16_t skeleton) {
    if (file != skeleton) {
        PRINT_ERROR("deserializeModel: wiring mismatch on '%s[%zu]' (file %u, skeleton %u)", field,
                    index, (unsigned)file, (unsigned)skeleton);
        exit(1);
    }
}

void deserializeModel(const model_t *m, FILE *f) {
    modelRequireValid(m, "deserializeModel");
    char magic[4];
    serialReadBytes(magic, 4, f);
    if (memcmp(magic, SERIALIZE_MAGIC, 4) != 0) {
        PRINT_ERROR("deserializeModel: bad magic bytes (expected \"ODTS\")");
        exit(1);
    }

    uint32_t version = serialReadU32LE(f);
    if (version != SERIALIZE_FORMAT_VERSION) {
        PRINT_ERROR("deserializeModel: unsupported format version %u (expected %u)",
                    (unsigned)version, (unsigned)SERIALIZE_FORMAT_VERSION);
        exit(1);
    }

    uint32_t layerCount = serialReadU32LE(f);
    if (layerCount != (uint32_t)m->numLayers) {
        PRINT_ERROR("deserializeModel: layerCount mismatch (file has %u, caller expects %zu)",
                    (unsigned)layerCount, m->numLayers);
        exit(1);
    }

    /* Comparing with the skeleton's kind also rejects every unknown kind. */
    uint8_t wiringKind = serialReadU8(f);
    uint8_t skeletonKind = modelIsChain(m) ? 0u : 1u;
    if (wiringKind != skeletonKind) {
        PRINT_ERROR("deserializeModel: wiring mismatch on 'kind' (file %u, skeleton %u)",
                    (unsigned)wiringKind, (unsigned)skeletonKind);
        exit(1);
    }
    if (skeletonKind == 1u) {
        /* The skeleton passed modelRequireValid, so equality validates the file's CSR; the bounds
         * come from the skeleton, never from the file. */
        for (size_t j = 0; j <= m->numLayers; j++) {
            requireWiringEntry("inputStart", j, serialReadU16LE(f), m->inputStart[j]);
        }
        for (size_t e = 0; e < m->inputStart[m->numLayers]; e++) {
            requireWiringEntry("inputIds", e, serialReadU16LE(f), m->inputIds[e]);
        }
    }

    for (size_t i = 0; i < m->numLayers; i++) {
        uint8_t tag = serialReadU8(f);
        if (tag != (uint8_t)m->layers[i]->type) {
            PRINT_ERROR("deserializeModel: record tag %u does not match expected layer type "
                        "%u at index %zu",
                        (unsigned)tag, (unsigned)m->layers[i]->type, i);
            exit(1);
        }
        deserializeLayer(m->layers[i], f);
    }
}

// Helper Functions

/* orderOfDimensions indexes dimensions[], so it must be a permutation of
 * 0..rank-1: an entry >= rank reads past the array, and a repeat leaves a
 * dimension unreachable. Quadratic scan: the rank is capped. */
static void deserializeAxisOrder(size_t *order, size_t rank, FILE *f, const char *what) {
    for (size_t d = 0; d < rank; d++) {
        uint32_t axis = serialReadU32LE(f);
        if ((size_t)axis >= rank) {
            PRINT_ERROR("%s: orderOfDimensions[%zu] = %u is out of range for rank %zu", what, d,
                        (unsigned)axis, rank);
            exit(1);
        }
        for (size_t prev = 0; prev < d; prev++) {
            if (order[prev] == (size_t)axis) {
                PRINT_ERROR("%s: orderOfDimensions[%zu] = %u repeats an earlier entry (not a "
                            "permutation of 0..%zu)",
                            what, d, (unsigned)axis, rank - 1);
                exit(1);
            }
        }
        order[d] = (size_t)axis;
    }
}

static void deserializeShape(shape_t *shape, FILE *f) {
    uint32_t fileRank = serialReadU32LE(f);
    /* The skeleton's dimension arrays were sized by its rank; a mismatch would
     * write past them, and an equal-element-count rank change would pass the
     * payload-size check. */
    if ((size_t)fileRank != shape->numberOfDimensions) {
        PRINT_ERROR("deserializeShape: file rank %u does not match the skeleton rank %zu",
                    (unsigned)fileRank, shape->numberOfDimensions);
        exit(1);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        shape->dimensions[d] = (size_t)serialReadU32LE(f);
    }
    deserializeAxisOrder(shape->orderOfDimensions, shape->numberOfDimensions, f,
                         "deserializeShape");
}

static void deserializeQuantization(quantization_t *q, FILE *f, size_t numberOfElements) {
    uint8_t type = serialReadU8(f);
    /* The skeleton's qConfig struct (NULL for FLOAT32/INT32/BOOL) is fixed by
     * its dtype; another file dtype would write through a mismatched or NULL
     * qConfig. */
    if ((qtype_t)type != q->type) {
        PRINT_ERROR("deserializeQuantization: file dtype %u does not match the skeleton dtype %u",
                    (unsigned)type, (unsigned)q->type);
        exit(1);
    }
    deserializeQConfig(q, f, numberOfElements);
}

static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    uint8_t typeTag = serialReadU8(f);
    if (typeTag > (uint8_t)ARITH_BFP) {
        PRINT_ERROR("deserializeArithmetic: unknown arithmetic wire tag %u (max %u)", typeTag,
                    (unsigned)ARITH_BFP);
        exit(1);
    }
    arithmetic->type = (arithmeticType_t)typeTag;

    uint8_t roundingTag = serialReadU8(f);
    if (roundingTag > (uint8_t)SR_HALF_AWAY) {
        PRINT_ERROR("deserializeArithmetic: unknown roundingMode wire tag %u", roundingTag);
        exit(1);
    }
    arithmetic->roundingMode = (roundingMode_t)roundingTag;
}

static void deserializeKernel(kernel_t *kernel, FILE *f) {
    kernel->size = (size_t)serialReadU32LE(f);
    uint8_t paddingTag = serialReadU8(f);
    if (paddingTag > (uint8_t)EXPLICIT) {
        PRINT_ERROR("deserializeKernel: paddingType %u is out of range (max %u)", paddingTag,
                    (unsigned)EXPLICIT);
        exit(1);
    }
    kernel->paddingType = (paddingType_t)paddingTag;
    kernel->stride = (size_t)serialReadU32LE(f);
    kernel->dilation = (size_t)serialReadU32LE(f);
    kernel->padding = (size_t)serialReadU32LE(f);
}

/* Bound untrusted group counts before allocation; the cap limits each scales
 * array to 256 KiB and prevents allocation-size overflow on 32-bit targets. */
#define SERIAL_MAX_QCONFIG_GROUPS 65536u

typedef struct {
    size_t numGroups;
    size_t groupSize;
} groupShape_t;

/* Reads and validates the group shape that precedes every grouped qConfig's
 * arrays, before the caller sizes any allocation from it. numberOfElements == 0
 * means no tensor geometry is available (layer wire configs): only the cap and
 * the per-tensor sentinel apply there. */
static groupShape_t deserializeGroupShape(FILE *f, size_t numberOfElements, const char *dtype) {
    groupShape_t shape;
    shape.numGroups = (size_t)serialReadU32LE(f);
    if (shape.numGroups == 0 || shape.numGroups > SERIAL_MAX_QCONFIG_GROUPS) {
        PRINT_ERROR("deserializeQConfig: %s file numGroups %zu is zero or exceeds the %u-group "
                    "sanity cap",
                    dtype, shape.numGroups, (unsigned)SERIAL_MAX_QCONFIG_GROUPS);
        exit(1);
    }
    if (numberOfElements != 0 && shape.numGroups > numberOfElements) {
        PRINT_ERROR("deserializeQConfig: %s file numGroups %zu exceeds the %zu-element tensor it "
                    "attaches to",
                    dtype, shape.numGroups, numberOfElements);
        exit(1);
    }
    shape.groupSize = (size_t)serialReadU32LE(f);
    if ((shape.numGroups == 1) != (shape.groupSize == 0)) {
        PRINT_ERROR("deserializeQConfig: %s file violates the numGroups==1<=>groupSize==0 "
                    "sentinel invariant (numGroups=%zu, groupSize=%zu)",
                    dtype, shape.numGroups, shape.groupSize);
        exit(1);
    }
    if (numberOfElements != 0) {
        validateGroupShape(shape.numGroups, shape.groupSize, numberOfElements,
                           "deserializeQConfig");
    }
    return shape;
}

/* v9 scale field: `u16 m`, `i16 e`, read into locals (never through a cast
 * of the stream buffer). Only the canonical form is accepted: a non-canonical
 * or out-of-range field is a corrupt record, not something to snap. */
static odtScale_t deserializeScale(FILE *f) {
    uint16_t m = serialReadU16LE(f);
    int16_t e = serialReadI16LE(f);
    if (m < ODT_SCALE_M_MIN || m > ODT_SCALE_M_MAX) {
        PRINT_ERROR("deserializeQConfig: scale mantissa %u outside [%d, %d]", (unsigned)m,
                    ODT_SCALE_M_MIN, ODT_SCALE_M_MAX);
        exit(1);
    }
    if (e < ODT_SCALE_E_MIN || e > ODT_SCALE_E_MAX) {
        PRINT_ERROR("deserializeQConfig: scale exponent %d outside [%d, %d]", (int)e,
                    ODT_SCALE_E_MIN, ODT_SCALE_E_MAX);
        exit(1);
    }
    return (odtScale_t){.m = m, .e = e};
}

static void deserializeQConfig(quantization_t *q, FILE *f, size_t numberOfElements) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symIntQC = q->qConfig;
        odtScaleStore(&symIntQC->scale, deserializeScale(f));
        symIntQC->roundingMode = (roundingMode_t)serialReadU8(f);
        symIntQC->qMaxBits = serialReadU8(f);
        if (symIntQC->qMaxBits == 0) {
            PRINT_ERROR("deserializeQConfig: SYM_INT32 file qMaxBits 0 has no representable "
                        "value");
            exit(1);
        }
        break;
    }
    case SYM: {
        symQConfig_t *symQC = q->qConfig;
        groupShape_t shape = deserializeGroupShape(f, numberOfElements, "SYM");
        if (shape.numGroups != symQC->numGroups) {
            freeReservedMemory(symQC->scales);
            symQC->scales = reserveMemory(shape.numGroups * sizeof(*symQC->scales));
            symQC->numGroups = shape.numGroups;
        }
        symQC->groupSize = shape.groupSize;
        for (size_t g = 0; g < shape.numGroups; g++) {
            odtScaleStore(&symQC->scales[g], deserializeScale(f));
        }
        uint8_t fileQBits = serialReadU8(f);
        /* 31 is the SYM storage ceiling; 1 has qMax == 0 (inf scale, NaN dequant). */
        if (fileQBits < 2 || fileQBits > 31) {
            PRINT_ERROR("deserializeQConfig: SYM file qBits %u outside [2, 31]",
                        (unsigned)fileQBits);
            exit(1);
        }
        symQC->qBits = fileQBits;
        symQC->roundingMode = (roundingMode_t)serialReadU8(f);
        break;
    }
    case ASYM: {
        asymQConfig_t *asymQC = q->qConfig;
        groupShape_t shape = deserializeGroupShape(f, numberOfElements, "ASYM");
        if (shape.numGroups != asymQC->numGroups) {
            freeReservedMemory(asymQC->scales);
            freeReservedMemory(asymQC->zeroPoints);
            asymQC->scales = reserveMemory(shape.numGroups * sizeof(*asymQC->scales));
            asymQC->zeroPoints = reserveMemory(shape.numGroups * sizeof(uint16_t));
            asymQC->numGroups = shape.numGroups;
        }
        asymQC->groupSize = shape.groupSize;
        for (size_t g = 0; g < shape.numGroups; g++) {
            odtScaleStore(&asymQC->scales[g], deserializeScale(f));
        }
        for (size_t g = 0; g < shape.numGroups; g++) {
            asymQC->zeroPoints[g] = serialReadU16LE(f);
        }
        uint8_t fileQBits = serialReadU8(f);
        /* uint16 code-domain zero points cap qBits at 16. */
        if (fileQBits == 0 || fileQBits > 16) {
            PRINT_ERROR("deserializeQConfig: ASYM file qBits %u outside [1, 16]",
                        (unsigned)fileQBits);
            exit(1);
        }
        asymQC->qBits = fileQBits;
        asymQC->roundingMode = (roundingMode_t)serialReadU8(f);
        break;
    }
    case BFP: {
        bfpQConfig_t *bfpQC = q->qConfig;
        groupShape_t shape = deserializeGroupShape(f, numberOfElements, "BFP");
        if (shape.numGroups != bfpQC->numGroups) {
            freeReservedMemory(bfpQC->exponents);
            bfpQC->exponents = reserveMemory(shape.numGroups * sizeof(uint8_t));
            bfpQC->numGroups = shape.numGroups;
        }
        bfpQC->groupSize = shape.groupSize;
        serialReadBytes(bfpQC->exponents, shape.numGroups, f);
        /* Validate widths before deriving the exponent bias: they are the
         * factory caps, and outside them the bias shift and qMax are undefined. */
        uint8_t fileMantissaBits = serialReadU8(f);
        if (fileMantissaBits < 2 || fileMantissaBits > 16) {
            PRINT_ERROR("deserializeQConfig: BFP file mantissaBits (%u) outside [2, 16]",
                        (unsigned)fileMantissaBits);
            exit(1);
        }
        uint8_t fileExponentBits = serialReadU8(f);
        if (fileExponentBits < 2 || fileExponentBits > 8) {
            PRINT_ERROR("deserializeQConfig: BFP file exponentBits (%u) outside [2, 8]",
                        (unsigned)fileExponentBits);
            exit(1);
        }
        bfpQC->mantissaBits = fileMantissaBits;
        bfpQC->exponentBits = fileExponentBits;
        /* Reject stored exponents whose scale would exceed the finite float32
         * range (0 * inf dequantizes to NaN, past every runtime guard) or the
         * record's exponent width (at narrow widths bias + 127 lies above 2^e - 1). */
        int32_t maxFiniteStored = bfpExponentBias(bfpQC) + 127;
        int32_t maxWidthStored = (int32_t)((1u << fileExponentBits) - 1u);
        int32_t maxStored = maxFiniteStored < maxWidthStored ? maxFiniteStored : maxWidthStored;
        for (size_t g = 0; g < shape.numGroups; g++) {
            if ((int32_t)bfpQC->exponents[g] > maxStored) {
                PRINT_ERROR("deserializeQConfig: BFP file exponent[%zu] (%u) exceeds the largest "
                            "valid stored exponent at exponentBits %u (min(2^e - 1, bias + 127) "
                            "== %d)",
                            g, (unsigned)bfpQC->exponents[g], (unsigned)fileExponentBits,
                            maxStored);
                exit(1);
            }
        }
        bfpQC->roundingMode = (roundingMode_t)serialReadU8(f);
        break;
    }
    case BFP_RAW:
        PRINT_ERROR("deserializeQConfig: BFP_RAW is executeOp scratch, never stored");
        exit(1);
    }
}

void readTensorRecordHeader(tensorRecordHeader_t *header, FILE *f) {
    uint32_t numDims = serialReadU32LE(f);
    if (numDims > ODTS_TENSOR_RECORD_MAX_RANK) {
        PRINT_ERROR("readTensorRecordHeader: rank %u exceeds the %u-dim cap", (unsigned)numDims,
                    (unsigned)ODTS_TENSOR_RECORD_MAX_RANK);
        exit(1);
    }
    header->numberOfDimensions = numDims;
    for (uint32_t d = 0; d < numDims; d++) {
        header->dimensions[d] = (size_t)serialReadU32LE(f);
    }
    size_t order[ODTS_TENSOR_RECORD_MAX_RANK];
    deserializeAxisOrder(order, numDims, f, "readTensorRecordHeader");
    shape_t wireShape = {
        .numberOfDimensions = numDims, .dimensions = header->dimensions, .orderOfDimensions = NULL};
    header->numberOfElements = calcNumberOfElementsByShape(&wireShape);

    uint8_t type = serialReadU8(f);
    header->quantization.type = (qtype_t)type;
    header->quantization.qConfig = NULL;
    /* Zeroed storage: group arrays start NULL with numGroups 0, so
     * deserializeQConfig reserves them at the file's group count. */
    memset(&header->qConfigStorage, 0, sizeof(header->qConfigStorage));
    switch (header->quantization.type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32:
        header->quantization.qConfig = &header->qConfigStorage.symInt32;
        break;
    case SYM:
        header->quantization.qConfig = &header->qConfigStorage.sym;
        break;
    case ASYM:
        header->quantization.qConfig = &header->qConfigStorage.asym;
        break;
    case BFP:
        header->quantization.qConfig = &header->qConfigStorage.bfp;
        break;
    default:
        PRINT_ERROR("readTensorRecordHeader: unknown qtype %u", (unsigned)type);
        exit(1);
    }
    deserializeQConfig(&header->quantization, f, header->numberOfElements);
    header->payloadBytes =
        calcNumberOfBytesForData(&header->quantization, header->numberOfElements);
}

void freeTensorRecordHeader(tensorRecordHeader_t *header) {
    switch (header->quantization.type) {
    case SYM:
        freeReservedMemory(header->qConfigStorage.sym.scales);
        header->qConfigStorage.sym.scales = NULL;
        break;
    case ASYM:
        freeReservedMemory(header->qConfigStorage.asym.scales);
        freeReservedMemory(header->qConfigStorage.asym.zeroPoints);
        header->qConfigStorage.asym.scales = NULL;
        header->qConfigStorage.asym.zeroPoints = NULL;
        break;
    case BFP:
        freeReservedMemory(header->qConfigStorage.bfp.exponents);
        header->qConfigStorage.bfp.exponents = NULL;
        break;
    default:
        break;
    }
}

static void skipSerializedTensor(FILE *f) {
    if (ftell(f) < 0) {
        PRINT_ERROR("skipSerializedTensor: stream not seekable (grad-record skip requires "
                    "fseek/ftell support)");
        exit(1);
    }

    tensorRecordHeader_t header;
    readTensorRecordHeader(&header, f);
    size_t payloadBytes = header.payloadBytes;
    freeTensorRecordHeader(&header);

    /* A larger value turns negative in the (long) cast and seeks backwards,
     * which the truncation check below cannot see. */
    if (payloadBytes > (size_t)LONG_MAX) {
        PRINT_ERROR("skipSerializedTensor: payload of %zu bytes exceeds the seekable range",
                    payloadBytes);
        exit(1);
    }
    if (fseek(f, (long)payloadBytes, SEEK_CUR) != 0) {
        PRINT_ERROR("skipSerializedTensor: seek past payload failed");
        exit(1);
    }
    /* fseek past a truncated file's real end succeeds silently (POSIX allows
     * seeking beyond EOF), so only comparing against the on-disk size
     * catches it. */
    long recordEnd = ftell(f);
    if (recordEnd < 0 || fseek(f, 0, SEEK_END) != 0) {
        PRINT_ERROR("skipSerializedTensor: position query failed after skip");
        exit(1);
    }
    long fileSize = ftell(f);
    if (fileSize < 0 || recordEnd > fileSize) {
        PRINT_ERROR("skipSerializedTensor: skipped record extends past end of file (truncated?)");
        exit(1);
    }
    if (fseek(f, recordEnd, SEEK_SET) != 0) {
        PRINT_ERROR("skipSerializedTensor: reposition after truncation check failed");
        exit(1);
    }
}

/* Unlike the tolerated grad byte, hasBias must match the skeleton: 1 into a
 * bias-free skeleton would write through a NULL parameter, 0 into a biased one
 * would silently keep its init bias. */
static void deserializeOptionalBias(parameter_t *skeletonBias, const char *layerName, FILE *f) {
    uint8_t hasBias = serialReadU8(f);
    if (hasBias > 1u || (hasBias == 1u) != (skeletonBias != NULL)) {
        PRINT_ERROR("deserializeModel: %s record hasBias %u does not match the skeleton "
                    "(bias %s)",
                    layerName, (unsigned)hasBias, skeletonBias != NULL ? "present" : "absent");
        exit(1);
    }
    if (hasBias) {
        deserializeParameter(skeletonBias, f);
    }
}

/* The FLOAT32 merge gate trusts addLayerInit, so the reader refuses what the factory refuses. */
static void requireAddFloat32Math(arithmetic_t a, const char *field) {
    if (a.type != ARITH_FLOAT32) {
        PRINT_ERROR("deserializeModel: ADD %s must be ARITH_FLOAT32 (got %d)", field, (int)a.type);
        exit(1);
    }
}

/* The record tail most layers share, in wire order. */
static void deserializeLayerTail(arithmetic_t *forwardMath, arithmetic_t *propLossMath,
                                 quantization_t *outputQ, quantization_t *propLossQ, FILE *f) {
    deserializeArithmetic(forwardMath, f);
    deserializeArithmetic(propLossMath, f);
    deserializeQuantization(outputQ, f, 0);
    deserializeQuantization(propLossQ, f, 0);
}

static void deserializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *linearConfig = layer->config->linear;
        deserializeParameter(linearConfig->weights, f);
        deserializeOptionalBias(linearConfig->bias, "LINEAR", f);
        deserializeArithmetic(&linearConfig->forwardMath, f);
        deserializeArithmetic(&linearConfig->weightGradMath, f);
        deserializeArithmetic(&linearConfig->biasGradMath, f);
        deserializeArithmetic(&linearConfig->propLossMath, f);
        deserializeQuantization(linearConfig->outputQ, f, 0);
        deserializeQuantization(linearConfig->propLossQ, f, 0);
        break;
    }
    case RELU: {
        reluConfig_t *reluConfig = layer->config->relu;
        deserializeLayerTail(&reluConfig->forwardMath, &reluConfig->propLossMath,
                             reluConfig->outputQ, reluConfig->propLossQ, f);
        reluValidateConfig(reluConfig, "deserializeModel: RELU");
        break;
    }
    case ADD: {
        addConfig_t *addConfig = layer->config->add;
        deserializeArithmetic(&addConfig->forwardMath, f);
        deserializeArithmetic(&addConfig->propLossMath, f);
        /* outputQ/propLossQ need no check: deserializeQuantization refuses a dtype that differs
         * from the FLOAT32 skeleton's. */
        requireAddFloat32Math(addConfig->forwardMath, "forwardMath");
        requireAddFloat32Math(addConfig->propLossMath, "propLossMath");
        deserializeQuantization(addConfig->outputQ, f, 0);
        deserializeQuantization(addConfig->propLossQ, f, 0);
        break;
    }
    case CONV1D: {
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        deserializeKernel(conv1dConfig->kernel, f);
        conv1dConfig->groups = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dConfig->weights, f);
        deserializeOptionalBias(conv1dConfig->bias, "CONV1D", f);
        deserializeArithmetic(&conv1dConfig->forwardMath, f);
        deserializeArithmetic(&conv1dConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dConfig->propLossMath, f);
        deserializeQuantization(conv1dConfig->outputQ, f, 0);
        deserializeQuantization(conv1dConfig->propLossQ, f, 0);
        conv1dValidateConfig(conv1dConfig, "deserializeModel: CONV1D");
        break;
    }
    case CONV1D_TRANSPOSED: {
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        deserializeKernel(conv1dTransposedConfig->kernel, f);
        conv1dTransposedConfig->groups = (size_t)serialReadU32LE(f);
        conv1dTransposedConfig->outputPadding = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dTransposedConfig->weights, f);
        deserializeOptionalBias(conv1dTransposedConfig->bias, "CONV1D_TRANSPOSED", f);
        deserializeArithmetic(&conv1dTransposedConfig->forwardMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->propLossMath, f);
        deserializeQuantization(conv1dTransposedConfig->outputQ, f, 0);
        deserializeQuantization(conv1dTransposedConfig->propLossQ, f, 0);
        conv1dTransposedValidateConfig(conv1dTransposedConfig,
                                       "deserializeModel: CONV1D_TRANSPOSED");
        break;
    }
    case MAXPOOL1D: {
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        deserializeKernel(maxPool1dConfig->kernel, f);
        deserializeLayerTail(&maxPool1dConfig->forwardMath, &maxPool1dConfig->propLossMath,
                             maxPool1dConfig->outputQ, maxPool1dConfig->propLossQ, f);
        break;
    }
    case AVGPOOL1D: {
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        deserializeKernel(avgPool1dConfig->kernel, f);
        deserializeLayerTail(&avgPool1dConfig->forwardMath, &avgPool1dConfig->propLossMath,
                             avgPool1dConfig->outputQ, avgPool1dConfig->propLossQ, f);
        break;
    }
    case SOFTMAX: {
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        deserializeLayerTail(&softmaxConfig->forwardMath, &softmaxConfig->propLossMath,
                             softmaxConfig->outputQ, softmaxConfig->propLossQ, f);
        softmaxValidateConfig(softmaxConfig, "deserializeModel: SOFTMAX");
        break;
    }
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION: {
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        deserializeQuantization(quantizationConfig->outputQ, f, 0);
        deserializeQuantization(quantizationConfig->propLossQ, f, 0);
        break;
    }
    case ADAPTIVE_AVGPOOL1D: {
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        adaptiveAvgPool1dConfig->outputSize = (size_t)serialReadU32LE(f);
        deserializeLayerTail(
            &adaptiveAvgPool1dConfig->forwardMath, &adaptiveAvgPool1dConfig->propLossMath,
            adaptiveAvgPool1dConfig->outputQ, adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    }
    case DROPOUT: {
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        dropoutConfig->p = serialReadF32LE(f);
        deserializeLayerTail(&dropoutConfig->forwardMath, &dropoutConfig->propLossMath,
                             dropoutConfig->outputQ, dropoutConfig->propLossQ, f);
        dropoutValidateConfig(dropoutConfig, "deserializeModel: DROPOUT");
        break;
    }
    case LAYERNORM: {
        layerNormConfig_t *layerNormConfig = layer->config->layerNorm;
        uint32_t layerNormNumNormDims = serialReadU32LE(f);
        /* numNormDims drives the normalizedShape read count; the skeleton's
         * array was sized by the build-time value — reject a mismatch before
         * file entries could be written past it. */
        if ((size_t)layerNormNumNormDims != layerNormConfig->numNormDims) {
            PRINT_ERROR("deserializeLayer: LayerNorm numNormDims mismatch (file %u, skeleton "
                        "%zu)",
                        (unsigned)layerNormNumNormDims, layerNormConfig->numNormDims);
            exit(1);
        }
        for (size_t d = 0; d < layerNormConfig->numNormDims; d++) {
            layerNormConfig->normalizedShape[d] = (size_t)serialReadU32LE(f);
        }
        layerNormConfig->eps = serialReadF32LE(f);
        deserializeParameter(layerNormConfig->gamma, f);
        deserializeParameter(layerNormConfig->beta, f);
        deserializeLayerTail(&layerNormConfig->forwardMath, &layerNormConfig->propLossMath,
                             layerNormConfig->outputQ, layerNormConfig->propLossQ, f);
        layerNormValidateConfig(layerNormConfig, "deserializeModel: LAYERNORM");
        break;
    }
    case GROUPNORM: {
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        groupNormConfig->numGroups = (size_t)serialReadU32LE(f);
        groupNormConfig->numChannels = (size_t)serialReadU32LE(f);
        groupNormConfig->eps = serialReadF32LE(f);
        deserializeParameter(groupNormConfig->gamma, f);
        deserializeParameter(groupNormConfig->beta, f);
        deserializeLayerTail(&groupNormConfig->forwardMath, &groupNormConfig->propLossMath,
                             groupNormConfig->outputQ, groupNormConfig->propLossQ, f);
        groupNormValidateConfig(groupNormConfig, "deserializeModel: GROUPNORM");
        break;
    }
    case BATCHNORM1D: {
        batchNorm1dConfig_t *c = layer->config->batchNorm1d;
        size_t fileC = (size_t)serialReadU32LE(f);
        float eps = serialReadF32LE(f);
        uint8_t mode = serialReadU8(f);
        float momentum = serialReadF32LE(f);
        uint8_t affine = serialReadU8(f);
        uint8_t track = serialReadU8(f);
        if (fileC != c->numChannels || affine > 1u || track > 1u || (affine == 1u) != c->affine ||
            (track == 1u) != c->trackRunningStats) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D record (C %zu, affine %u, track %u) does "
                        "not match the skeleton (C %zu, affine %d, track %d)",
                        fileC, (unsigned)affine, (unsigned)track, c->numChannels, (int)c->affine,
                        (int)c->trackRunningStats);
            exit(1);
        }
        if (!isfinite(eps) || eps <= 0.0f) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D eps must be finite and > 0");
            exit(1);
        }
        if (mode != BN_MOMENTUM_VALUE && mode != BN_MOMENTUM_CUMULATIVE) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D momentumMode %u unknown", (unsigned)mode);
            exit(1);
        }
        if (mode == BN_MOMENTUM_VALUE &&
            (!isfinite(momentum) || momentum < 0.0f || momentum > 1.0f)) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D momentum must be finite and in [0, 1]");
            exit(1);
        }
        c->eps = eps;
        c->momentumMode = (bnMomentumMode_t)mode;
        c->momentum = momentum;
        if (c->affine) {
            deserializeParameter(c->gamma, f);
            deserializeParameter(c->beta, f);
        }
        if (c->trackRunningStats) {
            float *rm = (float *)c->runningMean->data;
            float *rv = (float *)c->runningVar->data;
            for (size_t ch = 0; ch < c->numChannels; ch++) {
                rm[ch] = serialReadF32LE(f);
                if (!isfinite(rm[ch])) {
                    PRINT_ERROR("deserializeModel: BATCHNORM1D runningMean[%zu] not finite", ch);
                    exit(1);
                }
            }
            for (size_t ch = 0; ch < c->numChannels; ch++) {
                rv[ch] = serialReadF32LE(f);
                if (!isfinite(rv[ch]) || rv[ch] < 0.0f) {
                    PRINT_ERROR("deserializeModel: BATCHNORM1D runningVar[%zu] must be finite "
                                "and >= 0",
                                ch);
                    exit(1);
                }
            }
            c->numBatchesTracked = serialReadU64LE(f);
        }
        deserializeLayerTail(&c->forwardMath, &c->propLossMath, c->outputQ, c->propLossQ, f);
        /* Mirror the factory rule (validateLayerQuantForBatchNorm1d): forward and backward
         * trust a FLOAT32-only config. */
        if (c->forwardMath.type != ARITH_FLOAT32) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D forwardMath must be ARITH_FLOAT32 (got %d) "
                        "-- BatchNorm1d is FLOAT32-only",
                        (int)c->forwardMath.type);
            exit(1);
        }
        if (c->propLossMath.type != ARITH_FLOAT32) {
            PRINT_ERROR("deserializeModel: BATCHNORM1D propLossMath must be ARITH_FLOAT32 (got "
                        "%d) -- BatchNorm1d is FLOAT32-only",
                        (int)c->propLossMath.type);
            exit(1);
        }
        break;
    }
    case LAYER_TYPE_COUNT:
        PRINT_ERROR("deserializeLayer: LAYER_TYPE_COUNT is the enum's count, not a layer type");
        exit(1);
    }
}
