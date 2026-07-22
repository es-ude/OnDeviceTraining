#define SOURCE_FILE "DESERIALIZE"

#include <stdlib.h>
#include <string.h>

#include "AdaptiveAvgPool1d.h"
#include "ArithmeticType.h"
#include "AvgPool1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Deserialize.h"
#include "DeserializeInternal.h"
#include "Dropout.h"
#include "GroupNorm.h"
#include "Kernel.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MaxPool1d.h"
#include "QuantizationLayer.h"
#include "Relu.h"
#include "Rounding.h"
#include "SerialWire.h"
#include "Softmax.h"
#include "Tensor.h"

/* Mirrors Serialize.c's locked format v2 constants (#370): fixed-width
 * little-endian scalars via the checked SerialWire primitives; no v1
 * back-compat shim — v1 files were host-local artifacts.
 * v3: parameter records carry a grad-presence byte (#380). The reader is
 * TOLERANT of a presence/skeleton mismatch (#380 PR3, superseding PR1's
 * fail-fast): see deserializeParameter / skipSerializedTensor below. */
#define SERIALIZE_MAGIC "ODTS"
#define SERIALIZE_FORMAT_VERSION 3u

void deserializeTensor(tensor_t *tensor, FILE *f) {
    /* #316: capture the skeleton's expected payload size BEFORE the shape /
     * quantization overwrites. tensor->data was sized by initTensor from this
     * build-time shape + quantization; a file record with a larger element count
     * or packed width (which the dtype check alone cannot see) would otherwise
     * fread past that allocation. */
    size_t expectedBytes =
        calcNumberOfBytesForData(tensor->quantization, calcNumberOfElementsByShape(tensor->shape));

    deserializeShape(tensor->shape, f);
    deserializeQuantization(tensor->quantization, f);

    size_t numberOfValues = calcNumberOfElementsByShape(tensor->shape);
    /* Mirrors Serialize.c: payload length is the packed size. */
    size_t dataBytes = calcNumberOfBytesForData(tensor->quantization, numberOfValues);

    if (dataBytes != expectedBytes) {
        PRINT_ERROR("deserializeTensor: file payload %zu bytes does not match the skeleton's "
                    "allocated %zu bytes (shape/qBits mismatch)",
                    dataBytes, expectedBytes);
        exit(1);
    }

    serialReadBytes(tensor->data, dataBytes, f);
    deserializeSparsity();
}

void deserializeParameter(parameter_t *parameter, FILE *f) {
    uint8_t hasGrad = serialReadU8(f);
    deserializeTensor(parameter->param, f);
    if (hasGrad && parameter->grad != NULL) {
        deserializeTensor(parameter->grad, f);
    } else if (hasGrad) {
        /* Frozen skeleton, full checkpoint (#380 PR3): grads are transient
         * (zeroed before every batch) and a frozen layer allocates none -
         * parse past the record so the stream stays synced. */
        skipSerializedTensor(f);
    }
    /* hasGrad == 0 with a trainable skeleton: grads stay as built (zeroed);
     * optimizerZeroGrad re-zeros before every batch anyway. */
}

void deserializeModel(layer_t **model, size_t sizeModel, FILE *f) {
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
    if (layerCount != (uint32_t)sizeModel) {
        PRINT_ERROR("deserializeModel: layerCount mismatch (file has %u, caller expects %zu)",
                    (unsigned)layerCount, sizeModel);
        exit(1);
    }

    for (size_t i = 0; i < sizeModel; i++) {
        /* Tag byte = layerType_t enum position (append-only wire contract; see
         * Layer.h and the pins in test/unit/serial/UnitTestSerialize.c). */
        uint8_t tag = serialReadU8(f);
        if (tag != (uint8_t)model[i]->type) {
            PRINT_ERROR("deserializeModel: record tag %u does not match expected layer type "
                        "%u at index %zu",
                        (unsigned)tag, (unsigned)model[i]->type, i);
            exit(1);
        }
        deserializeLayer(model[i], f);
    }
}

// Helper Functions

static void deserializeShape(shape_t *shape, FILE *f) {
    uint32_t fileRank = serialReadU32LE(f);
    /* The skeleton's dimensions/orderOfDimensions arrays were sized by the
     * build-time rank; a mismatched record would otherwise write file dims
     * past them (and an equal-element-count rank change would slip past the
     * #316 payload-size check below). */
    if ((size_t)fileRank != shape->numberOfDimensions) {
        PRINT_ERROR("deserializeShape: file rank %u does not match the skeleton rank %zu",
                    (unsigned)fileRank, shape->numberOfDimensions);
        exit(1);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        shape->dimensions[d] = (size_t)serialReadU32LE(f);
    }
    for (size_t d = 0; d < shape->numberOfDimensions; d++) {
        shape->orderOfDimensions[d] = (size_t)serialReadU32LE(f);
    }
}

static void deserializeQuantization(quantization_t *q, FILE *f) {
    uint8_t type = serialReadU8(f);
    /* #316: the skeleton was built with a fixed dtype whose qConfig struct (or
     * NULL, for FLOAT32/INT32/BOOL) is fixed. A file record claiming a different
     * dtype would make deserializeQConfig write scale/qBits/... through a
     * mismatched or NULL qConfig — segfault on host, silent low-address writes on
     * an MMU-less MCU. Reject the mismatch before the overwrite; running here (not
     * in deserializeQConfig) forecloses the NULL-qConfig deref. */
    if ((qtype_t)type != q->type) {
        PRINT_ERROR("deserializeQuantization: file dtype %u does not match the skeleton dtype %u",
                    (unsigned)type, (unsigned)q->type);
        exit(1);
    }
    q->type = (qtype_t)type;
    deserializeQConfig(q, f);
}

static void deserializeArithmetic(arithmetic_t *arithmetic, FILE *f) {
    arithmetic->type = (arithmeticType_t)serialReadU8(f);
    arithmetic->roundingMode = (roundingMode_t)serialReadU8(f);
}

static void deserializeKernel(kernel_t *kernel, FILE *f) {
    kernel->size = (size_t)serialReadU32LE(f);
    kernel->paddingType = (paddingType_t)serialReadU8(f);
    kernel->stride = (size_t)serialReadU32LE(f);
    kernel->dilation = (size_t)serialReadU32LE(f);
    kernel->padding = (size_t)serialReadU32LE(f);
}

static void deserializeQConfig(quantization_t *q, FILE *f) {
    switch (q->type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32: {
        symInt32QConfig_t *symIntQC = q->qConfig;
        symIntQC->scale = serialReadF32LE(f);
        symIntQC->roundingMode = (roundingMode_t)serialReadU8(f);
        symIntQC->qMaxBits = serialReadU8(f);
        break;
    }
    case SYM: {
        symQConfig_t *symQC = q->qConfig;
        symQC->scale = serialReadF32LE(f);
        symQC->qBits = serialReadU8(f);
        symQC->roundingMode = (roundingMode_t)serialReadU8(f);
        break;
    }
    case ASYM: {
        asymQConfig_t *asymQC = q->qConfig;
        asymQC->scale = serialReadF32LE(f);
        asymQC->qBits = serialReadU8(f);
        asymQC->roundingMode = (roundingMode_t)serialReadU8(f);
        asymQC->zeroPoint = serialReadI32LE(f);
        break;
    }
    default:
        PRINT_ERROR("Unknown qType!");
        exit(1);
    }
}

/* Sanity cap on skipSerializedTensor's untrusted file rank: no shipped tensor
 * exceeds rank 8, so a larger value is a corrupt/malicious record. Bounds the
 * dims-read loop below -- no dims[] array is sized off this value (#380 PR3
 * review: the array was write-only, dropped in favor of an inline product). */
#define SKIP_TENSOR_MAX_DIMS 8

static void skipSerializedTensor(FILE *f) {
    long recordStart = ftell(f);
    if (recordStart < 0) {
        PRINT_ERROR("skipSerializedTensor: stream not seekable (grad-record skip requires "
                    "fseek/ftell support, #380 PR3)");
        exit(1);
    }

    uint32_t numDims = serialReadU32LE(f);
    if (numDims > SKIP_TENSOR_MAX_DIMS) {
        PRINT_ERROR("skipSerializedTensor: rank %u exceeds the %d-dim skip-helper cap",
                    (unsigned)numDims, SKIP_TENSOR_MAX_DIMS);
        exit(1);
    }
    size_t numberOfElements = 1;
    for (uint32_t d = 0; d < numDims; d++) {
        numberOfElements *= (size_t)serialReadU32LE(f);
    }
    /* orderOfDimensions: positional only, irrelevant to a discarded record. */
    if (fseek(f, (long)numDims * (long)sizeof(uint32_t), SEEK_CUR) != 0) {
        PRINT_ERROR("skipSerializedTensor: seek past orderOfDimensions failed");
        exit(1);
    }

    uint8_t type = serialReadU8(f);
    quantization_t scratchQ = {.type = (qtype_t)type, .qConfig = NULL};
    symInt32QConfig_t symIntScratch;
    symQConfig_t symScratch;
    asymQConfig_t asymScratch;
    switch (scratchQ.type) {
    case INT32:
    case FLOAT32:
    case BOOL:
        break;
    case SYM_INT32:
        scratchQ.qConfig = &symIntScratch;
        break;
    case SYM:
        scratchQ.qConfig = &symScratch;
        break;
    case ASYM:
        scratchQ.qConfig = &asymScratch;
        break;
    default:
        PRINT_ERROR("skipSerializedTensor: unknown qtype %u", (unsigned)type);
        exit(1);
    }
    deserializeQConfig(&scratchQ, f);

    size_t payloadBytes = calcNumberOfBytesForData(&scratchQ, numberOfElements);
    if (fseek(f, (long)payloadBytes, SEEK_CUR) != 0) {
        PRINT_ERROR("skipSerializedTensor: seek past payload failed");
        exit(1);
    }
    /* deserializeSparsity() is a zero-byte TODO stub -- nothing to skip. */

    /* Post-skip truncation guard (mirrors the ODTR/PPCA ftell precedent,
     * #316 wave): fseek past a genuinely truncated file's real end succeeds
     * silently (POSIX allows seeking beyond EOF), so only comparing against
     * the actual on-disk size catches it. */
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

// TODO
static void deserializeSparsity() {}

static void deserializeLayer(layer_t *layer, FILE *f) {
    switch (layer->type) {
    case LINEAR: {
        linearConfig_t *linearConfig = layer->config->linear;
        deserializeParameter(linearConfig->weights, f);
        deserializeParameter(linearConfig->bias, f);
        deserializeArithmetic(&linearConfig->forwardMath, f);
        deserializeArithmetic(&linearConfig->weightGradMath, f);
        deserializeArithmetic(&linearConfig->biasGradMath, f);
        deserializeArithmetic(&linearConfig->propLossMath, f);
        deserializeQuantization(linearConfig->outputQ, f);
        deserializeQuantization(linearConfig->propLossQ, f);
        break;
    }
    case RELU: {
        reluConfig_t *reluConfig = layer->config->relu;
        deserializeArithmetic(&reluConfig->forwardMath, f);
        deserializeArithmetic(&reluConfig->propLossMath, f);
        deserializeQuantization(reluConfig->outputQ, f);
        deserializeQuantization(reluConfig->propLossQ, f);
        break;
    }
    case CONV1D: {
        conv1dConfig_t *conv1dConfig = layer->config->conv1d;
        deserializeKernel(conv1dConfig->kernel, f);
        conv1dConfig->groups = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dConfig->weights, f);
        uint8_t conv1dHasBias = serialReadU8(f);
        if (conv1dHasBias) {
            deserializeParameter(conv1dConfig->bias, f);
        }
        deserializeArithmetic(&conv1dConfig->forwardMath, f);
        deserializeArithmetic(&conv1dConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dConfig->propLossMath, f);
        deserializeQuantization(conv1dConfig->outputQ, f);
        deserializeQuantization(conv1dConfig->propLossQ, f);
        break;
    }
    case CONV1D_TRANSPOSED: {
        conv1dTransposedConfig_t *conv1dTransposedConfig = layer->config->conv1dTransposed;
        deserializeKernel(conv1dTransposedConfig->kernel, f);
        conv1dTransposedConfig->groups = (size_t)serialReadU32LE(f);
        conv1dTransposedConfig->outputPadding = (size_t)serialReadU32LE(f);
        deserializeParameter(conv1dTransposedConfig->weights, f);
        uint8_t conv1dTransposedHasBias = serialReadU8(f);
        if (conv1dTransposedHasBias) {
            deserializeParameter(conv1dTransposedConfig->bias, f);
        }
        deserializeArithmetic(&conv1dTransposedConfig->forwardMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->weightGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->biasGradMath, f);
        deserializeArithmetic(&conv1dTransposedConfig->propLossMath, f);
        deserializeQuantization(conv1dTransposedConfig->outputQ, f);
        deserializeQuantization(conv1dTransposedConfig->propLossQ, f);
        break;
    }
    case MAXPOOL1D: {
        maxPool1dConfig_t *maxPool1dConfig = layer->config->maxPool1d;
        deserializeKernel(maxPool1dConfig->kernel, f);
        deserializeArithmetic(&maxPool1dConfig->forwardMath, f);
        deserializeArithmetic(&maxPool1dConfig->propLossMath, f);
        deserializeQuantization(maxPool1dConfig->outputQ, f);
        deserializeQuantization(maxPool1dConfig->propLossQ, f);
        break;
    }
    case AVGPOOL1D: {
        avgPool1dConfig_t *avgPool1dConfig = layer->config->avgPool1d;
        deserializeKernel(avgPool1dConfig->kernel, f);
        deserializeArithmetic(&avgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&avgPool1dConfig->propLossMath, f);
        deserializeQuantization(avgPool1dConfig->outputQ, f);
        deserializeQuantization(avgPool1dConfig->propLossQ, f);
        break;
    }
    case SOFTMAX: {
        softmaxConfig_t *softmaxConfig = layer->config->softmax;
        deserializeArithmetic(&softmaxConfig->forwardMath, f);
        deserializeArithmetic(&softmaxConfig->propLossMath, f);
        deserializeQuantization(softmaxConfig->outputQ, f);
        deserializeQuantization(softmaxConfig->propLossQ, f);
        break;
    }
    case FLATTEN:
        // Flatten carries no state (no parameters, no quantization).
        break;
    case QUANTIZATION: {
        quantizationConfig_t *quantizationConfig = layer->config->quantization;
        deserializeQuantization(quantizationConfig->outputQ, f);
        deserializeQuantization(quantizationConfig->propLossQ, f);
        break;
    }
    case ADAPTIVE_AVGPOOL1D: {
        adaptiveAvgPool1dConfig_t *adaptiveAvgPool1dConfig = layer->config->adaptiveAvgPool1d;
        adaptiveAvgPool1dConfig->outputSize = (size_t)serialReadU32LE(f);
        deserializeArithmetic(&adaptiveAvgPool1dConfig->forwardMath, f);
        deserializeArithmetic(&adaptiveAvgPool1dConfig->propLossMath, f);
        deserializeQuantization(adaptiveAvgPool1dConfig->outputQ, f);
        deserializeQuantization(adaptiveAvgPool1dConfig->propLossQ, f);
        break;
    }
    case DROPOUT: {
        dropoutConfig_t *dropoutConfig = layer->config->dropout;
        dropoutConfig->p = serialReadF32LE(f);
        deserializeArithmetic(&dropoutConfig->forwardMath, f);
        deserializeArithmetic(&dropoutConfig->propLossMath, f);
        deserializeQuantization(dropoutConfig->outputQ, f);
        deserializeQuantization(dropoutConfig->propLossQ, f);
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
        deserializeArithmetic(&layerNormConfig->forwardMath, f);
        deserializeArithmetic(&layerNormConfig->propLossMath, f);
        deserializeQuantization(layerNormConfig->outputQ, f);
        deserializeQuantization(layerNormConfig->propLossQ, f);
        break;
    }
    case GROUPNORM: {
        groupNormConfig_t *groupNormConfig = layer->config->groupNorm;
        groupNormConfig->numGroups = (size_t)serialReadU32LE(f);
        groupNormConfig->numChannels = (size_t)serialReadU32LE(f);
        groupNormConfig->eps = serialReadF32LE(f);
        deserializeParameter(groupNormConfig->gamma, f);
        deserializeParameter(groupNormConfig->beta, f);
        deserializeArithmetic(&groupNormConfig->forwardMath, f);
        deserializeArithmetic(&groupNormConfig->propLossMath, f);
        deserializeQuantization(groupNormConfig->outputQ, f);
        deserializeQuantization(groupNormConfig->propLossQ, f);
        break;
    }
    default:
        PRINT_ERROR("Unsupported layer type!\n");
        exit(1);
    }
}
