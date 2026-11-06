#define SOURCE_FILE "BFP_DATASET"

#include "bfp_dataset.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CheckedSize.h"
#include "Common.h"
#include "Quantization.h"
#include "QuantizationApi.h"
#include "SerialWire.h"
#include "Serialize.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"
#include "TensorRecord.h"

#define ODTA_MAGIC "ODTA"

static tensorArray_t *newTensorArray(size_t n) {
    size_t bytes;
    if (!checkedMulSize(n > 0 ? n : 1, sizeof(tensor_t *), &bytes)) {
        PRINT_ERROR("bfp_dataset: an array of %zu tensors overflows size_t", n);
        exit(1);
    }
    tensorArray_t *arr = reserveMemory(sizeof(tensorArray_t));
    tensor_t **slots = reserveMemory(bytes);
    if (arr == NULL || slots == NULL) {
        PRINT_ERROR("bfp_dataset: cannot reserve an array of %zu tensors", n);
        exit(1);
    }
    arr->size = n;
    arr->array = slots;
    return arr;
}

bool tensorArrayCountFits(uint64_t count, uint64_t remainingBytes, uint64_t sizeMax,
                          size_t pointerBytes) {
    if (count > remainingBytes / ODTA_MIN_RECORD_BYTES) {
        return false;
    }
    return pointerBytes == 0 || count <= sizeMax / pointerBytes;
}

/* Bytes between the stream position and the end of the file. */
static uint64_t bytesLeft(FILE *f, const char *path) {
    long here = ftell(f);
    if (here < 0 || fseek(f, 0, SEEK_END) != 0) {
        PRINT_ERROR("tensorArrayReadFile: %s: cannot determine the file size", path);
        exit(1);
    }
    long end = ftell(f);
    if (end < here || fseek(f, here, SEEK_SET) != 0) {
        PRINT_ERROR("tensorArrayReadFile: %s: cannot determine the file size", path);
        exit(1);
    }
    return (uint64_t)(end - here);
}

/* A fresh BFP quantization of q's storage format (widths, group shape,
 * rounding); its exponents are written by the conversion or the reader. */
static quantization_t *bfpQuantizationLike(const bfpQConfig_t *qc) {
    if (qc->numGroups > 1) {
        return quantizationInitBfpGrouped(qc->mantissaBits, qc->exponentBits, qc->roundingMode,
                                          qc->numGroups, qc->groupSize);
    }
    return quantizationInitBfp(qc->mantissaBits, qc->exponentBits, qc->roundingMode);
}

static shape_t *shapeLike(const shape_t *s) {
    size_t rank = s->numberOfDimensions;
    size_t *dims = reserveMemory((rank > 0 ? rank : 1) * sizeof(size_t));
    size_t *order = reserveMemory((rank > 0 ? rank : 1) * sizeof(size_t));
    for (size_t d = 0; d < rank; d++) {
        dims[d] = s->dimensions[d];
    }
    setOrderOfDimsForNewTensor(rank, order);
    shape_t *shape = reserveMemory(sizeof(shape_t));
    setShape(shape, dims, rank, order);
    return shape;
}

static const bfpQConfig_t *requireDeterministicBfp(const quantization_t *templateQ,
                                                   const char *what) {
    if (templateQ == NULL || templateQ->type != BFP) {
        PRINT_ERROR("%s: needs a BFP template, got dtype %d", what,
                    templateQ == NULL ? -1 : (int)templateQ->type);
        exit(1);
    }
    const bfpQConfig_t *qc = templateQ->qConfig;
    if (qc->roundingMode != HALF_AWAY) {
        PRINT_ERROR("%s: the offline conversion needs a HALF_AWAY template (got rounding mode %d): "
                    "it has no RNG stream, SR would need the online layer's per-call stream",
                    what, (int)qc->roundingMode);
        exit(1);
    }
    return qc;
}

static tensor_t *convertOne(tensor_t *src, const bfpQConfig_t *qc, size_t index, const char *what) {
    if (src->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: source tensor %zu has dtype %d, expected FLOAT32", what, index,
                    (int)src->quantization->type);
        exit(1);
    }
    tensor_t *dst = initTensor(shapeLike(src->shape), bfpQuantizationLike(qc), NULL);
    /* conversionMatrix[FLOAT32][BFP]: the cell the input Quantization layer
     * runs online; HALF_AWAY draws nothing, so no stream. */
    convertTensor(src, dst, NULL);
    return dst;
}

tensorArray_t *bfpDatasetConvert(const tensorArray_t *src, const quantization_t *templateQ) {
    const char *what = "bfpDatasetConvert";
    const bfpQConfig_t *qc = requireDeterministicBfp(templateQ, what);
    tensorArray_t *out = newTensorArray(src->size);
    for (size_t i = 0; i < src->size; i++) {
        out->array[i] = convertOne(src->array[i], qc, i, what);
    }
    return out;
}

tensorArray_t *bfpDatasetOneHotLabels(const tensorArray_t *classes, size_t numClasses,
                                      const quantization_t *templateQ) {
    const char *what = "bfpDatasetOneHotLabels";
    const bfpQConfig_t *qc = requireDeterministicBfp(templateQ, what);
    tensorArray_t *out = newTensorArray(classes->size);
    size_t dims[1] = {numClasses};
    size_t order[1] = {0};
    shape_t oneHotShape = {.dimensions = dims, .numberOfDimensions = 1, .orderOfDimensions = order};
    for (size_t i = 0; i < classes->size; i++) {
        tensor_t *c = classes->array[i];
        if (c->quantization->type != INT32 || calcNumberOfElementsByTensor(c) != 1) {
            PRINT_ERROR("%s: class tensor %zu must be one INT32 element", what, i);
            exit(1);
        }
        int32_t cls = ((int32_t *)c->data)[0];
        if (cls < 0 || (size_t)cls >= numClasses) {
            PRINT_ERROR("%s: class %ld of sample %zu is out of range [0, %zu)", what, (long)cls, i,
                        numClasses);
            exit(1);
        }
        tensor_t *f = initTensor(shapeLike(&oneHotShape), quantizationInitFloat(), NULL);
        float *v = (float *)f->data;
        for (size_t k = 0; k < numClasses; k++) {
            v[k] = (k == (size_t)cls) ? 1.0f : 0.0f;
        }
        out->array[i] = convertOne(f, qc, i, what);
        freeTensor(f);
        size_t got = 0;
        if (!bfpOneHotClass(out->array[i], &got) || got != (size_t)cls) {
            PRINT_ERROR("%s: the BFP label of sample %zu is not an exact one-hot of class %ld "
                        "(mantissa bits %u)",
                        what, i, (long)cls, (unsigned)qc->mantissaBits);
            exit(1);
        }
    }
    return out;
}

bool bfpOneHotClass(const tensor_t *label, size_t *cls) {
    if (label == NULL || label->quantization->type != BFP) {
        return false;
    }
    const bfpQConfig_t *qc = label->quantization->qConfig;
    size_t n = calcNumberOfElementsByTensor((tensor_t *)label);
    const int32_t bias = bfpExponentBias(qc);
    size_t groupSize = qc->groupSize == 0 ? n : qc->groupSize;
    size_t ones = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t c;
        unpackSignExtend(label->data, qc->mantissaBits, i * qc->mantissaBits, &c, 1);
        if (c == 0) {
            continue;
        }
        int e = (int)qc->exponents[i / groupSize] - bias;
        if (!(e <= 0 && e > -31 && c == ((int32_t)1 << (unsigned)(-e)))) {
            return false;
        }
        ones++;
        *cls = i;
    }
    return ones == 1;
}

void tensorArrayWriteFile(const char *path, const tensorArray_t *arr) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        PRINT_ERROR("tensorArrayWriteFile: cannot open %s for writing", path);
        exit(1);
    }
    serialWriteBytes(ODTA_MAGIC, 4, f);
    serialWriteU32LE(ODTS_TENSOR_RECORD_VERSION, f);
    serialWriteSizeAsU32LE(arr->size, f);
    for (size_t i = 0; i < arr->size; i++) {
        const shape_t *s = arr->array[i]->shape;
        for (size_t d = 0; d < s->numberOfDimensions; d++) {
            if (s->orderOfDimensions[d] != d) {
                PRINT_ERROR("tensorArrayWriteFile: tensor %zu is not in identity dimension order",
                            i);
                exit(1);
            }
        }
        serializeTensor(arr->array[i], f);
    }
    if (fclose(f) != 0) {
        PRINT_ERROR("tensorArrayWriteFile: closing %s failed", path);
        exit(1);
    }
}

static tensor_t *tensorFromRecord(tensorRecordHeader_t *h, size_t index, const char *path,
                                  uint64_t remaining) {
    /* Before anything is built: the payload must be in the file and within
     * the per-tensor limit (a tiny record must not request a huge buffer). */
    if ((uint64_t)h->payloadBytes > remaining || h->payloadBytes > ODTA_MAX_TENSOR_BYTES) {
        PRINT_ERROR("tensorArrayReadFile: %s: record %zu needs %zu payload bytes, the file has "
                    "%llu left (per-tensor limit %zu)",
                    path, index, h->payloadBytes, (unsigned long long)remaining,
                    (size_t)ODTA_MAX_TENSOR_BYTES);
        exit(1);
    }
    shape_t wire = {.dimensions = h->dimensions,
                    .numberOfDimensions = h->numberOfDimensions,
                    .orderOfDimensions = NULL};
    quantization_t *q;
    switch (h->quantization.type) {
    case FLOAT32:
        q = quantizationInitFloat();
        break;
    case INT32:
        q = quantizationInitInt32();
        break;
    case BFP:
        q = bfpQuantizationLike(&h->qConfigStorage.bfp);
        break;
    default:
        PRINT_ERROR("tensorArrayReadFile: %s: record %zu has dtype %d (FLOAT32, INT32 or BFP only)",
                    path, index, (int)h->quantization.type);
        exit(1);
    }
    tensor_t *t = initTensor(shapeLike(&wire), q, NULL);
    if (t == NULL || (h->payloadBytes > 0 && t->data == NULL)) {
        PRINT_ERROR("tensorArrayReadFile: %s: cannot reserve record %zu (%zu payload bytes)", path,
                    index, h->payloadBytes);
        exit(1);
    }
    if (q->type == BFP) {
        const bfpQConfig_t *src = &h->qConfigStorage.bfp;
        memcpy(((bfpQConfig_t *)q->qConfig)->exponents, src->exponents, src->numGroups);
    }
    size_t expected = calcNumberOfBytesForData(q, h->numberOfElements);
    if (expected != h->payloadBytes) {
        PRINT_ERROR("tensorArrayReadFile: %s: record %zu payload %zu bytes, expected %zu", path,
                    index, h->payloadBytes, expected);
        exit(1);
    }
    return t;
}

tensorArray_t *tensorArrayReadFile(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        PRINT_ERROR("tensorArrayReadFile: cannot open %s", path);
        exit(1);
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, ODTA_MAGIC, 4) != 0) {
        PRINT_ERROR("tensorArrayReadFile: %s is not an ODTA tensor-array file", path);
        exit(1);
    }
    uint32_t version = serialReadU32LE(f);
    if (version != ODTS_TENSOR_RECORD_VERSION) {
        PRINT_ERROR("tensorArrayReadFile: %s has tensor record version %u, this build reads %u",
                    path, (unsigned)version, (unsigned)ODTS_TENSOR_RECORD_VERSION);
        exit(1);
    }
    uint32_t fileCount = serialReadU32LE(f);
    if (!tensorArrayCountFits(fileCount, bytesLeft(f, path), SIZE_MAX, sizeof(tensor_t *))) {
        PRINT_ERROR("tensorArrayReadFile: %s declares %lu records, more records than the file can "
                    "hold",
                    path, (unsigned long)fileCount);
        exit(1);
    }
    size_t count = (size_t)fileCount;
    tensorArray_t *arr = newTensorArray(count);
    for (size_t i = 0; i < count; i++) {
        tensorRecordHeader_t h;
        readTensorRecordHeader(&h, f);
        tensor_t *t = tensorFromRecord(&h, i, path, bytesLeft(f, path));
        serialReadBytes(t->data, h.payloadBytes, f);
        freeTensorRecordHeader(&h);
        arr->array[i] = t;
    }
    int extra = fgetc(f);
    if (extra != EOF) {
        PRINT_ERROR("tensorArrayReadFile: %s has trailing bytes after its %zu records", path,
                    count);
        exit(1);
    }
    fclose(f);
    return arr;
}
