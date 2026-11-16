#define SOURCE_FILE "EXEMPLAR_BUFFER"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "CheckedSize.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "ExemplarBuffer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

exemplarBuffer_t *exemplarBufferCreate(size_t numClasses, size_t capacity) {
    if (numClasses == 0 || capacity == 0) {
        PRINT_ERROR("exemplarBufferCreate: numClasses and capacity must be >= 1");
        exit(1);
    }
    /* counts[c] is uint32_t and runs up to capacity */
    if ((uint64_t)capacity > UINT32_MAX) {
        PRINT_ERROR("exemplarBufferCreate: capacity %zu exceeds the uint32_t per-class count",
                    capacity);
        exit(1);
    }
    /* the counts array cannot wrap once the slot bytes fit: numClasses <= slots
     * and sizeof(uint32_t) <= sizeof(tensor_t *) */
    size_t slots, slotBytes;
    if (!checkedMulSize(numClasses, capacity, &slots) ||
        !checkedMulSize(slots, sizeof(tensor_t *), &slotBytes)) {
        PRINT_ERROR("exemplarBufferCreate: numClasses %zu * capacity %zu overflows size_t",
                    numClasses, capacity);
        exit(1);
    }
    exemplarBuffer_t *buf = reserveMemory(sizeof(exemplarBuffer_t));
    buf->numClasses = numClasses;
    buf->capacity = capacity;
    buf->items = reserveMemory(slotBytes);
    buf->counts = reserveMemory(numClasses * sizeof(uint32_t));
    for (size_t i = 0; i < slots; i++) {
        buf->items[i] = NULL;
    }
    for (size_t c = 0; c < numClasses; c++) {
        buf->counts[c] = 0;
    }
    return buf;
}

static void requireStoredShape(const tensor_t *item, const tensor_t *stored) {
    const shape_t *is = item->shape;
    const shape_t *ss = stored->shape;
    if (is->numberOfDimensions != ss->numberOfDimensions) {
        PRINT_ERROR("exemplarBufferAdd: item rank %zu != stored rank %zu", is->numberOfDimensions,
                    ss->numberOfDimensions);
        exit(1);
    }
    /* both orders are identity (requireIdentityOrder), so the dimensions alone decide */
    for (size_t d = 0; d < ss->numberOfDimensions; d++) {
        if (is->dimensions[d] != ss->dimensions[d]) {
            PRINT_ERROR("exemplarBufferAdd: item dimension %zu has size %zu, the stored exemplars "
                        "have %zu",
                        d, is->dimensions[d], ss->dimensions[d]);
            exit(1);
        }
    }
}

/* The copy converts element-wise in storage order into an identity-order
 * shape (getShapeLike), so a transposed view would silently be stored as its
 * untransposed base. */
static void requireIdentityOrder(const tensor_t *item) {
    const shape_t *is = item->shape;
    for (size_t d = 0; d < is->numberOfDimensions; d++) {
        if (is->orderOfDimensions[d] != d) {
            PRINT_ERROR("exemplarBufferAdd: item is not in identity dimension order (dimension "
                        "%zu has order %zu); transposed views cannot be stored",
                        d, is->orderOfDimensions[d]);
            exit(1);
        }
    }
}

void exemplarBufferAdd(exemplarBuffer_t *buf, const tensor_t *item, size_t classIndex) {
    if (classIndex >= buf->numClasses) {
        PRINT_ERROR("exemplarBufferAdd: classIndex %zu out of range (numClasses %zu)", classIndex,
                    buf->numClasses);
        exit(1);
    }
    requireIdentityOrder(item);
    /* one model input shape per buffer: the replay loader lends stored
     * exemplars zero-copy into batches of real samples */
    for (size_t c = 0; c < buf->numClasses; c++) {
        if (buf->counts[c] > 0) {
            requireStoredShape(item, buf->items[c * buf->capacity]);
            break;
        }
    }
    if (buf->counts[classIndex] == buf->capacity) {
        return; /* first-K: class full, later samples are dropped */
    }
    tensor_t *copy = initTensor(getShapeLike(item->shape), getQLike(item->quantization), NULL);
    /* sources-never-mutated funnel contract: stripping const is safe */
    executeConvert((tensor_t *)item, copy, NULL);
    buf->items[classIndex * buf->capacity + buf->counts[classIndex]] = copy;
    buf->counts[classIndex]++;
}

void freeExemplarBuffer(exemplarBuffer_t *buf) {
    if (buf == NULL) {
        return;
    }
    for (size_t c = 0; c < buf->numClasses; c++) {
        for (size_t i = 0; i < buf->counts[c]; i++) {
            freeTensor(buf->items[c * buf->capacity + i]);
        }
    }
    freeReservedMemory(buf->items);
    freeReservedMemory(buf->counts);
    freeReservedMemory(buf);
}
