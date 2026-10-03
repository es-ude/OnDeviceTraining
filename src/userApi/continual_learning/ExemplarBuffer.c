#define SOURCE_FILE "EXEMPLAR_BUFFER"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "Common.h"
#include "ExecuteOp.h"
#include "ExemplarBuffer.h"
#include "QuantizationApi.h"
#include "StorageApi.h"
#include "TensorApi.h"

static bool exemplarMulFits(size_t a, size_t b, size_t *product) {
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    *product = a * b;
    return true;
}

static void *reserveOrExit(size_t bytes, const char *what) {
    void *p = reserveMemory(bytes);
    if (p == NULL) {
        PRINT_ERROR("exemplarBufferCreate: cannot reserve %zu bytes for the %s", bytes, what);
        exit(1);
    }
    return p;
}

exemplarBuffer_t *exemplarBufferCreate(size_t numClasses, size_t capacity) {
    if (numClasses == 0 || capacity == 0) {
        PRINT_ERROR("exemplarBufferCreate: numClasses and capacity must be >= 1");
        exit(1);
    }
    /* the counts array cannot wrap once the slot bytes fit: numClasses <= slots
     * and sizeof(uint32_t) <= sizeof(tensor_t *) */
    size_t slots, slotBytes;
    if (!exemplarMulFits(numClasses, capacity, &slots) ||
        !exemplarMulFits(slots, sizeof(tensor_t *), &slotBytes)) {
        PRINT_ERROR("exemplarBufferCreate: numClasses %zu * capacity %zu overflows size_t",
                    numClasses, capacity);
        exit(1);
    }
    exemplarBuffer_t *buf = reserveOrExit(sizeof(exemplarBuffer_t), "buffer");
    buf->numClasses = numClasses;
    buf->capacity = capacity;
    buf->items = reserveOrExit(slotBytes, "exemplar slots");
    buf->counts = reserveOrExit(numClasses * sizeof(uint32_t), "class counts");
    for (size_t i = 0; i < slots; i++) {
        buf->items[i] = NULL;
    }
    for (size_t c = 0; c < numClasses; c++) {
        buf->counts[c] = 0;
    }
    return buf;
}

void exemplarBufferAdd(exemplarBuffer_t *buf, const tensor_t *item, size_t classIndex) {
    if (classIndex >= buf->numClasses) {
        PRINT_ERROR("exemplarBufferAdd: classIndex %zu out of range (numClasses %zu)", classIndex,
                    buf->numClasses);
        exit(1);
    }
    /* every stored exemplar must match the first one's element count (one
     * model input shape per buffer) */
    for (size_t c = 0; c < buf->numClasses; c++) {
        if (buf->counts[c] > 0) {
            size_t stored = calcNumberOfElementsByTensor(buf->items[c * buf->capacity]);
            if (calcNumberOfElementsByTensor((tensor_t *)item) != stored) {
                PRINT_ERROR("exemplarBufferAdd: item element count %zu != stored %zu",
                            calcNumberOfElementsByTensor((tensor_t *)item), stored);
                exit(1);
            }
            break;
        }
    }
    if (buf->counts[classIndex] == buf->capacity) {
        return; /* first-K: class full, later samples are dropped */
    }
    tensor_t *copy = initTensor(getShapeLike(item->shape), getQLike(item->quantization), NULL);
    /* sources-never-mutated funnel contract: stripping const is safe */
    executeConvert((tensor_t *)item, copy);
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
