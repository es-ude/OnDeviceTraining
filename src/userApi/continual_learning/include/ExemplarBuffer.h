#ifndef ODT_EXEMPLAR_BUFFER_H
#define ODT_EXEMPLAR_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "Tensor.h"

/* Raw-rehearsal buffer: per class, the FIRST `capacity` samples ever offered
 * are stored (later adds are dropped — first-K, no reservoir). Stored
 * exemplars are copies in the item's own storage; the replay loader lends
 * pointers into the buffer, so it must outlive the wrapped loader. Every item
 * offered must be in identity dimension order (no transposed views) and have
 * the first stored exemplar's rank and dimensions, across all classes; a
 * mismatch exits. */
typedef struct {
    size_t numClasses, capacity;
    tensor_t **items; /* [numClasses * capacity]; NULL until stored */
    uint32_t *counts; /* [numClasses] stored exemplars (<= capacity) */
} exemplarBuffer_t;

exemplarBuffer_t *exemplarBufferCreate(size_t numClasses, size_t capacity);
void exemplarBufferAdd(exemplarBuffer_t *buf, const tensor_t *item, size_t classIndex);
void freeExemplarBuffer(exemplarBuffer_t *buf);

#endif // ODT_EXEMPLAR_BUFFER_H
