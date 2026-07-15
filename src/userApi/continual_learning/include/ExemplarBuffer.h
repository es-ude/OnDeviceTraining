#ifndef ODT_EXEMPLAR_BUFFER_H
#define ODT_EXEMPLAR_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "Tensor.h"

/* Raw-rehearsal baseline (#326 fair comparison): per class, the FIRST
 * `capacity` samples ever offered are stored (adds beyond capacity are
 * dropped — first-K policy, no reservoir). Stored exemplars are COPIES
 * (executeConvert into cloned storage: any item dtype, mirroring the
 * loader-pool contract); the replay loader hands out POINTERS into the
 * buffer, so stored tensors must outlive the wrapped loader. */
typedef struct {
    size_t numClasses, capacity;
    tensor_t **items; /* [numClasses * capacity]; NULL until stored */
    uint32_t *counts; /* [numClasses] stored exemplars (<= capacity) */
} exemplarBuffer_t;

exemplarBuffer_t *exemplarBufferCreate(size_t numClasses, size_t capacity);
void exemplarBufferAdd(exemplarBuffer_t *buf, const tensor_t *item, size_t classIndex);
void freeExemplarBuffer(exemplarBuffer_t *buf);

#endif // ODT_EXEMPLAR_BUFFER_H
