#ifndef TENSORRECORD_H
#define TENSORRECORD_H

#include <stdio.h>

#include "Tensor.h"

/* Untrusted wire rank above this is a corrupt record: no shipped tensor
 * exceeds rank 8. */
#define ODTS_TENSOR_RECORD_MAX_RANK 8u

typedef struct {
    size_t numberOfDimensions;
    size_t dimensions[ODTS_TENSOR_RECORD_MAX_RANK];
    size_t numberOfElements;
    quantization_t quantization; /* qConfig points into qConfigStorage: never copy the struct */
    union {
        symInt32QConfig_t symInt32;
        symQConfig_t sym;
        asymQConfig_t asym;
        bfpQConfig_t bfp;
    } qConfigStorage;
    size_t payloadBytes;
} tensorRecordHeader_t;

/*! Parses shape + quantization of one ODTS tensor record into caller-owned
 *  scratch, with every check deserializeQConfig applies (caps, sentinel,
 *  divisibility against the record's own element count). Leaves the stream
 *  at the payload. Writes no tensor. Use it to validate or discard a record
 *  before (or instead of) deserializing it into a skeleton. Fails fast on a
 *  rank above ODTS_TENSOR_RECORD_MAX_RANK, an unknown dtype, an overflowing
 *  element count, or any qConfig check. Pair with freeTensorRecordHeader.
 */
void readTensorRecordHeader(tensorRecordHeader_t *header, FILE *f);

/*! Frees the SYM/ASYM/BFP per-group arrays readTensorRecordHeader reserved.
 *  Shape fields, quantization.type and payloadBytes stay readable. */
void freeTensorRecordHeader(tensorRecordHeader_t *header);

#endif // TENSORRECORD_H
