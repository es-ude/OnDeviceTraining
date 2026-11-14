#ifndef TENSORRECORD_H
#define TENSORRECORD_H

#include <stdio.h>

#include "Tensor.h"

/* Untrusted wire rank above this is a corrupt record: no shipped tensor
 * exceeds rank 8. */
#define ODTS_TENSOR_RECORD_MAX_RANK 8u

/* Layout version of the bytes serializeTensor writes: the ODTS version that
 * last changed them. Containers that embed tensor records (ODTR, ODTA) store it, so
 * a file written under another record layout fails at load by name. A change
 * to those bytes bumps this together with SERIALIZE_FORMAT_VERSION. */
#define ODTS_TENSOR_RECORD_VERSION 9u

/* ODTS container format version, written by serializeModel and the only value
 * deserializeModel accepts. */
#define SERIALIZE_FORMAT_VERSION 9u
_Static_assert(SERIALIZE_FORMAT_VERSION >= ODTS_TENSOR_RECORD_VERSION,
               "the tensor-record version names the ODTS version that introduced its layout");

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
 *  rank above ODTS_TENSOR_RECORD_MAX_RANK, an axis order that is not a
 *  permutation, an unknown dtype, an overflowing
 *  element count, or any qConfig check. Pair with freeTensorRecordHeader.
 */
void readTensorRecordHeader(tensorRecordHeader_t *header, FILE *f);

/*! Frees the SYM/ASYM/BFP per-group arrays readTensorRecordHeader reserved.
 *  Shape fields, quantization.type and payloadBytes stay readable. */
void freeTensorRecordHeader(tensorRecordHeader_t *header);

#endif // TENSORRECORD_H
