#ifndef TENSORRECORD_H
#define TENSORRECORD_H

#include <stdio.h>

#include "Tensor.h"

/* Untrusted wire rank above this is a corrupt record: no shipped tensor
 * exceeds rank 8. */
#define ODTS_TENSOR_RECORD_MAX_RANK 8u

/* Layout version of the bytes serializeTensor writes: 5 is the ODTS version
 * that last changed them (v5 SYM/ASYM/BFP qconfig records). Containers that
 * embed tensor records (ODTR) store it so a file written under another record
 * layout fails at load by name. Changing those bytes is caught by the tensor
 * golden-bytes tests in UnitTestSerialize.c going RED; the fix is to bump this
 * together with SERIALIZE_FORMAT_VERSION and regenerate the goldens. */
#define ODTS_TENSOR_RECORD_VERSION 5u

/* ODTS container format version, written by serializeModel and the only value
 * deserializeModel accepts. One definition shared by writer, reader and the
 * golden pins in UnitTestSerialize.c (version history: Serialize.c). */
#define SERIALIZE_FORMAT_VERSION 7u
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
 *  rank above ODTS_TENSOR_RECORD_MAX_RANK, an unknown dtype, an overflowing
 *  element count, or any qConfig check. Pair with freeTensorRecordHeader.
 */
void readTensorRecordHeader(tensorRecordHeader_t *header, FILE *f);

/*! Frees the SYM/ASYM/BFP per-group arrays readTensorRecordHeader reserved.
 *  Shape fields, quantization.type and payloadBytes stay readable. */
void freeTensorRecordHeader(tensorRecordHeader_t *header);

#endif // TENSORRECORD_H
