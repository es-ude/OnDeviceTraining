#ifndef ODT_TENSOR_H
#define ODT_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Quantization.h"

typedef void *tensorStorageId;

typedef struct shape {
    size_t numberOfDimensions;
    size_t *dimensions;
    size_t *orderOfDimensions;
} shape_t;

typedef enum { SPARSITY_TYPE_1, SPARSITY_TYPE_2 } sparsityType_t;

typedef struct sparsityConfig {
} sparsityConfig;

typedef struct sparsity {
    sparsityType_t type;
    sparsityConfig *config;
} sparsity_t;

typedef struct tensor {
    uint8_t *data;
    shape_t *shape;
    quantization_t *quantization;
    sparsity_t *sparsity;
} tensor_t;

typedef struct parameter {
    tensor_t *param;
    tensor_t *grad;
} parameter_t;

uint32_t getBitmask(uint32_t startbit, uint32_t endbit);

uint8_t writeByte(uint8_t existingData, uint8_t data, uint8_t startbit, uint8_t endbit);

uint8_t readByte(uint8_t data, uint8_t startbit, uint8_t endbit);

void byteConversion(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut, size_t dataOutBits,
                    size_t numValues);

/* Like byteConversion, but both cursors are bit-granular: writes into
 * dataOut starting at BIT position dstStartBit, reads from dataIn starting
 * at BIT position srcStartBit, and performs no leading memset. Bits outside
 * the written range [dstStartBit, dstStartBit + numValues*dataOutBits) are
 * preserved, stale in-range bits are overwritten. Enables packing AND
 * decoding mixed-width segments (e.g. delta compression: wide base values +
 * narrow deltas) at non-byte-aligned boundaries. Reads exactly
 * numValues*dataInBits bits from srcStartBit, so a segment may also END
 * mid-byte. */
void byteConversionAppend(uint8_t *dataIn, size_t dataInBits, uint8_t *dataOut, size_t dataOutBits,
                          size_t numValues, size_t dstStartBit, size_t srcStartBit);

bool tensorBoolGet(tensor_t const *tensor, size_t flatIndex);
void tensorBoolSet(tensor_t *tensor, size_t flatIndex, bool value);

tensor_t *getParamFromParameter(parameter_t *parameter);
tensor_t *getGradFromParameter(parameter_t *parameter);

size_t calcBytesPerElement(quantization_t *quantization);
size_t calcBitsPerElement(quantization_t *quantization);
size_t calcBytesPerTensor(tensor_t *tensor);
/* A byte count that overflows size_t stops the program. */
size_t calcNumberOfBytesForData(quantization_t *q, size_t numberOfElements);

/* An element count that overflows size_t stops the program. */
size_t calcNumberOfElementsByShape(shape_t *shape);
size_t calcNumberOfElementsByTensor(tensor_t *tensor);
size_t calcNumberOfElementsByParameter(parameter_t *parameter);

void transposeTensor(tensor_t *tensor, size_t dim0Index, size_t dim1Index);

void setTensorValuesForConversion(uint8_t *data, quantization_t *q, tensor_t *originalTensor,
                                  tensor_t *outputTensor);
void setTensorValues(tensor_t *tensor, uint8_t *data, shape_t *shape, quantization_t *quantization,
                     sparsity_t *sparsity);
void setParameterValues(parameter_t *parameter, tensor_t *param, tensor_t *grad);
void setOrderOfDimsForNewTensor(size_t numberOfDimensions, size_t *orderOfDimensions);
void setShape(shape_t *shape, size_t *dims, size_t numberOfDims, size_t *orderOfDims);

void printTensor(tensor_t *t);
void printShape(shape_t *shape);

void copyTensor(tensor_t *dest, tensor_t *src);

#endif // ODT_TENSOR_H
