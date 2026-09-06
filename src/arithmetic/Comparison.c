#define SOURCE_FILE "COMPARISON"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Comparison.h"
#include "DTypes.h"
#include "Quantization.h"
#include "TensorConversion.h"

void gteInt32Value(tensor_t *a, int32_t b, int32_t altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    int32_t values[numberOfValues];
    readBytesAsInt32Array(numberOfValues, a->data, values);

    for (size_t i = 0; i < numberOfValues; i++) {
        if (values[i] < b) {
            values[i] = altNumber;
        }
    }

    writeInt32ArrayToByteArray(numberOfValues, values, result->data);
}

void gteInt32Tensor(tensor_t *a, tensor_t *b, int32_t altNumber, tensor_t *result) {
    size_t aNumberOfValues = calcNumberOfElementsByTensor(a);
    size_t bNumberOfValues = calcNumberOfElementsByTensor(b);
    if (aNumberOfValues != bNumberOfValues) {
        PRINT_ERROR("Mismatched number of values!");
        exit(1);
    }

    int32_t aValues[aNumberOfValues];
    int32_t bValues[bNumberOfValues];
    readBytesAsInt32Array(aNumberOfValues, a->data, aValues);
    readBytesAsInt32Array(bNumberOfValues, b->data, bValues);

    for (size_t i = 0; i < aNumberOfValues; i++) {
        if (aValues[i] < bValues[i]) {
            aValues[i] = altNumber;
        }
    }
    writeInt32ArrayToByteArray(aNumberOfValues, aValues, result->data);
}

void gteFloatValue(tensor_t *a, float b, float altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    float *inputValues = (float *)a->data;
    float *outputValues = (float *)result->data;

    for (size_t i = 0; i < numberOfValues; i++) {
        if (inputValues[i] < b) {
            outputValues[i] = altNumber;
        } else {
            outputValues[i] = inputValues[i];
        }
    }
}

void gteFloatTensor(tensor_t *a, tensor_t *b, float altNumber, tensor_t *result) {
    size_t aNumberOfValues = calcNumberOfElementsByTensor(a);
    size_t bNumberOfValues = calcNumberOfElementsByTensor(b);
    if (aNumberOfValues != bNumberOfValues) {
        PRINT_ERROR("Mismatched number of values!");
        exit(1);
    }

    float aValues[aNumberOfValues];
    float bValues[bNumberOfValues];
    readBytesAsFloatArray(aNumberOfValues, a->data, aValues);
    readBytesAsFloatArray(bNumberOfValues, b->data, bValues);

    for (size_t i = 0; i < aNumberOfValues; i++) {
        if (aValues[i] < bValues[i]) {
            aValues[i] = altNumber;
        }
    }
    writeFloatArrayToByteArray(aNumberOfValues, aValues, result->data);
}

void gteSymInt32Zero(tensor_t *a, int32_t altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);

    int32_t *inputValues = (int32_t *)a->data;
    int32_t *outputValues = (int32_t *)result->data;

    for (size_t i = 0; i < numberOfValues; i++) {
        int32_t currentValue = inputValues[i];
        if (currentValue < 0) {
            currentValue = altNumber;
        }
        outputValues[i] = currentValue;
    }
}

void gteSymInt32Value(tensor_t *a, int32_t b, int32_t altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    int32_t values[numberOfValues];
    readBytesAsInt32Array(numberOfValues, a->data, values);

    symInt32QConfig_t *aSymInt32QC = a->quantization->qConfig;
    float scale = aSymInt32QC->scale;
    float scaledB = (float)b / scale;

    for (size_t i = 0; i < numberOfValues; i++) {
        if ((float)values[i] < scaledB) {
            values[i] = altNumber;
        }
    }

    writeInt32ArrayToByteArray(numberOfValues, values, result->data);
}

void gteSymInt32Tensor(tensor_t *a, tensor_t *b, int32_t altNumber, tensor_t *result) {
    size_t aNumberOfValues = calcNumberOfElementsByTensor(a);
    size_t bNumberOfValues = calcNumberOfElementsByTensor(b);
    if (aNumberOfValues != bNumberOfValues) {
        PRINT_ERROR("Mismatched number of values!");
        exit(1);
    }

    float aValues[aNumberOfValues];
    float bValues[bNumberOfValues];
    readBytesAsFloatArray(aNumberOfValues, a->data, aValues);
    readBytesAsFloatArray(bNumberOfValues, b->data, bValues);

    for (size_t i = 0; i < aNumberOfValues; i++) {
        if (aValues[i] < bValues[i]) {
            aValues[i] = altNumber;
        }
    }
    writeFloatArrayToByteArray(aNumberOfValues, aValues, result->data);
}

/* BFP epic PR4 (R-P2): ReLU on PACKED BFP storage. Clamp negative mantissa
 * codes to 0 in the CODE domain and copy the group exponents VERBATIM: zeroing
 * a code only shrinks its block's absmax, so every group's 2^E grid stays
 * valid -- no longer absmax-tight, which is the documented utilization drop
 * (spec §10 deviation 6) and the exact analog of gteSymInt32Zero's scale copy.
 * Routing this through executeOp instead would re-derive the target's
 * exponents at OUT_WRITE: a SECOND quantization of unchanged values, which
 * spec §9 / D8 forbid. The clamp only shrinks magnitude, so the pack can never
 * overflow the code width and needs no guard. This function dereferences BOTH
 * packed buffers, so it gates both itself (the PR4 idiom) rather than trusting
 * its caller: element counts AND grid AND widths. */
void gteBfpZero(tensor_t *a, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    const bfpQConfig_t *inQC = a->quantization->qConfig;
    bfpQConfig_t *outQC = result->quantization->qConfig;
    bfpRequireSameGeometry(inQC, numberOfValues, outQC, calcNumberOfElementsByTensor(result),
                           "gteBfpZero (packed-BFP ReLU)");

    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < numberOfValues; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = numberOfValues - off < ODT_CONVERSION_CHUNK_ELEMS
                           ? numberOfValues - off
                           : ODT_CONVERSION_CHUNK_ELEMS;
        /* off is a multiple of 256, so off*mantissaBits is a whole number of
         * bytes for every legal width -- the packed-chunk alignment contract. */
        unpackSignExtend((const uint8_t *)a->data + off * inQC->mantissaBits / 8,
                         inQC->mantissaBits, 0, codes, count);
        for (size_t i = 0; i < count; i++) {
            if (codes[i] < 0) {
                codes[i] = 0;
            }
        }
        byteConversion((uint8_t *)codes, 32,
                       (uint8_t *)result->data + off * outQC->mantissaBits / 8, outQC->mantissaBits,
                       count);
    }
    memcpy(outQC->exponents, inQC->exponents, inQC->numGroups);
}
