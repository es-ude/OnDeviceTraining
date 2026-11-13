#define SOURCE_FILE "COMPARISON"

#include <string.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Comparison.h"
#include "DTypes.h"
#include "Quantization.h"
#include "TensorConversion.h"

void gteFloatValue(tensor_t *a, float b, float altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    float *inputValues = (float *)a->data;
    float *outputValues = (float *)result->data;

    for (size_t i = 0; i < numberOfValues; i++) {
        outputValues[i] = inputValues[i] < b ? altNumber : inputValues[i];
    }
}

void gteSymInt32Zero(tensor_t *a, int32_t altNumber, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    int32_t *inputValues = (int32_t *)a->data;
    int32_t *outputValues = (int32_t *)result->data;

    for (size_t i = 0; i < numberOfValues; i++) {
        outputValues[i] = inputValues[i] < 0 ? altNumber : inputValues[i];
    }
}

/* Clamps in the code domain and copies the group exponents: zeroing a code only
 * shrinks its block's absmax, so every group's 2^E grid stays valid (no longer
 * absmax-tight). Re-deriving the exponents through executeOp would quantize unchanged
 * values a second time. */
void gteBfpZero(tensor_t *a, tensor_t *result) {
    size_t numberOfValues = calcNumberOfElementsByTensor(a);
    const bfpQConfig_t *inQC = a->quantization->qConfig;
    bfpQConfig_t *outQC = result->quantization->qConfig;

    int32_t codes[ODT_CONVERSION_CHUNK_ELEMS];
    for (size_t off = 0; off < numberOfValues; off += ODT_CONVERSION_CHUNK_ELEMS) {
        size_t count = numberOfValues - off < ODT_CONVERSION_CHUNK_ELEMS
                           ? numberOfValues - off
                           : ODT_CONVERSION_CHUNK_ELEMS;
        /* off is a multiple of 256, so off * mantissaBits is a whole number of bytes. */
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
