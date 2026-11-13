#define SOURCE_FILE "POINTWISE_FUSED"

#include <math.h>
#include <stdint.h>

#include "Common.h"
#include "PointwiseFused.h"
#include "Tensor.h"

#pragma STDC FP_CONTRACT OFF

size_t lerpInstructionCounter = 0;
size_t addcmulInstructionCounter = 0;
size_t addcdivDenomInstructionCounter = 0;

void lerpFloat32TensorsInplace(tensor_t *a, tensor_t *b, float weight) {
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float diff = bData[i] - aData[i];
        aData[i] = fmaf(weight, diff, aData[i]);
#ifdef TRACK_INSTRUCTIONS
        ++lerpInstructionCounter;
#endif
    }
}

void addcmulFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *c, float s) {
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    const float *cData = (const float *)c->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float t = s * bData[i];
        float t2 = t * cData[i];
        aData[i] = aData[i] + t2;
#ifdef TRACK_INSTRUCTIONS
        ++addcmulInstructionCounter;
#endif
    }
}

void addcdivDenomFloat32TensorsInplace(tensor_t *a, tensor_t *b, tensor_t *v, float dScale,
                                       float eps, float s) {
    size_t numberOfElements = calcNumberOfElementsByTensor(a);
    float *aData = (float *)a->data;
    const float *bData = (const float *)b->data;
    const float *vData = (const float *)v->data;
    for (size_t i = 0; i < numberOfElements; i++) {
        float d = sqrtf(vData[i]);
        d = d / dScale;
        d = d + eps;
        float numer = s * bData[i];
        float quotient = numer / d;
        aData[i] = aData[i] + quotient;
#ifdef TRACK_INSTRUCTIONS
        ++addcdivDenomInstructionCounter;
#endif
    }
}

size_t getLerpInstructionCounter(void) {
    return lerpInstructionCounter;
}

size_t getAddcmulInstructionCounter(void) {
    return addcmulInstructionCounter;
}

size_t getAddcdivDenomInstructionCounter(void) {
    return addcdivDenomInstructionCounter;
}
