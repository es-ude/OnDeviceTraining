#define SOURCE_FILE "DIV"

#ifdef TRACK_INSTRUCTIONS
#define DIV_FUNC_INT divIntsWithInstructionCounter
#define DIV_FUNC_FLOAT divFloatsWithInstructionCounter
#else
#define DIV_FUNC_INT divInts
#define DIV_FUNC_FLOAT divFloats
#endif

#include "Div.h"
#include "Common.h"
#include "DTypes.h"
#include "ScaleCanon.h"

/* a / b of two stored scales, rounded once (NEAREST): MulRatio forms
 * ma * 2^ea * 2^14 / mb exactly rounded, and the power-of-two factor
 * 2^(-14 - eb) is exact. b >= 2^127 (eb = 113) has no POW2 factor and fails
 * fast. */
static float canonQuotientScale(float a, float b) {
    odtScale_t sb = odtScaleFromFloat(b, ODT_SCALE_NEAREST);
    odtScale_t q = odtScaleMulRatio(odtScaleFromFloat(a, ODT_SCALE_NEAREST), 1u << 14,
                                    (uint32_t)sb.m, ODT_SCALE_NEAREST);
    return odtScaleToFloat(odtScaleMul(q, ODT_SCALE_POW2(-14 - (int32_t)sb.e), ODT_SCALE_NEAREST));
}

size_t divInstructionCounter = 0;

int32_t divInts(int32_t a, int32_t b) {
    return a / b;
}

int32_t divIntsWithInstructionCounter(int32_t a, int32_t b) {
    ++divInstructionCounter;
    return a / b;
}

float divFloats(float a, float b) {
    return a / b;
}

float divFloatsWithInstructionCounter(float a, float b) {
    ++divInstructionCounter;
    return a / b;
}

float divFloat32s(float a, float b) {
    return DIV_FUNC_FLOAT(a, b);
}

void divSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    size_t numberOfValues = calcNumberOfElementsByTensor(aTensor);

    int32_t aValues[numberOfValues];
    int32_t bValues[numberOfValues];
    readBytesAsInt32Array(numberOfValues, aTensor->data, aValues);
    readBytesAsInt32Array(numberOfValues, bTensor->data, bValues);

    int32_t output[numberOfValues];
    for (size_t i = 0; i < numberOfValues; i++) {
        output[i] = DIV_FUNC_INT(aValues[i], bValues[i]);
    }

    symInt32QConfig_t *aConfig = aTensor->quantization->qConfig;
    symInt32QConfig_t *bConfig = bTensor->quantization->qConfig;

    writeInt32ArrayToByteArray(numberOfValues, output, outputTensor->data);
    symInt32QConfig_t *outputConfig = outputTensor->quantization->qConfig;
    storeCanonScale(&outputConfig->scale, canonQuotientScale(loadCanonScale(&aConfig->scale),
                                                             loadCanonScale(&bConfig->scale)));
}

size_t getDivInstructionCounter() {
    return divInstructionCounter;
}
