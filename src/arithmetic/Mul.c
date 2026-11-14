#define SOURCE_FILE "MUL"

#ifdef TRACK_INSTRUCTIONS
#define MUL_FUNC_FLOAT mulFloatsWithInstructionCounter
#else
#define MUL_FUNC_FLOAT mulFloats
#endif

#include "Mul.h"
#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"
#include "ScaleCanon.h"
#include "SymAccumGuard.h"

size_t mulInstructionCounter = 0;

int32_t mulIntsWithInstructionCounter(int32_t a, int32_t b) {
    ++mulInstructionCounter;
    return a * b;
}

float mulFloats(float a, float b) {
    return a * b;
}

float mulFloatsWithInstructionCounter(float a, float b) {
    ++mulInstructionCounter;
    return a * b;
}

float mulFloat32s(float a, float b) {
    return MUL_FUNC_FLOAT(a, b);
}

void mulFloat32ElementWithFloat32TensorInplace(tensor_t *a, float b) {
    floatElementWithTensorArithmeticInplace(a, b, MUL_FUNC_FLOAT);
}

/* Width a raw elementwise SYM product declares (#516): a * b reaches the
 * corner 2^(qa-1) * 2^(qb-1) (both operands at their negative extreme), which
 * can exceed the operands' width, so the output carries the exact width of the
 * product's range (32 = the raw int32 range). */
static uint8_t symProductWidth(uint8_t aBits, uint8_t bBits) {
    int32_t aMin, aMax, bMin, bMax;
    uint64_t maxPos, maxNegMag;
    symAccumWidthRange(aBits, &aMin, &aMax);
    symAccumWidthRange(bBits, &bMin, &bMax);
    symAccumCornerRange(aMin, aMax, bMin, bMax, &maxPos, &maxNegMag);
    return symAccumWidthForRange(maxPos, maxNegMag);
}

void mulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    size_t numberOfValues = calcNumberOfElementsByTensor(aTensor);

    int32_t aValues[numberOfValues];
    int32_t bValues[numberOfValues];
    readBytesAsInt32Array(numberOfValues, aTensor->data, aValues);
    readBytesAsInt32Array(numberOfValues, bTensor->data, bValues);

    int32_t output[numberOfValues];
    for (size_t i = 0; i < numberOfValues; i++) {
        output[i] = mulInt32s(aValues[i], bValues[i]);
    }

    symInt32QConfig_t *aConfig = aTensor->quantization->qConfig;
    symInt32QConfig_t *bConfig = bTensor->quantization->qConfig;
    /* read both widths before any config write: the output may alias an operand */
    uint8_t width = symProductWidth(aConfig->qMaxBits, bConfig->qMaxBits);

    writeInt32ArrayToByteArray(numberOfValues, output, outputTensor->data);
    symInt32QConfig_t *outputConfig = outputTensor->quantization->qConfig;
    storeCanonScale(&outputConfig->scale,
                    canonProductScale(loadCanonScale(&aConfig->scale),
                                      loadCanonScale(&bConfig->scale), ODT_SCALE_NEAREST));
    outputConfig->qMaxBits = width;
}

size_t getMulInstructionCounter() {
    return mulInstructionCounter;
}
