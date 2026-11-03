#define SOURCE_FILE "MUL"

#ifdef TRACK_INSTRUCTIONS
#define MUL_FUNC_INT mulIntsWithInstructionCounter
#define MUL_FUNC_FLOAT mulFloatsWithInstructionCounter
#else
#define MUL_FUNC_INT mulInts
#define MUL_FUNC_FLOAT mulFloats
#endif

#include <stdio.h>
#include <stdlib.h>

#include "Arithmetic.h"
#include "DTypes.h"
#include "Mul.h"
#include "SymAccumGuard.h"

#include <Common.h>

size_t mulInstructionCounter = 0;

int32_t mulInts(int32_t a, int32_t b) {
    return a * b;
}

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

void mulInt32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor) {
    int32ElementArithmeticFunc_t mul = MUL_FUNC_INT;
    int32PointWiseArithmetic(a, b, mul, outputTensor);
}

void mulInt32TensorsInplace(tensor_t *a, tensor_t *b) {
    int32ElementArithmeticFunc_t mul = MUL_FUNC_INT;
    int32PointWiseArithmeticInplace(a, b, mul);
}

void mulInt32ElementWithInt32Tensor(tensor_t *a, int32_t b, tensor_t *outputTensor) {
    int32ElementArithmeticFunc_t mul = MUL_FUNC_INT;
    int32ElementWithTensorArithmetic(a, b, mul, outputTensor);
}

void mulInt32ElementWithInt32TensorInplace(tensor_t *a, int32_t b) {
    int32ElementArithmeticFunc_t mul = MUL_FUNC_INT;
    int32ElementWithTensorArithmeticInplace(a, b, mul);
}

void mulFloat32Tensors(tensor_t *a, tensor_t *b, tensor_t *outputTensor) {
    floatElementArithmeticFunc_t mul = MUL_FUNC_FLOAT;
    floatPointWiseArithmetic(a, b, mul, outputTensor);
}

void mulFloat32TensorsInplace(tensor_t *a, tensor_t *b) {
    floatElementArithmeticFunc_t mul = MUL_FUNC_FLOAT;
    floatPointWiseArithmeticInplace(a, b, mul);
}

void mulFloat32ElementWithFloat32Tensor(tensor_t *a, float b, tensor_t *outputTensor) {
    floatElementArithmeticFunc_t mul = MUL_FUNC_FLOAT;
    floatElementWithTensorArithmetic(a, b, mul, outputTensor);
}

void mulFloat32ElementWithFloat32TensorInplace(tensor_t *a, float b) {
    floatElementArithmeticFunc_t mul = MUL_FUNC_FLOAT;
    floatElementWithTensorArithmeticInplace(a, b, mul);
}

/* Width a raw elementwise SYM product declares (#516): |a * b| is at most
 * 2^(qa-1) * 2^(qb-1), which can exceed the operands' width, so the output
 * carries that static bound under the strict rule (32 = the raw int32 range)
 * and a consumer that reads it as an operand proves its own budget from it.
 * The int32 product itself is not checked: with qa + qb > 32 it can wrap. */
static uint8_t symProductWidth(uint8_t aBits, uint8_t bBits) {
    unsigned qa = symAccumClampedWidth(aBits);
    unsigned qb = symAccumClampedWidth(bBits);
    return symAccumWidthForMagnitude((uint64_t)1 << (qa + qb - 2u));
}

void mulSymInt32Tensors(tensor_t *aTensor, tensor_t *bTensor, tensor_t *outputTensor) {
    size_t aNumberOfValues = calcNumberOfElementsByTensor(aTensor);
    size_t bNumberOfValues = calcNumberOfElementsByTensor(bTensor);

    if (aNumberOfValues != bNumberOfValues) {
        PRINT_ERROR("Mismatched number of values!");
        exit(1);
    }

    int32_t aValues[aNumberOfValues];
    int32_t bValues[aNumberOfValues];
    readBytesAsInt32Array(aNumberOfValues, aTensor->data, aValues);
    readBytesAsInt32Array(aNumberOfValues, bTensor->data, bValues);

    int32_t output[aNumberOfValues];
    for (size_t i = 0; i < aNumberOfValues; i++) {
        output[i] = MUL_FUNC_INT(aValues[i], bValues[i]);
    }

    symInt32QConfig_t *aConfig = aTensor->quantization->qConfig;
    symInt32QConfig_t *bConfig = bTensor->quantization->qConfig;
    /* read both widths before any config write: the output may alias an operand */
    uint8_t width = symProductWidth(aConfig->qMaxBits, bConfig->qMaxBits);

    writeInt32ArrayToByteArray(aNumberOfValues, output, outputTensor->data);
    symInt32QConfig_t *outputConfig = outputTensor->quantization->qConfig;
    outputConfig->scale = MUL_FUNC_FLOAT(aConfig->scale, bConfig->scale);
    outputConfig->qMaxBits = width;
}

void mulSymInt32TensorsInplace(tensor_t *aTensor, tensor_t *bTensor) {
    size_t aNumberOfValues = calcNumberOfElementsByTensor(aTensor);
    size_t bNumberOfValues = calcNumberOfElementsByTensor(bTensor);

    if (aNumberOfValues != bNumberOfValues) {
        PRINT_ERROR("Mismatched number of values!");
        exit(1);
    }

    int32_t aValues[aNumberOfValues];
    int32_t bValues[aNumberOfValues];
    readBytesAsInt32Array(aNumberOfValues, aTensor->data, aValues);
    readBytesAsInt32Array(aNumberOfValues, bTensor->data, bValues);

    int32_t output[aNumberOfValues];
    for (size_t i = 0; i < aNumberOfValues; i++) {
        output[i] = MUL_FUNC_INT(aValues[i], bValues[i]);
    }

    writeInt32ArrayToByteArray(aNumberOfValues, output, aTensor->data);

    symInt32QConfig_t *aConfig = aTensor->quantization->qConfig;
    symInt32QConfig_t *bConfig = bTensor->quantization->qConfig;

    /* a's config is the output: read both widths before writing it */
    uint8_t width = symProductWidth(aConfig->qMaxBits, bConfig->qMaxBits);
    aConfig->scale = MUL_FUNC_FLOAT(aConfig->scale, bConfig->scale);
    aConfig->qMaxBits = width;
}

size_t getMulInstructionCounter() {
    return mulInstructionCounter;
}
