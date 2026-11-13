#define SOURCE_FILE "ADD"

#ifdef TRACK_INSTRUCTIONS
#define ADD_FUNC_INT addIntsWithInstructionCounter
#define ADD_FUNC_FLOAT addFloatsWithInstructionCounter
#else
#define ADD_FUNC_INT addInts
#define ADD_FUNC_FLOAT addFloats
#endif

#include "Add.h"
#include "Arithmetic.h"
#include "Common.h"
#include "DTypes.h"
#include "Tensor.h"
#include "TensorConversion.h"

size_t addInstructionCounter = 0;

int32_t addInts(int32_t a, int32_t b) {
    return a + b;
}

int32_t addIntsWithInstructionCounter(int32_t a, int32_t b) {
    ++addInstructionCounter;
    return a + b;
}

int32_t addInt32s(int32_t a, int32_t b) {
    return ADD_FUNC_INT(a, b);
}

float addFloats(float a, float b) {
    return a + b;
}

float addFloatsWithInstructionCounter(float a, float b) {
    ++addInstructionCounter;
    return a + b;
}

float addFloat32s(float a, float b) {
    return ADD_FUNC_FLOAT(a, b);
}

void addFloat32TensorsInplace(tensor_t *a, tensor_t *b) {
    floatPointWiseArithmeticInplace(a, b, ADD_FUNC_FLOAT);
}

void addSymInt32TensorsInplace(tensor_t *aTensor, tensor_t *bTensor, rngStream_t *rs) {
    size_t numberOfValues = calcNumberOfElementsByTensor(aTensor);

    quantization_t floatQ;
    initFloat32Quantization(&floatQ);

    uint8_t aFloatData[numberOfValues * sizeof(float)];
    tensor_t aFloat;
    setTensorValuesForConversion(aFloatData, &floatQ, aTensor, &aFloat);
    convertTensor(aTensor, &aFloat, rs);

    uint8_t bFloatData[numberOfValues * sizeof(float)];
    tensor_t bFloat;
    setTensorValuesForConversion(bFloatData, &floatQ, bTensor, &bFloat);
    convertTensor(bTensor, &bFloat, rs);

    addFloat32TensorsInplace(&aFloat, &bFloat);

    convertTensor(&aFloat, aTensor, rs);
}

size_t getAddInstructionCounter() {
    return addInstructionCounter;
}
