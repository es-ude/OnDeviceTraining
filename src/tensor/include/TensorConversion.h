#ifndef TENSOR_CONVERSION_H
#define TENSOR_CONVERSION_H

#include "Tensor.h"

typedef void (*conversionFunction_t)(tensor_t* inputTensor, tensor_t* outputTensor);

void convertTensor(tensor_t* inputTensor, tensor_t* outputTensor);

extern conversionFunction_t conversionMatrix[5][5];

#endif // TENSOR_CONVERSION_H
