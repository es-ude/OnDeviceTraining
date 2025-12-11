#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

tensor_t* tensorInitFloat(float* data, size_t* dims, size_t numberOfDims, bool isSparse);

tensor_t *tensorInit(float *data, size_t *dims, size_t numberOfDims, quantization_t *quantization, bool isSparse);

tensor_t* gradInitFloat(tensor_t* param);

parameter_t* parameterInit(tensor_t* param, tensor_t* grad);

tensor_t *getTensorLike(tensor_t *tensor);

// IMPORTANT: these are needed for trainingAPI.c
void freeData(tensor_t *tensor);
void freeShape(shape_t *shape);
void freeQuantization(quantization_t *quantization);

void freeTensor(tensor_t *tensor);

#endif //TENSOR_H
