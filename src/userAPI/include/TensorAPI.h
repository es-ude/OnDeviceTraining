#ifndef TENSOR_H
#define TENSOR_H

#include "Tensor.h"

tensor_t* tensorInitInt32(int32_t* data, size_t* dims, size_t numberOfDims, sparsity_t* sparsity);
tensor_t* tensorInitFloat(float* data, size_t* dims, size_t numberOfDims, sparsity_t* sparsity);
tensor_t* tensorInitSymInt32(float* data, size_t* dims, size_t numberOfDims,
                             roundingMode_t roundingMode, sparsity_t* sparsity);
tensor_t* tensorInitAsym(float* data, size_t* dims, size_t numberOfDims, uint8_t qBits,
                         roundingMode_t roundingMode, sparsity_t* sparsity);
tensor_t* tensorInit(float* data, size_t* dims, size_t numberOfDims, quantization_t* quantization,
                     sparsity_t* sparsity);

tensor_t* gradInitInt32(tensor_t* param, sparsity_t* sparsity);
tensor_t* gradInitFloat(tensor_t* param, sparsity_t* sparsity);
tensor_t* gradInitSymInt32(tensor_t* param, roundingMode_t roundingMode, sparsity_t* sparsity);
tensor_t* gradInitAsym(tensor_t* param, uint8_t qBits, roundingMode_t roundingMode, sparsity_t* sparsity);

parameter_t* parameterInit(tensor_t* param, tensor_t* grad);

quantization_t* getQLike(quantization_t* quantization);
tensor_t* getTensorLike(tensor_t* tensor);

// IMPORTANT: these are needed for trainingAPI.c to free gradTensors without freeing the actual Pointer
void freeData(tensor_t* tensor);
void freeShape(shape_t* shape);
void freeQuantization(quantization_t* quantization);

void freeTensor(tensor_t* tensor);
void freeParameter(parameter_t *parameter);

#endif //TENSOR_H
