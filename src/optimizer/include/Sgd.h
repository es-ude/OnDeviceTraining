#ifndef SGD_H
#define SGD_H

#include <stddef.h>
#include <stdint.h>

#include "Tensor.h"
#include "Layer.h"

typedef struct momentumBuffer
{
    parameter_t* parameter;
    tensor_t* momentums;
} momentumBuffer_t;

typedef struct sgd
{
    float learningRate;
    float momentumFactor;
    float weightDecay;
    momentumBuffer_t** momentumBuffers;
    size_t sizeMomentumBuffers;
} sgd_t;

void initMomentumBuffer(momentumBuffer_t* momentumBuffer, parameter_t* parameter, tensor_t* momentums);

uint32_t calcTotalNumberOfMomentumBuffers(layer_t* model, size_t sizeModel);

void initSGDConfig(sgd_t* sgd, float learningRate, float momentumFactor, float weightDecay,
                   momentumBuffer_t** momentumBuffers, size_t sizeMomentumBuffers);

void SGDStepFloat(sgd_t* sgd);

void SGDStepAsym(sgd_t* sgd);

void SGDZeroGrad(sgd_t* sgdConfig);

#endif
