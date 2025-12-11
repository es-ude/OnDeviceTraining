#ifndef SGDAPI_H
#define SGDAPI_H

#include "Sgd.h"

sgd_t* sgdInit(layer_t* model, size_t sizeModel, float learningRate, float momentumFactor, float weightDecay);

#endif //SGDAPI_H
