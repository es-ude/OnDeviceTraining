#include "Optimizer.h"
#include "Sgd.h"

optimizerFunctions_t optimizerFunctions[] = {
    [SGD] = {sgdStep, sgdZeroGrad},
    [SGD_M] = {sgdStepM, sgdZeroGrad}
};

static size_t calcNumberOfStatesByLayerType(const layerType_t type) {
    switch (type) {
    case LINEAR:
        return 2;
    case RELU:
        return 0;
    case CONV1D:
        return 2;
    default:
        return 0;
    }
}

size_t calcTotalNumberOfStates(layer_t **model, size_t sizeModel) {
    size_t number = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        number += calcNumberOfStatesByLayerType(model[i]->type);
    }
    return number;
}
