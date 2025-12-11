#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Tensor.h"
#include "Quantization.h"
#include "Layer.h"

typedef struct sgd sgd_t;

typedef struct states
{
    tensor_t** stateBuffers;
    size_t statesPerParameter;
} states_t;

typedef union optimImpl
{
    sgd_t* sgd;
} optimImpl_t;

typedef enum
{
    SGD,
    SGD_M
} optimizerType_t;

typedef struct optimizer
{
    optimizerType_t type;
    qtype_t qtype;
    optimImpl_t* impl;
    parameter_t** parameter;
    states_t** states;
    size_t sizeStates;
} optimizer_t;

typedef void (*stepFn_t)(optimizer_t* optim);
typedef void (*zeroFn_t)(optimizer_t* optim);

typedef struct optimizerFunctions
{
    stepFn_t step;
    zeroFn_t zero;
} optimizerFunctions_t;

extern optimizerFunctions_t optimizerFunctions[];

size_t calcTotalNumberOfStates(layer_t** model, size_t sizeModel);

#endif //OPTIMIZER_H
