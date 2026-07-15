#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Layer.h"
#include "Quantization.h"
#include "Tensor.h"

typedef struct sgd sgd_t;
typedef struct adamW adamW_t;

typedef struct states {
    tensor_t **stateBuffers;
    size_t statesPerParameter;
} states_t;

typedef union optimImpl {
    sgd_t *sgd;
    adamW_t *adamW;
} optimImpl_t;

typedef enum { SGD_M, ADAM_W } optimizerType_t;

typedef struct optimizer {
    optimizerType_t type;
    optimImpl_t *impl;
    parameter_t **parameter;
    states_t **states;
    size_t sizeStates;
} optimizer_t;

typedef void (*stepFn_t)(optimizer_t *optim);
typedef void (*zeroFn_t)(optimizer_t *optim);
/* #327: optimizer-agnostic LR access for the scheduler. LR stays in the impl
 * structs; these accessors are the only sanctioned cross-impl path. */
typedef float (*getLrFn_t)(optimizer_t *optim);
typedef void (*setLrFn_t)(optimizer_t *optim, float learningRate);

typedef struct optimizerFunctions {
    stepFn_t step;
    zeroFn_t zero;
    getLrFn_t getLr;
    setLrFn_t setLr;
} optimizerFunctions_t;

extern optimizerFunctions_t optimizerFunctions[];

/* optimizer-agnostic: reads only optim->parameter; every vtable row points
 * here unless an impl needs custom zeroing. */
void optimizerZeroGrad(optimizer_t *optimizer);

size_t calcTotalNumberOfStates(layer_t **model, size_t sizeModel);

#endif // OPTIMIZER_H
