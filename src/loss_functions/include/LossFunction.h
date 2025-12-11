#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "Tensor.h"

typedef float (*lossFwdFn_t)(tensor_t* modelOutput, tensor_t* label);
typedef void (*lossBwdFn_t)(tensor_t* modelOutput, tensor_t* label, tensor_t* result);

typedef struct lossFunctions {
    lossFwdFn_t forward;
    lossBwdFn_t backward;
} lossFunctions_t;

typedef enum lossType
{
    MSE,
    CROSS_ENTROPY
} lossType_t;

extern lossFunctions_t lossFunctions[];

#endif //LOSSFUNCTION_H
