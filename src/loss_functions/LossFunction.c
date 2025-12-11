#include "LossFunction.h"
#include "CrossEntropy.h"
#include "MSE.h"

lossFunctions_t lossFunctions[] = {
    [MSE] = {mseLossForward, mseLossBackward},
    [CROSS_ENTROPY] = {crossEntropyForwardFloat, crossEntropySoftmaxBackward}
};
