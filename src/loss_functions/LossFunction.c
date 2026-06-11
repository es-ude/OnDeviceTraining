#define SOURCE_FILE "LOSS_FUNCTION"

#include "LossFunction.h"
#include "CrossEntropy.h"
#include "MSE.h"

lossFunctions_t lossFunctions[] = {
    [MSE] = {mseLossForward, mseLossBackward, computeMeanScaleMSE},
    [CROSS_ENTROPY] = {crossEntropyForward, crossEntropySoftmaxBackward, computeMeanScaleCE},
};

lossConfig_t defaultLossConfig(lossFuncType_t funcType) {
    lossConfig_t cfg;
    cfg.funcType = funcType;
    cfg.backwardReduction = REDUCTION_MEAN;
    cfg.classWeights = NULL;
    return cfg;
}
