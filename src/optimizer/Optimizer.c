#define SOURCE_FILE "OPTIMIZER"

#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Linear.h"
#include "Optimizer.h"
#include "Sgd.h"

void optimizerZeroGrad(optimizer_t *optimizer) {
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        size_t paramSize = calcNumberOfElementsByParameter(param);
        size_t totalNumberOfBytes = calcNumberOfBytesForData(param->grad->quantization, paramSize);

        memset(param->grad->data, 0, totalNumberOfBytes);

        /* Byte-zero the mantissa/code storage above is necessary but, for
         * SYM/ASYM, not sufficient for VALUE-zero: config-reset the grid so
         * code 0 decodes to exactly 0.0f (spec §5.3). SYM_INT32's scale reset
         * is hygiene (the first-store trigger is the all-zero mantissa state,
         * not the scale); ASYM's zeroPoint reset is load-bearing - without it,
         * code 0 would decode to zeroPoint*scale, not 0 (PR2 watch-list item). */
        switch (param->grad->quantization->type) {
        case SYM_INT32: {
            symInt32QConfig_t *symIntQ = param->grad->quantization->qConfig;
            symIntQ->scale = 1.f;
            break;
        }
        case SYM: {
            symQConfig_t *symQ = param->grad->quantization->qConfig;
            symQ->scale = 1.f;
            break;
        }
        case ASYM: {
            asymQConfig_t *asymQ = param->grad->quantization->qConfig;
            asymQ->scale = 1.f;
            asymQ->zeroPoint = 0;
            break;
        }
        default:
            break;
        }
    }
}

optimizerFunctions_t optimizerFunctions[] = {
    [SGD_M] = {.step = sgdStepM, .zero = optimizerZeroGrad, .getLr = sgdGetLr, .setLr = sgdSetLr},
    [ADAM_W] = {
        .step = adamWStep, .zero = optimizerZeroGrad, .getLr = adamWGetLr, .setLr = adamWSetLr}};

/* Linear/Conv1d/Conv1dTransposed are bias-optional (BIAS_FALSE,
 * header-sanctioned): a bias-less layer contributes only its weight state,
 * not a weight+bias pair. Every other trainable layer type still has a
 * fixed contribution. */
static size_t calcNumberOfStatesByLayer(const layer_t *layer) {
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->bias != NULL ? 2 : 1;
    case LAYERNORM:
    case GROUPNORM:
        return 2;
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
    case QUANTIZATION:
        return 0;
    case CONV1D:
        return layer->config->conv1d->bias != NULL ? 2 : 1;
    case CONV1D_TRANSPOSED:
        return layer->config->conv1dTransposed->bias != NULL ? 2 : 1;
    default:
        PRINT_ERROR("Unknown Layer Type!");
        exit(1);
    }
}

size_t calcTotalNumberOfStates(layer_t **model, size_t sizeModel) {
    size_t number = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        number += calcNumberOfStatesByLayer(model[i]);
    }
    return number;
}
