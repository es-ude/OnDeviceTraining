#define SOURCE_FILE "OPTIMIZER"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "BatchNorm1d.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "DyadicScale.h"
#include "Layer.h"
#include "Linear.h"
#include "OdtHook.h"
#include "Optimizer.h"
#include "Sgd.h"

void optimizerStep(optimizer_t *optimizer, const rngKey_t *updateKey) {
    /* rngFold's index is 32 bits wide: a larger parameter index would reuse
     * the update keys of an earlier parameter. Checked before the BEGIN
     * event, so an exit never leaves the phase open. */
    if (updateKey != NULL && optimizer->sizeStates > 0 && optimizer->sizeStates - 1 > UINT32_MAX) {
        PRINT_ERROR("optimizerStep: parameter index exceeds UINT32_MAX");
        exit(1);
    }
    odtHookFire(ODT_EVENT_OPTIMIZER_BEGIN);
    optimizerFunctions[optimizer->type].step(optimizer, updateKey);
    odtHookFire(ODT_EVENT_OPTIMIZER_END);
}

void optimizerZeroGrad(optimizer_t *optimizer) {
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        size_t paramSize = calcNumberOfElementsByParameter(param);
        size_t totalNumberOfBytes = calcNumberOfBytesForData(param->grad->quantization, paramSize);

        memset(param->grad->data, 0, totalNumberOfBytes);

        /* Reset quantization metadata so zeroed codes represent zero. Grads are per-tensor, so
         * element 0 is the whole SYM/ASYM grid. */
        switch (param->grad->quantization->type) {
        case SYM_INT32: {
            symInt32QConfig_t *symIntQ = param->grad->quantization->qConfig;
            odtScaleStore(&symIntQ->scale, ODT_SCALE_ONE);
            break;
        }
        case SYM: {
            symQConfig_t *symQ = param->grad->quantization->qConfig;
            odtScaleStore(&symQ->scales[0], ODT_SCALE_ONE);
            break;
        }
        case ASYM: {
            /* Decode is (code - zeroPoint) * scale: the zero point must be reset. */
            asymQConfig_t *asymQ = param->grad->quantization->qConfig;
            odtScaleStore(&asymQ->scales[0], ODT_SCALE_ONE);
            asymQ->zeroPoints[0] = 0;
            break;
        }
        case BFP: {
            /* Reset BFP exponents to the canonical zero grid; byte-zeroed mantissas already
             * represent zero. */
            bfpQConfig_t *bfpQ = param->grad->quantization->qConfig;
            for (size_t g = 0; g < bfpQ->numGroups; g++) {
                bfpQ->exponents[g] = (uint8_t)bfpExponentBias(bfpQ);
            }
            break;
        }
        default:
            break;
        }
    }
}

void optimizerSetWriteBackRounding(optimizer_t *optimizer, roundingMode_t writeBackRounding) {
    optimizer->writeBackRounding = writeBackRounding;
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
    if (layerIsFrozen(layer)) {
        return 0;
    }
    switch (layer->type) {
    case LINEAR:
        return layer->config->linear->bias != NULL ? 2 : 1;
    case LAYERNORM:
    case GROUPNORM:
        return 2;
    case BATCHNORM1D:
        return layer->config->batchNorm1d->affine ? 2 : 0;
    case RELU:
    case SOFTMAX:
    case FLATTEN:
    case MAXPOOL1D:
    case AVGPOOL1D:
    case ADAPTIVE_AVGPOOL1D:
    case DROPOUT:
    case QUANTIZATION:
    case ADD:
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

size_t calcTotalNumberOfStates(const model_t *m) {
    size_t number = 0;
    for (size_t i = 0; i < m->numLayers; i++) {
        number += calcNumberOfStatesByLayer(m->layers[i]);
    }
    return number;
}
