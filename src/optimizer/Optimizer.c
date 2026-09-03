#define SOURCE_FILE "OPTIMIZER"

#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "Layer.h"
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
            symQ->scales[0] = 1.f;
            break;
        }
        case ASYM: {
            /* grads are per-tensor (gradInit carrier gate), so element 0 IS
             * the whole grid; zeroPoints[0]=0 is the load-bearing half (code
             * 0 must decode to exactly 0.0f -- (0 - 0)*scale under the D6
             * code-domain decode), scales[0]=1.f is hygiene. */
            asymQConfig_t *asymQ = param->grad->quantization->qConfig;
            asymQ->scales[0] = 1.f;
            asymQ->zeroPoints[0] = 0;
            break;
        }
        case BFP: {
            /* BFP epic PR3 Task 6: grads are per-tensor (gradInit's carrier
             * gate), but exponents is still a [numGroups]-array -- loop
             * generically. Byte-zero above already zero-values every packed
             * mantissa (code 0 decodes to exactly 0.0f regardless of
             * exponent), and the next accumulate does not key on exponents
             * either: FixedGrid's fresh-vs-carry decision (Task 5) is a
             * codes-only scan, so a byte-zeroed grad is classified fresh
             * whatever its exponents say. This reset is pure hygiene like
             * SYM_INT32's scale reset above -- the canonical zero state
             * (stored = bias) for serialization/inspection and any future
             * exponent-reading consumer; pinned by the Task 6 e2e's
             * exponent assertion (UnitTestMultiLayerTraining.c). */
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
