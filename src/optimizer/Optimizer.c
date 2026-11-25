#define SOURCE_FILE "OPTIMIZER"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AdamW.h"
#include "Common.h"
#include "DyadicScale.h"
#include "Layer.h"
#include "OdtHook.h"
#include "Optimizer.h"
#include "ParameterWalk.h"
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

static void countParameter(const paramRef_t *ref, void *ctx) {
    (void)ref;
    (*(size_t *)ctx)++;
}

size_t calcTotalNumberOfStates(const model_t *m) {
    size_t number = 0;
    forEachTrainableParameter(m, countParameter, &number);
    return number;
}
