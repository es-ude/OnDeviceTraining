#define SOURCE_FILE "OPTIMIZER_API"

#include <math.h>
#include <stdlib.h>

#include "Common.h"
#include "Conv1d.h"
#include "Conv1dTransposed.h"
#include "GroupNorm.h"
#include "Layer.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "OptimizerApi.h"
#include "Quantization.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"

void scaleOptimizerGradients(optimizer_t *optimizer, float factor) {
    /* Validation: warn (currently via PRINT_ERROR — see #151 for unified
     * warn/assert macros) on non-positive or non-finite factor. */
    if (!(factor > 0.0f && isfinite(factor))) {
        PRINT_ERROR("scaleOptimizerGradients: suspicious factor %f "
                    "(expected positive, finite)",
                    (double)factor);
    }

    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];

        switch (param->grad->quantization->type) {
        case FLOAT32: {
            size_t numberOfValues = calcNumberOfElementsByParameter(param);
            float *gradArr = (float *)param->grad->data;
            for (size_t j = 0; j < numberOfValues; j++) {
                gradArr[j] *= factor;
            }
            break;
        }
        case SYM_INT32: {
            /* float_value = int32_value * scale ⇒ multiplicative scaling can
             * be absorbed into the per-tensor scale, leaving the int32 storage
             * untouched. O(1) and avoids quantization round-trip loss. */
            symInt32QConfig_t *gradQ = param->grad->quantization->qConfig;
            gradQ->scale *= factor;
            break;
        }
        case SYM: {
            /* Packed-SYM dequant (mantissa * scale) is linear in scale exactly
             * like the SYM_INT32 case above — fold the factor into the
             * per-tensor scale, packed codes untouched (O(1), exact).
             * Defensive (belt-and-suspenders, group-quant PR3 Task 4):
             * gradInit's own carrier gate already rejects a grouped SYM
             * template before a grad tensor is ever built (grads are
             * per-tensor unconditionally, #300 axis), so numGroups > 1
             * should be unreachable here — but a hand-assembled optimizer
             * (this file's own comment above, and the pattern every
             * hand-built optimizer_t test in this tree exercises) could
             * still hand it a grad tensor that bypassed that gate. Folding
             * `factor` into scales[0] alone would silently scale ONLY group
             * 0 and leave every other group's scale untouched — fail fast
             * instead of corrupting the gradient. */
            symQConfig_t *gradQ = param->grad->quantization->qConfig;
            if (gradQ->numGroups > 1) {
                PRINT_ERROR("scaleOptimizerGradients: grouped SYM grad storage "
                            "(numGroups=%zu) is not supported — grads are per-tensor "
                            "unconditionally (gradInit's carrier gate, #300 axis)",
                            gradQ->numGroups);
                exit(1);
            }
            gradQ->scales[0] *= factor;
            break;
        }
        case ASYM: {
            /* Packed-ASYM dequant is (code - zeroPoint) * scale (D6 code
             * domain): still linear in scale, so the fold is exact the same
             * way; zeroPoint is an additive offset on the code axis and is
             * untouched. Same defensive grouped gate as the SYM arm above
             * (grads are per-tensor by gradInit's carrier gate, but a
             * hand-assembled optimizer could bypass it -- folding into
             * scales[0] alone would corrupt every other group). */
            asymQConfig_t *gradQ = param->grad->quantization->qConfig;
            if (gradQ->numGroups > 1) {
                PRINT_ERROR("scaleOptimizerGradients: grouped ASYM grad storage "
                            "(numGroups=%zu) is not supported — grads are per-tensor "
                            "unconditionally (gradInit's carrier gate, #300 axis)",
                            gradQ->numGroups);
                exit(1);
            }
            gradQ->scales[0] *= factor;
            break;
        }
        case BFP: {
            /* No O(1) fold here: BFP dequant is mantissa * 2^(E-bias) per
             * group, and an arbitrary factor is not a power of two -- an
             * honest O(n) value-domain repack (scaleBfpTensorInPlace: fresh
             * exponents from the scaled absmax, requant with the grad
             * config's own storage roundingMode). No grouped gate, unlike
             * the SYM/ASYM fold arms above: the primitive handles grouped
             * tensors correctly, so a hand-assembled grouped grad is scaled
             * right rather than corrupted (in-tree grads are per-tensor
             * anyway, gradInit's carrier gate, #300 axis).
             * Non-finite-factor asymmetry: every OTHER arm warns (the check
             * at the top of this function) and then PROPAGATES the non-finite
             * factor -- a float grad element or a per-tensor scale represents
             * NaN/inf fine, so the failure stays loud downstream. BFP warns
             * the same way and then HARD-FAILS inside the primitive, because
             * a (mantissa, shared exponent) grid has no non-finite code:
             * silently skipping would mask the caller's bug and saturating
             * would invent data. Precedent for the exit: optimizerClipGradNorm
             * below. */
            scaleBfpTensorInPlace(param->grad, factor);
            break;
        }
        default:
            PRINT_ERROR("scaleOptimizerGradients: unsupported gradient qtype "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM, BFP; INT32/BOOL "
                        "grad storage remains unsupported, #261)");
            exit(1);
        }
    }
}

float optimizerClipGradNorm(optimizer_t *optimizer, float maxNorm) {
    if (!(maxNorm > 0.0f && isfinite(maxNorm))) {
        PRINT_ERROR("optimizerClipGradNorm: invalid maxNorm %f (expected positive, finite)",
                    (double)maxNorm);
        exit(1);
    }

    /* Joint norm: ONE running sum of squares over every element of every
     * tracked grad (not a per-tensor norm) -- double accumulator, one sqrt
     * cast to float32 at the very end. */
    double sumSquares = 0.0;
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        tensor_t *grad = param->grad;

        switch (grad->quantization->type) {
        case FLOAT32: {
            size_t numberOfValues = calcNumberOfElementsByParameter(param);
            float *gradArr = (float *)grad->data;
            for (size_t j = 0; j < numberOfValues; j++) {
                double v = (double)gradArr[j];
                sumSquares += v * v;
            }
            break;
        }
        case SYM_INT32: {
            /* scale^2 * sum(mantissa^2): mantissas widen to double BEFORE
             * squaring (no int32*int32 product, no int64) -- mirrors the
             * SYM-kernel int32-accumulator rule in spirit. */
            size_t numberOfValues = calcNumberOfElementsByParameter(param);
            int32_t *mantissas = (int32_t *)grad->data;
            symInt32QConfig_t *gradQ = grad->quantization->qConfig;
            double scale = (double)gradQ->scale;
            double tensorSumSquares = 0.0;
            for (size_t j = 0; j < numberOfValues; j++) {
                double m = (double)mantissas[j];
                tensorSumSquares += m * m;
            }
            sumSquares += scale * scale * tensorSumSquares;
            break;
        }
        case SYM:
        case ASYM:
        case BFP:
            PRINT_ERROR("optimizerClipGradNorm: packed SYM/ASYM/BFP grad storage not supported "
                        "(v1) -- computing a norm needs unpacked element values; the O(1) "
                        "scale-fold only helps APPLYING an already-computed clip coefficient, "
                        "not computing the norm itself (follow-up, not implemented). accepted: "
                        "FLOAT32, SYM_INT32");
            exit(1);
        default:
            PRINT_ERROR("optimizerClipGradNorm: unsupported gradient qtype (accepted: FLOAT32, "
                        "SYM_INT32; packed SYM/ASYM/BFP rejected above, INT32/BOOL grad storage "
                        "remains unsupported, #261)");
            exit(1);
        }
    }

    float totalNorm = (float)sqrt(sumSquares);
    float clipCoef = maxNorm / (totalNorm + 1e-6f);
    if (clipCoef < 1.0f) {
        scaleOptimizerGradients(optimizer, clipCoef);
    }
    return totalNorm;
}

void collectTrainableParameters(layer_t **model, size_t sizeModel, parameter_t **slots) {
    size_t paramSlot = 0;
    for (size_t i = 0; i < sizeModel; i++) {
        layer_t *currentLayer = model[i];
        if (layerIsFrozen(currentLayer)) {
            continue;
        }
        layerConfig_t *layerConfig = currentLayer->config;

        switch (currentLayer->type) {
        case LINEAR: {
            linearConfig_t *linearConfig = layerConfig->linear;

            slots[paramSlot] = linearConfig->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (linearConfig->bias != NULL) {
                slots[paramSlot + 1] = linearConfig->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case CONV1D: {
            conv1dConfig_t *conv1dCfg = layerConfig->conv1d;

            slots[paramSlot] = conv1dCfg->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (conv1dCfg->bias != NULL) {
                slots[paramSlot + 1] = conv1dCfg->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case CONV1D_TRANSPOSED: {
            conv1dTransposedConfig_t *ctCfg = layerConfig->conv1dTransposed;

            slots[paramSlot] = ctCfg->weights;

            /* BIAS_FALSE (header-sanctioned): no bias parameter to collect. */
            if (ctCfg->bias != NULL) {
                slots[paramSlot + 1] = ctCfg->bias;
                paramSlot += 2;
            } else {
                paramSlot += 1;
            }
            break;
        }
        case LAYERNORM: {
            layerNormConfig_t *lnCfg = layerConfig->layerNorm;

            slots[paramSlot] = lnCfg->gamma;
            slots[paramSlot + 1] = lnCfg->beta;

            paramSlot += 2;
            break;
        }
        case GROUPNORM: {
            groupNormConfig_t *gnCfg = layerConfig->groupNorm;

            slots[paramSlot] = gnCfg->gamma;
            slots[paramSlot + 1] = gnCfg->beta;

            paramSlot += 2;
            break;
        }
        case RELU:
        case SOFTMAX:
        case FLATTEN:
        case MAXPOOL1D:
        case AVGPOOL1D:
        case ADAPTIVE_AVGPOOL1D:
        case DROPOUT:
        case QUANTIZATION:
            break;
        default:
            PRINT_ERROR("Unknown Layer Type");
            exit(1);
        }
    }
}

bool modelHasFrozenLayer(layer_t **model, size_t sizeModel) {
    for (size_t i = 0; i < sizeModel; i++) {
        if (layerIsFrozen(model[i])) {
            return true;
        }
    }
    return false;
}

void validateOptimizerGradStorage(optimizer_t *optim, const char *factoryName) {
    /* #261, PR3: grads may be stored FLOAT32 (default), SYM_INT32 (explicit
     * low-level knob), packed SYM/ASYM, or per-tensor BFP (explicit
     * grad-storage knob, memory-constrained targets; BFP epic PR3 Task 6 --
     * grouped BFP grads stay rejected at gradInit's own carrier gate, #300
     * axis, so a grad tensor reaching here is always per-tensor). INT32/BOOL
     * grad storage remains unimplemented - fail fast rather than silently
     * misread bytes in an unsupported layout. A NULL grad in a collected slot
     * is a mis-built model (frozen layers are skipped before collection
     * (#380); a collected slot must always carry an allocated grad) - fail
     * fast here instead of crashing mid-training (PR #366 review). */
    for (size_t s = 0; s < optim->sizeStates; s++) {
        tensor_t *grad = optim->parameter[s]->grad;
        if (grad == NULL) {
            PRINT_ERROR("%s: trainable parameter slot %zu has no grad tensor "
                        "(mis-built model; every trainable param must carry an "
                        "allocated grad)",
                        factoryName, s);
            exit(1);
        }
        qtype_t gradType = grad->quantization->type;
        if (gradType != FLOAT32 && gradType != SYM_INT32 && gradType != SYM && gradType != ASYM &&
            gradType != BFP) {
            PRINT_ERROR("%s: gradient storage dtype %d not supported "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM, BFP; INT32/BOOL grad "
                        "storage remains unsupported, #261)",
                        factoryName, (int)gradType);
            exit(1);
        }
    }
}

void freeState(states_t *state) {
    for (size_t i = 0; i < state->statesPerParameter; i++) {
        freeTensor(state->stateBuffers[i]);
    }
    freeReservedMemory(state->stateBuffers);
    freeReservedMemory(state);
}

void freeOptim(optimizer_t *optim) {
    for (size_t i = 0; i < optim->sizeStates; i++) {
        freeParameter(optim->parameter[i]);
        if (optim->states != NULL) {
            freeState(optim->states[i]);
        }
    }
    freeReservedMemory(optim->parameter);
    if (optim->states != NULL) {
        freeReservedMemory(optim->states);
    }
    /* optimImpl_t is a union of pointers: freeing through any member names
     * the same reserveMemory block, so this is type-agnostic by layout. */
    freeReservedMemory(optim->impl->sgd);
    freeReservedMemory(optim->impl);
    freeReservedMemory(optim);
}
