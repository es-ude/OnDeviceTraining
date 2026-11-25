#define SOURCE_FILE "OPTIMIZER_API"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "BfpKernelSupport.h"
#include "Common.h"
#include "Layer.h"
#include "OptimizerApi.h"
#include "OptimizerKeys.h"
#include "ParameterWalk.h"
#include "Quantization.h"
#include "ScaleCanon.h"
#include "StorageApi.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "TensorConversion.h"

/* rngFold's index is 32 bits wide: a larger parameter index would reuse
 * the scale stream of an earlier parameter. */
static void requireParamIndexFitsKey(const optimizer_t *optimizer, const rngKey_t *updateKey,
                                     const char *what) {
    if (updateKey != NULL && optimizer->sizeStates > 0 && optimizer->sizeStates - 1 > UINT32_MAX) {
        PRINT_ERROR("%s: parameter index exceeds UINT32_MAX", what);
        exit(1);
    }
}

/* What one fold multiplies by. FLOAT32 grads multiply by `factor`. A
 * SYM_INT32/SYM/ASYM grad folds into its stored scale, which must stay an
 * exact dyadic value: a mean (divisor != 0) folds s * 1 / divisor rounded once
 * (canonRatioScale, NEAREST), so the integer divisor is never first turned
 * into a float 1/B; an arbitrary factor (divisor == 0) is canonicalized NEAREST
 * and multiplied in with odtScaleMul (canonCoeffScale). */
typedef struct {
    float factor;
    size_t divisor;
} gradFold_t;

static float foldStoredScale(float s, gradFold_t fold, const char *what) {
    if (fold.divisor == 0) {
        return canonCoeffScale(s, fold.factor);
    }
    if (fold.divisor > 0x80000000u) {
        PRINT_ERROR("%s: a mean divisor above 2^31 (%zu) has no dyadic scale fold", what,
                    fold.divisor);
        exit(1);
    }
    return canonRatioScale(s, 1u, (uint32_t)fold.divisor);
}

/* The fold of every non-BFP grad dtype: FLOAT32 multiplies, SYM_INT32/SYM/ASYM
 * fold into the per-tensor scale. Kept out of line so the integer mean-scale
 * entry below carries no float arithmetic in its own body (the census reads
 * it as a `self` root). */
static __attribute__((noinline)) void scaleNonBfpGradient(parameter_t *param, gradFold_t fold,
                                                          const char *what) {
    switch (param->grad->quantization->type) {
    case FLOAT32: {
        size_t numberOfValues = calcNumberOfElementsByParameter(param);
        float *gradArr = (float *)param->grad->data;
        for (size_t j = 0; j < numberOfValues; j++) {
            gradArr[j] *= fold.factor;
        }
        break;
    }
    case SYM_INT32: {
        /* Dequant is linear in scale: fold the factor into it, codes untouched. */
        symInt32QConfig_t *gradQ = param->grad->quantization->qConfig;
        storeCanonScale(&gradQ->scale, foldStoredScale(loadCanonScale(&gradQ->scale), fold, what));
        break;
    }
    case SYM: {
        /* Fold scaling into the per-tensor scale; leave the codes unchanged. */
        symQConfig_t *gradQ = param->grad->quantization->qConfig;
        storeCanonScale(&gradQ->scales[0],
                        foldStoredScale(loadCanonScale(&gradQ->scales[0]), fold, what));
        break;
    }
    case ASYM: {
        /* Fold scaling into the per-tensor scale; leave codes and zero point unchanged. */
        asymQConfig_t *gradQ = param->grad->quantization->qConfig;
        storeCanonScale(&gradQ->scales[0],
                        foldStoredScale(loadCanonScale(&gradQ->scales[0]), fold, what));
        break;
    }
    default:
        PRINT_ERROR("%s: unsupported gradient qtype "
                    "(accepted: FLOAT32, SYM_INT32, SYM, ASYM, BFP)",
                    what);
        exit(1);
    }
}

/* 1.0f / (float)divisor: exactly the float computeMeanScale returns for the
 * same divisor, so the non-BFP arms of scaleOptimizerGradientsMean match the
 * float entry bit for bit. Out of line for the same census reason. */
static __attribute__((noinline)) float meanFactorOf(size_t divisor) {
    return 1.0f / (float)divisor;
}

void scaleOptimizerGradients(optimizer_t *optimizer, float factor, const rngKey_t *updateKey) {
    const char *what = "scaleOptimizerGradients";
    requireParamIndexFitsKey(optimizer, updateKey, what);
    /* Warns only: FLOAT32 grads carry a non-finite factor on; scale folds and BFP reject it. */
    if (!(factor > 0.0f && isfinite(factor))) {
        PRINT_ERROR("scaleOptimizerGradients: suspicious factor %f "
                    "(expected positive, finite)",
                    (double)factor);
    }

    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        if (param->grad->quantization->type != BFP) {
            scaleNonBfpGradient(param, (gradFold_t){.factor = factor, .divisor = 0}, what);
            continue;
        }
        /* BFP requires value-domain repacking and storage rounding; non-finite factors cannot
         * be represented. */
        rngStream_t scaleStream;
        scaleBfpTensorInPlace(param->grad, factor,
                              optimizerScaleStream(updateKey, (uint32_t)i, &scaleStream));
    }
}

void scaleOptimizerGradientsMean(optimizer_t *optimizer, size_t divisor,
                                 const rngKey_t *updateKey) {
    const char *what = "scaleOptimizerGradientsMean";
    requireParamIndexFitsKey(optimizer, updateKey, what);
    if (divisor == 0) {
        PRINT_ERROR("%s: a mean over zero samples has no factor", what);
        exit(1);
    }
    bool haveScalar = false;
    bfpScalar_t inverse = {.q = 0, .e = 0};
    for (size_t i = 0; i < optimizer->sizeStates; i++) {
        parameter_t *param = optimizer->parameter[i];
        if (param->grad->quantization->type != BFP) {
            scaleNonBfpGradient(
                param, (gradFold_t){.factor = meanFactorOf(divisor), .divisor = divisor}, what);
            continue;
        }
        /* Integer only: 1/divisor once per call, then the in-place integer
         * scale on the same per-parameter SCALE stream the float arm draws. */
        if (!haveScalar) {
            inverse = bfpScalarFromCount(divisor, what);
            haveScalar = true;
        }
        rngStream_t scaleStream;
        scaleBfpTensorByScalarInPlace(param->grad, inverse,
                                      optimizerScaleStream(updateKey, (uint32_t)i, &scaleStream));
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
            double scale = (double)loadCanonScale(&gradQ->scale);
            double tensorSumSquares = 0.0;
            for (size_t j = 0; j < numberOfValues; j++) {
                double m = (double)mantissas[j];
                tensorSumSquares += m * m;
            }
            sumSquares += scale * scale * tensorSumSquares;
            break;
        }
        default:
            /* SYM, ASYM and BFP: valid optimizer grad storage, but the norm needs unpacked
             * element values this clip does not compute. */
            PRINT_ERROR("optimizerClipGradNorm: packed SYM/ASYM/BFP grad storage not supported "
                        "-- computing a norm needs unpacked element values. accepted: "
                        "FLOAT32, SYM_INT32");
            exit(1);
        }
    }

    float totalNorm = (float)sqrt(sumSquares);
    float clipCoef = maxNorm / (totalNorm + 1e-6f);
    if (clipCoef < 1.0f) {
        /* The clip accepts only FLOAT32 and SYM_INT32 grads, which rescale
         * without rounding: no key. */
        scaleOptimizerGradients(optimizer, clipCoef, NULL);
    }
    return totalNorm;
}

typedef struct slotFill {
    parameter_t **slots;
    size_t next;
} slotFill_t;

static void fillSlot(const paramRef_t *ref, void *ctx) {
    slotFill_t *fill = ctx;
    fill->slots[fill->next++] = ref->parameter;
}

void collectTrainableParameters(const model_t *m, parameter_t **slots) {
    slotFill_t fill = {.slots = slots, .next = 0};
    forEachTrainableParameter(m, fillSlot, &fill);
}

bool modelHasFrozenLayer(const model_t *m) {
    for (size_t i = 0; i < m->numLayers; i++) {
        if (layerIsFrozen(m->layers[i])) {
            return true;
        }
    }
    return false;
}

static void requirePerTensorMomentTemplate(const quantization_t *momentQuant,
                                           const char *factoryName) {
    size_t numGroups = 1;
    const char *dtypeName = "";
    switch (momentQuant->type) {
    case SYM:
        numGroups = ((const symQConfig_t *)momentQuant->qConfig)->numGroups;
        dtypeName = "SYM";
        break;
    case ASYM:
        numGroups = ((const asymQConfig_t *)momentQuant->qConfig)->numGroups;
        dtypeName = "ASYM";
        break;
    case BFP:
        numGroups = ((const bfpQConfig_t *)momentQuant->qConfig)->numGroups;
        dtypeName = "BFP";
        break;
    default:
        break;
    }
    if (numGroups > 1) {
        PRINT_ERROR("%s: grouped %s moment templates are unsupported -- moment storage is "
                    "per-tensor",
                    factoryName, dtypeName);
        exit(1);
    }
}

states_t **initMomentStates(optimizer_t *optim, size_t statesPerParameter,
                            quantization_t *momentQuant, const char *factoryName) {
    if (optim->sizeStates > 0) {
        requirePerTensorMomentTemplate(momentQuant, factoryName);
    }
    states_t **states = reserveMemory(optim->sizeStates * sizeof(states_t *));
    for (size_t s = 0; s < optim->sizeStates; s++) {
        states_t *paramStates = reserveMemory(sizeof(states_t));
        paramStates->statesPerParameter = statesPerParameter;
        paramStates->stateBuffers = reserveMemory(statesPerParameter * sizeof(tensor_t *));
        tensor_t *param = optim->parameter[s]->param;
        for (size_t k = 0; k < statesPerParameter; k++) {
            paramStates->stateBuffers[k] =
                initTensor(getShapeLike(param->shape), getQLike(momentQuant), NULL);
        }
        states[s] = paramStates;
    }
    return states;
}

/* Scaling and zeroGrad touch only group 0, so a grouped SYM/ASYM grad would be silently
 * corrupted. gradInit never builds one; this catches a hand-built parameter. */
static void requirePerTensorGradAt(const tensor_t *grad, const char *factoryName, size_t slot) {
    const quantization_t *q = grad->quantization;
    size_t numGroups = 1;
    if (q->type == SYM) {
        numGroups = ((const symQConfig_t *)q->qConfig)->numGroups;
    } else if (q->type == ASYM) {
        numGroups = ((const asymQConfig_t *)q->qConfig)->numGroups;
    }
    if (numGroups > 1) {
        PRINT_ERROR("%s: trainable parameter slot %zu has a grouped gradient (numGroups=%zu); "
                    "gradients must be per-tensor",
                    factoryName, slot, numGroups);
        exit(1);
    }
}

void validateOptimizerGradStorage(optimizer_t *optim, const char *factoryName) {
    for (size_t s = 0; s < optim->sizeStates; s++) {
        tensor_t *grad = optim->parameter[s]->grad;
        if (grad == NULL) {
            PRINT_ERROR("%s: trainable parameter slot %zu has no grad tensor "
                        "(mis-built model; every trainable param must carry an "
                        "allocated grad)",
                        factoryName, s);
            exit(1);
        }
        requirePerTensorGradAt(grad, factoryName, s);
        qtype_t gradType = grad->quantization->type;
        if (gradType != FLOAT32 && gradType != SYM_INT32 && gradType != SYM && gradType != ASYM &&
            gradType != BFP) {
            PRINT_ERROR("%s: gradient storage dtype %d not supported "
                        "(accepted: FLOAT32, SYM_INT32, SYM, ASYM, BFP)",
                        factoryName, (int)gradType);
            exit(1);
        }
    }
}

bool paramSelectNormOrBias(const paramRef_t *ref, void *ctx) {
    (void)ctx;
    return ref->role == PARAM_ROLE_BIAS || ref->role == PARAM_ROLE_NORM_SCALE ||
           ref->role == PARAM_ROLE_NORM_SHIFT;
}

/* groupOf is uint8_t. */
#define MAX_PARAMETER_GROUPS 256u

typedef struct groupAssign {
    optimizer_t *optim;
    paramSelectFn_t select;
    void *selectCtx;
    uint8_t group;
    size_t next;
    size_t matched;
    const char *what;
} groupAssign_t;

static void assignVisited(const paramRef_t *ref, void *ctxv) {
    groupAssign_t *a = ctxv;
    optimizer_t *optim = a->optim;
    const size_t k = a->next++;
    if (k >= optim->sizeStates || ref->parameter != optim->parameter[k]) {
        PRINT_ERROR("%s: the model is not the one the optimizer was built from "
                    "(parameter %zu differs)",
                    a->what, k);
        exit(1);
    }
    if (!a->select(ref, a->selectCtx)) {
        return;
    }
    if (optim->groupOf[k] != 0) {
        PRINT_ERROR("%s: parameter %zu is already in group %u; every parameter belongs to "
                    "exactly one group",
                    a->what, k, (unsigned)optim->groupOf[k]);
        exit(1);
    }
    optim->groupOf[k] = a->group;
    a->matched++;
}

size_t assignParameterGroup(optimizer_t *optim, const model_t *m, paramSelectFn_t select, void *ctx,
                            float lrMultiplier, const char *what) {
    if (!(isfinite(lrMultiplier) && lrMultiplier > 0.0f)) {
        PRINT_ERROR("%s: lrMultiplier must be finite and > 0 (got %g)", what, (double)lrMultiplier);
        exit(1);
    }
    const size_t group = optim->numGroups == 0 ? 1 : optim->numGroups;
    if (group >= MAX_PARAMETER_GROUPS) {
        PRINT_ERROR("%s: at most %u parameter groups", what, MAX_PARAMETER_GROUPS);
        exit(1);
    }
    if (optim->sizeStates == 0) {
        PRINT_ERROR("%s: the optimizer has no parameter to select", what);
        exit(1);
    }
    if (optim->groupOf == NULL) {
        optim->groupOf = reserveMemory(optim->sizeStates * sizeof(uint8_t));
    }
    groupAssign_t a = {.optim = optim,
                       .select = select,
                       .selectCtx = ctx,
                       .group = (uint8_t)group,
                       .next = 0,
                       .matched = 0,
                       .what = what};
    forEachTrainableParameter(m, assignVisited, &a);
    if (a.next != optim->sizeStates) {
        PRINT_ERROR("%s: the model is not the one the optimizer was built from "
                    "(%zu trainable parameters, the optimizer has %zu)",
                    what, a.next, optim->sizeStates);
        exit(1);
    }
    if (a.matched == 0) {
        PRINT_ERROR("%s: the selection matches no parameter", what);
        exit(1);
    }
    return group;
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
    freeReservedMemory(optim->groupOf);
    /* optimImpl_t is a union of pointers: freeing through any member names
     * the same reserveMemory block, so this is type-agnostic by layout. */
    freeReservedMemory(optim->impl->sgd);
    freeReservedMemory(optim->impl);
    freeReservedMemory(optim);
}
