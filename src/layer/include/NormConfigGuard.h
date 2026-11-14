#ifndef ODT_NORM_CONFIG_GUARD_H
#define ODT_NORM_CONFIG_GUARD_H

#include <stdbool.h>
#include <stdlib.h>

#include "ArithmeticType.h"
#include "Common.h"
#include "ExecuteOp.h"
#include "Quantization.h"
#include "Tensor.h"

static inline bool normIsArithmetic(arithmeticType_t type) {
    return type == ARITH_FLOAT32 || type == ARITH_SYM_INT32 || type == ARITH_BFP;
}

static inline bool normIsBfpOperand(qtype_t type) {
    return type == FLOAT32 || type == BFP;
}

/* OUT_WRITE is the zero value, so an unset mode would overwrite the grads. */
static inline void normRequireAccMode(outputMode_t mode, const char *who, const char *field) {
    if (mode != OUT_ACC_DYNAMIC_RESCALE && mode != OUT_ACC_FIXED_SCALE) {
        PRINT_ERROR("%s: %s must be a grad accumulate mode (got %d)", who, field, (int)mode);
        exit(1);
    }
}

/* The arithmetic/storage rules GroupNorm and LayerNorm share: supported maths, param and
 * wire storage coherent with each math, and, unless frozen, grad modes and grad storage.
 * Call it from the layer's ValidateConfig after the layer's own geometry checks; `who`
 * prefixes the message. gamma/beta must be allocated; their grads may be NULL only when
 * frozen. */
static inline void
normValidateAffineCoherence(const parameter_t *gamma, const parameter_t *beta, arithmeticType_t fwd,
                            arithmeticType_t bwd, const quantization_t *outputQ,
                            const quantization_t *propLossQ, outputMode_t weightGradAccMode,
                            outputMode_t biasGradAccMode, bool frozen, const char *who) {
    if (!normIsArithmetic(fwd) || !normIsArithmetic(bwd)) {
        PRINT_ERROR("%s: forwardMath and propLossMath must be ARITH_FLOAT32, ARITH_SYM_INT32 or "
                    "ARITH_BFP (got %d, %d)",
                    who, (int)fwd, (int)bwd);
        exit(1);
    }
    qtype_t gammaType = gamma->param->quantization->type;
    qtype_t betaType = beta->param->quantization->type;
    /* The SYM_INT32 kernels read gamma/beta as int32 mantissas. */
    if (fwd == ARITH_SYM_INT32 && (gammaType != SYM_INT32 || betaType != SYM_INT32)) {
        PRINT_ERROR("%s: SYM_INT32 forwardMath requires SYM_INT32 gamma AND beta storage", who);
        exit(1);
    }
    /* The ARITH_BFP kernels borrow BFP params and stage FLOAT32 ones at the produced-wire
     * config, which must therefore be BFP. */
    if (fwd == ARITH_BFP &&
        (!normIsBfpOperand(gammaType) || !normIsBfpOperand(betaType) || outputQ->type != BFP)) {
        PRINT_ERROR("%s: ARITH_BFP forwardMath requires FLOAT32 or BFP gamma AND beta storage and "
                    "a BFP-typed outputQ (the staging width anchor)",
                    who);
        exit(1);
    }
    /* FLOAT32 math over quantized params would forward fine and die at the first backward:
     * the FLOAT32 backward raw-casts gamma. */
    if (fwd == ARITH_FLOAT32 && (gammaType != FLOAT32 || betaType != FLOAT32)) {
        PRINT_ERROR("%s: ARITH_FLOAT32 forwardMath requires FLOAT32 gamma AND beta storage -- "
                    "declare ARITH_BFP / ARITH_SYM_INT32 math, or insert a Quantization layer",
                    who);
        exit(1);
    }
    /* The SYM_INT32 backward reads forwardInput and gamma as mantissas. SYM forward + FLOAT32
     * backward stays constructible as the inference-only profile. */
    if (bwd == ARITH_SYM_INT32 && fwd != ARITH_SYM_INT32) {
        PRINT_ERROR("%s: SYM_INT32 propLossMath requires SYM_INT32 forwardMath", who);
        exit(1);
    }
    if (bwd == ARITH_BFP &&
        (!normIsBfpOperand(gammaType) || !normIsBfpOperand(betaType) || propLossQ->type != BFP)) {
        PRINT_ERROR("%s: ARITH_BFP propLossMath requires FLOAT32 or BFP gamma AND beta storage "
                    "and a BFP-typed propLossQ (the staging width anchor)",
                    who);
        exit(1);
    }
    if (bwd == ARITH_FLOAT32 &&
        (gammaType == BFP || betaType == BFP || outputQ->type == BFP || propLossQ->type == BFP)) {
        PRINT_ERROR("%s: ARITH_FLOAT32 propLossMath rejects BFP gamma/beta storage and BFP wire "
                    "configs -- the FLOAT32 backward raw-casts",
                    who);
        exit(1);
    }
    if (frozen) {
        return; /* no grads: the modes and grad storage are never used */
    }
    if (bwd != ARITH_FLOAT32) {
        normRequireAccMode(weightGradAccMode, who, "weightGradAccMode");
        normRequireAccMode(biasGradAccMode, who, "biasGradAccMode");
        return;
    }
    /* The FLOAT32 backward ignores the modes: it adds into the grads through a raw float *
     * cast. */
    if (gamma->grad->quantization->type != FLOAT32 || beta->grad->quantization->type != FLOAT32) {
        PRINT_ERROR("%s: ARITH_FLOAT32 propLossMath requires FLOAT32 gamma/beta grad storage "
                    "(got %d, %d)",
                    who, (int)gamma->grad->quantization->type, (int)beta->grad->quantization->type);
        exit(1);
    }
}

#endif // ODT_NORM_CONFIG_GUARD_H
