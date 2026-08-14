#ifndef LAYER_QUANT_H
#define LAYER_QUANT_H

#include "ArithmeticType.h"
#include "ExecuteOp.h"
#include "Quantization.h"

/*! Per-layer quantization profile: 4 by-value declared-arithmetic slots +
 *  6 storage-config pointers (spec 2026-07-02 arithmetic-type-split, D5).
 *
 *  Each layer factory reads only the fields it needs (documented in each
 *  layer's header).  Arithmetic fields are by-value: there is no "unset"
 *  sentinel, so factories validate the storage pointers they consume
 *  instead (NULL is a fatal error via PRINT_ERROR + exit).  Fields the
 *  layer doesn't read are ignored, so a single fully-populated
 *  layerQuant_t can be shared across all layers in a model.  */
typedef struct layerQuant {
    arithmetic_t forwardMath;    /* declared forward compute representation */
    arithmetic_t weightGradMath; /* declared weight-grad arithmetic (GEMM family) */
    arithmetic_t biasGradMath;   /* declared bias-grad arithmetic (GEMM family) */
    arithmetic_t propLossMath;   /* declared dx-wire arithmetic (kernel selection) */

    quantization_t *outputQ;   /* REQUIRED for every layer except Flatten: forward-wire storage */
    quantization_t *propLossQ; /* REQUIRED for every layer except Flatten: dx-wire storage */
    quantization_t *weightStorage;     /* REQUIRED for Conv1d / Conv1dTransposed / Linear /
                                           LayerNorm (gamma) */
    quantization_t *biasStorage;       /* REQUIRED for Conv1d / Conv1dTransposed / Linear
                                           (bias) / LayerNorm (beta) */
    quantization_t *weightGradStorage; /* grad storage knob; NULL = per-layer default */
    quantization_t *biasGradStorage;   /* grad storage knob; NULL = per-layer default */

    outputMode_t weightGradAccMode; /* accumulate mode for weight/gamma grads (PR3 spec D1) */
    outputMode_t biasGradAccMode;   /* accumulate mode for bias/beta grads (PR3 spec D1) */
} layerQuant_t;

/*! Populate `lq` from a single `q`: all four arithmetic slots derive via
 *  `arithmeticFromQuantization(q)`; `outputQ`/`propLossQ`/`weightStorage`/
 *  `biasStorage` all alias `q`; both grad-storage slots are NULL (factory
 *  default); weightGradAccMode/biasGradAccMode are BOTH OUT_ACC_DYNAMIC_RESCALE.
 *  DYNAMIC is the only accumulate mode valid for every layer's grad
 *  intermediate (FLOAT32 or SYM_INT32), so this convenience default can never
 *  break a layer — including LayerNorm, whose FLOAT32 beta-grad intermediate
 *  OUT_ACC_FIXED_SCALE would abort. Layers with a specialized
 *  fixed-scale-integer bias-grad scheme (Linear/Conv1d/ConvT1d, per their init
 *  functions) keep FIXED on the hand-wired path; opt back into it here by
 *  setting biasGradAccMode = OUT_ACC_FIXED_SCALE explicitly (PR3 spec D1).
 *  Convenience for the common all-same-quantization case.
 *  Caller retains ownership of `q`. */
void layerQuantInitUniform(layerQuant_t *lq, quantization_t *q);

/*! Deep-copy a `quantization_t` and its `qConfig`. Returns NULL if `src` is NULL.
 *
 *  Caller owns the returned allocation. Free via:
 *      freeQuantization(result);
 *
 *  NOT the qConfig/wrapper pair-free this used to prescribe: SYM, ASYM and BFP
 *  qConfigs each own further heap blocks (scales / scales+zeroPoints /
 *  exponents) that this function deep-copies into the result, and only
 *  freeQuantization frees them (BFP epic PR2 Task 8 — the pair-free leaked one
 *  block per Owning layer).
 *
 *  The `qConfig` size is dispatched by `src->type`; BOOL/INT32/FLOAT32 have
 *  no qConfig (result->qConfig == NULL). Unknown types fire PRINT_ERROR +
 *  exit(1).
 *
 *  Used by every `*LayerInitOwning` factory to materialize per-layer copies
 *  of the four math quantizations referenced by `layerQuant_t`. */
quantization_t *deepCopyQuantization(quantization_t *src);

#endif /* LAYER_QUANT_H */
